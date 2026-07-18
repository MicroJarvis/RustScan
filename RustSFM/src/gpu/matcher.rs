use super::WgpuContext;
use crate::sift::SiftMatchingOptions;
use anyhow::{Context, Result};
use rustslam::Match;
use std::collections::HashSet;
use std::sync::Arc;
use wgpu::util::DeviceExt;

const MATCHING_SHADER: &str = include_str!("shaders/sift_matching.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatchParams {
    query_count: u32,
    target_count: u32,
    max_l2_distance: f32,
    ratio_squared: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatchCandidate {
    best_index: u32,
    second_index: u32,
    best_distance: f32,
    second_distance: f32,
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

pub struct WgpuSiftMatcher {
    context: Arc<WgpuContext>,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl WgpuSiftMatcher {
    pub fn try_new() -> Result<Self> {
        Self::from_context(WgpuContext::try_new()?)
    }

    pub fn from_context(context: Arc<WgpuContext>) -> Result<Self> {
        let device = context.device();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rustsfm SIFT matcher bind group layout"),
            entries: &[
                storage_layout_entry(0, true),
                storage_layout_entry(1, true),
                storage_layout_entry(2, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rustsfm SIFT matcher shader"),
            source: wgpu::ShaderSource::Wgsl(MATCHING_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rustsfm SIFT matcher pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rustsfm SIFT matcher pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("match_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Ok(Self {
            context,
            bind_group_layout,
            pipeline,
        })
    }

    pub fn match_descriptors(
        &self,
        queries: &[[u8; 128]],
        targets: &[[u8; 128]],
        options: &SiftMatchingOptions,
    ) -> Result<Vec<Match>> {
        options.check()?;
        if queries.is_empty() || targets.is_empty() {
            return Ok(Vec::new());
        }
        let forward = self.match_one_way(queries, targets, options)?;
        let mut matches = if options.cross_check {
            let reverse = self.match_one_way(targets, queries, options)?;
            let reverse_pairs = reverse
                .into_iter()
                .map(|value| (value.query_idx, value.train_idx))
                .collect::<HashSet<_>>();
            forward
                .into_iter()
                .filter(|value| reverse_pairs.contains(&(value.train_idx, value.query_idx)))
                .collect()
        } else {
            forward
        };
        matches.sort_by(|left, right| {
            left.distance
                .total_cmp(&right.distance)
                .then_with(|| left.query_idx.cmp(&right.query_idx))
                .then_with(|| left.train_idx.cmp(&right.train_idx))
        });
        if options.max_num_matches > 0 {
            matches.truncate(options.max_num_matches);
        }
        Ok(matches)
    }

    fn match_one_way(
        &self,
        queries: &[[u8; 128]],
        targets: &[[u8; 128]],
        options: &SiftMatchingOptions,
    ) -> Result<Vec<Match>> {
        let query_count =
            u32::try_from(queries.len()).context("GPU SIFT query count exceeds u32")?;
        let target_count =
            u32::try_from(targets.len()).context("GPU SIFT target count exceeds u32")?;
        let query_values = pack_descriptors(queries);
        let target_values = pack_descriptors(targets);
        let params = MatchParams {
            query_count,
            target_count,
            max_l2_distance: 512.0 * 512.0 * options.max_distance * options.max_distance,
            ratio_squared: options.max_ratio * options.max_ratio,
            pad0: 0,
            pad1: 0,
            pad2: 0,
            pad3: 0,
        };
        let device = self.context.device();
        let query_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm SIFT matcher queries"),
            contents: bytemuck::cast_slice(&query_values),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let target_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm SIFT matcher targets"),
            contents: bytemuck::cast_slice(&target_values),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm SIFT matcher candidates"),
            size: u64::from(query_count) * std::mem::size_of::<MatchCandidate>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm SIFT matcher params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustsfm SIFT matcher bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: query_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: target_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rustsfm SIFT matcher encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustsfm SIFT matcher pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(query_count.div_ceil(64), 1, 1);
        }
        self.context.queue().submit(Some(encoder.finish()));
        let candidates = self
            .context
            .read_buffer::<MatchCandidate>(&output, queries.len())?;
        Ok(candidates
            .into_iter()
            .enumerate()
            .filter(|(_, candidate)| candidate.valid != 0)
            .map(|(query_idx, candidate)| Match {
                query_idx: query_idx as u32,
                train_idx: candidate.best_index,
                distance: candidate.best_distance.sqrt() / 512.0,
            })
            .collect())
    }
}

fn pack_descriptors(descriptors: &[[u8; 128]]) -> Vec<u32> {
    descriptors
        .iter()
        .flat_map(|descriptor| {
            descriptor
                .chunks_exact(4)
                .map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        })
        .collect()
}

fn storage_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
