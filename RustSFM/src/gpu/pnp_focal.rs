use super::{WgpuContext, WgpuPnpModelScorer};
use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};
use rustslam::SE3;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuPnpFocalModel {
    pub(crate) row0: [f32; 4],
    pub(crate) row1: [f32; 4],
    pub(crate) row2: [f32; 4],
    pub(crate) log_focal_and_padding: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuPnpFocalResult {
    pub(crate) selected_model: u32,
    pub(crate) inliers: u32,
    pub(crate) valid: u32,
    pub(crate) pad: u32,
    pub(crate) residual_sum: f32,
    pub(crate) focal: f32,
    pub(crate) pad1: f32,
    pub(crate) pad2: f32,
}

pub(crate) const PNP_FOCAL_SHADER: &str = include_str!("shaders/pnp_focal.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Pod, Zeroable)]
struct GpuPnpFocalSample {
    indices: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuPnpFocalSamplingParams {
    seed: u32,
    trial_count: u32,
    observation_count: u32,
    pad: u32,
}

pub(crate) struct WgpuPnPFocalSampler {
    context: Arc<WgpuContext>,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl WgpuPnPFocalSampler {
    pub(crate) fn from_context(context: Arc<WgpuContext>) -> Result<Self> {
        let device = context.device();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rustsfm gpu pnp-focal sampler bind group layout"),
            entries: &[storage_layout_entry(0, false), uniform_layout_entry(1)],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rustsfm gpu pnp-focal shader"),
            source: wgpu::ShaderSource::Wgsl(PNP_FOCAL_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rustsfm gpu pnp-focal sampler pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rustsfm gpu pnp-focal sampler pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("sample_four_points"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        if let Some(error) = pollster::block_on(error_scope.pop()) {
            bail!("gpu pnp-focal sampler pipeline creation failed: {error}");
        }
        Ok(Self {
            context,
            bind_group_layout,
            pipeline,
        })
    }

    pub(crate) fn sample_indices(
        &self,
        seed: u32,
        trial_count: usize,
        observation_count: usize,
    ) -> Result<Vec<[u32; 4]>> {
        if observation_count < 4 {
            bail!("gpu pnp-focal sampling needs at least four observations");
        }
        let trial_count =
            u32::try_from(trial_count).context("gpu pnp-focal trial count exceeds u32")?;
        let observation_count = u32::try_from(observation_count)
            .context("gpu pnp-focal observation count exceeds u32")?;
        if trial_count == 0 {
            return Ok(Vec::new());
        }
        let max_workgroups = self
            .context
            .device()
            .limits()
            .max_compute_workgroups_per_dimension;
        if trial_count > max_workgroups {
            bail!("gpu pnp-focal sampling requires {trial_count} workgroups, limit is {max_workgroups}");
        }
        let byte_len = u64::try_from(
            usize::try_from(trial_count)
                .context("gpu pnp-focal trial count does not fit usize")?
                .checked_mul(std::mem::size_of::<GpuPnpFocalSample>())
                .context("gpu pnp-focal sample buffer size overflow")?,
        )?;
        let limits = self.context.device().limits();
        let storage_limit = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size);
        if byte_len > storage_limit {
            bail!(
                "gpu pnp-focal sample buffer requires {byte_len} bytes, limit is {storage_limit}"
            );
        }
        let device = self.context.device();
        let samples = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal samples"),
            size: byte_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = GpuPnpFocalSamplingParams {
            seed,
            trial_count,
            observation_count,
            pad: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp-focal sampling params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustsfm gpu pnp-focal sampler bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: samples.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rustsfm gpu pnp-focal sampler encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustsfm gpu pnp-focal sampler pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(trial_count, 1, 1);
        }
        self.context.queue().submit(Some(encoder.finish()));
        let samples = self.context.read_buffer::<GpuPnpFocalSample>(
            &samples,
            usize::try_from(trial_count).context("gpu pnp-focal trial count does not fit usize")?,
        )?;
        for sample in &samples {
            if sample
                .indices
                .iter()
                .any(|&index| index >= observation_count)
            {
                bail!("gpu pnp-focal sampler returned an out-of-range index");
            }
            for left in 0..sample.indices.len() {
                for right in (left + 1)..sample.indices.len() {
                    if sample.indices[left] == sample.indices[right] {
                        bail!("gpu pnp-focal sampler returned duplicate sample indices");
                    }
                }
            }
        }
        Ok(samples.into_iter().map(|sample| sample.indices).collect())
    }
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

fn uniform_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuPnpFocalCandidate {
    pub(crate) pose: SE3,
    pub(crate) focal: f32,
}

pub(crate) struct WgpuPnPFocalScorer {
    scorer: WgpuPnpModelScorer,
    centered_points: Vec<[f32; 2]>,
    object_points: Vec<[f32; 3]>,
    threshold_px: f32,
}

impl WgpuPnPFocalScorer {
    pub(crate) fn from_context(context: Arc<WgpuContext>) -> Result<Self> {
        Ok(Self {
            scorer: WgpuPnpModelScorer::from_context(context)?,
            centered_points: Vec::new(),
            object_points: Vec::new(),
            threshold_px: 0.0,
        })
    }

    pub(crate) fn prepare(
        &mut self,
        centered_points: &[[f32; 2]],
        object_points: &[[f32; 3]],
        threshold_px: f32,
    ) -> Result<()> {
        if centered_points.len() != object_points.len()
            || centered_points.len() < 4
            || !threshold_px.is_finite()
            || threshold_px <= 0.0
        {
            bail!("invalid gpu focal pnp observations");
        }
        self.centered_points = centered_points.to_vec();
        self.object_points = object_points.to_vec();
        self.threshold_px = threshold_px;
        Ok(())
    }

    pub(crate) fn score(
        &mut self,
        candidate: GpuPnpFocalCandidate,
    ) -> Result<rustslam::tracker::PnPModelSupport> {
        if !candidate.focal.is_finite() || candidate.focal <= 0.0 {
            bail!("gpu focal pnp candidate has invalid focal length");
        }
        let normalized = self
            .centered_points
            .iter()
            .map(|p| [p[0] / candidate.focal, p[1] / candidate.focal])
            .collect::<Vec<_>>();
        self.scorer.prepare(
            &normalized,
            &self.object_points,
            self.threshold_px / candidate.focal,
        )?;
        self.scorer
            .score_models(&[candidate.pose])
            .map(|mut values| values.remove(0))
    }
}
