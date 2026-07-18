use super::WgpuContext;
use anyhow::{bail, Context, Result};
use std::sync::Arc;
use wgpu::util::DeviceExt;

const MODEL_SCORING_SHADER: &str = include_str!("shaders/model_scoring.wgsl");

/// Compact support read back for one model.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuModelSupport {
    pub inliers: u32,
    pub residual_sum: f32,
}

/// Residual definition used to score a row-major 3x3 two-view model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TwoViewModelKind {
    Sampson,
    HomographyForward,
}

impl TwoViewModelKind {
    fn shader_value(self) -> u32 {
        match self {
            Self::HomographyForward => 0,
            Self::Sampson => 1,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScoringParams {
    model_count: u32,
    observation_count: u32,
    model_kind: u32,
    selected_model: u32,
    max_residual: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

pub struct WgpuModelScorer {
    context: Arc<WgpuContext>,
    bind_group_layout: wgpu::BindGroupLayout,
    score_pipeline: wgpu::ComputePipeline,
    mask_pipeline: wgpu::ComputePipeline,
}

impl WgpuModelScorer {
    pub fn try_new() -> Result<Self> {
        Self::from_context(WgpuContext::try_new()?)
    }

    pub fn from_context(context: Arc<WgpuContext>) -> Result<Self> {
        let device = context.device();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rustsfm model scorer bind group layout"),
            entries: &[
                storage_layout_entry(0, true),
                storage_layout_entry(1, true),
                storage_layout_entry(2, true),
                storage_layout_entry(3, false),
                storage_layout_entry(4, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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
            label: Some("rustsfm model scorer shader"),
            source: wgpu::ShaderSource::Wgsl(MODEL_SCORING_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rustsfm model scorer pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let score_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rustsfm model scorer support pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("score_models"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let mask_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rustsfm model scorer mask pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("write_mask"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Ok(Self {
            context,
            bind_group_layout,
            score_pipeline,
            mask_pipeline,
        })
    }

    /// Scores row-major models using the same unsquared threshold convention as the CPU path.
    pub fn score_two_view_models(
        &self,
        models: &[[f32; 9]],
        points1: &[[f32; 2]],
        points2: &[[f32; 2]],
        threshold: f32,
        kind: TwoViewModelKind,
    ) -> Result<Vec<GpuModelSupport>> {
        let observation_count = validate_observations(points1, points2, threshold)?;
        let model_count = u32::try_from(models.len()).context("GPU model count exceeds u32")?;
        if models.is_empty() {
            return Ok(Vec::new());
        }
        self.validate_dispatch_count(model_count, "model")?;
        if points1.is_empty() {
            return Ok(vec![GpuModelSupport::default(); models.len()]);
        }
        self.validate_storage_slice("models", models)?;
        self.validate_storage_slice("first points", points1)?;
        self.validate_storage_slice("second points", points2)?;
        self.validate_storage_elements::<GpuModelSupport>("support summaries", models.len())?;

        let params = ScoringParams {
            model_count,
            observation_count,
            model_kind: kind.shader_value(),
            selected_model: 0,
            max_residual: squared_threshold(threshold),
            pad0: 0.0,
            pad1: 0.0,
            pad2: 0.0,
        };
        let (bind_group, summaries, _mask) = self.create_bind_group(
            models,
            points1,
            points2,
            &params,
            buffer_size::<GpuModelSupport>(models.len())?,
            std::mem::size_of::<u32>() as u64,
        );
        let mut encoder =
            self.context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rustsfm model scorer support encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustsfm model scorer support pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.score_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(model_count, 1, 1);
        }
        self.context.queue().submit(Some(encoder.finish()));
        self.context
            .read_buffer::<GpuModelSupport>(&summaries, models.len())
    }

    /// Returns the observation mask for one row-major model and an unsquared threshold.
    pub fn inlier_mask(
        &self,
        model: &[f32; 9],
        points1: &[[f32; 2]],
        points2: &[[f32; 2]],
        threshold: f32,
        kind: TwoViewModelKind,
    ) -> Result<Vec<bool>> {
        let observation_count = validate_observations(points1, points2, threshold)?;
        if points1.is_empty() {
            return Ok(Vec::new());
        }
        let workgroups = observation_count.div_ceil(64);
        self.validate_dispatch_count(workgroups, "mask")?;
        self.validate_storage_slice("model", std::slice::from_ref(model))?;
        self.validate_storage_slice("first points", points1)?;
        self.validate_storage_slice("second points", points2)?;
        self.validate_storage_elements::<u32>("inlier mask", points1.len())?;

        let params = ScoringParams {
            model_count: 1,
            observation_count,
            model_kind: kind.shader_value(),
            selected_model: 0,
            max_residual: squared_threshold(threshold),
            pad0: 0.0,
            pad1: 0.0,
            pad2: 0.0,
        };
        let (bind_group, _summaries, mask) = self.create_bind_group(
            std::slice::from_ref(model),
            points1,
            points2,
            &params,
            std::mem::size_of::<GpuModelSupport>() as u64,
            buffer_size::<u32>(points1.len())?,
        );
        let mut encoder =
            self.context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rustsfm model scorer mask encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustsfm model scorer mask pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.mask_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.context.queue().submit(Some(encoder.finish()));
        Ok(self
            .context
            .read_buffer::<u32>(&mask, points1.len())?
            .into_iter()
            .map(|value| value != 0)
            .collect())
    }

    fn create_bind_group(
        &self,
        models: &[[f32; 9]],
        points1: &[[f32; 2]],
        points2: &[[f32; 2]],
        params: &ScoringParams,
        summary_size: u64,
        mask_size: u64,
    ) -> (wgpu::BindGroup, wgpu::Buffer, wgpu::Buffer) {
        let device = self.context.device();
        let model_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm model scorer models"),
            contents: bytemuck::cast_slice(models),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let points1_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm model scorer first points"),
            contents: bytemuck::cast_slice(points1),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let points2_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm model scorer second points"),
            contents: bytemuck::cast_slice(points2),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let summaries = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm model scorer summaries"),
            size: summary_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mask = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm model scorer mask"),
            size: mask_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm model scorer params"),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustsfm model scorer bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: points1_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: points2_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: summaries.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: mask.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        (bind_group, summaries, mask)
    }

    fn validate_dispatch_count(&self, workgroups: u32, label: &str) -> Result<()> {
        let limit = self
            .context
            .device()
            .limits()
            .max_compute_workgroups_per_dimension;
        if workgroups > limit {
            bail!(
                "GPU model scorer {label} dispatch requires {workgroups} workgroups, limit is {limit}"
            );
        }
        Ok(())
    }

    fn validate_storage_slice<T>(&self, label: &str, values: &[T]) -> Result<()> {
        self.validate_storage_elements::<T>(label, values.len())
    }

    fn validate_storage_elements<T>(&self, label: &str, count: usize) -> Result<()> {
        let bytes = buffer_size::<T>(count)?;
        let limits = self.context.device().limits();
        let limit = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size);
        if bytes > limit {
            bail!("GPU model scorer {label} buffer requires {bytes} bytes, limit is {limit}");
        }
        Ok(())
    }
}

fn validate_observations(
    points1: &[[f32; 2]],
    points2: &[[f32; 2]],
    threshold: f32,
) -> Result<u32> {
    if points1.len() != points2.len() {
        bail!(
            "GPU model scorer point count mismatch: {} != {}",
            points1.len(),
            points2.len()
        );
    }
    if !threshold.is_finite() || threshold < 0.0 {
        bail!("GPU model scorer threshold must be finite and non-negative");
    }
    u32::try_from(points1.len()).context("GPU model scorer observation count exceeds u32")
}

fn squared_threshold(threshold: f32) -> f32 {
    threshold.max(1.0e-12).powi(2)
}

fn buffer_size<T>(count: usize) -> Result<u64> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .context("GPU model scorer buffer size overflow")
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
