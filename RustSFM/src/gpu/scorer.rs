use super::{WgpuContext, WgpuModelScorerTiming};
use anyhow::{bail, Context, Result};
use std::sync::Arc;
use std::time::Instant;
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuHomogeneousPoint {
    x: f32,
    y: f32,
    z: f32,
    pad: f32,
}

impl GpuHomogeneousPoint {
    fn from_xy(point: [f32; 2]) -> Self {
        Self {
            x: point[0],
            y: point[1],
            z: 1.0,
            pad: 0.0,
        }
    }

    fn from_xyz(point: [f32; 3]) -> Self {
        Self {
            x: point[0],
            y: point[1],
            z: point[2],
            pad: 0.0,
        }
    }
}

pub struct WgpuModelScorer {
    context: Arc<WgpuContext>,
    bind_group_layout: wgpu::BindGroupLayout,
    score_pipeline: wgpu::ComputePipeline,
    mask_pipeline: wgpu::ComputePipeline,
}

pub(crate) struct WgpuModelScoringSession<'a> {
    scorer: &'a WgpuModelScorer,
    points1: wgpu::Buffer,
    points2: wgpu::Buffer,
    observation_count: u32,
    observation_len: usize,
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
        validate_point_counts(points1.len(), points2.len())?;
        validate_threshold(threshold)?;
        if models.is_empty() {
            return Ok(Vec::new());
        }
        if points1.is_empty() {
            return Ok(vec![GpuModelSupport::default(); models.len()]);
        }
        let points1 = points1
            .iter()
            .copied()
            .map(GpuHomogeneousPoint::from_xy)
            .collect::<Vec<_>>();
        let points2 = points2
            .iter()
            .copied()
            .map(GpuHomogeneousPoint::from_xy)
            .collect::<Vec<_>>();
        self.prepare_session(&points1, &points2)?
            .score_two_view_models(models, threshold, kind)
    }

    #[cfg(test)]
    pub(crate) fn score_homogeneous_two_view_models(
        &self,
        models: &[[f32; 9]],
        points1: &[[f32; 3]],
        points2: &[[f32; 3]],
        threshold: f32,
        kind: TwoViewModelKind,
    ) -> Result<Vec<GpuModelSupport>> {
        validate_point_counts(points1.len(), points2.len())?;
        validate_threshold(threshold)?;
        if models.is_empty() {
            return Ok(Vec::new());
        }
        if points1.is_empty() {
            return Ok(vec![GpuModelSupport::default(); models.len()]);
        }
        self.prepare_homogeneous_session(points1, points2)?
            .score_two_view_models(models, threshold, kind)
    }

    pub(crate) fn prepare_homogeneous_session(
        &self,
        points1: &[[f32; 3]],
        points2: &[[f32; 3]],
    ) -> Result<WgpuModelScoringSession<'_>> {
        validate_point_counts(points1.len(), points2.len())?;
        if points1.is_empty() {
            bail!("GPU model scoring session requires at least one observation");
        }
        let points1 = points1
            .iter()
            .copied()
            .map(GpuHomogeneousPoint::from_xyz)
            .collect::<Vec<_>>();
        let points2 = points2
            .iter()
            .copied()
            .map(GpuHomogeneousPoint::from_xyz)
            .collect::<Vec<_>>();
        self.prepare_session(&points1, &points2)
    }

    fn prepare_session(
        &self,
        points1: &[GpuHomogeneousPoint],
        points2: &[GpuHomogeneousPoint],
    ) -> Result<WgpuModelScoringSession<'_>> {
        let observation_count = validate_point_counts(points1.len(), points2.len())?;
        self.validate_storage_slice("first points", points1)?;
        self.validate_storage_slice("second points", points2)?;
        let device = self.context.device();
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
        Ok(WgpuModelScoringSession {
            scorer: self,
            points1: points1_buffer,
            points2: points2_buffer,
            observation_count,
            observation_len: points1.len(),
        })
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
        validate_point_counts(points1.len(), points2.len())?;
        validate_threshold(threshold)?;
        if points1.is_empty() {
            return Ok(Vec::new());
        }
        let points1 = points1
            .iter()
            .copied()
            .map(GpuHomogeneousPoint::from_xy)
            .collect::<Vec<_>>();
        let points2 = points2
            .iter()
            .copied()
            .map(GpuHomogeneousPoint::from_xy)
            .collect::<Vec<_>>();
        self.prepare_session(&points1, &points2)?
            .inlier_mask(model, threshold, kind)
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

impl WgpuModelScoringSession<'_> {
    pub(crate) fn score_two_view_models(
        &self,
        models: &[[f32; 9]],
        threshold: f32,
        kind: TwoViewModelKind,
    ) -> Result<Vec<GpuModelSupport>> {
        self.score_two_view_models_profiled(models, threshold, kind)
            .map(|(supports, _)| supports)
    }

    pub(crate) fn score_two_view_models_profiled(
        &self,
        models: &[[f32; 9]],
        threshold: f32,
        kind: TwoViewModelKind,
    ) -> Result<(Vec<GpuModelSupport>, WgpuModelScorerTiming)> {
        validate_threshold(threshold)?;
        let model_count = u32::try_from(models.len()).context("GPU model count exceeds u32")?;
        if models.is_empty() {
            return Ok((Vec::new(), WgpuModelScorerTiming::default()));
        }
        self.scorer.validate_dispatch_count(model_count, "model")?;
        self.scorer.validate_storage_slice("models", models)?;
        self.scorer
            .validate_storage_elements::<GpuModelSupport>("support summaries", models.len())?;

        let buffer_prepare_started = Instant::now();
        let params = ScoringParams {
            model_count,
            observation_count: self.observation_count,
            model_kind: kind.shader_value(),
            selected_model: 0,
            max_residual: squared_threshold(threshold),
            pad0: 0.0,
            pad1: 0.0,
            pad2: 0.0,
        };
        let (bind_group, summaries, _mask) = self.create_bind_group(
            models,
            &params,
            buffer_size::<GpuModelSupport>(models.len())?,
            std::mem::size_of::<u32>() as u64,
        );
        let mut encoder =
            self.scorer
                .context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rustsfm model scorer support encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustsfm model scorer support pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scorer.score_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(model_count, 1, 1);
        }
        let command_buffer = encoder.finish();
        let buffer_prepare_seconds = buffer_prepare_started.elapsed().as_secs_f64();
        let submit_started = Instant::now();
        self.scorer.context.queue().submit(Some(command_buffer));
        let submit_seconds = submit_started.elapsed().as_secs_f64();
        let (supports, readback) = self
            .scorer
            .context
            .read_buffer_profiled::<GpuModelSupport>(&summaries, models.len())?;
        Ok((
            supports,
            WgpuModelScorerTiming {
                buffer_prepare_seconds,
                submit_seconds,
                readback_total_seconds: readback.total_seconds,
                readback_copy_submit_seconds: readback.copy_submit_seconds,
                readback_wait_seconds: readback.wait_seconds,
                readback_map_decode_seconds: readback.map_decode_seconds,
                score_calls: 1,
                mask_calls: 0,
                models_scored: models.len(),
                readback_calls: readback.calls,
                readback_bytes: readback.bytes,
            },
        ))
    }

    pub(crate) fn inlier_mask(
        &self,
        model: &[f32; 9],
        threshold: f32,
        kind: TwoViewModelKind,
    ) -> Result<Vec<bool>> {
        self.inlier_mask_profiled(model, threshold, kind)
            .map(|(mask, _)| mask)
    }

    pub(crate) fn inlier_mask_profiled(
        &self,
        model: &[f32; 9],
        threshold: f32,
        kind: TwoViewModelKind,
    ) -> Result<(Vec<bool>, WgpuModelScorerTiming)> {
        validate_threshold(threshold)?;
        let workgroups = self.observation_count.div_ceil(64);
        self.scorer.validate_dispatch_count(workgroups, "mask")?;
        self.scorer
            .validate_storage_slice("model", std::slice::from_ref(model))?;
        self.scorer
            .validate_storage_elements::<u32>("inlier mask", self.observation_len)?;

        let buffer_prepare_started = Instant::now();
        let params = ScoringParams {
            model_count: 1,
            observation_count: self.observation_count,
            model_kind: kind.shader_value(),
            selected_model: 0,
            max_residual: squared_threshold(threshold),
            pad0: 0.0,
            pad1: 0.0,
            pad2: 0.0,
        };
        let (bind_group, _summaries, mask) = self.create_bind_group(
            std::slice::from_ref(model),
            &params,
            std::mem::size_of::<GpuModelSupport>() as u64,
            buffer_size::<u32>(self.observation_len)?,
        );
        let mut encoder =
            self.scorer
                .context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rustsfm model scorer mask encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustsfm model scorer mask pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scorer.mask_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        let command_buffer = encoder.finish();
        let buffer_prepare_seconds = buffer_prepare_started.elapsed().as_secs_f64();
        let submit_started = Instant::now();
        self.scorer.context.queue().submit(Some(command_buffer));
        let submit_seconds = submit_started.elapsed().as_secs_f64();
        let (mask, readback) = self
            .scorer
            .context
            .read_buffer_profiled::<u32>(&mask, self.observation_len)?;
        Ok((
            mask.into_iter().map(|value| value != 0).collect(),
            WgpuModelScorerTiming {
                buffer_prepare_seconds,
                submit_seconds,
                readback_total_seconds: readback.total_seconds,
                readback_copy_submit_seconds: readback.copy_submit_seconds,
                readback_wait_seconds: readback.wait_seconds,
                readback_map_decode_seconds: readback.map_decode_seconds,
                score_calls: 0,
                mask_calls: 1,
                models_scored: 0,
                readback_calls: readback.calls,
                readback_bytes: readback.bytes,
            },
        ))
    }

    fn create_bind_group(
        &self,
        models: &[[f32; 9]],
        params: &ScoringParams,
        summary_size: u64,
        mask_size: u64,
    ) -> (wgpu::BindGroup, wgpu::Buffer, wgpu::Buffer) {
        let device = self.scorer.context.device();
        let model_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm model scorer models"),
            contents: bytemuck::cast_slice(models),
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
            layout: &self.scorer.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.points1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.points2.as_entire_binding(),
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
}

fn validate_point_counts(points1_len: usize, points2_len: usize) -> Result<u32> {
    if points1_len != points2_len {
        bail!(
            "GPU model scorer point count mismatch: {} != {}",
            points1_len,
            points2_len
        );
    }
    u32::try_from(points1_len).context("GPU model scorer observation count exceeds u32")
}

fn validate_threshold(threshold: f32) -> Result<()> {
    if !threshold.is_finite() || threshold < 0.0 {
        bail!("GPU model scorer threshold must be finite and non-negative");
    }
    Ok(())
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
