use super::WgpuContext;
use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};
use rustslam::tracker::{PnPModelScorer, PnPModelSupport};
use rustslam::SE3;
use std::sync::Arc;
use wgpu::util::DeviceExt;

const PNP_SCORING_SHADER: &str = include_str!("shaders/pnp_scoring.wgsl");
const PNP_WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuPnpImagePoint {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) pad0: f32,
    pub(crate) pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuPnpObjectPoint {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) z: f32,
    pub(crate) pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct GpuPnpModel {
    pub(crate) row0: [f32; 4],
    pub(crate) row1: [f32; 4],
    pub(crate) row2: [f32; 4],
}

impl GpuPnpModel {
    fn from_se3(model: &SE3) -> Result<Self> {
        let matrix = model.to_matrix();
        let rows = [matrix[0], matrix[1], matrix[2]];
        if rows.iter().flatten().any(|value| !value.is_finite()) {
            bail!("gpu pnp model contains non-finite values");
        }
        Ok(Self {
            row0: rows[0],
            row1: rows[1],
            row2: rows[2],
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PnpScoringParams {
    model_count: u32,
    observation_count: u32,
    selected_model: u32,
    pad0: u32,
    max_residual: f32,
    pad1: f32,
    pad2: f32,
    pad3: f32,
}

pub struct WgpuPnpModelScorer {
    context: Arc<WgpuContext>,
    bind_group_layout: wgpu::BindGroupLayout,
    score_pipeline: wgpu::ComputePipeline,
    mask_pipeline: wgpu::ComputePipeline,
    image_buffer: Option<wgpu::Buffer>,
    object_buffer: Option<wgpu::Buffer>,
    observation_count: usize,
    threshold: f32,
    last_scored_models: Vec<GpuPnpModel>,
    last_supports: Vec<PnPModelSupport>,
}

impl WgpuPnpModelScorer {
    pub fn try_new() -> Result<Self> {
        Self::from_context(WgpuContext::try_new()?)
    }

    pub fn from_context(context: Arc<WgpuContext>) -> Result<Self> {
        let device = context.device();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rustsfm gpu pnp bind group layout"),
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
            label: Some("rustsfm gpu pnp scoring shader"),
            source: wgpu::ShaderSource::Wgsl(PNP_SCORING_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rustsfm gpu pnp pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let score_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rustsfm gpu pnp support pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("score_models"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let mask_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rustsfm gpu pnp mask pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("write_mask"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        if let Some(error) = pollster::block_on(error_scope.pop()) {
            bail!("gpu pnp pipeline creation failed: {error}");
        }

        Ok(Self {
            context,
            bind_group_layout,
            score_pipeline,
            mask_pipeline,
            image_buffer: None,
            object_buffer: None,
            observation_count: 0,
            threshold: 0.0,
            last_scored_models: Vec::new(),
            last_supports: Vec::new(),
        })
    }

    pub fn prepare(
        &mut self,
        normalized_points: &[[f32; 2]],
        object_points: &[[f32; 3]],
        threshold: f32,
    ) -> Result<()> {
        validate_observations(normalized_points, object_points)?;
        validate_threshold(threshold)?;
        let image_points = normalized_points
            .iter()
            .copied()
            .map(|point| GpuPnpImagePoint {
                x: point[0],
                y: point[1],
                pad0: 0.0,
                pad1: 0.0,
            })
            .collect::<Vec<_>>();
        let object_points = object_points
            .iter()
            .copied()
            .map(|point| GpuPnpObjectPoint {
                x: point[0],
                y: point[1],
                z: point[2],
                pad: 0.0,
            })
            .collect::<Vec<_>>();
        self.validate_storage_slice("image observations", &image_points)?;
        self.validate_storage_slice("object observations", &object_points)?;
        let device = self.context.device();
        self.image_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rustsfm gpu pnp image observations"),
                contents: bytemuck::cast_slice(&image_points),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        );
        self.object_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rustsfm gpu pnp object observations"),
                contents: bytemuck::cast_slice(&object_points),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        );
        self.observation_count = normalized_points.len();
        self.threshold = threshold;
        self.last_scored_models.clear();
        self.last_supports.clear();
        Ok(())
    }

    pub fn score_models(&mut self, models: &[SE3]) -> Result<Vec<PnPModelSupport>> {
        let models = models
            .iter()
            .map(GpuPnpModel::from_se3)
            .collect::<Result<Vec<_>>>()?;
        self.last_scored_models.clear();
        self.last_supports.clear();
        let supports = self.score_gpu_models(&models)?;
        self.last_scored_models = models;
        self.last_supports = supports.clone();
        Ok(supports)
    }

    pub fn inlier_mask(&mut self, model: &SE3) -> Result<Vec<bool>> {
        let model = GpuPnpModel::from_se3(model)?;
        let expected = self
            .last_scored_models
            .iter()
            .zip(&self.last_supports)
            .find_map(|(scored_model, support)| (*scored_model == model).then_some(*support))
            .context("gpu pnp selected model is missing from the latest scoring batch")?;
        let mask = self.read_gpu_mask(&model)?;
        let mask_inliers = mask.iter().filter(|&&value| value).count();
        if mask_inliers != expected.inliers {
            bail!(
                "gpu pnp selected model support mismatch: summary has {}, mask has {}",
                expected.inliers,
                mask_inliers
            );
        }
        Ok(mask)
    }

    fn score_gpu_models(&self, models: &[GpuPnpModel]) -> Result<Vec<PnPModelSupport>> {
        let observation_count = self.prepared_observation_count()?;
        if models.is_empty() {
            return Ok(Vec::new());
        }
        let model_count = u32::try_from(models.len()).context("gpu pnp model count exceeds u32")?;
        self.validate_dispatch_count(model_count, "model")?;
        self.validate_storage_slice("models", models)?;
        self.validate_storage_elements::<super::GpuModelSupport>(
            "support summaries",
            models.len(),
        )?;

        let params = PnpScoringParams {
            model_count,
            observation_count: u32::try_from(observation_count)
                .context("gpu pnp observation count exceeds u32")?,
            selected_model: 0,
            pad0: 0,
            max_residual: squared_threshold(self.threshold),
            pad1: 0.0,
            pad2: 0.0,
            pad3: 0.0,
        };
        let (bind_group, summaries, _mask) = self.create_bind_group(
            models,
            &params,
            buffer_size::<super::GpuModelSupport>(models.len())?,
            std::mem::size_of::<u32>() as u64,
        )?;
        let device = self.context.device();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rustsfm gpu pnp support encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustsfm gpu pnp support pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.score_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(model_count, 1, 1);
        }
        self.context.queue().submit(Some(encoder.finish()));
        let summaries = self
            .context
            .read_buffer::<super::GpuModelSupport>(&summaries, models.len())
            .context("gpu pnp support readback")?;
        if summaries.len() != models.len() {
            bail!(
                "gpu pnp support readback count mismatch: expected {}, got {}",
                models.len(),
                summaries.len()
            );
        }
        summaries
            .into_iter()
            .enumerate()
            .map(|(index, summary)| {
                if summary.inliers as usize > observation_count {
                    bail!(
                        "gpu pnp model {index} reports {} inliers for {} observations",
                        summary.inliers,
                        observation_count
                    );
                }
                if !summary.residual_sum.is_finite() || summary.residual_sum < 0.0 {
                    bail!("gpu pnp model {index} reports invalid residual sum");
                }
                Ok(PnPModelSupport {
                    inliers: summary.inliers as usize,
                    residual_sum: f64::from(summary.residual_sum),
                })
            })
            .collect()
    }

    fn read_gpu_mask(&self, model: &GpuPnpModel) -> Result<Vec<bool>> {
        let observation_count = self.prepared_observation_count()?;
        let workgroups = u32::try_from(observation_count.div_ceil(PNP_WORKGROUP_SIZE as usize))
            .context("gpu pnp mask dispatch count exceeds u32")?;
        self.validate_dispatch_count(workgroups, "mask")?;
        self.validate_storage_slice("mask model", std::slice::from_ref(model))?;
        self.validate_storage_elements::<u32>("inlier mask", observation_count)?;
        let params = PnpScoringParams {
            model_count: 1,
            observation_count: u32::try_from(observation_count)
                .context("gpu pnp observation count exceeds u32")?,
            selected_model: 0,
            pad0: 0,
            max_residual: squared_threshold(self.threshold),
            pad1: 0.0,
            pad2: 0.0,
            pad3: 0.0,
        };
        let (bind_group, _summaries, mask) = self.create_bind_group(
            std::slice::from_ref(model),
            &params,
            std::mem::size_of::<super::GpuModelSupport>() as u64,
            buffer_size::<u32>(observation_count)?,
        )?;
        let device = self.context.device();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rustsfm gpu pnp mask encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustsfm gpu pnp mask pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.mask_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.context.queue().submit(Some(encoder.finish()));
        let values = self
            .context
            .read_buffer::<u32>(&mask, observation_count)
            .context("gpu pnp mask readback")?;
        if values.len() != observation_count {
            bail!(
                "gpu pnp mask readback count mismatch: expected {}, got {}",
                observation_count,
                values.len()
            );
        }
        Ok(values.into_iter().map(|value| value != 0).collect())
    }

    fn create_bind_group(
        &self,
        models: &[GpuPnpModel],
        params: &PnpScoringParams,
        summary_size: u64,
        mask_size: u64,
    ) -> Result<(wgpu::BindGroup, wgpu::Buffer, wgpu::Buffer)> {
        let image_buffer = self
            .image_buffer
            .as_ref()
            .context("gpu pnp image observations are not prepared")?;
        let object_buffer = self
            .object_buffer
            .as_ref()
            .context("gpu pnp object observations are not prepared")?;
        let device = self.context.device();
        let model_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp models"),
            contents: bytemuck::cast_slice(models),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let summaries = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp support summaries"),
            size: summary_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mask = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp inlier mask"),
            size: mask_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp params"),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustsfm gpu pnp bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                buffer_entry(0, &model_buffer),
                buffer_entry(1, image_buffer),
                buffer_entry(2, object_buffer),
                buffer_entry(3, &summaries),
                buffer_entry(4, &mask),
                buffer_entry(5, &params_buffer),
            ],
        });
        Ok((bind_group, summaries, mask))
    }

    fn prepared_observation_count(&self) -> Result<usize> {
        if self.observation_count == 0
            || self.image_buffer.is_none()
            || self.object_buffer.is_none()
        {
            bail!("gpu pnp scorer has not been prepared");
        }
        Ok(self.observation_count)
    }

    fn validate_dispatch_count(&self, workgroups: u32, label: &str) -> Result<()> {
        let limit = self
            .context
            .device()
            .limits()
            .max_compute_workgroups_per_dimension;
        if workgroups > limit {
            bail!("gpu pnp {label} dispatch requires {workgroups} workgroups, limit is {limit}");
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
            bail!("gpu pnp {label} buffer requires {bytes} bytes, limit is {limit}");
        }
        Ok(())
    }
}

impl PnPModelScorer for WgpuPnpModelScorer {
    type Error = anyhow::Error;

    fn prepare(
        &mut self,
        normalized_points: &[[f32; 2]],
        object_points: &[[f32; 3]],
        threshold: f32,
    ) -> Result<(), Self::Error> {
        WgpuPnpModelScorer::prepare(self, normalized_points, object_points, threshold)
    }

    fn score_models(&mut self, models: &[SE3]) -> Result<Vec<PnPModelSupport>, Self::Error> {
        WgpuPnpModelScorer::score_models(self, models)
    }

    fn inlier_mask(&mut self, model: &SE3) -> Result<Vec<bool>, Self::Error> {
        WgpuPnpModelScorer::inlier_mask(self, model)
    }
}

fn validate_observations(image_points: &[[f32; 2]], object_points: &[[f32; 3]]) -> Result<()> {
    if image_points.is_empty() || object_points.is_empty() {
        bail!("gpu pnp observations must be non-empty");
    }
    if image_points.len() != object_points.len() {
        bail!(
            "gpu pnp observation count mismatch: {} != {}",
            image_points.len(),
            object_points.len()
        );
    }
    if image_points
        .iter()
        .flatten()
        .chain(object_points.iter().flatten())
        .any(|value| !value.is_finite())
    {
        bail!("gpu pnp observations contain non-finite values");
    }
    u32::try_from(image_points.len()).context("gpu pnp observation count exceeds u32")?;
    Ok(())
}

fn validate_threshold(threshold: f32) -> Result<()> {
    if !threshold.is_finite() || threshold < 0.0 {
        bail!("gpu pnp threshold must be finite and non-negative");
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
        .context("gpu pnp buffer size overflow")
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

fn buffer_entry<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
