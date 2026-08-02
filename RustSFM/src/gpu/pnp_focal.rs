use super::{
    pnp_scorer::{GpuPnpImagePoint, GpuPnpObjectPoint},
    WgpuContext,
};
use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};
use rustslam::SE3;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
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

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuPnpFocalSupport {
    inliers: u32,
    pad0: u32,
    residual_sum: f32,
    pad1: f32,
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

pub(crate) struct GpuPnpFocalSolution {
    pub(crate) pose: SE3,
    pub(crate) focal: f32,
    pub(crate) inliers: usize,
    pub(crate) initial_inliers: usize,
    pub(crate) inlier_mask: Vec<bool>,
}

pub(crate) struct WgpuPnPFocalSolver {
    sampler: WgpuPnPFocalSampler,
    generator: WgpuPnPFocalCandidateGenerator,
    refiner: WgpuPnPFocalRefiner,
    context: Arc<WgpuContext>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuPnpFocalCandidateParams {
    sample: [u32; 4],
    focal: f32,
    triple: u32,
    observation_count: u32,
    model_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuPnpFocalScoringParams {
    model_count: u32,
    observation_count: u32,
    selected_model: u32,
    pad1: u32,
    threshold_squared: f32,
    pad2: f32,
    pad3: f32,
    pad4: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuPnpFocalSelectionParams {
    model_count: u32,
    observation_count: u32,
    pad0: u32,
    pad1: u32,
    min_focal: f32,
    max_focal: f32,
    pad2: f32,
    pad3: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuPnpFocalAcceptanceParams {
    observation_count: u32,
    pad0: u32,
    min_focal: f32,
    max_focal: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuPnpFocalRefineParams {
    observation_count: u32,
    pad0: u32,
    damping: f32,
    pad1: f32,
}

pub(crate) struct WgpuPnPFocalRefiner {
    context: Arc<WgpuContext>,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

struct GpuPnpFocalRefinementState {
    current_model: wgpu::Buffer,
    current_support: wgpu::Buffer,
    current_mask: wgpu::Buffer,
    candidate_model: wgpu::Buffer,
    candidate_support: wgpu::Buffer,
    candidate_mask: wgpu::Buffer,
    candidate_status: wgpu::Buffer,
}

impl GpuPnpFocalRefinementState {
    fn new(context: &WgpuContext, observation_count: usize) -> Result<Self> {
        let device = context.device();
        let storage_limit = gpu_storage_limit(context);
        let model_bytes = checked_gpu_storage_bytes::<GpuPnpFocalModel>(1, storage_limit, "model")?;
        let support_bytes =
            checked_gpu_storage_bytes::<GpuPnpFocalSupport>(1, storage_limit, "support")?;
        let mask_bytes =
            checked_gpu_storage_bytes::<u32>(observation_count, storage_limit, "inlier mask")?;
        let current_model = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal current model"),
            size: model_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let current_support = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal current support"),
            size: support_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let current_mask = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal current mask"),
            size: mask_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let candidate_model = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal candidate model"),
            size: model_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let candidate_support = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal candidate support"),
            size: support_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let candidate_mask = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal candidate mask"),
            size: mask_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let candidate_status = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal candidate refinement status"),
            size: checked_gpu_storage_bytes::<u32>(1, storage_limit, "refinement status")?,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            current_model,
            current_support,
            current_mask,
            candidate_model,
            candidate_support,
            candidate_mask,
            candidate_status,
        })
    }
}

pub(crate) struct WgpuPnPFocalCandidateGenerator {
    context: Arc<WgpuContext>,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    batch_bind_group_layout: wgpu::BindGroupLayout,
    batch_pipeline: wgpu::ComputePipeline,
}

impl WgpuPnPFocalCandidateGenerator {
    pub(crate) fn from_context(context: Arc<WgpuContext>) -> Result<Self> {
        let device = context.device();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rustsfm gpu pnp-focal candidate bind group layout"),
            entries: &[
                storage_layout_entry(0, true),
                storage_layout_entry(1, true),
                storage_layout_entry(2, false),
                uniform_layout_entry(3),
            ],
        });
        let batch_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rustsfm gpu pnp-focal candidate batch bind group layout"),
                entries: &[
                    storage_layout_entry(0, true),
                    storage_layout_entry(1, true),
                    storage_layout_entry(2, true),
                    storage_layout_entry(3, false),
                ],
            });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rustsfm gpu pnp-focal candidate shader"),
            source: wgpu::ShaderSource::Wgsl(PNP_FOCAL_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rustsfm gpu pnp-focal candidate pipeline layout"),
            bind_group_layouts: &[
                None,
                Some(&bind_group_layout),
                None,
                Some(&batch_bind_group_layout),
            ],
            immediate_size: 0,
        });
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rustsfm gpu pnp-focal P3P candidate pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("generate_p3p_candidates"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        if let Some(error) = pollster::block_on(error_scope.pop()) {
            bail!("gpu pnp-focal candidate pipeline creation failed: {error}");
        }
        let batch_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rustsfm gpu pnp-focal candidate batch pipeline layout"),
                bind_group_layouts: &[
                    None,
                    Some(&bind_group_layout),
                    None,
                    Some(&batch_bind_group_layout),
                ],
                immediate_size: 0,
            });
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let batch_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rustsfm gpu pnp-focal P3P candidate batch pipeline"),
            layout: Some(&batch_pipeline_layout),
            module: &shader,
            entry_point: Some("generate_p3p_candidate_batch"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        if let Some(error) = pollster::block_on(error_scope.pop()) {
            bail!("gpu pnp-focal candidate batch pipeline creation failed: {error}");
        }
        Ok(Self {
            context,
            bind_group_layout,
            pipeline,
            batch_bind_group_layout,
            batch_pipeline,
        })
    }

    pub(crate) fn generate_p3p(
        &self,
        centered_points: &[[f32; 2]],
        object_points: &[[f32; 3]],
        sample: [u32; 4],
        focal: f32,
        triple: u32,
    ) -> Result<Vec<GpuPnpFocalModel>> {
        validate_solver_inputs(centered_points, object_points, 1.0, focal, focal)?;
        if triple >= 4
            || sample
                .iter()
                .any(|&index| index as usize >= centered_points.len())
        {
            bail!("gpu pnp-focal candidate sample is invalid");
        }
        let images = centered_points
            .iter()
            .map(|&point| GpuPnpImagePoint {
                x: point[0],
                y: point[1],
                pad0: 0.0,
                pad1: 0.0,
            })
            .collect::<Vec<_>>();
        let objects = object_points
            .iter()
            .map(|&point| GpuPnpObjectPoint {
                x: point[0],
                y: point[1],
                z: point[2],
                pad: 0.0,
            })
            .collect::<Vec<_>>();
        let device = self.context.device();
        let image_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp-focal candidate images"),
            contents: bytemuck::cast_slice(&images),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let object_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp-focal candidate objects"),
            contents: bytemuck::cast_slice(&objects),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let models = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal candidate models"),
            size: (std::mem::size_of::<GpuPnpFocalModel>() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = GpuPnpFocalCandidateParams {
            sample,
            focal,
            triple,
            observation_count: u32::try_from(centered_points.len())?,
            model_offset: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp-focal candidate params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustsfm gpu pnp-focal candidate bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: image_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: object_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: models.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let batch_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp-focal single candidate batch parameters"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let batch_models = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal single candidate batch models"),
            size: (std::mem::size_of::<GpuPnpFocalModel>() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let batch_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustsfm gpu pnp-focal single candidate batch bind group"),
            layout: &self.batch_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: image_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: object_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: batch_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: batch_models.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rustsfm gpu pnp-focal candidate encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustsfm gpu pnp-focal candidate pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(1, &bind_group, &[]);
            pass.set_bind_group(3, &batch_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.context.queue().submit(Some(encoder.finish()));
        self.context.read_buffer::<GpuPnpFocalModel>(&models, 4)
    }

    pub(crate) fn generate_p3p_batch(
        &self,
        centered_points: &[[f32; 2]],
        object_points: &[[f32; 3]],
        parameters: &[([u32; 4], f32, u32)],
    ) -> Result<Vec<GpuPnpFocalModel>> {
        if parameters.is_empty() {
            return Ok(Vec::new());
        }
        for &(sample, focal, triple) in parameters {
            validate_solver_inputs(centered_points, object_points, 1.0, focal, focal)?;
            if triple >= 4
                || sample
                    .iter()
                    .any(|&index| index as usize >= centered_points.len())
            {
                bail!("gpu pnp-focal batch candidate sample is invalid");
            }
        }
        let dispatch_count = u32::try_from(parameters.len())?;
        if dispatch_count
            > self
                .context
                .device()
                .limits()
                .max_compute_workgroups_per_dimension
        {
            bail!("gpu pnp-focal batch candidate dispatch exceeds device workgroup limit");
        }
        let images = centered_points
            .iter()
            .map(|&point| GpuPnpImagePoint {
                x: point[0],
                y: point[1],
                pad0: 0.0,
                pad1: 0.0,
            })
            .collect::<Vec<_>>();
        let objects = object_points
            .iter()
            .map(|&point| GpuPnpObjectPoint {
                x: point[0],
                y: point[1],
                z: point[2],
                pad: 0.0,
            })
            .collect::<Vec<_>>();
        let observation_count = u32::try_from(centered_points.len())
            .context("gpu pnp-focal observation count exceeds u32")?;
        let parameters = parameters
            .iter()
            .map(|&(sample, focal, triple)| GpuPnpFocalCandidateParams {
                sample,
                focal,
                triple,
                observation_count,
                model_offset: 0,
            })
            .collect::<Vec<_>>();
        let model_count = parameters
            .len()
            .checked_mul(4)
            .context("gpu pnp-focal batch candidate count overflow")?;
        let device = self.context.device();
        let image_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp-focal batch candidate images"),
            contents: bytemuck::cast_slice(&images),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let object_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp-focal batch candidate objects"),
            contents: bytemuck::cast_slice(&objects),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let parameters_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp-focal batch candidate parameters"),
            contents: bytemuck::cast_slice(&parameters),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let models = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal batch candidate models"),
            size: u64::try_from(model_count * std::mem::size_of::<GpuPnpFocalModel>())?,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustsfm gpu pnp-focal batch candidate bind group"),
            layout: &self.batch_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: image_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: object_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: parameters_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: models.as_entire_binding(),
                },
            ],
        });
        let single_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp-focal batch candidate single parameters"),
            contents: bytemuck::bytes_of(&parameters[0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let single_models = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal batch candidate single models"),
            size: (std::mem::size_of::<GpuPnpFocalModel>() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let single_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustsfm gpu pnp-focal batch candidate single bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: image_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: object_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: single_models.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: single_params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rustsfm gpu pnp-focal batch candidate encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustsfm gpu pnp-focal batch candidate pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.batch_pipeline);
            pass.set_bind_group(1, &single_bind_group, &[]);
            pass.set_bind_group(3, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count, 1, 1);
        }
        self.context.queue().submit(Some(encoder.finish()));
        self.context
            .read_buffer::<GpuPnpFocalModel>(&models, model_count)
    }
}

impl WgpuPnPFocalSolver {
    pub(crate) fn from_context(context: Arc<WgpuContext>) -> Result<Self> {
        Ok(Self {
            sampler: WgpuPnPFocalSampler::from_context(context.clone())?,
            generator: WgpuPnPFocalCandidateGenerator::from_context(context.clone())?,
            refiner: WgpuPnPFocalRefiner::from_context(context.clone())?,
            context,
        })
    }

    fn generate_p3p_models(
        &self,
        centered_points: &[[f32; 2]],
        object_points: &[[f32; 3]],
        parameters: &[([u32; 4], f32, u32)],
    ) -> Result<(wgpu::Buffer, usize)> {
        if parameters.is_empty() {
            bail!("gpu pnp-focal candidate parameters are empty");
        }
        for &(sample, focal, triple) in parameters {
            validate_solver_inputs(centered_points, object_points, 1.0, focal, focal)?;
            if triple >= 4
                || sample
                    .iter()
                    .any(|&index| index as usize >= centered_points.len())
            {
                bail!("gpu pnp-focal candidate sample is invalid");
            }
        }
        let model_count = parameters
            .len()
            .checked_mul(4)
            .context("gpu pnp-focal candidate count overflow")?;
        let storage_limit = gpu_storage_limit(&self.context);
        let model_bytes =
            checked_gpu_storage_bytes::<GpuPnpFocalModel>(model_count, storage_limit, "models")?;
        let image_points = centered_points
            .iter()
            .map(|&point| GpuPnpImagePoint {
                x: point[0],
                y: point[1],
                pad0: 0.0,
                pad1: 0.0,
            })
            .collect::<Vec<_>>();
        let object_points = object_points
            .iter()
            .map(|&point| GpuPnpObjectPoint {
                x: point[0],
                y: point[1],
                z: point[2],
                pad: 0.0,
            })
            .collect::<Vec<_>>();
        checked_gpu_storage_bytes::<GpuPnpImagePoint>(
            image_points.len(),
            storage_limit,
            "image observations",
        )?;
        checked_gpu_storage_bytes::<GpuPnpObjectPoint>(
            object_points.len(),
            storage_limit,
            "object observations",
        )?;
        let device = self.context.device();
        let image_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp-focal candidate image observations"),
            contents: bytemuck::cast_slice(&image_points),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let object_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp-focal candidate object observations"),
            contents: bytemuck::cast_slice(&object_points),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let models = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal generated models"),
            size: model_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let placeholder = GpuPnpFocalCandidateParams {
            sample: parameters[0].0,
            focal: parameters[0].1,
            triple: parameters[0].2,
            observation_count: u32::try_from(centered_points.len())?,
            model_offset: 0,
        };
        let single_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp-focal candidate placeholder params"),
            contents: bytemuck::bytes_of(&placeholder),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let placeholder_models = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal candidate placeholder models"),
            size: checked_gpu_storage_bytes::<GpuPnpFocalModel>(4, storage_limit, "models")?,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let single_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustsfm gpu pnp-focal candidate placeholder bind group"),
            layout: &self.generator.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: image_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: object_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: placeholder_models.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: single_params.as_entire_binding(),
                },
            ],
        });
        let max_workgroups = usize::try_from(
            self.context
                .device()
                .limits()
                .max_compute_workgroups_per_dimension,
        )
        .context("gpu pnp-focal workgroup limit does not fit usize")?;
        let mut parameter_buffers = Vec::new();
        let mut bind_groups = Vec::new();
        for (chunk_index, chunk) in parameters.chunks(max_workgroups).enumerate() {
            let base_model = chunk_index
                .checked_mul(max_workgroups)
                .and_then(|index| index.checked_mul(4))
                .context("gpu pnp-focal model offset overflow")?;
            let params = chunk
                .iter()
                .enumerate()
                .map(|(index, &(sample, focal, triple))| {
                    Ok(GpuPnpFocalCandidateParams {
                        sample,
                        focal,
                        triple,
                        observation_count: u32::try_from(centered_points.len())?,
                        model_offset: u32::try_from(base_model + index * 4)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            checked_gpu_storage_bytes::<GpuPnpFocalCandidateParams>(
                params.len(),
                storage_limit,
                "candidate parameters",
            )?;
            parameter_buffers.push(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("rustsfm gpu pnp-focal candidate batch parameters"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
            );
            let parameters = parameter_buffers
                .last()
                .expect("parameter buffer was just pushed");
            bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rustsfm gpu pnp-focal candidate batch bind group"),
                layout: &self.generator.batch_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: image_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: object_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: parameters.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: models.as_entire_binding(),
                    },
                ],
            }));
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rustsfm gpu pnp-focal candidate generation encoder"),
        });
        for (chunk, bind_group) in parameters.chunks(max_workgroups).zip(&bind_groups) {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustsfm gpu pnp-focal candidate generation pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.generator.batch_pipeline);
            pass.set_bind_group(1, &single_bind_group, &[]);
            pass.set_bind_group(3, bind_group, &[]);
            pass.dispatch_workgroups(u32::try_from(chunk.len())?, 1, 1);
        }
        self.context.queue().submit(Some(encoder.finish()));
        Ok((models, model_count))
    }

    pub(crate) fn solve(
        &self,
        centered_points: &[[f32; 2]],
        object_points: &[[f32; 3]],
        threshold_px: f32,
        seed: u32,
        trial_count: usize,
        min_focal: f32,
        max_focal: f32,
    ) -> Result<Option<GpuPnpFocalSolution>> {
        validate_solver_inputs(
            centered_points,
            object_points,
            threshold_px,
            min_focal,
            max_focal,
        )?;
        let samples = self
            .sampler
            .sample_indices(seed, trial_count, centered_points.len())?;
        let focal_grid = focal_search_grid(min_focal, max_focal, 64)?;
        let triples = [0u32, 1, 2, 3];
        let parameter_count = validate_p3p_candidate_capacity(
            samples.len(),
            focal_grid.len(),
            triples.len(),
            gpu_storage_limit(&self.context),
            self.context
                .device()
                .limits()
                .max_compute_workgroups_per_dimension,
        )?;
        if parameter_count == 0 {
            return Ok(None);
        }
        let mut parameters = Vec::with_capacity(parameter_count);
        for sample in samples {
            for &focal in &focal_grid {
                for &triple in &triples {
                    parameters.push((sample, focal, triple));
                }
            }
        }
        let mut scorer = WgpuPnPFocalScorer::from_context(self.context.clone())?;
        scorer.prepare(centered_points, object_points, threshold_px)?;
        let (models, model_count) =
            self.generate_p3p_models(centered_points, object_points, &parameters)?;
        let state =
            scorer.initialize_refinement_state(&models, model_count, min_focal, max_focal)?;
        for _ in 0..10 {
            let image_points = scorer
                .image_buffer
                .as_ref()
                .context("gpu focal pnp observations are not prepared")?;
            let object_points = scorer
                .object_buffer
                .as_ref()
                .context("gpu focal pnp observations are not prepared")?;
            let mut encoder =
                self.context
                    .device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("rustsfm gpu pnp-focal refinement iteration encoder"),
                    });
            self.refiner.encode_refinement(
                &mut encoder,
                &state.current_model,
                image_points,
                object_points,
                &state.current_mask,
                &state.candidate_model,
                &state.candidate_status,
                centered_points.len(),
            )?;
            scorer.encode_score_and_mask(
                &mut encoder,
                &state.candidate_model,
                &state.candidate_support,
                &state.candidate_mask,
            )?;
            scorer.encode_acceptance(&mut encoder, &state, min_focal, max_focal)?;
            self.context.queue().submit(Some(encoder.finish()));
        }
        let support = self
            .context
            .read_buffer::<GpuPnpFocalSupport>(&state.current_support, 1)?[0];
        let mask = self
            .context
            .read_buffer::<u32>(&state.current_mask, centered_points.len())?;
        if support.inliers < 4
            || support.inliers as usize > centered_points.len()
            || mask.iter().filter(|&&inlier| inlier != 0).count() != support.inliers as usize
        {
            return Ok(None);
        }
        let model = self
            .context
            .read_buffer::<GpuPnpFocalModel>(&state.current_model, 1)?[0];
        let candidate = focal_model_to_candidate(model)
            .context("gpu focal pnp final model is invalid after GPU validation")?;
        let initial_inliers = support.pad0 as usize;
        if initial_inliers > centered_points.len() {
            bail!("gpu focal pnp initial support is invalid");
        }
        Ok(Some(GpuPnpFocalSolution {
            pose: candidate.pose,
            focal: candidate.focal,
            inliers: support.inliers as usize,
            initial_inliers,
            inlier_mask: mask.into_iter().map(|inlier| inlier != 0).collect(),
        }))
    }
}

impl WgpuPnPFocalRefiner {
    fn from_context(context: Arc<WgpuContext>) -> Result<Self> {
        let device = context.device();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rustsfm gpu pnp-focal refinement bind group layout"),
            entries: &[
                storage_layout_entry(0, true),
                storage_layout_entry(1, true),
                storage_layout_entry(2, true),
                storage_layout_entry(3, true),
                storage_layout_entry(4, false),
                storage_layout_entry(5, false),
                uniform_layout_entry(6),
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rustsfm gpu pnp-focal refinement shader"),
            source: wgpu::ShaderSource::Wgsl(PNP_FOCAL_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rustsfm gpu pnp-focal refinement pipeline layout"),
            bind_group_layouts: &[None, None, None, None, Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rustsfm gpu pnp-focal refinement pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("refine_focal_model"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        if let Some(error) = pollster::block_on(error_scope.pop()) {
            bail!("gpu pnp-focal refinement pipeline creation failed: {error}");
        }
        Ok(Self {
            context,
            bind_group_layout,
            pipeline,
        })
    }

    fn refine_once(
        &self,
        centered_points: &[[f32; 2]],
        object_points: &[[f32; 3]],
        candidate: GpuPnpFocalCandidate,
        inlier_mask: &[bool],
    ) -> Result<Option<GpuPnpFocalCandidate>> {
        if centered_points.len() != object_points.len()
            || centered_points.len() != inlier_mask.len()
            || inlier_mask.iter().filter(|&&inlier| inlier).count() < 4
        {
            return Ok(None);
        }
        let model = focal_candidate_to_model(candidate)?;
        let image_points = centered_points
            .iter()
            .map(|&point| GpuPnpImagePoint {
                x: point[0],
                y: point[1],
                pad0: 0.0,
                pad1: 0.0,
            })
            .collect::<Vec<_>>();
        let object_points = object_points
            .iter()
            .map(|&point| GpuPnpObjectPoint {
                x: point[0],
                y: point[1],
                z: point[2],
                pad: 0.0,
            })
            .collect::<Vec<_>>();
        let mask = inlier_mask
            .iter()
            .map(|&inlier| u32::from(inlier))
            .collect::<Vec<_>>();
        let device = self.context.device();
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp-focal refinement input model"),
            contents: bytemuck::bytes_of(&model),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let images = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp-focal refinement images"),
            contents: bytemuck::cast_slice(&image_points),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let objects = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp-focal refinement objects"),
            contents: bytemuck::cast_slice(&object_points),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mask = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp-focal refinement mask"),
            contents: bytemuck::cast_slice(&mask),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal refinement output model"),
            size: std::mem::size_of::<GpuPnpFocalModel>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let status = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal refinement status"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = GpuPnpFocalRefineParams {
            observation_count: u32::try_from(centered_points.len())?,
            pad0: 0,
            damping: 1.0e-3,
            pad1: 0.0,
        };
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp-focal refinement params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustsfm gpu pnp-focal refinement bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: images.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: objects.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: mask.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: status.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rustsfm gpu pnp-focal refinement encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustsfm gpu pnp-focal refinement pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(4, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.context.queue().submit(Some(encoder.finish()));
        if self.context.read_buffer::<u32>(&status, 1)? != [1] {
            return Ok(None);
        }
        Ok(self
            .context
            .read_buffer::<GpuPnpFocalModel>(&output, 1)?
            .into_iter()
            .next()
            .and_then(focal_model_to_candidate))
    }

    fn encode_refinement(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        image_points: &wgpu::Buffer,
        object_points: &wgpu::Buffer,
        mask: &wgpu::Buffer,
        output: &wgpu::Buffer,
        status: &wgpu::Buffer,
        observation_count: usize,
    ) -> Result<()> {
        let params = GpuPnpFocalRefineParams {
            observation_count: u32::try_from(observation_count)?,
            pad0: 0,
            damping: 1.0e-3,
            pad1: 0.0,
        };
        let params = self
            .context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rustsfm gpu pnp-focal refinement params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self
            .context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rustsfm gpu pnp-focal persistent refinement bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: image_points.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: object_points.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: mask.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: status.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rustsfm gpu pnp-focal persistent refinement pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(4, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        Ok(())
    }
}

fn validate_solver_inputs(
    centered_points: &[[f32; 2]],
    object_points: &[[f32; 3]],
    threshold_px: f32,
    min_focal: f32,
    max_focal: f32,
) -> Result<()> {
    if centered_points.len() != object_points.len() || centered_points.len() < 4 {
        bail!("gpu pnp-focal solver needs at least four paired observations");
    }
    if centered_points
        .iter()
        .flatten()
        .chain(object_points.iter().flatten())
        .any(|value| !value.is_finite())
    {
        bail!("gpu pnp-focal solver observations contain non-finite values");
    }
    if !threshold_px.is_finite() || threshold_px <= 0.0 {
        bail!("gpu pnp-focal solver threshold must be positive and finite");
    }
    if !min_focal.is_finite() || !max_focal.is_finite() || min_focal <= 0.0 || max_focal < min_focal
    {
        bail!("gpu pnp-focal solver focal bounds are invalid");
    }
    Ok(())
}

pub(crate) fn focal_search_grid(min_focal: f32, max_focal: f32, count: usize) -> Result<Vec<f32>> {
    if !min_focal.is_finite()
        || !max_focal.is_finite()
        || min_focal <= 0.0
        || max_focal < min_focal
        || count == 0
    {
        bail!("gpu pnp-focal search grid parameters are invalid");
    }
    if count == 1 || min_focal == max_focal {
        return Ok(vec![min_focal]);
    }

    let log_min = min_focal.ln();
    let log_step = (max_focal.ln() - log_min) / (count - 1) as f32;
    Ok((0..count)
        .map(|index| (log_min + log_step * index as f32).exp())
        .collect())
}

pub(crate) fn select_best_focal_support(
    supports: &[rustslam::tracker::PnPModelSupport],
) -> Option<usize> {
    supports
        .iter()
        .enumerate()
        .filter(|(_, support)| support.residual_sum.is_finite() && support.residual_sum >= 0.0)
        .max_by(|(left_index, left), (right_index, right)| {
            left.inliers
                .cmp(&right.inliers)
                .then_with(|| right.residual_sum.total_cmp(&left.residual_sum))
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
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

fn checked_gpu_storage_bytes<T>(count: usize, storage_limit: u64, label: &str) -> Result<u64> {
    let bytes = count
        .checked_mul(std::mem::size_of::<T>())
        .with_context(|| format!("gpu pnp-focal {label} buffer size overflow"))?;
    let bytes = u64::try_from(bytes)
        .with_context(|| format!("gpu pnp-focal {label} buffer size exceeds u64"))?;
    if bytes > storage_limit {
        bail!("gpu pnp-focal {label} buffer requires {bytes} bytes, limit is {storage_limit}");
    }
    Ok(bytes)
}

fn checked_mask_workgroups(
    observation_count: usize,
    max_workgroups: u32,
    label: &str,
) -> Result<u32> {
    let workgroups = u32::try_from(observation_count.div_ceil(64))
        .with_context(|| format!("gpu pnp-focal {label} mask dispatch count exceeds u32"))?;
    if workgroups > max_workgroups {
        bail!(
            "gpu pnp-focal {label} mask dispatch requires {workgroups} workgroups, limit is {max_workgroups}"
        );
    }
    Ok(workgroups)
}

fn validate_p3p_candidate_capacity(
    trial_count: usize,
    focal_count: usize,
    triple_count: usize,
    storage_limit: u64,
    max_workgroups: u32,
) -> Result<usize> {
    let parameter_count = trial_count
        .checked_mul(focal_count)
        .and_then(|count| count.checked_mul(triple_count))
        .context("gpu pnp-focal candidate parameter count overflow")?;
    let model_count = parameter_count
        .checked_mul(4)
        .context("gpu pnp-focal candidate count overflow")?;
    checked_gpu_storage_bytes::<GpuPnpFocalModel>(model_count, storage_limit, "models")?;
    let model_count =
        u32::try_from(model_count).context("gpu pnp-focal candidate count exceeds u32")?;
    if model_count > max_workgroups {
        bail!("gpu pnp-focal scoring requires {model_count} workgroups, limit is {max_workgroups}");
    }
    Ok(parameter_count)
}

fn gpu_storage_limit(context: &WgpuContext) -> u64 {
    let limits = context.device().limits();
    limits
        .max_buffer_size
        .min(limits.max_storage_buffer_binding_size)
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

fn focal_model_to_candidate(model: GpuPnpFocalModel) -> Option<GpuPnpFocalCandidate> {
    let focal = model.log_focal_and_padding[0].exp();
    let rows = [model.row0, model.row1, model.row2];
    if !focal.is_finite() || focal <= 0.0 || rows.iter().flatten().any(|value| !value.is_finite()) {
        return None;
    }
    Some(GpuPnpFocalCandidate {
        pose: SE3::from_rotation_translation(
            &[
                [rows[0][0], rows[0][1], rows[0][2]],
                [rows[1][0], rows[1][1], rows[1][2]],
                [rows[2][0], rows[2][1], rows[2][2]],
            ],
            &[rows[0][3], rows[1][3], rows[2][3]],
        ),
        focal,
    })
}

pub(crate) struct WgpuPnPFocalScorer {
    context: Arc<WgpuContext>,
    bind_group_layout: wgpu::BindGroupLayout,
    score_pipeline: wgpu::ComputePipeline,
    mask_pipeline: wgpu::ComputePipeline,
    selection_bind_group_layout: wgpu::BindGroupLayout,
    selection_pipeline: wgpu::ComputePipeline,
    acceptance_bind_group_layout: wgpu::BindGroupLayout,
    acceptance_pipeline: wgpu::ComputePipeline,
    image_buffer: Option<wgpu::Buffer>,
    object_buffer: Option<wgpu::Buffer>,
    observation_count: usize,
    threshold_px: f32,
    last_scored_models: Vec<GpuPnpFocalModel>,
    last_supports: Vec<rustslam::tracker::PnPModelSupport>,
}

impl WgpuPnPFocalScorer {
    pub(crate) fn from_context(context: Arc<WgpuContext>) -> Result<Self> {
        let device = context.device();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rustsfm gpu pnp-focal scoring bind group layout"),
            entries: &[
                storage_layout_entry(0, true),
                storage_layout_entry(1, true),
                storage_layout_entry(2, true),
                storage_layout_entry(3, false),
                storage_layout_entry(4, false),
                uniform_layout_entry(5),
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rustsfm gpu pnp-focal scoring shader"),
            source: wgpu::ShaderSource::Wgsl(PNP_FOCAL_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rustsfm gpu pnp-focal scoring pipeline layout"),
            bind_group_layouts: &[None, None, Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let score_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rustsfm gpu pnp-focal support pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("score_focal_models"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let mask_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rustsfm gpu pnp-focal mask pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("write_focal_mask"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let selection_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rustsfm gpu pnp-focal selection bind group layout"),
                entries: &[
                    storage_layout_entry(0, true),
                    storage_layout_entry(1, true),
                    storage_layout_entry(2, false),
                    storage_layout_entry(3, false),
                    uniform_layout_entry(4),
                    storage_layout_entry(5, false),
                ],
            });
        let selection_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rustsfm gpu pnp-focal selection pipeline layout"),
                bind_group_layouts: &[
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some(&selection_bind_group_layout),
                ],
                immediate_size: 0,
            });
        let selection_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rustsfm gpu pnp-focal selection pipeline"),
            layout: Some(&selection_pipeline_layout),
            module: &shader,
            entry_point: Some("select_focal_model"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let acceptance_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rustsfm gpu pnp-focal acceptance bind group layout"),
                entries: &[
                    storage_layout_entry(0, false),
                    storage_layout_entry(1, false),
                    storage_layout_entry(2, false),
                    storage_layout_entry(3, true),
                    storage_layout_entry(4, true),
                    storage_layout_entry(5, true),
                    storage_layout_entry(6, true),
                    uniform_layout_entry(7),
                ],
            });
        let acceptance_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rustsfm gpu pnp-focal acceptance pipeline layout"),
                bind_group_layouts: &[
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some(&acceptance_bind_group_layout),
                ],
                immediate_size: 0,
            });
        let acceptance_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("rustsfm gpu pnp-focal acceptance pipeline"),
                layout: Some(&acceptance_pipeline_layout),
                module: &shader,
                entry_point: Some("accept_refined_focal_candidate"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        if let Some(error) = pollster::block_on(error_scope.pop()) {
            bail!("gpu pnp-focal scoring pipeline creation failed: {error}");
        }
        Ok(Self {
            context,
            bind_group_layout,
            score_pipeline,
            mask_pipeline,
            selection_bind_group_layout,
            selection_pipeline,
            acceptance_bind_group_layout,
            acceptance_pipeline,
            image_buffer: None,
            object_buffer: None,
            observation_count: 0,
            threshold_px: 0.0,
            last_scored_models: Vec::new(),
            last_supports: Vec::new(),
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
        let image_points = centered_points
            .iter()
            .map(|&point| GpuPnpImagePoint {
                x: point[0],
                y: point[1],
                pad0: 0.0,
                pad1: 0.0,
            })
            .collect::<Vec<_>>();
        let object_points = object_points
            .iter()
            .map(|&point| GpuPnpObjectPoint {
                x: point[0],
                y: point[1],
                z: point[2],
                pad: 0.0,
            })
            .collect::<Vec<_>>();
        let storage_limit = gpu_storage_limit(&self.context);
        checked_gpu_storage_bytes::<GpuPnpImagePoint>(
            image_points.len(),
            storage_limit,
            "image observations",
        )?;
        checked_gpu_storage_bytes::<GpuPnpObjectPoint>(
            object_points.len(),
            storage_limit,
            "object observations",
        )?;
        let device = self.context.device();
        self.image_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rustsfm gpu pnp-focal image observations"),
                contents: bytemuck::cast_slice(&image_points),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        );
        self.object_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rustsfm gpu pnp-focal object observations"),
                contents: bytemuck::cast_slice(&object_points),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        );
        self.observation_count = centered_points.len();
        self.threshold_px = threshold_px;
        self.last_scored_models.clear();
        self.last_supports.clear();
        Ok(())
    }

    pub(crate) fn score(
        &mut self,
        candidate: GpuPnpFocalCandidate,
    ) -> Result<rustslam::tracker::PnPModelSupport> {
        self.score_many(&[candidate])
            .map(|mut supports| supports.remove(0))
    }

    pub(crate) fn score_many(
        &mut self,
        candidates: &[GpuPnpFocalCandidate],
    ) -> Result<Vec<rustslam::tracker::PnPModelSupport>> {
        let image_buffer = self
            .image_buffer
            .as_ref()
            .context("gpu focal pnp observations are not prepared")?;
        let object_buffer = self
            .object_buffer
            .as_ref()
            .context("gpu focal pnp observations are not prepared")?;
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let model_count =
            u32::try_from(candidates.len()).context("gpu focal pnp model count exceeds u32")?;
        if model_count
            > self
                .context
                .device()
                .limits()
                .max_compute_workgroups_per_dimension
        {
            bail!("gpu focal pnp scoring dispatch exceeds device workgroup limit");
        }
        self.last_scored_models.clear();
        self.last_supports.clear();
        let models = candidates
            .iter()
            .copied()
            .map(focal_candidate_to_model)
            .collect::<Result<Vec<_>>>()?;
        let device = self.context.device();
        let storage_limit = gpu_storage_limit(&self.context);
        checked_gpu_storage_bytes::<GpuPnpFocalModel>(models.len(), storage_limit, "models")?;
        let support_bytes = checked_gpu_storage_bytes::<GpuPnpFocalSupport>(
            candidates.len(),
            storage_limit,
            "support summaries",
        )?;
        let models_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp-focal scoring models"),
            contents: bytemuck::cast_slice(&models),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let supports = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal support summaries"),
            size: support_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mask_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal scoring mask scratch"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let params = GpuPnpFocalScoringParams {
            model_count,
            observation_count: u32::try_from(self.observation_count)?,
            selected_model: 0,
            pad1: 0,
            threshold_squared: self.threshold_px * self.threshold_px,
            pad2: 0.0,
            pad3: 0.0,
            pad4: 0.0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp-focal scoring params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustsfm gpu pnp-focal scoring bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: models_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: image_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: object_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: supports.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: mask_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rustsfm gpu pnp-focal scoring encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustsfm gpu pnp-focal scoring pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.score_pipeline);
            pass.set_bind_group(2, &bind_group, &[]);
            pass.dispatch_workgroups(model_count, 1, 1);
        }
        self.context.queue().submit(Some(encoder.finish()));
        let supports = self
            .context
            .read_buffer::<GpuPnpFocalSupport>(&supports, candidates.len())?
            .into_iter()
            .enumerate()
            .map(|(index, support)| {
                if support.inliers as usize > self.observation_count
                    || !support.residual_sum.is_finite()
                    || support.residual_sum < 0.0
                {
                    bail!("gpu focal pnp support {index} is invalid");
                }
                Ok(rustslam::tracker::PnPModelSupport {
                    inliers: support.inliers as usize,
                    residual_sum: support.residual_sum as f64,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.last_scored_models = models;
        self.last_supports = supports.clone();
        Ok(supports)
    }

    fn initialize_refinement_state(
        &self,
        models: &wgpu::Buffer,
        model_count: usize,
        min_focal: f32,
        max_focal: f32,
    ) -> Result<GpuPnpFocalRefinementState> {
        let image_buffer = self
            .image_buffer
            .as_ref()
            .context("gpu focal pnp observations are not prepared")?;
        let object_buffer = self
            .object_buffer
            .as_ref()
            .context("gpu focal pnp observations are not prepared")?;
        let model_count =
            u32::try_from(model_count).context("gpu focal pnp model count exceeds u32")?;
        if model_count
            > self
                .context
                .device()
                .limits()
                .max_compute_workgroups_per_dimension
        {
            bail!("gpu focal pnp scoring dispatch exceeds device workgroup limit");
        }
        let state = GpuPnpFocalRefinementState::new(&self.context, self.observation_count)?;
        let storage_limit = gpu_storage_limit(&self.context);
        let support_bytes = checked_gpu_storage_bytes::<GpuPnpFocalSupport>(
            usize::try_from(model_count)?,
            storage_limit,
            "support summaries",
        )?;
        let device = self.context.device();
        let supports = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal generated support summaries"),
            size: support_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let mask_scratch = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal generated mask scratch"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let scoring_params = GpuPnpFocalScoringParams {
            model_count,
            observation_count: u32::try_from(self.observation_count)?,
            selected_model: 0,
            pad1: 0,
            threshold_squared: self.threshold_px * self.threshold_px,
            pad2: 0.0,
            pad3: 0.0,
            pad4: 0.0,
        };
        let scoring_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp-focal generated scoring params"),
            contents: bytemuck::bytes_of(&scoring_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let scoring_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustsfm gpu pnp-focal generated scoring bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: models.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: image_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: object_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: supports.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: mask_scratch.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: scoring_params.as_entire_binding(),
                },
            ],
        });
        let selection_result = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal selected result"),
            size: std::mem::size_of::<GpuPnpFocalResult>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let selection_params = GpuPnpFocalSelectionParams {
            model_count,
            observation_count: u32::try_from(self.observation_count)?,
            pad0: 0,
            pad1: 0,
            min_focal,
            max_focal,
            pad2: 0.0,
            pad3: 0.0,
        };
        let selection_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp-focal selection params"),
            contents: bytemuck::bytes_of(&selection_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let selection_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustsfm gpu pnp-focal persistent selection bind group"),
            layout: &self.selection_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: models.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: supports.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: selection_result.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: state.current_model.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: selection_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: state.current_support.as_entire_binding(),
                },
            ],
        });
        let mask_params = GpuPnpFocalScoringParams {
            model_count: 1,
            observation_count: u32::try_from(self.observation_count)?,
            selected_model: 0,
            pad1: 0,
            threshold_squared: self.threshold_px * self.threshold_px,
            pad2: 0.0,
            pad3: 0.0,
            pad4: 0.0,
        };
        let mask_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp-focal initial mask params"),
            contents: bytemuck::bytes_of(&mask_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mask_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustsfm gpu pnp-focal initial mask bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state.current_model.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: image_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: object_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: state.current_support.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: state.current_mask.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: mask_params.as_entire_binding(),
                },
            ],
        });
        let mask_workgroups = checked_mask_workgroups(
            self.observation_count,
            device.limits().max_compute_workgroups_per_dimension,
            "initial",
        )?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rustsfm gpu pnp-focal persistent selection encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustsfm gpu pnp-focal generated scoring pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.score_pipeline);
            pass.set_bind_group(2, &scoring_bind_group, &[]);
            pass.dispatch_workgroups(model_count, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustsfm gpu pnp-focal persistent selection pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.selection_pipeline);
            pass.set_bind_group(5, &selection_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustsfm gpu pnp-focal initial mask pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.mask_pipeline);
            pass.set_bind_group(2, &mask_bind_group, &[]);
            pass.dispatch_workgroups(mask_workgroups, 1, 1);
        }
        self.context.queue().submit(Some(encoder.finish()));
        Ok(state)
    }

    fn encode_score_and_mask(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        model: &wgpu::Buffer,
        support: &wgpu::Buffer,
        mask: &wgpu::Buffer,
    ) -> Result<()> {
        let image_buffer = self
            .image_buffer
            .as_ref()
            .context("gpu focal pnp observations are not prepared")?;
        let object_buffer = self
            .object_buffer
            .as_ref()
            .context("gpu focal pnp observations are not prepared")?;
        let params = GpuPnpFocalScoringParams {
            model_count: 1,
            observation_count: u32::try_from(self.observation_count)?,
            selected_model: 0,
            pad1: 0,
            threshold_squared: self.threshold_px * self.threshold_px,
            pad2: 0.0,
            pad3: 0.0,
            pad4: 0.0,
        };
        let params = self
            .context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rustsfm gpu pnp-focal candidate scoring params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self
            .context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rustsfm gpu pnp-focal candidate scoring bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: model.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: image_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: object_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: support.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: mask.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustsfm gpu pnp-focal candidate scoring pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.score_pipeline);
            pass.set_bind_group(2, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        let mask_workgroups = checked_mask_workgroups(
            self.observation_count,
            self.context
                .device()
                .limits()
                .max_compute_workgroups_per_dimension,
            "candidate",
        )?;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rustsfm gpu pnp-focal candidate mask pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.mask_pipeline);
        pass.set_bind_group(2, &bind_group, &[]);
        pass.dispatch_workgroups(mask_workgroups, 1, 1);
        Ok(())
    }

    fn encode_acceptance(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        state: &GpuPnpFocalRefinementState,
        min_focal: f32,
        max_focal: f32,
    ) -> Result<()> {
        let params = GpuPnpFocalAcceptanceParams {
            observation_count: u32::try_from(self.observation_count)?,
            pad0: 0,
            min_focal,
            max_focal,
        };
        let params = self
            .context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rustsfm gpu pnp-focal acceptance params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self
            .context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rustsfm gpu pnp-focal acceptance bind group"),
                layout: &self.acceptance_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: state.current_model.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: state.current_support.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: state.current_mask.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: state.candidate_model.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: state.candidate_support.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: state.candidate_mask.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: state.candidate_status.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rustsfm gpu pnp-focal acceptance pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.acceptance_pipeline);
        pass.set_bind_group(6, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        Ok(())
    }

    fn accept_for_test(
        &self,
        current_model: GpuPnpFocalModel,
        current_support: GpuPnpFocalSupport,
        current_mask: &[u32],
        candidate_model: GpuPnpFocalModel,
        candidate_support: GpuPnpFocalSupport,
        candidate_mask: &[u32],
        candidate_status: u32,
    ) -> Result<(GpuPnpFocalModel, GpuPnpFocalSupport, Vec<u32>)> {
        if current_mask.len() != candidate_mask.len() || current_mask.len() < 4 {
            bail!("gpu focal pnp acceptance test masks are invalid");
        }
        let state = GpuPnpFocalRefinementState::new(&self.context, current_mask.len())?;
        let device = self.context.device();
        self.context.queue().write_buffer(
            &state.current_model,
            0,
            bytemuck::bytes_of(&current_model),
        );
        self.context.queue().write_buffer(
            &state.current_support,
            0,
            bytemuck::bytes_of(&current_support),
        );
        self.context.queue().write_buffer(
            &state.current_mask,
            0,
            bytemuck::cast_slice(current_mask),
        );
        self.context.queue().write_buffer(
            &state.candidate_model,
            0,
            bytemuck::bytes_of(&candidate_model),
        );
        self.context.queue().write_buffer(
            &state.candidate_support,
            0,
            bytemuck::bytes_of(&candidate_support),
        );
        self.context.queue().write_buffer(
            &state.candidate_mask,
            0,
            bytemuck::cast_slice(candidate_mask),
        );
        self.context.queue().write_buffer(
            &state.candidate_status,
            0,
            bytemuck::bytes_of(&candidate_status),
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rustsfm gpu pnp-focal acceptance test encoder"),
        });
        self.encode_acceptance(&mut encoder, &state, 1.0, 2_000.0)?;
        self.context.queue().submit(Some(encoder.finish()));
        Ok((
            self.context
                .read_buffer::<GpuPnpFocalModel>(&state.current_model, 1)?[0],
            self.context
                .read_buffer::<GpuPnpFocalSupport>(&state.current_support, 1)?[0],
            self.context
                .read_buffer::<u32>(&state.current_mask, current_mask.len())?,
        ))
    }

    fn score_generated_models_and_select(
        &mut self,
        models: &wgpu::Buffer,
        model_count: usize,
        min_focal: f32,
        max_focal: f32,
    ) -> Result<
        Option<(
            usize,
            GpuPnpFocalCandidate,
            rustslam::tracker::PnPModelSupport,
            Vec<bool>,
        )>,
    > {
        let image_buffer = self
            .image_buffer
            .as_ref()
            .context("gpu focal pnp observations are not prepared")?;
        let object_buffer = self
            .object_buffer
            .as_ref()
            .context("gpu focal pnp observations are not prepared")?;
        let model_count_u32 =
            u32::try_from(model_count).context("gpu focal pnp model count exceeds u32")?;
        if model_count_u32
            > self
                .context
                .device()
                .limits()
                .max_compute_workgroups_per_dimension
        {
            bail!("gpu focal pnp scoring dispatch exceeds device workgroup limit");
        }
        let storage_limit = gpu_storage_limit(&self.context);
        let support_bytes = checked_gpu_storage_bytes::<GpuPnpFocalSupport>(
            model_count,
            storage_limit,
            "support summaries",
        )?;
        let _mask_bytes =
            checked_gpu_storage_bytes::<u32>(self.observation_count, storage_limit, "inlier mask")?;
        let device = self.context.device();
        let supports = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal generated support summaries"),
            size: support_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let mask_scratch = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal generated mask scratch"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let scoring_params = GpuPnpFocalScoringParams {
            model_count: model_count_u32,
            observation_count: u32::try_from(self.observation_count)?,
            selected_model: 0,
            pad1: 0,
            threshold_squared: self.threshold_px * self.threshold_px,
            pad2: 0.0,
            pad3: 0.0,
            pad4: 0.0,
        };
        let scoring_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp-focal generated scoring params"),
            contents: bytemuck::bytes_of(&scoring_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let scoring_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustsfm gpu pnp-focal generated scoring bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: models.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: image_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: object_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: supports.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: mask_scratch.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: scoring_params.as_entire_binding(),
                },
            ],
        });
        let result = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal selected result"),
            size: std::mem::size_of::<GpuPnpFocalResult>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let selected_model = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal selected model"),
            size: std::mem::size_of::<GpuPnpFocalModel>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let selected_support = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal selected support"),
            size: std::mem::size_of::<GpuPnpFocalSupport>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let selection_params = GpuPnpFocalSelectionParams {
            model_count: model_count_u32,
            observation_count: u32::try_from(self.observation_count)?,
            pad0: 0,
            pad1: 0,
            min_focal,
            max_focal,
            pad2: 0.0,
            pad3: 0.0,
        };
        let selection_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp-focal selection params"),
            contents: bytemuck::bytes_of(&selection_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let selection_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustsfm gpu pnp-focal selection bind group"),
            layout: &self.selection_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: models.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: supports.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: selected_model.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: selection_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: selected_support.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rustsfm gpu pnp-focal generated selection encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustsfm gpu pnp-focal generated scoring pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.score_pipeline);
            pass.set_bind_group(2, &scoring_bind_group, &[]);
            pass.dispatch_workgroups(model_count_u32, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustsfm gpu pnp-focal selection pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.selection_pipeline);
            pass.set_bind_group(5, &selection_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.context.queue().submit(Some(encoder.finish()));
        let result = self.context.read_buffer::<GpuPnpFocalResult>(&result, 1)?[0];
        if result.valid == 0 || result.inliers < 4 || !result.residual_sum.is_finite() {
            return Ok(None);
        }
        let model = self
            .context
            .read_buffer::<GpuPnpFocalModel>(&selected_model, 1)?[0];
        let candidate = focal_model_to_candidate(model)
            .context("gpu focal pnp selected model is invalid after GPU validation")?;
        let support = rustslam::tracker::PnPModelSupport {
            inliers: result.inliers as usize,
            residual_sum: result.residual_sum as f64,
        };
        self.last_scored_models = vec![model];
        self.last_supports = vec![support];
        let mask = self.inlier_mask(candidate)?;
        if mask.iter().filter(|&&inlier| inlier).count() != support.inliers {
            bail!("gpu focal pnp selected mask does not match GPU support");
        }
        Ok(Some((
            result.selected_model as usize,
            candidate,
            support,
            mask,
        )))
    }

    pub(crate) fn inlier_mask(&self, candidate: GpuPnpFocalCandidate) -> Result<Vec<bool>> {
        let model = focal_candidate_to_model(candidate)?;
        let expected = self
            .last_scored_models
            .iter()
            .zip(&self.last_supports)
            .find_map(|(scored, support)| (*scored == model).then_some(*support))
            .context("gpu focal pnp selected model is missing from the latest scoring batch")?;
        let image_buffer = self
            .image_buffer
            .as_ref()
            .context("gpu focal pnp observations are not prepared")?;
        let object_buffer = self
            .object_buffer
            .as_ref()
            .context("gpu focal pnp observations are not prepared")?;
        let observation_count = self.observation_count;
        let storage_limit = gpu_storage_limit(&self.context);
        let workgroups = checked_mask_workgroups(
            observation_count,
            self.context
                .device()
                .limits()
                .max_compute_workgroups_per_dimension,
            "selected",
        )?;
        let device = self.context.device();
        let model_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp-focal mask model"),
            contents: bytemuck::bytes_of(&model),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let supports = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal mask support"),
            size: std::mem::size_of::<GpuPnpFocalSupport>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let mask = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm gpu pnp-focal inlier mask"),
            size: checked_gpu_storage_bytes::<u32>(
                observation_count,
                storage_limit,
                "inlier mask",
            )?,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = GpuPnpFocalScoringParams {
            model_count: 1,
            observation_count: u32::try_from(observation_count)?,
            selected_model: 0,
            pad1: 0,
            threshold_squared: self.threshold_px * self.threshold_px,
            pad2: 0.0,
            pad3: 0.0,
            pad4: 0.0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustsfm gpu pnp-focal mask params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustsfm gpu pnp-focal mask bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: image_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: object_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: supports.as_entire_binding(),
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
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rustsfm gpu pnp-focal mask encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustsfm gpu pnp-focal mask pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.mask_pipeline);
            pass.set_bind_group(2, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.context.queue().submit(Some(encoder.finish()));
        let mask = self
            .context
            .read_buffer::<u32>(&mask, observation_count)?
            .into_iter()
            .map(|value| value != 0)
            .collect::<Vec<_>>();
        let inliers = mask.iter().filter(|&&value| value).count();
        if inliers != expected.inliers {
            bail!(
                "gpu focal pnp selected model support mismatch: summary has {}, mask has {inliers}",
                expected.inliers
            );
        }
        Ok(mask)
    }
}

fn focal_candidate_to_model(candidate: GpuPnpFocalCandidate) -> Result<GpuPnpFocalModel> {
    if !candidate.focal.is_finite() || candidate.focal <= 0.0 {
        bail!("gpu focal pnp candidate has invalid focal length");
    }
    let matrix = candidate.pose.to_matrix();
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        bail!("gpu focal pnp candidate pose contains non-finite values");
    }
    Ok(GpuPnpFocalModel {
        row0: matrix[0],
        row1: matrix[1],
        row2: matrix[2],
        log_focal_and_padding: [candidate.focal.ln(), 0.0, 0.0, 0.0],
    })
}

#[cfg(all(test, feature = "gpu-wgpu"))]
mod tests {
    use super::*;

    fn dot3(left: [f32; 3], right: [f32; 3]) -> f32 {
        left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
    }

    fn normalized3(value: [f32; 3]) -> [f32; 3] {
        let scale = dot3(value, value).sqrt();
        [value[0] / scale, value[1] / scale, value[2] / scale]
    }

    fn squared_distance3(left: [f32; 3], right: [f32; 3]) -> f32 {
        let delta = [left[0] - right[0], left[1] - right[1], left[2] - right[2]];
        dot3(delta, delta)
    }

    fn p3p_single_real_root(c2: f32, c1: f32, c0: f32) -> f32 {
        let a = c1 - c2 * c2 / 3.0;
        let b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
        let discriminant = b * b / 4.0 + a * a * a / 27.0;
        if discriminant > 0.0 {
            let root_discriminant = discriminant.sqrt();
            return (-0.5 * b + root_discriminant).cbrt() + (-0.5 * b - root_discriminant).cbrt()
                - c2 / 3.0;
        }
        if a >= -1.0e-6 {
            return -c2 / 3.0;
        }
        2.0 * (-a / 3.0).sqrt()
            * (3.0 * b / (2.0 * a) * (-3.0 / a).sqrt())
                .clamp(-1.0, 1.0)
                .acos()
            / 3.0
            - c2 / 3.0
    }

    fn p3p_compute_pq_column_for_identity_pose(world: &[[f32; 3]; 4]) -> u32 {
        let a01 = squared_distance3(world[0], world[1]);
        let a02 = squared_distance3(world[0], world[2]);
        let a12 = squared_distance3(world[1], world[2]);
        let (first, second, third) = if a02 >= a01 && a02 >= a12 {
            (1, 0, 2)
        } else if a01 >= a02 && a01 >= a12 {
            (2, 0, 1)
        } else {
            (0, 1, 2)
        };
        let a01 = squared_distance3(world[first], world[second]);
        let a02 = squared_distance3(world[first], world[third]);
        let a12 = squared_distance3(world[second], world[third]);
        let ray0 = normalized3(world[first]);
        let ray1 = normalized3(world[second]);
        let ray2 = normalized3(world[third]);
        let m01 = dot3(ray0, ray1);
        let m02 = dot3(ray0, ray2);
        let m12 = dot3(ray1, ray2);
        let a = a01 / a12;
        let b = a02 / a12;
        let m12_squared = 1.0 - m12 * m12;
        let m02_squared = m02 * m02 - 1.0;
        let m01_squared = m01 * m01 - 1.0;
        let denominator = b * b * m12_squared + b * m02_squared;
        let mixed = -2.0 + 2.0 * m01 * m02 * m12;
        let c2 =
            ((a - 1.0) * m02_squared + 2.0 * a * b * m12_squared + b * b * m12_squared + b * mixed)
                / denominator;
        let c1 =
            (a * a * m12_squared + 2.0 * a * b * m12_squared + a * mixed + (b - 1.0) * m01_squared)
                / denominator;
        let c0 = (a * a * m12_squared + a * m01_squared) / denominator;
        let root = p3p_single_real_root(c2, c1, c0);
        let c = [
            [-a + root * (1.0 - b), -m02 * root, a * m12 + b * m12 * root],
            [-m02 * root, root + 1.0, -m01],
            [a * m12 + b * m12 * root, -m01, -a - b * root + 1.0],
        ];
        let adj0 = [
            c[1][2] * c[2][1] - c[1][1] * c[2][2],
            c[0][1] * c[2][2] - c[0][2] * c[2][1],
            c[0][1] * c[1][2] - c[0][2] * c[1][1],
        ];
        let adj1 = [
            adj0[1],
            c[0][2] * c[2][0] - c[0][0] * c[2][2],
            c[0][0] * c[1][2] - c[0][2] * c[1][0],
        ];
        let adj2 = [adj0[2], adj1[2], c[0][1] * c[1][0] - c[0][0] * c[1][1]];
        let diagonal = [adj0[0], adj1[1], adj2[2]];
        if diagonal[1] > diagonal[0] && diagonal[1] >= diagonal[2] {
            1
        } else if diagonal[2] > diagonal[0] && diagonal[2] > diagonal[1] {
            2
        } else {
            0
        }
    }

    fn candidate_reprojects_sample(
        model: GpuPnpFocalModel,
        image: &[[f32; 2]],
        world: &[[f32; 3]],
    ) -> bool {
        focal_model_to_candidate(model).is_some_and(|candidate| {
            let matrix = candidate.pose.to_matrix();
            world.iter().zip(image).all(|(world, image)| {
                let depth = matrix[2][0] * world[0]
                    + matrix[2][1] * world[1]
                    + matrix[2][2] * world[2]
                    + matrix[2][3];
                depth > 0.0
                    && (candidate.focal
                        * (matrix[0][0] * world[0]
                            + matrix[0][1] * world[1]
                            + matrix[0][2] * world[2]
                            + matrix[0][3])
                        / depth
                        - image[0])
                        .abs()
                        < 1.0e-2
                    && (candidate.focal
                        * (matrix[1][0] * world[0]
                            + matrix[1][1] * world[1]
                            + matrix[1][2] * world[2]
                            + matrix[1][3])
                        / depth
                        - image[1])
                        .abs()
                        < 1.0e-2
            })
        })
    }

    #[test]
    fn wgpu_pnp_focal_refinement_rejects_singular_normal_equations() -> Result<()> {
        let Some(context) = WgpuContext::try_new_optional()? else {
            eprintln!("skipping GPU PnP-focal singular-refinement test: no compatible adapter");
            return Ok(());
        };
        let refiner = WgpuPnPFocalRefiner::from_context(context)?;
        let candidate = GpuPnpFocalCandidate {
            pose: SE3::identity(),
            focal: 700.0,
        };
        let image = [[0.0, 0.0]; 4];
        let world = [[0.0, 0.0, 3.0]; 4];

        assert!(refiner
            .refine_once(&image, &world, candidate, &[true; 4])?
            .is_none());
        Ok(())
    }

    #[test]
    fn wgpu_pnp_focal_refinement_rejects_masked_behind_camera_observation() -> Result<()> {
        let Some(context) = WgpuContext::try_new_optional()? else {
            eprintln!(
                "skipping GPU PnP-focal behind-camera refinement test: no compatible adapter"
            );
            return Ok(());
        };
        let refiner = WgpuPnPFocalRefiner::from_context(context)?;
        let candidate = GpuPnpFocalCandidate {
            pose: SE3::identity(),
            focal: 700.0,
        };
        let world = [
            [-0.9, -0.4, 2.1],
            [0.7, -0.2, 2.7],
            [-0.3, 0.8, 3.4],
            [0.5, 0.6, 4.1],
            [-0.7, 0.3, 4.8],
            [0.2, -0.8, 3.0],
            [0.9, 0.4, 3.7],
            [0.1, -0.1, -2.0],
        ];
        let image = world
            .iter()
            .map(|point| {
                if point[2] > 0.0 {
                    [700.0 * point[0] / point[2], 700.0 * point[1] / point[2]]
                } else {
                    [0.0, 0.0]
                }
            })
            .collect::<Vec<_>>();

        assert!(refiner
            .refine_once(&image, &world, candidate, &[true; 8])?
            .is_none());
        Ok(())
    }

    #[test]
    fn gpu_pnp_focal_storage_bytes_rejects_overflow_and_device_limit() {
        assert_eq!(checked_gpu_storage_bytes::<u32>(4, 32, "mask").unwrap(), 16);
        assert!(checked_gpu_storage_bytes::<u32>(usize::MAX, u64::MAX, "mask").is_err());
        assert!(checked_gpu_storage_bytes::<u32>(9, 32, "mask").is_err());
    }

    #[test]
    fn gpu_pnp_focal_mask_workgroups_reject_device_limit() {
        assert_eq!(checked_mask_workgroups(128, 2, "candidate").unwrap(), 2);
        assert!(checked_mask_workgroups(129, 2, "candidate").is_err());
    }

    #[test]
    fn gpu_pnp_focal_p3p_capacity_rejects_oversized_trial_count_before_parameters() {
        assert!(validate_p3p_candidate_capacity(1_000, 64, 4, u64::MAX, 1_024).is_err());
    }

    #[test]
    fn gpu_pnp_focal_p3p_fixture_selects_compute_pq_column_zero() {
        let world = [
            [0.0, 0.0, 2.0],
            [0.054_054_86, -0.915_494_2, 4.964_504_7],
            [-0.108_016_22, -1.015_053, 2.909_355],
            [0.7, 0.3, 3.4],
        ];
        assert_eq!(p3p_compute_pq_column_for_identity_pose(&world), 0);
    }

    #[test]
    fn wgpu_pnp_focal_p3p_column_zero_candidate_reprojects_sample() -> Result<()> {
        let focal = 700.0f32;
        let world = [
            [0.0, 0.0, 2.0],
            [0.054_054_86, -0.915_494_2, 4.964_504_7],
            [-0.108_016_22, -1.015_053, 2.909_355],
            [0.7, 0.3, 3.4],
        ];
        assert_eq!(p3p_compute_pq_column_for_identity_pose(&world), 0);
        let image = world
            .iter()
            .map(|point| [focal * point[0] / point[2], focal * point[1] / point[2]])
            .collect::<Vec<_>>();
        let Some(context) = WgpuContext::try_new_optional()? else {
            eprintln!("skipping GPU PnP-focal column-zero test: no compatible adapter");
            return Ok(());
        };
        let generator = WgpuPnPFocalCandidateGenerator::from_context(context)?;
        let candidates = generator.generate_p3p(&image, &world, [0, 1, 2, 3], focal, 0)?;

        assert!(candidates
            .iter()
            .any(|model| candidate_reprojects_sample(*model, &image, &world)));
        Ok(())
    }

    #[test]
    fn wgpu_pnp_focal_gpu_selection_breaks_exact_ties_by_index() -> Result<()> {
        let Some(context) = WgpuContext::try_new_optional()? else {
            eprintln!("skipping GPU PnP-focal selection test: no compatible adapter");
            return Ok(());
        };
        let candidate = GpuPnpFocalCandidate {
            pose: SE3::identity(),
            focal: 700.0,
        };
        let model = focal_candidate_to_model(candidate)?;
        let models = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rustsfm GPU PnP-focal selection tie models"),
                contents: bytemuck::cast_slice(&[model; 4]),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let world = [
            [-0.3, -0.2, 2.5],
            [0.4, -0.1, 3.0],
            [-0.2, 0.5, 3.5],
            [0.3, 0.4, 4.0],
        ];
        let image = world
            .iter()
            .map(|point| [700.0 * point[0] / point[2], 700.0 * point[1] / point[2]])
            .collect::<Vec<_>>();
        let mut scorer = WgpuPnPFocalScorer::from_context(context)?;
        scorer.prepare(&image, &world, 0.1)?;

        let selected = scorer
            .score_generated_models_and_select(&models, 4, 100.0, 1_400.0)?
            .expect("all tied identity models are valid");
        assert_eq!(selected.0, 0);
        assert_eq!(selected.2.inliers, 4);
        Ok(())
    }

    #[test]
    fn wgpu_pnp_focal_gpu_acceptance_retains_current_on_tie() -> Result<()> {
        let Some(context) = WgpuContext::try_new_optional()? else {
            eprintln!("skipping GPU PnP-focal acceptance test: no compatible adapter");
            return Ok(());
        };
        let scorer = WgpuPnPFocalScorer::from_context(context.clone())?;
        let current = GpuPnpFocalCandidate {
            pose: SE3::identity(),
            focal: 700.0,
        };
        let candidate = GpuPnpFocalCandidate {
            pose: SE3::identity(),
            focal: 900.0,
        };
        let current_support = GpuPnpFocalSupport {
            inliers: 4,
            pad0: 4,
            residual_sum: 1.0,
            pad1: 0.0,
        };
        let candidate_support = GpuPnpFocalSupport {
            inliers: 4,
            pad0: 0,
            residual_sum: 1.0,
            pad1: 0.0,
        };
        let retained = scorer.accept_for_test(
            focal_candidate_to_model(current)?,
            current_support,
            &[1, 0, 1, 0],
            focal_candidate_to_model(candidate)?,
            candidate_support,
            &[0, 1, 0, 1],
            1,
        )?;

        assert_eq!(retained.0, focal_candidate_to_model(current)?);
        assert_eq!(retained.1.inliers, 4);
        assert_eq!(retained.1.pad0, 4);
        assert_eq!(retained.2, vec![1, 0, 1, 0]);
        Ok(())
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn wgpu_pnp_focal_p3p_reorders_adverse_baseline_before_solving() -> Result<()> {
        let Some(context) = WgpuContext::try_new_optional()? else {
            eprintln!("skipping GPU PnP-focal P3P ordering test: no compatible adapter");
            return Ok(());
        };
        let focal = 700.0f32;
        let world = [
            [0.0, 0.0, 3.0],
            [0.0002, 0.0, 3.0],
            [1.2, 0.4, 3.5],
            [-0.8, 0.7, 4.2],
        ];
        let image = world
            .iter()
            .map(|point| [focal * point[0] / point[2], focal * point[1] / point[2]])
            .collect::<Vec<_>>();
        let generator = WgpuPnPFocalCandidateGenerator::from_context(context)?;
        let candidates = generator.generate_p3p(&image, &world, [0, 1, 2, 3], focal, 0)?;

        assert!(candidates.iter().any(|model| {
            focal_model_to_candidate(*model).is_some_and(|candidate| {
                let matrix = candidate.pose.to_matrix();
                world.iter().zip(&image).all(|(world, image)| {
                    let depth = matrix[2][0] * world[0]
                        + matrix[2][1] * world[1]
                        + matrix[2][2] * world[2]
                        + matrix[2][3];
                    depth > 0.0
                        && (candidate.focal
                            * (matrix[0][0] * world[0]
                                + matrix[0][1] * world[1]
                                + matrix[0][2] * world[2]
                                + matrix[0][3])
                            / depth
                            - image[0])
                            .abs()
                            < 1.0e-2
                        && (candidate.focal
                            * (matrix[1][0] * world[0]
                                + matrix[1][1] * world[1]
                                + matrix[1][2] * world[2]
                                + matrix[1][3])
                            / depth
                            - image[1])
                            .abs()
                            < 1.0e-2
                })
            })
        }));
        Ok(())
    }
}
