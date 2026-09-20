//! Independent f32 experiment; no production solver integration or CPU fallback.
use super::WgpuContext;
use anyhow::{ensure, Context, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("shaders/five_point_f32.wgsl");
const ALGEBRA: &str = concat!(
    include_str!("shaders/five_point_generated.wgsl"),
    "\n",
    include_str!("shaders/five_point_algebra.wgsl")
);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Ray {
    xyz: [f32; 3],
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FivePointParams {
    count: u32,
    _pad: [u32; 3],
}

/// Row-major 5×9 epipolar constraints (each essential matrix is flattened row-major).
pub type FivePointConstraintMatrix = [f32; 45];
/// Column-major 9×4 basis: four consecutive nine-element right-nullspace vectors.
pub type FivePointNullspaceBasis = [f32; 36];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FivePointStatus {
    Success,
    NotConverged,
    RankDeficient,
    SingularElimination,
    NonFinite,
    InvalidBasis,
}

impl FivePointStatus {
    fn decode(value: f32) -> Result<Self> {
        Ok(match value {
            0.0 => Self::Success,
            1.0 => Self::NotConverged,
            2.0 => Self::RankDeficient,
            3.0 => Self::SingularElimination,
            4.0 => Self::NonFinite,
            5.0 => Self::InvalidBasis,
            _ => anyhow::bail!("invalid GPU five-point status {value}"),
        })
    }
}

#[derive(Debug, Clone)]
pub struct FivePointNullspaceResult {
    /// None on failure: a zero-filled failed record is never a successful basis.
    pub basis: Option<FivePointNullspaceBasis>,
    pub status: FivePointStatus,
    pub sweeps: u32,
    /// σ₅/σ₁ from Jacobi on the 5×5 AAᵀ (singular values), not from AᵀA.
    pub min_diagonal_ratio: f32,
    /// 5 on success; on RankDeficient, #{ σ_i > 1e-5 · σ₁ } over the five AAᵀ values.
    pub rank: u32,
}

#[derive(Debug, Clone)]
pub struct FivePointAlgebraResult {
    /// Column-major 10×20, exactly the generated CPU indexing.
    pub elimination: [f32; 200],
    /// Column-major 10×10 solution of left * solved = right (not negated).
    pub solved: [f32; 100],
    /// Column-major 13×3 determinant matrix.
    pub determinant: [f32; 39],
    /// Unnormalized coefficients, descending powers z^10 through z^0.
    pub coefficients: [f32; 11],
    /// Only Success makes all arrays valid; elimination remains diagnostic on failure.
    pub status: FivePointStatus,
    pub minimum_relative_pivot: f32,
}

struct Kernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

#[path = "five_point_f32_complete.rs"]
mod complete;
pub use complete::{
    FivePointModelSlot, FivePointProfile, FivePointSlotStatus, FivePointTrialResult,
    FIVE_POINT_PASS_NAMES,
};

#[path = "five_point_f32_session.rs"]
mod session;
pub use session::FivePointSession;

/// Independent GPU-only batched five-point solver and diagnostic stages.
pub struct WgpuFivePointF32 {
    context: Arc<WgpuContext>,
    constraints: Kernel,
    basis: Kernel,
    elimination: Kernel,
    algebra: Kernel,
    polynomial: Kernel,
    validate_polynomial: Kernel,
    complete: complete::CompleteKernels,
}

impl WgpuFivePointF32 {
    pub fn try_new() -> Result<Self> {
        Self::from_context(WgpuContext::try_new()?)
    }

    pub fn from_context(context: Arc<WgpuContext>) -> Result<Self> {
        let geometry_module = shader_module(context.device(), SHADER)?;
        let algebra_module = shader_module(context.device(), ALGEBRA)?;
        let constraints = kernel(context.device(), &geometry_module, "main", 2, false)?;
        let basis = kernel(context.device(), &geometry_module, "nullspace", 1, true)?;
        let elimination = kernel(context.device(), &algebra_module, "elimination", 1, false)?;
        let algebra = kernel(context.device(), &algebra_module, "algebra", 1, true)?;
        let polynomial = kernel(context.device(), &algebra_module, "polynomial", 1, false)?;
        let validate_polynomial = kernel(
            context.device(),
            &algebra_module,
            "validate_polynomial",
            1,
            false,
        )?;
        let complete = complete::CompleteKernels::new(context.device())?;
        Ok(Self {
            complete,
            context,
            constraints,
            basis,
            elimination,
            algebra,
            polynomial,
            validate_polynomial,
        })
    }

    /// Rejects nonfinite inputs; dispatches QR with explicit rank status.
    pub fn compute_nullspace_diagnostics(
        &self,
        matrices: &[FivePointConstraintMatrix],
    ) -> Result<Vec<FivePointNullspaceResult>> {
        ensure!(
            matrices.iter().flatten().all(|x| x.is_finite()),
            "nonfinite constraint input"
        );
        let values = self.dispatch(
            &[(&self.basis, 1)],
            &[bytemuck::cast_slice(matrices)],
            matrices.len(),
            40,
        )?;
        values
            .chunks_exact(40)
            .map(|v| {
                let status = FivePointStatus::decode(v[36])?;
                let basis = if status == FivePointStatus::Success {
                    Some(v[..36].try_into().unwrap())
                } else {
                    None
                };
                Ok(FivePointNullspaceResult {
                    basis,
                    status,
                    sweeps: v[37] as u32,
                    min_diagonal_ratio: v[38],
                    rank: v[39] as u32,
                })
            })
            .collect()
    }

    /// Convenience API: fails the batch if any sample lacks a valid four-vector basis.
    pub fn compute_nullspace_basis(
        &self,
        matrices: &[FivePointConstraintMatrix],
    ) -> Result<Vec<FivePointNullspaceBasis>> {
        self.compute_nullspace_diagnostics(matrices)?
            .into_iter()
            .enumerate()
            .map(|(i, result)| {
                result
                    .basis
                    .with_context(|| format!("five-point sample {i}: {:?}", result.status))
            })
            .collect()
    }

    /// Uses the supplied basis unchanged, permitting same-basis CPU f64 comparisons.
    /// Singular/unstable samples return a failure status, never a CPU solution.
    pub fn compute_algebra(
        &self,
        bases: &[FivePointNullspaceBasis],
    ) -> Result<Vec<FivePointAlgebraResult>> {
        ensure!(
            bases.iter().flatten().all(|x| x.is_finite()),
            "nonfinite basis input"
        );
        let values = self.dispatch(
            &[
                (&self.elimination, 200u32.div_ceil(32)),
                (&self.algebra, 1),
                (&self.polynomial, 11),
                (&self.validate_polynomial, 1),
            ],
            &[bytemuck::cast_slice(bases)],
            bases.len(),
            352,
        )?;
        values
            .chunks_exact(352)
            .map(|v| {
                Ok(FivePointAlgebraResult {
                    elimination: v[..200].try_into().unwrap(),
                    solved: v[200..300].try_into().unwrap(),
                    determinant: v[300..339].try_into().unwrap(),
                    coefficients: v[339..350].try_into().unwrap(),
                    status: FivePointStatus::decode(v[350])?,
                    minimum_relative_pivot: v[351],
                })
            })
            .collect()
    }

    pub fn compute_constraint_matrices(
        &self,
        rays1: &[[f32; 3]],
        rays2: &[[f32; 3]],
    ) -> Result<Vec<FivePointConstraintMatrix>> {
        ensure!(
            rays1.len() == rays2.len(),
            "ray batches have different lengths"
        );
        ensure!(rays1.len() % 5 == 0, "ray count must be divisible by five");
        ensure!(
            rays1.iter().chain(rays2).flatten().all(|x| x.is_finite()),
            "nonfinite ray input"
        );
        let a: Vec<Ray> = rays1.iter().map(|&xyz| Ray { xyz, _pad: 0.0 }).collect();
        let b: Vec<Ray> = rays2.iter().map(|&xyz| Ray { xyz, _pad: 0.0 }).collect();
        let values = self.dispatch(
            &[(&self.constraints, 1)],
            &[bytemuck::cast_slice(&a), bytemuck::cast_slice(&b)],
            rays1.len() / 5,
            45,
        )?;
        ensure!(
            values.iter().all(|x| x.is_finite()),
            "constraint construction overflow"
        );
        Ok(values
            .chunks_exact(45)
            .map(|v| v.try_into().unwrap())
            .collect())
    }

    fn dispatch(
        &self,
        kernels: &[(&Kernel, u32)],
        inputs: &[&[u8]],
        batches: usize,
        stride: usize,
    ) -> Result<Vec<f32>> {
        if batches == 0 {
            return Ok(Vec::new());
        }
        let device = self.context.device();
        let limits = device.limits();
        ensure!(
            batches <= limits.max_compute_workgroups_per_dimension as usize,
            "five-point batch exceeds dispatch limit"
        );
        let count = batches
            .checked_mul(stride)
            .context("five-point output size overflow")?;
        let bytes = count
            .checked_mul(4)
            .context("five-point output bytes overflow")?;
        for size in inputs.iter().map(|x| x.len()).chain(std::iter::once(bytes)) {
            ensure!(
                size as u64
                    <= u64::from(limits.max_storage_buffer_binding_size)
                        .min(limits.max_buffer_size),
                "five-point storage limit exceeded"
            );
        }
        let buffers: Vec<_> = inputs
            .iter()
            .map(|data| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("five-point input"),
                    contents: data,
                    usage: wgpu::BufferUsages::STORAGE,
                })
            })
            .collect();
        let out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("five-point readback output"),
            size: bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("five-point params"),
            contents: bytemuck::bytes_of(&FivePointParams {
                count: batches as u32,
                _pad: [0; 3],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("five-point encoder"),
        });
        // Separate passes give storage dependencies between expansion, solve and polynomial.
        for &(kernel, rows) in kernels {
            let needs_params =
                std::ptr::eq(kernel, &self.basis) || std::ptr::eq(kernel, &self.algebra);
            let mut entries: Vec<_> = buffers
                .iter()
                .chain(std::iter::once(&out))
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect();
            if needs_params {
                entries.push(wgpu::BindGroupEntry {
                    binding: entries.len() as u32,
                    resource: params.as_entire_binding(),
                });
            }
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("five-point bind group"),
                layout: &kernel.layout,
                entries: &entries,
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("five-point pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernel.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let groups = if std::ptr::eq(kernel, &self.basis) || std::ptr::eq(kernel, &self.algebra)
            {
                batches.div_ceil(32)
            } else {
                batches
            };
            pass.dispatch_workgroups(groups as u32, rows, 1);
        }
        self.context
            .wait_for(self.context.queue().submit(Some(encoder.finish())))?;
        self.context
            .read_buffer(&out, count)
            .context("five-point GPU readback")
    }
}

fn shader_module(device: &wgpu::Device, source: &str) -> Result<wgpu::ShaderModule> {
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("five-point WGSL"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    if let Some(error) = pollster::block_on(validation.pop()) {
        return Err(anyhow::Error::new(error).context("five-point WGSL validation"));
    }
    Ok(module)
}

fn kernel(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    entry: &str,
    inputs: u32,
    with_params: bool,
) -> Result<Kernel> {
    #[cfg(test)]
    let started = std::time::Instant::now();
    #[cfg(test)]
    eprintln!("compiling five-point pipeline {entry}");
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let mut entries: Vec<_> = (0..=inputs)
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: binding < inputs,
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    if with_params {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: inputs + 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(entry),
        entries: &entries,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(entry),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            }),
        ),
        module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    });
    let errors = [
        pollster::block_on(internal.pop()),
        pollster::block_on(memory.pop()),
        pollster::block_on(validation.pop()),
    ];
    if let Some(error) = errors.into_iter().flatten().next() {
        return Err(
            anyhow::Error::new(error).context(format!("five-point shader/pipeline {entry}"))
        );
    }
    #[cfg(test)]
    eprintln!(
        "compiled five-point pipeline {entry} in {:?}",
        started.elapsed()
    );
    Ok(Kernel { pipeline, layout })
}

#[cfg(test)]
#[path = "five_point_f32_tests.rs"]
mod tests;
