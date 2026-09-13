use super::*;

const STRIDE: usize = 176;
const PACK: &str = "
@group(0) @binding(0) var<storage,read> diagnostics:array<f32>;
@group(0) @binding(1) var<storage,read_write> bases:array<f32>;
@compute @workgroup_size(1)
fn pack(@builtin(workgroup_id) id:vec3<u32>) {
  if(diagnostics[id.x*40u+36u]!=0.0) { return; }
  for(var i=0u;i<36u;i++) { bases[id.x*36u+i]=diagnostics[id.x*40u+i]; }
}";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FivePointSlotStatus {
    Unused,
    Accepted,
    Complex,
    NotConverged,
    DuplicateRoot,
    NullVectorFailure,
    InvalidEssential,
    DuplicateModel,
    RealRoot,
}

#[derive(Debug, Clone)]
pub struct FivePointModelSlot {
    pub slot: usize,
    pub status: FivePointSlotStatus,
    pub root: [f32; 2],
    pub root_backward_error: f32,
    /// Row-major unit Frobenius norm; present only for Accepted slots.
    pub essential: Option<[f32; 9]>,
    pub null_residual: f32,
    pub constraint_residual: f32,
    pub essential_residual: f32,
}

#[derive(Debug, Clone)]
pub struct FivePointTrialResult {
    /// Input trial index within this call, never compacted after a failure.
    pub trial: usize,
    pub upstream_status: FivePointStatus,
    pub algebra_status: FivePointStatus,
    pub polynomial_failed: bool,
    pub model_count: usize,
    pub degree: usize,
    pub root_iterations: usize,
    pub unconverged: usize,
    pub complex_filtered: usize,
    pub duplicate_roots: usize,
    pub recovery_rejected: usize,
    pub duplicate_models: usize,
    pub dropped_leading: usize,
    pub rank: usize,
    pub jacobi_sweeps: usize,
    pub slots: [FivePointModelSlot; 10],
}

pub(super) struct CompleteKernels {
    pack: Kernel,
    roots: Kernel,
    recover: Kernel,
}

impl CompleteKernels {
    pub(super) fn new(device: &wgpu::Device) -> Result<Self> {
        let packing = shader_module(device, PACK)?;
        let recovery = shader_module(device, include_str!("shaders/five_point_recovery.wgsl"))?;
        Ok(Self {
            pack: kernel(device, &packing, "pack", 1)?,
            roots: kernel(device, &recovery, "roots", 4)?,
            recover: kernel(device, &recovery, "recover", 4)?,
        })
    }
}

impl WgpuFivePointF32 {
    /// Complete GPU f32 minimal solver. Uploads rays, executes all stages without
    /// intermediate host readback, and reads fixed ten-slot records per trial.
    /// Host work is validation, buffer management and decoding only, never solving.
    pub fn solve_essential(
        &self,
        rays1: &[[f32; 3]],
        rays2: &[[f32; 3]],
    ) -> Result<Vec<FivePointTrialResult>> {
        ensure!(
            rays1.len() == rays2.len() && rays1.len() % 5 == 0,
            "equal five-ray batches required"
        );
        ensure!(
            rays1
                .iter()
                .chain(rays2)
                .flatten()
                .all(|x| x.is_finite() && x.abs() <= 1e8),
            "finite bounded rays required (abs <= 1e8)"
        );
        let count = rays1.len() / 5;
        if count == 0 {
            return Ok(Vec::new());
        }
        self.complete_size(count, STRIDE)?;
        let device = self.context.device();
        let ray_bytes = count
            .checked_mul(5)
            .and_then(|n| n.checked_mul(std::mem::size_of::<Ray>()))
            .context("five-point ray size overflow")?;
        self.complete_size_bytes(ray_bytes)?;
        let upload = |rays: &[[f32; 3]]| {
            let padded: Vec<Ray> = rays.iter().map(|&xyz| Ray { xyz, _pad: 0.0 }).collect();
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("five-point full rays"),
                contents: bytemuck::cast_slice(&padded),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let left = upload(rays1);
        let right = upload(rays2);
        let constraints = self.complete_buffer(count, 45)?;
        let diagnostics = self.complete_buffer(count, 40)?;
        let basis = self.complete_buffer(count, 36)?;
        let algebra = self.complete_buffer(count, 352)?;
        let out = self.complete_buffer(count, STRIDE)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("complete GPU five-point"),
        });
        self.complete_pass(
            &mut encoder,
            &self.constraints,
            &[&left, &right, &constraints],
            count,
            1,
        );
        self.complete_pass(
            &mut encoder,
            &self.basis,
            &[&constraints, &diagnostics],
            count,
            1,
        );
        self.complete_pass(
            &mut encoder,
            &self.complete.pack,
            &[&diagnostics, &basis],
            count,
            1,
        );
        for (kernel, rows) in [
            (&self.elimination, 200),
            (&self.algebra, 1),
            (&self.polynomial, 11),
            (&self.validate_polynomial, 1),
        ] {
            self.complete_pass(&mut encoder, kernel, &[&basis, &algebra], count, rows);
        }
        let buffers = [&constraints, &diagnostics, &algebra, &basis, &out];
        self.complete_pass(&mut encoder, &self.complete.roots, &buffers, count, 1);
        self.complete_pass(&mut encoder, &self.complete.recover, &buffers, count, 1);
        self.context
            .wait_for(self.context.queue().submit(Some(encoder.finish())))?;
        decode(&self.context.read_buffer::<f32>(&out, count * STRIDE)?)
    }

    /// Diagnostic GPU root-only API (descending coefficients); no CPU root solver.
    /// RealRoot slots have passed realness/polish/deduplication, not E recovery.
    pub fn compute_polynomial_roots(
        &self,
        coefficients: &[[f32; 11]],
    ) -> Result<Vec<FivePointTrialResult>> {
        if coefficients.is_empty() {
            return Ok(Vec::new());
        }
        let count = coefficients.len();
        self.complete_size(count, 352)?;
        ensure!(
            coefficients.iter().flatten().all(|x| x.is_finite()),
            "nonfinite polynomial input"
        );
        let mut records = vec![0.0f32; count * 352];
        for (record, c) in records.chunks_exact_mut(352).zip(coefficients) {
            record[339..350].copy_from_slice(c);
        }
        let algebra = self
            .context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("five-point diagnostic polynomials"),
                contents: bytemuck::cast_slice(&records),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let dummy = self.complete_buffer(count, 45)?;
        let out = self.complete_buffer(count, STRIDE)?;
        let mut encoder = self
            .context
            .device()
            .create_command_encoder(&Default::default());
        self.complete_pass(
            &mut encoder,
            &self.complete.roots,
            &[&dummy, &dummy, &algebra, &dummy, &out],
            count,
            1,
        );
        self.context
            .wait_for(self.context.queue().submit(Some(encoder.finish())))?;
        decode(&self.context.read_buffer::<f32>(&out, count * STRIDE)?)
    }

    fn complete_size(&self, count: usize, stride: usize) -> Result<u64> {
        ensure!(
            count
                <= self
                    .context
                    .device()
                    .limits()
                    .max_compute_workgroups_per_dimension as usize,
            "five-point dispatch limit"
        );
        let bytes = count
            .checked_mul(stride)
            .and_then(|n| n.checked_mul(4))
            .context("five-point size overflow")?;
        self.complete_size_bytes(bytes)
    }
    fn complete_size_bytes(&self, bytes: usize) -> Result<u64> {
        let limits = self.context.device().limits();
        let bytes = u64::try_from(bytes).context("five-point size does not fit u64")?;
        ensure!(
            bytes <= u64::from(limits.max_storage_buffer_binding_size),
            "five-point storage binding limit"
        );
        ensure!(
            bytes <= limits.max_buffer_size,
            "five-point device buffer limit"
        );
        Ok(bytes)
    }
    fn complete_buffer(&self, count: usize, stride: usize) -> Result<wgpu::Buffer> {
        let size = self.complete_size(count, stride)?;
        let zeroes = vec![0u8; usize::try_from(size).context("five-point zero buffer too large")?];
        Ok(self
            .context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("five-point zero-initialized intermediate"),
                contents: &zeroes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            }))
    }
    fn complete_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        kernel: &Kernel,
        buffers: &[&wgpu::Buffer],
        count: usize,
        rows: u32,
    ) {
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let group = self
            .context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("five-point full stage"),
                layout: &kernel.layout,
                entries: &entries,
            });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("five-point full stage"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(count as u32, rows, 1);
    }
}

fn decode(values: &[f32]) -> Result<Vec<FivePointTrialResult>> {
    values
        .chunks_exact(STRIDE)
        .enumerate()
        .map(|(trial, v)| {
            let slots: Result<Vec<_>> = (0..10)
                .map(|slot| {
                    let s = &v[16 + slot * 16..32 + slot * 16];
                    let status = match s[0] {
                        0.0 => FivePointSlotStatus::Unused,
                        1.0 => FivePointSlotStatus::Accepted,
                        2.0 => FivePointSlotStatus::Complex,
                        3.0 => FivePointSlotStatus::NotConverged,
                        4.0 => FivePointSlotStatus::DuplicateRoot,
                        5.0 => FivePointSlotStatus::NullVectorFailure,
                        6.0 => FivePointSlotStatus::InvalidEssential,
                        9.0 => FivePointSlotStatus::DuplicateModel,
                        10.0 => FivePointSlotStatus::RealRoot,
                        x => anyhow::bail!("invalid GPU slot status {x}"),
                    };
                    Ok(FivePointModelSlot {
                        slot,
                        status,
                        root: [s[1], s[2]],
                        root_backward_error: s[3],
                        essential: if status == FivePointSlotStatus::Accepted {
                            Some(s[4..13].try_into().unwrap())
                        } else {
                            None
                        },
                        null_residual: s[13],
                        constraint_residual: s[14],
                        essential_residual: s[15],
                    })
                })
                .collect();
            Ok(FivePointTrialResult {
                trial,
                upstream_status: FivePointStatus::decode(v[0])?,
                algebra_status: FivePointStatus::decode(v[13])?,
                polynomial_failed: v[15] != 0.0,
                model_count: v[1] as usize,
                degree: v[2] as usize,
                root_iterations: v[3] as usize,
                unconverged: v[4] as usize,
                complex_filtered: v[5] as usize,
                duplicate_roots: v[6] as usize,
                recovery_rejected: v[7] as usize,
                duplicate_models: v[8] as usize,
                dropped_leading: v[9] as usize,
                rank: v[11] as usize,
                jacobi_sweeps: v[12] as usize,
                slots: slots?.try_into().unwrap(),
            })
        })
        .collect()
}
