use super::*;
use std::time::Instant;

pub const FIVE_POINT_PASS_NAMES: [&str; 9] = [
    "constraints",
    "basis",
    "pack",
    "elimination",
    "algebra",
    "polynomial",
    "validate_polynomial",
    "roots",
    "recover",
];

/// Host intervals are disjoint wall times; GPU intervals overlap host wait.
/// None means timestamp queries were unavailable/disabled, not zero GPU time.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct FivePointProfile {
    pub gpu_pass_seconds: Option<[Option<f64>; 9]>,
    /// Raw evidence retained even when equal boundary ticks make a duration unusable.
    pub timestamp_ticks: Option<[u64; 18]>,
    pub timestamp_period_ns: Option<f32>,
    pub prepare_seconds: f64,
    pub encode_seconds: f64,
    pub submit_seconds: f64,
    pub wait_seconds: f64,
    pub readback_seconds: f64,
    pub decode_seconds: f64,
    pub timestamp_readback_seconds: f64,
    pub total_seconds: f64,
}

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
        self.solve_essential_inner(rays1, rays2, None)
    }

    /// Same nine dispatches and buffers as the unprofiled experiment. All queries
    /// resolve after the last pass; no intermediate stage readback or host solving.
    pub fn solve_essential_profiled(
        &self,
        rays1: &[[f32; 3]],
        rays2: &[[f32; 3]],
    ) -> Result<(Vec<FivePointTrialResult>, FivePointProfile)> {
        let mut profile = FivePointProfile::default();
        let results = self.solve_essential_inner(rays1, rays2, Some(&mut profile))?;
        Ok((results, profile))
    }

    fn solve_essential_inner(
        &self,
        rays1: &[[f32; 3]],
        rays2: &[[f32; 3]],
        profile: Option<&mut FivePointProfile>,
    ) -> Result<Vec<FivePointTrialResult>> {
        let started = Instant::now();
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
        let queries = (profile.is_some() && self.context.timestamp_queries_enabled()).then(|| {
            device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("experimental five-point pass timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 18,
            })
        });
        let resolved = queries.as_ref().map(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("five-point resolved timestamps"),
                size: 18 * 8,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        });
        let prepare_seconds = started.elapsed().as_secs_f64();
        let encode_started = Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("complete GPU five-point"),
        });
        let mut stage = 0;
        let mut dispatch = |encoder: &mut wgpu::CommandEncoder,
                            kernel: &Kernel,
                            buffers: &[&wgpu::Buffer],
                            count,
                            rows| {
            self.complete_pass_timed(
                encoder,
                kernel,
                buffers,
                count,
                rows,
                queries.as_ref().map(|q| (q, stage)),
            );
            stage += 1;
        };
        dispatch(
            &mut encoder,
            &self.constraints,
            &[&left, &right, &constraints],
            count,
            1,
        );
        dispatch(
            &mut encoder,
            &self.basis,
            &[&constraints, &diagnostics],
            count,
            1,
        );
        dispatch(
            &mut encoder,
            &self.complete.pack,
            &[&diagnostics, &basis],
            count,
            1,
        );
        for (kernel, rows) in [
            (&self.elimination, 200u32.div_ceil(32)),
            (&self.algebra, 1),
            (&self.polynomial, 11),
            (&self.validate_polynomial, 1),
        ] {
            dispatch(&mut encoder, kernel, &[&basis, &algebra], count, rows);
        }
        let buffers = [&constraints, &diagnostics, &algebra, &basis, &out];
        dispatch(
            &mut encoder,
            &self.complete.roots,
            &buffers,
            count.div_ceil(32),
            1,
        );
        dispatch(&mut encoder, &self.complete.recover, &buffers, count, 1);
        let commands = encoder.finish();
        let encode_seconds = encode_started.elapsed().as_secs_f64();
        let submit_started = Instant::now();
        let submission = self.context.queue().submit(Some(commands));
        let submit_seconds = submit_started.elapsed().as_secs_f64();
        let wait_started = Instant::now();
        self.context.wait_for(submission)?;
        let wait_seconds = wait_started.elapsed().as_secs_f64();
        let readback_started = Instant::now();
        let values = self.context.read_buffer::<f32>(&out, count * STRIDE)?;
        let readback_seconds = readback_started.elapsed().as_secs_f64();
        let decode_started = Instant::now();
        let results = decode(&values)?;
        let decode_seconds = decode_started.elapsed().as_secs_f64();
        if let Some(profile) = profile {
            let query_started = Instant::now();
            let mut timestamp_ticks = None;
            let mut timestamp_period_ns = None;
            let gpu_pass_seconds = if let (Some(q), Some(buffer)) = (&queries, &resolved) {
                // Resolve only after completion of the entire compute submission.
                // Same-submission Metal resolves have returned zero last-pass
                // samples in this experiment. This is one final resolve, not
                // per-stage synchronization or a change to the compute passes.
                let mut resolve_encoder = device.create_command_encoder(&Default::default());
                resolve_encoder.resolve_query_set(q, 0..18, buffer, 0);
                self.context
                    .wait_for(self.context.queue().submit(Some(resolve_encoder.finish())))?;
                let ticks: [u64; 18] = self
                    .context
                    .read_buffer::<u64>(buffer, 18)?
                    .try_into()
                    .unwrap();
                let period_ns = self.context.queue().get_timestamp_period();
                let period = period_ns as f64 * 1e-9;
                timestamp_ticks = Some(ticks);
                timestamp_period_ns = Some(period_ns);
                ensure!(
                    ticks.chunks_exact(2).all(|p| p[1] >= p[0]),
                    "nonmonotonic GPU timestamps"
                );
                Some(std::array::from_fn(|i| {
                    // Equal samples have been observed on Metal for a nonempty
                    // recovery dispatch. Do not present them as zero-cost work.
                    (ticks[2 * i + 1] > ticks[2 * i])
                        .then(|| (ticks[2 * i + 1] - ticks[2 * i]) as f64 * period)
                }))
            } else {
                None
            };
            *profile = FivePointProfile {
                gpu_pass_seconds,
                timestamp_ticks,
                timestamp_period_ns,
                prepare_seconds,
                encode_seconds,
                submit_seconds,
                wait_seconds,
                readback_seconds,
                decode_seconds,
                timestamp_readback_seconds: query_started.elapsed().as_secs_f64(),
                total_seconds: started.elapsed().as_secs_f64(),
            };
        }
        Ok(results)
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
            count.div_ceil(32),
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
        self.complete_pass_timed(encoder, kernel, buffers, count, rows, None);
    }

    #[allow(clippy::too_many_arguments)]
    fn complete_pass_timed(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        kernel: &Kernel,
        buffers: &[&wgpu::Buffer],
        count: usize,
        rows: u32,
        timestamp: Option<(&wgpu::QuerySet, u32)>,
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
            timestamp_writes: timestamp.map(|(query_set, stage)| {
                wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: Some(stage * 2),
                    end_of_pass_write_index: Some(stage * 2 + 1),
                }
            }),
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
