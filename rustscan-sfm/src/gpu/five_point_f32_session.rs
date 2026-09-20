//! P2/P3 persistent five-point session: reuse buffers and staging, clear
//! zero-dependent ranges each solve, merge compute+copy into one submit/wait
//! (P3-A), and optionally pipeline two slots (P3-B double-buffer).
use super::*;
use anyhow::{bail, ensure, Context, Result};
use std::time::Instant;

const RAYS_PER_TRIAL: usize = 5;

struct SessionSlot {
    left: wgpu::Buffer,
    right: wgpu::Buffer,
    constraints: wgpu::Buffer,
    diagnostics: wgpu::Buffer,
    basis: wgpu::Buffer,
    algebra: wgpu::Buffer,
    out: wgpu::Buffer,
    params: wgpu::Buffer,
    staging: wgpu::Buffer,
}

/// Persistent GPU resources for repeated `solve_essential` calls.
///
/// Capacity may exceed the active trial count; shaders read `params.count`.
/// At most one unprofiled solve is in flight on the single-slot path; the
/// double-buffer path allows one GPU batch queued while the previous is mapped.
pub struct FivePointSession<'a> {
    solver: &'a WgpuFivePointF32,
    capacity: usize,
    slots: Vec<SessionSlot>,
    busy: bool,
    submit_count: u64,
    wait_count: u64,
    double_buffer: bool,
}

struct Inflight {
    slot: usize,
    submission: wgpu::SubmissionIndex,
    count: usize,
    trial_base: usize,
}

impl WgpuFivePointF32 {
    /// Create a reusable session with at least `initial_capacity` trials.
    pub fn session(&self, initial_capacity: usize) -> Result<FivePointSession<'_>> {
        FivePointSession::new(self, initial_capacity.max(1), false)
    }
}

impl<'a> FivePointSession<'a> {
    fn new(solver: &'a WgpuFivePointF32, capacity: usize, double_buffer: bool) -> Result<Self> {
        let n = if double_buffer { 2 } else { 1 };
        let mut slots = Vec::with_capacity(n);
        for i in 0..n {
            slots.push(SessionSlot::new(solver, capacity, i)?);
        }
        Ok(Self {
            solver,
            capacity,
            slots,
            busy: false,
            submit_count: 0,
            wait_count: 0,
            double_buffer,
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn submit_count(&self) -> u64 {
        self.submit_count
    }

    pub fn wait_count(&self) -> u64 {
        self.wait_count
    }

    pub fn double_buffer_enabled(&self) -> bool {
        self.double_buffer
    }

    pub fn reset_sync_counters(&mut self) {
        self.submit_count = 0;
        self.wait_count = 0;
    }

    /// Allocate a second buffer set for bounded pipelining (at most 2 in flight).
    pub fn enable_double_buffer(&mut self) -> Result<()> {
        if self.double_buffer {
            return Ok(());
        }
        ensure!(!self.busy, "cannot enable double-buffer while busy");
        self.slots
            .push(SessionSlot::new(self.solver, self.capacity, 1)?);
        self.double_buffer = true;
        Ok(())
    }

    /// Grow-only resize. Shrinking a later call reuses the larger capacity.
    pub fn ensure_capacity(&mut self, trials: usize) -> Result<()> {
        if trials <= self.capacity {
            return Ok(());
        }
        let limits = self.solver.context.device().limits();
        let limit = limits.max_compute_workgroups_per_dimension as usize;
        ensure!(trials <= limit, "five-point session exceeds dispatch limit");
        let mut next = self.capacity.max(1);
        while next < trials {
            next = next
                .checked_mul(2)
                .context("five-point session capacity overflow")?
                .min(limit);
            if next < trials && next == limit {
                bail!("five-point session cannot grow to {trials} under device limit {limit}");
            }
        }
        let double = self.double_buffer;
        let (submits, waits, busy) = (self.submit_count, self.wait_count, self.busy);
        *self = Self::new(self.solver, next.max(trials), double)?;
        self.submit_count = submits;
        self.wait_count = waits;
        self.busy = busy;
        Ok(())
    }

    pub fn solve_essential(
        &mut self,
        rays1: &[[f32; 3]],
        rays2: &[[f32; 3]],
    ) -> Result<Vec<FivePointTrialResult>> {
        self.solve_essential_inner(rays1, rays2, None)
    }

    pub fn solve_essential_profiled(
        &mut self,
        rays1: &[[f32; 3]],
        rays2: &[[f32; 3]],
    ) -> Result<(Vec<FivePointTrialResult>, FivePointProfile)> {
        let mut profile = FivePointProfile::default();
        let results = self.solve_essential_inner(rays1, rays2, Some(&mut profile))?;
        Ok((results, profile))
    }

    /// Pipeline many batches with at most two slots in flight (requires
    /// `enable_double_buffer`). Results are concatenated in input order.
    pub fn solve_essential_batches(
        &mut self,
        batches: &[(Vec<[f32; 3]>, Vec<[f32; 3]>)],
    ) -> Result<Vec<FivePointTrialResult>> {
        ensure!(
            self.double_buffer && self.slots.len() == 2,
            "double-buffer not enabled"
        );
        ensure!(
            !self.busy,
            "five-point session already has an in-flight solve"
        );
        if batches.is_empty() {
            return Ok(Vec::new());
        }
        self.busy = true;
        let result = (|| {
            let mut all = Vec::new();
            let mut inflight: Option<Inflight> = None;
            let mut slot = 0usize;
            let mut trial_base = 0usize;
            for (left, right) in batches {
                ensure!(
                    left.len() == right.len() && left.len() % 5 == 0,
                    "equal five-ray batches required"
                );
                ensure!(
                    left.iter()
                        .chain(right)
                        .flatten()
                        .all(|x| x.is_finite() && x.abs() <= 1e8),
                    "finite bounded rays required (abs <= 1e8)"
                );
                let count = left.len() / 5;
                if count == 0 {
                    continue;
                }
                if count > self.capacity {
                    // Growing replaces every slot (and its staging buffer), so
                    // drain the in-flight batch first or its results would be
                    // read from a freshly zeroed staging buffer.
                    if let Some(prev) = inflight.take() {
                        let mut part = self.finish_slot(prev.slot, prev.submission, prev.count)?;
                        for r in &mut part {
                            r.trial += prev.trial_base;
                        }
                        all.extend(part);
                    }
                    self.ensure_capacity(count)?;
                }
                let submission = self.submit_slot(slot, left, right, count)?;
                if let Some(prev) = inflight.take() {
                    let mut part = self.finish_slot(prev.slot, prev.submission, prev.count)?;
                    for r in &mut part {
                        r.trial += prev.trial_base;
                    }
                    all.extend(part);
                }
                inflight = Some(Inflight {
                    slot,
                    submission,
                    count,
                    trial_base,
                });
                trial_base += count;
                slot = 1 - slot;
            }
            if let Some(prev) = inflight {
                let mut part = self.finish_slot(prev.slot, prev.submission, prev.count)?;
                for r in &mut part {
                    r.trial += prev.trial_base;
                }
                all.extend(part);
            }
            Ok(all)
        })();
        self.busy = false;
        result
    }

    fn solve_essential_inner(
        &mut self,
        rays1: &[[f32; 3]],
        rays2: &[[f32; 3]],
        profile: Option<&mut FivePointProfile>,
    ) -> Result<Vec<FivePointTrialResult>> {
        ensure!(
            !self.busy,
            "five-point session already has an in-flight solve"
        );
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
        self.ensure_capacity(count)?;
        self.busy = true;
        let result = if profile.is_some() {
            self.encode_submit_read_profiled(0, rays1, rays2, count, started, profile)
        } else {
            let submission = self.submit_slot(0, rays1, rays2, count);
            match submission {
                Ok(sub) => self.finish_slot(0, sub, count),
                Err(e) => Err(e),
            }
        };
        self.busy = false;
        result
    }

    fn submit_slot(
        &mut self,
        slot: usize,
        rays1: &[[f32; 3]],
        rays2: &[[f32; 3]],
        count: usize,
    ) -> Result<wgpu::SubmissionIndex> {
        let device = self.solver.context.device();
        let queue = self.solver.context.queue();
        let left_rays: Vec<Ray> = rays1.iter().map(|&xyz| Ray { xyz, _pad: 0.0 }).collect();
        let right_rays: Vec<Ray> = rays2.iter().map(|&xyz| Ray { xyz, _pad: 0.0 }).collect();
        {
            let s = &self.slots[slot];
            queue.write_buffer(&s.left, 0, bytemuck::cast_slice(&left_rays));
            queue.write_buffer(&s.right, 0, bytemuck::cast_slice(&right_rays));
            queue.write_buffer(
                &s.params,
                0,
                bytemuck::bytes_of(&FivePointParams {
                    count: count as u32,
                    _pad: [0; 3],
                }),
            );
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("five-point session solve"),
        });
        {
            let s = &self.slots[slot];
            clear_trials(&mut encoder, &s.diagnostics, count, 40);
            clear_trials(&mut encoder, &s.basis, count, 36);
            clear_trials(&mut encoder, &s.algebra, count, 352);
            clear_trials(&mut encoder, &s.out, count, complete::STRIDE);
            self.encode_passes(&mut encoder, s, count, None)?;
            let byte_len = (count * complete::STRIDE * 4) as u64;
            encoder.copy_buffer_to_buffer(&s.out, 0, &s.staging, 0, byte_len);
        }
        let submission = queue.submit(Some(encoder.finish()));
        self.submit_count += 1;
        Ok(submission)
    }

    fn finish_slot(
        &mut self,
        slot: usize,
        submission: wgpu::SubmissionIndex,
        count: usize,
    ) -> Result<Vec<FivePointTrialResult>> {
        let byte_len = (count * complete::STRIDE * 4) as u64;
        let slice = self.slots[slot].staging.slice(0..byte_len);
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.solver.context.wait_for(submission)?;
        self.wait_count += 1;
        receiver
            .recv()
            .context("five-point session readback callback dropped")?
            .context("five-point session readback mapping failed")?;
        let mapped = slice.get_mapped_range();
        let values: Vec<f32> = mapped
            .chunks_exact(4)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        drop(mapped);
        self.slots[slot].staging.unmap();
        complete::decode(&values)
    }

    fn encode_passes(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        slot: &SessionSlot,
        count: usize,
        timestamp: Option<&wgpu::QuerySet>,
    ) -> Result<()> {
        let solver = self.solver;
        let mut stage = 0u32;
        let mut dispatch = |encoder: &mut wgpu::CommandEncoder,
                            kernel: &Kernel,
                            buffers: &[&wgpu::Buffer],
                            groups: usize,
                            rows: u32,
                            with_params: bool| {
            solver.complete_pass_timed(
                encoder,
                kernel,
                buffers,
                if with_params {
                    Some(&slot.params)
                } else {
                    None
                },
                groups,
                rows,
                timestamp.map(|q| (q, stage)),
            );
            stage += 1;
        };

        dispatch(
            encoder,
            &solver.constraints,
            &[&slot.left, &slot.right, &slot.constraints],
            count,
            1,
            false,
        );
        dispatch(
            encoder,
            &solver.basis,
            &[&slot.constraints, &slot.diagnostics],
            count.div_ceil(32),
            1,
            true,
        );
        dispatch(
            encoder,
            &solver.complete.pack,
            &[&slot.diagnostics, &slot.basis],
            count,
            1,
            false,
        );
        for (kernel, rows) in [
            (&solver.elimination, 200u32.div_ceil(32)),
            (&solver.algebra, 1),
            (&solver.polynomial, 11),
            (&solver.validate_polynomial, 1),
        ] {
            let groups = if std::ptr::eq(kernel, &solver.algebra) {
                count.div_ceil(32)
            } else {
                count
            };
            dispatch(
                encoder,
                kernel,
                &[&slot.basis, &slot.algebra],
                groups,
                rows,
                std::ptr::eq(kernel, &solver.algebra),
            );
        }
        let buffers = [
            &slot.constraints,
            &slot.diagnostics,
            &slot.algebra,
            &slot.basis,
            &slot.out,
        ];
        dispatch(
            encoder,
            &solver.complete.roots,
            &buffers,
            count.div_ceil(32),
            1,
            true,
        );
        dispatch(encoder, &solver.complete.recover, &buffers, count, 1, false);
        Ok(())
    }

    /// Profiled path stays single-slot and single-flight (timestamp resolve is
    /// intentionally separate from the result copy).
    fn encode_submit_read_profiled(
        &mut self,
        slot: usize,
        rays1: &[[f32; 3]],
        rays2: &[[f32; 3]],
        count: usize,
        started: Instant,
        profile: Option<&mut FivePointProfile>,
    ) -> Result<Vec<FivePointTrialResult>> {
        let device = self.solver.context.device();
        let queue = self.solver.context.queue();
        let left_rays: Vec<Ray> = rays1.iter().map(|&xyz| Ray { xyz, _pad: 0.0 }).collect();
        let right_rays: Vec<Ray> = rays2.iter().map(|&xyz| Ray { xyz, _pad: 0.0 }).collect();
        {
            let s = &self.slots[slot];
            queue.write_buffer(&s.left, 0, bytemuck::cast_slice(&left_rays));
            queue.write_buffer(&s.right, 0, bytemuck::cast_slice(&right_rays));
            queue.write_buffer(
                &s.params,
                0,
                bytemuck::bytes_of(&FivePointParams {
                    count: count as u32,
                    _pad: [0; 3],
                }),
            );
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("five-point session profiled solve"),
        });
        let prepare_seconds = started.elapsed().as_secs_f64();
        let encode_started = Instant::now();
        let queries =
            (profile.is_some() && self.solver.context.timestamp_queries_enabled()).then(|| {
                device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("five-point session timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 18,
                })
            });
        let resolved = queries.as_ref().map(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("five-point session resolved timestamps"),
                size: 18 * 8,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        });
        {
            let s = &self.slots[slot];
            clear_trials(&mut encoder, &s.diagnostics, count, 40);
            clear_trials(&mut encoder, &s.basis, count, 36);
            clear_trials(&mut encoder, &s.algebra, count, 352);
            clear_trials(&mut encoder, &s.out, count, complete::STRIDE);
            self.encode_passes(&mut encoder, s, count, queries.as_ref())?;
            let byte_len = (count * complete::STRIDE * 4) as u64;
            encoder.copy_buffer_to_buffer(&s.out, 0, &s.staging, 0, byte_len);
        }
        let encode_seconds = encode_started.elapsed().as_secs_f64();
        let submit_started = Instant::now();
        let submission = queue.submit(Some(encoder.finish()));
        self.submit_count += 1;
        let submit_seconds = submit_started.elapsed().as_secs_f64();

        let readback_started = Instant::now();
        let byte_len = (count * complete::STRIDE * 4) as u64;
        let slice = self.slots[slot].staging.slice(0..byte_len);
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let wait_started = Instant::now();
        self.solver.context.wait_for(submission)?;
        self.wait_count += 1;
        let wait_seconds = wait_started.elapsed().as_secs_f64();
        receiver
            .recv()
            .context("five-point session readback callback dropped")?
            .context("five-point session readback mapping failed")?;
        let mapped = slice.get_mapped_range();
        let values: Vec<f32> = mapped
            .chunks_exact(4)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        drop(mapped);
        self.slots[slot].staging.unmap();
        let readback_seconds = readback_started.elapsed().as_secs_f64();
        let decode_started = Instant::now();
        let results = complete::decode(&values)?;
        let decode_seconds = decode_started.elapsed().as_secs_f64();

        if let Some(profile) = profile {
            let query_started = Instant::now();
            let mut timestamp_ticks = None;
            let mut timestamp_period_ns = None;
            let gpu_pass_seconds = if let (Some(q), Some(buffer)) = (&queries, &resolved) {
                let mut resolve_encoder = device.create_command_encoder(&Default::default());
                resolve_encoder.resolve_query_set(q, 0..18, buffer, 0);
                self.solver
                    .context
                    .wait_for(queue.submit(Some(resolve_encoder.finish())))?;
                self.submit_count += 1;
                self.wait_count += 1;
                let ticks: [u64; 18] = self
                    .solver
                    .context
                    .read_buffer::<u64>(buffer, 18)?
                    .try_into()
                    .unwrap();
                let period_ns = queue.get_timestamp_period();
                let period = period_ns as f64 * 1e-9;
                timestamp_ticks = Some(ticks);
                timestamp_period_ns = Some(period_ns);
                ensure!(
                    ticks.chunks_exact(2).all(|p| p[1] >= p[0]),
                    "nonmonotonic GPU timestamps"
                );
                Some(std::array::from_fn(|i| {
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
}

impl SessionSlot {
    fn new(solver: &WgpuFivePointF32, capacity: usize, index: usize) -> Result<Self> {
        solver.complete_size(capacity, complete::STRIDE)?;
        let device = solver.context.device();
        let ray_bytes = capacity
            .checked_mul(RAYS_PER_TRIAL)
            .and_then(|n| n.checked_mul(std::mem::size_of::<Ray>()))
            .context("five-point session ray size overflow")?;
        solver.complete_size_bytes(ray_bytes)?;
        let tag = format!("five-point session slot{index}");
        Ok(Self {
            left: storage_buffer(device, &format!("{tag} left"), ray_bytes, true)?,
            right: storage_buffer(device, &format!("{tag} right"), ray_bytes, true)?,
            constraints: solver.complete_buffer(capacity, 45)?,
            diagnostics: solver.complete_buffer(capacity, 40)?,
            basis: solver.complete_buffer(capacity, 36)?,
            algebra: solver.complete_buffer(capacity, 352)?,
            out: solver.complete_buffer(capacity, complete::STRIDE)?,
            params: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{tag} params")),
                size: std::mem::size_of::<FivePointParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{tag} staging")),
                size: (capacity * complete::STRIDE * 4) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
        })
    }
}

fn storage_buffer(
    device: &wgpu::Device,
    label: &str,
    bytes: usize,
    upload: bool,
) -> Result<wgpu::Buffer> {
    let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    if !upload {
        usage |= wgpu::BufferUsages::COPY_SRC;
    }
    Ok(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes as u64,
        usage,
        mapped_at_creation: false,
    }))
}

fn clear_trials(
    encoder: &mut wgpu::CommandEncoder,
    buffer: &wgpu::Buffer,
    count: usize,
    stride: usize,
) {
    let bytes = (count * stride * 4) as u64;
    encoder.clear_buffer(buffer, 0, Some(bytes));
}
