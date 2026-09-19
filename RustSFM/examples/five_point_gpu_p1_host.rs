//! P1 host allocation/initialization evidence for the independent GPU f32 experiment.
//! Reuses the existing fixed input loader and complete-signature helpers unchanged.
//! Not RANSAC, not the 960-image pipeline, and no CPU runtime fallback.
#[allow(dead_code)]
#[path = "five_point_gpu_capacity_replay.rs"]
mod shared;

use anyhow::{ensure, Result};
use clap::Parser;
use rustsfm::gpu::{FivePointTrialResult, WgpuContext, WgpuFivePointF32, FIVE_POINT_PASS_NAMES};
use serde_json::json;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    path::PathBuf,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::Instant,
};

const INPUT: &str = "af07459d637b5b8c2b20c76d3ef103f058505fe0e8307578ee7e8750f3e230c5";
const GPU_SIGNATURE: &str = "1f6f3f8d88a17e4e1c42abfc85aad44a66c4b864091805594e086ca80bf9b339";
/// 512 covers the common ordinary-production physical window; the two large
/// batches match the existing offline candidate-generation reports.
const BATCHES: [usize; 3] = [512, 32768, 65024];
const WARMUPS: usize = 3;
const MEASURED: usize = 5;
/// Prefix sizes exercising partial workgroups and a near-capacity nonmultiple.
const BOUNDARIES: [usize; 7] = [1, 31, 32, 33, 63, 65, 65023];

/// Host allocation accounting. Counting is off during timed replays so the
/// atomics cannot be mistaken for part of the measured host cost; both the
/// baseline and candidate builds carry the identical gate.
struct Counting;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ZEROED_BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_LIVE_BYTES: AtomicU64 = AtomicU64::new(0);

fn record_allocation(bytes: usize, zeroed: bool) {
    if !COUNTING.load(Ordering::Relaxed) {
        return;
    }
    ALLOCATED_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    if zeroed {
        ZEROED_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    }
    let live = LIVE_BYTES.fetch_add(bytes as u64, Ordering::Relaxed) + bytes as u64;
    PEAK_LIVE_BYTES.fetch_max(live, Ordering::Relaxed);
}

fn record_release(bytes: usize) {
    if COUNTING.load(Ordering::Relaxed) {
        LIVE_BYTES.fetch_sub(bytes as u64, Ordering::Relaxed);
    }
}

// SAFETY: every method forwards to the system allocator with the same layout
// and pointer; the counters never alter allocation behaviour.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = System.alloc(layout);
        if !pointer.is_null() {
            record_allocation(layout.size(), false);
        }
        pointer
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = System.alloc_zeroed(layout);
        if !pointer.is_null() {
            record_allocation(layout.size(), true);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        record_release(layout.size());
        System.dealloc(pointer, layout);
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let next = System.realloc(pointer, layout, new_size);
        if !next.is_null() {
            record_release(layout.size());
            record_allocation(new_size, false);
        }
        next
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[derive(Debug, Clone, Copy, Default)]
struct Accounting {
    allocated_bytes: u64,
    allocations: u64,
    zeroed_bytes: u64,
    peak_live_bytes: u64,
}

/// Measures one replay in isolation; the counters are enabled only here.
fn account<T>(body: impl FnOnce() -> Result<T>) -> Result<(T, Accounting)> {
    LIVE_BYTES.store(0, Ordering::SeqCst);
    PEAK_LIVE_BYTES.store(0, Ordering::SeqCst);
    ALLOCATED_BYTES.store(0, Ordering::SeqCst);
    ALLOCATIONS.store(0, Ordering::SeqCst);
    ZEROED_BYTES.store(0, Ordering::SeqCst);
    COUNTING.store(true, Ordering::SeqCst);
    let value = body();
    COUNTING.store(false, Ordering::SeqCst);
    let accounting = Accounting {
        allocated_bytes: ALLOCATED_BYTES.load(Ordering::SeqCst),
        allocations: ALLOCATIONS.load(Ordering::SeqCst),
        zeroed_bytes: ZEROED_BYTES.load(Ordering::SeqCst),
        peak_live_bytes: PEAK_LIVE_BYTES.load(Ordering::SeqCst),
    };
    Ok((value?, accounting))
}

/// Disjoint host wall times summed across the calls of one replay. GPU pass
/// intervals overlap the host wait and are reported separately, never added.
#[derive(Debug, Clone, Copy, Default)]
struct Phases {
    prepare: f64,
    encode: f64,
    submit: f64,
    wait: f64,
    readback: f64,
    decode: f64,
    total: f64,
    calls: usize,
}

impl Phases {
    fn json(&self) -> serde_json::Value {
        json!({"prepare_ms":self.prepare*1000.0,"encode_ms":self.encode*1000.0,
            "submit_ms":self.submit*1000.0,"wait_ms":self.wait*1000.0,
            "readback_ms":self.readback*1000.0,"decode_ms":self.decode*1000.0,
            "solver_total_ms":self.total*1000.0,"calls":self.calls})
    }
}

fn rays(inputs: &[shared::cpu::Input]) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let convert = |left: bool| {
        inputs
            .iter()
            .flat_map(move |i| {
                (if left { &i.left } else { &i.right })
                    .iter()
                    .map(|r| [r.x as f32, r.y as f32, r.z as f32])
            })
            .collect()
    };
    (convert(true), convert(false))
}

/// Same chunking, conversion and trial renumbering as `shared::gpu_replay`,
/// but through the profiled entry point so host phases are attributable.
fn profiled_replay(
    gpu: &WgpuFivePointF32,
    inputs: &[shared::cpu::Input],
    batch: usize,
) -> Result<(Vec<FivePointTrialResult>, Phases)> {
    let mut all = Vec::with_capacity(inputs.len());
    let mut phases = Phases::default();
    for (chunk, inputs) in inputs.chunks(batch).enumerate() {
        let (left, right) = rays(inputs);
        let (mut results, profile) = gpu.solve_essential_profiled(&left, &right)?;
        for result in &mut results {
            result.trial += chunk * batch;
        }
        all.extend(results);
        phases.prepare += profile.prepare_seconds;
        phases.encode += profile.encode_seconds;
        phases.submit += profile.submit_seconds;
        phases.wait += profile.wait_seconds;
        phases.readback += profile.readback_seconds;
        phases.decode += profile.decode_seconds;
        phases.total += profile.total_seconds;
        phases.calls += 1;
    }
    Ok((all, phases))
}

fn median(values: &[f64]) -> f64 {
    assert!(!values.is_empty());
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
}

fn spread(values: &[f64]) -> serde_json::Value {
    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    json!({"median":median(values),"min":minimum,"max":maximum,
        "max_over_min":maximum/minimum,"samples":values})
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    database: PathBuf,
    /// Free-form build identity recorded in the report; does not change behaviour.
    #[arg(long)]
    label: String,
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.output.exists(), "refusing to overwrite experiment");
    let (inputs, pairs, _, observations) =
        shared::cpu::load_with_observations(&shared::cpu::Args {
            database: args.database.clone(),
            pairs: 127,
            trials: 512,
            batch_size: 512,
        })?;
    drop(observations);
    let digest = shared::cpu::input_digest(&inputs);
    ensure!(
        inputs.len() == 65024 && digest == INPUT,
        "input provenance changed"
    );
    let initialization = Instant::now();
    let context = WgpuContext::try_new()?;
    ensure!(
        !context.timestamp_queries_enabled(),
        "unprofiled context required for host phase attribution"
    );
    let device = format!("{:?}", context.capabilities());
    let gpu = WgpuFivePointF32::from_context(context)?;
    let initialization_seconds = initialization.elapsed().as_secs_f64();

    // Cold first call, before any warmup, at the smallest batch.
    let cold = Instant::now();
    let first = shared::gpu_replay(&gpu, &inputs[..BATCHES[0]], BATCHES[0])?;
    let cold_first_call_ms = cold.elapsed().as_secs_f64() * 1000.0;
    ensure!(first.len() == BATCHES[0], "cold call trials");
    drop(first);

    let reference = shared::gpu_replay(&gpu, &inputs, inputs.len())?;
    ensure!(reference.len() == inputs.len(), "missing trials");
    let bits = shared::gpu_bits(&reference);
    let signature = shared::signature(&bits);
    ensure!(
        signature == GPU_SIGNATURE,
        "historical signature changed: {signature}"
    );
    drop(reference);

    let mut boundaries = Vec::new();
    for n in BOUNDARIES {
        let result = shared::gpu_replay(&gpu, &inputs[..n], n)?;
        ensure!(
            result.len() == n && shared::gpu_bits(&result) == bits[..n],
            "prefix {n} differs from the full-run prefix"
        );
        // Repeating the same prefix on reused resources must not import
        // state from the preceding larger or smaller call.
        let repeat = shared::gpu_replay(&gpu, &inputs[..n], n)?;
        ensure!(
            shared::gpu_bits(&repeat) == bits[..n],
            "prefix {n} not reproducible on reuse"
        );
        boundaries
            .push(json!({"count":n,"signature":shared::signature(&shared::gpu_bits(&result))}));
    }

    // Size alternation and failure-then-success reuse on one solver instance.
    let mut alternation = Vec::new();
    for &n in &[65024usize, 1, 32768, 65, 512, 65023, 32] {
        let result = shared::gpu_replay(&gpu, &inputs[..n], n)?;
        ensure!(
            shared::gpu_bits(&result) == bits[..n],
            "alternating size {n} differs"
        );
        alternation.push(n);
    }
    let degenerate = gpu.solve_essential(&[[0.0; 3]; 40], &[[0.0; 3]; 40])?;
    ensure!(
        degenerate.len() == 8 && degenerate.iter().all(|r| r.model_count == 0),
        "degenerate batch must produce no models"
    );
    ensure!(gpu.solve_essential(&[], &[])?.is_empty(), "empty input");
    let after_failure = shared::gpu_replay(&gpu, &inputs[..512], 512)?;
    ensure!(
        shared::gpu_bits(&after_failure) == bits[..512],
        "success after failure differs"
    );

    let mut reports = Vec::new();
    for batch in BATCHES {
        let mut warmup_ms = Vec::new();
        let mut measured_ms = Vec::new();
        for iteration in 0..WARMUPS + MEASURED {
            let started = Instant::now();
            let result = shared::gpu_replay(&gpu, &inputs, batch)?;
            let ms = started.elapsed().as_secs_f64() * 1000.0;
            ensure!(
                result.len() == inputs.len()
                    && shared::signature(&shared::gpu_bits(&result)) == signature,
                "timed replay signature changed at batch {batch}"
            );
            if iteration < WARMUPS {
                warmup_ms.push(ms);
            } else {
                measured_ms.push(ms);
            }
        }
        // Separate host-phase series: the profiled entry point on a
        // timestamp-disabled context adds no query work.
        let mut phase_records = Vec::new();
        let mut prepare_ms = Vec::new();
        let mut decode_ms = Vec::new();
        let mut solver_ms = Vec::new();
        for iteration in 0..WARMUPS + MEASURED {
            let (result, phases) = profiled_replay(&gpu, &inputs, batch)?;
            ensure!(
                shared::signature(&shared::gpu_bits(&result)) == signature,
                "profiled replay signature changed at batch {batch}"
            );
            if iteration >= WARMUPS {
                prepare_ms.push(phases.prepare * 1000.0);
                decode_ms.push(phases.decode * 1000.0);
                solver_ms.push(phases.total * 1000.0);
                phase_records.push(phases.json());
            }
        }
        // Only the replay is counted; the harness's own verification work is
        // deliberately outside the counted region.
        let (accounted, accounting) = account(|| shared::gpu_replay(&gpu, &inputs, batch))?;
        ensure!(
            accounted.len() == inputs.len()
                && shared::signature(&shared::gpu_bits(&accounted)) == signature,
            "accounting replay signature changed at batch {batch}"
        );
        drop(accounted);
        eprintln!(
            "batch={batch} endtoend_median={:.3}ms prepare_median={:.3}ms decode_median={:.3}ms allocated={}MiB peak_live={}MiB",
            median(&measured_ms),
            median(&prepare_ms),
            median(&decode_ms),
            accounting.allocated_bytes / (1024 * 1024),
            accounting.peak_live_bytes / (1024 * 1024)
        );
        reports.push(
            json!({"batch":batch,"solve_calls":inputs.len().div_ceil(batch),
            "warmup_ms":warmup_ms,"endtoend_ms":spread(&measured_ms),
            "prepare_ms":spread(&prepare_ms),"decode_ms":spread(&decode_ms),
            "solver_total_ms":spread(&solver_ms),"phases":phase_records,
            "host_allocation":{"allocated_bytes":accounting.allocated_bytes,
                "allocations":accounting.allocations,
                "zeroed_allocation_bytes":accounting.zeroed_bytes,
                "peak_live_bytes":accounting.peak_live_bytes}}),
        );
    }

    // Separate timestamp-enabled context: confirms the kernels are untouched.
    let timestamps = WgpuContext::try_new_experimental_timestamps()?;
    ensure!(
        timestamps.timestamp_queries_enabled(),
        "actual GPU timestamps required"
    );
    let profiled = WgpuFivePointF32::from_context(timestamps)?;
    let (left, right) = rays(&inputs);
    let mut kernels = Vec::new();
    for iteration in 0..WARMUPS + MEASURED {
        let (result, profile) = profiled.solve_essential_profiled(&left, &right)?;
        ensure!(
            shared::signature(&shared::gpu_bits(&result)) == signature,
            "timestamp replay signature changed"
        );
        if iteration >= WARMUPS {
            let sum: Option<f64> = profile
                .gpu_pass_seconds
                .map(|passes| passes.iter().map(|p| p.unwrap_or(0.0)).sum());
            kernels.push(json!({"seconds":profile.gpu_pass_seconds,
                "nine_kernel_sum_ms":sum.map(|s| s*1000.0),
                "ticks":profile.timestamp_ticks,"period_ns":profile.timestamp_period_ns}));
        }
    }

    let report = json!({"schema":"five-point-p1-host-v1","label":args.label,
        "source_commit":"3ab9c09","database":args.database,"device":device,
        "available_parallelism":std::thread::available_parallelism()?.get(),
        "input_digest":digest,"trials":inputs.len(),"pairs":pairs.len(),
        "unique_images":shared::unique_images(&pairs).len(),
        "unique_image_ids":shared::unique_images(&pairs),
        "signature":signature,"boundaries":boundaries,
        "size_alternation":alternation,
        "initialization_seconds":initialization_seconds,
        "cold_first_call_ms":cold_first_call_ms,
        "warmup_rounds":WARMUPS,"measured_rounds":MEASURED,
        "batches":reports,"pass_names":FIVE_POINT_PASS_NAMES,"kernels":kernels,
        "timing_contract":"End-to-end uses the existing shared replay on a timestamp-disabled context: f64-to-f32 rays, buffer creation, uploads, all kernels, waits, readback, decoding and ordered collection. Excludes database load/sampling, GPU initialization, signature and boundary checks, allocation accounting and final result destruction. Host phases come from a separate series through the profiled entry point on the same timestamp-disabled context; they are disjoint host wall times and do not include GPU pass intervals, which overlap the host wait and are reported only from the separate timestamp-enabled context. Medians are within-process over five measured replays after three warmups; no round is selected.",
        "allocation_contract":"Counting global allocator, enabled only for one dedicated untimed replay per batch, covering the solver calls and the harness ray conversion but not the signature verification. Records requested bytes, not resident pages: zeroed requests may be served by lazily mapped pages, so a byte reduction is not itself a speedup.",
        "scope":"Independent GPU f32 candidate generation only. Not RANSAC, not the 960-image pipeline, no CPU runtime fallback, no production routing change."});
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    eprintln!(
        "label={} signature={signature}; wrote {}",
        args.label,
        args.output.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accounting_counts_only_inside_the_measured_body() {
        let outside = vec![0u8; 4096];
        let ((), accounting) = account(|| {
            let inside = vec![0u8; 1 << 20];
            assert_eq!(inside.len(), 1 << 20);
            Ok(())
        })
        .unwrap();
        assert!(accounting.allocated_bytes >= 1 << 20);
        assert!(accounting.zeroed_bytes >= 1 << 20);
        assert!(accounting.peak_live_bytes >= 1 << 20);
        assert!(accounting.allocations >= 1);
        // Released inside the body, so the peak exceeds the final live total.
        assert_eq!(LIVE_BYTES.load(Ordering::SeqCst), 0);
        assert!(!COUNTING.load(Ordering::SeqCst));
        let ((), untracked) = account(|| Ok(())).unwrap();
        assert_eq!(untracked.allocated_bytes, 0);
        assert_eq!(outside.len(), 4096);
    }

    #[test]
    fn batches_and_boundaries_cover_the_required_protocol() {
        assert_eq!(BATCHES, [512, 32768, 65024]);
        assert!(BATCHES.iter().all(|&b| 65024usize.div_ceil(b) >= 1));
        assert_eq!(65024usize.div_ceil(512), 127);
        assert_eq!(65024usize.div_ceil(32768), 2);
        assert_eq!(65024usize.div_ceil(65024), 1);
        // Partial final workgroups and a near-capacity nonmultiple.
        assert!(BOUNDARIES.iter().any(|n| n % 32 != 0));
        assert!(BOUNDARIES.contains(&65023));
        assert_eq!(MEASURED, 5);
        assert_eq!(WARMUPS, 3);
    }

    #[test]
    fn median_and_spread_do_not_select_the_best_round() {
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median(&[5.0, 1.0, 3.0]), 3.0);
        let value = spread(&[2.0, 1.0, 4.0]);
        assert_eq!(value["median"], 2.0);
        assert_eq!(value["min"], 1.0);
        assert_eq!(value["max"], 4.0);
        assert_eq!(value["samples"].as_array().unwrap().len(), 3);
    }
}
