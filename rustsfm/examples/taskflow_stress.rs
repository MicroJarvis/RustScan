//! Closed-loop CPU contention benchmark, not a full SfM DAG or GPU benchmark.
//! Use tools/taskflow_stress.py for isolated release runs and OS resource metrics.
#[allow(dead_code)]
#[path = "taskflow_ba.rs"]
mod demo;

use anyhow::{ensure, Context, Result};
use clap::{Parser, ValueEnum};
use rustscan_taskflow::{
    Budget, CpuRequest, ResourceRequest, Runtime, RuntimeConfig, TaskGraph, TaskVariant,
};
use rustsfm::ba::{try_refine_bundle_adjustment, BundleAdjustmentOptions};
use rustsfm::sift::SiftExtractionOptions;
use rustsfm::types::{ImageFrame, Reconstruction};
use rustsfm::CeresBaTaskflow;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::{Arc, Barrier, Condvar, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Mode {
    Uncontrolled,
    Fixed,
    Fixed4,
    Taskflow,
}

impl Mode {
    fn fixed_slots(self) -> Option<usize> {
        match self {
            Self::Fixed => Some(2),
            Self::Fixed4 => Some(4),
            _ => None,
        }
    }

    fn ba_threads(self) -> isize {
        match self {
            Self::Fixed => 2,
            Self::Fixed4 => 1,
            _ => 4,
        }
    }
}

#[derive(Parser)]
struct Args {
    #[arg(long, value_enum)]
    mode: Mode,
    #[arg(long)]
    workflows: usize,
    #[arg(long, default_value_t = 3)]
    cycles: usize,
    #[arg(long)]
    output: PathBuf,
}

// Benchmark-only FIFO admission: fixed uses 2 slots x 2 BA threads; fixed4
// uses 4 slots x 1 BA thread. Every job occupies one slot, regardless of kind.
struct Gate {
    state: Mutex<(usize, usize, usize)>, // next ticket, serving ticket, active
    changed: Condvar,
    capacity: usize,
}
struct Permit<'a>(&'a Gate);
impl Gate {
    fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            state: Mutex::new((0, 0, 0)),
            changed: Condvar::new(),
            capacity,
        }
    }

    fn acquire(&self) -> Permit<'_> {
        let mut state = self.state.lock().unwrap();
        let ticket = state.0;
        state.0 += 1;
        while ticket != state.1 || state.2 == self.capacity {
            state = self.changed.wait(state).unwrap();
        }
        state.1 += 1;
        state.2 += 1;
        self.changed.notify_all();
        Permit(self)
    }
}
impl Drop for Permit<'_> {
    fn drop(&mut self) {
        self.0.state.lock().unwrap().2 -= 1;
        self.0.changed.notify_all();
    }
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn rmse(model: &Reconstruction, frames: &[ImageFrame]) -> f64 {
    let mut error = 0.0;
    let mut observations = 0;
    for point in &model.points {
        for obs in &point.track {
            let p = model.poses[obs.image]
                .as_ref()
                .unwrap()
                .transform_point(&point.xyz);
            let xy = model
                .camera
                .img_from_cam(p[0] as f64, p[1] as f64, p[2] as f64)
                .unwrap();
            let kp = &frames[obs.image].keypoints[obs.feature];
            error += (xy[0] - kp.x() as f64).powi(2) + (xy[1] - kp.y() as f64).powi(2);
            observations += 1;
        }
    }
    (error / observations as f64).sqrt()
}

fn extract(gray: &[u8]) -> Result<(usize, u64)> {
    let features = rustsfm::sift::extract_sift_from_grayscale_u8(
        gray,
        512,
        512,
        &SiftExtractionOptions::default(),
    )?;
    let hash = features
        .descriptors_u8
        .iter()
        .flatten()
        .fold(0xcbf29ce484222325u64, |hash, &byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        });
    Ok((features.keypoints.len(), hash))
}

struct Bench {
    mode: Mode,
    gate: Gate,
    runtime: Option<Arc<Runtime>>,
    adapter: Option<CeresBaTaskflow>,
    large: (Vec<ImageFrame>, Reconstruction),
    small: (Vec<ImageFrame>, Reconstruction),
    gray: Arc<Vec<u8>>,
}
impl Bench {
    fn ba(&self, small: bool) -> Result<Value> {
        let (frames, base) = if small { &self.small } else { &self.large };
        // Input clone precedes submission: excluded from job latency, included in
        // overall workload time and process RSS. Identical for all modes.
        let mut model = base.clone();
        let start = Instant::now();
        let permit = self.mode.fixed_slots().map(|_| self.gate.acquire());
        let gate_ms = if permit.is_some() { ms(start) } else { 0.0 };
        let report = try_refine_bundle_adjustment(
            frames,
            &mut model,
            BundleAdjustmentOptions {
                iterations: 15,
                constant_images: vec![0, 1],
                num_threads: self.mode.ba_threads(),
                taskflow: self.adapter.clone(),
                ..Default::default()
            },
        )?
        .context("BA returned no report")?;
        drop(permit);
        let elapsed_ms = ms(start);
        let queue_ms = report.scheduling.as_ref().map_or(gate_ms, |s| s.queue_ms);
        let error = rmse(&model, frames);
        ensure!(
            report.is_solution_usable() && report.final_cost <= report.initial_cost,
            "unusable BA"
        );
        ensure!(
            error.is_finite() && error < 1e-3,
            "BA RMSE too large: {error}"
        );
        for image in 0..2 {
            for probe in [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
                ensure!(
                    model.poses[image].as_ref().unwrap().transform_point(&probe)
                        == base.poses[image].as_ref().unwrap().transform_point(&probe),
                    "fixed pose changed"
                );
            }
        }
        Ok(json!({"kind": if small {"small_ba"} else {"large_ba"},
            "queue_ms": queue_ms, "elapsed_ms": elapsed_ms, "compute_ms": elapsed_ms - queue_ms,
            "solver_threads": report.solver_num_threads,
            "grant": report.scheduling.as_ref().map(|s| s.granted_cpu_threads),
            "initial_cost": report.initial_cost, "final_cost": report.final_cost,
            "rmse_px": error, "usable": report.is_solution_usable(),
            "termination": format!("{:?}/{:?}", report.termination_type, report.termination_reason)}))
    }

    fn sift(&self) -> Result<Value> {
        let start = Instant::now();
        let (queue_ms, (count, hash)) = if let Some(runtime) = &self.runtime {
            let gray = self.gray.clone();
            let mut request = ResourceRequest::cpu(CpuRequest::fixed(1));
            request.working_memory_bytes = 256 * 1024 * 1024;
            let mut graph = TaskGraph::new();
            let node = graph.task(
                "SIFT",
                vec![TaskVariant::cpu("cpu", request, move |_| {
                    let queue_ms = ms(start);
                    extract(&gray)
                        .map(|result| (queue_ms, result))
                        .map_err(|error| rustscan_taskflow::TaskError::Failed(format!("{error:#}")))
                })],
            )?;
            let run = runtime.submit(graph)?;
            ensure!(
                run.wait_timeout(Duration::from_secs(90))
                    .context("SIFT timeout")?
                    .succeeded(),
                "SIFT failed"
            );
            *run.output(&node)?
        } else {
            let permit = self.mode.fixed_slots().map(|_| self.gate.acquire());
            let queue_ms = if permit.is_some() { ms(start) } else { 0.0 };
            let result = extract(&self.gray)?;
            drop(permit);
            (queue_ms, result)
        };
        let elapsed_ms = ms(start);
        // Matches the same image/backend in the previous paired comparison.
        ensure!(
            count == 7179 && hash == 12054017411631133491,
            "SIFT output mismatch: {count}/{hash}"
        );
        Ok(
            json!({"kind": "sift", "queue_ms": queue_ms, "elapsed_ms": elapsed_ms,
            "compute_ms": elapsed_ms - queue_ms, "solver_threads": 1,
            "keypoints": count, "descriptor_hash": hash}),
        )
    }
}

fn main() -> Result<()> {
    ensure!(!cfg!(debug_assertions), "run with --release");
    ensure!(
        cfg!(feature = "ceres-ba") && cfg!(feature = "vlfeat-sift"),
        "enable ceres-ba and vlfeat-sift"
    );
    let args = Args::parse();
    ensure!(
        args.workflows > 0 && args.cycles > 0,
        "workflows and cycles must be positive"
    );
    let steps = args.cycles.checked_mul(4).context("cycles overflow")?;
    args.workflows
        .checked_mul(steps)
        .context("job count overflow")?;
    for key in [
        "VECLIB_MAXIMUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
    ] {
        ensure!(
            std::env::var(key).as_deref() == Ok("1"),
            "set {key}=1 before launching"
        );
    }
    let output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    let runtime = if args.mode == Mode::Taskflow {
        Some(Arc::new(Runtime::new(RuntimeConfig {
            budget: Budget {
                cpu_threads: 4,
                memory_bytes: 1024 * 1024 * 1024,
                io_slots: 2,
            },
            ..Default::default()
        })?))
    } else {
        None
    };
    let adapter = runtime
        .as_ref()
        .map(|runtime| CeresBaTaskflow::new(runtime.clone(), 4, 256 * 1024 * 1024))
        .transpose()?;
    let bench = Bench {
        mode: args.mode,
        gate: Gate::new(args.mode.fixed_slots().unwrap_or(1)),
        runtime,
        adapter,
        large: demo::fixture_with_points(8000),
        small: demo::fixture_with_points(100),
        gray: Arc::new(
            (0..512 * 512)
                .map(|index| {
                    let x = index % 512;
                    let y = index / 512;
                    (((x * 13) ^ (y * 17) ^ ((x / 23 + y / 19) * 43)) % 256) as u8
                })
                .collect(),
        ),
    };
    let barrier = Barrier::new(args.workflows);
    let start = Instant::now();
    let jobs = std::thread::scope(|scope| -> Result<Vec<Value>> {
        let workers: Vec<_> = (0..args.workflows)
            .map(|workflow| {
                let bench = &bench;
                let barrier = &barrier;
                scope.spawn(move || -> Result<Vec<Value>> {
                    barrier.wait();
                    (0..steps)
                        .map(|sequence| {
                            let submitted_ms = ms(start);
                            // Same counts in every workflow; phase rotation avoids forcing
                            // all workflows through a synchronized all-BA/all-SIFT wave.
                            let mut record = match (sequence + workflow) % 4 {
                                0 => bench.ba(false)?,
                                2 => bench.sift()?,
                                _ => bench.ba(true)?,
                            };
                            record["workflow"] = json!(workflow);
                            record["sequence"] = json!(sequence);
                            record["preparation_started_ms"] = json!(submitted_ms);
                            Ok(record)
                        })
                        .collect()
                })
            })
            .collect();
        let mut jobs = Vec::new();
        for worker in workers {
            jobs.extend(worker.join().expect("workflow panic")?);
        }
        Ok(jobs)
    })?;
    let workload_ms = ms(start);
    if let Some(runtime) = &bench.runtime {
        let used = runtime.snapshot()?;
        ensure!(
            used.cpu_threads == 0 && used.memory_bytes == 0 && used.pending_tasks == 0,
            "reservations leaked: {used:?}"
        );
    }
    // Overall timer includes clones, verification and joins, but excludes fixture
    // and Runtime creation and final JSON serialization. compute_ms means service
    // wall time including dispatch/cleanup, NOT measured CPU time.
    serde_json::to_writer_pretty(
        output,
        &json!({
            "mode": args.mode.to_possible_value().unwrap().get_name(), "workflows": args.workflows,
            "cycles": args.cycles, "workload_ms": workload_ms, "jobs": jobs,
            "cpu_budget": if args.mode == Mode::Uncontrolled {None} else {Some(4)},
            "ba_thread_cap": args.mode.ba_threads(), "fixed_slots": args.mode.fixed_slots(),
            "ba_iterations": 15, "residual_gate": 50000, "reservations_released": true,
        }),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn fixed_gate_limits_active_jobs_and_releases_on_unwind() {
        check_gate(2);
    }

    #[test]
    fn fixed4_gate_limits_active_jobs_and_releases_on_unwind() {
        check_gate(4);
    }

    #[test]
    fn fixed_policies_respect_four_thread_allowance() {
        for (mode, slots, threads) in [(Mode::Fixed, 2, 2), (Mode::Fixed4, 4, 1)] {
            assert_eq!(mode.fixed_slots(), Some(slots));
            assert_eq!(mode.ba_threads(), threads);
            assert_eq!(slots * threads as usize, 4);
        }
        assert_eq!(Mode::Taskflow.fixed_slots(), None);
        assert_eq!(Mode::Taskflow.ba_threads(), 4);
        assert_eq!(Mode::Uncontrolled.fixed_slots(), None);
    }

    fn check_gate(capacity: usize) {
        let gate = Gate::new(capacity);
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let admitted = Barrier::new(capacity);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    let _permit = gate.acquire();
                    let n = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(n, Ordering::SeqCst);
                    admitted.wait();
                    active.fetch_sub(1, Ordering::SeqCst);
                });
            }
        });
        assert_eq!(peak.load(Ordering::SeqCst), capacity);
        assert!(std::panic::catch_unwind(|| {
            let _permit = gate.acquire();
            panic!("exercise release");
        })
        .is_err());
        assert_eq!(gate.state.lock().unwrap().2, 0);
        let _permit = gate.acquire();
    }
}
