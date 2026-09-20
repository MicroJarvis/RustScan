//! Paired release-only direct Ceres vs shared Taskflow benchmark.
//! Run from the workspace root; optional first argument is a JSON output path.
#[allow(dead_code)]
#[path = "taskflow_ba.rs"]
mod demo;

use anyhow::{ensure, Context, Result};
use rustscan_sfm::ba::{
    try_refine_bundle_adjustment, BundleAdjustmentOptions, BundleAdjustmentReport,
};
use rustscan_sfm::sift::SiftExtractionOptions;
use rustscan_sfm::types::{ImageFrame, Reconstruction};
use rustscan_sfm::CeresBaTaskflow;
use rustscan_taskflow::{
    Budget, CpuRequest, ResourceRequest, Runtime, RuntimeConfig, TaskGraph, TaskVariant,
};
use serde_json::{json, Value};
use std::sync::{mpsc, Arc, Barrier};
use std::time::{Duration, Instant};

const WARMUP: usize = 2;
const ROUNDS: usize = 7;
const SCRATCH: u64 = 256 * 1024 * 1024;

#[derive(Clone, Copy)]
struct Scenario {
    name: &'static str,
    points: usize,
    jobs: usize,
    mixed: bool,
}

struct BaResult {
    model: Reconstruction,
    report: BundleAdjustmentReport,
    wall_ms: f64,
}

struct Trial {
    wall_ms: f64,
    ba: Vec<BaResult>,
    sift: Option<(f64, usize, u64)>,
}

fn sift(gray: &[u8]) -> Result<(f64, usize, u64)> {
    let start = Instant::now();
    let features = rustscan_sfm::sift::extract_sift_from_grayscale_u8(
        gray,
        512,
        512,
        &SiftExtractionOptions::default(),
    )?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    // A deterministic checksum verifies descriptor contents, not just the count.
    let hash = features
        .descriptors_u8
        .iter()
        .flatten()
        .fold(0xcbf29ce484222325u64, |hash, &byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        });
    Ok((ms, features.keypoints.len(), hash))
}

fn run(
    scenario: Scenario,
    scheduled: bool,
    runtime: &Arc<Runtime>,
    frames: &[ImageFrame],
    base: &Reconstruction,
    gray: &Arc<Vec<u8>>,
) -> Result<Trial> {
    // Setup is excluded for both paths; the timer includes thread spawn, admission,
    // complete BA calls and joins. Runtime creation is a one-off, outside all trials.
    let models: Vec<_> = (0..scenario.jobs).map(|_| base.clone()).collect();
    let cap = if scenario.jobs == 1 { 4 } else { 2 };
    let executor = CeresBaTaskflow::new(runtime.clone(), cap, SCRATCH)?;
    let mut graph = TaskGraph::new();
    let (entered, started) = mpsc::sync_channel(1);
    let mut sift_node = None;
    let mut direct_entered = Some(entered.clone());
    if scenario.mixed && scheduled {
        let gray = gray.clone();
        let mut request = ResourceRequest::cpu(CpuRequest::fixed(1));
        request.working_memory_bytes = SCRATCH;
        sift_node = Some(graph.task(
            "SIFT",
            vec![TaskVariant::cpu("cpu", request, move |_| {
                entered.send(()).unwrap();
                sift(&gray)
                    .map_err(|error| rustscan_taskflow::TaskError::Failed(format!("{error:#}")))
            })],
        )?);
    }
    let barrier = Barrier::new(scenario.jobs);
    let start = Instant::now();
    let (ba, sift_result) = std::thread::scope(|scope| -> Result<_> {
        let sift_run = if scenario.mixed && scheduled {
            Some(runtime.submit(graph)?)
        } else {
            None
        };
        let sift_worker = if scenario.mixed && !scheduled {
            let entered = direct_entered.take().unwrap();
            Some(scope.spawn(move || {
                entered.send(()).unwrap();
                sift(gray)
            }))
        } else {
            None
        };
        if scenario.mixed {
            started.recv_timeout(Duration::from_secs(10))?;
        }
        let mut workers = Vec::new();
        for (index, mut model) in models.into_iter().enumerate() {
            let executor = executor.clone();
            let barrier = &barrier;
            workers.push(scope.spawn(move || -> Result<_> {
                barrier.wait();
                // Direct mixed baseline manually budgets 2+1 BA threads plus SIFT1.
                // Taskflow can grant 2+1, or 2+2 if SIFT has already completed.
                let threads = if scenario.mixed && !scheduled && index == 1 {
                    1
                } else {
                    cap
                };
                let options = BundleAdjustmentOptions {
                    iterations: 15,
                    constant_images: vec![0, 1],
                    num_threads: threads as isize,
                    taskflow: scheduled.then_some(executor),
                    ..Default::default()
                };
                let start = Instant::now();
                let report = try_refine_bundle_adjustment(frames, &mut model, options)?
                    .context("Ceres returned no report")?;
                Ok(BaResult {
                    model,
                    report,
                    wall_ms: start.elapsed().as_secs_f64() * 1000.0,
                })
            }));
        }
        let ba = workers
            .into_iter()
            .map(|worker| worker.join().expect("BA panic"))
            .collect::<Result<Vec<_>>>()?;
        let sift_result = if let Some(run) = sift_run {
            ensure!(
                run.wait_timeout(Duration::from_secs(30))
                    .context("SIFT timeout")?
                    .succeeded(),
                "SIFT failed"
            );
            Some(*run.output(&sift_node.unwrap())?)
        } else if let Some(worker) = sift_worker {
            Some(worker.join().expect("SIFT panic")?)
        } else {
            None
        };
        Ok((ba, sift_result))
    })?;
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    let used = runtime.snapshot()?;
    ensure!(
        used.cpu_threads == 0 && used.memory_bytes == 0 && used.pending_tasks == 0,
        "leaked reservations: {used:?}"
    );
    Ok(Trial {
        wall_ms,
        ba,
        sift: sift_result,
    })
}

fn rmse(model: &Reconstruction, frames: &[ImageFrame]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0;
    for point in &model.points {
        for observation in &point.track {
            let p = model.poses[observation.image]
                .as_ref()
                .unwrap()
                .transform_point(&point.xyz);
            let xy = model
                .camera
                .img_from_cam(p[0] as f64, p[1] as f64, p[2] as f64)
                .unwrap();
            let kp = &frames[observation.image].keypoints[observation.feature];
            sum += (xy[0] - kp.x() as f64).powi(2) + (xy[1] - kp.y() as f64).powi(2);
            count += 1;
        }
    }
    (sum / count as f64).sqrt()
}

fn pose_delta(a: &Reconstruction, b: &Reconstruction, count: usize) -> f32 {
    let mut delta = 0.0f32;
    for (a, b) in a.poses.iter().zip(&b.poses).take(count) {
        for probe in [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
            let a = a.as_ref().unwrap().transform_point(&probe);
            let b = b.as_ref().unwrap().transform_point(&probe);
            for (a, b) in a.into_iter().zip(b) {
                delta = delta.max((a - b).abs());
            }
        }
    }
    delta
}

fn validate(a: &Trial, b: &Trial, base: &Reconstruction, frames: &[ImageFrame]) -> Result<Value> {
    let mut points = 0.0f32;
    let mut poses = 0.0f32;
    let mut cost = 0.0f64;
    for (a, b) in a.ba.iter().zip(&b.ba) {
        for result in [a, b] {
            ensure!(result.report.is_solution_usable(), "unusable solution");
            ensure!(
                result.report.final_cost <= result.report.initial_cost,
                "cost increased"
            );
            ensure!(rmse(&result.model, frames) < 1e-3, "RMSE exceeds 0.001px");
            ensure!(
                pose_delta(&result.model, base, 2) == 0.0,
                "fixed poses changed"
            );
        }
        for (a, b) in a.model.points.iter().zip(&b.model.points) {
            for (&a, &b) in a.xyz.iter().zip(&b.xyz) {
                points = points.max((a - b).abs());
            }
        }
        poses = poses.max(pose_delta(&a.model, &b.model, base.poses.len()));
        let diff = (a.report.final_cost - b.report.final_cost).abs();
        ensure!(
            diff < 1e-5 * a.report.initial_cost.max(1.0),
            "cost mismatch"
        );
        cost = cost.max(diff);
    }
    ensure!(
        points < 1e-4 && poses < 1e-4,
        "geometry mismatch: points={points} poses={poses}"
    );
    ensure!(
        a.sift.map(|(_, n, hash)| (n, hash)) == b.sift.map(|(_, n, hash)| (n, hash)),
        "SIFT mismatch"
    );
    Ok(json!({"max_point_delta": points, "max_pose_probe_delta": poses, "max_cost_delta": cost}))
}

fn record(trial: &Trial, frames: &[ImageFrame]) -> Value {
    json!({
        "wall_ms": trial.wall_ms,
        "ba": trial.ba.iter().map(|b| json!({
            "wall_ms": b.wall_ms, "setup_ms": b.report.setup_ms, "solve_ms": b.report.solve_ms,
            "postprocess_ms": b.report.postprocess_ms,
            "solver_threads": b.report.solver_num_threads,
            "grant": b.report.scheduling.as_ref().map(|s| s.granted_cpu_threads),
            "queue_ms": b.report.scheduling.as_ref().map(|s| s.queue_ms),
            "initial_cost": b.report.initial_cost, "final_cost": b.report.final_cost,
            "residuals": b.report.residuals, "rmse_px": rmse(&b.model, frames),
            "usable": b.report.is_solution_usable(),
            "termination": format!("{:?}/{:?}", b.report.termination_type, b.report.termination_reason),
        })).collect::<Vec<_>>(),
        "sift": trial.sift.map(|(ms, n, hash)| json!({"ms": ms, "keypoints": n, "descriptor_hash": hash})),
    })
}

fn stats(values: &[f64]) -> Value {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    json!({"median_ms": sorted[sorted.len()/2], "min_ms": sorted[0], "max_ms": sorted[sorted.len()-1]})
}

fn main() -> Result<()> {
    ensure!(!cfg!(debug_assertions), "run with --release");
    ensure!(cfg!(feature = "ceres-ba"), "enable ceres-ba");
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
    let runtime = Arc::new(Runtime::new(RuntimeConfig {
        budget: Budget {
            cpu_threads: 4,
            memory_bytes: 1024 * 1024 * 1024,
            io_slots: 2,
        },
        ..Default::default()
    })?);
    let gray = Arc::new(
        (0..512 * 512)
            .map(|index| {
                let x = index % 512;
                let y = index / 512;
                (((x * 13) ^ (y * 17) ^ ((x / 23 + y / 19) * 43)) % 256) as u8
            })
            .collect::<Vec<_>>(),
    );
    let mut scenarios = Vec::new();
    for scenario in [
        Scenario {
            name: "small_ba",
            points: 100,
            jobs: 1,
            mixed: false,
        },
        Scenario {
            name: "large_ba",
            points: 8000,
            jobs: 1,
            mixed: false,
        },
        Scenario {
            name: "two_ba",
            points: 8000,
            jobs: 2,
            mixed: false,
        },
        Scenario {
            name: "two_ba_plus_sift",
            points: 8000,
            jobs: 2,
            mixed: true,
        },
    ] {
        let (frames, base) = demo::fixture_with_points(scenario.points);
        let mut trials = Vec::new();
        let mut direct_ms = Vec::new();
        let mut scheduled_ms = Vec::new();
        for round in 0..WARMUP + ROUNDS {
            let scheduled_first = round % 2 == 1;
            let first = run(scenario, scheduled_first, &runtime, &frames, &base, &gray)?;
            let second = run(scenario, !scheduled_first, &runtime, &frames, &base, &gray)?;
            let (direct, scheduled) = if scheduled_first {
                (second, first)
            } else {
                (first, second)
            };
            let quality = validate(&direct, &scheduled, &base, &frames)?;
            if round < WARMUP {
                continue;
            }
            println!(
                "{} round{} direct={:.3}ms taskflow={:.3}ms threads={:?}/{:?}",
                scenario.name,
                round - WARMUP + 1,
                direct.wall_ms,
                scheduled.wall_ms,
                direct
                    .ba
                    .iter()
                    .map(|b| b.report.solver_num_threads)
                    .collect::<Vec<_>>(),
                scheduled
                    .ba
                    .iter()
                    .map(|b| b.report.solver_num_threads)
                    .collect::<Vec<_>>()
            );
            direct_ms.push(direct.wall_ms);
            scheduled_ms.push(scheduled.wall_ms);
            trials.push(json!({"scheduled_first": scheduled_first, "direct": record(&direct, &frames), "taskflow": record(&scheduled, &frames), "comparison": quality}));
        }
        let direct = stats(&direct_ms);
        let scheduled = stats(&scheduled_ms);
        let ratio =
            scheduled["median_ms"].as_f64().unwrap() / direct["median_ms"].as_f64().unwrap();
        println!(
            "{} median direct={}ms taskflow={}ms change={:+.2}%",
            scenario.name,
            direct["median_ms"],
            scheduled["median_ms"],
            (ratio - 1.0) * 100.0
        );
        scenarios.push(json!({"name": scenario.name, "points": scenario.points, "jobs": scenario.jobs, "direct": direct, "taskflow": scheduled, "median_ratio": ratio, "trials": trials}));
    }
    let output = json!({"warmup_pairs": WARMUP, "measured_pairs": ROUNDS, "cpu_budget": 4, "ba_iterations": 15,
        "multi_thread_residual_gate": 50000, "scenarios": scenarios});
    if let Some(path) = std::env::args().nth(1) {
        std::fs::write(&path, serde_json::to_vec_pretty(&output)?)?;
        println!("Results saved to {path}");
    }
    Ok(())
}
