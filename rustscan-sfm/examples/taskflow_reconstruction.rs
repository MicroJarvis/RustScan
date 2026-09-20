//! Real database-first workflows sharing one Runtime; not a scheduler policy change.
use anyhow::{ensure, Context, Result};
use clap::{Parser, ValueEnum};
use rustscan_sfm::mapper::{run_reconstruction_with_task, MapperConfig};
use rustscan_sfm::{
    SfmTaskContext, SfmTaskControl, SfmTaskEvent, SfmTaskEventKind, SfmTaskStop, SfmTaskflow,
};
use rustscan_taskflow::{Budget, Runtime, RuntimeConfig};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Barrier,
};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Scenario {
    Simultaneous,
    LateArrival,
    QueuedCancel,
}

#[path = "c4/mod.rs"]
mod c4;

const WAIT: Duration = Duration::from_secs(30);

fn snapshot(runtime: &Runtime, started: Instant) -> Result<Value> {
    let s = runtime.snapshot()?;
    Ok(json!({"at_ms": started.elapsed().as_secs_f64()*1000.0,
        "cpu_threads": s.cpu_threads, "memory_bytes": s.memory_bytes,
        "pending_tasks": s.pending_tasks, "io_slots": s.io_slots,
        "exclusive_count": s.exclusive.len(), "gpu_entries": s.gpus.len()}))
}

fn wait_snapshot(runtime: &Runtime, started: Instant, pending: usize) -> Result<Value> {
    let deadline = Instant::now() + WAIT;
    loop {
        let s = snapshot(runtime, started)?;
        if s["cpu_threads"] == 8
            && s["memory_bytes"] == 1073741824_u64
            && s["pending_tasks"] == pending
        {
            return Ok(s);
        }
        ensure!(
            Instant::now() < deadline,
            "timed out waiting for 8 CPU / 1GiB / {pending} pending: {s}"
        );
        // Poll interval only; the predicate, not elapsed time, establishes readiness.
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    /// Directory containing independent, pre-populated job-N.db files.
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_delimiter = ',')]
    sizes: Vec<usize>,
    #[arg(long, default_value_t = 8)]
    cpu_budget: usize,
    #[arg(long, value_enum, default_value_t = Scenario::Simultaneous)]
    scenario: Scenario,
    /// C4 comparator; omitted preserves the existing C1/C2/C3 harness.
    #[arg(long, value_enum)]
    mode: Option<c4::Mode>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(mode) = args.mode {
        return c4::run(&args, mode);
    }
    ensure!(
        !args.sizes.is_empty() && args.sizes.iter().all(|n| *n >= 2),
        "invalid sizes"
    );
    ensure!(args.cpu_budget >= 4, "each workflow requests four threads");
    ensure!(
        !args.output.join("report.json").exists(),
        "refusing report overwrite"
    );
    for id in 0..args.sizes.len() {
        ensure!(
            args.output.join(format!("job-{id}.db")).is_file(),
            "missing job-{id}.db"
        );
        ensure!(
            !args.output.join(format!("model-{id}")).exists(),
            "refusing model overwrite"
        );
    }
    let focused = args.scenario != Scenario::Simultaneous;
    ensure!(
        !focused || (args.sizes == [48, 48, 12] && args.cpu_budget == 8),
        "focused scenarios require --sizes 48,48,12 --cpu-budget 8"
    );
    let memory_budget = 2 * 1024 * 1024 * 1024;
    let runtime = Arc::new(Runtime::new(RuntimeConfig {
        budget: Budget {
            cpu_threads: args.cpu_budget,
            memory_bytes: memory_budget,
            io_slots: 2,
        },
        ..Default::default()
    })?);
    let executor = SfmTaskflow::new(runtime.clone(), 512 * 1024 * 1024)?;
    let simultaneous_barrier = Arc::new(Barrier::new(args.sizes.len() + 1));
    let controls: Vec<_> = args.sizes.iter().map(|_| SfmTaskControl::new()).collect();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (small_done_tx, small_done_rx) = mpsc::channel();
    let mut starts = Vec::new();
    let mut releases = Vec::new();
    let stop = Arc::new(AtomicBool::new(false));
    let started = Instant::now();
    let monitor_runtime = runtime.clone();
    let monitor_stop = stop.clone();
    let monitor = std::thread::spawn(move || -> Result<Vec<Value>> {
        let mut samples = Vec::new();
        while !monitor_stop.load(Ordering::Relaxed) {
            let s = monitor_runtime.snapshot()?;
            samples.push(json!({"at_ms": started.elapsed().as_secs_f64()*1000.0,
                "cpu_threads": s.cpu_threads, "memory_bytes": s.memory_bytes,
                "pending_tasks": s.pending_tasks, "io_slots": s.io_slots,
                "exclusive_count": s.exclusive.len(), "gpu_entries": s.gpus.len()}));
            std::thread::sleep(Duration::from_millis(10));
        }
        Ok(samples)
    });
    let (jobs, coordination) = std::thread::scope(|scope| {
        let handles: Vec<_> = args.sizes.iter().enumerate().map(|(id, &size)| {
            let executor = executor.clone();
            let simultaneous_barrier = simultaneous_barrier.clone();
            let (start_tx, start_rx) = mpsc::channel();
            starts.push(start_tx);
            let (release_tx, release_rx) = mpsc::channel();
            releases.push(release_tx);
            let entered_tx = entered_tx.clone();
            let small_done_tx = small_done_tx.clone();
            let control = controls[id].clone();
            let input = args.input.clone();
            let output = args.output.clone();
            scope.spawn(move || -> Result<Value> {
                let mut events = Vec::new();
                let mut held = false;
                let mut barrier_timeout = false;
                let mut hold_ms = 0.0;
                let mut sink = |event: SfmTaskEvent| {
                    let at_ms = started.elapsed().as_secs_f64()*1000.0;
                    if focused && id < 2 && !held && event.kind == SfmTaskEventKind::Started {
                        held = true;
                        let hold_start = Instant::now();
                        let _ = entered_tx.send(json!({"id":id,"at_ms":at_ms,"event":event}));
                        if release_rx.recv_timeout(WAIT).is_err() {
                            barrier_timeout = true;
                            control.request_cancel();
                        }
                        hold_ms = hold_start.elapsed().as_secs_f64()*1000.0;
                    }
                    events.push(json!({"at_ms":at_ms,"event":event}));
                };
                let mut task = SfmTaskContext::new(&control, &mut sink).with_taskflow(executor);
                let config = MapperConfig {
                    input, output: output.join(format!("model-{id}")),
                    database: Some(output.join(format!("job-{id}.db"))),
                    max_images: Some(size), random_seed: 1, threads: Some(4),
                    ..Default::default()
                };
                std::fs::write(output.join(format!("config-{id}.txt")), format!("{config:#?}"))?;
                start_rx.recv_timeout(WAIT).context("submission gate timed out/disconnected")?;
                if !focused { simultaneous_barrier.wait(); }
                let submitted_ms = started.elapsed().as_secs_f64()*1000.0;
                let result = run_reconstruction_with_task(&config, &mut task);
                let completed_ms = started.elapsed().as_secs_f64()*1000.0;
                let stages = task.stage_reports();
                drop(task);
                drop(sink);
                if id == 2 { let _ = small_done_tx.send(completed_ms); }
                ensure!(!barrier_timeout, "job {id}: stage-entry release timed out/disconnected");
                let common = json!({"events":events,"barrier_hold_ms":hold_ms});
                match result {
                    Ok(summary) => {
                        std::fs::write(output.join(format!("job-{id}.json")), serde_json::to_vec_pretty(&summary)?)?;
                        Ok(json!({"id":id,"size":size,"submitted_ms":submitted_ms,"completed_ms":completed_ms,
                            "latency_ms":completed_ms-submitted_ms,"stages":stages,"observation":common,
                            "registered":summary.registered_images,"pairs":summary.pairs,
                            "points":summary.points,"models":summary.models,"success":true}))
                    }
                    Err(error) => Ok(json!({"id":id,"size":size,"submitted_ms":submitted_ms,
                        "completed_ms":completed_ms,"latency_ms":completed_ms-submitted_ms,
                                                "stages":stages,"observation":common,"success":false,
                                                "typed_cancelled":error.downcast_ref::<SfmTaskStop>() == Some(&SfmTaskStop::Cancelled),
                                                "error":format!("{error:#}")})),
                }
            })
        }).collect();
        let coordination = (|| -> Result<Value> {
            for tx in starts.iter().take(if focused { 2 } else { starts.len() }) {
                tx.send(())?;
            }
            if !focused {
                simultaneous_barrier.wait();
                return Ok(json!({}));
            }
            let first = entered_rx
                .recv_timeout(WAIT)
                .context("first large Started event missing")?;
            let second = entered_rx
                .recv_timeout(WAIT)
                .context("second large Started event missing")?;
            ensure!(first["id"] != second["id"], "duplicate entry barrier");
            let occupied = wait_snapshot(&runtime, started, 2)?;
            starts[2].send(())?;
            let queued = wait_snapshot(&runtime, started, 3)?;
            let mut evidence =
                json!({"large_started":[first,second],"occupied":occupied,"queued":queued});
            if args.scenario == Scenario::QueuedCancel {
                let cancel_ms = started.elapsed().as_secs_f64() * 1000.0;
                controls[2].request_cancel();
                let returned_ms = small_done_rx
                    .recv_timeout(WAIT)
                    .context("queued cancellation did not return while large stages held")?;
                evidence["cancel_requested_ms"] = json!(cancel_ms);
                evidence["cancel_returned_ms"] = json!(returned_ms);
                evidence["cancel_response_ms"] = json!(returned_ms - cancel_ms);
                evidence["after_cancel"] = wait_snapshot(&runtime, started, 2)?;
            }
            evidence["release_large_ms"] = json!(started.elapsed().as_secs_f64() * 1000.0);
            Ok(evidence)
        })();
        // Always unblock callers, including on a failed coordination assertion.
        if coordination.is_err() {
            for control in &controls {
                control.request_cancel();
            }
        }
        for tx in &releases {
            let _ = tx.send(());
        }
        drop(starts);
        let jobs = handles
            .into_iter()
            .map(|h| {
                h.join()
                    .map_err(|_| anyhow::anyhow!("workflow panicked"))
                    .and_then(|r| r)
            })
            .collect::<Result<Vec<_>>>();
        (jobs, coordination)
    });
    stop.store(true, Ordering::Relaxed);
    let samples = monitor
        .join()
        .map_err(|_| anyhow::anyhow!("monitor panicked"))??;
    let jobs = jobs?;
    let final_snapshot = runtime.snapshot()?;
    std::fs::write(
        args.output.join("report.json"),
        serde_json::to_vec_pretty(&json!({
            "scenario":format!("{:?}",args.scenario),
                        "coordination":coordination.as_ref().map(|v| v.clone()).unwrap_or_else(|e| json!({"error":format!("{e:#}")})),
                        "cpu_budget":args.cpu_budget,"memory_budget":memory_budget,"workflow_threads":4,
            "makespan_ms":started.elapsed().as_secs_f64()*1000.0,"jobs":jobs,"samples":samples,
            "final":{"cpu_threads":final_snapshot.cpu_threads,"memory_bytes":final_snapshot.memory_bytes,
                "pending_tasks":final_snapshot.pending_tasks,"io_slots":final_snapshot.io_slots,
                "exclusive_count":final_snapshot.exclusive.len(),"gpu_entries":final_snapshot.gpus.len()}
        }))?,
    )?;
    ensure!(
        final_snapshot.cpu_threads == 0
            && final_snapshot.memory_bytes == 0
            && final_snapshot.pending_tasks == 0
            && final_snapshot.io_slots == 0
            && final_snapshot.exclusive.is_empty()
            && final_snapshot.gpus.is_empty(),
        "resources not drained"
    );
    ensure!(
        samples.iter().all(
            |s| s["cpu_threads"].as_u64().unwrap() <= args.cpu_budget as u64
                && s["memory_bytes"].as_u64().unwrap() <= memory_budget
        ),
        "budget exceeded"
    );
    coordination?;
    for job in &jobs {
        if args.scenario == Scenario::QueuedCancel && job["id"] == 2 {
            ensure!(
                job["success"] == false && job["typed_cancelled"] == true,
                "not typed admission cancellation: {job}"
            );
            let stages = job["stages"]
                .as_array()
                .context("missing cancelled stages")?;
            ensure!(
                stages.len() == 1
                    && stages[0]["stage_name"] == "reconstruction"
                    && stages[0]["granted_threads"] == 0
                    && stages[0]["granted_memory"] == 0
                    && stages[0]["service_ms"] == 0.0
                    && stages[0]["cancelled_or_failed"] == true
                    && job["observation"]["events"].as_array().unwrap().is_empty(),
                "cancelled reconstruction was admitted or emitted events: {job}"
            );
            ensure!(
                !args.output.join("model-2").exists(),
                "cancelled job exported a model"
            );
            continue;
        }
        ensure!(job["success"] == true, "workflow failed: {job}");
        let stages = job["stages"].as_array().context("missing stages")?;
        ensure!(
            stages.len() == 1 && stages[0]["stage_name"] == "reconstruction",
            "unexpected stages"
        );
        ensure!(
            stages[0]["granted_threads"] == 4 && stages[0]["cancelled_or_failed"] == false,
            "bad grant"
        );
    }
    println!("{} workflows returned; reservations drained", jobs.len());
    Ok(())
}
