//! Empty-task scheduling smoke benchmark, not a kernel or SfM speed comparison.
use rustscan_taskflow::*;
use std::time::{Duration, Instant};

fn measure(runtime: &Runtime, count: usize, chain: bool) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let mut graph = TaskGraph::new();
    let mut previous = None;
    for index in 0..count {
        let task = graph.task(
            "noop",
            vec![TaskVariant::cpu(
                "cpu",
                ResourceRequest::cpu(CpuRequest::fixed(1)),
                move |_| {
                    std::hint::black_box(index);
                    Ok(())
                },
            )],
        )?;
        if chain {
            if let Some(previous) = previous {
                graph.depends_on(task.id(), previous)?;
            }
        }
        previous = Some(task.id());
    }
    let build = start.elapsed();
    let start = Instant::now();
    let run = runtime.submit(graph)?;
    let report = run
        .wait_timeout(Duration::from_secs(30))
        .ok_or("benchmark timed out")?;
    let execution = start.elapsed();
    assert!(report.succeeded());
    assert_eq!(report.tasks.len(), count);
    let snapshot = runtime.snapshot()?;
    assert_eq!(snapshot.pending_tasks, 0);
    assert_eq!(snapshot.cpu_threads, 0);
    println!(
        "{}: tasks={count}, build={build:?}, submit+execute+report={execution:?}, tasks/s={:.0}",
        if chain { "chain" } else { "wide" },
        count as f64 / execution.as_secs_f64(),
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        return Err("run this benchmark with --release".into());
    }
    let mut config = RuntimeConfig::default();
    config.budget.cpu_threads = config.budget.cpu_threads.min(8);
    config.event_capacity = 0;
    println!(
        "CPU budget: {}; no per-task parallel pools",
        config.budget.cpu_threads
    );
    let runtime = Runtime::new(config)?;
    measure(&runtime, 100, false)?;
    for _ in 0..3 {
        measure(&runtime, 10_000, false)?;
        measure(&runtime, 10_000, true)?;
    }
    Ok(())
}
