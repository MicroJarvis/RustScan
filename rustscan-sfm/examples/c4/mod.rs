//! C4-only comparators. Independent means no cross-workflow admission, not no Taskflow.
use super::*;
use std::sync::{Condvar, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(super) enum Mode {
    Independent,
    Fixed2,
    Shared,
}

impl Mode {
    fn runtime_count(self, jobs: usize) -> usize {
        if self == Self::Shared {
            1
        } else {
            jobs
        }
    }
    fn runtime_id(self, job: usize) -> usize {
        if self == Self::Shared {
            0
        } else {
            job
        }
    }
}

#[derive(Default)]
struct Slots {
    state: Mutex<(usize, usize)>, // next FIFO ticket, occupied slots
    changed: Condvar,
}
struct Permit<'a>(&'a Slots);
impl Slots {
    fn acquire(&self, ticket: usize) -> Result<Permit<'_>> {
        let state = self.state.lock().unwrap();
        let (mut state, timeout) = self
            .changed
            .wait_timeout_while(state, WAIT, |s| s.0 != ticket || s.1 == 2)
            .unwrap();
        ensure!(!timeout.timed_out(), "external slot deadline exceeded");
        state.0 += 1;
        state.1 += 1;
        self.changed.notify_all();
        Ok(Permit(self))
    }
}
impl Drop for Permit<'_> {
    fn drop(&mut self) {
        self.0.state.lock().unwrap().1 -= 1;
        self.0.changed.notify_all();
    }
}
fn drained(s: &Value) -> bool {
    [
        "cpu_threads",
        "memory_bytes",
        "pending_tasks",
        "io_slots",
        "exclusive_count",
        "gpu_entries",
    ]
    .iter()
    .all(|k| s[k] == 0)
}

pub(super) fn run(args: &Args, mode: Mode) -> Result<()> {
    ensure!(
        args.scenario == Scenario::Simultaneous
            && args.cpu_budget == 8
            && args.sizes == [48, 48, 12, 12, 12, 12, 12, 12],
        "C4 requires simultaneous, cpu8, sizes 48,48,12,12,12,12,12,12"
    );
    ensure!(
        !args.output.join("report.json").exists(),
        "refusing report overwrite"
    );
    for id in 0..args.sizes.len() {
        ensure!(
            args.output.join(format!("job-{id}.db")).is_file(),
            "missing database"
        );
        for name in [
            format!("model-{id}"),
            format!("job-{id}.json"),
            format!("config-{id}.txt"),
        ] {
            ensure!(
                !args.output.join(name).exists(),
                "refusing artifact overwrite"
            );
        }
    }
    let runtimes = (0..mode.runtime_count(args.sizes.len()))
        .map(|_| {
            Runtime::new(RuntimeConfig {
                budget: Budget {
                    cpu_threads: if mode == Mode::Shared { 8 } else { 4 },
                    memory_bytes: 2 * 1024 * 1024 * 1024,
                    io_slots: 2,
                },
                ..Default::default()
            })
            .map(Arc::new)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let slots = Slots::default();
    // Prepare callers/configs before publishing the common submission epoch.
    let (ready_tx, ready_rx) = mpsc::channel();
    let mut gates = Vec::new();
    let jobs = std::thread::scope(|scope| -> Result<Vec<Value>> {
        let mut handles = Vec::new();
        for (id, &size) in args.sizes.iter().enumerate() {
            let executor =
                SfmTaskflow::new(runtimes[mode.runtime_id(id)].clone(), 512 * 1024 * 1024)?;
            let (tx, rx) = mpsc::channel::<Instant>();
            gates.push(tx);
            let ready = ready_tx.clone();
            let slots = &slots;
            handles.push(scope.spawn(move || -> Result<Value> {
                let control = SfmTaskControl::new();
                let mut sink = |_: SfmTaskEvent| {};
                let mut task = SfmTaskContext::new(&control, &mut sink).with_taskflow(executor);
                let config = MapperConfig { input: args.input.clone(), output: args.output.join(format!("model-{id}")), database: Some(args.output.join(format!("job-{id}.db"))), max_images: Some(size), random_seed: 1, threads: Some(4), ..Default::default() };
                std::fs::write(args.output.join(format!("config-{id}.txt")), format!("{config:#?}"))?;
                ready.send(())?;
                let epoch = rx.recv_timeout(WAIT)?;
                let submitted_ms = id as f64 * 10.0;
                let due = epoch + Duration::from_millis(id as u64 * 10);
                std::thread::sleep(due.saturating_duration_since(Instant::now()));
                let dispatch_ms = epoch.elapsed().as_secs_f64()*1000.0;
                let permit = if mode == Mode::Fixed2 { Some(slots.acquire(id)?) } else { None };
                let slot_ms = epoch.elapsed().as_secs_f64()*1000.0;
                let result = run_reconstruction_with_task(&config, &mut task);
                let completed_ms = epoch.elapsed().as_secs_f64()*1000.0;
                drop(permit);
                let stages = task.stage_reports();
                let summary = result?;
                std::fs::write(args.output.join(format!("job-{id}.json")), serde_json::to_vec_pretty(&summary)?)?;
                ensure!(stages.len() == 1 && stages[0].granted_threads == 4 && !stages[0].cancelled_or_failed, "unexpected stage grant");
                let stage = &stages[0];
                Ok(json!({"id":id,"size":size,"runtime_id":mode.runtime_id(id),"submitted_ms":submitted_ms,"dispatch_ms":dispatch_ms,"slot_ms":slot_ms,"completed_ms":completed_ms,"dispatch_jitter_ms":dispatch_ms-submitted_ms,"external_slot_queue_ms":slot_ms-dispatch_ms,"total_queue_ms":slot_ms-submitted_ms+stage.queue_ms,"latency_ms":completed_ms-submitted_ms,"stages":stages,"registered":summary.registered_images,"pairs":summary.pairs,"points":summary.points,"models":summary.models}))
            }));
        }
        for _ in &args.sizes {
            ready_rx.recv_timeout(WAIT)?;
        }
        let epoch = Instant::now() + Duration::from_millis(100);
        for gate in &gates {
            gate.send(epoch)?;
        }
        // Join every caller even if one fails, before taking final runtime snapshots.
        let results: Vec<_> = handles
            .into_iter()
            .map(|h| {
                h.join()
                    .map_err(|_| anyhow::anyhow!("caller panicked"))
                    .and_then(|r| r)
            })
            .collect();
        results.into_iter().collect()
    });
    let final_snapshots = runtimes
        .iter()
        .enumerate()
        .map(|(id, r)| {
            let mut s = snapshot(r, Instant::now())?;
            s["runtime_id"] = json!(id);
            Ok(s)
        })
        .collect::<Result<Vec<_>>>()?;
    let clean = final_snapshots.iter().all(drained) && slots.state.lock().unwrap().1 == 0;
    let report = match &jobs {
        Ok(jobs) => {
            json!({"mode":format!("{mode:?}"),"cpu_budget_per_runtime":if mode == Mode::Shared {8} else {4},"memory_budget_per_runtime":2147483648_u64,"workflow_threads":4,"submission_interval_ms":10,"makespan_ms":jobs.iter().map(|j|j["completed_ms"].as_f64().unwrap()).fold(0.0,f64::max),"jobs":jobs,"final_snapshots":final_snapshots,"slots_occupied_final":slots.state.lock().unwrap().1,"clean":clean})
        }
        Err(e) => json!({"error":format!("{e:#}"),"final_snapshots":final_snapshots,"clean":clean}),
    };
    std::fs::write(
        args.output.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    ensure!(clean, "resources not drained");
    jobs?;
    println!("C4 {mode:?}: eight workflows completed; every runtime drained");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mode_dispatch() {
        for mode in [Mode::Independent, Mode::Fixed2] {
            assert_eq!(mode.runtime_count(8), 8);
            assert_eq!(mode.runtime_id(7), 7);
        }
        assert_eq!(Mode::Shared.runtime_count(8), 1);
        assert_eq!(Mode::Shared.runtime_id(7), 0);
        assert!(Args::try_parse_from([
            "example", "--input", "in", "--output", "out", "--mode", "fake"
        ])
        .is_err());
    }
    #[test]
    fn fixed_slots_wait_and_release() {
        let slots = Slots::default();
        let first = slots.acquire(0).unwrap();
        let second = slots.acquire(1).unwrap();
        let (tx, rx) = mpsc::channel();
        std::thread::scope(|s| {
            s.spawn(|| {
                let _third = slots.acquire(2).unwrap();
                tx.send(Instant::now()).unwrap();
            });
            assert!(rx.recv_timeout(Duration::from_millis(20)).is_err());
            let release = Instant::now();
            drop(first);
            assert!(rx.recv_timeout(WAIT).unwrap() >= release);
        });
        drop(second);
        assert_eq!(*slots.state.lock().unwrap(), (3, 0));
    }
    #[test]
    fn cleanup_each_independent_runtime() -> Result<()> {
        for _ in 0..8 {
            let runtime = Runtime::new(RuntimeConfig::default())?;
            assert!(drained(&snapshot(&runtime, Instant::now())?));
        }
        Ok(())
    }
}
