use rayon::prelude::*;
use rustscan_taskflow::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    mpsc, Arc,
};
use std::time::Duration;

const TIMEOUT: Duration = Duration::from_secs(5);
fn config(cpu: usize) -> RuntimeConfig {
    RuntimeConfig {
        budget: Budget {
            cpu_threads: cpu,
            memory_bytes: 1000,
            io_slots: 2,
        },
        gpus: vec![GpuCapacity {
            id: DeviceId(7),
            memory_bytes: 500,
            max_in_flight: 1,
            shared_host_memory: true,
        }],
        max_pending_tasks: 10_000,
        event_capacity: 1000,
        max_bypasses: 4,
    }
}
fn cpu(n: usize) -> ResourceRequest {
    ResourceRequest::cpu(CpuRequest::fixed(n))
}
fn gpu() -> ResourceRequest {
    let mut request = cpu(1);
    request.gpu = Some(GpuRequest {
        device: None,
        working_memory_bytes: 100,
        output_memory_bytes: 0,
    });
    request
}
fn report(run: &RunHandle) -> RunReport {
    run.wait_timeout(TIMEOUT).expect("workflow did not finish")
}
fn one_task(request: ResourceRequest) -> TaskGraph {
    let mut graph = TaskGraph::new();
    graph
        .task("task", vec![TaskVariant::cpu("cpu", request, |_| Ok(()))])
        .unwrap();
    graph
}
fn sync_runtime(runtime: &Runtime) {
    let _ = runtime.snapshot().unwrap();
}

#[test]
fn diamond_typed_artifacts_and_parallel_allowance() {
    let runtime = Runtime::new(config(4)).unwrap();
    let mut graph = TaskGraph::new();
    let a = graph
        .task("a", vec![TaskVariant::cpu("cpu", cpu(1), |_| Ok(3u64))])
        .unwrap();
    let a1 = a.clone();
    let b = graph
        .task(
            "b",
            vec![TaskVariant::cpu(
                "cpu",
                ResourceRequest::cpu(CpuRequest::scalable(1, 3, 4)),
                move |ctx| {
                    let value = *ctx.input(&a1)?;
                    ctx.parallel(|| {
                        assert!(rayon::current_num_threads() <= 3);
                        (0..100u64).into_par_iter().map(|n| n * value).sum::<u64>()
                    })
                },
            )],
        )
        .unwrap();
    let a2 = a.clone();
    let c = graph
        .task(
            "c",
            vec![TaskVariant::cpu("cpu", cpu(1), move |ctx| {
                Ok(*ctx.input(&a2)? + 1)
            })],
        )
        .unwrap();
    let b1 = b.clone();
    let c1 = c.clone();
    let d = graph
        .task(
            "d",
            vec![TaskVariant::cpu("cpu", cpu(1), move |ctx| {
                Ok(*ctx.input(&b1)? + *ctx.input(&c1)?)
            })],
        )
        .unwrap();
    graph.depends_on(b.id(), a.id()).unwrap();
    graph.depends_on(c.id(), a.id()).unwrap();
    graph.depends_on(d.id(), b.id()).unwrap();
    graph.depends_on(d.id(), c.id()).unwrap();
    let run = runtime.submit(graph).unwrap();
    assert!(report(&run).succeeded());
    assert_eq!(*run.output(&d).unwrap(), 14854);
    assert_eq!(runtime.snapshot().unwrap().cpu_threads, 0);
}

#[test]
fn rejects_foreign_handles_cycles_duplicates_and_bad_requests() {
    let mut a = TaskGraph::new();
    let mut b = TaskGraph::new();
    let x = a
        .task("x", vec![TaskVariant::cpu("cpu", cpu(1), |_| Ok(()))])
        .unwrap();
    let y = a
        .task("y", vec![TaskVariant::cpu("cpu", cpu(1), |_| Ok(()))])
        .unwrap();
    let z = b
        .task("z", vec![TaskVariant::cpu("cpu", cpu(1), |_| Ok(()))])
        .unwrap();
    assert_eq!(a.depends_on(x.id(), z.id()), Err(Error::ForeignTask));
    assert_eq!(a.depends_on(x.id(), x.id()), Err(Error::Cycle));
    a.depends_on(y.id(), x.id()).unwrap();
    a.depends_on(y.id(), x.id()).unwrap();
    assert!(a.validate().is_ok());
    a.depends_on(x.id(), y.id()).unwrap();
    assert_eq!(a.validate(), Err(Error::Cycle));
    assert!(a.task::<()>("empty", vec![]).is_err());
    assert!(a
        .task("zero", vec![TaskVariant::cpu("cpu", cpu(0), |_| Ok(()))])
        .is_err());
    assert!(a
        .task(
            "bad-range",
            vec![TaskVariant::cpu(
                "cpu",
                ResourceRequest::cpu(CpuRequest::scalable(4, 2, 8)),
                |_| Ok(())
            )]
        )
        .is_err());
    let mut overflow = cpu(1);
    overflow.working_memory_bytes = u64::MAX;
    overflow.output_memory_bytes = 1;
    assert!(a
        .task(
            "overflow",
            vec![TaskVariant::cpu("cpu", overflow, |_| Ok(()))]
        )
        .is_err());
    let mut exclusive = cpu(1);
    exclusive.exclusive = vec!["db".into(), "db".into()];
    assert!(a
        .task(
            "duplicate",
            vec![TaskVariant::cpu("cpu", exclusive, |_| Ok(()))]
        )
        .is_err());
}

#[test]
fn rejects_unsatisfiable_resources_before_any_work_runs() {
    let runtime = Runtime::new(config(2)).unwrap();
    assert!(matches!(
        runtime.submit(one_task(cpu(3))),
        Err(Error::Unschedulable(_))
    ));
    let mut request = gpu();
    request.gpu.as_mut().unwrap().device = Some(DeviceId(99));
    assert!(matches!(
        runtime.submit(one_task(request)),
        Err(Error::Unschedulable(_))
    ));
    let mut request = gpu();
    request.working_memory_bytes = 950;
    assert!(matches!(
        runtime.submit(one_task(request)),
        Err(Error::Unschedulable(_))
    ));
    let mut request = cpu(1);
    request.io_slots = 3;
    assert!(matches!(
        runtime.submit(one_task(request)),
        Err(Error::Unschedulable(_))
    ));
    assert!(report(&runtime.submit(TaskGraph::new()).unwrap()).succeeded());
}

#[test]
fn rejects_dependency_memory_deadlock_before_execution() {
    let runtime = Runtime::new(config(1)).unwrap();
    let mut graph = TaskGraph::new();
    let mut producer_request = cpu(1);
    producer_request.output_memory_bytes = 900;
    let producer = graph
        .task(
            "producer",
            vec![TaskVariant::cpu("cpu", producer_request, |_| Ok(42u64))],
        )
        .unwrap();
    let producer_id = producer.id();
    let input = producer.clone();
    let mut consumer_request = cpu(1);
    consumer_request.working_memory_bytes = 200;
    let consumer = graph
        .task(
            "consumer",
            vec![TaskVariant::cpu("cpu", consumer_request, move |ctx| {
                let value = ctx.input(&input)?;
                drop(value);
                Ok(())
            })],
        )
        .unwrap();
    graph.depends_on(consumer.id(), producer_id).unwrap();

    assert!(matches!(
        runtime.submit(graph),
        Err(Error::Unschedulable(name)) if name == "consumer"
    ));
    assert_eq!(runtime.snapshot().unwrap().pending_tasks, 0);
}

#[test]
fn asynchronous_gpu_releases_cpu_but_not_gpu_or_memory() {
    let runtime = Runtime::new(config(1)).unwrap();
    let (tx, rx) = mpsc::channel();
    let mut graph = TaskGraph::new();
    let output = graph
        .task(
            "gpu",
            vec![TaskVariant::asynchronous("wgpu", gpu(), move |_, done| {
                tx.send(done).unwrap();
            })],
        )
        .unwrap();
    let next = graph
        .task("next", vec![TaskVariant::cpu("cpu", cpu(1), |_| Ok(()))])
        .unwrap();
    graph.depends_on(next.id(), output.id()).unwrap();
    let run = runtime.submit(graph).unwrap();
    let done: Completion<u32> = rx.recv_timeout(TIMEOUT).unwrap();
    // A CPU workflow can complete while the GPU workflow remains in flight.
    assert!(report(&runtime.submit(one_task(cpu(1))).unwrap()).succeeded());
    let snapshot = runtime.snapshot().unwrap();
    assert_eq!(snapshot.cpu_threads, 0);
    assert_eq!(snapshot.memory_bytes, 100);
    assert_eq!(snapshot.gpus[&DeviceId(7)].in_flight, 1);
    assert!(run.wait_timeout(Duration::from_millis(10)).is_none());
    done.complete(Ok(42));
    assert!(report(&run).succeeded());
    assert_eq!(*run.output(&output).unwrap(), 42);
    let snapshot = runtime.snapshot().unwrap();
    assert_eq!(snapshot.memory_bytes, 0);
    assert_eq!(snapshot.gpus[&DeviceId(7)].in_flight, 0);
}

#[test]
fn cpu_fallback_when_gpu_busy_and_gpu_preference_when_available() {
    let runtime = Runtime::new(config(2)).unwrap();
    let (tx, rx) = mpsc::channel();
    let mut hold = TaskGraph::new();
    hold.task(
        "hold",
        vec![TaskVariant::<()>::asynchronous(
            "gpu",
            gpu(),
            move |_, done| {
                tx.send(done).unwrap();
            },
        )],
    )
    .unwrap();
    let held = runtime.submit(hold).unwrap();
    let done = rx.recv_timeout(TIMEOUT).unwrap();
    let build = || {
        let mut graph = TaskGraph::new();
        let output = graph
            .task(
                "choose",
                vec![
                    TaskVariant::cpu("gpu", gpu(), |ctx| {
                        Ok(ctx.grant().gpu.as_ref().unwrap().device.0)
                    }),
                    TaskVariant::cpu("cpu", cpu(1), |_| Ok(99u32)),
                ],
            )
            .unwrap();
        (graph, output)
    };
    let (graph, output) = build();
    let run = runtime.submit(graph).unwrap();
    assert!(report(&run).succeeded());
    assert_eq!(*run.output(&output).unwrap(), 99);
    done.complete(Ok(()));
    report(&held);
    let (graph, output) = build();
    let run = runtime.submit(graph).unwrap();
    assert!(report(&run).succeeded());
    assert_eq!(*run.output(&output).unwrap(), 7);
}

#[test]
fn cancellation_keeps_external_reservation_until_completion_and_discards_output() {
    let runtime = Runtime::new(config(2)).unwrap();
    let (tx, rx) = mpsc::channel();
    let mut graph = TaskGraph::new();
    let out = graph
        .task(
            "gpu",
            vec![TaskVariant::asynchronous("gpu", gpu(), move |ctx, done| {
                tx.send((done, ctx.cancellation())).unwrap();
            })],
        )
        .unwrap();
    let next = graph
        .task(
            "after",
            vec![TaskVariant::<()>::cpu("cpu", cpu(1), |_| {
                panic!("cancelled downstream ran")
            })],
        )
        .unwrap();
    graph.depends_on(next.id(), out.id()).unwrap();
    let run = runtime.submit(graph).unwrap();
    let (done, cancel): (Completion<u32>, _) = rx.recv_timeout(TIMEOUT).unwrap();
    run.cancel();
    sync_runtime(&runtime);
    assert!(cancel.is_cancelled());
    assert_eq!(runtime.snapshot().unwrap().gpus[&DeviceId(7)].in_flight, 1);
    assert!(run.wait_timeout(Duration::from_millis(10)).is_none());
    done.complete(Ok(12));
    assert!(report(&run)
        .tasks
        .iter()
        .all(|t| t.status == TaskStatus::Cancelled));
    assert!(run.output(&out).is_err());
    assert_eq!(runtime.snapshot().unwrap().memory_bytes, 0);
}

#[test]
fn completion_before_submit_returns_does_not_release_resources_early() {
    let runtime = Runtime::new(config(2)).unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let mut graph = TaskGraph::new();
    graph
        .task(
            "early",
            vec![TaskVariant::asynchronous("gpu", gpu(), move |_, done| {
                done.complete(Ok(()));
                started_tx.send(()).unwrap();
                release_rx.recv_timeout(TIMEOUT).unwrap();
            })],
        )
        .unwrap();
    let run = runtime.submit(graph).unwrap();
    started_rx.recv_timeout(TIMEOUT).unwrap();
    let snapshot = runtime.snapshot().unwrap();
    assert_eq!(snapshot.cpu_threads, 1);
    assert_eq!(snapshot.gpus[&DeviceId(7)].in_flight, 1);
    assert!(run.wait_timeout(Duration::from_millis(10)).is_none());
    release_tx.send(()).unwrap();
    assert!(report(&run).succeeded());
}

#[test]
fn async_submission_panic_waits_for_escaped_completion() {
    let runtime = Runtime::new(config(1)).unwrap();
    let (tx, rx) = mpsc::channel();
    let mut graph = TaskGraph::new();
    graph
        .task(
            "panic-submit",
            vec![TaskVariant::<()>::asynchronous(
                "gpu",
                gpu(),
                move |_, done| {
                    tx.send(done).unwrap();
                    panic!("submission panic after starting GPU");
                },
            )],
        )
        .unwrap();
    let run = runtime.submit(graph).unwrap();
    let done = rx.recv_timeout(TIMEOUT).unwrap();
    assert!(run.wait_timeout(Duration::from_millis(10)).is_none());
    assert_eq!(runtime.snapshot().unwrap().gpus[&DeviceId(7)].in_flight, 1);
    done.complete(Ok(()));
    assert!(matches!(
        report(&run).tasks[0].error,
        Some(TaskError::Panicked(_))
    ));
}

#[test]
fn failures_skip_only_descendants_and_panic_is_reported() {
    let runtime = Runtime::new(config(4)).unwrap();
    for panic in [false, true] {
        let mut graph = TaskGraph::new();
        let a = graph
            .task(
                "bad",
                vec![TaskVariant::<()>::cpu("cpu", cpu(1), move |_| {
                    if panic {
                        panic!("expected panic");
                    }
                    Err(TaskError::Failed("algorithm error".into()))
                })],
            )
            .unwrap();
        let b = graph
            .task(
                "child",
                vec![TaskVariant::<()>::cpu("cpu", cpu(1), |_| {
                    panic!("child ran")
                })],
            )
            .unwrap();
        let c = graph
            .task(
                "independent",
                vec![TaskVariant::cpu("cpu", cpu(1), |_| Ok(()))],
            )
            .unwrap();
        let d = graph
            .task(
                "join",
                vec![TaskVariant::<()>::cpu("cpu", cpu(1), |_| {
                    panic!("join ran")
                })],
            )
            .unwrap();
        graph.depends_on(b.id(), a.id()).unwrap();
        graph.depends_on(d.id(), b.id()).unwrap();
        graph.depends_on(d.id(), c.id()).unwrap();
        let run = runtime.submit(graph).unwrap();
        let result = report(&run);
        assert_eq!(result.tasks[0].status, TaskStatus::Failed);
        assert_eq!(result.tasks[1].status, TaskStatus::DependencyFailed);
        assert_eq!(result.tasks[2].status, TaskStatus::Succeeded);
        assert_eq!(result.tasks[3].status, TaskStatus::DependencyFailed);
        assert_eq!(runtime.snapshot().unwrap().cpu_threads, 0);
    }
}

#[test]
fn dropped_completion_fails_and_releases_resources() {
    let runtime = Runtime::new(config(1)).unwrap();
    let mut graph = TaskGraph::new();
    graph
        .task(
            "lost",
            vec![TaskVariant::<()>::asynchronous("gpu", gpu(), |_, _done| {})],
        )
        .unwrap();
    let result = report(&runtime.submit(graph).unwrap());
    assert_eq!(result.tasks[0].error, Some(TaskError::CompletionDropped));
    assert_eq!(runtime.snapshot().unwrap().memory_bytes, 0);
}

#[test]
fn artifact_memory_survives_workflow_completion_and_cloned_readers() {
    let runtime = Runtime::new(config(2)).unwrap();
    let mut request = gpu();
    request.output_memory_bytes = 20;
    request.gpu.as_mut().unwrap().output_memory_bytes = 80;
    let mut graph = TaskGraph::new();
    let out = graph
        .task(
            "output",
            vec![TaskVariant::cpu("gpu", request, |_| Ok(vec![1, 2]))],
        )
        .unwrap();
    let run = runtime.submit(graph).unwrap();
    assert!(report(&run).succeeded());
    let artifact = run.output(&out).unwrap();
    let reader = artifact.clone();
    drop(out);
    drop(run);
    drop(artifact);
    let snapshot = runtime.snapshot().unwrap();
    assert_eq!(snapshot.memory_bytes, 100);
    assert_eq!(snapshot.gpus[&DeviceId(7)].memory_bytes, 80);
    assert_eq!(snapshot.gpus[&DeviceId(7)].in_flight, 0);
    drop(reader);
    let snapshot = runtime.snapshot().unwrap();
    assert_eq!(snapshot.memory_bytes, 0);
    assert_eq!(snapshot.gpus[&DeviceId(7)].memory_bytes, 0);
}

#[test]
fn output_reservations_apply_backpressure_until_handles_are_dropped() {
    let runtime = Runtime::new(config(2)).unwrap();
    let mut request = cpu(1);
    request.output_memory_bytes = 900;
    let mut graph = TaskGraph::new();
    let output = graph
        .task("large", vec![TaskVariant::cpu("cpu", request, |_| Ok(()))])
        .unwrap();
    let first = runtime.submit(graph).unwrap();
    report(&first);
    let mut request = cpu(1);
    request.working_memory_bytes = 200;
    let second = runtime.submit(one_task(request)).unwrap();
    assert!(second.wait_timeout(Duration::from_millis(10)).is_none());
    drop(output);
    assert!(report(&second).succeeded());
}

#[test]
fn undeclared_and_foreign_artifact_access_is_rejected() {
    let runtime = Runtime::new(config(2)).unwrap();
    let mut graph = TaskGraph::new();
    let a = graph
        .task("a", vec![TaskVariant::cpu("cpu", cpu(1), |_| Ok(1))])
        .unwrap();
    let copy = a.clone();
    graph
        .task(
            "bad-read",
            vec![TaskVariant::cpu("cpu", cpu(1), move |ctx| {
                assert!(matches!(ctx.input(&copy), Err(TaskError::InvalidArtifact)));
                Ok(())
            })],
        )
        .unwrap();
    let first = runtime.submit(graph).unwrap();
    assert!(report(&first).succeeded());
    let second = runtime.submit(TaskGraph::new()).unwrap();
    report(&second);
    assert!(matches!(second.output(&a), Err(TaskError::InvalidArtifact)));
}

#[test]
fn isolated_rayons_use_granted_threads_across_multiple_graphs() {
    let runtime = Runtime::new(config(4)).unwrap();
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let mut runs = Vec::new();
    for _ in 0..6 {
        let mut graph = TaskGraph::new();
        for _ in 0..10 {
            let active = active.clone();
            let peak = peak.clone();
            graph
                .task(
                    "parallel",
                    vec![TaskVariant::cpu(
                        "cpu",
                        ResourceRequest::cpu(CpuRequest::scalable(1, 2, 4)),
                        move |ctx| {
                            let allowance = ctx.grant().cpu_threads;
                            ctx.parallel(|| {
                                assert_eq!(rayon::current_num_threads(), allowance);
                                (0..20).into_par_iter().for_each(|_| {
                                    let count = active.fetch_add(1, Ordering::SeqCst) + 1;
                                    peak.fetch_max(count, Ordering::SeqCst);
                                    std::thread::sleep(Duration::from_micros(50));
                                    active.fetch_sub(1, Ordering::SeqCst);
                                });
                            })?;
                            Ok(())
                        },
                    )],
                )
                .unwrap();
        }
        runs.push(runtime.submit(graph).unwrap());
    }
    for run in runs {
        assert!(report(&run).succeeded());
    }
    assert!(peak.load(Ordering::SeqCst) <= 4);
    assert_eq!(active.load(Ordering::SeqCst), 0);
}

#[test]
fn exclusive_resources_are_global_across_workflows() {
    let runtime = Runtime::new(config(4)).unwrap();
    let active = Arc::new(AtomicUsize::new(0));
    let mut runs = Vec::new();
    for _ in 0..5 {
        let mut graph = TaskGraph::new();
        for _ in 0..5 {
            let active = active.clone();
            let mut request = cpu(1);
            request.exclusive = vec!["shared-db-writer".into()];
            request.io_slots = 1;
            graph
                .task(
                    "write",
                    vec![TaskVariant::cpu("cpu", request, move |_| {
                        assert_eq!(active.fetch_add(1, Ordering::SeqCst), 0);
                        std::thread::sleep(Duration::from_micros(100));
                        assert_eq!(active.fetch_sub(1, Ordering::SeqCst), 1);
                        Ok(())
                    })],
                )
                .unwrap();
        }
        runs.push(runtime.submit(graph).unwrap());
    }
    for run in runs {
        assert!(report(&run).succeeded());
    }
    assert!(runtime.snapshot().unwrap().exclusive.is_empty());
}

#[test]
fn budget_reduction_never_revokes_running_work_and_recovery_resumes_queue() {
    let runtime = Runtime::new(config(3)).unwrap();
    let (tx, rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let mut graph = TaskGraph::new();
    graph
        .task(
            "running",
            vec![TaskVariant::cpu("cpu", cpu(3), move |_| {
                tx.send(()).unwrap();
                release_rx.recv_timeout(TIMEOUT).unwrap();
                Ok(())
            })],
        )
        .unwrap();
    let first = runtime.submit(graph).unwrap();
    rx.recv_timeout(TIMEOUT).unwrap();
    let mut budget = config(3).budget;
    budget.cpu_threads = 0;
    runtime.set_budget(budget).unwrap();
    assert_eq!(runtime.snapshot().unwrap().cpu_threads, 3);
    let second = runtime.submit(one_task(cpu(1))).unwrap();
    release_tx.send(()).unwrap();
    assert!(report(&first).succeeded());
    assert!(second.wait_timeout(Duration::from_millis(10)).is_none());
    runtime.set_budget(config(3).budget).unwrap();
    assert!(report(&second).succeeded());
    assert!(runtime.set_budget(config(4).budget).is_err());
}

#[test]
fn queue_capacity_and_nonblocking_telemetry() {
    let mut conf = config(1);
    conf.max_pending_tasks = 2;
    conf.event_capacity = 1;
    let runtime = Runtime::new(conf).unwrap();
    let mut budget = config(1).budget;
    budget.cpu_threads = 0;
    runtime.set_budget(budget).unwrap();
    let a = runtime.submit(one_task(cpu(1))).unwrap();
    let b = runtime.submit(one_task(cpu(1))).unwrap();
    assert!(matches!(
        runtime.submit(one_task(cpu(1))),
        Err(Error::QueueFull)
    ));
    a.cancel();
    assert_eq!(report(&a).tasks[0].status, TaskStatus::Cancelled);
    runtime.set_budget(config(1).budget).unwrap();
    assert!(report(&b).succeeded());
    assert!(report(&b).dropped_events > 0);
    assert!(b.try_event().is_some());
}

#[test]
fn bounded_overtaking_prevents_large_cpu_task_starvation() {
    let mut conf = config(3);
    conf.max_bypasses = 2;
    let runtime = Runtime::new(conf).unwrap();
    let (hold_tx, hold_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let mut hold = TaskGraph::new();
    hold.task(
        "hold",
        vec![TaskVariant::cpu("cpu", cpu(1), move |_| {
            hold_tx.send(()).unwrap();
            release_rx.recv_timeout(TIMEOUT).unwrap();
            Ok(())
        })],
    )
    .unwrap();
    let hold_run = runtime.submit(hold).unwrap();
    hold_rx.recv_timeout(TIMEOUT).unwrap();
    let large = runtime.submit(one_task(cpu(3))).unwrap();
    let (small_tx, small_rx) = mpsc::channel();
    let mut small = TaskGraph::new();
    for _ in 0..20 {
        let tx = small_tx.clone();
        small
            .task(
                "small",
                vec![TaskVariant::cpu("cpu", cpu(1), move |_| {
                    tx.send(()).unwrap();
                    Ok(())
                })],
            )
            .unwrap();
    }
    let small_run = runtime.submit(small).unwrap();
    small_rx.recv_timeout(TIMEOUT).unwrap();
    small_rx.recv_timeout(TIMEOUT).unwrap();
    assert!(small_rx.recv_timeout(Duration::from_millis(30)).is_err());
    release_tx.send(()).unwrap();
    assert!(report(&large).succeeded());
    assert!(report(&small_run).succeeded());
    report(&hold_run);
}

#[test]
fn reports_dependency_and_resource_wait_separately() {
    let runtime = Runtime::new(config(1)).unwrap();
    let (hold_started, hold_started_rx) = mpsc::channel();
    let (release_hold, release_hold_rx) = mpsc::channel();
    let mut hold = TaskGraph::new();
    hold.task(
        "hold",
        vec![TaskVariant::cpu("cpu", cpu(1), move |_| {
            hold_started.send(()).unwrap();
            release_hold_rx.recv_timeout(TIMEOUT).unwrap();
            Ok(())
        })],
    )
    .unwrap();
    let hold_run = runtime.submit(hold).unwrap();
    hold_started_rx.recv_timeout(TIMEOUT).unwrap();

    let mut graph = TaskGraph::new();
    let producer = graph
        .task(
            "producer",
            vec![TaskVariant::cpu("cpu", cpu(1), |_| {
                std::thread::sleep(Duration::from_millis(25));
                Ok(())
            })],
        )
        .unwrap();
    let consumer = graph
        .task(
            "consumer",
            vec![TaskVariant::cpu("cpu", cpu(1), |_| Ok(()))],
        )
        .unwrap();
    graph.depends_on(consumer.id(), producer.id()).unwrap();
    let run = runtime.submit(graph).unwrap();

    std::thread::sleep(Duration::from_millis(30));
    release_hold.send(()).unwrap();
    let result = report(&run);
    report(&hold_run);

    let producer_report = &result.tasks[producer.id().index()];
    let consumer_report = &result.tasks[consumer.id().index()];
    let producer_execution = producer_report.execution_time.unwrap();
    assert!(
        producer_report.resource_wait_time.unwrap() >= Duration::from_millis(15),
        "producer resource wait was {:?}",
        producer_report.resource_wait_time
    );
    assert_eq!(producer_report.dependency_wait_time, Some(Duration::ZERO));
    assert!(consumer_report.dependency_wait_time.unwrap() >= producer_execution);
    assert!(consumer_report.queue_time.unwrap() >= consumer_report.dependency_wait_time.unwrap());
    assert!(
        consumer_report.resource_wait_time.unwrap() < consumer_report.dependency_wait_time.unwrap()
    );
}

#[test]
fn memory_waiter_does_not_block_consumer_that_releases_its_input() {
    let mut conf = config(1);
    conf.max_bypasses = 2;
    let runtime = Runtime::new(conf).unwrap();
    let mut graph = TaskGraph::new();
    let mut allocation = cpu(1);
    allocation.output_memory_bytes = 900;
    let producer = graph
        .task(
            "producer",
            vec![TaskVariant::cpu("cpu", allocation, |_| Ok(42))],
        )
        .unwrap();
    let producer_id = producer.id();
    let mut waiting = cpu(1);
    waiting.working_memory_bytes = 200;
    graph
        .task(
            "memory waiter",
            vec![TaskVariant::cpu("cpu", waiting, |_| Ok(()))],
        )
        .unwrap();
    for _ in 0..3 {
        graph
            .task("bypass", vec![TaskVariant::cpu("cpu", cpu(1), |_| Ok(()))])
            .unwrap();
    }
    let consumer = graph
        .task(
            "consumer",
            vec![TaskVariant::cpu("cpu", cpu(1), move |ctx| {
                assert_eq!(*ctx.input(&producer)?, 42);
                drop(producer);
                Ok(())
            })],
        )
        .unwrap();
    graph.depends_on(consumer.id(), producer_id).unwrap();
    let run = runtime.submit(graph).unwrap();
    assert!(report(&run).succeeded());
    assert_eq!(runtime.snapshot().unwrap().memory_bytes, 0);
}

#[test]
fn any_gpu_skips_busy_device_without_partially_acquiring_resources() {
    let mut conf = config(2);
    conf.gpus.push(GpuCapacity {
        id: DeviceId(8),
        memory_bytes: 500,
        max_in_flight: 1,
        shared_host_memory: false,
    });
    let runtime = Runtime::new(conf).unwrap();
    let (tx, rx) = mpsc::channel();
    let mut graph = TaskGraph::new();
    graph
        .task(
            "hold first GPU",
            vec![TaskVariant::asynchronous("gpu", gpu(), move |_, done| {
                tx.send(done).unwrap();
            })],
        )
        .unwrap();
    let hold = runtime.submit(graph).unwrap();
    let completion = rx.recv_timeout(TIMEOUT).unwrap();
    let mut request = gpu();
    request.io_slots = 2;
    request.exclusive.push("shared-output".into());
    let mut graph = TaskGraph::new();
    graph
        .task(
            "second GPU",
            vec![TaskVariant::cpu("gpu", request, |ctx| {
                assert_eq!(ctx.grant().gpu.as_ref().unwrap().device, DeviceId(8));
                Ok(())
            })],
        )
        .unwrap();
    let second = runtime.submit(graph).unwrap();
    let result = second.wait_timeout(TIMEOUT);
    completion.complete(Ok(()));
    assert!(result.expect("second device was not selected").succeeded());
    assert!(report(&hold).succeeded());
    let snapshot = runtime.snapshot().unwrap();
    assert_eq!(snapshot.cpu_threads, 0);
    assert_eq!(snapshot.memory_bytes, 0);
    assert_eq!(snapshot.io_slots, 0);
    assert!(snapshot.exclusive.is_empty());
    assert!(snapshot
        .gpus
        .values()
        .all(|gpu| gpu.memory_bytes == 0 && gpu.in_flight == 0));
}

#[test]
fn concurrent_submitters_share_one_runtime() {
    let runtime = Arc::new(Runtime::new(config(4)).unwrap());
    let count = Arc::new(AtomicUsize::new(0));
    std::thread::scope(|scope| {
        for _ in 0..8 {
            let runtime = runtime.clone();
            let count = count.clone();
            scope.spawn(move || {
                for _ in 0..20 {
                    let count = count.clone();
                    let mut graph = TaskGraph::new();
                    graph
                        .task(
                            "concurrent",
                            vec![TaskVariant::cpu("cpu", cpu(1), move |_| {
                                count.fetch_add(1, Ordering::Relaxed);
                                Ok(())
                            })],
                        )
                        .unwrap();
                    assert!(report(&runtime.submit(graph).unwrap()).succeeded());
                }
            });
        }
    });
    assert_eq!(count.load(Ordering::Relaxed), 160);
    assert_eq!(runtime.snapshot().unwrap().pending_tasks, 0);
}

#[test]
fn invalid_runtime_capacities_are_rejected() {
    let mut conf = config(1);
    conf.gpus.push(conf.gpus[0].clone());
    assert!(matches!(Runtime::new(conf), Err(Error::Invalid(_))));
    let mut conf = config(1);
    conf.gpus[0].max_in_flight = 0;
    assert!(matches!(Runtime::new(conf), Err(Error::Invalid(_))));
    let mut conf = config(1);
    conf.budget.cpu_threads = 0;
    assert!(matches!(Runtime::new(conf), Err(Error::Invalid(_))));
}

#[test]
fn drop_cancels_queued_graph_and_runs_closure_destructors_once() {
    struct Count(Arc<AtomicUsize>);
    impl Drop for Count {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let runtime = Runtime::new(config(1)).unwrap();
    let mut budget = config(1).budget;
    budget.cpu_threads = 0;
    runtime.set_budget(budget).unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let mut graph = TaskGraph::new();
    let guard = Count(count.clone());
    graph
        .task(
            "never",
            vec![TaskVariant::cpu("cpu", cpu(1), move |_| {
                drop(guard);
                Ok(())
            })],
        )
        .unwrap();
    let run = runtime.submit(graph).unwrap();
    drop(runtime);
    assert_eq!(report(&run).tasks[0].status, TaskStatus::Cancelled);
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn pressure_policy_has_hysteresis_and_respects_ceiling() {
    let runtime = Runtime::new(config(8)).unwrap();
    let mut snapshot = runtime.snapshot().unwrap();
    let mut policy = AdaptivePolicy::new(config(8).budget, 100);
    let high = PressureSample {
        external_cpu_pressure: 0.75,
        available_memory_bytes: 150,
    };
    let budget = policy.update(&snapshot, high);
    assert_eq!(budget.cpu_threads, 7);
    assert_eq!(budget.memory_bytes, 50);
    snapshot.budget = budget;
    for _ in 0..3 {
        snapshot.budget = policy.update(
            &snapshot,
            PressureSample {
                external_cpu_pressure: 0.0,
                available_memory_bytes: u64::MAX,
            },
        );
        assert_eq!(snapshot.budget.cpu_threads, 7);
    }
    snapshot.budget = policy.update(
        &snapshot,
        PressureSample {
            external_cpu_pressure: 0.0,
            available_memory_bytes: u64::MAX,
        },
    );
    assert_eq!(snapshot.budget.cpu_threads, 8);
    assert_eq!(snapshot.budget.memory_bytes, 1000);
    let invalid = policy.update(
        &snapshot,
        PressureSample {
            external_cpu_pressure: f32::NAN,
            available_memory_bytes: 0,
        },
    );
    assert_eq!(invalid.cpu_threads, 7);
    assert_eq!(invalid.memory_bytes, 0);
}

#[test]
fn release_stress_ten_thousand_tasks() {
    let runtime = Runtime::new(config(4)).unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let mut graph = TaskGraph::new();
    let mut previous = None;
    for index in 0..10_000 {
        let count = count.clone();
        let task = graph
            .task(
                "stress",
                vec![TaskVariant::cpu("cpu", cpu(1), move |_| {
                    count.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                })],
            )
            .unwrap();
        if index % 20 != 0 {
            graph.depends_on(task.id(), previous.unwrap()).unwrap();
        }
        previous = Some(task.id());
    }
    let run = runtime.submit(graph).unwrap();
    assert!(report(&run).succeeded());
    assert_eq!(count.load(Ordering::Relaxed), 10_000);
    assert_eq!(runtime.snapshot().unwrap().pending_tasks, 0);
}

#[cfg(feature = "system-monitor")]
#[test]
fn system_monitor_can_sample_and_stop() {
    let runtime = Runtime::new(config(2)).unwrap();
    assert!(SystemMonitor::start(&runtime, Duration::from_millis(1), 0).is_err());
    let monitor = SystemMonitor::start(&runtime, Duration::from_millis(250), 0).unwrap();
    std::thread::sleep(Duration::from_millis(550));
    drop(monitor);
    assert!(report(&runtime.submit(one_task(cpu(1))).unwrap()).succeeded());
}
