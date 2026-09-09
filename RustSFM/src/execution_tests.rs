use super::*;
use crate::{SfmTaskContext, SfmTaskStop};
use rustscan_taskflow::Budget;
use std::cell::Cell;

fn executor(cpu: usize) -> SfmTaskflow {
    SfmTaskflow::new(
        Arc::new(
            Runtime::new(RuntimeConfig {
                budget: Budget {
                    cpu_threads: cpu,
                    memory_bytes: 1024,
                    io_slots: 2,
                },
                ..Default::default()
            })
            .unwrap(),
        ),
        128,
    )
    .unwrap()
}
fn released(executor: &SfmTaskflow) {
    let state = executor.runtime.snapshot().unwrap();
    assert_eq!(
        (state.cpu_threads, state.memory_bytes, state.pending_tasks),
        (0, 0, 0)
    );
    assert_eq!(state.io_slots, 0);
    assert!(state
        .gpus
        .values()
        .all(|gpu| gpu.memory_bytes == 0 && gpu.in_flight == 0));
    assert!(state.exclusive.is_empty());
    assert!(active_threads().is_none());
    assert!(active_control().is_none());
    assert!(active_memory_bytes().is_none());
    assert!(ACTIVE.with(|active| active.borrow().is_none()));
}

#[test]
fn shared_runtime_and_borrowed_callbacks_with_nested_bounded_cpu() -> Result<()> {
    assert!(Arc::ptr_eq(
        SfmTaskflow::shared()?.runtime(),
        SfmTaskflow::shared()?.runtime()
    ));
    let executor = executor(2);
    let control = SfmTaskControl::new();
    let value = Cell::new(0);
    let caller = std::thread::current().id();
    executor.run("outer", false, 8, &control, || {
        assert_eq!(active_threads(), Some(2));
        assert_eq!(parallel(rayon::current_num_threads), 2);
        executor.run("nested", false, 8, &control, || {
            assert_eq!(std::thread::current().id(), caller);
            value.set(1);
            assert_eq!(executor.runtime.snapshot()?.pending_tasks, 1);
            let request = ResourceRequest::cpu(CpuRequest::fixed(1));
            admit(
                &executor.runtime,
                "nested BA",
                request,
                &control,
                |grant, _| {
                    assert_eq!(grant.cpu_threads, 1);
                    Ok(())
                },
            )
        })
    })?;
    assert_eq!(value.get(), 1);
    released(&executor);
    Ok(())
}

#[test]
fn queued_cancel_is_typed_and_does_not_execute_stage() -> Result<()> {
    let executor = executor(1);
    let mut budget = executor.runtime.snapshot()?.budget;
    budget.cpu_threads = 0;
    executor.runtime.set_budget(budget)?;
    let control = SfmTaskControl::new();
    std::thread::scope(|scope| -> Result<()> {
        let worker = scope.spawn(|| {
            executor.run("queued", false, 1, &control, || -> Result<()> {
                panic!("cancelled stage must not run")
            })
        });
        let deadline = Instant::now() + Duration::from_secs(5);
        while executor.runtime.snapshot()?.pending_tasks == 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        let queued = executor.runtime.snapshot()?.pending_tasks;
        control.request_cancel();
        let error = worker.join().unwrap().unwrap_err();
        assert_eq!(queued, 1);
        assert_eq!(
            error.downcast_ref::<SfmTaskStop>(),
            Some(&SfmTaskStop::Cancelled)
        );
        Ok(())
    })?;
    released(&executor);
    Ok(())
}

#[test]
fn stage_panic_and_error_release_and_restore_scope() {
    let executor = executor(1);
    let control = SfmTaskControl::new();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        executor.run::<()>("panic", true, 1, &control, || panic!("test unwind"))
    }))
    .is_err());
    released(&executor);
    assert!(executor
        .run::<()>("error", true, 1, &control, || bail!("test failure"))
        .is_err());
    released(&executor);
}

#[test]
fn gpu_admission_remains_exclusive_until_work_finishes() -> Result<()> {
    let executor = executor(2);
    let control = SfmTaskControl::new();
    let (started, entered) = mpsc::channel();
    let (finish, finished) = mpsc::channel();
    std::thread::scope(|scope| -> Result<()> {
        let executor_ref = &executor;
        let control_ref = &control;
        let first = scope.spawn(move || {
            executor_ref.run("gpu first", true, 1, control_ref, || {
                started.send(())?;
                finished.recv_timeout(Duration::from_secs(5))?;
                Ok(())
            })
        });
        entered.recv_timeout(Duration::from_secs(5))?;
        let (started, entered) = mpsc::channel();
        let second = scope.spawn(move || {
            executor_ref.run("gpu second", true, 1, control_ref, || {
                started.send(()).unwrap();
                Ok(())
            })
        });
        let blocked = entered.recv_timeout(Duration::from_millis(30)).is_err();
        finish.send(())?;
        first.join().unwrap()?;
        second.join().unwrap()?;
        assert!(blocked);
        entered.recv_timeout(Duration::from_secs(5))?;
        Ok(())
    })?;
    released(&executor);
    Ok(())
}

#[test]
fn matching_entry_queues_before_database_open_and_respects_pause() -> Result<()> {
    let executor = executor(1);
    let control = SfmTaskControl::new();
    control.request_pause();
    let mut sink = |_| {};
    let mut task = SfmTaskContext::new(&control, &mut sink).with_taskflow(executor.clone());
    let error = crate::match_features_to_database_with_task(
        std::path::Path::new("does-not-exist.db"),
        &Default::default(),
        &mut task,
    )
    .unwrap_err();
    assert_eq!(
        error.downcast_ref::<SfmTaskStop>(),
        Some(&SfmTaskStop::Paused)
    );
    released(&executor);
    Ok(())
}

#[test]
fn stage_dag_orders_borrowed_work_and_stops_on_failure() -> Result<()> {
    let executor = executor(1);
    let control = SfmTaskControl::new();
    let stages = [("cpu", false, 4), ("gpu", true, 1)];
    let mut visited = Vec::new();
    executor.sequence(&stages, &control, |index| {
        assert_eq!(active_threads(), Some(1));
        assert_eq!(
            executor.runtime.snapshot()?.exclusive.is_empty(),
            index == 0
        );
        visited.push(index);
        Ok(())
    })?;
    assert_eq!(visited, [0, 1]);
    released(&executor);
    visited.clear();
    assert!(executor
        .sequence(&stages, &control, |index| {
            visited.push(index);
            bail!("predecessor failed")
        })
        .is_err());
    assert_eq!(visited, [0]);
    released(&executor);
    Ok(())
}

#[test]
fn stage_dag_cancel_and_panic_drain_before_return() -> Result<()> {
    let executor = executor(1);
    let stages = [("first", false, 1), ("must not run", false, 1)];
    let control = SfmTaskControl::new();
    let error = executor
        .sequence(&stages, &control, |index| {
            assert_eq!(index, 0);
            control.request_cancel();
            Ok(())
        })
        .unwrap_err();
    assert_eq!(
        error.downcast_ref::<SfmTaskStop>(),
        Some(&SfmTaskStop::Cancelled)
    );
    released(&executor);

    let control = SfmTaskControl::new();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        executor.sequence(&stages, &control, |index| {
            assert_eq!(index, 0);
            panic!("DAG callback panic")
        })
    }))
    .is_err());
    released(&executor);
    Ok(())
}

#[test]
fn component_adapters_share_stage_budget_and_reject_mixed_runtimes() -> Result<()> {
    let execution = executor(1);
    let features = crate::CpuFeatureTaskflow::new(execution.runtime.clone(), 128, 4)?;
    let ba = crate::CeresBaTaskflow::new(execution.runtime.clone(), 1, 256)?;
    let control = SfmTaskControl::new();
    let mut sink = |_| {};
    let mut task = SfmTaskContext::new(&control, &mut sink)
        .with_cpu_feature_taskflow(&features)
        .with_ceres_ba_taskflow(&ba);
    task.execute("composed", false, 4, |task| {
        assert_eq!(execution.runtime.snapshot()?.memory_bytes, 256);
        assert_eq!(task.feature_batch_threads(), 1);
        assert!(task.cpu_feature_taskflow()?.is_none());
        Ok(())
    })?;
    released(&execution);
    let foreign = executor(1);
    let mut task = task.with_taskflow(foreign);
    assert!(task.execute("invalid mix", false, 1, |_| Ok(())).is_err());
    released(&execution);
    Ok(())
}

#[test]
fn standalone_image_dag_rejects_mixed_runtimes_before_database_open() -> Result<()> {
    let execution = executor(1);
    let foreign = executor(1);
    let features = crate::CpuFeatureTaskflow::new(foreign.runtime.clone(), 128, 4)?;
    let control = SfmTaskControl::new();
    let directory = tempfile::tempdir()?;
    let database = directory.path().join("must-not-be-created.db");
    for selected in [false, true] {
        let mut sink = |_| {};
        let mut task = SfmTaskContext::new(&control, &mut sink)
            .with_taskflow(execution.clone())
            .with_cpu_feature_taskflow(&features);
        let error = if selected {
            crate::feature_extraction::extract_selected_features_to_database_with_task(
                &database,
                directory.path(),
                &Default::default(),
                &[1],
                &mut task,
            )
        } else {
            crate::extract_features_to_database_with_task(
                &database,
                directory.path(),
                &Default::default(),
                &mut task,
            )
        }
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("must share one Taskflow Runtime"));
        assert!(!database.exists());
    }
    released(&execution);
    released(&foreign);
    Ok(())
}

#[test]
fn stage_reports_capture_grant_and_service_time() -> Result<()> {
    let executor = executor(1);
    let control = SfmTaskControl::new();
    let reports = new_stage_report_sink();
    executor.run_with_reports("reported", false, 4, &control, &reports, || {
        std::thread::sleep(Duration::from_millis(5));
        Ok(())
    })?;
    let reports = stage_reports(&reports);
    assert_eq!(reports.len(), 1);
    let report = &reports[0];
    assert_eq!(report.stage_name, "reported");
    assert_eq!(report.requested_threads, 4);
    assert_eq!(report.granted_threads, 1);
    assert_eq!(report.requested_memory, 128);
    assert_eq!(report.granted_memory, 128);
    assert!(report.queue_ms.is_finite() && report.queue_ms >= 0.0);
    assert!(report.service_ms >= 5.0);
    assert!(report.total_ms >= report.service_ms);
    assert!(!report.cancelled_or_failed);
    released(&executor);
    Ok(())
}

#[test]
fn sequence_reports_capture_each_stage_in_order() -> Result<()> {
    let executor = executor(1);
    let control = SfmTaskControl::new();
    let reports = new_stage_report_sink();
    let stages = [("prepare", false, 4), ("register", false, 2)];
    let mut visited = Vec::new();

    executor.sequence_with_reports(&stages, &control, &reports, |index| {
        std::thread::sleep(Duration::from_millis(2));
        assert_eq!(active_threads(), Some(1));
        visited.push(index);
        Ok(())
    })?;
    assert_eq!(visited, [0, 1]);

    let reports = stage_reports(&reports);
    assert_eq!(
        reports
            .iter()
            .map(|report| report.stage_name.as_str())
            .collect::<Vec<_>>(),
        ["prepare", "register"]
    );
    for (report, requested_threads) in reports.iter().zip([4, 2]) {
        assert_eq!(report.requested_threads, requested_threads);
        assert_eq!(report.granted_threads, 1);
        assert_eq!(report.requested_memory, 128);
        assert_eq!(report.granted_memory, 128);
        assert!(report.queue_ms.is_finite() && report.queue_ms >= 0.0);
        assert!(report.service_ms >= 2.0);
        assert!(report.total_ms >= report.service_ms);
        assert!(!report.cancelled_or_failed);
    }
    assert_eq!(executor.runtime.snapshot()?.pending_tasks, 0);
    released(&executor);
    Ok(())
}

#[test]
fn stage_reports_capture_pre_admission_cancel() -> Result<()> {
    let executor = executor(1);
    let control = SfmTaskControl::new();
    control.request_cancel();
    let reports = new_stage_report_sink();
    let error = executor
        .run_with_reports("cancelled", false, 1, &control, &reports, || Ok(()))
        .unwrap_err();
    assert_eq!(
        error.downcast_ref::<SfmTaskStop>(),
        Some(&SfmTaskStop::Cancelled)
    );
    let reports = stage_reports(&reports);
    assert_eq!(reports.len(), 1);
    assert!(reports[0].cancelled_or_failed);
    assert_eq!(reports[0].granted_threads, 0);
    released(&executor);
    Ok(())
}

#[cfg(feature = "gpu-wgpu")]
fn real_gpu_stage_drains_caller_owned_context_on_error() -> Result<()> {
    let executor = executor(1);
    let control = SfmTaskControl::new();
    let context = crate::gpu::WgpuContext::try_new()?;
    println!("GPU integration device: {:?}", context.capabilities());
    let result = executor.run::<()>("GPU extraction", true, 1, &control, || {
        let extractor = crate::gpu::WgpuSiftExtractor::from_context(context.clone())?;
        let gray: Vec<u8> = (0..256 * 256)
            .map(|i| ((i * 13) ^ ((i / 256) * 17)) as u8)
            .collect();
        let features = extractor.extract_grayscale(
            &gray,
            256,
            256,
            &crate::sift::SiftExtractionOptions {
                use_gpu: true,
                max_num_features: 256,
                ..Default::default()
            },
        )?;
        assert!(!features.keypoints.is_empty());
        assert_eq!(executor.runtime.snapshot()?.exclusive, [GPU_KEY]);
        assert_eq!(active_scope().unwrap().devices.lock().unwrap().len(), 1);
        let encoder = context.device().create_command_encoder(&Default::default());
        context.queue().submit(Some(encoder.finish()));
        bail!("error after GPU submission")
    });
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("error after GPU submission"));
    released(&executor);
    // Caller deliberately keeps the context alive past the stage boundary.
    assert!(Arc::strong_count(&context) >= 1);
    Ok(())
}

// A regression must fail the test rather than leave the test harness joining a
// stuck coordinator forever. Cancel first to break accidental child admission
// waits; detach only if cleanup itself fails its second, bounded deadline.
fn bounded_executor_test(
    cpu: usize,
    work: impl FnOnce(&SfmTaskflow, &SfmTaskControl) -> Result<()> + Send + 'static,
) -> Result<()> {
    let executor = executor(cpu);
    let control = SfmTaskControl::new();
    let cancel = control.clone();
    let (send, receive) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            work(&executor, &control)?;
            released(&executor);
            // Runtime teardown is inside the deadline too, before sending completion.
            drop(executor);
            Ok(())
        }));
        let _ = send.send(result);
    });
    let result = match receive.recv_timeout(Duration::from_secs(10)) {
        Ok(result) => result,
        Err(error) => {
            cancel.request_cancel();
            if receive.recv_timeout(Duration::from_secs(5)).is_ok() {
                worker.join().unwrap();
            }
            bail!("bounded execution test did not complete: {error}");
        }
    };
    worker.join().unwrap();
    match result {
        Ok(result) => result,
        Err(panic) => std::panic::resume_unwind(panic),
    }
}

#[test]
fn stage_memory_floor_preserves_runtime_and_never_lowers_estimate() -> Result<()> {
    let executor = executor(1);
    for floor in [0, 64, 128, 256, u64::MAX] {
        let adjusted = executor.clone().with_stage_memory_floor(floor)?;
        assert!(Arc::ptr_eq(executor.runtime(), adjusted.runtime()));
        assert_eq!(adjusted.stage_memory_bytes(), 128.max(floor));
        assert_eq!(
            adjusted.with_stage_memory_floor(0)?.stage_memory_bytes(),
            128.max(floor)
        );
    }
    assert_eq!(executor.stage_memory_bytes(), 128);
    released(&executor);
    Ok(())
}

#[test]
fn active_memory_is_actual_grant_on_caller_workers_and_nested_paths() -> Result<()> {
    use rayon::prelude::*;
    bounded_executor_test(2, |executor, control| {
        let adjusted = executor.clone().with_stage_memory_floor(256)?;
        let reports = new_stage_report_sink();
        adjusted.run_with_reports("memory floor", false, 2, control, &reports, || {
            assert_eq!(active_memory_bytes(), Some(256));
            let check = || -> Result<()> {
                assert_eq!(active_memory_bytes(), Some(256));
                executor.run("smaller estimate", false, 1, control, || {
                    assert_eq!(executor.stage_memory_bytes(), 128);
                    assert_eq!(active_memory_bytes(), Some(256));
                    Ok(())
                })?;
                let too_large = executor.clone().with_stage_memory_floor(257)?;
                let called = Cell::new(false);
                assert!(too_large
                    .run("over parent", false, 1, control, || {
                        called.set(true);
                        Ok(())
                    })
                    .is_err());
                assert!(!called.get());
                assert_eq!(executor.runtime.snapshot()?.pending_tasks, 1);
                Ok(())
            };
            for result in parallel(|| rayon::broadcast(|_| check())) {
                result?;
            }
            parallel(|| (0..8).into_par_iter().try_for_each(|_| check()))
        })?;
        let report = stage_reports(&reports).pop().unwrap();
        assert_eq!((report.requested_memory, report.granted_memory), (256, 256));
        released(executor);
        adjusted.sequence(&[("one", false, 2), ("two", false, 1)], control, |_| {
            assert_eq!(active_memory_bytes(), Some(256));
            Ok(())
        })?;
        released(executor);
        let mut request = ResourceRequest::cpu(CpuRequest::fixed(1));
        request.working_memory_bytes = 192;
        admit(
            executor.runtime(),
            "direct memory",
            request,
            control,
            |grant, _| {
                assert_eq!(grant.working_memory_bytes, 192);
                assert_eq!(active_memory_bytes(), Some(192));
                assert_eq!(parallel(active_memory_bytes), Some(192));
                Ok(())
            },
        )
    })
}

#[test]
fn independent_admissions_each_have_four_live_workers() -> Result<()> {
    bounded_executor_test(8, |executor, control| {
        let (entered, receive) = mpsc::channel();
        let gate = (Mutex::new(false), std::sync::Condvar::new());
        std::thread::scope(|threads| -> Result<()> {
            let mut callers = Vec::new();
            for job in 0..2 {
                let entered = &entered;
                let gate = &gate;
                callers.push(threads.spawn(move || {
                    executor.run("four workers", false, 4, control, || {
                        assert_eq!(active_threads(), Some(4));
                        let owner = active_scope().unwrap();
                        let results = parallel(|| {
                            rayon::broadcast(|_| -> Result<()> {
                                assert_eq!(active_threads(), Some(4));
                                assert!(Arc::ptr_eq(&owner, &active_scope().unwrap()));
                                entered.send((job, std::thread::current().id()))?;
                                let (open, _) = gate
                                    .1
                                    .wait_timeout_while(
                                        gate.0.lock().unwrap(),
                                        Duration::from_secs(5),
                                        |open| !*open,
                                    )
                                    .unwrap();
                                anyhow::ensure!(*open, "worker capacity gate timed out");
                                Ok(())
                            })
                        });
                        for result in results {
                            result?;
                        }
                        Ok(())
                    })
                }));
            }
            let observed = (|| -> Result<()> {
                let deadline = Instant::now() + Duration::from_secs(4);
                let mut workers = [
                    std::collections::HashSet::new(),
                    std::collections::HashSet::new(),
                ];
                for _ in 0..8 {
                    let (job, thread) =
                        receive.recv_timeout(deadline.saturating_duration_since(Instant::now()))?;
                    workers[job].insert(thread);
                }
                assert_eq!(workers[0].len(), 4);
                assert_eq!(workers[1].len(), 4);
                assert!(workers[0].is_disjoint(&workers[1]));
                let state = executor.runtime.snapshot()?;
                assert_eq!(
                    (state.cpu_threads, state.memory_bytes, state.pending_tasks),
                    (8, 256, 2)
                );
                Ok(())
            })();
            // Release even when the capacity observation failed.
            *gate.0.lock().unwrap() = true;
            gate.1.notify_all();
            for caller in callers {
                caller.join().unwrap()?;
            }
            observed
        })
    })
}

#[test]
fn all_workers_and_par_iter_reenter_without_new_admission() -> Result<()> {
    use rayon::prelude::*;
    bounded_executor_test(4, |executor, control| {
        let mut owner = None;
        executor.run("outer", false, 4, control, || {
            let context = active_scope().unwrap();
            assert!(context.pool.get().is_none());
            owner = Some(Arc::downgrade(&context));
            let reports = new_stage_report_sink();
            let check = || -> Result<()> {
                let inherited = active_scope().unwrap();
                assert!(Arc::ptr_eq(&context, &inherited));
                assert_eq!(active_threads(), Some(4));
                active_control().unwrap().checkpoint()?;
                executor.run_with_reports("nested", false, 8, control, &reports, || {
                    assert_eq!(parallel(rayon::current_num_threads), 4);
                    executor.sequence(&[("one", false, 2), ("two", false, 4)], control, |_| {
                        let request = ResourceRequest::cpu(CpuRequest::fixed(1));
                        admit(
                            &executor.runtime,
                            "borrow",
                            request,
                            control,
                            |grant, queue_ms| {
                                assert_eq!(grant.cpu_threads, 1);
                                assert_eq!(queue_ms, 0.0);
                                let state = executor.runtime.snapshot()?;
                                assert_eq!(
                                    (state.cpu_threads, state.memory_bytes, state.pending_tasks),
                                    (4, 128, 1)
                                );
                                Ok(())
                            },
                        )
                    })
                })
            };
            for result in parallel(|| rayon::broadcast(|_| check())) {
                result?;
            }
            parallel(|| (0..32).into_par_iter().try_for_each(|_| check()))?;
            let reports = stage_reports(&reports);
            assert_eq!(reports.len(), 36);
            assert!(reports
                .iter()
                .all(|report| report.queue_ms == 0.0 && report.granted_threads == 4));
            Ok(())
        })?;
        assert!(
            owner.unwrap().upgrade().is_none(),
            "pool/TLS retained the owner"
        );
        Ok(())
    })
}

#[test]
fn native_admit_on_compute_workers_uses_one_cpu_per_worker() -> Result<()> {
    bounded_executor_test(4, |executor, control| {
        executor.run("outer", false, 4, control, || {
            let caller_native = || {
                admit(
                    executor.runtime(),
                    "caller native",
                    ResourceRequest::cpu(CpuRequest::fixed(4)),
                    control,
                    |grant, queue| {
                        assert_eq!(grant.cpu_threads, 4);
                        assert_eq!(queue, 0.0);
                        Ok(())
                    },
                )
            };
            caller_native()?;
            let entered = (Mutex::new(0usize), std::sync::Condvar::new());
            let results = parallel(|| {
                rayon::broadcast(|_| -> Result<std::thread::ThreadId> {
                    assert_eq!(active_threads(), Some(4));
                    let called = Cell::new(false);
                    let error = admit(
                        executor.runtime(),
                        "fixed native",
                        ResourceRequest::cpu(CpuRequest::fixed(4)),
                        control,
                        |_, _| {
                            called.set(true);
                            Ok(())
                        },
                    )
                    .unwrap_err();
                    assert!(!called.get());
                    assert!(error.to_string().contains("worker allowance is 1"));
                    executor.run("nested stage", false, 4, control, || {
                        assert_eq!(active_threads(), Some(4));
                        assert_eq!(parallel(rayon::current_num_threads), 4);
                        admit(
                            executor.runtime(),
                            "scalable native",
                            ResourceRequest::cpu(CpuRequest::scalable(1, 4, 4)),
                            control,
                            |grant, queue| {
                                assert_eq!(grant.cpu_threads, 1);
                                assert_eq!(queue, 0.0);
                                // The bridge narrows only this native grant, not
                                // the context used by structured Rayon work.
                                assert_eq!(active_threads(), Some(4));
                                let mut count = entered.0.lock().unwrap();
                                *count += 1;
                                entered.1.notify_all();
                                let (count, _) = entered
                                    .1
                                    .wait_timeout_while(count, Duration::from_secs(3), |count| {
                                        *count < 4
                                    })
                                    .unwrap();
                                anyhow::ensure!(
                                    *count == 4,
                                    "parallel native admission gate timed out"
                                );
                                let state = executor.runtime.snapshot()?;
                                assert_eq!(
                                    (state.cpu_threads, state.memory_bytes, state.pending_tasks),
                                    (4, 128, 1)
                                );
                                Ok(std::thread::current().id())
                            },
                        )
                    })
                })
            });
            let workers = results
                .into_iter()
                .collect::<Result<std::collections::HashSet<_>>>()?;
            assert_eq!(workers.len(), 4);
            // The pool now exists; the original caller must still get CPU4.
            caller_native()
        })
    })
}

#[test]
fn native_admit_does_not_limit_a_caller_on_an_unrelated_rayon_pool() -> Result<()> {
    bounded_executor_test(4, |executor, control| {
        let caller_pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
        caller_pool.install(|| {
            executor.run("Rayon caller", false, 4, control, || {
                assert!(rayon::current_thread_index().is_some());
                assert_eq!(parallel(rayon::current_num_threads), 4);
                assert!(active_scope()
                    .unwrap()
                    .pool
                    .get()
                    .unwrap()
                    .current_thread_index()
                    .is_none());
                admit(
                    executor.runtime(),
                    "caller native",
                    ResourceRequest::cpu(CpuRequest::fixed(4)),
                    control,
                    |grant, queue| {
                        assert_eq!(grant.cpu_threads, 4);
                        assert_eq!(queue, 0.0);
                        Ok(())
                    },
                )
            })
        })
    })
}

#[test]
fn worker_reentry_rejects_foreign_runtime_and_excess_resources() -> Result<()> {
    bounded_executor_test(2, |execution, control| {
        let foreign = executor(2);
        let larger = SfmTaskflow::new(execution.runtime.clone(), 256)?;
        execution.run("outer", false, 2, control, || {
            for result in parallel(|| {
                rayon::broadcast(|_| -> Result<()> {
                    assert!(foreign
                        .run("foreign", false, 1, control, || Ok(()))
                        .is_err());
                    assert!(foreign
                        .sequence(&[("foreign", false, 1)], control, |_| Ok(()))
                        .is_err());
                    assert!(execution
                        .run("GPU upgrade", true, 1, control, || Ok(()))
                        .is_err());
                    assert!(larger.run("memory", false, 1, control, || Ok(())).is_err());
                    assert!(larger
                        .sequence(&[("memory", false, 1)], control, |_| Ok(()))
                        .is_err());
                    assert!(admit(
                        &foreign.runtime,
                        "foreign",
                        ResourceRequest::cpu(CpuRequest::fixed(1)),
                        control,
                        |_, _| Ok(())
                    )
                    .is_err());
                    let base = ResourceRequest::cpu(CpuRequest::fixed(1));
                    let mut requests = vec![ResourceRequest::cpu(CpuRequest::fixed(3))];
                    let mut memory = base.clone();
                    memory.working_memory_bytes = 129;
                    requests.push(memory);
                    let mut output = base.clone();
                    output.output_memory_bytes = 1;
                    requests.push(output);
                    let mut io = base.clone();
                    io.io_slots = 1;
                    requests.push(io);
                    let mut exclusive = base;
                    exclusive.exclusive.push("unreserved".into());
                    requests.push(exclusive);
                    for request in requests {
                        let called = Cell::new(false);
                        assert!(
                            admit(&execution.runtime, "excess", request, control, |_, _| {
                                called.set(true);
                                Ok(())
                            })
                            .is_err()
                        );
                        assert!(!called.get());
                    }
                    assert_eq!(execution.runtime.snapshot()?.pending_tasks, 1);
                    assert_eq!(foreign.runtime.snapshot()?.pending_tasks, 0);
                    Ok(())
                })
            }) {
                result?;
            }
            Ok(())
        })?;
        released(&foreign);
        Ok(())
    })
}

#[test]
fn worker_cancel_and_panic_release_owner_and_do_not_leak_tls() -> Result<()> {
    bounded_executor_test(2, |executor, control| {
        let mut cancelled_owner = None;
        let error = executor
            .run("cancel", false, 2, control, || {
                cancelled_owner = Some(Arc::downgrade(&active_scope().unwrap()));
                parallel(|| active_control().unwrap().request_cancel());
                for result in parallel(|| {
                    rayon::broadcast(|_| -> Result<()> {
                        // A fresh child control must not hide cancellation of its parent.
                        let fresh = SfmTaskControl::new();
                        let errors = [
                            executor
                                .run("cancelled child", false, 1, &fresh, || Ok(()))
                                .unwrap_err(),
                            executor
                                .sequence(&[("cancelled sequence", false, 1)], &fresh, |_| Ok(()))
                                .unwrap_err(),
                            admit(
                                &executor.runtime,
                                "cancelled borrow",
                                ResourceRequest::cpu(CpuRequest::fixed(1)),
                                &fresh,
                                |_, _| Ok(()),
                            )
                            .unwrap_err(),
                        ];
                        for error in errors {
                            assert_eq!(
                                error.downcast_ref::<SfmTaskStop>(),
                                Some(&SfmTaskStop::Cancelled)
                            );
                        }
                        Ok(())
                    })
                }) {
                    result?;
                }
                control.checkpoint()?;
                Ok(())
            })
            .unwrap_err();
        assert_eq!(
            error.downcast_ref::<SfmTaskStop>(),
            Some(&SfmTaskStop::Cancelled)
        );
        assert!(cancelled_owner.unwrap().upgrade().is_none());
        released(executor);

        let fresh = SfmTaskControl::new();
        let mut failed_owner = None;
        let error = executor
            .run::<()>("worker error", false, 2, &fresh, || {
                failed_owner = Some(Arc::downgrade(&active_scope().unwrap()));
                parallel(|| bail!("synthetic worker error"))
            })
            .unwrap_err();
        assert!(error.to_string().contains("synthetic worker error"));
        assert!(failed_owner.unwrap().upgrade().is_none());
        released(executor);

        let mut panicked_owner = None;
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            executor.run::<()>("worker panic", false, 2, &fresh, || {
                panicked_owner = Some(Arc::downgrade(&active_scope().unwrap()));
                parallel(|| {
                    rayon::broadcast(|worker| {
                        assert!(active_scope().is_some());
                        if worker.index() == 0 {
                            panic!("synthetic worker panic");
                        }
                    })
                });
                Ok(())
            })
        }));
        assert!(panic.is_err());
        assert!(panicked_owner.unwrap().upgrade().is_none());
        released(executor);
        executor.run("fresh admission", false, 2, &fresh, || {
            for result in parallel(|| rayon::broadcast(|_| active_control().unwrap().checkpoint()))
            {
                result?;
            }
            Ok(())
        })
    })
}

#[test]
fn direct_admit_and_sequence_own_distinct_lazy_contexts() -> Result<()> {
    bounded_executor_test(2, |executor, control| {
        let request = ResourceRequest::cpu(CpuRequest::fixed(2));
        let mut previous = None;
        admit(&executor.runtime, "direct", request, control, |grant, _| {
            let owner = active_scope().unwrap();
            assert_eq!(owner.grant.cpu_threads, grant.cpu_threads);
            assert!(owner.pool.get().is_none());
            previous = Some(Arc::downgrade(&owner));
            parallel(|| {
                admit(
                    &executor.runtime,
                    "nested direct",
                    ResourceRequest::cpu(CpuRequest::fixed(1)),
                    control,
                    |grant, queue| {
                        assert_eq!(grant.cpu_threads, 1);
                        assert_eq!(queue, 0.0);
                        assert_eq!(executor.runtime.snapshot()?.pending_tasks, 1);
                        Ok(())
                    },
                )
            })
        })?;
        assert!(previous.as_ref().unwrap().upgrade().is_none());
        executor.sequence(
            &[("first", false, 2), ("second", false, 1)],
            control,
            |index| {
                assert!(previous.as_ref().unwrap().upgrade().is_none());
                let owner = active_scope().unwrap();
                assert!(owner.pool.get().is_none());
                assert_eq!(parallel(rayon::current_num_threads), [2, 1][index]);
                previous = Some(Arc::downgrade(&owner));
                Ok(())
            },
        )?;
        assert!(previous.unwrap().upgrade().is_none());
        Ok(())
    })
}

#[test]
fn expired_worker_marker_does_not_fall_back_to_fresh_admission() {
    let previous = ACTIVE.with(|active| active.replace(Some(Weak::new())));
    let outcome = std::panic::catch_unwind(active_threads);
    ACTIVE.with(|active| active.replace(previous));
    assert!(outcome.is_err());
    assert!(active_threads().is_none());
}

#[test]
fn nested_gpu_upgrade_and_runtime_switch_fail_without_deadlock() -> Result<()> {
    let executor = executor(1);
    let other = super::tests::executor(1);
    let control = SfmTaskControl::new();
    executor.run("cpu", false, 1, &control, || {
        assert!(executor.run("gpu", true, 1, &control, || Ok(())).is_err());
        assert!(other
            .run("other runtime", false, 1, &control, || Ok(()))
            .is_err());
        Ok(())
    })?;
    released(&executor);
    Ok(())
}
