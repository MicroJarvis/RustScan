use super::*;
use crate::ba::{try_refine_bundle_adjustment, BundleAdjustmentLoss};
use crate::task::SfmTaskStop;
use rustscan_taskflow::{Budget, RuntimeConfig};

const TIMEOUT: Duration = Duration::from_secs(5);
fn runtime(cpu: usize) -> Arc<Runtime> {
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
    )
}
fn fixture() -> (Vec<ImageFrame>, Reconstruction, BundleAdjustmentOptions) {
    let (frames, reconstruction, ..) = crate::ba::ceres::tests::rig_sensor_ba_fixture();
    let options = BundleAdjustmentOptions {
        iterations: 30,
        max_observation_error_px: 200.0,
        loss_function: BundleAdjustmentLoss::Huber { scale: 4.0 },
        variable_images: Some(vec![0, 1]),
        constant_images: vec![2, 3],
        point_ids: Some((0..8).collect()),
        ..Default::default()
    };
    (frames, reconstruction, options)
}
fn assert_released(runtime: &Runtime) {
    let used = runtime.snapshot().unwrap();
    assert_eq!(used.cpu_threads, 0);
    assert_eq!(used.memory_bytes, 0);
    assert_eq!(used.pending_tasks, 0);
}

#[test]
fn taskflow_ceres_matches_direct_solve_and_preserves_fixed_poses() -> Result<()> {
    let (frames, base, options) = fixture();
    let mut direct = base.clone();
    let expected = try_refine_bundle_adjustment(&frames, &mut direct, options.clone())?.unwrap();
    let runtime = runtime(4);
    let executor = CeresBaTaskflow::new(runtime.clone(), 4, 512)?;
    assert_eq!(executor.preferred_threads(&base, &options), 1);
    let mut scheduled = base.clone();
    let report = try_refine_bundle_adjustment(
        &frames,
        &mut scheduled,
        BundleAdjustmentOptions {
            taskflow: Some(executor),
            ..options
        },
    )?
    .unwrap();
    assert_eq!(report.solver_num_threads, 1);
    let grant = report.scheduling.as_ref().unwrap();
    assert_eq!(grant.granted_cpu_threads, 1);
    assert!(grant.queue_ms.is_finite() && grant.queue_ms >= 0.0);
    assert_eq!(report.residuals, expected.residuals);
    assert_eq!(report.linear_solver, expected.linear_solver);
    assert!(report.is_solution_usable());
    assert!(
        (report.final_cost - expected.final_cost).abs() < 1e-5 * expected.initial_cost.max(1.0)
    );
    for (left, right) in scheduled.points.iter().zip(&direct.points) {
        for (left, right) in left.xyz.iter().zip(right.xyz) {
            assert!((left - right).abs() < 1e-4);
        }
    }
    for id in [2, 3] {
        assert_eq!(
            scheduled.poses[id].unwrap().translation(),
            base.poses[id].unwrap().translation()
        );
        assert_eq!(
            scheduled.poses[id].unwrap().quaternion(),
            base.poses[id].unwrap().quaternion()
        );
    }
    assert_released(&runtime);
    Ok(())
}

#[test]
fn taskflow_ceres_uses_partial_grant_and_respects_explicit_thread_cap() -> Result<()> {
    let runtime = runtime(3);
    let (entered, started) = mpsc::sync_channel(1);
    let (release, released) = mpsc::sync_channel(1);
    let mut graph = TaskGraph::new();
    graph.task(
        "other workflow",
        vec![TaskVariant::cpu(
            "cpu",
            ResourceRequest::cpu(CpuRequest::fixed(1)),
            move |_| {
                entered.send(()).unwrap();
                released.recv_timeout(TIMEOUT).unwrap();
                Ok(())
            },
        )],
    )?;
    let peer = runtime.submit(graph)?;
    started.recv_timeout(TIMEOUT)?;
    let (frames, base, mut options) = fixture();
    options.min_num_residuals_for_multi_threading = 0;
    let executor = CeresBaTaskflow::new(runtime.clone(), 4, 512)?;
    let mut reconstruction = base.clone();
    let report = executor
        .refine(&frames, &mut reconstruction, options.clone())?
        .unwrap();
    assert_eq!(report.scheduling.unwrap().granted_cpu_threads, 2);
    assert_eq!(report.solver_num_threads, 2);
    assert!(report.final_cost <= report.initial_cost);
    options.num_threads = 1;
    let report = executor
        .refine(&frames, &mut reconstruction, options)?
        .unwrap();
    assert_eq!(report.scheduling.unwrap().granted_cpu_threads, 1);
    assert_eq!(report.solver_num_threads, 1);
    release.send(())?;
    assert!(peer.wait_timeout(TIMEOUT).unwrap().succeeded());
    assert_released(&runtime);
    Ok(())
}

#[test]
fn taskflow_ceres_cancel_queued_solve_does_not_mutate_model() -> Result<()> {
    let runtime = runtime(2);
    let mut budget = runtime.snapshot()?.budget;
    budget.cpu_threads = 0;
    runtime.set_budget(budget)?;
    let control = SfmTaskControl::new();
    let executor = CeresBaTaskflow::new(runtime.clone(), 2, 512)?.with_control(control.clone());
    let (frames, mut reconstruction, options) = fixture();
    let before = format!("{reconstruction:?}");
    std::thread::scope(|scope| -> Result<()> {
        let solve = scope.spawn(|| executor.refine(&frames, &mut reconstruction, options));
        let deadline = Instant::now() + TIMEOUT;
        while runtime.snapshot()?.pending_tasks == 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        let queued = runtime.snapshot()?.pending_tasks;
        control.request_cancel();
        let error = solve.join().unwrap().unwrap_err();
        assert_eq!(queued, 1);
        assert_eq!(
            error.downcast_ref::<SfmTaskStop>(),
            Some(&SfmTaskStop::Cancelled)
        );
        Ok(())
    })?;
    assert_eq!(format!("{reconstruction:?}"), before);
    assert_released(&runtime);
    Ok(())
}

#[test]
fn taskflow_ceres_does_not_revoke_running_resources_on_cancel_or_budget_reduction() -> Result<()> {
    let runtime = runtime(2);
    let control = SfmTaskControl::new();
    let executor = CeresBaTaskflow::new(runtime.clone(), 2, 512)?.with_control(control.clone());
    let (entered, started) = mpsc::sync_channel(1);
    let (release, released) = mpsc::sync_channel(1);
    std::thread::scope(|scope| -> Result<()> {
        let running_control = control.clone();
        let running = scope.spawn(move || {
            executor.run_admitted(2, |grant, _| {
                assert_eq!(grant.cpu_threads, 2);
                entered.send(())?;
                released.recv_timeout(TIMEOUT)?;
                running_control.checkpoint()?;
                Ok(())
            })
        });
        started.recv_timeout(TIMEOUT)?;
        control.request_cancel();
        let mut budget = runtime.snapshot()?.budget;
        budget.cpu_threads = 0;
        budget.memory_bytes = 0;
        runtime.set_budget(budget)?;
        let used = runtime.snapshot()?;
        assert_eq!(used.cpu_threads, 2);
        assert_eq!(used.memory_bytes, 512);
        release.send(())?;
        assert_eq!(
            running
                .join()
                .unwrap()
                .unwrap_err()
                .downcast_ref::<SfmTaskStop>(),
            Some(&SfmTaskStop::Cancelled)
        );
        Ok(())
    })?;
    assert_released(&runtime);
    Ok(())
}

#[test]
fn taskflow_ceres_panic_and_error_release_scoped_reservation() -> Result<()> {
    let runtime = runtime(2);
    let executor = CeresBaTaskflow::new(runtime.clone(), 2, 512)?;
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        executor.run_admitted::<()>(2, |_, _| panic!("native scope panic"))
    }));
    assert!(panic.is_err());
    assert_released(&runtime);
    assert!(executor
        .run_admitted::<()>(2, |_, _| bail!("backend error"))
        .is_err());
    assert_released(&runtime);
    Ok(())
}

#[test]
fn taskflow_ceres_pre_pause_and_unschedulable_memory_fail_before_mutation() -> Result<()> {
    let runtime = runtime(1);
    let (frames, mut reconstruction, options) = fixture();
    let before = format!("{reconstruction:?}");
    let executor = CeresBaTaskflow::new(runtime.clone(), 1, 2048)?;
    assert!(executor
        .refine(&frames, &mut reconstruction, options.clone())
        .is_err());
    let control = SfmTaskControl::new();
    control.request_pause();
    let executor = executor.with_control(control);
    assert_eq!(
        executor
            .refine(&frames, &mut reconstruction, options)
            .unwrap_err()
            .downcast_ref::<SfmTaskStop>(),
        Some(&SfmTaskStop::Paused)
    );
    assert_eq!(format!("{reconstruction:?}"), before);
    assert_released(&runtime);
    Ok(())
}

#[test]
fn taskflow_context_binding_preserves_ba_policy_and_uses_workflow_control() -> Result<()> {
    let runtime = runtime(2);
    let executor = CeresBaTaskflow::new(runtime, 2, 512)?;
    let control = SfmTaskControl::new();
    let mut sink = |_| {};
    let task = crate::SfmTaskContext::new(&control, &mut sink).with_ceres_ba_taskflow(&executor);
    let mut config = crate::MapperConfig {
        global_ba: false,
        local_ba: true,
        global_ba_iterations: 17,
        ..Default::default()
    };
    task.bind_ba_taskflow(&mut config);
    assert!(!config.global_ba && config.local_ba);
    assert_eq!(config.global_ba_iterations, 17);
    control.request_pause();
    let (frames, mut reconstruction, options) = fixture();
    assert_eq!(
        config
            .ba_taskflow
            .unwrap()
            .refine(&frames, &mut reconstruction, options)
            .unwrap_err()
            .downcast_ref::<SfmTaskStop>(),
        Some(&SfmTaskStop::Paused)
    );
    Ok(())
}
