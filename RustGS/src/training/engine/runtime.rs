//! WGPU training runtime orchestration.

use std::time::Instant;

use glam::Vec3;

use crate::core::GaussianCamera;
use crate::core::HostSplats;
use crate::training::data::frame_loader::{
    ordered_frame_indices, FrameLoaderOptions, PrefetchFrameLoader,
};
use crate::training::data::init_map::build_initial_splats;
use crate::training::evaluation::{scaled_dimensions, SharedWgpuContext};
use crate::training::events::{
    emit_training_event, TrainingCheckpointPolicy, TrainingCheckpointReady,
    TrainingCheckpointReason, TrainingControl, TrainingEvent, TrainingEventCadence,
    TrainingEventRoute, TrainingIterationProgress, TrainingOptions, TrainingPlanSelected,
    TrainingRun, TrainingRunCancelled, TrainingRunCompleted, TrainingRunDisposition,
    TrainingRunFailed, TrainingRunPaused, TrainingRunReport, TrainingRunStarted,
    TrainingSnapshotReady,
};
use crate::training::reporting::telemetry::store_last_training_telemetry;
use crate::training::TrainingConfig;
use crate::{Intrinsics, TrainingDataset, TrainingError};

use super::backend::{GsDevice, GsDiffBackend};
use super::splats::{device_splats_to_host, host_splats_to_device};
use super::trainer::{
    target_image_tensor, TrainingIterationMetrics, TrainingLoopObserver, WgpuTrainer,
    WgpuTrainingReport,
};

fn prepare_resume_runtime<T, SharedFactory, DefaultFactory>(
    config: &TrainingConfig,
    current_identity: Option<&crate::TrainingIdentity>,
    resume_checkpoint: Option<&crate::TrainingCheckpoint>,
    shared_device: Option<SharedFactory>,
    default_device: DefaultFactory,
) -> Result<(usize, T), TrainingError>
where
    SharedFactory: FnOnce() -> T,
    DefaultFactory: FnOnce() -> T,
{
    let start_iteration = if let Some(checkpoint) = resume_checkpoint {
        checkpoint.validate()?;
        let current_identity = current_identity.ok_or_else(|| {
            TrainingError::InvalidInput(
                "resuming training requires the current training identity".to_string(),
            )
        })?;
        if checkpoint.identity.dataset != current_identity.dataset {
            return Err(TrainingError::InvalidInput(
                "checkpoint dataset does not match the current training dataset".to_string(),
            ));
        }
        if checkpoint.identity.reconstruction != current_identity.reconstruction {
            return Err(TrainingError::InvalidInput(
                "checkpoint reconstruction does not match the current sparse reconstruction"
                    .to_string(),
            ));
        }
        if checkpoint.identity.config != current_identity.config {
            return Err(TrainingError::InvalidInput(
                "checkpoint configuration does not match the current training configuration"
                    .to_string(),
            ));
        }
        if config.iterations < checkpoint.completed_iterations {
            return Err(TrainingError::InvalidInput(format!(
                "training iteration target {} is lower than checkpoint completed iterations {}",
                config.iterations, checkpoint.completed_iterations
            )));
        }
        checkpoint.completed_iterations
    } else {
        0
    };

    let device = match shared_device {
        Some(shared_device) => shared_device(),
        None => default_device(),
    };
    Ok((start_iteration, device))
}

pub fn train_splats(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
    options: TrainingOptions<'_>,
) -> Result<TrainingRun, TrainingError> {
    train_splats_with_device_factory(dataset, config, options, GsDevice::default)
}

fn train_splats_with_device_factory<DefaultFactory>(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
    options: TrainingOptions<'_>,
    default_device: DefaultFactory,
) -> Result<TrainingRun, TrainingError>
where
    DefaultFactory: FnOnce() -> GsDevice,
{
    let run_started_at = Instant::now();
    let TrainingOptions {
        control,
        identity,
        resume_checkpoint,
        checkpoint_policy,
        shared_wgpu_context,
        mut on_event,
        mut on_checkpoint,
    } = options;
    let mut noop = |_event| {};
    let mut noop_checkpoint = |_ready: &TrainingCheckpointReady| Ok(());
    let emit_iteration_events = on_event.is_some();
    let on_event = on_event.as_deref_mut();
    let on_event = on_event.unwrap_or(&mut noop);
    let on_checkpoint = on_checkpoint.as_deref_mut();
    let on_checkpoint = on_checkpoint.unwrap_or(&mut noop_checkpoint);

    emit_training_event(
        on_event,
        TrainingEvent::RunStarted(TrainingRunStarted {
            iterations: config.iterations,
            frame_count: dataset.poses.len(),
            input_point_count: dataset.initial_points.len(),
        }),
    );
    emit_training_event(
        on_event,
        TrainingEvent::PlanSelected(TrainingPlanSelected {
            route: TrainingEventRoute::Standard,
        }),
    );

    let run = match run_training(
        dataset,
        config,
        &control,
        identity.as_ref(),
        resume_checkpoint.as_ref(),
        checkpoint_policy,
        shared_wgpu_context.as_ref(),
        emit_iteration_events,
        on_event,
        on_checkpoint,
        default_device,
    ) {
        Ok(run) => run,
        Err(error) => {
            emit_training_event(
                on_event,
                TrainingEvent::RunFailed(TrainingRunFailed {
                    error: error.to_string(),
                    elapsed: run_started_at.elapsed(),
                }),
            );
            return Err(error);
        }
    };

    match run.report.disposition {
        TrainingRunDisposition::Cancelled => {
            emit_training_event(
                on_event,
                TrainingEvent::RunCancelled(TrainingRunCancelled {
                    completed_iterations: run.report.completed_iterations,
                    elapsed: run.report.elapsed,
                }),
            );
        }
        TrainingRunDisposition::Paused => {
            emit_training_event(
                on_event,
                TrainingEvent::RunPaused(TrainingRunPaused {
                    completed_iterations: run.report.completed_iterations,
                    elapsed: run.report.elapsed,
                }),
            );
        }
        TrainingRunDisposition::Completed => {}
    }

    emit_training_event(
        on_event,
        TrainingEvent::RunCompleted(TrainingRunCompleted {
            report: run.report.clone(),
        }),
    );
    Ok(run)
}

#[allow(clippy::too_many_arguments)]
fn run_training<F, C, DefaultFactory>(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
    control: &TrainingControl,
    identity: Option<&crate::TrainingIdentity>,
    resume_checkpoint: Option<&crate::TrainingCheckpoint>,
    checkpoint_policy: TrainingCheckpointPolicy,
    shared_wgpu_context: Option<&SharedWgpuContext>,
    emit_iteration_events: bool,
    on_event: &mut F,
    on_checkpoint: &mut C,
    default_device: DefaultFactory,
) -> Result<TrainingRun, TrainingError>
where
    F: FnMut(TrainingEvent) + ?Sized,
    C: FnMut(&TrainingCheckpointReady) -> Result<(), TrainingError> + ?Sized,
    DefaultFactory: FnOnce() -> GsDevice,
{
    let shared_device = shared_wgpu_context.map(|context| || context.training_device());
    let (start_iteration, device) = prepare_resume_runtime(
        config,
        identity,
        resume_checkpoint,
        shared_device,
        default_device,
    )?;

    if dataset.poses.is_empty() {
        return Err(TrainingError::InvalidInput(
            "training dataset does not contain any poses".to_string(),
        ));
    }

    let started_at = Instant::now();
    let input_width = dataset.intrinsics.width as usize;
    let input_height = dataset.intrinsics.height as usize;
    let (target_width, target_height) =
        scaled_dimensions(input_width, input_height, config.raster.render_scale);
    let initial_splats = resume_checkpoint
        .is_none()
        .then(|| build_initial_splats(dataset, config))
        .transpose()?;
    let training_splats = resume_checkpoint
        .map(|checkpoint| &checkpoint.splats)
        .or(initial_splats.as_ref())
        .expect("new and resumed training both provide splats");

    let mut loader = PrefetchFrameLoader::new(
        dataset,
        config,
        FrameLoaderOptions {
            cache_capacity: config.data.frame_cache_capacity,
            prefetch_ahead: config.data.frame_prefetch_ahead,
            rgb_target_size: Some((target_width, target_height)),
        },
    )?;

    let frame_shuffle_seed = resume_checkpoint
        .map(|checkpoint| checkpoint.frame_shuffle_seed)
        .unwrap_or(config.data.frame_shuffle_seed);
    let frame_order = ordered_frame_indices(dataset.poses.len(), 1, frame_shuffle_seed);
    let mut cameras = Vec::with_capacity(frame_order.len());

    for &pose_idx in &frame_order {
        let pose = &dataset.poses[pose_idx];
        cameras.push(gaussian_camera_from_scene_pose(
            &pose.pose,
            dataset.intrinsics,
            target_width,
            target_height,
        ));
    }
    let scene_scale = camera_scene_scale(dataset, &frame_order);
    log::info!("WGPU training scene scale: {:.6}", scene_scale);

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| {
            TrainingError::TrainingFailed(format!("failed to build tokio runtime: {err}"))
        })?
        .block_on(async move {
            warm_up_training_kernels(
                training_splats,
                config,
                &device,
                &cameras,
                &frame_order,
                &mut loader,
                (target_width, target_height),
                scene_scale,
                start_iteration,
            )
            .await?;
            let (mut trainer, mut device_splats) = match resume_checkpoint {
                Some(checkpoint) => {
                    WgpuTrainer::from_checkpoint(
                        config.clone(),
                        device.clone(),
                        scene_scale,
                        checkpoint,
                    )
                    .await?
                }
                None => {
                    let device_splats =
                        host_splats_to_device::<GsDiffBackend>(training_splats, &device);
                    let sh_coeffs = device_splats.sh_coeffs.val().dims()[1];
                    let trainer = WgpuTrainer::new(
                        config.clone(),
                        device.clone(),
                        device_splats.num_splats(),
                        sh_coeffs,
                        scene_scale,
                    );
                    (trainer, device_splats)
                }
            };
            let mut observer = TrainingEventObserver {
                control,
                cadence: control.cadence(),
                checkpoint_policy,
                identity,
                emit_iteration_events,
                started_at,
                on_event,
                on_checkpoint,
            };
            let mut report = trainer
                .train_with_frame_loader(
                    &mut device_splats,
                    &cameras,
                    &frame_order,
                    &mut loader,
                    (target_width, target_height),
                    start_iteration,
                    config.iterations,
                    &mut observer,
                )
                .await?;
            if report.final_loss.is_none() {
                report.final_loss = resume_checkpoint.and_then(|checkpoint| checkpoint.latest_loss);
                report.final_step_loss = report.final_loss;
            }
            let splats = device_splats_to_host(&device_splats).await;
            Ok(build_training_run(splats, report, started_at.elapsed()))
        })
}

#[allow(clippy::too_many_arguments)]
async fn warm_up_training_kernels(
    initial_splats: &HostSplats,
    config: &TrainingConfig,
    device: &GsDevice,
    cameras: &[GaussianCamera],
    frame_order: &[usize],
    frame_loader: &mut PrefetchFrameLoader,
    image_dims: (usize, usize),
    scene_scale: f32,
    start_iteration: usize,
) -> Result<(), TrainingError> {
    if start_iteration >= config.iterations || cameras.is_empty() || frame_order.is_empty() {
        return Ok(());
    }

    let sample_idx = start_iteration % cameras.len();
    let frame_idx = frame_order[sample_idx];
    frame_loader.prefetch_order_window(frame_order, sample_idx)?;
    let decoded = frame_loader.get(frame_idx)?;
    let target_image = decoded.target_rgb.clone().ok_or_else(|| {
        TrainingError::TrainingFailed(format!(
            "frame loader did not prepare target_rgb for warmup frame {frame_idx}"
        ))
    })?;
    let target_img = target_image_tensor(&target_image, image_dims, device);

    let mut warmup_splats = host_splats_to_device::<GsDiffBackend>(initial_splats, device);
    let sh_coeffs = warmup_splats.sh_coeffs.val().dims()[1];
    let mut warmup_trainer = WgpuTrainer::new(
        config.clone(),
        device.clone(),
        warmup_splats.num_splats(),
        sh_coeffs,
        scene_scale,
    );
    let started_at = Instant::now();
    warmup_trainer
        .train_step(
            &mut warmup_splats,
            &cameras[sample_idx],
            target_img,
            image_dims,
            start_iteration + 1,
            cameras.len(),
            false,
        )
        .await?;
    log::debug!(
        "WGPU training warmup completed in {:.3}ms",
        started_at.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

struct TrainingEventObserver<'a, F, C>
where
    F: FnMut(TrainingEvent) + ?Sized,
    C: FnMut(&TrainingCheckpointReady) -> Result<(), TrainingError> + ?Sized,
{
    control: &'a TrainingControl,
    cadence: TrainingEventCadence,
    checkpoint_policy: TrainingCheckpointPolicy,
    identity: Option<&'a crate::TrainingIdentity>,
    emit_iteration_events: bool,
    started_at: Instant,
    on_event: &'a mut F,
    on_checkpoint: &'a mut C,
}

impl<F, C> TrainingLoopObserver for TrainingEventObserver<'_, F, C>
where
    F: FnMut(TrainingEvent) + ?Sized,
    C: FnMut(&TrainingCheckpointReady) -> Result<(), TrainingError> + ?Sized,
{
    fn should_cancel(&self) -> bool {
        self.control.is_cancel_requested()
    }

    fn should_emit_progress(&self, iteration: usize) -> bool {
        self.emit_iteration_events && self.cadence.should_emit_progress(iteration)
    }

    fn should_emit_snapshot(&self, iteration: usize) -> bool {
        self.emit_iteration_events && self.cadence.should_emit_snapshot(iteration)
    }

    fn checkpoint_reason(&self, iteration: usize) -> Option<TrainingCheckpointReason> {
        if self.control.is_cancel_requested() {
            None
        } else if self.control.is_pause_requested() {
            Some(TrainingCheckpointReason::Pause)
        } else if self.checkpoint_policy.should_checkpoint(iteration) {
            Some(TrainingCheckpointReason::Periodic)
        } else {
            None
        }
    }

    fn checkpoint_identity(&self) -> Option<&crate::TrainingIdentity> {
        self.identity
    }

    fn on_iteration(&mut self, metrics: TrainingIterationMetrics) {
        (self.on_event)(TrainingEvent::IterationProgress(
            TrainingIterationProgress {
                iteration: metrics.iteration,
                latest_loss: metrics.loss,
                gaussian_count: metrics.gaussian_count,
                elapsed: self.started_at.elapsed(),
            },
        ));
    }

    fn on_snapshot(&mut self, metrics: TrainingIterationMetrics, splats: HostSplats) {
        (self.on_event)(TrainingEvent::SnapshotReady(TrainingSnapshotReady {
            iteration: metrics.iteration,
            latest_loss: metrics.loss,
            gaussian_count: metrics.gaussian_count,
            elapsed: self.started_at.elapsed(),
            splats,
        }));
    }

    fn on_checkpoint(&mut self, ready: TrainingCheckpointReady) -> Result<(), TrainingError> {
        (self.on_checkpoint)(&ready)?;
        (self.on_event)(TrainingEvent::CheckpointReady(ready));
        Ok(())
    }
}

fn gaussian_camera_from_scene_pose(
    pose: &crate::SE3,
    intrinsics: crate::Intrinsics,
    target_width: usize,
    target_height: usize,
) -> GaussianCamera {
    let sx = target_width as f32 / intrinsics.width as f32;
    let sy = target_height as f32 / intrinsics.height as f32;
    let scaled_intrinsics = Intrinsics::new(
        intrinsics.fx * sx,
        intrinsics.fy * sy,
        intrinsics.cx * sx,
        intrinsics.cy * sy,
        target_width as u32,
        target_height as u32,
    );
    GaussianCamera::new(scaled_intrinsics, pose.inverse())
}

fn camera_scene_scale(dataset: &TrainingDataset, frame_order: &[usize]) -> f32 {
    let mut center = Vec3::ZERO;
    let mut count = 0usize;
    for &pose_idx in frame_order {
        let Some(pose) = dataset.poses.get(pose_idx) else {
            continue;
        };
        let position = pose.pose.vec();
        if position.x.is_finite() && position.y.is_finite() && position.z.is_finite() {
            center += position;
            count += 1;
        }
    }
    if count == 0 {
        return 1.0;
    }
    center /= count as f32;

    let mut radius = 0.0f32;
    for &pose_idx in frame_order {
        let Some(pose) = dataset.poses.get(pose_idx) else {
            continue;
        };
        let position = pose.pose.vec();
        if position.x.is_finite() && position.y.is_finite() && position.z.is_finite() {
            radius = radius.max(position.distance(center));
        }
    }

    let scene_scale = radius * 1.1;
    if scene_scale.is_finite() && scene_scale > 1e-8 {
        scene_scale
    } else {
        1.0
    }
}

fn build_training_run(
    splats: HostSplats,
    report: WgpuTrainingReport,
    elapsed: std::time::Duration,
) -> TrainingRun {
    let final_loss = report.final_loss;
    let gaussian_count = if report.final_gaussian_count == 0 {
        splats.len()
    } else {
        report.final_gaussian_count
    };

    let run = TrainingRun {
        report: TrainingRunReport {
            elapsed,
            training_loop_elapsed: report.training_loop_elapsed,
            final_loss,
            final_step_loss: report.final_step_loss.or(final_loss),
            gaussian_count,
            sh_degree: splats.sh_degree(),
            completed_iterations: report.completed_iterations,
            cancelled: report.cancelled,
            disposition: report.disposition,
            telemetry: Some(report.telemetry.clone()),
        },
        splats,
    };
    store_last_training_telemetry(run.report.telemetry.clone());
    run
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AdamCheckpoint, AdamParameterCheckpoint, TensorCheckpoint, TopologyCheckpoint,
        TrainingCheckpoint, TrainingIdentity, TRAINING_CHECKPOINT_VERSION,
    };

    fn resume_checkpoint(
        identity: TrainingIdentity,
        completed_iterations: usize,
    ) -> TrainingCheckpoint {
        let parameter = AdamParameterCheckpoint {
            moment1: None,
            moment2: None,
            scaling: None,
            step: 0,
        };
        let topology_tensor = TensorCheckpoint {
            shape: vec![0],
            values: Vec::new(),
        };
        TrainingCheckpoint {
            version: TRAINING_CHECKPOINT_VERSION,
            identity,
            completed_iterations,
            latest_loss: None,
            splats: HostSplats::default(),
            optimizer: AdamCheckpoint {
                transforms: parameter.clone(),
                sh_coeffs: parameter.clone(),
                raw_opacities: parameter,
            },
            topology: TopologyCheckpoint {
                grad_2d: topology_tensor.clone(),
                screen_grad_2d: topology_tensor.clone(),
                abs_grad_2d: topology_tensor.clone(),
                abs_pixel_grad_2d: topology_tensor.clone(),
                pixel_coverage: topology_tensor.clone(),
                camera_depth: topology_tensor.clone(),
                grad_color: topology_tensor.clone(),
                num_observations: topology_tensor.clone(),
                visible_observations: topology_tensor.clone(),
                actual_visible_observations: topology_tensor,
                splat_birth_iterations: Vec::new(),
                splat_invisible_windows: Vec::new(),
            },
            frame_shuffle_seed: 17,
            active_sh_degree: 0,
        }
    }

    fn identity() -> TrainingIdentity {
        TrainingIdentity {
            dataset: "dataset".to_string(),
            reconstruction: "reconstruction".to_string(),
            config: "config".to_string(),
        }
    }

    fn invalid_input_message(error: TrainingError) -> String {
        match error {
            TrainingError::InvalidInput(message) => message,
            other => panic!("expected invalid input, got {other:?}"),
        }
    }

    #[test]
    fn resume_identity_mismatches_fail_before_requesting_a_device() {
        let config = TrainingConfig::default();
        let current = identity();

        for (checkpoint_identity, expected) in [
            (
                TrainingIdentity {
                    dataset: "other".to_string(),
                    ..current.clone()
                },
                "checkpoint dataset does not match the current training dataset",
            ),
            (
                TrainingIdentity {
                    reconstruction: "other".to_string(),
                    ..current.clone()
                },
                "checkpoint reconstruction does not match the current sparse reconstruction",
            ),
            (
                TrainingIdentity {
                    config: "other".to_string(),
                    ..current.clone()
                },
                "checkpoint configuration does not match the current training configuration",
            ),
        ] {
            let checkpoint = resume_checkpoint(checkpoint_identity, 7);
            let error = prepare_resume_runtime(
                &config,
                Some(&current),
                Some(&checkpoint),
                None::<fn()>,
                || panic!("default device must not be requested"),
            )
            .unwrap_err();
            assert_eq!(invalid_input_message(error), expected);
        }
    }

    #[test]
    fn resume_requires_identity_and_rejects_a_lower_iteration_target() {
        let current = identity();
        let checkpoint = resume_checkpoint(current.clone(), 7);

        let missing_identity = prepare_resume_runtime(
            &TrainingConfig::default(),
            None,
            Some(&checkpoint),
            None::<fn()>,
            || panic!("device must not be requested"),
        )
        .unwrap_err();
        assert!(matches!(missing_identity, TrainingError::InvalidInput(_)));

        let lower_target = TrainingConfig {
            iterations: 6,
            ..Default::default()
        };
        let error = prepare_resume_runtime(
            &lower_target,
            Some(&current),
            Some(&checkpoint),
            None::<fn()>,
            || panic!("device must not be requested"),
        )
        .unwrap_err();
        assert!(invalid_input_message(error).contains("lower than checkpoint"));
    }

    #[test]
    fn shared_training_device_selection_precedes_the_default_factory() {
        let (start_iteration, device) = prepare_resume_runtime(
            &TrainingConfig::default(),
            None,
            None,
            Some(|| "shared"),
            || panic!("default factory must not run when shared device exists"),
        )
        .unwrap();

        assert_eq!(start_iteration, 0);
        assert_eq!(device, "shared");
    }

    #[test]
    fn train_splats_resume_validation_uses_the_entry_path_before_device_factory() {
        let dataset = TrainingDataset::new(Intrinsics::default());
        let config = TrainingConfig::default();
        let current = identity();

        let cases = [
            (
                Some(current.clone()),
                TrainingIdentity {
                    dataset: "other".to_string(),
                    ..current.clone()
                },
                "checkpoint dataset does not match the current training dataset",
            ),
            (
                Some(current.clone()),
                TrainingIdentity {
                    reconstruction: "other".to_string(),
                    ..current.clone()
                },
                "checkpoint reconstruction does not match the current sparse reconstruction",
            ),
            (
                Some(current.clone()),
                TrainingIdentity {
                    config: "other".to_string(),
                    ..current.clone()
                },
                "checkpoint configuration does not match the current training configuration",
            ),
            (
                None,
                current.clone(),
                "resuming training requires the current training identity",
            ),
        ];

        for (current_identity, checkpoint_identity, expected) in cases {
            let checkpoint = resume_checkpoint(checkpoint_identity, 7);
            let mut options = TrainingOptions::new().with_resume_checkpoint(checkpoint);
            if let Some(current_identity) = current_identity {
                options = options.with_identity(current_identity);
            }

            let error = train_splats_with_device_factory(&dataset, &config, options, || {
                panic!("device factory must not run before resume validation")
            })
            .unwrap_err();
            assert_eq!(invalid_input_message(error), expected);
        }
    }

    #[test]
    fn training_failure_emits_run_failed_as_the_terminal_event() {
        let dataset = TrainingDataset::new(Intrinsics::default());
        let config = TrainingConfig::default();
        let mut events = Vec::new();

        let result = train_splats(
            &dataset,
            &config,
            TrainingOptions::new().with_event_sink(|event| events.push(event)),
        );

        assert!(result.is_err());
        assert!(matches!(events.last(), Some(TrainingEvent::RunFailed(_))));
        assert!(!events
            .iter()
            .any(|event| matches!(event, TrainingEvent::RunCompleted(_))));
    }
}
