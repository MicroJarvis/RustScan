//! WGPU training runtime orchestration.

use std::time::Instant;

use glam::Vec3;

use crate::core::GaussianCamera;
use crate::core::HostSplats;
use crate::training::data::frame_loader::{
    ordered_frame_indices, FrameLoaderOptions, PrefetchFrameLoader,
};
use crate::training::data::init_map::build_initial_splats;
use crate::training::evaluation::scaled_dimensions;
use crate::training::events::{
    emit_training_event, TrainingControl, TrainingEvent, TrainingEventCadence, TrainingEventRoute,
    TrainingIterationProgress, TrainingOptions, TrainingPlanSelected, TrainingRun,
    TrainingRunCancelled, TrainingRunCompleted, TrainingRunFailed, TrainingRunReport,
    TrainingRunStarted, TrainingSnapshotReady,
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

pub fn train_splats(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
    mut options: TrainingOptions<'_>,
) -> Result<TrainingRun, TrainingError> {
    let run_started_at = Instant::now();
    let control = options.control;
    let mut noop = |_event| {};
    let emit_iteration_events = options.on_event.is_some();
    let on_event = options.on_event.as_deref_mut();
    let on_event = on_event.unwrap_or(&mut noop);

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

    let run = match run_training(dataset, config, &control, emit_iteration_events, on_event) {
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

    if run.report.cancelled {
        emit_training_event(
            on_event,
            TrainingEvent::RunCancelled(TrainingRunCancelled {
                completed_iterations: run.report.completed_iterations,
                elapsed: run.report.elapsed,
            }),
        );
    }

    emit_training_event(
        on_event,
        TrainingEvent::RunCompleted(TrainingRunCompleted {
            report: run.report.clone(),
        }),
    );
    Ok(run)
}

fn run_training<F>(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
    control: &TrainingControl,
    emit_iteration_events: bool,
    on_event: &mut F,
) -> Result<TrainingRun, TrainingError>
where
    F: FnMut(TrainingEvent) + ?Sized,
{
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
    let initial_splats = build_initial_splats(dataset, config)?;

    let mut loader = PrefetchFrameLoader::new(
        dataset,
        config,
        FrameLoaderOptions {
            cache_capacity: config.data.frame_cache_capacity,
            prefetch_ahead: config.data.frame_prefetch_ahead,
            rgb_target_size: Some((target_width, target_height)),
        },
    )?;

    let frame_order = ordered_frame_indices(dataset.poses.len(), 1, config.data.frame_shuffle_seed);
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
            let device = GsDevice::default();
            warm_up_training_kernels(
                &initial_splats,
                config,
                &device,
                &cameras,
                &frame_order,
                &mut loader,
                (target_width, target_height),
                scene_scale,
            )
            .await?;
            let mut device_splats =
                host_splats_to_device::<GsDiffBackend>(&initial_splats, &device);
            let sh_coeffs = device_splats.sh_coeffs.val().dims()[1];
            let mut trainer = WgpuTrainer::new(
                config.clone(),
                device.clone(),
                device_splats.num_splats(),
                sh_coeffs,
                scene_scale,
            );
            let mut observer = TrainingEventObserver {
                control,
                cadence: control.cadence(),
                emit_iteration_events,
                started_at,
                on_event,
            };
            let report = trainer
                .train_with_frame_loader(
                    &mut device_splats,
                    &cameras,
                    &frame_order,
                    &mut loader,
                    (target_width, target_height),
                    config.iterations,
                    &mut observer,
                )
                .await?;
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
) -> Result<(), TrainingError> {
    if config.iterations == 0 || cameras.is_empty() || frame_order.is_empty() {
        return Ok(());
    }

    let frame_idx = frame_order[0];
    frame_loader.prefetch_order_window(frame_order, 0)?;
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
            &cameras[0],
            target_img,
            image_dims,
            1,
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

struct TrainingEventObserver<'a, F>
where
    F: FnMut(TrainingEvent) + ?Sized,
{
    control: &'a TrainingControl,
    cadence: TrainingEventCadence,
    emit_iteration_events: bool,
    started_at: Instant,
    on_event: &'a mut F,
}

impl<F> TrainingLoopObserver for TrainingEventObserver<'_, F>
where
    F: FnMut(TrainingEvent) + ?Sized,
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
