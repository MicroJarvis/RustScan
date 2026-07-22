#![allow(clippy::too_many_arguments)]

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use burn::prelude::*;
use burn::tensor::{s, AllocationProperty, Bytes as BurnBytes, DType, Shape, TensorData};
use bytes::Bytes as SharedBytes;

use crate::core::GaussianCamera;
use crate::core::HostSplats;
use crate::training::backward;
use crate::training::data::frame_loader::PrefetchFrameLoader;
use crate::training::reporting::metrics::{ParityLossCurveSample, ParityTopologyMetrics};
use crate::training::reporting::telemetry::{LiteGsOptimizerLrs, LiteGsTrainingTelemetry};
use crate::training::topology::TopologyMutationPlan;
use crate::training::topology::{apply_mutations, plan_mutations, snapshot_for_topology};
use crate::training::topology::{apply_topology_metrics_delta, should_apply_topology_step};
use crate::training::{
    LiteGsPruneMode, TrainingCheckpoint, TrainingCheckpointReady, TrainingCheckpointReason,
    TrainingConfig, TrainingIdentity, TrainingRunDisposition, TRAINING_CHECKPOINT_VERSION,
};
use crate::TrainingError;

use super::backend::{GsBackendBase, GsDevice, GsDiffBackend};
use super::loss::{combined_loss_with_kernel, gaussian_kernel_1d, SsimConfig};
use super::optimizer::{AdamScaled, AdamScaledConfig};
use super::splats::{
    device_splats_to_host, host_splats_to_device, try_device_splats_to_host, DeviceSplats,
};
use super::topology_accum::{accumulate_topology_stats, TopologyAccumulatorSet};

#[derive(Debug, Clone, Default)]
pub struct WgpuTrainingReport {
    pub final_loss: Option<f32>,
    pub final_step_loss: Option<f32>,
    pub final_gaussian_count: usize,
    pub completed_iterations: usize,
    pub cancelled: bool,
    pub disposition: TrainingRunDisposition,
    pub training_loop_elapsed: Duration,
    pub telemetry: LiteGsTrainingTelemetry,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TrainingIterationMetrics {
    pub iteration: usize,
    pub loss: f32,
    pub gaussian_count: usize,
}

fn validate_loss_value(loss: f32, iteration: usize) -> Result<f32, TrainingError> {
    if loss.is_finite() {
        Ok(loss)
    } else {
        Err(TrainingError::TrainingFailed(format!(
            "non-finite loss {loss} at iteration {iteration}"
        )))
    }
}

fn record_completed_step(
    report: &mut WgpuTrainingReport,
    iteration: usize,
    gaussian_count: usize,
    loss: f32,
) {
    report.completed_iterations = iteration;
    report.final_gaussian_count = gaussian_count;
    report.final_loss = Some(loss);
    report.final_step_loss = Some(loss);
}

pub(crate) trait TrainingLoopObserver {
    fn should_cancel(&self) -> bool {
        false
    }

    fn should_emit_progress(&self, _iteration: usize) -> bool {
        false
    }

    fn should_emit_snapshot(&self, _iteration: usize) -> bool {
        false
    }

    fn checkpoint_reason(&self, _iteration: usize) -> Option<TrainingCheckpointReason> {
        None
    }

    fn checkpoint_identity(&self) -> Option<&TrainingIdentity> {
        None
    }

    fn on_iteration(&mut self, _metrics: TrainingIterationMetrics) {}

    fn on_snapshot(&mut self, _metrics: TrainingIterationMetrics, _splats: HostSplats) {}

    fn on_checkpoint(&mut self, _ready: TrainingCheckpointReady) -> Result<(), TrainingError> {
        Ok(())
    }
}

pub struct WgpuTrainer {
    config: TrainingConfig,
    optimizer: AdamScaled<GsBackendBase>,
    device: GsDevice,
    grad_2d_accum: Tensor<GsBackendBase, 1>,
    screen_grad_2d_accum: Tensor<GsBackendBase, 1>,
    abs_grad_2d_accum: Tensor<GsBackendBase, 1>,
    abs_pixel_grad_2d_accum: Tensor<GsBackendBase, 1>,
    pixel_coverage_accum: Tensor<GsBackendBase, 1>,
    camera_depth_accum: Tensor<GsBackendBase, 1>,
    grad_color_accum: Tensor<GsBackendBase, 1>,
    num_observations: Tensor<GsBackendBase, 1>,
    visible_observations: Tensor<GsBackendBase, 1>,
    actual_visible_observations: Tensor<GsBackendBase, 1>,
    splat_birth_iterations: Vec<usize>,
    splat_invisible_windows: Vec<usize>,
    ssim_config: SsimConfig,
    ssim_kernel: Tensor<GsDiffBackend, 1>,
    telemetry: LiteGsTrainingTelemetry,
    position_lr_scene_scale: f32,
    optimizer_lr_state: Option<OptimizerLrState>,
}

#[derive(Clone)]
struct SharedTargetImageBytes {
    data: Arc<Vec<f32>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct OptimizerLrState {
    sh_coeffs: usize,
    pos_lr: f32,
    rotation_lr: f32,
    scale_lr: f32,
    opacity_lr: f32,
    color_lr: f32,
    color_rest_lr: f32,
}

impl AsRef<[u8]> for SharedTargetImageBytes {
    fn as_ref(&self) -> &[u8] {
        bytemuck::cast_slice(self.data.as_slice())
    }
}

fn target_image_tensor_data(
    target_image: &Arc<Vec<f32>>,
    image_dims: (usize, usize),
) -> TensorData {
    let (width, height) = image_dims;
    let shared = SharedBytes::from_owner(SharedTargetImageBytes {
        data: Arc::clone(target_image),
    });
    TensorData::from_bytes(
        BurnBytes::from_shared(shared, AllocationProperty::Native),
        Shape::new([height, width, 3]),
        DType::F32,
    )
}

pub(crate) fn target_image_tensor(
    target_image: &Arc<Vec<f32>>,
    image_dims: (usize, usize),
    device: &GsDevice,
) -> Tensor<GsDiffBackend, 3> {
    Tensor::<GsDiffBackend, 3>::from_data(
        target_image_tensor_data(target_image, image_dims),
        device,
    )
}

impl WgpuTrainer {
    pub fn new(
        config: TrainingConfig,
        device: GsDevice,
        initial_splats: usize,
        sh_coeffs: usize,
        scene_scale: f32,
    ) -> Self {
        let mut optimizer = AdamScaled::<GsBackendBase>::new(AdamScaledConfig {
            lr: 1.0,
            eps: 1e-15,
            ..AdamScaledConfig::default()
        });
        let position_lr_scene_scale = effective_position_lr_scene_scale(&config, scene_scale);
        let position_lr = config.optimizer.lr_position * position_lr_scene_scale;

        let transform_scales = Tensor::<GsBackendBase, 2>::from_data(
            TensorData::from([[
                position_lr,
                position_lr,
                position_lr,
                config.optimizer.lr_rotation,
                config.optimizer.lr_rotation,
                config.optimizer.lr_rotation,
                config.optimizer.lr_rotation,
                config.optimizer.lr_scale,
                config.optimizer.lr_scale,
                config.optimizer.lr_scale,
            ]]),
            &device,
        );
        let sh_scale_values = sh_lr_values(
            sh_coeffs,
            config.optimizer.lr_color,
            config.optimizer.lr_color_rest,
        );
        let sh_scales = Tensor::<GsBackendBase, 3>::from_data(
            TensorData::new(sh_scale_values, [1, sh_coeffs.max(1), 1]),
            &device,
        );
        let opacity_scales =
            Tensor::<GsBackendBase, 1>::from_floats([config.optimizer.lr_opacity], &device);

        optimizer.set_transform_scaling(transform_scales);
        optimizer.set_sh_scaling(sh_scales);
        optimizer.set_opacity_scaling(opacity_scales);
        let ssim_config = SsimConfig::default();
        let ssim_kernel = gaussian_kernel_1d::<GsDiffBackend>(&ssim_config, &device);

        let telemetry =
            initial_training_telemetry(&config, initial_splats, position_lr_scene_scale);

        Self {
            config,
            optimizer,
            device: device.clone(),
            grad_2d_accum: Tensor::zeros([initial_splats], &device),
            screen_grad_2d_accum: Tensor::zeros([initial_splats], &device),
            abs_grad_2d_accum: Tensor::zeros([initial_splats], &device),
            abs_pixel_grad_2d_accum: Tensor::zeros([initial_splats], &device),
            pixel_coverage_accum: Tensor::zeros([initial_splats], &device),
            camera_depth_accum: Tensor::zeros([initial_splats], &device),
            grad_color_accum: Tensor::zeros([initial_splats], &device),
            num_observations: Tensor::zeros([initial_splats], &device),
            visible_observations: Tensor::zeros([initial_splats], &device),
            actual_visible_observations: Tensor::zeros([initial_splats], &device),
            splat_birth_iterations: vec![0; initial_splats],
            splat_invisible_windows: vec![0; initial_splats],
            ssim_config,
            ssim_kernel,
            telemetry,
            position_lr_scene_scale,
            optimizer_lr_state: None,
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub async fn checkpoint(
        &self,
        splats: &DeviceSplats<GsDiffBackend>,
        identity: TrainingIdentity,
        completed_iterations: usize,
        latest_loss: Option<f32>,
    ) -> Result<TrainingCheckpoint, TrainingError> {
        let host_splats = try_device_splats_to_host(splats).await?;
        let active_sh_degree =
            self.active_sh_degree_at(completed_iterations, splats.sh_degree) as usize;
        let topology = TopologyAccumulatorSet {
            grad_2d: self.grad_2d_accum.clone(),
            screen_grad_2d: self.screen_grad_2d_accum.clone(),
            abs_grad_2d: self.abs_grad_2d_accum.clone(),
            abs_pixel_grad_2d: self.abs_pixel_grad_2d_accum.clone(),
            pixel_coverage: self.pixel_coverage_accum.clone(),
            camera_depth: self.camera_depth_accum.clone(),
            grad_color: self.grad_color_accum.clone(),
            num_observations: self.num_observations.clone(),
            visible_observations: self.visible_observations.clone(),
            actual_visible_observations: self.actual_visible_observations.clone(),
        }
        .checkpoint(&self.splat_birth_iterations, &self.splat_invisible_windows)
        .await?;
        let checkpoint = TrainingCheckpoint {
            version: TRAINING_CHECKPOINT_VERSION,
            identity,
            completed_iterations,
            latest_loss,
            active_sh_degree,
            splats: host_splats,
            optimizer: self.optimizer.checkpoint().await?,
            topology,
            frame_shuffle_seed: self.config.data.frame_shuffle_seed,
        };
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub async fn from_checkpoint(
        mut config: TrainingConfig,
        device: GsDevice,
        scene_scale: f32,
        checkpoint: &TrainingCheckpoint,
    ) -> Result<(Self, DeviceSplats<GsDiffBackend>), TrainingError> {
        checkpoint.validate()?;
        config.validate()?;

        config.data.frame_shuffle_seed = checkpoint.frame_shuffle_seed;
        let splat_count = checkpoint.splats.len();
        let sh_coeffs = checkpoint.splats.sh_coeffs_row_width() / 3;
        let splats = host_splats_to_device::<GsDiffBackend>(&checkpoint.splats, &device);
        let mut trainer = Self::new(config, device.clone(), splat_count, sh_coeffs, scene_scale);
        trainer
            .optimizer
            .restore(&checkpoint.optimizer, &splats, &device)?;
        let topology =
            TopologyAccumulatorSet::from_checkpoint(&checkpoint.topology, splat_count, &device)?;
        trainer.grad_2d_accum = topology.grad_2d;
        trainer.screen_grad_2d_accum = topology.screen_grad_2d;
        trainer.abs_grad_2d_accum = topology.abs_grad_2d;
        trainer.abs_pixel_grad_2d_accum = topology.abs_pixel_grad_2d;
        trainer.pixel_coverage_accum = topology.pixel_coverage;
        trainer.camera_depth_accum = topology.camera_depth;
        trainer.grad_color_accum = topology.grad_color;
        trainer.num_observations = topology.num_observations;
        trainer.visible_observations = topology.visible_observations;
        trainer.actual_visible_observations = topology.actual_visible_observations;
        trainer.splat_birth_iterations = checkpoint.topology.splat_birth_iterations.clone();
        trainer.splat_invisible_windows = checkpoint.topology.splat_invisible_windows.clone();
        trainer.telemetry.active_sh_degree = Some(checkpoint.active_sh_degree);

        Ok((trainer, splats))
    }

    fn lr_at(&self, initial: f32, final_value: f32, iteration: usize) -> f32 {
        if final_value <= 0.0 || final_value >= initial || self.config.iterations == 0 {
            return initial;
        }

        let decay_iterations = self
            .config
            .optimizer
            .lr_decay_iterations
            .unwrap_or(self.config.iterations)
            .max(1);
        let t = (iteration.min(decay_iterations) as f32) / (decay_iterations as f32);
        initial * ((final_value / initial).ln() * t).exp()
    }

    fn position_lr_at(&self, iteration: usize) -> f32 {
        self.lr_at(
            self.config.optimizer.lr_position,
            self.config.optimizer.lr_pos_final,
            iteration,
        ) * self.position_lr_scene_scale
    }

    fn scale_lr_at(&self, iteration: usize) -> f32 {
        self.lr_at(
            self.config.optimizer.lr_scale,
            self.config.optimizer.lr_scale_final,
            iteration,
        )
    }

    fn rotation_lr_at(&self, iteration: usize) -> f32 {
        self.lr_at(
            self.config.optimizer.lr_rotation,
            self.config.optimizer.lr_rotation_final,
            iteration,
        )
    }

    fn opacity_lr_at(&self, iteration: usize) -> f32 {
        self.lr_at(
            self.config.optimizer.lr_opacity,
            self.config.optimizer.lr_opacity_final,
            iteration,
        )
    }

    fn color_lr_at(&self, iteration: usize) -> f32 {
        self.lr_at(
            self.config.optimizer.lr_color,
            self.config.optimizer.lr_color_final,
            iteration,
        )
    }

    fn color_rest_lr_at(&self, iteration: usize) -> f32 {
        self.lr_at(
            self.config.optimizer.lr_color_rest,
            self.config.optimizer.lr_color_rest_final,
            iteration,
        )
    }

    fn active_sh_degree_at(&self, iteration: usize, storage_sh_degree: u32) -> u32 {
        let scheduled = iteration.saturating_sub(1) / 1000;
        (scheduled as u32).min(storage_sh_degree)
    }

    fn update_optimizer_lrs(&mut self, iteration: usize, sh_coeffs: usize) {
        let pos_lr = self.position_lr_at(iteration);
        let rotation_lr = self.rotation_lr_at(iteration);
        let scale_lr = self.scale_lr_at(iteration);
        let opacity_lr = self.opacity_lr_at(iteration);
        let color_lr = self.color_lr_at(iteration);
        let color_rest_lr = self.color_rest_lr_at(iteration);
        let lr_state = OptimizerLrState {
            sh_coeffs,
            pos_lr,
            rotation_lr,
            scale_lr,
            opacity_lr,
            color_lr,
            color_rest_lr,
        };

        if self.optimizer_lr_state == Some(lr_state) {
            return;
        }

        let transform_scales = Tensor::<GsBackendBase, 2>::from_data(
            TensorData::from([[
                pos_lr,
                pos_lr,
                pos_lr,
                rotation_lr,
                rotation_lr,
                rotation_lr,
                rotation_lr,
                scale_lr,
                scale_lr,
                scale_lr,
            ]]),
            &self.device,
        );
        let sh_scale_values = sh_lr_values(sh_coeffs, color_lr, color_rest_lr);
        let sh_scales = Tensor::<GsBackendBase, 3>::from_data(
            TensorData::new(sh_scale_values, [1, sh_coeffs.max(1), 1]),
            &self.device,
        );
        let opacity_scales = Tensor::<GsBackendBase, 1>::from_floats([opacity_lr], &self.device);

        self.optimizer.set_transform_scaling(transform_scales);
        self.optimizer.set_sh_scaling(sh_scales);
        self.optimizer.set_opacity_scaling(opacity_scales);
        self.optimizer_lr_state = Some(lr_state);
        self.telemetry.learning_rates.xyz = Some(pos_lr);
        self.telemetry.learning_rates.sh_0 = Some(color_lr);
        self.telemetry.learning_rates.sh_rest = Some(color_rest_lr);
        self.telemetry.learning_rates.opacity = Some(opacity_lr);
        self.telemetry.learning_rates.scale = Some(scale_lr);
        self.telemetry.learning_rates.rot = Some(rotation_lr);
    }

    pub async fn train_step(
        &mut self,
        splats: &mut DeviceSplats<GsDiffBackend>,
        camera: &GaussianCamera,
        target_img: Tensor<GsDiffBackend, 3>,
        image_dims: (usize, usize),
        iteration: usize,
        frame_count: usize,
        collect_topology_stats: bool,
    ) -> Result<f32, TrainingError> {
        let profile_step = log::log_enabled!(log::Level::Debug)
            && (iteration <= 3 || iteration.is_multiple_of(100));
        let step_started_at = Instant::now();
        let (width, height) = image_dims;
        let background = [0.0, 0.0, 0.0];
        let target_ready_elapsed = step_started_at.elapsed();

        let active_sh_degree = self.active_sh_degree_at(iteration, splats.sh_degree);
        self.telemetry.active_sh_degree = Some(active_sh_degree as usize);
        let rendered = backward::render_splats_with_visibility_active_sh(
            splats,
            active_sh_degree,
            camera,
            (width as u32, height as u32),
            background,
            self.raster_cov_blur_at(iteration, frame_count),
        )
        .await;
        let forward_elapsed = if profile_step {
            let started = Instant::now();
            let _ = rendered
                .image
                .clone()
                .mean()
                .into_scalar_async()
                .await
                .expect("render profile sync");
            Some(started.elapsed())
        } else {
            None
        };
        let pred_rgb = rendered.image.slice(s![.., .., 0..3]);
        let dynamic_mask = self.dynamic_loss_mask_at(iteration, frame_count);
        let loss = combined_loss_with_kernel(
            pred_rgb,
            target_img,
            self.config.loss.loss_l1_weight as f64,
            self.config.loss.loss_ssim_weight as f64,
            self.config.loss.loss_gradient_weight as f64,
            self.config.loss.loss_robust_delta as f64,
            self.config.loss.loss_outlier_threshold as f64,
            self.config.loss.loss_outlier_weight as f64,
            dynamic_mask.map(|mask| mask.0).unwrap_or(0.0) as f64,
            dynamic_mask.map(|mask| mask.1).unwrap_or(0.0) as f64,
            dynamic_mask.map(|mask| mask.2).unwrap_or(1.0) as f64,
            &self.ssim_config,
            self.ssim_kernel.clone(),
        );
        let loss_sync_started_at = Instant::now();
        let loss_value =
            loss.clone().into_scalar_async().await.map_err(|err| {
                TrainingError::TrainingFailed(format!("failed to read loss: {err}"))
            })?;
        let loss_value = validate_loss_value(loss_value, iteration)?;
        let loss_elapsed = loss_sync_started_at.elapsed();
        let mut grads = loss.backward();

        let transforms_grad = splats
            .transforms
            .grad_remove(&mut grads)
            .unwrap_or_else(|| splats.transforms.val().inner().zeros_like());
        let sh_grad = splats
            .sh_coeffs
            .grad_remove(&mut grads)
            .unwrap_or_else(|| splats.sh_coeffs.val().inner().zeros_like());
        let opacity_grad = splats
            .raw_opacities
            .grad_remove(&mut grads)
            .unwrap_or_else(|| splats.raw_opacities.val().inner().zeros_like());
        let screen_grad_stats = rendered
            .screen_grad_stats
            .grad_remove(&mut grads)
            .unwrap_or_else(|| {
                Tensor::<GsBackendBase, 2>::zeros([splats.num_splats(), 7], &self.device)
            });
        let backward_elapsed = if profile_step {
            let started = Instant::now();
            let _ = transforms_grad
                .clone()
                .abs()
                .mean()
                .into_scalar_async()
                .await
                .expect("backward profile sync");
            Some(started.elapsed())
        } else {
            None
        };

        // Brush keeps a strong gradient-validation path; mirror that observability here
        // so we can quickly spot silent no-op training regressions.
        let should_log_diagnostics = log::log_enabled!(log::Level::Debug)
            && (iteration <= 3 || iteration.is_multiple_of(100));
        let grad_transforms_for_diag = if should_log_diagnostics {
            Some(transforms_grad.clone())
        } else {
            None
        };
        let grad_sh_for_diag = if should_log_diagnostics {
            Some(sh_grad.clone())
        } else {
            None
        };
        let grad_opacity_for_diag = if should_log_diagnostics {
            Some(opacity_grad.clone())
        } else {
            None
        };
        let prev_transforms = if should_log_diagnostics {
            Some(splats.transforms.val().inner())
        } else {
            None
        };
        let prev_sh = if should_log_diagnostics {
            Some(splats.sh_coeffs.val().inner())
        } else {
            None
        };
        let prev_opacity = if should_log_diagnostics {
            Some(splats.raw_opacities.val().inner())
        } else {
            None
        };

        self.update_optimizer_lrs(
            iteration.saturating_sub(1),
            splats.sh_coeffs.val().dims()[1],
        );
        if collect_topology_stats {
            self.accumulate_gradients(
                &transforms_grad,
                &screen_grad_stats,
                &sh_grad,
                &rendered.visible,
                self.uses_visibility_pruning(),
                self.collects_actual_visibility_diagnostics(),
            );
        }
        self.optimizer
            .step_device_splats(splats, transforms_grad, sh_grad, opacity_grad);
        let optimizer_elapsed = if profile_step {
            let started = Instant::now();
            let _ = splats
                .transforms
                .val()
                .inner()
                .abs()
                .mean()
                .into_scalar_async()
                .await
                .expect("optimizer profile sync");
            Some(started.elapsed())
        } else {
            None
        };

        if profile_step {
            log::debug!(
                "WGPU train profile step {} | target={:.3}ms | forward_sync={:.3}ms | loss_sync={:.3}ms | backward_sync={:.3}ms | optimizer_sync={:.3}ms | total_so_far={:.3}ms",
                iteration,
                target_ready_elapsed.as_secs_f64() * 1000.0,
                forward_elapsed.unwrap_or_default().as_secs_f64() * 1000.0,
                loss_elapsed.as_secs_f64() * 1000.0,
                backward_elapsed.unwrap_or_default().as_secs_f64() * 1000.0,
                optimizer_elapsed.unwrap_or_default().as_secs_f64() * 1000.0,
                step_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }

        if should_log_diagnostics {
            let grad_transforms_mean_abs = grad_transforms_for_diag
                .expect("transforms grad for diagnostics")
                .abs()
                .mean()
                .into_scalar_async()
                .await
                .expect("transforms grad mean");
            let grad_sh_mean_abs = grad_sh_for_diag
                .expect("sh grad for diagnostics")
                .abs()
                .mean()
                .into_scalar_async()
                .await
                .expect("sh grad mean");
            let grad_opacity_mean_abs = grad_opacity_for_diag
                .expect("opacity grad for diagnostics")
                .abs()
                .mean()
                .into_scalar_async()
                .await
                .expect("opacity grad mean");

            let delta_transforms_mean_abs = (splats.transforms.val().inner()
                - prev_transforms.expect("prev transforms for diagnostics"))
            .abs()
            .mean()
            .into_scalar_async()
            .await
            .expect("transforms delta mean");
            let delta_sh_mean_abs = (splats.sh_coeffs.val().inner()
                - prev_sh.expect("prev sh for diagnostics"))
            .abs()
            .mean()
            .into_scalar_async()
            .await
            .expect("sh delta mean");
            let delta_opacity_mean_abs = (splats.raw_opacities.val().inner()
                - prev_opacity.expect("prev opacity for diagnostics"))
            .abs()
            .mean()
            .into_scalar_async()
            .await
            .expect("opacity delta mean");

            log::info!(
                "WGPU train diagnostics step {} | grad_mean_abs: transforms={:.6e}, sh={:.6e}, opacity={:.6e} | delta_mean_abs: transforms={:.6e}, sh={:.6e}, opacity={:.6e}",
                iteration,
                grad_transforms_mean_abs,
                grad_sh_mean_abs,
                grad_opacity_mean_abs,
                delta_transforms_mean_abs,
                delta_sh_mean_abs,
                delta_opacity_mean_abs,
            );
        }

        if self.should_apply_topology(iteration, frame_count) {
            self.apply_topology_mutations(splats, iteration, frame_count)
                .await;
        }

        Ok(loss_value)
    }

    pub(crate) async fn train_with_frame_loader(
        &mut self,
        splats: &mut DeviceSplats<GsDiffBackend>,
        cameras: &[GaussianCamera],
        frame_order: &[usize],
        frame_loader: &mut PrefetchFrameLoader,
        image_dims: (usize, usize),
        start_iteration: usize,
        num_iterations: usize,
        observer: &mut dyn TrainingLoopObserver,
    ) -> Result<WgpuTrainingReport, TrainingError> {
        if cameras.is_empty() || cameras.len() != frame_order.len() {
            return Err(TrainingError::InvalidInput(format!(
                "training frame order length ({}) must match camera count ({}) and be non-empty",
                frame_order.len(),
                cameras.len()
            )));
        }

        let mut report = WgpuTrainingReport {
            completed_iterations: start_iteration,
            final_gaussian_count: splats.num_splats(),
            ..Default::default()
        };
        self.telemetry.topology.total_epochs =
            Some(training_epoch_count(num_iterations, cameras.len()));
        let collect_topology_stats =
            training_uses_topology_stats(&self.config, num_iterations, cameras.len());
        let mut target_tensor_cache = HashMap::<usize, Tensor<GsDiffBackend, 3>>::new();
        let mut target_tensor_lru = VecDeque::<usize>::new();
        let target_tensor_cache_capacity = self.config.data.frame_cache_capacity.max(1);
        let training_loop_started_at = Instant::now();

        for zero_based in start_iteration..num_iterations {
            if observer.should_cancel() {
                report.cancelled = true;
                report.disposition = TrainingRunDisposition::Cancelled;
                break;
            }

            let sample_idx = zero_based % cameras.len();
            let frame_idx = frame_order[sample_idx];
            frame_loader.prefetch_order_window(frame_order, sample_idx)?;
            let decoded = frame_loader.get(frame_idx)?;
            let target_img = match target_tensor_cache.get(&frame_idx).cloned() {
                Some(cached) => {
                    touch_target_tensor_cache(&mut target_tensor_lru, frame_idx);
                    cached
                }
                None => {
                    let target_image = decoded.target_rgb.clone().ok_or_else(|| {
                        TrainingError::TrainingFailed(format!(
                            "frame loader did not prepare target_rgb for frame {frame_idx}"
                        ))
                    })?;
                    let tensor = target_image_tensor(&target_image, image_dims, &self.device);
                    target_tensor_cache.insert(frame_idx, tensor.clone());
                    touch_target_tensor_cache(&mut target_tensor_lru, frame_idx);
                    while target_tensor_cache.len() > target_tensor_cache_capacity {
                        if let Some(evicted) = target_tensor_lru.pop_front() {
                            target_tensor_cache.remove(&evicted);
                        }
                    }
                    tensor
                }
            };

            let iteration_idx = zero_based + 1;
            let emit_progress = observer.should_emit_progress(iteration_idx);
            let emit_snapshot = observer.should_emit_snapshot(iteration_idx);
            let should_log_step = iteration_idx.is_multiple_of(100);
            let loss = self
                .train_step(
                    splats,
                    &cameras[sample_idx],
                    target_img,
                    image_dims,
                    iteration_idx,
                    cameras.len(),
                    collect_topology_stats,
                )
                .await?;
            record_completed_step(&mut report, iteration_idx, splats.num_splats(), loss);
            self.record_loss_sample(
                iteration_idx,
                frame_idx,
                loss,
                should_log_step || iteration_idx == num_iterations,
            );
            let metrics = TrainingIterationMetrics {
                iteration: iteration_idx,
                loss,
                gaussian_count: splats.num_splats(),
            };
            if emit_progress {
                observer.on_iteration(metrics);
            }
            if emit_snapshot {
                let host = device_splats_to_host(splats).await;
                observer.on_snapshot(metrics, host);
            }
            if should_log_step {
                log::info!(
                    "WGPU training step {} | loss={:.6} | splats={}",
                    iteration_idx,
                    loss,
                    splats.num_splats()
                );
            }

            if observer.should_cancel() {
                report.cancelled = true;
                report.disposition = TrainingRunDisposition::Cancelled;
                break;
            }

            if let Some(reason) = observer.checkpoint_reason(iteration_idx) {
                let identity = observer.checkpoint_identity().cloned().ok_or_else(|| {
                    TrainingError::InvalidInput(
                        "checkpointing training requires the current training identity".to_string(),
                    )
                })?;
                let checkpoint = self
                    .checkpoint(splats, identity, iteration_idx, Some(loss))
                    .await?;
                observer.on_checkpoint(TrainingCheckpointReady {
                    iteration: iteration_idx,
                    reason,
                    checkpoint,
                })?;
                if reason == TrainingCheckpointReason::Pause {
                    report.disposition = TrainingRunDisposition::Paused;
                    break;
                }
            }
        }

        report.training_loop_elapsed = training_loop_started_at.elapsed();
        self.finish_report(&mut report);
        Ok(report)
    }

    fn should_apply_topology(&self, iteration: usize, frame_count: usize) -> bool {
        should_apply_topology_step(&self.config, iteration.max(1), frame_count)
    }

    fn accumulate_gradients(
        &mut self,
        transforms_grad: &Tensor<GsBackendBase, 2>,
        screen_grad_stats: &Tensor<GsBackendBase, 2>,
        sh_grad: &Tensor<GsBackendBase, 3>,
        visible: &Tensor<GsDiffBackend, 1>,
        use_actual_visibility: bool,
        collect_actual_visibility_diagnostics: bool,
    ) {
        // This is the post-projection transform gradient, not the per-pixel
        // screen-space mean gradient required by AbsGS-style densification. The
        // fused kernel preserves the old statistics while avoiding a long chain
        // of tiny Burn tensor ops after every backward pass.
        let accum = TopologyAccumulatorSet {
            grad_2d: self.grad_2d_accum.clone(),
            screen_grad_2d: self.screen_grad_2d_accum.clone(),
            abs_grad_2d: self.abs_grad_2d_accum.clone(),
            abs_pixel_grad_2d: self.abs_pixel_grad_2d_accum.clone(),
            pixel_coverage: self.pixel_coverage_accum.clone(),
            camera_depth: self.camera_depth_accum.clone(),
            grad_color: self.grad_color_accum.clone(),
            num_observations: self.num_observations.clone(),
            visible_observations: self.visible_observations.clone(),
            actual_visible_observations: self.actual_visible_observations.clone(),
        };
        let updated = accumulate_topology_stats(
            transforms_grad.clone(),
            screen_grad_stats.clone(),
            sh_grad.clone(),
            visible.clone().inner(),
            accum,
            use_actual_visibility,
            collect_actual_visibility_diagnostics,
        );

        self.grad_2d_accum = updated.grad_2d;
        self.screen_grad_2d_accum = updated.screen_grad_2d;
        self.abs_grad_2d_accum = updated.abs_grad_2d;
        self.abs_pixel_grad_2d_accum = updated.abs_pixel_grad_2d;
        self.pixel_coverage_accum = updated.pixel_coverage;
        self.camera_depth_accum = updated.camera_depth;
        self.grad_color_accum = updated.grad_color;
        self.num_observations = updated.num_observations;
        self.visible_observations = updated.visible_observations;
        self.actual_visible_observations = updated.actual_visible_observations;
    }

    fn uses_visibility_pruning(&self) -> bool {
        matches!(
            self.config.litegs.pruning.prune_mode,
            LiteGsPruneMode::Threshold
        )
    }

    fn collects_actual_visibility_diagnostics(&self) -> bool {
        self.config.litegs.pruning.prune_visibility_dry_run
            || self.uses_visibility_pruning()
            || matches!(
                self.config.litegs.pruning.prune_mode,
                LiteGsPruneMode::VisibilityWeight
            )
    }

    fn raster_cov_blur_at(&self, iteration: usize, frame_count: usize) -> f32 {
        let Some(final_blur) = self.config.raster.raster_cov_blur_final else {
            return self.config.raster.raster_cov_blur;
        };
        let Some(start_epoch) = self.config.raster.raster_cov_blur_final_after_epoch.or(self
            .config
            .litegs
            .topology
            .topology_freeze_after_epoch)
        else {
            return self.config.raster.raster_cov_blur;
        };
        if frame_count == 0 {
            return self.config.raster.raster_cov_blur;
        }
        let completed_epoch = iteration.saturating_sub(1) / frame_count;
        if completed_epoch >= start_epoch {
            final_blur
        } else {
            self.config.raster.raster_cov_blur
        }
    }

    fn dynamic_loss_mask_at(
        &self,
        iteration: usize,
        frame_count: usize,
    ) -> Option<(f32, f32, f32)> {
        if self.config.loss.loss_dynamic_mask_threshold_high
            <= self.config.loss.loss_dynamic_mask_threshold_low
            || self.config.loss.loss_dynamic_mask_min_weight >= 1.0
            || frame_count == 0
        {
            return None;
        }
        let start_epoch = self.config.loss.loss_dynamic_mask_start_epoch.or(self
            .config
            .litegs
            .topology
            .topology_freeze_after_epoch)?;
        let completed_epoch = iteration.saturating_sub(1) / frame_count;
        if completed_epoch < start_epoch {
            return None;
        }
        Some((
            self.config.loss.loss_dynamic_mask_threshold_low,
            self.config.loss.loss_dynamic_mask_threshold_high,
            self.config.loss.loss_dynamic_mask_min_weight,
        ))
    }

    async fn apply_topology_mutations(
        &mut self,
        splats: &mut DeviceSplats<GsDiffBackend>,
        iteration: usize,
        frame_count: usize,
    ) {
        let mut snapshot = snapshot_for_topology(
            splats,
            &self.grad_2d_accum,
            &self.screen_grad_2d_accum,
            &self.abs_grad_2d_accum,
            &self.abs_pixel_grad_2d_accum,
            &self.pixel_coverage_accum,
            &self.camera_depth_accum,
            &self.grad_color_accum,
            &self.num_observations,
            &self.visible_observations,
            self.collects_actual_visibility_diagnostics()
                .then_some(&self.actual_visible_observations),
        )
        .await;
        self.update_topology_visibility_state(
            snapshot.splats.len(),
            &snapshot.visible_observations,
            iteration,
        );
        snapshot.splat_ages = self.splat_ages_at(iteration, snapshot.splats.len());
        snapshot.invisible_windows = self
            .splat_invisible_windows
            .iter()
            .copied()
            .take(snapshot.splats.len())
            .collect();
        let plan = plan_mutations(&snapshot, &self.config, iteration, frame_count);
        if let Some(sample) = plan.telemetry_sample.clone() {
            log::info!(
                "Topology diagnostics | iter={} | epoch={:?} | splats={} | growth={} | clone={} | split={} | prune={} | large_low_grad={}/{} ({:.3}) | low_vis={} | near_low_vis={} | high_opacity_low_vis={} | vis_prune_dry_run={}",
                sample.iteration,
                sample.completed_epoch,
                sample.gaussian_count,
                sample.growth_candidates,
                sample.clone_candidates,
                sample.split_candidates,
                sample.prune_candidates,
                sample.large_low_grad_count,
                sample.large_splat_count,
                sample.large_low_grad_ratio.unwrap_or(0.0),
                sample.low_visibility_splats,
                sample.near_low_visibility_splats,
                sample.high_opacity_low_visibility_splats,
                sample.visibility_prune_dry_run_candidates,
            );
            self.telemetry.topology.topology_step_samples.push(sample);
        }
        apply_topology_metrics_delta(&mut self.telemetry.topology, plan.aftermath.metrics_delta);
        if plan.mutates_splats() {
            apply_mutations(splats, &snapshot.splats, &plan, &self.device);
            self.remap_topology_visibility_state(&plan, iteration);
        }
        if plan.aftermath.requires_adam_rebuild || plan.aftermath.apply_opacity_reset {
            self.optimizer.reset();
        }
        self.reset_accumulators(
            splats.num_splats(),
            splats.sh_coeffs.val().dims()[1],
            iteration,
        );
    }

    fn reset_accumulators(&mut self, num_splats: usize, sh_coeffs: usize, iteration: usize) {
        self.grad_2d_accum = Tensor::zeros([num_splats], &self.device);
        self.screen_grad_2d_accum = Tensor::zeros([num_splats], &self.device);
        self.abs_grad_2d_accum = Tensor::zeros([num_splats], &self.device);
        self.abs_pixel_grad_2d_accum = Tensor::zeros([num_splats], &self.device);
        self.pixel_coverage_accum = Tensor::zeros([num_splats], &self.device);
        self.camera_depth_accum = Tensor::zeros([num_splats], &self.device);
        self.grad_color_accum = Tensor::zeros([num_splats], &self.device);
        self.num_observations = Tensor::zeros([num_splats], &self.device);
        self.visible_observations = Tensor::zeros([num_splats], &self.device);
        self.actual_visible_observations = Tensor::zeros([num_splats], &self.device);

        self.update_optimizer_lrs(iteration.saturating_sub(1), sh_coeffs);
    }

    fn update_topology_visibility_state(
        &mut self,
        num_splats: usize,
        visible_observations: &[f32],
        iteration: usize,
    ) {
        self.ensure_topology_visibility_state(num_splats, iteration);
        for idx in 0..num_splats {
            let visible = visible_observations
                .get(idx)
                .copied()
                .is_some_and(|value| value.is_finite() && value > 0.0);
            if visible {
                self.splat_invisible_windows[idx] = 0;
            } else {
                self.splat_invisible_windows[idx] =
                    self.splat_invisible_windows[idx].saturating_add(1);
            }
        }
    }

    fn ensure_topology_visibility_state(&mut self, num_splats: usize, iteration: usize) {
        if self.splat_birth_iterations.len() < num_splats {
            self.splat_birth_iterations.resize(num_splats, iteration);
        } else {
            self.splat_birth_iterations.truncate(num_splats);
        }

        if self.splat_invisible_windows.len() < num_splats {
            self.splat_invisible_windows.resize(num_splats, 0);
        } else {
            self.splat_invisible_windows.truncate(num_splats);
        }
    }

    fn splat_ages_at(&self, iteration: usize, num_splats: usize) -> Vec<usize> {
        (0..num_splats)
            .map(|idx| {
                iteration.saturating_sub(
                    self.splat_birth_iterations
                        .get(idx)
                        .copied()
                        .unwrap_or(iteration),
                )
            })
            .collect()
    }

    fn remap_topology_visibility_state(&mut self, plan: &TopologyMutationPlan, iteration: usize) {
        let previous_birth_iterations = self.splat_birth_iterations.clone();
        let previous_invisible_windows = self.splat_invisible_windows.clone();
        let origins = plan.origins();
        self.splat_birth_iterations = origins
            .iter()
            .map(|origin| {
                origin
                    .and_then(|idx| previous_birth_iterations.get(idx).copied())
                    .unwrap_or(iteration)
            })
            .collect();
        self.splat_invisible_windows = origins
            .iter()
            .map(|origin| {
                origin
                    .and_then(|idx| previous_invisible_windows.get(idx).copied())
                    .unwrap_or(0)
            })
            .collect();
    }

    fn record_loss_sample(
        &mut self,
        iteration: usize,
        frame_idx: usize,
        loss: f32,
        keep_curve_sample: bool,
    ) {
        self.telemetry.final_loss = Some(loss);
        self.telemetry.final_step_loss = Some(loss);
        self.telemetry.loss_terms.total = Some(loss);
        if keep_curve_sample {
            self.telemetry
                .loss_curve_samples
                .push(ParityLossCurveSample {
                    iteration,
                    frame_idx,
                    l1: None,
                    ssim: None,
                    depth: None,
                    total: Some(loss),
                    depth_valid_pixels: self.telemetry.depth_valid_pixels,
                });
        }
    }

    fn finish_report(&mut self, report: &mut WgpuTrainingReport) {
        self.telemetry.final_loss = report.final_loss;
        self.telemetry.final_step_loss = report.final_step_loss;
        self.telemetry.topology.final_gaussians = Some(report.final_gaussian_count);
        report.telemetry = self.telemetry.clone();
    }
}

fn touch_target_tensor_cache(lru: &mut VecDeque<usize>, frame_idx: usize) {
    if let Some(position) = lru.iter().position(|cached| *cached == frame_idx) {
        lru.remove(position);
    }
    lru.push_back(frame_idx);
}

fn sh_lr_values(sh_coeffs: usize, dc_lr: f32, rest_lr: f32) -> Vec<f32> {
    let coeffs = sh_coeffs.max(1);
    let mut values = vec![rest_lr; coeffs];
    values[0] = dc_lr;
    values
}

fn effective_position_lr_scene_scale(config: &TrainingConfig, scene_scale: f32) -> f32 {
    if config.optimizer.lr_position_scene_scale && scene_scale.is_finite() && scene_scale > 1e-8 {
        scene_scale
    } else {
        1.0
    }
}

fn training_uses_topology_stats(
    config: &TrainingConfig,
    num_iterations: usize,
    frame_count: usize,
) -> bool {
    if num_iterations == 0 || frame_count == 0 {
        return false;
    }
    (1..=num_iterations).any(|iteration| should_apply_topology_step(config, iteration, frame_count))
}

fn initial_training_telemetry(
    config: &TrainingConfig,
    initial_splats: usize,
    position_lr_scene_scale: f32,
) -> LiteGsTrainingTelemetry {
    LiteGsTrainingTelemetry {
        active_sh_degree: Some(config.litegs.rendering.sh_degree),
        rotation_frozen: config.optimizer.lr_rotation == 0.0,
        learning_rates: LiteGsOptimizerLrs {
            xyz: Some(config.optimizer.lr_position * position_lr_scene_scale),
            sh_0: Some(config.optimizer.lr_color),
            sh_rest: Some(config.optimizer.lr_color_rest),
            opacity: Some(config.optimizer.lr_opacity),
            scale: Some(config.optimizer.lr_scale),
            rot: Some(config.optimizer.lr_rotation),
        },
        topology: ParityTopologyMetrics {
            initialization_gaussians: Some(initial_splats),
            topology_freeze_epoch: config.litegs.topology.topology_freeze_after_epoch,
            ..ParityTopologyMetrics::default()
        },
        ..LiteGsTrainingTelemetry::default()
    }
}

fn training_epoch_count(iterations: usize, frame_count: usize) -> usize {
    iterations
        .checked_div(frame_count)
        .map(|epochs| epochs.max(1))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::engine::splats::host_splats_to_device;
    use crate::training::{TensorCheckpoint, TrainingCheckpoint, TrainingIdentity};
    use burn::module::Param;

    const CHECKPOINT_ITERATIONS: usize = 8;

    fn trainer_checkpoint_config() -> TrainingConfig {
        let mut config = TrainingConfig::default();
        config.data.frame_shuffle_seed = 0x5eed_cafe;
        config.optimizer.lr_pos_final = config.optimizer.lr_position;
        config.optimizer.lr_scale_final = config.optimizer.lr_scale;
        config.optimizer.lr_rotation_final = config.optimizer.lr_rotation;
        config.optimizer.lr_opacity_final = config.optimizer.lr_opacity;
        config.optimizer.lr_color_final = config.optimizer.lr_color;
        config.optimizer.lr_color_rest_final = config.optimizer.lr_color_rest;
        config
    }

    fn trainer_checkpoint_identity() -> TrainingIdentity {
        TrainingIdentity {
            dataset: "dataset-hash".to_string(),
            reconstruction: "reconstruction-hash".to_string(),
            config: "config-hash".to_string(),
        }
    }

    fn trainer_checkpoint_host_splats() -> HostSplats {
        HostSplats::from_components(
            vec![0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2],
            vec![-2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2],
            vec![1.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.8, 0.0, 0.2, 0.0],
            vec![-0.5, 0.0, 0.5],
            (0..36).map(|value| value as f32 * 0.01).collect(),
            1,
        )
        .expect("valid test splats")
    }

    fn install_trainer_checkpoint_state(trainer: &mut WgpuTrainer) {
        let tensor = |values| Tensor::<GsBackendBase, 1>::from_floats(values, &trainer.device);
        trainer.grad_2d_accum = tensor([1.0, 2.0, 3.0]);
        trainer.screen_grad_2d_accum = tensor([11.0, 12.0, 13.0]);
        trainer.abs_grad_2d_accum = tensor([21.0, 22.0, 23.0]);
        trainer.abs_pixel_grad_2d_accum = tensor([31.0, 32.0, 33.0]);
        trainer.pixel_coverage_accum = tensor([41.0, 42.0, 43.0]);
        trainer.camera_depth_accum = tensor([51.0, 52.0, 53.0]);
        trainer.grad_color_accum = tensor([61.0, 62.0, 63.0]);
        trainer.num_observations = tensor([71.0, 72.0, 73.0]);
        trainer.visible_observations = tensor([81.0, 82.0, 83.0]);
        trainer.actual_visible_observations = tensor([91.0, 92.0, 93.0]);
        trainer.splat_birth_iterations = vec![0, 4, 8];
        trainer.splat_invisible_windows = vec![1, 2, 3];
    }

    fn step_trainer_optimizer(trainer: &mut WgpuTrainer, splats: &mut DeviceSplats<GsDiffBackend>) {
        trainer.update_optimizer_lrs(CHECKPOINT_ITERATIONS, 4);
        trainer.optimizer.step_device_splats(
            splats,
            Tensor::ones([3, 10], &trainer.device).mul_scalar(0.1),
            Tensor::ones([3, 4, 3], &trainer.device).mul_scalar(-0.2),
            Tensor::ones([3], &trainer.device).mul_scalar(0.3),
        );
    }

    async fn populated_trainer_checkpoint(
        device: &GsDevice,
    ) -> (TrainingConfig, TrainingCheckpoint) {
        let config = trainer_checkpoint_config();
        let host_splats = trainer_checkpoint_host_splats();
        let mut splats = host_splats_to_device::<GsDiffBackend>(&host_splats, device);
        let mut trainer = WgpuTrainer::new(config.clone(), device.clone(), 3, 4, 2.5);
        install_trainer_checkpoint_state(&mut trainer);
        step_trainer_optimizer(&mut trainer, &mut splats);
        let checkpoint = trainer
            .checkpoint(
                &splats,
                trainer_checkpoint_identity(),
                CHECKPOINT_ITERATIONS,
                Some(0.125),
            )
            .await
            .expect("export trainer checkpoint");
        (config, checkpoint)
    }

    fn assert_topology_tensor(tensor: &TensorCheckpoint, values: [f32; 3]) {
        assert_eq!(tensor.shape, [3]);
        assert_eq!(tensor.values, values);
    }

    async fn trainer_checkpoint_restore_error(
        config: TrainingConfig,
        device: GsDevice,
        checkpoint: &TrainingCheckpoint,
    ) -> TrainingError {
        match WgpuTrainer::from_checkpoint(config, device, 2.5, checkpoint).await {
            Ok(_) => panic!("malformed trainer checkpoint must be rejected"),
            Err(error) => error,
        }
    }

    #[test]
    fn non_finite_loss_is_rejected_before_gradient_work() {
        assert!(validate_loss_value(f32::NAN, 7).is_err());
        assert!(validate_loss_value(f32::INFINITY, 7).is_err());
        assert_eq!(validate_loss_value(0.25, 7).unwrap(), 0.25);
    }

    #[test]
    fn completed_step_always_replaces_reported_final_loss() {
        let mut report = WgpuTrainingReport::default();
        record_completed_step(&mut report, 1, 10, 0.5);
        record_completed_step(&mut report, 2, 11, 0.25);
        assert_eq!(report.final_loss, Some(0.25));
        assert_eq!(report.final_step_loss, Some(0.25));
        assert_eq!(report.completed_iterations, 2);
        assert_eq!(report.final_gaussian_count, 11);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn trainer_checkpoint_roundtrips_splats_optimizer_and_all_topology_state() {
        let device = GsDevice::default();
        let (config, checkpoint) = populated_trainer_checkpoint(&device).await;

        assert_eq!(checkpoint.completed_iterations, CHECKPOINT_ITERATIONS);
        assert_eq!(checkpoint.latest_loss, Some(0.125));
        assert_eq!(checkpoint.frame_shuffle_seed, 0x5eed_cafe);
        assert_eq!(checkpoint.active_sh_degree, 0);
        assert_eq!(checkpoint.splats.len(), 3);
        assert_eq!(checkpoint.optimizer.transforms.step, 1);
        assert_topology_tensor(&checkpoint.topology.grad_2d, [1.0, 2.0, 3.0]);
        assert_topology_tensor(&checkpoint.topology.screen_grad_2d, [11.0, 12.0, 13.0]);
        assert_topology_tensor(&checkpoint.topology.abs_grad_2d, [21.0, 22.0, 23.0]);
        assert_topology_tensor(&checkpoint.topology.abs_pixel_grad_2d, [31.0, 32.0, 33.0]);
        assert_topology_tensor(&checkpoint.topology.pixel_coverage, [41.0, 42.0, 43.0]);
        assert_topology_tensor(&checkpoint.topology.camera_depth, [51.0, 52.0, 53.0]);
        assert_topology_tensor(&checkpoint.topology.grad_color, [61.0, 62.0, 63.0]);
        assert_topology_tensor(&checkpoint.topology.num_observations, [71.0, 72.0, 73.0]);
        assert_topology_tensor(
            &checkpoint.topology.visible_observations,
            [81.0, 82.0, 83.0],
        );
        assert_topology_tensor(
            &checkpoint.topology.actual_visible_observations,
            [91.0, 92.0, 93.0],
        );
        assert_eq!(checkpoint.topology.splat_birth_iterations, [0, 4, 8]);
        assert_eq!(checkpoint.topology.splat_invisible_windows, [1, 2, 3]);

        let (restored, restored_splats) =
            WgpuTrainer::from_checkpoint(config, device, 2.5, &checkpoint)
                .await
                .expect("restore trainer checkpoint");
        assert_eq!(
            device_splats_to_host(&restored_splats).await,
            checkpoint.splats
        );

        let reexported = restored
            .checkpoint(
                &restored_splats,
                checkpoint.identity.clone(),
                checkpoint.completed_iterations,
                checkpoint.latest_loss,
            )
            .await
            .expect("re-export restored trainer checkpoint");
        assert_eq!(reexported, checkpoint);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn trainer_checkpoint_restore_rejects_malformed_topology_before_allocation() {
        let device = GsDevice::default();
        let (config, checkpoint) = populated_trainer_checkpoint(&device).await;

        let mut wrong_rank = checkpoint.clone();
        wrong_rank.topology.grad_2d.shape = vec![1, 3];
        let error = trainer_checkpoint_restore_error(config.clone(), device.clone(), &wrong_rank)
            .await
            .to_string();
        assert!(error.contains("topology.grad_2d must have shape [3]"));

        let mut wrong_len = checkpoint.clone();
        wrong_len.topology.screen_grad_2d.values.pop();
        let error = trainer_checkpoint_restore_error(config.clone(), device.clone(), &wrong_len)
            .await
            .to_string();
        assert!(error.contains("shape expects 3 values, got 2"));

        let mut non_finite = checkpoint.clone();
        non_finite.topology.abs_grad_2d.values[1] = f32::NAN;
        let error = trainer_checkpoint_restore_error(config.clone(), device.clone(), &non_finite)
            .await
            .to_string();
        assert!(error.contains("tensor values must be finite"));

        let mut splat_mismatch = checkpoint;
        splat_mismatch.topology.splat_birth_iterations.pop();
        let error = trainer_checkpoint_restore_error(config, device, &splat_mismatch)
            .await
            .to_string();
        assert!(error.contains("must contain 3 values, got 2"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn trainer_checkpoint_roundtrips_reset_optimizer_state() {
        let device = GsDevice::default();
        let config = trainer_checkpoint_config();
        let host_splats = trainer_checkpoint_host_splats();
        let splats = host_splats_to_device::<GsDiffBackend>(&host_splats, &device);
        let mut trainer = WgpuTrainer::new(config.clone(), device.clone(), 3, 4, 2.5);
        install_trainer_checkpoint_state(&mut trainer);
        trainer.optimizer.reset();
        let checkpoint = trainer
            .checkpoint(
                &splats,
                trainer_checkpoint_identity(),
                CHECKPOINT_ITERATIONS,
                None,
            )
            .await
            .expect("export reset trainer checkpoint");
        assert_eq!(checkpoint.optimizer.transforms.step, 0);
        assert!(checkpoint.optimizer.transforms.moment1.is_none());
        assert!(checkpoint.optimizer.transforms.scaling.is_some());

        let (restored, restored_splats) =
            WgpuTrainer::from_checkpoint(config, device, 2.5, &checkpoint)
                .await
                .expect("restore reset trainer checkpoint");
        let reexported = restored
            .checkpoint(
                &restored_splats,
                checkpoint.identity.clone(),
                checkpoint.completed_iterations,
                checkpoint.latest_loss,
            )
            .await
            .expect("re-export reset trainer checkpoint");
        assert_eq!(reexported.optimizer, checkpoint.optimizer);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn trainer_checkpoint_preserves_decaying_lr_boundary_until_next_iteration() {
        let device = GsDevice::default();
        let mut config = trainer_checkpoint_config();
        config.optimizer.lr_decay_iterations = Some(100);
        config.optimizer.lr_pos_final = config.optimizer.lr_position * 0.1;
        config.optimizer.lr_scale_final = config.optimizer.lr_scale * 0.1;
        config.optimizer.lr_rotation_final = config.optimizer.lr_rotation * 0.1;
        config.optimizer.lr_opacity_final = config.optimizer.lr_opacity * 0.1;
        config.optimizer.lr_color_final = config.optimizer.lr_color * 0.1;
        config.optimizer.lr_color_rest_final = config.optimizer.lr_color_rest * 0.1;
        let host_splats = trainer_checkpoint_host_splats();
        let mut splats = host_splats_to_device::<GsDiffBackend>(&host_splats, &device);
        let mut trainer = WgpuTrainer::new(config.clone(), device.clone(), 3, 4, 2.5);
        install_trainer_checkpoint_state(&mut trainer);
        trainer.update_optimizer_lrs(CHECKPOINT_ITERATIONS - 1, 4);
        trainer.optimizer.step_device_splats(
            &mut splats,
            Tensor::ones([3, 10], &trainer.device).mul_scalar(0.1),
            Tensor::ones([3, 4, 3], &trainer.device).mul_scalar(-0.2),
            Tensor::ones([3], &trainer.device).mul_scalar(0.3),
        );
        let checkpoint = trainer
            .checkpoint(
                &splats,
                trainer_checkpoint_identity(),
                CHECKPOINT_ITERATIONS,
                Some(0.125),
            )
            .await
            .expect("export decaying-LR checkpoint");

        let (mut restored, restored_splats) =
            WgpuTrainer::from_checkpoint(config, device, 2.5, &checkpoint)
                .await
                .expect("restore decaying-LR checkpoint");
        let reexported = restored
            .checkpoint(
                &restored_splats,
                checkpoint.identity.clone(),
                checkpoint.completed_iterations,
                checkpoint.latest_loss,
            )
            .await
            .expect("re-export decaying-LR checkpoint");
        assert_eq!(reexported.optimizer, checkpoint.optimizer);

        trainer.update_optimizer_lrs(CHECKPOINT_ITERATIONS, 4);
        restored.update_optimizer_lrs(CHECKPOINT_ITERATIONS, 4);
        assert_eq!(
            restored
                .optimizer
                .checkpoint()
                .await
                .expect("restored optimizer after next LR update"),
            trainer
                .optimizer
                .checkpoint()
                .await
                .expect("original optimizer after next LR update")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn trainer_checkpoint_exports_scheduled_active_sh_degree_boundaries() {
        let device = GsDevice::default();
        let config = trainer_checkpoint_config();
        let host_splats = trainer_checkpoint_host_splats();
        let splats = host_splats_to_device::<GsDiffBackend>(&host_splats, &device);
        let trainer = WgpuTrainer::new(config, device, 3, 4, 2.5);

        for (completed_iterations, expected_degree) in [(0, 0), (1000, 0), (1001, 1)] {
            let checkpoint = trainer
                .checkpoint(
                    &splats,
                    trainer_checkpoint_identity(),
                    completed_iterations,
                    None,
                )
                .await
                .expect("export active-SH checkpoint");
            assert_eq!(
                checkpoint.active_sh_degree, expected_degree,
                "completed iterations {completed_iterations}"
            );
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn trainer_checkpoint_rejects_malformed_device_splat_shapes_without_panicking() {
        let device = GsDevice::default();
        let config = trainer_checkpoint_config();
        let host_splats = trainer_checkpoint_host_splats();
        let mut splats = host_splats_to_device::<GsDiffBackend>(&host_splats, &device);
        splats.transforms = Param::from_tensor(Tensor::zeros([3, 9], &device));
        let trainer = WgpuTrainer::new(config, device, 3, 4, 2.5);

        let error = trainer
            .checkpoint(&splats, trainer_checkpoint_identity(), 0, None)
            .await
            .expect_err("malformed device splats must return an error");
        assert!(matches!(
            error,
            TrainingError::InvalidInput(message)
                if message.contains("transforms") && message.contains("[N, 10]")
        ));
    }
}
