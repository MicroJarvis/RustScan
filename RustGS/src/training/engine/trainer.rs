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
use crate::training::gpu_primitives::device_radix::{
    radix_sort_dispatch_count, radix_sort_workspace_bytes,
};
use crate::training::gpu_primitives::prefix_sum::{
    prefix_sum_dispatch_count, prefix_sum_workspace_bytes,
};
use crate::training::reporting::metrics::{
    step_intersection_overflowed, ParityLossCurveSample, ParityTopologyMetrics,
};
use crate::training::reporting::optimization_report::{
    duration_millis, percentile_f64, percentile_usize,
};
use crate::training::reporting::telemetry::{LiteGsOptimizerLrs, LiteGsTrainingTelemetry};
use crate::training::topology::{
    apply_mutations, apply_topology_metrics_delta, plan_mutations, should_apply_topology_step,
    snapshot_for_topology, visibility_window_baseline_from_cumulative, visibility_window_delta,
    TopologyMutationPlan,
};
use crate::training::{
    LiteGsPruneMode, TrainingCheckpoint, TrainingCheckpointReady, TrainingCheckpointReason,
    TrainingConfig, TrainingIdentity, TrainingRunDisposition, TRAINING_CHECKPOINT_VERSION,
};
use crate::TrainingError;

use super::backend::{GsBackendBase, GsDevice, GsDiffBackend};
use super::device_status::{DeviceTrainingStatus, TrainingStatusSnapshot};
use super::loss::{combined_loss_with_kernel, gaussian_kernel_1d, LossStatusBackend, SsimConfig};
use super::optimizer::{AdamScaled, AdamScaledConfig};
use super::splats::{
    device_splats_to_host, host_splats_to_device, try_device_splats_to_host, DeviceSplats,
};
use super::topology_accum::{accumulate_topology_stats, TopologyAccumulatorSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StatusReadbackReason {
    LossCadence,
    TopologyBoundary,
    Checkpoint,
    Pause,
    Cancel,
    TrainingEnd,
}

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
    pub loop_duration: Duration,
    pub loss_readback: bool,
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

pub(crate) const LOSS_SCALAR_READBACK_INTERVAL: usize = 20;

pub(crate) fn should_read_loss(
    iteration: usize,
    total_iterations: usize,
    cadence: usize,
    checkpoint_due: bool,
    paused: bool,
) -> bool {
    let cadence = cadence.max(1);
    checkpoint_due
        || paused
        || iteration == 1
        || iteration >= total_iterations.max(1)
        || iteration.is_multiple_of(cadence)
}

fn record_completed_step(
    report: &mut WgpuTrainingReport,
    iteration: usize,
    gaussian_count: usize,
    loss: Option<f32>,
) {
    report.completed_iterations = iteration;
    report.final_gaussian_count = gaussian_count;
    if let Some(loss) = loss {
        report.final_loss = Some(loss);
        report.final_step_loss = Some(loss);
    }
}

pub(crate) trait TrainingLoopObserver {
    fn should_cancel(&self) -> bool {
        false
    }

    fn should_pause(&self) -> bool {
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

fn complete_checkpoint_boundary(
    observer: &mut dyn TrainingLoopObserver,
    mut ready: TrainingCheckpointReady,
) -> Result<Option<TrainingRunDisposition>, TrainingError> {
    if observer.should_cancel() {
        return Ok(Some(TrainingRunDisposition::Cancelled));
    }

    if ready.reason == TrainingCheckpointReason::Periodic && observer.should_pause() {
        ready.reason = TrainingCheckpointReason::Pause;
    }
    let reason = ready.reason;
    observer.on_checkpoint(ready)?;
    if observer.should_cancel() {
        Ok(Some(TrainingRunDisposition::Cancelled))
    } else if reason == TrainingCheckpointReason::Pause || observer.should_pause() {
        Ok(Some(TrainingRunDisposition::Paused))
    } else {
        Ok(None)
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
    /// Cumulative visible_observations at the previous topology step. Window
    /// visibility for prune / invisible-window advancement is the delta.
    visibility_window_baseline: Vec<f32>,
    actual_visibility_window_baseline: Vec<f32>,
    ssim_config: SsimConfig,
    ssim_kernel: Tensor<GsDiffBackend, 1>,
    telemetry: LiteGsTrainingTelemetry,
    position_lr_scene_scale: f32,
    optimizer_lr_state: Option<OptimizerLrState>,
    intersection_capacity: usize,
    device_status: DeviceTrainingStatus<GsBackendBase>,
    optimization_samples: OptimizationTimingSamples,
    /// When set (tests / parity harness only), forces forward capacity below the
    /// planned size so overflow sticky flags can be exercised through `train_step`.
    intersection_capacity_override: Option<usize>,
}

#[derive(Debug, Clone, Default)]
struct OptimizationTimingSamples {
    loop_ms: Vec<f64>,
    loss_readbacks: usize,
    count_readbacks: usize,
    status_readbacks: usize,
    status_readbacks_loss_cadence: usize,
    status_readbacks_topology: usize,
    status_readbacks_checkpoint: usize,
    status_readbacks_pause: usize,
    status_readbacks_cancel: usize,
    status_readbacks_training_end: usize,
    capacity_telemetry_readbacks: usize,
    loss_value_readbacks: usize,
    checkpoint_tensor_readbacks: usize,
    sort_dispatches: Vec<usize>,
    scan_dispatches: Vec<usize>,
    sort_workspace_bytes: Option<usize>,
    scan_workspace_bytes: Option<usize>,
    topology_snapshot_ms: Vec<f64>,
    topology_plan_ms: Vec<f64>,
    topology_apply_ms: Vec<f64>,
    topology_snapshot_readback_bytes: Option<usize>,
}

impl OptimizationTimingSamples {
    fn record_loop_step(
        &mut self,
        loop_duration: Duration,
        loss_readback: bool,
        splat_count: usize,
        intersection_capacity: usize,
    ) {
        self.loop_ms.push(duration_millis(loop_duration));
        if loss_readback {
            self.loss_readbacks = self.loss_readbacks.saturating_add(1);
        }
        let sort_len = splat_count;
        let scan_len = splat_count;
        self.sort_dispatches
            .push(radix_sort_dispatch_count(sort_len));
        self.scan_dispatches
            .push(prefix_sum_dispatch_count(scan_len));
        let sort_bytes = radix_sort_workspace_bytes(intersection_capacity.max(sort_len));
        let scan_bytes = prefix_sum_workspace_bytes(scan_len);
        self.sort_workspace_bytes = Some(self.sort_workspace_bytes.unwrap_or(0).max(sort_bytes));
        self.scan_workspace_bytes = Some(self.scan_workspace_bytes.unwrap_or(0).max(scan_bytes));
    }

    fn record_count_readbacks(&mut self, count: usize) {
        self.count_readbacks = self.count_readbacks.saturating_add(count);
    }

    fn record_status_readback(&mut self, reason: StatusReadbackReason) {
        self.status_readbacks = self.status_readbacks.saturating_add(1);
        match reason {
            StatusReadbackReason::LossCadence => {
                self.status_readbacks_loss_cadence =
                    self.status_readbacks_loss_cadence.saturating_add(1);
            }
            StatusReadbackReason::TopologyBoundary => {
                self.status_readbacks_topology = self.status_readbacks_topology.saturating_add(1);
            }
            StatusReadbackReason::Checkpoint => {
                self.status_readbacks_checkpoint =
                    self.status_readbacks_checkpoint.saturating_add(1);
            }
            StatusReadbackReason::Pause => {
                self.status_readbacks_pause = self.status_readbacks_pause.saturating_add(1);
            }
            StatusReadbackReason::Cancel => {
                self.status_readbacks_cancel = self.status_readbacks_cancel.saturating_add(1);
            }
            StatusReadbackReason::TrainingEnd => {
                self.status_readbacks_training_end =
                    self.status_readbacks_training_end.saturating_add(1);
            }
        }
    }

    fn flush_into(&self, telemetry: &mut LiteGsTrainingTelemetry) {
        let mut loop_sorted = self.loop_ms.clone();
        loop_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        telemetry.loop_duration_p50_ms = percentile_f64(&loop_sorted, 50.0);
        telemetry.loop_duration_p95_ms = percentile_f64(&loop_sorted, 95.0);
        telemetry.loop_timing_kind = Some("cpu_submit_instant".into());
        telemetry.loss_readback_count = Some(self.loss_readbacks);
        telemetry.count_readback_count = Some(self.count_readbacks);
        telemetry.status_readbacks = Some(self.status_readbacks);
        telemetry.status_readbacks_loss_cadence = Some(self.status_readbacks_loss_cadence);
        telemetry.status_readbacks_topology = Some(self.status_readbacks_topology);
        telemetry.status_readbacks_checkpoint = Some(self.status_readbacks_checkpoint);
        telemetry.status_readbacks_pause = Some(self.status_readbacks_pause);
        telemetry.status_readbacks_cancel = Some(self.status_readbacks_cancel);
        telemetry.status_readbacks_training_end = Some(self.status_readbacks_training_end);
        telemetry.capacity_telemetry_readbacks = Some(self.capacity_telemetry_readbacks);
        telemetry.loss_value_readbacks = Some(self.loss_value_readbacks);
        telemetry.checkpoint_tensor_readbacks = Some(self.checkpoint_tensor_readbacks);
        telemetry.radix_dispatch_count_p50 = percentile_usize(&self.sort_dispatches, 50.0);
        telemetry.radix_dispatch_count_p95 = percentile_usize(&self.sort_dispatches, 95.0);
        telemetry.scan_dispatch_count_p50 = percentile_usize(&self.scan_dispatches, 50.0);
        telemetry.scan_dispatch_count_p95 = percentile_usize(&self.scan_dispatches, 95.0);
        telemetry.sort_workspace_bytes = self.sort_workspace_bytes;
        telemetry.scan_workspace_bytes = self.scan_workspace_bytes;

        let mut snapshot_sorted = self.topology_snapshot_ms.clone();
        snapshot_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut plan_sorted = self.topology_plan_ms.clone();
        plan_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut apply_sorted = self.topology_apply_ms.clone();
        apply_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        telemetry.topology_snapshot_ms_p50 = percentile_f64(&snapshot_sorted, 50.0);
        telemetry.topology_plan_ms_p50 = percentile_f64(&plan_sorted, 50.0);
        telemetry.topology_apply_ms_p50 = percentile_f64(&apply_sorted, 50.0);
        telemetry.topology_snapshot_readback_bytes = self.topology_snapshot_readback_bytes;
    }
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
            visibility_window_baseline: vec![0.0; initial_splats],
            actual_visibility_window_baseline: vec![0.0; initial_splats],
            ssim_config,
            ssim_kernel,
            telemetry,
            position_lr_scene_scale,
            optimizer_lr_state: None,
            intersection_capacity: 0,
            device_status: DeviceTrainingStatus::new(&device, 0),
            optimization_samples: OptimizationTimingSamples::default(),
            intersection_capacity_override: None,
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub async fn checkpoint(
        &mut self,
        splats: &DeviceSplats<GsDiffBackend>,
        identity: TrainingIdentity,
        completed_iterations: usize,
        latest_loss: Option<f32>,
    ) -> Result<TrainingCheckpoint, TrainingError> {
        self.checkpoint_with_status_reason(
            splats,
            identity,
            completed_iterations,
            latest_loss,
            StatusReadbackReason::Checkpoint,
        )
        .await
    }

    async fn checkpoint_with_status_reason(
        &mut self,
        splats: &DeviceSplats<GsDiffBackend>,
        identity: TrainingIdentity,
        completed_iterations: usize,
        latest_loss: Option<f32>,
        status_reason: StatusReadbackReason,
    ) -> Result<TrainingCheckpoint, TrainingError> {
        self.ensure_device_status_healthy(status_reason).await?;
        let host_splats = try_device_splats_to_host(splats).await?;
        self.optimization_samples.checkpoint_tensor_readbacks = self
            .optimization_samples
            .checkpoint_tensor_readbacks
            .saturating_add(1);
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
        .checkpoint(
            &self.splat_birth_iterations,
            &self.splat_invisible_windows,
            &self.visibility_window_baseline,
            &self.actual_visibility_window_baseline,
        )
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
        trainer.device_status =
            DeviceTrainingStatus::new(&device, checkpoint.optimizer.transforms.step as u32);
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
        // v2 checkpoints own the mid-window baselines; do not rebuild from cumulative.
        trainer.visibility_window_baseline = checkpoint.topology.visibility_window_baseline.clone();
        trainer.actual_visibility_window_baseline = checkpoint
            .topology
            .actual_visibility_window_baseline
            .clone();
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
        read_loss: bool,
    ) -> Result<Option<f32>, TrainingError> {
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
            self.intersection_capacity_for(splats.num_splats(), (width as u32, height as u32)),
            Some((iteration as u32, self.device_status.buffer().clone())),
        )
        .await;
        // Overflow sticky bits are written on-device by write_dispatch (no per-step
        // host readback). Host mirror catches up at safety-point reads; device
        // mutation gates land in Task 1.4.
        // Fail before loss/backward/optimizer so a truncated forward cannot
        // update Adam moments, parameters, or topology statistics.
        self.ensure_forward_capacity_before_update(read_loss)
            .await?;
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
            self.config.loss.loss_dynamic_mask_gradient,
            &self.ssim_config,
            self.ssim_kernel.clone(),
        );
        // Mark sticky non-finite on device before backward / mutation.
        GsBackendBase::mark_non_finite_loss(
            loss.clone().inner().into_primitive().tensor(),
            self.device_status.buffer().clone().into_primitive(),
            iteration as u32,
        );
        let loss_for_read = read_loss.then(|| loss.clone());
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
        // Defense in depth: sticky must still be clear immediately before mutation.
        self.ensure_forward_capacity_before_update(read_loss)
            .await?;
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
        self.optimizer.step_device_splats(
            splats,
            transforms_grad,
            sh_grad,
            opacity_grad,
            self.device_status.buffer().clone(),
        );
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
                0.0,
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
            self.ensure_device_status_healthy(StatusReadbackReason::TopologyBoundary)
                .await?;
            self.apply_topology_mutations(splats, iteration, frame_count)
                .await;
        }

        if !read_loss {
            return Ok(None);
        }

        // Sample current-frame capacity telemetry on loss cadence only. Sticky
        // overflow was already consumed before state updates above.
        let capacity_telemetry = self
            .sample_forward_capacity(
                &rendered.logical_visible,
                &rendered.requested_intersections,
                &rendered.intersection_overflow,
                rendered.intersection_capacity,
            )
            .await?;
        debug_assert!(
            !capacity_telemetry.overflowed
                || self.device_status.host_snapshot().has_forward_overflow(),
            "loss-cadence status sync must observe sticky overflow before reporting"
        );
        let loss_value = loss_for_read
            .expect("loss retained for scalar readback")
            .into_scalar_async()
            .await
            .map_err(|err| TrainingError::TrainingFailed(format!("failed to read loss: {err}")))?;
        self.optimization_samples.loss_value_readbacks = self
            .optimization_samples
            .loss_value_readbacks
            .saturating_add(1);
        if self.device_status.host_snapshot().has_non_finite_loss() || !loss_value.is_finite() {
            let first = if self.device_status.host_snapshot().has_non_finite_loss() {
                self.device_status.host_snapshot().first_invalid_iteration
            } else {
                (iteration as u32).max(1)
            };
            return Err(TrainingError::NonFiniteLoss {
                first_iteration: first,
            });
        }
        Ok(Some(validate_loss_value(loss_value, iteration)?))
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
        let mut last_sampled_loss = 0.0;

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
            let step_started_at = Instant::now();
            let emit_progress = observer.should_emit_progress(iteration_idx);
            let emit_snapshot = observer.should_emit_snapshot(iteration_idx);
            let should_log_step = iteration_idx.is_multiple_of(100)
                || (start_iteration > 0 && zero_based == start_iteration);
            let checkpoint_due = observer.checkpoint_reason(iteration_idx).is_some();
            let read_loss = should_read_loss(
                iteration_idx,
                num_iterations,
                LOSS_SCALAR_READBACK_INTERVAL,
                checkpoint_due,
                observer.should_pause(),
            ) || should_log_step;
            let loss = self
                .train_step(
                    splats,
                    &cameras[sample_idx],
                    target_img,
                    image_dims,
                    iteration_idx,
                    cameras.len(),
                    collect_topology_stats,
                    read_loss,
                )
                .await?;
            let loop_duration = step_started_at.elapsed();
            self.optimization_samples.record_loop_step(
                loop_duration,
                read_loss,
                splats.num_splats(),
                self.intersection_capacity,
            );
            if let Some(loss) = loss {
                last_sampled_loss = loss;
            }
            record_completed_step(&mut report, iteration_idx, splats.num_splats(), loss);
            if let Some(loss) = loss {
                self.record_loss_sample(
                    iteration_idx,
                    frame_idx,
                    loss,
                    should_log_step || iteration_idx == num_iterations,
                );
            }
            let metrics = TrainingIterationMetrics {
                iteration: iteration_idx,
                loss: last_sampled_loss,
                gaussian_count: splats.num_splats(),
                loop_duration,
                loss_readback: read_loss,
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
                    last_sampled_loss,
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
                    .checkpoint_with_status_reason(
                        splats,
                        identity,
                        iteration_idx,
                        Some(last_sampled_loss),
                        Self::checkpoint_status_reason(reason),
                    )
                    .await?;
                if let Some(disposition) = complete_checkpoint_boundary(
                    observer,
                    TrainingCheckpointReady {
                        iteration: iteration_idx,
                        reason,
                        checkpoint,
                    },
                )? {
                    report.cancelled = disposition == TrainingRunDisposition::Cancelled;
                    report.disposition = disposition;
                    break;
                }
            }
        }

        report.training_loop_elapsed = training_loop_started_at.elapsed();
        self.ensure_device_status_healthy(StatusReadbackReason::TrainingEnd)
            .await?;
        self.finish_report(&mut report);
        Ok(report)
    }

    fn intersection_capacity_for(&mut self, splat_count: usize, img_size: (u32, u32)) -> usize {
        if let Some(capacity) = self.intersection_capacity_override {
            self.intersection_capacity = capacity.max(1);
            return self.intersection_capacity;
        }
        let tile_bounds = crate::training::forward::calc_tile_bounds(img_size);
        let hard = crate::training::forward::hard_intersection_capacity(
            splat_count,
            tile_bounds.0 * tile_bounds.1,
        );
        let planned = crate::training::forward::planned_intersection_capacity(
            splat_count,
            tile_bounds.0 * tile_bounds.1,
        );
        let capacity = self.intersection_capacity.max(planned).min(hard);
        self.intersection_capacity = capacity;
        capacity
    }

    /// Force a fixed intersection workspace capacity (parity / fault-injection).
    pub fn force_intersection_capacity_for_test(&mut self, capacity: usize) {
        self.intersection_capacity_override = Some(capacity.max(1));
    }

    /// Stage-1 overflow hard-stop reused by bounded-forward parity (capacity − 1).
    pub async fn assert_overflow_capacity_minus_one_mutates_nothing_for_test(
        &mut self,
        splats: &mut DeviceSplats<GsDiffBackend>,
        camera: &GaussianCamera,
        target: Tensor<GsDiffBackend, 3>,
        image_dims: (usize, usize),
        capacity: usize,
    ) -> Result<(), TrainingError> {
        let healthy = self
            .train_step(
                splats,
                camera,
                target.clone(),
                image_dims,
                1,
                1,
                true,
                false,
            )
            .await?;
        debug_assert!(healthy.is_none());

        let before = self.snapshot_mutation_state_for_test(splats).await?;
        self.force_intersection_capacity_for_test(capacity);

        let overflowed = self
            .train_step(splats, camera, target, image_dims, 2, 1, true, false)
            .await?;
        debug_assert!(overflowed.is_none());

        let after = self.snapshot_mutation_state_for_test(splats).await?;
        let mut mismatches = Vec::new();
        if after.0 != before.0 {
            mismatches.push("transforms");
        }
        if after.1 != before.1 {
            mismatches.push("sh");
        }
        if after.2 != before.2 {
            mismatches.push("opacity");
        }
        if after.3 != before.3 {
            mismatches.push("adam");
        }
        if after.4 != before.4 {
            mismatches.push("topology");
        }
        if after.6 != before.6 {
            mismatches.push("birth");
        }
        if after.7 != before.7 {
            mismatches.push("invisible");
        }
        if after.5.committed_optimizer_steps != before.5.committed_optimizer_steps {
            mismatches.push("committed_optimizer_steps");
        }
        if !mismatches.is_empty() {
            return Err(TrainingError::TrainingFailed(format!(
                "capacity-1 overflow mutated trainer state ({})",
                mismatches.join(",")
            )));
        }
        if !after.5.has_forward_overflow() {
            return Err(TrainingError::TrainingFailed(format!(
                "capacity-1 overflow missing sticky forward overflow flag (flags={:#x}, requested={}, capacity={})",
                after.5.flags,
                after.5.requested_intersections,
                after.5.intersection_capacity
            )));
        }
        Ok(())
    }

    async fn snapshot_mutation_state_for_test(
        &mut self,
        splats: &DeviceSplats<GsDiffBackend>,
    ) -> Result<
        (
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            crate::training::AdamCheckpoint,
            crate::training::TopologyCheckpoint,
            TrainingStatusSnapshot,
            Vec<usize>,
            Vec<usize>,
        ),
        TrainingError,
    > {
        let status = self.device_status.read().await?;
        self.device_status.adopt_device_snapshot(status);
        self.optimizer
            .sync_committed_steps(status.committed_optimizer_steps as usize);
        let transforms = splats
            .transforms
            .val()
            .into_data_async()
            .await
            .map_err(|e| TrainingError::TrainingFailed(format!("transforms read: {e}")))?
            .into_vec::<f32>()
            .map_err(|e| TrainingError::TrainingFailed(format!("transforms cast: {e:?}")))?;
        let sh = splats
            .sh_coeffs
            .val()
            .into_data_async()
            .await
            .map_err(|e| TrainingError::TrainingFailed(format!("sh read: {e}")))?
            .into_vec::<f32>()
            .map_err(|e| TrainingError::TrainingFailed(format!("sh cast: {e:?}")))?;
        let opacity = splats
            .raw_opacities
            .val()
            .into_data_async()
            .await
            .map_err(|e| TrainingError::TrainingFailed(format!("opacity read: {e}")))?
            .into_vec::<f32>()
            .map_err(|e| TrainingError::TrainingFailed(format!("opacity cast: {e:?}")))?;
        let adam = self.optimizer.checkpoint().await?;
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
        .checkpoint(
            &self.splat_birth_iterations,
            &self.splat_invisible_windows,
            &self.visibility_window_baseline,
            &self.actual_visibility_window_baseline,
        )
        .await?;
        Ok((
            transforms,
            sh,
            opacity,
            adam,
            topology,
            status,
            self.splat_birth_iterations.clone(),
            self.splat_invisible_windows.clone(),
        ))
    }

    async fn ensure_forward_capacity_before_update(
        &self,
        read_loss: bool,
    ) -> Result<(), TrainingError> {
        let _ = read_loss;
        if let Some(err) = self.device_status.host_snapshot().to_error() {
            return Err(err);
        }
        Ok(())
    }

    async fn ensure_device_status_healthy(
        &mut self,
        reason: StatusReadbackReason,
    ) -> Result<TrainingStatusSnapshot, TrainingError> {
        let status = self.device_status.read().await?;
        self.device_status.adopt_device_snapshot(status);
        self.optimization_samples.record_status_readback(reason);
        self.optimizer
            .sync_committed_steps(status.committed_optimizer_steps as usize);
        if let Some(err) = status.to_error() {
            return Err(err);
        }
        Ok(status)
    }

    fn checkpoint_status_reason(reason: TrainingCheckpointReason) -> StatusReadbackReason {
        match reason {
            TrainingCheckpointReason::Periodic => StatusReadbackReason::Checkpoint,
            TrainingCheckpointReason::Pause => StatusReadbackReason::Pause,
            TrainingCheckpointReason::Shutdown => StatusReadbackReason::Cancel,
        }
    }

    async fn sample_forward_capacity(
        &mut self,
        logical_visible: &Tensor<GsDiffBackend, 1, Int>,
        requested: &Tensor<GsDiffBackend, 1, Int>,
        overflow: &Tensor<GsDiffBackend, 1, Int>,
        capacity: usize,
    ) -> Result<crate::training::reporting::metrics::ForwardCapacityTelemetry, TrainingError> {
        // Loss cadence is a safety point: pull sticky status without relying on
        // per-step overflow scalar readback.
        let status = self
            .ensure_device_status_healthy(StatusReadbackReason::LossCadence)
            .await?;

        let visible_value = logical_visible
            .clone()
            .inner()
            .into_scalar_async()
            .await
            .map_err(|err| {
                TrainingError::TrainingFailed(format!("failed to read logical visible: {err}"))
            })?;
        let requested_value = requested
            .clone()
            .inner()
            .into_scalar_async()
            .await
            .map_err(|err| {
                TrainingError::TrainingFailed(format!(
                    "failed to read requested intersections: {err}"
                ))
            })?;
        let overflow_value = overflow
            .clone()
            .inner()
            .into_scalar_async()
            .await
            .map_err(|err| {
                TrainingError::TrainingFailed(format!(
                    "failed to read intersection overflow: {err}"
                ))
            })?;
        let telemetry = crate::training::reporting::metrics::ForwardCapacityTelemetry {
            logical_visible: visible_value.max(0) as u32,
            logical_intersections: requested_value.max(0) as u32,
            capacity: capacity as u32,
            overflowed: overflow_value != 0
                || step_intersection_overflowed(requested_value.max(0) as u32, capacity as u32)
                || status.has_forward_overflow(),
        };
        self.telemetry.forward_capacity = Some(telemetry);
        self.optimization_samples.record_count_readbacks(3);
        self.optimization_samples.capacity_telemetry_readbacks = self
            .optimization_samples
            .capacity_telemetry_readbacks
            .saturating_add(3);
        Ok(telemetry)
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
            self.device_status.buffer().clone(),
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
        // Weight used to force visible_observations += 1, so history-invisible
        // pruning could never see a zero count. All prune modes now accumulate
        // the rasterizer visibility bit.
        matches!(
            self.config.litegs.pruning.prune_mode,
            LiteGsPruneMode::Threshold
                | LiteGsPruneMode::Weight
                | LiteGsPruneMode::VisibilityWeight
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
        let snapshot_started = Instant::now();
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
        let accumulator_fields = if self.collects_actual_visibility_diagnostics() {
            10
        } else {
            9
        };
        self.optimization_samples
            .topology_snapshot_ms
            .push(duration_millis(snapshot_started.elapsed()));
        let snapshot_readback_bytes = snapshot
            .splats
            .len()
            .saturating_mul(accumulator_fields * std::mem::size_of::<f32>());
        self.optimization_samples.topology_snapshot_readback_bytes = Some(
            self.optimization_samples
                .topology_snapshot_readback_bytes
                .unwrap_or(0)
                .max(snapshot_readback_bytes),
        );
        // Densify keeps cumulative visible_observations. Prune / invisible windows
        // use the delta since the previous topology step so SkipNoEligibleCandidates
        // can retain densify accumulators without freezing invisibility clocks.
        snapshot.window_visible_observations = visibility_window_delta(
            &snapshot.visible_observations,
            &self.visibility_window_baseline,
        );
        snapshot.window_actual_visible_observations = visibility_window_delta(
            &snapshot.actual_visible_observations,
            &self.actual_visibility_window_baseline,
        );
        self.update_topology_visibility_state(
            snapshot.splats.len(),
            &snapshot.window_visible_observations,
            iteration,
        );
        snapshot.splat_ages = self.splat_ages_at(iteration, snapshot.splats.len());
        snapshot.invisible_windows = self
            .splat_invisible_windows
            .iter()
            .copied()
            .take(snapshot.splats.len())
            .collect();
        let plan_started = Instant::now();
        let plan = plan_mutations(&snapshot, &self.config, iteration, frame_count);
        self.optimization_samples
            .topology_plan_ms
            .push(duration_millis(plan_started.elapsed()));
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
        self.telemetry.topology.scheduled_steps =
            self.telemetry.topology.scheduled_steps.saturating_add(1);
        let apply_started = Instant::now();
        if plan.mutates_splats() {
            apply_mutations(splats, &snapshot.splats, &plan, &self.device);
            self.remap_topology_visibility_state(&plan, iteration);
        }
        if plan.aftermath.requires_adam_rebuild {
            let sh_dims = splats.sh_coeffs.val().dims();
            self.optimizer.remap_origins(
                &plan.origins(),
                sh_dims[1],
                sh_dims.get(2).copied().unwrap_or(3),
                &self.device,
            );
        }
        if plan.aftermath.apply_opacity_reset {
            self.optimizer.clear_opacity_moments();
        }
        if plan.should_retain_accumulators() {
            self.telemetry.topology.skipped_no_eligible_candidates = self
                .telemetry
                .topology
                .skipped_no_eligible_candidates
                .saturating_add(1);
            log::info!(
                "Topology step {} retained accumulators: no eligible candidates",
                iteration
            );
            // Advance the visibility window even when densify accumulators stay.
            self.visibility_window_baseline =
                visibility_window_baseline_from_cumulative(&snapshot.visible_observations);
            self.actual_visibility_window_baseline =
                visibility_window_baseline_from_cumulative(&snapshot.actual_visible_observations);
        } else {
            self.telemetry.topology.accumulator_resets =
                self.telemetry.topology.accumulator_resets.saturating_add(1);
            self.reset_accumulators(
                splats.num_splats(),
                splats.sh_coeffs.val().dims()[1],
                iteration,
            );
        }
        self.optimization_samples
            .topology_apply_ms
            .push(duration_millis(apply_started.elapsed()));
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
        self.visibility_window_baseline = vec![0.0; num_splats];
        self.actual_visibility_window_baseline = vec![0.0; num_splats];

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

        if self.visibility_window_baseline.len() < num_splats {
            self.visibility_window_baseline.resize(num_splats, 0.0);
        } else {
            self.visibility_window_baseline.truncate(num_splats);
        }

        if self.actual_visibility_window_baseline.len() < num_splats {
            self.actual_visibility_window_baseline
                .resize(num_splats, 0.0);
        } else {
            self.actual_visibility_window_baseline.truncate(num_splats);
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
        let previous_visibility_baseline = self.visibility_window_baseline.clone();
        let previous_actual_baseline = self.actual_visibility_window_baseline.clone();
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
        self.visibility_window_baseline = origins
            .iter()
            .map(|origin| {
                origin
                    .and_then(|idx| previous_visibility_baseline.get(idx).copied())
                    .unwrap_or(0.0)
            })
            .collect();
        self.actual_visibility_window_baseline = origins
            .iter()
            .map(|origin| {
                origin
                    .and_then(|idx| previous_actual_baseline.get(idx).copied())
                    .unwrap_or(0.0)
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
        self.optimization_samples.flush_into(&mut self.telemetry);
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
    use crate::training::{
        TensorCheckpoint, TopologyCheckpoint, TrainingCheckpoint, TrainingIdentity,
    };
    use burn::module::Param;

    #[test]
    fn loss_scalar_readback_is_not_every_step() {
        assert!(should_read_loss(
            1,
            3_000,
            LOSS_SCALAR_READBACK_INTERVAL,
            false,
            false
        ));
        assert!(!should_read_loss(
            2,
            3_000,
            LOSS_SCALAR_READBACK_INTERVAL,
            false,
            false
        ));
        assert!(should_read_loss(
            20,
            3_000,
            LOSS_SCALAR_READBACK_INTERVAL,
            false,
            false
        ));
        assert!(should_read_loss(
            19,
            3_000,
            LOSS_SCALAR_READBACK_INTERVAL,
            true,
            false
        ));
        assert!(should_read_loss(
            19,
            3_000,
            LOSS_SCALAR_READBACK_INTERVAL,
            false,
            true
        ));
        assert!(should_read_loss(
            3_000,
            3_000,
            LOSS_SCALAR_READBACK_INTERVAL,
            false,
            false
        ));
        assert!(!should_read_loss(
            2_999,
            3_000,
            LOSS_SCALAR_READBACK_INTERVAL,
            false,
            false
        ));
    }

    #[test]
    fn sticky_overflow_blocks_mutation_and_checkpoint_identity() {
        let snapshot = super::super::device_status::note_forward_overflow(
            TrainingStatusSnapshot::default(),
            true,
            2,
            9_000,
            8_000,
        );
        assert!(!snapshot.is_healthy());
        let err = snapshot.to_error().expect("overflow must produce an error");
        assert!(matches!(
            err,
            TrainingError::ForwardCapacityExceeded {
                first_iteration: 2,
                logical_intersections: 9_000,
                capacity: 8_000,
            }
        ));
    }

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
        trainer.visibility_window_baseline = vec![10.0, 20.0, 30.0];
        trainer.actual_visibility_window_baseline = vec![11.0, 21.0, 31.0];
    }

    fn step_trainer_optimizer(trainer: &mut WgpuTrainer, splats: &mut DeviceSplats<GsDiffBackend>) {
        trainer.update_optimizer_lrs(CHECKPOINT_ITERATIONS, 4);
        trainer.optimizer.step_device_splats(
            splats,
            Tensor::ones([3, 10], &trainer.device).mul_scalar(0.1),
            Tensor::ones([3, 4, 3], &trainer.device).mul_scalar(-0.2),
            Tensor::ones([3], &trainer.device).mul_scalar(0.3),
            trainer.device_status.buffer().clone(),
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

    struct CancelledCheckpointObserver {
        checkpoint_calls: usize,
    }

    impl TrainingLoopObserver for CancelledCheckpointObserver {
        fn should_cancel(&self) -> bool {
            true
        }

        fn on_checkpoint(&mut self, _ready: TrainingCheckpointReady) -> Result<(), TrainingError> {
            self.checkpoint_calls += 1;
            Ok(())
        }
    }

    struct PausedCheckpointObserver {
        committed_reason: Option<TrainingCheckpointReason>,
    }

    impl TrainingLoopObserver for PausedCheckpointObserver {
        fn should_pause(&self) -> bool {
            true
        }

        fn on_checkpoint(&mut self, ready: TrainingCheckpointReady) -> Result<(), TrainingError> {
            self.committed_reason = Some(ready.reason);
            Ok(())
        }
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
        record_completed_step(&mut report, 1, 10, Some(0.5));
        record_completed_step(&mut report, 2, 11, Some(0.25));
        assert_eq!(report.final_loss, Some(0.25));
        assert_eq!(report.final_step_loss, Some(0.25));
        assert_eq!(report.completed_iterations, 2);
        assert_eq!(report.final_gaussian_count, 11);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn checkpoint_cancel_after_build_skips_commit_and_finishes_cancelled() {
        let device = GsDevice::default();
        let (_, checkpoint) = populated_trainer_checkpoint(&device).await;
        let mut observer = CancelledCheckpointObserver {
            checkpoint_calls: 0,
        };

        let disposition = complete_checkpoint_boundary(
            &mut observer,
            TrainingCheckpointReady {
                iteration: CHECKPOINT_ITERATIONS,
                reason: TrainingCheckpointReason::Pause,
                checkpoint,
            },
        )
        .unwrap();

        assert_eq!(disposition, Some(TrainingRunDisposition::Cancelled));
        assert_eq!(observer.checkpoint_calls, 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn periodic_checkpoint_upgrades_to_pause_after_build_before_commit() {
        let device = GsDevice::default();
        let (_, checkpoint) = populated_trainer_checkpoint(&device).await;
        let mut observer = PausedCheckpointObserver {
            committed_reason: None,
        };

        let disposition = complete_checkpoint_boundary(
            &mut observer,
            TrainingCheckpointReady {
                iteration: CHECKPOINT_ITERATIONS,
                reason: TrainingCheckpointReason::Periodic,
                checkpoint,
            },
        )
        .unwrap();

        assert_eq!(
            observer.committed_reason,
            Some(TrainingCheckpointReason::Pause)
        );
        assert_eq!(disposition, Some(TrainingRunDisposition::Paused));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn sticky_overflow_rejects_successful_checkpoint_export() {
        let device = GsDevice::default();
        let config = trainer_checkpoint_config();
        let host_splats = trainer_checkpoint_host_splats();
        let splats = host_splats_to_device::<GsDiffBackend>(&host_splats, &device);
        let mut trainer = WgpuTrainer::new(config, device, 3, 4, 2.5);
        trainer.device_status.set_host_snapshot(
            super::super::device_status::note_forward_overflow(
                TrainingStatusSnapshot::default(),
                true,
                2,
                9_000,
                8_000,
            ),
        );
        let err = trainer
            .checkpoint(
                &splats,
                trainer_checkpoint_identity(),
                CHECKPOINT_ITERATIONS,
                Some(0.125),
            )
            .await
            .expect_err("sticky overflow must reject checkpoint export");
        assert!(matches!(
            err,
            TrainingError::ForwardCapacityExceeded {
                first_iteration: 2,
                logical_intersections: 9_000,
                capacity: 8_000,
            }
        ));
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
        assert_eq!(
            checkpoint.topology.visibility_window_baseline,
            [10.0, 20.0, 30.0]
        );
        assert_eq!(
            checkpoint.topology.actual_visibility_window_baseline,
            [11.0, 21.0, 31.0]
        );

        let (mut restored, restored_splats) =
            WgpuTrainer::from_checkpoint(config, device, 2.5, &checkpoint)
                .await
                .expect("restore trainer checkpoint");
        assert_eq!(
            restored.visibility_window_baseline,
            checkpoint.topology.visibility_window_baseline
        );
        assert_eq!(
            restored.actual_visibility_window_baseline,
            checkpoint.topology.actual_visibility_window_baseline
        );
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

        let (mut restored, restored_splats) =
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
            trainer.device_status.buffer().clone(),
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
        let mut trainer = WgpuTrainer::new(config, device, 3, 4, 2.5);

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
        let mut trainer = WgpuTrainer::new(config, device, 3, 4, 2.5);

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

    async fn snapshot_mutation_state(
        trainer: &mut WgpuTrainer,
        splats: &DeviceSplats<GsDiffBackend>,
    ) -> (
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        crate::training::AdamCheckpoint,
        TopologyCheckpoint,
        TrainingStatusSnapshot,
        Vec<usize>,
        Vec<usize>,
    ) {
        let status = trainer.device_status.read().await.expect("status read");
        trainer
            .optimizer
            .sync_committed_steps(status.committed_optimizer_steps as usize);
        let transforms = splats
            .transforms
            .val()
            .into_data_async()
            .await
            .expect("transforms")
            .into_vec::<f32>()
            .expect("f32");
        let sh = splats
            .sh_coeffs
            .val()
            .into_data_async()
            .await
            .expect("sh")
            .into_vec::<f32>()
            .expect("f32");
        let opacity = splats
            .raw_opacities
            .val()
            .into_data_async()
            .await
            .expect("opacity")
            .into_vec::<f32>()
            .expect("f32");
        let adam = trainer
            .optimizer
            .checkpoint()
            .await
            .expect("adam checkpoint");
        let topology = TopologyAccumulatorSet {
            grad_2d: trainer.grad_2d_accum.clone(),
            screen_grad_2d: trainer.screen_grad_2d_accum.clone(),
            abs_grad_2d: trainer.abs_grad_2d_accum.clone(),
            abs_pixel_grad_2d: trainer.abs_pixel_grad_2d_accum.clone(),
            pixel_coverage: trainer.pixel_coverage_accum.clone(),
            camera_depth: trainer.camera_depth_accum.clone(),
            grad_color: trainer.grad_color_accum.clone(),
            num_observations: trainer.num_observations.clone(),
            visible_observations: trainer.visible_observations.clone(),
            actual_visible_observations: trainer.actual_visible_observations.clone(),
        }
        .checkpoint(
            &trainer.splat_birth_iterations,
            &trainer.splat_invisible_windows,
            &trainer.visibility_window_baseline,
            &trainer.actual_visibility_window_baseline,
        )
        .await
        .expect("topology checkpoint");
        (
            transforms,
            sh,
            opacity,
            adam,
            topology,
            status,
            trainer.splat_birth_iterations.clone(),
            trainer.splat_invisible_windows.clone(),
        )
    }

    fn fault_injection_camera() -> GaussianCamera {
        use crate::{Intrinsics, SE3};
        GaussianCamera::new(
            Intrinsics::new(8.0, 8.0, 4.0, 4.0, 8, 8),
            SE3::new(&[0.0, 0.0, 0.0, 1.0], &[0.0, 0.0, 0.0]),
        )
    }

    fn fault_injection_target(device: &GsDevice, fill: f32) -> Tensor<GsDiffBackend, 3> {
        Tensor::<GsDiffBackend, 3>::full([8, 8, 3], fill, device)
    }

    fn fault_injection_config() -> TrainingConfig {
        let mut config = trainer_checkpoint_config();
        config.iterations = 4;
        config.litegs.topology.refine_every = 10_000;
        config.litegs.topology.opacity_reset_interval = 10_000;
        config
    }

    #[tokio::test(flavor = "current_thread")]
    async fn train_step_nan_injection_with_read_loss_false_mutates_nothing() {
        let device = GsDevice::default();
        let config = fault_injection_config();
        let host_splats = trainer_checkpoint_host_splats();
        let mut splats = host_splats_to_device::<GsDiffBackend>(&host_splats, &device);
        let mut trainer = WgpuTrainer::new(config, device.clone(), 3, 4, 2.5);
        install_trainer_checkpoint_state(&mut trainer);
        let camera = fault_injection_camera();

        let healthy = trainer
            .train_step(
                &mut splats,
                &camera,
                fault_injection_target(&device, 0.4),
                (8, 8),
                1,
                1,
                true,
                false,
            )
            .await
            .expect("healthy step");
        assert!(healthy.is_none());

        let before = snapshot_mutation_state(&mut trainer, &splats).await;
        assert_eq!(before.5.committed_optimizer_steps, 1);

        let poisoned = trainer
            .train_step(
                &mut splats,
                &camera,
                fault_injection_target(&device, f32::NAN),
                (8, 8),
                2,
                1,
                true,
                false,
            )
            .await
            .expect("gated nan step returns Ok when loss is not sampled");
        assert!(poisoned.is_none());

        let after = snapshot_mutation_state(&mut trainer, &splats).await;
        assert_eq!(after.0, before.0, "transforms must not change");
        assert_eq!(after.1, before.1, "sh must not change");
        assert_eq!(after.2, before.2, "opacity must not change");
        assert_eq!(after.3, before.3, "adam state must not change");
        assert_eq!(after.4, before.4, "topology accumulators must not change");
        assert_eq!(after.6, before.6, "birth iterations must not change");
        assert_eq!(after.7, before.7, "invisible windows must not change");
        assert_eq!(
            after.5.committed_optimizer_steps,
            before.5.committed_optimizer_steps
        );
        assert!(after.5.has_non_finite_loss());
        assert_eq!(after.5.first_invalid_iteration, 2);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn train_step_overflow_injection_with_read_loss_false_mutates_nothing() {
        let device = GsDevice::default();
        let config = fault_injection_config();
        let host_splats = trainer_checkpoint_host_splats();
        let mut splats = host_splats_to_device::<GsDiffBackend>(&host_splats, &device);
        let mut trainer = WgpuTrainer::new(config, device.clone(), 3, 4, 2.5);
        install_trainer_checkpoint_state(&mut trainer);
        let camera = fault_injection_camera();

        let healthy = trainer
            .train_step(
                &mut splats,
                &camera,
                fault_injection_target(&device, 0.4),
                (8, 8),
                1,
                1,
                true,
                false,
            )
            .await
            .expect("healthy step");
        assert!(healthy.is_none());

        let before = snapshot_mutation_state(&mut trainer, &splats).await;
        trainer.intersection_capacity_override = Some(1);

        let overflowed = trainer
            .train_step(
                &mut splats,
                &camera,
                fault_injection_target(&device, 0.4),
                (8, 8),
                2,
                1,
                true,
                false,
            )
            .await
            .expect("gated overflow step returns Ok when loss is not sampled");
        assert!(overflowed.is_none());

        let after = snapshot_mutation_state(&mut trainer, &splats).await;
        assert_eq!(after.0, before.0, "transforms must not change");
        assert_eq!(after.1, before.1, "sh must not change");
        assert_eq!(after.2, before.2, "opacity must not change");
        assert_eq!(after.3, before.3, "adam state must not change");
        assert_eq!(after.4, before.4, "topology accumulators must not change");
        assert_eq!(after.6, before.6);
        assert_eq!(after.7, before.7);
        assert_eq!(
            after.5.committed_optimizer_steps,
            before.5.committed_optimizer_steps
        );
        assert!(after.5.has_forward_overflow());
        assert_eq!(after.5.first_invalid_iteration, 2);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn checkpoint_outside_loss_cadence_still_reads_device_status() {
        let device = GsDevice::default();
        let config = fault_injection_config();
        let host_splats = trainer_checkpoint_host_splats();
        let mut splats = host_splats_to_device::<GsDiffBackend>(&host_splats, &device);
        let mut trainer = WgpuTrainer::new(config, device.clone(), 3, 4, 2.5);
        let camera = fault_injection_camera();

        for iteration in 1..=3 {
            let read_loss =
                should_read_loss(iteration, 100, LOSS_SCALAR_READBACK_INTERVAL, false, false);
            trainer
                .train_step(
                    &mut splats,
                    &camera,
                    fault_injection_target(&device, 0.4),
                    (8, 8),
                    iteration,
                    1,
                    false,
                    read_loss,
                )
                .await
                .expect("healthy train step");
        }
        assert!(!should_read_loss(
            3,
            100,
            LOSS_SCALAR_READBACK_INTERVAL,
            false,
            false
        ));
        assert_eq!(
            trainer.optimization_samples.status_readbacks_loss_cadence,
            1
        );
        assert_eq!(trainer.optimization_samples.status_readbacks_checkpoint, 0);

        trainer
            .checkpoint(&splats, trainer_checkpoint_identity(), 3, None)
            .await
            .expect("checkpoint outside loss cadence");
        assert_eq!(trainer.optimization_samples.status_readbacks_checkpoint, 1);
        assert_eq!(trainer.optimization_samples.status_readbacks, 2);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn hundred_step_status_readbacks_match_safety_points_not_steps() {
        let device = GsDevice::default();
        let mut config = fault_injection_config();
        config.iterations = 100;
        let host_splats = trainer_checkpoint_host_splats();
        let mut splats = host_splats_to_device::<GsDiffBackend>(&host_splats, &device);
        let mut trainer = WgpuTrainer::new(config, device.clone(), 3, 4, 2.5);
        let camera = fault_injection_camera();

        let mut expected_loss_cadence = 0usize;
        for iteration in 1..=100 {
            let read_loss =
                should_read_loss(iteration, 100, LOSS_SCALAR_READBACK_INTERVAL, false, false);
            if read_loss {
                expected_loss_cadence += 1;
            }
            trainer
                .train_step(
                    &mut splats,
                    &camera,
                    fault_injection_target(&device, 0.4),
                    (8, 8),
                    iteration,
                    1,
                    false,
                    read_loss,
                )
                .await
                .expect("healthy hundred-step training");
        }
        trainer
            .ensure_device_status_healthy(StatusReadbackReason::TrainingEnd)
            .await
            .expect("training end status");

        assert_eq!(
            trainer.optimization_samples.status_readbacks_loss_cadence,
            expected_loss_cadence
        );
        assert_eq!(trainer.optimization_samples.status_readbacks_topology, 0);
        assert_eq!(trainer.optimization_samples.status_readbacks_checkpoint, 0);
        assert_eq!(
            trainer.optimization_samples.status_readbacks_training_end,
            1
        );
        assert_eq!(
            trainer.optimization_samples.status_readbacks,
            expected_loss_cadence + 1
        );
        assert!(
            trainer.optimization_samples.status_readbacks < 100,
            "status readbacks must not scale with every step"
        );
        assert_eq!(expected_loss_cadence, 6);
    }
}
