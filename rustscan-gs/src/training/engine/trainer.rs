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
use crate::training::gpu_primitives::prefix_sum::{prefix_sum_dispatch_count, PrefixSumWorkspace};
use crate::training::reporting::gpu_profiler::{
    probe_training_device, profile_device_gpu_step, runtime_device_bytes_in_use,
    should_profile_gpu, span, CpuSpanTimer, PipelineTimingCollector,
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
    device_splats_to_host, empty_device_splats_placeholder, host_splats_to_device,
    try_device_splats_to_host, DeviceSplats,
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
    /// Diagnostic read after the device gate already marked the step unhealthy.
    /// Never used on healthy steps.
    ForwardAbort,
}

/// Result of one logical training step after device work finishes.
///
/// `Committed.loss == None` means the step submitted successfully but the loss
/// scalar was not read. That must not be confused with an aborted step.
#[derive(Debug)]
#[must_use]
pub(crate) enum TrainStepDisposition {
    /// Safety-point path already synced device status; host may confirm commits.
    ConfirmedCommitted {
        loss: Option<f32>,
    },
    /// Unread-loss step submitted without a status read; not yet a confirmed commit.
    SubmittedUnconfirmed,
    Aborted {
        error: TrainingError,
    },
}

/// Tracks submitted vs device-confirmed optimizer commits for the outer loop.
#[derive(Debug, Clone, Copy)]
struct CommitConfirmationState {
    start_iteration: usize,
    /// Device `committed_optimizer_steps` at loop entry (absolute).
    committed_baseline: usize,
    /// Highest logical iteration index that was submitted (pending or confirmed).
    highest_submitted: usize,
    /// Last confirmed `completed_iterations`.
    last_confirmed: usize,
}

impl CommitConfirmationState {
    fn new(start_iteration: usize, committed_baseline: usize) -> Self {
        Self {
            start_iteration,
            committed_baseline,
            highest_submitted: start_iteration,
            last_confirmed: start_iteration,
        }
    }

    fn note_submitted(&mut self, iteration: usize) {
        self.highest_submitted = self.highest_submitted.max(iteration);
    }

    /// Apply device word-4 commits. Returns newly confirmed iteration indices.
    fn apply_device_committed(&mut self, device_committed: usize) -> Vec<usize> {
        let confirmed = self
            .start_iteration
            .saturating_add(device_committed.saturating_sub(self.committed_baseline))
            .min(self.highest_submitted);
        if confirmed <= self.last_confirmed {
            return Vec::new();
        }
        let newly: Vec<usize> = ((self.last_confirmed + 1)..=confirmed).collect();
        self.last_confirmed = confirmed;
        newly
    }
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
    prefix_sum_workspace: PrefixSumWorkspace,
    pipeline_timing: PipelineTimingCollector,
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
    status_readbacks_forward_abort: usize,
    status_readbacks_step_disposition: usize,
    capacity_telemetry_readbacks: usize,
    loss_value_readbacks: usize,
    checkpoint_tensor_readbacks: usize,
    /// Device prepare_optimizer launches that blocked (from status word 6).
    gpu_gate_optimizer_skips: usize,
    /// Host-inferred backward skips once a sticky anomaly is observed.
    gpu_gate_backward_skips: usize,
    /// Host-inferred topology accumulate skips once a sticky anomaly is observed.
    gpu_gate_topology_skips: usize,
    /// Safety-point / ForwardAbort paths that returned a structured training error.
    host_safety_point_aborts: usize,
    sort_dispatches: Vec<usize>,
    scan_dispatches: Vec<usize>,
    sort_workspace_bytes: Option<usize>,
    scan_workspace_bytes: Option<usize>,
    scan_workspace_scratch_bytes: Option<usize>,
    scan_workspace_output_bytes: Option<usize>,
    scan_workspace_growth_count: usize,
    scan_workspace_output_growth_count: usize,
    scan_workspace_step_fresh_allocations: Vec<usize>,
    scan_workspace_step_scratch_fresh: Vec<usize>,
    scan_workspace_step_output_fresh: Vec<usize>,
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
        // Dispatch counts are host-side estimates from splat_count (not the
        // intersection-count scan length). Measured scan workspace bytes come
        // only from record_scan_workspace_stats / PrefixSumWorkspace.
        let sort_len = splat_count;
        self.sort_dispatches
            .push(radix_sort_dispatch_count(sort_len));
        self.scan_dispatches
            .push(prefix_sum_dispatch_count(splat_count));
        let sort_bytes = radix_sort_workspace_bytes(intersection_capacity.max(sort_len));
        self.sort_workspace_bytes = Some(self.sort_workspace_bytes.unwrap_or(0).max(sort_bytes));
    }

    fn record_scan_workspace_stats(
        &mut self,
        reserved_bytes: usize,
        scratch_bytes: usize,
        output_bytes: usize,
        growth_count: usize,
        output_growth_count: usize,
        step_fresh: usize,
        step_scratch_fresh: usize,
        step_output_fresh: usize,
    ) {
        self.scan_workspace_bytes =
            Some(self.scan_workspace_bytes.unwrap_or(0).max(reserved_bytes));
        self.scan_workspace_scratch_bytes = Some(
            self.scan_workspace_scratch_bytes
                .unwrap_or(0)
                .max(scratch_bytes),
        );
        self.scan_workspace_output_bytes = Some(
            self.scan_workspace_output_bytes
                .unwrap_or(0)
                .max(output_bytes),
        );
        self.scan_workspace_growth_count = growth_count;
        self.scan_workspace_output_growth_count = output_growth_count;
        self.scan_workspace_step_fresh_allocations.push(step_fresh);
        self.scan_workspace_step_scratch_fresh
            .push(step_scratch_fresh);
        self.scan_workspace_step_output_fresh
            .push(step_output_fresh);
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
            StatusReadbackReason::ForwardAbort => {
                self.status_readbacks_forward_abort =
                    self.status_readbacks_forward_abort.saturating_add(1);
            }
        }
    }

    fn record_host_safety_point_abort(&mut self) {
        self.host_safety_point_aborts = self.host_safety_point_aborts.saturating_add(1);
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
        telemetry.status_readbacks_forward_abort = Some(self.status_readbacks_forward_abort);
        telemetry.status_readbacks_step_disposition = Some(self.status_readbacks_step_disposition);
        telemetry.capacity_telemetry_readbacks = Some(self.capacity_telemetry_readbacks);
        telemetry.loss_value_readbacks = Some(self.loss_value_readbacks);
        telemetry.checkpoint_tensor_readbacks = Some(self.checkpoint_tensor_readbacks);
        telemetry.gpu_gate_optimizer_skips = Some(self.gpu_gate_optimizer_skips);
        telemetry.gpu_gate_backward_skips = Some(self.gpu_gate_backward_skips);
        telemetry.gpu_gate_topology_skips = Some(self.gpu_gate_topology_skips);
        telemetry.host_safety_point_aborts = Some(self.host_safety_point_aborts);
        telemetry.radix_dispatch_count_p50 = percentile_usize(&self.sort_dispatches, 50.0);
        telemetry.radix_dispatch_count_p95 = percentile_usize(&self.sort_dispatches, 95.0);
        telemetry.scan_dispatch_count_p50 = percentile_usize(&self.scan_dispatches, 50.0);
        telemetry.scan_dispatch_count_p95 = percentile_usize(&self.scan_dispatches, 95.0);
        telemetry.sort_workspace_bytes = self.sort_workspace_bytes;
        telemetry.scan_workspace_bytes = self.scan_workspace_bytes;
        telemetry.scan_workspace_bytes_reason = if self.scan_workspace_bytes.is_some() {
            None
        } else {
            Some("scan_workspace_not_observed".into())
        };
        telemetry.scan_workspace_scratch_bytes = self.scan_workspace_scratch_bytes;
        telemetry.scan_workspace_output_bytes = self.scan_workspace_output_bytes;
        telemetry.scan_workspace_growth_count = Some(self.scan_workspace_growth_count);
        telemetry.scan_workspace_output_growth_count =
            Some(self.scan_workspace_output_growth_count);
        telemetry.scan_workspace_step_fresh_allocations_p50 =
            percentile_usize(&self.scan_workspace_step_fresh_allocations, 50.0);
        telemetry.scan_workspace_step_fresh_allocations_p95 =
            percentile_usize(&self.scan_workspace_step_fresh_allocations, 95.0);
        telemetry.scan_workspace_step_scratch_fresh_p50 =
            percentile_usize(&self.scan_workspace_step_scratch_fresh, 50.0);
        telemetry.scan_workspace_step_output_fresh_p50 =
            percentile_usize(&self.scan_workspace_step_output_fresh, 50.0);

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

        let mut pipeline_timing = PipelineTimingCollector::new(probe_training_device(&device));
        pipeline_timing.set_profiler_mode(
            config.profiler.enabled,
            config.profiler.gpu_timing_enabled,
            config.profiler.gpu_sample_every,
        );

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
            prefix_sum_workspace: PrefixSumWorkspace::new(),
            pipeline_timing,
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
        // Prior safety-point may already own sticky overflow / non-finite on the
        // host mirror. Refuse to start another logical iteration; refresh device
        // diagnostics under ForwardAbort without running forward/loss/backward.
        if self.device_status.host_snapshot().to_error().is_some() {
            return self
                .ensure_device_status_healthy(StatusReadbackReason::ForwardAbort)
                .await
                .map(|_| None);
        }

        self.prefix_sum_workspace.begin_step();

        let profile_step = log::log_enabled!(log::Level::Debug)
            && (iteration <= 3 || iteration.is_multiple_of(100));
        let step_started_at = Instant::now();
        let (width, height) = image_dims;
        let background = [0.0, 0.0, 0.0];
        let target_ready_elapsed = step_started_at.elapsed();

        let active_sh_degree = self.active_sh_degree_at(iteration, splats.sh_degree);
        self.telemetry.active_sh_degree = Some(active_sh_degree as usize);

        let forward_cpu = Instant::now();
        let profile_gpu = self.config.profiler.enabled
            && self.config.profiler.gpu_timing_enabled
            && self.pipeline_timing.probe().gpu_timing_supported()
            && should_profile_gpu(
                iteration,
                self.config.profiler.gpu_sample_every,
                Some(self.config.iterations),
            );
        let rendered = if profile_gpu {
            let device = self.device.clone();
            let capacity =
                self.intersection_capacity_for(splats.num_splats(), (width as u32, height as u32));
            let cov_blur = self.raster_cov_blur_at(iteration, frame_count);
            let status_buf = self.device_status.buffer().clone();
            let sh_degree = splats.sh_degree;
            let mut owned_splats =
                std::mem::replace(splats, empty_device_splats_placeholder(&device, sh_degree));
            let mut owned_ws = std::mem::take(&mut self.prefix_sum_workspace);
            let owned_camera = camera.clone();
            let ((stolen_splats, stolen_ws, rendered), gpu_ms) =
                profile_device_gpu_step(&device, move || async move {
                    let rendered = backward::render_splats_with_visibility_active_sh(
                        &mut owned_splats,
                        active_sh_degree,
                        &owned_camera,
                        (width as u32, height as u32),
                        background,
                        cov_blur,
                        capacity,
                        Some((iteration as u32, status_buf)),
                        Some(&mut owned_ws),
                    )
                    .await;
                    (owned_splats, owned_ws, rendered)
                })
                .await;
            *splats = stolen_splats;
            self.prefix_sum_workspace = stolen_ws;
            if let Some(ms) = gpu_ms {
                self.pipeline_timing.record_gpu_step_ms(ms);
            }
            rendered
        } else {
            backward::render_splats_with_visibility_active_sh(
                splats,
                active_sh_degree,
                camera,
                (width as u32, height as u32),
                background,
                self.raster_cov_blur_at(iteration, frame_count),
                self.intersection_capacity_for(splats.num_splats(), (width as u32, height as u32)),
                Some((iteration as u32, self.device_status.buffer().clone())),
                Some(&mut self.prefix_sum_workspace),
            )
            .await
        };
        self.pipeline_timing
            .record_span(span::FORWARD, forward_cpu.elapsed());
        self.optimization_samples.record_scan_workspace_stats(
            self.prefix_sum_workspace.reserved_bytes(),
            self.prefix_sum_workspace.scratch_bytes(),
            self.prefix_sum_workspace.output_bytes(),
            self.prefix_sum_workspace.growth_count(),
            self.prefix_sum_workspace.output_growth_count(),
            self.prefix_sum_workspace.step_fresh_allocations(),
            self.prefix_sum_workspace.step_allocations().scratch_fresh,
            self.prefix_sum_workspace.step_allocations().output_fresh,
        );
        self.pipeline_timing.observe_workspace(
            self.prefix_sum_workspace.reserved_bytes() as u64,
            self.prefix_sum_workspace.growth_count() as u64,
            self.prefix_sum_workspace.step_fresh_allocations() as u64,
        );
        self.pipeline_timing
            .observe_runtime_device_bytes(runtime_device_bytes_in_use(&self.device));
        debug_assert_eq!(
            self.prefix_sum_workspace.reserved_bytes(),
            self.prefix_sum_workspace
                .scratch_bytes()
                .saturating_add(self.prefix_sum_workspace.output_bytes()),
            "scan workspace bytes must equal scratch + output"
        );
        debug_assert!(
            self.prefix_sum_workspace.capacity() == 0
                || self.prefix_sum_workspace.output_capacity()
                    >= self.prefix_sum_workspace.capacity().min(1),
            "output capacity tracks owned scan buffer size"
        );
        // Overflow sticky bits are written on-device by write_dispatch together with
        // the same-step mutation_gate clear. Loss/backward/Adam/topology consult the
        // device status buffer directly — no mid-step host-mirror branch.
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
        let backward_cpu = Instant::now();
        let mut grads = loss.backward();
        self.pipeline_timing
            .record_span(span::BACKWARD, backward_cpu.elapsed());

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
        let optimizer_cpu = Instant::now();
        self.optimizer.step_device_splats(
            splats,
            transforms_grad,
            sh_grad,
            opacity_grad,
            self.device_status.buffer().clone(),
        );
        self.pipeline_timing
            .record_span(span::OPTIMIZER, optimizer_cpu.elapsed());
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
            match self
                .ensure_device_status_healthy(StatusReadbackReason::TopologyBoundary)
                .await
            {
                Ok(_) => {
                    self.apply_topology_mutations(splats, iteration, frame_count)
                        .await;
                }
                Err(err) => {
                    self.optimization_samples.gpu_gate_topology_skips = self
                        .optimization_samples
                        .gpu_gate_topology_skips
                        .saturating_add(1);
                    return Err(err);
                }
            }
        }

        if !read_loss {
            // Zero stepwise readback: do not probe status on healthy unread steps.
            // Device sticky flags + mutation_gate still block optimizer commits;
            // the outer loop confirms via word 4 at the next safety point.
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

    /// Confirm pending commits from device word 4 after a safety-point status sync.
    fn confirm_commits_from_host_mirror(
        &self,
        state: &mut CommitConfirmationState,
        report: &mut WgpuTrainingReport,
        splat_count: usize,
        last_loss: Option<f32>,
    ) -> Vec<usize> {
        let device_committed =
            self.device_status.host_snapshot().committed_optimizer_steps as usize;
        let newly = state.apply_device_committed(device_committed);
        if let Some(&last) = newly.last() {
            record_completed_step(report, last, splat_count, last_loss);
        }
        newly
    }

    /// Explicit committed/aborted/pending result for outer-loop accounting.
    pub async fn train_step_disposition(
        &mut self,
        splats: &mut DeviceSplats<GsDiffBackend>,
        camera: &GaussianCamera,
        target_img: Tensor<GsDiffBackend, 3>,
        image_dims: (usize, usize),
        iteration: usize,
        frame_count: usize,
        collect_topology_stats: bool,
        read_loss: bool,
    ) -> Result<TrainStepDisposition, TrainingError> {
        match self
            .train_step(
                splats,
                camera,
                target_img,
                image_dims,
                iteration,
                frame_count,
                collect_topology_stats,
                read_loss,
            )
            .await
        {
            Ok(loss) => {
                if read_loss {
                    Ok(TrainStepDisposition::ConfirmedCommitted { loss })
                } else {
                    Ok(TrainStepDisposition::SubmittedUnconfirmed)
                }
            }
            Err(error) => match error {
                TrainingError::ForwardCapacityExceeded { .. }
                | TrainingError::NonFiniteLoss { .. } => {
                    Ok(TrainStepDisposition::Aborted { error })
                }
                other => Err(other),
            },
        }
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
        let committed_baseline =
            self.device_status.host_snapshot().committed_optimizer_steps as usize;
        let mut commit_state = CommitConfirmationState::new(start_iteration, committed_baseline);

        for zero_based in start_iteration..num_iterations {
            if observer.should_cancel() {
                report.cancelled = true;
                report.disposition = TrainingRunDisposition::Cancelled;
                break;
            }

            let sample_idx = zero_based % cameras.len();
            let frame_idx = frame_order[sample_idx];
            let frame_wait_started = Instant::now();
            frame_loader.prefetch_order_window(frame_order, sample_idx)?;
            let decoded = frame_loader.get(frame_idx)?;
            self.pipeline_timing
                .record_span(span::FRAME_WAIT, frame_wait_started.elapsed());
            // Decode/resize happen on prefetch workers; attribute residual host
            // cache-miss preparation under decode/resize/upload without double-counting
            // into step_cpu child sums (step_cpu is the parent Instant for the iteration).
            let target_img = match target_tensor_cache.get(&frame_idx).cloned() {
                Some(cached) => {
                    touch_target_tensor_cache(&mut target_tensor_lru, frame_idx);
                    // Cache hit: decode/resize already ran on the prefetch worker.
                    // Do not push 0 ms samples that dilute real miss timings.
                    cached
                }
                None => {
                    // Prefer worker-measured decode/resize; fall back only when
                    // timings were not attached (should not happen for prefetch path).
                    if let Some(ms) = decoded.decode_ms {
                        self.pipeline_timing.record_span_ms(span::DECODE, ms);
                    }
                    if let Some(ms) = decoded.resize_ms {
                        self.pipeline_timing.record_span_ms(span::RESIZE, ms);
                    }
                    let target_image = decoded.target_rgb.clone().ok_or_else(|| {
                        TrainingError::TrainingFailed(format!(
                            "frame loader did not prepare target_rgb for frame {frame_idx}"
                        ))
                    })?;
                    let upload = CpuSpanTimer::start(span::UPLOAD);
                    let tensor = target_image_tensor(&target_image, image_dims, &self.device);
                    upload.finish(&mut self.pipeline_timing);
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
            let disposition = match self
                .train_step_disposition(
                    splats,
                    &cameras[sample_idx],
                    target_img,
                    image_dims,
                    iteration_idx,
                    cameras.len(),
                    collect_topology_stats,
                    read_loss,
                )
                .await
            {
                Ok(disposition) => disposition,
                Err(error) => {
                    self.finish_report(&mut report);
                    return Err(error);
                }
            };
            commit_state.note_submitted(iteration_idx);
            let (loss, newly_confirmed) = match disposition {
                TrainStepDisposition::ConfirmedCommitted { loss } => {
                    // Loss-cadence / checkpoint path already synced device status.
                    let newly = self.confirm_commits_from_host_mirror(
                        &mut commit_state,
                        &mut report,
                        splats.num_splats(),
                        loss,
                    );
                    (loss, newly)
                }
                TrainStepDisposition::SubmittedUnconfirmed => (None, Vec::new()),
                TrainStepDisposition::Aborted { error } => {
                    self.finish_report(&mut report);
                    return Err(error);
                }
            };
            let loop_duration = step_started_at.elapsed();
            self.pipeline_timing.record_cpu_step(loop_duration);
            self.optimization_samples.record_loop_step(
                loop_duration,
                read_loss,
                splats.num_splats(),
                self.intersection_capacity,
            );
            if let Some(loss) = loss {
                last_sampled_loss = loss;
            }
            if let Some(loss) = loss {
                self.record_loss_sample(
                    iteration_idx,
                    frame_idx,
                    loss,
                    should_log_step || iteration_idx == num_iterations,
                );
            }
            let metrics_for = |iteration: usize| TrainingIterationMetrics {
                iteration,
                loss: last_sampled_loss,
                gaussian_count: splats.num_splats(),
                loop_duration,
                loss_readback: read_loss && iteration == iteration_idx,
            };
            for confirmed in &newly_confirmed {
                if observer.should_emit_progress(*confirmed) {
                    observer.on_iteration(metrics_for(*confirmed));
                }
                if observer.should_cancel() {
                    report.cancelled = true;
                    report.disposition = TrainingRunDisposition::Cancelled;
                    break;
                }
            }
            if report.cancelled {
                break;
            }
            if newly_confirmed
                .iter()
                .any(|c| observer.should_emit_snapshot(*c))
            {
                let host = device_splats_to_host(splats).await;
                let snap_iter = *newly_confirmed
                    .iter()
                    .rev()
                    .find(|c| observer.should_emit_snapshot(**c))
                    .unwrap_or(&iteration_idx);
                observer.on_snapshot(metrics_for(snap_iter), host);
            }
            if should_log_step && commit_state.last_confirmed >= iteration_idx {
                log::info!(
                    "WGPU training step {} | loss={:.6} | splats={}",
                    iteration_idx,
                    last_sampled_loss,
                    splats.num_splats()
                );
            }

            if let Some(reason) = observer.checkpoint_reason(iteration_idx) {
                // Checkpoint is a safety point; only write after confirmation.
                if commit_state.last_confirmed < iteration_idx {
                    self.finish_report(&mut report);
                    return Err(TrainingError::TrainingFailed(format!(
                        "checkpoint at iteration {iteration_idx} requested before commit confirmation (confirmed={})",
                        commit_state.last_confirmed
                    )));
                }
                let identity = observer.checkpoint_identity().cloned().ok_or_else(|| {
                    TrainingError::InvalidInput(
                        "checkpointing training requires the current training identity".to_string(),
                    )
                })?;
                let checkpoint = self
                    .checkpoint_with_status_reason(
                        splats,
                        identity,
                        commit_state.last_confirmed,
                        Some(last_sampled_loss),
                        Self::checkpoint_status_reason(reason),
                    )
                    .await?;
                if let Some(disposition) = complete_checkpoint_boundary(
                    observer,
                    TrainingCheckpointReady {
                        iteration: commit_state.last_confirmed,
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
        match self
            .ensure_device_status_healthy(StatusReadbackReason::TrainingEnd)
            .await
        {
            Ok(_) => {
                let end_loss = report.final_loss.or(Some(last_sampled_loss));
                let _ = self.confirm_commits_from_host_mirror(
                    &mut commit_state,
                    &mut report,
                    splats.num_splats(),
                    end_loss,
                );
            }
            Err(error) => {
                self.finish_report(&mut report);
                return Err(error);
            }
        }
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

        let overflow_err = self
            .train_step(splats, camera, target, image_dims, 2, 1, true, false)
            .await
            .expect_err("capacity-1 overflow must abort even when loss is unread");
        assert!(
            matches!(overflow_err, TrainingError::ForwardCapacityExceeded { .. }),
            "got {overflow_err:?}"
        );

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

    async fn ensure_device_status_healthy(
        &mut self,
        reason: StatusReadbackReason,
    ) -> Result<TrainingStatusSnapshot, TrainingError> {
        let status = self.device_status.read().await?;
        self.device_status.adopt_device_snapshot(status);
        // Once sticky flags are set on device, classify diagnostic reads as
        // ForwardAbort so healthy LossCadence counters stay honest.
        let effective_reason = if !status.is_healthy()
            && matches!(
                reason,
                StatusReadbackReason::LossCadence
                    | StatusReadbackReason::TopologyBoundary
                    | StatusReadbackReason::TrainingEnd
            ) {
            StatusReadbackReason::ForwardAbort
        } else {
            reason
        };
        self.optimization_samples
            .record_status_readback(effective_reason);
        self.optimizer
            .sync_committed_steps(status.committed_optimizer_steps as usize);
        self.optimization_samples.gpu_gate_optimizer_skips =
            status.gpu_gate_optimizer_skips as usize;
        if let Some(err) = status.to_error() {
            self.optimization_samples.record_host_safety_point_abort();
            if status.has_forward_overflow() || status.has_non_finite_loss() {
                self.optimization_samples.gpu_gate_backward_skips = self
                    .optimization_samples
                    .gpu_gate_backward_skips
                    .saturating_add(1);
            }
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
        let snapshot_elapsed = snapshot_started.elapsed();
        self.optimization_samples
            .topology_snapshot_ms
            .push(duration_millis(snapshot_elapsed));
        self.pipeline_timing
            .record_span(span::TOPOLOGY_SNAPSHOT, snapshot_elapsed);
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
        let plan_elapsed = plan_started.elapsed();
        self.optimization_samples
            .topology_plan_ms
            .push(duration_millis(plan_elapsed));
        self.pipeline_timing
            .record_span(span::TOPOLOGY_PLAN, plan_elapsed);
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
            let upload_started = Instant::now();
            apply_mutations(splats, &snapshot.splats, &plan, &self.device);
            self.pipeline_timing
                .record_span(span::TOPOLOGY_UPLOAD, upload_started.elapsed());
            let remap_started = Instant::now();
            self.remap_topology_visibility_state(&plan, iteration);
            self.pipeline_timing
                .record_span(span::TOPOLOGY_REMAP_VISIBILITY, remap_started.elapsed());
        }
        if plan.aftermath.requires_adam_rebuild {
            let sh_dims = splats.sh_coeffs.val().dims();
            let remap_started = Instant::now();
            self.optimizer.remap_origins(
                &plan.origins(),
                sh_dims[1],
                sh_dims.get(2).copied().unwrap_or(3),
                &self.device,
            );
            self.pipeline_timing
                .record_span(span::TOPOLOGY_REMAP_OPTIMIZER, remap_started.elapsed());
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
        let apply_elapsed = apply_started.elapsed();
        self.optimization_samples
            .topology_apply_ms
            .push(duration_millis(apply_elapsed));
        self.pipeline_timing
            .record_span(span::TOPOLOGY_APPLY, apply_elapsed);
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
        self.pipeline_timing
            .observe_runtime_device_bytes(runtime_device_bytes_in_use(&self.device));
        let gpu_report = self.pipeline_timing.build_report();
        self.telemetry.gpu_profiler = Some(gpu_report);
        report.telemetry = self.telemetry.clone();
    }

    /// Replace the environment probe (e.g. SharedWgpuContext adapter metadata).
    pub(crate) fn set_pipeline_probe(
        &mut self,
        probe: crate::training::reporting::gpu_profiler::GpuEnvironmentProbe,
    ) {
        self.pipeline_timing.set_probe(probe);
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

    #[derive(Debug)]
    struct SyntheticOuterLoopAbort {
        error: TrainingError,
        report: Box<WgpuTrainingReport>,
    }

    /// Mirror train_with_frame_loader commit rules without PrefetchFrameLoader.
    async fn run_synthetic_outer_loop(
        trainer: &mut WgpuTrainer,
        splats: &mut DeviceSplats<GsDiffBackend>,
        camera: &GaussianCamera,
        device: &GsDevice,
        start_iteration: usize,
        num_iterations: usize,
        target_fill: f32,
        force_capacity: Option<usize>,
        observer: &mut dyn TrainingLoopObserver,
    ) -> Result<WgpuTrainingReport, SyntheticOuterLoopAbort> {
        let mut report = WgpuTrainingReport {
            completed_iterations: start_iteration,
            final_gaussian_count: splats.num_splats(),
            ..Default::default()
        };
        let mut last_sampled_loss = 0.0;
        if let Some(capacity) = force_capacity {
            trainer.force_intersection_capacity_for_test(capacity);
        }
        let committed_baseline = trainer
            .device_status
            .host_snapshot()
            .committed_optimizer_steps as usize;
        let mut commit_state = CommitConfirmationState::new(start_iteration, committed_baseline);
        for zero_based in start_iteration..num_iterations {
            if observer.should_cancel() {
                report.cancelled = true;
                report.disposition = TrainingRunDisposition::Cancelled;
                break;
            }
            let iteration_idx = zero_based + 1;
            let checkpoint_due = observer.checkpoint_reason(iteration_idx).is_some();
            let read_loss = should_read_loss(
                iteration_idx,
                num_iterations,
                LOSS_SCALAR_READBACK_INTERVAL,
                checkpoint_due,
                observer.should_pause(),
            );
            let disposition = match trainer
                .train_step_disposition(
                    splats,
                    camera,
                    fault_injection_target(device, target_fill),
                    (8, 8),
                    iteration_idx,
                    1,
                    false,
                    read_loss,
                )
                .await
            {
                Ok(disposition) => disposition,
                Err(error) => {
                    trainer.finish_report(&mut report);
                    return Err(SyntheticOuterLoopAbort {
                        error,
                        report: Box::new(report),
                    });
                }
            };
            commit_state.note_submitted(iteration_idx);
            let (loss, newly_confirmed) = match disposition {
                TrainStepDisposition::ConfirmedCommitted { loss } => {
                    let newly = trainer.confirm_commits_from_host_mirror(
                        &mut commit_state,
                        &mut report,
                        splats.num_splats(),
                        loss,
                    );
                    (loss, newly)
                }
                TrainStepDisposition::SubmittedUnconfirmed => (None, Vec::new()),
                TrainStepDisposition::Aborted { error } => {
                    trainer.finish_report(&mut report);
                    return Err(SyntheticOuterLoopAbort {
                        error,
                        report: Box::new(report),
                    });
                }
            };
            if let Some(loss) = loss {
                last_sampled_loss = loss;
            }
            let metrics_for = |iteration: usize| TrainingIterationMetrics {
                iteration,
                loss: last_sampled_loss,
                gaussian_count: splats.num_splats(),
                loop_duration: Duration::from_millis(0),
                loss_readback: read_loss && iteration == iteration_idx,
            };
            for confirmed in &newly_confirmed {
                if observer.should_emit_progress(*confirmed) {
                    observer.on_iteration(metrics_for(*confirmed));
                }
                if observer.should_cancel() {
                    report.cancelled = true;
                    report.disposition = TrainingRunDisposition::Cancelled;
                    break;
                }
            }
            if report.cancelled {
                break;
            }
            if newly_confirmed
                .iter()
                .any(|c| observer.should_emit_snapshot(*c))
            {
                let host = device_splats_to_host(splats).await;
                let snap_iter = *newly_confirmed
                    .iter()
                    .rev()
                    .find(|c| observer.should_emit_snapshot(**c))
                    .unwrap_or(&iteration_idx);
                observer.on_snapshot(metrics_for(snap_iter), host);
            }
            if let Some(reason) = observer.checkpoint_reason(iteration_idx) {
                if commit_state.last_confirmed < iteration_idx {
                    trainer.finish_report(&mut report);
                    return Err(SyntheticOuterLoopAbort {
                        error: TrainingError::TrainingFailed(format!(
                            "checkpoint at iteration {iteration_idx} before confirmation"
                        )),
                        report: Box::new(report),
                    });
                }
                let identity = match observer.checkpoint_identity().cloned() {
                    Some(identity) => identity,
                    None => {
                        trainer.finish_report(&mut report);
                        return Err(SyntheticOuterLoopAbort {
                            error: TrainingError::InvalidInput(
                                "checkpointing training requires the current training identity"
                                    .to_string(),
                            ),
                            report: Box::new(report),
                        });
                    }
                };
                let checkpoint = match trainer
                    .checkpoint_with_status_reason(
                        splats,
                        identity,
                        commit_state.last_confirmed,
                        Some(last_sampled_loss),
                        WgpuTrainer::checkpoint_status_reason(reason),
                    )
                    .await
                {
                    Ok(checkpoint) => checkpoint,
                    Err(error) => {
                        trainer.finish_report(&mut report);
                        return Err(SyntheticOuterLoopAbort {
                            error,
                            report: Box::new(report),
                        });
                    }
                };
                match complete_checkpoint_boundary(
                    observer,
                    TrainingCheckpointReady {
                        iteration: commit_state.last_confirmed,
                        reason,
                        checkpoint,
                    },
                ) {
                    Ok(Some(disposition)) => {
                        report.cancelled = disposition == TrainingRunDisposition::Cancelled;
                        report.disposition = disposition;
                        break;
                    }
                    Ok(None) => {}
                    Err(error) => {
                        trainer.finish_report(&mut report);
                        return Err(SyntheticOuterLoopAbort {
                            error,
                            report: Box::new(report),
                        });
                    }
                }
            }
        }
        match trainer
            .ensure_device_status_healthy(StatusReadbackReason::TrainingEnd)
            .await
        {
            Ok(_) => {
                let end_loss = report.final_loss.or(Some(last_sampled_loss));
                let _ = trainer.confirm_commits_from_host_mirror(
                    &mut commit_state,
                    &mut report,
                    splats.num_splats(),
                    end_loss,
                );
            }
            Err(error) => {
                trainer.finish_report(&mut report);
                return Err(SyntheticOuterLoopAbort {
                    error,
                    report: Box::new(report),
                });
            }
        }
        trainer.finish_report(&mut report);
        Ok(report)
    }

    struct OuterLoopProbeObserver {
        progress_iters: Vec<usize>,
        snapshot_iters: Vec<usize>,
        checkpoint_iters: Vec<usize>,
        cancel_after: Option<usize>,
        pause_at: Option<usize>,
        checkpoint_every: Option<usize>,
        identity: TrainingIdentity,
        last_checkpoint: Option<TrainingCheckpoint>,
    }

    impl OuterLoopProbeObserver {
        fn new() -> Self {
            Self {
                progress_iters: Vec::new(),
                snapshot_iters: Vec::new(),
                checkpoint_iters: Vec::new(),
                cancel_after: None,
                pause_at: None,
                checkpoint_every: None,
                identity: trainer_checkpoint_identity(),
                last_checkpoint: None,
            }
        }
    }

    impl TrainingLoopObserver for OuterLoopProbeObserver {
        fn should_cancel(&self) -> bool {
            self.cancel_after
                .is_some_and(|after| self.progress_iters.last().copied().unwrap_or(0) >= after)
        }

        fn should_pause(&self) -> bool {
            self.pause_at
                .is_some_and(|at| self.progress_iters.last().copied().unwrap_or(0) >= at)
        }

        fn should_emit_progress(&self, _iteration: usize) -> bool {
            true
        }

        fn should_emit_snapshot(&self, iteration: usize) -> bool {
            iteration == 1
                || self
                    .checkpoint_every
                    .is_some_and(|every| iteration.is_multiple_of(every))
        }

        fn checkpoint_reason(&self, iteration: usize) -> Option<TrainingCheckpointReason> {
            if self.pause_at == Some(iteration) {
                return Some(TrainingCheckpointReason::Pause);
            }
            self.checkpoint_every.and_then(|every| {
                iteration
                    .is_multiple_of(every)
                    .then_some(TrainingCheckpointReason::Periodic)
            })
        }

        fn checkpoint_identity(&self) -> Option<&TrainingIdentity> {
            Some(&self.identity)
        }

        fn on_iteration(&mut self, metrics: TrainingIterationMetrics) {
            self.progress_iters.push(metrics.iteration);
        }

        fn on_snapshot(&mut self, metrics: TrainingIterationMetrics, _splats: HostSplats) {
            self.snapshot_iters.push(metrics.iteration);
        }

        fn on_checkpoint(&mut self, ready: TrainingCheckpointReady) -> Result<(), TrainingError> {
            self.checkpoint_iters.push(ready.iteration);
            self.last_checkpoint = Some(ready.checkpoint);
            Ok(())
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn outer_loop_overflow_read_loss_false_keeps_completed_iterations() {
        let device = GsDevice::default();
        let config = fault_injection_config();
        let host_splats = trainer_checkpoint_host_splats();
        let mut splats = host_splats_to_device::<GsDiffBackend>(&host_splats, &device);
        let mut trainer = WgpuTrainer::new(config, device.clone(), 3, 4, 2.5);
        install_trainer_checkpoint_state(&mut trainer);
        let camera = fault_injection_camera();
        let mut observer = OuterLoopProbeObserver::new();

        // Commit iteration 1 with normal capacity.
        let healthy = run_synthetic_outer_loop(
            &mut trainer,
            &mut splats,
            &camera,
            &device,
            0,
            1,
            0.4,
            None,
            &mut observer,
        )
        .await
        .expect("healthy first iteration");
        assert_eq!(healthy.completed_iterations, 1);
        assert_eq!(observer.progress_iters, vec![1]);

        let before = snapshot_mutation_state(&mut trainer, &splats).await;
        let mut observer = OuterLoopProbeObserver::new();
        let aborted = run_synthetic_outer_loop(
            &mut trainer,
            &mut splats,
            &camera,
            &device,
            1,
            3,
            0.4,
            Some(1),
            &mut observer,
        )
        .await
        .expect_err("overflow with unread loss must abort outer loop");
        assert!(
            matches!(aborted.error, TrainingError::ForwardCapacityExceeded { .. }),
            "got {:?}",
            aborted.error
        );
        assert_eq!(
            aborted.report.completed_iterations, 1,
            "aborted unread overflow must not advance completed_iterations"
        );
        let after = snapshot_mutation_state(&mut trainer, &splats).await;
        assert_eq!(after.0, before.0, "transforms must not change");
        assert_eq!(after.3, before.3, "adam must not change");
        assert_eq!(after.4, before.4, "topology accumulators must not change");
        assert_eq!(after.6, before.6);
        assert_eq!(after.7, before.7);
        assert_eq!(
            after.5.committed_optimizer_steps,
            before.5.committed_optimizer_steps
        );
        assert!(observer.progress_iters.is_empty());
        assert!(observer.snapshot_iters.is_empty());
        assert!(observer.checkpoint_iters.is_empty());
        assert_eq!(after.5.first_invalid_iteration, 2);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn outer_loop_non_finite_read_loss_false_keeps_completed_iterations() {
        let device = GsDevice::default();
        let config = fault_injection_config();
        let host_splats = trainer_checkpoint_host_splats();
        let mut splats = host_splats_to_device::<GsDiffBackend>(&host_splats, &device);
        let mut trainer = WgpuTrainer::new(config, device.clone(), 3, 4, 2.5);
        install_trainer_checkpoint_state(&mut trainer);
        let camera = fault_injection_camera();
        let mut observer = OuterLoopProbeObserver::new();
        let healthy = run_synthetic_outer_loop(
            &mut trainer,
            &mut splats,
            &camera,
            &device,
            0,
            1,
            0.4,
            None,
            &mut observer,
        )
        .await
        .expect("healthy first iteration");
        assert_eq!(healthy.completed_iterations, 1);

        let before = snapshot_mutation_state(&mut trainer, &splats).await;
        let mut observer = OuterLoopProbeObserver::new();
        let aborted = run_synthetic_outer_loop(
            &mut trainer,
            &mut splats,
            &camera,
            &device,
            1,
            3,
            f32::NAN,
            None,
            &mut observer,
        )
        .await
        .expect_err("non-finite unread loss must abort outer loop");
        assert!(matches!(aborted.error, TrainingError::NonFiniteLoss { .. }));
        assert_eq!(aborted.report.completed_iterations, 1);
        let after = snapshot_mutation_state(&mut trainer, &splats).await;
        assert_eq!(after.0, before.0);
        assert_eq!(after.3, before.3);
        assert_eq!(
            after.5.committed_optimizer_steps,
            before.5.committed_optimizer_steps
        );
        assert!(observer.progress_iters.is_empty());
        assert!(observer.checkpoint_iters.is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn outer_loop_continuous_overflow_never_advances_report() {
        let device = GsDevice::default();
        let config = fault_injection_config();
        let host_splats = trainer_checkpoint_host_splats();
        let mut splats = host_splats_to_device::<GsDiffBackend>(&host_splats, &device);
        let mut trainer = WgpuTrainer::new(config, device.clone(), 3, 4, 2.5);
        let camera = fault_injection_camera();
        let mut observer = OuterLoopProbeObserver::new();
        let healthy = run_synthetic_outer_loop(
            &mut trainer,
            &mut splats,
            &camera,
            &device,
            0,
            1,
            0.4,
            None,
            &mut observer,
        )
        .await
        .expect("healthy");
        assert_eq!(healthy.completed_iterations, 1);

        for _ in 0..3 {
            let mut observer = OuterLoopProbeObserver::new();
            let err = run_synthetic_outer_loop(
                &mut trainer,
                &mut splats,
                &camera,
                &device,
                1,
                4,
                0.4,
                Some(1),
                &mut observer,
            )
            .await
            .expect_err("sticky overflow");
            assert!(matches!(
                err.error,
                TrainingError::ForwardCapacityExceeded {
                    first_iteration: 2,
                    ..
                }
            ));
            assert_eq!(err.report.completed_iterations, 1);
            assert!(observer.progress_iters.is_empty());
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn outer_loop_committed_unread_loss_advances_and_keeps_cadence() {
        let device = GsDevice::default();
        let config = fault_injection_config();
        let host_splats = trainer_checkpoint_host_splats();
        let mut splats = host_splats_to_device::<GsDiffBackend>(&host_splats, &device);
        let mut trainer = WgpuTrainer::new(config, device.clone(), 3, 4, 2.5);
        let camera = fault_injection_camera();
        let mut observer = OuterLoopProbeObserver::new();
        let report = run_synthetic_outer_loop(
            &mut trainer,
            &mut splats,
            &camera,
            &device,
            0,
            5,
            0.4,
            None,
            &mut observer,
        )
        .await
        .expect("committed unread steps");
        assert_eq!(report.completed_iterations, 5);
        assert_eq!(observer.progress_iters, vec![1, 2, 3, 4, 5]);
        assert_eq!(
            trainer.optimization_samples.loss_value_readbacks, 2,
            "iterations 1 and 5 are on the loss cadence for a 5-step run"
        );
        assert_eq!(
            trainer
                .optimization_samples
                .status_readbacks_step_disposition,
            0,
            "healthy unread steps must not perform StepDisposition status readbacks"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn outer_loop_cancel_and_checkpoint_use_committed_iterations_only() {
        let device = GsDevice::default();
        let config = fault_injection_config();
        let host_splats = trainer_checkpoint_host_splats();
        let mut splats = host_splats_to_device::<GsDiffBackend>(&host_splats, &device);
        let mut trainer = WgpuTrainer::new(config, device.clone(), 3, 4, 2.5);
        let camera = fault_injection_camera();

        let mut observer = OuterLoopProbeObserver::new();
        observer.checkpoint_every = Some(2);
        let report = run_synthetic_outer_loop(
            &mut trainer,
            &mut splats,
            &camera,
            &device,
            0,
            2,
            0.4,
            None,
            &mut observer,
        )
        .await
        .expect("checkpoint boundary");
        assert_eq!(report.completed_iterations, 2);
        assert_eq!(observer.checkpoint_iters, vec![2]);
        assert_eq!(observer.snapshot_iters, vec![1, 2]);
        let checkpoint = observer
            .last_checkpoint
            .expect("periodic checkpoint must commit");
        assert_eq!(checkpoint.completed_iterations, 2);

        let mut observer = OuterLoopProbeObserver::new();
        // Cancel once progress reaches iteration 3 (confirmed via loss/end cadence).
        observer.cancel_after = Some(3);
        let cancelled = run_synthetic_outer_loop(
            &mut trainer,
            &mut splats,
            &camera,
            &device,
            2,
            3,
            0.4,
            None,
            &mut observer,
        )
        .await
        .expect("cancel after committed progress");
        assert!(cancelled.cancelled);
        assert_eq!(cancelled.disposition, TrainingRunDisposition::Cancelled);
        assert_eq!(cancelled.completed_iterations, 3);
        assert_eq!(observer.progress_iters, vec![3]);
        assert!(observer.checkpoint_iters.is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn outer_loop_pause_checkpoint_keeps_committed_iterations() {
        let device = GsDevice::default();
        let config = fault_injection_config();
        let host_splats = trainer_checkpoint_host_splats();
        let mut splats = host_splats_to_device::<GsDiffBackend>(&host_splats, &device);
        let mut trainer = WgpuTrainer::new(config, device.clone(), 3, 4, 2.5);
        let camera = fault_injection_camera();
        let mut observer = OuterLoopProbeObserver::new();
        observer.pause_at = Some(2);
        let paused = run_synthetic_outer_loop(
            &mut trainer,
            &mut splats,
            &camera,
            &device,
            0,
            5,
            0.4,
            None,
            &mut observer,
        )
        .await
        .expect("pause checkpoint boundary");
        assert_eq!(paused.disposition, TrainingRunDisposition::Paused);
        assert_eq!(paused.completed_iterations, 2);
        assert_eq!(observer.checkpoint_iters, vec![2]);
        let checkpoint = observer
            .last_checkpoint
            .expect("pause must emit checkpoint");
        assert_eq!(checkpoint.completed_iterations, 2);
        assert_eq!(observer.progress_iters, vec![1, 2]);
    }

    fn production_outer_loop_fixture(
        config: &TrainingConfig,
    ) -> (
        tempfile::TempDir,
        PrefetchFrameLoader,
        Vec<GaussianCamera>,
        Vec<usize>,
    ) {
        use crate::training::data::frame_loader::FrameLoaderOptions;
        use crate::{Intrinsics, ScenePose, TrainingDataset, SE3};

        let temp = tempfile::tempdir().expect("tempdir");
        let image_path = temp.path().join("frame.rgb");
        std::fs::write(&image_path, vec![40u8; 8 * 8 * 3]).expect("write rgb");
        let mut dataset = TrainingDataset::new(Intrinsics::new(8.0, 8.0, 4.0, 4.0, 8, 8));
        dataset.add_pose(ScenePose::new(0, image_path, SE3::identity(), 0.0));
        let loader = PrefetchFrameLoader::new(
            &dataset,
            config,
            FrameLoaderOptions {
                cache_capacity: 2,
                prefetch_ahead: 1,
                rgb_target_size: Some((8, 8)),
            },
        )
        .expect("prefetch loader");
        (temp, loader, vec![fault_injection_camera()], vec![0usize])
    }

    #[tokio::test(flavor = "current_thread")]
    async fn production_outer_loop_overflow_keeps_completed_iterations() {
        let device = GsDevice::default();
        let config = fault_injection_config();
        let host_splats = trainer_checkpoint_host_splats();
        let mut splats = host_splats_to_device::<GsDiffBackend>(&host_splats, &device);
        let mut trainer = WgpuTrainer::new(config.clone(), device.clone(), 3, 4, 2.5);
        install_trainer_checkpoint_state(&mut trainer);
        let (_temp, mut loader, cameras, order) = production_outer_loop_fixture(&config);
        let mut observer = OuterLoopProbeObserver::new();

        let healthy = trainer
            .train_with_frame_loader(
                &mut splats,
                &cameras,
                &order,
                &mut loader,
                (8, 8),
                0,
                1,
                &mut observer,
            )
            .await
            .expect("healthy production iteration");
        assert_eq!(healthy.completed_iterations, 1);
        assert_eq!(observer.progress_iters, vec![1]);

        trainer.force_intersection_capacity_for_test(1);
        let before = snapshot_mutation_state(&mut trainer, &splats).await;
        let mut observer = OuterLoopProbeObserver::new();
        let err = trainer
            .train_with_frame_loader(
                &mut splats,
                &cameras,
                &order,
                &mut loader,
                (8, 8),
                1,
                3,
                &mut observer,
            )
            .await
            .expect_err("production overflow must abort");
        assert!(
            matches!(err, TrainingError::ForwardCapacityExceeded { .. }),
            "got {err:?}"
        );
        // Host report is only available via finish_report path inside Err; re-run
        // status from trainer telemetry / device mirror.
        assert_eq!(
            trainer
                .device_status
                .host_snapshot()
                .committed_optimizer_steps,
            before.5.committed_optimizer_steps
        );
        assert!(observer.progress_iters.is_empty());
        let after = snapshot_mutation_state(&mut trainer, &splats).await;
        assert_eq!(after.0, before.0);
        assert_eq!(after.3, before.3);
        assert_eq!(
            after.5.committed_optimizer_steps,
            before.5.committed_optimizer_steps
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn production_outer_loop_non_finite_keeps_completed_iterations() {
        use crate::training::engine::device_status::{
            TrainingStatusSnapshot, STATUS_NON_FINITE_LOSS,
        };

        let device = GsDevice::default();
        let config = fault_injection_config();
        let host_splats = trainer_checkpoint_host_splats();
        let mut splats = host_splats_to_device::<GsDiffBackend>(&host_splats, &device);
        let mut trainer = WgpuTrainer::new(config.clone(), device.clone(), 3, 4, 2.5);
        install_trainer_checkpoint_state(&mut trainer);
        let (_temp, mut loader, cameras, order) = production_outer_loop_fixture(&config);
        let mut observer = OuterLoopProbeObserver::new();
        let healthy = trainer
            .train_with_frame_loader(
                &mut splats,
                &cameras,
                &order,
                &mut loader,
                (8, 8),
                0,
                1,
                &mut observer,
            )
            .await
            .expect("healthy production iteration");
        assert_eq!(healthy.completed_iterations, 1);

        let before = snapshot_mutation_state(&mut trainer, &splats).await;
        let mut sticky = TrainingStatusSnapshot::default();
        sticky.flags = STATUS_NON_FINITE_LOSS;
        sticky.first_invalid_iteration = 2;
        sticky.committed_optimizer_steps = before.5.committed_optimizer_steps;
        trainer.device_status.set_host_snapshot(sticky);

        let mut observer = OuterLoopProbeObserver::new();
        let err = trainer
            .train_with_frame_loader(
                &mut splats,
                &cameras,
                &order,
                &mut loader,
                (8, 8),
                1,
                3,
                &mut observer,
            )
            .await
            .expect_err("sticky non-finite must abort production loop");
        assert!(
            matches!(err, TrainingError::NonFiniteLoss { .. }),
            "got {err:?}"
        );
        assert!(observer.progress_iters.is_empty());
        let after = snapshot_mutation_state(&mut trainer, &splats).await;
        assert_eq!(
            after.5.committed_optimizer_steps,
            before.5.committed_optimizer_steps
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn status_readback_reason_totals_are_recomputable() {
        let device = GsDevice::default();
        let mut config = fault_injection_config();
        config.iterations = 20;
        let host_splats = trainer_checkpoint_host_splats();
        let mut splats = host_splats_to_device::<GsDiffBackend>(&host_splats, &device);
        let mut trainer = WgpuTrainer::new(config, device.clone(), 3, 4, 2.5);
        let camera = fault_injection_camera();
        for iteration in 1..=20 {
            let read_loss =
                should_read_loss(iteration, 20, LOSS_SCALAR_READBACK_INTERVAL, false, false);
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
                .expect("step");
        }
        trainer
            .ensure_device_status_healthy(StatusReadbackReason::TrainingEnd)
            .await
            .expect("end");
        let samples = &trainer.optimization_samples;
        let parts = samples.status_readbacks_loss_cadence
            + samples.status_readbacks_topology
            + samples.status_readbacks_checkpoint
            + samples.status_readbacks_pause
            + samples.status_readbacks_cancel
            + samples.status_readbacks_training_end
            + samples.status_readbacks_forward_abort
            + samples.status_readbacks_step_disposition;
        assert_eq!(samples.status_readbacks, parts);
        assert_eq!(samples.status_readbacks_step_disposition, 0);
        let mut report = WgpuTrainingReport::default();
        trainer.finish_report(&mut report);
        let telemetry = report.telemetry;
        let telem_parts = telemetry.status_readbacks_loss_cadence.unwrap_or(0)
            + telemetry.status_readbacks_topology.unwrap_or(0)
            + telemetry.status_readbacks_checkpoint.unwrap_or(0)
            + telemetry.status_readbacks_pause.unwrap_or(0)
            + telemetry.status_readbacks_cancel.unwrap_or(0)
            + telemetry.status_readbacks_training_end.unwrap_or(0)
            + telemetry.status_readbacks_forward_abort.unwrap_or(0)
            + telemetry.status_readbacks_step_disposition.unwrap_or(0);
        assert_eq!(telemetry.status_readbacks, Some(telem_parts));
        assert_eq!(telemetry.status_readbacks_step_disposition, Some(0));
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

        // C2: unread steps return Ok(None) without status readback; device gate
        // still blocks mutation. Sticky NonFinite surfaces at the next safety point.
        let unread = trainer
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
            .expect("unread non-finite stays SubmittedUnconfirmed");
        assert!(unread.is_none());

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
        // snapshot_mutation_state syncs status; sticky must be visible after that.
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

        let unread = trainer
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
            .expect("unread overflow stays SubmittedUnconfirmed");
        assert!(unread.is_none());

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
        assert_eq!(
            after.5.mutation_gate, 0,
            "overflow step must leave mutation_gate blocked"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn train_step_continuous_overflow_with_read_loss_false_mutates_nothing() {
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

        for iteration in 2..=4 {
            let unread = trainer
                .train_step(
                    &mut splats,
                    &camera,
                    fault_injection_target(&device, 0.4),
                    (8, 8),
                    iteration,
                    1,
                    true,
                    false,
                )
                .await
                .expect("continuous unread overflow stays SubmittedUnconfirmed");
            assert!(
                unread.is_none(),
                "iteration {iteration} must not invent a loss scalar"
            );
        }

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
        assert_eq!(
            after.5.first_invalid_iteration, 2,
            "sticky first overflow iteration must survive continuous overflows"
        );
        assert_eq!(after.5.mutation_gate, 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn sticky_host_overflow_aborts_next_step_without_mutation() {
        let device = GsDevice::default();
        let config = fault_injection_config();
        let host_splats = trainer_checkpoint_host_splats();
        let mut splats = host_splats_to_device::<GsDiffBackend>(&host_splats, &device);
        let mut trainer = WgpuTrainer::new(config, device.clone(), 3, 4, 2.5);
        install_trainer_checkpoint_state(&mut trainer);
        let camera = fault_injection_camera();

        trainer
            .train_step(
                &mut splats,
                &camera,
                fault_injection_target(&device, 0.4),
                (8, 8),
                1,
                1,
                true,
                true,
            )
            .await
            .expect("healthy step with loss read");

        trainer.intersection_capacity_override = Some(1);
        let overflow_err = trainer
            .train_step(
                &mut splats,
                &camera,
                fault_injection_target(&device, 0.4),
                (8, 8),
                2,
                1,
                true,
                true,
            )
            .await
            .expect_err("loss-cadence overflow must surface ForwardCapacityExceeded");
        assert!(
            matches!(
                overflow_err,
                TrainingError::ForwardCapacityExceeded {
                    first_iteration: 2,
                    ..
                }
            ),
            "got {overflow_err:?}"
        );
        assert!(trainer.optimization_samples.status_readbacks_forward_abort >= 1);
        assert!(trainer.optimization_samples.host_safety_point_aborts >= 1);
        assert!(trainer.optimization_samples.gpu_gate_optimizer_skips >= 1);
        assert!(trainer.optimization_samples.gpu_gate_backward_skips >= 1);

        let before = snapshot_mutation_state(&mut trainer, &splats).await;
        let aborted = trainer
            .train_step(
                &mut splats,
                &camera,
                fault_injection_target(&device, 0.4),
                (8, 8),
                3,
                1,
                true,
                false,
            )
            .await
            .expect_err("sticky host overflow must abort the next logical iteration");
        assert!(
            matches!(aborted, TrainingError::ForwardCapacityExceeded { .. }),
            "got {aborted:?}"
        );
        let after = snapshot_mutation_state(&mut trainer, &splats).await;
        assert_eq!(after.0, before.0);
        assert_eq!(after.3, before.3);
        assert_eq!(
            after.5.committed_optimizer_steps,
            before.5.committed_optimizer_steps
        );
        assert!(trainer.optimization_samples.status_readbacks_forward_abort >= 2);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn train_step_nan_with_read_loss_returns_unique_non_finite_error() {
        let device = GsDevice::default();
        let config = fault_injection_config();
        let host_splats = trainer_checkpoint_host_splats();
        let mut splats = host_splats_to_device::<GsDiffBackend>(&host_splats, &device);
        let mut trainer = WgpuTrainer::new(config, device.clone(), 3, 4, 2.5);
        install_trainer_checkpoint_state(&mut trainer);
        let camera = fault_injection_camera();

        trainer
            .train_step(
                &mut splats,
                &camera,
                fault_injection_target(&device, 0.4),
                (8, 8),
                1,
                1,
                true,
                true,
            )
            .await
            .expect("healthy step");

        let err = trainer
            .train_step(
                &mut splats,
                &camera,
                fault_injection_target(&device, f32::NAN),
                (8, 8),
                2,
                1,
                true,
                true,
            )
            .await
            .expect_err("loss-cadence NaN must surface NonFiniteLoss");
        assert!(
            matches!(err, TrainingError::NonFiniteLoss { first_iteration: 2 }),
            "got {err:?}"
        );
        assert!(trainer.optimization_samples.host_safety_point_aborts >= 1);
        assert!(trainer.optimization_samples.gpu_gate_optimizer_skips >= 1);

        let follow = trainer
            .train_step(
                &mut splats,
                &camera,
                fault_injection_target(&device, 0.4),
                (8, 8),
                3,
                1,
                true,
                false,
            )
            .await
            .expect_err("sticky non-finite must not be cleared by a later step");
        assert!(matches!(follow, TrainingError::NonFiniteLoss { .. }));
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
        assert_eq!(
            trainer
                .optimization_samples
                .status_readbacks_step_disposition,
            0,
            "unread steps must not perform StepDisposition status readbacks"
        );
        assert_eq!(trainer.optimization_samples.status_readbacks_checkpoint, 0);

        trainer
            .checkpoint(&splats, trainer_checkpoint_identity(), 3, None)
            .await
            .expect("checkpoint outside loss cadence");
        assert_eq!(trainer.optimization_samples.status_readbacks_checkpoint, 1);
        assert_eq!(
            trainer.optimization_samples.status_readbacks, 2,
            "loss cadence + checkpoint only"
        );
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
        assert_eq!(
            trainer
                .optimization_samples
                .status_readbacks_step_disposition,
            0,
            "healthy unread steps must not use StepDisposition status readbacks"
        );
        assert_eq!(
            trainer.optimization_samples.loss_value_readbacks, expected_loss_cadence,
            "unread steps must not sample the loss scalar"
        );
        assert_eq!(trainer.optimization_samples.status_readbacks_topology, 0);
        assert_eq!(trainer.optimization_samples.status_readbacks_checkpoint, 0);
        assert_eq!(
            trainer.optimization_samples.status_readbacks_training_end,
            1
        );
        assert_eq!(
            trainer.optimization_samples.status_readbacks,
            expected_loss_cadence + 1,
            "status readbacks are safety points only (loss cadence + training end)"
        );
        assert_eq!(expected_loss_cadence, 6);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn scan_workspace_telemetry_tracks_measured_bytes_not_splat_estimates() {
        use crate::training::gpu_primitives::prefix_sum::prefix_sum_total_reserved_bytes;

        let device = GsDevice::default();
        let config = fault_injection_config();
        let host_splats = trainer_checkpoint_host_splats();
        let mut splats = host_splats_to_device::<GsDiffBackend>(&host_splats, &device);
        let mut trainer = WgpuTrainer::new(config, device.clone(), 3, 4, 2.5);
        let camera = fault_injection_camera();

        // No scan observed yet: measured bytes stay unset.
        assert!(trainer.optimization_samples.scan_workspace_bytes.is_none());
        trainer.optimization_samples.record_loop_step(
            Duration::from_millis(1),
            false,
            /*splat_count=*/ 64,
            /*intersection_capacity=*/ 128,
        );
        assert!(
            trainer.optimization_samples.scan_workspace_bytes.is_none(),
            "loop-step splat estimates must not invent scan_workspace_bytes"
        );

        trainer
            .train_step(
                &mut splats,
                &camera,
                fault_injection_target(&device, 0.4),
                (8, 8),
                1,
                1,
                false,
                false,
            )
            .await
            .expect("first train step");

        let reserved = trainer
            .optimization_samples
            .scan_workspace_bytes
            .expect("train step must record measured scan workspace");
        let scratch = trainer
            .optimization_samples
            .scan_workspace_scratch_bytes
            .expect("scratch bytes");
        let output = trainer
            .optimization_samples
            .scan_workspace_output_bytes
            .expect("output bytes");
        assert_eq!(reserved, scratch + output);
        assert_eq!(
            reserved,
            trainer.prefix_sum_workspace.reserved_bytes(),
            "telemetry must mirror PrefixSumWorkspace::reserved_bytes"
        );
        assert_eq!(
            reserved,
            prefix_sum_total_reserved_bytes(trainer.prefix_sum_workspace.capacity())
        );
        // Visible / splat counts can differ from a naive host estimate; measured
        // bytes still come from the owned workspace capacities.
        assert_ne!(
            reserved,
            crate::training::gpu_primitives::prefix_sum::prefix_sum_workspace_bytes(64),
            "must not equal the loop-step splat_count=64 estimate"
        );

        let growth_after_first = trainer.prefix_sum_workspace.growth_count();
        let output_growth_after_first = trainer.prefix_sum_workspace.output_growth_count();
        assert!(growth_after_first >= 1);
        assert!(output_growth_after_first >= 1);

        trainer
            .train_step(
                &mut splats,
                &camera,
                fault_injection_target(&device, 0.4),
                (8, 8),
                2,
                1,
                false,
                false,
            )
            .await
            .expect("steady train step");
        assert_eq!(
            trainer.prefix_sum_workspace.step_fresh_allocations(),
            0,
            "steady reuse must not fresh-allocate"
        );
        assert_eq!(
            trainer.prefix_sum_workspace.growth_count(),
            growth_after_first
        );
        assert_eq!(
            trainer.prefix_sum_workspace.output_growth_count(),
            output_growth_after_first
        );
        assert_eq!(
            trainer.optimization_samples.scan_workspace_bytes,
            Some(trainer.prefix_sum_workspace.reserved_bytes())
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn train_step_hundred_step_prefix_workspace_reaches_steady_state() {
        use crate::training::gpu_primitives::prefix_sum::prefix_sum_total_reserved_bytes;

        let device = GsDevice::default();
        let mut config = fault_injection_config();
        config.iterations = 100;
        let host_splats = trainer_checkpoint_host_splats();
        let mut splats = host_splats_to_device::<GsDiffBackend>(&host_splats, &device);
        let mut trainer = WgpuTrainer::new(config, device.clone(), 3, 4, 2.5);
        let camera = fault_injection_camera();

        let mut growth_series = Vec::new();
        for iteration in 1..=100 {
            trainer
                .train_step(
                    &mut splats,
                    &camera,
                    fault_injection_target(&device, 0.4),
                    (8, 8),
                    iteration,
                    1,
                    false,
                    should_read_loss(iteration, 100, LOSS_SCALAR_READBACK_INTERVAL, false, false),
                )
                .await
                .expect("train step");
            growth_series.push(trainer.prefix_sum_workspace.growth_count());
            if iteration == 1 {
                assert!(
                    trainer.prefix_sum_workspace.step_fresh_allocations() > 0,
                    "first step may allocate scratch/output"
                );
                assert!(trainer.prefix_sum_workspace.growth_count() >= 1);
                assert!(trainer.prefix_sum_workspace.output_growth_count() >= 1);
            } else {
                assert_eq!(
                    trainer.prefix_sum_workspace.step_fresh_allocations(),
                    0,
                    "steady-state train step {iteration} must not fresh-allocate"
                );
                assert_eq!(
                    trainer
                        .prefix_sum_workspace
                        .step_allocations()
                        .scratch_fresh,
                    0
                );
                assert_eq!(
                    trainer.prefix_sum_workspace.step_allocations().output_fresh,
                    0
                );
            }
            assert_eq!(
                trainer.prefix_sum_workspace.reserved_bytes(),
                trainer.prefix_sum_workspace.scratch_bytes()
                    + trainer.prefix_sum_workspace.output_bytes()
            );
        }

        assert!(growth_series.windows(2).all(|w| w[1] >= w[0]));
        assert_eq!(growth_series.first(), growth_series.last());
        assert_eq!(
            trainer.optimization_samples.scan_workspace_bytes,
            Some(trainer.prefix_sum_workspace.reserved_bytes())
        );
        assert_eq!(
            trainer.prefix_sum_workspace.reserved_bytes(),
            prefix_sum_total_reserved_bytes(trainer.prefix_sum_workspace.capacity())
        );

        // Short → long → short on the same workspace the train loop owns.
        {
            use crate::training::gpu_primitives::prefix_sum::PrefixSumBackend;
            use burn::prelude::*;
            use burn::tensor::{Int, TensorData};

            async fn scan_values(
                ws: &mut PrefixSumWorkspace,
                device: &GsDevice,
                values: &[i32],
            ) -> Vec<i32> {
                ws.begin_step();
                let input = Tensor::<GsBackendBase, 1, Int>::from_data(
                    TensorData::new(values.to_vec(), [values.len()]),
                    device,
                );
                let scanned =
                    GsBackendBase::prefix_sum_u32_with_workspace(ws, input.into_primitive())
                        .expect("workspace scan");
                Tensor::<GsBackendBase, 1, Int>::from_primitive(scanned)
                    .into_data_async()
                    .await
                    .expect("read")
                    .into_vec::<i32>()
                    .expect("data")
            }

            async fn fresh_values(device: &GsDevice, values: &[i32]) -> Vec<i32> {
                let input = Tensor::<GsBackendBase, 1, Int>::from_data(
                    TensorData::new(values.to_vec(), [values.len()]),
                    device,
                );
                let scanned =
                    GsBackendBase::prefix_sum_u32_primitive(input.into_primitive()).expect("fresh");
                Tensor::<GsBackendBase, 1, Int>::from_primitive(scanned)
                    .into_data_async()
                    .await
                    .expect("read")
                    .into_vec::<i32>()
                    .expect("data")
            }

            let short = [1_i32, 2, 3, 4];
            let long: Vec<i32> = (0..300).map(|i| i % 3 + 1).collect();
            let short_again = [4_i32, 5, 6];
            for values in [&short[..], long.as_slice(), &short_again[..]] {
                let ws_vals = scan_values(&mut trainer.prefix_sum_workspace, &device, values).await;
                let fresh_vals = fresh_values(&device, values).await;
                assert_eq!(ws_vals, fresh_vals);
            }
            assert!(
                trainer.prefix_sum_workspace.capacity() >= long.len(),
                "long scan must raise owned capacity"
            );
            assert_eq!(
                trainer.prefix_sum_workspace.reserved_bytes(),
                trainer.prefix_sum_workspace.scratch_bytes()
                    + trainer.prefix_sum_workspace.output_bytes()
            );
        }

        // Hold a prior output across the next scan: must not overwrite.
        {
            use crate::training::gpu_primitives::prefix_sum::PrefixSumBackend;
            use burn::prelude::*;
            use burn::tensor::{Int, TensorData};

            trainer.prefix_sum_workspace.begin_step();
            let held_input = Tensor::<GsBackendBase, 1, Int>::from_data(
                TensorData::new(vec![1_i32, 2, 3], [3]),
                &device,
            );
            let held = GsBackendBase::prefix_sum_u32_with_workspace(
                &mut trainer.prefix_sum_workspace,
                held_input.into_primitive(),
            )
            .expect("held scan");

            trainer.prefix_sum_workspace.begin_step();
            let next_input = Tensor::<GsBackendBase, 1, Int>::from_data(
                TensorData::new(vec![9_i32, 8, 7], [3]),
                &device,
            );
            let _ = GsBackendBase::prefix_sum_u32_with_workspace(
                &mut trainer.prefix_sum_workspace,
                next_input.into_primitive(),
            )
            .expect("scan while prior output held");
            assert!(
                trainer.prefix_sum_workspace.step_allocations().output_fresh >= 1,
                "in-use train workspace output must force a fresh buffer"
            );

            let held_vals = Tensor::<GsBackendBase, 1, Int>::from_primitive(held)
                .into_data_async()
                .await
                .expect("held read")
                .into_vec::<i32>()
                .expect("held data");
            assert_eq!(held_vals, vec![1, 3, 6]);
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn gpu_profiler_hundred_step_smoke_sample_counts_and_workspace_monotonic() {
        use crate::training::reporting::gpu_profiler::assert_report_self_consistent;

        let device = GsDevice::default();
        let mut config = fault_injection_config();
        config.iterations = 100;
        config.profiler.gpu_timing_enabled = true;
        config.profiler.gpu_sample_every = 1;
        let host_splats = trainer_checkpoint_host_splats();
        let mut splats = host_splats_to_device::<GsDiffBackend>(&host_splats, &device);
        let mut trainer = WgpuTrainer::new(config, device.clone(), 3, 4, 2.5);
        let camera = fault_injection_camera();

        let mut growth_series = Vec::new();
        let mut fresh_series = Vec::new();
        for iteration in 1..=100 {
            let step_started = Instant::now();
            trainer
                .train_step(
                    &mut splats,
                    &camera,
                    fault_injection_target(&device, 0.4),
                    (8, 8),
                    iteration,
                    1,
                    false,
                    should_read_loss(iteration, 100, LOSS_SCALAR_READBACK_INTERVAL, false, false),
                )
                .await
                .expect("train step");
            trainer
                .pipeline_timing
                .record_cpu_step(step_started.elapsed());
            growth_series.push(
                trainer
                    .pipeline_timing
                    .build_report()
                    .workspace_growth_count,
            );
            fresh_series.push(
                trainer
                    .pipeline_timing
                    .build_report()
                    .fresh_step_allocations,
            );
        }

        let report = trainer.pipeline_timing.build_report();
        assert_report_self_consistent(&report).expect("profiler report self-consistent");
        assert!(
            report.cpu_step_p50_ms.is_some(),
            "CPU step percentiles must be present"
        );
        assert_eq!(
            report.cpu_timing_kind.as_deref(),
            Some("cpu_submit_instant")
        );
        assert!(
            !report.adapter.as_deref().unwrap_or("").is_empty()
                || report.adapter_unavailable_reason.is_some()
        );
        assert!(!report.backend.is_empty());

        if report.supported {
            assert!(report.unsupported_reason.is_none());
            assert!(
                report.sample_count > 0,
                "device timing must collect GPU samples"
            );
            assert!(report.gpu_step_p50_ms.is_some());
            assert!(report.gpu_step_p95_ms.is_some());
            assert!(report.gpu_step_p95_ms.unwrap() >= report.gpu_step_p50_ms.unwrap());
        } else {
            assert!(
                report.unsupported_reason.is_some(),
                "unsupported path must keep a stable reason"
            );
            assert!(report.gpu_step_p50_ms.is_none());
            assert!(report.gpu_step_p95_ms.is_none());
            assert_eq!(report.sample_count, 0);
        }

        assert!(growth_series.windows(2).all(|w| w[1] >= w[0]));
        assert!(fresh_series.windows(2).all(|w| w[1] >= w[0]));
        assert_eq!(
            report.workspace_growth_count,
            *growth_series.last().expect("growth series")
        );
        assert_eq!(
            report.fresh_step_allocations,
            *fresh_series.last().expect("fresh series")
        );

        let forward = report
            .pipeline_spans
            .get(span::FORWARD)
            .expect("forward span");
        assert!(forward.sample_count > 0);
        assert_eq!(forward.timing_kind, "synchronized_boundary");
        assert_eq!(
            report.gpu_timing_scope.as_deref(),
            Some(crate::training::reporting::gpu_profiler::GPU_TIMING_SCOPE_FORWARD)
        );
        assert_eq!(
            report.workspace_scope,
            crate::training::reporting::gpu_profiler::WORKSPACE_SCOPE_PREFIX_SUM
        );
        if report.sample_count > 0 {
            let sum = report.gpu_forward_sum_ms.expect("forward sum");
            let p50 = report.gpu_step_p50_ms.expect("p50");
            assert!(sum.is_finite() && sum >= 0.0);
            assert!(p50.is_finite());
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn c4_production_path_profiler_maps_to_optimization_json() {
        use crate::training::reporting::gpu_profiler::{
            assert_report_self_consistent, optimization_gpu_fields_from_profiler,
            GpuEnvironmentProbe, PipelineTimingCollector, GPU_TIMING_SCOPE_FORWARD,
            WORKSPACE_SCOPE_PREFIX_SUM,
        };

        let device = GsDevice::default();
        let mut config = fault_injection_config();
        config.iterations = 4;
        config.profiler.enabled = true;
        config.profiler.gpu_timing_enabled = true;
        config.profiler.gpu_sample_every = 1;
        config.data.frame_cache_capacity = 2;
        let host_splats = trainer_checkpoint_host_splats();
        let mut splats = host_splats_to_device::<GsDiffBackend>(&host_splats, &device);
        let mut trainer = WgpuTrainer::new(config.clone(), device.clone(), 3, 4, 2.5);
        install_trainer_checkpoint_state(&mut trainer);
        let (_temp, mut loader, cameras, order) = production_outer_loop_fixture(&config);
        let mut observer = OuterLoopProbeObserver::new();

        let report = trainer
            .train_with_frame_loader(
                &mut splats,
                &cameras,
                &order,
                &mut loader,
                (8, 8),
                0,
                4,
                &mut observer,
            )
            .await
            .expect("production outer loop");
        assert_eq!(report.completed_iterations, 4);

        let profiler = report
            .telemetry
            .gpu_profiler
            .as_ref()
            .expect("finish_report must attach gpu_profiler");
        assert_report_self_consistent(profiler).expect("profiler self-consistent");
        assert_eq!(
            profiler.gpu_timing_scope.as_deref(),
            Some(GPU_TIMING_SCOPE_FORWARD)
        );
        assert_eq!(profiler.workspace_scope, WORKSPACE_SCOPE_PREFIX_SUM);
        assert!(profiler.timestamp_query_available || profiler.unsupported_reason.is_some());

        // Uneven accepted samples: sum must not equal p50 × N.
        let mut uneven = PipelineTimingCollector::new(GpuEnvironmentProbe {
            backend: profiler.backend.clone(),
            adapter: profiler.adapter.clone(),
            driver: profiler.driver.clone(),
            timestamp_query_available: true,
            timing_method_device: true,
            ..Default::default()
        });
        for ms in [1.0, 10.0, 100.0, 1000.0] {
            uneven.record_gpu_step_ms(ms);
        }
        let uneven_report = uneven.build_report();
        let sum = uneven_report.gpu_forward_sum_ms.expect("sum");
        let p50 = uneven_report.gpu_step_p50_ms.expect("p50");
        assert!((sum - p50 * uneven_report.sample_count as f64).abs() > 1.0);

        let fields = optimization_gpu_fields_from_profiler(profiler);
        assert!(fields.gpu_completion_seconds.is_none());
        assert_eq!(fields.gpu_timing_scope.as_deref(), Some("forward"));
        assert_eq!(
            fields.timestamp_query_available,
            Some(profiler.timestamp_query_available)
        );
        assert_eq!(fields.workspace_scope.as_deref(), Some("prefix_sum"));
        if let Some(forward_sum_ms) = profiler.gpu_forward_sum_ms {
            assert_eq!(
                fields.gpu_forward_sum_seconds,
                Some(forward_sum_ms / 1000.0)
            );
        }

        // Status readback identity: total equals sum of reason counters.
        let telemetry = &report.telemetry;
        let total = telemetry.status_readbacks.unwrap_or(0);
        let parts = telemetry.status_readbacks_loss_cadence.unwrap_or(0)
            + telemetry.status_readbacks_topology.unwrap_or(0)
            + telemetry.status_readbacks_checkpoint.unwrap_or(0)
            + telemetry.status_readbacks_pause.unwrap_or(0)
            + telemetry.status_readbacks_cancel.unwrap_or(0)
            + telemetry.status_readbacks_training_end.unwrap_or(0)
            + telemetry.status_readbacks_forward_abort.unwrap_or(0)
            + telemetry.status_readbacks_step_disposition.unwrap_or(0);
        assert_eq!(total, parts, "status_readbacks total must equal reason sum");
        // C2 contract: healthy unread steps do not add disposition reads.
        assert_eq!(telemetry.status_readbacks_step_disposition.unwrap_or(0), 0);

        // Prefetch miss records real decode/resize; cache hits must not push 0ms.
        let decode = profiler
            .pipeline_spans
            .get(span::DECODE)
            .expect("decode span");
        let resize = profiler
            .pipeline_spans
            .get(span::RESIZE)
            .expect("resize span");
        assert!(
            decode.sample_count <= 1,
            "cache hits must not push 0ms decode samples, got {}",
            decode.sample_count
        );
        assert!(
            resize.sample_count <= 1,
            "cache hits must not push 0ms resize samples, got {}",
            resize.sample_count
        );
    }
}
