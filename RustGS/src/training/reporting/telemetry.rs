use super::metrics::{
    ForwardCapacityTelemetry, ParityLossCurveSample, ParityLossTerms, ParityTopologyMetrics,
};
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct LiteGsOptimizerLrs {
    pub xyz: Option<f32>,
    pub sh_0: Option<f32>,
    pub sh_rest: Option<f32>,
    pub opacity: Option<f32>,
    pub scale: Option<f32>,
    pub rot: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct LiteGsTrainingTelemetry {
    pub loss_terms: ParityLossTerms,
    pub loss_curve_samples: Vec<ParityLossCurveSample>,
    pub topology: ParityTopologyMetrics,
    pub active_sh_degree: Option<usize>,
    pub final_loss: Option<f32>,
    pub final_step_loss: Option<f32>,
    pub depth_valid_pixels: Option<usize>,
    pub depth_grad_scale: Option<f32>,
    pub rotation_frozen: bool,
    pub learning_rates: LiteGsOptimizerLrs,
    pub forward_capacity: Option<ForwardCapacityTelemetry>,
    pub radix_dispatch_count_p50: Option<usize>,
    pub radix_dispatch_count_p95: Option<usize>,
    pub scan_dispatch_count_p50: Option<usize>,
    pub scan_dispatch_count_p95: Option<usize>,
    pub sort_workspace_bytes: Option<usize>,
    pub scan_workspace_bytes: Option<usize>,
    /// CPU submit-side loop duration percentiles (`Instant`); not GPU completion.
    pub loop_duration_p50_ms: Option<f64>,
    pub loop_duration_p95_ms: Option<f64>,
    pub loop_timing_kind: Option<String>,
    pub loss_readback_count: Option<usize>,
    pub count_readback_count: Option<usize>,
    pub status_readbacks: Option<usize>,
    pub status_readbacks_loss_cadence: Option<usize>,
    pub status_readbacks_topology: Option<usize>,
    pub status_readbacks_checkpoint: Option<usize>,
    pub status_readbacks_pause: Option<usize>,
    pub status_readbacks_cancel: Option<usize>,
    pub status_readbacks_training_end: Option<usize>,
    pub capacity_telemetry_readbacks: Option<usize>,
    pub loss_value_readbacks: Option<usize>,
    pub checkpoint_tensor_readbacks: Option<usize>,
    pub topology_snapshot_ms_p50: Option<f64>,
    pub topology_plan_ms_p50: Option<f64>,
    pub topology_apply_ms_p50: Option<f64>,
    pub topology_snapshot_readback_bytes: Option<usize>,
    pub checkpoint_migration: Option<String>,
}

static LAST_TRAINING_TELEMETRY: OnceLock<Mutex<Option<LiteGsTrainingTelemetry>>> = OnceLock::new();

pub(crate) fn store_last_training_telemetry(telemetry: Option<LiteGsTrainingTelemetry>) {
    let slot = LAST_TRAINING_TELEMETRY.get_or_init(|| Mutex::new(None));
    let mut guard = slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    *guard = telemetry;
}

pub fn last_training_telemetry() -> Option<LiteGsTrainingTelemetry> {
    let slot = LAST_TRAINING_TELEMETRY.get_or_init(|| Mutex::new(None));
    slot.lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}
