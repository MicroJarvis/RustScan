use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ForwardCapacityTelemetry {
    pub logical_visible: u32,
    pub logical_intersections: u32,
    pub capacity: u32,
    pub overflowed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ParityLossTerms {
    pub l1: Option<f32>,
    pub ssim: Option<f32>,
    pub scale_regularization: Option<f32>,
    pub transmittance: Option<f32>,
    pub depth: Option<f32>,
    pub total: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ParityFloatDistribution {
    pub count: usize,
    pub min: Option<f32>,
    pub p10: Option<f32>,
    pub p50: Option<f32>,
    pub p90: Option<f32>,
    pub p99: Option<f32>,
    pub max: Option<f32>,
    pub mean: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ParityTopologyStepSample {
    pub iteration: usize,
    pub completed_epoch: Option<usize>,
    pub gaussian_count: usize,
    pub clone_candidates: usize,
    pub split_candidates: usize,
    pub prune_candidates: usize,
    pub growth_candidates: usize,
    pub active_grad_stats: usize,
    pub small_scale_stats: usize,
    pub opacity_ready_stats: usize,
    pub large_splat_count: usize,
    pub large_low_grad_count: usize,
    pub large_low_grad_ratio: Option<f32>,
    #[serde(default)]
    pub low_visibility_splats: usize,
    #[serde(default)]
    pub near_low_visibility_splats: usize,
    #[serde(default)]
    pub high_opacity_low_visibility_splats: usize,
    #[serde(default)]
    pub visibility_prune_dry_run_candidates: usize,
    pub mean2d_grad: ParityFloatDistribution,
    #[serde(default)]
    pub screen_mean2d_grad: ParityFloatDistribution,
    #[serde(default)]
    pub abs_mean2d_grad: ParityFloatDistribution,
    #[serde(default)]
    pub abs_pixel_mean2d_grad: ParityFloatDistribution,
    #[serde(default)]
    pub pixel_coverage: ParityFloatDistribution,
    #[serde(default)]
    pub camera_depth: ParityFloatDistribution,
    #[serde(default)]
    pub depth_scale: ParityFloatDistribution,
    #[serde(default)]
    pub split_score: ParityFloatDistribution,
    #[serde(default)]
    pub actual_visible_count: ParityFloatDistribution,
    #[serde(default)]
    pub actual_visibility_ratio: ParityFloatDistribution,
    pub max_scale: ParityFloatDistribution,
    pub opacity: ParityFloatDistribution,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ParityTopologyMetrics {
    pub initialization_gaussians: Option<usize>,
    pub final_gaussians: Option<usize>,
    pub total_epochs: Option<usize>,
    pub densify_until_epoch: Option<usize>,
    pub late_stage_start_epoch: Option<usize>,
    pub topology_freeze_epoch: Option<usize>,
    pub densify_events: usize,
    pub densify_added: usize,
    pub first_densify_epoch: Option<usize>,
    pub last_densify_epoch: Option<usize>,
    pub late_stage_densify_events: usize,
    pub late_stage_densify_added: usize,
    pub prune_events: usize,
    pub prune_removed: usize,
    pub first_prune_epoch: Option<usize>,
    pub last_prune_epoch: Option<usize>,
    pub late_stage_prune_events: usize,
    pub late_stage_prune_removed: usize,
    pub opacity_reset_events: usize,
    pub first_opacity_reset_epoch: Option<usize>,
    pub last_opacity_reset_epoch: Option<usize>,
    pub late_stage_opacity_reset_events: usize,
    #[serde(default)]
    pub topology_step_samples: Vec<ParityTopologyStepSample>,
    pub export_outputs: usize,
    pub checkpoint_roundtrips: usize,
    /// Scheduled topology steps, including steps that retained accumulators.
    #[serde(default)]
    pub scheduled_steps: usize,
    /// Scheduled steps with no eligible candidates and no opacity reset.
    #[serde(default)]
    pub skipped_no_eligible_candidates: usize,
    /// Steps that cleared topology accumulators and started a new window.
    #[serde(default)]
    pub accumulator_resets: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ParityLossCurveSample {
    pub iteration: usize,
    pub frame_idx: usize,
    pub l1: Option<f32>,
    pub ssim: Option<f32>,
    pub depth: Option<f32>,
    pub total: Option<f32>,
    pub depth_valid_pixels: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::ForwardCapacityTelemetry;
    use crate::TrainingError;

    #[test]
    fn overflow_telemetry_maps_to_forward_capacity_exceeded() {
        let telemetry = ForwardCapacityTelemetry {
            logical_visible: 128,
            logical_intersections: 9_000,
            capacity: 8_000,
            overflowed: true,
        };
        assert!(telemetry.overflowed);
        let err = TrainingError::ForwardCapacityExceeded {
            logical_intersections: telemetry.logical_intersections,
            capacity: telemetry.capacity,
        };
        assert!(err.to_string().contains("9000"));
        assert!(err.to_string().contains("8000"));
    }
}
