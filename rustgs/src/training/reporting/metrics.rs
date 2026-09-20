use serde::{Deserialize, Serialize};

use crate::TrainingError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ForwardCapacityTelemetry {
    pub logical_visible: u32,
    pub logical_intersections: u32,
    pub capacity: u32,
    pub overflowed: bool,
}

/// Cross-step sticky overflow record. Sampling cadence must not drop earlier
/// capacity failures, and mutation is blocked while this stays set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StickyForwardOverflow {
    pub overflowed: bool,
    pub first_iteration: u32,
    pub logical_intersections: u32,
    pub capacity: u32,
}

impl StickyForwardOverflow {
    pub fn to_error(self) -> TrainingError {
        TrainingError::ForwardCapacityExceeded {
            logical_intersections: self.logical_intersections,
            capacity: self.capacity,
            first_iteration: self.first_iteration,
        }
    }
}

/// Exact-full (`requested == capacity`) is valid; only a strict excess fails.
pub fn step_intersection_overflowed(requested: u32, capacity: u32) -> bool {
    requested > capacity
}

/// Merge a step overflow into sticky state. The first anomaly wins.
pub fn accumulate_sticky_forward_overflow(
    sticky: StickyForwardOverflow,
    step_overflowed: bool,
    iteration: u32,
    logical_intersections: u32,
    capacity: u32,
) -> StickyForwardOverflow {
    if !step_overflowed {
        return sticky;
    }
    if sticky.overflowed {
        return sticky;
    }
    StickyForwardOverflow {
        overflowed: true,
        first_iteration: iteration.max(1),
        logical_intersections,
        capacity,
    }
}

pub fn allows_state_mutation(sticky: StickyForwardOverflow) -> bool {
    !sticky.overflowed
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
    use super::*;
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
            first_iteration: 1,
        };
        assert!(err.to_string().contains("9000"));
        assert!(err.to_string().contains("8000"));
        assert!(err.to_string().contains("first_iteration=1"));
    }

    #[test]
    fn exact_full_capacity_is_not_overflow() {
        assert!(!step_intersection_overflowed(1_024, 1_024));
        assert!(step_intersection_overflowed(1_025, 1_024));
    }

    #[test]
    fn sticky_keeps_first_anomaly_across_later_healthy_steps() {
        let mut sticky = StickyForwardOverflow::default();
        sticky = accumulate_sticky_forward_overflow(sticky, false, 1, 100, 1_000);
        assert!(allows_state_mutation(sticky));

        sticky = accumulate_sticky_forward_overflow(sticky, true, 2, 9_000, 8_000);
        assert!(!allows_state_mutation(sticky));
        assert_eq!(sticky.first_iteration, 2);
        assert_eq!(sticky.logical_intersections, 9_000);
        assert_eq!(sticky.capacity, 8_000);

        sticky = accumulate_sticky_forward_overflow(sticky, false, 20, 100, 8_000);
        assert!(!allows_state_mutation(sticky));
        assert_eq!(sticky.first_iteration, 2);
        assert_eq!(sticky.logical_intersections, 9_000);

        let err = sticky.to_error();
        assert!(matches!(
            err,
            TrainingError::ForwardCapacityExceeded {
                logical_intersections: 9_000,
                capacity: 8_000,
                first_iteration: 2,
            }
        ));
        let message = err.to_string();
        assert!(message.contains("9000"));
        assert!(message.contains("8000"));
        assert!(message.contains("first_iteration=2"));
    }

    #[test]
    fn consecutive_overflows_preserve_first_iteration() {
        let mut sticky = StickyForwardOverflow::default();
        sticky = accumulate_sticky_forward_overflow(sticky, true, 3, 5_000, 4_000);
        sticky = accumulate_sticky_forward_overflow(sticky, true, 4, 6_000, 4_000);
        assert_eq!(sticky.first_iteration, 3);
        assert_eq!(sticky.logical_intersections, 5_000);
        assert_eq!(sticky.capacity, 4_000);
    }

    #[test]
    fn policy_blocks_mutation_on_overflow_step_before_later_sample() {
        let mut sticky = StickyForwardOverflow::default();
        let mut mutations = 0usize;
        let mut failed_at = None;
        for iteration in 1u32..=20 {
            let overflowed = iteration == 2;
            let requested = if overflowed { 9_000 } else { 100 };
            sticky =
                accumulate_sticky_forward_overflow(sticky, overflowed, iteration, requested, 8_000);
            if !allows_state_mutation(sticky) {
                failed_at = Some(iteration);
                break;
            }
            mutations += 1;
        }
        assert_eq!(failed_at, Some(2));
        assert_eq!(mutations, 1);
        assert_eq!(sticky.first_iteration, 2);
    }

    #[test]
    fn buggy_sample_only_current_frame_policy_misses_nonsample_overflow() {
        let mut saw_error = false;
        let mut mutations = 0usize;
        for iteration in 1u32..=20 {
            let overflowed = iteration == 2;
            let sample = iteration == 20;
            mutations += 1;
            if sample && overflowed {
                saw_error = true;
            }
        }
        assert_eq!(mutations, 20);
        assert!(!saw_error);
    }
}
