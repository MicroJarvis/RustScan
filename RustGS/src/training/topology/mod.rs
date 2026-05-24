mod apply;
mod bridge;
mod density_controller;
mod reference;
mod schedule;
mod splat_metrics;

use rand::seq::index::sample_weighted;
use rand::{rngs::StdRng, Rng, SeedableRng};

pub(crate) use self::schedule::should_apply_topology_step;
pub(crate) use apply::apply_mutations;
pub(crate) use bridge::{plan_mutations, snapshot_for_topology};

use self::density_controller::{DensityControllerConfig, PruneMode};
use self::schedule::{plan_topology_execution, schedule_topology, TopologyStepContext};
use self::splat_metrics::TopologySplatMetrics;
use super::reporting::metrics::{ParityFloatDistribution, ParityTopologyMetrics, ParityTopologyStepSample};
use super::{LiteGsConfig, LiteGsPruneMode, LiteGsSplitScoreMode, TrainingConfig};
use crate::core::HostSplats;

const LITEGS_OPACITY_THRESHOLD: f32 = 0.005;
const LITEGS_PERCENT_DENSE: f32 = 0.01;
const BRUSH_MIN_OPACITY: f32 = 1.0 / 255.0;
const LITEGS_OPACITY_RESET_CAP: f32 = 0.01;
const BRUSH_MIN_SCALE: f32 = 1e-10;
const BRUSH_REFINE_PROGRESS_LIMIT: f32 = 0.95;
const TOPOLOGY_SELECTION_SALT: u64 = 0x6c69_7465_6773_7365;
const TOPOLOGY_REFINE_SALT: u64 = 0x6272_7573_685f_7266;

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct RunningMoments {
    pub(super) mean: f32,
    pub(super) m2: f32,
    pub(super) count: usize,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct MetalGaussianStats {
    pub(super) mean2d_grad: RunningMoments,
    pub(super) fragment_weight: RunningMoments,
    pub(super) fragment_err: RunningMoments,
    pub(super) refine_weight_max: f32,
    pub(super) screen_refine_weight_max: f32,
    pub(super) abs_refine_weight_max: f32,
    pub(super) abs_pixel_refine_weight_max: f32,
    pub(super) pixel_coverage_mean: f32,
    pub(super) camera_depth_mean: f32,
    pub(super) visible_count: usize,
    pub(super) actual_visible_count: usize,
    pub(super) actual_visibility_ratio: f32,
    pub(super) age: usize,
    pub(super) consecutive_invisible_epochs: usize,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct TopologyCandidateInfo {
    pub(super) opacity: f32,
    pub(super) mean2d_grad: f32,
    pub(super) screen_mean2d_grad: f32,
    pub(super) abs_mean2d_grad: f32,
    pub(super) abs_pixel_mean2d_grad: f32,
    pub(super) pixel_coverage: f32,
    pub(super) camera_depth: f32,
    pub(super) depth_scale: f32,
    pub(super) split_score: f32,
    pub(super) growth_weight: f32,
    pub(super) visible_count: usize,
    pub(super) actual_visible_count: usize,
    pub(super) actual_visibility_ratio: f32,
    pub(super) age: usize,
    pub(super) consecutive_invisible_epochs: usize,
    pub(super) prune_candidate: bool,
    pub(super) growth_candidate: bool,
    pub(super) split_candidate: bool,
    pub(super) clone_candidate: bool,
}

#[derive(Debug, Default)]
pub(super) struct TopologyAnalysis {
    pub(super) infos: Vec<TopologyCandidateInfo>,
    pub(super) clone_candidates: usize,
    pub(super) split_candidates: usize,
    pub(super) prune_candidates: usize,
    pub(super) growth_candidates: usize,
    pub(super) active_grad_stats: usize,
    pub(super) small_scale_stats: usize,
    pub(super) opacity_ready_stats: usize,
    pub(super) low_visibility_splats: usize,
    pub(super) near_low_visibility_splats: usize,
    pub(super) high_opacity_low_visibility_splats: usize,
    pub(super) visibility_prune_dry_run_candidates: usize,
    pub(super) max_grad: f32,
    pub(super) mean_grad: f32,
}

#[derive(Debug, Default, Clone)]
pub(super) struct LiteGsDensifySelection {
    pub(super) selected_indices: Vec<usize>,
    pub(super) replacement_count: usize,
    pub(super) extra_growth_count: usize,
}

#[derive(Debug, Clone)]
pub(super) struct TopologyPolicy {
    pub(super) litegs: LiteGsConfig,
    pub(super) max_gaussian_budget: usize,
    pub(super) scene_extent: f32,
    pub(super) max_iterations: usize,
}

impl TopologyPolicy {
    pub(crate) fn from_training_config(config: &TrainingConfig, scene_extent: f32) -> Self {
        Self {
            litegs: config.litegs.clone(),
            max_gaussian_budget: config.initialization.max_initial_gaussians.max(1),
            scene_extent,
            max_iterations: config.iterations,
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn litegs_total_epochs(&self, frame_count: usize) -> usize {
        if frame_count == 0 {
            0
        } else {
            (self.max_iterations / frame_count).max(1)
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn litegs_effective_densify_from_epoch(&self, frame_count: usize) -> usize {
        let total_epochs = self.litegs_total_epochs(frame_count);
        if total_epochs == 0 || self.litegs.topology.densify_from >= total_epochs {
            total_epochs
        } else {
            self.litegs.topology.densify_from
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn litegs_densify_until_epoch(&self, frame_count: usize) -> usize {
        let total_epochs = self.litegs_total_epochs(frame_count);
        let densify_from = self.litegs_effective_densify_from_epoch(frame_count);
        if densify_from >= total_epochs {
            return total_epochs;
        }
        if let Some(until) = self.litegs.topology.densify_until {
            return until.max(densify_from.saturating_add(1)).min(total_epochs);
        }

        let reset_interval = self.litegs.topology.opacity_reset_interval.max(1);
        let scaled = ((total_epochs as f32) * 0.8).floor() as usize;
        let computed = (scaled / reset_interval) * reset_interval + 1;
        computed
            .max(densify_from.saturating_add(1))
            .min(total_epochs)
    }

    pub(super) fn litegs_clone_scale_threshold(&self) -> f32 {
        (self.scene_extent * LITEGS_PERCENT_DENSE).max(1e-4)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn density_controller_reference_config(
        &self,
        current_gaussians: usize,
    ) -> DensityControllerConfig {
        DensityControllerConfig {
            densify_grad_threshold: self.litegs.growth.growth_grad_threshold,
            opacity_threshold: LITEGS_OPACITY_THRESHOLD,
            percent_dense: LITEGS_PERCENT_DENSE,
            screen_extent: self.scene_extent.max(1e-6),
            init_points_num: current_gaussians.max(1),
            target_primitives: self.litegs.topology.target_primitives.max(current_gaussians.max(1)),
            densify_from: self.litegs.topology.densify_from,
            densify_until: self.litegs.topology.densify_until.unwrap_or(self.max_iterations),
            densification_interval: self.litegs.topology.densification_interval.max(1),
            prune_mode: density_controller_prune_mode(self.litegs.pruning.prune_mode),
        }
    }
}

pub(crate) fn plan_topology_from_host_snapshot(
    config: &TrainingConfig,
    splats: &HostSplats,
    grad_2d_accum: &[f32],
    screen_grad_2d_accum: &[f32],
    abs_grad_2d_accum: &[f32],
    abs_pixel_grad_2d_accum: &[f32],
    pixel_coverage_accum: &[f32],
    camera_depth_accum: &[f32],
    grad_color_accum: &[f32],
    num_observations: &[f32],
    visible_observations: &[f32],
    actual_visible_observations: &[f32],
    splat_ages: &[usize],
    invisible_windows: &[usize],
    iteration: usize,
    frame_count: usize,
) -> TopologyMutationPlan {
    let metrics = TopologySplatMetrics::from_snapshot(splats);
    let policy = TopologyPolicy::from_training_config(config, splats.scene_extent());
    let schedule = schedule_topology(
        &policy,
        TopologyStepContext {
            iteration,
            frame_count,
        },
    );

    let stats = build_host_snapshot_stats(
        metrics.len(),
        grad_2d_accum,
        screen_grad_2d_accum,
        abs_grad_2d_accum,
        abs_pixel_grad_2d_accum,
        pixel_coverage_accum,
        camera_depth_accum,
        grad_color_accum,
        num_observations,
        visible_observations,
        actual_visible_observations,
        splat_ages,
        invisible_windows,
    );
    let analysis = analyze_topology_candidates(&policy, &metrics, &stats, schedule.densify);

    let requested_additions = litegs_requested_additions(
        &analysis.infos,
        policy.litegs.growth.growth_select_fraction,
        schedule.allow_extra_growth,
    );
    let max_gaussians = requested_gaussian_cap(&policy, metrics.len(), requested_additions);
    let max_new = max_gaussians.saturating_sub(metrics.len());
    let topology_seed = topology_rng_seed(config.data.frame_shuffle_seed, iteration);
    let litegs_selection = litegs_select_densify_candidates_seeded(
        &analysis.infos,
        max_new,
        policy.litegs.growth.growth_select_fraction,
        schedule.allow_extra_growth,
        topology_seed,
    );

    let execution = plan_topology_execution(&policy, schedule, &analysis, &litegs_selection);
    let request = TopologyMutationRequest {
        should_densify: execution.should_densify,
        should_reset_opacity: execution.should_reset_opacity,
        refine_decay: refine_decay_for_schedule(
            config,
            schedule.densify || schedule.prune,
            iteration,
        ),
        completed_epoch: execution.completed_epoch,
        late_stage: iteration >= config.iterations.saturating_mul(9) / 10,
        infos: &analysis.infos,
        litegs_selection: &litegs_selection,
        random_seed: topology_seed,
    };

    let mut plan = plan_topology_mutation(&metrics, request);
    plan.telemetry_sample = Some(topology_step_sample(
        &policy,
        &metrics,
        &analysis,
        iteration,
        schedule.completed_epoch,
    ));
    plan
}

pub(super) fn analyze_topology_candidates(
    policy: &TopologyPolicy,
    splats: &TopologySplatMetrics,
    stats: &[MetalGaussianStats],
    opacity_prune_enabled: bool,
) -> TopologyAnalysis {
    let mut analysis = TopologyAnalysis {
        infos: Vec::with_capacity(splats.len()),
        ..TopologyAnalysis::default()
    };
    let mut grad_sum = 0.0f32;
    let clone_scale_threshold = policy.litegs_clone_scale_threshold();
    let (brush_center, brush_extent) = splats.brush_bounds_center_extent();
    let brush_max_allowed_bounds = brush_extent.max(policy.scene_extent.max(1e-6)) * 100.0;

    for idx in 0..splats.len() {
        let position = splats.position(idx);
        let scale = splats.scale(idx);
        let max_scale = splats.max_scale(idx);
        let opacity = splats.opacity(idx);
        let gaussian_stats = stats.get(idx).copied().unwrap_or_default();
        let mean2d_grad = gaussian_stats.refine_weight_max;
        let screen_mean2d_grad = gaussian_stats.screen_refine_weight_max;
        let abs_mean2d_grad = gaussian_stats.abs_refine_weight_max;
        let abs_pixel_mean2d_grad = gaussian_stats.abs_pixel_refine_weight_max;
        let pixel_coverage = gaussian_stats.pixel_coverage_mean;
        let camera_depth = gaussian_stats.camera_depth_mean;
        let actual_visibility_ratio = gaussian_stats.actual_visibility_ratio;
        let depth_scale = depth_scale_factor(
            camera_depth,
            policy.litegs.growth.depth_scale_gamma,
            policy.scene_extent,
        );
        let split_score = match policy.litegs.growth.split_score_mode {
            LiteGsSplitScoreMode::Baseline => mean2d_grad,
            LiteGsSplitScoreMode::Abs => abs_mean2d_grad,
            LiteGsSplitScoreMode::AbsPixel => abs_pixel_mean2d_grad,
            LiteGsSplitScoreMode::AbsPixelDepth => abs_pixel_mean2d_grad * depth_scale,
        };
        let growth_threshold = policy.litegs.growth.growth_grad_threshold;
        let split_threshold = match policy.litegs.growth.split_score_mode {
            LiteGsSplitScoreMode::Baseline => growth_threshold,
            LiteGsSplitScoreMode::Abs
            | LiteGsSplitScoreMode::AbsPixel
            | LiteGsSplitScoreMode::AbsPixelDepth => policy.litegs.growth.split_grad_threshold,
        };
        let clone_candidate = max_scale <= clone_scale_threshold
            && mean2d_grad.is_finite()
            && mean2d_grad >= growth_threshold;
        let split_opacity_ready = opacity.is_finite() && opacity > BRUSH_MIN_OPACITY;
        let baseline_split_candidate = max_scale > clone_scale_threshold
            && mean2d_grad.is_finite()
            && mean2d_grad >= growth_threshold;
        let mode_uses_augmented_baseline = matches!(
            policy.litegs.growth.split_score_mode,
            LiteGsSplitScoreMode::Abs | LiteGsSplitScoreMode::AbsPixel
        );
        let mode_uses_depth_scaled_score =
            policy.litegs.growth.split_score_mode == LiteGsSplitScoreMode::AbsPixelDepth;
        let score_split_candidate = policy.litegs.growth.split_score_mode
            != LiteGsSplitScoreMode::Baseline
            && split_score.is_finite()
            && split_score >= split_threshold
            && split_opacity_ready;
        let split_candidate = max_scale > clone_scale_threshold
            && if mode_uses_depth_scaled_score {
                score_split_candidate
            } else if mode_uses_augmented_baseline {
                baseline_split_candidate || score_split_candidate
            } else {
                baseline_split_candidate
            };
        let growth_weight = if clone_candidate {
            mean2d_grad.max(0.0)
        } else if split_candidate {
            match policy.litegs.growth.split_score_mode {
                LiteGsSplitScoreMode::Baseline => mean2d_grad.max(0.0),
                LiteGsSplitScoreMode::Abs | LiteGsSplitScoreMode::AbsPixel => {
                    split_score.max(mean2d_grad).max(0.0)
                }
                LiteGsSplitScoreMode::AbsPixelDepth => split_score.max(0.0),
            }
        } else {
            0.0
        };
        let growth_candidate = clone_candidate || split_candidate;
        let candidate_info = TopologyCandidateInfo {
            opacity,
            mean2d_grad,
            screen_mean2d_grad,
            abs_mean2d_grad,
            abs_pixel_mean2d_grad,
            pixel_coverage,
            camera_depth,
            depth_scale,
            split_score,
            growth_weight,
            visible_count: gaussian_stats.visible_count,
            actual_visible_count: gaussian_stats.actual_visible_count,
            actual_visibility_ratio,
            age: gaussian_stats.age,
            consecutive_invisible_epochs: gaussian_stats.consecutive_invisible_epochs,
            prune_candidate: false,
            growth_candidate,
            split_candidate,
            clone_candidate,
        };
        let prune_candidate = litegs_should_prune_candidate(
            policy,
            &candidate_info,
            position,
            scale,
            brush_center,
            brush_max_allowed_bounds,
            splats.retainable(idx),
            opacity_prune_enabled,
        );

        analysis.infos.push(TopologyCandidateInfo {
            prune_candidate,
            ..candidate_info
        });
        if mean2d_grad > growth_threshold {
            analysis.active_grad_stats += 1;
        }
        if max_scale <= clone_scale_threshold {
            analysis.small_scale_stats += 1;
        }
        if opacity > LITEGS_OPACITY_THRESHOLD {
            analysis.opacity_ready_stats += 1;
        }
        if is_low_visibility(policy, &candidate_info) {
            analysis.low_visibility_splats += 1;
            if is_near_camera_low_visibility(policy, &candidate_info) {
                analysis.near_low_visibility_splats += 1;
            }
            if candidate_info.opacity >= policy.litegs.pruning.prune_high_opacity_threshold {
                analysis.high_opacity_low_visibility_splats += 1;
            }
            if policy.litegs.pruning.prune_visibility_dry_run
                && candidate_info.opacity < policy.litegs.pruning.prune_high_opacity_threshold
            {
                analysis.visibility_prune_dry_run_candidates += 1;
            }
        }
        if mean2d_grad.is_finite() {
            analysis.max_grad = analysis.max_grad.max(mean2d_grad);
            grad_sum += mean2d_grad;
        }

        if growth_candidate {
            analysis.growth_candidates += 1;
            if candidate_info.clone_candidate {
                analysis.clone_candidates += 1;
            }
            if candidate_info.split_candidate {
                analysis.split_candidates += 1;
            }
        }
        if prune_candidate {
            analysis.prune_candidates += 1;
        }
    }

    if !analysis.infos.is_empty() {
        analysis.mean_grad = grad_sum / analysis.infos.len() as f32;
    }
    analysis
}

fn is_low_visibility(policy: &TopologyPolicy, info: &TopologyCandidateInfo) -> bool {
    collects_actual_visibility_diagnostics(policy)
        && info.age >= policy.litegs.pruning.prune_min_age
        && info.actual_visibility_ratio.is_finite()
        && info.actual_visibility_ratio < policy.litegs.pruning.prune_visibility_threshold
}

fn collects_actual_visibility_diagnostics(policy: &TopologyPolicy) -> bool {
    policy.litegs.pruning.prune_visibility_dry_run
        || matches!(
            policy.litegs.pruning.prune_mode,
            LiteGsPruneMode::Threshold | LiteGsPruneMode::VisibilityWeight
        )
}

fn is_near_camera_low_visibility(policy: &TopologyPolicy, info: &TopologyCandidateInfo) -> bool {
    let near_depth = policy.scene_extent.max(1e-6) * policy.litegs.growth.depth_scale_gamma;
    is_low_visibility(policy, info)
        && info.camera_depth.is_finite()
        && info.camera_depth > 0.0
        && info.camera_depth < near_depth
}

fn topology_step_sample(
    policy: &TopologyPolicy,
    splats: &TopologySplatMetrics,
    analysis: &TopologyAnalysis,
    iteration: usize,
    completed_epoch: Option<usize>,
) -> ParityTopologyStepSample {
    let clone_scale_threshold = policy.litegs_clone_scale_threshold();
    let growth_threshold = policy.litegs.growth.growth_grad_threshold;
    let mut max_scales = Vec::with_capacity(splats.len());
    let mut opacities = Vec::with_capacity(splats.len());
    let mut large_splat_count = 0usize;
    let mut large_low_grad_count = 0usize;

    for idx in 0..splats.len() {
        let max_scale = splats.max_scale(idx);
        let opacity = splats.opacity(idx);
        max_scales.push(max_scale);
        opacities.push(opacity);

        if max_scale > clone_scale_threshold {
            large_splat_count = large_splat_count.saturating_add(1);
            let mean2d_grad = analysis
                .infos
                .get(idx)
                .map(|info| info.mean2d_grad)
                .unwrap_or_default();
            if mean2d_grad.is_finite() && mean2d_grad < growth_threshold {
                large_low_grad_count = large_low_grad_count.saturating_add(1);
            }
        }
    }

    let large_low_grad_ratio =
        (large_splat_count > 0).then_some(large_low_grad_count as f32 / large_splat_count as f32);
    ParityTopologyStepSample {
        iteration,
        completed_epoch,
        gaussian_count: splats.len(),
        clone_candidates: analysis.clone_candidates,
        split_candidates: analysis.split_candidates,
        prune_candidates: analysis.prune_candidates,
        growth_candidates: analysis.growth_candidates,
        active_grad_stats: analysis.active_grad_stats,
        small_scale_stats: analysis.small_scale_stats,
        opacity_ready_stats: analysis.opacity_ready_stats,
        large_splat_count,
        large_low_grad_count,
        large_low_grad_ratio,
        low_visibility_splats: analysis.low_visibility_splats,
        near_low_visibility_splats: analysis.near_low_visibility_splats,
        high_opacity_low_visibility_splats: analysis.high_opacity_low_visibility_splats,
        visibility_prune_dry_run_candidates: analysis.visibility_prune_dry_run_candidates,
        mean2d_grad: finite_distribution(analysis.infos.iter().map(|info| info.mean2d_grad)),
        screen_mean2d_grad: finite_distribution(
            analysis.infos.iter().map(|info| info.screen_mean2d_grad),
        ),
        abs_mean2d_grad: finite_distribution(
            analysis.infos.iter().map(|info| info.abs_mean2d_grad),
        ),
        abs_pixel_mean2d_grad: finite_distribution(
            analysis.infos.iter().map(|info| info.abs_pixel_mean2d_grad),
        ),
        pixel_coverage: finite_distribution(analysis.infos.iter().map(|info| info.pixel_coverage)),
        camera_depth: finite_distribution(analysis.infos.iter().map(|info| info.camera_depth)),
        depth_scale: finite_distribution(analysis.infos.iter().map(|info| info.depth_scale)),
        split_score: finite_distribution(analysis.infos.iter().map(|info| info.split_score)),
        actual_visible_count: finite_distribution(
            analysis
                .infos
                .iter()
                .map(|info| info.actual_visible_count as f32),
        ),
        actual_visibility_ratio: finite_distribution(
            analysis
                .infos
                .iter()
                .map(|info| info.actual_visibility_ratio),
        ),
        max_scale: finite_distribution(max_scales),
        opacity: finite_distribution(opacities),
    }
}

fn depth_scale_factor(camera_depth: f32, gamma: f32, scene_radius: f32) -> f32 {
    if !camera_depth.is_finite() || camera_depth <= 0.0 {
        return 0.0;
    }
    let denom = (gamma * scene_radius.max(1e-6)).max(1e-6);
    let ratio = camera_depth / denom;
    (ratio * ratio).clamp(0.0, 1.0)
}

fn finite_distribution(values: impl IntoIterator<Item = f32>) -> ParityFloatDistribution {
    let mut values: Vec<f32> = values
        .into_iter()
        .filter(|value| value.is_finite())
        .collect();
    if values.is_empty() {
        return ParityFloatDistribution::default();
    }

    values.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let count = values.len();
    let mean = values.iter().copied().sum::<f32>() / count as f32;
    ParityFloatDistribution {
        count,
        min: values.first().copied(),
        p10: percentile(&values, 0.10),
        p50: percentile(&values, 0.50),
        p90: percentile(&values, 0.90),
        p99: percentile(&values, 0.99),
        max: values.last().copied(),
        mean: Some(mean),
    }
}

fn percentile(sorted_values: &[f32], quantile: f32) -> Option<f32> {
    if sorted_values.is_empty() {
        return None;
    }
    let last = sorted_values.len().saturating_sub(1);
    let idx = ((last as f32) * quantile.clamp(0.0, 1.0)).round() as usize;
    sorted_values.get(idx.min(last)).copied()
}

pub(super) fn litegs_requested_additions(
    infos: &[TopologyCandidateInfo],
    growth_select_fraction: f32,
    allow_extra_growth: bool,
) -> usize {
    let prune_candidates = infos.iter().filter(|info| info.prune_candidate).count();
    if !allow_extra_growth {
        return prune_candidates;
    }

    let threshold_count = infos.iter().filter(|info| info.growth_candidate).count();
    let grow_count = (threshold_count as f32 * growth_select_fraction).round() as usize;

    prune_candidates.saturating_add(grow_count.saturating_sub(prune_candidates))
}

#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn litegs_select_densify_candidates(
    infos: &[TopologyCandidateInfo],
    max_new: usize,
    growth_select_fraction: f32,
    allow_extra_growth: bool,
) -> LiteGsDensifySelection {
    litegs_select_densify_candidates_seeded(
        infos,
        max_new,
        growth_select_fraction,
        allow_extra_growth,
        0,
    )
}

fn litegs_select_densify_candidates_seeded(
    infos: &[TopologyCandidateInfo],
    max_new: usize,
    growth_select_fraction: f32,
    allow_extra_growth: bool,
    seed: u64,
) -> LiteGsDensifySelection {
    let mut rng = StdRng::seed_from_u64(seed ^ TOPOLOGY_SELECTION_SALT);
    litegs_select_densify_candidates_with_rng(
        infos,
        max_new,
        growth_select_fraction,
        allow_extra_growth,
        &mut rng,
    )
}

fn litegs_select_densify_candidates_with_rng<R: Rng + ?Sized>(
    infos: &[TopologyCandidateInfo],
    max_new: usize,
    growth_select_fraction: f32,
    allow_extra_growth: bool,
    rng: &mut R,
) -> LiteGsDensifySelection {
    if max_new == 0 || infos.is_empty() {
        return LiteGsDensifySelection::default();
    }

    let prune_candidates = infos.iter().filter(|info| info.prune_candidate).count();
    let replacement_sources: Vec<usize> = infos
        .iter()
        .enumerate()
        .filter_map(|(idx, info)| {
            (!info.prune_candidate && info.opacity.is_finite() && info.opacity > 0.0).then_some(idx)
        })
        .collect();
    let replacement_weights: Vec<f32> = replacement_sources
        .iter()
        .map(|&idx| infos[idx].opacity.max(0.0))
        .collect();

    let replacement_count = prune_candidates.min(max_new).min(replacement_sources.len());
    let mut selection = LiteGsDensifySelection {
        selected_indices: Vec::with_capacity(max_new.min(infos.len())),
        replacement_count: 0,
        extra_growth_count: 0,
    };
    let mut used_sources = vec![false; infos.len()];

    if replacement_count > 0 {
        for sampled in sample_weighted_indices(&replacement_weights, replacement_count, rng) {
            let source_idx = replacement_sources[sampled];
            selection.selected_indices.push(source_idx);
            selection.replacement_count += 1;
            used_sources[source_idx] = true;
        }
    }

    if allow_extra_growth && selection.selected_indices.len() < max_new {
        let threshold_count = infos.iter().filter(|info| info.growth_candidate).count();
        let grow_count = (threshold_count as f32 * growth_select_fraction).round() as usize;
        let sample_high_grad = grow_count.saturating_sub(prune_candidates);
        let extra_growth_limit =
            sample_high_grad.min(max_new.saturating_sub(selection.selected_indices.len()));

        if extra_growth_limit > 0 {
            let growth_sources: Vec<usize> = infos
                .iter()
                .enumerate()
                .filter_map(|(idx, info)| {
                    (!info.prune_candidate && info.growth_candidate && !used_sources[idx])
                        .then_some(idx)
                })
                .collect();
            let growth_weights: Vec<f32> = growth_sources
                .iter()
                .map(|&idx| infos[idx].growth_weight.max(0.0))
                .collect();
            for sampled in sample_weighted_indices(
                &growth_weights,
                extra_growth_limit.min(growth_sources.len()),
                rng,
            ) {
                let source_idx = growth_sources[sampled];
                selection.selected_indices.push(source_idx);
                selection.extra_growth_count += 1;
            }
        }
    }

    selection
}

pub(super) fn requested_gaussian_cap(
    policy: &TopologyPolicy,
    current_len: usize,
    litegs_requested_additions: usize,
) -> usize {
    policy
        .max_gaussian_budget
        .max(current_len.saturating_add(litegs_requested_additions))
}

fn refine_decay_for_schedule(
    config: &TrainingConfig,
    should_refine: bool,
    iteration: usize,
) -> Option<TopologyRefineDecay> {
    if !should_refine || (config.litegs.refine.opacity_decay == 0.0 && config.litegs.refine.scale_decay == 0.0) {
        return None;
    }

    let train_t = if config.iterations == 0 {
        1.0
    } else {
        iteration as f32 / config.iterations as f32
    };
    Some(TopologyRefineDecay {
        opacity_decay: config.litegs.refine.opacity_decay,
        scale_decay: config.litegs.refine.scale_decay,
        train_t: train_t.clamp(0.0, 1.0),
    })
}



#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TopologyStatsAction {
    KeepCurrent,
    UseMutated,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct TopologyRefineDecay {
    pub(crate) opacity_decay: f32,
    pub(crate) scale_decay: f32,
    pub(crate) train_t: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct TopologyMetricsDelta {
    pub(super) final_gaussians: usize,
    pub(super) added: usize,
    pub(super) pruned: usize,
    pub(super) opacity_reset: bool,
    pub(super) completed_epoch: Option<usize>,
    pub(super) late_stage: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TopologyMutationAftermath {
    pub(super) requires_runtime_rebuild: bool,
    pub(super) requires_adam_rebuild: bool,
    pub(super) requires_cluster_resync: bool,
    pub(super) requires_runtime_reserve: bool,
    pub(super) apply_opacity_reset: bool,
    pub(super) reset_refine_window_stats: bool,
    pub(super) gaussian_stats_action: TopologyStatsAction,
    pub(super) metrics_delta: TopologyMetricsDelta,
}

impl Default for TopologyMutationAftermath {
    fn default() -> Self {
        Self {
            requires_runtime_rebuild: false,
            requires_adam_rebuild: false,
            requires_cluster_resync: false,
            requires_runtime_reserve: false,
            apply_opacity_reset: false,
            reset_refine_window_stats: false,
            gaussian_stats_action: TopologyStatsAction::KeepCurrent,
            metrics_delta: TopologyMetricsDelta::default(),
        }
    }
}

pub(super) struct TopologyMutationRequest<'a> {
    pub(super) should_densify: bool,
    pub(super) should_reset_opacity: bool,
    pub(super) refine_decay: Option<TopologyRefineDecay>,
    pub(super) completed_epoch: Option<usize>,
    pub(super) late_stage: bool,
    pub(super) infos: &'a [TopologyCandidateInfo],
    pub(super) litegs_selection: &'a LiteGsDensifySelection,
    pub(super) random_seed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum TopologyPlanRow {
    Existing {
        source_idx: usize,
    },
    BrushRefineExisting {
        source_idx: usize,
        sample_scalar: f32,
    },
    BrushRefineNew {
        source_idx: usize,
        sample_scalar: f32,
    },
}

impl TopologyPlanRow {
    pub(crate) fn source_idx(&self) -> usize {
        match *self {
            Self::Existing { source_idx }
            | Self::BrushRefineExisting { source_idx, .. }
            | Self::BrushRefineNew { source_idx, .. } => source_idx,
        }
    }

    pub(crate) fn is_existing(&self) -> bool {
        matches!(
            self,
            Self::Existing { .. } | Self::BrushRefineExisting { .. }
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct TopologyMutationPlan {
    pub(crate) rows: Vec<TopologyPlanRow>,
    pub(crate) added: usize,
    pub(crate) pruned: usize,
    pub(crate) refine_decay: Option<TopologyRefineDecay>,
    pub(crate) telemetry_sample: Option<ParityTopologyStepSample>,
    pub(super) aftermath: TopologyMutationAftermath,
}

impl TopologyMutationPlan {
    pub(crate) fn mutates_splats(&self) -> bool {
        self.added > 0
            || self.pruned > 0
            || self.refine_decay.is_some()
            || self.aftermath.apply_opacity_reset
    }

    pub(crate) fn origins(&self) -> Vec<Option<usize>> {
        self.rows
            .iter()
            .map(|row| row.is_existing().then_some(row.source_idx()))
            .collect()
    }
}

pub(crate) fn plan_topology_mutation(
    metrics: &TopologySplatMetrics,
    request: TopologyMutationRequest<'_>,
) -> TopologyMutationPlan {
    plan_brush_refine_mutation(metrics, request)
}

fn plan_brush_refine_mutation(
    metrics: &TopologySplatMetrics,
    request: TopologyMutationRequest<'_>,
) -> TopologyMutationPlan {
    let mut surviving_indices: Vec<usize> = (0..metrics.len())
        .filter(|&idx| {
            !request
                .infos
                .get(idx)
                .map(|info| info.prune_candidate)
                .unwrap_or(false)
        })
        .collect();
    if surviving_indices.is_empty() {
        surviving_indices = (0..metrics.len()).collect();
    }

    let mut rows: Vec<TopologyPlanRow> = surviving_indices
        .iter()
        .copied()
        .map(|source_idx| TopologyPlanRow::Existing { source_idx })
        .collect();
    let pruned = metrics.len().saturating_sub(surviving_indices.len());

    let mut added = 0usize;
    if request.should_densify && !request.litegs_selection.selected_indices.is_empty() {
        let mut rng = StdRng::seed_from_u64(request.random_seed ^ TOPOLOGY_REFINE_SALT);
        for &source_idx in &request.litegs_selection.selected_indices {
            let Some(row_idx) = surviving_indices.iter().position(|&idx| idx == source_idx) else {
                continue;
            };
            let sample_scalar = brush_refine_sample_scalar(&mut rng);
            rows[row_idx] = TopologyPlanRow::BrushRefineExisting {
                source_idx,
                sample_scalar,
            };
            rows.push(TopologyPlanRow::BrushRefineNew {
                source_idx,
                sample_scalar,
            });
            added = added.saturating_add(1);
        }
    }

    let aftermath = topology_mutation_aftermath(&request, rows.len(), added, pruned);
    TopologyMutationPlan {
        rows,
        added,
        pruned,
        refine_decay: request.refine_decay,
        telemetry_sample: None,
        aftermath,
    }
}


fn topology_mutation_aftermath(
    request: &TopologyMutationRequest<'_>,
    final_gaussians: usize,
    added: usize,
    pruned: usize,
) -> TopologyMutationAftermath {
    let topology_changed = added > 0 || pruned > 0;
    let splat_values_changed =
        topology_changed || request.refine_decay.is_some() || request.should_reset_opacity;
    let gaussian_stats_action = if topology_changed {
        TopologyStatsAction::UseMutated
    } else {
        TopologyStatsAction::KeepCurrent
    };

    TopologyMutationAftermath {
        requires_runtime_rebuild: topology_changed,
        requires_adam_rebuild: splat_values_changed,
        requires_cluster_resync: topology_changed,
        requires_runtime_reserve: topology_changed,
        apply_opacity_reset: request.should_reset_opacity,
        reset_refine_window_stats: true,
        gaussian_stats_action,
        metrics_delta: TopologyMetricsDelta {
            final_gaussians,
            added,
            pruned,
            opacity_reset: request.should_reset_opacity,
            completed_epoch: request.completed_epoch,
            late_stage: request.late_stage,
        },
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn apply_topology_metrics_delta(
    metrics: &mut ParityTopologyMetrics,
    delta: TopologyMetricsDelta,
) {
    metrics.final_gaussians = Some(delta.final_gaussians);
    if delta.added > 0 {
        metrics.densify_events = metrics.densify_events.saturating_add(1);
        metrics.densify_added = metrics.densify_added.saturating_add(delta.added);
        record_topology_epoch(
            &mut metrics.first_densify_epoch,
            &mut metrics.last_densify_epoch,
            delta.completed_epoch,
        );
        if delta.late_stage {
            metrics.late_stage_densify_events = metrics.late_stage_densify_events.saturating_add(1);
            metrics.late_stage_densify_added =
                metrics.late_stage_densify_added.saturating_add(delta.added);
        }
    }
    if delta.pruned > 0 {
        metrics.prune_events = metrics.prune_events.saturating_add(1);
        metrics.prune_removed = metrics.prune_removed.saturating_add(delta.pruned);
        record_topology_epoch(
            &mut metrics.first_prune_epoch,
            &mut metrics.last_prune_epoch,
            delta.completed_epoch,
        );
        if delta.late_stage {
            metrics.late_stage_prune_events = metrics.late_stage_prune_events.saturating_add(1);
            metrics.late_stage_prune_removed = metrics
                .late_stage_prune_removed
                .saturating_add(delta.pruned);
        }
    }
    if delta.opacity_reset {
        metrics.opacity_reset_events = metrics.opacity_reset_events.saturating_add(1);
        record_topology_epoch(
            &mut metrics.first_opacity_reset_epoch,
            &mut metrics.last_opacity_reset_epoch,
            delta.completed_epoch,
        );
        if delta.late_stage {
            metrics.late_stage_opacity_reset_events =
                metrics.late_stage_opacity_reset_events.saturating_add(1);
        }
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn density_controller_prune_mode(mode: LiteGsPruneMode) -> PruneMode {
    match mode {
        LiteGsPruneMode::Threshold => PruneMode::Threshold,
        LiteGsPruneMode::Weight | LiteGsPruneMode::VisibilityWeight => PruneMode::Weight,
    }
}

fn sample_weighted_indices<R: Rng + ?Sized>(
    weights: &[f32],
    count: usize,
    rng: &mut R,
) -> Vec<usize> {
    if count == 0 || weights.is_empty() {
        return Vec::new();
    }

    let sanitized: Vec<f32> = weights
        .iter()
        .map(|weight| {
            if weight.is_finite() && *weight > 0.0 {
                *weight
            } else {
                0.0
            }
        })
        .collect();
    sample_weighted(
        rng,
        sanitized.len(),
        |idx| sanitized[idx],
        count.min(sanitized.len()),
    )
    .map(|sample| sample.into_iter().collect())
    .unwrap_or_default()
}

fn brush_refine_sample_scalar<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    let u1 = rng.gen::<f32>().clamp(1e-6, 1.0 - 1e-6);
    let u2 = rng.gen::<f32>();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

fn topology_rng_seed(base_seed: u64, iteration: usize) -> u64 {
    base_seed
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(iteration as u64)
        .rotate_left(17)
}




#[cfg_attr(not(test), allow(dead_code))]
fn record_topology_epoch(
    first: &mut Option<usize>,
    last: &mut Option<usize>,
    epoch: Option<usize>,
) {
    let Some(epoch) = epoch else {
        return;
    };
    *first = Some(first.map_or(epoch, |current| current.min(epoch)));
    *last = Some(last.map_or(epoch, |current| current.max(epoch)));
}

fn litegs_should_prune_candidate(
    policy: &TopologyPolicy,
    info: &TopologyCandidateInfo,
    position: [f32; 3],
    scale: [f32; 3],
    center: [f32; 3],
    max_allowed_bounds: f32,
    retainable: bool,
    opacity_prune_enabled: bool,
) -> bool {
    let opacity_threshold = policy.litegs.pruning.prune_opacity_threshold.max(BRUSH_MIN_OPACITY);
    let old_enough = info.age >= policy.litegs.pruning.prune_min_age;
    let opacity_prune =
        old_enough && (!info.opacity.is_finite() || info.opacity < opacity_threshold);
    let history_invisible_prune = old_enough
        && info.visible_count == 0
        && info.consecutive_invisible_epochs >= policy.litegs.pruning.prune_invisible_epochs;
    let contribution_prune = match policy.litegs.pruning.prune_mode {
        LiteGsPruneMode::Threshold => opacity_prune || history_invisible_prune,
        LiteGsPruneMode::Weight => {
            (opacity_prune_enabled && opacity_prune)
                || (!opacity_prune_enabled && history_invisible_prune)
        }
        LiteGsPruneMode::VisibilityWeight => {
            let visibility_prune = old_enough
                && info.actual_visibility_ratio.is_finite()
                && info.actual_visibility_ratio < policy.litegs.pruning.prune_visibility_threshold
                && info.opacity < policy.litegs.pruning.prune_high_opacity_threshold;
            (opacity_prune_enabled && opacity_prune)
                || (!opacity_prune_enabled && history_invisible_prune)
                || visibility_prune
        }
    };
    let scale_small = scale
        .iter()
        .any(|value| !value.is_finite() || *value < BRUSH_MIN_SCALE);
    let scale_large = scale
        .iter()
        .any(|value| value.is_finite() && *value > max_allowed_bounds);
    let out_of_bounds = position
        .iter()
        .zip(center.iter())
        .any(|(position, center)| (*position - *center).abs() > max_allowed_bounds);
    contribution_prune || scale_small || scale_large || out_of_bounds || !retainable
}

fn build_host_snapshot_stats(
    len: usize,
    grad_2d_accum: &[f32],
    screen_grad_2d_accum: &[f32],
    abs_grad_2d_accum: &[f32],
    abs_pixel_grad_2d_accum: &[f32],
    pixel_coverage_accum: &[f32],
    camera_depth_accum: &[f32],
    grad_color_accum: &[f32],
    num_observations: &[f32],
    visible_observations: &[f32],
    actual_visible_observations: &[f32],
    splat_ages: &[usize],
    invisible_windows: &[usize],
) -> Vec<MetalGaussianStats> {
    (0..len)
        .map(|idx| {
            let observations = num_observations
                .get(idx)
                .copied()
                .unwrap_or_default()
                .max(0.0);
            let denom = observations.max(1.0);
            let visible_observations = visible_observations
                .get(idx)
                .copied()
                .unwrap_or_default()
                .max(0.0);
            let visible_count = visible_observations.round().min(usize::MAX as f32) as usize;
            let visible_denom = visible_observations.max(1.0);
            let actual_visible_observations = actual_visible_observations
                .get(idx)
                .copied()
                .unwrap_or_default()
                .max(0.0);
            let actual_visible_count =
                actual_visible_observations.round().min(usize::MAX as f32) as usize;
            let actual_visibility_ratio = if observations > 0.0 {
                (actual_visible_observations / observations).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let grad_2d = grad_2d_accum.get(idx).copied().unwrap_or_default();
            let screen_grad_2d = screen_grad_2d_accum.get(idx).copied().unwrap_or_default();
            let abs_grad_2d = abs_grad_2d_accum.get(idx).copied().unwrap_or_default();
            let abs_pixel_grad_2d = abs_pixel_grad_2d_accum
                .get(idx)
                .copied()
                .unwrap_or_default();
            let pixel_coverage = pixel_coverage_accum
                .get(idx)
                .copied()
                .unwrap_or_default()
                .max(0.0);
            let camera_depth = camera_depth_accum.get(idx).copied().unwrap_or_default();
            let grad_color = grad_color_accum.get(idx).copied().unwrap_or_default();
            let age = splat_ages.get(idx).copied().unwrap_or_default();
            let consecutive_invisible_epochs =
                invisible_windows.get(idx).copied().unwrap_or_default();
            let coverage_denom = pixel_coverage.max(1e-6);

            MetalGaussianStats {
                mean2d_grad: RunningMoments {
                    mean: grad_2d / denom,
                    m2: 0.0,
                    count: visible_count,
                },
                fragment_weight: RunningMoments {
                    mean: grad_color / denom,
                    m2: 0.0,
                    count: visible_count,
                },
                refine_weight_max: grad_2d / denom,
                screen_refine_weight_max: screen_grad_2d / visible_denom,
                abs_refine_weight_max: abs_grad_2d / visible_denom,
                abs_pixel_refine_weight_max: abs_pixel_grad_2d / coverage_denom,
                pixel_coverage_mean: pixel_coverage / visible_denom,
                camera_depth_mean: camera_depth / coverage_denom,
                visible_count,
                actual_visible_count,
                actual_visibility_ratio,
                age,
                consecutive_invisible_epochs,
                ..MetalGaussianStats::default()
            }
        })
        .collect()
}


#[cfg(test)]
mod tests;
