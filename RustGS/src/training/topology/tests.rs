use std::cmp::Ordering;

use glam::{Quat, Vec3};
use super::reference::density_controller_reference_summary;
use super::schedule::{schedule_topology, TopologyStepContext};
use super::splat_metrics::TopologySplatMetrics;
use super::{
    analyze_topology_candidates, apply_topology_metrics_delta, litegs_requested_additions,
    litegs_select_densify_candidates,
    litegs_select_densify_candidates_seeded, plan_topology_mutation, should_apply_topology_step,
    LiteGsDensifySelection, MetalGaussianStats, RunningMoments, TopologyCandidateInfo,
    TopologyMutationAftermath, TopologyMutationPlan, TopologyMutationRequest, TopologyPlanRow,
    TopologyPolicy, TopologyRefineDecay, TopologyStatsAction, BRUSH_MIN_OPACITY,
};
use crate::core::HostSplats as Splats;
use crate::sh::{rgb_to_sh0_value, sh_coeff_count_for_degree, SplatColorRepresentation};
use crate::training::reporting::metrics::ParityTopologyMetrics;
use crate::training::{LiteGsConfig, LiteGsOpacityResetMode, TrainingConfig};

pub(super) fn densify_snapshot_litegs(
    policy: &TopologyPolicy,
    snapshot: &mut Splats,
    stats: &mut Vec<MetalGaussianStats>,
    origins: &mut Vec<Option<usize>>,
    max_gaussians: usize,
    selected_indices: &[usize],
) -> usize {
    if snapshot.len() >= max_gaussians || selected_indices.is_empty() {
        return 0;
    }

    let clone_scale_threshold = policy.litegs_clone_scale_threshold();
    let mut clone_indices = Vec::new();
    let mut split_indices = Vec::new();

    for &idx in selected_indices
        .iter()
        .take(max_gaussians.saturating_sub(snapshot.len()))
    {
        let max_scale = snapshot.scale(idx).into_iter().fold(0.0f32, f32::max);
        if max_scale <= clone_scale_threshold {
            clone_indices.push(idx);
        } else {
            split_indices.push(idx);
        }
    }

    let mut added = 0usize;

    for idx in &split_indices {
        if snapshot.len() >= max_gaussians {
            break;
        }

        let mut position = snapshot.position(*idx);
        let mut log_scale = snapshot.log_scale(*idx);
        let rotation = snapshot.rotation(*idx);
        let opacity_logit = snapshot.opacity_logits[*idx];
        let sh_coeffs = snapshot.sh_coeffs_row(*idx).to_vec();
        let scale = snapshot.scale(*idx);
        let (max_axis, max_axis_scale) = scale
            .into_iter()
            .enumerate()
            .max_by(|lhs, rhs| lhs.1.partial_cmp(&rhs.1).unwrap_or(Ordering::Equal))
            .unwrap_or((0, 0.0));
        position[max_axis] += max_axis_scale * 0.5;
        log_scale[max_axis] = (max_axis_scale / 1.6f32).max(1e-6f32).ln();

        append_splat(snapshot, position, log_scale, rotation, opacity_logit, &sh_coeffs);
        stats.push(MetalGaussianStats::default());
        origins.push(None);
        added = added.saturating_add(1);
    }

    for idx in &clone_indices {
        if snapshot.len() >= max_gaussians {
            break;
        }

        let position = snapshot.position(*idx);
        let log_scale = snapshot.log_scale(*idx);
        let rotation = snapshot.rotation(*idx);
        let opacity_logit = snapshot.opacity_logits[*idx];
        let sh_coeffs = snapshot.sh_coeffs_row(*idx).to_vec();
        append_splat(snapshot, position, log_scale, rotation, opacity_logit, &sh_coeffs);
        stats.push(MetalGaussianStats::default());
        origins.push(None);
        added = added.saturating_add(1);
    }

    added
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct TopologyMutationResult {
    pub(super) added: usize,
    pub(super) pruned: usize,
    pub(super) aftermath: TopologyMutationAftermath,
}

impl TopologyPlanRow {
    fn position(&self, metrics: &TopologySplatMetrics) -> [f32; 3] {
        let source_idx = self.source_idx();
        let mut position = metrics.position(source_idx);
        match *self {
            Self::Existing { .. } => position,
            Self::BrushRefineExisting { sample_scalar, .. } => {
                let offset = brush_refine_offset(metrics, source_idx, sample_scalar);
                position[0] -= offset[0];
                position[1] -= offset[1];
                position[2] -= offset[2];
                position
            }
            Self::BrushRefineNew { sample_scalar, .. } => {
                let offset = brush_refine_offset(metrics, source_idx, sample_scalar);
                position[0] += offset[0];
                position[1] += offset[1];
                position[2] += offset[2];
                position
            }
        }
    }

    fn scale(&self, metrics: &TopologySplatMetrics) -> [f32; 3] {
        let source_idx = self.source_idx();
        let mut scale = metrics.scale(source_idx);
        match *self {
            Self::Existing { .. } => scale,
            Self::BrushRefineExisting { .. } | Self::BrushRefineNew { .. } => {
                scale = brush_refine_scale(scale);
                scale
            }
        }
    }

    fn opacity(&self, metrics: &TopologySplatMetrics) -> f32 {
        match *self {
            Self::BrushRefineExisting { .. } | Self::BrushRefineNew { .. } => {
                brush_refine_opacity(metrics.opacity(self.source_idx()))
            }
            _ => metrics.opacity(self.source_idx()),
        }
    }
}

impl TopologyMutationPlan {
    pub(super) fn remap_stats(&self, current: &[MetalGaussianStats]) -> Vec<MetalGaussianStats> {
        self.rows
            .iter()
            .map(|row| {
                if row.is_existing() {
                    current.get(row.source_idx()).copied().unwrap_or_default()
                } else {
                    MetalGaussianStats::default()
                }
            })
            .collect()
    }
}

pub(super) fn apply_snapshot_mutations(
    snapshot: &mut Splats,
    stats: &mut Vec<MetalGaussianStats>,
    origins: &mut Vec<Option<usize>>,
    request: TopologyMutationRequest<'_>,
) -> TopologyMutationResult {
    let metrics = TopologySplatMetrics::from_snapshot(snapshot);
    let plan = plan_topology_mutation(&metrics, request);
    if plan.added > 0
        || plan.pruned > 0
        || plan.rows.len() != snapshot.len()
        || plan.refine_decay.is_some()
        || plan.aftermath.apply_opacity_reset
    {
        *snapshot = rebuild_snapshot_from_plan(snapshot, &metrics, &plan);
        *stats = plan.remap_stats(stats);
        *origins = plan.origins();
    }

    TopologyMutationResult {
        added: plan.added,
        pruned: plan.pruned,
        aftermath: plan.aftermath,
    }
}

fn brush_refine_scale(scale: [f32; 3]) -> [f32; 3] {
    let mut refined = scale;
    let mut max_axis = 0usize;
    for axis in 1..3 {
        if refined[axis] > refined[max_axis] {
            max_axis = axis;
        }
    }
    refined[max_axis] *= 0.5;
    refined
}

fn brush_refine_opacity(opacity: f32) -> f32 {
    let clamped = opacity.clamp(BRUSH_MIN_OPACITY, 1.0 - BRUSH_MIN_OPACITY);
    let inverted = (1.0 - clamped).max(0.0);
    (1.0 - inverted.sqrt()).clamp(BRUSH_MIN_OPACITY, 1.0 - BRUSH_MIN_OPACITY)
}

fn brush_refine_offset(
    metrics: &TopologySplatMetrics,
    source_idx: usize,
    sample_scalar: f32,
) -> [f32; 3] {
    let rotation = metrics.rotation(source_idx);
    let quat = Quat::from_xyzw(rotation[1], rotation[2], rotation[3], rotation[0]);
    let quat = if quat.length_squared() > 0.0 {
        quat.normalize()
    } else {
        Quat::IDENTITY
    };
    (quat * (Vec3::from_array(metrics.scale(source_idx)) * sample_scalar)).to_array()
}

fn rebuild_snapshot_from_plan(
    snapshot: &Splats,
    metrics: &TopologySplatMetrics,
    plan: &TopologyMutationPlan,
) -> Splats {
    let mut rebuilt = empty_splats_like(snapshot, plan.rows.len());
    for row in &plan.rows {
        let source_idx = row.source_idx();
        let position = row.position(metrics);
        let mut scale = row.scale(metrics);
        let mut opacity = row.opacity(metrics).clamp(1e-6, 1.0 - 1e-6);
        if let Some(decay) = plan.refine_decay {
            apply_refine_decay_for_test(&mut scale, &mut opacity, decay);
        }
        if plan.aftermath.apply_opacity_reset {
            opacity = opacity.min(super::LITEGS_OPACITY_RESET_CAP);
        }
        let log_scale = scale.map(|value| value.max(1e-6).ln());
        let rotation = snapshot.rotation(source_idx);
        let opacity_logit = (opacity / (1.0 - opacity)).ln();
        append_splat(
            &mut rebuilt,
            position,
            log_scale,
            rotation,
            opacity_logit,
            snapshot.sh_coeffs_row(source_idx),
        );
    }
    rebuilt
}

fn empty_splats_like(snapshot: &Splats, row_count: usize) -> Splats {
    let sh_degree = snapshot.sh_degree();
    let sh_row_width = sh_coeff_count_for_degree(sh_degree) * 3;
    Splats::from_components(
        Vec::with_capacity(row_count * 3),
        Vec::with_capacity(row_count * 3),
        Vec::with_capacity(row_count * 4),
        Vec::with_capacity(row_count),
        Vec::with_capacity(row_count * sh_row_width),
        sh_degree,
    )
    .expect("empty topology test snapshot")
}

fn append_splat(
    splats: &mut Splats,
    position: [f32; 3],
    log_scale: [f32; 3],
    rotation: [f32; 4],
    opacity_logit: f32,
    sh_coeffs: &[f32],
) {
    let row_width = sh_coeff_count_for_degree(splats.sh_degree()) * 3;
    let copied = sh_coeffs.len().min(row_width);

    splats.positions.extend_from_slice(&position);
    splats.log_scales.extend_from_slice(&log_scale);
    splats.rotations.extend_from_slice(&rotation);
    splats.opacity_logits.push(opacity_logit);
    splats.sh_coeffs.extend_from_slice(&sh_coeffs[..copied]);
    if copied < row_width {
        splats
            .sh_coeffs
            .resize(splats.sh_coeffs.len() + (row_width - copied), 0.0);
    }
}

fn apply_refine_decay_for_test(
    scale: &mut [f32; 3],
    opacity: &mut f32,
    decay: TopologyRefineDecay,
) {
    let shrink_strength = 1.0 - decay.train_t.clamp(0.0, 1.0);
    if decay.scale_decay > 0.0 {
        let scale_scaling = (1.0 - decay.scale_decay * shrink_strength).max(1e-6);
        for value in scale {
            *value = (*value * scale_scaling).max(1e-12);
        }
    }
    if decay.opacity_decay > 0.0 {
        *opacity = (*opacity - decay.opacity_decay * shrink_strength).clamp(1e-12, 1.0 - 1e-12);
    }
}

fn test_snapshot(
    positions: Vec<f32>,
    log_scales: Vec<f32>,
    rotations: Vec<f32>,
    opacity_logits: Vec<f32>,
    colors: Vec<f32>,
    sh_rest: Vec<f32>,
    color_representation: SplatColorRepresentation,
) -> Splats {
    let sh_degree = color_representation.sh_degree();
    let row_count = opacity_logits.len();
    let sh_rest_row_width = sh_coeff_count_for_degree(sh_degree).saturating_sub(1) * 3;
    let mut sh_coeffs =
        Vec::with_capacity(row_count * sh_coeff_count_for_degree(sh_degree) * 3);
    for idx in 0..row_count {
        let color_base = idx * 3;
        match color_representation {
            SplatColorRepresentation::Rgb => {
                sh_coeffs.extend(
                    colors[color_base..color_base + 3]
                        .iter()
                        .copied()
                        .map(rgb_to_sh0_value),
                );
            }
            SplatColorRepresentation::SphericalHarmonics { .. } => {
                sh_coeffs.extend_from_slice(&colors[color_base..color_base + 3]);
                let rest = crate::core::splats::row_slice(&sh_rest, sh_rest_row_width, idx);
                sh_coeffs.extend_from_slice(rest);
            }
        }
    }
    Splats {
        positions,
        log_scales,
        rotations,
        opacity_logits,
        sh_coeffs,
        sh_degree,
    }
}

fn litegs_with(mut configure: impl FnMut(&mut LiteGsConfig)) -> LiteGsConfig {
    let mut litegs = LiteGsConfig::default();
    configure(&mut litegs);
    litegs
}

#[test]
fn litegs_schedule_respects_refine_cadence_and_freeze() {
    let config = TrainingConfig {
        iterations: 30,
        litegs: litegs_with(|litegs| {
            litegs.topology.densify_from = 1;
            litegs.topology.densify_until = Some(6);
            litegs.topology.refine_every = 2;
            litegs.topology.opacity_reset_interval = 2;
            litegs.topology.topology_freeze_after_epoch = Some(4);
        }),
        ..TrainingConfig::default()
    };
    let policy = TopologyPolicy::from_training_config(&config, 1.0);

    let epoch2 = schedule_topology(
        &policy,
        TopologyStepContext {
            iteration: 7,
            frame_count: 2,
        },
    );
    let epoch4 = schedule_topology(
        &policy,
        TopologyStepContext {
            iteration: 11,
            frame_count: 2,
        },
    );

    assert_eq!(epoch2.completed_epoch, Some(3));
    assert!(epoch2.densify);
    assert!(epoch2.prune);
    assert!(!epoch2.reset_opacity);
    assert!(epoch2.allow_extra_growth);

    assert_eq!(epoch4.completed_epoch, Some(5));
    assert!(!epoch4.densify);
    assert!(!epoch4.prune);
    assert!(!epoch4.reset_opacity);
}

#[test]
fn litegs_reset_mode_schedules_opacity_reset_on_epoch_cadence() {
    let config = TrainingConfig {
        iterations: 30,
        litegs: litegs_with(|litegs| {
            litegs.topology.refine_every = 2;
            litegs.topology.opacity_reset_interval = 2;
            litegs.topology.opacity_reset_mode = LiteGsOpacityResetMode::Reset;
        }),
        ..TrainingConfig::default()
    };
    let policy = TopologyPolicy::from_training_config(&config, 1.0);

    let epoch2 = schedule_topology(
        &policy,
        TopologyStepContext {
            iteration: 5,
            frame_count: 2,
        },
    );
    let epoch3 = schedule_topology(
        &policy,
        TopologyStepContext {
            iteration: 7,
            frame_count: 2,
        },
    );

    assert_eq!(epoch2.completed_epoch, Some(2));
    assert!(epoch2.reset_opacity);
    assert!(should_apply_topology_step(&config, 5, 2));

    assert_eq!(epoch3.completed_epoch, Some(3));
    assert!(!epoch3.reset_opacity);
}

#[test]
fn litegs_growth_freeze_still_allows_pruning() {
    let config = TrainingConfig {
        iterations: 30_000,
        litegs: litegs_with(|litegs| {
            litegs.topology.growth_freeze_after_epoch = Some(4);
            litegs.pruning.prune_until_epoch = Some(30);
            litegs.topology.refine_every = 160;
        }),
        ..TrainingConfig::default()
    };
    let policy = TopologyPolicy::from_training_config(&config, 1.0);

    let after_growth_freeze = schedule_topology(
        &policy,
        TopologyStepContext {
            iteration: 1_600 + 1,
            frame_count: 180,
        },
    );

    assert_eq!(after_growth_freeze.completed_epoch, Some(8));
    assert!(!after_growth_freeze.densify);
    assert!(after_growth_freeze.prune);
    assert!(!after_growth_freeze.allow_extra_growth);
}

#[test]
fn litegs_short_run_schedule_disables_densify_window() {
    let config = TrainingConfig {
        iterations: 500,
        litegs: litegs_with(|litegs| {
            litegs.topology.densify_from = 3;
            litegs.topology.refine_every = 200;
        }),
        ..TrainingConfig::default()
    };
    let policy = TopologyPolicy::from_training_config(&config, 1.0);

    assert_eq!(policy.litegs_total_epochs(638), 1);
    assert_eq!(policy.litegs_effective_densify_from_epoch(638), 1);
    assert_eq!(policy.litegs_densify_until_epoch(638), 1);

    let early = schedule_topology(
        &policy,
        TopologyStepContext {
            iteration: 100,
            frame_count: 638,
        },
    );
    let first_refine = schedule_topology(
        &policy,
        TopologyStepContext {
            iteration: 201,
            frame_count: 638,
        },
    );
    let second_refine = schedule_topology(
        &policy,
        TopologyStepContext {
            iteration: 401,
            frame_count: 638,
        },
    );

    assert_eq!(early.completed_epoch, Some(0));
    assert!(!early.densify);
    assert!(!early.prune);

    assert_eq!(first_refine.completed_epoch, Some(0));
    assert!(first_refine.densify);
    assert!(first_refine.prune);
    assert!(first_refine.allow_extra_growth);
    assert!(!first_refine.reset_opacity);

    assert_eq!(second_refine.completed_epoch, Some(0));
    assert!(second_refine.densify);
    assert!(second_refine.prune);
    assert!(second_refine.allow_extra_growth);
    assert!(!second_refine.reset_opacity);
}

#[test]
fn should_apply_topology_step_uses_litegs_refine_cadence() {
    let config = TrainingConfig {
        iterations: 500,
        litegs: litegs_with(|litegs| {
            litegs.topology.refine_every = 160;
        }),
        ..TrainingConfig::default()
    };

    assert!(
        !should_apply_topology_step(&config, 100, 638),
        "legacy cadence checkpoint should not trigger LiteGS refine scheduling"
    );
    assert!(
        should_apply_topology_step(&config, 161, 638),
        "LiteGS refine should trigger at phase_iter=160"
    );
    assert!(
        !should_apply_topology_step(&config, 201, 638),
        "non-refine iterations should not trigger topology work"
    );
}

#[test]
fn litegs_requested_additions_only_adds_extra_growth_when_enabled() {
    let infos = vec![
        candidate(true, false, 0, 0.0, 0.0),
        candidate(false, true, 1, 0.2, 1.0),
        candidate(false, true, 1, 0.3, 2.0),
        candidate(false, false, 1, 0.4, 0.0),
    ];

    assert_eq!(litegs_requested_additions(&infos, 0.5, false), 1);
    assert_eq!(litegs_requested_additions(&infos, 0.5, true), 1);
    assert_eq!(litegs_requested_additions(&infos, 1.0, true), 2);
}

#[test]
fn litegs_selection_prefers_replacements_before_extra_growth() {
    let infos = vec![
        candidate(true, false, 0, 0.0, 0.0),
        candidate(false, true, 1, 0.9, 2.0),
        candidate(false, true, 1, 0.8, 3.0),
        candidate(false, true, 1, 0.7, 1.0),
    ];

    let selection = litegs_select_densify_candidates(&infos, 3, 1.0, true);

    assert_eq!(selection.replacement_count, 1);
    assert_eq!(selection.extra_growth_count, 2);
    assert_eq!(selection.selected_indices.len(), 3);
    assert!(selection.selected_indices.iter().all(|idx| *idx > 0));
}

#[test]
fn litegs_selection_is_repeatable_for_same_seed() {
    let infos = vec![
        candidate(true, false, 0, 0.0, 0.0),
        candidate(false, true, 1, 0.9, 2.0),
        candidate(false, true, 1, 0.8, 3.0),
        candidate(false, true, 1, 0.7, 1.0),
        candidate(false, true, 1, 0.6, 4.0),
    ];

    let lhs = litegs_select_densify_candidates_seeded(&infos, 3, 1.0, true, 42);
    let rhs = litegs_select_densify_candidates_seeded(&infos, 3, 1.0, true, 42);

    assert_eq!(lhs.selected_indices, rhs.selected_indices);
    assert_eq!(lhs.replacement_count, rhs.replacement_count);
    assert_eq!(lhs.extra_growth_count, rhs.extra_growth_count);
}

#[test]
fn litegs_densify_preserves_sh_layout() {
    let config = TrainingConfig {
        litegs: litegs_with(|litegs| {
            litegs.rendering.sh_degree = 3;
        }),
        ..TrainingConfig::default()
    };
    let policy = TopologyPolicy::from_training_config(&config, 1.0);
    let mut snapshot = test_snapshot(
        vec![0.0, 0.0, 1.0],
        vec![0.05f32.ln(), 0.05f32.ln(), 0.05f32.ln()],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.5],
        vec![0.2, 0.4, 0.6],
        vec![0.1; 15 * 3],
        SplatColorRepresentation::SphericalHarmonics { degree: 3 },
    );
    let mut stats = vec![MetalGaussianStats::default()];
    let mut origins = vec![Some(0)];

    let added =
        densify_snapshot_litegs(&policy, &mut snapshot, &mut stats, &mut origins, 2, &[0]);

    assert_eq!(added, 1);
    assert_eq!(snapshot.len(), 2);
    assert_eq!(snapshot.sh_coeffs.len(), 2 * 16 * 3);
}

#[test]
fn analyze_topology_candidates_marks_litegs_growth_and_prune() {
    let config = TrainingConfig {
        litegs: litegs_with(|litegs| {
            litegs.pruning.prune_min_age = 1;
            litegs.pruning.prune_invisible_epochs = 1;
        }),
        ..TrainingConfig::default()
    };
    let policy = TopologyPolicy::from_training_config(&config, 1.0);
    let snapshot = test_snapshot(
        vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        vec![
            0.005f32.ln(),
            0.005f32.ln(),
            0.005f32.ln(),
            0.5f32.ln(),
            0.5f32.ln(),
            0.5f32.ln(),
        ],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        vec![2.0, -10.0],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        vec![0.0; 2 * 15 * 3],
        SplatColorRepresentation::SphericalHarmonics { degree: 3 },
    );
    let stats = vec![
        MetalGaussianStats {
            mean2d_grad: RunningMoments {
                mean: 1.0,
                m2: 0.0,
                count: 1,
            },
            refine_weight_max: 1.0,
            visible_count: 1,
            age: 1,
            ..Default::default()
        },
        MetalGaussianStats {
            age: 1,
            consecutive_invisible_epochs: 2,
            ..Default::default()
        },
    ];

    let metrics = TopologySplatMetrics::from_snapshot(&snapshot);
    let analysis = analyze_topology_candidates(&policy, &metrics, &stats, true);

    assert_eq!(analysis.clone_candidates, 1);
    assert_eq!(analysis.prune_candidates, 1);
    assert_eq!(analysis.growth_candidates, 1);
    assert!(analysis.infos[0].growth_candidate);
    assert!(analysis.infos[1].prune_candidate);
}

#[test]
fn analyze_topology_candidates_uses_configured_prune_opacity_threshold() {
    let config = TrainingConfig {
        litegs: litegs_with(|litegs| {
            litegs.pruning.prune_opacity_threshold = 0.02;
            litegs.pruning.prune_mode = crate::training::LiteGsPruneMode::Threshold;
            litegs.pruning.prune_min_age = 1;
            litegs.pruning.prune_invisible_epochs = 100;
        }),
        ..TrainingConfig::default()
    };
    let policy = TopologyPolicy::from_training_config(&config, 1.0);
    let snapshot = test_snapshot(
        vec![0.0, 0.0, 1.0],
        vec![0.005f32.ln(), 0.005f32.ln(), 0.005f32.ln()],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![(0.01f32 / 0.99f32).ln()],
        vec![1.0, 0.0, 0.0],
        Vec::new(),
        SplatColorRepresentation::Rgb,
    );
    let stats = vec![MetalGaussianStats {
        visible_count: 1,
        age: 1,
        ..Default::default()
    }];

    let metrics = TopologySplatMetrics::from_snapshot(&snapshot);
    let analysis = analyze_topology_candidates(&policy, &metrics, &stats, true);

    assert_eq!(analysis.prune_candidates, 1);
    assert!(analysis.infos[0].prune_candidate);
}

#[test]
fn weight_pruning_keeps_visible_low_opacity_splats_after_growth_freeze() {
    let config = TrainingConfig {
        litegs: litegs_with(|litegs| {
            litegs.pruning.prune_opacity_threshold = 0.02;
            litegs.pruning.prune_mode = crate::training::LiteGsPruneMode::Weight;
        }),
        ..TrainingConfig::default()
    };
    let policy = TopologyPolicy::from_training_config(&config, 1.0);
    let snapshot = test_snapshot(
        vec![0.0, 0.0, 1.0],
        vec![0.005f32.ln(), 0.005f32.ln(), 0.005f32.ln()],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![(0.01f32 / 0.99f32).ln()],
        vec![1.0, 0.0, 0.0],
        Vec::new(),
        SplatColorRepresentation::Rgb,
    );
    let stats = vec![MetalGaussianStats {
        visible_count: 1,
        age: 10,
        ..Default::default()
    }];

    let metrics = TopologySplatMetrics::from_snapshot(&snapshot);
    let analysis = analyze_topology_candidates(&policy, &metrics, &stats, false);

    assert_eq!(analysis.prune_candidates, 0);
    assert!(!analysis.infos[0].prune_candidate);
}

#[test]
fn weight_pruning_uses_opacity_during_growth() {
    let config = TrainingConfig {
        litegs: litegs_with(|litegs| {
            litegs.pruning.prune_opacity_threshold = 0.02;
            litegs.pruning.prune_mode = crate::training::LiteGsPruneMode::Weight;
        }),
        ..TrainingConfig::default()
    };
    let policy = TopologyPolicy::from_training_config(&config, 1.0);
    let snapshot = test_snapshot(
        vec![0.0, 0.0, 1.0],
        vec![0.005f32.ln(), 0.005f32.ln(), 0.005f32.ln()],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![(0.01f32 / 0.99f32).ln()],
        vec![1.0, 0.0, 0.0],
        Vec::new(),
        SplatColorRepresentation::Rgb,
    );
    let stats = vec![MetalGaussianStats {
        visible_count: 1,
        age: 10,
        ..Default::default()
    }];

    let metrics = TopologySplatMetrics::from_snapshot(&snapshot);
    let analysis = analyze_topology_candidates(&policy, &metrics, &stats, true);

    assert_eq!(analysis.prune_candidates, 1);
    assert!(analysis.infos[0].prune_candidate);
}

#[test]
fn invisible_pruning_requires_configured_consecutive_windows() {
    let config = TrainingConfig {
        litegs: litegs_with(|litegs| {
            litegs.pruning.prune_min_age = 1;
            litegs.pruning.prune_invisible_epochs = 3;
        }),
        ..TrainingConfig::default()
    };
    let policy = TopologyPolicy::from_training_config(&config, 1.0);
    let snapshot = test_snapshot(
        vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        vec![
            0.005f32.ln(),
            0.005f32.ln(),
            0.005f32.ln(),
            0.005f32.ln(),
            0.005f32.ln(),
            0.005f32.ln(),
        ],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        vec![1.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        Vec::new(),
        SplatColorRepresentation::Rgb,
    );
    let stats = vec![
        MetalGaussianStats {
            age: 10,
            consecutive_invisible_epochs: 2,
            ..Default::default()
        },
        MetalGaussianStats {
            age: 10,
            consecutive_invisible_epochs: 3,
            ..Default::default()
        },
    ];

    let metrics = TopologySplatMetrics::from_snapshot(&snapshot);
    let analysis = analyze_topology_candidates(&policy, &metrics, &stats, false);

    assert_eq!(analysis.prune_candidates, 1);
    assert!(!analysis.infos[0].prune_candidate);
    assert!(analysis.infos[1].prune_candidate);

    let growth_analysis = analyze_topology_candidates(&policy, &metrics, &stats, true);
    assert_eq!(growth_analysis.prune_candidates, 0);
    assert!(!growth_analysis.infos[1].prune_candidate);
}

#[test]
fn opacity_pruning_respects_min_age() {
    let config = TrainingConfig {
        litegs: litegs_with(|litegs| {
            litegs.pruning.prune_min_age = 5;
            litegs.pruning.prune_opacity_threshold = 0.02;
        }),
        ..TrainingConfig::default()
    };
    let policy = TopologyPolicy::from_training_config(&config, 1.0);
    let snapshot = test_snapshot(
        vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        vec![
            0.005f32.ln(),
            0.005f32.ln(),
            0.005f32.ln(),
            0.005f32.ln(),
            0.005f32.ln(),
            0.005f32.ln(),
        ],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        vec![(0.01f32 / 0.99f32).ln(), (0.01f32 / 0.99f32).ln()],
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        Vec::new(),
        SplatColorRepresentation::Rgb,
    );
    let stats = vec![
        MetalGaussianStats {
            visible_count: 1,
            age: 4,
            ..Default::default()
        },
        MetalGaussianStats {
            visible_count: 1,
            age: 5,
            ..Default::default()
        },
    ];

    let metrics = TopologySplatMetrics::from_snapshot(&snapshot);
    let analysis = analyze_topology_candidates(&policy, &metrics, &stats, true);

    assert_eq!(analysis.prune_candidates, 1);
    assert!(!analysis.infos[0].prune_candidate);
    assert!(analysis.infos[1].prune_candidate);
}

#[test]
fn density_controller_reference_summary_tracks_threshold_masks() {
    let config = TrainingConfig {
        iterations: 12,
        litegs: litegs_with(|litegs| {
            litegs.topology.densify_from = 1;
            litegs.topology.densify_until = Some(8);
            litegs.topology.densification_interval = 1;
            litegs.growth.growth_grad_threshold = 0.001;
            litegs.pruning.prune_mode = crate::training::LiteGsPruneMode::Threshold;
        }),
        ..TrainingConfig::default()
    };
    let policy = TopologyPolicy::from_training_config(&config, 2.0);
    let snapshot = test_snapshot(
        vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0],
        vec![
            0.005f32.ln(),
            0.005f32.ln(),
            0.005f32.ln(),
            0.05f32.ln(),
            0.05f32.ln(),
            0.05f32.ln(),
            0.01f32.ln(),
            0.01f32.ln(),
            0.01f32.ln(),
        ],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        vec![2.0, 2.0, -10.0],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        vec![0.0; 3 * 15 * 3],
        SplatColorRepresentation::SphericalHarmonics { degree: 3 },
    );
    let stats = vec![
        MetalGaussianStats {
            mean2d_grad: RunningMoments {
                mean: 0.01,
                m2: 0.0,
                count: 1,
            },
            visible_count: 1,
            ..Default::default()
        },
        MetalGaussianStats {
            mean2d_grad: RunningMoments {
                mean: 0.02,
                m2: 0.0,
                count: 1,
            },
            visible_count: 1,
            ..Default::default()
        },
        MetalGaussianStats::default(),
    ];

    let metrics = TopologySplatMetrics::from_snapshot(&snapshot);
    let summary = density_controller_reference_summary(&policy, &metrics, &stats, Some(2));

    assert_eq!(summary.clone_candidates(), 1);
    assert_eq!(summary.split_candidates(), 1);
    assert_eq!(summary.prune_candidates(), 1);
    assert_eq!(summary.densify_budget, Some(3));
}

#[test]
fn density_controller_reference_summary_tracks_weight_budget() {
    let config = TrainingConfig {
        iterations: 12,
        litegs: litegs_with(|litegs| {
            litegs.topology.densify_from = 1;
            litegs.topology.densify_until = Some(5);
            litegs.topology.densification_interval = 1;
            litegs.growth.growth_grad_threshold = 0.001;
            litegs.pruning.prune_mode = crate::training::LiteGsPruneMode::Weight;
            litegs.topology.target_primitives = 6;
        }),
        ..TrainingConfig::default()
    };
    let policy = TopologyPolicy::from_training_config(&config, 1.0);
    let snapshot = test_snapshot(
        vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0],
        vec![0.01f32.ln(); 9],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        vec![2.0, 2.0, 2.0],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        vec![0.0; 3 * 15 * 3],
        SplatColorRepresentation::SphericalHarmonics { degree: 3 },
    );
    let stats = vec![
        MetalGaussianStats {
            mean2d_grad: RunningMoments {
                mean: 0.01,
                m2: 0.0,
                count: 1,
            },
            fragment_weight: RunningMoments {
                mean: 0.5,
                m2: 0.0,
                count: 1,
            },
            fragment_err: RunningMoments {
                mean: 0.6,
                m2: 0.1,
                count: 2,
            },
            visible_count: 2,
            ..Default::default()
        },
        MetalGaussianStats {
            mean2d_grad: RunningMoments {
                mean: 0.01,
                m2: 0.0,
                count: 1,
            },
            fragment_weight: RunningMoments {
                mean: 0.4,
                m2: 0.0,
                count: 1,
            },
            fragment_err: RunningMoments {
                mean: 0.5,
                m2: 0.05,
                count: 2,
            },
            visible_count: 2,
            ..Default::default()
        },
        MetalGaussianStats {
            mean2d_grad: RunningMoments {
                mean: 0.01,
                m2: 0.0,
                count: 1,
            },
            visible_count: 3,
            ..Default::default()
        },
    ];

    let metrics = TopologySplatMetrics::from_snapshot(&snapshot);
    let summary = density_controller_reference_summary(&policy, &metrics, &stats, Some(3));

    assert_eq!(summary.prune_candidates(), 1);
    assert_eq!(summary.clone_candidates(), 3);
    assert_eq!(summary.split_candidates(), 0);
    assert_eq!(summary.densify_budget, Some(3));
}

#[test]
fn mutation_aftermath_requests_structural_rebuild_and_metrics() {
    let mut snapshot = test_snapshot(
        vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        vec![
            0.05f32.ln(),
            0.05f32.ln(),
            0.05f32.ln(),
            0.05f32.ln(),
            0.05f32.ln(),
            0.05f32.ln(),
        ],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        vec![2.0, -10.0],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        Vec::new(),
        SplatColorRepresentation::Rgb,
    );
    let mut stats = vec![
        MetalGaussianStats {
            mean2d_grad: RunningMoments {
                mean: 1.0,
                m2: 0.0,
                count: 1,
            },
            age: 5,
            ..Default::default()
        },
        MetalGaussianStats::default(),
    ];
    let mut origins = vec![Some(0), Some(1)];
    let infos = vec![
        candidate(false, true, 1, 0.9, 1.0),
        candidate(true, false, 0, 0.0, 0.0),
    ];
    let selection = LiteGsDensifySelection {
        selected_indices: vec![0],
        replacement_count: 1,
        extra_growth_count: 0,
    };

    let mutation = apply_snapshot_mutations(
        &mut snapshot,
        &mut stats,
        &mut origins,
        TopologyMutationRequest {
            should_densify: true,
            should_reset_opacity: false,
            refine_decay: None,
            completed_epoch: Some(2),
            late_stage: true,
            infos: &infos,
            litegs_selection: &selection,
            random_seed: 0,
        },
    );

    assert_eq!(mutation.added, 1);
    assert_eq!(mutation.pruned, 1);
    assert_eq!(snapshot.len(), 2);
    assert_eq!(
        mutation.aftermath.gaussian_stats_action,
        TopologyStatsAction::UseMutated
    );
    assert!(mutation.aftermath.requires_runtime_rebuild);
    assert!(mutation.aftermath.requires_adam_rebuild);
    assert!(mutation.aftermath.requires_cluster_resync);
    assert!(mutation.aftermath.requires_runtime_reserve);
    assert!(!mutation.aftermath.apply_opacity_reset);

    let mut metrics = ParityTopologyMetrics::default();
    apply_topology_metrics_delta(&mut metrics, mutation.aftermath.metrics_delta);
    assert_eq!(metrics.final_gaussians, Some(2));
    assert_eq!(metrics.densify_events, 1);
    assert_eq!(metrics.densify_added, 1);
    assert_eq!(metrics.prune_events, 1);
    assert_eq!(metrics.prune_removed, 1);
    assert_eq!(metrics.first_densify_epoch, Some(2));
    assert_eq!(metrics.last_prune_epoch, Some(2));
    assert_eq!(metrics.late_stage_densify_events, 1);
    assert_eq!(metrics.late_stage_prune_events, 1);
}

#[test]
fn mutation_aftermath_marks_litegs_opacity_reset_without_structural_change() {
    let mut snapshot = test_snapshot(
        vec![0.0, 0.0, 1.0],
        vec![0.05f32.ln(), 0.05f32.ln(), 0.05f32.ln()],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![2.0],
        vec![1.0, 0.0, 0.0],
        Vec::new(),
        SplatColorRepresentation::Rgb,
    );
    let mut stats = vec![MetalGaussianStats::default()];
    let mut origins = vec![Some(0)];
    let infos = vec![candidate(false, false, 1, 0.9, 0.0)];

    let mutation = apply_snapshot_mutations(
        &mut snapshot,
        &mut stats,
        &mut origins,
        TopologyMutationRequest {
            should_densify: false,
            should_reset_opacity: true,
            refine_decay: None,
            completed_epoch: Some(3),
            late_stage: true,
            infos: &infos,
            litegs_selection: &LiteGsDensifySelection::default(),
            random_seed: 0,
        },
    );

    assert_eq!(mutation.added, 0);
    assert_eq!(mutation.pruned, 0);
    assert_eq!(
        mutation.aftermath.gaussian_stats_action,
        TopologyStatsAction::KeepCurrent
    );
    assert!(!mutation.aftermath.requires_runtime_rebuild);
    assert!(mutation.aftermath.apply_opacity_reset);
    assert!(mutation.aftermath.reset_refine_window_stats);

    let mut metrics = ParityTopologyMetrics::default();
    apply_topology_metrics_delta(&mut metrics, mutation.aftermath.metrics_delta);
    assert_eq!(metrics.final_gaussians, Some(1));
    assert_eq!(metrics.opacity_reset_events, 1);
    assert_eq!(metrics.first_opacity_reset_epoch, Some(3));
    assert_eq!(metrics.late_stage_opacity_reset_events, 1);
}

#[test]
fn opacity_reset_caps_high_opacity_without_raising_low_opacity() {
    let mut snapshot = test_snapshot(
        vec![0.0, 0.0, 1.0, 0.1, 0.0, 1.0],
        vec![
            0.05f32.ln(),
            0.05f32.ln(),
            0.05f32.ln(),
            0.05f32.ln(),
            0.05f32.ln(),
            0.05f32.ln(),
        ],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        vec![
            (0.8f32 / (1.0 - 0.8f32)).ln(),
            (0.005f32 / (1.0 - 0.005f32)).ln(),
        ],
        vec![1.0, 0.0, 0.0, 0.5, 0.5, 0.5],
        Vec::new(),
        SplatColorRepresentation::Rgb,
    );
    let mut stats = vec![MetalGaussianStats::default(); 2];
    let mut origins = vec![Some(0), Some(1)];
    let infos = vec![
        candidate(false, false, 1, 0.8, 0.0),
        candidate(false, false, 1, 0.005, 0.0),
    ];

    let mutation = apply_snapshot_mutations(
        &mut snapshot,
        &mut stats,
        &mut origins,
        TopologyMutationRequest {
            should_densify: false,
            should_reset_opacity: true,
            refine_decay: None,
            completed_epoch: Some(2),
            late_stage: false,
            infos: &infos,
            litegs_selection: &LiteGsDensifySelection::default(),
            random_seed: 0,
        },
    );

    assert!(mutation.aftermath.apply_opacity_reset);
    assert!(mutation.aftermath.requires_adam_rebuild);
    assert!(plan_topology_mutation(
        &TopologySplatMetrics::from_snapshot(&snapshot),
        TopologyMutationRequest {
            should_densify: false,
            should_reset_opacity: true,
            refine_decay: None,
            completed_epoch: Some(2),
            late_stage: false,
            infos: &infos,
            litegs_selection: &LiteGsDensifySelection::default(),
            random_seed: 0,
        }
    )
    .mutates_splats());
    assert!((snapshot.opacity(0) - super::LITEGS_OPACITY_RESET_CAP).abs() < 1e-6);
    assert!((snapshot.opacity(1) - 0.005).abs() < 1e-6);
}

#[test]
fn no_op_topology_plan_does_not_mutate_splats() {
    let metrics = TopologySplatMetrics::from_snapshot(&test_snapshot(
        vec![0.0, 0.0, 1.0],
        vec![0.05f32.ln(), 0.05f32.ln(), 0.05f32.ln()],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![2.0],
        vec![1.0, 0.0, 0.0],
        Vec::new(),
        SplatColorRepresentation::Rgb,
    ));
    let infos = vec![candidate(false, false, 1, 0.9, 0.0)];
    let plan = plan_topology_mutation(
        &metrics,
        TopologyMutationRequest {
            should_densify: false,
            should_reset_opacity: false,
            refine_decay: None,
            completed_epoch: Some(3),
            late_stage: false,
            infos: &infos,
            litegs_selection: &LiteGsDensifySelection::default(),
            random_seed: 0,
        },
    );

    assert_eq!(plan.added, 0);
    assert_eq!(plan.pruned, 0);
    assert!(!plan.mutates_splats());
}

#[test]
fn refine_decay_shrinks_existing_splats_without_structural_change() {
    let mut snapshot = test_snapshot(
        vec![0.0, 0.0, 1.0],
        vec![0.1f32.ln(), 0.1f32.ln(), 0.1f32.ln()],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0],
        vec![1.0, 0.0, 0.0],
        Vec::new(),
        SplatColorRepresentation::Rgb,
    );
    let mut stats = vec![MetalGaussianStats::default()];
    let mut origins = vec![Some(0)];
    let infos = vec![candidate(false, false, 1, 0.5, 0.0)];

    let mutation = apply_snapshot_mutations(
        &mut snapshot,
        &mut stats,
        &mut origins,
        TopologyMutationRequest {
            should_densify: false,
            should_reset_opacity: false,
            refine_decay: Some(TopologyRefineDecay {
                opacity_decay: 0.1,
                scale_decay: 0.2,
                train_t: 0.5,
            }),
            completed_epoch: Some(3),
            late_stage: false,
            infos: &infos,
            litegs_selection: &LiteGsDensifySelection::default(),
            random_seed: 0,
        },
    );

    assert_eq!(mutation.added, 0);
    assert_eq!(mutation.pruned, 0);
    assert_eq!(snapshot.len(), 1);
    assert!((snapshot.scale(0)[0] - 0.09).abs() < 1e-6);
    assert!((snapshot.opacity(0) - 0.45).abs() < 1e-6);
}

fn candidate(
    prune_candidate: bool,
    growth_candidate: bool,
    visible_count: usize,
    opacity: f32,
    mean2d_grad: f32,
) -> TopologyCandidateInfo {
    TopologyCandidateInfo {
        opacity,
        mean2d_grad,
        screen_mean2d_grad: mean2d_grad,
        abs_mean2d_grad: mean2d_grad,
        abs_pixel_mean2d_grad: mean2d_grad,
        pixel_coverage: 1.0,
        camera_depth: 1.0,
        depth_scale: 1.0,
        split_score: mean2d_grad,
        growth_weight: mean2d_grad,
        visible_count,
        actual_visible_count: visible_count,
        actual_visibility_ratio: if visible_count == 0 { 0.0 } else { 1.0 },
        age: 10,
        consecutive_invisible_epochs: if visible_count == 0 { 10 } else { 0 },
        prune_candidate,
        growth_candidate,
        split_candidate: growth_candidate,
        clone_candidate: growth_candidate,
    }
}
