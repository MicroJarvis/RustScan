use super::{LiteGsDensifySelection, TopologyAnalysis, TopologyPolicy};
use crate::training::{LiteGsOpacityResetMode, TrainingConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TopologyExecutionDisposition {
    Apply,
    SkipNoEligibleCandidates,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct TopologyExecutionPlan {
    pub(super) completed_epoch: Option<usize>,
    pub(super) should_densify: bool,
    pub(super) should_reset_opacity: bool,
    pub(super) disposition: TopologyExecutionDisposition,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct TopologySchedule {
    pub(super) completed_epoch: Option<usize>,
    pub(super) densify: bool,
    pub(super) prune: bool,
    pub(super) reset_opacity: bool,
    pub(super) allow_extra_growth: bool,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct TopologyStepContext {
    pub(super) iteration: usize,
    pub(super) frame_count: usize,
}

pub(super) fn schedule_topology(
    policy: &TopologyPolicy,
    step: TopologyStepContext,
) -> TopologySchedule {
    let Some(epoch) = litegs_current_epoch(step.iteration, step.frame_count) else {
        return TopologySchedule::default();
    };
    let phase_iter = step.iteration.saturating_sub(1);
    let topology_frozen = policy
        .litegs
        .topology
        .topology_freeze_after_epoch
        .map(|freeze_epoch| epoch >= freeze_epoch)
        .unwrap_or(false);
    let growth_frozen = topology_frozen
        || policy
            .litegs
            .topology
            .growth_freeze_after_epoch
            .map(|freeze_epoch| epoch >= freeze_epoch)
            .unwrap_or(false);
    let refine_every = policy.litegs.topology.refine_every.max(1);
    let on_refine_cadence = phase_iter > 0 && phase_iter.is_multiple_of(refine_every);
    let densify_from = policy.litegs_effective_densify_from_epoch(step.frame_count);
    let densify_until = policy.litegs_densify_until_epoch(step.frame_count);
    let densification_interval = policy.litegs.topology.densification_interval.max(1);
    let in_densify_window = epoch >= densify_from && epoch < densify_until;
    let on_densify_epoch = in_densify_window
        && epoch
            .saturating_sub(densify_from)
            .is_multiple_of(densification_interval);
    let prune_start = densify_from.saturating_add(policy.litegs.pruning.prune_offset_epochs);
    let on_prune_epoch = epoch >= prune_start
        && epoch
            .saturating_sub(prune_start)
            .is_multiple_of(densification_interval);
    let default_prune_until = densify_until.max(prune_start.saturating_add(1));
    let prune_window = policy
        .litegs
        .pruning
        .prune_until_epoch
        .map(|until_epoch| epoch < until_epoch)
        .unwrap_or(epoch < default_prune_until);
    let densify = on_refine_cadence && !growth_frozen && on_densify_epoch;
    let prune = on_refine_cadence && !topology_frozen && on_prune_epoch && prune_window;
    let reset_opacity = on_refine_cadence
        && !topology_frozen
        && matches!(
            policy.litegs.topology.opacity_reset_mode,
            LiteGsOpacityResetMode::Reset
        )
        && policy.litegs.topology.opacity_reset_interval > 0
        && epoch > 0
        && epoch.is_multiple_of(policy.litegs.topology.opacity_reset_interval);
    TopologySchedule {
        completed_epoch: Some(epoch),
        densify,
        prune,
        reset_opacity,
        allow_extra_growth: densify && phase_iter < policy.litegs.growth.growth_stop_iter,
    }
}

pub(crate) fn should_apply_topology_step(
    config: &TrainingConfig,
    iteration: usize,
    frame_count: usize,
) -> bool {
    let policy = TopologyPolicy::from_training_config(config, 1.0);
    let schedule = schedule_topology(
        &policy,
        TopologyStepContext {
            iteration,
            frame_count,
        },
    );
    schedule.densify || schedule.prune || schedule.reset_opacity
}

pub(super) fn plan_topology_execution(
    _policy: &TopologyPolicy,
    schedule: TopologySchedule,
    analysis: &TopologyAnalysis,
    litegs_selection: &LiteGsDensifySelection,
) -> TopologyExecutionPlan {
    let mut plan = TopologyExecutionPlan {
        completed_epoch: schedule.completed_epoch,
        should_densify: schedule.densify,
        should_reset_opacity: schedule.reset_opacity,
        disposition: TopologyExecutionDisposition::Apply,
    };

    let has_candidates =
        !litegs_selection.selected_indices.is_empty() || analysis.prune_candidates > 0;
    if !has_candidates && !plan.should_reset_opacity {
        plan.disposition = TopologyExecutionDisposition::SkipNoEligibleCandidates;
    }

    plan
}

fn litegs_current_epoch(iteration: usize, frame_count: usize) -> Option<usize> {
    if frame_count == 0 || iteration == 0 {
        return None;
    }
    Some(iteration.saturating_sub(1) / frame_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_consumes_densify_and_prune_windows() {
        let config = TrainingConfig {
            iterations: 1_000,
            litegs: crate::training::LiteGsConfig {
                topology: crate::training::LiteGsTopologyConfig {
                    densify_from: 3,
                    densify_until: Some(8),
                    densification_interval: 2,
                    refine_every: 1,
                    ..Default::default()
                },
                pruning: crate::training::LiteGsPruningConfig {
                    prune_offset_epochs: 1,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let policy = TopologyPolicy::from_training_config(&config, 1.0);
        let at_epoch = |epoch: usize| {
            schedule_topology(
                &policy,
                TopologyStepContext {
                    iteration: epoch * 10 + 1,
                    frame_count: 10,
                },
            )
        };

        assert!(!at_epoch(2).densify);
        assert!(at_epoch(3).densify);
        assert!(!at_epoch(3).prune);
        assert!(!at_epoch(4).densify);
        assert!(at_epoch(4).prune);
        assert!(at_epoch(5).densify);
        assert!(!at_epoch(5).prune);
        assert!(!at_epoch(8).densify);
        assert!(!at_epoch(8).prune);
    }
}
