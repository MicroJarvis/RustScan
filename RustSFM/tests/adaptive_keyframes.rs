use rustsfm::{
    select_adaptive_keyframes_from_metrics, AdaptiveKeyframePairMetrics,
    AdaptiveKeyframeSelectionConfig, AdaptiveKeyframeSelectionDecision,
    AdaptiveKeyframeSelectionError,
};
use std::collections::BTreeSet;

fn config() -> AdaptiveKeyframeSelectionConfig {
    AdaptiveKeyframeSelectionConfig {
        retention_feature_coverage: 0.35,
        min_inliers: 15,
        min_inlier_ratio: 0.20,
        min_triangulated: 4,
    }
}

fn metrics(
    anchor: u32,
    candidate: u32,
    coverage: f64,
    connected: bool,
) -> AdaptiveKeyframePairMetrics {
    AdaptiveKeyframePairMetrics {
        anchor_frame_id: anchor,
        candidate_frame_id: candidate,
        descriptor_matches: if connected { 100 } else { 10 },
        inliers: if connected { 50 } else { 2 },
        triangulated: if connected { 40 } else { 0 },
        inlier_ratio: if connected { 0.5 } else { 0.2 },
        feature_coverage: coverage,
    }
}

#[test]
fn highly_redundant_sequence_selects_only_boundaries() {
    let evidence = [
        metrics(1, 2, 0.80, true),
        metrics(1, 3, 0.75, true),
        metrics(1, 4, 0.70, true),
    ];

    let result =
        select_adaptive_keyframes_from_metrics(&[1, 2, 3, 4], &evidence, &config()).unwrap();

    assert_eq!(result.selected_frame_ids, vec![1, 4]);
    assert_eq!(result.evaluated_pairs, 3);
    assert!(result
        .diagnostics
        .iter()
        .all(|diagnostic| diagnostic.decision == AdaptiveKeyframeSelectionDecision::Redundant));
}

#[test]
fn low_coverage_valid_geometry_selects_connected_transition_and_advances_anchor() {
    let evidence = [
        metrics(1, 2, 0.80, true),
        metrics(1, 3, 0.25, true),
        metrics(3, 4, 0.80, true),
    ];

    let result =
        select_adaptive_keyframes_from_metrics(&[1, 2, 3, 4], &evidence, &config()).unwrap();

    assert_eq!(result.selected_frame_ids, vec![1, 3, 4]);
    assert!(result.diagnostics.iter().any(|diagnostic| {
        diagnostic.metrics.anchor_frame_id == 1
            && diagnostic.metrics.candidate_frame_id == 3
            && diagnostic.decision == AdaptiveKeyframeSelectionDecision::ConnectedTransition
    }));
    assert!(result.diagnostics.iter().any(|diagnostic| {
        diagnostic.metrics.anchor_frame_id == 3 && diagnostic.metrics.candidate_frame_id == 4
    }));
}

#[test]
fn connectivity_loss_retains_bridge_retries_candidate_and_terminates() {
    let evidence = [
        metrics(1, 2, 0.80, true),
        metrics(1, 3, 0.10, false),
        metrics(2, 3, 0.25, true),
        metrics(3, 4, 0.80, true),
    ];

    let result =
        select_adaptive_keyframes_from_metrics(&[1, 2, 3, 4], &evidence, &config()).unwrap();

    assert_eq!(result.selected_frame_ids, vec![1, 2, 3, 4]);
    assert_eq!(result.evaluated_pairs, 4);
    assert_eq!(
        result
            .selected_frame_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len(),
        result.selected_frame_ids.len()
    );
    assert!(result.diagnostics.iter().any(|diagnostic| {
        diagnostic.metrics.anchor_frame_id == 1
            && diagnostic.metrics.candidate_frame_id == 3
            && diagnostic.decision == AdaptiveKeyframeSelectionDecision::ConnectivityBridge
    }));
    assert!(result.diagnostics.iter().any(|diagnostic| {
        diagnostic.metrics.anchor_frame_id == 2 && diagnostic.metrics.candidate_frame_id == 3
    }));
}

#[test]
fn fast_changing_sequence_selects_more_than_redundant_sequence_of_same_length() {
    let redundant = [
        metrics(1, 2, 0.8, true),
        metrics(1, 3, 0.8, true),
        metrics(1, 4, 0.8, true),
        metrics(1, 5, 0.8, true),
        metrics(1, 6, 0.8, true),
    ];
    let changing = [
        metrics(1, 2, 0.2, true),
        metrics(2, 3, 0.2, true),
        metrics(3, 4, 0.2, true),
        metrics(4, 5, 0.2, true),
        metrics(5, 6, 0.2, true),
    ];

    let redundant_result =
        select_adaptive_keyframes_from_metrics(&[1, 2, 3, 4, 5, 6], &redundant, &config()).unwrap();
    let changing_result =
        select_adaptive_keyframes_from_metrics(&[1, 2, 3, 4, 5, 6], &changing, &config()).unwrap();

    assert!(changing_result.selected_frame_ids.len() > redundant_result.selected_frame_ids.len());
}

#[test]
fn policy_is_deterministic_ordered_unique_and_preserves_boundaries() {
    let evidence = [metrics(10, 20, 0.8, true), metrics(10, 30, 0.8, true)];

    let first =
        select_adaptive_keyframes_from_metrics(&[10, 20, 30], &evidence, &config()).unwrap();
    let second =
        select_adaptive_keyframes_from_metrics(&[10, 20, 30], &evidence, &config()).unwrap();

    assert_eq!(first, second);
    assert_eq!(first.selected_frame_ids, vec![10, 30]);
    assert_eq!(first.selected_frame_ids.first(), Some(&10));
    assert_eq!(first.selected_frame_ids.last(), Some(&30));
    assert_eq!(
        first
            .selected_frame_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len(),
        first.selected_frame_ids.len()
    );
}

#[test]
fn policy_rejects_invalid_config_and_input() {
    let mut invalid_coverage = config();
    invalid_coverage.retention_feature_coverage = f64::NAN;
    assert_eq!(
        invalid_coverage.validate(),
        Err(AdaptiveKeyframeSelectionError::InvalidConfigMetric {
            field: "retention_feature_coverage"
        })
    );

    let mut out_of_range_coverage = config();
    out_of_range_coverage.retention_feature_coverage = 1.01;
    assert_eq!(
        out_of_range_coverage.validate(),
        Err(AdaptiveKeyframeSelectionError::InvalidConfigMetric {
            field: "retention_feature_coverage"
        })
    );

    let mut invalid_ratio = config();
    invalid_ratio.min_inlier_ratio = -0.1;
    assert_eq!(
        invalid_ratio.validate(),
        Err(AdaptiveKeyframeSelectionError::InvalidConfigMetric {
            field: "min_inlier_ratio"
        })
    );

    let mut zero_inliers = config();
    zero_inliers.min_inliers = 0;
    assert_eq!(
        zero_inliers.validate(),
        Err(AdaptiveKeyframeSelectionError::InvalidConfigMetric {
            field: "min_inliers"
        })
    );

    let mut zero_triangulated = config();
    zero_triangulated.min_triangulated = 0;
    assert_eq!(
        zero_triangulated.validate(),
        Err(AdaptiveKeyframeSelectionError::InvalidConfigMetric {
            field: "min_triangulated"
        })
    );

    assert_eq!(
        select_adaptive_keyframes_from_metrics(&[1], &[], &config()),
        Err(AdaptiveKeyframeSelectionError::InsufficientFrames { usable_frames: 1 })
    );
    assert_eq!(
        select_adaptive_keyframes_from_metrics(&[1, 1], &[], &config()),
        Err(AdaptiveKeyframeSelectionError::DuplicateFrameId { frame_id: 1 })
    );
}

#[test]
fn policy_rejects_missing_or_non_finite_pair_evidence() {
    assert_eq!(
        select_adaptive_keyframes_from_metrics(&[1, 2], &[], &config()),
        Err(AdaptiveKeyframeSelectionError::MissingPairEvidence {
            anchor_frame_id: 1,
            candidate_frame_id: 2,
        })
    );

    let mut non_finite_ratio = metrics(1, 2, 0.2, true);
    non_finite_ratio.inlier_ratio = f64::NAN;
    assert_eq!(
        select_adaptive_keyframes_from_metrics(&[1, 2], &[non_finite_ratio], &config()),
        Err(AdaptiveKeyframeSelectionError::NonFinitePairMetric {
            anchor_frame_id: 1,
            candidate_frame_id: 2,
            field: "inlier_ratio",
        })
    );

    let mut non_finite_coverage = metrics(1, 2, f64::INFINITY, true);
    non_finite_coverage.feature_coverage = f64::INFINITY;
    assert_eq!(
        select_adaptive_keyframes_from_metrics(&[1, 2], &[non_finite_coverage], &config()),
        Err(AdaptiveKeyframeSelectionError::NonFinitePairMetric {
            anchor_frame_id: 1,
            candidate_frame_id: 2,
            field: "feature_coverage",
        })
    );
}
