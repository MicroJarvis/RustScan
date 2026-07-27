use rustsfm::{
    run_adaptive_keyframe_selection, select_adaptive_keyframes_from_metrics,
    AdaptiveKeyframePairMetrics, AdaptiveKeyframeSelectionConfig,
    AdaptiveKeyframeSelectionDecision, AdaptiveKeyframeSelectionError, MapperConfig, SequenceFrame,
    SfmTaskContext, SfmTaskControl, SfmTaskOperation, SfmTaskStage, SfmTaskStop,
};
use std::collections::BTreeSet;
use std::path::Path;

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

#[test]
fn runtime_selection_honors_cancellation_before_image_processing() {
    let temp = tempfile::tempdir().unwrap();
    let image_dir = temp.path().join("images");
    std::fs::create_dir(&image_dir).unwrap();
    let frames = [
        write_runtime_frame(&image_dir, 1, "000.png"),
        write_runtime_frame(&image_dir, 2, "001.png"),
    ];
    let output = temp.path().join("output");
    let control = SfmTaskControl::new();
    control.request_cancel();
    let mut sink = |_| {};
    let mut task = SfmTaskContext::new(&control, &mut sink);

    let error = run_adaptive_keyframe_selection(
        &frames,
        &config(),
        &MapperConfig::default(),
        &output,
        &mut task,
    )
    .unwrap_err();

    assert_eq!(
        error.downcast_ref::<SfmTaskStop>(),
        Some(&SfmTaskStop::Cancelled)
    );
    assert!(!output.exists());
}

#[test]
fn runtime_selection_acquires_finite_metrics_and_reports_selection_progress() {
    let temp = tempfile::tempdir().unwrap();
    let image_dir = temp.path().join("images");
    std::fs::create_dir(&image_dir).unwrap();
    let frames = [
        write_textured_runtime_frame(&image_dir, 10, "000.png", 0),
        write_textured_runtime_frame(&image_dir, 20, "001.png", 3),
        write_textured_runtime_frame(&image_dir, 30, "002.png", 6),
    ];
    let output = temp.path().join("output");
    let mut mapper = MapperConfig::default();
    mapper.max_features = 512;
    mapper.min_matches = 4;
    mapper.min_inliers = 4;
    mapper.min_triangulated = 1;
    mapper.essential_iterations = 250;
    mapper.sift_extraction.use_gpu = false;
    mapper.sift_matching.use_gpu = false;
    let selection = AdaptiveKeyframeSelectionConfig {
        retention_feature_coverage: 0.35,
        min_inliers: 4,
        min_inlier_ratio: 0.01,
        min_triangulated: 1,
    };
    let control = SfmTaskControl::new();
    let mut events = Vec::new();
    let mut sink = |event| events.push(event);
    let mut task = SfmTaskContext::new(&control, &mut sink);

    let result =
        run_adaptive_keyframe_selection(&frames, &selection, &mapper, &output, &mut task).unwrap();
    drop(task);
    drop(sink);

    assert_eq!(result.imported_frames, 3);
    assert_eq!(result.usable_frames, 3);
    assert_eq!(result.selected_frame_ids.first(), Some(&10));
    assert_eq!(result.selected_frame_ids.last(), Some(&30));
    assert!(result.evaluated_pairs >= 2);
    assert!(result.diagnostics.iter().all(|diagnostic| {
        diagnostic.metrics.inlier_ratio.is_finite()
            && diagnostic.metrics.feature_coverage.is_finite()
    }));
    assert!(output.join("Cache/database.db").is_file());
    let evaluations = events
        .iter()
        .filter(|event| {
            event.stage == SfmTaskStage::KeyframeSelection
                && event.operation == SfmTaskOperation::EvaluateKeyframePair
        })
        .collect::<Vec<_>>();
    assert_eq!(evaluations.len(), result.evaluated_pairs);
    assert!(evaluations.iter().all(|event| {
        event.pair.is_some()
            && event.completed.is_some_and(|selected| selected >= 1)
            && event.total == Some(frames.len())
    }));
    assert!(events.iter().any(|event| {
        event.stage == SfmTaskStage::KeyframeSelection
            && event.operation == SfmTaskOperation::Complete
            && event.completed == Some(result.selected_frame_ids.len())
    }));
}

fn write_runtime_frame(directory: &Path, id: u32, name: &str) -> SequenceFrame {
    let path = directory.join(name);
    image::GrayImage::from_pixel(8, 8, image::Luma([128]))
        .save(&path)
        .unwrap();
    SequenceFrame {
        id,
        image_path: path,
        timestamp_us: Some(i64::from(id)),
    }
}

fn write_textured_runtime_frame(
    directory: &Path,
    id: u32,
    name: &str,
    horizontal_shift: u32,
) -> SequenceFrame {
    let width = 320;
    let height = 240;
    let image = image::GrayImage::from_fn(width, height, |x, y| {
        let source_x = (x + horizontal_shift) % width;
        let checker = if ((source_x / 12) + (y / 12)) % 2 == 0 {
            48i32
        } else {
            -48i32
        };
        let hash = source_x
            .wrapping_mul(73_856_093)
            .wrapping_add(y.wrapping_mul(19_349_663));
        let noise = ((hash ^ (hash >> 13)) & 31) as i32 - 15;
        image::Luma([(128 + checker + noise).clamp(0, 255) as u8])
    });
    let path = directory.join(name);
    image.save(&path).unwrap();
    SequenceFrame {
        id,
        image_path: path,
        timestamp_us: Some(i64::from(id)),
    }
}
