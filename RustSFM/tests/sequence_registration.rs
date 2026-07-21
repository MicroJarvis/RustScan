use rustsfm::{
    FrameRegistrationDiagnostic, FrameRegistrationStatus, RegistrationRound, SequenceFrame,
    SequenceRegistrationConfig, SequenceRegistrationError, SequenceRegistrationPlan,
    SequenceRegistrationResult,
};
use serde::{de::DeserializeOwned, Serialize};
use std::fmt::Debug;
use std::path::PathBuf;

fn assert_json_round_trip<T>(value: &T)
where
    T: Debug + DeserializeOwned + PartialEq + Serialize,
{
    let json = serde_json::to_string(value).unwrap();
    assert_eq!(&serde_json::from_str::<T>(&json).unwrap(), value);
}

#[test]
fn temporal_sample_plan_uses_nearest_keyframes_in_deterministic_order() {
    let plan = SequenceRegistrationPlan::build(12, &[0, 3, 6, 9, 11], 2, 4).unwrap();

    assert_eq!(
        plan.attempts_for(4, RegistrationRound::Narrow),
        &[3, 6, 0, 9]
    );
    assert_eq!(
        plan.attempts_for(4, RegistrationRound::Wide),
        &[3, 6, 0, 9, 11]
    );
    assert_eq!(plan.pending_frames(), &[1, 2, 4, 5, 7, 8, 10]);
}

#[test]
fn temporal_plan_json_round_trip_rebuilds_equivalent_attempts() {
    let plan = SequenceRegistrationPlan::build(12, &[0, 3, 6, 9, 11], 2, 4).unwrap();

    let json = serde_json::to_string(&plan).unwrap();
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(value.as_object().unwrap().len(), 4);
    assert!(value.get("pending").is_none());
    assert!(value.get("narrow_support").is_none());
    assert!(value.get("wide_support").is_none());
    let restored: SequenceRegistrationPlan = serde_json::from_str(&json).unwrap();

    assert_eq!(restored, plan);
    assert_eq!(
        restored.attempts_for(4, RegistrationRound::Narrow),
        &[3, 6, 0, 9]
    );
    assert_eq!(
        restored.attempts_for(4, RegistrationRound::Wide),
        &[3, 6, 0, 9, 11]
    );
    assert_eq!(restored.pending_frames(), &[1, 2, 4, 5, 7, 8, 10]);
}

#[test]
fn temporal_plan_json_rejects_invalid_keyframe_inputs() {
    let invalid = serde_json::json!({
        "frame_count": 4,
        "keyframes": [0, 2, 1],
        "narrow_neighbors_each_side": 2,
        "wide_neighbors_each_side": 4,
    });

    assert!(serde_json::from_value::<SequenceRegistrationPlan>(invalid).is_err());
}

#[test]
fn temporal_rounds_limit_each_side_then_sort_by_distance_and_frame_id() {
    let first = SequenceRegistrationPlan::build(10, &[0, 2, 7, 9], 1, 3).unwrap();
    let second = SequenceRegistrationPlan::build(10, &[0, 2, 7, 9], 1, 3).unwrap();

    assert_eq!(first.attempts_for(5, RegistrationRound::Narrow), &[7, 2]);
    assert_eq!(
        first.attempts_for(5, RegistrationRound::Wide),
        &[7, 2, 9, 0]
    );
    assert_eq!(first, second);
}

#[test]
fn temporal_plan_support_lists_only_contain_keyframes() {
    let plan = SequenceRegistrationPlan::build(8, &[0, 4, 7], 2, 4).unwrap();

    for frame in plan.pending_frames() {
        for round in [RegistrationRound::Narrow, RegistrationRound::Wide] {
            assert!(plan
                .attempts_for(*frame, round)
                .iter()
                .all(|support| [0, 4, 7].contains(support)));
        }
    }
}

#[test]
fn invalid_temporal_plan_rejects_empty_sequences_and_keyframes() {
    assert!(SequenceRegistrationPlan::build(0, &[], 2, 4).is_err());
    assert!(SequenceRegistrationPlan::build(4, &[], 2, 4).is_err());
}

#[test]
fn invalid_temporal_plan_rejects_duplicate_keyframes() {
    assert!(SequenceRegistrationPlan::build(4, &[0, 1, 1], 2, 4).is_err());
}

#[test]
fn invalid_temporal_plan_rejects_out_of_range_keyframes() {
    assert!(SequenceRegistrationPlan::build(4, &[0, 4], 2, 4).is_err());
}

#[test]
fn invalid_temporal_plan_rejects_unsorted_keyframes() {
    assert!(SequenceRegistrationPlan::build(4, &[0, 2, 1], 2, 4).is_err());
}

#[test]
fn registration_status_identifies_pose_coverage() {
    assert!(FrameRegistrationStatus::Keyframe.is_registered());
    assert!(FrameRegistrationStatus::Registered.is_registered());
    assert!(!FrameRegistrationStatus::Unresolved.is_registered());
    assert!(!FrameRegistrationStatus::Excluded.is_registered());
}

#[test]
fn sequence_config_defaults_match_registration_policy() {
    let config = SequenceRegistrationConfig::default();

    assert_eq!(config.narrow_neighbors_each_side, 2);
    assert_eq!(config.wide_neighbors_each_side, 4);
    assert_eq!(config.min_inliers, 24);
    assert_eq!(config.min_inlier_ratio, 0.20);
    assert_eq!(config.max_reprojection_error, 4.0);
    assert!(config.use_gpu_pnp);
    assert_json_round_trip(&config);
}

#[test]
fn sequence_frame_and_round_round_trip_through_json() {
    let frame = SequenceFrame {
        id: 42,
        image_path: PathBuf::from("images/000042.jpg"),
        timestamp_us: Some(1_234_567),
    };

    assert_json_round_trip(&frame);
    assert_json_round_trip(&RegistrationRound::Narrow);
    assert_json_round_trip(&RegistrationRound::Wide);
    assert_eq!(
        serde_json::to_value(RegistrationRound::Narrow).unwrap(),
        "narrow"
    );
}

#[test]
fn diagnostic_update_preserves_attempt_state_and_metrics_in_json() {
    let mut diagnostic = FrameRegistrationDiagnostic::new(4, FrameRegistrationStatus::Unresolved);

    diagnostic.record_attempt(
        FrameRegistrationStatus::Unresolved,
        vec![3, 6],
        18,
        0.36,
        Some(3.25),
        Some("narrow support was insufficient".to_owned()),
    );
    diagnostic.record_attempt(
        FrameRegistrationStatus::Registered,
        vec![3, 6, 0, 9, 11],
        31,
        0.62,
        Some(1.5),
        Some("registered in wide round".to_owned()),
    );

    assert_eq!(diagnostic.frame_id, 4);
    assert_eq!(diagnostic.status, FrameRegistrationStatus::Registered);
    assert_eq!(diagnostic.attempts, 2);
    assert_eq!(diagnostic.support_frame_ids, vec![3, 6, 0, 9, 11]);
    assert_eq!(diagnostic.inlier_count, 31);
    assert_eq!(diagnostic.inlier_ratio, 0.62);
    assert_eq!(diagnostic.mean_reprojection_error, Some(1.5));
    assert_eq!(
        diagnostic.message.as_deref(),
        Some("registered in wide round")
    );
    assert_json_round_trip(&diagnostic);
}

#[test]
fn sequence_result_round_trip_preserves_diagnostics() {
    let result = SequenceRegistrationResult {
        imported_frames: 2,
        registered_frames: 1,
        diagnostics: vec![
            FrameRegistrationDiagnostic::new(0, FrameRegistrationStatus::Keyframe),
            FrameRegistrationDiagnostic {
                frame_id: 1,
                status: FrameRegistrationStatus::Unresolved,
                attempts: 2,
                support_frame_ids: vec![0],
                inlier_count: 11,
                inlier_ratio: 0.15,
                mean_reprojection_error: Some(4.5),
                message: Some("inlier threshold not met".to_owned()),
            },
        ],
        sparse_model: PathBuf::from("sparse/0"),
    };

    assert_json_round_trip(&result);
}

#[test]
fn complete_coverage_accepts_keyframes_and_registered_frames() {
    let result = SequenceRegistrationResult {
        imported_frames: 2,
        registered_frames: 2,
        diagnostics: vec![
            FrameRegistrationDiagnostic::new(0, FrameRegistrationStatus::Keyframe),
            FrameRegistrationDiagnostic::new(1, FrameRegistrationStatus::Registered),
        ],
        sparse_model: PathBuf::from("sparse/0"),
    };

    assert!(result.has_complete_coverage());
    assert_eq!(result.validate_complete_coverage(), Ok(()));
}

#[test]
fn incomplete_frame_count_returns_an_explicit_error() {
    let result = SequenceRegistrationResult {
        imported_frames: 3,
        registered_frames: 2,
        diagnostics: vec![
            FrameRegistrationDiagnostic::new(0, FrameRegistrationStatus::Keyframe),
            FrameRegistrationDiagnostic::new(1, FrameRegistrationStatus::Registered),
            FrameRegistrationDiagnostic::new(2, FrameRegistrationStatus::Unresolved),
        ],
        sparse_model: PathBuf::from("sparse/0"),
    };

    assert!(!result.has_complete_coverage());
    let error = result.validate_complete_coverage().unwrap_err();
    assert!(error.to_string().contains("2 of 3"));
    assert!(matches!(
        error,
        SequenceRegistrationError::IncompleteCoverage {
            imported_frames: 3,
            registered_frames: 2,
            unresolved_frame_ids,
        } if unresolved_frame_ids == vec![2]
    ));
}

#[test]
fn unresolved_diagnostic_fails_coverage_even_when_counts_match() {
    let result = SequenceRegistrationResult {
        imported_frames: 3,
        registered_frames: 3,
        diagnostics: vec![
            FrameRegistrationDiagnostic::new(0, FrameRegistrationStatus::Keyframe),
            FrameRegistrationDiagnostic::new(1, FrameRegistrationStatus::Registered),
            FrameRegistrationDiagnostic::new(2, FrameRegistrationStatus::Unresolved),
        ],
        sparse_model: PathBuf::from("sparse/0"),
    };

    assert!(!result.has_complete_coverage());
    let error = result.validate_complete_coverage().unwrap_err();
    assert!(error.to_string().contains("frame 2"));
    assert!(matches!(
        error,
        SequenceRegistrationError::RegistrationStatusCountMismatch {
            registered_frames: 3,
            diagnostic_registered_frames: 2,
            unresolved_frame_ids,
        } if unresolved_frame_ids == vec![2]
    ));
}

#[test]
fn complete_counts_with_missing_diagnostic_fail_coverage() {
    let result = SequenceRegistrationResult {
        imported_frames: 3,
        registered_frames: 3,
        diagnostics: vec![
            FrameRegistrationDiagnostic::new(0, FrameRegistrationStatus::Keyframe),
            FrameRegistrationDiagnostic::new(1, FrameRegistrationStatus::Registered),
        ],
        sparse_model: PathBuf::from("sparse/0"),
    };

    assert!(!result.has_complete_coverage());
    assert!(matches!(
        result.validate_complete_coverage(),
        Err(SequenceRegistrationError::InvalidDiagnostics {
            imported_frames: 3,
            diagnostic_count: 2,
            missing_frame_ids,
            duplicate_frame_ids,
            unexpected_frame_ids,
        }) if missing_frame_ids == vec![2]
            && duplicate_frame_ids.is_empty()
            && unexpected_frame_ids.is_empty()
    ));
}

#[test]
fn complete_counts_with_duplicate_diagnostic_fail_coverage() {
    let result = SequenceRegistrationResult {
        imported_frames: 3,
        registered_frames: 3,
        diagnostics: vec![
            FrameRegistrationDiagnostic::new(0, FrameRegistrationStatus::Keyframe),
            FrameRegistrationDiagnostic::new(1, FrameRegistrationStatus::Registered),
            FrameRegistrationDiagnostic::new(1, FrameRegistrationStatus::Registered),
        ],
        sparse_model: PathBuf::from("sparse/0"),
    };

    assert!(!result.has_complete_coverage());
    assert!(matches!(
        result.validate_complete_coverage(),
        Err(SequenceRegistrationError::InvalidDiagnostics {
            imported_frames: 3,
            diagnostic_count: 3,
            missing_frame_ids,
            duplicate_frame_ids,
            unexpected_frame_ids,
        }) if missing_frame_ids == vec![2]
            && duplicate_frame_ids == vec![1]
            && unexpected_frame_ids.is_empty()
    ));
}
