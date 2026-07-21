use rustsfm::{
    SfmControlState, SfmTaskContext, SfmTaskControl, SfmTaskEvent, SfmTaskEventKind,
    SfmTaskOperation, SfmTaskStage, SfmTaskStop,
};
use serde::{de::DeserializeOwned, Serialize};
use std::fmt::Debug;

fn assert_wire_round_trips<T>(cases: &[(T, &str)])
where
    T: Debug + DeserializeOwned + PartialEq + Serialize,
{
    for (value, expected) in cases {
        let json = serde_json::to_value(value).unwrap();
        assert_eq!(json, *expected);
        assert_eq!(serde_json::from_value::<T>(json).unwrap(), *value);
    }
}

fn progress_event(sequence: u64, elapsed_ms: u64) -> SfmTaskEvent {
    SfmTaskEvent {
        sequence,
        elapsed_ms,
        stage: SfmTaskStage::FeatureExtraction,
        operation: SfmTaskOperation::ExtractImage,
        kind: SfmTaskEventKind::Progress,
        completed: Some(3),
        total: Some(10),
        registered_images: None,
        sparse_points: None,
        image_id: Some(12),
        pair: None,
        message: None,
        issue: None,
    }
}

#[test]
fn control_prioritizes_cancel_over_pause() {
    let control = SfmTaskControl::new();
    control.request_pause();
    assert_eq!(control.checkpoint(), Err(SfmTaskStop::Paused));
    control.request_cancel();
    assert_eq!(control.checkpoint(), Err(SfmTaskStop::Cancelled));
    assert_eq!(control.state(), SfmControlState::CancelRequested);
}

#[test]
fn default_control_is_running() {
    let control = SfmTaskControl::default();

    assert_eq!(control.state(), SfmControlState::Running);
    assert_eq!(control.checkpoint(), Ok(()));
}

#[test]
fn cloned_control_shares_cancel_and_pause_cannot_downgrade_it() {
    let control = SfmTaskControl::new();
    let cloned = control.clone();

    control.request_cancel();
    cloned.request_pause();

    assert_eq!(control.state(), SfmControlState::CancelRequested);
    assert_eq!(cloned.state(), SfmControlState::CancelRequested);
    assert_eq!(control.checkpoint(), Err(SfmTaskStop::Cancelled));
    assert_eq!(cloned.checkpoint(), Err(SfmTaskStop::Cancelled));
}

#[test]
fn task_stages_round_trip_through_snake_case_json() {
    assert_wire_round_trips(&[
        (SfmTaskStage::FeatureExtraction, "feature_extraction"),
        (SfmTaskStage::FeatureMatching, "feature_matching"),
        (SfmTaskStage::IncrementalMapping, "incremental_mapping"),
        (SfmTaskStage::BundleAdjustment, "bundle_adjustment"),
        (
            SfmTaskStage::FullFrameRegistration,
            "full_frame_registration",
        ),
        (SfmTaskStage::Export, "export"),
    ]);
}

#[test]
fn task_operations_round_trip_through_snake_case_json() {
    assert_wire_round_trips(&[
        (SfmTaskOperation::Begin, "begin"),
        (SfmTaskOperation::ExtractImage, "extract_image"),
        (SfmTaskOperation::MatchPairBatch, "match_pair_batch"),
        (
            SfmTaskOperation::RegisterInitialPair,
            "register_initial_pair",
        ),
        (SfmTaskOperation::RegisterImage, "register_image"),
        (
            SfmTaskOperation::LocalBundleAdjustment,
            "local_bundle_adjustment",
        ),
        (
            SfmTaskOperation::GlobalBundleAdjustment,
            "global_bundle_adjustment",
        ),
        (
            SfmTaskOperation::RegisterFrameAttempt,
            "register_frame_attempt",
        ),
        (SfmTaskOperation::ValidateArtifacts, "validate_artifacts"),
        (SfmTaskOperation::WriteArtifacts, "write_artifacts"),
        (SfmTaskOperation::Complete, "complete"),
    ]);
}

#[test]
fn task_event_kinds_round_trip_through_snake_case_json() {
    assert_wire_round_trips(&[
        (SfmTaskEventKind::Started, "started"),
        (SfmTaskEventKind::Progress, "progress"),
        (SfmTaskEventKind::Warning, "warning"),
        (SfmTaskEventKind::Error, "error"),
        (SfmTaskEventKind::Completed, "completed"),
    ]);
}

#[test]
fn event_progress_is_machine_readable() {
    let event = SfmTaskEvent {
        sequence: 7,
        elapsed_ms: 42,
        stage: SfmTaskStage::FeatureExtraction,
        operation: SfmTaskOperation::ExtractImage,
        kind: SfmTaskEventKind::Progress,
        completed: Some(3),
        total: Some(10),
        registered_images: None,
        sparse_points: None,
        image_id: Some(12),
        pair: None,
        message: None,
        issue: None,
    };
    let json = serde_json::to_value(event).unwrap();
    assert_eq!(json["sequence"], 7);
    assert_eq!(json["stage"], "feature_extraction");
    assert_eq!(json["operation"], "extract_image");
    assert_eq!(json["kind"], "progress");
}

#[test]
fn context_assigns_monotonic_event_metadata() {
    let control = SfmTaskControl::new();
    let mut events = Vec::new();

    {
        let mut sink = |event| events.push(event);
        let mut context = SfmTaskContext::new(&control, &mut sink);
        context.emit(progress_event(41, u64::MAX));
        context.emit(progress_event(7, u64::MAX));
    }

    assert_eq!(events.len(), 2);
    assert_eq!(events[0].sequence, 0);
    assert_eq!(events[1].sequence, 1);
    assert_ne!(events[0].elapsed_ms, u64::MAX);
    assert_ne!(events[1].elapsed_ms, u64::MAX);
    assert!(events[1].elapsed_ms >= events[0].elapsed_ms);
}

#[test]
fn controlled_mapper_entry_point_is_public() {
    let _entry = rustsfm::run_reconstruction_with_task;
}
