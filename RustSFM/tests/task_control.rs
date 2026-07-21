use rustsfm::{
    SfmControlState, SfmTaskContext, SfmTaskControl, SfmTaskEvent, SfmTaskEventKind,
    SfmTaskOperation, SfmTaskStage, SfmTaskStop,
};

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
