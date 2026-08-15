use rustsfm::sift::SiftFeatures;
use rustsfm::{
    extract_features_to_database, extract_features_to_database_with_extractor,
    extract_features_to_database_with_extractor_and_task, extract_features_to_database_with_task,
    match_features_to_database, match_features_to_database_with_task,
    register_remaining_sequence_frames, run_adaptive_keyframe_selection,
    run_keyframe_reconstruction, run_reconstruction, run_reconstruction_with_callbacks,
    run_reconstruction_with_task, run_sequence_registration, AdaptiveKeyframeSelectionConfig,
    AdaptiveKeyframeSelectionResult, ExtractFeaturesReport, KeyframeReconstructionResult,
    MapperConfig, MatchFeaturesOptions, MatchFeaturesReport, PipelineCallbackSink,
    ReconstructionSummary, SequenceFrame, SequenceRegistrationConfig, SequenceRegistrationResult,
    SfmControlState, SfmTaskContext, SfmTaskControl, SfmTaskEvent, SfmTaskEventKind,
    SfmTaskOperation, SfmTaskStage, SfmTaskStop, SiftExtractionOptions, SiftFeatureExtractor,
};
use serde::{de::DeserializeOwned, Serialize};
use std::fmt::Debug;
use std::path::Path;
use std::rc::Rc;

struct BorrowedPublicApiExtractor<'a> {
    _not_send_or_sync: &'a Rc<()>,
}

impl SiftFeatureExtractor for BorrowedPublicApiExtractor<'_> {
    fn backend_name(&self) -> &'static str {
        "public-api-test"
    }

    fn extract_grayscale(
        &self,
        _gray: &[u8],
        _width: u32,
        _height: u32,
        _options: &SiftExtractionOptions,
    ) -> anyhow::Result<SiftFeatures> {
        unreachable!("compile-time API test does not extract features")
    }
}

type LegacyBorrowedExtractorApi<'a> =
    for<'database, 'images, 'options, 'extractor> fn(
        &'database Path,
        &'images Path,
        &'options SiftExtractionOptions,
        &'extractor BorrowedPublicApiExtractor<'a>,
    ) -> anyhow::Result<ExtractFeaturesReport>;
type ControlledBorrowedExtractorApi<'a> =
    for<'database, 'images, 'options, 'extractor, 'context, 'task> fn(
        &'database Path,
        &'images Path,
        &'options SiftExtractionOptions,
        &'extractor BorrowedPublicApiExtractor<'a>,
        &'context mut SfmTaskContext<'task>,
    ) -> anyhow::Result<
        ExtractFeaturesReport,
    >;

fn assert_borrowed_extractor_apis<'a>(
    database_path: &Path,
    images_dir: &Path,
    options: &SiftExtractionOptions,
    extractor: &BorrowedPublicApiExtractor<'a>,
    task: &mut SfmTaskContext<'_>,
) {
    let _: LegacyBorrowedExtractorApi<'a> =
        extract_features_to_database_with_extractor::<BorrowedPublicApiExtractor<'a>>;
    let _: ControlledBorrowedExtractorApi<'a> =
        extract_features_to_database_with_extractor_and_task::<BorrowedPublicApiExtractor<'a>>;

    let _ =
        extract_features_to_database_with_extractor(database_path, images_dir, options, extractor);
    let _ = extract_features_to_database_with_extractor_and_task(
        database_path,
        images_dir,
        options,
        extractor,
        task,
    );
}

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
        (SfmTaskStage::KeyframeSelection, "keyframe_selection"),
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
            SfmTaskOperation::EvaluateKeyframePair,
            "evaluate_keyframe_pair",
        ),
        (SfmTaskOperation::SelectKeyframe, "select_keyframe"),
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

#[test]
fn legacy_and_controlled_public_entry_points_keep_their_signatures() {
    type LegacyExtractionApi =
        for<'database, 'images, 'options> fn(
            &'database Path,
            &'images Path,
            &'options SiftExtractionOptions,
        ) -> anyhow::Result<ExtractFeaturesReport>;
    type LegacyMapperApi = fn(&MapperConfig) -> anyhow::Result<ReconstructionSummary>;
    type LegacyCallbackMapperApi = for<'config, 'sink> fn(
        &'config MapperConfig,
        Option<&'sink mut dyn PipelineCallbackSink>,
    )
        -> anyhow::Result<ReconstructionSummary>;
    type ControlledExtractionApi =
        for<'database, 'images, 'options, 'context, 'task> fn(
            &'database Path,
            &'images Path,
            &'options SiftExtractionOptions,
            &'context mut SfmTaskContext<'task>,
        ) -> anyhow::Result<
            ExtractFeaturesReport,
        >;
    type LegacyMatchingApi = for<'database, 'options> fn(
        &'database Path,
        &'options MatchFeaturesOptions,
    )
        -> anyhow::Result<MatchFeaturesReport>;
    type ControlledMatchingApi =
        for<'database, 'options, 'context, 'task> fn(
            &'database Path,
            &'options MatchFeaturesOptions,
            &'context mut SfmTaskContext<'task>,
        )
            -> anyhow::Result<MatchFeaturesReport>;
    type ControlledMapperApi =
        for<'config, 'context, 'task> fn(
            &'config MapperConfig,
            &'context mut SfmTaskContext<'task>,
        ) -> anyhow::Result<ReconstructionSummary>;
    type ControlledKeyframeApi =
        for<'frames, 'keyframes, 'mapper, 'output, 'context, 'task> fn(
            &'frames [SequenceFrame],
            &'keyframes [u32],
            &'mapper MapperConfig,
            &'output Path,
            &'context mut SfmTaskContext<'task>,
        ) -> anyhow::Result<
            KeyframeReconstructionResult,
        >;
    type ControlledAdaptiveSelectionApi =
        for<'frames, 'selection, 'mapper, 'output, 'context, 'task> fn(
            &'frames [SequenceFrame],
            &'selection AdaptiveKeyframeSelectionConfig,
            &'mapper MapperConfig,
            &'output Path,
            &'context mut SfmTaskContext<'task>,
        ) -> anyhow::Result<
            AdaptiveKeyframeSelectionResult,
        >;
    type ControlledRemainingSequenceApi =
        for<'frames, 'keyframes, 'result, 'mapper, 'config, 'output, 'context, 'task> fn(
            &'frames [SequenceFrame],
            &'keyframes [u32],
            &'result KeyframeReconstructionResult,
            &'mapper MapperConfig,
            &'config SequenceRegistrationConfig,
            &'output Path,
            &'context mut SfmTaskContext<'task>,
        ) -> anyhow::Result<SequenceRegistrationResult>;
    type ControlledSequenceApi =
        for<'frames, 'keyframes, 'mapper, 'config, 'output, 'context, 'task> fn(
            &'frames [SequenceFrame],
            &'keyframes [u32],
            &'mapper MapperConfig,
            &'config SequenceRegistrationConfig,
            &'output Path,
            &'context mut SfmTaskContext<'task>,
        )
            -> anyhow::Result<
            SequenceRegistrationResult,
        >;

    let _: LegacyExtractionApi = extract_features_to_database;
    let _: LegacyMatchingApi = match_features_to_database;
    let _compile_only = assert_borrowed_extractor_apis;
    let _: LegacyCallbackMapperApi = run_reconstruction_with_callbacks;
    let _: LegacyMapperApi = run_reconstruction;
    let _: ControlledExtractionApi = extract_features_to_database_with_task;
    let _: ControlledMatchingApi = match_features_to_database_with_task;
    let _: ControlledMapperApi = run_reconstruction_with_task;
    let _: ControlledAdaptiveSelectionApi = run_adaptive_keyframe_selection;
    let _: ControlledKeyframeApi = run_keyframe_reconstruction;
    let _: ControlledRemainingSequenceApi = register_remaining_sequence_frames;
    let _: ControlledSequenceApi = run_sequence_registration;
}
