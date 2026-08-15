use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};

use image::GenericImageView;
use rust_viewer::loader::load_colmap_training_dataset;
use rust_viewer::media::{
    import_image_sequence, import_video, select_keyframes, DecodedVideoFrame,
    ImageSequenceImportRequest, ImportedFrame, KeyframeSelectionConfig, MediaEventSink,
    MediaImportError, MediaImportEvent, VideoDecoder, VideoMetadata,
};
use rust_viewer::pipeline::{
    ArtifactValidation, ImportWorker, PendingArtifact, PipelineCommand, PipelineCoordinator,
    PipelineWorkers, PnpWorker, SfmWorker, StageRequest, TrainingWorker, WorkerControl,
    WorkerEventSink, WorkerOutcome,
};
use rust_viewer::project::{
    ProjectCreateRequest, ProjectStage, ProjectStore, ProjectStoreWarning, SourceKind,
    SourceOwnership, SourceSpec, StageState, SuggestedAction,
};
use rustgs::ColmapConfig;

#[derive(Default)]
struct RecordingSink {
    events: Vec<MediaImportEvent>,
}

struct MutatingSink {
    events: Vec<MediaImportEvent>,
    mutate_on_first_frame: PathBuf,
}

struct RemovingStagedPayloadSink {
    staged_payload: PathBuf,
}

impl MediaEventSink for MutatingSink {
    fn on_media_event(&mut self, event: MediaImportEvent) {
        if matches!(event, MediaImportEvent::FrameCommitted { frame_id: 0, .. }) {
            image::RgbaImage::from_pixel(1, 1, image::Rgba([200, 100, 50, 255]))
                .save(&self.mutate_on_first_frame)
                .unwrap();
        }
        self.events.push(event);
    }
}

impl MediaEventSink for RemovingStagedPayloadSink {
    fn on_media_event(&mut self, event: MediaImportEvent) {
        if matches!(event, MediaImportEvent::FrameCommitted { frame_id: 0, .. }) {
            fs::remove_file(&self.staged_payload).unwrap();
        }
    }
}

#[test]
fn image_sequence_requires_at_least_two_sources_without_committed_metadata() {
    let temp = tempfile::tempdir().unwrap();
    let sources = temp.path().join("source-images");
    fs::create_dir(&sources).unwrap();
    let only_image = write_fixture_png(&sources, "frame1.png");
    let project = temp.path().join("Incomplete.rustscanproject");
    let mut store = ProjectStore::create(
        &project,
        ProjectCreateRequest::new("Incomplete", SourceSpec::managed_images("pending-import")),
    )
    .unwrap();
    let mut sink = RecordingSink::default();

    let error = import_image_sequence(
        &ImageSequenceImportRequest::managed(vec![only_image]),
        &mut store,
        &mut sink,
    )
    .unwrap_err();

    assert!(matches!(error, MediaImportError::InvalidSource(_)));
    assert_eq!(
        store
            .manifest()
            .try_stage(ProjectStage::Import)
            .unwrap()
            .state(),
        StageState::Failed
    );
    assert!(store
        .manifest()
        .try_stage(ProjectStage::Import)
        .unwrap()
        .error()
        .unwrap()
        .suggested_actions
        .contains(&SuggestedAction::Retry));
    assert!(!project.join("Artifacts/import").exists());
}

#[test]
fn image_sequence_missing_preflight_source_persists_a_retryable_import_failure() {
    let temp = tempfile::tempdir().unwrap();
    let sources = temp.path().join("source-images");
    fs::create_dir(&sources).unwrap();
    let first = write_fixture_png(&sources, "frame1.png");
    let missing = sources.join("frame2.png");
    let project = temp.path().join("Missing.rustscanproject");
    let mut store = ProjectStore::create(
        &project,
        ProjectCreateRequest::new("Missing", SourceSpec::managed_images("pending-import")),
    )
    .unwrap();

    let error = import_image_sequence(
        &ImageSequenceImportRequest::managed(vec![first, missing]),
        &mut store,
        &mut RecordingSink::default(),
    )
    .unwrap_err();

    assert!(matches!(error, MediaImportError::InvalidSource(_)));
    let stage = store.manifest().try_stage(ProjectStage::Import).unwrap();
    assert_eq!(stage.state(), StageState::Failed);
    assert!(stage.artifacts().is_empty());
    let failure = stage.error().unwrap();
    assert!(failure.retryable);
    assert!(failure.suggested_actions.contains(&SuggestedAction::Retry));
    assert!(failure
        .suggested_actions
        .contains(&SuggestedAction::RevealSource));
    assert!(!project.join("Artifacts/import").exists());
}

#[test]
fn image_sequence_missing_staged_payload_cannot_commit_import_metadata() {
    let temp = tempfile::tempdir().unwrap();
    let sources = temp.path().join("source-images");
    fs::create_dir(&sources).unwrap();
    let first = write_fixture_png(&sources, "frame1.png");
    let second = write_fixture_png(&sources, "frame2.png");
    let project = temp.path().join("MissingPayload.rustscanproject");
    let mut store = ProjectStore::create(
        &project,
        ProjectCreateRequest::new(
            "Missing payload",
            SourceSpec::managed_images("pending-import"),
        ),
    )
    .unwrap();
    let staged_payload = project.join("Cache/.staging/import-1/Cache/frames/00000000.png");
    let mut sink = RemovingStagedPayloadSink { staged_payload };

    let error = import_image_sequence(
        &ImageSequenceImportRequest::managed(vec![first, second]),
        &mut store,
        &mut sink,
    )
    .unwrap_err();

    assert!(matches!(error, MediaImportError::Project(_)));
    let stage = store.manifest().try_stage(ProjectStage::Import).unwrap();
    assert_eq!(stage.state(), StageState::Failed);
    assert!(stage.artifacts().is_empty());
    assert!(!project.join("Artifacts/import").exists());
}

#[test]
fn image_sequence_stale_import_preflight_failure_preserves_the_prior_attempt() {
    let temp = tempfile::tempdir().unwrap();
    let sources = temp.path().join("source-images");
    fs::create_dir(&sources).unwrap();
    let first = write_fixture_png(&sources, "frame1.png");
    let second = write_fixture_png(&sources, "frame2.png");
    let project = temp.path().join("Stale.rustscanproject");
    let mut store = ProjectStore::create(
        &project,
        ProjectCreateRequest::new("Stale", SourceSpec::managed_images("pending-import")),
    )
    .unwrap();

    import_image_sequence(
        &ImageSequenceImportRequest::managed(vec![first.clone(), second]),
        &mut store,
        &mut RecordingSink::default(),
    )
    .unwrap();
    let prior = store
        .manifest()
        .try_stage(ProjectStage::Import)
        .unwrap()
        .clone();
    let mut config = store.manifest().import_config.clone();
    config.video_keyframes_per_second += 1.0;
    store.update_import_config(config).unwrap();
    assert_eq!(
        store
            .manifest()
            .try_stage(ProjectStage::Import)
            .unwrap()
            .state(),
        StageState::Stale
    );

    let error = import_image_sequence(
        &ImageSequenceImportRequest::managed(vec![first, sources.join("missing.png")]),
        &mut store,
        &mut RecordingSink::default(),
    )
    .unwrap_err();

    assert!(matches!(error, MediaImportError::InvalidSource(_)));
    let stage = store.manifest().try_stage(ProjectStage::Import).unwrap();
    assert_eq!(stage.state(), StageState::Failed);
    assert_eq!(stage.attempt(), prior.attempt());
    assert_eq!(stage.artifacts(), prior.artifacts());
    assert!(stage.error().unwrap().retryable);
}

#[test]
fn image_sequence_decode_failure_leaves_no_committed_import_artifacts() {
    let temp = tempfile::tempdir().unwrap();
    let sources = temp.path().join("source-images");
    fs::create_dir(&sources).unwrap();
    let first = write_fixture_png(&sources, "frame1.png");
    let broken = sources.join("frame2.png");
    fs::write(&broken, b"not a PNG").unwrap();
    let project = temp.path().join("Broken.rustscanproject");
    let mut store = ProjectStore::create(
        &project,
        ProjectCreateRequest::new("Broken", SourceSpec::managed_images("pending-import")),
    )
    .unwrap();
    let mut sink = RecordingSink::default();

    let error = import_image_sequence(
        &ImageSequenceImportRequest::managed(vec![first, broken]),
        &mut store,
        &mut sink,
    )
    .unwrap_err();

    assert!(matches!(error, MediaImportError::Decode(_)));
    assert_eq!(
        store
            .manifest()
            .try_stage(ProjectStage::Import)
            .unwrap()
            .state(),
        StageState::Failed
    );
    assert!(!project.join("Artifacts/import").exists());
    assert!(!project.join("Cache/.staging/import-1").exists());
    assert!(fs::read_dir(project.join("Logs/recovery"))
        .unwrap()
        .any(|entry| entry.unwrap().path().is_dir()));
}

#[test]
fn image_sequence_discards_an_attempt_when_source_identity_changes_during_import() {
    let temp = tempfile::tempdir().unwrap();
    let sources = temp.path().join("source-images");
    fs::create_dir(&sources).unwrap();
    let first = write_fixture_png(&sources, "frame1.png");
    let second = write_fixture_png(&sources, "frame2.png");
    let project = temp.path().join("Changing.rustscanproject");
    let mut store = ProjectStore::create(
        &project,
        ProjectCreateRequest::new("Changing", SourceSpec::managed_images("pending-import")),
    )
    .unwrap();
    let mut sink = MutatingSink {
        events: Vec::new(),
        mutate_on_first_frame: second.clone(),
    };

    let error = import_image_sequence(
        &ImageSequenceImportRequest::managed(vec![first, second]),
        &mut store,
        &mut sink,
    )
    .unwrap_err();

    assert!(matches!(
        error,
        MediaImportError::SourceChangedDuringImport { .. }
    ));
    assert!(!project.join("Artifacts/import").exists());
    assert!(!project.join("Cache/.staging/import-1").exists());
    assert!(fs::read_dir(project.join("Logs/recovery"))
        .unwrap()
        .any(|entry| entry.unwrap().path().is_dir()));
}

#[test]
fn image_sequence_invalid_reimport_preserves_the_succeeded_import() {
    let temp = tempfile::tempdir().unwrap();
    let sources = temp.path().join("source-images");
    fs::create_dir(&sources).unwrap();
    let first = write_fixture_png(&sources, "frame1.png");
    let second = write_fixture_png(&sources, "frame2.png");
    let project = temp.path().join("Succeeded.rustscanproject");
    let mut store = ProjectStore::create(
        &project,
        ProjectCreateRequest::new("Succeeded", SourceSpec::managed_images("pending-import")),
    )
    .unwrap();

    import_image_sequence(
        &ImageSequenceImportRequest::managed(vec![first.clone(), second]),
        &mut store,
        &mut RecordingSink::default(),
    )
    .unwrap();
    let prior = store
        .manifest()
        .try_stage(ProjectStage::Import)
        .unwrap()
        .clone();

    let error = import_image_sequence(
        &ImageSequenceImportRequest::managed(vec![first, sources.join("missing.png")]),
        &mut store,
        &mut RecordingSink::default(),
    )
    .unwrap_err();

    assert!(matches!(error, MediaImportError::InvalidSource(_)));
    assert_eq!(
        store.manifest().try_stage(ProjectStage::Import).unwrap(),
        &prior
    );
    assert!(!project.join("Cache/.staging/import-2").exists());
}

#[test]
fn missing_or_changed_referenced_source_opens_with_a_recoverable_warning_and_can_be_reimported() {
    let temp = tempfile::tempdir().unwrap();
    let sources = temp.path().join("source-images");
    fs::create_dir(&sources).unwrap();
    let first = write_fixture_png(&sources, "frame1.png");
    let second = write_fixture_png(&sources, "frame2.png");
    let project = temp.path().join("Referenced.rustscanproject");
    let mut store = ProjectStore::create(
        &project,
        ProjectCreateRequest::new("Referenced", SourceSpec::managed_images("pending-import")),
    )
    .unwrap();
    let mut sink = RecordingSink::default();

    let result = import_image_sequence(
        &ImageSequenceImportRequest::referenced(vec![first.clone(), second.clone()]),
        &mut store,
        &mut sink,
    )
    .unwrap();

    assert_eq!(
        store.manifest().source.ownership,
        SourceOwnership::Referenced
    );
    assert!(store.manifest().source.bookmark.is_some());
    let source_json: serde_json::Value =
        serde_json::from_slice(&fs::read(project.join(&result.source_metadata)).unwrap()).unwrap();
    assert_eq!(source_json["ownership"], "referenced");
    assert!(source_json["frames"]
        .as_array()
        .unwrap()
        .iter()
        .all(|frame| frame["managed_copy"].is_null()));
    assert!(!project
        .join("Artifacts/import/attempt-00000001/Sources/managed")
        .exists());
    drop(store);

    let relocated = temp.path().join("relocated");
    fs::create_dir(&relocated).unwrap();
    let relocated_second = write_fixture_png(&relocated, "frame2.png");
    fs::remove_file(&second).unwrap();
    let mut reopened = ProjectStore::open(&project).unwrap();
    assert!(matches!(
        reopened.warnings(),
        [ProjectStoreWarning::ReferencedSourceUnavailable { .. }]
    ));

    import_image_sequence(
        &ImageSequenceImportRequest::referenced(vec![first, relocated_second]),
        &mut reopened,
        &mut RecordingSink::default(),
    )
    .unwrap();
    drop(reopened);
    let recovered = ProjectStore::open(&project).unwrap();
    assert!(!recovered.warnings().iter().any(|warning| matches!(
        warning,
        ProjectStoreWarning::ReferencedSourceUnavailable { .. }
            | ProjectStoreWarning::ReferencedSourceChanged { .. }
    )));
}

#[test]
fn changed_referenced_source_opens_with_an_identity_warning() {
    let temp = tempfile::tempdir().unwrap();
    let sources = temp.path().join("source-images");
    fs::create_dir(&sources).unwrap();
    let first = write_fixture_png(&sources, "frame1.png");
    let second = write_fixture_png(&sources, "frame2.png");
    let project = temp.path().join("Changed.rustscanproject");
    let mut store = ProjectStore::create(
        &project,
        ProjectCreateRequest::new("Changed", SourceSpec::managed_images("pending-import")),
    )
    .unwrap();

    import_image_sequence(
        &ImageSequenceImportRequest::referenced(vec![first, second.clone()]),
        &mut store,
        &mut RecordingSink::default(),
    )
    .unwrap();
    drop(store);
    image::RgbaImage::from_pixel(1, 1, image::Rgba([201, 100, 50, 255]))
        .save(&second)
        .unwrap();

    let reopened = ProjectStore::open(&project).unwrap();
    assert!(matches!(
        reopened.warnings(),
        [ProjectStoreWarning::ReferencedSourceChanged { .. }]
    ));
}

#[test]
fn invalid_referenced_bookmark_opens_with_a_recoverable_source_warning() {
    let temp = tempfile::tempdir().unwrap();
    let project = temp.path().join("InvalidBookmark.rustscanproject");
    let mut store = ProjectStore::create(
        &project,
        ProjectCreateRequest::new(
            "Invalid bookmark",
            SourceSpec::managed_images("pending-import"),
        ),
    )
    .unwrap();
    store
        .update_source(SourceSpec {
            kind: SourceKind::ImageSequence,
            ownership: SourceOwnership::Referenced,
            identity: "fixture-identity".to_owned(),
            display_paths: Vec::new(),
            bookmark: Some(b"not a bookmark".to_vec()),
        })
        .unwrap();
    drop(store);

    let reopened = ProjectStore::open(&project).unwrap();

    assert!(matches!(
        reopened.warnings(),
        [ProjectStoreWarning::ReferencedSourceUnavailable { detail, .. }]
            if detail.contains("bookmark")
    ));
}

impl MediaEventSink for RecordingSink {
    fn on_media_event(&mut self, event: MediaImportEvent) {
        self.events.push(event);
    }
}

fn write_fixture_png(directory: &Path, name: &str) -> PathBuf {
    let path = directory.join(name);
    image::RgbaImage::from_pixel(1, 1, image::Rgba([12, 34, 56, 255]))
        .save(&path)
        .unwrap();
    path
}

#[test]
fn image_project_full_frame_result_is_a_loadable_colmap_dataset() {
    let fixture = completed_image_project_fixture();

    let loaded = load_colmap_training_dataset(&fixture.colmap_root, &ColmapConfig::default())
        .expect("committed FullFramePnp artifact should load without a GPU adapter");

    assert_eq!(loaded.summary.frame_count, 2);
}

struct CompletedImageProjectFixture {
    _temporary: tempfile::TempDir,
    colmap_root: PathBuf,
}

fn completed_image_project_fixture() -> CompletedImageProjectFixture {
    let temporary = tempfile::tempdir().unwrap();
    let sources = temporary.path().join("source-images");
    fs::create_dir(&sources).unwrap();
    let project = temporary.path().join("Completed.rustscanproject");
    let mut store = ProjectStore::create(
        &project,
        ProjectCreateRequest::new("Completed", SourceSpec::managed_images("pending-import")),
    )
    .unwrap();
    import_image_sequence(
        &ImageSequenceImportRequest::managed(vec![
            write_fixture_png(&sources, "frame1.png"),
            write_fixture_png(&sources, "frame2.png"),
        ]),
        &mut store,
        &mut RecordingSink::default(),
    )
    .unwrap();

    let workers = PipelineWorkers::new(
        UnexpectedProjectWorker,
        FixtureSfmWorker,
        FixturePnpWorker,
        UnexpectedProjectWorker,
    );
    let mut pipeline = PipelineCoordinator::new(store, workers).unwrap();
    pipeline
        .send(PipelineCommand::StartThrough {
            stage: ProjectStage::FullFramePnp,
        })
        .unwrap();
    pipeline.drive_until_idle().unwrap();
    assert_eq!(
        pipeline
            .store()
            .manifest()
            .try_stage(ProjectStage::FullFramePnp)
            .unwrap()
            .state(),
        StageState::Succeeded
    );

    CompletedImageProjectFixture {
        colmap_root: pipeline
            .store()
            .root()
            .join("Artifacts/full_frame_pnp/attempt-00000001/colmap"),
        _temporary: temporary,
    }
}

struct UnexpectedProjectWorker;

impl ImportWorker for UnexpectedProjectWorker {
    fn run(
        &self,
        _request: StageRequest,
        _control: WorkerControl,
        _events: WorkerEventSink,
    ) -> WorkerOutcome {
        panic!("the imported project must not rerun its import stage")
    }
}

impl TrainingWorker for UnexpectedProjectWorker {
    fn run(
        &self,
        _request: StageRequest,
        _control: WorkerControl,
        _events: WorkerEventSink,
    ) -> WorkerOutcome {
        panic!("reconstruction must stop before the training stage")
    }
}

struct FixtureSfmWorker;

impl SfmWorker for FixtureSfmWorker {
    fn run(
        &self,
        _request: StageRequest,
        _control: WorkerControl,
        _events: WorkerEventSink,
    ) -> WorkerOutcome {
        WorkerOutcome::Succeeded(vec![PendingArtifact::new(
            "keyframe-result.json",
            br#"{}"#.to_vec(),
            ArtifactValidation::Json,
        )])
    }
}

struct FixturePnpWorker;

impl PnpWorker for FixturePnpWorker {
    fn run(
        &self,
        request: StageRequest,
        _control: WorkerControl,
        _events: WorkerEventSink,
    ) -> WorkerOutcome {
        let imported_images = request
            .project_root
            .join("Artifacts/import/attempt-00000001/Cache/frames");
        let first = fs::read(imported_images.join("00000000.png")).unwrap();
        let second = fs::read(imported_images.join("00000001.png")).unwrap();
        WorkerOutcome::Succeeded(vec![
            PendingArtifact::new(
                "colmap/sparse/0/cameras.txt",
                b"1 PINHOLE 1 1 1.0 1.0 0.5 0.5\n".to_vec(),
                ArtifactValidation::ReadableFile,
            ),
            PendingArtifact::new(
                "colmap/sparse/0/images.txt",
                concat!(
                    "1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 00000000.png\n\n",
                    "2 1.0 0.0 0.0 0.0 1.0 0.0 0.0 1 00000001.png\n\n",
                )
                .as_bytes()
                .to_vec(),
                ArtifactValidation::ReadableFile,
            ),
            PendingArtifact::new(
                "colmap/sparse/0/points3D.txt",
                b"1 0.0 0.0 1.0 128 128 128 0.1 1 0\n".to_vec(),
                ArtifactValidation::ReadableFile,
            ),
            PendingArtifact::new(
                "colmap/images/00000000.png",
                first,
                ArtifactValidation::ReadableFile,
            ),
            PendingArtifact::new(
                "colmap/images/00000001.png",
                second,
                ArtifactValidation::ReadableFile,
            ),
            PendingArtifact::new(
                "pnp-result.json",
                br#"{"imported_frames":2,"registered_frames":2,"complete":true}"#.to_vec(),
                ArtifactValidation::PnpCoverage {
                    imported_frames: 2,
                    registered_frames: 2,
                },
            ),
        ])
    }
}

#[test]
fn image_sequence_naturally_sorts_normalizes_and_commits_every_frame() {
    let temp = tempfile::tempdir().unwrap();
    let sources = temp.path().join("source-images");
    fs::create_dir(&sources).unwrap();
    let paths = vec![
        write_fixture_png(&sources, "frame10.png"),
        write_fixture_png(&sources, "frame2.png"),
        write_fixture_png(&sources, "frame1.png"),
    ];
    let project = temp.path().join("Flowers.rustscanproject");
    let mut store = ProjectStore::create(
        &project,
        ProjectCreateRequest::new("Flowers", SourceSpec::managed_images("pending-import")),
    )
    .unwrap();
    let mut sink = RecordingSink::default();

    let result = import_image_sequence(
        &ImageSequenceImportRequest::managed(paths),
        &mut store,
        &mut sink,
    )
    .unwrap();

    assert_eq!(
        result
            .frames
            .iter()
            .map(|frame| frame.source_name.as_str())
            .collect::<Vec<_>>(),
        ["frame1.png", "frame2.png", "frame10.png"]
    );
    assert_eq!(
        result
            .frames
            .iter()
            .map(|frame| frame.id)
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );
    assert!(result.frames.iter().all(|frame| frame.is_keyframe));
    assert!(result
        .frames
        .iter()
        .all(|frame| project.join(&frame.normalized_image).is_file()));
    assert!(result
        .frames
        .iter()
        .all(|frame| project.join(&frame.thumbnail).is_file()));
    for frame in &result.frames {
        assert_eq!(
            image::open(project.join(&frame.normalized_image))
                .unwrap()
                .dimensions(),
            (frame.width, frame.height)
        );
        assert!(image::open(project.join(&frame.thumbnail)).is_ok());
    }
    assert!(result
        .frames
        .iter()
        .all(|frame| frame.width == 1 && frame.height == 1));
    assert!(project.join(&result.source_metadata).is_file());
    assert!(project.join(&result.frames_metadata).is_file());
    assert!(project.join(&result.keyframes_metadata).is_file());
    let source_metadata: serde_json::Value =
        serde_json::from_slice(&fs::read(project.join(&result.source_metadata)).unwrap()).unwrap();
    assert_eq!(source_metadata["ownership"], "managed_copy");
    assert_eq!(source_metadata["identity"], result.source_identity);
    for frame in source_metadata["frames"].as_array().unwrap() {
        let managed_copy = frame["managed_copy"].as_str().unwrap();
        assert!(project.join(managed_copy).is_file());
    }
    let frames_metadata: serde_json::Value =
        serde_json::from_slice(&fs::read(project.join(&result.frames_metadata)).unwrap()).unwrap();
    assert_eq!(frames_metadata.as_array().unwrap().len(), 3);
    let keyframes_metadata: serde_json::Value =
        serde_json::from_slice(&fs::read(project.join(&result.keyframes_metadata)).unwrap())
            .unwrap();
    assert_eq!(keyframes_metadata, serde_json::json!([0, 1, 2]));
    assert_eq!(
        store
            .manifest()
            .try_stage(ProjectStage::Import)
            .unwrap()
            .state(),
        StageState::Succeeded
    );
    assert_eq!(
        store
            .manifest()
            .try_stage(ProjectStage::Import)
            .unwrap()
            .artifacts()
            .iter()
            .map(|artifact| artifact.relative_path.as_str())
            .collect::<Vec<_>>(),
        [
            "Artifacts/import/attempt-00000001/Sources/source.json",
            "Artifacts/import/attempt-00000001/Cache/frames.json",
            "Artifacts/import/attempt-00000001/Cache/keyframes.json",
        ]
    );
    assert!(matches!(
        sink.events.as_slice(),
        [
            MediaImportEvent::Started { total: Some(3) },
            MediaImportEvent::FrameCommitted {
                frame_id: 0,
                completed: 1,
                total: Some(3)
            },
            MediaImportEvent::FrameCommitted {
                frame_id: 1,
                completed: 2,
                total: Some(3)
            },
            MediaImportEvent::FrameCommitted {
                frame_id: 2,
                completed: 3,
                total: Some(3)
            },
            MediaImportEvent::Completed {
                frame_count: 3,
                keyframe_count: 3
            },
        ]
    ));
}

#[test]
fn keyframe_selection_is_deterministic_bounded_and_close_to_target_rate() {
    let frames = (0..120)
        .map(|id| metadata_frame(id, i64::from(id) * 1_000_000 / 30, 10.0, unique_hash(id)))
        .collect::<Vec<_>>();
    let config = KeyframeSelectionConfig::default();

    let selected = select_keyframes(&frames, config).unwrap();

    assert_eq!(selected, select_keyframes(&frames, config).unwrap());
    assert_eq!(selected.first(), Some(&0));
    assert_eq!(selected.last(), Some(&119));
    assert!(selected.windows(2).all(|pair| {
        frame_by_id(&frames, pair[1]).presentation_time_us.unwrap()
            - frame_by_id(&frames, pair[0]).presentation_time_us.unwrap()
            <= config.max_gap_us
    }));
    assert!((12..=14).contains(&selected.len()));
}

#[test]
fn keyframe_selection_prefers_the_sharpest_near_duplicate_in_a_window() {
    let frames = vec![
        metadata_frame(0, 0, 1.0, 0),
        metadata_frame(1, 333_334, 2.0, u64::MAX),
        metadata_frame(2, 400_000, 9.0, u64::MAX ^ 1),
        metadata_frame(3, 800_000, 1.0, 0x0123_4567_89ab_cdef),
    ];

    let selected = select_keyframes(&frames, KeyframeSelectionConfig::default()).unwrap();

    assert!(selected.contains(&2));
    assert!(!selected.contains(&1));
}

#[test]
fn keyframe_selection_keeps_a_window_winner_when_its_forced_endpoints_share_the_window() {
    let frames = vec![
        metadata_frame(0, 0, 1.0, 0),
        metadata_frame(1, 100_000, 10.0, u64::MAX),
        metadata_frame(2, 200_000, 1.0, 0x0123_4567_89ab_cdef),
    ];

    let selected = select_keyframes(&frames, KeyframeSelectionConfig::default()).unwrap();

    assert_eq!(selected, vec![0, 1, 2]);
}

#[test]
fn keyframe_selection_suppresses_near_duplicates_without_exceeding_the_gap() {
    let frames = (0..10)
        .map(|id| metadata_frame(id, i64::from(id) * 333_333, f64::from(id), 0xfeed_face))
        .collect::<Vec<_>>();
    let config = KeyframeSelectionConfig::default();

    let selected = select_keyframes(&frames, config).unwrap();

    assert_eq!(selected, vec![0, 3, 6, 9]);
    assert!(selected.len() < frames.len());
    assert!(selected.windows(2).all(|pair| {
        frame_by_id(&frames, pair[1]).presentation_time_us.unwrap()
            - frame_by_id(&frames, pair[0]).presentation_time_us.unwrap()
            <= config.max_gap_us
    }));
}

#[test]
fn keyframe_selection_rejects_invalid_configuration_and_metadata() {
    let frame = metadata_frame(0, 0, 1.0, 0);

    let error = select_keyframes(
        &[frame.clone()],
        KeyframeSelectionConfig {
            target_per_second: 0.0,
            ..KeyframeSelectionConfig::default()
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("target_per_second"));

    let mut missing_timestamp = frame;
    missing_timestamp.presentation_time_us = None;
    let error =
        select_keyframes(&[missing_timestamp], KeyframeSelectionConfig::default()).unwrap_err();
    assert!(error.to_string().contains("presentation timestamp"));
}

#[test]
fn video_import_decodes_every_bgra_frame_and_marks_only_selected_keyframes() {
    let temp = tempfile::tempdir().unwrap();
    let project = temp.path().join("Video.rustscanproject");
    let mut store = ProjectStore::create(
        &project,
        ProjectCreateRequest::new("Video", SourceSpec::managed_images("pending-import")),
    )
    .unwrap();
    let mut decoder = FakeVideoDecoder::new(
        VideoMetadata {
            duration_us: 400_000,
            width: 2,
            height: 2,
            nominal_fps: 10.0,
        },
        (0..5)
            .map(|index| DecodedVideoFrame {
                presentation_time_us: i64::from(index) * 100_000,
                width: 2,
                height: 2,
                bgra: [
                    20 + index as u8,
                    30,
                    40,
                    255,
                    50,
                    60,
                    70,
                    255,
                    0,
                    0,
                    0,
                    0,
                    80,
                    90,
                    100,
                    255,
                    110,
                    120,
                    130,
                    255,
                    0,
                    0,
                    0,
                    0,
                ]
                .to_vec(),
                bytes_per_row: 12,
            })
            .collect(),
    );

    let result = import_video(
        &mut decoder,
        SourceSpec {
            kind: SourceKind::Video,
            ownership: SourceOwnership::ManagedCopy,
            identity: "fake-video".to_owned(),
            display_paths: Vec::new(),
            bookmark: None,
        },
        &mut store,
        &mut RecordingSink::default(),
    )
    .unwrap();

    assert_eq!(result.frames.len(), 5);
    assert!(result.frames.windows(2).all(|pair| {
        pair[0].presentation_time_us.unwrap() < pair[1].presentation_time_us.unwrap()
    }));
    assert!(result.frames.iter().any(|frame| !frame.is_keyframe));
    assert!(result.frames.iter().all(|frame| {
        image::open(project.join(&frame.normalized_image)).is_ok()
            && image::open(project.join(&frame.thumbnail)).is_ok()
    }));
    assert_eq!(
        image::open(project.join(&result.frames[0].normalized_image))
            .unwrap()
            .to_rgb8()
            .get_pixel(0, 0)
            .0,
        [40, 30, 20]
    );
    assert_eq!(decoder.frames_remaining(), 0);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "set RUSTSCAN_VIDEO_FIXTURE to run against a local MOV or MP4 fixture"]
fn video_decoder_avfoundation_fixture_emits_monotonic_frames() {
    use rust_viewer::media::AvFoundationVideoDecoder;

    let fixture = std::env::var_os("RUSTSCAN_VIDEO_FIXTURE")
        .expect("RUSTSCAN_VIDEO_FIXTURE must point to a MOV or MP4 fixture");
    let (mut decoder, source) = AvFoundationVideoDecoder::open_referenced(fixture).unwrap();
    assert_eq!(source.kind, SourceKind::Video);
    assert_eq!(source.ownership, SourceOwnership::Referenced);
    assert!(source.bookmark.is_some());
    let metadata = decoder.metadata().unwrap();
    let mut previous_timestamp = None;
    let mut frame_count = 0;
    while let Some(frame) = decoder.next_frame().unwrap() {
        assert!(frame.width > 0 && frame.height > 0);
        assert!(frame.bytes_per_row >= frame.width as usize * 4);
        assert!(frame.bgra.len() >= frame.bytes_per_row * frame.height as usize);
        if let Some(previous_timestamp) = previous_timestamp {
            assert!(frame.presentation_time_us > previous_timestamp);
        }
        previous_timestamp = Some(frame.presentation_time_us);
        frame_count += 1;
    }
    assert!(metadata.width > 0 && metadata.height > 0);
    assert!(frame_count > 0);
}

#[cfg(target_os = "macos")]
#[test]
fn video_decoder_saved_referenced_requires_a_persisted_bookmark() {
    use rust_viewer::media::AvFoundationVideoDecoder;

    let source = SourceSpec {
        kind: SourceKind::Video,
        ownership: SourceOwnership::Referenced,
        identity: "saved-video".to_owned(),
        display_paths: vec!["/missing/capture.mov".to_owned()],
        bookmark: None,
    };

    let error = match AvFoundationVideoDecoder::open_saved_referenced(&source) {
        Ok(_) => panic!("saved referenced video without a bookmark unexpectedly opened"),
        Err(error) => error,
    };

    assert!(
        matches!(error, MediaImportError::InvalidSource(detail) if detail.contains("bookmark"))
    );
}

struct FakeVideoDecoder {
    metadata: VideoMetadata,
    frames: VecDeque<DecodedVideoFrame>,
}

impl FakeVideoDecoder {
    fn new(metadata: VideoMetadata, frames: Vec<DecodedVideoFrame>) -> Self {
        Self {
            metadata,
            frames: frames.into(),
        }
    }

    fn frames_remaining(&self) -> usize {
        self.frames.len()
    }
}

impl VideoDecoder for FakeVideoDecoder {
    fn metadata(&self) -> Result<VideoMetadata, MediaImportError> {
        Ok(self.metadata)
    }

    fn next_frame(&mut self) -> Result<Option<DecodedVideoFrame>, MediaImportError> {
        Ok(self.frames.pop_front())
    }
}

fn metadata_frame(
    id: u32,
    presentation_time_us: i64,
    sharpness: f64,
    perceptual_hash: u64,
) -> ImportedFrame {
    ImportedFrame {
        id,
        source_name: format!("frame{id:04}.png"),
        presentation_time_us: Some(presentation_time_us),
        normalized_image: String::new(),
        thumbnail: String::new(),
        width: 1,
        height: 1,
        sharpness,
        perceptual_hash,
        is_keyframe: false,
    }
}

fn frame_by_id(frames: &[ImportedFrame], id: u32) -> &ImportedFrame {
    frames.iter().find(|frame| frame.id == id).unwrap()
}

fn unique_hash(id: u32) -> u64 {
    u64::from(id)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .rotate_left(17)
}
