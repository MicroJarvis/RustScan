use std::fs;
use std::path::{Path, PathBuf};

use image::GenericImageView;
use rust_viewer::media::{
    import_image_sequence, ImageSequenceImportRequest, MediaEventSink, MediaImportError,
    MediaImportEvent,
};
use rust_viewer::project::{
    ProjectCreateRequest, ProjectStage, ProjectStore, ProjectStoreWarning, SourceKind,
    SourceOwnership, SourceSpec, StageState, SuggestedAction,
};

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
