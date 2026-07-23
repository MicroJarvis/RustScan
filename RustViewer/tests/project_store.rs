use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::process::Command;
#[cfg(unix)]
use std::sync::mpsc;
#[cfg(unix)]
use std::time::Duration;

use fs2::FileExt;
use rust_viewer::project::{
    ArtifactRef, ChangeKind, ImportConfigSnapshot, PnpConfigSnapshot, ProjectCreateRequest,
    ProjectLease, ProjectManifest, ProjectManifestValidationError, ProjectStage, ProjectStore,
    ProjectStoreError, SfmConfigSnapshot, SourceKind, SourceOwnership, SourceSpec, StageState,
    PROJECT_SCHEMA_VERSION,
};

const STAGES: [(ProjectStage, &str); 5] = [
    (ProjectStage::Import, "import"),
    (ProjectStage::KeyframeSfm, "keyframe_sfm"),
    (ProjectStage::FullFramePnp, "full_frame_pnp"),
    (ProjectStage::Training, "training"),
    (ProjectStage::Complete, "complete"),
];

fn create_request(name: &str) -> ProjectCreateRequest {
    ProjectCreateRequest::new(name, SourceSpec::managed_images("source-a"))
}

fn assert_invalid(manifest: &ProjectManifest) {
    assert!(manifest.validate().is_err());
}

fn read_manifest_value(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&fs::read(path.join("project.json")).unwrap()).unwrap()
}

fn write_manifest_value(path: &Path, value: &serde_json::Value) {
    fs::write(
        path.join("project.json"),
        serde_json::to_vec_pretty(value).unwrap(),
    )
    .unwrap();
}

fn artifact_ref(relative_path: impl Into<String>, bytes: &[u8]) -> ArtifactRef {
    ArtifactRef {
        relative_path: relative_path.into(),
        content_hash: blake3::hash(bytes).to_hex().to_string(),
        byte_len: bytes.len() as u64,
    }
}

fn create_succeeded_project(base: &Path, name: &str) -> PathBuf {
    let path = base.join(format!("{name}.rustscanproject"));
    let store = ProjectStore::create(&path, create_request(name)).unwrap();
    drop(store);
    let mut value = read_manifest_value(&path);
    for (_, key) in STAGES {
        let bytes = format!("artifact-{key}").into_bytes();
        let relative = format!("Artifacts/{key}/attempt-00000001/result.bin");
        let artifact_path = path.join(&relative);
        fs::create_dir_all(artifact_path.parent().unwrap()).unwrap();
        fs::write(&artifact_path, &bytes).unwrap();
        value["stages"][key]["state"] = serde_json::json!("succeeded");
        value["stages"][key]["attempt"] = serde_json::json!(1);
        value["stages"][key]["artifacts"] =
            serde_json::to_value([artifact_ref(relative, &bytes)]).unwrap();
    }
    write_manifest_value(&path, &value);
    path
}

fn assert_stale_from(store: &ProjectStore, first_stale: ProjectStage) {
    let mut stale = false;
    for (stage, _) in STAGES {
        if stage == first_stale {
            stale = true;
        }
        assert_eq!(
            store.manifest().try_stage(stage).unwrap().state(),
            if stale {
                StageState::Stale
            } else {
                StageState::Succeeded
            }
        );
    }
}

#[test]
fn manifest_round_trip_preserves_initial_stage_records() {
    let manifest = ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));
    let json = serde_json::to_string_pretty(&manifest).unwrap();
    let restored: ProjectManifest = serde_json::from_str(&json).unwrap();

    restored.validate().unwrap();
    let import = restored.try_stage(ProjectStage::Import).unwrap();
    assert!(json.contains("\"import\""));
    assert_eq!(import.state(), StageState::Ready);
    assert_eq!(import.attempt(), 0);
    assert_eq!(import.completed(), None);
    assert_eq!(import.total(), None);
    assert_eq!(import.started_unix_ms(), None);
    assert!(import.updated_unix_ms() > 0);
    assert!(import.artifacts().is_empty());
    assert_eq!(import.error(), None);
    assert_eq!(restored, manifest);
}

#[test]
fn new_manifest_uses_the_declared_schema_and_config_defaults() {
    let source = SourceSpec::managed_images("source-a");
    let manifest = ProjectManifest::new("Flowers", source.clone());

    manifest.validate().unwrap();
    assert_eq!(manifest.schema_version(), PROJECT_SCHEMA_VERSION);
    assert_ne!(manifest.id(), uuid::Uuid::nil());
    assert_eq!(manifest.lease(), None);
    assert_eq!(source.kind, SourceKind::ImageSequence);
    assert_eq!(source.ownership, SourceOwnership::ManagedCopy);
    assert_eq!(source.identity, "source-a");
    assert!(source.display_paths.is_empty());
    assert_eq!(source.bookmark, None);
    assert_eq!(
        manifest.import_config,
        ImportConfigSnapshot {
            video_keyframes_per_second: 3.0,
            maximum_keyframe_gap_us: 1_000_000,
            thumbnail_long_edge: 256,
        }
    );
    assert_eq!(
        manifest.sfm_config,
        SfmConfigSnapshot {
            use_all_images: true,
            use_gpu_sift: true,
            use_gpu_matching: true,
        }
    );
    assert_eq!(
        manifest.pnp_config,
        PnpConfigSnapshot {
            narrow_neighbors_each_side: 2,
            wide_neighbors_each_side: 4,
            min_inliers: 24,
            min_inlier_ratio: 0.20,
            max_reprojection_error: 4.0,
            use_gpu_pnp: true,
        }
    );
    assert_eq!(
        manifest.compatibility.rustgs_checkpoint_version,
        rustgs::TRAINING_CHECKPOINT_VERSION
    );
}

#[test]
fn partial_stage_map_deserializes_but_fails_structural_validation_without_panicking() {
    let manifest = ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));
    let mut value = serde_json::to_value(manifest).unwrap();
    value["stages"].as_object_mut().unwrap().remove("training");
    let partial: ProjectManifest = serde_json::from_value(value).unwrap();

    assert!(matches!(
        partial.try_stage(ProjectStage::Training),
        Err(ProjectManifestValidationError::MissingStage {
            stage: ProjectStage::Training
        })
    ));
    assert!(matches!(
        partial.validate(),
        Err(ProjectManifestValidationError::MissingStage {
            stage: ProjectStage::Training
        })
    ));
}

#[test]
fn validation_rejects_schema_identity_progress_and_config_errors() {
    let manifest = ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));

    let mut wrong_schema = serde_json::to_value(&manifest).unwrap();
    wrong_schema["schema_version"] = serde_json::json!(PROJECT_SCHEMA_VERSION + 1);
    let wrong_schema: ProjectManifest = serde_json::from_value(wrong_schema).unwrap();
    assert!(matches!(
        wrong_schema.validate(),
        Err(ProjectManifestValidationError::UnsupportedSchemaVersion { .. })
    ));

    let mut empty_name = manifest.clone();
    empty_name.display_name = "  ".to_owned();
    assert!(matches!(
        empty_name.validate(),
        Err(ProjectManifestValidationError::EmptyDisplayName)
    ));

    let mut empty_identity = manifest.clone();
    empty_identity.source.identity = "\t".to_owned();
    assert!(matches!(
        empty_identity.validate(),
        Err(ProjectManifestValidationError::EmptySourceIdentity)
    ));

    let mut value = serde_json::to_value(&manifest).unwrap();
    value["stages"]["import"]["completed"] = serde_json::json!(2);
    value["stages"]["import"]["total"] = serde_json::json!(1);
    let invalid_progress: ProjectManifest = serde_json::from_value(value).unwrap();
    assert!(matches!(
        invalid_progress.validate(),
        Err(ProjectManifestValidationError::InvalidProgress {
            stage: ProjectStage::Import,
            completed: 2,
            total: 1,
        })
    ));

    for fps in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut invalid = manifest.clone();
        invalid.import_config.video_keyframes_per_second = fps;
        assert_invalid(&invalid);
    }
    let mut invalid = manifest.clone();
    invalid.import_config.maximum_keyframe_gap_us = 0;
    assert_invalid(&invalid);
    let mut invalid = manifest.clone();
    invalid.import_config.thumbnail_long_edge = 0;
    assert_invalid(&invalid);

    let mut invalid = manifest.clone();
    invalid.pnp_config.narrow_neighbors_each_side = 0;
    assert_invalid(&invalid);
    let mut invalid = manifest.clone();
    invalid.pnp_config.wide_neighbors_each_side = 1;
    invalid.pnp_config.narrow_neighbors_each_side = 2;
    assert_invalid(&invalid);
    let mut invalid = manifest.clone();
    invalid.pnp_config.min_inliers = 0;
    assert_invalid(&invalid);
    for ratio in [0.0, -0.1, 1.1, f64::NAN, f64::INFINITY] {
        let mut invalid = manifest.clone();
        invalid.pnp_config.min_inlier_ratio = ratio;
        assert_invalid(&invalid);
    }
    for error in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut invalid = manifest.clone();
        invalid.pnp_config.max_reprojection_error = error;
        assert_invalid(&invalid);
    }
    let mut invalid = manifest;
    invalid.training_config.iterations = 0;
    assert_invalid(&invalid);
}

#[test]
fn validation_rejects_missing_or_malformed_artifact_references() {
    let manifest = ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));

    let mut value = serde_json::to_value(&manifest).unwrap();
    value["stages"]["import"]["state"] = serde_json::json!("succeeded");
    let missing: ProjectManifest = serde_json::from_value(value).unwrap();
    assert!(matches!(
        missing.validate(),
        Err(
            ProjectManifestValidationError::SucceededStageWithoutArtifacts {
                stage: ProjectStage::Import
            }
        )
    ));

    let mut value = serde_json::to_value(&manifest).unwrap();
    value["stages"]["import"]["artifacts"] = serde_json::json!([{
        "relative_path": "Cache/frames.json",
        "content_hash": "not-a-blake3-hash",
        "byte_len": 0
    }]);
    let malformed: ProjectManifest = serde_json::from_value(value).unwrap();
    assert!(matches!(
        malformed.validate(),
        Err(ProjectManifestValidationError::InvalidArtifact { .. })
    ));

    for hash in ["A".repeat(64), "a".repeat(63), "g".repeat(64)] {
        let mut invalid = manifest.clone();
        invalid.active_scene = Some(ArtifactRef {
            relative_path: "Training/scene.ply".to_owned(),
            content_hash: hash,
            byte_len: 0,
        });
        assert_invalid(&invalid);
    }

    let mut invalid = manifest;
    invalid.final_scene = Some(ArtifactRef {
        relative_path: "../scene.ply".to_owned(),
        content_hash: "a".repeat(64),
        byte_len: 0,
    });
    assert_invalid(&invalid);
}

#[test]
fn validation_requires_exactly_one_matching_lease_for_one_active_stage() {
    let manifest = ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));
    let mut active_without_lease = serde_json::to_value(&manifest).unwrap();
    active_without_lease["stages"]["import"]["state"] = serde_json::json!("running");
    active_without_lease["stages"]["import"]["attempt"] = serde_json::json!(1);
    let active_without_lease: ProjectManifest =
        serde_json::from_value(active_without_lease).unwrap();
    assert!(matches!(
        active_without_lease.validate(),
        Err(ProjectManifestValidationError::ActiveStageWithoutLease {
            stage: ProjectStage::Import
        })
    ));

    let mut two_active = serde_json::to_value(&manifest).unwrap();
    two_active["stages"]["import"]["state"] = serde_json::json!("running");
    two_active["stages"]["import"]["attempt"] = serde_json::json!(1);
    two_active["stages"]["keyframe_sfm"]["state"] = serde_json::json!("pause_requested");
    two_active["stages"]["keyframe_sfm"]["attempt"] = serde_json::json!(1);
    let two_active: ProjectManifest = serde_json::from_value(two_active).unwrap();
    assert!(matches!(
        two_active.validate(),
        Err(ProjectManifestValidationError::MultipleActiveStages { .. })
    ));

    let mut lease_without_active = serde_json::to_value(&manifest).unwrap();
    lease_without_active["stages"]["import"]["state"] = serde_json::json!("queued");
    lease_without_active["stages"]["import"]["attempt"] = serde_json::json!(1);
    lease_without_active["lease"] = serde_json::to_value(ProjectLease {
        project_id: manifest.id(),
        stage: ProjectStage::Import,
        attempt: 1,
        process_id: 42,
        started_unix_ms: 100,
    })
    .unwrap();
    let lease_without_active: ProjectManifest =
        serde_json::from_value(lease_without_active).unwrap();
    assert!(matches!(
        lease_without_active.validate(),
        Err(ProjectManifestValidationError::LeaseWithoutActiveStage {
            stage: ProjectStage::Import
        })
    ));

    let mut valid = serde_json::to_value(&manifest).unwrap();
    valid["stages"]["import"]["state"] = serde_json::json!("running");
    valid["stages"]["import"]["attempt"] = serde_json::json!(1);
    valid["lease"] = serde_json::to_value(ProjectLease {
        project_id: manifest.id(),
        stage: ProjectStage::Import,
        attempt: 1,
        process_id: 42,
        started_unix_ms: 100,
    })
    .unwrap();
    let valid: ProjectManifest = serde_json::from_value(valid).unwrap();
    valid.validate().unwrap();

    let mut wrong_project = serde_json::to_value(&valid).unwrap();
    wrong_project["lease"]["project_id"] = serde_json::to_value(uuid::Uuid::new_v4()).unwrap();
    let wrong_project: ProjectManifest = serde_json::from_value(wrong_project).unwrap();
    assert!(matches!(
        wrong_project.validate(),
        Err(ProjectManifestValidationError::LeaseProjectMismatch { .. })
    ));

    let mut wrong_attempt = serde_json::to_value(&valid).unwrap();
    wrong_attempt["lease"]["attempt"] = serde_json::json!(2);
    let wrong_attempt: ProjectManifest = serde_json::from_value(wrong_attempt).unwrap();
    assert!(matches!(
        wrong_attempt.validate(),
        Err(ProjectManifestValidationError::LeaseAttemptMismatch { .. })
    ));

    let mut wrong_stage = serde_json::to_value(&valid).unwrap();
    wrong_stage["lease"]["stage"] = serde_json::json!("keyframe_sfm");
    let wrong_stage: ProjectManifest = serde_json::from_value(wrong_stage).unwrap();
    assert!(matches!(
        wrong_stage.validate(),
        Err(ProjectManifestValidationError::LeaseActiveStageMismatch { .. })
    ));
}

#[test]
fn project_store_create_builds_the_exact_package_tree_and_validates_destination() {
    let temp = tempfile::tempdir().unwrap();
    let invalid_suffix = temp.path().join("Flowers.project");
    assert!(matches!(
        ProjectStore::create(&invalid_suffix, create_request("Flowers")),
        Err(ProjectStoreError::InvalidPackageSuffix { .. })
    ));
    assert!(!invalid_suffix.exists());

    let nonempty = temp.path().join("Existing.rustscanproject");
    fs::create_dir(&nonempty).unwrap();
    fs::write(nonempty.join("keep.txt"), b"keep").unwrap();
    assert!(matches!(
        ProjectStore::create(&nonempty, create_request("Existing")),
        Err(ProjectStoreError::DestinationNotEmpty { .. })
    ));
    assert_eq!(fs::read(nonempty.join("keep.txt")).unwrap(), b"keep");

    let project = temp.path().join("Flowers.rustscanproject");
    fs::create_dir(&project).unwrap();
    let store = ProjectStore::create(&project, create_request("Flowers")).unwrap();
    assert_eq!(store.root(), fs::canonicalize(&project).unwrap());
    assert_eq!(store.manifest().schema_version(), PROJECT_SCHEMA_VERSION);
    for relative in [
        "Sources",
        "Sources/managed",
        "Cache/frames",
        "Cache/thumbnails",
        "Cache/.staging",
        "Reconstruction",
        "Training/checkpoints",
        "Artifacts",
        "Logs",
        "Logs/recovery",
    ] {
        assert!(store.root().join(relative).is_dir(), "missing {relative}");
    }
    let top_level = fs::read_dir(store.root())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        top_level,
        [
            "Artifacts",
            "Cache",
            "Logs",
            "Reconstruction",
            "Sources",
            "Training",
            "project.json",
            "project.lock",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    );
}

#[test]
fn project_store_create_cleans_initialization_failures() {
    let temp = tempfile::tempdir().unwrap();
    let invalid = temp.path().join("Invalid.rustscanproject");
    assert!(matches!(
        ProjectStore::create(&invalid, create_request("  ")),
        Err(ProjectStoreError::InvalidManifest(_))
    ));
    assert!(!invalid.exists());

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let existing = temp.path().join("ReadOnly.rustscanproject");
        fs::create_dir(&existing).unwrap();
        fs::set_permissions(&existing, fs::Permissions::from_mode(0o500)).unwrap();
        let probe = existing.join("permission-probe");
        match fs::write(&probe, b"probe") {
            Ok(()) => {
                fs::remove_file(probe).unwrap();
                fs::set_permissions(&existing, fs::Permissions::from_mode(0o700)).unwrap();
                return;
            }
            Err(error) => assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied),
        }
        assert!(matches!(
            ProjectStore::create(&existing, create_request("Read Only")),
            Err(ProjectStoreError::Io(_))
        ));
        assert!(fs::read_dir(&existing).unwrap().next().is_none());
        fs::set_permissions(&existing, fs::Permissions::from_mode(0o700)).unwrap();
    }
}

#[test]
fn project_store_create_accepts_the_orphan_lock_left_by_a_failed_open() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("Orphan.rustscanproject");
    fs::create_dir(&path).unwrap();

    assert!(matches!(
        ProjectStore::open(&path),
        Err(ProjectStoreError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound
    ));
    assert!(path.join("project.lock").is_file());

    let store = ProjectStore::create(&path, create_request("Recovered")).unwrap();
    assert_eq!(store.manifest().display_name, "Recovered");
}

#[test]
fn project_store_create_rejects_an_orphan_lock_alongside_other_content() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("Occupied.rustscanproject");
    fs::create_dir(&path).unwrap();
    fs::write(path.join("project.lock"), b"").unwrap();
    fs::write(path.join("keep.txt"), b"keep").unwrap();

    assert!(matches!(
        ProjectStore::create(&path, create_request("Occupied")),
        Err(ProjectStoreError::DestinationNotEmpty { .. })
    ));
    assert_eq!(fs::read(path.join("keep.txt")).unwrap(), b"keep");
}

#[test]
fn project_store_create_rejects_a_locked_orphan_lock() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("Locked.rustscanproject");
    fs::create_dir(&path).unwrap();
    let lock = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(path.join("project.lock"))
        .unwrap();
    lock.try_lock_exclusive().unwrap();

    assert!(matches!(
        ProjectStore::create(&path, create_request("Locked")),
        Err(ProjectStoreError::AlreadyOpen { .. })
    ));
}

#[test]
fn project_store_allows_only_one_writer() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("Flowers.rustscanproject");
    let first = ProjectStore::create(&path, create_request("Flowers")).unwrap();
    assert!(matches!(
        ProjectStore::open(&path),
        Err(ProjectStoreError::AlreadyOpen { .. })
    ));
    drop(first);
    ProjectStore::open(&path).unwrap();
}

#[cfg(unix)]
#[test]
fn project_store_lock_replacement_does_not_admit_a_second_writer() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("Flowers.rustscanproject");
    let first = ProjectStore::create(&path, create_request("Flowers")).unwrap();
    let lock_path = path.join("project.lock");
    fs::remove_file(&lock_path).unwrap();
    fs::write(&lock_path, b"").unwrap();

    let second = ProjectStore::open(&path);

    assert!(matches!(second, Err(ProjectStoreError::AlreadyOpen { .. })));
    drop(first);
}

#[test]
fn project_store_open_dispatches_schema_before_typed_deserialization() {
    let temp = tempfile::tempdir().unwrap();
    let future_path = temp.path().join("Future.rustscanproject");
    let future = ProjectStore::create(&future_path, create_request("Future")).unwrap();
    drop(future);
    let mut value = read_manifest_value(&future_path);
    value["schema_version"] = serde_json::json!(PROJECT_SCHEMA_VERSION + 1);
    value["stages"] = serde_json::json!("not a stage map");
    write_manifest_value(&future_path, &value);
    assert!(matches!(
        ProjectStore::open(&future_path),
        Err(ProjectStoreError::FutureSchemaVersion { .. })
    ));

    let old_path = temp.path().join("Old.rustscanproject");
    let old = ProjectStore::create(&old_path, create_request("Old")).unwrap();
    drop(old);
    let mut value = read_manifest_value(&old_path);
    value["schema_version"] = serde_json::json!(0);
    write_manifest_value(&old_path, &value);
    assert!(matches!(
        ProjectStore::open(&old_path),
        Err(ProjectStoreError::MigrationUnavailable {
            from: 0,
            to: PROJECT_SCHEMA_VERSION
        })
    ));
}

#[test]
fn project_store_open_returns_structured_errors_for_malformed_and_partial_manifests() {
    let temp = tempfile::tempdir().unwrap();
    let malformed_path = temp.path().join("Malformed.rustscanproject");
    let malformed = ProjectStore::create(&malformed_path, create_request("Malformed")).unwrap();
    drop(malformed);
    fs::write(malformed_path.join("project.json"), b"{not json").unwrap();
    assert!(matches!(
        ProjectStore::open(&malformed_path),
        Err(ProjectStoreError::MalformedJson { .. })
    ));

    let partial_path = temp.path().join("Partial.rustscanproject");
    let partial = ProjectStore::create(&partial_path, create_request("Partial")).unwrap();
    drop(partial);
    let mut value = read_manifest_value(&partial_path);
    value["stages"].as_object_mut().unwrap().remove("training");
    write_manifest_value(&partial_path, &value);
    assert!(matches!(
        ProjectStore::open(&partial_path),
        Err(ProjectStoreError::InvalidManifest(
            ProjectManifestValidationError::MissingStage {
                stage: ProjectStage::Training
            }
        ))
    ));

    let missing_schema_path = temp.path().join("MissingSchema.rustscanproject");
    let missing_schema =
        ProjectStore::create(&missing_schema_path, create_request("Missing Schema")).unwrap();
    drop(missing_schema);
    let mut value = read_manifest_value(&missing_schema_path);
    value.as_object_mut().unwrap().remove("schema_version");
    write_manifest_value(&missing_schema_path, &value);
    assert!(matches!(
        ProjectStore::open(&missing_schema_path),
        Err(ProjectStoreError::InvalidSchemaVersion)
    ));
}

#[cfg(unix)]
#[test]
fn project_store_open_rejects_a_symlink_package_root() {
    use std::os::unix::fs::symlink;

    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("Flowers.rustscanproject");
    let store = ProjectStore::create(&path, create_request("Flowers")).unwrap();
    drop(store);
    let link = temp.path().join("Link.rustscanproject");
    symlink(&path, &link).unwrap();
    assert!(matches!(
        ProjectStore::open(&link),
        Err(ProjectStoreError::SymlinkPackageRoot { .. })
    ));
}

#[test]
fn project_store_typed_updates_persist_identity_and_apply_invalidation_boundaries() {
    let temp = tempfile::tempdir().unwrap();

    let path = create_succeeded_project(temp.path(), "Source");
    let mut store = ProjectStore::open(&path).unwrap();
    let id = store.manifest().id();
    store
        .update_source(SourceSpec::managed_images("source-b"))
        .unwrap();
    assert_eq!(store.manifest().id(), id);
    assert_stale_from(&store, ProjectStage::Import);
    drop(store);
    let reopened = ProjectStore::open(&path).unwrap();
    assert_eq!(reopened.manifest().source.identity, "source-b");
    drop(reopened);

    let path = create_succeeded_project(temp.path(), "Import");
    let mut store = ProjectStore::open(&path).unwrap();
    let mut config = store.manifest().import_config.clone();
    config.video_keyframes_per_second = 4.0;
    store.update_import_config(config).unwrap();
    assert_stale_from(&store, ProjectStage::Import);
    drop(store);

    let path = create_succeeded_project(temp.path(), "Sfm");
    let mut store = ProjectStore::open(&path).unwrap();
    let mut config = store.manifest().sfm_config.clone();
    config.use_all_images = false;
    store.update_sfm_config(config).unwrap();
    assert_stale_from(&store, ProjectStage::KeyframeSfm);
    drop(store);

    let path = create_succeeded_project(temp.path(), "Pnp");
    let mut store = ProjectStore::open(&path).unwrap();
    let mut config = store.manifest().pnp_config.clone();
    config.min_inliers += 1;
    store.update_pnp_config(config).unwrap();
    assert_stale_from(&store, ProjectStage::FullFramePnp);
    drop(store);

    let path = create_succeeded_project(temp.path(), "Training");
    let mut store = ProjectStore::open(&path).unwrap();
    let mut config = store.manifest().training_config.clone();
    config.iterations += 1;
    store.update_training_config(config).unwrap();
    assert_stale_from(&store, ProjectStage::Training);
}

#[test]
fn project_store_typed_updates_reject_only_changes_that_invalidate_the_active_stage() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("Active.rustscanproject");
    let store = ProjectStore::create(&path, create_request("Active")).unwrap();
    let id = store.manifest().id();
    drop(store);

    let mut value = read_manifest_value(&path);
    value["stages"]["import"]["state"] = serde_json::json!("running");
    value["stages"]["import"]["attempt"] = serde_json::json!(1);
    value["lease"] = serde_json::to_value(ProjectLease {
        project_id: id,
        stage: ProjectStage::Import,
        attempt: 1,
        process_id: 42,
        started_unix_ms: 100,
    })
    .unwrap();
    write_manifest_value(&path, &value);

    let mut store = ProjectStore::open(&path).unwrap();
    let before_bytes = fs::read(path.join("project.json")).unwrap();
    let before_import = store.manifest().import_config.clone();
    let mut import = before_import.clone();
    import.video_keyframes_per_second += 1.0;

    assert!(matches!(
        store.update_import_config(import),
        Err(ProjectStoreError::StageActive {
            stage: ProjectStage::Import,
            change: ChangeKind::ImportConfig,
        })
    ));
    assert_eq!(store.manifest().import_config, before_import);
    assert_eq!(fs::read(path.join("project.json")).unwrap(), before_bytes);

    let mut training = store.manifest().training_config.clone();
    training.iterations += 1;
    store.update_training_config(training.clone()).unwrap();
    assert_eq!(store.manifest().training_config, training);
    assert_eq!(
        store
            .manifest()
            .try_stage(ProjectStage::Import)
            .unwrap()
            .state(),
        StageState::Running
    );
    assert_eq!(
        store.manifest().lease().unwrap().stage,
        ProjectStage::Import
    );
}

#[test]
fn project_store_typed_update_validates_before_atomic_manifest_replacement() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("Flowers.rustscanproject");
    let mut store = ProjectStore::create(&path, create_request("Flowers")).unwrap();
    let before = fs::read(path.join("project.json")).unwrap();
    let id = store.manifest().id();
    let mut invalid = store.manifest().import_config.clone();
    invalid.video_keyframes_per_second = f64::NAN;

    assert!(matches!(
        store.update_import_config(invalid),
        Err(ProjectStoreError::InvalidManifest(_))
    ));
    assert_eq!(store.manifest().id(), id);
    assert_eq!(fs::read(path.join("project.json")).unwrap(), before);
    assert!(fs::read_dir(&path).unwrap().all(|entry| {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        !name.ends_with(".tmp")
    }));
}

#[cfg(unix)]
#[test]
fn project_store_typed_update_preserves_state_on_pre_rename_io_failure() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("Flowers.rustscanproject");
    let mut store = ProjectStore::create(&path, create_request("Flowers")).unwrap();
    let before_bytes = fs::read(path.join("project.json")).unwrap();
    let before_config = store.manifest().import_config.clone();
    let mut updated = before_config.clone();
    updated.video_keyframes_per_second = 4.0;

    fs::set_permissions(&path, fs::Permissions::from_mode(0o500)).unwrap();
    let probe = path.join("permission-probe");
    match fs::write(&probe, b"probe") {
        Ok(()) => {
            fs::remove_file(probe).unwrap();
            fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
            return;
        }
        Err(error) => assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied),
    }
    let result = store.update_import_config(updated);
    fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();

    assert!(matches!(result, Err(ProjectStoreError::Io(_))));
    assert_eq!(store.manifest().import_config, before_config);
    assert_eq!(fs::read(path.join("project.json")).unwrap(), before_bytes);
    assert!(fs::read_dir(&path).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .ends_with(".tmp")
    }));
}

fn create_project_with_artifact(base: &Path, name: &str, reference: &ArtifactRef) -> PathBuf {
    let path = base.join(format!("{name}.rustscanproject"));
    let store = ProjectStore::create(&path, create_request(name)).unwrap();
    drop(store);
    let mut value = read_manifest_value(&path);
    value["stages"]["import"]["artifacts"] = serde_json::to_value([reference.clone()]).unwrap();
    write_manifest_value(&path, &value);
    path
}

#[test]
fn project_store_open_stream_validates_stage_and_scene_artifacts() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("Valid.rustscanproject");
    let store = ProjectStore::create(&path, create_request("Valid")).unwrap();
    drop(store);
    let bytes = (0..200_000)
        .map(|index| (index % 251) as u8)
        .collect::<Vec<_>>();
    let relative = "Artifacts/import/attempt-00000001/result.bin";
    fs::create_dir_all(path.join(relative).parent().unwrap()).unwrap();
    fs::write(path.join(relative), &bytes).unwrap();
    let reference = artifact_ref(relative, &bytes);
    let mut value = read_manifest_value(&path);
    value["stages"]["import"]["artifacts"] = serde_json::to_value([reference.clone()]).unwrap();
    value["active_scene"] = serde_json::to_value(&reference).unwrap();
    value["final_scene"] = serde_json::to_value(&reference).unwrap();
    write_manifest_value(&path, &value);

    ProjectStore::open(&path).unwrap();
}

#[test]
fn project_store_open_rejects_missing_length_hash_and_non_file_artifacts() {
    let temp = tempfile::tempdir().unwrap();
    let bytes = b"artifact";
    let relative = "Artifacts/import/attempt-00000001/result.bin";

    let missing =
        create_project_with_artifact(temp.path(), "Missing", &artifact_ref(relative, bytes));
    assert!(matches!(
        ProjectStore::open(&missing),
        Err(ProjectStoreError::ArtifactMissing { .. })
    ));

    let length = create_project_with_artifact(
        temp.path(),
        "Length",
        &ArtifactRef {
            byte_len: bytes.len() as u64 + 1,
            ..artifact_ref(relative, bytes)
        },
    );
    fs::create_dir_all(length.join(relative).parent().unwrap()).unwrap();
    fs::write(length.join(relative), bytes).unwrap();
    assert!(matches!(
        ProjectStore::open(&length),
        Err(ProjectStoreError::ArtifactLengthMismatch { .. })
    ));

    let hash = create_project_with_artifact(
        temp.path(),
        "Hash",
        &ArtifactRef {
            content_hash: "a".repeat(64),
            ..artifact_ref(relative, bytes)
        },
    );
    fs::create_dir_all(hash.join(relative).parent().unwrap()).unwrap();
    fs::write(hash.join(relative), bytes).unwrap();
    assert!(matches!(
        ProjectStore::open(&hash),
        Err(ProjectStoreError::ArtifactHashMismatch { .. })
    ));

    let directory =
        create_project_with_artifact(temp.path(), "Directory", &artifact_ref(relative, bytes));
    fs::create_dir_all(directory.join(relative)).unwrap();
    assert!(matches!(
        ProjectStore::open(&directory),
        Err(ProjectStoreError::ArtifactNotRegularFile { .. })
    ));
}

#[cfg(unix)]
#[test]
fn project_store_open_rejects_a_fifo_without_waiting_for_a_writer() {
    let temp = tempfile::tempdir().unwrap();
    let bytes = b"artifact";
    let relative = "Artifacts/import/attempt-00000001/result.fifo";
    let path = create_project_with_artifact(
        temp.path(),
        "Fifo",
        &ArtifactRef {
            relative_path: relative.to_owned(),
            content_hash: blake3::hash(bytes).to_hex().to_string(),
            byte_len: bytes.len() as u64,
        },
    );
    fs::create_dir_all(path.join(relative).parent().unwrap()).unwrap();
    assert!(Command::new("mkfifo")
        .arg(path.join(relative))
        .status()
        .unwrap()
        .success());

    let (tx, rx) = mpsc::sync_channel(1);
    let open_path = path.clone();
    let handle = std::thread::spawn(move || {
        let _ = tx.send(ProjectStore::open(open_path));
    });
    let result = match rx.recv_timeout(Duration::from_secs(2)) {
        Ok(result) => result,
        Err(error) => {
            let writer = OpenOptions::new()
                .write(true)
                .open(path.join(relative))
                .unwrap();
            let _ = rx.recv_timeout(Duration::from_secs(2));
            drop(writer);
            handle.join().unwrap();
            panic!("artifact validation blocked while opening a FIFO: {error}");
        }
    };
    handle.join().unwrap();

    assert!(matches!(
        result,
        Err(ProjectStoreError::ArtifactNotRegularFile { .. })
    ));
}

#[test]
fn project_store_rejects_unsafe_committed_artifact_paths() {
    let temp = tempfile::tempdir().unwrap();
    let reference = ArtifactRef {
        relative_path: "../outside.bin".to_owned(),
        content_hash: "a".repeat(64),
        byte_len: 1,
    };
    let path = create_project_with_artifact(temp.path(), "Traversal", &reference);
    assert!(matches!(
        ProjectStore::open(&path),
        Err(ProjectStoreError::InvalidManifest(
            ProjectManifestValidationError::InvalidArtifact { .. }
        ))
    ));

    let reference = ArtifactRef {
        relative_path: r"Artifacts\..\outside.bin".to_owned(),
        content_hash: "a".repeat(64),
        byte_len: 1,
    };
    let path = create_project_with_artifact(temp.path(), "Backslash", &reference);
    assert!(matches!(
        ProjectStore::open(&path),
        Err(ProjectStoreError::InvalidManifest(
            ProjectManifestValidationError::InvalidArtifact { .. }
        ))
    ));
}

#[cfg(unix)]
#[test]
fn project_store_rejects_symlinked_committed_artifacts_and_ancestor_escapes() {
    use std::os::unix::fs::symlink;

    let temp = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let bytes = b"artifact";
    fs::write(outside.path().join("result.bin"), bytes).unwrap();
    let relative = "Artifacts/import/attempt-00000001/result.bin";

    let leaf =
        create_project_with_artifact(temp.path(), "LeafSymlink", &artifact_ref(relative, bytes));
    fs::create_dir_all(leaf.join(relative).parent().unwrap()).unwrap();
    symlink(outside.path().join("result.bin"), leaf.join(relative)).unwrap();
    assert!(matches!(
        ProjectStore::open(&leaf),
        Err(ProjectStoreError::ArtifactSymlink { .. })
    ));

    let ancestor = create_project_with_artifact(
        temp.path(),
        "AncestorSymlink",
        &artifact_ref(relative, bytes),
    );
    fs::create_dir_all(ancestor.join("Artifacts")).unwrap();
    symlink(outside.path(), ancestor.join("Artifacts/import")).unwrap();
    assert!(matches!(
        ProjectStore::open(&ancestor),
        Err(ProjectStoreError::ArtifactSymlink { .. })
    ));
}
