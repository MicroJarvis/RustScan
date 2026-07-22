use rust_viewer::project::{
    ArtifactRef, ChangeKind, ImportConfigSnapshot, PnpConfigSnapshot, ProjectLease,
    ProjectManifest, ProjectManifestValidationError, ProjectStage, SfmConfigSnapshot, SourceKind,
    SourceOwnership, SourceSpec, StageState, PROJECT_SCHEMA_VERSION,
};

const STAGES: [(ProjectStage, &str); 5] = [
    (ProjectStage::Import, "import"),
    (ProjectStage::KeyframeSfm, "keyframe_sfm"),
    (ProjectStage::FullFramePnp, "full_frame_pnp"),
    (ProjectStage::Training, "training"),
    (ProjectStage::Complete, "complete"),
];

fn artifact_for(stage: ProjectStage) -> ArtifactRef {
    ArtifactRef {
        relative_path: format!("artifacts/{stage:?}.json").to_lowercase(),
        content_hash: "a".repeat(64),
        byte_len: 42,
    }
}

fn persisted_manifest_succeeded_through(last: ProjectStage) -> ProjectManifest {
    let manifest = ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));
    let mut value = serde_json::to_value(manifest).unwrap();
    for (stage, key) in STAGES {
        let record = &mut value["stages"][key];
        record["state"] = serde_json::json!("succeeded");
        record["attempt"] = serde_json::json!(1);
        record["artifacts"] = serde_json::to_value([artifact_for(stage)]).unwrap();
        if stage == last {
            break;
        }
    }
    let manifest: ProjectManifest = serde_json::from_value(value).unwrap();
    manifest.validate().unwrap();
    manifest
}

fn assert_invalid(manifest: &ProjectManifest) {
    assert!(manifest.validate().is_err());
}

#[test]
fn manifest_round_trip_preserves_stage_records() {
    let mut manifest = ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));
    manifest
        .transition(ProjectStage::Import, StageState::Queued)
        .unwrap();
    manifest
        .transition(ProjectStage::Import, StageState::Running)
        .unwrap();

    let json = serde_json::to_string_pretty(&manifest).unwrap();
    let restored: ProjectManifest = serde_json::from_str(&json).unwrap();

    restored.validate().unwrap();
    let import = restored.try_stage(ProjectStage::Import).unwrap();
    assert!(json.contains("\"import\""));
    assert!(json.contains("\"running\""));
    assert_eq!(import.state(), StageState::Running);
    assert_eq!(import.attempt(), 1);
    assert_eq!(import.completed(), None);
    assert_eq!(import.total(), None);
    assert!(import.started_unix_ms().is_some());
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
    assert_eq!(manifest.schema_version, PROJECT_SCHEMA_VERSION);
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
fn succeeded_sfm_becomes_stale_when_keyframes_change() {
    let mut manifest = persisted_manifest_succeeded_through(ProjectStage::Training);

    manifest.invalidate(ChangeKind::KeyframeSelection);

    assert_eq!(
        manifest
            .try_stage(ProjectStage::KeyframeSfm)
            .unwrap()
            .state(),
        StageState::Stale
    );
    assert_eq!(
        manifest
            .try_stage(ProjectStage::FullFramePnp)
            .unwrap()
            .state(),
        StageState::Stale
    );
    assert_eq!(
        manifest.try_stage(ProjectStage::Training).unwrap().state(),
        StageState::Stale
    );
    assert_eq!(
        manifest.try_stage(ProjectStage::Import).unwrap().state(),
        StageState::Succeeded
    );
}

#[test]
fn request_transitions_are_public_but_direct_pause_and_success_remain_forbidden() {
    let mut manifest = ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));
    manifest
        .transition(ProjectStage::Import, StageState::Queued)
        .unwrap();
    manifest
        .transition(ProjectStage::Import, StageState::Running)
        .unwrap();
    assert!(manifest
        .transition(ProjectStage::Import, StageState::Paused)
        .is_err());
    assert!(manifest
        .transition(ProjectStage::Import, StageState::Succeeded)
        .is_err());

    manifest
        .transition(ProjectStage::Import, StageState::PauseRequested)
        .unwrap();
    manifest
        .transition(ProjectStage::Import, StageState::CancelRequested)
        .unwrap();
    manifest
        .transition(ProjectStage::Import, StageState::Cancelled)
        .unwrap();
    assert_eq!(
        manifest.try_stage(ProjectStage::Import).unwrap().state(),
        StageState::Cancelled
    );
}

#[test]
fn invalidation_follows_every_change_category_dependency_boundary() {
    let cases = [
        (ChangeKind::Source, Some(ProjectStage::Import)),
        (
            ChangeKind::KeyframeSelection,
            Some(ProjectStage::KeyframeSfm),
        ),
        (ChangeKind::SfmConfig, Some(ProjectStage::KeyframeSfm)),
        (ChangeKind::PnpConfig, Some(ProjectStage::FullFramePnp)),
        (ChangeKind::TrainingConfig, Some(ProjectStage::Training)),
        (ChangeKind::ViewerAppearance, None),
    ];

    for (change, first_invalidated) in cases {
        let mut manifest = persisted_manifest_succeeded_through(ProjectStage::Complete);
        manifest.invalidate(change);

        let mut invalidated = false;
        for (stage, _) in STAGES {
            if Some(stage) == first_invalidated {
                invalidated = true;
            }
            let expected = if invalidated {
                StageState::Stale
            } else {
                StageState::Succeeded
            };
            assert_eq!(
                manifest.try_stage(stage).unwrap().state(),
                expected,
                "unexpected state for {stage:?} after {change:?}"
            );
        }
    }
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
fn validation_rejects_schema_identity_and_progress_errors() {
    let manifest = ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));

    let mut wrong_schema = manifest.clone();
    wrong_schema.schema_version += 1;
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

    let mut value = serde_json::to_value(manifest).unwrap();
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
}

#[test]
fn validation_rejects_invalid_import_and_pnp_configs() {
    let manifest = ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));

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
}

#[test]
fn validation_delegates_to_training_config_validation() {
    let mut manifest = ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));
    manifest.training_config.iterations = 0;

    assert!(matches!(
        manifest.validate(),
        Err(ProjectManifestValidationError::InvalidTrainingConfig { .. })
    ));
}

#[test]
fn validation_rejects_missing_or_malformed_artifacts() {
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
    let malformed_stage_artifact: ProjectManifest = serde_json::from_value(value).unwrap();
    assert!(matches!(
        malformed_stage_artifact.validate(),
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

    let mut invalid = manifest.clone();
    invalid.final_scene = Some(ArtifactRef {
        relative_path: "../scene.ply".to_owned(),
        content_hash: "a".repeat(64),
        byte_len: 0,
    });
    assert_invalid(&invalid);
}

#[test]
fn validation_checks_lease_identity_attempt_and_active_state() {
    let mut running = ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));
    running
        .transition(ProjectStage::Import, StageState::Queued)
        .unwrap();
    running
        .transition(ProjectStage::Import, StageState::Running)
        .unwrap();
    let attempt = running.try_stage(ProjectStage::Import).unwrap().attempt();
    running.lease = Some(ProjectLease {
        project_id: running.id,
        stage: ProjectStage::Import,
        attempt,
        process_id: 42,
        started_unix_ms: 100,
    });
    running.validate().unwrap();

    let mut wrong_project = running.clone();
    wrong_project.lease.as_mut().unwrap().project_id = uuid::Uuid::new_v4();
    assert!(matches!(
        wrong_project.validate(),
        Err(ProjectManifestValidationError::LeaseProjectMismatch { .. })
    ));

    let mut wrong_attempt = running.clone();
    wrong_attempt.lease.as_mut().unwrap().attempt += 1;
    assert!(matches!(
        wrong_attempt.validate(),
        Err(ProjectManifestValidationError::LeaseAttemptMismatch { .. })
    ));

    let mut queued = ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));
    queued
        .transition(ProjectStage::Import, StageState::Queued)
        .unwrap();
    queued.lease = Some(ProjectLease {
        project_id: queued.id,
        stage: ProjectStage::Import,
        attempt: queued.try_stage(ProjectStage::Import).unwrap().attempt(),
        process_id: 42,
        started_unix_ms: 100,
    });
    assert!(matches!(
        queued.validate(),
        Err(ProjectManifestValidationError::LeaseStageNotActive { .. })
    ));
}
