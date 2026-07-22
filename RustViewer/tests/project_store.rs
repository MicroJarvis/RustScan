use rust_viewer::project::{
    ArtifactRef, ChangeKind, ImportConfigSnapshot, PnpConfigSnapshot, ProjectErrorRecord,
    ProjectManifest, ProjectStage, SfmConfigSnapshot, SourceKind, SourceOwnership, SourceSpec,
    StageState, SuggestedAction, ValidatedArtifacts, PROJECT_SCHEMA_VERSION,
};

fn artifact_for(stage: ProjectStage) -> ArtifactRef {
    ArtifactRef {
        relative_path: format!("artifacts/{stage:?}.json").to_lowercase(),
        content_hash: "a".repeat(64),
        byte_len: 42,
    }
}

fn commit_ready_stage(manifest: &mut ProjectManifest, stage: ProjectStage) {
    manifest.transition(stage, StageState::Queued).unwrap();
    manifest.transition(stage, StageState::Running).unwrap();
    let artifacts = ValidatedArtifacts::try_new(vec![artifact_for(stage)]).unwrap();
    manifest.commit_stage_success(stage, artifacts).unwrap();
}

fn succeeded_manifest_through(last: ProjectStage) -> ProjectManifest {
    let mut manifest = ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));
    for stage in ProjectStage::ORDER {
        commit_ready_stage(&mut manifest, stage);
        if stage == last {
            break;
        }
    }
    manifest
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

    assert!(json.contains("\"import\""));
    assert!(json.contains("\"running\""));
    assert_eq!(
        serde_json::from_str::<ProjectManifest>(&json).unwrap(),
        manifest
    );
}

#[test]
fn new_manifest_uses_the_declared_schema_and_config_defaults() {
    let source = SourceSpec::managed_images("source-a");
    let manifest = ProjectManifest::new("Flowers", source.clone());

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
}

#[test]
fn succeeded_sfm_becomes_stale_when_keyframes_change() {
    let mut manifest = succeeded_manifest_through(ProjectStage::Training);

    manifest.invalidate(ChangeKind::KeyframeSelection);

    assert_eq!(
        manifest.stage(ProjectStage::KeyframeSfm).state,
        StageState::Stale
    );
    assert_eq!(
        manifest.stage(ProjectStage::FullFramePnp).state,
        StageState::Stale
    );
    assert_eq!(
        manifest.stage(ProjectStage::Training).state,
        StageState::Stale
    );
    assert_eq!(
        manifest.stage(ProjectStage::Import).state,
        StageState::Succeeded
    );
}

#[test]
fn running_cannot_jump_directly_to_paused() {
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
}

#[test]
fn every_declared_generic_transition_is_legal() {
    let legal = [
        (StageState::NotStarted, StageState::Ready),
        (StageState::Ready, StageState::Queued),
        (StageState::Queued, StageState::Running),
        (StageState::Running, StageState::PauseRequested),
        (StageState::PauseRequested, StageState::Paused),
        (StageState::Paused, StageState::Queued),
        (StageState::Running, StageState::CancelRequested),
        (StageState::CancelRequested, StageState::Cancelled),
        (StageState::Cancelled, StageState::Queued),
        (StageState::Running, StageState::Failed),
        (StageState::Failed, StageState::Queued),
        (StageState::Succeeded, StageState::Stale),
        (StageState::Stale, StageState::Ready),
    ];

    for (from, to) in legal {
        let mut manifest = ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));
        manifest.stage_mut(ProjectStage::Import).state = from;
        assert!(
            manifest.transition(ProjectStage::Import, to).is_ok(),
            "expected {from:?} -> {to:?} to be legal"
        );
    }
}

#[test]
fn representative_undeclared_transition_jumps_are_illegal() {
    let illegal = [
        (StageState::NotStarted, StageState::Queued),
        (StageState::Ready, StageState::Running),
        (StageState::Queued, StageState::Paused),
        (StageState::Running, StageState::Paused),
        (StageState::Running, StageState::Succeeded),
        (StageState::Paused, StageState::Succeeded),
        (StageState::CancelRequested, StageState::Queued),
        (StageState::Succeeded, StageState::Queued),
        (StageState::Failed, StageState::Running),
        (StageState::Stale, StageState::Running),
    ];

    for (from, to) in illegal {
        let mut manifest = ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));
        manifest.stage_mut(ProjectStage::Import).state = from;
        assert!(
            manifest.transition(ProjectStage::Import, to).is_err(),
            "expected {from:?} -> {to:?} to be illegal"
        );
    }
}

#[test]
fn successful_stages_make_only_their_direct_dependant_ready() {
    let mut manifest = ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));
    assert_eq!(
        manifest.stage(ProjectStage::Import).state,
        StageState::Ready
    );
    for stage in [
        ProjectStage::KeyframeSfm,
        ProjectStage::FullFramePnp,
        ProjectStage::Training,
        ProjectStage::Complete,
    ] {
        assert_eq!(manifest.stage(stage).state, StageState::NotStarted);
    }
    assert!(manifest
        .transition(ProjectStage::KeyframeSfm, StageState::Ready)
        .is_err());

    for (completed, newly_ready) in [
        (ProjectStage::Import, ProjectStage::KeyframeSfm),
        (ProjectStage::KeyframeSfm, ProjectStage::FullFramePnp),
        (ProjectStage::FullFramePnp, ProjectStage::Training),
        (ProjectStage::Training, ProjectStage::Complete),
    ] {
        commit_ready_stage(&mut manifest, completed);
        assert_eq!(manifest.stage(newly_ready).state, StageState::Ready);
    }
}

#[test]
fn invalidation_follows_each_change_category_dependency_boundary() {
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
        let mut manifest = succeeded_manifest_through(ProjectStage::Complete);
        manifest.invalidate(change);

        let mut invalidated = false;
        for stage in ProjectStage::ORDER {
            if Some(stage) == first_invalidated {
                invalidated = true;
            }
            let expected = if invalidated {
                StageState::Stale
            } else {
                StageState::Succeeded
            };
            assert_eq!(
                manifest.stage(stage).state,
                expected,
                "unexpected state for {stage:?} after {change:?}"
            );
        }
    }
}

#[test]
fn invalidation_readies_incomplete_work_but_leaves_not_started_work_alone() {
    let mut manifest = ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));
    manifest.stage_mut(ProjectStage::KeyframeSfm).state = StageState::Failed;
    manifest.stage_mut(ProjectStage::FullFramePnp).state = StageState::Queued;

    manifest.invalidate(ChangeKind::KeyframeSelection);

    assert_eq!(
        manifest.stage(ProjectStage::KeyframeSfm).state,
        StageState::Ready
    );
    assert_eq!(
        manifest.stage(ProjectStage::FullFramePnp).state,
        StageState::Ready
    );
    assert_eq!(
        manifest.stage(ProjectStage::Training).state,
        StageState::NotStarted
    );
}

#[test]
fn retry_resets_transient_stage_metadata_and_increments_attempt() {
    let mut manifest = ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));
    manifest
        .transition(ProjectStage::Import, StageState::Queued)
        .unwrap();
    assert_eq!(manifest.stage(ProjectStage::Import).attempt, 1);
    manifest
        .transition(ProjectStage::Import, StageState::Running)
        .unwrap();
    {
        let record = manifest.stage_mut(ProjectStage::Import);
        record.completed = Some(7);
        record.total = Some(10);
        record.error = Some(ProjectErrorRecord {
            code: "import_failed".to_owned(),
            stage: ProjectStage::Import,
            summary: "Import failed".to_owned(),
            detail: "Unreadable source".to_owned(),
            frame_id: Some(7),
            pair: None,
            retryable: true,
            suggested_actions: vec![SuggestedAction::Retry],
        });
    }
    manifest
        .transition(ProjectStage::Import, StageState::Failed)
        .unwrap();
    manifest
        .transition(ProjectStage::Import, StageState::Queued)
        .unwrap();

    let retried = manifest.stage(ProjectStage::Import);
    assert_eq!(retried.attempt, 2);
    assert_eq!(retried.completed, None);
    assert_eq!(retried.total, None);
    assert_eq!(retried.started_unix_ms, None);
    assert_eq!(retried.error, None);
}

#[test]
fn stage_success_requires_nonempty_validated_artifacts() {
    let mut manifest = ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));
    manifest
        .transition(ProjectStage::Import, StageState::Queued)
        .unwrap();
    manifest
        .transition(ProjectStage::Import, StageState::Running)
        .unwrap();

    assert!(manifest
        .transition(ProjectStage::Import, StageState::Succeeded)
        .is_err());
    assert!(ValidatedArtifacts::try_new(Vec::new()).is_err());
    assert!(ValidatedArtifacts::try_new(vec![ArtifactRef {
        relative_path: String::new(),
        content_hash: String::new(),
        byte_len: 0,
    }])
    .is_err());

    let artifact = artifact_for(ProjectStage::Import);
    let validated = ValidatedArtifacts::try_new(vec![artifact.clone()]).unwrap();
    manifest
        .commit_stage_success(ProjectStage::Import, validated)
        .unwrap();

    let import = manifest.stage(ProjectStage::Import);
    assert_eq!(import.state, StageState::Succeeded);
    assert_eq!(import.artifacts, vec![artifact]);
    assert_eq!(import.error, None);
    assert_eq!(
        manifest.stage(ProjectStage::KeyframeSfm).state,
        StageState::Ready
    );
}
