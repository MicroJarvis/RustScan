use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::manifest::{
    unix_time_ms, validate_artifact_ref, ArtifactRef, ArtifactValidationError, ProjectManifest,
    ProjectManifestValidationError, ProjectStage, StageState,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeKind {
    Source,
    ImportConfig,
    KeyframeSelection,
    SfmConfig,
    PnpConfig,
    TrainingConfig,
    ViewerAppearance,
}

impl ChangeKind {
    pub(crate) fn invalidates(self, stage: ProjectStage) -> bool {
        let first = match self {
            Self::Source | Self::ImportConfig => ProjectStage::Import,
            Self::KeyframeSelection | Self::SfmConfig => ProjectStage::KeyframeSfm,
            Self::PnpConfig => ProjectStage::FullFramePnp,
            Self::TrainingConfig => ProjectStage::Training,
            Self::ViewerAppearance => return false,
        };
        let first_index = ProjectStage::ORDER
            .iter()
            .position(|candidate| *candidate == first)
            .expect("declared project stage");
        let stage_index = ProjectStage::ORDER
            .iter()
            .position(|candidate| *candidate == stage)
            .expect("declared project stage");
        stage_index >= first_index
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct ValidatedArtifacts {
    artifacts: Vec<ArtifactRef>,
}

#[cfg_attr(not(test), allow(dead_code))]
impl ValidatedArtifacts {
    pub(crate) fn try_new(artifacts: Vec<ArtifactRef>) -> Result<Self, ArtifactValidationError> {
        if artifacts.is_empty() {
            return Err(ArtifactValidationError::Empty);
        }

        for artifact in &artifacts {
            validate_artifact_ref(artifact)?;
        }

        Ok(Self { artifacts })
    }

    fn into_inner(self) -> Vec<ArtifactRef> {
        self.artifacts
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ProjectStateError {
    #[error(transparent)]
    InvalidManifest(#[from] ProjectManifestValidationError),
    #[error("illegal transition for {stage:?}: {from:?} -> {to:?}")]
    IllegalTransition {
        stage: ProjectStage,
        from: StageState,
        to: StageState,
    },
    #[error("{stage:?} cannot become ready before {predecessor:?} succeeds")]
    DependencyNotReady {
        stage: ProjectStage,
        predecessor: ProjectStage,
    },
    #[error("stage {stage:?} attempt counter overflowed")]
    AttemptOverflow { stage: ProjectStage },
}

impl ProjectManifest {
    pub fn dependencies_ready(&self, stage: ProjectStage) -> bool {
        predecessor(stage).is_none_or(|required| {
            self.try_stage(required)
                .is_ok_and(|record| record.state == StageState::Succeeded)
        })
    }

    /// Applies a legal transition to a manifest that has already passed `validate`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn transition(
        &mut self,
        stage: ProjectStage,
        to: StageState,
    ) -> Result<(), ProjectStateError> {
        self.transition_at(stage, to, unix_time_ms())
    }

    fn transition_at(
        &mut self,
        stage: ProjectStage,
        to: StageState,
        now_unix_ms: u64,
    ) -> Result<(), ProjectStateError> {
        let from = self.try_stage(stage)?.state;
        #[allow(clippy::match_like_matches_macro)]
        // Keep the transition table as an explicit match.
        let legal = match (from, to) {
            (StageState::NotStarted, StageState::Ready)
            | (StageState::Ready, StageState::Queued)
            | (StageState::Ready, StageState::Failed)
            | (StageState::Queued, StageState::Running)
            | (StageState::Running, StageState::PauseRequested)
            | (StageState::PauseRequested, StageState::Paused)
            | (StageState::PauseRequested, StageState::CancelRequested)
            | (StageState::PauseRequested, StageState::Failed)
            | (StageState::Paused, StageState::Queued)
            | (StageState::Running, StageState::CancelRequested)
            | (StageState::CancelRequested, StageState::Cancelled)
            | (StageState::CancelRequested, StageState::Failed)
            | (StageState::Cancelled, StageState::Queued)
            | (StageState::Running, StageState::Failed)
            | (StageState::Failed, StageState::Queued)
            | (StageState::Succeeded, StageState::Stale)
            | (StageState::Stale, StageState::Ready) => true,
            _ => false,
        };
        if !legal {
            return Err(ProjectStateError::IllegalTransition { stage, from, to });
        }

        if matches!(to, StageState::Ready | StageState::Queued) {
            self.require_dependencies(stage)?;
        }

        let next_attempt = if to == StageState::Queued {
            Some(
                self.stage(stage)
                    .attempt
                    .checked_add(1)
                    .ok_or(ProjectStateError::AttemptOverflow { stage })?,
            )
        } else {
            None
        };
        let record = self.stage_mut(stage);
        if let Some(next_attempt) = next_attempt {
            record.attempt = next_attempt;
            record.reset_transient_work();
        } else if to == StageState::Running {
            record.started_unix_ms = Some(now_unix_ms);
        } else if to == StageState::Ready {
            record.reset_transient_work();
        }
        record.state = to;
        record.updated_unix_ms = now_unix_ms;
        self.updated_unix_ms = now_unix_ms;
        Ok(())
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn commit_stage_success(
        &mut self,
        stage: ProjectStage,
        validated_artifacts: ValidatedArtifacts,
    ) -> Result<(), ProjectStateError> {
        let from = self.try_stage(stage)?.state;
        if !matches!(
            from,
            StageState::Running | StageState::PauseRequested | StageState::CancelRequested
        ) {
            return Err(ProjectStateError::IllegalTransition {
                stage,
                from,
                to: StageState::Succeeded,
            });
        }

        let now_unix_ms = unix_time_ms();
        let record = self.stage_mut(stage);
        record.state = StageState::Succeeded;
        record.updated_unix_ms = now_unix_ms;
        record.artifacts = validated_artifacts.into_inner();
        record.error = None;
        if let Some(total) = record.total {
            record.completed = Some(total);
        }
        self.updated_unix_ms = now_unix_ms;
        self.refresh_readiness_at(now_unix_ms);
        Ok(())
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn refresh_readiness_at(&mut self, now_unix_ms: u64) {
        let mut changed = false;
        for stage in ProjectStage::ORDER {
            if self.stage(stage).state == StageState::NotStarted && self.dependencies_ready(stage) {
                let record = self.stage_mut(stage);
                record.state = StageState::Ready;
                record.updated_unix_ms = now_unix_ms;
                changed = true;
            }
        }
        if changed {
            self.updated_unix_ms = now_unix_ms;
        }
    }

    /// Invalidates a manifest that has already passed `validate`.
    pub(crate) fn invalidate(&mut self, change: ChangeKind) {
        let first = match change {
            ChangeKind::Source | ChangeKind::ImportConfig => ProjectStage::Import,
            ChangeKind::KeyframeSelection | ChangeKind::SfmConfig => ProjectStage::KeyframeSfm,
            ChangeKind::PnpConfig => ProjectStage::FullFramePnp,
            ChangeKind::TrainingConfig => ProjectStage::Training,
            ChangeKind::ViewerAppearance => return,
        };
        let now_unix_ms = unix_time_ms();
        let mut affected = false;

        for stage in ProjectStage::ORDER
            .into_iter()
            .skip_while(|stage| *stage != first)
        {
            let state = self.stage(stage).state;
            match state {
                StageState::Succeeded => {
                    let record = self.stage_mut(stage);
                    record.state = StageState::Stale;
                    record.updated_unix_ms = now_unix_ms;
                    affected = true;
                }
                StageState::NotStarted => {}
                _ => {
                    let record = self.stage_mut(stage);
                    record.state = StageState::Ready;
                    record.updated_unix_ms = now_unix_ms;
                    record.reset_transient_work();
                    affected = true;
                }
            }
        }

        if affected {
            self.updated_unix_ms = now_unix_ms;
        }
    }

    fn require_dependencies(&self, stage: ProjectStage) -> Result<(), ProjectStateError> {
        if let Some(required) = predecessor(stage) {
            if self.try_stage(required)?.state != StageState::Succeeded {
                return Err(ProjectStateError::DependencyNotReady {
                    stage,
                    predecessor: required,
                });
            }
        }
        Ok(())
    }
}

fn predecessor(stage: ProjectStage) -> Option<ProjectStage> {
    match stage {
        ProjectStage::Import => None,
        ProjectStage::KeyframeSfm => Some(ProjectStage::Import),
        ProjectStage::FullFramePnp => Some(ProjectStage::KeyframeSfm),
        ProjectStage::Training => Some(ProjectStage::FullFramePnp),
        ProjectStage::Complete => Some(ProjectStage::Training),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::manifest::{ProjectErrorRecord, SourceSpec, SuggestedAction};

    fn artifact_for(stage: ProjectStage) -> ArtifactRef {
        ArtifactRef {
            relative_path: format!(
                "Artifacts/{}/attempt-00000001/result.bin",
                stage.artifact_directory()
            ),
            content_hash: "a".repeat(64),
            byte_len: 0,
        }
    }

    fn validated_artifacts(stage: ProjectStage) -> ValidatedArtifacts {
        ValidatedArtifacts::try_new(vec![artifact_for(stage)]).unwrap()
    }

    fn force_state(manifest: &mut ProjectManifest, stage: ProjectStage, state: StageState) {
        manifest.stage_mut(stage).state = state;
    }

    #[test]
    fn every_declared_generic_transition_is_legal() {
        let legal = [
            (StageState::NotStarted, StageState::Ready),
            (StageState::Ready, StageState::Queued),
            (StageState::Queued, StageState::Running),
            (StageState::Running, StageState::PauseRequested),
            (StageState::PauseRequested, StageState::Paused),
            (StageState::PauseRequested, StageState::CancelRequested),
            (StageState::PauseRequested, StageState::Failed),
            (StageState::Paused, StageState::Queued),
            (StageState::Running, StageState::CancelRequested),
            (StageState::CancelRequested, StageState::Cancelled),
            (StageState::CancelRequested, StageState::Failed),
            (StageState::Cancelled, StageState::Queued),
            (StageState::Running, StageState::Failed),
            (StageState::Failed, StageState::Queued),
            (StageState::Succeeded, StageState::Stale),
            (StageState::Stale, StageState::Ready),
        ];

        for (from, to) in legal {
            let mut manifest =
                ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));
            force_state(&mut manifest, ProjectStage::Import, from);
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
            (StageState::Queued, StageState::Failed),
            (StageState::Running, StageState::Paused),
            (StageState::Running, StageState::Succeeded),
            (StageState::Paused, StageState::Succeeded),
            (StageState::CancelRequested, StageState::Queued),
            (StageState::Succeeded, StageState::Queued),
            (StageState::Failed, StageState::Running),
            (StageState::Stale, StageState::Running),
        ];

        for (from, to) in illegal {
            let mut manifest =
                ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));
            force_state(&mut manifest, ProjectStage::Import, from);
            assert!(
                manifest.transition(ProjectStage::Import, to).is_err(),
                "expected {from:?} -> {to:?} to be illegal"
            );
        }
    }

    #[test]
    fn dependency_readiness_rejects_early_work_and_promotes_only_the_direct_successor() {
        let mut manifest = ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));

        assert_eq!(
            manifest.transition(ProjectStage::KeyframeSfm, StageState::Ready),
            Err(ProjectStateError::DependencyNotReady {
                stage: ProjectStage::KeyframeSfm,
                predecessor: ProjectStage::Import,
            })
        );
        force_state(&mut manifest, ProjectStage::KeyframeSfm, StageState::Ready);
        assert_eq!(
            manifest.transition(ProjectStage::KeyframeSfm, StageState::Queued),
            Err(ProjectStateError::DependencyNotReady {
                stage: ProjectStage::KeyframeSfm,
                predecessor: ProjectStage::Import,
            })
        );
        force_state(
            &mut manifest,
            ProjectStage::KeyframeSfm,
            StageState::NotStarted,
        );

        manifest
            .transition(ProjectStage::Import, StageState::Queued)
            .unwrap();
        manifest
            .transition(ProjectStage::Import, StageState::Running)
            .unwrap();
        manifest
            .commit_stage_success(
                ProjectStage::Import,
                validated_artifacts(ProjectStage::Import),
            )
            .unwrap();

        assert_eq!(
            manifest.stage(ProjectStage::Import).state,
            StageState::Succeeded
        );
        assert_eq!(
            manifest.stage(ProjectStage::KeyframeSfm).state,
            StageState::Ready
        );
        for stage in [
            ProjectStage::FullFramePnp,
            ProjectStage::Training,
            ProjectStage::Complete,
        ] {
            assert_eq!(manifest.stage(stage).state, StageState::NotStarted);
            assert!(!manifest.dependencies_ready(stage));
        }
    }

    #[test]
    fn invalidation_handles_incomplete_states_and_viewer_appearance_is_a_no_op() {
        let mut manifest = ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));
        manifest
            .transition(ProjectStage::Import, StageState::Queued)
            .unwrap();
        manifest
            .transition(ProjectStage::Import, StageState::Running)
            .unwrap();
        manifest
            .commit_stage_success(
                ProjectStage::Import,
                validated_artifacts(ProjectStage::Import),
            )
            .unwrap();
        force_state(&mut manifest, ProjectStage::KeyframeSfm, StageState::Failed);
        force_state(
            &mut manifest,
            ProjectStage::FullFramePnp,
            StageState::Queued,
        );
        force_state(
            &mut manifest,
            ProjectStage::Training,
            StageState::NotStarted,
        );
        force_state(
            &mut manifest,
            ProjectStage::Complete,
            StageState::PauseRequested,
        );

        let before_viewer_change = manifest.clone();
        manifest.invalidate(ChangeKind::ViewerAppearance);
        assert_eq!(manifest, before_viewer_change);

        manifest.invalidate(ChangeKind::KeyframeSelection);
        assert_eq!(
            manifest.stage(ProjectStage::Import).state,
            StageState::Succeeded
        );
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
        assert_eq!(
            manifest.stage(ProjectStage::Complete).state,
            StageState::Ready
        );
    }

    #[test]
    fn invalidation_follows_every_change_category_dependency_boundary() {
        let cases = [
            (ChangeKind::Source, Some(ProjectStage::Import)),
            (ChangeKind::ImportConfig, Some(ProjectStage::Import)),
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
            let mut manifest =
                ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));
            for stage in ProjectStage::ORDER {
                let record = manifest.stage_mut(stage);
                record.state = StageState::Succeeded;
                record.attempt = 1;
                record.artifacts = vec![artifact_for(stage)];
            }
            manifest.invalidate(change);

            let mut invalidated = false;
            for stage in ProjectStage::ORDER {
                if Some(stage) == first_invalidated {
                    invalidated = true;
                }
                assert_eq!(
                    manifest.stage(stage).state,
                    if invalidated {
                        StageState::Stale
                    } else {
                        StageState::Succeeded
                    },
                    "unexpected state for {stage:?} after {change:?}"
                );
            }
        }
    }

    #[test]
    fn success_commit_accepts_running_and_both_request_races() {
        for terminal_race in [
            StageState::Running,
            StageState::PauseRequested,
            StageState::CancelRequested,
        ] {
            let mut manifest =
                ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));
            manifest
                .transition(ProjectStage::Import, StageState::Queued)
                .unwrap();
            manifest
                .transition(ProjectStage::Import, StageState::Running)
                .unwrap();
            if terminal_race != StageState::Running {
                manifest
                    .transition(ProjectStage::Import, terminal_race)
                    .unwrap();
            }

            manifest
                .commit_stage_success(
                    ProjectStage::Import,
                    validated_artifacts(ProjectStage::Import),
                )
                .unwrap();

            assert_eq!(
                manifest.stage(ProjectStage::Import).state,
                StageState::Succeeded
            );
        }
    }

    #[test]
    fn success_commit_rejects_every_other_starting_state() {
        for from in [
            StageState::NotStarted,
            StageState::Ready,
            StageState::Queued,
            StageState::Paused,
            StageState::Cancelled,
            StageState::Succeeded,
            StageState::Failed,
            StageState::Stale,
        ] {
            let mut manifest =
                ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));
            force_state(&mut manifest, ProjectStage::Import, from);
            assert!(manifest
                .commit_stage_success(
                    ProjectStage::Import,
                    validated_artifacts(ProjectStage::Import),
                )
                .is_err());
        }
    }

    #[test]
    fn validated_artifacts_require_canonical_hashes_and_safe_paths() {
        assert!(ValidatedArtifacts::try_new(Vec::new()).is_err());
        for artifact in [
            ArtifactRef {
                relative_path: "../scene.ply".to_owned(),
                content_hash: "a".repeat(64),
                byte_len: 0,
            },
            ArtifactRef {
                relative_path: "scene.ply".to_owned(),
                content_hash: "A".repeat(64),
                byte_len: 0,
            },
        ] {
            assert!(ValidatedArtifacts::try_new(vec![artifact]).is_err());
        }
    }

    #[test]
    fn queue_attempt_overflow_is_an_error_and_does_not_change_the_record() {
        let mut manifest = ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));
        manifest.stage_mut(ProjectStage::Import).attempt = u32::MAX;

        assert_eq!(
            manifest.transition(ProjectStage::Import, StageState::Queued),
            Err(ProjectStateError::AttemptOverflow {
                stage: ProjectStage::Import
            })
        );
        let record = manifest.stage(ProjectStage::Import);
        assert_eq!(record.state, StageState::Ready);
        assert_eq!(record.attempt, u32::MAX);
    }

    #[test]
    fn request_failure_preserves_attempt_artifacts_and_attached_error() {
        for requested in [StageState::PauseRequested, StageState::CancelRequested] {
            let mut manifest =
                ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));
            manifest
                .transition(ProjectStage::Import, StageState::Queued)
                .unwrap();
            manifest
                .transition(ProjectStage::Import, StageState::Running)
                .unwrap();
            manifest
                .transition(ProjectStage::Import, requested)
                .unwrap();
            let error = ProjectErrorRecord {
                code: "worker_failed".to_owned(),
                stage: ProjectStage::Import,
                summary: "Worker failed".to_owned(),
                detail: "Failure after request".to_owned(),
                frame_id: None,
                pair: None,
                retryable: true,
                suggested_actions: vec![SuggestedAction::Retry],
            };
            let record = manifest.stage_mut(ProjectStage::Import);
            record.artifacts = vec![artifact_for(ProjectStage::Import)];
            record.error = Some(error.clone());
            record.updated_unix_ms = 0;
            let attempt = record.attempt;

            manifest
                .transition(ProjectStage::Import, StageState::Failed)
                .unwrap();

            let failed = manifest.stage(ProjectStage::Import);
            assert_eq!(failed.state, StageState::Failed);
            assert_eq!(failed.attempt, attempt);
            assert_eq!(failed.artifacts, vec![artifact_for(ProjectStage::Import)]);
            assert_eq!(failed.error, Some(error));
            assert!(failed.updated_unix_ms > 0);
        }
    }
}
