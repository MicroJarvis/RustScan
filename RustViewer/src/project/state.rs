use std::path::{Component, Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::manifest::{unix_time_ms, ArtifactRef, ProjectManifest, ProjectStage, StageState};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeKind {
    Source,
    KeyframeSelection,
    SfmConfig,
    PnpConfig,
    TrainingConfig,
    ViewerAppearance,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ArtifactValidationError {
    #[error("a successful stage must commit at least one artifact")]
    Empty,
    #[error("artifact path must be a non-empty relative path without traversal: {0:?}")]
    InvalidRelativePath(String),
    #[error("artifact {relative_path:?} has no content hash")]
    MissingContentHash { relative_path: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedArtifacts {
    artifacts: Vec<ArtifactRef>,
}

impl ValidatedArtifacts {
    pub fn try_new(artifacts: Vec<ArtifactRef>) -> Result<Self, ArtifactValidationError> {
        if artifacts.is_empty() {
            return Err(ArtifactValidationError::Empty);
        }

        for artifact in &artifacts {
            let path = Path::new(&artifact.relative_path);
            if artifact.relative_path.trim().is_empty()
                || path.is_absolute()
                || path
                    .components()
                    .any(|component| !matches!(component, Component::Normal(_)))
            {
                return Err(ArtifactValidationError::InvalidRelativePath(
                    artifact.relative_path.clone(),
                ));
            }
            if artifact.content_hash.trim().is_empty() {
                return Err(ArtifactValidationError::MissingContentHash {
                    relative_path: artifact.relative_path.clone(),
                });
            }
        }

        Ok(Self { artifacts })
    }

    fn into_inner(self) -> Vec<ArtifactRef> {
        self.artifacts
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ProjectStateError {
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
}

impl ProjectManifest {
    pub fn stage(&self, stage: ProjectStage) -> &super::manifest::StageRecord {
        self.stages
            .get(&stage)
            .expect("every manifest must contain every project stage")
    }

    pub fn stage_mut(&mut self, stage: ProjectStage) -> &mut super::manifest::StageRecord {
        self.stages
            .get_mut(&stage)
            .expect("every manifest must contain every project stage")
    }

    pub fn dependencies_ready(&self, stage: ProjectStage) -> bool {
        predecessor(stage)
            .is_none_or(|required| self.stage(required).state == StageState::Succeeded)
    }

    pub fn transition(
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
        let from = self.stage(stage).state;
        #[allow(clippy::match_like_matches_macro)]
        // Keep the transition table as an explicit match.
        let legal = match (from, to) {
            (StageState::NotStarted, StageState::Ready)
            | (StageState::Ready, StageState::Queued)
            | (StageState::Queued, StageState::Running)
            | (StageState::Running, StageState::PauseRequested)
            | (StageState::PauseRequested, StageState::Paused)
            | (StageState::Paused, StageState::Queued)
            | (StageState::Running, StageState::CancelRequested)
            | (StageState::CancelRequested, StageState::Cancelled)
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

        let record = self.stage_mut(stage);
        if to == StageState::Queued {
            record.attempt = record.attempt.saturating_add(1);
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

    pub fn commit_stage_success(
        &mut self,
        stage: ProjectStage,
        validated_artifacts: ValidatedArtifacts,
    ) -> Result<(), ProjectStateError> {
        let from = self.stage(stage).state;
        if from != StageState::Running {
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

    pub fn refresh_readiness(&mut self) {
        self.refresh_readiness_at(unix_time_ms());
    }

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

    pub fn invalidate(&mut self, change: ChangeKind) {
        let first = match change {
            ChangeKind::Source => ProjectStage::Import,
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
            if self.stage(required).state != StageState::Succeeded {
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
