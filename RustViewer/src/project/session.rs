use super::{ProjectManifest, ProjectStage, SourceKind, StageState};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectStagePresentation {
    Ready,
    Waiting,
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectSessionSummary {
    pub source_kind: SourceKind,
    pub imported_frame_count: Option<u64>,
    pub import_state: ProjectStagePresentation,
    pub pose_state: ProjectStagePresentation,
    pub pose_detail: String,
    pub training_state: ProjectStagePresentation,
    pub training_detail: String,
}

impl ProjectSessionSummary {
    pub fn from_manifest(manifest: &ProjectManifest) -> Self {
        let import = manifest.stage(ProjectStage::Import);
        let sfm = manifest.stage(ProjectStage::KeyframeSfm);
        let pnp = manifest.stage(ProjectStage::FullFramePnp);
        let training = manifest.stage(ProjectStage::Training);

        let mut summary = Self::from_states(
            manifest.source.kind,
            import.state(),
            sfm.state(),
            pnp.state(),
            training.state(),
        )
        .with_imported_frame_count(import.completed());

        if summary.pose_state == ProjectStagePresentation::Failed {
            if let Some(detail) = [sfm, pnp]
                .into_iter()
                .find(|stage| stage.state() == StageState::Failed)
                .and_then(|stage| stage.error.as_ref())
                .map(|error| error.detail.trim())
                .filter(|detail| !detail.is_empty())
            {
                summary.pose_detail = detail.to_owned();
            }
        }

        summary
    }

    pub fn from_states(
        source_kind: SourceKind,
        import: StageState,
        sfm: StageState,
        pnp: StageState,
        training: StageState,
    ) -> Self {
        let import_state = present_stage(import);
        let (pose_state, pose_detail) = pose_presentation(source_kind, import, sfm, pnp);
        let (training_state, training_detail) = if pose_state != ProjectStagePresentation::Completed
        {
            (
                ProjectStagePresentation::Waiting,
                training_waiting_detail(source_kind).to_owned(),
            )
        } else {
            (
                present_stage(training),
                training_detail(training).to_owned(),
            )
        };

        Self {
            source_kind,
            imported_frame_count: None,
            import_state,
            pose_state,
            pose_detail,
            training_state,
            training_detail,
        }
    }

    fn with_imported_frame_count(mut self, imported_frame_count: Option<u64>) -> Self {
        self.imported_frame_count = imported_frame_count;
        self
    }
}

fn pose_presentation(
    source_kind: SourceKind,
    import: StageState,
    sfm: StageState,
    pnp: StageState,
) -> (ProjectStagePresentation, String) {
    if import != StageState::Succeeded {
        return (
            present_stage(import),
            "Waiting for imported frames".to_owned(),
        );
    }
    if sfm != StageState::Succeeded {
        return (present_stage(sfm), stage_detail("RustSFM", sfm).to_owned());
    }
    if pnp == StageState::Succeeded {
        return (
            ProjectStagePresentation::Completed,
            "All frames have poses".to_owned(),
        );
    }

    (
        present_stage(pnp),
        match source_kind {
            SourceKind::Video => "Waiting for full-frame poses".to_owned(),
            SourceKind::ImageSequence => "Waiting for pose coverage".to_owned(),
        },
    )
}

fn present_stage(state: StageState) -> ProjectStagePresentation {
    match state {
        StageState::Ready => ProjectStagePresentation::Ready,
        StageState::Queued
        | StageState::Running
        | StageState::PauseRequested
        | StageState::CancelRequested => ProjectStagePresentation::Running,
        StageState::Succeeded => ProjectStagePresentation::Completed,
        StageState::Failed => ProjectStagePresentation::Failed,
        StageState::NotStarted | StageState::Paused | StageState::Cancelled | StageState::Stale => {
            ProjectStagePresentation::Waiting
        }
    }
}

fn training_waiting_detail(source_kind: SourceKind) -> &'static str {
    match source_kind {
        SourceKind::Video => "Waiting for full-frame poses",
        SourceKind::ImageSequence => "Waiting for pose coverage",
    }
}

fn training_detail(state: StageState) -> &'static str {
    stage_detail("RustGS", state)
}

fn stage_detail(task: &'static str, state: StageState) -> &'static str {
    match state {
        StageState::NotStarted => "Waiting to start",
        StageState::Ready => "Ready to start",
        StageState::Queued => "Queued",
        StageState::Running => "Running",
        StageState::PauseRequested => "Pausing at a safe boundary",
        StageState::Paused => "Paused",
        StageState::CancelRequested => "Cancelling",
        StageState::Cancelled => "Cancelled",
        StageState::Succeeded => "Completed",
        StageState::Failed => match task {
            "RustSFM" => "RustSFM failed",
            "RustGS" => "RustGS failed",
            _ => "Stage failed",
        },
        StageState::Stale => "Requires rerun",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::{SourceKind, StageState};

    #[test]
    fn video_import_blocks_training_until_full_frame_registration_succeeds() {
        let summary = ProjectSessionSummary::from_states(
            SourceKind::Video,
            StageState::Succeeded,
            StageState::Succeeded,
            StageState::Ready,
            StageState::NotStarted,
        );

        assert_eq!(summary.training_state, ProjectStagePresentation::Waiting);
        assert_eq!(summary.training_detail, "Waiting for full-frame poses");
    }

    #[test]
    fn project_session_failed_pose_uses_persisted_stage_error() {
        let mut manifest = ProjectManifest::new(
            "Failure fixture",
            crate::project::SourceSpec::managed_images("fixture"),
        );
        manifest.stage_mut(ProjectStage::Import).state = StageState::Succeeded;
        let stage = manifest.stage_mut(ProjectStage::KeyframeSfm);
        stage.state = StageState::Failed;
        stage.error = Some(crate::project::ProjectErrorRecord {
            code: "rustsfm_failed".to_owned(),
            stage: ProjectStage::KeyframeSfm,
            summary: "RustSFM pose solve failed".to_owned(),
            detail: "GPU PnP-focal fallback could not solve".to_owned(),
            frame_id: None,
            pair: None,
            retryable: true,
            suggested_actions: Vec::new(),
        });

        let summary = ProjectSessionSummary::from_manifest(&manifest);

        assert_eq!(summary.pose_state, ProjectStagePresentation::Failed);
        assert_eq!(
            summary.pose_detail,
            "GPU PnP-focal fallback could not solve"
        );
    }
}
