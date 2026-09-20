use std::path::PathBuf;

use std::sync::Arc;

use crossbeam_channel::Sender;

use crate::pipeline::{PipelineEvent, PipelineProgressDetail};
use crate::project::{ProjectErrorRecord, ProjectManifest, ProjectStage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactValidation {
    Json,
    ReadableFile,
    PnpCoverage {
        imported_frames: usize,
        registered_frames: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingArtifact {
    pub relative_path: String,
    pub payload: Vec<u8>,
    pub validation: ArtifactValidation,
}

impl PendingArtifact {
    pub fn new(
        relative_path: impl Into<String>,
        payload: Vec<u8>,
        validation: ArtifactValidation,
    ) -> Self {
        Self {
            relative_path: relative_path.into(),
            payload,
            validation,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerOutcome {
    Succeeded(Vec<PendingArtifact>),
    Paused(Vec<PendingArtifact>),
    Cancelled(Vec<PendingArtifact>),
    Failed(ProjectErrorRecord),
}

#[derive(Debug, Clone)]
pub struct StageRequest {
    pub stage: ProjectStage,
    pub attempt: u32,
    pub project_root: PathBuf,
    pub workspace_path: PathBuf,
    pub manifest: ProjectManifest,
}

#[derive(Debug, Clone)]
pub struct WorkerControl(rustscan_sfm::SfmTaskControl);

impl WorkerControl {
    pub(crate) fn new() -> Self {
        Self(rustscan_sfm::SfmTaskControl::new())
    }
    pub(crate) fn request_pause(&self) {
        self.0.request_pause();
    }
    pub(crate) fn request_cancel(&self) {
        self.0.request_cancel();
    }
    pub(crate) fn sfm_control(&self) -> rustscan_sfm::SfmTaskControl {
        self.0.clone()
    }
    pub fn pause_requested(&self) -> bool {
        self.0.state() == rustscan_sfm::SfmControlState::PauseRequested
    }
    pub fn cancel_requested(&self) -> bool {
        self.0.state() == rustscan_sfm::SfmControlState::CancelRequested
    }
}

#[derive(Clone)]
pub struct WorkerEventSink {
    stage: ProjectStage,
    attempt: u32,
    sender: Sender<PipelineEvent>,
}

impl WorkerEventSink {
    pub(crate) fn new(stage: ProjectStage, attempt: u32, sender: Sender<PipelineEvent>) -> Self {
        Self {
            stage,
            attempt,
            sender,
        }
    }
    pub fn progress(
        &self,
        completed: Option<u64>,
        total: Option<u64>,
        detail: PipelineProgressDetail,
    ) {
        let _ = self.sender.try_send(PipelineEvent::StageProgress {
            stage: self.stage,
            attempt: self.attempt,
            completed,
            total,
            detail,
        });
    }
    pub fn scene_snapshot(&self, splats: Arc<rustscan_gs::HostSplats>) {
        let _ = self.sender.try_send(PipelineEvent::SceneSnapshot(splats));
    }
}

pub trait ImportWorker: Send + Sync + 'static {
    fn run(
        &self,
        request: StageRequest,
        control: WorkerControl,
        events: WorkerEventSink,
    ) -> WorkerOutcome;
}
pub trait SfmWorker: Send + Sync + 'static {
    fn run(
        &self,
        request: StageRequest,
        control: WorkerControl,
        events: WorkerEventSink,
    ) -> WorkerOutcome;
}
pub trait PnpWorker: Send + Sync + 'static {
    fn run(
        &self,
        request: StageRequest,
        control: WorkerControl,
        events: WorkerEventSink,
    ) -> WorkerOutcome;
}
pub trait TrainingWorker: Send + Sync + 'static {
    fn run(
        &self,
        request: StageRequest,
        control: WorkerControl,
        events: WorkerEventSink,
    ) -> WorkerOutcome;
}

#[derive(Clone)]
pub struct PipelineWorkers {
    pub(crate) import: Arc<dyn ImportWorker>,
    pub(crate) sfm: Arc<dyn SfmWorker>,
    pub(crate) pnp: Arc<dyn PnpWorker>,
    pub(crate) training: Arc<dyn TrainingWorker>,
}

impl PipelineWorkers {
    pub fn new<I, S, P, T>(import: I, sfm: S, pnp: P, training: T) -> Self
    where
        I: ImportWorker,
        S: SfmWorker,
        P: PnpWorker,
        T: TrainingWorker,
    {
        Self {
            import: Arc::new(import),
            sfm: Arc::new(sfm),
            pnp: Arc::new(pnp),
            training: Arc::new(training),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sfm_control_observes_stop_without_progress_events_and_cancel_is_final() {
        let control = WorkerControl::new();
        let sfm = control.sfm_control();
        control.request_pause();
        assert_eq!(sfm.checkpoint(), Err(rustscan_sfm::SfmTaskStop::Paused));
        let peer = control.clone();
        std::thread::spawn(move || peer.request_cancel())
            .join()
            .unwrap();
        assert_eq!(sfm.checkpoint(), Err(rustscan_sfm::SfmTaskStop::Cancelled));
        control.request_pause();
        assert!(control.cancel_requested());
        assert!(!control.pause_requested());
    }
}
