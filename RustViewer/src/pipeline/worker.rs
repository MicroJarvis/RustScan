use std::sync::atomic::{AtomicU8, Ordering};
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
    pub manifest: ProjectManifest,
}

#[derive(Debug, Clone)]
pub struct WorkerControl(Arc<AtomicU8>);

impl WorkerControl {
    pub(crate) fn new() -> Self {
        Self(Arc::new(AtomicU8::new(0)))
    }
    pub(crate) fn request_pause(&self) {
        self.0.store(1, Ordering::Release);
    }
    pub(crate) fn request_cancel(&self) {
        self.0.store(2, Ordering::Release);
    }
    pub fn pause_requested(&self) -> bool {
        self.0.load(Ordering::Acquire) == 1
    }
    pub fn cancel_requested(&self) -> bool {
        self.0.load(Ordering::Acquire) == 2
    }
}

#[derive(Clone)]
pub struct WorkerEventSink {
    stage: ProjectStage,
    sender: Sender<PipelineEvent>,
}

impl WorkerEventSink {
    pub(crate) fn new(stage: ProjectStage, sender: Sender<PipelineEvent>) -> Self {
        Self { stage, sender }
    }
    pub fn progress(
        &self,
        completed: Option<u64>,
        total: Option<u64>,
        detail: PipelineProgressDetail,
    ) {
        let _ = self.sender.try_send(PipelineEvent::StageProgress {
            stage: self.stage,
            completed,
            total,
            detail,
        });
    }
    pub fn scene_snapshot(&self, splats: Arc<rustgs::HostSplats>) {
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
