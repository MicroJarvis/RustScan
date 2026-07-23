use std::fs;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Receiver, Sender, TryRecvError, TrySendError};
use thiserror::Error;

use crate::pipeline::{
    ArtifactValidation, PendingArtifact, PipelineCommand, PipelineEvent, PipelineWorkers,
    ShutdownDisposition, StageRequest, WorkerControl, WorkerEventSink, WorkerOutcome,
};
use crate::project::artifacts::{ArtifactValidationKind, StageWorkspace, StagedArtifact};
use crate::project::{
    ProjectErrorRecord, ProjectStage, ProjectStore, ProjectStoreError, StageState, SuggestedAction,
};

const EVENT_CAPACITY: usize = 64;
const COMMAND_CAPACITY: usize = 8;

#[derive(Debug, Error)]
pub enum PipelineCoordinatorError {
    #[error(transparent)]
    Store(#[from] ProjectStoreError),
    #[error("pipeline command queue is full")]
    CommandQueueFull,
    #[error("pipeline command queue is closed")]
    CommandQueueClosed,
    #[error("pipeline worker thread panicked")]
    WorkerPanicked,
    #[error("worker artifact path is invalid: {0}")]
    InvalidArtifact(String),
    #[error("worker artifact I/O failed: {0}")]
    Io(#[from] std::io::Error),
}

pub struct PipelineCoordinator {
    store: ProjectStore,
    workers: PipelineWorkers,
    command_sender: Sender<PipelineCommand>,
    command_receiver: Receiver<PipelineCommand>,
    event_sender: Sender<PipelineEvent>,
    event_receiver: Receiver<PipelineEvent>,
    worker_event_sender: Sender<PipelineEvent>,
    worker_event_receiver: Receiver<PipelineEvent>,
    active: Option<ActiveWorker>,
    automatic: bool,
    max_concurrent_workers: usize,
    last_progress_persisted: Option<Instant>,
}

struct ActiveWorker {
    stage: ProjectStage,
    workspace: StageWorkspace,
    control: WorkerControl,
    outcome_receiver: Receiver<WorkerOutcome>,
    handle: Option<JoinHandle<()>>,
}

impl PipelineCoordinator {
    pub fn new(
        mut store: ProjectStore,
        workers: PipelineWorkers,
    ) -> Result<Self, PipelineCoordinatorError> {
        store.recover_interrupted_stage()?;
        let (command_sender, command_receiver) = bounded(COMMAND_CAPACITY);
        let (event_sender, event_receiver) = bounded(EVENT_CAPACITY);
        let (worker_event_sender, worker_event_receiver) = bounded(EVENT_CAPACITY);
        Ok(Self {
            store,
            workers,
            command_sender,
            command_receiver,
            event_sender,
            event_receiver,
            worker_event_sender,
            worker_event_receiver,
            active: None,
            automatic: false,
            max_concurrent_workers: 0,
            last_progress_persisted: None,
        })
    }

    pub fn store(&self) -> &ProjectStore {
        &self.store
    }
    pub fn store_mut(&mut self) -> &mut ProjectStore {
        &mut self.store
    }
    pub fn max_concurrent_workers(&self) -> usize {
        self.max_concurrent_workers
    }

    pub fn send(&self, command: PipelineCommand) -> Result<(), PipelineCoordinatorError> {
        self.command_sender
            .try_send(command)
            .map_err(|error| match error {
                TrySendError::Full(_) => PipelineCoordinatorError::CommandQueueFull,
                TrySendError::Disconnected(_) => PipelineCoordinatorError::CommandQueueClosed,
            })
    }

    pub fn try_next_event(&self) -> Option<PipelineEvent> {
        self.event_receiver.try_recv().ok()
    }

    pub fn drive_until_idle(&mut self) -> Result<(), PipelineCoordinatorError> {
        loop {
            self.drive_once()?;
            if self.active.is_none() && !self.automatic && self.command_receiver.is_empty() {
                let _ = self.event_sender.try_send(PipelineEvent::Idle);
                return Ok(());
            }
            thread::sleep(Duration::from_millis(1));
        }
    }

    pub fn drive_once(&mut self) -> Result<(), PipelineCoordinatorError> {
        while let Ok(command) = self.command_receiver.try_recv() {
            self.apply_command(command)?;
        }
        self.forward_worker_events()?;
        if self.finish_active_worker()? {
            return Ok(());
        }
        if self.active.is_none() && self.automatic {
            if !self.start_next_ready_stage()? {
                self.automatic = false;
            }
        }
        Ok(())
    }

    fn apply_command(&mut self, command: PipelineCommand) -> Result<(), PipelineCoordinatorError> {
        match command {
            PipelineCommand::StartAutomatic => {
                if self.active.is_none() {
                    self.automatic = true;
                }
            }
            PipelineCommand::Pause => {
                self.automatic = false;
                if let Some(active) = &self.active {
                    self.store.request_stage_pause(active.stage)?;
                    active.control.request_pause();
                }
            }
            PipelineCommand::Cancel => {
                self.automatic = false;
                if let Some(active) = &self.active {
                    self.store.request_stage_cancel(active.stage)?;
                    active.control.request_cancel();
                }
            }
            PipelineCommand::Retry { stage } => {
                if self.active.is_none()
                    && self
                        .store
                        .manifest()
                        .try_stage(stage)
                        .map(|record| record.state() == StageState::Failed)
                        .unwrap_or(false)
                {
                    self.automatic = false;
                    self.start_stage(stage)?;
                }
            }
            PipelineCommand::RestartFrom { stage, confirmed } => {
                if confirmed && self.active.is_none() {
                    self.store.restart_from_stage(stage)?;
                    self.automatic = true;
                }
            }
            PipelineCommand::Shutdown { disposition } => match disposition {
                ShutdownDisposition::PauseAndQuit => self.apply_command(PipelineCommand::Pause)?,
                ShutdownDisposition::CancelAndQuit => {
                    self.apply_command(PipelineCommand::Cancel)?
                }
                ShutdownDisposition::KeepRunning => {}
            },
        }
        self.emit_manifest();
        Ok(())
    }

    fn start_next_ready_stage(&mut self) -> Result<bool, PipelineCoordinatorError> {
        for stage in [
            ProjectStage::Import,
            ProjectStage::KeyframeSfm,
            ProjectStage::FullFramePnp,
            ProjectStage::Training,
        ] {
            if self
                .store
                .manifest()
                .try_stage(stage)
                .map(|record| matches!(record.state(), StageState::Ready | StageState::Stale))
                .unwrap_or(false)
            {
                self.start_stage(stage)?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn start_stage(&mut self, stage: ProjectStage) -> Result<(), PipelineCoordinatorError> {
        if self.active.is_some() {
            return Ok(());
        }
        let workspace = self.store.begin_stage(stage)?;
        let request = StageRequest {
            stage,
            attempt: workspace.attempt(),
            manifest: self.store.manifest().clone(),
        };
        let control = WorkerControl::new();
        let sink = WorkerEventSink::new(stage, self.worker_event_sender.clone());
        let (outcome_sender, outcome_receiver) = bounded(1);
        let workers = self.workers.clone();
        let worker_control = control.clone();
        let handle = thread::Builder::new()
            .name(format!("rustscan-{stage:?}"))
            .spawn(move || {
                let outcome = match stage {
                    ProjectStage::Import => workers.import.run(request, worker_control, sink),
                    ProjectStage::KeyframeSfm => workers.sfm.run(request, worker_control, sink),
                    ProjectStage::FullFramePnp => workers.pnp.run(request, worker_control, sink),
                    ProjectStage::Training => workers.training.run(request, worker_control, sink),
                    ProjectStage::Complete => unreachable!("complete has no worker"),
                };
                let _ = outcome_sender.send(outcome);
            })?;
        self.active = Some(ActiveWorker {
            stage,
            workspace,
            control,
            outcome_receiver,
            handle: Some(handle),
        });
        self.max_concurrent_workers = self.max_concurrent_workers.max(1);
        self.emit_manifest();
        Ok(())
    }

    fn finish_active_worker(&mut self) -> Result<bool, PipelineCoordinatorError> {
        let Some(active) = self.active.as_ref() else {
            return Ok(false);
        };
        let outcome = match active.outcome_receiver.try_recv() {
            Ok(outcome) => outcome,
            Err(TryRecvError::Empty) => return Ok(false),
            Err(TryRecvError::Disconnected) => {
                return Err(PipelineCoordinatorError::WorkerPanicked)
            }
        };
        let mut active = self.active.take().expect("active worker");
        active
            .handle
            .take()
            .expect("worker handle")
            .join()
            .map_err(|_| PipelineCoordinatorError::WorkerPanicked)?;
        match outcome {
            WorkerOutcome::Succeeded(artifacts) => self.commit_success(active, artifacts)?,
            WorkerOutcome::Paused(artifacts) => {
                self.retain_pending_artifacts(&active.workspace, artifacts)?;
                self.store.mark_stage_paused(active.stage)?;
            }
            WorkerOutcome::Cancelled(artifacts) => {
                self.retain_pending_artifacts(&active.workspace, artifacts)?;
                self.store.mark_stage_cancelled(active.stage)?;
            }
            WorkerOutcome::Failed(error) => self.store.mark_stage_failed(active.stage, error)?,
        }
        self.emit_manifest();
        Ok(true)
    }

    fn commit_success(
        &mut self,
        active: ActiveWorker,
        artifacts: Vec<PendingArtifact>,
    ) -> Result<(), PipelineCoordinatorError> {
        if artifacts.is_empty() {
            return self
                .store
                .mark_stage_failed(
                    active.stage,
                    worker_error(active.stage, "worker returned no artifacts"),
                )
                .map_err(Into::into);
        }
        for artifact in &artifacts {
            if let ArtifactValidation::PnpCoverage {
                imported_frames,
                registered_frames,
            } = artifact.validation
            {
                if imported_frames != registered_frames {
                    let error =
                        worker_error(active.stage, "PnP did not register every imported frame");
                    self.store.mark_stage_failed(active.stage, error.clone())?;
                    let _ = self
                        .event_sender
                        .try_send(PipelineEvent::NeedsAttention(error));
                    self.automatic = false;
                    return Ok(());
                }
            }
        }
        let declarations = self.retain_pending_artifacts(&active.workspace, artifacts)?;
        self.store
            .commit_stage_success(&active.workspace, &declarations, false)?;
        Ok(())
    }

    fn retain_pending_artifacts(
        &self,
        workspace: &StageWorkspace,
        artifacts: Vec<PendingArtifact>,
    ) -> Result<Vec<StagedArtifact>, PipelineCoordinatorError> {
        let mut declarations = Vec::with_capacity(artifacts.len());
        for artifact in artifacts {
            let kind = match artifact.validation {
                ArtifactValidation::Json | ArtifactValidation::PnpCoverage { .. } => {
                    ArtifactValidationKind::Json
                }
                ArtifactValidation::ReadableFile => ArtifactValidationKind::ReadableFile,
            };
            let declaration = StagedArtifact::new(&artifact.relative_path, kind)
                .map_err(|error| PipelineCoordinatorError::InvalidArtifact(error.to_string()))?;
            let path = workspace.path().join(&artifact.relative_path);
            let parent = path.parent().ok_or_else(|| {
                PipelineCoordinatorError::InvalidArtifact(artifact.relative_path.clone())
            })?;
            fs::create_dir_all(parent)?;
            fs::write(path, artifact.payload)?;
            declarations.push(declaration);
        }
        if !declarations.is_empty() {
            self.store
                .validate_stage_payloads(workspace, &declarations)?;
        }
        Ok(declarations)
    }

    fn forward_worker_events(&mut self) -> Result<(), PipelineCoordinatorError> {
        while let Ok(event) = self.worker_event_receiver.try_recv() {
            if let PipelineEvent::StageProgress {
                stage,
                completed,
                total,
                ..
            } = event
            {
                if self
                    .last_progress_persisted
                    .is_none_or(|last| last.elapsed() >= Duration::from_secs(1))
                {
                    if completed.is_some() || total.is_some() {
                        self.store.record_stage_progress(stage, completed, total)?;
                        self.last_progress_persisted = Some(Instant::now());
                    }
                }
            }
            let _ = self.event_sender.try_send(event);
        }
        Ok(())
    }

    fn emit_manifest(&self) {
        let _ = self.event_sender.try_send(PipelineEvent::ManifestChanged(
            self.store.manifest().clone(),
        ));
    }
}

fn worker_error(stage: ProjectStage, detail: &str) -> ProjectErrorRecord {
    ProjectErrorRecord {
        code: "pipeline_worker_failed".to_owned(),
        stage,
        summary: "Pipeline worker failed".to_owned(),
        detail: detail.to_owned(),
        frame_id: None,
        pair: None,
        retryable: true,
        suggested_actions: vec![SuggestedAction::Retry],
    }
}
