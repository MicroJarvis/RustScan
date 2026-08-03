use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender};
use rust_viewer::pipeline::{
    ArtifactValidation, ImportWorker, PendingArtifact, PipelineCommand, PipelineCoordinator,
    PipelineEvent, PipelineProgressDetail, PipelineWorkers, PnpWorker, SfmWorker, StageRequest,
    TrainingWorker, WorkerControl, WorkerEventSink, WorkerOutcome,
};
use rust_viewer::project::{
    ProjectCreateRequest, ProjectStage, ProjectStore, SourceSpec, StageState,
};
use rustgs::HostSplats;

#[test]
fn worker_request_includes_the_locked_project_root_and_stage_workspace() {
    let temporary = tempfile::tempdir().unwrap();
    let project_root = temporary.path().join("fixture.rustscanproject");
    let store = ProjectStore::create(
        &project_root,
        ProjectCreateRequest::new("Fixture", SourceSpec::managed_images("fixture")),
    )
    .unwrap();
    let (request_sender, request_receiver) = mpsc::channel();
    let trace = Arc::new(Mutex::new(Vec::new()));
    let workers = PipelineWorkers::new(
        FakeStageWorker::new(ProjectStage::Import, Arc::clone(&trace)),
        CapturingSfmWorker { request_sender },
        FakeStageWorker::new(ProjectStage::FullFramePnp, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::Training, trace),
    );
    let mut coordinator = PipelineCoordinator::new(store, workers).unwrap();

    coordinator.send(PipelineCommand::StartAutomatic).unwrap();
    coordinator.drive_until_idle().unwrap();

    let request = request_receiver.recv().unwrap();
    assert!(request.project_root.ends_with("fixture.rustscanproject"));
    assert!(request.workspace_path.starts_with(&request.project_root));
    assert!(request
        .workspace_path
        .ends_with("Cache/.staging/keyframe_sfm-1"));
}

#[test]
fn automatic_pipeline_runs_stages_in_order_with_one_worker_at_a_time() {
    let temporary = tempfile::tempdir().unwrap();
    let store = ProjectStore::create(
        temporary.path().join("Pipeline.rustscanproject"),
        ProjectCreateRequest::new("Pipeline", SourceSpec::managed_images("fixture")),
    )
    .unwrap();
    let trace = Arc::new(Mutex::new(Vec::new()));
    let workers = PipelineWorkers::new(
        FakeStageWorker::new(ProjectStage::Import, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::KeyframeSfm, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::FullFramePnp, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::Training, Arc::clone(&trace)),
    );
    let mut coordinator = PipelineCoordinator::new(store, workers).unwrap();

    coordinator.send(PipelineCommand::StartAutomatic).unwrap();
    coordinator.send(PipelineCommand::StartAutomatic).unwrap();
    coordinator.drive_until_idle().unwrap();

    assert_eq!(
        coordinator
            .store()
            .manifest()
            .try_stage(ProjectStage::Training)
            .unwrap()
            .state(),
        StageState::Succeeded
    );
    assert_eq!(
        *trace.lock().unwrap(),
        [
            ProjectStage::Import,
            ProjectStage::KeyframeSfm,
            ProjectStage::FullFramePnp,
            ProjectStage::Training,
        ]
    );
    assert_eq!(coordinator.max_concurrent_workers(), 1);
}

#[test]
fn reconstruction_target_stops_after_full_frame_pnp() {
    let temporary = tempfile::tempdir().unwrap();
    let store = ProjectStore::create(
        temporary.path().join("Target.rustscanproject"),
        ProjectCreateRequest::new("Target", SourceSpec::managed_images("fixture")),
    )
    .unwrap();
    let trace = Arc::new(Mutex::new(Vec::new()));
    let workers = PipelineWorkers::new(
        FakeStageWorker::new(ProjectStage::Import, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::KeyframeSfm, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::FullFramePnp, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::Training, Arc::clone(&trace)),
    );
    let mut coordinator = PipelineCoordinator::new(store, workers).unwrap();

    coordinator
        .send(PipelineCommand::StartThrough {
            stage: ProjectStage::FullFramePnp,
        })
        .unwrap();
    coordinator.drive_until_idle().unwrap();

    assert_eq!(
        *trace.lock().unwrap(),
        [
            ProjectStage::Import,
            ProjectStage::KeyframeSfm,
            ProjectStage::FullFramePnp,
        ]
    );
    assert_eq!(
        coordinator
            .store()
            .manifest()
            .try_stage(ProjectStage::Training)
            .unwrap()
            .state(),
        StageState::Ready
    );
}

#[test]
fn reconstruction_pipeline_stops_after_full_frame_pose_coverage() {
    let temporary = tempfile::tempdir().unwrap();
    let store = ProjectStore::create(
        temporary.path().join("Reconstruction.rustscanproject"),
        ProjectCreateRequest::new("Reconstruction", SourceSpec::managed_images("fixture")),
    )
    .unwrap();
    let trace = Arc::new(Mutex::new(Vec::new()));
    let workers = PipelineWorkers::new(
        FakeStageWorker::new(ProjectStage::Import, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::KeyframeSfm, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::FullFramePnp, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::Training, Arc::clone(&trace)),
    );
    let mut coordinator = PipelineCoordinator::new(store, workers).unwrap();

    coordinator
        .send(PipelineCommand::StartThrough {
            stage: ProjectStage::FullFramePnp,
        })
        .unwrap();
    coordinator.drive_until_idle().unwrap();

    assert_eq!(
        coordinator
            .store()
            .manifest()
            .try_stage(ProjectStage::FullFramePnp)
            .unwrap()
            .state(),
        StageState::Succeeded
    );
    assert_eq!(
        coordinator
            .store()
            .manifest()
            .try_stage(ProjectStage::Training)
            .unwrap()
            .state(),
        StageState::Ready
    );
    assert!(!trace.lock().unwrap().contains(&ProjectStage::Training));
}

#[test]
fn incomplete_pnp_coverage_needs_attention_and_never_starts_training() {
    let temporary = tempfile::tempdir().unwrap();
    let store = ProjectStore::create(
        temporary.path().join("Coverage.rustscanproject"),
        ProjectCreateRequest::new("Coverage", SourceSpec::managed_images("fixture")),
    )
    .unwrap();
    let trace = Arc::new(Mutex::new(Vec::new()));
    let workers = PipelineWorkers::new(
        FakeStageWorker::new(ProjectStage::Import, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::KeyframeSfm, Arc::clone(&trace)),
        IncompletePnpWorker {
            trace: Arc::clone(&trace),
        },
        FakeStageWorker::new(ProjectStage::Training, Arc::clone(&trace)),
    );
    let mut coordinator = PipelineCoordinator::new(store, workers).unwrap();

    coordinator.send(PipelineCommand::StartAutomatic).unwrap();
    coordinator.drive_until_idle().unwrap();

    assert_eq!(
        coordinator
            .store()
            .manifest()
            .try_stage(ProjectStage::FullFramePnp)
            .unwrap()
            .state(),
        StageState::Failed
    );
    assert_eq!(
        coordinator
            .store()
            .manifest()
            .try_stage(ProjectStage::Training)
            .unwrap()
            .state(),
        StageState::NotStarted
    );
    assert!(!trace.lock().unwrap().contains(&ProjectStage::Training));
    assert!(
        std::iter::from_fn(|| coordinator.try_next_event()).any(|event| matches!(
            event,
            rust_viewer::pipeline::PipelineEvent::NeedsAttention(_)
        ))
    );
}

#[test]
fn stale_progress_after_pnp_failure_is_discarded_without_replacing_the_error() {
    let temporary = tempfile::tempdir().unwrap();
    let store = ProjectStore::create(
        temporary.path().join("LateProgress.rustscanproject"),
        ProjectCreateRequest::new("Late progress", SourceSpec::managed_images("fixture")),
    )
    .unwrap();
    let trace = Arc::new(Mutex::new(Vec::new()));
    let (trigger_sender, trigger_receiver) = crossbeam_channel::bounded(1);
    let (emitted_sender, emitted_receiver) = crossbeam_channel::bounded(1);
    let workers = PipelineWorkers::new(
        FakeStageWorker::new(ProjectStage::Import, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::KeyframeSfm, Arc::clone(&trace)),
        LateProgressFailingPnpWorker {
            trigger: trigger_receiver,
            emitted: emitted_sender,
        },
        FakeStageWorker::new(ProjectStage::Training, trace),
    );
    let mut coordinator = PipelineCoordinator::new(store, workers).unwrap();

    coordinator.send(PipelineCommand::StartAutomatic).unwrap();
    coordinator.drive_until_idle().unwrap();
    let pnp = coordinator
        .store()
        .manifest()
        .try_stage(ProjectStage::FullFramePnp)
        .unwrap();
    assert_eq!(pnp.state(), StageState::Failed);
    assert_eq!(pnp.error().unwrap().detail, "terminal PnP error");

    trigger_sender.send(()).unwrap();
    emitted_receiver
        .recv_timeout(Duration::from_secs(1))
        .unwrap();
    coordinator.drive_once().unwrap();

    let pnp = coordinator
        .store()
        .manifest()
        .try_stage(ProjectStage::FullFramePnp)
        .unwrap();
    assert_eq!(pnp.state(), StageState::Failed);
    assert_eq!(pnp.error().unwrap().detail, "terminal PnP error");
    assert!(
        !std::iter::from_fn(|| coordinator.try_next_event()).any(|event| matches!(
            event,
            PipelineEvent::StageProgress {
                stage: ProjectStage::FullFramePnp,
                ..
            }
        ))
    );
}

#[test]
fn stale_progress_from_a_previous_pnp_attempt_is_discarded_during_retry() {
    let temporary = tempfile::tempdir().unwrap();
    let store = ProjectStore::create(
        temporary.path().join("RetryProgress.rustscanproject"),
        ProjectCreateRequest::new("Retry progress", SourceSpec::managed_images("fixture")),
    )
    .unwrap();
    let trace = Arc::new(Mutex::new(Vec::new()));
    let (stale_trigger_sender, stale_trigger_receiver) = crossbeam_channel::bounded(1);
    let (stale_emitted_sender, stale_emitted_receiver) = crossbeam_channel::bounded(1);
    let (current_emitted_sender, current_emitted_receiver) = crossbeam_channel::bounded(1);
    let (finish_sender, finish_receiver) = crossbeam_channel::bounded(1);
    let workers = PipelineWorkers::new(
        FakeStageWorker::new(ProjectStage::Import, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::KeyframeSfm, Arc::clone(&trace)),
        RetriedPnpWorker {
            stale_trigger: stale_trigger_receiver,
            stale_emitted: stale_emitted_sender,
            current_emitted: current_emitted_sender,
            finish: finish_receiver,
        },
        FakeStageWorker::new(ProjectStage::Training, trace),
    );
    let mut coordinator = PipelineCoordinator::new(store, workers).unwrap();

    coordinator.send(PipelineCommand::StartAutomatic).unwrap();
    coordinator.drive_until_idle().unwrap();
    assert_eq!(
        coordinator
            .store()
            .manifest()
            .try_stage(ProjectStage::FullFramePnp)
            .unwrap()
            .state(),
        StageState::Failed
    );

    coordinator
        .send(PipelineCommand::Retry {
            stage: ProjectStage::FullFramePnp,
        })
        .unwrap();
    coordinator.drive_once().unwrap();
    current_emitted_receiver
        .recv_timeout(Duration::from_secs(1))
        .unwrap();
    stale_trigger_sender.send(()).unwrap();
    stale_emitted_receiver
        .recv_timeout(Duration::from_secs(1))
        .unwrap();
    coordinator.drive_once().unwrap();

    let pnp = coordinator
        .store()
        .manifest()
        .try_stage(ProjectStage::FullFramePnp)
        .unwrap();
    assert_eq!(pnp.state(), StageState::Running);
    assert_eq!(pnp.attempt(), 2);
    assert_eq!(pnp.completed(), Some(4));
    assert_eq!(pnp.total(), Some(4));
    let progress = std::iter::from_fn(|| coordinator.try_next_event())
        .filter_map(|event| match event {
            PipelineEvent::StageProgress {
                stage: ProjectStage::FullFramePnp,
                attempt,
                detail: PipelineProgressDetail::Sfm { operation, .. },
                ..
            } => Some((attempt, operation)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(progress, vec![(2, "CurrentAttempt".to_owned())]);

    finish_sender.send(()).unwrap();
    coordinator.drive_until_idle().unwrap();
}

#[test]
fn pause_requests_worker_control_and_persists_paused_state() {
    let temporary = tempfile::tempdir().unwrap();
    let store = ProjectStore::create(
        temporary.path().join("Pause.rustscanproject"),
        ProjectCreateRequest::new("Pause", SourceSpec::managed_images("fixture")),
    )
    .unwrap();
    let trace = Arc::new(Mutex::new(Vec::new()));
    let workers = PipelineWorkers::new(
        PausingImportWorker,
        FakeStageWorker::new(ProjectStage::KeyframeSfm, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::FullFramePnp, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::Training, trace),
    );
    let mut coordinator = PipelineCoordinator::new(store, workers).unwrap();

    coordinator.send(PipelineCommand::StartAutomatic).unwrap();
    coordinator.drive_once().unwrap();
    coordinator.send(PipelineCommand::Pause).unwrap();
    coordinator.drive_until_idle().unwrap();

    assert_eq!(
        coordinator
            .store()
            .manifest()
            .try_stage(ProjectStage::Import)
            .unwrap()
            .state(),
        StageState::Paused
    );
    assert!(coordinator.store().manifest().lease().is_none());
}

#[test]
fn cancel_requests_worker_control_and_stops_automatic_pipeline() {
    let temporary = tempfile::tempdir().unwrap();
    let store = ProjectStore::create(
        temporary.path().join("Cancel.rustscanproject"),
        ProjectCreateRequest::new("Cancel", SourceSpec::managed_images("fixture")),
    )
    .unwrap();
    let trace = Arc::new(Mutex::new(Vec::new()));
    let workers = PipelineWorkers::new(
        PausingImportWorker,
        FakeStageWorker::new(ProjectStage::KeyframeSfm, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::FullFramePnp, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::Training, trace),
    );
    let mut coordinator = PipelineCoordinator::new(store, workers).unwrap();

    coordinator.send(PipelineCommand::StartAutomatic).unwrap();
    coordinator.drive_once().unwrap();
    coordinator.send(PipelineCommand::Cancel).unwrap();
    coordinator.drive_until_idle().unwrap();

    assert_eq!(
        coordinator
            .store()
            .manifest()
            .try_stage(ProjectStage::Import)
            .unwrap()
            .state(),
        StageState::Cancelled
    );
    assert!(coordinator.store().manifest().lease().is_none());
    assert_eq!(
        coordinator
            .store()
            .manifest()
            .try_stage(ProjectStage::KeyframeSfm)
            .unwrap()
            .state(),
        StageState::NotStarted
    );
}

#[test]
fn retry_runs_only_the_failed_stage() {
    let temporary = tempfile::tempdir().unwrap();
    let store = ProjectStore::create(
        temporary.path().join("Retry.rustscanproject"),
        ProjectCreateRequest::new("Retry", SourceSpec::managed_images("fixture")),
    )
    .unwrap();
    let trace = Arc::new(Mutex::new(Vec::new()));
    let failures_remaining = Arc::new(Mutex::new(1));
    let workers = PipelineWorkers::new(
        FakeStageWorker::new(ProjectStage::Import, Arc::clone(&trace)),
        FailingThenSucceedingSfmWorker {
            trace: Arc::clone(&trace),
            failures_remaining: Arc::clone(&failures_remaining),
        },
        FakeStageWorker::new(ProjectStage::FullFramePnp, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::Training, Arc::clone(&trace)),
    );
    let mut coordinator = PipelineCoordinator::new(store, workers).unwrap();

    coordinator.send(PipelineCommand::StartAutomatic).unwrap();
    coordinator.drive_until_idle().unwrap();
    assert_eq!(
        coordinator
            .store()
            .manifest()
            .try_stage(ProjectStage::KeyframeSfm)
            .unwrap()
            .state(),
        StageState::Failed
    );

    coordinator
        .send(PipelineCommand::Retry {
            stage: ProjectStage::KeyframeSfm,
        })
        .unwrap();
    coordinator.drive_until_idle().unwrap();

    assert_eq!(
        *trace.lock().unwrap(),
        [
            ProjectStage::Import,
            ProjectStage::KeyframeSfm,
            ProjectStage::KeyframeSfm
        ]
    );
    assert_eq!(
        coordinator
            .store()
            .manifest()
            .try_stage(ProjectStage::KeyframeSfm)
            .unwrap()
            .state(),
        StageState::Succeeded
    );
    assert_eq!(
        coordinator
            .store()
            .manifest()
            .try_stage(ProjectStage::FullFramePnp)
            .unwrap()
            .state(),
        StageState::Ready
    );
}

#[test]
fn confirmed_restart_invalidates_downstream_and_keeps_old_artifacts_until_replaced() {
    let temporary = tempfile::tempdir().unwrap();
    let store = ProjectStore::create(
        temporary.path().join("Restart.rustscanproject"),
        ProjectCreateRequest::new("Restart", SourceSpec::managed_images("fixture")),
    )
    .unwrap();
    let trace = Arc::new(Mutex::new(Vec::new()));
    let workers = PipelineWorkers::new(
        FakeStageWorker::new(ProjectStage::Import, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::KeyframeSfm, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::FullFramePnp, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::Training, Arc::clone(&trace)),
    );
    let mut coordinator = PipelineCoordinator::new(store, workers).unwrap();

    coordinator.send(PipelineCommand::StartAutomatic).unwrap();
    coordinator.drive_until_idle().unwrap();
    let previous_artifacts = coordinator
        .store()
        .manifest()
        .try_stage(ProjectStage::KeyframeSfm)
        .unwrap()
        .artifacts()
        .to_vec();
    let previous_attempt = coordinator
        .store()
        .manifest()
        .try_stage(ProjectStage::KeyframeSfm)
        .unwrap()
        .attempt();

    coordinator
        .send(PipelineCommand::RestartFrom {
            stage: ProjectStage::KeyframeSfm,
            confirmed: false,
        })
        .unwrap();
    coordinator.drive_until_idle().unwrap();
    assert_eq!(
        coordinator
            .store()
            .manifest()
            .try_stage(ProjectStage::KeyframeSfm)
            .unwrap()
            .attempt(),
        previous_attempt
    );

    coordinator
        .send(PipelineCommand::RestartFrom {
            stage: ProjectStage::KeyframeSfm,
            confirmed: true,
        })
        .unwrap();
    coordinator.drive_once().unwrap();
    let restarting_sfm = coordinator
        .store()
        .manifest()
        .try_stage(ProjectStage::KeyframeSfm)
        .unwrap();
    assert_eq!(restarting_sfm.state(), StageState::Running);
    assert_eq!(restarting_sfm.artifacts(), previous_artifacts.as_slice());
    coordinator.drive_until_idle().unwrap();

    let sfm = coordinator
        .store()
        .manifest()
        .try_stage(ProjectStage::KeyframeSfm)
        .unwrap();
    assert_eq!(sfm.state(), StageState::Succeeded);
    assert_eq!(sfm.attempt(), previous_attempt + 1);
    assert_ne!(sfm.artifacts(), previous_artifacts.as_slice());
    assert_eq!(
        *trace.lock().unwrap(),
        [
            ProjectStage::Import,
            ProjectStage::KeyframeSfm,
            ProjectStage::FullFramePnp,
            ProjectStage::Training,
            ProjectStage::KeyframeSfm,
            ProjectStage::FullFramePnp,
            ProjectStage::Training,
        ]
    );
}

#[test]
fn recovered_interrupted_worker_becomes_retryable_failure() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("Recovery.rustscanproject");
    let store = ProjectStore::create(
        &root,
        ProjectCreateRequest::new("Recovery", SourceSpec::managed_images("fixture")),
    )
    .unwrap();
    let trace = Arc::new(Mutex::new(Vec::new()));
    let workers = PipelineWorkers::new(
        FakeStageWorker::new(ProjectStage::Import, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::KeyframeSfm, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::FullFramePnp, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::Training, trace),
    );
    let mut interrupted = PipelineCoordinator::new(store, workers.clone()).unwrap();
    interrupted.send(PipelineCommand::StartAutomatic).unwrap();
    interrupted.drive_once().unwrap();
    drop(interrupted);

    let reopened = ProjectStore::open(&root).unwrap();
    let mut recovered = PipelineCoordinator::new(reopened, workers).unwrap();
    let import = recovered
        .store()
        .manifest()
        .try_stage(ProjectStage::Import)
        .unwrap();
    assert_eq!(import.state(), StageState::Failed);
    assert_eq!(import.error().unwrap().code, "interrupted");
    assert!(recovered.store().manifest().lease().is_none());

    recovered
        .send(PipelineCommand::Retry {
            stage: ProjectStage::Import,
        })
        .unwrap();
    recovered.drive_until_idle().unwrap();
    assert_eq!(
        recovered
            .store()
            .manifest()
            .try_stage(ProjectStage::Import)
            .unwrap()
            .state(),
        StageState::Succeeded
    );
}

#[test]
fn progress_counts_and_snapshots_reach_ui_while_persistence_is_throttled() {
    let temporary = tempfile::tempdir().unwrap();
    let store = ProjectStore::create(
        temporary.path().join("Progress.rustscanproject"),
        ProjectCreateRequest::new("Progress", SourceSpec::managed_images("fixture")),
    )
    .unwrap();
    let trace = Arc::new(Mutex::new(Vec::new()));
    let snapshot = Arc::new(HostSplats::default());
    let workers = PipelineWorkers::new(
        ProgressPausingImportWorker {
            snapshot: Arc::clone(&snapshot),
        },
        FakeStageWorker::new(ProjectStage::KeyframeSfm, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::FullFramePnp, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::Training, trace),
    );
    let mut coordinator = PipelineCoordinator::new(store, workers).unwrap();

    coordinator.send(PipelineCommand::StartAutomatic).unwrap();
    coordinator.drive_once().unwrap();
    coordinator.send(PipelineCommand::Pause).unwrap();
    coordinator.drive_until_idle().unwrap();

    let import = coordinator
        .store()
        .manifest()
        .try_stage(ProjectStage::Import)
        .unwrap();
    assert_eq!(import.completed(), Some(1));
    assert_eq!(import.total(), Some(10));
    let events = std::iter::from_fn(|| coordinator.try_next_event()).collect::<Vec<_>>();
    let progress_counts = events
        .iter()
        .filter_map(|event| match event {
            PipelineEvent::StageProgress {
                stage: ProjectStage::Import,
                completed,
                total,
                ..
            } => Some((*completed, *total)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        progress_counts,
        vec![(Some(1), Some(10)), (Some(10), Some(10))]
    );
    assert!(events.iter().any(|event| matches!(
        event,
        PipelineEvent::SceneSnapshot(found) if Arc::ptr_eq(found, &snapshot)
    )));
}

#[test]
fn pausing_worker_retains_valid_pending_artifacts_in_its_workspace() {
    let temporary = tempfile::tempdir().unwrap();
    let store = ProjectStore::create(
        temporary.path().join("PauseArtifacts.rustscanproject"),
        ProjectCreateRequest::new("Pause Artifacts", SourceSpec::managed_images("fixture")),
    )
    .unwrap();
    let trace = Arc::new(Mutex::new(Vec::new()));
    let workers = PipelineWorkers::new(
        ArtifactPausingImportWorker,
        FakeStageWorker::new(ProjectStage::KeyframeSfm, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::FullFramePnp, Arc::clone(&trace)),
        FakeStageWorker::new(ProjectStage::Training, trace),
    );
    let mut coordinator = PipelineCoordinator::new(store, workers).unwrap();

    coordinator.send(PipelineCommand::StartAutomatic).unwrap();
    coordinator.drive_once().unwrap();
    coordinator.send(PipelineCommand::Pause).unwrap();
    coordinator.drive_until_idle().unwrap();

    assert_eq!(
        std::fs::read(
            coordinator
                .store()
                .root()
                .join("Cache/.staging/import-1/checkpoint.json"),
        )
        .unwrap(),
        br#"{"complete":false}"#,
    );
}

#[derive(Clone)]
struct FakeStageWorker {
    stage: ProjectStage,
    trace: Arc<Mutex<Vec<ProjectStage>>>,
}

struct CapturingSfmWorker {
    request_sender: mpsc::Sender<StageRequest>,
}

impl SfmWorker for CapturingSfmWorker {
    fn run(
        &self,
        request: StageRequest,
        _control: WorkerControl,
        _events: WorkerEventSink,
    ) -> WorkerOutcome {
        self.request_sender.send(request).unwrap();
        WorkerOutcome::Failed(rust_viewer::project::ProjectErrorRecord {
            code: "captured".to_owned(),
            stage: ProjectStage::KeyframeSfm,
            summary: "Captured request".to_owned(),
            detail: "The request was captured for this contract test.".to_owned(),
            frame_id: None,
            pair: None,
            retryable: false,
            suggested_actions: Vec::new(),
        })
    }
}

impl FakeStageWorker {
    fn new(stage: ProjectStage, trace: Arc<Mutex<Vec<ProjectStage>>>) -> Self {
        Self { stage, trace }
    }

    fn run_fake(
        &self,
        request: StageRequest,
        _control: WorkerControl,
        _events: WorkerEventSink,
    ) -> WorkerOutcome {
        assert_eq!(request.stage, self.stage);
        self.trace.lock().unwrap().push(self.stage);
        WorkerOutcome::Succeeded(vec![PendingArtifact::new(
            "result.json",
            br#"{}"#.to_vec(),
            ArtifactValidation::Json,
        )])
    }
}

impl ImportWorker for FakeStageWorker {
    fn run(
        &self,
        request: StageRequest,
        control: WorkerControl,
        events: WorkerEventSink,
    ) -> WorkerOutcome {
        self.run_fake(request, control, events)
    }
}

impl SfmWorker for FakeStageWorker {
    fn run(
        &self,
        request: StageRequest,
        control: WorkerControl,
        events: WorkerEventSink,
    ) -> WorkerOutcome {
        self.run_fake(request, control, events)
    }
}

impl PnpWorker for FakeStageWorker {
    fn run(
        &self,
        request: StageRequest,
        control: WorkerControl,
        events: WorkerEventSink,
    ) -> WorkerOutcome {
        self.run_fake(request, control, events)
    }
}

impl TrainingWorker for FakeStageWorker {
    fn run(
        &self,
        request: StageRequest,
        control: WorkerControl,
        events: WorkerEventSink,
    ) -> WorkerOutcome {
        self.run_fake(request, control, events)
    }
}

struct PausingImportWorker;

impl ImportWorker for PausingImportWorker {
    fn run(
        &self,
        _request: StageRequest,
        control: WorkerControl,
        _events: WorkerEventSink,
    ) -> WorkerOutcome {
        while !control.pause_requested() && !control.cancel_requested() {
            thread::sleep(Duration::from_millis(1));
        }
        if control.pause_requested() {
            WorkerOutcome::Paused(Vec::new())
        } else {
            WorkerOutcome::Cancelled(Vec::new())
        }
    }
}

struct IncompletePnpWorker {
    trace: Arc<Mutex<Vec<ProjectStage>>>,
}

struct LateProgressFailingPnpWorker {
    trigger: Receiver<()>,
    emitted: Sender<()>,
}

struct RetriedPnpWorker {
    stale_trigger: Receiver<()>,
    stale_emitted: Sender<()>,
    current_emitted: Sender<()>,
    finish: Receiver<()>,
}

struct FailingThenSucceedingSfmWorker {
    trace: Arc<Mutex<Vec<ProjectStage>>>,
    failures_remaining: Arc<Mutex<usize>>,
}

struct ProgressPausingImportWorker {
    snapshot: Arc<HostSplats>,
}

impl ImportWorker for ProgressPausingImportWorker {
    fn run(
        &self,
        _request: StageRequest,
        control: WorkerControl,
        events: WorkerEventSink,
    ) -> WorkerOutcome {
        events.progress(
            Some(1),
            Some(10),
            PipelineProgressDetail::Media { frame_id: Some(0) },
        );
        events.progress(
            Some(10),
            Some(10),
            PipelineProgressDetail::Media { frame_id: Some(9) },
        );
        events.scene_snapshot(Arc::clone(&self.snapshot));
        while !control.pause_requested() {
            thread::sleep(Duration::from_millis(1));
        }
        WorkerOutcome::Paused(Vec::new())
    }
}

struct ArtifactPausingImportWorker;

impl ImportWorker for ArtifactPausingImportWorker {
    fn run(
        &self,
        _request: StageRequest,
        control: WorkerControl,
        _events: WorkerEventSink,
    ) -> WorkerOutcome {
        while !control.pause_requested() {
            thread::sleep(Duration::from_millis(1));
        }
        WorkerOutcome::Paused(vec![PendingArtifact::new(
            "checkpoint.json",
            br#"{"complete":false}"#.to_vec(),
            ArtifactValidation::Json,
        )])
    }
}

impl SfmWorker for FailingThenSucceedingSfmWorker {
    fn run(
        &self,
        request: StageRequest,
        _control: WorkerControl,
        _events: WorkerEventSink,
    ) -> WorkerOutcome {
        self.trace.lock().unwrap().push(request.stage);
        let mut failures_remaining = self.failures_remaining.lock().unwrap();
        if *failures_remaining > 0 {
            *failures_remaining -= 1;
            return WorkerOutcome::Failed(rust_viewer::project::ProjectErrorRecord {
                code: "sfm_failed".to_owned(),
                stage: ProjectStage::KeyframeSfm,
                summary: "SFM failed".to_owned(),
                detail: "test failure".to_owned(),
                frame_id: None,
                pair: None,
                retryable: true,
                suggested_actions: vec![rust_viewer::project::SuggestedAction::Retry],
            });
        }
        WorkerOutcome::Succeeded(vec![PendingArtifact::new(
            "result.json",
            br#"{}"#.to_vec(),
            ArtifactValidation::Json,
        )])
    }
}

impl PnpWorker for IncompletePnpWorker {
    fn run(
        &self,
        request: StageRequest,
        _control: WorkerControl,
        _events: WorkerEventSink,
    ) -> WorkerOutcome {
        self.trace.lock().unwrap().push(request.stage);
        WorkerOutcome::Succeeded(vec![PendingArtifact::new(
            "registration.json",
            br#"{}"#.to_vec(),
            ArtifactValidation::PnpCoverage {
                imported_frames: 5,
                registered_frames: 4,
            },
        )])
    }
}

impl PnpWorker for LateProgressFailingPnpWorker {
    fn run(
        &self,
        _request: StageRequest,
        _control: WorkerControl,
        events: WorkerEventSink,
    ) -> WorkerOutcome {
        let trigger = self.trigger.clone();
        let emitted = self.emitted.clone();
        thread::spawn(move || {
            if trigger.recv().is_ok() {
                events.progress(
                    Some(9),
                    Some(10),
                    PipelineProgressDetail::Sfm {
                        operation: "MatchPairBatch".to_owned(),
                        image_id: None,
                        pair: None,
                        registered_images: None,
                        sparse_points: None,
                    },
                );
                let _ = emitted.send(());
            }
        });
        WorkerOutcome::Failed(rust_viewer::project::ProjectErrorRecord {
            code: "terminal_failure".to_owned(),
            stage: ProjectStage::FullFramePnp,
            summary: "PnP failed".to_owned(),
            detail: "terminal PnP error".to_owned(),
            frame_id: None,
            pair: None,
            retryable: false,
            suggested_actions: Vec::new(),
        })
    }
}

impl PnpWorker for RetriedPnpWorker {
    fn run(
        &self,
        request: StageRequest,
        _control: WorkerControl,
        events: WorkerEventSink,
    ) -> WorkerOutcome {
        match request.attempt {
            1 => {
                let trigger = self.stale_trigger.clone();
                let emitted = self.stale_emitted.clone();
                thread::spawn(move || {
                    if trigger.recv().is_ok() {
                        events.progress(
                            Some(9),
                            Some(10),
                            PipelineProgressDetail::Sfm {
                                operation: "StaleAttempt".to_owned(),
                                image_id: None,
                                pair: None,
                                registered_images: None,
                                sparse_points: None,
                            },
                        );
                        let _ = emitted.send(());
                    }
                });
                WorkerOutcome::Failed(rust_viewer::project::ProjectErrorRecord {
                    code: "first_attempt_failed".to_owned(),
                    stage: ProjectStage::FullFramePnp,
                    summary: "PnP failed".to_owned(),
                    detail: "attempt one failed".to_owned(),
                    frame_id: None,
                    pair: None,
                    retryable: true,
                    suggested_actions: Vec::new(),
                })
            }
            2 => {
                events.progress(
                    Some(4),
                    Some(4),
                    PipelineProgressDetail::Sfm {
                        operation: "CurrentAttempt".to_owned(),
                        image_id: None,
                        pair: None,
                        registered_images: None,
                        sparse_points: None,
                    },
                );
                let _ = self.current_emitted.send(());
                let _ = self.finish.recv();
                WorkerOutcome::Succeeded(vec![PendingArtifact::new(
                    "result.json",
                    br#"{}"#.to_vec(),
                    ArtifactValidation::Json,
                )])
            }
            attempt => panic!("unexpected PnP attempt {attempt}"),
        }
    }
}
