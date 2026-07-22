use std::collections::VecDeque;
use std::sync::{mpsc, Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use rustgs::{
    train_splats, HostSplats, TrainingCheckpointReady, TrainingConfig, TrainingControl,
    TrainingDataset, TrainingError, TrainingEvent, TrainingEventCadence, TrainingIterationProgress,
    TrainingOptions, TrainingRun, TrainingRunPaused, TrainingRunReport, TrainingSnapshotReady,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingSessionState {
    Idle,
    Loading,
    Starting,
    Training,
    Stopping,
    Completed,
    Failed,
    Cancelled,
}

impl TrainingSessionState {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingProgress {
    pub total_iterations: Option<usize>,
    pub latest_iteration: Option<usize>,
    pub latest_loss: Option<f32>,
    pub gaussian_count: Option<usize>,
    pub elapsed: Duration,
    pub event_count: usize,
}

impl Default for TrainingProgress {
    fn default() -> Self {
        Self {
            total_iterations: None,
            latest_iteration: None,
            latest_loss: None,
            gaussian_count: None,
            elapsed: Duration::from_secs(0),
            event_count: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingControlOptions {
    pub progress_every: usize,
    pub snapshot_every: Option<usize>,
    /// Keep the latest host-side splat snapshot even when stop/cancel was requested.
    pub retain_snapshot_on_cancel: bool,
}

impl Default for TrainingControlOptions {
    fn default() -> Self {
        Self {
            progress_every: 1,
            snapshot_every: Some(25),
            retain_snapshot_on_cancel: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrainingSessionEvent {
    StateChanged {
        from: TrainingSessionState,
        to: TrainingSessionState,
    },
    BackendEvent(TrainingEvent),
    ProgressUpdated(TrainingProgress),
    SnapshotUpdated {
        gaussian_count: usize,
        sh_degree: usize,
    },
    Completed(TrainingRunReport),
    Failed(String),
    Cancelled,
}

#[derive(Debug, thiserror::Error)]
pub enum TrainingSessionError {
    #[error("training session is already running")]
    AlreadyRunning,
    #[error("no active training session")]
    NoActiveSession,
    #[error("failed to send control command to worker")]
    ControlChannelClosed,
}

pub trait TrainingRunner: Send + 'static {
    fn run(
        &self,
        dataset: &TrainingDataset,
        config: &TrainingConfig,
        control: TrainingControl,
        on_event: &mut dyn FnMut(TrainingEvent),
    ) -> Result<TrainingRun, TrainingError>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RustGsTrainingRunner;

impl TrainingRunner for RustGsTrainingRunner {
    fn run(
        &self,
        dataset: &TrainingDataset,
        config: &TrainingConfig,
        control: TrainingControl,
        on_event: &mut dyn FnMut(TrainingEvent),
    ) -> Result<TrainingRun, TrainingError> {
        train_splats(
            dataset,
            config,
            TrainingOptions::new()
                .with_control(control)
                .with_event_sink(on_event),
        )
    }
}

#[derive(Debug)]
pub struct TrainingSession {
    shared: Arc<Mutex<SessionShared>>,
    training_control: TrainingControl,
    control_tx: mpsc::Sender<WorkerControlMessage>,
    event_rx: mpsc::Receiver<TrainingSessionEvent>,
    worker: Option<JoinHandle<()>>,
    pending_events: VecDeque<TrainingSessionEvent>,
}

impl TrainingSession {
    pub fn start(
        dataset: TrainingDataset,
        config: TrainingConfig,
        options: TrainingControlOptions,
    ) -> Self {
        Self::start_with_runner(dataset, config, options, RustGsTrainingRunner)
    }

    pub(crate) fn start_with_runner<R>(
        dataset: TrainingDataset,
        config: TrainingConfig,
        options: TrainingControlOptions,
        runner: R,
    ) -> Self
    where
        R: TrainingRunner,
    {
        let training_control = TrainingControl::new(TrainingEventCadence {
            progress_every: options.progress_every.max(1),
            snapshot_every: options.snapshot_every.map(|value| value.max(1)),
        });
        let (control_tx, control_rx) = mpsc::channel::<WorkerControlMessage>();
        let (event_tx, event_rx) = mpsc::channel::<TrainingSessionEvent>();
        let shared = Arc::new(Mutex::new(SessionShared::default()));

        let shared_clone = Arc::clone(&shared);
        let worker_control = training_control.clone();
        let worker = thread::spawn(move || {
            run_training_worker(
                dataset,
                config,
                options,
                runner,
                worker_control,
                shared_clone,
                control_rx,
                event_tx,
            );
        });

        Self {
            shared,
            training_control,
            control_tx,
            event_rx,
            worker: Some(worker),
            pending_events: VecDeque::new(),
        }
    }

    pub fn stop(&mut self) -> Result<(), TrainingSessionError> {
        self.cancel()
    }

    pub fn cancel(&mut self) -> Result<(), TrainingSessionError> {
        let mut shared = lock_unpoison(&self.shared);
        if shared.state.is_terminal() {
            return Ok(());
        }

        let from = shared.state;
        if matches!(
            shared.state,
            TrainingSessionState::Loading
                | TrainingSessionState::Starting
                | TrainingSessionState::Training
        ) {
            shared.state = TrainingSessionState::Stopping;
            self.pending_events
                .push_back(TrainingSessionEvent::StateChanged {
                    from,
                    to: TrainingSessionState::Stopping,
                });
        }
        drop(shared);

        self.training_control.request_cancel();
        self.control_tx
            .send(WorkerControlMessage::Stop)
            .map_err(|_| TrainingSessionError::ControlChannelClosed)
    }

    pub fn poll_events(&mut self) -> Vec<TrainingSessionEvent> {
        let mut events = Vec::new();

        while let Some(event) = self.pending_events.pop_front() {
            events.push(event);
        }

        while let Ok(event) = self.event_rx.try_recv() {
            events.push(event);
        }

        self.try_join_finished_worker();
        events
    }

    pub fn recv_event(&mut self, timeout: Duration) -> Option<TrainingSessionEvent> {
        if let Some(event) = self.pending_events.pop_front() {
            return Some(event);
        }

        match self.event_rx.recv_timeout(timeout) {
            Ok(event) => {
                self.try_join_finished_worker();
                Some(event)
            }
            Err(_) => None,
        }
    }

    pub fn state(&self) -> TrainingSessionState {
        lock_unpoison(&self.shared).state
    }

    pub fn progress(&self) -> TrainingProgress {
        lock_unpoison(&self.shared).progress.clone()
    }

    pub fn latest_snapshot(&self) -> Option<Arc<HostSplats>> {
        lock_unpoison(&self.shared).latest_snapshot.clone()
    }

    pub fn latest_report(&self) -> Option<TrainingRunReport> {
        lock_unpoison(&self.shared).latest_report.clone()
    }

    pub fn latest_error(&self) -> Option<String> {
        lock_unpoison(&self.shared).last_error.clone()
    }

    pub fn is_terminal(&self) -> bool {
        self.state().is_terminal()
    }

    fn try_join_finished_worker(&mut self) {
        if let Some(handle) = self.worker.as_ref() {
            if !handle.is_finished() {
                return;
            }
        } else {
            return;
        }

        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }
}

#[derive(Debug, Default)]
pub struct TrainingManager {
    session: Option<TrainingSession>,
}

impl TrainingManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn start(
        &mut self,
        dataset: TrainingDataset,
        config: TrainingConfig,
        options: TrainingControlOptions,
    ) -> Result<(), TrainingSessionError> {
        self.start_with_runner(dataset, config, options, RustGsTrainingRunner)
    }

    pub(crate) fn start_with_runner<R>(
        &mut self,
        dataset: TrainingDataset,
        config: TrainingConfig,
        options: TrainingControlOptions,
        runner: R,
    ) -> Result<(), TrainingSessionError>
    where
        R: TrainingRunner,
    {
        if self
            .session
            .as_ref()
            .map(|session| !session.is_terminal())
            .unwrap_or(false)
        {
            return Err(TrainingSessionError::AlreadyRunning);
        }

        self.session = Some(TrainingSession::start_with_runner(
            dataset, config, options, runner,
        ));
        Ok(())
    }

    pub fn stop(&mut self) -> Result<(), TrainingSessionError> {
        self.session
            .as_mut()
            .ok_or(TrainingSessionError::NoActiveSession)?
            .stop()
    }

    pub fn cancel(&mut self) -> Result<(), TrainingSessionError> {
        self.stop()
    }

    pub fn poll_events(&mut self) -> Vec<TrainingSessionEvent> {
        match self.session.as_mut() {
            Some(session) => session.poll_events(),
            None => Vec::new(),
        }
    }

    pub fn state(&self) -> TrainingSessionState {
        self.session
            .as_ref()
            .map(TrainingSession::state)
            .unwrap_or(TrainingSessionState::Idle)
    }

    pub fn progress(&self) -> TrainingProgress {
        self.session
            .as_ref()
            .map(TrainingSession::progress)
            .unwrap_or_default()
    }

    pub fn latest_snapshot(&self) -> Option<Arc<HostSplats>> {
        self.session
            .as_ref()
            .and_then(TrainingSession::latest_snapshot)
    }

    pub fn latest_report(&self) -> Option<TrainingRunReport> {
        self.session
            .as_ref()
            .and_then(TrainingSession::latest_report)
    }

    pub fn latest_error(&self) -> Option<String> {
        self.session
            .as_ref()
            .and_then(TrainingSession::latest_error)
    }

    pub fn has_active_session(&self) -> bool {
        self.session.is_some()
    }

    pub fn clear_terminal_session(&mut self) {
        let should_clear = self
            .session
            .as_ref()
            .map(TrainingSession::is_terminal)
            .unwrap_or(false);
        if should_clear {
            self.session = None;
        }
    }
}

#[derive(Debug, Clone)]
struct SessionShared {
    state: TrainingSessionState,
    progress: TrainingProgress,
    latest_snapshot: Option<Arc<HostSplats>>,
    latest_report: Option<TrainingRunReport>,
    last_error: Option<String>,
}

impl Default for SessionShared {
    fn default() -> Self {
        Self {
            state: TrainingSessionState::Idle,
            progress: TrainingProgress::default(),
            latest_snapshot: None,
            latest_report: None,
            last_error: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkerControlMessage {
    Stop,
}

fn run_training_worker<R>(
    dataset: TrainingDataset,
    config: TrainingConfig,
    options: TrainingControlOptions,
    runner: R,
    training_control: TrainingControl,
    shared: Arc<Mutex<SessionShared>>,
    control_rx: mpsc::Receiver<WorkerControlMessage>,
    event_tx: mpsc::Sender<TrainingSessionEvent>,
) where
    R: TrainingRunner,
{
    transition_state(&shared, &event_tx, TrainingSessionState::Loading);
    if stop_requested(&control_rx) {
        finalize_cancelled(&shared, &event_tx);
        return;
    }

    transition_state(&shared, &event_tx, TrainingSessionState::Starting);
    if stop_requested(&control_rx) {
        finalize_cancelled(&shared, &event_tx);
        return;
    }

    {
        let mut guard = lock_unpoison(&shared);
        guard.progress.total_iterations = Some(config.iterations);
    }
    emit_progress_update(&shared, &event_tx);
    transition_state(&shared, &event_tx, TrainingSessionState::Training);

    let mut cancel_requested = false;
    let mut on_event = |event: TrainingEvent| {
        apply_backend_event(&shared, &event, &event_tx);
        let _ = event_tx.send(TrainingSessionEvent::BackendEvent(event));
        emit_progress_update(&shared, &event_tx);

        if !cancel_requested && stop_requested(&control_rx) {
            cancel_requested = true;
            training_control.request_cancel();
            transition_state(&shared, &event_tx, TrainingSessionState::Stopping);
        }
    };

    let result = runner.run(&dataset, &config, training_control.clone(), &mut on_event);
    if !cancel_requested && stop_requested(&control_rx) {
        cancel_requested = true;
        training_control.request_cancel();
        transition_state(&shared, &event_tx, TrainingSessionState::Stopping);
    }

    match result {
        Ok(run) if cancel_requested || run.report.cancelled => {
            if options.retain_snapshot_on_cancel {
                store_snapshot(&shared, &event_tx, Arc::new(run.splats));
            }
            finalize_cancelled(&shared, &event_tx);
        }
        Ok(run) => finalize_completed(&shared, &event_tx, run),
        Err(err) if cancel_requested => {
            let _ = err;
            finalize_cancelled(&shared, &event_tx);
        }
        Err(err) => finalize_failed(&shared, &event_tx, err.to_string()),
    }
}

fn lock_unpoison<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn transition_state(
    shared: &Arc<Mutex<SessionShared>>,
    event_tx: &mpsc::Sender<TrainingSessionEvent>,
    next: TrainingSessionState,
) {
    let from = {
        let mut guard = lock_unpoison(shared);
        if guard.state == next {
            return;
        }
        let previous = guard.state;
        guard.state = next;
        previous
    };

    let _ = event_tx.send(TrainingSessionEvent::StateChanged { from, to: next });
}

fn emit_progress_update(
    shared: &Arc<Mutex<SessionShared>>,
    event_tx: &mpsc::Sender<TrainingSessionEvent>,
) {
    let progress = lock_unpoison(shared).progress.clone();
    let _ = event_tx.send(TrainingSessionEvent::ProgressUpdated(progress));
}

fn apply_backend_event(
    shared: &Arc<Mutex<SessionShared>>,
    event: &TrainingEvent,
    event_tx: &mpsc::Sender<TrainingSessionEvent>,
) {
    let mut guard = lock_unpoison(shared);
    guard.progress.event_count = guard.progress.event_count.saturating_add(1);

    match event {
        TrainingEvent::RunStarted(started) => {
            guard.progress.total_iterations = Some(started.iterations);
        }
        TrainingEvent::IterationProgress(TrainingIterationProgress {
            iteration,
            latest_loss,
            gaussian_count,
            elapsed,
        }) => {
            guard.progress.latest_iteration = Some(*iteration);
            guard.progress.latest_loss = Some(*latest_loss);
            guard.progress.gaussian_count = Some(*gaussian_count);
            guard.progress.elapsed = *elapsed;
        }
        TrainingEvent::SnapshotReady(TrainingSnapshotReady {
            iteration,
            latest_loss,
            gaussian_count,
            elapsed,
            splats,
        }) => {
            guard.progress.latest_iteration = Some(*iteration);
            guard.progress.latest_loss = Some(*latest_loss);
            guard.progress.gaussian_count = Some(*gaussian_count);
            guard.progress.elapsed = *elapsed;
            let snapshot = Arc::new(splats.clone());
            guard.latest_snapshot = Some(snapshot.clone());
            drop(guard);
            let _ = event_tx.send(TrainingSessionEvent::SnapshotUpdated {
                gaussian_count: snapshot.len(),
                sh_degree: snapshot.sh_degree(),
            });
            return;
        }
        TrainingEvent::CheckpointReady(TrainingCheckpointReady {
            iteration,
            checkpoint,
            ..
        }) => {
            guard.progress.latest_iteration = Some(*iteration);
            guard.progress.latest_loss = checkpoint.latest_loss;
            guard.progress.gaussian_count = Some(checkpoint.splats.len());
        }
        TrainingEvent::RunPaused(TrainingRunPaused {
            completed_iterations,
            elapsed,
        }) => {
            guard.progress.elapsed = *elapsed;
            guard.progress.latest_iteration = Some(*completed_iterations);
        }
        TrainingEvent::RunCancelled(cancelled) => {
            guard.progress.elapsed = cancelled.elapsed;
            guard.progress.latest_iteration = Some(cancelled.completed_iterations);
        }
        TrainingEvent::RunFailed(failed) => {
            guard.progress.elapsed = failed.elapsed;
            guard.last_error = Some(failed.error.clone());
        }
        TrainingEvent::RunCompleted(completed) => {
            guard.progress.latest_loss = completed.report.final_loss;
            guard.progress.gaussian_count = Some(completed.report.gaussian_count);
            guard.progress.elapsed = completed.report.elapsed;
            guard.progress.latest_iteration = Some(completed.report.completed_iterations);
        }
        TrainingEvent::PlanSelected(_) => {}
    }
}

fn store_snapshot(
    shared: &Arc<Mutex<SessionShared>>,
    event_tx: &mpsc::Sender<TrainingSessionEvent>,
    snapshot: Arc<HostSplats>,
) {
    let summary = TrainingSessionEvent::SnapshotUpdated {
        gaussian_count: snapshot.len(),
        sh_degree: snapshot.sh_degree(),
    };

    {
        let mut guard = lock_unpoison(shared);
        guard.latest_snapshot = Some(snapshot);
    }

    let _ = event_tx.send(summary);
}

fn finalize_completed(
    shared: &Arc<Mutex<SessionShared>>,
    event_tx: &mpsc::Sender<TrainingSessionEvent>,
    run: TrainingRun,
) {
    {
        let mut guard = lock_unpoison(shared);
        guard.progress.latest_loss = run.report.final_loss;
        guard.progress.gaussian_count = Some(run.report.gaussian_count);
        guard.progress.elapsed = run.report.elapsed;
        guard.progress.latest_iteration = guard.progress.total_iterations;
        guard.latest_report = Some(run.report.clone());
        guard.last_error = None;
    }
    emit_progress_update(shared, event_tx);
    store_snapshot(shared, event_tx, Arc::new(run.splats));

    transition_state(shared, event_tx, TrainingSessionState::Completed);
    let _ = event_tx.send(TrainingSessionEvent::Completed(run.report));
}

fn finalize_failed(
    shared: &Arc<Mutex<SessionShared>>,
    event_tx: &mpsc::Sender<TrainingSessionEvent>,
    error: String,
) {
    {
        let mut guard = lock_unpoison(shared);
        guard.last_error = Some(error.clone());
    }
    transition_state(shared, event_tx, TrainingSessionState::Failed);
    let _ = event_tx.send(TrainingSessionEvent::Failed(error));
}

fn finalize_cancelled(
    shared: &Arc<Mutex<SessionShared>>,
    event_tx: &mpsc::Sender<TrainingSessionEvent>,
) {
    transition_state(shared, event_tx, TrainingSessionState::Cancelled);
    let _ = event_tx.send(TrainingSessionEvent::Cancelled);
}

fn stop_requested(control_rx: &mpsc::Receiver<WorkerControlMessage>) -> bool {
    let mut stop = false;
    while let Ok(msg) = control_rx.try_recv() {
        if msg == WorkerControlMessage::Stop {
            stop = true;
        }
    }
    stop
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::Instant;

    use rustgs::{
        Intrinsics, ScenePose, TrainingEventRoute, TrainingPlanSelected, TrainingRunCompleted,
        TrainingRunDisposition, TrainingRunStarted, SE3,
    };

    struct SuccessRunner;

    impl TrainingRunner for SuccessRunner {
        fn run(
            &self,
            _dataset: &TrainingDataset,
            config: &TrainingConfig,
            _control: TrainingControl,
            on_event: &mut dyn FnMut(TrainingEvent),
        ) -> Result<TrainingRun, TrainingError> {
            on_event(TrainingEvent::RunStarted(TrainingRunStarted {
                iterations: config.iterations,
                frame_count: 1,
                input_point_count: 1,
            }));
            on_event(TrainingEvent::PlanSelected(TrainingPlanSelected {
                route: TrainingEventRoute::Standard,
            }));

            let report = TrainingRunReport {
                elapsed: Duration::from_millis(12),
                training_loop_elapsed: Duration::from_millis(10),
                final_loss: Some(0.42),
                final_step_loss: Some(0.42),
                gaussian_count: 64,
                sh_degree: 0,
                completed_iterations: config.iterations,
                cancelled: false,
                disposition: TrainingRunDisposition::Completed,
                telemetry: None,
            };
            on_event(TrainingEvent::RunCompleted(TrainingRunCompleted {
                report: report.clone(),
            }));

            Ok(TrainingRun {
                splats: HostSplats::default(),
                report,
            })
        }
    }

    struct SlowRunner;

    impl TrainingRunner for SlowRunner {
        fn run(
            &self,
            _dataset: &TrainingDataset,
            config: &TrainingConfig,
            control: TrainingControl,
            on_event: &mut dyn FnMut(TrainingEvent),
        ) -> Result<TrainingRun, TrainingError> {
            on_event(TrainingEvent::RunStarted(TrainingRunStarted {
                iterations: config.iterations,
                frame_count: 1,
                input_point_count: 1,
            }));
            for _ in 0..20 {
                thread::sleep(Duration::from_millis(10));
                if control.is_cancel_requested() {
                    return Ok(TrainingRun {
                        splats: HostSplats::default(),
                        report: TrainingRunReport {
                            elapsed: Duration::from_millis(125),
                            training_loop_elapsed: Duration::from_millis(100),
                            final_loss: None,
                            final_step_loss: None,
                            gaussian_count: 0,
                            sh_degree: 0,
                            completed_iterations: 0,
                            cancelled: true,
                            disposition: TrainingRunDisposition::Cancelled,
                            telemetry: None,
                        },
                    });
                }
                on_event(TrainingEvent::PlanSelected(TrainingPlanSelected {
                    route: TrainingEventRoute::Standard,
                }));
            }

            let report = TrainingRunReport {
                elapsed: Duration::from_millis(250),
                training_loop_elapsed: Duration::from_millis(225),
                final_loss: Some(0.11),
                final_step_loss: Some(0.11),
                gaussian_count: 128,
                sh_degree: 0,
                completed_iterations: config.iterations,
                cancelled: false,
                disposition: TrainingRunDisposition::Completed,
                telemetry: None,
            };
            on_event(TrainingEvent::RunCompleted(TrainingRunCompleted {
                report: report.clone(),
            }));
            Ok(TrainingRun {
                splats: HostSplats::default(),
                report,
            })
        }
    }

    #[test]
    fn session_reaches_completed_and_stores_report() {
        let dataset = dummy_dataset();
        let config = TrainingConfig {
            iterations: 12,
            ..TrainingConfig::default()
        };

        let mut session = TrainingSession::start_with_runner(
            dataset,
            config,
            TrainingControlOptions::default(),
            SuccessRunner,
        );

        let events = collect_until_terminal(&mut session, Duration::from_secs(2));
        assert_eq!(session.state(), TrainingSessionState::Completed);
        assert!(session.latest_snapshot().is_some());
        assert!(session.latest_report().is_some());
        assert!(events.iter().any(|event| matches!(
            event,
            TrainingSessionEvent::StateChanged {
                to: TrainingSessionState::Completed,
                ..
            }
        )));
        assert!(events
            .iter()
            .any(|event| matches!(event, TrainingSessionEvent::Completed(_))));
    }

    #[test]
    fn manager_stop_moves_session_to_cancelled() {
        let dataset = dummy_dataset();
        let config = TrainingConfig {
            iterations: 1000,
            ..TrainingConfig::default()
        };

        let mut manager = TrainingManager::new();
        manager
            .start_with_runner(
                dataset,
                config,
                TrainingControlOptions {
                    progress_every: 1,
                    snapshot_every: Some(10),
                    retain_snapshot_on_cancel: false,
                },
                SlowRunner,
            )
            .unwrap();

        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            manager.poll_events();
            if manager.state() == TrainingSessionState::Training {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(manager.state(), TrainingSessionState::Training);

        manager.stop().unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline {
            manager.poll_events();
            if manager.state().is_terminal() {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }

        assert_eq!(manager.state(), TrainingSessionState::Cancelled);
        assert!(manager.latest_report().is_none());
        assert!(manager.latest_snapshot().is_none());
    }

    #[test]
    fn paused_backend_event_updates_progress_without_adding_a_session_state() {
        let shared = Arc::new(Mutex::new(SessionShared::default()));
        let (event_tx, _event_rx) = mpsc::channel();

        apply_backend_event(
            &shared,
            &TrainingEvent::RunPaused(TrainingRunPaused {
                completed_iterations: 7,
                elapsed: Duration::from_millis(33),
            }),
            &event_tx,
        );

        let guard = lock_unpoison(&shared);
        assert_eq!(guard.progress.latest_iteration, Some(7));
        assert_eq!(guard.progress.elapsed, Duration::from_millis(33));
        assert_eq!(guard.progress.event_count, 1);
        assert_eq!(guard.state, TrainingSessionState::Idle);
    }

    fn collect_until_terminal(
        session: &mut TrainingSession,
        timeout: Duration,
    ) -> Vec<TrainingSessionEvent> {
        let mut events = Vec::new();
        let deadline = Instant::now() + timeout;

        while Instant::now() < deadline {
            events.extend(session.poll_events());
            if session.state().is_terminal() {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }

        events
    }

    fn dummy_dataset() -> TrainingDataset {
        let intrinsics = Intrinsics::from_focal(500.0, 64, 48);
        let mut dataset = TrainingDataset::new(intrinsics);
        dataset.add_pose(ScenePose::new(
            0,
            PathBuf::from("frame_0000.png"),
            SE3::identity(),
            0.0,
        ));
        dataset.add_point([0.0, 0.0, 1.0], Some([1.0, 1.0, 1.0]));
        dataset
    }
}
