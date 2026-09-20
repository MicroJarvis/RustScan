use crate::core::HostSplats;
use crate::training::checkpoint::{TrainingCheckpoint, TrainingIdentity};
use crate::training::evaluation::SharedWgpuContext;
use crate::training::reporting::telemetry::LiteGsTrainingTelemetry;
use crate::TrainingError;
use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc,
};
use std::time::Duration;

const TRAINING_RUNNING: u8 = 0;
const TRAINING_PAUSE_REQUESTED: u8 = 1;
const TRAINING_CANCEL_REQUESTED: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingEventRoute {
    Standard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingEventCadence {
    pub progress_every: usize,
    pub snapshot_every: Option<usize>,
}

impl Default for TrainingEventCadence {
    fn default() -> Self {
        Self {
            progress_every: 1,
            snapshot_every: None,
        }
    }
}

impl TrainingEventCadence {
    pub fn should_emit_progress(&self, iteration: usize) -> bool {
        let every = self.progress_every.max(1);
        iteration > 0 && iteration.is_multiple_of(every)
    }

    pub fn should_emit_snapshot(&self, iteration: usize) -> bool {
        let Some(every) = self.snapshot_every else {
            return false;
        };
        let every = every.max(1);
        iteration > 0 && iteration.is_multiple_of(every)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TrainingCheckpointPolicy {
    pub every: Option<usize>,
}

impl TrainingCheckpointPolicy {
    pub fn should_checkpoint(self, iteration: usize) -> bool {
        self.every
            .is_some_and(|every| iteration > 0 && iteration.is_multiple_of(every.max(1)))
    }
}

#[derive(Debug, Clone)]
pub struct TrainingControl {
    state: Arc<AtomicU8>,
    cadence: TrainingEventCadence,
}

impl Default for TrainingControl {
    fn default() -> Self {
        Self::new(TrainingEventCadence::default())
    }
}

impl TrainingControl {
    pub fn new(cadence: TrainingEventCadence) -> Self {
        Self {
            state: Arc::new(AtomicU8::new(TRAINING_RUNNING)),
            cadence,
        }
    }

    pub fn with_progress_cadence(mut self, every: usize) -> Self {
        self.cadence.progress_every = every.max(1);
        self
    }

    pub fn with_snapshot_cadence(mut self, every: Option<usize>) -> Self {
        self.cadence.snapshot_every = every.map(|value| value.max(1));
        self
    }

    pub fn request_cancel(&self) {
        self.state
            .store(TRAINING_CANCEL_REQUESTED, Ordering::Release);
    }

    pub fn is_cancel_requested(&self) -> bool {
        self.state.load(Ordering::Acquire) == TRAINING_CANCEL_REQUESTED
    }

    pub fn request_pause(&self) {
        let _ = self.state.compare_exchange(
            TRAINING_RUNNING,
            TRAINING_PAUSE_REQUESTED,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    pub fn is_pause_requested(&self) -> bool {
        self.state.load(Ordering::Acquire) == TRAINING_PAUSE_REQUESTED
    }

    pub fn cadence(&self) -> TrainingEventCadence {
        self.cadence
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingRunStarted {
    pub iterations: usize,
    pub frame_count: usize,
    pub input_point_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingPlanSelected {
    pub route: TrainingEventRoute,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingRunCompleted {
    pub report: TrainingRunReport,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingCheckpointReason {
    Periodic,
    Pause,
    Shutdown,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingCheckpointReady {
    pub iteration: usize,
    pub reason: TrainingCheckpointReason,
    pub checkpoint: TrainingCheckpoint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrainingRunDisposition {
    #[default]
    Completed,
    Paused,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainingIterationProgress {
    pub iteration: usize,
    pub latest_loss: f32,
    pub gaussian_count: usize,
    pub elapsed: Duration,
}

#[derive(Debug, Clone)]
pub struct TrainingSnapshotReady {
    pub iteration: usize,
    pub latest_loss: f32,
    pub gaussian_count: usize,
    pub elapsed: Duration,
    pub splats: HostSplats,
}

impl PartialEq for TrainingSnapshotReady {
    fn eq(&self, other: &Self) -> bool {
        self.iteration == other.iteration
            && self.latest_loss.to_bits() == other.latest_loss.to_bits()
            && self.gaussian_count == other.gaussian_count
            && self.elapsed == other.elapsed
            && self.splats.len() == other.splats.len()
            && self.splats.sh_degree() == other.splats.sh_degree()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingRunCancelled {
    pub completed_iterations: usize,
    pub elapsed: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingRunPaused {
    pub completed_iterations: usize,
    pub elapsed: Duration,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingRunFailed {
    pub error: String,
    pub elapsed: Duration,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum TrainingEvent {
    RunStarted(TrainingRunStarted),
    PlanSelected(TrainingPlanSelected),
    IterationProgress(TrainingIterationProgress),
    SnapshotReady(TrainingSnapshotReady),
    CheckpointReady(TrainingCheckpointReady),
    RunPaused(TrainingRunPaused),
    RunCancelled(TrainingRunCancelled),
    RunFailed(TrainingRunFailed),
    RunCompleted(TrainingRunCompleted),
}

pub type TrainingEventSink<'a> = dyn FnMut(TrainingEvent) + 'a;
pub type TrainingCheckpointSink<'a> =
    dyn FnMut(&TrainingCheckpointReady) -> Result<(), TrainingError> + 'a;

#[derive(Default)]
pub struct TrainingOptions<'a> {
    pub control: TrainingControl,
    pub identity: Option<TrainingIdentity>,
    pub resume_checkpoint: Option<TrainingCheckpoint>,
    pub checkpoint_policy: TrainingCheckpointPolicy,
    pub shared_wgpu_context: Option<SharedWgpuContext>,
    pub on_event: Option<Box<TrainingEventSink<'a>>>,
    pub on_checkpoint: Option<Box<TrainingCheckpointSink<'a>>>,
}

impl<'a> TrainingOptions<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_control(mut self, control: TrainingControl) -> Self {
        self.control = control;
        self
    }

    pub fn with_identity(mut self, identity: TrainingIdentity) -> Self {
        self.identity = Some(identity);
        self
    }

    pub fn with_resume_checkpoint(mut self, checkpoint: TrainingCheckpoint) -> Self {
        self.resume_checkpoint = Some(checkpoint);
        self
    }

    pub fn with_checkpoint_policy(mut self, policy: TrainingCheckpointPolicy) -> Self {
        self.checkpoint_policy = policy;
        self
    }

    pub fn with_shared_wgpu_context(mut self, context: SharedWgpuContext) -> Self {
        self.shared_wgpu_context = Some(context);
        self
    }

    pub fn with_event_sink<F>(mut self, on_event: F) -> Self
    where
        F: FnMut(TrainingEvent) + 'a,
    {
        self.on_event = Some(Box::new(on_event));
        self
    }

    pub fn with_checkpoint_sink<F>(mut self, sink: F) -> Self
    where
        F: FnMut(&TrainingCheckpointReady) -> Result<(), TrainingError> + 'a,
    {
        self.on_checkpoint = Some(Box::new(sink));
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingRunReport {
    pub elapsed: Duration,
    pub training_loop_elapsed: Duration,
    pub final_loss: Option<f32>,
    pub final_step_loss: Option<f32>,
    pub gaussian_count: usize,
    pub sh_degree: usize,
    pub completed_iterations: usize,
    pub cancelled: bool,
    pub disposition: TrainingRunDisposition,
    pub telemetry: Option<LiteGsTrainingTelemetry>,
}

impl TrainingRunReport {
    pub fn metadata_final_loss_or(&self, default: f32) -> f32 {
        self.final_loss.unwrap_or(default)
    }
}

#[derive(Debug)]
pub struct TrainingRun {
    pub splats: HostSplats,
    pub report: TrainingRunReport,
}

impl TrainingRun {
    pub fn into_splats(self) -> HostSplats {
        self.splats
    }
}

pub(crate) fn emit_training_event(sink: &mut TrainingEventSink<'_>, event: TrainingEvent) {
    sink(event);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_control_cancel_overrides_pause() {
        let control = TrainingControl::new(TrainingEventCadence {
            progress_every: 5,
            snapshot_every: Some(100),
        });

        control.request_pause();
        assert!(control.is_pause_requested());
        assert!(!control.is_cancel_requested());

        control.request_cancel();
        assert!(control.is_cancel_requested());
        assert!(!control.is_pause_requested());
    }

    #[test]
    fn training_control_checkpoint_policy_honors_cadence_and_iteration_zero() {
        let policy = TrainingCheckpointPolicy { every: Some(1_000) };

        assert!(!policy.should_checkpoint(0));
        assert!(!policy.should_checkpoint(999));
        assert!(policy.should_checkpoint(1_000));
        assert!(policy.should_checkpoint(2_000));
        assert!(TrainingCheckpointPolicy { every: Some(0) }.should_checkpoint(1));
        assert!(!TrainingCheckpointPolicy::default().should_checkpoint(1_000));
    }

    #[test]
    fn training_options_checkpoint_builders_keep_checkpoint_sink_separate() {
        let identity = crate::TrainingIdentity {
            dataset: "dataset".to_string(),
            reconstruction: "reconstruction".to_string(),
            config: "config".to_string(),
        };
        let options = TrainingOptions::new()
            .with_identity(identity.clone())
            .with_checkpoint_policy(TrainingCheckpointPolicy { every: Some(5) })
            .with_checkpoint_sink(|_ready| Ok(()));

        assert_eq!(options.identity, Some(identity));
        assert!(options.resume_checkpoint.is_none());
        assert!(options.checkpoint_policy.should_checkpoint(5));
        assert!(options.on_checkpoint.is_some());
    }

    #[test]
    fn training_run_dispositions_distinguish_all_terminal_states() {
        assert_ne!(
            TrainingRunDisposition::Paused,
            TrainingRunDisposition::Cancelled
        );
        assert_ne!(
            TrainingRunDisposition::Completed,
            TrainingRunDisposition::Paused
        );
    }
}
