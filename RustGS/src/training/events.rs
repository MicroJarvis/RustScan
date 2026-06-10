use crate::core::HostSplats;
use crate::training::reporting::telemetry::LiteGsTrainingTelemetry;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;

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

#[derive(Debug, Clone)]
pub struct TrainingControl {
    cancelled: Arc<AtomicBool>,
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
            cancelled: Arc::new(AtomicBool::new(false)),
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
        self.cancelled.store(true, Ordering::Relaxed);
    }

    pub fn is_cancel_requested(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
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

#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum TrainingEvent {
    RunStarted(TrainingRunStarted),
    PlanSelected(TrainingPlanSelected),
    IterationProgress(TrainingIterationProgress),
    SnapshotReady(TrainingSnapshotReady),
    RunCancelled(TrainingRunCancelled),
    RunCompleted(TrainingRunCompleted),
}

pub type TrainingEventSink<'a> = dyn FnMut(TrainingEvent) + 'a;

pub struct TrainingOptions<'a> {
    pub control: TrainingControl,
    pub on_event: Option<Box<TrainingEventSink<'a>>>,
}

impl<'a> Default for TrainingOptions<'a> {
    fn default() -> Self {
        Self {
            control: TrainingControl::default(),
            on_event: None,
        }
    }
}

impl<'a> TrainingOptions<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_control(mut self, control: TrainingControl) -> Self {
        self.control = control;
        self
    }

    pub fn with_event_sink<F>(mut self, on_event: F) -> Self
    where
        F: FnMut(TrainingEvent) + 'a,
    {
        self.on_event = Some(Box::new(on_event));
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
