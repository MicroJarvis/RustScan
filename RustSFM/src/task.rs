use serde::{Deserialize, Serialize};
use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc,
};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SfmTaskStage {
    FeatureExtraction,
    FeatureMatching,
    IncrementalMapping,
    BundleAdjustment,
    FullFrameRegistration,
    Export,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SfmTaskOperation {
    Begin,
    ExtractImage,
    MatchPairBatch,
    RegisterInitialPair,
    RegisterImage,
    LocalBundleAdjustment,
    GlobalBundleAdjustment,
    RegisterFrameAttempt,
    ValidateArtifacts,
    WriteArtifacts,
    Complete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SfmTaskEventKind {
    Started,
    Progress,
    Warning,
    Error,
    Completed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SfmTaskIssue {
    pub code: String,
    pub summary: String,
    pub detail: String,
    pub retryable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SfmTaskEvent {
    pub sequence: u64,
    pub elapsed_ms: u64,
    pub stage: SfmTaskStage,
    pub operation: SfmTaskOperation,
    pub kind: SfmTaskEventKind,
    pub completed: Option<usize>,
    pub total: Option<usize>,
    pub registered_images: Option<usize>,
    pub sparse_points: Option<usize>,
    pub image_id: Option<u32>,
    pub pair: Option<(u32, u32)>,
    pub message: Option<String>,
    pub issue: Option<SfmTaskIssue>,
}

pub trait SfmTaskEventSink {
    fn on_sfm_event(&mut self, event: SfmTaskEvent);
}

impl<F> SfmTaskEventSink for F
where
    F: FnMut(SfmTaskEvent),
{
    fn on_sfm_event(&mut self, event: SfmTaskEvent) {
        self(event);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SfmControlState {
    Running = 0,
    PauseRequested = 1,
    CancelRequested = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SfmTaskStop {
    #[error("SFM task paused at a safe boundary")]
    Paused,
    #[error("SFM task cancelled at a safe boundary")]
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct SfmTaskControl {
    state: Arc<AtomicU8>,
}

impl SfmTaskControl {
    pub fn new() -> Self {
        Self {
            state: Arc::new(AtomicU8::new(SfmControlState::Running as u8)),
        }
    }

    pub fn request_pause(&self) {
        let _ = self.state.compare_exchange(
            SfmControlState::Running as u8,
            SfmControlState::PauseRequested as u8,
            Ordering::SeqCst,
            Ordering::SeqCst,
        );
    }

    pub fn request_cancel(&self) {
        self.state
            .store(SfmControlState::CancelRequested as u8, Ordering::SeqCst);
    }

    pub fn state(&self) -> SfmControlState {
        match self.state.load(Ordering::SeqCst) {
            0 => SfmControlState::Running,
            1 => SfmControlState::PauseRequested,
            2 => SfmControlState::CancelRequested,
            state => unreachable!("invalid SFM control state: {state}"),
        }
    }

    pub fn checkpoint(&self) -> Result<(), SfmTaskStop> {
        match self.state() {
            SfmControlState::Running => Ok(()),
            SfmControlState::PauseRequested => Err(SfmTaskStop::Paused),
            SfmControlState::CancelRequested => Err(SfmTaskStop::Cancelled),
        }
    }
}

pub struct SfmTaskContext<'a> {
    control: &'a SfmTaskControl,
    sink: &'a mut dyn SfmTaskEventSink,
    started_at: Instant,
    next_sequence: u64,
}

impl<'a> SfmTaskContext<'a> {
    pub fn new(control: &'a SfmTaskControl, sink: &'a mut dyn SfmTaskEventSink) -> Self {
        Self {
            control,
            sink,
            started_at: Instant::now(),
            next_sequence: 0,
        }
    }

    pub fn emit(&mut self, mut event: SfmTaskEvent) {
        event.sequence = self.next_sequence;
        event.elapsed_ms = self.started_at.elapsed().as_millis().min(u64::MAX as u128) as u64;
        self.next_sequence = self.next_sequence.saturating_add(1);
        self.sink.on_sfm_event(event);
    }

    pub fn checkpoint(&self) -> Result<(), SfmTaskStop> {
        self.control.checkpoint()
    }
}
