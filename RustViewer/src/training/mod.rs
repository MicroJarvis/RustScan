//! Training-specific bridge layers for RustViewer.

pub mod gpu_viewport;
pub mod preview;
pub mod session;

pub use session::{
    TrainingControlOptions, TrainingManager, TrainingProgress, TrainingRunner, TrainingSession,
    TrainingSessionError, TrainingSessionEvent, TrainingSessionState,
};
