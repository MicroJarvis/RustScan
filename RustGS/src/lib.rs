//! RustGS - 3D Gaussian Splatting Training Library
//!
//! This crate provides offline 3DGS training capabilities for RustScan.
//! It takes images and camera poses as input and outputs trained splat sets.
//!
//! # Architecture
//!
//! - `core`: shared training-neutral types such as cameras
//! - `training`: Training loops and optimizers
//! - `io`: Scene file I/O (.splat, PLY, checkpoints)
//! - `init`: Splat initialization from point clouds
//!
//! # Example
//!
//! ```no_run
//! use rustgs::{load_colmap_training_dataset, train_splats, ColmapConfig, TrainingConfig, TrainingOptions};
//! use std::path::PathBuf;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let path = PathBuf::from("colmap_dataset");
//!
//! // Load a COLMAP reconstruction with sparse points.
//! let dataset = load_colmap_training_dataset(&path, &ColmapConfig::default())?;
//!
//! // Train 3DGS splats.
//! let config = TrainingConfig::default();
//! let run = train_splats(&dataset, &config, TrainingOptions::default())?;
//!
//! // Save the trained splats.
//! rustgs::save_splats(
//!     "scene.splat".as_ref(),
//!     &run.splats,
//!     &rustgs::SplatMetadata::default(),
//! )?;
//! # Ok(())
//! # }
//! ```

pub mod core;
pub mod init;
pub mod io;
mod sh;
pub mod training;
#[cfg(feature = "gpu")]
pub mod viewport;

use std::path::Path;

pub use rustscan_types::{Intrinsics, MapPointData, ScenePose, TrainingDataset, SE3};

// Re-export core types
pub use crate::core::{GaussianCamera, HostSplats, SplatView};

// Re-export training types
pub use crate::training::{
    compare_loss_curve_samples, default_litegs_parity_fixtures, default_parity_report_path,
    parity_fixture_id_for_input_path, resolve_litegs_parity_fixture_input_path,
    resolve_litegs_parity_reference_report_path, EvaluationDevice, EvaluationFrameMetric,
    FinalTrainingMetrics, LiteGsCameraConfig, LiteGsConfig, LiteGsFeatureConfig,
    LiteGsGrowthConfig, LiteGsOpacityResetMode, LiteGsPruneMode, LiteGsPruningConfig,
    LiteGsRefineConfig, LiteGsRenderingConfig, LiteGsSplitScoreMode, LiteGsTileSize,
    LiteGsTopologyConfig, LiteGsTrainingProfile, ParityCheckOutcome, ParityCheckStatus,
    ParityFixtureKind, ParityFixtureSpec, ParityFloatDistribution, ParityGateEvaluation,
    ParityGateStatus, ParityHarnessReport, ParityLossCurveSample, ParityLossTerms,
    ParityMetricSnapshot, ParityReferenceComparison, ParityThresholds, ParityTimingMetrics,
    ParityTopologyMetrics, ParityTopologyStepSample, PsnrSummary, SplatEvaluationConfig,
    SplatEvaluationError, SplatEvaluationResult, SplatEvaluationSummary, TrainingDataConfig,
    TrainingInitializationConfig, TrainingLossConfig, TrainingOptimizerConfig,
    TrainingRasterConfig, DEFAULT_CONVERGENCE_FIXTURE_ID, DEFAULT_RASTER_COV_BLUR,
    DEFAULT_TINY_FIXTURE_ID,
};
pub use crate::training::{
    compute_psnr_f32, scaled_dimensions, select_evaluation_frames, summarize_psnr_samples,
    summarize_training_metrics, worst_frame_metrics,
};
#[cfg(feature = "gpu")]
pub use crate::training::{
    evaluate_splats, evaluation_device, last_training_telemetry, render_evaluation_frame,
    runtime_from_splats, LiteGsOptimizerLrs, LiteGsTrainingTelemetry, TrainingControl,
    TrainingEvent, TrainingEventCadence, TrainingEventRoute, TrainingIterationProgress,
    TrainingOptions, TrainingPlanSelected, TrainingRun, TrainingRunCancelled, TrainingRunCompleted,
    TrainingRunReport, TrainingRunStarted, TrainingSnapshotReady,
};
#[cfg(feature = "gpu")]
pub use crate::training::{SharedWgpuContext, SplatEvaluationRenderer};
pub use crate::training::{TrainingBackend, TrainingConfig, TrainingResult};
#[cfg(feature = "gpu")]
pub use crate::viewport::{BurnViewportRenderer, BurnViewportResolution};

// Re-export IO types
pub use crate::io::colmap_dataset::{load_colmap_dataset, ColmapConfig};
#[cfg(feature = "gpu")]
pub use crate::io::scene_io::{
    load_splats, load_splats_ply, load_splats_splat, save_splats, save_splats_ply,
    save_splats_splat,
};
pub use crate::io::scene_io::{SceneIoError, SplatMetadata};
#[cfg(feature = "gpu")]
pub use crate::io::TrainingCheckpoint;

// Re-export initialization types
#[cfg(feature = "gpu")]
pub use crate::init::initialize_host_splats_from_points;
pub use crate::init::GaussianInitConfig;

#[cfg(not(feature = "gpu"))]
pub fn gpu_available() -> bool {
    false
}

#[cfg(feature = "gpu")]
pub fn gpu_available() -> bool {
    true
}

/// Supported training input format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingInputKind {
    Colmap,
}

impl std::fmt::Display for TrainingInputKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Colmap => write!(f, "COLMAP dataset"),
        }
    }
}

/// Load a COLMAP training dataset and report the resolved input kind.
pub fn load_colmap_training_dataset_with_source(
    input: &Path,
    colmap_config: &ColmapConfig,
) -> Result<(TrainingDataset, TrainingInputKind), TrainingError> {
    if !input.is_dir() {
        return Err(TrainingError::InvalidInput(format!(
            "{} is not a COLMAP dataset directory",
            input.display()
        )));
    }

    load_colmap_dataset(input, colmap_config).map(|dataset| (dataset, TrainingInputKind::Colmap))
}

/// Load a COLMAP training dataset.
pub fn load_colmap_training_dataset(
    input: &Path,
    colmap_config: &ColmapConfig,
) -> Result<TrainingDataset, TrainingError> {
    load_colmap_training_dataset_with_source(input, colmap_config).map(|(dataset, _)| dataset)
}

/// Train 3DGS splats from a prepared training dataset.
#[cfg(feature = "gpu")]
pub fn train_splats(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
    options: TrainingOptions<'_>,
) -> Result<TrainingRun, TrainingError> {
    training::train_splats(dataset, config, options)
}

/// Training error type.
#[derive(Debug, thiserror::Error)]
pub enum TrainingError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("GPU error: {0}")]
    Gpu(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Training failed: {0}")]
    TrainingFailed(String),
}
