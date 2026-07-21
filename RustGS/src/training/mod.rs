//! Training module for 3D Gaussian Splatting.

mod checkpoint;
mod config;
mod evaluation;
mod reporting;

macro_rules! gpu_modules {
    ($($module:ident),+ $(,)?) => {
        $(
            #[cfg(feature = "gpu")]
            mod $module;
        )+
    };
}

gpu_modules!(backward, data, events, gpu_primitives, topology,);

#[cfg(feature = "gpu")]
pub(crate) mod engine;
#[cfg(feature = "gpu")]
pub(crate) mod forward;

#[cfg(feature = "gpu")]
use crate::{TrainingDataset, TrainingError};

pub use checkpoint::{
    load_training_checkpoint, save_training_checkpoint, AdamCheckpoint, AdamParameterCheckpoint,
    TensorCheckpoint, TopologyCheckpoint, TrainingCheckpoint, TrainingIdentity,
    TRAINING_CHECKPOINT_VERSION,
};
pub use evaluation::MIN_RENDER_SCALE;
pub use evaluation::{
    compare_loss_curve_samples, default_litegs_parity_fixtures, default_parity_report_path,
    parity_fixture_id_for_input_path, resolve_litegs_parity_fixture_input_path,
    resolve_litegs_parity_reference_report_path, ParityCheckOutcome, ParityCheckStatus,
    ParityFixtureKind, ParityFixtureSpec, ParityGateEvaluation, ParityGateStatus,
    ParityHarnessReport, ParityMetricSnapshot, ParityReferenceComparison, ParityThresholds,
    ParityTimingMetrics, DEFAULT_CONVERGENCE_FIXTURE_ID, DEFAULT_TINY_FIXTURE_ID,
};
pub use evaluation::{
    compute_psnr_f32, scaled_dimensions, select_evaluation_frames, summarize_psnr_samples,
    summarize_training_metrics, worst_frame_metrics, EvaluationDevice, EvaluationFrameMetric,
    FinalTrainingMetrics, PsnrSummary, SplatEvaluationConfig, SplatEvaluationError,
    SplatEvaluationResult, SplatEvaluationSummary,
};
#[cfg(feature = "gpu")]
pub use evaluation::{
    evaluate_splats, evaluation_device, render_evaluation_frame, runtime_from_splats,
};
#[cfg(feature = "gpu")]
pub use evaluation::{SharedWgpuContext, SplatEvaluationRenderer};
#[cfg(feature = "gpu")]
pub use events::{
    TrainingControl, TrainingEvent, TrainingEventCadence, TrainingEventRoute,
    TrainingIterationProgress, TrainingOptions, TrainingPlanSelected, TrainingRun,
    TrainingRunCancelled, TrainingRunCompleted, TrainingRunFailed, TrainingRunReport,
    TrainingRunStarted, TrainingSnapshotReady,
};
pub use reporting::metrics::{
    ParityFloatDistribution, ParityLossCurveSample, ParityLossTerms, ParityTopologyMetrics,
    ParityTopologyStepSample,
};

pub use config::{
    LiteGsCameraConfig, LiteGsConfig, LiteGsFeatureConfig, LiteGsGrowthConfig,
    LiteGsOpacityResetMode, LiteGsPruneMode, LiteGsPruningConfig, LiteGsRefineConfig,
    LiteGsRenderingConfig, LiteGsSplitScoreMode, LiteGsTileSize, LiteGsTopologyConfig,
    LiteGsTrainingProfile, TrainingBackend, TrainingConfig, TrainingDataConfig,
    TrainingInitializationConfig, TrainingLossConfig, TrainingOptimizerConfig,
    TrainingRasterConfig, TrainingResult, DEFAULT_RASTER_COV_BLUR,
};
#[cfg(feature = "gpu")]
pub use reporting::telemetry::{
    last_training_telemetry, LiteGsOptimizerLrs, LiteGsTrainingTelemetry,
};

#[cfg(feature = "gpu")]
pub fn train_splats(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
    options: TrainingOptions<'_>,
) -> Result<TrainingRun, TrainingError> {
    reporting::telemetry::store_last_training_telemetry(None);
    config.validate()?;
    engine::train_splats(dataset, config, options)
}
