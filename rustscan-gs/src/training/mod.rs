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
    load_training_checkpoint, load_training_checkpoint_with_migration, save_training_checkpoint,
    AdamCheckpoint, AdamParameterCheckpoint, CheckpointMigration, TensorCheckpoint,
    TopologyCheckpoint, TrainingCheckpoint, TrainingIdentity, MAX_TRAINING_CHECKPOINT_BYTES,
    MAX_TRAINING_CHECKPOINT_SPLATS, MAX_TRAINING_CHECKPOINT_TENSOR_ELEMENTS,
    MAX_TRAINING_CHECKPOINT_TENSOR_RANK, MAX_TRAINING_IDENTITY_BYTES,
    TRAINING_CHECKPOINT_FORMAT_VERSION, TRAINING_CHECKPOINT_MAGIC, TRAINING_CHECKPOINT_VERSION,
    TRAINING_CHECKPOINT_VERSION_V1,
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
pub use evaluation::{SharedWgpuContext, SplatEvaluationRenderOutput, SplatEvaluationRenderer};
#[cfg(feature = "gpu")]
pub use events::{
    TrainingCheckpointPolicy, TrainingCheckpointReady, TrainingCheckpointReason,
    TrainingCheckpointSink, TrainingControl, TrainingEvent, TrainingEventCadence,
    TrainingEventRoute, TrainingIterationProgress, TrainingOptions, TrainingPlanSelected,
    TrainingRun, TrainingRunCancelled, TrainingRunCompleted, TrainingRunDisposition,
    TrainingRunFailed, TrainingRunPaused, TrainingRunReport, TrainingRunStarted,
    TrainingSnapshotReady,
};
#[cfg(feature = "gpu")]
pub use reporting::gpu_profiler::{
    assert_report_self_consistent, optimization_gpu_fields_from_profiler, probe_wgpu_environment,
    span as gpu_profiler_span, GpuEnvironmentProbe, GpuProfilerReport, OptimizationGpuFields,
    PipelineSpanStats, PipelineTimingCollector, PIPELINE_TIMING_WARMUP_SAMPLES,
    UNSUPPORTED_TIMESTAMP_QUERY_UNAVAILABLE,
};
pub use reporting::metrics::{
    accumulate_sticky_forward_overflow, allows_state_mutation, step_intersection_overflowed,
    ForwardCapacityTelemetry, ParityFloatDistribution, ParityLossCurveSample, ParityLossTerms,
    ParityTopologyMetrics, ParityTopologyStepSample, StickyForwardOverflow,
};
pub use reporting::optimization_report::{
    build_optimization_report, canonical_config_fingerprint, compare_optimization_reports,
    current_peak_rss_bytes, default_optimization_report_path, duration_millis,
    load_optimization_report, percentile_f64, percentile_usize, sh_schedule_fingerprint,
    write_optimization_report, OptimizationCommand, OptimizationCompareDecision,
    OptimizationCompareResult, OptimizationEnvironment, OptimizationEvalFrame,
    OptimizationEvaluationMetrics, OptimizationMemoryMetrics, OptimizationMetricDelta,
    OptimizationReport, OptimizationTopologyMetrics, OptimizationTrainMetrics,
};

pub use config::{
    DynamicMaskGradient, LiteGsCameraConfig, LiteGsConfig, LiteGsFeatureConfig, LiteGsGrowthConfig,
    LiteGsOpacityResetMode, LiteGsPruneMode, LiteGsPruningConfig, LiteGsRefineConfig,
    LiteGsRenderingConfig, LiteGsSplitScoreMode, LiteGsTileSize, LiteGsTopologyConfig,
    LiteGsTrainingProfile, TrainingBackend, TrainingConfig, TrainingDataConfig,
    TrainingInitializationConfig, TrainingLossConfig, TrainingOptimizerConfig,
    TrainingProfilerConfig, TrainingRasterConfig, TrainingResult, DEFAULT_RASTER_COV_BLUR,
    MAX_TRAINING_ITERATIONS,
};
#[cfg(feature = "gpu")]
pub use reporting::telemetry::{
    last_training_telemetry, LiteGsOptimizerLrs, LiteGsTrainingTelemetry,
};

#[cfg(feature = "gpu")]
pub use data::frame_loader::training_frame_order;
#[cfg(feature = "gpu")]
pub use forward::parity::run_bounded_forward_parity_suite;

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
