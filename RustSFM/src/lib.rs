pub mod ba;
pub mod sequence_registration;
pub mod task;

pub use ba::{BundleAdjustmentLinearSolverPreference, BundleAdjustmentSparseLinearAlgebra};
pub use sequence_registration::{
    FrameRegistrationDiagnostic, FrameRegistrationStatus, RegistrationRound, SequenceFrame,
    SequenceRegistrationConfig, SequenceRegistrationError, SequenceRegistrationPlan,
    SequenceRegistrationResult, MAX_SEQUENCE_PLAN_FRAMES,
};
pub use task::{
    SfmControlState, SfmTaskContext, SfmTaskControl, SfmTaskEvent, SfmTaskEventKind,
    SfmTaskEventSink, SfmTaskIssue, SfmTaskOperation, SfmTaskStage, SfmTaskStop,
};

// I/O, COLMAP compatibility, and persistent graph/cache formats.
#[path = "io/colmap.rs"]
pub mod colmap;
#[path = "io/colmap_image.rs"]
pub mod colmap_image;
#[path = "io/database.rs"]
pub mod database;

// Core shared data structures.
#[path = "core/correspondence_graph.rs"]
pub mod correspondence_graph;
#[path = "core/types.rs"]
pub mod types;

// Feature extraction, matching, and retrieval.
#[path = "feature/feature_extraction.rs"]
pub mod feature_extraction;
#[path = "feature/feature_matching.rs"]
pub mod feature_matching;
#[path = "feature/feature_matching_db.rs"]
mod feature_matching_db;
#[path = "feature/retrieval.rs"]
pub mod retrieval;
#[path = "feature/sift.rs"]
pub mod sift;
#[path = "feature/sift_index.rs"]
mod sift_index;
#[path = "feature/wide.rs"]
pub mod wide;

// Geometry, estimators, robust fitting, and numerical helpers.
#[path = "geometry/colmap_eigen.rs"]
mod colmap_eigen;
#[path = "geometry/five_point.rs"]
pub mod five_point;
#[path = "geometry/five_point_generated.rs"]
mod five_point_generated;
#[path = "geometry/generalized_pose.rs"]
pub mod generalized_pose;
#[path = "geometry/geometry.rs"]
pub mod geometry;
#[path = "geometry/least_absolute_deviations.rs"]
pub mod least_absolute_deviations;
#[path = "geometry/polynomial.rs"]
pub mod polynomial;
#[path = "geometry/sparse_cholesky.rs"]
pub mod sparse_cholesky;
#[path = "geometry/sprt.rs"]
pub mod sprt;
#[path = "geometry/support_measurement.rs"]
pub mod support_measurement;
#[path = "geometry/triangulation.rs"]
pub mod triangulation;
#[path = "geometry/triangulation_estimator.rs"]
pub mod triangulation_estimator;
#[path = "geometry/two_view.rs"]
pub mod two_view;

// Sparse SfM pipelines and reconstruction orchestration.
#[path = "sfm/global_mapper.rs"]
pub mod global_mapper;
#[path = "sfm/global_positioning.rs"]
pub mod global_positioning;
#[path = "sfm/incremental_triangulator.rs"]
pub mod incremental_triangulator;
#[path = "sfm/joint_global_positioning.rs"]
pub mod joint_global_positioning;
#[path = "sfm/mapper.rs"]
pub mod mapper;
#[path = "sfm/observation_manager.rs"]
pub mod observation_manager;
#[path = "sfm/pose_graph.rs"]
pub mod pose_graph;
#[path = "sfm/rotation_averaging.rs"]
pub mod rotation_averaging;
#[path = "sfm/track_establishment.rs"]
pub mod track_establishment;
#[path = "sfm/track_triangulation.rs"]
pub mod track_triangulation;
#[path = "sfm/view_graph_calibration.rs"]
pub mod view_graph_calibration;
#[path = "sfm/view_graph_splitting.rs"]
pub mod view_graph_splitting;
#[path = "sfm/visibility_pyramid.rs"]
pub mod visibility_pyramid;

// Diagnostics and validation harnesses.
#[path = "diagnostics/compare.rs"]
pub mod compare;
#[path = "diagnostics/parity.rs"]
pub mod parity;

pub mod gpu;

pub use compare::{
    compare_colmap, compare_colmap_stages, parse_compare_stages, CompareReport, CompareStage,
    CompareStagesReport,
};
pub use feature_extraction::{
    compare_extracted_sift_features, extract_features_to_database,
    extract_features_to_database_with_extractor,
    extract_features_to_database_with_extractor_and_task, ExtractFeaturesReport,
    SiftFeatureExtractor,
};
pub use feature_matching::{generate_matching_pairs, MatchingPairStrategy};
pub use feature_matching_db::{
    debug_two_view_database_pair, match_features_to_database, match_features_to_database_with_task,
    DebugTwoViewOptions, DebugTwoViewReport, MatchFeaturesOptions, MatchFeaturesReport,
    MatchFeaturesVerifierEvent, MatchFeaturesVerifierTrace,
};
pub use global_mapper::{
    pairwise_matches_from_pairs, run_global_mapper, run_global_reconstruction,
    run_global_reconstructions, GlobalMapperOptions, GlobalMapperResult,
    GlobalReconstructionOptions, GlobalReconstructionResult, GlobalReconstructionsResult,
    GlobalRefinementOptions, GlobalStructureRefinementStats,
};
pub use global_positioning::{
    estimate_global_positions, relative_translations_from_pairs, GlobalPositioningOptions,
    GlobalPositioningResult, RelativeTranslation,
};
pub use joint_global_positioning::{
    build_ray_observations, estimate_joint_global_positions, JointGlobalPositioningOptions,
    JointGlobalPositioningResult, JointGlobalPositioningSolver,
};
pub use mapper::{
    reference_camera_setup, run_incremental_pipeline, run_reconstruction,
    run_reconstruction_with_callbacks, run_reconstruction_with_task, FeatureType,
    ImageSelectionMethod, IncrementalPipelineCallback, IncrementalPipelineResult,
    IncrementalPipelineStatus, MapperConfig, PipelineCallbackEvent, PipelineCallbackSink,
    ReconstructionSeed, ReconstructionSummary, ReferenceCameraSetup,
};
pub use parity::{compare_database_parity, ParityReport};
pub use retrieval::{
    build_vocab_tree_pairs, descriptors_u8_to_f32, generate_vocab_tree_pairs, ImageScore,
    VisualIndex, VocabTree, VocabTreeBuildOptions, VocabTreePairOptions,
};
pub use rotation_averaging::{
    estimate_global_rotations, relative_rotations_from_pairs, RelativeRotation,
    RotationAveragingOptions, RotationAveragingResult,
};
pub use sift::{benchmark_sift_extraction, SiftBenchmarkReport, SiftExtractionOptions};
pub use track_establishment::{
    establish_tracks, FeatureNode, PairwiseMatches, Track, TrackEstablishmentOptions,
    TrackEstablishmentStats,
};
pub use track_triangulation::{
    track_to_observations, triangulate_tracks, TrackTriangulationOptions, TrackTriangulationStats,
};
pub use view_graph_calibration::{
    calibrate_view_graph, filter_rotation_inconsistent_pairs, rotation_edge_error_deg,
    ViewGraphCalibrationOptions, ViewGraphCalibrationStats,
};
pub use view_graph_splitting::{
    components_for_reconstruction, find_view_graph_components, remap_pairs_for_component,
    subset_frames_for_component, ViewGraphComponent, ViewGraphComponentSplittingOptions,
    ViewGraphComponentSplittingStats,
};
