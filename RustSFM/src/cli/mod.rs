mod commands;
mod feature_database;
mod project;
mod support;

use clap::{Parser, Subcommand};
use rustsfm::{
    BundleAdjustmentLinearSolverPreference, BundleAdjustmentSparseLinearAlgebra, FeatureType,
    ImageSelectionMethod,
};
use std::path::PathBuf;
#[derive(Debug, Parser)]
#[command(name = "rustsfm")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Reconstruct(ReconstructArgs),
    Compare(CompareArgs),
    CompareExtract(CompareExtractArgs),
    Parity(ParityArgs),
    BenchmarkSift(BenchmarkSiftArgs),
    BenchmarkMatchPairs(BenchmarkMatchPairsArgs),
    ExtractFeatures(ExtractFeaturesArgs),
    MatchFeatures(MatchFeaturesArgs),
    DebugTwoview(DebugTwoViewArgs),
    #[command(name = "feature_extractor")]
    FeatureExtractor(ColmapFeatureExtractorArgs),
    #[command(name = "exhaustive_matcher")]
    ExhaustiveMatcher(ColmapMatcherArgs),
    #[command(name = "sequential_matcher")]
    SequentialMatcher(ColmapSequentialMatcherArgs),
    #[command(name = "vocab_tree_matcher")]
    VocabTreeMatcher(ColmapVocabTreeMatcherArgs),
    #[command(name = "geometric_verifier")]
    GeometricVerifier(ColmapGeometricVerifierArgs),
    Mapper(ColmapMapperArgs),
    #[command(name = "model_converter")]
    ModelConverter(ModelConverterArgs),
}

#[derive(Debug, Parser)]
struct ReconstructArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    reference: Option<PathBuf>,
    #[arg(long)]
    database: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    write_two_view_geometries: bool,
    #[arg(long, default_value_t = false)]
    ignore_database_two_view_poses: bool,
    #[arg(long, default_value_t = false)]
    write_database: bool,
    #[arg(long)]
    max_images: Option<usize>,
    #[arg(long, default_value_t = false)]
    single_model: bool,
    #[arg(long, default_value = "50")]
    max_num_models: usize,
    #[arg(long, default_value = "20")]
    max_model_overlap: usize,
    #[arg(long, default_value = "10")]
    min_model_size: usize,
    #[arg(long)]
    snapshot_path: Option<PathBuf>,
    #[arg(long, default_value = "0")]
    snapshot_frames_freq: usize,
    #[arg(long, default_value_t = false)]
    no_extract_colors: bool,
    #[arg(long, default_value_t = false)]
    fix_existing_frames: bool,
    #[arg(long, default_value = "sift")]
    feature_type: FeatureType,
    #[arg(long, default_value = "8192")]
    max_features: usize,
    #[arg(long, default_value_t = false)]
    sift_estimate_affine_shape: bool,
    #[arg(long, default_value_t = false)]
    sift_domain_size_pooling: bool,
    #[arg(long, default_value_t = false)]
    sift_force_covariant: bool,
    #[arg(long, default_value_t = false)]
    sift_cpu_brute_force_matcher: bool,
    #[arg(long, default_value_t = false)]
    local_matching: bool,
    #[arg(long, default_value = "3")]
    local_window: usize,
    #[arg(long, default_value = "sequential")]
    matching_strategy: String,
    #[arg(long, default_value = "10")]
    sequential_overlap: usize,
    #[arg(long, default_value_t = true)]
    sequential_quadratic_overlap: bool,
    #[arg(long, default_value_t = false)]
    sequential_loop_detection: bool,
    #[arg(long, default_value = "10")]
    sequential_loop_detection_period: usize,
    #[arg(long, default_value = "100")]
    vocab_tree_num_images: usize,
    #[arg(long, default_value_t = false)]
    guided_matching: bool,
    #[arg(long, default_value = "2.0")]
    guided_max_epipolar_error_px: f32,
    #[arg(long, default_value_t = false)]
    experimental_sequence_heuristics: bool,
    #[arg(long, default_value_t = false)]
    experimental_ring_closure: bool,
    #[arg(long, default_value_t = false)]
    experimental_structureless_pair_pose_fallback: bool,
    #[arg(long, default_value = "0.8")]
    match_ratio: f64,
    #[arg(long, default_value = "160")]
    max_hamming_distance: f32,
    #[arg(long, default_value = "15")]
    min_matches: usize,
    #[arg(long, default_value = "15")]
    min_inliers: usize,
    #[arg(long, default_value = "4")]
    min_triangulated: usize,
    #[arg(long, default_value = "100")]
    init_min_num_inliers: usize,
    #[arg(long, default_value = "16.0")]
    init_min_tri_angle_deg: f32,
    #[arg(long, default_value = "0.95")]
    init_max_forward_motion: f32,
    #[arg(long, default_value = "200")]
    init_num_trials: usize,
    #[arg(long, default_value = "2")]
    init_max_reg_trials: usize,
    #[arg(long, default_value = "4.0")]
    essential_threshold_px: f32,
    #[arg(long, default_value = "10000")]
    essential_iterations: u32,
    #[arg(long, default_value = "12.0")]
    pnp_threshold_px: f32,
    #[arg(long, default_value = "10000")]
    pnp_iterations: u32,
    #[arg(long, default_value_t = false)]
    use_gpu_pnp: bool,
    #[arg(long, default_value = "30")]
    abs_pose_min_num_inliers: usize,
    #[arg(long, default_value = "0.25")]
    abs_pose_min_inlier_ratio: f32,
    #[arg(long, default_value = "min_uncertainty")]
    image_selection_method: ImageSelectionMethod,
    #[arg(long, default_value = "-1")]
    random_seed: i32,
    #[arg(long, default_value = "3")]
    max_reg_trials: usize,
    #[arg(long, default_value_t = false)]
    no_local_ba: bool,
    #[arg(long, default_value = "6")]
    local_ba_num_images: usize,
    #[arg(long, default_value = "15")]
    local_ba_min_shared_points: usize,
    #[arg(long, default_value = "25")]
    local_ba_iterations: usize,
    #[arg(long, default_value = "2")]
    local_ba_max_refinements: usize,
    #[arg(long, default_value = "0.001")]
    local_ba_max_refinement_change: f32,
    #[arg(long, default_value_t = false)]
    no_global_ba: bool,
    #[arg(long, default_value = "auto")]
    ba_linear_solver: BundleAdjustmentLinearSolverPreference,
    #[arg(long, default_value = "auto")]
    ba_sparse_backend: BundleAdjustmentSparseLinearAlgebra,
    #[arg(long, default_value = "50")]
    global_ba_iterations: usize,
    #[arg(long, default_value = "1.5")]
    global_ba_images_ratio: f32,
    #[arg(long, default_value = "1.5")]
    global_ba_points_ratio: f32,
    #[arg(long, default_value = "500")]
    global_ba_images_freq: usize,
    #[arg(long, default_value = "250000")]
    global_ba_points_freq: usize,
    #[arg(long, default_value = "5")]
    global_ba_max_refinements: usize,
    #[arg(long, default_value = "0.0005")]
    global_ba_max_refinement_change: f32,
    #[arg(long, default_value_t = false)]
    global_ba_ignore_redundant_points3d: bool,
    #[arg(long, default_value = "0.05")]
    global_ba_ignore_redundant_points3d_min_coverage_gain: f64,
    #[arg(long, default_value_t = false)]
    no_ba_refine_focal_length: bool,
    #[arg(long, default_value_t = false)]
    ba_refine_principal_point: bool,
    #[arg(long, default_value_t = false)]
    no_ba_refine_extra_params: bool,
    #[arg(long = "ba-constant-rig-id", value_delimiter = ',')]
    ba_constant_rig_ids: Vec<u32>,
    #[arg(long = "ba-constant-camera-id", value_delimiter = ',')]
    ba_constant_camera_ids: Vec<u32>,
    #[arg(long, default_value = "0.1")]
    min_focal_length_ratio: f64,
    #[arg(long, default_value = "10.0")]
    max_focal_length_ratio: f64,
    #[arg(long, default_value = "1.0")]
    max_extra_param: f64,
    #[arg(long, default_value = "8.0")]
    max_reprojection_error_px: f32,
    #[arg(long, default_value_t = true)]
    ignore_two_view_tracks: bool,
    #[arg(long, default_value_t = false)]
    pose_graph: bool,
    #[arg(long, default_value_t = false)]
    global_mapper: bool,
    #[arg(long)]
    fx: Option<f32>,
    #[arg(long)]
    fy: Option<f32>,
    #[arg(long)]
    cx: Option<f32>,
    #[arg(long)]
    cy: Option<f32>,
    #[arg(long)]
    threads: Option<usize>,
    #[arg(long, default_value_t = false)]
    no_copy_images: bool,
    #[arg(long)]
    summary_json: Option<PathBuf>,
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Debug, Parser)]
struct CompareArgs {
    #[arg(long)]
    reference: PathBuf,
    #[arg(long)]
    candidate: PathBuf,
    #[arg(long)]
    reference_database: Option<PathBuf>,
    #[arg(long)]
    candidate_database: Option<PathBuf>,
    #[arg(long, value_delimiter = ',', default_value = "all")]
    stages: Vec<String>,
    #[arg(long)]
    output_json: Option<PathBuf>,
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Debug, Parser)]
struct CompareExtractArgs {
    #[arg(long)]
    reference_database: PathBuf,
    #[arg(long)]
    images: PathBuf,
    #[arg(long, default_value = "8192")]
    max_features: usize,
    #[arg(long, default_value_t = false)]
    sift_estimate_affine_shape: bool,
    #[arg(long, default_value_t = false)]
    sift_domain_size_pooling: bool,
    #[arg(long, default_value_t = false)]
    sift_force_covariant: bool,
    #[arg(long)]
    output_json: Option<PathBuf>,
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Debug, Parser)]
struct ExtractFeaturesArgs {
    #[arg(long)]
    database: PathBuf,
    #[arg(long)]
    images: PathBuf,
    #[arg(long, default_value = "8192")]
    max_features: usize,
    #[arg(long, default_value_t = false)]
    sift_estimate_affine_shape: bool,
    #[arg(long, default_value_t = false)]
    sift_domain_size_pooling: bool,
    #[arg(long, default_value_t = false)]
    sift_force_covariant: bool,
    #[arg(long, default_value_t = false)]
    use_gpu: bool,
    #[arg(long)]
    output_json: Option<PathBuf>,
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Debug, Parser)]
struct MatchFeaturesArgs {
    #[arg(long)]
    database: PathBuf,
    #[arg(long, default_value = "sequential")]
    matching_strategy: String,
    #[arg(long, default_value = "10")]
    sequential_overlap: usize,
    #[arg(long, default_value_t = true)]
    sequential_quadratic_overlap: bool,
    #[arg(long, default_value_t = false)]
    sequential_loop_detection: bool,
    #[arg(long, default_value = "10")]
    sequential_loop_detection_period: usize,
    #[arg(long, default_value = "100")]
    vocab_tree_num_images: usize,
    #[arg(long, default_value = "0.8")]
    match_ratio: f64,
    #[arg(long, default_value_t = false)]
    sift_cpu_brute_force_matcher: bool,
    #[arg(long, default_value_t = false)]
    use_gpu: bool,
    #[arg(long, default_value = "15")]
    min_num_matches: usize,
    #[arg(long, default_value = "4.0")]
    essential_threshold_px: f32,
    #[arg(long, default_value = "10000")]
    essential_iterations: u32,
    #[arg(long, default_value_t = true)]
    clear_existing: bool,
    #[arg(long, default_value_t = false)]
    use_existing_matches: bool,
    #[arg(long, default_value = "1000")]
    existing_match_batch_size: usize,
    #[arg(long, default_value = "-1")]
    random_seed: i32,
    #[arg(long)]
    output_json: Option<PathBuf>,
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Debug, Parser)]
struct BenchmarkMatchPairsArgs {
    #[arg(long)]
    database: PathBuf,
    #[arg(long, default_value = "5")]
    window: usize,
    #[arg(long)]
    pair_limit: Option<usize>,
    #[arg(long, default_value = "1")]
    repetitions: usize,
    #[arg(long, default_value_t = false)]
    use_gpu: bool,
    #[arg(long, default_value = "0")]
    random_seed: i32,
    #[arg(long)]
    output_json: Option<PathBuf>,
    #[arg(long)]
    artifacts_dir: Option<PathBuf>,
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Debug, Parser)]
struct ColmapFeatureExtractorArgs {
    #[arg(long = "database_path", alias = "database-path")]
    database_path: PathBuf,
    #[arg(long = "image_path", alias = "image-path")]
    image_path: PathBuf,
    #[arg(long = "ImageReader.single_camera", default_value = "1")]
    single_camera: i32,
    #[arg(long = "ImageReader.camera_model", default_value = "PINHOLE")]
    camera_model: String,
    #[arg(long = "SiftExtraction.max_num_features", default_value = "8192")]
    max_num_features: usize,
    #[arg(long = "SiftExtraction.estimate_affine_shape", default_value = "0")]
    estimate_affine_shape: i32,
    #[arg(long = "SiftExtraction.domain_size_pooling", default_value = "0")]
    domain_size_pooling: i32,
    #[arg(long = "SiftExtraction.use_gpu")]
    use_gpu: Option<i32>,
    #[arg(long)]
    output_json: Option<PathBuf>,
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Debug, Parser)]
struct ColmapMatcherArgs {
    #[arg(long = "database_path", alias = "database-path")]
    database_path: PathBuf,
    #[arg(long = "SiftMatching.max_ratio", default_value = "0.8")]
    max_ratio: f64,
    #[arg(long = "SiftMatching.use_gpu")]
    use_gpu: Option<i32>,
    #[arg(long = "SiftMatching.guided_matching", default_value = "0")]
    guided_matching: i32,
    #[arg(long = "TwoViewGeometry.min_num_inliers", default_value = "15")]
    min_num_inliers: usize,
    #[arg(long = "TwoViewGeometry.max_error", default_value = "4.0")]
    max_error: f32,
    #[arg(long = "TwoViewGeometry.max_num_trials", default_value = "10000")]
    max_num_trials: u32,
    #[arg(long = "TwoViewGeometry.random_seed", default_value = "-1")]
    random_seed: i32,
    #[arg(long)]
    output_json: Option<PathBuf>,
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Debug, Parser)]
struct ColmapSequentialMatcherArgs {
    #[arg(long = "database_path", alias = "database-path")]
    database_path: PathBuf,
    #[arg(long = "SequentialMatching.overlap", default_value = "10")]
    overlap: usize,
    #[arg(long = "SequentialMatching.quadratic_overlap", default_value = "1")]
    quadratic_overlap: i32,
    #[arg(long = "SequentialMatching.loop_detection", default_value = "0")]
    loop_detection: i32,
    #[arg(
        long = "SequentialMatching.loop_detection_period",
        default_value = "10"
    )]
    loop_detection_period: usize,
    #[arg(long = "SiftMatching.max_ratio", default_value = "0.8")]
    max_ratio: f64,
    #[arg(long = "SiftMatching.use_gpu")]
    use_gpu: Option<i32>,
    #[arg(long = "SiftMatching.guided_matching", default_value = "0")]
    guided_matching: i32,
    #[arg(long = "TwoViewGeometry.min_num_inliers", default_value = "15")]
    min_num_inliers: usize,
    #[arg(long = "TwoViewGeometry.max_error", default_value = "4.0")]
    max_error: f32,
    #[arg(long = "TwoViewGeometry.max_num_trials", default_value = "10000")]
    max_num_trials: u32,
    #[arg(long = "TwoViewGeometry.random_seed", default_value = "-1")]
    random_seed: i32,
    #[arg(long)]
    output_json: Option<PathBuf>,
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Debug, Parser)]
struct ColmapVocabTreeMatcherArgs {
    #[arg(long = "database_path", alias = "database-path")]
    database_path: PathBuf,
    #[arg(long = "VocabTreeMatching.num_images", default_value = "100")]
    num_images: usize,
    #[arg(long = "SiftMatching.max_ratio", default_value = "0.8")]
    max_ratio: f64,
    #[arg(long = "SiftMatching.use_gpu")]
    use_gpu: Option<i32>,
    #[arg(long = "SiftMatching.guided_matching", default_value = "0")]
    guided_matching: i32,
    #[arg(long = "TwoViewGeometry.min_num_inliers", default_value = "15")]
    min_num_inliers: usize,
    #[arg(long = "TwoViewGeometry.max_error", default_value = "4.0")]
    max_error: f32,
    #[arg(long = "TwoViewGeometry.max_num_trials", default_value = "10000")]
    max_num_trials: u32,
    #[arg(long = "TwoViewGeometry.random_seed", default_value = "-1")]
    random_seed: i32,
    #[arg(long)]
    output_json: Option<PathBuf>,
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Debug, Parser)]
struct ColmapGeometricVerifierArgs {
    #[arg(long = "database_path", alias = "database-path")]
    database_path: PathBuf,
    #[arg(long = "TwoViewGeometry.min_num_inliers", default_value = "15")]
    min_num_inliers: usize,
    #[arg(long = "TwoViewGeometry.max_error", default_value = "4.0")]
    max_error: f32,
    #[arg(long = "TwoViewGeometry.max_num_trials", default_value = "10000")]
    max_num_trials: u32,
    #[arg(long = "TwoViewGeometry.random_seed", default_value = "-1")]
    random_seed: i32,
    #[arg(long = "SiftMatching.guided_matching", default_value = "0")]
    guided_matching: i32,
    #[arg(long)]
    output_json: Option<PathBuf>,
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Debug, Parser)]
struct ColmapMapperArgs {
    #[arg(long = "project_path", alias = "project-path")]
    project_path: Option<PathBuf>,
    #[arg(long = "database_path", alias = "database-path")]
    database_path: Option<PathBuf>,
    #[arg(long = "image_path", alias = "image-path")]
    image_path: Option<PathBuf>,
    #[arg(long = "output_path", alias = "output-path")]
    output_path: Option<PathBuf>,
    #[arg(long = "Mapper.ba_refine_focal_length")]
    ba_refine_focal_length: Option<i32>,
    #[arg(long = "Mapper.ba_refine_principal_point")]
    ba_refine_principal_point: Option<i32>,
    #[arg(long = "Mapper.ba_refine_extra_params")]
    ba_refine_extra_params: Option<i32>,
    #[arg(long = "Mapper.ba_linear_solver")]
    ba_linear_solver: Option<String>,
    #[arg(long = "Mapper.ba_sparse_backend")]
    ba_sparse_backend: Option<String>,
    #[arg(long = "Mapper.multiple_models")]
    multiple_models: Option<i32>,
    #[arg(long = "Mapper.min_num_matches")]
    min_num_matches: Option<usize>,
    #[arg(long = "Mapper.max_num_models")]
    max_num_models: Option<usize>,
    #[arg(long = "Mapper.max_model_overlap")]
    max_model_overlap: Option<usize>,
    #[arg(long = "Mapper.min_model_size")]
    min_model_size: Option<usize>,
    #[arg(long = "Mapper.snapshot_path")]
    snapshot_path: Option<PathBuf>,
    #[arg(long = "Mapper.snapshot_frames_freq")]
    snapshot_frames_freq: Option<usize>,
    #[arg(long = "Mapper.fix_existing_frames")]
    fix_existing_frames: Option<i32>,
    #[arg(long = "Mapper.init_num_trials")]
    init_num_trials: Option<usize>,
    #[arg(long = "Mapper.init_min_num_inliers")]
    init_min_num_inliers: Option<usize>,
    #[arg(long = "Mapper.init_max_error")]
    init_max_error: Option<f32>,
    #[arg(long = "Mapper.init_max_forward_motion")]
    init_max_forward_motion: Option<f32>,
    #[arg(long = "Mapper.init_min_tri_angle")]
    init_min_tri_angle: Option<f32>,
    #[arg(long = "Mapper.init_max_reg_trials")]
    init_max_reg_trials: Option<usize>,
    #[arg(long = "Mapper.abs_pose_max_error")]
    abs_pose_max_error: Option<f32>,
    #[arg(long = "Mapper.abs_pose_min_num_inliers")]
    abs_pose_min_num_inliers: Option<usize>,
    #[arg(long = "Mapper.abs_pose_min_inlier_ratio")]
    abs_pose_min_inlier_ratio: Option<f32>,
    #[arg(long = "Mapper.use_gpu_pnp")]
    use_gpu_pnp: Option<i32>,
    #[arg(long = "Mapper.max_reg_trials")]
    max_reg_trials: Option<usize>,
    #[arg(long = "Mapper.ba_local_num_images")]
    local_ba_num_images: Option<usize>,
    #[arg(long = "Mapper.ba_global_frames_ratio")]
    global_ba_images_ratio: Option<f32>,
    #[arg(long = "Mapper.ba_global_points_ratio")]
    global_ba_points_ratio: Option<f32>,
    #[arg(long = "Mapper.ba_global_frames_freq")]
    global_ba_images_freq: Option<usize>,
    #[arg(long = "Mapper.ba_global_points_freq")]
    global_ba_points_freq: Option<usize>,
    #[arg(long = "Mapper.ba_global_max_num_iterations")]
    global_ba_iterations: Option<usize>,
    #[arg(long = "Mapper.ba_local_max_num_iterations")]
    local_ba_iterations: Option<usize>,
    #[arg(long = "Mapper.ba_global_max_refinements")]
    global_ba_max_refinements: Option<usize>,
    #[arg(long = "Mapper.ba_local_max_refinements")]
    local_ba_max_refinements: Option<usize>,
    #[arg(long = "Mapper.ba_global_max_refinement_change")]
    global_ba_max_refinement_change: Option<f32>,
    #[arg(long = "Mapper.ba_local_max_refinement_change")]
    local_ba_max_refinement_change: Option<f32>,
    #[arg(long = "Mapper.ba_global_ignore_redundant_points3D")]
    global_ba_ignore_redundant_points3d: Option<i32>,
    #[arg(long = "Mapper.ba_global_ignore_redundant_points3D_min_coverage_gain")]
    global_ba_ignore_redundant_points3d_min_coverage_gain: Option<f64>,
    #[arg(long = "Mapper.extract_colors")]
    extract_colors: Option<i32>,
    #[arg(long = "Mapper.min_focal_length_ratio")]
    min_focal_length_ratio: Option<f64>,
    #[arg(long = "Mapper.max_focal_length_ratio")]
    max_focal_length_ratio: Option<f64>,
    #[arg(long = "Mapper.max_extra_param")]
    max_extra_param: Option<f64>,
    #[arg(long = "Mapper.filter_max_reproj_error")]
    filter_max_reproj_error: Option<f32>,
    #[arg(long = "Mapper.tri_ignore_two_view_tracks")]
    tri_ignore_two_view_tracks: Option<i32>,
    #[arg(long = "Mapper.random_seed")]
    random_seed: Option<i32>,
    #[arg(long = "Mapper.num_threads")]
    num_threads: Option<isize>,
    #[arg(long)]
    summary_json: Option<PathBuf>,
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Debug, Parser)]
struct ModelConverterArgs {
    #[arg(long = "input_path", alias = "input-path")]
    input_path: PathBuf,
    #[arg(long = "output_path", alias = "output-path")]
    output_path: PathBuf,
    #[arg(long = "output_type", alias = "output-type", default_value = "TXT")]
    output_type: String,
    #[arg(long = "input_type", alias = "input-type")]
    input_type: Option<String>,
}

#[derive(Debug, Parser)]
struct DebugTwoViewArgs {
    #[arg(long)]
    database: PathBuf,
    #[arg(long)]
    image1: String,
    #[arg(long)]
    image2: String,
    #[arg(long, default_value = "4.0")]
    essential_threshold_px: f32,
    #[arg(long, default_value = "10000")]
    essential_iterations: u32,
    #[arg(long, default_value = "15")]
    min_inliers: usize,
    #[arg(long, default_value = "4")]
    min_triangulated: usize,
    #[arg(long, default_value = "-1")]
    random_seed: i32,
    #[arg(long)]
    output_json: Option<PathBuf>,
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Debug, Parser)]
struct ParityArgs {
    #[arg(long)]
    database: PathBuf,
    #[arg(long)]
    images: Option<PathBuf>,
    #[arg(long = "image-name")]
    image_names: Vec<String>,
    #[arg(long, default_value = "15")]
    min_matches: usize,
    #[arg(long, default_value_t = false)]
    ignore_watermarks: bool,
    #[arg(long, default_value_t = false)]
    load_all_images: bool,
    #[arg(long, default_value_t = false)]
    convert_pose_priors_to_enu: bool,
    #[arg(long)]
    output_json: Option<PathBuf>,
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Debug, Parser)]
struct BenchmarkSiftArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output_json: Option<PathBuf>,
    #[arg(long, default_value = "8192")]
    max_features: usize,
    #[arg(long, default_value_t = false)]
    sift_estimate_affine_shape: bool,
    #[arg(long, default_value_t = false)]
    sift_domain_size_pooling: bool,
    #[arg(long, default_value_t = false)]
    sift_force_covariant: bool,
    #[arg(long, default_value_t = false)]
    use_gpu: bool,
    #[arg(long, default_value = "info")]
    log_level: String,
}

pub fn run() -> anyhow::Result<()> {
    commands::run()
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn colmap_feature_extractor_accepts_gpu_one() {
        let cli = Cli::try_parse_from([
            "rustsfm",
            "feature_extractor",
            "--database_path",
            "database.db",
            "--image_path",
            "images",
            "--SiftExtraction.use_gpu",
            "1",
        ])
        .unwrap();
        let Commands::FeatureExtractor(args) = cli.command else {
            panic!("wrong command")
        };
        assert_eq!(args.use_gpu, Some(1));
    }

    #[test]
    fn native_extract_features_parses_use_gpu() {
        let cli = Cli::try_parse_from([
            "rustsfm",
            "extract-features",
            "--database",
            "database.db",
            "--images",
            "images",
            "--use-gpu",
        ])
        .unwrap();
        let Commands::ExtractFeatures(args) = cli.command else {
            panic!("wrong command")
        };
        assert!(args.use_gpu);
    }

    #[test]
    fn native_match_features_parses_use_gpu() {
        let cli = Cli::try_parse_from([
            "rustsfm",
            "match-features",
            "--database",
            "database.db",
            "--use-gpu",
        ])
        .unwrap();
        let Commands::MatchFeatures(args) = cli.command else {
            panic!("wrong command")
        };
        assert!(args.use_gpu);
    }

    #[test]
    fn benchmark_match_pairs_parses_bounded_repetitions() {
        let cli = Cli::try_parse_from([
            "rustsfm",
            "benchmark-match-pairs",
            "--database",
            "database.db",
            "--window",
            "5",
            "--pair-limit",
            "96",
            "--repetitions",
            "3",
            "--use-gpu",
            "--output-json",
            "report.json",
        ])
        .unwrap();
        let Commands::BenchmarkMatchPairs(args) = cli.command else {
            panic!("wrong command")
        };
        assert_eq!(args.window, 5);
        assert_eq!(args.pair_limit, Some(96));
        assert_eq!(args.repetitions, 3);
        assert!(args.use_gpu);
        assert_eq!(args.output_json, Some(PathBuf::from("report.json")));
    }

    #[test]
    fn benchmark_match_pairs_parses_artifacts_directory() {
        let cli = Cli::try_parse_from([
            "rustsfm",
            "benchmark-match-pairs",
            "--database",
            "database.db",
            "--artifacts-dir",
            "benchmark-artifacts",
        ])
        .unwrap();

        let Commands::BenchmarkMatchPairs(args) = cli.command else {
            panic!("wrong command")
        };
        assert_eq!(
            args.artifacts_dir,
            Some(PathBuf::from("benchmark-artifacts"))
        );
    }

    #[test]
    fn native_reconstruct_parses_gpu_pnp() {
        let cli = Cli::try_parse_from([
            "rustsfm",
            "reconstruct",
            "--input",
            "in",
            "--output",
            "out",
            "--use-gpu-pnp",
        ])
        .expect("native GPU PnP flag");
        let Commands::Reconstruct(args) = cli.command else {
            panic!("reconstruct command")
        };
        assert!(args.use_gpu_pnp);
    }

    #[test]
    fn native_reconstruct_parses_ba_backend_options() {
        let cli = Cli::try_parse_from([
            "rustsfm",
            "reconstruct",
            "--input",
            "in",
            "--output",
            "out",
            "--ba-linear-solver",
            "iterative_schur",
            "--ba-sparse-backend",
            "accelerate-sparse",
        ])
        .expect("native BA backend flags");
        let Commands::Reconstruct(args) = cli.command else {
            panic!("reconstruct command")
        };
        assert_eq!(
            args.ba_linear_solver,
            BundleAdjustmentLinearSolverPreference::IterativeSchur
        );
        assert_eq!(
            args.ba_sparse_backend,
            BundleAdjustmentSparseLinearAlgebra::AccelerateSparse
        );
    }

    #[test]
    fn native_reconstruct_uses_less_aggressive_global_ba_ratio_defaults() {
        let cli =
            Cli::try_parse_from(["rustsfm", "reconstruct", "--input", "in", "--output", "out"])
                .expect("native reconstruct defaults");
        let Commands::Reconstruct(args) = cli.command else {
            panic!("reconstruct command")
        };

        assert_eq!(args.global_ba_images_ratio, 1.5);
        assert_eq!(args.global_ba_points_ratio, 1.5);
    }

    #[test]
    fn colmap_mapper_parses_gpu_pnp() {
        let cli = Cli::try_parse_from(["rustsfm", "mapper", "--Mapper.use_gpu_pnp", "1"])
            .expect("COLMAP GPU PnP flag");
        let Commands::Mapper(args) = cli.command else {
            panic!("mapper command")
        };
        assert_eq!(args.use_gpu_pnp, Some(1));
    }

    #[test]
    fn colmap_mapper_parses_ba_backend_options() {
        let cli = Cli::try_parse_from([
            "rustsfm",
            "mapper",
            "--Mapper.ba_linear_solver",
            "iterative_schur",
            "--Mapper.ba_sparse_backend",
            "accelerate-sparse",
        ])
        .expect("COLMAP BA backend flags");
        let Commands::Mapper(args) = cli.command else {
            panic!("mapper command")
        };
        assert_eq!(args.ba_linear_solver.as_deref(), Some("iterative_schur"));
        assert_eq!(args.ba_sparse_backend.as_deref(), Some("accelerate-sparse"));
    }
}
