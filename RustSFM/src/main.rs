use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use image::ImageReader;
use rustsfm::colmap::{
    read_colmap_sparse_files, write_colmap_sparse_model, ColmapCamera, ColmapSparseFormat,
};
use rustsfm::database::{ColmapDatabase, ColmapDatabaseCamera, ColmapDatabaseImage};
use rustsfm::feature_matching::MatchingPairStrategy;
use rustsfm::sift::{SiftExtractionOptions, SiftMatchingOptions};
use rustsfm::{
    benchmark_sift_extraction, compare_colmap_stages, compare_database_parity,
    compare_extracted_sift_features, debug_two_view_database_pair, extract_features_to_database,
    match_features_to_database, parse_compare_stages, run_reconstruction, DebugTwoViewOptions,
    FeatureType, ImageSelectionMethod, MapperConfig, MatchFeaturesOptions,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

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
    #[arg(long, default_value = "50")]
    global_ba_iterations: usize,
    #[arg(long, default_value = "1.1")]
    global_ba_images_ratio: f32,
    #[arg(long, default_value = "1.1")]
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
    #[arg(long, default_value = "info")]
    log_level: String,
}

fn sift_matching_from_args(
    match_ratio: f64,
    sift_cpu_brute_force_matcher: bool,
) -> SiftMatchingOptions {
    SiftMatchingOptions {
        max_ratio: match_ratio as f32,
        cpu_brute_force_matcher: sift_cpu_brute_force_matcher,
        ..Default::default()
    }
}

fn sift_extraction_from_args(
    max_features: usize,
    sift_estimate_affine_shape: bool,
    sift_domain_size_pooling: bool,
    sift_force_covariant: bool,
) -> SiftExtractionOptions {
    SiftExtractionOptions {
        max_num_features: max_features,
        estimate_affine_shape: sift_estimate_affine_shape,
        domain_size_pooling: sift_domain_size_pooling,
        force_covariant_extractor: sift_force_covariant,
        ..Default::default()
    }
}

fn colmap_bool(value: i32, name: &str) -> Result<bool> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        other => bail!("{name} expects a COLMAP-style boolean value of 0 or 1, got {other}"),
    }
}

fn colmap_optional_bool(value: Option<i32>, name: &str) -> Result<Option<bool>> {
    value.map(|value| colmap_bool(value, name)).transpose()
}

fn parse_colmap_project(path: &Path) -> Result<HashMap<String, String>> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read project_path {}", path.display()))?;
    let mut values = HashMap::new();
    let mut section: Option<String> = None;
    for (line_index, raw_line) in contents.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        if let Some(section_name) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            section = Some(section_name.trim().to_string());
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            bail!(
                "invalid project_path line {} in {}: expected key=value",
                line_index + 1,
                path.display()
            );
        };
        let key = key.trim();
        if key.is_empty() {
            bail!(
                "invalid project_path line {} in {}: empty key",
                line_index + 1,
                path.display()
            );
        }
        let full_key = if let Some(section) = &section {
            format!("{section}.{}", key)
        } else {
            key.to_string()
        };
        values.insert(full_key, value.trim().to_string());
    }
    Ok(values)
}

fn project_value<'a>(project: &'a HashMap<String, String>, key: &str) -> Option<&'a str> {
    project.get(key).map(String::as_str)
}

fn parse_project_value<T>(project: &HashMap<String, String>, key: &str) -> Result<Option<T>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    project
        .get(key)
        .map(|value| {
            value.parse::<T>().map_err(|err| {
                anyhow::anyhow!("failed to parse project_path option {key}={value:?}: {err}")
            })
        })
        .transpose()
}

fn parse_project_bool(project: &HashMap<String, String>, key: &str) -> Result<Option<bool>> {
    let Some(value) = project.get(key) else {
        return Ok(None);
    };
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" => Ok(Some(true)),
        "0" | "false" => Ok(Some(false)),
        _ => bail!("failed to parse project_path option {key}={value:?} as boolean"),
    }
}

fn colmap_project_bool_to_i32(value: Option<bool>) -> Option<i32> {
    value.map(|value| if value { 1 } else { 0 })
}

fn colmap_num_threads(value: isize, name: &str) -> Result<Option<usize>> {
    if value < 0 {
        Ok(None)
    } else {
        usize::try_from(value)
            .with_context(|| format!("{name} is out of range: {value}"))
            .map(Some)
    }
}

#[derive(Debug)]
struct ResolvedColmapMapperArgs {
    database_path: PathBuf,
    image_path: PathBuf,
    output_path: PathBuf,
    ba_refine_focal_length: i32,
    ba_refine_principal_point: i32,
    ba_refine_extra_params: i32,
    multiple_models: i32,
    min_num_matches: usize,
    max_num_models: usize,
    max_model_overlap: usize,
    min_model_size: usize,
    snapshot_path: Option<PathBuf>,
    snapshot_frames_freq: usize,
    fix_existing_frames: i32,
    init_num_trials: usize,
    init_min_num_inliers: usize,
    init_max_error: f32,
    init_max_forward_motion: f32,
    init_min_tri_angle: f32,
    init_max_reg_trials: usize,
    abs_pose_max_error: f32,
    abs_pose_min_num_inliers: usize,
    abs_pose_min_inlier_ratio: f32,
    max_reg_trials: usize,
    local_ba_num_images: usize,
    global_ba_images_ratio: f32,
    global_ba_points_ratio: f32,
    global_ba_images_freq: usize,
    global_ba_points_freq: usize,
    global_ba_iterations: usize,
    local_ba_iterations: usize,
    global_ba_max_refinements: usize,
    local_ba_max_refinements: usize,
    global_ba_max_refinement_change: f32,
    local_ba_max_refinement_change: f32,
    global_ba_ignore_redundant_points3d: i32,
    global_ba_ignore_redundant_points3d_min_coverage_gain: f64,
    extract_colors: i32,
    min_focal_length_ratio: f64,
    max_focal_length_ratio: f64,
    max_extra_param: f64,
    filter_max_reproj_error: f32,
    tri_ignore_two_view_tracks: i32,
    random_seed: i32,
    num_threads: Option<usize>,
}

fn resolve_colmap_mapper_args(args: &ColmapMapperArgs) -> Result<ResolvedColmapMapperArgs> {
    let project = if let Some(path) = &args.project_path {
        parse_colmap_project(path)?
    } else {
        HashMap::new()
    };

    let database_path = args
        .database_path
        .clone()
        .or_else(|| project_value(&project, "database_path").map(PathBuf::from))
        .context("missing required --database_path (or database_path in --project_path)")?;
    let image_path = args
        .image_path
        .clone()
        .or_else(|| project_value(&project, "image_path").map(PathBuf::from))
        .context("missing required --image_path (or image_path in --project_path)")?;
    let output_path = args
        .output_path
        .clone()
        .or_else(|| project_value(&project, "output_path").map(PathBuf::from))
        .context("missing required --output_path (or output_path in --project_path)")?;

    let project_num_threads = parse_project_value::<isize>(&project, "Mapper.num_threads")?;
    let num_threads = match args.num_threads.or(project_num_threads) {
        Some(value) => colmap_num_threads(value, "Mapper.num_threads")?,
        None => None,
    };
    let project_snapshot_path = project_value(&project, "Mapper.snapshot_path")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);

    Ok(ResolvedColmapMapperArgs {
        database_path,
        image_path,
        output_path,
        ba_refine_focal_length: args
            .ba_refine_focal_length
            .or(colmap_project_bool_to_i32(parse_project_bool(
                &project,
                "Mapper.ba_refine_focal_length",
            )?))
            .unwrap_or(1),
        ba_refine_principal_point: args
            .ba_refine_principal_point
            .or(colmap_project_bool_to_i32(parse_project_bool(
                &project,
                "Mapper.ba_refine_principal_point",
            )?))
            .unwrap_or(0),
        ba_refine_extra_params: args
            .ba_refine_extra_params
            .or(colmap_project_bool_to_i32(parse_project_bool(
                &project,
                "Mapper.ba_refine_extra_params",
            )?))
            .unwrap_or(1),
        multiple_models: args
            .multiple_models
            .or(colmap_project_bool_to_i32(parse_project_bool(
                &project,
                "Mapper.multiple_models",
            )?))
            .unwrap_or(1),
        min_num_matches: args
            .min_num_matches
            .or(parse_project_value(&project, "Mapper.min_num_matches")?)
            .unwrap_or(15),
        max_num_models: args
            .max_num_models
            .or(parse_project_value(&project, "Mapper.max_num_models")?)
            .unwrap_or(50),
        max_model_overlap: args
            .max_model_overlap
            .or(parse_project_value(&project, "Mapper.max_model_overlap")?)
            .unwrap_or(20),
        min_model_size: args
            .min_model_size
            .or(parse_project_value(&project, "Mapper.min_model_size")?)
            .unwrap_or(10),
        snapshot_path: args.snapshot_path.clone().or(project_snapshot_path),
        snapshot_frames_freq: args
            .snapshot_frames_freq
            .or(parse_project_value(
                &project,
                "Mapper.snapshot_frames_freq",
            )?)
            .unwrap_or(0),
        fix_existing_frames: args
            .fix_existing_frames
            .or(colmap_project_bool_to_i32(parse_project_bool(
                &project,
                "Mapper.fix_existing_frames",
            )?))
            .unwrap_or(0),
        init_num_trials: args
            .init_num_trials
            .or(parse_project_value(&project, "Mapper.init_num_trials")?)
            .unwrap_or(200),
        init_min_num_inliers: args
            .init_min_num_inliers
            .or(parse_project_value(
                &project,
                "Mapper.init_min_num_inliers",
            )?)
            .unwrap_or(100),
        init_max_error: args
            .init_max_error
            .or(parse_project_value(&project, "Mapper.init_max_error")?)
            .unwrap_or(4.0),
        init_max_forward_motion: args
            .init_max_forward_motion
            .or(parse_project_value(
                &project,
                "Mapper.init_max_forward_motion",
            )?)
            .unwrap_or(0.95),
        init_min_tri_angle: args
            .init_min_tri_angle
            .or(parse_project_value(&project, "Mapper.init_min_tri_angle")?)
            .unwrap_or(16.0),
        init_max_reg_trials: args
            .init_max_reg_trials
            .or(parse_project_value(&project, "Mapper.init_max_reg_trials")?)
            .unwrap_or(2),
        abs_pose_max_error: args
            .abs_pose_max_error
            .or(parse_project_value(&project, "Mapper.abs_pose_max_error")?)
            .unwrap_or(12.0),
        abs_pose_min_num_inliers: args
            .abs_pose_min_num_inliers
            .or(parse_project_value(
                &project,
                "Mapper.abs_pose_min_num_inliers",
            )?)
            .unwrap_or(30),
        abs_pose_min_inlier_ratio: args
            .abs_pose_min_inlier_ratio
            .or(parse_project_value(
                &project,
                "Mapper.abs_pose_min_inlier_ratio",
            )?)
            .unwrap_or(0.25),
        max_reg_trials: args
            .max_reg_trials
            .or(parse_project_value(&project, "Mapper.max_reg_trials")?)
            .unwrap_or(3),
        local_ba_num_images: args
            .local_ba_num_images
            .or(parse_project_value(&project, "Mapper.ba_local_num_images")?)
            .unwrap_or(6),
        global_ba_images_ratio: args
            .global_ba_images_ratio
            .or(parse_project_value(
                &project,
                "Mapper.ba_global_frames_ratio",
            )?)
            .unwrap_or(1.1),
        global_ba_points_ratio: args
            .global_ba_points_ratio
            .or(parse_project_value(
                &project,
                "Mapper.ba_global_points_ratio",
            )?)
            .unwrap_or(1.1),
        global_ba_images_freq: args
            .global_ba_images_freq
            .or(parse_project_value(
                &project,
                "Mapper.ba_global_frames_freq",
            )?)
            .unwrap_or(500),
        global_ba_points_freq: args
            .global_ba_points_freq
            .or(parse_project_value(
                &project,
                "Mapper.ba_global_points_freq",
            )?)
            .unwrap_or(250_000),
        global_ba_iterations: args
            .global_ba_iterations
            .or(parse_project_value(
                &project,
                "Mapper.ba_global_max_num_iterations",
            )?)
            .unwrap_or(50),
        local_ba_iterations: args
            .local_ba_iterations
            .or(parse_project_value(
                &project,
                "Mapper.ba_local_max_num_iterations",
            )?)
            .unwrap_or(25),
        global_ba_max_refinements: args
            .global_ba_max_refinements
            .or(parse_project_value(
                &project,
                "Mapper.ba_global_max_refinements",
            )?)
            .unwrap_or(5),
        local_ba_max_refinements: args
            .local_ba_max_refinements
            .or(parse_project_value(
                &project,
                "Mapper.ba_local_max_refinements",
            )?)
            .unwrap_or(2),
        global_ba_max_refinement_change: args
            .global_ba_max_refinement_change
            .or(parse_project_value(
                &project,
                "Mapper.ba_global_max_refinement_change",
            )?)
            .unwrap_or(0.0005),
        local_ba_max_refinement_change: args
            .local_ba_max_refinement_change
            .or(parse_project_value(
                &project,
                "Mapper.ba_local_max_refinement_change",
            )?)
            .unwrap_or(0.001),
        global_ba_ignore_redundant_points3d: args
            .global_ba_ignore_redundant_points3d
            .or(colmap_project_bool_to_i32(parse_project_bool(
                &project,
                "Mapper.ba_global_ignore_redundant_points3D",
            )?))
            .unwrap_or(0),
        global_ba_ignore_redundant_points3d_min_coverage_gain: args
            .global_ba_ignore_redundant_points3d_min_coverage_gain
            .or(parse_project_value(
                &project,
                "Mapper.ba_global_ignore_redundant_points3D_min_coverage_gain",
            )?)
            .unwrap_or(0.05),
        extract_colors: args
            .extract_colors
            .or(colmap_project_bool_to_i32(parse_project_bool(
                &project,
                "Mapper.extract_colors",
            )?))
            .unwrap_or(1),
        min_focal_length_ratio: args
            .min_focal_length_ratio
            .or(parse_project_value(
                &project,
                "Mapper.min_focal_length_ratio",
            )?)
            .unwrap_or(0.1),
        max_focal_length_ratio: args
            .max_focal_length_ratio
            .or(parse_project_value(
                &project,
                "Mapper.max_focal_length_ratio",
            )?)
            .unwrap_or(10.0),
        max_extra_param: args
            .max_extra_param
            .or(parse_project_value(&project, "Mapper.max_extra_param")?)
            .unwrap_or(1.0),
        filter_max_reproj_error: args
            .filter_max_reproj_error
            .or(parse_project_value(
                &project,
                "Mapper.filter_max_reproj_error",
            )?)
            .unwrap_or(4.0),
        tri_ignore_two_view_tracks: args
            .tri_ignore_two_view_tracks
            .or(colmap_project_bool_to_i32(parse_project_bool(
                &project,
                "Mapper.tri_ignore_two_view_tracks",
            )?))
            .unwrap_or(1),
        random_seed: args
            .random_seed
            .or(parse_project_value(&project, "Mapper.random_seed")?)
            .unwrap_or(-1),
        num_threads,
    })
}

fn matching_pair_strategy_from_name(
    matching_strategy: &str,
    local_window: usize,
    sequential_overlap: usize,
    sequential_quadratic_overlap: bool,
    sequential_loop_detection: bool,
    sequential_loop_detection_period: usize,
    vocab_tree_num_images: usize,
) -> MatchingPairStrategy {
    match matching_strategy.to_ascii_lowercase().as_str() {
        "exhaustive" => MatchingPairStrategy::Exhaustive,
        "local-window" | "local_window" => MatchingPairStrategy::LocalWindow {
            window: local_window.max(1),
        },
        "vocab-tree" | "vocab_tree" | "vocabtree" => MatchingPairStrategy::VocabTree {
            num_images: vocab_tree_num_images.max(1),
        },
        _ => MatchingPairStrategy::Sequential {
            overlap: sequential_overlap.max(1),
            quadratic_overlap: sequential_quadratic_overlap,
            loop_detection: sequential_loop_detection,
            loop_detection_period: sequential_loop_detection_period.max(1),
        },
    }
}

fn run_reconstruct(args: ReconstructArgs) -> Result<()> {
    env_logger::Builder::new()
        .parse_filters(&args.log_level)
        .init();
    let matching_pair_strategy = matching_pair_strategy_from_name(
        &args.matching_strategy,
        args.local_window,
        args.sequential_overlap,
        args.sequential_quadratic_overlap,
        args.sequential_loop_detection,
        args.sequential_loop_detection_period,
        args.vocab_tree_num_images,
    );
    let config = MapperConfig {
        input: args.input,
        output: args.output,
        reference: args.reference,
        database: args.database,
        write_two_view_geometries: args.write_two_view_geometries,
        ignore_database_two_view_poses: args.ignore_database_two_view_poses,
        write_database: args.write_database,
        max_images: args.max_images,
        multiple_models: !args.single_model,
        max_num_models: args.max_num_models,
        max_model_overlap: args.max_model_overlap,
        min_model_size: args.min_model_size,
        snapshot_path: args.snapshot_path,
        snapshot_frames_freq: args.snapshot_frames_freq,
        extract_colors: !args.no_extract_colors,
        fix_existing_frames: args.fix_existing_frames,
        feature_type: args.feature_type,
        max_features: args.max_features,
        match_ratio: args.match_ratio,
        max_hamming_distance: args.max_hamming_distance,
        local_matching: args.local_matching,
        local_window: args.local_window,
        matching_pair_strategy,
        sift_matching: SiftMatchingOptions {
            guided_matching: args.guided_matching,
            max_guided_epipolar_error_px: args.guided_max_epipolar_error_px,
            cpu_brute_force_matcher: args.sift_cpu_brute_force_matcher,
            max_ratio: args.match_ratio as f32,
            ..Default::default()
        },
        sift_extraction: sift_extraction_from_args(
            args.max_features,
            args.sift_estimate_affine_shape,
            args.sift_domain_size_pooling,
            args.sift_force_covariant,
        ),
        experimental_sequence_heuristics: args.experimental_sequence_heuristics,
        experimental_ring_closure: args.experimental_ring_closure,
        experimental_structureless_pair_pose_fallback: args
            .experimental_structureless_pair_pose_fallback,
        min_matches: args.min_matches,
        min_inliers: args.min_inliers,
        min_triangulated: args.min_triangulated,
        init_min_num_inliers: args.init_min_num_inliers,
        init_min_tri_angle_deg: args.init_min_tri_angle_deg,
        init_max_forward_motion: args.init_max_forward_motion,
        init_num_trials: args.init_num_trials,
        init_max_reg_trials: args.init_max_reg_trials,
        essential_threshold_px: args.essential_threshold_px,
        essential_iterations: args.essential_iterations,
        pnp_threshold_px: args.pnp_threshold_px,
        pnp_iterations: args.pnp_iterations,
        abs_pose_min_num_inliers: args.abs_pose_min_num_inliers,
        abs_pose_min_inlier_ratio: args.abs_pose_min_inlier_ratio,
        image_selection_method: args.image_selection_method,
        random_seed: args.random_seed,
        max_reg_trials: args.max_reg_trials,
        local_ba: !args.no_local_ba,
        local_ba_num_images: args.local_ba_num_images,
        local_ba_min_shared_points: args.local_ba_min_shared_points,
        local_ba_iterations: args.local_ba_iterations,
        local_ba_max_refinements: args.local_ba_max_refinements,
        local_ba_max_refinement_change: args.local_ba_max_refinement_change,
        global_ba: !args.no_global_ba,
        global_ba_iterations: args.global_ba_iterations,
        global_ba_images_ratio: args.global_ba_images_ratio,
        global_ba_points_ratio: args.global_ba_points_ratio,
        global_ba_images_freq: args.global_ba_images_freq,
        global_ba_points_freq: args.global_ba_points_freq,
        global_ba_max_refinements: args.global_ba_max_refinements,
        global_ba_max_refinement_change: args.global_ba_max_refinement_change,
        global_ba_ignore_redundant_points3d: args.global_ba_ignore_redundant_points3d,
        global_ba_ignore_redundant_points3d_min_coverage_gain: args
            .global_ba_ignore_redundant_points3d_min_coverage_gain,
        ba_refine_focal_length: !args.no_ba_refine_focal_length,
        ba_refine_principal_point: args.ba_refine_principal_point,
        ba_refine_extra_params: !args.no_ba_refine_extra_params,
        ba_constant_rig_ids: args.ba_constant_rig_ids,
        ba_constant_camera_ids: args.ba_constant_camera_ids,
        min_focal_length_ratio: args.min_focal_length_ratio,
        max_focal_length_ratio: args.max_focal_length_ratio,
        max_extra_param: args.max_extra_param,
        max_reprojection_error_px: args.max_reprojection_error_px,
        ignore_two_view_tracks: args.ignore_two_view_tracks,
        global_mapper: args.global_mapper,
        pose_graph: args.pose_graph,
        copy_images: !args.no_copy_images,
        threads: args.threads,
        fx: args.fx,
        fy: args.fy,
        cx: args.cx,
        cy: args.cy,
        ..Default::default()
    };
    let summary = run_reconstruction(&config)?;
    println!(
        "RustSFM completed: images={} registered={} points={} pairs={} elapsed_ms={:.2}",
        summary.images,
        summary.registered_images,
        summary.points,
        summary.pairs,
        summary.elapsed_ms
    );
    if let Some(path) = args.summary_json {
        std::fs::write(path, serde_json::to_string_pretty(&summary)?)?;
    }
    Ok(())
}

fn run_match_features(args: MatchFeaturesArgs) -> Result<()> {
    env_logger::Builder::new()
        .parse_filters(&args.log_level)
        .init();
    let report = match_features_to_database(
        &args.database,
        &MatchFeaturesOptions {
            pair_strategy: matching_pair_strategy_from_name(
                &args.matching_strategy,
                3,
                args.sequential_overlap,
                args.sequential_quadratic_overlap,
                args.sequential_loop_detection,
                args.sequential_loop_detection_period,
                args.vocab_tree_num_images,
            ),
            sift_matching: sift_matching_from_args(
                args.match_ratio,
                args.sift_cpu_brute_force_matcher,
            ),
            min_num_matches: args.min_num_matches,
            min_inliers: args.min_num_matches,
            essential_threshold_px: args.essential_threshold_px,
            essential_iterations: args.essential_iterations,
            clear_existing: args.clear_existing,
            use_existing_matches: args.use_existing_matches,
            existing_match_batch_size: args.existing_match_batch_size,
            random_seed: args.random_seed,
            ..MatchFeaturesOptions::default()
        },
    )?;
    println!(
        "match-features: pairs={} matched={} verified={} total_matches={} seconds={:.3}",
        report.pair_count,
        report.matched_pairs,
        report.verified_pairs,
        report.total_matches,
        report.matching_seconds
    );
    if let Some(path) = args.output_json {
        std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }
    Ok(())
}

fn run_colmap_feature_extractor(args: ColmapFeatureExtractorArgs) -> Result<()> {
    env_logger::Builder::new()
        .parse_filters(&args.log_level)
        .init();
    let use_gpu = colmap_optional_bool(args.use_gpu, "SiftExtraction.use_gpu")?;
    if use_gpu == Some(true) {
        bail!("SiftExtraction.use_gpu is not implemented in RustSFM; use CPU extraction");
    }
    let single_camera = colmap_bool(args.single_camera, "ImageReader.single_camera")?;
    let estimate_affine_shape = colmap_bool(
        args.estimate_affine_shape,
        "SiftExtraction.estimate_affine_shape",
    )?;
    let domain_size_pooling = colmap_bool(
        args.domain_size_pooling,
        "SiftExtraction.domain_size_pooling",
    )?;
    ensure_feature_extractor_database(
        &args.database_path,
        &args.image_path,
        &args.camera_model,
        single_camera,
    )?;
    let options = sift_extraction_from_args(
        args.max_num_features,
        estimate_affine_shape,
        domain_size_pooling,
        estimate_affine_shape || domain_size_pooling,
    );
    let report = extract_features_to_database(&args.database_path, &args.image_path, &options)?;
    println!(
        "feature_extractor: database={} images={} total_keypoints={} seconds={:.3}",
        args.database_path.display(),
        report.image_count,
        report.total_keypoints,
        report.extraction_seconds
    );
    if let Some(path) = args.output_json {
        std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }
    Ok(())
}

fn run_colmap_matcher(
    database_path: PathBuf,
    strategy: MatchingPairStrategy,
    max_ratio: f64,
    use_gpu: Option<i32>,
    guided_matching: i32,
    min_num_inliers: usize,
    max_error: f32,
    max_num_trials: u32,
    random_seed: i32,
    output_json: Option<PathBuf>,
    log_level: String,
) -> Result<()> {
    let use_gpu = colmap_optional_bool(use_gpu, "SiftMatching.use_gpu")?;
    if use_gpu == Some(true) {
        bail!("SiftMatching.use_gpu is not implemented in RustSFM; use CPU matching");
    }
    let guided_matching = colmap_bool(guided_matching, "SiftMatching.guided_matching")?;
    let args = MatchFeaturesArgs {
        database: database_path,
        matching_strategy: "sequential".to_string(),
        sequential_overlap: 10,
        sequential_quadratic_overlap: true,
        sequential_loop_detection: false,
        sequential_loop_detection_period: 10,
        vocab_tree_num_images: 100,
        match_ratio: max_ratio,
        sift_cpu_brute_force_matcher: false,
        min_num_matches: min_num_inliers,
        essential_threshold_px: max_error,
        essential_iterations: max_num_trials,
        clear_existing: true,
        use_existing_matches: false,
        existing_match_batch_size: 1000,
        random_seed,
        output_json,
        log_level,
    };
    let mut options = MatchFeaturesOptions {
        pair_strategy: strategy,
        sift_matching: SiftMatchingOptions {
            guided_matching,
            max_ratio: max_ratio as f32,
            ..Default::default()
        },
        min_num_matches: min_num_inliers,
        min_inliers: min_num_inliers,
        essential_threshold_px: max_error,
        essential_iterations: max_num_trials,
        random_seed,
        ..MatchFeaturesOptions::default()
    };
    options.clear_existing = args.clear_existing;
    let report = match_features_to_database(&args.database, &options)?;
    println!(
        "matcher: pairs={} matched={} verified={} total_matches={} seconds={:.3}",
        report.pair_count,
        report.matched_pairs,
        report.verified_pairs,
        report.total_matches,
        report.matching_seconds
    );
    if let Some(path) = args.output_json {
        std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }
    Ok(())
}

fn run_colmap_geometric_verifier(args: ColmapGeometricVerifierArgs) -> Result<()> {
    env_logger::Builder::new()
        .parse_filters(&args.log_level)
        .init();
    let guided_matching = colmap_bool(args.guided_matching, "SiftMatching.guided_matching")?;
    let report = match_features_to_database(
        &args.database_path,
        &MatchFeaturesOptions {
            sift_matching: SiftMatchingOptions {
                guided_matching,
                ..Default::default()
            },
            min_num_matches: args.min_num_inliers,
            min_inliers: args.min_num_inliers,
            essential_threshold_px: args.max_error,
            essential_iterations: args.max_num_trials,
            clear_existing: false,
            use_existing_matches: true,
            random_seed: args.random_seed,
            ..MatchFeaturesOptions::default()
        },
    )?;
    println!(
        "geometric_verifier: pairs={} verified={} total_matches={} seconds={:.3}",
        report.pair_count, report.verified_pairs, report.total_matches, report.matching_seconds
    );
    if let Some(path) = args.output_json {
        std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }
    Ok(())
}

fn run_colmap_mapper(args: ColmapMapperArgs) -> Result<()> {
    let resolved = resolve_colmap_mapper_args(&args)?;
    let ba_refine_focal_length = colmap_bool(
        resolved.ba_refine_focal_length,
        "Mapper.ba_refine_focal_length",
    )?;
    let ba_refine_principal_point = colmap_bool(
        resolved.ba_refine_principal_point,
        "Mapper.ba_refine_principal_point",
    )?;
    let ba_refine_extra_params = colmap_bool(
        resolved.ba_refine_extra_params,
        "Mapper.ba_refine_extra_params",
    )?;
    let multiple_models = colmap_bool(resolved.multiple_models, "Mapper.multiple_models")?;
    let extract_colors = colmap_bool(resolved.extract_colors, "Mapper.extract_colors")?;
    let fix_existing_frames =
        colmap_bool(resolved.fix_existing_frames, "Mapper.fix_existing_frames")?;
    let global_ba_ignore_redundant_points3d = colmap_bool(
        resolved.global_ba_ignore_redundant_points3d,
        "Mapper.ba_global_ignore_redundant_points3D",
    )?;
    let ignore_two_view_tracks = colmap_bool(
        resolved.tri_ignore_two_view_tracks,
        "Mapper.tri_ignore_two_view_tracks",
    )?;
    let reconstruct_args = ReconstructArgs {
        input: resolved.image_path,
        output: resolved.output_path,
        reference: None,
        database: Some(resolved.database_path),
        write_two_view_geometries: false,
        ignore_database_two_view_poses: false,
        write_database: false,
        max_images: None,
        single_model: !multiple_models,
        max_num_models: resolved.max_num_models,
        max_model_overlap: resolved.max_model_overlap,
        min_model_size: resolved.min_model_size,
        snapshot_path: resolved.snapshot_path,
        snapshot_frames_freq: resolved.snapshot_frames_freq,
        no_extract_colors: !extract_colors,
        fix_existing_frames,
        feature_type: FeatureType::Sift,
        max_features: 8192,
        sift_estimate_affine_shape: false,
        sift_domain_size_pooling: false,
        sift_force_covariant: false,
        sift_cpu_brute_force_matcher: false,
        local_matching: false,
        local_window: 3,
        matching_strategy: "sequential".to_string(),
        sequential_overlap: 10,
        sequential_quadratic_overlap: true,
        sequential_loop_detection: false,
        sequential_loop_detection_period: 10,
        vocab_tree_num_images: 100,
        guided_matching: false,
        guided_max_epipolar_error_px: 2.0,
        experimental_sequence_heuristics: false,
        experimental_ring_closure: false,
        experimental_structureless_pair_pose_fallback: false,
        match_ratio: 0.8,
        max_hamming_distance: 160.0,
        min_matches: resolved.min_num_matches,
        min_inliers: resolved.min_num_matches,
        min_triangulated: 4,
        init_min_num_inliers: resolved.init_min_num_inliers,
        init_min_tri_angle_deg: resolved.init_min_tri_angle,
        init_max_forward_motion: resolved.init_max_forward_motion,
        init_num_trials: resolved.init_num_trials,
        init_max_reg_trials: resolved.init_max_reg_trials,
        essential_threshold_px: resolved.init_max_error,
        essential_iterations: 10_000,
        pnp_threshold_px: resolved.abs_pose_max_error,
        pnp_iterations: 10_000,
        abs_pose_min_num_inliers: resolved.abs_pose_min_num_inliers,
        abs_pose_min_inlier_ratio: resolved.abs_pose_min_inlier_ratio,
        image_selection_method: ImageSelectionMethod::MinUncertainty,
        random_seed: resolved.random_seed,
        max_reg_trials: resolved.max_reg_trials,
        no_local_ba: false,
        local_ba_num_images: resolved.local_ba_num_images,
        local_ba_min_shared_points: 15,
        local_ba_iterations: resolved.local_ba_iterations,
        local_ba_max_refinements: resolved.local_ba_max_refinements,
        local_ba_max_refinement_change: resolved.local_ba_max_refinement_change,
        no_global_ba: false,
        global_ba_iterations: resolved.global_ba_iterations,
        global_ba_images_ratio: resolved.global_ba_images_ratio,
        global_ba_points_ratio: resolved.global_ba_points_ratio,
        global_ba_images_freq: resolved.global_ba_images_freq,
        global_ba_points_freq: resolved.global_ba_points_freq,
        global_ba_max_refinements: resolved.global_ba_max_refinements,
        global_ba_max_refinement_change: resolved.global_ba_max_refinement_change,
        global_ba_ignore_redundant_points3d,
        global_ba_ignore_redundant_points3d_min_coverage_gain: resolved
            .global_ba_ignore_redundant_points3d_min_coverage_gain,
        no_ba_refine_focal_length: !ba_refine_focal_length,
        ba_refine_principal_point: ba_refine_principal_point,
        no_ba_refine_extra_params: !ba_refine_extra_params,
        ba_constant_rig_ids: Vec::new(),
        ba_constant_camera_ids: Vec::new(),
        min_focal_length_ratio: resolved.min_focal_length_ratio,
        max_focal_length_ratio: resolved.max_focal_length_ratio,
        max_extra_param: resolved.max_extra_param,
        max_reprojection_error_px: resolved.filter_max_reproj_error,
        ignore_two_view_tracks,
        pose_graph: false,
        global_mapper: false,
        fx: None,
        fy: None,
        cx: None,
        cy: None,
        threads: resolved.num_threads,
        no_copy_images: false,
        summary_json: args.summary_json,
        log_level: args.log_level,
    };
    run_reconstruct(reconstruct_args)
}

fn run_model_converter(args: ModelConverterArgs) -> Result<()> {
    let input_format = parse_sparse_format(args.input_type.as_deref())?;
    let output_format = parse_sparse_format(Some(&args.output_type))?;
    let sparse = if let Some(format) = input_format {
        rustsfm::colmap::read_colmap_sparse_files_with_format(&args.input_path, format)?
    } else {
        read_colmap_sparse_files(&args.input_path)?
    };
    write_colmap_sparse_model(
        &args.output_path,
        &sparse,
        output_format.with_context(|| {
            format!(
                "unsupported model_converter output_type '{}'; supported: TXT, BIN",
                args.output_type
            )
        })?,
    )?;
    println!(
        "model_converter: cameras={} images={} points3D={} output={}",
        sparse.cameras.len(),
        sparse.images.len(),
        sparse.points3d.len(),
        args.output_path.display()
    );
    Ok(())
}

fn parse_sparse_format(value: Option<&str>) -> Result<Option<ColmapSparseFormat>> {
    let Some(value) = value else {
        return Ok(None);
    };
    match value.to_ascii_uppercase().as_str() {
        "TXT" | "TEXT" => Ok(Some(ColmapSparseFormat::Text)),
        "BIN" | "BINARY" => Ok(Some(ColmapSparseFormat::Binary)),
        other => bail!("unsupported sparse model format '{other}'; supported: TXT, BIN"),
    }
}

fn ensure_feature_extractor_database(
    database_path: &Path,
    image_path: &Path,
    camera_model: &str,
    single_camera: bool,
) -> Result<()> {
    if !matches!(camera_model, "PINHOLE" | "SIMPLE_PINHOLE") {
        bail!("RustSFM feature_extractor currently supports PINHOLE/SIMPLE_PINHOLE only");
    }
    let db = ColmapDatabase::open(database_path)?;
    if !db.read_all_images()?.is_empty() {
        return Ok(());
    }
    let mut images = std::fs::read_dir(image_path)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| {
                    matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png")
                })
        })
        .collect::<Vec<_>>();
    images.sort();
    if images.is_empty() {
        bail!("no images found under {}", image_path.display());
    }

    let mut shared_camera_id = None;
    for (index, path) in images.iter().enumerate() {
        let (width, height) = image_dimensions(path)?;
        let camera_id = if single_camera {
            if let Some(camera_id) = shared_camera_id {
                camera_id
            } else {
                let id = write_database_camera(&db, 0, width, height, camera_model)?;
                shared_camera_id = Some(id);
                id
            }
        } else {
            write_database_camera(&db, 0, width, height, camera_model)?
        };
        let name = path
            .file_name()
            .context("image path has no file name")?
            .to_string_lossy()
            .into_owned();
        db.write_image(
            &ColmapDatabaseImage {
                image_id: (index + 1) as u32,
                name,
                camera_id,
                frame_id: None,
            },
            true,
        )?;
    }
    Ok(())
}

fn image_dimensions(path: &Path) -> Result<(u32, u32)> {
    let image = ImageReader::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?
        .with_guessed_format()
        .with_context(|| format!("failed to guess image format for {}", path.display()))?
        .decode()
        .with_context(|| format!("failed to decode {}", path.display()))?;
    Ok((image.width(), image.height()))
}

fn write_database_camera(
    db: &ColmapDatabase,
    camera_id: u32,
    width: u32,
    height: u32,
    camera_model: &str,
) -> Result<u32> {
    let focal = width.max(height) as f64 * 1.2;
    let (model_id, params) = if camera_model == "SIMPLE_PINHOLE" {
        (
            rustsfm::types::COLMAP_SIMPLE_PINHOLE,
            vec![focal, width as f64 * 0.5, height as f64 * 0.5],
        )
    } else {
        (
            rustsfm::types::COLMAP_PINHOLE,
            vec![focal, focal, width as f64 * 0.5, height as f64 * 0.5],
        )
    };
    db.write_camera(
        &ColmapDatabaseCamera {
            camera: ColmapCamera {
                camera_id,
                model_id,
                width,
                height,
                params,
            },
            has_prior_focal_length: false,
        },
        camera_id != 0,
    )
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Reconstruct(args) => run_reconstruct(args)?,
        Commands::Compare(args) => {
            env_logger::Builder::new()
                .parse_filters(&args.log_level)
                .init();
            let stages = parse_compare_stages(&args.stages)?;
            let report = compare_colmap_stages(
                &args.reference,
                &args.candidate,
                args.reference_database.as_deref(),
                args.candidate_database.as_deref(),
                &stages,
            )?;
            if let Some(features) = &report.features {
                println!(
                    "features: common_images={} ref_kpts={} cand_kpts={} max_diff={} pct_exact={:.3}",
                    features.common_images,
                    features.reference_total_keypoints,
                    features.candidate_total_keypoints,
                    features.keypoint_count_diff.max_abs_diff,
                    features.keypoint_count_diff.pct_exact
                );
            }
            if let Some(matches) = &report.matches {
                println!(
                    "matches: common_pairs={} ref_matches={} cand_matches={} max_diff={} pct_exact={:.3}",
                    matches.common_pairs,
                    matches.reference_total_matches,
                    matches.candidate_total_matches,
                    matches.common_pair_match_count_diff.max_abs_diff,
                    matches.common_pair_match_count_diff.pct_exact
                );
            }
            if let Some(twoview) = &report.twoview {
                println!(
                    "twoview: common_pairs={} config_agreement={:.3} max_inlier_diff={} inlier_overlap_mean={:.3} mismatches={}",
                    twoview.common_pairs,
                    twoview.config_agreement_rate,
                    twoview.inlier_count_diff.max_abs_diff,
                    twoview.inlier_set_overlap.mean_rate,
                    twoview.config_mismatches.len()
                );
            }
            if let Some(registration) = &report.registration {
                println!(
                    "registration: ref={} cand={} common={} ref_only={} cand_only={}",
                    registration.reference_registered,
                    registration.candidate_registered,
                    registration.common_registered,
                    registration.reference_only.len(),
                    registration.candidate_only.len()
                );
            }
            if let Some(tracks) = &report.tracks {
                println!(
                    "tracks: ref_points={} cand_points={} ref_len2={} cand_len2={} ref_len3_4={} cand_len3_4={}",
                    tracks.reference_points,
                    tracks.candidate_points,
                    tracks.reference_track_length_histogram.length_2,
                    tracks.candidate_track_length_histogram.length_2,
                    tracks.reference_track_length_histogram.length_3_4,
                    tracks.candidate_track_length_histogram.length_3_4
                );
            }
            if let Some(ba) = &report.ba {
                println!(
                    "ba: common={} translation_rmse={:.6} rotation_mean_deg={:.6} rotation_rmse_deg={:.6} rotation_max_deg={:.6} rel_rot_mean_deg={:.6} scale={:.6}",
                    ba.common_images,
                    ba.translation_error.rmse,
                    ba.rotation_error_deg.mean,
                    ba.rotation_error_deg.rmse,
                    ba.rotation_error_deg.max,
                    ba.adjacent_relative_rotation_error_deg.mean,
                    ba.similarity_scale
                );
            }
            if let Some(path) = args.output_json {
                std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
            }
        }
        Commands::Parity(args) => {
            env_logger::Builder::new()
                .parse_filters(&args.log_level)
                .init();
            let image_names = parity_image_names(args.images.as_ref(), &args.image_names)?;
            let report = compare_database_parity(
                &args.database,
                image_names,
                args.min_matches,
                args.ignore_watermarks,
                args.load_all_images,
                args.convert_pose_priors_to_enu,
            )?;
            println!(
                "COLMAP database parity: raw_images={} cache_images={} raw_frames={} cache_frames={} raw_frame_data={} cache_frame_data={} raw_pairs={} cache_pairs={} bridge_pairs={} bridge_matches={} differences={}",
                report.raw.images,
                report.cache.images,
                report.raw.frames,
                report.cache.frames,
                report.raw.frame_data,
                report.cache.frame_data,
                report.raw.two_view_pairs,
                report.cache.two_view_pairs,
                report.bridge.frame_pairs,
                report.bridge.matches,
                report.differences.len()
            );
            if !report.differences.is_empty() {
                for diff in &report.differences {
                    println!("difference {}: {}", diff.kind, diff.detail);
                }
            }
            if let Some(path) = args.output_json {
                std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
            }
        }
        Commands::BenchmarkSift(args) => {
            env_logger::Builder::new()
                .parse_filters(&args.log_level)
                .init();
            let options = sift_extraction_from_args(
                args.max_features,
                args.sift_estimate_affine_shape,
                args.sift_domain_size_pooling,
                args.sift_force_covariant,
            );
            let report = benchmark_sift_extraction(&args.input, &options)?;
            println!(
                "SIFT benchmark: backend={} images={} total_features={} mean_features={:.1} seconds={:.3} covariant={}",
                report.backend,
                report.image_count,
                report.total_features,
                report.mean_features,
                report.extraction_seconds,
                report.uses_covariant_extractor
            );
            if let Some(path) = args.output_json {
                std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
            }
        }
        Commands::CompareExtract(args) => {
            env_logger::Builder::new()
                .parse_filters(&args.log_level)
                .init();
            let options = sift_extraction_from_args(
                args.max_features,
                args.sift_estimate_affine_shape,
                args.sift_domain_size_pooling,
                args.sift_force_covariant,
            );
            let report =
                compare_extracted_sift_features(&args.reference_database, &args.images, &options)?;
            println!(
                "SIFT extract compare: common_images={} ref_kpts={} cand_kpts={} max_diff={} pct_exact={:.3} mean_abs_diff={:.1}",
                report.common_images,
                report.reference_total_keypoints,
                report.candidate_total_keypoints,
                report.keypoint_count_diff.max_abs_diff,
                report.keypoint_count_diff.pct_exact,
                report.keypoint_count_diff.mean_abs_diff
            );
            if let Some(path) = args.output_json {
                std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
            }
        }
        Commands::ExtractFeatures(args) => {
            env_logger::Builder::new()
                .parse_filters(&args.log_level)
                .init();
            let options = sift_extraction_from_args(
                args.max_features,
                args.sift_estimate_affine_shape,
                args.sift_domain_size_pooling,
                args.sift_force_covariant,
            );
            let report = extract_features_to_database(&args.database, &args.images, &options)?;
            println!(
                "extract-features: backend={} images={} total_keypoints={} mean_keypoints={:.1} seconds={:.3}",
                report.backend,
                report.image_count,
                report.total_keypoints,
                report.mean_keypoints,
                report.extraction_seconds
            );
            if let Some(path) = args.output_json {
                std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
            }
        }
        Commands::MatchFeatures(args) => run_match_features(args)?,
        Commands::DebugTwoview(args) => {
            env_logger::Builder::new()
                .parse_filters(&args.log_level)
                .init();
            let report = debug_two_view_database_pair(
                &args.database,
                &args.image1,
                &args.image2,
                &DebugTwoViewOptions {
                    essential_threshold_px: args.essential_threshold_px,
                    essential_iterations: args.essential_iterations,
                    min_inliers: args.min_inliers,
                    min_triangulated: args.min_triangulated,
                    random_seed: args.random_seed,
                },
            )?;
            if let Some(diagnostics) = &report.diagnostics {
                let f_inliers = diagnostics
                    .fundamental
                    .as_ref()
                    .map(|support| support.inliers)
                    .unwrap_or(0);
                let h_inliers = diagnostics
                    .homography
                    .as_ref()
                    .map(|support| support.inliers)
                    .unwrap_or(0);
                let ef_overlap = diagnostics
                    .e_f_mask_overlap
                    .as_ref()
                    .map(format_mask_overlap)
                    .unwrap_or_else(|| "n/a".to_string());
                let eh_overlap = diagnostics
                    .e_h_mask_overlap
                    .as_ref()
                    .map(format_mask_overlap)
                    .unwrap_or_else(|| "n/a".to_string());
                let fh_overlap = diagnostics
                    .f_h_mask_overlap
                    .as_ref()
                    .map(format_mask_overlap)
                    .unwrap_or_else(|| "n/a".to_string());
                println!(
                    "debug-twoview: pair={} / {} indices=({}, {}) seed={} matches={} active={} E={} F={} H={} ratios(E/F={:.3},H/F={:.3},H/E={:.3}) overlaps(EF={},EH={},FH={}) selected={:?} config={:?} inliers={} estimate_config={:?} estimate_inliers={:?} estimate_triangulated={:?} estimate_tri_angle={:?} estimate_rot={:?} estimate_error={:?}",
                    report.left_image,
                    report.right_image,
                    report.left_index,
                    report.right_index,
                    report.sampler_seed,
                    report.num_matches,
                    diagnostics.active_observations,
                    diagnostics.essential.inliers,
                    f_inliers,
                    h_inliers,
                    diagnostics.e_f_inlier_ratio,
                    diagnostics.h_f_inlier_ratio,
                    diagnostics.h_e_inlier_ratio,
                    ef_overlap,
                    eh_overlap,
                    fh_overlap,
                    diagnostics.selected_source,
                    diagnostics.classified_config,
                    diagnostics.selected_inliers,
                    report.estimate_config,
                    report.estimate_inliers,
                    report.estimate_triangulated,
                    report.estimate_median_triangulation_angle_deg,
                    report.estimate_rotation_deg,
                    report.estimate_mean_reprojection_error_px
                );
            } else {
                println!(
                    "debug-twoview: pair={} / {} matches={} no estimate",
                    report.left_image, report.right_image, report.num_matches
                );
            }
            if let Some(stored_models) = &report.stored_models {
                let stored_e = stored_models
                    .essential
                    .as_ref()
                    .map(|support| support.inliers)
                    .unwrap_or(0);
                let stored_f = stored_models
                    .fundamental
                    .as_ref()
                    .map(|support| support.inliers)
                    .unwrap_or(0);
                let stored_h = stored_models
                    .homography
                    .as_ref()
                    .map(|support| support.inliers)
                    .unwrap_or(0);
                println!(
                    "debug-twoview stored-models: E={} F={} H={}",
                    stored_e, stored_f, stored_h
                );
            }
            if let Some(path) = args.output_json {
                std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
            }
        }
        Commands::FeatureExtractor(args) => run_colmap_feature_extractor(args)?,
        Commands::ExhaustiveMatcher(args) => run_colmap_matcher(
            args.database_path,
            MatchingPairStrategy::Exhaustive,
            args.max_ratio,
            args.use_gpu,
            args.guided_matching,
            args.min_num_inliers,
            args.max_error,
            args.max_num_trials,
            args.random_seed,
            args.output_json,
            args.log_level,
        )?,
        Commands::SequentialMatcher(args) => {
            let quadratic_overlap = colmap_bool(
                args.quadratic_overlap,
                "SequentialMatching.quadratic_overlap",
            )?;
            let loop_detection =
                colmap_bool(args.loop_detection, "SequentialMatching.loop_detection")?;
            run_colmap_matcher(
                args.database_path,
                MatchingPairStrategy::Sequential {
                    overlap: args.overlap.max(1),
                    quadratic_overlap,
                    loop_detection,
                    loop_detection_period: args.loop_detection_period.max(1),
                },
                args.max_ratio,
                args.use_gpu,
                args.guided_matching,
                args.min_num_inliers,
                args.max_error,
                args.max_num_trials,
                args.random_seed,
                args.output_json,
                args.log_level,
            )?
        }
        Commands::VocabTreeMatcher(args) => run_colmap_matcher(
            args.database_path,
            MatchingPairStrategy::VocabTree {
                num_images: args.num_images.max(1),
            },
            args.max_ratio,
            args.use_gpu,
            args.guided_matching,
            args.min_num_inliers,
            args.max_error,
            args.max_num_trials,
            args.random_seed,
            args.output_json,
            args.log_level,
        )?,
        Commands::GeometricVerifier(args) => run_colmap_geometric_verifier(args)?,
        Commands::Mapper(args) => run_colmap_mapper(args)?,
        Commands::ModelConverter(args) => run_model_converter(args)?,
    }
    Ok(())
}

fn format_mask_overlap(overlap: &rustsfm::two_view::TwoViewMaskOverlapDiagnostics) -> String {
    format!(
        "{}/{}@{:.3}",
        overlap.intersection, overlap.union, overlap.jaccard
    )
}

fn parity_image_names(images: Option<&PathBuf>, explicit_names: &[String]) -> Result<Vec<String>> {
    let mut names = explicit_names.to_vec();
    if let Some(root) = images {
        let mut from_dir = std::fs::read_dir(root)?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"))
                    .unwrap_or(false)
            })
            .filter_map(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.to_string())
            })
            .collect::<Vec<_>>();
        from_dir.sort();
        names.extend(from_dir);
    }
    names.sort();
    names.dedup();
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_mapper_args(project_path: PathBuf) -> ColmapMapperArgs {
        ColmapMapperArgs {
            project_path: Some(project_path),
            database_path: None,
            image_path: None,
            output_path: None,
            ba_refine_focal_length: None,
            ba_refine_principal_point: None,
            ba_refine_extra_params: None,
            multiple_models: None,
            min_num_matches: None,
            max_num_models: None,
            max_model_overlap: None,
            min_model_size: None,
            snapshot_path: None,
            snapshot_frames_freq: None,
            fix_existing_frames: None,
            init_num_trials: None,
            init_min_num_inliers: None,
            init_max_error: None,
            init_max_forward_motion: None,
            init_min_tri_angle: None,
            init_max_reg_trials: None,
            abs_pose_max_error: None,
            abs_pose_min_num_inliers: None,
            abs_pose_min_inlier_ratio: None,
            max_reg_trials: None,
            local_ba_num_images: None,
            global_ba_images_ratio: None,
            global_ba_points_ratio: None,
            global_ba_images_freq: None,
            global_ba_points_freq: None,
            global_ba_iterations: None,
            local_ba_iterations: None,
            global_ba_max_refinements: None,
            local_ba_max_refinements: None,
            global_ba_max_refinement_change: None,
            local_ba_max_refinement_change: None,
            global_ba_ignore_redundant_points3d: None,
            global_ba_ignore_redundant_points3d_min_coverage_gain: None,
            extract_colors: None,
            min_focal_length_ratio: None,
            max_focal_length_ratio: None,
            max_extra_param: None,
            filter_max_reproj_error: None,
            tri_ignore_two_view_tracks: None,
            random_seed: None,
            num_threads: None,
            summary_json: None,
            log_level: "info".to_string(),
        }
    }

    #[test]
    fn colmap_mapper_project_path_overrides_defaults() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let project_path = dir.path().join("project.ini");
        std::fs::write(
            &project_path,
            "\
database_path=/tmp/project.db
image_path=/tmp/images
output_path=/tmp/sparse
[Mapper]
ba_refine_focal_length=false
ba_refine_extra_params=false
multiple_models=false
extract_colors=false
filter_max_reproj_error=4
tri_ignore_two_view_tracks=true
num_threads=-1
",
        )?;

        let resolved = resolve_colmap_mapper_args(&base_mapper_args(project_path))?;

        assert_eq!(resolved.database_path, PathBuf::from("/tmp/project.db"));
        assert_eq!(resolved.image_path, PathBuf::from("/tmp/images"));
        assert_eq!(resolved.output_path, PathBuf::from("/tmp/sparse"));
        assert_eq!(resolved.ba_refine_focal_length, 0);
        assert_eq!(resolved.ba_refine_extra_params, 0);
        assert_eq!(resolved.multiple_models, 0);
        assert_eq!(resolved.extract_colors, 0);
        assert_eq!(resolved.filter_max_reproj_error, 4.0);
        assert_eq!(resolved.tri_ignore_two_view_tracks, 1);
        assert_eq!(resolved.num_threads, None);
        Ok(())
    }

    #[test]
    fn colmap_mapper_cli_values_override_project_path() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let project_path = dir.path().join("project.ini");
        std::fs::write(
            &project_path,
            "\
database_path=/tmp/project.db
image_path=/tmp/images
output_path=/tmp/sparse
[Mapper]
ba_refine_focal_length=false
extract_colors=false
num_threads=-1
",
        )?;
        let mut args = base_mapper_args(project_path);
        args.database_path = Some(PathBuf::from("/tmp/cli.db"));
        args.ba_refine_focal_length = Some(1);
        args.extract_colors = Some(1);
        args.num_threads = Some(4);

        let resolved = resolve_colmap_mapper_args(&args)?;

        assert_eq!(resolved.database_path, PathBuf::from("/tmp/cli.db"));
        assert_eq!(resolved.ba_refine_focal_length, 1);
        assert_eq!(resolved.extract_colors, 1);
        assert_eq!(resolved.num_threads, Some(4));
        Ok(())
    }
}
