use crate::database::ColmapPosePrior;
use crate::feature_matching::MatchingPairStrategy;
use crate::sift::{SiftExtractionOptions, SiftMatchingOptions};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureType {
    Orb,
    Sift,
}

impl std::str::FromStr for FeatureType {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "orb" => Ok(Self::Orb),
            "sift" => Ok(Self::Sift),
            _ => bail!("unsupported feature type '{value}', expected orb or sift"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageSelectionMethod {
    MaxVisiblePointsNum,
    MaxVisiblePointsRatio,
    MinUncertainty,
}

impl std::str::FromStr for ImageSelectionMethod {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().replace('-', "_").as_str() {
            "max_visible_points_num" | "max_visible_points" | "num_visible_points" => {
                Ok(Self::MaxVisiblePointsNum)
            }
            "max_visible_points_ratio" | "visible_points_ratio" => {
                Ok(Self::MaxVisiblePointsRatio)
            }
            "min_uncertainty" | "uncertainty" => Ok(Self::MinUncertainty),
            _ => bail!(
                "unsupported image selection method '{value}', expected max_visible_points_num, max_visible_points_ratio, or min_uncertainty"
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MapperConfig {
    pub input: PathBuf,
    pub output: PathBuf,
    pub reference: Option<PathBuf>,
    pub database: Option<PathBuf>,
    pub write_two_view_geometries: bool,
    pub ignore_database_two_view_poses: bool,
    pub write_database: bool,
    pub max_images: Option<usize>,
    pub multiple_models: bool,
    pub max_num_models: usize,
    pub max_model_overlap: usize,
    pub min_model_size: usize,
    pub snapshot_path: Option<PathBuf>,
    pub snapshot_frames_freq: usize,
    pub extract_colors: bool,
    pub fix_existing_frames: bool,
    pub feature_type: FeatureType,
    pub max_features: usize,
    pub match_ratio: f64,
    pub sift_extraction: SiftExtractionOptions,
    pub sift_matching: SiftMatchingOptions,
    pub max_hamming_distance: f32,
    pub local_matching: bool,
    pub local_window: usize,
    pub matching_pair_strategy: MatchingPairStrategy,
    pub experimental_sequence_heuristics: bool,
    pub experimental_ring_closure: bool,
    pub experimental_structureless_pair_pose_fallback: bool,
    pub min_matches: usize,
    pub min_inliers: usize,
    pub min_triangulated: usize,
    pub init_min_num_inliers: usize,
    pub init_min_tri_angle_deg: f32,
    pub init_max_forward_motion: f32,
    pub init_num_trials: usize,
    pub init_max_reg_trials: usize,
    pub essential_threshold_px: f32,
    pub essential_iterations: u32,
    pub pnp_threshold_px: f32,
    pub pnp_iterations: u32,
    pub use_gpu_pnp: bool,
    pub abs_pose_min_num_inliers: usize,
    pub abs_pose_min_inlier_ratio: f32,
    pub image_selection_method: ImageSelectionMethod,
    pub pose_priors: Vec<ColmapPosePrior>,
    pub random_seed: i32,
    pub max_reg_trials: usize,
    pub local_ba: bool,
    pub local_ba_num_images: usize,
    pub local_ba_min_shared_points: usize,
    pub local_ba_iterations: usize,
    pub local_ba_max_refinements: usize,
    pub local_ba_max_refinement_change: f32,
    pub global_ba: bool,
    pub global_ba_iterations: usize,
    pub global_ba_images_ratio: f32,
    pub global_ba_points_ratio: f32,
    pub global_ba_images_freq: usize,
    pub global_ba_points_freq: usize,
    pub global_ba_max_refinements: usize,
    pub global_ba_max_refinement_change: f32,
    pub global_ba_ignore_redundant_points3d: bool,
    pub global_ba_ignore_redundant_points3d_min_coverage_gain: f64,
    pub ba_refine_focal_length: bool,
    pub ba_refine_principal_point: bool,
    pub ba_refine_extra_params: bool,
    pub ba_constant_rig_ids: Vec<u32>,
    pub ba_constant_camera_ids: Vec<u32>,
    pub min_focal_length_ratio: f64,
    pub max_focal_length_ratio: f64,
    pub max_extra_param: f64,
    pub max_reprojection_error_px: f32,
    pub ignore_two_view_tracks: bool,
    /// Run the GLOMAP-style global mapper pipeline instead of incremental SfM.
    pub global_mapper: bool,
    pub pose_graph: bool,
    pub copy_images: bool,
    pub threads: Option<usize>,
    pub fx: Option<f32>,
    pub fy: Option<f32>,
    pub cx: Option<f32>,
    pub cy: Option<f32>,
}

impl Default for MapperConfig {
    fn default() -> Self {
        Self {
            input: PathBuf::new(),
            output: PathBuf::new(),
            reference: None,
            database: None,
            write_two_view_geometries: false,
            ignore_database_two_view_poses: false,
            write_database: false,
            max_images: None,
            multiple_models: true,
            max_num_models: 50,
            max_model_overlap: 20,
            min_model_size: 10,
            snapshot_path: None,
            snapshot_frames_freq: 0,
            extract_colors: true,
            fix_existing_frames: false,
            feature_type: FeatureType::Sift,
            max_features: 8192,
            match_ratio: 0.8,
            sift_extraction: SiftExtractionOptions::default(),
            sift_matching: SiftMatchingOptions::default(),
            max_hamming_distance: 160.0,
            local_matching: false,
            local_window: 0,
            matching_pair_strategy: MatchingPairStrategy::default(),
            experimental_sequence_heuristics: false,
            experimental_ring_closure: false,
            experimental_structureless_pair_pose_fallback: true,
            min_matches: 15,
            min_inliers: 15,
            min_triangulated: 4,
            init_min_num_inliers: 100,
            init_min_tri_angle_deg: 16.0,
            init_max_forward_motion: 0.95,
            init_num_trials: 200,
            init_max_reg_trials: 2,
            essential_threshold_px: 4.0,
            essential_iterations: 10000,
            pnp_threshold_px: 12.0,
            pnp_iterations: 10000,
            use_gpu_pnp: false,
            abs_pose_min_num_inliers: 30,
            abs_pose_min_inlier_ratio: 0.25,
            image_selection_method: ImageSelectionMethod::MinUncertainty,
            pose_priors: Vec::new(),
            random_seed: -1,
            max_reg_trials: 3,
            local_ba: true,
            local_ba_num_images: 6,
            local_ba_min_shared_points: 15,
            local_ba_iterations: 25,
            local_ba_max_refinements: 2,
            local_ba_max_refinement_change: 0.001,
            global_ba: true,
            global_ba_iterations: 50,
            global_ba_images_ratio: 1.5,
            global_ba_points_ratio: 1.5,
            global_ba_images_freq: 500,
            global_ba_points_freq: 250_000,
            global_ba_max_refinements: 5,
            global_ba_max_refinement_change: 0.0005,
            global_ba_ignore_redundant_points3d: false,
            global_ba_ignore_redundant_points3d_min_coverage_gain: 0.05,
            ba_refine_focal_length: true,
            ba_refine_principal_point: false,
            ba_refine_extra_params: true,
            ba_constant_rig_ids: Vec::new(),
            ba_constant_camera_ids: Vec::new(),
            min_focal_length_ratio: 0.1,
            max_focal_length_ratio: 10.0,
            max_extra_param: 1.0,
            max_reprojection_error_px: 8.0,
            ignore_two_view_tracks: true,
            global_mapper: false,
            pose_graph: false,
            copy_images: true,
            threads: None,
            fx: None,
            fy: None,
            cx: None,
            cy: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconstructionSummary {
    pub images: usize,
    pub registered_images: usize,
    pub points: usize,
    pub pairs: usize,
    pub models: usize,
    pub elapsed_ms: f64,
    pub debug_log: Vec<String>,
}
