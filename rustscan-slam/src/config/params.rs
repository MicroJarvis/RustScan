//! Parameter structures for SLAM components

use serde::{Deserialize, Serialize};

/// Collect validation errors into a list of human-readable messages.
pub type ValidationErrors = Vec<String>;

/// Feature extractor type for tracking
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FeatureType {
    Orb,
    Harris,
    Fast,
}

impl Default for FeatureType {
    fn default() -> Self {
        Self::Orb
    }
}

/// Tracker/VO parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackerParams {
    /// Feature extractor type
    #[serde(default)]
    pub feature_type: FeatureType,
    /// Maximum number of features to detect
    pub max_features: usize,
    /// Minimum number of features to maintain
    pub min_features: usize,
    /// Number of pyramid levels
    pub pyramid_levels: u32,
    /// Patch size for ORB
    pub patch_size: u32,
    /// Scale factor between pyramid levels
    pub scale_factor: f32,
    /// FAST threshold
    pub fast_threshold: u32,
    /// Matching ratio threshold (Lowe's ratio)
    pub match_ratio: f32,
    /// Minimum matches to proceed
    pub min_matches: usize,
    /// Minimum inliers after PnP
    pub min_inliers: usize,
    /// Maximum iterations for PnP
    pub pnp_max_iterations: usize,
}

impl Default for TrackerParams {
    fn default() -> Self {
        Self {
            feature_type: FeatureType::Orb,
            max_features: 2000,
            min_features: 500,
            pyramid_levels: 8,
            patch_size: 31,
            scale_factor: 1.2,
            fast_threshold: 20,
            match_ratio: 0.75,
            min_matches: 50,
            min_inliers: 10,
            pnp_max_iterations: 20,
        }
    }
}

impl TrackerParams {
    pub fn validate(&self) -> ValidationErrors {
        let mut errs = Vec::new();
        if self.max_features == 0 {
            errs.push("tracker.max_features must be > 0".into());
        }
        if self.min_features == 0 {
            errs.push("tracker.min_features must be > 0".into());
        }
        if self.max_features > 0 && self.min_features > 0 && self.max_features <= self.min_features
        {
            errs.push(format!(
                "tracker.max_features ({}) must be > min_features ({})",
                self.max_features, self.min_features
            ));
        }
        if self.pyramid_levels == 0 {
            errs.push("tracker.pyramid_levels must be > 0".into());
        }
        if self.patch_size == 0 {
            errs.push("tracker.patch_size must be > 0".into());
        }
        if self.scale_factor <= 1.0 {
            errs.push(format!(
                "tracker.scale_factor ({}) must be > 1.0",
                self.scale_factor
            ));
        }
        if self.match_ratio <= 0.0 || self.match_ratio > 1.0 {
            errs.push(format!(
                "tracker.match_ratio ({}) must be in (0, 1]",
                self.match_ratio
            ));
        }
        if self.min_matches == 0 {
            errs.push("tracker.min_matches must be > 0".into());
        }
        if self.min_inliers == 0 {
            errs.push("tracker.min_inliers must be > 0".into());
        }
        if self.pnp_max_iterations == 0 {
            errs.push("tracker.pnp_max_iterations must be > 0".into());
        }
        errs
    }
}

/// Mapper parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapperParams {
    /// Maximum number of keyframes
    pub max_keyframes: usize,
    /// Minimum number of keyframes
    pub min_keyframes: usize,
    /// Keyframe interval (frames)
    pub keyframe_interval: usize,
    /// Maximum MapPoints per keyframe
    pub max_points_per_keyframe: usize,
    /// Maximum distance for new MapPoints
    pub max_point_distance: f32,
    /// Minimum distance for new MapPoints
    pub min_point_distance: f32,
    /// Maximum reprojection error
    pub max_reproj_error: f32,
    /// Minimum triangulation angle (degrees)
    pub min_triangulation_angle: f32,
    /// Whether to use local mapping
    pub use_local_mapping: bool,
    /// Local mapping window size
    pub local_mapping_window: usize,
}

impl Default for MapperParams {
    fn default() -> Self {
        Self {
            max_keyframes: 100,
            min_keyframes: 5,
            keyframe_interval: 5,
            max_points_per_keyframe: 500,
            max_point_distance: 50.0,
            min_point_distance: 0.1,
            max_reproj_error: 4.0,
            min_triangulation_angle: 3.0,
            use_local_mapping: true,
            local_mapping_window: 10,
        }
    }
}

impl MapperParams {
    pub fn validate(&self) -> ValidationErrors {
        let mut errs = Vec::new();
        if self.max_keyframes == 0 {
            errs.push("mapper.max_keyframes must be > 0".into());
        }
        if self.max_keyframes > 0
            && self.min_keyframes > 0
            && self.max_keyframes <= self.min_keyframes
        {
            errs.push(format!(
                "mapper.max_keyframes ({}) must be > min_keyframes ({})",
                self.max_keyframes, self.min_keyframes
            ));
        }
        if self.keyframe_interval == 0 {
            errs.push("mapper.keyframe_interval must be > 0".into());
        }
        if self.max_point_distance <= self.min_point_distance {
            errs.push(format!(
                "mapper.max_point_distance ({}) must be > min_point_distance ({})",
                self.max_point_distance, self.min_point_distance
            ));
        }
        if self.min_point_distance < 0.0 {
            errs.push("mapper.min_point_distance must be >= 0".into());
        }
        if self.max_reproj_error <= 0.0 {
            errs.push("mapper.max_reproj_error must be > 0".into());
        }
        if self.min_triangulation_angle <= 0.0 {
            errs.push("mapper.min_triangulation_angle must be > 0".into());
        }
        errs
    }
}

/// Optimizer parameters (Bundle Adjustment)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerParams {
    /// Number of iterations for local BA
    pub local_ba_iterations: usize,
    /// Number of iterations for full BA
    pub full_ba_iterations: usize,
    /// Number of iterations for pose optimization
    pub pose_iterations: usize,
    /// Maximum reprojection error (pixels)
    pub max_reproj_error: f32,
    /// Robust kernel threshold
    pub robust_kernel_threshold: f32,
    /// Use parallel BA
    pub use_parallel: bool,
    /// Number of threads for parallel BA
    pub num_threads: usize,
    /// Convergence threshold
    pub convergence_threshold: f32,
}

impl Default for OptimizerParams {
    fn default() -> Self {
        Self {
            local_ba_iterations: 20,
            full_ba_iterations: 100,
            pose_iterations: 10,
            max_reproj_error: 4.0,
            robust_kernel_threshold: 5.0,
            use_parallel: true,
            num_threads: 4,
            convergence_threshold: 1e-6,
        }
    }
}

impl OptimizerParams {
    pub fn validate(&self) -> ValidationErrors {
        let mut errs = Vec::new();
        if self.local_ba_iterations == 0 {
            errs.push("optimizer.local_ba_iterations must be > 0".into());
        }
        if self.full_ba_iterations == 0 {
            errs.push("optimizer.full_ba_iterations must be > 0".into());
        }
        if self.pose_iterations == 0 {
            errs.push("optimizer.pose_iterations must be > 0".into());
        }
        if self.max_reproj_error <= 0.0 {
            errs.push("optimizer.max_reproj_error must be > 0".into());
        }
        if self.robust_kernel_threshold <= 0.0 {
            errs.push("optimizer.robust_kernel_threshold must be > 0".into());
        }
        if self.use_parallel && self.num_threads == 0 {
            errs.push("optimizer.num_threads must be > 0 when use_parallel is true".into());
        }
        if self.convergence_threshold <= 0.0 {
            errs.push("optimizer.convergence_threshold must be > 0".into());
        }
        errs
    }
}

/// Loop closing parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopClosingParams {
    /// Minimum score for loop detection
    pub min_loop_score: f32,
    /// Minimum matches for geometric verification
    pub min_matches: usize,
    /// Minimum inliers after geometric verification
    pub min_inliers: usize,
    /// Minimum distance (keyframes) between current and loop candidate
    pub min_distance: usize,
    /// RANSAC iterations for geometric verification
    pub ransac_iterations: usize,
    /// Inlier threshold (pixels)
    pub inlier_threshold: f32,
    /// Use SIMD for descriptor matching
    pub use_simd: bool,
    /// Covisibility threshold for keyframe selection
    pub covisibility_threshold: f32,
    /// Enable similarity transform (Sim3) optimization
    pub use_sim3: bool,
    /// Number of iterations for essential matrix optimization
    pub essential_iterations: usize,
}

impl Default for LoopClosingParams {
    fn default() -> Self {
        Self {
            min_loop_score: 0.05,
            min_matches: 20,
            min_inliers: 15,
            min_distance: 30,
            ransac_iterations: 200,
            inlier_threshold: 3.0,
            use_simd: true,
            covisibility_threshold: 0.6,
            use_sim3: true,
            essential_iterations: 100,
        }
    }
}

impl LoopClosingParams {
    pub fn validate(&self) -> ValidationErrors {
        let mut errs = Vec::new();
        if self.min_loop_score <= 0.0 {
            errs.push("loop_closing.min_loop_score must be > 0".into());
        }
        if self.min_matches == 0 {
            errs.push("loop_closing.min_matches must be > 0".into());
        }
        if self.min_inliers == 0 {
            errs.push("loop_closing.min_inliers must be > 0".into());
        }
        if self.ransac_iterations == 0 {
            errs.push("loop_closing.ransac_iterations must be > 0".into());
        }
        if self.inlier_threshold <= 0.0 {
            errs.push("loop_closing.inlier_threshold must be > 0".into());
        }
        errs
    }
}

/// Dataset parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetParams {
    /// Dataset type (tum, kitti, euroc, realsense)
    pub dataset_type: String,
    /// Dataset root path
    pub root_path: String,
    /// Load depth images
    pub load_depth: bool,
    /// Load ground truth
    pub load_ground_truth: bool,
    /// Maximum frames to process
    pub max_frames: usize,
    /// Frame stride (process every N frames)
    pub stride: usize,
    /// Depth scale factor
    pub depth_scale: f32,
    /// Depth truncation threshold
    pub depth_trunc: f32,
}

impl Default for DatasetParams {
    fn default() -> Self {
        Self {
            dataset_type: "tum".to_string(),
            root_path: "".to_string(),
            load_depth: true,
            load_ground_truth: true,
            max_frames: 0,
            stride: 1,
            depth_scale: 1000.0,
            depth_trunc: 10.0,
        }
    }
}

impl DatasetParams {
    pub fn validate(&self) -> ValidationErrors {
        let mut errs = Vec::new();
        if self.stride == 0 {
            errs.push("dataset.stride must be > 0".into());
        }
        if self.depth_scale <= 0.0 {
            errs.push("dataset.depth_scale must be > 0".into());
        }
        if self.depth_trunc <= 0.0 {
            errs.push("dataset.depth_trunc must be > 0".into());
        }
        errs
    }
}

/// Viewer parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewerParams {
    /// Enable viewer
    pub enabled: bool,
    /// Viewer width
    pub width: u32,
    /// Viewer height
    pub height: u32,
    /// Point size
    pub point_size: f32,
    /// Line width
    pub line_width: f32,
    /// Camera view width
    pub camera_view_width: f32,
    /// Camera view height
    pub camera_view_height: f32,
    /// Background color (R, G, B)
    pub background_color: [f32; 3],
    /// Show keyframes
    pub show_keyframes: bool,
    /// Show map points
    pub show_points: bool,
    /// Show current frame
    pub show_current_frame: bool,
    /// Show trajectory
    pub show_trajectory: bool,
    /// Update period (ms)
    pub update_period_ms: u32,
}

impl Default for ViewerParams {
    fn default() -> Self {
        Self {
            enabled: true,
            width: 1024,
            height: 768,
            point_size: 2.0,
            line_width: 1.0,
            camera_view_width: 0.1,
            camera_view_height: 0.1,
            background_color: [0.8, 0.8, 0.8],
            show_keyframes: true,
            show_points: true,
            show_current_frame: true,
            show_trajectory: true,
            update_period_ms: 50,
        }
    }
}

impl ViewerParams {
    pub fn validate(&self) -> ValidationErrors {
        let mut errs = Vec::new();
        if self.width == 0 {
            errs.push("viewer.width must be > 0".into());
        }
        if self.height == 0 {
            errs.push("viewer.height must be > 0".into());
        }
        if self.point_size <= 0.0 {
            errs.push("viewer.point_size must be > 0".into());
        }
        if self.line_width <= 0.0 {
            errs.push("viewer.line_width must be > 0".into());
        }
        errs
    }
}

/// 3DGS (Gaussian Splatting) parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianSplattingParams {
    /// Maximum number of Gaussians
    pub max_gaussians: usize,
    /// Initial number of Gaussians
    pub init_gaussians: usize,
    /// Densify interval (iterations)
    pub densify_interval: usize,
    /// Densify threshold
    pub densify_threshold: f32,
    /// Prune interval (iterations)
    pub prune_interval: usize,
    /// Prune opacity threshold
    pub prune_opacity: f32,
    /// Learning rate for position
    pub lr_position: f32,
    /// Learning rate for rotation
    pub lr_rotation: f32,
    /// Learning rate for scaling
    pub lr_scale: f32,
    /// Learning rate for opacity
    pub lr_opacity: f32,
    /// Learning rate for color
    pub lr_color: f32,
    /// Batch size for training
    pub batch_size: usize,
    /// Number of training iterations
    pub num_iterations: usize,
    /// Use GPU acceleration
    pub use_gpu: bool,
}

impl Default for GaussianSplattingParams {
    fn default() -> Self {
        Self {
            max_gaussians: 100_000,
            init_gaussians: 1000,
            densify_interval: 100,
            densify_threshold: 0.0002,
            prune_interval: 100,
            prune_opacity: 0.05,
            lr_position: 0.00016,
            lr_rotation: 0.002,
            lr_scale: 0.005,
            lr_opacity: 0.05,
            lr_color: 0.0025,
            batch_size: 4096,
            num_iterations: 3_000,
            use_gpu: true,
        }
    }
}

impl GaussianSplattingParams {
    pub fn validate(&self) -> ValidationErrors {
        let mut errs = Vec::new();
        if self.max_gaussians == 0 {
            errs.push("gaussian_splatting.max_gaussians must be > 0".into());
        }
        if self.init_gaussians == 0 {
            errs.push("gaussian_splatting.init_gaussians must be > 0".into());
        }
        if self.max_gaussians > 0 && self.init_gaussians > self.max_gaussians {
            errs.push(format!(
                "gaussian_splatting.init_gaussians ({}) must be <= max_gaussians ({})",
                self.init_gaussians, self.max_gaussians
            ));
        }
        if self.lr_position <= 0.0 {
            errs.push("gaussian_splatting.lr_position must be > 0".into());
        }
        if self.lr_rotation <= 0.0 {
            errs.push("gaussian_splatting.lr_rotation must be > 0".into());
        }
        if self.lr_scale <= 0.0 {
            errs.push("gaussian_splatting.lr_scale must be > 0".into());
        }
        if self.lr_opacity <= 0.0 {
            errs.push("gaussian_splatting.lr_opacity must be > 0".into());
        }
        if self.lr_color <= 0.0 {
            errs.push("gaussian_splatting.lr_color must be > 0".into());
        }
        if self.batch_size == 0 {
            errs.push("gaussian_splatting.batch_size must be > 0".into());
        }
        if self.num_iterations == 0 {
            errs.push("gaussian_splatting.num_iterations must be > 0".into());
        }
        errs
    }
}

/// TSDF volume parameters for mesh extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsdfParams {
    /// Voxel size (meters)
    pub voxel_size: f32,
    /// Truncation distance
    pub trunc_dist: f32,
    /// Volume size (meters)
    pub volume_size: f32,
    /// Marching cubes isolevel
    pub isolevel: f32,
    /// Maximum weight per voxel
    pub max_weight: u32,
}

impl Default for TsdfParams {
    fn default() -> Self {
        Self {
            voxel_size: 0.01,
            trunc_dist: 0.03,
            volume_size: 2.0,
            isolevel: 0.0,
            max_weight: 100,
        }
    }
}

impl TsdfParams {
    pub fn validate(&self) -> ValidationErrors {
        let mut errs = Vec::new();
        if self.voxel_size <= 0.0 {
            errs.push("tsdf.voxel_size must be > 0".into());
        }
        if self.trunc_dist <= 0.0 {
            errs.push("tsdf.trunc_dist must be > 0".into());
        }
        if self.volume_size <= 0.0 {
            errs.push("tsdf.volume_size must be > 0".into());
        }
        if self.max_weight == 0 {
            errs.push("tsdf.max_weight must be > 0".into());
        }
        errs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_params_default() {
        let params = TrackerParams::default();
        assert_eq!(params.max_features, 2000);
        assert!(matches!(params.feature_type, FeatureType::Orb));
    }

    #[test]
    fn test_mapper_params_default() {
        let params = MapperParams::default();
        assert_eq!(params.max_keyframes, 100);
    }

    #[test]
    fn test_optimizer_params_default() {
        let params = OptimizerParams::default();
        assert_eq!(params.local_ba_iterations, 20);
    }

    #[test]
    fn test_loop_closing_params_default() {
        let params = LoopClosingParams::default();
        assert_eq!(params.min_loop_score, 0.05);
    }

    #[test]
    fn test_dataset_params_default() {
        let params = DatasetParams::default();
        assert_eq!(params.dataset_type, "tum");
    }

    #[test]
    fn test_viewer_params_default() {
        let params = ViewerParams::default();
        assert!(params.enabled);
    }

    #[test]
    fn test_gaussian_splatting_params_default() {
        let params = GaussianSplattingParams::default();
        assert_eq!(params.max_gaussians, 100_000);
    }

    #[test]
    fn test_tsdf_params_default() {
        let params = TsdfParams::default();
        assert_eq!(params.voxel_size, 0.01);
    }

    #[test]
    fn test_defaults_pass_validation() {
        assert!(TrackerParams::default().validate().is_empty());
        assert!(MapperParams::default().validate().is_empty());
        assert!(OptimizerParams::default().validate().is_empty());
        assert!(LoopClosingParams::default().validate().is_empty());
        assert!(DatasetParams::default().validate().is_empty());
        assert!(ViewerParams::default().validate().is_empty());
        assert!(GaussianSplattingParams::default().validate().is_empty());
        assert!(TsdfParams::default().validate().is_empty());
    }

    #[test]
    fn test_tracker_validation_max_lte_min() {
        let mut p = TrackerParams::default();
        p.max_features = 100;
        p.min_features = 200;
        let errs = p.validate();
        assert!(errs.iter().any(|e| e.contains("max_features")));
    }

    #[test]
    fn test_tracker_validation_scale_factor() {
        let mut p = TrackerParams::default();
        p.scale_factor = 0.5;
        let errs = p.validate();
        assert!(errs.iter().any(|e| e.contains("scale_factor")));
    }

    #[test]
    fn test_tracker_validation_match_ratio() {
        let mut p = TrackerParams::default();
        p.match_ratio = 1.5;
        let errs = p.validate();
        assert!(errs.iter().any(|e| e.contains("match_ratio")));
    }

    #[test]
    fn test_mapper_validation_distances() {
        let mut p = MapperParams::default();
        p.max_point_distance = 0.05;
        p.min_point_distance = 0.1;
        let errs = p.validate();
        assert!(errs.iter().any(|e| e.contains("max_point_distance")));
    }

    #[test]
    fn test_optimizer_validation_threads() {
        let mut p = OptimizerParams::default();
        p.use_parallel = true;
        p.num_threads = 0;
        let errs = p.validate();
        assert!(errs.iter().any(|e| e.contains("num_threads")));
    }

    #[test]
    fn test_gaussian_splatting_init_gt_max() {
        let mut p = GaussianSplattingParams::default();
        p.init_gaussians = 200_000;
        p.max_gaussians = 100_000;
        let errs = p.validate();
        assert!(errs.iter().any(|e| e.contains("init_gaussians")));
    }

    #[test]
    fn test_tsdf_validation_voxel_size() {
        let mut p = TsdfParams::default();
        p.voxel_size = -1.0;
        let errs = p.validate();
        assert!(errs.iter().any(|e| e.contains("voxel_size")));
    }
}
