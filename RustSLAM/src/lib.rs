//! RustSLAM - A pure Rust Visual SLAM library
//!
//! ## Quick Start
//!
//! ```rust
//! use rustslam::{SE3, Frame};
//! ```

// Re-export core types
pub use colmap_rng::{
    colmap_ransac_num_trials, n_choose_k, sample_unique_indices, ColmapCombinationSampler,
    ColmapMt19937, ColmapProgressiveSampler, ColmapRandomSampler, ColmapRansacOptions,
    COLMAP_RANSAC_DEFAULT_MAX_NUM_TRIALS,
};
pub use core::{Camera, Frame, FrameFeatures, KeyFrame, Map, MapPoint, SE3};
pub use features::{
    Descriptors, DistanceMetric, FeatureExtractor, FeatureMatcher, KeyPoint, KnnMatcher, Match,
    OrbExtractor,
};
pub use tracker::{VOResult, VOState, VisualOdometry};

// Modules
#[cfg(feature = "slam-pipeline")]
pub mod cli;
pub mod colmap_rng;
pub mod config;
pub mod core;
pub mod depth;
pub mod features;
pub mod fusion;
#[cfg(feature = "slam-pipeline")]
pub mod io;
pub mod loop_closing;
pub mod mapping;
pub mod optimizer;
pub mod pipeline;
pub mod test_utils;
pub mod tracker;
pub mod viewer;
