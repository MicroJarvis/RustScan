//! RustSLAM - A pure Rust Visual SLAM library
//!
//! ## Quick Start
//!
//! ```rust
//! use rustslam::{SE3, Frame};
//! ```

// Re-export core types
pub use core::{Camera, Frame, FrameFeatures, KeyFrame, Map, MapPoint, SE3};
pub use features::{
    Descriptors, DistanceMetric, FeatureExtractor, FeatureMatcher, KeyPoint, KnnMatcher, Match,
    OrbExtractor,
};
pub use colmap_rng::{sample_unique_indices, ColmapMt19937, ColmapRandomSampler};
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
