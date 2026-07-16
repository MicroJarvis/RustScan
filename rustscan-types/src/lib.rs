//! Shared types for RustScan crates.
//!
//! This crate provides common types shared between RustSLAM, RustGS, RustMesh, and RustViewer:
//! - `SE3`: 3D pose representation (rotation + translation)
//! - `Intrinsics`: Camera intrinsic parameters
//! - `TrainingDataset`: Dataset for 3DGS offline training
//! - `SlamOutput`: SLAM output format for downstream processing

pub mod camera;
pub mod colmap;
pub mod pose;
pub mod scene;

// Re-exports
pub use camera::{Intrinsics, ScenePose, TrainingDataset};
pub use pose::SE3;
pub use scene::{MapPointData, SlamOutput};
