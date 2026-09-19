//! Shared types for RustScan crates.
//!
//! This crate provides common types shared between RustSLAM, RustGS, RustMesh, and RustViewer:
//! - `SE3`: 3D pose representation (rotation + translation)
//! - `Intrinsics`: Camera intrinsic parameters
//! - `TrainingDataset`: Dataset for 3DGS offline training
//! - `SlamOutput`: SLAM output format for downstream processing

pub mod camera;
pub mod colmap;
pub mod math;
pub mod pose;
pub mod scene;

// Re-exports
pub use camera::{Intrinsics, ScenePose, TrainingDataset};
pub use math::{
    matrix3_from_column_major_array, matrix3_from_rotation, matrix3_from_row_major_array,
    matrix3_to_column_major_array, matrix3_to_row_major_array, matrix4_from_column_major_array,
    matrix4_from_row_major_array, matrix4_to_column_major_array, matrix4_to_row_major_array,
    point3_from_array, point3_to_array, rotation_from_axis_angle_x, rotation_from_axis_angle_y,
    rotation_from_axis_angle_z, rotation_from_matrix3, rotation_from_scaled_axis,
    rotation_from_wxyz, rotation_from_xyzw, rotation_to_wxyz, rotation_to_xyzw, vector3_from_array,
    vector3_to_array, Mat3d, Mat3f, Mat4d, Mat4f, Point2d, Point2f, Point3d, Point3f, Rotation3d,
    Rotation3f, Vec2d, Vec2f, Vec3d, Vec3f, Vec4d, Vec4f,
};
pub use pose::SE3;
pub use scene::{MapPointData, SlamOutput};
