//! Gaussian camera for rendering.
//!
//! This module will be populated from RustSLAM/src/fusion/gaussian.rs in Story 9-3.

use crate::Intrinsics;
use crate::SE3;
use glam::{Mat4, Quat, Vec3};

/// Camera for Gaussian rendering.
///
/// Combines intrinsics and extrinsics for rendering.
#[derive(Debug, Clone)]
pub struct GaussianCamera {
    /// Camera intrinsics
    pub intrinsics: Intrinsics,
    /// Camera extrinsics (world-to-camera transform)
    pub extrinsics: SE3,
}

impl GaussianCamera {
    /// Create a new Gaussian camera.
    pub fn new(intrinsics: Intrinsics, extrinsics: SE3) -> Self {
        Self {
            intrinsics,
            extrinsics,
        }
    }

    /// Get the view matrix (world-to-camera).
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::from_cols_array_2d(&self.extrinsics.to_matrix())
    }

    /// Get the world-space camera position.
    pub fn position(&self) -> Vec3 {
        self.extrinsics.inverse().vec()
    }

    /// Get the projection matrix.
    pub fn projection_matrix(&self) -> Mat4 {
        let Intrinsics {
            fx,
            fy,
            cx,
            cy,
            width,
            height,
        } = self.intrinsics;
        let near = 0.01;
        let far = 100.0;

        // OpenGL-style projection matrix
        let w = width as f32;
        let h = height as f32;

        Mat4::from_cols_array(&[
            2.0 * fx / w,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0 * fy / h,
            0.0,
            0.0,
            1.0 - 2.0 * cx / w,
            1.0 - 2.0 * cy / h,
            -(far + near) / (far - near),
            -1.0,
            0.0,
            0.0,
            -2.0 * far * near / (far - near),
            0.0,
        ])
    }

    /// Project a 3D point to screen coordinates.
    pub fn project(&self, point: [f32; 3]) -> Option<[f32; 2]> {
        let p = self.extrinsics.transform_point(&point);
        let z = p[2];
        if z <= 0.0 {
            return None;
        }

        let Intrinsics { fx, fy, cx, cy, .. } = self.intrinsics;
        let u = fx * p[0] / z + cx;
        let v = fy * p[1] / z + cy;

        Some([u, v])
    }
}

/// Build a world-to-camera view matrix from a pose.
pub fn viewmat_from_pose(rotation: Quat, translation: Vec3) -> Mat4 {
    Mat4::from_rotation_translation(rotation, translation)
}
