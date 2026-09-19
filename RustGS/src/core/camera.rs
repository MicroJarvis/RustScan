//! Gaussian camera for rendering.
//!
//! This module will be populated from RustSLAM/src/fusion/gaussian.rs in Story 9-3.

use crate::core::matrix::{homogeneous_from_quat_translation, se3_to_homogeneous_matrix};
use crate::Intrinsics;
use crate::SE3;
use nalgebra::{Matrix4, Point3, Vector3};
use rustscan_types::Rotation3f;

/// Camera for Gaussian rendering.
///
/// Combines intrinsics and extrinsics for rendering.
#[derive(Debug, Clone)]
pub struct GaussianCamera {
    /// Camera intrinsics
    pub intrinsics: Intrinsics,
    /// Camera extrinsics (`camera_from_world`)
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

    /// View matrix `camera_from_world`.
    pub fn view_matrix(&self) -> Matrix4<f32> {
        se3_to_homogeneous_matrix(self.extrinsics)
    }

    /// Get the world-space camera position.
    pub fn position(&self) -> Vector3<f32> {
        self.extrinsics.inverse().vec()
    }

    /// OpenGL-style projection matrix.
    pub fn projection_matrix(&self) -> Matrix4<f32> {
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

        let w = width as f32;
        let h = height as f32;

        Matrix4::new(
            2.0 * fx / w,
            0.0,
            1.0 - 2.0 * cx / w,
            0.0,
            0.0,
            2.0 * fy / h,
            1.0 - 2.0 * cy / h,
            0.0,
            0.0,
            0.0,
            -(far + near) / (far - near),
            -2.0 * far * near / (far - near),
            0.0,
            0.0,
            -1.0,
            0.0,
        )
    }

    /// Project a 3D point to screen coordinates.
    pub fn project(&self, point: [f32; 3]) -> Option<[f32; 2]> {
        let p = self
            .extrinsics
            .transform_point3(Point3::new(point[0], point[1], point[2]));
        let z = p.z;
        if z <= 0.0 {
            return None;
        }

        let Intrinsics { fx, fy, cx, cy, .. } = self.intrinsics;
        let u = fx * p.x / z + cx;
        let v = fy * p.y / z + cy;

        Some([u, v])
    }
}

/// Build a `camera_from_world` view matrix from a pose.
pub fn viewmat_from_pose(rotation: Rotation3f, translation: Vector3<f32>) -> Matrix4<f32> {
    homogeneous_from_quat_translation(rotation, translation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector4;
    use rustscan_types::matrix4_to_column_major_array;

    fn non_symmetric_camera() -> (GaussianCamera, Rotation3f, Vector3<f32>) {
        let rotation = Rotation3f::from_euler_angles(0.3, -0.4, 0.5);
        let translation = Vector3::new(1.25, -2.5, 3.75);
        let camera = GaussianCamera::new(
            Intrinsics::new(500.0, 510.0, 321.5, 239.25, 640, 480),
            SE3::from_quat_translation(rotation, translation),
        );
        (camera, rotation, translation)
    }

    #[test]
    fn view_matrix_matches_pose_and_gpu_column_major_layout() {
        let (camera, rotation, translation) = non_symmetric_camera();
        let view = camera.view_matrix();
        let from_pose = viewmat_from_pose(rotation, translation);
        let expected = camera.extrinsics.to_homogeneous_matrix();
        let packed = matrix4_to_column_major_array(view);

        for col in 0..4 {
            for row in 0..4 {
                let actual = view[(row, col)];
                assert!((actual - from_pose[(row, col)]).abs() < 1e-6);
                assert!((actual - expected[(row, col)]).abs() < 1e-6);
                assert!((packed[col][row] - actual).abs() < 1e-7);
            }
        }

        let point = Vector4::new(0.4, -1.2, 2.8, 1.0);
        let transformed = view * point;
        let expected_point = camera
            .extrinsics
            .transform_point3(Point3::new(point.x, point.y, point.z));
        assert!((transformed.x - expected_point.x).abs() < 1e-5);
        assert!((transformed.y - expected_point.y).abs() < 1e-5);
        assert!((transformed.z - expected_point.z).abs() < 1e-5);
        assert!((transformed.w - 1.0).abs() < 1e-5);
    }

    #[test]
    fn projection_matrix_keeps_opengl_layout_for_offset_principal_point() {
        let (camera, _, _) = non_symmetric_camera();
        let projection = camera.projection_matrix();
        let w = camera.intrinsics.width as f32;
        let h = camera.intrinsics.height as f32;
        let near = 0.01;
        let far = 100.0;

        assert!((projection[(0, 0)] - 2.0 * camera.intrinsics.fx / w).abs() < 1e-6);
        assert!((projection[(1, 1)] - 2.0 * camera.intrinsics.fy / h).abs() < 1e-6);
        assert!((projection[(0, 2)] - (1.0 - 2.0 * camera.intrinsics.cx / w)).abs() < 1e-6);
        assert!((projection[(1, 2)] - (1.0 - 2.0 * camera.intrinsics.cy / h)).abs() < 1e-6);
        assert!((projection[(2, 2)] - (-(far + near) / (far - near))).abs() < 1e-6);
        assert!((projection[(2, 3)] - (-2.0 * far * near / (far - near))).abs() < 1e-6);
        assert!((projection[(3, 2)] + 1.0).abs() < 1e-6);
        assert!((projection[(0, 3)]).abs() < 1e-7);
        assert!((projection[(3, 0)]).abs() < 1e-7);
        assert!((projection[(3, 3)]).abs() < 1e-7);
    }
}
