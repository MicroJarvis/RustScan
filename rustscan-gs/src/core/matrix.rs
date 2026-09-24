//! CPU matrix helpers for RustGS.
//!
//! Camera and scene transforms use `nalgebra`. GPU uniforms still use
//! column-major arrays, converted with explicit names.

use nalgebra::{Isometry3, Matrix3, Matrix4, Translation3, UnitQuaternion, Vector3};
use rustscan_types::{Rotation3f, SE3};

pub use rustscan_types::matrix4_to_column_major_array;

/// Build a homogeneous rigid transform from a unit quaternion and translation.
///
/// The result maps column vectors as `p' = M * p`.
pub fn homogeneous_from_quat_translation(
    rotation: Rotation3f,
    translation: Vector3<f32>,
) -> Matrix4<f32> {
    Isometry3::from_parts(Translation3::from(translation), rotation).to_homogeneous()
}

/// Homogeneous matrix for an `SE3` pose: `p' = M * p`.
///
/// For `GaussianCamera` extrinsics this is `camera_from_world`.
pub fn se3_to_homogeneous_matrix(pose: SE3) -> Matrix4<f32> {
    pose.to_homogeneous_matrix()
}

pub(crate) fn matrix3_from_quat(rotation: Rotation3f) -> Matrix3<f32> {
    rotation.to_rotation_matrix().into_inner()
}

pub(crate) fn quat_from_matrix3(rotation: &Matrix3<f32>) -> Rotation3f {
    UnitQuaternion::from_matrix(rotation)
}

pub(crate) fn rotate_vec3(rotation: Matrix3<f32>, vector: Vector3<f32>) -> Vector3<f32> {
    rotation * vector
}

pub(crate) fn homogeneous_from_linear_translation(
    linear: Matrix3<f32>,
    translation: Vector3<f32>,
) -> Matrix4<f32> {
    Matrix4::new(
        linear[(0, 0)],
        linear[(0, 1)],
        linear[(0, 2)],
        translation.x,
        linear[(1, 0)],
        linear[(1, 1)],
        linear[(1, 2)],
        translation.y,
        linear[(2, 0)],
        linear[(2, 1)],
        linear[(2, 2)],
        translation.z,
        0.0,
        0.0,
        0.0,
        1.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector4;

    fn non_symmetric_pose() -> (Rotation3f, Vector3<f32>, SE3) {
        let rotation = Rotation3f::from_euler_angles(0.3, -0.4, 0.5);
        let translation = Vector3::new(1.25, -2.5, 3.75);
        (
            rotation,
            translation,
            SE3::from_quat_translation(rotation, translation),
        )
    }

    #[test]
    fn se3_homogeneous_matrix_matches_column_major_pose_array() {
        let (_, _, pose) = non_symmetric_pose();
        let matrix = se3_to_homogeneous_matrix(pose);
        let expected = pose.to_column_major_array();

        for col in 0..4 {
            for row in 0..4 {
                let actual = matrix[(row, col)];
                let expected = expected[col][row];
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "mismatch at ({row}, {col}): {actual} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn column_major_array_round_trips_non_symmetric_matrix() {
        let (_, _, pose) = non_symmetric_pose();
        let matrix = se3_to_homogeneous_matrix(pose);
        let packed = matrix4_to_column_major_array(matrix);
        let restored = matrix4_from_column_major_array(&packed);

        for col in 0..4 {
            for row in 0..4 {
                assert!((matrix[(row, col)] - restored[(row, col)]).abs() < 1e-7);
                assert!((packed[col][row] - matrix[(row, col)]).abs() < 1e-7);
            }
        }

        let point = Vector4::new(0.7, -1.1, 2.3, 1.0);
        let expected = matrix * point;
        let actual = restored * point;
        for idx in 0..4 {
            assert!((expected[idx] - actual[idx]).abs() < 1e-6);
        }
    }
}
