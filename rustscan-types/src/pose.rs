//! Shared SE(3) pose: `UnitQuaternion<f32>` + `Vector3<f32>`.
//!
//! This type is the single CPU pose representation for RustScan. Array accessors
//! are explicit boundary conversions and name their layout.

use crate::math::{
    matrix3_from_row_major_array, matrix3_to_column_major_array, matrix3_to_row_major_array,
    matrix4_from_column_major_array, matrix4_from_row_major_array, matrix4_to_column_major_array,
    matrix4_to_row_major_array, point3_from_array, point3_to_array, rotation_from_scaled_axis,
    rotation_from_xyzw, rotation_to_xyzw, vector3_from_array, vector3_to_array, Mat3f, Mat4f,
    Point3f, Rotation3f, Vec3f,
};
use nalgebra::{Isometry3, Matrix3, Translation3, Vector3};
use serde::{Deserialize, Serialize};

/// Rigid pose `p' = R * p + t`.
///
/// Callers must document whether a stored pose is `world_from_camera` or
/// `camera_from_world`. `ScenePose.pose` is camera-to-world.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SE3 {
    rotation: Rotation3f,
    translation: Vec3f,
}

#[derive(Serialize, Deserialize)]
struct SE3Serde {
    rotation: QuatSerde,
    translation: Vec3Serde,
}

#[derive(Serialize, Deserialize)]
struct QuatSerde {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

#[derive(Serialize, Deserialize)]
struct Vec3Serde {
    x: f32,
    y: f32,
    z: f32,
}

impl Serialize for SE3 {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        SE3Serde::from(*self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SE3 {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        SE3Serde::deserialize(deserializer).map(Self::from)
    }
}

impl From<SE3> for SE3Serde {
    fn from(pose: SE3) -> Self {
        let xyzw = rotation_to_xyzw(pose.rotation);
        Self {
            rotation: QuatSerde {
                x: xyzw[0],
                y: xyzw[1],
                z: xyzw[2],
                w: xyzw[3],
            },
            translation: Vec3Serde {
                x: pose.translation.x,
                y: pose.translation.y,
                z: pose.translation.z,
            },
        }
    }
}

impl From<SE3Serde> for SE3 {
    fn from(value: SE3Serde) -> Self {
        SE3 {
            rotation: rotation_from_xyzw([
                value.rotation.x,
                value.rotation.y,
                value.rotation.z,
                value.rotation.w,
            ]),
            translation: Vector3::new(
                value.translation.x,
                value.translation.y,
                value.translation.z,
            ),
        }
    }
}

impl SE3 {
    /// Create from `[x, y, z, w]` and translation `[x, y, z]`.
    pub fn new(quaternion_xyzw: &[f32; 4], translation: &[f32; 3]) -> Self {
        Self {
            rotation: rotation_from_xyzw(*quaternion_xyzw),
            translation: vector3_from_array(*translation),
        }
    }

    pub fn from_axis_angle(axis_angle: &[f32; 3], translation: &[f32; 3]) -> Self {
        Self {
            rotation: rotation_from_scaled_axis(vector3_from_array(*axis_angle)),
            translation: vector3_from_array(*translation),
        }
    }

    pub fn identity() -> Self {
        Self {
            rotation: Rotation3f::identity(),
            translation: Vector3::zeros(),
        }
    }

    /// Create from a row-major `[[row]; 3]` rotation and translation.
    ///
    /// This matches the previous RustSLAM array convention: `rotation[row][col]`.
    pub fn from_rotation_translation(rotation: &[[f32; 3]; 3], translation: &[f32; 3]) -> Self {
        Self::from_rotation_matrix(
            matrix3_from_row_major_array(rotation),
            vector3_from_array(*translation),
        )
    }

    pub fn from_rotation_matrix(rotation: Mat3f, translation: Vec3f) -> Self {
        Self {
            rotation: Rotation3f::from_matrix(&rotation),
            translation,
        }
    }

    pub fn from_quat_translation(rotation: Rotation3f, translation: Vec3f) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    pub fn from_row_major_array(rows: &[[f32; 4]; 4]) -> Self {
        Self::from_homogeneous_matrix(matrix4_from_row_major_array(rows))
    }

    pub fn from_column_major_array(columns: &[[f32; 4]; 4]) -> Self {
        Self::from_homogeneous_matrix(matrix4_from_column_major_array(columns))
    }

    pub fn from_homogeneous_matrix(matrix: Mat4f) -> Self {
        let rotation = matrix.fixed_view::<3, 3>(0, 0).into_owned();
        let translation = Vector3::new(matrix[(0, 3)], matrix[(1, 3)], matrix[(2, 3)]);
        Self::from_rotation_matrix(rotation, translation)
    }

    /// Homogeneous rigid matrix, `p' = M * p`.
    pub fn to_homogeneous_matrix(&self) -> Mat4f {
        self.isometry().to_homogeneous()
    }

    /// Row-major `[[row]; 4]` array matching the previous RustSLAM `to_matrix()` layout.
    pub fn to_matrix(&self) -> [[f32; 4]; 4] {
        self.to_row_major_array()
    }

    /// Row-major `[[row]; 4]` GPU/wire array: `array[row][col]`.
    pub fn to_row_major_array(&self) -> [[f32; 4]; 4] {
        matrix4_to_row_major_array(self.to_homogeneous_matrix())
    }

    /// Column-major `[[column]; 4]` GPU/wire array: `array[col][row]`.
    pub fn to_column_major_array(&self) -> [[f32; 4]; 4] {
        matrix4_to_column_major_array(self.to_homogeneous_matrix())
    }

    pub fn compose(&self, other: &SE3) -> SE3 {
        SE3 {
            rotation: self.rotation * other.rotation,
            translation: self.translation + self.rotation * other.translation,
        }
    }

    pub fn inverse(&self) -> SE3 {
        let rotation = self.rotation.inverse();
        SE3 {
            translation: -(rotation * self.translation),
            rotation,
        }
    }

    pub fn transform_point3(&self, point: Point3f) -> Point3f {
        self.rotation * point + self.translation
    }

    pub fn transform_vector3(&self, vector: Vec3f) -> Vec3f {
        self.rotation * vector
    }

    /// Transform a point given as `[x, y, z]` coordinates.
    pub fn transform_point(&self, point: &[f32; 3]) -> [f32; 3] {
        point3_to_array(self.transform_point3(point3_from_array(*point)))
    }

    /// Transform a direction given as `[x, y, z]` coordinates.
    pub fn transform_vector(&self, vector: &[f32; 3]) -> [f32; 3] {
        vector3_to_array(self.transform_vector3(vector3_from_array(*vector)))
    }

    pub fn transform_point_array(&self, point: &[f32; 3]) -> [f32; 3] {
        self.transform_point(point)
    }

    pub fn transform_vector_array(&self, vector: &[f32; 3]) -> [f32; 3] {
        self.transform_vector(vector)
    }

    pub fn rotation_matrix3(&self) -> Mat3f {
        self.rotation.to_rotation_matrix().into_inner()
    }

    /// Row-major `[[row]; 3]` array matching the previous RustSLAM accessor.
    pub fn rotation_matrix(&self) -> [[f32; 3]; 3] {
        matrix3_to_row_major_array(self.rotation_matrix3())
    }

    pub fn rotation_matrix_row_major_array(&self) -> [[f32; 3]; 3] {
        self.rotation_matrix()
    }

    pub fn rotation_matrix_column_major_array(&self) -> [[f32; 3]; 3] {
        matrix3_to_column_major_array(self.rotation_matrix3())
    }

    pub fn translation_vector(&self) -> Vec3f {
        self.translation
    }

    pub fn translation(&self) -> [f32; 3] {
        vector3_to_array(self.translation)
    }

    /// Quaternion as `[x, y, z, w]`.
    pub fn quaternion(&self) -> [f32; 4] {
        rotation_to_xyzw(self.rotation)
    }

    pub fn rotation_quat(&self) -> Rotation3f {
        self.rotation
    }

    pub fn quat(&self) -> Rotation3f {
        self.rotation
    }

    pub fn vec(&self) -> Vec3f {
        self.translation
    }

    pub fn axis_angle(&self) -> [f32; 3] {
        vector3_to_array(self.rotation.scaled_axis())
    }

    /// Lie exponential: tangent is `[ωx, ωy, ωz, vx, vy, vz]`.
    pub fn exp(tangent: &[f32; 6]) -> Self {
        let omega = Vector3::new(tangent[0], tangent[1], tangent[2]);
        let v = Vector3::new(tangent[3], tangent[4], tangent[5]);
        let angle = omega.norm();
        let rotation = rotation_from_scaled_axis(omega);
        let translation = if angle < 1e-10 {
            v
        } else {
            let c1 = (1.0 - angle.cos()) / (angle * angle);
            let c2 = (angle - angle.sin()) / (angle * angle * angle);
            let omega_hat = skew_symmetric(omega);
            v + (c1 * omega_hat * v) + (c2 * omega_hat * omega_hat * v)
        };
        Self {
            rotation,
            translation,
        }
    }

    pub fn log(&self) -> [f32; 6] {
        let omega = self.rotation.scaled_axis();
        let angle = omega.norm();
        let t = self.translation;
        let v = if angle < 1e-10 {
            t
        } else {
            let c1 = (1.0 - angle.cos()) / (angle * angle);
            let c2 = (angle - angle.sin()) / (angle * angle * angle);
            let omega_hat = skew_symmetric(omega);
            t - (c1 * omega_hat * t) + (c2 * omega_hat * omega_hat * t)
        };
        [omega.x, omega.y, omega.z, v.x, v.y, v.z]
    }

    pub fn rotation(&self) -> [[f32; 3]; 3] {
        self.rotation_matrix_row_major_array()
    }

    fn isometry(&self) -> Isometry3<f32> {
        Isometry3::from_parts(Translation3::from(self.translation), self.rotation)
    }
}

impl Default for SE3 {
    fn default() -> Self {
        Self::identity()
    }
}

fn skew_symmetric(omega: Vec3f) -> Matrix3<f32> {
    Matrix3::new(
        0.0, -omega.z, omega.y, omega.z, 0.0, -omega.x, -omega.y, omega.x, 0.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{matrix4_from_column_major_array, matrix4_from_row_major_array};
    use nalgebra::{Point3, Vector4};

    fn non_symmetric_pose() -> SE3 {
        SE3::from_quat_translation(
            Rotation3f::from_euler_angles(0.3, -0.4, 0.5),
            Vector3::new(1.25, -2.5, 3.75),
        )
    }

    #[test]
    fn identity_homogeneous_matrix_is_identity() {
        let matrix = SE3::identity().to_homogeneous_matrix();
        assert!((matrix - Mat4f::identity()).norm() < 1e-10);
    }

    #[test]
    fn compose_applies_translation() {
        let a = SE3::identity();
        let b = SE3::from_axis_angle(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0]);
        let c = a.compose(&b);
        assert!((c.translation()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn inverse_round_trips_non_symmetric_pose() {
        let pose = non_symmetric_pose();
        let composed = pose.compose(&pose.inverse());
        assert!((composed.to_homogeneous_matrix() - Mat4f::identity()).norm() < 1e-5);
    }

    #[test]
    fn transform_point_and_vector_use_distinct_semantics() {
        let pose = non_symmetric_pose();
        let point = Point3::new(0.7, -1.1, 2.3);
        let direction = Vector3::new(0.7, -1.1, 2.3);
        let rotated = pose.rotation_quat() * direction;
        let translated = rotated + pose.translation_vector();
        let transformed_point = pose.transform_point3(point);
        let transformed_vector = pose.transform_vector3(direction);
        assert!((transformed_point.coords - translated).norm() < 1e-6);
        assert!((transformed_vector - rotated).norm() < 1e-6);
        assert!((transformed_point.coords - transformed_vector).norm() > 0.5);
    }

    #[test]
    fn row_and_column_major_arrays_round_trip_non_symmetric_pose() {
        let pose = non_symmetric_pose();
        let matrix = pose.to_homogeneous_matrix();
        let rows = pose.to_row_major_array();
        let cols = pose.to_column_major_array();
        assert!((rows[0][3] - pose.translation_vector().x).abs() < 1e-6);
        assert!((cols[3][0] - pose.translation_vector().x).abs() < 1e-6);
        assert!((rows[0][1] - cols[1][0]).abs() < 1e-7);

        let from_rows = SE3::from_row_major_array(&rows);
        let from_cols = SE3::from_column_major_array(&cols);
        let point = Point3::new(0.4, -1.2, 2.8);
        let expected = pose.transform_point3(point);
        assert!((from_rows.transform_point3(point) - expected).norm() < 1e-5);
        assert!((from_cols.transform_point3(point) - expected).norm() < 1e-5);

        let packed_point = Vector4::new(point.x, point.y, point.z, 1.0);
        assert!(
            (matrix4_from_row_major_array(&rows) * packed_point - matrix * packed_point).norm()
                < 1e-6
        );
        assert!(
            (matrix4_from_column_major_array(&cols) * packed_point - matrix * packed_point).norm()
                < 1e-6
        );
    }

    #[test]
    fn from_rotation_translation_keeps_row_major_convention() {
        let rotation = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let translation = [1.0, 2.0, 3.0];
        let pose = SE3::from_rotation_translation(&rotation, &translation);
        assert_eq!(pose.translation(), translation);
        let recovered = pose.rotation_matrix_row_major_array();
        for row in 0..3 {
            for col in 0..3 {
                assert!((recovered[row][col] - rotation[row][col]).abs() < 1e-5);
            }
        }
        let transformed = pose.transform_point_array(&[1.0, 0.0, 2.0]);
        assert!((transformed[0] - 1.0).abs() < 1e-6);
        assert!((transformed[1] - 3.0).abs() < 1e-6);
        assert!((transformed[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn xyzw_quaternion_round_trip_matches_point_transform() {
        let pose = non_symmetric_pose();
        let restored = SE3::new(&pose.quaternion(), &pose.translation());
        let point = Point3::new(-0.3, 1.4, 0.9);
        assert!((restored.transform_point3(point) - pose.transform_point3(point)).norm() < 1e-6);
    }

    #[test]
    fn exp_log_roundtrip() {
        let tangent = [0.1, 0.2, 0.3, 1.0, 2.0, 3.0];
        let pose = SE3::exp(&tangent);
        let log_tangent = pose.log();
        for i in 0..6 {
            assert!((tangent[i] - log_tangent[i]).abs() < 1e-4);
        }
    }

    /// SE3 serializes as an explicit object layout
    /// (`rotation: {x,y,z,w}`, `translation: {x,y,z}`).
    /// Legacy glam array forms (`[x,y,z,w]` / `[x,y,z]`) are intentionally
    /// unsupported — a breaking format change, not a compatibility layer.
    #[test]
    fn serde_uses_explicit_object_layout() {
        let pose = non_symmetric_pose();
        let json = serde_json::to_value(pose).unwrap();
        assert!(json["rotation"]["x"].is_number());
        assert!(json["translation"]["y"].is_number());
        let decoded: SE3 = serde_json::from_value(json).unwrap();
        let point = Point3::new(0.2, -0.8, 1.6);
        assert!((decoded.transform_point3(point) - pose.transform_point3(point)).norm() < 1e-6);
    }
}
