//! Shared CPU math aliases and quaternion/array conversions.
//!
//! Positions use `Point3*`. Directions and displacements use `Vec3*`.
//! Rotations use `UnitQuaternion`; `Quaternion` is only for raw component math.

use nalgebra::{
    Matrix3, Matrix4, Point2, Point3, Quaternion, Unit, UnitQuaternion, Vector2, Vector3, Vector4,
};

pub type Vec2f = Vector2<f32>;
pub type Vec3f = Vector3<f32>;
pub type Vec4f = Vector4<f32>;
pub type Vec2d = Vector2<f64>;
pub type Vec3d = Vector3<f64>;
pub type Vec4d = Vector4<f64>;

pub type Point2f = Point2<f32>;
pub type Point3f = Point3<f32>;
pub type Point2d = Point2<f64>;
pub type Point3d = Point3<f64>;

pub type Rotation3f = UnitQuaternion<f32>;
pub type Rotation3d = UnitQuaternion<f64>;

pub type Mat3f = Matrix3<f32>;
pub type Mat4f = Matrix4<f32>;
pub type Mat3d = Matrix3<f64>;
pub type Mat4d = Matrix4<f64>;

/// Build a unit quaternion from glam/JSON `[x, y, z, w]`.
pub fn rotation_from_xyzw(xyzw: [f32; 4]) -> Rotation3f {
    UnitQuaternion::new_normalize(Quaternion::new(xyzw[3], xyzw[0], xyzw[1], xyzw[2]))
}

/// Build a unit quaternion from COLMAP `[w, x, y, z]`.
pub fn rotation_from_wxyz(wxyz: [f32; 4]) -> Rotation3f {
    UnitQuaternion::new_normalize(Quaternion::new(wxyz[0], wxyz[1], wxyz[2], wxyz[3]))
}

/// Serialize a unit quaternion as `[x, y, z, w]`.
pub fn rotation_to_xyzw(rotation: Rotation3f) -> [f32; 4] {
    let q = rotation.into_inner();
    [q.i, q.j, q.k, q.w]
}

/// Serialize a unit quaternion as COLMAP `[w, x, y, z]`.
pub fn rotation_to_wxyz(rotation: Rotation3f) -> [f32; 4] {
    let q = rotation.into_inner();
    [q.w, q.i, q.j, q.k]
}

pub fn point3_from_array(coords: [f32; 3]) -> Point3f {
    Point3::new(coords[0], coords[1], coords[2])
}

pub fn vector3_from_array(coords: [f32; 3]) -> Vec3f {
    Vector3::new(coords[0], coords[1], coords[2])
}

pub fn point3_to_array(point: Point3f) -> [f32; 3] {
    [point.x, point.y, point.z]
}

pub fn vector3_to_array(vector: Vec3f) -> [f32; 3] {
    [vector.x, vector.y, vector.z]
}

pub fn matrix3_from_row_major_array(rows: &[[f32; 3]; 3]) -> Mat3f {
    Matrix3::from_fn(|row, col| rows[row][col])
}

pub fn matrix3_from_column_major_array(columns: &[[f32; 3]; 3]) -> Mat3f {
    Matrix3::from_fn(|row, col| columns[col][row])
}

pub fn matrix3_to_row_major_array(matrix: Mat3f) -> [[f32; 3]; 3] {
    std::array::from_fn(|row| std::array::from_fn(|col| matrix[(row, col)]))
}

pub fn matrix3_to_column_major_array(matrix: Mat3f) -> [[f32; 3]; 3] {
    std::array::from_fn(|col| std::array::from_fn(|row| matrix[(row, col)]))
}

pub fn matrix4_from_row_major_array(rows: &[[f32; 4]; 4]) -> Mat4f {
    Matrix4::from_fn(|row, col| rows[row][col])
}

pub fn matrix4_from_column_major_array(columns: &[[f32; 4]; 4]) -> Mat4f {
    Matrix4::from_fn(|row, col| columns[col][row])
}

pub fn matrix4_to_row_major_array(matrix: Mat4f) -> [[f32; 4]; 4] {
    std::array::from_fn(|row| std::array::from_fn(|col| matrix[(row, col)]))
}

pub fn matrix4_to_column_major_array(matrix: Mat4f) -> [[f32; 4]; 4] {
    std::array::from_fn(|col| std::array::from_fn(|row| matrix[(row, col)]))
}

pub fn rotation_from_scaled_axis(axis_angle: Vec3f) -> Rotation3f {
    let angle = axis_angle.norm();
    if angle > 1e-10 {
        UnitQuaternion::from_axis_angle(&Unit::new_normalize(axis_angle), angle)
    } else {
        UnitQuaternion::identity()
    }
}

pub fn rotation_from_axis_angle_x(angle: f32) -> Rotation3f {
    UnitQuaternion::from_axis_angle(&Vector3::x_axis(), angle)
}

pub fn rotation_from_axis_angle_y(angle: f32) -> Rotation3f {
    UnitQuaternion::from_axis_angle(&Vector3::y_axis(), angle)
}

pub fn rotation_from_axis_angle_z(angle: f32) -> Rotation3f {
    UnitQuaternion::from_axis_angle(&Vector3::z_axis(), angle)
}

pub fn rotation_from_matrix3(rotation: &Mat3f) -> Rotation3f {
    UnitQuaternion::from_matrix(rotation)
}

pub fn matrix3_from_rotation(rotation: Rotation3f) -> Mat3f {
    rotation.to_rotation_matrix().into_inner()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector4;

    fn non_symmetric_rotation() -> Rotation3f {
        UnitQuaternion::from_euler_angles(0.3, -0.4, 0.5)
    }

    #[test]
    fn xyzw_and_wxyz_round_trip_non_symmetric_rotation() {
        let rotation = non_symmetric_rotation();
        let xyzw = rotation_to_xyzw(rotation);
        let wxyz = rotation_to_wxyz(rotation);
        assert!((xyzw[3] - wxyz[0]).abs() < 1e-7);
        assert!((xyzw[0] - wxyz[1]).abs() < 1e-7);

        let from_xyzw = rotation_from_xyzw(xyzw);
        let from_wxyz = rotation_from_wxyz(wxyz);
        let point = Point3::new(0.7, -1.2, 2.4);
        let expected = rotation * point;
        assert!((from_xyzw * point - expected).norm() < 1e-6);
        assert!((from_wxyz * point - expected).norm() < 1e-6);
    }

    #[test]
    fn matrix4_row_and_column_major_arrays_round_trip() {
        let rotation = non_symmetric_rotation();
        let translation = Vector3::new(1.25, -2.5, 3.75);
        let mut rigid = rotation.to_homogeneous();
        rigid[(0, 3)] = translation.x;
        rigid[(1, 3)] = translation.y;
        rigid[(2, 3)] = translation.z;

        let columns = matrix4_to_column_major_array(rigid);
        let rows = matrix4_to_row_major_array(rigid);
        assert!((columns[3][0] - translation.x).abs() < 1e-7);
        assert!((rows[0][3] - translation.x).abs() < 1e-7);
        assert!((rows[0][1] - columns[1][0]).abs() < 1e-7);

        let from_cols = matrix4_from_column_major_array(&columns);
        let from_rows = matrix4_from_row_major_array(&rows);
        let point = Vector4::new(0.4, -1.1, 2.2, 1.0);
        let expected = rigid * point;
        assert!((from_cols * point - expected).norm() < 1e-6);
        assert!((from_rows * point - expected).norm() < 1e-6);
    }

    #[test]
    fn point_and_vector_arrays_round_trip() {
        let point = Point3::new(1.25, -2.5, 3.75);
        let vector = Vector3::new(0.3, -0.4, 0.5);
        assert_eq!(point3_from_array(point3_to_array(point)), point);
        assert_eq!(vector3_from_array(vector3_to_array(vector)), vector);
    }
}
