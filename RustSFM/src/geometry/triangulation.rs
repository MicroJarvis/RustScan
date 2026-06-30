//! Faithful Rust port of COLMAP's `geometry/triangulation` module.
//!
//! This module reproduces COLMAP's `src/colmap/geometry/triangulation.{h,cc}`
//! triangulation primitives (plus the two essential-matrix helpers from
//! `geometry/essential_matrix.cc` that `TriangulateOptimalPoint` depends on),
//! using `f64` throughout to match COLMAP's `double` precision.
//!
//! Conventions mirror COLMAP exactly:
//! - `cam_from_world` is the `3x4` camera pose matrix `[R | t]` mapping world
//!   points into the camera frame (`x_cam = R * X_world + t`).
//! - `cam_point` / `cam_ray` are normalized image-plane observations, i.e.
//!   homogeneous bearing coordinates `(x, y)` with implicit `z = 1`.
//! - The returned 3D point is expressed in world coordinates (or, for
//!   [`triangulate_mid_point`], in the first camera frame, matching COLMAP).

use nalgebra::{
    Matrix2, Matrix2x3, Matrix3, Matrix3x4, Matrix4, SymmetricEigen, Vector2, Vector3, Vector4,
};

/// Eigen `Vector4d::hnormalized()`: divide by the last component and drop it.
#[inline]
fn hnormalized4(v: &Vector4<f64>) -> Vector3<f64> {
    Vector3::new(v[0] / v[3], v[1] / v[3], v[2] / v[3])
}

/// Eigen `Vector3d::hnormalized()`: divide by the last component and drop it.
#[inline]
fn hnormalized3(v: &Vector3<f64>) -> Vector2<f64> {
    Vector2::new(v[0] / v[2], v[1] / v[2])
}

/// COLMAP `CrossProductMatrix`: the skew-symmetric matrix `[v]_x`.
#[inline]
fn cross_product_matrix(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -v[2], v[1], //
        v[2], 0.0, -v[0], //
        -v[1], v[0], 0.0, //
    )
}

/// COLMAP `EssentialMatrixFromPose`:
/// `E = [t_normalized]_x * R` for the relative pose `cam2_from_cam1`.
fn essential_matrix_from_pose(
    cam2_from_cam1_rotation: &Matrix3<f64>,
    cam2_from_cam1_translation: &Vector3<f64>,
) -> Matrix3<f64> {
    cross_product_matrix(&cam2_from_cam1_translation.normalize()) * cam2_from_cam1_rotation
}

/// COLMAP `FindOptimalImageObservations` (Lindstrom, "Triangulation Made Easy",
/// CVPR 2010, single-iteration `niter1`). Corrects the two observations so they
/// exactly satisfy the epipolar constraint of `E` with minimal geometric shift.
fn find_optimal_image_observations(
    e: &Matrix3<f64>,
    point1: &Vector2<f64>,
    point2: &Vector2<f64>,
) -> (Vector2<f64>, Vector2<f64>) {
    let point1_homogeneous = Vector3::new(point1[0], point1[1], 1.0);
    let point2_homogeneous = Vector3::new(point2[0], point2[1], 1.0);

    let s = Matrix2x3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    // Epipolar lines.
    let mut n1: Vector2<f64> = s * e * point2_homogeneous;
    let mut n2: Vector2<f64> = s * e.transpose() * point1_homogeneous;

    let e_tilde: Matrix2<f64> = e.fixed_view::<2, 2>(0, 0).into_owned();

    let a = (n1.transpose() * e_tilde * n2)[(0, 0)];
    let b = (n1.norm_squared() + n2.norm_squared()) / 2.0;
    let c = (point1_homogeneous.transpose() * e * point2_homogeneous)[(0, 0)];
    let d = (b * b - a * c).sqrt();
    let mut lambda = c / (b + d);

    let delta1 = lambda * n1;
    let delta2 = lambda * n2;

    n1 -= e_tilde * delta2;
    n2 -= e_tilde.transpose() * delta1;

    lambda *= (2.0 * d) / (n1.norm_squared() + n2.norm_squared());

    let optimal_point1 = hnormalized3(&(point1_homogeneous - s.transpose() * (lambda * n1)));
    let optimal_point2 = hnormalized3(&(point2_homogeneous - s.transpose() * (lambda * n2)));
    (optimal_point1, optimal_point2)
}

/// COLMAP `TriangulatePoint`: two-view DLT triangulation.
///
/// Returns `None` when the homogeneous solution is at infinity (last component
/// is exactly zero), matching COLMAP's `svd.matrixV()(3, 3) == 0` guard.
pub fn triangulate_point(
    cam1_from_world: &Matrix3x4<f64>,
    cam2_from_world: &Matrix3x4<f64>,
    cam_point1: &Vector2<f64>,
    cam_point2: &Vector2<f64>,
) -> Option<Vector3<f64>> {
    let mut rows = [0.0f64; 16];
    for col in 0..4 {
        rows[col] = cam_point1[0] * cam1_from_world[(2, col)] - cam1_from_world[(0, col)];
        rows[4 + col] = cam_point1[1] * cam1_from_world[(2, col)] - cam1_from_world[(1, col)];
        rows[8 + col] = cam_point2[0] * cam2_from_world[(2, col)] - cam2_from_world[(0, col)];
        rows[12 + col] = cam_point2[1] * cam2_from_world[(2, col)] - cam2_from_world[(1, col)];
    }
    let a = Matrix4::<f64>::from_row_slice(&rows);

    let svd = a.svd(false, true);
    let v_t = svd.v_t?;
    // Eigen `matrixV().col(3)` is the right-singular vector of the smallest
    // singular value; nalgebra sorts singular values descending, so that is the
    // last row of `V^T`.
    let v_col3 = v_t.row(3);
    if v_col3[3] == 0.0 {
        return None;
    }
    Some(hnormalized4(&Vector4::new(
        v_col3[0], v_col3[1], v_col3[2], v_col3[3],
    )))
}

/// COLMAP `TriangulateMidPoint`: midpoint triangulation expressed in the first
/// camera frame, with a cheirality check on both ray scales.
pub fn triangulate_mid_point(
    cam2_from_cam1_rotation: &Matrix3<f64>,
    cam2_from_cam1_translation: &Vector3<f64>,
    cam_ray1: &Vector3<f64>,
    cam_ray2: &Vector3<f64>,
) -> Option<Vector3<f64>> {
    // `cam1_from_cam2_rotation = R^T`.
    let cam1_from_cam2_rotation = cam2_from_cam1_rotation.transpose();
    let cam_ray2_in_cam1 = cam1_from_cam2_rotation * cam_ray2;
    let cam2_in_cam1 = cam1_from_cam2_rotation * (-cam2_from_cam1_translation);

    let a = Matrix3::from_columns(&[*cam_ray1, -cam_ray2_in_cam1, -cam2_in_cam1]);

    let svd = a.svd(false, true);
    let v_t = svd.v_t?;
    // Eigen `matrixV().col(2)` -> last row of `V^T`.
    let v_col2 = v_t.row(2);
    if v_col2[2] == 0.0 {
        return None;
    }
    let lambda = Vector2::new(v_col2[0] / v_col2[2], v_col2[1] / v_col2[2]);

    // Check if point is behind cameras.
    if lambda[0] <= f64::EPSILON || lambda[1] <= f64::EPSILON {
        return None;
    }

    Some(0.5 * (lambda[0] * cam_ray1 + cam2_in_cam1 + lambda[1] * cam_ray2_in_cam1))
}

/// COLMAP `TriangulateMultiViewPoint`: multi-view DLT via the smallest
/// eigenvector of the accumulated normal-equation matrix.
pub fn triangulate_multi_view_point(
    cams_from_world: &[Matrix3x4<f64>],
    cam_points: &[Vector2<f64>],
) -> Option<Vector3<f64>> {
    if cams_from_world.len() != cam_points.len() || cam_points.is_empty() {
        return None;
    }

    let mut a = Matrix4::<f64>::zeros();
    for i in 0..cam_points.len() {
        let point = Vector3::new(cam_points[i][0], cam_points[i][1], 1.0).normalize();
        let term = cams_from_world[i] - point * point.transpose() * cams_from_world[i];
        a += term.transpose() * term;
    }

    let eigen = SymmetricEigen::new(a);
    // Eigen's `SelfAdjointEigenSolver` sorts eigenvalues ascending and uses
    // `eigenvectors().col(0)` (smallest). nalgebra does not guarantee an
    // ordering, so select the smallest-eigenvalue column explicitly.
    let mut min_index = 0;
    let mut min_value = eigen.eigenvalues[0];
    for k in 1..4 {
        if eigen.eigenvalues[k] < min_value {
            min_value = eigen.eigenvalues[k];
            min_index = k;
        }
    }
    let v = eigen.eigenvectors.column(min_index);
    if v[3] == 0.0 {
        return None;
    }
    Some(Vector3::new(v[0] / v[3], v[1] / v[3], v[2] / v[3]))
}

/// COLMAP `TriangulateOptimalPoint`: corrects the observations to the optimal
/// epipolar-consistent positions and then triangulates with the DLT.
pub fn triangulate_optimal_point(
    cam1_from_world: &Matrix3x4<f64>,
    cam2_from_world: &Matrix3x4<f64>,
    cam_point1: &Vector2<f64>,
    cam_point2: &Vector2<f64>,
) -> Option<Vector3<f64>> {
    let r1: Matrix3<f64> = cam1_from_world.fixed_view::<3, 3>(0, 0).into_owned();
    let t1: Vector3<f64> = cam1_from_world.column(3).into_owned();
    let r2: Matrix3<f64> = cam2_from_world.fixed_view::<3, 3>(0, 0).into_owned();
    let t2: Vector3<f64> = cam2_from_world.column(3).into_owned();

    // cam2_from_cam1 = cam2_from_world * Inverse(cam1_from_world).
    let cam2_from_cam1_rotation = r2 * r1.transpose();
    let cam2_from_cam1_translation = t2 - cam2_from_cam1_rotation * t1;

    let e = essential_matrix_from_pose(&cam2_from_cam1_rotation, &cam2_from_cam1_translation);

    // `essential_matrix_from_pose` returns the standard essential matrix `E`
    // with `point2^T E point1 = 0`. `find_optimal_image_observations` (the
    // Lindstrom formulas) is written for the transposed convention
    // `point1^T E point2 = 0` (`n1 = S E point2`, `c = point1^T E point2`), i.e.
    // the essential matrix of the reverse pose. Pass `E^T` so the corrected
    // observations satisfy the true epipolar constraint.
    let (optimal_point1, optimal_point2) =
        find_optimal_image_observations(&e.transpose(), cam_point1, cam_point2);

    triangulate_point(
        cam1_from_world,
        cam2_from_world,
        &optimal_point1,
        &optimal_point2,
    )
}

/// COLMAP `CalculateAngleBetweenVectors`: clamped `acos` of the normalized dot
/// product, returning `0` for degenerate (zero-length) inputs.
pub fn calculate_angle_between_vectors(v1: &Vector3<f64>, v2: &Vector3<f64>) -> f64 {
    let squared_norm1 = v1.norm_squared();
    let squared_norm2 = v2.norm_squared();
    if squared_norm1 == 0.0 || squared_norm2 == 0.0 {
        return 0.0;
    }
    (v1.dot(v2) / (squared_norm1 * squared_norm2).sqrt())
        .clamp(-1.0, 1.0)
        .acos()
}

/// COLMAP `CalculateTriangulationAngle`: the minimum of the enclosing ray angle
/// and its supplement (triangulation is unstable for acute/obtuse angles).
pub fn calculate_triangulation_angle(
    proj_center1: &Vector3<f64>,
    proj_center2: &Vector3<f64>,
    point3d: &Vector3<f64>,
) -> f64 {
    let angle =
        calculate_angle_between_vectors(&(point3d - proj_center1), &(point3d - proj_center2));
    angle.min(std::f64::consts::PI - angle)
}

/// COLMAP `CalculateTriangulationAngles`: batched [`calculate_triangulation_angle`].
pub fn calculate_triangulation_angles(
    proj_center1: &Vector3<f64>,
    proj_center2: &Vector3<f64>,
    points3d: &[Vector3<f64>],
) -> Vec<f64> {
    points3d
        .iter()
        .map(|point3d| calculate_triangulation_angle(proj_center1, proj_center2, point3d))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, Unit};

    /// Build a `[R | t]` camera pose matrix from a rotation and camera center.
    fn pose_from_center(rotation: &Matrix3<f64>, center: &Vector3<f64>) -> Matrix3x4<f64> {
        let t = -(rotation * center);
        let mut m = Matrix3x4::<f64>::zeros();
        m.fixed_view_mut::<3, 3>(0, 0).copy_from(rotation);
        m.column_mut(3).copy_from(&t);
        m
    }

    /// Project a world point into normalized image coordinates `(x/z, y/z)`.
    fn project(cam_from_world: &Matrix3x4<f64>, point: &Vector3<f64>) -> Vector2<f64> {
        // NOTE: nalgebra's `Vector3::to_homogeneous` appends `0` (direction
        // semantics), so build the position-homogeneous vector explicitly.
        let point_h = Vector4::new(point[0], point[1], point[2], 1.0);
        let p = cam_from_world * point_h;
        Vector2::new(p[0] / p[2], p[1] / p[2])
    }

    fn rot(axis: Vector3<f64>, angle: f64) -> Matrix3<f64> {
        Rotation3::from_axis_angle(&Unit::new_normalize(axis), angle).into_inner()
    }

    #[test]
    fn triangulate_point_recovers_exact_point_noise_free() {
        let cam1 = pose_from_center(&Matrix3::identity(), &Vector3::new(0.0, 0.0, 0.0));
        let cam2 = pose_from_center(
            &rot(Vector3::new(0.0, 1.0, 0.0), 0.2),
            &Vector3::new(1.0, 0.0, 0.0),
        );
        let truth = Vector3::new(0.3, -0.4, 5.0);
        let p1 = project(&cam1, &truth);
        let p2 = project(&cam2, &truth);

        let xyz = triangulate_point(&cam1, &cam2, &p1, &p2).expect("triangulation");
        assert!((xyz - truth).norm() < 1e-9, "{xyz:?} vs {truth:?}");
    }

    #[test]
    fn triangulate_multi_view_point_recovers_point_from_three_views() {
        let cams = [
            pose_from_center(&Matrix3::identity(), &Vector3::new(0.0, 0.0, 0.0)),
            pose_from_center(
                &rot(Vector3::new(0.0, 1.0, 0.0), 0.25),
                &Vector3::new(1.0, 0.0, 0.0),
            ),
            pose_from_center(
                &rot(Vector3::new(1.0, 0.0, 0.0), -0.15),
                &Vector3::new(0.2, 1.0, -0.3),
            ),
        ];
        let truth = Vector3::new(-0.5, 0.6, 4.0);
        let points: Vec<Vector2<f64>> = cams.iter().map(|c| project(c, &truth)).collect();

        let xyz = triangulate_multi_view_point(&cams, &points).expect("triangulation");
        assert!((xyz - truth).norm() < 1e-9, "{xyz:?} vs {truth:?}");
    }

    #[test]
    fn triangulate_multi_view_point_matches_two_view_for_two_observations() {
        let cam1 = pose_from_center(&Matrix3::identity(), &Vector3::new(0.0, 0.0, 0.0));
        let cam2 = pose_from_center(
            &rot(Vector3::new(0.1, 1.0, 0.0), 0.3),
            &Vector3::new(0.8, 0.1, 0.0),
        );
        let truth = Vector3::new(0.2, 0.1, 6.0);
        let p1 = project(&cam1, &truth);
        let p2 = project(&cam2, &truth);

        let two = triangulate_point(&cam1, &cam2, &p1, &p2).unwrap();
        let multi = triangulate_multi_view_point(&[cam1, cam2], &[p1, p2]).unwrap();
        assert!((two - multi).norm() < 1e-7, "{two:?} vs {multi:?}");
    }

    #[test]
    fn triangulate_optimal_point_equals_dlt_when_noise_free() {
        let cam1 = pose_from_center(&Matrix3::identity(), &Vector3::new(0.0, 0.0, 0.0));
        let cam2 = pose_from_center(
            &rot(Vector3::new(0.0, 1.0, 0.0), 0.2),
            &Vector3::new(1.0, 0.0, 0.0),
        );
        let truth = Vector3::new(0.3, -0.4, 5.0);
        let p1 = project(&cam1, &truth);
        let p2 = project(&cam2, &truth);

        let optimal = triangulate_optimal_point(&cam1, &cam2, &p1, &p2).expect("triangulation");
        assert!((optimal - truth).norm() < 1e-7, "{optimal:?} vs {truth:?}");
    }

    #[test]
    fn triangulate_optimal_point_beats_dlt_under_noise() {
        let cam1 = pose_from_center(&Matrix3::identity(), &Vector3::new(0.0, 0.0, 0.0));
        let cam2 = pose_from_center(
            &rot(Vector3::new(0.0, 1.0, 0.0), 0.35),
            &Vector3::new(1.0, 0.0, 0.0),
        );
        let truth = Vector3::new(0.4, 0.2, 4.0);
        let mut p1 = project(&cam1, &truth);
        let mut p2 = project(&cam2, &truth);
        // Add asymmetric observation noise.
        p1 += Vector2::new(0.004, -0.003);
        p2 += Vector2::new(-0.0035, 0.0045);

        let dlt = triangulate_point(&cam1, &cam2, &p1, &p2).unwrap();
        let optimal = triangulate_optimal_point(&cam1, &cam2, &p1, &p2).unwrap();
        assert!(
            (optimal - truth).norm() <= (dlt - truth).norm() + 1e-9,
            "optimal {optimal:?} should be at least as accurate as dlt {dlt:?}",
        );
    }

    #[test]
    fn triangulate_mid_point_recovers_point_in_first_camera_frame() {
        let cam2_from_cam1_rotation = rot(Vector3::new(0.0, 1.0, 0.0), 0.3);
        let cam2_from_cam1_translation = Vector3::new(-1.0, 0.05, 0.0);
        let truth_in_cam1 = Vector3::new(0.25, -0.1, 5.0);
        let point_in_cam2 = cam2_from_cam1_rotation * truth_in_cam1 + cam2_from_cam1_translation;

        let ray1 = truth_in_cam1; // bearing (z not necessarily 1, direction matters)
        let ray2 = point_in_cam2;

        let xyz = triangulate_mid_point(
            &cam2_from_cam1_rotation,
            &cam2_from_cam1_translation,
            &ray1,
            &ray2,
        )
        .expect("triangulation");
        assert!((xyz - truth_in_cam1).norm() < 1e-9, "{xyz:?}");
    }

    #[test]
    fn triangulation_angle_uses_min_of_angle_and_supplement() {
        // Two cameras symmetric about a point: 90-degree enclosed angle.
        let c1 = Vector3::new(-1.0, 0.0, 0.0);
        let c2 = Vector3::new(1.0, 0.0, 0.0);
        let point = Vector3::new(0.0, 0.0, 1.0);
        let angle = calculate_triangulation_angle(&c1, &c2, &point);
        assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 1e-12);

        // A very wide (near 180-degree) configuration collapses to its small
        // supplement.
        let far = Vector3::new(0.0, 0.0, 0.001);
        let wide = calculate_triangulation_angle(&c1, &c2, &far);
        assert!(wide < std::f64::consts::FRAC_PI_2);
    }

    #[test]
    fn angle_between_vectors_handles_degenerate_and_clamped_inputs() {
        assert_eq!(
            calculate_angle_between_vectors(&Vector3::zeros(), &Vector3::new(1.0, 0.0, 0.0)),
            0.0
        );
        // Parallel vectors -> 0 angle even with clamping at +1.
        let v = Vector3::new(2.0, 0.0, 0.0);
        assert!(calculate_angle_between_vectors(&v, &v).abs() < 1e-12);
        // Anti-parallel vectors -> PI angle (clamped at -1).
        let anti = calculate_angle_between_vectors(&v, &(-v));
        assert!((anti - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn triangulation_angles_batch_matches_scalar() {
        let c1 = Vector3::new(0.0, 0.0, 0.0);
        let c2 = Vector3::new(2.0, 0.0, 0.0);
        let points = [
            Vector3::new(0.5, 0.0, 3.0),
            Vector3::new(1.0, 1.0, 4.0),
            Vector3::new(-0.5, 0.2, 2.0),
        ];
        let batch = calculate_triangulation_angles(&c1, &c2, &points);
        for (i, point) in points.iter().enumerate() {
            assert_eq!(batch[i], calculate_triangulation_angle(&c1, &c2, point));
        }
    }
}
