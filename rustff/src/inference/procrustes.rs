//! Weighted Procrustes alignment for pose estimation from pointmaps.
//!
//! Given predicted 3D pointmaps and confidence weights, estimate the
//! optimal rigid body transformation (rotation + translation).
//!
//! Reference: Python implementation in scripts/procrustes.py

use nalgebra::{Matrix3, Matrix4, Vector3};

/// Estimate rigid body transformation from source to target point cloud.
///
/// Returns (R, t) where R is rotation [3x3] and t is translation [3].
pub fn weighted_procrustes(
    src: &[[f32; 3]],
    tgt: &[[f32; 3]],
    weights: &[f32],
) -> (Matrix3<f32>, Vector3<f32>) {
    assert_eq!(src.len(), tgt.len());
    assert_eq!(src.len(), weights.len());
    assert!(!src.is_empty());

    let w_sum: f32 = weights.iter().sum();
    let w_norm: Vec<f32> = weights.iter().map(|w| w / w_sum).collect();

    // Weighted centroids
    let mut src_centroid = [0.0f32; 3];
    let mut tgt_centroid = [0.0f32; 3];
    for i in 0..src.len() {
        for j in 0..3 {
            src_centroid[j] += w_norm[i] * src[i][j];
            tgt_centroid[j] += w_norm[i] * tgt[i][j];
        }
    }

    // Centered points
    let src_centered: Vec<[f32; 3]> = src
        .iter()
        .map(|p| {
            [
                p[0] - src_centroid[0],
                p[1] - src_centroid[1],
                p[2] - src_centroid[2],
            ]
        })
        .collect();
    let tgt_centered: Vec<[f32; 3]> = tgt
        .iter()
        .map(|p| {
            [
                p[0] - tgt_centroid[0],
                p[1] - tgt_centroid[1],
                p[2] - tgt_centroid[2],
            ]
        })
        .collect();

    // Weighted cross-covariance H = sum src * tgt^T, stored row-major.
    let mut h = [[0.0f32; 3]; 3];
    for i in 0..src.len() {
        for r in 0..3 {
            for c in 0..3 {
                h[r][c] += w_norm[i] * src_centered[i][r] * tgt_centered[i][c];
            }
        }
    }

    let h_mat = Matrix3::from_fn(|row, col| h[row][col]);
    let svd = nalgebra::SVD::new(h_mat, true, true);
    let u = svd.u.expect("3x3 SVD must produce U");
    let v_t = svd.v_t.expect("3x3 SVD must produce V^T");
    let mut r = &v_t.transpose() * u.transpose();
    if r.determinant() < 0.0 {
        let mut v = v_t.transpose();
        v.set_column(2, &(-v.column(2)));
        r = v * u.transpose();
    }

    // Translation: t = tgt_centroid - R * src_centroid
    let src_centroid = Vector3::from(src_centroid);
    let tgt_centroid = Vector3::from(tgt_centroid);
    let t = tgt_centroid - r * src_centroid;

    (r, t)
}

/// Extract camera pose from pointmap via weighted Procrustes.
///
/// Given a predicted pointmap and confidence, estimate the camera-to-world pose.
pub fn pointmap_to_pose(points: &[[f32; 3]], confidence: &[f32]) -> Matrix4<f32> {
    // Filter by confidence
    let valid: Vec<(&[f32; 3], f32)> = points
        .iter()
        .zip(confidence.iter())
        .filter(|(_, &c)| c > 0.1)
        .map(|(p, &c)| (p, c))
        .collect();

    if valid.len() < 10 {
        return Matrix4::identity();
    }

    // Weighted centroid
    let w_sum: f32 = valid.iter().map(|(_, c)| *c).sum();
    let mut centroid = [0.0f32; 3];
    for (p, c) in valid.iter() {
        for j in 0..3 {
            centroid[j] += *c * p[j] / w_sum;
        }
    }

    // Compute covariance for principal axes
    let mut cov = [[0.0f32; 3]; 3];
    for (p, c) in valid.iter() {
        let d = [p[0] - centroid[0], p[1] - centroid[1], p[2] - centroid[2]];
        for r in 0..3 {
            for col in 0..3 {
                cov[r][col] += *c * d[r] * d[col];
            }
        }
    }

    // Eigendecomposition (power iteration for dominant eigenvector)
    let (_evals, evecs) = eigendecompose_3x3(&cov);

    // Build rotation from eigenvectors (column vectors)
    let r = Matrix3::from_fn(|row, col| evecs[row][col]);

    Matrix4::new(
        r[(0, 0)],
        r[(0, 1)],
        r[(0, 2)],
        centroid[0],
        r[(1, 0)],
        r[(1, 1)],
        r[(1, 2)],
        centroid[1],
        r[(2, 0)],
        r[(2, 1)],
        r[(2, 2)],
        centroid[2],
        0.0,
        0.0,
        0.0,
        1.0,
    )
}

// ============================================================
// Linear algebra helpers
// ============================================================

/// 3x3 eigendecomposition using Jacobi rotations
fn eigendecompose_3x3(m: &[[f32; 3]; 3]) -> ([f32; 3], [[f32; 3]; 3]) {
    let mut a = *m;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    for _ in 0..50 {
        // Find largest off-diagonal
        let mut max_val = 0.0f32;
        let mut p = 0;
        let mut q = 1;
        for i in 0..3 {
            for j in (i + 1)..3 {
                if a[i][j].abs() > max_val {
                    max_val = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-7 {
            break;
        }

        // Jacobi rotation
        let theta = 0.5 * (2.0 * a[p][q]).atan2(a[p][p] - a[q][q]);
        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation to a
        let mut new_a = a;
        for i in 0..3 {
            new_a[i][p] = c * a[i][p] + s * a[i][q];
            new_a[i][q] = -s * a[i][p] + c * a[i][q];
        }
        let mut new_a2 = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                new_a2[p][j] = c * new_a[p][j] + s * new_a[q][j];
                new_a2[q][j] = -s * new_a[p][j] + c * new_a[q][j];
                if i != p && i != q {
                    new_a2[i][j] = new_a[i][j];
                }
            }
        }
        a = new_a2;

        // Update eigenvectors
        for i in 0..3 {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = c * vip + s * viq;
            v[i][q] = -s * vip + c * viq;
        }
    }

    ([a[0][0], a[1][1], a[2][2]], v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;

    #[test]
    fn test_procrustes_identity() {
        let points = vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        let weights = vec![1.0; 4];

        // Same points -> R = I, t = 0
        let (r, t) = weighted_procrustes(&points, &points, &weights);

        assert!(
            (r - Matrix3::identity()).norm() < 0.01,
            "Rotation should be identity, diff={}",
            (r - Matrix3::identity()).norm()
        );
        assert!(t.norm() < 0.01, "Translation should be zero, t={t:?}");
    }

    #[test]
    fn test_procrustes_translation() {
        let src = vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        let offset = [2.0, 3.0, 4.0];
        let tgt: Vec<[f32; 3]> = src
            .iter()
            .map(|p| [p[0] + offset[0], p[1] + offset[1], p[2] + offset[2]])
            .collect();
        let weights = vec![1.0; 4];

        let (r, t) = weighted_procrustes(&src, &tgt, &weights);

        assert!(
            (r - Matrix3::identity()).norm() < 0.01,
            "Rotation should be identity, diff={}",
            (r - Matrix3::identity()).norm()
        );

        let t_err = (t - Vector3::from(offset)).norm();
        assert!(
            t_err < 0.01,
            "Translation should be {offset:?}, err={t_err}"
        );
    }

    #[test]
    fn test_procrustes_non_symmetric_rotation_and_translation() {
        let rotation = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let translation = Vector3::new(1.25, -2.5, 3.75);
        let src_points = [
            Point3::new(0.7, -1.1, 2.3),
            Point3::new(-0.4, 1.8, 0.2),
            Point3::new(1.5, 0.3, -0.9),
            Point3::new(0.1, -0.6, 1.4),
        ];
        let src: Vec<[f32; 3]> = src_points.iter().map(|p| [p.x, p.y, p.z]).collect();
        let tgt: Vec<[f32; 3]> = src_points
            .iter()
            .map(|p| {
                let q = rotation * p.coords + translation;
                [q.x, q.y, q.z]
            })
            .collect();
        let weights = vec![1.0; 4];

        let (r, t) = weighted_procrustes(&src, &tgt, &weights);
        assert!(
            (r - rotation).norm() < 0.05,
            "rotation err={}",
            (r - rotation).norm()
        );
        assert!(
            (t - translation).norm() < 0.05,
            "translation err={}",
            (t - translation).norm()
        );

        let pose = Matrix4::new(
            r[(0, 0)],
            r[(0, 1)],
            r[(0, 2)],
            t.x,
            r[(1, 0)],
            r[(1, 1)],
            r[(1, 2)],
            t.y,
            r[(2, 0)],
            r[(2, 1)],
            r[(2, 2)],
            t.z,
            0.0,
            0.0,
            0.0,
            1.0,
        );
        let p = nalgebra::Vector4::new(0.7, -1.1, 2.3, 1.0);
        let expected = {
            let q = rotation * Vector3::new(0.7, -1.1, 2.3) + translation;
            nalgebra::Vector4::new(q.x, q.y, q.z, 1.0)
        };
        assert!((pose * p - expected).norm() < 0.05);
    }
}
