use crate::five_point_generated;
use crate::polynomial;
use nalgebra::{DMatrix, Matrix3, SMatrix, SVector, Vector3};

pub type EssentialBasis = SMatrix<f64, 9, 4>;

pub fn estimate_five_point_essential(
    rays1: &[Vector3<f64>],
    rays2: &[Vector3<f64>],
) -> Vec<Matrix3<f64>> {
    let Some(basis) = five_point_nullspace(rays1, rays2) else {
        return Vec::new();
    };
    let basis_data = basis_as_colmap_data(&basis);
    let a_data = five_point_generated::build_elimination_matrix(&basis_data);
    let a = DMatrix::<f64>::from_column_slice(10, 20, &a_data);
    let left = a.view((0, 0), (10, 10)).into_owned();
    let right = a.view((0, 10), (10, 10)).into_owned();
    let Some(aa) = left.lu().solve(&right) else {
        return Vec::new();
    };

    let b_data = build_determinant_matrix_data(&aa);
    let coeffs = five_point_generated::determinant_coeffs(&b_data);
    let roots = polynomial::complex_roots_companion_matrix(&coeffs);

    let mut models = Vec::new();
    for root in roots {
        if root.im.abs() > 1.0e-10 {
            continue;
        }
        let z1 = root.re;
        let z2 = z1 * z1;
        let z3 = z2 * z1;
        let z4 = z3 * z1;
        let mut bz = Matrix3::<f64>::zeros();
        for j in 0..3 {
            bz[(j, 0)] = b_at(&b_data, 0, j) * z3
                + b_at(&b_data, 1, j) * z2
                + b_at(&b_data, 2, j) * z1
                + b_at(&b_data, 3, j);
            bz[(j, 1)] = b_at(&b_data, 4, j) * z3
                + b_at(&b_data, 5, j) * z2
                + b_at(&b_data, 6, j) * z1
                + b_at(&b_data, 7, j);
            bz[(j, 2)] = b_at(&b_data, 8, j) * z4
                + b_at(&b_data, 9, j) * z3
                + b_at(&b_data, 10, j) * z2
                + b_at(&b_data, 11, j) * z1
                + b_at(&b_data, 12, j);
        }
        let svd = bz.svd(false, true);
        let Some(vt) = svd.v_t else { continue };
        let x = vt.row(2).transpose();
        if x[2].abs() < 1.0e-10 {
            continue;
        }
        let e_vec = basis.column(0) * (x[0] / x[2])
            + basis.column(1) * (x[1] / x[2])
            + basis.column(2) * z1
            + basis.column(3);
        let norm = e_vec.norm();
        if !norm.is_finite() || norm <= 1.0e-12 {
            continue;
        }
        let e = vec9_to_matrix(e_vec / norm);
        if e.iter().all(|v| v.is_finite()) {
            models.push(e);
        }
    }
    models
}

pub fn five_point_nullspace(
    rays1: &[Vector3<f64>],
    rays2: &[Vector3<f64>],
) -> Option<EssentialBasis> {
    if rays1.len().min(rays2.len()) < 5 {
        return None;
    }
    let n = rays1.len().min(rays2.len());
    let mut rows = Vec::with_capacity(n * 9);
    for (x1, x2) in rays1.iter().zip(rays2.iter()) {
        rows.extend_from_slice(&[
            x2.x * x1.x,
            x2.x * x1.y,
            x2.x * x1.z,
            x2.y * x1.x,
            x2.y * x1.y,
            x2.y * x1.z,
            x2.z * x1.x,
            x2.z * x1.y,
            x2.z * x1.z,
        ]);
    }
    let q = DMatrix::<f64>::from_row_slice(n, 9, &rows);
    if n == 5 {
        five_point_minimal_nullspace(&q)
    } else {
        five_point_svd_nullspace(&q)
    }
}

fn five_point_minimal_nullspace(q: &DMatrix<f64>) -> Option<EssentialBasis> {
    if q.nrows() != 5 || q.ncols() != 9 {
        return None;
    }
    let qr = q.transpose().qr();
    let mut q_t = DMatrix::<f64>::identity(9, 9);
    qr.q_tr_mul(&mut q_t);
    let mut basis = EssentialBasis::zeros();
    for basis_col in 0..4 {
        for idx in 0..9 {
            basis[(idx, basis_col)] = q_t[(5 + basis_col, idx)];
        }
    }
    Some(basis)
}

fn five_point_svd_nullspace(q: &DMatrix<f64>) -> Option<EssentialBasis> {
    if q.ncols() != 9 || q.nrows() <= 5 {
        return None;
    }
    let svd = q.clone().svd(false, true);
    let vt = complete_right_singular_rows(svd.v_t?)?;
    let mut basis = EssentialBasis::zeros();
    for basis_col in 0..4 {
        for idx in 0..9 {
            basis[(idx, basis_col)] = vt[(5 + basis_col, idx)];
        }
    }
    Some(basis)
}

fn complete_right_singular_rows(vt: DMatrix<f64>) -> Option<DMatrix<f64>> {
    if vt.ncols() != 9 || vt.nrows() > 9 {
        return None;
    }
    if vt.nrows() == 9 {
        return Some(vt);
    }
    let qr = vt.transpose().qr();
    let mut q_t = DMatrix::<f64>::identity(9, 9);
    qr.q_tr_mul(&mut q_t);
    let mut full = DMatrix::<f64>::zeros(9, 9);
    for row in 0..vt.nrows() {
        for col in 0..9 {
            full[(row, col)] = vt[(row, col)];
        }
    }
    for row in vt.nrows()..9 {
        for col in 0..9 {
            full[(row, col)] = q_t[(row, col)];
        }
    }
    Some(full)
}

fn basis_as_colmap_data(basis: &EssentialBasis) -> [f64; 36] {
    let mut data = [0.0f64; 36];
    for col in 0..4 {
        for row in 0..9 {
            data[col * 9 + row] = basis[(row, col)];
        }
    }
    data
}

fn build_determinant_matrix_data(aa: &DMatrix<f64>) -> [f64; 39] {
    let mut b = [0.0f64; 39];
    for i in 0..3 {
        b_set(&mut b, 0, i, 0.0);
        b_set(&mut b, 4, i, 0.0);
        b_set(&mut b, 8, i, 0.0);

        for k in 0..3 {
            let v = aa[(i * 2 + 4, k)];
            b_set(&mut b, 1 + k, i, v);
        }
        for k in 0..3 {
            let v = aa[(i * 2 + 4, 3 + k)];
            b_set(&mut b, 5 + k, i, v);
        }
        for k in 0..4 {
            let v = aa[(i * 2 + 4, 6 + k)];
            b_set(&mut b, 9 + k, i, v);
        }
        for k in 0..3 {
            let v = b_at(&b, k, i) - aa[(i * 2 + 5, k)];
            b_set(&mut b, k, i, v);
        }
        for k in 0..3 {
            let row = 4 + k;
            let v = b_at(&b, row, i) - aa[(i * 2 + 5, 3 + k)];
            b_set(&mut b, row, i, v);
        }
        for k in 0..4 {
            let row = 8 + k;
            let v = b_at(&b, row, i) - aa[(i * 2 + 5, 6 + k)];
            b_set(&mut b, row, i, v);
        }
    }
    b
}

fn b_at(b: &[f64; 39], row: usize, col: usize) -> f64 {
    b[col * 13 + row]
}

fn b_set(b: &mut [f64; 39], row: usize, col: usize, value: f64) {
    b[col * 13 + row] = value;
}

fn vec9_to_matrix(v: SVector<f64, 9>) -> Matrix3<f64> {
    Matrix3::from_row_slice(&[v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn five_point_nullspace_contains_true_essential() {
        let rotation = Matrix3::<f64>::identity();
        let translation = Vector3::new(0.2, -0.03, 0.01).normalize();
        let essential = skew(translation) * rotation;
        let points = [
            Vector3::new(-0.4, -0.2, 3.0),
            Vector3::new(0.1, -0.3, 4.0),
            Vector3::new(0.5, -0.1, 3.5),
            Vector3::new(-0.2, 0.4, 4.2),
            Vector3::new(0.4, 0.3, 3.8),
        ];
        let rays1 = points.iter().map(|p| p / p.z).collect::<Vec<_>>();
        let rays2 = points
            .iter()
            .map(|p| {
                let q = rotation * p + translation;
                q / q.z
            })
            .collect::<Vec<_>>();
        let basis = five_point_nullspace(&rays1, &rays2).expect("basis");
        let e_vec = matrix_to_vec9(essential.normalize());
        let projection = basis * (basis.transpose() * e_vec);
        let residual = (e_vec - projection).norm();
        assert!(residual < 1.0e-8, "residual={residual}");
    }

    #[test]
    fn five_point_overdetermined_nullspace_contains_true_essential() {
        let rotation = nalgebra::Rotation3::from_euler_angles(0.02, -0.03, 0.04).into_inner();
        let translation = Vector3::new(0.12, -0.08, 0.03).normalize();
        let essential = skew(translation) * rotation;
        let points = [
            Vector3::new(-0.4, -0.2, 3.0),
            Vector3::new(0.1, -0.3, 4.0),
            Vector3::new(0.5, -0.1, 3.5),
            Vector3::new(-0.2, 0.4, 4.2),
            Vector3::new(0.4, 0.3, 3.8),
            Vector3::new(-0.1, 0.2, 5.1),
            Vector3::new(0.3, -0.4, 4.8),
            Vector3::new(-0.5, 0.1, 3.6),
        ];
        let rays1 = points.iter().map(|p| p / p.z).collect::<Vec<_>>();
        let rays2 = points
            .iter()
            .map(|p| {
                let q = rotation * p + translation;
                q / q.z
            })
            .collect::<Vec<_>>();
        let basis = five_point_nullspace(&rays1, &rays2).expect("basis");
        let e_vec = matrix_to_vec9(essential.normalize());
        let projection = basis * (basis.transpose() * e_vec);
        let residual = (e_vec - projection).norm();
        assert!(residual < 1.0e-8, "residual={residual}");
    }

    #[test]
    fn five_point_recovers_synthetic_essential() {
        let rotation = nalgebra::Rotation3::from_euler_angles(0.03, -0.04, 0.02).into_inner();
        let translation = Vector3::new(0.2, -0.03, 0.05).normalize();
        let essential = (skew(translation) * rotation).normalize();
        let points = [
            Vector3::new(-0.4, -0.2, 3.0),
            Vector3::new(0.1, -0.3, 4.0),
            Vector3::new(0.5, -0.1, 3.5),
            Vector3::new(-0.2, 0.4, 4.2),
            Vector3::new(0.4, 0.3, 3.8),
        ];
        let rays1 = points.iter().map(|p| p / p.z).collect::<Vec<_>>();
        let rays2 = points
            .iter()
            .map(|p| {
                let q = rotation * p + translation;
                q / q.z
            })
            .collect::<Vec<_>>();
        let models = estimate_five_point_essential(&rays1, &rays2);
        assert!(!models.is_empty(), "no models returned");
        let best = models
            .iter()
            .map(|model| essential_distance(*model, essential))
            .fold(f64::INFINITY, f64::min);
        assert!(
            best < 1.0e-5,
            "best distance={best}, models={}",
            models.len()
        );
    }

    #[test]
    fn five_point_root_filter_uses_colmap_imaginary_threshold() {
        let roots = polynomial::complex_roots_companion_matrix(&[1.0, 0.0, 1.0]);
        let accepted = roots
            .into_iter()
            .filter(|root| root.im.abs() <= 1.0e-10)
            .collect::<Vec<_>>();

        assert!(accepted.is_empty());
    }

    fn matrix_to_vec9(m: Matrix3<f64>) -> nalgebra::SVector<f64, 9> {
        nalgebra::SVector::<f64, 9>::from_row_slice(&[
            m[(0, 0)],
            m[(0, 1)],
            m[(0, 2)],
            m[(1, 0)],
            m[(1, 1)],
            m[(1, 2)],
            m[(2, 0)],
            m[(2, 1)],
            m[(2, 2)],
        ])
    }

    fn skew(t: Vector3<f64>) -> Matrix3<f64> {
        Matrix3::new(0.0, -t.z, t.y, t.z, 0.0, -t.x, -t.y, t.x, 0.0)
    }

    fn essential_distance(a: Matrix3<f64>, b: Matrix3<f64>) -> f64 {
        let a = a.normalize();
        let b = b.normalize();
        (a - b).norm().min((a + b).norm())
    }
}
