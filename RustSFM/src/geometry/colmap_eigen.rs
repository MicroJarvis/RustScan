use nalgebra::{DMatrix, DVector};

#[cfg(colmap_eigen)]
mod ffi {
    use std::os::raw::{c_double, c_int};

    extern "C" {
        pub fn rustsfm_eigen_right_nullspace_9(
            row_major: *const c_double,
            rows: usize,
            nullity: usize,
            output: *mut c_double,
        ) -> c_int;

        pub fn rustsfm_eigen_full_piv_right_nullspace_9(
            row_major: *const c_double,
            rows: usize,
            nullity: usize,
            output: *mut c_double,
        ) -> c_int;

        pub fn rustsfm_eigen_jacobi_svd_vt_9(
            row_major: *const c_double,
            rows: usize,
            output_vt: *mut c_double,
            output_singular_values: *mut c_double,
            output_num_singular_values: *mut usize,
        ) -> c_int;

        pub fn rustsfm_eigen_companion_roots(
            coeffs: *const c_double,
            len: usize,
            output_interleaved_complex: *mut c_double,
            output_num_roots: *mut usize,
        ) -> c_int;

        pub fn rustsfm_eigen_fundamental_seven_point(
            points1_xy: *const c_double,
            points2_xy: *const c_double,
            output_row_major_models: *mut c_double,
            output_num_models: *mut usize,
        ) -> c_int;

        pub fn rustsfm_eigen_partial_piv_lu_solve_10x10(
            lhs_row_major: *const c_double,
            rhs_row_major: *const c_double,
            output_row_major: *mut c_double,
        ) -> c_int;

        pub fn rustsfm_eigen_jacobi_svd_right_null_vector_3(
            matrix_row_major: *const c_double,
            output_vector: *mut c_double,
        ) -> c_int;
    }
}

pub fn right_nullspace_9(a: &DMatrix<f64>, nullity: usize) -> Option<Vec<[f64; 9]>> {
    if a.ncols() != 9 || a.nrows() == 0 || nullity == 0 || a.nrows() + nullity != 9 {
        return None;
    }

    #[cfg(colmap_eigen)]
    {
        let row_major = row_major_9(a)?;
        let mut output = vec![0.0f64; nullity * 9];
        let ok = unsafe {
            ffi::rustsfm_eigen_right_nullspace_9(
                row_major.as_ptr(),
                a.nrows(),
                nullity,
                output.as_mut_ptr(),
            )
        };
        if ok != 0 && output.iter().all(|value| value.is_finite()) {
            return Some(
                output
                    .chunks_exact(9)
                    .map(|chunk| {
                        [
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7], chunk[8],
                        ]
                    })
                    .collect(),
            );
        }
    }

    None
}

pub fn full_piv_right_nullspace_9(a: &DMatrix<f64>, nullity: usize) -> Option<Vec<[f64; 9]>> {
    if a.ncols() != 9 || a.nrows() == 0 || nullity == 0 || a.nrows() + nullity != 9 {
        return None;
    }

    #[cfg(colmap_eigen)]
    {
        let row_major = row_major_9(a)?;
        let mut output = vec![0.0f64; nullity * 9];
        let ok = unsafe {
            ffi::rustsfm_eigen_full_piv_right_nullspace_9(
                row_major.as_ptr(),
                a.nrows(),
                nullity,
                output.as_mut_ptr(),
            )
        };
        if ok != 0 && output.iter().all(|value| value.is_finite()) {
            return Some(
                output
                    .chunks_exact(9)
                    .map(|chunk| {
                        [
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7], chunk[8],
                        ]
                    })
                    .collect(),
            );
        }
    }

    None
}

pub fn jacobi_svd_vt_9(a: &DMatrix<f64>) -> Option<(DMatrix<f64>, DVector<f64>)> {
    if a.ncols() != 9 || a.nrows() == 0 {
        return None;
    }

    #[cfg(colmap_eigen)]
    {
        let row_major = row_major_9(a)?;
        let mut vt = [0.0f64; 81];
        let mut singular_values = [0.0f64; 9];
        let mut num_singular_values = 0usize;
        let ok = unsafe {
            ffi::rustsfm_eigen_jacobi_svd_vt_9(
                row_major.as_ptr(),
                a.nrows(),
                vt.as_mut_ptr(),
                singular_values.as_mut_ptr(),
                &mut num_singular_values,
            )
        };
        if ok != 0
            && (1..=9).contains(&num_singular_values)
            && vt.iter().all(|value| value.is_finite())
            && singular_values[..num_singular_values]
                .iter()
                .all(|value| value.is_finite())
        {
            return Some((
                DMatrix::from_row_slice(9, 9, &vt),
                DVector::from_row_slice(&singular_values[..num_singular_values]),
            ));
        }
    }

    None
}

pub fn companion_roots(coeffs: &[f64]) -> Option<Vec<(f64, f64)>> {
    if coeffs.len() < 3
        || coeffs[0].abs() < 1.0e-15
        || coeffs.iter().any(|value| !value.is_finite())
    {
        return None;
    }

    #[cfg(colmap_eigen)]
    {
        let degree = coeffs.len() - 1;
        let mut output = vec![0.0f64; degree * 2];
        let mut num_roots = 0usize;
        let ok = unsafe {
            ffi::rustsfm_eigen_companion_roots(
                coeffs.as_ptr(),
                coeffs.len(),
                output.as_mut_ptr(),
                &mut num_roots,
            )
        };
        if ok != 0 && num_roots == degree && output.iter().all(|value| value.is_finite()) {
            return Some(
                output
                    .chunks_exact(2)
                    .map(|chunk| (chunk[0], chunk[1]))
                    .collect(),
            );
        }
    }

    None
}

pub fn fundamental_seven_point(
    points1_xy: &[[f64; 2]; 7],
    points2_xy: &[[f64; 2]; 7],
) -> Option<Vec<[f64; 9]>> {
    if points1_xy
        .iter()
        .flatten()
        .chain(points2_xy.iter().flatten())
        .any(|value| !value.is_finite())
    {
        return None;
    }

    #[cfg(colmap_eigen)]
    {
        let mut flat1 = [0.0f64; 14];
        let mut flat2 = [0.0f64; 14];
        for idx in 0..7 {
            flat1[2 * idx] = points1_xy[idx][0];
            flat1[2 * idx + 1] = points1_xy[idx][1];
            flat2[2 * idx] = points2_xy[idx][0];
            flat2[2 * idx + 1] = points2_xy[idx][1];
        }
        let mut output = [0.0f64; 27];
        let mut num_models = 0usize;
        let ok = unsafe {
            ffi::rustsfm_eigen_fundamental_seven_point(
                flat1.as_ptr(),
                flat2.as_ptr(),
                output.as_mut_ptr(),
                &mut num_models,
            )
        };
        if ok != 0
            && num_models <= 3
            && output[..num_models * 9]
                .iter()
                .all(|value| value.is_finite())
        {
            return Some(
                output[..num_models * 9]
                    .chunks_exact(9)
                    .map(|chunk| {
                        [
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7], chunk[8],
                        ]
                    })
                    .collect(),
            );
        }
    }

    None
}

pub fn partial_piv_lu_solve_10x10(lhs: &DMatrix<f64>, rhs: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    if lhs.shape() != (10, 10)
        || rhs.shape() != (10, 10)
        || lhs.iter().chain(rhs.iter()).any(|value| !value.is_finite())
    {
        return None;
    }

    #[cfg(colmap_eigen)]
    {
        let lhs_row_major = row_major(lhs)?;
        let rhs_row_major = row_major(rhs)?;
        let mut output = [0.0f64; 100];
        let ok = unsafe {
            ffi::rustsfm_eigen_partial_piv_lu_solve_10x10(
                lhs_row_major.as_ptr(),
                rhs_row_major.as_ptr(),
                output.as_mut_ptr(),
            )
        };
        if ok != 0 && output.iter().all(|value| value.is_finite()) {
            return Some(DMatrix::from_row_slice(10, 10, &output));
        }
    }

    None
}

pub fn jacobi_svd_right_null_vector_3(matrix_row_major: &[f64; 9]) -> Option<[f64; 3]> {
    if matrix_row_major.iter().any(|value| !value.is_finite()) {
        return None;
    }

    #[cfg(colmap_eigen)]
    {
        let mut output = [0.0f64; 3];
        let ok = unsafe {
            ffi::rustsfm_eigen_jacobi_svd_right_null_vector_3(
                matrix_row_major.as_ptr(),
                output.as_mut_ptr(),
            )
        };
        if ok != 0 && output.iter().all(|value| value.is_finite()) {
            return Some(output);
        }
    }

    None
}

#[cfg(colmap_eigen)]
fn row_major_9(a: &DMatrix<f64>) -> Option<Vec<f64>> {
    if a.ncols() != 9 || a.iter().any(|value| !value.is_finite()) {
        return None;
    }
    row_major(a)
}

#[cfg(colmap_eigen)]
fn row_major(a: &DMatrix<f64>) -> Option<Vec<f64>> {
    if a.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut row_major = Vec::with_capacity(a.nrows() * a.ncols());
    for row in 0..a.nrows() {
        for col in 0..a.ncols() {
            row_major.push(a[(row, col)]);
        }
    }
    Some(row_major)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(colmap_eigen)]
    #[test]
    fn right_nullspace_9_returns_valid_basis_when_bridge_available() {
        let a = DMatrix::from_row_slice(
            8,
            9,
            &[
                1.0, 0.2, -0.1, 0.4, 0.5, -0.3, 0.7, -0.2, 0.1, //
                0.3, 1.1, 0.4, -0.6, 0.2, 0.8, -0.1, 0.5, 0.9, //
                -0.4, 0.6, 1.2, 0.1, -0.7, 0.3, 0.2, 0.8, -0.5, //
                0.9, -0.3, 0.2, 1.0, 0.4, -0.8, 0.6, 0.1, 0.7, //
                0.5, 0.7, -0.6, 0.2, 1.3, 0.4, -0.9, 0.3, 0.2, //
                -0.2, 0.4, 0.9, 0.7, -0.1, 1.1, 0.5, -0.6, 0.3, //
                0.8, -0.5, 0.1, -0.3, 0.6, 0.2, 1.4, 0.7, -0.4, //
                0.1, 0.9, -0.2, 0.5, -0.4, 0.6, 0.3, 1.2, 0.8,
            ],
        );
        let basis = right_nullspace_9(&a, 1).unwrap();
        let x = DVector::from_row_slice(&basis[0]);
        let residual = &a * x;
        assert!(residual.norm() < 1.0e-10, "nullspace residual {residual:?}");
    }

    #[cfg(colmap_eigen)]
    #[test]
    fn jacobi_svd_vt_9_returns_full_vt_when_bridge_available() {
        let a = DMatrix::from_fn(10, 9, |row, col| ((row + 1) * (col + 2)) as f64);
        let (vt, singular_values) = jacobi_svd_vt_9(&a).unwrap();
        assert_eq!(vt.shape(), (9, 9));
        assert_eq!(singular_values.len(), 9);
    }

    #[cfg(colmap_eigen)]
    #[test]
    fn companion_roots_returns_complex_roots_when_bridge_available() {
        let roots = companion_roots(&[1.0, 0.0, 1.0]).unwrap();
        assert_eq!(roots.len(), 2);
        assert!(roots.iter().all(|(re, _)| re.abs() < 1.0e-12));
        assert!(roots.iter().any(|(_, im)| (im - 1.0).abs() < 1.0e-12));
        assert!(roots.iter().any(|(_, im)| (im + 1.0).abs() < 1.0e-12));
    }
}
