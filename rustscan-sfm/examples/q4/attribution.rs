//! Offline-only arithmetic attribution; production/reference library is unchanged.
#[allow(dead_code)]
#[path = "../../src/geometry/colmap_eigen.rs"]
mod eigen;
#[allow(dead_code)]
#[path = "../../src/geometry/five_point_generated.rs"]
mod generated;
use nalgebra::DMatrix;
use rustscan_sfm::gpu::FivePointAlgebraResult;
use serde_json::{json, Value};

pub const NAMES: [&str; 8] = [
    "A_gpu_vs_f64_same_basis",
    "solve_gpu_vs_f64_same_A",
    "solve_gpu_vs_f64_same_basis",
    "B_gpu_vs_f64_same_solve",
    "B_gpu_vs_f64_same_basis",
    "polynomial_gpu_vs_f64_same_B",
    "polynomial_f64_gpu_B_vs_f64_basis",
    "polynomial_total",
];
fn error(a: &[f64], b: &[f64]) -> f64 {
    let scale = b.iter().fold(0.0f64, |s, x| s.max(x.abs()));
    if scale == 0.0 || a.iter().chain(b).any(|x| !x.is_finite()) {
        return f64::NAN;
    }
    a.iter()
        .zip(b)
        .fold(0.0f64, |s, (a, b)| s.max((a - b).abs()))
        / scale
}
fn solve(a: &[f64; 200]) -> Option<DMatrix<f64>> {
    let a = DMatrix::from_column_slice(10, 20, a);
    let left = a.view((0, 0), (10, 10)).into_owned();
    let right = a.view((0, 10), (10, 10)).into_owned();
    eigen::partial_piv_lu_solve_10x10(&left, &right).or_else(|| left.lu().solve(&right))
}
fn determinant(s: &DMatrix<f64>) -> [f64; 39] {
    let mut b = [0.0; 39];
    for col in 0..3 {
        for (offset, start, count) in [(0, 0, 3), (4, 3, 3), (8, 6, 4)] {
            for k in 0..count {
                b[13 * col + offset + k + 1] += s[(2 * col + 4, start + k)];
                b[13 * col + offset + k] -= s[(2 * col + 5, start + k)];
            }
        }
    }
    b
}
pub fn measure(basis: &[f32; 36], gpu: &FivePointAlgebraResult) -> [f64; 8] {
    let a = generated::build_elimination_matrix(&basis.map(f64::from));
    let Some(s) = solve(&a) else {
        return [f64::NAN; 8];
    };
    let Some(sa) = solve(&gpu.elimination.map(f64::from)) else {
        return [f64::NAN; 8];
    };
    let b = determinant(&s);
    let bs = determinant(&DMatrix::from_column_slice(
        10,
        10,
        &gpu.solved.map(f64::from),
    ));
    let gb = gpu.determinant.map(f64::from);
    let c = generated::determinant_coeffs(&b);
    let cb = generated::determinant_coeffs(&gb);
    [
        error(&gpu.elimination.map(f64::from), &a),
        error(&gpu.solved.map(f64::from), sa.as_slice()),
        error(&gpu.solved.map(f64::from), s.as_slice()),
        error(&gb, &bs),
        error(&gb, &b),
        error(&gpu.coefficients.map(f64::from), &cb),
        error(&cb, &c),
        error(&gpu.coefficients.map(f64::from), &c),
    ]
}
pub fn summary(values: impl Iterator<Item = f64>) -> Value {
    let all: Vec<_> = values.collect();
    let mut v: Vec<_> = all.iter().copied().filter(|x| x.is_finite()).collect();
    v.sort_by(f64::total_cmp);
    let q = |p: f64| {
        v.get(((v.len().saturating_sub(1)) as f64 * p).round() as usize)
            .copied()
    };
    json!({"n":v.len(),"nonfinite":all.len()-v.len(),"p50":q(0.5),"p90":q(0.9),"p99":q(0.99)})
}
