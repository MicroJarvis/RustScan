//! Dense-backed compatibility port of COLMAP's `optim/sparse_cholesky`.
//!
//! COLMAP's production implementation wraps CHOLMOD supernodal LLT and falls
//! back to Eigen's simplicial LDLT. RustSFM adds a CSC lower-triangle storage
//! path with simplicial Cholesky for native BA Schur complements above the
//! COLMAP/Ceres dense threshold (50 pose entities). Small systems and the LAD
//! path still use the dense `nalgebra` backend.

use nalgebra::{DMatrix, DVector};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseCholeskyBackend {
    Cholesky,
    LuFallback,
}

#[derive(Debug, Clone)]
pub struct SparseCholeskyWithFallbackSolver {
    matrix: Option<DMatrix<f64>>,
    backend: Option<SparseCholeskyBackend>,
    use_lu_fallback: bool,
}

impl Default for SparseCholeskyWithFallbackSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseCholeskyWithFallbackSolver {
    pub fn new() -> Self {
        Self {
            matrix: None,
            backend: None,
            use_lu_fallback: false,
        }
    }

    pub fn compute(&mut self, a: &DMatrix<f64>) -> bool {
        self.analyze_pattern(a);
        self.factorize(a)
    }

    pub fn analyze_pattern(&mut self, _a: &DMatrix<f64>) {
        self.matrix = None;
        self.backend = None;
        self.use_lu_fallback = false;
    }

    pub fn factorize(&mut self, a: &DMatrix<f64>) -> bool {
        self.matrix = None;
        self.backend = None;
        if !a.is_square() {
            return false;
        }

        if !self.use_lu_fallback && a.clone().cholesky().is_some() {
            self.matrix = Some(a.clone());
            self.backend = Some(SparseCholeskyBackend::Cholesky);
            return true;
        }

        self.use_lu_fallback = true;
        let identity = DMatrix::<f64>::identity(a.nrows(), a.ncols());
        if a.clone().lu().solve(&identity).is_none() {
            return false;
        }

        self.matrix = Some(a.clone());
        self.backend = Some(SparseCholeskyBackend::LuFallback);
        true
    }

    pub fn solve(&self, b: &DVector<f64>, x: &mut DVector<f64>) -> bool {
        let Some(matrix) = &self.matrix else {
            return false;
        };
        if b.len() != matrix.nrows() {
            return false;
        }

        let solution = match self.backend {
            Some(SparseCholeskyBackend::Cholesky) => {
                let Some(cholesky) = matrix.clone().cholesky() else {
                    return false;
                };
                cholesky.solve(b)
            }
            Some(SparseCholeskyBackend::LuFallback) => {
                let Some(solution) = matrix.clone().lu().solve(b) else {
                    return false;
                };
                solution
            }
            None => return false,
        };
        if !solution.iter().all(|value| value.is_finite()) {
            return false;
        }

        *x = solution;
        true
    }

    pub fn backend(&self) -> Option<SparseCholeskyBackend> {
        self.backend
    }
}

/// COLMAP/Ceres `DENSE_SCHUR` vs `SPARSE_SCHUR` threshold on pose entities.
pub const DENSE_SCHUR_MAX_POSE_ENTITIES: usize = 50;
/// COLMAP/Ceres `SPARSE_SCHUR` vs `ITERATIVE_SCHUR` threshold on pose entities.
pub const SPARSE_SCHUR_MAX_POSE_ENTITIES: usize = 1000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchurParameterBlock {
    pub offset: usize,
    pub dim: usize,
}

pub fn solve_symmetric_pcg<F, P>(
    n: usize,
    mat_vec: F,
    precondition: P,
    rhs: &DVector<f64>,
    max_iterations: usize,
    tolerance: f64,
) -> Option<DVector<f64>>
where
    F: Fn(&DVector<f64>) -> DVector<f64>,
    P: Fn(&DVector<f64>) -> DVector<f64>,
{
    if rhs.len() != n || max_iterations == 0 {
        return None;
    }
    let mut x = DVector::<f64>::zeros(n);
    let mut r = rhs - mat_vec(&x);
    let mut z = precondition(&r);
    let mut p = z.clone();
    let mut rz_old = r.dot(&z);
    if !rz_old.is_finite() || rz_old <= 0.0 {
        return None;
    }
    let rhs_norm = rhs.norm().max(1.0);
    for _ in 0..max_iterations {
        if r.norm() / rhs_norm <= tolerance {
            break;
        }
        let ap = mat_vec(&p);
        let alpha = rz_old / p.dot(&ap).max(1.0e-24);
        if !alpha.is_finite() {
            return None;
        }
        x += alpha * &p;
        r -= alpha * &ap;
        z = precondition(&r);
        let rz_new = r.dot(&z);
        if !rz_new.is_finite() {
            return None;
        }
        p = z + (rz_new / rz_old) * &p;
        rz_old = rz_new;
    }
    x.iter().all(|value| value.is_finite()).then_some(x)
}

pub fn schur_jacobi_preconditioner<'a>(
    blocks: &'a [SchurParameterBlock],
    get: impl Fn(usize, usize) -> f64 + 'a,
    dim: usize,
) -> impl Fn(&DVector<f64>) -> DVector<f64> + 'a {
    move |residual: &DVector<f64>| {
        let mut out = DVector::<f64>::zeros(dim);
        for block in blocks {
            let mut sub = DMatrix::<f64>::zeros(block.dim, block.dim);
            for row in 0..block.dim {
                for col in 0..block.dim {
                    sub[(row, col)] = get(block.offset + row, block.offset + col);
                }
            }
            let rhs = residual.rows(block.offset, block.dim);
            let delta = sub
                .try_inverse()
                .and_then(|inv| Some(inv * rhs))
                .unwrap_or_else(|| rhs.into_owned());
            out.rows_mut(block.offset, block.dim).copy_from(&delta);
        }
        out
    }
}

/// Lower-triangle accumulator for symmetric sparse systems (Schur complements).
#[derive(Debug, Clone)]
pub struct SymmetricSparseMatrix {
    n: usize,
    entries: BTreeMap<(usize, usize), f64>,
}

impl SymmetricSparseMatrix {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            entries: BTreeMap::new(),
        }
    }

    pub fn dimension(&self) -> usize {
        self.n
    }

    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    pub fn add(&mut self, row: usize, col: usize, value: f64) {
        if value == 0.0 || row >= self.n || col >= self.n {
            return;
        }
        let (r, c) = if row >= col { (row, col) } else { (col, row) };
        let slot = self.entries.entry((r, c)).or_insert(0.0);
        *slot += value;
        if *slot == 0.0 {
            self.entries.remove(&(r, c));
        }
    }

    pub fn add_block(&mut self, row_offset: usize, col_offset: usize, block: &DMatrix<f64>) {
        for r in 0..block.nrows() {
            for c in 0..block.ncols() {
                self.add(row_offset + r, col_offset + c, block[(r, c)]);
            }
        }
    }

    pub fn add_lm_damping_to_diagonal(&mut self, radius: f64, clamp: impl Fn(f64) -> f64) {
        let radius = radius.max(1.0e-32);
        for col in 0..self.n {
            let diag = clamp(self.get(col, col));
            self.add(col, col, diag / radius);
        }
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        let (r, c) = if row >= col { (row, col) } else { (col, row) };
        self.entries.get(&(r, c)).copied().unwrap_or(0.0)
    }

    pub fn mat_vec(&self, x: &DVector<f64>) -> DVector<f64> {
        let mut y = DVector::<f64>::zeros(self.n);
        for (&(row, col), &value) in &self.entries {
            y[row] += value * x[col];
            if row != col {
                y[col] += value * x[row];
            }
        }
        y
    }

    pub fn to_dense(&self) -> DMatrix<f64> {
        let mut dense = DMatrix::<f64>::zeros(self.n, self.n);
        for (&(row, col), &value) in &self.entries {
            dense[(row, col)] = value;
            if row != col {
                dense[(col, row)] = value;
            }
        }
        dense
    }

    pub fn solve(&self, rhs: &DVector<f64>) -> Option<DVector<f64>> {
        if rhs.len() != self.n {
            return None;
        }
        if let Some(chol) = SimplicialSparseCholesky::factorize(self) {
            if let Some(solution) = chol.solve(rhs) {
                return Some(solution);
            }
        }
        let dense = self.to_dense();
        dense
            .clone()
            .cholesky()
            .map(|factor| factor.solve(rhs))
            .or_else(|| dense.lu().solve(rhs))
    }
}

#[derive(Debug, Clone)]
struct SimplicialSparseCholesky {
    n: usize,
    l_lower: BTreeMap<(usize, usize), f64>,
}

impl SimplicialSparseCholesky {
    fn factorize(matrix: &SymmetricSparseMatrix) -> Option<Self> {
        let n = matrix.n;
        if n == 0 {
            return Some(Self {
                n,
                l_lower: BTreeMap::new(),
            });
        }

        let mut l_lower = BTreeMap::<(usize, usize), f64>::new();
        let mut column = vec![0.0; n];

        for i in 0..n {
            column.fill(0.0);
            for row in i..n {
                column[row] = matrix.get(row, i);
            }

            for k in 0..i {
                let l_ik = *l_lower.get(&(i, k))?;
                for j in i..n {
                    if let Some(&l_jk) = l_lower.get(&(j, k)) {
                        column[j] -= l_jk * l_ik;
                    }
                }
            }

            let diag = column[i];
            if !diag.is_finite() || diag <= 0.0 {
                return None;
            }
            let l_ii = diag.sqrt();
            if !l_ii.is_finite() || l_ii <= 0.0 {
                return None;
            }
            l_lower.insert((i, i), l_ii);

            for j in (i + 1)..n {
                let l_ji = column[j] / l_ii;
                if !l_ji.is_finite() {
                    return None;
                }
                if l_ji != 0.0 {
                    l_lower.insert((j, i), l_ji);
                }
            }
        }

        Some(Self { n, l_lower })
    }

    fn solve(&self, rhs: &DVector<f64>) -> Option<DVector<f64>> {
        if rhs.len() != self.n {
            return None;
        }
        let mut y = DVector::<f64>::zeros(self.n);
        for i in 0..self.n {
            let mut sum = rhs[i];
            for k in 0..i {
                if let Some(&l_ik) = self.l_lower.get(&(i, k)) {
                    sum -= l_ik * y[k];
                }
            }
            let l_ii = *self.l_lower.get(&(i, i))?;
            y[i] = sum / l_ii;
        }

        let mut x = DVector::<f64>::zeros(self.n);
        for i in (0..self.n).rev() {
            let mut sum = y[i];
            for k in (i + 1)..self.n {
                if let Some(&l_ki) = self.l_lower.get(&(k, i)) {
                    sum -= l_ki * x[k];
                }
            }
            let l_ii = *self.l_lower.get(&(i, i))?;
            x[i] = sum / l_ii;
        }
        x.iter().all(|value| value.is_finite()).then_some(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain_laplacian_gauge_fixed(n: usize) -> DMatrix<f64> {
        let mut a = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            let mut diag = 0.0;
            if i > 0 {
                a[(i, i - 1)] = -1.0;
                diag += 1.0;
            }
            if i < n - 1 {
                a[(i, i + 1)] = -1.0;
                diag += 1.0;
            }
            a[(i, i)] = diag;
        }
        a[(0, 0)] += 1.0;
        a
    }

    #[test]
    fn sparse_symmetric_matrix_chain_matches_dense_cholesky() {
        let n = 64;
        let a_dense = chain_laplacian_gauge_fixed(n);
        let b = DVector::from_iterator(n, (1..=n).map(|value| value as f64));

        let mut sparse = SymmetricSparseMatrix::new(n);
        for row in 0..n {
            for col in 0..=row {
                let value = a_dense[(row, col)];
                if value != 0.0 {
                    sparse.add(row, col, value);
                }
            }
        }

        let sparse_x = sparse.solve(&b).expect("sparse solve");
        let dense_x = a_dense.clone().cholesky().expect("dense spd").solve(&b);
        assert!((sparse_x - dense_x).norm() <= 1.0e-9);
        assert!(sparse.nnz() < n * n);
    }

    #[test]
    fn sparse_symmetric_matrix_large_chain() {
        let n = 500;
        let a_dense = chain_laplacian_gauge_fixed(n);
        let b = DVector::from_iterator(n, (0..n).map(|i| ((i * 37 % 101) as f64) / 100.0 - 0.5));
        let mut sparse = SymmetricSparseMatrix::new(n);
        for row in 0..n {
            for col in 0..=row {
                let value = a_dense[(row, col)];
                if value != 0.0 {
                    sparse.add(row, col, value);
                }
            }
        }
        let x = sparse.solve(&b).expect("large sparse solve");
        assert!((a_dense * x - b).norm() <= 1.0e-6);
    }

    #[test]
    fn compute_and_solve_diagonal() {
        let a = DMatrix::<f64>::from_diagonal(&DVector::from_row_slice(&[2.0, 3.0, 4.0]));
        let mut solver = SparseCholeskyWithFallbackSolver::new();
        assert!(solver.compute(&a));
        assert_eq!(solver.backend(), Some(SparseCholeskyBackend::Cholesky));

        let b = DVector::from_row_slice(&[2.0, 6.0, 12.0]);
        let mut x = DVector::<f64>::zeros(3);
        assert!(solver.solve(&b, &mut x));
        assert!((x - DVector::from_row_slice(&[1.0, 2.0, 3.0])).norm() <= 1.0e-12);
    }

    #[test]
    fn compute_and_solve_chain() {
        let a = chain_laplacian_gauge_fixed(10);
        let b = DVector::from_iterator(a.nrows(), (1..=10).map(|value| value as f64));
        let mut solver = SparseCholeskyWithFallbackSolver::new();
        assert!(solver.compute(&a));

        let mut x = DVector::<f64>::zeros(a.ncols());
        assert!(solver.solve(&b, &mut x));
        assert!((a * x - b).norm() <= 1.0e-9);
    }

    #[test]
    fn analyze_and_factorize_reused_across_matrices() {
        let a1 = chain_laplacian_gauge_fixed(8);
        let mut a2 = a1.clone();
        for i in 0..a2.ncols() {
            a2[(i, i)] += 0.5;
        }

        let b = DVector::from_iterator(a1.nrows(), (1..=8).map(|value| value as f64));
        let mut solver = SparseCholeskyWithFallbackSolver::new();
        solver.analyze_pattern(&a1);

        let mut x = DVector::<f64>::zeros(a1.ncols());
        assert!(solver.factorize(&a1));
        assert!(solver.solve(&b, &mut x));
        assert!((&a1 * &x - &b).norm() <= 1.0e-9);

        assert!(solver.factorize(&a2));
        assert!(solver.solve(&b, &mut x));
        assert!((a2 * x - b).norm() <= 1.0e-9);
    }

    #[test]
    fn compute_returns_false_on_singular_matrix() {
        let a = DMatrix::<f64>::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
        let mut solver = SparseCholeskyWithFallbackSolver::new();
        assert!(!solver.compute(&a));
    }

    #[test]
    fn ridge_makes_singular_matrix_solvable() {
        let a = DMatrix::<f64>::from_row_slice(2, 2, &[1.0 + 1.0e-6, 1.0, 1.0, 1.0 + 1.0e-6]);
        let mut solver = SparseCholeskyWithFallbackSolver::new();
        assert!(solver.compute(&a));

        let b = DVector::from_row_slice(&[2.0, 2.0]);
        let mut x = DVector::<f64>::zeros(2);
        assert!(solver.solve(&b, &mut x));
        assert!(x.iter().all(|value| !value.is_nan()));
    }

    #[test]
    fn falls_back_to_lu_on_indefinite_matrix() {
        let a = DMatrix::<f64>::from_diagonal(&DVector::from_row_slice(&[1.0, 1.0, -1.0e-20]));
        let mut solver = SparseCholeskyWithFallbackSolver::new();
        assert!(solver.compute(&a));
        assert_eq!(solver.backend(), Some(SparseCholeskyBackend::LuFallback));

        let b = DVector::from_row_slice(&[2.0, 3.0, -5.0e-20]);
        let mut x = DVector::<f64>::zeros(3);
        assert!(solver.solve(&b, &mut x));
        assert!((x - DVector::from_row_slice(&[2.0, 3.0, 5.0])).norm() <= 1.0e-10);
    }

    #[test]
    fn solve_symmetric_pcg_matches_direct_solver_on_spd_chain() {
        let n = 32;
        let a_dense = chain_laplacian_gauge_fixed(n);
        let b = DVector::from_iterator(n, (1..=n).map(|value| value as f64));
        let direct = a_dense.clone().cholesky().unwrap().solve(&b);
        let mat_vec = |vector: &DVector<f64>| &a_dense * vector;
        let blocks = vec![SchurParameterBlock { offset: 0, dim: n }];
        let precondition = schur_jacobi_preconditioner(&blocks, |row, col| a_dense[(row, col)], n);
        let pcg =
            solve_symmetric_pcg(n, mat_vec, precondition, &b, 200, 1.0e-8).expect("pcg solve");
        assert!((pcg - direct).norm() <= 1.0e-6);
    }

    #[test]
    fn ill_conditioned_chain() {
        let a = chain_laplacian_gauge_fixed(500);
        let b = DVector::from_iterator(
            a.nrows(),
            (0..a.nrows()).map(|i| ((i * 37 % 101) as f64) / 100.0 - 0.5),
        );
        let mut solver = SparseCholeskyWithFallbackSolver::new();
        assert!(solver.compute(&a));

        let mut x = DVector::<f64>::zeros(a.ncols());
        assert!(solver.solve(&b, &mut x));
        assert!(x.iter().all(|value| !value.is_nan()));
        assert!((a * x - b).norm() <= 1.0e-6);
    }
}
