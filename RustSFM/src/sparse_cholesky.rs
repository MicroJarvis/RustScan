//! Dense-backed compatibility port of COLMAP's `optim/sparse_cholesky`.
//!
//! COLMAP's production implementation wraps CHOLMOD supernodal LLT and falls
//! back to Eigen's simplicial LDLT. RustSFM does not currently depend on a
//! sparse matrix or CHOLMOD binding, so this module preserves the solver state
//! machine and success/failure semantics with a dense `nalgebra` backend:
//! Cholesky for positive-definite systems and LU as the fallback for full-rank
//! non-positive-definite systems.

use nalgebra::{DMatrix, DVector};

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
