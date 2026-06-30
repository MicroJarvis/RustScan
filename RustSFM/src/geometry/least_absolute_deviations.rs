//! Rust port of COLMAP's `optim/least_absolute_deviations` ADMM solver.
//!
//! COLMAP stores `A` as a sparse matrix and can use CHOLMOD-backed sparse
//! Cholesky. RustSFM currently uses a dense `nalgebra` backend, while preserving
//! the ADMM update equations, options, convergence tests, validity semantics,
//! and ridge regularization behavior.

use crate::sparse_cholesky::SparseCholeskyWithFallbackSolver;
use nalgebra::{DMatrix, DVector};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeastAbsoluteDeviationSolverType {
    SimplicialLlt,
    SupernodalCholmodLlt,
}

/// COLMAP `LeastAbsoluteDeviationSolver::Options`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeastAbsoluteDeviationOptions {
    /// Augmented Lagrangian parameter.
    pub rho: f64,
    /// Over-relaxation parameter, typical values are between 1.0 and 1.8.
    pub alpha: f64,
    /// Maximum solver iterations.
    pub max_num_iterations: usize,
    /// Absolute solution threshold.
    pub absolute_tolerance: f64,
    /// Relative solution threshold.
    pub relative_tolerance: f64,
    /// Tikhonov ridge added to the diagonal of A^T A before factorization.
    pub ridge_regularization: f64,
    /// Requested COLMAP linear solver family. Both variants use RustSFM's
    /// dense Cholesky backend until a sparse/CHOLMOD Rust backend is available.
    pub solver_type: LeastAbsoluteDeviationSolverType,
}

impl Default for LeastAbsoluteDeviationOptions {
    fn default() -> Self {
        Self {
            rho: 1.0,
            alpha: 1.0,
            max_num_iterations: 1000,
            absolute_tolerance: 1.0e-4,
            relative_tolerance: 1.0e-2,
            ridge_regularization: 0.0,
            solver_type: LeastAbsoluteDeviationSolverType::SimplicialLlt,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeastAbsoluteDeviationError {
    InvalidOptions(&'static str),
    Underdetermined,
    DimensionMismatch,
}

impl fmt::Display for LeastAbsoluteDeviationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(message) => write!(f, "invalid LAD options: {message}"),
            Self::Underdetermined => write!(f, "underdetermined systems are not supported"),
            Self::DimensionMismatch => write!(f, "LAD input dimensions do not match"),
        }
    }
}

impl std::error::Error for LeastAbsoluteDeviationError {}

#[derive(Debug, Clone)]
pub struct LeastAbsoluteDeviationSolver {
    options: LeastAbsoluteDeviationOptions,
    a: DMatrix<f64>,
    at: DMatrix<f64>,
    linear_solver: LeastAbsoluteDeviationLinearSolver,
    valid: bool,
}

impl LeastAbsoluteDeviationSolver {
    pub fn new(
        options: LeastAbsoluteDeviationOptions,
        a: DMatrix<f64>,
    ) -> Result<Self, LeastAbsoluteDeviationError> {
        validate_options(options)?;
        if a.nrows() < a.ncols() {
            return Err(LeastAbsoluteDeviationError::Underdetermined);
        }

        let at = a.transpose();
        let ata = normal_equations(&a, options.ridge_regularization);
        let linear_solver =
            LeastAbsoluteDeviationLinearSolver::factorize(options.solver_type, &ata);
        let valid = linear_solver.is_some();

        Ok(Self {
            options,
            a,
            at,
            linear_solver: linear_solver.unwrap_or(LeastAbsoluteDeviationLinearSolver::Invalid),
            valid,
        })
    }

    pub fn valid(&self) -> bool {
        self.valid
    }

    pub fn solve(&self, b: &DVector<f64>, x: &mut DVector<f64>) -> bool {
        if !self.valid || b.len() != self.a.nrows() || x.len() != self.a.ncols() {
            return false;
        }

        let mut z = DVector::<f64>::zeros(self.a.nrows());
        let mut z_old = DVector::<f64>::zeros(self.a.nrows());
        let mut u = DVector::<f64>::zeros(self.a.nrows());

        let mut ax;
        let mut ax_hat;

        let b_norm = b.norm();
        let eps_pri_threshold = (self.a.nrows() as f64).sqrt() * self.options.absolute_tolerance;
        let eps_dual_threshold = (self.a.ncols() as f64).sqrt() * self.options.absolute_tolerance;

        for _ in 0..self.options.max_num_iterations {
            let rhs = &self.at * (b + &z - &u);
            let Some(solution) = self.linear_solver.solve(&rhs) else {
                return false;
            };
            *x = solution;
            if !x.iter().all(|v| v.is_finite()) {
                return false;
            }

            ax = &self.a * &*x;
            ax_hat = self.options.alpha * &ax + (1.0 - self.options.alpha) * (&z + b);

            std::mem::swap(&mut z, &mut z_old);
            z = shrinkage(&(ax_hat.clone() - b + &u), 1.0 / self.options.rho);

            u += ax_hat - &z - b;

            let r_norm = (&ax - &z - b).norm();
            let s_norm = (-self.options.rho * (&self.at * (&z - &z_old))).norm();
            let eps_pri = eps_pri_threshold
                + self.options.relative_tolerance * b_norm.max(ax.norm()).max(z.norm());
            let eps_dual = eps_dual_threshold
                + self.options.relative_tolerance * (self.options.rho * (&self.at * &u)).norm();

            if r_norm < eps_pri && s_norm < eps_dual {
                break;
            }
        }

        true
    }
}

#[derive(Debug, Clone)]
enum LeastAbsoluteDeviationLinearSolver {
    SimplicialLlt(DMatrix<f64>),
    SupernodalCholmodLlt(SparseCholeskyWithFallbackSolver),
    Invalid,
}

impl LeastAbsoluteDeviationLinearSolver {
    fn factorize(
        solver_type: LeastAbsoluteDeviationSolverType,
        ata: &DMatrix<f64>,
    ) -> Option<Self> {
        match solver_type {
            LeastAbsoluteDeviationSolverType::SimplicialLlt => {
                if ata.clone().cholesky().is_some() {
                    Some(Self::SimplicialLlt(ata.clone()))
                } else {
                    None
                }
            }
            LeastAbsoluteDeviationSolverType::SupernodalCholmodLlt => {
                let mut solver = SparseCholeskyWithFallbackSolver::new();
                if solver.compute(ata) {
                    Some(Self::SupernodalCholmodLlt(solver))
                } else {
                    None
                }
            }
        }
    }

    fn solve(&self, rhs: &DVector<f64>) -> Option<DVector<f64>> {
        match self {
            Self::SimplicialLlt(matrix) => matrix
                .clone()
                .cholesky()
                .map(|cholesky| cholesky.solve(rhs)),
            Self::SupernodalCholmodLlt(solver) => {
                let mut x = DVector::<f64>::zeros(rhs.len());
                solver.solve(rhs, &mut x).then_some(x)
            }
            Self::Invalid => None,
        }
    }
}

fn validate_options(
    options: LeastAbsoluteDeviationOptions,
) -> Result<(), LeastAbsoluteDeviationError> {
    if options.ridge_regularization < 0.0 {
        return Err(LeastAbsoluteDeviationError::InvalidOptions(
            "ridge_regularization must be non-negative",
        ));
    }
    if options.rho <= 0.0 {
        return Err(LeastAbsoluteDeviationError::InvalidOptions(
            "rho must be positive",
        ));
    }
    if options.alpha <= 0.0 {
        return Err(LeastAbsoluteDeviationError::InvalidOptions(
            "alpha must be positive",
        ));
    }
    if options.max_num_iterations == 0 {
        return Err(LeastAbsoluteDeviationError::InvalidOptions(
            "max_num_iterations must be positive",
        ));
    }
    if options.absolute_tolerance < 0.0 {
        return Err(LeastAbsoluteDeviationError::InvalidOptions(
            "absolute_tolerance must be non-negative",
        ));
    }
    if options.relative_tolerance < 0.0 {
        return Err(LeastAbsoluteDeviationError::InvalidOptions(
            "relative_tolerance must be non-negative",
        ));
    }
    Ok(())
}

fn shrinkage(a: &DVector<f64>, kappa: f64) -> DVector<f64> {
    a.map(|value| (value + kappa).min(0.0) + (value - kappa).max(0.0))
}

fn normal_equations(a: &DMatrix<f64>, ridge_regularization: f64) -> DMatrix<f64> {
    let mut ata = a.transpose() * a;
    if ridge_regularization > 0.0 {
        for i in 0..ata.ncols() {
            ata[(i, i)] += ridge_regularization;
        }
    }
    ata
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solver_types() -> [LeastAbsoluteDeviationSolverType; 2] {
        [
            LeastAbsoluteDeviationSolverType::SimplicialLlt,
            LeastAbsoluteDeviationSolverType::SupernodalCholmodLlt,
        ]
    }

    fn options(solver_type: LeastAbsoluteDeviationSolverType) -> LeastAbsoluteDeviationOptions {
        LeastAbsoluteDeviationOptions {
            solver_type,
            ..LeastAbsoluteDeviationOptions::default()
        }
    }

    fn dense_matrix(rows: usize, cols: usize, values: &[f64]) -> DMatrix<f64> {
        DMatrix::from_row_slice(rows, cols, values)
    }

    #[test]
    fn overdetermined_matches_colmap_reference_solution() {
        for solver_type in solver_types() {
            let mut values = Vec::new();
            for i in 0..4 {
                for j in 0..3 {
                    values.push((i * 3 + j + 1) as f64);
                }
            }
            let mut a = dense_matrix(4, 3, &values);
            a[(0, 0)] = 10.0;
            let b = DVector::from_row_slice(&[1.0, 2.0, 3.0, 4.0]);
            let mut x = DVector::<f64>::zeros(3);

            let solver =
                LeastAbsoluteDeviationSolver::new(options(solver_type), a.clone()).unwrap();
            assert!(solver.valid());
            assert!(solver.solve(&b, &mut x));

            assert!((x[0] - 0.0).abs() < 1.0e-6, "{x:?}");
            assert!((x[1] - 0.0).abs() < 1.0e-6, "{x:?}");
            assert!((x[2] - 1.0 / 3.0).abs() < 1.0e-6, "{x:?}");
            assert!((&a * &x - b).norm() <= 1.0e-6);
        }
    }

    #[test]
    fn well_determined_matches_colmap_reference_solution() {
        for solver_type in solver_types() {
            let mut values = Vec::new();
            for i in 0..3 {
                for j in 0..3 {
                    values.push((i * 3 + j + 1) as f64);
                }
            }
            let mut a = dense_matrix(3, 3, &values);
            a[(0, 0)] = 10.0;
            let b = DVector::from_row_slice(&[1.0, 2.0, 3.0]);
            let mut x = DVector::<f64>::zeros(3);

            let solver =
                LeastAbsoluteDeviationSolver::new(options(solver_type), a.clone()).unwrap();
            assert!(solver.valid());
            assert!(solver.solve(&b, &mut x));

            assert!((x[0] - 0.0).abs() < 1.0e-6, "{x:?}");
            assert!((x[1] - 0.0).abs() < 1.0e-6, "{x:?}");
            assert!((x[2] - 1.0 / 3.0).abs() < 1.0e-6, "{x:?}");
            assert!((&a * &x - b).norm() <= 1.0e-6);
        }
    }

    #[test]
    fn underdetermined_system_is_rejected() {
        let a = DMatrix::<f64>::zeros(2, 3);
        for solver_type in solver_types() {
            assert!(matches!(
                LeastAbsoluteDeviationSolver::new(options(solver_type), a.clone()),
                Err(LeastAbsoluteDeviationError::Underdetermined)
            ));
        }
    }

    #[test]
    fn simple_overdetermined_system_solves_exactly() {
        for solver_type in solver_types() {
            let a = dense_matrix(
                4,
                3,
                &[
                    1.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0, //
                    0.0, 0.0, 1.0, //
                    1.0, 1.0, 1.0,
                ],
            );
            let b = DVector::from_row_slice(&[1.0, 2.0, 3.0, 6.0]);
            let mut x = DVector::<f64>::zeros(3);
            assert!((&a * &x - &b).abs().sum() > 1.0e-1);

            let solver =
                LeastAbsoluteDeviationSolver::new(options(solver_type), a.clone()).unwrap();
            assert!(solver.solve(&b, &mut x));
            assert!((&a * &x - b).abs().sum() <= 1.0e-6, "{x:?}");
        }
    }

    #[test]
    fn diagonal_system_matches_colmap_expected_solution() {
        for solver_type in solver_types() {
            let mut a = DMatrix::<f64>::zeros(5, 5);
            for i in 0..5 {
                a[(i, i)] = i as f64 + 1.0;
            }
            let b = DVector::from_iterator(5, (0..5).map(|i| (i as f64 + 1.0) * (i as f64 + 2.0)));
            let mut x = DVector::<f64>::zeros(5);
            let opts = LeastAbsoluteDeviationOptions {
                max_num_iterations: 100,
                solver_type,
                ..LeastAbsoluteDeviationOptions::default()
            };

            let solver = LeastAbsoluteDeviationSolver::new(opts, a.clone()).unwrap();
            assert!(solver.solve(&b, &mut x));
            let expected = DVector::from_iterator(5, (0..5).map(|i| b[i] / a[(i, i)]));
            assert!((x - expected).abs().sum() <= 1.0e-6);
        }
    }

    #[test]
    fn overdetermined_with_outlier_prefers_l1_solution() {
        for solver_type in solver_types() {
            let mut values = Vec::new();
            for i in 0..6 {
                values.push(1.0);
                values.push(i as f64);
            }
            let a = dense_matrix(6, 2, &values);
            let b = DVector::from_row_slice(&[2.0, 5.0, 8.0, 11.0, 1000.0, 17.0]);
            let mut x = DVector::<f64>::zeros(2);
            let opts = LeastAbsoluteDeviationOptions {
                max_num_iterations: 1000,
                solver_type,
                ..LeastAbsoluteDeviationOptions::default()
            };

            let solver = LeastAbsoluteDeviationSolver::new(opts, a).unwrap();
            assert!(solver.solve(&b, &mut x));
            assert!((x[0] - 2.0).abs() <= 1.0e-3, "{x:?}");
            assert!((x[1] - 3.0).abs() <= 1.0e-3, "{x:?}");
        }
    }

    #[test]
    fn identity_and_scaled_identity_match_expected_solution() {
        for solver_type in solver_types() {
            let identity = DMatrix::<f64>::identity(4, 4);
            let b = DVector::from_row_slice(&[1.0, 2.0, 3.0, 4.0]);
            let mut x = DVector::<f64>::zeros(4);
            let solver = LeastAbsoluteDeviationSolver::new(options(solver_type), identity).unwrap();
            assert!(solver.solve(&b, &mut x));
            assert!((x - &b).abs().sum() <= 1.0e-3);

            let scaled = DMatrix::<f64>::identity(3, 3) * 5.0;
            let b = DVector::from_row_slice(&[5.0, 10.0, 15.0]);
            let mut x = DVector::<f64>::zeros(3);
            let solver = LeastAbsoluteDeviationSolver::new(options(solver_type), scaled).unwrap();
            assert!(solver.solve(&b, &mut x));
            assert!((x - b / 5.0).abs().sum() <= 1.0e-3);
        }
    }

    #[test]
    fn tighter_tolerances_reduce_l1_residual() {
        let a = dense_matrix(
            5,
            3,
            &[
                3.0, 0.5, 0.2, //
                0.5, 2.5, 0.3, //
                0.2, 0.3, 2.0, //
                1.0, 1.5, 0.5, //
                0.7, 0.6, 1.8,
            ],
        );
        let b = DVector::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        for solver_type in solver_types() {
            let mut loose_x = DVector::<f64>::zeros(3);
            let loose_options = LeastAbsoluteDeviationOptions {
                absolute_tolerance: 1.0e-1,
                relative_tolerance: 1.0e-1,
                solver_type,
                ..LeastAbsoluteDeviationOptions::default()
            };
            let loose_solver = LeastAbsoluteDeviationSolver::new(loose_options, a.clone()).unwrap();
            assert!(loose_solver.solve(&b, &mut loose_x));

            let mut tight_x = DVector::<f64>::zeros(3);
            let tight_options = LeastAbsoluteDeviationOptions {
                absolute_tolerance: 1.0e-6,
                relative_tolerance: 1.0e-4,
                max_num_iterations: 2000,
                solver_type,
                ..LeastAbsoluteDeviationOptions::default()
            };
            let tight_solver = LeastAbsoluteDeviationSolver::new(tight_options, a.clone()).unwrap();
            assert!(tight_solver.solve(&b, &mut tight_x));

            let loose_residual = (&a * loose_x - &b).abs().sum();
            let tight_residual = (&a * tight_x - &b).abs().sum();
            assert!(
                tight_residual < 0.99 * loose_residual,
                "loose={loose_residual}, tight={tight_residual}"
            );
        }
    }

    #[test]
    fn ridge_regularization_matches_colmap_validity_behavior() {
        let a = dense_matrix(
            3,
            2,
            &[
                1.0, 1.0, //
                2.0, 2.0, //
                3.0, 3.0,
            ],
        );
        let b = DVector::from_row_slice(&[2.0, 4.0, 6.0]);

        for solver_type in solver_types() {
            let solver =
                LeastAbsoluteDeviationSolver::new(options(solver_type), a.clone()).unwrap();
            assert!(!solver.valid());
            let mut x = DVector::<f64>::zeros(2);
            assert!(!solver.solve(&b, &mut x));

            let opts = LeastAbsoluteDeviationOptions {
                ridge_regularization: 1.0e-9,
                solver_type,
                ..LeastAbsoluteDeviationOptions::default()
            };
            let solver = LeastAbsoluteDeviationSolver::new(opts, a.clone()).unwrap();
            assert!(solver.valid());
            assert!(solver.solve(&b, &mut x));
            assert!(x.iter().all(|v| !v.is_nan()));
            assert!((a.clone() * &x - &b).abs().sum() <= 1.0e-3, "{x:?}");
        }
    }
}
