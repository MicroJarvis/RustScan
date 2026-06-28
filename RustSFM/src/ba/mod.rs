use crate::types::{ImageFrame, Reconstruction, SensorId};

mod covariance;
mod native;
mod pose_prior;
mod shared;

#[cfg(feature = "ceres-ba")]
mod ceres;
#[cfg(feature = "ceres-ba")]
mod ceres_problem;

pub use covariance::{compute_pose_covariances, BundleAdjustmentCovariance, CovariancePoseBlock};
pub(crate) use pose_prior::POSE_PRIOR_JACOBIAN_EPS;
pub use pose_prior::{
    camera_center_pose_jacobian, camera_center_world, position_prior_information_matrix,
    BundleAdjustmentPosePrior,
};

/// Ceres-equivalent robust loss functions for bundle adjustment.
///
/// Each variant maps to a Ceres `LossFunction` with a robustification scale.
/// The methods operate on `s`, the squared residual norm `||r||^2`, mirroring
/// Ceres' `rho(s)` convention. `weight` returns the IRLS weight `rho'(s)` that
/// scales the residual/Jacobian rows (applied as `sqrt(weight)`), and `cost`
/// returns `0.5 * rho(s)` so the reported objective matches Ceres' cost.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BundleAdjustmentLoss {
    /// `rho(s) = s` (plain squared error, no robustification).
    Trivial,
    /// Ceres `HuberLoss(scale)`.
    Huber { scale: f64 },
    /// Ceres `SoftLOneLoss(scale)`.
    SoftL1 { scale: f64 },
    /// Ceres `CauchyLoss(scale)` (COLMAP incremental mapper default).
    Cauchy { scale: f64 },
}

impl BundleAdjustmentLoss {
    /// Match COLMAP's `CeresBundleAdjustmentOptions::Check()` scale gate.
    #[inline]
    pub fn has_colmap_valid_scale(self) -> bool {
        match self {
            Self::Trivial => true,
            Self::Huber { scale } | Self::SoftL1 { scale } | Self::Cauchy { scale } => {
                scale.is_finite() && scale >= 0.0
            }
        }
    }

    /// IRLS weight `rho'(s)` for a squared residual `s = ||r||^2`.
    #[inline]
    pub fn weight(self, s: f64) -> f64 {
        let s = s.max(0.0);
        match self {
            Self::Trivial => 1.0,
            Self::Huber { scale } => {
                let b2 = scale * scale;
                if s <= b2 {
                    1.0
                } else {
                    (scale / s.max(1.0e-24).sqrt()).max(0.0)
                }
            }
            Self::SoftL1 { scale } => {
                let a2 = (scale * scale).max(1.0e-24);
                1.0 / (1.0 + s / a2).sqrt()
            }
            Self::Cauchy { scale } => {
                let a2 = (scale * scale).max(1.0e-24);
                1.0 / (1.0 + s / a2)
            }
        }
    }

    /// Ceres objective contribution `0.5 * rho(s)` for `s = ||r||^2`.
    #[inline]
    pub fn cost(self, s: f64) -> f64 {
        let s = s.max(0.0);
        match self {
            Self::Trivial => 0.5 * s,
            Self::Huber { scale } => {
                let b2 = scale * scale;
                if s <= b2 {
                    0.5 * s
                } else {
                    scale * s.sqrt() - 0.5 * b2
                }
            }
            Self::SoftL1 { scale } => {
                let a2 = (scale * scale).max(1.0e-24);
                a2 * ((1.0 + s / a2).sqrt() - 1.0)
            }
            Self::Cauchy { scale } => {
                let a2 = (scale * scale).max(1.0e-24);
                0.5 * a2 * (1.0 + s / a2).ln()
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct BundleAdjustmentOptions {
    pub iterations: usize,
    pub function_tolerance: f64,
    pub gradient_tolerance: f64,
    pub parameter_tolerance: f64,
    pub max_linear_solver_iterations: usize,
    pub num_threads: isize,
    pub min_num_residuals_for_multi_threading: usize,
    pub max_num_consecutive_invalid_steps: usize,
    pub max_consecutive_nonmonotonic_steps: usize,
    pub loss_function: BundleAdjustmentLoss,
    pub max_observation_error_px: f64,
    pub variable_images: Option<Vec<usize>>,
    pub constant_images: Vec<usize>,
    pub gauge: BundleAdjustmentGauge,
    pub variable_cameras: Option<Vec<usize>>,
    pub constant_cameras: Vec<usize>,
    pub constant_rigs: Vec<u32>,
    pub constant_sensor_from_rig: Vec<SensorId>,
    pub refine_focal_length: bool,
    pub refine_principal_point: bool,
    pub refine_extra_params: bool,
    pub point_ids: Option<Vec<usize>>,
    pub constant_point_ids: Option<Vec<usize>>,
    pub allow_single_observation_points: bool,
    pub pose_priors: Vec<BundleAdjustmentPosePrior>,
    pub prior_position_fallback_stddev: f64,
    pub compute_covariance: bool,
}

impl Default for BundleAdjustmentOptions {
    fn default() -> Self {
        Self {
            iterations: 100,
            function_tolerance: 0.0,
            gradient_tolerance: 1.0e-4,
            parameter_tolerance: 0.0,
            max_linear_solver_iterations: 200,
            num_threads: -1,
            min_num_residuals_for_multi_threading: 50_000,
            max_num_consecutive_invalid_steps: 10,
            max_consecutive_nonmonotonic_steps: 10,
            loss_function: BundleAdjustmentLoss::Huber { scale: 4.0 },
            max_observation_error_px: 16.0,
            variable_images: None,
            constant_images: Vec::new(),
            gauge: BundleAdjustmentGauge::Default,
            variable_cameras: None,
            constant_cameras: Vec::new(),
            constant_rigs: Vec::new(),
            constant_sensor_from_rig: Vec::new(),
            refine_focal_length: false,
            refine_principal_point: false,
            refine_extra_params: false,
            point_ids: None,
            constant_point_ids: None,
            allow_single_observation_points: false,
            pose_priors: Vec::new(),
            prior_position_fallback_stddev: 1.0,
            compute_covariance: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundleAdjustmentLinearSolver {
    DenseSchur,
    SparseSchur,
    IterativeSchur,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundleAdjustmentPreconditioner {
    SchurJacobi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundleAdjustmentGauge {
    None,
    Default,
    ThreePoints,
    TwoCamsFromWorld,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundleAdjustmentTerminationType {
    Convergence,
    NoConvergence,
    Failure,
    UserSuccess,
    UserFailure,
}

impl BundleAdjustmentTerminationType {
    pub fn is_solution_usable(self) -> bool {
        matches!(
            self,
            Self::Convergence | Self::NoConvergence | Self::UserSuccess
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundleAdjustmentTerminationReason {
    GradientTolerance,
    FunctionTolerance,
    ParameterTolerance,
    MaxIterations,
    LinearizationFailure,
    LinearSolveFailure,
    InvalidStep,
    NoAcceptedStep,
    MaxConsecutiveInvalidSteps,
    MaxConsecutiveNonmonotonicSteps,
}

#[derive(Debug, Clone)]
pub struct BundleAdjustmentReport {
    pub iterations: usize,
    pub attempted_iterations: usize,
    pub successful_steps: usize,
    pub unsuccessful_steps: usize,
    pub linear_solver_iterations: usize,
    pub linearization_failures: usize,
    pub linear_solve_failures: usize,
    pub invalid_steps: usize,
    pub rejected_steps: usize,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub observations: usize,
    pub residuals: usize,
    pub effective_parameters: usize,
    pub gradient_max_norm: f64,
    pub step_norm: f64,
    pub step_quality: f64,
    pub damping: f64,
    pub linear_solver: BundleAdjustmentLinearSolver,
    pub preconditioner: Option<BundleAdjustmentPreconditioner>,
    pub covariance: Option<BundleAdjustmentCovariance>,
    pub termination_type: BundleAdjustmentTerminationType,
    pub termination_reason: BundleAdjustmentTerminationReason,
}

impl BundleAdjustmentReport {
    pub fn is_solution_usable(&self) -> bool {
        self.termination_type.is_solution_usable()
    }

    pub fn brief_report(&self) -> String {
        format!(
            "termination={:?} reason={:?} solver={:?} residuals={} parameters={} iterations={}/{} linear_iterations={} cost={:.6}->{:.6} step_quality={:.6}",
            self.termination_type,
            self.termination_reason,
            self.linear_solver,
            self.residuals,
            self.effective_parameters,
            self.iterations,
            self.attempted_iterations,
            self.linear_solver_iterations,
            self.initial_cost,
            self.final_cost,
            self.step_quality
        )
    }
}

pub fn refine_bundle_adjustment(
    frames: &[ImageFrame],
    reconstruction: &mut Reconstruction,
    options: BundleAdjustmentOptions,
) -> Option<BundleAdjustmentReport> {
    if !options.loss_function.has_colmap_valid_scale() {
        return None;
    }

    #[cfg(feature = "ceres-ba")]
    {
        return ceres::refine_bundle_adjustment_ceres(frames, reconstruction, options);
    }
    #[cfg(not(feature = "ceres-ba"))]
    native::refine_bundle_adjustment_native(frames, reconstruction, options)
}

#[cfg(test)]
mod tests {
    use super::BundleAdjustmentLoss;

    #[test]
    fn loss_scale_check_matches_colmap_ceres_options() {
        assert!(BundleAdjustmentLoss::Trivial.has_colmap_valid_scale());
        assert!(BundleAdjustmentLoss::Huber { scale: 0.0 }.has_colmap_valid_scale());
        assert!(BundleAdjustmentLoss::SoftL1 { scale: 1.0 }.has_colmap_valid_scale());
        assert!(BundleAdjustmentLoss::Cauchy { scale: 1.0 }.has_colmap_valid_scale());

        assert!(!BundleAdjustmentLoss::Huber { scale: -1.0 }.has_colmap_valid_scale());
        assert!(!BundleAdjustmentLoss::SoftL1 { scale: f64::NAN }.has_colmap_valid_scale());
        assert!(!BundleAdjustmentLoss::Cauchy {
            scale: f64::INFINITY
        }
        .has_colmap_valid_scale());
    }
}
