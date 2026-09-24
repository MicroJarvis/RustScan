use crate::types::{ImageFrame, Reconstruction, SensorId};
use anyhow::anyhow;
use std::fmt;
use std::str::FromStr;

mod covariance;
mod pose_prior;
mod taskflow;
pub use taskflow::{BaSchedulingReport, CeresBaTaskflow};

#[cfg(feature = "ceres-ba")]
mod ceres;
#[cfg(feature = "ceres-ba")]
mod ceres_problem;
#[cfg(all(test, feature = "ceres-ba"))]
pub(crate) use ceres::tests::rig_sensor_ba_fixture;
#[cfg(feature = "ceres-ba")]
mod ceres_support;
#[cfg(feature = "ceres-ba")]
mod shared;

pub use covariance::{compute_pose_covariances, BundleAdjustmentCovariance, CovariancePoseBlock};
#[cfg(feature = "ceres-ba")]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundleAdjustmentLinearSolverPreference {
    Auto,
    DenseSchur,
    SparseSchur,
    IterativeSchur,
}

impl Default for BundleAdjustmentLinearSolverPreference {
    fn default() -> Self {
        Self::Auto
    }
}

impl fmt::Display for BundleAdjustmentLinearSolverPreference {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::Auto => "auto",
            Self::DenseSchur => "dense-schur",
            Self::SparseSchur => "sparse-schur",
            Self::IterativeSchur => "iterative-schur",
        };
        formatter.write_str(value)
    }
}

impl FromStr for BundleAdjustmentLinearSolverPreference {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().replace('-', "_").as_str() {
            "auto" => Ok(Self::Auto),
            "dense_schur" => Ok(Self::DenseSchur),
            "sparse_schur" => Ok(Self::SparseSchur),
            "iterative_schur" => Ok(Self::IterativeSchur),
            _ => Err(anyhow!(
                "unsupported BA linear solver '{value}', expected auto, dense-schur, sparse-schur, or iterative-schur"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundleAdjustmentSparseLinearAlgebra {
    Auto,
    SuiteSparse,
    AccelerateSparse,
    EigenSparse,
}

impl Default for BundleAdjustmentSparseLinearAlgebra {
    fn default() -> Self {
        Self::Auto
    }
}

impl fmt::Display for BundleAdjustmentSparseLinearAlgebra {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::Auto => "auto",
            Self::SuiteSparse => "suite-sparse",
            Self::AccelerateSparse => "accelerate-sparse",
            Self::EigenSparse => "eigen-sparse",
        };
        formatter.write_str(value)
    }
}

impl FromStr for BundleAdjustmentSparseLinearAlgebra {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().replace('-', "_").as_str() {
            "auto" => Ok(Self::Auto),
            "suite_sparse" => Ok(Self::SuiteSparse),
            "accelerate_sparse" => Ok(Self::AccelerateSparse),
            "eigen_sparse" => Ok(Self::EigenSparse),
            _ => Err(anyhow!(
                "unsupported BA sparse backend '{value}', expected auto, suite-sparse, accelerate-sparse, or eigen-sparse"
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BundleAdjustmentOptions {
    pub iterations: usize,
    pub linear_solver: BundleAdjustmentLinearSolverPreference,
    pub sparse_linear_algebra: BundleAdjustmentSparseLinearAlgebra,
    pub function_tolerance: f64,
    pub gradient_tolerance: f64,
    pub parameter_tolerance: f64,
    pub max_linear_solver_iterations: usize,
    pub num_threads: isize,
    /// Optional shared admission controller; None preserves direct Ceres execution.
    pub taskflow: Option<CeresBaTaskflow>,
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
            linear_solver: BundleAdjustmentLinearSolverPreference::Auto,
            sparse_linear_algebra: BundleAdjustmentSparseLinearAlgebra::Auto,
            function_tolerance: 0.0,
            gradient_tolerance: 1.0e-4,
            parameter_tolerance: 0.0,
            max_linear_solver_iterations: 200,
            num_threads: -1,
            taskflow: None,
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

/// Decides whether a solved BA parameter vector may mutate the caller's
/// [`crate::types::Reconstruction`].
///
/// A solution is committable only when Ceres reports it usable, the mapped
/// termination type is usable (`Convergence`, `NoConvergence`, or
/// `UserSuccess`), and the staged candidate (parameters + derived point
/// errors) is valid. `NoConvergence` remains eligible when Ceres marks the
/// solution usable; `Failure` / `UserFailure` never commit.
pub(crate) fn should_commit_ba_solution(
    ceres_summary_usable: bool,
    termination_type: BundleAdjustmentTerminationType,
    candidate_valid: bool,
) -> bool {
    ceres_summary_usable && termination_type.is_solution_usable() && candidate_valid
}

/// Test-only seams for BA commit-gate coverage. Compiled only under `cfg(test)`.
#[cfg(test)]
pub(crate) mod commit_test_hooks {
    use super::BundleAdjustmentTerminationType;
    use std::cell::RefCell;

    /// Deterministic overrides applied after Ceres solve and before candidate
    /// validation / commit. Production options never expose these fields.
    #[derive(Debug, Clone, Default)]
    pub(crate) struct BaCommitTestHooks {
        pub force_ceres_usable: Option<bool>,
        pub force_termination: Option<BundleAdjustmentTerminationType>,
        pub corrupt_first_camera_param: Option<f64>,
        /// Overwrites translation-x of the first frame (else first) pose block.
        pub corrupt_first_pose_translation_x: Option<f64>,
        /// Overwrites translation-x of every registered pose block (frame/sensor/image).
        pub corrupt_all_pose_translations_x: Option<f64>,
        /// After cloning the candidate, set every non-ref sensor translation-x
        /// before applying solved parameters (for composed-pose overflow tests).
        pub seed_candidate_sensor_translation_x: Option<f64>,
        /// Request cancel after candidate validation and before live install.
        pub cancel_before_commit: bool,
    }

    thread_local! {
        static ACTIVE: RefCell<Option<BaCommitTestHooks>> = const { RefCell::new(None) };
    }

    pub(crate) fn with_hooks<R>(hooks: BaCommitTestHooks, f: impl FnOnce() -> R) -> R {
        ACTIVE.with(|cell| {
            *cell.borrow_mut() = Some(hooks);
        });
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        ACTIVE.with(|cell| {
            *cell.borrow_mut() = None;
        });
        match result {
            Ok(value) => value,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    pub(crate) fn current() -> Option<BaCommitTestHooks> {
        ACTIVE.with(|cell| cell.borrow().clone())
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
    /// Thread limit passed to Ceres, not a measurement of active OS threads.
    pub solver_num_threads: usize,
    pub scheduling: Option<BaSchedulingReport>,
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
    pub sparse_backend: Option<BundleAdjustmentSparseLinearAlgebra>,
    pub setup_ms: f64,
    pub solve_ms: f64,
    pub postprocess_ms: f64,
    pub elapsed_ms: f64,
    pub covariance: Option<BundleAdjustmentCovariance>,
    pub termination_type: BundleAdjustmentTerminationType,
    pub termination_reason: BundleAdjustmentTerminationReason,
    /// Present when post-BA cameras were reset to the pre-BA snapshot and poses/points were kept.
    pub camera_reset_audit: Option<String>,
}

impl BundleAdjustmentReport {
    pub fn is_solution_usable(&self) -> bool {
        self.termination_type.is_solution_usable()
    }

    pub fn brief_report(&self) -> String {
        format!(
            "termination={:?} reason={:?} solver={:?} sparse_backend={:?} residuals={} parameters={} iterations={}/{} linear_iterations={} cost={:.6}->{:.6} step_quality={:.6} setup_ms={:.2} solve_ms={:.2} postprocess_ms={:.2} elapsed_ms={:.2} solver_threads={} scheduling={:?}",
            self.termination_type,
            self.termination_reason,
            self.linear_solver,
            self.sparse_backend,
            self.residuals,
            self.effective_parameters,
            self.iterations,
            self.attempted_iterations,
            self.linear_solver_iterations,
            self.initial_cost,
            self.final_cost,
            self.step_quality,
            self.setup_ms,
            self.solve_ms,
            self.postprocess_ms,
            self.elapsed_ms,
            self.solver_num_threads,
            self.scheduling
        )
    }
}

pub fn refine_bundle_adjustment(
    frames: &[ImageFrame],
    reconstruction: &mut Reconstruction,
    options: BundleAdjustmentOptions,
) -> Option<BundleAdjustmentReport> {
    try_refine_bundle_adjustment(frames, reconstruction, options).unwrap_or_else(|error| {
        log::error!("BA admission/execution failed: {error:#}");
        None
    })
}

/// Like `refine_bundle_adjustment`, but preserves admission and cooperative-stop
/// errors instead of mapping them to the legacy Option return type.
pub fn try_refine_bundle_adjustment(
    frames: &[ImageFrame],
    reconstruction: &mut Reconstruction,
    mut options: BundleAdjustmentOptions,
) -> anyhow::Result<Option<BundleAdjustmentReport>> {
    if !options.loss_function.has_colmap_valid_scale() {
        return Ok(None);
    }
    if let Some(executor) = options.taskflow.take() {
        return executor.refine(frames, reconstruction, options);
    }
    #[cfg(feature = "ceres-ba")]
    {
        if let Some(control) = crate::execution::active_control() {
            control.checkpoint()?;
            let result = ceres_problem::solve_bundle_adjustment_ceres(
                frames,
                reconstruction,
                options,
                Some(&control),
            );
            if result.is_none() {
                control.checkpoint()?;
            }
            return Ok(result);
        }
        Ok(ceres::refine_bundle_adjustment_ceres(
            frames,
            reconstruction,
            options,
        ))
    }
    #[cfg(not(feature = "ceres-ba"))]
    {
        let _ = (frames, reconstruction, options);
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        should_commit_ba_solution, BundleAdjustmentLinearSolverPreference, BundleAdjustmentLoss,
        BundleAdjustmentSparseLinearAlgebra, BundleAdjustmentTerminationType,
    };

    #[test]
    fn commit_gate_accepts_usable_no_convergence_and_rejects_failures() {
        assert!(should_commit_ba_solution(
            true,
            BundleAdjustmentTerminationType::Convergence,
            true
        ));
        assert!(should_commit_ba_solution(
            true,
            BundleAdjustmentTerminationType::NoConvergence,
            true
        ));
        assert!(should_commit_ba_solution(
            true,
            BundleAdjustmentTerminationType::UserSuccess,
            true
        ));
        assert!(!should_commit_ba_solution(
            false,
            BundleAdjustmentTerminationType::Convergence,
            true
        ));
        assert!(!should_commit_ba_solution(
            true,
            BundleAdjustmentTerminationType::Failure,
            true
        ));
        assert!(!should_commit_ba_solution(
            true,
            BundleAdjustmentTerminationType::UserFailure,
            true
        ));
        assert!(!should_commit_ba_solution(
            true,
            BundleAdjustmentTerminationType::Convergence,
            false
        ));
    }

    #[test]
    fn bundle_adjustment_preference_parsing_accepts_cli_spellings() {
        assert!(matches!(
            "accelerate-sparse".parse(),
            Ok(BundleAdjustmentSparseLinearAlgebra::AccelerateSparse)
        ));
        assert!(matches!(
            "ACCELERATE_SPARSE".parse(),
            Ok(BundleAdjustmentSparseLinearAlgebra::AccelerateSparse)
        ));
        assert!(matches!(
            "iterative_schur".parse(),
            Ok(BundleAdjustmentLinearSolverPreference::IterativeSchur)
        ));
        assert!("cuda"
            .parse::<BundleAdjustmentSparseLinearAlgebra>()
            .is_err());
    }

    #[test]
    fn bundle_adjustment_preference_defaults_are_auto() {
        assert_eq!(
            BundleAdjustmentLinearSolverPreference::default(),
            BundleAdjustmentLinearSolverPreference::Auto
        );
        assert_eq!(
            BundleAdjustmentSparseLinearAlgebra::default(),
            BundleAdjustmentSparseLinearAlgebra::Auto
        );
    }

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
