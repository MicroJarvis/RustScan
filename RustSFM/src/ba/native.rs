use super::shared::{
    add_three_point_gauge, bundle_adjustment_point_filter, collect_observations, project_point,
    refresh_point_errors, BaObservation,
};
use super::{
    camera_center_pose_jacobian, camera_center_world, compute_pose_covariances,
    position_prior_information_matrix, BundleAdjustmentCovariance, BundleAdjustmentGauge,
    BundleAdjustmentLinearSolver, BundleAdjustmentLinearSolverPreference, BundleAdjustmentLoss,
    BundleAdjustmentOptions, BundleAdjustmentPreconditioner, BundleAdjustmentReport,
    BundleAdjustmentTerminationReason, BundleAdjustmentTerminationType, CovariancePoseBlock,
    POSE_PRIOR_JACOBIAN_EPS,
};
use crate::sparse_cholesky::{
    schur_jacobi_preconditioner, solve_symmetric_pcg, SchurParameterBlock, SymmetricSparseMatrix,
    DENSE_SCHUR_MAX_POSE_ENTITIES, SPARSE_SCHUR_MAX_POSE_ENTITIES,
};
use crate::types::{
    colmap_camera_model_focal_idxs, colmap_camera_model_principal_point_idxs, CameraModel, Frame,
    ImageFrame, Reconstruction, Rig, Rigid3, SensorId, COLMAP_DIVISION, COLMAP_EUCM,
    COLMAP_FISHEYE, COLMAP_FOV, COLMAP_FULL_OPENCV, COLMAP_OPENCV, COLMAP_OPENCV_FISHEYE,
    COLMAP_PINHOLE, COLMAP_RADIAL, COLMAP_RADIAL_FISHEYE, COLMAP_RAD_TAN_THIN_PRISM_FISHEYE,
    COLMAP_SIMPLE_DIVISION, COLMAP_SIMPLE_FISHEYE, COLMAP_SIMPLE_PINHOLE, COLMAP_SIMPLE_RADIAL,
    COLMAP_SIMPLE_RADIAL_FISHEYE, COLMAP_THIN_PRISM_FISHEYE,
};
use glam::{Quat, Vec3};
use nalgebra::{DMatrix, DVector, SMatrix, SVector};
use rustslam::SE3;
use std::collections::{BTreeMap, HashSet};

type Mat2x3 = SMatrix<f64, 2, 3>;
type Mat2x6 = SMatrix<f64, 2, 6>;
type Mat2 = SMatrix<f64, 2, 2>;
type Mat3 = SMatrix<f64, 3, 3>;
type Vec2 = SVector<f64, 2>;
type Vec3d = SVector<f64, 3>;
type Vec6 = SVector<f64, 6>;

pub(crate) fn refine_bundle_adjustment_native(
    frames: &[ImageFrame],
    reconstruction: &mut Reconstruction,
    options: BundleAdjustmentOptions,
) -> Option<BundleAdjustmentReport> {
    let point_filter = bundle_adjustment_point_filter(
        options.point_ids.as_deref(),
        options.constant_point_ids.as_deref(),
    );
    let mut constant_point_filter = options
        .constant_point_ids
        .as_ref()
        .map(|ids| ids.iter().copied().collect::<HashSet<_>>())
        .unwrap_or_default();
    let mut pose_blocks = variable_pose_blocks(
        reconstruction,
        options.variable_images.as_deref(),
        &options.constant_images,
        &options.constant_rigs,
        matches!(options.gauge, BundleAdjustmentGauge::Default),
    );
    if reconstruction.points.is_empty() {
        return None;
    }
    let observations = collect_observations(
        frames,
        reconstruction,
        options.max_observation_error_px,
        point_filter.as_ref(),
        options.allow_single_observation_points,
    );
    if matches!(options.gauge, BundleAdjustmentGauge::TwoCamsFromWorld) {
        let fixed = apply_two_cams_from_world_gauge(
            &mut pose_blocks,
            reconstruction,
            &options,
            &observations,
        );
        if !fixed {
            add_three_point_gauge(&mut constant_point_filter, reconstruction, &observations);
        }
    } else if matches!(options.gauge, BundleAdjustmentGauge::ThreePoints) {
        add_three_point_gauge(&mut constant_point_filter, reconstruction, &observations);
    }
    reindex_pose_blocks(&mut pose_blocks);

    let sensor_pose_specs = sensor_pose_specs(reconstruction, &pose_blocks, &options);
    let camera_param_specs = camera_param_specs(
        reconstruction,
        &observations,
        &options,
        pose_blocks.dim + sensor_pose_specs.len() * 6,
    );
    let nonpoint_dim = pose_blocks.dim + sensor_pose_specs.len() * 6 + camera_param_specs.len();
    let residuals = count_variable_residuals(
        reconstruction,
        &observations,
        &pose_blocks,
        &sensor_pose_specs,
        &camera_param_specs,
        &constant_point_filter,
    );
    if residuals == 0 {
        return None;
    }
    if nonpoint_dim == 0 {
        return refine_point_only_bundle_adjustment_native(
            frames,
            reconstruction,
            &options,
            observations,
            constant_point_filter,
            residuals,
        );
    }
    let pose_entities = pose_blocks.blocks.len() + sensor_pose_specs.len();
    let (linear_solver, preconditioner) =
        native_linear_solver_policy_for_preference(options.linear_solver, pose_entities);
    let schur_blocks =
        schur_parameter_blocks(&pose_blocks, &sensor_pose_specs, &camera_param_specs);
    if observations.len() * 2 < nonpoint_dim {
        return None;
    }
    let initial_poses = reconstruction.poses.clone();
    let initial_frames = reconstruction.frames.clone();
    sync_frame_pose_blocks_from_images(reconstruction, &pose_blocks);
    let initial_cost = total_cost(
        reconstruction,
        &observations,
        options.loss_function,
        &options.pose_priors,
        options.prior_position_fallback_stddev,
    );
    if !initial_cost.is_finite() {
        reconstruction.poses.clone_from_slice(&initial_poses);
        reconstruction.frames.clone_from_slice(&initial_frames);
        return None;
    }
    let mut final_cost = initial_cost;
    let mut completed = 0usize;
    let mut attempted = 0usize;
    let mut unsuccessful_steps = 0usize;
    let mut linear_solver_iterations = 0usize;
    let mut linearization_failures = 0usize;
    let mut linear_solve_failures = 0usize;
    let mut invalid_steps = 0usize;
    let mut rejected_steps = 0usize;
    let mut consecutive_invalid_steps = 0usize;
    let mut consecutive_nonmonotonic_steps = 0usize;
    let mut gradient_max_norm = f64::INFINITY;
    let mut step_norm = 0.0;
    let mut step_quality = f64::NAN;
    let mut termination_type = BundleAdjustmentTerminationType::NoConvergence;
    let mut termination_reason = BundleAdjustmentTerminationReason::MaxIterations;
    let mut trust_region = TrustRegionState::ceres_initial();

    for _ in 0..options.iterations {
        attempted += 1;
        let Some(system) = build_schur_system(
            reconstruction,
            &observations,
            &pose_blocks,
            &sensor_pose_specs,
            &camera_param_specs,
            &constant_point_filter,
            options.loss_function,
            trust_region.radius,
            &options.pose_priors,
            options.prior_position_fallback_stddev,
        ) else {
            linearization_failures += 1;
            unsuccessful_steps += 1;
            termination_type = BundleAdjustmentTerminationType::Failure;
            termination_reason = BundleAdjustmentTerminationReason::LinearizationFailure;
            break;
        };
        gradient_max_norm = system.g.amax();
        if options.gradient_tolerance > 0.0 && gradient_max_norm <= options.gradient_tolerance {
            termination_type = BundleAdjustmentTerminationType::Convergence;
            termination_reason = BundleAdjustmentTerminationReason::GradientTolerance;
            break;
        }
        let linear_solution = solve_linear_system(
            &system.h,
            &system.g,
            linear_solver,
            preconditioner,
            &schur_blocks,
            options.max_linear_solver_iterations,
        );
        linear_solver_iterations += linear_solution.iterations;
        let Some(delta) = linear_solution.delta else {
            linear_solve_failures += 1;
            unsuccessful_steps += 1;
            consecutive_invalid_steps += 1;
            consecutive_nonmonotonic_steps = 0;
            trust_region.on_step_rejected();
            termination_reason = BundleAdjustmentTerminationReason::LinearSolveFailure;
            if options.max_num_consecutive_invalid_steps > 0
                && consecutive_invalid_steps >= options.max_num_consecutive_invalid_steps
            {
                termination_type = BundleAdjustmentTerminationType::Failure;
                termination_reason = BundleAdjustmentTerminationReason::MaxConsecutiveInvalidSteps;
                break;
            }
            continue;
        };
        let delta_norm = delta.norm();
        step_norm = delta_norm;
        if !delta.iter().all(|v| v.is_finite()) || delta_norm > 20.0 {
            invalid_steps += 1;
            unsuccessful_steps += 1;
            consecutive_invalid_steps += 1;
            consecutive_nonmonotonic_steps = 0;
            trust_region.on_step_rejected();
            termination_reason = BundleAdjustmentTerminationReason::InvalidStep;
            if options.max_num_consecutive_invalid_steps > 0
                && consecutive_invalid_steps >= options.max_num_consecutive_invalid_steps
            {
                termination_type = BundleAdjustmentTerminationType::Failure;
                termination_reason = BundleAdjustmentTerminationReason::MaxConsecutiveInvalidSteps;
                break;
            }
            continue;
        }
        if options.parameter_tolerance > 0.0 && delta_norm <= options.parameter_tolerance {
            termination_type = BundleAdjustmentTerminationType::Convergence;
            termination_reason = BundleAdjustmentTerminationReason::ParameterTolerance;
            break;
        }

        let base_poses = reconstruction.poses.clone();
        let base_frames = reconstruction.frames.clone();
        let base_rigs = reconstruction.rigs.clone();
        let base_points = reconstruction
            .points
            .iter()
            .map(|p| p.xyz)
            .collect::<Vec<_>>();
        let base_camera = reconstruction.camera;
        let base_cameras = reconstruction.cameras.clone();
        let mut accepted = false;
        for step in [1.0, 0.5, 0.25, 0.125, 0.0625] {
            let predicted_decrease = predicted_model_decrease(&system.h, &system.g, &delta, step);
            apply_schur_delta(
                reconstruction,
                &observations,
                &pose_blocks,
                &sensor_pose_specs,
                &camera_param_specs,
                &system.point_blocks,
                &delta,
                step,
            );
            let candidate_cost = total_cost(
                reconstruction,
                &observations,
                options.loss_function,
                &options.pose_priors,
                options.prior_position_fallback_stddev,
            );
            let actual_decrease = final_cost - candidate_cost;
            step_quality = step_quality_ratio(actual_decrease, predicted_decrease);
            if candidate_cost.is_finite()
                && predicted_decrease > 0.0
                && actual_decrease > 0.0
                && step_quality > 0.0
            {
                let previous_cost = final_cost;
                final_cost = candidate_cost;
                trust_region.on_step_accepted(step_quality);
                completed += 1;
                accepted = true;
                consecutive_invalid_steps = 0;
                consecutive_nonmonotonic_steps = 0;
                if relative_cost_change(previous_cost, final_cost) <= options.function_tolerance {
                    termination_type = BundleAdjustmentTerminationType::Convergence;
                    termination_reason = BundleAdjustmentTerminationReason::FunctionTolerance;
                }
                break;
            }
            restore_state(
                reconstruction,
                &base_poses,
                &base_frames,
                &base_rigs,
                &base_points,
                base_camera,
                &base_cameras,
            );
        }
        if !accepted {
            unsuccessful_steps += 1;
            rejected_steps += 1;
            consecutive_invalid_steps = 0;
            consecutive_nonmonotonic_steps += 1;
            trust_region.on_step_rejected();
            termination_reason = BundleAdjustmentTerminationReason::NoAcceptedStep;
            if options.max_consecutive_nonmonotonic_steps > 0
                && consecutive_nonmonotonic_steps >= options.max_consecutive_nonmonotonic_steps
            {
                termination_type = BundleAdjustmentTerminationType::Failure;
                termination_reason =
                    BundleAdjustmentTerminationReason::MaxConsecutiveNonmonotonicSteps;
                break;
            }
        } else if termination_type == BundleAdjustmentTerminationType::Convergence {
            break;
        }
    }
    if termination_type != BundleAdjustmentTerminationType::Failure
        && termination_type != BundleAdjustmentTerminationType::Convergence
    {
        termination_type = if completed == 0
            && attempted > 0
            && linear_solve_failures + invalid_steps == attempted
        {
            BundleAdjustmentTerminationType::Failure
        } else {
            BundleAdjustmentTerminationType::NoConvergence
        };
    }

    refresh_point_errors(frames, reconstruction);
    let covariance = options
        .compute_covariance
        .then(|| {
            compute_bundle_adjustment_covariance(
                reconstruction,
                &observations,
                &pose_blocks,
                &sensor_pose_specs,
                &camera_param_specs,
                &constant_point_filter,
                options.loss_function,
                &options.pose_priors,
                options.prior_position_fallback_stddev,
            )
        })
        .flatten();
    Some(BundleAdjustmentReport {
        iterations: completed,
        attempted_iterations: attempted,
        successful_steps: completed,
        unsuccessful_steps,
        linear_solver_iterations,
        linearization_failures,
        linear_solve_failures,
        invalid_steps,
        rejected_steps,
        initial_cost,
        final_cost,
        observations: observations.len(),
        residuals,
        effective_parameters: nonpoint_dim
            + point_effective_parameter_count(&observations, &constant_point_filter),
        gradient_max_norm,
        step_norm,
        step_quality,
        damping: trust_region.radius,
        linear_solver,
        preconditioner,
        sparse_backend: None,
        setup_ms: 0.0,
        solve_ms: 0.0,
        postprocess_ms: 0.0,
        elapsed_ms: 0.0,
        covariance,
        termination_type,
        termination_reason,
    })
}

pub(crate) fn count_variable_residuals(
    reconstruction: &Reconstruction,
    observations: &[BaObservation],
    pose_blocks: &PoseBlockSet,
    sensor_pose_specs: &[SensorPoseSpec],
    camera_param_specs: &[CameraParamSpec],
    constant_point_filter: &HashSet<usize>,
) -> usize {
    let variable_sensors = sensor_pose_specs
        .iter()
        .map(|spec| spec.key.clone())
        .collect::<HashSet<_>>();
    let variable_cameras = camera_param_specs
        .iter()
        .map(|spec| spec.camera)
        .collect::<HashSet<_>>();
    let variable_observations = observations
        .iter()
        .filter(|obs| {
            !constant_point_filter.contains(&obs.point)
                || pose_blocks
                    .image_to_block
                    .get(obs.image)
                    .copied()
                    .flatten()
                    .and_then(|block| pose_blocks.blocks.get(block))
                    .is_some_and(|block| pose_block_dim(block) > 0)
                || frame_sensor_key_for_image(reconstruction, obs.image)
                    .is_some_and(|key| variable_sensors.contains(&key))
                || camera_index_for_image(reconstruction, obs.image)
                    .is_some_and(|camera| variable_cameras.contains(&camera))
        })
        .count();
    variable_observations * 2
}

fn refine_point_only_bundle_adjustment_native(
    frames: &[ImageFrame],
    reconstruction: &mut Reconstruction,
    options: &BundleAdjustmentOptions,
    observations: Vec<BaObservation>,
    constant_point_filter: HashSet<usize>,
    residuals: usize,
) -> Option<BundleAdjustmentReport> {
    let initial_cost = total_cost(
        reconstruction,
        &observations,
        options.loss_function,
        &[],
        1.0,
    );
    if !initial_cost.is_finite() {
        return None;
    }

    let mut final_cost = initial_cost;
    let mut completed = 0usize;
    let mut attempted = 0usize;
    let mut unsuccessful_steps = 0usize;
    let mut linear_solver_iterations = 0usize;
    let mut linearization_failures = 0usize;
    let mut linear_solve_failures = 0usize;
    let mut invalid_steps = 0usize;
    let mut rejected_steps = 0usize;
    let mut consecutive_invalid_steps = 0usize;
    let mut consecutive_nonmonotonic_steps = 0usize;
    let mut gradient_max_norm = f64::INFINITY;
    let mut step_norm = 0.0;
    let mut step_quality = f64::NAN;
    let mut termination_type = BundleAdjustmentTerminationType::NoConvergence;
    let mut termination_reason = BundleAdjustmentTerminationReason::MaxIterations;
    let mut trust_region = TrustRegionState::ceres_initial();

    for _ in 0..options.iterations {
        attempted += 1;
        let Some(system) = build_point_only_system(
            reconstruction,
            &observations,
            &constant_point_filter,
            options.loss_function,
            trust_region.radius,
        ) else {
            linearization_failures += 1;
            unsuccessful_steps += 1;
            termination_type = BundleAdjustmentTerminationType::Failure;
            termination_reason = BundleAdjustmentTerminationReason::LinearizationFailure;
            break;
        };
        gradient_max_norm = system
            .point_blocks
            .iter()
            .filter(|block| block.active)
            .map(|block| block.g.amax())
            .fold(0.0, f64::max);
        if options.gradient_tolerance > 0.0 && gradient_max_norm <= options.gradient_tolerance {
            termination_type = BundleAdjustmentTerminationType::Convergence;
            termination_reason = BundleAdjustmentTerminationReason::GradientTolerance;
            break;
        }
        let point_deltas =
            solve_point_only_system(&system.point_blocks, options.max_linear_solver_iterations);
        linear_solver_iterations += point_deltas.iterations;
        let Some(point_deltas) = point_deltas.deltas else {
            linear_solve_failures += 1;
            unsuccessful_steps += 1;
            consecutive_invalid_steps += 1;
            consecutive_nonmonotonic_steps = 0;
            trust_region.on_step_rejected();
            termination_reason = BundleAdjustmentTerminationReason::LinearSolveFailure;
            if options.max_num_consecutive_invalid_steps > 0
                && consecutive_invalid_steps >= options.max_num_consecutive_invalid_steps
            {
                termination_type = BundleAdjustmentTerminationType::Failure;
                termination_reason = BundleAdjustmentTerminationReason::MaxConsecutiveInvalidSteps;
                break;
            }
            continue;
        };
        step_norm = point_deltas
            .iter()
            .map(|delta| delta.dot(delta))
            .sum::<f64>()
            .sqrt();
        if !step_norm.is_finite() || step_norm > 20.0 {
            invalid_steps += 1;
            unsuccessful_steps += 1;
            consecutive_invalid_steps += 1;
            consecutive_nonmonotonic_steps = 0;
            trust_region.on_step_rejected();
            termination_reason = BundleAdjustmentTerminationReason::InvalidStep;
            if options.max_num_consecutive_invalid_steps > 0
                && consecutive_invalid_steps >= options.max_num_consecutive_invalid_steps
            {
                termination_type = BundleAdjustmentTerminationType::Failure;
                termination_reason = BundleAdjustmentTerminationReason::MaxConsecutiveInvalidSteps;
                break;
            }
            continue;
        }
        if options.parameter_tolerance > 0.0 && step_norm <= options.parameter_tolerance {
            termination_type = BundleAdjustmentTerminationType::Convergence;
            termination_reason = BundleAdjustmentTerminationReason::ParameterTolerance;
            break;
        }

        let base_points = reconstruction
            .points
            .iter()
            .map(|point| point.xyz)
            .collect::<Vec<_>>();
        let mut accepted = false;
        for step in [1.0, 0.5, 0.25, 0.125, 0.0625] {
            let predicted_decrease =
                point_only_predicted_model_decrease(&system.point_blocks, &point_deltas, step);
            apply_point_only_delta(reconstruction, &point_deltas, step);
            let candidate_cost = total_cost(
                reconstruction,
                &observations,
                options.loss_function,
                &[],
                1.0,
            );
            let actual_decrease = final_cost - candidate_cost;
            step_quality = step_quality_ratio(actual_decrease, predicted_decrease);
            if candidate_cost.is_finite()
                && predicted_decrease > 0.0
                && actual_decrease > 0.0
                && step_quality > 0.0
            {
                let previous_cost = final_cost;
                final_cost = candidate_cost;
                trust_region.on_step_accepted(step_quality);
                completed += 1;
                accepted = true;
                consecutive_invalid_steps = 0;
                consecutive_nonmonotonic_steps = 0;
                if relative_cost_change(previous_cost, final_cost) <= options.function_tolerance {
                    termination_type = BundleAdjustmentTerminationType::Convergence;
                    termination_reason = BundleAdjustmentTerminationReason::FunctionTolerance;
                }
                break;
            }
            for (point, xyz) in reconstruction.points.iter_mut().zip(base_points.iter()) {
                point.xyz = *xyz;
            }
        }
        if !accepted {
            unsuccessful_steps += 1;
            rejected_steps += 1;
            consecutive_invalid_steps = 0;
            consecutive_nonmonotonic_steps += 1;
            trust_region.on_step_rejected();
            termination_reason = BundleAdjustmentTerminationReason::NoAcceptedStep;
            if options.max_consecutive_nonmonotonic_steps > 0
                && consecutive_nonmonotonic_steps >= options.max_consecutive_nonmonotonic_steps
            {
                termination_type = BundleAdjustmentTerminationType::Failure;
                termination_reason =
                    BundleAdjustmentTerminationReason::MaxConsecutiveNonmonotonicSteps;
                break;
            }
        } else if termination_type == BundleAdjustmentTerminationType::Convergence {
            break;
        }
    }

    if termination_type != BundleAdjustmentTerminationType::Failure
        && termination_type != BundleAdjustmentTerminationType::Convergence
    {
        termination_type = if completed == 0
            && attempted > 0
            && linear_solve_failures + invalid_steps == attempted
        {
            BundleAdjustmentTerminationType::Failure
        } else {
            BundleAdjustmentTerminationType::NoConvergence
        };
    }

    refresh_point_errors(frames, reconstruction);
    Some(BundleAdjustmentReport {
        iterations: completed,
        attempted_iterations: attempted,
        successful_steps: completed,
        unsuccessful_steps,
        linear_solver_iterations,
        linearization_failures,
        linear_solve_failures,
        invalid_steps,
        rejected_steps,
        initial_cost,
        final_cost,
        observations: observations.len(),
        residuals,
        effective_parameters: point_effective_parameter_count(
            &observations,
            &constant_point_filter,
        ),
        gradient_max_norm,
        step_norm,
        step_quality,
        damping: trust_region.radius,
        linear_solver: BundleAdjustmentLinearSolver::DenseSchur,
        preconditioner: None,
        sparse_backend: None,
        setup_ms: 0.0,
        solve_ms: 0.0,
        postprocess_ms: 0.0,
        elapsed_ms: 0.0,
        covariance: None,
        termination_type,
        termination_reason,
    })
}

pub(crate) fn point_effective_parameter_count(
    observations: &[BaObservation],
    constant_point_filter: &HashSet<usize>,
) -> usize {
    let mut points = observations
        .iter()
        .filter_map(|obs| (!constant_point_filter.contains(&obs.point)).then_some(obs.point))
        .collect::<Vec<_>>();
    points.sort_unstable();
    points.dedup();
    points.len() * 3
}

#[derive(Clone)]
struct SchurSystem {
    h: SchurHessian,
    g: DVector<f64>,
    point_blocks: Vec<PointBlock>,
}

#[derive(Debug, Clone)]
enum SchurHessian {
    Dense(DMatrix<f64>),
    Sparse(SymmetricSparseMatrix),
}

impl SchurHessian {
    fn new_dense(n: usize) -> Self {
        Self::Dense(DMatrix::zeros(n, n))
    }

    fn new_for_pose_entities(n: usize, pose_entities: usize) -> Self {
        if pose_entities > DENSE_SCHUR_MAX_POSE_ENTITIES {
            Self::Sparse(SymmetricSparseMatrix::new(n))
        } else {
            Self::new_dense(n)
        }
    }

    fn add(&mut self, row: usize, col: usize, value: f64) {
        match self {
            Self::Dense(matrix) => matrix[(row, col)] += value,
            Self::Sparse(matrix) => matrix.add(row, col, value),
        }
    }

    fn add_lm_damping(&mut self, radius: f64) {
        match self {
            Self::Dense(matrix) => add_lm_damping_to_matrix_diagonal(matrix, radius),
            Self::Sparse(matrix) => {
                matrix.add_lm_damping_to_diagonal(radius, clamp_lm_diagonal);
            }
        }
    }

    fn get(&self, row: usize, col: usize) -> f64 {
        match self {
            Self::Dense(matrix) => matrix[(row, col)],
            Self::Sparse(matrix) => matrix.get(row, col),
        }
    }

    fn dimension(&self) -> usize {
        match self {
            Self::Dense(matrix) => matrix.nrows(),
            Self::Sparse(matrix) => matrix.dimension(),
        }
    }

    fn mat_vec(&self, vector: &DVector<f64>) -> DVector<f64> {
        match self {
            Self::Dense(matrix) => matrix * vector,
            Self::Sparse(matrix) => matrix.mat_vec(vector),
        }
    }

    fn to_dense(&self) -> DMatrix<f64> {
        match self {
            Self::Dense(matrix) => matrix.clone(),
            Self::Sparse(matrix) => matrix.to_dense(),
        }
    }
}

fn native_linear_solver_policy(
    pose_entities: usize,
) -> (
    BundleAdjustmentLinearSolver,
    Option<BundleAdjustmentPreconditioner>,
) {
    native_linear_solver_policy_for_preference(
        BundleAdjustmentLinearSolverPreference::Auto,
        pose_entities,
    )
}

fn native_linear_solver_policy_for_preference(
    preference: BundleAdjustmentLinearSolverPreference,
    pose_entities: usize,
) -> (
    BundleAdjustmentLinearSolver,
    Option<BundleAdjustmentPreconditioner>,
) {
    match preference {
        BundleAdjustmentLinearSolverPreference::DenseSchur => {
            (BundleAdjustmentLinearSolver::DenseSchur, None)
        }
        BundleAdjustmentLinearSolverPreference::SparseSchur => {
            (BundleAdjustmentLinearSolver::SparseSchur, None)
        }
        BundleAdjustmentLinearSolverPreference::IterativeSchur => (
            BundleAdjustmentLinearSolver::IterativeSchur,
            Some(BundleAdjustmentPreconditioner::SchurJacobi),
        ),
        BundleAdjustmentLinearSolverPreference::Auto => {
            if pose_entities <= DENSE_SCHUR_MAX_POSE_ENTITIES {
                (BundleAdjustmentLinearSolver::DenseSchur, None)
            } else if pose_entities <= SPARSE_SCHUR_MAX_POSE_ENTITIES {
                (BundleAdjustmentLinearSolver::SparseSchur, None)
            } else {
                (
                    BundleAdjustmentLinearSolver::IterativeSchur,
                    Some(BundleAdjustmentPreconditioner::SchurJacobi),
                )
            }
        }
    }
}

fn schur_parameter_blocks(
    pose_blocks: &PoseBlockSet,
    sensor_pose_specs: &[SensorPoseSpec],
    camera_param_specs: &[CameraParamSpec],
) -> Vec<SchurParameterBlock> {
    let mut blocks = pose_blocks
        .blocks
        .iter()
        .map(|block| SchurParameterBlock {
            offset: block.offset,
            dim: pose_block_dim(block),
        })
        .filter(|block| block.dim > 0)
        .collect::<Vec<_>>();
    blocks.extend(sensor_pose_specs.iter().map(|spec| SchurParameterBlock {
        offset: spec.offset,
        dim: 6,
    }));
    blocks.extend(camera_param_specs.iter().map(|spec| SchurParameterBlock {
        offset: spec.offset,
        dim: 1,
    }));
    blocks
}

fn schur_covariance_pose_blocks(pose_blocks: &PoseBlockSet) -> Vec<CovariancePoseBlock> {
    pose_blocks
        .blocks
        .iter()
        .filter_map(|block| {
            let PoseBlockKind::Image(image) = block.kind else {
                return None;
            };
            (pose_block_dim(block) == 6).then_some(CovariancePoseBlock {
                image,
                offset: block.offset,
                dim: 6,
            })
        })
        .collect()
}

pub(crate) fn compute_bundle_adjustment_covariance(
    reconstruction: &Reconstruction,
    observations: &[BaObservation],
    pose_blocks: &PoseBlockSet,
    sensor_pose_specs: &[SensorPoseSpec],
    camera_param_specs: &[CameraParamSpec],
    constant_point_filter: &HashSet<usize>,
    loss_function: BundleAdjustmentLoss,
    pose_priors: &[super::BundleAdjustmentPosePrior],
    prior_position_fallback_stddev: f64,
) -> Option<BundleAdjustmentCovariance> {
    const COVARIANCE_TRUST_REGION_RADIUS: f64 = 1.0e32;
    let system = build_schur_system(
        reconstruction,
        observations,
        pose_blocks,
        sensor_pose_specs,
        camera_param_specs,
        constant_point_filter,
        loss_function,
        COVARIANCE_TRUST_REGION_RADIUS,
        pose_priors,
        prior_position_fallback_stddev,
    )?;
    compute_pose_covariances(
        &system.h.to_dense(),
        &schur_covariance_pose_blocks(pose_blocks),
    )
}

fn pose_block_jacobian_3d(
    jacobian: crate::ba::pose_prior::Mat3x6,
    block: &PoseBlock,
) -> Option<DMatrix<f64>> {
    let dim = pose_block_dim(block);
    if dim == 0 {
        return None;
    }
    let mut out = DMatrix::<f64>::zeros(3, dim);
    let mut local_col = 0usize;
    for axis in 0..6 {
        if block.free_axes[axis] {
            out.set_column(local_col, &jacobian.column(axis));
            local_col += 1;
        }
    }
    Some(out)
}

fn mat3x6_to_dmatrix(matrix: crate::ba::pose_prior::Mat3x6) -> DMatrix<f64> {
    DMatrix::from_fn(3, 6, |row, col| matrix[(row, col)])
}

fn accumulate_pose_priors(
    reconstruction: &Reconstruction,
    pose_blocks: &PoseBlockSet,
    sensor_pose_specs: &[SensorPoseSpec],
    pose_priors: &[super::BundleAdjustmentPosePrior],
    fallback_stddev: f64,
    h_cc: &mut SchurHessian,
    g_c: &mut DVector<f64>,
) {
    let sensor_pose_lookup = sensor_pose_specs
        .iter()
        .map(|spec| (spec.key.clone(), spec.offset))
        .collect::<BTreeMap<_, _>>();
    for prior in pose_priors {
        let Some(pose) = reconstruction.poses.get(prior.image).copied().flatten() else {
            continue;
        };
        let mut nonpoint_jacobians = Vec::new();
        if let Some(block_idx) = pose_blocks
            .image_to_block
            .get(prior.image)
            .copied()
            .flatten()
        {
            let block = &pose_blocks.blocks[block_idx];
            let jacobian = match block.kind {
                PoseBlockKind::Image(_) => camera_center_pose_jacobian(pose),
                PoseBlockKind::Frame(frame_idx) => {
                    let Some(frame_jacobian) =
                        frame_camera_center_jacobian(reconstruction, frame_idx, prior.image)
                    else {
                        continue;
                    };
                    frame_jacobian
                }
            };
            if let Some(jacobian) = pose_block_jacobian_3d(jacobian, block) {
                nonpoint_jacobians.push((block.offset, jacobian));
            }
        }
        if let Some(key) = frame_sensor_key_for_image(reconstruction, prior.image) {
            if let Some(&offset) = sensor_pose_lookup.get(&key) {
                let Some(jacobian) = sensor_camera_center_jacobian(reconstruction, prior.image)
                else {
                    continue;
                };
                nonpoint_jacobians.push((offset, mat3x6_to_dmatrix(jacobian)));
            }
        }
        if nonpoint_jacobians.is_empty() {
            continue;
        }
        let center = camera_center_world(pose);
        let residual = center - DVector::from_column_slice(&prior.position);
        let information =
            position_prior_information_matrix(&prior.position_covariance, fallback_stddev);
        let info_residual = information * residual;
        for (offset, jacobian) in &nonpoint_jacobians {
            for col in 0..jacobian.ncols() {
                g_c[*offset + col] += jacobian.column(col).dot(&info_residual);
            }
        }
        for (offset_i, jacobian_i) in &nonpoint_jacobians {
            for (offset_j, jacobian_j) in &nonpoint_jacobians {
                for row in 0..jacobian_i.ncols() {
                    for col in 0..jacobian_j.ncols() {
                        h_cc.add(
                            *offset_i + row,
                            *offset_j + col,
                            jacobian_i
                                .column(row)
                                .dot(&(information * jacobian_j.column(col))),
                        );
                    }
                }
            }
        }
    }
}

struct LinearSolveResult {
    delta: Option<DVector<f64>>,
    iterations: usize,
}

struct PointOnlySystem {
    point_blocks: Vec<PointOnlyBlock>,
}

#[derive(Debug, Clone)]
struct PointOnlyBlock {
    h: Mat3,
    g: Vec3d,
    active: bool,
}

struct PointOnlySolveResult {
    deltas: Option<Vec<Vec3d>>,
    iterations: usize,
}

fn solve_linear_system(
    hessian: &SchurHessian,
    gradient: &DVector<f64>,
    linear_solver: BundleAdjustmentLinearSolver,
    preconditioner: Option<BundleAdjustmentPreconditioner>,
    blocks: &[SchurParameterBlock],
    max_iterations: usize,
) -> LinearSolveResult {
    if max_iterations == 0 {
        return LinearSolveResult {
            delta: None,
            iterations: 0,
        };
    }
    let rhs = -gradient;
    match linear_solver {
        BundleAdjustmentLinearSolver::IterativeSchur => {
            let dim = hessian.dimension();
            let mat_vec = |vector: &DVector<f64>| hessian.mat_vec(vector);
            let delta = if matches!(
                preconditioner,
                Some(BundleAdjustmentPreconditioner::SchurJacobi)
            ) {
                let precondition =
                    schur_jacobi_preconditioner(blocks, |row, col| hessian.get(row, col), dim);
                solve_symmetric_pcg(dim, mat_vec, precondition, &rhs, max_iterations, 1.0e-6)
            } else {
                solve_symmetric_pcg(
                    dim,
                    mat_vec,
                    |vector: &DVector<f64>| vector.clone(),
                    &rhs,
                    max_iterations,
                    1.0e-6,
                )
            };
            let solved = delta.is_some();
            LinearSolveResult {
                delta,
                iterations: if solved { max_iterations.max(1) } else { 0 },
            }
        }
        BundleAdjustmentLinearSolver::SparseSchur => match hessian {
            SchurHessian::Sparse(hessian) => LinearSolveResult {
                delta: hessian.solve(&rhs),
                iterations: 1,
            },
            SchurHessian::Dense(hessian) => direct_dense_linear_solve(hessian, &rhs),
        },
        BundleAdjustmentLinearSolver::DenseSchur => match hessian {
            SchurHessian::Dense(hessian) => direct_dense_linear_solve(hessian, &rhs),
            SchurHessian::Sparse(hessian) => LinearSolveResult {
                delta: hessian.solve(&rhs),
                iterations: 1,
            },
        },
    }
}

fn direct_dense_linear_solve(hessian: &DMatrix<f64>, rhs: &DVector<f64>) -> LinearSolveResult {
    if let Some(cholesky) = hessian.clone().cholesky() {
        return LinearSolveResult {
            delta: Some(cholesky.solve(rhs)),
            iterations: 1,
        };
    }
    LinearSolveResult {
        delta: hessian.clone().lu().solve(rhs),
        iterations: 1,
    }
}

fn solve_dense_linear_system(
    hessian: &DMatrix<f64>,
    gradient: &DVector<f64>,
    max_iterations: usize,
) -> LinearSolveResult {
    solve_linear_system(
        &SchurHessian::Dense(hessian.clone()),
        gradient,
        BundleAdjustmentLinearSolver::DenseSchur,
        None,
        &[],
        max_iterations,
    )
}

fn build_point_only_system(
    reconstruction: &Reconstruction,
    observations: &[BaObservation],
    constant_point_filter: &HashSet<usize>,
    loss_function: BundleAdjustmentLoss,
    radius: f64,
) -> Option<PointOnlySystem> {
    let mut point_blocks = (0..reconstruction.points.len())
        .map(|_| PointOnlyBlock {
            h: Mat3::zeros(),
            g: Vec3d::zeros(),
            active: false,
        })
        .collect::<Vec<_>>();

    for obs in observations {
        if constant_point_filter.contains(&obs.point) {
            continue;
        }
        let pose = reconstruction.poses[obs.image]?;
        let point = reconstruction.points.get(obs.point)?.xyz;
        let (residual, _, j_point) = residual_and_jacobians(
            reconstruction.camera_for_image(obs.image),
            pose,
            point,
            obs.xy,
        )?;
        let err = residual.norm();
        let weight = loss_function.weight(err * err);
        let sqrt_w = weight.sqrt();
        let residual = residual * sqrt_w;
        let j_point = j_point * sqrt_w;
        let block = &mut point_blocks[obs.point];
        block.h += j_point.transpose() * j_point;
        block.g += j_point.transpose() * residual;
        block.active = true;
    }

    for block in &mut point_blocks {
        if !block.active {
            continue;
        }
        add_lm_damping_to_mat3_diagonal(&mut block.h, radius);
    }

    Some(PointOnlySystem { point_blocks })
}

fn solve_point_only_system(
    point_blocks: &[PointOnlyBlock],
    max_iterations: usize,
) -> PointOnlySolveResult {
    if max_iterations == 0 {
        return PointOnlySolveResult {
            deltas: None,
            iterations: 0,
        };
    }
    let mut deltas = vec![Vec3d::zeros(); point_blocks.len()];
    let mut solved = false;
    for (point_id, block) in point_blocks.iter().enumerate() {
        if !block.active {
            continue;
        }
        solved = true;
        let rhs = -block.g;
        let Some(delta) = block
            .h
            .cholesky()
            .map(|cholesky| cholesky.solve(&rhs))
            .or_else(|| block.h.lu().solve(&rhs))
        else {
            return PointOnlySolveResult {
                deltas: None,
                iterations: 1,
            };
        };
        deltas[point_id] = delta;
    }
    PointOnlySolveResult {
        deltas: solved.then_some(deltas),
        iterations: 1,
    }
}

fn point_only_predicted_model_decrease(
    point_blocks: &[PointOnlyBlock],
    point_deltas: &[Vec3d],
    step: f64,
) -> f64 {
    let mut decrease = 0.0;
    for (block, delta) in point_blocks.iter().zip(point_deltas.iter()) {
        if !block.active {
            continue;
        }
        let scaled_delta = delta * step;
        let linear = block.g.dot(&scaled_delta);
        let quadratic = 0.5 * scaled_delta.dot(&(block.h * scaled_delta));
        decrease -= linear + quadratic;
    }
    decrease
}

fn apply_point_only_delta(reconstruction: &mut Reconstruction, point_deltas: &[Vec3d], step: f64) {
    for (point, delta) in reconstruction.points.iter_mut().zip(point_deltas.iter()) {
        point.xyz[0] += (delta[0] * step) as f32;
        point.xyz[1] += (delta[1] * step) as f32;
        point.xyz[2] += (delta[2] * step) as f32;
    }
}

fn predicted_model_decrease(
    hessian: &SchurHessian,
    gradient: &DVector<f64>,
    delta: &DVector<f64>,
    step: f64,
) -> f64 {
    let scaled_delta = delta * step;
    let linear = gradient.dot(&scaled_delta);
    let quadratic = match hessian {
        SchurHessian::Dense(hessian) => 0.5 * scaled_delta.dot(&(hessian * &scaled_delta)),
        SchurHessian::Sparse(hessian) => 0.5 * scaled_delta.dot(&hessian.mat_vec(&scaled_delta)),
    };
    -(linear + quadratic)
}

fn step_quality_ratio(actual_decrease: f64, predicted_decrease: f64) -> f64 {
    if !actual_decrease.is_finite() || !predicted_decrease.is_finite() || predicted_decrease <= 0.0
    {
        return f64::NEG_INFINITY;
    }
    actual_decrease / predicted_decrease
}

const LM_MIN_DIAGONAL: f64 = 1.0e-6;
const LM_MAX_DIAGONAL: f64 = 1.0e32;
const INITIAL_TRUST_REGION_RADIUS: f64 = 1.0e4;
const MAX_TRUST_REGION_RADIUS: f64 = 1.0e16;
const MIN_TRUST_REGION_RADIUS: f64 = 1.0e-32;

fn clamp_lm_diagonal(value: f64) -> f64 {
    if !value.is_finite() {
        return LM_MIN_DIAGONAL;
    }
    value.clamp(LM_MIN_DIAGONAL, LM_MAX_DIAGONAL)
}

fn add_lm_damping_to_matrix_diagonal(matrix: &mut DMatrix<f64>, radius: f64) {
    let radius = radius.max(MIN_TRUST_REGION_RADIUS);
    for idx in 0..matrix.nrows() {
        let diag = clamp_lm_diagonal(matrix[(idx, idx)]);
        matrix[(idx, idx)] += diag / radius;
    }
}

fn add_lm_damping_to_mat3_diagonal(matrix: &mut Mat3, radius: f64) {
    let radius = radius.max(MIN_TRUST_REGION_RADIUS);
    for d in 0..3 {
        let diag = clamp_lm_diagonal(matrix[(d, d)]);
        matrix[(d, d)] += diag / radius;
    }
}

#[derive(Debug, Clone, Copy)]
struct TrustRegionState {
    radius: f64,
    decrease_factor: f64,
}

impl TrustRegionState {
    fn ceres_initial() -> Self {
        Self {
            radius: INITIAL_TRUST_REGION_RADIUS,
            decrease_factor: 2.0,
        }
    }

    fn on_step_accepted(&mut self, step_quality: f64) {
        let shrink = 1.0 - (2.0 * step_quality - 1.0).powi(3);
        self.radius = (self.radius / shrink.max(1.0 / 3.0)).min(MAX_TRUST_REGION_RADIUS);
        self.decrease_factor = 2.0;
    }

    fn on_step_rejected(&mut self) {
        self.radius = (self.radius / self.decrease_factor).max(MIN_TRUST_REGION_RADIUS);
        self.decrease_factor = (self.decrease_factor * 2.0).min(1.0e16);
    }
}

fn ceres_step_accepted_radius(radius: f64, step_quality: f64) -> f64 {
    let shrink = 1.0 - (2.0 * step_quality - 1.0).powi(3);
    (radius / shrink.max(1.0 / 3.0)).min(MAX_TRUST_REGION_RADIUS)
}

fn ceres_step_rejected_radius(radius: f64, decrease_factor: f64) -> f64 {
    (radius / decrease_factor).max(MIN_TRUST_REGION_RADIUS)
}

#[derive(Debug, Clone)]
struct PointBlock {
    h_inv: Mat3,
    g: Vec3d,
    nonpoint_blocks: Vec<NonPointBlock>,
}

#[derive(Debug, Clone)]
struct NonPointBlock {
    offset: usize,
    jacobian: DMatrix<f64>,
}

#[derive(Debug, Clone)]
pub(crate) struct PoseBlockSet {
    pub(crate) blocks: Vec<PoseBlock>,
    pub(crate) image_to_block: Vec<Option<usize>>,
    pub(crate) dim: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct PoseBlock {
    pub(crate) kind: PoseBlockKind,
    pub(crate) images: Vec<usize>,
    pub(crate) offset: usize,
    pub(crate) free_axes: [bool; 6],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PoseBlockKind {
    Image(usize),
    Frame(usize),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct SensorPoseKey {
    pub(crate) rig_id: u32,
    pub(crate) sensor_id: SensorId,
}

#[derive(Debug, Clone)]
pub(crate) struct SensorPoseSpec {
    pub(crate) key: SensorPoseKey,
    pub(crate) offset: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CameraParamSpec {
    pub(crate) camera: usize,
    pub(crate) param: usize,
    pub(crate) offset: usize,
}

pub(crate) fn variable_pose_blocks(
    reconstruction: &Reconstruction,
    variable_images: Option<&[usize]>,
    constant_images: &[usize],
    constant_rigs: &[u32],
    apply_default_gauge: bool,
) -> PoseBlockSet {
    let mut constant_images = constant_images.iter().copied().collect::<HashSet<_>>();
    let constant_rigs = constant_rigs.iter().copied().collect::<HashSet<_>>();
    let mut constant_frames = constant_images
        .iter()
        .filter_map(|&image| reconstruction.frame_index_for_image(image))
        .collect::<HashSet<_>>();
    for (frame_idx, frame) in reconstruction.frames.iter().enumerate() {
        if constant_rigs.contains(&frame.rig_id) {
            constant_frames.insert(frame_idx);
        }
    }
    if apply_default_gauge
        && variable_images.is_none()
        && constant_images.is_empty()
        && constant_frames.is_empty()
        && !reconstruction.poses.is_empty()
    {
        constant_images.insert(0);
        if let Some(frame_idx) = reconstruction.frame_index_for_image(0) {
            constant_frames.insert(frame_idx);
        }
    }
    let candidate_images = if let Some(images) = variable_images {
        images.to_vec()
    } else {
        reconstruction
            .poses
            .iter()
            .enumerate()
            .filter_map(|(idx, pose)| pose.is_some().then_some(idx))
            .collect()
    };

    let mut frame_candidates = BTreeMap::<usize, ()>::new();
    let mut image_candidates = Vec::new();
    for image in candidate_images {
        if image >= reconstruction.poses.len() || reconstruction.poses[image].is_none() {
            continue;
        }
        if let Some(frame_idx) = reconstruction.frame_index_for_image(image) {
            if constant_frames.contains(&frame_idx)
                || frame_registered_images_with_sensors(reconstruction, frame_idx).is_none()
            {
                continue;
            }
            frame_candidates.insert(frame_idx, ());
        } else if !constant_images.contains(&image) {
            image_candidates.push(image);
        }
    }

    image_candidates.sort_unstable();
    image_candidates.dedup();
    let mut blocks = Vec::new();
    let mut image_to_block = vec![None; reconstruction.poses.len()];
    for frame_idx in frame_candidates.keys().copied() {
        let Some(images) = frame_registered_images_with_sensors(reconstruction, frame_idx) else {
            continue;
        };
        let block_idx = blocks.len();
        for &image in &images {
            image_to_block[image] = Some(block_idx);
        }
        blocks.push(PoseBlock {
            kind: PoseBlockKind::Frame(frame_idx),
            images,
            offset: block_idx * 6,
            free_axes: [true; 6],
        });
    }
    for image in image_candidates {
        let block_idx = blocks.len();
        image_to_block[image] = Some(block_idx);
        blocks.push(PoseBlock {
            kind: PoseBlockKind::Image(image),
            images: vec![image],
            offset: block_idx * 6,
            free_axes: [true; 6],
        });
    }

    let mut pose_blocks = PoseBlockSet {
        blocks,
        image_to_block,
        dim: 0,
    };
    reindex_pose_blocks(&mut pose_blocks);
    pose_blocks
}

fn reindex_pose_blocks(pose_blocks: &mut PoseBlockSet) {
    let mut offset = 0usize;
    for block in &mut pose_blocks.blocks {
        block.offset = offset;
        offset += pose_block_dim(block);
    }
    pose_blocks.dim = offset;
}

pub(crate) fn pose_block_dim(block: &PoseBlock) -> usize {
    block.free_axes.iter().filter(|&&free| free).count()
}

fn pose_block_active_axes(block: &PoseBlock) -> Vec<usize> {
    block
        .free_axes
        .iter()
        .enumerate()
        .filter_map(|(axis, &free)| free.then_some(axis))
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum GaugeUnitKey {
    Image(usize),
    Frame(usize),
}

#[derive(Debug, Clone, Copy)]
struct GaugePoseUnit {
    key: GaugeUnitKey,
    block: Option<usize>,
    pose: SE3,
}

pub(crate) fn apply_two_cams_from_world_gauge(
    pose_blocks: &mut PoseBlockSet,
    reconstruction: &Reconstruction,
    options: &BundleAdjustmentOptions,
    observations: &[BaObservation],
) -> bool {
    if pose_blocks.dim == 0 {
        return true;
    }

    let units = gauge_pose_units(reconstruction, pose_blocks, options, observations);
    let mut fixed_unit: Option<GaugePoseUnit> = None;
    for unit in &units {
        if !gauge_unit_pose_is_constant(*unit, pose_blocks) {
            continue;
        }
        if let Some(first) = fixed_unit {
            if first.key != unit.key {
                return true;
            }
        } else {
            fixed_unit = Some(*unit);
        }
    }

    let mut first_unit = fixed_unit;
    let mut second_unit = None;
    let mut fixed_dim = 0usize;
    for unit in units {
        if first_unit.is_none() {
            first_unit = Some(unit);
            continue;
        }
        let first = first_unit.unwrap();
        if first.key == unit.key || !gauge_unit_pose_is_variable(unit, pose_blocks) {
            continue;
        }
        if let Some(dim) = two_cam_gauge_fixed_translation_dim(first.pose, unit.pose) {
            second_unit = Some(unit);
            fixed_dim = dim;
            break;
        }
    }

    let (Some(first), Some(second)) = (first_unit, second_unit) else {
        return false;
    };
    if let Some(block_idx) = first.block {
        pose_blocks.blocks[block_idx].free_axes = [false; 6];
    }
    if let Some(block_idx) = second.block {
        pose_blocks.blocks[block_idx].free_axes[3 + fixed_dim] = false;
    }
    true
}

fn gauge_pose_units(
    reconstruction: &Reconstruction,
    pose_blocks: &PoseBlockSet,
    options: &BundleAdjustmentOptions,
    observations: &[BaObservation],
) -> Vec<GaugePoseUnit> {
    let mut units = Vec::new();
    let mut seen = HashSet::new();
    let mut images = observations.iter().map(|obs| obs.image).collect::<Vec<_>>();
    images.sort_unstable();
    images.dedup();
    for image in images {
        if reconstruction.poses[image].is_none()
            || !gauge_image_sensor_is_constant(reconstruction, image, options)
        {
            continue;
        }
        let key = reconstruction
            .frame_index_for_image(image)
            .map(GaugeUnitKey::Frame)
            .unwrap_or(GaugeUnitKey::Image(image));
        if !seen.insert(key) {
            continue;
        }
        let pose = match key {
            GaugeUnitKey::Image(_) => reconstruction.poses[image],
            GaugeUnitKey::Frame(frame_idx) => reconstruction
                .frames
                .get(frame_idx)
                .map(|frame| frame.rig_from_world.to_se3()),
        };
        let Some(pose) = pose else {
            continue;
        };
        let block = pose_blocks.image_to_block.get(image).copied().flatten();
        units.push(GaugePoseUnit { key, block, pose });
    }
    units
}

fn gauge_image_sensor_is_constant(
    reconstruction: &Reconstruction,
    image: usize,
    options: &BundleAdjustmentOptions,
) -> bool {
    let Some(key) = frame_sensor_key_for_image(reconstruction, image) else {
        return true;
    };
    ref_sensor_key(reconstruction, &key)
        || options
            .constant_sensor_from_rig
            .iter()
            .any(|sensor_id| sensor_id == &key.sensor_id)
}

fn gauge_unit_pose_is_constant(unit: GaugePoseUnit, pose_blocks: &PoseBlockSet) -> bool {
    unit.block
        .and_then(|block| pose_blocks.blocks.get(block))
        .is_none_or(|block| pose_block_dim(block) == 0)
}

fn gauge_unit_pose_is_variable(unit: GaugePoseUnit, pose_blocks: &PoseBlockSet) -> bool {
    unit.block
        .and_then(|block| pose_blocks.blocks.get(block))
        .is_some_and(|block| pose_block_dim(block) > 0)
}

fn two_cam_gauge_fixed_translation_dim(first: SE3, second: SE3) -> Option<usize> {
    let baseline = first.compose(&second.inverse()).translation();
    let mut fixed_dim = 0usize;
    let mut max_abs = baseline[0].abs();
    for dim in 1..3 {
        let value = baseline[dim].abs();
        if value > max_abs {
            max_abs = value;
            fixed_dim = dim;
        }
    }
    (max_abs > 1.0e-9).then_some(fixed_dim)
}

fn frame_registered_images_with_sensors(
    reconstruction: &Reconstruction,
    frame_idx: usize,
) -> Option<Vec<usize>> {
    let images = reconstruction
        .image_indices_for_frame_index(frame_idx)
        .into_iter()
        .filter(|&image| reconstruction.poses.get(image).copied().flatten().is_some())
        .collect::<Vec<_>>();
    if images.is_empty()
        || images
            .iter()
            .any(|&image| frame_sensor_from_rig(reconstruction, frame_idx, image).is_none())
    {
        None
    } else {
        Some(images)
    }
}

pub(crate) fn camera_param_specs(
    reconstruction: &Reconstruction,
    observations: &[BaObservation],
    options: &BundleAdjustmentOptions,
    first_offset: usize,
) -> Vec<CameraParamSpec> {
    if !(options.refine_focal_length
        || options.refine_principal_point
        || options.refine_extra_params)
    {
        return Vec::new();
    }

    let constant_cameras = options
        .constant_cameras
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let mut camera_indices = if let Some(cameras) = options.variable_cameras.as_deref() {
        cameras.to_vec()
    } else {
        observations
            .iter()
            .filter_map(|obs| camera_index_for_image(reconstruction, obs.image))
            .collect::<Vec<_>>()
    };
    camera_indices.sort_unstable();
    camera_indices.dedup();

    let mut specs = Vec::new();
    for camera_idx in camera_indices {
        if constant_cameras.contains(&camera_idx) {
            continue;
        }
        let Some(camera) = camera_by_index(reconstruction, camera_idx) else {
            continue;
        };
        for param in selected_camera_params(
            camera,
            options.refine_focal_length,
            options.refine_principal_point,
            options.refine_extra_params,
        ) {
            specs.push(CameraParamSpec {
                camera: camera_idx,
                param,
                offset: first_offset + specs.len(),
            });
        }
    }
    specs
}

pub(crate) fn sensor_pose_specs(
    reconstruction: &Reconstruction,
    pose_blocks: &PoseBlockSet,
    options: &BundleAdjustmentOptions,
) -> Vec<SensorPoseSpec> {
    let constant_sensors = options
        .constant_sensor_from_rig
        .iter()
        .cloned()
        .collect::<HashSet<_>>();
    let mut keys = BTreeMap::<SensorPoseKey, ()>::new();
    for block in &pose_blocks.blocks {
        let PoseBlockKind::Frame(_) = block.kind else {
            continue;
        };
        for &image in &block.images {
            let Some(key) = frame_sensor_key_for_image(reconstruction, image) else {
                continue;
            };
            if constant_sensors.contains(&key.sensor_id) || ref_sensor_key(reconstruction, &key) {
                continue;
            }
            keys.insert(key, ());
        }
    }
    keys.keys()
        .cloned()
        .enumerate()
        .map(|(idx, key)| SensorPoseSpec {
            key,
            offset: pose_blocks.dim + idx * 6,
        })
        .collect()
}

fn ref_sensor_key(reconstruction: &Reconstruction, key: &SensorPoseKey) -> bool {
    reconstruction
        .rigs
        .iter()
        .find(|rig| rig.rig_id == key.rig_id)
        .and_then(|rig| rig.ref_sensor_id.as_ref())
        .is_some_and(|ref_sensor_id| ref_sensor_id == &key.sensor_id)
}

fn selected_camera_params(
    camera: CameraModel,
    refine_focal_length: bool,
    refine_principal_point: bool,
    refine_extra_params: bool,
) -> Vec<usize> {
    let focal = colmap_camera_model_focal_idxs(camera.model_id).unwrap_or(&[]);
    let principal = colmap_camera_model_principal_point_idxs(camera.model_id)
        .map(|idxs| idxs.to_vec())
        .unwrap_or_default();
    let mut selected = Vec::new();
    if refine_focal_length {
        selected.extend(focal.iter().copied());
    }
    if refine_principal_point {
        selected.extend(principal.iter().copied());
    }
    if refine_extra_params {
        selected.extend(
            (0..camera.num_params).filter(|idx| !focal.contains(idx) && !principal.contains(idx)),
        );
    }
    selected.retain(|&idx| idx < camera.num_params);
    selected.sort_unstable();
    selected.dedup();
    selected
}

pub(crate) fn camera_index_for_image(
    reconstruction: &Reconstruction,
    image: usize,
) -> Option<usize> {
    match reconstruction.image_camera_indices.get(image).copied() {
        Some(camera_idx) if camera_idx < reconstruction.cameras.len() => Some(camera_idx),
        Some(0) if reconstruction.cameras.is_empty() => Some(0),
        Some(_) => None,
        None => {
            if reconstruction.cameras.is_empty() || !reconstruction.poses.is_empty() {
                Some(0)
            } else {
                None
            }
        }
    }
}

pub(crate) fn camera_by_index(
    reconstruction: &Reconstruction,
    camera_idx: usize,
) -> Option<CameraModel> {
    reconstruction
        .cameras
        .get(camera_idx)
        .copied()
        .or_else(|| (camera_idx == 0).then_some(reconstruction.camera))
}

fn build_schur_system(
    reconstruction: &Reconstruction,
    observations: &[BaObservation],
    pose_blocks: &PoseBlockSet,
    sensor_pose_specs: &[SensorPoseSpec],
    camera_param_specs: &[CameraParamSpec],
    constant_point_filter: &HashSet<usize>,
    loss_function: BundleAdjustmentLoss,
    radius: f64,
    pose_priors: &[super::BundleAdjustmentPosePrior],
    prior_position_fallback_stddev: f64,
) -> Option<SchurSystem> {
    let mut camera_param_lookup = vec![
        Vec::new();
        camera_param_specs
            .iter()
            .map(|s| s.camera)
            .max()
            .unwrap_or(0)
            + 1
    ];
    for (idx, spec) in camera_param_specs.iter().enumerate() {
        camera_param_lookup[spec.camera].push(idx);
    }
    let sensor_pose_lookup = sensor_pose_specs
        .iter()
        .map(|spec| (spec.key.clone(), spec.offset))
        .collect::<BTreeMap<_, _>>();

    let nonpoint_dim = pose_blocks.dim + sensor_pose_specs.len() * 6 + camera_param_specs.len();
    let pose_entities = pose_blocks.blocks.len() + sensor_pose_specs.len();
    let mut h_cc = SchurHessian::new_for_pose_entities(nonpoint_dim, pose_entities);
    let mut g_c = DVector::<f64>::zeros(nonpoint_dim);
    let mut point_blocks = (0..reconstruction.points.len())
        .map(|_| PointBlock {
            h_inv: Mat3::zeros(),
            g: Vec3d::zeros(),
            nonpoint_blocks: Vec::new(),
        })
        .collect::<Vec<_>>();

    for obs in observations {
        let pose = reconstruction.poses[obs.image]?;
        let point = reconstruction.points.get(obs.point)?.xyz;
        let (residual, j_pose, j_point) = residual_and_jacobians(
            reconstruction.camera_for_image(obs.image),
            pose,
            point,
            obs.xy,
        )?;
        let err = residual.norm();
        let weight = loss_function.weight(err * err);
        let sqrt_w = weight.sqrt();
        let residual = residual * sqrt_w;
        let j_pose = j_pose * sqrt_w;
        let j_point = j_point * sqrt_w;

        let mut nonpoint_jacobians = Vec::new();
        if let Some(block_idx) = pose_blocks.image_to_block.get(obs.image).copied().flatten() {
            let block = &pose_blocks.blocks[block_idx];
            let j_pose = match block.kind {
                PoseBlockKind::Image(_) => j_pose,
                PoseBlockKind::Frame(frame_idx) => {
                    frame_pose_jacobian(
                        reconstruction,
                        frame_idx,
                        obs.image,
                        reconstruction.camera_for_image(obs.image),
                        point,
                    )? * sqrt_w
                }
            };
            if let Some(j_pose) = pose_block_jacobian(j_pose, block) {
                nonpoint_jacobians.push((block.offset, j_pose));
            }
        }
        if let Some(key) = frame_sensor_key_for_image(reconstruction, obs.image) {
            if let Some(&offset) = sensor_pose_lookup.get(&key) {
                let j_sensor = sensor_pose_jacobian(
                    reconstruction,
                    obs.image,
                    reconstruction.camera_for_image(obs.image),
                    point,
                )? * sqrt_w;
                nonpoint_jacobians.push((offset, mat2x6_to_dmatrix(j_sensor)));
            }
        }
        if let Some(camera_idx) = camera_index_for_image(reconstruction, obs.image) {
            if camera_idx < camera_param_lookup.len() {
                let camera = camera_by_index(reconstruction, camera_idx)?;
                for &spec_idx in &camera_param_lookup[camera_idx] {
                    let spec = camera_param_specs[spec_idx];
                    if let Some(j_param) = camera_param_jacobian(camera, spec.param, pose, point) {
                        nonpoint_jacobians.push((spec.offset, vec2_to_dmatrix(j_param * sqrt_w)));
                    }
                }
            }
        }

        let residual_d = DVector::from_column_slice(&[residual[0], residual[1]]);
        let point_is_constant = constant_point_filter.contains(&obs.point);
        if !point_is_constant {
            let point_block = &mut point_blocks[obs.point];
            point_block.h_inv += j_point.transpose() * j_point;
            point_block.g += j_point.transpose() * residual;
            for (offset, jacobian) in &nonpoint_jacobians {
                point_block.nonpoint_blocks.push(NonPointBlock {
                    offset: *offset,
                    jacobian: point_nonpoint_cross(j_point, jacobian),
                });
            }
        }
        for (offset, jacobian) in &nonpoint_jacobians {
            let g = jacobian.transpose() * &residual_d;
            for r in 0..jacobian.ncols() {
                g_c[*offset + r] += g[r];
            }
        }
        for (offset_i, jacobian_i) in &nonpoint_jacobians {
            for (offset_j, jacobian_j) in &nonpoint_jacobians {
                let h = jacobian_i.transpose() * jacobian_j;
                for r in 0..jacobian_i.ncols() {
                    for c in 0..jacobian_j.ncols() {
                        h_cc.add(*offset_i + r, *offset_j + c, h[(r, c)]);
                    }
                }
            }
        }
    }

    h_cc.add_lm_damping(radius);
    accumulate_pose_priors(
        reconstruction,
        pose_blocks,
        sensor_pose_specs,
        pose_priors,
        prior_position_fallback_stddev,
        &mut h_cc,
        &mut g_c,
    );

    for block in &mut point_blocks {
        if block.nonpoint_blocks.is_empty() {
            continue;
        }
        add_lm_damping_to_mat3_diagonal(&mut block.h_inv, radius);
        block.h_inv = block.h_inv.try_inverse()?;
        for e_i in &block.nonpoint_blocks {
            let schur_g = e_i.jacobian.transpose() * block.h_inv * block.g;
            for r in 0..e_i.jacobian.ncols() {
                g_c[e_i.offset + r] -= schur_g[r];
            }
            for e_j in &block.nonpoint_blocks {
                let schur_h = e_i.jacobian.transpose() * block.h_inv * &e_j.jacobian;
                for r in 0..e_i.jacobian.ncols() {
                    for c in 0..e_j.jacobian.ncols() {
                        h_cc.add(e_i.offset + r, e_j.offset + c, -schur_h[(r, c)]);
                    }
                }
            }
        }
    }

    Some(SchurSystem {
        h: h_cc,
        g: g_c,
        point_blocks,
    })
}

fn apply_schur_delta(
    reconstruction: &mut Reconstruction,
    observations: &[BaObservation],
    pose_blocks: &PoseBlockSet,
    sensor_pose_specs: &[SensorPoseSpec],
    camera_param_specs: &[CameraParamSpec],
    point_blocks: &[PointBlock],
    nonpoint_delta: &DVector<f64>,
    step: f64,
) {
    for block in &pose_blocks.blocks {
        let mut delta = Vec6::zeros();
        let mut local_col = 0usize;
        for axis in 0..6 {
            if block.free_axes[axis] {
                delta[axis] = nonpoint_delta[block.offset + local_col] * step;
                local_col += 1;
            }
        }
        apply_pose_block_delta(reconstruction, block, delta);
    }

    let changed_sensors = sensor_pose_specs
        .iter()
        .filter_map(|spec| {
            let delta = Vec6::from_iterator((0..6).map(|k| nonpoint_delta[spec.offset + k] * step));
            apply_sensor_pose_delta(reconstruction, &spec.key, delta).then_some(spec.key.clone())
        })
        .collect::<Vec<_>>();
    if !changed_sensors.is_empty() {
        sync_pose_blocks_for_sensor_changes(reconstruction, pose_blocks, &changed_sensors);
    }

    for spec in camera_param_specs {
        let delta = nonpoint_delta[spec.offset] * step;
        apply_camera_param_delta(reconstruction, *spec, delta);
    }

    for (point_idx, block) in point_blocks.iter().enumerate() {
        if block.nonpoint_blocks.is_empty() || point_idx >= reconstruction.points.len() {
            continue;
        }
        let mut rhs = block.g;
        for nonpoint_block in &block.nonpoint_blocks {
            for row in 0..3 {
                let mut value = 0.0;
                for col in 0..nonpoint_block.jacobian.ncols() {
                    value += nonpoint_block.jacobian[(row, col)]
                        * nonpoint_delta[nonpoint_block.offset + col];
                }
                rhs[row] += value;
            }
        }
        let point_delta = -(block.h_inv * rhs) * step;
        let point = &mut reconstruction.points[point_idx].xyz;
        point[0] += point_delta[0] as f32;
        point[1] += point_delta[1] as f32;
        point[2] += point_delta[2] as f32;
    }

    // The observation list is fixed within one BA call, so no track topology update is needed here.
    let _ = observations;
}

fn sync_frame_pose_blocks_from_images(
    reconstruction: &mut Reconstruction,
    pose_blocks: &PoseBlockSet,
) {
    for block in &pose_blocks.blocks {
        let PoseBlockKind::Frame(frame_idx) = block.kind else {
            continue;
        };
        let Some(&reference_image) = block.images.first() else {
            continue;
        };
        let Some(rig_from_world) =
            frame_rig_from_world_from_image(reconstruction, frame_idx, reference_image)
        else {
            continue;
        };
        set_frame_pose_block(reconstruction, frame_idx, &block.images, rig_from_world);
    }
}

fn apply_pose_block_delta(reconstruction: &mut Reconstruction, block: &PoseBlock, delta: Vec6) {
    match block.kind {
        PoseBlockKind::Image(image) => {
            if let Some(pose) = reconstruction.poses.get(image).copied().flatten() {
                reconstruction.poses[image] = Some(apply_pose_delta_f64(pose, delta));
            }
        }
        PoseBlockKind::Frame(frame_idx) => {
            let Some(frame) = reconstruction.frames.get(frame_idx) else {
                return;
            };
            let rig_from_world = apply_pose_delta_f64(frame.rig_from_world.to_se3(), delta);
            set_frame_pose_block(reconstruction, frame_idx, &block.images, rig_from_world);
        }
    }
}

pub(crate) fn set_frame_pose_block(
    reconstruction: &mut Reconstruction,
    frame_idx: usize,
    images: &[usize],
    rig_from_world: SE3,
) {
    let image_poses = images
        .iter()
        .filter_map(|&image| {
            let sensor_from_rig = frame_sensor_from_rig(reconstruction, frame_idx, image)?;
            Some((image, sensor_from_rig.compose(&rig_from_world)))
        })
        .collect::<Vec<_>>();
    if let Some(frame) = reconstruction.frames.get_mut(frame_idx) {
        frame.rig_from_world = Rigid3::from_se3(rig_from_world);
    }
    for (image, pose) in image_poses {
        if let Some(slot) = reconstruction.poses.get_mut(image) {
            *slot = Some(pose);
        }
    }
}

fn frame_rig_from_world_from_image(
    reconstruction: &Reconstruction,
    frame_idx: usize,
    image: usize,
) -> Option<SE3> {
    let image_pose = reconstruction.poses.get(image).copied().flatten()?;
    let sensor_from_rig = frame_sensor_from_rig(reconstruction, frame_idx, image)?;
    Some(sensor_from_rig.inverse().compose(&image_pose))
}

pub(crate) fn frame_sensor_from_rig(
    reconstruction: &Reconstruction,
    frame_idx: usize,
    image: usize,
) -> Option<SE3> {
    let frame = reconstruction.frames.get(frame_idx)?;
    let sensor_id = reconstruction.frame_sensor_id_for_image(frame_idx, image)?;
    reconstruction.sensor_from_rig(frame.rig_id, sensor_id)
}

pub(crate) fn frame_sensor_key_for_image(
    reconstruction: &Reconstruction,
    image: usize,
) -> Option<SensorPoseKey> {
    let frame_idx = reconstruction.frame_index_for_image(image)?;
    let frame = reconstruction.frames.get(frame_idx)?;
    let sensor_id = reconstruction
        .frame_sensor_id_for_image(frame_idx, image)?
        .clone();
    Some(SensorPoseKey {
        rig_id: frame.rig_id,
        sensor_id,
    })
}

fn frame_pose_jacobian(
    reconstruction: &Reconstruction,
    frame_idx: usize,
    image: usize,
    camera: CameraModel,
    point: [f32; 3],
) -> Option<Mat2x6> {
    let frame = reconstruction.frames.get(frame_idx)?;
    let rig_from_world = frame.rig_from_world.to_se3();
    let sensor_from_rig = frame_sensor_from_rig(reconstruction, frame_idx, image)?;
    analytic_frame_pose_jacobian(camera, sensor_from_rig, rig_from_world, point)
        .or_else(|| numerical_frame_pose_jacobian(camera, sensor_from_rig, rig_from_world, point))
}

fn sensor_pose_jacobian(
    reconstruction: &Reconstruction,
    image: usize,
    camera: CameraModel,
    point: [f32; 3],
) -> Option<Mat2x6> {
    let frame_idx = reconstruction.frame_index_for_image(image)?;
    let frame = reconstruction.frames.get(frame_idx)?;
    let rig_from_world = frame.rig_from_world.to_se3();
    let sensor_from_rig = frame_sensor_from_rig(reconstruction, frame_idx, image)?;
    analytic_sensor_pose_jacobian(camera, sensor_from_rig, rig_from_world, point)
        .or_else(|| numerical_sensor_pose_jacobian(camera, sensor_from_rig, rig_from_world, point))
}

fn frame_camera_center_jacobian(
    reconstruction: &Reconstruction,
    frame_idx: usize,
    image: usize,
) -> Option<crate::ba::pose_prior::Mat3x6> {
    let frame = reconstruction.frames.get(frame_idx)?;
    let rig_from_world = frame.rig_from_world.to_se3();
    let sensor_from_rig = frame_sensor_from_rig(reconstruction, frame_idx, image)?;
    let mut jacobian = crate::ba::pose_prior::Mat3x6::zeros();
    for axis in 0..6 {
        let mut plus = Vec6::zeros();
        plus[axis] = POSE_PRIOR_JACOBIAN_EPS;
        let mut minus = Vec6::zeros();
        minus[axis] = -POSE_PRIOR_JACOBIAN_EPS;
        let plus_center = camera_center_world(
            sensor_from_rig.compose(&apply_pose_delta_f64(rig_from_world, plus)),
        );
        let minus_center = camera_center_world(
            sensor_from_rig.compose(&apply_pose_delta_f64(rig_from_world, minus)),
        );
        jacobian.set_column(
            axis,
            &((plus_center - minus_center) / (2.0 * POSE_PRIOR_JACOBIAN_EPS)),
        );
    }
    Some(jacobian)
}

fn sensor_camera_center_jacobian(
    reconstruction: &Reconstruction,
    image: usize,
) -> Option<crate::ba::pose_prior::Mat3x6> {
    let frame_idx = reconstruction.frame_index_for_image(image)?;
    let frame = reconstruction.frames.get(frame_idx)?;
    let rig_from_world = frame.rig_from_world.to_se3();
    let sensor_from_rig = frame_sensor_from_rig(reconstruction, frame_idx, image)?;
    let mut jacobian = crate::ba::pose_prior::Mat3x6::zeros();
    for axis in 0..6 {
        let mut plus = Vec6::zeros();
        plus[axis] = POSE_PRIOR_JACOBIAN_EPS;
        let mut minus = Vec6::zeros();
        minus[axis] = -POSE_PRIOR_JACOBIAN_EPS;
        let plus_center = camera_center_world(
            apply_pose_delta_f64(sensor_from_rig, plus).compose(&rig_from_world),
        );
        let minus_center = camera_center_world(
            apply_pose_delta_f64(sensor_from_rig, minus).compose(&rig_from_world),
        );
        jacobian.set_column(
            axis,
            &((plus_center - minus_center) / (2.0 * POSE_PRIOR_JACOBIAN_EPS)),
        );
    }
    Some(jacobian)
}

pub(crate) fn analytic_frame_pose_jacobian(
    camera: CameraModel,
    sensor_from_rig: SE3,
    rig_from_world: SE3,
    point: [f32; 3],
) -> Option<Mat2x6> {
    let image_pose = sensor_from_rig.compose(&rig_from_world);
    let (j_image_pose, _) = analytic_projection_jacobians(camera, image_pose, point)?;
    let r_sensor = mat3_from_pose_rotation(sensor_from_rig);
    let mut jacobian = Mat2x6::zeros();
    for row in 0..2 {
        for col in 0..3 {
            jacobian[(row, col)] = j_image_pose[(row, 0)] * r_sensor[(0, col)]
                + j_image_pose[(row, 1)] * r_sensor[(1, col)]
                + j_image_pose[(row, 2)] * r_sensor[(2, col)];
            jacobian[(row, col + 3)] = j_image_pose[(row, 3)] * r_sensor[(0, col)]
                + j_image_pose[(row, 4)] * r_sensor[(1, col)]
                + j_image_pose[(row, 5)] * r_sensor[(2, col)];
        }
    }
    Some(jacobian)
}

fn numerical_frame_pose_jacobian(
    camera: CameraModel,
    sensor_from_rig: SE3,
    rig_from_world: SE3,
    point: [f32; 3],
) -> Option<Mat2x6> {
    let mut jacobian = Mat2x6::zeros();
    let eps = [1.0e-4; 6];
    for axis in 0..6 {
        let mut plus = Vec6::zeros();
        plus[axis] = eps[axis];
        let mut minus = Vec6::zeros();
        minus[axis] = -eps[axis];
        let p_plus = project_point(
            camera,
            sensor_from_rig.compose(&apply_pose_delta_f64(rig_from_world, plus)),
            point,
        )?;
        let p_minus = project_point(
            camera,
            sensor_from_rig.compose(&apply_pose_delta_f64(rig_from_world, minus)),
            point,
        )?;
        jacobian[(0, axis)] = (p_plus[0] - p_minus[0]) / (2.0 * eps[axis]);
        jacobian[(1, axis)] = (p_plus[1] - p_minus[1]) / (2.0 * eps[axis]);
    }
    Some(jacobian)
}

pub(crate) fn analytic_sensor_pose_jacobian(
    camera: CameraModel,
    sensor_from_rig: SE3,
    rig_from_world: SE3,
    point: [f32; 3],
) -> Option<Mat2x6> {
    let image_pose = sensor_from_rig.compose(&rig_from_world);
    let (j_image_pose, _) = analytic_projection_jacobians(camera, image_pose, point)?;
    let r_sensor = mat3_from_pose_rotation(sensor_from_rig);
    let t_rig = vec3_from_pose_translation(rig_from_world);
    let dt_domega = -cross_matrix(&(r_sensor * t_rig));
    let mut jacobian = Mat2x6::zeros();
    for row in 0..2 {
        for col in 0..3 {
            jacobian[(row, col)] = j_image_pose[(row, col)]
                + j_image_pose[(row, 3)] * dt_domega[(0, col)]
                + j_image_pose[(row, 4)] * dt_domega[(1, col)]
                + j_image_pose[(row, 5)] * dt_domega[(2, col)];
            jacobian[(row, col + 3)] = j_image_pose[(row, col + 3)];
        }
    }
    Some(jacobian)
}

fn numerical_sensor_pose_jacobian(
    camera: CameraModel,
    sensor_from_rig: SE3,
    rig_from_world: SE3,
    point: [f32; 3],
) -> Option<Mat2x6> {
    let mut jacobian = Mat2x6::zeros();
    let eps = [1.0e-4; 6];
    for axis in 0..6 {
        let mut plus = Vec6::zeros();
        plus[axis] = eps[axis];
        let mut minus = Vec6::zeros();
        minus[axis] = -eps[axis];
        let p_plus = project_point(
            camera,
            apply_pose_delta_f64(sensor_from_rig, plus).compose(&rig_from_world),
            point,
        )?;
        let p_minus = project_point(
            camera,
            apply_pose_delta_f64(sensor_from_rig, minus).compose(&rig_from_world),
            point,
        )?;
        jacobian[(0, axis)] = (p_plus[0] - p_minus[0]) / (2.0 * eps[axis]);
        jacobian[(1, axis)] = (p_plus[1] - p_minus[1]) / (2.0 * eps[axis]);
    }
    Some(jacobian)
}

fn mat3_from_pose_rotation(pose: SE3) -> Mat3 {
    let rotation = pose.rotation_matrix();
    Mat3::from_row_slice(&[
        rotation[0][0] as f64,
        rotation[0][1] as f64,
        rotation[0][2] as f64,
        rotation[1][0] as f64,
        rotation[1][1] as f64,
        rotation[1][2] as f64,
        rotation[2][0] as f64,
        rotation[2][1] as f64,
        rotation[2][2] as f64,
    ])
}

fn vec3_from_pose_translation(pose: SE3) -> Vec3d {
    let translation = pose.translation();
    Vec3d::new(
        translation[0] as f64,
        translation[1] as f64,
        translation[2] as f64,
    )
}

fn cross_matrix(vector: &Vec3d) -> Mat3 {
    Mat3::new(
        0.0, -vector[2], vector[1], vector[2], 0.0, -vector[0], -vector[1], vector[0], 0.0,
    )
}

pub(crate) fn apply_sensor_pose_delta(
    reconstruction: &mut Reconstruction,
    key: &SensorPoseKey,
    delta: Vec6,
) -> bool {
    let Some(rig) = reconstruction
        .rigs
        .iter_mut()
        .find(|rig| rig.rig_id == key.rig_id)
    else {
        return false;
    };
    if rig
        .ref_sensor_id
        .as_ref()
        .is_some_and(|ref_sensor_id| ref_sensor_id == &key.sensor_id)
    {
        return false;
    }
    let Some(sensor) = rig
        .sensors
        .iter_mut()
        .find(|sensor| sensor.sensor_id == key.sensor_id)
    else {
        return false;
    };
    let current = sensor
        .sensor_from_rig
        .as_ref()
        .map(Rigid3::to_se3)
        .unwrap_or_else(SE3::identity);
    sensor.sensor_from_rig = Some(Rigid3::from_se3(apply_pose_delta_f64(current, delta)));
    true
}

pub(crate) fn sync_pose_blocks_for_sensor_changes(
    reconstruction: &mut Reconstruction,
    pose_blocks: &PoseBlockSet,
    changed_sensors: &[SensorPoseKey],
) {
    let changed_sensors = changed_sensors.iter().collect::<HashSet<_>>();
    for block in &pose_blocks.blocks {
        let PoseBlockKind::Frame(frame_idx) = block.kind else {
            continue;
        };
        let frame_uses_changed_sensor = block
            .images
            .iter()
            .filter_map(|&image| frame_sensor_key_for_image(reconstruction, image))
            .any(|key| changed_sensors.contains(&key));
        if frame_uses_changed_sensor {
            let Some(frame) = reconstruction.frames.get(frame_idx) else {
                continue;
            };
            set_frame_pose_block(
                reconstruction,
                frame_idx,
                &block.images,
                frame.rig_from_world.to_se3(),
            );
        }
    }
}

pub(crate) fn projection_jacobians(
    camera: CameraModel,
    pose: SE3,
    point: [f32; 3],
) -> Option<(Mat2x6, Mat2x3)> {
    analytic_projection_jacobians(camera, pose, point).or_else(|| {
        Some((
            numerical_pose_jacobian(camera, pose, point)?,
            numerical_point_jacobian(camera, pose, point)?,
        ))
    })
}

fn residual_and_jacobians(
    camera: CameraModel,
    pose: SE3,
    point: [f32; 3],
    xy: [f64; 2],
) -> Option<(Vec2, Mat2x6, Mat2x3)> {
    let predicted = project_point(camera, pose, point)?;
    let residual = Vec2::new(predicted[0] - xy[0], predicted[1] - xy[1]);
    if !residual.iter().all(|v| v.is_finite()) {
        return None;
    }
    let (j_pose, j_point) = analytic_projection_jacobians(camera, pose, point).unwrap_or((
        numerical_pose_jacobian(camera, pose, point)?,
        numerical_point_jacobian(camera, pose, point)?,
    ));
    Some((residual, j_pose, j_point))
}

fn analytic_projection_jacobians(
    camera: CameraModel,
    pose: SE3,
    point: [f32; 3],
) -> Option<(Mat2x6, Mat2x3)> {
    let cam_point = pose.transform_point(&point);
    let x = cam_point[0] as f64;
    let y = cam_point[1] as f64;
    let z = cam_point[2] as f64;
    let j_cam = analytic_img_from_cam_jacobian(camera, x, y, z)?;
    if z <= f64::EPSILON || ![x, y, z].iter().all(|v| v.is_finite()) {
        return None;
    }

    let translation = pose.translation();
    let rx = x - translation[0] as f64;
    let ry = y - translation[1] as f64;
    let rz = z - translation[2] as f64;

    // COLMAP/Ceres stores pose as separate rotation and translation parameter
    // blocks: R <- delta_R * R, t <- t + delta_t.
    let dcam_dpose = [
        [0.0, rz, -ry, 1.0, 0.0, 0.0],
        [-rz, 0.0, rx, 0.0, 1.0, 0.0],
        [ry, -rx, 0.0, 0.0, 0.0, 1.0],
    ];
    let mut j_pose = Mat2x6::zeros();
    for col in 0..6 {
        j_pose[(0, col)] = j_cam[(0, 0)] * dcam_dpose[0][col]
            + j_cam[(0, 1)] * dcam_dpose[1][col]
            + j_cam[(0, 2)] * dcam_dpose[2][col];
        j_pose[(1, col)] = j_cam[(1, 0)] * dcam_dpose[0][col]
            + j_cam[(1, 1)] * dcam_dpose[1][col]
            + j_cam[(1, 2)] * dcam_dpose[2][col];
    }

    let rotation = pose.rotation_matrix();
    let mut j_point = Mat2x3::zeros();
    for col in 0..3 {
        j_point[(0, col)] = j_cam[(0, 0)] * rotation[0][col] as f64
            + j_cam[(0, 1)] * rotation[1][col] as f64
            + j_cam[(0, 2)] * rotation[2][col] as f64;
        j_point[(1, col)] = j_cam[(1, 0)] * rotation[0][col] as f64
            + j_cam[(1, 1)] * rotation[1][col] as f64
            + j_cam[(1, 2)] * rotation[2][col] as f64;
    }

    Some((j_pose, j_point))
}

pub(crate) fn analytic_img_from_cam_jacobian(
    camera: CameraModel,
    x: f64,
    y: f64,
    z: f64,
) -> Option<Mat2x3> {
    if z <= f64::EPSILON || ![x, y, z].iter().all(|v| v.is_finite()) {
        return None;
    }
    let inv_z = 1.0 / z;
    let inv_z2 = inv_z * inv_z;
    let u = x * inv_z;
    let v = y * inv_z;
    let du_dcam = [inv_z, 0.0, -x * inv_z2];
    let dv_dcam = [0.0, inv_z, -y * inv_z2];

    let mut j_norm = SMatrix::<f64, 2, 2>::zeros();
    match camera.model_id {
        COLMAP_SIMPLE_PINHOLE => {
            let f = camera.params[0];
            j_norm[(0, 0)] = f;
            j_norm[(1, 1)] = f;
        }
        COLMAP_PINHOLE => {
            j_norm[(0, 0)] = camera.params[0];
            j_norm[(1, 1)] = camera.params[1];
        }
        COLMAP_SIMPLE_RADIAL | COLMAP_RADIAL => {
            let f = camera.params[0];
            let k1 = camera.params[3];
            let k2 = if camera.model_id == COLMAP_RADIAL {
                camera.params[4]
            } else {
                0.0
            };
            let r2 = u * u + v * v;
            let radial = 1.0 + k1 * r2 + k2 * r2 * r2;
            let radial_derivative = k1 + 2.0 * k2 * r2;
            j_norm[(0, 0)] = f * (radial + 2.0 * u * u * radial_derivative);
            j_norm[(0, 1)] = f * (2.0 * u * v * radial_derivative);
            j_norm[(1, 0)] = f * (2.0 * u * v * radial_derivative);
            j_norm[(1, 1)] = f * (radial + 2.0 * v * v * radial_derivative);
        }
        COLMAP_OPENCV => {
            let fx = camera.params[0];
            let fy = camera.params[1];
            let k1 = camera.params[4];
            let k2 = camera.params[5];
            let p1 = camera.params[6];
            let p2 = camera.params[7];
            let u2 = u * u;
            let v2 = v * v;
            let r2 = u2 + v2;
            let radial = 1.0 + k1 * r2 + k2 * r2 * r2;
            let radial_derivative = k1 + 2.0 * k2 * r2;
            let dx_du = radial + 2.0 * u2 * radial_derivative + 2.0 * p1 * v + 6.0 * p2 * u;
            let dx_dv = 2.0 * u * v * radial_derivative + 2.0 * p1 * u + 2.0 * p2 * v;
            let dy_du = 2.0 * u * v * radial_derivative + 2.0 * p2 * v + 2.0 * p1 * u;
            let dy_dv = radial + 2.0 * v2 * radial_derivative + 6.0 * p1 * v + 2.0 * p2 * u;
            j_norm[(0, 0)] = fx * dx_du;
            j_norm[(0, 1)] = fx * dx_dv;
            j_norm[(1, 0)] = fy * dy_du;
            j_norm[(1, 1)] = fy * dy_dv;
        }
        COLMAP_FULL_OPENCV => {
            let fx = camera.params[0];
            let fy = camera.params[1];
            let k1 = camera.params[4];
            let k2 = camera.params[5];
            let p1 = camera.params[6];
            let p2 = camera.params[7];
            let k3 = camera.params[8];
            let k4 = camera.params[9];
            let k5 = camera.params[10];
            let k6 = camera.params[11];
            let u2 = u * u;
            let v2 = v * v;
            let terms = full_opencv_radial_terms(u, v, k1, k2, k3, k4, k5, k6)?;
            let dx_du =
                terms.radial + 2.0 * u2 * terms.radial_derivative + 2.0 * p1 * v + 6.0 * p2 * u;
            let dx_dv = 2.0 * u * v * terms.radial_derivative + 2.0 * p1 * u + 2.0 * p2 * v;
            let dy_du = 2.0 * u * v * terms.radial_derivative + 2.0 * p2 * v + 2.0 * p1 * u;
            let dy_dv =
                terms.radial + 2.0 * v2 * terms.radial_derivative + 6.0 * p1 * v + 2.0 * p2 * u;
            j_norm[(0, 0)] = fx * dx_du;
            j_norm[(0, 1)] = fx * dx_dv;
            j_norm[(1, 0)] = fy * dy_du;
            j_norm[(1, 1)] = fy * dy_dv;
        }
        COLMAP_FOV => {
            let fx = camera.params[0];
            let fy = camera.params[1];
            let terms = fov_distortion_terms(camera.params[4], u, v)?;
            j_norm[(0, 0)] = fx * (terms.factor + 2.0 * u * u * terms.factor_derivative_r2);
            j_norm[(0, 1)] = fx * (2.0 * u * v * terms.factor_derivative_r2);
            j_norm[(1, 0)] = fy * (2.0 * u * v * terms.factor_derivative_r2);
            j_norm[(1, 1)] = fy * (terms.factor + 2.0 * v * v * terms.factor_derivative_r2);
        }
        COLMAP_SIMPLE_FISHEYE | COLMAP_FISHEYE => {
            let fx = camera.params[0];
            let fy = if camera.model_id == COLMAP_SIMPLE_FISHEYE {
                camera.params[0]
            } else {
                camera.params[1]
            };
            let fisheye = fisheye_normal_terms(u, v)?;
            j_norm[(0, 0)] = fx * fisheye.jacobian[(0, 0)];
            j_norm[(0, 1)] = fx * fisheye.jacobian[(0, 1)];
            j_norm[(1, 0)] = fy * fisheye.jacobian[(1, 0)];
            j_norm[(1, 1)] = fy * fisheye.jacobian[(1, 1)];
        }
        COLMAP_SIMPLE_RADIAL_FISHEYE | COLMAP_RADIAL_FISHEYE | COLMAP_OPENCV_FISHEYE => {
            let fx = camera.params[0];
            let fy = match camera.model_id {
                COLMAP_OPENCV_FISHEYE => camera.params[1],
                _ => camera.params[0],
            };
            let fisheye = fisheye_normal_terms(u, v)?;
            let terms = fisheye_radial_terms(camera, fisheye.u, fisheye.v)?;
            let dx_duu = terms.radial + 2.0 * fisheye.u * fisheye.u * terms.radial_derivative;
            let dx_dvv = 2.0 * fisheye.u * fisheye.v * terms.radial_derivative;
            let dy_duu = dx_dvv;
            let dy_dvv = terms.radial + 2.0 * fisheye.v * fisheye.v * terms.radial_derivative;
            j_norm[(0, 0)] =
                fx * (dx_duu * fisheye.jacobian[(0, 0)] + dx_dvv * fisheye.jacobian[(1, 0)]);
            j_norm[(0, 1)] =
                fx * (dx_duu * fisheye.jacobian[(0, 1)] + dx_dvv * fisheye.jacobian[(1, 1)]);
            j_norm[(1, 0)] =
                fy * (dy_duu * fisheye.jacobian[(0, 0)] + dy_dvv * fisheye.jacobian[(1, 0)]);
            j_norm[(1, 1)] =
                fy * (dy_duu * fisheye.jacobian[(0, 1)] + dy_dvv * fisheye.jacobian[(1, 1)]);
        }
        COLMAP_THIN_PRISM_FISHEYE | COLMAP_RAD_TAN_THIN_PRISM_FISHEYE => {
            let fx = camera.params[0];
            let fy = camera.params[1];
            let fisheye = fisheye_normal_terms(u, v)?;
            let distortion = fisheye_distortion_terms(camera, fisheye.u, fisheye.v)?;
            let j_total = Mat2::identity() + distortion.jacobian;
            j_norm[(0, 0)] = fx
                * (j_total[(0, 0)] * fisheye.jacobian[(0, 0)]
                    + j_total[(0, 1)] * fisheye.jacobian[(1, 0)]);
            j_norm[(0, 1)] = fx
                * (j_total[(0, 0)] * fisheye.jacobian[(0, 1)]
                    + j_total[(0, 1)] * fisheye.jacobian[(1, 1)]);
            j_norm[(1, 0)] = fy
                * (j_total[(1, 0)] * fisheye.jacobian[(0, 0)]
                    + j_total[(1, 1)] * fisheye.jacobian[(1, 0)]);
            j_norm[(1, 1)] = fy
                * (j_total[(1, 0)] * fisheye.jacobian[(0, 1)]
                    + j_total[(1, 1)] * fisheye.jacobian[(1, 1)]);
        }
        COLMAP_SIMPLE_DIVISION | COLMAP_DIVISION => {
            let fx = camera.params[0];
            let fy = if camera.model_id == COLMAP_SIMPLE_DIVISION {
                camera.params[0]
            } else {
                camera.params[1]
            };
            let k = if camera.model_id == COLMAP_SIMPLE_DIVISION {
                camera.params[3]
            } else {
                camera.params[4]
            };
            let terms = division_projection_terms(x, y, z, k)?;
            return Some(Mat2x3::from_row_slice(&[
                fx * terms.j_cam[(0, 0)],
                fx * terms.j_cam[(0, 1)],
                fx * terms.j_cam[(0, 2)],
                fy * terms.j_cam[(1, 0)],
                fy * terms.j_cam[(1, 1)],
                fy * terms.j_cam[(1, 2)],
            ]));
        }
        COLMAP_EUCM => {
            let fx = camera.params[0];
            let fy = camera.params[1];
            let terms = eucm_projection_terms(x, y, z, camera.params[4], camera.params[5])?;
            return Some(Mat2x3::from_row_slice(&[
                fx * terms.j_cam[(0, 0)],
                fx * terms.j_cam[(0, 1)],
                fx * terms.j_cam[(0, 2)],
                fy * terms.j_cam[(1, 0)],
                fy * terms.j_cam[(1, 1)],
                fy * terms.j_cam[(1, 2)],
            ]));
        }
        _ => return None,
    }
    if !j_norm.iter().all(|value| value.is_finite()) {
        return None;
    }

    let mut j_cam = Mat2x3::zeros();
    for col in 0..3 {
        j_cam[(0, col)] = j_norm[(0, 0)] * du_dcam[col] + j_norm[(0, 1)] * dv_dcam[col];
        j_cam[(1, col)] = j_norm[(1, 0)] * du_dcam[col] + j_norm[(1, 1)] * dv_dcam[col];
    }
    Some(j_cam)
}

fn numerical_pose_jacobian(camera: CameraModel, pose: SE3, point: [f32; 3]) -> Option<Mat2x6> {
    let mut jacobian = Mat2x6::zeros();
    let eps = [1.0e-4; 6];
    for axis in 0..6 {
        let mut plus = Vec6::zeros();
        plus[axis] = eps[axis];
        let mut minus = Vec6::zeros();
        minus[axis] = -eps[axis];
        let p_plus = project_point(camera, apply_pose_delta_f64(pose, plus), point)?;
        let p_minus = project_point(camera, apply_pose_delta_f64(pose, minus), point)?;
        jacobian[(0, axis)] = (p_plus[0] - p_minus[0]) / (2.0 * eps[axis]);
        jacobian[(1, axis)] = (p_plus[1] - p_minus[1]) / (2.0 * eps[axis]);
    }
    Some(jacobian)
}

fn numerical_point_jacobian(camera: CameraModel, pose: SE3, point: [f32; 3]) -> Option<Mat2x3> {
    let mut jacobian = Mat2x3::zeros();
    let eps = 1.0e-4;
    for axis in 0..3 {
        let mut plus = point;
        let mut minus = point;
        plus[axis] += eps as f32;
        minus[axis] -= eps as f32;
        let p_plus = project_point(camera, pose, plus)?;
        let p_minus = project_point(camera, pose, minus)?;
        jacobian[(0, axis)] = (p_plus[0] - p_minus[0]) / (2.0 * eps);
        jacobian[(1, axis)] = (p_plus[1] - p_minus[1]) / (2.0 * eps);
    }
    Some(jacobian)
}

pub(crate) fn camera_param_jacobian(
    camera: CameraModel,
    param: usize,
    pose: SE3,
    point: [f32; 3],
) -> Option<Vec2> {
    analytic_camera_param_jacobian(camera, param, pose, point)
        .or_else(|| finite_difference_camera_param_jacobian(camera, param, pose, point))
}

fn analytic_camera_param_jacobian(
    camera: CameraModel,
    param: usize,
    pose: SE3,
    point: [f32; 3],
) -> Option<Vec2> {
    if param >= camera.num_params {
        return None;
    }
    let cam_point = pose.transform_point(&point);
    let x = cam_point[0] as f64;
    let y = cam_point[1] as f64;
    let z = cam_point[2] as f64;
    if z <= f64::EPSILON || ![x, y, z].iter().all(|v| v.is_finite()) {
        return None;
    }
    let nx = x / z;
    let ny = y / z;
    let r2 = nx * nx + ny * ny;
    match (camera.model_id, param) {
        (COLMAP_SIMPLE_PINHOLE, 0) => Some(Vec2::new(nx, ny)),
        (COLMAP_SIMPLE_PINHOLE, 1) => Some(Vec2::new(1.0, 0.0)),
        (COLMAP_SIMPLE_PINHOLE, 2) => Some(Vec2::new(0.0, 1.0)),
        (COLMAP_PINHOLE, 0) => Some(Vec2::new(nx, 0.0)),
        (COLMAP_PINHOLE, 1) => Some(Vec2::new(0.0, ny)),
        (COLMAP_PINHOLE, 2) => Some(Vec2::new(1.0, 0.0)),
        (COLMAP_PINHOLE, 3) => Some(Vec2::new(0.0, 1.0)),
        (COLMAP_SIMPLE_RADIAL, 0) => {
            let radial = 1.0 + camera.params[3] * r2;
            Some(Vec2::new(nx * radial, ny * radial))
        }
        (COLMAP_SIMPLE_RADIAL, 1) => Some(Vec2::new(1.0, 0.0)),
        (COLMAP_SIMPLE_RADIAL, 2) => Some(Vec2::new(0.0, 1.0)),
        (COLMAP_SIMPLE_RADIAL, 3) => {
            let f = camera.params[0];
            Some(Vec2::new(f * nx * r2, f * ny * r2))
        }
        (COLMAP_RADIAL, 0) => {
            let radial = 1.0 + camera.params[3] * r2 + camera.params[4] * r2 * r2;
            Some(Vec2::new(nx * radial, ny * radial))
        }
        (COLMAP_RADIAL, 1) => Some(Vec2::new(1.0, 0.0)),
        (COLMAP_RADIAL, 2) => Some(Vec2::new(0.0, 1.0)),
        (COLMAP_RADIAL, 3) => {
            let f = camera.params[0];
            Some(Vec2::new(f * nx * r2, f * ny * r2))
        }
        (COLMAP_RADIAL, 4) => {
            let f = camera.params[0];
            Some(Vec2::new(f * nx * r2 * r2, f * ny * r2 * r2))
        }
        (COLMAP_OPENCV, 0) => {
            let k1 = camera.params[4];
            let k2 = camera.params[5];
            let p1 = camera.params[6];
            let p2 = camera.params[7];
            let distorted = opencv_distorted_normal(nx, ny, k1, k2, p1, p2);
            Some(Vec2::new(distorted[0], 0.0))
        }
        (COLMAP_OPENCV, 1) => {
            let k1 = camera.params[4];
            let k2 = camera.params[5];
            let p1 = camera.params[6];
            let p2 = camera.params[7];
            let distorted = opencv_distorted_normal(nx, ny, k1, k2, p1, p2);
            Some(Vec2::new(0.0, distorted[1]))
        }
        (COLMAP_OPENCV, 2) => Some(Vec2::new(1.0, 0.0)),
        (COLMAP_OPENCV, 3) => Some(Vec2::new(0.0, 1.0)),
        (COLMAP_OPENCV, 4) => Some(Vec2::new(
            camera.params[0] * nx * r2,
            camera.params[1] * ny * r2,
        )),
        (COLMAP_OPENCV, 5) => Some(Vec2::new(
            camera.params[0] * nx * r2 * r2,
            camera.params[1] * ny * r2 * r2,
        )),
        (COLMAP_OPENCV, 6) => Some(Vec2::new(
            camera.params[0] * 2.0 * nx * ny,
            camera.params[1] * (r2 + 2.0 * ny * ny),
        )),
        (COLMAP_OPENCV, 7) => Some(Vec2::new(
            camera.params[0] * (r2 + 2.0 * nx * nx),
            camera.params[1] * 2.0 * nx * ny,
        )),
        (COLMAP_FOV, 0..=4) => {
            let terms = fov_distortion_terms(camera.params[4], nx, ny)?;
            match param {
                0 => Some(Vec2::new(terms.x, 0.0)),
                1 => Some(Vec2::new(0.0, terms.y)),
                2 => Some(Vec2::new(1.0, 0.0)),
                3 => Some(Vec2::new(0.0, 1.0)),
                4 => Some(Vec2::new(
                    camera.params[0] * nx * terms.factor_derivative_omega,
                    camera.params[1] * ny * terms.factor_derivative_omega,
                )),
                _ => None,
            }
        }
        (COLMAP_SIMPLE_FISHEYE, 0..=2) => {
            let fisheye = fisheye_normal_terms(nx, ny)?;
            match param {
                0 => Some(Vec2::new(fisheye.u, fisheye.v)),
                1 => Some(Vec2::new(1.0, 0.0)),
                2 => Some(Vec2::new(0.0, 1.0)),
                _ => None,
            }
        }
        (COLMAP_FISHEYE, 0..=3) => {
            let fisheye = fisheye_normal_terms(nx, ny)?;
            match param {
                0 => Some(Vec2::new(fisheye.u, 0.0)),
                1 => Some(Vec2::new(0.0, fisheye.v)),
                2 => Some(Vec2::new(1.0, 0.0)),
                3 => Some(Vec2::new(0.0, 1.0)),
                _ => None,
            }
        }
        (COLMAP_SIMPLE_RADIAL_FISHEYE, 0..=3) => {
            let fisheye = fisheye_normal_terms(nx, ny)?;
            let terms = fisheye_radial_terms(camera, fisheye.u, fisheye.v)?;
            let distorted_u = fisheye.u * terms.radial;
            let distorted_v = fisheye.v * terms.radial;
            match param {
                0 => Some(Vec2::new(distorted_u, distorted_v)),
                1 => Some(Vec2::new(1.0, 0.0)),
                2 => Some(Vec2::new(0.0, 1.0)),
                3 => Some(Vec2::new(
                    camera.params[0] * fisheye.u * terms.r2,
                    camera.params[0] * fisheye.v * terms.r2,
                )),
                _ => None,
            }
        }
        (COLMAP_RADIAL_FISHEYE, 0..=4) => {
            let fisheye = fisheye_normal_terms(nx, ny)?;
            let terms = fisheye_radial_terms(camera, fisheye.u, fisheye.v)?;
            let distorted_u = fisheye.u * terms.radial;
            let distorted_v = fisheye.v * terms.radial;
            match param {
                0 => Some(Vec2::new(distorted_u, distorted_v)),
                1 => Some(Vec2::new(1.0, 0.0)),
                2 => Some(Vec2::new(0.0, 1.0)),
                3 => Some(Vec2::new(
                    camera.params[0] * fisheye.u * terms.r2,
                    camera.params[0] * fisheye.v * terms.r2,
                )),
                4 => Some(Vec2::new(
                    camera.params[0] * fisheye.u * terms.r4,
                    camera.params[0] * fisheye.v * terms.r4,
                )),
                _ => None,
            }
        }
        (COLMAP_OPENCV_FISHEYE, 0..=7) => {
            let fisheye = fisheye_normal_terms(nx, ny)?;
            let terms = fisheye_radial_terms(camera, fisheye.u, fisheye.v)?;
            let distorted_u = fisheye.u * terms.radial;
            let distorted_v = fisheye.v * terms.radial;
            match param {
                0 => Some(Vec2::new(distorted_u, 0.0)),
                1 => Some(Vec2::new(0.0, distorted_v)),
                2 => Some(Vec2::new(1.0, 0.0)),
                3 => Some(Vec2::new(0.0, 1.0)),
                4 => Some(Vec2::new(
                    camera.params[0] * fisheye.u * terms.r2,
                    camera.params[1] * fisheye.v * terms.r2,
                )),
                5 => Some(Vec2::new(
                    camera.params[0] * fisheye.u * terms.r4,
                    camera.params[1] * fisheye.v * terms.r4,
                )),
                6 => Some(Vec2::new(
                    camera.params[0] * fisheye.u * terms.r6,
                    camera.params[1] * fisheye.v * terms.r6,
                )),
                7 => Some(Vec2::new(
                    camera.params[0] * fisheye.u * terms.r8,
                    camera.params[1] * fisheye.v * terms.r8,
                )),
                _ => None,
            }
        }
        (COLMAP_THIN_PRISM_FISHEYE, 0..=11) => {
            let fx = camera.params[0];
            let fy = camera.params[1];
            let fisheye = fisheye_normal_terms(nx, ny)?;
            let terms = fisheye_distortion_terms(camera, fisheye.u, fisheye.v)?;
            let r2 = fisheye.u * fisheye.u + fisheye.v * fisheye.v;
            let r4 = r2 * r2;
            let r6 = r4 * r2;
            let r8 = r4 * r4;
            match param {
                0 => Some(Vec2::new(terms.x, 0.0)),
                1 => Some(Vec2::new(0.0, terms.y)),
                2 => Some(Vec2::new(1.0, 0.0)),
                3 => Some(Vec2::new(0.0, 1.0)),
                4 => Some(Vec2::new(fx * fisheye.u * r2, fy * fisheye.v * r2)),
                5 => Some(Vec2::new(fx * fisheye.u * r4, fy * fisheye.v * r4)),
                6 => Some(Vec2::new(
                    fx * 2.0 * fisheye.u * fisheye.v,
                    fy * (r2 + 2.0 * fisheye.v * fisheye.v),
                )),
                7 => Some(Vec2::new(
                    fx * (r2 + 2.0 * fisheye.u * fisheye.u),
                    fy * 2.0 * fisheye.u * fisheye.v,
                )),
                8 => Some(Vec2::new(fx * fisheye.u * r6, fy * fisheye.v * r6)),
                9 => Some(Vec2::new(fx * fisheye.u * r8, fy * fisheye.v * r8)),
                10 => Some(Vec2::new(fx * r2, 0.0)),
                11 => Some(Vec2::new(0.0, fy * r2)),
                _ => None,
            }
        }
        (COLMAP_RAD_TAN_THIN_PRISM_FISHEYE, 0..=15) => {
            let fx = camera.params[0];
            let fy = camera.params[1];
            let fisheye = fisheye_normal_terms(nx, ny)?;
            let terms = fisheye_distortion_terms(camera, fisheye.u, fisheye.v)?;
            let theta2 = fisheye.u * fisheye.u + fisheye.v * fisheye.v;
            let mut th_radial = 1.0;
            let mut theta_power = 1.0;
            for coeff in &camera.params[4..10] {
                theta_power *= theta2;
                th_radial += coeff * theta_power;
            }
            let x_dist = th_radial * fisheye.u;
            let y_dist = th_radial * fisheye.v;
            let x2 = x_dist * x_dist;
            let y2 = y_dist * y_dist;
            let xy = x_dist * y_dist;
            let r2_dist = x2 + y2;
            let r4_dist = r2_dist * r2_dist;
            match param {
                0 => Some(Vec2::new(terms.x, 0.0)),
                1 => Some(Vec2::new(0.0, terms.y)),
                2 => Some(Vec2::new(1.0, 0.0)),
                3 => Some(Vec2::new(0.0, 1.0)),
                4..=9 => {
                    let intermediate =
                        rad_tan_thin_prism_intermediate_terms(camera, fisheye.u, fisheye.v)?;
                    let power = theta2.powi((param - 3) as i32);
                    let radial_du = intermediate.j_xy[(0, 0)] * fisheye.u * power
                        + intermediate.j_xy[(0, 1)] * fisheye.v * power;
                    let radial_dv = intermediate.j_xy[(1, 0)] * fisheye.u * power
                        + intermediate.j_xy[(1, 1)] * fisheye.v * power;
                    Some(Vec2::new(fx * radial_du, fy * radial_dv))
                }
                10 => Some(Vec2::new(fx * (r2_dist + 2.0 * x2), fy * 2.0 * xy)),
                11 => Some(Vec2::new(fx * 2.0 * xy, fy * (r2_dist + 2.0 * y2))),
                12 => Some(Vec2::new(fx * r2_dist, 0.0)),
                13 => Some(Vec2::new(fx * r4_dist, 0.0)),
                14 => Some(Vec2::new(0.0, fy * r2_dist)),
                15 => Some(Vec2::new(0.0, fy * r4_dist)),
                _ => None,
            }
        }
        (COLMAP_SIMPLE_DIVISION, 0..=3) => {
            let terms = division_projection_terms(x, y, z, camera.params[3])?;
            match param {
                0 => Some(Vec2::new(terms.x, terms.y)),
                1 => Some(Vec2::new(1.0, 0.0)),
                2 => Some(Vec2::new(0.0, 1.0)),
                3 => {
                    let q = x * x + y * y;
                    let dscale_dk = 4.0 * q / (terms.disc_sqrt * (z + terms.disc_sqrt).powi(2));
                    Some(Vec2::new(
                        camera.params[0] * x * dscale_dk,
                        camera.params[0] * y * dscale_dk,
                    ))
                }
                _ => None,
            }
        }
        (COLMAP_DIVISION, 0..=4) => {
            let terms = division_projection_terms(x, y, z, camera.params[4])?;
            match param {
                0 => Some(Vec2::new(terms.x, 0.0)),
                1 => Some(Vec2::new(0.0, terms.y)),
                2 => Some(Vec2::new(1.0, 0.0)),
                3 => Some(Vec2::new(0.0, 1.0)),
                4 => {
                    let q = x * x + y * y;
                    let dscale_dk = 4.0 * q / (terms.disc_sqrt * (z + terms.disc_sqrt).powi(2));
                    Some(Vec2::new(
                        camera.params[0] * x * dscale_dk,
                        camera.params[1] * y * dscale_dk,
                    ))
                }
                _ => None,
            }
        }
        (COLMAP_EUCM, 0..=5) => {
            let alpha = camera.params[4];
            let beta = camera.params[5];
            let terms = eucm_projection_terms(x, y, z, alpha, beta)?;
            match param {
                0 => Some(Vec2::new(terms.x, 0.0)),
                1 => Some(Vec2::new(0.0, terms.y)),
                2 => Some(Vec2::new(1.0, 0.0)),
                3 => Some(Vec2::new(0.0, 1.0)),
                4 => {
                    let dden_dalpha = terms.rho - z;
                    Some(Vec2::new(
                        -camera.params[0] * x * dden_dalpha / (terms.den * terms.den),
                        -camera.params[1] * y * dden_dalpha / (terms.den * terms.den),
                    ))
                }
                5 => {
                    let q = x * x + y * y;
                    let dden_dbeta = alpha * q / (2.0 * terms.rho);
                    Some(Vec2::new(
                        -camera.params[0] * x * dden_dbeta / (terms.den * terms.den),
                        -camera.params[1] * y * dden_dbeta / (terms.den * terms.den),
                    ))
                }
                _ => None,
            }
        }
        (COLMAP_FULL_OPENCV, 0..=11) => {
            let fx = camera.params[0];
            let fy = camera.params[1];
            let k1 = camera.params[4];
            let k2 = camera.params[5];
            let p1 = camera.params[6];
            let p2 = camera.params[7];
            let k3 = camera.params[8];
            let k4 = camera.params[9];
            let k5 = camera.params[10];
            let k6 = camera.params[11];
            let terms = full_opencv_radial_terms(nx, ny, k1, k2, k3, k4, k5, k6)?;
            let distorted = opencv_distorted_normal_from_radial(nx, ny, terms.radial, p1, p2);
            match param {
                0 => Some(Vec2::new(distorted[0], 0.0)),
                1 => Some(Vec2::new(0.0, distorted[1])),
                2 => Some(Vec2::new(1.0, 0.0)),
                3 => Some(Vec2::new(0.0, 1.0)),
                4 => Some(Vec2::new(
                    fx * nx * terms.r2 / terms.den,
                    fy * ny * terms.r2 / terms.den,
                )),
                5 => Some(Vec2::new(
                    fx * nx * terms.r4 / terms.den,
                    fy * ny * terms.r4 / terms.den,
                )),
                6 => Some(Vec2::new(fx * 2.0 * nx * ny, fy * (r2 + 2.0 * ny * ny))),
                7 => Some(Vec2::new(fx * (r2 + 2.0 * nx * nx), fy * 2.0 * nx * ny)),
                8 => Some(Vec2::new(
                    fx * nx * terms.r6 / terms.den,
                    fy * ny * terms.r6 / terms.den,
                )),
                9 => {
                    let scale = -terms.num * terms.r2 / (terms.den * terms.den);
                    Some(Vec2::new(fx * nx * scale, fy * ny * scale))
                }
                10 => {
                    let scale = -terms.num * terms.r4 / (terms.den * terms.den);
                    Some(Vec2::new(fx * nx * scale, fy * ny * scale))
                }
                11 => {
                    let scale = -terms.num * terms.r6 / (terms.den * terms.den);
                    Some(Vec2::new(fx * nx * scale, fy * ny * scale))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn opencv_distorted_normal(u: f64, v: f64, k1: f64, k2: f64, p1: f64, p2: f64) -> [f64; 2] {
    let r2 = u * u + v * v;
    let radial = 1.0 + k1 * r2 + k2 * r2 * r2;
    opencv_distorted_normal_from_radial(u, v, radial, p1, p2)
}

fn opencv_distorted_normal_from_radial(u: f64, v: f64, radial: f64, p1: f64, p2: f64) -> [f64; 2] {
    let u2 = u * u;
    let uv = u * v;
    let v2 = v * v;
    let r2 = u2 + v2;
    [
        u * radial + 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2),
        v * radial + 2.0 * p2 * uv + p1 * (r2 + 2.0 * v2),
    ]
}

struct FullOpenCvRadialTerms {
    r2: f64,
    r4: f64,
    r6: f64,
    num: f64,
    den: f64,
    radial: f64,
    radial_derivative: f64,
}

fn full_opencv_radial_terms(
    u: f64,
    v: f64,
    k1: f64,
    k2: f64,
    k3: f64,
    k4: f64,
    k5: f64,
    k6: f64,
) -> Option<FullOpenCvRadialTerms> {
    let r2 = u * u + v * v;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let num = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
    let den = 1.0 + k4 * r2 + k5 * r4 + k6 * r6;
    if den.abs() <= f64::EPSILON {
        return None;
    }
    let radial = num / den;
    let dnum_dr2 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;
    let dden_dr2 = k4 + 2.0 * k5 * r2 + 3.0 * k6 * r4;
    let radial_derivative = (dnum_dr2 * den - num * dden_dr2) / (den * den);
    let terms = FullOpenCvRadialTerms {
        r2,
        r4,
        r6,
        num,
        den,
        radial,
        radial_derivative,
    };
    if [
        terms.r2,
        terms.r4,
        terms.r6,
        terms.num,
        terms.den,
        terms.radial,
        terms.radial_derivative,
    ]
    .iter()
    .all(|value| value.is_finite())
    {
        Some(terms)
    } else {
        None
    }
}

struct FisheyeNormalTerms {
    u: f64,
    v: f64,
    jacobian: Mat2,
}

fn fisheye_normal_terms(u: f64, v: f64) -> Option<FisheyeNormalTerms> {
    let r2 = u * u + v * v;
    let r = r2.sqrt();
    let mut uu = u;
    let mut vv = v;
    let mut jacobian = Mat2::identity();
    if r > f64::EPSILON {
        let theta = r.atan();
        let scale = theta / r;
        uu *= scale;
        vv *= scale;
        let dscale_dr = (r / (1.0 + r2) - theta) / r2;
        let dscale_du = dscale_dr * u / r;
        let dscale_dv = dscale_dr * v / r;
        jacobian[(0, 0)] = scale + u * dscale_du;
        jacobian[(0, 1)] = u * dscale_dv;
        jacobian[(1, 0)] = v * dscale_du;
        jacobian[(1, 1)] = scale + v * dscale_dv;
    }
    if [uu, vv].iter().all(|value| value.is_finite())
        && jacobian.iter().all(|value| value.is_finite())
    {
        Some(FisheyeNormalTerms {
            u: uu,
            v: vv,
            jacobian,
        })
    } else {
        None
    }
}

struct FisheyeRadialTerms {
    r2: f64,
    r4: f64,
    r6: f64,
    r8: f64,
    radial: f64,
    radial_derivative: f64,
}

fn fisheye_radial_terms(camera: CameraModel, u: f64, v: f64) -> Option<FisheyeRadialTerms> {
    let r2 = u * u + v * v;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let r8 = r4 * r4;
    let (k1, k2, k3, k4) = match camera.model_id {
        COLMAP_SIMPLE_RADIAL_FISHEYE => (camera.params[3], 0.0, 0.0, 0.0),
        COLMAP_RADIAL_FISHEYE => (camera.params[3], camera.params[4], 0.0, 0.0),
        COLMAP_OPENCV_FISHEYE => (
            camera.params[4],
            camera.params[5],
            camera.params[6],
            camera.params[7],
        ),
        _ => return None,
    };
    let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
    let radial_derivative = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4 + 4.0 * k4 * r6;
    let terms = FisheyeRadialTerms {
        r2,
        r4,
        r6,
        r8,
        radial,
        radial_derivative,
    };
    if [
        terms.r2,
        terms.r4,
        terms.r6,
        terms.r8,
        terms.radial,
        terms.radial_derivative,
    ]
    .iter()
    .all(|value| value.is_finite())
    {
        Some(terms)
    } else {
        None
    }
}

struct FovDistortionTerms {
    x: f64,
    y: f64,
    factor: f64,
    factor_derivative_r2: f64,
    factor_derivative_omega: f64,
}

fn fov_distortion_terms(omega: f64, u: f64, v: f64) -> Option<FovDistortionTerms> {
    const EPSILON: f64 = 1.0e-4;
    let r2 = u * u + v * v;
    let omega2 = omega * omega;
    let (factor, factor_derivative_r2, factor_derivative_omega) = if omega2 < EPSILON {
        (
            omega2 * r2 / 3.0 - omega2 / 12.0 + 1.0,
            omega2 / 3.0,
            2.0 * omega * (r2 / 3.0 - 1.0 / 12.0),
        )
    } else if r2 < EPSILON {
        let t = (omega / 2.0).tan();
        let dt_domega = 0.5 * (1.0 + t * t);
        let factor = -2.0 * t * (4.0 * r2 * t * t - 3.0) / (3.0 * omega);
        let factor_derivative_r2 = -8.0 * t * t * t / (3.0 * omega);
        let numerator = -8.0 * r2 * t * t * t + 6.0 * t;
        let numerator_derivative = dt_domega * (6.0 - 24.0 * r2 * t * t);
        let factor_derivative_omega =
            (numerator_derivative * omega - numerator) / (3.0 * omega * omega);
        (factor, factor_derivative_r2, factor_derivative_omega)
    } else {
        let r = r2.sqrt();
        let t = (omega / 2.0).tan();
        let a = 2.0 * r * t;
        let numerator = a.atan();
        let den = r * omega;
        let factor = numerator / den;
        let dnum_dr = 2.0 * t / (1.0 + a * a);
        let dfactor_dr = (dnum_dr * den - numerator * omega) / (den * den);
        let factor_derivative_r2 = dfactor_dr / (2.0 * r);
        let da_domega = r * (1.0 + t * t);
        let dnum_domega = da_domega / (1.0 + a * a);
        let factor_derivative_omega = (dnum_domega * den - numerator * r) / (den * den);
        (factor, factor_derivative_r2, factor_derivative_omega)
    };
    let terms = FovDistortionTerms {
        x: u * factor,
        y: v * factor,
        factor,
        factor_derivative_r2,
        factor_derivative_omega,
    };
    if [
        terms.x,
        terms.y,
        terms.factor,
        terms.factor_derivative_r2,
        terms.factor_derivative_omega,
    ]
    .iter()
    .all(|value| value.is_finite())
    {
        Some(terms)
    } else {
        None
    }
}

struct FisheyeDistortionTerms {
    x: f64,
    y: f64,
    dx: f64,
    dy: f64,
    jacobian: Mat2,
}

fn fisheye_distortion_terms(camera: CameraModel, u: f64, v: f64) -> Option<FisheyeDistortionTerms> {
    match camera.model_id {
        COLMAP_THIN_PRISM_FISHEYE => thin_prism_fisheye_distortion_terms(camera, u, v),
        COLMAP_RAD_TAN_THIN_PRISM_FISHEYE => {
            rad_tan_thin_prism_fisheye_distortion_terms(camera, u, v)
        }
        _ => None,
    }
}

fn thin_prism_fisheye_distortion_terms(
    camera: CameraModel,
    u: f64,
    v: f64,
) -> Option<FisheyeDistortionTerms> {
    let k1 = camera.params[4];
    let k2 = camera.params[5];
    let p1 = camera.params[6];
    let p2 = camera.params[7];
    let k3 = camera.params[8];
    let k4 = camera.params[9];
    let sx1 = camera.params[10];
    let sy1 = camera.params[11];
    let r2 = u * u + v * v;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let r8 = r4 * r4;
    let radial_offset = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
    let radial_derivative = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4 + 4.0 * k4 * r6;
    let dx = u * radial_offset + 2.0 * p1 * u * v + p2 * (r2 + 2.0 * u * u) + sx1 * r2;
    let dy = v * radial_offset + 2.0 * p2 * u * v + p1 * (r2 + 2.0 * v * v) + sy1 * r2;
    let mut jacobian =
        radial_tangential_offset_jacobian(u, v, radial_offset, radial_derivative, p1, p2);
    jacobian[(0, 0)] += 2.0 * sx1 * u;
    jacobian[(0, 1)] += 2.0 * sx1 * v;
    jacobian[(1, 0)] += 2.0 * sy1 * u;
    jacobian[(1, 1)] += 2.0 * sy1 * v;
    finite_fisheye_distortion_terms(u, v, dx, dy, jacobian)
}

fn rad_tan_thin_prism_fisheye_distortion_terms(
    camera: CameraModel,
    u: f64,
    v: f64,
) -> Option<FisheyeDistortionTerms> {
    let p0 = camera.params[10];
    let p1 = camera.params[11];
    let s0 = camera.params[12];
    let s1 = camera.params[13];
    let s2 = camera.params[14];
    let s3 = camera.params[15];
    let theta2 = u * u + v * v;
    let mut th_radial = 1.0;
    let mut th_radial_derivative = 0.0;
    let mut theta_power = 1.0;
    for (idx, coeff) in camera.params[4..10].iter().enumerate() {
        th_radial_derivative += (idx as f64 + 1.0) * coeff * theta_power;
        theta_power *= theta2;
        th_radial += coeff * theta_power;
    }

    let x = th_radial * u;
    let y = th_radial * v;
    let dx_du = th_radial + 2.0 * u * u * th_radial_derivative;
    let dx_dv = 2.0 * u * v * th_radial_derivative;
    let dy_du = dx_dv;
    let dy_dv = th_radial + 2.0 * v * v * th_radial_derivative;

    let x2 = x * x;
    let y2 = y * y;
    let xy = x * y;
    let r2 = x2 + y2;
    let r4 = r2 * r2;
    let dx_tang = 2.0 * p1 * xy + p0 * (r2 + 2.0 * x2);
    let dy_tang = 2.0 * p0 * xy + p1 * (r2 + 2.0 * y2);
    let dx_tp = s0 * r2 + s1 * r4;
    let dy_tp = s2 * r2 + s3 * r4;
    let dx = x + dx_tang + dx_tp - u;
    let dy = y + dy_tang + dy_tp - v;

    let dtx_dx = 2.0 * p1 * y + 6.0 * p0 * x + 2.0 * s0 * x + 4.0 * s1 * r2 * x;
    let dtx_dy = 2.0 * p1 * x + 2.0 * p0 * y + 2.0 * s0 * y + 4.0 * s1 * r2 * y;
    let dty_dx = 2.0 * p0 * y + 2.0 * p1 * x + 2.0 * s2 * x + 4.0 * s3 * r2 * x;
    let dty_dy = 2.0 * p0 * x + 6.0 * p1 * y + 2.0 * s2 * y + 4.0 * s3 * r2 * y;
    let jacobian = Mat2::from_row_slice(&[
        (1.0 + dtx_dx) * dx_du + dtx_dy * dy_du - 1.0,
        (1.0 + dtx_dx) * dx_dv + dtx_dy * dy_dv,
        dty_dx * dx_du + (1.0 + dty_dy) * dy_du,
        dty_dx * dx_dv + (1.0 + dty_dy) * dy_dv - 1.0,
    ]);
    finite_fisheye_distortion_terms(u, v, dx, dy, jacobian)
}

struct RadTanThinPrismIntermediateTerms {
    x: f64,
    y: f64,
    r2: f64,
    r4: f64,
    j_xy: Mat2,
}

fn rad_tan_thin_prism_intermediate_terms(
    camera: CameraModel,
    u: f64,
    v: f64,
) -> Option<RadTanThinPrismIntermediateTerms> {
    let p0 = camera.params[10];
    let p1 = camera.params[11];
    let s0 = camera.params[12];
    let s1 = camera.params[13];
    let s2 = camera.params[14];
    let s3 = camera.params[15];
    let theta2 = u * u + v * v;
    let mut th_radial = 1.0;
    let mut theta_power = 1.0;
    for coeff in &camera.params[4..10] {
        theta_power *= theta2;
        th_radial += coeff * theta_power;
    }
    let x = th_radial * u;
    let y = th_radial * v;
    let r2 = x * x + y * y;
    let r4 = r2 * r2;
    let dtx_dx = 2.0 * p1 * y + 6.0 * p0 * x + 2.0 * s0 * x + 4.0 * s1 * r2 * x;
    let dtx_dy = 2.0 * p1 * x + 2.0 * p0 * y + 2.0 * s0 * y + 4.0 * s1 * r2 * y;
    let dty_dx = 2.0 * p0 * y + 2.0 * p1 * x + 2.0 * s2 * x + 4.0 * s3 * r2 * x;
    let dty_dy = 2.0 * p0 * x + 6.0 * p1 * y + 2.0 * s2 * y + 4.0 * s3 * r2 * y;
    let terms = RadTanThinPrismIntermediateTerms {
        x,
        y,
        r2,
        r4,
        j_xy: Mat2::from_row_slice(&[1.0 + dtx_dx, dtx_dy, dty_dx, 1.0 + dty_dy]),
    };
    if [terms.x, terms.y, terms.r2, terms.r4]
        .iter()
        .all(|value| value.is_finite())
        && terms.j_xy.iter().all(|value| value.is_finite())
    {
        Some(terms)
    } else {
        None
    }
}

fn finite_fisheye_distortion_terms(
    u: f64,
    v: f64,
    dx: f64,
    dy: f64,
    jacobian: Mat2,
) -> Option<FisheyeDistortionTerms> {
    let terms = FisheyeDistortionTerms {
        x: u + dx,
        y: v + dy,
        dx,
        dy,
        jacobian,
    };
    if [terms.x, terms.y, terms.dx, terms.dy]
        .iter()
        .all(|value| value.is_finite())
        && terms.jacobian.iter().all(|value| value.is_finite())
    {
        Some(terms)
    } else {
        None
    }
}

fn radial_tangential_offset_jacobian(
    u: f64,
    v: f64,
    radial: f64,
    radial_derivative: f64,
    p1: f64,
    p2: f64,
) -> Mat2 {
    Mat2::from_row_slice(&[
        radial + 2.0 * u * u * radial_derivative + 2.0 * p1 * v + 6.0 * p2 * u,
        2.0 * u * v * radial_derivative + 2.0 * p1 * u + 2.0 * p2 * v,
        2.0 * u * v * radial_derivative + 2.0 * p2 * v + 2.0 * p1 * u,
        radial + 2.0 * v * v * radial_derivative + 6.0 * p1 * v + 2.0 * p2 * u,
    ])
}

struct DivisionProjectionTerms {
    x: f64,
    y: f64,
    scale: f64,
    disc_sqrt: f64,
    j_cam: Mat2x3,
}

fn division_projection_terms(x: f64, y: f64, z: f64, k: f64) -> Option<DivisionProjectionTerms> {
    let q = x * x + y * y;
    let disc_sq = z * z - 4.0 * k * q;
    if disc_sq < 0.0 {
        return None;
    }
    let disc_sqrt = disc_sq.sqrt();
    let den = z + disc_sqrt;
    if den.abs() <= f64::EPSILON {
        return None;
    }
    let scale = 2.0 / den;
    let den_derivative_x = -4.0 * k * x / disc_sqrt;
    let den_derivative_y = -4.0 * k * y / disc_sqrt;
    let den_derivative_z = 1.0 + z / disc_sqrt;
    let scale_derivative_x = -2.0 * den_derivative_x / (den * den);
    let scale_derivative_y = -2.0 * den_derivative_y / (den * den);
    let scale_derivative_z = -2.0 * den_derivative_z / (den * den);
    let j_cam = Mat2x3::from_row_slice(&[
        scale + x * scale_derivative_x,
        x * scale_derivative_y,
        x * scale_derivative_z,
        y * scale_derivative_x,
        scale + y * scale_derivative_y,
        y * scale_derivative_z,
    ]);
    let terms = DivisionProjectionTerms {
        x: scale * x,
        y: scale * y,
        scale,
        disc_sqrt,
        j_cam,
    };
    if [terms.x, terms.y, terms.scale, terms.disc_sqrt]
        .iter()
        .all(|value| value.is_finite())
        && terms.j_cam.iter().all(|value| value.is_finite())
    {
        Some(terms)
    } else {
        None
    }
}

struct EucmProjectionTerms {
    x: f64,
    y: f64,
    rho: f64,
    den: f64,
    j_cam: Mat2x3,
}

fn eucm_projection_terms(
    x: f64,
    y: f64,
    z: f64,
    alpha: f64,
    beta: f64,
) -> Option<EucmProjectionTerms> {
    let q = x * x + y * y;
    let rho2 = beta * q + z * z;
    if rho2 < 0.0 {
        return None;
    }
    let rho = rho2.sqrt();
    let den = alpha * rho + (1.0 - alpha) * z;
    if den < f64::EPSILON {
        return None;
    }
    let inv_den = 1.0 / den;
    let dden_dx = alpha * beta * x / rho;
    let dden_dy = alpha * beta * y / rho;
    let dden_dz = alpha * z / rho + (1.0 - alpha);
    let inv_den2 = inv_den * inv_den;
    let j_cam = Mat2x3::from_row_slice(&[
        inv_den - x * dden_dx * inv_den2,
        -x * dden_dy * inv_den2,
        -x * dden_dz * inv_den2,
        -y * dden_dx * inv_den2,
        inv_den - y * dden_dy * inv_den2,
        -y * dden_dz * inv_den2,
    ]);
    let terms = EucmProjectionTerms {
        x: x * inv_den,
        y: y * inv_den,
        rho,
        den,
        j_cam,
    };
    if [terms.x, terms.y, terms.rho, terms.den]
        .iter()
        .all(|value| value.is_finite())
        && terms.j_cam.iter().all(|value| value.is_finite())
    {
        Some(terms)
    } else {
        None
    }
}

fn finite_difference_camera_param_jacobian(
    camera: CameraModel,
    param: usize,
    pose: SE3,
    point: [f32; 3],
) -> Option<Vec2> {
    if param >= camera.num_params {
        return None;
    }
    let eps = camera.params[param].abs().max(1.0) * 1.0e-6;
    let mut plus = camera;
    let mut minus = camera;
    plus.params[param] += eps;
    minus.params[param] -= eps;
    sync_camera_intrinsics_from_params(&mut plus);
    sync_camera_intrinsics_from_params(&mut minus);
    let p_plus = project_point(plus, pose, point)?;
    let p_minus = project_point(minus, pose, point)?;
    Some(Vec2::new(
        (p_plus[0] - p_minus[0]) / (2.0 * eps),
        (p_plus[1] - p_minus[1]) / (2.0 * eps),
    ))
}

fn mat2x6_to_dmatrix(matrix: Mat2x6) -> DMatrix<f64> {
    DMatrix::from_fn(2, 6, |row, col| matrix[(row, col)])
}

fn pose_block_jacobian(matrix: Mat2x6, block: &PoseBlock) -> Option<DMatrix<f64>> {
    let axes = pose_block_active_axes(block);
    if axes.is_empty() {
        return None;
    }
    Some(DMatrix::from_fn(2, axes.len(), |row, col| {
        matrix[(row, axes[col])]
    }))
}

fn vec2_to_dmatrix(vector: Vec2) -> DMatrix<f64> {
    DMatrix::from_column_slice(2, 1, &[vector[0], vector[1]])
}

fn point_nonpoint_cross(j_point: Mat2x3, j_nonpoint: &DMatrix<f64>) -> DMatrix<f64> {
    DMatrix::from_fn(3, j_nonpoint.ncols(), |row, col| {
        j_point[(0, row)] * j_nonpoint[(0, col)] + j_point[(1, row)] * j_nonpoint[(1, col)]
    })
}

pub(crate) fn apply_camera_param_delta(
    reconstruction: &mut Reconstruction,
    spec: CameraParamSpec,
    delta: f64,
) {
    if spec.camera < reconstruction.cameras.len() {
        let camera = &mut reconstruction.cameras[spec.camera];
        if spec.param < camera.num_params {
            camera.params[spec.param] += delta;
            sync_camera_intrinsics_from_params(camera);
        }
        if spec.camera == 0 {
            reconstruction.camera = *camera;
        }
    } else if spec.camera == 0 && spec.param < reconstruction.camera.num_params {
        reconstruction.camera.params[spec.param] += delta;
        sync_camera_intrinsics_from_params(&mut reconstruction.camera);
    }
}

pub(crate) fn sync_camera_intrinsics_from_params(camera: &mut CameraModel) {
    if let Some(focal_idxs) = colmap_camera_model_focal_idxs(camera.model_id) {
        match focal_idxs {
            [idx] if *idx < camera.num_params => {
                camera.fx = camera.params[*idx] as f32;
                camera.fy = camera.params[*idx] as f32;
            }
            [idx_x, idx_y] if *idx_x < camera.num_params && *idx_y < camera.num_params => {
                camera.fx = camera.params[*idx_x] as f32;
                camera.fy = camera.params[*idx_y] as f32;
            }
            _ => {}
        }
    }
    if let Some([idx_x, idx_y]) = colmap_camera_model_principal_point_idxs(camera.model_id) {
        if idx_x < camera.num_params && idx_y < camera.num_params {
            camera.cx = camera.params[idx_x] as f32;
            camera.cy = camera.params[idx_y] as f32;
        }
    }
}

pub(crate) fn apply_pose_delta_f64(pose: SE3, delta: Vec6) -> SE3 {
    let q = pose.quaternion();
    let base_rotation = Quat::from_xyzw(q[0], q[1], q[2], q[3]).normalize();
    let omega = Vec3::new(delta[0] as f32, delta[1] as f32, delta[2] as f32);
    let angle = omega.length();
    let delta_rotation = if angle > 1.0e-12 {
        Quat::from_axis_angle(omega / angle, angle)
    } else {
        Quat::IDENTITY
    };
    let t = pose.translation();
    let translation = Vec3::new(
        t[0] + delta[3] as f32,
        t[1] + delta[4] as f32,
        t[2] + delta[5] as f32,
    );
    SE3::from_quat_translation((delta_rotation * base_rotation).normalize(), translation)
}

fn total_cost(
    reconstruction: &Reconstruction,
    observations: &[BaObservation],
    loss_function: BundleAdjustmentLoss,
    pose_priors: &[super::BundleAdjustmentPosePrior],
    prior_position_fallback_stddev: f64,
) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for obs in observations {
        let Some(pose) = reconstruction.poses.get(obs.image).copied().flatten() else {
            continue;
        };
        let Some(point) = reconstruction.points.get(obs.point) else {
            continue;
        };
        let Some(predicted) =
            project_point(reconstruction.camera_for_image(obs.image), pose, point.xyz)
        else {
            continue;
        };
        let s = (predicted[0] - obs.xy[0]).powi(2) + (predicted[1] - obs.xy[1]).powi(2);
        if s.is_finite() {
            total += loss_function.cost(s);
            count += 1;
        }
    }
    if count == 0 {
        f64::INFINITY
    } else {
        total / count as f64
            + pose_prior_total_cost(reconstruction, pose_priors, prior_position_fallback_stddev)
    }
}

fn pose_prior_total_cost(
    reconstruction: &Reconstruction,
    pose_priors: &[super::BundleAdjustmentPosePrior],
    fallback_stddev: f64,
) -> f64 {
    let mut total = 0.0;
    for prior in pose_priors {
        let Some(pose) = reconstruction.poses.get(prior.image).copied().flatten() else {
            continue;
        };
        let center = camera_center_world(pose);
        let prior_position = Vec3d::from_column_slice(&prior.position);
        let residual = center - prior_position;
        let information =
            position_prior_information_matrix(&prior.position_covariance, fallback_stddev);
        let cost = 0.5 * residual.dot(&(information * residual));
        if cost.is_finite() {
            total += cost;
        }
    }
    total
}

fn relative_cost_change(previous: f64, current: f64) -> f64 {
    if !previous.is_finite() || !current.is_finite() {
        return f64::INFINITY;
    }
    ((previous - current) / previous.abs().max(1.0)).abs()
}

fn restore_state(
    reconstruction: &mut Reconstruction,
    poses: &[Option<SE3>],
    frames: &[Frame],
    rigs: &[Rig],
    points: &[[f32; 3]],
    camera: CameraModel,
    cameras: &[CameraModel],
) {
    reconstruction.poses.clone_from_slice(poses);
    reconstruction.frames.clone_from_slice(frames);
    reconstruction.rigs.clone_from_slice(rigs);
    for (point, xyz) in reconstruction.points.iter_mut().zip(points.iter()) {
        point.xyz = *xyz;
    }
    reconstruction.camera = camera;
    reconstruction.cameras.clone_from_slice(cameras);
}

#[allow(dead_code)]
fn pose_from_parts(rotation: Quat, translation: Vec3) -> SE3 {
    SE3::from_quat_translation(rotation.normalize(), translation)
}

#[cfg(test)]
mod tests {
    use super::super::{
        refine_bundle_adjustment, BundleAdjustmentLinearSolver, BundleAdjustmentLoss,
        BundleAdjustmentOptions, BundleAdjustmentPreconditioner, BundleAdjustmentTerminationReason,
        BundleAdjustmentTerminationType,
    };
    use super::*;
    use crate::ba::shared::project_point;
    use crate::types::{
        CameraModel, DataId, ImageFrame, Point3D, Reconstruction, Rig, RigSensor, SensorType,
        TrackObservation,
    };
    use rustslam::Descriptors;
    use std::path::PathBuf;

    #[test]
    fn bundle_adjustment_defaults_match_colmap_ceres_convergence_options() {
        let options = BundleAdjustmentOptions::default();

        assert_eq!(options.iterations, 100);
        assert_eq!(options.function_tolerance, 0.0);
        assert_eq!(options.gradient_tolerance, 1.0e-4);
        assert_eq!(options.parameter_tolerance, 0.0);
        assert_eq!(options.max_linear_solver_iterations, 200);
        assert_eq!(options.num_threads, -1);
        assert_eq!(options.min_num_residuals_for_multi_threading, 50_000);
        assert_eq!(options.max_num_consecutive_invalid_steps, 10);
        assert_eq!(options.max_consecutive_nonmonotonic_steps, 10);
    }

    #[test]
    fn loss_functions_match_ceres_rho_and_weight_formulas() {
        let scale = 2.0;
        let a2 = scale * scale;

        // Inlier region: s <= scale^2 → all robust losses behave near-quadratic
        // and Huber matches trivial exactly.
        let s_in = 1.0;
        assert!((BundleAdjustmentLoss::Trivial.weight(s_in) - 1.0).abs() < 1e-12);
        assert!((BundleAdjustmentLoss::Huber { scale }.weight(s_in) - 1.0).abs() < 1e-12);
        assert!((BundleAdjustmentLoss::Trivial.cost(s_in) - 0.5 * s_in).abs() < 1e-12);
        assert!((BundleAdjustmentLoss::Huber { scale }.cost(s_in) - 0.5 * s_in).abs() < 1e-12);

        // Outlier region: s > scale^2.
        let s_out = 16.0; // err = 4, scale = 2
        let huber_w = BundleAdjustmentLoss::Huber { scale }.weight(s_out);
        assert!((huber_w - scale / s_out.sqrt()).abs() < 1e-12);
        let huber_c = BundleAdjustmentLoss::Huber { scale }.cost(s_out);
        assert!((huber_c - (scale * s_out.sqrt() - 0.5 * a2)).abs() < 1e-12);

        let cauchy_w = BundleAdjustmentLoss::Cauchy { scale }.weight(s_out);
        assert!((cauchy_w - 1.0 / (1.0 + s_out / a2)).abs() < 1e-12);
        let cauchy_c = BundleAdjustmentLoss::Cauchy { scale }.cost(s_out);
        assert!((cauchy_c - 0.5 * a2 * (1.0 + s_out / a2).ln()).abs() < 1e-12);

        let soft_w = BundleAdjustmentLoss::SoftL1 { scale }.weight(s_out);
        assert!((soft_w - 1.0 / (1.0 + s_out / a2).sqrt()).abs() < 1e-12);
        let soft_c = BundleAdjustmentLoss::SoftL1 { scale }.cost(s_out);
        assert!((soft_c - a2 * ((1.0 + s_out / a2).sqrt() - 1.0)).abs() < 1e-12);

        // Robust losses must down-weight large residuals more aggressively than
        // Huber, and all weights are in (0, 1].
        assert!(cauchy_w < soft_w);
        assert!(soft_w < huber_w);
        assert!(cauchy_w > 0.0 && huber_w <= 1.0);
    }

    #[test]
    fn trust_region_radius_updates_match_ceres_levenberg_marquardt() {
        assert!(
            (ceres_step_accepted_radius(1.0e4, 0.9) - 1.0e4 / (1.0 - 0.8_f64.powi(3))).abs()
                < 1.0e-6
        );
        assert_eq!(ceres_step_accepted_radius(1.0e4, 0.5), 1.0e4);
        assert!(
            (ceres_step_accepted_radius(1.0e4, 0.1) - 1.0e4 / (1.0 - (-0.8_f64).powi(3))).abs()
                < 1.0e-6
        );
        assert_eq!(ceres_step_rejected_radius(1.0e4, 2.0), 5.0e3);
        assert_eq!(
            ceres_step_rejected_radius(1.0e-40, 2.0),
            MIN_TRUST_REGION_RADIUS
        );
    }

    #[test]
    fn lm_diagonal_clamp_matches_ceres_bounds() {
        assert_eq!(clamp_lm_diagonal(0.0), LM_MIN_DIAGONAL);
        assert_eq!(clamp_lm_diagonal(1.0e40), LM_MAX_DIAGONAL);
        assert_eq!(clamp_lm_diagonal(1.0), 1.0);
    }

    #[test]
    fn schur_hessian_uses_sparse_storage_above_ceres_dense_threshold() {
        let dense = SchurHessian::new_for_pose_entities(120, DENSE_SCHUR_MAX_POSE_ENTITIES);
        assert!(matches!(dense, SchurHessian::Dense(_)));
        let sparse = SchurHessian::new_for_pose_entities(120, DENSE_SCHUR_MAX_POSE_ENTITIES + 1);
        assert!(matches!(sparse, SchurHessian::Sparse(_)));
    }

    #[test]
    fn native_linear_solver_policy_matches_ceres_thresholds() {
        assert_eq!(
            native_linear_solver_policy(DENSE_SCHUR_MAX_POSE_ENTITIES).0,
            BundleAdjustmentLinearSolver::DenseSchur
        );
        assert_eq!(
            native_linear_solver_policy(DENSE_SCHUR_MAX_POSE_ENTITIES + 1).0,
            BundleAdjustmentLinearSolver::SparseSchur
        );
        assert_eq!(
            native_linear_solver_policy(SPARSE_SCHUR_MAX_POSE_ENTITIES).0,
            BundleAdjustmentLinearSolver::SparseSchur
        );
        let (solver, preconditioner) =
            native_linear_solver_policy(SPARSE_SCHUR_MAX_POSE_ENTITIES + 1);
        assert_eq!(solver, BundleAdjustmentLinearSolver::IterativeSchur);
        assert_eq!(
            preconditioner,
            Some(BundleAdjustmentPreconditioner::SchurJacobi)
        );
    }

    #[test]
    fn native_linear_solver_policy_honors_explicit_preference() {
        use crate::ba::BundleAdjustmentLinearSolverPreference as Preference;

        assert_eq!(
            native_linear_solver_policy_for_preference(Preference::DenseSchur, 1001).0,
            BundleAdjustmentLinearSolver::DenseSchur
        );
        assert_eq!(
            native_linear_solver_policy_for_preference(Preference::SparseSchur, 10).0,
            BundleAdjustmentLinearSolver::SparseSchur
        );
        let (solver, preconditioner) =
            native_linear_solver_policy_for_preference(Preference::IterativeSchur, 10);
        assert_eq!(solver, BundleAdjustmentLinearSolver::IterativeSchur);
        assert_eq!(
            preconditioner,
            Some(BundleAdjustmentPreconditioner::SchurJacobi)
        );
    }

    #[test]
    fn iterative_schur_linear_solver_solves_spd_reduced_system() {
        let hessian = SchurHessian::Dense(DMatrix::from_row_slice(
            3,
            3,
            &[4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0],
        ));
        let expected = DVector::from_column_slice(&[1.0, -2.0, 0.5]);
        let gradient = -(&hessian.to_dense() * &expected);
        let blocks = vec![
            SchurParameterBlock { offset: 0, dim: 2 },
            SchurParameterBlock { offset: 2, dim: 1 },
        ];
        let result = solve_linear_system(
            &hessian,
            &gradient,
            BundleAdjustmentLinearSolver::IterativeSchur,
            Some(BundleAdjustmentPreconditioner::SchurJacobi),
            &blocks,
            50,
        );
        let delta = result.delta.expect("iterative schur solve");
        for i in 0..3 {
            assert!((delta[i] - expected[i]).abs() < 1.0e-4, "i={i}");
        }
    }

    #[test]
    fn bundle_adjustment_pose_prior_pulls_camera_center_toward_prior() {
        use super::super::BundleAdjustmentPosePrior;

        let true_camera = CameraModel::new_pinhole(100, 100, 60.0, 60.0, 50.0, 50.0);
        let end_pose = SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(0.45, 0.02, 0.0));
        let prior_center = {
            let center = camera_center_world(end_pose);
            [center[0] as f64, center[1] as f64, center[2] as f64]
        };
        let start_pose = SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(2.45, 0.02, 0.0));
        let scene_points = [
            [-0.6, -0.4, 3.2],
            [-0.2, -0.4, 3.0],
            [0.3, -0.35, 3.4],
            [0.7, -0.2, 3.6],
        ];
        let mut frames = vec![frame(0), frame(1)];
        for point in scene_points {
            let projected0 = project_point(true_camera, SE3::identity(), point).unwrap();
            let projected1 = project_point(true_camera, end_pose, point).unwrap();
            frames[0].keypoints.push(rustslam::KeyPoint::new(
                projected0[0] as f32,
                projected0[1] as f32,
            ));
            frames[1].keypoints.push(rustslam::KeyPoint::new(
                projected1[0] as f32,
                projected1[1] as f32,
            ));
        }
        let mut reconstruction = Reconstruction {
            camera: true_camera,
            cameras: vec![true_camera],
            camera_ids: vec![1],
            rigs: Vec::new(),
            frames: Vec::new(),
            image_names: vec!["0.jpg".into(), "1.jpg".into()],
            image_paths: vec![PathBuf::from("0.jpg"), PathBuf::from("1.jpg")],
            image_ids: vec![1, 2],
            image_camera_indices: vec![0, 0],
            image_frame_indices: vec![None, None],
            poses: vec![Some(SE3::identity()), Some(start_pose)],
            observations: vec![
                (0..scene_points.len()).map(Some).collect(),
                (0..scene_points.len()).map(Some).collect(),
            ],
            keypoints: frames.iter().map(|f| f.keypoints.clone()).collect(),
            point_ids: (1..=scene_points.len() as u64).collect(),
            points: scene_points
                .iter()
                .enumerate()
                .map(|(idx, xyz)| Point3D {
                    xyz: *xyz,
                    color: [0, 0, 0],
                    error: 0.0,
                    track: vec![
                        TrackObservation {
                            image: 0,
                            feature: idx,
                        },
                        TrackObservation {
                            image: 1,
                            feature: idx,
                        },
                    ],
                })
                .collect(),
        };
        let start_center = camera_center_world(start_pose);
        let report = refine_bundle_adjustment_native(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 20,
                loss_function: BundleAdjustmentLoss::Trivial,
                max_observation_error_px: 50.0,
                constant_images: vec![0],
                pose_priors: vec![BundleAdjustmentPosePrior::new(1, prior_center)],
                prior_position_fallback_stddev: 0.05,
                allow_single_observation_points: true,
                ..BundleAdjustmentOptions::default()
            },
        )
        .expect("ba with pose prior");
        assert!(report.is_solution_usable());
        let optimized = reconstruction.poses[1].expect("optimized pose");
        let end_center = camera_center_world(optimized);
        let start_error =
            (start_center - nalgebra::DVector::from_column_slice(&prior_center)).norm();
        let end_error = (end_center - nalgebra::DVector::from_column_slice(&prior_center)).norm();
        assert!(
            end_error < start_error,
            "start_error={start_error} end_error={end_error}"
        );
    }

    #[test]
    fn bundle_adjustment_compute_covariance_returns_pose_blocks() {
        let true_camera = CameraModel::new_pinhole(100, 100, 60.0, 60.0, 50.0, 50.0);
        let poses = [
            SE3::identity(),
            SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(0.45, 0.02, 0.0)),
        ];
        let scene_points = [[0.0, 0.0, 3.0], [0.4, 0.1, 3.2]];
        let mut frames = vec![frame(0), frame(1)];
        for point in scene_points {
            let projected0 = project_point(true_camera, poses[0], point).unwrap();
            let projected1 = project_point(true_camera, poses[1], point).unwrap();
            frames[0].keypoints.push(rustslam::KeyPoint::new(
                projected0[0] as f32,
                projected0[1] as f32,
            ));
            frames[1].keypoints.push(rustslam::KeyPoint::new(
                projected1[0] as f32,
                projected1[1] as f32,
            ));
        }
        let mut reconstruction = Reconstruction {
            camera: true_camera,
            cameras: vec![true_camera],
            camera_ids: vec![1],
            rigs: Vec::new(),
            frames: Vec::new(),
            image_names: vec!["0.jpg".into(), "1.jpg".into()],
            image_paths: vec![PathBuf::from("0.jpg"), PathBuf::from("1.jpg")],
            image_ids: vec![1, 2],
            image_camera_indices: vec![0, 0],
            image_frame_indices: vec![None, None],
            poses: vec![Some(poses[0]), Some(poses[1])],
            observations: vec![vec![Some(0), Some(1)], vec![Some(0), Some(1)]],
            keypoints: frames.iter().map(|f| f.keypoints.clone()).collect(),
            point_ids: vec![1, 2],
            points: scene_points
                .iter()
                .enumerate()
                .map(|(idx, xyz)| Point3D {
                    xyz: *xyz,
                    color: [0, 0, 0],
                    error: 0.0,
                    track: vec![
                        TrackObservation {
                            image: 0,
                            feature: idx,
                        },
                        TrackObservation {
                            image: 1,
                            feature: idx,
                        },
                    ],
                })
                .collect(),
        };
        let report = refine_bundle_adjustment_native(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 8,
                loss_function: BundleAdjustmentLoss::Trivial,
                max_observation_error_px: 50.0,
                constant_images: vec![0],
                allow_single_observation_points: true,
                compute_covariance: true,
                ..BundleAdjustmentOptions::default()
            },
        )
        .expect("ba with covariance");
        let covariance = report.covariance.expect("covariance report");
        let block1 = covariance
            .pose_blocks
            .get(&1)
            .expect("image 1 covariance block");
        assert!(block1[(0, 0)].is_finite());
    }

    #[test]
    fn solve_linear_system_uses_cholesky_for_spd_reduced_camera_matrix() {
        // A symmetric positive definite reduced camera matrix solves the same
        // via Ceres-style Cholesky as via the LU fallback.
        let hessian = DMatrix::from_row_slice(3, 3, &[4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0]);
        let expected = DVector::from_column_slice(&[1.0, -2.0, 0.5]);
        let gradient = -(&hessian * &expected);

        let result = solve_dense_linear_system(&hessian, &gradient, 1);
        let delta = result.delta.expect("SPD system must solve");
        assert_eq!(result.iterations, 1);
        for i in 0..3 {
            assert!((delta[i] - expected[i]).abs() < 1e-9);
        }

        // A zero iteration budget yields no step, matching Ceres' zero
        // linear-solver iteration guard.
        let none = solve_dense_linear_system(&hessian, &gradient, 0);
        assert!(none.delta.is_none());
        assert_eq!(none.iterations, 0);
    }

    #[test]
    fn analytic_projection_jacobians_match_numerical_differences() {
        let cameras = [
            CameraModel::from_colmap(COLMAP_SIMPLE_PINHOLE, 200, 160, &[95.0, 100.0, 80.0])
                .unwrap(),
            CameraModel::from_colmap(COLMAP_PINHOLE, 200, 160, &[90.0, 96.0, 100.0, 80.0]).unwrap(),
            CameraModel::from_colmap(COLMAP_SIMPLE_RADIAL, 200, 160, &[95.0, 100.0, 80.0, 0.02])
                .unwrap(),
            CameraModel::from_colmap(COLMAP_RADIAL, 200, 160, &[95.0, 100.0, 80.0, 0.02, -0.001])
                .unwrap(),
            CameraModel::from_colmap(
                COLMAP_OPENCV,
                200,
                160,
                &[90.0, 96.0, 100.0, 80.0, 0.02, -0.001, 0.0005, -0.0003],
            )
            .unwrap(),
            CameraModel::from_colmap(
                COLMAP_FULL_OPENCV,
                200,
                160,
                &[
                    90.0, 96.0, 100.0, 80.0, 0.02, -0.001, 0.0005, -0.0003, 0.00001, 0.00002,
                    -0.00001, 0.000005,
                ],
            )
            .unwrap(),
            CameraModel::from_colmap(COLMAP_FOV, 200, 160, &[90.0, 96.0, 100.0, 80.0, 0.25])
                .unwrap(),
            CameraModel::from_colmap(COLMAP_SIMPLE_FISHEYE, 200, 160, &[95.0, 100.0, 80.0])
                .unwrap(),
            CameraModel::from_colmap(COLMAP_FISHEYE, 200, 160, &[90.0, 96.0, 100.0, 80.0]).unwrap(),
            CameraModel::from_colmap(
                COLMAP_SIMPLE_RADIAL_FISHEYE,
                200,
                160,
                &[95.0, 100.0, 80.0, 0.02],
            )
            .unwrap(),
            CameraModel::from_colmap(
                COLMAP_RADIAL_FISHEYE,
                200,
                160,
                &[95.0, 100.0, 80.0, 0.02, -0.001],
            )
            .unwrap(),
            CameraModel::from_colmap(
                COLMAP_OPENCV_FISHEYE,
                200,
                160,
                &[90.0, 96.0, 100.0, 80.0, 0.02, -0.001, 0.0001, -0.00001],
            )
            .unwrap(),
            CameraModel::from_colmap(
                COLMAP_THIN_PRISM_FISHEYE,
                200,
                160,
                &[
                    90.0, 96.0, 100.0, 80.0, 0.01, -0.0005, 0.0001, -0.0001, 0.00001, -0.00001,
                    0.00002, -0.00002,
                ],
            )
            .unwrap(),
            CameraModel::from_colmap(
                COLMAP_RAD_TAN_THIN_PRISM_FISHEYE,
                200,
                160,
                &[
                    90.0, 96.0, 100.0, 80.0, 0.01, -0.0005, 0.00001, -0.000001, 0.0, 0.0, 0.0001,
                    -0.0001, 0.00002, -0.00002, 0.00001, -0.00001,
                ],
            )
            .unwrap(),
            CameraModel::from_colmap(COLMAP_SIMPLE_DIVISION, 200, 160, &[95.0, 100.0, 80.0, 0.02])
                .unwrap(),
            CameraModel::from_colmap(COLMAP_DIVISION, 200, 160, &[90.0, 96.0, 100.0, 80.0, 0.02])
                .unwrap(),
            CameraModel::from_colmap(COLMAP_EUCM, 200, 160, &[90.0, 96.0, 100.0, 80.0, 0.4, 1.2])
                .unwrap(),
        ];
        let pose = SE3::from_quat_translation(
            Quat::from_rotation_y(0.12) * Quat::from_rotation_x(-0.04),
            Vec3::new(0.2, -0.1, 0.05),
        );
        let point = [0.3, -0.2, 3.2];

        for camera in cameras {
            let (analytic_pose, analytic_point) =
                analytic_projection_jacobians(camera, pose, point).unwrap();
            let numerical_pose = numerical_pose_jacobian(camera, pose, point).unwrap();
            let numerical_point = numerical_point_jacobian(camera, pose, point).unwrap();

            for row in 0..2 {
                for col in 0..6 {
                    assert!(
                        (analytic_pose[(row, col)] - numerical_pose[(row, col)]).abs() < 2.0e-2,
                        "pose row={row} col={col} analytic={} numerical={}",
                        analytic_pose[(row, col)],
                        numerical_pose[(row, col)]
                    );
                }
                for col in 0..3 {
                    assert!(
                        (analytic_point[(row, col)] - numerical_point[(row, col)]).abs() < 2.0e-2,
                        "point row={row} col={col} analytic={} numerical={}",
                        analytic_point[(row, col)],
                        numerical_point[(row, col)]
                    );
                }
            }
        }
    }

    #[test]
    fn analytic_frame_and_sensor_pose_jacobians_match_numerical_differences() {
        let camera = CameraModel::from_colmap(
            COLMAP_OPENCV,
            220,
            180,
            &[96.0, 101.0, 110.0, 90.0, 0.01, -0.0008, 0.0004, -0.0002],
        )
        .unwrap();
        let sensor_from_rig = SE3::from_quat_translation(
            Quat::from_rotation_z(0.08) * Quat::from_rotation_y(-0.05),
            Vec3::new(0.3, -0.04, 0.02),
        );
        let rig_from_world = SE3::from_quat_translation(
            Quat::from_rotation_y(0.14) * Quat::from_rotation_x(-0.06),
            Vec3::new(0.2, 0.1, -0.03),
        );
        let point = [0.4, -0.25, 3.4];

        let analytic_frame =
            analytic_frame_pose_jacobian(camera, sensor_from_rig, rig_from_world, point).unwrap();
        let numerical_frame =
            numerical_frame_pose_jacobian(camera, sensor_from_rig, rig_from_world, point).unwrap();
        assert_jacobian_close(analytic_frame, numerical_frame, 5.0e-2);

        let analytic_sensor =
            analytic_sensor_pose_jacobian(camera, sensor_from_rig, rig_from_world, point).unwrap();
        let numerical_sensor =
            numerical_sensor_pose_jacobian(camera, sensor_from_rig, rig_from_world, point).unwrap();
        assert_jacobian_close(analytic_sensor, numerical_sensor, 5.0e-2);
    }

    #[test]
    fn ba_pose_delta_updates_rotation_and_translation_as_separate_blocks() {
        let pose =
            SE3::from_quat_translation(Quat::from_rotation_y(0.2), Vec3::new(1.0, -2.0, 3.0));
        let mut rotation_only = Vec6::zeros();
        rotation_only[2] = 0.1;

        let updated = apply_pose_delta_f64(pose, rotation_only);

        assert_eq!(updated.translation(), pose.translation());

        let mut translation_only = Vec6::zeros();
        translation_only[3] = 0.4;
        translation_only[4] = -0.2;
        translation_only[5] = 0.1;
        let translated = apply_pose_delta_f64(pose, translation_only);

        assert_eq!(translated.quaternion(), pose.quaternion());
        assert_eq!(translated.translation(), [1.4_f32, -2.2_f32, 3.1_f32]);
    }

    #[test]
    fn analytic_camera_param_jacobians_match_finite_differences() {
        let cameras = [
            CameraModel::from_colmap(COLMAP_SIMPLE_PINHOLE, 200, 160, &[95.0, 100.0, 80.0])
                .unwrap(),
            CameraModel::from_colmap(COLMAP_PINHOLE, 200, 160, &[90.0, 96.0, 100.0, 80.0]).unwrap(),
            CameraModel::from_colmap(COLMAP_SIMPLE_RADIAL, 200, 160, &[95.0, 100.0, 80.0, 0.02])
                .unwrap(),
            CameraModel::from_colmap(COLMAP_RADIAL, 200, 160, &[95.0, 100.0, 80.0, 0.02, -0.001])
                .unwrap(),
            CameraModel::from_colmap(
                COLMAP_OPENCV,
                200,
                160,
                &[90.0, 96.0, 100.0, 80.0, 0.02, -0.001, 0.0005, -0.0003],
            )
            .unwrap(),
            CameraModel::from_colmap(
                COLMAP_FULL_OPENCV,
                200,
                160,
                &[
                    90.0, 96.0, 100.0, 80.0, 0.02, -0.001, 0.0005, -0.0003, 0.00001, 0.00002,
                    -0.00001, 0.000005,
                ],
            )
            .unwrap(),
            CameraModel::from_colmap(COLMAP_FOV, 200, 160, &[90.0, 96.0, 100.0, 80.0, 0.25])
                .unwrap(),
            CameraModel::from_colmap(COLMAP_SIMPLE_FISHEYE, 200, 160, &[95.0, 100.0, 80.0])
                .unwrap(),
            CameraModel::from_colmap(COLMAP_FISHEYE, 200, 160, &[90.0, 96.0, 100.0, 80.0]).unwrap(),
            CameraModel::from_colmap(
                COLMAP_SIMPLE_RADIAL_FISHEYE,
                200,
                160,
                &[95.0, 100.0, 80.0, 0.02],
            )
            .unwrap(),
            CameraModel::from_colmap(
                COLMAP_RADIAL_FISHEYE,
                200,
                160,
                &[95.0, 100.0, 80.0, 0.02, -0.001],
            )
            .unwrap(),
            CameraModel::from_colmap(
                COLMAP_OPENCV_FISHEYE,
                200,
                160,
                &[90.0, 96.0, 100.0, 80.0, 0.02, -0.001, 0.0001, -0.00001],
            )
            .unwrap(),
            CameraModel::from_colmap(
                COLMAP_THIN_PRISM_FISHEYE,
                200,
                160,
                &[
                    90.0, 96.0, 100.0, 80.0, 0.01, -0.0005, 0.0001, -0.0001, 0.00001, -0.00001,
                    0.00002, -0.00002,
                ],
            )
            .unwrap(),
            CameraModel::from_colmap(
                COLMAP_RAD_TAN_THIN_PRISM_FISHEYE,
                200,
                160,
                &[
                    90.0, 96.0, 100.0, 80.0, 0.01, -0.0005, 0.00001, -0.000001, 0.0, 0.0, 0.0001,
                    -0.0001, 0.00002, -0.00002, 0.00001, -0.00001,
                ],
            )
            .unwrap(),
            CameraModel::from_colmap(COLMAP_SIMPLE_DIVISION, 200, 160, &[95.0, 100.0, 80.0, 0.02])
                .unwrap(),
            CameraModel::from_colmap(COLMAP_DIVISION, 200, 160, &[90.0, 96.0, 100.0, 80.0, 0.02])
                .unwrap(),
            CameraModel::from_colmap(COLMAP_EUCM, 200, 160, &[90.0, 96.0, 100.0, 80.0, 0.4, 1.2])
                .unwrap(),
        ];
        let pose = SE3::from_quat_translation(
            Quat::from_rotation_y(0.12) * Quat::from_rotation_x(-0.04),
            Vec3::new(0.2, -0.1, 0.05),
        );
        let point = [0.3, -0.2, 3.2];

        for camera in cameras {
            for param in 0..camera.num_params {
                let analytic = analytic_camera_param_jacobian(camera, param, pose, point).unwrap();
                let numerical =
                    finite_difference_camera_param_jacobian(camera, param, pose, point).unwrap();

                assert!(
                    (analytic[0] - numerical[0]).abs() < 1.0e-6,
                    "param={param} du analytic={} numerical={}",
                    analytic[0],
                    numerical[0]
                );
                assert!(
                    (analytic[1] - numerical[1]).abs() < 1.0e-6,
                    "param={param} dv analytic={} numerical={}",
                    analytic[1],
                    numerical[1]
                );
            }
        }
    }

    #[test]
    fn local_bundle_adjustment_keeps_non_variable_images_fixed() {
        let mut frames = vec![frame(0), frame(1)];
        frames[0].keypoints = vec![
            rustslam::KeyPoint::new(45.0, 45.0),
            rustslam::KeyPoint::new(55.0, 45.0),
            rustslam::KeyPoint::new(45.0, 55.0),
            rustslam::KeyPoint::new(55.0, 55.0),
            rustslam::KeyPoint::new(50.0, 50.0),
            rustslam::KeyPoint::new(60.0, 50.0),
        ];
        frames[1].keypoints = vec![
            rustslam::KeyPoint::new(70.0, 45.0),
            rustslam::KeyPoint::new(80.0, 45.0),
            rustslam::KeyPoint::new(70.0, 55.0),
            rustslam::KeyPoint::new(80.0, 55.0),
            rustslam::KeyPoint::new(75.0, 50.0),
            rustslam::KeyPoint::new(85.0, 50.0),
        ];
        let mut reconstruction = reconstruction(&frames);
        let fixed_pose = SE3::identity();
        reconstruction.poses[0] = Some(fixed_pose);
        reconstruction.poses[1] = Some(SE3::from_quat_translation(
            Quat::IDENTITY,
            Vec3::new(0.8, 0.05, 0.0),
        ));
        for (idx, xyz) in [
            [-0.2, -0.2, 2.0],
            [0.2, -0.2, 2.0],
            [-0.2, 0.2, 2.0],
            [0.2, 0.2, 2.0],
            [0.0, 0.0, 2.0],
            [0.4, 0.0, 2.0],
        ]
        .into_iter()
        .enumerate()
        {
            reconstruction.observations[0][idx] = Some(idx);
            reconstruction.observations[1][idx] = Some(idx);
            reconstruction.points.push(Point3D {
                xyz,
                color: [0, 0, 0],
                error: 0.0,
                track: vec![
                    TrackObservation {
                        image: 0,
                        feature: idx,
                    },
                    TrackObservation {
                        image: 1,
                        feature: idx,
                    },
                ],
            });
            reconstruction.point_ids.push(idx as u64 + 1);
        }

        let report = refine_bundle_adjustment(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 2,
                loss_function: BundleAdjustmentLoss::Huber { scale: 4.0 },
                max_observation_error_px: 50.0,
                variable_images: Some(vec![1]),
                constant_images: vec![0],
                point_ids: Some((0..6).collect()),
                ..BundleAdjustmentOptions::default()
            },
        );

        assert!(report.is_some());
        assert_eq!(
            reconstruction.poses[0].unwrap().translation(),
            fixed_pose.translation()
        );
    }

    #[test]
    fn explicit_constant_images_override_the_default_global_gauge() {
        let frames = vec![frame(0), frame(1), frame(2)];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::from_quat_translation(
            Quat::IDENTITY,
            Vec3::new(0.8, 0.0, 0.0),
        ));
        reconstruction.poses[2] = Some(SE3::from_quat_translation(
            Quat::IDENTITY,
            Vec3::new(1.6, 0.0, 0.0),
        ));

        let variable = variable_pose_blocks(&reconstruction, None, &[2], &[], true);
        let variable_images = variable
            .blocks
            .iter()
            .map(|block| block.images.clone())
            .collect::<Vec<_>>();

        assert_eq!(variable_images, vec![vec![0], vec![1]]);
    }

    #[test]
    fn three_point_gauge_promotes_independent_points_to_constant() {
        let frames = vec![frame(0), frame(1)];
        let mut reconstruction = reconstruction(&frames);
        for xyz in [
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [2.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
        ] {
            reconstruction.points.push(Point3D {
                xyz,
                color: [0, 0, 0],
                error: 0.0,
                track: Vec::new(),
            });
        }
        let observations = (0..4)
            .map(|point| BaObservation {
                image: 0,
                point,
                xy: [0.0, 0.0],
            })
            .collect::<Vec<_>>();
        let mut constant_points = HashSet::new();

        add_three_point_gauge(&mut constant_points, &reconstruction, &observations);

        assert_eq!(constant_points.len(), 3);
        assert!(constant_points.contains(&0));
        assert!(constant_points.contains(&1));
        assert!(!constant_points.contains(&2));
        assert!(constant_points.contains(&3));
    }

    #[test]
    fn two_cams_from_world_gauge_fixes_first_pose_and_one_translation_axis() {
        let frames = vec![frame(0), frame(1), frame(2)];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::from_quat_translation(
            Quat::IDENTITY,
            Vec3::new(0.1, 0.8, 0.2),
        ));
        reconstruction.poses[2] = Some(SE3::from_quat_translation(
            Quat::IDENTITY,
            Vec3::new(1.4, 0.2, 0.1),
        ));
        let mut pose_blocks = variable_pose_blocks(&reconstruction, None, &[], &[], false);

        assert!(apply_two_cams_from_world_gauge(
            &mut pose_blocks,
            &reconstruction,
            &BundleAdjustmentOptions::default(),
            &[
                BaObservation {
                    image: 0,
                    point: 0,
                    xy: [0.0, 0.0],
                },
                BaObservation {
                    image: 1,
                    point: 0,
                    xy: [0.0, 0.0],
                },
                BaObservation {
                    image: 2,
                    point: 0,
                    xy: [0.0, 0.0],
                },
            ],
        ));
        reindex_pose_blocks(&mut pose_blocks);

        assert_eq!(pose_blocks.dim, 11);
        assert_eq!(pose_blocks.blocks[0].free_axes, [false; 6]);
        assert_eq!(
            pose_blocks.blocks[1].free_axes,
            [true, true, true, true, false, true]
        );
        assert_eq!(pose_blocks.blocks[1].offset, 0);
        assert_eq!(pose_blocks.blocks[2].offset, 5);
    }

    #[test]
    fn two_cams_from_world_gauge_falls_back_when_baseline_is_degenerate() {
        let frames = vec![frame(0), frame(1)];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::identity());
        let mut pose_blocks = variable_pose_blocks(&reconstruction, None, &[], &[], false);

        assert!(!apply_two_cams_from_world_gauge(
            &mut pose_blocks,
            &reconstruction,
            &BundleAdjustmentOptions::default(),
            &[
                BaObservation {
                    image: 0,
                    point: 0,
                    xy: [0.0, 0.0],
                },
                BaObservation {
                    image: 1,
                    point: 0,
                    xy: [0.0, 0.0],
                },
            ],
        ));
        reindex_pose_blocks(&mut pose_blocks);

        assert_eq!(pose_blocks.dim, 12);
        assert!(pose_blocks
            .blocks
            .iter()
            .all(|block| block.free_axes == [true; 6]));
    }

    #[test]
    fn bundle_adjustment_uses_frame_pose_blocks_for_rig_images() {
        let mut frames = vec![frame(0), frame(1), frame(2)];
        let camera = CameraModel::new_pinhole(120, 100, 80.0, 80.0, 60.0, 50.0);
        let ref_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 11,
        };
        let aux_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 12,
        };
        let sensor_from_rig = SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(0.25, 0.0, 0.0));
        let rig_from_world = SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(0.1, 0.0, 0.0));
        let outside_pose = SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(0.8, 0.02, 0.0));
        let poses = [
            rig_from_world,
            sensor_from_rig.compose(&rig_from_world),
            outside_pose,
        ];
        let scene_points = vec![
            [-0.3, -0.2, 2.2],
            [0.0, -0.2, 2.1],
            [0.3, -0.1, 2.3],
            [-0.2, 0.2, 2.0],
            [0.2, 0.2, 2.4],
            [0.0, 0.0, 2.6],
        ];
        for image in 0..frames.len() {
            frames[image].keypoints = scene_points
                .iter()
                .map(|&point| {
                    let xy = project_point(camera, poses[image], point).unwrap();
                    rustslam::KeyPoint::new(xy[0] as f32, xy[1] as f32)
                })
                .collect();
            frames[image].colors = vec![[0, 0, 0]; frames[image].keypoints.len()];
        }

        let mut reconstruction = reconstruction(&frames);
        reconstruction.camera = camera;
        reconstruction.cameras = vec![camera];
        reconstruction.rigs = vec![Rig {
            rig_id: 3,
            ref_sensor_id: Some(ref_sensor.clone()),
            sensors: vec![
                RigSensor {
                    sensor_id: ref_sensor.clone(),
                    sensor_from_rig: None,
                },
                RigSensor {
                    sensor_id: aux_sensor.clone(),
                    sensor_from_rig: Some(Rigid3::from_se3(sensor_from_rig)),
                },
            ],
        }];
        reconstruction.frames = vec![Frame {
            frame_id: 9,
            rig_id: 3,
            rig_from_world: Rigid3::from_se3(rig_from_world),
            data_ids: vec![
                DataId {
                    sensor_id: ref_sensor,
                    data_id: reconstruction.image_id(0) as u64,
                },
                DataId {
                    sensor_id: aux_sensor.clone(),
                    data_id: reconstruction.image_id(1) as u64,
                },
            ],
        }];
        reconstruction.image_frame_indices = vec![Some(0), Some(0), None];
        reconstruction.poses[0] = Some(SE3::from_quat_translation(
            Quat::IDENTITY,
            Vec3::new(0.0, 0.0, 0.0),
        ));
        reconstruction.poses[1] = Some(SE3::from_quat_translation(
            Quat::IDENTITY,
            Vec3::new(3.0, 0.0, 0.0),
        ));
        reconstruction.poses[2] = Some(outside_pose);
        for (idx, xyz) in scene_points.into_iter().enumerate() {
            for image in 0..frames.len() {
                reconstruction.observations[image][idx] = Some(idx);
            }
            reconstruction.points.push(Point3D {
                xyz,
                color: [0, 0, 0],
                error: 0.0,
                track: vec![
                    TrackObservation {
                        image: 0,
                        feature: idx,
                    },
                    TrackObservation {
                        image: 1,
                        feature: idx,
                    },
                    TrackObservation {
                        image: 2,
                        feature: idx,
                    },
                ],
            });
            reconstruction.point_ids.push(idx as u64 + 1);
        }

        let report = refine_bundle_adjustment_native(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 4,
                loss_function: BundleAdjustmentLoss::Huber { scale: 4.0 },
                max_observation_error_px: 200.0,
                variable_images: Some(vec![0, 1]),
                constant_images: vec![2],
                constant_sensor_from_rig: vec![aux_sensor.clone()],
                point_ids: Some((0..6).collect()),
                ..BundleAdjustmentOptions::default()
            },
        )
        .unwrap();

        assert!(report.final_cost <= report.initial_cost);
        let rig_pose = reconstruction.frames[0].rig_from_world.to_se3();
        let expected_aux_pose = sensor_from_rig.compose(&rig_pose);
        assert_eq!(
            reconstruction.poses[0].unwrap().translation(),
            rig_pose.translation()
        );
        assert!(translation_distance(reconstruction.poses[1].unwrap(), expected_aux_pose) < 1.0e-5);
    }

    #[test]
    fn bundle_adjustment_refines_sensor_from_rig_when_not_constant() {
        let SensorBaFixture {
            frames,
            mut reconstruction,
            aux_sensor,
            initial_sensor_from_rig,
            true_sensor_from_rig,
            ..
        } = sensor_ba_fixture();
        let initial_error = translation_distance(initial_sensor_from_rig, true_sensor_from_rig);

        let report = refine_bundle_adjustment_native(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 8,
                loss_function: BundleAdjustmentLoss::Huber { scale: 4.0 },
                max_observation_error_px: 200.0,
                variable_images: Some(vec![0, 1]),
                constant_images: vec![2, 3],
                point_ids: Some((0..8).collect()),
                ..BundleAdjustmentOptions::default()
            },
        )
        .unwrap();

        let refined_sensor_from_rig = sensor_from_rig_pose(&reconstruction, 3, &aux_sensor);
        assert!(report.final_cost < report.initial_cost);
        assert!(
            translation_distance(refined_sensor_from_rig, true_sensor_from_rig) < initial_error
        );
    }

    #[test]
    fn bundle_adjustment_frame_pose_prior_pulls_rig_aux_camera_center() {
        use super::super::BundleAdjustmentPosePrior;

        let SensorBaFixture {
            frames,
            mut reconstruction,
            ..
        } = sensor_ba_fixture();
        let start_center = camera_center_world(reconstruction.poses[1].unwrap());
        let prior_center = [
            start_center[0] + 0.35,
            start_center[1] + 0.08,
            start_center[2],
        ];
        let start_error = center_error(start_center, prior_center);

        let report = refine_bundle_adjustment_native(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 12,
                gradient_tolerance: 0.0,
                loss_function: BundleAdjustmentLoss::Huber { scale: 4.0 },
                max_observation_error_px: 200.0,
                variable_images: Some(vec![0, 1]),
                constant_images: vec![2, 3],
                point_ids: Some((0..8).collect()),
                pose_priors: vec![BundleAdjustmentPosePrior::new(1, prior_center)],
                prior_position_fallback_stddev: 0.01,
                ..BundleAdjustmentOptions::default()
            },
        )
        .unwrap();

        let end_center = camera_center_world(reconstruction.poses[1].unwrap());
        let end_error = center_error(end_center, prior_center);
        assert!(report.is_solution_usable());
        assert!(
            end_error < start_error,
            "start_error={start_error} end_error={end_error}"
        );
    }

    #[test]
    fn bundle_adjustment_keeps_constant_sensor_from_rig_fixed() {
        let SensorBaFixture {
            frames,
            mut reconstruction,
            aux_sensor,
            initial_sensor_from_rig,
            ..
        } = sensor_ba_fixture();

        let report = refine_bundle_adjustment_native(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 4,
                loss_function: BundleAdjustmentLoss::Huber { scale: 4.0 },
                max_observation_error_px: 200.0,
                variable_images: Some(vec![0, 1]),
                constant_images: vec![2, 3],
                constant_sensor_from_rig: vec![aux_sensor.clone()],
                point_ids: Some((0..8).collect()),
                ..BundleAdjustmentOptions::default()
            },
        )
        .unwrap();

        assert!(report.is_solution_usable());
        assert!(
            translation_distance(
                sensor_from_rig_pose(&reconstruction, 3, &aux_sensor),
                initial_sensor_from_rig,
            ) < 1.0e-6
        );
    }

    #[test]
    fn bundle_adjustment_uses_constant_points_without_moving_them() {
        let mut frames = vec![frame(0), frame(1)];
        frames[0].keypoints = vec![
            rustslam::KeyPoint::new(45.0, 45.0),
            rustslam::KeyPoint::new(55.0, 45.0),
            rustslam::KeyPoint::new(45.0, 55.0),
            rustslam::KeyPoint::new(55.0, 55.0),
            rustslam::KeyPoint::new(50.0, 50.0),
            rustslam::KeyPoint::new(60.0, 50.0),
        ];
        frames[1].keypoints = vec![
            rustslam::KeyPoint::new(70.0, 45.0),
            rustslam::KeyPoint::new(80.0, 45.0),
            rustslam::KeyPoint::new(70.0, 55.0),
            rustslam::KeyPoint::new(80.0, 55.0),
            rustslam::KeyPoint::new(75.0, 50.0),
            rustslam::KeyPoint::new(85.0, 50.0),
        ];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::from_quat_translation(
            Quat::IDENTITY,
            Vec3::new(0.6, 0.0, 0.0),
        ));
        for (idx, xyz) in [
            [-0.2, -0.2, 2.0],
            [0.2, -0.2, 2.0],
            [-0.2, 0.2, 2.0],
            [0.2, 0.2, 2.0],
            [0.0, 0.0, 2.0],
            [0.4, 0.0, 2.0],
        ]
        .into_iter()
        .enumerate()
        {
            reconstruction.observations[0][idx] = Some(idx);
            reconstruction.observations[1][idx] = Some(idx);
            reconstruction.points.push(Point3D {
                xyz,
                color: [0, 0, 0],
                error: 0.0,
                track: vec![
                    TrackObservation {
                        image: 0,
                        feature: idx,
                    },
                    TrackObservation {
                        image: 1,
                        feature: idx,
                    },
                ],
            });
            reconstruction.point_ids.push(idx as u64 + 1);
        }
        let original_points = reconstruction
            .points
            .iter()
            .map(|point| point.xyz)
            .collect::<Vec<_>>();

        let report = refine_bundle_adjustment_native(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 3,
                loss_function: BundleAdjustmentLoss::Huber { scale: 4.0 },
                max_observation_error_px: 50.0,
                variable_images: Some(vec![1]),
                constant_images: vec![0],
                constant_point_ids: Some((0..6).collect()),
                ..BundleAdjustmentOptions::default()
            },
        )
        .unwrap();

        assert!(report.final_cost < report.initial_cost);
        assert!(report.is_solution_usable());
        assert_eq!(report.residuals, frames[1].keypoints.len() * 2);
        assert!(report.residuals < report.observations * 2);
        assert!(report.attempted_iterations >= report.iterations);
        assert_eq!(report.successful_steps, report.iterations);
        assert_ne!(
            report.termination_type,
            BundleAdjustmentTerminationType::Failure
        );
        assert_eq!(
            reconstruction
                .points
                .iter()
                .map(|point| point.xyz)
                .collect::<Vec<_>>(),
            original_points
        );
    }

    #[test]
    fn bundle_adjustment_refines_points_when_all_nonpoint_parameters_are_fixed() {
        let camera = CameraModel::new_pinhole(120, 100, 70.0, 70.0, 60.0, 50.0);
        let poses = [
            SE3::identity(),
            SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(0.5, 0.0, 0.0)),
        ];
        let true_point = [0.2, 0.1, 3.0];
        let mut frames = vec![frame(0), frame(1)];
        for image in 0..2 {
            let xy = project_point(camera, poses[image], true_point).unwrap();
            frames[image].keypoints = vec![rustslam::KeyPoint::new(xy[0] as f32, xy[1] as f32)];
        }
        let mut reconstruction = reconstruction(&frames);
        reconstruction.camera = camera;
        reconstruction.cameras = vec![camera];
        reconstruction.poses[0] = Some(poses[0]);
        reconstruction.poses[1] = Some(poses[1]);
        reconstruction.observations[0][0] = Some(0);
        reconstruction.observations[1][0] = Some(0);
        reconstruction.points.push(Point3D {
            xyz: [0.45, -0.15, 2.6],
            color: [0, 0, 0],
            error: 0.0,
            track: vec![
                TrackObservation {
                    image: 0,
                    feature: 0,
                },
                TrackObservation {
                    image: 1,
                    feature: 0,
                },
            ],
        });
        reconstruction.point_ids.push(1);
        let initial_pose0 = reconstruction.poses[0].unwrap();
        let initial_pose1 = reconstruction.poses[1].unwrap();
        let initial_camera = reconstruction.cameras[0];
        let initial_distance =
            Vec3::from_array(reconstruction.points[0].xyz).distance(Vec3::from_array(true_point));

        let report = refine_bundle_adjustment_native(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 8,
                gradient_tolerance: 0.0,
                loss_function: BundleAdjustmentLoss::Huber { scale: 4.0 },
                max_observation_error_px: 200.0,
                variable_images: Some(Vec::new()),
                constant_images: vec![0, 1],
                constant_cameras: vec![0],
                point_ids: Some(vec![0]),
                ..BundleAdjustmentOptions::default()
            },
        )
        .unwrap();

        assert!(report.final_cost < report.initial_cost);
        assert!(report.is_solution_usable());
        assert_eq!(report.effective_parameters, 3);
        assert_eq!(
            reconstruction.poses[0].unwrap().translation(),
            initial_pose0.translation()
        );
        assert_eq!(
            reconstruction.poses[1].unwrap().translation(),
            initial_pose1.translation()
        );
        assert_eq!(reconstruction.cameras[0].fx, initial_camera.fx);
        let final_distance =
            Vec3::from_array(reconstruction.points[0].xyz).distance(Vec3::from_array(true_point));
        assert!(final_distance < initial_distance);
    }

    #[test]
    fn bundle_adjustment_can_refine_pose_from_single_observation_points_when_enabled() {
        let camera = CameraModel::new_pinhole(120, 100, 80.0, 80.0, 60.0, 50.0);
        let true_pose =
            SE3::from_quat_translation(Quat::from_rotation_y(0.04), Vec3::new(0.05, -0.02, 0.03));
        let initial_pose =
            SE3::from_quat_translation(Quat::from_rotation_y(0.08), Vec3::new(0.22, -0.05, 0.08));
        let scene_points = [
            [-0.5, -0.3, 3.0],
            [-0.2, -0.2, 3.2],
            [0.1, -0.25, 3.1],
            [0.4, -0.1, 3.4],
            [-0.4, 0.1, 3.3],
            [0.0, 0.0, 3.0],
            [0.35, 0.15, 3.5],
            [-0.15, 0.35, 3.2],
        ];
        let mut frames = vec![frame(0)];
        frames[0].keypoints = scene_points
            .iter()
            .map(|&point| {
                let xy = project_point(camera, true_pose, point).unwrap();
                rustslam::KeyPoint::new(xy[0] as f32, xy[1] as f32)
            })
            .collect();
        let mut reconstruction = reconstruction(&frames);
        reconstruction.camera = camera;
        reconstruction.cameras = vec![camera];
        reconstruction.poses[0] = Some(initial_pose);
        for (idx, xyz) in scene_points.into_iter().enumerate() {
            reconstruction.observations[0][idx] = Some(idx);
            reconstruction.points.push(Point3D {
                xyz,
                color: [0, 0, 0],
                error: 0.0,
                track: vec![TrackObservation {
                    image: 0,
                    feature: idx,
                }],
            });
            reconstruction.point_ids.push(idx as u64 + 1);
        }
        let initial_error = translation_distance(initial_pose, true_pose);

        let skipped = refine_bundle_adjustment(
            &frames,
            &mut reconstruction.clone(),
            BundleAdjustmentOptions {
                iterations: 4,
                max_observation_error_px: 100.0,
                variable_images: Some(vec![0]),
                constant_point_ids: Some((0..scene_points.len()).collect()),
                ..BundleAdjustmentOptions::default()
            },
        );
        assert!(skipped.is_none());

        let report = refine_bundle_adjustment(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 8,
                max_observation_error_px: 100.0,
                variable_images: Some(vec![0]),
                constant_point_ids: Some((0..scene_points.len()).collect()),
                allow_single_observation_points: true,
                ..BundleAdjustmentOptions::default()
            },
        )
        .unwrap();

        assert!(report.is_solution_usable());
        assert!(report.final_cost < report.initial_cost);
        assert!(translation_distance(reconstruction.poses[0].unwrap(), true_pose) < initial_error);
    }

    #[test]
    fn bundle_adjustment_refines_focal_length_when_enabled() {
        let true_camera = CameraModel::new_pinhole(100, 100, 60.0, 60.0, 50.0, 50.0);
        let initial_camera = CameraModel::new_pinhole(100, 100, 45.0, 45.0, 50.0, 50.0);
        let poses = [
            SE3::identity(),
            SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(0.45, 0.02, 0.0)),
        ];
        let scene_points = [
            [-0.6, -0.4, 3.2],
            [-0.2, -0.4, 3.0],
            [0.3, -0.35, 3.4],
            [0.7, -0.2, 3.6],
            [-0.5, 0.1, 3.3],
            [0.0, 0.0, 3.1],
            [0.45, 0.15, 3.5],
            [-0.25, 0.45, 3.7],
            [0.55, 0.5, 3.8],
        ];
        let mut frames = vec![frame(0), frame(1)];
        for image in 0..2 {
            frames[image].keypoints = scene_points
                .iter()
                .map(|&point| {
                    let xy = project_point(true_camera, poses[image], point).unwrap();
                    rustslam::KeyPoint::new(xy[0] as f32, xy[1] as f32)
                })
                .collect();
        }

        let mut reconstruction = reconstruction(&frames);
        reconstruction.camera = initial_camera;
        reconstruction.cameras = vec![initial_camera];
        reconstruction.poses[0] = Some(poses[0]);
        reconstruction.poses[1] = Some(poses[1]);
        for (idx, xyz) in scene_points.into_iter().enumerate() {
            reconstruction.observations[0][idx] = Some(idx);
            reconstruction.observations[1][idx] = Some(idx);
            reconstruction.points.push(Point3D {
                xyz,
                color: [0, 0, 0],
                error: 0.0,
                track: vec![
                    TrackObservation {
                        image: 0,
                        feature: idx,
                    },
                    TrackObservation {
                        image: 1,
                        feature: idx,
                    },
                ],
            });
            reconstruction.point_ids.push(idx as u64 + 1);
        }

        let report = refine_bundle_adjustment_native(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 8,
                loss_function: BundleAdjustmentLoss::Huber { scale: 4.0 },
                max_observation_error_px: 50.0,
                variable_images: Some(Vec::new()),
                variable_cameras: Some(vec![0]),
                refine_focal_length: true,
                ..BundleAdjustmentOptions::default()
            },
        )
        .unwrap();

        assert!(report.final_cost < report.initial_cost);
        assert!(report.is_solution_usable());
        assert_eq!(report.successful_steps, report.iterations);
        assert!(report.attempted_iterations >= report.iterations);
        assert!(report.residuals <= report.observations * 2);
        assert!(report.brief_report().contains("termination="));
        assert!(report.brief_report().contains("step_quality="));
        assert_eq!(report.effective_parameters, 2 + scene_points.len() * 3);
        assert!(report.linear_solver_iterations >= report.successful_steps);
        assert!(report.gradient_max_norm.is_finite());
        assert!(report.step_norm.is_finite());
        assert!(report.step_quality.is_finite());
        assert!(report.step_quality > 0.0);
        assert!(report.damping.is_finite());
        assert!(
            (60.0 - reconstruction.cameras[0].fx as f64).abs()
                < (60.0 - initial_camera.fx as f64).abs()
        );
        assert_eq!(reconstruction.camera.fx, reconstruction.cameras[0].fx);
    }

    #[test]
    fn bundle_adjustment_uses_function_tolerance_for_convergence() {
        let true_camera = CameraModel::new_pinhole(100, 100, 60.0, 60.0, 50.0, 50.0);
        let initial_camera = CameraModel::new_pinhole(100, 100, 45.0, 45.0, 50.0, 50.0);
        let poses = [
            SE3::identity(),
            SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(0.45, 0.02, 0.0)),
        ];
        let scene_points = [
            [-0.6, -0.4, 3.2],
            [-0.2, -0.4, 3.0],
            [0.3, -0.35, 3.4],
            [0.7, -0.2, 3.6],
            [-0.5, 0.1, 3.3],
            [0.0, 0.0, 3.1],
            [0.45, 0.15, 3.5],
            [-0.25, 0.45, 3.7],
            [0.55, 0.5, 3.8],
        ];
        let mut frames = vec![frame(0), frame(1)];
        for image in 0..2 {
            frames[image].keypoints = scene_points
                .iter()
                .map(|&point| {
                    let xy = project_point(true_camera, poses[image], point).unwrap();
                    rustslam::KeyPoint::new(xy[0] as f32, xy[1] as f32)
                })
                .collect();
        }

        let mut reconstruction = reconstruction(&frames);
        reconstruction.camera = initial_camera;
        reconstruction.cameras = vec![initial_camera];
        reconstruction.poses[0] = Some(poses[0]);
        reconstruction.poses[1] = Some(poses[1]);
        for (idx, xyz) in scene_points.into_iter().enumerate() {
            reconstruction.observations[0][idx] = Some(idx);
            reconstruction.observations[1][idx] = Some(idx);
            reconstruction.points.push(Point3D {
                xyz,
                color: [0, 0, 0],
                error: 0.0,
                track: vec![
                    TrackObservation {
                        image: 0,
                        feature: idx,
                    },
                    TrackObservation {
                        image: 1,
                        feature: idx,
                    },
                ],
            });
            reconstruction.point_ids.push(idx as u64 + 1);
        }

        let report = refine_bundle_adjustment_native(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 8,
                function_tolerance: 1.0,
                gradient_tolerance: 0.0,
                loss_function: BundleAdjustmentLoss::Huber { scale: 4.0 },
                max_observation_error_px: 50.0,
                variable_images: Some(Vec::new()),
                variable_cameras: Some(vec![0]),
                refine_focal_length: true,
                ..BundleAdjustmentOptions::default()
            },
        )
        .unwrap();

        assert_eq!(
            report.termination_type,
            BundleAdjustmentTerminationType::Convergence
        );
        assert_eq!(
            report.termination_reason,
            BundleAdjustmentTerminationReason::FunctionTolerance
        );
        assert!(report.attempted_iterations < 8);
    }

    #[test]
    fn bundle_adjustment_uses_parameter_tolerance_for_convergence() {
        let true_camera = CameraModel::new_pinhole(100, 100, 60.0, 60.0, 50.0, 50.0);
        let initial_camera = CameraModel::new_pinhole(100, 100, 45.0, 45.0, 50.0, 50.0);
        let poses = [
            SE3::identity(),
            SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(0.45, 0.02, 0.0)),
        ];
        let scene_points = [
            [-0.6, -0.4, 3.2],
            [-0.2, -0.4, 3.0],
            [0.3, -0.35, 3.4],
            [0.7, -0.2, 3.6],
            [-0.5, 0.1, 3.3],
            [0.0, 0.0, 3.1],
            [0.45, 0.15, 3.5],
            [-0.25, 0.45, 3.7],
            [0.55, 0.5, 3.8],
        ];
        let mut frames = vec![frame(0), frame(1)];
        for image in 0..2 {
            frames[image].keypoints = scene_points
                .iter()
                .map(|&point| {
                    let xy = project_point(true_camera, poses[image], point).unwrap();
                    rustslam::KeyPoint::new(xy[0] as f32, xy[1] as f32)
                })
                .collect();
        }

        let mut reconstruction = reconstruction(&frames);
        reconstruction.camera = initial_camera;
        reconstruction.cameras = vec![initial_camera];
        reconstruction.poses[0] = Some(poses[0]);
        reconstruction.poses[1] = Some(poses[1]);
        for (idx, xyz) in scene_points.into_iter().enumerate() {
            reconstruction.observations[0][idx] = Some(idx);
            reconstruction.observations[1][idx] = Some(idx);
            reconstruction.points.push(Point3D {
                xyz,
                color: [0, 0, 0],
                error: 0.0,
                track: vec![
                    TrackObservation {
                        image: 0,
                        feature: idx,
                    },
                    TrackObservation {
                        image: 1,
                        feature: idx,
                    },
                ],
            });
            reconstruction.point_ids.push(idx as u64 + 1);
        }

        let report = refine_bundle_adjustment_native(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 8,
                gradient_tolerance: 0.0,
                parameter_tolerance: 1.0e6,
                loss_function: BundleAdjustmentLoss::Huber { scale: 4.0 },
                max_observation_error_px: 50.0,
                variable_images: Some(Vec::new()),
                variable_cameras: Some(vec![0]),
                refine_focal_length: true,
                ..BundleAdjustmentOptions::default()
            },
        )
        .unwrap();

        assert_eq!(
            report.termination_type,
            BundleAdjustmentTerminationType::Convergence
        );
        assert_eq!(
            report.termination_reason,
            BundleAdjustmentTerminationReason::ParameterTolerance
        );
        assert_eq!(report.attempted_iterations, 1);
        assert!(report.step_norm <= 1.0e6);
        assert!(report.gradient_max_norm.is_finite());
    }

    #[test]
    fn bundle_adjustment_respects_zero_linear_solver_iteration_budget() {
        let true_camera = CameraModel::new_pinhole(100, 100, 60.0, 60.0, 50.0, 50.0);
        let initial_camera = CameraModel::new_pinhole(100, 100, 45.0, 45.0, 50.0, 50.0);
        let poses = [
            SE3::identity(),
            SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(0.45, 0.02, 0.0)),
        ];
        let scene_points = [
            [-0.6, -0.4, 3.2],
            [-0.2, -0.4, 3.0],
            [0.3, -0.35, 3.4],
            [0.7, -0.2, 3.6],
            [-0.5, 0.1, 3.3],
            [0.0, 0.0, 3.1],
            [0.45, 0.15, 3.5],
            [-0.25, 0.45, 3.7],
            [0.55, 0.5, 3.8],
        ];
        let mut frames = vec![frame(0), frame(1)];
        for image in 0..2 {
            frames[image].keypoints = scene_points
                .iter()
                .map(|&point| {
                    let xy = project_point(true_camera, poses[image], point).unwrap();
                    rustslam::KeyPoint::new(xy[0] as f32, xy[1] as f32)
                })
                .collect();
        }

        let mut reconstruction = reconstruction(&frames);
        reconstruction.camera = initial_camera;
        reconstruction.cameras = vec![initial_camera];
        reconstruction.poses[0] = Some(poses[0]);
        reconstruction.poses[1] = Some(poses[1]);
        for (idx, xyz) in scene_points.into_iter().enumerate() {
            reconstruction.observations[0][idx] = Some(idx);
            reconstruction.observations[1][idx] = Some(idx);
            reconstruction.points.push(Point3D {
                xyz,
                color: [0, 0, 0],
                error: 0.0,
                track: vec![
                    TrackObservation {
                        image: 0,
                        feature: idx,
                    },
                    TrackObservation {
                        image: 1,
                        feature: idx,
                    },
                ],
            });
            reconstruction.point_ids.push(idx as u64 + 1);
        }

        let report = refine_bundle_adjustment_native(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 3,
                gradient_tolerance: 0.0,
                max_linear_solver_iterations: 0,
                max_num_consecutive_invalid_steps: 1,
                loss_function: BundleAdjustmentLoss::Huber { scale: 4.0 },
                max_observation_error_px: 50.0,
                variable_images: Some(Vec::new()),
                variable_cameras: Some(vec![0]),
                refine_focal_length: true,
                ..BundleAdjustmentOptions::default()
            },
        )
        .unwrap();

        assert_eq!(
            report.termination_type,
            BundleAdjustmentTerminationType::Failure
        );
        assert_eq!(
            report.termination_reason,
            BundleAdjustmentTerminationReason::MaxConsecutiveInvalidSteps
        );
        assert_eq!(report.attempted_iterations, 1);
        assert_eq!(report.linear_solver_iterations, 0);
        assert_eq!(report.linear_solve_failures, 1);
    }

    #[test]
    fn bundle_adjustment_report_zero_iterations_matches_no_convergence() {
        let mut frames = vec![frame(0), frame(1)];
        frames[0].keypoints = vec![
            rustslam::KeyPoint::new(45.0, 45.0),
            rustslam::KeyPoint::new(55.0, 45.0),
            rustslam::KeyPoint::new(45.0, 55.0),
            rustslam::KeyPoint::new(55.0, 55.0),
            rustslam::KeyPoint::new(50.0, 50.0),
            rustslam::KeyPoint::new(60.0, 50.0),
        ];
        frames[1].keypoints = vec![
            rustslam::KeyPoint::new(70.0, 45.0),
            rustslam::KeyPoint::new(80.0, 45.0),
            rustslam::KeyPoint::new(70.0, 55.0),
            rustslam::KeyPoint::new(80.0, 55.0),
            rustslam::KeyPoint::new(75.0, 50.0),
            rustslam::KeyPoint::new(85.0, 50.0),
        ];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::from_quat_translation(
            Quat::IDENTITY,
            Vec3::new(0.8, 0.05, 0.0),
        ));
        for (idx, xyz) in [
            [-0.2, -0.2, 2.0],
            [0.2, -0.2, 2.0],
            [-0.2, 0.2, 2.0],
            [0.2, 0.2, 2.0],
            [0.0, 0.0, 2.0],
            [0.4, 0.0, 2.0],
        ]
        .into_iter()
        .enumerate()
        {
            reconstruction.observations[0][idx] = Some(idx);
            reconstruction.observations[1][idx] = Some(idx);
            reconstruction.points.push(Point3D {
                xyz,
                color: [0, 0, 0],
                error: 0.0,
                track: vec![
                    TrackObservation {
                        image: 0,
                        feature: idx,
                    },
                    TrackObservation {
                        image: 1,
                        feature: idx,
                    },
                ],
            });
            reconstruction.point_ids.push(idx as u64 + 1);
        }

        let report = refine_bundle_adjustment_native(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 0,
                loss_function: BundleAdjustmentLoss::Huber { scale: 4.0 },
                max_observation_error_px: 50.0,
                variable_images: Some(vec![1]),
                constant_images: vec![0],
                point_ids: Some((0..6).collect()),
                ..BundleAdjustmentOptions::default()
            },
        )
        .unwrap();

        assert!(report.is_solution_usable());
        assert_eq!(report.iterations, 0);
        assert_eq!(report.attempted_iterations, 0);
        assert_eq!(report.successful_steps, 0);
        assert_eq!(report.unsuccessful_steps, 0);
        assert_eq!(report.residuals, report.observations * 2);
        assert_eq!(
            report.termination_type,
            BundleAdjustmentTerminationType::NoConvergence
        );
        assert_eq!(
            report.termination_reason,
            BundleAdjustmentTerminationReason::MaxIterations
        );
    }

    fn reconstruction(frames: &[ImageFrame]) -> Reconstruction {
        Reconstruction {
            camera: CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0),
            cameras: vec![CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0)],
            camera_ids: vec![1],
            rigs: Vec::new(),
            frames: Vec::new(),
            image_names: frames.iter().map(|frame| frame.name.clone()).collect(),
            image_paths: frames.iter().map(|frame| frame.path.clone()).collect(),
            image_ids: (0..frames.len()).map(|idx| idx as u32 + 1).collect(),
            image_camera_indices: vec![0; frames.len()],
            image_frame_indices: vec![None; frames.len()],
            poses: vec![None; frames.len()],
            observations: frames
                .iter()
                .map(|frame| vec![None; frame.keypoints.len()])
                .collect(),
            keypoints: frames.iter().map(|frame| frame.keypoints.clone()).collect(),
            point_ids: Vec::new(),
            points: Vec::new(),
        }
    }

    struct SensorBaFixture {
        frames: Vec<ImageFrame>,
        reconstruction: Reconstruction,
        aux_sensor: SensorId,
        initial_sensor_from_rig: SE3,
        true_sensor_from_rig: SE3,
    }

    fn sensor_ba_fixture() -> SensorBaFixture {
        let mut frames = vec![frame(0), frame(1), frame(2), frame(3)];
        let camera = CameraModel::new_pinhole(160, 120, 90.0, 90.0, 80.0, 60.0);
        let ref_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 11,
        };
        let aux_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 12,
        };
        let true_sensor_from_rig =
            SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(0.35, 0.02, 0.0));
        let initial_sensor_from_rig =
            SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(0.1, -0.04, 0.0));
        let rig_poses = [
            SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(0.0, 0.0, 0.0)),
            SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(0.18, 0.03, 0.0)),
        ];
        let outside_poses = [
            SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(0.75, 0.02, 0.0)),
            SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(-0.65, 0.01, 0.0)),
        ];
        let poses = [
            rig_poses[0],
            true_sensor_from_rig.compose(&rig_poses[0]),
            outside_poses[0],
            outside_poses[1],
        ];
        let scene_points = vec![
            [-0.35, -0.25, 2.5],
            [-0.05, -0.2, 2.3],
            [0.25, -0.15, 2.7],
            [0.45, 0.05, 2.9],
            [-0.25, 0.2, 2.4],
            [0.05, 0.25, 2.6],
            [0.35, 0.3, 3.0],
            [0.0, 0.0, 3.2],
        ];
        for image in 0..frames.len() {
            frames[image].keypoints = scene_points
                .iter()
                .map(|&point| {
                    let xy = project_point(camera, poses[image], point).unwrap();
                    rustslam::KeyPoint::new(xy[0] as f32, xy[1] as f32)
                })
                .collect();
            frames[image].colors = vec![[0, 0, 0]; frames[image].keypoints.len()];
        }

        let mut reconstruction = reconstruction(&frames);
        reconstruction.camera = camera;
        reconstruction.cameras = vec![camera];
        reconstruction.rigs = vec![Rig {
            rig_id: 3,
            ref_sensor_id: Some(ref_sensor.clone()),
            sensors: vec![
                RigSensor {
                    sensor_id: ref_sensor.clone(),
                    sensor_from_rig: None,
                },
                RigSensor {
                    sensor_id: aux_sensor.clone(),
                    sensor_from_rig: Some(Rigid3::from_se3(initial_sensor_from_rig)),
                },
            ],
        }];
        reconstruction.frames = vec![
            Frame {
                frame_id: 9,
                rig_id: 3,
                rig_from_world: Rigid3::from_se3(rig_poses[0]),
                data_ids: vec![
                    DataId {
                        sensor_id: ref_sensor.clone(),
                        data_id: reconstruction.image_id(0) as u64,
                    },
                    DataId {
                        sensor_id: aux_sensor.clone(),
                        data_id: reconstruction.image_id(1) as u64,
                    },
                ],
            },
            Frame {
                frame_id: 10,
                rig_id: 3,
                rig_from_world: Rigid3::from_se3(rig_poses[1]),
                data_ids: Vec::new(),
            },
        ];
        reconstruction.image_frame_indices = vec![Some(0), Some(0), None, None];
        reconstruction.poses[0] = Some(rig_poses[0]);
        reconstruction.poses[1] = Some(initial_sensor_from_rig.compose(&rig_poses[0]));
        reconstruction.poses[2] = Some(outside_poses[0]);
        reconstruction.poses[3] = Some(outside_poses[1]);
        for (idx, xyz) in scene_points.into_iter().enumerate() {
            for image in 0..frames.len() {
                reconstruction.observations[image][idx] = Some(idx);
            }
            reconstruction.points.push(Point3D {
                xyz,
                color: [0, 0, 0],
                error: 0.0,
                track: vec![
                    TrackObservation {
                        image: 0,
                        feature: idx,
                    },
                    TrackObservation {
                        image: 1,
                        feature: idx,
                    },
                    TrackObservation {
                        image: 2,
                        feature: idx,
                    },
                    TrackObservation {
                        image: 3,
                        feature: idx,
                    },
                ],
            });
            reconstruction.point_ids.push(idx as u64 + 1);
        }

        SensorBaFixture {
            frames,
            reconstruction,
            aux_sensor,
            initial_sensor_from_rig,
            true_sensor_from_rig,
        }
    }

    fn sensor_from_rig_pose(
        reconstruction: &Reconstruction,
        rig_id: u32,
        sensor_id: &SensorId,
    ) -> SE3 {
        reconstruction
            .rigs
            .iter()
            .find(|rig| rig.rig_id == rig_id)
            .and_then(|rig| {
                rig.sensors
                    .iter()
                    .find(|sensor| &sensor.sensor_id == sensor_id)
            })
            .and_then(|sensor| sensor.sensor_from_rig.as_ref())
            .map(Rigid3::to_se3)
            .unwrap()
    }

    fn translation_distance(left: SE3, right: SE3) -> f32 {
        let left = left.translation();
        let right = right.translation();
        ((left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2) + (left[2] - right[2]).powi(2))
            .sqrt()
    }

    fn center_error(center: Vec3d, target: [f64; 3]) -> f64 {
        ((center[0] - target[0]).powi(2)
            + (center[1] - target[1]).powi(2)
            + (center[2] - target[2]).powi(2))
        .sqrt()
    }

    fn assert_jacobian_close(left: Mat2x6, right: Mat2x6, tolerance: f64) {
        let mut max_error = 0.0f64;
        let mut max_row = 0usize;
        let mut max_col = 0usize;
        for row in 0..2 {
            for col in 0..6 {
                let error = (left[(row, col)] - right[(row, col)]).abs();
                if error > max_error {
                    max_error = error;
                    max_row = row;
                    max_col = col;
                }
            }
        }
        assert!(
            max_error < tolerance,
            "row={max_row} col={max_col} left={} right={} max_error={max_error}",
            left[(max_row, max_col)],
            right[(max_row, max_col)]
        );
    }

    fn frame(id: usize) -> ImageFrame {
        ImageFrame {
            id,
            name: format!("{id}.jpg"),
            path: PathBuf::from(format!("{id}.jpg")),
            width: 100,
            height: 100,
            keypoints: Vec::new(),
            descriptors: Descriptors::new(),
            sift: crate::sift::SiftFeatures::default(),
            wide_descriptors: crate::wide::WideDescriptors {
                data: Vec::new(),
                dim: 0,
                count: 0,
            },
            strong_feature_indices: Vec::new(),
            colors: Vec::new(),
        }
    }
}
