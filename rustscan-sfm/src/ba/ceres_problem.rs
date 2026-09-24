use super::ceres_support::{
    analytic_frame_pose_jacobian, analytic_img_from_cam_jacobian, analytic_sensor_pose_jacobian,
    apply_two_cams_from_world_gauge, camera_by_index, camera_param_jacobian, camera_param_specs,
    count_variable_residuals, frame_sensor_from_rig, frame_sensor_key_for_image,
    projection_jacobians, sensor_pose_specs, set_frame_pose_block,
    sync_pose_blocks_for_sensor_changes, variable_pose_blocks, CameraParamSpec, PoseBlockKind,
    SensorPoseKey,
};
use super::shared::{
    add_three_point_gauge, bundle_adjustment_point_filter, collect_observations, project_point,
};
use super::{
    camera_center_world, position_prior_information_matrix, should_commit_ba_solution,
    BundleAdjustmentGauge, BundleAdjustmentLinearSolver, BundleAdjustmentLinearSolverPreference,
    BundleAdjustmentLoss, BundleAdjustmentOptions, BundleAdjustmentPreconditioner,
    BundleAdjustmentReport, BundleAdjustmentSparseLinearAlgebra, BundleAdjustmentTerminationReason,
    BundleAdjustmentTerminationType, POSE_PRIOR_JACOBIAN_EPS,
};
use crate::types::{CameraModel, ImageFrame, Reconstruction, Rigid3};
use ceres_solver::loss::LossFunction;
use ceres_solver::parameter_block::ParameterBlockOrIndex;
use ceres_solver::solver::{
    LinearSolverType, PreconditionerType, SolverOptions, SparseLinearAlgebraLibraryType,
    TerminationType,
};
use ceres_solver::{CostFunctionType, NllsProblem};
type Quat = nalgebra::UnitQuaternion<f32>;
type Vec3 = nalgebra::Vector3<f32>;
use nalgebra::SMatrix;
use rustscan_slam::SE3;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

type Mat3 = SMatrix<f64, 3, 3>;
type Mat3x7 = SMatrix<f64, 3, 7>;
type Mat2x7 = SMatrix<f64, 2, 7>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum PoseEntityKey {
    Image(usize),
    Frame(usize),
    Sensor(SensorPoseKey),
}

#[derive(Debug, Clone)]
enum PoseEval {
    Fixed(SE3),
    Image {
        handle: usize,
    },
    Frame {
        frame_handle: usize,
        sensor: FrameSensorEval,
    },
}

#[derive(Debug, Clone)]
enum FrameSensorEval {
    Fixed(SE3),
    Variable { handle: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParamRole {
    ImagePose,
    FramePose,
    SensorPose,
    Point,
    CameraParam(usize),
}

#[derive(Debug, Clone)]
struct ResidualBinding {
    xy: [f64; 2],
    param_roles: Vec<ParamRole>,
    pose_eval: PoseEval,
    camera_base: CameraModel,
}

#[derive(Debug, Clone)]
struct PosePriorBinding {
    prior_position: [f64; 3],
    sqrt_information: Mat3,
    param_roles: Vec<ParamRole>,
    pose_eval: PoseEval,
}

pub fn solve_bundle_adjustment_ceres(
    frames: &[ImageFrame],
    reconstruction: &mut Reconstruction,
    options: BundleAdjustmentOptions,
    control: Option<&crate::task::SfmTaskControl>,
) -> Option<BundleAdjustmentReport> {
    let ba_started = Instant::now();
    let setup_started = Instant::now();
    if reconstruction.points.is_empty() {
        return None;
    }

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

    let observations = collect_observations(
        frames,
        reconstruction,
        options.max_observation_error_px,
        point_filter.as_ref(),
        options.allow_single_observation_points,
    );
    if observations.is_empty() {
        return None;
    }

    if matches!(options.gauge, BundleAdjustmentGauge::ThreePoints) {
        add_three_point_gauge(&mut constant_point_filter, reconstruction, &observations);
    } else if matches!(options.gauge, BundleAdjustmentGauge::TwoCamsFromWorld) {
        apply_two_cams_from_world_gauge(&mut pose_blocks, reconstruction, &options, &observations);
    }

    let sensor_pose_specs = sensor_pose_specs(reconstruction, &pose_blocks, &options);
    let camera_param_specs = camera_param_specs(
        reconstruction,
        &observations,
        &options,
        pose_blocks.dim + sensor_pose_specs.len() * 6,
    );
    let sensor_lookup = sensor_pose_specs
        .iter()
        .map(|spec| (spec.key.clone(), spec))
        .collect::<HashMap<_, _>>();

    let mut block_values = HashMap::<usize, Vec<f64>>::new();
    let mut pose_entity_registry = HashMap::<PoseEntityKey, usize>::new();
    let mut pose_free_axes = HashMap::<usize, [bool; 6]>::new();
    let mut point_registry = HashMap::<usize, usize>::new();
    let mut camera_param_registry = HashMap::<(usize, usize), usize>::new();
    let mut constant_blocks = HashSet::<usize>::new();
    let mut next_param_index = 0usize;
    let mut frame_images = HashMap::<usize, Vec<usize>>::new();

    for block in &pose_blocks.blocks {
        match block.kind {
            PoseBlockKind::Image(image) => {
                let Some(pose) = reconstruction.poses.get(image).copied().flatten() else {
                    continue;
                };
                register_pose_entity(
                    PoseEntityKey::Image(image),
                    block.free_axes,
                    se3_to_pose_params(pose),
                    &mut pose_entity_registry,
                    &mut block_values,
                    &mut next_param_index,
                    &mut constant_blocks,
                    &mut pose_free_axes,
                );
            }
            PoseBlockKind::Frame(frame_idx) => {
                frame_images.insert(frame_idx, block.images.clone());
                let Some(frame) = reconstruction.frames.get(frame_idx) else {
                    continue;
                };
                register_pose_entity(
                    PoseEntityKey::Frame(frame_idx),
                    block.free_axes,
                    se3_to_pose_params(frame.rig_from_world.to_se3()),
                    &mut pose_entity_registry,
                    &mut block_values,
                    &mut next_param_index,
                    &mut constant_blocks,
                    &mut pose_free_axes,
                );
            }
        }
    }

    for spec in &sensor_pose_specs {
        let Some(pose) = reconstruction.sensor_from_rig(spec.key.rig_id, &spec.key.sensor_id)
        else {
            continue;
        };
        register_pose_entity(
            PoseEntityKey::Sensor(spec.key.clone()),
            [true; 6],
            se3_to_pose_params(pose),
            &mut pose_entity_registry,
            &mut block_values,
            &mut next_param_index,
            &mut constant_blocks,
            &mut pose_free_axes,
        );
    }

    for spec in &camera_param_specs {
        let Some(camera) = camera_by_index(reconstruction, spec.camera) else {
            continue;
        };
        if spec.param >= camera.num_params {
            continue;
        }
        let key = (spec.camera, spec.param);
        if camera_param_registry.contains_key(&key) {
            continue;
        }
        let idx = next_param_index;
        next_param_index += 1;
        block_values.insert(idx, vec![camera.params_slice()[spec.param]]);
        camera_param_registry.insert(key, idx);
    }

    let mut problem = NllsProblem::new();
    let mut bindings = 0usize;
    let mut internal_to_storage = HashMap::<usize, usize>::new();
    let mut next_storage_index = 0usize;

    for obs in &observations {
        let Some(point) = reconstruction.points.get(obs.point) else {
            continue;
        };
        let camera_base = reconstruction.camera_for_image(obs.image);
        let Some(pose_eval) = build_pose_eval(
            reconstruction,
            obs.image,
            &pose_blocks,
            &sensor_lookup,
            &pose_entity_registry,
        ) else {
            continue;
        };
        let mut param_indices = Vec::new();
        let mut param_roles = Vec::new();
        append_pose_parameters(&pose_eval, &mut param_indices, &mut param_roles);
        append_camera_parameters(
            reconstruction,
            obs.image,
            &camera_param_specs,
            &camera_param_registry,
            &mut param_indices,
            &mut param_roles,
        );
        let point_idx = register_point(
            obs.point,
            point.xyz,
            constant_point_filter.contains(&obs.point),
            &mut point_registry,
            &mut block_values,
            &mut next_param_index,
            &mut constant_blocks,
        );
        param_indices.push(point_idx);
        param_roles.push(ParamRole::Point);

        let (param_indices, param_roles) = dedup_residual_parameters(&param_indices, &param_roles);

        let binding = ResidualBinding {
            xy: obs.xy,
            param_roles,
            pose_eval,
            camera_base,
        };
        let cost = build_cost_function(binding);

        let mut builder = problem.residual_block_builder().set_cost(cost, 2);
        for &idx in &param_indices {
            builder = builder.add_parameter(param_ref(
                idx,
                block_values.get(&idx).expect("parameter block must exist"),
                &mut internal_to_storage,
                &mut next_storage_index,
            ));
        }
        builder = builder.set_loss(ceres_loss(options.loss_function));
        problem = builder.build_into_problem().ok()?.0;
        bindings += 1;
    }

    for prior in &options.pose_priors {
        let Some(pose_eval) = build_pose_eval(
            reconstruction,
            prior.image,
            &pose_blocks,
            &sensor_lookup,
            &pose_entity_registry,
        ) else {
            continue;
        };
        let mut param_indices = Vec::new();
        let mut param_roles = Vec::new();
        append_pose_parameters(&pose_eval, &mut param_indices, &mut param_roles);
        let (param_indices, param_roles) = dedup_residual_parameters(&param_indices, &param_roles);
        if param_indices.is_empty()
            || param_indices
                .iter()
                .all(|idx| constant_blocks.contains(idx))
        {
            continue;
        }
        let information = position_prior_information_matrix(
            &prior.position_covariance,
            options.prior_position_fallback_stddev,
        );
        let sqrt_information = information
            .cholesky()
            .map(|factor| factor.l().clone())
            .unwrap_or_else(Mat3::identity);
        let cost = build_pose_prior_cost(PosePriorBinding {
            prior_position: prior.position,
            sqrt_information,
            param_roles,
            pose_eval,
        });
        let mut builder = problem.residual_block_builder().set_cost(cost, 3);
        for &idx in &param_indices {
            builder = builder.add_parameter(param_ref(
                idx,
                block_values
                    .get(&idx)
                    .expect("pose prior parameter block must exist"),
                &mut internal_to_storage,
                &mut next_storage_index,
            ));
        }
        problem = builder.build_into_problem().ok()?.0;
    }

    if point_registry.is_empty() || bindings == 0 {
        return None;
    }

    let effective_parameters = count_variable_blocks(&constant_blocks, &block_values);
    let residuals = count_variable_residuals(
        reconstruction,
        &observations,
        &pose_blocks,
        &sensor_pose_specs,
        &camera_param_specs,
        &constant_point_filter,
    );
    if effective_parameters == 0 || residuals == 0 {
        return None;
    }

    for (&internal_idx, &free_axes) in &pose_free_axes {
        let Some(&storage_idx) = internal_to_storage.get(&internal_idx) else {
            continue;
        };
        if constant_blocks.contains(&internal_idx) {
            continue;
        }
        let constant_translation = pose_manifold_constant_translation_indices(free_axes)?;
        problem
            .set_pose_manifold(storage_idx, &constant_translation)
            .ok()?;
    }

    for internal_idx in constant_blocks {
        let Some(&storage_idx) = internal_to_storage.get(&internal_idx) else {
            continue;
        };
        problem.set_parameter_block_constant(storage_idx).ok()?;
    }

    let (solver_options, solver_policy, sparse_backend) =
        ceres_solver_options(&options, pose_entity_registry.len(), bindings * 2)?;
    let setup_ms = setup_started.elapsed().as_secs_f64() * 1000.0;
    if let Some(control) = control {
        control.checkpoint().ok()?;
    }
    let solve_started = Instant::now();
    let solution = problem.solve(&solver_options).ok()?;
    // Ceres optimizes separate parameter storage. A stop requested during Solve
    // can therefore discard it before mutating the caller's reconstruction.
    if let Some(control) = control {
        control.checkpoint().ok()?;
    }
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1000.0;
    let postprocess_started = Instant::now();

    let summary = &solution.summary;
    let (
        mut termination_type,
        termination_reason,
        gradient_max_norm,
        step_norm,
        step_quality,
        damping,
    ) = map_ceres_summary(summary);
    let parameters = {
        #[cfg(test)]
        {
            let mut parameters = solution.parameters;
            if let Some(hooks) = super::commit_test_hooks::current() {
                if let Some(corrupt) = hooks.corrupt_first_camera_param {
                    if let Some(first_spec) = camera_param_specs.first() {
                        let key = (first_spec.camera, first_spec.param);
                        if let Some(&idx) = camera_param_registry.get(&key) {
                            if let Some(storage_idx) = internal_to_storage.get(&idx) {
                                if let Some(block) = parameters.get_mut(*storage_idx) {
                                    if let Some(slot) = block.get_mut(0) {
                                        *slot = corrupt;
                                    }
                                }
                            }
                        }
                    }
                }
                if let Some(tx) = hooks.corrupt_first_pose_translation_x {
                    let handle = pose_entity_registry
                        .iter()
                        .find(|(key, _)| matches!(key, PoseEntityKey::Frame(_)))
                        .or_else(|| pose_entity_registry.iter().next())
                        .map(|(_, handle)| *handle);
                    if let Some(handle) = handle {
                        if let Some(&storage_idx) = internal_to_storage.get(&handle) {
                            if let Some(block) = parameters.get_mut(storage_idx) {
                                if block.len() >= 5 {
                                    block[4] = tx;
                                }
                            }
                        }
                    }
                }
                if let Some(tx) = hooks.corrupt_all_pose_translations_x {
                    for &handle in pose_entity_registry.values() {
                        if let Some(&storage_idx) = internal_to_storage.get(&handle) {
                            if let Some(block) = parameters.get_mut(storage_idx) {
                                if block.len() >= 5 {
                                    block[4] = tx;
                                }
                            }
                        }
                    }
                }
            }
            parameters
        }
        #[cfg(not(test))]
        {
            solution.parameters
        }
    };
    let ceres_usable = {
        #[cfg(test)]
        {
            let mut ceres_usable = summary.is_solution_usable();
            if let Some(hooks) = super::commit_test_hooks::current() {
                if let Some(force_usable) = hooks.force_ceres_usable {
                    ceres_usable = force_usable;
                }
            }
            ceres_usable
        }
        #[cfg(not(test))]
        {
            summary.is_solution_usable()
        }
    };
    #[cfg(test)]
    let cancel_before_commit = super::commit_test_hooks::current()
        .map(|hooks| hooks.cancel_before_commit)
        .unwrap_or(false);
    #[cfg(test)]
    if let Some(hooks) = super::commit_test_hooks::current() {
        if let Some(forced_termination) = hooks.force_termination {
            termination_type = forced_termination;
        }
    }

    // Build the full candidate (poses, cameras, points, derived point errors)
    // before touching the caller's Reconstruction. Any stage failure keeps the
    // live model unchanged.
    let candidate = build_validated_candidate(
        frames,
        reconstruction,
        &parameters,
        &internal_to_storage,
        &pose_entity_registry,
        &frame_images,
        &camera_param_registry,
        &camera_param_specs,
        &point_registry,
        &constant_point_filter,
        &pose_blocks,
    );
    let candidate_valid = candidate.is_ok();

    let mut committed = should_commit_ba_solution(ceres_usable, termination_type, candidate_valid);
    if !candidate_valid && termination_type.is_solution_usable() {
        // Solver claimed usability, but the candidate would install illegal state.
        termination_type = BundleAdjustmentTerminationType::Failure;
        committed = false;
    }

    let covariance = if committed {
        // Cancel after solve / candidate validation must still discard write-back.
        #[cfg(test)]
        if cancel_before_commit {
            if let Some(control) = control {
                control.request_cancel();
            } else {
                // No control binding: treat as cancelled without mutating.
                return None;
            }
        }
        if let Some(control) = control {
            control.checkpoint().ok()?;
        }
        let candidate = candidate.expect("committed requires a validated candidate");
        install_ba_candidate(reconstruction, candidate);
        options
            .compute_covariance
            .then(|| {
                super::ceres_support::compute_bundle_adjustment_covariance(
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
            .flatten()
    } else {
        let _ = candidate;
        None
    };

    let successful_steps = summary.num_successful_steps().max(0) as usize;
    let unsuccessful_steps = summary.num_unsuccessful_steps().max(0) as usize;
    let residuals_reduced = summary.num_residuals_reduced().max(0) as usize;
    let effective_parameters_reduced = summary.num_effective_parameters_reduced().max(0) as usize;
    let postprocess_ms = postprocess_started.elapsed().as_secs_f64() * 1000.0;
    let elapsed_ms = ba_started.elapsed().as_secs_f64() * 1000.0;
    // Emit after the existing timer boundaries; diagnostic formatting and I/O
    // must not be attributed to native solve or postprocessing.
    if std::env::var("RUSTSFM_PROFILE_CERES").is_ok_and(|value| value == "1") {
        eprintln!(
            "RUSTSFM_CERES_PROFILE {}",
            serde_json::json!({
                "observations": observations.len(),
                "residuals": residuals_reduced,
                "effective_parameters": effective_parameters_reduced,
                "initial_cost": summary.initial_cost(),
                "final_cost": summary.final_cost(),
                "solve_wrapper_ms": solve_ms,
                "ba_elapsed_ms": elapsed_ms,
                "message": summary.message(),
                "full_report": summary.full_report(),
                "committed": committed,
            })
        );
    }
    Some(BundleAdjustmentReport {
        solver_num_threads: ceres_num_threads(&options, bindings * 2) as usize,
        scheduling: None,
        iterations: successful_steps,
        attempted_iterations: successful_steps + unsuccessful_steps,
        successful_steps,
        unsuccessful_steps,
        linear_solver_iterations: summary.num_inner_iteration_steps().max(0) as usize,
        linearization_failures: 0,
        linear_solve_failures: 0,
        invalid_steps: 0,
        rejected_steps: unsuccessful_steps,
        initial_cost: summary.initial_cost(),
        final_cost: summary.final_cost(),
        observations: observations.len(),
        residuals: residuals_reduced,
        effective_parameters: effective_parameters_reduced,
        gradient_max_norm,
        step_norm,
        step_quality,
        damping,
        linear_solver: map_bundle_adjustment_linear_solver(solver_policy),
        preconditioner: map_bundle_adjustment_preconditioner(solver_policy),
        sparse_backend,
        setup_ms,
        solve_ms,
        postprocess_ms,
        elapsed_ms,
        covariance,
        termination_type,
        termination_reason,
        camera_reset_audit: None,
    })
}

fn register_pose_entity(
    key: PoseEntityKey,
    free_axes: [bool; 6],
    values: [f64; 7],
    registry: &mut HashMap<PoseEntityKey, usize>,
    block_values: &mut HashMap<usize, Vec<f64>>,
    next_param_index: &mut usize,
    constant_blocks: &mut HashSet<usize>,
    pose_free_axes: &mut HashMap<usize, [bool; 6]>,
) -> usize {
    if let Some(handle) = registry.get(&key).copied() {
        return handle;
    }
    let idx = *next_param_index;
    *next_param_index += 1;
    block_values.insert(idx, values.to_vec());
    if free_axes.iter().all(|free| !*free) {
        constant_blocks.insert(idx);
    }
    pose_free_axes.insert(idx, free_axes);
    registry.insert(key, idx);
    idx
}

fn pose_manifold_constant_translation_indices(free_axes: [bool; 6]) -> Option<Vec<usize>> {
    if free_axes[0..3].iter().any(|free| !*free) {
        return None;
    }
    Some(
        free_axes[3..6]
            .iter()
            .enumerate()
            .filter_map(|(idx, &free)| (!free).then_some(idx))
            .collect(),
    )
}

fn register_point(
    point_id: usize,
    xyz: [f32; 3],
    is_constant: bool,
    registry: &mut HashMap<usize, usize>,
    block_values: &mut HashMap<usize, Vec<f64>>,
    next_param_index: &mut usize,
    constant_blocks: &mut HashSet<usize>,
) -> usize {
    if let Some(&idx) = registry.get(&point_id) {
        return idx;
    }
    let idx = *next_param_index;
    *next_param_index += 1;
    block_values.insert(idx, vec![xyz[0] as f64, xyz[1] as f64, xyz[2] as f64]);
    registry.insert(point_id, idx);
    if is_constant {
        constant_blocks.insert(idx);
    }
    idx
}

fn param_ref(
    idx: usize,
    values: &[f64],
    internal_to_storage: &mut HashMap<usize, usize>,
    next_storage_index: &mut usize,
) -> ParameterBlockOrIndex {
    if let Some(&storage_idx) = internal_to_storage.get(&idx) {
        return storage_idx.into();
    }
    let storage_idx = *next_storage_index;
    *next_storage_index += 1;
    internal_to_storage.insert(idx, storage_idx);
    values.to_vec().into()
}

fn build_pose_eval(
    reconstruction: &Reconstruction,
    image: usize,
    pose_blocks: &super::ceres_support::PoseBlockSet,
    sensor_lookup: &HashMap<SensorPoseKey, &super::ceres_support::SensorPoseSpec>,
    pose_entity_registry: &HashMap<PoseEntityKey, usize>,
) -> Option<PoseEval> {
    let Some(block_idx) = pose_blocks.image_to_block.get(image).copied().flatten() else {
        let pose = reconstruction.poses.get(image).copied().flatten()?;
        return Some(PoseEval::Fixed(pose));
    };
    let block = pose_blocks.blocks.get(block_idx)?;
    match block.kind {
        PoseBlockKind::Image(image) => Some(PoseEval::Image {
            handle: *pose_entity_registry.get(&PoseEntityKey::Image(image))?,
        }),
        PoseBlockKind::Frame(frame_idx) => {
            let frame_handle = *pose_entity_registry.get(&PoseEntityKey::Frame(frame_idx))?;
            if let Some(key) = frame_sensor_key_for_image(reconstruction, image) {
                if sensor_lookup.contains_key(&key) {
                    Some(PoseEval::Frame {
                        frame_handle,
                        sensor: FrameSensorEval::Variable {
                            handle: *pose_entity_registry.get(&PoseEntityKey::Sensor(key))?,
                        },
                    })
                } else {
                    let fixed = frame_sensor_from_rig(reconstruction, frame_idx, image)?;
                    Some(PoseEval::Frame {
                        frame_handle,
                        sensor: FrameSensorEval::Fixed(fixed),
                    })
                }
            } else {
                None
            }
        }
    }
}

fn dedup_residual_parameters(
    param_indices: &[usize],
    param_roles: &[ParamRole],
) -> (Vec<usize>, Vec<ParamRole>) {
    let mut indices = Vec::new();
    let mut roles = Vec::new();
    for (&idx, &role) in param_indices.iter().zip(param_roles.iter()) {
        if indices.contains(&idx) {
            continue;
        }
        indices.push(idx);
        roles.push(role);
    }
    (indices, roles)
}

fn append_pose_parameters(
    pose_eval: &PoseEval,
    param_indices: &mut Vec<usize>,
    param_roles: &mut Vec<ParamRole>,
) {
    match pose_eval {
        PoseEval::Fixed(_) => {}
        PoseEval::Image { handle } => {
            param_indices.push(*handle);
            param_roles.push(ParamRole::ImagePose);
        }
        PoseEval::Frame {
            frame_handle,
            sensor,
        } => {
            param_indices.push(*frame_handle);
            param_roles.push(ParamRole::FramePose);
            if let FrameSensorEval::Variable { handle } = sensor {
                param_indices.push(*handle);
                param_roles.push(ParamRole::SensorPose);
            }
        }
    }
}

fn append_camera_parameters(
    reconstruction: &Reconstruction,
    image: usize,
    camera_param_specs: &[CameraParamSpec],
    camera_param_registry: &HashMap<(usize, usize), usize>,
    param_indices: &mut Vec<usize>,
    param_roles: &mut Vec<ParamRole>,
) {
    let Some(camera_idx) = super::ceres_support::camera_index_for_image(reconstruction, image)
    else {
        return;
    };
    for spec in camera_param_specs {
        if spec.camera != camera_idx {
            continue;
        }
        let key = (spec.camera, spec.param);
        let Some(&idx) = camera_param_registry.get(&key) else {
            continue;
        };
        if !param_indices.contains(&idx) {
            param_indices.push(idx);
            param_roles.push(ParamRole::CameraParam(spec.param));
        }
    }
}

fn build_cost_function(binding: ResidualBinding) -> CostFunctionType<'static> {
    Box::new(
        move |parameters: &[&[f64]], residuals: &mut [f64], jacobians| {
            let Some(residual) = eval_residual(parameters, &binding) else {
                residuals[0] = 0.0;
                residuals[1] = 0.0;
                return jacobians.is_none();
            };
            residuals.copy_from_slice(&residual);
            if let Some(jacobians) = jacobians {
                return fill_jacobians(parameters, &binding, jacobians);
            }
            true
        },
    )
}

fn build_pose_prior_cost(binding: PosePriorBinding) -> CostFunctionType<'static> {
    Box::new(
        move |parameters: &[&[f64]], residuals: &mut [f64], jacobians| {
            let Some(center) = pose_prior_camera_center(parameters, &binding) else {
                for residual in residuals.iter_mut() {
                    *residual = 0.0;
                }
                return jacobians.is_none();
            };
            let diff = center - nalgebra::DVector::from_column_slice(&binding.prior_position);
            let weighted = binding.sqrt_information * diff;
            for (residual, value) in residuals.iter_mut().zip(weighted.iter()) {
                *residual = *value;
            }
            if let Some(jacobians) = jacobians {
                for (p_idx, role) in binding.param_roles.iter().enumerate() {
                    let Some(jacobian) = jacobians.get_mut(p_idx).and_then(|block| block.as_mut())
                    else {
                        continue;
                    };
                    let Some(ambient) =
                        camera_center_ambient_jacobian_for_role(parameters, &binding, p_idx, *role)
                    else {
                        return false;
                    };
                    let product = binding.sqrt_information * ambient;
                    for row in 0..3 {
                        for col in 0..7 {
                            jacobian[row][col] = product[(row, col)];
                        }
                    }
                }
            }
            true
        },
    )
}

fn pose_prior_camera_center(
    parameters: &[&[f64]],
    binding: &PosePriorBinding,
) -> Option<nalgebra::SVector<f64, 3>> {
    let pose = match &binding.pose_eval {
        PoseEval::Fixed(pose) => *pose,
        PoseEval::Image { .. } => {
            let pose_params =
                pose_params_for_roles(parameters, &binding.param_roles, ParamRole::ImagePose)?;
            pose_params_to_se3(pose_params)
        }
        PoseEval::Frame { sensor, .. } => {
            let rig_pose =
                pose_params_for_roles(parameters, &binding.param_roles, ParamRole::FramePose)?;
            let rig = pose_params_to_se3(rig_pose);
            let sensor_pose = match sensor {
                FrameSensorEval::Fixed(pose) => *pose,
                FrameSensorEval::Variable { .. } => {
                    let sensor_pose = pose_params_for_roles(
                        parameters,
                        &binding.param_roles,
                        ParamRole::SensorPose,
                    )?;
                    pose_params_to_se3(sensor_pose)
                }
            };
            sensor_pose.compose(&rig)
        }
    };
    Some(camera_center_world(pose))
}

fn camera_center_ambient_jacobian_for_role(
    parameters: &[&[f64]],
    binding: &PosePriorBinding,
    p_idx: usize,
    role: ParamRole,
) -> Option<Mat3x7> {
    let mut base = [0.0; 7];
    copy_pose_params(parameters.get(p_idx)?, &mut base)?;
    let mut jacobian = Mat3x7::zeros();
    for col in 0..7 {
        let mut plus = base;
        let mut minus = base;
        plus[col] += POSE_PRIOR_JACOBIAN_EPS;
        minus[col] -= POSE_PRIOR_JACOBIAN_EPS;
        let plus_center =
            pose_prior_camera_center_with_replaced_block(parameters, binding, role, &plus)?;
        let minus_center =
            pose_prior_camera_center_with_replaced_block(parameters, binding, role, &minus)?;
        jacobian.set_column(
            col,
            &((plus_center - minus_center) / (2.0 * POSE_PRIOR_JACOBIAN_EPS)),
        );
    }
    Some(jacobian)
}

fn pose_prior_camera_center_with_replaced_block(
    parameters: &[&[f64]],
    binding: &PosePriorBinding,
    role: ParamRole,
    replacement: &[f64; 7],
) -> Option<nalgebra::SVector<f64, 3>> {
    let pose = match &binding.pose_eval {
        PoseEval::Fixed(pose) => *pose,
        PoseEval::Image { .. } => {
            if role != ParamRole::ImagePose {
                return None;
            }
            pose_params_to_se3(replacement)
        }
        PoseEval::Frame { sensor, .. } => {
            let rig = if role == ParamRole::FramePose {
                pose_params_to_se3(replacement)
            } else {
                pose_params_to_se3(pose_params_for_roles(
                    parameters,
                    &binding.param_roles,
                    ParamRole::FramePose,
                )?)
            };
            let sensor_pose = match sensor {
                FrameSensorEval::Fixed(pose) => *pose,
                FrameSensorEval::Variable { .. } => {
                    if role == ParamRole::SensorPose {
                        pose_params_to_se3(replacement)
                    } else {
                        pose_params_to_se3(pose_params_for_roles(
                            parameters,
                            &binding.param_roles,
                            ParamRole::SensorPose,
                        )?)
                    }
                }
            };
            sensor_pose.compose(&rig)
        }
    };
    Some(camera_center_world(pose))
}

fn eval_residual(parameters: &[&[f64]], binding: &ResidualBinding) -> Option<[f64; 2]> {
    let state = assemble_state(parameters, binding)?;
    let predicted = match &binding.pose_eval {
        PoseEval::Image { .. } => {
            let pose_params = pose_params_for_role(parameters, binding, ParamRole::ImagePose)?;
            project_image_pose_point(state.camera, pose_params, state.point)?
        }
        PoseEval::Frame { sensor, .. } => {
            let rig_pose = pose_params_for_role(parameters, binding, ParamRole::FramePose)?;
            match sensor {
                FrameSensorEval::Fixed(pose) => {
                    let sensor_pose = se3_to_pose_params(*pose);
                    project_frame_pose_point(state.camera, &sensor_pose, rig_pose, state.point)?
                }
                FrameSensorEval::Variable { .. } => {
                    let sensor_pose =
                        pose_params_for_role(parameters, binding, ParamRole::SensorPose)?;
                    project_frame_pose_point(state.camera, sensor_pose, rig_pose, state.point)?
                }
            }
        }
        _ => project_point(state.camera, state.pose, state.point)?,
    };
    Some([predicted[0] - binding.xy[0], predicted[1] - binding.xy[1]])
}

struct AssembledState {
    pose: SE3,
    rig_from_world: Option<SE3>,
    sensor_from_rig: Option<SE3>,
    point: [f32; 3],
    camera: CameraModel,
}

fn assemble_state(parameters: &[&[f64]], binding: &ResidualBinding) -> Option<AssembledState> {
    let mut image_pose = [0.0; 7];
    let mut frame_pose = [0.0; 7];
    let mut sensor_pose = [0.0; 7];
    let mut point = [0.0f64; 3];
    let mut has_point = false;
    let mut camera = binding.camera_base;

    for (p_idx, role) in binding.param_roles.iter().enumerate() {
        let slice = parameters.get(p_idx)?;
        match role {
            ParamRole::ImagePose => copy_pose_params(slice, &mut image_pose)?,
            ParamRole::FramePose => copy_pose_params(slice, &mut frame_pose)?,
            ParamRole::SensorPose => copy_pose_params(slice, &mut sensor_pose)?,
            ParamRole::Point => {
                point[0] = slice.first().copied().unwrap_or(0.0);
                point[1] = slice.get(1).copied().unwrap_or(0.0);
                point[2] = slice.get(2).copied().unwrap_or(0.0);
                has_point = true;
            }
            ParamRole::CameraParam(param) => {
                if *param < camera.num_params {
                    camera.set_param(*param, slice[0]).ok()?;
                }
            }
        }
    }

    if !has_point {
        return None;
    }

    let pose = match &binding.pose_eval {
        PoseEval::Fixed(pose) => {
            return Some(AssembledState {
                pose: *pose,
                rig_from_world: None,
                sensor_from_rig: None,
                point: [point[0] as f32, point[1] as f32, point[2] as f32],
                camera,
            });
        }
        PoseEval::Image { .. } => pose_params_to_se3(&image_pose),
        PoseEval::Frame { sensor, .. } => {
            let rig = pose_params_to_se3(&frame_pose);
            let sensor_pose = match sensor {
                FrameSensorEval::Fixed(pose) => *pose,
                FrameSensorEval::Variable { .. } => pose_params_to_se3(&sensor_pose),
            };
            sensor_pose.compose(&rig)
        }
    };
    let point = [point[0] as f32, point[1] as f32, point[2] as f32];
    let (rig_from_world, sensor_from_rig) = match &binding.pose_eval {
        PoseEval::Frame { sensor, .. } => {
            let rig = pose_params_to_se3(&frame_pose);
            let sensor_pose = match sensor {
                FrameSensorEval::Fixed(pose) => *pose,
                FrameSensorEval::Variable { .. } => pose_params_to_se3(&sensor_pose),
            };
            (Some(rig), Some(sensor_pose))
        }
        _ => (None, None),
    };
    Some(AssembledState {
        pose,
        rig_from_world,
        sensor_from_rig,
        point,
        camera,
    })
}

fn fill_jacobians(
    parameters: &[&[f64]],
    binding: &ResidualBinding,
    jacobians: &mut [Option<&mut [&mut [f64]]>],
) -> bool {
    if fill_analytic_jacobians(parameters, binding, jacobians).is_some() {
        true
    } else {
        fill_numeric_jacobians(parameters, binding, jacobians)
    }
}

fn fill_analytic_jacobians(
    parameters: &[&[f64]],
    binding: &ResidualBinding,
    jacobians: &mut [Option<&mut [&mut [f64]]>],
) -> Option<()> {
    let state = assemble_state(parameters, binding)?;
    let (_, j_point) = projection_jacobians(state.camera, state.pose, state.point)?;

    match &binding.pose_eval {
        PoseEval::Frame { sensor, .. } => {
            let rig = state.rig_from_world?;
            let sensor_pose = state.sensor_from_rig?;
            let j_frame =
                analytic_frame_pose_jacobian(state.camera, sensor_pose, rig, state.point)?;
            let j_sensor = match sensor {
                FrameSensorEval::Variable { .. } => Some(analytic_sensor_pose_jacobian(
                    state.camera,
                    sensor_pose,
                    rig,
                    state.point,
                )?),
                FrameSensorEval::Fixed(_) => None,
            };
            fill_pose_jacobians(
                binding,
                parameters,
                jacobians,
                &state,
                |role| match role {
                    ParamRole::FramePose => Some((j_frame[(0, 0)], j_frame[(1, 0)])),
                    ParamRole::SensorPose => {
                        let j = j_sensor.as_ref()?;
                        Some((j[(0, 0)], j[(1, 0)]))
                    }
                    _ => None,
                },
                &j_point,
            )?;
            return Some(());
        }
        _ => {}
    }

    fill_pose_jacobians(
        binding,
        parameters,
        jacobians,
        &state,
        |_role| None,
        &j_point,
    )
}

fn fill_pose_jacobians(
    binding: &ResidualBinding,
    parameters: &[&[f64]],
    jacobians: &mut [Option<&mut [&mut [f64]]>],
    state: &AssembledState,
    mut pose_col: impl FnMut(ParamRole) -> Option<(f64, f64)>,
    j_point: &nalgebra::SMatrix<f64, 2, 3>,
) -> Option<()> {
    for (p_idx, role) in binding.param_roles.iter().enumerate() {
        let Some(jac) = jacobians.get_mut(p_idx).and_then(|j| j.as_mut()) else {
            continue;
        };
        match role {
            ParamRole::ImagePose | ParamRole::FramePose | ParamRole::SensorPose => {
                if parameters
                    .get(p_idx)
                    .is_some_and(|params| params.len() == 7)
                {
                    if *role == ParamRole::ImagePose {
                        if let Some(j_pose) = analytic_image_pose_jacobian_block(
                            parameters,
                            binding,
                            p_idx,
                            state.camera,
                            state.point,
                        ) {
                            for k in 0..7 {
                                jac[0][k] = j_pose[(0, k)];
                                jac[1][k] = j_pose[(1, k)];
                            }
                        } else {
                            fill_numeric_jacobian_block(parameters, binding, p_idx, jac)?;
                        }
                    } else if *role == ParamRole::FramePose {
                        if let Some(j_pose) = analytic_frame_pose_jacobian_block(
                            parameters,
                            binding,
                            p_idx,
                            state.camera,
                            state.point,
                        ) {
                            for k in 0..7 {
                                jac[0][k] = j_pose[(0, k)];
                                jac[1][k] = j_pose[(1, k)];
                            }
                        } else {
                            fill_numeric_jacobian_block(parameters, binding, p_idx, jac)?;
                        }
                    } else if *role == ParamRole::SensorPose {
                        if let Some(j_pose) = analytic_sensor_pose_jacobian_block(
                            parameters,
                            binding,
                            p_idx,
                            state.camera,
                            state.point,
                        ) {
                            for k in 0..7 {
                                jac[0][k] = j_pose[(0, k)];
                                jac[1][k] = j_pose[(1, k)];
                            }
                        } else {
                            fill_numeric_jacobian_block(parameters, binding, p_idx, jac)?;
                        }
                    } else {
                        fill_numeric_jacobian_block(parameters, binding, p_idx, jac)?;
                    }
                } else {
                    let (d0, d1) = pose_col(*role)?;
                    jac[0][0] = d0;
                    jac[1][0] = d1;
                }
            }
            ParamRole::Point => {
                for k in 0..3 {
                    jac[0][k] = j_point[(0, k)];
                    jac[1][k] = j_point[(1, k)];
                }
            }
            ParamRole::CameraParam(param) => {
                let j_param = camera_param_jacobian(state.camera, *param, state.pose, state.point)?;
                jac[0][0] = j_param[0];
                jac[1][0] = j_param[1];
            }
        }
    }
    Some(())
}

fn analytic_image_pose_jacobian_block(
    parameters: &[&[f64]],
    binding: &ResidualBinding,
    p_idx: usize,
    camera: CameraModel,
    point: [f32; 3],
) -> Option<Mat2x7> {
    if binding.param_roles.get(p_idx) != Some(&ParamRole::ImagePose) {
        return None;
    }
    let pose_params = parameters.get(p_idx)?;
    if pose_params.len() != 7 {
        return None;
    }
    analytic_image_pose_jacobian_ambient(camera, point, pose_params)
}

fn analytic_frame_pose_jacobian_block(
    parameters: &[&[f64]],
    binding: &ResidualBinding,
    p_idx: usize,
    camera: CameraModel,
    point: [f32; 3],
) -> Option<Mat2x7> {
    if binding.param_roles.get(p_idx) != Some(&ParamRole::FramePose) {
        return None;
    }
    let rig_pose = parameters.get(p_idx)?;
    if rig_pose.len() != 7 {
        return None;
    }
    let sensor_pose = sensor_pose_params_for_binding(parameters, binding)?;
    analytic_frame_pose_jacobian_ambient(camera, &sensor_pose, rig_pose, point)
}

fn analytic_sensor_pose_jacobian_block(
    parameters: &[&[f64]],
    binding: &ResidualBinding,
    p_idx: usize,
    camera: CameraModel,
    point: [f32; 3],
) -> Option<Mat2x7> {
    if binding.param_roles.get(p_idx) != Some(&ParamRole::SensorPose) {
        return None;
    }
    let sensor_pose = parameters.get(p_idx)?;
    if sensor_pose.len() != 7 {
        return None;
    }
    let rig_pose = pose_params_array_for_role(parameters, binding, ParamRole::FramePose)?;
    analytic_sensor_pose_jacobian_ambient(camera, sensor_pose, &rig_pose, point)
}

fn analytic_image_pose_jacobian_ambient(
    camera: CameraModel,
    point: [f32; 3],
    pose_params: &[f64],
) -> Option<Mat2x7> {
    let cam_point = image_pose_cam_point(pose_params, point)?;
    let x = cam_point[0];
    let y = cam_point[1];
    let z = cam_point[2];
    let j_cam = analytic_img_from_cam_jacobian(camera, x, y, z)?;
    let j_rot = quaternion_rotate_point_jacobian(pose_params, point_f64(point))?;

    let mut j_pose = Mat2x7::zeros();
    for col in 0..4 {
        j_pose[(0, col)] = j_cam[(0, 0)] * j_rot[(0, col)]
            + j_cam[(0, 1)] * j_rot[(1, col)]
            + j_cam[(0, 2)] * j_rot[(2, col)];
        j_pose[(1, col)] = j_cam[(1, 0)] * j_rot[(0, col)]
            + j_cam[(1, 1)] * j_rot[(1, col)]
            + j_cam[(1, 2)] * j_rot[(2, col)];
    }
    for col in 0..3 {
        j_pose[(0, col + 4)] = j_cam[(0, col)];
        j_pose[(1, col + 4)] = j_cam[(1, col)];
    }
    Some(j_pose)
}

fn analytic_frame_pose_jacobian_ambient(
    camera: CameraModel,
    sensor_pose_params: &[f64],
    rig_pose_params: &[f64],
    point: [f32; 3],
) -> Option<Mat2x7> {
    let cam_point = frame_pose_cam_point(sensor_pose_params, rig_pose_params, point)?;
    let j_cam = analytic_img_from_cam_jacobian(camera, cam_point[0], cam_point[1], cam_point[2])?;
    let j_rig_rot = quaternion_rotate_point_jacobian(rig_pose_params, point_f64(point))?;
    let r_sensor = quaternion_rotation_matrix_colmap(sensor_pose_params)?;

    let mut dcam_drig = Mat3x7::zeros();
    for col in 0..4 {
        for row in 0..3 {
            dcam_drig[(row, col)] = r_sensor[(row, 0)] * j_rig_rot[(0, col)]
                + r_sensor[(row, 1)] * j_rig_rot[(1, col)]
                + r_sensor[(row, 2)] * j_rig_rot[(2, col)];
        }
    }
    for col in 0..3 {
        for row in 0..3 {
            dcam_drig[(row, col + 4)] = r_sensor[(row, col)];
        }
    }

    Some(mul_img_from_cam_jacobian(&j_cam, &dcam_drig))
}

fn analytic_sensor_pose_jacobian_ambient(
    camera: CameraModel,
    sensor_pose_params: &[f64],
    rig_pose_params: &[f64],
    point: [f32; 3],
) -> Option<Mat2x7> {
    let point_in_rig = rig_pose_point(rig_pose_params, point)?;
    let cam_point = frame_pose_cam_point(sensor_pose_params, rig_pose_params, point)?;
    let j_cam = analytic_img_from_cam_jacobian(camera, cam_point[0], cam_point[1], cam_point[2])?;
    let j_sensor_rot = quaternion_rotate_point_jacobian(sensor_pose_params, point_in_rig)?;

    let mut dcam_dsensor = Mat3x7::zeros();
    for col in 0..4 {
        for row in 0..3 {
            dcam_dsensor[(row, col)] = j_sensor_rot[(row, col)];
        }
    }
    for col in 0..3 {
        dcam_dsensor[(col, col + 4)] = 1.0;
    }

    Some(mul_img_from_cam_jacobian(&j_cam, &dcam_dsensor))
}

fn mul_img_from_cam_jacobian(j_cam: &SMatrix<f64, 2, 3>, dcam_dpose: &Mat3x7) -> Mat2x7 {
    let mut jacobian = Mat2x7::zeros();
    for col in 0..7 {
        jacobian[(0, col)] = j_cam[(0, 0)] * dcam_dpose[(0, col)]
            + j_cam[(0, 1)] * dcam_dpose[(1, col)]
            + j_cam[(0, 2)] * dcam_dpose[(2, col)];
        jacobian[(1, col)] = j_cam[(1, 0)] * dcam_dpose[(0, col)]
            + j_cam[(1, 1)] * dcam_dpose[(1, col)]
            + j_cam[(1, 2)] * dcam_dpose[(2, col)];
    }
    jacobian
}

fn pose_params_for_role<'a>(
    parameters: &'a [&'a [f64]],
    binding: &ResidualBinding,
    role: ParamRole,
) -> Option<&'a [f64]> {
    pose_params_for_roles(parameters, &binding.param_roles, role)
}

fn pose_params_for_roles<'a>(
    parameters: &'a [&'a [f64]],
    param_roles: &[ParamRole],
    role: ParamRole,
) -> Option<&'a [f64]> {
    let p_idx = param_roles
        .iter()
        .position(|candidate| *candidate == role)?;
    let pose_params = parameters.get(p_idx)?;
    (pose_params.len() == 7).then_some(*pose_params)
}

fn pose_params_array_for_role(
    parameters: &[&[f64]],
    binding: &ResidualBinding,
    role: ParamRole,
) -> Option<[f64; 7]> {
    let pose_params = pose_params_for_role(parameters, binding, role)?;
    let mut out = [0.0; 7];
    out.copy_from_slice(pose_params);
    Some(out)
}

fn sensor_pose_params_for_binding(
    parameters: &[&[f64]],
    binding: &ResidualBinding,
) -> Option<[f64; 7]> {
    match &binding.pose_eval {
        PoseEval::Frame { sensor, .. } => match sensor {
            FrameSensorEval::Fixed(pose) => Some(se3_to_pose_params(*pose)),
            FrameSensorEval::Variable { .. } => {
                pose_params_array_for_role(parameters, binding, ParamRole::SensorPose)
            }
        },
        _ => None,
    }
}

fn project_image_pose_point(
    camera: CameraModel,
    pose_params: &[f64],
    point: [f32; 3],
) -> Option<[f64; 2]> {
    let cam_point = image_pose_cam_point(pose_params, point)?;
    camera.img_from_cam_unchecked(cam_point[0], cam_point[1], cam_point[2])
}

fn project_frame_pose_point(
    camera: CameraModel,
    sensor_pose_params: &[f64],
    rig_pose_params: &[f64],
    point: [f32; 3],
) -> Option<[f64; 2]> {
    let cam_point = frame_pose_cam_point(sensor_pose_params, rig_pose_params, point)?;
    camera.img_from_cam_unchecked(cam_point[0], cam_point[1], cam_point[2])
}

fn image_pose_cam_point(pose_params: &[f64], point: [f32; 3]) -> Option<[f64; 3]> {
    if pose_params.len() != 7 {
        return None;
    }
    let rotated = quaternion_rotate_point_colmap(pose_params, point_f64(point))?;
    let cam_point = [
        rotated[0] + pose_params[4],
        rotated[1] + pose_params[5],
        rotated[2] + pose_params[6],
    ];
    cam_point
        .iter()
        .all(|value| value.is_finite())
        .then_some(cam_point)
}

fn frame_pose_cam_point(
    sensor_pose_params: &[f64],
    rig_pose_params: &[f64],
    point: [f32; 3],
) -> Option<[f64; 3]> {
    let point_in_rig = rig_pose_point(rig_pose_params, point)?;
    let rotated = quaternion_rotate_point_colmap(sensor_pose_params, point_in_rig)?;
    let cam_point = [
        rotated[0] + sensor_pose_params[4],
        rotated[1] + sensor_pose_params[5],
        rotated[2] + sensor_pose_params[6],
    ];
    cam_point
        .iter()
        .all(|value| value.is_finite())
        .then_some(cam_point)
}

fn rig_pose_point(rig_pose_params: &[f64], point: [f32; 3]) -> Option<[f64; 3]> {
    if rig_pose_params.len() != 7 {
        return None;
    }
    let rotated = quaternion_rotate_point_colmap(rig_pose_params, point_f64(point))?;
    let point_in_rig = [
        rotated[0] + rig_pose_params[4],
        rotated[1] + rig_pose_params[5],
        rotated[2] + rig_pose_params[6],
    ];
    point_in_rig
        .iter()
        .all(|value| value.is_finite())
        .then_some(point_in_rig)
}

fn point_f64(point: [f32; 3]) -> [f64; 3] {
    [point[0] as f64, point[1] as f64, point[2] as f64]
}

fn quaternion_rotation_matrix_colmap(q: &[f64]) -> Option<Mat3> {
    let ex = quaternion_rotate_point_colmap(q, [1.0, 0.0, 0.0])?;
    let ey = quaternion_rotate_point_colmap(q, [0.0, 1.0, 0.0])?;
    let ez = quaternion_rotate_point_colmap(q, [0.0, 0.0, 1.0])?;
    Some(Mat3::from_row_slice(&[
        ex[0], ey[0], ez[0], ex[1], ey[1], ez[1], ex[2], ey[2], ez[2],
    ]))
}

fn quaternion_rotate_point_colmap(q: &[f64], point: [f64; 3]) -> Option<[f64; 3]> {
    if q.len() < 4 {
        return None;
    }
    let qx = q[0];
    let qy = q[1];
    let qz = q[2];
    let qw = q[3];
    let px = point[0];
    let py = point[1];
    let pz = point[2];

    let v_x_p0 = qy * pz - qz * py;
    let v_x_p1 = qz * px - qx * pz;
    let v_x_p2 = qx * py - qy * px;
    let v_x_v_x_p0 = qy * v_x_p2 - qz * v_x_p1;
    let v_x_v_x_p1 = qz * v_x_p0 - qx * v_x_p2;
    let v_x_v_x_p2 = qx * v_x_p1 - qy * v_x_p0;

    let rotated = [
        px + 2.0 * (qw * v_x_p0 + v_x_v_x_p0),
        py + 2.0 * (qw * v_x_p1 + v_x_v_x_p1),
        pz + 2.0 * (qw * v_x_p2 + v_x_v_x_p2),
    ];
    rotated
        .iter()
        .all(|value| value.is_finite())
        .then_some(rotated)
}

fn quaternion_rotate_point_jacobian(q: &[f64], point: [f64; 3]) -> Option<SMatrix<f64, 3, 4>> {
    if q.len() < 4 {
        return None;
    }
    let qx = q[0];
    let qy = q[1];
    let qz = q[2];
    let qw = q[3];
    let px = point[0];
    let py = point[1];
    let pz = point[2];
    let qx_px = qx * px;
    let qx_py = qx * py;
    let qx_pz = qx * pz;
    let qy_px = qy * px;
    let qy_py = qy * py;
    let qy_pz = qy * pz;
    let qz_px = qz * px;
    let qz_py = qz * py;
    let qz_pz = qz * pz;
    let qw_px = qw * px;
    let qw_py = qw * py;
    let qw_pz = qw * pz;

    let jacobian = SMatrix::<f64, 3, 4>::from_row_slice(&[
        2.0 * (qy_py + qz_pz),
        2.0 * (-2.0 * qy_px + qx_py + qw_pz),
        2.0 * (-2.0 * qz_px - qw_py + qx_pz),
        2.0 * (-qz_py + qy_pz),
        2.0 * (qy_px - 2.0 * qx_py - qw_pz),
        2.0 * (qx_px + qz_pz),
        2.0 * (qw_px - 2.0 * qz_py + qy_pz),
        2.0 * (qz_px - qx_pz),
        2.0 * (qz_px + qw_py - 2.0 * qx_pz),
        2.0 * (-qw_px + qz_py - 2.0 * qy_pz),
        2.0 * (qx_px + qy_py),
        2.0 * (-qy_px + qx_py),
    ]);
    jacobian
        .iter()
        .all(|value| value.is_finite())
        .then_some(jacobian)
}

fn fill_numeric_jacobians(
    parameters: &[&[f64]],
    binding: &ResidualBinding,
    jacobians: &mut [Option<&mut [&mut [f64]]>],
) -> bool {
    for (p_idx, jac_opt) in jacobians.iter_mut().enumerate() {
        let Some(jac) = jac_opt else {
            continue;
        };
        if fill_numeric_jacobian_block(parameters, binding, p_idx, jac).is_none() {
            return false;
        }
    }
    true
}

fn fill_numeric_jacobian_block(
    parameters: &[&[f64]],
    binding: &ResidualBinding,
    p_idx: usize,
    jac: &mut [&mut [f64]],
) -> Option<()> {
    const EPS: f64 = 1.0e-8;
    let mut params_storage = parameters.iter().map(|p| p.to_vec()).collect::<Vec<_>>();
    let param_len = params_storage.get(p_idx)?.len();
    for k in 0..param_len {
        params_storage[p_idx][k] += EPS;
        let plus = eval_residual_from_storage(&params_storage, binding, p_idx);
        params_storage[p_idx][k] -= 2.0 * EPS;
        let minus = eval_residual_from_storage(&params_storage, binding, p_idx);
        params_storage[p_idx][k] += EPS;
        let (Some(plus), Some(minus)) = (plus, minus) else {
            return None;
        };
        for r in 0..2 {
            jac[r][k] = (plus[r] - minus[r]) / (2.0 * EPS);
        }
    }
    Some(())
}

fn eval_residual_from_storage(
    params_storage: &[Vec<f64>],
    binding: &ResidualBinding,
    _perturbed_block: usize,
) -> Option<[f64; 2]> {
    let params = params_storage
        .iter()
        .map(|p| p.as_slice())
        .collect::<Vec<_>>();
    eval_residual(&params, binding)
}

struct PreparedBaWriteBack {
    image_poses: Vec<(usize, SE3)>,
    frame_poses: Vec<(usize, Vec<usize>, SE3)>,
    sensor_poses: Vec<(SensorPoseKey, SE3)>,
    cameras: Vec<CameraModel>,
    legacy_camera: CameraModel,
    points: Vec<(usize, [f32; 3])>,
}

fn prepare_write_back(
    reconstruction: &Reconstruction,
    parameters: &[Vec<f64>],
    internal_to_storage: &HashMap<usize, usize>,
    pose_entity_registry: &HashMap<PoseEntityKey, usize>,
    camera_param_registry: &HashMap<(usize, usize), usize>,
    camera_param_specs: &[CameraParamSpec],
    point_registry: &HashMap<usize, usize>,
    constant_point_filter: &HashSet<usize>,
) -> Result<PreparedBaWriteBack, String> {
    let mut image_poses = Vec::new();
    let mut frame_poses = Vec::new();
    let mut sensor_poses = Vec::new();

    for (key, &handle) in pose_entity_registry {
        let values = parameter_values(parameters, handle, internal_to_storage)
            .ok_or_else(|| format!("missing pose parameter block for handle {handle}"))?;
        let pose = pose_params_to_se3_checked(values)?;
        match key {
            PoseEntityKey::Image(image) => image_poses.push((*image, pose)),
            PoseEntityKey::Frame(frame_idx) => {
                // Image membership is resolved at apply time from the live map.
                frame_poses.push((*frame_idx, Vec::new(), pose));
            }
            PoseEntityKey::Sensor(sensor_key) => sensor_poses.push((sensor_key.clone(), pose)),
        }
    }

    let mut cameras = reconstruction.cameras.clone();
    if cameras.is_empty() {
        cameras.push(reconstruction.camera);
    }
    for &spec in camera_param_specs {
        let key = (spec.camera, spec.param);
        let Some(&idx) = camera_param_registry.get(&key) else {
            return Err(format!(
                "missing camera parameter registry entry for camera {} param {}",
                spec.camera, spec.param
            ));
        };
        if spec.camera >= cameras.len() {
            return Err(format!(
                "camera index {} out of range for write-back",
                spec.camera
            ));
        }
        if spec.param >= cameras[spec.camera].num_params {
            return Err(format!(
                "camera {} param {} out of range for num_params={}",
                spec.camera, spec.param, cameras[spec.camera].num_params
            ));
        }
        let params = parameter_values(parameters, idx, internal_to_storage).ok_or_else(|| {
            format!(
                "missing camera parameter block for camera {} param {}",
                spec.camera, spec.param
            )
        })?;
        let value = params.first().copied().ok_or_else(|| {
            format!(
                "empty camera parameter block for camera {} param {}",
                spec.camera, spec.param
            )
        })?;
        cameras[spec.camera]
            .set_param(spec.param, value)
            .map_err(|err| {
                format!(
                    "illegal camera write-back for camera {} param {}: {err}",
                    spec.camera, spec.param
                )
            })?;
    }
    let legacy_camera = cameras.first().copied().unwrap_or(reconstruction.camera);

    let mut points = Vec::new();
    for (&point_id, &idx) in point_registry {
        if constant_point_filter.contains(&point_id) {
            continue;
        }
        if point_id >= reconstruction.points.len() {
            return Err(format!(
                "point index {point_id} out of range for write-back"
            ));
        }
        let params = parameter_values(parameters, idx, internal_to_storage)
            .ok_or_else(|| format!("missing point parameter block for point {point_id}"))?;
        let xyz = point_params_to_xyz_checked(params)?;
        points.push((point_id, xyz));
    }

    Ok(PreparedBaWriteBack {
        image_poses,
        frame_poses,
        sensor_poses,
        cameras,
        legacy_camera,
        points,
    })
}

fn apply_prepared_write_back(
    reconstruction: &mut Reconstruction,
    prepared: PreparedBaWriteBack,
    frame_images: &HashMap<usize, Vec<usize>>,
    pose_blocks: &super::ceres_support::PoseBlockSet,
) {
    for (image, pose) in prepared.image_poses {
        if let Some(slot) = reconstruction.poses.get_mut(image) {
            *slot = Some(pose);
        }
    }

    for (frame_idx, _, pose) in prepared.frame_poses {
        if let Some(images) = frame_images.get(&frame_idx) {
            set_frame_pose_block(reconstruction, frame_idx, images, pose);
        }
    }

    let mut changed_sensors = Vec::new();
    for (sensor_key, pose) in prepared.sensor_poses {
        if let Some(rig) = reconstruction
            .rigs
            .iter_mut()
            .find(|rig| rig.rig_id == sensor_key.rig_id)
        {
            if let Some(sensor) = rig
                .sensors
                .iter_mut()
                .find(|sensor| sensor.sensor_id == sensor_key.sensor_id)
            {
                sensor.sensor_from_rig = Some(Rigid3::from_se3(pose));
                changed_sensors.push(sensor_key);
            }
        }
    }
    if !changed_sensors.is_empty() {
        sync_pose_blocks_for_sensor_changes(reconstruction, pose_blocks, &changed_sensors);
    }

    reconstruction.cameras = prepared.cameras;
    reconstruction.camera = prepared.legacy_camera;

    for (point_id, xyz) in prepared.points {
        if let Some(point) = reconstruction.points.get_mut(point_id) {
            point.xyz = xyz;
        }
    }
}

/// Recompute point errors on a candidate reconstruction. Rejects the candidate
/// when any observation residual, f64→f32 narrowing, or mean accumulation is
/// non-finite. Distinguishes finite behind-camera geometry (skipped under the
/// existing policy) from non-finite camera coordinates or projection overflow.
fn refresh_point_errors_checked(
    frames: &[ImageFrame],
    reconstruction: &mut Reconstruction,
) -> Result<(), String> {
    let image_cameras = (0..reconstruction.poses.len())
        .map(|image| reconstruction.camera_for_image(image))
        .collect::<Vec<_>>();
    for (point_id, point) in reconstruction.points.iter_mut().enumerate() {
        let mut total = 0.0f32;
        let mut count = 0usize;
        for obs in &point.track {
            let Some(pose) = reconstruction.poses.get(obs.image).copied().flatten() else {
                continue;
            };
            if obs.image >= frames.len() || obs.feature >= frames[obs.image].keypoints.len() {
                continue;
            }
            let kp = &frames[obs.image].keypoints[obs.feature];
            let Some(predicted) = project_point_for_candidate_error(
                image_cameras[obs.image],
                pose,
                point.xyz,
                point_id,
                obs.image,
                obs.feature,
            )?
            else {
                // Finite geometric non-projectability (e.g. behind camera).
                continue;
            };
            let err = ((predicted[0] - kp.x() as f64).powi(2)
                + (predicted[1] - kp.y() as f64).powi(2))
            .sqrt();
            if !err.is_finite() {
                return Err(format!(
                    "point {point_id} observation ({},{}) has non-finite residual {err}",
                    obs.image, obs.feature
                ));
            }
            let err_f32 = err as f32;
            if !err_f32.is_finite() {
                return Err(format!(
                    "point {point_id} observation ({},{}) residual overflows f32 ({err})",
                    obs.image, obs.feature
                ));
            }
            total += err_f32;
            if !total.is_finite() {
                return Err(format!(
                    "point {point_id} residual accumulation overflowed f32"
                ));
            }
            count += 1;
        }
        if count > 0 {
            let mean = total / count as f32;
            if !mean.is_finite() {
                return Err(format!(
                    "point {point_id} mean residual is non-finite ({mean})"
                ));
            }
            point.error = mean;
        }
        if !point.error.is_finite() {
            return Err(format!(
                "point {point_id} error is non-finite ({})",
                point.error
            ));
        }
    }
    Ok(())
}

/// Project for candidate error refresh.
///
/// - `Ok(Some(xy))`: finite image projection
/// - `Ok(None)`: finite geometry that the existing policy does not project
///   (behind/at the camera, or in-model geometric domain rejection)
/// - `Err`: non-finite camera coordinates or projection overflow/NaN/Inf
fn project_point_for_candidate_error(
    camera: CameraModel,
    pose: SE3,
    point: [f32; 3],
    point_id: usize,
    image: usize,
    feature: usize,
) -> Result<Option<[f64; 2]>, String> {
    use crate::types::CameraProjectionOutcome;

    let p = pose.transform_point(&point);
    let u = p[0] as f64;
    let v = p[1] as f64;
    let w = p[2] as f64;
    if !u.is_finite() || !v.is_finite() || !w.is_finite() {
        return Err(format!(
            "point {point_id} observation ({image},{feature}) has non-finite camera coordinates ({u}, {v}, {w})"
        ));
    }
    // Cheirality policy matches CameraModel::img_from_cam: skip behind/at camera.
    if w < f64::EPSILON {
        return Ok(None);
    }
    let uu = u / w;
    let vv = v / w;
    if !uu.is_finite() || !vv.is_finite() {
        return Err(format!(
            "point {point_id} observation ({image},{feature}) normalized camera coords overflow ({uu}, {vv})"
        ));
    }
    match camera.classify_img_from_cam_unchecked(u, v, w) {
        CameraProjectionOutcome::FiniteProjection(xy) => {
            // Residual refresh narrows to f32; reject overflow here instead of
            // letting a later cast silently become Inf after a Some(...) path.
            let xy_f32 = [xy[0] as f32, xy[1] as f32];
            if xy_f32.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "point {point_id} observation ({image},{feature}) projection overflows f32 ({}, {})",
                    xy[0], xy[1]
                ));
            }
            Ok(Some(xy))
        }
        CameraProjectionOutcome::FiniteGeometricDomainSkip => Ok(None),
        CameraProjectionOutcome::NonFiniteProjection => Err(format!(
            "point {point_id} observation ({image},{feature}) projection is non-finite for camera model {}",
            camera.model_id
        )),
    }
}

fn validate_se3_pose(pose: SE3, label: &str) -> Result<(), String> {
    let translation = pose.translation();
    let quaternion = pose.quaternion();
    if translation.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label}: non-finite translation {translation:?}"));
    }
    if quaternion.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label}: non-finite quaternion {quaternion:?}"));
    }
    let norm = (quaternion[0] * quaternion[0]
        + quaternion[1] * quaternion[1]
        + quaternion[2] * quaternion[2]
        + quaternion[3] * quaternion[3])
        .sqrt();
    if !norm.is_finite() || norm <= f32::EPSILON {
        return Err(format!("{label}: invalid quaternion norm {norm}"));
    }
    let normalized = [
        quaternion[0] / norm,
        quaternion[1] / norm,
        quaternion[2] / norm,
        quaternion[3] / norm,
    ];
    if normalized.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "{label}: normalized quaternion is non-finite {normalized:?}"
        ));
    }
    Ok(())
}

fn validate_rigid3_pose(rigid: &Rigid3, label: &str) -> Result<(), String> {
    if rigid.qvec.iter().any(|value| !value.is_finite())
        || rigid.tvec.iter().any(|value| !value.is_finite())
    {
        return Err(format!(
            "{label}: non-finite rigid components q={:?} t={:?}",
            rigid.qvec, rigid.tvec
        ));
    }
    let [w, x, y, z] = rigid.qvec;
    let norm = (w * w + x * x + y * y + z * z).sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(format!("{label}: invalid quaternion norm {norm}"));
    }
    let normalized = [w / norm, x / norm, y / norm, z / norm];
    if normalized.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "{label}: normalized quaternion is non-finite {normalized:?}"
        ));
    }
    // Also require the f32 SE3 conversion (layout used after write-back) to be
    // finite; do not rely on Rigid3::to_se3 silently substituting identity.
    validate_se3_pose(rigid.to_se3(), label)
}

fn validate_candidate_poses(reconstruction: &Reconstruction) -> Result<(), String> {
    for (image, pose) in reconstruction.poses.iter().enumerate() {
        if let Some(pose) = pose {
            validate_se3_pose(*pose, &format!("image pose {image}"))?;
        }
    }
    for (frame_idx, frame) in reconstruction.frames.iter().enumerate() {
        validate_rigid3_pose(
            &frame.rig_from_world,
            &format!("frame {frame_idx} rig_from_world"),
        )?;
    }
    for rig in &reconstruction.rigs {
        for sensor in &rig.sensors {
            if let Some(rigid) = sensor.sensor_from_rig.as_ref() {
                validate_rigid3_pose(
                    rigid,
                    &format!(
                        "rig {} sensor {} sensor_from_rig",
                        rig.rig_id, sensor.sensor_id.sensor_id
                    ),
                )?;
            }
        }
    }
    Ok(())
}

fn build_validated_candidate(
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
    parameters: &[Vec<f64>],
    internal_to_storage: &HashMap<usize, usize>,
    pose_entity_registry: &HashMap<PoseEntityKey, usize>,
    frame_images: &HashMap<usize, Vec<usize>>,
    camera_param_registry: &HashMap<(usize, usize), usize>,
    camera_param_specs: &[CameraParamSpec],
    point_registry: &HashMap<usize, usize>,
    constant_point_filter: &HashSet<usize>,
    pose_blocks: &super::ceres_support::PoseBlockSet,
) -> Result<Reconstruction, String> {
    let prepared = prepare_write_back(
        reconstruction,
        parameters,
        internal_to_storage,
        pose_entity_registry,
        camera_param_registry,
        camera_param_specs,
        point_registry,
        constant_point_filter,
    )?;
    let mut candidate = reconstruction.clone();
    #[cfg(test)]
    if let Some(tx) = super::commit_test_hooks::current()
        .and_then(|hooks| hooks.seed_candidate_sensor_translation_x)
    {
        for rig in &mut candidate.rigs {
            let ref_id = rig.ref_sensor_id.clone();
            for sensor in &mut rig.sensors {
                if ref_id.as_ref() == Some(&sensor.sensor_id) {
                    continue;
                }
                let mut rigid = sensor
                    .sensor_from_rig
                    .clone()
                    .unwrap_or_else(Rigid3::identity);
                rigid.tvec[0] = tx;
                sensor.sensor_from_rig = Some(rigid);
            }
        }
    }
    apply_prepared_write_back(&mut candidate, prepared, frame_images, pose_blocks);
    // Composition (frame ∘ sensor → image) can overflow even when each operand
    // was individually finite; validate the final derived poses before errors.
    validate_candidate_poses(&candidate)?;
    refresh_point_errors_checked(frames, &mut candidate)?;
    Ok(candidate)
}

fn install_ba_candidate(dst: &mut Reconstruction, src: Reconstruction) {
    dst.camera = src.camera;
    dst.cameras = src.cameras;
    dst.rigs = src.rigs;
    dst.frames = src.frames;
    dst.poses = src.poses;
    dst.points = src.points;
}

fn pose_params_to_se3_checked(params: &[f64]) -> Result<SE3, String> {
    if params.len() < 7 {
        return Err(format!(
            "pose parameter block has {} values, expected 7",
            params.len()
        ));
    }
    for (idx, value) in params.iter().take(7).enumerate() {
        if !value.is_finite() {
            return Err(format!("pose parameter[{idx}] is non-finite ({value})"));
        }
    }
    let q = [
        params[0] as f32,
        params[1] as f32,
        params[2] as f32,
        params[3] as f32,
    ];
    let t = [params[4] as f32, params[5] as f32, params[6] as f32];
    if q.iter().any(|v| !v.is_finite()) || t.iter().any(|v| !v.is_finite()) {
        return Err("pose parameters become non-finite after f32 narrowing".into());
    }
    Ok(pose_params_to_se3(params))
}

fn point_params_to_xyz_checked(params: &[f64]) -> Result<[f32; 3], String> {
    let x = params.first().copied().unwrap_or(0.0);
    let y = params.get(1).copied().unwrap_or(0.0);
    let z = params.get(2).copied().unwrap_or(0.0);
    if !x.is_finite() || !y.is_finite() || !z.is_finite() {
        return Err(format!("point parameters are non-finite ({x}, {y}, {z})"));
    }
    let xyz = [x as f32, y as f32, z as f32];
    if xyz.iter().any(|v| !v.is_finite()) {
        return Err("point parameters become non-finite after f32 narrowing".into());
    }
    Ok(xyz)
}

fn parameter_values<'a>(
    parameters: &'a [Vec<f64>],
    internal_idx: usize,
    internal_to_storage: &HashMap<usize, usize>,
) -> Option<&'a [f64]> {
    let storage_idx = internal_to_storage.get(&internal_idx)?;
    parameters.get(*storage_idx).map(|values| values.as_slice())
}

fn pose_params_from_solution(
    parameters: &[Vec<f64>],
    handle: usize,
    internal_to_storage: &HashMap<usize, usize>,
) -> Option<SE3> {
    let values = parameter_values(parameters, handle, internal_to_storage)?;
    pose_params_to_se3_checked(values).ok()
}

fn count_variable_blocks(
    constant_blocks: &HashSet<usize>,
    block_values: &HashMap<usize, Vec<f64>>,
) -> usize {
    block_values
        .iter()
        .filter(|(idx, _)| !constant_blocks.contains(idx))
        .map(|(_, values)| values.len())
        .sum()
}

fn ceres_loss(loss: BundleAdjustmentLoss) -> LossFunction {
    match loss {
        BundleAdjustmentLoss::Trivial => LossFunction::trivial(),
        BundleAdjustmentLoss::Huber { scale } => LossFunction::huber(scale),
        BundleAdjustmentLoss::SoftL1 { scale } => LossFunction::soft_l1(scale),
        BundleAdjustmentLoss::Cauchy { scale } => LossFunction::cauchy(scale),
    }
}

fn ceres_solver_options(
    options: &BundleAdjustmentOptions,
    num_pose_entities: usize,
    num_residuals: usize,
) -> Option<(
    SolverOptions,
    CeresSolverPolicy,
    Option<BundleAdjustmentSparseLinearAlgebra>,
)> {
    let has_sparse_backend = match options.sparse_linear_algebra {
        BundleAdjustmentSparseLinearAlgebra::Auto => ceres_has_sparse_backend(),
        BundleAdjustmentSparseLinearAlgebra::SuiteSparse
        | BundleAdjustmentSparseLinearAlgebra::AccelerateSparse
        | BundleAdjustmentSparseLinearAlgebra::EigenSparse => true,
    };
    let solver_policy = ceres_solver_policy_for_preference(
        options.linear_solver,
        num_pose_entities,
        has_sparse_backend,
    );
    let max_num_iterations = ceres_i32_option(options.iterations)?;
    let max_linear_solver_iterations = ceres_i32_option(options.max_linear_solver_iterations)?;
    let max_num_consecutive_invalid_steps =
        ceres_i32_option(options.max_num_consecutive_invalid_steps)?;
    let max_consecutive_nonmonotonic_steps =
        ceres_i32_option(options.max_consecutive_nonmonotonic_steps)?;
    let mut builder = SolverOptions::builder()
        .max_num_iterations(max_num_iterations)
        .function_tolerance(options.function_tolerance)
        .gradient_tolerance(options.gradient_tolerance)
        .parameter_tolerance(options.parameter_tolerance)
        .max_linear_solver_iterations(max_linear_solver_iterations)
        .num_threads(ceres_num_threads(options, num_residuals))
        .max_num_consecutive_invalid_steps(max_num_consecutive_invalid_steps)
        .max_consecutive_nonmonotonic_steps(max_consecutive_nonmonotonic_steps)
        .linear_solver_type(solver_policy.linear_solver);
    if let Some(preconditioner) = solver_policy.preconditioner {
        builder = builder.preconditioner_type(preconditioner);
    }
    if let Some(sparse_backend) = ceres_sparse_backend_for_preference(options.sparse_linear_algebra)
    {
        builder = builder.sparse_linear_algebra_library_type(sparse_backend);
    }
    let selected_sparse_backend =
        map_ceres_sparse_backend(builder.current_sparse_linear_algebra_library_type());
    let solver_options = builder.build().ok()?;
    Some((solver_options, solver_policy, selected_sparse_backend))
}

fn ceres_i32_option(value: usize) -> Option<i32> {
    i32::try_from(value).ok()
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct CeresSolverPolicy {
    linear_solver: LinearSolverType,
    preconditioner: Option<PreconditionerType>,
}

fn ceres_solver_policy_for_preference(
    preference: BundleAdjustmentLinearSolverPreference,
    num_pose_entities: usize,
    has_sparse_backend: bool,
) -> CeresSolverPolicy {
    match preference {
        BundleAdjustmentLinearSolverPreference::DenseSchur => CeresSolverPolicy {
            linear_solver: LinearSolverType::DENSE_SCHUR,
            preconditioner: None,
        },
        BundleAdjustmentLinearSolverPreference::SparseSchur => CeresSolverPolicy {
            linear_solver: LinearSolverType::SPARSE_SCHUR,
            preconditioner: None,
        },
        BundleAdjustmentLinearSolverPreference::IterativeSchur => CeresSolverPolicy {
            linear_solver: LinearSolverType::ITERATIVE_SCHUR,
            preconditioner: Some(PreconditionerType::SCHUR_JACOBI),
        },
        BundleAdjustmentLinearSolverPreference::Auto => {
            if num_pose_entities <= 50 {
                CeresSolverPolicy {
                    linear_solver: LinearSolverType::DENSE_SCHUR,
                    preconditioner: None,
                }
            } else if has_sparse_backend && num_pose_entities <= 1000 {
                CeresSolverPolicy {
                    linear_solver: LinearSolverType::SPARSE_SCHUR,
                    preconditioner: None,
                }
            } else {
                CeresSolverPolicy {
                    linear_solver: LinearSolverType::ITERATIVE_SCHUR,
                    preconditioner: Some(PreconditionerType::SCHUR_JACOBI),
                }
            }
        }
    }
}

fn ceres_sparse_backend_for_preference(
    preference: BundleAdjustmentSparseLinearAlgebra,
) -> Option<SparseLinearAlgebraLibraryType> {
    match preference {
        BundleAdjustmentSparseLinearAlgebra::Auto => None,
        BundleAdjustmentSparseLinearAlgebra::SuiteSparse => {
            Some(SparseLinearAlgebraLibraryType::SUITE_SPARSE)
        }
        BundleAdjustmentSparseLinearAlgebra::AccelerateSparse => {
            Some(SparseLinearAlgebraLibraryType::ACCELERATE_SPARSE)
        }
        BundleAdjustmentSparseLinearAlgebra::EigenSparse => {
            Some(SparseLinearAlgebraLibraryType::EIGEN_SPARSE)
        }
    }
}

fn map_ceres_sparse_backend(
    backend: SparseLinearAlgebraLibraryType,
) -> Option<BundleAdjustmentSparseLinearAlgebra> {
    match backend {
        SparseLinearAlgebraLibraryType::SUITE_SPARSE => {
            Some(BundleAdjustmentSparseLinearAlgebra::SuiteSparse)
        }
        SparseLinearAlgebraLibraryType::ACCELERATE_SPARSE => {
            Some(BundleAdjustmentSparseLinearAlgebra::AccelerateSparse)
        }
        SparseLinearAlgebraLibraryType::EIGEN_SPARSE => {
            Some(BundleAdjustmentSparseLinearAlgebra::EigenSparse)
        }
        SparseLinearAlgebraLibraryType::CUDA_SPARSE | SparseLinearAlgebraLibraryType::NO_SPARSE => {
            None
        }
        _ => None,
    }
}

fn map_bundle_adjustment_linear_solver(policy: CeresSolverPolicy) -> BundleAdjustmentLinearSolver {
    match policy.linear_solver {
        LinearSolverType::DENSE_SCHUR => BundleAdjustmentLinearSolver::DenseSchur,
        LinearSolverType::SPARSE_SCHUR => BundleAdjustmentLinearSolver::SparseSchur,
        LinearSolverType::ITERATIVE_SCHUR => BundleAdjustmentLinearSolver::IterativeSchur,
        _ => BundleAdjustmentLinearSolver::DenseSchur,
    }
}

fn map_bundle_adjustment_preconditioner(
    policy: CeresSolverPolicy,
) -> Option<BundleAdjustmentPreconditioner> {
    match policy.preconditioner {
        Some(PreconditionerType::SCHUR_JACOBI) => Some(BundleAdjustmentPreconditioner::SchurJacobi),
        _ => None,
    }
}

fn ceres_sparse_backend() -> SparseLinearAlgebraLibraryType {
    SolverOptions::builder().current_sparse_linear_algebra_library_type()
}

fn ceres_has_sparse_backend() -> bool {
    ceres_sparse_backend() != SparseLinearAlgebraLibraryType::NO_SPARSE
}

fn ceres_num_threads(options: &BundleAdjustmentOptions, num_residuals: usize) -> i32 {
    if let Some(granted) = crate::execution::active_threads() {
        let requested = if options.num_threads > 0 {
            options.num_threads as usize
        } else {
            granted
        };
        return if num_residuals < options.min_num_residuals_for_multi_threading {
            1
        } else {
            granted.min(requested).min(i32::MAX as usize) as i32
        };
    }
    if num_residuals < options.min_num_residuals_for_multi_threading {
        1
    } else if options.num_threads <= 0 {
        std::thread::available_parallelism()
            .map(|threads| threads.get())
            .unwrap_or(1)
            .max(1) as i32
    } else {
        options.num_threads.min(i32::MAX as isize) as i32
    }
}

fn copy_pose_params(slice: &[f64], target: &mut [f64; 7]) -> Option<()> {
    if slice.len() != 7 {
        return None;
    }
    target.copy_from_slice(slice);
    Some(())
}

pub(crate) fn se3_to_pose_params(pose: SE3) -> [f64; 7] {
    let q = pose.quaternion();
    let t = pose.translation();
    [
        q[0] as f64,
        q[1] as f64,
        q[2] as f64,
        q[3] as f64,
        t[0] as f64,
        t[1] as f64,
        t[2] as f64,
    ]
}

fn pose_params_to_se3(params: &[f64]) -> SE3 {
    let rotation = if params.len() >= 4 {
        crate::geometry::quat_from_xyzw(
            params[0] as f32,
            params[1] as f32,
            params[2] as f32,
            params[3] as f32,
        )
    } else {
        Quat::identity()
    };
    SE3::from_quat_translation(
        rotation,
        Vec3::new(
            params.get(4).copied().unwrap_or(0.0) as f32,
            params.get(5).copied().unwrap_or(0.0) as f32,
            params.get(6).copied().unwrap_or(0.0) as f32,
        ),
    )
}

fn map_ceres_summary(
    summary: &ceres_solver::solver::SolverSummary,
) -> (
    BundleAdjustmentTerminationType,
    BundleAdjustmentTerminationReason,
    f64,
    f64,
    f64,
    f64,
) {
    let full = summary.full_report();
    let brief = summary.brief_report();
    let source = if full.contains("Termination:") {
        &full
    } else {
        &brief
    };

    let termination_type = map_ceres_termination_type(summary.termination_type());
    let termination_reason = parse_ceres_termination_reason(source, summary);
    let gradient_max_norm = finite_or_none(summary.last_gradient_max_norm())
        .or_else(|| parse_ceres_gradient_max_norm(&full))
        .or_else(|| parse_ceres_gradient_max_norm(&brief))
        .or_else(|| parse_ceres_gradient_from_brief_table(&brief))
        .or_else(|| parse_ceres_gradient_from_brief_table(&full))
        .unwrap_or(f64::NAN);
    let mut gradient_max_norm = gradient_max_norm;
    if gradient_max_norm.is_nan() && summary.is_solution_usable() {
        let initial = summary.initial_cost();
        let final_cost = summary.final_cost();
        if initial > 0.0 && final_cost / initial <= 1.0e-6 {
            gradient_max_norm = 0.0;
        }
    }
    let step_norm = finite_or_none(summary.last_step_norm())
        .or_else(|| parse_ceres_scalar_field(source, "Step norm"))
        .or_else(|| parse_ceres_step_norm_from_table(source))
        .or_else(|| parse_ceres_step_norm_from_table(&brief))
        .unwrap_or(f64::NAN);
    let step_quality = finite_or_none(summary.last_relative_decrease()).unwrap_or(f64::NAN);
    let damping = finite_or_none(summary.last_trust_region_radius())
        .filter(|radius| *radius > 0.0)
        .unwrap_or(f64::NAN);
    (
        termination_type,
        termination_reason,
        gradient_max_norm,
        step_norm,
        step_quality,
        damping,
    )
}

fn finite_or_none(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn map_ceres_termination_type(ceres_type: TerminationType) -> BundleAdjustmentTerminationType {
    match ceres_type {
        TerminationType::Convergence => BundleAdjustmentTerminationType::Convergence,
        TerminationType::NoConvergence => BundleAdjustmentTerminationType::NoConvergence,
        TerminationType::Failure => BundleAdjustmentTerminationType::Failure,
        TerminationType::UserSuccess => BundleAdjustmentTerminationType::UserSuccess,
        TerminationType::UserFailure => BundleAdjustmentTerminationType::UserFailure,
        TerminationType::Unknown(_) => BundleAdjustmentTerminationType::Failure,
    }
}

fn parse_ceres_termination_reason(
    report: &str,
    summary: &ceres_solver::solver::SolverSummary,
) -> BundleAdjustmentTerminationReason {
    let termination_line = report
        .lines()
        .find(|line| line.contains("Termination:"))
        .unwrap_or("");
    let message = summary.message();
    let upper = if termination_line.is_empty() {
        message.to_ascii_uppercase()
    } else {
        format!(
            "{} {}",
            termination_line.to_ascii_uppercase(),
            message.to_ascii_uppercase()
        )
    };

    if upper.contains("MAXIMUM") || upper.contains("MAX NUM") {
        BundleAdjustmentTerminationReason::MaxIterations
    } else if upper.contains("GRADIENT") {
        BundleAdjustmentTerminationReason::GradientTolerance
    } else if upper.contains("FUNCTION") {
        BundleAdjustmentTerminationReason::FunctionTolerance
    } else if upper.contains("PARAMETER") {
        BundleAdjustmentTerminationReason::ParameterTolerance
    } else if upper.contains("NO CONVERGENCE") || upper.contains("MAXIMUM") {
        BundleAdjustmentTerminationReason::MaxIterations
    } else if summary.is_solution_usable() {
        BundleAdjustmentTerminationReason::GradientTolerance
    } else {
        BundleAdjustmentTerminationReason::MaxIterations
    }
}

fn parse_ceres_gradient_max_norm(report: &str) -> Option<f64> {
    for line in report.lines() {
        let Some(idx) = line.find("Gradient max norm:") else {
            continue;
        };
        let rest = line[idx + "Gradient max norm:".len()..].trim();
        let token = rest.split_whitespace().next()?;
        return token.parse().ok();
    }
    None
}

fn parse_ceres_step_norm_from_table(report: &str) -> Option<f64> {
    for line in report.lines() {
        if !line.trim_start().starts_with('0') && !line.trim_start().starts_with('1') {
            continue;
        }
        let cols: Vec<&str> = line.split_whitespace().collect();
        // Iteration table: iter cost cost_change |gradient| |step| ...
        if cols.len() >= 5 {
            if let Ok(step) = cols[4].parse::<f64>() {
                return Some(step);
            }
        }
    }
    None
}

fn parse_ceres_gradient_from_brief_table(report: &str) -> Option<f64> {
    let mut last = None;
    for line in report.lines() {
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() >= 4 && cols[0].parse::<i32>().is_ok() {
            if let Ok(value) = cols[3].parse::<f64>() {
                last = Some(value);
            }
        }
    }
    last
}

fn parse_ceres_scalar_field(report: &str, label: &str) -> Option<f64> {
    for line in report.lines() {
        if !line.contains(label) {
            continue;
        }
        let value = line
            .split_whitespace()
            .last()
            .or_else(|| line.rsplit(':').next())?;
        return value.trim().parse().ok();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::super::{
        commit_test_hooks::{with_hooks, BaCommitTestHooks},
        BundleAdjustmentLinearSolver, BundleAdjustmentTerminationType,
    };
    use super::*;
    use crate::sift::SiftFeatures;
    use crate::task::SfmTaskControl;
    use crate::types::{CameraModel, ImageFrame, Point3D, TrackObservation};
    use crate::wide::WideDescriptors;
    type Quat = nalgebra::UnitQuaternion<f32>;
    type Vec3 = nalgebra::Vector3<f32>;
    use rustscan_slam::Descriptors;
    use rustscan_slam::KeyPoint;
    use rustscan_slam::SE3;
    use std::path::PathBuf;

    #[derive(Clone)]
    struct BaMutableSnapshot {
        legacy_camera_params: Vec<f64>,
        camera_params: Vec<Vec<f64>>,
        poses: Vec<[f64; 7]>,
        pose_present: Vec<bool>,
        points_xyz: Vec<[f32; 3]>,
        point_errors: Vec<f32>,
        frame_poses: Vec<[f64; 7]>,
        sensor_poses: Vec<Option<[f64; 7]>>,
    }

    fn snapshot_ba_state(reconstruction: &Reconstruction) -> BaMutableSnapshot {
        let camera_params = reconstruction
            .cameras
            .iter()
            .map(|camera| camera.params_slice().to_vec())
            .collect();
        let mut poses = Vec::with_capacity(reconstruction.poses.len());
        let mut pose_present = Vec::with_capacity(reconstruction.poses.len());
        for pose in &reconstruction.poses {
            match pose {
                Some(pose) => {
                    pose_present.push(true);
                    poses.push(se3_to_pose_params(*pose));
                }
                None => {
                    pose_present.push(false);
                    poses.push([0.0; 7]);
                }
            }
        }
        let frame_poses = reconstruction
            .frames
            .iter()
            .map(|frame| se3_to_pose_params(frame.rig_from_world.to_se3()))
            .collect();
        let sensor_poses = reconstruction
            .rigs
            .iter()
            .flat_map(|rig| {
                rig.sensors.iter().map(|sensor| {
                    sensor
                        .sensor_from_rig
                        .as_ref()
                        .map(|rigid| se3_to_pose_params(rigid.to_se3()))
                })
            })
            .collect();
        BaMutableSnapshot {
            legacy_camera_params: reconstruction.camera.params_slice().to_vec(),
            camera_params,
            poses,
            pose_present,
            points_xyz: reconstruction.points.iter().map(|p| p.xyz).collect(),
            point_errors: reconstruction.points.iter().map(|p| p.error).collect(),
            frame_poses,
            sensor_poses,
        }
    }

    fn assert_bits_eq_f64(label: &str, left: &[f64], right: &[f64]) {
        assert_eq!(left.len(), right.len(), "{label}: length mismatch");
        for (idx, (a, b)) in left.iter().zip(right.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{label}[{idx}]: left={a:?} right={b:?}"
            );
        }
    }

    fn assert_bits_eq_f32(label: &str, left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len(), "{label}: length mismatch");
        for (idx, (a, b)) in left.iter().zip(right.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{label}[{idx}]: left={a:?} right={b:?}"
            );
        }
    }

    fn assert_ba_state_unchanged(before: &BaMutableSnapshot, after: &Reconstruction) {
        let after = snapshot_ba_state(after);
        assert_bits_eq_f64(
            "legacy_camera",
            &before.legacy_camera_params,
            &after.legacy_camera_params,
        );
        assert_eq!(before.camera_params.len(), after.camera_params.len());
        for (idx, (left, right)) in before
            .camera_params
            .iter()
            .zip(after.camera_params.iter())
            .enumerate()
        {
            assert_bits_eq_f64(&format!("camera[{idx}]"), left, right);
        }
        assert_eq!(before.pose_present, after.pose_present);
        assert_eq!(before.poses.len(), after.poses.len());
        for (idx, (left, right)) in before.poses.iter().zip(after.poses.iter()).enumerate() {
            assert_bits_eq_f64(&format!("pose[{idx}]"), left, right);
        }
        assert_eq!(before.points_xyz.len(), after.points_xyz.len());
        for (idx, (left, right)) in before
            .points_xyz
            .iter()
            .zip(after.points_xyz.iter())
            .enumerate()
        {
            assert_bits_eq_f32(&format!("point_xyz[{idx}]"), left, right);
        }
        assert_bits_eq_f32("point_errors", &before.point_errors, &after.point_errors);
        assert_eq!(before.frame_poses.len(), after.frame_poses.len());
        for (idx, (left, right)) in before
            .frame_poses
            .iter()
            .zip(after.frame_poses.iter())
            .enumerate()
        {
            assert_bits_eq_f64(&format!("frame_pose[{idx}]"), left, right);
        }
        assert_eq!(before.sensor_poses.len(), after.sensor_poses.len());
        for (idx, (left, right)) in before
            .sensor_poses
            .iter()
            .zip(after.sensor_poses.iter())
            .enumerate()
        {
            match (left, right) {
                (None, None) => {}
                (Some(left), Some(right)) => {
                    assert_bits_eq_f64(&format!("sensor_pose[{idx}]"), left, right);
                }
                _ => panic!("sensor_pose[{idx}] presence mismatch"),
            }
        }
    }

    fn trivial_single_observation_scene() -> (Vec<ImageFrame>, Reconstruction) {
        let frames = vec![ImageFrame {
            id: 0,
            name: "0.jpg".into(),
            path: PathBuf::from("0.jpg"),
            width: 100,
            height: 100,
            keypoints: vec![KeyPoint::new(50.0, 50.0)],
            descriptors: Descriptors::new(),
            sift: SiftFeatures::default(),
            wide_descriptors: WideDescriptors {
                data: Vec::new(),
                dim: 0,
                count: 0,
            },
            strong_feature_indices: Vec::new(),
            colors: Vec::new(),
        }];
        let reconstruction = Reconstruction {
            camera: CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0),
            cameras: vec![CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0)],
            camera_ids: vec![1],
            rigs: Vec::new(),
            frames: Vec::new(),
            image_names: vec!["0.jpg".into()],
            image_paths: vec![PathBuf::from("0.jpg")],
            image_ids: vec![1],
            image_camera_indices: vec![0],
            image_frame_indices: vec![None],
            poses: vec![Some(SE3::identity())],
            observations: vec![vec![Some(0)]],
            keypoints: frames.iter().map(|f| f.keypoints.clone()).collect(),
            point_ids: vec![1],
            points: vec![Point3D {
                xyz: [0.0, 0.0, 2.0],
                color: [0, 0, 0],
                error: 1.25,
                track: vec![TrackObservation {
                    image: 0,
                    feature: 0,
                }],
            }],
        };
        (frames, reconstruction)
    }

    fn non_identity_multi_camera_scene() -> (Vec<ImageFrame>, Reconstruction) {
        let pose0 = SE3::from_quat_translation(Quat::identity(), Vec3::new(0.1, -0.05, 0.02));
        let pose1 = SE3::from_quat_translation(Quat::identity(), Vec3::new(0.85, 0.03, -0.01));
        let cam0 = CameraModel::new_pinhole(120, 90, 70.0, 71.0, 60.0, 45.0);
        let cam1 = CameraModel::new_pinhole(120, 90, 68.0, 69.0, 58.0, 44.0);
        // Intentionally diverge legacy camera from cameras[0] so snapshot must
        // compare both independently.
        let legacy = CameraModel::new_pinhole(120, 90, 65.0, 66.0, 55.0, 40.0);
        let frames = vec![
            ImageFrame {
                id: 0,
                name: "0.jpg".into(),
                path: PathBuf::from("0.jpg"),
                width: 120,
                height: 90,
                keypoints: vec![KeyPoint::new(62.0, 44.0)],
                descriptors: Descriptors::new(),
                sift: SiftFeatures::default(),
                wide_descriptors: WideDescriptors {
                    data: Vec::new(),
                    dim: 0,
                    count: 0,
                },
                strong_feature_indices: Vec::new(),
                colors: Vec::new(),
            },
            ImageFrame {
                id: 1,
                name: "1.jpg".into(),
                path: PathBuf::from("1.jpg"),
                width: 120,
                height: 90,
                keypoints: vec![KeyPoint::new(40.0, 46.0)],
                descriptors: Descriptors::new(),
                sift: SiftFeatures::default(),
                wide_descriptors: WideDescriptors {
                    data: Vec::new(),
                    dim: 0,
                    count: 0,
                },
                strong_feature_indices: Vec::new(),
                colors: Vec::new(),
            },
        ];
        let sensor0 = crate::types::SensorId {
            sensor_type: crate::types::SensorType::Camera,
            sensor_id: 1,
        };
        let sensor1 = crate::types::SensorId {
            sensor_type: crate::types::SensorType::Camera,
            sensor_id: 2,
        };
        let reconstruction = Reconstruction {
            camera: legacy,
            cameras: vec![cam0, cam1],
            camera_ids: vec![1, 2],
            rigs: vec![crate::types::Rig {
                rig_id: 7,
                ref_sensor_id: Some(sensor0.clone()),
                sensors: vec![
                    crate::types::RigSensor {
                        sensor_id: sensor0.clone(),
                        sensor_from_rig: None,
                    },
                    crate::types::RigSensor {
                        sensor_id: sensor1.clone(),
                        sensor_from_rig: Some(Rigid3::from_se3(SE3::from_quat_translation(
                            Quat::identity(),
                            Vec3::new(0.2, 0.0, 0.0),
                        ))),
                    },
                ],
            }],
            frames: vec![crate::types::Frame {
                frame_id: 3,
                rig_id: 7,
                rig_from_world: Rigid3::from_se3(pose0),
                data_ids: vec![
                    crate::types::DataId {
                        sensor_id: sensor0,
                        data_id: 1,
                    },
                    crate::types::DataId {
                        sensor_id: sensor1,
                        data_id: 2,
                    },
                ],
            }],
            image_names: vec!["0.jpg".into(), "1.jpg".into()],
            image_paths: vec![PathBuf::from("0.jpg"), PathBuf::from("1.jpg")],
            image_ids: vec![1, 2],
            image_camera_indices: vec![0, 1],
            image_frame_indices: vec![Some(0), Some(0)],
            poses: vec![Some(pose0), Some(pose1)],
            observations: vec![vec![Some(0)], vec![Some(0)]],
            keypoints: frames.iter().map(|f| f.keypoints.clone()).collect(),
            point_ids: vec![1],
            points: vec![Point3D {
                xyz: [0.4, 0.0, 2.0],
                color: [0, 0, 0],
                error: 2.5,
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
            }],
        };
        (frames, reconstruction)
    }

    fn solve_with_hooks(
        frames: &[ImageFrame],
        reconstruction: &mut Reconstruction,
        options: BundleAdjustmentOptions,
        hooks: BaCommitTestHooks,
        control: Option<&SfmTaskControl>,
    ) -> Option<BundleAdjustmentReport> {
        with_hooks(hooks, || {
            solve_bundle_adjustment_ceres(frames, reconstruction, options, control)
        })
    }

    #[test]
    fn rejected_ba_solutions_leave_reconstruction_unchanged() {
        let (frames, mut reconstruction) = trivial_single_observation_scene();

        for (label, hooks) in [
            (
                "failure",
                BaCommitTestHooks {
                    force_ceres_usable: Some(false),
                    force_termination: Some(BundleAdjustmentTerminationType::Failure),
                    corrupt_first_camera_param: None,
                    corrupt_first_pose_translation_x: None,
                    corrupt_all_pose_translations_x: None,
                    seed_candidate_sensor_translation_x: None,
                    cancel_before_commit: false,
                },
            ),
            (
                "user_failure",
                BaCommitTestHooks {
                    force_ceres_usable: Some(false),
                    force_termination: Some(BundleAdjustmentTerminationType::UserFailure),
                    corrupt_first_camera_param: None,
                    corrupt_first_pose_translation_x: None,
                    corrupt_all_pose_translations_x: None,
                    seed_candidate_sensor_translation_x: None,
                    cancel_before_commit: false,
                },
            ),
            (
                "nan_camera",
                BaCommitTestHooks {
                    force_ceres_usable: Some(true),
                    force_termination: Some(BundleAdjustmentTerminationType::Convergence),
                    corrupt_first_camera_param: Some(f64::NAN),
                    corrupt_first_pose_translation_x: None,
                    corrupt_all_pose_translations_x: None,
                    seed_candidate_sensor_translation_x: None,
                    cancel_before_commit: false,
                },
            ),
            (
                "inf_camera",
                BaCommitTestHooks {
                    force_ceres_usable: Some(true),
                    force_termination: Some(BundleAdjustmentTerminationType::NoConvergence),
                    corrupt_first_camera_param: Some(f64::INFINITY),
                    corrupt_first_pose_translation_x: None,
                    corrupt_all_pose_translations_x: None,
                    seed_candidate_sensor_translation_x: None,
                    cancel_before_commit: false,
                },
            ),
            (
                "non_positive_focal",
                BaCommitTestHooks {
                    force_ceres_usable: Some(true),
                    force_termination: Some(BundleAdjustmentTerminationType::UserSuccess),
                    corrupt_first_camera_param: Some(-1.0),
                    corrupt_first_pose_translation_x: None,
                    corrupt_all_pose_translations_x: None,
                    seed_candidate_sensor_translation_x: None,
                    cancel_before_commit: false,
                },
            ),
            (
                "finite_focal_infinite_point_error",
                BaCommitTestHooks {
                    force_ceres_usable: Some(true),
                    force_termination: Some(BundleAdjustmentTerminationType::Convergence),
                    corrupt_first_camera_param: Some(3.0e38),
                    corrupt_first_pose_translation_x: None,
                    corrupt_all_pose_translations_x: None,
                    seed_candidate_sensor_translation_x: None,
                    cancel_before_commit: false,
                },
            ),
        ] {
            let (mut case_frames, mut case_reconstruction) = trivial_single_observation_scene();
            if label == "finite_focal_infinite_point_error" {
                case_reconstruction.points[0].xyz = [4.0, 0.0, 2.0];
                case_frames[0].keypoints[0] = KeyPoint::new(150.0, 50.0);
            }
            let before = snapshot_ba_state(&case_reconstruction);
            let report = solve_with_hooks(
                &case_frames,
                &mut case_reconstruction,
                BundleAdjustmentOptions {
                    iterations: 5,
                    allow_single_observation_points: true,
                    constant_images: if label == "finite_focal_infinite_point_error" {
                        vec![0]
                    } else {
                        Vec::new()
                    },
                    constant_point_ids: if label == "finite_focal_infinite_point_error" {
                        Some(vec![0])
                    } else {
                        None
                    },
                    refine_focal_length: true,
                    ..BundleAdjustmentOptions::default()
                },
                hooks,
                None,
            )
            .unwrap_or_else(|| panic!("{label}: expected report"));
            assert!(
                !report.is_solution_usable(),
                "{label}: report must be unusable"
            );
            assert_ba_state_unchanged(&before, &case_reconstruction);
            let _ = (&frames, &mut reconstruction);
        }
    }

    #[test]
    fn rejected_ba_on_rig_scene_leaves_all_mutable_state_unchanged() {
        let (frames, mut reconstruction) = non_identity_multi_camera_scene();
        let before = snapshot_ba_state(&reconstruction);
        assert!(
            !before.frame_poses.is_empty() && !before.sensor_poses.is_empty(),
            "fixture must exercise frames/sensors"
        );
        assert_ne!(
            before.legacy_camera_params, before.camera_params[0],
            "fixture must keep independent legacy camera"
        );
        let report = solve_with_hooks(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 5,
                refine_focal_length: true,
                variable_images: Some(vec![0, 1]),
                constant_images: vec![0],
                ..BundleAdjustmentOptions::default()
            },
            BaCommitTestHooks {
                force_ceres_usable: Some(false),
                force_termination: Some(BundleAdjustmentTerminationType::Failure),
                corrupt_first_camera_param: Some(3.0e38),
                corrupt_first_pose_translation_x: None,
                corrupt_all_pose_translations_x: None,
                seed_candidate_sensor_translation_x: None,
                cancel_before_commit: false,
            },
            None,
        )
        .expect("report");
        assert!(!report.is_solution_usable());
        assert_ba_state_unchanged(&before, &reconstruction);
    }

    #[test]
    fn candidate_rejects_nonfinite_composed_frame_sensor_pose() {
        let (frames, reconstruction) = non_identity_multi_camera_scene();
        let mut reconstruction = reconstruction;
        reconstruction.rigs[0].sensors[1].sensor_from_rig = Some(Rigid3::from_se3(
            SE3::from_quat_translation(Quat::identity(), Vec3::new(3.0e38, 0.0, 0.0)),
        ));
        reconstruction.points[0].track.retain(|obs| obs.image == 1);
        let poses = variable_pose_blocks(&reconstruction, None, &[], &[], false);
        let parameters = vec![vec![0.0, 0.0, 0.0, 1.0, 3.0e38, 0.0, 0.0]];
        let storage = HashMap::from([(0usize, 0usize)]);
        let registry = HashMap::from([(PoseEntityKey::Frame(0), 0usize)]);
        let frame_images = HashMap::from([(0usize, vec![0usize, 1usize])]);
        let candidate = build_validated_candidate(
            &frames,
            &reconstruction,
            &parameters,
            &storage,
            &registry,
            &frame_images,
            &HashMap::new(),
            &[],
            &HashMap::new(),
            &HashSet::new(),
            &poses,
        );
        assert!(
            candidate.is_err(),
            "non-finite composed pose must be rejected before install: {candidate:?}"
        );
    }

    #[test]
    fn usable_solve_rejects_nonfinite_composed_pose_without_partial_write() {
        let (frames, mut reconstruction) = non_identity_multi_camera_scene();
        let before = snapshot_ba_state(&reconstruction);
        let report = solve_with_hooks(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 5,
                refine_focal_length: true,
                variable_images: Some(vec![0, 1]),
                // Both images share one frame; keeping image 0 constant would
                // drop the entire frame (and sensor) from the pose registry.
                constant_images: Vec::new(),
                gauge: BundleAdjustmentGauge::None,
                ..BundleAdjustmentOptions::default()
            },
            BaCommitTestHooks {
                force_ceres_usable: Some(true),
                force_termination: Some(BundleAdjustmentTerminationType::Convergence),
                corrupt_first_camera_param: None,
                // Finite frame translation 3e38 plus seeded sensor 3e38 compose
                // to Inf; usable Convergence must still reject install.
                corrupt_first_pose_translation_x: Some(3.0e38),
                corrupt_all_pose_translations_x: None,
                seed_candidate_sensor_translation_x: Some(3.0e38),
                cancel_before_commit: false,
            },
            None,
        )
        .expect("report");
        assert!(
            !report.is_solution_usable(),
            "usable solver summary must not install a non-finite composed candidate"
        );
        assert_eq!(
            report.termination_type,
            BundleAdjustmentTerminationType::Failure
        );
        assert_ba_state_unchanged(&before, &reconstruction);
    }

    #[test]
    fn candidate_error_refresh_rejects_nonfinite_camera_coordinates() {
        let pose = SE3::from_quat_translation(Quat::identity(), Vec3::new(f32::INFINITY, 0.0, 0.0));
        let camera = CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0);
        let err = project_point_for_candidate_error(camera, pose, [0.0, 0.0, 2.0], 0, 0, 0);
        assert!(
            err.is_err(),
            "non-finite camera coordinates must fail candidate error refresh: {err:?}"
        );
    }

    #[test]
    fn candidate_error_refresh_skips_finite_behind_camera_geometry() {
        let pose = SE3::identity();
        let camera = CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0);
        let projected = project_point_for_candidate_error(camera, pose, [0.0, 0.0, -2.0], 0, 0, 0)
            .expect("behind-camera must remain a geometric skip, not a numerical failure");
        assert!(projected.is_none());

        let (frames, mut reconstruction) = trivial_single_observation_scene();
        reconstruction.points[0].xyz = [0.0, 0.0, -2.0];
        let before_error = reconstruction.points[0].error;
        refresh_point_errors_checked(&frames, &mut reconstruction)
            .expect("finite behind-camera observations must not reject the candidate");
        assert_eq!(
            reconstruction.points[0].error.to_bits(),
            before_error.to_bits(),
            "skipped geometric observations must retain the prior finite error"
        );
    }

    #[test]
    fn candidate_error_refresh_rejects_projection_overflow_returning_none() {
        // Finite depth with extreme lateral offset: f64 projection stays finite
        // but overflows f32. That must fail candidate validation (same as a
        // finite2-filtered None from Inf image coords).
        let pose = SE3::identity();
        let camera = CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0);
        let err = project_point_for_candidate_error(camera, pose, [3.0e38, 0.0, 1.0], 0, 0, 0);
        assert!(
            err.is_err(),
            "projection overflow must reject the candidate: {err:?}"
        );
    }

    #[test]
    fn candidate_error_refresh_rejects_simple_radial_distortion_overflow() {
        let camera = CameraModel::from_colmap(
            crate::types::COLMAP_SIMPLE_RADIAL,
            100,
            100,
            &[50.0, 50.0, 50.0, 1e308],
        )
        .expect("finite extreme radial camera");
        assert!(
            camera.img_from_cam_unchecked(2.0, 0.0, 1.0).is_none(),
            "actual distorted projection must overflow"
        );
        assert_eq!(
            camera.classify_img_from_cam_unchecked(2.0, 0.0, 1.0),
            crate::types::CameraProjectionOutcome::NonFiniteProjection
        );
        let err =
            project_point_for_candidate_error(camera, SE3::identity(), [2.0, 0.0, 1.0], 0, 0, 0);
        assert!(
            err.is_err(),
            "numeric distortion failure must not be classified as geometric skip: {err:?}"
        );
    }

    #[test]
    fn candidate_error_refresh_skips_finite_division_domain_rejection() {
        let camera = CameraModel::from_colmap(
            crate::types::COLMAP_SIMPLE_DIVISION,
            100,
            100,
            &[50.0, 50.0, 50.0, 10.0],
        )
        .expect("division camera");
        assert_eq!(
            camera.classify_img_from_cam_unchecked(10.0, 0.0, 1.0),
            crate::types::CameraProjectionOutcome::FiniteGeometricDomainSkip
        );
        let projected =
            project_point_for_candidate_error(camera, SE3::identity(), [10.0, 0.0, 1.0], 0, 0, 0)
                .expect("finite geometric domain rejection must remain a skip");
        assert!(projected.is_none());
    }

    #[test]
    fn candidate_error_refresh_rejects_opencv_distortion_overflow() {
        let camera = CameraModel::from_colmap(
            crate::types::COLMAP_OPENCV,
            100,
            100,
            &[50.0, 50.0, 50.0, 50.0, 1e308, 0.0, 0.0, 0.0],
        )
        .expect("finite extreme opencv camera");
        assert_eq!(
            camera.classify_img_from_cam_unchecked(2.0, 0.0, 1.0),
            crate::types::CameraProjectionOutcome::NonFiniteProjection
        );
        let err =
            project_point_for_candidate_error(camera, SE3::identity(), [2.0, 0.0, 1.0], 0, 0, 0);
        assert!(
            err.is_err(),
            "opencv distortion overflow must reject: {err:?}"
        );
    }

    #[test]
    fn usable_solve_rejects_radial_distortion_overflow_without_partial_write() {
        let (mut frames, mut reconstruction) = trivial_single_observation_scene();
        // Start from a finite SIMPLE_RADIAL model; inject extreme k after the
        // usable solve (same pattern as finite_focal_infinite_point_error).
        let radial = CameraModel::from_colmap(
            crate::types::COLMAP_SIMPLE_RADIAL,
            100,
            100,
            &[50.0, 50.0, 50.0, 0.0],
        )
        .expect("finite radial camera");
        reconstruction.camera = radial;
        reconstruction.cameras = vec![radial];
        // Point [2,0,1] projects to (150,50) with k=0; keep the observation
        // inside max_observation_error_px so the residual is included.
        reconstruction.points[0].xyz = [2.0, 0.0, 1.0];
        frames[0].keypoints[0] = KeyPoint::new(150.0, 50.0);
        reconstruction.keypoints[0][0] = KeyPoint::new(150.0, 50.0);
        let before = snapshot_ba_state(&reconstruction);
        let report = solve_with_hooks(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 5,
                allow_single_observation_points: true,
                constant_images: vec![0],
                constant_point_ids: Some(vec![0]),
                // Only the distortion (extra) block is free, so
                // corrupt_first_camera_param targets k rather than focal.
                refine_extra_params: true,
                ..BundleAdjustmentOptions::default()
            },
            BaCommitTestHooks {
                force_ceres_usable: Some(true),
                force_termination: Some(BundleAdjustmentTerminationType::Convergence),
                corrupt_first_camera_param: Some(1e308),
                corrupt_first_pose_translation_x: None,
                corrupt_all_pose_translations_x: None,
                seed_candidate_sensor_translation_x: None,
                cancel_before_commit: false,
            },
            None,
        )
        .expect("report");
        assert!(
            !report.is_solution_usable(),
            "usable solver summary must not install a non-finite distortion candidate"
        );
        assert_eq!(
            report.termination_type,
            BundleAdjustmentTerminationType::Failure
        );
        assert_ba_state_unchanged(&before, &reconstruction);
    }

    #[test]
    fn cancel_after_solve_before_commit_does_not_mutate() {
        let (frames, mut reconstruction) = non_identity_multi_camera_scene();
        let before = snapshot_ba_state(&reconstruction);
        let control = SfmTaskControl::new();
        let report = solve_with_hooks(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 5,
                allow_single_observation_points: true,
                ..BundleAdjustmentOptions::default()
            },
            BaCommitTestHooks {
                force_ceres_usable: Some(true),
                force_termination: Some(BundleAdjustmentTerminationType::Convergence),
                corrupt_first_camera_param: None,
                corrupt_first_pose_translation_x: None,
                corrupt_all_pose_translations_x: None,
                seed_candidate_sensor_translation_x: None,
                cancel_before_commit: true,
            },
            Some(&control),
        );
        assert!(
            report.is_none(),
            "post-solve pre-commit cancel must discard the report"
        );
        assert_ba_state_unchanged(&before, &reconstruction);
    }

    #[test]
    fn usable_no_convergence_still_commits_when_candidate_valid() {
        let (frames, mut reconstruction) = trivial_single_observation_scene();
        let before = snapshot_ba_state(&reconstruction);
        let report = solve_with_hooks(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 5,
                allow_single_observation_points: true,
                refine_focal_length: true,
                ..BundleAdjustmentOptions::default()
            },
            BaCommitTestHooks {
                force_ceres_usable: Some(true),
                force_termination: Some(BundleAdjustmentTerminationType::NoConvergence),
                corrupt_first_camera_param: None,
                corrupt_first_pose_translation_x: None,
                corrupt_all_pose_translations_x: None,
                seed_candidate_sensor_translation_x: None,
                cancel_before_commit: false,
            },
            None,
        )
        .expect("usable no-convergence should return a report");
        assert!(report.is_solution_usable());
        assert_eq!(
            report.termination_type,
            BundleAdjustmentTerminationType::NoConvergence
        );
        assert_ne!(
            reconstruction.points[0].error.to_bits(),
            before.point_errors[0].to_bits()
        );
        assert!(reconstruction.points[0].error.is_finite());
    }

    #[test]
    fn usable_user_success_commits_when_candidate_valid() {
        let (frames, mut reconstruction) = trivial_single_observation_scene();
        let before_error = reconstruction.points[0].error;
        let report = solve_with_hooks(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 5,
                allow_single_observation_points: true,
                ..BundleAdjustmentOptions::default()
            },
            BaCommitTestHooks {
                force_ceres_usable: Some(true),
                force_termination: Some(BundleAdjustmentTerminationType::UserSuccess),
                corrupt_first_camera_param: None,
                corrupt_first_pose_translation_x: None,
                corrupt_all_pose_translations_x: None,
                seed_candidate_sensor_translation_x: None,
                cancel_before_commit: false,
            },
            None,
        )
        .expect("usable user-success should return a report");
        assert!(report.is_solution_usable());
        assert_eq!(
            report.termination_type,
            BundleAdjustmentTerminationType::UserSuccess
        );
        assert!(reconstruction.points[0].error.is_finite());
        assert_ne!(
            reconstruction.points[0].error.to_bits(),
            before_error.to_bits()
        );
    }

    #[test]
    fn successful_convergence_commits_geometry_and_point_errors() {
        let (frames, mut reconstruction) = trivial_single_observation_scene();
        // Deliberately wrong point so a successful solve must move geometry.
        reconstruction.points[0].xyz = [0.4, -0.3, 2.5];
        reconstruction.points[0].error = 9.0;
        let before_xyz = reconstruction.points[0].xyz;
        let before_error = reconstruction.points[0].error;
        let report = solve_bundle_adjustment_ceres(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 25,
                allow_single_observation_points: true,
                ..BundleAdjustmentOptions::default()
            },
            None,
        )
        .expect("convergence path");
        assert!(report.is_solution_usable());
        assert_eq!(
            report.termination_type,
            BundleAdjustmentTerminationType::Convergence
        );
        assert!(reconstruction.points[0].error.is_finite());
        assert!(
            reconstruction.points[0].error < before_error,
            "committed point error must improve: before={before_error} after={}",
            reconstruction.points[0].error
        );
        let moved =
            (0..3).any(|i| reconstruction.points[0].xyz[i].to_bits() != before_xyz[i].to_bits());
        assert!(moved, "successful BA must update point geometry");
    }

    #[test]
    fn ceres_full_report_contains_termination_and_gradient_fields() {
        let frames = vec![ImageFrame {
            id: 0,
            name: "0.jpg".into(),
            path: PathBuf::from("0.jpg"),
            width: 100,
            height: 100,
            keypoints: vec![KeyPoint::new(50.0, 50.0)],
            descriptors: Descriptors::new(),
            sift: SiftFeatures::default(),
            wide_descriptors: WideDescriptors {
                data: Vec::new(),
                dim: 0,
                count: 0,
            },
            strong_feature_indices: Vec::new(),
            colors: Vec::new(),
        }];
        let mut reconstruction = Reconstruction {
            camera: CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0),
            cameras: vec![CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0)],
            camera_ids: vec![1],
            rigs: Vec::new(),
            frames: Vec::new(),
            image_names: vec!["0.jpg".into()],
            image_paths: vec![PathBuf::from("0.jpg")],
            image_ids: vec![1],
            image_camera_indices: vec![0],
            image_frame_indices: vec![None],
            poses: vec![Some(SE3::identity())],
            observations: vec![vec![Some(0)]],
            keypoints: frames.iter().map(|f| f.keypoints.clone()).collect(),
            point_ids: vec![1],
            points: vec![Point3D {
                xyz: [0.0, 0.0, 2.0],
                color: [0, 0, 0],
                error: 0.0,
                track: vec![TrackObservation {
                    image: 0,
                    feature: 0,
                }],
            }],
        };
        let report = solve_bundle_adjustment_ceres(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 5,
                allow_single_observation_points: true,
                ..BundleAdjustmentOptions::default()
            },
            None,
        )
        .expect("ba should succeed");
        assert!(report.gradient_max_norm.is_finite());
        assert!(report.step_norm.is_finite());
        assert!(report.step_quality.is_finite());
        assert!(report.damping.is_finite());
        assert_eq!(
            report.termination_reason,
            BundleAdjustmentTerminationReason::GradientTolerance
        );
        assert_eq!(report.residuals, 2);
        assert_eq!(report.effective_parameters, 3);
        assert_eq!(
            report.linear_solver,
            BundleAdjustmentLinearSolver::DenseSchur
        );
        assert!(report.preconditioner.is_none());
    }

    #[test]
    fn ceres_trivial_loss_matches_colmap_explicit_loss_function() {
        let frames = vec![ImageFrame {
            id: 0,
            name: "0.jpg".into(),
            path: PathBuf::from("0.jpg"),
            width: 100,
            height: 100,
            keypoints: vec![KeyPoint::new(50.0, 50.0)],
            descriptors: Descriptors::new(),
            sift: SiftFeatures::default(),
            wide_descriptors: WideDescriptors {
                data: Vec::new(),
                dim: 0,
                count: 0,
            },
            strong_feature_indices: Vec::new(),
            colors: Vec::new(),
        }];
        let mut reconstruction = Reconstruction {
            camera: CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0),
            cameras: vec![CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0)],
            camera_ids: vec![1],
            rigs: Vec::new(),
            frames: Vec::new(),
            image_names: vec!["0.jpg".into()],
            image_paths: vec![PathBuf::from("0.jpg")],
            image_ids: vec![1],
            image_camera_indices: vec![0],
            image_frame_indices: vec![None],
            poses: vec![Some(SE3::identity())],
            observations: vec![vec![Some(0)]],
            keypoints: frames.iter().map(|f| f.keypoints.clone()).collect(),
            point_ids: vec![1],
            points: vec![Point3D {
                xyz: [0.0, 0.0, 2.0],
                color: [0, 0, 0],
                error: 0.0,
                track: vec![TrackObservation {
                    image: 0,
                    feature: 0,
                }],
            }],
        };

        let report = solve_bundle_adjustment_ceres(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 5,
                allow_single_observation_points: true,
                loss_function: BundleAdjustmentLoss::Trivial,
                ..BundleAdjustmentOptions::default()
            },
            None,
        )
        .expect("trivial loss should be explicit and Ceres-compatible");

        assert!(report.is_solution_usable());
        assert!(report.final_cost <= report.initial_cost + 1.0e-12);
    }

    #[test]
    fn ceres_termination_type_mapping_matches_colmap_summary_bridge() {
        assert_eq!(
            map_ceres_termination_type(TerminationType::Convergence),
            BundleAdjustmentTerminationType::Convergence
        );
        assert_eq!(
            map_ceres_termination_type(TerminationType::NoConvergence),
            BundleAdjustmentTerminationType::NoConvergence
        );
        assert_eq!(
            map_ceres_termination_type(TerminationType::Failure),
            BundleAdjustmentTerminationType::Failure
        );
        assert_eq!(
            map_ceres_termination_type(TerminationType::UserSuccess),
            BundleAdjustmentTerminationType::UserSuccess
        );
        assert_eq!(
            map_ceres_termination_type(TerminationType::UserFailure),
            BundleAdjustmentTerminationType::UserFailure
        );
        assert_eq!(
            map_ceres_termination_type(TerminationType::Unknown(99)),
            BundleAdjustmentTerminationType::Failure
        );
    }

    #[test]
    fn ceres_options_forward_max_linear_solver_iterations_to_validation() {
        assert!(ceres_solver_options(
            &BundleAdjustmentOptions {
                max_linear_solver_iterations: 0,
                ..BundleAdjustmentOptions::default()
            },
            1,
            2,
        )
        .is_some());

        assert!(ceres_solver_options(
            &BundleAdjustmentOptions {
                max_linear_solver_iterations: usize::MAX,
                ..BundleAdjustmentOptions::default()
            },
            1,
            2,
        )
        .is_none());
    }

    #[test]
    fn ceres_options_reject_usize_fields_before_i32_wraparound() {
        let Some(wraps_to_positive) = (u32::MAX as usize).checked_add(101) else {
            return;
        };
        assert_eq!(wraps_to_positive as i32, 100);

        assert!(ceres_solver_options(
            &BundleAdjustmentOptions {
                iterations: wraps_to_positive,
                ..BundleAdjustmentOptions::default()
            },
            1,
            2,
        )
        .is_none());
        assert!(ceres_solver_options(
            &BundleAdjustmentOptions {
                max_linear_solver_iterations: wraps_to_positive,
                ..BundleAdjustmentOptions::default()
            },
            1,
            2,
        )
        .is_none());
        assert!(ceres_solver_options(
            &BundleAdjustmentOptions {
                max_num_consecutive_invalid_steps: wraps_to_positive,
                ..BundleAdjustmentOptions::default()
            },
            1,
            2,
        )
        .is_none());
        assert!(ceres_solver_options(
            &BundleAdjustmentOptions {
                max_consecutive_nonmonotonic_steps: wraps_to_positive,
                ..BundleAdjustmentOptions::default()
            },
            1,
            2,
        )
        .is_none());
    }

    #[test]
    fn ceres_auto_solver_policy_covers_thresholds_and_sparse_availability() {
        for (poses, sparse, expected) in [
            (50, true, LinearSolverType::DENSE_SCHUR),
            (51, true, LinearSolverType::SPARSE_SCHUR),
            (1000, true, LinearSolverType::SPARSE_SCHUR),
            (1001, true, LinearSolverType::ITERATIVE_SCHUR),
            (51, false, LinearSolverType::ITERATIVE_SCHUR),
            (1000, false, LinearSolverType::ITERATIVE_SCHUR),
        ] {
            let policy = ceres_solver_policy_for_preference(
                BundleAdjustmentLinearSolverPreference::Auto,
                poses,
                sparse,
            );
            assert!(
                policy.linear_solver == expected,
                "poses={poses} sparse={sparse}"
            );
            assert!(
                policy.preconditioner
                    == (expected == LinearSolverType::ITERATIVE_SCHUR)
                        .then_some(PreconditionerType::SCHUR_JACOBI)
            );
        }
        assert!(ceres_solver_options(&BundleAdjustmentOptions::default(), 1001, 2).is_some());
    }

    #[test]
    fn ceres_solver_policy_honors_explicit_preferences() {
        use crate::ba::BundleAdjustmentLinearSolverPreference as Preference;

        let dense = ceres_solver_policy_for_preference(Preference::DenseSchur, 1001, true);
        assert!(dense.linear_solver == LinearSolverType::DENSE_SCHUR);
        assert!(dense.preconditioner.is_none());

        let sparse = ceres_solver_policy_for_preference(Preference::SparseSchur, 10, true);
        assert!(sparse.linear_solver == LinearSolverType::SPARSE_SCHUR);
        assert!(sparse.preconditioner.is_none());

        let iterative = ceres_solver_policy_for_preference(Preference::IterativeSchur, 10, true);
        assert!(iterative.linear_solver == LinearSolverType::ITERATIVE_SCHUR);
        assert!(iterative.preconditioner == Some(PreconditionerType::SCHUR_JACOBI));
    }

    #[test]
    fn ceres_sparse_backend_preference_maps_to_ceres() {
        use crate::ba::BundleAdjustmentSparseLinearAlgebra as Backend;

        assert!(
            ceres_sparse_backend_for_preference(Backend::SuiteSparse)
                == Some(SparseLinearAlgebraLibraryType::SUITE_SPARSE)
        );
        assert!(
            ceres_sparse_backend_for_preference(Backend::AccelerateSparse)
                == Some(SparseLinearAlgebraLibraryType::ACCELERATE_SPARSE)
        );
        assert!(
            ceres_sparse_backend_for_preference(Backend::EigenSparse)
                == Some(SparseLinearAlgebraLibraryType::EIGEN_SPARSE)
        );
        assert!(ceres_sparse_backend_for_preference(Backend::Auto).is_none());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn ceres_sparse_backend_uses_optimized_macos_build() {
        let backend = ceres_sparse_backend();

        assert!(matches!(
            backend,
            SparseLinearAlgebraLibraryType::SUITE_SPARSE
                | SparseLinearAlgebraLibraryType::ACCELERATE_SPARSE
        ));
    }

    #[test]
    fn ceres_options_match_colmap_threading_gate() {
        let options = BundleAdjustmentOptions {
            num_threads: 7,
            min_num_residuals_for_multi_threading: 100,
            ..BundleAdjustmentOptions::default()
        };

        assert_eq!(ceres_num_threads(&options, 99), 1);
        assert_eq!(ceres_num_threads(&options, 100), 7);
        assert!(
            ceres_num_threads(
                &BundleAdjustmentOptions {
                    num_threads: -1,
                    min_num_residuals_for_multi_threading: 0,
                    ..BundleAdjustmentOptions::default()
                },
                0,
            ) >= 1
        );
    }

    #[test]
    fn image_pose_ambient_jacobian_matches_numeric_ceres_block() {
        let camera = CameraModel::new_pinhole(200, 160, 90.0, 96.0, 100.0, 80.0);
        let pose = SE3::from_quat_translation(
            crate::geometry::quat_from_rotation_y(0.17),
            Vec3::new(0.2, -0.1, 0.05),
        );
        let point = [0.25, -0.1, 2.5];
        let pose_params = se3_to_pose_params(pose);
        let binding = ResidualBinding {
            xy: [70.0, 82.0],
            param_roles: vec![ParamRole::ImagePose, ParamRole::Point],
            pose_eval: PoseEval::Image { handle: 0 },
            camera_base: camera,
        };
        let analytic = analytic_image_pose_jacobian_ambient(camera, point, &pose_params).unwrap();
        let point_params = [point[0] as f64, point[1] as f64, point[2] as f64];
        let parameters = [pose_params.as_slice(), point_params.as_slice()];
        let numeric = numeric_jacobian_for_block(&parameters, &binding, 0);
        assert_mat2x7_close(analytic, numeric, 1.0e-3);
    }

    #[test]
    fn frame_pose_ambient_jacobian_matches_numeric_ceres_block() {
        let camera = CameraModel::new_pinhole(200, 160, 90.0, 96.0, 100.0, 80.0);
        let sensor_pose = SE3::from_quat_translation(
            crate::geometry::quat_from_rotation_x(-0.11),
            Vec3::new(0.15, 0.03, -0.02),
        );
        let rig_pose = SE3::from_quat_translation(
            crate::geometry::quat_from_rotation_y(0.17),
            Vec3::new(0.2, -0.1, 0.05),
        );
        let point = [0.25, -0.1, 2.5];
        let sensor_params = se3_to_pose_params(sensor_pose);
        let rig_params = se3_to_pose_params(rig_pose);
        let binding = ResidualBinding {
            xy: [70.0, 82.0],
            param_roles: vec![
                ParamRole::SensorPose,
                ParamRole::FramePose,
                ParamRole::Point,
            ],
            pose_eval: PoseEval::Frame {
                frame_handle: 1,
                sensor: FrameSensorEval::Variable { handle: 0 },
            },
            camera_base: camera,
        };
        let analytic =
            analytic_frame_pose_jacobian_ambient(camera, &sensor_params, &rig_params, point)
                .unwrap();
        let point_params = [point[0] as f64, point[1] as f64, point[2] as f64];
        let parameters = [
            sensor_params.as_slice(),
            rig_params.as_slice(),
            point_params.as_slice(),
        ];
        let numeric = numeric_jacobian_for_block(&parameters, &binding, 1);
        assert_mat2x7_close(analytic, numeric, 1.0e-3);
    }

    #[test]
    fn sensor_pose_ambient_jacobian_matches_numeric_ceres_block() {
        let camera = CameraModel::new_pinhole(200, 160, 90.0, 96.0, 100.0, 80.0);
        let sensor_pose = SE3::from_quat_translation(
            crate::geometry::quat_from_rotation_x(-0.11),
            Vec3::new(0.15, 0.03, -0.02),
        );
        let rig_pose = SE3::from_quat_translation(
            crate::geometry::quat_from_rotation_y(0.17),
            Vec3::new(0.2, -0.1, 0.05),
        );
        let point = [0.25, -0.1, 2.5];
        let sensor_params = se3_to_pose_params(sensor_pose);
        let rig_params = se3_to_pose_params(rig_pose);
        let binding = ResidualBinding {
            xy: [70.0, 82.0],
            param_roles: vec![
                ParamRole::SensorPose,
                ParamRole::FramePose,
                ParamRole::Point,
            ],
            pose_eval: PoseEval::Frame {
                frame_handle: 1,
                sensor: FrameSensorEval::Variable { handle: 0 },
            },
            camera_base: camera,
        };
        let analytic =
            analytic_sensor_pose_jacobian_ambient(camera, &sensor_params, &rig_params, point)
                .unwrap();
        let point_params = [point[0] as f64, point[1] as f64, point[2] as f64];
        let parameters = [
            sensor_params.as_slice(),
            rig_params.as_slice(),
            point_params.as_slice(),
        ];
        let numeric = numeric_jacobian_for_block(&parameters, &binding, 0);
        assert_mat2x7_close(analytic, numeric, 1.0e-3);
    }

    fn numeric_jacobian_for_block(
        parameters: &[&[f64]],
        binding: &ResidualBinding,
        block: usize,
    ) -> Mat2x7 {
        let mut numeric = Mat2x7::zeros();
        let eps = 1.0e-6;
        for col in 0..7 {
            let mut plus_params = parameters.iter().map(|p| p.to_vec()).collect::<Vec<_>>();
            plus_params[block][col] += eps;
            let plus_slices = plus_params.iter().map(|p| p.as_slice()).collect::<Vec<_>>();
            let plus = eval_residual(&plus_slices, binding).unwrap();
            let mut minus_params = parameters.iter().map(|p| p.to_vec()).collect::<Vec<_>>();
            minus_params[block][col] -= eps;
            let minus_slices = minus_params
                .iter()
                .map(|p| p.as_slice())
                .collect::<Vec<_>>();
            let minus = eval_residual(&minus_slices, binding).unwrap();
            numeric[(0, col)] = (plus[0] - minus[0]) / (2.0 * eps);
            numeric[(1, col)] = (plus[1] - minus[1]) / (2.0 * eps);
        }
        numeric
    }

    fn assert_mat2x7_close(analytic: Mat2x7, numeric: Mat2x7, tolerance: f64) {
        for row in 0..2 {
            for col in 0..7 {
                assert!(
                    (analytic[(row, col)] - numeric[(row, col)]).abs() < tolerance,
                    "row={row} col={col} analytic={} numeric={}",
                    analytic[(row, col)],
                    numeric[(row, col)]
                );
            }
        }
    }

    #[test]
    fn ceres_eigen_quaternion_manifold_matches_colmap_binding() {
        let target = [
            0.0,
            0.0,
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
        ];
        let cost: CostFunctionType = Box::new(move |parameters, residuals, mut jacobians| {
            for i in 0..4 {
                residuals[i] = parameters[0][i] - target[i];
            }
            if let Some(jacobians) = jacobians.as_mut() {
                if let Some(d_dq) = jacobians[0].as_mut() {
                    for r in 0..4 {
                        for c in 0..4 {
                            d_dq[r][c] = if r == c { 1.0 } else { 0.0 };
                        }
                    }
                }
            }
            true
        });

        let (mut problem, _) = NllsProblem::new()
            .residual_block_builder()
            .set_cost(cost, 4)
            .set_parameters([vec![0.0, 0.0, 0.0, 1.0]])
            .build_into_problem()
            .unwrap();
        problem.set_eigen_quaternion_manifold(0).unwrap();

        let solution = problem.solve(&SolverOptions::default()).unwrap();
        assert!(solution.summary.is_solution_usable());
        let q = &solution.parameters[0];
        let norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1.0e-12);
        for i in 0..4 {
            assert!((q[i] - target[i]).abs() < 1.0e-8);
        }
    }

    #[test]
    fn ceres_pose_manifold_supports_colmap_fixed_translation_axis() {
        let target = [
            0.0,
            0.0,
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
            1.0,
            3.0,
            3.0,
        ];
        let cost: CostFunctionType = Box::new(move |parameters, residuals, mut jacobians| {
            for i in 0..7 {
                residuals[i] = parameters[0][i] - target[i];
            }
            if let Some(jacobians) = jacobians.as_mut() {
                if let Some(d_dp) = jacobians[0].as_mut() {
                    for r in 0..7 {
                        for c in 0..7 {
                            d_dp[r][c] = if r == c { 1.0 } else { 0.0 };
                        }
                    }
                }
            }
            true
        });

        let (mut problem, _) = NllsProblem::new()
            .residual_block_builder()
            .set_cost(cost, 7)
            .set_parameters([vec![0.0, 0.0, 0.0, 1.0, -5.0, 2.0, -9.0]])
            .build_into_problem()
            .unwrap();
        problem.set_pose_manifold(0, &[1]).unwrap();

        let solver_options = SolverOptions::builder()
            .max_num_iterations(100)
            .function_tolerance(1.0e-12)
            .gradient_tolerance(1.0e-12)
            .parameter_tolerance(1.0e-12)
            .build()
            .unwrap();
        let solution = problem.solve(&solver_options).unwrap();
        assert!(solution.summary.is_solution_usable());
        let pose = &solution.parameters[0];
        let norm = pose[0..4].iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1.0e-12);
        assert!((pose[0] - target[0]).abs() < 1.0e-8);
        assert!((pose[1] - target[1]).abs() < 1.0e-8);
        assert!((pose[2] - target[2]).abs() < 1.0e-8);
        assert!((pose[3] - target[3]).abs() < 1.0e-8);
        assert!((pose[4] - target[4]).abs() < 1.0e-8);
        assert!((pose[5] - 2.0).abs() < 1.0e-12);
        assert!((pose[6] - target[6]).abs() < 1.0e-8);
    }

    #[test]
    fn ceres_pose_prior_cost_moves_camera_center() {
        let binding = PosePriorBinding {
            prior_position: [0.0, 0.0, 0.0],
            sqrt_information: Mat3::identity() * 100.0,
            param_roles: vec![ParamRole::ImagePose],
            pose_eval: PoseEval::Image { handle: 0 },
        };
        let cost = build_pose_prior_cost(binding);
        let (mut problem, _) = NllsProblem::new()
            .residual_block_builder()
            .set_cost(cost, 3)
            .set_parameters([se3_to_pose_params(SE3::from_quat_translation(
                Quat::identity(),
                Vec3::new(-4.0, 0.0, 0.0),
            ))
            .to_vec()])
            .build_into_problem()
            .unwrap();
        problem.set_pose_manifold(0, &[]).unwrap();

        let solution = problem.solve(&SolverOptions::default()).unwrap();
        assert!(solution.summary.is_solution_usable());
        let pose = pose_params_to_se3(&solution.parameters[0]);
        let center = camera_center_world(pose);
        assert!(center.norm() < 1.0e-3, "center={center:?}");
    }
}
