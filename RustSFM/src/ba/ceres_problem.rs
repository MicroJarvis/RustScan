use super::native::{
    analytic_frame_pose_jacobian, analytic_img_from_cam_jacobian, analytic_sensor_pose_jacobian,
    apply_two_cams_from_world_gauge, camera_by_index, camera_param_jacobian, camera_param_specs,
    count_variable_residuals, frame_sensor_from_rig, frame_sensor_key_for_image,
    projection_jacobians, sensor_pose_specs, set_frame_pose_block,
    sync_camera_intrinsics_from_params, sync_pose_blocks_for_sensor_changes, variable_pose_blocks,
    CameraParamSpec, PoseBlockKind, SensorPoseKey,
};
use super::shared::{
    add_three_point_gauge, bundle_adjustment_point_filter, collect_observations, project_point,
    refresh_point_errors,
};
use super::{
    BundleAdjustmentGauge, BundleAdjustmentLoss, BundleAdjustmentOptions, BundleAdjustmentReport,
    BundleAdjustmentTerminationReason, BundleAdjustmentTerminationType,
};
use crate::types::{CameraModel, ImageFrame, Reconstruction, Rigid3};
use ceres_solver::loss::LossFunction;
use ceres_solver::parameter_block::ParameterBlockOrIndex;
use ceres_solver::solver::{
    LinearSolverType, PreconditionerType, SolverOptions, SparseLinearAlgebraLibraryType,
    TerminationType,
};
use ceres_solver::{CostFunctionType, NllsProblem};
use glam::{Quat, Vec3};
use nalgebra::SMatrix;
use rustslam::SE3;
use std::collections::{HashMap, HashSet};

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

pub fn solve_bundle_adjustment_ceres(
    frames: &[ImageFrame],
    reconstruction: &mut Reconstruction,
    options: BundleAdjustmentOptions,
) -> Option<BundleAdjustmentReport> {
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
        block_values.insert(idx, vec![camera.params[spec.param]]);
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

    let solver_options = ceres_solver_options(&options, pose_entity_registry.len(), bindings * 2)?;
    let solution = problem.solve(&solver_options).ok()?;

    write_back_solution(
        reconstruction,
        &solution.parameters,
        &internal_to_storage,
        &pose_entity_registry,
        &frame_images,
        &sensor_pose_specs,
        &camera_param_registry,
        &camera_param_specs,
        &point_registry,
        &constant_point_filter,
        &pose_blocks,
    );

    refresh_point_errors(frames, reconstruction);

    let summary = solution.summary;
    let successful_steps = summary.num_successful_steps().max(0) as usize;
    let unsuccessful_steps = summary.num_unsuccessful_steps().max(0) as usize;
    let (termination_type, termination_reason, gradient_max_norm, step_norm, step_quality, damping) =
        map_ceres_summary(&summary);
    let residuals_reduced = summary.num_residuals_reduced().max(0) as usize;
    let effective_parameters_reduced = summary.num_effective_parameters_reduced().max(0) as usize;
    Some(BundleAdjustmentReport {
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
        termination_type,
        termination_reason,
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
    pose_blocks: &super::native::PoseBlockSet,
    sensor_lookup: &HashMap<SensorPoseKey, &super::native::SensorPoseSpec>,
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
    let Some(camera_idx) = super::native::camera_index_for_image(reconstruction, image) else {
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
                    camera.params[*param] = slice[0];
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
    sync_camera_intrinsics_from_params(&mut camera);
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
    let p_idx = binding
        .param_roles
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
    camera.img_from_cam(cam_point[0], cam_point[1], cam_point[2])
}

fn project_frame_pose_point(
    camera: CameraModel,
    sensor_pose_params: &[f64],
    rig_pose_params: &[f64],
    point: [f32; 3],
) -> Option<[f64; 2]> {
    let cam_point = frame_pose_cam_point(sensor_pose_params, rig_pose_params, point)?;
    camera.img_from_cam(cam_point[0], cam_point[1], cam_point[2])
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

fn write_back_solution(
    reconstruction: &mut Reconstruction,
    parameters: &[Vec<f64>],
    internal_to_storage: &HashMap<usize, usize>,
    pose_entity_registry: &HashMap<PoseEntityKey, usize>,
    frame_images: &HashMap<usize, Vec<usize>>,
    sensor_pose_specs: &[super::native::SensorPoseSpec],
    camera_param_registry: &HashMap<(usize, usize), usize>,
    camera_param_specs: &[CameraParamSpec],
    point_registry: &HashMap<usize, usize>,
    constant_point_filter: &HashSet<usize>,
    pose_blocks: &super::native::PoseBlockSet,
) {
    let mut changed_sensors = Vec::new();

    for (key, &handle) in pose_entity_registry {
        let Some(pose) = pose_params_from_solution(parameters, handle, internal_to_storage) else {
            continue;
        };
        match key {
            PoseEntityKey::Image(image) => {
                if let Some(slot) = reconstruction.poses.get_mut(*image) {
                    *slot = Some(pose);
                }
            }
            PoseEntityKey::Frame(frame_idx) => {
                if let Some(images) = frame_images.get(frame_idx) {
                    set_frame_pose_block(reconstruction, *frame_idx, images, pose);
                }
            }
            PoseEntityKey::Sensor(sensor_key) => {
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
                        changed_sensors.push(sensor_key.clone());
                    }
                }
            }
        }
    }

    if !changed_sensors.is_empty() {
        sync_pose_blocks_for_sensor_changes(reconstruction, pose_blocks, &changed_sensors);
    }

    let mut cameras = reconstruction.cameras.clone();
    if cameras.is_empty() {
        cameras.push(reconstruction.camera);
    }
    for &spec in camera_param_specs {
        let key = (spec.camera, spec.param);
        let Some(&idx) = camera_param_registry.get(&key) else {
            continue;
        };
        if spec.param >= cameras[spec.camera].num_params {
            continue;
        }
        let Some(params) = parameter_values(parameters, idx, internal_to_storage) else {
            continue;
        };
        cameras[spec.camera].params[spec.param] = params[0];
        sync_camera_intrinsics_from_params(&mut cameras[spec.camera]);
    }
    reconstruction.cameras = cameras.clone();
    if let Some(camera) = cameras.first() {
        reconstruction.camera = *camera;
    }

    for (&point_id, &idx) in point_registry {
        if constant_point_filter.contains(&point_id) {
            continue;
        }
        if let Some(point) = reconstruction.points.get_mut(point_id) {
            let Some(params) = parameter_values(parameters, idx, internal_to_storage) else {
                continue;
            };
            point.xyz = [
                params[0] as f32,
                params.get(1).copied().unwrap_or(0.0) as f32,
                params.get(2).copied().unwrap_or(0.0) as f32,
            ];
        }
    }

    let _ = sensor_pose_specs;
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
    Some(pose_params_to_se3(values))
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
) -> Option<SolverOptions> {
    let solver_policy = ceres_solver_policy(num_pose_entities, ceres_has_sparse_backend());
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
    builder.build().ok()
}

fn ceres_i32_option(value: usize) -> Option<i32> {
    i32::try_from(value).ok()
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct CeresSolverPolicy {
    linear_solver: LinearSolverType,
    preconditioner: Option<PreconditionerType>,
}

fn ceres_solver_policy(num_pose_entities: usize, has_sparse_backend: bool) -> CeresSolverPolicy {
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

fn ceres_has_sparse_backend() -> bool {
    SolverOptions::builder().current_sparse_linear_algebra_library_type()
        != SparseLinearAlgebraLibraryType::NO_SPARSE
}

fn ceres_num_threads(options: &BundleAdjustmentOptions, num_residuals: usize) -> i32 {
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
        Quat::from_xyzw(
            params[0] as f32,
            params[1] as f32,
            params[2] as f32,
            params[3] as f32,
        )
    } else {
        Quat::IDENTITY
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
        .map(|radius| 1.0 / radius)
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
    use super::*;
    use crate::sift::SiftFeatures;
    use crate::types::{CameraModel, ImageFrame, Point3D, TrackObservation};
    use crate::wide::WideDescriptors;
    use glam::{Quat, Vec3};
    use rustslam::Descriptors;
    use rustslam::KeyPoint;
    use rustslam::SE3;
    use std::path::PathBuf;

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
    fn ceres_options_match_colmap_solver_type_thresholds() {
        let policy = ceres_solver_policy(50, true);
        assert!(policy.linear_solver == LinearSolverType::DENSE_SCHUR);
        assert!(policy.preconditioner.is_none());

        let policy = ceres_solver_policy(51, true);
        assert!(policy.linear_solver == LinearSolverType::SPARSE_SCHUR);
        assert!(policy.preconditioner.is_none());

        let policy = ceres_solver_policy(1000, true);
        assert!(policy.linear_solver == LinearSolverType::SPARSE_SCHUR);
        assert!(policy.preconditioner.is_none());

        let policy = ceres_solver_policy(1001, true);
        assert!(policy.linear_solver == LinearSolverType::ITERATIVE_SCHUR);
        assert!(policy.preconditioner == Some(PreconditionerType::SCHUR_JACOBI));

        assert!(ceres_solver_options(&BundleAdjustmentOptions::default(), 1001, 2).is_some());
    }

    #[test]
    fn ceres_options_match_colmap_sparse_backend_gate() {
        let policy = ceres_solver_policy(51, false);
        assert!(policy.linear_solver == LinearSolverType::ITERATIVE_SCHUR);
        assert!(policy.preconditioner == Some(PreconditionerType::SCHUR_JACOBI));

        let policy = ceres_solver_policy(1000, false);
        assert!(policy.linear_solver == LinearSolverType::ITERATIVE_SCHUR);
        assert!(policy.preconditioner == Some(PreconditionerType::SCHUR_JACOBI));
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
            Quat::from_rotation_y(0.17).normalize(),
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
            Quat::from_rotation_x(-0.11).normalize(),
            Vec3::new(0.15, 0.03, -0.02),
        );
        let rig_pose = SE3::from_quat_translation(
            Quat::from_rotation_y(0.17).normalize(),
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
            Quat::from_rotation_x(-0.11).normalize(),
            Vec3::new(0.15, 0.03, -0.02),
        );
        let rig_pose = SE3::from_quat_translation(
            Quat::from_rotation_y(0.17).normalize(),
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
}
