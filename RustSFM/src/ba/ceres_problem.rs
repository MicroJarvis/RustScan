use super::native::{
    apply_two_cams_from_world_gauge, camera_by_index, camera_param_specs,
    count_variable_residuals, frame_sensor_from_rig, frame_sensor_key_for_image,
    point_effective_parameter_count, sensor_pose_specs, set_frame_pose_block,
    sync_camera_intrinsics_from_params, sync_pose_blocks_for_sensor_changes,
    variable_pose_blocks, CameraParamSpec, PoseBlockKind, SensorPoseKey,
};
use super::shared::{
    add_three_point_gauge, bundle_adjustment_point_filter, collect_observations,
    project_point, refresh_point_errors,
};
use super::{
    BundleAdjustmentGauge, BundleAdjustmentLoss, BundleAdjustmentOptions, BundleAdjustmentReport,
    BundleAdjustmentTerminationReason, BundleAdjustmentTerminationType,
};
use crate::types::{CameraModel, ImageFrame, Reconstruction, Rigid3};
use ceres_solver::loss::LossFunction;
use ceres_solver::parameter_block::ParameterBlockOrIndex;
use ceres_solver::solver::{LinearSolverType, SolverOptions};
use ceres_solver::{CostFunctionType, NllsProblem};
use glam::{Quat, Vec3};
use rustslam::SE3;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum PoseEntityKey {
    Image(usize),
    Frame(usize),
    Sensor(SensorPoseKey),
}

#[derive(Debug, Clone)]
enum PoseEval {
    Fixed(SE3),
    Image { handles: [usize; 6] },
    Frame {
        frame_handles: [usize; 6],
        sensor: FrameSensorEval,
    },
}

#[derive(Debug, Clone)]
enum FrameSensorEval {
    Fixed(SE3),
    Variable { handles: [usize; 6] },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParamRole {
    ImagePoseAxis(usize),
    FramePoseAxis(usize),
    SensorPoseAxis(usize),
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
    let mut pose_entity_registry = HashMap::<PoseEntityKey, [usize; 6]>::new();
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
                    se3_to_params(pose),
                    &mut pose_entity_registry,
                    &mut block_values,
                    &mut next_param_index,
                    &mut constant_blocks,
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
                    se3_to_params(frame.rig_from_world.to_se3()),
                    &mut pose_entity_registry,
                    &mut block_values,
                    &mut next_param_index,
                    &mut constant_blocks,
                );
            }
        }
    }

    for spec in &sensor_pose_specs {
        let Some(pose) =
            reconstruction.sensor_from_rig(spec.key.rig_id, &spec.key.sensor_id)
        else {
            continue;
        };
        register_pose_entity(
            PoseEntityKey::Sensor(spec.key.clone()),
            [true; 6],
            se3_to_params(pose),
            &mut pose_entity_registry,
            &mut block_values,
            &mut next_param_index,
            &mut constant_blocks,
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
        if let Some(loss) = ceres_loss(options.loss_function) {
            builder = builder.set_loss(loss);
        }
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

    for internal_idx in constant_blocks {
        let Some(&storage_idx) = internal_to_storage.get(&internal_idx) else {
            continue;
        };
        problem.set_parameter_block_constant(storage_idx).ok()?;
    }

    let solution = problem.solve(&ceres_solver_options(
        &options,
        pose_entity_registry.len(),
    )).ok()?;

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
        residuals,
        effective_parameters: pose_blocks.dim
            + sensor_pose_specs.len() * 6
            + camera_param_specs.len()
            + point_effective_parameter_count(&observations, &constant_point_filter),
        gradient_max_norm: f64::NAN,
        step_norm: f64::NAN,
        step_quality: f64::NAN,
        damping: f64::NAN,
        termination_type: if summary.is_solution_usable() {
            BundleAdjustmentTerminationType::Convergence
        } else {
            BundleAdjustmentTerminationType::Failure
        },
        termination_reason: BundleAdjustmentTerminationReason::GradientTolerance,
    })
}

fn register_pose_entity(
    key: PoseEntityKey,
    free_axes: [bool; 6],
    values: [f64; 6],
    registry: &mut HashMap<PoseEntityKey, [usize; 6]>,
    block_values: &mut HashMap<usize, Vec<f64>>,
    next_param_index: &mut usize,
    constant_blocks: &mut HashSet<usize>,
) -> [usize; 6] {
    if let Some(handles) = registry.get(&key).copied() {
        return handles;
    }
    let mut handles = [0usize; 6];
    for axis in 0..6 {
        let idx = *next_param_index;
        *next_param_index += 1;
        handles[axis] = idx;
        block_values.insert(idx, vec![values[axis]]);
        if !free_axes[axis] {
            constant_blocks.insert(idx);
        }
    }
    registry.insert(key, handles);
    handles
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
    block_values.insert(
        idx,
        vec![xyz[0] as f64, xyz[1] as f64, xyz[2] as f64],
    );
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
    pose_entity_registry: &HashMap<PoseEntityKey, [usize; 6]>,
) -> Option<PoseEval> {
    let Some(block_idx) = pose_blocks.image_to_block.get(image).copied().flatten() else {
        let pose = reconstruction.poses.get(image).copied().flatten()?;
        return Some(PoseEval::Fixed(pose));
    };
    let block = pose_blocks.blocks.get(block_idx)?;
    match block.kind {
        PoseBlockKind::Image(image) => Some(PoseEval::Image {
            handles: *pose_entity_registry.get(&PoseEntityKey::Image(image))?,
        }),
        PoseBlockKind::Frame(frame_idx) => {
            let frame_handles = *pose_entity_registry.get(&PoseEntityKey::Frame(frame_idx))?;
            if let Some(key) = frame_sensor_key_for_image(reconstruction, image) {
                if sensor_lookup.contains_key(&key) {
                    Some(PoseEval::Frame {
                        frame_handles,
                        sensor: FrameSensorEval::Variable {
                            handles: *pose_entity_registry.get(&PoseEntityKey::Sensor(key))?,
                        },
                    })
                } else {
                    let fixed = frame_sensor_from_rig(reconstruction, frame_idx, image)?;
                    Some(PoseEval::Frame {
                        frame_handles,
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
        PoseEval::Image { handles } => {
            for axis in 0..6 {
                param_indices.push(handles[axis]);
                param_roles.push(ParamRole::ImagePoseAxis(axis));
            }
        }
        PoseEval::Frame {
            frame_handles,
            sensor,
        } => {
            for axis in 0..6 {
                param_indices.push(frame_handles[axis]);
                param_roles.push(ParamRole::FramePoseAxis(axis));
            }
            if let FrameSensorEval::Variable { handles } = sensor {
                for axis in 0..6 {
                    param_indices.push(handles[axis]);
                    param_roles.push(ParamRole::SensorPoseAxis(axis));
                }
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
    Box::new(move |parameters: &[&[f64]], residuals: &mut [f64], jacobians| {
        let Some(residual) = eval_residual(parameters, &binding) else {
            residuals[0] = 0.0;
            residuals[1] = 0.0;
            return jacobians.is_none();
        };
        residuals.copy_from_slice(&residual);
        if let Some(jacobians) = jacobians {
            return fill_numeric_jacobians(parameters, &binding, jacobians);
        }
        true
    })
}

fn eval_residual(parameters: &[&[f64]], binding: &ResidualBinding) -> Option<[f64; 2]> {
    let (pose, point, camera) = assemble_state(parameters, binding)?;
    let predicted = project_point(camera, pose, point)?;
    Some([predicted[0] - binding.xy[0], predicted[1] - binding.xy[1]])
}

fn assemble_state(
    parameters: &[&[f64]],
    binding: &ResidualBinding,
) -> Option<(SE3, [f32; 3], CameraModel)> {
    let mut image_pose = [0.0; 6];
    let mut frame_pose = [0.0; 6];
    let mut sensor_pose = [0.0; 6];
    let mut point = [0.0f64; 3];
    let mut has_point = false;
    let mut camera = binding.camera_base;

    for (p_idx, role) in binding.param_roles.iter().enumerate() {
        let slice = parameters.get(p_idx)?;
        match role {
            ParamRole::ImagePoseAxis(axis) => image_pose[*axis] = slice[0],
            ParamRole::FramePoseAxis(axis) => frame_pose[*axis] = slice[0],
            ParamRole::SensorPoseAxis(axis) => sensor_pose[*axis] = slice[0],
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
        PoseEval::Fixed(pose) => *pose,
        PoseEval::Image { .. } => params_to_se3(&image_pose),
        PoseEval::Frame { sensor, .. } => {
            let rig = params_to_se3(&frame_pose);
            match sensor {
                FrameSensorEval::Fixed(pose) => pose.compose(&rig),
                FrameSensorEval::Variable { .. } => params_to_se3(&sensor_pose).compose(&rig),
            }
        }
    };
    sync_camera_intrinsics_from_params(&mut camera);
    Some((
        pose,
        [point[0] as f32, point[1] as f32, point[2] as f32],
        camera,
    ))
}

fn fill_numeric_jacobians(
    parameters: &[&[f64]],
    binding: &ResidualBinding,
    jacobians: &mut [Option<&mut [&mut [f64]]>],
) -> bool {
    const EPS: f64 = 1.0e-8;
    let mut params_storage = parameters
        .iter()
        .map(|p| p.to_vec())
        .collect::<Vec<_>>();

    for (p_idx, jac_opt) in jacobians.iter_mut().enumerate() {
        let Some(jac) = jac_opt else {
            continue;
        };
        let param_len = params_storage[p_idx].len();
        for k in 0..param_len {
            params_storage[p_idx][k] += EPS;
            let plus = eval_residual_from_storage(&params_storage, binding, p_idx);
            params_storage[p_idx][k] -= 2.0 * EPS;
            let minus = eval_residual_from_storage(&params_storage, binding, p_idx);
            params_storage[p_idx][k] += EPS;
            let (Some(plus), Some(minus)) = (plus, minus) else {
                return false;
            };
            for r in 0..2 {
                jac[r][k] = (plus[r] - minus[r]) / (2.0 * EPS);
            }
        }
    }
    true
}

fn eval_residual_from_storage(
    params_storage: &[Vec<f64>],
    binding: &ResidualBinding,
    _perturbed_block: usize,
) -> Option<[f64; 2]> {
    let params = params_storage.iter().map(|p| p.as_slice()).collect::<Vec<_>>();
    eval_residual(&params, binding)
}

fn write_back_solution(
    reconstruction: &mut Reconstruction,
    parameters: &[Vec<f64>],
    internal_to_storage: &HashMap<usize, usize>,
    pose_entity_registry: &HashMap<PoseEntityKey, [usize; 6]>,
    frame_images: &HashMap<usize, Vec<usize>>,
    sensor_pose_specs: &[super::native::SensorPoseSpec],
    camera_param_registry: &HashMap<(usize, usize), usize>,
    camera_param_specs: &[CameraParamSpec],
    point_registry: &HashMap<usize, usize>,
    constant_point_filter: &HashSet<usize>,
    pose_blocks: &super::native::PoseBlockSet,
) {
    let mut changed_sensors = Vec::new();

    for (key, handles) in pose_entity_registry {
        let Some(pose) = params_from_solution(parameters, handles, internal_to_storage) else {
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

fn params_from_solution(
    parameters: &[Vec<f64>],
    handles: &[usize; 6],
    internal_to_storage: &HashMap<usize, usize>,
) -> Option<SE3> {
    let mut values = [0.0; 6];
    for (axis, &internal_idx) in handles.iter().enumerate() {
        values[axis] = parameter_values(parameters, internal_idx, internal_to_storage)?[0];
    }
    Some(params_to_se3(&values))
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

fn ceres_loss(loss: BundleAdjustmentLoss) -> Option<LossFunction> {
    match loss {
        BundleAdjustmentLoss::Trivial => None,
        BundleAdjustmentLoss::Huber { scale } => Some(LossFunction::huber(scale)),
        BundleAdjustmentLoss::SoftL1 { scale } => Some(LossFunction::soft_l1(scale)),
        BundleAdjustmentLoss::Cauchy { scale } => Some(LossFunction::cauchy(scale)),
    }
}

fn ceres_solver_options(options: &BundleAdjustmentOptions, num_pose_entities: usize) -> SolverOptions {
    let linear_solver = if num_pose_entities >= 50 {
        LinearSolverType::SPARSE_SCHUR
    } else {
        LinearSolverType::DENSE_SCHUR
    };
    SolverOptions::builder()
        .max_num_iterations(options.iterations as i32)
        .function_tolerance(options.function_tolerance)
        .gradient_tolerance(options.gradient_tolerance)
        .parameter_tolerance(options.parameter_tolerance)
        .max_num_consecutive_invalid_steps(options.max_num_consecutive_invalid_steps as i32)
        .max_consecutive_nonmonotonic_steps(options.max_consecutive_nonmonotonic_steps as i32)
        .linear_solver_type(linear_solver)
        .build()
        .unwrap_or_default()
}

pub(crate) fn se3_to_params(pose: SE3) -> [f64; 6] {
    let q = pose.quaternion();
    let quat = Quat::from_xyzw(q[0], q[1], q[2], q[3]).normalize();
    let (axis, angle) = quat.to_axis_angle();
    let aa = axis * angle;
    let t = pose.translation();
    [
        aa.x as f64,
        aa.y as f64,
        aa.z as f64,
        t[0] as f64,
        t[1] as f64,
        t[2] as f64,
    ]
}

fn params_to_se3(params: &[f64]) -> SE3 {
    let omega = Vec3::new(params[0] as f32, params[1] as f32, params[2] as f32);
    let angle = omega.length();
    let rotation = if angle > 1.0e-12 {
        Quat::from_axis_angle(omega / angle, angle)
    } else {
        Quat::IDENTITY
    };
    SE3::from_quat_translation(
        rotation,
        Vec3::new(params[3] as f32, params[4] as f32, params[5] as f32),
    )
}
