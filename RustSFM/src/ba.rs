use crate::types::{
    colmap_camera_model_focal_idxs, colmap_camera_model_principal_point_idxs, CameraModel,
    ImageFrame, Reconstruction, COLMAP_FULL_OPENCV, COLMAP_OPENCV, COLMAP_PINHOLE, COLMAP_RADIAL,
    COLMAP_SIMPLE_PINHOLE, COLMAP_SIMPLE_RADIAL,
};
use glam::{Quat, Vec3};
use nalgebra::{DMatrix, DVector, SMatrix, SVector};
use rustslam::SE3;
use std::collections::HashSet;

type Mat2x3 = SMatrix<f64, 2, 3>;
type Mat2x6 = SMatrix<f64, 2, 6>;
type Mat3 = SMatrix<f64, 3, 3>;
type Vec2 = SVector<f64, 2>;
type Vec3d = SVector<f64, 3>;
type Vec6 = SVector<f64, 6>;

#[derive(Debug, Clone)]
pub struct BundleAdjustmentOptions {
    pub iterations: usize,
    pub function_tolerance: f64,
    pub gradient_tolerance: f64,
    pub parameter_tolerance: f64,
    pub max_linear_solver_iterations: usize,
    pub max_num_consecutive_invalid_steps: usize,
    pub max_consecutive_nonmonotonic_steps: usize,
    pub huber_delta_px: f64,
    pub max_observation_error_px: f64,
    pub variable_images: Option<Vec<usize>>,
    pub constant_images: Vec<usize>,
    pub variable_cameras: Option<Vec<usize>>,
    pub constant_cameras: Vec<usize>,
    pub refine_focal_length: bool,
    pub refine_principal_point: bool,
    pub refine_extra_params: bool,
    pub point_ids: Option<Vec<usize>>,
    pub constant_point_ids: Option<Vec<usize>>,
}

impl Default for BundleAdjustmentOptions {
    fn default() -> Self {
        Self {
            iterations: 100,
            function_tolerance: 0.0,
            gradient_tolerance: 1.0e-4,
            parameter_tolerance: 0.0,
            max_linear_solver_iterations: 200,
            max_num_consecutive_invalid_steps: 10,
            max_consecutive_nonmonotonic_steps: 10,
            huber_delta_px: 4.0,
            max_observation_error_px: 16.0,
            variable_images: None,
            constant_images: Vec::new(),
            variable_cameras: None,
            constant_cameras: Vec::new(),
            refine_focal_length: false,
            refine_principal_point: false,
            refine_extra_params: false,
            point_ids: None,
            constant_point_ids: None,
        }
    }
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
    MaxIterations,
    LinearizationFailure,
    LinearSolveFailure,
    InvalidStep,
    NoAcceptedStep,
    MaxConsecutiveInvalidSteps,
    MaxConsecutiveNonmonotonicSteps,
}

#[derive(Debug, Clone, Copy)]
pub struct BundleAdjustmentReport {
    pub iterations: usize,
    pub attempted_iterations: usize,
    pub successful_steps: usize,
    pub unsuccessful_steps: usize,
    pub linearization_failures: usize,
    pub linear_solve_failures: usize,
    pub invalid_steps: usize,
    pub rejected_steps: usize,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub observations: usize,
    pub residuals: usize,
    pub termination_type: BundleAdjustmentTerminationType,
    pub termination_reason: BundleAdjustmentTerminationReason,
}

impl BundleAdjustmentReport {
    pub fn is_solution_usable(&self) -> bool {
        self.termination_type.is_solution_usable()
    }

    pub fn brief_report(&self) -> String {
        format!(
            "termination={:?} reason={:?} residuals={} iterations={}/{} cost={:.6}->{:.6}",
            self.termination_type,
            self.termination_reason,
            self.residuals,
            self.iterations,
            self.attempted_iterations,
            self.initial_cost,
            self.final_cost
        )
    }
}

pub fn refine_bundle_adjustment(
    frames: &[ImageFrame],
    reconstruction: &mut Reconstruction,
    options: BundleAdjustmentOptions,
) -> Option<BundleAdjustmentReport> {
    let point_filter = bundle_adjustment_point_filter(
        options.point_ids.as_deref(),
        options.constant_point_ids.as_deref(),
    );
    let constant_point_filter = options
        .constant_point_ids
        .as_ref()
        .map(|ids| ids.iter().copied().collect::<HashSet<_>>())
        .unwrap_or_default();
    let pose_indices = variable_camera_indices(
        reconstruction,
        options.variable_images.as_deref(),
        &options.constant_images,
    );
    if reconstruction.points.is_empty() {
        return None;
    }
    let observations = collect_observations(
        frames,
        reconstruction,
        options.max_observation_error_px,
        point_filter.as_ref(),
    );
    let camera_param_specs = camera_param_specs(
        reconstruction,
        &observations,
        &options,
        pose_indices.len() * 6,
    );
    let nonpoint_dim = pose_indices.len() * 6 + camera_param_specs.len();
    if nonpoint_dim == 0 || observations.len() * 2 < nonpoint_dim {
        return None;
    }
    let residuals = count_variable_residuals(
        reconstruction,
        &observations,
        &pose_indices,
        &camera_param_specs,
        &constant_point_filter,
    );
    if residuals == 0 {
        return None;
    }

    let initial_cost = total_cost(reconstruction, &observations, options.huber_delta_px);
    if !initial_cost.is_finite() {
        return None;
    }
    let mut final_cost = initial_cost;
    let mut completed = 0usize;
    let mut attempted = 0usize;
    let mut unsuccessful_steps = 0usize;
    let mut linearization_failures = 0usize;
    let mut linear_solve_failures = 0usize;
    let mut invalid_steps = 0usize;
    let mut rejected_steps = 0usize;
    let mut consecutive_invalid_steps = 0usize;
    let mut consecutive_nonmonotonic_steps = 0usize;
    let mut termination_type = BundleAdjustmentTerminationType::NoConvergence;
    let mut termination_reason = BundleAdjustmentTerminationReason::MaxIterations;
    let mut damping = 1.0e-3;

    for _ in 0..options.iterations {
        attempted += 1;
        let Some(system) = build_schur_system(
            reconstruction,
            &observations,
            &pose_indices,
            &camera_param_specs,
            &constant_point_filter,
            options.huber_delta_px,
            damping,
        ) else {
            linearization_failures += 1;
            unsuccessful_steps += 1;
            termination_type = BundleAdjustmentTerminationType::Failure;
            termination_reason = BundleAdjustmentTerminationReason::LinearizationFailure;
            break;
        };
        let Some(delta) = system.h.lu().solve(&(-system.g)) else {
            linear_solve_failures += 1;
            unsuccessful_steps += 1;
            consecutive_invalid_steps += 1;
            consecutive_nonmonotonic_steps = 0;
            damping *= 10.0;
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
        if !delta.iter().all(|v| v.is_finite()) || delta_norm > 20.0 {
            invalid_steps += 1;
            unsuccessful_steps += 1;
            consecutive_invalid_steps += 1;
            consecutive_nonmonotonic_steps = 0;
            damping *= 10.0;
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
        if delta_norm <= options.gradient_tolerance.max(options.parameter_tolerance) {
            termination_type = BundleAdjustmentTerminationType::Convergence;
            termination_reason = BundleAdjustmentTerminationReason::GradientTolerance;
            break;
        }

        let base_poses = reconstruction.poses.clone();
        let base_points = reconstruction
            .points
            .iter()
            .map(|p| p.xyz)
            .collect::<Vec<_>>();
        let base_camera = reconstruction.camera;
        let base_cameras = reconstruction.cameras.clone();
        let mut accepted = false;
        for step in [1.0, 0.5, 0.25, 0.125, 0.0625] {
            apply_schur_delta(
                reconstruction,
                &observations,
                &pose_indices,
                &camera_param_specs,
                &system.point_blocks,
                &delta,
                step,
            );
            let candidate_cost = total_cost(reconstruction, &observations, options.huber_delta_px);
            if candidate_cost.is_finite() && candidate_cost + 1.0e-8 < final_cost {
                let previous_cost = final_cost;
                final_cost = candidate_cost;
                damping = (damping * 0.5).max(1.0e-8);
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
            damping *= 4.0;
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
        linearization_failures,
        linear_solve_failures,
        invalid_steps,
        rejected_steps,
        initial_cost,
        final_cost,
        observations: observations.len(),
        residuals,
        termination_type,
        termination_reason,
    })
}

fn bundle_adjustment_point_filter(
    variable_points: Option<&[usize]>,
    constant_points: Option<&[usize]>,
) -> Option<HashSet<usize>> {
    match (variable_points, constant_points) {
        (None, None) => None,
        (variable_points, constant_points) => {
            let mut filter = HashSet::new();
            if let Some(points) = variable_points {
                filter.extend(points.iter().copied());
            }
            if let Some(points) = constant_points {
                filter.extend(points.iter().copied());
            }
            Some(filter)
        }
    }
}

fn count_variable_residuals(
    reconstruction: &Reconstruction,
    observations: &[BaObservation],
    pose_indices: &[usize],
    camera_param_specs: &[CameraParamSpec],
    constant_point_filter: &HashSet<usize>,
) -> usize {
    let variable_poses = pose_indices.iter().copied().collect::<HashSet<_>>();
    let variable_cameras = camera_param_specs
        .iter()
        .map(|spec| spec.camera)
        .collect::<HashSet<_>>();
    let variable_observations = observations
        .iter()
        .filter(|obs| {
            !constant_point_filter.contains(&obs.point)
                || variable_poses.contains(&obs.image)
                || camera_index_for_image(reconstruction, obs.image)
                    .is_some_and(|camera| variable_cameras.contains(&camera))
        })
        .count();
    variable_observations * 2
}

#[derive(Debug, Clone, Copy)]
struct BaObservation {
    image: usize,
    point: usize,
    xy: [f64; 2],
}

struct SchurSystem {
    h: DMatrix<f64>,
    g: DVector<f64>,
    point_blocks: Vec<PointBlock>,
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

#[derive(Debug, Clone, Copy)]
struct CameraParamSpec {
    camera: usize,
    param: usize,
    offset: usize,
}

fn variable_camera_indices(
    reconstruction: &Reconstruction,
    variable_images: Option<&[usize]>,
    constant_images: &[usize],
) -> Vec<usize> {
    let constant_images = constant_images.iter().copied().collect::<HashSet<_>>();
    if let Some(images) = variable_images {
        let mut unique = images
            .iter()
            .copied()
            .filter(|&idx| idx < reconstruction.poses.len() && reconstruction.poses[idx].is_some())
            .filter(|idx| !constant_images.contains(idx))
            .collect::<Vec<_>>();
        unique.sort_unstable();
        unique.dedup();
        unique
    } else {
        let has_explicit_gauge = !constant_images.is_empty();
        reconstruction
            .poses
            .iter()
            .enumerate()
            .filter_map(|(idx, pose)| {
                (pose.is_some() && (has_explicit_gauge || idx > 0)).then_some(idx)
            })
            .filter(|idx| !constant_images.contains(idx))
            .collect()
    }
}

fn collect_observations(
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
    max_error_px: f64,
    point_filter: Option<&HashSet<usize>>,
) -> Vec<BaObservation> {
    let mut observations = Vec::new();
    for (point_id, point) in reconstruction.points.iter().enumerate() {
        if point_filter.is_some_and(|filter| !filter.contains(&point_id)) {
            continue;
        }
        if point.track.len() < 2 {
            continue;
        }
        for obs in &point.track {
            if obs.image >= reconstruction.poses.len()
                || obs.image >= frames.len()
                || reconstruction.poses[obs.image].is_none()
                || obs.feature >= frames[obs.image].keypoints.len()
            {
                continue;
            }
            let kp = &frames[obs.image].keypoints[obs.feature];
            let Some(pose) = reconstruction.poses[obs.image] else {
                continue;
            };
            let Some(predicted) =
                project_point(reconstruction.camera_for_image(obs.image), pose, point.xyz)
            else {
                continue;
            };
            let err = ((predicted[0] - kp.x() as f64).powi(2)
                + (predicted[1] - kp.y() as f64).powi(2))
            .sqrt();
            if err.is_finite() && err <= max_error_px {
                observations.push(BaObservation {
                    image: obs.image,
                    point: point_id,
                    xy: [kp.x() as f64, kp.y() as f64],
                });
            }
        }
    }
    observations
}

fn camera_param_specs(
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

fn camera_index_for_image(reconstruction: &Reconstruction, image: usize) -> Option<usize> {
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

fn camera_by_index(reconstruction: &Reconstruction, camera_idx: usize) -> Option<CameraModel> {
    reconstruction
        .cameras
        .get(camera_idx)
        .copied()
        .or_else(|| (camera_idx == 0).then_some(reconstruction.camera))
}

fn build_schur_system(
    reconstruction: &Reconstruction,
    observations: &[BaObservation],
    pose_indices: &[usize],
    camera_param_specs: &[CameraParamSpec],
    constant_point_filter: &HashSet<usize>,
    huber_delta_px: f64,
    damping: f64,
) -> Option<SchurSystem> {
    let mut pose_lookup = vec![None; reconstruction.poses.len()];
    for (var_idx, &image) in pose_indices.iter().enumerate() {
        pose_lookup[image] = Some(var_idx * 6);
    }
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

    let nonpoint_dim = pose_indices.len() * 6 + camera_param_specs.len();
    let mut h_cc = DMatrix::<f64>::zeros(nonpoint_dim, nonpoint_dim);
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
        let weight = huber_weight(err, huber_delta_px);
        let sqrt_w = weight.sqrt();
        let residual = residual * sqrt_w;
        let j_pose = j_pose * sqrt_w;
        let j_point = j_point * sqrt_w;

        let mut nonpoint_jacobians = Vec::new();
        if let Some(offset) = pose_lookup[obs.image] {
            nonpoint_jacobians.push((offset, mat2x6_to_dmatrix(j_pose)));
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
                        h_cc[(*offset_i + r, *offset_j + c)] += h[(r, c)];
                    }
                }
            }
        }
    }

    for idx in 0..nonpoint_dim {
        h_cc[(idx, idx)] += damping;
    }

    for block in &mut point_blocks {
        if block.nonpoint_blocks.is_empty() {
            continue;
        }
        for d in 0..3 {
            block.h_inv[(d, d)] += damping;
        }
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
                        h_cc[(e_i.offset + r, e_j.offset + c)] -= schur_h[(r, c)];
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
    pose_indices: &[usize],
    camera_param_specs: &[CameraParamSpec],
    point_blocks: &[PointBlock],
    nonpoint_delta: &DVector<f64>,
    step: f64,
) {
    for (var_idx, &image) in pose_indices.iter().enumerate() {
        let delta = Vec6::from_iterator((0..6).map(|k| nonpoint_delta[var_idx * 6 + k] * step));
        if let Some(pose) = reconstruction.poses[image] {
            reconstruction.poses[image] = Some(apply_pose_delta_f64(pose, delta));
        }
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

fn analytic_img_from_cam_jacobian(camera: CameraModel, x: f64, y: f64, z: f64) -> Option<Mat2x3> {
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

fn camera_param_jacobian(
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

fn vec2_to_dmatrix(vector: Vec2) -> DMatrix<f64> {
    DMatrix::from_column_slice(2, 1, &[vector[0], vector[1]])
}

fn point_nonpoint_cross(j_point: Mat2x3, j_nonpoint: &DMatrix<f64>) -> DMatrix<f64> {
    DMatrix::from_fn(3, j_nonpoint.ncols(), |row, col| {
        j_point[(0, row)] * j_nonpoint[(0, col)] + j_point[(1, row)] * j_nonpoint[(1, col)]
    })
}

fn apply_camera_param_delta(
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

fn sync_camera_intrinsics_from_params(camera: &mut CameraModel) {
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

fn project_point(camera: CameraModel, pose: SE3, point: [f32; 3]) -> Option<[f64; 2]> {
    let p = pose.transform_point(&point);
    camera.img_from_cam(p[0] as f64, p[1] as f64, p[2] as f64)
}

fn apply_pose_delta_f64(pose: SE3, delta: Vec6) -> SE3 {
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
    huber_delta_px: f64,
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
        let err = ((predicted[0] - obs.xy[0]).powi(2) + (predicted[1] - obs.xy[1]).powi(2)).sqrt();
        if err.is_finite() {
            total += huber_cost(err, huber_delta_px);
            count += 1;
        }
    }
    if count == 0 {
        f64::INFINITY
    } else {
        total / count as f64
    }
}

fn huber_weight(err: f64, delta: f64) -> f64 {
    if err <= delta {
        1.0
    } else {
        delta / err.max(1.0e-12)
    }
}

fn huber_cost(err: f64, delta: f64) -> f64 {
    if err <= delta {
        0.5 * err * err
    } else {
        delta * (err - 0.5 * delta)
    }
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
    points: &[[f32; 3]],
    camera: CameraModel,
    cameras: &[CameraModel],
) {
    reconstruction.poses.clone_from_slice(poses);
    for (point, xyz) in reconstruction.points.iter_mut().zip(points.iter()) {
        point.xyz = *xyz;
    }
    reconstruction.camera = camera;
    reconstruction.cameras.clone_from_slice(cameras);
}

fn refresh_point_errors(frames: &[ImageFrame], reconstruction: &mut Reconstruction) {
    let image_cameras = (0..reconstruction.poses.len())
        .map(|image| reconstruction.camera_for_image(image))
        .collect::<Vec<_>>();
    for point in &mut reconstruction.points {
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
            if let Some(predicted) = project_point(image_cameras[obs.image], pose, point.xyz) {
                let err = ((predicted[0] - kp.x() as f64).powi(2)
                    + (predicted[1] - kp.y() as f64).powi(2))
                .sqrt();
                if err.is_finite() {
                    total += err as f32;
                    count += 1;
                }
            }
        }
        if count > 0 {
            point.error = total / count as f32;
        }
    }
}

#[allow(dead_code)]
fn pose_from_parts(rotation: Quat, translation: Vec3) -> SE3 {
    SE3::from_quat_translation(rotation.normalize(), translation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CameraModel, ImageFrame, Point3D, Reconstruction, TrackObservation};
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
        assert_eq!(options.max_num_consecutive_invalid_steps, 10);
        assert_eq!(options.max_consecutive_nonmonotonic_steps, 10);
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
                huber_delta_px: 4.0,
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

        let variable = variable_camera_indices(&reconstruction, None, &[2]);

        assert_eq!(variable, vec![0, 1]);
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

        let report = refine_bundle_adjustment(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 3,
                huber_delta_px: 4.0,
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

        let report = refine_bundle_adjustment(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 8,
                huber_delta_px: 4.0,
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

        let report = refine_bundle_adjustment(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 8,
                function_tolerance: 1.0,
                gradient_tolerance: 0.0,
                huber_delta_px: 4.0,
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

        let report = refine_bundle_adjustment(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 0,
                huber_delta_px: 4.0,
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
            image_names: frames.iter().map(|frame| frame.name.clone()).collect(),
            image_paths: frames.iter().map(|frame| frame.path.clone()).collect(),
            image_ids: (0..frames.len()).map(|idx| idx as u32 + 1).collect(),
            image_camera_indices: vec![0; frames.len()],
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
