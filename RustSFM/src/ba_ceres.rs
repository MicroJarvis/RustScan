//! Ceres-backed bundle adjustment (`feature = "ceres-ba"`, enabled by default).
//!
//! Handles image-level poses with fixed intrinsics. Unsupported configurations
//! (rig/frame pose blocks, sensor extrinsics refinement, camera intrinsics
//! refinement, and `TWO_CAMS_FROM_WORLD` gauge partial axes) return `None`.

use crate::ba::{
    add_three_point_gauge, bundle_adjustment_point_filter, collect_observations,
    refresh_point_errors, BaObservation, BundleAdjustmentGauge, BundleAdjustmentLoss,
    BundleAdjustmentOptions, BundleAdjustmentReport, BundleAdjustmentTerminationReason,
    BundleAdjustmentTerminationType,
};
use crate::types::{CameraModel, ImageFrame, Reconstruction};
use ceres_solver::loss::LossFunction;
use ceres_solver::parameter_block::ParameterBlockOrIndex;
use ceres_solver::solver::{LinearSolverType, SolverOptions};
use ceres_solver::{CostFunctionType, NllsProblem};
use glam::{Quat, Vec3};
use rustslam::SE3;
use std::collections::{HashMap, HashSet};

/// Returns true when the Ceres backend can handle this problem configuration.
pub fn supports_ceres_ba(reconstruction: &Reconstruction, options: &BundleAdjustmentOptions) -> bool {
    if options.refine_focal_length
        || options.refine_principal_point
        || options.refine_extra_params
    {
        return false;
    }
    if matches!(options.gauge, BundleAdjustmentGauge::TwoCamsFromWorld) {
        return false;
    }
    if !options.constant_rigs.is_empty() || !options.constant_sensor_from_rig.is_empty() {
        return false;
    }
    if reconstruction
        .frames
        .iter()
        .any(|frame| frame.rig_id != 0 || !frame.data_ids.is_empty())
    {
        return false;
    }

    let variable_images = variable_image_candidates(reconstruction, options);
    if variable_images
        .iter()
        .any(|&image| reconstruction.frame_index_for_image(image).is_some())
    {
        return false;
    }
    true
}

pub fn refine_bundle_adjustment_ceres(
    frames: &[ImageFrame],
    reconstruction: &mut Reconstruction,
    options: BundleAdjustmentOptions,
) -> Option<BundleAdjustmentReport> {
    if !supports_ceres_ba(reconstruction, &options) {
        return None;
    }
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
    let constant_images = constant_image_set(reconstruction, &options);

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
    }

    let mut image_to_ceres_idx = HashMap::<usize, usize>::new();
    let mut point_to_ceres_idx = HashMap::<usize, usize>::new();
    let mut constant_blocks = HashSet::<usize>::new();
    let mut next_param_index = 0usize;

    let mut problem = NllsProblem::new();
    for obs in &observations {
        let Some(pose) = reconstruction.poses.get(obs.image).copied().flatten() else {
            continue;
        };
        let Some(point) = reconstruction.points.get(obs.point) else {
            continue;
        };
        let camera = reconstruction.camera_for_image(obs.image);
        let xy = obs.xy;

        let pose_param = parameter_ref(
            &mut image_to_ceres_idx,
            &mut next_param_index,
            obs.image,
            se3_to_params(pose),
            constant_images.contains(&obs.image),
            &mut constant_blocks,
        );
        let point_param = parameter_ref(
            &mut point_to_ceres_idx,
            &mut next_param_index,
            obs.point,
            vec![
                point.xyz[0] as f64,
                point.xyz[1] as f64,
                point.xyz[2] as f64,
            ],
            constant_point_filter.contains(&obs.point),
            &mut constant_blocks,
        );

        let cost: CostFunctionType = Box::new(
            move |parameters: &[&[f64]], residuals: &mut [f64], jacobians| {
                let Some(residual) =
                    reprojection_residual(camera, parameters[0], parameters[1], xy)
                else {
                    residuals[0] = 0.0;
                    residuals[1] = 0.0;
                    return jacobians.is_none();
                };
                residuals.copy_from_slice(&residual);
                if let Some(jacobians) = jacobians {
                    return fill_reprojection_jacobians(camera, parameters, xy, jacobians);
                }
                true
            },
        );

        let mut builder = problem
            .residual_block_builder()
            .set_cost(cost, 2)
            .add_parameter(pose_param)
            .add_parameter(point_param);
        if let Some(loss) = ceres_loss(options.loss_function) {
            builder = builder.set_loss(loss);
        }
        problem = builder.build_into_problem().ok()?.0;
    }

    if image_to_ceres_idx.is_empty() || point_to_ceres_idx.is_empty() {
        return None;
    }

    let nonpoint_dim = count_variable_pose_params(&image_to_ceres_idx, &constant_blocks);
    let point_dim = count_variable_point_params(&point_to_ceres_idx, &constant_blocks);
    let effective_parameters = nonpoint_dim + point_dim;
    let residuals = count_variable_residuals(
        &observations,
        &image_to_ceres_idx,
        &point_to_ceres_idx,
        &constant_blocks,
    );
    if effective_parameters == 0 || residuals == 0 {
        return None;
    }

    for idx in constant_blocks {
        problem.set_parameter_block_constant(idx).ok()?;
    }

    let solver_options = ceres_solver_options(&options, image_to_ceres_idx.len());
    let solution = problem.solve(&solver_options).ok()?;
    if !solution.summary.is_solution_usable() {
        return None;
    }

    for (&image, &idx) in &image_to_ceres_idx {
        if let Some(pose) = reconstruction.poses.get_mut(image).and_then(|p| p.as_mut()) {
            *pose = params_to_se3(&solution.parameters[idx]);
        }
    }
    for (&point_id, &idx) in &point_to_ceres_idx {
        if let Some(point) = reconstruction.points.get_mut(point_id) {
            let params = &solution.parameters[idx];
            point.xyz = [params[0] as f32, params[1] as f32, params[2] as f32];
        }
    }

    refresh_point_errors(frames, reconstruction);

    let successful_steps = solution.summary.num_successful_steps().max(0) as usize;
    let unsuccessful_steps = solution.summary.num_unsuccessful_steps().max(0) as usize;
    Some(BundleAdjustmentReport {
        iterations: successful_steps,
        attempted_iterations: successful_steps + unsuccessful_steps,
        successful_steps,
        unsuccessful_steps,
        linear_solver_iterations: solution.summary.num_inner_iteration_steps().max(0) as usize,
        linearization_failures: 0,
        linear_solve_failures: 0,
        invalid_steps: 0,
        rejected_steps: unsuccessful_steps,
        initial_cost: solution.summary.initial_cost(),
        final_cost: solution.summary.final_cost(),
        observations: observations.len(),
        residuals,
        effective_parameters,
        gradient_max_norm: f64::NAN,
        step_norm: f64::NAN,
        step_quality: f64::NAN,
        damping: f64::NAN,
        termination_type: if solution.summary.is_solution_usable() {
            BundleAdjustmentTerminationType::Convergence
        } else {
            BundleAdjustmentTerminationType::Failure
        },
        termination_reason: BundleAdjustmentTerminationReason::GradientTolerance,
    })
}

fn parameter_ref(
    registry: &mut HashMap<usize, usize>,
    next_param_index: &mut usize,
    key: usize,
    values: Vec<f64>,
    is_constant: bool,
    constant_blocks: &mut HashSet<usize>,
) -> ParameterBlockOrIndex {
    if let Some(&idx) = registry.get(&key) {
        return idx.into();
    }
    let idx = *next_param_index;
    *next_param_index += 1;
    registry.insert(key, idx);
    if is_constant {
        constant_blocks.insert(idx);
    }
    values.into()
}

fn variable_image_candidates(
    reconstruction: &Reconstruction,
    options: &BundleAdjustmentOptions,
) -> Vec<usize> {
    if let Some(images) = options.variable_images.as_deref() {
        images.to_vec()
    } else {
        reconstruction
            .poses
            .iter()
            .enumerate()
            .filter_map(|(idx, pose)| pose.is_some().then_some(idx))
            .collect()
    }
}

fn constant_image_set(
    reconstruction: &Reconstruction,
    options: &BundleAdjustmentOptions,
) -> HashSet<usize> {
    let mut constant_images = options.constant_images.iter().copied().collect::<HashSet<_>>();
    if options.variable_images.is_none()
        && options.constant_images.is_empty()
        && options.constant_rigs.is_empty()
        && matches!(options.gauge, BundleAdjustmentGauge::Default)
        && !reconstruction.poses.is_empty()
    {
        constant_images.insert(0);
    }
    constant_images
}

fn ceres_loss(loss: BundleAdjustmentLoss) -> Option<LossFunction> {
    match loss {
        BundleAdjustmentLoss::Trivial => None,
        BundleAdjustmentLoss::Huber { scale } => Some(LossFunction::huber(scale)),
        BundleAdjustmentLoss::SoftL1 { scale } => Some(LossFunction::soft_l1(scale)),
        BundleAdjustmentLoss::Cauchy { scale } => Some(LossFunction::cauchy(scale)),
    }
}

fn ceres_solver_options(options: &BundleAdjustmentOptions, num_poses: usize) -> SolverOptions {
    let linear_solver = if num_poses >= 50 {
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

fn se3_to_params(pose: SE3) -> Vec<f64> {
    let q = pose.quaternion();
    let quat = Quat::from_xyzw(q[0], q[1], q[2], q[3]).normalize();
    let (axis, angle) = quat.to_axis_angle();
    let aa = axis * angle;
    let t = pose.translation();
    vec![
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

fn project(camera: CameraModel, pose_params: &[f64], point_params: &[f64]) -> Option<[f64; 2]> {
    let pose = params_to_se3(pose_params);
    let point = [
        point_params[0] as f32,
        point_params[1] as f32,
        point_params[2] as f32,
    ];
    let p = pose.transform_point(&point);
    camera.img_from_cam(p[0] as f64, p[1] as f64, p[2] as f64)
}

fn reprojection_residual(
    camera: CameraModel,
    pose_params: &[f64],
    point_params: &[f64],
    xy: [f64; 2],
) -> Option<[f64; 2]> {
    let predicted = project(camera, pose_params, point_params)?;
    Some([predicted[0] - xy[0], predicted[1] - xy[1]])
}

fn fill_reprojection_jacobians(
    camera: CameraModel,
    parameters: &[&[f64]],
    xy: [f64; 2],
    jacobians: &mut [Option<&mut [&mut [f64]]>],
) -> bool {
    const EPS: f64 = 1.0e-8;
    let mut pose_params = parameters[0].to_vec();
    let mut point_params = parameters[1].to_vec();

    for (param_idx, jac_opt) in jacobians.iter_mut().enumerate() {
        let Some(jac) = jac_opt else {
            continue;
        };
        let param_len = match param_idx {
            0 => pose_params.len(),
            1 => point_params.len(),
            _ => return false,
        };
        for k in 0..param_len {
            match param_idx {
                0 => {
                    pose_params[k] += EPS;
                    let plus = reprojection_residual(camera, &pose_params, parameters[1], xy);
                    pose_params[k] -= 2.0 * EPS;
                    let minus = reprojection_residual(camera, &pose_params, parameters[1], xy);
                    pose_params[k] += EPS;
                    let (Some(plus), Some(minus)) = (plus, minus) else {
                        return false;
                    };
                    for r in 0..2 {
                        jac[r][k] = (plus[r] - minus[r]) / (2.0 * EPS);
                    }
                }
                1 => {
                    point_params[k] += EPS;
                    let plus = reprojection_residual(camera, parameters[0], &point_params, xy);
                    point_params[k] -= 2.0 * EPS;
                    let minus = reprojection_residual(camera, parameters[0], &point_params, xy);
                    point_params[k] += EPS;
                    let (Some(plus), Some(minus)) = (plus, minus) else {
                        return false;
                    };
                    for r in 0..2 {
                        jac[r][k] = (plus[r] - minus[r]) / (2.0 * EPS);
                    }
                }
                _ => return false,
            }
        }
    }
    true
}

fn count_variable_pose_params(
    image_to_ceres_idx: &HashMap<usize, usize>,
    constant_blocks: &HashSet<usize>,
) -> usize {
    image_to_ceres_idx
        .values()
        .filter(|idx| !constant_blocks.contains(idx))
        .count()
        * 6
}

fn count_variable_point_params(
    point_to_ceres_idx: &HashMap<usize, usize>,
    constant_blocks: &HashSet<usize>,
) -> usize {
    point_to_ceres_idx
        .values()
        .filter(|idx| !constant_blocks.contains(idx))
        .count()
        * 3
}

fn count_variable_residuals(
    observations: &[BaObservation],
    image_to_ceres_idx: &HashMap<usize, usize>,
    point_to_ceres_idx: &HashMap<usize, usize>,
    constant_blocks: &HashSet<usize>,
) -> usize {
    observations
        .iter()
        .filter(|obs| {
            image_to_ceres_idx.contains_key(&obs.image)
                && point_to_ceres_idx.contains_key(&obs.point)
                && (!constant_blocks.contains(&image_to_ceres_idx[&obs.image])
                    || !constant_blocks.contains(&point_to_ceres_idx[&obs.point]))
        })
        .count()
        * 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ba::{refine_bundle_adjustment, BundleAdjustmentOptions};
    use crate::sift::SiftFeatures;
    use crate::types::{Point3D, TrackObservation};
    use rustslam::Descriptors;
    use crate::wide::WideDescriptors;
    use rustslam::KeyPoint;
    use std::path::PathBuf;

    fn frame(id: usize) -> ImageFrame {
        ImageFrame {
            id,
            name: format!("{id}.jpg"),
            path: PathBuf::from(format!("{id}.jpg")),
            width: 100,
            height: 100,
            keypoints: Vec::new(),
            descriptors: Descriptors::new(),
            sift: SiftFeatures::default(),
            wide_descriptors: WideDescriptors {
                data: Vec::new(),
                dim: 0,
                count: 0,
            },
            strong_feature_indices: Vec::new(),
            colors: Vec::new(),
        }
    }

    fn reconstruction(frames: &[ImageFrame]) -> Reconstruction {
        use crate::types::CameraModel;
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

    #[test]
    fn ceres_local_ba_keeps_constant_image_fixed() {
        let mut frames = vec![frame(0), frame(1)];
        frames[0].keypoints = vec![
            KeyPoint::new(45.0, 45.0),
            KeyPoint::new(55.0, 45.0),
            KeyPoint::new(45.0, 55.0),
            KeyPoint::new(55.0, 55.0),
        ];
        frames[1].keypoints = vec![
            KeyPoint::new(70.0, 45.0),
            KeyPoint::new(80.0, 45.0),
            KeyPoint::new(70.0, 55.0),
            KeyPoint::new(80.0, 55.0),
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

        assert!(supports_ceres_ba(
            &reconstruction,
            &BundleAdjustmentOptions {
                variable_images: Some(vec![1]),
                constant_images: vec![0],
                ..BundleAdjustmentOptions::default()
            }
        ));

        let report = refine_bundle_adjustment(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 20,
                loss_function: BundleAdjustmentLoss::Huber { scale: 4.0 },
                max_observation_error_px: 50.0,
                variable_images: Some(vec![1]),
                constant_images: vec![0],
                point_ids: Some((0..4).collect()),
                ..BundleAdjustmentOptions::default()
            },
        )
        .expect("ceres ba should succeed");
        assert_eq!(
            reconstruction.poses[0].unwrap().translation(),
            fixed_pose.translation()
        );
        assert!(report.final_cost <= report.initial_cost);
    }
}
