//! Ceres Levenberg-Marquardt solver for joint global positioning (GLOMAP §3.2).

use super::{
    apply_origin_gauge, inverse_depth_along_ray, mean_residual, normalize_joint_scale,
    JointGlobalPositioningOptions, RayObservation,
};
use ceres_solver::loss::LossFunction;
use ceres_solver::parameter_block::ParameterBlockOrIndex;
use ceres_solver::solver::{LinearSolverType, SolverOptions};
use ceres_solver::{CostFunctionType, NllsProblem};
use glam::Vec3;
use std::collections::HashMap;

struct RayResidualBinding {
    ray: [f64; 3],
    sqrt_weight: f64,
}

pub(crate) fn solve_joint_global_positioning_ceres(
    observations: &[RayObservation],
    num_views: usize,
    num_tracks: usize,
    mut centers: Vec<Vec3>,
    mut points: Vec<Vec3>,
    options: &JointGlobalPositioningOptions,
) -> Option<(Vec<Vec3>, Vec<Vec3>, usize, f64)> {
    if observations.is_empty() || num_views < 2 || num_tracks == 0 {
        return None;
    }

    apply_origin_gauge(&mut centers, &mut points);
    centers[0] = Vec3::ZERO;

    let mut block_values = HashMap::<usize, Vec<f64>>::new();
    for view in 0..num_views {
        let center = centers.get(view).copied().unwrap_or(Vec3::ZERO);
        block_values.insert(
            camera_param_index(view),
            vec![center.x as f64, center.y as f64, center.z as f64],
        );
    }
    for track in 0..num_tracks {
        let point = points.get(track).copied().unwrap_or(Vec3::ZERO);
        block_values.insert(
            point_param_index(track, num_views),
            vec![point.x as f64, point.y as f64, point.z as f64],
        );
    }

    let mut problem = NllsProblem::new();
    let mut internal_to_storage = HashMap::<usize, usize>::new();
    let mut next_storage_index = 0usize;

    for obs in observations {
        let binding = RayResidualBinding {
            ray: [
                obs.ray_world.x as f64,
                obs.ray_world.y as f64,
                obs.ray_world.z as f64,
            ],
            sqrt_weight: obs.weight.sqrt(),
        };
        let cost = build_ray_residual_cost(binding);
        let camera_idx = camera_param_index(obs.camera);
        let point_idx = point_param_index(obs.track, num_views);
        let mut builder = problem.residual_block_builder().set_cost(cost, 3);
        builder = builder.add_parameter(param_ref(
            camera_idx,
            block_values.get(&camera_idx).expect("camera block"),
            &mut internal_to_storage,
            &mut next_storage_index,
        ));
        builder = builder.add_parameter(param_ref(
            point_idx,
            block_values.get(&point_idx).expect("point block"),
            &mut internal_to_storage,
            &mut next_storage_index,
        ));
        builder = builder.set_loss(LossFunction::huber(options.huber_threshold));
        problem = builder.build_into_problem().ok()?.0;
    }

    if let Some(&storage_idx) = internal_to_storage.get(&camera_param_index(0)) {
        problem.set_parameter_block_constant(storage_idx).ok()?;
    }

    if options.max_num_iterations == 0 {
        let mut depths = vec![1.0; observations.len()];
        for (obs_idx, obs) in observations.iter().enumerate() {
            let delta = points[obs.track] - centers[obs.camera];
            depths[obs_idx] = inverse_depth_along_ray(obs.ray_world, delta);
        }
        let residual = mean_residual(observations, &centers, &points, &depths);
        return Some((centers, points, 0, residual));
    }

    let max_num_iterations = i32::try_from(options.max_num_iterations).ok()?;
    let solver_options = SolverOptions::builder()
        .max_num_iterations(max_num_iterations)
        .function_tolerance(options.convergence)
        .gradient_tolerance(options.convergence)
        .parameter_tolerance(options.convergence)
        .linear_solver_type(LinearSolverType::DENSE_NORMAL_CHOLESKY)
        .num_threads(1)
        .build()
        .ok()?;
    let solution = problem.solve(&solver_options).ok()?;
    if !solution.summary.is_solution_usable() {
        return None;
    }

    write_back_joint_solution(
        &solution.parameters,
        &internal_to_storage,
        num_views,
        num_tracks,
        &mut centers,
        &mut points,
    );

    if !options.use_translation_averaging_init {
        let mut depths = vec![1.0; observations.len()];
        for (obs_idx, obs) in observations.iter().enumerate() {
            let delta = points[obs.track] - centers[obs.camera];
            depths[obs_idx] = inverse_depth_along_ray(obs.ray_world, delta);
        }
        normalize_joint_scale(&mut centers, &mut points, &mut depths);
    }

    let mut depths = vec![1.0; observations.len()];
    for (obs_idx, obs) in observations.iter().enumerate() {
        let delta = points[obs.track] - centers[obs.camera];
        depths[obs_idx] = inverse_depth_along_ray(obs.ray_world, delta);
    }
    let num_iterations = solution.summary.num_successful_steps().max(0) as usize;
    let residual = mean_residual(observations, &centers, &points, &depths);
    Some((centers, points, num_iterations, residual))
}

fn camera_param_index(view: usize) -> usize {
    view
}

fn point_param_index(track: usize, num_views: usize) -> usize {
    num_views + track
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

fn build_ray_residual_cost(binding: RayResidualBinding) -> CostFunctionType<'static> {
    Box::new(
        move |parameters: &[&[f64]], residuals: &mut [f64], jacobians| {
            let center = parameter_xyz(parameters[0]);
            let point = parameter_xyz(parameters[1]);
            let Some((residual, _)) = ray_direction_residual_and_jacobian(
                center,
                point,
                binding.ray,
                binding.sqrt_weight,
            ) else {
                residuals.fill(0.0);
                return jacobians.is_none();
            };
            residuals.copy_from_slice(&residual);
            if let Some(jacobians) = jacobians {
                fill_numeric_ray_jacobians(
                    center,
                    point,
                    binding.ray,
                    binding.sqrt_weight,
                    jacobians,
                );
            }
            true
        },
    )
}

fn fill_numeric_ray_jacobians(
    center: [f64; 3],
    point: [f64; 3],
    ray: [f64; 3],
    sqrt_weight: f64,
    jacobians: &mut [Option<&mut [&mut [f64]]>],
) {
    const EPS: f64 = 1.0e-7;
    let base = ray_direction_residual_and_jacobian(center, point, ray, sqrt_weight)
        .map(|(residual, _)| residual)
        .unwrap_or([0.0; 3]);

    for (block_idx, jac) in jacobians.iter_mut().enumerate() {
        let Some(jac) = jac.as_mut() else {
            continue;
        };
        for col in 0..3 {
            let mut plus_center = center;
            let mut minus_center = center;
            let mut plus_point = point;
            let mut minus_point = point;
            match block_idx {
                0 => {
                    plus_center[col] += EPS;
                    minus_center[col] -= EPS;
                }
                1 => {
                    plus_point[col] += EPS;
                    minus_point[col] -= EPS;
                }
                _ => continue,
            }
            let plus = if block_idx == 0 {
                ray_direction_residual_and_jacobian(plus_center, point, ray, sqrt_weight)
            } else {
                ray_direction_residual_and_jacobian(center, plus_point, ray, sqrt_weight)
            }
            .map(|(residual, _)| residual)
            .unwrap_or(base);
            let minus = if block_idx == 0 {
                ray_direction_residual_and_jacobian(minus_center, point, ray, sqrt_weight)
            } else {
                ray_direction_residual_and_jacobian(center, minus_point, ray, sqrt_weight)
            }
            .map(|(residual, _)| residual)
            .unwrap_or(base);
            for row in 0..3 {
                jac[row][col] = (plus[row] - minus[row]) / (2.0 * EPS);
            }
        }
    }
}

fn parameter_xyz(values: &[f64]) -> [f64; 3] {
    [values[0], values[1], values[2]]
}

fn ray_direction_residual_and_jacobian(
    center: [f64; 3],
    point: [f64; 3],
    ray: [f64; 3],
    sqrt_weight: f64,
) -> Option<([f64; 3], Option<([[f64; 3]; 3], [[f64; 3]; 3])>)> {
    let delta = [
        point[0] - center[0],
        point[1] - center[1],
        point[2] - center[2],
    ];
    let den = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];

    let mut residual = ray;
    let mut jr_center = [[0.0; 3]; 3];
    let mut jr_point = [[0.0; 3]; 3];

    if den > 1.0e-12 {
        let num_dot = ray[0] * delta[0] + ray[1] * delta[1] + ray[2] * delta[2];
        if num_dot > 0.0 {
            let d = num_dot / den;
            residual[0] -= d * delta[0];
            residual[1] -= d * delta[1];
            residual[2] -= d * delta[2];

            let grad = [
                (ray[0] - 2.0 * d * delta[0]) / den,
                (ray[1] - 2.0 * d * delta[1]) / den,
                (ray[2] - 2.0 * d * delta[2]) / den,
            ];
            for i in 0..3 {
                for j in 0..3 {
                    let dr_ddelta = -delta[i] * grad[j] - d * (if i == j { 1.0 } else { 0.0 });
                    jr_point[i][j] = dr_ddelta;
                    jr_center[i][j] = -dr_ddelta;
                }
            }
        }
    }

    for i in 0..3 {
        residual[i] *= sqrt_weight;
        for j in 0..3 {
            jr_center[i][j] *= sqrt_weight;
            jr_point[i][j] *= sqrt_weight;
        }
    }

    Some((residual, Some((jr_center, jr_point))))
}

fn write_back_joint_solution(
    parameters: &[Vec<f64>],
    internal_to_storage: &HashMap<usize, usize>,
    num_views: usize,
    num_tracks: usize,
    centers: &mut [Vec3],
    points: &mut [Vec3],
) {
    let _ = num_tracks;
    let mut storage_to_internal = HashMap::<usize, usize>::new();
    for (&internal, &storage) in internal_to_storage {
        storage_to_internal.insert(storage, internal);
    }
    for (storage_idx, values) in parameters.iter().enumerate() {
        let Some(&internal) = storage_to_internal.get(&storage_idx) else {
            continue;
        };
        let xyz = Vec3::new(values[0] as f32, values[1] as f32, values[2] as f32);
        if internal < num_views {
            centers[internal] = xyz;
        } else {
            points[internal - num_views] = xyz;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint_global_positioning::{
        build_ray_observations, solve_joint_global_positioning_alternating,
        JointGlobalPositioningSolver,
    };
    use crate::track_establishment::{FeatureNode, Track};
    use crate::types::{CameraModel, PairGeometry};
    use glam::Quat;
    use rustslam::SE3;

    fn test_camera() -> CameraModel {
        CameraModel::new_pinhole(640, 480, 500.0, 500.0, 320.0, 240.0)
    }

    fn synth_frame(id: usize, keypoints: Vec<(f32, f32)>) -> crate::types::ImageFrame {
        crate::types::ImageFrame {
            id,
            name: format!("img_{id}.jpg"),
            path: std::path::PathBuf::from(format!("img_{id}.jpg")),
            width: 640,
            height: 480,
            keypoints: keypoints
                .into_iter()
                .map(|(x, y)| rustslam::KeyPoint::new(x, y))
                .collect(),
            descriptors: rustslam::Descriptors::new(),
            sift: crate::sift::SiftFeatures::default(),
            wide_descriptors: crate::wide::WideDescriptors {
                data: Vec::new(),
                dim: 0,
                count: 0,
            },
            strong_feature_indices: Vec::new(),
            colors: vec![[128, 128, 128]],
        }
    }

    fn synth_pair(
        i: usize,
        j: usize,
        rotations: &[Quat],
        centers: &[glam::Vec3],
        inliers: usize,
    ) -> PairGeometry {
        let r_ij = (rotations[j] * rotations[i].inverse()).normalize();
        let t_ij = -(rotations[j] * (centers[j] - centers[i]));
        PairGeometry {
            left: i,
            right: j,
            two_view_config: 2,
            f_matrix: None,
            e_matrix: None,
            h_matrix: None,
            qvec: None,
            tvec: None,
            matches: Vec::new(),
            inlier_matches: Vec::new(),
            relative_pose: SE3::from_quat_translation(r_ij, t_ij),
            inliers,
            triangulated: inliers,
            mean_reprojection_error_px: 0.5,
            rotation_deg: 0.0,
            median_triangulation_angle_deg: 5.0,
            pose_graph_only: false,
        }
    }

    #[test]
    fn ceres_joint_positioning_improves_bata_warm_start() {
        let camera = test_camera();
        let n = 5;
        let rotations = vec![Quat::IDENTITY; n];
        let centers = vec![
            glam::Vec3::ZERO,
            glam::Vec3::new(0.5, 0.0, 0.0),
            glam::Vec3::new(1.0, 0.0, 0.0),
            glam::Vec3::new(1.5, 0.0, 0.0),
            glam::Vec3::new(2.0, 0.0, 0.0),
        ];
        let gt_points = vec![
            glam::Vec3::new(0.5, 0.0, 4.0),
            glam::Vec3::new(1.0, 0.25, 5.0),
            glam::Vec3::new(1.5, -0.1, 4.5),
        ];

        let mut frames = vec![synth_frame(0, Vec::new()); n];
        let mut tracks = Vec::new();
        for point in &gt_points {
            let mut observations = Vec::new();
            for view in 0..n {
                let translation = -(rotations[view] * centers[view]);
                let pose = SE3::from_quat_translation(rotations[view], translation);
                let p = pose.transform_point(&[point.x, point.y, point.z]);
                let Some(xy) = camera.img_from_cam(p[0] as f64, p[1] as f64, p[2] as f64) else {
                    continue;
                };
                let kp = (xy[0] as f32, xy[1] as f32);
                if observations.is_empty() {
                    frames[view] = synth_frame(view, vec![kp]);
                    observations.push(FeatureNode::new(view, 0));
                } else {
                    let feature_idx = frames[view].keypoints.len();
                    frames[view]
                        .keypoints
                        .push(rustslam::KeyPoint::new(kp.0, kp.1));
                    frames[view].colors.push([128, 128, 128]);
                    observations.push(FeatureNode::new(view, feature_idx));
                }
            }
            tracks.push(Track { observations });
        }

        let mut pairs = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                pairs.push(synth_pair(i, j, &rotations, &centers, 100));
            }
        }

        let init = crate::joint_global_positioning::estimate_joint_global_positions(
            &rotations,
            &tracks,
            &frames,
            camera,
            &pairs,
            &JointGlobalPositioningOptions {
                solver: JointGlobalPositioningSolver::Alternating,
                max_num_iterations: 0,
                ..JointGlobalPositioningOptions::default()
            },
        )
        .unwrap();
        let refined = crate::joint_global_positioning::estimate_joint_global_positions(
            &rotations,
            &tracks,
            &frames,
            camera,
            &pairs,
            &JointGlobalPositioningOptions {
                solver: JointGlobalPositioningSolver::CeresLevenbergMarquardt,
                ..JointGlobalPositioningOptions::default()
            },
        )
        .unwrap();
        assert!(refined.num_iterations > 0);
        assert!(refined.mean_residual <= init.mean_residual + 1.0e-6);
        assert!(refined.mean_residual < 0.05);
    }

    #[test]
    fn ceres_joint_positioning_matches_or_beats_alternating() {
        let camera = test_camera();
        let n = 5;
        let rotations = vec![Quat::IDENTITY; n];
        let centers = vec![
            glam::Vec3::ZERO,
            glam::Vec3::new(0.5, 0.0, 0.0),
            glam::Vec3::new(1.0, 0.0, 0.0),
            glam::Vec3::new(1.5, 0.0, 0.0),
            glam::Vec3::new(2.0, 0.0, 0.0),
        ];
        let gt_points = vec![
            glam::Vec3::new(0.5, 0.0, 4.0),
            glam::Vec3::new(1.0, 0.25, 5.0),
        ];

        let mut frames = vec![synth_frame(0, Vec::new()); n];
        let mut tracks = Vec::new();
        for point in &gt_points {
            let mut observations = Vec::new();
            for view in 0..n {
                let translation = -(rotations[view] * centers[view]);
                let pose = SE3::from_quat_translation(rotations[view], translation);
                let p = pose.transform_point(&[point.x, point.y, point.z]);
                let Some(xy) = camera.img_from_cam(p[0] as f64, p[1] as f64, p[2] as f64) else {
                    continue;
                };
                let kp = (xy[0] as f32, xy[1] as f32);
                if observations.is_empty() {
                    frames[view] = synth_frame(view, vec![kp]);
                    observations.push(FeatureNode::new(view, 0));
                } else {
                    let feature_idx = frames[view].keypoints.len();
                    frames[view]
                        .keypoints
                        .push(rustslam::KeyPoint::new(kp.0, kp.1));
                    frames[view].colors.push([128, 128, 128]);
                    observations.push(FeatureNode::new(view, feature_idx));
                }
            }
            tracks.push(Track { observations });
        }

        let observations = build_ray_observations(&rotations, &tracks, &frames, camera);
        let options = JointGlobalPositioningOptions::default();
        let noisy_centers = centers
            .iter()
            .map(|c| *c + glam::Vec3::new(0.05, -0.03, 0.02))
            .collect::<Vec<_>>();
        let noisy_points = gt_points
            .iter()
            .map(|p| *p + glam::Vec3::new(-0.1, 0.08, -0.15))
            .collect::<Vec<_>>();

        let (_, _, _, alt_residual) = solve_joint_global_positioning_alternating(
            &observations,
            n,
            tracks.len(),
            noisy_centers.clone(),
            noisy_points.clone(),
            &options,
        )
        .unwrap();
        let ceres_options = JointGlobalPositioningOptions {
            solver: JointGlobalPositioningSolver::CeresLevenbergMarquardt,
            ..options
        };
        let (_, _, _, ceres_residual) = solve_joint_global_positioning_ceres(
            &observations,
            n,
            tracks.len(),
            noisy_centers,
            noisy_points,
            &ceres_options,
        )
        .unwrap();
        assert!(ceres_residual <= alt_residual + 1.0e-6);
    }
}
