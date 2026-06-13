use crate::types::{CameraModel, ImageFrame, Reconstruction};
use glam::{Quat, Vec3};
use nalgebra::{DMatrix, DVector, SMatrix, SVector};
use rustslam::SE3;

type Mat2x3 = SMatrix<f64, 2, 3>;
type Mat2x6 = SMatrix<f64, 2, 6>;
type Mat3 = SMatrix<f64, 3, 3>;
type Mat3x6 = SMatrix<f64, 3, 6>;
type Mat6 = SMatrix<f64, 6, 6>;
type Vec2 = SVector<f64, 2>;
type Vec3d = SVector<f64, 3>;
type Vec6 = SVector<f64, 6>;

#[derive(Debug, Clone, Copy)]
pub struct BundleAdjustmentOptions {
    pub iterations: usize,
    pub huber_delta_px: f64,
    pub max_observation_error_px: f64,
}

impl Default for BundleAdjustmentOptions {
    fn default() -> Self {
        Self {
            iterations: 8,
            huber_delta_px: 4.0,
            max_observation_error_px: 16.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BundleAdjustmentReport {
    pub iterations: usize,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub observations: usize,
}

pub fn refine_bundle_adjustment(
    frames: &[ImageFrame],
    reconstruction: &mut Reconstruction,
    options: BundleAdjustmentOptions,
) -> Option<BundleAdjustmentReport> {
    let camera_indices = variable_camera_indices(reconstruction);
    if camera_indices.is_empty() || reconstruction.points.is_empty() {
        return None;
    }
    let observations =
        collect_observations(frames, reconstruction, options.max_observation_error_px);
    if observations.len() < camera_indices.len() * 6 {
        return None;
    }

    let initial_cost = total_cost(reconstruction, &observations, options.huber_delta_px);
    if !initial_cost.is_finite() {
        return None;
    }
    let mut final_cost = initial_cost;
    let mut completed = 0usize;
    let mut damping = 1.0e-3;

    for _ in 0..options.iterations {
        let system = build_schur_system(
            reconstruction,
            &observations,
            &camera_indices,
            options.huber_delta_px,
            damping,
        )?;
        let Some(delta) = system.h.lu().solve(&(-system.g)) else {
            damping *= 10.0;
            continue;
        };
        if !delta.iter().all(|v| v.is_finite()) || delta.norm() > 20.0 {
            damping *= 10.0;
            continue;
        }

        let base_poses = reconstruction.poses.clone();
        let base_points = reconstruction
            .points
            .iter()
            .map(|p| p.xyz)
            .collect::<Vec<_>>();
        let mut accepted = false;
        for step in [1.0, 0.5, 0.25, 0.125, 0.0625] {
            apply_schur_delta(
                reconstruction,
                &observations,
                &camera_indices,
                &system.point_blocks,
                &delta,
                step,
            );
            let candidate_cost = total_cost(reconstruction, &observations, options.huber_delta_px);
            if candidate_cost.is_finite() && candidate_cost + 1.0e-8 < final_cost {
                final_cost = candidate_cost;
                damping = (damping * 0.5).max(1.0e-8);
                completed += 1;
                accepted = true;
                break;
            }
            restore_state(reconstruction, &base_poses, &base_points);
        }
        if !accepted {
            damping *= 4.0;
        }
        if delta.norm() < 1.0e-8 {
            break;
        }
    }

    refresh_point_errors(frames, reconstruction);
    Some(BundleAdjustmentReport {
        iterations: completed,
        initial_cost,
        final_cost,
        observations: observations.len(),
    })
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
    camera_blocks: Vec<(usize, Mat3x6)>,
}

fn variable_camera_indices(reconstruction: &Reconstruction) -> Vec<usize> {
    reconstruction
        .poses
        .iter()
        .enumerate()
        .filter_map(|(idx, pose)| (idx > 0 && pose.is_some()).then_some(idx))
        .collect()
}

fn collect_observations(
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
    max_error_px: f64,
) -> Vec<BaObservation> {
    let mut observations = Vec::new();
    for (point_id, point) in reconstruction.points.iter().enumerate() {
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

fn build_schur_system(
    reconstruction: &Reconstruction,
    observations: &[BaObservation],
    camera_indices: &[usize],
    huber_delta_px: f64,
    damping: f64,
) -> Option<SchurSystem> {
    let mut camera_lookup = vec![None; reconstruction.poses.len()];
    for (var_idx, &image) in camera_indices.iter().enumerate() {
        camera_lookup[image] = Some(var_idx);
    }

    let camera_dim = camera_indices.len() * 6;
    let mut h_cc = DMatrix::<f64>::zeros(camera_dim, camera_dim);
    let mut g_c = DVector::<f64>::zeros(camera_dim);
    let mut point_blocks = (0..reconstruction.points.len())
        .map(|_| PointBlock {
            h_inv: Mat3::zeros(),
            g: Vec3d::zeros(),
            camera_blocks: Vec::new(),
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

        let point_block = &mut point_blocks[obs.point];
        point_block.h_inv += j_point.transpose() * j_point;
        point_block.g += j_point.transpose() * residual;

        if let Some(cam_idx) = camera_lookup[obs.image] {
            let c0 = cam_idx * 6;
            let h = j_pose.transpose() * j_pose;
            let g = j_pose.transpose() * residual;
            for r in 0..6 {
                g_c[c0 + r] += g[r];
                for c in 0..6 {
                    h_cc[(c0 + r, c0 + c)] += h[(r, c)];
                }
            }
            point_block
                .camera_blocks
                .push((cam_idx, j_point.transpose() * j_pose));
        }
    }

    for cam in 0..camera_indices.len() {
        for d in 0..6 {
            let idx = cam * 6 + d;
            h_cc[(idx, idx)] += damping;
        }
    }

    for block in &mut point_blocks {
        if block.camera_blocks.is_empty() {
            continue;
        }
        for d in 0..3 {
            block.h_inv[(d, d)] += damping;
        }
        block.h_inv = block.h_inv.try_inverse()?;
        for &(ci, ref e_i) in &block.camera_blocks {
            let c0 = ci * 6;
            let schur_g = e_i.transpose() * block.h_inv * block.g;
            for r in 0..6 {
                g_c[c0 + r] -= schur_g[r];
            }
            for &(cj, ref e_j) in &block.camera_blocks {
                let c1 = cj * 6;
                let schur_h: Mat6 = e_i.transpose() * block.h_inv * e_j;
                for r in 0..6 {
                    for c in 0..6 {
                        h_cc[(c0 + r, c1 + c)] -= schur_h[(r, c)];
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
    camera_indices: &[usize],
    point_blocks: &[PointBlock],
    camera_delta: &DVector<f64>,
    step: f64,
) {
    let mut camera_lookup = vec![None; reconstruction.poses.len()];
    for (var_idx, &image) in camera_indices.iter().enumerate() {
        camera_lookup[image] = Some(var_idx);
        let delta = Vec6::from_iterator((0..6).map(|k| camera_delta[var_idx * 6 + k] * step));
        if let Some(pose) = reconstruction.poses[image] {
            reconstruction.poses[image] = Some(apply_pose_delta_f64(pose, delta));
        }
    }

    for (point_idx, block) in point_blocks.iter().enumerate() {
        if block.camera_blocks.is_empty() || point_idx >= reconstruction.points.len() {
            continue;
        }
        let mut rhs = block.g;
        for &(cam_idx, ref e) in &block.camera_blocks {
            let delta = Vec6::from_iterator((0..6).map(|k| camera_delta[cam_idx * 6 + k]));
            rhs += e * delta;
        }
        let point_delta = -(block.h_inv * rhs) * step;
        let point = &mut reconstruction.points[point_idx].xyz;
        point[0] += point_delta[0] as f32;
        point[1] += point_delta[1] as f32;
        point[2] += point_delta[2] as f32;
    }

    // The observation list is fixed within one BA call, so no track topology update is needed here.
    let _ = observations;
    let _ = camera_lookup;
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
    let j_pose = numerical_pose_jacobian(camera, pose, point)?;
    let j_point = numerical_point_jacobian(camera, pose, point)?;
    Some((residual, j_pose, j_point))
}

fn numerical_pose_jacobian(camera: CameraModel, pose: SE3, point: [f32; 3]) -> Option<Mat2x6> {
    let mut jacobian = Mat2x6::zeros();
    let eps = [1.0e-6, 1.0e-6, 1.0e-6, 1.0e-5, 1.0e-5, 1.0e-5];
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
    let eps = 1.0e-5;
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

fn project_point(camera: CameraModel, pose: SE3, point: [f32; 3]) -> Option<[f64; 2]> {
    let p = pose.transform_point(&point);
    camera.img_from_cam(p[0] as f64, p[1] as f64, p[2] as f64)
}

fn apply_pose_delta_f64(pose: SE3, delta: Vec6) -> SE3 {
    let tangent = [
        delta[0] as f32,
        delta[1] as f32,
        delta[2] as f32,
        delta[3] as f32,
        delta[4] as f32,
        delta[5] as f32,
    ];
    SE3::exp(&tangent).compose(&pose)
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

fn restore_state(reconstruction: &mut Reconstruction, poses: &[Option<SE3>], points: &[[f32; 3]]) {
    reconstruction.poses.clone_from_slice(poses);
    for (point, xyz) in reconstruction.points.iter_mut().zip(points.iter()) {
        point.xyz = *xyz;
    }
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
