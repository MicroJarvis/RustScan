use crate::two_view::{
    estimate_calibrated_two_view_with_observations_and_cameras, triangulate_world_point,
    TwoViewOptions,
};
use crate::types::{CameraModel, ImageFrame, PairGeometry};
use glam::{Quat, Vec3};
use nalgebra::{Matrix3, SMatrix, SVector, Vector3};
use rustslam::{Match, SE3};

pub fn camera_center(pose: SE3) -> Vec3 {
    let q = pose.quaternion();
    let r = Quat::from_xyzw(q[0], q[1], q[2], q[3]).normalize();
    let t = Vec3::from_array(pose.translation());
    -(r.inverse() * t)
}

pub fn pose_from_rotation_center(rotation: Quat, center: Vec3) -> SE3 {
    SE3::from_quat_translation(rotation.normalize(), -(rotation.normalize() * center))
}

pub fn pose_rotation(pose: SE3) -> Quat {
    let q = pose.quaternion();
    Quat::from_xyzw(q[0], q[1], q[2], q[3]).normalize()
}

pub fn pose_with_flipped_translation(pose: SE3) -> SE3 {
    let rotation = pose_rotation(pose);
    let translation = -Vec3::from_array(pose.translation());
    SE3::from_quat_translation(rotation, translation)
}

pub fn relative_rotation_deg(a: SE3, b: SE3) -> f32 {
    let qa = a.quaternion();
    let qb = b.quaternion();
    let dot = (qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3])
        .abs()
        .clamp(0.0, 1.0);
    (2.0 * dot.acos()).to_degrees()
}

pub fn reprojection_error_px(point: [f32; 3], pose: SE3, xy: [f32; 2], camera: CameraModel) -> f32 {
    let p = pose.transform_point(&point);
    let Some([u, v]) = camera.img_from_cam_f32(p[0], p[1], p[2]) else {
        return f32::INFINITY;
    };
    ((u - xy[0]).powi(2) + (v - xy[1]).powi(2)).sqrt()
}

pub fn estimate_pair_geometry(
    left_idx: usize,
    right_idx: usize,
    left: &ImageFrame,
    right: &ImageFrame,
    matches: &[Match],
    camera: CameraModel,
    essential_threshold: f32,
    essential_iterations: u32,
    min_inliers: usize,
    min_triangulated: usize,
) -> Option<PairGeometry> {
    estimate_pair_geometry_with_cameras(
        left_idx,
        right_idx,
        left,
        right,
        matches,
        camera,
        camera,
        essential_threshold,
        essential_iterations,
        min_inliers,
        min_triangulated,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn estimate_pair_geometry_with_cameras(
    left_idx: usize,
    right_idx: usize,
    left: &ImageFrame,
    right: &ImageFrame,
    matches: &[Match],
    left_camera: CameraModel,
    right_camera: CameraModel,
    essential_threshold: f32,
    essential_iterations: u32,
    min_inliers: usize,
    min_triangulated: usize,
) -> Option<PairGeometry> {
    estimate_pair_geometry_with_options_and_cameras(
        left_idx,
        right_idx,
        left,
        right,
        matches,
        left_camera,
        right_camera,
        essential_threshold,
        essential_iterations,
        min_inliers,
        min_triangulated,
        PairEstimationOptions::default(),
    )
}

#[derive(Debug, Clone, Copy)]
pub struct PairEstimationOptions {
    pub max_pose_matches: usize,
    pub use_hartley_refinement: bool,
    pub use_five_point: bool,
    pub refine_sampson: bool,
}

impl Default for PairEstimationOptions {
    fn default() -> Self {
        Self {
            max_pose_matches: 1024,
            use_hartley_refinement: true,
            use_five_point: true,
            refine_sampson: true,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn estimate_pair_geometry_with_options(
    left_idx: usize,
    right_idx: usize,
    left: &ImageFrame,
    right: &ImageFrame,
    matches: &[Match],
    camera: CameraModel,
    essential_threshold: f32,
    essential_iterations: u32,
    min_inliers: usize,
    min_triangulated: usize,
    options: PairEstimationOptions,
) -> Option<PairGeometry> {
    estimate_pair_geometry_with_options_and_cameras(
        left_idx,
        right_idx,
        left,
        right,
        matches,
        camera,
        camera,
        essential_threshold,
        essential_iterations,
        min_inliers,
        min_triangulated,
        options,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn estimate_pair_geometry_with_options_and_cameras(
    left_idx: usize,
    right_idx: usize,
    left: &ImageFrame,
    right: &ImageFrame,
    matches: &[Match],
    left_camera: CameraModel,
    right_camera: CameraModel,
    essential_threshold: f32,
    essential_iterations: u32,
    min_inliers: usize,
    min_triangulated: usize,
    options: PairEstimationOptions,
) -> Option<PairGeometry> {
    if matches.len() < min_inliers.max(8) {
        return None;
    }
    let pose_matches = select_pose_matches(matches, left, options.max_pose_matches);
    let mut norm_left = Vec::with_capacity(pose_matches.len());
    let mut norm_right = Vec::with_capacity(pose_matches.len());
    let mut obs_left_px = Vec::with_capacity(pose_matches.len());
    let mut obs_right_px = Vec::with_capacity(pose_matches.len());
    let mut valid_matches = Vec::with_capacity(pose_matches.len());
    for m in &pose_matches {
        let li = m.query_idx as usize;
        let ri = m.train_idx as usize;
        if li >= left.keypoints.len() || ri >= right.keypoints.len() {
            continue;
        }
        let lk = &left.keypoints[li];
        let rk = &right.keypoints[ri];
        let Some(left_xy) = left_camera.cam_from_img_f32(lk.x(), lk.y()) else {
            continue;
        };
        let Some(right_xy) = right_camera.cam_from_img_f32(rk.x(), rk.y()) else {
            continue;
        };
        norm_left.push(left_xy);
        norm_right.push(right_xy);
        obs_left_px.push([lk.x(), lk.y()]);
        obs_right_px.push([rk.x(), rk.y()]);
        valid_matches.push(m.clone());
    }
    if valid_matches.len() < min_inliers.max(8) {
        return None;
    }
    let normalized_threshold =
        mean_cam_from_img_threshold(left_camera, right_camera, essential_threshold as f64);
    let estimate = estimate_calibrated_two_view_with_observations_and_cameras(
        &norm_left,
        &norm_right,
        &obs_left_px,
        &obs_right_px,
        left_camera,
        right_camera,
        &TwoViewOptions {
            ransac_max_error_px: essential_threshold as f64,
            ransac_threshold: normalized_threshold,
            ransac_min_iterations: 100,
            ransac_max_iterations: essential_iterations,
            random_seed: ((left_idx as u64) << 32) ^ right_idx as u64 ^ 0x243f_6a88_85a3_08d3,
            loransac_num_lo_steps: 6,
            min_inliers,
            min_inlier_ratio: 0.0,
            min_triangulated,
            min_e_f_inlier_ratio: 0.95,
            max_h_inlier_ratio: 0.8,
            force_h_use: false,
            multiple_models: false,
            multiple_ignore_watermark: true,
            detect_watermark: true,
            watermark_min_inlier_ratio: 0.7,
            watermark_border_size: 0.1,
            watermark_detection_max_error_px: 4.0,
            filter_stationary_matches: false,
            stationary_matches_max_error_px: 4.0,
            use_hartley_refinement: options.use_hartley_refinement
                && right_idx.abs_diff(left_idx) <= 2,
            use_five_point: options.use_five_point && right_idx.abs_diff(left_idx) <= 3,
        },
    )?;
    let inlier_mask = estimate.inlier_mask;
    let mut pose_inlier_matches = Vec::new();
    let mut in_left = Vec::new();
    let mut in_right = Vec::new();
    for (idx, &is_inlier) in inlier_mask.iter().enumerate().take(valid_matches.len()) {
        if is_inlier {
            pose_inlier_matches.push(valid_matches[idx].clone());
            in_left.push(norm_left[idx]);
            in_right.push(norm_right[idx]);
        }
    }
    if pose_inlier_matches.len() < min_inliers {
        return None;
    }
    let mut relative_pose = estimate.pose;
    let mut best_count = estimate.triangulated;
    let mut best_mean_reproj = estimate.mean_reprojection_error_px;
    let mut best_rotation_deg = estimate.rotation_deg;
    if options.refine_sampson && std::env::var_os("RUSTSFM_DISABLE_SAMPSON_REFINE").is_none() {
        let refined = if std::env::var_os("RUSTSFM_REPROJ_RELATIVE_REFINE").is_some() {
            refine_relative_pose_reprojection(relative_pose, &in_left, &in_right)
                .or_else(|| refine_relative_pose_sampson(relative_pose, &in_left, &in_right))
        } else {
            refine_relative_pose_sampson(relative_pose, &in_left, &in_right)
        };
        if let Some(refined) = refined {
            let use_guard = std::env::var_os("RUSTSFM_GUARD_SAMPSON_REFINE").is_some();
            if !use_guard {
                relative_pose = refined;
            }
            let mut refined_count = 0usize;
            let mut reproj_sum = 0.0f32;
            let mut refined_angles = Vec::new();
            for (idx, (&p1, &p2)) in in_left.iter().zip(in_right.iter()).enumerate() {
                let Some(xyz) = triangulate_world_point(SE3::identity(), refined, p1, p2) else {
                    continue;
                };
                let m = &pose_inlier_matches[idx];
                let lk = &left.keypoints[m.query_idx as usize];
                let rk = &right.keypoints[m.train_idx as usize];
                let err = mean_pair_reprojection_error_with_cameras(
                    xyz,
                    SE3::identity(),
                    refined,
                    [lk.x(), lk.y()],
                    [rk.x(), rk.y()],
                    left_camera,
                    right_camera,
                );
                if err.is_finite() {
                    reproj_sum += err;
                    refined_count += 1;
                    if let Some(angle) = triangulation_angle_deg(SE3::identity(), refined, xyz) {
                        refined_angles.push(angle);
                    }
                }
            }
            let refined_reproj = if refined_count > 0 {
                reproj_sum / refined_count as f32
            } else {
                f32::INFINITY
            };
            let accept_refined = if use_guard {
                let refined_median_angle = median_f32(&mut refined_angles);
                let rotation_step = relative_rotation_deg(refined, relative_pose);
                refined_count >= best_count.saturating_sub(best_count / 10)
                    && refined_reproj <= best_mean_reproj * 1.5
                    && refined_median_angle + 0.05 >= estimate.median_triangulation_angle_deg
                    && rotation_step <= sampson_refine_rotation_limit_deg()
            } else {
                refined_count >= best_count.saturating_sub(best_count / 10)
                    && refined_reproj <= best_mean_reproj * 1.5
            };
            if accept_refined {
                if use_guard {
                    relative_pose = refined;
                }
                best_count = refined_count;
                best_mean_reproj = refined_reproj;
                best_rotation_deg = relative_rotation_deg(relative_pose, SE3::identity());
            }
        }
    }
    if best_count < min_triangulated {
        return None;
    }
    let mut output_inlier_matches = pose_inlier_matches;
    if std::env::var_os("RUSTSFM_DENSE_PAIR_INLIERS").is_some() {
        let dense_inlier_matches = collect_pose_consistent_matches(
            matches,
            left,
            right,
            left_camera,
            right_camera,
            relative_pose,
        );
        if dense_inlier_matches.len() > output_inlier_matches.len() {
            output_inlier_matches = limit_dense_pair_inliers(dense_inlier_matches);
        }
    }
    Some(PairGeometry {
        left: left_idx,
        right: right_idx,
        two_view_config: estimate.two_view_config,
        f_matrix: estimate.fundamental.map(matrix3_to_row_array),
        e_matrix: estimate.e_matrix,
        h_matrix: estimate.homography.map(matrix3_to_row_array),
        qvec: estimate.qvec,
        tvec: estimate.tvec,
        matches: valid_matches,
        inlier_matches: output_inlier_matches,
        relative_pose,
        inliers: inlier_mask.iter().filter(|&&x| x).count(),
        triangulated: best_count,
        mean_reprojection_error_px: best_mean_reproj,
        rotation_deg: best_rotation_deg,
        median_triangulation_angle_deg: estimate.median_triangulation_angle_deg,
        pose_graph_only: false,
    })
}

fn matrix3_to_row_array(matrix: nalgebra::Matrix3<f64>) -> [f64; 9] {
    [
        matrix[(0, 0)],
        matrix[(0, 1)],
        matrix[(0, 2)],
        matrix[(1, 0)],
        matrix[(1, 1)],
        matrix[(1, 2)],
        matrix[(2, 0)],
        matrix[(2, 1)],
        matrix[(2, 2)],
    ]
}

fn mean_cam_from_img_threshold(
    left_camera: CameraModel,
    right_camera: CameraModel,
    threshold_px: f64,
) -> f64 {
    0.5 * (left_camera.cam_from_img_threshold(threshold_px)
        + right_camera.cam_from_img_threshold(threshold_px))
}

fn limit_dense_pair_inliers(mut matches: Vec<Match>) -> Vec<Match> {
    let limit = dense_pair_inlier_limit();
    if matches.len() <= limit || limit == 0 {
        return matches;
    }
    matches.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    matches.truncate(limit);
    matches
}

fn dense_pair_inlier_limit() -> usize {
    std::env::var("RUSTSFM_DENSE_PAIR_INLIER_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 32)
        .unwrap_or(2048)
}

fn collect_pose_consistent_matches(
    matches: &[Match],
    left: &ImageFrame,
    right: &ImageFrame,
    left_camera: CameraModel,
    right_camera: CameraModel,
    relative_pose: SE3,
) -> Vec<Match> {
    let mut inliers = Vec::new();
    let max_reproj = dense_pair_reprojection_threshold_px();
    let max_sampson = dense_pair_sampson_threshold(left_camera, right_camera);
    for m in matches {
        let li = m.query_idx as usize;
        let ri = m.train_idx as usize;
        if li >= left.keypoints.len() || ri >= right.keypoints.len() {
            continue;
        }
        let lk = &left.keypoints[li];
        let rk = &right.keypoints[ri];
        let Some(left_xy) = left_camera.cam_from_img_f32(lk.x(), lk.y()) else {
            continue;
        };
        let Some(right_xy) = right_camera.cam_from_img_f32(rk.x(), rk.y()) else {
            continue;
        };
        let residual = sampson_residual(relative_pose, left_xy, right_xy);
        if !residual.is_finite() || residual.abs() > max_sampson {
            continue;
        }
        let Some(xyz) = crate::two_view::triangulate_world_point(
            SE3::identity(),
            relative_pose,
            left_xy,
            right_xy,
        ) else {
            continue;
        };
        let err = mean_pair_reprojection_error_with_cameras(
            xyz,
            SE3::identity(),
            relative_pose,
            [lk.x(), lk.y()],
            [rk.x(), rk.y()],
            left_camera,
            right_camera,
        );
        if err.is_finite() && err <= max_reproj {
            inliers.push(m.clone());
        }
    }
    inliers
}

fn dense_pair_reprojection_threshold_px() -> f32 {
    std::env::var("RUSTSFM_DENSE_PAIR_REPROJ_PX")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(2.0)
}

fn dense_pair_sampson_threshold(left_camera: CameraModel, right_camera: CameraModel) -> f32 {
    let px = std::env::var("RUSTSFM_DENSE_PAIR_SAMPSON_PX")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(2.0);
    mean_cam_from_img_threshold(left_camera, right_camera, px as f64) as f32
}

fn sampson_refine_rotation_limit_deg() -> f32 {
    std::env::var("RUSTSFM_SAMPSON_REFINE_ROT_LIMIT_DEG")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(0.25)
}

fn select_pose_matches(matches: &[Match], frame: &ImageFrame, max_count: usize) -> Vec<Match> {
    if matches.len() <= max_count || max_count == 0 {
        return matches.to_vec();
    }
    const GRID: usize = 8;
    let per_cell = (max_count / (GRID * GRID)).max(4);
    let mut order = (0..matches.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| {
        matches[a]
            .distance
            .partial_cmp(&matches[b].distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut cell_counts = [0usize; GRID * GRID];
    let mut selected = Vec::with_capacity(max_count);
    let mut used = vec![false; matches.len()];
    for &idx in &order {
        let m = &matches[idx];
        let Some(kp) = frame.keypoints.get(m.query_idx as usize) else {
            continue;
        };
        let cell_x = ((kp.x() / frame.width.max(1) as f32) * GRID as f32)
            .floor()
            .clamp(0.0, (GRID - 1) as f32) as usize;
        let cell_y = ((kp.y() / frame.height.max(1) as f32) * GRID as f32)
            .floor()
            .clamp(0.0, (GRID - 1) as f32) as usize;
        let cell = cell_y * GRID + cell_x;
        if cell_counts[cell] >= per_cell {
            continue;
        }
        selected.push(m.clone());
        used[idx] = true;
        cell_counts[cell] += 1;
        if selected.len() == max_count {
            return selected;
        }
    }
    for &idx in &order {
        if used[idx] {
            continue;
        }
        selected.push(matches[idx].clone());
        if selected.len() == max_count {
            break;
        }
    }
    selected
}

fn refine_relative_pose_sampson(initial: SE3, pts1: &[[f32; 2]], pts2: &[[f32; 2]]) -> Option<SE3> {
    if pts1.len().min(pts2.len()) < 16 {
        return None;
    }
    let mut pose = normalize_relative_translation(initial)?;
    let mut last_cost = sampson_cost(pose, pts1, pts2);
    if !last_cost.is_finite() {
        return None;
    }
    for _ in 0..10 {
        let mut h = SMatrix::<f32, 6, 6>::zeros();
        let mut b = SVector::<f32, 6>::zeros();
        let mut used = 0usize;
        for (&p1, &p2) in sampled_relative_refine_points(pts1, pts2).iter() {
            let residual = sampson_residual(pose, p1, p2);
            if !residual.is_finite() {
                continue;
            }
            let weight = huber_weight(residual.abs(), 2.0e-3);
            let jacobian = numerical_sampson_jacobian(pose, p1, p2)?;
            h += jacobian.transpose() * jacobian * weight;
            b += jacobian.transpose() * SVector::<f32, 1>::new(residual) * weight;
            used += 1;
        }
        if used < 16 {
            return None;
        }
        let lambda = 1.0e-4 * (h.trace() / 6.0).max(1.0);
        for axis in 0..6 {
            h[(axis, axis)] += lambda;
        }
        let delta = h.lu().solve(&(-b))?;
        if !delta.iter().all(|v| v.is_finite()) || delta.norm() > 0.5 {
            return None;
        }
        let mut accepted = false;
        for scale in [1.0f32, 0.5, 0.25, 0.125, 0.0625] {
            let candidate = perturb_relative_pose(pose, delta * scale)?;
            let cost = sampson_cost(candidate, pts1, pts2);
            if cost.is_finite() && cost < last_cost {
                pose = candidate;
                last_cost = cost;
                accepted = true;
                break;
            }
        }
        if !accepted || delta.norm() < 1.0e-6 {
            break;
        }
    }
    Some(pose)
}

fn refine_relative_pose_reprojection(
    initial: SE3,
    pts1: &[[f32; 2]],
    pts2: &[[f32; 2]],
) -> Option<SE3> {
    if pts1.len().min(pts2.len()) < 16 {
        return None;
    }
    let mut pose = normalize_relative_translation(initial)?;
    let mut last_cost = relative_reprojection_cost(pose, pts1, pts2);
    if !last_cost.is_finite() {
        return None;
    }
    for _ in 0..12 {
        let mut h = SMatrix::<f32, 6, 6>::zeros();
        let mut b = SVector::<f32, 6>::zeros();
        let mut used = 0usize;
        for (&p1, &p2) in pts1.iter().zip(pts2.iter()) {
            let residual = relative_reprojection_residual4(pose, p1, p2)?;
            let err = residual.norm();
            if !err.is_finite() || err > 8.0e-3 {
                continue;
            }
            let weight = huber_weight(err, 2.0e-3);
            let jacobian = numerical_relative_reprojection_jacobian4(pose, p1, p2)?;
            h += jacobian.transpose() * jacobian * weight;
            b += jacobian.transpose() * residual * weight;
            used += 1;
        }
        if used < 16 {
            return None;
        }
        let lambda = 1.0e-4 * (h.trace() / 6.0).max(1.0);
        for axis in 0..6 {
            h[(axis, axis)] += lambda;
        }
        let delta = h.lu().solve(&(-b))?;
        if !delta.iter().all(|v| v.is_finite()) || delta.norm() > 0.25 {
            return None;
        }
        let mut accepted = false;
        for scale in [1.0f32, 0.5, 0.25, 0.125, 0.0625] {
            let candidate = perturb_relative_pose(pose, delta * scale)?;
            let cost = relative_reprojection_cost(candidate, pts1, pts2);
            if cost.is_finite() && cost < last_cost {
                pose = candidate;
                last_cost = cost;
                accepted = true;
                break;
            }
        }
        if !accepted || delta.norm() < 1.0e-6 {
            break;
        }
    }
    Some(pose)
}

fn sampled_relative_refine_points<'a>(
    pts1: &'a [[f32; 2]],
    pts2: &'a [[f32; 2]],
) -> Vec<(&'a [f32; 2], &'a [f32; 2])> {
    let n = pts1.len().min(pts2.len());
    let limit = relative_refine_point_limit().min(n);
    if limit == n {
        return (0..n).map(|idx| (&pts1[idx], &pts2[idx])).collect();
    }
    (0..limit)
        .map(|k| {
            let idx = k * n / limit;
            (&pts1[idx], &pts2[idx])
        })
        .collect()
}

fn relative_refine_point_limit() -> usize {
    std::env::var("RUSTSFM_RELATIVE_REFINE_POINTS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 16)
        .unwrap_or(256)
}

fn relative_reprojection_cost(pose: SE3, pts1: &[[f32; 2]], pts2: &[[f32; 2]]) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0usize;
    for (&p1, &p2) in pts1.iter().zip(pts2.iter()) {
        let Some(residual) = relative_reprojection_residual4(pose, p1, p2) else {
            continue;
        };
        let err = residual.norm();
        if err.is_finite() {
            let robust = err.min(1.0e-2);
            total += robust * robust;
            count += 1;
        }
    }
    if count == 0 {
        f32::INFINITY
    } else {
        total / count as f32
    }
}

fn relative_reprojection_residual4(
    pose: SE3,
    p1: [f32; 2],
    p2: [f32; 2],
) -> Option<SVector<f32, 4>> {
    let point = crate::two_view::triangulate_relative_pose_point(pose, p1, p2)?;
    let p_left = Vec3::from_array(point);
    let p_right = Vec3::from_array(pose.transform_point(&point));
    if p_left.z <= 1.0e-6 || p_right.z <= 1.0e-6 {
        return None;
    }
    let pred1 = Vec3::new(p_left.x / p_left.z, p_left.y / p_left.z, 1.0);
    let pred2 = Vec3::new(p_right.x / p_right.z, p_right.y / p_right.z, 1.0);
    Some(SVector::<f32, 4>::new(
        pred1.x - p1[0],
        pred1.y - p1[1],
        pred2.x - p2[0],
        pred2.y - p2[1],
    ))
}

fn numerical_relative_reprojection_jacobian4(
    pose: SE3,
    p1: [f32; 2],
    p2: [f32; 2],
) -> Option<SMatrix<f32, 4, 6>> {
    let mut jacobian = SMatrix::<f32, 4, 6>::zeros();
    let eps = [1.0e-5, 1.0e-5, 1.0e-5, 1.0e-4, 1.0e-4, 1.0e-4];
    for axis in 0..6 {
        let mut plus = SVector::<f32, 6>::zeros();
        plus[axis] = eps[axis];
        let mut minus = SVector::<f32, 6>::zeros();
        minus[axis] = -eps[axis];
        let r_plus = relative_reprojection_residual4(perturb_relative_pose(pose, plus)?, p1, p2)?;
        let r_minus = relative_reprojection_residual4(perturb_relative_pose(pose, minus)?, p1, p2)?;
        for row in 0..4 {
            jacobian[(row, axis)] = (r_plus[row] - r_minus[row]) / (2.0 * eps[axis]);
        }
    }
    Some(jacobian)
}

fn triangulation_angle_deg(left_pose: SE3, right_pose: SE3, point: [f32; 3]) -> Option<f32> {
    let left_center = camera_center(left_pose);
    let right_center = camera_center(right_pose);
    let point = Vec3::from_array(point);
    let left_ray = (point - left_center).try_normalize()?;
    let right_ray = (point - right_center).try_normalize()?;
    let angle = left_ray.dot(right_ray).clamp(-1.0, 1.0).acos();
    Some(angle.min(std::f32::consts::PI - angle).to_degrees())
}

fn median_f32(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values[values.len() / 2]
}

fn sampson_cost(pose: SE3, pts1: &[[f32; 2]], pts2: &[[f32; 2]]) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0usize;
    for (&p1, &p2) in pts1.iter().zip(pts2.iter()) {
        let residual = sampson_residual(pose, p1, p2);
        if residual.is_finite() {
            let robust = residual.abs().min(1.0e-2);
            total += robust * robust;
            count += 1;
        }
    }
    if count == 0 {
        f32::INFINITY
    } else {
        total / count as f32
    }
}

fn sampson_residual(pose: SE3, p1: [f32; 2], p2: [f32; 2]) -> f32 {
    let (r, t) = relative_rotation_translation(pose);
    let e = skew(t) * r;
    let x1 = Vector3::new(p1[0], p1[1], 1.0);
    let x2 = Vector3::new(p2[0], p2[1], 1.0);
    let ex1 = e * x1;
    let etx2 = e.transpose() * x2;
    let numerator = x2.dot(&(e * x1));
    let denom = (ex1[0] * ex1[0] + ex1[1] * ex1[1] + etx2[0] * etx2[0] + etx2[1] * etx2[1]).sqrt();
    if denom <= 1.0e-12 {
        f32::INFINITY
    } else {
        numerator / denom
    }
}

fn numerical_sampson_jacobian(pose: SE3, p1: [f32; 2], p2: [f32; 2]) -> Option<SMatrix<f32, 1, 6>> {
    let mut jacobian = SMatrix::<f32, 1, 6>::zeros();
    let eps = [1.0e-5, 1.0e-5, 1.0e-5, 1.0e-4, 1.0e-4, 1.0e-4];
    for axis in 0..6 {
        let mut plus = SVector::<f32, 6>::zeros();
        plus[axis] = eps[axis];
        let mut minus = SVector::<f32, 6>::zeros();
        minus[axis] = -eps[axis];
        let r_plus = sampson_residual(perturb_relative_pose(pose, plus)?, p1, p2);
        let r_minus = sampson_residual(perturb_relative_pose(pose, minus)?, p1, p2);
        if !r_plus.is_finite() || !r_minus.is_finite() {
            return None;
        }
        jacobian[(0, axis)] = (r_plus - r_minus) / (2.0 * eps[axis]);
    }
    Some(jacobian)
}

fn perturb_relative_pose(pose: SE3, delta: SVector<f32, 6>) -> Option<SE3> {
    let rotation = pose_rotation(pose);
    let dr = Quat::from_scaled_axis(Vec3::new(delta[0], delta[1], delta[2]));
    let t = Vec3::from_array(pose.translation()) + Vec3::new(delta[3], delta[4], delta[5]);
    let t = t.try_normalize()?;
    Some(SE3::from_quat_translation((dr * rotation).normalize(), t))
}

fn normalize_relative_translation(pose: SE3) -> Option<SE3> {
    let t = Vec3::from_array(pose.translation()).try_normalize()?;
    Some(SE3::from_quat_translation(pose_rotation(pose), t))
}

fn relative_rotation_translation(pose: SE3) -> (Matrix3<f32>, Vector3<f32>) {
    let r = pose.rotation_matrix();
    let t = pose.translation();
    (
        Matrix3::from_row_slice(&[
            r[0][0], r[0][1], r[0][2], r[1][0], r[1][1], r[1][2], r[2][0], r[2][1], r[2][2],
        ]),
        Vector3::new(t[0], t[1], t[2]),
    )
}

fn skew(t: Vector3<f32>) -> Matrix3<f32> {
    Matrix3::new(0.0, -t[2], t[1], t[2], 0.0, -t[0], -t[1], t[0], 0.0)
}

fn huber_weight(err: f32, delta: f32) -> f32 {
    if err <= delta {
        1.0
    } else {
        delta / err.max(1.0e-12)
    }
}

pub fn mean_pair_reprojection_error(
    point: [f32; 3],
    left_pose: SE3,
    right_pose: SE3,
    left_xy: [f32; 2],
    right_xy: [f32; 2],
    camera: CameraModel,
) -> f32 {
    mean_pair_reprojection_error_with_cameras(
        point, left_pose, right_pose, left_xy, right_xy, camera, camera,
    )
}

pub fn mean_pair_reprojection_error_with_cameras(
    point: [f32; 3],
    left_pose: SE3,
    right_pose: SE3,
    left_xy: [f32; 2],
    right_xy: [f32; 2],
    left_camera: CameraModel,
    right_camera: CameraModel,
) -> f32 {
    0.5 * (reprojection_error_px(point, left_pose, left_xy, left_camera)
        + reprojection_error_px(point, right_pose, right_xy, right_camera))
}
