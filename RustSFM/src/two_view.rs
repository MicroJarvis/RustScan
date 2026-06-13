use crate::five_point::estimate_five_point_essential;
use crate::geometry::relative_rotation_deg;
use crate::types::CameraModel;
use glam::{Quat, Vec3};
use nalgebra::{DMatrix, Matrix3, Matrix3x4, Rotation3, UnitQuaternion, Vector3};
use rustslam::SE3;

#[derive(Debug, Clone)]
pub struct TwoViewOptions {
    pub ransac_threshold: f64,
    pub ransac_max_iterations: u32,
    pub min_inliers: usize,
    pub min_triangulated: usize,
    pub use_hartley_refinement: bool,
    pub use_five_point: bool,
}

#[derive(Debug, Clone)]
pub struct TwoViewEstimate {
    pub essential: Matrix3<f64>,
    pub inlier_mask: Vec<bool>,
    pub pose: SE3,
    pub triangulated: usize,
    pub mean_reprojection_error_px: f32,
    pub rotation_deg: f32,
    pub median_triangulation_angle_deg: f32,
}

#[derive(Debug, Clone)]
struct ModelSupport {
    inlier_mask: Vec<bool>,
    inliers: usize,
    mean_residual: f64,
    median_residual: f64,
}

#[derive(Debug, Clone)]
struct PoseCandidateScore {
    pose: SE3,
    triangulated: usize,
    mean_reprojection_error_px: f32,
    median_angle_deg: f64,
}

pub fn estimate_calibrated_two_view(
    pts1: &[[f32; 2]],
    pts2: &[[f32; 2]],
    camera: CameraModel,
    options: &TwoViewOptions,
) -> Option<TwoViewEstimate> {
    let obs1_px = normalized_observations_to_pixels(pts1, camera);
    let obs2_px = normalized_observations_to_pixels(pts2, camera);
    estimate_calibrated_two_view_with_observations(pts1, pts2, &obs1_px, &obs2_px, camera, options)
}

pub fn estimate_calibrated_two_view_with_observations(
    pts1: &[[f32; 2]],
    pts2: &[[f32; 2]],
    obs1_px: &[[f32; 2]],
    obs2_px: &[[f32; 2]],
    camera: CameraModel,
    options: &TwoViewOptions,
) -> Option<TwoViewEstimate> {
    let n = pts1.len().min(pts2.len());
    if n < options.min_inliers.max(5) {
        return None;
    }
    let obs1_px = if obs1_px.len() >= n { obs1_px } else { pts1 };
    let obs2_px = if obs2_px.len() >= n { obs2_px } else { pts2 };

    let pts1 = pts1
        .iter()
        .take(n)
        .map(|p| Vector3::new(p[0] as f64, p[1] as f64, 1.0))
        .collect::<Vec<_>>();
    let pts2 = pts2
        .iter()
        .take(n)
        .map(|p| Vector3::new(p[0] as f64, p[1] as f64, 1.0))
        .collect::<Vec<_>>();
    let support_limit = ransac_support_limit();
    let support_indices = if n > support_limit {
        (0..support_limit)
            .map(|k| k * n / support_limit)
            .collect::<Vec<_>>()
    } else {
        (0..n).collect::<Vec<_>>()
    };

    let mut rng = Lcg::new(0x9e37_79b9_7f4a_7c15 ^ n as u64);
    let mut best: Option<(Matrix3<f64>, ModelSupport)> = None;
    let mut max_iterations = options.ransac_max_iterations.max(1);
    let mut iteration = 0u32;
    while iteration < max_iterations {
        iteration += 1;
        let (sample_size, models) = if options.use_five_point {
            let sample = rng.sample_unique(n, 5);
            if sample.len() != 5 {
                continue;
            }
            (
                5,
                estimate_essential_five_point_indexed(&pts1, &pts2, &sample),
            )
        } else {
            let sample = rng.sample_unique(n, 8);
            if sample.len() != 8 {
                continue;
            }
            let model = estimate_essential_eight_point_indexed_lightweight(&pts1, &pts2, &sample);
            (8, model.into_iter().collect::<Vec<_>>())
        };
        if models.is_empty() {
            continue;
        }
        for model in models {
            let support = model_support_indexed(
                &pts1,
                &pts2,
                &support_indices,
                &model,
                options.ransac_threshold,
            );
            if support.inliers < 5 {
                continue;
            }
            if is_better_support(&support, best.as_ref().map(|(_, s)| s)) {
                let estimated_inliers =
                    (support.inliers * n / support_indices.len().max(1)).clamp(0, n);
                max_iterations = max_iterations.min(adaptive_ransac_iterations(
                    estimated_inliers,
                    n,
                    options.ransac_max_iterations,
                    0.999,
                    sample_size,
                ));
                best = Some((model, support));
            }
        }
    }

    let (mut essential, _) = best.or_else(|| {
        estimate_essential_eight_point(&pts1, &pts2).map(|model| {
            let support = model_support(&pts1, &pts2, &model, options.ransac_threshold);
            (model, support)
        })
    })?;
    let mut support = model_support(&pts1, &pts2, &essential, options.ransac_threshold);

    for _ in 0..6 {
        if support.inliers < 8 {
            break;
        }
        let inliers = support
            .inlier_mask
            .iter()
            .enumerate()
            .filter_map(|(idx, &is_inlier)| is_inlier.then_some(idx))
            .collect::<Vec<_>>();
        let sampled_inliers = sample_indices_evenly(&inliers, essential_refit_inlier_limit());
        let refined = if options.use_hartley_refinement {
            estimate_essential_eight_point_indexed(&pts1, &pts2, &sampled_inliers)
        } else {
            estimate_essential_eight_point_indexed_lightweight(&pts1, &pts2, &sampled_inliers)
        };
        let Some(refined) = refined else { break };
        let refined_support = model_support(&pts1, &pts2, &refined, options.ransac_threshold);
        if is_better_support(&refined_support, Some(&support)) {
            essential = refined;
            support = refined_support;
        } else {
            break;
        }
    }

    if support.inliers < options.min_inliers {
        return None;
    }

    let pose_score = choose_pose_from_essential(
        &essential,
        &pts1,
        &pts2,
        obs1_px,
        obs2_px,
        &support.inlier_mask,
        camera,
    )?;
    if pose_score.triangulated < options.min_triangulated {
        return None;
    }

    Some(TwoViewEstimate {
        essential,
        inlier_mask: support.inlier_mask,
        pose: pose_score.pose,
        triangulated: pose_score.triangulated,
        mean_reprojection_error_px: pose_score.mean_reprojection_error_px,
        rotation_deg: relative_rotation_deg(pose_score.pose, SE3::identity()),
        median_triangulation_angle_deg: pose_score.median_angle_deg as f32,
    })
}

fn ransac_support_limit() -> usize {
    std::env::var("RUSTSFM_RANSAC_SUPPORT_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 32)
        .unwrap_or(512)
}

fn normalized_observations_to_pixels(pts: &[[f32; 2]], camera: CameraModel) -> Vec<[f32; 2]> {
    pts.iter()
        .map(|p| camera.img_from_cam_f32(p[0], p[1], 1.0).unwrap_or(*p))
        .collect()
}

fn essential_refit_inlier_limit() -> usize {
    std::env::var("RUSTSFM_ESSENTIAL_REFIT_INLIERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 8)
        .unwrap_or(512)
}

fn sample_indices_evenly(indices: &[usize], max_count: usize) -> Vec<usize> {
    if indices.len() <= max_count || max_count == 0 {
        return indices.to_vec();
    }
    let mut sampled = Vec::with_capacity(max_count);
    for k in 0..max_count {
        let idx = k * indices.len() / max_count;
        sampled.push(indices[idx]);
    }
    sampled.dedup();
    sampled
}

pub(crate) fn triangulate_relative_pose_point(
    pose: SE3,
    left_xy: [f32; 2],
    right_xy: [f32; 2],
) -> Option<[f32; 3]> {
    let r = pose.rotation_matrix();
    let t = pose.translation();
    let r = Matrix3::from_row_slice(&[
        r[0][0] as f64,
        r[0][1] as f64,
        r[0][2] as f64,
        r[1][0] as f64,
        r[1][1] as f64,
        r[1][2] as f64,
        r[2][0] as f64,
        r[2][1] as f64,
        r[2][2] as f64,
    ]);
    let t = Vector3::new(t[0] as f64, t[1] as f64, t[2] as f64);
    let x1 = Vector3::new(left_xy[0] as f64, left_xy[1] as f64, 1.0);
    let x2 = Vector3::new(right_xy[0] as f64, right_xy[1] as f64, 1.0);
    triangulate_normalized_pair(&r, &t, &x1, &x2)
        .map(|point| [point.x as f32, point.y as f32, point.z as f32])
}

pub(crate) fn triangulate_world_point(
    left_pose: SE3,
    right_pose: SE3,
    left_xy: [f32; 2],
    right_xy: [f32; 2],
) -> Option<[f32; 3]> {
    let relative_pose = right_pose.compose(&left_pose.inverse());
    let point_left = triangulate_relative_pose_point(relative_pose, left_xy, right_xy)?;
    Some(left_pose.inverse().transform_point(&point_left))
}

fn is_better_support(candidate: &ModelSupport, current: Option<&ModelSupport>) -> bool {
    let Some(current) = current else {
        return true;
    };
    candidate.inliers > current.inliers
        || (candidate.inliers == current.inliers
            && candidate.median_residual < current.median_residual)
        || (candidate.inliers == current.inliers
            && (candidate.median_residual - current.median_residual).abs() < 1.0e-12
            && candidate.mean_residual < current.mean_residual)
}

fn adaptive_ransac_iterations(
    inliers: usize,
    total: usize,
    max_iterations: u32,
    confidence: f64,
    sample_size: usize,
) -> u32 {
    if inliers == 0 || total == 0 || inliers >= total {
        return 1;
    }
    let inlier_ratio = inliers as f64 / total as f64;
    let success_prob = inlier_ratio
        .powi(sample_size as i32)
        .clamp(1.0e-12, 1.0 - 1.0e-12);
    let denom = (1.0 - success_prob).ln();
    if denom >= 0.0 {
        max_iterations
    } else {
        ((1.0 - confidence).ln() / denom)
            .ceil()
            .clamp(1.0, max_iterations as f64) as u32
    }
}

fn estimate_essential_eight_point(
    pts1: &[Vector3<f64>],
    pts2: &[Vector3<f64>],
) -> Option<Matrix3<f64>> {
    let indices = (0..pts1.len().min(pts2.len())).collect::<Vec<_>>();
    estimate_essential_eight_point_indexed(pts1, pts2, &indices)
}

fn estimate_essential_five_point_indexed(
    pts1: &[Vector3<f64>],
    pts2: &[Vector3<f64>],
    indices: &[usize],
) -> Vec<Matrix3<f64>> {
    if indices.len() < 5 {
        return Vec::new();
    }
    let mut rays1 = Vec::with_capacity(indices.len());
    let mut rays2 = Vec::with_capacity(indices.len());
    for &idx in indices {
        let Some(x1) = pts1.get(idx) else {
            return Vec::new();
        };
        let Some(x2) = pts2.get(idx) else {
            return Vec::new();
        };
        rays1.push(*x1);
        rays2.push(*x2);
    }
    estimate_five_point_essential(&rays1, &rays2)
}

fn estimate_essential_eight_point_indexed(
    pts1: &[Vector3<f64>],
    pts2: &[Vector3<f64>],
    indices: &[usize],
) -> Option<Matrix3<f64>> {
    if indices.len() < 8 {
        return None;
    }
    let (norm1, t1) = normalize_points_indexed(pts1, indices)?;
    let (norm2, t2) = normalize_points_indexed(pts2, indices)?;
    let mut rows = Vec::with_capacity(indices.len() * 9);
    for (x1, x2) in norm1.iter().zip(norm2.iter()) {
        rows.extend_from_slice(&[
            x2.x * x1.x,
            x2.x * x1.y,
            x2.x * x1.z,
            x2.y * x1.x,
            x2.y * x1.y,
            x2.y * x1.z,
            x2.z * x1.x,
            x2.z * x1.y,
            x2.z * x1.z,
        ]);
    }
    let a = DMatrix::<f64>::from_row_slice(indices.len(), 9, &rows);
    let svd = a.svd(false, true);
    let vt = svd.v_t?;
    let q = vt.row(vt.nrows() - 1);
    let e_norm = Matrix3::from_row_slice(&[q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8]]);
    let e = t2.transpose() * e_norm * t1;
    enforce_essential_constraints(e)
}

fn estimate_essential_eight_point_indexed_lightweight(
    pts1: &[Vector3<f64>],
    pts2: &[Vector3<f64>],
    indices: &[usize],
) -> Option<Matrix3<f64>> {
    if indices.len() < 8 {
        return None;
    }
    let mut rows = Vec::with_capacity(indices.len() * 9);
    for &idx in indices {
        let x1 = pts1.get(idx)?;
        let x2 = pts2.get(idx)?;
        rows.extend_from_slice(&[
            x2.x * x1.x,
            x2.x * x1.y,
            x2.x * x1.z,
            x2.y * x1.x,
            x2.y * x1.y,
            x2.y * x1.z,
            x2.z * x1.x,
            x2.z * x1.y,
            x2.z * x1.z,
        ]);
    }
    let a = DMatrix::<f64>::from_row_slice(indices.len(), 9, &rows);
    let svd = a.svd(false, true);
    let vt = svd.v_t?;
    let q = vt.row(vt.nrows() - 1);
    let e = Matrix3::from_row_slice(&[q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8]]);
    enforce_essential_constraints(e)
}

fn normalize_points_indexed(
    pts: &[Vector3<f64>],
    indices: &[usize],
) -> Option<(Vec<Vector3<f64>>, Matrix3<f64>)> {
    if indices.is_empty() {
        return None;
    }
    let mut centroid = Vector3::<f64>::zeros();
    for &idx in indices {
        let p = pts.get(idx)?;
        centroid.x += p.x / p.z;
        centroid.y += p.y / p.z;
    }
    centroid.x /= indices.len() as f64;
    centroid.y /= indices.len() as f64;

    let mut mean_dist = 0.0f64;
    for &idx in indices {
        let p = pts.get(idx)?;
        let x = p.x / p.z - centroid.x;
        let y = p.y / p.z - centroid.y;
        mean_dist += (x * x + y * y).sqrt();
    }
    mean_dist /= indices.len() as f64;
    let scale = std::f64::consts::SQRT_2 / mean_dist.max(1.0e-12);
    let transform = Matrix3::new(
        scale,
        0.0,
        -scale * centroid.x,
        0.0,
        scale,
        -scale * centroid.y,
        0.0,
        0.0,
        1.0,
    );
    let normalized = indices
        .iter()
        .map(|&idx| transform * pts[idx])
        .collect::<Vec<_>>();
    Some((normalized, transform))
}

fn enforce_essential_constraints(e: Matrix3<f64>) -> Option<Matrix3<f64>> {
    let svd = e.svd(true, true);
    let mut u = svd.u?;
    let mut vt = svd.v_t?;
    if u.determinant() < 0.0 {
        u.column_mut(2).scale_mut(-1.0);
    }
    if vt.determinant() < 0.0 {
        vt.row_mut(2).scale_mut(-1.0);
    }
    let s = (svd.singular_values[0] + svd.singular_values[1]) * 0.5;
    let sigma = Matrix3::from_diagonal(&Vector3::new(s, s, 0.0));
    let e = u * sigma * vt;
    let norm = e.norm();
    (norm > 1.0e-12 && norm.is_finite()).then_some(e / norm)
}

fn model_support(
    pts1: &[Vector3<f64>],
    pts2: &[Vector3<f64>],
    essential: &Matrix3<f64>,
    threshold: f64,
) -> ModelSupport {
    let indices = (0..pts1.len().min(pts2.len())).collect::<Vec<_>>();
    model_support_indexed(pts1, pts2, &indices, essential, threshold)
}

fn model_support_indexed(
    pts1: &[Vector3<f64>],
    pts2: &[Vector3<f64>],
    indices: &[usize],
    essential: &Matrix3<f64>,
    threshold: f64,
) -> ModelSupport {
    let threshold_sq = threshold.max(1.0e-12).powi(2);
    let n = pts1.len().min(pts2.len());
    let mut inlier_mask = vec![false; n];
    let mut residuals = Vec::new();
    for &idx in indices {
        if idx >= n {
            continue;
        }
        let residual = squared_sampson_error(&pts1[idx], &pts2[idx], essential);
        if residual.is_finite() && residual <= threshold_sq {
            inlier_mask[idx] = true;
            residuals.push(residual);
        }
    }
    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let inliers = residuals.len();
    let mean_residual = if inliers > 0 {
        residuals.iter().sum::<f64>() / inliers as f64
    } else {
        f64::INFINITY
    };
    let median_residual = residuals
        .get(inliers.saturating_sub(1) / 2)
        .copied()
        .unwrap_or(f64::INFINITY);
    ModelSupport {
        inlier_mask,
        inliers,
        mean_residual,
        median_residual,
    }
}

fn squared_sampson_error(x1: &Vector3<f64>, x2: &Vector3<f64>, essential: &Matrix3<f64>) -> f64 {
    let ex1 = essential * x1;
    let etx2 = essential.transpose() * x2;
    let num = x2.dot(&(essential * x1));
    let denom = ex1.x * ex1.x + ex1.y * ex1.y + etx2.x * etx2.x + etx2.y * etx2.y;
    if denom <= 1.0e-24 {
        f64::INFINITY
    } else {
        num * num / denom
    }
}

fn choose_pose_from_essential(
    essential: &Matrix3<f64>,
    pts1: &[Vector3<f64>],
    pts2: &[Vector3<f64>],
    obs1_px: &[[f32; 2]],
    obs2_px: &[[f32; 2]],
    inlier_mask: &[bool],
    camera: CameraModel,
) -> Option<PoseCandidateScore> {
    let candidates = decompose_essential_matrix(essential)?;
    candidates
        .into_iter()
        .filter_map(|(r, t)| {
            let pose = se3_from_parts(&r, &t)?;
            let mut reproj_sum = 0.0f64;
            let mut triangulated = 0usize;
            let mut angles = Vec::new();
            for (idx, &is_inlier) in inlier_mask.iter().enumerate() {
                if !is_inlier {
                    continue;
                }
                let Some(point) = triangulate_normalized_pair(&r, &t, &pts1[idx], &pts2[idx])
                else {
                    continue;
                };
                let point2 = r * point + t;
                if point.z <= 1.0e-8 || point2.z <= 1.0e-8 {
                    continue;
                }
                let err = pair_reprojection_error_px(
                    &point,
                    &point2,
                    obs1_px
                        .get(idx)
                        .copied()
                        .unwrap_or([pts1[idx].x as f32, pts1[idx].y as f32]),
                    obs2_px
                        .get(idx)
                        .copied()
                        .unwrap_or([pts2[idx].x as f32, pts2[idx].y as f32]),
                    camera,
                );
                if !err.is_finite() || err > 16.0 {
                    continue;
                }
                let angle =
                    triangulation_angle_deg(&Vector3::zeros(), &(-r.transpose() * t), &point);
                if !angle.is_finite() {
                    continue;
                }
                reproj_sum += err;
                angles.push(angle);
                triangulated += 1;
            }
            if triangulated == 0 {
                return None;
            }
            angles.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            Some(PoseCandidateScore {
                pose,
                triangulated,
                mean_reprojection_error_px: (reproj_sum / triangulated as f64) as f32,
                median_angle_deg: angles[angles.len() / 2],
            })
        })
        .max_by(|a, b| compare_pose_scores(a, b))
}

fn compare_pose_scores(a: &PoseCandidateScore, b: &PoseCandidateScore) -> std::cmp::Ordering {
    a.triangulated
        .cmp(&b.triangulated)
        .then_with(|| {
            b.mean_reprojection_error_px
                .partial_cmp(&a.mean_reprojection_error_px)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| {
            a.median_angle_deg
                .partial_cmp(&b.median_angle_deg)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

fn decompose_essential_matrix(
    essential: &Matrix3<f64>,
) -> Option<Vec<(Matrix3<f64>, Vector3<f64>)>> {
    let svd = essential.svd(true, true);
    let mut u = svd.u?;
    let mut vt = svd.v_t?;
    if u.determinant() < 0.0 {
        u.column_mut(2).scale_mut(-1.0);
    }
    if vt.determinant() < 0.0 {
        vt.row_mut(2).scale_mut(-1.0);
    }
    let w = Matrix3::new(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let mut r1 = u * w * vt;
    let mut r2 = u * w.transpose() * vt;
    if r1.determinant() < 0.0 {
        r1 *= -1.0;
    }
    if r2.determinant() < 0.0 {
        r2 *= -1.0;
    }
    let t = u.column(2).into_owned().normalize();
    Some(vec![(r1, t), (r2, t), (r1, -t), (r2, -t)])
}

fn triangulate_normalized_pair(
    r: &Matrix3<f64>,
    t: &Vector3<f64>,
    x1: &Vector3<f64>,
    x2: &Vector3<f64>,
) -> Option<Vector3<f64>> {
    let p1 = Matrix3x4::<f64>::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    let p2 = Matrix3x4::<f64>::new(
        r[(0, 0)],
        r[(0, 1)],
        r[(0, 2)],
        t.x,
        r[(1, 0)],
        r[(1, 1)],
        r[(1, 2)],
        t.y,
        r[(2, 0)],
        r[(2, 1)],
        r[(2, 2)],
        t.z,
    );
    let mut rows = Vec::with_capacity(16);
    for col in 0..4 {
        rows.push(x1.x * p1[(2, col)] - p1[(0, col)]);
    }
    for col in 0..4 {
        rows.push(x1.y * p1[(2, col)] - p1[(1, col)]);
    }
    for col in 0..4 {
        rows.push(x2.x * p2[(2, col)] - p2[(0, col)]);
    }
    for col in 0..4 {
        rows.push(x2.y * p2[(2, col)] - p2[(1, col)]);
    }
    let a = nalgebra::Matrix4::<f64>::from_row_slice(&rows);
    let svd = a.svd(false, true);
    let vt = svd.v_t?;
    let x = vt.row(3);
    if x[3].abs() <= 1.0e-12 || !x[3].is_finite() {
        return None;
    }
    let point = Vector3::new(x[0] / x[3], x[1] / x[3], x[2] / x[3]);
    point.iter().all(|v| v.is_finite()).then_some(point)
}

fn pair_reprojection_error_px(
    point1: &Vector3<f64>,
    point2: &Vector3<f64>,
    observation1_px: [f32; 2],
    observation2_px: [f32; 2],
    camera: CameraModel,
) -> f64 {
    let err1 = camera_reprojection_error_px(point1, observation1_px, camera);
    let err2 = camera_reprojection_error_px(point2, observation2_px, camera);
    0.5 * (err1 + err2)
}

fn camera_reprojection_error_px(
    point: &Vector3<f64>,
    observation_px: [f32; 2],
    camera: CameraModel,
) -> f64 {
    let Some(predicted) = camera.img_from_cam(point.x, point.y, point.z) else {
        return f64::INFINITY;
    };
    ((predicted[0] - observation_px[0] as f64).powi(2)
        + (predicted[1] - observation_px[1] as f64).powi(2))
    .sqrt()
}

fn triangulation_angle_deg(
    center1: &Vector3<f64>,
    center2: &Vector3<f64>,
    point: &Vector3<f64>,
) -> f64 {
    let v1 = point - center1;
    let v2 = point - center2;
    let denom = v1.norm() * v2.norm();
    if denom <= 1.0e-12 {
        return f64::INFINITY;
    }
    let angle = (v1.dot(&v2) / denom).clamp(-1.0, 1.0).acos();
    angle.min(std::f64::consts::PI - angle).to_degrees()
}

fn se3_from_parts(r: &Matrix3<f64>, t: &Vector3<f64>) -> Option<SE3> {
    let rotation = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(*r));
    let q = rotation.into_inner();
    let quat = Quat::from_xyzw(q.i as f32, q.j as f32, q.k as f32, q.w as f32).normalize();
    let translation = Vec3::new(t.x as f32, t.y as f32, t.z as f32);
    (translation.is_finite() && quat.is_finite())
        .then_some(SE3::from_quat_translation(quat, translation))
}

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }

    fn gen_range(&mut self, max: usize) -> usize {
        if max == 0 {
            0
        } else {
            (self.next_u32() as usize) % max
        }
    }

    fn sample_unique(&mut self, n: usize, k: usize) -> Vec<usize> {
        if k > n {
            return Vec::new();
        }
        let mut sample = Vec::with_capacity(k);
        while sample.len() < k {
            let idx = self.gen_range(n);
            if !sample.contains(&idx) {
                sample.push(idx);
            }
        }
        sample
    }
}
