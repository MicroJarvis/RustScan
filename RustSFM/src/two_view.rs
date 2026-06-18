use crate::five_point::estimate_five_point_essential;
use crate::geometry::relative_rotation_deg;
use crate::types::CameraModel;
use glam::{Quat, Vec3};
use nalgebra::{
    DMatrix, DVector, Matrix3, Matrix3x4, Rotation3, SymmetricEigen, UnitQuaternion, Vector3,
};
use rustslam::SE3;

#[derive(Debug, Clone)]
pub struct TwoViewOptions {
    pub ransac_max_error_px: f64,
    pub ransac_threshold: f64,
    pub ransac_max_iterations: u32,
    pub random_seed: u64,
    pub loransac_num_lo_steps: usize,
    pub min_inliers: usize,
    pub min_inlier_ratio: f64,
    pub min_triangulated: usize,
    pub min_e_f_inlier_ratio: f64,
    pub max_h_inlier_ratio: f64,
    pub force_h_use: bool,
    pub multiple_models: bool,
    pub multiple_ignore_watermark: bool,
    pub detect_watermark: bool,
    pub watermark_min_inlier_ratio: f64,
    pub watermark_border_size: f64,
    pub watermark_detection_max_error_px: f64,
    pub filter_stationary_matches: bool,
    pub stationary_matches_max_error_px: f64,
    pub use_hartley_refinement: bool,
    pub use_five_point: bool,
}

#[derive(Debug, Clone)]
pub struct TwoViewEstimate {
    pub essential: Matrix3<f64>,
    pub two_view_config: i32,
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
    residual_sum: f64,
}

#[derive(Debug, Clone)]
struct PoseCandidateScore {
    pose: SE3,
    triangulated: usize,
    mean_reprojection_error_px: f32,
    median_angle_deg: f64,
}

const COLMAP_RANSAC_DYN_NUM_TRIALS_MULTIPLIER: f64 = 3.0;

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
    estimate_calibrated_two_view_with_observations_and_cameras(
        pts1, pts2, obs1_px, obs2_px, camera, camera, options,
    )
}

pub fn estimate_calibrated_two_view_with_observations_and_cameras(
    pts1: &[[f32; 2]],
    pts2: &[[f32; 2]],
    obs1_px: &[[f32; 2]],
    obs2_px: &[[f32; 2]],
    camera1: CameraModel,
    camera2: CameraModel,
    options: &TwoViewOptions,
) -> Option<TwoViewEstimate> {
    let n = pts1.len().min(pts2.len());
    if n < options.min_inliers.max(5) {
        return None;
    }
    let obs1_px = if obs1_px.len() >= n { obs1_px } else { pts1 };
    let obs2_px = if obs2_px.len() >= n { obs2_px } else { pts2 };
    let active_indices = active_match_indices(
        obs1_px,
        obs2_px,
        n,
        options.filter_stationary_matches,
        options.stationary_matches_max_error_px,
    );
    if active_indices.len() < options.min_inliers.max(5) {
        return None;
    }
    if options.multiple_models {
        return estimate_multiple_calibrated_two_view_with_observations_and_cameras(
            pts1,
            pts2,
            obs1_px,
            obs2_px,
            camera1,
            camera2,
            &active_indices,
            options,
        );
    }

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
    let img_pts1 = obs1_px
        .iter()
        .take(n)
        .map(|p| Vector3::new(p[0] as f64, p[1] as f64, 1.0))
        .collect::<Vec<_>>();
    let img_pts2 = obs2_px
        .iter()
        .take(n)
        .map(|p| Vector3::new(p[0] as f64, p[1] as f64, 1.0))
        .collect::<Vec<_>>();
    if options.force_h_use {
        return estimate_force_h_two_view(
            &pts1,
            &pts2,
            &img_pts1,
            &img_pts2,
            obs1_px,
            obs2_px,
            &active_indices,
            camera1,
            camera2,
            options,
        );
    }
    let support_limit = ransac_support_limit();
    let support_indices = if active_indices.len() > support_limit {
        (0..support_limit)
            .map(|k| active_indices[k * active_indices.len() / support_limit])
            .collect::<Vec<_>>()
    } else {
        active_indices.clone()
    };

    let mut sampler = ColmapRandomSampler::new(
        options.random_seed ^ 0x9e37_79b9_7f4a_7c15 ^ n as u64,
        &active_indices,
    );
    let mut best: Option<(Matrix3<f64>, ModelSupport)> = None;
    let mut max_iterations = options.ransac_max_iterations.max(1);
    let mut iteration = 0u32;
    while iteration < max_iterations {
        iteration += 1;
        let (sample_size, models) = if options.use_five_point {
            let sample = sampler.sample(5);
            if sample.len() != 5 {
                continue;
            }
            (
                5,
                estimate_essential_five_point_indexed(&pts1, &pts2, &sample),
            )
        } else {
            let sample = sampler.sample(8);
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
                let estimated_inliers = (support.inliers * active_indices.len()
                    / support_indices.len().max(1))
                .clamp(0, n);
                max_iterations = max_iterations.min(adaptive_ransac_iterations(
                    estimated_inliers,
                    active_indices.len(),
                    options.ransac_max_iterations,
                    0.999,
                    sample_size,
                ));
                best = Some((model, support));
            }
        }
    }

    let (mut essential, _) = best.or_else(|| {
        estimate_essential_eight_point_indexed(&pts1, &pts2, &active_indices).map(|model| {
            let support = model_support_indexed(
                &pts1,
                &pts2,
                &active_indices,
                &model,
                options.ransac_threshold,
            );
            (model, support)
        })
    })?;
    let mut support = model_support_indexed(
        &pts1,
        &pts2,
        &active_indices,
        &essential,
        options.ransac_threshold,
    );

    for _ in 0..options.loransac_num_lo_steps {
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
        let refined_support = model_support_indexed(
            &pts1,
            &pts2,
            &active_indices,
            &refined,
            options.ransac_threshold,
        );
        if is_better_support(&refined_support, Some(&support)) {
            essential = refined;
            support = refined_support;
        } else {
            break;
        }
    }

    let f_support = estimate_fundamental_ransac(
        &img_pts1,
        &img_pts2,
        &active_indices,
        options.ransac_max_error_px,
        options.ransac_max_iterations,
        options.random_seed,
        options.loransac_num_lo_steps,
    );
    let h_support = estimate_homography_ransac(
        &img_pts1,
        &img_pts2,
        &active_indices,
        options.ransac_max_error_px,
        options.ransac_max_iterations,
        options.random_seed,
        options.loransac_num_lo_steps,
    );
    let Some((mut two_view_config, selected_mask, selected_inliers)) = classify_calibrated_two_view(
        &support,
        f_support.as_ref().map(|(_, support)| support),
        h_support.as_ref().map(|(_, support)| support),
        options,
    ) else {
        return None;
    };
    if options.min_inlier_ratio > 0.0
        && selected_inliers as f64 / (active_indices.len().max(1) as f64) < options.min_inlier_ratio
    {
        return None;
    }
    if two_view_config == crate::database::COLMAP_TWO_VIEW_PLANAR_OR_PANORAMIC {
        if let Some((homography, _)) = h_support.as_ref() {
            two_view_config = classify_homography_motion(homography, camera1, camera2);
        }
    }
    if options.detect_watermark
        && detect_watermark_matches(
            camera1,
            camera2,
            obs1_px,
            obs2_px,
            &selected_mask,
            selected_inliers,
            options,
        )
    {
        two_view_config = crate::database::COLMAP_TWO_VIEW_WATERMARK;
    }

    let (pose_essential, pose_mask) = pose_essential_and_mask(
        essential,
        f_support.as_ref().map(|(model, _)| model),
        &support,
        &selected_mask,
        selected_inliers,
        two_view_config,
        &pts1,
        &pts2,
        &active_indices,
        camera1,
        camera2,
        options,
    );
    let pose_score = choose_pose_from_essential(
        &pose_essential,
        &pts1,
        &pts2,
        obs1_px,
        obs2_px,
        &pose_mask,
        camera1,
        camera2,
    )?;
    if pose_score.triangulated < options.min_triangulated {
        return None;
    }

    Some(TwoViewEstimate {
        essential: pose_essential,
        two_view_config,
        inlier_mask: selected_mask,
        pose: pose_score.pose,
        triangulated: pose_score.triangulated,
        mean_reprojection_error_px: pose_score.mean_reprojection_error_px,
        rotation_deg: relative_rotation_deg(pose_score.pose, SE3::identity()),
        median_triangulation_angle_deg: pose_score.median_angle_deg as f32,
    })
}

#[allow(clippy::too_many_arguments)]
fn estimate_force_h_two_view(
    pts1: &[Vector3<f64>],
    pts2: &[Vector3<f64>],
    img_pts1: &[Vector3<f64>],
    img_pts2: &[Vector3<f64>],
    obs1_px: &[[f32; 2]],
    obs2_px: &[[f32; 2]],
    active_indices: &[usize],
    camera1: CameraModel,
    camera2: CameraModel,
    options: &TwoViewOptions,
) -> Option<TwoViewEstimate> {
    let (homography, h_support) = estimate_homography_ransac(
        img_pts1,
        img_pts2,
        active_indices,
        options.ransac_max_error_px,
        options.ransac_max_iterations,
        options.random_seed,
        options.loransac_num_lo_steps,
    )?;
    if h_support.inliers < options.min_inliers {
        return None;
    }
    if options.min_inlier_ratio > 0.0
        && h_support.inliers as f64 / (active_indices.len().max(1) as f64)
            < options.min_inlier_ratio
    {
        return None;
    }

    let mut two_view_config = classify_homography_motion(&homography, camera1, camera2);
    if options.detect_watermark
        && detect_watermark_matches(
            camera1,
            camera2,
            obs1_px,
            obs2_px,
            &h_support.inlier_mask,
            h_support.inliers,
            options,
        )
    {
        two_view_config = crate::database::COLMAP_TWO_VIEW_WATERMARK;
    }

    let inlier_indices = h_support
        .inlier_mask
        .iter()
        .enumerate()
        .filter_map(|(idx, &is_inlier)| is_inlier.then_some(idx))
        .collect::<Vec<_>>();
    let essential = estimate_essential_eight_point_indexed(pts1, pts2, &inlier_indices)
        .or_else(|| estimate_essential_eight_point_indexed(pts1, pts2, active_indices))?;
    let essential_support = model_support_indexed(
        pts1,
        pts2,
        active_indices,
        &essential,
        options.ransac_threshold,
    );
    let pose_mask = if essential_support.inliers >= options.min_inliers {
        essential_support.inlier_mask
    } else {
        h_support.inlier_mask.clone()
    };
    let pose_score = choose_pose_from_essential(
        &essential, pts1, pts2, obs1_px, obs2_px, &pose_mask, camera1, camera2,
    )?;
    if pose_score.triangulated < options.min_triangulated {
        return None;
    }

    Some(TwoViewEstimate {
        essential,
        two_view_config,
        inlier_mask: h_support.inlier_mask,
        pose: pose_score.pose,
        triangulated: pose_score.triangulated,
        mean_reprojection_error_px: pose_score.mean_reprojection_error_px,
        rotation_deg: relative_rotation_deg(pose_score.pose, SE3::identity()),
        median_triangulation_angle_deg: pose_score.median_angle_deg as f32,
    })
}

#[allow(clippy::too_many_arguments)]
fn estimate_multiple_calibrated_two_view_with_observations_and_cameras(
    pts1: &[[f32; 2]],
    pts2: &[[f32; 2]],
    obs1_px: &[[f32; 2]],
    obs2_px: &[[f32; 2]],
    camera1: CameraModel,
    camera2: CameraModel,
    active_indices: &[usize],
    options: &TwoViewOptions,
) -> Option<TwoViewEstimate> {
    let n = pts1.len().min(pts2.len());
    let mut remaining = active_indices.to_vec();
    let mut models = Vec::<TwoViewEstimate>::new();
    let max_models = multiple_model_limit();

    for model_idx in 0..max_models {
        if remaining.len() < options.min_inliers.max(5) {
            break;
        }

        let sub_pts1 = remaining.iter().map(|&idx| pts1[idx]).collect::<Vec<_>>();
        let sub_pts2 = remaining.iter().map(|&idx| pts2[idx]).collect::<Vec<_>>();
        let sub_obs1 = remaining
            .iter()
            .map(|&idx| obs1_px[idx])
            .collect::<Vec<_>>();
        let sub_obs2 = remaining
            .iter()
            .map(|&idx| obs2_px[idx])
            .collect::<Vec<_>>();

        let mut sub_options = options.clone();
        sub_options.multiple_models = false;
        sub_options.filter_stationary_matches = false;
        sub_options.random_seed = options.random_seed
            ^ 0x6a09_e667_f3bc_c909
            ^ ((model_idx as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));

        let mut estimate = match estimate_calibrated_two_view_with_observations_and_cameras(
            &sub_pts1,
            &sub_pts2,
            &sub_obs1,
            &sub_obs2,
            camera1,
            camera2,
            &sub_options,
        ) {
            Some(estimate) => estimate,
            None => break,
        };

        let mut full_mask = vec![false; n];
        let mut inlier_count = 0usize;
        for (sub_idx, &is_inlier) in estimate.inlier_mask.iter().enumerate() {
            if !is_inlier {
                continue;
            }
            let Some(&global_idx) = remaining.get(sub_idx) else {
                continue;
            };
            full_mask[global_idx] = true;
            inlier_count += 1;
        }
        if inlier_count < options.min_inliers {
            break;
        }

        remaining.retain(|&idx| !full_mask[idx]);
        estimate.inlier_mask = full_mask;
        if options.multiple_ignore_watermark
            && estimate.two_view_config == crate::database::COLMAP_TWO_VIEW_WATERMARK
        {
            continue;
        }
        models.push(estimate);
    }

    if models.is_empty() {
        return None;
    }
    if models.len() == 1 {
        return models.into_iter().next();
    }

    let best_idx = models
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| compare_multiple_model_estimates(a, b))
        .map(|(idx, _)| idx)?;
    let mut best = models.swap_remove(best_idx);
    let mut union_mask = best.inlier_mask.clone();
    for model in &models {
        for (idx, &is_inlier) in model.inlier_mask.iter().enumerate() {
            union_mask[idx] |= is_inlier;
        }
    }
    best.two_view_config = crate::database::COLMAP_TWO_VIEW_MULTIPLE;
    best.inlier_mask = union_mask;
    Some(best)
}

fn multiple_model_limit() -> usize {
    std::env::var("RUSTSFM_MULTIPLE_MODEL_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(8)
}

fn compare_multiple_model_estimates(
    a: &TwoViewEstimate,
    b: &TwoViewEstimate,
) -> std::cmp::Ordering {
    a.inlier_mask
        .iter()
        .filter(|&&is_inlier| is_inlier)
        .count()
        .cmp(&b.inlier_mask.iter().filter(|&&is_inlier| is_inlier).count())
        .then_with(|| a.triangulated.cmp(&b.triangulated))
        .then_with(|| {
            b.mean_reprojection_error_px
                .partial_cmp(&a.mean_reprojection_error_px)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

fn ransac_support_limit() -> usize {
    std::env::var("RUSTSFM_RANSAC_SUPPORT_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 32)
        .unwrap_or(usize::MAX)
}

fn active_match_indices(
    obs1_px: &[[f32; 2]],
    obs2_px: &[[f32; 2]],
    n: usize,
    filter_stationary_matches: bool,
    stationary_matches_max_error_px: f64,
) -> Vec<usize> {
    if !filter_stationary_matches {
        return (0..n).collect();
    }
    let max_error_sq = stationary_matches_max_error_px.max(0.0).powi(2);
    (0..n)
        .filter(|&idx| {
            let p1 = obs1_px[idx];
            let p2 = obs2_px[idx];
            let dx = p1[0] as f64 - p2[0] as f64;
            let dy = p1[1] as f64 - p2[1] as f64;
            dx * dx + dy * dy > max_error_sq
        })
        .collect()
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
        || (candidate.inliers == current.inliers && candidate.residual_sum < current.residual_sum)
}

fn adaptive_ransac_iterations(
    inliers: usize,
    total: usize,
    max_iterations: u32,
    confidence: f64,
    sample_size: usize,
) -> u32 {
    let max_iterations = max_iterations.max(1);
    if sample_size == 0 || inliers >= total && total >= sample_size {
        return 1;
    }
    if inliers < sample_size || total < sample_size {
        max_iterations
    } else {
        let prob_failure = 1.0 - confidence;
        if prob_failure <= 0.0 {
            return max_iterations;
        }
        let mut prob_inlier = 1.0;
        for i in 0..sample_size {
            prob_inlier *= (inliers - i) as f64 / (total - i) as f64;
        }
        let prob_outlier = 1.0 - prob_inlier;
        if prob_outlier <= 0.0 {
            return 1;
        }
        if prob_outlier >= 1.0 {
            return max_iterations;
        }
        let num_trials = (prob_failure.ln() / prob_outlier.ln()
            * COLMAP_RANSAC_DYN_NUM_TRIALS_MULTIPLIER)
            .ceil();
        if !num_trials.is_finite() {
            max_iterations
        } else {
            num_trials.clamp(1.0, max_iterations as f64) as u32
        }
    }
}

fn estimate_fundamental_ransac(
    pts1: &[Vector3<f64>],
    pts2: &[Vector3<f64>],
    active_indices: &[usize],
    threshold: f64,
    max_iterations: u32,
    random_seed: u64,
    lo_steps: usize,
) -> Option<(Matrix3<f64>, ModelSupport)> {
    if active_indices.len() < 7 {
        return None;
    }
    let mut sampler = ColmapRandomSampler::new(
        random_seed ^ 0x517c_c1b7_2722_0a95 ^ active_indices.len() as u64,
        active_indices,
    );
    let mut best: Option<(Matrix3<f64>, ModelSupport)> = None;
    let mut max_iterations = max_iterations.max(1);
    let mut iteration = 0u32;
    while iteration < max_iterations {
        iteration += 1;
        let sample = sampler.sample(7);
        if sample.len() != 7 {
            continue;
        }
        let models = estimate_fundamental_seven_point_indexed(pts1, pts2, &sample);
        for model in models {
            let support = model_support_indexed(pts1, pts2, active_indices, &model, threshold);
            if support.inliers >= 7 && is_better_support(&support, best.as_ref().map(|(_, s)| s)) {
                max_iterations = max_iterations.min(adaptive_ransac_iterations(
                    support.inliers,
                    active_indices.len(),
                    max_iterations,
                    0.999,
                    7,
                ));
                best = Some((model, support));
            }
        }
    }
    let (model, support) = best.or_else(|| {
        estimate_fundamental_eight_point_indexed(pts1, pts2, active_indices).map(|model| {
            let support = model_support_indexed(pts1, pts2, active_indices, &model, threshold);
            (model, support)
        })
    })?;
    Some(refine_fundamental_support(
        pts1,
        pts2,
        active_indices,
        threshold,
        model,
        support,
        lo_steps,
    ))
}

fn estimate_fundamental_seven_point_indexed(
    pts1: &[Vector3<f64>],
    pts2: &[Vector3<f64>],
    indices: &[usize],
) -> Vec<Matrix3<f64>> {
    if indices.len() != 7 {
        return Vec::new();
    }
    let mut rows = Vec::with_capacity(indices.len() * 9);
    for &idx in indices {
        let Some(x1) = pts1.get(idx) else {
            return Vec::new();
        };
        let Some(x2) = pts2.get(idx) else {
            return Vec::new();
        };
        let x = x1.x / x1.z;
        let y = x1.y / x1.z;
        let u = x2.x / x2.z;
        let v = x2.y / x2.z;
        rows.extend_from_slice(&[u * x, u * y, u, v * x, v * y, v, x, y, 1.0]);
    }
    let a = DMatrix::<f64>::from_row_slice(indices.len(), 9, &rows);
    let ata = a.transpose() * a;
    let eigen = SymmetricEigen::new(ata);
    let mut order = (0..eigen.eigenvalues.len()).collect::<Vec<_>>();
    order.sort_by(|&lhs, &rhs| {
        eigen.eigenvalues[lhs]
            .partial_cmp(&eigen.eigenvalues[rhs])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if order.len() < 2 {
        return Vec::new();
    }
    let f2_col = eigen.eigenvectors.column(order[0]);
    let f1_col = eigen.eigenvectors.column(order[1]);
    let f2 = [
        f2_col[0], f2_col[1], f2_col[2], f2_col[3], f2_col[4], f2_col[5], f2_col[6], f2_col[7],
        f2_col[8],
    ];
    let mut f1 = [
        f1_col[0], f1_col[1], f1_col[2], f1_col[3], f1_col[4], f1_col[5], f1_col[6], f1_col[7],
        f1_col[8],
    ];
    for (a, b) in f1.iter_mut().zip(f2.iter()) {
        *a -= *b;
    }

    let Some(roots) = colmap_fundamental_cubic_roots(&f1, &f2) else {
        return Vec::new();
    };

    let mut models = Vec::with_capacity(roots.len());
    for root in roots {
        let f_vec = [
            f1[0] * root + f2[0],
            f1[1] * root + f2[1],
            f1[2] * root + f2[2],
            f1[3] * root + f2[3],
            f1[4] * root + f2[4],
            f1[5] * root + f2[5],
            f1[6] * root + f2[6],
            f1[7] * root + f2[7],
            f1[8] * root + f2[8],
        ];
        let norm = f_vec.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1.0e-12 && norm.is_finite() {
            models.push(Matrix3::from_row_slice(&f_vec) / norm);
        }
    }
    models
}

fn colmap_fundamental_cubic_roots(f1: &[f64; 9], f2: &[f64; 9]) -> Option<Vec<f64>> {
    let t0 = f1[4] * f1[8] - f1[5] * f1[7];
    let t1 = f1[3] * f1[8] - f1[5] * f1[6];
    let t2 = f1[3] * f1[7] - f1[4] * f1[6];
    let t3 = f2[4] * f2[8] - f2[5] * f2[7];
    let t4 = f2[3] * f2[8] - f2[5] * f2[6];
    let t5 = f2[3] * f2[7] - f2[4] * f2[6];

    let c3 = f1[0] * t0 - f1[1] * t1 + f1[2] * t2;
    if c3.abs() < 1.0e-16 {
        return None;
    }

    let c2 = f2[0] * t0 - f2[1] * t1 + f2[2] * t2 - f2[3] * (f1[1] * f1[8] - f1[2] * f1[7])
        + f2[4] * (f1[0] * f1[8] - f1[2] * f1[6])
        - f2[5] * (f1[0] * f1[7] - f1[1] * f1[6])
        + f2[6] * (f1[1] * f1[5] - f1[2] * f1[4])
        - f2[7] * (f1[0] * f1[5] - f1[2] * f1[3])
        + f2[8] * (f1[0] * f1[4] - f1[1] * f1[3]);
    let c1 = f1[0] * t3 - f1[1] * t4 + f1[2] * t5 - f1[3] * (f2[1] * f2[8] - f2[2] * f2[7])
        + f1[4] * (f2[0] * f2[8] - f2[2] * f2[6])
        - f1[5] * (f2[0] * f2[7] - f2[1] * f2[6])
        + f1[6] * (f2[1] * f2[5] - f2[2] * f2[4])
        - f1[7] * (f2[0] * f2[5] - f2[2] * f2[3])
        + f1[8] * (f2[0] * f2[4] - f2[1] * f2[3]);
    let c0 = f2[0] * t3 - f2[1] * t4 + f2[2] * t5;

    Some(colmap_cubic_polynomial_roots(c2 / c3, c1 / c3, c0 / c3))
}

fn colmap_cubic_polynomial_roots(c2: f64, c1: f64, c0: f64) -> Vec<f64> {
    const K2_PI_OVER_3: f64 = 2.09439510239319526263557236234192;
    const K4_PI_OVER_3: f64 = 4.18879020478639052527114472468384;

    let c2_over_3 = c2 / 3.0;
    let a = c1 - c2 * c2_over_3;
    let mut b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
    let mut c = b * b / 4.0 + a * a * a / 27.0;
    let mut roots = Vec::with_capacity(3);
    if c > 0.0 {
        c = c.sqrt();
        b *= -0.5;
        roots.push((b + c).cbrt() + (b - c).cbrt() - c2_over_3);
    } else if a.abs() > 1.0e-24 {
        c = 3.0 * b / (2.0 * a) * (-3.0 / a).sqrt();
        let d = 2.0 * (-a / 3.0).sqrt();
        let acos_over_3 = c.clamp(-1.0, 1.0).acos() / 3.0;
        roots.push(d * acos_over_3.cos() - c2_over_3);
        roots.push(d * (acos_over_3 - K2_PI_OVER_3).cos() - c2_over_3);
        roots.push(d * (acos_over_3 - K4_PI_OVER_3).cos() - c2_over_3);
    } else {
        roots.push(-c2_over_3);
    }

    for root in roots.iter_mut() {
        let x = *root;
        let x2 = x * x;
        let x3 = x * x2;
        let denom = 3.0 * x2 + 2.0 * c2 * x + c1;
        if denom.abs() > 1.0e-24 {
            *root += -(x3 + c2 * x2 + c1 * x + c0) / denom;
        }
    }
    roots.retain(|root| root.is_finite());
    roots
}

fn estimate_fundamental_eight_point_indexed(
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
    let q = colmap_eight_point_nullspace(&a)?;
    let f_norm = Matrix3::from_row_slice(&q);
    let svd_f = f_norm.svd(true, true);
    let u = svd_f.u?;
    let vt = svd_f.v_t?;
    let mut s = svd_f.singular_values;
    s[2] = 0.0;
    let f_rank2 = u * Matrix3::from_diagonal(&s) * vt;
    let f = t2.transpose() * f_rank2 * t1;
    let norm = f.norm();
    (norm > 1.0e-12 && norm.is_finite()).then_some(f / norm)
}

fn colmap_eight_point_nullspace(a: &DMatrix<f64>) -> Option<[f64; 9]> {
    if a.ncols() != 9 || a.nrows() < 8 {
        return None;
    }
    if a.nrows() == 8 {
        return eight_point_minimal_nullspace(a);
    }
    let svd = a.clone().svd(false, true);
    let vt = svd.v_t?;
    let q = vt.row(8);
    Some([q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8]])
}

fn eight_point_minimal_nullspace(a: &DMatrix<f64>) -> Option<[f64; 9]> {
    let mut best: Option<([f64; 9], f64)> = None;
    for free_col in 0..9 {
        let lhs = DMatrix::<f64>::from_fn(8, 8, |row, col| {
            let source_col = if col < free_col { col } else { col + 1 };
            a[(row, source_col)]
        });
        let rhs = DVector::<f64>::from_fn(8, |row, _| -a[(row, free_col)]);
        let Some(solution) = lhs.lu().solve(&rhs) else {
            continue;
        };
        let mut q = [0.0f64; 9];
        q[free_col] = 1.0;
        for col in 0..8 {
            let target_col = if col < free_col { col } else { col + 1 };
            q[target_col] = solution[col];
        }
        let norm = q.iter().map(|value| value * value).sum::<f64>().sqrt();
        if !norm.is_finite() || norm <= 1.0e-12 {
            continue;
        }
        for value in q.iter_mut() {
            *value /= norm;
        }
        let residual = (a * DVector::<f64>::from_column_slice(&q)).norm();
        if residual.is_finite()
            && best
                .as_ref()
                .is_none_or(|(_, best_residual)| residual < *best_residual)
        {
            best = Some((q, residual));
        }
    }
    best.map(|(q, _)| q)
}

fn refine_fundamental_support(
    pts1: &[Vector3<f64>],
    pts2: &[Vector3<f64>],
    active_indices: &[usize],
    threshold: f64,
    mut model: Matrix3<f64>,
    mut support: ModelSupport,
    lo_steps: usize,
) -> (Matrix3<f64>, ModelSupport) {
    for _ in 0..lo_steps {
        if support.inliers < 8 {
            break;
        }
        let inliers = support
            .inlier_mask
            .iter()
            .enumerate()
            .filter_map(|(idx, &is_inlier)| is_inlier.then_some(idx))
            .collect::<Vec<_>>();
        let sampled_inliers = sample_indices_evenly(&inliers, fundamental_refit_inlier_limit());
        let Some(refined) = estimate_fundamental_eight_point_indexed(pts1, pts2, &sampled_inliers)
        else {
            break;
        };
        let refined_support =
            model_support_indexed(pts1, pts2, active_indices, &refined, threshold);
        if is_better_support(&refined_support, Some(&support)) {
            model = refined;
            support = refined_support;
        } else {
            break;
        }
    }
    (model, support)
}

fn fundamental_refit_inlier_limit() -> usize {
    std::env::var("RUSTSFM_FUNDAMENTAL_REFIT_INLIERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 8)
        .unwrap_or(usize::MAX)
}

fn estimate_homography_ransac(
    pts1: &[Vector3<f64>],
    pts2: &[Vector3<f64>],
    active_indices: &[usize],
    threshold: f64,
    max_iterations: u32,
    random_seed: u64,
    lo_steps: usize,
) -> Option<(Matrix3<f64>, ModelSupport)> {
    if active_indices.len() < 4 {
        return None;
    }
    let mut sampler = ColmapRandomSampler::new(
        random_seed ^ 0x94d0_49bb_1331_11eb ^ active_indices.len() as u64,
        active_indices,
    );
    let mut best: Option<(Matrix3<f64>, ModelSupport)> = None;
    let mut max_iterations = max_iterations.max(1);
    let mut iteration = 0u32;
    while iteration < max_iterations {
        iteration += 1;
        let sample = sampler.sample(4);
        if sample.len() != 4 {
            continue;
        }
        let Some(model) = estimate_homography_dlt_indexed(pts1, pts2, &sample) else {
            continue;
        };
        let support = homography_support_indexed(pts1, pts2, active_indices, &model, threshold);
        if support.inliers >= 4 && is_better_support(&support, best.as_ref().map(|(_, s)| s)) {
            max_iterations = max_iterations.min(adaptive_ransac_iterations(
                support.inliers,
                active_indices.len(),
                max_iterations,
                0.999,
                4,
            ));
            best = Some((model, support));
        }
    }
    let (model, support) = best.or_else(|| {
        estimate_homography_dlt_indexed(pts1, pts2, active_indices).map(|model| {
            let support = homography_support_indexed(pts1, pts2, active_indices, &model, threshold);
            (model, support)
        })
    })?;
    Some(refine_homography_support(
        pts1,
        pts2,
        active_indices,
        threshold,
        model,
        support,
        lo_steps,
    ))
}

fn estimate_homography_dlt_indexed(
    pts1: &[Vector3<f64>],
    pts2: &[Vector3<f64>],
    indices: &[usize],
) -> Option<Matrix3<f64>> {
    if indices.len() < 4 {
        return None;
    }
    let mut rows = Vec::with_capacity(indices.len() * 18);
    for &idx in indices {
        let x1 = pts1.get(idx)?;
        let x2 = pts2.get(idx)?;
        let x = x1.x / x1.z;
        let y = x1.y / x1.z;
        let u = x2.x / x2.z;
        let v = x2.y / x2.z;
        rows.extend_from_slice(&[x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u]);
        rows.extend_from_slice(&[0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, -v]);
    }
    let a = DMatrix::<f64>::from_row_slice(indices.len() * 2, 9, &rows);
    let h = if indices.len() == 4 {
        let lhs = DMatrix::<f64>::from_fn(8, 8, |row, col| a[(row, col)]);
        let rhs = DVector::<f64>::from_fn(8, |row, _| -a[(row, 8)]);
        let h = lhs.lu().solve(&rhs)?;
        if h.iter().any(|value| value.is_nan()) {
            return None;
        }
        Matrix3::from_row_slice(&[h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0])
    } else {
        let svd = a.svd(false, true);
        if colmap_svd_rank(&svd.singular_values) < 8 {
            return None;
        }
        let vt = svd.v_t?;
        let q = vt.row(8);
        Matrix3::from_row_slice(&[q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8]])
    };
    (h.determinant().abs() >= 1.0e-8).then_some(h)
}

fn colmap_svd_rank(singular_values: &DVector<f64>) -> usize {
    if singular_values.is_empty() {
        return 0;
    }
    let threshold = (singular_values[0] * singular_values.len().max(1) as f64 * f64::EPSILON)
        .max(f64::MIN_POSITIVE);
    singular_values
        .iter()
        .filter(|value| **value >= threshold)
        .count()
}

fn refine_homography_support(
    pts1: &[Vector3<f64>],
    pts2: &[Vector3<f64>],
    active_indices: &[usize],
    threshold: f64,
    mut model: Matrix3<f64>,
    mut support: ModelSupport,
    lo_steps: usize,
) -> (Matrix3<f64>, ModelSupport) {
    for _ in 0..lo_steps {
        if support.inliers < 4 {
            break;
        }
        let inliers = support
            .inlier_mask
            .iter()
            .enumerate()
            .filter_map(|(idx, &is_inlier)| is_inlier.then_some(idx))
            .collect::<Vec<_>>();
        let sampled_inliers = sample_indices_evenly(&inliers, homography_refit_inlier_limit());
        let Some(refined) = estimate_homography_dlt_indexed(pts1, pts2, &sampled_inliers) else {
            break;
        };
        let refined_support =
            homography_support_indexed(pts1, pts2, active_indices, &refined, threshold);
        if is_better_support(&refined_support, Some(&support)) {
            model = refined;
            support = refined_support;
        } else {
            break;
        }
    }
    (model, support)
}

fn homography_refit_inlier_limit() -> usize {
    std::env::var("RUSTSFM_HOMOGRAPHY_REFIT_INLIERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 4)
        .unwrap_or(usize::MAX)
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
    let a = essential_eight_point_constraint_matrix(pts1, pts2, indices)?;
    let q = colmap_eight_point_nullspace(&a)?;
    enforce_essential_constraints(Matrix3::from_row_slice(&q))
}

fn estimate_essential_eight_point_indexed_lightweight(
    pts1: &[Vector3<f64>],
    pts2: &[Vector3<f64>],
    indices: &[usize],
) -> Option<Matrix3<f64>> {
    if indices.len() < 8 {
        return None;
    }
    let a = essential_eight_point_constraint_matrix(pts1, pts2, indices)?;
    let svd = a.svd(false, true);
    let vt = svd.v_t?;
    let q = vt.row(vt.nrows() - 1);
    let e = Matrix3::from_row_slice(&[q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8]]);
    enforce_essential_constraints(e)
}

fn essential_eight_point_constraint_matrix(
    pts1: &[Vector3<f64>],
    pts2: &[Vector3<f64>],
    indices: &[usize],
) -> Option<DMatrix<f64>> {
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
    Some(DMatrix::<f64>::from_row_slice(indices.len(), 9, &rows))
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

    let mut rms_dist_sq = 0.0f64;
    for &idx in indices {
        let p = pts.get(idx)?;
        let x = p.x / p.z - centroid.x;
        let y = p.y / p.z - centroid.y;
        rms_dist_sq += x * x + y * y;
    }
    let rms_dist = (rms_dist_sq / indices.len() as f64).sqrt();
    let scale = std::f64::consts::SQRT_2 / rms_dist.max(1.0e-12);
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
    let mut inliers = 0usize;
    let mut residual_sum = 0.0f64;
    for &idx in indices {
        if idx >= n {
            continue;
        }
        let residual = squared_sampson_error(&pts1[idx], &pts2[idx], essential);
        if residual.is_finite() && residual <= threshold_sq {
            inlier_mask[idx] = true;
            inliers += 1;
            residual_sum += residual;
        }
    }
    ModelSupport {
        inlier_mask,
        inliers,
        residual_sum,
    }
}

fn homography_support_indexed(
    pts1: &[Vector3<f64>],
    pts2: &[Vector3<f64>],
    indices: &[usize],
    homography: &Matrix3<f64>,
    threshold: f64,
) -> ModelSupport {
    let threshold_sq = threshold.max(1.0e-12).powi(2);
    let n = pts1.len().min(pts2.len());
    let mut inlier_mask = vec![false; n];
    let mut inliers = 0usize;
    let mut residual_sum = 0.0f64;
    for &idx in indices {
        if idx >= n {
            continue;
        }
        let residual = homography_forward_error(&pts1[idx], &pts2[idx], homography);
        if residual.is_finite() && residual <= threshold_sq {
            inlier_mask[idx] = true;
            inliers += 1;
            residual_sum += residual;
        }
    }
    ModelSupport {
        inlier_mask,
        inliers,
        residual_sum,
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

fn homography_forward_error(
    x1: &Vector3<f64>,
    x2: &Vector3<f64>,
    homography: &Matrix3<f64>,
) -> f64 {
    let Some(p2) = dehomogeneous(&(homography * x1)) else {
        return f64::INFINITY;
    };
    let x2x = x2.x / x2.z;
    let x2y = x2.y / x2.z;
    (p2[0] - x2x).powi(2) + (p2[1] - x2y).powi(2)
}

fn dehomogeneous(p: &Vector3<f64>) -> Option<[f64; 2]> {
    (p.z.abs() > 1.0e-12 && p.z.is_finite()).then_some([p.x / p.z, p.y / p.z])
}

fn classify_calibrated_two_view(
    e_support: &ModelSupport,
    f_support: Option<&ModelSupport>,
    h_support: Option<&ModelSupport>,
    options: &TwoViewOptions,
) -> Option<(i32, Vec<bool>, usize)> {
    let min_num_inliers = options.min_inliers;
    let f_inliers = f_support.map(|s| s.inliers).unwrap_or(0);
    let h_inliers = h_support.map(|s| s.inliers).unwrap_or(0);
    if e_support.inliers < min_num_inliers
        && f_inliers < min_num_inliers
        && h_inliers < min_num_inliers
    {
        return None;
    }
    if options.force_h_use {
        return h_support.filter(|s| s.inliers >= min_num_inliers).map(|s| {
            (
                crate::database::COLMAP_TWO_VIEW_PLANAR_OR_PANORAMIC,
                s.inlier_mask.clone(),
                s.inliers,
            )
        });
    }

    let e_f_ratio = ratio(e_support.inliers, f_inliers);
    let h_f_ratio = ratio(h_inliers, f_inliers);
    let h_e_ratio = ratio(h_inliers, e_support.inliers);

    if e_support.inliers >= min_num_inliers && e_f_ratio > options.min_e_f_inlier_ratio {
        let mut config = crate::database::COLMAP_TWO_VIEW_CALIBRATED;
        let mut mask = e_support.inlier_mask.clone();
        let mut inliers = e_support.inliers;
        if let Some(f_support) = f_support {
            if f_support.inliers > inliers {
                mask = f_support.inlier_mask.clone();
                inliers = f_support.inliers;
            }
        }
        if h_e_ratio > options.max_h_inlier_ratio {
            config = crate::database::COLMAP_TWO_VIEW_PLANAR_OR_PANORAMIC;
            if let Some(h_support) = h_support {
                if h_support.inliers > inliers {
                    mask = h_support.inlier_mask.clone();
                    inliers = h_support.inliers;
                }
            }
        }
        Some((config, mask, inliers))
    } else if let Some(f_support) = f_support.filter(|s| s.inliers >= min_num_inliers) {
        let mut config = crate::database::COLMAP_TWO_VIEW_UNCALIBRATED;
        let mut mask = f_support.inlier_mask.clone();
        let mut inliers = f_support.inliers;
        if h_f_ratio > options.max_h_inlier_ratio {
            config = crate::database::COLMAP_TWO_VIEW_PLANAR_OR_PANORAMIC;
            if let Some(h_support) = h_support {
                if h_support.inliers > inliers {
                    mask = h_support.inlier_mask.clone();
                    inliers = h_support.inliers;
                }
            }
        }
        Some((config, mask, inliers))
    } else {
        h_support.filter(|s| s.inliers >= min_num_inliers).map(|s| {
            (
                crate::database::COLMAP_TWO_VIEW_PLANAR_OR_PANORAMIC,
                s.inlier_mask.clone(),
                s.inliers,
            )
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn pose_essential_and_mask(
    essential: Matrix3<f64>,
    fundamental: Option<&Matrix3<f64>>,
    e_support: &ModelSupport,
    selected_mask: &[bool],
    selected_inliers: usize,
    two_view_config: i32,
    pts1: &[Vector3<f64>],
    pts2: &[Vector3<f64>],
    active_indices: &[usize],
    camera1: CameraModel,
    camera2: CameraModel,
    options: &TwoViewOptions,
) -> (Matrix3<f64>, Vec<bool>) {
    let use_fundamental_pose = matches!(
        two_view_config,
        crate::database::COLMAP_TWO_VIEW_UNCALIBRATED
            | crate::database::COLMAP_TWO_VIEW_PLANAR
            | crate::database::COLMAP_TWO_VIEW_PANORAMIC
            | crate::database::COLMAP_TWO_VIEW_PLANAR_OR_PANORAMIC
    ) && selected_inliers > e_support.inliers;
    let pose_essential = if use_fundamental_pose {
        fundamental
            .and_then(|f| fundamental_to_essential(f, camera1, camera2))
            .unwrap_or(essential)
    } else {
        essential
    };
    let pose_mask = if use_fundamental_pose {
        selected_mask.to_vec()
    } else if e_support.inliers >= options.min_inliers {
        e_support.inlier_mask.clone()
    } else {
        selected_mask.to_vec()
    };
    let support = model_support_indexed(
        pts1,
        pts2,
        active_indices,
        &pose_essential,
        options.ransac_threshold,
    );
    if support.inliers >= options.min_inliers
        && support.inliers >= pose_mask.iter().filter(|&&v| v).count()
    {
        (pose_essential, support.inlier_mask)
    } else {
        (pose_essential, pose_mask)
    }
}

fn fundamental_to_essential(
    fundamental: &Matrix3<f64>,
    camera1: CameraModel,
    camera2: CameraModel,
) -> Option<Matrix3<f64>> {
    let k1 = camera_intrinsic_matrix(camera1);
    let k2 = camera_intrinsic_matrix(camera2);
    enforce_essential_constraints(k2.transpose() * fundamental * k1)
}

fn classify_homography_motion(
    homography: &Matrix3<f64>,
    camera1: CameraModel,
    camera2: CameraModel,
) -> i32 {
    let Some(h_norm) = normalize_pixel_homography(homography, camera1, camera2) else {
        return crate::database::COLMAP_TWO_VIEW_PLANAR_OR_PANORAMIC;
    };
    let Some(rotation) = closest_rotation(&h_norm) else {
        return crate::database::COLMAP_TWO_VIEW_PLANAR_OR_PANORAMIC;
    };
    let rotation_residual = (h_norm - rotation).norm() / 3.0f64.sqrt();
    if rotation_residual <= homography_panoramic_residual_threshold() {
        crate::database::COLMAP_TWO_VIEW_PANORAMIC
    } else {
        crate::database::COLMAP_TWO_VIEW_PLANAR
    }
}

fn normalize_pixel_homography(
    homography: &Matrix3<f64>,
    camera1: CameraModel,
    camera2: CameraModel,
) -> Option<Matrix3<f64>> {
    let k1 = camera_intrinsic_matrix(camera1);
    let k2_inv = camera_intrinsic_matrix(camera2).try_inverse()?;
    let mut h_norm = k2_inv * homography * k1;
    let scale = h_norm.determinant().abs().powf(1.0 / 3.0);
    if !scale.is_finite() || scale <= 1.0e-12 {
        return None;
    }
    h_norm /= scale;
    if h_norm.determinant() < 0.0 {
        h_norm *= -1.0;
    }
    h_norm.iter().all(|v| v.is_finite()).then_some(h_norm)
}

fn camera_intrinsic_matrix(camera: CameraModel) -> Matrix3<f64> {
    Matrix3::new(
        camera.fx as f64,
        0.0,
        camera.cx as f64,
        0.0,
        camera.fy as f64,
        camera.cy as f64,
        0.0,
        0.0,
        1.0,
    )
}

fn closest_rotation(matrix: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    let svd = matrix.svd(true, true);
    let mut u = svd.u?;
    let vt = svd.v_t?;
    let mut rotation = u * vt;
    if rotation.determinant() < 0.0 {
        u.column_mut(2).scale_mut(-1.0);
        rotation = u * vt;
    }
    rotation.iter().all(|v| v.is_finite()).then_some(rotation)
}

fn homography_panoramic_residual_threshold() -> f64 {
    std::env::var("RUSTSFM_HOMOGRAPHY_PANORAMIC_RESIDUAL")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| *value > 0.0)
        .unwrap_or(0.05)
}

fn ratio(num: usize, denom: usize) -> f64 {
    if denom == 0 {
        f64::INFINITY
    } else {
        num as f64 / denom as f64
    }
}

fn detect_watermark_matches(
    camera1: CameraModel,
    camera2: CameraModel,
    points1: &[[f32; 2]],
    points2: &[[f32; 2]],
    inlier_mask: &[bool],
    num_inliers: usize,
    options: &TwoViewOptions,
) -> bool {
    if num_inliers == 0 {
        return false;
    }
    let diagonal1 = ((camera1.width as f64).powi(2) + (camera1.height as f64).powi(2)).sqrt();
    let diagonal2 = ((camera2.width as f64).powi(2) + (camera2.height as f64).powi(2)).sqrt();
    let border1 = options.watermark_border_size.clamp(0.0, 1.0) * diagonal1;
    let border2 = options.watermark_border_size.clamp(0.0, 1.0) * diagonal2;
    let mut inlier_points = Vec::with_capacity(num_inliers);
    let mut num_border = 0usize;
    for idx in 0..inlier_mask.len().min(points1.len()).min(points2.len()) {
        if !inlier_mask[idx] {
            continue;
        }
        let p1 = points1[idx];
        let p2 = points2[idx];
        if is_in_watermark_border(p1, camera1, border1)
            || is_in_watermark_border(p2, camera2, border2)
        {
            num_border += 1;
        }
        inlier_points.push((p1, p2));
    }
    if num_border as f64 / (num_inliers as f64) < options.watermark_min_inlier_ratio {
        return false;
    }
    let translations = inlier_points
        .iter()
        .map(|(p1, p2)| [p2[0] as f64 - p1[0] as f64, p2[1] as f64 - p1[1] as f64])
        .collect::<Vec<_>>();
    if translations.is_empty() {
        return false;
    }
    let mut best = 0usize;
    let max_error_sq = options.watermark_detection_max_error_px.max(0.0).powi(2);
    for candidate in &translations {
        let count = translations
            .iter()
            .filter(|tr| {
                let dx = tr[0] - candidate[0];
                let dy = tr[1] - candidate[1];
                dx * dx + dy * dy <= max_error_sq
            })
            .count();
        best = best.max(count);
    }
    best as f64 / num_inliers as f64 >= options.watermark_min_inlier_ratio
}

fn is_in_watermark_border(point: [f32; 2], camera: CameraModel, border: f64) -> bool {
    let x = point[0] as f64;
    let y = point[1] as f64;
    x < border
        || y < border
        || x > camera.width as f64 - border
        || y > camera.height as f64 - border
}

fn choose_pose_from_essential(
    essential: &Matrix3<f64>,
    pts1: &[Vector3<f64>],
    pts2: &[Vector3<f64>],
    obs1_px: &[[f32; 2]],
    obs2_px: &[[f32; 2]],
    inlier_mask: &[bool],
    camera1: CameraModel,
    camera2: CameraModel,
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
                    camera1,
                    camera2,
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
    camera1: CameraModel,
    camera2: CameraModel,
) -> f64 {
    let err1 = camera_reprojection_error_px(point1, observation1_px, camera1);
    let err2 = camera_reprojection_error_px(point2, observation2_px, camera2);
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

struct ColmapMt19937 {
    state: [u32; 624],
    index: usize,
}

impl ColmapMt19937 {
    fn new(seed: u64) -> Self {
        let mut rng = Self {
            state: [0; 624],
            index: 624,
        };
        rng.state[0] = seed as u32;
        for i in 1..624 {
            rng.state[i] = 1_812_433_253u32
                .wrapping_mul(rng.state[i - 1] ^ (rng.state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        rng
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    fn twist(&mut self) {
        const UPPER_MASK: u32 = 0x8000_0000;
        const LOWER_MASK: u32 = 0x7fff_ffff;
        const MATRIX_A: u32 = 0x9908_b0df;

        for i in 0..624 {
            let x = (self.state[i] & UPPER_MASK) | (self.state[(i + 1) % 624] & LOWER_MASK);
            let mut xa = x >> 1;
            if x & 1 != 0 {
                xa ^= MATRIX_A;
            }
            self.state[i] = self.state[(i + 397) % 624] ^ xa;
        }
        self.index = 0;
    }

    fn uniform_u32(&mut self, min: u32, max: u32) -> u32 {
        let range = max.wrapping_sub(min).wrapping_add(1);
        if range == 1 {
            return min;
        }
        let width = if range == 0 {
            u32::BITS
        } else {
            let floor_log2 = u32::BITS - range.leading_zeros() - 1;
            let is_power_of_two = range & ((u32::MAX) >> (u32::BITS - floor_log2)) == 0;
            floor_log2 + u32::from(!is_power_of_two)
        };
        loop {
            let sample = self.independent_bits(width);
            if range == 0 || sample < range {
                return sample.wrapping_add(min);
            }
        }
    }

    fn independent_bits(&mut self, width: u32) -> u32 {
        if width == 0 {
            return 0;
        }
        let mask = if width < u32::BITS {
            u32::MAX >> (u32::BITS - width)
        } else {
            u32::MAX
        };
        self.next_u32() & mask
    }
}

struct ColmapRandomSampler {
    rng: ColmapMt19937,
    sample_indices: Vec<usize>,
}

impl ColmapRandomSampler {
    fn new(seed: u64, indices: &[usize]) -> Self {
        Self {
            rng: ColmapMt19937::new(seed),
            sample_indices: indices.to_vec(),
        }
    }

    fn sample(&mut self, k: usize) -> Vec<usize> {
        if k > self.sample_indices.len() {
            return Vec::new();
        }
        let last = self.sample_indices.len() - 1;
        for i in 0..k {
            let j = self.rng.uniform_u32(i as u32, last as u32) as usize;
            self.sample_indices.swap(i, j);
        }
        self.sample_indices[..k].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_test_options() -> TwoViewOptions {
        TwoViewOptions {
            ransac_max_error_px: 4.0,
            ransac_threshold: 0.01,
            ransac_max_iterations: 128,
            random_seed: 42,
            loransac_num_lo_steps: 6,
            min_inliers: 15,
            min_inlier_ratio: 0.0,
            min_triangulated: 1,
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
            use_hartley_refinement: true,
            use_five_point: false,
        }
    }

    fn skew_matrix(t: Vector3<f64>) -> Matrix3<f64> {
        Matrix3::new(0.0, -t.z, t.y, t.z, 0.0, -t.x, -t.y, t.x, 0.0)
    }

    fn essential_distance(a: Matrix3<f64>, b: Matrix3<f64>) -> f64 {
        let a = a.normalize();
        let b = b.normalize();
        (a - b).norm().min((a + b).norm())
    }

    #[test]
    fn adaptive_ransac_iterations_matches_colmap_trial_formula() {
        assert_eq!(adaptive_ransac_iterations(50, 100, 10_000, 0.999, 5), 726);
        assert_eq!(adaptive_ransac_iterations(50, 100, 10_000, 0.999, 8), 7173);
        assert_eq!(adaptive_ransac_iterations(90, 100, 10_000, 0.999, 5), 24);
    }

    #[test]
    fn adaptive_ransac_iterations_keeps_full_budget_for_invalid_support() {
        assert_eq!(adaptive_ransac_iterations(3, 100, 123, 0.999, 5), 123);
        assert_eq!(adaptive_ransac_iterations(10, 4, 123, 0.999, 5), 123);
        assert_eq!(adaptive_ransac_iterations(10, 10, 123, 0.999, 5), 1);
    }

    #[test]
    fn colmap_random_sampler_uses_stateful_partial_shuffle() {
        let mut sampler = ColmapRandomSampler::new(42, &[0, 1, 2, 3, 4, 5]);

        assert_eq!(sampler.sample(3), vec![3, 5, 4]);
        assert_eq!(sampler.sample(3), vec![4, 1, 3]);

        let mut sampler = ColmapRandomSampler::new(1, &[0, 1, 2, 3, 4, 5]);
        assert_eq!(sampler.sample(3), vec![5, 4, 2]);
        assert_eq!(sampler.sample(3), vec![5, 2, 0]);
    }

    #[test]
    fn colmap_random_sampler_rejects_oversized_samples() {
        let mut sampler = ColmapRandomSampler::new(1, &[10, 20]);

        assert!(sampler.sample(3).is_empty());
    }

    #[test]
    fn colmap_mt19937_matches_reference_outputs() {
        let mut rng = ColmapMt19937::new(42);

        let outputs = [
            rng.next_u32(),
            rng.next_u32(),
            rng.next_u32(),
            rng.next_u32(),
            rng.next_u32(),
            rng.next_u32(),
        ];

        assert_eq!(
            outputs,
            [
                1_608_637_542,
                3_421_126_067,
                4_083_286_876,
                787_846_414,
                3_143_890_026,
                3_348_747_335,
            ]
        );
    }

    #[test]
    fn fundamental_seven_point_matches_colmap_reference() {
        let points1_raw = [
            0.4964, 1.0577, 0.3650, -0.0919, -0.5412, 0.0159, -0.5239, 0.9467, 0.3467, 0.5301,
            0.2797, 0.0012, -0.1986, 0.0460,
        ];
        let points2_raw = [
            0.7570, 2.7340, 0.3961, 0.6981, -0.6014, 0.7110, -0.7385, 2.2712, 0.4177, 1.2132,
            0.3052, 0.4835, -0.2171, 0.5057,
        ];
        let pts1 = (0..7)
            .map(|i| Vector3::new(points1_raw[2 * i], points1_raw[2 * i + 1], 1.0))
            .collect::<Vec<_>>();
        let pts2 = (0..7)
            .map(|i| Vector3::new(points2_raw[2 * i], points2_raw[2 * i + 1], 1.0))
            .collect::<Vec<_>>();

        let models = estimate_fundamental_seven_point_indexed(&pts1, &pts2, &[0, 1, 2, 3, 4, 5, 6]);

        assert_eq!(models.len(), 1);
        let f = models[0] / models[0][(2, 2)];
        let expected = Matrix3::new(
            4.81441976,
            -8.16978909,
            6.73133404,
            5.16247992,
            0.19325606,
            -2.87239381,
            -9.92570126,
            3.64159554,
            1.0,
        );
        for (actual, expected) in f.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
    }

    #[test]
    fn fundamental_seven_point_solver_recovers_epipolar_models() {
        let points_world = [
            Vector3::new(-0.8, -0.5, 4.0),
            Vector3::new(-0.2, 0.4, 4.6),
            Vector3::new(0.5, -0.3, 5.2),
            Vector3::new(0.9, 0.7, 5.8),
            Vector3::new(-0.6, 0.9, 6.1),
            Vector3::new(0.1, -0.8, 4.9),
            Vector3::new(0.7, 0.1, 6.4),
        ];
        let translation = Vector3::new(0.6, -0.1, 0.2);
        let rotation = Rotation3::from_euler_angles(0.04, -0.03, 0.02);
        let mut pts1 = Vec::new();
        let mut pts2 = Vec::new();
        for point in points_world {
            pts1.push(Vector3::new(point.x / point.z, point.y / point.z, 1.0));
            let p2 = rotation * (point - translation);
            pts2.push(Vector3::new(p2.x / p2.z, p2.y / p2.z, 1.0));
        }

        let models = estimate_fundamental_seven_point_indexed(&pts1, &pts2, &[0, 1, 2, 3, 4, 5, 6]);

        assert!(!models.is_empty());
        assert!(models.iter().any(|model| {
            let det = model.determinant().abs();
            let max_residual = pts1
                .iter()
                .zip(pts2.iter())
                .map(|(x1, x2)| squared_sampson_error(x1, x2, model))
                .fold(0.0, f64::max);
            det < 1.0e-8 && max_residual < 1.0e-10
        }));
    }

    #[test]
    fn normalize_points_indexed_matches_colmap_rms_scaling() {
        let points = (0..11)
            .map(|i| Vector3::new(i as f64, i as f64, 1.0))
            .collect::<Vec<_>>();
        let indices = (0..points.len()).collect::<Vec<_>>();

        let (normalized, transform) = normalize_points_indexed(&points, &indices).unwrap();

        assert!((transform[(0, 0)] - 0.31622776601683794).abs() < 1.0e-15);
        assert!((transform[(1, 1)] - 0.31622776601683794).abs() < 1.0e-15);
        assert!((transform[(0, 2)] + 1.5811388300841898).abs() < 1.0e-15);
        assert!((transform[(1, 2)] + 1.5811388300841898).abs() < 1.0e-15);
        let mean_x = normalized.iter().map(|p| p.x / p.z).sum::<f64>() / normalized.len() as f64;
        let mean_y = normalized.iter().map(|p| p.y / p.z).sum::<f64>() / normalized.len() as f64;
        assert!(mean_x.abs() < 1.0e-12);
        assert!(mean_y.abs() < 1.0e-12);
    }

    #[test]
    fn fundamental_eight_point_matches_colmap_reference() {
        let points1_raw = [
            1.839035, 1.924743, 0.543582, 0.375221, 0.473240, 0.142522, 0.964910, 0.598376,
            0.102388, 0.140092, 15.994343, 9.622164, 0.285901, 0.430055, 0.091150, 0.254594,
        ];
        let points2_raw = [
            1.002114, 1.129644, 1.521742, 1.846002, 1.084332, 0.275134, 0.293328, 0.588992,
            0.839509, 0.087290, 1.779735, 1.116857, 0.878616, 0.602447, 0.642616, 1.028681,
        ];
        let pts1 = (0..8)
            .map(|i| Vector3::new(points1_raw[2 * i], points1_raw[2 * i + 1], 1.0))
            .collect::<Vec<_>>();
        let pts2 = (0..8)
            .map(|i| Vector3::new(points2_raw[2 * i], points2_raw[2 * i + 1], 1.0))
            .collect::<Vec<_>>();

        let f = estimate_fundamental_eight_point_indexed(&pts1, &pts2, &[0, 1, 2, 3, 4, 5, 6, 7])
            .unwrap();
        let f = f / f[(2, 2)];
        let expected = Matrix3::new(
            -9.85701, 18.97038, -1.55224, -3.24832, 2.04346, 0.977619, 11.22355, -19.43171, 1.0,
        );

        for (actual, expected) in f.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1.0e-5);
        }
    }

    #[test]
    fn essential_eight_point_uses_colmap_raw_ray_estimator() {
        let rotation = Rotation3::from_euler_angles(0.03, -0.04, 0.02);
        let translation = Vector3::new(0.2, -0.03, 0.05).normalize();
        let expected = (skew_matrix(translation) * rotation.matrix()).normalize();
        let points_world = [
            Vector3::new(-0.4, -0.2, 3.0),
            Vector3::new(0.1, -0.3, 4.0),
            Vector3::new(0.5, -0.1, 3.5),
            Vector3::new(-0.2, 0.4, 4.2),
            Vector3::new(0.4, 0.3, 3.8),
            Vector3::new(-0.6, 0.15, 4.5),
            Vector3::new(0.25, -0.45, 3.7),
            Vector3::new(0.7, 0.25, 4.8),
        ];
        let pts1 = points_world
            .iter()
            .map(|p| Vector3::new(p.x / p.z, p.y / p.z, 1.0))
            .collect::<Vec<_>>();
        let pts2 = points_world
            .iter()
            .map(|p| {
                let q = rotation * p + translation;
                Vector3::new(q.x / q.z, q.y / q.z, 1.0)
            })
            .collect::<Vec<_>>();

        let estimated =
            estimate_essential_eight_point_indexed(&pts1, &pts2, &[0, 1, 2, 3, 4, 5, 6, 7])
                .unwrap();
        let direct = estimate_essential_eight_point_indexed_lightweight(
            &pts1,
            &pts2,
            &[0, 1, 2, 3, 4, 5, 6, 7],
        )
        .unwrap();

        assert!(essential_distance(estimated, expected) < 1.0e-8);
        assert!(direct.iter().all(|value| value.is_finite()));
    }

    fn transform_homography_point(h: &Matrix3<f64>, x: f64, y: f64) -> Vector3<f64> {
        let p = h * Vector3::new(x, y, 1.0);
        Vector3::new(p.x / p.z, p.y / p.z, 1.0)
    }

    #[test]
    fn homography_four_point_estimator_matches_colmap_lu_path() {
        let expected = Matrix3::new(1.2, 0.15, 3.0, -0.08, 0.9, -2.0, 0.001, -0.002, 1.0);
        let pts1 = vec![
            Vector3::new(-1.0, -1.0, 1.0),
            Vector3::new(2.0, -0.5, 1.0),
            Vector3::new(1.4, 1.3, 1.0),
            Vector3::new(-0.7, 1.1, 1.0),
        ];
        let pts2 = pts1
            .iter()
            .map(|p| transform_homography_point(&expected, p.x, p.y))
            .collect::<Vec<_>>();

        let estimated = estimate_homography_dlt_indexed(&pts1, &pts2, &[0, 1, 2, 3]).unwrap();

        for (actual, expected) in estimated.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1.0e-9);
        }
    }

    #[test]
    fn homography_multi_point_estimator_matches_colmap_svd_path() {
        let expected = Matrix3::new(0.8, -0.2, 4.0, 0.12, 1.1, 1.5, -0.003, 0.001, 1.0);
        let pts1 = vec![
            Vector3::new(-2.0, -1.0, 1.0),
            Vector3::new(1.5, -0.8, 1.0),
            Vector3::new(2.0, 1.2, 1.0),
            Vector3::new(-1.3, 1.4, 1.0),
            Vector3::new(0.2, 0.3, 1.0),
        ];
        let pts2 = pts1
            .iter()
            .map(|p| transform_homography_point(&expected, p.x, p.y))
            .collect::<Vec<_>>();

        let estimated = estimate_homography_dlt_indexed(&pts1, &pts2, &[0, 1, 2, 3, 4]).unwrap();
        let estimated = estimated / estimated[(2, 2)];

        for (actual, expected) in estimated.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1.0e-8);
        }
    }

    #[test]
    fn homography_estimator_rejects_singular_colmap_models() {
        let singular = Matrix3::new(1.0, 0.2, 0.5, 0.0, 0.0, 0.0, 0.01, -0.02, 1.0);
        let pts1 = vec![
            Vector3::new(-2.0, -1.0, 1.0),
            Vector3::new(1.5, -0.8, 1.0),
            Vector3::new(2.0, 1.2, 1.0),
            Vector3::new(-1.3, 1.4, 1.0),
            Vector3::new(0.2, 0.3, 1.0),
        ];
        let pts2 = pts1
            .iter()
            .map(|p| transform_homography_point(&singular, p.x, p.y))
            .collect::<Vec<_>>();

        assert!(estimate_homography_dlt_indexed(&pts1, &pts2, &[0, 1, 2, 3, 4]).is_none());
    }

    #[test]
    fn homography_support_uses_colmap_forward_projection_error() {
        let homography = Matrix3::new(0.01, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let pts1 = vec![Vector3::new(100.0, 0.0, 1.0), Vector3::new(0.0, 1.0, 1.0)];
        let pts2 = vec![Vector3::new(2.0, 0.0, 1.0), Vector3::new(0.0, 1.0, 1.0)];

        let support = homography_support_indexed(&pts1, &pts2, &[0, 1], &homography, 2.0);
        let inverse_homography = homography.try_inverse().unwrap();
        let forward_residual = homography_forward_error(&pts1[0], &pts2[0], &homography);
        let symmetric_residual = {
            let p2 = dehomogeneous(&(homography * pts1[0])).unwrap();
            let p1 = dehomogeneous(&(inverse_homography * pts2[0])).unwrap();
            0.5 * ((p2[0] - pts2[0].x / pts2[0].z).powi(2)
                + (p2[1] - pts2[0].y / pts2[0].z).powi(2)
                + (p1[0] - pts1[0].x / pts1[0].z).powi(2)
                + (p1[1] - pts1[0].y / pts1[0].z).powi(2))
        };

        assert_eq!(support.inliers, 2);
        assert_eq!(support.inlier_mask, vec![true, true]);
        assert!((support.residual_sum - 1.0).abs() < 1.0e-12);
        assert!(forward_residual <= 4.0);
        assert!(symmetric_residual > 4.0);
    }

    #[test]
    fn support_ordering_matches_colmap_inlier_residual_sum() {
        let current = ModelSupport {
            inlier_mask: vec![true, true, true],
            inliers: 3,
            residual_sum: 1.0,
        };
        let more_inliers = ModelSupport {
            inlier_mask: vec![true, true, true, true],
            inliers: 4,
            residual_sum: 10.0,
        };
        let lower_residual_sum = ModelSupport {
            inlier_mask: vec![true, true, true],
            inliers: 3,
            residual_sum: 0.9,
        };
        let higher_residual_sum = ModelSupport {
            inlier_mask: vec![true, true, true],
            inliers: 3,
            residual_sum: 1.1,
        };

        assert!(is_better_support(&more_inliers, Some(&current)));
        assert!(is_better_support(&lower_residual_sum, Some(&current)));
        assert!(!is_better_support(&higher_residual_sum, Some(&current)));
    }

    #[test]
    fn pair_reprojection_error_uses_each_image_camera() {
        let camera1 = CameraModel::new_pinhole(640, 480, 500.0, 500.0, 320.0, 240.0);
        let camera2 = CameraModel::new_pinhole(800, 600, 900.0, 700.0, 30.0, 40.0);
        let point1 = Vector3::new(0.1, -0.05, 1.0);
        let point2 = Vector3::new(0.2, -0.1, 1.4);
        let obs1 = camera1.img_from_cam(point1.x, point1.y, point1.z).unwrap();
        let obs2 = camera2.img_from_cam(point2.x, point2.y, point2.z).unwrap();

        let err = pair_reprojection_error_px(
            &point1,
            &point2,
            [obs1[0] as f32, obs1[1] as f32],
            [obs2[0] as f32, obs2[1] as f32],
            camera1,
            camera2,
        );
        let wrong_right_camera_err = pair_reprojection_error_px(
            &point1,
            &point2,
            [obs1[0] as f32, obs1[1] as f32],
            [obs2[0] as f32, obs2[1] as f32],
            camera1,
            camera1,
        );

        assert!(err < 1.0e-4);
        assert!(wrong_right_camera_err > 100.0);
    }

    #[test]
    fn stationary_filter_removes_near_identical_image_matches() {
        let points1 = vec![[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]];
        let points2 = vec![[11.0, 10.0], [25.0, 20.0], [30.5, 30.5], [50.0, 40.0]];
        let active = active_match_indices(&points1, &points2, 4, true, 2.0);

        assert_eq!(active, vec![1, 3]);
    }

    #[test]
    fn watermark_detection_requires_border_translation_consensus() {
        let camera = CameraModel::new_pinhole(1000, 800, 700.0, 700.0, 500.0, 400.0);
        let mut points1 = Vec::new();
        let mut points2 = Vec::new();
        for i in 0..20 {
            let x = 20.0 + i as f32 * 2.0;
            let y = if i % 2 == 0 { 18.0 } else { 780.0 };
            points1.push([x, y]);
            points2.push([x + 12.0, y - 3.0]);
        }
        let mask = vec![true; points1.len()];
        let options = default_test_options();

        assert!(detect_watermark_matches(
            camera,
            camera,
            &points1,
            &points2,
            &mask,
            points1.len(),
            &options,
        ));
    }

    #[test]
    fn calibrated_classification_marks_homography_dominant_pairs_planar() {
        let e_support = ModelSupport {
            inlier_mask: vec![true, true, false, false],
            inliers: 2,
            residual_sum: 0.0,
        };
        let f_support = ModelSupport {
            inlier_mask: vec![true, true, true, false],
            inliers: 3,
            residual_sum: 0.0,
        };
        let h_support = ModelSupport {
            inlier_mask: vec![true, true, true, true],
            inliers: 4,
            residual_sum: 0.0,
        };
        let mut options = default_test_options();
        options.min_inliers = 3;

        let (config, mask, inliers) =
            classify_calibrated_two_view(&e_support, Some(&f_support), Some(&h_support), &options)
                .unwrap();

        assert_eq!(config, crate::database::COLMAP_TWO_VIEW_PLANAR_OR_PANORAMIC);
        assert_eq!(inliers, 4);
        assert_eq!(mask, h_support.inlier_mask);
    }

    #[test]
    fn watermark_detection_preserves_geometry_with_watermark_config() {
        let camera = CameraModel::new_pinhole(1000, 800, 700.0, 700.0, 500.0, 400.0);
        let mut pts1 = Vec::new();
        let mut pts2 = Vec::new();
        let mut obs1 = Vec::new();
        let mut obs2 = Vec::new();
        let pose = SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(1.0, 0.0, 0.0));
        for i in 0..40 {
            let x = 20.0 + i as f32 * 11.0;
            let y = if i % 2 == 0 {
                18.0 + (i % 7) as f32 * 8.0
            } else {
                780.0 - (i % 7) as f32 * 8.0
            };
            let p1 = camera.cam_from_img_f32(x, y).unwrap();
            let z = 4.0 + (i % 5) as f32 * 0.01;
            let point = [p1[0] * z, p1[1] * z, z];
            let p2_norm = pose.transform_point(&point);
            let p2_px = camera
                .img_from_cam_f32(p2_norm[0], p2_norm[1], p2_norm[2])
                .unwrap();
            pts1.push(p1);
            pts2.push([p2_norm[0] / p2_norm[2], p2_norm[1] / p2_norm[2]]);
            obs1.push([x, y]);
            obs2.push(p2_px);
        }
        let mut options = default_test_options();
        options.min_inliers = 20;
        options.min_triangulated = 8;
        options.ransac_threshold = 0.01;
        options.ransac_max_iterations = 1024;

        let estimate = estimate_calibrated_two_view_with_observations_and_cameras(
            &pts1, &pts2, &obs1, &obs2, camera, camera, &options,
        )
        .unwrap();

        assert_eq!(
            estimate.two_view_config,
            crate::database::COLMAP_TWO_VIEW_WATERMARK
        );
    }
}
