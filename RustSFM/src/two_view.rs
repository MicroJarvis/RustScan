use crate::five_point::estimate_five_point_essential;
use crate::geometry::relative_rotation_deg;
use crate::types::CameraModel;
use glam::{Quat, Vec3};
use nalgebra::{DMatrix, DVector, Matrix3, Matrix3x4, Rotation3, UnitQuaternion, Vector3};
use rustslam::{ColmapRandomSampler, ColmapRansacOptions, SE3};

#[derive(Debug, Clone)]
pub struct TwoViewOptions {
    pub ransac_max_error_px: f64,
    pub ransac_threshold: f64,
    pub ransac_min_inlier_ratio: f64,
    pub ransac_min_iterations: u32,
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
    pub fundamental: Option<Matrix3<f64>>,
    pub homography: Option<Matrix3<f64>>,
    pub e_matrix: Option<[f64; 9]>,
    pub qvec: Option<[f64; 4]>,
    pub tvec: Option<[f64; 3]>,
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
    let essential_sample_size = if options.use_five_point { 5 } else { 8 };
    let essential_ransac_options =
        two_view_ransac_options(options.ransac_threshold, options, essential_sample_size)?;
    let mut best: Option<(Matrix3<f64>, ModelSupport)> = None;
    let mut max_iterations = essential_ransac_options.max_num_trials.max(1) as u32;
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
                    &essential_ransac_options,
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
        two_view_ransac_options(options.ransac_max_error_px, options, 7)?,
        options.random_seed,
        options.loransac_num_lo_steps,
    );
    let h_support = estimate_homography_ransac(
        &img_pts1,
        &img_pts2,
        &active_indices,
        two_view_ransac_options(options.ransac_max_error_px, options, 4)?,
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
            two_view_config = classify_homography_motion(
                homography,
                camera1,
                camera2,
                &pts1,
                &pts2,
                &selected_mask,
            );
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

    let homography_pose_score = if matches!(
        two_view_config,
        crate::database::COLMAP_TWO_VIEW_PLANAR | crate::database::COLMAP_TWO_VIEW_PANORAMIC
    ) {
        h_support.as_ref().and_then(|(homography, _)| {
            choose_pose_from_homography(
                homography,
                camera1,
                camera2,
                &pts1,
                &pts2,
                &selected_mask,
                two_view_config,
            )
        })
    } else {
        None
    };
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
    let pose_score = if let Some(score) = homography_pose_score {
        score
    } else {
        choose_pose_from_essential(
            &pose_essential,
            &pts1,
            &pts2,
            obs1_px,
            obs2_px,
            &pose_mask,
            camera1,
            camera2,
        )?
    };
    if pose_score.triangulated < options.min_triangulated {
        return None;
    }
    let pose_rigid = rigid3_from_se3(pose_score.pose);

    Some(TwoViewEstimate {
        essential: pose_essential,
        fundamental: f_support.as_ref().map(|(model, _)| *model),
        homography: h_support.as_ref().map(|(model, _)| *model),
        e_matrix: Some(matrix3_to_row_array(pose_essential)),
        qvec: Some(pose_rigid.qvec),
        tvec: Some(pose_rigid.tvec),
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
        two_view_ransac_options(options.ransac_max_error_px, options, 4)?,
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

    let mut two_view_config = classify_homography_motion(
        &homography,
        camera1,
        camera2,
        pts1,
        pts2,
        &h_support.inlier_mask,
    );
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

    let homography_pose_score = if matches!(
        two_view_config,
        crate::database::COLMAP_TWO_VIEW_PLANAR | crate::database::COLMAP_TWO_VIEW_PANORAMIC
    ) {
        choose_pose_from_homography(
            &homography,
            camera1,
            camera2,
            pts1,
            pts2,
            &h_support.inlier_mask,
            two_view_config,
        )
    } else {
        None
    };

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
    let pose_score = if let Some(score) = homography_pose_score {
        score
    } else {
        choose_pose_from_essential(
            &essential, pts1, pts2, obs1_px, obs2_px, &pose_mask, camera1, camera2,
        )?
    };
    if pose_score.triangulated < options.min_triangulated {
        return None;
    }
    let pose_rigid = rigid3_from_se3(pose_score.pose);

    Some(TwoViewEstimate {
        essential,
        fundamental: None,
        homography: Some(homography),
        e_matrix: Some(matrix3_to_row_array(essential)),
        qvec: Some(pose_rigid.qvec),
        tvec: Some(pose_rigid.tvec),
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

pub(crate) fn estimate_relative_pose_from_rays(
    rays1: &[[f64; 3]],
    rays2: &[[f64; 3]],
    max_error: f64,
    min_inlier_ratio: f64,
    min_num_trials: usize,
    max_num_trials: usize,
    confidence: f64,
    dyn_num_trials_multiplier: f64,
    random_seed: i32,
) -> Option<(SE3, usize, Vec<bool>)> {
    const SAMPLE_SIZE: usize = 5;
    let n = rays1.len().min(rays2.len());
    if n < SAMPLE_SIZE {
        return None;
    }

    let pts1 = rays1
        .iter()
        .take(n)
        .map(|ray| Vector3::new(ray[0], ray[1], ray[2]))
        .collect::<Vec<_>>();
    let pts2 = rays2
        .iter()
        .take(n)
        .map(|ray| Vector3::new(ray[0], ray[1], ray[2]))
        .collect::<Vec<_>>();
    let active_indices = (0..n).collect::<Vec<_>>();
    let seed = if random_seed >= 0 {
        random_seed as u64
    } else {
        0x6a09_e667_f3bc_c909
    };
    let mut sampler = ColmapRandomSampler::new(seed, &active_indices);
    let ransac_options = relative_pose_ransac_options(
        max_error,
        min_inlier_ratio,
        min_num_trials,
        max_num_trials,
        confidence,
        dyn_num_trials_multiplier,
        random_seed,
        SAMPLE_SIZE,
    )?;
    let mut max_iterations = ransac_options.max_num_trials;

    let mut best: Option<(Matrix3<f64>, ModelSupport)> = None;
    let mut iteration = 0usize;
    while iteration < max_iterations {
        iteration += 1;
        let sample = sampler.sample(SAMPLE_SIZE);
        if sample.len() != SAMPLE_SIZE {
            break;
        }
        for model in estimate_essential_five_point_indexed(&pts1, &pts2, &sample) {
            let mut support =
                model_support_indexed(&pts1, &pts2, &active_indices, &model, max_error);
            if support.inliers < SAMPLE_SIZE {
                continue;
            }

            if is_better_support(&support, best.as_ref().map(|(_, support)| support)) {
                let mut local_model = model;
                for _ in 0..10 {
                    let previous_inliers = support.inliers;
                    if previous_inliers < SAMPLE_SIZE {
                        break;
                    }
                    let inliers = support
                        .inlier_mask
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, &is_inlier)| is_inlier.then_some(idx))
                        .collect::<Vec<_>>();
                    let local_models =
                        estimate_essential_five_point_indexed(&pts1, &pts2, &inliers);
                    let mut improved = false;
                    for candidate_model in local_models {
                        let candidate_support = model_support_indexed(
                            &pts1,
                            &pts2,
                            &active_indices,
                            &candidate_model,
                            max_error,
                        );
                        if is_better_support(&candidate_support, Some(&support)) {
                            local_model = candidate_model;
                            support = candidate_support;
                            improved = true;
                        }
                    }
                    if support.inliers <= previous_inliers || !improved {
                        break;
                    }
                }

                max_iterations = max_iterations.min(ransac_options.dynamic_max_num_trials(
                    support.inliers,
                    n,
                    SAMPLE_SIZE,
                ));
                best = Some((local_model, support));
            }
        }
    }

    let (essential, support) = best?;
    if support.inliers < SAMPLE_SIZE {
        return None;
    }
    let pose_score = choose_pose_from_essential(
        &essential,
        &pts1,
        &pts2,
        &rays_to_pixel_like_observations(rays1, n),
        &rays_to_pixel_like_observations(rays2, n),
        &support.inlier_mask,
        CameraModel::new_pinhole(1, 1, 1.0, 1.0, 0.0, 0.0),
        CameraModel::new_pinhole(1, 1, 1.0, 1.0, 0.0, 0.0),
    )?;
    Some((pose_score.pose, support.inliers, support.inlier_mask))
}

fn relative_pose_ransac_options(
    max_error: f64,
    min_inlier_ratio: f64,
    min_num_trials: usize,
    max_num_trials: usize,
    confidence: f64,
    dyn_num_trials_multiplier: f64,
    random_seed: i32,
    sample_size: usize,
) -> Option<ColmapRansacOptions> {
    ColmapRansacOptions {
        max_error,
        min_inlier_ratio,
        confidence,
        dyn_num_trials_multiplier,
        min_num_trials,
        max_num_trials,
        random_seed,
        num_threads: 1,
    }
    .with_initial_max_num_trials(sample_size)
    .ok()
}

fn two_view_ransac_options(
    max_error: f64,
    options: &TwoViewOptions,
    sample_size: usize,
) -> Option<ColmapRansacOptions> {
    let min_inlier_ratio = if options.min_inlier_ratio > 0.0 {
        options.min_inlier_ratio
    } else {
        options.ransac_min_inlier_ratio
    };
    let max_num_trials = options.ransac_max_iterations.max(1) as usize;
    let min_num_trials = (options.ransac_min_iterations as usize).min(max_num_trials);
    ColmapRansacOptions {
        max_error,
        min_inlier_ratio,
        confidence: 0.999,
        min_num_trials,
        max_num_trials,
        ..ColmapRansacOptions::default()
    }
    .with_initial_max_num_trials(sample_size)
    .ok()
}

fn rays_to_pixel_like_observations(rays: &[[f64; 3]], n: usize) -> Vec<[f32; 2]> {
    rays.iter()
        .take(n)
        .map(|ray| {
            let z = if ray[2].abs() > 1.0e-12 { ray[2] } else { 1.0 };
            [(ray[0] / z) as f32, (ray[1] / z) as f32]
        })
        .collect()
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
    options: &ColmapRansacOptions,
    sample_size: usize,
) -> u32 {
    let max_iterations = options.max_num_trials.max(1);
    let num_trials = options.dynamic_max_num_trials(inliers, total, sample_size);
    num_trials.clamp(1, max_iterations) as u32
}

fn estimate_fundamental_ransac(
    pts1: &[Vector3<f64>],
    pts2: &[Vector3<f64>],
    active_indices: &[usize],
    ransac_options: ColmapRansacOptions,
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
    let mut max_iterations = ransac_options.max_num_trials.max(1) as u32;
    let mut iteration = 0u32;
    while iteration < max_iterations {
        iteration += 1;
        let sample = sampler.sample(7);
        if sample.len() != 7 {
            continue;
        }
        let models = estimate_fundamental_seven_point_indexed(pts1, pts2, &sample);
        for model in models {
            let support =
                model_support_indexed(pts1, pts2, active_indices, &model, ransac_options.max_error);
            if support.inliers >= 7 && is_better_support(&support, best.as_ref().map(|(_, s)| s)) {
                max_iterations = max_iterations.min(adaptive_ransac_iterations(
                    support.inliers,
                    active_indices.len(),
                    &ransac_options,
                    7,
                ));
                best = Some((model, support));
            }
        }
    }
    let (model, support) = best.or_else(|| {
        estimate_fundamental_eight_point_indexed(pts1, pts2, active_indices).map(|model| {
            let support =
                model_support_indexed(pts1, pts2, active_indices, &model, ransac_options.max_error);
            (model, support)
        })
    })?;
    Some(refine_fundamental_support(
        pts1,
        pts2,
        active_indices,
        ransac_options.max_error,
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
    let Some(basis) = right_nullspace_householder(&a, 2) else {
        return Vec::new();
    };
    let mut f1 = basis[0];
    let f2 = basis[1];
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
    let basis = right_nullspace_householder(a, 1)?;
    let q = basis.first()?;
    Some(*q)
}

fn right_nullspace_householder(a: &DMatrix<f64>, nullity: usize) -> Option<Vec<[f64; 9]>> {
    if a.ncols() != 9 || a.nrows() + nullity != 9 {
        return None;
    }
    let qr = a.transpose().qr();
    let mut q_t = DMatrix::<f64>::identity(9, 9);
    qr.q_tr_mul(&mut q_t);
    let first = 9 - nullity;
    Some(
        (first..9)
            .map(|row| {
                [
                    q_t[(row, 0)],
                    q_t[(row, 1)],
                    q_t[(row, 2)],
                    q_t[(row, 3)],
                    q_t[(row, 4)],
                    q_t[(row, 5)],
                    q_t[(row, 6)],
                    q_t[(row, 7)],
                    q_t[(row, 8)],
                ]
            })
            .collect(),
    )
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
    ransac_options: ColmapRansacOptions,
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
    let mut max_iterations = ransac_options.max_num_trials.max(1) as u32;
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
        let support = homography_support_indexed(
            pts1,
            pts2,
            active_indices,
            &model,
            ransac_options.max_error,
        );
        if support.inliers >= 4 && is_better_support(&support, best.as_ref().map(|(_, s)| s)) {
            max_iterations = max_iterations.min(adaptive_ransac_iterations(
                support.inliers,
                active_indices.len(),
                &ransac_options,
                4,
            ));
            best = Some((model, support));
        }
    }
    let (model, support) = best.or_else(|| {
        estimate_homography_dlt_indexed(pts1, pts2, active_indices).map(|model| {
            let support = homography_support_indexed(
                pts1,
                pts2,
                active_indices,
                &model,
                ransac_options.max_error,
            );
            (model, support)
        })
    })?;
    Some(refine_homography_support(
        pts1,
        pts2,
        active_indices,
        ransac_options.max_error,
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
    rays1: &[Vector3<f64>],
    rays2: &[Vector3<f64>],
    inlier_mask: &[bool],
) -> i32 {
    if let Some((translation_norm_sq, _triangulated)) =
        pose_from_homography_matrix(homography, camera1, camera2, rays1, rays2, inlier_mask)
    {
        if translation_norm_sq < 1.0e-12 {
            return crate::database::COLMAP_TWO_VIEW_PANORAMIC;
        }
        return crate::database::COLMAP_TWO_VIEW_PLANAR;
    }
    classify_homography_motion_by_rotation(homography, camera1, camera2)
}

fn classify_homography_motion_by_rotation(
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

fn choose_pose_from_homography(
    homography: &Matrix3<f64>,
    camera1: CameraModel,
    camera2: CameraModel,
    rays1: &[Vector3<f64>],
    rays2: &[Vector3<f64>],
    inlier_mask: &[bool],
    two_view_config: i32,
) -> Option<PoseCandidateScore> {
    let candidates = decompose_homography_matrix(homography, camera1, camera2)?;
    let mut best: Option<(HomographyPoseCandidate, usize, f64, Vec<Vector3<f64>>)> = None;
    for candidate in candidates {
        let mut points = Vec::new();
        let mut residual_sum = 0.0;
        for (idx, &is_inlier) in inlier_mask.iter().enumerate() {
            if !is_inlier {
                continue;
            }
            let (Some(ray1), Some(ray2)) = (rays1.get(idx), rays2.get(idx)) else {
                continue;
            };
            let Some(point) =
                triangulate_midpoint(&candidate.rotation, &candidate.translation, ray1, ray2)
            else {
                continue;
            };
            let point2 = candidate.rotation * point + candidate.translation;
            residual_sum += 1.0 - clamp_unit(ray1.normalize().dot(&point.normalize()));
            residual_sum += 1.0 - clamp_unit(ray2.normalize().dot(&point2.normalize()));
            points.push(point);
        }
        if best.as_ref().is_none_or(
            |(_, best_count, best_residual, _): &(
                HomographyPoseCandidate,
                usize,
                f64,
                Vec<Vector3<f64>>,
            )| {
                points.len() > *best_count
                    || (points.len() == *best_count && residual_sum < *best_residual)
            },
        ) {
            best = Some((candidate, points.len(), residual_sum, points));
        }
    }
    let (candidate, triangulated, residual_sum, points) = best?;
    if two_view_config == crate::database::COLMAP_TWO_VIEW_PLANAR && triangulated == 0 {
        return None;
    }
    let pose = se3_from_parts(&candidate.rotation, &candidate.translation)?;
    let median_angle_deg = if two_view_config == crate::database::COLMAP_TWO_VIEW_PANORAMIC {
        0.0
    } else {
        let center2 = -candidate.rotation.transpose() * candidate.translation;
        let mut angles = points
            .iter()
            .map(|point| triangulation_angle_deg(&Vector3::zeros(), &center2, point))
            .filter(|angle| angle.is_finite())
            .collect::<Vec<_>>();
        if angles.is_empty() {
            0.0
        } else {
            angles.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            angles[angles.len() / 2]
        }
    };
    Some(PoseCandidateScore {
        pose,
        triangulated,
        mean_reprojection_error_px: if triangulated == 0 {
            0.0
        } else {
            (residual_sum / triangulated as f64) as f32
        },
        median_angle_deg,
    })
}

fn pose_from_homography_matrix(
    homography: &Matrix3<f64>,
    camera1: CameraModel,
    camera2: CameraModel,
    rays1: &[Vector3<f64>],
    rays2: &[Vector3<f64>],
    inlier_mask: &[bool],
) -> Option<(f64, usize)> {
    let candidates = decompose_homography_matrix(homography, camera1, camera2)?;
    let mut best_translation_norm_sq = 0.0;
    let mut best_points = 0usize;
    let mut best_residual_sum = f64::MAX;
    for candidate in candidates {
        let mut points = 0usize;
        let mut residual_sum = 0.0;
        for (idx, &is_inlier) in inlier_mask.iter().enumerate() {
            if !is_inlier {
                continue;
            }
            let (Some(ray1), Some(ray2)) = (rays1.get(idx), rays2.get(idx)) else {
                continue;
            };
            let Some(point) =
                triangulate_midpoint(&candidate.rotation, &candidate.translation, ray1, ray2)
            else {
                continue;
            };
            let point2 = candidate.rotation * point + candidate.translation;
            let err1 = 1.0 - clamp_unit(ray1.normalize().dot(&point.normalize()));
            let err2 = 1.0 - clamp_unit(ray2.normalize().dot(&point2.normalize()));
            residual_sum += err1 + err2;
            points += 1;
        }
        if points > best_points || (points == best_points && residual_sum < best_residual_sum) {
            best_points = points;
            best_residual_sum = residual_sum;
            best_translation_norm_sq = candidate.translation.norm_squared();
        }
    }
    (best_points > 0).then_some((best_translation_norm_sq, best_points))
}

#[derive(Clone, Copy, Debug)]
struct HomographyPoseCandidate {
    rotation: Matrix3<f64>,
    translation: Vector3<f64>,
    #[allow(dead_code)]
    normal: Vector3<f64>,
}

fn decompose_homography_matrix(
    homography: &Matrix3<f64>,
    camera1: CameraModel,
    camera2: CameraModel,
) -> Option<Vec<HomographyPoseCandidate>> {
    let k1 = camera_intrinsic_matrix(camera1);
    let k2_inv = camera_intrinsic_matrix(camera2).try_inverse()?;
    let mut h_norm = k2_inv * homography * k1;
    let svd = h_norm.svd(false, false);
    if svd.singular_values.len() < 2 || svd.singular_values[1].abs() <= 1.0e-12 {
        return None;
    }
    h_norm /= svd.singular_values[1];
    if h_norm.determinant() < 0.0 {
        h_norm *= -1.0;
    }

    let s = h_norm.transpose() * h_norm - Matrix3::<f64>::identity();
    if max_abs_coeff(&s) < 1.0e-3 {
        return Some(vec![HomographyPoseCandidate {
            rotation: h_norm,
            translation: Vector3::zeros(),
            normal: Vector3::zeros(),
        }]);
    }

    let m00 = opposite_minor(&s, 0, 0);
    let m11 = opposite_minor(&s, 1, 1);
    let m22 = opposite_minor(&s, 2, 2);
    let rtm00 = m00.max(0.0).sqrt();
    let rtm11 = m11.max(0.0).sqrt();
    let rtm22 = m22.max(0.0).sqrt();
    let m01 = opposite_minor(&s, 0, 1);
    let m12 = opposite_minor(&s, 1, 2);
    let m02 = opposite_minor(&s, 0, 2);
    let e12 = sign_of_number(m12);
    let e02 = sign_of_number(m02);
    let e01 = sign_of_number(m01);
    let ns = [s[(0, 0)].abs(), s[(1, 1)].abs(), s[(2, 2)].abs()];
    let idx = ns
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)?;

    let (np1, np2) = match idx {
        0 => (
            Vector3::new(s[(0, 0)], s[(0, 1)] + rtm22, s[(0, 2)] + e12 * rtm11),
            Vector3::new(s[(0, 0)], s[(0, 1)] - rtm22, s[(0, 2)] - e12 * rtm11),
        ),
        1 => (
            Vector3::new(s[(0, 1)] + rtm22, s[(1, 1)], s[(1, 2)] - e02 * rtm00),
            Vector3::new(s[(0, 1)] - rtm22, s[(1, 1)], s[(1, 2)] + e02 * rtm00),
        ),
        2 => (
            Vector3::new(s[(0, 2)] + e01 * rtm11, s[(1, 2)] + rtm00, s[(2, 2)]),
            Vector3::new(s[(0, 2)] - e01 * rtm11, s[(1, 2)] - rtm00, s[(2, 2)]),
        ),
        _ => return None,
    };
    let trace_s = s.trace();
    let v = 2.0 * (1.0 + trace_s - m00 - m11 - m22).max(0.0).sqrt();
    if !v.is_finite() || v.abs() <= 1.0e-12 {
        return None;
    }
    let esii = sign_of_number(s[(idx, idx)]);
    let r = (2.0 + trace_s + v).max(0.0).sqrt();
    let n_t = (2.0 + trace_s - v).max(0.0).sqrt();
    let n1 = np1.try_normalize(1.0e-12)?;
    let n2 = np2.try_normalize(1.0e-12)?;
    let half_nt = 0.5 * n_t;
    let esii_t_r = esii * r;
    let t1_star = half_nt * (esii_t_r * n2 - n_t * n1);
    let t2_star = half_nt * (esii_t_r * n1 - n_t * n2);
    let r1 = compute_homography_rotation(&h_norm, &t1_star, &n1, v);
    let t1 = r1 * t1_star;
    let r2 = compute_homography_rotation(&h_norm, &t2_star, &n2, v);
    let t2 = r2 * t2_star;
    Some(vec![
        HomographyPoseCandidate {
            rotation: r1,
            translation: t1,
            normal: -n1,
        },
        HomographyPoseCandidate {
            rotation: r1,
            translation: -t1,
            normal: n1,
        },
        HomographyPoseCandidate {
            rotation: r2,
            translation: t2,
            normal: -n2,
        },
        HomographyPoseCandidate {
            rotation: r2,
            translation: -t2,
            normal: n2,
        },
    ])
}

fn compute_homography_rotation(
    h_normalized: &Matrix3<f64>,
    tstar: &Vector3<f64>,
    normal: &Vector3<f64>,
    v: f64,
) -> Matrix3<f64> {
    h_normalized * (Matrix3::<f64>::identity() - (2.0 / v) * (tstar * normal.transpose()))
}

fn triangulate_midpoint(
    r: &Matrix3<f64>,
    t: &Vector3<f64>,
    ray1: &Vector3<f64>,
    ray2: &Vector3<f64>,
) -> Option<Vector3<f64>> {
    let cam1_from_cam2_rotation = r.transpose();
    let ray2_in_cam1 = cam1_from_cam2_rotation * ray2;
    let cam2_in_cam1 = cam1_from_cam2_rotation * -t;
    let a = Matrix3::<f64>::from_columns(&[*ray1, -ray2_in_cam1, -cam2_in_cam1]);
    let svd = a.svd(false, true);
    let vt = svd.v_t?;
    if vt.nrows() < 3 || vt[(2, 2)].abs() <= f64::EPSILON {
        return None;
    }
    let lambda0 = vt[(2, 0)] / vt[(2, 2)];
    let lambda1 = vt[(2, 1)] / vt[(2, 2)];
    if lambda0 <= f64::EPSILON || lambda1 <= f64::EPSILON {
        return None;
    }
    Some(0.5 * (lambda0 * ray1 + cam2_in_cam1 + lambda1 * ray2_in_cam1))
}

fn opposite_minor(matrix: &Matrix3<f64>, row: usize, col: usize) -> f64 {
    let col1 = if col == 0 { 1 } else { 0 };
    let col2 = if col == 2 { 1 } else { 2 };
    let row1 = if row == 0 { 1 } else { 0 };
    let row2 = if row == 2 { 1 } else { 2 };
    matrix[(row1, col2)] * matrix[(row2, col1)] - matrix[(row1, col1)] * matrix[(row2, col2)]
}

fn sign_of_number(value: f64) -> f64 {
    if value >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

fn max_abs_coeff(matrix: &Matrix3<f64>) -> f64 {
    matrix
        .iter()
        .fold(0.0f64, |max, value| max.max(value.abs()))
}

fn clamp_unit(value: f64) -> f64 {
    value.clamp(-1.0, 1.0)
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
    // First camera at the origin (`[I | 0]`), second at the relative pose
    // `[R | t]`. Delegate the DLT to the COLMAP-faithful triangulation module.
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
    let point = crate::triangulation::triangulate_point(
        &p1,
        &p2,
        &nalgebra::Vector2::new(x1.x, x1.y),
        &nalgebra::Vector2::new(x2.x, x2.y),
    )?;
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

fn rigid3_from_se3(pose: SE3) -> crate::types::Rigid3 {
    let q = pose.quaternion();
    let t = pose.translation();
    crate::types::Rigid3 {
        qvec: [q[3] as f64, q[0] as f64, q[1] as f64, q[2] as f64],
        tvec: [t[0] as f64, t[1] as f64, t[2] as f64],
    }
}

fn matrix3_to_row_array(matrix: Matrix3<f64>) -> [f64; 9] {
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

#[cfg(test)]
mod tests {
    use super::*;
    use rustslam::{colmap_ransac_num_trials, ColmapMt19937, ColmapRandomSampler};

    fn default_test_options() -> TwoViewOptions {
        TwoViewOptions {
            ransac_max_error_px: 4.0,
            ransac_threshold: 0.01,
            ransac_min_inlier_ratio: 0.25,
            ransac_min_iterations: 100,
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

    fn test_ransac_options(
        min_iterations: u32,
        max_iterations: u32,
        confidence: f64,
    ) -> ColmapRansacOptions {
        ColmapRansacOptions {
            max_error: 1.0,
            confidence,
            min_num_trials: min_iterations as usize,
            max_num_trials: max_iterations as usize,
            ..ColmapRansacOptions::default()
        }
    }

    #[test]
    fn adaptive_ransac_iterations_matches_colmap_trial_formula() {
        let options = test_ransac_options(0, 10_000, 0.999);
        assert_eq!(adaptive_ransac_iterations(50, 100, &options, 5), 726);
        assert_eq!(adaptive_ransac_iterations(50, 100, &options, 8), 7173);
        assert_eq!(adaptive_ransac_iterations(90, 100, &options, 5), 24);
    }

    #[test]
    fn adaptive_ransac_iterations_respects_colmap_min_trial_floor() {
        let options = test_ransac_options(100, 10_000, 0.999);
        assert_eq!(adaptive_ransac_iterations(90, 100, &options, 5), 100);
        let options = test_ransac_options(128, 128, 0.999);
        assert_eq!(adaptive_ransac_iterations(90, 100, &options, 5), 128);
    }

    #[test]
    fn adaptive_ransac_iterations_keeps_full_budget_for_invalid_support() {
        let options = test_ransac_options(0, 123, 0.999);
        assert_eq!(adaptive_ransac_iterations(3, 100, &options, 5), 123);
        assert_eq!(adaptive_ransac_iterations(10, 4, &options, 5), 123);
        assert_eq!(adaptive_ransac_iterations(10, 10, &options, 5), 1);
    }

    #[test]
    fn two_view_ransac_options_use_colmap_prior_inlier_initial_clamp() {
        let options = default_test_options();

        let essential_options = two_view_ransac_options(0.01, &options, 5).unwrap();
        assert_eq!(essential_options.min_inlier_ratio, 0.25);
        assert_eq!(essential_options.min_num_trials, 100);
        assert_eq!(essential_options.max_num_trials, 128);

        let mut full_budget_options = options.clone();
        full_budget_options.ransac_max_iterations = 10_000;
        let essential_options = two_view_ransac_options(0.01, &full_budget_options, 5).unwrap();
        assert_eq!(essential_options.max_num_trials, 10_000);

        let homography_options = two_view_ransac_options(4.0, &full_budget_options, 4).unwrap();
        assert_eq!(
            homography_options.max_num_trials,
            colmap_ransac_num_trials(25_000, 100_000, 4, 0.999, 3.0)
        );

        let mut filtered_options = full_budget_options.clone();
        filtered_options.min_inlier_ratio = 0.5;
        let filtered = two_view_ransac_options(0.01, &filtered_options, 5).unwrap();
        assert_eq!(filtered.min_inlier_ratio, 0.5);
        assert_eq!(
            filtered.max_num_trials,
            colmap_ransac_num_trials(50_000, 100_000, 5, 0.999, 3.0).min(10_000)
        );

        let mut tiny_budget_options = options.clone();
        tiny_budget_options.ransac_max_iterations = 50;
        let tiny = two_view_ransac_options(0.01, &tiny_budget_options, 5).unwrap();
        assert_eq!(tiny.min_num_trials, 50);
        assert_eq!(tiny.max_num_trials, 50);
    }

    #[test]
    fn ray_relative_pose_ransac_options_use_shared_colmap_initialization() {
        let options =
            relative_pose_ransac_options(0.01, 0.5, 3, 10_000, 0.999, 1.0, 42, 5).unwrap();
        assert_eq!(
            options.max_num_trials,
            colmap_ransac_num_trials(50_000, 100_000, 5, 0.999, 1.0)
        );
        assert_eq!(options.min_num_trials, 3);
        assert_eq!(options.random_seed, 42);

        assert!(relative_pose_ransac_options(0.0, 0.5, 0, 10_000, 0.999, 1.0, -1, 5).is_none());
        let zero_prior =
            relative_pose_ransac_options(0.01, 0.0, 0, 10_000, 0.999, 1.0, -1, 5).unwrap();
        assert_eq!(zero_prior.max_num_trials, 10_000);
    }

    #[test]
    fn ray_relative_pose_uses_shared_dynamic_trial_bounds() {
        let options =
            relative_pose_ransac_options(0.01, 0.25, 100, 10_000, 0.999, 3.0, 42, 5).unwrap();
        assert_eq!(options.dynamic_max_num_trials(50, 100, 5), 726);
        assert_eq!(options.dynamic_max_num_trials(100, 100, 5), 100);
        assert_eq!(options.dynamic_max_num_trials(3, 100, 5), 10_000);
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
    fn eight_point_minimal_nullspace_uses_full_householder_q_column() {
        let a = DMatrix::<f64>::from_row_slice(
            8,
            9,
            &[
                0.2, -0.1, 1.0, 0.3, 0.5, -0.4, 1.1, -0.7, 0.9, -0.6, 0.8, 0.2, 0.4, -1.2, 0.7,
                -0.3, 0.6, 1.0, 1.4, -0.9, 0.5, -0.8, 0.1, 0.3, 0.7, 1.2, -0.5, -1.1, 0.4, 0.6,
                0.9, -0.2, 1.3, -0.7, 0.5, 0.8, 0.3, 1.1, -0.4, -0.6, 0.2, 0.9, -1.0, 0.7, 1.5,
                -0.2, 0.5, 1.0, -0.8, 0.4, 0.6, -1.1, 0.2, 0.9, -0.3, 0.6, -0.5, 1.2, 0.7, -0.9,
                0.8, 0.3, 1.0, -0.4, 0.2, -1.3, 0.5, 1.1, -0.7, 0.9, -0.1, 0.4,
            ],
        );

        let q = eight_point_minimal_nullspace(&a).unwrap();
        let residual = (&a * DVector::<f64>::from_column_slice(&q)).norm();
        let norm = q.iter().map(|value| value * value).sum::<f64>().sqrt();

        assert!(residual < 1.0e-10, "residual={residual}");
        assert!((norm - 1.0).abs() < 1.0e-12);
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

    #[test]
    fn homography_motion_classification_uses_pose_translation() {
        let camera = CameraModel::new_pinhole(640, 480, 500.0, 500.0, 320.0, 240.0);
        let k = camera_intrinsic_matrix(camera);
        let rotation = Rotation3::from_euler_angles(0.03, -0.02, 0.05).into_inner();
        let pure_rotation_h = k * rotation * k.try_inverse().unwrap();
        let scene_points = [
            Vector3::new(-0.4, -0.2, 3.0),
            Vector3::new(0.1, -0.3, 4.0),
            Vector3::new(0.5, -0.1, 3.5),
            Vector3::new(-0.2, 0.4, 4.2),
            Vector3::new(0.4, 0.3, 3.8),
            Vector3::new(-0.6, 0.15, 4.5),
            Vector3::new(0.25, -0.45, 3.7),
            Vector3::new(0.7, 0.25, 4.8),
        ];
        let rays1 = scene_points
            .iter()
            .map(|p| Vector3::new(p.x / p.z, p.y / p.z, 1.0).normalize())
            .collect::<Vec<_>>();
        let rays2 = rays1
            .iter()
            .map(|ray| (rotation * ray).normalize())
            .collect::<Vec<_>>();
        let inliers = vec![true; rays1.len()];

        assert_eq!(
            classify_homography_motion(&pure_rotation_h, camera, camera, &rays1, &rays2, &inliers),
            crate::database::COLMAP_TWO_VIEW_PANORAMIC
        );

        let translation = Vector3::new(0.2, -0.03, 0.05);
        let normal = Vector3::new(0.0, 0.0, 1.0);
        let distance = 3.0;
        let planar_h =
            k * (rotation - translation * normal.transpose() / distance) * k.try_inverse().unwrap();
        let rays2 = rays1
            .iter()
            .map(|ray| {
                let point = distance * ray / normal.dot(ray);
                (rotation * point + translation).normalize()
            })
            .collect::<Vec<_>>();

        assert_eq!(
            classify_homography_motion(&planar_h, camera, camera, &rays1, &rays2, &inliers),
            crate::database::COLMAP_TWO_VIEW_PLANAR
        );
    }

    #[test]
    fn homography_decomposition_matches_colmap_nominal_reference() {
        let mut h = Matrix3::new(
            2.649157564634028,
            4.583875997496426,
            70.694447785121326,
            -1.072756858861583,
            3.533262150437228,
            1513.656999614321649,
            0.001303887589576,
            0.003042206876298,
            1.0,
        );
        h *= 3.0;
        let camera = CameraModel::new_pinhole(640, 480, 640.0, 640.0, 320.0, 240.0);
        let candidates = decompose_homography_matrix(&h, camera, camera).expect("candidates");
        assert_eq!(candidates.len(), 4);

        let ref_rotation = Matrix3::new(
            0.43307983549125,
            0.545749113549648,
            -0.717356090899523,
            -0.85630229674426,
            0.497582023798831,
            -0.138414255706431,
            0.281404038139784,
            0.67421809131173,
            0.682818960388909,
        );
        let ref_translation = Vector3::new(1.826751712278038, 1.264718492450820, 0.195080809998819);
        let ref_normal = Vector3::new(-0.244875830334816, -0.480857890778889, -0.841909446789566);

        assert!(
            candidates.iter().any(|candidate| {
                (candidate.rotation - ref_rotation).norm() < 1.0e-6
                    && (candidate.translation - ref_translation).norm() < 1.0e-6
                    && (candidate.normal - ref_normal).norm() < 1.0e-6
            }),
            "reference homography solution missing: {candidates:?}"
        );
    }

    #[test]
    fn pose_from_homography_matches_colmap_nominal_reference() {
        let rotation =
            UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(1.0, 0.1, 0.2, 0.3))
                .to_rotation_matrix()
                .into_inner();
        let ref_translation = Vector3::new(1.0, 0.0, 0.0);
        let ref_normal = Vector3::new(0.0, 0.0, -1.0);
        let h = rotation - ref_translation * ref_normal.transpose();
        let rays1 = vec![
            Vector3::new(0.1, 0.1, 1.0).normalize(),
            Vector3::new(0.4, 0.1, 1.0).normalize(),
            Vector3::new(0.1, 0.4, 1.0).normalize(),
            Vector3::new(0.4, 0.4, 1.0).normalize(),
            Vector3::new(0.0, 0.0, 1.0).normalize(),
        ];
        let rays2 = rays1
            .iter()
            .map(|ray| (h * ray).normalize())
            .collect::<Vec<_>>();
        let camera = CameraModel::new_pinhole(1, 1, 1.0, 1.0, 0.0, 0.0);
        let candidates = decompose_homography_matrix(&h, camera, camera).expect("candidates");
        let inliers = vec![true; rays1.len()];
        let mut best: Option<(HomographyPoseCandidate, usize, f64)> = None;
        for candidate in candidates {
            let mut points = 0usize;
            let mut residual_sum = 0.0;
            for idx in 0..rays1.len() {
                let Some(point) = triangulate_midpoint(
                    &candidate.rotation,
                    &candidate.translation,
                    &rays1[idx],
                    &rays2[idx],
                ) else {
                    continue;
                };
                let point2 = candidate.rotation * point + candidate.translation;
                residual_sum += 1.0 - clamp_unit(rays1[idx].dot(&point.normalize()));
                residual_sum += 1.0 - clamp_unit(rays2[idx].dot(&point2.normalize()));
                points += 1;
            }
            if best.as_ref().is_none_or(|(_, best_points, best_residual)| {
                points > *best_points || (points == *best_points && residual_sum < *best_residual)
            }) {
                best = Some((candidate, points, residual_sum));
            }
        }
        let (best, points, _) = best.expect("best pose");
        assert_eq!(points, inliers.len());
        assert!((best.rotation - rotation).norm() < 1.0e-6);
        assert!((best.translation.normalize() - ref_translation).norm() < 1.0e-6);
        assert!((best.normal - ref_normal).norm() < 1.0e-5);
    }

    #[test]
    fn planar_geometry_uses_homography_pose_handoff() {
        let rotation =
            UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(1.0, 0.1, 0.2, 0.3))
                .to_rotation_matrix()
                .into_inner();
        let ref_translation = Vector3::new(1.0, 0.0, 0.0);
        let ref_normal = Vector3::new(0.0, 0.0, -1.0);
        let h = rotation - ref_translation * ref_normal.transpose();
        let rays1 = vec![
            Vector3::new(0.1, 0.1, 1.0).normalize(),
            Vector3::new(0.4, 0.1, 1.0).normalize(),
            Vector3::new(0.1, 0.4, 1.0).normalize(),
            Vector3::new(0.4, 0.4, 1.0).normalize(),
            Vector3::new(0.0, 0.0, 1.0).normalize(),
        ];
        let rays2 = rays1
            .iter()
            .map(|ray| (h * ray).normalize())
            .collect::<Vec<_>>();
        let camera = CameraModel::new_pinhole(1, 1, 1.0, 1.0, 0.0, 0.0);
        let inliers = vec![true; rays1.len()];
        let score = choose_pose_from_homography(
            &h,
            camera,
            camera,
            &rays1,
            &rays2,
            &inliers,
            crate::database::COLMAP_TWO_VIEW_PLANAR,
        )
        .expect("homography pose");

        assert_eq!(score.triangulated, rays1.len());
        let pose_rotation = Matrix3::from_row_slice(&[
            score.pose.rotation_matrix()[0][0] as f64,
            score.pose.rotation_matrix()[0][1] as f64,
            score.pose.rotation_matrix()[0][2] as f64,
            score.pose.rotation_matrix()[1][0] as f64,
            score.pose.rotation_matrix()[1][1] as f64,
            score.pose.rotation_matrix()[1][2] as f64,
            score.pose.rotation_matrix()[2][0] as f64,
            score.pose.rotation_matrix()[2][1] as f64,
            score.pose.rotation_matrix()[2][2] as f64,
        ]);
        let pose_translation = Vector3::new(
            score.pose.translation()[0] as f64,
            score.pose.translation()[1] as f64,
            score.pose.translation()[2] as f64,
        );
        assert!((pose_rotation - rotation).norm() < 1.0e-6);
        assert!((pose_translation.normalize() - ref_translation).norm() < 1.0e-6);
        assert!(score.median_angle_deg > 0.0);
    }

    #[test]
    fn two_view_estimate_carries_colmap_pose_metadata() {
        let rotation = Rotation3::from_euler_angles(0.03, -0.04, 0.02).into_inner();
        let translation = Vector3::new(0.2, -0.03, 0.05).normalize();
        let points_world = [
            Vector3::new(-0.4, -0.2, 3.0),
            Vector3::new(0.1, -0.3, 4.0),
            Vector3::new(0.5, -0.1, 3.5),
            Vector3::new(-0.2, 0.4, 4.2),
            Vector3::new(0.4, 0.3, 3.8),
            Vector3::new(-0.6, 0.15, 4.5),
            Vector3::new(0.25, -0.45, 3.7),
            Vector3::new(0.7, 0.25, 4.8),
            Vector3::new(-0.35, 0.45, 5.0),
            Vector3::new(0.55, -0.25, 4.4),
        ];
        let pts1 = points_world
            .iter()
            .map(|p| [p.x as f32 / p.z as f32, p.y as f32 / p.z as f32])
            .collect::<Vec<_>>();
        let pts2 = points_world
            .iter()
            .map(|p| {
                let q = rotation * p + translation;
                [q.x as f32 / q.z as f32, q.y as f32 / q.z as f32]
            })
            .collect::<Vec<_>>();
        let camera = CameraModel::new_pinhole(640, 480, 500.0, 500.0, 320.0, 240.0);
        let mut options = default_test_options();
        options.min_inliers = 8;
        options.min_triangulated = 4;
        options.ransac_max_iterations = 128;
        let estimate = estimate_calibrated_two_view(&pts1, &pts2, camera, &options).unwrap();

        assert_eq!(
            estimate.e_matrix.unwrap(),
            matrix3_to_row_array(estimate.essential)
        );
        assert!(estimate.qvec.is_some());
        assert!(estimate.tvec.is_some());
        assert!(estimate.qvec.unwrap().iter().all(|value| value.is_finite()));
        assert!(estimate.tvec.unwrap().iter().all(|value| value.is_finite()));
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
