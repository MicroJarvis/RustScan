//! GLOMAP-style view graph calibration.
//!
//! This stage refines the two-view graph before global rotation averaging and
//! track establishment:
//!
//! 1. **Pairwise match filtering** — keep only inliers that are consistent with
//!    the classified two-view model (E/F/H), pass a cheirality test, exceed a
//!    minimum triangulation angle, and are not too close to an epipole.
//! 2. **Optional intrinsics refinement** — scale shared focal length to reduce
//!    symmetric epipolar error across calibrated edges (Sweeney-style lite).
//! 3. **Optional relative-pose re-estimation** from the filtered matches.
//! 4. **Rotation-consistency filtering** — drop edges whose relative rotation
//!    disagrees with a global rotation-averaging solution (GLOMAP §3).

use crate::database::{
    COLMAP_TWO_VIEW_CALIBRATED, COLMAP_TWO_VIEW_CALIBRATED_RIG, COLMAP_TWO_VIEW_DEGENERATE,
    COLMAP_TWO_VIEW_MULTIPLE, COLMAP_TWO_VIEW_PANORAMIC, COLMAP_TWO_VIEW_PLANAR,
    COLMAP_TWO_VIEW_PLANAR_OR_PANORAMIC, COLMAP_TWO_VIEW_UNCALIBRATED, COLMAP_TWO_VIEW_UNDEFINED,
    COLMAP_TWO_VIEW_WATERMARK,
};
use crate::geometry::{estimate_pair_geometry_with_options, pose_rotation, PairEstimationOptions};
use crate::rotation_averaging::{
    estimate_global_rotations, relative_rotations_from_pairs, RotationAveragingOptions,
};
use crate::triangulation::triangulate_mid_point;
use crate::types::{CameraModel, ImageFrame, PairGeometry};
use glam::{Quat, Vec3};
use nalgebra::{Matrix3, Vector3};
use rustslam::{Match, SE3};

/// Options for [`calibrate_view_graph`].
#[derive(Debug, Clone, Copy)]
pub struct ViewGraphCalibrationOptions {
    /// Run view-graph calibration. When false, pairs and camera are returned
    /// unchanged.
    pub enabled: bool,
    /// Maximum Sampson / transfer error when re-checking inliers (pixels).
    pub max_epipolar_error_px: f32,
    /// Minimum per-match triangulation angle (degrees).
    pub min_triangulation_angle_deg: f32,
    /// Reject matches whose pixel distance to either epipole falls below this.
    pub max_epipole_distance_px: f32,
    /// Drop edges whose relative rotation disagrees with global averaging by
    /// more than this angle (degrees).
    pub max_rotation_error_deg: f64,
    /// Minimum surviving inliers required to keep an edge.
    pub min_inliers_per_pair: usize,
    /// Re-estimate two-view geometry from filtered matches.
    pub refine_relative_poses: bool,
    /// Refine shared focal length with a 1D scale search on calibrated edges.
    pub refine_intrinsics: bool,
    /// Two-view RANSAC threshold passed to pose re-estimation (pixels).
    pub essential_threshold_px: f32,
    /// Two-view RANSAC iteration budget for pose re-estimation.
    pub essential_iterations: u32,
    /// Minimum triangulated inliers required when re-estimating a pair.
    pub min_triangulated_per_pair: usize,
}

impl Default for ViewGraphCalibrationOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            max_epipolar_error_px: 4.0,
            min_triangulation_angle_deg: 1.5,
            max_epipole_distance_px: 10.0,
            max_rotation_error_deg: 5.0,
            min_inliers_per_pair: 15,
            refine_relative_poses: true,
            refine_intrinsics: false,
            essential_threshold_px: 4.0,
            essential_iterations: 256,
            min_triangulated_per_pair: 10,
        }
    }
}

/// Statistics from [`calibrate_view_graph`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ViewGraphCalibrationStats {
    pub pairs_in: usize,
    pub pairs_out: usize,
    pub matches_in: usize,
    pub matches_out: usize,
    pub rotation_filtered_pairs: usize,
    pub intrinsics_refined: bool,
}

/// Calibrate the view graph: filter pairwise matches, optionally refine
/// intrinsics and relative poses, and remove rotation-inconsistent edges.
pub fn calibrate_view_graph(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    mut camera: CameraModel,
    options: &ViewGraphCalibrationOptions,
) -> (Vec<PairGeometry>, CameraModel, ViewGraphCalibrationStats) {
    let mut stats = ViewGraphCalibrationStats {
        pairs_in: pairs.len(),
        matches_in: pairs.iter().map(|p| p.inlier_matches.len()).sum(),
        ..ViewGraphCalibrationStats::default()
    };
    if !options.enabled {
        stats.pairs_out = pairs.len();
        stats.matches_out = stats.matches_in;
        return (pairs.to_vec(), camera, stats);
    }

    let mut calibrated: Vec<PairGeometry> = pairs
        .iter()
        .filter(|pair| pair_is_supported(pair) && !pair.pose_graph_only)
        .filter_map(|pair| {
            filter_pair_matches(
                pair,
                frames.get(pair.left)?,
                frames.get(pair.right)?,
                camera,
                options,
            )
        })
        .filter(|pair| pair.inlier_matches.len() >= options.min_inliers_per_pair)
        .collect();

    stats.matches_out = calibrated.iter().map(|p| p.inlier_matches.len()).sum();

    if options.refine_intrinsics && !calibrated.is_empty() {
        if let Some(refined) = refine_shared_focal_length(&calibrated, frames, camera, options) {
            camera = refined;
            stats.intrinsics_refined = true;
        }
    }

    if options.refine_relative_poses {
        calibrated = calibrated
            .into_iter()
            .filter_map(|pair| {
                let left = frames.get(pair.left)?;
                let right = frames.get(pair.right)?;
                estimate_pair_geometry_with_options(
                    pair.left,
                    pair.right,
                    left,
                    right,
                    &pair.inlier_matches,
                    camera,
                    options.essential_threshold_px,
                    options.essential_iterations,
                    options.min_inliers_per_pair,
                    options.min_triangulated_per_pair,
                    PairEstimationOptions {
                        ransac_random_seed: pair.left as i32 ^ pair.right as i32,
                        ..PairEstimationOptions::default()
                    },
                )
            })
            .collect();
        stats.matches_out = calibrated.iter().map(|p| p.inlier_matches.len()).sum();
    }

    let num_views = frames.len();
    if num_views >= 2 && !calibrated.is_empty() {
        let before = calibrated.len();
        calibrated = filter_rotation_inconsistent_pairs(
            num_views,
            &calibrated,
            options.max_rotation_error_deg,
        );
        stats.rotation_filtered_pairs = before.saturating_sub(calibrated.len());
    }

    stats.pairs_out = calibrated.len();
    (calibrated, camera, stats)
}

fn pair_is_supported(pair: &PairGeometry) -> bool {
    !matches!(
        pair.two_view_config,
        COLMAP_TWO_VIEW_UNDEFINED
            | COLMAP_TWO_VIEW_DEGENERATE
            | COLMAP_TWO_VIEW_WATERMARK
            | COLMAP_TWO_VIEW_MULTIPLE
    )
}

fn filter_pair_matches(
    pair: &PairGeometry,
    left: &ImageFrame,
    right: &ImageFrame,
    camera: CameraModel,
    options: &ViewGraphCalibrationOptions,
) -> Option<PairGeometry> {
    let left_pose = SE3::identity();
    let right_pose = pair.relative_pose;
    let (ep1, ep2) = epipoles_from_pair(pair, camera);

    let filtered = pair
        .inlier_matches
        .iter()
        .filter(|m| {
            geometry_consistent(pair, left, right, m, camera, options.max_epipolar_error_px)
                && cheiral_with_min_angle(
                    left_pose,
                    right_pose,
                    left,
                    right,
                    m,
                    camera,
                    options.min_triangulation_angle_deg,
                    pair.two_view_config,
                )
                && epipole_distance_ok(left, right, m, ep1.as_ref(), ep2.as_ref(), options)
        })
        .cloned()
        .collect::<Vec<_>>();

    if filtered.len() < options.min_inliers_per_pair {
        return None;
    }

    let mut updated = pair.clone();
    updated.inlier_matches = filtered;
    updated.inliers = updated.inlier_matches.len();
    Some(updated)
}

fn geometry_consistent(
    pair: &PairGeometry,
    left: &ImageFrame,
    right: &ImageFrame,
    m: &Match,
    camera: CameraModel,
    max_error_px: f32,
) -> bool {
    let li = m.query_idx as usize;
    let ri = m.train_idx as usize;
    let Some(lk) = left.keypoints.get(li) else {
        return false;
    };
    let Some(rk) = right.keypoints.get(ri) else {
        return false;
    };
    let threshold_sq = (max_error_px as f64).max(1.0e-12).powi(2);

    match pair.two_view_config {
        COLMAP_TWO_VIEW_PLANAR
        | COLMAP_TWO_VIEW_PANORAMIC
        | COLMAP_TWO_VIEW_PLANAR_OR_PANORAMIC => {
            let Some(h) = pair.h_matrix.map(matrix3_from_row_array) else {
                return true;
            };
            homography_transfer_error_sq(lk.x(), lk.y(), rk.x(), rk.y(), &h) <= threshold_sq
        }
        COLMAP_TWO_VIEW_UNCALIBRATED => {
            let Some(f) = pair.f_matrix.map(matrix3_from_row_array) else {
                return true;
            };
            squared_sampson_error_pixels(lk.x(), lk.y(), rk.x(), rk.y(), &f) <= threshold_sq
        }
        _ => {
            if let Some(e) = pair.e_matrix.map(matrix3_from_row_array) {
                let Some(x1) = normalized_point(camera, lk.x(), lk.y()) else {
                    return false;
                };
                let Some(x2) = normalized_point(camera, rk.x(), rk.y()) else {
                    return false;
                };
                squared_sampson_error_normalized(&x1, &x2, &e) <= threshold_sq
            } else if let Some(f) = pair.f_matrix.map(matrix3_from_row_array) {
                squared_sampson_error_pixels(lk.x(), lk.y(), rk.x(), rk.y(), &f) <= threshold_sq
            } else {
                true
            }
        }
    }
}

fn cheiral_with_min_angle(
    left_pose: SE3,
    right_pose: SE3,
    left: &ImageFrame,
    right: &ImageFrame,
    m: &Match,
    camera: CameraModel,
    min_angle_deg: f32,
    two_view_config: i32,
) -> bool {
    if matches!(
        two_view_config,
        COLMAP_TWO_VIEW_PANORAMIC | COLMAP_TWO_VIEW_PLANAR_OR_PANORAMIC
    ) {
        return true;
    }
    let li = m.query_idx as usize;
    let ri = m.train_idx as usize;
    let Some(lk) = left.keypoints.get(li) else {
        return false;
    };
    let Some(rk) = right.keypoints.get(ri) else {
        return false;
    };
    let Some(ray1) = normalized_ray(camera, lk.x(), lk.y()) else {
        return false;
    };
    let Some(ray2) = normalized_ray(camera, rk.x(), rk.y()) else {
        return false;
    };

    let r1 = rotation_matrix(left_pose);
    let t1 = translation_vector(left_pose);
    let r2 = rotation_matrix(right_pose);
    let t2 = translation_vector(right_pose);
    let r21 = r2 * r1.transpose();
    let t21 = t2 - r21 * t1;

    let Some(point) = triangulate_mid_point(&r21, &t21, &ray1, &ray2) else {
        return false;
    };
    if min_angle_deg <= 0.0 {
        return true;
    }
    let c1 = Vector3::zeros();
    let c2 = r21.transpose() * (-t21);
    let angle = triangulation_angle_deg(&c1, &c2, &point);
    angle.is_finite() && angle >= min_angle_deg as f64
}

fn epipole_distance_ok(
    left: &ImageFrame,
    right: &ImageFrame,
    m: &Match,
    ep1: Option<&Vector3<f64>>,
    ep2: Option<&Vector3<f64>>,
    options: &ViewGraphCalibrationOptions,
) -> bool {
    if options.max_epipole_distance_px <= 0.0 {
        return true;
    }
    let li = m.query_idx as usize;
    let ri = m.train_idx as usize;
    let Some(lk) = left.keypoints.get(li) else {
        return false;
    };
    let Some(rk) = right.keypoints.get(ri) else {
        return false;
    };
    let max_dist = options.max_epipole_distance_px as f64;
    if let Some(ep) = ep1 {
        if point_epipole_distance_px(lk.x(), lk.y(), ep) < max_dist {
            return false;
        }
    }
    if let Some(ep) = ep2 {
        if point_epipole_distance_px(rk.x(), rk.y(), ep) < max_dist {
            return false;
        }
    }
    true
}

fn epipoles_from_pair(
    pair: &PairGeometry,
    camera: CameraModel,
) -> (Option<Vector3<f64>>, Option<Vector3<f64>>) {
    let Some(matrix) = pair
        .e_matrix
        .map(matrix3_from_row_array)
        .or_else(|| pair.f_matrix.map(matrix3_from_row_array))
    else {
        return (None, None);
    };
    if matches!(
        pair.two_view_config,
        COLMAP_TWO_VIEW_PLANAR
            | COLMAP_TWO_VIEW_PANORAMIC
            | COLMAP_TWO_VIEW_PLANAR_OR_PANORAMIC
    ) {
        return (None, None);
    }
    let (ep1_norm, ep2_norm) = epipoles_from_matrix(&matrix);
    (
        ep1_norm.map(|ep| epipole_to_pixels(ep, camera)),
        ep2_norm.map(|ep| epipole_to_pixels(ep, camera)),
    )
}

fn epipoles_from_matrix(m: &Matrix3<f64>) -> (Option<Vector3<f64>>, Option<Vector3<f64>>) {
    let ep1 = null_vector(m);
    let ep2 = null_vector(&m.transpose());
    (ep1, ep2)
}

fn null_vector(m: &Matrix3<f64>) -> Option<Vector3<f64>> {
    let svd = m.svd(true, true);
    let v_t = svd.v_t?;
    let row = v_t.row(2);
    let v = Vector3::new(row[0], row[1], row[2]);
    if v.norm_squared() < 1.0e-12 {
        None
    } else {
        Some(v.normalize())
    }
}

fn epipole_to_pixels(ep: Vector3<f64>, camera: CameraModel) -> Vector3<f64> {
    if ep.z.abs() < 1.0e-12 {
        return ep;
    }
    let u = ep.x / ep.z;
    let v = ep.y / ep.z;
    if let Some(px) = camera.img_from_cam(u, v, 1.0) {
        Vector3::new(px[0], px[1], 1.0)
    } else {
        ep
    }
}

fn point_epipole_distance_px(x: f32, y: f32, ep: &Vector3<f64>) -> f64 {
    if ep.z.abs() < 1.0e-12 {
        return f64::INFINITY;
    }
    let ex = ep.x / ep.z;
    let ey = ep.y / ep.z;
    ((x as f64 - ex).powi(2) + (y as f64 - ey).powi(2)).sqrt()
}

fn refine_shared_focal_length(
    pairs: &[PairGeometry],
    frames: &[ImageFrame],
    camera: CameraModel,
    options: &ViewGraphCalibrationOptions,
) -> Option<CameraModel> {
    let mut best_scale = 1.0f64;
    let mut best_cost = f64::INFINITY;
    for step in 0..21 {
        let scale = 0.9 + 0.01 * step as f64;
        let trial = scaled_camera(camera, scale);
        let cost = total_epipolar_cost(pairs, frames, trial, options.max_epipolar_error_px);
        if cost < best_cost {
            best_cost = cost;
            best_scale = scale;
        }
    }
    (best_scale != 1.0).then(|| scaled_camera(camera, best_scale))
}

fn scaled_camera(mut camera: CameraModel, scale: f64) -> CameraModel {
    camera.fx = (camera.fx as f64 * scale) as f32;
    camera.fy = (camera.fy as f64 * scale) as f32;
    camera
}

fn total_epipolar_cost(
    pairs: &[PairGeometry],
    frames: &[ImageFrame],
    camera: CameraModel,
    max_error_px: f32,
) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for pair in pairs {
        if !matches!(
            pair.two_view_config,
            COLMAP_TWO_VIEW_CALIBRATED | COLMAP_TWO_VIEW_CALIBRATED_RIG
        ) {
            continue;
        }
        let Some(left) = frames.get(pair.left) else {
            continue;
        };
        let Some(right) = frames.get(pair.right) else {
            continue;
        };
        for m in &pair.inlier_matches {
            if geometry_consistent(pair, left, right, m, camera, max_error_px) {
                count += 1;
            } else {
                total += 1.0;
            }
        }
    }
    if count == 0 {
        f64::INFINITY
    } else {
        total / count as f64
    }
}

pub fn filter_rotation_inconsistent_pairs(
    num_views: usize,
    pairs: &[PairGeometry],
    max_error_deg: f64,
) -> Vec<PairGeometry> {
    if pairs.is_empty() || num_views < 2 {
        return pairs.to_vec();
    }
    let Some(rotation) = estimate_global_rotations(
        num_views,
        &relative_rotations_from_pairs(pairs),
        &RotationAveragingOptions::default(),
    ) else {
        return pairs.to_vec();
    };

    pairs
        .iter()
        .filter(|pair| {
            rotation_edge_error_deg(pair, &rotation.global_rotations) <= max_error_deg
        })
        .cloned()
        .collect()
}

pub fn rotation_edge_error_deg(pair: &PairGeometry, global_rotations: &[Quat]) -> f64 {
    if pair.left >= global_rotations.len() || pair.right >= global_rotations.len() {
        return f64::INFINITY;
    }
    let observed = pose_rotation(pair.relative_pose);
    let predicted =
        (global_rotations[pair.right] * global_rotations[pair.left].inverse()).normalize();
    quat_angle_deg(observed, predicted) as f64
}

fn quat_angle_deg(a: Quat, b: Quat) -> f32 {
    let delta = (a * b.inverse()).normalize();
    (2.0 * delta.w.abs().clamp(-1.0, 1.0).acos()).to_degrees()
}

fn matrix3_from_row_array(values: [f64; 9]) -> Matrix3<f64> {
    Matrix3::from_row_slice(&values)
}

fn normalized_point(camera: CameraModel, x: f32, y: f32) -> Option<Vector3<f64>> {
    let uv = camera.cam_from_img(x as f64, y as f64)?;
    Some(Vector3::new(uv[0], uv[1], 1.0))
}

fn normalized_ray(camera: CameraModel, x: f32, y: f32) -> Option<Vector3<f64>> {
    normalized_point(camera, x, y).map(|v| v.normalize())
}

fn rotation_matrix(pose: SE3) -> Matrix3<f64> {
    let r = pose.rotation_matrix();
    Matrix3::from_row_slice(&[
        r[0][0] as f64,
        r[0][1] as f64,
        r[0][2] as f64,
        r[1][0] as f64,
        r[1][1] as f64,
        r[1][2] as f64,
        r[2][0] as f64,
        r[2][1] as f64,
        r[2][2] as f64,
    ])
}

fn translation_vector(pose: SE3) -> Vector3<f64> {
    let t = pose.translation();
    Vector3::new(t[0] as f64, t[1] as f64, t[2] as f64)
}

fn squared_sampson_error_pixels(x1: f32, y1: f32, x2: f32, y2: f32, f: &Matrix3<f64>) -> f64 {
    let p1 = Vector3::new(x1 as f64, y1 as f64, 1.0);
    let p2 = Vector3::new(x2 as f64, y2 as f64, 1.0);
    squared_sampson_error_normalized(&p1, &p2, f)
}

fn squared_sampson_error_normalized(x1: &Vector3<f64>, x2: &Vector3<f64>, e: &Matrix3<f64>) -> f64 {
    let ex1 = e * x1;
    let etx2 = e.transpose() * x2;
    let num = x2.dot(&(e * x1));
    let den = ex1[0].powi(2) + ex1[1].powi(2) + etx2[0].powi(2) + etx2[1].powi(2);
    if den <= 1.0e-12 {
        f64::INFINITY
    } else {
        (num * num) / den
    }
}

fn homography_transfer_error_sq(x1: f32, y1: f32, x2: f32, y2: f32, h: &Matrix3<f64>) -> f64 {
    let p1 = Vector3::new(x1 as f64, y1 as f64, 1.0);
    let p2 = Vector3::new(x2 as f64, y2 as f64, 1.0);
    let hp = h * p1;
    if hp.z.abs() < 1.0e-12 {
        return f64::INFINITY;
    }
    let pred = Vector3::new(hp.x / hp.z, hp.y / hp.z, 1.0);
    let dx = pred.x - p2.x;
    let dy = pred.y - p2.y;
    dx * dx + dy * dy
}

fn triangulation_angle_deg(center1: &Vector3<f64>, center2: &Vector3<f64>, point: &Vector3<f64>) -> f64 {
    let v1 = point - center1;
    let v2 = point - center2;
    let denom = v1.norm() * v2.norm();
    if denom <= 1.0e-12 {
        return f64::INFINITY;
    }
    let angle = (v1.dot(&v2) / denom).clamp(-1.0, 1.0).acos();
    angle.min(std::f64::consts::PI - angle).to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::COLMAP_TWO_VIEW_CALIBRATED;
    use glam::Quat;

    fn test_camera() -> CameraModel {
        CameraModel::new_pinhole(640, 480, 500.0, 500.0, 320.0, 240.0)
    }

    fn test_frame(id: usize, x: f32, y: f32) -> ImageFrame {
        ImageFrame {
            id,
            name: format!("img_{id}.jpg"),
            path: std::path::PathBuf::from(format!("img_{id}.jpg")),
            width: 640,
            height: 480,
            keypoints: vec![rustslam::KeyPoint::new(x, y)],
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

    fn synth_calibrated_pair(
        left_idx: usize,
        right_idx: usize,
        left_kp: (f32, f32),
        right_kp: (f32, f32),
        relative_pose: SE3,
        e_matrix: [f64; 9],
    ) -> PairGeometry {
        PairGeometry {
            left: left_idx,
            right: right_idx,
            two_view_config: COLMAP_TWO_VIEW_CALIBRATED,
            f_matrix: None,
            e_matrix: Some(e_matrix),
            h_matrix: None,
            qvec: None,
            tvec: None,
            matches: Vec::new(),
            inlier_matches: vec![Match {
                query_idx: 0,
                train_idx: 0,
                distance: 0.0,
            }],
            relative_pose,
            inliers: 1,
            triangulated: 1,
            mean_reprojection_error_px: 0.5,
            rotation_deg: 0.0,
            median_triangulation_angle_deg: 5.0,
            pose_graph_only: false,
        }
    }

    #[test]
    fn rejects_matches_with_small_triangulation_angle() {
        let camera = test_camera();
        let options = ViewGraphCalibrationOptions {
            min_triangulation_angle_deg: 5.0,
            min_inliers_per_pair: 1,
            refine_relative_poses: false,
            ..ViewGraphCalibrationOptions::default()
        };
        // Nearly identical viewing directions -> tiny triangulation angle.
        let left_kp = (320.0, 240.0);
        let right_kp = (322.0, 240.0);
        let pair = synth_calibrated_pair(
            0,
            1,
            left_kp,
            right_kp,
            SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(0.01, 0.0, 0.0)),
            [0.0, -0.01, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        let frames = vec![
            test_frame(0, left_kp.0, left_kp.1),
            test_frame(1, right_kp.0, right_kp.1),
        ];
        let (_, _, stats) = calibrate_view_graph(&frames, &[pair], camera, &options);
        assert_eq!(stats.pairs_out, 0);
    }

    #[test]
    fn filter_rotation_inconsistent_pairs_removes_outlier_edge() {
        let pose01 = SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(1.0, 0.0, 0.0));
        let pose02 = SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(2.0, 0.0, 0.0));
        let bad_pose12 = SE3::from_quat_translation(
            Quat::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2),
            Vec3::new(1.0, 0.0, 0.0),
        );
        let mk = |left: usize, right: usize, pose: SE3| PairGeometry {
            left,
            right,
            two_view_config: COLMAP_TWO_VIEW_CALIBRATED,
            f_matrix: None,
            e_matrix: None,
            h_matrix: None,
            qvec: None,
            tvec: None,
            matches: Vec::new(),
            inlier_matches: vec![Match {
                query_idx: 0,
                train_idx: 0,
                distance: 0.0,
            }; 30],
            relative_pose: pose,
            inliers: 30,
            triangulated: 30,
            mean_reprojection_error_px: 0.5,
            rotation_deg: 0.0,
            median_triangulation_angle_deg: 5.0,
            pose_graph_only: false,
        };
        let pairs = vec![mk(0, 1, pose01), mk(0, 2, pose02), mk(1, 2, bad_pose12)];
        let filtered = filter_rotation_inconsistent_pairs(3, &pairs, 5.0);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|p| !(p.left == 1 && p.right == 2)));
    }
}
