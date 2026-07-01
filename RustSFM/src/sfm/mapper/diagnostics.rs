use super::pair_geometry_to_colmap_two_view_geometry;
use crate::colmap::{read_colmap_poses, world_to_camera_rotation};
use crate::types::{ImageFrame, PairGeometry};
use nalgebra::{Matrix3, Vector3};
use rustslam::SE3;
use std::collections::HashMap;
use std::path::Path;

pub(super) fn pair_quality_summary(pairs: &[PairGeometry]) -> Vec<String> {
    if pairs.is_empty() {
        return Vec::new();
    }
    let mean_reproj = pairs
        .iter()
        .map(|p| p.mean_reprojection_error_px)
        .sum::<f32>()
        / pairs.len() as f32;
    let mean_inliers = pairs.iter().map(|p| p.inliers).sum::<usize>() as f32 / pairs.len() as f32;
    let high_error = pairs
        .iter()
        .filter(|p| p.mean_reprojection_error_px > 4.0 || p.rotation_deg > 25.0)
        .count();
    let mean_triangulation_angle = pairs
        .iter()
        .map(|p| p.median_triangulation_angle_deg)
        .sum::<f32>()
        / pairs.len() as f32;
    vec![format!(
        "pair_quality mean_inliers={:.1} mean_reproj={:.3} mean_tri_angle={:.3}deg high_error_pairs={}/{}",
        mean_inliers,
        mean_reproj,
        mean_triangulation_angle,
        high_error,
        pairs.len()
    )]
}

pub(super) fn pair_config_summary(pairs: &[PairGeometry]) -> Vec<String> {
    if pairs.is_empty() {
        return Vec::new();
    }
    let mut counts = pairs
        .iter()
        .fold(HashMap::<i32, usize>::new(), |mut counts, pair| {
            *counts.entry(pair.two_view_config).or_default() += 1;
            counts
        });
    let mut configs = counts.keys().copied().collect::<Vec<_>>();
    configs.sort_unstable();
    let parts = configs
        .into_iter()
        .map(|config| {
            let count = counts.remove(&config).unwrap_or(0);
            format!("{}={}", colmap_two_view_config_name(config), count)
        })
        .collect::<Vec<_>>();
    vec![format!("pair_config {}", parts.join(" "))]
}

pub(super) fn pair_two_view_metadata_summary(pairs: &[PairGeometry]) -> Vec<String> {
    if pairs.is_empty() {
        return Vec::new();
    }
    let mut with_f = 0usize;
    let mut with_e = 0usize;
    let mut with_h = 0usize;
    let mut with_qvec = 0usize;
    let mut with_tvec = 0usize;
    let mut with_pose = 0usize;
    for pair in pairs {
        let geometry = pair_geometry_to_colmap_two_view_geometry(pair);
        with_f += usize::from(geometry.f_matrix.is_some());
        with_e += usize::from(geometry.e_matrix.is_some());
        with_h += usize::from(geometry.h_matrix.is_some());
        with_qvec += usize::from(geometry.qvec.is_some());
        with_tvec += usize::from(geometry.tvec.is_some());
        with_pose += usize::from(geometry.qvec.is_some() && geometry.tvec.is_some());
    }
    vec![format!(
        "pair_two_view_metadata F={} E={} H={} qvec={} tvec={} pose={} total={}",
        with_f,
        with_e,
        with_h,
        with_qvec,
        with_tvec,
        with_pose,
        pairs.len()
    )]
}

fn colmap_two_view_config_name(config: i32) -> &'static str {
    match config {
        crate::database::COLMAP_TWO_VIEW_UNDEFINED => "UNDEFINED",
        crate::database::COLMAP_TWO_VIEW_DEGENERATE => "DEGENERATE",
        crate::database::COLMAP_TWO_VIEW_CALIBRATED => "CALIBRATED",
        crate::database::COLMAP_TWO_VIEW_UNCALIBRATED => "UNCALIBRATED",
        crate::database::COLMAP_TWO_VIEW_PLANAR => "PLANAR",
        crate::database::COLMAP_TWO_VIEW_PANORAMIC => "PANORAMIC",
        crate::database::COLMAP_TWO_VIEW_PLANAR_OR_PANORAMIC => "PLANAR_OR_PANORAMIC",
        crate::database::COLMAP_TWO_VIEW_WATERMARK => "WATERMARK",
        crate::database::COLMAP_TWO_VIEW_MULTIPLE => "MULTIPLE",
        crate::database::COLMAP_TWO_VIEW_CALIBRATED_RIG => "CALIBRATED_RIG",
        _ => "UNKNOWN",
    }
}

pub(super) fn pair_connectivity_summary(
    pairs: &[PairGeometry],
    frames: &[ImageFrame],
) -> Vec<String> {
    let mut degree = vec![0usize; frames.len()];
    let mut first_edges = Vec::new();
    for pair in pairs {
        degree[pair.left] += 1;
        degree[pair.right] += 1;
        if pair.left < 6 || pair.right < 6 {
            first_edges.push(format!(
                "{}->{}(in={},tri={},err={:.2})",
                pair.left + 1,
                pair.right + 1,
                pair.inliers,
                pair.triangulated,
                pair.mean_reprojection_error_px
            ));
        }
    }
    let isolated = degree.iter().filter(|&&d| d == 0).count();
    let min_degree = degree.iter().copied().min().unwrap_or(0);
    let max_degree = degree.iter().copied().max().unwrap_or(0);
    let mean_degree = if degree.is_empty() {
        0.0
    } else {
        degree.iter().sum::<usize>() as f32 / degree.len() as f32
    };
    first_edges.truncate(18);
    vec![
        format!(
            "pair_connectivity isolated={} min_degree={} mean_degree={:.2} max_degree={}",
            isolated, min_degree, mean_degree, max_degree
        ),
        format!("pair_first_edges {}", first_edges.join(" ")),
    ]
}

pub(super) fn pair_reference_error_summary(
    pairs: &[PairGeometry],
    frames: &[ImageFrame],
    reference: &Path,
) -> Vec<String> {
    let Ok(poses) = read_colmap_poses(reference) else {
        return Vec::new();
    };
    let by_name = poses
        .iter()
        .map(|pose| (pose.name.as_str(), pose))
        .collect::<HashMap<_, _>>();
    let mut rot_errors = Vec::new();
    let mut trans_errors = Vec::new();
    let mut worst_pairs = Vec::<(f64, String)>::new();
    for pair in pairs {
        let Some(left_ref) = by_name.get(frames[pair.left].name.as_str()) else {
            continue;
        };
        let Some(right_ref) = by_name.get(frames[pair.right].name.as_str()) else {
            continue;
        };
        let ref_rel = reference_relative_pose(left_ref, right_ref);
        let cand_rel = rust_relative_pose_parts(pair.relative_pose);
        let rot_error = rotation_angle_deg_na(ref_rel.0.transpose() * cand_rel.0);
        let trans_error = if let (Some(a), Some(b)) = (
            ref_rel.1.try_normalize(1.0e-12),
            cand_rel.1.try_normalize(1.0e-12),
        ) {
            a.dot(&b).clamp(-1.0, 1.0).acos().to_degrees()
        } else {
            f64::INFINITY
        };
        if trans_error.is_finite() {
            trans_errors.push(trans_error);
        }
        rot_errors.push(rot_error);
        let score = rot_error + trans_error.min(180.0);
        let label = format!(
            "{}->{} rot={:.4}deg trans={:.4}deg inliers={} reproj={:.3} tri_angle={:.4}deg",
            frames[pair.left].name,
            frames[pair.right].name,
            rot_error,
            trans_error,
            pair.inliers,
            pair.mean_reprojection_error_px,
            pair.median_triangulation_angle_deg
        );
        worst_pairs.push((score, label));
    }
    if rot_errors.is_empty() {
        return Vec::new();
    }
    let rot_rmse = rmse(&rot_errors);
    let trans_rmse = rmse(&trans_errors);
    let mut out = vec![format!(
        "pair_reference_error pairs={} rot_mean={:.4}deg rot_rmse={:.4}deg trans_mean={:.4}deg trans_rmse={:.4}deg",
        rot_errors.len(),
        mean(&rot_errors),
        rot_rmse,
        mean(&trans_errors),
        trans_rmse
    )];
    worst_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    if let Some((_, label)) = worst_pairs.first() {
        out.push(format!("pair_reference_worst {label}"));
    }
    let top = worst_pairs
        .iter()
        .take(12)
        .map(|(_, label)| label.as_str())
        .collect::<Vec<_>>();
    if !top.is_empty() {
        out.push(format!("pair_reference_worst_top {}", top.join(" | ")));
    }
    out
}

fn reference_relative_pose(
    left: &crate::colmap::ColmapPose,
    right: &crate::colmap::ColmapPose,
) -> (Matrix3<f64>, Vector3<f64>) {
    let left_r = world_to_camera_rotation(left);
    let right_r = world_to_camera_rotation(right);
    let left_t = Vector3::new(left.tvec[0], left.tvec[1], left.tvec[2]);
    let right_t = Vector3::new(right.tvec[0], right.tvec[1], right.tvec[2]);
    let rotation = right_r * left_r.transpose();
    let translation = right_t - rotation * left_t;
    (rotation, translation)
}

fn rust_relative_pose_parts(pose: SE3) -> (Matrix3<f64>, Vector3<f64>) {
    let r = pose.rotation_matrix();
    let t = pose.translation();
    (
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
        ]),
        Vector3::new(t[0] as f64, t[1] as f64, t[2] as f64),
    )
}

fn rotation_angle_deg_na(delta: Matrix3<f64>) -> f64 {
    ((delta.trace() - 1.0) * 0.5)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn rmse(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt()
    }
}
