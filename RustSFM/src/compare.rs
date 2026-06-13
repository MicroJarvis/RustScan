use crate::colmap::{camera_center, read_colmap_poses, world_to_camera_rotation, ColmapPose};
use anyhow::{bail, Context, Result};
use nalgebra::{
    Matrix3, Matrix4, Quaternion, Rotation3, SymmetricEigen, UnitQuaternion, Vector3, Vector4,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ErrorStats {
    pub mean: f64,
    pub rmse: f64,
    pub max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerImageError {
    pub image_name: String,
    pub translation_error: f64,
    pub rotation_error_deg: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerAdjacentError {
    pub left_image_name: String,
    pub right_image_name: String,
    pub relative_rotation_error_deg: f64,
    pub relative_translation_angle_deg: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareReport {
    pub common_images: usize,
    pub similarity_scale: f64,
    pub translation_error: ErrorStats,
    pub rotation_error_deg: ErrorStats,
    pub adjacent_relative_rotation_error_deg: ErrorStats,
    pub adjacent_relative_translation_angle_deg: ErrorStats,
    pub per_image: Vec<PerImageError>,
    pub per_adjacent: Vec<PerAdjacentError>,
}

pub fn compare_colmap(reference: &Path, candidate: &Path) -> Result<CompareReport> {
    let ref_poses = read_colmap_poses(reference)?;
    let cand_poses = read_colmap_poses(candidate)?;
    let ref_by_name = ref_poses
        .iter()
        .map(|p| (p.name.as_str(), p))
        .collect::<HashMap<_, _>>();
    let cand_by_name = cand_poses
        .iter()
        .map(|p| (p.name.as_str(), p))
        .collect::<HashMap<_, _>>();
    let mut names = ref_by_name
        .keys()
        .filter(|name| cand_by_name.contains_key(**name))
        .copied()
        .collect::<Vec<_>>();
    names.sort_unstable();
    if names.len() < 3 {
        bail!("need at least 3 common images for pose comparison");
    }
    let ref_centers = names
        .iter()
        .map(|name| camera_center(ref_by_name[name]))
        .collect::<Vec<_>>();
    let cand_centers = names
        .iter()
        .map(|name| camera_center(cand_by_name[name]))
        .collect::<Vec<_>>();
    let sim = estimate_similarity(&cand_centers, &ref_centers)?;
    let orientation_alignment =
        estimate_orientation_alignment(&names, &ref_by_name, &cand_by_name)?;

    let mut trans_errors = Vec::new();
    let mut rot_errors = Vec::new();
    let mut per_image = Vec::new();
    for name in &names {
        let r = ref_by_name[name];
        let c = cand_by_name[name];
        let aligned_c = sim.scale * sim.rotation * camera_center(c) + sim.translation;
        let translation_error = (aligned_c - camera_center(r)).norm();
        // Camera centers and orientations have independent gauges in a sparse reconstruction.
        // Use a center Sim3 for translation and a quaternion average for the orientation gauge.
        let aligned_rwc = world_to_camera_rotation(c) * orientation_alignment.transpose();
        let rotation_error_deg =
            rotation_angle_deg(world_to_camera_rotation(r).transpose() * aligned_rwc);
        trans_errors.push(translation_error);
        rot_errors.push(rotation_error_deg);
        per_image.push(PerImageError {
            image_name: name.to_string(),
            translation_error,
            rotation_error_deg,
        });
    }
    let mut rel_rot_errors = Vec::new();
    let mut rel_trans_angle_errors = Vec::new();
    let mut per_adjacent = Vec::new();
    for pair in names.windows(2) {
        let r1 = ref_by_name[pair[0]];
        let r2 = ref_by_name[pair[1]];
        let c1 = cand_by_name[pair[0]];
        let c2 = cand_by_name[pair[1]];
        let ref_rel = relative_pose_parts(r1, r2);
        let cand_rel = relative_pose_parts(c1, c2);
        let relative_rotation_error_deg =
            rotation_angle_deg(ref_rel.rotation.transpose() * cand_rel.rotation);
        rel_rot_errors.push(relative_rotation_error_deg);
        let mut relative_translation_angle_deg = None;
        if let (Some(rt), Some(ct)) = (
            ref_rel.translation.try_normalize(1.0e-12),
            cand_rel.translation.try_normalize(1.0e-12),
        ) {
            let angle = rt.dot(&ct).clamp(-1.0, 1.0).acos().to_degrees();
            rel_trans_angle_errors.push(angle);
            relative_translation_angle_deg = Some(angle);
        }
        per_adjacent.push(PerAdjacentError {
            left_image_name: pair[0].to_string(),
            right_image_name: pair[1].to_string(),
            relative_rotation_error_deg,
            relative_translation_angle_deg,
        });
    }
    Ok(CompareReport {
        common_images: per_image.len(),
        similarity_scale: sim.scale,
        translation_error: stats(&trans_errors),
        rotation_error_deg: stats(&rot_errors),
        adjacent_relative_rotation_error_deg: stats(&rel_rot_errors),
        adjacent_relative_translation_angle_deg: stats(&rel_trans_angle_errors),
        per_image,
        per_adjacent,
    })
}

fn estimate_orientation_alignment(
    names: &[&str],
    ref_by_name: &HashMap<&str, &ColmapPose>,
    cand_by_name: &HashMap<&str, &ColmapPose>,
) -> Result<Matrix3<f64>> {
    let mut accum = Matrix4::<f64>::zeros();
    for name in names {
        let ref_r = world_to_camera_rotation(ref_by_name[name]);
        let cand_r = world_to_camera_rotation(cand_by_name[name]);
        // Candidate and reference reconstructions may differ by a global world rotation G:
        // Rcw_ref ~= Rcw_cand * G^T, so each image votes for G ~= Rcw_ref^T * Rcw_cand.
        let gauge = ref_r.transpose() * cand_r;
        let quat = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(gauge))
            .into_inner();
        let mut v = Vector4::new(quat.w, quat.i, quat.j, quat.k);
        if v[0] < 0.0 {
            v = -v;
        }
        accum += v * v.transpose();
    }
    let eig = SymmetricEigen::new(accum);
    let mut best = 0usize;
    for idx in 1..4 {
        if eig.eigenvalues[idx] > eig.eigenvalues[best] {
            best = idx;
        }
    }
    let q = eig.eigenvectors.column(best);
    let quat = Quaternion::new(q[0], q[1], q[2], q[3]).normalize();
    Ok(UnitQuaternion::from_quaternion(quat)
        .to_rotation_matrix()
        .into_inner())
}

struct RelativePoseParts {
    rotation: Matrix3<f64>,
    translation: Vector3<f64>,
}

fn relative_pose_parts(left: &ColmapPose, right: &ColmapPose) -> RelativePoseParts {
    let left_r = world_to_camera_rotation(left);
    let right_r = world_to_camera_rotation(right);
    let left_t = Vector3::new(left.tvec[0], left.tvec[1], left.tvec[2]);
    let right_t = Vector3::new(right.tvec[0], right.tvec[1], right.tvec[2]);
    let rotation = right_r * left_r.transpose();
    let translation = right_t - rotation * left_t;
    RelativePoseParts {
        rotation,
        translation,
    }
}

struct Similarity {
    scale: f64,
    rotation: Matrix3<f64>,
    translation: Vector3<f64>,
}

fn estimate_similarity(
    candidate: &[Vector3<f64>],
    reference: &[Vector3<f64>],
) -> Result<Similarity> {
    let n = candidate.len();
    if n != reference.len() || n < 3 {
        bail!("invalid similarity inputs");
    }
    let n_f = n as f64;
    let cm = candidate.iter().fold(Vector3::zeros(), |a, p| a + p) / n_f;
    let rm = reference.iter().fold(Vector3::zeros(), |a, p| a + p) / n_f;
    let mut cov = Matrix3::zeros();
    let mut var = 0.0;
    for (c, r) in candidate.iter().zip(reference.iter()) {
        let dc = c - cm;
        let dr = r - rm;
        cov += dr * dc.transpose();
        var += dc.norm_squared();
    }
    cov /= n_f;
    var /= n_f;
    let svd = cov.svd(true, true);
    let u = svd.u.context("missing U")?;
    let vt = svd.v_t.context("missing Vt")?;
    let mut d = Matrix3::identity();
    if (u * vt).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    let rotation = u * d * vt;
    let scale = (svd.singular_values[0] * d[(0, 0)]
        + svd.singular_values[1]
        + svd.singular_values[2] * d[(2, 2)])
        / var.max(1.0e-12);
    let translation = rm - scale * rotation * cm;
    Ok(Similarity {
        scale,
        rotation,
        translation,
    })
}

fn rotation_angle_deg(delta: Matrix3<f64>) -> f64 {
    ((delta.trace() - 1.0) * 0.5)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

fn stats(values: &[f64]) -> ErrorStats {
    if values.is_empty() {
        return ErrorStats::default();
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let rmse = (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt();
    let max = values.iter().copied().fold(0.0, f64::max);
    ErrorStats { mean, rmse, max }
}
