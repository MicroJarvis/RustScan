//! Faithful Rust port of COLMAP's `estimators/triangulation` module.
//!
//! Reproduces `src/colmap/estimators/triangulation.{h,cc}`: the
//! `TriangulationEstimator` (two-view / multi-view DLT model estimation with
//! cheirality + triangulation-angle gating, and angular / reprojection
//! residuals), `EstimateTriangulationOptions`, and the robust
//! `estimate_triangulation` entry point.
//!
//! The geometric primitives come from the COLMAP-faithful [`crate::triangulation`]
//! module. The robust loop mirrors COLMAP's LORANSAC + `CombinationSampler` +
//! `InlierSupportMeasurer` structure: it enumerates the 2-view combinations,
//! scores each candidate by inlier support, applies the COLMAP dynamic
//! stopping criterion, and performs a final local-optimization refit over all
//! inliers via the multi-view DLT.

use crate::triangulation::{
    calculate_triangulation_angle, triangulate_multi_view_point, triangulate_point,
};
use crate::types::CameraModel;
use nalgebra::{Matrix3, Matrix3x4, Vector2, Vector3};
use rustslam::ColmapCombinationSampler;

/// COLMAP `TriangulationEstimator::ResidualType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualType {
    AngularError,
    ReprojectionError,
}

/// COLMAP `TriangulationEstimator::PointData`.
#[derive(Debug, Clone, Copy)]
pub struct PointData {
    /// Image observation in pixels. Only needed for `ReprojectionError`.
    pub img_point: Vector2<f64>,
    /// Normalized camera coordinates. Must always be set.
    pub cam_point: Vector2<f64>,
}

/// COLMAP `TriangulationEstimator::PoseData`.
#[derive(Debug, Clone, Copy)]
pub struct PoseData {
    /// `cam_from_world` projection matrix for the observation's image.
    pub cam_from_world: Matrix3x4<f64>,
    /// Projection center (camera origin in world coordinates).
    pub proj_center: Vector3<f64>,
    /// Camera model for the observation's image.
    pub camera: CameraModel,
}

/// COLMAP `scene/projection.cc::HasPointPositiveDepth`.
pub fn has_point_positive_depth(cam_from_world: &Matrix3x4<f64>, point3d: &Vector3<f64>) -> bool {
    let row2 = cam_from_world.row(2);
    let depth = row2[0] * point3d[0] + row2[1] * point3d[1] + row2[2] * point3d[2] + row2[3];
    depth >= f64::EPSILON
}

/// COLMAP `scene/projection.cc::CalculateSquaredReprojectionError` (matrix form).
pub fn calculate_squared_reprojection_error(
    point2d: &Vector2<f64>,
    point3d: &Vector3<f64>,
    cam_from_world: &Matrix3x4<f64>,
    camera: &CameraModel,
) -> f64 {
    let point3d_in_cam = project_into_cam(cam_from_world, point3d);
    let Some(proj) = camera.img_from_cam(point3d_in_cam[0], point3d_in_cam[1], point3d_in_cam[2])
    else {
        return f64::MAX;
    };
    let dx = proj[0] - point2d[0];
    let dy = proj[1] - point2d[1];
    dx * dx + dy * dy
}

/// COLMAP `scene/projection.cc::CalculateAngularReprojectionError`
/// (pre-normalized bearing form).
pub fn calculate_angular_reprojection_error(
    cam_ray: &Vector3<f64>,
    point3d: &Vector3<f64>,
    cam_from_world: &Matrix3x4<f64>,
) -> f64 {
    let point3d_in_cam = project_into_cam(cam_from_world, point3d);
    let normalized = point3d_in_cam.normalize();
    let cos_angle = cam_ray.dot(&normalized);
    cos_angle.clamp(-1.0, 1.0).acos()
}

#[inline]
fn project_into_cam(cam_from_world: &Matrix3x4<f64>, point3d: &Vector3<f64>) -> Vector3<f64> {
    let r: Matrix3<f64> = cam_from_world.fixed_view::<3, 3>(0, 0).into_owned();
    let t: Vector3<f64> = cam_from_world.column(3).into_owned();
    r * point3d + t
}

/// Projection center of a `cam_from_world = [R | t]` matrix: `-R^T t`.
pub fn projection_center(cam_from_world: &Matrix3x4<f64>) -> Vector3<f64> {
    let r: Matrix3<f64> = cam_from_world.fixed_view::<3, 3>(0, 0).into_owned();
    let t: Vector3<f64> = cam_from_world.column(3).into_owned();
    -(r.transpose() * t)
}

/// COLMAP `TriangulationEstimator`.
#[derive(Debug, Clone, Copy)]
pub struct TriangulationEstimator {
    min_tri_angle: f64,
    residual_type: ResidualType,
}

impl TriangulationEstimator {
    /// Minimum number of samples needed to estimate a model.
    pub const MIN_NUM_SAMPLES: usize = 2;

    pub fn new(min_tri_angle: f64, residual_type: ResidualType) -> Self {
        debug_assert!(min_tri_angle >= 0.0);
        Self {
            min_tri_angle,
            residual_type,
        }
    }

    /// COLMAP `TriangulationEstimator::Estimate`. Returns at most one model.
    pub fn estimate(
        &self,
        point_data: &[PointData],
        pose_data: &[PoseData],
    ) -> Option<Vector3<f64>> {
        debug_assert!(point_data.len() >= 2);
        debug_assert_eq!(point_data.len(), pose_data.len());

        if point_data.len() == 2 {
            // Two-view triangulation.
            let xyz = triangulate_point(
                &pose_data[0].cam_from_world,
                &pose_data[1].cam_from_world,
                &point_data[0].cam_point,
                &point_data[1].cam_point,
            )?;
            if has_point_positive_depth(&pose_data[0].cam_from_world, &xyz)
                && has_point_positive_depth(&pose_data[1].cam_from_world, &xyz)
                && calculate_triangulation_angle(
                    &pose_data[0].proj_center,
                    &pose_data[1].proj_center,
                    &xyz,
                ) >= self.min_tri_angle
            {
                return Some(xyz);
            }
            None
        } else {
            // Multi-view triangulation.
            let cams_from_world: Vec<Matrix3x4<f64>> =
                pose_data.iter().map(|pose| pose.cam_from_world).collect();
            let cam_points: Vec<Vector2<f64>> =
                point_data.iter().map(|point| point.cam_point).collect();
            let xyz = triangulate_multi_view_point(&cams_from_world, &cam_points)?;

            // Cheirality constraint for every view.
            for pose in pose_data {
                if !has_point_positive_depth(&pose.cam_from_world, &xyz) {
                    return None;
                }
            }

            // Sufficient triangulation angle for at least one pair.
            for i in 0..pose_data.len() {
                for j in 0..i {
                    if calculate_triangulation_angle(
                        &pose_data[i].proj_center,
                        &pose_data[j].proj_center,
                        &xyz,
                    ) >= self.min_tri_angle
                    {
                        return Some(xyz);
                    }
                }
            }
            None
        }
    }

    /// COLMAP `TriangulationEstimator::Residuals` (squared residuals).
    pub fn residuals(
        &self,
        point_data: &[PointData],
        pose_data: &[PoseData],
        xyz: &Vector3<f64>,
    ) -> Vec<f64> {
        debug_assert_eq!(point_data.len(), pose_data.len());
        point_data
            .iter()
            .zip(pose_data.iter())
            .map(|(point, pose)| match self.residual_type {
                ResidualType::ReprojectionError => calculate_squared_reprojection_error(
                    &point.img_point,
                    xyz,
                    &pose.cam_from_world,
                    &pose.camera,
                ),
                ResidualType::AngularError => {
                    let cam_ray =
                        Vector3::new(point.cam_point[0], point.cam_point[1], 1.0).normalize();
                    let angular_error =
                        calculate_angular_reprojection_error(&cam_ray, xyz, &pose.cam_from_world);
                    angular_error * angular_error
                }
            })
            .collect()
    }
}

/// COLMAP `EstimateTriangulationOptions` (RANSAC knobs flattened in).
#[derive(Debug, Clone, Copy)]
pub struct EstimateTriangulationOptions {
    /// Minimum triangulation angle in radians.
    pub min_tri_angle: f64,
    pub residual_type: ResidualType,
    /// RANSAC inlier threshold (same units as the residual: radians for angular,
    /// pixels for reprojection). Compared against the squared residual as
    /// `max_error^2`.
    pub max_error: f64,
    pub confidence: f64,
    pub dyn_num_trials_multiplier: f64,
    pub min_inlier_ratio: f64,
    pub min_num_trials: usize,
    pub max_num_trials: usize,
    /// COLMAP `random_seed`; ignored for `CombinationSampler`, which is
    /// deterministic and non-randomized in COLMAP.
    pub random_seed: i32,
}

impl Default for EstimateTriangulationOptions {
    fn default() -> Self {
        // Matches COLMAP `EstimateTriangulationOptions` defaults.
        Self {
            min_tri_angle: 0.0,
            residual_type: ResidualType::AngularError,
            max_error: 2.0_f64.to_radians(),
            confidence: 0.9999,
            dyn_num_trials_multiplier: 3.0,
            min_inlier_ratio: 0.02,
            min_num_trials: 0,
            max_num_trials: 10000,
            random_seed: -1,
        }
    }
}

/// COLMAP `RANSAC::ComputeNumTrials` with `min_num_samples = 2`.
fn compute_num_trials(
    num_inliers: usize,
    num_samples: usize,
    confidence: f64,
    dyn_num_trials_multiplier: f64,
) -> usize {
    let prob_failure = 1.0 - confidence;
    if prob_failure <= 0.0 {
        return usize::MAX;
    }

    if num_inliers < TriangulationEstimator::MIN_NUM_SAMPLES
        || num_samples < TriangulationEstimator::MIN_NUM_SAMPLES
    {
        return usize::MAX;
    }

    let mut prob_inlier = 1.0;
    for idx in 0..TriangulationEstimator::MIN_NUM_SAMPLES {
        prob_inlier *= (num_inliers - idx) as f64 / (num_samples - idx) as f64;
    }

    let prob_outlier = 1.0 - prob_inlier;
    if prob_outlier <= 0.0 {
        return 1;
    }
    if prob_outlier == 1.0 {
        return usize::MAX;
    }

    let num_trials = (prob_failure.ln() / prob_outlier.ln() * dyn_num_trials_multiplier).ceil();
    if !num_trials.is_finite() {
        return usize::MAX;
    }
    num_trials.max(1.0) as usize
}

fn initial_max_num_trials(
    options: &EstimateTriangulationOptions,
    max_sampler_samples: usize,
) -> usize {
    let assumed_samples = 100_000usize;
    let assumed_inliers =
        (options.min_inlier_ratio.clamp(0.0, 1.0) * assumed_samples as f64) as usize;
    let dyn_max_num_trials = compute_num_trials(
        assumed_inliers,
        assumed_samples,
        options.confidence,
        options.dyn_num_trials_multiplier,
    );
    options
        .max_num_trials
        .min(dyn_max_num_trials)
        .min(max_sampler_samples)
}

/// Inlier support under COLMAP's `InlierSupportMeasurer`: more inliers is
/// better; ties broken by smaller summed inlier residual.
struct Support {
    num_inliers: usize,
    residual_sum: f64,
}

fn evaluate_support(residuals: &[f64], max_residual: f64) -> Support {
    let mut num_inliers = 0;
    let mut residual_sum = 0.0;
    for &r in residuals {
        if r <= max_residual {
            num_inliers += 1;
            residual_sum += r;
        }
    }
    Support {
        num_inliers,
        residual_sum,
    }
}

fn support_is_better(candidate: &Support, best: &Support) -> bool {
    candidate.num_inliers > best.num_inliers
        || (candidate.num_inliers == best.num_inliers && candidate.residual_sum < best.residual_sum)
}

pub(crate) fn triangulation_sample_pairs(
    num_samples: usize,
    random_seed: i32,
) -> Vec<(usize, usize)> {
    let _ = random_seed;
    let mut sampler = ColmapCombinationSampler::new(TriangulationEstimator::MIN_NUM_SAMPLES);
    if !sampler.initialize(num_samples) {
        return Vec::new();
    }
    let max_num_samples = sampler.max_num_samples() as usize;
    let mut pairs = Vec::with_capacity(max_num_samples);
    for _ in 0..max_num_samples {
        let sample = sampler.sample();
        if sample.len() == 2 {
            pairs.push((sample[0], sample[1]));
        }
    }
    pairs
}

/// COLMAP `EstimateTriangulation`: robustly estimate a 3D point from
/// observations in multiple views via LORANSAC plus a final inlier refit.
///
/// `points` are pixel observations, `cams_from_world` the `[R | t]` pose
/// matrices, and `cameras` the per-observation camera models. On success,
/// returns the inlier mask (one flag per observation) and the 3D point.
pub fn estimate_triangulation(
    options: &EstimateTriangulationOptions,
    points: &[Vector2<f64>],
    cams_from_world: &[Matrix3x4<f64>],
    cameras: &[CameraModel],
) -> Option<(Vec<bool>, Vector3<f64>)> {
    let num_samples = points.len();
    if num_samples < TriangulationEstimator::MIN_NUM_SAMPLES
        || cams_from_world.len() != num_samples
        || cameras.len() != num_samples
    {
        return None;
    }

    let mut point_data = Vec::with_capacity(num_samples);
    let mut pose_data = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let cam_point = cameras[i]
            .cam_from_img(points[i][0], points[i][1])
            .map(|c| Vector2::new(c[0], c[1]))
            .unwrap_or_else(Vector2::zeros);
        point_data.push(PointData {
            img_point: points[i],
            cam_point,
        });
        pose_data.push(PoseData {
            cam_from_world: cams_from_world[i],
            proj_center: projection_center(&cams_from_world[i]),
            camera: cameras[i],
        });
    }

    let estimator = TriangulationEstimator::new(options.min_tri_angle, options.residual_type);
    let max_residual = options.max_error * options.max_error;

    // Robust estimation over the 2-view combinations (CombinationSampler order),
    // scored by inlier support with COLMAP's dynamic stopping criterion.
    let mut best_model: Option<Vector3<f64>> = None;
    let mut best_support = Support {
        num_inliers: 0,
        residual_sum: f64::MAX,
    };
    let sample_pairs = triangulation_sample_pairs(num_samples, options.random_seed);
    let max_num_trials = initial_max_num_trials(options, sample_pairs.len());
    let mut dynamic_max_trials = max_num_trials;
    let mut abort = false;

    for (curr_thread_trial, (i, j)) in sample_pairs.into_iter().enumerate() {
        if curr_thread_trial >= max_num_trials || abort {
            break;
        }

        let sample_points = [point_data[i], point_data[j]];
        let sample_poses = [pose_data[i], pose_data[j]];
        let Some(xyz) = estimator.estimate(&sample_points, &sample_poses) else {
            continue;
        };

        let residuals = estimator.residuals(&point_data, &pose_data, &xyz);
        let support = evaluate_support(&residuals, max_residual);
        if support_is_better(&support, &best_support) {
            dynamic_max_trials = compute_num_trials(
                support.num_inliers,
                num_samples,
                options.confidence,
                options.dyn_num_trials_multiplier,
            );
            best_support = support;
            best_model = Some(xyz);
        }

        if curr_thread_trial >= dynamic_max_trials && curr_thread_trial >= options.min_num_trials {
            abort = true;
        }
    }

    let mut best_xyz = best_model?;

    // Local optimization: refit using all current inliers via the multi-view
    // DLT, and keep the refit if its support is at least as good.
    let inlier_indices =
        inlier_indices(&estimator, &point_data, &pose_data, &best_xyz, max_residual);
    if inlier_indices.len() > TriangulationEstimator::MIN_NUM_SAMPLES {
        let lo_points: Vec<PointData> = inlier_indices.iter().map(|&k| point_data[k]).collect();
        let lo_poses: Vec<PoseData> = inlier_indices.iter().map(|&k| pose_data[k]).collect();
        if let Some(refit) = estimator.estimate(&lo_points, &lo_poses) {
            let residuals = estimator.residuals(&point_data, &pose_data, &refit);
            let support = evaluate_support(&residuals, max_residual);
            if !support_is_better(&best_support, &support) {
                best_support = support;
                best_xyz = refit;
            }
        }
    }

    if best_support.num_inliers < TriangulationEstimator::MIN_NUM_SAMPLES {
        return None;
    }

    let residuals = estimator.residuals(&point_data, &pose_data, &best_xyz);
    let inlier_mask: Vec<bool> = residuals.iter().map(|&r| r <= max_residual).collect();
    Some((inlier_mask, best_xyz))
}

fn inlier_indices(
    estimator: &TriangulationEstimator,
    point_data: &[PointData],
    pose_data: &[PoseData],
    xyz: &Vector3<f64>,
    max_residual: f64,
) -> Vec<usize> {
    estimator
        .residuals(point_data, pose_data, xyz)
        .into_iter()
        .enumerate()
        .filter_map(|(idx, r)| (r <= max_residual).then_some(idx))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::COLMAP_PINHOLE;
    use nalgebra::{Rotation3, Unit};

    fn rot(axis: Vector3<f64>, angle: f64) -> Matrix3<f64> {
        Rotation3::from_axis_angle(&Unit::new_normalize(axis), angle).into_inner()
    }

    fn pose_from_center(rotation: &Matrix3<f64>, center: &Vector3<f64>) -> Matrix3x4<f64> {
        let t = -(rotation * center);
        let mut m = Matrix3x4::<f64>::zeros();
        m.fixed_view_mut::<3, 3>(0, 0).copy_from(rotation);
        m.column_mut(3).copy_from(&t);
        m
    }

    fn pinhole() -> CameraModel {
        CameraModel::from_colmap(COLMAP_PINHOLE, 640, 480, &[500.0, 500.0, 320.0, 240.0]).unwrap()
    }

    fn project_pixel(
        camera: &CameraModel,
        cam_from_world: &Matrix3x4<f64>,
        xyz: &Vector3<f64>,
    ) -> Vector2<f64> {
        let p = project_into_cam(cam_from_world, xyz);
        let uv = camera.img_from_cam(p[0], p[1], p[2]).unwrap();
        Vector2::new(uv[0], uv[1])
    }

    fn three_view_setup() -> (Vec<Matrix3x4<f64>>, Vec<CameraModel>, Vector3<f64>) {
        let cams = vec![
            pose_from_center(&Matrix3::identity(), &Vector3::new(0.0, 0.0, 0.0)),
            pose_from_center(
                &rot(Vector3::new(0.0, 1.0, 0.0), 0.3),
                &Vector3::new(1.0, 0.0, 0.0),
            ),
            pose_from_center(
                &rot(Vector3::new(1.0, 0.0, 0.0), -0.2),
                &Vector3::new(0.3, 0.8, -0.2),
            ),
        ];
        let cameras = vec![pinhole(), pinhole(), pinhole()];
        let truth = Vector3::new(-0.4, 0.5, 4.0);
        (cams, cameras, truth)
    }

    #[test]
    fn estimate_triangulation_recovers_point_with_all_inliers() {
        let (cams, cameras, truth) = three_view_setup();
        let points: Vec<Vector2<f64>> = cams
            .iter()
            .zip(cameras.iter())
            .map(|(cam, camera)| project_pixel(camera, cam, &truth))
            .collect();

        let options = EstimateTriangulationOptions {
            residual_type: ResidualType::ReprojectionError,
            max_error: 1.0,
            ..EstimateTriangulationOptions::default()
        };
        let (mask, xyz) = estimate_triangulation(&options, &points, &cams, &cameras)
            .expect("estimation succeeds");
        assert!(
            mask.iter().all(|&m| m),
            "all observations should be inliers"
        );
        assert!((xyz - truth).norm() < 1e-6, "{xyz:?} vs {truth:?}");
    }

    #[test]
    fn estimate_triangulation_rejects_outlier_observation() {
        let (cams, cameras, truth) = three_view_setup();
        // Add a fourth camera but feed it a grossly wrong observation.
        let mut cams = cams;
        let mut cameras = cameras;
        cams.push(pose_from_center(
            &rot(Vector3::new(0.0, 1.0, 0.0), -0.4),
            &Vector3::new(-1.0, 0.1, 0.0),
        ));
        cameras.push(pinhole());

        let mut points: Vec<Vector2<f64>> = cams
            .iter()
            .zip(cameras.iter())
            .map(|(cam, camera)| project_pixel(camera, cam, &truth))
            .collect();
        // Corrupt the last observation far outside the inlier threshold.
        points[3] = points[3] + Vector2::new(120.0, -90.0);

        let options = EstimateTriangulationOptions {
            residual_type: ResidualType::ReprojectionError,
            max_error: 1.0,
            ..EstimateTriangulationOptions::default()
        };
        let (mask, xyz) = estimate_triangulation(&options, &points, &cams, &cameras)
            .expect("estimation succeeds despite outlier");
        assert_eq!(mask.len(), 4);
        assert!(
            mask[0] && mask[1] && mask[2],
            "true observations are inliers"
        );
        assert!(!mask[3], "corrupted observation is an outlier");
        assert!((xyz - truth).norm() < 1e-4, "{xyz:?} vs {truth:?}");
    }

    #[test]
    fn estimate_triangulation_angular_residual_recovers_point() {
        let (cams, cameras, truth) = three_view_setup();
        let points: Vec<Vector2<f64>> = cams
            .iter()
            .zip(cameras.iter())
            .map(|(cam, camera)| project_pixel(camera, cam, &truth))
            .collect();

        let options = EstimateTriangulationOptions {
            residual_type: ResidualType::AngularError,
            ..EstimateTriangulationOptions::default()
        };
        let (mask, xyz) = estimate_triangulation(&options, &points, &cams, &cameras)
            .expect("estimation succeeds");
        assert!(mask.iter().all(|&m| m));
        assert!((xyz - truth).norm() < 1e-6, "{xyz:?} vs {truth:?}");
    }

    #[test]
    fn estimate_triangulation_enforces_min_triangulation_angle() {
        let (cams, cameras, truth) = three_view_setup();
        let points: Vec<Vector2<f64>> = cams
            .iter()
            .zip(cameras.iter())
            .map(|(cam, camera)| project_pixel(camera, cam, &truth))
            .collect();

        // An impossibly large minimum angle rejects every candidate model.
        let options = EstimateTriangulationOptions {
            min_tri_angle: 3.0, // radians (> 170 degrees)
            residual_type: ResidualType::ReprojectionError,
            max_error: 1.0,
            ..EstimateTriangulationOptions::default()
        };
        assert!(estimate_triangulation(&options, &points, &cams, &cameras).is_none());
    }

    #[test]
    fn has_point_positive_depth_matches_sign_of_camera_depth() {
        let cam = pose_from_center(&Matrix3::identity(), &Vector3::new(0.0, 0.0, 0.0));
        assert!(has_point_positive_depth(&cam, &Vector3::new(0.0, 0.0, 5.0)));
        assert!(!has_point_positive_depth(
            &cam,
            &Vector3::new(0.0, 0.0, -5.0)
        ));
    }

    #[test]
    fn triangulation_sample_pairs_match_colmap_combination_sampler() {
        let exhaustive = super::triangulation_sample_pairs(10, -1);
        let seeded_a = super::triangulation_sample_pairs(16, 42);
        let seeded_b = super::triangulation_sample_pairs(16, 123);
        assert_eq!(exhaustive.len(), 45);
        assert_eq!(seeded_a.len(), 120);
        assert_eq!(seeded_a, seeded_b);
        assert_eq!(
            &seeded_a[..6],
            &[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)]
        );
        assert_eq!(seeded_a.last().copied(), Some((14, 15)));
    }

    #[test]
    fn ransac_trial_count_matches_colmap_without_replacement_formula() {
        assert_eq!(super::compute_num_trials(1, 100, 0.99, 1.0), usize::MAX);
        assert_eq!(super::compute_num_trials(10, 100, 0.99, 1.0), 505);
        assert_eq!(super::compute_num_trials(10, 100, 0.9999, 3.0), 3026);
        assert_eq!(super::compute_num_trials(100, 100, 0.9999, 3.0), 1);

        let options = EstimateTriangulationOptions {
            max_num_trials: 10_000,
            min_inlier_ratio: 0.02,
            ..EstimateTriangulationOptions::default()
        };
        assert_eq!(super::initial_max_num_trials(&options, 45), 45);
        assert_eq!(super::initial_max_num_trials(&options, 50_000), 10_000);
    }
}
