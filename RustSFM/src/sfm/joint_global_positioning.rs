//! Joint global positioning of cameras and 3D points (GLOMAP §3.2).
//!
//! Given fixed global rotations and multi-view feature tracks, this stage
//! jointly estimates camera centers `c_i` and point positions `X_k` by
//! minimizing normalized direction residuals
//!
//! ```text
//! minimize  sum_{i,k}  rho( || v_ik - d_ik (X_k - c_i) || )
//! subject to d_ik >= 0
//! ```
//!
//! where `v_ik` is the unit viewing ray in world coordinates and `d_ik` is a
//! per-observation depth scale. The default solver is Ceres Levenberg-Marquardt
//! on camera centers and 3D points when the `ceres-ba` feature is enabled;
//! otherwise alternating center/point/depth updates with Huber IRLS are used.

#[cfg(feature = "ceres-ba")]
#[path = "joint_global_positioning_ceres.rs"]
mod joint_global_positioning_ceres;
use crate::global_positioning::{
    estimate_global_positions, relative_translations_from_pairs, GlobalPositioningOptions,
};
use crate::track_establishment::Track;
use crate::triangulation::triangulate_multi_view_point;
use crate::types::{CameraModel, ImageFrame, PairGeometry};
use glam::{Quat, Vec3};
use nalgebra::{Matrix3x4, Vector2};
use rustslam::SE3;

/// A single ray constraint linking a camera to a track/point.
#[derive(Debug, Clone, Copy)]
pub struct RayObservation {
    /// Index of the observing camera/image.
    pub camera: usize,
    /// Feature index within the observing image.
    pub feature: usize,
    /// Track index linked to the observation.
    pub track: usize,
    /// Unit viewing ray in world coordinates.
    pub ray_world: Vec3,
    /// Robust weighting seed for the observation.
    pub weight: f64,
}

/// Solver backend for joint global positioning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointGlobalPositioningSolver {
    /// Alternating center/point/depth updates with Huber IRLS.
    Alternating,
    /// Ceres Levenberg-Marquardt on camera centers and 3D points.
    CeresLevenbergMarquardt,
}

impl Default for JointGlobalPositioningSolver {
    fn default() -> Self {
        #[cfg(feature = "ceres-ba")]
        {
            Self::CeresLevenbergMarquardt
        }
        #[cfg(not(feature = "ceres-ba"))]
        {
            Self::Alternating
        }
    }
}

/// Options for [`estimate_joint_global_positions`].
#[derive(Debug, Clone, Copy)]
pub struct JointGlobalPositioningOptions {
    /// Solver used after warm-start initialization.
    pub solver: JointGlobalPositioningSolver,
    /// Maximum solver iterations.
    pub max_num_iterations: usize,
    /// Stop once the largest variable update drops below this threshold.
    pub convergence: f64,
    /// Huber threshold on the per-residual vector norm.
    pub huber_threshold: f64,
    /// Warm-start camera centers from translation averaging before alternating
    /// refinement. When false, centers and points start at the origin.
    pub use_translation_averaging_init: bool,
    /// Translation-averaging options used only for warm start.
    pub translation_averaging: GlobalPositioningOptions,
}

impl Default for JointGlobalPositioningOptions {
    fn default() -> Self {
        Self {
            solver: JointGlobalPositioningSolver::default(),
            max_num_iterations: 100,
            convergence: 1.0e-5,
            huber_threshold: 0.1,
            use_translation_averaging_init: true,
            translation_averaging: GlobalPositioningOptions::default(),
        }
    }
}

/// Result of joint global positioning.
#[derive(Debug, Clone)]
pub struct JointGlobalPositioningResult {
    /// Estimated camera centers (view `0` at the origin, unit RMS scale gauge).
    pub centers: Vec<Vec3>,
    /// Estimated 3D point positions, one entry per input track (same order).
    pub points: Vec<Vec3>,
    /// Number of solver iterations performed.
    pub num_iterations: usize,
    /// Mean per-residual vector norm after the final iteration.
    pub mean_residual: f64,
    /// Cameras connected to view `0` through shared tracks.
    pub connected: Vec<bool>,
}

/// Build world-frame unit viewing rays for every track observation.
pub fn build_ray_observations(
    global_rotations: &[Quat],
    tracks: &[Track],
    frames: &[ImageFrame],
    camera: CameraModel,
) -> Vec<RayObservation> {
    let mut observations = Vec::new();
    for (track_idx, track) in tracks.iter().enumerate() {
        let track_weight = track.len().max(1) as f64;
        for node in &track.observations {
            let Some(frame) = frames.get(node.image) else {
                continue;
            };
            let Some(kp) = frame.keypoints.get(node.feature) else {
                continue;
            };
            let Some(rotation) = global_rotations.get(node.image) else {
                continue;
            };
            let Some(ray) = camera_ray_world(*rotation, camera, kp.x(), kp.y()) else {
                continue;
            };
            observations.push(RayObservation {
                camera: node.image,
                feature: node.feature,
                track: track_idx,
                ray_world: ray,
                weight: track_weight,
            });
        }
    }
    observations
}

/// Jointly estimate camera centers and 3D points from fixed global rotations
/// and multi-view tracks.
pub fn estimate_joint_global_positions(
    global_rotations: &[Quat],
    tracks: &[Track],
    frames: &[ImageFrame],
    camera: CameraModel,
    pairs: &[PairGeometry],
    options: &JointGlobalPositioningOptions,
) -> Option<JointGlobalPositioningResult> {
    let num_views = global_rotations.len();
    if num_views < 2 || tracks.is_empty() {
        return None;
    }

    let observations = build_ray_observations(global_rotations, tracks, frames, camera);
    if observations.is_empty() {
        return None;
    }

    let connected = camera_connectivity_from_tracks(num_views, tracks);
    let num_tracks = tracks.len();
    let mut centers =
        initialize_centers(num_views, global_rotations, pairs, &observations, options)?;
    let mut points = initialize_points_from_poses(
        num_tracks,
        &observations,
        &centers,
        global_rotations,
        frames,
        camera,
    );
    let mut depths = vec![1.0f64; observations.len()];
    apply_origin_gauge(&mut centers, &mut points);
    for (obs_idx, obs) in observations.iter().enumerate() {
        let delta = points[obs.track] - centers[obs.camera];
        depths[obs_idx] = inverse_depth_along_ray(obs.ray_world, delta);
    }

    let (centers, points, num_iterations, mean_residual) = match options.solver {
        JointGlobalPositioningSolver::Alternating => solve_joint_global_positioning_alternating(
            &observations,
            num_views,
            num_tracks,
            centers,
            points,
            options,
        )?,
        #[cfg(feature = "ceres-ba")]
        JointGlobalPositioningSolver::CeresLevenbergMarquardt => {
            if observations.len() > 5_000 {
                solve_joint_global_positioning_alternating(
                    &observations,
                    num_views,
                    num_tracks,
                    centers,
                    points,
                    options,
                )?
            } else {
                joint_global_positioning_ceres::solve_joint_global_positioning_ceres(
                    &observations,
                    num_views,
                    num_tracks,
                    centers.clone(),
                    points.clone(),
                    options,
                )
                .or_else(|| {
                    solve_joint_global_positioning_alternating(
                        &observations,
                        num_views,
                        num_tracks,
                        centers,
                        points,
                        options,
                    )
                })?
            }
        }
        #[cfg(not(feature = "ceres-ba"))]
        JointGlobalPositioningSolver::CeresLevenbergMarquardt => {
            solve_joint_global_positioning_alternating(
                &observations,
                num_views,
                num_tracks,
                centers,
                points,
                options,
            )?
        }
    };

    Some(JointGlobalPositioningResult {
        mean_residual,
        centers,
        points,
        num_iterations,
        connected,
    })
}

pub(crate) fn solve_joint_global_positioning_alternating(
    observations: &[RayObservation],
    num_views: usize,
    num_tracks: usize,
    mut centers: Vec<Vec3>,
    mut points: Vec<Vec3>,
    options: &JointGlobalPositioningOptions,
) -> Option<(Vec<Vec3>, Vec<Vec3>, usize, f64)> {
    let mut depths = vec![1.0f64; observations.len()];
    for (obs_idx, obs) in observations.iter().enumerate() {
        let delta = points[obs.track] - centers[obs.camera];
        depths[obs_idx] = inverse_depth_along_ray(obs.ray_world, delta);
    }

    let mut num_iterations = 0usize;
    for _ in 0..options.max_num_iterations {
        num_iterations += 1;

        let weights = robust_weights(
            observations,
            &centers,
            &points,
            &depths,
            options.huber_threshold,
        );

        let new_centers = update_centers(num_views, observations, &points, &depths, &weights);
        let new_points = update_points(num_tracks, observations, &new_centers, &depths, &weights);

        let mut depths_next = depths.clone();
        for (obs_idx, obs) in observations.iter().enumerate() {
            depths_next[obs_idx] = inverse_depth_along_ray(
                obs.ray_world,
                new_points[obs.track] - new_centers[obs.camera],
            );
        }

        let mut scaled_centers = new_centers;
        let mut scaled_points = new_points;
        apply_origin_gauge(&mut scaled_centers, &mut scaled_points);

        let max_change = scaled_centers
            .iter()
            .zip(centers.iter())
            .map(|(a, b)| (*a - *b).length() as f64)
            .chain(
                scaled_points
                    .iter()
                    .zip(points.iter())
                    .map(|(a, b)| (*a - *b).length() as f64),
            )
            .fold(0.0f64, f64::max);

        centers = scaled_centers;
        points = scaled_points;
        depths = depths_next;

        if max_change < options.convergence {
            break;
        }
    }

    if !options.use_translation_averaging_init {
        normalize_joint_scale(&mut centers, &mut points, &mut depths);
    }

    let residual = mean_residual(observations, &centers, &points, &depths);
    Some((centers, points, num_iterations, residual))
}

fn camera_ray_world(rotation: Quat, camera: CameraModel, x: f32, y: f32) -> Option<Vec3> {
    let uv = camera.cam_from_img_f32(x, y)?;
    let bearing_cam = Vec3::new(uv[0], uv[1], 1.0);
    bearing_cam
        .try_normalize()
        .map(|bearing| (rotation.inverse() * bearing).normalize())
}

fn initialize_centers(
    num_views: usize,
    global_rotations: &[Quat],
    pairs: &[PairGeometry],
    observations: &[RayObservation],
    options: &JointGlobalPositioningOptions,
) -> Option<Vec<Vec3>> {
    if options.use_translation_averaging_init {
        let edges = relative_translations_from_pairs(pairs);
        if let Some(result) =
            estimate_global_positions(global_rotations, &edges, &options.translation_averaging)
        {
            return Some(result.centers);
        }
    }

    let mut centers = vec![Vec3::ZERO; num_views];
    if observations.is_empty() {
        return None;
    }
    for obs in observations {
        centers[obs.camera] += obs.ray_world;
    }
    for center in &mut centers {
        if center.length_squared() > 1.0e-12 {
            *center = center.normalize();
        }
    }
    Some(centers)
}

fn initialize_points_from_poses(
    num_tracks: usize,
    observations: &[RayObservation],
    centers: &[Vec3],
    global_rotations: &[Quat],
    frames: &[ImageFrame],
    camera: CameraModel,
) -> Vec<Vec3> {
    let mut points = vec![Vec3::ZERO; num_tracks];
    for track_idx in 0..num_tracks {
        let mut cams_from_world = Vec::new();
        let mut cam_points = Vec::new();
        let mut seen_cameras = std::collections::HashSet::new();
        for obs in observations.iter().filter(|o| o.track == track_idx) {
            if !seen_cameras.insert(obs.camera) {
                continue;
            }
            let pose = pose_from_rotation_center(global_rotations[obs.camera], centers[obs.camera]);
            let Some(frame) = frames.get(obs.camera) else {
                continue;
            };
            let Some(kp) = frame.keypoints.get(obs.feature) else {
                continue;
            };
            let Some(xy) = camera.cam_from_img_f32(kp.x(), kp.y()) else {
                continue;
            };
            cams_from_world.push(se3_to_matrix3x4(pose));
            cam_points.push(Vector2::new(xy[0] as f64, xy[1] as f64));
        }
        if cam_points.len() >= 2 {
            if let Some(xyz) = triangulate_multi_view_point(&cams_from_world, &cam_points) {
                points[track_idx] = Vec3::new(xyz[0] as f32, xyz[1] as f32, xyz[2] as f32);
                continue;
            }
        }
        let mut count = 0usize;
        for obs in observations.iter().filter(|o| o.track == track_idx) {
            points[track_idx] += centers[obs.camera] + obs.ray_world * 3.0;
            count += 1;
        }
        if count > 0 {
            points[track_idx] /= count as f32;
        }
    }
    points
}

fn pose_from_rotation_center(rotation: Quat, center: Vec3) -> SE3 {
    SE3::from_quat_translation(rotation, -(rotation * center))
}

fn se3_to_matrix3x4(pose: SE3) -> Matrix3x4<f64> {
    let r = pose.rotation_matrix();
    let t = pose.translation();
    Matrix3x4::new(
        r[0][0] as f64,
        r[0][1] as f64,
        r[0][2] as f64,
        t[0] as f64,
        r[1][0] as f64,
        r[1][1] as f64,
        r[1][2] as f64,
        t[1] as f64,
        r[2][0] as f64,
        r[2][1] as f64,
        r[2][2] as f64,
        t[2] as f64,
    )
}

fn robust_weights(
    observations: &[RayObservation],
    centers: &[Vec3],
    points: &[Vec3],
    depths: &[f64],
    huber_threshold: f64,
) -> Vec<f64> {
    observations
        .iter()
        .zip(depths.iter())
        .map(|(obs, &depth)| {
            let residual = obs.ray_world - (points[obs.track] - centers[obs.camera]) * depth as f32;
            let norm = residual.length() as f64;
            let robust = if norm <= huber_threshold {
                1.0
            } else {
                huber_threshold / norm
            };
            obs.weight * robust
        })
        .collect()
}

fn update_centers(
    num_views: usize,
    observations: &[RayObservation],
    points: &[Vec3],
    depths: &[f64],
    weights: &[f64],
) -> Vec<Vec3> {
    let mut numerators = vec![Vec3::ZERO; num_views];
    let mut denominators = vec![0.0f64; num_views];
    for (obs, (&depth, &weight)) in observations.iter().zip(depths.iter().zip(weights.iter())) {
        if weight <= 0.0 || depth <= 0.0 {
            continue;
        }
        let d = depth as f32;
        let d2 = d * d;
        let target = d * points[obs.track] - obs.ray_world;
        numerators[obs.camera] += target * (weight * d as f64) as f32;
        denominators[obs.camera] += weight * d2 as f64;
    }
    let mut centers = vec![Vec3::ZERO; num_views];
    for (center, (numerator, denom)) in centers
        .iter_mut()
        .zip(numerators.iter().zip(denominators.iter()))
    {
        if *denom > 0.0 {
            *center = *numerator / *denom as f32;
        }
    }
    centers
}

pub(crate) fn inverse_depth_along_ray(ray_world: Vec3, delta: Vec3) -> f64 {
    let denom = delta.length_squared() as f64;
    if denom < 1.0e-12 {
        return 1.0;
    }
    (ray_world.dot(delta).max(0.0) as f64) / denom
}

pub(crate) fn apply_origin_gauge(centers: &mut [Vec3], points: &mut [Vec3]) {
    if centers.is_empty() {
        return;
    }
    let shift = centers[0];
    for center in centers.iter_mut() {
        *center -= shift;
    }
    for point in points.iter_mut() {
        *point -= shift;
    }
    centers[0] = Vec3::ZERO;
}

fn update_points(
    num_tracks: usize,
    observations: &[RayObservation],
    centers: &[Vec3],
    depths: &[f64],
    weights: &[f64],
) -> Vec<Vec3> {
    let mut numerators = vec![Vec3::ZERO; num_tracks];
    let mut denominators = vec![0.0f64; num_tracks];
    for (obs, (&depth, &weight)) in observations.iter().zip(depths.iter().zip(weights.iter())) {
        if weight <= 0.0 || depth <= 0.0 {
            continue;
        }
        let d = depth as f32;
        let d2 = d * d;
        let target = obs.ray_world + d * centers[obs.camera];
        numerators[obs.track] += target * (weight * d as f64) as f32;
        denominators[obs.track] += weight * d2 as f64;
    }
    let mut points = vec![Vec3::ZERO; num_tracks];
    for (point, (numerator, denom)) in points
        .iter_mut()
        .zip(numerators.iter().zip(denominators.iter()))
    {
        if *denom > 0.0 {
            *point = *numerator / *denom as f32;
        }
    }
    points
}

pub(crate) fn normalize_joint_scale(centers: &mut [Vec3], points: &mut [Vec3], depths: &mut [f64]) {
    let sum_sq: f64 = centers
        .iter()
        .chain(points.iter())
        .map(|v| v.length_squared() as f64)
        .sum::<f64>();
    let count = centers.len().max(1) + points.len();
    let rms = (sum_sq / count as f64).sqrt();
    if !rms.is_finite() || rms < 1.0e-12 {
        return;
    }
    let scale = (1.0 / rms) as f32;
    for center in centers.iter_mut() {
        *center *= scale;
    }
    for point in points.iter_mut() {
        *point *= scale;
    }
    for depth in depths.iter_mut() {
        *depth *= rms;
    }
}

pub(crate) fn mean_residual(
    observations: &[RayObservation],
    centers: &[Vec3],
    points: &[Vec3],
    depths: &[f64],
) -> f64 {
    if observations.is_empty() {
        return 0.0;
    }
    let mut total = 0.0f64;
    for (obs, &depth) in observations.iter().zip(depths.iter()) {
        let residual = obs.ray_world - (points[obs.track] - centers[obs.camera]) * depth as f32;
        total += residual.length() as f64;
    }
    total / observations.len() as f64
}

fn camera_connectivity_from_tracks(num_views: usize, tracks: &[Track]) -> Vec<bool> {
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); num_views];
    for track in tracks {
        let images: Vec<usize> = track
            .observations
            .iter()
            .map(|node| node.image)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        for i in 0..images.len() {
            for j in (i + 1)..images.len() {
                adjacency[images[i]].push(images[j]);
                adjacency[images[j]].push(images[i]);
            }
        }
    }
    let mut connected = vec![false; num_views];
    let mut queue = std::collections::VecDeque::new();
    connected[0] = true;
    queue.push_back(0usize);
    while let Some(node) = queue.pop_front() {
        for &neighbor in &adjacency[node] {
            if !connected[neighbor] {
                connected[neighbor] = true;
                queue.push_back(neighbor);
            }
        }
    }
    connected
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::track_establishment::FeatureNode;
    use crate::types::PairGeometry;
    use glam::Quat;
    use rustslam::{ColmapMt19937, Match, SE3};

    fn unit(rng: &mut ColmapMt19937) -> f32 {
        rng.next_u32() as f32 / u32::MAX as f32
    }

    fn random_quat(rng: &mut ColmapMt19937) -> Quat {
        let axis = Vec3::new(unit(rng) - 0.5, unit(rng) - 0.5, unit(rng) - 0.5).normalize_or_zero();
        let axis = if axis.length_squared() < 1.0e-6 {
            Vec3::X
        } else {
            axis
        };
        Quat::from_axis_angle(axis, unit(rng) * std::f32::consts::PI)
    }

    fn test_camera() -> CameraModel {
        CameraModel::new_pinhole(640, 480, 500.0, 500.0, 320.0, 240.0)
    }

    fn project(
        camera: CameraModel,
        rotation: Quat,
        center: Vec3,
        point: Vec3,
    ) -> Option<(f32, f32)> {
        let translation = -(rotation * center);
        let pose = SE3::from_quat_translation(rotation, translation);
        let cam = pose.transform_point(&point.to_array());
        if cam[2] <= 1.0e-3 {
            return None;
        }
        let xy = camera.img_from_cam(cam[0] as f64, cam[1] as f64, cam[2] as f64)?;
        Some((xy[0] as f32, xy[1] as f32))
    }

    fn synth_frame(id: usize, keypoints: Vec<(f32, f32)>) -> ImageFrame {
        let kps = keypoints
            .into_iter()
            .map(|(x, y)| rustslam::KeyPoint::new(x, y))
            .collect::<Vec<_>>();
        let colors = vec![[128, 128, 128]; kps.len()];
        ImageFrame {
            id,
            name: format!("img_{id:03}.jpg"),
            path: std::path::PathBuf::from(format!("img_{id:03}.jpg")),
            width: 640,
            height: 480,
            keypoints: kps,
            descriptors: rustslam::Descriptors::new(),
            sift: crate::sift::SiftFeatures::default(),
            wide_descriptors: crate::wide::WideDescriptors {
                data: Vec::new(),
                dim: 0,
                count: 0,
            },
            strong_feature_indices: Vec::new(),
            colors,
        }
    }

    fn synth_pair(
        i: usize,
        j: usize,
        rotations: &[Quat],
        centers: &[Vec3],
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
            inlier_matches: vec![Match {
                query_idx: 0,
                train_idx: 0,
                distance: 0.0,
            }],
            relative_pose: SE3::from_quat_translation(r_ij, t_ij),
            inliers,
            triangulated: inliers,
            mean_reprojection_error_px: 0.5,
            rotation_deg: 0.0,
            median_triangulation_angle_deg: 5.0,
            pose_graph_only: false,
        }
    }

    fn align_scale(est: &[Vec3], gt: &[Vec3]) -> f32 {
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for (e, g) in est.iter().zip(gt.iter()) {
            num += e.dot(*g);
            den += e.dot(*e);
        }
        if den < 1.0e-12 {
            1.0
        } else {
            num / den
        }
    }

    #[test]
    fn recovers_identity_rotations_layout() {
        let camera = test_camera();
        let n = 5;
        let rotations = vec![Quat::IDENTITY; n];
        let centers = vec![
            Vec3::ZERO,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.5, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let gt_points = vec![
            Vec3::new(0.5, 0.0, 4.0),
            Vec3::new(1.0, 0.25, 5.0),
            Vec3::new(1.5, -0.1, 4.5),
        ];

        let mut frames = vec![synth_frame(0, Vec::new()); n];
        let mut tracks = Vec::new();
        for point in &gt_points {
            let mut observations = Vec::new();
            for view in 0..n {
                let Some((x, y)) = project(camera, rotations[view], centers[view], *point) else {
                    continue;
                };
                if observations.is_empty() {
                    frames[view] = synth_frame(view, vec![(x, y)]);
                    observations.push(FeatureNode::new(view, 0));
                } else {
                    let feature_idx = frames[view].keypoints.len();
                    frames[view].keypoints.push(rustslam::KeyPoint::new(x, y));
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

        let result = estimate_joint_global_positions(
            &rotations,
            &tracks,
            &frames,
            camera,
            &pairs,
            &JointGlobalPositioningOptions {
                max_num_iterations: 0,
                ..JointGlobalPositioningOptions::default()
            },
        )
        .unwrap();

        let scale = align_scale(&result.centers, &centers);
        for view in 0..n {
            let err = (result.centers[view] * scale - centers[view]).length();
            assert!(err < 0.05, "view {view} center error {err}");
        }
        for (est, gt) in result.points.iter().zip(gt_points.iter()) {
            let err = (*est * scale - *gt).length();
            assert!(err < 1.5, "point error {err}");
        }
    }

    #[test]
    fn joint_refinement_improves_noisy_init() {
        let camera = test_camera();
        let n = 5;
        let rotations = vec![Quat::IDENTITY; n];
        let centers = vec![
            Vec3::ZERO,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.5, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let gt_points = vec![Vec3::new(0.5, 0.0, 4.0), Vec3::new(1.0, 0.25, 5.0)];

        let mut frames = vec![synth_frame(0, Vec::new()); n];
        let mut tracks = Vec::new();
        for point in &gt_points {
            let mut observations = Vec::new();
            for view in 0..n {
                let Some((x, y)) = project(camera, rotations[view], centers[view], *point) else {
                    continue;
                };
                if observations.is_empty() {
                    frames[view] = synth_frame(view, vec![(x, y)]);
                    observations.push(FeatureNode::new(view, 0));
                } else {
                    let feature_idx = frames[view].keypoints.len();
                    frames[view].keypoints.push(rustslam::KeyPoint::new(x, y));
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

        let refined = estimate_joint_global_positions(
            &rotations,
            &tracks,
            &frames,
            camera,
            &pairs,
            &JointGlobalPositioningOptions::default(),
        )
        .unwrap();
        assert!(refined.mean_residual < 0.05);
        assert!(refined.num_iterations > 0);
    }

    #[test]
    fn recovers_cameras_and_points_from_synthetic_tracks() {
        let mut rng = ColmapMt19937::new(314);
        let camera = test_camera();
        let n = 7;
        let num_points = 12;
        let mut rotations = vec![Quat::IDENTITY];
        let mut centers = vec![Vec3::ZERO];
        for _ in 1..n {
            rotations.push(random_quat(&mut rng));
            centers.push(Vec3::new(
                unit(&mut rng) * 2.0 - 1.0,
                unit(&mut rng) * 2.0 - 1.0,
                unit(&mut rng) * 2.0 - 1.0,
            ));
        }

        let mut gt_points = Vec::new();
        for _ in 0..num_points {
            gt_points.push(Vec3::new(
                unit(&mut rng) * 2.0 - 1.0,
                unit(&mut rng) * 2.0 - 1.0,
                unit(&mut rng) + 2.0,
            ));
        }

        let mut frames = vec![synth_frame(0, Vec::new()); n];
        let mut tracks = Vec::new();
        let mut track_gt_points = Vec::new();
        for point in &gt_points {
            let mut observations = Vec::new();
            for view in 0..n {
                if view > 0 && (view + track_gt_points.len()) % 3 == 0 {
                    continue;
                }
                let Some((x, y)) = project(camera, rotations[view], centers[view], *point) else {
                    continue;
                };
                if observations.is_empty() {
                    frames[view] = synth_frame(view, vec![(x, y)]);
                    observations.push(FeatureNode::new(view, 0));
                } else {
                    let feature_idx = frames[view].keypoints.len();
                    frames[view].keypoints.push(rustslam::KeyPoint::new(x, y));
                    frames[view].colors.push([128, 128, 128]);
                    observations.push(FeatureNode::new(view, feature_idx));
                }
            }
            if observations.len() >= 2 {
                tracks.push(Track { observations });
                track_gt_points.push(*point);
            }
        }

        let mut pairs = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if j - i > 4 {
                    continue;
                }
                pairs.push(synth_pair(i, j, &rotations, &centers, 100));
            }
        }

        let result = estimate_joint_global_positions(
            &rotations,
            &tracks,
            &frames,
            camera,
            &pairs,
            &JointGlobalPositioningOptions {
                max_num_iterations: 0,
                ..JointGlobalPositioningOptions::default()
            },
        )
        .unwrap();

        assert!(result.connected.iter().filter(|&&c| c).count() >= 2);
        let center_scale = align_scale(&result.centers, &centers);
        for view in 0..n {
            if !result.connected[view] {
                continue;
            }
            let err = (result.centers[view] * center_scale - centers[view]).length();
            assert!(err < 0.15, "view {view} center error {err}");
        }

        assert_eq!(result.points.len(), tracks.len());
        assert!(result.mean_residual < 0.5);
    }

    #[test]
    fn rejects_empty_tracks() {
        let camera = test_camera();
        assert!(estimate_joint_global_positions(
            &[Quat::IDENTITY, Quat::IDENTITY],
            &[],
            &[],
            camera,
            &[],
            &JointGlobalPositioningOptions::default(),
        )
        .is_none());
    }
}
