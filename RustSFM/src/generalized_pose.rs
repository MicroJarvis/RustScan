use crate::geometry::camera_center;
use crate::types::CameraModel;
#[cfg(feature = "poselib")]
use crate::types::Rigid3;
use glam::Vec3;
#[cfg(feature = "poselib")]
use rustslam::{colmap_ransac_num_trials, ColmapRandomSampler};
use rustslam::{ColmapRansacOptions, SE3};
use std::collections::BTreeSet;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

static GENERALIZED_POSE_RANSAC_SEED: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RansacOptions {
    pub max_error: f64,
    pub min_inlier_ratio: f64,
    pub confidence: f64,
    pub dyn_num_trials_multiplier: f64,
    pub min_num_trials: usize,
    pub max_num_trials: usize,
    pub random_seed: i32,
    pub num_threads: isize,
}

impl Default for RansacOptions {
    fn default() -> Self {
        let options = ColmapRansacOptions::default();
        Self {
            max_error: options.max_error,
            min_inlier_ratio: options.min_inlier_ratio,
            confidence: options.confidence,
            dyn_num_trials_multiplier: options.dyn_num_trials_multiplier,
            min_num_trials: options.min_num_trials,
            max_num_trials: options.max_num_trials,
            random_seed: options.random_seed,
            num_threads: options.num_threads,
        }
    }
}

impl RansacOptions {
    pub fn check(&self) -> Result<(), GeneralizedPoseError> {
        if self.max_error <= 0.0 {
            return Err(GeneralizedPoseError::InvalidOptions(
                "ransac max_error must be positive",
            ));
        }
        if !(0.0..=1.0).contains(&self.min_inlier_ratio) {
            return Err(GeneralizedPoseError::InvalidOptions(
                "ransac min_inlier_ratio must be in [0, 1]",
            ));
        }
        if !(0.0..=1.0).contains(&self.confidence) {
            return Err(GeneralizedPoseError::InvalidOptions(
                "ransac confidence must be in [0, 1]",
            ));
        }
        if self.min_num_trials > self.max_num_trials {
            return Err(GeneralizedPoseError::InvalidOptions(
                "ransac min_num_trials must not exceed max_num_trials",
            ));
        }
        if self.random_seed < -1 {
            return Err(GeneralizedPoseError::InvalidOptions(
                "ransac random_seed must be >= -1",
            ));
        }
        if self.num_threads == 0 || self.num_threads < -1 {
            return Err(GeneralizedPoseError::InvalidOptions(
                "ransac num_threads must be -1 or positive",
            ));
        }
        Ok(())
    }

    #[cfg(feature = "poselib")]
    fn as_colmap_options(&self) -> ColmapRansacOptions {
        ColmapRansacOptions {
            max_error: self.max_error,
            min_inlier_ratio: self.min_inlier_ratio,
            confidence: self.confidence,
            dyn_num_trials_multiplier: self.dyn_num_trials_multiplier,
            min_num_trials: self.min_num_trials,
            max_num_trials: self.max_num_trials,
            random_seed: self.random_seed,
            num_threads: self.num_threads,
        }
    }
}

fn generalized_pose_sampler_seed(random_seed: i32) -> u64 {
    if random_seed >= 0 {
        random_seed as u64
    } else {
        next_generalized_pose_ransac_seed()
    }
}

fn next_generalized_pose_ransac_seed() -> u64 {
    GENERALIZED_POSE_RANSAC_SEED.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructureLessAbsolutePoseEstimationOptions {
    pub ransac_options: RansacOptions,
}

impl Default for StructureLessAbsolutePoseEstimationOptions {
    fn default() -> Self {
        let mut ransac_options = RansacOptions::default();
        ransac_options.max_error = 6.0;
        ransac_options.min_num_trials = 100;
        ransac_options.max_num_trials = 10000;
        ransac_options.confidence = 0.99999;
        Self { ransac_options }
    }
}

impl StructureLessAbsolutePoseEstimationOptions {
    pub fn check(&self) -> Result<(), GeneralizedPoseError> {
        self.ransac_options.check()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneralizedAbsolutePoseEstimationOptions {
    pub ransac_options: RansacOptions,
}

impl Default for GeneralizedAbsolutePoseEstimationOptions {
    fn default() -> Self {
        let mut ransac_options = RansacOptions::default();
        ransac_options.max_error = 12.0;
        ransac_options.min_num_trials = 100;
        ransac_options.max_num_trials = 10000;
        ransac_options.confidence = 0.99999;
        Self { ransac_options }
    }
}

impl GeneralizedAbsolutePoseEstimationOptions {
    pub fn check(&self) -> Result<(), GeneralizedPoseError> {
        self.ransac_options.check()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneralizedRelativePoseEstimationOptions {
    pub ransac_options: RansacOptions,
}

impl Default for GeneralizedRelativePoseEstimationOptions {
    fn default() -> Self {
        let mut ransac_options = RansacOptions::default();
        ransac_options.max_error = 4.0;
        ransac_options.min_num_trials = 30;
        Self { ransac_options }
    }
}

impl GeneralizedRelativePoseEstimationOptions {
    pub fn check(&self) -> Result<(), GeneralizedPoseError> {
        self.ransac_options.check()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GRNPObservation {
    pub cam_from_rig: SE3,
    pub ray_in_cam: [f64; 3],
}

#[derive(Debug, Clone, Copy)]
pub struct StructureLessAbsolutePoseProblem<'a> {
    pub query_points2d: &'a [[f64; 2]],
    pub world_points2d: &'a [[f64; 2]],
    pub world_camera_idxs: &'a [usize],
    pub world_cams_from_world: &'a [SE3],
    pub world_cameras: &'a [CameraModel],
    pub query_camera: CameraModel,
}

#[derive(Debug, Clone, Copy)]
pub struct GeneralizedAbsolutePoseProblem<'a> {
    pub points2d: &'a [[f64; 2]],
    pub points3d: &'a [[f64; 3]],
    pub camera_idxs: &'a [usize],
    pub cams_from_rig: &'a [SE3],
    pub cameras: &'a [CameraModel],
}

#[derive(Debug, Clone, Copy)]
pub struct GeneralizedRelativePoseProblem<'a> {
    pub points2d1: &'a [[f64; 2]],
    pub points2d2: &'a [[f64; 2]],
    pub camera_idxs1: &'a [usize],
    pub camera_idxs2: &'a [usize],
    pub cams_from_rig: &'a [SE3],
    pub cameras: &'a [CameraModel],
}

#[derive(Debug, Clone)]
pub struct StructureLessAbsolutePoseObservations {
    pub world_obs: Vec<GRNPObservation>,
    pub query_obs: Vec<GRNPObservation>,
    pub normalized_max_error: f64,
}

#[derive(Debug, Clone)]
pub struct StructureLessAbsolutePoseEstimate {
    pub query_cam_from_world: SE3,
    pub num_inliers: usize,
    pub inlier_mask: Vec<bool>,
}

#[derive(Debug, Clone)]
pub struct GeneralizedAbsolutePoseObservations {
    pub observations: Vec<GRNPObservation>,
    pub points3d: Vec<[f64; 3]>,
    pub unique_point3d_ids: Vec<usize>,
    pub normalized_max_error: f64,
}

#[derive(Debug, Clone)]
pub struct GeneralizedAbsolutePoseEstimate {
    pub rig_from_world: SE3,
    pub num_inliers: usize,
    pub num_unique_inliers: usize,
    pub inlier_mask: Vec<bool>,
}

#[derive(Debug, Clone)]
pub struct GeneralizedRelativePoseObservations {
    pub observations1: Vec<GRNPObservation>,
    pub observations2: Vec<GRNPObservation>,
    pub normalized_max_error: f64,
    pub both_rigs_panoramic: bool,
}

#[derive(Debug, Clone)]
pub struct GeneralizedRelativePoseEstimate {
    pub rig2_from_rig1: Option<SE3>,
    pub pano2_from_pano1: Option<SE3>,
    pub num_inliers: usize,
    pub inlier_mask: Vec<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeneralizedPoseError {
    InvalidOptions(&'static str),
    InvalidInput(&'static str),
    MissingGeneralizedRelativePoseSolver,
    SolverFailed(&'static str),
}

impl fmt::Display for GeneralizedPoseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(message) | Self::InvalidInput(message) => f.write_str(message),
            Self::MissingGeneralizedRelativePoseSolver => f.write_str(
                "PoseLib generalized relative pose solver is not enabled; rebuild with --features poselib",
            ),
            Self::SolverFailed(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for GeneralizedPoseError {}

pub fn prepare_structureless_absolute_pose_observations(
    options: &StructureLessAbsolutePoseEstimationOptions,
    problem: StructureLessAbsolutePoseProblem<'_>,
) -> Result<Option<StructureLessAbsolutePoseObservations>, GeneralizedPoseError> {
    if problem.world_points2d.len() != problem.query_points2d.len() {
        return Err(GeneralizedPoseError::InvalidInput(
            "world and query point counts differ",
        ));
    }
    if problem.world_points2d.len() != problem.world_camera_idxs.len() {
        return Err(GeneralizedPoseError::InvalidInput(
            "world point and camera index counts differ",
        ));
    }
    if problem.world_cams_from_world.len() != problem.world_cameras.len() {
        return Err(GeneralizedPoseError::InvalidInput(
            "world pose and camera counts differ",
        ));
    }
    throw_check_cameras(
        problem.world_camera_idxs,
        problem.world_cams_from_world,
        problem.world_cameras,
    )?;
    options.check()?;

    if is_panoramic_rig(problem.world_camera_idxs, problem.world_cams_from_world)? {
        return Ok(None);
    }

    let num_points = problem.world_points2d.len();
    let mut world_obs = Vec::with_capacity(num_points);
    let mut query_obs = Vec::with_capacity(num_points);
    for idx in 0..num_points {
        let world_camera_idx = problem.world_camera_idxs[idx];
        let world_xy = problem.world_points2d[idx];
        let query_xy = problem.query_points2d[idx];
        world_obs.push(GRNPObservation {
            cam_from_rig: problem.world_cams_from_world[world_camera_idx],
            ray_in_cam: problem.world_cameras[world_camera_idx]
                .cam_ray_from_img(world_xy[0], world_xy[1])
                .unwrap_or([0.0, 0.0, 0.0]),
        });
        query_obs.push(GRNPObservation {
            cam_from_rig: SE3::identity(),
            ray_in_cam: problem
                .query_camera
                .cam_ray_from_img(query_xy[0], query_xy[1])
                .unwrap_or([0.0, 0.0, 0.0]),
        });
    }

    Ok(Some(StructureLessAbsolutePoseObservations {
        world_obs,
        query_obs,
        normalized_max_error: problem
            .query_camera
            .cam_from_img_threshold(options.ransac_options.max_error),
    }))
}

pub fn prepare_generalized_absolute_pose_observations(
    options: &GeneralizedAbsolutePoseEstimationOptions,
    problem: GeneralizedAbsolutePoseProblem<'_>,
) -> Result<Option<GeneralizedAbsolutePoseObservations>, GeneralizedPoseError> {
    if problem.points2d.len() != problem.points3d.len() {
        return Err(GeneralizedPoseError::InvalidInput(
            "2D and 3D point counts differ",
        ));
    }
    if problem.points2d.len() != problem.camera_idxs.len() {
        return Err(GeneralizedPoseError::InvalidInput(
            "point and camera index counts differ",
        ));
    }
    throw_check_cameras(problem.camera_idxs, problem.cams_from_rig, problem.cameras)?;
    options.check()?;

    let num_points = problem.points2d.len();
    if num_points == 0 {
        return Ok(None);
    }

    let mut observations = Vec::with_capacity(num_points);
    for idx in 0..num_points {
        let camera_idx = problem.camera_idxs[idx];
        let ray_in_cam = problem.cameras[camera_idx]
            .cam_ray_from_img(problem.points2d[idx][0], problem.points2d[idx][1])
            .unwrap_or([0.0, 0.0, 0.0]);
        observations.push(GRNPObservation {
            cam_from_rig: problem.cams_from_rig[camera_idx],
            ray_in_cam,
        });
    }

    Ok(Some(GeneralizedAbsolutePoseObservations {
        observations,
        points3d: problem.points3d.to_vec(),
        unique_point3d_ids: compute_unique_point3d_ids(problem.points3d),
        normalized_max_error: normalized_generalized_absolute_max_error(
            problem.camera_idxs,
            problem.cameras,
            options.ransac_options.max_error,
        )?,
    }))
}

pub fn prepare_generalized_relative_pose_observations(
    options: &GeneralizedRelativePoseEstimationOptions,
    problem: GeneralizedRelativePoseProblem<'_>,
) -> Result<Option<GeneralizedRelativePoseObservations>, GeneralizedPoseError> {
    if problem.points2d1.len() != problem.points2d2.len() {
        return Err(GeneralizedPoseError::InvalidInput(
            "two-view point counts differ",
        ));
    }
    if problem.points2d1.len() != problem.camera_idxs1.len()
        || problem.points2d1.len() != problem.camera_idxs2.len()
    {
        return Err(GeneralizedPoseError::InvalidInput(
            "two-view point and camera index counts differ",
        ));
    }
    throw_check_cameras(problem.camera_idxs1, problem.cams_from_rig, problem.cameras)?;
    throw_check_cameras(problem.camera_idxs2, problem.cams_from_rig, problem.cameras)?;
    options.check()?;

    let num_points = problem.points2d1.len();
    if num_points == 0 {
        return Ok(None);
    }

    let both_rigs_panoramic = is_panoramic_rig(problem.camera_idxs1, problem.cams_from_rig)?
        && is_panoramic_rig(problem.camera_idxs2, problem.cams_from_rig)?;
    let mut observations1 = Vec::with_capacity(num_points);
    let mut observations2 = Vec::with_capacity(num_points);
    for idx in 0..num_points {
        let camera_idx1 = problem.camera_idxs1[idx];
        let camera_idx2 = problem.camera_idxs2[idx];
        let ray1 = problem.cameras[camera_idx1]
            .cam_ray_from_img(problem.points2d1[idx][0], problem.points2d1[idx][1])
            .unwrap_or([0.0, 0.0, 0.0]);
        let ray2 = problem.cameras[camera_idx2]
            .cam_ray_from_img(problem.points2d2[idx][0], problem.points2d2[idx][1])
            .unwrap_or([0.0, 0.0, 0.0]);
        observations1.push(GRNPObservation {
            cam_from_rig: problem.cams_from_rig[camera_idx1],
            ray_in_cam: if both_rigs_panoramic {
                rotate_ray_to_rig(problem.cams_from_rig[camera_idx1], ray1)
            } else {
                ray1
            },
        });
        observations2.push(GRNPObservation {
            cam_from_rig: problem.cams_from_rig[camera_idx2],
            ray_in_cam: if both_rigs_panoramic {
                rotate_ray_to_rig(problem.cams_from_rig[camera_idx2], ray2)
            } else {
                ray2
            },
        });
    }

    Ok(Some(GeneralizedRelativePoseObservations {
        observations1,
        observations2,
        normalized_max_error: normalized_generalized_relative_max_error(
            problem.cameras,
            options.ransac_options.max_error,
        )?,
        both_rigs_panoramic,
    }))
}

pub fn estimate_generalized_relative_pose(
    options: &GeneralizedRelativePoseEstimationOptions,
    problem: GeneralizedRelativePoseProblem<'_>,
) -> Result<Option<GeneralizedRelativePoseEstimate>, GeneralizedPoseError> {
    let Some(observations) = prepare_generalized_relative_pose_observations(options, problem)?
    else {
        return Ok(None);
    };

    if observations.both_rigs_panoramic {
        return estimate_panoramic_generalized_relative_pose(options, &observations);
    }

    estimate_non_panoramic_generalized_relative_pose(options, &observations)
}

pub fn original_camera_relative_pose_from_rig_relative_pose(
    rig2_from_rig1: SE3,
    orig_cam1_from_rig1: SE3,
    orig_cam2_from_rig2: SE3,
) -> SE3 {
    orig_cam2_from_rig2
        .compose(&rig2_from_rig1)
        .compose(&orig_cam1_from_rig1.inverse())
}

pub fn estimate_structureless_absolute_pose(
    options: &StructureLessAbsolutePoseEstimationOptions,
    problem: StructureLessAbsolutePoseProblem<'_>,
) -> Result<Option<StructureLessAbsolutePoseEstimate>, GeneralizedPoseError> {
    let Some(observations) = prepare_structureless_absolute_pose_observations(options, problem)?
    else {
        return Ok(None);
    };

    estimate_structureless_absolute_pose_from_observations(options, &observations)
}

pub fn estimate_generalized_absolute_pose(
    options: &GeneralizedAbsolutePoseEstimationOptions,
    problem: GeneralizedAbsolutePoseProblem<'_>,
) -> Result<Option<GeneralizedAbsolutePoseEstimate>, GeneralizedPoseError> {
    let Some(observations) = prepare_generalized_absolute_pose_observations(options, problem)?
    else {
        return Ok(None);
    };

    estimate_generalized_absolute_pose_from_observations(options, &observations)
}

#[cfg(feature = "poselib")]
fn estimate_generalized_absolute_pose_from_observations(
    options: &GeneralizedAbsolutePoseEstimationOptions,
    observations: &GeneralizedAbsolutePoseObservations,
) -> Result<Option<GeneralizedAbsolutePoseEstimate>, GeneralizedPoseError> {
    estimate_generalized_absolute_pose_ransac(options, observations)
}

#[cfg(not(feature = "poselib"))]
fn estimate_generalized_absolute_pose_from_observations(
    _options: &GeneralizedAbsolutePoseEstimationOptions,
    _observations: &GeneralizedAbsolutePoseObservations,
) -> Result<Option<GeneralizedAbsolutePoseEstimate>, GeneralizedPoseError> {
    Err(GeneralizedPoseError::MissingGeneralizedRelativePoseSolver)
}

#[cfg(feature = "poselib")]
fn estimate_structureless_absolute_pose_from_observations(
    options: &StructureLessAbsolutePoseEstimationOptions,
    observations: &StructureLessAbsolutePoseObservations,
) -> Result<Option<StructureLessAbsolutePoseEstimate>, GeneralizedPoseError> {
    let generalized_options = GeneralizedRelativePoseEstimationOptions {
        ransac_options: RansacOptions {
            max_error: observations.normalized_max_error,
            ..options.ransac_options
        },
    };
    let generalized_observations = GeneralizedRelativePoseObservations {
        observations1: observations.world_obs.clone(),
        observations2: observations.query_obs.clone(),
        normalized_max_error: observations.normalized_max_error,
        both_rigs_panoramic: false,
    };
    let Some(estimate) =
        estimate_generalized_relative_pose_ransac(&generalized_options, &generalized_observations)?
    else {
        return Ok(None);
    };
    let Some(query_cam_from_world) = estimate.rig2_from_rig1 else {
        return Ok(None);
    };
    Ok(Some(StructureLessAbsolutePoseEstimate {
        query_cam_from_world,
        num_inliers: estimate.num_inliers,
        inlier_mask: estimate.inlier_mask,
    }))
}

#[cfg(not(feature = "poselib"))]
fn estimate_structureless_absolute_pose_from_observations(
    _options: &StructureLessAbsolutePoseEstimationOptions,
    _observations: &StructureLessAbsolutePoseObservations,
) -> Result<Option<StructureLessAbsolutePoseEstimate>, GeneralizedPoseError> {
    Err(GeneralizedPoseError::MissingGeneralizedRelativePoseSolver)
}

fn throw_check_cameras(
    camera_idxs: &[usize],
    cams_from_rig: &[SE3],
    cameras: &[CameraModel],
) -> Result<(), GeneralizedPoseError> {
    if cameras.is_empty() {
        return Err(GeneralizedPoseError::InvalidInput(
            "world cameras are empty",
        ));
    }
    if cams_from_rig.len() != cameras.len() {
        return Err(GeneralizedPoseError::InvalidInput(
            "world pose and camera counts differ",
        ));
    }
    let Some(max_camera_idx) = camera_idxs.iter().copied().max() else {
        return Err(GeneralizedPoseError::InvalidInput(
            "world camera indices are empty",
        ));
    };
    if max_camera_idx >= cameras.len() {
        return Err(GeneralizedPoseError::InvalidInput(
            "world camera index is out of range",
        ));
    }
    Ok(())
}

fn is_panoramic_rig(
    camera_idxs: &[usize],
    cams_from_rig: &[SE3],
) -> Result<bool, GeneralizedPoseError> {
    let camera_idx_set = camera_idxs.iter().copied().collect::<BTreeSet<_>>();
    let Some(first_camera_idx) = camera_idx_set.iter().next().copied() else {
        return Err(GeneralizedPoseError::InvalidInput(
            "world camera indices are empty",
        ));
    };
    let first_origin = camera_center(cams_from_rig[first_camera_idx]);
    for camera_idx in camera_idx_set.iter().skip(1).copied() {
        let other_origin = camera_center(cams_from_rig[camera_idx]);
        if (first_origin - other_origin).length() > 1.0e-6 {
            return Ok(false);
        }
    }
    Ok(true)
}

fn normalized_generalized_relative_max_error(
    cameras: &[CameraModel],
    max_error_px: f64,
) -> Result<f64, GeneralizedPoseError> {
    if cameras.is_empty() {
        return Err(GeneralizedPoseError::InvalidInput(
            "world cameras are empty",
        ));
    }
    let mut normalized = 0.0;
    for camera in cameras {
        normalized += camera.cam_from_img_threshold(max_error_px);
    }
    Ok(normalized / cameras.len() as f64)
}

fn normalized_generalized_absolute_max_error(
    camera_idxs: &[usize],
    cameras: &[CameraModel],
    max_error_px: f64,
) -> Result<f64, GeneralizedPoseError> {
    if camera_idxs.is_empty() {
        return Err(GeneralizedPoseError::InvalidInput(
            "world camera indices are empty",
        ));
    }
    let mut normalized = 0.0;
    for &camera_idx in camera_idxs {
        let Some(camera) = cameras.get(camera_idx) else {
            return Err(GeneralizedPoseError::InvalidInput(
                "world camera index is out of range",
            ));
        };
        normalized += camera.cam_from_img_threshold(max_error_px);
    }
    Ok(normalized / camera_idxs.len() as f64)
}

fn compute_unique_point3d_ids(points3d: &[[f64; 3]]) -> Vec<usize> {
    let mut order = (0..points3d.len()).collect::<Vec<_>>();
    order.sort_by(|&i, &j| {
        points3d[i][0]
            .total_cmp(&points3d[j][0])
            .then(points3d[i][1].total_cmp(&points3d[j][1]))
            .then(points3d[i][2].total_cmp(&points3d[j][2]))
    });

    let mut unique_ids = vec![0usize; points3d.len()];
    let Some(&first) = order.first() else {
        return unique_ids;
    };
    let mut unique_point = first;
    for idx in order {
        if !point3d_is_approx(points3d[unique_point], points3d[idx], 1.0e-5) {
            unique_point = idx;
        }
        unique_ids[idx] = unique_point;
    }
    unique_ids
}

fn point3d_is_approx(left: [f64; 3], right: [f64; 3], eps: f64) -> bool {
    (left[0] - right[0]).abs() <= eps
        && (left[1] - right[1]).abs() <= eps
        && (left[2] - right[2]).abs() <= eps
}

fn rotate_ray_to_rig(cam_from_rig: SE3, ray_in_cam: [f64; 3]) -> [f64; 3] {
    let q = cam_from_rig.quaternion();
    let rotation = glam::Quat::from_xyzw(q[0], q[1], q[2], q[3]).normalize();
    let ray = Vec3::new(
        ray_in_cam[0] as f32,
        ray_in_cam[1] as f32,
        ray_in_cam[2] as f32,
    );
    let ray = rotation.inverse() * ray;
    [ray.x as f64, ray.y as f64, ray.z as f64]
}

#[cfg(feature = "poselib")]
fn estimate_non_panoramic_generalized_relative_pose(
    options: &GeneralizedRelativePoseEstimationOptions,
    observations: &GeneralizedRelativePoseObservations,
) -> Result<Option<GeneralizedRelativePoseEstimate>, GeneralizedPoseError> {
    estimate_generalized_relative_pose_ransac(options, observations)
}

#[cfg(not(feature = "poselib"))]
fn estimate_non_panoramic_generalized_relative_pose(
    _options: &GeneralizedRelativePoseEstimationOptions,
    _observations: &GeneralizedRelativePoseObservations,
) -> Result<Option<GeneralizedRelativePoseEstimate>, GeneralizedPoseError> {
    Err(GeneralizedPoseError::MissingGeneralizedRelativePoseSolver)
}

fn estimate_panoramic_generalized_relative_pose(
    options: &GeneralizedRelativePoseEstimationOptions,
    observations: &GeneralizedRelativePoseObservations,
) -> Result<Option<GeneralizedRelativePoseEstimate>, GeneralizedPoseError> {
    let rays1 = observations
        .observations1
        .iter()
        .map(|obs| obs.ray_in_cam)
        .collect::<Vec<_>>();
    let rays2 = observations
        .observations2
        .iter()
        .map(|obs| obs.ray_in_cam)
        .collect::<Vec<_>>();
    let Some((pose, num_inliers, inlier_mask)) = crate::two_view::estimate_relative_pose_from_rays(
        &rays1,
        &rays2,
        observations.normalized_max_error,
        options.ransac_options.min_inlier_ratio,
        options.ransac_options.min_num_trials,
        options.ransac_options.max_num_trials,
        options.ransac_options.confidence,
        options.ransac_options.dyn_num_trials_multiplier,
        options.ransac_options.random_seed,
    ) else {
        return Ok(None);
    };

    Ok(Some(GeneralizedRelativePoseEstimate {
        rig2_from_rig1: None,
        pano2_from_pano1: Some(pose),
        num_inliers,
        inlier_mask,
    }))
}

#[cfg(feature = "poselib")]
fn estimate_generalized_absolute_pose_ransac(
    options: &GeneralizedAbsolutePoseEstimationOptions,
    observations: &GeneralizedAbsolutePoseObservations,
) -> Result<Option<GeneralizedAbsolutePoseEstimate>, GeneralizedPoseError> {
    const SAMPLE_SIZE: usize = 3;
    let num_points = observations.observations.len();
    if num_points < SAMPLE_SIZE {
        return Ok(None);
    }

    let active_indices = (0..num_points).collect::<Vec<_>>();
    let seed = generalized_pose_sampler_seed(options.ransac_options.random_seed);
    let mut sampler = ColmapRandomSampler::new(seed, &active_indices);
    let threshold_sq = observations.normalized_max_error * observations.normalized_max_error;
    let ransac_options = options
        .ransac_options
        .as_colmap_options()
        .with_initial_max_num_trials(SAMPLE_SIZE)
        .map_err(GeneralizedPoseError::InvalidOptions)?;
    let max_num_trials = ransac_options.max_num_trials;
    let mut dynamic_max_trials = max_num_trials;

    let mut best = None::<GeneralizedAbsoluteModelSupport>;
    let mut trial = 0usize;
    let mut abort = false;
    while trial < max_num_trials && !abort {
        let curr_thread_trial = trial;
        trial += 1;
        let sample = sampler.sample(SAMPLE_SIZE);
        if sample.len() != SAMPLE_SIZE {
            break;
        }

        for pose in
            poselib_gp3p_estimate(&observations.observations, &observations.points3d, &sample)?
        {
            let support = score_generalized_absolute_pose(
                pose,
                &observations.observations,
                &observations.points3d,
                &observations.unique_point3d_ids,
                threshold_sq,
            );
            if generalized_absolute_support_passes_colmap_success_gate(&support, SAMPLE_SIZE)
                && best
                    .as_ref()
                    .is_none_or(|current| support.is_better_than(current))
            {
                dynamic_max_trials = ransac_trials_from_counts(
                    support.num_inliers,
                    num_points,
                    SAMPLE_SIZE,
                    ransac_options.confidence,
                    ransac_options.dyn_num_trials_multiplier,
                );
                best = Some(support);
            }
            if colmap_ransac_abort_after_trial(
                curr_thread_trial,
                dynamic_max_trials,
                ransac_options.min_num_trials,
            ) {
                abort = true;
                break;
            }
        }
    }

    let Some(best) = best else {
        return Ok(None);
    };
    Ok(Some(GeneralizedAbsolutePoseEstimate {
        rig_from_world: best.pose,
        num_inliers: generalized_absolute_colmap_report_num_inliers(&best),
        num_unique_inliers: best.num_unique_inliers,
        inlier_mask: best.inlier_mask,
    }))
}

#[cfg(feature = "poselib")]
#[derive(Debug, Clone)]
struct GeneralizedAbsoluteModelSupport {
    pose: SE3,
    num_unique_inliers: usize,
    num_inliers: usize,
    residual_sum: f64,
    inlier_mask: Vec<bool>,
}

#[cfg(feature = "poselib")]
impl GeneralizedAbsoluteModelSupport {
    fn is_better_than(&self, other: &Self) -> bool {
        self.num_unique_inliers > other.num_unique_inliers
            || (self.num_unique_inliers == other.num_unique_inliers
                && (self.num_inliers > other.num_inliers
                    || (self.num_inliers == other.num_inliers
                        && self.residual_sum < other.residual_sum)))
    }
}

#[cfg(feature = "poselib")]
fn generalized_absolute_support_passes_colmap_success_gate(
    support: &GeneralizedAbsoluteModelSupport,
    min_num_samples: usize,
) -> bool {
    support.num_inliers >= min_num_samples
}

#[cfg(feature = "poselib")]
fn generalized_absolute_colmap_report_num_inliers(
    support: &GeneralizedAbsoluteModelSupport,
) -> usize {
    support.num_unique_inliers
}

#[cfg(feature = "poselib")]
fn colmap_ransac_abort_after_trial(
    curr_thread_trial: usize,
    dynamic_max_trials: usize,
    min_num_trials: usize,
) -> bool {
    curr_thread_trial >= dynamic_max_trials && curr_thread_trial >= min_num_trials
}

#[cfg(feature = "poselib")]
fn score_generalized_absolute_pose(
    rig_from_world: SE3,
    observations: &[GRNPObservation],
    points3d: &[[f64; 3]],
    unique_point3d_ids: &[usize],
    threshold_sq: f64,
) -> GeneralizedAbsoluteModelSupport {
    let mut inlier_mask = Vec::with_capacity(observations.len());
    let mut unique_inliers = BTreeSet::new();
    let mut num_inliers = 0usize;
    let mut residual_sum = 0.0;

    for (idx, (obs, point3d)) in observations.iter().zip(points3d.iter()).enumerate() {
        let residual =
            generalized_absolute_reprojection_residual_sq(rig_from_world, *obs, *point3d)
                .unwrap_or(f64::INFINITY);
        let is_inlier = residual <= threshold_sq;
        if is_inlier {
            num_inliers += 1;
            if let Some(&unique_id) = unique_point3d_ids.get(idx) {
                unique_inliers.insert(unique_id);
            }
            residual_sum += residual;
        }
        inlier_mask.push(is_inlier);
    }

    GeneralizedAbsoluteModelSupport {
        pose: rig_from_world,
        num_unique_inliers: unique_inliers.len(),
        num_inliers,
        residual_sum,
        inlier_mask,
    }
}

#[cfg(feature = "poselib")]
fn generalized_absolute_reprojection_residual_sq(
    rig_from_world: SE3,
    obs: GRNPObservation,
    point3d: [f64; 3],
) -> Option<f64> {
    let point = [point3d[0] as f32, point3d[1] as f32, point3d[2] as f32];
    let point_in_rig = rig_from_world.transform_point(&point);
    let point_in_cam = obs.cam_from_rig.transform_point(&point_in_rig);
    if point_in_cam[2] <= f32::EPSILON || !point_in_cam.iter().all(|v| v.is_finite()) {
        return None;
    }
    let x = point_in_cam[0] as f64 / point_in_cam[2] as f64;
    let y = point_in_cam[1] as f64 / point_in_cam[2] as f64;
    let ray_x = obs.ray_in_cam[0] / obs.ray_in_cam[2];
    let ray_y = obs.ray_in_cam[1] / obs.ray_in_cam[2];
    let dx = ray_x - x;
    let dy = ray_y - y;
    let residual = dx * dx + dy * dy;
    residual.is_finite().then_some(residual)
}

#[cfg(feature = "poselib")]
fn estimate_generalized_relative_pose_ransac(
    options: &GeneralizedRelativePoseEstimationOptions,
    observations: &GeneralizedRelativePoseObservations,
) -> Result<Option<GeneralizedRelativePoseEstimate>, GeneralizedPoseError> {
    const SAMPLE_SIZE: usize = 6;
    let num_points = observations.observations1.len();
    if num_points < SAMPLE_SIZE {
        return Ok(None);
    }

    let active_indices = (0..num_points).collect::<Vec<_>>();
    let seed = generalized_pose_sampler_seed(options.ransac_options.random_seed);
    let mut sampler = ColmapRandomSampler::new(seed, &active_indices);
    let threshold_sq = observations.normalized_max_error * observations.normalized_max_error;
    let ransac_options = options
        .ransac_options
        .as_colmap_options()
        .with_initial_max_num_trials(SAMPLE_SIZE)
        .map_err(GeneralizedPoseError::InvalidOptions)?;
    let max_num_trials = ransac_options.max_num_trials;
    let mut dynamic_max_trials = max_num_trials;

    let mut best: Option<GeneralizedRelativeModelSupport> = None;
    let mut trial = 0usize;
    let mut abort = false;
    while trial < max_num_trials && !abort {
        let curr_thread_trial = trial;
        trial += 1;
        let sample = sampler.sample(SAMPLE_SIZE);
        if sample.len() != SAMPLE_SIZE {
            break;
        }

        for pose in poselib_gr6p_estimate(
            &observations.observations1,
            &observations.observations2,
            &sample,
        )? {
            let mut support = score_generalized_relative_pose(
                pose,
                &observations.observations1,
                &observations.observations2,
                threshold_sq,
            );
            if support.num_inliers >= SAMPLE_SIZE
                && best
                    .as_ref()
                    .is_none_or(|current| support.is_better_than(current))
            {
                support = refine_generalized_relative_pose_with_gr8p(
                    &observations.observations1,
                    &observations.observations2,
                    threshold_sq,
                    support,
                    seed.wrapping_add(trial as u64) as u32,
                )?;
                dynamic_max_trials = ransac_trials_from_counts(
                    support.num_inliers,
                    num_points,
                    SAMPLE_SIZE,
                    ransac_options.confidence,
                    ransac_options.dyn_num_trials_multiplier,
                );
                best = Some(support);
            }
            if colmap_ransac_abort_after_trial(
                curr_thread_trial,
                dynamic_max_trials,
                ransac_options.min_num_trials,
            ) {
                abort = true;
                break;
            }
        }
    }

    let Some(best) = best else {
        return Ok(None);
    };
    Ok(Some(GeneralizedRelativePoseEstimate {
        rig2_from_rig1: Some(best.pose),
        pano2_from_pano1: None,
        num_inliers: best.num_inliers,
        inlier_mask: best.inlier_mask,
    }))
}

#[cfg(feature = "poselib")]
#[derive(Debug, Clone)]
struct GeneralizedRelativeModelSupport {
    pose: SE3,
    num_inliers: usize,
    residual_sum: f64,
    inlier_mask: Vec<bool>,
}

#[cfg(feature = "poselib")]
impl GeneralizedRelativeModelSupport {
    fn is_better_than(&self, other: &Self) -> bool {
        self.num_inliers > other.num_inliers
            || (self.num_inliers == other.num_inliers && self.residual_sum < other.residual_sum)
    }
}

#[cfg(feature = "poselib")]
fn score_generalized_relative_pose(
    rig2_from_rig1: SE3,
    observations1: &[GRNPObservation],
    observations2: &[GRNPObservation],
    threshold_sq: f64,
) -> GeneralizedRelativeModelSupport {
    let mut inlier_mask = Vec::with_capacity(observations1.len());
    let mut num_inliers = 0usize;
    let mut residual_sum = 0.0;

    for (obs1, obs2) in observations1.iter().zip(observations2.iter()) {
        let residual =
            generalized_sampson_residual_sq(rig2_from_rig1, *obs1, *obs2).unwrap_or(f64::INFINITY);
        let is_inlier = residual <= threshold_sq;
        if is_inlier {
            num_inliers += 1;
            residual_sum += residual;
        }
        inlier_mask.push(is_inlier);
    }

    GeneralizedRelativeModelSupport {
        pose: rig2_from_rig1,
        num_inliers,
        residual_sum,
        inlier_mask,
    }
}

#[cfg(feature = "poselib")]
fn refine_generalized_relative_pose_with_gr8p(
    observations1: &[GRNPObservation],
    observations2: &[GRNPObservation],
    threshold_sq: f64,
    support: GeneralizedRelativeModelSupport,
    random_seed: u32,
) -> Result<GeneralizedRelativeModelSupport, GeneralizedPoseError> {
    let mut best = support;
    const MAX_LOCAL_TRIALS: usize = 10;
    for local_trial in 0..MAX_LOCAL_TRIALS {
        let prev_best_num_inliers = best.num_inliers;
        let inlier_indices = best
            .inlier_mask
            .iter()
            .enumerate()
            .filter_map(|(idx, &is_inlier)| is_inlier.then_some(idx))
            .collect::<Vec<_>>();
        if inlier_indices.len() < 8 || best.num_inliers <= 6 {
            break;
        }

        for pose in poselib_gr8p_estimate(
            observations1,
            observations2,
            &inlier_indices,
            random_seed.wrapping_add(local_trial as u32),
        )? {
            let candidate =
                score_generalized_relative_pose(pose, observations1, observations2, threshold_sq);
            if candidate.is_better_than(&best) {
                best = candidate;
            }
        }

        if best.num_inliers <= prev_best_num_inliers {
            break;
        }
    }
    Ok(best)
}

#[cfg(feature = "poselib")]
fn generalized_sampson_residual_sq(
    rig2_from_rig1: SE3,
    obs1: GRNPObservation,
    obs2: GRNPObservation,
) -> Option<f64> {
    let cam2_from_cam1 = obs2
        .cam_from_rig
        .compose(&rig2_from_rig1)
        .compose(&obs1.cam_from_rig.inverse());
    let rotation = rotation_matrix_f64(cam2_from_cam1);
    let translation = normalized_translation_f64(cam2_from_cam1)?;
    let e = skew_f64(translation).mul_mat3(rotation);
    let ray1 = Vec3d::new(obs1.ray_in_cam[0], obs1.ray_in_cam[1], obs1.ray_in_cam[2]);
    let ray2 = Vec3d::new(obs2.ray_in_cam[0], obs2.ray_in_cam[1], obs2.ray_in_cam[2]);
    let epipolar_line1 = e.mul_vec3(ray1);
    let e_col0 = e.col(0);
    let e_col1 = e.col(1);
    let num = ray2.dot(epipolar_line1);
    let denom = ray2.dot(e_col0).powi(2)
        + ray2.dot(e_col1).powi(2)
        + epipolar_line1.x.powi(2)
        + epipolar_line1.y.powi(2);
    (denom > 1.0e-24 && denom.is_finite()).then_some(num * num / denom)
}

#[cfg(feature = "poselib")]
fn poselib_gr6p_estimate(
    observations1: &[GRNPObservation],
    observations2: &[GRNPObservation],
    sample: &[usize],
) -> Result<Vec<SE3>, GeneralizedPoseError> {
    let mut origins1 = [0.0; 18];
    let mut origins2 = [0.0; 18];
    let mut rays1 = [0.0; 18];
    let mut rays2 = [0.0; 18];

    for (sample_idx, &obs_idx) in sample.iter().enumerate() {
        let obs1 = observations1[obs_idx];
        let obs2 = observations2[obs_idx];
        let origin1 = camera_center_f64(obs1.cam_from_rig);
        let origin2 = camera_center_f64(obs2.cam_from_rig);
        let Some(ray1) = rotate_ray_to_rig_f64(obs1.cam_from_rig, obs1.ray_in_cam) else {
            return Ok(Vec::new());
        };
        let Some(ray2) = rotate_ray_to_rig_f64(obs2.cam_from_rig, obs2.ray_in_cam) else {
            return Ok(Vec::new());
        };
        for axis in 0..3 {
            origins1[3 * sample_idx + axis] = origin1[axis];
            origins2[3 * sample_idx + axis] = origin2[axis];
            rays1[3 * sample_idx + axis] = ray1[axis];
            rays2[3 * sample_idx + axis] = ray2[axis];
        }
    }

    poselib_ffi::gen_relpose_6pt(&origins1, &rays1, &origins2, &rays2)
}

#[cfg(feature = "poselib")]
fn poselib_gr8p_estimate(
    observations1: &[GRNPObservation],
    observations2: &[GRNPObservation],
    sample: &[usize],
    random_seed: u32,
) -> Result<Vec<SE3>, GeneralizedPoseError> {
    let num_points = sample.len();
    if num_points < 8 {
        return Ok(Vec::new());
    }

    let mut origins1 = vec![0.0; 3 * num_points];
    let mut origins2 = vec![0.0; 3 * num_points];
    let mut rays1 = vec![0.0; 3 * num_points];
    let mut rays2 = vec![0.0; 3 * num_points];
    let mut cam_qvecs1 = vec![0.0; 4 * num_points];
    let mut cam_qvecs2 = vec![0.0; 4 * num_points];
    let mut cam_tvecs1 = vec![0.0; 3 * num_points];
    let mut cam_tvecs2 = vec![0.0; 3 * num_points];

    for (sample_idx, &obs_idx) in sample.iter().enumerate() {
        let obs1 = observations1[obs_idx];
        let obs2 = observations2[obs_idx];
        let origin1 = camera_center_f64(obs1.cam_from_rig);
        let origin2 = camera_center_f64(obs2.cam_from_rig);
        let (qvec1, tvec1) = cam_from_rig_qvec_tvec_f64(obs1.cam_from_rig);
        let (qvec2, tvec2) = cam_from_rig_qvec_tvec_f64(obs2.cam_from_rig);
        for axis in 0..3 {
            origins1[3 * sample_idx + axis] = origin1[axis];
            origins2[3 * sample_idx + axis] = origin2[axis];
            rays1[3 * sample_idx + axis] = obs1.ray_in_cam[axis];
            rays2[3 * sample_idx + axis] = obs2.ray_in_cam[axis];
            cam_tvecs1[3 * sample_idx + axis] = tvec1[axis];
            cam_tvecs2[3 * sample_idx + axis] = tvec2[axis];
        }
        for axis in 0..4 {
            cam_qvecs1[4 * sample_idx + axis] = qvec1[axis];
            cam_qvecs2[4 * sample_idx + axis] = qvec2[axis];
        }
    }

    poselib_ffi::gen_relpose_8pt(
        &origins1,
        &rays1,
        &origins2,
        &rays2,
        &cam_qvecs1,
        &cam_tvecs1,
        &cam_qvecs2,
        &cam_tvecs2,
        random_seed,
    )
}

#[cfg(feature = "poselib")]
fn poselib_gp3p_estimate(
    observations: &[GRNPObservation],
    points3d: &[[f64; 3]],
    sample: &[usize],
) -> Result<Vec<SE3>, GeneralizedPoseError> {
    if sample.len() != 3 {
        return Ok(Vec::new());
    }

    let mut origins = [0.0; 9];
    let mut rays = [0.0; 9];
    let mut world_points = [0.0; 9];

    for (sample_idx, &obs_idx) in sample.iter().enumerate() {
        let obs = observations[obs_idx];
        let origin = camera_center_f64(obs.cam_from_rig);
        let Some(ray) = rotate_ray_to_rig_f64(obs.cam_from_rig, obs.ray_in_cam) else {
            return Ok(Vec::new());
        };
        for axis in 0..3 {
            origins[3 * sample_idx + axis] = origin[axis];
            rays[3 * sample_idx + axis] = ray[axis];
            world_points[3 * sample_idx + axis] = points3d[obs_idx][axis];
        }
    }

    poselib_ffi::gp3p(&origins, &rays, &world_points)
}

#[cfg(feature = "poselib")]
fn camera_center_f64(pose: SE3) -> [f64; 3] {
    let rotation_inv = rotation_matrix_f64(pose).transpose();
    let t = pose.translation();
    let center = rotation_inv.mul_vec3(Vec3d::new(-(t[0] as f64), -(t[1] as f64), -(t[2] as f64)));
    [center.x, center.y, center.z]
}

#[cfg(feature = "poselib")]
fn cam_from_rig_qvec_tvec_f64(pose: SE3) -> ([f64; 4], [f64; 3]) {
    let q = pose.quaternion();
    let t = pose.translation();
    (
        [q[3] as f64, q[0] as f64, q[1] as f64, q[2] as f64],
        [t[0] as f64, t[1] as f64, t[2] as f64],
    )
}

#[cfg(feature = "poselib")]
fn rotate_ray_to_rig_f64(cam_from_rig: SE3, ray_in_cam: [f64; 3]) -> Option<[f64; 3]> {
    let rotation_inv = rotation_matrix_f64(cam_from_rig).transpose();
    let ray = rotation_inv.mul_vec3(Vec3d::new(ray_in_cam[0], ray_in_cam[1], ray_in_cam[2]));
    let ray = ray.normalized()?;
    Some([ray.x, ray.y, ray.z])
}

#[cfg(feature = "poselib")]
fn normalized_translation_f64(pose: SE3) -> Option<Vec3d> {
    let t = pose.translation();
    Vec3d::new(t[0] as f64, t[1] as f64, t[2] as f64).normalized()
}

#[cfg(feature = "poselib")]
fn rotation_matrix_f64(pose: SE3) -> Mat3d {
    let r = pose.rotation_matrix();
    Mat3d {
        m: [
            [r[0][0] as f64, r[0][1] as f64, r[0][2] as f64],
            [r[1][0] as f64, r[1][1] as f64, r[1][2] as f64],
            [r[2][0] as f64, r[2][1] as f64, r[2][2] as f64],
        ],
    }
}

#[cfg(feature = "poselib")]
fn skew_f64(v: Vec3d) -> Mat3d {
    Mat3d {
        m: [[0.0, -v.z, v.y], [v.z, 0.0, -v.x], [-v.y, v.x, 0.0]],
    }
}

#[cfg(feature = "poselib")]
fn ransac_trials_from_counts(
    num_inliers: usize,
    num_samples: usize,
    sample_size: usize,
    confidence: f64,
    multiplier: f64,
) -> usize {
    colmap_ransac_num_trials(
        num_inliers,
        num_samples,
        sample_size,
        confidence,
        multiplier,
    )
}

#[cfg(feature = "poselib")]
#[derive(Debug, Clone, Copy)]
struct Vec3d {
    x: f64,
    y: f64,
    z: f64,
}

#[cfg(feature = "poselib")]
impl Vec3d {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    fn normalized(self) -> Option<Self> {
        let norm = self.norm();
        (norm > 1.0e-24 && norm.is_finite()).then_some(Self::new(
            self.x / norm,
            self.y / norm,
            self.z / norm,
        ))
    }
}

#[cfg(feature = "poselib")]
#[derive(Debug, Clone, Copy)]
struct Mat3d {
    m: [[f64; 3]; 3],
}

#[cfg(feature = "poselib")]
impl Mat3d {
    fn mul_vec3(self, v: Vec3d) -> Vec3d {
        Vec3d::new(
            self.m[0][0] * v.x + self.m[0][1] * v.y + self.m[0][2] * v.z,
            self.m[1][0] * v.x + self.m[1][1] * v.y + self.m[1][2] * v.z,
            self.m[2][0] * v.x + self.m[2][1] * v.y + self.m[2][2] * v.z,
        )
    }

    fn mul_mat3(self, rhs: Self) -> Self {
        let mut m = [[0.0; 3]; 3];
        for (r, row) in m.iter_mut().enumerate() {
            for (c, value) in row.iter_mut().enumerate() {
                *value = self.m[r][0] * rhs.m[0][c]
                    + self.m[r][1] * rhs.m[1][c]
                    + self.m[r][2] * rhs.m[2][c];
            }
        }
        Self { m }
    }

    fn transpose(self) -> Self {
        Self {
            m: [
                [self.m[0][0], self.m[1][0], self.m[2][0]],
                [self.m[0][1], self.m[1][1], self.m[2][1]],
                [self.m[0][2], self.m[1][2], self.m[2][2]],
            ],
        }
    }

    fn col(self, idx: usize) -> Vec3d {
        Vec3d::new(self.m[0][idx], self.m[1][idx], self.m[2][idx])
    }
}

#[cfg(feature = "poselib")]
mod poselib_ffi {
    use super::*;
    use std::os::raw::{c_double, c_int};

    const MAX_POSELIB_POSES: usize = 128;

    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    struct RustSfmPoseLibPose {
        qvec: [c_double; 4],
        tvec: [c_double; 3],
    }

    extern "C" {
        fn rustsfm_poselib_gen_relpose_6pt(
            origins1: *const c_double,
            rays1: *const c_double,
            origins2: *const c_double,
            rays2: *const c_double,
            num_points: usize,
            output: *mut RustSfmPoseLibPose,
            max_output: usize,
            num_output: *mut usize,
        ) -> c_int;

        fn rustsfm_poselib_gen_relpose_8pt(
            origins1: *const c_double,
            rays1: *const c_double,
            origins2: *const c_double,
            rays2: *const c_double,
            cam_qvecs1: *const c_double,
            cam_tvecs1: *const c_double,
            cam_qvecs2: *const c_double,
            cam_tvecs2: *const c_double,
            num_points: usize,
            random_seed: u32,
            output: *mut RustSfmPoseLibPose,
            max_output: usize,
            num_output: *mut usize,
        ) -> c_int;

        fn rustsfm_poselib_gp3p(
            origins: *const c_double,
            rays: *const c_double,
            points3d: *const c_double,
            num_points: usize,
            output: *mut RustSfmPoseLibPose,
            max_output: usize,
            num_output: *mut usize,
        ) -> c_int;
    }

    pub(super) fn gen_relpose_6pt(
        origins1: &[f64; 18],
        rays1: &[f64; 18],
        origins2: &[f64; 18],
        rays2: &[f64; 18],
    ) -> Result<Vec<SE3>, GeneralizedPoseError> {
        let mut output = [RustSfmPoseLibPose {
            qvec: [0.0; 4],
            tvec: [0.0; 3],
        }; MAX_POSELIB_POSES];
        let mut num_output = 0usize;
        let status = unsafe {
            rustsfm_poselib_gen_relpose_6pt(
                origins1.as_ptr(),
                rays1.as_ptr(),
                origins2.as_ptr(),
                rays2.as_ptr(),
                6,
                output.as_mut_ptr(),
                output.len(),
                &mut num_output,
            )
        };
        if status != 0 {
            return Err(GeneralizedPoseError::SolverFailed(
                "PoseLib gen_relpose_6pt failed",
            ));
        }
        Ok(output
            .iter()
            .take(num_output.min(output.len()))
            .filter_map(|pose| pose_from_poselib(*pose))
            .collect())
    }

    pub(super) fn gen_relpose_8pt(
        origins1: &[f64],
        rays1: &[f64],
        origins2: &[f64],
        rays2: &[f64],
        cam_qvecs1: &[f64],
        cam_tvecs1: &[f64],
        cam_qvecs2: &[f64],
        cam_tvecs2: &[f64],
        random_seed: u32,
    ) -> Result<Vec<SE3>, GeneralizedPoseError> {
        if origins1.len() % 3 != 0 {
            return Err(GeneralizedPoseError::InvalidInput(
                "generalized relative pose origin data is malformed",
            ));
        }
        let num_points = origins1.len() / 3;
        if num_points < 8 {
            return Ok(Vec::new());
        }
        if origins2.len() != origins1.len()
            || rays1.len() != origins1.len()
            || rays2.len() != origins1.len()
            || cam_tvecs1.len() != origins1.len()
            || cam_tvecs2.len() != origins1.len()
            || cam_qvecs1.len() != 4 * num_points
            || cam_qvecs2.len() != 4 * num_points
        {
            return Err(GeneralizedPoseError::InvalidInput(
                "generalized relative pose observation data is malformed",
            ));
        }

        let mut output = [RustSfmPoseLibPose {
            qvec: [0.0; 4],
            tvec: [0.0; 3],
        }; MAX_POSELIB_POSES];
        let mut num_output = 0usize;
        let status = unsafe {
            rustsfm_poselib_gen_relpose_8pt(
                origins1.as_ptr(),
                rays1.as_ptr(),
                origins2.as_ptr(),
                rays2.as_ptr(),
                cam_qvecs1.as_ptr(),
                cam_tvecs1.as_ptr(),
                cam_qvecs2.as_ptr(),
                cam_tvecs2.as_ptr(),
                num_points,
                random_seed,
                output.as_mut_ptr(),
                output.len(),
                &mut num_output,
            )
        };
        if status != 0 {
            return Err(GeneralizedPoseError::SolverFailed(
                "PoseLib gen_relpose_8pt failed",
            ));
        }
        Ok(output
            .iter()
            .take(num_output.min(output.len()))
            .filter_map(|pose| pose_from_poselib(*pose))
            .collect())
    }

    pub(super) fn gp3p(
        origins: &[f64; 9],
        rays: &[f64; 9],
        points3d: &[f64; 9],
    ) -> Result<Vec<SE3>, GeneralizedPoseError> {
        let mut output = [RustSfmPoseLibPose {
            qvec: [0.0; 4],
            tvec: [0.0; 3],
        }; MAX_POSELIB_POSES];
        let mut num_output = 0usize;
        let status = unsafe {
            rustsfm_poselib_gp3p(
                origins.as_ptr(),
                rays.as_ptr(),
                points3d.as_ptr(),
                3,
                output.as_mut_ptr(),
                output.len(),
                &mut num_output,
            )
        };
        if status != 0 {
            return Err(GeneralizedPoseError::SolverFailed("PoseLib gp3p failed"));
        }
        Ok(output
            .iter()
            .take(num_output.min(output.len()))
            .filter_map(|pose| pose_from_poselib(*pose))
            .collect())
    }

    fn pose_from_poselib(pose: RustSfmPoseLibPose) -> Option<SE3> {
        let rigid = Rigid3 {
            qvec: pose.qvec,
            tvec: pose.tvec,
        };
        let pose = rigid.to_se3();
        let translation = pose.translation();
        let quaternion = pose.quaternion();
        translation
            .iter()
            .chain(quaternion.iter())
            .all(|value| value.is_finite())
            .then_some(pose)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structureless_options_match_colmap_defaults() {
        let options = StructureLessAbsolutePoseEstimationOptions::default();

        assert_eq!(options.ransac_options.max_error, 6.0);
        assert_eq!(options.ransac_options.min_num_trials, 100);
        assert_eq!(options.ransac_options.max_num_trials, 10000);
        assert_eq!(options.ransac_options.confidence, 0.99999);
    }

    #[test]
    fn generalized_absolute_options_match_colmap_defaults() {
        let options = GeneralizedAbsolutePoseEstimationOptions::default();

        assert_eq!(options.ransac_options.max_error, 12.0);
        assert_eq!(options.ransac_options.min_num_trials, 100);
        assert_eq!(options.ransac_options.max_num_trials, 10000);
        assert_eq!(options.ransac_options.confidence, 0.99999);
    }

    #[test]
    fn generalized_relative_options_match_colmap_initial_two_view_defaults() {
        let options = GeneralizedRelativePoseEstimationOptions::default();

        assert_eq!(options.ransac_options.max_error, 4.0);
        assert_eq!(options.ransac_options.min_num_trials, 30);
        assert_eq!(options.ransac_options.random_seed, -1);
    }

    #[test]
    fn generalized_pose_sampler_seed_honors_colmap_signed_seed() {
        assert_eq!(generalized_pose_sampler_seed(42), 42);
        assert_eq!(generalized_pose_sampler_seed(42), 42);
    }

    #[test]
    fn generalized_pose_sampler_seed_changes_for_colmap_default_seed() {
        let first = generalized_pose_sampler_seed(-1);
        let second = generalized_pose_sampler_seed(-1);

        assert_ne!(first, second);
    }

    #[test]
    fn structureless_preparation_matches_colmap_ray_and_threshold_conversion() {
        let world_camera = CameraModel::new_pinhole(200, 100, 100.0, 100.0, 100.0, 50.0);
        let query_camera = CameraModel::new_pinhole(200, 100, 50.0, 50.0, 100.0, 50.0);
        let world_points2d = [[100.0, 50.0], [110.0, 50.0]];
        let query_points2d = [[100.0, 50.0], [100.0, 60.0]];
        let world_camera_idxs = [0usize, 1usize];
        let world_cams_from_world = [
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(-1.0, 0.0, 0.0)),
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(1.0, 0.0, 0.0)),
        ];
        let world_cameras = [world_camera, world_camera];
        let mut options = StructureLessAbsolutePoseEstimationOptions::default();
        options.ransac_options.max_error = 4.0;

        let prepared = prepare_structureless_absolute_pose_observations(
            &options,
            StructureLessAbsolutePoseProblem {
                query_points2d: &query_points2d,
                world_points2d: &world_points2d,
                world_camera_idxs: &world_camera_idxs,
                world_cams_from_world: &world_cams_from_world,
                world_cameras: &world_cameras,
                query_camera,
            },
        )
        .unwrap()
        .unwrap();

        assert_eq!(prepared.world_obs.len(), 2);
        assert_eq!(prepared.query_obs.len(), 2);
        assert!((prepared.normalized_max_error - 0.08).abs() < 1.0e-12);
        assert_eq!(prepared.world_obs[0].ray_in_cam, [0.0, 0.0, 1.0]);
        assert!(prepared.query_obs[1].ray_in_cam[1] > 0.0);
    }

    #[test]
    fn generalized_absolute_preparation_matches_colmap_threshold_and_unique_ids() {
        let cam_a = CameraModel::new_pinhole(200, 100, 100.0, 100.0, 100.0, 50.0);
        let cam_b = CameraModel::new_pinhole(200, 100, 50.0, 50.0, 100.0, 50.0);
        let cameras = [cam_a, cam_b];
        let cams_from_rig = [
            SE3::identity(),
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(0.2, 0.0, 0.0)),
        ];
        let points2d = [[100.0, 50.0], [110.0, 50.0], [100.0, 60.0]];
        let points3d = [[0.0, 0.0, 4.0], [1.0, 0.0, 4.0], [1.0 + 5.0e-6, 0.0, 4.0]];
        let camera_idxs = [0usize, 1, 1];
        let mut options = GeneralizedAbsolutePoseEstimationOptions::default();
        options.ransac_options.max_error = 4.0;

        let prepared = prepare_generalized_absolute_pose_observations(
            &options,
            GeneralizedAbsolutePoseProblem {
                points2d: &points2d,
                points3d: &points3d,
                camera_idxs: &camera_idxs,
                cams_from_rig: &cams_from_rig,
                cameras: &cameras,
            },
        )
        .unwrap()
        .unwrap();

        assert_eq!(prepared.observations.len(), 3);
        assert!((prepared.normalized_max_error - ((0.04 + 0.08 + 0.08) / 3.0)).abs() < 1.0e-12);
        assert_eq!(
            prepared.unique_point3d_ids[1],
            prepared.unique_point3d_ids[2]
        );
        assert_ne!(
            prepared.unique_point3d_ids[0],
            prepared.unique_point3d_ids[1]
        );
    }

    #[test]
    fn structureless_preparation_rejects_panoramic_world_rig() {
        let camera = CameraModel::new_pinhole(200, 100, 100.0, 100.0, 100.0, 50.0);
        let points2d = [[100.0, 50.0], [110.0, 50.0]];
        let camera_idxs = [0usize, 1usize];
        let cams_from_world = [SE3::identity(), SE3::identity()];
        let cameras = [camera, camera];

        let prepared = prepare_structureless_absolute_pose_observations(
            &StructureLessAbsolutePoseEstimationOptions::default(),
            StructureLessAbsolutePoseProblem {
                query_points2d: &points2d,
                world_points2d: &points2d,
                world_camera_idxs: &camera_idxs,
                world_cams_from_world: &cams_from_world,
                world_cameras: &cameras,
                query_camera: camera,
            },
        )
        .unwrap();

        assert!(prepared.is_none());
    }

    #[test]
    fn generalized_relative_preparation_matches_colmap_threshold_and_rays() {
        let cam_a = CameraModel::new_pinhole(200, 100, 100.0, 100.0, 100.0, 50.0);
        let cam_b = CameraModel::new_pinhole(200, 100, 50.0, 50.0, 100.0, 50.0);
        let cams_from_rig = [
            SE3::identity(),
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(0.2, 0.0, 0.0)),
        ];
        let points1 = [[100.0, 50.0], [110.0, 50.0]];
        let points2 = [[100.0, 60.0], [100.0, 50.0]];
        let camera_idxs1 = [0usize, 1usize];
        let camera_idxs2 = [1usize, 0usize];
        let cameras = [cam_a, cam_b];
        let mut options = GeneralizedRelativePoseEstimationOptions::default();
        options.ransac_options.max_error = 4.0;

        let prepared = prepare_generalized_relative_pose_observations(
            &options,
            GeneralizedRelativePoseProblem {
                points2d1: &points1,
                points2d2: &points2,
                camera_idxs1: &camera_idxs1,
                camera_idxs2: &camera_idxs2,
                cams_from_rig: &cams_from_rig,
                cameras: &cameras,
            },
        )
        .unwrap()
        .unwrap();

        assert_eq!(prepared.observations1.len(), 2);
        assert_eq!(prepared.observations2.len(), 2);
        assert!(!prepared.both_rigs_panoramic);
        assert!((prepared.normalized_max_error - 0.06).abs() < 1.0e-12);
        assert_eq!(prepared.observations1[0].ray_in_cam, [0.0, 0.0, 1.0]);
        assert!(prepared.observations2[0].ray_in_cam[1] > 0.0);
    }

    #[test]
    fn generalized_relative_threshold_averages_all_rig_cameras_like_colmap() {
        let cam_a = CameraModel::new_pinhole(200, 100, 100.0, 100.0, 100.0, 50.0);
        let cam_b = CameraModel::new_pinhole(200, 100, 50.0, 50.0, 100.0, 50.0);
        let cam_c = CameraModel::new_pinhole(200, 100, 25.0, 25.0, 100.0, 50.0);
        let cameras = [cam_a, cam_b, cam_c];
        let cams_from_rig = [
            SE3::identity(),
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(0.2, 0.0, 0.0)),
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(-0.2, 0.0, 0.0)),
        ];
        let points = [[100.0, 50.0], [110.0, 50.0]];
        let camera_idxs1 = [0usize, 0usize];
        let camera_idxs2 = [0usize, 0usize];
        let mut options = GeneralizedRelativePoseEstimationOptions::default();
        options.ransac_options.max_error = 4.0;

        let prepared = prepare_generalized_relative_pose_observations(
            &options,
            GeneralizedRelativePoseProblem {
                points2d1: &points,
                points2d2: &points,
                camera_idxs1: &camera_idxs1,
                camera_idxs2: &camera_idxs2,
                cams_from_rig: &cams_from_rig,
                cameras: &cameras,
            },
        )
        .unwrap()
        .unwrap();

        assert!((prepared.normalized_max_error - ((0.04 + 0.08 + 0.16) / 3.0)).abs() < 1.0e-12);
    }

    #[test]
    fn generalized_relative_preparation_rotates_panoramic_rays_into_rig() {
        let camera = CameraModel::new_pinhole(200, 100, 100.0, 100.0, 100.0, 50.0);
        let rotation = glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let cams_from_rig = [
            SE3::identity(),
            SE3::from_quat_translation(rotation, glam::Vec3::ZERO),
        ];
        let points = [[100.0, 50.0], [100.0, 50.0]];
        let camera_idxs = [0usize, 1usize];
        let cameras = [camera, camera];

        let prepared = prepare_generalized_relative_pose_observations(
            &GeneralizedRelativePoseEstimationOptions::default(),
            GeneralizedRelativePoseProblem {
                points2d1: &points,
                points2d2: &points,
                camera_idxs1: &camera_idxs,
                camera_idxs2: &camera_idxs,
                cams_from_rig: &cams_from_rig,
                cameras: &cameras,
            },
        )
        .unwrap()
        .unwrap();

        assert!(prepared.both_rigs_panoramic);
        assert_eq!(prepared.observations1[0].ray_in_cam, [0.0, 0.0, 1.0]);
        assert!((prepared.observations1[1].ray_in_cam[0] + 1.0).abs() < 1.0e-6);
        assert!(prepared.observations1[1].ray_in_cam[2].abs() < 1.0e-6);
    }

    #[test]
    fn panoramic_generalized_relative_pose_uses_relative_pose_branch() {
        let camera = CameraModel::new_pinhole(640, 480, 500.0, 500.0, 320.0, 240.0);
        let cams_from_rig = [
            SE3::identity(),
            SE3::from_quat_translation(glam::Quat::from_rotation_y(0.4), glam::Vec3::ZERO),
        ];
        let pano2_from_pano1 = SE3::from_quat_translation(
            glam::Quat::from_rotation_y(0.08) * glam::Quat::from_rotation_x(-0.03),
            glam::Vec3::new(0.45, -0.04, 0.18).normalize(),
        );
        let points_in_pano1 = [
            [-0.7, -0.3, 5.0],
            [-0.3, 0.2, 4.5],
            [0.1, -0.4, 5.5],
            [0.4, 0.3, 6.0],
            [0.8, -0.1, 5.8],
            [-0.9, 0.4, 6.3],
            [0.2, 0.6, 7.0],
            [-0.5, -0.7, 6.8],
            [1.0, 0.5, 7.4],
            [-1.1, 0.0, 5.9],
            [0.6, -0.6, 6.5],
            [-0.2, 0.8, 7.2],
            [1.2, -0.2, 8.0],
            [-1.3, -0.4, 7.7],
            [0.3, 1.0, 8.4],
            [-0.8, 0.9, 7.9],
        ];
        let camera_idxs1 = [0usize, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
        let camera_idxs2 = [1usize, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let mut points2d1 = Vec::new();
        let mut points2d2 = Vec::new();
        for idx in 0..points_in_pano1.len() {
            let p1 = points_in_pano1[idx];
            let p2 = pano2_from_pano1.transform_point(&p1);
            points2d1.push(project_rig_point(
                camera,
                cams_from_rig[camera_idxs1[idx]],
                p1,
            ));
            points2d2.push(project_rig_point(
                camera,
                cams_from_rig[camera_idxs2[idx]],
                p2,
            ));
        }

        let mut options = GeneralizedRelativePoseEstimationOptions::default();
        options.ransac_options.max_error = 1.0;
        options.ransac_options.min_num_trials = 1;
        options.ransac_options.max_num_trials = 512;
        options.ransac_options.random_seed = 9;
        let estimate = estimate_generalized_relative_pose(
            &options,
            GeneralizedRelativePoseProblem {
                points2d1: &points2d1,
                points2d2: &points2d2,
                camera_idxs1: &camera_idxs1,
                camera_idxs2: &camera_idxs2,
                cams_from_rig: &cams_from_rig,
                cameras: &[camera, camera],
            },
        )
        .unwrap()
        .unwrap();

        assert!(estimate.rig2_from_rig1.is_none());
        assert_eq!(estimate.num_inliers, points_in_pano1.len());
        assert!(estimate.inlier_mask.iter().all(|&is_inlier| is_inlier));
        assert_pose_close(estimate.pano2_from_pano1.unwrap(), pano2_from_pano1, 5.0e-2);
    }

    #[cfg(not(feature = "poselib"))]
    #[test]
    fn generalized_relative_estimator_reports_missing_solver_without_poselib() {
        let camera = CameraModel::new_pinhole(200, 100, 100.0, 100.0, 100.0, 50.0);
        let points = [
            [100.0, 50.0],
            [110.0, 50.0],
            [100.0, 60.0],
            [90.0, 55.0],
            [105.0, 45.0],
            [95.0, 65.0],
        ];
        let camera_idxs1 = [0usize, 1, 0, 1, 0, 1];
        let camera_idxs2 = [1usize, 0, 1, 0, 1, 0];
        let cameras = [camera, camera];
        let cams_from_rig = [
            SE3::identity(),
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(0.3, 0.0, 0.0)),
        ];

        let err = estimate_generalized_relative_pose(
            &GeneralizedRelativePoseEstimationOptions::default(),
            GeneralizedRelativePoseProblem {
                points2d1: &points,
                points2d2: &points,
                camera_idxs1: &camera_idxs1,
                camera_idxs2: &camera_idxs2,
                cams_from_rig: &cams_from_rig,
                cameras: &cameras,
            },
        )
        .unwrap_err();

        assert_eq!(
            err,
            GeneralizedPoseError::MissingGeneralizedRelativePoseSolver
        );
    }

    #[cfg(feature = "poselib")]
    #[test]
    fn poselib_generalized_relative_pose_estimates_non_panoramic_rig() {
        let scene = synthetic_generalized_relative_scene();
        let mut options = GeneralizedRelativePoseEstimationOptions::default();
        options.ransac_options.max_error = 1.0;
        options.ransac_options.min_num_trials = 1;
        options.ransac_options.max_num_trials = 128;
        options.ransac_options.random_seed = 7;
        let estimate = estimate_generalized_relative_pose(
            &options,
            GeneralizedRelativePoseProblem {
                points2d1: &scene.points2d1,
                points2d2: &scene.points2d2,
                camera_idxs1: &scene.camera_idxs1,
                camera_idxs2: &scene.camera_idxs2,
                cams_from_rig: &scene.cams_from_rig,
                cameras: &scene.cameras,
            },
        )
        .unwrap()
        .unwrap();

        assert_eq!(estimate.num_inliers, scene.num_points);
        assert!(estimate.inlier_mask.iter().all(|&is_inlier| is_inlier));
        assert!(estimate.pano2_from_pano1.is_none());
        assert_pose_close(
            estimate.rig2_from_rig1.unwrap(),
            scene.rig2_from_rig1,
            2.0e-2,
        );
    }

    #[cfg(feature = "poselib")]
    #[test]
    fn poselib_gr8p_local_refit_estimates_non_panoramic_rig_from_inliers() {
        let scene = synthetic_generalized_relative_scene();
        let mut options = GeneralizedRelativePoseEstimationOptions::default();
        options.ransac_options.max_error = 1.0;
        let observations = prepare_generalized_relative_pose_observations(
            &options,
            GeneralizedRelativePoseProblem {
                points2d1: &scene.points2d1,
                points2d2: &scene.points2d2,
                camera_idxs1: &scene.camera_idxs1,
                camera_idxs2: &scene.camera_idxs2,
                cams_from_rig: &scene.cams_from_rig,
                cameras: &scene.cameras,
            },
        )
        .unwrap()
        .unwrap();
        let inlier_indices = (0..scene.num_points).collect::<Vec<_>>();
        let poses = poselib_gr8p_estimate(
            &observations.observations1,
            &observations.observations2,
            &inlier_indices,
            11,
        )
        .unwrap();

        assert!(!poses.is_empty());
        let threshold_sq = observations.normalized_max_error * observations.normalized_max_error;
        let mut best = None;
        for pose in poses {
            let candidate = score_generalized_relative_pose(
                pose,
                &observations.observations1,
                &observations.observations2,
                threshold_sq,
            );
            if best
                .as_ref()
                .is_none_or(|current| candidate.is_better_than(current))
            {
                best = Some(candidate);
            }
        }
        let best = best.unwrap();
        assert_eq!(best.num_inliers, scene.num_points);
        assert_pose_close(best.pose, scene.rig2_from_rig1, 5.0e-2);
    }

    #[cfg(feature = "poselib")]
    #[test]
    fn generalized_relative_support_residual_sum_ignores_outliers_like_colmap() {
        let rig2_from_rig1 =
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(1.0, 0.0, 0.0));
        let observations1 = [
            GRNPObservation {
                cam_from_rig: SE3::identity(),
                ray_in_cam: [0.0, 0.0, 1.0],
            },
            GRNPObservation {
                cam_from_rig: SE3::identity(),
                ray_in_cam: [0.0, 0.0, 1.0],
            },
        ];
        let observations2 = [
            GRNPObservation {
                cam_from_rig: SE3::identity(),
                ray_in_cam: [0.0, 0.0, 1.0],
            },
            GRNPObservation {
                cam_from_rig: SE3::identity(),
                ray_in_cam: [0.0, 1.0, 1.0],
            },
        ];

        let support =
            score_generalized_relative_pose(rig2_from_rig1, &observations1, &observations2, 1.0e-4);

        assert_eq!(support.num_inliers, 1);
        assert_eq!(support.inlier_mask, vec![true, false]);
        assert!(support.residual_sum.abs() < 1.0e-12);
    }

    #[cfg(feature = "poselib")]
    #[test]
    fn generalized_absolute_success_gate_uses_total_inliers_like_colmap() {
        let support = GeneralizedAbsoluteModelSupport {
            pose: SE3::identity(),
            num_unique_inliers: 2,
            num_inliers: 3,
            residual_sum: 0.0,
            inlier_mask: vec![true, true, true],
        };
        let stronger_unique_support = GeneralizedAbsoluteModelSupport {
            pose: SE3::identity(),
            num_unique_inliers: 3,
            num_inliers: 3,
            residual_sum: 1.0,
            inlier_mask: vec![true, true, true],
        };

        assert!(generalized_absolute_support_passes_colmap_success_gate(
            &support, 3
        ));
        assert!(stronger_unique_support.is_better_than(&support));
    }

    #[cfg(feature = "poselib")]
    #[test]
    fn generalized_absolute_report_num_inliers_uses_unique_inliers_like_colmap() {
        let support = GeneralizedAbsoluteModelSupport {
            pose: SE3::identity(),
            num_unique_inliers: 2,
            num_inliers: 5,
            residual_sum: 0.0,
            inlier_mask: vec![true; 5],
        };

        assert_eq!(generalized_absolute_colmap_report_num_inliers(&support), 2);
    }

    #[cfg(feature = "poselib")]
    #[test]
    fn generalized_ransac_dynamic_abort_uses_colmap_zero_based_trial_gate() {
        assert!(!colmap_ransac_abort_after_trial(0, 1, 0));
        assert!(colmap_ransac_abort_after_trial(1, 1, 0));

        assert!(!colmap_ransac_abort_after_trial(99, 1, 100));
        assert!(colmap_ransac_abort_after_trial(100, 1, 100));
    }

    #[cfg(feature = "poselib")]
    #[test]
    fn poselib_generalized_absolute_pose_estimates_non_panoramic_rig() {
        let camera = CameraModel::new_pinhole(640, 480, 500.0, 500.0, 320.0, 240.0);
        let cams_from_rig = [
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(0.25, 0.0, 0.0)),
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(-0.15, 0.22, 0.0)),
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(0.05, -0.18, 0.12)),
        ];
        let rig_from_world = SE3::from_quat_translation(
            glam::Quat::from_rotation_y(0.08) * glam::Quat::from_rotation_x(-0.03),
            glam::Vec3::new(0.45, -0.04, 0.18),
        );
        let points3d = [
            [-0.7, -0.3, 5.0],
            [-0.3, 0.2, 4.5],
            [0.1, -0.4, 5.5],
            [0.4, 0.3, 6.0],
            [0.8, -0.1, 5.8],
            [-0.9, 0.4, 6.3],
            [0.2, 0.6, 7.0],
            [-0.5, -0.7, 6.8],
            [1.0, 0.5, 7.4],
            [-1.1, 0.0, 5.9],
            [0.6, -0.6, 6.5],
            [-0.2, 0.8, 7.2],
            [1.2, -0.2, 8.0],
            [-1.3, -0.4, 7.7],
            [0.3, 1.0, 8.4],
            [-0.8, 0.9, 7.9],
            [0.9, -0.9, 8.2],
            [-0.1, -1.0, 7.5],
        ];
        let camera_idxs = vec![0usize, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
        let mut points2d = Vec::new();
        for (idx, point) in points3d.iter().enumerate() {
            let point_f32 = [point[0] as f32, point[1] as f32, point[2] as f32];
            let point_in_rig = rig_from_world.transform_point(&point_f32);
            points2d.push(project_rig_point(
                camera,
                cams_from_rig[camera_idxs[idx]],
                point_in_rig,
            ));
        }

        let mut options = GeneralizedAbsolutePoseEstimationOptions::default();
        options.ransac_options.max_error = 1.0;
        options.ransac_options.min_num_trials = 1;
        options.ransac_options.max_num_trials = 256;
        options.ransac_options.random_seed = 17;
        let estimate = estimate_generalized_absolute_pose(
            &options,
            GeneralizedAbsolutePoseProblem {
                points2d: &points2d,
                points3d: &points3d,
                camera_idxs: &camera_idxs,
                cams_from_rig: &cams_from_rig,
                cameras: &[camera, camera, camera],
            },
        )
        .unwrap()
        .unwrap();

        assert_eq!(estimate.num_unique_inliers, points3d.len());
        assert_eq!(estimate.num_inliers, points3d.len());
        assert!(estimate.inlier_mask.iter().all(|&is_inlier| is_inlier));
        assert_pose_close(estimate.rig_from_world, rig_from_world, 3.0e-2);
    }

    #[cfg(feature = "poselib")]
    #[test]
    fn poselib_structureless_absolute_pose_reuses_generalized_relative_solver() {
        let camera = CameraModel::new_pinhole(640, 480, 500.0, 500.0, 320.0, 240.0);
        let world_cams_from_world = [
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(0.25, 0.0, 0.0)),
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(-0.15, 0.22, 0.0)),
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(0.05, -0.18, 0.12)),
        ];
        let query_cam_from_world = SE3::from_quat_translation(
            glam::Quat::from_rotation_y(0.08) * glam::Quat::from_rotation_x(-0.03),
            glam::Vec3::new(0.45, -0.04, 0.18),
        );
        let points_in_world = [
            [-0.7, -0.3, 5.0],
            [-0.3, 0.2, 4.5],
            [0.1, -0.4, 5.5],
            [0.4, 0.3, 6.0],
            [0.8, -0.1, 5.8],
            [-0.9, 0.4, 6.3],
            [0.2, 0.6, 7.0],
            [-0.5, -0.7, 6.8],
            [1.0, 0.5, 7.4],
            [-1.1, 0.0, 5.9],
            [0.6, -0.6, 6.5],
            [-0.2, 0.8, 7.2],
            [1.2, -0.2, 8.0],
            [-1.3, -0.4, 7.7],
            [0.3, 1.0, 8.4],
            [-0.8, 0.9, 7.9],
            [0.9, -0.9, 8.2],
            [-0.1, -1.0, 7.5],
        ];
        let world_camera_idxs = vec![0usize, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
        let mut world_points2d = Vec::new();
        let mut query_points2d = Vec::new();
        for (idx, point) in points_in_world.iter().enumerate() {
            world_points2d.push(project_rig_point(
                camera,
                world_cams_from_world[world_camera_idxs[idx]],
                *point,
            ));
            query_points2d.push(project_rig_point(camera, query_cam_from_world, *point));
        }

        let mut options = StructureLessAbsolutePoseEstimationOptions::default();
        options.ransac_options.max_error = 1.0;
        options.ransac_options.min_num_trials = 1;
        options.ransac_options.max_num_trials = 128;
        options.ransac_options.random_seed = 13;
        let estimate = estimate_structureless_absolute_pose(
            &options,
            StructureLessAbsolutePoseProblem {
                query_points2d: &query_points2d,
                world_points2d: &world_points2d,
                world_camera_idxs: &world_camera_idxs,
                world_cams_from_world: &world_cams_from_world,
                world_cameras: &[camera, camera, camera],
                query_camera: camera,
            },
        )
        .unwrap()
        .unwrap();

        assert_eq!(estimate.num_inliers, points_in_world.len());
        assert!(estimate.inlier_mask.iter().all(|&is_inlier| is_inlier));
        assert_pose_close(estimate.query_cam_from_world, query_cam_from_world, 3.0e-2);
    }

    #[cfg(feature = "poselib")]
    #[test]
    fn generalized_loransac_trial_count_matches_colmap_without_replacement_formula() {
        let trials = ransac_trials_from_counts(35, 100, 6, 0.99, 3.0);
        let mut prob_inlier = 1.0;
        for idx in 0..6 {
            prob_inlier *= (35 - idx) as f64 / (100 - idx) as f64;
        }
        let expected = ((1.0f64 - 0.99).ln() / (1.0f64 - prob_inlier).ln() * 3.0).ceil() as usize;

        assert_eq!(trials, expected);
        assert_ne!(
            trials,
            ((1.0f64 - 0.99).ln() / (1.0f64 - 0.35f64.powi(6)).ln() * 3.0).ceil() as usize
        );
    }

    #[cfg(feature = "poselib")]
    struct SyntheticGeneralizedRelativeScene {
        cameras: [CameraModel; 3],
        cams_from_rig: [SE3; 3],
        rig2_from_rig1: SE3,
        points2d1: Vec<[f64; 2]>,
        points2d2: Vec<[f64; 2]>,
        camera_idxs1: Vec<usize>,
        camera_idxs2: Vec<usize>,
        num_points: usize,
    }

    #[cfg(feature = "poselib")]
    fn synthetic_generalized_relative_scene() -> SyntheticGeneralizedRelativeScene {
        let camera = CameraModel::new_pinhole(640, 480, 500.0, 500.0, 320.0, 240.0);
        let cams_from_rig = [
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(0.25, 0.0, 0.0)),
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(-0.15, 0.22, 0.0)),
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(0.05, -0.18, 0.12)),
        ];
        let rig2_from_rig1 = SE3::from_quat_translation(
            glam::Quat::from_rotation_y(0.08) * glam::Quat::from_rotation_x(-0.03),
            glam::Vec3::new(0.45, -0.04, 0.18),
        );
        let points_in_rig1 = [
            [-0.7, -0.3, 5.0],
            [-0.3, 0.2, 4.5],
            [0.1, -0.4, 5.5],
            [0.4, 0.3, 6.0],
            [0.8, -0.1, 5.8],
            [-0.9, 0.4, 6.3],
            [0.2, 0.6, 7.0],
            [-0.5, -0.7, 6.8],
            [1.0, 0.5, 7.4],
            [-1.1, 0.0, 5.9],
            [0.6, -0.6, 6.5],
            [-0.2, 0.8, 7.2],
            [1.2, -0.2, 8.0],
            [-1.3, -0.4, 7.7],
            [0.3, 1.0, 8.4],
            [-0.8, 0.9, 7.9],
            [0.9, -0.9, 8.2],
            [-0.1, -1.0, 7.5],
        ];
        let camera_idxs1 = vec![0usize, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
        let camera_idxs2 = vec![1usize, 2, 0, 2, 0, 1, 1, 2, 0, 2, 0, 1, 1, 2, 0, 2, 0, 1];
        let mut points2d1 = Vec::new();
        let mut points2d2 = Vec::new();
        for idx in 0..points_in_rig1.len() {
            let p1 = points_in_rig1[idx];
            let p2 = rig2_from_rig1.transform_point(&p1);
            points2d1.push(project_rig_point(
                camera,
                cams_from_rig[camera_idxs1[idx]],
                p1,
            ));
            points2d2.push(project_rig_point(
                camera,
                cams_from_rig[camera_idxs2[idx]],
                p2,
            ));
        }

        SyntheticGeneralizedRelativeScene {
            cameras: [camera, camera, camera],
            cams_from_rig,
            rig2_from_rig1,
            points2d1,
            points2d2,
            camera_idxs1,
            camera_idxs2,
            num_points: points_in_rig1.len(),
        }
    }

    #[test]
    fn recomposes_original_camera_pose_from_generalized_rig_pose() {
        let orig_cam1_from_rig1 = SE3::from_quat_translation(
            glam::Quat::from_rotation_y(0.1),
            glam::Vec3::new(0.3, 0.0, 0.0),
        );
        let orig_cam2_from_rig2 = SE3::from_quat_translation(
            glam::Quat::from_rotation_x(-0.2),
            glam::Vec3::new(-0.1, 0.2, 0.0),
        );
        let rig2_from_rig1 = SE3::from_quat_translation(
            glam::Quat::from_rotation_z(0.3),
            glam::Vec3::new(1.0, 0.1, 0.2),
        );

        let recomposed = original_camera_relative_pose_from_rig_relative_pose(
            rig2_from_rig1,
            orig_cam1_from_rig1,
            orig_cam2_from_rig2,
        );
        let expected = orig_cam2_from_rig2
            .compose(&rig2_from_rig1)
            .compose(&orig_cam1_from_rig1.inverse());

        assert_pose_close(recomposed, expected, 1.0e-6);
    }

    #[cfg(not(feature = "poselib"))]
    #[test]
    fn structureless_estimator_reports_missing_gr6p_gr8p_solver() {
        let camera = CameraModel::new_pinhole(200, 100, 100.0, 100.0, 100.0, 50.0);
        let points2d = [[100.0, 50.0], [110.0, 50.0]];
        let camera_idxs = [0usize, 1usize];
        let cams_from_world = [
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(-1.0, 0.0, 0.0)),
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(1.0, 0.0, 0.0)),
        ];
        let cameras = [camera, camera];

        let err = estimate_structureless_absolute_pose(
            &StructureLessAbsolutePoseEstimationOptions::default(),
            StructureLessAbsolutePoseProblem {
                query_points2d: &points2d,
                world_points2d: &points2d,
                world_camera_idxs: &camera_idxs,
                world_cams_from_world: &cams_from_world,
                world_cameras: &cameras,
                query_camera: camera,
            },
        )
        .unwrap_err();

        assert_eq!(
            err,
            GeneralizedPoseError::MissingGeneralizedRelativePoseSolver
        );
    }

    fn assert_pose_close(actual: SE3, expected: SE3, eps: f32) {
        let actual_t = actual.translation();
        let expected_t = expected.translation();
        for idx in 0..3 {
            assert!((actual_t[idx] - expected_t[idx]).abs() < eps);
        }
        let actual_q = actual.quaternion();
        let expected_q = expected.quaternion();
        let dot = (actual_q[0] * expected_q[0]
            + actual_q[1] * expected_q[1]
            + actual_q[2] * expected_q[2]
            + actual_q[3] * expected_q[3])
            .abs();
        assert!((1.0 - dot) < eps);
    }

    fn project_rig_point(
        camera: CameraModel,
        cam_from_rig: SE3,
        point_in_rig: [f32; 3],
    ) -> [f64; 2] {
        let point_in_cam = cam_from_rig.transform_point(&point_in_rig);
        camera
            .img_from_cam(
                point_in_cam[0] as f64,
                point_in_cam[1] as f64,
                point_in_cam[2] as f64,
            )
            .unwrap()
    }
}
