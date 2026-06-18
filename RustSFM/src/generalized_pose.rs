use crate::geometry::camera_center;
use crate::types::CameraModel;
use rustslam::SE3;
use std::collections::BTreeSet;
use std::fmt;

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
        Self {
            max_error: 0.0,
            min_inlier_ratio: 0.1,
            confidence: 0.99,
            dyn_num_trials_multiplier: 3.0,
            min_num_trials: 0,
            max_num_trials: usize::MAX,
            random_seed: -1,
            num_threads: 1,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeneralizedPoseError {
    InvalidOptions(&'static str),
    InvalidInput(&'static str),
    MissingGeneralizedRelativePoseSolver,
}

impl fmt::Display for GeneralizedPoseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(message) | Self::InvalidInput(message) => f.write_str(message),
            Self::MissingGeneralizedRelativePoseSolver => f.write_str(
                "COLMAP GR6P/GR8P generalized relative pose LORANSAC is not implemented yet",
            ),
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

pub fn estimate_structureless_absolute_pose(
    options: &StructureLessAbsolutePoseEstimationOptions,
    problem: StructureLessAbsolutePoseProblem<'_>,
) -> Result<Option<StructureLessAbsolutePoseEstimate>, GeneralizedPoseError> {
    let Some(_observations) = prepare_structureless_absolute_pose_observations(options, problem)?
    else {
        return Ok(None);
    };

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
}
