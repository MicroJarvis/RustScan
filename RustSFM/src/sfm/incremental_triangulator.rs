use crate::correspondence_graph::{
    build_correspondence_graph_from_pairs, CorrespondenceGraph, ImageId, Point2DIdx,
};
use crate::observation_manager::ObservationManager;
use crate::triangulation_estimator::{
    calculate_angular_reprojection_error, calculate_squared_reprojection_error,
    estimate_triangulation, EstimateTriangulationOptions, ResidualType,
};
use crate::types::{
    CameraModel, ImageFrame, PairGeometry, Point3D, Reconstruction, TrackObservation,
};
use nalgebra::{Matrix3x4, Vector2, Vector3};
use rustslam::SE3;
use std::collections::{HashMap, HashSet};

const EXHAUSTIVE_TRIANGULATION_SAMPLING_THRESHOLD: usize = 15;
const MIN_RECURSIVE_CREATE_TRACK_LENGTH: usize = 3;

#[derive(Debug, Clone, Copy)]
pub struct IncrementalTriangulatorOptions {
    pub max_transitivity: usize,
    pub create_max_angle_error_deg: f32,
    pub continue_max_angle_error_deg: f32,
    pub merge_max_reproj_error_px: f32,
    pub complete_max_reproj_error_px: f32,
    pub complete_max_transitivity: usize,
    pub re_max_angle_error_deg: f32,
    pub re_min_ratio: f32,
    pub re_max_trials: usize,
    pub min_angle_deg: f32,
    pub ignore_two_view_tracks: bool,
    pub min_focal_length_ratio: f64,
    pub max_focal_length_ratio: f64,
    pub max_extra_param: f64,
    pub random_seed: i32,
    pub num_threads: isize,
}

impl Default for IncrementalTriangulatorOptions {
    fn default() -> Self {
        Self {
            max_transitivity: 1,
            create_max_angle_error_deg: 2.0,
            continue_max_angle_error_deg: 2.0,
            merge_max_reproj_error_px: 4.0,
            complete_max_reproj_error_px: 4.0,
            complete_max_transitivity: 5,
            re_max_angle_error_deg: 5.0,
            re_min_ratio: 0.2,
            re_max_trials: 1,
            min_angle_deg: 1.5,
            ignore_two_view_tracks: true,
            min_focal_length_ratio: 0.1,
            max_focal_length_ratio: 10.0,
            max_extra_param: 1.0,
            random_seed: -1,
            num_threads: 1,
        }
    }
}

impl IncrementalTriangulatorOptions {
    pub fn from_mapper_threshold(max_reprojection_error_px: f32) -> Self {
        Self {
            merge_max_reproj_error_px: max_reprojection_error_px,
            complete_max_reproj_error_px: max_reprojection_error_px,
            ..Self::default()
        }
    }

    pub fn from_mapper_config_values(
        max_reprojection_error_px: f32,
        min_focal_length_ratio: f64,
        max_focal_length_ratio: f64,
        max_extra_param: f64,
        random_seed: i32,
        ignore_two_view_tracks: bool,
        num_threads: isize,
    ) -> Self {
        Self {
            merge_max_reproj_error_px: max_reprojection_error_px,
            complete_max_reproj_error_px: max_reprojection_error_px,
            ignore_two_view_tracks,
            min_focal_length_ratio,
            max_focal_length_ratio,
            max_extra_param,
            random_seed,
            num_threads,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TriangulationReport {
    pub created_points: usize,
    pub continued_observations: usize,
    pub completed_observations: usize,
}

impl TriangulationReport {
    pub fn total_observations(self) -> usize {
        self.created_points * 2 + self.continued_observations + self.completed_observations
    }
}

/// Session-scoped triangulation state kept alive for the full incremental mapping
/// attempt, matching COLMAP's long-lived `IncrementalTriangulator`.
pub struct IncrementalTriangulatorState {
    observation_manager: ObservationManager,
    merge_trials: HashSet<(usize, usize)>,
    retriangulation_trials: HashMap<(usize, usize), usize>,
    camera_has_bogus_params: HashMap<usize, bool>,
}

impl IncrementalTriangulatorState {
    pub fn new(
        frames: &[ImageFrame],
        pairs: &[PairGeometry],
        reconstruction: &Reconstruction,
    ) -> Self {
        Self {
            observation_manager: ObservationManager::new(frames, pairs, reconstruction),
            merge_trials: HashSet::new(),
            retriangulation_trials: HashMap::new(),
            camera_has_bogus_params: HashMap::new(),
        }
    }

    pub fn rebuild(
        &mut self,
        frames: &[ImageFrame],
        pairs: &[PairGeometry],
        reconstruction: &Reconstruction,
    ) {
        self.observation_manager
            .rebuild(frames, pairs, reconstruction);
        self.observation_manager
            .install_correspondence_graph(build_correspondence_graph_from_pairs(frames, pairs));
        self.merge_trials.clear();
        self.retriangulation_trials.clear();
        self.camera_has_bogus_params.clear();
    }

    pub fn sync_after_reconstruction_rollback(
        &mut self,
        frames: &[ImageFrame],
        pairs: &[PairGeometry],
        reconstruction: &Reconstruction,
    ) {
        self.observation_manager
            .rebuild(frames, pairs, reconstruction);
        self.observation_manager
            .install_correspondence_graph(build_correspondence_graph_from_pairs(frames, pairs));
        self.camera_has_bogus_params.clear();
    }

    pub fn correspondence_graph(&self) -> &CorrespondenceGraph {
        self.observation_manager
            .correspondence_graph()
            .expect("correspondence graph must be installed")
    }

    pub fn observation_manager(&self) -> &ObservationManager {
        &self.observation_manager
    }

    pub fn observation_manager_mut(&mut self) -> &mut ObservationManager {
        &mut self.observation_manager
    }

    pub fn retriangulation_trials(&self) -> &HashMap<(usize, usize), usize> {
        &self.retriangulation_trials
    }

    fn sync_merge_trials_after_point_merge(
        &mut self,
        keep_id: usize,
        remove_id: usize,
        moved_from: Option<usize>,
    ) {
        self.merge_trials = self
            .merge_trials
            .iter()
            .filter_map(|&(left, right)| {
                if left == keep_id || right == keep_id || left == remove_id || right == remove_id {
                    return None;
                }
                let remap = |point_id| {
                    if moved_from == Some(point_id) {
                        remove_id
                    } else {
                        point_id
                    }
                };
                let left = remap(left);
                let right = remap(right);
                (left != right).then_some(ordered_point_pair(left, right))
            })
            .collect();
    }
}

pub struct IncrementalTriangulator<'a> {
    frames: &'a [ImageFrame],
    pairs: &'a [PairGeometry],
    reconstruction: &'a mut Reconstruction,
    state: &'a mut IncrementalTriangulatorState,
}

impl<'a> IncrementalTriangulator<'a> {
    pub fn new(
        frames: &'a [ImageFrame],
        pairs: &'a [PairGeometry],
        reconstruction: &'a mut Reconstruction,
        state: &'a mut IncrementalTriangulatorState,
    ) -> Self {
        Self {
            frames,
            pairs,
            reconstruction,
            state,
        }
    }

    pub fn triangulate_image(
        &mut self,
        options: &IncrementalTriangulatorOptions,
        image: usize,
    ) -> TriangulationReport {
        if self
            .reconstruction
            .poses
            .get(image)
            .copied()
            .flatten()
            .is_none()
        {
            return TriangulationReport::default();
        }

        let camera_index = self
            .reconstruction
            .image_camera_indices
            .get(image)
            .copied()
            .unwrap_or(0);
        if self.has_camera_bogus_params(camera_index, options) {
            return TriangulationReport::default();
        }

        self.clear_triangulate_caches();

        let mut report = TriangulationReport::default();
        let num_features = self
            .frames
            .get(image)
            .map(|frame| frame.keypoints.len())
            .unwrap_or(0);
        let mut corrs_data = Vec::new();

        for feature in 0..num_features {
            if !self.valid_observation(image, feature) {
                continue;
            }

            let num_triangulated =
                self.find_correspondences(options, image, feature, &mut corrs_data);
            if corrs_data.is_empty() {
                continue;
            }

            if num_triangulated == 0 {
                corrs_data.push((image, feature));
                if self.create_from_correspondences(options, &corrs_data) > 0 {
                    report.created_points += 1;
                }
            } else {
                if self.continue_reference_observation(options, image, feature, &corrs_data) {
                    report.continued_observations += 1;
                }
                corrs_data.push((image, feature));
                if self.create_from_correspondences(options, &corrs_data) > 0 {
                    report.created_points += 1;
                }
            }
        }
        report
    }

    /// COLMAP `CompleteImage`: complete existing tracks on the image and create
    /// new points for untriangulated observations whose correspondences are all
    /// still untriangulated.
    pub fn complete_image(
        &mut self,
        options: &IncrementalTriangulatorOptions,
        image: usize,
    ) -> TriangulationReport {
        if self
            .reconstruction
            .poses
            .get(image)
            .copied()
            .flatten()
            .is_none()
        {
            return TriangulationReport::default();
        }

        let camera_index = self
            .reconstruction
            .image_camera_indices
            .get(image)
            .copied()
            .unwrap_or(0);
        if self.has_camera_bogus_params(camera_index, options) {
            return TriangulationReport::default();
        }

        self.clear_triangulate_caches();

        let mut report = TriangulationReport::default();
        let num_features = self
            .frames
            .get(image)
            .map(|frame| frame.keypoints.len())
            .unwrap_or(0);
        let mut corrs_data = Vec::new();
        let complete_est_options = complete_estimate_options(options);

        for feature in 0..num_features {
            if !self.valid_observation(image, feature) {
                continue;
            }

            if let Some(point_id) = self.reconstruction.observations[image][feature] {
                report.completed_observations += self.complete_track(options, point_id);
                continue;
            }

            if options.ignore_two_view_tracks
                && self
                    .state
                    .correspondence_graph()
                    .is_two_view_observation(image as ImageId, feature as Point2DIdx)
                    .unwrap_or(false)
            {
                continue;
            }

            let num_triangulated =
                self.find_correspondences(options, image, feature, &mut corrs_data);
            if num_triangulated > 0 || corrs_data.is_empty() {
                continue;
            }

            corrs_data.push((image, feature));
            let mut est_options = complete_est_options;
            if corrs_data.len() <= EXHAUSTIVE_TRIANGULATION_SAMPLING_THRESHOLD {
                est_options.min_num_trials = n_choose_k(corrs_data.len(), 2);
            }
            if let Some((inlier_mask, xyz)) =
                self.estimate_point_from_correspondences(&corrs_data, &est_options)
            {
                if self.add_point_from_inliers(&corrs_data, &inlier_mask, xyz) {
                    report.created_points += 1;
                }
            }
        }
        report
    }

    pub fn get_modified_points3d(&self) -> &HashSet<usize> {
        self.state.observation_manager().modified_point3d_ids()
    }

    pub fn complete_tracks(
        &mut self,
        options: &IncrementalTriangulatorOptions,
        point_ids: &HashSet<usize>,
    ) -> usize {
        self.clear_triangulate_caches();
        point_ids
            .iter()
            .copied()
            .map(|point_id| self.complete_track(options, point_id))
            .sum()
    }

    pub fn complete_all_tracks(&mut self, options: &IncrementalTriangulatorOptions) -> usize {
        let point_ids = (0..self.reconstruction.points.len()).collect::<HashSet<_>>();
        self.complete_tracks(options, &point_ids)
    }

    pub fn merge_tracks(
        &mut self,
        options: &IncrementalTriangulatorOptions,
        point_ids: &HashSet<usize>,
    ) -> usize {
        self.clear_triangulate_caches();
        let mut point_ids = point_ids.iter().copied().collect::<Vec<_>>();
        point_ids.sort_unstable_by(|a, b| b.cmp(a));
        point_ids
            .into_iter()
            .map(|point_id| self.merge_track(options, point_id))
            .sum()
    }

    pub fn merge_all_tracks(&mut self, options: &IncrementalTriangulatorOptions) -> usize {
        let point_ids = (0..self.reconstruction.points.len()).collect::<HashSet<_>>();
        self.merge_tracks(options, &point_ids)
    }

    pub fn retriangulate(&mut self, options: &IncrementalTriangulatorOptions) -> usize {
        self.clear_triangulate_caches();
        let re_options = IncrementalTriangulatorOptions {
            continue_max_angle_error_deg: options.re_max_angle_error_deg,
            ..*options
        };
        let mut total = 0usize;
        for pair in self.pairs {
            if self
                .reconstruction
                .poses
                .get(pair.left)
                .copied()
                .flatten()
                .is_none()
                || self
                    .reconstruction
                    .poses
                    .get(pair.right)
                    .copied()
                    .flatten()
                    .is_none()
            {
                continue;
            }
            let pair_key = ordered_point_pair(pair.left, pair.right);
            if options.re_max_trials == 0 {
                continue;
            }
            let trials = self
                .state
                .retriangulation_trials
                .get(&pair_key)
                .copied()
                .unwrap_or(0);
            if trials >= options.re_max_trials {
                continue;
            }
            let (tri_corrs, total_corrs) = self.pair_triangulation_counts(pair);
            if total_corrs == 0 || tri_corrs as f32 / total_corrs as f32 >= options.re_min_ratio {
                continue;
            }
            self.state
                .retriangulation_trials
                .insert(pair_key, trials + 1);
            for match_ in &pair.inlier_matches {
                let left_feature = match_.query_idx as usize;
                let right_feature = match_.train_idx as usize;
                if !self.valid_observation(pair.left, left_feature)
                    || !self.valid_observation(pair.right, right_feature)
                {
                    continue;
                }
                let left_point = self.reconstruction.observations[pair.left][left_feature];
                let right_point = self.reconstruction.observations[pair.right][right_feature];
                match (left_point, right_point) {
                    (Some(_), Some(_)) => {}
                    (Some(point_id), None) => {
                        if point_id < self.reconstruction.points.len()
                            && self.continue_reference_observation(
                                &re_options,
                                pair.right,
                                right_feature,
                                &[(pair.left, left_feature)],
                            )
                        {
                            total += 1;
                        }
                    }
                    (None, Some(point_id)) => {
                        if point_id < self.reconstruction.points.len()
                            && self.continue_reference_observation(
                                &re_options,
                                pair.left,
                                left_feature,
                                &[(pair.right, right_feature)],
                            )
                        {
                            total += 1;
                        }
                    }
                    (None, None) => {
                        let corrs = vec![(pair.left, left_feature), (pair.right, right_feature)];
                        total += self.create_from_correspondences(options, &corrs);
                    }
                }
            }
        }
        total
    }

    pub fn clear_modified_points3d(&mut self) {
        self.state
            .observation_manager_mut()
            .clear_modified_point3d_ids();
    }

    fn valid_observation(&self, image: usize, feature: usize) -> bool {
        self.frames
            .get(image)
            .map(|frame| feature < frame.keypoints.len())
            .unwrap_or(false)
            && self
                .reconstruction
                .observations
                .get(image)
                .map(|observations| feature < observations.len())
                .unwrap_or(false)
    }

    fn clear_triangulate_caches(&mut self) {
        self.state.merge_trials.clear();
        self.state.camera_has_bogus_params.clear();
    }

    fn has_camera_bogus_params(
        &mut self,
        camera_index: usize,
        options: &IncrementalTriangulatorOptions,
    ) -> bool {
        if let Some(&cached) = self.state.camera_has_bogus_params.get(&camera_index) {
            return cached;
        }
        let camera = self
            .reconstruction
            .cameras
            .get(camera_index)
            .copied()
            .unwrap_or(self.reconstruction.camera);
        let bogus = camera_has_bogus_params_for_triangulation(camera, options);
        self.state
            .camera_has_bogus_params
            .insert(camera_index, bogus);
        bogus
    }

    fn find_correspondences(
        &mut self,
        options: &IncrementalTriangulatorOptions,
        image: usize,
        feature: usize,
        corrs_data: &mut Vec<(usize, usize)>,
    ) -> usize {
        corrs_data.clear();
        let graph_corrs = self
            .state
            .correspondence_graph()
            .extract_transitive_correspondences(
                image as ImageId,
                feature as Point2DIdx,
                options.max_transitivity,
            )
            .unwrap_or_default();

        let mut num_triangulated = 0usize;
        for corr in graph_corrs {
            let corr_image = corr.image_id as usize;
            let corr_feature = corr.point2d_idx as usize;
            if self
                .reconstruction
                .poses
                .get(corr_image)
                .copied()
                .flatten()
                .is_none()
                || !self.valid_observation(corr_image, corr_feature)
            {
                continue;
            }
            let camera_index = self
                .reconstruction
                .image_camera_indices
                .get(corr_image)
                .copied()
                .unwrap_or(0);
            if self.has_camera_bogus_params(camera_index, options) {
                continue;
            }
            corrs_data.push((corr_image, corr_feature));
            if self.reconstruction.observations[corr_image][corr_feature].is_some() {
                num_triangulated += 1;
            }
        }
        num_triangulated
    }

    fn continue_reference_observation(
        &mut self,
        options: &IncrementalTriangulatorOptions,
        ref_image: usize,
        ref_feature: usize,
        corrs_data: &[(usize, usize)],
    ) -> bool {
        if self.reconstruction.observations[ref_image][ref_feature].is_some() {
            return false;
        }
        let Some(ref_pose) = self.reconstruction.poses[ref_image] else {
            return false;
        };
        let ref_camera = self.reconstruction.camera_for_image(ref_image);
        let ref_keypoint = &self.frames[ref_image].keypoints[ref_feature];
        let Some(ref_cam_ray) =
            normalized_camera_ray(ref_camera, ref_keypoint.x(), ref_keypoint.y())
        else {
            return false;
        };
        let ref_cam_from_world = se3_to_matrix3x4(ref_pose);

        let mut best_angle_error = f64::MAX;
        let mut best_point_id = None;
        for &(corr_image, corr_feature) in corrs_data {
            let Some(point_id) = self.reconstruction.observations[corr_image][corr_feature] else {
                continue;
            };
            let point_xyz = Vector3::from(self.reconstruction.points[point_id].xyz.map(f64::from));
            let angle_error =
                calculate_angular_reprojection_error(&ref_cam_ray, &point_xyz, &ref_cam_from_world);
            if angle_error < best_angle_error {
                best_angle_error = angle_error;
                best_point_id = Some(point_id);
            }
        }

        let max_angle_error = (options.continue_max_angle_error_deg as f64).to_radians();
        if best_angle_error > max_angle_error || best_point_id.is_none() {
            return false;
        }

        let point_id = best_point_id.expect("checked above");
        if !self.state.observation_manager_mut().add_observation(
            self.frames,
            self.pairs,
            self.reconstruction,
            point_id,
            TrackObservation {
                image: ref_image,
                feature: ref_feature,
            },
        ) {
            return false;
        }
        self.state
            .observation_manager_mut()
            .mark_point3d_modified(point_id);
        true
    }

    fn create_from_correspondences(
        &mut self,
        options: &IncrementalTriangulatorOptions,
        corrs_data: &[(usize, usize)],
    ) -> usize {
        let create_corrs: Vec<(usize, usize)> = corrs_data
            .iter()
            .copied()
            .filter(|&(image, feature)| self.reconstruction.observations[image][feature].is_none())
            .collect();
        if create_corrs.len() < 2 {
            return 0;
        }
        if options.ignore_two_view_tracks && create_corrs.len() == 2 {
            let (image, feature) = create_corrs[0];
            if self
                .state
                .correspondence_graph()
                .is_two_view_observation(image as ImageId, feature as Point2DIdx)
                .unwrap_or(false)
            {
                return 0;
            }
        }

        let mut est_options = create_estimate_options(options);
        if create_corrs.len() <= EXHAUSTIVE_TRIANGULATION_SAMPLING_THRESHOLD {
            est_options.min_num_trials = n_choose_k(create_corrs.len(), 2);
        }
        let Some((inlier_mask, xyz)) =
            self.estimate_point_from_correspondences(&create_corrs, &est_options)
        else {
            return 0;
        };

        let track_length = inlier_mask.iter().filter(|&&is_inlier| is_inlier).count();
        if track_length < 2 {
            return 0;
        }
        if !self.add_point_from_inliers(&create_corrs, &inlier_mask, xyz) {
            return 0;
        }

        if create_corrs.len().saturating_sub(track_length) >= MIN_RECURSIVE_CREATE_TRACK_LENGTH {
            track_length + self.create_from_correspondences(options, corrs_data)
        } else {
            track_length
        }
    }

    fn estimate_point_from_correspondences(
        &self,
        corrs: &[(usize, usize)],
        est_options: &EstimateTriangulationOptions,
    ) -> Option<(Vec<bool>, Vector3<f64>)> {
        let mut points = Vec::with_capacity(corrs.len());
        let mut cams_from_world = Vec::with_capacity(corrs.len());
        let mut cameras = Vec::with_capacity(corrs.len());
        for &(obs_image, obs_feature) in corrs {
            let pose = self.reconstruction.poses[obs_image]?;
            let kp = &self.frames[obs_image].keypoints[obs_feature];
            points.push(Vector2::new(kp.x() as f64, kp.y() as f64));
            cams_from_world.push(se3_to_matrix3x4(pose));
            cameras.push(self.reconstruction.camera_for_image(obs_image));
        }
        estimate_triangulation(est_options, &points, &cams_from_world, &cameras)
    }

    fn add_point_from_inliers(
        &mut self,
        corrs: &[(usize, usize)],
        inlier_mask: &[bool],
        xyz: Vector3<f64>,
    ) -> bool {
        let track: Vec<TrackObservation> = corrs
            .iter()
            .zip(inlier_mask.iter())
            .filter_map(|(&(obs_image, obs_feature), &is_inlier)| {
                is_inlier.then_some(TrackObservation {
                    image: obs_image,
                    feature: obs_feature,
                })
            })
            .collect();
        if track.len() < 2 {
            return false;
        }

        let xyz_f32 = [xyz[0] as f32, xyz[1] as f32, xyz[2] as f32];
        if !xyz_f32.iter().all(|v| v.is_finite()) {
            return false;
        }
        let error =
            mean_track_reprojection_error(xyz_f32, &track, self.frames, self.reconstruction)
                .unwrap_or(0.0);
        let point = Point3D {
            xyz: xyz_f32,
            color: [0, 0, 0],
            error,
            track,
        };
        self.state
            .observation_manager_mut()
            .add_point3d(self.frames, self.pairs, self.reconstruction, point)
            .is_some()
    }

    fn complete_track(
        &mut self,
        options: &IncrementalTriangulatorOptions,
        point_id: usize,
    ) -> usize {
        if point_id >= self.reconstruction.points.len() {
            return 0;
        }

        let max_squared_reproj_error = (options.complete_max_reproj_error_px as f64).powi(2);
        let point_xyz = Vector3::from(self.reconstruction.points[point_id].xyz.map(f64::from));

        let mut curr_queue = self.reconstruction.points[point_id]
            .track
            .iter()
            .map(|obs| (obs.image, obs.feature))
            .collect::<Vec<_>>();
        let mut next_queue = Vec::new();
        let mut visited = curr_queue.iter().copied().collect::<HashSet<_>>();
        let mut completed = 0usize;

        for transitivity in 1..=options.complete_max_transitivity {
            while let Some((image, feature)) = curr_queue.pop() {
                let direct_corrs = self
                    .state
                    .correspondence_graph()
                    .extract_correspondences(image as ImageId, feature as Point2DIdx)
                    .unwrap_or_default();
                for corr in direct_corrs {
                    let corr_image = corr.image_id as usize;
                    let corr_feature = corr.point2d_idx as usize;
                    if !visited.insert((corr_image, corr_feature)) {
                        continue;
                    }
                    if self
                        .reconstruction
                        .poses
                        .get(corr_image)
                        .copied()
                        .flatten()
                        .is_none()
                        || !self.valid_observation(corr_image, corr_feature)
                        || self.reconstruction.observations[corr_image][corr_feature].is_some()
                    {
                        continue;
                    }
                    let camera_index = self
                        .reconstruction
                        .image_camera_indices
                        .get(corr_image)
                        .copied()
                        .unwrap_or(0);
                    if self.has_camera_bogus_params(camera_index, options) {
                        continue;
                    }
                    let Some(pose) = self.reconstruction.poses[corr_image] else {
                        continue;
                    };
                    let keypoint = &self.frames[corr_image].keypoints[corr_feature];
                    let camera = self.reconstruction.camera_for_image(corr_image);
                    let squared_error = calculate_squared_reprojection_error(
                        &Vector2::new(keypoint.x() as f64, keypoint.y() as f64),
                        &point_xyz,
                        &se3_to_matrix3x4(pose),
                        &camera,
                    );
                    if squared_error > max_squared_reproj_error {
                        continue;
                    }
                    if !self.state.observation_manager_mut().add_observation(
                        self.frames,
                        self.pairs,
                        self.reconstruction,
                        point_id,
                        TrackObservation {
                            image: corr_image,
                            feature: corr_feature,
                        },
                    ) {
                        continue;
                    }
                    self.state
                        .observation_manager_mut()
                        .mark_point3d_modified(point_id);
                    completed += 1;
                    if transitivity < options.complete_max_transitivity {
                        next_queue.push((corr_image, corr_feature));
                    }
                }
            }
            if next_queue.is_empty() {
                break;
            }
            curr_queue = next_queue;
            next_queue = Vec::new();
        }
        completed
    }

    fn merge_track(&mut self, options: &IncrementalTriangulatorOptions, point_id: usize) -> usize {
        if point_id >= self.reconstruction.points.len() {
            return 0;
        }
        let track = self.reconstruction.points[point_id].track.clone();
        for obs in track {
            let direct_corrs = self
                .state
                .correspondence_graph()
                .extract_correspondences(obs.image as ImageId, obs.feature as Point2DIdx)
                .unwrap_or_default();
            for corr in direct_corrs {
                let corr_image = corr.image_id as usize;
                let corr_feature = corr.point2d_idx as usize;
                if !self.valid_observation(corr_image, corr_feature) {
                    continue;
                }
                let Some(other_point_id) =
                    self.reconstruction.observations[corr_image][corr_feature]
                else {
                    continue;
                };
                if other_point_id == point_id || other_point_id >= self.reconstruction.points.len()
                {
                    continue;
                }
                let key = ordered_point_pair(point_id, other_point_id);
                if !self.state.merge_trials.insert(key) {
                    continue;
                }
                if self.try_merge_pair(options, point_id, other_point_id) {
                    let merged = key.0;
                    let merged_count = self.reconstruction.points[merged].track.len();
                    let recursive_count = self.merge_track(options, merged);
                    return recursive_count.max(merged_count);
                }
            }
        }
        0
    }

    fn pair_triangulation_counts(&self, pair: &PairGeometry) -> (usize, usize) {
        let mut tri_corrs = 0usize;
        let mut total_corrs = 0usize;
        for match_ in &pair.inlier_matches {
            let left_feature = match_.query_idx as usize;
            let right_feature = match_.train_idx as usize;
            if !self.valid_observation(pair.left, left_feature)
                || !self.valid_observation(pair.right, right_feature)
            {
                continue;
            }
            total_corrs += 1;
            let left_point = self.reconstruction.observations[pair.left][left_feature];
            let right_point = self.reconstruction.observations[pair.right][right_feature];
            if left_point.is_some() && left_point == right_point {
                tri_corrs += 1;
            }
        }
        (tri_corrs, total_corrs)
    }

    fn try_merge_pair(
        &mut self,
        options: &IncrementalTriangulatorOptions,
        point_id1: usize,
        point_id2: usize,
    ) -> bool {
        if point_id1 == point_id2
            || point_id1 >= self.reconstruction.points.len()
            || point_id2 >= self.reconstruction.points.len()
        {
            return false;
        }
        let (keep_id, remove_id) = ordered_point_pair(point_id1, point_id2);
        let keep = self.reconstruction.points[keep_id].clone();
        let remove = self.reconstruction.points[remove_id].clone();
        let mut merged_track = keep.track.clone();
        merged_track.extend(remove.track.iter().cloned());
        let keep_weight = keep.track.len() as f32;
        let remove_weight = remove.track.len() as f32;
        let total_weight = keep_weight + remove_weight;
        let merged_xyz = [
            (keep.xyz[0] * keep_weight + remove.xyz[0] * remove_weight) / total_weight,
            (keep.xyz[1] * keep_weight + remove.xyz[1] * remove_weight) / total_weight,
            (keep.xyz[2] * keep_weight + remove.xyz[2] * remove_weight) / total_weight,
        ];
        if !merged_xyz.iter().all(|value| value.is_finite()) {
            return false;
        }
        if !self.track_reprojects(merged_xyz, &merged_track, options.merge_max_reproj_error_px) {
            return false;
        }

        let merged_color = average_color(&merged_track, self.frames);
        let merged_error = mean_track_reprojection_error(
            merged_xyz,
            &merged_track,
            self.frames,
            self.reconstruction,
        )
        .unwrap_or(0.0);
        let last_point_id = self.reconstruction.points.len() - 1;
        let moved_from = (remove_id != last_point_id).then_some(last_point_id);
        let merged = self
            .state
            .observation_manager_mut()
            .merge_points3d(
                self.frames,
                self.pairs,
                self.reconstruction,
                keep_id,
                remove_id,
                Point3D {
                    xyz: merged_xyz,
                    color: merged_color,
                    error: merged_error,
                    track: merged_track,
                },
            )
            .is_some();
        if merged {
            self.state
                .sync_merge_trials_after_point_merge(keep_id, remove_id, moved_from);
        }
        merged
    }

    fn track_reprojects(
        &self,
        xyz: [f32; 3],
        track: &[TrackObservation],
        max_reproj_error_px: f32,
    ) -> bool {
        track.iter().all(|obs| {
            let Some(pose) = self.reconstruction.poses[obs.image] else {
                return false;
            };
            let keypoint = &self.frames[obs.image].keypoints[obs.feature];
            let error = crate::geometry::reprojection_error_px(
                xyz,
                pose,
                [keypoint.x(), keypoint.y()],
                self.reconstruction.camera_for_image(obs.image),
            );
            error.is_finite() && error <= max_reproj_error_px
        })
    }
}

fn camera_has_bogus_params_for_triangulation(
    camera: CameraModel,
    options: &IncrementalTriangulatorOptions,
) -> bool {
    camera.has_bogus_params(
        options.min_focal_length_ratio,
        options.max_focal_length_ratio,
        options.max_extra_param,
    )
}

fn normalized_camera_ray(camera: CameraModel, x: f32, y: f32) -> Option<Vector3<f64>> {
    let xy = camera.cam_from_img_f32(x, y)?;
    Vector3::new(xy[0] as f64, xy[1] as f64, 1.0).try_normalize(1e-12)
}

fn n_choose_k(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result = 1usize;
    for i in 0..k {
        result = result.saturating_mul(n - i) / (i + 1);
    }
    result
}

fn se3_to_matrix3x4(pose: SE3) -> Matrix3x4<f64> {
    let r = pose.rotation_matrix();
    let t = pose.translation();
    nalgebra::Matrix3x4::new(
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

fn ordered_point_pair(a: usize, b: usize) -> (usize, usize) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn mean_track_reprojection_error(
    xyz: [f32; 3],
    track: &[TrackObservation],
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
) -> Option<f32> {
    let mut total = 0.0f32;
    let mut count = 0usize;
    for obs in track {
        let pose = reconstruction.poses[obs.image]?;
        let keypoint = &frames[obs.image].keypoints[obs.feature];
        let error = crate::geometry::reprojection_error_px(
            xyz,
            pose,
            [keypoint.x(), keypoint.y()],
            reconstruction.camera_for_image(obs.image),
        );
        if !error.is_finite() {
            return None;
        }
        total += error;
        count += 1;
    }
    (count > 0).then_some(total / count as f32)
}

/// Build COLMAP `EstimateTriangulationOptions` from the incremental
/// triangulator options, matching `IncrementalTriangulator::Create`'s
/// configuration (angular residual; `create_max_angle_error` as the inlier
/// threshold; `min_angle` as the minimum triangulation angle).
fn create_estimate_options(
    options: &IncrementalTriangulatorOptions,
) -> EstimateTriangulationOptions {
    EstimateTriangulationOptions {
        min_tri_angle: (options.min_angle_deg as f64).to_radians(),
        residual_type: ResidualType::AngularError,
        max_error: (options.create_max_angle_error_deg as f64).to_radians(),
        confidence: 0.9999,
        dyn_num_trials_multiplier: 3.0,
        min_inlier_ratio: 0.02,
        min_num_trials: 0,
        max_num_trials: 10000,
        random_seed: options.random_seed,
        num_threads: options.num_threads,
    }
}

fn complete_estimate_options(
    options: &IncrementalTriangulatorOptions,
) -> EstimateTriangulationOptions {
    EstimateTriangulationOptions {
        min_tri_angle: (options.min_angle_deg as f64).to_radians(),
        residual_type: ResidualType::ReprojectionError,
        max_error: options.complete_max_reproj_error_px as f64,
        confidence: 0.9999,
        dyn_num_trials_multiplier: 3.0,
        min_inlier_ratio: 0.02,
        min_num_trials: 0,
        max_num_trials: 10000,
        random_seed: options.random_seed,
        num_threads: options.num_threads,
    }
}

fn average_color(observations: &[TrackObservation], frames: &[ImageFrame]) -> [u8; 3] {
    let mut color = [0usize; 3];
    let mut count = 0usize;
    for obs in observations {
        let Some(c) = frames
            .get(obs.image)
            .and_then(|frame| frame.colors.get(obs.feature))
        else {
            continue;
        };
        color[0] += c[0] as usize;
        color[1] += c[1] as usize;
        color[2] += c[2] as usize;
        count += 1;
    }
    if count == 0 {
        return [0, 0, 0];
    }
    [
        (color[0] / count) as u8,
        (color[1] / count) as u8,
        (color[2] / count) as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CameraModel, ImageFrame, PairGeometry, Reconstruction};
    use rustslam::{Descriptors, Match};
    use std::path::PathBuf;

    #[test]
    fn average_color_returns_black_when_frame_colors_are_lazy() {
        let mut frames = vec![frame(0)];
        frames[0].colors.clear();
        let observations = [TrackObservation {
            image: 0,
            feature: 0,
        }];

        assert_eq!(average_color(&observations, &frames), [0, 0, 0]);
    }

    #[test]
    fn average_color_averages_only_available_colors() {
        let mut frames = vec![frame(0), frame(1)];
        frames[0].colors.clear();
        frames[1].colors = vec![[10, 20, 30], [30, 40, 50]];
        let observations = [
            TrackObservation {
                image: 0,
                feature: 0,
            },
            TrackObservation {
                image: 1,
                feature: 0,
            },
            TrackObservation {
                image: 1,
                feature: 1,
            },
        ];

        assert_eq!(average_color(&observations, &frames), [20, 30, 40]);
    }

    #[test]
    fn mapper_triangulator_options_keep_colmap_ignore_two_view_tracks_default() {
        let options = IncrementalTriangulatorOptions::from_mapper_threshold(4.0);
        assert!(options.ignore_two_view_tracks);
        let options = IncrementalTriangulatorOptions::from_mapper_config_values(
            4.0, 0.1, 10.0, 1.0, -1, false, 1,
        );
        assert!(!options.ignore_two_view_tracks);
    }

    #[test]
    fn triangulate_image_creates_pair_tracks() {
        let mut frames = vec![frame(0), frame(1)];
        frames[1].keypoints = vec![
            rustslam::KeyPoint::new(55.0, 50.0),
            rustslam::KeyPoint::new(60.0, 50.0),
        ];
        let pairs = vec![pair(0, 1, &[(0, 0), (1, 1)])];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::from_quat_translation(
            glam::Quat::IDENTITY,
            glam::Vec3::new(1.0, 0.0, 0.0),
        ));

        let mut tri_state = IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        let mut triangulator =
            IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction, &mut tri_state);
        let report = triangulator.triangulate_image(
            &IncrementalTriangulatorOptions {
                min_angle_deg: 0.1,
                merge_max_reproj_error_px: 10.0,
                ignore_two_view_tracks: false,
                ..IncrementalTriangulatorOptions::default()
            },
            0,
        );

        assert_eq!(report.created_points, 2);
        assert_eq!(triangulator.reconstruction.points.len(), 2);
        assert_eq!(triangulator.get_modified_points3d().len(), 2);
    }

    #[test]
    fn triangulate_image_ignores_two_view_tracks_by_default() {
        let mut frames = vec![frame(0), frame(1)];
        frames[1].keypoints = vec![rustslam::KeyPoint::new(55.0, 50.0)];
        let pairs = vec![pair(0, 1, &[(0, 0)])];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::from_quat_translation(
            glam::Quat::IDENTITY,
            glam::Vec3::new(1.0, 0.0, 0.0),
        ));

        let mut tri_state = IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        let mut triangulator =
            IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction, &mut tri_state);
        let report = triangulator.triangulate_image(
            &IncrementalTriangulatorOptions {
                min_angle_deg: 0.1,
                merge_max_reproj_error_px: 10.0,
                ..IncrementalTriangulatorOptions::default()
            },
            0,
        );

        assert_eq!(report.created_points, 0);
        assert!(triangulator.reconstruction.points.is_empty());
    }

    #[test]
    fn create_pair_track_builds_multiview_track_via_estimator() {
        // Three registered views observing one world point. Creating from the
        // (0,1) seed should transitively gather image 2 and emit a single
        // three-view track through `estimate_triangulation`.
        let mut frames = vec![frame(0), frame(1), frame(2)];
        frames[0].keypoints[0] = rustslam::KeyPoint::new(50.0, 50.0);
        frames[1].keypoints[0] = rustslam::KeyPoint::new(62.5, 50.0);
        frames[2].keypoints[0] = rustslam::KeyPoint::new(50.0, 62.5);
        let pairs = vec![
            pair(0, 1, &[(0, 0)]),
            pair(0, 2, &[(0, 0)]),
            pair(1, 2, &[(0, 0)]),
        ];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::from_quat_translation(
            glam::Quat::IDENTITY,
            glam::Vec3::new(1.0, 0.0, 0.0),
        ));
        reconstruction.poses[2] = Some(SE3::from_quat_translation(
            glam::Quat::IDENTITY,
            glam::Vec3::new(0.0, 1.0, 0.0),
        ));

        let mut tri_state = IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        let mut triangulator =
            IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction, &mut tri_state);
        let report = triangulator.triangulate_image(
            &IncrementalTriangulatorOptions {
                min_angle_deg: 0.5,
                ignore_two_view_tracks: false,
                ..IncrementalTriangulatorOptions::default()
            },
            0,
        );

        assert_eq!(report.created_points, 1);
        assert_eq!(triangulator.reconstruction.points.len(), 1);
        assert_eq!(triangulator.reconstruction.points[0].track.len(), 3);
        let xyz = triangulator.reconstruction.points[0].xyz;
        assert!(
            xyz[0].abs() < 1e-2 && xyz[1].abs() < 1e-2 && (xyz[2] - 4.0).abs() < 1e-2,
            "{xyz:?}"
        );
    }

    #[test]
    fn triangulate_image_continues_existing_track() {
        let mut frames = vec![frame(0), frame(1)];
        frames[1].keypoints[0] = rustslam::KeyPoint::new(75.0, 50.0);
        let pairs = vec![pair(0, 1, &[(0, 0)])];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(baseline_pose());
        reconstruction.observations[0][0] = Some(0);
        reconstruction.point_ids.push(1);
        reconstruction.points.push(Point3D {
            xyz: [0.0, 0.0, 2.0],
            color: [0, 0, 0],
            error: 0.0,
            track: vec![TrackObservation {
                image: 0,
                feature: 0,
            }],
        });

        let mut tri_state = IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        let mut triangulator =
            IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction, &mut tri_state);
        let report = triangulator.triangulate_image(
            &IncrementalTriangulatorOptions {
                complete_max_reproj_error_px: 10.0,
                min_angle_deg: 0.1,
                ..IncrementalTriangulatorOptions::default()
            },
            1,
        );

        assert_eq!(report.continued_observations, 1);
        assert_eq!(triangulator.reconstruction.points[0].track.len(), 2);
        assert_eq!(triangulator.get_modified_points3d(), &HashSet::from([0]));
    }

    #[test]
    fn complete_tracks_adds_transitive_observations() {
        let mut frames = vec![frame(0), frame(1), frame(2)];
        frames[1].keypoints[0] = rustslam::KeyPoint::new(75.0, 50.0);
        frames[2].keypoints[0] = rustslam::KeyPoint::new(100.0, 50.0);
        let pairs = vec![pair(0, 1, &[(0, 0)]), pair(1, 2, &[(0, 0)])];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(baseline_pose());
        reconstruction.poses[2] = Some(SE3::from_quat_translation(
            glam::Quat::IDENTITY,
            glam::Vec3::new(2.0, 0.0, 0.0),
        ));
        reconstruction.observations[0][0] = Some(0);
        reconstruction.observations[1][0] = Some(0);
        reconstruction.point_ids.push(1);
        reconstruction.points.push(Point3D {
            xyz: [0.0, 0.0, 2.0],
            color: [0, 0, 0],
            error: 0.0,
            track: vec![
                TrackObservation {
                    image: 0,
                    feature: 0,
                },
                TrackObservation {
                    image: 1,
                    feature: 0,
                },
            ],
        });

        let mut tri_state = IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        let mut triangulator =
            IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction, &mut tri_state);
        let completed = triangulator.complete_tracks(
            &IncrementalTriangulatorOptions {
                complete_max_transitivity: 2,
                complete_max_reproj_error_px: 10.0,
                min_angle_deg: 0.1,
                ..IncrementalTriangulatorOptions::default()
            },
            &HashSet::from([0]),
        );

        assert_eq!(completed, 1);
        assert_eq!(triangulator.reconstruction.observations[2][0], Some(0));
        assert_eq!(reconstruction.points[0].track.len(), 3);
    }

    #[test]
    fn complete_image_creates_track_for_untriangulated_two_view_cluster() {
        let mut frames = vec![frame(0), frame(1), frame(2)];
        frames[1].keypoints[0] = rustslam::KeyPoint::new(75.0, 50.0);
        frames[2].keypoints[0] = rustslam::KeyPoint::new(100.0, 50.0);
        let pairs = vec![pair(0, 1, &[(0, 0)]), pair(1, 2, &[(0, 0)])];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(baseline_pose());
        reconstruction.poses[2] = Some(SE3::from_quat_translation(
            glam::Quat::IDENTITY,
            glam::Vec3::new(2.0, 0.0, 0.0),
        ));

        let mut tri_state = IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        let mut triangulator =
            IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction, &mut tri_state);
        let report = triangulator.complete_image(
            &IncrementalTriangulatorOptions {
                complete_max_reproj_error_px: 10.0,
                min_angle_deg: 0.1,
                ignore_two_view_tracks: false,
                ..IncrementalTriangulatorOptions::default()
            },
            2,
        );

        assert_eq!(report.created_points, 1);
        assert_eq!(reconstruction.observations[2][0], Some(0));
        assert!(reconstruction.points[0]
            .track
            .iter()
            .any(|obs| obs.image == 2));
    }

    #[test]
    fn triangulate_image_skips_bogus_camera() {
        let mut frames = vec![frame(0), frame(1)];
        frames[1].keypoints = vec![rustslam::KeyPoint::new(55.0, 50.0)];
        let pairs = vec![pair(0, 1, &[(0, 0)])];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.cameras = vec![CameraModel::new_pinhole(100, 100, 0.5, 0.5, 50.0, 50.0)];
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(baseline_pose());

        let mut tri_state = IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        let mut triangulator =
            IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction, &mut tri_state);
        let report = triangulator.triangulate_image(
            &IncrementalTriangulatorOptions {
                min_focal_length_ratio: 0.1,
                max_focal_length_ratio: 10.0,
                ignore_two_view_tracks: false,
                min_angle_deg: 0.1,
                ..IncrementalTriangulatorOptions::default()
            },
            0,
        );

        assert_eq!(report.created_points, 0);
        assert!(reconstruction.points.is_empty());
    }

    #[test]
    fn merge_tracks_combines_corresponding_points() {
        let mut frames = vec![frame(0), frame(1)];
        frames[1].keypoints[0] = rustslam::KeyPoint::new(75.0, 50.0);
        let pairs = vec![pair(0, 1, &[(0, 0)])];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(baseline_pose());
        reconstruction.observations[0][0] = Some(0);
        reconstruction.observations[1][0] = Some(1);
        reconstruction.point_ids.extend([1, 2]);
        reconstruction.points.push(Point3D {
            xyz: [0.0, 0.0, 2.0],
            color: [10, 0, 0],
            error: 0.0,
            track: vec![TrackObservation {
                image: 0,
                feature: 0,
            }],
        });
        reconstruction.points.push(Point3D {
            xyz: [0.0, 0.0, 2.0],
            color: [0, 10, 0],
            error: 0.0,
            track: vec![TrackObservation {
                image: 1,
                feature: 0,
            }],
        });

        let mut tri_state = IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        let mut triangulator =
            IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction, &mut tri_state);
        let merged = triangulator.merge_tracks(
            &IncrementalTriangulatorOptions {
                merge_max_reproj_error_px: 10.0,
                min_angle_deg: 0.1,
                ..IncrementalTriangulatorOptions::default()
            },
            &HashSet::from([0]),
        );

        assert_eq!(merged, 2);
        assert_eq!(triangulator.reconstruction.points.len(), 1);
        assert_eq!(triangulator.reconstruction.points[0].track.len(), 2);
        assert_eq!(triangulator.reconstruction.observations[0][0], Some(0));
        assert_eq!(triangulator.reconstruction.observations[1][0], Some(0));
        assert_eq!(triangulator.get_modified_points3d(), &HashSet::from([0]));
    }

    #[test]
    fn merge_trial_cache_remaps_only_the_swapped_tail_point() {
        let frames = vec![frame(0), frame(1)];
        let reconstruction = reconstruction(&frames);
        let mut state = IncrementalTriangulatorState::new(&frames, &[], &reconstruction);
        state.merge_trials = HashSet::from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]);

        state.sync_merge_trials_after_point_merge(0, 2, Some(5));

        assert_eq!(state.merge_trials, HashSet::from([(2, 4), (3, 4)]));
    }

    #[test]
    fn merge_tracks_uses_colmap_track_length_weighted_xyz() {
        let keep_xyz = [-0.05, 0.0, 2.5];
        let remove_xyz = [0.1, 0.0, 4.0];
        let expected_xyz = [0.0, 0.0, 3.0];
        let measured_xyz = [0.06, -0.02, 3.05];
        let pose0 = SE3::identity();
        let pose1 = baseline_pose();
        let pose2 =
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(2.0, 0.0, 0.0));
        let mut frames = vec![frame(0), frame(1), frame(2)];
        frames[0].keypoints[0] = project_keypoint(pose0, measured_xyz);
        frames[1].keypoints[0] = project_keypoint(pose1, measured_xyz);
        frames[2].keypoints[0] = project_keypoint(pose2, measured_xyz);
        let pairs = vec![pair(1, 2, &[(0, 0)])];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(pose0);
        reconstruction.poses[1] = Some(pose1);
        reconstruction.poses[2] = Some(pose2);
        reconstruction.observations[0][0] = Some(0);
        reconstruction.observations[1][0] = Some(0);
        reconstruction.observations[2][0] = Some(1);
        reconstruction.point_ids.extend([1, 2]);
        reconstruction.points.push(Point3D {
            xyz: keep_xyz,
            color: [10, 0, 0],
            error: 0.0,
            track: vec![
                TrackObservation {
                    image: 0,
                    feature: 0,
                },
                TrackObservation {
                    image: 1,
                    feature: 0,
                },
            ],
        });
        reconstruction.points.push(Point3D {
            xyz: remove_xyz,
            color: [0, 10, 0],
            error: 0.0,
            track: vec![TrackObservation {
                image: 2,
                feature: 0,
            }],
        });

        let mut tri_state = IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        let mut triangulator =
            IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction, &mut tri_state);
        let merged = triangulator.merge_tracks(
            &IncrementalTriangulatorOptions {
                merge_max_reproj_error_px: 2.0,
                ..IncrementalTriangulatorOptions::default()
            },
            &HashSet::from([0]),
        );

        assert_eq!(merged, 3);
        assert_eq!(triangulator.reconstruction.points.len(), 1);
        let merged_xyz = triangulator.reconstruction.points[0].xyz;
        for axis in 0..3 {
            assert!(
                (merged_xyz[axis] - expected_xyz[axis]).abs() < 1.0e-6,
                "{merged_xyz:?}"
            );
        }
        assert_eq!(triangulator.reconstruction.points[0].track.len(), 3);
    }

    #[test]
    fn merge_tracks_retries_pairs_after_failed_public_call() {
        let keep_xyz = [-0.05, 0.0, 2.5];
        let remove_xyz = [0.1, 0.0, 4.0];
        let measured_xyz = [0.06, -0.02, 3.05];
        let pose0 = SE3::identity();
        let pose1 = baseline_pose();
        let pose2 =
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(2.0, 0.0, 0.0));
        let mut frames = vec![frame(0), frame(1), frame(2)];
        frames[0].keypoints[0] = project_keypoint(pose0, measured_xyz);
        frames[1].keypoints[0] = project_keypoint(pose1, measured_xyz);
        frames[2].keypoints[0] = project_keypoint(pose2, measured_xyz);
        let pairs = vec![pair(1, 2, &[(0, 0)])];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(pose0);
        reconstruction.poses[1] = Some(pose1);
        reconstruction.poses[2] = Some(pose2);
        reconstruction.observations[0][0] = Some(0);
        reconstruction.observations[1][0] = Some(0);
        reconstruction.observations[2][0] = Some(1);
        reconstruction.point_ids.extend([1, 2]);
        reconstruction.points.push(Point3D {
            xyz: keep_xyz,
            color: [10, 0, 0],
            error: 0.0,
            track: vec![
                TrackObservation {
                    image: 0,
                    feature: 0,
                },
                TrackObservation {
                    image: 1,
                    feature: 0,
                },
            ],
        });
        reconstruction.points.push(Point3D {
            xyz: remove_xyz,
            color: [0, 10, 0],
            error: 0.0,
            track: vec![TrackObservation {
                image: 2,
                feature: 0,
            }],
        });

        let mut tri_state = IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        let mut triangulator =
            IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction, &mut tri_state);

        let failed = triangulator.merge_tracks(
            &IncrementalTriangulatorOptions {
                merge_max_reproj_error_px: 0.001,
                ..IncrementalTriangulatorOptions::default()
            },
            &HashSet::from([0]),
        );
        assert_eq!(failed, 0);
        assert_eq!(triangulator.reconstruction.points.len(), 2);

        let merged = triangulator.merge_tracks(
            &IncrementalTriangulatorOptions {
                merge_max_reproj_error_px: 2.0,
                ..IncrementalTriangulatorOptions::default()
            },
            &HashSet::from([0]),
        );
        assert_eq!(merged, 3);
        assert_eq!(triangulator.reconstruction.points.len(), 1);
    }

    #[test]
    fn merge_tracks_retries_failed_pairs_after_successful_merge_creates_new_point_identity() {
        let pose0 = SE3::identity();
        let pose1 = baseline_pose();
        let pose2 =
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(2.0, 0.0, 0.0));
        let pose3 =
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(3.0, 0.0, 0.0));
        let mut frames = vec![frame(0), frame(1), frame(2), frame(3)];
        let final_xyz = [0.0, 0.0, 4.0];
        for (image, pose) in [pose0, pose1, pose2, pose3].into_iter().enumerate() {
            frames[image].keypoints[0] = project_keypoint(pose, final_xyz);
        }
        let pairs = vec![
            pair(0, 2, &[(0, 0)]),
            pair(0, 1, &[(0, 0)]),
            pair(1, 2, &[(0, 0)]),
        ];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(pose0);
        reconstruction.poses[1] = Some(pose1);
        reconstruction.poses[2] = Some(pose2);
        reconstruction.poses[3] = Some(pose3);
        reconstruction.observations[0][0] = Some(0);
        reconstruction.observations[1][0] = Some(1);
        reconstruction.observations[2][0] = Some(2);
        reconstruction.observations[3][0] = Some(2);
        reconstruction.point_ids.extend([1, 2, 3]);
        reconstruction.points.push(Point3D {
            xyz: [0.0, 0.0, 2.0],
            color: [0, 0, 0],
            error: 0.0,
            track: vec![TrackObservation {
                image: 0,
                feature: 0,
            }],
        });
        reconstruction.points.push(Point3D {
            xyz: [0.0, 0.0, 6.0],
            color: [0, 0, 0],
            error: 0.0,
            track: vec![TrackObservation {
                image: 1,
                feature: 0,
            }],
        });
        reconstruction.points.push(Point3D {
            xyz: [0.0, 0.0, 4.0],
            color: [0, 0, 0],
            error: 0.0,
            track: vec![
                TrackObservation {
                    image: 2,
                    feature: 0,
                },
                TrackObservation {
                    image: 3,
                    feature: 0,
                },
            ],
        });

        let mut tri_state = IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        let mut triangulator =
            IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction, &mut tri_state);
        let merged = triangulator.merge_tracks(
            &IncrementalTriangulatorOptions {
                merge_max_reproj_error_px: 1.0,
                ..IncrementalTriangulatorOptions::default()
            },
            &HashSet::from([0]),
        );

        assert_eq!(merged, 4);
        assert_eq!(triangulator.reconstruction.points.len(), 1);
        assert_eq!(triangulator.reconstruction.points[0].track.len(), 4);
        for image in 0..4 {
            assert_eq!(
                triangulator.reconstruction.observations[image][0],
                Some(0),
                "image {image}"
            );
        }
    }

    #[test]
    fn merge_tracks_allows_colmap_same_image_track_elements() {
        let frames = vec![frame(0), frame(1)];
        let pairs = vec![pair(0, 1, &[(0, 0)])];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::identity());
        reconstruction.observations[0][0] = Some(0);
        reconstruction.observations[1][0] = Some(1);
        reconstruction.point_ids.extend([1, 2]);
        reconstruction.points.push(Point3D {
            xyz: [0.0, 0.0, 1.0],
            color: [0, 0, 0],
            error: 0.0,
            track: vec![
                TrackObservation {
                    image: 0,
                    feature: 0,
                },
                TrackObservation {
                    image: 1,
                    feature: 1,
                },
            ],
        });
        reconstruction.points.push(Point3D {
            xyz: [0.0, 0.0, 1.0],
            color: [0, 0, 0],
            error: 0.0,
            track: vec![TrackObservation {
                image: 1,
                feature: 0,
            }],
        });

        let mut tri_state = IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        let mut triangulator =
            IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction, &mut tri_state);
        let merged = triangulator.merge_tracks(
            &IncrementalTriangulatorOptions {
                merge_max_reproj_error_px: 10.0,
                ..IncrementalTriangulatorOptions::default()
            },
            &HashSet::from([0]),
        );

        assert_eq!(merged, 3);
        assert_eq!(triangulator.reconstruction.points.len(), 1);
        assert_eq!(triangulator.reconstruction.points[0].track.len(), 3);
        assert_eq!(triangulator.reconstruction.observations[1][0], Some(0));
        assert_eq!(triangulator.reconstruction.observations[1][1], Some(0));
    }

    #[test]
    fn retriangulate_creates_under_reconstructed_pair_points() {
        let mut frames = vec![frame(0), frame(1)];
        frames[1].keypoints = vec![
            rustslam::KeyPoint::new(55.0, 50.0),
            rustslam::KeyPoint::new(60.0, 50.0),
        ];
        let pairs = vec![pair(0, 1, &[(0, 0), (1, 1)])];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::from_quat_translation(
            glam::Quat::IDENTITY,
            glam::Vec3::new(1.0, 0.0, 0.0),
        ));

        let mut tri_state = IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        let mut triangulator =
            IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction, &mut tri_state);
        let added = triangulator.retriangulate(&IncrementalTriangulatorOptions {
            re_min_ratio: 0.5,
            min_angle_deg: 0.1,
            merge_max_reproj_error_px: 10.0,
            ignore_two_view_tracks: false,
            ..IncrementalTriangulatorOptions::default()
        });

        assert_eq!(added, 4);
        assert_eq!(triangulator.reconstruction.points.len(), 2);
    }

    #[test]
    fn retriangulate_limits_trials_per_pair_to_re_max_trials() {
        let mut frames = vec![frame(0), frame(1)];
        frames[1].keypoints = vec![
            rustslam::KeyPoint::new(55.0, 50.0),
            rustslam::KeyPoint::new(60.0, 50.0),
        ];
        let pairs = vec![pair(0, 1, &[(0, 0), (1, 1)])];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::from_quat_translation(
            glam::Quat::IDENTITY,
            glam::Vec3::new(1.0, 0.0, 0.0),
        ));

        let mut tri_state = IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        let options = IncrementalTriangulatorOptions {
            re_min_ratio: 0.5,
            re_max_trials: 2,
            min_angle_deg: 0.1,
            merge_max_reproj_error_px: 10.0,
            ignore_two_view_tracks: false,
            ..IncrementalTriangulatorOptions::default()
        };

        let first = {
            let mut triangulator =
                IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction, &mut tri_state);
            triangulator.retriangulate(&options)
        };
        assert_eq!(first, 4);
        assert_eq!(
            tri_state
                .retriangulation_trials()
                .get(&ordered_point_pair(0, 1))
                .copied(),
            Some(1)
        );

        // The pair is now fully triangulated, so a second pass records no new
        // trial because the ratio gate is satisfied before the trial counter
        // is incremented.
        let second = {
            let mut triangulator =
                IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction, &mut tri_state);
            triangulator.retriangulate(&options)
        };
        assert_eq!(second, 0);
        assert_eq!(
            tri_state
                .retriangulation_trials()
                .get(&ordered_point_pair(0, 1))
                .copied(),
            Some(1)
        );
    }

    #[test]
    fn retriangulation_trials_persist_across_triangulator_instances() {
        let mut frames = vec![frame(0), frame(1)];
        frames[1].keypoints = vec![
            rustslam::KeyPoint::new(55.0, 50.0),
            rustslam::KeyPoint::new(60.0, 50.0),
        ];
        let pairs = vec![pair(0, 1, &[(0, 0), (1, 1)])];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::from_quat_translation(
            glam::Quat::IDENTITY,
            glam::Vec3::new(1.0, 0.0, 0.0),
        ));

        let mut tri_state = IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        let options = IncrementalTriangulatorOptions {
            re_min_ratio: 0.5,
            re_max_trials: 1,
            min_angle_deg: 0.1,
            merge_max_reproj_error_px: 10.0,
            ignore_two_view_tracks: false,
            ..IncrementalTriangulatorOptions::default()
        };

        {
            let mut triangulator =
                IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction, &mut tri_state);
            assert_eq!(triangulator.retriangulate(&options), 4);
        }
        assert_eq!(
            tri_state
                .retriangulation_trials()
                .get(&ordered_point_pair(0, 1))
                .copied(),
            Some(1)
        );

        {
            let mut triangulator =
                IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction, &mut tri_state);
            assert_eq!(triangulator.retriangulate(&options), 0);
        }
    }

    #[test]
    fn rollback_sync_preserves_retriangulation_trials_and_refreshes_stats() {
        let mut frames = vec![frame(0), frame(1)];
        frames[1].keypoints = vec![
            rustslam::KeyPoint::new(55.0, 50.0),
            rustslam::KeyPoint::new(60.0, 50.0),
        ];
        let pairs = vec![pair(0, 1, &[(0, 0), (1, 1)])];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(baseline_pose());
        let rollback_reconstruction = reconstruction.clone();
        let mut tri_state = IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        let options = IncrementalTriangulatorOptions {
            re_min_ratio: 0.5,
            re_max_trials: 1,
            min_angle_deg: 0.1,
            merge_max_reproj_error_px: 10.0,
            ignore_two_view_tracks: false,
            ..IncrementalTriangulatorOptions::default()
        };

        {
            let mut triangulator =
                IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction, &mut tri_state);
            assert_eq!(triangulator.retriangulate(&options), 4);
        }
        assert_eq!(
            tri_state
                .retriangulation_trials()
                .get(&ordered_point_pair(0, 1))
                .copied(),
            Some(1)
        );
        assert_eq!(tri_state.observation_manager().num_visible_points3d(1), 2);

        reconstruction = rollback_reconstruction;
        tri_state.sync_after_reconstruction_rollback(&frames, &pairs, &reconstruction);

        assert_eq!(
            tri_state
                .retriangulation_trials()
                .get(&ordered_point_pair(0, 1))
                .copied(),
            Some(1)
        );
        assert_eq!(tri_state.observation_manager().num_visible_points3d(1), 0);
        assert_eq!(tri_state.observation_manager().num_observations(1), 2);
        assert_eq!(
            tri_state
                .observation_manager()
                .num_visible_correspondences(1),
            2
        );
    }

    fn reconstruction(frames: &[ImageFrame]) -> Reconstruction {
        Reconstruction {
            camera: CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0),
            cameras: vec![CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0)],
            camera_ids: vec![1],
            rigs: Vec::new(),
            frames: Vec::new(),
            image_names: frames.iter().map(|frame| frame.name.clone()).collect(),
            image_paths: frames.iter().map(|frame| frame.path.clone()).collect(),
            image_ids: (0..frames.len()).map(|idx| idx as u32 + 1).collect(),
            image_camera_indices: vec![0; frames.len()],
            image_frame_indices: vec![None; frames.len()],
            poses: vec![None; frames.len()],
            observations: frames
                .iter()
                .map(|frame| vec![None; frame.keypoints.len()])
                .collect(),
            keypoints: frames.iter().map(|frame| frame.keypoints.clone()).collect(),
            point_ids: Vec::new(),
            points: Vec::new(),
        }
    }

    fn pair(left: usize, right: usize, matches: &[(u32, u32)]) -> PairGeometry {
        PairGeometry {
            left,
            right,
            two_view_config: crate::database::COLMAP_TWO_VIEW_CALIBRATED,
            f_matrix: None,
            e_matrix: None,
            h_matrix: None,
            qvec: None,
            tvec: None,
            matches: Vec::new(),
            inlier_matches: matches
                .iter()
                .map(|&(query_idx, train_idx)| Match {
                    query_idx,
                    train_idx,
                    distance: 0.0,
                })
                .collect(),
            relative_pose: SE3::identity(),
            inliers: matches.len(),
            triangulated: 0,
            mean_reprojection_error_px: 0.0,
            rotation_deg: 0.0,
            median_triangulation_angle_deg: 0.0,
            pose_graph_only: false,
        }
    }

    fn baseline_pose() -> SE3 {
        SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(1.0, 0.0, 0.0))
    }

    fn project_keypoint(pose: SE3, point: [f32; 3]) -> rustslam::KeyPoint {
        let camera = CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0);
        let point = pose.transform_point(&point);
        let xy = camera
            .img_from_cam_f32(point[0], point[1], point[2])
            .expect("point projects in front of camera");
        rustslam::KeyPoint::new(xy[0], xy[1])
    }

    fn frame(id: usize) -> ImageFrame {
        ImageFrame {
            id,
            name: format!("{id}.jpg"),
            path: PathBuf::from(format!("{id}.jpg")),
            width: 100,
            height: 100,
            keypoints: vec![
                rustslam::KeyPoint::new(50.0, 50.0),
                rustslam::KeyPoint::new(55.0, 50.0),
            ],
            descriptors: Descriptors::new(),
            sift: crate::sift::SiftFeatures::default(),
            wide_descriptors: crate::wide::WideDescriptors {
                data: Vec::new(),
                dim: 0,
                count: 0,
            },
            strong_feature_indices: Vec::new(),
            colors: vec![[0, 0, 0]; 2],
        }
    }
}
