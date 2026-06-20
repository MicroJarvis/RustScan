use crate::correspondence_graph::{
    CorrespondenceGraph, FeatureMatch, ImageId, Point2DIdx, TwoViewGeometryRecord,
};
use crate::geometry::camera_center;
use crate::triangulation_estimator::{
    calculate_angular_reprojection_error, estimate_triangulation, EstimateTriangulationOptions,
    ResidualType,
};
use crate::observation_manager::ObservationManager;
use crate::types::{CameraModel, ImageFrame, PairGeometry, Point3D, Reconstruction, TrackObservation};
use nalgebra::{Matrix3x4, Vector3};
use rustslam::SE3;
use std::collections::{HashMap, HashSet, VecDeque};

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
        }
    }
}

impl IncrementalTriangulatorOptions {
    pub fn from_mapper_threshold(max_reprojection_error_px: f32) -> Self {
        Self {
            merge_max_reproj_error_px: max_reprojection_error_px,
            complete_max_reproj_error_px: max_reprojection_error_px,
            ignore_two_view_tracks: false,
            ..Self::default()
        }
    }

    pub fn from_mapper_config_values(
        max_reprojection_error_px: f32,
        min_focal_length_ratio: f64,
        max_focal_length_ratio: f64,
        max_extra_param: f64,
        random_seed: i32,
    ) -> Self {
        Self {
            merge_max_reproj_error_px: max_reprojection_error_px,
            complete_max_reproj_error_px: max_reprojection_error_px,
            ignore_two_view_tracks: false,
            min_focal_length_ratio,
            max_focal_length_ratio,
            max_extra_param,
            random_seed,
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
    correspondence_graph: CorrespondenceGraph,
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
            correspondence_graph: build_correspondence_graph_from_pairs(frames, pairs),
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
        self.correspondence_graph = build_correspondence_graph_from_pairs(frames, pairs);
        self.merge_trials.clear();
        self.retriangulation_trials.clear();
        self.camera_has_bogus_params.clear();
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
                if self.create_from_correspondences(options, &corrs_data) {
                    report.created_points += 1;
                }
            } else {
                if self.continue_reference_observation(options, image, feature, &corrs_data) {
                    report.continued_observations += 1;
                }
                corrs_data.push((image, feature));
                if self.create_from_correspondences(options, &corrs_data) {
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
                        if self.continue_track(
                            options,
                            point_id,
                            pair.right,
                            right_feature,
                            options.complete_max_reproj_error_px,
                            options.re_max_angle_error_deg,
                        ) {
                            total += 1;
                        }
                    }
                    (None, Some(point_id)) => {
                        if self.continue_track(
                            options,
                            point_id,
                            pair.left,
                            left_feature,
                            options.complete_max_reproj_error_px,
                            options.re_max_angle_error_deg,
                        ) {
                            total += 1;
                        }
                    }
                    (None, None) => {
                        if self.create_pair_track(
                            options,
                            pair,
                            pair.left,
                            left_feature,
                            right_feature,
                        ) {
                            total += 2;
                        }
                    }
                }
            }
        }
        total
    }

    pub fn clear_modified_points3d(&mut self) {
        self.state.observation_manager_mut().clear_modified_point3d_ids();
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
            .correspondence_graph
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
        let Some(ref_cam_ray) = normalized_camera_ray(ref_camera, ref_keypoint.x(), ref_keypoint.y())
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
            let angle_error = calculate_angular_reprojection_error(
                &ref_cam_ray,
                &point_xyz,
                &ref_cam_from_world,
            );
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
    ) -> bool {
        let create_corrs: Vec<(usize, usize)> = corrs_data
            .iter()
            .copied()
            .filter(|&(image, feature)| {
                self.reconstruction.observations[image][feature].is_none()
            })
            .collect();
        if create_corrs.len() < 2 {
            return false;
        }
        if options.ignore_two_view_tracks && create_corrs.len() == 2 {
            let (image, feature) = create_corrs[0];
            if self
                .state
                .correspondence_graph
                .is_two_view_observation(image as ImageId, feature as Point2DIdx)
                .unwrap_or(false)
            {
                return false;
            }
        }

        let mut points = Vec::with_capacity(create_corrs.len());
        let mut cams_from_world = Vec::with_capacity(create_corrs.len());
        let mut cameras = Vec::with_capacity(create_corrs.len());
        for &(obs_image, obs_feature) in &create_corrs {
            let Some(pose) = self.reconstruction.poses[obs_image] else {
                return false;
            };
            let kp = &self.frames[obs_image].keypoints[obs_feature];
            points.push(nalgebra::Vector2::new(kp.x() as f64, kp.y() as f64));
            cams_from_world.push(se3_to_matrix3x4(pose));
            cameras.push(self.reconstruction.camera_for_image(obs_image));
        }

        let mut est_options = create_estimate_options(options);
        if points.len() <= EXHAUSTIVE_TRIANGULATION_SAMPLING_THRESHOLD {
            est_options.min_num_trials = n_choose_k(points.len(), 2);
        }
        let Some((inlier_mask, xyz)) =
            estimate_triangulation(&est_options, &points, &cams_from_world, &cameras)
        else {
            return false;
        };

        let track: Vec<TrackObservation> = create_corrs
            .iter()
            .zip(inlier_mask.iter())
            .filter_map(|(&(obs_image, obs_feature), &is_inlier)| {
                is_inlier.then_some(TrackObservation {
                    image: obs_image,
                    feature: obs_feature,
                })
            })
            .collect();
        let track_length = track.len();
        if track_length < 2 {
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
        if self
            .state
            .observation_manager_mut()
            .add_point3d(self.frames, self.pairs, self.reconstruction, point)
            .is_none()
        {
            return false;
        }

        if create_corrs.len().saturating_sub(track_length) >= MIN_RECURSIVE_CREATE_TRACK_LENGTH {
            let _ = self.create_from_correspondences(options, corrs_data);
        }
        true
    }

    fn continue_track(
        &mut self,
        options: &IncrementalTriangulatorOptions,
        point_id: usize,
        image: usize,
        feature: usize,
        max_reproj_error_px: f32,
        max_angle_error_deg: f32,
    ) -> bool {
        if point_id >= self.reconstruction.points.len()
            || self.reconstruction.observations[image][feature].is_some()
            || self.reconstruction.points[point_id]
                .track
                .iter()
                .any(|obs| obs.image == image)
        {
            return false;
        }
        let Some(pose) = self.reconstruction.poses[image] else {
            return false;
        };
        let keypoint = &self.frames[image].keypoints[feature];
        let error = crate::geometry::reprojection_error_px(
            self.reconstruction.points[point_id].xyz,
            pose,
            [keypoint.x(), keypoint.y()],
            self.reconstruction.camera_for_image(image),
        );
        if !error.is_finite() || error > max_reproj_error_px {
            return false;
        }
        if !self.track_observation_angles_consistent(point_id, image, feature, max_angle_error_deg)
        {
            return false;
        }
        if !self.state.observation_manager_mut().add_observation(
            self.frames,
            self.pairs,
            self.reconstruction,
            point_id,
            TrackObservation { image, feature },
        ) {
            return false;
        }
        if !self.refine_point_from_track(point_id, options, max_reproj_error_px) {
            self.state.observation_manager_mut().delete_observation(
                self.frames,
                self.pairs,
                self.reconstruction,
                image,
                feature,
            );
            return false;
        }
        self.state.observation_manager_mut().mark_point3d_modified(point_id);
        true
    }

    fn create_pair_track(
        &mut self,
        options: &IncrementalTriangulatorOptions,
        pair: &PairGeometry,
        image: usize,
        feature: usize,
        other_feature: usize,
    ) -> bool {
        if self.reconstruction.poses[pair.left].is_none()
            || self.reconstruction.poses[pair.right].is_none()
        {
            return false;
        }
        let left_feature = if pair.left == image {
            feature
        } else {
            other_feature
        };
        let right_feature = if pair.right == image {
            feature
        } else {
            other_feature
        };
        if self.reconstruction.observations[pair.left][left_feature].is_some()
            || self.reconstruction.observations[pair.right][right_feature].is_some()
        {
            return false;
        }

        let corrs = self.gather_create_observations(
            pair.left,
            left_feature,
            pair.right,
            right_feature,
            options.max_transitivity,
        );
        self.create_from_correspondences(options, &corrs)
    }

    /// Collect the seed observation pair plus transitively-corresponding
    /// observations in registered images that are not yet assigned to a 3D
    /// point, bounded by `max_transitivity`. One observation per image; the seed
    /// left/right observations are always the first two returned entries.
    fn gather_create_observations(
        &self,
        left: usize,
        left_feature: usize,
        right: usize,
        right_feature: usize,
        max_transitivity: usize,
    ) -> Vec<(usize, usize)> {
        let mut result = vec![(left, left_feature), (right, right_feature)];
        let mut images_used: HashSet<usize> = HashSet::from([left, right]);
        let mut visited: HashSet<(usize, usize)> =
            HashSet::from([(left, left_feature), (right, right_feature)]);
        let mut current: VecDeque<(usize, usize)> =
            VecDeque::from([(left, left_feature), (right, right_feature)]);
        for _ in 0..max_transitivity {
            if current.is_empty() {
                break;
            }
            let mut next = VecDeque::new();
            while let Some((image, feature)) = current.pop_front() {
                for (corr_image, corr_feature) in self.corresponding_features(image, feature) {
                    if !visited.insert((corr_image, corr_feature)) {
                        continue;
                    }
                    next.push_back((corr_image, corr_feature));
                    if images_used.contains(&corr_image)
                        || self
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
                    images_used.insert(corr_image);
                    result.push((corr_image, corr_feature));
                }
            }
            current = next;
        }
        result
    }

    fn complete_track(
        &mut self,
        options: &IncrementalTriangulatorOptions,
        point_id: usize,
    ) -> usize {
        if point_id >= self.reconstruction.points.len() {
            return 0;
        }
        let mut completed = 0usize;
        let mut visited = self.reconstruction.points[point_id]
            .track
            .iter()
            .map(|obs| (obs.image, obs.feature))
            .collect::<HashSet<_>>();
        let mut current = self.reconstruction.points[point_id]
            .track
            .iter()
            .map(|obs| (obs.image, obs.feature))
            .collect::<VecDeque<_>>();

        for _ in 0..options.complete_max_transitivity {
            if current.is_empty() {
                break;
            }
            let mut next = VecDeque::new();
            while let Some((image, feature)) = current.pop_front() {
                for (corr_image, corr_feature) in self.corresponding_features(image, feature) {
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
                    if self.continue_track(
                        options,
                        point_id,
                        corr_image,
                        corr_feature,
                        options.complete_max_reproj_error_px,
                        options.continue_max_angle_error_deg,
                    ) {
                        completed += 1;
                        next.push_back((corr_image, corr_feature));
                    }
                }
            }
            current = next;
        }
        completed
    }

    fn corresponding_features(&self, image: usize, feature: usize) -> Vec<(usize, usize)> {
        let mut correspondences = Vec::new();
        for pair in self
            .pairs
            .iter()
            .filter(|pair| pair.left == image || pair.right == image)
        {
            for match_ in &pair.inlier_matches {
                if pair.left == image && match_.query_idx as usize == feature {
                    correspondences.push((pair.right, match_.train_idx as usize));
                } else if pair.right == image && match_.train_idx as usize == feature {
                    correspondences.push((pair.left, match_.query_idx as usize));
                }
            }
        }
        correspondences
    }

    fn merge_track(&mut self, options: &IncrementalTriangulatorOptions, point_id: usize) -> usize {
        if point_id >= self.reconstruction.points.len() {
            return 0;
        }
        let track = self.reconstruction.points[point_id].track.clone();
        for obs in track {
            for (corr_image, corr_feature) in self.corresponding_features(obs.image, obs.feature) {
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
        if tracks_conflict_by_image(&keep.track, &remove.track) {
            return false;
        }
        let mut merged_track = keep.track.clone();
        merged_track.extend(remove.track.iter().cloned());
        let Some(merged_xyz) = self.triangulate_track_observations(&merged_track) else {
            return false;
        };
        if !track_has_min_triangulation_angle(
            merged_xyz,
            &merged_track,
            self.reconstruction,
            options.min_angle_deg,
        ) {
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
        self.state.observation_manager_mut()
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
            .is_some()
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

    fn track_observation_angles_consistent(
        &self,
        point_id: usize,
        image: usize,
        feature: usize,
        max_angle_error_deg: f32,
    ) -> bool {
        if max_angle_error_deg <= 0.0 {
            return true;
        }
        let Some(pose) = self.reconstruction.poses[image] else {
            return false;
        };
        let keypoint = &self.frames[image].keypoints[feature];
        let Some(xy) = self
            .reconstruction
            .camera_for_image(image)
            .cam_from_img_f32(keypoint.x(), keypoint.y())
        else {
            return false;
        };
        let point = self.reconstruction.points[point_id].xyz;
        let center = camera_center(pose);
        let ray_to_point = (glam::Vec3::from_array(point) - center).try_normalize();
        let ray_from_obs = observation_world_ray(pose, xy);
        match (ray_to_point, ray_from_obs) {
            (Some(a), Some(b)) => angular_error_deg(a, b) <= max_angle_error_deg,
            _ => false,
        }
    }

    fn refine_point_from_track(
        &mut self,
        point_id: usize,
        options: &IncrementalTriangulatorOptions,
        max_reproj_error_px: f32,
    ) -> bool {
        if point_id >= self.reconstruction.points.len() {
            return false;
        }
        let track = self.reconstruction.points[point_id].track.clone();
        let Some(xyz) = self.triangulate_track_observations(&track) else {
            return false;
        };
        if !track_has_min_triangulation_angle(
            xyz,
            &track,
            self.reconstruction,
            options.min_angle_deg,
        ) {
            return false;
        }
        if !self.track_reprojects(xyz, &track, max_reproj_error_px) {
            return false;
        }
        let Some(error) =
            mean_track_reprojection_error(xyz, &track, self.frames, self.reconstruction)
        else {
            return false;
        };
        self.reconstruction.points[point_id].xyz = xyz;
        self.reconstruction.points[point_id].error = error;
        true
    }

    fn triangulate_track_observations(&self, track: &[TrackObservation]) -> Option<[f32; 3]> {
        // Gather one normalized observation per distinct image and run COLMAP's
        // multi-view DLT (`TriangulateMultiViewPoint`) over all of them, instead
        // of triangulating only the widest-baseline pair.
        let mut cams_from_world = Vec::with_capacity(track.len());
        let mut cam_points = Vec::with_capacity(track.len());
        let mut seen_images = HashSet::new();
        for obs in track {
            if !seen_images.insert(obs.image) {
                continue;
            }
            let pose = self.reconstruction.poses[obs.image]?;
            let kp = self.frames[obs.image].keypoints.get(obs.feature)?;
            let xy = self
                .reconstruction
                .camera_for_image(obs.image)
                .cam_from_img_f32(kp.x(), kp.y())?;
            cams_from_world.push(se3_to_matrix3x4(pose));
            cam_points.push(nalgebra::Vector2::new(xy[0] as f64, xy[1] as f64));
        }
        if cam_points.len() < 2 {
            return None;
        }
        let xyz = crate::triangulation::triangulate_multi_view_point(&cams_from_world, &cam_points)?;
        xyz.iter()
            .all(|v| v.is_finite())
            .then_some([xyz[0] as f32, xyz[1] as f32, xyz[2] as f32])
    }
}

fn build_correspondence_graph_from_pairs(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
) -> CorrespondenceGraph {
    let mut graph = CorrespondenceGraph::new();
    for (idx, frame) in frames.iter().enumerate() {
        let _ = graph.add_image(idx as ImageId, frame.keypoints.len());
    }
    for pair in pairs {
        if pair.left >= frames.len() || pair.right >= frames.len() {
            continue;
        }
        let matches = pair
            .inlier_matches
            .iter()
            .map(FeatureMatch::from)
            .collect::<Vec<_>>();
        let _ = graph.add_two_view_geometry(
            pair.left as ImageId,
            pair.right as ImageId,
            TwoViewGeometryRecord::with_inlier_matches(matches),
        );
    }
    let _ = graph.finalize();
    graph
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

fn tracks_conflict_by_image(left: &[TrackObservation], right: &[TrackObservation]) -> bool {
    let mut images = left.iter().map(|obs| obs.image).collect::<HashSet<_>>();
    right.iter().any(|obs| !images.insert(obs.image))
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

fn triangulation_angle_deg(left_pose: SE3, right_pose: SE3, point: [f32; 3]) -> Option<f32> {
    let c1 = camera_center(left_pose);
    let c2 = camera_center(right_pose);
    let p = glam::Vec3::from_array(point);
    let v1 = (p - c1).try_normalize()?;
    let v2 = (p - c2).try_normalize()?;
    Some(v1.dot(v2).abs().clamp(-1.0, 1.0).acos().to_degrees())
}

/// Build COLMAP `EstimateTriangulationOptions` from the incremental
/// triangulator options, matching `IncrementalTriangulator::Create`'s
/// configuration (angular residual; `create_max_angle_error` as the inlier
/// threshold; `min_angle` as the minimum triangulation angle).
fn create_estimate_options(options: &IncrementalTriangulatorOptions) -> EstimateTriangulationOptions {
    EstimateTriangulationOptions {
        min_tri_angle: (options.min_angle_deg as f64).to_radians(),
        residual_type: ResidualType::AngularError,
        max_error: (options.create_max_angle_error_deg as f64).to_radians(),
        confidence: 0.9999,
        min_inlier_ratio: 0.02,
        min_num_trials: 0,
        max_num_trials: 10000,
    }
}

fn track_has_min_triangulation_angle(
    point: [f32; 3],
    track: &[TrackObservation],
    reconstruction: &Reconstruction,
    min_angle_deg: f32,
) -> bool {
    if min_angle_deg <= 0.0 {
        return true;
    }
    let mut best = 0.0f32;
    for i in 0..track.len() {
        for j in i + 1..track.len() {
            let Some(pose_i) = reconstruction.poses[track[i].image] else {
                continue;
            };
            let Some(pose_j) = reconstruction.poses[track[j].image] else {
                continue;
            };
            if let Some(angle) = triangulation_angle_deg(pose_i, pose_j, point) {
                best = best.max(angle);
            }
        }
    }
    best >= min_angle_deg
}

fn observation_world_ray(pose: SE3, xy: [f32; 2]) -> Option<glam::Vec3> {
    let q = pose.quaternion();
    let rotation = glam::Quat::from_xyzw(q[0], q[1], q[2], q[3]).normalize();
    (rotation.inverse() * glam::Vec3::new(xy[0], xy[1], 1.0)).try_normalize()
}

fn angular_error_deg(a: glam::Vec3, b: glam::Vec3) -> f32 {
    a.dot(b).clamp(-1.0, 1.0).acos().to_degrees()
}

fn average_color(observations: &[TrackObservation], frames: &[ImageFrame]) -> [u8; 3] {
    let mut color = [0usize; 3];
    for obs in observations {
        let c = frames[obs.image].colors[obs.feature];
        color[0] += c[0] as usize;
        color[1] += c[1] as usize;
        color[2] += c[2] as usize;
    }
    let n = observations.len().max(1);
    [
        (color[0] / n) as u8,
        (color[1] / n) as u8,
        (color[2] / n) as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CameraModel, ImageFrame, PairGeometry, Reconstruction};
    use rustslam::{Descriptors, Match};
    use std::path::PathBuf;

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
    fn merge_tracks_rejects_same_image_track_conflicts() {
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

        assert_eq!(merged, 0);
        assert_eq!(triangulator.reconstruction.points.len(), 2);
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
