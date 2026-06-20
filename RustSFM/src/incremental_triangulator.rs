use crate::geometry::{camera_center, mean_pair_reprojection_error_with_cameras};
use crate::observation_manager::ObservationManager;
use crate::types::{ImageFrame, PairGeometry, Point3D, Reconstruction, TrackObservation};
use rustslam::SE3;
use std::collections::{HashSet, VecDeque};

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

pub struct IncrementalTriangulator<'a> {
    frames: &'a [ImageFrame],
    pairs: &'a [PairGeometry],
    reconstruction: &'a mut Reconstruction,
    observation_manager: ObservationManager,
    merge_trials: HashSet<(usize, usize)>,
    retriangulation_trials: HashSet<(usize, usize)>,
}

impl<'a> IncrementalTriangulator<'a> {
    pub fn new(
        frames: &'a [ImageFrame],
        pairs: &'a [PairGeometry],
        reconstruction: &'a mut Reconstruction,
    ) -> Self {
        let observation_manager = ObservationManager::new(frames, pairs, reconstruction);
        Self {
            frames,
            pairs,
            reconstruction,
            observation_manager,
            merge_trials: HashSet::new(),
            retriangulation_trials: HashSet::new(),
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

        let mut report = TriangulationReport::default();
        for pair in self
            .pairs
            .iter()
            .filter(|pair| pair.left == image || pair.right == image)
        {
            let other = if pair.left == image {
                pair.right
            } else {
                pair.left
            };
            if self
                .reconstruction
                .poses
                .get(other)
                .copied()
                .flatten()
                .is_none()
            {
                continue;
            }

            for match_ in &pair.inlier_matches {
                let (feature, other_feature) = if pair.left == image {
                    (match_.query_idx as usize, match_.train_idx as usize)
                } else {
                    (match_.train_idx as usize, match_.query_idx as usize)
                };
                if !self.valid_observation(image, feature)
                    || !self.valid_observation(other, other_feature)
                {
                    continue;
                }

                let point_id = self.reconstruction.observations[image][feature];
                let other_point_id = self.reconstruction.observations[other][other_feature];
                match (point_id, other_point_id) {
                    (None, Some(existing_point_id)) => {
                        if self.continue_track(
                            options,
                            existing_point_id,
                            image,
                            feature,
                            options.complete_max_reproj_error_px,
                            options.continue_max_angle_error_deg,
                        ) {
                            report.continued_observations += 1;
                        }
                    }
                    (Some(existing_point_id), None) => {
                        if self.continue_track(
                            options,
                            existing_point_id,
                            other,
                            other_feature,
                            options.complete_max_reproj_error_px,
                            options.continue_max_angle_error_deg,
                        ) {
                            report.continued_observations += 1;
                        }
                    }
                    (None, None) => {
                        if self.create_pair_track(options, pair, image, feature, other_feature) {
                            report.created_points += 1;
                        }
                    }
                    (Some(_), Some(_)) => {}
                }
            }
        }
        report
    }

    pub fn get_modified_points3d(&self) -> &HashSet<usize> {
        self.observation_manager.modified_point3d_ids()
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
            if self.retriangulation_trials.contains(&pair_key) {
                continue;
            }
            let (tri_corrs, total_corrs) = self.pair_triangulation_counts(pair);
            if total_corrs == 0 || tri_corrs as f32 / total_corrs as f32 >= options.re_min_ratio {
                continue;
            }
            if options.re_max_trials == 0 {
                continue;
            }
            self.retriangulation_trials.insert(pair_key);
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
        self.observation_manager.clear_modified_point3d_ids();
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
        if !self.observation_manager.add_observation(
            self.frames,
            self.pairs,
            self.reconstruction,
            point_id,
            TrackObservation { image, feature },
        ) {
            return false;
        }
        if !self.refine_point_from_track(point_id, options, max_reproj_error_px) {
            self.observation_manager.delete_observation(
                self.frames,
                self.pairs,
                self.reconstruction,
                image,
                feature,
            );
            return false;
        }
        self.observation_manager.mark_point3d_modified(point_id);
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
        let Some(left_pose) = self.reconstruction.poses[pair.left] else {
            return false;
        };
        let Some(right_pose) = self.reconstruction.poses[pair.right] else {
            return false;
        };
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
        if options.ignore_two_view_tracks
            && !self.pair_observation_has_registered_transitive_context(
                pair.left,
                left_feature,
                pair.right,
                right_feature,
                options.max_transitivity,
            )
        {
            return false;
        }

        let left_kp = &self.frames[pair.left].keypoints[left_feature];
        let right_kp = &self.frames[pair.right].keypoints[right_feature];
        let Some(left_xy) = self
            .reconstruction
            .camera_for_image(pair.left)
            .cam_from_img_f32(left_kp.x(), left_kp.y())
        else {
            return false;
        };
        let Some(right_xy) = self
            .reconstruction
            .camera_for_image(pair.right)
            .cam_from_img_f32(right_kp.x(), right_kp.y())
        else {
            return false;
        };
        let Some(xyz) =
            crate::two_view::triangulate_world_point(left_pose, right_pose, left_xy, right_xy)
        else {
            return false;
        };
        let Some(angle) = triangulation_angle_deg(left_pose, right_pose, xyz) else {
            return false;
        };
        if angle < options.min_angle_deg {
            return false;
        }
        if !pair_angle_consistent(
            left_pose,
            right_pose,
            left_xy,
            right_xy,
            xyz,
            options.create_max_angle_error_deg,
        ) {
            return false;
        }
        let error = mean_pair_reprojection_error_with_cameras(
            xyz,
            left_pose,
            right_pose,
            [left_kp.x(), left_kp.y()],
            [right_kp.x(), right_kp.y()],
            self.reconstruction.camera_for_image(pair.left),
            self.reconstruction.camera_for_image(pair.right),
        );
        if !error.is_finite() || error > options.merge_max_reproj_error_px {
            return false;
        }

        let point = Point3D {
            xyz,
            color: [0, 0, 0],
            error,
            track: vec![
                TrackObservation {
                    image: pair.left,
                    feature: left_feature,
                },
                TrackObservation {
                    image: pair.right,
                    feature: right_feature,
                },
            ],
        };
        self.observation_manager
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

    fn pair_observation_has_registered_transitive_context(
        &self,
        left: usize,
        left_feature: usize,
        right: usize,
        right_feature: usize,
        max_transitivity: usize,
    ) -> bool {
        if max_transitivity == 0 {
            return false;
        }
        let mut visited = HashSet::from([(left, left_feature), (right, right_feature)]);
        let mut current = VecDeque::from([(left, left_feature), (right, right_feature)]);
        for _ in 0..max_transitivity {
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
                        .is_some()
                    {
                        return true;
                    }
                    next.push_back((corr_image, corr_feature));
                }
            }
            current = next;
            if current.is_empty() {
                break;
            }
        }
        false
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
                if !self.merge_trials.insert(key) {
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
        self.observation_manager
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
        let mut best = None::<(f32, [f32; 3])>;
        for i in 0..track.len() {
            for j in i + 1..track.len() {
                let obs1 = &track[i];
                let obs2 = &track[j];
                if obs1.image == obs2.image {
                    continue;
                }
                let pose1 = self.reconstruction.poses[obs1.image]?;
                let pose2 = self.reconstruction.poses[obs2.image]?;
                let kp1 = self.frames[obs1.image].keypoints.get(obs1.feature)?;
                let kp2 = self.frames[obs2.image].keypoints.get(obs2.feature)?;
                let xy1 = self
                    .reconstruction
                    .camera_for_image(obs1.image)
                    .cam_from_img_f32(kp1.x(), kp1.y())?;
                let xy2 = self
                    .reconstruction
                    .camera_for_image(obs2.image)
                    .cam_from_img_f32(kp2.x(), kp2.y())?;
                let Some(xyz) = crate::two_view::triangulate_world_point(pose1, pose2, xy1, xy2)
                else {
                    continue;
                };
                let Some(angle) = triangulation_angle_deg(pose1, pose2, xyz) else {
                    continue;
                };
                if best
                    .as_ref()
                    .map(|(best_angle, _)| angle > *best_angle)
                    .unwrap_or(true)
                {
                    best = Some((angle, xyz));
                }
            }
        }
        best.map(|(_, xyz)| xyz)
    }
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

fn pair_angle_consistent(
    left_pose: SE3,
    right_pose: SE3,
    left_xy: [f32; 2],
    right_xy: [f32; 2],
    point: [f32; 3],
    max_angle_error_deg: f32,
) -> bool {
    if max_angle_error_deg <= 0.0 {
        return true;
    }
    let Some(left_ray) = observation_world_ray(left_pose, left_xy) else {
        return false;
    };
    let Some(right_ray) = observation_world_ray(right_pose, right_xy) else {
        return false;
    };
    let p = glam::Vec3::from_array(point);
    let Some(left_to_point) = (p - camera_center(left_pose)).try_normalize() else {
        return false;
    };
    let Some(right_to_point) = (p - camera_center(right_pose)).try_normalize() else {
        return false;
    };
    angular_error_deg(left_ray, left_to_point).max(angular_error_deg(right_ray, right_to_point))
        <= max_angle_error_deg
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

        let mut triangulator = IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction);
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

        let mut triangulator = IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction);
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

        let mut triangulator = IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction);
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

        let mut triangulator = IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction);
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
        assert_eq!(triangulator.reconstruction.points[0].track.len(), 3);
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

        let mut triangulator = IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction);
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

        let mut triangulator = IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction);
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

        let mut triangulator = IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction);
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
