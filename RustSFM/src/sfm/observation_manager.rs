use crate::correspondence_graph::{build_correspondence_graph_from_pairs, CorrespondenceGraph};
use crate::types::{ImageFrame, PairGeometry, Point3D, Reconstruction, TrackObservation};
use crate::visibility_pyramid::VisibilityPyramid;
use rustslam::SE3;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ImagePairStat {
    pub num_tri_corrs: usize,
    pub num_total_corrs: usize,
}

#[derive(Debug, Clone)]
pub struct ImageStat {
    pub num_observations: usize,
    pub num_correspondences: usize,
    pub num_visible_correspondences: usize,
    pub num_visible_points3d: usize,
    visibility_pyramid: VisibilityPyramid,
}

impl Default for ImageStat {
    fn default() -> Self {
        Self {
            num_observations: 0,
            num_correspondences: 0,
            num_visible_correspondences: 0,
            num_visible_points3d: 0,
            visibility_pyramid: VisibilityPyramid::default(),
        }
    }
}

impl ImageStat {
    pub fn point3d_visibility_score(&self) -> usize {
        self.visibility_pyramid.score()
    }

    fn init_for_frame(frame: Option<&ImageFrame>) -> Self {
        let (width, height) = frame
            .map(|image| (image.width as usize, image.height as usize))
            .unwrap_or((0, 0));
        Self {
            visibility_pyramid: VisibilityPyramid::new(width, height),
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ObservationManager {
    image_stats: Vec<ImageStat>,
    image_pair_stats: HashMap<(usize, usize), ImagePairStat>,
    point3d_correspondence_counts: Vec<Vec<usize>>,
    modified_point3d_ids: HashSet<usize>,
    correspondence_graph: Option<CorrespondenceGraph>,
}

impl ObservationManager {
    pub fn new(
        frames: &[ImageFrame],
        pairs: &[PairGeometry],
        reconstruction: &Reconstruction,
    ) -> Self {
        let mut manager = Self::default();
        manager.install_correspondence_graph(build_correspondence_graph_from_pairs(frames, pairs));
        manager.rebuild(frames, pairs, reconstruction);
        manager
    }

    pub fn install_correspondence_graph(&mut self, graph: CorrespondenceGraph) {
        self.correspondence_graph = Some(graph);
    }

    pub fn correspondence_graph(&self) -> Option<&CorrespondenceGraph> {
        self.correspondence_graph.as_ref()
    }

    pub fn rebuild(
        &mut self,
        frames: &[ImageFrame],
        pairs: &[PairGeometry],
        reconstruction: &Reconstruction,
    ) {
        let modified_point3d_ids = std::mem::take(&mut self.modified_point3d_ids);
        let mut image_stats = frames
            .iter()
            .map(|frame| ImageStat::init_for_frame(Some(frame)))
            .collect::<Vec<_>>();
        let mut observed_features = frames
            .iter()
            .map(|frame| vec![false; frame.keypoints.len()])
            .collect::<Vec<_>>();
        let mut point3d_correspondence_counts = frames
            .iter()
            .map(|frame| vec![0usize; frame.keypoints.len()])
            .collect::<Vec<_>>();
        let mut image_pair_stats = HashMap::new();

        for pair in pairs {
            if pair.left >= frames.len() || pair.right >= frames.len() {
                continue;
            }
            let key = ordered_pair(pair.left, pair.right);
            let mut stat = ImagePairStat {
                num_total_corrs: pair.inlier_matches.len(),
                num_tri_corrs: 0,
            };
            for m in &pair.inlier_matches {
                let left_feature = m.query_idx as usize;
                let right_feature = m.train_idx as usize;
                if left_feature >= frames[pair.left].keypoints.len()
                    || right_feature >= frames[pair.right].keypoints.len()
                {
                    continue;
                }
                mark_observation(
                    &mut image_stats,
                    &mut observed_features,
                    pair.left,
                    left_feature,
                );
                mark_observation(
                    &mut image_stats,
                    &mut observed_features,
                    pair.right,
                    right_feature,
                );
                image_stats[pair.left].num_correspondences += 1;
                image_stats[pair.right].num_correspondences += 1;

                let left_registered = reconstruction
                    .poses
                    .get(pair.left)
                    .copied()
                    .flatten()
                    .is_some();
                let right_registered = reconstruction
                    .poses
                    .get(pair.right)
                    .copied()
                    .flatten()
                    .is_some();
                let left_point = reconstruction
                    .observations
                    .get(pair.left)
                    .and_then(|obs| obs.get(left_feature))
                    .copied()
                    .flatten();
                let right_point = reconstruction
                    .observations
                    .get(pair.right)
                    .and_then(|obs| obs.get(right_feature))
                    .copied()
                    .flatten();
                if left_registered {
                    image_stats[pair.right].num_visible_correspondences += 1;
                }
                if right_registered {
                    image_stats[pair.left].num_visible_correspondences += 1;
                }
                if left_point.is_some() && left_registered {
                    increment_correspondence_has_point3d(
                        &mut image_stats,
                        &mut point3d_correspondence_counts,
                        frames,
                        pair.right,
                        right_feature,
                    );
                }
                if right_point.is_some() && right_registered {
                    increment_correspondence_has_point3d(
                        &mut image_stats,
                        &mut point3d_correspondence_counts,
                        frames,
                        pair.left,
                        left_feature,
                    );
                }
                if left_point.is_some() && left_point == right_point {
                    stat.num_tri_corrs += 1;
                }
            }
            image_pair_stats.insert(key, stat);
        }

        self.image_stats = image_stats;
        self.image_pair_stats = image_pair_stats;
        self.point3d_correspondence_counts = point3d_correspondence_counts;
        self.modified_point3d_ids = modified_point3d_ids;
    }

    pub fn register_image(
        &mut self,
        frames: &[ImageFrame],
        pairs: &[PairGeometry],
        reconstruction: &mut Reconstruction,
        image: usize,
        pose: SE3,
    ) -> bool {
        self.register_frame_for_image(frames, pairs, reconstruction, image, pose)
    }

    pub fn register_frame_for_image(
        &mut self,
        frames: &[ImageFrame],
        pairs: &[PairGeometry],
        reconstruction: &mut Reconstruction,
        image: usize,
        pose: SE3,
    ) -> bool {
        if image >= reconstruction.poses.len() {
            return false;
        }
        let frame_images = if let Some(frame_registration) =
            reconstruction.frame_registration_poses_for_image(image, pose)
        {
            if frame_registration.image_poses.is_empty() {
                return false;
            }
            for (frame_image, frame_pose) in frame_registration.image_poses {
                if let Some(slot) = reconstruction.poses.get_mut(frame_image) {
                    *slot = Some(frame_pose);
                }
            }
            if let Some(frame) = reconstruction.frames.get_mut(frame_registration.frame_idx) {
                frame.rig_from_world =
                    crate::types::Rigid3::from_se3(frame_registration.rig_from_world);
            }
            reconstruction.image_indices_for_registration_unit(image)
        } else if let Some(slot) = reconstruction.poses.get_mut(image) {
            *slot = Some(pose);
            vec![image]
        } else {
            return false;
        };

        if let Some(graph) = self.correspondence_graph.clone() {
            self.propagate_visible_correspondences_on_register(
                frames,
                &graph,
                reconstruction,
                &frame_images,
            );
        } else {
            self.rebuild(frames, pairs, reconstruction);
        }
        true
    }

    pub fn deregister_image(
        &mut self,
        frames: &[ImageFrame],
        pairs: &[PairGeometry],
        reconstruction: &mut Reconstruction,
        image: usize,
    ) -> bool {
        self.deregister_frame_for_image(frames, pairs, reconstruction, image)
    }

    pub fn deregister_frame_for_image(
        &mut self,
        frames: &[ImageFrame],
        pairs: &[PairGeometry],
        reconstruction: &mut Reconstruction,
        image: usize,
    ) -> bool {
        if image >= reconstruction.poses.len() {
            return false;
        }
        let frame_images = reconstruction.image_indices_for_registration_unit(image);
        if frame_images.iter().all(|&frame_image| {
            reconstruction
                .poses
                .get(frame_image)
                .is_none_or(|pose| pose.is_none())
        }) {
            return false;
        }

        if let Some(graph) = self.correspondence_graph.clone() {
            for &frame_image in &frame_images {
                let num_features = frames
                    .get(frame_image)
                    .map(|frame| frame.keypoints.len())
                    .unwrap_or(0);
                for feature in 0..num_features {
                    for (corr_image, _corr_feature) in
                        correspondences_for(&graph, frame_image, feature)
                    {
                        let stats = &mut self.image_stats[corr_image];
                        debug_assert!(stats.num_visible_correspondences > 0);
                        stats.num_visible_correspondences =
                            stats.num_visible_correspondences.saturating_sub(1);
                    }
                    if reconstruction.observations[frame_image][feature].is_some() {
                        self.delete_observation_internal(
                            frames,
                            pairs,
                            reconstruction,
                            frame_image,
                            feature,
                        );
                    }
                }
            }
        } else {
            for frame_image in frame_images.clone() {
                let features = reconstruction
                    .observations
                    .get(frame_image)
                    .map(|observations| {
                        observations
                            .iter()
                            .enumerate()
                            .filter_map(|(feature, point_id)| point_id.is_some().then_some(feature))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                for feature in features {
                    self.delete_observation_internal(
                        frames,
                        pairs,
                        reconstruction,
                        frame_image,
                        feature,
                    );
                }
            }
        }

        for frame_image in frame_images {
            if let Some(slot) = reconstruction.poses.get_mut(frame_image) {
                *slot = None;
            }
        }
        if let Some(frame_idx) = reconstruction.frame_index_for_image(image) {
            if let Some(frame) = reconstruction.frames.get_mut(frame_idx) {
                frame.rig_from_world = crate::types::Rigid3::identity();
            }
        }
        if self.correspondence_graph.is_none() {
            self.rebuild(frames, pairs, reconstruction);
        }
        true
    }

    pub fn add_point3d(
        &mut self,
        frames: &[ImageFrame],
        pairs: &[PairGeometry],
        reconstruction: &mut Reconstruction,
        point: Point3D,
    ) -> Option<usize> {
        if point.track.is_empty()
            || !track_observations_are_valid(reconstruction, &point.track)
            || point
                .track
                .iter()
                .any(|obs| reconstruction.observations[obs.image][obs.feature].is_some())
        {
            return None;
        }

        ensure_point_id_table(reconstruction);
        let point_id = reconstruction.points.len();
        for obs in &point.track {
            reconstruction.observations[obs.image][obs.feature] = Some(point_id);
        }
        reconstruction
            .point_ids
            .push(next_point3d_id(reconstruction));
        reconstruction.points.push(point);
        self.mark_point3d_modified(point_id);

        if let Some(graph) = self.correspondence_graph.clone() {
            let track = reconstruction.points[point_id].track.clone();
            for obs in track {
                self.set_observation_as_triangulated(
                    &graph,
                    frames,
                    reconstruction,
                    obs.image,
                    obs.feature,
                    false,
                );
            }
        } else {
            self.rebuild(frames, pairs, reconstruction);
        }
        Some(point_id)
    }

    pub fn add_observation(
        &mut self,
        frames: &[ImageFrame],
        pairs: &[PairGeometry],
        reconstruction: &mut Reconstruction,
        point_id: usize,
        observation: TrackObservation,
    ) -> bool {
        if point_id >= reconstruction.points.len()
            || !track_observation_is_valid(reconstruction, &observation)
            || reconstruction.observations[observation.image][observation.feature].is_some()
        {
            return false;
        }

        reconstruction.observations[observation.image][observation.feature] = Some(point_id);
        let obs_image = observation.image;
        let obs_feature = observation.feature;
        reconstruction.points[point_id].track.push(observation);
        self.mark_point3d_modified(point_id);

        if let Some(graph) = self.correspondence_graph.clone() {
            self.set_observation_as_triangulated(
                &graph,
                frames,
                reconstruction,
                obs_image,
                obs_feature,
                true,
            );
        } else {
            self.rebuild(frames, pairs, reconstruction);
        }
        true
    }

    pub fn delete_point3d(
        &mut self,
        frames: &[ImageFrame],
        pairs: &[PairGeometry],
        reconstruction: &mut Reconstruction,
        point_id: usize,
    ) -> bool {
        if point_id >= reconstruction.points.len() {
            return false;
        }

        if let Some(graph) = self.correspondence_graph.clone() {
            let track = reconstruction.points[point_id].track.clone();
            for obs in track {
                self.reset_tri_observations(
                    &graph,
                    frames,
                    reconstruction,
                    obs.image,
                    obs.feature,
                    true,
                );
            }
            if !self.delete_point3d_internal(reconstruction, point_id) {
                return false;
            }
        } else if !self.delete_point3d_internal(reconstruction, point_id) {
            return false;
        } else {
            self.rebuild(frames, pairs, reconstruction);
            return true;
        }
        true
    }

    pub fn delete_observation(
        &mut self,
        frames: &[ImageFrame],
        pairs: &[PairGeometry],
        reconstruction: &mut Reconstruction,
        image: usize,
        feature: usize,
    ) -> bool {
        self.delete_observation_internal(frames, pairs, reconstruction, image, feature)
    }

    fn delete_observation_internal(
        &mut self,
        frames: &[ImageFrame],
        pairs: &[PairGeometry],
        reconstruction: &mut Reconstruction,
        image: usize,
        feature: usize,
    ) -> bool {
        let Some(point_id) = reconstruction
            .observations
            .get(image)
            .and_then(|observations| observations.get(feature))
            .copied()
            .flatten()
        else {
            return false;
        };
        if point_id >= reconstruction.points.len() {
            if let Some(observations) = reconstruction.observations.get_mut(image) {
                if let Some(observation) = observations.get_mut(feature) {
                    *observation = None;
                }
            }
            if self.correspondence_graph.is_none() {
                self.rebuild(frames, pairs, reconstruction);
            }
            return true;
        }

        if reconstruction.points[point_id].track.len() <= 2 {
            return self.delete_point3d(frames, pairs, reconstruction, point_id);
        }

        if let Some(graph) = self.correspondence_graph.clone() {
            self.reset_tri_observations(&graph, frames, reconstruction, image, feature, false);
        }

        reconstruction.observations[image][feature] = None;
        if let Some(pos) = reconstruction.points[point_id]
            .track
            .iter()
            .position(|obs| obs.image == image && obs.feature == feature)
        {
            reconstruction.points[point_id].track.remove(pos);
        }
        self.mark_point3d_modified(point_id);
        if self.correspondence_graph.is_none() {
            self.rebuild(frames, pairs, reconstruction);
        }
        true
    }

    pub fn merge_points3d(
        &mut self,
        frames: &[ImageFrame],
        pairs: &[PairGeometry],
        reconstruction: &mut Reconstruction,
        point_id1: usize,
        point_id2: usize,
        merged_point: Point3D,
    ) -> Option<usize> {
        if point_id1 == point_id2
            || point_id1 >= reconstruction.points.len()
            || point_id2 >= reconstruction.points.len()
            || !track_observations_are_valid(reconstruction, &merged_point.track)
        {
            return None;
        }

        for obs in &merged_point.track {
            let assigned = reconstruction.observations[obs.image][obs.feature];
            if assigned.is_some_and(|point_id| point_id != point_id1 && point_id != point_id2) {
                return None;
            }
        }

        if let Some(graph) = self.correspondence_graph.clone() {
            let track1 = reconstruction.points[point_id1].track.clone();
            for obs in track1 {
                self.reset_tri_observations(
                    &graph,
                    frames,
                    reconstruction,
                    obs.image,
                    obs.feature,
                    true,
                );
            }
            let track2 = reconstruction.points[point_id2].track.clone();
            for obs in track2 {
                self.reset_tri_observations(
                    &graph,
                    frames,
                    reconstruction,
                    obs.image,
                    obs.feature,
                    true,
                );
            }
        }

        let (keep_id, remove_id) = ordered_pair(point_id1, point_id2);
        ensure_point_id_table(reconstruction);
        reconstruction.points[keep_id] = merged_point;
        for obs in &reconstruction.points[keep_id].track {
            reconstruction.observations[obs.image][obs.feature] = Some(keep_id);
        }
        if !self.delete_point3d_internal(reconstruction, remove_id) {
            return None;
        }
        self.mark_point3d_modified(keep_id);

        if let Some(graph) = self.correspondence_graph.clone() {
            let track = reconstruction.points[keep_id].track.clone();
            for obs in track {
                self.set_observation_as_triangulated(
                    &graph,
                    frames,
                    reconstruction,
                    obs.image,
                    obs.feature,
                    false,
                );
            }
        } else {
            self.rebuild(frames, pairs, reconstruction);
        }
        Some(keep_id)
    }

    fn propagate_visible_correspondences_on_register(
        &mut self,
        frames: &[ImageFrame],
        graph: &CorrespondenceGraph,
        reconstruction: &Reconstruction,
        frame_images: &[usize],
    ) {
        for &image in frame_images {
            let num_features = frames
                .get(image)
                .map(|frame| frame.keypoints.len())
                .unwrap_or(0);
            for feature in 0..num_features {
                for (corr_image, corr_feature) in correspondences_for(graph, image, feature) {
                    if corr_image < self.image_stats.len() {
                        self.image_stats[corr_image].num_visible_correspondences += 1;
                    }
                    if reconstruction.observations[image][feature].is_some() {
                        increment_correspondence_has_point3d(
                            &mut self.image_stats,
                            &mut self.point3d_correspondence_counts,
                            frames,
                            corr_image,
                            corr_feature,
                        );
                    }
                }
            }
        }
    }

    fn set_observation_as_triangulated(
        &mut self,
        graph: &CorrespondenceGraph,
        frames: &[ImageFrame],
        reconstruction: &Reconstruction,
        image: usize,
        feature: usize,
        is_continued_point3d: bool,
    ) {
        if reconstruction.poses.get(image).copied().flatten().is_none() {
            return;
        }
        let Some(point_id) = reconstruction.observations[image][feature] else {
            return;
        };

        for (corr_image, corr_feature) in correspondences_for(graph, image, feature) {
            increment_correspondence_has_point3d(
                &mut self.image_stats,
                &mut self.point3d_correspondence_counts,
                frames,
                corr_image,
                corr_feature,
            );

            let corr_point = reconstruction.observations[corr_image][corr_feature];
            if corr_point == Some(point_id) && (is_continued_point3d || image < corr_image) {
                let key = ordered_pair(image, corr_image);
                let num_total = graph
                    .num_matches_between_images(image as u32, corr_image as u32)
                    .unwrap_or(0) as usize;
                let stat = self.image_pair_stats.entry(key).or_insert(ImagePairStat {
                    num_total_corrs: num_total,
                    num_tri_corrs: 0,
                });
                stat.num_tri_corrs += 1;
            }
        }
    }

    fn reset_tri_observations(
        &mut self,
        graph: &CorrespondenceGraph,
        frames: &[ImageFrame],
        reconstruction: &Reconstruction,
        image: usize,
        feature: usize,
        is_deleted_point3d: bool,
    ) {
        if reconstruction.poses.get(image).copied().flatten().is_none() {
            return;
        }
        let Some(point_id) = reconstruction.observations[image][feature] else {
            return;
        };

        for (corr_image, corr_feature) in correspondences_for(graph, image, feature) {
            decrement_correspondence_has_point3d(
                &mut self.image_stats,
                &mut self.point3d_correspondence_counts,
                frames,
                corr_image,
                corr_feature,
            );

            let corr_point = reconstruction.observations[corr_image][corr_feature];
            if corr_point == Some(point_id) && (!is_deleted_point3d || image < corr_image) {
                let key = ordered_pair(image, corr_image);
                if let Some(stat) = self.image_pair_stats.get_mut(&key) {
                    debug_assert!(stat.num_tri_corrs > 0);
                    stat.num_tri_corrs = stat.num_tri_corrs.saturating_sub(1);
                }
            }
        }
    }

    pub fn mark_point3d_modified(&mut self, point_id: usize) {
        self.modified_point3d_ids.insert(point_id);
    }

    pub fn modified_point3d_ids(&self) -> &HashSet<usize> {
        &self.modified_point3d_ids
    }

    pub fn clear_modified_point3d_ids(&mut self) {
        self.modified_point3d_ids.clear();
    }

    pub fn image_pairs(&self) -> &HashMap<(usize, usize), ImagePairStat> {
        &self.image_pair_stats
    }

    pub fn num_observations(&self, image: usize) -> usize {
        self.image_stats
            .get(image)
            .map(|stat| stat.num_observations)
            .unwrap_or(0)
    }

    pub fn num_correspondences(&self, image: usize) -> usize {
        self.image_stats
            .get(image)
            .map(|stat| stat.num_correspondences)
            .unwrap_or(0)
    }

    pub fn num_visible_correspondences(&self, image: usize) -> usize {
        self.image_stats
            .get(image)
            .map(|stat| stat.num_visible_correspondences)
            .unwrap_or(0)
    }

    pub fn num_visible_points3d(&self, image: usize) -> usize {
        self.image_stats
            .get(image)
            .map(|stat| stat.num_visible_points3d)
            .unwrap_or(0)
    }

    pub fn point3d_visibility_score(&self, image: usize) -> usize {
        self.image_stats
            .get(image)
            .map(ImageStat::point3d_visibility_score)
            .unwrap_or(0)
    }

    pub fn num_correspondences_have_point3d(&self, image: usize, feature: usize) -> usize {
        self.point3d_correspondence_counts
            .get(image)
            .and_then(|counts| counts.get(feature))
            .copied()
            .unwrap_or(0)
    }

    fn delete_point3d_internal(
        &mut self,
        reconstruction: &mut Reconstruction,
        point_id: usize,
    ) -> bool {
        if point_id >= reconstruction.points.len() {
            return false;
        }

        ensure_point_id_table(reconstruction);
        reconstruction.points.remove(point_id);
        if point_id < reconstruction.point_ids.len() {
            reconstruction.point_ids.remove(point_id);
        }
        for observations in &mut reconstruction.observations {
            for observation in observations {
                if let Some(id) = *observation {
                    if id == point_id {
                        *observation = None;
                    } else if id > point_id {
                        *observation = Some(id - 1);
                    }
                }
            }
        }
        self.shift_modified_point_ids_after_delete(point_id);
        true
    }

    fn shift_modified_point_ids_after_delete(&mut self, point_id: usize) {
        self.modified_point3d_ids = self
            .modified_point3d_ids
            .iter()
            .filter_map(|&id| {
                if id == point_id {
                    None
                } else if id > point_id {
                    Some(id - 1)
                } else {
                    Some(id)
                }
            })
            .collect();
    }
}

fn ordered_pair(left: usize, right: usize) -> (usize, usize) {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

fn mark_observation(
    image_stats: &mut [ImageStat],
    observed_features: &mut [Vec<bool>],
    image: usize,
    feature: usize,
) {
    if !observed_features[image][feature] {
        observed_features[image][feature] = true;
        image_stats[image].num_observations += 1;
    }
}

fn increment_correspondence_has_point3d(
    image_stats: &mut [ImageStat],
    point3d_correspondence_counts: &mut [Vec<usize>],
    frames: &[ImageFrame],
    image: usize,
    feature: usize,
) {
    point3d_correspondence_counts[image][feature] += 1;
    if point3d_correspondence_counts[image][feature] == 1 {
        image_stats[image].num_visible_points3d += 1;
        if let Some(keypoint) = frames
            .get(image)
            .and_then(|frame| frame.keypoints.get(feature))
        {
            image_stats[image]
                .visibility_pyramid
                .set_point(keypoint.x(), keypoint.y());
        }
    }
}

fn decrement_correspondence_has_point3d(
    image_stats: &mut [ImageStat],
    point3d_correspondence_counts: &mut [Vec<usize>],
    frames: &[ImageFrame],
    image: usize,
    feature: usize,
) {
    debug_assert!(point3d_correspondence_counts[image][feature] > 0);
    point3d_correspondence_counts[image][feature] -= 1;
    if point3d_correspondence_counts[image][feature] == 0 {
        image_stats[image].num_visible_points3d =
            image_stats[image].num_visible_points3d.saturating_sub(1);
        if let Some(keypoint) = frames
            .get(image)
            .and_then(|frame| frame.keypoints.get(feature))
        {
            image_stats[image]
                .visibility_pyramid
                .reset_point(keypoint.x(), keypoint.y());
        }
    }
}

fn correspondences_for(
    graph: &CorrespondenceGraph,
    image: usize,
    feature: usize,
) -> Vec<(usize, usize)> {
    graph
        .find_correspondences(image as u32, feature as u32)
        .map(|corrs| {
            corrs
                .iter()
                .map(|corr| (corr.image_id as usize, corr.point2d_idx as usize))
                .collect()
        })
        .unwrap_or_default()
}

fn track_observation_is_valid(
    reconstruction: &Reconstruction,
    observation: &TrackObservation,
) -> bool {
    reconstruction
        .observations
        .get(observation.image)
        .and_then(|observations| observations.get(observation.feature))
        .is_some()
}

fn track_observations_are_valid(
    reconstruction: &Reconstruction,
    track: &[TrackObservation],
) -> bool {
    let mut seen_observations = HashSet::new();
    track.iter().all(|obs| {
        track_observation_is_valid(reconstruction, obs)
            && seen_observations.insert((obs.image, obs.feature))
    })
}

fn ensure_point_id_table(reconstruction: &mut Reconstruction) {
    let mut used = reconstruction
        .point_ids
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    while reconstruction.point_ids.len() < reconstruction.points.len() {
        let mut point3d_id = reconstruction.point_ids.len() as u64 + 1;
        while used.contains(&point3d_id) {
            point3d_id += 1;
        }
        reconstruction.point_ids.push(point3d_id);
        used.insert(point3d_id);
    }
}

fn next_point3d_id(reconstruction: &Reconstruction) -> u64 {
    let mut point3d_id = reconstruction.point_ids.iter().copied().max().unwrap_or(0) + 1;
    let used = reconstruction
        .point_ids
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    while used.contains(&point3d_id) {
        point3d_id += 1;
    }
    point3d_id
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        CameraModel, DataId, Frame, ImageFrame, Point3D, Reconstruction, Rig, RigSensor, Rigid3,
        SensorId, SensorType, TrackObservation,
    };
    use rustslam::{Descriptors, Match, SE3};
    use std::path::PathBuf;

    #[test]
    fn counts_visible_points_for_unregistered_images() {
        let frames = vec![frame(0, 100, 100), frame(1, 100, 100), frame(2, 100, 100)];
        let pairs = vec![pair(0, 1, &[(0, 0), (1, 1)]), pair(1, 2, &[(0, 0), (1, 1)])];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::identity());
        reconstruction.observations[0][0] = Some(0);
        reconstruction.observations[1][0] = Some(0);
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
                    feature: 0,
                },
            ],
        });

        let manager = ObservationManager::new(&frames, &pairs, &reconstruction);

        assert_eq!(manager.num_observations(2), 2);
        assert_eq!(manager.num_visible_points3d(2), 1);
        assert!(manager.point3d_visibility_score(2) > 0);
    }

    #[test]
    fn counts_triangulated_correspondences_per_pair() {
        let frames = vec![frame(0, 100, 100), frame(1, 100, 100)];
        let pairs = vec![pair(0, 1, &[(0, 0), (1, 1)])];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::identity());
        reconstruction.observations[0][0] = Some(3);
        reconstruction.observations[1][0] = Some(3);

        let manager = ObservationManager::new(&frames, &pairs, &reconstruction);

        let stat = manager.image_pairs().get(&(0, 1)).unwrap();
        assert_eq!(stat.num_total_corrs, 2);
        assert_eq!(stat.num_tri_corrs, 1);
    }

    #[test]
    fn registered_images_propagate_visible_correspondences() {
        let frames = vec![frame(0, 100, 100), frame(1, 100, 100)];
        let pairs = vec![pair(0, 1, &[(0, 0), (1, 1)])];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());

        let manager = ObservationManager::new(&frames, &pairs, &reconstruction);

        assert_eq!(manager.num_visible_correspondences(1), 2);
        assert_eq!(manager.num_visible_correspondences(0), 0);
    }

    #[test]
    fn register_image_updates_visible_correspondence_stats() {
        let frames = vec![frame(0, 100, 100), frame(1, 100, 100)];
        let pairs = vec![pair(0, 1, &[(0, 0), (1, 1)])];
        let mut reconstruction = reconstruction(&frames);
        let mut manager = ObservationManager::new(&frames, &pairs, &reconstruction);

        assert!(manager.register_image(&frames, &pairs, &mut reconstruction, 0, SE3::identity()));

        assert!(reconstruction.poses[0].is_some());
        assert_eq!(manager.num_visible_correspondences(1), 2);
        assert_eq!(manager.num_visible_correspondences(0), 0);
    }

    #[test]
    fn register_image_does_not_double_count_existing_point3d_correspondences() {
        let frames = vec![frame(0, 100, 100), frame(1, 100, 100), frame(2, 100, 100)];
        let pairs = vec![pair(0, 2, &[(0, 0)]), pair(1, 2, &[(0, 0)])];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::identity());
        reconstruction.observations[0][0] = Some(0);
        reconstruction.observations[1][0] = Some(0);
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
                    feature: 0,
                },
            ],
        });
        let mut manager = ObservationManager::new(&frames, &pairs, &reconstruction);
        assert_eq!(manager.num_correspondences_have_point3d(2, 0), 2);

        assert!(manager.register_image(&frames, &pairs, &mut reconstruction, 2, SE3::identity()));

        let fresh_manager = ObservationManager::new(&frames, &pairs, &reconstruction);
        assert_eq!(manager.num_correspondences_have_point3d(2, 0), 2);
        assert_eq!(
            manager.num_correspondences_have_point3d(2, 0),
            fresh_manager.num_correspondences_have_point3d(2, 0)
        );
    }

    #[test]
    fn register_image_registers_whole_rig_frame_with_sensor_poses() {
        let frames = vec![frame(0, 100, 100), frame(1, 100, 100), frame(2, 100, 100)];
        let pairs = vec![pair(1, 2, &[(0, 0), (1, 1)])];
        let mut reconstruction = reconstruction(&frames);
        add_two_camera_rig_frame(&mut reconstruction, 0, 1);
        let mut manager = ObservationManager::new(&frames, &pairs, &reconstruction);
        let selected_pose =
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(10.0, 0.0, 0.0));

        assert!(manager.register_image(&frames, &pairs, &mut reconstruction, 1, selected_pose));

        assert_eq!(
            reconstruction.poses[0].unwrap().translation(),
            [9.0, 0.0, 0.0]
        );
        assert_eq!(
            reconstruction.poses[1].unwrap().translation(),
            [10.0, 0.0, 0.0]
        );
        assert!(reconstruction.poses[2].is_none());
        assert_eq!(
            reconstruction.frames[0].rig_from_world.tvec,
            [9.0, 0.0, 0.0]
        );
        assert_eq!(manager.num_visible_correspondences(2), 2);
    }

    #[test]
    fn deregister_image_deletes_its_observations_and_refreshes_stats() {
        let frames = vec![frame(0, 100, 100), frame(1, 100, 100), frame(2, 100, 100)];
        let pairs = vec![pair(0, 1, &[(0, 0)]), pair(1, 2, &[(0, 0)])];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::identity());
        reconstruction.poses[2] = Some(SE3::identity());
        let mut manager = ObservationManager::new(&frames, &pairs, &reconstruction);
        manager
            .add_point3d(
                &frames,
                &pairs,
                &mut reconstruction,
                Point3D {
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
                            feature: 0,
                        },
                        TrackObservation {
                            image: 2,
                            feature: 0,
                        },
                    ],
                },
            )
            .unwrap();

        assert!(manager.deregister_image(&frames, &pairs, &mut reconstruction, 0));

        assert!(reconstruction.poses[0].is_none());
        assert_eq!(reconstruction.points.len(), 1);
        assert_eq!(reconstruction.points[0].track.len(), 2);
        assert_eq!(reconstruction.observations[0][0], None);
        assert_eq!(reconstruction.observations[1][0], Some(0));
        assert_eq!(reconstruction.observations[2][0], Some(0));
        assert_eq!(manager.num_visible_correspondences(1), 1);
        assert_eq!(manager.num_visible_correspondences(0), 1);
    }

    #[test]
    fn deregister_image_deregisters_whole_rig_frame() {
        let frames = vec![frame(0, 100, 100), frame(1, 100, 100), frame(2, 100, 100)];
        let pairs = vec![pair(0, 2, &[(0, 0)]), pair(1, 2, &[(0, 0)])];
        let mut reconstruction = reconstruction(&frames);
        add_two_camera_rig_frame(&mut reconstruction, 0, 1);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::identity());
        reconstruction.poses[2] = Some(SE3::identity());
        let mut manager = ObservationManager::new(&frames, &pairs, &reconstruction);
        manager
            .add_point3d(
                &frames,
                &pairs,
                &mut reconstruction,
                Point3D {
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
                            feature: 0,
                        },
                        TrackObservation {
                            image: 2,
                            feature: 0,
                        },
                    ],
                },
            )
            .unwrap();

        assert!(manager.deregister_image(&frames, &pairs, &mut reconstruction, 1));

        assert!(reconstruction.poses[0].is_none());
        assert!(reconstruction.poses[1].is_none());
        assert!(reconstruction.poses[2].is_some());
        assert_eq!(reconstruction.observations[0][0], None);
        assert_eq!(reconstruction.observations[1][0], None);
        assert_eq!(reconstruction.observations[2][0], None);
        assert!(reconstruction.points.is_empty());
        assert_eq!(reconstruction.frames[0].rig_from_world, Rigid3::identity());
    }

    #[test]
    fn add_point3d_updates_pair_visibility_and_modified_state() {
        let frames = vec![frame(0, 100, 100), frame(1, 100, 100), frame(2, 100, 100)];
        let pairs = vec![pair(0, 1, &[(0, 0)]), pair(1, 2, &[(0, 0)])];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::identity());
        let mut manager = ObservationManager::new(&frames, &pairs, &reconstruction);

        let point_id = manager
            .add_point3d(
                &frames,
                &pairs,
                &mut reconstruction,
                Point3D {
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
                            feature: 0,
                        },
                    ],
                },
            )
            .unwrap();

        assert_eq!(point_id, 0);
        assert_eq!(reconstruction.observations[0][0], Some(0));
        assert_eq!(reconstruction.observations[1][0], Some(0));
        assert_eq!(manager.image_pairs().get(&(0, 1)).unwrap().num_tri_corrs, 1);
        assert_eq!(manager.num_visible_points3d(2), 1);
        assert_eq!(manager.num_correspondences_have_point3d(2, 0), 1);
        assert!(manager.point3d_visibility_score(2) > 0);
        assert!(manager.modified_point3d_ids().contains(&0));
    }

    #[test]
    fn delete_observation_on_two_view_track_deletes_point3d() {
        let frames = vec![frame(0, 100, 100), frame(1, 100, 100)];
        let pairs = vec![pair(0, 1, &[(0, 0)])];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::identity());
        let mut manager = ObservationManager::new(&frames, &pairs, &reconstruction);
        manager
            .add_point3d(
                &frames,
                &pairs,
                &mut reconstruction,
                Point3D {
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
                            feature: 0,
                        },
                    ],
                },
            )
            .unwrap();

        assert!(manager.delete_observation(&frames, &pairs, &mut reconstruction, 0, 0));

        assert!(reconstruction.points.is_empty());
        assert!(reconstruction.point_ids.is_empty());
        assert_eq!(reconstruction.observations[0][0], None);
        assert_eq!(reconstruction.observations[1][0], None);
        assert_eq!(manager.image_pairs().get(&(0, 1)).unwrap().num_tri_corrs, 0);
    }

    #[test]
    fn delete_observation_on_three_view_track_keeps_two_view_point() {
        let frames = vec![frame(0, 100, 100), frame(1, 100, 100), frame(2, 100, 100)];
        let pairs = vec![
            pair(0, 1, &[(0, 0)]),
            pair(1, 2, &[(0, 0)]),
            pair(0, 2, &[(0, 0)]),
        ];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::identity());
        reconstruction.poses[2] = Some(SE3::identity());
        let mut manager = ObservationManager::new(&frames, &pairs, &reconstruction);
        manager
            .add_point3d(
                &frames,
                &pairs,
                &mut reconstruction,
                Point3D {
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
                            feature: 0,
                        },
                        TrackObservation {
                            image: 2,
                            feature: 0,
                        },
                    ],
                },
            )
            .unwrap();

        assert!(manager.delete_observation(&frames, &pairs, &mut reconstruction, 2, 0));

        assert_eq!(reconstruction.points.len(), 1);
        assert_eq!(reconstruction.points[0].track.len(), 2);
        assert_eq!(reconstruction.observations[2][0], None);
        assert_eq!(manager.image_pairs().get(&(0, 1)).unwrap().num_tri_corrs, 1);
        assert_eq!(manager.image_pairs().get(&(0, 2)).unwrap().num_tri_corrs, 0);
    }

    #[test]
    fn merge_points3d_reassigns_tracks_and_pair_stats() {
        let frames = vec![
            frame(0, 100, 100),
            frame(1, 100, 100),
            frame(2, 100, 100),
            frame(3, 100, 100),
        ];
        let pairs = vec![
            pair(0, 1, &[(0, 0)]),
            pair(1, 2, &[(0, 0)]),
            pair(2, 3, &[(0, 0)]),
        ];
        let mut reconstruction = reconstruction(&frames);
        for pose in &mut reconstruction.poses {
            *pose = Some(SE3::identity());
        }
        let mut manager = ObservationManager::new(&frames, &pairs, &reconstruction);
        let first = manager
            .add_point3d(
                &frames,
                &pairs,
                &mut reconstruction,
                Point3D {
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
                            feature: 0,
                        },
                    ],
                },
            )
            .unwrap();
        let second = manager
            .add_point3d(
                &frames,
                &pairs,
                &mut reconstruction,
                Point3D {
                    xyz: [0.1, 0.0, 1.0],
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
                },
            )
            .unwrap();

        let merged = manager
            .merge_points3d(
                &frames,
                &pairs,
                &mut reconstruction,
                first,
                second,
                Point3D {
                    xyz: [0.05, 0.0, 1.0],
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
                        TrackObservation {
                            image: 2,
                            feature: 0,
                        },
                        TrackObservation {
                            image: 3,
                            feature: 0,
                        },
                    ],
                },
            )
            .unwrap();

        assert_eq!(merged, 0);
        assert_eq!(reconstruction.points.len(), 1);
        assert_eq!(reconstruction.points[0].track.len(), 4);
        assert_eq!(reconstruction.observations[2][0], Some(0));
        assert_eq!(reconstruction.observations[3][0], Some(0));
        assert_eq!(manager.image_pairs().get(&(1, 2)).unwrap().num_tri_corrs, 1);
        assert_eq!(manager.image_pairs().get(&(2, 3)).unwrap().num_tri_corrs, 1);
        assert!(manager.modified_point3d_ids().contains(&0));
    }

    #[test]
    fn incremental_event_paths_match_rebuild_stats() {
        let frames = vec![frame(0, 100, 100), frame(1, 100, 100), frame(2, 100, 100)];
        let pairs = vec![pair(0, 1, &[(0, 0)]), pair(1, 2, &[(0, 0)])];
        let mut incremental_recon = reconstruction(&frames);
        incremental_recon.poses[0] = Some(SE3::identity());
        incremental_recon.poses[1] = Some(SE3::identity());
        let mut incremental = ObservationManager::new(&frames, &pairs, &incremental_recon);
        incremental
            .add_point3d(
                &frames,
                &pairs,
                &mut incremental_recon,
                Point3D {
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
                            feature: 0,
                        },
                    ],
                },
            )
            .unwrap();
        assert!(incremental.register_image(
            &frames,
            &pairs,
            &mut incremental_recon,
            2,
            SE3::identity()
        ));

        let mut rebuild_recon = reconstruction(&frames);
        rebuild_recon.poses[0] = Some(SE3::identity());
        rebuild_recon.poses[1] = Some(SE3::identity());
        rebuild_recon.poses[2] = Some(SE3::identity());
        rebuild_recon.observations[0][0] = Some(0);
        rebuild_recon.observations[1][0] = Some(0);
        rebuild_recon.points.push(Point3D {
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
                    feature: 0,
                },
            ],
        });
        let rebuild = ObservationManager::new(&frames, &pairs, &rebuild_recon);

        for image in 0..frames.len() {
            assert_eq!(
                incremental.num_visible_correspondences(image),
                rebuild.num_visible_correspondences(image),
                "visible corrs image {image}"
            );
            assert_eq!(
                incremental.num_visible_points3d(image),
                rebuild.num_visible_points3d(image),
                "visible points3d image {image}"
            );
            assert_eq!(
                incremental.point3d_visibility_score(image),
                rebuild.point3d_visibility_score(image),
                "visibility score image {image}"
            );
        }
        assert_eq!(
            incremental
                .image_pairs()
                .get(&(0, 1))
                .unwrap()
                .num_tri_corrs,
            rebuild.image_pairs().get(&(0, 1)).unwrap().num_tri_corrs
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

    fn add_two_camera_rig_frame(reconstruction: &mut Reconstruction, left: usize, right: usize) {
        let ref_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 11,
        };
        let right_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 12,
        };
        reconstruction.rigs = vec![Rig {
            rig_id: 3,
            ref_sensor_id: Some(ref_sensor.clone()),
            sensors: vec![
                RigSensor {
                    sensor_id: ref_sensor.clone(),
                    sensor_from_rig: None,
                },
                RigSensor {
                    sensor_id: right_sensor.clone(),
                    sensor_from_rig: Some(Rigid3 {
                        qvec: [1.0, 0.0, 0.0, 0.0],
                        tvec: [1.0, 0.0, 0.0],
                    }),
                },
            ],
        }];
        reconstruction.frames = vec![Frame {
            frame_id: 9,
            rig_id: 3,
            rig_from_world: Rigid3::identity(),
            data_ids: vec![
                DataId {
                    sensor_id: ref_sensor,
                    data_id: reconstruction.image_id(left) as u64,
                },
                DataId {
                    sensor_id: right_sensor,
                    data_id: reconstruction.image_id(right) as u64,
                },
            ],
        }];
        reconstruction.image_frame_indices[left] = Some(0);
        reconstruction.image_frame_indices[right] = Some(0);
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

    fn frame(id: usize, width: u32, height: u32) -> ImageFrame {
        ImageFrame {
            id,
            name: format!("{id}.jpg"),
            path: PathBuf::from(format!("{id}.jpg")),
            width,
            height,
            keypoints: vec![
                rustslam::KeyPoint::new(10.0, 10.0),
                rustslam::KeyPoint::new(80.0, 80.0),
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
