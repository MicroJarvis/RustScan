use crate::types::{ImageFrame, PairGeometry, Point3D, Reconstruction, TrackObservation};
use rustslam::SE3;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ImagePairStat {
    pub num_tri_corrs: usize,
    pub num_total_corrs: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImageStat {
    pub num_observations: usize,
    pub num_correspondences: usize,
    pub num_visible_correspondences: usize,
    pub num_visible_points3d: usize,
    pub point3d_visibility_score: usize,
}

#[derive(Debug, Clone, Default)]
pub struct ObservationManager {
    image_stats: Vec<ImageStat>,
    image_pair_stats: HashMap<(usize, usize), ImagePairStat>,
    point3d_correspondence_counts: Vec<Vec<usize>>,
    modified_point3d_ids: HashSet<usize>,
}

impl ObservationManager {
    pub fn new(
        frames: &[ImageFrame],
        pairs: &[PairGeometry],
        reconstruction: &Reconstruction,
    ) -> Self {
        let mut manager = Self::default();
        manager.rebuild(frames, pairs, reconstruction);
        manager
    }

    pub fn rebuild(
        &mut self,
        frames: &[ImageFrame],
        pairs: &[PairGeometry],
        reconstruction: &Reconstruction,
    ) {
        let modified_point3d_ids = std::mem::take(&mut self.modified_point3d_ids);
        let mut image_stats = vec![ImageStat::default(); frames.len()];
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
                        pair.right,
                        right_feature,
                    );
                }
                if right_point.is_some() && right_registered {
                    increment_correspondence_has_point3d(
                        &mut image_stats,
                        &mut point3d_correspondence_counts,
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

        for image in 0..frames.len() {
            let positions = point3d_correspondence_counts[image]
                .iter()
                .enumerate()
                .filter_map(|(feature, &count)| {
                    (count > 0).then(|| {
                        let keypoint = &frames[image].keypoints[feature];
                        (keypoint.x(), keypoint.y())
                    })
                })
                .collect::<Vec<_>>();
            image_stats[image].point3d_visibility_score =
                visibility_pyramid_score(frames.get(image), &positions);
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
        let Some(slot) = reconstruction.poses.get_mut(image) else {
            return false;
        };
        *slot = Some(pose);
        self.rebuild(frames, pairs, reconstruction);
        true
    }

    pub fn deregister_image(
        &mut self,
        frames: &[ImageFrame],
        pairs: &[PairGeometry],
        reconstruction: &mut Reconstruction,
        image: usize,
    ) -> bool {
        if image >= reconstruction.poses.len() {
            return false;
        }
        let features = reconstruction
            .observations
            .get(image)
            .map(|observations| {
                observations
                    .iter()
                    .enumerate()
                    .filter_map(|(feature, point_id)| point_id.is_some().then_some(feature))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        for feature in features {
            self.delete_observation(frames, pairs, reconstruction, image, feature);
        }
        reconstruction.poses[image] = None;
        self.rebuild(frames, pairs, reconstruction);
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
        self.rebuild(frames, pairs, reconstruction);
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
            || reconstruction.points[point_id]
                .track
                .iter()
                .any(|obs| obs.image == observation.image)
        {
            return false;
        }

        reconstruction.observations[observation.image][observation.feature] = Some(point_id);
        reconstruction.points[point_id].track.push(observation);
        self.mark_point3d_modified(point_id);
        self.rebuild(frames, pairs, reconstruction);
        true
    }

    pub fn delete_point3d(
        &mut self,
        frames: &[ImageFrame],
        pairs: &[PairGeometry],
        reconstruction: &mut Reconstruction,
        point_id: usize,
    ) -> bool {
        if !self.delete_point3d_internal(reconstruction, point_id) {
            return false;
        }
        self.rebuild(frames, pairs, reconstruction);
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
            self.rebuild(frames, pairs, reconstruction);
            return true;
        }

        if reconstruction.points[point_id].track.len() <= 2 {
            return self.delete_point3d(frames, pairs, reconstruction, point_id);
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
        self.rebuild(frames, pairs, reconstruction);
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

        let (keep_id, remove_id) = ordered_pair(point_id1, point_id2);
        for obs in &merged_point.track {
            let assigned = reconstruction.observations[obs.image][obs.feature];
            if assigned.is_some_and(|point_id| point_id != point_id1 && point_id != point_id2) {
                return None;
            }
        }

        ensure_point_id_table(reconstruction);
        reconstruction.points[keep_id] = merged_point;
        for obs in &reconstruction.points[keep_id].track {
            reconstruction.observations[obs.image][obs.feature] = Some(keep_id);
        }
        if !self.delete_point3d_internal(reconstruction, remove_id) {
            return None;
        }
        self.mark_point3d_modified(keep_id);
        self.rebuild(frames, pairs, reconstruction);
        Some(keep_id)
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
            .map(|stat| stat.point3d_visibility_score)
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
    image: usize,
    feature: usize,
) {
    point3d_correspondence_counts[image][feature] += 1;
    if point3d_correspondence_counts[image][feature] == 1 {
        image_stats[image].num_visible_points3d += 1;
    }
}

fn visibility_pyramid_score(frame: Option<&ImageFrame>, positions: &[(f32, f32)]) -> usize {
    let Some(frame) = frame else {
        return 0;
    };
    if frame.width == 0 || frame.height == 0 || positions.is_empty() {
        return 0;
    }
    let mut score = 0usize;
    for level in 0..6 {
        let bins = 1usize << level;
        let mut occupied = HashSet::new();
        for &(x, y) in positions {
            let bx = ((x.max(0.0) / frame.width as f32) * bins as f32).floor() as usize;
            let by = ((y.max(0.0) / frame.height as f32) * bins as f32).floor() as usize;
            occupied.insert((bx.min(bins - 1), by.min(bins - 1)));
        }
        score += occupied.len();
    }
    score
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
    let mut seen_images = HashSet::new();
    track.iter().all(|obs| {
        track_observation_is_valid(reconstruction, obs)
            && seen_observations.insert((obs.image, obs.feature))
            && seen_images.insert(obs.image)
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
    use crate::types::{CameraModel, ImageFrame, Point3D, Reconstruction, TrackObservation};
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
