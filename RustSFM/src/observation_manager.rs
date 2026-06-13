use crate::types::{ImageFrame, PairGeometry, Reconstruction};
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
}

impl ObservationManager {
    pub fn new(
        frames: &[ImageFrame],
        pairs: &[PairGeometry],
        reconstruction: &Reconstruction,
    ) -> Self {
        let mut image_stats = vec![ImageStat::default(); frames.len()];
        let mut observed_features = frames
            .iter()
            .map(|frame| vec![false; frame.keypoints.len()])
            .collect::<Vec<_>>();
        let mut visible_point_features = frames
            .iter()
            .map(|frame| vec![false; frame.keypoints.len()])
            .collect::<Vec<_>>();
        let mut visible_feature_positions = vec![Vec::<(f32, f32)>::new(); frames.len()];
        let mut image_pair_stats = HashMap::new();

        for pair in pairs {
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
                if left_point.is_some() && reconstruction.poses[pair.right].is_some() {
                    image_stats[pair.left].num_visible_correspondences += 1;
                }
                if right_point.is_some() && reconstruction.poses[pair.left].is_some() {
                    image_stats[pair.right].num_visible_correspondences += 1;
                }
                if right_point.is_some() {
                    mark_visible_point(
                        frames,
                        &mut image_stats,
                        &mut visible_point_features,
                        &mut visible_feature_positions,
                        pair.left,
                        left_feature,
                    );
                }
                if left_point.is_some() {
                    mark_visible_point(
                        frames,
                        &mut image_stats,
                        &mut visible_point_features,
                        &mut visible_feature_positions,
                        pair.right,
                        right_feature,
                    );
                }
                if left_point.is_some() && left_point == right_point {
                    stat.num_tri_corrs += 1;
                }
            }
            image_pair_stats.insert(key, stat);
        }

        for (image, positions) in visible_feature_positions.iter().enumerate() {
            image_stats[image].point3d_visibility_score =
                visibility_pyramid_score(frames.get(image), positions);
        }

        Self {
            image_stats,
            image_pair_stats,
        }
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

fn mark_visible_point(
    frames: &[ImageFrame],
    image_stats: &mut [ImageStat],
    visible_point_features: &mut [Vec<bool>],
    visible_feature_positions: &mut [Vec<(f32, f32)>],
    image: usize,
    feature: usize,
) {
    if visible_point_features[image][feature] {
        return;
    }
    visible_point_features[image][feature] = true;
    image_stats[image].num_visible_points3d += 1;
    let keypoint = &frames[image].keypoints[feature];
    visible_feature_positions[image].push((keypoint.x(), keypoint.y()));
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

    fn reconstruction(frames: &[ImageFrame]) -> Reconstruction {
        Reconstruction {
            camera: CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0),
            cameras: vec![CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0)],
            camera_ids: vec![1],
            image_names: frames.iter().map(|frame| frame.name.clone()).collect(),
            image_paths: frames.iter().map(|frame| frame.path.clone()).collect(),
            image_ids: (0..frames.len()).map(|idx| idx as u32 + 1).collect(),
            image_camera_indices: vec![0; frames.len()],
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
