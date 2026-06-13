use crate::geometry::{camera_center, mean_pair_reprojection_error_with_cameras};
use crate::types::{ImageFrame, PairGeometry, Point3D, Reconstruction, TrackObservation};
use rustslam::SE3;
use std::collections::HashSet;

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
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TriangulationReport {
    pub created_points: usize,
    pub continued_observations: usize,
}

impl TriangulationReport {
    pub fn total_observations(self) -> usize {
        self.created_points * 2 + self.continued_observations
    }
}

pub struct IncrementalTriangulator<'a> {
    frames: &'a [ImageFrame],
    pairs: &'a [PairGeometry],
    reconstruction: &'a mut Reconstruction,
    modified_point3d_ids: HashSet<usize>,
}

impl<'a> IncrementalTriangulator<'a> {
    pub fn new(
        frames: &'a [ImageFrame],
        pairs: &'a [PairGeometry],
        reconstruction: &'a mut Reconstruction,
    ) -> Self {
        Self {
            frames,
            pairs,
            reconstruction,
            modified_point3d_ids: HashSet::new(),
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
        &self.modified_point3d_ids
    }

    pub fn clear_modified_points3d(&mut self) {
        self.modified_point3d_ids.clear();
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
        _options: &IncrementalTriangulatorOptions,
        point_id: usize,
        image: usize,
        feature: usize,
        max_reproj_error_px: f32,
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
        self.reconstruction.observations[image][feature] = Some(point_id);
        self.reconstruction.points[point_id]
            .track
            .push(TrackObservation { image, feature });
        self.modified_point3d_ids.insert(point_id);
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

        let left_kp = &self.frames[pair.left].keypoints[left_feature];
        let right_kp = &self.frames[pair.right].keypoints[right_feature];
        let left_xy = self
            .reconstruction
            .camera_for_image(pair.left)
            .normalize(left_kp.x(), left_kp.y());
        let right_xy = self
            .reconstruction
            .camera_for_image(pair.right)
            .normalize(right_kp.x(), right_kp.y());
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

        let point_id = self.reconstruction.points.len();
        self.reconstruction.observations[pair.left][left_feature] = Some(point_id);
        self.reconstruction.observations[pair.right][right_feature] = Some(point_id);
        self.reconstruction.point_ids.push(point_id as u64 + 1);
        self.reconstruction.points.push(Point3D {
            xyz,
            color: average_color(
                &[
                    TrackObservation {
                        image: pair.left,
                        feature: left_feature,
                    },
                    TrackObservation {
                        image: pair.right,
                        feature: right_feature,
                    },
                ],
                self.frames,
            ),
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
        });
        self.modified_point3d_ids.insert(point_id);
        true
    }
}

fn triangulation_angle_deg(left_pose: SE3, right_pose: SE3, point: [f32; 3]) -> Option<f32> {
    let c1 = camera_center(left_pose);
    let c2 = camera_center(right_pose);
    let p = glam::Vec3::from_array(point);
    let v1 = (p - c1).try_normalize()?;
    let v2 = (p - c2).try_normalize()?;
    Some(v1.dot(v2).abs().clamp(-1.0, 1.0).acos().to_degrees())
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
                ..IncrementalTriangulatorOptions::default()
            },
            0,
        );

        assert_eq!(report.created_points, 2);
        assert_eq!(triangulator.reconstruction.points.len(), 2);
        assert_eq!(triangulator.get_modified_points3d().len(), 2);
    }

    #[test]
    fn triangulate_image_continues_existing_track() {
        let frames = vec![frame(0), frame(1)];
        let pairs = vec![pair(0, 1, &[(0, 0)])];
        let mut reconstruction = reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::identity());
        reconstruction.observations[0][0] = Some(0);
        reconstruction.point_ids.push(1);
        reconstruction.points.push(Point3D {
            xyz: [0.0, 0.0, 1.0],
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
                ..IncrementalTriangulatorOptions::default()
            },
            1,
        );

        assert_eq!(report.continued_observations, 1);
        assert_eq!(triangulator.reconstruction.points[0].track.len(), 2);
        assert_eq!(triangulator.get_modified_points3d(), &HashSet::from([0]));
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
