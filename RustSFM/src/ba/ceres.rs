//! Ceres-backed bundle adjustment (`feature = "ceres-ba"`, enabled by default).
//!
//! Delegates to [`ceres_problem`] for full pose-block, rig/sensor extrinsics,
//! intrinsics refinement, and gauge support.

use super::ceres_problem;
use super::{BundleAdjustmentOptions, BundleAdjustmentReport};
use crate::types::{ImageFrame, Reconstruction};

/// Returns true when the Ceres backend can handle this problem configuration.
pub fn supports_ceres_ba(_reconstruction: &Reconstruction, _options: &BundleAdjustmentOptions) -> bool {
    true
}

pub fn refine_bundle_adjustment_ceres(
    frames: &[ImageFrame],
    reconstruction: &mut Reconstruction,
    options: BundleAdjustmentOptions,
) -> Option<BundleAdjustmentReport> {
    ceres_problem::solve_bundle_adjustment_ceres(frames, reconstruction, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{refine_bundle_adjustment, BundleAdjustmentLoss};
    use crate::sift::SiftFeatures;
    use crate::types::{CameraModel, Point3D, TrackObservation};
    use rustslam::Descriptors;
    use crate::wide::WideDescriptors;
    use rustslam::KeyPoint;
    use glam::{Quat, Vec3};
    use rustslam::SE3;
    use std::path::PathBuf;

    fn frame(id: usize) -> ImageFrame {
        ImageFrame {
            id,
            name: format!("{id}.jpg"),
            path: PathBuf::from(format!("{id}.jpg")),
            width: 100,
            height: 100,
            keypoints: Vec::new(),
            descriptors: Descriptors::new(),
            sift: SiftFeatures::default(),
            wide_descriptors: WideDescriptors {
                data: Vec::new(),
                dim: 0,
                count: 0,
            },
            strong_feature_indices: Vec::new(),
            colors: Vec::new(),
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

    #[test]
    fn ceres_local_ba_keeps_constant_image_fixed() {
        let mut frames = vec![frame(0), frame(1)];
        frames[0].keypoints = vec![
            KeyPoint::new(45.0, 45.0),
            KeyPoint::new(55.0, 45.0),
            KeyPoint::new(45.0, 55.0),
            KeyPoint::new(55.0, 55.0),
        ];
        frames[1].keypoints = vec![
            KeyPoint::new(70.0, 45.0),
            KeyPoint::new(80.0, 45.0),
            KeyPoint::new(70.0, 55.0),
            KeyPoint::new(80.0, 55.0),
        ];
        let mut reconstruction = reconstruction(&frames);
        let fixed_pose = SE3::identity();
        reconstruction.poses[0] = Some(fixed_pose);
        reconstruction.poses[1] = Some(SE3::from_quat_translation(
            Quat::IDENTITY,
            Vec3::new(0.8, 0.05, 0.0),
        ));
        for (idx, xyz) in [
            [-0.2, -0.2, 2.0],
            [0.2, -0.2, 2.0],
            [-0.2, 0.2, 2.0],
            [0.2, 0.2, 2.0],
        ]
        .into_iter()
        .enumerate()
        {
            reconstruction.observations[0][idx] = Some(idx);
            reconstruction.observations[1][idx] = Some(idx);
            reconstruction.points.push(Point3D {
                xyz,
                color: [0, 0, 0],
                error: 0.0,
                track: vec![
                    TrackObservation {
                        image: 0,
                        feature: idx,
                    },
                    TrackObservation {
                        image: 1,
                        feature: idx,
                    },
                ],
            });
            reconstruction.point_ids.push(idx as u64 + 1);
        }

        assert!(supports_ceres_ba(
            &reconstruction,
            &BundleAdjustmentOptions {
                variable_images: Some(vec![1]),
                constant_images: vec![0],
                ..BundleAdjustmentOptions::default()
            }
        ));

        let report = refine_bundle_adjustment(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 20,
                loss_function: BundleAdjustmentLoss::Huber { scale: 4.0 },
                max_observation_error_px: 50.0,
                variable_images: Some(vec![1]),
                constant_images: vec![0],
                point_ids: Some((0..4).collect()),
                ..BundleAdjustmentOptions::default()
            },
        )
        .expect("ceres ba should succeed");
        assert_eq!(
            reconstruction.poses[0].unwrap().translation(),
            fixed_pose.translation()
        );
        assert!(report.final_cost <= report.initial_cost);
    }
}
