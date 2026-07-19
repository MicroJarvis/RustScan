//! Ceres-backed bundle adjustment (`feature = "ceres-ba"`, enabled by default).
//!
//! Delegates to [`ceres_problem`] for full pose-block, rig/sensor extrinsics,
//! intrinsics refinement, and gauge support.

use super::ceres_problem;
use super::{BundleAdjustmentOptions, BundleAdjustmentReport};
use crate::types::{ImageFrame, Reconstruction};

/// Returns true when the Ceres backend can handle this problem configuration.
pub fn supports_ceres_ba(
    _reconstruction: &Reconstruction,
    _options: &BundleAdjustmentOptions,
) -> bool {
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
    use super::super::shared::project_point;
    use super::super::{
        camera_center_world, refine_bundle_adjustment, BundleAdjustmentLoss,
        BundleAdjustmentPosePrior,
    };
    use super::*;
    use crate::sift::SiftFeatures;
    use crate::types::{
        CameraModel, DataId, Frame, Point3D, Rig, RigSensor, Rigid3, SensorId, SensorType,
        TrackObservation,
    };
    use crate::wide::WideDescriptors;
    use glam::{Quat, Vec3};
    use rustslam::Descriptors;
    use rustslam::KeyPoint;
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
        assert_eq!(
            reconstruction.poses[0].unwrap().quaternion(),
            fixed_pose.quaternion()
        );
        assert!((quaternion_norm(reconstruction.poses[1].unwrap()) - 1.0).abs() < 1.0e-6);
        assert!(report.final_cost <= report.initial_cost);
        assert!(report.gradient_max_norm.is_finite());
        assert!(report.setup_ms.is_finite() && report.setup_ms >= 0.0);
        assert!(report.solve_ms.is_finite() && report.solve_ms >= 0.0);
        assert!(report.postprocess_ms.is_finite() && report.postprocess_ms >= 0.0);
        assert!(report.elapsed_ms >= report.solve_ms);
    }

    fn translation_distance(left: SE3, right: SE3) -> f32 {
        let left = left.translation();
        let right = right.translation();
        ((left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2) + (left[2] - right[2]).powi(2))
            .sqrt()
    }

    fn quaternion_norm(pose: SE3) -> f32 {
        let q = pose.quaternion();
        (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt()
    }

    fn sensor_from_rig_pose(
        reconstruction: &Reconstruction,
        rig_id: u32,
        sensor_id: &SensorId,
    ) -> SE3 {
        reconstruction
            .rigs
            .iter()
            .find(|rig| rig.rig_id == rig_id)
            .and_then(|rig| {
                rig.sensors
                    .iter()
                    .find(|sensor| &sensor.sensor_id == sensor_id)
            })
            .and_then(|sensor| sensor.sensor_from_rig.as_ref())
            .map(Rigid3::to_se3)
            .unwrap()
    }

    fn rig_sensor_ba_fixture() -> (Vec<ImageFrame>, Reconstruction, SensorId, SE3, SE3) {
        let mut frames = vec![frame(0), frame(1), frame(2), frame(3)];
        let camera = CameraModel::new_pinhole(160, 120, 90.0, 90.0, 80.0, 60.0);
        let ref_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 11,
        };
        let aux_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 12,
        };
        let true_sensor_from_rig =
            SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(0.35, 0.02, 0.0));
        let initial_sensor_from_rig =
            SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(0.1, -0.04, 0.0));
        let rig_poses = [
            SE3::from_quat_translation(Quat::IDENTITY, Vec3::ZERO),
            SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(0.18, 0.03, 0.0)),
        ];
        let outside_poses = [
            SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(0.75, 0.02, 0.0)),
            SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(-0.65, 0.01, 0.0)),
        ];
        let poses = [
            rig_poses[0],
            true_sensor_from_rig.compose(&rig_poses[0]),
            outside_poses[0],
            outside_poses[1],
        ];
        let scene_points = [
            [-0.35, -0.25, 2.5],
            [-0.05, -0.2, 2.3],
            [0.25, -0.15, 2.7],
            [0.45, 0.05, 2.9],
            [-0.25, 0.2, 2.4],
            [0.05, 0.25, 2.6],
            [0.35, 0.3, 3.0],
            [0.0, 0.0, 3.2],
        ];
        for image in 0..frames.len() {
            frames[image].keypoints = scene_points
                .iter()
                .map(|&point| {
                    let xy = project_point(camera, poses[image], point).unwrap();
                    KeyPoint::new(xy[0] as f32, xy[1] as f32)
                })
                .collect();
        }

        let mut reconstruction = reconstruction(&frames);
        reconstruction.camera = camera;
        reconstruction.cameras = vec![camera];
        reconstruction.rigs = vec![Rig {
            rig_id: 3,
            ref_sensor_id: Some(ref_sensor.clone()),
            sensors: vec![
                RigSensor {
                    sensor_id: ref_sensor,
                    sensor_from_rig: None,
                },
                RigSensor {
                    sensor_id: aux_sensor.clone(),
                    sensor_from_rig: Some(Rigid3::from_se3(initial_sensor_from_rig)),
                },
            ],
        }];
        reconstruction.frames = vec![
            Frame {
                frame_id: 9,
                rig_id: 3,
                rig_from_world: Rigid3::from_se3(rig_poses[0]),
                data_ids: vec![
                    DataId {
                        sensor_id: SensorId {
                            sensor_type: SensorType::Camera,
                            sensor_id: 11,
                        },
                        data_id: reconstruction.image_id(0) as u64,
                    },
                    DataId {
                        sensor_id: aux_sensor.clone(),
                        data_id: reconstruction.image_id(1) as u64,
                    },
                ],
            },
            Frame {
                frame_id: 10,
                rig_id: 3,
                rig_from_world: Rigid3::from_se3(rig_poses[1]),
                data_ids: Vec::new(),
            },
        ];
        reconstruction.image_frame_indices = vec![Some(0), Some(0), None, None];
        reconstruction.poses[0] = Some(rig_poses[0]);
        reconstruction.poses[1] = Some(initial_sensor_from_rig.compose(&rig_poses[0]));
        reconstruction.poses[2] = Some(outside_poses[0]);
        reconstruction.poses[3] = Some(outside_poses[1]);
        for (idx, xyz) in scene_points.into_iter().enumerate() {
            for image in 0..frames.len() {
                reconstruction.observations[image][idx] = Some(idx);
            }
            reconstruction.points.push(Point3D {
                xyz,
                color: [0, 0, 0],
                error: 0.0,
                track: (0..frames.len())
                    .map(|image| TrackObservation {
                        image,
                        feature: idx,
                    })
                    .collect(),
            });
            reconstruction.point_ids.push(idx as u64 + 1);
        }

        (
            frames,
            reconstruction,
            aux_sensor,
            initial_sensor_from_rig,
            true_sensor_from_rig,
        )
    }

    #[test]
    fn ceres_rig_ba_refines_sensor_from_rig() {
        let (frames, mut reconstruction, aux_sensor, initial_sensor_from_rig, true_sensor_from_rig) =
            rig_sensor_ba_fixture();
        let initial_error = translation_distance(initial_sensor_from_rig, true_sensor_from_rig);

        let report = refine_bundle_adjustment(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 30,
                loss_function: BundleAdjustmentLoss::Huber { scale: 4.0 },
                max_observation_error_px: 200.0,
                variable_images: Some(vec![0, 1]),
                constant_images: vec![2, 3],
                point_ids: Some((0..8).collect()),
                ..BundleAdjustmentOptions::default()
            },
        )
        .expect("ceres rig ba should succeed");

        let refined = sensor_from_rig_pose(&reconstruction, 3, &aux_sensor);
        assert!(report.final_cost < report.initial_cost);
        assert!(report.gradient_max_norm.is_finite());
        assert!(translation_distance(refined, true_sensor_from_rig) < initial_error);
    }

    #[test]
    fn ceres_frame_pose_prior_pulls_rig_aux_camera_center() {
        let (frames, mut reconstruction, ..) = rig_sensor_ba_fixture();
        let start_center = camera_center_world(reconstruction.poses[1].unwrap());
        let prior_center = [
            start_center[0] + 0.35,
            start_center[1] + 0.08,
            start_center[2],
        ];
        let start_error = center_error(start_center, prior_center);

        let report = refine_bundle_adjustment(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 30,
                loss_function: BundleAdjustmentLoss::Huber { scale: 4.0 },
                max_observation_error_px: 200.0,
                variable_images: Some(vec![0, 1]),
                constant_images: vec![2, 3],
                point_ids: Some((0..8).collect()),
                pose_priors: vec![BundleAdjustmentPosePrior::new(1, prior_center)],
                prior_position_fallback_stddev: 0.01,
                ..BundleAdjustmentOptions::default()
            },
        )
        .expect("ceres rig ba with pose prior should succeed");

        let end_center = camera_center_world(reconstruction.poses[1].unwrap());
        let end_error = center_error(end_center, prior_center);
        assert!(report.is_solution_usable());
        assert!(
            end_error < start_error,
            "start_error={start_error} end_error={end_error}"
        );
    }

    #[test]
    fn ceres_pose_prior_without_explicit_gauge_pins_camera_centers() {
        let true_camera = CameraModel::new_pinhole(100, 100, 60.0, 60.0, 50.0, 50.0);
        let poses = [
            SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(0.0, 0.0, 0.0)),
            SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(0.35, 0.0, 0.0)),
            SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(0.7, 0.0, 0.0)),
        ];
        let points = [
            [-0.4, -0.3, 3.0],
            [0.0, -0.2, 3.2],
            [0.35, 0.1, 3.4],
            [0.7, 0.2, 3.1],
        ];
        let mut frames = vec![frame(0), frame(1), frame(2)];
        for image in 0..3 {
            frames[image].keypoints = points
                .iter()
                .map(|&point| {
                    let projected =
                        super::super::shared::project_point(true_camera, poses[image], point)
                            .expect("projected point");
                    KeyPoint::new(projected[0] as f32, projected[1] as f32)
                })
                .collect();
        }
        let mut reconstruction = reconstruction(&frames);
        reconstruction.camera = true_camera;
        reconstruction.cameras = vec![true_camera];
        reconstruction.poses = poses
            .iter()
            .map(|pose| {
                let center = camera_center_world(*pose);
                let shifted_center =
                    Vec3::new(center[0] as f32 + 4.0, center[1] as f32, center[2] as f32);
                Some(crate::geometry::pose_from_rotation_center(
                    Quat::IDENTITY,
                    shifted_center,
                ))
            })
            .collect();
        for (idx, xyz) in points.into_iter().enumerate() {
            for image in 0..3 {
                reconstruction.observations[image][idx] = Some(idx);
            }
            reconstruction.points.push(Point3D {
                xyz: [xyz[0] + 4.0, xyz[1], xyz[2]],
                color: [0, 0, 0],
                error: 0.0,
                track: (0..3)
                    .map(|image| TrackObservation {
                        image,
                        feature: idx,
                    })
                    .collect(),
            });
            reconstruction.point_ids.push(idx as u64 + 1);
        }
        let priors = poses
            .iter()
            .enumerate()
            .map(|(image, pose)| {
                let center = camera_center_world(*pose);
                BundleAdjustmentPosePrior::new(image, [center[0], center[1], center[2]])
            })
            .collect::<Vec<_>>();

        let report = refine_bundle_adjustment(
            &frames,
            &mut reconstruction,
            BundleAdjustmentOptions {
                iterations: 50,
                loss_function: BundleAdjustmentLoss::Trivial,
                max_observation_error_px: 100.0,
                gauge: super::super::BundleAdjustmentGauge::None,
                pose_priors: priors,
                prior_position_fallback_stddev: 0.01,
                ..BundleAdjustmentOptions::default()
            },
        )
        .expect("ceres prior-position BA should succeed");

        assert!(report.is_solution_usable());
        for (image, pose) in poses.iter().enumerate() {
            let expected = camera_center_world(*pose);
            let actual = camera_center_world(reconstruction.poses[image].unwrap());
            assert!(
                center_error(actual, [expected[0], expected[1], expected[2]]) < 0.1,
                "image={image} actual={actual:?} expected={expected:?}"
            );
        }
    }

    fn center_error(center: nalgebra::SVector<f64, 3>, target: [f64; 3]) -> f64 {
        ((center[0] - target[0]).powi(2)
            + (center[1] - target[1]).powi(2)
            + (center[2] - target[2]).powi(2))
        .sqrt()
    }
}
