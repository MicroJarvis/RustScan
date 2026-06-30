//! Triangulate multi-view feature tracks given fixed global camera poses.
//!
//! This is the point-establishment stage of a GLOMAP-style global mapper: once
//! global rotations/translations and multi-view tracks are known, each track is
//! triangulated with COLMAP's multi-view DLT
//! ([`crate::triangulation::triangulate_multi_view_point`]) and filtered by
//! triangulation angle, cheirality, and reprojection error before being added
//! to a [`crate::types::Reconstruction`].

use crate::track_establishment::{FeatureNode, Track};
use crate::triangulation::{calculate_triangulation_angle, triangulate_multi_view_point};
use crate::types::{ImageFrame, Point3D, Reconstruction, TrackObservation};
use nalgebra::{Matrix3x4, Vector2, Vector3};
use rustslam::SE3;
use std::collections::HashSet;

/// Options controlling [`triangulate_tracks`].
#[derive(Debug, Clone, Copy)]
pub struct TrackTriangulationOptions {
    /// Minimum triangulation angle (degrees) between any observation pair.
    pub min_triangulation_angle_deg: f32,
    /// Maximum mean reprojection error (pixels) across track observations.
    pub max_reprojection_error_px: f32,
    /// Reject points with non-positive depth in any observing camera.
    pub require_cheirality: bool,
}

impl Default for TrackTriangulationOptions {
    fn default() -> Self {
        Self {
            min_triangulation_angle_deg: 1.5,
            max_reprojection_error_px: 4.0,
            require_cheirality: true,
        }
    }
}

/// Statistics from a track triangulation run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TrackTriangulationStats {
    pub num_attempted: usize,
    pub num_triangulated: usize,
    pub num_rejected_pose: usize,
    pub num_rejected_triangulation: usize,
    pub num_rejected_angle: usize,
    pub num_rejected_cheirality: usize,
    pub num_rejected_reprojection: usize,
}

/// Triangulate feature tracks and append accepted 3D points to `reconstruction`.
///
/// `reconstruction.poses` must already contain the global poses for registered
/// views. On success, `reconstruction.points`, `reconstruction.point_ids`, and
/// `reconstruction.observations` are updated for each accepted track.
pub fn triangulate_tracks(
    tracks: &[Track],
    frames: &[ImageFrame],
    reconstruction: &mut Reconstruction,
    options: &TrackTriangulationOptions,
) -> TrackTriangulationStats {
    let mut stats = TrackTriangulationStats::default();
    stats.num_attempted = tracks.len();

    for track in tracks {
        let observations = track
            .observations
            .iter()
            .map(|node| TrackObservation {
                image: node.image,
                feature: node.feature,
            })
            .collect::<Vec<_>>();

        if !track_has_registered_poses(&observations, reconstruction) {
            stats.num_rejected_pose += 1;
            continue;
        }

        let Some(xyz) = triangulate_track_observations(&observations, frames, reconstruction)
        else {
            stats.num_rejected_triangulation += 1;
            continue;
        };

        if !track_has_min_triangulation_angle(
            xyz,
            &observations,
            reconstruction,
            options.min_triangulation_angle_deg,
        ) {
            stats.num_rejected_angle += 1;
            continue;
        }

        if options.require_cheirality && !track_is_cheiral(xyz, &observations, reconstruction) {
            stats.num_rejected_cheirality += 1;
            continue;
        }

        let Some(error) = mean_track_reprojection_error(xyz, &observations, frames, reconstruction)
        else {
            stats.num_rejected_reprojection += 1;
            continue;
        };
        if error > options.max_reprojection_error_px {
            stats.num_rejected_reprojection += 1;
            continue;
        }

        add_point3d_to_reconstruction(reconstruction, xyz, error, &observations, frames);
        stats.num_triangulated += 1;
    }

    stats
}

fn triangulate_track_observations(
    track: &[TrackObservation],
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
) -> Option<[f32; 3]> {
    let mut cams_from_world = Vec::with_capacity(track.len());
    let mut cam_points = Vec::with_capacity(track.len());
    let mut seen_images = HashSet::new();
    for obs in track {
        if !seen_images.insert(obs.image) {
            continue;
        }
        let pose = (*reconstruction.poses.get(obs.image)?)?;
        let kp = frames.get(obs.image)?.keypoints.get(obs.feature)?;
        let xy = reconstruction
            .camera_for_image(obs.image)
            .cam_from_img_f32(kp.x(), kp.y())?;
        cams_from_world.push(se3_to_matrix3x4(pose));
        cam_points.push(Vector2::new(xy[0] as f64, xy[1] as f64));
    }
    if cam_points.len() < 2 {
        return None;
    }
    let xyz = triangulate_multi_view_point(&cams_from_world, &cam_points)?;
    xyz.iter()
        .all(|v| v.is_finite())
        .then_some([xyz[0] as f32, xyz[1] as f32, xyz[2] as f32])
}

fn track_has_registered_poses(track: &[TrackObservation], reconstruction: &Reconstruction) -> bool {
    track.iter().all(|obs| {
        reconstruction
            .poses
            .get(obs.image)
            .and_then(|pose| *pose)
            .is_some()
    })
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
    let point3 = Vector3::new(point[0] as f64, point[1] as f64, point[2] as f64);
    let mut best = 0.0f64;
    for i in 0..track.len() {
        for j in i + 1..track.len() {
            let Some(pose_i) = reconstruction.poses[track[i].image] else {
                continue;
            };
            let Some(pose_j) = reconstruction.poses[track[j].image] else {
                continue;
            };
            let c1 = camera_center_vec3(pose_i);
            let c2 = camera_center_vec3(pose_j);
            let angle_rad = calculate_triangulation_angle(&c1, &c2, &point3);
            best = best.max(angle_rad);
        }
    }
    best.to_degrees() >= min_angle_deg as f64
}

fn track_is_cheiral(
    point: [f32; 3],
    track: &[TrackObservation],
    reconstruction: &Reconstruction,
) -> bool {
    track.iter().all(|obs| {
        let Some(pose) = reconstruction.poses[obs.image] else {
            return false;
        };
        let cam = pose.transform_point(&point);
        cam[2] > 1.0e-6
    })
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

fn add_point3d_to_reconstruction(
    reconstruction: &mut Reconstruction,
    xyz: [f32; 3],
    error: f32,
    track: &[TrackObservation],
    frames: &[ImageFrame],
) {
    let point_id = reconstruction.points.len();
    for obs in track {
        if obs.image >= reconstruction.observations.len() {
            continue;
        }
        if obs.feature >= reconstruction.observations[obs.image].len() {
            reconstruction.observations[obs.image].resize(obs.feature + 1, None);
        }
        reconstruction.observations[obs.image][obs.feature] = Some(point_id);
    }
    reconstruction.point_ids.push(point_id as u64 + 1);
    reconstruction.points.push(Point3D {
        xyz,
        color: average_track_color(track, frames),
        error,
        track: track.to_vec(),
    });
}

fn average_track_color(track: &[TrackObservation], frames: &[ImageFrame]) -> [u8; 3] {
    let mut color = [0usize; 3];
    let mut count = 0usize;
    for obs in track {
        let Some(frame) = frames.get(obs.image) else {
            continue;
        };
        let Some(sample) = frame.colors.get(obs.feature) else {
            continue;
        };
        color[0] += sample[0] as usize;
        color[1] += sample[1] as usize;
        color[2] += sample[2] as usize;
        count += 1;
    }
    if count == 0 {
        return [128, 128, 128];
    }
    [
        (color[0] / count) as u8,
        (color[1] / count) as u8,
        (color[2] / count) as u8,
    ]
}

fn camera_center_vec3(pose: SE3) -> Vector3<f64> {
    let c = crate::geometry::camera_center(pose);
    Vector3::new(c.x as f64, c.y as f64, c.z as f64)
}

fn se3_to_matrix3x4(pose: SE3) -> Matrix3x4<f64> {
    let r = pose.rotation_matrix();
    let t = pose.translation();
    Matrix3x4::new(
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

/// Convert a [`Track`] to [`TrackObservation`]s (same indices, stable ordering).
pub fn track_to_observations(track: &Track) -> Vec<TrackObservation> {
    track
        .observations
        .iter()
        .map(|FeatureNode { image, feature }| TrackObservation {
            image: *image,
            feature: *feature,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CameraModel;
    use glam::{Quat, Vec3};
    use std::path::PathBuf;

    fn test_camera() -> CameraModel {
        CameraModel::new_pinhole(640, 480, 500.0, 500.0, 320.0, 240.0)
    }

    fn project_keypoint(camera: CameraModel, pose: SE3, point: [f32; 3]) -> rustslam::KeyPoint {
        let p = pose.transform_point(&point);
        let xy = camera
            .img_from_cam(p[0] as f64, p[1] as f64, p[2] as f64)
            .unwrap();
        rustslam::KeyPoint::new(xy[0] as f32, xy[1] as f32)
    }

    fn test_frame(id: usize, keypoints: Vec<rustslam::KeyPoint>) -> ImageFrame {
        let colors = vec![[200, 100, 50]; keypoints.len()];
        ImageFrame {
            id,
            name: format!("img_{id:03}.jpg"),
            path: PathBuf::from(format!("img_{id:03}.jpg")),
            width: 640,
            height: 480,
            keypoints,
            descriptors: rustslam::Descriptors::new(),
            sift: crate::sift::SiftFeatures::default(),
            wide_descriptors: crate::wide::WideDescriptors {
                data: Vec::new(),
                dim: 0,
                count: 0,
            },
            strong_feature_indices: Vec::new(),
            colors,
        }
    }

    fn empty_reconstruction(num_views: usize, poses: Vec<Option<SE3>>) -> Reconstruction {
        Reconstruction {
            camera: test_camera(),
            cameras: vec![test_camera()],
            camera_ids: vec![1],
            rigs: Vec::new(),
            frames: Vec::new(),
            image_names: (0..num_views)
                .map(|idx| format!("img_{idx:03}.jpg"))
                .collect(),
            image_paths: (0..num_views)
                .map(|idx| PathBuf::from(format!("img_{idx:03}.jpg")))
                .collect(),
            image_ids: (0..num_views).map(|idx| idx as u32 + 1).collect(),
            image_camera_indices: vec![0; num_views],
            image_frame_indices: vec![None; num_views],
            poses,
            observations: vec![Vec::new(); num_views],
            keypoints: vec![Vec::new(); num_views],
            point_ids: Vec::new(),
            points: Vec::new(),
        }
    }

    #[test]
    fn triangulates_synthetic_track_with_known_point() {
        let camera = test_camera();
        let point = [0.0, 0.0, 5.0];
        let n = 4;
        let mut frames = Vec::new();
        let mut poses = Vec::new();
        for view in 0..n {
            let pose =
                SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(view as f32 * 0.25, 0.0, 0.0));
            poses.push(Some(pose));
            frames.push(test_frame(
                view,
                vec![project_keypoint(camera, pose, point)],
            ));
        }

        let mut reconstruction = empty_reconstruction(n, poses);
        for (view, frame) in frames.iter().enumerate() {
            reconstruction.keypoints[view] = frame.keypoints.clone();
            reconstruction.observations[view] = vec![None; frame.keypoints.len()];
        }

        let track = Track {
            observations: (0..n).map(|view| FeatureNode::new(view, 0)).collect(),
        };
        let stats = triangulate_tracks(
            std::slice::from_ref(&track),
            &frames,
            &mut reconstruction,
            &TrackTriangulationOptions {
                min_triangulation_angle_deg: 0.5,
                max_reprojection_error_px: 2.0,
                require_cheirality: true,
            },
        );
        assert_eq!(stats.num_triangulated, 1);
        assert_eq!(reconstruction.points.len(), 1);
        let recovered = reconstruction.points[0].xyz;
        let err = Vec3::from_array(recovered).distance(Vec3::from_array(point));
        assert!(err < 1.0e-2, "triangulation error {err}");
    }

    #[test]
    fn rejects_track_with_missing_pose() {
        let camera = test_camera();
        let point = [0.0, 0.0, 2.0];
        let pose0 = SE3::identity();
        let pose1 = SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(0.5, 0.0, 0.0));
        let frames = vec![
            test_frame(0, vec![project_keypoint(camera, pose0, point)]),
            test_frame(1, vec![project_keypoint(camera, pose1, point)]),
        ];
        let mut reconstruction = empty_reconstruction(2, vec![Some(pose0), None]);
        for (view, frame) in frames.iter().enumerate() {
            reconstruction.keypoints[view] = frame.keypoints.clone();
            reconstruction.observations[view] = vec![None; frame.keypoints.len()];
        }
        let track = Track {
            observations: vec![FeatureNode::new(0, 0), FeatureNode::new(1, 0)],
        };
        let stats = triangulate_tracks(
            std::slice::from_ref(&track),
            &frames,
            &mut reconstruction,
            &TrackTriangulationOptions::default(),
        );
        assert_eq!(stats.num_triangulated, 0);
        assert_eq!(stats.num_rejected_pose, 1);
    }
}
