use crate::types::{CameraModel, ImageFrame, Reconstruction};
use rustslam::SE3;
use std::collections::HashSet;

pub(crate) fn bundle_adjustment_point_filter(
    variable_points: Option<&[usize]>,
    constant_points: Option<&[usize]>,
) -> Option<HashSet<usize>> {
    match (variable_points, constant_points) {
        (None, None) => None,
        (variable_points, constant_points) => {
            let mut filter = HashSet::new();
            if let Some(points) = variable_points {
                filter.extend(points.iter().copied());
            }
            if let Some(points) = constant_points {
                filter.extend(points.iter().copied());
            }
            Some(filter)
        }
    }
}

pub(crate) fn add_three_point_gauge(
    constant_point_filter: &mut HashSet<usize>,
    reconstruction: &Reconstruction,
    observations: &[BaObservation],
) {
    let mut fixed_points = Vec::<[f32; 3]>::new();
    let mut observed_points = observations.iter().map(|obs| obs.point).collect::<Vec<_>>();
    observed_points.sort_unstable();
    observed_points.dedup();

    for point_id in &observed_points {
        if !constant_point_filter.contains(point_id) {
            continue;
        }
        if let Some(point) = reconstruction.points.get(*point_id) {
            maybe_add_gauge_point(&mut fixed_points, point.xyz);
        }
        if fixed_points.len() >= 3 {
            return;
        }
    }

    for point_id in observed_points {
        if constant_point_filter.contains(&point_id) {
            continue;
        }
        let Some(point) = reconstruction.points.get(point_id) else {
            continue;
        };
        if maybe_add_gauge_point(&mut fixed_points, point.xyz) {
            constant_point_filter.insert(point_id);
            if fixed_points.len() >= 3 {
                return;
            }
        }
    }
}

fn maybe_add_gauge_point(points: &mut Vec<[f32; 3]>, candidate: [f32; 3]) -> bool {
    if points.len() >= 3 || !candidate.iter().all(|value| value.is_finite()) {
        return false;
    }
    let independent = match points.len() {
        0 => true,
        1 => distance3(points[0], candidate) > 1.0e-9,
        2 => triangle_area2(points[0], points[1], candidate) > 1.0e-12,
        _ => false,
    };
    if independent {
        points.push(candidate);
    }
    independent
}

fn distance3(left: [f32; 3], right: [f32; 3]) -> f64 {
    let dx = left[0] as f64 - right[0] as f64;
    let dy = left[1] as f64 - right[1] as f64;
    let dz = left[2] as f64 - right[2] as f64;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn triangle_area2(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f64 {
    let ab = [
        b[0] as f64 - a[0] as f64,
        b[1] as f64 - a[1] as f64,
        b[2] as f64 - a[2] as f64,
    ];
    let ac = [
        c[0] as f64 - a[0] as f64,
        c[1] as f64 - a[1] as f64,
        c[2] as f64 - a[2] as f64,
    ];
    let cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]
}

pub(crate) struct BaObservation {
    pub(crate) image: usize,
    pub(crate) point: usize,
    pub(crate) xy: [f64; 2],
}

pub(crate) fn collect_observations(
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
    max_error_px: f64,
    point_filter: Option<&HashSet<usize>>,
    allow_single_observation_points: bool,
) -> Vec<BaObservation> {
    let mut observations = Vec::new();
    for (point_id, point) in reconstruction.points.iter().enumerate() {
        if point_filter.is_some_and(|filter| !filter.contains(&point_id)) {
            continue;
        }
        if !allow_single_observation_points && point.track.len() < 2 {
            continue;
        }
        for obs in &point.track {
            if obs.image >= reconstruction.poses.len()
                || obs.image >= frames.len()
                || reconstruction.poses[obs.image].is_none()
                || obs.feature >= frames[obs.image].keypoints.len()
            {
                continue;
            }
            let kp = &frames[obs.image].keypoints[obs.feature];
            let Some(pose) = reconstruction.poses[obs.image] else {
                continue;
            };
            let Some(predicted) =
                project_point(reconstruction.camera_for_image(obs.image), pose, point.xyz)
            else {
                continue;
            };
            let err = ((predicted[0] - kp.x() as f64).powi(2)
                + (predicted[1] - kp.y() as f64).powi(2))
            .sqrt();
            if err.is_finite() && err <= max_error_px {
                observations.push(BaObservation {
                    image: obs.image,
                    point: point_id,
                    xy: [kp.x() as f64, kp.y() as f64],
                });
            }
        }
    }
    observations
}

pub(crate) fn project_point(camera: CameraModel, pose: SE3, point: [f32; 3]) -> Option<[f64; 2]> {
    let p = pose.transform_point(&point);
    camera.img_from_cam(p[0] as f64, p[1] as f64, p[2] as f64)
}

pub(crate) fn refresh_point_errors(frames: &[ImageFrame], reconstruction: &mut Reconstruction) {
    let image_cameras = (0..reconstruction.poses.len())
        .map(|image| reconstruction.camera_for_image(image))
        .collect::<Vec<_>>();
    for point in &mut reconstruction.points {
        let mut total = 0.0f32;
        let mut count = 0usize;
        for obs in &point.track {
            let Some(pose) = reconstruction.poses.get(obs.image).copied().flatten() else {
                continue;
            };
            if obs.image >= frames.len() || obs.feature >= frames[obs.image].keypoints.len() {
                continue;
            }
            let kp = &frames[obs.image].keypoints[obs.feature];
            if let Some(predicted) = project_point(image_cameras[obs.image], pose, point.xyz) {
                let err = ((predicted[0] - kp.x() as f64).powi(2)
                    + (predicted[1] - kp.y() as f64).powi(2))
                .sqrt();
                if err.is_finite() {
                    total += err as f32;
                    count += 1;
                }
            }
        }
        if count > 0 {
            point.error = total / count as f32;
        }
    }
}
