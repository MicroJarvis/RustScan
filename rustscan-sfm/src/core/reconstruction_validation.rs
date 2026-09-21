//! Structural checks and occupied-ID allocation for [`Reconstruction`](crate::types::Reconstruction).

use crate::types::{
    CameraModel, DataId, Frame, Point3D, Reconstruction, Rig, SensorId, SensorType,
    COLMAP_MAX_CAMERA_PARAMS,
};
use rustscan_slam::{KeyPoint, SE3};
use std::collections::{BTreeSet, HashMap, HashSet};
use thiserror::Error;

/// Exclusive upper bound used by the COLMAP database `image_id` check.
pub const COLMAP_ID_EXCLUSIVE_LIMIT: u64 = 2_147_483_647;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PersistentIdDomain {
    /// Camera, image, rig, and frame IDs: `1..COLMAP_ID_EXCLUSIVE_LIMIT`.
    ColmapRecord,
    /// Point3D IDs: any non-zero `u64`.
    Point3D,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[error("{record_type} ids={ids} field={field} reason={reason}")]
pub struct ReconstructionValidationError {
    pub record_type: &'static str,
    pub ids: String,
    pub field: &'static str,
    pub reason: String,
}

fn validation_error(
    record_type: &'static str,
    ids: impl Into<String>,
    field: &'static str,
    reason: impl Into<String>,
) -> ReconstructionValidationError {
    ReconstructionValidationError {
        record_type,
        ids: ids.into(),
        field,
        reason: reason.into(),
    }
}

pub(crate) fn lookup_error(
    record_type: &'static str,
    ids: impl Into<String>,
    field: &'static str,
    reason: impl Into<String>,
) -> ReconstructionValidationError {
    validation_error(record_type, ids, field, reason)
}

/// Checks reconstruction invariants that do not depend on a COLMAP export.
pub fn validate_structure(
    reconstruction: &Reconstruction,
) -> Result<(), ReconstructionValidationError> {
    validate_parallel_lengths(reconstruction)?;
    validate_identifiers(reconstruction)?;
    validate_camera_indexes(reconstruction)?;
    validate_observation_lengths(reconstruction)?;
    validate_observation_points(reconstruction)?;
    validate_tracks(reconstruction)?;
    validate_observation_track_agreement(reconstruction)?;
    validate_rig_frame_references(reconstruction)?;
    validate_finite_values(reconstruction)?;
    Ok(())
}

/// Structural validation plus export-only registered-image checks.
pub fn validate_for_colmap_export(
    reconstruction: &Reconstruction,
) -> Result<(), ReconstructionValidationError> {
    validate_structure(reconstruction)?;
    for (point_index, point) in reconstruction.points.iter().enumerate() {
        for observation in &point.track {
            if reconstruction
                .poses
                .get(observation.image)
                .and_then(|pose| pose.as_ref())
                .is_none()
            {
                return Err(validation_error(
                    "point",
                    format!(
                        "point_index={point_index},point_id={},image_index={}",
                        reconstruction.point_ids[point_index], observation.image
                    ),
                    "track",
                    "track references an image that is not registered for export",
                ));
            }
        }
    }
    Ok(())
}

fn validate_parallel_lengths(
    reconstruction: &Reconstruction,
) -> Result<(), ReconstructionValidationError> {
    let image_count = reconstruction.image_names.len();
    let image_fields: [(&str, usize); 7] = [
        ("image_paths", reconstruction.image_paths.len()),
        ("image_ids", reconstruction.image_ids.len()),
        (
            "image_camera_indices",
            reconstruction.image_camera_indices.len(),
        ),
        (
            "image_frame_indices",
            reconstruction.image_frame_indices.len(),
        ),
        ("poses", reconstruction.poses.len()),
        ("observations", reconstruction.observations.len()),
        ("keypoints", reconstruction.keypoints.len()),
    ];
    for (field, actual) in image_fields {
        if actual != image_count {
            return Err(validation_error(
                "image",
                format!("expected={image_count},actual={actual}"),
                field,
                "parallel image metadata lengths differ",
            ));
        }
    }
    if reconstruction.camera_ids.len() != reconstruction.cameras.len() {
        return Err(validation_error(
            "camera",
            format!(
                "cameras={},camera_ids={}",
                reconstruction.cameras.len(),
                reconstruction.camera_ids.len()
            ),
            "camera_ids",
            "camera and camera id lengths differ",
        ));
    }
    if reconstruction.point_ids.len() != reconstruction.points.len() {
        return Err(validation_error(
            "point",
            format!(
                "points={},point_ids={}",
                reconstruction.points.len(),
                reconstruction.point_ids.len()
            ),
            "point_ids",
            "point and point id lengths differ",
        ));
    }
    Ok(())
}

fn validate_identifiers(
    reconstruction: &Reconstruction,
) -> Result<(), ReconstructionValidationError> {
    require_unique_ids(
        "camera",
        "camera_id",
        PersistentIdDomain::ColmapRecord,
        reconstruction.camera_ids.iter().map(|id| u64::from(*id)),
    )?;
    require_unique_ids(
        "image",
        "image_id",
        PersistentIdDomain::ColmapRecord,
        reconstruction.image_ids.iter().map(|id| u64::from(*id)),
    )?;
    require_unique_ids(
        "point",
        "point_id",
        PersistentIdDomain::Point3D,
        reconstruction.point_ids.iter().copied(),
    )?;
    require_unique_ids(
        "rig",
        "rig_id",
        PersistentIdDomain::ColmapRecord,
        reconstruction.rigs.iter().map(|rig| u64::from(rig.rig_id)),
    )?;
    require_unique_ids(
        "frame",
        "frame_id",
        PersistentIdDomain::ColmapRecord,
        reconstruction
            .frames
            .iter()
            .map(|frame| u64::from(frame.frame_id)),
    )?;
    Ok(())
}

fn require_unique_ids(
    record_type: &'static str,
    field: &'static str,
    domain: PersistentIdDomain,
    ids: impl IntoIterator<Item = u64>,
) -> Result<(), ReconstructionValidationError> {
    let mut seen = HashSet::new();
    for id in ids {
        ensure_id_legal(domain, id, record_type, field)?;
        if !seen.insert(id) {
            return Err(validation_error(
                record_type,
                format!("{field}={id}"),
                field,
                "duplicate id",
            ));
        }
    }
    Ok(())
}

fn ensure_id_legal(
    domain: PersistentIdDomain,
    id: u64,
    record_type: &'static str,
    field: &'static str,
) -> Result<(), ReconstructionValidationError> {
    if id == 0 {
        return Err(validation_error(
            record_type,
            format!("{field}=0"),
            field,
            "id 0 is not a legal COLMAP id",
        ));
    }
    if domain == PersistentIdDomain::ColmapRecord && id >= COLMAP_ID_EXCLUSIVE_LIMIT {
        return Err(validation_error(
            record_type,
            format!("{field}={id}"),
            field,
            format!("id is outside the COLMAP range 1..{COLMAP_ID_EXCLUSIVE_LIMIT}"),
        ));
    }
    Ok(())
}

fn validate_camera_indexes(
    reconstruction: &Reconstruction,
) -> Result<(), ReconstructionValidationError> {
    for (image_index, &camera_index) in reconstruction.image_camera_indices.iter().enumerate() {
        if camera_index >= reconstruction.cameras.len() {
            return Err(validation_error(
                "image",
                format!(
                    "image_index={image_index},image_id={},camera_index={camera_index}",
                    reconstruction.image_ids[image_index]
                ),
                "image_camera_indices",
                "camera index does not reference a camera",
            ));
        }
    }
    Ok(())
}

fn validate_observation_lengths(
    reconstruction: &Reconstruction,
) -> Result<(), ReconstructionValidationError> {
    for (image_index, observations) in reconstruction.observations.iter().enumerate() {
        let keypoint_count = reconstruction.keypoints[image_index].len();
        if observations.len() != keypoint_count {
            return Err(validation_error(
                "image",
                format!(
                    "image_index={image_index},image_id={},observations={},keypoints={keypoint_count}",
                    reconstruction.image_ids[image_index],
                    observations.len()
                ),
                "observations",
                "observation count does not match keypoint count",
            ));
        }
    }
    Ok(())
}

fn validate_observation_points(
    reconstruction: &Reconstruction,
) -> Result<(), ReconstructionValidationError> {
    for (image_index, observations) in reconstruction.observations.iter().enumerate() {
        for (feature_index, observation) in observations.iter().enumerate() {
            if let Some(point_index) = observation {
                if *point_index >= reconstruction.points.len() {
                    return Err(validation_error(
                        "observation",
                        format!(
                            "image_index={image_index},image_id={},feature_index={feature_index},point_index={point_index}",
                            reconstruction.image_ids[image_index]
                        ),
                        "point_index",
                        "observation references a missing point",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn validate_tracks(reconstruction: &Reconstruction) -> Result<(), ReconstructionValidationError> {
    let mut owners = HashSet::new();
    for (point_index, point) in reconstruction.points.iter().enumerate() {
        for observation in &point.track {
            if observation.image >= reconstruction.observations.len() {
                return Err(validation_error(
                    "point",
                    format!(
                        "point_index={point_index},point_id={},image_index={}",
                        reconstruction.point_ids[point_index], observation.image
                    ),
                    "track",
                    "track image index is out of range",
                ));
            }
            if observation.feature >= reconstruction.observations[observation.image].len() {
                return Err(validation_error(
                    "point",
                    format!(
                        "point_index={point_index},point_id={},image_index={},feature_index={}",
                        reconstruction.point_ids[point_index],
                        observation.image,
                        observation.feature
                    ),
                    "track",
                    "track feature index is out of range",
                ));
            }
            if !owners.insert((observation.image, observation.feature)) {
                return Err(validation_error(
                    "point",
                    format!(
                        "point_index={point_index},point_id={},image_index={},feature_index={}",
                        reconstruction.point_ids[point_index],
                        observation.image,
                        observation.feature
                    ),
                    "track",
                    "image feature belongs to more than one point track",
                ));
            }
        }
    }
    Ok(())
}

fn validate_observation_track_agreement(
    reconstruction: &Reconstruction,
) -> Result<(), ReconstructionValidationError> {
    let mut track_owner = HashMap::<(usize, usize), usize>::new();
    for (point_index, point) in reconstruction.points.iter().enumerate() {
        for observation in &point.track {
            track_owner.insert((observation.image, observation.feature), point_index);
        }
    }
    for (image_index, observations) in reconstruction.observations.iter().enumerate() {
        for (feature_index, observation) in observations.iter().enumerate() {
            let tracked = track_owner.get(&(image_index, feature_index)).copied();
            match (observation, tracked) {
                (Some(point_index), Some(track_point)) if *point_index == track_point => {}
                (Some(point_index), Some(track_point)) => {
                    return Err(validation_error(
                        "observation",
                        format!(
                            "image_index={image_index},image_id={},feature_index={feature_index},observation_point={point_index},track_point={track_point}",
                            reconstruction.image_ids[image_index]
                        ),
                        "point_index",
                        "observation and track point disagree",
                    ));
                }
                (Some(point_index), None) => {
                    return Err(validation_error(
                        "observation",
                        format!(
                            "image_index={image_index},image_id={},feature_index={feature_index},point_index={point_index}",
                            reconstruction.image_ids[image_index]
                        ),
                        "observations",
                        "observation has no matching point track",
                    ));
                }
                (None, Some(track_point)) => {
                    return Err(validation_error(
                        "point",
                        format!(
                            "point_index={track_point},point_id={},image_index={image_index},feature_index={feature_index}",
                            reconstruction.point_ids[track_point]
                        ),
                        "track",
                        "track has no matching observation",
                    ));
                }
                (None, None) => {}
            }
        }
    }
    Ok(())
}

fn validate_rig_frame_references(
    reconstruction: &Reconstruction,
) -> Result<(), ReconstructionValidationError> {
    for (frame_index, frame) in reconstruction.frames.iter().enumerate() {
        let Some(rig) = reconstruction
            .rigs
            .iter()
            .find(|rig| rig.rig_id == frame.rig_id)
        else {
            return Err(validation_error(
                "frame",
                format!(
                    "frame_index={frame_index},frame_id={},rig_id={}",
                    frame.frame_id, frame.rig_id
                ),
                "rig_id",
                "frame rig id does not reference a rig",
            ));
        };
        for data_id in &frame.data_ids {
            validate_frame_data(reconstruction, frame_index, frame, rig, data_id)?;
        }
    }
    for (image_index, frame_index) in reconstruction.image_frame_indices.iter().enumerate() {
        let Some(frame_index) = frame_index else {
            continue;
        };
        let Some(frame) = reconstruction.frames.get(*frame_index) else {
            return Err(validation_error(
                "image",
                format!(
                    "image_index={image_index},image_id={},frame_index={frame_index}",
                    reconstruction.image_ids[image_index]
                ),
                "image_frame_indices",
                "image frame index does not reference a frame",
            ));
        };
        let image_id = u64::from(reconstruction.image_ids[image_index]);
        if !frame.data_ids.iter().any(|data| {
            data.sensor_id.sensor_type == SensorType::Camera && data.data_id == image_id
        }) {
            return Err(validation_error(
                "image",
                format!(
                    "image_index={image_index},image_id={image_id},frame_id={}",
                    frame.frame_id
                ),
                "image_frame_indices",
                "image frame does not list this image data id",
            ));
        }
    }
    Ok(())
}

fn validate_frame_data(
    reconstruction: &Reconstruction,
    frame_index: usize,
    frame: &Frame,
    rig: &Rig,
    data_id: &DataId,
) -> Result<(), ReconstructionValidationError> {
    if data_id.data_id == 0 {
        return Err(validation_error(
            "frame",
            format!(
                "frame_index={frame_index},frame_id={},data_id=0",
                frame.frame_id
            ),
            "data_id",
            "data id 0 is not resolvable",
        ));
    }
    if data_id.sensor_id.sensor_type == SensorType::Invalid
        || !sensor_on_rig(rig, &data_id.sensor_id)
    {
        return Err(validation_error(
            "frame",
            format!(
                "frame_index={frame_index},frame_id={},sensor_id={},sensor_type={:?}",
                frame.frame_id, data_id.sensor_id.sensor_id, data_id.sensor_id.sensor_type
            ),
            "sensor_id",
            "frame sensor is not on the referenced rig",
        ));
    }
    if data_id.sensor_id.sensor_type == SensorType::Camera {
        let Some(image_index) = reconstruction
            .image_ids
            .iter()
            .position(|image_id| u64::from(*image_id) == data_id.data_id)
        else {
            return Err(validation_error(
                "frame",
                format!(
                    "frame_index={frame_index},frame_id={},data_id={}",
                    frame.frame_id, data_id.data_id
                ),
                "data_id",
                "camera data id does not reference an image",
            ));
        };
        if reconstruction.image_frame_indices[image_index] != Some(frame_index) {
            return Err(validation_error(
                "frame",
                format!(
                    "frame_index={frame_index},frame_id={},image_index={image_index},data_id={}",
                    frame.frame_id, data_id.data_id
                ),
                "data_id",
                "camera data id does not point back at this frame",
            ));
        }
    }
    Ok(())
}

fn sensor_on_rig(rig: &Rig, sensor_id: &SensorId) -> bool {
    rig.ref_sensor_id
        .as_ref()
        .is_some_and(|reference| reference == sensor_id)
        || rig
            .sensors
            .iter()
            .any(|sensor| &sensor.sensor_id == sensor_id)
}

fn validate_finite_values(
    reconstruction: &Reconstruction,
) -> Result<(), ReconstructionValidationError> {
    validate_camera_finite("camera", "legacy", &reconstruction.camera)?;
    for (camera_index, camera) in reconstruction.cameras.iter().enumerate() {
        validate_camera_finite(
            "camera",
            &format!(
                "camera_index={camera_index},camera_id={}",
                reconstruction.camera_ids[camera_index]
            ),
            camera,
        )?;
    }
    for (image_index, pose) in reconstruction.poses.iter().enumerate() {
        if let Some(pose) = pose {
            validate_pose_finite(reconstruction, image_index, pose)?;
        }
    }
    for (image_index, keypoints) in reconstruction.keypoints.iter().enumerate() {
        for (feature_index, keypoint) in keypoints.iter().enumerate() {
            validate_keypoint_finite(reconstruction, image_index, feature_index, keypoint)?;
        }
    }
    for (point_index, point) in reconstruction.points.iter().enumerate() {
        validate_point_finite(reconstruction, point_index, point)?;
    }
    for (frame_index, frame) in reconstruction.frames.iter().enumerate() {
        if !frame
            .rig_from_world
            .qvec
            .iter()
            .all(|value| value.is_finite())
            || !frame
                .rig_from_world
                .tvec
                .iter()
                .all(|value| value.is_finite())
        {
            return Err(validation_error(
                "frame",
                format!("frame_index={frame_index},frame_id={}", frame.frame_id),
                "rig_from_world",
                "frame pose is non-finite",
            ));
        }
    }
    Ok(())
}

fn validate_camera_finite(
    record_type: &'static str,
    ids: &str,
    camera: &CameraModel,
) -> Result<(), ReconstructionValidationError> {
    if camera.num_params > COLMAP_MAX_CAMERA_PARAMS {
        return Err(validation_error(
            record_type,
            ids,
            "num_params",
            "camera parameter count exceeds the fixed parameter array",
        ));
    }
    let scalars = [camera.fx, camera.fy, camera.cx, camera.cy];
    if scalars.iter().any(|value| !value.is_finite()) {
        return Err(validation_error(
            record_type,
            ids,
            "fx",
            "camera intrinsics are non-finite",
        ));
    }
    if camera.params[..camera.num_params]
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(validation_error(
            record_type,
            ids,
            "params",
            "camera parameters are non-finite",
        ));
    }
    Ok(())
}

fn validate_pose_finite(
    reconstruction: &Reconstruction,
    image_index: usize,
    pose: &SE3,
) -> Result<(), ReconstructionValidationError> {
    if pose.translation().iter().any(|value| !value.is_finite()) {
        return Err(validation_error(
            "image",
            format!(
                "image_index={image_index},image_id={}",
                reconstruction.image_ids[image_index]
            ),
            "translation",
            "pose translation is non-finite",
        ));
    }
    if pose.quaternion().iter().any(|value| !value.is_finite()) {
        return Err(validation_error(
            "image",
            format!(
                "image_index={image_index},image_id={}",
                reconstruction.image_ids[image_index]
            ),
            "quaternion",
            "pose quaternion is non-finite",
        ));
    }
    Ok(())
}

fn validate_keypoint_finite(
    reconstruction: &Reconstruction,
    image_index: usize,
    feature_index: usize,
    keypoint: &KeyPoint,
) -> Result<(), ReconstructionValidationError> {
    let values = [
        keypoint.pt.0,
        keypoint.pt.1,
        keypoint.size,
        keypoint.angle,
        keypoint.response,
    ];
    if values.iter().any(|value| !value.is_finite()) {
        return Err(validation_error(
            "keypoint",
            format!(
                "image_index={image_index},image_id={},feature_index={feature_index}",
                reconstruction.image_ids[image_index]
            ),
            "pt",
            "keypoint is non-finite",
        ));
    }
    Ok(())
}

fn validate_point_finite(
    reconstruction: &Reconstruction,
    point_index: usize,
    point: &Point3D,
) -> Result<(), ReconstructionValidationError> {
    if point.xyz.iter().any(|value| !value.is_finite()) {
        return Err(validation_error(
            "point",
            format!(
                "point_index={point_index},point_id={}",
                reconstruction.point_ids[point_index]
            ),
            "xyz",
            "point position is non-finite",
        ));
    }
    if !point.error.is_finite() {
        return Err(validation_error(
            "point",
            format!(
                "point_index={point_index},point_id={}",
                reconstruction.point_ids[point_index]
            ),
            "error",
            "point error is non-finite",
        ));
    }
    Ok(())
}

/// Allocates persistent IDs from the occupied set, never from a vector index.
#[derive(Debug, Clone)]
pub struct OccupiedIdAllocator {
    domain: PersistentIdDomain,
    occupied: BTreeSet<u64>,
    /// Smallest id that may be issued. This never decreases, so deleted ids
    /// are not reused.
    lower_bound: u64,
    overflowed: bool,
    next: Option<u64>,
}

impl OccupiedIdAllocator {
    pub fn new(domain: PersistentIdDomain) -> Self {
        Self {
            domain,
            occupied: BTreeSet::new(),
            lower_bound: 1,
            overflowed: false,
            next: Some(1),
        }
    }

    pub fn replace_occupied(
        &mut self,
        ids: impl IntoIterator<Item = u64>,
    ) -> Result<(), ReconstructionValidationError> {
        let mut occupied = BTreeSet::new();
        for id in ids {
            let field = match self.domain {
                PersistentIdDomain::ColmapRecord => "id",
                PersistentIdDomain::Point3D => "point_id",
            };
            let record_type = match self.domain {
                PersistentIdDomain::ColmapRecord => "record",
                PersistentIdDomain::Point3D => "point",
            };
            ensure_id_legal(self.domain, id, record_type, field)?;
            if !occupied.insert(id) {
                return Err(validation_error(
                    record_type,
                    format!("{field}={id}"),
                    field,
                    "duplicate occupied id",
                ));
            }
        }
        self.occupied = occupied;
        let mut lower_bound = self.lower_bound;
        let mut overflowed = self.overflowed;
        for id in &self.occupied {
            match id.checked_add(1) {
                Some(after) => lower_bound = lower_bound.max(after),
                None => overflowed = true,
            }
        }
        self.lower_bound = lower_bound;
        self.overflowed = overflowed;
        self.next = if overflowed { None } else { Some(lower_bound) };
        Ok(())
    }

    pub fn mark_exhausted(&mut self) {
        self.occupied.clear();
        self.next = None;
    }

    pub fn allocate(&mut self) -> Result<u64, ReconstructionValidationError> {
        let Some(id) = self.next else {
            return Err(validation_error(
                "id",
                "next=overflow".to_owned(),
                "id",
                "id allocation overflowed",
            ));
        };
        let field = match self.domain {
            PersistentIdDomain::ColmapRecord => "id",
            PersistentIdDomain::Point3D => "point_id",
        };
        let record_type = match self.domain {
            PersistentIdDomain::ColmapRecord => "record",
            PersistentIdDomain::Point3D => "point",
        };
        ensure_id_legal(self.domain, id, record_type, field)?;
        if !self.occupied.insert(id) {
            return Err(validation_error(
                record_type,
                format!("{field}={id}"),
                field,
                "allocated id collides with an occupied id",
            ));
        }
        self.next = match id.checked_add(1) {
            Some(after) => {
                self.lower_bound = self.lower_bound.max(after);
                Some(self.lower_bound)
            }
            None => {
                self.overflowed = true;
                None
            }
        };
        Ok(id)
    }
}

/// Allocates `count` fresh COLMAP record IDs starting at 1.
pub fn fresh_colmap_record_ids(count: usize) -> Result<Vec<u32>, ReconstructionValidationError> {
    let mut allocator = OccupiedIdAllocator::new(PersistentIdDomain::ColmapRecord);
    let mut ids = Vec::with_capacity(count);
    for _ in 0..count {
        let id = allocator.allocate()?;
        let id = u32::try_from(id).map_err(|_| {
            validation_error(
                "record",
                format!("id={id}"),
                "id",
                "allocated COLMAP id does not fit in u32",
            )
        })?;
        ids.push(id);
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Rigid3, SensorId, SensorType, TrackObservation};
    use std::path::PathBuf;

    fn camera() -> CameraModel {
        CameraModel::new_pinhole(640, 480, 500.0, 500.0, 320.0, 240.0)
    }

    fn keypoint() -> KeyPoint {
        KeyPoint::new(1.0, 2.0)
    }

    fn point(xyz: [f32; 3], error: f32, image: usize) -> Point3D {
        Point3D {
            xyz,
            color: [0, 0, 0],
            error,
            track: vec![TrackObservation { image, feature: 0 }],
        }
    }

    fn two_images() -> Reconstruction {
        let camera = camera();
        Reconstruction {
            camera,
            cameras: vec![camera],
            camera_ids: vec![1],
            rigs: Vec::new(),
            frames: Vec::new(),
            image_names: vec!["a.png".to_owned(), "b.png".to_owned()],
            image_paths: vec![PathBuf::from("a.png"), PathBuf::from("b.png")],
            image_ids: vec![1, 2],
            image_camera_indices: vec![0, 0],
            image_frame_indices: vec![None, None],
            poses: vec![Some(SE3::identity()), Some(SE3::identity())],
            observations: vec![vec![Some(0)], vec![Some(1)]],
            keypoints: vec![vec![keypoint()], vec![keypoint()]],
            point_ids: vec![1, 2],
            points: vec![
                point([1.0, 2.0, 3.0], 0.1, 0),
                point([4.0, 5.0, 6.0], 0.2, 1),
            ],
        }
    }

    fn assert_error(
        result: Result<impl std::fmt::Debug, ReconstructionValidationError>,
        field: &str,
    ) {
        let error = result.expect_err(field);
        assert_eq!(error.field, field, "{error}");
        assert!(!error.record_type.is_empty(), "{error}");
        assert!(!error.ids.is_empty(), "{error}");
        assert!(!error.reason.is_empty(), "{error}");
    }

    #[test]
    fn valid_structure_and_export_pass() {
        let reconstruction = two_images();
        validate_structure(&reconstruction).expect("structure");
        validate_for_colmap_export(&reconstruction).expect("export");
    }

    #[test]
    fn structure_errors_name_the_failing_field() {
        let mut cases: Vec<(&str, Reconstruction)> = Vec::new();

        let mut lengths = two_images();
        lengths.image_names.pop();
        cases.push(("image_paths", lengths));

        let mut duplicate_images = two_images();
        duplicate_images.image_ids[1] = 1;
        cases.push(("image_id", duplicate_images));

        let mut duplicate_cameras = two_images();
        duplicate_cameras.cameras.push(camera());
        duplicate_cameras.camera_ids.push(1);
        cases.push(("camera_id", duplicate_cameras));

        let mut duplicate_points = two_images();
        duplicate_points.point_ids[1] = 1;
        cases.push(("point_id", duplicate_points));

        let mut zero_image = two_images();
        zero_image.image_ids[0] = 0;
        cases.push(("image_id", zero_image));

        let mut zero_camera = two_images();
        zero_camera.camera_ids[0] = 0;
        cases.push(("camera_id", zero_camera));

        let mut zero_point = two_images();
        zero_point.point_ids[0] = 0;
        cases.push(("point_id", zero_point));

        let mut huge_image = two_images();
        huge_image.image_ids[0] = COLMAP_ID_EXCLUSIVE_LIMIT as u32;
        cases.push(("image_id", huge_image));

        let mut camera_index = two_images();
        camera_index.image_camera_indices[0] = 4;
        cases.push(("image_camera_indices", camera_index));

        let mut observation_len = two_images();
        observation_len.observations[0].push(None);
        cases.push(("observations", observation_len));

        let mut missing_point = two_images();
        missing_point.observations[0][0] = Some(9);
        missing_point.points[0].track.clear();
        cases.push(("point_index", missing_point));

        let mut track_image = two_images();
        track_image.points[0].track[0].image = 9;
        cases.push(("track", track_image));

        let mut track_feature = two_images();
        track_feature.points[0].track[0].feature = 3;
        cases.push(("track", track_feature));

        let mut shared_feature = two_images();
        shared_feature.points[1].track[0].image = 0;
        cases.push(("track", shared_feature));

        let mut observation_only = two_images();
        observation_only.points[0].track.clear();
        cases.push(("observations", observation_only));

        let mut track_only = two_images();
        track_only.observations[0][0] = None;
        cases.push(("track", track_only));

        let mut disagreement = two_images();
        disagreement.observations[0][0] = Some(1);
        cases.push(("point_index", disagreement));

        let mut non_finite_camera = two_images();
        non_finite_camera.cameras[0].fx = f32::NAN;
        cases.push(("fx", non_finite_camera));

        let mut non_finite_params = two_images();
        non_finite_params.cameras[0].params[0] = f64::INFINITY;
        cases.push(("params", non_finite_params));

        let mut non_finite_pose = two_images();
        non_finite_pose.poses[0] = Some(SE3::new(&[0.0, 0.0, 0.0, 1.0], &[f32::NAN, 0.0, 0.0]));
        cases.push(("translation", non_finite_pose));

        let mut non_finite_point = two_images();
        non_finite_point.points[0].xyz[2] = f32::INFINITY;
        cases.push(("xyz", non_finite_point));

        let mut non_finite_error = two_images();
        non_finite_error.points[1].error = f32::NAN;
        cases.push(("error", non_finite_error));

        let mut non_finite_keypoint = two_images();
        non_finite_keypoint.keypoints[1][0].pt.1 = f32::NAN;
        cases.push(("pt", non_finite_keypoint));

        for (field, reconstruction) in cases {
            assert_error(validate_structure(&reconstruction), field);
        }
    }

    #[test]
    fn rig_frame_reference_errors_name_the_failing_field() {
        let mut cases: Vec<(&str, Reconstruction)> = Vec::new();
        let mut base = two_images();
        let sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 1,
        };
        base.rigs.push(Rig {
            rig_id: 1,
            ref_sensor_id: Some(sensor.clone()),
            sensors: Vec::new(),
        });
        base.frames.push(Frame {
            frame_id: 1,
            rig_id: 1,
            rig_from_world: Rigid3::identity(),
            data_ids: vec![
                DataId {
                    sensor_id: sensor.clone(),
                    data_id: 1,
                },
                DataId {
                    sensor_id: sensor,
                    data_id: 2,
                },
            ],
        });
        base.image_frame_indices = vec![Some(0), Some(0)];
        validate_structure(&base).expect("rigged reconstruction");

        let mut duplicate_rigs = base.clone();
        duplicate_rigs.rigs.push(Rig {
            rig_id: 1,
            ref_sensor_id: None,
            sensors: Vec::new(),
        });
        cases.push(("rig_id", duplicate_rigs));

        let mut zero_rig = base.clone();
        zero_rig.rigs[0].rig_id = 0;
        zero_rig.frames[0].rig_id = 0;
        cases.push(("rig_id", zero_rig));

        let mut duplicate_frames = base.clone();
        duplicate_frames
            .frames
            .push(duplicate_frames.frames[0].clone());
        cases.push(("frame_id", duplicate_frames));

        let mut zero_frame = base.clone();
        zero_frame.frames[0].frame_id = 0;
        cases.push(("frame_id", zero_frame));

        let mut missing_rig = base.clone();
        missing_rig.frames[0].rig_id = 9;
        cases.push(("rig_id", missing_rig));

        let mut missing_image = base.clone();
        missing_image.frames[0].data_ids[0].data_id = 44;
        cases.push(("data_id", missing_image));

        let mut missing_sensor = base.clone();
        missing_sensor.frames[0].data_ids[0].sensor_id.sensor_id = 8;
        cases.push(("sensor_id", missing_sensor));

        let mut bad_frame_index = base.clone();
        bad_frame_index.frames[0].data_ids.remove(0);
        bad_frame_index.image_frame_indices[0] = Some(3);
        cases.push(("image_frame_indices", bad_frame_index));

        let mut one_way = base.clone();
        one_way.image_frame_indices[1] = None;
        cases.push(("data_id", one_way));

        let mut non_finite_frame = base;
        non_finite_frame.frames[0].rig_from_world.tvec[0] = f64::NAN;
        cases.push(("rig_from_world", non_finite_frame));

        for (field, reconstruction) in cases {
            assert_error(validate_structure(&reconstruction), field);
        }
    }

    #[test]
    fn export_rejects_unregistered_track_after_structure_passes() {
        let mut reconstruction = two_images();
        reconstruction.poses[0] = None;
        validate_structure(&reconstruction).expect("unregistered image is structurally legal");
        assert_error(validate_for_colmap_export(&reconstruction), "track");
    }

    #[test]
    fn export_reports_structural_errors_before_export_rules() {
        let mut reconstruction = two_images();
        reconstruction.image_ids[1] = 1;
        reconstruction.poses[0] = None;
        assert_error(validate_for_colmap_export(&reconstruction), "image_id");
    }

    #[test]
    fn allocator_uses_occupied_ids_and_rejects_boundaries() {
        let mut images = OccupiedIdAllocator::new(PersistentIdDomain::ColmapRecord);
        images.replace_occupied([2, 9, 4]).expect("sparse ids");
        assert_eq!(images.allocate().expect("next"), 10);
        assert_eq!(images.allocate().expect("following"), 11);
        images
            .replace_occupied([])
            .expect("deleted ids stay reserved");
        assert_eq!(images.allocate().expect("not reused"), 12);

        let mut zero = OccupiedIdAllocator::new(PersistentIdDomain::Point3D);
        assert_error(zero.replace_occupied([0]), "point_id");

        let mut duplicate = OccupiedIdAllocator::new(PersistentIdDomain::Point3D);
        assert_error(duplicate.replace_occupied([4, 4]), "point_id");

        let mut saturated = OccupiedIdAllocator::new(PersistentIdDomain::ColmapRecord);
        saturated
            .replace_occupied([COLMAP_ID_EXCLUSIVE_LIMIT - 1])
            .expect("max legal image id");
        assert_error(saturated.allocate(), "id");

        let mut points = OccupiedIdAllocator::new(PersistentIdDomain::Point3D);
        points
            .replace_occupied([u64::MAX])
            .expect("max point id is representable");
        assert_error(points.allocate(), "id");

        assert_eq!(
            fresh_colmap_record_ids(0).expect("empty"),
            Vec::<u32>::new()
        );
        assert_eq!(fresh_colmap_record_ids(3).expect("fresh"), vec![1, 2, 3]);
    }

    #[test]
    fn strict_lookups_do_not_fabricate_ids_or_cameras() {
        let mut reconstruction = two_images();
        assert_eq!(reconstruction.try_image_id(0).expect("image"), 1);
        assert_eq!(
            reconstruction
                .try_camera_id_for_image(1)
                .expect("camera id"),
            1
        );
        assert_eq!(
            reconstruction.try_camera_for_image(0).expect("camera").fx,
            500.0
        );
        assert_eq!(reconstruction.try_point3d_id(1).expect("point"), 2);

        reconstruction.image_ids.clear();
        reconstruction.image_camera_indices.clear();
        reconstruction.camera_ids.clear();
        reconstruction.cameras.clear();
        reconstruction.point_ids.clear();
        assert_error(reconstruction.try_image_id(0), "image_id");
        assert_error(reconstruction.try_camera_id_for_image(0), "camera_id");
        assert_error(reconstruction.try_camera_for_image(0), "camera");
        assert_error(reconstruction.try_point3d_id(0), "point_id");

        #[allow(deprecated)]
        {
            assert_eq!(reconstruction.image_id(0), 1);
            assert_eq!(reconstruction.camera_id_for_image(0), 1);
            assert_eq!(reconstruction.point3d_id(0), 1);
            assert_eq!(
                reconstruction.camera_for_image(0).fx,
                reconstruction.camera.fx
            );
        }
    }
}
