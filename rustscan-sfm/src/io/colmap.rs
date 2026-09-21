use crate::geometry::{UnitQuatNormalize, Vec3GlamExt};
use crate::reconstruction_validation::{validate_for_colmap_export, COLMAP_ID_EXCLUSIVE_LIMIT};
use crate::types::{
    colmap_camera_model_focal_idxs, colmap_camera_model_id, colmap_camera_model_name,
    colmap_camera_model_num_params, CameraModel, DataId, Frame, Point3D, Reconstruction, Rig,
    RigSensor, Rigid3, SensorId, SensorType, TrackObservation,
};
use anyhow::{bail, Context, Result};
use nalgebra::{Matrix3, Quaternion, UnitQuaternion, Vector3};
use rustscan_slam::SE3;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Display;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, Write};
use std::path::{Path, PathBuf};

/// Quaternions at or below this Euclidean norm are rejected.
///
/// A finite norm above the epsilon may be normalized after validation. A zero
/// quaternion is never replaced with the identity.
const QUATERNION_NORM_EPSILON: f64 = 1e-8;

#[derive(Debug, thiserror::Error)]
#[error(
    "source={data_source} record={record_type} id={record_id} referenced={referenced_id} feature={feature_index} reason={reason}"
)]
struct ColmapIoError {
    data_source: String,
    record_type: &'static str,
    record_id: String,
    referenced_id: String,
    feature_index: String,
    reason: String,
}

fn colmap_io_error(
    source: &Path,
    record_type: &'static str,
    record_id: impl Display,
    referenced_id: impl Display,
    feature_index: impl Display,
    reason: impl Display,
) -> ColmapIoError {
    ColmapIoError {
        data_source: source.display().to_string(),
        record_type,
        record_id: record_id.to_string(),
        referenced_id: referenced_id.to_string(),
        feature_index: feature_index.to_string(),
        reason: reason.to_string(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColmapPose {
    pub image_id: u32,
    pub camera_id: u32,
    pub name: String,
    pub qvec: [f64; 4],
    pub tvec: [f64; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColmapPoint2D {
    pub xy: [f64; 2],
    pub point3d_id: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColmapImage {
    pub image_id: u32,
    pub camera_id: u32,
    pub name: String,
    pub qvec: [f64; 4],
    pub tvec: [f64; 3],
    pub points2d: Vec<ColmapPoint2D>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColmapCamera {
    pub camera_id: u32,
    pub model_id: i32,
    pub width: u32,
    pub height: u32,
    pub params: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColmapTrackElement {
    pub image_id: u32,
    pub point2d_idx: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColmapPoint3D {
    pub point3d_id: u64,
    pub xyz: [f64; 3],
    pub color: [u8; 3],
    pub error: f64,
    pub track: Vec<ColmapTrackElement>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColmapSensorType {
    Invalid,
    Camera,
    Imu,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColmapSensorId {
    pub sensor_type: ColmapSensorType,
    pub sensor_id: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColmapRigid3 {
    pub qvec: [f64; 4],
    pub tvec: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColmapRigSensor {
    pub sensor_id: ColmapSensorId,
    pub sensor_from_rig: Option<ColmapRigid3>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColmapRig {
    pub rig_id: u32,
    pub ref_sensor_id: Option<ColmapSensorId>,
    pub sensors: Vec<ColmapRigSensor>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColmapDataId {
    pub sensor_id: ColmapSensorId,
    pub data_id: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColmapFrame {
    pub frame_id: u32,
    pub rig_id: u32,
    pub rig_from_world: ColmapRigid3,
    pub data_ids: Vec<ColmapDataId>,
}

#[derive(Debug, Clone)]
pub struct ColmapSparseModel {
    pub reconstruction: Reconstruction,
    pub rigs: Vec<ColmapRig>,
    pub frames: Vec<ColmapFrame>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColmapSparseFiles {
    pub cameras: Vec<ColmapCamera>,
    pub rigs: Vec<ColmapRig>,
    pub frames: Vec<ColmapFrame>,
    pub images: Vec<ColmapImage>,
    pub points3d: Vec<ColmapPoint3D>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColmapSparseFormat {
    Text,
    Binary,
}

pub fn read_camera_model(root: &Path) -> Result<CameraModel> {
    let sparse = resolve_sparse_dir(root)?;
    let bin = sparse.join("cameras.bin");
    if bin.exists() {
        let camera = first_camera(read_cameras_bin(&bin)?, &bin)?;
        return camera_model_from_colmap(camera);
    }
    let txt = sparse.join("cameras.txt");
    if txt.exists() {
        let camera = first_camera(read_cameras_txt(&txt)?, &txt)?;
        return camera_model_from_colmap(camera);
    }
    bail!("missing cameras.bin/cameras.txt under {}", sparse.display())
}

pub fn read_colmap_cameras(root: &Path) -> Result<Vec<(u32, CameraModel)>> {
    let sparse = resolve_sparse_dir(root)?;
    let bin = sparse.join("cameras.bin");
    let cameras = if bin.exists() {
        read_cameras_bin(&bin)?
    } else {
        let txt = sparse.join("cameras.txt");
        if !txt.exists() {
            bail!("missing cameras.bin/cameras.txt under {}", sparse.display());
        }
        read_cameras_txt(&txt)?
    };
    cameras
        .into_iter()
        .map(|camera| {
            let camera_id = camera.camera_id;
            Ok((camera_id, camera_model_from_colmap(camera)?))
        })
        .collect()
}

fn first_camera(cameras: Vec<ColmapCamera>, path: &Path) -> Result<ColmapCamera> {
    cameras
        .into_iter()
        .next()
        .with_context(|| format!("no camera in {}", path.display()))
}

fn camera_model_from_colmap(camera: ColmapCamera) -> Result<CameraModel> {
    CameraModel::from_colmap(camera.model_id, camera.width, camera.height, &camera.params)
        .with_context(|| {
            format!(
                "invalid COLMAP camera model_id={} width={} height={} num_params={}",
                camera.model_id,
                camera.width,
                camera.height,
                camera.params.len()
            )
        })
}

pub fn read_colmap_poses(root: &Path) -> Result<Vec<ColmapPose>> {
    Ok(read_colmap_images(root)?
        .into_iter()
        .map(|image| ColmapPose {
            image_id: image.image_id,
            camera_id: image.camera_id,
            name: image.name,
            qvec: image.qvec,
            tvec: image.tvec,
        })
        .collect())
}

pub fn read_colmap_images(root: &Path) -> Result<Vec<ColmapImage>> {
    let sparse = resolve_sparse_dir(root)?;
    let bin = sparse.join("images.bin");
    if bin.exists() {
        return read_images_bin(&bin);
    }
    let txt = sparse.join("images.txt");
    if txt.exists() {
        return read_images_txt(&txt);
    }
    bail!("missing images.bin/images.txt under {}", sparse.display())
}

pub fn read_colmap_points3d(root: &Path) -> Result<Vec<ColmapPoint3D>> {
    let sparse = resolve_sparse_dir(root)?;
    let bin = sparse.join("points3D.bin");
    if bin.exists() {
        return read_points3d_bin(&bin);
    }
    let txt = sparse.join("points3D.txt");
    if txt.exists() {
        return read_points3d_txt(&txt);
    }
    bail!(
        "missing points3D.bin/points3D.txt under {}",
        sparse.display()
    )
}

pub fn read_colmap_sparse_files(root: &Path) -> Result<ColmapSparseFiles> {
    let sparse = resolve_sparse_dir(root)?;
    let has_required_bin = sparse.join("cameras.bin").exists()
        && sparse.join("images.bin").exists()
        && sparse.join("points3D.bin").exists();
    let has_required_txt = sparse.join("cameras.txt").exists()
        && sparse.join("images.txt").exists()
        && sparse.join("points3D.txt").exists();
    if has_required_bin {
        read_colmap_sparse_files_with_format(&sparse, ColmapSparseFormat::Binary)
    } else if has_required_txt {
        read_colmap_sparse_files_with_format(&sparse, ColmapSparseFormat::Text)
    } else {
        bail!(
            "missing COLMAP cameras/images/points3D sparse model files under {}",
            sparse.display()
        )
    }
}

pub fn read_colmap_sparse_files_with_format(
    root: &Path,
    format: ColmapSparseFormat,
) -> Result<ColmapSparseFiles> {
    let sparse = resolve_sparse_dir(root)?;
    let (cameras, rigs, frames, images, points3d) = match format {
        ColmapSparseFormat::Text => (
            read_cameras_txt(&sparse.join("cameras.txt"))?,
            read_optional_rigs_txt(&sparse)?,
            read_optional_frames_txt(&sparse)?,
            read_images_txt(&sparse.join("images.txt"))?,
            read_points3d_txt(&sparse.join("points3D.txt"))?,
        ),
        ColmapSparseFormat::Binary => (
            read_cameras_bin(&sparse.join("cameras.bin"))?,
            read_optional_rigs_bin(&sparse)?,
            read_optional_frames_bin(&sparse)?,
            read_images_bin(&sparse.join("images.bin"))?,
            read_points3d_bin(&sparse.join("points3D.bin"))?,
        ),
    };
    Ok(ColmapSparseFiles {
        cameras,
        rigs,
        frames,
        images,
        points3d,
    })
}

struct LocatedSparseFiles {
    files: ColmapSparseFiles,
    directory: PathBuf,
    format: ColmapSparseFormat,
}

fn read_located_sparse_files(root: &Path) -> Result<LocatedSparseFiles> {
    let directory = resolve_sparse_dir(root)?;
    let has_required_bin = directory.join("cameras.bin").exists()
        && directory.join("images.bin").exists()
        && directory.join("points3D.bin").exists();
    let has_required_txt = directory.join("cameras.txt").exists()
        && directory.join("images.txt").exists()
        && directory.join("points3D.txt").exists();
    let format = if has_required_bin {
        ColmapSparseFormat::Binary
    } else if has_required_txt {
        ColmapSparseFormat::Text
    } else {
        bail!(
            "missing COLMAP cameras/images/points3D sparse model files under {}",
            directory.display()
        )
    };
    let files = read_colmap_sparse_files_with_format(&directory, format)?;
    Ok(LocatedSparseFiles {
        files,
        directory,
        format,
    })
}

pub fn read_colmap_reconstruction(root: &Path) -> Result<Reconstruction> {
    let located = read_located_sparse_files(root)?;
    reconstruction_from_colmap_files(&located.files, &located.directory, located.format)
}

pub fn read_colmap_sparse_model(root: &Path) -> Result<ColmapSparseModel> {
    let located = read_located_sparse_files(root)?;
    let mut reconstruction =
        reconstruction_from_colmap_files(&located.files, &located.directory, located.format)?;
    apply_rig_frame_metadata_to_reconstruction(
        &mut reconstruction,
        &located.files.rigs,
        &located.files.frames,
        &sparse_record_path(&located.directory, "frames", located.format),
    )?;
    Ok(ColmapSparseModel {
        reconstruction,
        rigs: located.files.rigs,
        frames: located.files.frames,
    })
}

fn read_optional_rigs_txt(sparse: &Path) -> Result<Vec<ColmapRig>> {
    let path = sparse.join("rigs.txt");
    if path.exists() {
        read_rigs_txt(&path)
    } else {
        Ok(Vec::new())
    }
}

fn read_optional_frames_txt(sparse: &Path) -> Result<Vec<ColmapFrame>> {
    let path = sparse.join("frames.txt");
    if path.exists() {
        read_frames_txt(&path)
    } else {
        Ok(Vec::new())
    }
}

fn read_optional_rigs_bin(sparse: &Path) -> Result<Vec<ColmapRig>> {
    let path = sparse.join("rigs.bin");
    if path.exists() {
        read_rigs_bin(&path)
    } else {
        Ok(Vec::new())
    }
}

fn read_optional_frames_bin(sparse: &Path) -> Result<Vec<ColmapFrame>> {
    let path = sparse.join("frames.bin");
    if path.exists() {
        read_frames_bin(&path)
    } else {
        Ok(Vec::new())
    }
}

pub fn read_colmap_rigs(root: &Path) -> Result<Vec<ColmapRig>> {
    let sparse = resolve_sparse_dir(root)?;
    let bin = sparse.join("rigs.bin");
    if bin.exists() {
        return read_rigs_bin(&bin);
    }
    let txt = sparse.join("rigs.txt");
    if txt.exists() {
        return read_rigs_txt(&txt);
    }
    bail!("missing rigs.txt under {}", sparse.display())
}

pub fn read_colmap_frames(root: &Path) -> Result<Vec<ColmapFrame>> {
    let sparse = resolve_sparse_dir(root)?;
    let bin = sparse.join("frames.bin");
    if bin.exists() {
        return read_frames_bin(&bin);
    }
    let txt = sparse.join("frames.txt");
    if txt.exists() {
        return read_frames_txt(&txt);
    }
    bail!("missing frames.txt under {}", sparse.display())
}

pub fn resolve_sparse_dir(root: &Path) -> Result<PathBuf> {
    if has_model_files(root) {
        return Ok(root.to_path_buf());
    }
    let sparse = root.join("sparse");
    if has_model_files(&sparse) {
        return Ok(sparse);
    }
    let sparse0 = sparse.join("0");
    if has_model_files(&sparse0) {
        return Ok(sparse0);
    }
    bail!(
        "could not resolve COLMAP sparse dir under {}",
        root.display()
    )
}

fn has_model_files(path: &Path) -> bool {
    path.join("images.bin").exists()
        || path.join("images.txt").exists()
        || path.join("cameras.bin").exists()
        || path.join("cameras.txt").exists()
        || path.join("points3D.bin").exists()
        || path.join("points3D.txt").exists()
        || path.join("rigs.bin").exists()
        || path.join("rigs.txt").exists()
        || path.join("frames.bin").exists()
        || path.join("frames.txt").exists()
}

pub fn camera_center(pose: &ColmapPose) -> Vector3<f64> {
    let r = world_to_camera_rotation(pose);
    let t = Vector3::new(pose.tvec[0], pose.tvec[1], pose.tvec[2]);
    -(r.transpose() * t)
}

pub fn world_to_camera_rotation(pose: &ColmapPose) -> Matrix3<f64> {
    UnitQuaternion::from_quaternion(Quaternion::new(
        pose.qvec[0],
        pose.qvec[1],
        pose.qvec[2],
        pose.qvec[3],
    ))
    .to_rotation_matrix()
    .into_inner()
}

fn sparse_record_path(directory: &Path, stem: &str, format: ColmapSparseFormat) -> PathBuf {
    let extension = match format {
        ColmapSparseFormat::Text => "txt",
        ColmapSparseFormat::Binary => "bin",
    };
    directory.join(format!("{stem}.{extension}"))
}

fn validate_colmap_sparse_files(
    sparse: &ColmapSparseFiles,
    directory: &Path,
    format: ColmapSparseFormat,
) -> Result<()> {
    let camera_ids = validate_cameras(
        &sparse.cameras,
        &sparse_record_path(directory, "cameras", format),
    )?;
    let images = validate_images(
        &sparse.images,
        &camera_ids,
        &sparse_record_path(directory, "images", format),
    )?;
    validate_points_and_tracks(
        &sparse.points3d,
        &images,
        &sparse_record_path(directory, "points3D", format),
        &sparse_record_path(directory, "images", format),
    )?;
    validate_rigs_and_frames(
        &sparse.rigs,
        &sparse.frames,
        &camera_ids,
        &images.keys().copied().collect(),
        &sparse_record_path(directory, "rigs", format),
        &sparse_record_path(directory, "frames", format),
        Some(&images),
    )?;
    Ok(())
}

fn validate_cameras(cameras: &[ColmapCamera], source: &Path) -> Result<HashSet<u32>> {
    let mut camera_ids = HashSet::new();
    for camera in cameras {
        if !camera_ids.insert(camera.camera_id) {
            return Err(colmap_io_error(
                source,
                "camera",
                camera.camera_id,
                "-",
                "-",
                "duplicate camera id",
            )
            .into());
        }
        ensure_colmap_record_id(
            u64::from(camera.camera_id),
            source,
            "camera",
            camera.camera_id,
        )?;
        if camera.width == 0 || camera.height == 0 {
            return Err(colmap_io_error(
                source,
                "camera",
                camera.camera_id,
                "-",
                "-",
                format!(
                    "image dimensions are illegal (width={}, height={})",
                    camera.width, camera.height
                ),
            )
            .into());
        }
        let Some(expected) = colmap_camera_model_num_params(camera.model_id) else {
            return Err(colmap_io_error(
                source,
                "camera",
                camera.camera_id,
                camera.model_id,
                "-",
                "unknown camera model",
            )
            .into());
        };
        if camera.params.len() != expected {
            return Err(colmap_io_error(
                source,
                "camera",
                camera.camera_id,
                camera.model_id,
                "-",
                format!(
                    "camera parameter count is {} but model {expected} parameters are required",
                    camera.params.len()
                ),
            )
            .into());
        }
        if camera.params.iter().any(|value| !value.is_finite()) {
            return Err(colmap_io_error(
                source,
                "camera",
                camera.camera_id,
                "-",
                "-",
                "camera parameter is non-finite",
            )
            .into());
        }
        let Some(focal_idxs) = colmap_camera_model_focal_idxs(camera.model_id) else {
            return Err(colmap_io_error(
                source,
                "camera",
                camera.camera_id,
                camera.model_id,
                "-",
                "camera model has no focal parameters",
            )
            .into());
        };
        for focal_index in focal_idxs {
            let focal = camera.params[*focal_index];
            if !focal.is_finite() || focal <= 0.0 {
                return Err(colmap_io_error(
                    source,
                    "camera",
                    camera.camera_id,
                    "-",
                    "-",
                    format!("focal parameter at index {focal_index} is illegal ({focal})"),
                )
                .into());
            }
        }
    }
    Ok(camera_ids)
}

fn validate_images<'a>(
    images: &'a [ColmapImage],
    camera_ids: &HashSet<u32>,
    source: &Path,
) -> Result<HashMap<u32, &'a ColmapImage>> {
    let mut by_id = HashMap::new();
    for image in images {
        if by_id.contains_key(&image.image_id) {
            return Err(colmap_io_error(
                source,
                "image",
                image.image_id,
                "-",
                "-",
                "duplicate image id",
            )
            .into());
        }
        ensure_colmap_record_id(u64::from(image.image_id), source, "image", image.image_id)?;
        if !camera_ids.contains(&image.camera_id) {
            return Err(colmap_io_error(
                source,
                "image",
                image.image_id,
                image.camera_id,
                "-",
                "image references a missing camera",
            )
            .into());
        }
        validate_quaternion(image.qvec, source, "image", image.image_id, "-")?;
        if image.tvec.iter().any(|value| !value.is_finite()) {
            return Err(colmap_io_error(
                source,
                "image",
                image.image_id,
                "-",
                "-",
                "translation is non-finite",
            )
            .into());
        }
        for (feature_index, point) in image.points2d.iter().enumerate() {
            if point.xy.iter().any(|value| !value.is_finite()) {
                return Err(colmap_io_error(
                    source,
                    "image",
                    image.image_id,
                    "-",
                    feature_index,
                    "observation coordinate is non-finite",
                )
                .into());
            }
        }
        by_id.insert(image.image_id, image);
    }
    Ok(by_id)
}

fn validate_points_and_tracks(
    points: &[ColmapPoint3D],
    images: &HashMap<u32, &ColmapImage>,
    points_source: &Path,
    images_source: &Path,
) -> Result<()> {
    let mut point_ids = HashSet::new();
    let mut points_by_id = HashMap::new();
    for point in points {
        if !point_ids.insert(point.point3d_id) {
            return Err(colmap_io_error(
                points_source,
                "point",
                point.point3d_id,
                "-",
                "-",
                "duplicate point id",
            )
            .into());
        }
        if point.point3d_id == 0 {
            return Err(colmap_io_error(
                points_source,
                "point",
                point.point3d_id,
                "-",
                "-",
                "point id 0 is illegal",
            )
            .into());
        }
        if point.xyz.iter().any(|value| !value.is_finite()) {
            return Err(colmap_io_error(
                points_source,
                "point",
                point.point3d_id,
                "-",
                "-",
                "point coordinate is non-finite",
            )
            .into());
        }
        if !point.error.is_finite() {
            return Err(colmap_io_error(
                points_source,
                "point",
                point.point3d_id,
                "-",
                "-",
                "reprojection error is non-finite",
            )
            .into());
        }
        points_by_id.insert(point.point3d_id, point);
    }

    let mut listed_observations = HashSet::new();
    for point in points {
        for element in &point.track {
            let feature_index = match usize::try_from(element.point2d_idx) {
                Ok(index) => index,
                Err(_) => {
                    return Err(colmap_io_error(
                        points_source,
                        "point",
                        point.point3d_id,
                        element.image_id,
                        element.point2d_idx,
                        "feature index overflows usize",
                    )
                    .into());
                }
            };
            if !listed_observations.insert((element.image_id, element.point2d_idx)) {
                return Err(colmap_io_error(
                    points_source,
                    "point",
                    point.point3d_id,
                    element.image_id,
                    element.point2d_idx,
                    "duplicate track observation",
                )
                .into());
            }
            let Some(image) = images.get(&element.image_id) else {
                return Err(colmap_io_error(
                    points_source,
                    "point",
                    point.point3d_id,
                    element.image_id,
                    element.point2d_idx,
                    "track references a missing image",
                )
                .into());
            };
            if feature_index >= image.points2d.len() {
                return Err(colmap_io_error(
                    points_source,
                    "point",
                    point.point3d_id,
                    element.image_id,
                    element.point2d_idx,
                    "feature index is out of range",
                )
                .into());
            }
            match image.points2d[feature_index].point3d_id {
                Some(point3d_id) if point3d_id == point.point3d_id => {}
                other => {
                    return Err(colmap_io_error(
                        points_source,
                        "point",
                        point.point3d_id,
                        other
                            .map(|id| id.to_string())
                            .unwrap_or_else(|| "-".to_string()),
                        element.point2d_idx,
                        "point track does not match the image observation",
                    )
                    .into());
                }
            }
        }
    }

    for image in images.values() {
        for (feature_index, point2d) in image.points2d.iter().enumerate() {
            let Some(point3d_id) = point2d.point3d_id else {
                continue;
            };
            let Some(point) = points_by_id.get(&point3d_id) else {
                return Err(colmap_io_error(
                    images_source,
                    "image",
                    image.image_id,
                    point3d_id,
                    feature_index,
                    "observation references a missing point",
                )
                .into());
            };
            let listed = point.track.iter().any(|element| {
                element.image_id == image.image_id && element.point2d_idx == feature_index as u64
            });
            if !listed {
                return Err(colmap_io_error(
                    images_source,
                    "image",
                    image.image_id,
                    point3d_id,
                    feature_index,
                    "image observation does not match the point track",
                )
                .into());
            }
        }
    }
    Ok(())
}

fn validate_rigs_and_frames(
    rigs: &[ColmapRig],
    frames: &[ColmapFrame],
    camera_ids: &HashSet<u32>,
    image_ids: &HashSet<u32>,
    rigs_source: &Path,
    frames_source: &Path,
    images: Option<&HashMap<u32, &ColmapImage>>,
) -> Result<()> {
    let mut rig_ids = HashSet::new();
    let mut sensors_by_rig: HashMap<u32, HashSet<(String, u32)>> = HashMap::new();
    for rig in rigs {
        if !rig_ids.insert(rig.rig_id) {
            return Err(colmap_io_error(
                rigs_source,
                "rig",
                rig.rig_id,
                "-",
                "-",
                "duplicate rig id",
            )
            .into());
        }
        ensure_colmap_record_id(u64::from(rig.rig_id), rigs_source, "rig", rig.rig_id)?;
        let mut sensors = HashSet::new();
        if let Some(reference) = &rig.ref_sensor_id {
            validate_sensor_reference(
                reference,
                camera_ids,
                rigs_source,
                "rig",
                rig.rig_id,
                "reference sensor",
            )?;
            if !sensors.insert(sensor_key(reference)) {
                return Err(colmap_io_error(
                    rigs_source,
                    "rig",
                    rig.rig_id,
                    reference.sensor_id,
                    "-",
                    "duplicate rig sensor id",
                )
                .into());
            }
        } else if !rig.sensors.is_empty() {
            return Err(colmap_io_error(
                rigs_source,
                "rig",
                rig.rig_id,
                "-",
                "-",
                "rig has sensors but no reference sensor",
            )
            .into());
        }
        for sensor in &rig.sensors {
            validate_sensor_reference(
                &sensor.sensor_id,
                camera_ids,
                rigs_source,
                "rig",
                rig.rig_id,
                "rig sensor",
            )?;
            if !sensors.insert(sensor_key(&sensor.sensor_id)) {
                return Err(colmap_io_error(
                    rigs_source,
                    "rig",
                    rig.rig_id,
                    sensor.sensor_id.sensor_id,
                    "-",
                    "duplicate rig sensor id",
                )
                .into());
            }
            if let Some(sensor_from_rig) = &sensor.sensor_from_rig {
                validate_quaternion(
                    sensor_from_rig.qvec,
                    rigs_source,
                    "rig",
                    rig.rig_id,
                    sensor.sensor_id.sensor_id,
                )?;
                if sensor_from_rig.tvec.iter().any(|value| !value.is_finite()) {
                    return Err(colmap_io_error(
                        rigs_source,
                        "rig",
                        rig.rig_id,
                        sensor.sensor_id.sensor_id,
                        "-",
                        "sensor translation is non-finite",
                    )
                    .into());
                }
            }
        }
        sensors_by_rig.insert(rig.rig_id, sensors);
    }

    let mut frame_ids = HashSet::new();
    let mut image_frame = HashMap::new();
    for frame in frames {
        if !frame_ids.insert(frame.frame_id) {
            return Err(colmap_io_error(
                frames_source,
                "frame",
                frame.frame_id,
                "-",
                "-",
                "duplicate frame id",
            )
            .into());
        }
        ensure_colmap_record_id(
            u64::from(frame.frame_id),
            frames_source,
            "frame",
            frame.frame_id,
        )?;
        let Some(sensors) = sensors_by_rig.get(&frame.rig_id) else {
            return Err(colmap_io_error(
                frames_source,
                "frame",
                frame.frame_id,
                frame.rig_id,
                "-",
                "frame references a missing rig",
            )
            .into());
        };
        validate_quaternion(
            frame.rig_from_world.qvec,
            frames_source,
            "frame",
            frame.frame_id,
            "-",
        )?;
        if frame
            .rig_from_world
            .tvec
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(colmap_io_error(
                frames_source,
                "frame",
                frame.frame_id,
                "-",
                "-",
                "frame translation is non-finite",
            )
            .into());
        }
        for data_id in &frame.data_ids {
            if !sensors.contains(&sensor_key(&data_id.sensor_id)) {
                return Err(colmap_io_error(
                    frames_source,
                    "frame",
                    frame.frame_id,
                    data_id.sensor_id.sensor_id,
                    "-",
                    "frame sensor is not on the referenced rig",
                )
                .into());
            }
            if data_id.data_id == 0 {
                return Err(colmap_io_error(
                    frames_source,
                    "frame",
                    frame.frame_id,
                    data_id.sensor_id.sensor_id,
                    "-",
                    "data id 0 is illegal",
                )
                .into());
            }
            if data_id.sensor_id.sensor_type != ColmapSensorType::Camera {
                continue;
            }
            let Ok(image_id) = u32::try_from(data_id.data_id) else {
                return Err(colmap_io_error(
                    frames_source,
                    "frame",
                    frame.frame_id,
                    data_id.data_id,
                    "-",
                    "camera data id does not fit an image id",
                )
                .into());
            };
            if !image_ids.contains(&image_id) {
                return Err(colmap_io_error(
                    frames_source,
                    "frame",
                    frame.frame_id,
                    image_id,
                    "-",
                    "camera data id references a missing image",
                )
                .into());
            }
            if let Some(images) = images {
                if let Some(image) = images.get(&image_id) {
                    if image.camera_id != data_id.sensor_id.sensor_id {
                        return Err(colmap_io_error(
                            frames_source,
                            "frame",
                            frame.frame_id,
                            image_id,
                            "-",
                            format!(
                                "camera data sensor {} does not match image camera {}",
                                data_id.sensor_id.sensor_id, image.camera_id
                            ),
                        )
                        .into());
                    }
                }
            }
            if image_frame.contains_key(&image_id) {
                return Err(colmap_io_error(
                    frames_source,
                    "frame",
                    frame.frame_id,
                    image_id,
                    "-",
                    "image is assigned to more than one frame",
                )
                .into());
            }
            image_frame.insert(image_id, frame.frame_id);
        }
    }
    Ok(())
}

fn validate_sensor_reference(
    sensor: &ColmapSensorId,
    camera_ids: &HashSet<u32>,
    source: &Path,
    record_type: &'static str,
    record_id: u32,
    role: &str,
) -> Result<()> {
    ensure_colmap_record_id(u64::from(sensor.sensor_id), source, record_type, record_id)?;
    if sensor.sensor_type == ColmapSensorType::Invalid {
        return Err(colmap_io_error(
            source,
            record_type,
            record_id,
            sensor.sensor_id,
            "-",
            format!("{role} type is invalid"),
        )
        .into());
    }
    if sensor.sensor_type == ColmapSensorType::Camera && !camera_ids.contains(&sensor.sensor_id) {
        return Err(colmap_io_error(
            source,
            record_type,
            record_id,
            sensor.sensor_id,
            "-",
            format!("{role} references a missing camera"),
        )
        .into());
    }
    Ok(())
}

fn validate_quaternion(
    qvec: [f64; 4],
    source: &Path,
    record_type: &'static str,
    record_id: impl Display,
    referenced_id: impl Display,
) -> Result<()> {
    if qvec.iter().any(|value| !value.is_finite()) {
        return Err(colmap_io_error(
            source,
            record_type,
            record_id,
            referenced_id,
            "-",
            "quaternion is non-finite",
        )
        .into());
    }
    let norm = qvec.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm <= QUATERNION_NORM_EPSILON {
        return Err(colmap_io_error(
            source,
            record_type,
            record_id,
            referenced_id,
            "-",
            "quaternion norm is zero",
        )
        .into());
    }
    Ok(())
}

fn ensure_colmap_record_id(
    id: u64,
    source: &Path,
    record_type: &'static str,
    record_id: impl Display,
) -> Result<()> {
    if id == 0 || id >= COLMAP_ID_EXCLUSIVE_LIMIT {
        return Err(colmap_io_error(
            source,
            record_type,
            record_id,
            id,
            "-",
            format!("id is outside 1..{COLMAP_ID_EXCLUSIVE_LIMIT}"),
        )
        .into());
    }
    Ok(())
}

fn sensor_key(sensor: &ColmapSensorId) -> (String, u32) {
    (
        sensor_type_name(&sensor.sensor_type).to_string(),
        sensor.sensor_id,
    )
}

fn insert_new<K, V>(map: &mut HashMap<K, V>, key: K, value: V) -> bool
where
    K: Eq + std::hash::Hash,
{
    if map.contains_key(&key) {
        return false;
    }
    map.insert(key, value);
    true
}

fn reconstruction_from_colmap_parts(
    colmap_cameras: Vec<(u32, CameraModel)>,
    images: Vec<ColmapImage>,
    points3d: Vec<ColmapPoint3D>,
    source: &Path,
) -> Result<Reconstruction> {
    if colmap_cameras.is_empty() {
        bail!("COLMAP reconstruction has no cameras");
    }
    let (camera_ids, cameras): (Vec<_>, Vec<_>) = colmap_cameras.into_iter().unzip();
    let mut camera_index_by_id = HashMap::new();
    for (idx, &camera_id) in camera_ids.iter().enumerate() {
        if !insert_new(&mut camera_index_by_id, camera_id, idx) {
            return Err(colmap_io_error(
                source,
                "camera",
                camera_id,
                "-",
                "-",
                "duplicate camera id",
            )
            .into());
        }
    }
    let mut image_index_by_id = HashMap::new();
    for (idx, image) in images.iter().enumerate() {
        if !insert_new(&mut image_index_by_id, image.image_id, idx) {
            return Err(colmap_io_error(
                source,
                "image",
                image.image_id,
                "-",
                "-",
                "duplicate image id",
            )
            .into());
        }
    }

    let mut image_camera_indices = Vec::with_capacity(images.len());
    for image in &images {
        let Some(camera_index) = camera_index_by_id.get(&image.camera_id).copied() else {
            return Err(colmap_io_error(
                source,
                "image",
                image.image_id,
                image.camera_id,
                "-",
                "image references a missing camera",
            )
            .into());
        };
        image_camera_indices.push(camera_index);
    }
    let (rigs, frames, image_frame_indices) = Reconstruction::empty_metadata(images.len());
    let keypoints = images
        .iter()
        .map(|image| {
            image
                .points2d
                .iter()
                .map(keypoint_from_colmap_point2d)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut point_index_by_id = BTreeMap::new();
    for (idx, point) in points3d.iter().enumerate() {
        if point_index_by_id.contains_key(&point.point3d_id) {
            return Err(colmap_io_error(
                source,
                "point",
                point.point3d_id,
                "-",
                "-",
                "duplicate point id",
            )
            .into());
        }
        point_index_by_id.insert(point.point3d_id, idx);
    }

    let mut observations = keypoints
        .iter()
        .map(|points| vec![None; points.len()])
        .collect::<Vec<_>>();
    for (image_idx, image) in images.iter().enumerate() {
        for (point2d_idx, point2d) in image.points2d.iter().enumerate() {
            let Some(point3d_id) = point2d.point3d_id else {
                continue;
            };
            let Some(&point_idx) = point_index_by_id.get(&point3d_id) else {
                return Err(colmap_io_error(
                    source,
                    "image",
                    image.image_id,
                    point3d_id,
                    point2d_idx,
                    "observation references a missing point",
                )
                .into());
            };
            observations[image_idx][point2d_idx] = Some(point_idx);
        }
    }

    let mut points = Vec::with_capacity(points3d.len());
    for point in &points3d {
        let mut track = Vec::with_capacity(point.track.len());
        for element in &point.track {
            let Some(&image) = image_index_by_id.get(&element.image_id) else {
                return Err(colmap_io_error(
                    source,
                    "point",
                    point.point3d_id,
                    element.image_id,
                    element.point2d_idx,
                    "track references a missing image",
                )
                .into());
            };
            let feature = match usize::try_from(element.point2d_idx) {
                Ok(feature) => feature,
                Err(_) => {
                    return Err(colmap_io_error(
                        source,
                        "point",
                        point.point3d_id,
                        element.image_id,
                        element.point2d_idx,
                        "feature index overflows usize",
                    )
                    .into());
                }
            };
            if keypoints
                .get(image)
                .is_none_or(|points| feature >= points.len())
            {
                return Err(colmap_io_error(
                    source,
                    "point",
                    point.point3d_id,
                    element.image_id,
                    element.point2d_idx,
                    "feature index is out of range",
                )
                .into());
            }
            track.push(TrackObservation { image, feature });
        }
        points.push(Point3D {
            xyz: [
                point.xyz[0] as f32,
                point.xyz[1] as f32,
                point.xyz[2] as f32,
            ],
            color: point.color,
            error: point.error as f32,
            track,
        });
    }

    let mut poses = Vec::with_capacity(images.len());
    for image in &images {
        poses.push(Some(se3_from_colmap_pose(image.qvec, image.tvec)?));
    }
    let image_names = images
        .iter()
        .map(|image| image.name.clone())
        .collect::<Vec<_>>();
    let image_paths = image_names.iter().map(PathBuf::from).collect::<Vec<_>>();
    let image_ids = images
        .iter()
        .map(|image| image.image_id)
        .collect::<Vec<_>>();
    let point_ids = points3d
        .iter()
        .map(|point| point.point3d_id)
        .collect::<Vec<_>>();

    Ok(Reconstruction {
        camera: cameras[0],
        cameras,
        camera_ids,
        rigs,
        frames,
        image_names,
        image_paths,
        image_ids,
        image_camera_indices,
        image_frame_indices,
        poses,
        observations,
        keypoints,
        point_ids,
        points,
    })
}

fn reconstruction_from_colmap_files(
    sparse: &ColmapSparseFiles,
    directory: &Path,
    format: ColmapSparseFormat,
) -> Result<Reconstruction> {
    validate_colmap_sparse_files(sparse, directory, format)?;
    let cameras = sparse
        .cameras
        .iter()
        .cloned()
        .map(|camera| {
            let camera_id = camera.camera_id;
            Ok((camera_id, camera_model_from_colmap(camera)?))
        })
        .collect::<Result<Vec<_>>>()?;
    reconstruction_from_colmap_parts(
        cameras,
        sparse.images.clone(),
        sparse.points3d.clone(),
        &sparse_record_path(directory, "cameras", format),
    )
}

fn apply_rig_frame_metadata_to_reconstruction(
    reconstruction: &mut Reconstruction,
    rigs: &[ColmapRig],
    frames: &[ColmapFrame],
    source: &Path,
) -> Result<()> {
    reconstruction.rigs = rigs.iter().map(rig_from_colmap).collect();
    reconstruction.frames = frames.iter().map(frame_from_colmap).collect();
    let mut frame_index_by_camera_data_id = HashMap::new();
    for (frame_idx, frame) in frames.iter().enumerate() {
        for data_id in &frame.data_ids {
            if data_id.sensor_id.sensor_type != ColmapSensorType::Camera {
                continue;
            }
            let Ok(image_id) = u32::try_from(data_id.data_id) else {
                return Err(colmap_io_error(
                    source,
                    "frame",
                    frame.frame_id,
                    data_id.data_id,
                    "-",
                    "camera data id does not fit an image id",
                )
                .into());
            };
            if !insert_new(&mut frame_index_by_camera_data_id, image_id, frame_idx) {
                return Err(colmap_io_error(
                    source,
                    "frame",
                    frame.frame_id,
                    image_id,
                    "-",
                    "image is assigned to more than one frame",
                )
                .into());
            }
        }
    }
    reconstruction.image_frame_indices = reconstruction
        .image_ids
        .iter()
        .map(|image_id| frame_index_by_camera_data_id.get(image_id).copied())
        .collect();
    Ok(())
}

fn rig_from_colmap(rig: &ColmapRig) -> Rig {
    let ref_sensor_id = rig.ref_sensor_id.as_ref().map(sensor_id_from_colmap);
    let mut sensors: Vec<_> = rig.sensors.iter().map(rig_sensor_from_colmap).collect();
    // COLMAP stores the reference sensor beside the non-reference sensor list.
    // Reconstruction validation also requires that sensor inside `sensors`.
    // Copy the existing reference; do not invent a missing rig.
    if let Some(reference) = &ref_sensor_id {
        if !sensors.iter().any(|sensor| &sensor.sensor_id == reference) {
            sensors.insert(
                0,
                RigSensor {
                    sensor_id: reference.clone(),
                    sensor_from_rig: None,
                },
            );
        }
    }
    Rig {
        rig_id: rig.rig_id,
        ref_sensor_id,
        sensors,
    }
}
fn rig_sensor_from_colmap(sensor: &ColmapRigSensor) -> RigSensor {
    RigSensor {
        sensor_id: sensor_id_from_colmap(&sensor.sensor_id),
        sensor_from_rig: sensor.sensor_from_rig.as_ref().map(rigid3_from_colmap),
    }
}

fn frame_from_colmap(frame: &ColmapFrame) -> Frame {
    Frame {
        frame_id: frame.frame_id,
        rig_id: frame.rig_id,
        rig_from_world: rigid3_from_colmap(&frame.rig_from_world),
        data_ids: frame.data_ids.iter().map(data_id_from_colmap).collect(),
    }
}

fn sensor_id_from_colmap(sensor_id: &ColmapSensorId) -> SensorId {
    SensorId {
        sensor_type: sensor_type_from_colmap(&sensor_id.sensor_type),
        sensor_id: sensor_id.sensor_id,
    }
}

fn sensor_type_from_colmap(sensor_type: &ColmapSensorType) -> SensorType {
    match sensor_type {
        ColmapSensorType::Invalid => SensorType::Invalid,
        ColmapSensorType::Camera => SensorType::Camera,
        ColmapSensorType::Imu => SensorType::Imu,
        ColmapSensorType::Other(value) => SensorType::Other(value.clone()),
    }
}

fn rigid3_from_colmap(rigid: &ColmapRigid3) -> Rigid3 {
    Rigid3 {
        qvec: rigid.qvec,
        tvec: rigid.tvec,
    }
}

fn data_id_from_colmap(data_id: &ColmapDataId) -> DataId {
    DataId {
        sensor_id: sensor_id_from_colmap(&data_id.sensor_id),
        data_id: data_id.data_id,
    }
}

fn keypoint_from_colmap_point2d(point: &ColmapPoint2D) -> rustscan_slam::KeyPoint {
    rustscan_slam::KeyPoint {
        pt: (point.xy[0] as f32, point.xy[1] as f32),
        size: 1.0,
        angle: 0.0,
        response: 1.0,
        octave: 0,
    }
}

fn se3_from_colmap_pose(qvec: [f64; 4], tvec: [f64; 3]) -> Result<SE3> {
    validate_quaternion(qvec, Path::new("colmap-pose"), "image", "-", "-")?;
    if tvec.iter().any(|value| !value.is_finite()) {
        return Err(colmap_io_error(
            Path::new("colmap-pose"),
            "image",
            "-",
            "-",
            "-",
            "translation is non-finite",
        )
        .into());
    }
    let rotation = crate::geometry::quat_from_xyzw(
        qvec[1] as f32,
        qvec[2] as f32,
        qvec[3] as f32,
        qvec[0] as f32,
    )
    .normalize();
    Ok(SE3::from_quat_translation(
        rotation,
        nalgebra::Vector3::new(tvec[0] as f32, tvec[1] as f32, tvec[2] as f32),
    ))
}

pub fn export_colmap(
    root: &Path,
    reconstruction: &Reconstruction,
    copy_images: bool,
) -> Result<()> {
    export_colmap_with_sparse_index(root, reconstruction, copy_images, 0)
}

fn reject_invalid_export(reconstruction: &Reconstruction) -> Result<()> {
    validate_for_colmap_export(reconstruction).map_err(|error| {
        anyhow::anyhow!(
            "source=reconstruction-export record=reconstruction id=- referenced=- feature=- reason={error}"
        )
    })
}

pub fn export_colmap_with_sparse_index(
    root: &Path,
    reconstruction: &Reconstruction,
    copy_images: bool,
    sparse_index: usize,
) -> Result<()> {
    reject_invalid_export(reconstruction)?;
    let sparse = sparse_files_from_reconstruction(reconstruction)?;
    let images_dir = root.join("images");
    let sparse_dir = root.join("sparse").join(sparse_index.to_string());
    fs::create_dir_all(&images_dir)?;
    fs::create_dir_all(&sparse_dir)?;
    if copy_images {
        for path in &reconstruction.image_paths {
            let name = path
                .file_name()
                .context("input image path has no file name")?;
            let dst = images_dir.join(name);
            if !dst.exists() {
                fs::copy(path, dst)?;
            }
        }
    }
    write_raw_cameras_txt(&sparse_dir.join("cameras.txt"), &sparse.cameras)?;
    write_raw_images_txt(&sparse_dir.join("images.txt"), &sparse)?;
    write_raw_points3d_txt(&sparse_dir.join("points3D.txt"), &sparse.points3d)?;
    if !sparse.rigs.is_empty() || !sparse.frames.is_empty() {
        write_raw_rigs_txt(&sparse_dir.join("rigs.txt"), &sparse.rigs)?;
        write_raw_frames_txt(&sparse_dir.join("frames.txt"), &sparse.frames)?;
    }
    Ok(())
}

pub fn export_colmap_sparse_snapshot(root: &Path, reconstruction: &Reconstruction) -> Result<()> {
    reject_invalid_export(reconstruction)?;
    let sparse = sparse_files_from_reconstruction(reconstruction)?;
    fs::create_dir_all(root)?;
    write_raw_cameras_txt(&root.join("cameras.txt"), &sparse.cameras)?;
    write_raw_images_txt(&root.join("images.txt"), &sparse)?;
    write_raw_points3d_txt(&root.join("points3D.txt"), &sparse.points3d)?;
    if !sparse.rigs.is_empty() || !sparse.frames.is_empty() {
        write_raw_rigs_txt(&root.join("rigs.txt"), &sparse.rigs)?;
        write_raw_frames_txt(&root.join("frames.txt"), &sparse.frames)?;
    }
    Ok(())
}

pub fn export_colmap_sparse_model(
    root: &Path,
    model: &ColmapSparseModel,
    copy_images: bool,
) -> Result<()> {
    reject_invalid_export(&model.reconstruction)?;
    let camera_ids = model.reconstruction.camera_ids.iter().copied().collect();
    let image_ids = model.reconstruction.image_ids.iter().copied().collect();
    let sparse_dir = root.join("sparse").join("0");
    validate_rigs_and_frames(
        &model.rigs,
        &model.frames,
        &camera_ids,
        &image_ids,
        &sparse_dir.join("rigs.txt"),
        &sparse_dir.join("frames.txt"),
        None,
    )?;
    for frame in &model.frames {
        for data_id in &frame.data_ids {
            if data_id.sensor_id.sensor_type != ColmapSensorType::Camera {
                continue;
            }
            let Some(image_index) = model
                .reconstruction
                .image_ids
                .iter()
                .position(|image_id| u64::from(*image_id) == data_id.data_id)
            else {
                continue;
            };
            let camera_id = model.reconstruction.try_camera_id_for_image(image_index)?;
            if camera_id != data_id.sensor_id.sensor_id {
                return Err(colmap_io_error(
                    &sparse_dir.join("frames.txt"),
                    "frame",
                    frame.frame_id,
                    data_id.data_id,
                    "-",
                    format!(
                        "camera data sensor {} does not match image camera {camera_id}",
                        data_id.sensor_id.sensor_id
                    ),
                )
                .into());
            }
        }
    }
    export_colmap(root, &model.reconstruction, copy_images)?;
    write_rigs_txt(&sparse_dir.join("rigs.txt"), &model.rigs)?;
    write_frames_txt(&sparse_dir.join("frames.txt"), &model.frames)?;
    Ok(())
}

pub fn write_colmap_sparse_text(root: &Path, sparse: &ColmapSparseFiles) -> Result<()> {
    validate_colmap_sparse_files(sparse, root, ColmapSparseFormat::Text)?;
    fs::create_dir_all(root)?;
    write_raw_rigs_txt(&root.join("rigs.txt"), &sparse.rigs)?;
    write_raw_cameras_txt(&root.join("cameras.txt"), &sparse.cameras)?;
    write_raw_frames_txt(&root.join("frames.txt"), &sparse.frames)?;
    write_raw_images_txt(&root.join("images.txt"), sparse)?;
    write_raw_points3d_txt(&root.join("points3D.txt"), &sparse.points3d)?;
    Ok(())
}

pub fn write_colmap_sparse_binary(root: &Path, sparse: &ColmapSparseFiles) -> Result<()> {
    validate_colmap_sparse_files(sparse, root, ColmapSparseFormat::Binary)?;
    fs::create_dir_all(root)?;
    write_raw_rigs_bin(&root.join("rigs.bin"), &sparse.rigs)?;
    write_raw_cameras_bin(&root.join("cameras.bin"), &sparse.cameras)?;
    write_raw_frames_bin(&root.join("frames.bin"), &sparse.frames)?;
    write_raw_images_bin(&root.join("images.bin"), sparse)?;
    write_raw_points3d_bin(&root.join("points3D.bin"), &sparse.points3d)?;
    Ok(())
}

pub fn write_colmap_sparse_model(
    root: &Path,
    sparse: &ColmapSparseFiles,
    format: ColmapSparseFormat,
) -> Result<()> {
    match format {
        ColmapSparseFormat::Text => write_colmap_sparse_text(root, sparse),
        ColmapSparseFormat::Binary => write_colmap_sparse_binary(root, sparse),
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn write_cameras_txt(path: &Path, reconstruction: &Reconstruction) -> Result<()> {
    let cameras = cameras_from_reconstruction(reconstruction)?;
    write_raw_cameras_txt(path, &cameras)
}

#[cfg_attr(not(test), allow(dead_code))]
fn write_images_txt(path: &Path, reconstruction: &Reconstruction) -> Result<()> {
    let sparse = sparse_files_from_reconstruction(reconstruction)?;
    write_raw_images_txt(path, &sparse)
}

#[cfg_attr(not(test), allow(dead_code))]
fn write_points3d_txt(path: &Path, reconstruction: &Reconstruction) -> Result<()> {
    let points = points3d_from_reconstruction(reconstruction)?;
    write_raw_points3d_txt(path, &points)
}

fn write_rigs_txt(path: &Path, rigs: &[ColmapRig]) -> Result<()> {
    write_raw_rigs_txt(path, rigs)
}

fn write_frames_txt(path: &Path, frames: &[ColmapFrame]) -> Result<()> {
    write_raw_frames_txt(path, frames)
}

fn cameras_from_reconstruction(reconstruction: &Reconstruction) -> Result<Vec<ColmapCamera>> {
    if reconstruction.cameras.len() != reconstruction.camera_ids.len() {
        bail!(
            "source=reconstruction-export record=camera id=- referenced=- feature=- reason=camera_ids length {} does not match cameras length {}",
            reconstruction.camera_ids.len(),
            reconstruction.cameras.len()
        );
    }
    Ok(reconstruction
        .cameras
        .iter()
        .zip(reconstruction.camera_ids.iter())
        .map(|(camera, &camera_id)| ColmapCamera {
            camera_id,
            model_id: camera.model_id,
            width: camera.width,
            height: camera.height,
            params: camera.params_slice().to_vec(),
        })
        .collect())
}

fn images_from_reconstruction(reconstruction: &Reconstruction) -> Result<Vec<ColmapImage>> {
    let mut images = Vec::new();
    for (idx, name) in reconstruction.image_names.iter().enumerate() {
        let Some(pose) = reconstruction.poses.get(idx).copied().flatten() else {
            continue;
        };
        let image_id = reconstruction.try_image_id(idx)?;
        let camera_id = reconstruction.try_camera_id_for_image(idx)?;
        reconstruction.try_camera_for_image(idx)?;
        let q = pose.quaternion();
        let t = pose.translation();
        let mut points2d = Vec::new();
        if let Some(keypoints) = reconstruction.keypoints.get(idx) {
            for (feature_idx, kp) in keypoints.iter().enumerate() {
                let point3d_id = match reconstruction
                    .observations
                    .get(idx)
                    .and_then(|obs| obs.get(feature_idx))
                    .copied()
                    .flatten()
                {
                    Some(point_index) => Some(reconstruction.try_point3d_id(point_index)?),
                    None => None,
                };
                points2d.push(ColmapPoint2D {
                    xy: [kp.x() as f64, kp.y() as f64],
                    point3d_id,
                });
            }
        }
        images.push(ColmapImage {
            image_id,
            camera_id,
            name: name.clone(),
            qvec: [q[3] as f64, q[0] as f64, q[1] as f64, q[2] as f64],
            tvec: [t[0] as f64, t[1] as f64, t[2] as f64],
            points2d,
        });
    }
    Ok(images)
}

fn points3d_from_reconstruction(reconstruction: &Reconstruction) -> Result<Vec<ColmapPoint3D>> {
    let mut points = Vec::with_capacity(reconstruction.points.len());
    for (idx, point) in reconstruction.points.iter().enumerate() {
        let mut track = Vec::with_capacity(point.track.len());
        for TrackObservation { image, feature } in &point.track {
            track.push(ColmapTrackElement {
                image_id: reconstruction.try_image_id(*image)?,
                point2d_idx: *feature as u64,
            });
        }
        points.push(ColmapPoint3D {
            point3d_id: reconstruction.try_point3d_id(idx)?,
            xyz: [
                point.xyz[0] as f64,
                point.xyz[1] as f64,
                point.xyz[2] as f64,
            ],
            color: point.color,
            error: point.error as f64,
            track,
        });
    }
    Ok(points)
}

fn sparse_files_from_reconstruction(reconstruction: &Reconstruction) -> Result<ColmapSparseFiles> {
    let mut rigs = Vec::with_capacity(reconstruction.rigs.len());
    for rig in &reconstruction.rigs {
        rigs.push(rig_to_colmap(rig)?);
    }
    Ok(ColmapSparseFiles {
        cameras: cameras_from_reconstruction(reconstruction)?,
        rigs,
        frames: reconstruction.frames.iter().map(frame_to_colmap).collect(),
        images: images_from_reconstruction(reconstruction)?,
        points3d: points3d_from_reconstruction(reconstruction)?,
    })
}

fn write_raw_cameras_txt(path: &Path, cameras: &[ColmapCamera]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "# Camera list with one line of data per camera:")?;
    writeln!(w, "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]")?;
    writeln!(w, "# Number of cameras: {}", cameras.len())?;
    for camera in sorted_cameras(cameras) {
        let model_name = colmap_camera_model_name(camera.model_id)
            .with_context(|| format!("unsupported COLMAP camera model id {}", camera.model_id))?;
        writeln!(
            w,
            "{} {} {} {} {}",
            camera.camera_id,
            model_name,
            camera.width,
            camera.height,
            format_f64_list(&camera.params)
        )?;
    }
    Ok(())
}

fn write_raw_images_txt(path: &Path, sparse: &ColmapSparseFiles) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "# Image list with two lines of data per image:")?;
    writeln!(
        w,
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"
    )?;
    writeln!(w, "#   POINTS2D[] as (X, Y, POINT3D_ID)")?;
    writeln!(
        w,
        "# Number of images: {}, mean observations per image: {}",
        sparse.images.len(),
        format_f64(mean_observations(&sparse.images))
    )?;
    for image in ordered_images_for_write(sparse) {
        validate_colmap_image_name(&image.name)?;
        writeln!(
            w,
            "{} {} {} {} {} {} {} {} {} {}",
            image.image_id,
            format_f64(image.qvec[0]),
            format_f64(image.qvec[1]),
            format_f64(image.qvec[2]),
            format_f64(image.qvec[3]),
            format_f64(image.tvec[0]),
            format_f64(image.tvec[1]),
            format_f64(image.tvec[2]),
            image.camera_id,
            image.name
        )?;
        for (feature_idx, point2d) in image.points2d.iter().enumerate() {
            if feature_idx > 0 {
                write!(w, " ")?;
            }
            let point_id = point2d
                .point3d_id
                .map(|id| id.to_string())
                .unwrap_or_else(|| "-1".to_string());
            write!(
                w,
                "{} {} {}",
                format_f64(point2d.xy[0]),
                format_f64(point2d.xy[1]),
                point_id
            )?;
        }
        writeln!(w)?;
    }
    Ok(())
}

fn write_raw_points3d_txt(path: &Path, points: &[ColmapPoint3D]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "# 3D point list with one line of data per point:")?;
    writeln!(
        w,
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)"
    )?;
    writeln!(
        w,
        "# Number of points: {}, mean track length: {}",
        points.len(),
        format_f64(mean_track_length(points))
    )?;
    for p in sorted_points3d(points) {
        write!(
            w,
            "{} {} {} {} {} {} {} {}",
            p.point3d_id,
            format_f64(p.xyz[0]),
            format_f64(p.xyz[1]),
            format_f64(p.xyz[2]),
            p.color[0],
            p.color[1],
            p.color[2],
            format_f64(p.error)
        )?;
        for elem in &p.track {
            write!(w, " {} {}", elem.image_id, elem.point2d_idx)?;
        }
        writeln!(w)?;
    }
    Ok(())
}

fn write_raw_rigs_txt(path: &Path, rigs: &[ColmapRig]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "# Rig calib list with one line of data per calib:")?;
    writeln!(
        w,
        "#   RIG_ID, NUM_SENSORS, REF_SENSOR_TYPE, REF_SENSOR_ID, SENSORS[] as (SENSOR_TYPE, SENSOR_ID, HAS_POSE, [QW, QX, QY, QZ, TX, TY, TZ])"
    )?;
    writeln!(w, "# Number of rigs: {}", rigs.len())?;
    for rig in sorted_rigs(rigs) {
        write!(w, "{} {}", rig.rig_id, rig.num_sensors()?)?;
        if let Some(ref_sensor_id) = &rig.ref_sensor_id {
            write!(w, " {}", format_sensor_id(ref_sensor_id))?;
        }
        for sensor in sorted_rig_sensors(&rig.sensors) {
            write!(w, " {}", format_sensor_id(&sensor.sensor_id))?;
            if let Some(sensor_from_rig) = &sensor.sensor_from_rig {
                write!(w, " 1 {}", format_rigid3(sensor_from_rig))?;
            } else {
                write!(w, " 0")?;
            }
        }
        writeln!(w)?;
    }
    Ok(())
}

fn write_raw_frames_txt(path: &Path, frames: &[ColmapFrame]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "# Frame list with one line of data per frame:")?;
    writeln!(
        w,
        "#   FRAME_ID, RIG_ID, RIG_FROM_WORLD[QW, QX, QY, QZ, TX, TY, TZ], NUM_DATA_IDS, DATA_IDS[] as (SENSOR_TYPE, SENSOR_ID, DATA_ID)"
    )?;
    writeln!(w, "# Number of frames: {}", frames.len())?;
    for frame in sorted_frames(frames) {
        write!(
            w,
            "{} {} {} {}",
            frame.frame_id,
            frame.rig_id,
            format_rigid3(&frame.rig_from_world),
            frame.data_ids.len()
        )?;
        for data_id in sorted_data_ids(&frame.data_ids) {
            write!(
                w,
                " {} {}",
                format_sensor_id(&data_id.sensor_id),
                data_id.data_id
            )?;
        }
        writeln!(w)?;
    }
    Ok(())
}

fn write_raw_cameras_bin(path: &Path, cameras: &[ColmapCamera]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    write_u64(&mut w, cameras.len() as u64)?;
    for camera in sorted_cameras(cameras) {
        let expected = colmap_camera_model_num_params(camera.model_id)
            .with_context(|| format!("unsupported COLMAP camera model id {}", camera.model_id))?;
        if camera.params.len() != expected {
            bail!(
                "camera_id={} model_id={} expects {} params, got {}",
                camera.camera_id,
                camera.model_id,
                expected,
                camera.params.len()
            );
        }
        write_u32(&mut w, camera.camera_id)?;
        write_i32(&mut w, camera.model_id)?;
        write_u64(&mut w, camera.width as u64)?;
        write_u64(&mut w, camera.height as u64)?;
        for &param in &camera.params {
            write_f64(&mut w, param)?;
        }
    }
    Ok(())
}

fn write_raw_images_bin(path: &Path, sparse: &ColmapSparseFiles) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    let images = ordered_images_for_write(sparse);
    write_u64(&mut w, images.len() as u64)?;
    for image in images {
        validate_colmap_image_name(&image.name)?;
        write_u32(&mut w, image.image_id)?;
        for &value in &image.qvec {
            write_f64(&mut w, value)?;
        }
        for &value in &image.tvec {
            write_f64(&mut w, value)?;
        }
        write_u32(&mut w, image.camera_id)?;
        w.write_all(image.name.as_bytes())?;
        write_u8(&mut w, 0)?;
        write_u64(&mut w, image.points2d.len() as u64)?;
        for point2d in &image.points2d {
            write_f64(&mut w, point2d.xy[0])?;
            write_f64(&mut w, point2d.xy[1])?;
            write_u64(&mut w, point2d.point3d_id.unwrap_or(u64::MAX))?;
        }
    }
    Ok(())
}

fn write_raw_points3d_bin(path: &Path, points: &[ColmapPoint3D]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    write_u64(&mut w, points.len() as u64)?;
    for point in sorted_points3d(points) {
        write_u64(&mut w, point.point3d_id)?;
        for &value in &point.xyz {
            write_f64(&mut w, value)?;
        }
        w.write_all(&point.color)?;
        write_f64(&mut w, point.error)?;
        write_u64(&mut w, point.track.len() as u64)?;
        for elem in &point.track {
            write_u32(&mut w, elem.image_id)?;
            let point2d_idx = u32::try_from(elem.point2d_idx).with_context(|| {
                format!(
                    "POINT2D_IDX {} exceeds COLMAP binary u32 range",
                    elem.point2d_idx
                )
            })?;
            write_u32(&mut w, point2d_idx)?;
        }
    }
    Ok(())
}

fn write_raw_rigs_bin(path: &Path, rigs: &[ColmapRig]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    write_u64(&mut w, rigs.len() as u64)?;
    for rig in sorted_rigs(rigs) {
        write_u32(&mut w, rig.rig_id)?;
        write_u32(&mut w, rig.num_sensors()? as u32)?;
        if let Some(ref_sensor_id) = &rig.ref_sensor_id {
            write_sensor_id_bin(&mut w, ref_sensor_id)?;
        }
        for sensor in sorted_rig_sensors(&rig.sensors) {
            write_sensor_id_bin(&mut w, &sensor.sensor_id)?;
            write_u8(&mut w, u8::from(sensor.sensor_from_rig.is_some()))?;
            if let Some(sensor_from_rig) = &sensor.sensor_from_rig {
                write_rigid3_bin(&mut w, sensor_from_rig)?;
            }
        }
    }
    Ok(())
}

fn write_raw_frames_bin(path: &Path, frames: &[ColmapFrame]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    write_u64(&mut w, frames.len() as u64)?;
    for frame in sorted_frames(frames) {
        write_u32(&mut w, frame.frame_id)?;
        write_u32(&mut w, frame.rig_id)?;
        write_rigid3_bin(&mut w, &frame.rig_from_world)?;
        write_u32(&mut w, frame.data_ids.len() as u32)?;
        for data_id in sorted_data_ids(&frame.data_ids) {
            write_sensor_id_bin(&mut w, &data_id.sensor_id)?;
            write_u64(&mut w, data_id.data_id)?;
        }
    }
    Ok(())
}

fn write_sensor_id_bin(w: &mut impl Write, sensor_id: &ColmapSensorId) -> Result<()> {
    write_i32(w, sensor_type_to_colmap_i32(&sensor_id.sensor_type))?;
    write_u32(w, sensor_id.sensor_id)?;
    Ok(())
}

fn write_rigid3_bin(w: &mut impl Write, rigid: &ColmapRigid3) -> Result<()> {
    for &value in &rigid.qvec {
        write_f64(w, value)?;
    }
    for &value in &rigid.tvec {
        write_f64(w, value)?;
    }
    Ok(())
}

impl ColmapRig {
    fn num_sensors(&self) -> Result<usize> {
        if self.ref_sensor_id.is_none() && !self.sensors.is_empty() {
            bail!(
                "COLMAP rig_id={} has non-reference sensors but no reference sensor",
                self.rig_id
            );
        }
        Ok(self.num_sensors_unchecked())
    }

    fn num_sensors_unchecked(&self) -> usize {
        self.sensors.len() + usize::from(self.ref_sensor_id.is_some())
    }
}

fn format_sensor_id(sensor_id: &ColmapSensorId) -> String {
    format!(
        "{} {}",
        sensor_type_name(&sensor_id.sensor_type),
        sensor_id.sensor_id
    )
}

fn sensor_type_name(sensor_type: &ColmapSensorType) -> &str {
    match sensor_type {
        ColmapSensorType::Invalid => "INVALID",
        ColmapSensorType::Camera => "CAMERA",
        ColmapSensorType::Imu => "IMU",
        ColmapSensorType::Other(value) => value.as_str(),
    }
}

fn format_rigid3(rigid: &ColmapRigid3) -> String {
    rigid
        .qvec
        .iter()
        .chain(rigid.tvec.iter())
        .map(|&value| format_f64(value))
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_f64(value: f64) -> String {
    format!("{value:.17}")
}

fn format_f64_list(values: &[f64]) -> String {
    values
        .iter()
        .map(|&value| format_f64(value))
        .collect::<Vec<_>>()
        .join(" ")
}

fn sorted_cameras(cameras: &[ColmapCamera]) -> Vec<&ColmapCamera> {
    let mut out = cameras.iter().collect::<Vec<_>>();
    out.sort_by_key(|camera| camera.camera_id);
    out
}

fn sorted_rigs(rigs: &[ColmapRig]) -> Vec<&ColmapRig> {
    let mut out = rigs.iter().collect::<Vec<_>>();
    out.sort_by_key(|rig| rig.rig_id);
    out
}

fn sorted_rig_sensors(sensors: &[ColmapRigSensor]) -> Vec<&ColmapRigSensor> {
    let mut out = sensors.iter().collect::<Vec<_>>();
    out.sort_by_key(|sensor| sensor_id_sort_key(&sensor.sensor_id));
    out
}

fn sorted_frames(frames: &[ColmapFrame]) -> Vec<&ColmapFrame> {
    let mut out = frames.iter().collect::<Vec<_>>();
    out.sort_by_key(|frame| frame.frame_id);
    out
}

fn sorted_data_ids(data_ids: &[ColmapDataId]) -> Vec<&ColmapDataId> {
    let mut out = data_ids.iter().collect::<Vec<_>>();
    out.sort_by_key(|data_id| (sensor_id_sort_key(&data_id.sensor_id), data_id.data_id));
    out
}

fn sorted_points3d(points: &[ColmapPoint3D]) -> Vec<&ColmapPoint3D> {
    let mut out = points.iter().collect::<Vec<_>>();
    out.sort_by_key(|point| point.point3d_id);
    out
}

fn sensor_id_sort_key(sensor_id: &ColmapSensorId) -> (i32, u32, String) {
    (
        sensor_type_to_colmap_i32(&sensor_id.sensor_type),
        sensor_id.sensor_id,
        match &sensor_id.sensor_type {
            ColmapSensorType::Other(value) => value.clone(),
            _ => String::new(),
        },
    )
}

fn ordered_images_for_write(sparse: &ColmapSparseFiles) -> Vec<&ColmapImage> {
    let mut ordered = Vec::new();
    let mut used = vec![false; sparse.images.len()];
    for frame in sorted_frames(&sparse.frames) {
        for data_id in sorted_data_ids(&frame.data_ids) {
            if data_id.sensor_id.sensor_type != ColmapSensorType::Camera {
                continue;
            }
            let Ok(image_id) = u32::try_from(data_id.data_id) else {
                continue;
            };
            if let Some((idx, image)) = sparse
                .images
                .iter()
                .enumerate()
                .find(|(idx, image)| !used[*idx] && image.image_id == image_id)
            {
                ordered.push(image);
                used[idx] = true;
            }
        }
    }
    let mut rest = sparse
        .images
        .iter()
        .enumerate()
        .filter(|(idx, _)| !used[*idx])
        .map(|(_, image)| image)
        .collect::<Vec<_>>();
    rest.sort_by_key(|image| image.image_id);
    ordered.extend(rest);
    ordered
}

fn mean_observations(images: &[ColmapImage]) -> f64 {
    if images.is_empty() {
        0.0
    } else {
        images
            .iter()
            .map(|image| image.points2d.len())
            .sum::<usize>() as f64
            / images.len() as f64
    }
}

fn mean_track_length(points: &[ColmapPoint3D]) -> f64 {
    if points.is_empty() {
        0.0
    } else {
        points.iter().map(|point| point.track.len()).sum::<usize>() as f64 / points.len() as f64
    }
}

fn sensor_type_to_colmap_i32(sensor_type: &ColmapSensorType) -> i32 {
    match sensor_type {
        ColmapSensorType::Invalid => -1,
        ColmapSensorType::Camera => 0,
        ColmapSensorType::Imu => 1,
        ColmapSensorType::Other(value) => value.parse::<i32>().unwrap_or(-1),
    }
}

fn rig_to_colmap(rig: &Rig) -> Result<ColmapRig> {
    let ref_sensor_id = rig.ref_sensor_id.as_ref().map(sensor_id_to_colmap);
    let mut sensors = Vec::new();
    for sensor in &rig.sensors {
        let sensor_id = sensor_id_to_colmap(&sensor.sensor_id);
        if ref_sensor_id.as_ref() == Some(&sensor_id) {
            if sensor.sensor_from_rig.is_some() {
                bail!(
                    "source=reconstruction-export record=rig id={} referenced={} feature=- reason=posed reference sensor is not representable in COLMAP",
                    rig.rig_id,
                    sensor.sensor_id.sensor_id
                );
            }
            continue;
        }
        sensors.push(rig_sensor_to_colmap(sensor));
    }
    Ok(ColmapRig {
        rig_id: rig.rig_id,
        ref_sensor_id,
        sensors,
    })
}

fn rig_sensor_to_colmap(sensor: &RigSensor) -> ColmapRigSensor {
    ColmapRigSensor {
        sensor_id: sensor_id_to_colmap(&sensor.sensor_id),
        sensor_from_rig: sensor.sensor_from_rig.as_ref().map(rigid3_to_colmap),
    }
}

fn frame_to_colmap(frame: &Frame) -> ColmapFrame {
    ColmapFrame {
        frame_id: frame.frame_id,
        rig_id: frame.rig_id,
        rig_from_world: rigid3_to_colmap(&frame.rig_from_world),
        data_ids: frame.data_ids.iter().map(data_id_to_colmap).collect(),
    }
}

fn sensor_id_to_colmap(sensor_id: &SensorId) -> ColmapSensorId {
    ColmapSensorId {
        sensor_type: sensor_type_to_colmap(&sensor_id.sensor_type),
        sensor_id: sensor_id.sensor_id,
    }
}

fn sensor_type_to_colmap(sensor_type: &SensorType) -> ColmapSensorType {
    match sensor_type {
        SensorType::Invalid => ColmapSensorType::Invalid,
        SensorType::Camera => ColmapSensorType::Camera,
        SensorType::Imu => ColmapSensorType::Imu,
        SensorType::Other(value) => ColmapSensorType::Other(value.clone()),
    }
}

fn rigid3_to_colmap(rigid: &Rigid3) -> ColmapRigid3 {
    ColmapRigid3 {
        qvec: rigid.qvec,
        tvec: rigid.tvec,
    }
}

fn data_id_to_colmap(data_id: &DataId) -> ColmapDataId {
    ColmapDataId {
        sensor_id: sensor_id_to_colmap(&data_id.sensor_id),
        data_id: data_id.data_id,
    }
}

fn read_images_txt(path: &Path) -> Result<Vec<ColmapImage>> {
    let reader = BufReader::new(File::open(path)?);
    let mut images = Vec::new();
    let mut lines = reader.lines();
    while let Some(line) = lines.next() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let prefix = trimmed.split_whitespace().take(9).collect::<Vec<_>>();
        if prefix.len() < 9 || prefix[0].parse::<u32>().is_err() {
            bail!("invalid image record in {}: {trimmed}", path.display());
        }
        let name = parse_image_name_from_header(trimmed)
            .with_context(|| format!("missing image name in {}", path.display()))?;
        validate_colmap_image_name(&name)?;
        let points_line = lines
            .next()
            .transpose()?
            .with_context(|| format!("missing points2D line after image in {}", path.display()))?;
        images.push(ColmapImage {
            image_id: prefix[0].parse()?,
            camera_id: prefix[8].parse()?,
            qvec: [
                prefix[1].parse()?,
                prefix[2].parse()?,
                prefix[3].parse()?,
                prefix[4].parse()?,
            ],
            tvec: [prefix[5].parse()?, prefix[6].parse()?, prefix[7].parse()?],
            name,
            points2d: parse_points2d_txt(&points_line, path)?,
        });
    }
    Ok(images)
}

fn parse_image_name_from_header(line: &str) -> Option<String> {
    let mut in_token = false;
    let mut tokens_seen = 0usize;
    for (idx, ch) in line.char_indices() {
        if ch.is_whitespace() {
            if in_token {
                tokens_seen += 1;
                in_token = false;
                if tokens_seen == 9 {
                    let name = line[idx..].trim_start();
                    return (!name.is_empty()).then(|| name.to_string());
                }
            }
        } else if !in_token {
            in_token = true;
        }
    }
    if in_token {
        tokens_seen += 1;
    }
    (tokens_seen > 9).then_some(String::new())
}

fn validate_colmap_image_name(name: &str) -> Result<()> {
    let path = Path::new(name);
    if name.is_empty()
        || name.contains('\\')
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        bail!(
            "unsafe COLMAP image name '{name}': expected a relative path without parent traversal"
        );
    }
    Ok(())
}

fn parse_points2d_txt(line: &str, path: &Path) -> Result<Vec<ColmapPoint2D>> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    let parts = trimmed.split_whitespace().collect::<Vec<_>>();
    if parts.len() % 3 != 0 {
        bail!(
            "points2D line must contain X/Y/POINT3D_ID triples in {}",
            path.display()
        );
    }
    parts
        .chunks(3)
        .map(|chunk| {
            let point3d_id = parse_optional_point3d_id_text(chunk[2])?;
            Ok(ColmapPoint2D {
                xy: [chunk[0].parse()?, chunk[1].parse()?],
                point3d_id,
            })
        })
        .collect()
}

fn parse_optional_point3d_id_text(value: &str) -> Result<Option<u64>> {
    let signed = value.parse::<i64>()?;
    if signed == -1 {
        Ok(None)
    } else if signed < 0 {
        bail!("invalid negative COLMAP point3D id {signed}")
    } else {
        Ok(Some(signed as u64))
    }
}

fn read_images_bin(path: &Path) -> Result<Vec<ColmapImage>> {
    let mut f = File::open(path)?;
    let n = checked_binary_count(read_u64(&mut f)?, &mut f, 73, "images", path)?;
    let mut images = reservable_vec(n, "images", path)?;
    for _ in 0..n {
        let image_id = read_u32(&mut f)?;
        let qvec = [
            read_f64(&mut f)?,
            read_f64(&mut f)?,
            read_f64(&mut f)?,
            read_f64(&mut f)?,
        ];
        let tvec = [read_f64(&mut f)?, read_f64(&mut f)?, read_f64(&mut f)?];
        let camera_id = read_u32(&mut f)?;
        let name = read_cstr(&mut f)?;
        validate_colmap_image_name(&name)?;
        let m = checked_binary_count(read_u64(&mut f)?, &mut f, 24, "points2D", path)?;
        let mut points2d = reservable_vec(m, "points2D", path)?;
        for _ in 0..m {
            let xy = [read_f64(&mut f)?, read_f64(&mut f)?];
            let raw_point3d_id = read_u64(&mut f)?;
            points2d.push(ColmapPoint2D {
                xy,
                point3d_id: optional_point3d_id_bin(raw_point3d_id),
            });
        }
        images.push(ColmapImage {
            image_id,
            camera_id,
            name,
            qvec,
            tvec,
            points2d,
        });
    }
    Ok(images)
}

fn optional_point3d_id_bin(point3d_id: u64) -> Option<u64> {
    (point3d_id != u64::MAX).then_some(point3d_id)
}

fn read_cameras_txt(path: &Path) -> Result<Vec<ColmapCamera>> {
    let reader = BufReader::new(File::open(path)?);
    let mut cameras = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts: Vec<_> = trimmed.split_whitespace().collect();
        if parts.len() < 5 || parts[0].parse::<u32>().is_err() {
            bail!("invalid camera record in {}: {trimmed}", path.display());
        }
        let camera_id = parts[0].parse()?;
        let model_id = colmap_camera_model_id(parts[1])
            .with_context(|| format!("unsupported COLMAP camera model '{}'", parts[1]))?;
        let width = parts[2].parse()?;
        let height = parts[3].parse()?;
        let params = parts[4..]
            .iter()
            .map(|p| p.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()?;
        let expected = colmap_camera_model_num_params(model_id)
            .context("unsupported COLMAP camera model id")?;
        if params.len() != expected {
            bail!(
                "camera model {} expects {} params, got {} in {}",
                parts[1],
                expected,
                params.len(),
                path.display()
            );
        }
        cameras.push(ColmapCamera {
            camera_id,
            model_id,
            width,
            height,
            params,
        });
    }
    if cameras.is_empty() {
        bail!("no camera in {}", path.display());
    }
    Ok(cameras)
}

fn read_cameras_bin(path: &Path) -> Result<Vec<ColmapCamera>> {
    let mut f = File::open(path)?;
    let n = checked_binary_count(read_u64(&mut f)?, &mut f, 24, "cameras", path)?;
    if n == 0 {
        bail!("empty camera file {}", path.display());
    }
    let mut cameras = reservable_vec(n, "cameras", path)?;
    for _ in 0..n {
        let camera_id = read_u32(&mut f)?;
        let model_id = read_i32(&mut f)?;
        let width = u32::try_from(read_u64(&mut f)?)
            .with_context(|| format!("camera width exceeds u32 in {}", path.display()))?;
        let height = u32::try_from(read_u64(&mut f)?)
            .with_context(|| format!("camera height exceeds u32 in {}", path.display()))?;
        let num_params = colmap_camera_model_num_params(model_id)
            .with_context(|| format!("unsupported COLMAP camera model id {model_id}"))?;
        let mut params = Vec::with_capacity(num_params);
        for _ in 0..num_params {
            params.push(read_f64(&mut f)?);
        }
        cameras.push(ColmapCamera {
            camera_id,
            model_id,
            width,
            height,
            params,
        });
    }
    Ok(cameras)
}

fn read_points3d_txt(path: &Path) -> Result<Vec<ColmapPoint3D>> {
    let reader = BufReader::new(File::open(path)?);
    let mut points = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts = trimmed.split_whitespace().collect::<Vec<_>>();
        if parts.len() < 8 || parts[0].parse::<u64>().is_err() {
            bail!("invalid point3D record in {}: {trimmed}", path.display());
        }
        let mut track = Vec::new();
        for chunk in parts[8..].chunks(2) {
            if chunk.len() != 2 {
                bail!(
                    "point3D track must contain IMAGE_ID/POINT2D_IDX pairs in {}",
                    path.display()
                );
            }
            track.push(ColmapTrackElement {
                image_id: chunk[0].parse()?,
                point2d_idx: chunk[1].parse()?,
            });
        }
        points.push(ColmapPoint3D {
            point3d_id: parts[0].parse()?,
            xyz: [parts[1].parse()?, parts[2].parse()?, parts[3].parse()?],
            color: [parts[4].parse()?, parts[5].parse()?, parts[6].parse()?],
            error: parts[7].parse()?,
            track,
        });
    }
    Ok(points)
}

fn read_points3d_bin(path: &Path) -> Result<Vec<ColmapPoint3D>> {
    let mut f = File::open(path)?;
    let n = checked_binary_count(read_u64(&mut f)?, &mut f, 51, "points3D", path)?;
    let mut points = reservable_vec(n, "points3D", path)?;
    for _ in 0..n {
        let point3d_id = read_u64(&mut f)?;
        let xyz = [read_f64(&mut f)?, read_f64(&mut f)?, read_f64(&mut f)?];
        let color = [read_u8(&mut f)?, read_u8(&mut f)?, read_u8(&mut f)?];
        let error = read_f64(&mut f)?;
        let track_length =
            checked_binary_count(read_u64(&mut f)?, &mut f, 8, "point3D track", path)?;
        let mut track = reservable_vec(track_length, "point3D track", path)?;
        for _ in 0..track_length {
            track.push(ColmapTrackElement {
                image_id: read_u32(&mut f)?,
                point2d_idx: u64::from(read_u32(&mut f)?),
            });
        }
        points.push(ColmapPoint3D {
            point3d_id,
            xyz,
            color,
            error,
            track,
        });
    }
    Ok(points)
}

fn read_rigs_txt(path: &Path) -> Result<Vec<ColmapRig>> {
    let reader = BufReader::new(File::open(path)?);
    let mut rigs = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts = trimmed.split_whitespace().collect::<Vec<_>>();
        if parts.len() < 2 || parts[0].parse::<u32>().is_err() {
            bail!("invalid rig record in {}: {trimmed}", path.display());
        }
        let rig_id = parts[0].parse()?;
        let num_sensors = parts[1].parse::<usize>()?;
        let mut cursor = 2usize;
        let min_sensor_fields = if num_sensors == 0 {
            0
        } else {
            2usize
                .checked_add(
                    num_sensors
                        .saturating_sub(1)
                        .checked_mul(3)
                        .context("rig sensor count overflows the minimum text field calculation")?,
                )
                .context("rig sensor count overflows the minimum text field calculation")?
        };
        if min_sensor_fields > parts.len().saturating_sub(cursor) {
            bail!(
                "rig declares {num_sensors} sensors but has too few fields in {}",
                path.display()
            );
        }
        let ref_sensor_id = if num_sensors > 0 {
            Some(parse_sensor_id(&parts, &mut cursor, path)?)
        } else {
            None
        };
        let mut sensors = Vec::new();
        for _ in 0..num_sensors.saturating_sub(1) {
            let sensor_id = parse_sensor_id(&parts, &mut cursor, path)?;
            let has_pose = parse_next::<u32>(&parts, &mut cursor, path)? == 1;
            let sensor_from_rig = if has_pose {
                Some(parse_rigid3(&parts, &mut cursor, path)?)
            } else {
                None
            };
            sensors.push(ColmapRigSensor {
                sensor_id,
                sensor_from_rig,
            });
        }
        if cursor != parts.len() {
            bail!("unexpected trailing rig fields in {}", path.display());
        }
        rigs.push(ColmapRig {
            rig_id,
            ref_sensor_id,
            sensors,
        });
    }
    Ok(rigs)
}

fn read_frames_txt(path: &Path) -> Result<Vec<ColmapFrame>> {
    let reader = BufReader::new(File::open(path)?);
    let mut frames = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts = trimmed.split_whitespace().collect::<Vec<_>>();
        if parts.len() < 10 || parts[0].parse::<u32>().is_err() {
            bail!("invalid frame record in {}: {trimmed}", path.display());
        }
        let mut cursor = 0usize;
        let frame_id = parse_next(&parts, &mut cursor, path)?;
        let rig_id = parse_next(&parts, &mut cursor, path)?;
        let rig_from_world = parse_rigid3(&parts, &mut cursor, path)?;
        let num_data_ids = parse_next::<usize>(&parts, &mut cursor, path)?;
        if num_data_ids > parts.len().saturating_sub(cursor) / 3 {
            bail!(
                "frame declares {num_data_ids} data ids but has too few fields in {}",
                path.display()
            );
        }
        let mut data_ids = reservable_vec(num_data_ids, "frame data ids", path)?;
        for _ in 0..num_data_ids {
            let sensor_id = parse_sensor_id(&parts, &mut cursor, path)?;
            let data_id = parse_next(&parts, &mut cursor, path)?;
            data_ids.push(ColmapDataId { sensor_id, data_id });
        }
        if cursor != parts.len() {
            bail!("unexpected trailing frame fields in {}", path.display());
        }
        frames.push(ColmapFrame {
            frame_id,
            rig_id,
            rig_from_world,
            data_ids,
        });
    }
    Ok(frames)
}

fn read_rigs_bin(path: &Path) -> Result<Vec<ColmapRig>> {
    let mut f = File::open(path)?;
    let n = checked_binary_count(read_u64(&mut f)?, &mut f, 8, "rigs", path)?;
    let mut rigs = reservable_vec(n, "rigs", path)?;
    for _ in 0..n {
        let rig_id = read_u32(&mut f)?;
        let num_sensors =
            checked_binary_count(u64::from(read_u32(&mut f)?), &mut f, 8, "rig sensors", path)?;
        let ref_sensor_id = if num_sensors > 0 {
            Some(read_sensor_id_bin(&mut f)?)
        } else {
            None
        };
        let mut sensors = reservable_vec(
            num_sensors.saturating_sub(1),
            "non-reference rig sensors",
            path,
        )?;
        for _ in 0..num_sensors.saturating_sub(1) {
            let sensor_id = read_sensor_id_bin(&mut f)?;
            let has_pose = read_u8(&mut f)? != 0;
            let sensor_from_rig = if has_pose {
                Some(read_rigid3_bin(&mut f)?)
            } else {
                None
            };
            sensors.push(ColmapRigSensor {
                sensor_id,
                sensor_from_rig,
            });
        }
        rigs.push(ColmapRig {
            rig_id,
            ref_sensor_id,
            sensors,
        });
    }
    Ok(rigs)
}

fn read_frames_bin(path: &Path) -> Result<Vec<ColmapFrame>> {
    let mut f = File::open(path)?;
    let n = checked_binary_count(read_u64(&mut f)?, &mut f, 68, "frames", path)?;
    let mut frames = reservable_vec(n, "frames", path)?;
    for _ in 0..n {
        let frame_id = read_u32(&mut f)?;
        let rig_id = read_u32(&mut f)?;
        let rig_from_world = read_rigid3_bin(&mut f)?;
        let num_data_ids = checked_binary_count(
            u64::from(read_u32(&mut f)?),
            &mut f,
            16,
            "frame data ids",
            path,
        )?;
        let mut data_ids = reservable_vec(num_data_ids, "frame data ids", path)?;
        for _ in 0..num_data_ids {
            data_ids.push(ColmapDataId {
                sensor_id: read_sensor_id_bin(&mut f)?,
                data_id: read_u64(&mut f)?,
            });
        }
        frames.push(ColmapFrame {
            frame_id,
            rig_id,
            rig_from_world,
            data_ids,
        });
    }
    Ok(frames)
}

fn read_sensor_id_bin(r: &mut impl Read) -> Result<ColmapSensorId> {
    Ok(ColmapSensorId {
        sensor_type: ColmapSensorType::from_colmap_i32(read_i32(r)?),
        sensor_id: read_u32(r)?,
    })
}

fn read_rigid3_bin(r: &mut impl Read) -> Result<ColmapRigid3> {
    Ok(ColmapRigid3 {
        qvec: [read_f64(r)?, read_f64(r)?, read_f64(r)?, read_f64(r)?],
        tvec: [read_f64(r)?, read_f64(r)?, read_f64(r)?],
    })
}

fn parse_rigid3(parts: &[&str], cursor: &mut usize, path: &Path) -> Result<ColmapRigid3> {
    Ok(ColmapRigid3 {
        qvec: [
            parse_next(parts, cursor, path)?,
            parse_next(parts, cursor, path)?,
            parse_next(parts, cursor, path)?,
            parse_next(parts, cursor, path)?,
        ],
        tvec: [
            parse_next(parts, cursor, path)?,
            parse_next(parts, cursor, path)?,
            parse_next(parts, cursor, path)?,
        ],
    })
}

fn parse_sensor_id(parts: &[&str], cursor: &mut usize, path: &Path) -> Result<ColmapSensorId> {
    let sensor_type = parse_next_str(parts, cursor, path)?;
    let sensor_id = parse_next(parts, cursor, path)?;
    Ok(ColmapSensorId {
        sensor_type: ColmapSensorType::from_colmap_str(sensor_type),
        sensor_id,
    })
}

fn parse_next<T>(parts: &[&str], cursor: &mut usize, path: &Path) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    let value = parse_next_str(parts, cursor, path)?;
    Ok(value.parse()?)
}

fn parse_next_str<'a>(parts: &'a [&'a str], cursor: &mut usize, path: &Path) -> Result<&'a str> {
    let value = parts
        .get(*cursor)
        .copied()
        .with_context(|| format!("truncated COLMAP text record in {}", path.display()))?;
    *cursor += 1;
    Ok(value)
}

impl ColmapSensorType {
    fn from_colmap_str(value: &str) -> Self {
        match value {
            "INVALID" => Self::Invalid,
            "CAMERA" => Self::Camera,
            "IMU" => Self::Imu,
            other => Self::Other(other.to_string()),
        }
    }

    fn from_colmap_i32(value: i32) -> Self {
        match value {
            -1 => Self::Invalid,
            0 => Self::Camera,
            1 => Self::Imu,
            other => Self::Other(other.to_string()),
        }
    }
}

fn checked_binary_count(
    raw_count: u64,
    file: &mut File,
    min_record_bytes: u64,
    label: &str,
    path: &Path,
) -> Result<usize> {
    let count = usize::try_from(raw_count)
        .with_context(|| format!("{label} count does not fit usize in {}", path.display()))?;
    let position = file.stream_position()?;
    let file_len = file.metadata()?.len();
    let remaining = file_len.checked_sub(position).with_context(|| {
        format!(
            "reader position {position} exceeds file length {file_len} in {}",
            path.display()
        )
    })?;
    if min_record_bytes == 0 || raw_count <= remaining / min_record_bytes {
        Ok(count)
    } else {
        bail!(
            "{label} count {raw_count} cannot fit in the {remaining} remaining bytes of {}",
            path.display()
        )
    }
}

fn reservable_vec<T>(count: usize, label: &str, path: &Path) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve(count).with_context(|| {
        format!(
            "cannot reserve capacity for {count} {label} entries from {}",
            path.display()
        )
    })?;
    Ok(values)
}

fn read_u8(r: &mut impl Read) -> std::io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_i32(r: &mut impl Read) -> std::io::Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}

fn read_f64(r: &mut impl Read) -> std::io::Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}

fn read_cstr(r: &mut impl Read) -> std::io::Result<String> {
    let mut bytes = Vec::new();
    loop {
        let mut b = [0u8; 1];
        r.read_exact(&mut b)?;
        if b[0] == 0 {
            break;
        }
        bytes.push(b[0]);
    }
    Ok(String::from_utf8_lossy(&bytes).to_string())
}

fn write_u8(w: &mut impl Write, value: u8) -> std::io::Result<()> {
    w.write_all(&[value])
}

fn write_u32(w: &mut impl Write, value: u32) -> std::io::Result<()> {
    w.write_all(&value.to_le_bytes())
}

fn write_u64(w: &mut impl Write, value: u64) -> std::io::Result<()> {
    w.write_all(&value.to_le_bytes())
}

fn write_i32(w: &mut impl Write, value: i32) -> std::io::Result<()> {
    w.write_all(&value.to_le_bytes())
}

fn write_f64(w: &mut impl Write, value: f64) -> std::io::Result<()> {
    w.write_all(&value.to_le_bytes())
}

#[allow(dead_code)]
pub fn se3_to_colmap_pose(name: String, pose: SE3) -> ColmapPose {
    let q = pose.quaternion();
    let t = pose.translation();
    ColmapPose {
        image_id: 0,
        camera_id: 1,
        name,
        qvec: [q[3] as f64, q[0] as f64, q[1] as f64, q[2] as f64],
        tvec: [t[0] as f64, t[1] as f64, t[2] as f64],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        Point3D, COLMAP_FULL_OPENCV, COLMAP_PINHOLE, COLMAP_SIMPLE_PINHOLE, COLMAP_SIMPLE_RADIAL,
    };
    use rustscan_slam::{tracker::PnPProblem, tracker::PnPSolver, KeyPoint, SE3};
    use std::collections::BTreeMap;
    use tempfile::tempdir;

    #[test]
    fn malformed_binary_collection_counts_return_errors_without_panicking() {
        fn assert_rejected_without_panic<T>(path: &Path, reader: fn(&Path) -> Result<Vec<T>>) {
            let attempt = std::panic::catch_unwind(|| reader(path));
            assert!(attempt.is_ok(), "{} count must not panic", path.display());
            assert!(
                attempt.unwrap().is_err(),
                "{} count must be rejected",
                path.display()
            );
        }

        let dir = tempdir().unwrap();
        for name in ["cameras.bin", "images.bin", "points3D.bin"] {
            let path = dir.path().join(name);
            std::fs::write(&path, u64::MAX.to_le_bytes()).unwrap();
        }
        assert_rejected_without_panic(&dir.path().join("cameras.bin"), read_cameras_bin);
        assert_rejected_without_panic(&dir.path().join("images.bin"), read_images_bin);
        assert_rejected_without_panic(&dir.path().join("points3D.bin"), read_points3d_bin);
    }

    #[test]
    fn reads_simple_pinhole_text_camera_without_param_shift() -> Result<()> {
        let dir = tempdir()?;
        let sparse = dir.path().join("sparse/0");
        fs::create_dir_all(&sparse)?;
        fs::write(
            sparse.join("cameras.txt"),
            "# cameras\n1 SIMPLE_PINHOLE 640 480 500 320 240\n",
        )?;

        let camera = read_camera_model(dir.path())?;

        assert_eq!(camera.model_id, COLMAP_SIMPLE_PINHOLE);
        assert_eq!(camera.model_name(), "SIMPLE_PINHOLE");
        assert_eq!(camera.num_params, 3);
        assert_eq!(camera.params_slice(), &[500.0, 320.0, 240.0]);
        assert_eq!(camera.fx, 500.0);
        assert_eq!(camera.fy, 500.0);
        assert_eq!(camera.cx, 320.0);
        assert_eq!(camera.cy, 240.0);
        Ok(())
    }

    #[test]
    fn reads_simple_radial_text_camera_intrinsics_and_distortion() -> Result<()> {
        let dir = tempdir()?;
        let sparse = dir.path().join("sparse/0");
        fs::create_dir_all(&sparse)?;
        fs::write(
            sparse.join("cameras.txt"),
            "7 SIMPLE_RADIAL 800 600 700 401 299 -0.0125\n",
        )?;

        let camera = read_camera_model(dir.path())?;

        assert_eq!(camera.model_id, COLMAP_SIMPLE_RADIAL);
        assert_eq!(camera.params_slice(), &[700.0, 401.0, 299.0, -0.0125]);
        assert_eq!(camera.fx, 700.0);
        assert_eq!(camera.fy, 700.0);
        assert_eq!(camera.cx, 401.0);
        assert_eq!(camera.cy, 299.0);
        Ok(())
    }

    #[test]
    fn reads_binary_camera_with_current_colmap_model_param_count() -> Result<()> {
        let dir = tempdir()?;
        let sparse = dir.path().join("sparse/0");
        fs::create_dir_all(&sparse)?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&(COLMAP_FULL_OPENCV as i32).to_le_bytes());
        bytes.extend_from_slice(&1024u64.to_le_bytes());
        bytes.extend_from_slice(&768u64.to_le_bytes());
        let params: [f64; 12] = [
            900.0, 901.0, 512.0, 384.0, 0.1, -0.02, 0.003, -0.004, 0.0005, 0.0, 0.0, 0.0,
        ];
        for param in params {
            bytes.extend_from_slice(&param.to_le_bytes());
        }
        fs::write(sparse.join("cameras.bin"), bytes)?;

        let camera = read_camera_model(dir.path())?;

        assert_eq!(camera.model_id, COLMAP_FULL_OPENCV);
        assert_eq!(camera.model_name(), "FULL_OPENCV");
        assert_eq!(camera.num_params, 12);
        assert_eq!(camera.fx, 900.0);
        assert_eq!(camera.fy, 901.0);
        assert_eq!(camera.cx, 512.0);
        assert_eq!(camera.cy, 384.0);
        assert_eq!(camera.params_slice()[11], 0.0);
        Ok(())
    }

    #[test]
    fn reads_text_images_with_points2d_and_optional_point3d_ids() -> Result<()> {
        let dir = tempdir()?;
        let sparse = dir.path().join("sparse/0");
        fs::create_dir_all(&sparse)?;
        fs::write(
            sparse.join("images.txt"),
            "# images\n7 1 0 0 0 0 0 0 11 image.jpg\n10.5 20.5 99 30.0 40.0 -1\n",
        )?;

        let images = read_colmap_images(dir.path())?;
        let poses = read_colmap_poses(dir.path())?;

        assert_eq!(images.len(), 1);
        assert_eq!(images[0].image_id, 7);
        assert_eq!(images[0].camera_id, 11);
        assert_eq!(images[0].name, "image.jpg");
        assert_eq!(images[0].points2d.len(), 2);
        assert_eq!(images[0].points2d[0].xy, [10.5, 20.5]);
        assert_eq!(images[0].points2d[0].point3d_id, Some(99));
        assert_eq!(images[0].points2d[1].point3d_id, None);
        assert_eq!(poses[0].image_id, 7);
        assert_eq!(poses[0].name, "image.jpg");
        Ok(())
    }

    #[test]
    fn reads_text_image_name_to_end_of_line() -> Result<()> {
        let dir = tempdir()?;
        let sparse = dir.path().join("sparse/0");
        fs::create_dir_all(&sparse)?;
        fs::write(
            sparse.join("images.txt"),
            "# images\n7 1 0 0 0 0 0 0 11 folder with spaces/image 01.jpg\n10.5 20.5 -1\n",
        )?;

        let images = read_colmap_images(dir.path())?;

        assert_eq!(images.len(), 1);
        assert_eq!(images[0].name, "folder with spaces/image 01.jpg");
        assert_eq!(images[0].points2d[0].xy, [10.5, 20.5]);
        Ok(())
    }

    #[test]
    fn rejects_absolute_and_parent_traversing_image_names() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("images.txt");
        for name in ["../escape.jpg", "/absolute.jpg", "C:\\absolute.jpg"] {
            fs::write(&path, format!("1 1 0 0 0 0 0 0 1 {name}\n\n"))?;
            assert!(read_images_txt(&path).is_err(), "{name}");
        }
        Ok(())
    }

    #[test]
    fn reads_binary_images_with_points2d() -> Result<()> {
        let dir = tempdir()?;
        let sparse = dir.path().join("sparse/0");
        fs::create_dir_all(&sparse)?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&7u32.to_le_bytes());
        for value in [1.0f64, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&11u32.to_le_bytes());
        bytes.extend_from_slice(b"image.jpg\0");
        bytes.extend_from_slice(&2u64.to_le_bytes());
        for (x, y, point3d_id) in [(10.5f64, 20.5f64, 99u64), (30.0, 40.0, u64::MAX)] {
            bytes.extend_from_slice(&x.to_le_bytes());
            bytes.extend_from_slice(&y.to_le_bytes());
            bytes.extend_from_slice(&point3d_id.to_le_bytes());
        }
        fs::write(sparse.join("images.bin"), bytes)?;

        let images = read_colmap_images(dir.path())?;

        assert_eq!(images.len(), 1);
        assert_eq!(images[0].image_id, 7);
        assert_eq!(images[0].camera_id, 11);
        assert_eq!(images[0].name, "image.jpg");
        assert_eq!(images[0].qvec, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(images[0].tvec, [0.1, 0.2, 0.3]);
        assert_eq!(images[0].points2d[0].xy, [10.5, 20.5]);
        assert_eq!(images[0].points2d[0].point3d_id, Some(99));
        assert_eq!(images[0].points2d[1].point3d_id, None);
        Ok(())
    }

    #[test]
    #[ignore = "requires the external artifacts/inputs/flowers2_colmap fixture"]
    fn real_colmap_sparse_tracks_recover_registered_image_pose_with_pnp() -> Result<()> {
        let sparse = read_colmap_sparse_files_with_format(
            Path::new("../artifacts/inputs/flowers2_colmap/sparse/text"),
            ColmapSparseFormat::Text,
        )?;
        let camera = sparse
            .cameras
            .iter()
            .find(|camera| camera.camera_id == 1)
            .cloned()
            .map(camera_model_from_colmap)
            .transpose()?
            .expect("fixture camera");
        assert_eq!(camera.model_id, COLMAP_PINHOLE);

        let image = sparse
            .images
            .iter()
            .find(|image| image.image_id == 3)
            .expect("fixture image");
        let points_by_id = sparse
            .points3d
            .iter()
            .map(|point| (point.point3d_id, point.xyz))
            .collect::<BTreeMap<_, _>>();

        let mut problem = PnPProblem::new();
        for point2d in &image.points2d {
            let Some(point3d_id) = point2d.point3d_id else {
                continue;
            };
            let Some(xyz) = points_by_id.get(&point3d_id) else {
                continue;
            };
            let Some(cam_point) =
                camera.cam_from_img_f32(point2d.xy[0] as f32, point2d.xy[1] as f32)
            else {
                continue;
            };
            problem.add_correspondence(cam_point, [xyz[0] as f32, xyz[1] as f32, xyz[2] as f32]);
            if problem.image_points.len() == 256 {
                break;
            }
        }
        assert_eq!(problem.image_points.len(), 256);

        let mut solver = PnPSolver::new(1.0, 1.0, 0.0, 0.0);
        solver.ransac_threshold = camera.cam_from_img_threshold(8.0) as f32;
        solver.ransac_confidence = 0.99999;
        solver.ransac_min_iterations = 100;
        solver.ransac_max_iterations = 10_000;
        solver.ransac_random_seed = Some(0);
        let (estimated, inliers) = solver.solve(&problem).expect("COLMAP fixture PnP pose");
        assert!(
            inliers.iter().filter(|&&value| value).count() >= 240,
            "real COLMAP tracks should be mostly consistent with the registered pose"
        );

        let reference = se3_from_colmap_pose(image.qvec, image.tvec)?;
        let rotation_error = rotation_error_deg(&reference, &estimated);
        let translation_error = translation_error(&reference, &estimated);

        assert!(
            rotation_error < 0.05,
            "rotation_error={rotation_error}deg reference={:?} estimated={:?}",
            reference.quaternion(),
            estimated.quaternion()
        );
        assert!(
            translation_error < 0.05,
            "translation_error={translation_error} reference={:?} estimated={:?}",
            reference.translation(),
            estimated.translation()
        );
        Ok(())
    }

    #[test]
    fn raw_sparse_text_and_binary_roundtrip_preserves_colmap_files() -> Result<()> {
        let dir = tempdir()?;
        let original = sample_sparse_files();
        let text_dir = dir.path().join("text");
        let bin_dir = dir.path().join("bin");

        write_colmap_sparse_text(&text_dir, &original)?;
        let text_roundtrip =
            read_colmap_sparse_files_with_format(&text_dir, ColmapSparseFormat::Text)?;
        assert_eq!(
            normalize_sparse_files(text_roundtrip),
            normalize_sparse_files(original.clone())
        );

        write_colmap_sparse_binary(&bin_dir, &original)?;
        let bin_roundtrip =
            read_colmap_sparse_files_with_format(&bin_dir, ColmapSparseFormat::Binary)?;
        assert_eq!(
            normalize_sparse_files(bin_roundtrip),
            normalize_sparse_files(original)
        );

        Ok(())
    }

    fn rotation_error_deg(a: &SE3, b: &SE3) -> f32 {
        let ra = a.rotation_matrix();
        let rb = b.rotation_matrix();
        let mut trace = 0.0f32;
        for row in 0..3 {
            for col in 0..3 {
                trace += ra[row][col] * rb[row][col];
            }
        }
        ((trace - 1.0) * 0.5).clamp(-1.0, 1.0).acos().to_degrees()
    }

    fn translation_error(a: &SE3, b: &SE3) -> f32 {
        let ta = a.translation();
        let tb = b.translation();
        ((ta[0] - tb[0]).powi(2) + (ta[1] - tb[1]).powi(2) + (ta[2] - tb[2]).powi(2)).sqrt()
    }

    #[test]
    fn read_sparse_files_prefers_complete_binary_model_over_text() -> Result<()> {
        let dir = tempdir()?;
        let text_model = ColmapSparseFiles {
            cameras: vec![ColmapCamera {
                camera_id: 1,
                model_id: COLMAP_PINHOLE,
                width: 10,
                height: 10,
                params: vec![1.0, 1.0, 5.0, 5.0],
            }],
            rigs: Vec::new(),
            frames: Vec::new(),
            images: vec![ColmapImage {
                image_id: 1,
                camera_id: 1,
                name: "text.jpg".to_string(),
                qvec: [1.0, 0.0, 0.0, 0.0],
                tvec: [0.0, 0.0, 0.0],
                points2d: Vec::new(),
            }],
            points3d: Vec::new(),
        };
        let bin_model = ColmapSparseFiles {
            images: vec![ColmapImage {
                image_id: 2,
                name: "binary.jpg".to_string(),
                ..text_model.images[0].clone()
            }],
            ..text_model.clone()
        };
        write_colmap_sparse_text(dir.path(), &text_model)?;
        write_colmap_sparse_binary(dir.path(), &bin_model)?;

        let loaded = read_colmap_sparse_files(dir.path())?;

        assert_eq!(loaded.images.len(), 1);
        assert_eq!(loaded.images[0].image_id, 2);
        assert_eq!(loaded.images[0].name, "binary.jpg");
        Ok(())
    }

    #[test]
    fn writes_camera_using_preserved_colmap_model_and_params() -> Result<()> {
        let dir = tempdir()?;
        let camera = CameraModel::from_colmap(
            COLMAP_SIMPLE_RADIAL,
            800,
            600,
            &[700.0, 401.0, 299.0, -0.0125],
        )
        .unwrap();

        let reconstruction = test_reconstruction_with_cameras(vec![(1, camera)], vec![0]);
        write_cameras_txt(&dir.path().join("cameras.txt"), &reconstruction)?;
        let text = fs::read_to_string(dir.path().join("cameras.txt"))?;

        assert!(text.contains("1 SIMPLE_RADIAL 800 600 700.00000000000000000 401.00000000000000000 299.00000000000000000 -0.01250000000000000"));
        Ok(())
    }

    #[test]
    fn exports_image_camera_ownership_without_forcing_camera_one() -> Result<()> {
        let dir = tempdir()?;
        let camera1 =
            CameraModel::from_colmap(COLMAP_PINHOLE, 640, 480, &[500.0, 501.0, 320.0, 240.0])
                .unwrap();
        let camera2 =
            CameraModel::from_colmap(COLMAP_SIMPLE_RADIAL, 800, 600, &[700.0, 401.0, 299.0, 0.0])
                .unwrap();
        let mut reconstruction =
            test_reconstruction_with_cameras(vec![(11, camera1), (42, camera2)], vec![0, 1]);
        reconstruction.poses = vec![Some(SE3::identity()), Some(SE3::identity())];
        reconstruction.image_ids = vec![7, 8];

        write_cameras_txt(&dir.path().join("cameras.txt"), &reconstruction)?;
        write_images_txt(&dir.path().join("images.txt"), &reconstruction)?;

        let cameras = fs::read_to_string(dir.path().join("cameras.txt"))?;
        let images = read_images_txt(&dir.path().join("images.txt"))?;
        assert!(cameras.contains("11 PINHOLE"));
        assert!(cameras.contains("42 SIMPLE_RADIAL"));
        assert_eq!(images.len(), 2);
        assert_eq!(images[0].image_id, 7);
        assert_eq!(images[0].camera_id, 11);
        assert_eq!(images[0].name, "image_0.jpg");
        assert_eq!(images[1].image_id, 8);
        assert_eq!(images[1].camera_id, 42);
        assert_eq!(images[1].name, "image_1.jpg");
        Ok(())
    }

    #[test]
    fn exports_preserved_non_contiguous_point3d_ids() -> Result<()> {
        let dir = tempdir()?;
        let camera =
            CameraModel::from_colmap(COLMAP_PINHOLE, 640, 480, &[500.0, 501.0, 320.0, 240.0])
                .unwrap();
        let mut reconstruction = test_reconstruction_with_cameras(vec![(1, camera)], vec![0, 0]);
        reconstruction.poses = vec![Some(SE3::identity()), Some(SE3::identity())];
        reconstruction.keypoints = vec![
            vec![test_keypoint(10.0, 20.0)],
            vec![test_keypoint(30.0, 40.0)],
        ];
        reconstruction.observations = vec![vec![Some(0)], vec![Some(0)]];
        reconstruction.point_ids = vec![99];
        reconstruction.points = vec![Point3D {
            xyz: [1.0, 2.0, 3.0],
            color: [4, 5, 6],
            error: 0.25,
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
        }];

        write_images_txt(&dir.path().join("images.txt"), &reconstruction)?;
        write_points3d_txt(&dir.path().join("points3D.txt"), &reconstruction)?;

        let images = read_images_txt(&dir.path().join("images.txt"))?;
        let points = read_points3d_txt(&dir.path().join("points3D.txt"))?;
        assert_eq!(images[0].points2d[0].xy, [10.0, 20.0]);
        assert_eq!(images[0].points2d[0].point3d_id, Some(99));
        assert_eq!(images[1].points2d[0].xy, [30.0, 40.0]);
        assert_eq!(images[1].points2d[0].point3d_id, Some(99));
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].point3d_id, 99);
        assert_eq!(points[0].track[0].image_id, 1);
        assert_eq!(points[0].track[1].image_id, 2);
        Ok(())
    }

    #[test]
    fn reads_text_points3d_with_non_contiguous_ids_and_tracks() -> Result<()> {
        let dir = tempdir()?;
        let sparse = dir.path().join("sparse/0");
        fs::create_dir_all(&sparse)?;
        fs::write(
            sparse.join("points3D.txt"),
            "# points\n99 1.0 2.0 3.0 4 5 6 0.25 7 12 8 13\n150 -1.0 0.5 9.0 9 8 7 1.5\n",
        )?;

        let points = read_colmap_points3d(dir.path())?;

        assert_eq!(points.len(), 2);
        assert_eq!(points[0].point3d_id, 99);
        assert_eq!(points[0].xyz, [1.0, 2.0, 3.0]);
        assert_eq!(points[0].color, [4, 5, 6]);
        assert_eq!(points[0].error, 0.25);
        assert_eq!(
            points[0].track,
            vec![
                ColmapTrackElement {
                    image_id: 7,
                    point2d_idx: 12,
                },
                ColmapTrackElement {
                    image_id: 8,
                    point2d_idx: 13,
                },
            ]
        );
        assert_eq!(points[1].point3d_id, 150);
        assert!(points[1].track.is_empty());
        Ok(())
    }

    #[test]
    fn reads_binary_points3d_with_tracks() -> Result<()> {
        let dir = tempdir()?;
        let sparse = dir.path().join("sparse/0");
        fs::create_dir_all(&sparse)?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&123u64.to_le_bytes());
        for value in [1.5f64, -2.0, 3.25] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&[10, 20, 30]);
        bytes.extend_from_slice(&0.75f64.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&7u32.to_le_bytes());
        bytes.extend_from_slice(&12u32.to_le_bytes());
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&13u32.to_le_bytes());
        fs::write(sparse.join("points3D.bin"), bytes)?;

        let points = read_colmap_points3d(dir.path())?;

        assert_eq!(points.len(), 1);
        assert_eq!(points[0].point3d_id, 123);
        assert_eq!(points[0].xyz, [1.5, -2.0, 3.25]);
        assert_eq!(points[0].color, [10, 20, 30]);
        assert_eq!(points[0].error, 0.75);
        assert_eq!(
            points[0].track,
            vec![
                ColmapTrackElement {
                    image_id: 7,
                    point2d_idx: 12,
                },
                ColmapTrackElement {
                    image_id: 8,
                    point2d_idx: 13,
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn reads_full_text_reconstruction_preserving_ids_tracks_and_cameras() -> Result<()> {
        let dir = tempdir()?;
        let sparse = dir.path().join("sparse/0");
        fs::create_dir_all(&sparse)?;
        fs::write(
            sparse.join("cameras.txt"),
            "# cameras\n11 PINHOLE 640 480 500 501 320 240\n42 SIMPLE_RADIAL 800 600 700 401 299 0.01\n",
        )?;
        fs::write(
            sparse.join("images.txt"),
            concat!(
                "# images\n",
                "7 1 0 0 0 0.1 0.2 0.3 11 left.jpg\n",
                "10 20 99 30 40 -1\n",
                "8 0.9238795325112867 0 0.3826834323650898 0 1 2 3 42 right.jpg\n",
                "15 25 99\n"
            ),
        )?;
        fs::write(
            sparse.join("points3D.txt"),
            "# points\n99 1.5 2.5 3.5 4 5 6 0.125 7 0 8 0\n",
        )?;

        let reconstruction = read_colmap_reconstruction(dir.path())?;

        assert_eq!(reconstruction.camera_ids, vec![11, 42]);
        assert_eq!(reconstruction.image_ids, vec![7, 8]);
        assert_eq!(reconstruction.image_camera_indices, vec![0, 1]);
        assert_eq!(
            reconstruction.image_names,
            vec!["left.jpg".to_string(), "right.jpg".to_string()]
        );
        assert_eq!(reconstruction.point_ids, vec![99]);
        assert_eq!(reconstruction.keypoints[0].len(), 2);
        assert_eq!(reconstruction.keypoints[1].len(), 1);
        assert_eq!(
            reconstruction.observations,
            vec![vec![Some(0), None], vec![Some(0)]]
        );
        assert_eq!(
            reconstruction.points[0].track,
            vec![
                TrackObservation {
                    image: 0,
                    feature: 0,
                },
                TrackObservation {
                    image: 1,
                    feature: 0,
                },
            ]
        );

        let exported = dir.path().join("exported");
        export_colmap(&exported, &reconstruction, false)?;
        let roundtrip = read_colmap_reconstruction(&exported)?;
        assert_eq!(roundtrip.camera_ids, vec![11, 42]);
        assert_eq!(roundtrip.image_ids, vec![7, 8]);
        assert_eq!(roundtrip.point_ids, vec![99]);
        assert_eq!(roundtrip.observations, reconstruction.observations);
        assert_eq!(roundtrip.points[0].track, reconstruction.points[0].track);
        Ok(())
    }

    #[test]
    fn sparse_model_roundtrip_preserves_rigs_and_frames() -> Result<()> {
        let dir = tempdir()?;
        let sparse = dir.path().join("sparse/0");
        fs::create_dir_all(&sparse)?;
        fs::write(
            sparse.join("cameras.txt"),
            "# cameras\n11 PINHOLE 640 480 500 501 320 240\n12 PINHOLE 640 480 500 501 320 240\n",
        )?;
        fs::write(
            sparse.join("images.txt"),
            "# images\n7 1 0 0 0 0 0 0 11 left.jpg\n10 20 -1\n",
        )?;
        fs::write(sparse.join("points3D.txt"), "# points\n")?;
        fs::write(
            sparse.join("rigs.txt"),
            "# rigs\n3 3 CAMERA 11 CAMERA 12 1 1 0 0 0 0.1 0.2 0.3 IMU 5 0\n",
        )?;
        fs::write(
            sparse.join("frames.txt"),
            "# frames\n9 3 1 0 0 0 0.4 0.5 0.6 2 CAMERA 11 7 IMU 5 99\n",
        )?;

        let model = read_colmap_sparse_model(dir.path())?;
        assert_eq!(model.reconstruction.rigs.len(), 1);
        assert_eq!(model.reconstruction.frames.len(), 1);
        assert_eq!(model.reconstruction.image_frame_indices, vec![Some(0)]);
        let exported = dir.path().join("exported_model");
        export_colmap_sparse_model(&exported, &model, false)?;
        let roundtrip = read_colmap_sparse_model(&exported)?;

        assert_eq!(
            roundtrip.reconstruction.camera_ids,
            model.reconstruction.camera_ids
        );
        assert_eq!(
            roundtrip.reconstruction.image_ids,
            model.reconstruction.image_ids
        );
        assert_eq!(roundtrip.rigs, model.rigs);
        assert_eq!(roundtrip.frames, model.frames);

        let exported_from_reconstruction = dir.path().join("exported_from_reconstruction");
        export_colmap(&exported_from_reconstruction, &model.reconstruction, false)?;
        let reconstruction_roundtrip = read_colmap_sparse_model(&exported_from_reconstruction)?;
        assert_eq!(reconstruction_roundtrip.rigs, model.rigs);
        assert_eq!(reconstruction_roundtrip.frames, model.frames);
        Ok(())
    }

    #[test]
    fn reads_text_rigs_with_reference_and_sensor_poses() -> Result<()> {
        let dir = tempdir()?;
        let sparse = dir.path().join("sparse/0");
        fs::create_dir_all(&sparse)?;
        fs::write(
            sparse.join("rigs.txt"),
            "# rigs\n3 3 CAMERA 11 CAMERA 12 1 1 0 0 0 0.1 0.2 0.3 IMU 5 0\n",
        )?;

        let rigs = read_colmap_rigs(dir.path())?;

        assert_eq!(rigs.len(), 1);
        assert_eq!(rigs[0].rig_id, 3);
        assert_eq!(
            rigs[0].ref_sensor_id,
            Some(ColmapSensorId {
                sensor_type: ColmapSensorType::Camera,
                sensor_id: 11,
            })
        );
        assert_eq!(rigs[0].sensors.len(), 2);
        assert_eq!(
            rigs[0].sensors[0].sensor_id,
            ColmapSensorId {
                sensor_type: ColmapSensorType::Camera,
                sensor_id: 12,
            }
        );
        assert_eq!(
            rigs[0].sensors[0].sensor_from_rig,
            Some(ColmapRigid3 {
                qvec: [1.0, 0.0, 0.0, 0.0],
                tvec: [0.1, 0.2, 0.3],
            })
        );
        assert_eq!(rigs[0].sensors[1].sensor_from_rig, None);
        Ok(())
    }

    #[test]
    fn reads_text_frames_with_data_ids() -> Result<()> {
        let dir = tempdir()?;
        let sparse = dir.path().join("sparse/0");
        fs::create_dir_all(&sparse)?;
        fs::write(
            sparse.join("frames.txt"),
            "# frames\n9 3 1 0 0 0 0.4 0.5 0.6 2 CAMERA 11 7 IMU 5 99\n",
        )?;

        let frames = read_colmap_frames(dir.path())?;

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].frame_id, 9);
        assert_eq!(frames[0].rig_id, 3);
        assert_eq!(
            frames[0].rig_from_world,
            ColmapRigid3 {
                qvec: [1.0, 0.0, 0.0, 0.0],
                tvec: [0.4, 0.5, 0.6],
            }
        );
        assert_eq!(
            frames[0].data_ids,
            vec![
                ColmapDataId {
                    sensor_id: ColmapSensorId {
                        sensor_type: ColmapSensorType::Camera,
                        sensor_id: 11,
                    },
                    data_id: 7,
                },
                ColmapDataId {
                    sensor_id: ColmapSensorId {
                        sensor_type: ColmapSensorType::Imu,
                        sensor_id: 5,
                    },
                    data_id: 99,
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn reads_binary_rigs_with_sensor_poses() -> Result<()> {
        let dir = tempdir()?;
        let sparse = dir.path().join("sparse/0");
        fs::create_dir_all(&sparse)?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&11u32.to_le_bytes());
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&12u32.to_le_bytes());
        bytes.push(1);
        for value in [1.0f64, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(sparse.join("rigs.bin"), bytes)?;

        let rigs = read_colmap_rigs(dir.path())?;

        assert_eq!(rigs.len(), 1);
        assert_eq!(rigs[0].rig_id, 3);
        assert_eq!(
            rigs[0].ref_sensor_id,
            Some(ColmapSensorId {
                sensor_type: ColmapSensorType::Camera,
                sensor_id: 11,
            })
        );
        assert_eq!(
            rigs[0].sensors,
            vec![ColmapRigSensor {
                sensor_id: ColmapSensorId {
                    sensor_type: ColmapSensorType::Camera,
                    sensor_id: 12,
                },
                sensor_from_rig: Some(ColmapRigid3 {
                    qvec: [1.0, 0.0, 0.0, 0.0],
                    tvec: [0.1, 0.2, 0.3],
                }),
            }]
        );
        Ok(())
    }

    #[test]
    fn reads_binary_frames_with_u64_data_ids() -> Result<()> {
        let dir = tempdir()?;
        let sparse = dir.path().join("sparse/0");
        fs::create_dir_all(&sparse)?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        for value in [1.0f64, 0.0, 0.0, 0.0, 0.4, 0.5, 0.6] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&11u32.to_le_bytes());
        bytes.extend_from_slice(&4_294_967_299u64.to_le_bytes());
        fs::write(sparse.join("frames.bin"), bytes)?;

        let frames = read_colmap_frames(dir.path())?;

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].frame_id, 9);
        assert_eq!(frames[0].rig_id, 3);
        assert_eq!(
            frames[0].data_ids,
            vec![ColmapDataId {
                sensor_id: ColmapSensorId {
                    sensor_type: ColmapSensorType::Camera,
                    sensor_id: 11,
                },
                data_id: 4_294_967_299,
            }]
        );
        Ok(())
    }

    #[test]
    fn manual_intrinsic_override_updates_exported_params() {
        let mut camera =
            CameraModel::from_colmap(COLMAP_PINHOLE, 640, 480, &[500.0, 501.0, 320.0, 240.0])
                .unwrap();

        camera.set_fx(600.0);
        camera.set_fy(601.0);
        camera.set_cx(321.0);
        camera.set_cy(241.0);

        assert_eq!(camera.params_slice(), &[600.0, 601.0, 321.0, 241.0]);
    }

    fn test_reconstruction_with_cameras(
        cameras: Vec<(u32, CameraModel)>,
        image_camera_indices: Vec<usize>,
    ) -> Reconstruction {
        let (camera_ids, camera_models): (Vec<_>, Vec<_>) = cameras.into_iter().unzip();
        let camera = camera_models[0];
        let image_count = image_camera_indices.len();
        Reconstruction {
            camera,
            cameras: camera_models,
            camera_ids,
            rigs: Vec::new(),
            frames: Vec::new(),
            image_names: (0..image_count)
                .map(|idx| format!("image_{idx}.jpg"))
                .collect(),
            image_paths: (0..image_count)
                .map(|idx| PathBuf::from(format!("image_{idx}.jpg")))
                .collect(),
            image_ids: (0..image_count).map(|idx| idx as u32 + 1).collect(),
            image_camera_indices,
            image_frame_indices: vec![None; image_count],
            poses: vec![None; image_count],
            observations: vec![Vec::new(); image_count],
            keypoints: vec![Vec::new(); image_count],
            point_ids: Vec::new(),
            points: Vec::new(),
        }
    }

    fn test_keypoint(x: f32, y: f32) -> KeyPoint {
        KeyPoint {
            pt: (x, y),
            size: 1.0,
            angle: 0.0,
            response: 1.0,
            octave: 0,
        }
    }

    fn sample_sparse_files() -> ColmapSparseFiles {
        ColmapSparseFiles {
            cameras: vec![
                ColmapCamera {
                    camera_id: 42,
                    model_id: COLMAP_SIMPLE_RADIAL,
                    width: 800,
                    height: 600,
                    params: vec![700.25, 401.5, 299.75, -0.0125],
                },
                ColmapCamera {
                    camera_id: 11,
                    model_id: COLMAP_PINHOLE,
                    width: 640,
                    height: 480,
                    params: vec![500.0, 501.0, 320.0, 240.0],
                },
            ],
            rigs: vec![ColmapRig {
                rig_id: 3,
                ref_sensor_id: Some(ColmapSensorId {
                    sensor_type: ColmapSensorType::Camera,
                    sensor_id: 11,
                }),
                sensors: vec![
                    ColmapRigSensor {
                        sensor_id: ColmapSensorId {
                            sensor_type: ColmapSensorType::Camera,
                            sensor_id: 42,
                        },
                        sensor_from_rig: Some(ColmapRigid3 {
                            qvec: [0.9238795325112867, 0.0, 0.3826834323650898, 0.0],
                            tvec: [0.1, -0.2, 0.3],
                        }),
                    },
                    ColmapRigSensor {
                        sensor_id: ColmapSensorId {
                            sensor_type: ColmapSensorType::Imu,
                            sensor_id: 5,
                        },
                        sensor_from_rig: None,
                    },
                ],
            }],
            frames: vec![ColmapFrame {
                frame_id: 9,
                rig_id: 3,
                rig_from_world: ColmapRigid3 {
                    qvec: [1.0, 0.0, 0.0, 0.0],
                    tvec: [0.4, 0.5, 0.6],
                },
                data_ids: vec![
                    ColmapDataId {
                        sensor_id: ColmapSensorId {
                            sensor_type: ColmapSensorType::Camera,
                            sensor_id: 42,
                        },
                        data_id: 8,
                    },
                    ColmapDataId {
                        sensor_id: ColmapSensorId {
                            sensor_type: ColmapSensorType::Camera,
                            sensor_id: 11,
                        },
                        data_id: 7,
                    },
                    ColmapDataId {
                        sensor_id: ColmapSensorId {
                            sensor_type: ColmapSensorType::Imu,
                            sensor_id: 5,
                        },
                        data_id: 4_294_967_299,
                    },
                ],
            }],
            images: vec![
                ColmapImage {
                    image_id: 8,
                    camera_id: 42,
                    name: "right image.jpg".to_string(),
                    qvec: [0.9238795325112867, 0.0, 0.3826834323650898, 0.0],
                    tvec: [1.0, 2.0, 3.0],
                    points2d: vec![ColmapPoint2D {
                        xy: [15.25, 25.5],
                        point3d_id: Some(99),
                    }],
                },
                ColmapImage {
                    image_id: 7,
                    camera_id: 11,
                    name: "left.jpg".to_string(),
                    qvec: [1.0, 0.0, 0.0, 0.0],
                    tvec: [0.1, 0.2, 0.3],
                    points2d: vec![
                        ColmapPoint2D {
                            xy: [10.5, 20.25],
                            point3d_id: Some(99),
                        },
                        ColmapPoint2D {
                            xy: [30.0, 40.0],
                            point3d_id: None,
                        },
                    ],
                },
            ],
            points3d: vec![
                ColmapPoint3D {
                    point3d_id: 150,
                    xyz: [-1.0, 0.5, 9.0],
                    color: [9, 8, 7],
                    error: 1.5,
                    track: Vec::new(),
                },
                ColmapPoint3D {
                    point3d_id: 99,
                    xyz: [1.5, 2.5, 3.5],
                    color: [4, 5, 6],
                    error: 0.125,
                    track: vec![
                        ColmapTrackElement {
                            image_id: 7,
                            point2d_idx: 0,
                        },
                        ColmapTrackElement {
                            image_id: 8,
                            point2d_idx: 0,
                        },
                    ],
                },
            ],
        }
    }

    fn normalize_sparse_files(mut sparse: ColmapSparseFiles) -> ColmapSparseFiles {
        sparse.cameras.sort_by_key(|camera| camera.camera_id);
        for rig in &mut sparse.rigs {
            rig.sensors
                .sort_by_key(|sensor| sensor_id_sort_key(&sensor.sensor_id));
        }
        sparse.rigs.sort_by_key(|rig| rig.rig_id);
        for frame in &mut sparse.frames {
            frame
                .data_ids
                .sort_by_key(|data_id| (sensor_id_sort_key(&data_id.sensor_id), data_id.data_id));
        }
        sparse.frames.sort_by_key(|frame| frame.frame_id);
        sparse.images.sort_by_key(|image| image.image_id);
        sparse.points3d.sort_by_key(|point| point.point3d_id);
        sparse
    }

    type MalformedImportCase = (
        &'static str,
        fn(&mut ColmapSparseFiles),
        &'static str,
        &'static [&'static str],
    );

    #[test]
    fn strict_text_and_binary_import_reject_the_same_malformed_models() -> Result<()> {
        let cases: Vec<MalformedImportCase> = vec![
            (
                "duplicate-camera",
                duplicate_camera,
                "cameras",
                &["record=camera", "id=1", "duplicate camera id"],
            ),
            (
                "duplicate-image",
                duplicate_image,
                "images",
                &["record=image", "id=1", "duplicate image id"],
            ),
            (
                "duplicate-point",
                duplicate_point,
                "points3D",
                &["record=point", "id=5", "duplicate point id"],
            ),
            (
                "unknown-camera",
                unknown_camera,
                "images",
                &["record=image", "id=1", "referenced=99", "missing camera"],
            ),
            (
                "unknown-image",
                unknown_image,
                "points3D",
                &[
                    "record=point",
                    "id=5",
                    "referenced=9",
                    "feature=0",
                    "missing image",
                ],
            ),
            (
                "unknown-point",
                unknown_point,
                "images",
                &[
                    "record=image",
                    "id=1",
                    "referenced=99",
                    "feature=0",
                    "missing point",
                ],
            ),
            (
                "conflicting-track",
                conflicting_track,
                "images",
                &[
                    "record=image",
                    "id=1",
                    "referenced=5",
                    "feature=0",
                    "does not match",
                ],
            ),
            (
                "zero-quaternion",
                zero_quaternion,
                "images",
                &["record=image", "id=1", "quaternion norm is zero"],
            ),
            (
                "non-finite",
                non_finite_translation,
                "images",
                &["record=image", "id=1", "non-finite"],
            ),
            (
                "feature-index",
                feature_index_overflow,
                "points3D",
                &[
                    "record=point",
                    "id=5",
                    "referenced=1",
                    "feature=50",
                    "feature index is out of range",
                ],
            ),
            (
                "invalid-dimensions",
                invalid_dimensions,
                "cameras",
                &["record=camera", "id=1", "dimensions are illegal"],
            ),
            (
                "invalid-focal",
                invalid_focal,
                "cameras",
                &["record=camera", "id=1", "focal parameter"],
            ),
            (
                "duplicate-rig",
                duplicate_rig,
                "rigs",
                &["record=rig", "id=3", "duplicate rig id"],
            ),
            (
                "duplicate-frame",
                duplicate_frame,
                "frames",
                &["record=frame", "id=9", "duplicate frame id"],
            ),
        ];

        for (name, mutate, file_stem, expected) in cases {
            for format in [ColmapSparseFormat::Text, ColmapSparseFormat::Binary] {
                let dir = tempdir()?;
                let mut sparse = minimal_valid_sparse();
                mutate(&mut sparse);
                write_unvalidated_sparse(dir.path(), &sparse, format)?;
                let error = read_colmap_reconstruction(dir.path()).expect_err(name);
                let message = format!("{error:#}");
                let extension = match format {
                    ColmapSparseFormat::Text => "txt",
                    ColmapSparseFormat::Binary => "bin",
                };
                assert!(
                    message.contains(&format!("{file_stem}.{extension}")),
                    "{name} {format:?} missing source file in {message}"
                );
                for fragment in expected {
                    assert!(
                        message.contains(fragment),
                        "{name} {format:?} missing {fragment} in {message}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn strict_roundtrip_preserves_asymmetric_pose_ids_and_tracks() -> Result<()> {
        let reconstruction = asymmetric_reconstruction();
        let dir = tempdir()?;
        let text_root = dir.path().join("text");
        export_colmap(&text_root, &reconstruction, false)?;
        assert_roundtrip(&reconstruction, &read_colmap_reconstruction(&text_root)?);

        let binary_root = dir.path().join("binary");
        let sparse = sparse_files_from_reconstruction(&reconstruction)?;
        write_colmap_sparse_binary(&binary_root, &sparse)?;
        assert_roundtrip(&reconstruction, &read_colmap_reconstruction(&binary_root)?);
        Ok(())
    }

    #[test]
    fn failed_export_does_not_create_or_truncate_target_files() -> Result<()> {
        let dir = tempdir()?;
        let mut invalid = asymmetric_reconstruction();
        invalid.camera_ids[0] = 0;

        let fresh = dir.path().join("fresh");
        let error = export_colmap(&fresh, &invalid, false).expect_err("invalid export");
        let message = format!("{error:#}");
        assert!(
            message.contains("source=reconstruction-export"),
            "{message}"
        );
        assert!(message.contains("record=reconstruction"), "{message}");
        assert!(!fresh.exists(), "validation created {}", fresh.display());

        let existing = dir.path().join("existing");
        let sparse = existing.join("sparse").join("0");
        fs::create_dir_all(&sparse)?;
        fs::write(sparse.join("cameras.txt"), b"ORIGINAL")?;
        let error = export_colmap_sparse_snapshot(&sparse, &invalid).expect_err("snapshot");
        assert!(format!("{error:#}").contains("source=reconstruction-export"));
        assert_eq!(fs::read(sparse.join("cameras.txt"))?, b"ORIGINAL");
        assert!(!sparse.join("images.txt").exists());

        let raw_dest = dir.path().join("raw-new");
        let mut raw = minimal_valid_sparse();
        raw.cameras.push(raw.cameras[0].clone());
        let error = write_colmap_sparse_model(&raw_dest, &raw, ColmapSparseFormat::Text)
            .expect_err("raw text");
        let message = format!("{error:#}");
        assert!(message.contains("cameras.txt"), "{message}");
        assert!(message.contains("duplicate camera id"), "{message}");
        assert!(!raw_dest.exists());

        let raw_existing = dir.path().join("raw-existing");
        fs::create_dir_all(&raw_existing)?;
        fs::write(raw_existing.join("points3D.bin"), b"ORIGINAL")?;
        let error = write_colmap_sparse_model(&raw_existing, &raw, ColmapSparseFormat::Binary)
            .expect_err("raw binary");
        assert!(format!("{error:#}").contains("cameras.bin"));
        assert_eq!(fs::read(raw_existing.join("points3D.bin"))?, b"ORIGINAL");
        assert!(!raw_existing.join("cameras.bin").exists());
        Ok(())
    }

    fn assert_roundtrip(original: &Reconstruction, loaded: &Reconstruction) {
        assert_eq!(loaded.camera_ids, original.camera_ids);
        assert_eq!(loaded.image_ids, original.image_ids);
        assert_eq!(loaded.image_camera_indices, original.image_camera_indices);
        assert_eq!(loaded.point_ids, original.point_ids);
        assert_eq!(loaded.observations, original.observations);
        assert_eq!(loaded.points[0].track, original.points[0].track);
        for (loaded_pose, original_pose) in loaded.poses.iter().zip(original.poses.iter()) {
            let loaded_pose = loaded_pose.expect("round-trip pose");
            let original_pose = original_pose.expect("original pose");
            assert!(
                rotation_error_deg(&loaded_pose, &original_pose) < 1e-3,
                "rotation drifted"
            );
            assert!(
                translation_error(&loaded_pose, &original_pose) < 1e-4,
                "translation drifted"
            );
        }
    }

    fn asymmetric_reconstruction() -> Reconstruction {
        let camera_a =
            CameraModel::from_colmap(COLMAP_PINHOLE, 640, 480, &[500.0, 510.0, 320.0, 240.0])
                .unwrap();
        let camera_b =
            CameraModel::from_colmap(COLMAP_SIMPLE_RADIAL, 800, 600, &[700.0, 401.0, 299.0, 0.01])
                .unwrap();
        let mut reconstruction =
            test_reconstruction_with_cameras(vec![(11, camera_a), (42, camera_b)], vec![0, 1]);
        reconstruction.image_ids = vec![7, 8];
        let axis = nalgebra::Unit::new_normalize(nalgebra::Vector3::new(0.2, 0.5, 0.8));
        reconstruction.poses = vec![
            Some(SE3::from_quat_translation(
                nalgebra::UnitQuaternion::from_axis_angle(&axis, 0.7),
                nalgebra::Vector3::new(1.25, -0.5, 2.0),
            )),
            Some(SE3::from_quat_translation(
                nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::y_axis(), -0.35),
                nalgebra::Vector3::new(-0.4, 0.8, 1.1),
            )),
        ];
        reconstruction.keypoints = vec![
            vec![test_keypoint(12.0, 24.0), test_keypoint(40.0, 50.0)],
            vec![test_keypoint(18.0, 22.0)],
        ];
        reconstruction.observations = vec![vec![Some(0), None], vec![Some(0)]];
        reconstruction.point_ids = vec![99];
        reconstruction.points = vec![Point3D {
            xyz: [1.0, 2.0, 3.5],
            color: [9, 8, 7],
            error: 0.2,
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
        }];
        reconstruction
    }

    fn minimal_valid_sparse() -> ColmapSparseFiles {
        ColmapSparseFiles {
            cameras: vec![ColmapCamera {
                camera_id: 1,
                model_id: COLMAP_PINHOLE,
                width: 640,
                height: 480,
                params: vec![500.0, 500.0, 320.0, 240.0],
            }],
            rigs: Vec::new(),
            frames: Vec::new(),
            images: vec![ColmapImage {
                image_id: 1,
                camera_id: 1,
                name: "a.jpg".to_string(),
                qvec: [1.0, 0.0, 0.0, 0.0],
                tvec: [0.1, 0.2, 0.3],
                points2d: vec![ColmapPoint2D {
                    xy: [10.0, 20.0],
                    point3d_id: Some(5),
                }],
            }],
            points3d: vec![ColmapPoint3D {
                point3d_id: 5,
                xyz: [1.0, 2.0, 3.0],
                color: [1, 2, 3],
                error: 0.1,
                track: vec![ColmapTrackElement {
                    image_id: 1,
                    point2d_idx: 0,
                }],
            }],
        }
    }

    fn duplicate_camera(sparse: &mut ColmapSparseFiles) {
        sparse.cameras.push(sparse.cameras[0].clone());
    }

    fn duplicate_image(sparse: &mut ColmapSparseFiles) {
        sparse.images.push(sparse.images[0].clone());
    }

    fn duplicate_point(sparse: &mut ColmapSparseFiles) {
        sparse.points3d.push(sparse.points3d[0].clone());
    }

    fn unknown_camera(sparse: &mut ColmapSparseFiles) {
        sparse.images[0].camera_id = 99;
    }

    fn unknown_image(sparse: &mut ColmapSparseFiles) {
        sparse.points3d[0].track[0].image_id = 9;
    }

    fn unknown_point(sparse: &mut ColmapSparseFiles) {
        sparse.images[0].points2d[0].point3d_id = Some(99);
        sparse.points3d[0].track.clear();
    }

    fn conflicting_track(sparse: &mut ColmapSparseFiles) {
        sparse.points3d[0].track.clear();
    }

    fn zero_quaternion(sparse: &mut ColmapSparseFiles) {
        sparse.images[0].qvec = [0.0, 0.0, 0.0, 0.0];
    }

    fn non_finite_translation(sparse: &mut ColmapSparseFiles) {
        sparse.images[0].tvec[0] = f64::NAN;
    }

    fn feature_index_overflow(sparse: &mut ColmapSparseFiles) {
        sparse.points3d[0].track[0].point2d_idx = 50;
    }

    fn invalid_dimensions(sparse: &mut ColmapSparseFiles) {
        sparse.cameras[0].width = 0;
    }

    fn invalid_focal(sparse: &mut ColmapSparseFiles) {
        sparse.cameras[0].params[0] = 0.0;
    }

    fn duplicate_rig(sparse: &mut ColmapSparseFiles) {
        sparse.rigs = vec![camera_rig(3), camera_rig(3)];
    }

    fn duplicate_frame(sparse: &mut ColmapSparseFiles) {
        sparse.rigs = vec![camera_rig(3)];
        sparse.frames = vec![camera_frame(9), camera_frame(9)];
    }

    fn camera_rig(rig_id: u32) -> ColmapRig {
        ColmapRig {
            rig_id,
            ref_sensor_id: Some(ColmapSensorId {
                sensor_type: ColmapSensorType::Camera,
                sensor_id: 1,
            }),
            sensors: Vec::new(),
        }
    }

    fn camera_frame(frame_id: u32) -> ColmapFrame {
        ColmapFrame {
            frame_id,
            rig_id: 3,
            rig_from_world: ColmapRigid3 {
                qvec: [1.0, 0.0, 0.0, 0.0],
                tvec: [0.2, 0.0, 0.0],
            },
            data_ids: vec![ColmapDataId {
                sensor_id: ColmapSensorId {
                    sensor_type: ColmapSensorType::Camera,
                    sensor_id: 1,
                },
                data_id: 1,
            }],
        }
    }

    fn write_unvalidated_sparse(
        dir: &Path,
        sparse: &ColmapSparseFiles,
        format: ColmapSparseFormat,
    ) -> Result<()> {
        fs::create_dir_all(dir)?;
        match format {
            ColmapSparseFormat::Text => {
                write_raw_cameras_txt(&dir.join("cameras.txt"), &sparse.cameras)?;
                write_raw_images_txt(&dir.join("images.txt"), sparse)?;
                write_raw_points3d_txt(&dir.join("points3D.txt"), &sparse.points3d)?;
                if !sparse.rigs.is_empty() {
                    write_raw_rigs_txt(&dir.join("rigs.txt"), &sparse.rigs)?;
                }
                if !sparse.frames.is_empty() {
                    write_raw_frames_txt(&dir.join("frames.txt"), &sparse.frames)?;
                }
            }
            ColmapSparseFormat::Binary => {
                write_raw_cameras_bin(&dir.join("cameras.bin"), &sparse.cameras)?;
                write_raw_images_bin(&dir.join("images.bin"), sparse)?;
                write_raw_points3d_bin(&dir.join("points3D.bin"), &sparse.points3d)?;
                if !sparse.rigs.is_empty() {
                    write_raw_rigs_bin(&dir.join("rigs.bin"), &sparse.rigs)?;
                }
                if !sparse.frames.is_empty() {
                    write_raw_frames_bin(&dir.join("frames.bin"), &sparse.frames)?;
                }
            }
        }
        Ok(())
    }
}
