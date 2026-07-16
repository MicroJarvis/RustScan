use crate::types::{
    colmap_camera_model_id, colmap_camera_model_name, colmap_camera_model_num_params, CameraModel,
    DataId, Frame, Point3D, Reconstruction, Rig, RigSensor, Rigid3, SensorId, SensorType,
    TrackObservation,
};
use anyhow::{bail, Context, Result};
use nalgebra::{Matrix3, Quaternion, UnitQuaternion, Vector3};
use rustslam::SE3;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, Write};
use std::path::{Path, PathBuf};

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

pub fn read_colmap_reconstruction(root: &Path) -> Result<Reconstruction> {
    let sparse = read_colmap_sparse_files(root)?;
    reconstruction_from_colmap_files(&sparse)
}

pub fn read_colmap_sparse_model(root: &Path) -> Result<ColmapSparseModel> {
    let sparse = read_colmap_sparse_files(root)?;
    let mut reconstruction = reconstruction_from_colmap_files(&sparse)?;
    apply_rig_frame_metadata_to_reconstruction(&mut reconstruction, &sparse.rigs, &sparse.frames);
    Ok(ColmapSparseModel {
        reconstruction,
        rigs: sparse.rigs,
        frames: sparse.frames,
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

fn reconstruction_from_colmap_parts(
    colmap_cameras: Vec<(u32, CameraModel)>,
    images: Vec<ColmapImage>,
    points3d: Vec<ColmapPoint3D>,
) -> Result<Reconstruction> {
    if colmap_cameras.is_empty() {
        bail!("COLMAP reconstruction has no cameras");
    }
    let (camera_ids, cameras): (Vec<_>, Vec<_>) = colmap_cameras.into_iter().unzip();
    let camera_index_by_id = camera_ids
        .iter()
        .enumerate()
        .map(|(idx, &camera_id)| (camera_id, idx))
        .collect::<HashMap<_, _>>();
    let image_index_by_id = images
        .iter()
        .enumerate()
        .map(|(idx, image)| (image.image_id, idx))
        .collect::<HashMap<_, _>>();

    let image_camera_indices = images
        .iter()
        .map(|image| {
            camera_index_by_id
                .get(&image.camera_id)
                .copied()
                .with_context(|| {
                    format!(
                        "image_id={} references missing camera_id={}",
                        image.image_id, image.camera_id
                    )
                })
        })
        .collect::<Result<Vec<_>>>()?;
    let (rigs, frames, image_frame_indices) = Reconstruction::empty_metadata(images.len());
    let keypoints = images
        .iter()
        .map(|image| {
            image
                .points2d
                .iter()
                .map(|point| keypoint_from_colmap_point2d(point))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut observations = keypoints
        .iter()
        .map(|points| vec![None; points.len()])
        .collect::<Vec<_>>();
    let point_index_by_id = points3d
        .iter()
        .enumerate()
        .map(|(idx, point)| (point.point3d_id, idx))
        .collect::<BTreeMap<_, _>>();

    for (image_idx, image) in images.iter().enumerate() {
        for (point2d_idx, point2d) in image.points2d.iter().enumerate() {
            let Some(point3d_id) = point2d.point3d_id else {
                continue;
            };
            let Some(&point_idx) = point_index_by_id.get(&point3d_id) else {
                continue;
            };
            observations[image_idx][point2d_idx] = Some(point_idx);
        }
    }

    let mut points = points3d
        .iter()
        .map(|point| {
            let track = point
                .track
                .iter()
                .filter_map(|elem| {
                    let image = *image_index_by_id.get(&elem.image_id)?;
                    let feature = elem.point2d_idx as usize;
                    keypoints
                        .get(image)
                        .filter(|points| feature < points.len())
                        .map(|_| TrackObservation { image, feature })
                })
                .collect::<Vec<_>>();
            Point3D {
                xyz: [
                    point.xyz[0] as f32,
                    point.xyz[1] as f32,
                    point.xyz[2] as f32,
                ],
                color: point.color,
                error: point.error as f32,
                track,
            }
        })
        .collect::<Vec<_>>();
    ensure_observations_have_point_tracks(&observations, &mut points);
    ensure_point_tracks_have_observations(&mut observations, &points);

    let poses = images
        .iter()
        .map(|image| Some(se3_from_colmap_pose(image.qvec, image.tvec)))
        .collect::<Vec<_>>();
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

fn reconstruction_from_colmap_files(sparse: &ColmapSparseFiles) -> Result<Reconstruction> {
    let cameras = sparse
        .cameras
        .iter()
        .cloned()
        .map(|camera| {
            let camera_id = camera.camera_id;
            Ok((camera_id, camera_model_from_colmap(camera)?))
        })
        .collect::<Result<Vec<_>>>()?;
    reconstruction_from_colmap_parts(cameras, sparse.images.clone(), sparse.points3d.clone())
}

fn apply_rig_frame_metadata_to_reconstruction(
    reconstruction: &mut Reconstruction,
    rigs: &[ColmapRig],
    frames: &[ColmapFrame],
) {
    reconstruction.rigs = rigs.iter().map(rig_from_colmap).collect();
    reconstruction.frames = frames.iter().map(frame_from_colmap).collect();
    let frame_index_by_camera_data_id = frames
        .iter()
        .enumerate()
        .flat_map(|(frame_idx, frame)| {
            frame
                .data_ids
                .iter()
                .filter(|data_id| data_id.sensor_id.sensor_type == ColmapSensorType::Camera)
                .map(move |data_id| (data_id.data_id as u32, frame_idx))
        })
        .collect::<HashMap<_, _>>();
    reconstruction.image_frame_indices = reconstruction
        .image_ids
        .iter()
        .map(|image_id| frame_index_by_camera_data_id.get(image_id).copied())
        .collect();
}

fn rig_from_colmap(rig: &ColmapRig) -> Rig {
    Rig {
        rig_id: rig.rig_id,
        ref_sensor_id: rig.ref_sensor_id.as_ref().map(sensor_id_from_colmap),
        sensors: rig.sensors.iter().map(rig_sensor_from_colmap).collect(),
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

fn keypoint_from_colmap_point2d(point: &ColmapPoint2D) -> rustslam::KeyPoint {
    rustslam::KeyPoint {
        pt: (point.xy[0] as f32, point.xy[1] as f32),
        size: 1.0,
        angle: 0.0,
        response: 1.0,
        octave: 0,
    }
}

fn se3_from_colmap_pose(qvec: [f64; 4], tvec: [f64; 3]) -> SE3 {
    let rotation = glam::Quat::from_xyzw(
        qvec[1] as f32,
        qvec[2] as f32,
        qvec[3] as f32,
        qvec[0] as f32,
    )
    .normalize();
    SE3::from_quat_translation(
        rotation,
        glam::Vec3::new(tvec[0] as f32, tvec[1] as f32, tvec[2] as f32),
    )
}

fn ensure_point_tracks_have_observations(
    observations: &mut [Vec<Option<usize>>],
    points: &[Point3D],
) {
    for (point_idx, point) in points.iter().enumerate() {
        for obs in &point.track {
            if let Some(slot) = observations
                .get_mut(obs.image)
                .and_then(|image_obs| image_obs.get_mut(obs.feature))
            {
                if slot.is_none() {
                    *slot = Some(point_idx);
                }
            }
        }
    }
}

fn ensure_observations_have_point_tracks(
    observations: &[Vec<Option<usize>>],
    points: &mut [Point3D],
) {
    for (image, image_observations) in observations.iter().enumerate() {
        for (feature, point_idx) in image_observations.iter().enumerate() {
            let Some(point_idx) = point_idx else {
                continue;
            };
            let Some(point) = points.get_mut(*point_idx) else {
                continue;
            };
            if !point
                .track
                .iter()
                .any(|obs| obs.image == image && obs.feature == feature)
            {
                point.track.push(TrackObservation { image, feature });
            }
        }
    }
}

pub fn export_colmap(
    root: &Path,
    reconstruction: &Reconstruction,
    copy_images: bool,
) -> Result<()> {
    export_colmap_with_sparse_index(root, reconstruction, copy_images, 0)
}

pub fn export_colmap_with_sparse_index(
    root: &Path,
    reconstruction: &Reconstruction,
    copy_images: bool,
    sparse_index: usize,
) -> Result<()> {
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
    write_cameras_txt(&sparse_dir.join("cameras.txt"), reconstruction)?;
    write_images_txt(&sparse_dir.join("images.txt"), reconstruction)?;
    write_points3d_txt(&sparse_dir.join("points3D.txt"), reconstruction)?;
    if !reconstruction.rigs.is_empty() || !reconstruction.frames.is_empty() {
        let rigs = reconstruction
            .rigs
            .iter()
            .map(rig_to_colmap)
            .collect::<Vec<_>>();
        let frames = reconstruction
            .frames
            .iter()
            .map(frame_to_colmap)
            .collect::<Vec<_>>();
        write_rigs_txt(&sparse_dir.join("rigs.txt"), &rigs)?;
        write_frames_txt(&sparse_dir.join("frames.txt"), &frames)?;
    }
    Ok(())
}

pub fn export_colmap_sparse_snapshot(root: &Path, reconstruction: &Reconstruction) -> Result<()> {
    fs::create_dir_all(root)?;
    write_cameras_txt(&root.join("cameras.txt"), reconstruction)?;
    write_images_txt(&root.join("images.txt"), reconstruction)?;
    write_points3d_txt(&root.join("points3D.txt"), reconstruction)?;
    if !reconstruction.rigs.is_empty() || !reconstruction.frames.is_empty() {
        let rigs = reconstruction
            .rigs
            .iter()
            .map(rig_to_colmap)
            .collect::<Vec<_>>();
        let frames = reconstruction
            .frames
            .iter()
            .map(frame_to_colmap)
            .collect::<Vec<_>>();
        write_rigs_txt(&root.join("rigs.txt"), &rigs)?;
        write_frames_txt(&root.join("frames.txt"), &frames)?;
    }
    Ok(())
}

pub fn export_colmap_sparse_model(
    root: &Path,
    model: &ColmapSparseModel,
    copy_images: bool,
) -> Result<()> {
    export_colmap(root, &model.reconstruction, copy_images)?;
    let sparse_dir = root.join("sparse").join("0");
    write_rigs_txt(&sparse_dir.join("rigs.txt"), &model.rigs)?;
    write_frames_txt(&sparse_dir.join("frames.txt"), &model.frames)?;
    Ok(())
}

pub fn write_colmap_sparse_text(root: &Path, sparse: &ColmapSparseFiles) -> Result<()> {
    fs::create_dir_all(root)?;
    write_raw_rigs_txt(&root.join("rigs.txt"), &sparse.rigs)?;
    write_raw_cameras_txt(&root.join("cameras.txt"), &sparse.cameras)?;
    write_raw_frames_txt(&root.join("frames.txt"), &sparse.frames)?;
    write_raw_images_txt(&root.join("images.txt"), sparse)?;
    write_raw_points3d_txt(&root.join("points3D.txt"), &sparse.points3d)?;
    Ok(())
}

pub fn write_colmap_sparse_binary(root: &Path, sparse: &ColmapSparseFiles) -> Result<()> {
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

fn write_cameras_txt(path: &Path, reconstruction: &Reconstruction) -> Result<()> {
    let cameras = cameras_from_reconstruction(reconstruction)?;
    write_raw_cameras_txt(path, &cameras)
}

fn write_images_txt(path: &Path, reconstruction: &Reconstruction) -> Result<()> {
    let sparse = sparse_files_from_reconstruction(reconstruction)?;
    write_raw_images_txt(path, &sparse)
}

fn write_points3d_txt(path: &Path, reconstruction: &Reconstruction) -> Result<()> {
    let points = points3d_from_reconstruction(reconstruction);
    write_raw_points3d_txt(path, &points)
}

fn write_rigs_txt(path: &Path, rigs: &[ColmapRig]) -> Result<()> {
    write_raw_rigs_txt(path, rigs)
}

fn write_frames_txt(path: &Path, frames: &[ColmapFrame]) -> Result<()> {
    write_raw_frames_txt(path, frames)
}

fn cameras_from_reconstruction(reconstruction: &Reconstruction) -> Result<Vec<ColmapCamera>> {
    let cameras = if reconstruction.cameras.is_empty() {
        vec![reconstruction.camera]
    } else {
        reconstruction.cameras.clone()
    };
    cameras
        .iter()
        .enumerate()
        .map(|(idx, camera)| {
            let camera_id = reconstruction
                .camera_ids
                .get(idx)
                .copied()
                .unwrap_or_else(|| idx as u32 + 1);
            Ok(ColmapCamera {
                camera_id,
                model_id: camera.model_id,
                width: camera.width,
                height: camera.height,
                params: camera.params_slice().to_vec(),
            })
        })
        .collect()
}

fn images_from_reconstruction(reconstruction: &Reconstruction) -> Vec<ColmapImage> {
    reconstruction
        .image_names
        .iter()
        .enumerate()
        .filter_map(|(idx, name)| {
            let pose = reconstruction.poses.get(idx).copied().flatten()?;
            let q = pose.quaternion();
            let t = pose.translation();
            let points2d = reconstruction
                .keypoints
                .get(idx)
                .into_iter()
                .flatten()
                .enumerate()
                .map(|(feature_idx, kp)| {
                    let point3d_id = reconstruction
                        .observations
                        .get(idx)
                        .and_then(|obs| obs.get(feature_idx))
                        .copied()
                        .flatten()
                        .map(|id| reconstruction.point3d_id(id));
                    ColmapPoint2D {
                        xy: [kp.x() as f64, kp.y() as f64],
                        point3d_id,
                    }
                })
                .collect();
            Some(ColmapImage {
                image_id: reconstruction.image_id(idx),
                camera_id: reconstruction.camera_id_for_image(idx),
                name: name.clone(),
                qvec: [q[3] as f64, q[0] as f64, q[1] as f64, q[2] as f64],
                tvec: [t[0] as f64, t[1] as f64, t[2] as f64],
                points2d,
            })
        })
        .collect()
}

fn points3d_from_reconstruction(reconstruction: &Reconstruction) -> Vec<ColmapPoint3D> {
    reconstruction
        .points
        .iter()
        .enumerate()
        .map(|(idx, p)| ColmapPoint3D {
            point3d_id: reconstruction.point3d_id(idx),
            xyz: [p.xyz[0] as f64, p.xyz[1] as f64, p.xyz[2] as f64],
            color: p.color,
            error: p.error as f64,
            track: p
                .track
                .iter()
                .map(|TrackObservation { image, feature }| ColmapTrackElement {
                    image_id: reconstruction.image_id(*image),
                    point2d_idx: *feature as u64,
                })
                .collect(),
        })
        .collect()
}

fn sparse_files_from_reconstruction(reconstruction: &Reconstruction) -> Result<ColmapSparseFiles> {
    Ok(ColmapSparseFiles {
        cameras: cameras_from_reconstruction(reconstruction)?,
        rigs: reconstruction.rigs.iter().map(rig_to_colmap).collect(),
        frames: reconstruction.frames.iter().map(frame_to_colmap).collect(),
        images: images_from_reconstruction(reconstruction),
        points3d: points3d_from_reconstruction(reconstruction),
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

fn ordered_images_for_write<'a>(sparse: &'a ColmapSparseFiles) -> Vec<&'a ColmapImage> {
    let image_by_id = sparse
        .images
        .iter()
        .map(|image| (image.image_id, image))
        .collect::<HashMap<_, _>>();
    let mut ordered = Vec::new();
    let mut seen = BTreeMap::<u32, ()>::new();
    for frame in sorted_frames(&sparse.frames) {
        for data_id in sorted_data_ids(&frame.data_ids) {
            if data_id.sensor_id.sensor_type != ColmapSensorType::Camera {
                continue;
            }
            let Ok(image_id) = u32::try_from(data_id.data_id) else {
                continue;
            };
            if let Some(image) = image_by_id.get(&image_id) {
                ordered.push(*image);
                seen.insert(image_id, ());
            }
        }
    }
    let mut rest = sparse
        .images
        .iter()
        .filter(|image| !seen.contains_key(&image.image_id))
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

fn rig_to_colmap(rig: &Rig) -> ColmapRig {
    ColmapRig {
        rig_id: rig.rig_id,
        ref_sensor_id: rig.ref_sensor_id.as_ref().map(sensor_id_to_colmap),
        sensors: rig.sensors.iter().map(rig_sensor_to_colmap).collect(),
    }
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
    use rustslam::{tracker::PnPProblem, tracker::PnPSolver, KeyPoint, SE3};
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
    fn real_colmap_sparse_tracks_recover_registered_image_pose_with_pnp() -> Result<()> {
        let sparse = read_colmap_sparse_files_with_format(
            Path::new("../test_data/flowers2_colmap/sparse/text"),
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

        let reference = se3_from_colmap_pose(image.qvec, image.tvec);
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
            "# cameras\n11 PINHOLE 640 480 500 501 320 240\n",
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
}
