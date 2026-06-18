use crate::types::{
    colmap_camera_model_id, colmap_camera_model_num_params, CameraModel, Point3D, Reconstruction,
    TrackObservation,
};
use anyhow::{bail, Context, Result};
use nalgebra::{Matrix3, Quaternion, UnitQuaternion, Vector3};
use rustslam::SE3;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
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

pub fn read_colmap_reconstruction(root: &Path) -> Result<Reconstruction> {
    let colmap_cameras = read_colmap_cameras(root)?;
    let images = read_colmap_images(root)?;
    let points3d = read_optional_colmap_points3d(root)?;
    reconstruction_from_colmap_parts(colmap_cameras, images, points3d)
}

fn read_optional_colmap_points3d(root: &Path) -> Result<Vec<ColmapPoint3D>> {
    let sparse = resolve_sparse_dir(root)?;
    let bin = sparse.join("points3D.bin");
    if bin.exists() {
        return read_points3d_bin(&bin);
    }
    let txt = sparse.join("points3D.txt");
    if txt.exists() {
        return read_points3d_txt(&txt);
    }
    Ok(Vec::new())
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
        image_names,
        image_paths,
        image_ids,
        image_camera_indices,
        poses,
        observations,
        keypoints,
        point_ids,
        points,
    })
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
    let images_dir = root.join("images");
    let sparse_dir = root.join("sparse").join("0");
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
    Ok(())
}

fn write_cameras_txt(path: &Path, reconstruction: &Reconstruction) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "# Camera list with one line of data per camera:")?;
    writeln!(w, "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]")?;
    let cameras = if reconstruction.cameras.is_empty() {
        vec![reconstruction.camera]
    } else {
        reconstruction.cameras.clone()
    };
    for (idx, camera) in cameras.iter().enumerate() {
        let camera_id = reconstruction
            .camera_ids
            .get(idx)
            .copied()
            .unwrap_or_else(|| idx as u32 + 1);
        writeln!(
            w,
            "{} {} {} {} {}",
            camera_id,
            camera.model_name(),
            camera.width,
            camera.height,
            camera
                .params_slice()
                .iter()
                .map(|p| format!("{p:.17}"))
                .collect::<Vec<_>>()
                .join(" ")
        )?;
    }
    Ok(())
}

fn write_images_txt(path: &Path, reconstruction: &Reconstruction) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "# Image list with two lines of data per image:")?;
    writeln!(w, "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME")?;
    writeln!(w, "# POINTS2D[] as (X, Y, POINT3D_ID)")?;
    for (idx, name) in reconstruction.image_names.iter().enumerate() {
        let Some(pose) = reconstruction.poses[idx] else {
            continue;
        };
        let q = pose.quaternion();
        let t = pose.translation();
        writeln!(
            w,
            "{} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {} {}",
            reconstruction.image_id(idx),
            q[3],
            q[0],
            q[1],
            q[2],
            t[0],
            t[1],
            t[2],
            reconstruction.camera_id_for_image(idx),
            name
        )?;
        for (feature_idx, kp) in reconstruction.keypoints[idx].iter().enumerate() {
            if feature_idx > 0 {
                write!(w, " ")?;
            }
            let point_id = reconstruction.observations[idx][feature_idx]
                .map(|id| reconstruction.point3d_id(id).to_string())
                .unwrap_or_else(|| "-1".to_string());
            write!(w, "{:.6} {:.6} {}", kp.x(), kp.y(), point_id)?;
        }
        writeln!(w)?;
    }
    Ok(())
}

fn write_points3d_txt(path: &Path, reconstruction: &Reconstruction) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "# 3D point list with one line of data per point:")?;
    writeln!(w, "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]")?;
    for (idx, p) in reconstruction.points.iter().enumerate() {
        write!(
            w,
            "{} {:.9} {:.9} {:.9} {} {} {} {:.6}",
            reconstruction.point3d_id(idx),
            p.xyz[0],
            p.xyz[1],
            p.xyz[2],
            p.color[0],
            p.color[1],
            p.color[2],
            p.error
        )?;
        for TrackObservation { image, feature } in &p.track {
            write!(w, " {} {}", reconstruction.image_id(*image), feature)?;
        }
        writeln!(w)?;
    }
    Ok(())
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
        let parts: Vec<_> = trimmed.split_whitespace().collect();
        if parts.len() < 10 || parts[0].parse::<u32>().is_err() {
            continue;
        }
        let points_line = lines
            .next()
            .transpose()?
            .with_context(|| format!("missing points2D line after image in {}", path.display()))?;
        images.push(ColmapImage {
            image_id: parts[0].parse()?,
            camera_id: parts[8].parse()?,
            qvec: [
                parts[1].parse()?,
                parts[2].parse()?,
                parts[3].parse()?,
                parts[4].parse()?,
            ],
            tvec: [parts[5].parse()?, parts[6].parse()?, parts[7].parse()?],
            name: parts[9].to_string(),
            points2d: parse_points2d_txt(&points_line, path)?,
        });
    }
    Ok(images)
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
    if signed < 0 {
        Ok(None)
    } else {
        Ok(Some(signed as u64))
    }
}

fn read_images_bin(path: &Path) -> Result<Vec<ColmapImage>> {
    let mut f = File::open(path)?;
    let n = read_u64(&mut f)? as usize;
    let mut images = Vec::with_capacity(n);
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
        let m = read_u64(&mut f)? as usize;
        let mut points2d = Vec::with_capacity(m);
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
            continue;
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
    let n = read_u64(&mut f)?;
    if n == 0 {
        bail!("empty camera file {}", path.display());
    }
    let mut cameras = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let camera_id = read_u32(&mut f)?;
        let model_id = read_i32(&mut f)?;
        let width = read_u64(&mut f)? as u32;
        let height = read_u64(&mut f)? as u32;
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
            continue;
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
    let n = read_u64(&mut f)? as usize;
    let mut points = Vec::with_capacity(n);
    for _ in 0..n {
        let point3d_id = read_u64(&mut f)?;
        let xyz = [read_f64(&mut f)?, read_f64(&mut f)?, read_f64(&mut f)?];
        let color = [read_u8(&mut f)?, read_u8(&mut f)?, read_u8(&mut f)?];
        let error = read_f64(&mut f)?;
        let track_length = read_u64(&mut f)? as usize;
        let mut track = Vec::with_capacity(track_length);
        for _ in 0..track_length {
            track.push(ColmapTrackElement {
                image_id: read_u32(&mut f)?,
                point2d_idx: read_u64(&mut f)?,
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
            continue;
        }
        let rig_id = parts[0].parse()?;
        let num_sensors = parts[1].parse::<usize>()?;
        let mut cursor = 2usize;
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
            continue;
        }
        let mut cursor = 0usize;
        let frame_id = parse_next(&parts, &mut cursor, path)?;
        let rig_id = parse_next(&parts, &mut cursor, path)?;
        let rig_from_world = parse_rigid3(&parts, &mut cursor, path)?;
        let num_data_ids = parse_next::<usize>(&parts, &mut cursor, path)?;
        let mut data_ids = Vec::with_capacity(num_data_ids);
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
    let n = read_u64(&mut f)? as usize;
    let mut rigs = Vec::with_capacity(n);
    for _ in 0..n {
        let rig_id = read_u32(&mut f)?;
        let num_sensors = read_u32(&mut f)? as usize;
        let ref_sensor_id = if num_sensors > 0 {
            Some(read_sensor_id_bin(&mut f)?)
        } else {
            None
        };
        let mut sensors = Vec::with_capacity(num_sensors.saturating_sub(1));
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
    let n = read_u64(&mut f)? as usize;
    let mut frames = Vec::with_capacity(n);
    for _ in 0..n {
        let frame_id = read_u32(&mut f)?;
        let rig_id = read_u32(&mut f)?;
        let rig_from_world = read_rigid3_bin(&mut f)?;
        let num_data_ids = read_u32(&mut f)? as usize;
        let mut data_ids = Vec::with_capacity(num_data_ids);
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
    use rustslam::{KeyPoint, SE3};
    use tempfile::tempdir;

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
        let images = fs::read_to_string(dir.path().join("images.txt"))?;
        assert!(cameras.contains("11 PINHOLE"));
        assert!(cameras.contains("42 SIMPLE_RADIAL"));
        assert!(images.contains("7 1.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 11 image_0.jpg"));
        assert!(images.contains("8 1.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 42 image_1.jpg"));
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

        let images = fs::read_to_string(dir.path().join("images.txt"))?;
        let points = fs::read_to_string(dir.path().join("points3D.txt"))?;
        assert!(images.contains("10.000000 20.000000 99"));
        assert!(images.contains("30.000000 40.000000 99"));
        assert!(
            points.contains("\n99 1.000000000 2.000000000 3.000000000 4 5 6 0.250000 1 0 2 0\n")
        );
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
        bytes.extend_from_slice(&12u64.to_le_bytes());
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&13u64.to_le_bytes());
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
            image_names: (0..image_count)
                .map(|idx| format!("image_{idx}.jpg"))
                .collect(),
            image_paths: (0..image_count)
                .map(|idx| PathBuf::from(format!("image_{idx}.jpg")))
                .collect(),
            image_ids: (0..image_count).map(|idx| idx as u32 + 1).collect(),
            image_camera_indices,
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
}
