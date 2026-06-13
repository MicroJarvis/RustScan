use crate::types::{
    colmap_camera_model_id, colmap_camera_model_num_params, CameraModel, Reconstruction,
    TrackObservation,
};
use anyhow::{bail, Context, Result};
use nalgebra::{Matrix3, Quaternion, UnitQuaternion, Vector3};
use rustslam::SE3;
use serde::{Deserialize, Serialize};
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

fn read_images_txt(path: &Path) -> Result<Vec<ColmapPose>> {
    let reader = BufReader::new(File::open(path)?);
    let mut poses = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts: Vec<_> = trimmed.split_whitespace().collect();
        if parts.len() < 10 || parts[0].parse::<u32>().is_err() {
            continue;
        }
        poses.push(ColmapPose {
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
        });
    }
    Ok(poses)
}

fn read_images_bin(path: &Path) -> Result<Vec<ColmapPose>> {
    let mut f = File::open(path)?;
    let n = read_u64(&mut f)? as usize;
    let mut poses = Vec::with_capacity(n);
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
        for _ in 0..m {
            read_f64(&mut f)?;
            read_f64(&mut f)?;
            read_u64(&mut f)?;
        }
        poses.push(ColmapPose {
            image_id,
            camera_id,
            name,
            qvec,
            tvec,
        });
    }
    Ok(poses)
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
