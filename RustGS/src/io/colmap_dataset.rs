//! COLMAP dataset loader for RustGS.
//!
//! Parses COLMAP format cameras.bin/images.bin/points3D.bin files
//! and converts to TrainingDataset for RustGS pipeline compatibility.
//!
//! Supports both binary and text formats.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use glam::{Mat3, Mat4, Vec3, Vec4};

use crate::{Intrinsics, ScenePose, TrainingDataset, TrainingError, SE3};

/// Configuration for loading a COLMAP dataset.
#[derive(Debug, Clone)]
pub struct ColmapConfig {
    /// Maximum number of frames to load (0 = all).
    pub max_frames: usize,
    /// Keep every Nth frame.
    pub frame_stride: usize,
    /// Depth scale for converting depth values to meters.
    pub depth_scale: f32,
    /// Apply VkSplat/Nerfstudio-style camera and point-cloud normalization.
    pub normalize_world_space: bool,
}

impl Default for ColmapConfig {
    fn default() -> Self {
        Self {
            max_frames: 0,
            frame_stride: 1,
            depth_scale: 1.0,
            normalize_world_space: false,
        }
    }
}

/// COLMAP camera model types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CameraModel {
    Pinhole,
    SimpleRadial,
    Radial,
    OpenCV,
    OpenCVFisheye,
    FullFisheye,
    ThinPrismFisheye,
}

impl CameraModel {
    fn from_id(id: u32) -> Option<Self> {
        match id {
            1 => Some(CameraModel::Pinhole),
            2 => Some(CameraModel::SimpleRadial),
            3 => Some(CameraModel::Radial),
            4 => Some(CameraModel::OpenCV),
            5 => Some(CameraModel::OpenCVFisheye),
            6 => Some(CameraModel::FullFisheye),
            7 => Some(CameraModel::ThinPrismFisheye),
            _ => None,
        }
    }

    /// Number of parameters for this camera model.
    fn num_params(&self) -> usize {
        match self {
            CameraModel::Pinhole => 4,       // fx, fy, cx, cy
            CameraModel::SimpleRadial => 4,  // f, cx, cy, k1
            CameraModel::Radial => 5,        // f, cx, cy, k1, k2
            CameraModel::OpenCV => 8,        // fx, fy, cx, cy, k1, k2, p1, p2
            CameraModel::OpenCVFisheye => 8, // fx, fy, cx, cy, k1, k2, k3, k4
            CameraModel::FullFisheye => 12,
            CameraModel::ThinPrismFisheye => 15,
        }
    }

    /// Extract intrinsics from camera parameters.
    fn intrinsics(&self, width: u32, height: u32, params: &[f64]) -> Option<Intrinsics> {
        match self {
            CameraModel::Pinhole => {
                if params.len() >= 4 {
                    Some(Intrinsics::new(
                        params[0] as f32, // fx
                        params[1] as f32, // fy
                        params[2] as f32, // cx
                        params[3] as f32, // cy
                        width,
                        height,
                    ))
                } else {
                    None
                }
            }
            CameraModel::SimpleRadial => {
                if params.len() >= 3 {
                    Some(Intrinsics::new(
                        params[0] as f32, // f
                        params[0] as f32, // f (same for both)
                        params[1] as f32, // cx
                        params[2] as f32, // cy
                        width,
                        height,
                    ))
                } else {
                    None
                }
            }
            CameraModel::Radial => {
                if params.len() >= 3 {
                    Some(Intrinsics::new(
                        params[0] as f32, // f
                        params[0] as f32, // f (same for both)
                        params[1] as f32, // cx
                        params[2] as f32, // cy
                        width,
                        height,
                    ))
                } else {
                    None
                }
            }
            CameraModel::OpenCV => {
                if params.len() >= 4 {
                    Some(Intrinsics::new(
                        params[0] as f32, // fx
                        params[1] as f32, // fy
                        params[2] as f32, // cx
                        params[3] as f32, // cy
                        width,
                        height,
                    ))
                } else {
                    None
                }
            }
            _ => None, // Unsupported models, fall back to default
        }
    }
}

/// COLMAP camera entry.
#[derive(Debug, Clone)]
struct ColmapCamera {
    model: CameraModel,
    width: u32,
    height: u32,
    params: Vec<f64>,
}

/// COLMAP image entry.
#[derive(Debug, Clone)]
struct ColmapImage {
    image_id: u32,
    qw: f64,
    qx: f64,
    qy: f64,
    qz: f64,
    tx: f64,
    ty: f64,
    tz: f64,
    name: String,
}

/// COLMAP 3D point entry.
#[derive(Debug, Clone)]
struct ColmapPoint3D {
    x: f64,
    y: f64,
    z: f64,
    r: u8,
    g: u8,
    b: u8,
}

/// Load a COLMAP dataset from a sparse reconstruction directory.
///
/// The directory should contain:
/// - cameras.bin (or cameras.txt)
/// - images.bin (or images.txt)
/// - points3D.bin (or points3D.txt)
/// - images/ directory with actual image files
///
/// # Arguments
/// * `input` - Path to COLMAP sparse reconstruction directory (e.g., "sparse/0")
/// * `config` - Loading configuration
///
/// # Returns
/// * `TrainingDataset` ready for RustGS training
pub fn load_colmap_dataset(
    input: &Path,
    config: &ColmapConfig,
) -> Result<TrainingDataset, TrainingError> {
    // Resolve the sparse directory
    let sparse_dir = resolve_colmap_sparse_dir(input)?;

    // Determine image directory
    let image_dir = resolve_image_dir(&sparse_dir)?;

    // Parse cameras
    let cameras = parse_colmap_cameras(&sparse_dir)?;

    // Parse images
    let images = parse_colmap_images(&sparse_dir)?;

    // Parse 3D points required for sparse-point initialization
    let points = parse_colmap_points3d(&sparse_dir)?;

    if cameras.is_empty() {
        return Err(TrainingError::InvalidInput(
            "no cameras found in COLMAP dataset".to_string(),
        ));
    }
    if images.is_empty() {
        return Err(TrainingError::InvalidInput(
            "no images found in COLMAP dataset".to_string(),
        ));
    }
    if points.is_empty() {
        return Err(TrainingError::InvalidInput(format!(
            "no sparse points found in {} (expected points3D.bin or points3D.txt with at least one point)",
            sparse_dir.display(),
        )));
    }

    // Use first camera for intrinsics (COLMAP datasets typically have one camera)
    let first_camera = &cameras[0];
    let intrinsics = first_camera
        .model
        .intrinsics(
            first_camera.width,
            first_camera.height,
            &first_camera.params,
        )
        .ok_or_else(|| {
            TrainingError::InvalidInput(format!(
                "unsupported camera model {:?} or missing parameters",
                first_camera.model
            ))
        })?;

    let mut poses: Vec<SE3> = images.iter().map(scene_pose_from_colmap_image).collect();
    let mut point_positions: Vec<Vec3> = points
        .iter()
        .map(|point| Vec3::new(point.x as f32, point.y as f32, point.z as f32))
        .collect();

    if config.normalize_world_space {
        normalize_world_space(&mut poses, &mut point_positions);
    }

    // Build dataset
    let mut dataset = TrainingDataset::new(intrinsics).with_depth_scale(config.depth_scale);
    // Add initial points from COLMAP sparse reconstruction
    for (point, position) in points.iter().zip(point_positions.iter()) {
        dataset.add_point(
            [position.x, position.y, position.z],
            Some([
                point.r as f32 / 255.0,
                point.g as f32 / 255.0,
                point.b as f32 / 255.0,
            ]),
        );
    }

    // Apply frame selection
    let considered = if config.max_frames > 0 {
        config.max_frames.min(images.len())
    } else {
        images.len()
    };
    let stride = config.frame_stride.max(1);
    let mut missing_image_count = 0usize;
    let mut missing_image_examples = Vec::new();

    // Add poses
    for (frame_idx, (image, pose)) in images
        .iter()
        .zip(poses.iter())
        .take(considered)
        .step_by(stride)
        .enumerate()
    {
        let image_path = image_dir.join(&image.name);
        if !image_path.exists() {
            missing_image_count += 1;
            if missing_image_examples.len() < 5 {
                missing_image_examples.push(image_path.display().to_string());
            }
            continue;
        }

        let scene_pose = ScenePose::new(frame_idx as u64, image_path, *pose, image.image_id as f64);
        dataset.add_pose(scene_pose);
    }

    if dataset.poses.is_empty() {
        return Err(TrainingError::InvalidInput(format!(
            "no valid frames found in {} after image path validation",
            sparse_dir.display(),
        )));
    }
    if missing_image_count > 0 {
        log::warn!(
            "COLMAP dataset {} skipped {} frames because image files were missing (showing up to 5): {}",
            sparse_dir.display(),
            missing_image_count,
            missing_image_examples.join(", ")
        );
    }

    log::info!(
        "Loaded COLMAP dataset {} | cameras={} | images_total={} | frames={} | missing_images={} | points={} | resolution={}x{}",
        sparse_dir.display(),
        cameras.len(),
        considered,
        dataset.poses.len(),
        missing_image_count,
        dataset.initial_points.len(),
        intrinsics.width,
        intrinsics.height,
    );

    Ok(dataset)
}

pub(crate) fn resolve_colmap_sparse_dir(input: &Path) -> Result<PathBuf, TrainingError> {
    // Check if input is directly a sparse directory
    if is_colmap_sparse_dir(input) {
        return Ok(input.to_path_buf());
    }

    // Check for sparse subdirectory
    let sparse = input.join("sparse");
    if is_colmap_sparse_dir(&sparse) {
        return Ok(sparse);
    }

    // Check for sparse/0 (common COLMAP output structure)
    let sparse0 = sparse.join("0");
    if is_colmap_sparse_dir(&sparse0) {
        return Ok(sparse0);
    }

    Err(TrainingError::InvalidInput(format!(
        "could not find COLMAP sparse reconstruction in {}",
        input.display(),
    )))
}

pub(crate) fn is_colmap_sparse_dir(path: &Path) -> bool {
    path.is_dir()
        && (path.join("cameras.bin").exists() || path.join("cameras.txt").exists())
        && (path.join("images.bin").exists() || path.join("images.txt").exists())
}

fn scene_pose_from_colmap_image(image: &ColmapImage) -> SE3 {
    let world_to_camera = SE3::new(
        &[
            image.qx as f32,
            image.qy as f32,
            image.qz as f32,
            image.qw as f32,
        ],
        &[image.tx as f32, image.ty as f32, image.tz as f32],
    );
    world_to_camera.inverse()
}

fn normalize_world_space(poses: &mut [SE3], points: &mut [Vec3]) {
    if poses.is_empty() || points.is_empty() {
        return;
    }

    let t1 = similarity_from_cameras(poses);
    transform_poses(t1, poses);
    transform_points(t1, points);

    let t2 = align_principal_axes(points);
    transform_poses(t2, poses);
    transform_points(t2, points);

    let z_median = median(points.iter().map(|point| point.z).collect());
    let z_mean = points.iter().map(|point| point.z).sum::<f32>() / points.len() as f32;
    if z_median > z_mean {
        let flip = Mat4::from_cols(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, -1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, -1.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        );
        transform_poses(flip, poses);
        transform_points(flip, points);
    }
}

fn transform_poses(transform: Mat4, poses: &mut [SE3]) {
    for pose in poses {
        let transformed = transform * se3_to_mat4(*pose);
        let scale = transformed.x_axis.truncate().length();
        let inv_scale = if scale > 1e-12 { scale.recip() } else { 1.0 };
        let rotation = Mat3::from_cols(
            transformed.x_axis.truncate() * inv_scale,
            transformed.y_axis.truncate() * inv_scale,
            transformed.z_axis.truncate() * inv_scale,
        );
        let translation = transformed.w_axis.truncate();
        *pose = SE3::from_quat_translation(glam::Quat::from_mat3(&rotation), translation);
    }
}

fn transform_points(transform: Mat4, points: &mut [Vec3]) {
    let linear = Mat3::from_cols(
        transform.x_axis.truncate(),
        transform.y_axis.truncate(),
        transform.z_axis.truncate(),
    );
    let translation = transform.w_axis.truncate();
    for point in points {
        *point = linear * *point + translation;
    }
}

fn se3_to_mat4(pose: SE3) -> Mat4 {
    Mat4::from_rotation_translation(pose.quat(), pose.vec())
}

fn similarity_from_cameras(poses: &[SE3]) -> Mat4 {
    let mut positions = Vec::with_capacity(poses.len());
    let mut ups = Vec::with_capacity(poses.len());
    let mut forwards = Vec::with_capacity(poses.len());

    for pose in poses {
        let rotation = Mat3::from_quat(pose.quat());
        positions.push(pose.vec());
        ups.push(rotation * Vec3::new(0.0, -1.0, 0.0));
        forwards.push(rotation * Vec3::new(0.0, 0.0, 1.0));
    }

    let mut world_up = Vec3::ZERO;
    for up in &ups {
        world_up += *up;
    }
    world_up /= ups.len() as f32;
    world_up = world_up
        .try_normalize()
        .unwrap_or(Vec3::new(0.0, -1.0, 0.0));

    let up_camspace = Vec3::new(0.0, -1.0, 0.0);
    let c = up_camspace.dot(world_up);
    let cross = world_up.cross(up_camspace);
    let align = if c > -1.0 {
        let skew = Mat3::from_cols(
            Vec3::new(0.0, -cross.z, cross.y),
            Vec3::new(cross.z, 0.0, -cross.x),
            Vec3::new(-cross.y, cross.x, 0.0),
        );
        Mat3::IDENTITY + skew + (skew * skew) * (1.0 / (1.0 + c))
    } else {
        Mat3::from_cols(
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        )
    };

    for position in &mut positions {
        *position = align * *position;
    }
    for forward in &mut forwards {
        *forward = align * *forward;
    }

    let nearest_points: Vec<Vec3> = positions
        .iter()
        .zip(forwards.iter())
        .map(|(position, forward)| *position + (-*position).dot(*forward) * *forward)
        .collect();
    let translate = -median_vec3(&nearest_points);

    let distances: Vec<f32> = positions
        .iter()
        .map(|position| (*position + translate).length())
        .collect();
    let median_distance = median(distances);
    let scale = if median_distance > 1e-12 {
        median_distance.recip()
    } else {
        1.0
    };

    Mat4::from_cols(
        (align.x_axis * scale).extend(0.0),
        (align.y_axis * scale).extend(0.0),
        (align.z_axis * scale).extend(0.0),
        (translate * scale).extend(1.0),
    )
}

fn align_principal_axes(points: &[Vec3]) -> Mat4 {
    if points.is_empty() {
        return Mat4::IDENTITY;
    }

    let centroid = median_vec3(points);
    let mut covariance = [[0.0_f64; 3]; 3];
    let mut mean = [0.0_f64; 3];

    for point in points {
        let translated = [
            (point.x - centroid.x) as f64,
            (point.y - centroid.y) as f64,
            (point.z - centroid.z) as f64,
        ];
        for i in 0..3 {
            mean[i] += translated[i];
            for j in 0..3 {
                covariance[i][j] += translated[i] * translated[j];
            }
        }
    }

    let denom = points.len().saturating_sub(1).max(1) as f64;
    for i in 0..3 {
        mean[i] /= points.len() as f64;
        for j in 0..3 {
            covariance[i][j] /= denom;
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            covariance[i][j] -= mean[i] * mean[j];
        }
    }

    let (eigenvectors, eigenvalues) = jacobi_eigen_3x3(covariance);
    let mut order = [0usize, 1, 2];
    order.sort_by(|&lhs, &rhs| {
        eigenvalues[rhs]
            .partial_cmp(&eigenvalues[lhs])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut eigen_cols = [
        vec3_from_f64_col(eigenvectors, order[0]),
        vec3_from_f64_col(eigenvectors, order[1]),
        vec3_from_f64_col(eigenvectors, order[2]),
    ];
    if eigen_cols[0].cross(eigen_cols[1]).dot(eigen_cols[2]) < 0.0 {
        eigen_cols[0] = -eigen_cols[0];
    }

    let eigen = Mat3::from_cols(eigen_cols[0], eigen_cols[1], eigen_cols[2]);
    let rotation = eigen.transpose();
    let translation = -(rotation * centroid);
    Mat4::from_cols(
        rotation.x_axis.extend(0.0),
        rotation.y_axis.extend(0.0),
        rotation.z_axis.extend(0.0),
        translation.extend(1.0),
    )
}

fn jacobi_eigen_3x3(mut a: [[f64; 3]; 3]) -> ([[f64; 3]; 3], [f32; 3]) {
    let mut eigenvectors = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        eigenvectors[i][i] = 1.0;
    }

    for _ in 0..12 {
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max_off_diag = a[0][1].abs();
        for i in 0..3 {
            for j in (i + 1)..3 {
                let value = a[i][j].abs();
                if value > max_off_diag {
                    max_off_diag = value;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off_diag < 1e-8 {
            break;
        }

        let theta = 0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p]);
        let c = theta.cos();
        let s = theta.sin();

        let mut g = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            g[i][i] = 1.0;
        }
        g[p][p] = c;
        g[p][q] = -s;
        g[q][p] = s;
        g[q][q] = c;

        a = mat3_mul(mat3_mul(mat3_transpose(g), a), g);
        eigenvectors = mat3_mul(eigenvectors, g);
    }

    (
        eigenvectors,
        [a[0][0] as f32, a[1][1] as f32, a[2][2] as f32],
    )
}

fn vec3_from_f64_col(matrix: [[f64; 3]; 3], col: usize) -> Vec3 {
    Vec3::new(
        matrix[0][col] as f32,
        matrix[1][col] as f32,
        matrix[2][col] as f32,
    )
    .normalize_or_zero()
}

fn mat3_transpose(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [matrix[0][0], matrix[1][0], matrix[2][0]],
        [matrix[0][1], matrix[1][1], matrix[2][1]],
        [matrix[0][2], matrix[1][2], matrix[2][2]],
    ]
}

fn mat3_mul(lhs: [[f64; 3]; 3], rhs: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = lhs[i][0] * rhs[0][j] + lhs[i][1] * rhs[1][j] + lhs[i][2] * rhs[2][j];
        }
    }
    out
}

fn median_vec3(values: &[Vec3]) -> Vec3 {
    if values.is_empty() {
        return Vec3::ZERO;
    }
    Vec3::new(
        median(values.iter().map(|value| value.x).collect()),
        median(values.iter().map(|value| value.y).collect()),
        median(values.iter().map(|value| value.z).collect()),
    )
}

fn median(mut values: Vec<f32>) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n.is_multiple_of(2) {
        (values[n / 2 - 1] + values[n / 2]) * 0.5
    } else {
        values[n / 2]
    }
}

fn resolve_image_dir(sparse_dir: &Path) -> Result<PathBuf, TrainingError> {
    // Try common image directory locations
    let candidates = [
        sparse_dir.join("images"),
        sparse_dir.parent().unwrap().join("images"),
        sparse_dir
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("images"),
    ];

    for candidate in &candidates {
        if candidate.is_dir() {
            return Ok(candidate.clone());
        }
    }

    Err(TrainingError::InvalidInput(
        "could not find images directory".to_string(),
    ))
}

// Binary parsing helpers
fn read_u64<T: Read>(reader: &mut T) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_u32<T: Read>(reader: &mut T) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u8<T: Read>(reader: &mut T) -> std::io::Result<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_f64<T: Read>(reader: &mut T) -> std::io::Result<f64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_null_terminated_string<T: Read>(reader: &mut T) -> std::io::Result<String> {
    let mut buf = Vec::new();
    loop {
        let byte = read_u8(reader)?;
        if byte == 0 {
            break;
        }
        buf.push(byte);
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn parse_colmap_cameras(dir: &Path) -> Result<Vec<ColmapCamera>, TrainingError> {
    let bin_path = dir.join("cameras.bin");
    let txt_path = dir.join("cameras.txt");

    if bin_path.exists() {
        parse_cameras_binary(&bin_path)
    } else if txt_path.exists() {
        parse_cameras_text(&txt_path)
    } else {
        Err(TrainingError::InvalidInput(format!(
            "no cameras file found in {}",
            dir.display(),
        )))
    }
}

fn parse_cameras_binary(path: &Path) -> Result<Vec<ColmapCamera>, TrainingError> {
    let mut file = File::open(path)?;
    let num_cameras = read_u64(&mut file)? as usize;

    let mut cameras = Vec::with_capacity(num_cameras);
    for _ in 0..num_cameras {
        read_u32(&mut file)?;
        let model_id = read_u32(&mut file)?;
        let width = read_u64(&mut file)? as u32;
        let height = read_u64(&mut file)? as u32;

        let model = CameraModel::from_id(model_id).ok_or_else(|| {
            TrainingError::InvalidInput(format!("unknown camera model ID {}", model_id))
        })?;

        let num_params = model.num_params();
        let params = (0..num_params)
            .map(|_| read_f64(&mut file))
            .collect::<std::io::Result<Vec<_>>>()?;

        cameras.push(ColmapCamera {
            model,
            width,
            height,
            params,
        });
    }

    Ok(cameras)
}

fn parse_cameras_text(path: &Path) -> Result<Vec<ColmapCamera>, TrainingError> {
    let reader = BufReader::new(File::open(path)?);
    let mut cameras = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() < 4 {
            continue;
        }

        parse_u32(path, line_num, "camera_id", parts[0])?;
        let model_name = parts[1];
        let width = parse_u32(path, line_num, "width", parts[2])?;
        let height = parse_u32(path, line_num, "height", parts[3])?;

        let model = parse_camera_model_name(model_name)?;
        let params: Vec<f64> = parts[4..]
            .iter()
            .enumerate()
            .map(|(i, p)| parse_f64(path, line_num, &format!("param[{}]", i), p))
            .collect::<Result<Vec<_>, _>>()?;

        cameras.push(ColmapCamera {
            model,
            width,
            height,
            params,
        });
    }

    Ok(cameras)
}

fn parse_u32(path: &Path, line_num: usize, field: &str, value: &str) -> Result<u32, TrainingError> {
    value.parse::<u32>().map_err(|err| {
        TrainingError::InvalidInput(format!(
            "{} line {}: invalid {} '{}': {}",
            path.display(),
            line_num + 1,
            field,
            value,
            err
        ))
    })
}

fn parse_f64(path: &Path, line_num: usize, field: &str, value: &str) -> Result<f64, TrainingError> {
    value.parse::<f64>().map_err(|err| {
        TrainingError::InvalidInput(format!(
            "{} line {}: invalid {} '{}': {}",
            path.display(),
            line_num + 1,
            field,
            value,
            err
        ))
    })
}

fn parse_camera_model_name(name: &str) -> Result<CameraModel, TrainingError> {
    match name.to_uppercase().as_str() {
        "PINHOLE" => Ok(CameraModel::Pinhole),
        "SIMPLE_RADIAL" => Ok(CameraModel::SimpleRadial),
        "RADIAL" => Ok(CameraModel::Radial),
        "OPENCV" => Ok(CameraModel::OpenCV),
        "OPENCV_FISHEYE" => Ok(CameraModel::OpenCVFisheye),
        "FULL_FISHEYE" => Ok(CameraModel::FullFisheye),
        "THIN_PRISM_FISHEYE" => Ok(CameraModel::ThinPrismFisheye),
        _ => Err(TrainingError::InvalidInput(format!(
            "unknown camera model '{}'",
            name
        ))),
    }
}

fn parse_colmap_images(dir: &Path) -> Result<Vec<ColmapImage>, TrainingError> {
    let bin_path = dir.join("images.bin");
    let txt_path = dir.join("images.txt");

    if bin_path.exists() {
        parse_images_binary(&bin_path)
    } else if txt_path.exists() {
        parse_images_text(&txt_path)
    } else {
        Err(TrainingError::InvalidInput(format!(
            "no images file found in {}",
            dir.display(),
        )))
    }
}

fn parse_images_binary(path: &Path) -> Result<Vec<ColmapImage>, TrainingError> {
    let mut file = File::open(path)?;
    let num_images = read_u64(&mut file)? as usize;

    let mut images = Vec::with_capacity(num_images);
    for _ in 0..num_images {
        let image_id = read_u32(&mut file)?;
        let qw = read_f64(&mut file)?;
        let qx = read_f64(&mut file)?;
        let qy = read_f64(&mut file)?;
        let qz = read_f64(&mut file)?;
        let tx = read_f64(&mut file)?;
        let ty = read_f64(&mut file)?;
        let tz = read_f64(&mut file)?;
        read_u32(&mut file)?;
        let name = read_null_terminated_string(&mut file)?;

        // Skip 2D point observations (we don't need them for dataset loading)
        let num_points2d = read_u64(&mut file)?;
        for _ in 0..num_points2d {
            // x, y, point3D_id
            read_f64(&mut file)?;
            read_f64(&mut file)?;
            read_u64(&mut file)?;
        }

        images.push(ColmapImage {
            image_id,
            qw,
            qx,
            qy,
            qz,
            tx,
            ty,
            tz,
            name,
        });
    }

    Ok(images)
}

fn parse_images_text(path: &Path) -> Result<Vec<ColmapImage>, TrainingError> {
    let reader = BufReader::new(File::open(path)?);
    let mut images = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Image line format: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        // Points line follows (we skip it)
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() < 10 {
            continue;
        }

        // Skip lines that don't start with a valid image ID
        if parts[0].parse::<u32>().is_err() {
            continue;
        }

        let image_id = parse_u32(path, line_num, "image_id", parts[0])?;
        let qw = parse_f64(path, line_num, "qw", parts[1])?;
        let qx = parse_f64(path, line_num, "qx", parts[2])?;
        let qy = parse_f64(path, line_num, "qy", parts[3])?;
        let qz = parse_f64(path, line_num, "qz", parts[4])?;
        let tx = parse_f64(path, line_num, "tx", parts[5])?;
        let ty = parse_f64(path, line_num, "ty", parts[6])?;
        let tz = parse_f64(path, line_num, "tz", parts[7])?;
        parse_u32(path, line_num, "camera_id", parts[8])?;
        let name = parts[9].to_string();

        images.push(ColmapImage {
            image_id,
            qw,
            qx,
            qy,
            qz,
            tx,
            ty,
            tz,
            name,
        });
    }

    Ok(images)
}

fn parse_colmap_points3d(dir: &Path) -> Result<Vec<ColmapPoint3D>, TrainingError> {
    let bin_path = dir.join("points3D.bin");
    let txt_path = dir.join("points3D.txt");

    if bin_path.exists() {
        parse_points3d_binary(&bin_path)
    } else if txt_path.exists() {
        parse_points3d_text(&txt_path)
    } else {
        Err(TrainingError::InvalidInput(format!(
            "missing COLMAP sparse points in {} (expected points3D.bin or points3D.txt)",
            dir.display(),
        )))
    }
}

fn parse_points3d_binary(path: &Path) -> Result<Vec<ColmapPoint3D>, TrainingError> {
    let mut file = File::open(path)?;
    let num_points = read_u64(&mut file)? as usize;

    let mut points = Vec::with_capacity(num_points);
    for _ in 0..num_points {
        read_u64(&mut file)?;
        let x = read_f64(&mut file)?;
        let y = read_f64(&mut file)?;
        let z = read_f64(&mut file)?;
        let r = read_u8(&mut file)?;
        let g = read_u8(&mut file)?;
        let b = read_u8(&mut file)?;

        // Skip error and track
        read_f64(&mut file)?; // error
        let track_len = read_u64(&mut file)?;
        for _ in 0..track_len {
            read_u32(&mut file)?; // image_id
            read_u32(&mut file)?; // point2d_idx
        }

        points.push(ColmapPoint3D { x, y, z, r, g, b });
    }

    Ok(points)
}

fn parse_points3d_text(path: &Path) -> Result<Vec<ColmapPoint3D>, TrainingError> {
    let reader = BufReader::new(File::open(path)?);
    let mut points = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Point line format: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() < 8 {
            continue;
        }

        parse_u64(path, line_num, "point_id", parts[0])?;
        let x = parse_f64(path, line_num, "x", parts[1])?;
        let y = parse_f64(path, line_num, "y", parts[2])?;
        let z = parse_f64(path, line_num, "z", parts[3])?;
        let r = parse_u8(path, line_num, "r", parts[4])?;
        let g = parse_u8(path, line_num, "g", parts[5])?;
        let b = parse_u8(path, line_num, "b", parts[6])?;

        points.push(ColmapPoint3D { x, y, z, r, g, b });
    }

    Ok(points)
}

fn parse_u64(path: &Path, line_num: usize, field: &str, value: &str) -> Result<u64, TrainingError> {
    value.parse::<u64>().map_err(|err| {
        TrainingError::InvalidInput(format!(
            "{} line {}: invalid {} '{}': {}",
            path.display(),
            line_num + 1,
            field,
            value,
            err
        ))
    })
}

fn parse_u8(path: &Path, line_num: usize, field: &str, value: &str) -> Result<u8, TrainingError> {
    value.parse::<u8>().map_err(|err| {
        TrainingError::InvalidInput(format!(
            "{} line {}: invalid {} '{}': {}",
            path.display(),
            line_num + 1,
            field,
            value,
            err
        ))
    })
}
