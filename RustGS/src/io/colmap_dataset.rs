//! COLMAP dataset loader for RustGS.
//!
//! Parses COLMAP format cameras.bin/images.bin/points3D.bin files
//! and converts to TrainingDataset for RustGS pipeline compatibility.
//!
//! Supports both binary and text formats.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek};
use std::path::{Path, PathBuf};

use glam::{Mat3, Mat4, Vec3, Vec4};
use rustscan_types::colmap::{
    colmap_camera_model_by_id, colmap_camera_model_by_name, ColmapCameraModelSpec, COLMAP_PINHOLE,
    COLMAP_SIMPLE_PINHOLE,
};

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
    /// Explicit root containing images referenced by the sparse model.
    pub image_root: Option<PathBuf>,
}

impl Default for ColmapConfig {
    fn default() -> Self {
        Self {
            max_frames: 0,
            frame_stride: 1,
            depth_scale: 1.0,
            normalize_world_space: false,
            image_root: None,
        }
    }
}

/// COLMAP camera entry.
#[derive(Debug, Clone)]
struct ColmapCamera {
    camera_id: u32,
    model: &'static ColmapCameraModelSpec,
    width: u32,
    height: u32,
    params: Vec<f64>,
}

/// COLMAP image entry.
#[derive(Debug, Clone)]
struct ColmapImage {
    image_id: u32,
    camera_id: u32,
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
    let image_dir = resolve_image_dir(&sparse_dir, config.image_root.as_deref())?;

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

    let cameras_by_id = cameras
        .iter()
        .map(|camera| (camera.camera_id, camera))
        .collect::<BTreeMap<_, _>>();
    let referenced_camera_ids = images
        .iter()
        .map(|image| image.camera_id)
        .collect::<BTreeSet<_>>();
    if referenced_camera_ids.len() != 1 {
        return Err(TrainingError::InvalidInput(format!(
            "RustGS currently requires one shared COLMAP CAMERA_ID, but images reference {:?}",
            referenced_camera_ids
        )));
    }
    let camera_id = *referenced_camera_ids.iter().next().expect("one camera id");
    let camera = cameras_by_id.get(&camera_id).ok_or_else(|| {
        TrainingError::InvalidInput(format!(
            "COLMAP images reference missing CAMERA_ID {camera_id}"
        ))
    })?;
    let intrinsics = pinhole_intrinsics(camera)?;

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
        validate_image_name(&image.name)?;
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

/// Resolve the COLMAP sparse model directory used when loading `input`.
///
/// Accepts a sparse model directory directly, a dataset root containing
/// `sparse`, or the common dataset layout containing `sparse/0`.
pub fn resolve_colmap_sparse_dir(input: &Path) -> Result<PathBuf, TrainingError> {
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

/// Fingerprint the exact COLMAP sparse model files selected by the loader.
///
/// The fingerprint covers cameras, images, and points3D. For each category,
/// the binary file is selected when present; otherwise the text file is used.
/// Other files and subdirectories are ignored. File contents are streamed into
/// the digest so large point clouds do not need to be buffered in memory.
pub fn fingerprint_colmap_sparse_model(input: &Path) -> Result<[u8; 32], TrainingError> {
    let sparse_dir = resolve_colmap_sparse_dir(input)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"RustGS COLMAP sparse model fingerprint\0v1");

    for stem in ["cameras", "images", "points3D"] {
        let (path, encoding) = select_colmap_model_file(&sparse_dir, stem).ok_or_else(|| {
            TrainingError::InvalidInput(format!(
                "missing COLMAP {stem} model in {}",
                sparse_dir.display()
            ))
        })?;
        let label = format!("{stem}.{}", encoding.extension());
        update_fingerprint_label(&mut hasher, label.as_bytes());
        update_fingerprint_file(&mut hasher, &path)?;
    }

    Ok(*hasher.finalize().as_bytes())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColmapModelEncoding {
    Binary,
    Text,
}

impl ColmapModelEncoding {
    fn extension(self) -> &'static str {
        match self {
            Self::Binary => "bin",
            Self::Text => "txt",
        }
    }
}

fn select_colmap_model_file(
    directory: &Path,
    stem: &str,
) -> Option<(PathBuf, ColmapModelEncoding)> {
    let binary = directory.join(format!("{stem}.bin"));
    if binary.exists() {
        return Some((binary, ColmapModelEncoding::Binary));
    }
    let text = directory.join(format!("{stem}.txt"));
    text.exists().then_some((text, ColmapModelEncoding::Text))
}

fn update_fingerprint_label(hasher: &mut blake3::Hasher, label: &[u8]) {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
}

fn update_fingerprint_file(hasher: &mut blake3::Hasher, path: &Path) -> Result<(), TrainingError> {
    let mut file = File::open(path)?;
    let expected_len = file.metadata()?.len();
    hasher.update(&expected_len.to_le_bytes());

    let mut actual_len = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        actual_len = actual_len.checked_add(read as u64).ok_or_else(|| {
            TrainingError::InvalidInput(format!(
                "COLMAP model file length overflow in {}",
                path.display()
            ))
        })?;
        hasher.update(&buffer[..read]);
    }
    if actual_len != expected_len {
        return Err(TrainingError::InvalidInput(format!(
            "COLMAP model file {} changed while fingerprinting",
            path.display()
        )));
    }
    Ok(())
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

fn pinhole_intrinsics(camera: &ColmapCamera) -> Result<Intrinsics, TrainingError> {
    if camera.width == 0 || camera.height == 0 {
        return Err(TrainingError::InvalidInput(format!(
            "COLMAP camera {} has zero dimensions",
            camera.camera_id
        )));
    }
    if camera.model.has_distortion() {
        return Err(TrainingError::InvalidInput(format!(
            "COLMAP camera {} uses distorted model {}; undistort images and export PINHOLE or SIMPLE_PINHOLE before training",
            camera.camera_id, camera.model.name
        )));
    }
    let values = match camera.model.id {
        COLMAP_SIMPLE_PINHOLE => [
            camera.params[0],
            camera.params[0],
            camera.params[1],
            camera.params[2],
        ],
        COLMAP_PINHOLE => [
            camera.params[0],
            camera.params[1],
            camera.params[2],
            camera.params[3],
        ],
        _ => {
            return Err(TrainingError::InvalidInput(format!(
                "COLMAP camera model {} is not supported by the pinhole RustGS rasterizer",
                camera.model.name
            )))
        }
    };
    if !values.iter().all(|value| value.is_finite())
        || values[0] <= 0.0
        || values[1] <= 0.0
        || values.iter().any(|value| value.abs() > f64::from(f32::MAX))
    {
        return Err(TrainingError::InvalidInput(format!(
            "COLMAP camera {} has invalid pinhole parameters",
            camera.camera_id
        )));
    }
    Ok(Intrinsics::new(
        values[0] as f32,
        values[1] as f32,
        values[2] as f32,
        values[3] as f32,
        camera.width,
        camera.height,
    ))
}

fn validate_image_name(name: &str) -> Result<(), TrainingError> {
    let path = Path::new(name);
    if name.is_empty()
        || name.contains('\\')
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(TrainingError::InvalidInput(format!(
            "unsafe COLMAP image name '{name}': expected a relative path without parent traversal"
        )));
    }
    Ok(())
}

fn image_name_from_header(line: &str) -> Option<String> {
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
    None
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
    for (mean_value, covariance_row) in mean.iter_mut().zip(covariance.iter_mut()) {
        *mean_value /= points.len() as f64;
        for covariance_value in covariance_row {
            *covariance_value /= denom;
        }
    }
    for (i, covariance_row) in covariance.iter_mut().enumerate() {
        for (j, covariance_value) in covariance_row.iter_mut().enumerate() {
            *covariance_value -= mean[i] * mean[j];
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
    for (i, row) in eigenvectors.iter_mut().enumerate() {
        row[i] = 1.0;
    }

    for _ in 0..12 {
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max_off_diag = a[0][1].abs();
        for (i, row) in a.iter().enumerate() {
            for (j, entry) in row.iter().enumerate().skip(i + 1) {
                let value = entry.abs();
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
        for (i, row) in g.iter_mut().enumerate() {
            row[i] = 1.0;
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

fn resolve_image_dir(
    sparse_dir: &Path,
    explicit_image_root: Option<&Path>,
) -> Result<PathBuf, TrainingError> {
    if let Some(image_root) = explicit_image_root {
        if image_root.is_dir() {
            return Ok(image_root.to_path_buf());
        }
        return Err(TrainingError::InvalidInput(format!(
            "explicit COLMAP image root is not a directory: {}",
            image_root.display()
        )));
    }

    let mut candidates = vec![sparse_dir.join("images")];
    candidates.extend(
        sparse_dir
            .ancestors()
            .skip(1)
            .take(2)
            .map(|ancestor| ancestor.join("images")),
    );
    for candidate in candidates {
        if candidate.is_dir() {
            return Ok(candidate);
        }
    }

    Err(TrainingError::InvalidInput(format!(
        "could not find images directory for {}; pass an explicit image_root",
        sparse_dir.display()
    )))
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

fn read_bounded_count(
    file: &mut File,
    path: &Path,
    label: &str,
    minimum_item_bytes: u64,
) -> Result<usize, TrainingError> {
    let raw_count = read_u64(file)?;
    let count = usize::try_from(raw_count).map_err(|_| {
        TrainingError::InvalidInput(format!(
            "{label} count {raw_count} does not fit usize in {}",
            path.display()
        ))
    })?;
    let remaining = file
        .metadata()?
        .len()
        .saturating_sub(file.stream_position()?);
    let maximum_count = remaining / minimum_item_bytes.max(1);
    if raw_count > maximum_count {
        return Err(TrainingError::InvalidInput(format!(
            "{label} count {raw_count} exceeds the maximum {maximum_count} allowed by the remaining bytes in {}",
            path.display()
        )));
    }
    Ok(count)
}

fn parse_colmap_cameras(dir: &Path) -> Result<Vec<ColmapCamera>, TrainingError> {
    match select_colmap_model_file(dir, "cameras") {
        Some((path, ColmapModelEncoding::Binary)) => parse_cameras_binary(&path),
        Some((path, ColmapModelEncoding::Text)) => parse_cameras_text(&path),
        None => Err(TrainingError::InvalidInput(format!(
            "no cameras file found in {}",
            dir.display(),
        ))),
    }
}

fn parse_cameras_binary(path: &Path) -> Result<Vec<ColmapCamera>, TrainingError> {
    let mut file = File::open(path)?;
    let num_cameras = read_bounded_count(&mut file, path, "camera", 24)?;

    let mut cameras = Vec::with_capacity(num_cameras);
    for _ in 0..num_cameras {
        let camera_id = read_u32(&mut file)?;
        let model_id = i32::try_from(read_u32(&mut file)?).map_err(|_| {
            TrainingError::InvalidInput(format!(
                "camera model ID exceeds i32 in {}",
                path.display()
            ))
        })?;
        let width = u32::try_from(read_u64(&mut file)?).map_err(|_| {
            TrainingError::InvalidInput(format!("camera width exceeds u32 in {}", path.display()))
        })?;
        let height = u32::try_from(read_u64(&mut file)?).map_err(|_| {
            TrainingError::InvalidInput(format!("camera height exceeds u32 in {}", path.display()))
        })?;

        let model = colmap_camera_model_by_id(model_id).ok_or_else(|| {
            TrainingError::InvalidInput(format!("unknown camera model ID {}", model_id))
        })?;

        let params = (0..model.num_params)
            .map(|_| read_f64(&mut file))
            .collect::<std::io::Result<Vec<_>>>()?;

        cameras.push(ColmapCamera {
            camera_id,
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
            return Err(TrainingError::InvalidInput(format!(
                "{} line {}: invalid camera record",
                path.display(),
                line_num + 1
            )));
        }

        let camera_id = parse_u32(path, line_num, "camera_id", parts[0])?;
        let model_name = parts[1];
        let width = parse_u32(path, line_num, "width", parts[2])?;
        let height = parse_u32(path, line_num, "height", parts[3])?;

        let model = parse_camera_model_name(model_name)?;
        let params: Vec<f64> = parts[4..]
            .iter()
            .enumerate()
            .map(|(i, p)| parse_f64(path, line_num, &format!("param[{}]", i), p))
            .collect::<Result<Vec<_>, _>>()?;
        if params.len() != model.num_params {
            return Err(TrainingError::InvalidInput(format!(
                "{} line {}: camera model {} expects {} parameters, got {}",
                path.display(),
                line_num + 1,
                model.name,
                model.num_params,
                params.len()
            )));
        }

        cameras.push(ColmapCamera {
            camera_id,
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

fn parse_camera_model_name(name: &str) -> Result<&'static ColmapCameraModelSpec, TrainingError> {
    colmap_camera_model_by_name(&name.to_ascii_uppercase())
        .ok_or_else(|| TrainingError::InvalidInput(format!("unknown camera model '{}'", name)))
}

fn parse_colmap_images(dir: &Path) -> Result<Vec<ColmapImage>, TrainingError> {
    match select_colmap_model_file(dir, "images") {
        Some((path, ColmapModelEncoding::Binary)) => parse_images_binary(&path),
        Some((path, ColmapModelEncoding::Text)) => parse_images_text(&path),
        None => Err(TrainingError::InvalidInput(format!(
            "no images file found in {}",
            dir.display(),
        ))),
    }
}

fn parse_images_binary(path: &Path) -> Result<Vec<ColmapImage>, TrainingError> {
    let mut file = File::open(path)?;
    let num_images = read_bounded_count(&mut file, path, "image", 73)?;

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
        let camera_id = read_u32(&mut file)?;
        let name = read_null_terminated_string(&mut file)?;
        validate_image_name(&name)?;

        // Skip 2D point observations (we don't need them for dataset loading)
        let num_points2d = read_bounded_count(&mut file, path, "points2D", 24)?;
        for _ in 0..num_points2d {
            // x, y, point3D_id
            read_f64(&mut file)?;
            read_f64(&mut file)?;
            read_u64(&mut file)?;
        }

        images.push(ColmapImage {
            image_id,
            camera_id,
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
    let mut lines = reader.lines().enumerate();

    while let Some((line_num, line)) = lines.next() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Image line format: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        // Points line follows (we skip it)
        let parts = trimmed.split_whitespace().take(9).collect::<Vec<_>>();
        if parts.len() != 9 || parts[0].parse::<u32>().is_err() {
            return Err(TrainingError::InvalidInput(format!(
                "{} line {}: invalid image record",
                path.display(),
                line_num + 1
            )));
        }

        let image_id = parse_u32(path, line_num, "image_id", parts[0])?;
        let qw = parse_f64(path, line_num, "qw", parts[1])?;
        let qx = parse_f64(path, line_num, "qx", parts[2])?;
        let qy = parse_f64(path, line_num, "qy", parts[3])?;
        let qz = parse_f64(path, line_num, "qz", parts[4])?;
        let tx = parse_f64(path, line_num, "tx", parts[5])?;
        let ty = parse_f64(path, line_num, "ty", parts[6])?;
        let tz = parse_f64(path, line_num, "tz", parts[7])?;
        let camera_id = parse_u32(path, line_num, "camera_id", parts[8])?;
        let name = image_name_from_header(trimmed).ok_or_else(|| {
            TrainingError::InvalidInput(format!(
                "{} line {}: missing image name",
                path.display(),
                line_num + 1
            ))
        })?;
        validate_image_name(&name)?;
        let (_, points_line) = lines.next().ok_or_else(|| {
            TrainingError::InvalidInput(format!(
                "{} line {}: missing points2D line",
                path.display(),
                line_num + 1
            ))
        })?;
        points_line?;

        images.push(ColmapImage {
            image_id,
            camera_id,
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
    match select_colmap_model_file(dir, "points3D") {
        Some((path, ColmapModelEncoding::Binary)) => parse_points3d_binary(&path),
        Some((path, ColmapModelEncoding::Text)) => parse_points3d_text(&path),
        None => Err(TrainingError::InvalidInput(format!(
            "missing COLMAP sparse points in {} (expected points3D.bin or points3D.txt)",
            dir.display(),
        ))),
    }
}

fn parse_points3d_binary(path: &Path) -> Result<Vec<ColmapPoint3D>, TrainingError> {
    let mut file = File::open(path)?;
    let num_points = read_bounded_count(&mut file, path, "point3D", 51)?;

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
        let track_len = read_bounded_count(&mut file, path, "point3D track", 8)?;
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
            return Err(TrainingError::InvalidInput(format!(
                "{} line {}: invalid point3D record",
                path.display(),
                line_num + 1
            )));
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

#[cfg(test)]
mod tests {
    use super::*;
    use rustscan_types::colmap::COLMAP_CAMERA_MODELS;
    use std::fs;
    use tempfile::tempdir;

    fn params(model: &ColmapCameraModelSpec) -> String {
        (0..model.num_params)
            .map(|idx| (idx + 1).to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn write_minimal_fixture(
        sparse: &Path,
        cameras: &str,
        images: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        fs::create_dir_all(sparse)?;
        fs::write(sparse.join("cameras.txt"), cameras)?;
        fs::write(sparse.join("images.txt"), images)?;
        fs::write(sparse.join("points3D.txt"), "1 0 0 1 255 0 0 0\n")?;
        Ok(())
    }

    #[test]
    fn sparse_model_fingerprint_prefers_binary_files_over_text_fallbacks() {
        let dir = tempdir().unwrap();
        for stem in ["cameras", "images", "points3D"] {
            fs::write(
                dir.path().join(format!("{stem}.bin")),
                format!("{stem} bin"),
            )
            .unwrap();
            fs::write(
                dir.path().join(format!("{stem}.txt")),
                format!("{stem} txt"),
            )
            .unwrap();
        }
        let original = fingerprint_colmap_sparse_model(dir.path()).unwrap();

        for stem in ["cameras", "images", "points3D"] {
            fs::write(
                dir.path().join(format!("{stem}.txt")),
                format!("changed {stem} txt"),
            )
            .unwrap();
        }
        let changed_text = fingerprint_colmap_sparse_model(dir.path()).unwrap();

        assert_eq!(changed_text, original);
    }

    #[test]
    fn sparse_model_fingerprint_matches_dataset_root_and_direct_sparse_input() {
        let dir = tempdir().unwrap();
        let sparse = dir.path().join("sparse/0");
        write_minimal_fixture(&sparse, "camera model", "image poses").unwrap();

        assert_eq!(
            fingerprint_colmap_sparse_model(dir.path()).unwrap(),
            fingerprint_colmap_sparse_model(&sparse).unwrap()
        );
    }

    #[test]
    fn parses_every_authoritative_camera_model_from_text_and_binary() {
        let dir = tempdir().unwrap();
        let text = COLMAP_CAMERA_MODELS
            .iter()
            .enumerate()
            .map(|(idx, model)| format!("{} {} 640 480 {}\n", idx + 1, model.name, params(model)))
            .collect::<String>();
        let text_path = dir.path().join("cameras.txt");
        fs::write(&text_path, text).unwrap();
        let parsed_text = parse_cameras_text(&text_path).unwrap();

        let mut binary = Vec::new();
        binary.extend_from_slice(&(COLMAP_CAMERA_MODELS.len() as u64).to_le_bytes());
        for (idx, model) in COLMAP_CAMERA_MODELS.iter().enumerate() {
            binary.extend_from_slice(&((idx + 1) as u32).to_le_bytes());
            binary.extend_from_slice(&(model.id as u32).to_le_bytes());
            binary.extend_from_slice(&640u64.to_le_bytes());
            binary.extend_from_slice(&480u64.to_le_bytes());
            for value in 0..model.num_params {
                binary.extend_from_slice(&((value + 1) as f64).to_le_bytes());
            }
        }
        let binary_path = dir.path().join("cameras.bin");
        fs::write(&binary_path, binary).unwrap();
        let parsed_binary = parse_cameras_binary(&binary_path).unwrap();

        for (expected, (text, binary)) in COLMAP_CAMERA_MODELS
            .iter()
            .zip(parsed_text.iter().zip(&parsed_binary))
        {
            assert_eq!(text.model, expected);
            assert_eq!(binary.model, expected);
            assert_eq!(text.params.len(), expected.num_params);
            assert_eq!(binary.params.len(), expected.num_params);
        }
        assert_eq!(parsed_text[0].model.id, COLMAP_SIMPLE_PINHOLE);
    }

    #[test]
    fn rejects_multi_camera_training_instead_of_using_the_first_camera() {
        let dir = tempdir().unwrap();
        let sparse = dir.path().join("sparse");
        let image_root = dir.path().join("source-images");
        fs::create_dir_all(&image_root).unwrap();
        fs::write(image_root.join("a.png"), []).unwrap();
        fs::write(image_root.join("b.png"), []).unwrap();
        write_minimal_fixture(
            &sparse,
            "11 PINHOLE 640 480 500 500 320 240\n42 PINHOLE 800 600 700 700 400 300\n",
            "1 1 0 0 0 0 0 0 11 a.png\n\n2 1 0 0 0 0 0 0 42 b.png\n\n",
        )
        .unwrap();

        let error = load_colmap_dataset(
            &sparse,
            &ColmapConfig {
                image_root: Some(image_root),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("one shared COLMAP CAMERA_ID"));
    }

    #[test]
    fn rejects_distorted_images_without_an_undistortion_path() {
        let dir = tempdir().unwrap();
        let sparse = dir.path().join("sparse");
        let image_root = dir.path().join("images");
        fs::create_dir_all(&image_root).unwrap();
        fs::write(image_root.join("a.png"), []).unwrap();
        write_minimal_fixture(
            &sparse,
            "1 SIMPLE_RADIAL 640 480 500 320 240 0.1\n",
            "1 1 0 0 0 0 0 0 1 a.png\n\n",
        )
        .unwrap();

        let error = load_colmap_dataset(
            &sparse,
            &ColmapConfig {
                image_root: Some(image_root),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("undistort images"));
    }

    #[test]
    fn rejects_images_that_reference_a_missing_camera() {
        let dir = tempdir().unwrap();
        let sparse = dir.path().join("sparse");
        let image_root = dir.path().join("images");
        fs::create_dir_all(&image_root).unwrap();
        fs::write(image_root.join("a.png"), []).unwrap();
        write_minimal_fixture(
            &sparse,
            "1 PINHOLE 640 480 500 500 320 240\n",
            "1 1 0 0 0 0 0 0 99 a.png\n\n",
        )
        .unwrap();

        let error = load_colmap_dataset(
            &sparse,
            &ColmapConfig {
                image_root: Some(image_root),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("missing CAMERA_ID 99"));
    }

    #[test]
    fn preserves_text_image_name_to_end_of_line_and_camera_id() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("images.txt");
        fs::write(&path, "7 1 0 0 0 0 0 0 42 nested/image name.png\n\n").unwrap();

        let images = parse_images_text(&path).unwrap();
        assert_eq!(images[0].camera_id, 42);
        assert_eq!(images[0].name, "nested/image name.png");
    }

    #[test]
    fn rejects_unsafe_image_names() {
        for name in ["../escape.png", "/absolute.png", "C:\\absolute.png"] {
            assert!(validate_image_name(name).is_err(), "{name}");
        }
    }

    #[test]
    fn explicit_image_root_supports_sparse_only_exports_without_parent_panics() {
        let dir = tempdir().unwrap();
        let sparse = dir.path().join("relative-sparse");
        let image_root = dir.path().join("original-images");
        fs::create_dir_all(&image_root).unwrap();
        fs::write(image_root.join("a.png"), []).unwrap();
        write_minimal_fixture(
            &sparse,
            "1 SIMPLE_PINHOLE 640 480 500 320 240\n",
            "1 1 0 0 0 0 0 0 1 a.png\n\n",
        )
        .unwrap();

        let dataset = load_colmap_dataset(
            &sparse,
            &ColmapConfig {
                image_root: Some(image_root.clone()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(dataset.poses[0].image_path, image_root.join("a.png"));
        assert_eq!(dataset.intrinsics.fx, 500.0);
    }

    #[test]
    fn binary_camera_dimensions_must_fit_u32() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("cameras.bin");
        let mut binary = Vec::new();
        binary.extend_from_slice(&1u64.to_le_bytes());
        binary.extend_from_slice(&1u32.to_le_bytes());
        binary.extend_from_slice(&(COLMAP_PINHOLE as u32).to_le_bytes());
        binary.extend_from_slice(&(u64::from(u32::MAX) + 1).to_le_bytes());
        binary.extend_from_slice(&480u64.to_le_bytes());
        fs::write(&path, binary).unwrap();

        assert!(parse_cameras_binary(&path).is_err());
    }

    #[test]
    fn malformed_text_camera_and_point_records_are_rejected() {
        let dir = tempdir().unwrap();
        let cameras = dir.path().join("cameras.txt");
        let points = dir.path().join("points3D.txt");
        fs::write(&cameras, "1 PINHOLE 640\n").unwrap();
        fs::write(&points, "1 0 0\n").unwrap();

        assert!(parse_cameras_text(&cameras).is_err());
        assert!(parse_points3d_text(&points).is_err());
    }

    #[test]
    fn binary_collection_counts_are_bounded_by_remaining_bytes() {
        let dir = tempdir().unwrap();
        let cameras = dir.path().join("cameras.bin");
        let images = dir.path().join("images.bin");
        let points = dir.path().join("points3D.bin");
        for path in [&cameras, &images, &points] {
            fs::write(path, u64::MAX.to_le_bytes()).unwrap();
        }

        assert!(parse_cameras_binary(&cameras).is_err());
        assert!(parse_images_binary(&images).is_err());
        assert!(parse_points3d_binary(&points).is_err());
    }

    #[cfg(feature = "rustsfm-contract-tests")]
    #[test]
    fn loads_sparse_text_fixture_written_by_rustsfm() {
        use rustsfm::colmap::{
            write_colmap_sparse_text, ColmapCamera, ColmapImage, ColmapPoint3D, ColmapSparseFiles,
        };

        let dir = tempdir().unwrap();
        let sparse = dir.path().join("sparse/0");
        let image_root = dir.path().join("source-images");
        fs::create_dir_all(&image_root).unwrap();
        fs::write(image_root.join("frame one.png"), []).unwrap();
        write_colmap_sparse_text(
            &sparse,
            &ColmapSparseFiles {
                cameras: vec![ColmapCamera {
                    camera_id: 7,
                    model_id: COLMAP_SIMPLE_PINHOLE,
                    width: 640,
                    height: 480,
                    params: vec![500.0, 320.0, 240.0],
                }],
                rigs: Vec::new(),
                frames: Vec::new(),
                images: vec![ColmapImage {
                    image_id: 11,
                    camera_id: 7,
                    name: "frame one.png".to_string(),
                    qvec: [1.0, 0.0, 0.0, 0.0],
                    tvec: [0.0, 0.0, 0.0],
                    points2d: Vec::new(),
                }],
                points3d: vec![ColmapPoint3D {
                    point3d_id: 99,
                    xyz: [1.0, 2.0, 3.0],
                    color: [10, 20, 30],
                    error: 0.0,
                    track: Vec::new(),
                }],
            },
        )
        .unwrap();

        let dataset = load_colmap_dataset(
            &sparse,
            &ColmapConfig {
                image_root: Some(image_root.clone()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(dataset.poses[0].frame_id, 0);
        assert_eq!(
            dataset.poses[0].image_path,
            image_root.join("frame one.png")
        );
        assert_eq!(
            dataset.intrinsics,
            Intrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480)
        );
        assert_eq!(dataset.initial_points[0].0, [1.0, 2.0, 3.0]);
    }
}
