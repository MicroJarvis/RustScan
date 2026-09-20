//! Dataset loading for SLAM
//!
//! This module provides interfaces and implementations for loading
//! standard SLAM datasets (TUM RGB-D, KITTI, EuRoC, etc.)

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::core::{Camera, SE3};

/// Errors that can occur when loading datasets
#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Invalid dataset format: {0}")]
    Format(String),

    #[error("Image loading error: {0}")]
    Image(String),

    #[error("Camera calibration missing or invalid")]
    Calibration,

    #[error("Frame index out of bounds: {0}")]
    FrameIndex(usize),
}

/// Result type for dataset operations
pub type Result<T> = std::result::Result<T, DatasetError>;

/// A single frame in a dataset
#[derive(Debug, Clone)]
pub struct Frame {
    /// Frame index (0-based)
    pub index: usize,
    /// Timestamp in seconds
    pub timestamp: f64,
    /// RGB color image (width * height * 3, row-major)
    pub color: Vec<u8>,
    /// Depth image in meters (width * height, row-major)
    /// None if depth not available
    pub depth: Option<Vec<f32>>,
    /// Camera intrinsic parameters
    pub camera: Camera,
    /// Ground truth pose (if available)
    pub ground_truth_pose: Option<SE3>,
    /// Image width
    pub width: u32,
    /// Image height
    pub height: u32,
}

impl Frame {
    /// Create a new frame
    pub fn new(
        index: usize,
        timestamp: f64,
        color: Vec<u8>,
        depth: Option<Vec<f32>>,
        camera: Camera,
        ground_truth_pose: Option<SE3>,
    ) -> Self {
        let width = camera.width;
        let height = camera.height;

        Self {
            index,
            timestamp,
            color,
            depth,
            camera,
            ground_truth_pose,
            width,
            height,
        }
    }

    /// Check if this frame has depth data
    pub fn has_depth(&self) -> bool {
        self.depth.is_some()
    }

    /// Check if this frame has ground truth pose
    pub fn has_ground_truth(&self) -> bool {
        self.ground_truth_pose.is_some()
    }
}

/// Trait for SLAM datasets
pub trait Dataset {
    /// Get the number of frames in the dataset
    fn len(&self) -> usize;

    /// Check if the dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a frame by index
    fn get_frame(&self, index: usize) -> Result<Frame>;

    /// Get the camera model (common for all frames)
    fn camera(&self) -> Camera;

    /// Get dataset metadata
    fn metadata(&self) -> &DatasetMetadata;

    /// Iterate over all frames
    fn frames(&self) -> DatasetIterator<'_>
    where
        Self: Sized,
    {
        DatasetIterator {
            dataset: self,
            current: 0,
        }
    }
}

/// Iterator over dataset frames
pub struct DatasetIterator<'a> {
    dataset: &'a dyn Dataset,
    current: usize,
}

impl<'a> Iterator for DatasetIterator<'a> {
    type Item = Result<Frame>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.dataset.len() {
            let result = self.dataset.get_frame(self.current);
            self.current += 1;
            Some(result)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.dataset.len().saturating_sub(self.current);
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for DatasetIterator<'a> {}

/// Metadata about a dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Dataset name (e.g., "TUM RGB-D", "KITTI Odometry")
    pub name: String,
    /// Sequence identifier
    pub sequence: String,
    /// Total number of frames
    pub total_frames: usize,
    /// Whether dataset provides depth images
    pub has_depth: bool,
    /// Whether dataset provides ground truth poses
    pub has_ground_truth: bool,
    /// Frame rate in Hz (if known)
    pub frame_rate: Option<f32>,
    /// Average translational speed (m/s, if known)
    pub avg_speed: Option<f32>,
    /// Total trajectory length (m, if known)
    pub trajectory_length: Option<f32>,
    /// Dataset-specific notes
    pub notes: String,
}

/// Configuration for loading a dataset
#[derive(Debug, Clone)]
pub struct DatasetConfig {
    /// Path to dataset root directory
    pub root_path: PathBuf,
    /// Whether to load depth images
    pub load_depth: bool,
    /// Whether to load ground truth poses
    pub load_ground_truth: bool,
    /// Maximum number of frames to load (0 for all)
    pub max_frames: usize,
    /// Frame stride (load every nth frame)
    pub stride: usize,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            root_path: PathBuf::new(),
            load_depth: true,
            load_ground_truth: false,
            max_frames: 0,
            stride: 1,
        }
    }
}

/// Parse an association file (TUM RGB-D format)
/// Format: timestamp filename timestamp filename
fn parse_association_file(path: &Path) -> Result<Vec<(f64, String)>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut entries = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 2 {
            return Err(DatasetError::Format(format!(
                "Line {}: expected 2 parts, got {}",
                line_num + 1,
                parts.len()
            )));
        }

        let timestamp: f64 = parts[0].parse().map_err(|e| {
            DatasetError::Format(format!(
                "Line {}: invalid timestamp '{}': {}",
                line_num + 1,
                parts[0],
                e
            ))
        })?;

        let filename = parts[1].to_string();
        entries.push((timestamp, filename));
    }

    Ok(entries)
}

/// Load a 16-bit PNG depth image and convert to meters
#[cfg(feature = "image")]
fn load_depth_image(path: &Path) -> Result<Vec<f32>> {
    use image::io::Reader as ImageReader;
    use image::ImageFormat;

    let img = ImageReader::open(path)?
        .with_guessed_format()?
        .decode()
        .map_err(|e| DatasetError::Image(e.to_string()))?;

    let width = img.width() as usize;
    let height = img.height() as usize;

    // TUM RGB-D stores depth in millimeters as 16-bit
    let depth_data: Vec<f32> = match img {
        image::DynamicImage::ImageLuma16(luma) => {
            luma.pixels()
                .map(|p| p.0[0] as f32 / 1000.0) // Convert mm to meters
                .collect()
        }
        _ => {
            return Err(DatasetError::Image(format!(
                "Expected 16-bit grayscale image, got {:?}",
                img.color()
            )))
        }
    };

    if depth_data.len() != width * height {
        return Err(DatasetError::Image(format!(
            "Depth image size mismatch: expected {} pixels, got {}",
            width * height,
            depth_data.len()
        )));
    }

    Ok(depth_data)
}

/// Load an RGB image
#[cfg(feature = "image")]
fn load_color_image(path: &Path) -> Result<Vec<u8>> {
    use image::io::Reader as ImageReader;

    let img = ImageReader::open(path)?
        .with_guessed_format()?
        .decode()
        .map_err(|e| DatasetError::Image(e.to_string()))?;

    let rgb = img.to_rgb8();
    Ok(rgb.as_raw().to_vec())
}

#[cfg(not(feature = "image"))]
fn load_depth_image(_path: &Path) -> Result<Vec<f32>> {
    Err(DatasetError::Image(
        "Depth image loading requires 'image' feature".to_string(),
    ))
}

#[cfg(not(feature = "image"))]
fn load_color_image(_path: &Path) -> Result<Vec<u8>> {
    Err(DatasetError::Image(
        "Color image loading requires 'image' feature".to_string(),
    ))
}

fn collect_image_entries(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut entries = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg"))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    entries.sort();
    Ok(entries)
}

fn parse_projection_camera(line: &str, default_width: u32, default_height: u32) -> Result<Camera> {
    let mut parts = line.split_whitespace();
    let _label = parts.next();
    let values = parts
        .map(|value| {
            value.parse::<f32>().map_err(|e| {
                DatasetError::Format(format!("invalid calibration value '{}': {}", value, e))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if values.len() < 12 {
        return Err(DatasetError::Format(format!(
            "expected 12 projection values, got {}",
            values.len()
        )));
    }

    Ok(Camera::new(
        values[0],
        values[5],
        values[2],
        values[6],
        default_width,
        default_height,
    ))
}

fn parse_pose_matrix_row(values: &[&str], timestamp: f64) -> Result<(f64, SE3)> {
    if values.len() < 12 {
        return Err(DatasetError::Format(format!(
            "expected at least 12 pose values, got {}",
            values.len()
        )));
    }

    let parsed = values
        .iter()
        .take(12)
        .map(|value| {
            value
                .parse::<f32>()
                .map_err(|e| DatasetError::Format(format!("invalid pose value '{}': {}", value, e)))
        })
        .collect::<Result<Vec<_>>>()?;

    let rotation = [
        [parsed[0], parsed[1], parsed[2]],
        [parsed[4], parsed[5], parsed[6]],
        [parsed[8], parsed[9], parsed[10]],
    ];
    let translation = [parsed[3], parsed[7], parsed[11]];
    Ok((
        timestamp,
        SE3::from_rotation_translation(&rotation, &translation),
    ))
}

fn find_closest_pose(poses: &[(f64, SE3)], timestamp: f64, threshold_seconds: f64) -> Option<SE3> {
    if poses.is_empty() {
        return None;
    }

    match poses.binary_search_by(|(ts, _)| {
        ts.partial_cmp(&timestamp)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        Ok(idx) => Some(poses[idx].1),
        Err(idx) => {
            let mut best: Option<(f64, SE3)> = None;
            for candidate in [idx.checked_sub(1), Some(idx)].into_iter().flatten() {
                if let Some((ts, pose)) = poses.get(candidate) {
                    let diff = (ts - timestamp).abs();
                    if best.map(|(best_diff, _)| diff < best_diff).unwrap_or(true) {
                        best = Some((diff, *pose));
                    }
                }
            }
            match best {
                Some((diff, pose)) if diff <= threshold_seconds => Some(pose),
                _ => None,
            }
        }
    }
}

/// TUM RGB-D dataset loader
///
/// The TUM RGB-D dataset format:
/// - rgb.txt: timestamps and filenames for color images
/// - depth.txt: timestamps and filenames for depth images
/// - groundtruth.txt: timestamps and ground truth poses
/// - camera calibration file (usually fixed parameters)
///
/// Reference: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
pub struct TumRgbdDataset {
    config: DatasetConfig,
    metadata: DatasetMetadata,
    camera: Camera,
    color_entries: Vec<(f64, PathBuf)>,
    depth_entries: Vec<(f64, PathBuf)>,
    ground_truth: Vec<(f64, SE3)>,
}

impl TumRgbdDataset {
    /// Load a TUM RGB-D dataset from the given path
    pub fn load(config: DatasetConfig) -> Result<Self> {
        let root = &config.root_path;

        // Check required files
        let rgb_file = root.join("rgb.txt");
        let depth_file = root.join("depth.txt");

        if !rgb_file.exists() {
            return Err(DatasetError::Format(format!(
                "rgb.txt not found in {}",
                root.display()
            )));
        }

        // Parse association files
        let color_entries = parse_association_file(&rgb_file)?
            .into_iter()
            .map(|(ts, fname)| (ts, root.join(fname)))
            .collect::<Vec<_>>();

        let depth_entries = if depth_file.exists() && config.load_depth {
            parse_association_file(&depth_file)?
                .into_iter()
                .map(|(ts, fname)| (ts, root.join(fname)))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        // Parse ground truth if available
        let ground_truth = if config.load_ground_truth {
            let gt_file = root.join("groundtruth.txt");
            if gt_file.exists() {
                Self::parse_ground_truth(&gt_file)?
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Load camera calibration
        let camera = Self::load_camera(root)?;

        // Apply frame limits and stride
        let max_frames = if config.max_frames > 0 {
            config.max_frames.min(color_entries.len())
        } else {
            color_entries.len()
        };

        let color_entries = color_entries
            .into_iter()
            .take(max_frames)
            .step_by(config.stride)
            .collect::<Vec<_>>();

        let depth_entries = depth_entries
            .into_iter()
            .take(max_frames)
            .step_by(config.stride)
            .collect::<Vec<_>>();

        // Create metadata
        let metadata = DatasetMetadata {
            name: "TUM RGB-D".to_string(),
            sequence: root
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string(),
            total_frames: color_entries.len(),
            has_depth: !depth_entries.is_empty(),
            has_ground_truth: !ground_truth.is_empty(),
            frame_rate: Some(30.0), // TUM RGB-D is typically 30Hz
            avg_speed: None,
            trajectory_length: None,
            notes: format!("Loaded from {}", root.display()),
        };

        Ok(Self {
            config,
            metadata,
            camera,
            color_entries,
            depth_entries,
            ground_truth,
        })
    }

    /// Parse ground truth file (TUM format: timestamp tx ty tz qx qy qz qw)
    fn parse_ground_truth(path: &Path) -> Result<Vec<(f64, SE3)>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut poses = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 8 {
                return Err(DatasetError::Format(format!(
                    "Line {}: expected 8 parts for ground truth, got {}",
                    line_num + 1,
                    parts.len()
                )));
            }

            let timestamp: f64 = parts[0].parse().map_err(|e| {
                DatasetError::Format(format!(
                    "Line {}: invalid timestamp '{}': {}",
                    line_num + 1,
                    parts[0],
                    e
                ))
            })?;

            // Parse translation
            let tx: f32 = parts[1].parse().map_err(|e| {
                DatasetError::Format(format!(
                    "Line {}: invalid tx '{}': {}",
                    line_num + 1,
                    parts[1],
                    e
                ))
            })?;
            let ty: f32 = parts[2].parse().map_err(|e| {
                DatasetError::Format(format!(
                    "Line {}: invalid ty '{}': {}",
                    line_num + 1,
                    parts[2],
                    e
                ))
            })?;
            let tz: f32 = parts[3].parse().map_err(|e| {
                DatasetError::Format(format!(
                    "Line {}: invalid tz '{}': {}",
                    line_num + 1,
                    parts[3],
                    e
                ))
            })?;

            // Parse quaternion (xyzw order in TUM)
            let qx: f32 = parts[4].parse().map_err(|e| {
                DatasetError::Format(format!(
                    "Line {}: invalid qx '{}': {}",
                    line_num + 1,
                    parts[4],
                    e
                ))
            })?;
            let qy: f32 = parts[5].parse().map_err(|e| {
                DatasetError::Format(format!(
                    "Line {}: invalid qy '{}': {}",
                    line_num + 1,
                    parts[5],
                    e
                ))
            })?;
            let qz: f32 = parts[6].parse().map_err(|e| {
                DatasetError::Format(format!(
                    "Line {}: invalid qz '{}': {}",
                    line_num + 1,
                    parts[6],
                    e
                ))
            })?;
            let qw: f32 = parts[7].parse().map_err(|e| {
                DatasetError::Format(format!(
                    "Line {}: invalid qw '{}': {}",
                    line_num + 1,
                    parts[7],
                    e
                ))
            })?;

            // Create SE3 from translation and quaternion
            let translation = nalgebra::Vector3::new(tx, ty, tz);
            let rotation = rustscan_types::rotation_from_xyzw([qx, qy, qz, qw]);
            let pose = SE3::from_quat_translation(rotation, translation);

            poses.push((timestamp, pose));
        }

        // Sort by timestamp for binary search
        poses.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(poses)
    }

    /// Load camera calibration
    /// TUM RGB-D uses fixed intrinsics for most sequences
    fn load_camera(root: &Path) -> Result<Camera> {
        // Try to load from calibration file
        let calib_file = root.join("calibration.txt");
        if calib_file.exists() {
            // Simple calibration file format: fx fy cx cy
            if let Ok(content) = std::fs::read_to_string(&calib_file) {
                let lines: Vec<&str> = content.lines().collect();
                if lines.len() >= 4 {
                    let fx: f32 = lines[0].parse().unwrap_or(525.0);
                    let fy: f32 = lines[1].parse().unwrap_or(525.0);
                    let cx: f32 = lines[2].parse().unwrap_or(319.5);
                    let cy: f32 = lines[3].parse().unwrap_or(239.5);

                    // Check for image dimensions
                    let width = 640;
                    let height = 480;

                    return Ok(Camera::new(fx, fy, cx, cy, width, height));
                }
            }
        }

        // Default TUM RGB-D intrinsics (freiburg1, freiburg2, freiburg3)
        // Most sequences are 640x480 with these parameters
        Ok(Camera::new(525.0, 525.0, 319.5, 239.5, 640, 480))
    }

    /// Find closest depth timestamp for a given color timestamp
    fn find_closest_depth(&self, color_timestamp: f64) -> Option<(f64, &PathBuf)> {
        if self.depth_entries.is_empty() {
            return None;
        }

        // Simple linear search (entries are small, OK for now)
        self.depth_entries
            .iter()
            .min_by_key(|(ts, _)| {
                // Use absolute difference as key
                let diff = (ts - color_timestamp).abs();
                (diff * 1e6) as i64 // Convert to microseconds for integer comparison
            })
            .map(|(ts, path)| (*ts, path))
    }

    /// Find closest ground truth pose for a given timestamp
    fn find_closest_ground_truth(&self, timestamp: f64) -> Option<SE3> {
        if self.ground_truth.is_empty() {
            return None;
        }

        // Binary search for closest timestamp (ground_truth is sorted)
        match self.ground_truth.binary_search_by(|(ts, _)| {
            ts.partial_cmp(&timestamp)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(idx) => {
                // Exact match
                Some(self.ground_truth[idx].1)
            }
            Err(idx) => {
                // idx is insertion point, check neighbors
                let mut best_idx = idx;
                let mut best_diff = f64::INFINITY;

                // Check idx (if within bounds)
                if idx < self.ground_truth.len() {
                    let diff = (self.ground_truth[idx].0 - timestamp).abs();
                    if diff < best_diff {
                        best_diff = diff;
                        best_idx = idx;
                    }
                }

                // Check idx-1 (if exists)
                if idx > 0 {
                    let diff = (self.ground_truth[idx - 1].0 - timestamp).abs();
                    if diff < best_diff {
                        best_diff = diff;
                        best_idx = idx - 1;
                    }
                }

                // Return best match if within reasonable threshold (100ms)
                if best_diff < 0.1 {
                    Some(self.ground_truth[best_idx].1)
                } else {
                    None
                }
            }
        }
    }
}

impl Dataset for TumRgbdDataset {
    fn len(&self) -> usize {
        self.color_entries.len()
    }

    fn get_frame(&self, index: usize) -> Result<Frame> {
        if index >= self.color_entries.len() {
            return Err(DatasetError::FrameIndex(index));
        }

        let (color_ts, color_path) = &self.color_entries[index];

        // Load color image
        let color = load_color_image(color_path)?;

        // Load depth image if available
        let depth = if self.config.load_depth && !self.depth_entries.is_empty() {
            if let Some((depth_ts, depth_path)) = self.find_closest_depth(*color_ts) {
                // Check if timestamp difference is acceptable (100ms threshold)
                if (depth_ts - *color_ts).abs() < 0.1 {
                    load_depth_image(depth_path).ok()
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Get ground truth pose if available
        let ground_truth_pose = if self.config.load_ground_truth {
            self.find_closest_ground_truth(*color_ts)
        } else {
            None
        };

        Ok(Frame::new(
            index,
            *color_ts,
            color,
            depth,
            self.camera.clone(),
            ground_truth_pose,
        ))
    }

    fn camera(&self) -> Camera {
        self.camera.clone()
    }

    fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }
}

/// KITTI Odometry dataset loader.
///
/// The loader currently reads the left monocular stream from `image_0/`,
/// camera intrinsics from `calib.txt`, and optional poses from `poses.txt`.
pub struct KittiDataset {
    config: DatasetConfig,
    metadata: DatasetMetadata,
    camera: Camera,
    image_entries: Vec<PathBuf>,
    poses: Vec<(f64, SE3)>,
}

impl KittiDataset {
    pub fn load(config: DatasetConfig) -> Result<Self> {
        let root = &config.root_path;
        let image_dir = root.join("image_0");
        if !image_dir.is_dir() {
            return Err(DatasetError::Format(format!(
                "KITTI image directory not found: {}",
                image_dir.display()
            )));
        }

        let calib_path = root.join("calib.txt");
        let camera = if calib_path.exists() {
            let calib = std::fs::read_to_string(&calib_path)?;
            let line = calib
                .lines()
                .find(|line| line.starts_with("P0:") || line.starts_with("P2:"))
                .ok_or_else(|| {
                    DatasetError::Format("calib.txt missing P0/P2 projection matrix".into())
                })?;
            parse_projection_camera(line, 1241, 376)?
        } else {
            Camera::new(718.856, 718.856, 607.1928, 185.2157, 1241, 376)
        };

        let image_entries = collect_image_entries(&image_dir)?;
        if image_entries.is_empty() {
            return Err(DatasetError::Format(format!(
                "no KITTI images found in {}",
                image_dir.display()
            )));
        }

        let poses = {
            let poses_path = root.join("poses.txt");
            if poses_path.exists() {
                let file = File::open(&poses_path)?;
                let reader = BufReader::new(file);
                let mut parsed = Vec::new();
                for (idx, line) in reader.lines().enumerate() {
                    let line = line?;
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let values = trimmed.split_whitespace().collect::<Vec<_>>();
                    parsed.push(parse_pose_matrix_row(&values, idx as f64)?);
                }
                parsed
            } else {
                Vec::new()
            }
        };

        let stride = config.stride.max(1);
        let max_frames = if config.max_frames > 0 {
            config.max_frames.min(image_entries.len())
        } else {
            image_entries.len()
        };
        let image_entries = image_entries
            .into_iter()
            .take(max_frames)
            .step_by(stride)
            .collect::<Vec<_>>();

        let metadata = DatasetMetadata {
            name: "KITTI Odometry".to_string(),
            sequence: root
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string(),
            total_frames: image_entries.len(),
            has_depth: false,
            has_ground_truth: !poses.is_empty(),
            frame_rate: Some(10.0),
            avg_speed: None,
            trajectory_length: None,
            notes: format!("Loaded monocular KITTI stream from {}", root.display()),
        };

        Ok(Self {
            config,
            metadata,
            camera,
            image_entries,
            poses,
        })
    }
}

impl Dataset for KittiDataset {
    fn len(&self) -> usize {
        self.image_entries.len()
    }

    fn get_frame(&self, index: usize) -> Result<Frame> {
        if index >= self.image_entries.len() {
            return Err(DatasetError::FrameIndex(index));
        }

        let color = load_color_image(&self.image_entries[index])?;
        let timestamp = index as f64 / self.metadata.frame_rate.unwrap_or(10.0) as f64;
        let ground_truth_pose = if self.config.load_ground_truth {
            self.poses.get(index).map(|(_, pose)| *pose)
        } else {
            None
        };

        Ok(Frame::new(
            index,
            timestamp,
            color,
            None,
            self.camera.clone(),
            ground_truth_pose,
        ))
    }

    fn camera(&self) -> Camera {
        self.camera.clone()
    }

    fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }
}

/// EuRoC MAV dataset loader.
///
/// The current implementation reads the `mav0/cam0` monocular stream,
/// parses optional intrinsics from `sensor.yaml`, and loads optional
/// ground-truth poses from `state_groundtruth_estimate0/data.csv`.
pub struct EurocDataset {
    config: DatasetConfig,
    metadata: DatasetMetadata,
    camera: Camera,
    image_entries: Vec<(f64, PathBuf)>,
    ground_truth: Vec<(f64, SE3)>,
}

impl EurocDataset {
    pub fn load(config: DatasetConfig) -> Result<Self> {
        let root = &config.root_path;
        let cam0_dir = root.join("mav0").join("cam0");
        let data_dir = cam0_dir.join("data");
        let csv_path = cam0_dir.join("data.csv");
        if !data_dir.is_dir() || !csv_path.exists() {
            return Err(DatasetError::Format(format!(
                "EuRoC cam0 data not found under {}",
                root.display()
            )));
        }

        let camera = {
            let sensor_path = cam0_dir.join("sensor.yaml");
            if sensor_path.exists() {
                let sensor = std::fs::read_to_string(&sensor_path)?;
                let intrinsics_line = sensor
                    .lines()
                    .find(|line| line.trim_start().starts_with("intrinsics:"));
                let resolution_line = sensor
                    .lines()
                    .find(|line| line.trim_start().starts_with("resolution:"));

                let (fx, fy, cx, cy) = intrinsics_line
                    .and_then(|line| line.split_once('[').map(|(_, rest)| rest))
                    .and_then(|rest| rest.split_once(']').map(|(vals, _)| vals))
                    .map(|vals| {
                        vals.split(',')
                            .map(|v| v.trim().parse::<f32>())
                            .collect::<std::result::Result<Vec<_>, _>>()
                    })
                    .transpose()
                    .map_err(|e| DatasetError::Format(format!("invalid EuRoC intrinsics: {}", e)))?
                    .and_then(|vals| {
                        (vals.len() == 4).then_some((vals[0], vals[1], vals[2], vals[3]))
                    })
                    .unwrap_or((458.654, 457.296, 367.215, 248.375));

                let (width, height) = resolution_line
                    .and_then(|line| line.split_once('[').map(|(_, rest)| rest))
                    .and_then(|rest| rest.split_once(']').map(|(vals, _)| vals))
                    .map(|vals| {
                        vals.split(',')
                            .map(|v| v.trim().parse::<u32>())
                            .collect::<std::result::Result<Vec<_>, _>>()
                    })
                    .transpose()
                    .map_err(|e| DatasetError::Format(format!("invalid EuRoC resolution: {}", e)))?
                    .and_then(|vals| (vals.len() == 2).then_some((vals[0], vals[1])))
                    .unwrap_or((752, 480));

                Camera::new(fx, fy, cx, cy, width, height)
            } else {
                Camera::new(458.654, 457.296, 367.215, 248.375, 752, 480)
            }
        };

        let file = File::open(&csv_path)?;
        let reader = BufReader::new(file);
        let mut image_entries = Vec::new();
        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts = line.split(',').map(|p| p.trim()).collect::<Vec<_>>();
            if parts.len() < 2 {
                return Err(DatasetError::Format(format!(
                    "invalid EuRoC cam0 csv row '{}'",
                    line
                )));
            }
            let timestamp_ns = parts[0].parse::<u64>().map_err(|e| {
                DatasetError::Format(format!("invalid EuRoC timestamp '{}': {}", parts[0], e))
            })?;
            image_entries.push((timestamp_ns as f64 / 1e9, data_dir.join(parts[1])));
        }
        if image_entries.is_empty() {
            return Err(DatasetError::Format("no EuRoC images found".into()));
        }

        let ground_truth = if config.load_ground_truth {
            let gt_path = root
                .join("mav0")
                .join("state_groundtruth_estimate0")
                .join("data.csv");
            if gt_path.exists() {
                let file = File::open(&gt_path)?;
                let reader = BufReader::new(file);
                let mut poses = Vec::new();
                for line in reader.lines() {
                    let line = line?;
                    let line = line.trim();
                    if line.is_empty() || line.starts_with('#') {
                        continue;
                    }
                    let parts = line.split(',').map(|p| p.trim()).collect::<Vec<_>>();
                    if parts.len() < 8 {
                        return Err(DatasetError::Format(format!(
                            "invalid EuRoC ground truth row '{}'",
                            line
                        )));
                    }
                    let timestamp = parts[0].parse::<u64>().map_err(|e| {
                        DatasetError::Format(format!(
                            "invalid EuRoC ground truth timestamp '{}': {}",
                            parts[0], e
                        ))
                    })? as f64
                        / 1e9;
                    let tx = parts[1].parse::<f32>().map_err(|e| {
                        DatasetError::Format(format!("invalid tx '{}': {}", parts[1], e))
                    })?;
                    let ty = parts[2].parse::<f32>().map_err(|e| {
                        DatasetError::Format(format!("invalid ty '{}': {}", parts[2], e))
                    })?;
                    let tz = parts[3].parse::<f32>().map_err(|e| {
                        DatasetError::Format(format!("invalid tz '{}': {}", parts[3], e))
                    })?;
                    let qw = parts[4].parse::<f32>().map_err(|e| {
                        DatasetError::Format(format!("invalid qw '{}': {}", parts[4], e))
                    })?;
                    let qx = parts[5].parse::<f32>().map_err(|e| {
                        DatasetError::Format(format!("invalid qx '{}': {}", parts[5], e))
                    })?;
                    let qy = parts[6].parse::<f32>().map_err(|e| {
                        DatasetError::Format(format!("invalid qy '{}': {}", parts[6], e))
                    })?;
                    let qz = parts[7].parse::<f32>().map_err(|e| {
                        DatasetError::Format(format!("invalid qz '{}': {}", parts[7], e))
                    })?;
                    poses.push((
                        timestamp,
                        SE3::from_quat_translation(
                            rustscan_types::rotation_from_xyzw([qx, qy, qz, qw]),
                            nalgebra::Vector3::new(tx, ty, tz),
                        ),
                    ));
                }
                poses.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                poses
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        let stride = config.stride.max(1);
        let max_frames = if config.max_frames > 0 {
            config.max_frames.min(image_entries.len())
        } else {
            image_entries.len()
        };
        let image_entries = image_entries
            .into_iter()
            .take(max_frames)
            .step_by(stride)
            .collect::<Vec<_>>();

        let metadata = DatasetMetadata {
            name: "EuRoC MAV".to_string(),
            sequence: root
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string(),
            total_frames: image_entries.len(),
            has_depth: false,
            has_ground_truth: !ground_truth.is_empty(),
            frame_rate: Some(20.0),
            avg_speed: None,
            trajectory_length: None,
            notes: format!("Loaded EuRoC cam0 stream from {}", root.display()),
        };

        Ok(Self {
            config,
            metadata,
            camera,
            image_entries,
            ground_truth,
        })
    }
}

impl Dataset for EurocDataset {
    fn len(&self) -> usize {
        self.image_entries.len()
    }

    fn get_frame(&self, index: usize) -> Result<Frame> {
        if index >= self.image_entries.len() {
            return Err(DatasetError::FrameIndex(index));
        }

        let (timestamp, path) = &self.image_entries[index];
        let color = load_color_image(path)?;
        let ground_truth_pose = if self.config.load_ground_truth {
            find_closest_pose(&self.ground_truth, *timestamp, 0.01)
        } else {
            None
        };

        Ok(Frame::new(
            index,
            *timestamp,
            color,
            None,
            self.camera.clone(),
            ground_truth_pose,
        ))
    }

    fn camera(&self) -> Camera {
        self.camera.clone()
    }

    fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_parse_association_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");

        let content = r#"# color images
1305031102.175304 rgb/1305031102.175304.png
1305031102.211214 rgb/1305031102.211214.png
1305031102.243738 rgb/1305031102.243738.png
"#;
        std::fs::write(&file_path, content).unwrap();

        let entries = parse_association_file(&file_path).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].0, 1305031102.175304);
        assert_eq!(entries[0].1, "rgb/1305031102.175304.png");
    }

    #[test]
    fn test_parse_ground_truth() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("groundtruth.txt");

        let content = r#"# ground truth trajectory
1305031102.175304 0.1 0.2 0.3 0.0 0.0 0.0 1.0
1305031102.211214 0.2 0.3 0.4 0.0 0.0 0.707 0.707
"#;
        std::fs::write(&file_path, content).unwrap();

        let poses = TumRgbdDataset::parse_ground_truth(&file_path).unwrap();
        assert_eq!(poses.len(), 2);
        assert_eq!(poses[0].0, 1305031102.175304);
        assert_eq!(poses[1].0, 1305031102.211214);

        // Check first pose translation
        let pose0 = poses[0].1;
        let t = pose0.translation();
        assert!((t[0] - 0.1).abs() < 1e-6);
        assert!((t[1] - 0.2).abs() < 1e-6);
        assert!((t[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_dataset_config_default() {
        let config = DatasetConfig::default();
        assert_eq!(config.load_depth, true);
        assert_eq!(config.load_ground_truth, false);
        assert_eq!(config.max_frames, 0);
        assert_eq!(config.stride, 1);
    }

    #[test]
    fn test_frame_creation() {
        let camera = Camera::new(525.0, 525.0, 320.0, 240.0, 640, 480);
        let color = vec![255u8; 640 * 480 * 3];
        let depth = vec![1.0f32; 640 * 480];

        let frame = Frame::new(
            0,
            1234.567,
            color.clone(),
            Some(depth.clone()),
            camera.clone(),
            None,
        );

        assert_eq!(frame.index, 0);
        assert_eq!(frame.timestamp, 1234.567);
        assert_eq!(frame.color.len(), 640 * 480 * 3);
        assert!(frame.has_depth());
        assert!(!frame.has_ground_truth());
        assert_eq!(frame.width, 640);
        assert_eq!(frame.height, 480);
    }

    #[test]
    fn test_kitti_loader_reads_metadata() {
        let dir = tempdir().unwrap();
        let image_dir = dir.path().join("image_0");
        std::fs::create_dir_all(&image_dir).unwrap();
        std::fs::write(
            dir.path().join("calib.txt"),
            "P0: 718.856 0.0 607.1928 0.0 0.0 718.856 185.2157 0.0 0.0 0.0 1.0 0.0\n",
        )
        .unwrap();
        std::fs::write(dir.path().join("poses.txt"), "1 0 0 0 0 1 0 0 0 0 1 0\n").unwrap();
        std::fs::write(image_dir.join("000000.png"), b"dummy").unwrap();

        let dataset = KittiDataset::load(DatasetConfig {
            root_path: dir.path().to_path_buf(),
            load_depth: false,
            load_ground_truth: true,
            max_frames: 0,
            stride: 1,
        })
        .unwrap();

        assert_eq!(dataset.len(), 1);
        assert!(dataset.metadata().has_ground_truth);
        assert_eq!(dataset.camera().width, 1241);
    }

    #[test]
    fn test_euroc_loader_reads_metadata() {
        let dir = tempdir().unwrap();
        let cam0 = dir.path().join("mav0").join("cam0");
        let data_dir = cam0.join("data");
        let gt_dir = dir.path().join("mav0").join("state_groundtruth_estimate0");
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::create_dir_all(&gt_dir).unwrap();
        std::fs::write(
            cam0.join("sensor.yaml"),
            "intrinsics: [458.654, 457.296, 367.215, 248.375]\nresolution: [752, 480]\n",
        )
        .unwrap();
        std::fs::write(
            cam0.join("data.csv"),
            "#timestamp,filename\n1403636579763555584,1403636579763555584.png\n",
        )
        .unwrap();
        std::fs::write(gt_dir.join("data.csv"), "#timestamp,p_RS_R_x,p_RS_R_y,p_RS_R_z,q_RS_w,q_RS_x,q_RS_y,q_RS_z\n1403636579763555584,0,0,0,1,0,0,0\n").unwrap();
        std::fs::write(data_dir.join("1403636579763555584.png"), b"dummy").unwrap();

        let dataset = EurocDataset::load(DatasetConfig {
            root_path: dir.path().to_path_buf(),
            load_depth: false,
            load_ground_truth: true,
            max_frames: 0,
            stride: 1,
        })
        .unwrap();

        assert_eq!(dataset.len(), 1);
        assert!(dataset.metadata().has_ground_truth);
        assert_eq!(dataset.camera().width, 752);
    }
}
