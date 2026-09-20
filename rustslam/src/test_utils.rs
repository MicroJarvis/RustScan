//! Test utilities for RustSLAM
//!
//! This module provides utilities for generating synthetic test data:
//! - TSDF volumes (sphere, cube)
//! - Mock video frames
//! - Synthetic camera poses and depth maps
//!
//! # Examples
//!
//! ```
//! use rustslam::test_utils::*;
//! use nalgebra::Point3;
//!
//! let tsdf = create_sphere_tsdf(Point3::origin(), 1.0, 0.1);
//! let frame = create_mock_frame(640, 480, FramePattern::Checkerboard);
//! ```

use crate::core::pose::SE3;
use crate::fusion::tsdf_volume::{TsdfConfig, TsdfVolume};
use nalgebra::{Point3, Vector3};
use std::f32::consts::PI;

/// Frame pattern for mock video frames
#[derive(Debug, Clone, Copy)]
pub enum FramePattern {
    /// Solid color (gray)
    Solid,
    /// Checkerboard pattern
    Checkerboard,
    /// Horizontal gradient
    Gradient,
}

/// Trajectory type for synthetic camera poses
#[derive(Debug, Clone, Copy)]
pub enum TrajectoryType {
    /// Circular trajectory around origin
    Circle,
    /// Linear trajectory along X axis
    Line,
    /// Random poses
    Random,
}

/// Depth pattern for synthetic depth maps
#[derive(Debug, Clone, Copy)]
pub enum DepthPattern {
    /// Constant depth
    Constant,
    /// Planar surface
    Planar,
    /// Spherical surface
    Spherical,
}

/// Mock video frame structure
#[derive(Debug, Clone)]
pub struct MockVideoFrame {
    /// RGB data (width * height * 3 bytes)
    pub rgb: Vec<u8>,
    /// Frame width in pixels
    pub width: u32,
    /// Frame height in pixels
    pub height: u32,
    /// Timestamp in seconds
    pub timestamp: f64,
}

/// Create a synthetic sphere TSDF volume
///
/// # Arguments
/// * `center` - Center position of the sphere
/// * `radius` - Radius of the sphere
/// * `voxel_size` - Size of each voxel
///
/// # Returns
/// A TSDF volume with a sphere signed distance field
pub fn create_sphere_tsdf(center: Point3<f32>, radius: f32, voxel_size: f32) -> TsdfVolume {
    let size = radius * 3.0; // Volume size to contain sphere
    let config = TsdfConfig {
        voxel_size,
        sdf_trunc: voxel_size * 4.0,
        min_bound: center - Vector3::repeat(size / 2.0),
        max_bound: center + Vector3::repeat(size / 2.0),
        max_weight: 100.0,
        integration_weight: 1.0,
    };

    let mut volume = TsdfVolume::new(config);

    // Fill volume with sphere SDF
    let dims = volume.dimensions();
    for x in 0..dims.0 {
        for y in 0..dims.1 {
            for z in 0..dims.2 {
                let pos = volume.voxel_to_world(x, y, z);
                let dist = (pos - center).norm() - radius;
                let tsdf = dist.clamp(-voxel_size * 4.0, voxel_size * 4.0) / (voxel_size * 4.0);

                let voxel = volume.get_voxel_mut(x, y, z);
                voxel.tsdf = tsdf;
                voxel.weight = 1.0;
                voxel.color = [0.8, 0.8, 0.8];
                voxel.color_weight = 1.0;
            }
        }
    }

    volume
}

/// Create a synthetic cube TSDF volume
///
/// # Arguments
/// * `center` - Center position of the cube
/// * `size` - Side length of the cube
/// * `voxel_size` - Size of each voxel
///
/// # Returns
/// A TSDF volume with a cube signed distance field
pub fn create_cube_tsdf(center: Point3<f32>, size: f32, voxel_size: f32) -> TsdfVolume {
    let volume_size = size * 2.0; // Volume size to contain cube
    let config = TsdfConfig {
        voxel_size,
        sdf_trunc: voxel_size * 4.0,
        min_bound: center - Vector3::repeat(volume_size / 2.0),
        max_bound: center + Vector3::repeat(volume_size / 2.0),
        max_weight: 100.0,
        integration_weight: 1.0,
    };

    let mut volume = TsdfVolume::new(config);

    // Fill volume with cube SDF
    let half_size = size / 2.0;
    let dims = volume.dimensions();
    for x in 0..dims.0 {
        for y in 0..dims.1 {
            for z in 0..dims.2 {
                let pos = volume.voxel_to_world(x, y, z);
                let local = pos - center;

                // Cube SDF: max distance to faces
                let dx = local.x.abs() - half_size;
                let dy = local.y.abs() - half_size;
                let dz = local.z.abs() - half_size;

                let dist = dx.max(dy).max(dz);
                let tsdf = dist.clamp(-voxel_size * 4.0, voxel_size * 4.0) / (voxel_size * 4.0);

                let voxel = volume.get_voxel_mut(x, y, z);
                voxel.tsdf = tsdf;
                voxel.weight = 1.0;
                voxel.color = [0.8, 0.8, 0.8];
                voxel.color_weight = 1.0;
            }
        }
    }

    volume
}

/// Create a mock video frame with specified pattern
///
/// # Arguments
/// * `width` - Frame width in pixels
/// * `height` - Frame height in pixels
/// * `pattern` - Pattern type for the frame
///
/// # Returns
/// A mock video frame with RGB data
pub fn create_mock_frame(width: u32, height: u32, pattern: FramePattern) -> MockVideoFrame {
    let size = (width * height * 3) as usize;
    let mut rgb = vec![0u8; size];

    match pattern {
        FramePattern::Solid => {
            // Fill with gray (128, 128, 128)
            for pixel in rgb.chunks_mut(3) {
                pixel[0] = 128;
                pixel[1] = 128;
                pixel[2] = 128;
            }
        }
        FramePattern::Checkerboard => {
            // 8x8 checkerboard pattern
            let block_size = 8;
            for y in 0..height {
                for x in 0..width {
                    let bx = x / block_size;
                    let by = y / block_size;
                    let is_white = (bx + by) % 2 == 0;
                    let color = if is_white { 255 } else { 0 };

                    let idx = ((y * width + x) * 3) as usize;
                    rgb[idx] = color;
                    rgb[idx + 1] = color;
                    rgb[idx + 2] = color;
                }
            }
        }
        FramePattern::Gradient => {
            // Horizontal gradient from black to white
            for y in 0..height {
                for x in 0..width {
                    let intensity = (x as f32 / width as f32 * 255.0) as u8;
                    let idx = ((y * width + x) * 3) as usize;
                    rgb[idx] = intensity;
                    rgb[idx + 1] = intensity;
                    rgb[idx + 2] = intensity;
                }
            }
        }
    }

    MockVideoFrame {
        rgb,
        width,
        height,
        timestamp: 0.0,
    }
}

/// Create synthetic camera poses along a trajectory
///
/// # Arguments
/// * `count` - Number of poses to generate
/// * `trajectory` - Type of trajectory
///
/// # Returns
/// Vector of SE3 poses
pub fn create_synthetic_poses(count: usize, trajectory: TrajectoryType) -> Vec<SE3> {
    let mut poses = Vec::with_capacity(count);

    match trajectory {
        TrajectoryType::Circle => {
            // Circular trajectory around origin
            let radius = 2.0;
            for i in 0..count {
                let angle = 2.0 * PI * (i as f32) / (count as f32);
                let x = radius * angle.cos();
                let z = radius * angle.sin();

                // Look at origin
                let translation = [x, 0.0, z];
                let rotation = [0.0, angle + PI / 2.0, 0.0, 1.0]; // Simplified rotation

                poses.push(SE3::new(&rotation, &translation));
            }
        }
        TrajectoryType::Line => {
            // Linear trajectory along X axis
            for i in 0..count {
                let x = (i as f32) * 0.1;
                let translation = [x, 0.0, 0.0];
                let rotation = [0.0, 0.0, 0.0, 1.0]; // Identity rotation

                poses.push(SE3::new(&rotation, &translation));
            }
        }
        TrajectoryType::Random => {
            // Random poses (simplified - not truly random without rand crate)
            for i in 0..count {
                let x = ((i * 17) % 100) as f32 / 50.0 - 1.0;
                let y = ((i * 31) % 100) as f32 / 50.0 - 1.0;
                let z = ((i * 47) % 100) as f32 / 50.0 - 1.0;

                let translation = [x, y, z];
                let rotation = [0.0, 0.0, 0.0, 1.0];

                poses.push(SE3::new(&rotation, &translation));
            }
        }
    }

    poses
}

/// Create synthetic depth map with specified pattern
///
/// # Arguments
/// * `width` - Depth map width
/// * `height` - Depth map height
/// * `pattern` - Depth pattern type
///
/// # Returns
/// Vector of depth values (width * height)
pub fn create_synthetic_depth(width: u32, height: u32, pattern: DepthPattern) -> Vec<f32> {
    let size = (width * height) as usize;
    let mut depth = vec![0.0f32; size];

    match pattern {
        DepthPattern::Constant => {
            // Constant depth of 1.0 meter
            depth.fill(1.0);
        }
        DepthPattern::Planar => {
            // Planar surface tilted slightly
            for y in 0..height {
                for x in 0..width {
                    let nx = (x as f32 / width as f32) - 0.5;
                    let ny = (y as f32 / height as f32) - 0.5;
                    let d = 1.0 + nx * 0.2 + ny * 0.2;
                    depth[(y * width + x) as usize] = d;
                }
            }
        }
        DepthPattern::Spherical => {
            // Spherical surface
            let cx = width as f32 / 2.0;
            let cy = height as f32 / 2.0;
            let radius = width.min(height) as f32 / 3.0;

            for y in 0..height {
                for x in 0..width {
                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    let r = (dx * dx + dy * dy).sqrt();

                    if r < radius {
                        let d = 1.0 + (1.0 - (r / radius).powi(2)).sqrt() * 0.5;
                        depth[(y * width + x) as usize] = d;
                    } else {
                        depth[(y * width + x) as usize] = 1.0;
                    }
                }
            }
        }
    }

    depth
}
