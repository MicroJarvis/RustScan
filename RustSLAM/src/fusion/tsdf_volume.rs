//! TSDF Volume for Mesh Extraction from 3DGS
//!
//! Implements Truncated Signed Distance Function (TSDF) volume fusion
//! for extracting mesh from Gaussian Splatting representations.
//!
//! Based on:
//! - Kinect Fusion algorithm
//! - PGSR (Planar-based Gaussian Splatting Reconstruction)

use nalgebra::{Matrix4, Point3, Vector3};
use std::collections::HashMap;

/// A 3D voxel cell in the TSDF volume
#[derive(Debug, Clone)]
pub struct Voxel {
    /// TSDF value: -1 (inside) to 1 (outside), 0 = surface
    pub tsdf: f32,
    /// Weighted sum for averaging
    pub weight: f32,
    /// RGB color (optional, for visualization)
    pub color: [f32; 3],
    /// Color weight
    pub color_weight: f32,
}

impl Default for Voxel {
    fn default() -> Self {
        Self {
            tsdf: 1.0,
            weight: 0.0,
            color: [0.0, 0.0, 0.0],
            color_weight: 0.0,
        }
    }
}

/// TSDF Volume configuration
#[derive(Debug, Clone)]
pub struct TsdfConfig {
    /// Voxel size in world units (e.g., 0.01 for 1cm)
    pub voxel_size: f32,
    /// Truncation distance (typically 3-5 * voxel_size)
    pub sdf_trunc: f32,
    /// Volume bounds
    pub min_bound: Point3<f32>,
    pub max_bound: Point3<f32>,
    /// Maximum weight per voxel
    pub max_weight: f32,
    /// Integration weight
    pub integration_weight: f32,
}

impl Default for TsdfConfig {
    fn default() -> Self {
        Self {
            voxel_size: 0.01, // 1cm voxels
            sdf_trunc: 0.03,  // 3cm truncation
            min_bound: Point3::new(-1.0, -1.0, -1.0),
            max_bound: Point3::new(1.0, 1.0, 1.0),
            max_weight: 100.0,
            integration_weight: 1.0,
        }
    }
}

/// TSDF Volume for volumetric fusion
#[derive(Debug)]
pub struct TsdfVolume {
    /// Configuration
    config: TsdfConfig,
    /// Voxel grid: key = (x, y, z) index, value = voxel data
    voxels: HashMap<(i32, i32, i32), Voxel>,
    /// Volume dimensions
    volume_dims: (i32, i32, i32),
    /// Number of integrated frames
    frame_count: usize,
}

impl TsdfVolume {
    /// Create a new TSDF volume from bounds
    pub fn new(config: TsdfConfig) -> Self {
        let dx = ((config.max_bound.x - config.min_bound.x) / config.voxel_size).ceil() as i32;
        let dy = ((config.max_bound.y - config.min_bound.y) / config.voxel_size).ceil() as i32;
        let dz = ((config.max_bound.z - config.min_bound.z) / config.voxel_size).ceil() as i32;

        Self {
            config,
            voxels: HashMap::new(),
            volume_dims: (dx, dy, dz),
            frame_count: 0,
        }
    }

    /// Create volume centered at origin with given size
    pub fn centered(size_meters: f32, voxel_size: f32) -> Self {
        let half = size_meters / 2.0;
        let config = TsdfConfig {
            voxel_size,
            sdf_trunc: voxel_size * 3.0,
            min_bound: Point3::new(-half, -half, -half),
            max_bound: Point3::new(half, half, half),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Convert world position to voxel index
    pub fn world_to_voxel(&self, pos: Point3<f32>) -> Option<(i32, i32, i32)> {
        let x = ((pos.x - self.config.min_bound.x) / self.config.voxel_size).floor() as i32;
        let y = ((pos.y - self.config.min_bound.y) / self.config.voxel_size).floor() as i32;
        let z = ((pos.z - self.config.min_bound.z) / self.config.voxel_size).floor() as i32;

        if x >= 0
            && x < self.volume_dims.0
            && y >= 0
            && y < self.volume_dims.1
            && z >= 0
            && z < self.volume_dims.2
        {
            Some((x, y, z))
        } else {
            None
        }
    }

    /// Convert voxel index to world position (center of voxel)
    pub fn voxel_to_world(&self, ix: i32, iy: i32, iz: i32) -> Point3<f32> {
        Point3::new(
            self.config.min_bound.x + (ix as f32 + 0.5) * self.config.voxel_size,
            self.config.min_bound.y + (iy as f32 + 0.5) * self.config.voxel_size,
            self.config.min_bound.z + (iz as f32 + 0.5) * self.config.voxel_size,
        )
    }

    /// Get voxel at index (create if not exists)
    pub fn get_voxel_mut(&mut self, ix: i32, iy: i32, iz: i32) -> &mut Voxel {
        self.voxels
            .entry((ix, iy, iz))
            .or_insert_with(Voxel::default)
    }

    /// Get voxel at index (read only)
    pub fn get_voxel(&self, ix: i32, iy: i32, iz: i32) -> Option<&Voxel> {
        self.voxels.get(&(ix, iy, iz))
    }

    /// Integrate a depth map into the TSDF volume
    ///
    /// # Arguments
    /// * `depth` - Depth map (in meters)
    /// * `color` - RGB color map (optional, can be None)
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `intrinsics` - Camera intrinsics [fx, fy, cx, cy]
    /// * `extrinsics` - Camera pose as 4x4 matrix (camera to world)
    pub fn integrate(
        &mut self,
        depth: &[f32],
        color: Option<&[[u8; 3]]>,
        width: usize,
        height: usize,
        intrinsics: [f32; 4],
        extrinsics: &Matrix4<f32>,
    ) {
        let depth_fn = |idx: usize| -> f32 { depth.get(idx).copied().unwrap_or(0.0) };
        let color_fn =
            color.map(|colors| move |idx: usize| -> Option<[u8; 3]> { colors.get(idx).copied() });
        self.integrate_depth_map(
            &depth_fn,
            color_fn
                .as_ref()
                .map(|f| f as &dyn Fn(usize) -> Option<[u8; 3]>),
            width,
            height,
            intrinsics,
            extrinsics,
        );
    }

    /// Integrate from Gaussian rendering (project Gaussians to depth)
    ///
    /// This is the key integration for 3DGS -> Mesh
    pub fn integrate_from_gaussians<F>(
        &mut self,
        get_depth_at: F,
        get_color_at: Option<&dyn Fn(usize) -> Option<[u8; 3]>>,
        width: usize,
        height: usize,
        intrinsics: [f32; 4],
        extrinsics: &Matrix4<f32>,
    ) where
        F: Fn(usize) -> f32,
    {
        self.integrate_depth_map(
            &get_depth_at,
            get_color_at,
            width,
            height,
            intrinsics,
            extrinsics,
        );
    }

    fn integrate_depth_map(
        &mut self,
        get_depth_at: &dyn Fn(usize) -> f32,
        get_color_at: Option<&dyn Fn(usize) -> Option<[u8; 3]>>,
        width: usize,
        height: usize,
        intrinsics: [f32; 4],
        extrinsics: &Matrix4<f32>,
    ) {
        let [fx, fy, cx, cy] = intrinsics;
        let trunc = self.config.sdf_trunc.max(self.config.voxel_size);
        let step = self.config.voxel_size.max(1e-6);

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let z = get_depth_at(idx);

                if z <= 0.001 || z > 10.0 {
                    continue;
                }

                let ray_dir = Vector3::new((x as f32 - cx) / fx, (y as f32 - cy) / fy, 1.0);
                let mut t = (z - trunc).max(0.0);
                let t_end = z + trunc;
                let mut last_voxel: Option<(i32, i32, i32)> = None;

                while t <= t_end {
                    let cam_pos = Point3::from(ray_dir * t);
                    let world_pos = extrinsics.transform_point(&cam_pos);
                    if let Some((vx, vy, vz)) = self.world_to_voxel(world_pos) {
                        if last_voxel != Some((vx, vy, vz)) {
                            let sdf = z - t;
                            let tsdf = (sdf / trunc).clamp(-1.0, 1.0);

                            let integration_weight = self.config.integration_weight;
                            if integration_weight <= 0.0 {
                                continue;
                            }
                            let max_weight = self.config.max_weight;
                            let voxel = self.get_voxel_mut(vx, vy, vz);
                            let tsdf_denom = (voxel.weight + integration_weight).max(1e-8);
                            voxel.tsdf = (voxel.tsdf * voxel.weight + tsdf * integration_weight)
                                / tsdf_denom;
                            voxel.weight = (voxel.weight + integration_weight).min(max_weight);

                            if let Some(color_fn) = get_color_at {
                                if let Some(c) = color_fn(idx) {
                                    let cr = c[0] as f32 / 255.0;
                                    let cg = c[1] as f32 / 255.0;
                                    let cb = c[2] as f32 / 255.0;
                                    let color_denom =
                                        (voxel.color_weight + integration_weight).max(1e-8);
                                    voxel.color = [
                                        (voxel.color[0] * voxel.color_weight
                                            + cr * integration_weight)
                                            / color_denom,
                                        (voxel.color[1] * voxel.color_weight
                                            + cg * integration_weight)
                                            / color_denom,
                                        (voxel.color[2] * voxel.color_weight
                                            + cb * integration_weight)
                                            / color_denom,
                                    ];
                                    voxel.color_weight += integration_weight;
                                }
                            }

                            last_voxel = Some((vx, vy, vz));
                        }
                    }
                    t += step;
                }
            }
        }

        self.frame_count += 1;
    }

    /// Get all voxels that are near the surface (|tsdf| < threshold)
    pub fn get_surface_voxels(&self, threshold: f32) -> Vec<(i32, i32, i32, &Voxel)> {
        self.voxels
            .iter()
            .filter(|(_, v)| v.weight > 0.0 && v.tsdf.abs() < threshold)
            .map(|((x, y, z), v)| (*x, *y, *z, v))
            .collect()
    }

    /// Get voxel grid dimensions
    pub fn dimensions(&self) -> (i32, i32, i32) {
        self.volume_dims
    }

    /// Get configuration
    pub fn config(&self) -> &TsdfConfig {
        &self.config
    }

    /// Get frame count
    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    /// Get number of active voxels
    pub fn num_voxels(&self) -> usize {
        self.voxels.len()
    }

    /// Iterate over all active voxels (sparse iteration)
    pub fn voxels_iter(&self) -> impl Iterator<Item = ((i32, i32, i32), &Voxel)> {
        self.voxels.iter().map(|(k, v)| (*k, v))
    }

    /// Get active voxel positions (for sparse marching cubes)
    pub fn active_voxel_positions(&self) -> Vec<(i32, i32, i32)> {
        self.voxels.keys().copied().collect()
    }

    /// Clear the volume
    pub fn clear(&mut self) {
        self.voxels.clear();
        self.frame_count = 0;
    }
}

impl Default for TsdfVolume {
    fn default() -> Self {
        Self::new(TsdfConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SE3;
    use nalgebra::{Point3, UnitQuaternion, Vector3};

    #[test]
    fn test_voxel_creation() {
        let config = TsdfConfig::default();
        let volume = TsdfVolume::new(config);

        assert_eq!(volume.dimensions(), (200, 200, 200));
    }

    #[test]
    fn test_world_voxel_conversion() {
        let volume = TsdfVolume::centered(1.0, 0.01);

        let pos = Point3::new(0.0, 0.0, 0.0);
        let voxel = volume.world_to_voxel(pos);
        assert!(voxel.is_some());

        let (vx, vy, vz) = voxel.unwrap();
        let world = volume.voxel_to_world(vx, vy, vz);

        assert!((world - pos).norm() < 0.01);
    }

    #[test]
    fn test_integrate_simple() {
        let mut volume = TsdfVolume::centered(0.5, 0.01);

        // Simple depth map: 10x10, all at z=1.0
        let depth = vec![1.0f32; 10 * 10];
        let intrinsics = [500.0, 500.0, 5.0, 5.0];
        let extrinsics = Matrix4::identity();

        volume.integrate(&depth, None, 10, 10, intrinsics, &extrinsics);

        assert!(volume.frame_count() > 0);
    }

    #[test]
    fn test_integrate_updates_multiple_voxels() {
        let mut volume = TsdfVolume::centered(0.5, 0.05);
        let depth = vec![0.2f32; 1];
        let intrinsics = [1.0, 1.0, 0.0, 0.0];
        let extrinsics = Matrix4::identity();

        volume.integrate(&depth, None, 1, 1, intrinsics, &extrinsics);

        assert!(volume.num_voxels() > 1);
    }

    #[test]
    fn extrinsics_match_se3_homogeneous_for_non_symmetric_pose() {
        let pose = SE3::from_quat_translation(
            UnitQuaternion::from_euler_angles(0.3, -0.4, 0.5),
            Vector3::new(1.25, -2.5, 3.75),
        );
        let matrix = pose.to_homogeneous_matrix();
        let point = Point3::new(0.4, -1.2, 2.8);
        let from_pose = pose.transform_point3(point);
        let from_matrix = matrix.transform_point(&point);
        assert!((from_pose - from_matrix).norm() < 1e-5);

        let rows = pose.to_row_major_array();
        let from_rows = rustscan_types::matrix4_from_row_major_array(&rows);
        assert!((from_rows.transform_point(&point) - from_pose).norm() < 1e-5);
    }
}
