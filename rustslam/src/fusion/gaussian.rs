//! 3D Gaussian Splatting for Dense SLAM
//!
//! This module provides the core data structures and operations
//! for 3D Gaussian Splatting based dense reconstruction.
//!
//! References:
//! - RTG-SLAM: Real-time 3D Reconstruction at Scale Using Gaussian Splatting
//! - SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM

use nalgebra::{Matrix3, Point3, UnitQuaternion, Vector3};
use rustscan_types::matrix3_from_column_major_array;

/// A single 3D Gaussian primitive
#[derive(Debug, Clone)]
pub struct Gaussian3D {
    /// Position (mean) of the Gaussian
    pub position: Point3<f32>,
    /// Scale factors (sx, sy, sz)
    pub scale: Vector3<f32>,
    /// Rotation quaternion
    pub rotation: UnitQuaternion<f32>,
    /// Opacity (0-1)
    pub opacity: f32,
    /// Spherical Harmonics coefficients for color
    /// Stored as [DC, C1, C2, ...] per channel (RGB)
    /// For simplicity, we use RGB (3 channels) with DC only for now
    pub color: [f32; 3],
    /// Feature vector (optional, for tracking)
    pub features: Option<Vec<f32>>,
    /// Gaussian state (for optimization)
    pub state: GaussianState,
}

/// State of a Gaussian (for selective optimization)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GaussianState {
    /// Newly created, needs optimization
    New,
    /// Unstable, needs optimization
    Unstable,
    /// Stable, converged
    Stable,
}

impl Gaussian3D {
    /// Create a new Gaussian from depth point
    pub fn from_depth_point(x: f32, y: f32, z: f32, color: [u8; 3]) -> Self {
        Self {
            position: Point3::new(x, y, z),
            scale: Vector3::repeat(0.01), // 1cm default scale
            rotation: UnitQuaternion::identity(),
            opacity: 0.5,
            color: [
                color[0] as f32 / 255.0,
                color[1] as f32 / 255.0,
                color[2] as f32 / 255.0,
            ],
            features: None,
            state: GaussianState::New,
        }
    }

    /// Get the approximate covariance matrix (simplified)
    /// In practice, we use diagonal approximation with scale
    pub fn covariance_diagonal(&self) -> Vector3<f32> {
        // Simplified: use scale as approximate std dev
        self.scale.component_mul(&self.scale)
    }

    /// Project Gaussian to 2D (for rendering)
    /// Returns: (center_x, center_y, radius)
    pub fn project(
        &self,
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        pose: &[[f32; 3]; 3],
        t: &[f32; 3],
    ) -> Option<[f32; 3]> {
        // Transform to camera frame. `pose` is interpreted as column-major to
        // match the previous glam `Mat3::from_cols(row0, row1, row2)` construction.
        let r = matrix3_from_stored_pose_array(pose);

        let translation = Vector3::new(t[0], t[1], t[2]);
        let cam_pos = r.transpose() * (self.position.coords - translation);

        // Behind camera
        if cam_pos.z <= 0.0 {
            return None;
        }

        // Project to image plane
        let u = fx * cam_pos.x / cam_pos.z + cx;
        let v = fy * cam_pos.y / cam_pos.z + cy;

        // Approximate radius using scale
        let radius = (fx + fy) * self.scale.x / cam_pos.z;

        Some([u, v, radius])
    }

    /// Apply opacity (for alpha blending)
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Apply scale
    pub fn with_scale(mut self, scale: Vector3<f32>) -> Self {
        self.scale = scale;
        self
    }
}

impl Default for Gaussian3D {
    fn default() -> Self {
        Self {
            position: Point3::origin(),
            scale: Vector3::repeat(0.01),
            rotation: UnitQuaternion::identity(),
            opacity: 0.5,
            color: [0.5, 0.5, 0.5],
            features: None,
            state: GaussianState::New,
        }
    }
}

/// Interpret a stored `[[f32; 3]; 3]` the way previous glam
/// `Mat3::from_cols(pose[0], pose[1], pose[2])` did: each inner array is a column.
pub(crate) fn matrix3_from_stored_pose_array(pose: &[[f32; 3]; 3]) -> Matrix3<f32> {
    matrix3_from_column_major_array(pose)
}

/// A collection of Gaussians (the scene representation)
#[derive(Debug, Clone)]
pub struct GaussianMap {
    /// All Gaussians in the map
    gaussians: Vec<Gaussian3D>,
    /// Maximum number of Gaussians
    max_gaussians: usize,
    /// Number of stable Gaussians
    stable_count: usize,
}

impl GaussianMap {
    /// Create a new Gaussian map
    pub fn new(max_gaussians: usize) -> Self {
        Self {
            gaussians: Vec::with_capacity(max_gaussians),
            max_gaussians,
            stable_count: 0,
        }
    }

    /// Add a new Gaussian
    pub fn add(&mut self, gaussian: Gaussian3D) -> Option<usize> {
        if self.gaussians.len() >= self.max_gaussians {
            return None;
        }

        let id = self.gaussians.len();
        self.gaussians.push(gaussian);

        // Remove oldest if over capacity
        if self.gaussians.len() > self.max_gaussians {
            self.gaussians.remove(0);
        }

        Some(id)
    }

    /// Add Gaussians from depth frame
    pub fn add_from_depth(
        &mut self,
        depth: &[f32],
        color: &[[u8; 3]],
        width: usize,
        height: usize,
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        pose: &[[f32; 3]; 3],
        t: &[f32; 3],
    ) -> usize {
        let r = matrix3_from_stored_pose_array(pose);
        let translation = Vector3::new(t[0], t[1], t[2]);

        let mut added = 0;

        // Sample points from depth
        for v in (0..height).step_by(2) {
            for u in (0..width).step_by(2) {
                let idx = v * width + u;
                let z = depth[idx];

                // Valid depth
                if z > 0.01 && z < 10.0 {
                    // Backproject to 3D
                    let x = (u as f32 - cx) * z / fx;
                    let y = (v as f32 - cy) * z / fy;

                    // Transform to world frame
                    let cam_pos = Vector3::new(x, y, z);
                    let world_pos = r * cam_pos + translation;

                    let gaussian = Gaussian3D::from_depth_point(
                        world_pos.x,
                        world_pos.y,
                        world_pos.z,
                        color[idx],
                    );

                    if self.add(gaussian).is_some() {
                        added += 1;
                    }
                }
            }
        }

        added
    }

    /// Get Gaussian by index
    pub fn get(&self, index: usize) -> Option<&Gaussian3D> {
        self.gaussians.get(index)
    }

    /// Get mutable Gaussian
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Gaussian3D> {
        self.gaussians.get_mut(index)
    }

    /// Get all Gaussians
    pub fn gaussians(&self) -> &[Gaussian3D] {
        &self.gaussians
    }

    /// Get number of Gaussians
    pub fn len(&self) -> usize {
        self.gaussians.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.gaussians.is_empty()
    }

    /// Get stable Gaussians count
    pub fn stable_count(&self) -> usize {
        self.stable_count
    }

    /// Update Gaussian states (call after optimization)
    pub fn update_states(&mut self) {
        self.stable_count = self
            .gaussians
            .iter()
            .filter(|g| g.state == GaussianState::Stable)
            .count();
    }

    /// Get unstable Gaussians for optimization
    pub fn get_unstable(&self) -> Vec<usize> {
        self.gaussians
            .iter()
            .enumerate()
            .filter(|(_, g)| g.state == GaussianState::Unstable || g.state == GaussianState::New)
            .map(|(i, _)| i)
            .collect()
    }

    /// Retain gaussians matching the predicate and return number removed.
    pub fn retain<F>(&mut self, mut keep: F) -> usize
    where
        F: FnMut(&Gaussian3D) -> bool,
    {
        let before = self.gaussians.len();
        self.gaussians.retain(|g| keep(g));
        self.update_states();
        before.saturating_sub(self.gaussians.len())
    }

    /// Clear all Gaussians
    pub fn clear(&mut self) {
        self.gaussians.clear();
        self.stable_count = 0;
    }
}

impl Default for GaussianMap {
    fn default() -> Self {
        Self::new(100_000) // 100k Gaussians default
    }
}

/// Camera pose for rendering
#[derive(Debug, Clone)]
pub struct GaussianCamera {
    /// Camera intrinsics
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    /// Camera pose (rotation matrix + translation)
    pub rotation: [[f32; 3]; 3],
    pub translation: [f32; 3],
}

impl GaussianCamera {
    /// Create new camera
    pub fn new(fx: f32, fy: f32, cx: f32, cy: f32) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
        }
    }

    /// Set pose
    pub fn with_pose(mut self, rotation: [[f32; 3]; 3], translation: [f32; 3]) -> Self {
        self.rotation = rotation;
        self.translation = translation;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SE3;
    use nalgebra::{Point3, Vector3};
    use rustscan_types::{
        matrix3_from_column_major_array, matrix3_to_column_major_array, Rotation3f,
    };

    #[test]
    fn test_gaussian_creation() {
        let g = Gaussian3D::from_depth_point(0.0, 0.0, 1.0, [255, 128, 64]);
        assert!((g.position - Point3::new(0.0, 0.0, 1.0)).norm() < 0.001);
        assert!(g.color[0] > 0.9);
    }

    #[test]
    fn test_gaussian_projection() {
        let g = Gaussian3D::from_depth_point(0.0, 0.0, 1.0, [255, 128, 64]);
        let camera = GaussianCamera::new(500.0, 500.0, 320.0, 240.0);

        // Camera at origin looking at +Z
        let result = g.project(
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &[0.0, 0.0, 0.0],
        );

        assert!(result.is_some());
        let [u, v, _] = result.unwrap();
        assert!((u - 320.0).abs() < 0.1);
        assert!((v - 240.0).abs() < 0.1);
    }

    #[test]
    fn stored_pose_array_round_trips_non_symmetric_rotation() {
        let rotation = Rotation3f::from_euler_angles(0.3, -0.4, 0.5);
        let translation = Vector3::new(1.25, -2.5, 3.75);
        let pose = SE3::from_quat_translation(rotation, translation);
        let stored = pose.rotation_matrix();
        let matrix = matrix3_from_stored_pose_array(&stored);
        let round_trip = matrix3_to_column_major_array(matrix);

        for col in 0..3 {
            for row in 0..3 {
                assert!((round_trip[col][row] - stored[col][row]).abs() < 1e-6);
            }
        }

        let point = Point3::new(0.4, -1.2, 2.8);
        let cam_pos = matrix.transpose() * (point.coords - translation);
        let expected =
            matrix3_from_column_major_array(&stored).transpose() * (point.coords - translation);
        assert!((cam_pos - expected).norm() < 1e-5);
    }

    #[test]
    fn test_gaussian_map() {
        let mut map = GaussianMap::new(100);
        assert!(map.is_empty());

        let g = Gaussian3D::default();
        map.add(g);

        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_gaussian_map_from_depth() {
        let mut map = GaussianMap::new(1000);

        // Simple 4x4 depth frame
        let depth = [1.0f32; 16];
        let color = [[255u8, 255, 255]; 16];

        let added = map.add_from_depth(
            &depth,
            &color,
            4,
            4,
            500.0,
            500.0,
            320.0,
            240.0,
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &[0.0, 0.0, 0.0],
        );

        assert!(added > 0);
    }
}
