//! MapPoint representation

use nalgebra::{Point3, Vector3};

/// A 3D map point
#[derive(Debug, Clone)]
pub struct MapPoint {
    /// Unique ID
    pub id: u64,
    /// 3D position in world frame
    pub position: Point3<f32>,
    /// Normal direction (optional)
    pub normal: Option<Vector3<f32>>,
    /// Optional color (normalized RGB, used for Gaussian initialization)
    pub color: Option<[f32; 3]>,
    /// Reference keyframe ID
    pub reference_kf: u64,
    /// Number of observed from keyframes
    pub observations: u32,
    /// Is outlier
    pub is_outlier: bool,
}

impl MapPoint {
    /// Create a new map point
    pub fn new(id: u64, position: Point3<f32>, reference_kf: u64) -> Self {
        Self {
            id,
            position,
            normal: None,
            color: None,
            reference_kf,
            observations: 1,
            is_outlier: false,
        }
    }

    /// Add an observation
    pub fn add_observation(&mut self) {
        self.observations += 1;
    }

    /// Mark as outlier
    pub fn mark_outlier(&mut self) {
        self.is_outlier = true;
    }

    /// Mark as inlier
    pub fn mark_inlier(&mut self) {
        self.is_outlier = false;
    }

    /// Set normal direction
    pub fn set_normal(&mut self, normal: Vector3<f32>) {
        self.normal = Some(normal);
    }

    /// Set color from normalized RGB
    pub fn set_color(&mut self, color: [f32; 3]) {
        self.color = Some([
            color[0].clamp(0.0, 1.0),
            color[1].clamp(0.0, 1.0),
            color[2].clamp(0.0, 1.0),
        ]);
    }

    /// Set color from 8-bit RGB
    pub fn set_color_u8(&mut self, color: [u8; 3]) {
        self.set_color([
            color[0] as f32 / 255.0,
            color[1] as f32 / 255.0,
            color[2] as f32 / 255.0,
        ]);
    }
}
