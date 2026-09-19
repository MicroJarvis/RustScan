//! Camera model representation

use crate::config::config::CameraConfig;
use nalgebra::{Matrix3, Vector3};

/// Camera intrinsic parameters
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    /// Focal length in pixels (fx, fy)
    pub focal: Vector3<f32>,
    /// Principal point (cx, cy)
    pub principal: Vector3<f32>,
    /// Image dimensions
    pub width: u32,
    pub height: u32,
    /// Distortion parameters (k1, k2, p1, p2, k3)
    pub distortion: Option<[f32; 5]>,
}

impl Camera {
    /// Create a new camera with given parameters
    pub fn new(fx: f32, fy: f32, cx: f32, cy: f32, width: u32, height: u32) -> Self {
        Self {
            focal: Vector3::new(fx, fy, 1.0),
            principal: Vector3::new(cx, cy, 0.0),
            width,
            height,
            distortion: None,
        }
    }

    /// Create a camera from K matrix.
    ///
    /// Reads `fx = K[(0, 0)]`, `fy = K[(1, 1)]`, and principal components from
    /// `K[(2, 0)]` / `K[(2, 1)]` to match the previous column-vector accessors
    /// `col(0).z` and `col(1).z`.
    pub fn from_k(k: &Matrix3<f32>, width: u32, height: u32) -> Self {
        Self {
            focal: Vector3::new(k[(0, 0)], k[(1, 1)], 1.0),
            principal: Vector3::new(k[(2, 0)], k[(2, 1)], 0.0),
            width,
            height,
            distortion: None,
        }
    }

    /// Create a camera from CameraConfig
    pub fn from_config(config: &CameraConfig) -> Self {
        Self::new(
            config.fx,
            config.fy,
            config.cx,
            config.cy,
            config.width,
            config.height,
        )
    }

    /// Get the intrinsic matrix.
    ///
    /// Layout is `p' = K * p` with `(row, column)` indexing. The previous
    /// `from_cols([fx,0,0], [0,fy,0], [cx,cy,0])` construction left `K[(2, 2)] = 0`.
    pub fn k(&self) -> Matrix3<f32> {
        Matrix3::from_columns(&[
            self.focal.x * Vector3::x(),
            self.focal.y * Vector3::y(),
            self.principal.x * Vector3::x() + self.principal.y * Vector3::y(),
        ])
    }

    /// Project a 3D camera-frame point to pixel coordinates
    pub fn project(&self, point: &Vector3<f32>) -> Option<Vector3<f32>> {
        if point.z <= 0.0 {
            return None;
        }

        let x = point.x / point.z;
        let y = point.y / point.z;

        let px = self.focal.x * x + self.principal.x;
        let py = self.focal.y * y + self.principal.y;

        Some(Vector3::new(px, py, point.z))
    }

    /// Unproject a pixel to 3D ray
    pub fn unproject(&self, pixel: &Vector3<f32>, depth: f32) -> Vector3<f32> {
        let x = (pixel.x - self.principal.x) / self.focal.x;
        let y = (pixel.y - self.principal.y) / self.focal.y;
        Vector3::new(x, y, 1.0) * depth
    }

    /// Check if a pixel is inside the image
    pub fn is_in_image(&self, pixel: &Vector3<f32>, margin: i32) -> bool {
        pixel.x >= margin as f32
            && pixel.x < (self.width as i32 - margin) as f32
            && pixel.y >= margin as f32
            && pixel.y < (self.height as i32 - margin) as f32
    }
}

/// Default camera (640x480, common parameters)
impl Default for Camera {
    fn default() -> Self {
        Self::new(525.0, 525.0, 319.5, 239.5, 640, 480)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn k_matrix_uses_row_column_indexing_for_offset_principal_point() {
        let camera = Camera::new(500.0, 510.0, 321.5, 239.25, 640, 480);
        let k = camera.k();
        assert!((k[(0, 0)] - 500.0).abs() < 1e-6);
        assert!((k[(1, 1)] - 510.0).abs() < 1e-6);
        assert!((k[(0, 2)] - 321.5).abs() < 1e-6);
        assert!((k[(1, 2)] - 239.25).abs() < 1e-6);
        assert!(k[(2, 2)].abs() < 1e-7);

        let from_k = Camera::from_k(&k, 640, 480);
        assert!((from_k.focal.x - 500.0).abs() < 1e-6);
        assert!((from_k.focal.y - 510.0).abs() < 1e-6);
        assert!(from_k.principal.x.abs() < 1e-7);
        assert!(from_k.principal.y.abs() < 1e-7);
    }

    #[test]
    fn project_and_unproject_round_trip_off_axis_point() {
        let camera = Camera::new(500.0, 510.0, 321.5, 239.25, 640, 480);
        let point = Vector3::new(0.4, -1.2, 2.8);
        let projected = camera.project(&point).unwrap();
        let unprojected = camera.unproject(&projected, point.z);
        assert!((unprojected - point).norm() < 1e-5);
    }
}
