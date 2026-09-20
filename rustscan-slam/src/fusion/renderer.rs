//! Differentiable Gaussian Renderer
//!
//! Implements the core rendering pipeline for 3D Gaussian Splatting.
//! Based on:
//! - "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
//! - RTG-SLAM: Real-time 3D Reconstruction

use crate::fusion::gaussian::{
    matrix3_from_stored_pose_array, Gaussian3D, GaussianCamera, GaussianMap,
};
use nalgebra::Vector3;

/// Output of rendering
#[derive(Debug, Clone)]
pub struct RenderOutput {
    /// Rendered color image (RGB)
    pub color: Vec<u8>,
    /// Rendered depth image
    pub depth: Vec<f32>,
    /// Rendered normal (optional)
    pub normal: Option<Vec<[f32; 3]>>,
    /// Image dimensions
    pub width: usize,
    pub height: usize,
}

/// Differentiable Gaussian Renderer
pub struct GaussianRenderer {
    /// Image width
    width: usize,
    /// Image height
    height: usize,
    /// Background color (RGB)
    background: [f32; 3],
}

impl GaussianRenderer {
    /// Create a new renderer
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            background: [0.0, 0.0, 0.0],
        }
    }

    /// Set background color
    pub fn with_background(mut self, r: f32, g: f32, b: f32) -> Self {
        self.background = [r, g, b];
        self
    }

    /// Render the Gaussian map from a camera view
    pub fn render(&self, map: &GaussianMap, camera: &GaussianCamera) -> RenderOutput {
        let mut color = vec![0u8; self.width * self.height * 3];
        let mut depth = vec![0.0f32; self.width * self.height];

        // Project all Gaussians and compute camera-space depth
        let mut gaussians_with_depth: Vec<(&Gaussian3D, f32, f32, f32, f32)> = map
            .gaussians()
            .iter()
            .filter_map(|g| {
                if let Some([ux, uy, radius]) = g.project(
                    camera.fx,
                    camera.fy,
                    camera.cx,
                    camera.cy,
                    &camera.rotation,
                    &camera.translation,
                ) {
                    // Compute camera-space depth from the rotation and translation
                    let r = matrix3_from_stored_pose_array(&camera.rotation);
                    let t = Vector3::from(camera.translation);
                    let cam_pos = r.transpose() * (g.position.coords - t);
                    let cam_depth = cam_pos.z;
                    if cam_depth > 0.0 && cam_depth < 100.0 {
                        Some((g, cam_depth, ux, uy, radius))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        // Sort by camera-space depth (far to near for alpha blending)
        gaussians_with_depth
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Render each Gaussian
        for (gaussian, cam_depth, ux, uy, radius) in gaussians_with_depth {
            let gc = [
                (gaussian.color[0].clamp(0.0, 1.0) * 255.0) as u8,
                (gaussian.color[1].clamp(0.0, 1.0) * 255.0) as u8,
                (gaussian.color[2].clamp(0.0, 1.0) * 255.0) as u8,
            ];
            self.render_gaussian(
                &mut color, &mut depth, cam_depth, gc, ux as i32, uy as i32, radius,
            );
        }

        RenderOutput {
            color,
            depth,
            normal: None,
            width: self.width,
            height: self.height,
        }
    }

    /// Render a single Gaussian as a filled circle using camera-space depth
    fn render_gaussian(
        &self,
        color: &mut [u8],
        depth: &mut [f32],
        cam_depth: f32,
        gc: [u8; 3],
        cx: i32,
        cy: i32,
        radius: f32,
    ) {
        let radius = radius.max(1.0);
        let r_sq = radius * radius;

        let min_x = ((cx as f32 - radius) as i32).max(0);
        let max_x = ((cx as f32 + radius) as i32).min(self.width as i32 - 1);
        let min_y = ((cy as f32 - radius) as i32).max(0);
        let max_y = ((cy as f32 + radius) as i32).min(self.height as i32 - 1);

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let dx = x as f32 - cx as f32;
                let dy = y as f32 - cy as f32;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq <= r_sq {
                    let idx = y as usize * self.width + x as usize;

                    // Keep minimum camera-space depth (nearest surface wins)
                    if color[idx * 3] == 0 || cam_depth < depth[idx] {
                        color[idx * 3] = gc[0];
                        color[idx * 3 + 1] = gc[1];
                        color[idx * 3 + 2] = gc[2];
                        depth[idx] = cam_depth;
                    }
                }
            }
        }
    }

    /// Render depth only (for TSDF integration)
    ///
    /// Returns a depth map in camera-space (z-distance from camera center).
    pub fn render_depth(&self, map: &GaussianMap, camera: &GaussianCamera) -> Vec<f32> {
        let mut depth = vec![0.0f32; self.width * self.height];

        let r = matrix3_from_stored_pose_array(&camera.rotation);
        let t = Vector3::from(camera.translation);

        // Project all Gaussians and compute camera-space depth
        let mut gaussians_projected: Vec<(&Gaussian3D, f32, f32, f32, f32)> = map
            .gaussians()
            .iter()
            .filter_map(|g| {
                if let Some([ux, uy, radius]) = g.project(
                    camera.fx,
                    camera.fy,
                    camera.cx,
                    camera.cy,
                    &camera.rotation,
                    &camera.translation,
                ) {
                    let cam_pos = r.transpose() * (g.position.coords - t);
                    let cam_depth = cam_pos.z;
                    if cam_depth > 0.001 && cam_depth < 100.0 {
                        Some((g, cam_depth, ux, uy, radius))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        // Sort front-to-back so nearest depth wins when writing
        gaussians_projected
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (_gaussian, cam_depth, ux, uy, radius) in gaussians_projected {
            self.render_depth_circle(&mut depth, cam_depth, ux as i32, uy as i32, radius);
        }

        depth
    }

    /// Render depth and color simultaneously (for TSDF integration with color)
    ///
    /// Returns (depth_map, color_map) where color is [u8; 3] per pixel.
    pub fn render_depth_and_color(
        &self,
        map: &GaussianMap,
        camera: &GaussianCamera,
    ) -> (Vec<f32>, Vec<[u8; 3]>) {
        let mut depth = vec![0.0f32; self.width * self.height];
        let mut color = vec![[0u8; 3]; self.width * self.height];

        let r = matrix3_from_stored_pose_array(&camera.rotation);
        let t = Vector3::from(camera.translation);

        let mut gaussians_projected: Vec<(&Gaussian3D, f32, f32, f32, f32)> = map
            .gaussians()
            .iter()
            .filter_map(|g| {
                if let Some([ux, uy, radius]) = g.project(
                    camera.fx,
                    camera.fy,
                    camera.cx,
                    camera.cy,
                    &camera.rotation,
                    &camera.translation,
                ) {
                    let cam_pos = r.transpose() * (g.position.coords - t);
                    let cam_depth = cam_pos.z;
                    if cam_depth > 0.001 && cam_depth < 100.0 {
                        Some((g, cam_depth, ux, uy, radius))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        // Sort front-to-back so nearest depth wins
        gaussians_projected
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (gaussian, cam_depth, ux, uy, radius) in gaussians_projected {
            let gc = [
                (gaussian.color[0].clamp(0.0, 1.0) * 255.0) as u8,
                (gaussian.color[1].clamp(0.0, 1.0) * 255.0) as u8,
                (gaussian.color[2].clamp(0.0, 1.0) * 255.0) as u8,
            ];
            self.render_depth_color_circle(
                &mut depth, &mut color, cam_depth, gc, ux as i32, uy as i32, radius,
            );
        }

        (depth, color)
    }

    /// Render depth and color as circle
    fn render_depth_color_circle(
        &self,
        depth: &mut [f32],
        color: &mut [[u8; 3]],
        z: f32,
        gc: [u8; 3],
        cx: i32,
        cy: i32,
        radius: f32,
    ) {
        let radius = radius.max(1.0);
        let r_sq = radius * radius;

        let min_x = ((cx as f32 - radius) as i32).max(0);
        let max_x = ((cx as f32 + radius) as i32).min(self.width as i32 - 1);
        let min_y = ((cy as f32 - radius) as i32).max(0);
        let max_y = ((cy as f32 + radius) as i32).min(self.height as i32 - 1);

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let dx = x as f32 - cx as f32;
                let dy = y as f32 - cy as f32;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq <= r_sq {
                    let idx = y as usize * self.width + x as usize;
                    if depth[idx] == 0.0 || z < depth[idx] {
                        depth[idx] = z;
                        color[idx] = gc;
                    }
                }
            }
        }
    }

    /// Render depth as circle
    fn render_depth_circle(&self, depth: &mut [f32], z: f32, cx: i32, cy: i32, radius: f32) {
        let radius = radius.max(1.0);
        let r_sq = radius * radius;

        let min_x = ((cx as f32 - radius) as i32).max(0);
        let max_x = ((cx as f32 + radius) as i32).min(self.width as i32 - 1);
        let min_y = ((cy as f32 - radius) as i32).max(0);
        let max_y = ((cy as f32 + radius) as i32).min(self.height as i32 - 1);

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let dx = x as f32 - cx as f32;
                let dy = y as f32 - cy as f32;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq <= r_sq {
                    let idx = y as usize * self.width + x as usize;
                    // Keep minimum depth
                    if depth[idx] == 0.0 || z < depth[idx] {
                        depth[idx] = z;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_renderer_creation() {
        let renderer = GaussianRenderer::new(640, 480);
        assert_eq!(renderer.width, 640);
        assert_eq!(renderer.height, 480);
    }

    #[test]
    fn test_render_empty_map() {
        let renderer = GaussianRenderer::new(64, 64);
        let map = GaussianMap::new(100);
        let camera = GaussianCamera::new(500.0, 500.0, 32.0, 32.0);

        let output = renderer.render(&map, &camera);
        assert_eq!(output.color.len(), 64 * 64 * 3);
        assert_eq!(output.depth.len(), 64 * 64);
    }

    #[test]
    fn test_render_depth() {
        let renderer = GaussianRenderer::new(64, 64);
        let mut map = GaussianMap::new(100);

        // Add a Gaussian at z=1.0
        let g = Gaussian3D::from_depth_point(0.0, 0.0, 1.0, [255, 128, 64]);
        map.add(g);

        let camera = GaussianCamera::new(500.0, 500.0, 32.0, 32.0);
        let depth = renderer.render_depth(&map, &camera);

        // Should have some depth values
        assert!(depth.iter().any(|&d| d > 0.0));
    }
}
