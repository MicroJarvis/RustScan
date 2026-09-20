//! Complete 3DGS Renderer - Tiled Rasterization
//!
//! Based on "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
//! Kerbl et al. SIGGRAPH 2023
//!
//! This implements:
//! 1. Gaussian projection to 2D
//! 2. Tiled rasterization
//! 3. Depth sorting
//! 4. Alpha blending

/// A single Gaussian with all parameters
#[derive(Debug, Clone)]
pub struct Gaussian {
    /// Position in world space [x, y, z]
    pub position: [f32; 3],
    /// Scale [sx, sy, sz]
    pub scale: [f32; 3],
    /// Rotation (quaternion) [w, x, y, z]
    pub rotation: [f32; 4],
    /// Opacity [0, 1]
    pub opacity: f32,
    /// Color RGB [r, g, b]
    pub color: [f32; 3],
}

impl Gaussian {
    pub fn new(
        position: [f32; 3],
        scale: [f32; 3],
        rotation: [f32; 4],
        opacity: f32,
        color: [f32; 3],
    ) -> Self {
        Self {
            position,
            scale,
            rotation,
            opacity,
            color,
        }
    }

    /// Create from depth point
    pub fn from_depth_point(x: f32, y: f32, z: f32, color: [u8; 3]) -> Self {
        Self {
            position: [x, y, z],
            scale: [0.01, 0.01, 0.01],
            rotation: [1.0, 0.0, 0.0, 0.0],
            opacity: 0.5,
            color: [
                color[0] as f32 / 255.0,
                color[1] as f32 / 255.0,
                color[2] as f32 / 255.0,
            ],
        }
    }
}

/// 2D projected Gaussian
#[derive(Debug, Clone)]
pub struct ProjectedGaussian {
    /// Center x in image coordinates
    pub x: f32,
    /// Center y in image coordinates
    pub y: f32,
    /// Depth (for sorting)
    pub depth: f32,
    /// 2D covariance xx
    pub cov_xx: f32,
    /// 2D covariance xy
    pub cov_xy: f32,
    /// 2D covariance yy
    pub cov_yy: f32,
    /// Opacity
    pub opacity: f32,
    /// Color
    pub color: [f32; 3],
    /// Original index (for debugging)
    pub orig_idx: usize,
}

/// Tiled rasterization renderer
pub struct TiledRenderer {
    pub width: usize,
    pub height: usize,
    tile_width: usize,
    tile_height: usize,
    num_tiles_x: usize,
    num_tiles_y: usize,
}

impl TiledRenderer {
    pub fn new(width: usize, height: usize) -> Self {
        let tile_width = 16;
        let tile_height = 16;
        let num_tiles_x = (width + tile_width - 1) / tile_width;
        let num_tiles_y = (height + tile_height - 1) / tile_height;

        Self {
            width,
            height,
            tile_width,
            tile_height,
            num_tiles_x,
            num_tiles_y,
        }
    }

    /// Project 3D Gaussians to 2D with covariance
    ///
    /// Based on Eq. 3 in the paper:
    /// Project 3D covariance to 2D using Jacobian of projection
    pub fn project_gaussians(
        &self,
        gaussians: &[Gaussian],
        fx: f32,
        fy: f32,
        _cx: f32,
        _cy: f32,
        rotation: &[[f32; 3]; 3],
        translation: &[f32; 3],
    ) -> Vec<ProjectedGaussian> {
        let mut projected = Vec::with_capacity(gaussians.len());

        // Extract rotation matrix
        let r = *rotation;

        for (idx, g) in gaussians.iter().enumerate() {
            // Transform position to camera space
            let wx = g.position[0];
            let wy = g.position[1];
            let wz = g.position[2];

            // Apply rotation
            let cx = r[0][0] * wx + r[0][1] * wy + r[0][2] * wz + translation[0];
            let cy = r[1][0] * wx + r[1][1] * wy + r[1][2] * wz + translation[1];
            let cz = r[2][0] * wx + r[2][1] * wy + r[2][2] * wz + translation[2];

            // Skip points behind camera
            if cz <= 0.0 {
                continue;
            }

            // Project to image plane
            let px = fx * cx / cz + cx;
            let py = fy * cy / cz + cy;

            // Compute 2D covariance (simplified)
            // Full implementation: transform 3D covariance by Jacobian
            let scale_x = g.scale[0].abs();
            let scale_y = g.scale[1].abs();

            let cov_xx = (scale_x * fx / cz).powi(2);
            let cov_yy = (scale_y * fy / cz).powi(2);
            let cov_xy = 0.0; // Simplified

            projected.push(ProjectedGaussian {
                x: px,
                y: py,
                depth: cz,
                cov_xx,
                cov_xy,
                cov_yy,
                opacity: g.opacity,
                color: g.color,
                orig_idx: idx,
            });
        }

        projected
    }

    /// Compute bounding box in tiles
    pub fn compute_tile_bounds(
        &self,
        g: &ProjectedGaussian,
        tile_alpha: f32,
    ) -> (usize, usize, usize, usize) {
        // Compute standard deviation
        let sigma_x = (g.cov_xx * tile_alpha).sqrt().max(1.0);
        let sigma_y = (g.cov_yy * tile_alpha).sqrt().max(1.0);

        // Bounding box in pixels
        let x_min = (g.x - 3.0 * sigma_x).max(0.0) as usize;
        let x_max = (g.x + 3.0 * sigma_x).min(self.width as f32 - 1.0) as usize;
        let y_min = (g.y - 3.0 * sigma_y).max(0.0) as usize;
        let y_max = (g.y + 3.0 * sigma_y).min(self.height as f32 - 1.0) as usize;

        // Convert to tile coordinates
        let tile_x_min = x_min / self.tile_width;
        let tile_x_max = (x_max + self.tile_width - 1) / self.tile_width;
        let tile_y_min = y_min / self.tile_height;
        let tile_y_max = (y_max + self.tile_height - 1) / self.tile_height;

        (tile_x_min, tile_x_max, tile_y_min, tile_y_max)
    }

    /// Sort Gaussians by depth (front to back)
    pub fn sort_by_depth(&self, gaussians: &mut [ProjectedGaussian]) {
        gaussians.sort_by(|a, b| {
            a.depth
                .partial_cmp(&b.depth)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Render with tiled rasterization
    ///
    /// Algorithm:
    /// 1. Project Gaussians to 2D
    /// 2. Compute tile bounds
    /// 3. For each tile, sort Gaussians that overlap it
    /// 4. Render with alpha blending (front to back)
    pub fn render(
        &self,
        gaussians: &[Gaussian],
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        rotation: &[[f32; 3]; 3],
        translation: &[f32; 3],
    ) -> RenderBuffer {
        // Project to 2D
        let projected = self.project_gaussians(gaussians, fx, fy, cx, cy, rotation, translation);

        // Initialize output buffers
        let mut color_buf = vec![0.0f32; self.width * self.height * 3];
        let mut depth_buf = vec![f32::MAX; self.width * self.height];
        let mut alpha_buf = vec![0.0f32; self.width * self.height];

        let tile_alpha = 4.0; // Alpha multiplier for tile assignment

        let tile_count = self.num_tiles_x * self.num_tiles_y;
        let mut tile_lists: Vec<Vec<usize>> = vec![Vec::new(); tile_count];

        // Assign Gaussians to tiles based on bounds
        for (idx, g) in projected.iter().enumerate() {
            let (tile_x_min, tile_x_max, tile_y_min, tile_y_max) =
                self.compute_tile_bounds(g, tile_alpha);

            for ty in tile_y_min..tile_y_max {
                for tx in tile_x_min..tile_x_max {
                    let tile_idx = ty * self.num_tiles_x + tx;
                    tile_lists[tile_idx].push(idx);
                }
            }
        }

        // Process each tile independently
        for ty in 0..self.num_tiles_y {
            for tx in 0..self.num_tiles_x {
                let tile_idx = ty * self.num_tiles_x + tx;
                let gaussians_in_tile = &mut tile_lists[tile_idx];
                if gaussians_in_tile.is_empty() {
                    continue;
                }

                // Front-to-back sort for correct alpha blending
                gaussians_in_tile.sort_by(|a, b| {
                    projected[*a]
                        .depth
                        .partial_cmp(&projected[*b].depth)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let px_start = tx * self.tile_width;
                let py_start = ty * self.tile_height;
                let px_end = (px_start + self.tile_width).min(self.width);
                let py_end = (py_start + self.tile_height).min(self.height);

                for &g_idx in gaussians_in_tile.iter() {
                    let g = &projected[g_idx];

                    // Precompute inverse covariance for Mahalanobis distance
                    // Σ⁻¹ = (1/det) * [[cov_yy, -cov_xy], [-cov_xy, cov_xx]]
                    let det = g.cov_xx * g.cov_yy - g.cov_xy * g.cov_xy;
                    if det < 1e-10 {
                        continue;
                    }
                    let inv_det = 1.0 / det;

                    for py in py_start..py_end {
                        for px in px_start..px_end {
                            let idx = py * self.width + px;

                            if alpha_buf[idx] >= 0.999 {
                                continue;
                            }

                            let dx = px as f32 - g.x;
                            let dy = py as f32 - g.y;
                            let d_sq = (g.cov_yy * dx * dx - 2.0 * g.cov_xy * dx * dy
                                + g.cov_xx * dy * dy)
                                * inv_det;

                            if d_sq < 9.0 {
                                let weight = (-0.5 * d_sq).exp() * g.opacity;
                                if weight > 0.001 {
                                    let transmittance = 1.0 - alpha_buf[idx];
                                    let alpha = weight * transmittance;
                                    if alpha > 0.0 {
                                        color_buf[idx * 3] += g.color[0] * alpha;
                                        color_buf[idx * 3 + 1] += g.color[1] * alpha;
                                        color_buf[idx * 3 + 2] += g.color[2] * alpha;
                                        alpha_buf[idx] += alpha;

                                        if depth_buf[idx] == f32::MAX && alpha > 0.01 {
                                            depth_buf[idx] = g.depth;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Clamp colors to [0, 1]
        for v in &mut color_buf {
            *v = v.clamp(0.0, 1.0);
        }

        RenderBuffer {
            color: color_buf,
            depth: depth_buf,
            width: self.width,
            height: self.height,
        }
    }
}

/// Render output buffer
pub struct RenderBuffer {
    pub color: Vec<f32>, // [H, W, 3] RGB
    pub depth: Vec<f32>, // [H, W]
    pub width: usize,
    pub height: usize,
}

impl RenderBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            color: vec![0.0f32; width * height * 3],
            depth: vec![f32::MAX; width * height],
            width,
            height,
        }
    }
}

/// Densification - add new Gaussians
pub fn densify(gaussians: &mut Vec<Gaussian>, grads: &[f32], threshold: f32) {
    let _n = gaussians.len();
    let mut new_gaussians = Vec::new();

    for (i, g) in gaussians.iter().enumerate() {
        let grad_mag = if i < grads.len() { grads[i].abs() } else { 0.0 };

        // Split large Gaussians with high gradient
        if grad_mag > threshold && g.scale[0] < 0.1 {
            // Create two smaller Gaussians
            let offset = g.scale[0] * 0.1;

            let mut g1 = g.clone();
            g1.position[0] += offset;
            g1.scale[0] *= 0.8;

            let mut g2 = g.clone();
            g2.position[0] -= offset;
            g2.scale[0] *= 0.8;

            new_gaussians.push(g1);
            new_gaussians.push(g2);
        }
    }

    gaussians.extend(new_gaussians);
}

/// Pruning - remove low opacity Gaussians
pub fn prune(gaussians: &mut Vec<Gaussian>, threshold: f32) {
    gaussians.retain(|g| g.opacity > threshold);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiled_renderer() {
        let renderer = TiledRenderer::new(640, 480);
        assert_eq!(renderer.num_tiles_x, 40);
        assert_eq!(renderer.num_tiles_y, 30);
    }

    #[test]
    fn test_gaussian_projection() {
        let renderer = TiledRenderer::new(64, 64);

        let gaussians = vec![Gaussian::new(
            [0.0, 0.0, 1.0],
            [0.01, 0.01, 0.01],
            [1.0, 0.0, 0.0, 0.0],
            0.5,
            [1.0, 0.5, 0.25],
        )];

        let rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let projected = renderer.project_gaussians(
            &gaussians,
            500.0,
            500.0,
            32.0,
            32.0,
            &rotation,
            &[0.0, 0.0, 0.0],
        );

        assert!(!projected.is_empty());
        assert!(projected[0].depth > 0.0);
    }

    #[test]
    fn test_depth_sorting() {
        let mut gaussians = vec![
            ProjectedGaussian {
                x: 0.0,
                y: 0.0,
                depth: 2.0,
                cov_xx: 1.0,
                cov_xy: 0.0,
                cov_yy: 1.0,
                opacity: 0.5,
                color: [1.0, 0.0, 0.0],
                orig_idx: 0,
            },
            ProjectedGaussian {
                x: 0.0,
                y: 0.0,
                depth: 1.0,
                cov_xx: 1.0,
                cov_xy: 0.0,
                cov_yy: 1.0,
                opacity: 0.5,
                color: [0.0, 1.0, 0.0],
                orig_idx: 1,
            },
            ProjectedGaussian {
                x: 0.0,
                y: 0.0,
                depth: 3.0,
                cov_xx: 1.0,
                cov_xy: 0.0,
                cov_yy: 1.0,
                opacity: 0.5,
                color: [0.0, 0.0, 1.0],
                orig_idx: 2,
            },
        ];

        let renderer = TiledRenderer::new(64, 64);
        renderer.sort_by_depth(&mut gaussians);

        // Should be sorted front to back (near to far)
        assert_eq!(gaussians[0].depth, 1.0);
        assert_eq!(gaussians[1].depth, 2.0);
        assert_eq!(gaussians[2].depth, 3.0);
    }
}
