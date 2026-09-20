//! Incremental Gaussian Mapping
//!
//! Handles adding new Gaussians from RGB-D frames and optimizing existing ones.
//! Based on RTG-SLAM approach:
//! - Add Gaussians for newly observed pixels
//! - Add Gaussians for pixels with large errors
//! - Classify Gaussians as Stable/Unstable
//! - Only optimize unstable Gaussians

use crate::fusion::gaussian::{
    matrix3_from_stored_pose_array, Gaussian3D, GaussianMap, GaussianState,
};
use crate::fusion::renderer::GaussianRenderer;
use nalgebra::Vector3;

/// Configuration for Gaussian mapping
#[derive(Debug, Clone)]
pub struct MapperConfig {
    /// Maximum number of Gaussians
    pub max_gaussians: usize,
    /// Minimum depth for new Gaussians
    pub min_depth: f32,
    /// Maximum depth for new Gaussians
    pub max_depth: f32,
    /// Sampling step for new Gaussians (pixels)
    pub sampling_step: usize,
    /// Error threshold for adding new Gaussians
    pub error_threshold: f32,
    /// Densification interval (frames)
    pub densify_interval: usize,
    /// Pruning threshold (opacity)
    pub prune_opacity_threshold: f32,
}

impl Default for MapperConfig {
    fn default() -> Self {
        Self {
            max_gaussians: 100_000,
            min_depth: 0.01, // 1cm
            max_depth: 10.0, // 10m
            sampling_step: 2,
            error_threshold: 0.1,
            densify_interval: 100,
            prune_opacity_threshold: 0.01,
        }
    }
}

/// Gaussian Mapper for incremental mapping
pub struct GaussianMapper {
    /// Configuration
    config: MapperConfig,
    /// Gaussian map
    map: GaussianMap,
    /// Renderer for error computation
    renderer: GaussianRenderer,
    /// Frame counter
    frame_count: usize,
    /// IDs of Gaussians to optimize
    optimize_ids: Vec<usize>,
}

impl GaussianMapper {
    /// Create a new mapper
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            config: MapperConfig::default(),
            map: GaussianMap::new(100_000),
            renderer: GaussianRenderer::new(width, height),
            frame_count: 0,
            optimize_ids: Vec::new(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: MapperConfig, width: usize, height: usize) -> Self {
        Self {
            map: GaussianMap::new(config.max_gaussians),
            renderer: GaussianRenderer::new(width, height),
            config,
            frame_count: 0,
            optimize_ids: Vec::new(),
        }
    }

    /// Get the Gaussian map
    pub fn map(&self) -> &GaussianMap {
        &self.map
    }

    /// Get mutable map
    pub fn map_mut(&mut self) -> &mut GaussianMap {
        &mut self.map
    }

    /// Process a new RGB-D frame
    ///
    /// 1. Add new Gaussians from newly observed pixels
    /// 2. Classify Gaussians as Stable/Unstable
    /// 3. Mark Gaussians for optimization
    pub fn update(
        &mut self,
        depth: &[f32],
        color: &[[u8; 3]],
        width: usize,
        height: usize,
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        rotation: &[[f32; 3]; 3],
        translation: &[f32; 3],
    ) -> MapperUpdateResult {
        self.frame_count += 1;

        let mut added = 0;
        let mut pruned = 0;

        // 1. Always add Gaussians for first few frames, then periodically
        if self.frame_count <= 5 || self.frame_count % self.config.densify_interval == 0 {
            added = self.add_from_depth(
                depth,
                color,
                width,
                height,
                fx,
                fy,
                cx,
                cy,
                rotation,
                translation,
            );
        }

        // 2. Compute rendering error and classify
        let (stable_count, unstable_ids) =
            self.classify_gaussians(depth, width, height, fx, fy, cx, cy, rotation, translation);

        let unstable_count = unstable_ids.len();
        self.optimize_ids = unstable_ids;
        if self.frame_count % self.config.densify_interval == 0 {
            pruned = self.prune();
        }

        MapperUpdateResult {
            added,
            pruned,
            stable_count,
            unstable_count,
            total_gaussians: self.map.len(),
        }
    }

    /// Add new Gaussians from depth frame
    fn add_from_depth(
        &mut self,
        depth: &[f32],
        color: &[[u8; 3]],
        width: usize,
        height: usize,
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        rotation: &[[f32; 3]; 3],
        translation: &[f32; 3],
    ) -> usize {
        let step = self.config.sampling_step;
        let mut added = 0;

        // Compute camera rotation matrix
        let r = matrix3_from_stored_pose_array(rotation);
        let t = Vector3::new(translation[0], translation[1], translation[2]);

        // Sample points from depth
        for y in (0..height).step_by(step) {
            for x in (0..width).step_by(step) {
                let idx = y * width + x;
                let z = depth[idx];

                // Check depth validity
                if z < self.config.min_depth || z > self.config.max_depth {
                    continue;
                }

                // Backproject to camera frame
                let x_cam = (x as f32 - cx) * z / fx;
                let y_cam = (y as f32 - cy) * z / fy;

                // Transform to world frame
                let p_cam = Vector3::new(x_cam, y_cam, z);
                let p_world = r * p_cam + t;

                // Create Gaussian
                let gaussian =
                    Gaussian3D::from_depth_point(p_world.x, p_world.y, p_world.z, color[idx]);

                if self.map.add(gaussian).is_some() {
                    added += 1;
                }
            }
        }

        added
    }

    /// Classify Gaussians as Stable or Unstable
    ///
    /// Stable: renders well to observed images
    /// Unstable: needs optimization
    fn classify_gaussians(
        &mut self,
        depth: &[f32],
        _width: usize,
        _height: usize,
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        rotation: &[[f32; 3]; 3],
        translation: &[f32; 3],
    ) -> (usize, Vec<usize>) {
        // Render current view
        let camera =
            crate::fusion::GaussianCamera::new(fx, fy, cx, cy).with_pose(*rotation, *translation);

        let rendered_depth = self.renderer.render_depth(&self.map, &camera);

        // Compare rendered vs observed depth
        let mut valid_pixels = 0usize;
        let mut total_error = 0.0f32;

        for i in 0..rendered_depth.len() {
            let rend_d = rendered_depth[i];
            let obs_d = depth[i];

            if rend_d > 0.0 && obs_d > 0.0 {
                total_error += (rend_d - obs_d).abs();
                valid_pixels += 1;
            }
        }

        let mean_error = if valid_pixels > 0 {
            total_error / valid_pixels as f32
        } else {
            self.config.error_threshold + 1.0
        };

        if mean_error <= self.config.error_threshold {
            for idx in 0..self.map.len() {
                if let Some(gaussian) = self.map.get_mut(idx) {
                    gaussian.state = GaussianState::Stable;
                }
            }
            self.map.update_states();
            (self.map.len(), Vec::new())
        } else {
            let mut unstable_ids = Vec::with_capacity(self.map.len());
            for idx in 0..self.map.len() {
                if let Some(gaussian) = self.map.get_mut(idx) {
                    gaussian.state = GaussianState::Unstable;
                    unstable_ids.push(idx);
                }
            }
            self.map.update_states();
            (0, unstable_ids)
        }
    }

    /// Get Gaussians that need optimization
    pub fn get_optimize_ids(&self) -> &[usize] {
        &self.optimize_ids
    }

    /// Optimize selected Gaussians (placeholder for full BA)
    ///
    /// In full implementation, this would:
    /// 1. Build optimization problem
    /// 2. Compute gradients
    /// 3. Update Gaussian parameters
    pub fn optimize(&mut self, iterations: usize) {
        let steps = iterations.max(1);
        for idx in self.optimize_ids.iter().copied() {
            if let Some(gaussian) = self.map.get_mut(idx) {
                gaussian.opacity = (gaussian.opacity + 0.02 * steps as f32).clamp(0.0, 1.0);
                let shrink = 0.99f32.powi(steps as i32);
                gaussian.scale *= shrink;
                gaussian.scale = gaussian.scale.map(|x| x.clamp(0.001, 0.1));
                gaussian.state = GaussianState::Stable;
            }
        }
        self.optimize_ids.clear();
        self.map.update_states();
    }

    /// Prune invisible or low-opacity Gaussians
    pub fn prune(&mut self) -> usize {
        let removed = self.map.retain(|gaussian| {
            gaussian.opacity >= self.config.prune_opacity_threshold
                && gaussian.scale.min() >= 0.001
                && gaussian.scale.max() <= 0.5
        });
        self.optimize_ids = self.map.get_unstable();
        removed
    }

    /// Get number of Gaussians
    pub fn num_gaussians(&self) -> usize {
        self.map.len()
    }

    /// Clear the map
    pub fn clear(&mut self) {
        self.map.clear();
        self.frame_count = 0;
        self.optimize_ids.clear();
    }
}

/// Result of mapper update
#[derive(Debug, Clone)]
pub struct MapperUpdateResult {
    /// Number of Gaussians added
    pub added: usize,
    /// Number of Gaussians pruned
    pub pruned: usize,
    /// Number of stable Gaussians
    pub stable_count: usize,
    /// Number of unstable Gaussians
    pub unstable_count: usize,
    /// Total Gaussians in map
    pub total_gaussians: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mapper_creation() {
        let mapper = GaussianMapper::new(640, 480);
        assert_eq!(mapper.num_gaussians(), 0);
    }

    #[test]
    fn test_mapper_update() {
        let mut mapper = GaussianMapper::new(64, 64);

        // Create test depth and color
        let depth = vec![1.0f32; 64 * 64];
        let color = vec![[255u8, 255, 255]; 64 * 64];

        let rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let translation = [0.0, 0.0, 0.0];

        let result = mapper.update(
            &depth,
            &color,
            64,
            64,
            500.0,
            500.0,
            32.0,
            32.0,
            &rotation,
            &translation,
        );

        assert!(result.total_gaussians > 0);
    }

    #[test]
    fn test_config() {
        let config = MapperConfig::default();
        assert_eq!(config.max_gaussians, 100_000);
        assert_eq!(config.sampling_step, 2);
    }

    #[test]
    fn test_optimize_marks_selected_gaussians_stable() {
        let mut mapper = GaussianMapper::new(32, 32);
        mapper.map.add(Gaussian3D::default());
        mapper.optimize_ids = vec![0];

        mapper.optimize(3);

        assert_eq!(mapper.map.get(0).unwrap().state, GaussianState::Stable);
        assert!(mapper.get_optimize_ids().is_empty());
    }

    #[test]
    fn test_prune_removes_low_opacity_gaussians() {
        let mut mapper = GaussianMapper::new(32, 32);
        let mut keep = Gaussian3D::default();
        keep.opacity = 0.5;
        let mut drop = Gaussian3D::default();
        drop.opacity = 0.0;
        mapper.map.add(keep);
        mapper.map.add(drop);

        let removed = mapper.prune();

        assert_eq!(removed, 1);
        assert_eq!(mapper.num_gaussians(), 1);
    }
}
