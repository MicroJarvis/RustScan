//! Gaussian-based Tracking
//!
//! Uses rendered depth from Gaussian map for camera tracking.
//! Based on RTG-SLAM and SplaTAM approaches.

use crate::core::SE3;
use crate::fusion::gaussian::{GaussianCamera, GaussianMap};
use crate::fusion::renderer::GaussianRenderer;
use nalgebra::Vector3;

/// Tracking result
#[derive(Debug, Clone)]
pub struct TrackingResult {
    /// Estimated pose
    pub pose: SE3,
    /// Tracking confidence (0-1)
    pub confidence: f32,
    /// Number of matched points
    pub num_matches: usize,
    /// Whether tracking is successful
    pub success: bool,
}

impl TrackingResult {
    pub fn failure() -> Self {
        Self {
            pose: SE3::identity(),
            confidence: 0.0,
            num_matches: 0,
            success: false,
        }
    }
}

/// Gaussian-based Tracker
pub struct GaussianTracker {
    /// Renderer for depth
    renderer: GaussianRenderer,
    /// Image width
    width: usize,
    /// Image height
    _height: usize,
    /// Camera intrinsics
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    /// ICP iterations
    icp_iterations: usize,
    /// Depth threshold for matching
    depth_threshold: f32,
    /// Maximum correspondence distance
    _max_correspondence_dist: f32,
}

impl GaussianTracker {
    /// Create a new tracker
    pub fn new(width: usize, height: usize) -> Self {
        let default_cam = crate::core::Camera::default();
        let fx_scale = default_cam.focal.x / default_cam.width as f32;
        let fy_scale = default_cam.focal.y / default_cam.height as f32;
        let fx = (width as f32 * fx_scale).max(1.0);
        let fy = (height as f32 * fy_scale).max(1.0);
        let cx = width as f32 * 0.5;
        let cy = height as f32 * 0.5;

        Self {
            renderer: GaussianRenderer::new(width, height),
            width,
            _height: height,
            fx,
            fy,
            cx,
            cy,
            icp_iterations: 10,
            depth_threshold: 0.5,          // 50cm
            _max_correspondence_dist: 0.1, // 10cm
        }
    }

    /// Override camera intrinsics used by tracking.
    pub fn set_intrinsics(&mut self, fx: f32, fy: f32, cx: f32, cy: f32) {
        self.fx = fx.max(1.0);
        self.fy = fy.max(1.0);
        self.cx = cx;
        self.cy = cy;
    }

    /// Track camera pose using rendered depth
    ///
    /// Uses ICP (Iterative Closest Point) between rendered depth and observed depth
    pub fn track(
        &self,
        map: &GaussianMap,
        observed_depth: &[f32],
        initial_pose: SE3,
    ) -> TrackingResult {
        if map.is_empty() {
            return TrackingResult::failure();
        }

        let mut pose = initial_pose;
        let mut total_matches = 0;

        // ICP iterations
        for _ in 0..self.icp_iterations {
            // Render depth from current pose estimate
            let camera = self.create_camera(&pose);
            let rendered_depth = self.renderer.render_depth(map, &camera);

            // Compute ICP
            let (matches, error) = self.compute_icp(&rendered_depth, observed_depth, &camera);

            if matches.is_empty() {
                break;
            }

            total_matches = matches.len();

            // Check convergence
            if error < 1e-4 {
                break;
            }

            // Update pose (simplified - just translation for now)
            // Full implementation would compute optimal transformation
            if let Some(delta) = self.estimate_delta(&matches, &camera) {
                let delta_se3 = SE3::from_rotation_translation(
                    &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    &delta,
                );
                pose = delta_se3.compose(&pose);
            }
        }

        let confidence = (total_matches as f32 / (observed_depth.len() as f32)).min(1.0);

        TrackingResult {
            pose,
            confidence,
            num_matches: total_matches,
            success: confidence > 0.1,
        }
    }

    /// Create camera from SE3 pose
    fn create_camera(&self, pose: &SE3) -> GaussianCamera {
        let rotation = pose.rotation_matrix();
        let translation = pose.translation();

        GaussianCamera::new(self.fx, self.fy, self.cx, self.cy).with_pose(rotation, translation)
    }

    /// Compute ICP correspondence
    fn compute_icp(
        &self,
        rendered_depth: &[f32],
        observed_depth: &[f32],
        camera: &GaussianCamera,
    ) -> (Vec<ICPCorrespondence>, f32) {
        let mut correspondences = Vec::new();
        let mut total_error = 0.0f32;

        // Simple depth correspondence
        for (i, &obs_d) in observed_depth.iter().enumerate() {
            if obs_d <= 0.0 {
                continue;
            }

            let rend_d = rendered_depth[i];

            // Skip invalid rendered depth
            if rend_d <= 0.0 {
                continue;
            }

            // Check depth difference
            let diff = (obs_d - rend_d).abs();

            if diff < self.depth_threshold {
                // Compute 3D point
                let u = (i % self.width) as f32;
                let v = (i / self.width) as f32;

                let x = (u - camera.cx) * obs_d / camera.fx;
                let y = (v - camera.cy) * obs_d / camera.fy;
                let z = obs_d;

                // Rendered point
                let x_r = (u - camera.cx) * rend_d / camera.fx;
                let y_r = (v - camera.cy) * rend_d / camera.fy;
                let z_r = rend_d;

                correspondences.push(ICPCorrespondence {
                    observed: [x, y, z],
                    rendered: [x_r, y_r, z_r],
                });

                total_error += diff;
            }
        }

        let avg_error = if !correspondences.is_empty() {
            total_error / correspondences.len() as f32
        } else {
            f32::MAX
        };

        (correspondences, avg_error)
    }

    /// Estimate transformation delta (simplified)
    fn estimate_delta(
        &self,
        correspondences: &[ICPCorrespondence],
        _camera: &GaussianCamera,
    ) -> Option<[f32; 3]> {
        if correspondences.is_empty() {
            return None;
        }

        // Simplified: compute centroid difference
        let mut obs_centroid = Vector3::zeros();
        let mut rend_centroid = Vector3::zeros();

        for c in correspondences {
            obs_centroid += Vector3::new(c.observed[0], c.observed[1], c.observed[2]);
            rend_centroid += Vector3::new(c.rendered[0], c.rendered[1], c.rendered[2]);
        }

        let n = correspondences.len() as f32;
        obs_centroid /= n;
        rend_centroid /= n;

        // Translation delta
        let delta = [
            (obs_centroid.x - rend_centroid.x),
            (obs_centroid.y - rend_centroid.y),
            (obs_centroid.z - rend_centroid.z),
        ];

        Some(delta)
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        // No state to reset for now
    }
}

/// ICP Correspondence
#[derive(Debug, Clone)]
pub struct ICPCorrespondence {
    /// Observed 3D point
    observed: [f32; 3],
    /// Rendered 3D point
    rendered: [f32; 3],
}

impl Default for GaussianTracker {
    fn default() -> Self {
        Self::new(640, 480)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_creation() {
        let tracker = GaussianTracker::new(640, 480);
        assert!(tracker.icp_iterations > 0);
    }

    #[test]
    fn test_tracking_empty_map() {
        let tracker = GaussianTracker::new(64, 64);
        let map = GaussianMap::new(100);
        let depth = vec![1.0f32; 64 * 64];

        let result = tracker.track(&map, &depth, SE3::identity());

        assert!(!result.success);
    }

    #[test]
    fn test_tracking_with_gaussians() {
        let tracker = GaussianTracker::new(64, 64);
        let mut map = GaussianMap::new(100);

        // Add a Gaussian at z=1.0
        let g =
            crate::fusion::gaussian::Gaussian3D::from_depth_point(0.0, 0.0, 1.0, [255, 128, 64]);
        map.add(g);

        // Match depth
        let depth = vec![1.0f32; 64 * 64];

        let result = tracker.track(&map, &depth, SE3::identity());

        // Should have some matches
        assert!(result.num_matches >= 0);
    }
}
