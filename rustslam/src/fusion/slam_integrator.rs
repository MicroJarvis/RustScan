//! Sparse-Dense SLAM Integration
//!
//! Combines sparse SLAM (feature-based) with dense 3DGS mapping.
//! Based on RTG-SLAM and SplaTAM approaches.
//!
//! Architecture:
//! Sparse Tracking → Pose Estimate → Dense Mapping → 3DGS Map
//!        ↑                              ↓
//!        ←←←←← Loop Closure ←←←←←←←←

use crate::core::SE3;
use crate::fusion::tiled_renderer::{densify, prune, Gaussian, RenderBuffer, TiledRenderer};

/// Configuration for Sparse-Dense SLAM
#[derive(Debug, Clone)]
pub struct SlamConfig {
    /// Whether to use dense mapping
    pub use_dense: bool,
    /// Dense mapping interval (frames)
    pub dense_interval: usize,
    /// Number of Gaussians to maintain
    pub max_gaussians: usize,
    /// Densification threshold
    pub densify_threshold: f32,
    /// Pruning threshold
    pub prune_threshold: f32,
    /// Keyframe distance threshold
    pub kf_dist_threshold: f32,
    /// Keyframe angle threshold
    pub kf_angle_threshold: f32,
}

impl Default for SlamConfig {
    fn default() -> Self {
        Self {
            use_dense: true,
            dense_interval: 1,
            max_gaussians: 100_000,
            densify_threshold: 0.0002,
            prune_threshold: 0.005,
            kf_dist_threshold: 0.5,   // meters
            kf_angle_threshold: 10.0, // degrees
        }
    }
}

/// A keyframe with both sparse and dense data
#[derive(Debug, Clone)]
pub struct KeyFrame {
    /// Frame ID
    pub id: usize,
    /// Camera pose
    pub pose: SE3,
    /// Timestamp
    pub timestamp: f64,
    /// RGB image data
    pub color: Vec<u8>,
    /// Depth image data
    pub depth: Vec<f32>,
    /// Image width
    pub width: usize,
    /// Image height
    pub height: usize,
    /// Camera intrinsics
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    /// Associated Gaussians (for dense)
    pub gaussians: Vec<Gaussian>,
}

impl KeyFrame {
    pub fn new(
        id: usize,
        pose: SE3,
        timestamp: f64,
        color: Vec<u8>,
        depth: Vec<f32>,
        width: usize,
        height: usize,
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
    ) -> Self {
        Self {
            id,
            pose,
            timestamp,
            color,
            depth,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            gaussians: Vec::new(),
        }
    }

    /// Check if should create new keyframe
    pub fn should_create_kf(&self, current_pose: &SE3) -> bool {
        // Check translation distance
        let t1 = self.pose.translation();
        let t2 = current_pose.translation();
        let dist =
            ((t1[0] - t2[0]).powi(2) + (t1[1] - t2[1]).powi(2) + (t1[2] - t2[2]).powi(2)).sqrt();

        // Check rotation angle
        let r1 = self.pose.rotation_matrix();
        let r2 = current_pose.rotation_matrix();

        // Simplified angle check
        let angle = ((r1[0][0] * r2[0][0] + r1[0][1] * r2[0][1] + r1[0][2] * r2[0][2]).acos()
            * 180.0
            / std::f32::consts::PI)
            .abs();

        dist > 0.5 || angle > 10.0
    }
}

/// Sparse-Dense SLAM System
pub struct SparseDenseSlam {
    config: SlamConfig,
    /// Current frame ID
    frame_id: usize,
    /// Current pose
    current_pose: SE3,
    /// Keyframes
    keyframes: Vec<KeyFrame>,
    /// Dense Gaussians
    gaussians: Vec<Gaussian>,
    /// Tiled renderer
    renderer: TiledRenderer,
    /// Whether tracking is lost
    tracking_lost: bool,
    /// Camera intrinsics
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
}

impl SparseDenseSlam {
    pub fn new(width: usize, height: usize) -> Self {
        let cam = crate::config::CameraConfig::default();
        Self {
            config: SlamConfig::default(),
            frame_id: 0,
            current_pose: SE3::identity(),
            keyframes: Vec::new(),
            gaussians: Vec::new(),
            renderer: TiledRenderer::new(width, height),
            tracking_lost: false,
            fx: cam.fx,
            fy: cam.fy,
            cx: cam.cx,
            cy: cam.cy,
        }
    }

    pub fn with_config(config: SlamConfig, width: usize, height: usize) -> Self {
        let cam = crate::config::CameraConfig::default();
        Self {
            config,
            frame_id: 0,
            current_pose: SE3::identity(),
            keyframes: Vec::new(),
            gaussians: Vec::new(),
            renderer: TiledRenderer::new(width, height),
            tracking_lost: false,
            fx: cam.fx,
            fy: cam.fy,
            cx: cam.cx,
            cy: cam.cy,
        }
    }

    /// Get current pose
    pub fn pose(&self) -> &SE3 {
        &self.current_pose
    }

    /// Get number of Gaussians
    pub fn num_gaussians(&self) -> usize {
        self.gaussians.len()
    }

    /// Get keyframes
    pub fn keyframes(&self) -> &[KeyFrame] {
        &self.keyframes
    }

    /// Process a new frame with sparse tracking result
    ///
    /// This is called after sparse tracking gives us a pose estimate
    pub fn process_frame(
        &mut self,
        color: Vec<u8>,
        depth: Vec<f32>,
        tracked_pose: SE3,
    ) -> SlamOutput {
        self.frame_id += 1;
        self.current_pose = tracked_pose;

        // Create keyframe if needed
        let is_keyframe = if self.keyframes.is_empty() {
            // First frame - always create keyframe
            true
        } else {
            // Check if should create new keyframe
            let last_kf = &self.keyframes[self.keyframes.len() - 1];
            last_kf.should_create_kf(&tracked_pose)
        };

        // Create keyframe if needed
        if is_keyframe {
            let kf = KeyFrame::new(
                self.frame_id,
                tracked_pose,
                self.frame_id as f64,
                color.clone(),
                depth.clone(),
                self.renderer.width,
                self.renderer.height,
                self.fx,
                self.fy,
                self.cx,
                self.cy,
            );
            self.keyframes.push(kf);
        }

        // Dense mapping
        if self.config.use_dense && self.frame_id % self.config.dense_interval == 0 {
            self.update_dense_map(&color, &depth, tracked_pose);
        }

        // Periodic densification
        if self.frame_id % 100 == 0 && !self.gaussians.is_empty() {
            // Use a placeholder gradient buffer for heuristic densification.
            let grads = vec![0.0f32; self.gaussians.len()];
            densify(&mut self.gaussians, &grads, self.config.densify_threshold);
        }

        // Periodic pruning
        if self.frame_id % 100 == 0 {
            prune(&mut self.gaussians, self.config.prune_threshold);
        }

        // Limit Gaussians
        if self.gaussians.len() > self.config.max_gaussians {
            // Remove oldest
            let excess = self.gaussians.len() - self.config.max_gaussians;
            self.gaussians.drain(0..excess);
        }

        SlamOutput {
            frame_id: self.frame_id,
            pose: self.current_pose.clone(),
            is_keyframe,
            num_gaussians: self.gaussians.len(),
            tracking_lost: self.tracking_lost,
        }
    }

    /// Update dense 3DGS map from RGB-D frame
    fn update_dense_map(&mut self, color: &[u8], depth: &[f32], pose: SE3) {
        let width = self.renderer.width;
        let height = self.renderer.height;

        // Extract rotation and translation
        let rot = pose.rotation_matrix();
        let trans = pose.translation();

        // Sample points from depth
        let step = 2; // Sample every other pixel
        let mut new_gaussians = Vec::new();

        for y in (0..height).step_by(step) {
            for x in (0..width).step_by(step) {
                let idx = y * width + x;
                let z = depth[idx];

                // Skip invalid depth
                if z <= 0.0 || z > 10.0 {
                    continue;
                }

                // Backproject to camera space using configured intrinsics
                let x_cam = (x as f32 - self.cx) * z / self.fx;
                let y_cam = (y as f32 - self.cy) * z / self.fy;

                // Transform to world space
                let wx = rot[0][0] * x_cam + rot[0][1] * y_cam + rot[0][2] * z + trans[0];
                let wy = rot[1][0] * x_cam + rot[1][1] * y_cam + rot[1][2] * z + trans[1];
                let wz = rot[2][0] * x_cam + rot[2][1] * y_cam + rot[2][2] * z + trans[2];

                // Get color
                let c_idx = idx * 3;
                let pixel_color = [color[c_idx], color[c_idx + 1], color[c_idx + 2]];

                // Create Gaussian
                let gaussian = Gaussian::from_depth_point(wx, wy, wz, pixel_color);
                new_gaussians.push(gaussian);
            }
        }

        // Add new Gaussians
        self.gaussians.extend(new_gaussians);
    }

    /// Render current view
    pub fn render(&self) -> RenderBuffer {
        let rot = self.current_pose.rotation_matrix();
        let trans = self.current_pose.translation();

        self.renderer.render(
            &self.gaussians,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            &rot,
            &trans,
        )
    }

    /// Reset the SLAM system
    pub fn reset(&mut self) {
        self.frame_id = 0;
        self.current_pose = SE3::identity();
        self.keyframes.clear();
        self.gaussians.clear();
        self.tracking_lost = false;
    }

    /// Mark tracking as lost
    pub fn set_tracking_lost(&mut self, lost: bool) {
        self.tracking_lost = lost;
    }
}

/// Output from processing a frame
#[derive(Debug, Clone)]
pub struct SlamOutput {
    pub frame_id: usize,
    pub pose: SE3,
    pub is_keyframe: bool,
    pub num_gaussians: usize,
    pub tracking_lost: bool,
}

/// Integration with existing RustSLAM sparse tracking
///
/// Usage:
/// ```ignore
/// // In your main loop:
///
/// // 1. Get sparse tracking result (from existing RustSLAM)
/// let tracked_pose = sparse_tracker.track(frame);
///
/// // 2. Feed to dense SLAM
/// let output = dense_slam.process_frame(color, depth, tracked_pose);
///
/// // 3. Get rendered view (optional)
/// let rendered = dense_slam.render();
/// ```
pub struct SlamIntegrator {
    sparse_dense: SparseDenseSlam,
}

impl SlamIntegrator {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            sparse_dense: SparseDenseSlam::new(width, height),
        }
    }

    pub fn with_config(config: SlamConfig, width: usize, height: usize) -> Self {
        Self {
            sparse_dense: SparseDenseSlam::with_config(config, width, height),
        }
    }

    /// Process frame from sparse tracking
    pub fn process(&mut self, color: Vec<u8>, depth: Vec<f32>, sparse_pose: SE3) -> SlamOutput {
        self.sparse_dense.process_frame(color, depth, sparse_pose)
    }

    /// Get current rendered view
    pub fn render(&self) -> RenderBuffer {
        self.sparse_dense.render()
    }

    /// Get current pose
    pub fn pose(&self) -> &SE3 {
        self.sparse_dense.pose()
    }

    /// Get Gaussian count
    pub fn num_gaussians(&self) -> usize {
        self.sparse_dense.num_gaussians()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slam_creation() {
        let slam = SparseDenseSlam::new(640, 480);
        assert_eq!(slam.num_gaussians(), 0);
    }

    #[test]
    fn test_keyframe_should_create() {
        let kf = KeyFrame::new(
            0,
            SE3::identity(),
            0.0,
            vec![],
            vec![],
            640,
            480,
            500.0,
            500.0,
            320.0,
            240.0,
        );

        // Far pose should trigger new keyframe
        let far_pose = SE3::from_rotation_translation(
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &[1.0, 0.0, 0.0],
        );

        assert!(kf.should_create_kf(&far_pose));
    }

    #[test]
    fn test_slam_integrator() {
        let mut integrator = SlamIntegrator::new(64, 64);

        let color = vec![128u8; 64 * 64 * 3];
        let depth = vec![1.0f32; 64 * 64];

        let output = integrator.process(color, depth, SE3::identity());

        assert_eq!(output.frame_id, 1);
        assert!(output.is_keyframe); // First frame
    }
}
