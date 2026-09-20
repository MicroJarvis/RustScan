//! Bundle Adjustment module
//!
//! Provides BA functionality for optimizing camera poses and 3D points.
//! Implements Bundle Adjustment using Gauss-Newton algorithm.

use crate::core::SE3;
/// Camera pose with intrinsics for BA
#[derive(Clone)]
pub struct BACamera {
    /// Camera intrinsics (fx, fy, cx, cy)
    pub intrinsics: [f64; 4],
    /// Camera pose (SE3)
    pub pose: SE3,
    /// Is pose fixed (not optimized)?
    pub fix_pose: bool,
}

impl BACamera {
    /// Create a new BA camera
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self {
            intrinsics: [fx, fy, cx, cy],
            pose: SE3::identity(),
            fix_pose: false,
        }
    }

    /// Create with pose
    pub fn with_pose(mut self, pose: SE3) -> Self {
        self.pose = pose;
        self
    }

    /// Fix the pose (not optimized)
    pub fn fix_pose(mut self) -> Self {
        self.fix_pose = true;
        self
    }

    /// Get focal length
    pub fn fx(&self) -> f64 {
        self.intrinsics[0]
    }
    pub fn fy(&self) -> f64 {
        self.intrinsics[1]
    }
    /// Get principal point
    pub fn cx(&self) -> f64 {
        self.intrinsics[2]
    }
    pub fn cy(&self) -> f64 {
        self.intrinsics[3]
    }
}

impl std::fmt::Debug for BACamera {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BACamera")
            .field("intrinsics", &self.intrinsics)
            .field("pose", &"SE3")
            .field("fix_pose", &self.fix_pose)
            .finish()
    }
}

/// 3D landmark for BA
#[derive(Debug, Clone)]
pub struct BALandmark {
    /// 3D position (X, Y, Z)
    pub position: [f64; 3],
    /// Is position fixed?
    pub fix_position: bool,
}

impl BALandmark {
    /// Create a new landmark
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            position: [x, y, z],
            fix_position: false,
        }
    }

    /// Fix position
    pub fn fix_position(mut self) -> Self {
        self.fix_position = true;
        self
    }
}

/// Observation (2D point in image)
#[derive(Debug, Clone, Copy)]
pub struct BAObservation {
    /// Pixel coordinates (u, v)
    pub uv: [f64; 2],
}

impl BAObservation {
    /// Create a new observation
    pub fn new(u: f64, v: f64) -> Self {
        Self { uv: [u, v] }
    }
}

/// Bundle Adjustment problem builder
///
/// This struct holds the BA problem data and provides methods to build
/// and solve the optimization problem using Gauss-Newton algorithm.
pub struct BundleAdjuster {
    /// Cameras to optimize
    cameras: Vec<BACamera>,
    /// Landmarks (3D points)
    landmarks: Vec<BALandmark>,
    /// Observations: (camera_idx, landmark_idx, observation)
    observations: Vec<(usize, usize, BAObservation)>,
    /// Whether optimization has been run
    optimized: bool,
    /// Optimization verbose
    verbose: bool,
    /// Initial cost before optimization
    initial_cost: Option<f64>,
    /// Final cost after optimization
    final_cost: Option<f64>,
    /// Number of iterations run
    iterations: usize,
}

impl BundleAdjuster {
    /// Create a new bundle adjuster
    pub fn new() -> Self {
        Self {
            cameras: Vec::new(),
            landmarks: Vec::new(),
            observations: Vec::new(),
            optimized: false,
            verbose: false,
            initial_cost: None,
            final_cost: None,
            iterations: 0,
        }
    }

    /// Enable verbose output
    pub fn with_verbose(mut self) -> Self {
        self.verbose = true;
        self
    }

    /// Add a camera to the problem
    pub fn add_camera(&mut self, camera: BACamera) -> usize {
        let idx = self.cameras.len();
        self.cameras.push(camera);
        idx
    }

    /// Add a landmark to the problem
    pub fn add_landmark(&mut self, landmark: BALandmark) -> usize {
        let idx = self.landmarks.len();
        self.landmarks.push(landmark);
        idx
    }

    /// Add an observation
    pub fn add_observation(&mut self, camera_idx: usize, landmark_idx: usize, obs: BAObservation) {
        self.observations.push((camera_idx, landmark_idx, obs));
    }

    /// Get initial cost
    pub fn initial_cost(&self) -> Option<f64> {
        self.initial_cost
    }

    /// Get final cost
    pub fn final_cost(&self) -> Option<f64> {
        self.final_cost
    }

    /// Get number of iterations
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Project a 3D point to 2D using camera model
    fn project(&self, cam_idx: usize, lm_idx: usize) -> Option<[f64; 2]> {
        let camera = &self.cameras[cam_idx];
        let landmark = &self.landmarks[lm_idx];

        let pose = camera.pose;
        let r = pose.rotation_matrix();
        let t = pose.translation();

        // Transform point to camera frame
        let px = landmark.position[0] as f32;
        let py = landmark.position[1] as f32;
        let pz = landmark.position[2] as f32;
        let tx = t[0] as f32;
        let ty = t[1] as f32;
        let tz = t[2] as f32;

        // p_cam = R^T * (p_world - t)
        let x = r[0][0] * (px - tx) + r[1][0] * (py - ty) + r[2][0] * (pz - tz);
        let y = r[0][1] * (px - tx) + r[1][1] * (py - ty) + r[2][1] * (pz - tz);
        let z = r[0][2] * (px - tx) + r[1][2] * (py - ty) + r[2][2] * (pz - tz);

        if z <= 1e-10 {
            return None;
        }

        // Project to image plane
        let fx = camera.fx() as f32;
        let fy = camera.fy() as f32;
        let cx = camera.cx() as f32;
        let cy = camera.cy() as f32;

        let u = fx * x / z + cx;
        let v = fy * y / z + cy;

        Some([u as f64, v as f64])
    }

    /// Compute all reprojection errors
    fn compute_errors(&self) -> Vec<f64> {
        self.observations
            .iter()
            .filter_map(|(cam_idx, lm_idx, obs)| {
                self.project(*cam_idx, *lm_idx).map(|proj| {
                    let dx = proj[0] - obs.uv[0];
                    let dy = proj[1] - obs.uv[1];
                    dx * dx + dy * dy
                })
            })
            .collect()
    }

    fn total_cost(&self) -> f64 {
        self.compute_errors().iter().sum()
    }

    fn perturb_pose(pose: &SE3, axis: usize, delta: f32) -> SE3 {
        let mut twist = [0.0f32; 6];
        twist[axis] = delta;
        SE3::exp(&twist).compose(pose)
    }

    fn try_pose_update(
        &mut self,
        cam_idx: usize,
        base_pose: SE3,
        update: [f32; 6],
        base_cost: f64,
    ) {
        let mut scale = 1.0f32;
        while scale >= 1.0 / 1024.0 {
            let mut scaled_update = [0.0f32; 6];
            for axis in 0..6 {
                scaled_update[axis] = update[axis] * scale;
            }
            self.cameras[cam_idx].pose = SE3::exp(&scaled_update).compose(&base_pose);
            let candidate_cost = self.total_cost();
            if candidate_cost.is_finite() && candidate_cost <= base_cost {
                return;
            }
            scale *= 0.5;
        }

        self.cameras[cam_idx].pose = base_pose;
    }

    fn try_landmark_update(
        &mut self,
        lm_idx: usize,
        base_position: [f64; 3],
        update: [f64; 3],
        base_cost: f64,
    ) {
        let mut scale = 1.0f64;
        while scale >= 1.0 / 1024.0 {
            for axis in 0..3 {
                self.landmarks[lm_idx].position[axis] = base_position[axis] + update[axis] * scale;
            }
            let candidate_cost = self.total_cost();
            if candidate_cost.is_finite() && candidate_cost <= base_cost {
                return;
            }
            scale *= 0.5;
        }

        self.landmarks[lm_idx].position = base_position;
    }

    /// Build and run BA optimization using Gauss-Newton
    pub fn optimize(
        &mut self,
        max_iterations: usize,
    ) -> Result<(Vec<BACamera>, Vec<BALandmark>), String> {
        if self.cameras.is_empty() {
            return Err("No cameras in BA problem".to_string());
        }
        if self.landmarks.is_empty() {
            return Err("No landmarks in BA problem".to_string());
        }
        if self.observations.is_empty() {
            return Err("No observations in BA problem".to_string());
        }

        // Compute initial cost
        let initial_errors = self.compute_errors();
        let initial_cost: f64 = initial_errors.iter().sum();
        self.initial_cost = Some(initial_cost);

        if self.verbose {
            println!("Initial cost: {:.6}", initial_cost);
        }

        // Joint optimization over camera poses and landmarks (finite-difference gradient descent)
        self.iterations = 0;
        let mut prev_cost = initial_cost;
        let eps_pose = 1e-4f64;
        let eps_point = 1e-4f64;

        for iter in 0..max_iterations {
            self.iterations = iter + 1;

            // Compute errors
            let errors = self.compute_errors();
            let total_cost: f64 = errors.iter().sum();

            if self.verbose {
                println!("Iteration {}: cost = {:.6}", iter + 1, total_cost);
            }

            // Check convergence
            if iter > 0 && (total_cost - prev_cost).abs() < 1e-6 {
                if self.verbose {
                    println!("Converged!");
                }
                break;
            }
            prev_cost = total_cost;

            let num_obs = self.num_observations().max(1) as f64;
            let pose_lr = 1e-3f64 / num_obs;
            let point_lr = 1e-3f64 / num_obs;

            // Optimize camera poses
            for cam_idx in 0..self.cameras.len() {
                if self.cameras[cam_idx].fix_pose {
                    continue;
                }

                let base_cost = self.total_cost();
                let base_pose = self.cameras[cam_idx].pose;
                let mut grad = [0.0f64; 6];
                for axis in 0..6 {
                    self.cameras[cam_idx].pose =
                        Self::perturb_pose(&base_pose, axis, eps_pose as f32);
                    let plus = self.total_cost();

                    self.cameras[cam_idx].pose =
                        Self::perturb_pose(&base_pose, axis, -(eps_pose as f32));
                    let minus = self.total_cost();

                    grad[axis] = (plus - minus) / (2.0 * eps_pose);
                }
                self.cameras[cam_idx].pose = base_pose;

                let mut update = [0.0f32; 6];
                for axis in 0..6 {
                    update[axis] = (-pose_lr * grad[axis]) as f32;
                }
                self.try_pose_update(cam_idx, base_pose, update, base_cost);
            }

            // Optimize landmark positions
            for lm_idx in 0..self.landmarks.len() {
                if self.landmarks[lm_idx].fix_position {
                    continue;
                }

                let base_cost = self.total_cost();
                let base = self.landmarks[lm_idx].position;
                let mut grad = [0.0f64; 3];
                for axis in 0..3 {
                    self.landmarks[lm_idx].position[axis] = base[axis] + eps_point;
                    let plus = self.total_cost();

                    self.landmarks[lm_idx].position[axis] = base[axis] - eps_point;
                    let minus = self.total_cost();

                    grad[axis] = (plus - minus) / (2.0 * eps_point);
                    self.landmarks[lm_idx].position[axis] = base[axis];
                }

                let mut update = [0.0f64; 3];
                for axis in 0..3 {
                    update[axis] = -point_lr * grad[axis];
                }
                self.try_landmark_update(lm_idx, base, update, base_cost);
            }
        }

        // Compute final cost
        let final_errors = self.compute_errors();
        let final_cost: f64 = final_errors.iter().sum();
        self.final_cost = Some(final_cost);

        if self.verbose {
            if let Some(init) = self.initial_cost {
                if init > 0.0 {
                    println!("Cost reduction: {:.2}%", (1.0 - final_cost / init) * 100.0);
                }
            }
        }

        self.optimized = true;
        Ok((self.cameras.clone(), self.landmarks.clone()))
    }

    pub fn is_optimized(&self) -> bool {
        self.optimized
    }

    pub fn num_cameras(&self) -> usize {
        self.cameras.len()
    }

    pub fn num_landmarks(&self) -> usize {
        self.landmarks.len()
    }

    pub fn num_observations(&self) -> usize {
        self.observations.len()
    }

    /// Compute total reprojection error
    pub fn compute_residual(&self) -> f64 {
        self.compute_errors().iter().map(|e| e.sqrt()).sum()
    }

    pub fn clear(&mut self) {
        self.cameras.clear();
        self.landmarks.clear();
        self.observations.clear();
        self.optimized = false;
        self.initial_cost = None;
        self.final_cost = None;
        self.iterations = 0;
    }
}

impl Default for BundleAdjuster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ba_camera_creation() {
        let camera = BACamera::new(500.0, 500.0, 320.0, 240.0);
        assert_eq!(camera.intrinsics, [500.0, 500.0, 320.0, 240.0]);
        assert_eq!(camera.fx(), 500.0);
    }

    #[test]
    fn test_ba_landmark_creation() {
        let landmark = BALandmark::new(1.0, 2.0, 3.0);
        assert_eq!(landmark.position, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_ba_observation() {
        let obs = BAObservation::new(100.0, 200.0);
        assert_eq!(obs.uv, [100.0, 200.0]);
    }

    #[test]
    fn test_bundle_adjuster_creation() {
        let adjuster = BundleAdjuster::new();
        assert_eq!(adjuster.num_cameras(), 0);
        assert_eq!(adjuster.num_landmarks(), 0);
        assert!(!adjuster.is_optimized());
    }

    #[test]
    fn test_add_camera() {
        let mut adjuster = BundleAdjuster::new();
        let camera = BACamera::new(500.0, 500.0, 320.0, 240.0);
        let idx = adjuster.add_camera(camera);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_add_landmark() {
        let mut adjuster = BundleAdjuster::new();
        let landmark = BALandmark::new(1.0, 2.0, 3.0);
        let idx = adjuster.add_landmark(landmark);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_add_observation() {
        let mut adjuster = BundleAdjuster::new();
        let camera = BACamera::new(500.0, 500.0, 320.0, 240.0);
        adjuster.add_camera(camera);
        let landmark = BALandmark::new(1.0, 2.0, 3.0);
        adjuster.add_landmark(landmark);
        let obs = BAObservation::new(100.0, 200.0);
        adjuster.add_observation(0, 0, obs);
        assert_eq!(adjuster.num_observations(), 1);
    }

    #[test]
    fn test_optimize_empty() {
        let mut adjuster = BundleAdjuster::new();
        let result = adjuster.optimize(10);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_residual() {
        let mut adjuster = BundleAdjuster::new();
        let camera = BACamera::new(500.0, 500.0, 320.0, 240.0);
        adjuster.add_camera(camera);
        let landmark = BALandmark::new(1.0, 2.0, 5.0);
        adjuster.add_landmark(landmark);
        // z=5: u = 500*1/5 + 320 = 420, v = 500*2/5 + 240 = 440
        let obs = BAObservation::new(420.0, 440.0);
        adjuster.add_observation(0, 0, obs);
        let residual = adjuster.compute_residual();
        assert!(residual < 0.01);
    }

    #[test]
    fn test_optimize_updates_camera_pose() {
        let mut adjuster = BundleAdjuster::new();
        let initial_pose = SE3::from_axis_angle(&[0.0, 0.0, 0.0], &[0.2, 0.0, 0.0]);
        let camera = BACamera::new(300.0, 300.0, 160.0, 120.0).with_pose(initial_pose);
        let cam_idx = adjuster.add_camera(camera);

        let points = vec![
            [0.0, 0.0, 3.0],
            [0.2, 0.1, 3.0],
            [-0.2, -0.1, 3.0],
            [0.1, -0.2, 3.2],
        ];

        for p in &points {
            let lm_idx = adjuster.add_landmark(BALandmark::new(p[0], p[1], p[2]));
            let u = 300.0 * p[0] / p[2] + 160.0;
            let v = 300.0 * p[1] / p[2] + 120.0;
            adjuster.add_observation(cam_idx, lm_idx, BAObservation::new(u, v));
        }

        let initial_cost = adjuster.total_cost();
        let (cams, _) = adjuster.optimize(8).unwrap();
        let final_cost = adjuster.total_cost();
        let t0 = initial_pose.translation();
        let t1 = cams[0].pose.translation();

        assert!(
            final_cost < initial_cost,
            "BA should reduce reprojection cost"
        );
        assert!(
            (t1[0] - t0[0]).abs() > 1e-5
                || (t1[1] - t0[1]).abs() > 1e-5
                || (t1[2] - t0[2]).abs() > 1e-5,
            "Camera pose should be updated during optimization"
        );
    }

    #[test]
    fn test_clear() {
        let mut adjuster = BundleAdjuster::new();
        let camera = BACamera::new(500.0, 500.0, 320.0, 240.0);
        adjuster.add_camera(camera);
        let landmark = BALandmark::new(1.0, 2.0, 3.0);
        adjuster.add_landmark(landmark);
        let obs = BAObservation::new(100.0, 200.0);
        adjuster.add_observation(0, 0, obs);
        adjuster.clear();
        assert_eq!(adjuster.num_cameras(), 0);
    }
}
