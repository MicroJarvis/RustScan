//! Geometric solvers for Visual Odometry
//!
//! Implements PnP, Essential Matrix, Triangulation, and Sim3 solvers with proper algorithms.

use crate::core::SE3;
use crate::features::base::Match;
use glam::{Mat3, Vec3};
use nalgebra::{DMatrix, Matrix3, Vector3 as NaVec3};
use std::f32::consts::PI;

/// 2D-3D correspondence for PnP
#[derive(Debug, Clone)]
pub struct PnPProblem {
    /// 2D points in image coordinates [x, y]
    pub image_points: Vec<[f32; 2]>,
    /// 3D points in world coordinates [X, Y, Z]
    pub object_points: Vec<[f32; 3]>,
}

impl PnPProblem {
    /// Create a new PnP problem
    pub fn new() -> Self {
        Self {
            image_points: Vec::new(),
            object_points: Vec::new(),
        }
    }

    /// Add a correspondence
    pub fn add_correspondence(&mut self, img: [f32; 2], obj: [f32; 3]) {
        self.image_points.push(img);
        self.object_points.push(obj);
    }

    /// Check if we have enough points
    pub fn is_solvable(&self) -> bool {
        self.image_points.len() >= 4
    }
}

impl Default for PnPProblem {
    fn default() -> Self {
        Self::new()
    }
}

/// RANSAC-based PnP solver using P3P + RANSAC
pub struct PnPSolver {
    /// Camera intrinsics
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    /// RANSAC parameters
    pub ransac_threshold: f32,
    pub ransac_confidence: f32,
    pub ransac_max_iterations: u32,
}

impl PnPSolver {
    /// Create a new PnP solver
    pub fn new(fx: f32, fy: f32, cx: f32, cy: f32) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            ransac_threshold: 3.0,
            ransac_confidence: 0.99,
            ransac_max_iterations: 200,
        }
    }

    /// Solve PnP using RANSAC with P3P
    ///
    /// Returns: (pose, inlier_mask)
    pub fn solve(&self, problem: &PnPProblem) -> Option<(SE3, Vec<bool>)> {
        if !problem.is_solvable() {
            return None;
        }

        let n = problem.image_points.len();

        // Normalize image coordinates
        let normalized_pts: Vec<[f32; 2]> = problem
            .image_points
            .iter()
            .map(|p| [(p[0] - self.cx) / self.fx, (p[1] - self.cy) / self.fy])
            .collect();

        let threshold = (self.ransac_threshold / self.fx.max(self.fy).max(1.0)).max(1.0e-6);
        let seed = deterministic_pnp_seed(&normalized_pts, &problem.object_points);
        let mut rng = Lcg::new(seed);

        // RANSAC loop
        let mut best_inliers: Vec<bool> = vec![false; n];
        let mut best_pose: Option<SE3> = None;
        let mut best_inlier_count = 0;
        let mut best_mean_error = f32::INFINITY;
        let sample_size = 3usize; // true P3P uses exactly 3 correspondences
        let mut adaptive_max_iter = self.ransac_max_iterations;

        for iter in 0..self.ransac_max_iterations {
            if iter >= adaptive_max_iter {
                break;
            }
            // Randomly select 3 points for true P3P hypothesis
            let indices = self.random_indices(&mut rng, n, sample_size);

            // Solve for this sample
            if let Some(poses) = self.solve_p3p(&normalized_pts, &problem.object_points, &indices) {
                // For each P3P solution, check all points
                for pose in poses {
                    let (inliers, inlier_count, mean_error) = self.evaluate_pose(
                        &pose,
                        &normalized_pts,
                        &problem.object_points,
                        threshold,
                    );

                    // Update best if more inliers
                    if inlier_count > best_inlier_count
                        || (inlier_count == best_inlier_count && mean_error < best_mean_error)
                    {
                        best_inlier_count = inlier_count;
                        best_inliers = inliers;
                        best_pose = Some(pose);
                        best_mean_error = mean_error;
                    }
                }
            }

            // Adaptive early termination: recompute the number of iterations
            // required so that at least one sample is all-inlier with probability
            // `ransac_confidence`, given the current observed inlier ratio.
            if best_inlier_count >= sample_size {
                let inlier_ratio = best_inlier_count as f32 / n as f32;
                let p_all_inlier = inlier_ratio.powi(sample_size as i32);
                if p_all_inlier > 1e-9 {
                    let needed = ((1.0f32 - self.ransac_confidence).ln()
                        / (1.0f32 - p_all_inlier).ln())
                    .ceil() as u32;
                    adaptive_max_iter = needed.min(self.ransac_max_iterations);
                }
            }
        }

        // Wide-baseline or nearly-planar image sets can make the minimal P3P
        // hypotheses brittle. Add a bounded DLT-RANSAC pass as a non-minimal
        // absolute-pose hypothesis, then keep the same inlier/refinement path.
        let dlt_sample_size = 6usize;
        let dlt_rescue_threshold = 12usize.min(n / 10).max(dlt_sample_size);
        if n >= dlt_sample_size && best_inlier_count < dlt_rescue_threshold {
            let dlt_max_iterations = self.ransac_max_iterations.clamp(64, 1024);
            let mut dlt_adaptive_max_iter = dlt_max_iterations;
            for iter in 0..dlt_max_iterations {
                if iter >= dlt_adaptive_max_iter {
                    break;
                }
                let sample = rng.sample_unique(n, dlt_sample_size);
                if sample.len() < dlt_sample_size {
                    continue;
                }
                let mut sample_img = Vec::with_capacity(sample.len());
                let mut sample_obj = Vec::with_capacity(sample.len());
                for idx in sample {
                    sample_img.push(normalized_pts[idx]);
                    sample_obj.push(problem.object_points[idx]);
                }
                let Some(pose) = self.estimate_pose_dlt(&sample_img, &sample_obj) else {
                    continue;
                };
                let (inliers, inlier_count, mean_error) =
                    self.evaluate_pose(&pose, &normalized_pts, &problem.object_points, threshold);
                let significant_gain = inlier_count > best_inlier_count + (4usize).max(n / 20);
                let rescue_hypothesis = best_inlier_count < dlt_sample_size.max(8)
                    && inlier_count >= dlt_sample_size.max(8);
                let cleaner_tie =
                    inlier_count > best_inlier_count && mean_error < best_mean_error * 0.75;
                if rescue_hypothesis || significant_gain || cleaner_tie {
                    best_inlier_count = inlier_count;
                    best_inliers = inliers;
                    best_pose = Some(pose);
                    best_mean_error = mean_error;
                }

                if best_inlier_count >= dlt_sample_size {
                    let inlier_ratio = best_inlier_count as f32 / n as f32;
                    let p_all_inlier = inlier_ratio.powi(dlt_sample_size as i32);
                    if p_all_inlier > 1e-9 {
                        let needed = ((1.0f32 - self.ransac_confidence).ln()
                            / (1.0f32 - p_all_inlier).ln())
                        .ceil() as u32;
                        dlt_adaptive_max_iter = needed.min(dlt_max_iterations);
                    }
                }
            }
        }

        if let Some(pose) = self.estimate_pose_dlt(&normalized_pts, &problem.object_points) {
            let (inliers, inlier_count, mean_error) =
                self.evaluate_pose(&pose, &normalized_pts, &problem.object_points, threshold);
            if inlier_count > best_inlier_count
                || (inlier_count == best_inlier_count && mean_error < best_mean_error)
            {
                best_pose = Some(pose);
                best_inliers = inliers;
                best_inlier_count = inlier_count;
            }
        }

        if best_pose.is_none() {
            return None;
        }

        // Refine pose with all inliers
        if let Some(ref mut pose) = best_pose {
            if best_inlier_count >= 4 {
                *pose =
                    self.refine_pose(pose, &normalized_pts, &problem.object_points, &best_inliers);
                let (refined_inliers, refined_inlier_count, _) =
                    self.evaluate_pose(pose, &normalized_pts, &problem.object_points, threshold);
                if refined_inlier_count >= 4 {
                    best_inliers = refined_inliers;
                }
            }
        }

        Some((best_pose.unwrap_or(SE3::identity()), best_inliers))
    }

    fn evaluate_pose(
        &self,
        pose: &SE3,
        normalized_pts: &[[f32; 2]],
        object_points: &[[f32; 3]],
        threshold: f32,
    ) -> (Vec<bool>, usize, f32) {
        let n = normalized_pts.len().min(object_points.len());
        let mut inliers = vec![false; n];
        let mut inlier_count = 0usize;
        let mut total_error = 0.0f32;

        for i in 0..n {
            let projected = self.project_point(pose, &object_points[i]);
            let error = self.reprojection_error(&normalized_pts[i], &projected);
            if error < threshold {
                inliers[i] = true;
                inlier_count += 1;
                total_error += error;
            }
        }

        let mean_error = if inlier_count > 0 {
            total_error / inlier_count as f32
        } else {
            f32::INFINITY
        };

        (inliers, inlier_count, mean_error)
    }

    /// Randomly select k indices
    fn random_indices(&self, rng: &mut Lcg, n: usize, k: usize) -> Vec<usize> {
        use std::collections::HashSet;
        let mut selected = HashSet::new();
        while selected.len() < k.min(n) {
            let idx = rng.gen_range(n);
            selected.insert(idx);
        }
        let mut indices: Vec<usize> = selected.into_iter().collect();
        indices.sort_unstable();
        indices
    }

    /// Solve P3P - Perspective-Three-Point Problem
    ///
    /// Given 3 2D-3D correspondences, compute up to 4 possible camera poses
    fn solve_p3p(
        &self,
        img_pts: &[[f32; 2]],
        obj_pts: &[[f32; 3]],
        indices: &[usize],
    ) -> Option<Vec<SE3>> {
        if indices.len() < 3 {
            return None;
        }
        let (i0, i1, i2) = (indices[0], indices[1], indices[2]);
        if i0 >= img_pts.len() || i1 >= img_pts.len() || i2 >= img_pts.len() {
            return None;
        }

        // Bearing vectors (unit direction in normalised camera coords)
        let f1 = Vec3::new(img_pts[i0][0], img_pts[i0][1], 1.0).normalize();
        let f2 = Vec3::new(img_pts[i1][0], img_pts[i1][1], 1.0).normalize();
        let f3 = Vec3::new(img_pts[i2][0], img_pts[i2][1], 1.0).normalize();

        let p1 = Vec3::from(obj_pts[i0]);
        let p2 = Vec3::from(obj_pts[i1]);
        let p3 = Vec3::from(obj_pts[i2]);

        // Squared inter-point distances in world space
        let d12_sq = (p2 - p1).length_squared();
        let d13_sq = (p3 - p1).length_squared();
        let d23_sq = (p3 - p2).length_squared();
        if d12_sq < 1e-10 || d13_sq < 1e-10 || d23_sq < 1e-10 {
            return None;
        }

        // Cosines of inter-ray angles
        let cos_alpha = f1.dot(f2); // angle between rays to P1, P2
        let cos_beta = f1.dot(f3); // angle between rays to P1, P3
        let cos_gamma = f2.dot(f3); // angle between rays to P2, P3

        // Grunert equations (Fischler & Bolles 1981):
        //   s1² + s2² - 2 s1 s2 cos_alpha = d12²   ... (1)
        //   s1² + s3² - 2 s1 s3 cos_beta  = d13²   ... (2)
        //   s2² + s3² - 2 s2 s3 cos_gamma = d23²   ... (3)
        //
        // Let u = s2/s1.  Divide (1) by s1²: 1 + u² - 2u cos_alpha = d12²/s1²
        // Let  K1 = d13²/d12²,  K2 = d23²/d12².
        //
        // From (1)/(2): 1 + u² - 2u cos_alpha = (d12²/d13²)(1 + v² - 2v cos_beta)
        //   where v = s3/s1.  This gives a linear-quadratic relation in (u, v).
        // From (1)/(3): (1 + u² - 2u cos_alpha) * bc²/d12² = u²s1²·? ... standard derivation
        //
        // We eliminate v using the quadratic from (1)=(2) and substitute into (3)
        // to obtain a degree-4 polynomial in u.

        let k1 = d13_sq / d12_sq;
        let k2 = d23_sq / d12_sq;

        // From equality of s1² in equations (1) and (2):
        //   (1 + u² - 2u·cos_alpha) / d12² = (1 + v² - 2v·cos_beta) / d13²
        // Rearranged (quadratic in v):
        //   k1·v² - 2k1·cos_beta·v + k1 - u² + 2u·cos_alpha - 1 = 0
        // Discriminant: Δ = k1²·cos_beta² - k1(k1 - u² + 2u·cos_alpha - 1)
        //            = k1(k1·cos_beta² - k1 + u² - 2u·cos_alpha + 1)
        // Solve for v: v = [k1·cos_beta ± √(k1·Δ_inner)] / k1
        //   where Δ_inner = u² - 2u·cos_alpha + 1 - k1·sin_beta²  (using cos²β-1 = -sin²β)
        // Two branches (±) → substitute each into (3) and clear the sqrt.
        //
        // After substitution and squaring we get a degree-4 polynomial in u.
        // Coefficients from the standard derivation:

        let sin_alpha_sq = (1.0 - cos_alpha * cos_alpha).max(0.0);
        let sin_beta_sq = (1.0 - cos_beta * cos_beta).max(0.0);
        let sin_gamma_sq = (1.0 - cos_gamma * cos_gamma).max(0.0);

        // Coefficient of the quartic G(u) = c4 u⁴ + c3 u³ + c2 u² + c1 u + c0 = 0
        // Derived by eliminating v from the Grunert system (see e.g. Sun & Wang 2010,
        // or the classic derivation in multiple-view geometry texts).
        let g1 = (1.0 - k1) * (1.0 - k1);
        let g2 = 4.0 * k1 * sin_beta_sq;
        let h1 = 2.0 * cos_alpha;
        let h2 = 2.0 * k1 * cos_beta;
        let h3 = k2 * (1.0 - k1 - k2 + k1 * k2 - 4.0 * k1 * sin_gamma_sq);

        let c4 = g1 - g2 + (1.0 + k2) * (1.0 + k2) - 4.0 * k2 * sin_gamma_sq;
        let c3 = -h1 * (g1 - g2 + (1.0 + k2) * (1.0 + k2) - 4.0 * k2 * sin_gamma_sq)
            + h2 * (1.0 + k2 - 2.0 * k2 * cos_gamma * cos_alpha / (cos_gamma.max(1e-8)))
            + h3 * h1;
        let c2 = -2.0 * k1 * sin_alpha_sq * (g1 - g2 + (1.0 + k2) * (1.0 + k2))
            + 4.0 * k1 * sin_alpha_sq * k2 * sin_gamma_sq
            + 2.0 * (1.0 - k1) * (1.0 - k1)
            - 4.0 * k1 * sin_beta_sq
            - 2.0 * (1.0 + k2) * (1.0 + k2)
            + 8.0 * k2 * sin_gamma_sq;
        let c1 = 2.0 * h1 * k1 * sin_alpha_sq;
        let c0 = g1 - g2;

        // Solve the quartic via its companion matrix eigenvalues (f32 precision
        // is fine here: we validate every root via reprojection afterwards).
        let roots = solve_quartic_companion(c4, c3, c2, c1, c0);

        let mut poses = Vec::with_capacity(8);
        for u in roots {
            if u <= 0.0 {
                continue; // s2 must be positive (in front of camera)
            }

            // v from the quadratic (both branches):
            // k1·v² - 2k1·cos_beta·v + (k1 - u² + 2u·cos_alpha - 1) = 0
            let qa = k1;
            let qb = -2.0 * k1 * cos_beta;
            let qc = k1 - u * u + 2.0 * u * cos_alpha - 1.0;
            let disc = qb * qb - 4.0 * qa * qc;
            if disc < 0.0 {
                continue;
            }
            let sqrt_disc = disc.sqrt();
            for &sign in &[1.0f32, -1.0] {
                let v = (-qb + sign * sqrt_disc) / (2.0 * qa);
                if v <= 0.0 {
                    continue;
                }

                // Recover s1 from equation (1):  s1² = d12² / (1 + u² - 2u cos_alpha)
                let denom = 1.0 + u * u - 2.0 * u * cos_alpha;
                if denom < 1e-8 {
                    continue;
                }
                let s1_sq = d12_sq / denom;
                if s1_sq < 0.0 {
                    continue;
                }
                let s1 = s1_sq.sqrt();
                let s2 = u * s1;
                let s3 = v * s1;

                // Camera-frame coordinates of the three world points
                let q1 = f1 * s1;
                let q2 = f2 * s2;
                let q3 = f3 * s3;

                // Recover R, t via Procrustes (SVD) on the 3-point set
                if let Some(pose) = recover_pose_from_3points(
                    &[q1, q2, q3], // camera-frame positions
                    &[p1, p2, p3], // world positions
                ) {
                    poses.push(pose);
                }
            }
        }

        if poses.is_empty() {
            None
        } else {
            Some(poses)
        }
    }

    /// Project a 3D point to normalized image coordinates
    fn project_point(&self, pose: &SE3, point: &[f32; 3]) -> [f32; 2] {
        let transformed = pose.transform_point(point);
        if transformed[2] > 0.0 {
            [
                transformed[0] / transformed[2],
                transformed[1] / transformed[2],
            ]
        } else {
            [0.0, 0.0]
        }
    }

    /// Compute reprojection error
    fn reprojection_error(&self, p1: &[f32; 2], p2: &[f32; 2]) -> f32 {
        let dx = p1[0] - p2[0];
        let dy = p1[1] - p2[1];
        (dx * dx + dy * dy).sqrt()
    }

    /// Refine pose using all inliers with Gauss-Newton
    fn refine_pose(
        &self,
        initial: &SE3,
        img_pts: &[[f32; 2]],
        obj_pts: &[[f32; 3]],
        inliers: &[bool],
    ) -> SE3 {
        // Collect inlier correspondences
        let mut inlier_img = Vec::new();
        let mut inlier_obj = Vec::new();

        for (i, &is_inlier) in inliers.iter().enumerate() {
            if is_inlier {
                inlier_img.push(img_pts[i]);
                inlier_obj.push(obj_pts[i]);
            }
        }

        if inlier_img.len() < 4 {
            return *initial;
        }

        let mut pose = *initial;
        let eps = 1e-4f32;
        for _ in 0..8 {
            let mut h = nalgebra::SMatrix::<f32, 6, 6>::zeros();
            let mut b = nalgebra::SVector::<f32, 6>::zeros();
            let mut valid_obs = 0usize;

            for (obs, obj) in inlier_img.iter().zip(inlier_obj.iter()) {
                let projected = self.project_point(&pose, obj);
                let e =
                    nalgebra::SVector::<f32, 2>::new(obs[0] - projected[0], obs[1] - projected[1]);

                let mut j = nalgebra::SMatrix::<f32, 2, 6>::zeros();
                for k in 0..6 {
                    let mut delta = [0.0f32; 6];
                    delta[k] = eps;
                    let pose_perturbed = SE3::exp(&delta).compose(&pose);
                    let p_plus = self.project_point(&pose_perturbed, obj);
                    j[(0, k)] = (p_plus[0] - projected[0]) / eps;
                    j[(1, k)] = (p_plus[1] - projected[1]) / eps;
                }

                h += j.transpose() * j;
                b += j.transpose() * e;
                valid_obs += 1;
            }

            if valid_obs < 4 {
                break;
            }

            for d in 0..6 {
                h[(d, d)] += 1e-6;
            }

            let Some(delta) = h.lu().solve(&b) else {
                break;
            };

            let mut twist = [0.0f32; 6];
            for k in 0..6 {
                twist[k] = delta[k];
            }
            pose = SE3::exp(&twist).compose(&pose);

            if delta.norm() < 1e-5 {
                break;
            }
        }

        pose
    }

    fn estimate_pose_dlt(&self, img_pts: &[[f32; 2]], obj_pts: &[[f32; 3]]) -> Option<SE3> {
        let n = img_pts.len().min(obj_pts.len());
        if n < 4 {
            return None;
        }

        let mut a_data = Vec::with_capacity(n * 2 * 12);
        for i in 0..n {
            let u = img_pts[i][0];
            let v = img_pts[i][1];
            let x = obj_pts[i][0];
            let y = obj_pts[i][1];
            let z = obj_pts[i][2];

            a_data.extend_from_slice(&[
                x,
                y,
                z,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -u * x,
                -u * y,
                -u * z,
                -u,
            ]);
            a_data.extend_from_slice(&[
                0.0,
                0.0,
                0.0,
                0.0,
                x,
                y,
                z,
                1.0,
                -v * x,
                -v * y,
                -v * z,
                -v,
            ]);
        }

        let a = DMatrix::<f32>::from_row_slice(n * 2, 12, &a_data);
        let svd = a.svd(true, true);
        let v_t = svd.v_t?;
        let p = v_t.row(v_t.nrows() - 1);

        let mut pmat = nalgebra::SMatrix::<f32, 3, 4>::zeros();
        for r in 0..3 {
            for c in 0..4 {
                pmat[(r, c)] = p[r * 4 + c];
            }
        }

        let m = pmat.fixed_view::<3, 3>(0, 0).into_owned();
        let scale = (m.row(0).norm() + m.row(1).norm() + m.row(2).norm()) / 3.0;
        if !scale.is_finite() || scale.abs() < 1e-8 {
            return None;
        }
        let scaled = pmat / scale;
        let candidates = [scaled, -scaled];

        let mut best_pose = None;
        let mut best_positive_depths = 0usize;
        let mut best_mean_error = f32::INFINITY;

        for candidate in candidates {
            let m_norm = candidate.fixed_view::<3, 3>(0, 0).into_owned();
            let svd_r = m_norm.svd(true, true);
            let u = svd_r.u?;
            let v_t = svd_r.v_t?;

            let mut correction = Matrix3::identity();
            if (u.clone() * v_t.clone()).determinant() < 0.0 {
                correction[(2, 2)] = -1.0;
            }
            let r = u * correction * v_t;
            let t = candidate.column(3);
            let pose = SE3::from_rotation_translation(&mat3_to_array(&r), &[t[0], t[1], t[2]]);

            let mut positive_depths = 0usize;
            let mut total_error = 0.0f32;
            let mut valid_projections = 0usize;

            for i in 0..n {
                let transformed = pose.transform_point(&obj_pts[i]);
                if transformed[2] > 0.0 {
                    positive_depths += 1;
                }
                if transformed[2].is_finite() && transformed[2] > 1e-6 {
                    let projected = [
                        transformed[0] / transformed[2],
                        transformed[1] / transformed[2],
                    ];
                    total_error += self.reprojection_error(&img_pts[i], &projected);
                    valid_projections += 1;
                } else {
                    total_error += 1e6;
                }
            }

            if positive_depths == 0 {
                continue;
            }

            let mean_error = if valid_projections > 0 {
                total_error / valid_projections as f32
            } else {
                f32::INFINITY
            };

            let better = best_pose.is_none()
                || positive_depths > best_positive_depths
                || (positive_depths == best_positive_depths && mean_error < best_mean_error);
            if better {
                best_pose = Some(pose);
                best_positive_depths = positive_depths;
                best_mean_error = mean_error;
            }
        }

        best_pose
    }
}

/// Essential Matrix solver for 2D-2D motion estimation
pub struct EssentialSolver {
    /// RANSAC parameters
    pub ransac_threshold: f32,
    pub ransac_max_iterations: u32,
}

impl EssentialSolver {
    /// Create a new essential matrix solver
    pub fn new() -> Self {
        Self {
            ransac_threshold: 0.01,
            ransac_max_iterations: 200,
        }
    }

    /// Compute essential matrix from matches using 8-point algorithm + RANSAC
    ///
    /// Returns: (essential_matrix, inlier_mask)
    pub fn compute(
        &self,
        _matches: &[Match],
        pts1: &[[f32; 2]],
        pts2: &[[f32; 2]],
    ) -> Option<(Mat3, Vec<bool>)> {
        if pts1.len() < 8 || pts2.len() < 8 {
            return None;
        }

        let n = pts1.len().min(pts2.len());
        let mut rng = Lcg::new(n as u64 + (self.ransac_max_iterations as u64));
        let mut best_inliers = Vec::new();
        let mut best_e = None;
        let mut best_inlier_count = 0usize;
        let mut best_error = f32::INFINITY;

        for _ in 0..self.ransac_max_iterations {
            let sample = rng.sample_unique(n, 8);
            if sample.len() < 8 {
                continue;
            }

            let mut s1 = Vec::with_capacity(8);
            let mut s2 = Vec::with_capacity(8);
            for &idx in &sample {
                s1.push(pts1[idx]);
                s2.push(pts2[idx]);
            }

            let e = self.compute_essential(&s1, &s2)?;
            let inliers = self.inlier_mask(pts1, pts2, &e);
            let (count, mean_error) = self.inlier_stats(pts1, pts2, &e, &inliers);
            if count > best_inlier_count || (count == best_inlier_count && mean_error < best_error)
            {
                best_inliers = inliers;
                best_e = Some(e);
                best_inlier_count = count;
                best_error = mean_error;
            }
        }

        let mut e = best_e.or_else(|| self.compute_essential(pts1, pts2))?;
        if best_inliers.is_empty() {
            best_inliers = self.inlier_mask(pts1, pts2, &e);
        }

        let inlier_count = best_inliers.iter().filter(|&&x| x).count();
        if inlier_count >= 8 {
            let mut refined_pts1 = Vec::with_capacity(inlier_count);
            let mut refined_pts2 = Vec::with_capacity(inlier_count);
            for i in 0..n {
                if best_inliers[i] {
                    refined_pts1.push(pts1[i]);
                    refined_pts2.push(pts2[i]);
                }
            }
            if let Some(refined_e) = self.compute_essential(&refined_pts1, &refined_pts2) {
                e = refined_e;
                best_inliers = self.inlier_mask(pts1, pts2, &e);
            }
        }

        Some((e, best_inliers))
    }

    /// Normalize points for numerical stability
    fn normalize_points(&self, pts: &[[f32; 2]]) -> (Vec<[f32; 2]>, Mat3) {
        let n = pts.len();
        if n == 0 {
            return (vec![], Mat3::IDENTITY);
        }

        // Compute centroid
        let mut cx = 0.0f32;
        let mut cy = 0.0f32;
        for p in pts {
            cx += p[0];
            cy += p[1];
        }
        cx /= n as f32;
        cy /= n as f32;

        // Compute scale
        let mut scale = 0.0f32;
        for p in pts {
            scale += ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt();
        }
        scale = (n as f32 * 1.414) / scale.max(1e-8);

        // Normalize
        let normalized: Vec<[f32; 2]> = pts
            .iter()
            .map(|p| [(p[0] - cx) * scale, (p[1] - cy) * scale])
            .collect();

        // Transformation matrix
        let t = Mat3::from_cols(
            Vec3::new(scale, 0.0, 0.0),
            Vec3::new(0.0, scale, 0.0),
            Vec3::new(-cx * scale, -cy * scale, 1.0),
        );

        (normalized, t)
    }

    /// Solve using 8-point algorithm
    fn solve_8point(&self, a: &[[f32; 9]]) -> Option<Mat3> {
        let n = a.len();
        if n < 8 {
            return None;
        }

        let mut data = Vec::with_capacity(n * 9);
        for row in a {
            data.extend_from_slice(row);
        }

        let mat = DMatrix::<f32>::from_row_slice(n, 9, &data);
        let svd = mat.svd(true, true);
        let v_t = svd.v_t?;
        if v_t.nrows() == 0 {
            return None;
        }
        let e_vec = v_t.row(v_t.nrows() - 1);
        let e = Matrix3::from_row_slice(&[
            e_vec[0], e_vec[1], e_vec[2], e_vec[3], e_vec[4], e_vec[5], e_vec[6], e_vec[7],
            e_vec[8],
        ]);

        Some(mat3_from_na(&e))
    }

    /// Enforce rank-2 constraint on essential matrix
    pub fn enforce_rank2(&self, e: Mat3) -> Mat3 {
        let na_e = mat3_to_na(&e);
        let svd = na_e.svd(true, true);
        let mut u = svd.u.unwrap_or(Matrix3::identity());
        let mut v_t = svd.v_t.unwrap_or(Matrix3::identity());
        if u.determinant() < 0.0 {
            u *= -1.0;
        }
        if v_t.determinant() < 0.0 {
            v_t *= -1.0;
        }

        let s = svd.singular_values;
        let essential_sigma = 0.5 * (s[0] + s[1]);
        let sigma = Matrix3::from_diagonal(&NaVec3::new(essential_sigma, essential_sigma, 0.0));
        let e_rank2 = u * sigma * v_t;
        mat3_from_na(&e_rank2)
    }

    /// RANSAC filtering
    fn inlier_mask(&self, pts1: &[[f32; 2]], pts2: &[[f32; 2]], e: &Mat3) -> Vec<bool> {
        let n = pts1.len().min(pts2.len());
        let na_e = mat3_to_na(e);
        let threshold = self.ransac_threshold.max(1e-6);
        let mut inliers = Vec::with_capacity(n);

        for i in 0..n {
            let x1 = NaVec3::new(pts1[i][0], pts1[i][1], 1.0);
            let x2 = NaVec3::new(pts2[i][0], pts2[i][1], 1.0);
            let ex1 = na_e * x1;
            let etx2 = na_e.transpose() * x2;
            let x2t_ex1 = x2.transpose() * na_e * x1;
            let denom = ex1[0] * ex1[0] + ex1[1] * ex1[1] + etx2[0] * etx2[0] + etx2[1] * etx2[1];
            let dist = if denom > 1e-12 {
                (x2t_ex1[(0, 0)] * x2t_ex1[(0, 0)]) / denom
            } else {
                f32::MAX
            };
            inliers.push(dist < threshold * threshold);
        }

        inliers
    }

    fn inlier_stats(
        &self,
        pts1: &[[f32; 2]],
        pts2: &[[f32; 2]],
        e: &Mat3,
        inliers: &[bool],
    ) -> (usize, f32) {
        let n = pts1.len().min(pts2.len()).min(inliers.len());
        let na_e = mat3_to_na(e);
        let mut count = 0usize;
        let mut total_error = 0.0f32;

        for i in 0..n {
            if !inliers[i] {
                continue;
            }
            let x1 = NaVec3::new(pts1[i][0], pts1[i][1], 1.0);
            let x2 = NaVec3::new(pts2[i][0], pts2[i][1], 1.0);
            let ex1 = na_e * x1;
            let etx2 = na_e.transpose() * x2;
            let x2t_ex1 = x2.transpose() * na_e * x1;
            let denom = ex1[0] * ex1[0] + ex1[1] * ex1[1] + etx2[0] * etx2[0] + etx2[1] * etx2[1];
            if denom > 1e-12 {
                total_error += (x2t_ex1[(0, 0)] * x2t_ex1[(0, 0)]) / denom;
                count += 1;
            }
        }

        let mean_error = if count > 0 {
            total_error / count as f32
        } else {
            f32::INFINITY
        };
        (count, mean_error)
    }

    /// Recover pose from essential matrix
    ///
    /// Returns: 4 possible pose solutions
    pub fn recover_pose(&self, e: Mat3) -> [SE3; 4] {
        let na_e = mat3_to_na(&e);
        let svd = na_e.svd(true, true);
        let mut u = svd.u.unwrap_or(Matrix3::identity());
        let mut v_t = svd.v_t.unwrap_or(Matrix3::identity());

        if u.determinant() < 0.0 {
            u *= -1.0;
        }
        if v_t.determinant() < 0.0 {
            v_t *= -1.0;
        }

        let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

        let mut r1 = u * w * v_t;
        let mut r2 = u * w.transpose() * v_t;
        if r1.determinant() < 0.0 {
            r1 = -r1;
        }
        if r2.determinant() < 0.0 {
            r2 = -r2;
        }
        let t = u.column(2);
        let t_vec = normalize_vec3([t[0], t[1], t[2]]);

        let pose1 = SE3::from_rotation_translation(&mat3_to_array(&r1), &t_vec);
        let pose2 = SE3::from_rotation_translation(&mat3_to_array(&r1), &negate_vec3(t_vec));
        let pose3 = SE3::from_rotation_translation(&mat3_to_array(&r2), &t_vec);
        let pose4 = SE3::from_rotation_translation(&mat3_to_array(&r2), &negate_vec3(t_vec));

        [pose1, pose2, pose3, pose4]
    }
}

impl EssentialSolver {
    fn compute_essential(&self, pts1: &[[f32; 2]], pts2: &[[f32; 2]]) -> Option<Mat3> {
        let n = pts1.len().min(pts2.len());
        if n < 8 {
            return None;
        }

        let (norm_pts1, t1) = self.normalize_points(pts1);
        let (norm_pts2, t2) = self.normalize_points(pts2);

        let mut a = Vec::with_capacity(n);
        for i in 0..n {
            let x1 = &norm_pts1[i];
            let x2 = &norm_pts2[i];
            a.push([
                x2[0] * x1[0],
                x2[0] * x1[1],
                x2[0],
                x2[1] * x1[0],
                x2[1] * x1[1],
                x2[1],
                x1[0],
                x1[1],
                1.0,
            ]);
        }

        let e_norm = self.solve_8point(&a)?;
        let e = t2.transpose() * e_norm * t1;
        Some(self.enforce_rank2(e))
    }
}

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn gen_range(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.next_u32() as usize) % max
    }

    fn sample_unique(&mut self, max: usize, count: usize) -> Vec<usize> {
        if max == 0 || count == 0 {
            return Vec::new();
        }
        let target = count.min(max);
        let mut picked = std::collections::HashSet::with_capacity(target);
        while picked.len() < target {
            picked.insert(self.gen_range(max));
        }
        let mut indices: Vec<usize> = picked.into_iter().collect();
        indices.sort_unstable();
        indices
    }
}

fn mat3_to_na(mat: &Mat3) -> Matrix3<f32> {
    Matrix3::from_column_slice(&mat.to_cols_array())
}

fn mat3_from_na(mat: &Matrix3<f32>) -> Mat3 {
    Mat3::from_cols_array(&[
        mat[(0, 0)],
        mat[(1, 0)],
        mat[(2, 0)],
        mat[(0, 1)],
        mat[(1, 1)],
        mat[(2, 1)],
        mat[(0, 2)],
        mat[(1, 2)],
        mat[(2, 2)],
    ])
}

fn mat3_to_array(mat: &Matrix3<f32>) -> [[f32; 3]; 3] {
    [
        [mat[(0, 0)], mat[(0, 1)], mat[(0, 2)]],
        [mat[(1, 0)], mat[(1, 1)], mat[(1, 2)]],
        [mat[(2, 0)], mat[(2, 1)], mat[(2, 2)]],
    ]
}

fn normalize_vec3(mut v: [f32; 3]) -> [f32; 3] {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norm > 1e-8 {
        v[0] /= norm;
        v[1] /= norm;
        v[2] /= norm;
    }
    v
}

fn negate_vec3(v: [f32; 3]) -> [f32; 3] {
    [-v[0], -v[1], -v[2]]
}

impl Default for EssentialSolver {
    fn default() -> Self {
        Self::new()
    }
}

fn deterministic_pnp_seed(img_pts: &[[f32; 2]], obj_pts: &[[f32; 3]]) -> u64 {
    let mut seed = 0xcbf29ce484222325u64;

    for point in img_pts {
        seed ^= point[0].to_bits() as u64;
        seed = seed.wrapping_mul(0x100000001b3);
        seed ^= point[1].to_bits() as u64;
        seed = seed.wrapping_mul(0x100000001b3);
    }

    for point in obj_pts {
        seed ^= point[0].to_bits() as u64;
        seed = seed.wrapping_mul(0x100000001b3);
        seed ^= point[1].to_bits() as u64;
        seed = seed.wrapping_mul(0x100000001b3);
        seed ^= point[2].to_bits() as u64;
        seed = seed.wrapping_mul(0x100000001b3);
    }

    seed ^ ((img_pts.len() as u64) << 32) ^ (obj_pts.len() as u64)
}

/// Triangulation solver using DLT (Direct Linear Transform)
pub struct Triangulator {
    /// Minimum triangulation angle (radians)
    pub min_angle: f32,
    /// Minimum triangulation distance
    pub min_dist: f32,
    /// Maximum reprojection error
    pub max_error: f32,
}

impl Triangulator {
    /// Create a new triangulator
    pub fn new() -> Self {
        Self {
            min_angle: (1.5 * PI / 180.0), // modest parallax still usable on handheld video
            min_dist: 0.02,
            max_error: 4.0,
        }
    }

    /// Triangulate 2D points from two views using DLT
    ///
    /// P1, P2: Camera poses (SE3)
    /// pts1, pts2: Corresponding 2D points
    pub fn triangulate(
        &self,
        pose1: &SE3,
        pose2: &SE3,
        pts1: &[[f32; 2]],
        pts2: &[[f32; 2]],
    ) -> Vec<Option<[f32; 3]>> {
        let n = pts1.len().min(pts2.len());
        let mut results = Vec::with_capacity(n);

        // Get camera centers
        let c1 = pose1.inverse().translation();
        let c2 = pose2.inverse().translation();

        // Check triangulation angle
        let baseline = (Vec3::from(c2) - Vec3::from(c1)).length();
        let max_range = baseline * 100.0;

        if baseline < self.min_dist {
            // Baseline too small, return None for all
            return vec![None; n];
        }

        for i in 0..n {
            let pt = self.triangulate_dlt(pose1, pose2, pts1[i], pts2[i]);

            // Check if point is valid
            if let Some(point) = pt {
                // Check if point is in front of both cameras
                let p = Vec3::from(point);
                let ray1 = p - Vec3::from(c1);
                let ray2 = p - Vec3::from(c2);
                let depth1 = pose1.transform_point(&point)[2];
                let depth2 = pose2.transform_point(&point)[2];
                let range1 = ray1.length();
                let range2 = ray2.length();

                // Check angle
                let angle = ray1.angle_between(ray2);

                if angle > self.min_angle
                    && depth1 > 0.0
                    && depth2 > 0.0
                    && range1.is_finite()
                    && range2.is_finite()
                    && range1 <= max_range
                    && range2 <= max_range
                {
                    results.push(Some(point));
                } else {
                    results.push(None);
                }
            } else {
                results.push(None);
            }
        }

        results
    }

    /// DLT-based triangulation
    fn triangulate_dlt(
        &self,
        pose1: &SE3,
        pose2: &SE3,
        p1: [f32; 2],
        p2: [f32; 2],
    ) -> Option<[f32; 3]> {
        let r1 = pose1.rotation_matrix();
        let t1 = pose1.translation();
        let r2 = pose2.rotation_matrix();
        let t2 = pose2.translation();

        let p1m = [
            [r1[0][0], r1[0][1], r1[0][2], t1[0]],
            [r1[1][0], r1[1][1], r1[1][2], t1[1]],
            [r1[2][0], r1[2][1], r1[2][2], t1[2]],
        ];
        let p2m = [
            [r2[0][0], r2[0][1], r2[0][2], t2[0]],
            [r2[1][0], r2[1][1], r2[1][2], t2[1]],
            [r2[2][0], r2[2][1], r2[2][2], t2[2]],
        ];

        let mut a_data = Vec::with_capacity(16);
        a_data.extend_from_slice(&[
            p1[0] * p1m[2][0] - p1m[0][0],
            p1[0] * p1m[2][1] - p1m[0][1],
            p1[0] * p1m[2][2] - p1m[0][2],
            p1[0] * p1m[2][3] - p1m[0][3],
        ]);
        a_data.extend_from_slice(&[
            p1[1] * p1m[2][0] - p1m[1][0],
            p1[1] * p1m[2][1] - p1m[1][1],
            p1[1] * p1m[2][2] - p1m[1][2],
            p1[1] * p1m[2][3] - p1m[1][3],
        ]);
        a_data.extend_from_slice(&[
            p2[0] * p2m[2][0] - p2m[0][0],
            p2[0] * p2m[2][1] - p2m[0][1],
            p2[0] * p2m[2][2] - p2m[0][2],
            p2[0] * p2m[2][3] - p2m[0][3],
        ]);
        a_data.extend_from_slice(&[
            p2[1] * p2m[2][0] - p2m[1][0],
            p2[1] * p2m[2][1] - p2m[1][1],
            p2[1] * p2m[2][2] - p2m[1][2],
            p2[1] * p2m[2][3] - p2m[1][3],
        ]);

        let a = DMatrix::<f32>::from_row_slice(4, 4, &a_data);
        let svd = a.svd(true, true);
        let v_t = svd.v_t?;
        let x = v_t.row(v_t.nrows() - 1);
        let w = x[3];
        if w.abs() < 1e-8 {
            return None;
        }

        Some([x[0] / w, x[1] / w, x[2] / w])
    }

    /// Check if a point is observable from a camera pose
    #[allow(dead_code)]
    fn is_observable(&self, point: &[f32; 3], pose: &SE3) -> bool {
        let cam_center = Vec3::from(pose.inverse().translation());
        let point_vec = Vec3::new(point[0], point[1], point[2]);
        let ray = point_vec - cam_center;

        // Point should be in front of camera (positive z in camera frame)
        let pose_inv = pose.inverse();
        let r = pose_inv.rotation_matrix();
        let z_dir = Vec3::new(r[0][2], r[1][2], r[2][2]);

        ray.dot(z_dir) > 0.0
    }
}

impl Default for Triangulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Sim3 Solver for similarity transform estimation (scale + rotation + translation)
pub struct Sim3Solver {
    /// RANSAC threshold
    pub ransac_threshold: f32,
}

impl Sim3Solver {
    /// Create a new Sim3 solver
    pub fn new(threshold: f32) -> Self {
        Self {
            ransac_threshold: threshold,
        }
    }

    /// Compute Sim3 transform between two sets of 3D points
    ///
    /// Uses the Umeyama algorithm (SVD-based) to compute rotation, scale, and translation.
    /// Returns: (sim3_transform, inliers)
    /// sim3: (scale, translation, rotation)
    pub fn compute(
        &self,
        pts1: &[[f32; 3]],
        pts2: &[[f32; 3]],
    ) -> Option<((f32, [f32; 3], Mat3), Vec<bool>)> {
        if pts1.len() < 3 || pts2.len() < 3 {
            return None;
        }

        let n = pts1.len().min(pts2.len());

        // Compute centroids
        let c1 = self.compute_centroid(pts1);
        let c2 = self.compute_centroid(pts2);

        // Compute scale
        let scale = self.compute_scale(pts1, c1, pts2, c2);

        // Compute rotation using SVD (Umeyama algorithm)
        let rotation = self.compute_rotation(pts1, c1, pts2, c2, n);

        // Compute translation: t = c2 - s * R * c1
        let rc1 = rotation * (Vec3::from(c1) * scale);
        let translation = [c2[0] - rc1.x, c2[1] - rc1.y, c2[2] - rc1.z];

        // Compute inliers
        let mut inliers = vec![false; n];
        for i in 0..n {
            let transformed = self.apply_sim3((scale, translation, rotation), pts1[i]);
            let error = (Vec3::from(transformed) - Vec3::from(pts2[i])).length();
            if error < self.ransac_threshold * 10.0 {
                inliers[i] = true;
            }
        }

        Some(((scale, translation, rotation), inliers))
    }

    /// Compute rotation between two centered point sets using SVD
    fn compute_rotation(
        &self,
        pts1: &[[f32; 3]],
        c1: [f32; 3],
        pts2: &[[f32; 3]],
        c2: [f32; 3],
        n: usize,
    ) -> Mat3 {
        // Build cross-covariance matrix H = sum(q2_i * q1_i^T)
        // where q1 = pts1 - c1, q2 = pts2 - c2
        let mut h = Matrix3::<f32>::zeros();

        for i in 0..n {
            let q1 = NaVec3::new(pts1[i][0] - c1[0], pts1[i][1] - c1[1], pts1[i][2] - c1[2]);
            let q2 = NaVec3::new(pts2[i][0] - c2[0], pts2[i][1] - c2[1], pts2[i][2] - c2[2]);

            // H += q2 * q1^T
            h += q2 * q1.transpose();
        }

        // SVD: H = U * S * V^T
        let svd = h.svd(true, true);
        let u = svd.u.unwrap_or(Matrix3::identity());
        let v_t = svd.v_t.unwrap_or(Matrix3::identity());

        // R = U * diag(1, 1, det(U*V^T)) * V^T
        // This ensures det(R) = +1 (proper rotation)
        let d = (u * v_t).determinant();
        let sign = if d < 0.0 { -1.0 } else { 1.0 };
        let correction = Matrix3::from_diagonal(&NaVec3::new(1.0, 1.0, sign));
        let r = u * correction * v_t;

        mat3_from_na(&r)
    }

    /// Compute centroid of points
    fn compute_centroid(&self, pts: &[[f32; 3]]) -> [f32; 3] {
        let n = pts.len() as f32;
        let mut c = [0.0f32; 3];
        for p in pts {
            c[0] += p[0] / n;
            c[1] += p[1] / n;
            c[2] += p[2] / n;
        }
        c
    }

    /// Compute scale between two point sets
    fn compute_scale(
        &self,
        pts1: &[[f32; 3]],
        c1: [f32; 3],
        pts2: &[[f32; 3]],
        c2: [f32; 3],
    ) -> f32 {
        let _n = pts1.len() as f32;

        let mut d1_sq = 0.0f32;
        let mut d2_sq = 0.0f32;

        for i in 0..pts1.len() {
            let dx = pts1[i][0] - c1[0];
            let dy = pts1[i][1] - c1[1];
            let dz = pts1[i][2] - c1[2];
            d1_sq += dx * dx + dy * dy + dz * dz;

            let dx = pts2[i][0] - c2[0];
            let dy = pts2[i][1] - c2[1];
            let dz = pts2[i][2] - c2[2];
            d2_sq += dx * dx + dy * dy + dz * dz;
        }

        if d1_sq > 1e-8 {
            (d2_sq / d1_sq).sqrt()
        } else {
            1.0
        }
    }

    /// Create a Sim3 transform
    pub fn create_sim3(
        &self,
        scale: f32,
        translation: Vec3,
        rotation: Mat3,
    ) -> (f32, [f32; 3], Mat3) {
        (
            scale,
            [translation.x, translation.y, translation.z],
            rotation,
        )
    }

    /// Apply Sim3 transform to a point
    pub fn apply_sim3(&self, sim3: (f32, [f32; 3], Mat3), point: [f32; 3]) -> [f32; 3] {
        let (scale, translation, rotation) = sim3;
        let p = Vec3::from(point);
        let transformed = rotation * (p * scale) + Vec3::from(translation);
        [transformed.x, transformed.y, transformed.z]
    }
}

impl Default for Sim3Solver {
    fn default() -> Self {
        Self::new(0.01)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// P3P helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Solve the quartic  c4·u⁴ + c3·u³ + c2·u² + c1·u + c0 = 0
/// using the companion matrix (4×4) eigenvalue method.
/// Only real roots are returned (imaginary part < threshold).
fn solve_quartic_companion(c4: f32, c3: f32, c2: f32, c1: f32, c0: f32) -> Vec<f32> {
    if c4.abs() < 1e-10 {
        // Degenerate: fall back to cubic / lower degree
        return solve_cubic_companion(c3, c2, c1, c0);
    }
    // Monic form: divide through by c4
    let a = c3 / c4;
    let b = c2 / c4;
    let c = c1 / c4;
    let d = c0 / c4;

    // Companion matrix of x⁴ + a·x³ + b·x² + c·x + d:
    //  [  0   0   0  -d ]
    //  [  1   0   0  -c ]
    //  [  0   1   0  -b ]
    //  [  0   0   1  -a ]
    //
    // Use the QR iteration (power-method style) approximation via
    // nalgebra's eigenvalue solver on the real 4×4 companion matrix.
    let companion = nalgebra::Matrix4::<f32>::new(
        0.0, 0.0, 0.0, -d, 1.0, 0.0, 0.0, -c, 0.0, 1.0, 0.0, -b, 0.0, 0.0, 1.0, -a,
    );
    let eig = companion.eigenvalues();
    match eig {
        Some(vals) => vals.iter().copied().filter(|v| v.is_finite()).collect(),
        None => Vec::new(),
    }
}

fn solve_cubic_companion(c3: f32, c2: f32, c1: f32, c0: f32) -> Vec<f32> {
    if c3.abs() < 1e-10 {
        return Vec::new();
    }
    let a = c2 / c3;
    let b = c1 / c3;
    let c = c0 / c3;
    let companion = nalgebra::Matrix3::<f32>::new(0.0, 0.0, -c, 1.0, 0.0, -b, 0.0, 1.0, -a);
    match companion.eigenvalues() {
        Some(vals) => vals.iter().copied().filter(|v| v.is_finite()).collect(),
        None => Vec::new(),
    }
}

/// Given the three camera-frame 3D positions of the world points and their
/// known world positions, recover the camera pose (R, t) via SVD Procrustes.
fn recover_pose_from_3points(cam_pts: &[Vec3; 3], world_pts: &[Vec3; 3]) -> Option<SE3> {
    // Centroids
    let cam_c = (cam_pts[0] + cam_pts[1] + cam_pts[2]) / 3.0;
    let world_c = (world_pts[0] + world_pts[1] + world_pts[2]) / 3.0;

    // Cross-covariance  H = Σ (cam_i - cam_c)(world_i - world_c)ᵀ
    // We want R such that cam_centred ≈ R · world_centred
    // i.e. H = Σ cam_centred_i · world_centred_iᵀ
    let mut h = nalgebra::Matrix3::<f32>::zeros();
    for i in 0..3 {
        let dc = cam_pts[i] - cam_c;
        let dw = world_pts[i] - world_c;
        h += nalgebra::Vector3::new(dc.x, dc.y, dc.z) * nalgebra::RowVector3::new(dw.x, dw.y, dw.z);
    }

    let svd = h.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;

    let mut correction = Matrix3::identity();
    if (u.clone() * v_t.clone()).determinant() < 0.0 {
        correction[(2, 2)] = -1.0;
    }
    let r_na = u * correction * v_t;

    // t = cam_centroid - R · world_centroid
    let wc = nalgebra::Vector3::new(world_c.x, world_c.y, world_c.z);
    let t_na = nalgebra::Vector3::new(cam_c.x, cam_c.y, cam_c.z) - r_na * wc;

    let r_arr = mat3_to_array(&r_na);
    Some(SE3::from_rotation_translation(
        &r_arr,
        &[t_na[0], t_na[1], t_na[2]],
    ))
}
