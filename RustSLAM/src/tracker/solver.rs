//! Geometric solvers for Visual Odometry
//!
//! Implements PnP, Essential Matrix, Triangulation, and Sim3 solvers with proper algorithms.

use crate::colmap_rng::{sample_unique_indices, ColmapMt19937};
use crate::core::SE3;
use crate::features::base::Match;
use glam::{Mat3, Vec3};
use nalgebra::{
    DMatrix, Matrix3, Matrix4, SMatrix, SVector, SymmetricEigen, Vector3 as NaVec3, Vector4,
};
use std::f32::consts::PI;

type Vec3d = nalgebra::Vector3<f64>;
type Vec4d = Vector4<f64>;
type Mat3d = Matrix3<f64>;
type Mat4d = Matrix4<f64>;
type Mat3x4d = SMatrix<f64, 3, 4>;
type Mat3x7d = SMatrix<f64, 3, 7>;
type Mat3x10d = SMatrix<f64, 3, 10>;
type Mat8x4d = SMatrix<f64, 8, 4>;
type Mat6x10d = SMatrix<f64, 6, 10>;
type Mat12d = SMatrix<f64, 12, 12>;

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
    pub ransac_min_inlier_ratio: f32,
    pub ransac_dyn_num_trials_multiplier: f32,
    pub ransac_min_iterations: u32,
    pub ransac_max_iterations: u32,
    pub ransac_random_seed: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
struct InlierSupport {
    num_inliers: usize,
    residual_sum: f64,
}

impl Default for InlierSupport {
    fn default() -> Self {
        Self {
            num_inliers: 0,
            residual_sum: f64::MAX,
        }
    }
}

impl InlierSupport {
    fn is_better_than(&self, other: &Self) -> bool {
        self.num_inliers > other.num_inliers
            || (self.num_inliers == other.num_inliers && self.residual_sum < other.residual_sum)
    }
}

struct PoseEvaluation {
    inliers: Vec<bool>,
    support: InlierSupport,
}

#[derive(Debug, Clone)]
pub struct PnPFocalResult {
    pub pose: SE3,
    pub inliers: Vec<bool>,
    pub focal: f32,
}

#[derive(Debug, Clone, Copy)]
struct FocalPoseModel {
    pose: SE3,
    focal: f64,
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
            ransac_min_inlier_ratio: 0.0,
            ransac_dyn_num_trials_multiplier: 3.0,
            ransac_min_iterations: 0,
            ransac_max_iterations: 200,
            ransac_random_seed: None,
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
        let seed = self
            .ransac_random_seed
            .unwrap_or_else(|| deterministic_pnp_seed(&normalized_pts, &problem.object_points));
        let mut rng = ColmapMt19937::new(seed);

        // RANSAC loop
        let mut best_inliers: Vec<bool> = vec![false; n];
        let mut best_pose: Option<SE3> = None;
        let mut best_support = InlierSupport::default();
        let sample_size = 3usize; // true P3P uses exactly 3 correspondences
        let max_iterations = self.initial_max_iterations(sample_size);
        let mut adaptive_max_iter = max_iterations;

        for iter in 0..max_iterations {
            if iter >= adaptive_max_iter {
                break;
            }
            // Randomly select 3 points for true P3P hypothesis
            let indices = self.random_indices(&mut rng, n, sample_size);

            // Solve for this sample
            if let Some(poses) = self.solve_p3p(&normalized_pts, &problem.object_points, &indices) {
                // For each P3P solution, check all points
                for pose in poses {
                    let eval = self.evaluate_pose(
                        &pose,
                        &normalized_pts,
                        &problem.object_points,
                        threshold,
                    );

                    if eval.support.is_better_than(&best_support) {
                        let (local_pose, local_eval) = self.local_optimize_pose(
                            pose,
                            eval,
                            &normalized_pts,
                            &problem.object_points,
                            threshold,
                        );

                        if local_eval.support.is_better_than(&best_support) {
                            best_support = local_eval.support;
                            best_inliers = local_eval.inliers;
                            best_pose = Some(local_pose);
                        }
                    }
                }
            }

            if best_support.num_inliers >= sample_size {
                adaptive_max_iter = self
                    .update_adaptive_max_iterations(best_support.num_inliers, n, sample_size)
                    .min(max_iterations);
            }
        }

        if best_pose.is_none() {
            return None;
        }

        // Refine pose with all inliers
        if let Some(ref mut pose) = best_pose {
            if best_support.num_inliers >= 4 {
                *pose =
                    self.refine_pose(pose, &normalized_pts, &problem.object_points, &best_inliers);
                let refined_eval =
                    self.evaluate_pose(pose, &normalized_pts, &problem.object_points, threshold);
                if refined_eval.support.num_inliers >= 4 {
                    best_inliers = refined_eval.inliers;
                }
            }
        }

        Some((best_pose.unwrap_or(SE3::identity()), best_inliers))
    }

    /// Solve absolute pose while estimating a shared focal length.
    ///
    /// This follows COLMAP's PNPF data flow at the RANSAC level: image points
    /// are centered by the principal point, hypotheses are scored in pixels,
    /// and the returned focal is a single value shared by all focal parameters.
    /// The model generator tries a PoseLib-style P4PF algebraic solver first
    /// and falls back to four-point P3P hypotheses plus focal updates for
    /// numerical coverage.
    pub fn solve_with_estimated_focal(&self, problem: &PnPProblem) -> Option<PnPFocalResult> {
        if !problem.is_solvable() {
            return None;
        }

        let n = problem.image_points.len();
        if n < 4 || n != problem.object_points.len() {
            return None;
        }

        let centered_pts: Vec<[f32; 2]> = problem
            .image_points
            .iter()
            .map(|p| [p[0] - self.cx, p[1] - self.cy])
            .collect();

        let threshold = self.ransac_threshold.max(1.0e-6);
        let seed = self
            .ransac_random_seed
            .unwrap_or_else(|| deterministic_pnp_seed(&centered_pts, &problem.object_points));
        let mut rng = ColmapMt19937::new(seed);

        let sample_size = 4usize;
        let max_iterations = self.initial_max_iterations(sample_size);
        let mut adaptive_max_iter = max_iterations;
        let mut best_model = None::<FocalPoseModel>;
        let mut best_eval = PoseEvaluation {
            inliers: vec![false; n],
            support: InlierSupport::default(),
        };

        for iter in 0..max_iterations {
            if iter >= adaptive_max_iter {
                break;
            }

            let indices = self.random_indices(&mut rng, n, sample_size);
            for model in
                self.estimate_focal_pose_models(&centered_pts, &problem.object_points, &indices)
            {
                let eval = self.evaluate_focal_pose(
                    &model,
                    &centered_pts,
                    &problem.object_points,
                    threshold,
                );

                if eval.support.is_better_than(&best_eval.support) {
                    best_eval = eval;
                    best_model = Some(model);
                }
            }

            if best_eval.support.num_inliers >= sample_size {
                adaptive_max_iter = self
                    .update_adaptive_max_iterations(best_eval.support.num_inliers, n, sample_size)
                    .min(max_iterations);
            }
        }

        let mut best_model = best_model?;
        for _ in 0..10 {
            if best_eval.support.num_inliers < 4 {
                break;
            }
            let prev_num_inliers = best_eval.support.num_inliers;
            let Some(refined) = self.refine_focal_pose_model(
                best_model,
                &centered_pts,
                &problem.object_points,
                &best_eval.inliers,
            ) else {
                break;
            };
            let refined_eval = self.evaluate_focal_pose(
                &refined,
                &centered_pts,
                &problem.object_points,
                threshold,
            );
            if refined_eval.support.is_better_than(&best_eval.support) {
                best_model = refined;
                best_eval = refined_eval;
            }
            if best_eval.support.num_inliers <= prev_num_inliers {
                break;
            }
        }

        if best_eval.support.num_inliers < sample_size || !best_model.focal.is_finite() {
            return None;
        }

        Some(PnPFocalResult {
            pose: best_model.pose,
            inliers: best_eval.inliers,
            focal: best_model.focal as f32,
        })
    }

    fn initial_max_iterations(&self, sample_size: usize) -> u32 {
        if self.ransac_min_inlier_ratio <= 0.0 {
            return self.ransac_max_iterations.max(self.ransac_min_iterations);
        }
        let assumed_samples = 100_000usize;
        let assumed_inliers =
            (self.ransac_min_inlier_ratio.clamp(0.0, 1.0) * assumed_samples as f32) as usize;
        let dyn_max = compute_ransac_num_trials(
            assumed_inliers,
            assumed_samples,
            sample_size,
            self.ransac_confidence,
            self.ransac_dyn_num_trials_multiplier,
        );
        self.ransac_max_iterations.min(dyn_max)
    }

    fn update_adaptive_max_iterations(
        &self,
        num_inliers: usize,
        num_samples: usize,
        sample_size: usize,
    ) -> u32 {
        compute_ransac_num_trials(
            num_inliers,
            num_samples,
            sample_size,
            self.ransac_confidence,
            self.ransac_dyn_num_trials_multiplier,
        )
        .max(self.ransac_min_iterations)
    }

    fn local_optimize_pose(
        &self,
        initial_pose: SE3,
        initial_eval: PoseEvaluation,
        normalized_pts: &[[f32; 2]],
        object_points: &[[f32; 3]],
        threshold: f32,
    ) -> (SE3, PoseEvaluation) {
        let mut best_pose = initial_pose;
        let mut best_eval = initial_eval;
        if best_eval.support.num_inliers <= 3 || best_eval.support.num_inliers < 4 {
            return (best_pose, best_eval);
        }

        for _ in 0..10 {
            let prev_best_num_inliers = best_eval.support.num_inliers;
            let mut inlier_img = Vec::with_capacity(best_eval.support.num_inliers);
            let mut inlier_obj = Vec::with_capacity(best_eval.support.num_inliers);
            for i in 0..best_eval.inliers.len() {
                if best_eval.inliers[i] {
                    inlier_img.push(normalized_pts[i]);
                    inlier_obj.push(object_points[i]);
                }
            }

            if inlier_img.len() < 4 {
                break;
            }

            let Some(local_pose) = self.estimate_pose_epnp(&inlier_img, &inlier_obj) else {
                break;
            };
            let local_eval =
                self.evaluate_pose(&local_pose, normalized_pts, object_points, threshold);
            if local_eval.support.is_better_than(&best_eval.support) {
                best_pose = local_pose;
                best_eval = local_eval;
            }

            if best_eval.support.num_inliers <= prev_best_num_inliers {
                break;
            }
        }

        (best_pose, best_eval)
    }

    fn evaluate_pose(
        &self,
        pose: &SE3,
        normalized_pts: &[[f32; 2]],
        object_points: &[[f32; 3]],
        threshold: f32,
    ) -> PoseEvaluation {
        let n = normalized_pts.len().min(object_points.len());
        let mut inliers = vec![false; n];
        let mut support = InlierSupport {
            num_inliers: 0,
            residual_sum: 0.0,
        };
        let max_residual = threshold * threshold;

        for i in 0..n {
            let projected = self.project_point(pose, &object_points[i]);
            let error = self.reprojection_error(&normalized_pts[i], &projected);
            let residual = error * error;
            if residual <= max_residual {
                inliers[i] = true;
                support.num_inliers += 1;
                support.residual_sum += residual as f64;
            }
        }

        PoseEvaluation { inliers, support }
    }

    fn evaluate_focal_pose(
        &self,
        model: &FocalPoseModel,
        centered_pts: &[[f32; 2]],
        object_points: &[[f32; 3]],
        threshold_px: f32,
    ) -> PoseEvaluation {
        let n = centered_pts.len().min(object_points.len());
        let mut inliers = vec![false; n];
        let mut support = InlierSupport {
            num_inliers: 0,
            residual_sum: 0.0,
        };
        let max_residual = threshold_px * threshold_px;

        for i in 0..n {
            let point_cam = model.pose.transform_point(&object_points[i]);
            let z = point_cam[2] as f64;
            let residual = if z > f64::EPSILON {
                let projected = [
                    (model.focal * point_cam[0] as f64 / z) as f32,
                    (model.focal * point_cam[1] as f64 / z) as f32,
                ];
                let dx = centered_pts[i][0] - projected[0];
                let dy = centered_pts[i][1] - projected[1];
                dx * dx + dy * dy
            } else {
                f32::MAX
            };
            if residual.is_finite() && residual <= max_residual {
                inliers[i] = true;
                support.num_inliers += 1;
                support.residual_sum += residual as f64;
            }
        }

        PoseEvaluation { inliers, support }
    }

    fn random_indices(&self, rng: &mut ColmapMt19937, n: usize, k: usize) -> Vec<usize> {
        sample_unique_indices(rng, n, k)
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

        let rays = [
            Vec3d::new(img_pts[i0][0] as f64, img_pts[i0][1] as f64, 1.0).normalize(),
            Vec3d::new(img_pts[i1][0] as f64, img_pts[i1][1] as f64, 1.0).normalize(),
            Vec3d::new(img_pts[i2][0] as f64, img_pts[i2][1] as f64, 1.0).normalize(),
        ];
        let world = [
            world_point_to_vec3d(&obj_pts[i0]),
            world_point_to_vec3d(&obj_pts[i1]),
            world_point_to_vec3d(&obj_pts[i2]),
        ];

        let poses = solve_p3p_poselib(&rays, &world);

        if poses.is_empty() {
            None
        } else {
            Some(poses)
        }
    }

    fn estimate_focal_pose_models(
        &self,
        centered_pts: &[[f32; 2]],
        object_points: &[[f32; 3]],
        indices: &[usize],
    ) -> Vec<FocalPoseModel> {
        if indices.len() < 4 {
            return Vec::new();
        }

        let mut sample_img = Vec::with_capacity(indices.len());
        let mut sample_obj = Vec::with_capacity(indices.len());
        for &idx in indices {
            if idx >= centered_pts.len() || idx >= object_points.len() {
                return Vec::new();
            }
            sample_img.push(centered_pts[idx]);
            sample_obj.push(object_points[idx]);
        }

        let mut models = estimate_focal_pose_p4pf(&sample_img, &sample_obj);
        models.extend(self.estimate_focal_pose_models_p3p(&sample_img, &sample_obj));

        let base_focal = self.initial_focal_guess(centered_pts);
        for scale in [1.0, 0.75, 4.0 / 3.0, 0.5, 2.0, 0.25, 4.0, 8.0] {
            let mut focal = base_focal * scale;
            if !focal.is_finite() || focal <= 1.0e-6 {
                continue;
            }

            let mut model = None;
            for _ in 0..4 {
                let normalized = sample_img
                    .iter()
                    .map(|p| [p[0] / focal as f32, p[1] / focal as f32])
                    .collect::<Vec<_>>();
                let Some(pose) = self.estimate_pose_epnp(&normalized, &sample_obj) else {
                    break;
                };
                let Some(updated_focal) =
                    estimate_focal_from_centered_pose(pose, &sample_img, &sample_obj)
                else {
                    break;
                };
                model = Some(FocalPoseModel {
                    pose,
                    focal: updated_focal,
                });
                if (updated_focal - focal).abs() <= 1.0e-7 * focal.max(1.0) {
                    break;
                }
                focal = updated_focal;
            }

            if let Some(model) = model {
                if model.focal.is_finite() && model.focal > 0.0 {
                    models.push(model);
                }
            }
        }

        if sample_img.len() >= 6 {
            if let Some(model) = estimate_focal_pose_dlt(&sample_img, &sample_obj) {
                models.push(model);
            }
        }

        deduplicate_focal_models(models)
    }

    fn estimate_focal_pose_models_p3p(
        &self,
        centered_pts: &[[f32; 2]],
        object_points: &[[f32; 3]],
    ) -> Vec<FocalPoseModel> {
        if centered_pts.len().min(object_points.len()) < 4 {
            return Vec::new();
        }

        let base_focal = self.initial_focal_guess(centered_pts);
        let mut models = Vec::new();
        let p3p_triples = [
            [0usize, 1usize, 2usize],
            [0usize, 1usize, 3usize],
            [0usize, 2usize, 3usize],
            [1usize, 2usize, 3usize],
        ];
        for scale in [0.25, 0.5, 0.75, 1.0, 4.0 / 3.0, 2.0, 4.0, 8.0] {
            let focal = base_focal * scale;
            if !focal.is_finite() || focal <= 1.0e-6 {
                continue;
            }
            let normalized = centered_pts
                .iter()
                .map(|p| [p[0] / focal as f32, p[1] / focal as f32])
                .collect::<Vec<_>>();
            for triple in p3p_triples {
                let Some(poses) = self.solve_p3p(&normalized, object_points, &triple) else {
                    continue;
                };
                for pose in poses {
                    let Some(updated_focal) =
                        estimate_focal_from_centered_pose(pose, centered_pts, object_points)
                    else {
                        continue;
                    };
                    if !updated_focal.is_finite() || updated_focal <= 0.0 {
                        continue;
                    }
                    let model = FocalPoseModel {
                        pose,
                        focal: updated_focal,
                    };
                    if let Some(refined) = self.refine_focal_pose_model_gauss_newton(
                        model,
                        centered_pts,
                        object_points,
                    ) {
                        models.push(refined);
                    } else if focal_pose_model_cost(model, centered_pts, object_points).is_some() {
                        models.push(model);
                    }
                }
            }
        }

        models
    }

    fn refine_focal_pose_model(
        &self,
        initial_model: FocalPoseModel,
        centered_pts: &[[f32; 2]],
        object_points: &[[f32; 3]],
        inliers: &[bool],
    ) -> Option<FocalPoseModel> {
        let mut inlier_img = Vec::new();
        let mut inlier_obj = Vec::new();
        for (idx, &is_inlier) in inliers.iter().enumerate() {
            if is_inlier && idx < centered_pts.len() && idx < object_points.len() {
                inlier_img.push(centered_pts[idx]);
                inlier_obj.push(object_points[idx]);
            }
        }
        if inlier_img.len() < 4 {
            return None;
        }

        let mut model = initial_model;
        let mut focal = model.focal;
        for _ in 0..6 {
            let normalized = inlier_img
                .iter()
                .map(|p| [p[0] / focal as f32, p[1] / focal as f32])
                .collect::<Vec<_>>();
            let pose = self.estimate_pose_epnp(&normalized, &inlier_obj)?;
            let updated_focal = estimate_focal_from_centered_pose(pose, &inlier_img, &inlier_obj)?;
            if !updated_focal.is_finite() || updated_focal <= 0.0 {
                return None;
            }
            model = FocalPoseModel {
                pose,
                focal: updated_focal,
            };
            if (updated_focal - focal).abs() <= 1.0e-7 * focal.max(1.0) {
                break;
            }
            focal = updated_focal;
        }

        self.refine_focal_pose_model_gauss_newton(model, &inlier_img, &inlier_obj)
            .or(Some(model))
    }

    fn refine_focal_pose_model_gauss_newton(
        &self,
        initial_model: FocalPoseModel,
        centered_pts: &[[f32; 2]],
        object_points: &[[f32; 3]],
    ) -> Option<FocalPoseModel> {
        if centered_pts.len().min(object_points.len()) < 4 {
            return None;
        }

        let mut model = initial_model;
        let mut best_cost = focal_pose_model_cost(model, centered_pts, object_points)?;
        let mut damping = 1.0e-6;
        let eps = 1.0e-4f64;
        for _ in 0..12 {
            let mut h = SMatrix::<f64, 7, 7>::zeros();
            let mut b = SVector::<f64, 7>::zeros();
            let mut valid_obs = 0usize;

            for (obs, obj) in centered_pts.iter().zip(object_points.iter()) {
                let projected = project_focal_pose_model(model, obj)?;
                let residual = [obs[0] as f64 - projected[0], obs[1] as f64 - projected[1]];

                let mut jac = SMatrix::<f64, 2, 7>::zeros();
                for k in 0..7 {
                    let perturbed = perturb_focal_pose_model(model, k, eps)?;
                    let projected_plus = project_focal_pose_model(perturbed, obj)?;
                    jac[(0, k)] = (projected_plus[0] - projected[0]) / eps;
                    jac[(1, k)] = (projected_plus[1] - projected[1]) / eps;
                }

                for i in 0..7 {
                    b[i] += jac[(0, i)] * residual[0] + jac[(1, i)] * residual[1];
                    for j in 0..7 {
                        h[(i, j)] += jac[(0, i)] * jac[(0, j)] + jac[(1, i)] * jac[(1, j)];
                    }
                }
                valid_obs += 1;
            }

            if valid_obs < 4 {
                break;
            }
            for i in 0..7 {
                h[(i, i)] += damping;
            }

            let Some(delta) = h.lu().solve(&b) else {
                damping *= 10.0;
                continue;
            };
            if !delta.iter().all(|value| value.is_finite()) || delta.norm() > 10.0 {
                damping *= 10.0;
                continue;
            }

            let mut accepted = false;
            for step in [1.0, 0.5, 0.25, 0.125, 0.0625] {
                let candidate = apply_focal_pose_delta(model, &delta, step)?;
                let Some(cost) = focal_pose_model_cost(candidate, centered_pts, object_points)
                else {
                    continue;
                };
                if cost + 1.0e-12 < best_cost {
                    model = candidate;
                    best_cost = cost;
                    damping = (damping * 0.5).max(1.0e-10);
                    accepted = true;
                    break;
                }
            }

            if delta.norm() < 1.0e-8 {
                break;
            }
            if !accepted {
                damping *= 4.0;
            }
        }

        Some(model)
    }

    fn initial_focal_guess(&self, centered_pts: &[[f32; 2]]) -> f64 {
        let configured = ((self.fx as f64).abs() + (self.fy as f64).abs()) * 0.5;
        if configured.is_finite() && configured > 1.0e-6 {
            return configured;
        }
        let mean_radius = centered_pts
            .iter()
            .map(|p| (p[0] as f64).hypot(p[1] as f64))
            .sum::<f64>()
            / centered_pts.len().max(1) as f64;
        mean_radius.max(1.0)
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

    pub(crate) fn estimate_pose_epnp(
        &self,
        img_pts: &[[f32; 2]],
        obj_pts: &[[f32; 3]],
    ) -> Option<SE3> {
        let mut estimator = EpnpEstimator::new(img_pts, obj_pts)?;
        let (rotation, translation) = estimator.compute_pose()?;

        if rotation.iter().any(|v| !v.is_finite()) || translation.iter().any(|v| !v.is_finite()) {
            return None;
        }

        let rotation = [
            [
                rotation[(0, 0)] as f32,
                rotation[(0, 1)] as f32,
                rotation[(0, 2)] as f32,
            ],
            [
                rotation[(1, 0)] as f32,
                rotation[(1, 1)] as f32,
                rotation[(1, 2)] as f32,
            ],
            [
                rotation[(2, 0)] as f32,
                rotation[(2, 1)] as f32,
                rotation[(2, 2)] as f32,
            ],
        ];
        let translation = [
            translation[0] as f32,
            translation[1] as f32,
            translation[2] as f32,
        ];

        Some(SE3::from_rotation_translation(&rotation, &translation))
    }
}

struct EpnpEstimator<'a> {
    image_points: &'a [[f32; 2]],
    world_points: &'a [[f32; 3]],
    cws: [Vec3d; 4],
    ccs: [Vec3d; 4],
    alphas: Vec<Vec4d>,
    pcs: Vec<Vec3d>,
}

fn solve_p3p_poselib(rays_in: &[Vec3d; 3], world_in: &[Vec3d; 3]) -> Vec<SE3> {
    let mut x01 = world_in[0] - world_in[1];
    let mut x02 = world_in[0] - world_in[2];
    let x12 = world_in[1] - world_in[2];

    let mut a01 = x01.norm_squared();
    let mut a02 = x02.norm_squared();
    let mut a12 = x12.norm_squared();
    if a01 <= 1.0e-16 || a02 <= 1.0e-16 || a12 <= 1.0e-16 {
        return Vec::new();
    }

    let mut world = *world_in;
    let mut rays = *rays_in;

    if a01 > a02 {
        if a01 > a12 {
            rays.swap(0, 2);
            world.swap(0, 2);
            std::mem::swap(&mut a01, &mut a12);
            x01 = -x12;
            x02 = -x02;
        }
    } else if a02 > a12 {
        rays.swap(0, 1);
        world.swap(0, 1);
        std::mem::swap(&mut a02, &mut a12);
        x01 = -x01;
        x02 = x12;
    }

    let a12_inv = 1.0 / a12;
    let a = a01 * a12_inv;
    let b = a02 * a12_inv;

    let m01 = rays[0].dot(&rays[1]);
    let m02 = rays[0].dot(&rays[2]);
    let m12 = rays[1].dot(&rays[2]);

    let m12sq = -m12 * m12 + 1.0;
    let m02sq = -1.0 + m02 * m02;
    let m01sq = -1.0 + m01 * m01;
    let ab = a * b;
    let bsq = b * b;
    let asq = a * a;
    let m013 = -2.0 + 2.0 * m01 * m02 * m12;
    let bsqm12sq = bsq * m12sq;
    let asqm12sq = asq * m12sq;
    let abm12sq = 2.0 * ab * m12sq;

    let k3_den = bsqm12sq + b * m02sq;
    if k3_den.abs() <= 1.0e-15 {
        return Vec::new();
    }
    let k3_inv = 1.0 / k3_den;
    let k2 = k3_inv * ((-1.0 + a) * m02sq + abm12sq + bsqm12sq + b * m013);
    let k1 = k3_inv * (asqm12sq + abm12sq + a * m013 + (-1.0 + b) * m01sq);
    let k0 = k3_inv * (asqm12sq + a * m01sq);

    let (single_real_root, s) = solve_cubic_single_real(k2, k1, k0);

    let mut c = Mat3d::zeros();
    c[(0, 0)] = -a + s * (1.0 - b);
    c[(0, 1)] = -m02 * s;
    c[(0, 2)] = a * m12 + b * m12 * s;
    c[(1, 0)] = c[(0, 1)];
    c[(1, 1)] = s + 1.0;
    c[(1, 2)] = -m01;
    c[(2, 0)] = c[(0, 2)];
    c[(2, 1)] = c[(1, 2)];
    c[(2, 2)] = -a - b * s + 1.0;

    let pq = compute_pq(c);
    let Some(xx) = Mat3d::from_columns(&[x01, x02, x01.cross(&x02)]).try_inverse() else {
        return Vec::new();
    };

    let mut poses = Vec::with_capacity(4);
    for p in pq {
        let p0 = p[0];
        let p1 = p[1];
        let p2 = p[2];
        if p0.abs() <= 1.0e-15 && p1.abs() <= 1.0e-15 {
            continue;
        }

        let switch_12 = p0.abs() <= p1.abs();
        if switch_12 {
            let w0 = -p0 / p1;
            let w1 = -p2 / p1;
            let ca_den = w1 * w1 - b;
            if ca_den.abs() <= 1.0e-15 {
                continue;
            }
            let ca = 1.0 / ca_den;
            let cb = 2.0 * (b * m12 - m02 * w1 + w0 * w1) * ca;
            let cc = (w0 * w0 - 2.0 * m02 * w0 - b + 1.0) * ca;
            for tau in root2real(cb, cc) {
                if tau <= 0.0 {
                    continue;
                }
                let d2_den = tau * (tau - 2.0 * m12) + 1.0;
                if d2_den <= 1.0e-15 {
                    continue;
                }
                let mut d2 = (a12 / d2_den).sqrt();
                let mut d1 = tau * d2;
                let mut d0 = w0 * d2 + w1 * d1;
                if d0 < 0.0 {
                    continue;
                }
                refine_lambda(&mut d0, &mut d1, &mut d2, a01, a02, a12, m01, m02, m12);
                push_p3p_pose(&mut poses, &rays, &world, &xx, d0, d1, d2);
            }
        } else {
            let w0 = -p1 / p0;
            let w1 = -p2 / p0;
            let ca_den = -a * w1 * w1 + 2.0 * a * m12 * w1 - a + 1.0;
            if ca_den.abs() <= 1.0e-15 {
                continue;
            }
            let ca = 1.0 / ca_den;
            let cb = 2.0 * (a * m12 * w0 - m01 - a * w0 * w1) * ca;
            let cc = (1.0 - a * w0 * w0) * ca;
            for tau in root2real(cb, cc) {
                if tau <= 0.0 {
                    continue;
                }
                let d0_den = tau * (tau - 2.0 * m01) + 1.0;
                if d0_den <= 1.0e-15 {
                    continue;
                }
                let mut d0 = (a01 / d0_den).sqrt();
                let mut d1 = tau * d0;
                let mut d2 = w0 * d0 + w1 * d1;
                if d2 < 0.0 {
                    continue;
                }
                refine_lambda(&mut d0, &mut d1, &mut d2, a01, a02, a12, m01, m02, m12);
                push_p3p_pose(&mut poses, &rays, &world, &xx, d0, d1, d2);
            }
        }

        if !poses.is_empty() && single_real_root {
            break;
        }
    }

    poses
}

fn push_p3p_pose(
    poses: &mut Vec<SE3>,
    rays: &[Vec3d; 3],
    world: &[Vec3d; 3],
    xx: &Mat3d,
    d0: f64,
    d1: f64,
    d2: f64,
) {
    if !d0.is_finite() || !d1.is_finite() || !d2.is_finite() {
        return;
    }
    let v1 = d0 * rays[0] - d1 * rays[1];
    let v2 = d0 * rays[0] - d2 * rays[2];
    let yy = Mat3d::from_columns(&[v1, v2, v1.cross(&v2)]);
    let rotation = yy * xx;
    if rotation.iter().any(|v| !v.is_finite()) || rotation.determinant().abs() < 1.0e-8 {
        return;
    }
    let translation = d0 * rays[0] - rotation * world[0];
    if translation.iter().any(|v| !v.is_finite()) {
        return;
    }
    poses.push(se3_from_f64_rt(&rotation, &translation));
}

fn compute_pq(c: Mat3d) -> [Vec3d; 2] {
    let mut c_adj = Mat3d::zeros();
    c_adj[(0, 0)] = c[(1, 2)] * c[(2, 1)] - c[(1, 1)] * c[(2, 2)];
    c_adj[(1, 1)] = c[(0, 2)] * c[(2, 0)] - c[(0, 0)] * c[(2, 2)];
    c_adj[(2, 2)] = c[(0, 1)] * c[(1, 0)] - c[(0, 0)] * c[(1, 1)];
    c_adj[(0, 1)] = c[(0, 1)] * c[(2, 2)] - c[(0, 2)] * c[(2, 1)];
    c_adj[(0, 2)] = c[(0, 2)] * c[(1, 1)] - c[(0, 1)] * c[(1, 2)];
    c_adj[(1, 0)] = c_adj[(0, 1)];
    c_adj[(1, 2)] = c[(0, 0)] * c[(1, 2)] - c[(0, 2)] * c[(1, 0)];
    c_adj[(2, 0)] = c_adj[(0, 2)];
    c_adj[(2, 1)] = c_adj[(1, 2)];

    let diag = [c_adj[(0, 0)], c_adj[(1, 1)], c_adj[(2, 2)]];
    let max_idx = if diag[0] > diag[1] {
        if diag[0] > diag[2] {
            0
        } else {
            2
        }
    } else if diag[1] > diag[2] {
        1
    } else {
        2
    };
    let denom = diag[max_idx].abs().sqrt().max(1.0e-15);
    let v = c_adj.column(max_idx).into_owned() / denom;

    let mut c_shifted = c;
    c_shifted[(0, 1)] -= v[2];
    c_shifted[(0, 2)] += v[1];
    c_shifted[(1, 2)] -= v[0];
    c_shifted[(1, 0)] += v[2];
    c_shifted[(2, 0)] -= v[1];
    c_shifted[(2, 1)] += v[0];

    [
        c_shifted.column(0).into_owned(),
        Vec3d::new(c_shifted[(0, 0)], c_shifted[(0, 1)], c_shifted[(0, 2)]),
    ]
}

fn root2real(b: f64, c: f64) -> Vec<f64> {
    let threshold = -1.0e-12;
    let discriminant = b * b - 4.0 * c;
    if discriminant < threshold {
        return Vec::new();
    }
    if discriminant > threshold && discriminant < 0.0 {
        return vec![-0.5 * b];
    }
    let sqrt_disc = discriminant.max(0.0).sqrt();
    if b < 0.0 {
        vec![0.5 * (-b + sqrt_disc), 0.5 * (-b - sqrt_disc)]
    } else {
        let r1 = if (-b + sqrt_disc).abs() > 1.0e-15 {
            2.0 * c / (-b + sqrt_disc)
        } else {
            -0.5 * b
        };
        let r2 = if (-b - sqrt_disc).abs() > 1.0e-15 {
            2.0 * c / (-b - sqrt_disc)
        } else {
            -0.5 * b
        };
        vec![r1, r2]
    }
}

fn solve_cubic_single_real(c2: f64, c1: f64, c0: f64) -> (bool, f64) {
    let a = c1 - c2 * c2 / 3.0;
    let mut b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
    let mut c = b * b / 4.0 + a * a * a / 27.0;
    if c != 0.0 {
        if c > 0.0 {
            c = c.sqrt();
            b *= -0.5;
            let root = (b + c).cbrt() + (b - c).cbrt() - c2 / 3.0;
            return (true, root);
        }
        let denom = 2.0 * a;
        if denom.abs() <= 1.0e-15 || a >= 0.0 {
            return (false, -c2 / 3.0);
        }
        c = 3.0 * b / denom * (-3.0 / a).sqrt();
        let root = 2.0 * (-a / 3.0).sqrt() * c.clamp(-1.0, 1.0).acos().cos() - c2 / 3.0;
        return (false, root);
    }
    let root = -c2 / 3.0 + if a != 0.0 { 3.0 * b / a } else { 0.0 };
    (false, root)
}

fn refine_lambda(
    lambda1: &mut f64,
    lambda2: &mut f64,
    lambda3: &mut f64,
    a12: f64,
    a13: f64,
    a23: f64,
    b12: f64,
    b13: f64,
    b23: f64,
) {
    for _ in 0..5 {
        let r1 = *lambda1 * *lambda1 - 2.0 * *lambda1 * *lambda2 * b12 + *lambda2 * *lambda2 - a12;
        let r2 = *lambda1 * *lambda1 - 2.0 * *lambda1 * *lambda3 * b13 + *lambda3 * *lambda3 - a13;
        let r3 = *lambda2 * *lambda2 - 2.0 * *lambda2 * *lambda3 * b23 + *lambda3 * *lambda3 - a23;
        if r1.abs() + r2.abs() + r3.abs() < 1.0e-10 {
            return;
        }
        let x11 = *lambda1 - *lambda2 * b12;
        let x12 = *lambda2 - *lambda1 * b12;
        let x21 = *lambda1 - *lambda3 * b13;
        let x23 = *lambda3 - *lambda1 * b13;
        let x32 = *lambda2 - *lambda3 * b23;
        let x33 = *lambda3 - *lambda2 * b23;
        let det_den = x11 * x23 * x32 + x12 * x21 * x33;
        if det_den.abs() <= 1.0e-15 {
            return;
        }
        let det_j = 0.5 / det_den;
        *lambda1 += (-x23 * x32 * r1 - x12 * x33 * r2 + x12 * x23 * r3) * det_j;
        *lambda2 += (-x21 * x33 * r1 + x11 * x33 * r2 - x11 * x23 * r3) * det_j;
        *lambda3 += (x21 * x32 * r1 - x11 * x32 * r2 - x12 * x21 * r3) * det_j;
    }
}

fn se3_from_f64_rt(rotation: &Mat3d, translation: &Vec3d) -> SE3 {
    let rotation = [
        [
            rotation[(0, 0)] as f32,
            rotation[(0, 1)] as f32,
            rotation[(0, 2)] as f32,
        ],
        [
            rotation[(1, 0)] as f32,
            rotation[(1, 1)] as f32,
            rotation[(1, 2)] as f32,
        ],
        [
            rotation[(2, 0)] as f32,
            rotation[(2, 1)] as f32,
            rotation[(2, 2)] as f32,
        ],
    ];
    let translation = [
        translation[0] as f32,
        translation[1] as f32,
        translation[2] as f32,
    ];
    SE3::from_rotation_translation(&rotation, &translation)
}

impl<'a> EpnpEstimator<'a> {
    fn new(image_points: &'a [[f32; 2]], world_points: &'a [[f32; 3]]) -> Option<Self> {
        let n = image_points.len().min(world_points.len());
        if n < 4 || image_points.len() != world_points.len() {
            return None;
        }

        Some(Self {
            image_points,
            world_points,
            cws: [Vec3d::zeros(); 4],
            ccs: [Vec3d::zeros(); 4],
            alphas: vec![Vec4d::zeros(); n],
            pcs: vec![Vec3d::zeros(); n],
        })
    }

    fn compute_pose(&mut self) -> Option<(Mat3d, Vec3d)> {
        self.choose_control_points();
        self.compute_barycentric_coordinates()?;

        let m = self.compute_m();
        let mtm = self.compute_mtm(&m);
        let svd = mtm.svd(true, true);
        let u = svd.u?;
        let ut = u.transpose();

        let l6x10 = self.compute_l6x10(&ut);
        let rho = self.compute_rho();

        let mut candidates: Vec<(f64, Mat3d, Vec3d)> = Vec::with_capacity(3);

        if let Some(mut betas) = self.find_betas_approx1(&l6x10, &rho) {
            self.run_gauss_newton(&l6x10, &rho, &mut betas)?;
            if let Some((rotation, translation, error)) = self.compute_rt(&ut, &betas) {
                candidates.push((error, rotation, translation));
            }
        }

        if let Some(mut betas) = self.find_betas_approx2(&l6x10, &rho) {
            self.run_gauss_newton(&l6x10, &rho, &mut betas)?;
            if let Some((rotation, translation, error)) = self.compute_rt(&ut, &betas) {
                candidates.push((error, rotation, translation));
            }
        }

        if let Some(mut betas) = self.find_betas_approx3(&l6x10, &rho) {
            self.run_gauss_newton(&l6x10, &rho, &mut betas)?;
            if let Some((rotation, translation, error)) = self.compute_rt(&ut, &betas) {
                candidates.push((error, rotation, translation));
            }
        }

        candidates
            .into_iter()
            .filter(|(error, rotation, translation)| {
                error.is_finite()
                    && rotation.iter().all(|v| v.is_finite())
                    && translation.iter().all(|v| v.is_finite())
            })
            .min_by(|left, right| left.0.total_cmp(&right.0))
            .map(|(_, rotation, translation)| (rotation, translation))
    }

    fn choose_control_points(&mut self) {
        let n = self.world_points.len();
        self.cws[0] = Vec3d::zeros();
        for point in self.world_points {
            self.cws[0] += world_point_to_vec3d(point);
        }
        self.cws[0] /= n as f64;

        let mut pw0tpw0 = Mat3d::zeros();
        for point in self.world_points {
            let centered = world_point_to_vec3d(point) - self.cws[0];
            pw0tpw0 += centered * centered.transpose();
        }

        let svd = pw0tpw0.svd(true, true);
        let Some(u) = svd.u else {
            return;
        };

        for i in 1..4 {
            let scale = (svd.singular_values[i - 1] / n as f64).max(0.0).sqrt();
            self.cws[i] = self.cws[0] + scale * u.column(i - 1).into_owned();
        }
    }

    fn compute_barycentric_coordinates(&mut self) -> Option<()> {
        let cc = Mat3d::from_columns(&[
            self.cws[1] - self.cws[0],
            self.cws[2] - self.cws[0],
            self.cws[3] - self.cws[0],
        ]);

        if cc.svd(false, false).rank(1.0e-12) < 3 {
            return None;
        }

        let cc_inv = cc.try_inverse()?;
        for (i, point) in self.world_points.iter().enumerate() {
            let bary = cc_inv * (world_point_to_vec3d(point) - self.cws[0]);
            self.alphas[i] =
                Vec4d::new(1.0 - bary[0] - bary[1] - bary[2], bary[0], bary[1], bary[2]);
        }

        Some(())
    }

    fn compute_m(&self) -> DMatrix<f64> {
        let mut m = DMatrix::<f64>::zeros(3 * self.image_points.len(), 12);
        for (i, image_point) in self.image_points.iter().enumerate() {
            let ray = Vec3d::new(image_point[0] as f64, image_point[1] as f64, 1.0).normalize();
            for j in 0..4 {
                let alpha = self.alphas[i][j];

                m[(3 * i, 3 * j)] = 0.0;
                m[(3 * i, 3 * j + 1)] = -alpha * ray[2];
                m[(3 * i, 3 * j + 2)] = alpha * ray[1];

                m[(3 * i + 1, 3 * j)] = alpha * ray[2];
                m[(3 * i + 1, 3 * j + 1)] = 0.0;
                m[(3 * i + 1, 3 * j + 2)] = -alpha * ray[0];

                m[(3 * i + 2, 3 * j)] = -alpha * ray[1];
                m[(3 * i + 2, 3 * j + 1)] = alpha * ray[0];
                m[(3 * i + 2, 3 * j + 2)] = 0.0;
            }
        }
        m
    }

    fn compute_mtm(&self, m: &DMatrix<f64>) -> Mat12d {
        let mut mtm = Mat12d::zeros();
        for row in 0..12 {
            for col in 0..12 {
                let mut value = 0.0;
                for k in 0..m.nrows() {
                    value += m[(k, row)] * m[(k, col)];
                }
                mtm[(row, col)] = value;
            }
        }
        mtm
    }

    fn compute_l6x10(&self, ut: &Mat12d) -> Mat6x10d {
        let mut dv = [[Vec3d::zeros(); 6]; 4];
        for i in 0..4 {
            let mut a = 0usize;
            let mut b = 1usize;
            for j in 0..6 {
                dv[i][j] = Vec3d::new(
                    ut[(11 - i, 3 * a)] - ut[(11 - i, 3 * b)],
                    ut[(11 - i, 3 * a + 1)] - ut[(11 - i, 3 * b + 1)],
                    ut[(11 - i, 3 * a + 2)] - ut[(11 - i, 3 * b + 2)],
                );

                b += 1;
                if b > 3 {
                    a += 1;
                    b = a + 1;
                }
            }
        }

        let mut l6x10 = Mat6x10d::zeros();
        for i in 0..6 {
            l6x10[(i, 0)] = dv[0][i].dot(&dv[0][i]);
            l6x10[(i, 1)] = 2.0 * dv[0][i].dot(&dv[1][i]);
            l6x10[(i, 2)] = dv[1][i].dot(&dv[1][i]);
            l6x10[(i, 3)] = 2.0 * dv[0][i].dot(&dv[2][i]);
            l6x10[(i, 4)] = 2.0 * dv[1][i].dot(&dv[2][i]);
            l6x10[(i, 5)] = dv[2][i].dot(&dv[2][i]);
            l6x10[(i, 6)] = 2.0 * dv[0][i].dot(&dv[3][i]);
            l6x10[(i, 7)] = 2.0 * dv[1][i].dot(&dv[3][i]);
            l6x10[(i, 8)] = 2.0 * dv[2][i].dot(&dv[3][i]);
            l6x10[(i, 9)] = dv[3][i].dot(&dv[3][i]);
        }
        l6x10
    }

    fn compute_rho(&self) -> SVector<f64, 6> {
        SVector::<f64, 6>::new(
            (self.cws[0] - self.cws[1]).norm_squared(),
            (self.cws[0] - self.cws[2]).norm_squared(),
            (self.cws[0] - self.cws[3]).norm_squared(),
            (self.cws[1] - self.cws[2]).norm_squared(),
            (self.cws[1] - self.cws[3]).norm_squared(),
            (self.cws[2] - self.cws[3]).norm_squared(),
        )
    }

    fn find_betas_approx1(&self, l6x10: &Mat6x10d, rho: &SVector<f64, 6>) -> Option<Vec4d> {
        let mut l6x4 = SMatrix::<f64, 6, 4>::zeros();
        for i in 0..6 {
            l6x4[(i, 0)] = l6x10[(i, 0)];
            l6x4[(i, 1)] = l6x10[(i, 1)];
            l6x4[(i, 2)] = l6x10[(i, 3)];
            l6x4[(i, 3)] = l6x10[(i, 6)];
        }

        let b4 = l6x4.svd(true, true).solve(rho, 1.0e-12).ok()?;
        let beta0_abs = b4[0].abs().sqrt();
        if beta0_abs <= 1.0e-12 {
            return None;
        }

        let sign = if b4[0] < 0.0 { -1.0 } else { 1.0 };
        Some(Vec4d::new(
            beta0_abs,
            sign * b4[1] / beta0_abs,
            sign * b4[2] / beta0_abs,
            sign * b4[3] / beta0_abs,
        ))
    }

    fn find_betas_approx2(&self, l6x10: &Mat6x10d, rho: &SVector<f64, 6>) -> Option<Vec4d> {
        let mut l6x3 = SMatrix::<f64, 6, 3>::zeros();
        for i in 0..6 {
            l6x3[(i, 0)] = l6x10[(i, 0)];
            l6x3[(i, 1)] = l6x10[(i, 1)];
            l6x3[(i, 2)] = l6x10[(i, 2)];
        }

        let b3 = l6x3.svd(true, true).solve(rho, 1.0e-12).ok()?;
        let mut betas = Vec4d::zeros();
        if b3[0] < 0.0 {
            betas[0] = (-b3[0]).sqrt();
            betas[1] = if b3[2] < 0.0 { (-b3[2]).sqrt() } else { 0.0 };
        } else {
            betas[0] = b3[0].sqrt();
            betas[1] = if b3[2] > 0.0 { b3[2].sqrt() } else { 0.0 };
        }

        if b3[1] < 0.0 {
            betas[0] = -betas[0];
        }
        Some(betas)
    }

    fn find_betas_approx3(&self, l6x10: &Mat6x10d, rho: &SVector<f64, 6>) -> Option<Vec4d> {
        let mut l6x5 = SMatrix::<f64, 6, 5>::zeros();
        for row in 0..6 {
            for col in 0..5 {
                l6x5[(row, col)] = l6x10[(row, col)];
            }
        }

        let b5 = l6x5.svd(true, true).solve(rho, 1.0e-12).ok()?;
        let mut betas = Vec4d::zeros();
        if b5[0] < 0.0 {
            betas[0] = (-b5[0]).sqrt();
            betas[1] = if b5[2] < 0.0 { (-b5[2]).sqrt() } else { 0.0 };
        } else {
            betas[0] = b5[0].sqrt();
            betas[1] = if b5[2] > 0.0 { b5[2].sqrt() } else { 0.0 };
        }

        if b5[1] < 0.0 {
            betas[0] = -betas[0];
        }
        if betas[0].abs() <= 1.0e-12 {
            return None;
        }
        betas[2] = b5[3] / betas[0];
        Some(betas)
    }

    fn run_gauss_newton(
        &self,
        l6x10: &Mat6x10d,
        rho: &SVector<f64, 6>,
        betas: &mut Vec4d,
    ) -> Option<()> {
        for _ in 0..5 {
            let mut a = SMatrix::<f64, 6, 4>::zeros();
            let mut b = SVector::<f64, 6>::zeros();

            for i in 0..6 {
                a[(i, 0)] = 2.0 * l6x10[(i, 0)] * betas[0]
                    + l6x10[(i, 1)] * betas[1]
                    + l6x10[(i, 3)] * betas[2]
                    + l6x10[(i, 6)] * betas[3];
                a[(i, 1)] = l6x10[(i, 1)] * betas[0]
                    + 2.0 * l6x10[(i, 2)] * betas[1]
                    + l6x10[(i, 4)] * betas[2]
                    + l6x10[(i, 7)] * betas[3];
                a[(i, 2)] = l6x10[(i, 3)] * betas[0]
                    + l6x10[(i, 4)] * betas[1]
                    + 2.0 * l6x10[(i, 5)] * betas[2]
                    + l6x10[(i, 8)] * betas[3];
                a[(i, 3)] = l6x10[(i, 6)] * betas[0]
                    + l6x10[(i, 7)] * betas[1]
                    + l6x10[(i, 8)] * betas[2]
                    + 2.0 * l6x10[(i, 9)] * betas[3];

                b[i] = rho[i]
                    - (l6x10[(i, 0)] * betas[0] * betas[0]
                        + l6x10[(i, 1)] * betas[0] * betas[1]
                        + l6x10[(i, 2)] * betas[1] * betas[1]
                        + l6x10[(i, 3)] * betas[0] * betas[2]
                        + l6x10[(i, 4)] * betas[1] * betas[2]
                        + l6x10[(i, 5)] * betas[2] * betas[2]
                        + l6x10[(i, 6)] * betas[0] * betas[3]
                        + l6x10[(i, 7)] * betas[1] * betas[3]
                        + l6x10[(i, 8)] * betas[2] * betas[3]
                        + l6x10[(i, 9)] * betas[3] * betas[3]);
            }

            let delta = a.svd(true, true).solve(&b, 1.0e-12).ok()?;
            *betas += delta;
        }

        Some(())
    }

    fn compute_rt(&mut self, ut: &Mat12d, betas: &Vec4d) -> Option<(Mat3d, Vec3d, f64)> {
        self.compute_ccs(ut, betas);
        self.compute_pcs();
        self.solve_for_sign();
        let (rotation, translation) = self.estimate_rt()?;
        let error = self.compute_total_error(&rotation, &translation);
        Some((rotation, translation, error))
    }

    fn compute_ccs(&mut self, ut: &Mat12d, betas: &Vec4d) {
        self.ccs = [Vec3d::zeros(); 4];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..3 {
                    self.ccs[j][k] += betas[i] * ut[(11 - i, 3 * j + k)];
                }
            }
        }
    }

    fn compute_pcs(&mut self) {
        for i in 0..self.world_points.len() {
            self.pcs[i] = self.alphas[i][0] * self.ccs[0]
                + self.alphas[i][1] * self.ccs[1]
                + self.alphas[i][2] * self.ccs[2]
                + self.alphas[i][3] * self.ccs[3];
        }
    }

    fn solve_for_sign(&mut self) {
        if self.pcs.first().map(|p| p[2] < 0.0).unwrap_or(false) {
            for cc in &mut self.ccs {
                *cc = -*cc;
            }
            for pc in &mut self.pcs {
                *pc = -*pc;
            }
        }
    }

    fn estimate_rt(&self) -> Option<(Mat3d, Vec3d)> {
        let n = self.world_points.len() as f64;
        let mut pc0 = Vec3d::zeros();
        let mut pw0 = Vec3d::zeros();
        for i in 0..self.world_points.len() {
            pc0 += self.pcs[i];
            pw0 += world_point_to_vec3d(&self.world_points[i]);
        }
        pc0 /= n;
        pw0 /= n;

        let mut abt = Mat3d::zeros();
        for i in 0..self.world_points.len() {
            let pc = self.pcs[i] - pc0;
            let pw = world_point_to_vec3d(&self.world_points[i]) - pw0;
            abt += pc * pw.transpose();
        }

        let svd = abt.svd(true, true);
        let u = svd.u?;
        let v_t = svd.v_t?;

        let mut correction = Mat3d::identity();
        if (u * v_t).determinant() < 0.0 {
            correction[(2, 2)] = -1.0;
        }
        let rotation = u * correction * v_t;
        let translation = pc0 - rotation * pw0;
        Some((rotation, translation))
    }

    fn compute_total_error(&self, rotation: &Mat3d, translation: &Vec3d) -> f64 {
        let mut error = 0.0;
        for (image_point, world_point) in self.image_points.iter().zip(self.world_points.iter()) {
            let point_cam = rotation * world_point_to_vec3d(world_point) + translation;
            if point_cam[2].abs() <= 1.0e-12 {
                return f64::MAX;
            }
            let projected = Vec3d::new(
                point_cam[0] / point_cam[2],
                point_cam[1] / point_cam[2],
                0.0,
            );
            let observed = Vec3d::new(image_point[0] as f64, image_point[1] as f64, 0.0);
            error += (projected - observed).norm();
        }
        error
    }
}

fn world_point_to_vec3d(point: &[f32; 3]) -> Vec3d {
    Vec3d::new(point[0] as f64, point[1] as f64, point[2] as f64)
}

fn estimate_focal_from_centered_pose(
    pose: SE3,
    centered_pts: &[[f32; 2]],
    object_points: &[[f32; 3]],
) -> Option<f64> {
    let n = centered_pts.len().min(object_points.len());
    if n < 4 {
        return None;
    }

    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    for i in 0..n {
        let p = pose.transform_point(&object_points[i]);
        let z = p[2] as f64;
        if z <= f64::EPSILON {
            continue;
        }
        let nx = p[0] as f64 / z;
        let ny = p[1] as f64 / z;
        numerator += nx * centered_pts[i][0] as f64 + ny * centered_pts[i][1] as f64;
        denominator += nx * nx + ny * ny;
    }

    if denominator <= 1.0e-12 {
        return None;
    }
    let focal = numerator / denominator;
    (focal.is_finite() && focal > 0.0).then_some(focal)
}

fn estimate_focal_pose_p4pf(
    centered_pts: &[[f32; 2]],
    object_points: &[[f32; 3]],
) -> Vec<FocalPoseModel> {
    if centered_pts.len().min(object_points.len()) != 4 {
        return Vec::new();
    }

    let f0 = centered_pts
        .iter()
        .map(|point| (point[0] as f64).hypot(point[1] as f64))
        .sum::<f64>()
        / 4.0;
    if !f0.is_finite() || f0 <= 1.0e-12 {
        return Vec::new();
    }

    let mut points2d = [[0.0f64; 2]; 4];
    let mut points3d = [Vec3d::zeros(); 4];
    for i in 0..4 {
        points2d[i] = [
            centered_pts[i][0] as f64 / f0,
            centered_pts[i][1] as f64 / f0,
        ];
        points3d[i] = world_point_to_vec3d(&object_points[i]);
    }

    let mut m = Mat8x4d::zeros();
    for i in 0..4 {
        let x = points2d[i][0];
        let y = points2d[i][1];
        let point = points3d[i];
        m[(0, i)] = -y * point[0];
        m[(2, i)] = -y * point[1];
        m[(4, i)] = -y * point[2];
        m[(6, i)] = -y;
        m[(1, i)] = x * point[0];
        m[(3, i)] = x * point[1];
        m[(5, i)] = x * point[2];
        m[(7, i)] = x;
    }

    let eigen = SymmetricEigen::new(m * m.transpose());
    let mut eigen_indices = (0..8).collect::<Vec<_>>();
    eigen_indices
        .sort_by(|&left, &right| eigen.eigenvalues[left].total_cmp(&eigen.eigenvalues[right]));
    let mut n = Mat8x4d::zeros();
    for (col, &idx) in eigen_indices.iter().take(4).enumerate() {
        n.set_column(col, &eigen.eigenvectors.column(idx));
    }

    let mut a = Mat4d::zeros();
    let mut b = Mat4d::zeros();
    for i in 0..4 {
        let x = points2d[i][0];
        let y = points2d[i][1];
        let point = points3d[i];
        if x.abs() < y.abs() {
            a[(i, 0)] = y * point[0];
            a[(i, 1)] = y * point[1];
            a[(i, 2)] = y * point[2];
            a[(i, 3)] = y;
            for j in 0..4 {
                b[(i, j)] =
                    point[0] * n[(1, j)] + point[1] * n[(3, j)] + point[2] * n[(5, j)] + n[(7, j)];
            }
        } else {
            a[(i, 0)] = x * point[0];
            a[(i, 1)] = x * point[1];
            a[(i, 2)] = x * point[2];
            a[(i, 3)] = x;
            for j in 0..4 {
                b[(i, j)] =
                    point[0] * n[(0, j)] + point[1] * n[(2, j)] + point[2] * n[(4, j)] + n[(6, j)];
            }
        }
    }

    let Some(a_inv) = a.try_inverse() else {
        return Vec::new();
    };
    b = a_inv * b;

    let mut d = [0.0f64; 48];
    for col in 0..4 {
        for row in 0..8 {
            d[col * 8 + row] = n[(row, col)];
        }
    }
    for col in 0..4 {
        for row in 0..4 {
            d[32 + col * 4 + row] = b[(row, col)];
        }
    }

    let mut coeffs = Mat3x10d::zeros();
    coeffs.row_mut(0).copy_from(
        &SVector::<f64, 10>::from_row_slice(&[
            d[0] * d[1] + d[2] * d[3] + d[4] * d[5],
            d[0] * d[9] + d[1] * d[8] + d[2] * d[11] + d[3] * d[10] + d[4] * d[13] + d[5] * d[12],
            d[0] * d[17] + d[1] * d[16] + d[2] * d[19] + d[3] * d[18] + d[4] * d[21] + d[5] * d[20],
            d[8] * d[9] + d[10] * d[11] + d[12] * d[13],
            d[8] * d[17]
                + d[9] * d[16]
                + d[10] * d[19]
                + d[11] * d[18]
                + d[12] * d[21]
                + d[13] * d[20],
            d[16] * d[17] + d[18] * d[19] + d[20] * d[21],
            d[0] * d[25] + d[1] * d[24] + d[2] * d[27] + d[3] * d[26] + d[4] * d[29] + d[5] * d[28],
            d[8] * d[25]
                + d[9] * d[24]
                + d[10] * d[27]
                + d[11] * d[26]
                + d[12] * d[29]
                + d[13] * d[28],
            d[16] * d[25]
                + d[17] * d[24]
                + d[18] * d[27]
                + d[19] * d[26]
                + d[20] * d[29]
                + d[21] * d[28],
            d[24] * d[25] + d[26] * d[27] + d[28] * d[29],
        ])
        .transpose(),
    );
    coeffs.row_mut(1).copy_from(
        &SVector::<f64, 10>::from_row_slice(&[
            d[0] * d[32] + d[2] * d[33] + d[4] * d[34],
            d[0] * d[36]
                + d[2] * d[37]
                + d[8] * d[32]
                + d[4] * d[38]
                + d[10] * d[33]
                + d[12] * d[34],
            d[0] * d[40]
                + d[2] * d[41]
                + d[4] * d[42]
                + d[16] * d[32]
                + d[18] * d[33]
                + d[20] * d[34],
            d[8] * d[36] + d[10] * d[37] + d[12] * d[38],
            d[8] * d[40]
                + d[10] * d[41]
                + d[16] * d[36]
                + d[12] * d[42]
                + d[18] * d[37]
                + d[20] * d[38],
            d[16] * d[40] + d[18] * d[41] + d[20] * d[42],
            d[0] * d[44]
                + d[2] * d[45]
                + d[4] * d[46]
                + d[24] * d[32]
                + d[26] * d[33]
                + d[28] * d[34],
            d[8] * d[44]
                + d[10] * d[45]
                + d[12] * d[46]
                + d[24] * d[36]
                + d[26] * d[37]
                + d[28] * d[38],
            d[16] * d[44]
                + d[18] * d[45]
                + d[24] * d[40]
                + d[20] * d[46]
                + d[26] * d[41]
                + d[28] * d[42],
            d[24] * d[44] + d[26] * d[45] + d[28] * d[46],
        ])
        .transpose(),
    );
    coeffs.row_mut(2).copy_from(
        &SVector::<f64, 10>::from_row_slice(&[
            d[1] * d[32] + d[3] * d[33] + d[5] * d[34],
            d[1] * d[36]
                + d[3] * d[37]
                + d[9] * d[32]
                + d[5] * d[38]
                + d[11] * d[33]
                + d[13] * d[34],
            d[1] * d[40]
                + d[3] * d[41]
                + d[5] * d[42]
                + d[17] * d[32]
                + d[19] * d[33]
                + d[21] * d[34],
            d[9] * d[36] + d[11] * d[37] + d[13] * d[38],
            d[9] * d[40]
                + d[11] * d[41]
                + d[17] * d[36]
                + d[13] * d[42]
                + d[19] * d[37]
                + d[21] * d[38],
            d[17] * d[40] + d[19] * d[41] + d[21] * d[42],
            d[1] * d[44]
                + d[3] * d[45]
                + d[5] * d[46]
                + d[25] * d[32]
                + d[27] * d[33]
                + d[29] * d[34],
            d[9] * d[44]
                + d[11] * d[45]
                + d[13] * d[46]
                + d[25] * d[36]
                + d[27] * d[37]
                + d[29] * d[38],
            d[17] * d[44]
                + d[19] * d[45]
                + d[25] * d[40]
                + d[21] * d[46]
                + d[27] * d[41]
                + d[29] * d[42],
            d[25] * d[44] + d[27] * d[45] + d[29] * d[46],
        ])
        .transpose(),
    );

    let solutions = solve_re3q3(coeffs);
    let mut candidates = Vec::new();
    for alpha_xyz in solutions {
        let alpha = Vec4d::new(alpha_xyz[0], alpha_xyz[1], alpha_xyz[2], 1.0);
        let p12 = n * alpha;
        let mut projection = Mat3x4d::zeros();
        for col in 0..4 {
            projection[(0, col)] = p12[col * 2];
            projection[(1, col)] = p12[col * 2 + 1];
            projection[(2, col)] = b[(col, 0)] * alpha[0]
                + b[(col, 1)] * alpha[1]
                + b[(col, 2)] * alpha[2]
                + b[(col, 3)] * alpha[3];
        }

        if projection.fixed_view::<3, 3>(0, 0).determinant() < 0.0 {
            projection = -projection;
        }

        let row2_norm = projection.fixed_view::<1, 3>(2, 0).norm();
        if !row2_norm.is_finite() || row2_norm <= 1.0e-12 {
            continue;
        }
        projection /= row2_norm;

        let fx = projection.fixed_view::<1, 3>(0, 0).norm() * f0;
        let fy = projection.fixed_view::<1, 3>(1, 0).norm() * f0;
        if !fx.is_finite() || !fy.is_finite() || fx <= 0.0 || fy <= 0.0 {
            continue;
        }

        for col in 0..4 {
            projection[(0, col)] /= fx / f0;
            projection[(1, col)] /= fy / f0;
        }
        let rotation = projection.fixed_view::<3, 3>(0, 0).into_owned();
        let translation = Vec3d::new(projection[(0, 3)], projection[(1, 3)], projection[(2, 3)]);

        let focal_ratio = fx / fy;
        let focal_ratio_error = (focal_ratio - 1.0)
            .abs()
            .max((1.0 / focal_ratio - 1.0).abs());
        if focal_ratio_error >= 1.0 {
            continue;
        }
        let focal = (fx + fy) * 0.5;
        if !has_positive_depth(&rotation, &translation, &points3d) {
            continue;
        }

        let pose = se3_from_f64_rt(&rotation, &translation);
        let model = FocalPoseModel { pose, focal };
        if focal_pose_model_cost(model, centered_pts, object_points).is_some() {
            candidates.push(model);
        }
    }

    deduplicate_focal_models(candidates)
}

fn has_positive_depth(rotation: &Mat3d, translation: &Vec3d, points: &[Vec3d; 4]) -> bool {
    points
        .iter()
        .all(|point| (rotation.row(2) * point)[0] + translation[2] >= 0.0)
}

fn solve_re3q3(coeffs: Mat3x10d) -> Vec<Vec3d> {
    let ax = Mat3d::from_columns(&[
        coeffs.column(3).into_owned(),
        coeffs.column(5).into_owned(),
        coeffs.column(4).into_owned(),
    ]);
    let ay = Mat3d::from_columns(&[
        coeffs.column(0).into_owned(),
        coeffs.column(5).into_owned(),
        coeffs.column(2).into_owned(),
    ]);
    let az = Mat3d::from_columns(&[
        coeffs.column(3).into_owned(),
        coeffs.column(0).into_owned(),
        coeffs.column(1).into_owned(),
    ]);

    let detx = ax.determinant().abs();
    let dety = ay.determinant().abs();
    let detz = az.determinant().abs();
    let mut elim_var = 0usize;
    let mut det = detx;
    if det < dety {
        det = dety;
        elim_var = 1;
    }
    if det < detz {
        elim_var = 2;
    }

    let mut p = Mat3x7d::zeros();
    let solved = if elim_var == 0 {
        for col in 0..7 {
            let src = [0usize, 1, 2, 6, 7, 8, 9][col];
            p.set_column(col, &coeffs.column(src));
        }
        ax.try_inverse().map(|inv| -inv * p)
    } else if elim_var == 1 {
        for col in 0..7 {
            let src = [3usize, 1, 4, 7, 6, 8, 9][col];
            p.set_column(col, &coeffs.column(src));
        }
        ay.try_inverse().map(|inv| -inv * p)
    } else {
        for col in 0..7 {
            let src = [5usize, 4, 2, 8, 7, 6, 9][col];
            p.set_column(col, &coeffs.column(src));
        }
        az.try_inverse().map(|inv| -inv * p)
    };
    let Some(p) = solved else {
        return Vec::new();
    };

    let a11 = p[(0, 1)] * p[(2, 1)] + p[(0, 2)] * p[(1, 1)]
        - p[(2, 1)] * p[(0, 1)]
        - p[(2, 2)] * p[(2, 1)]
        - p[(2, 0)];
    let a12 = p[(0, 1)] * p[(2, 4)]
        + p[(0, 4)] * p[(2, 1)]
        + p[(0, 2)] * p[(1, 4)]
        + p[(0, 5)] * p[(1, 1)]
        - p[(2, 1)] * p[(0, 4)]
        - p[(2, 4)] * p[(0, 1)]
        - p[(2, 2)] * p[(2, 4)]
        - p[(2, 5)] * p[(2, 1)]
        - p[(2, 3)];
    let a13 = p[(0, 4)] * p[(2, 4)] + p[(0, 5)] * p[(1, 4)]
        - p[(2, 4)] * p[(0, 4)]
        - p[(2, 5)] * p[(2, 4)]
        - p[(2, 6)];
    let a14 = p[(0, 1)] * p[(2, 2)] + p[(0, 2)] * p[(1, 2)]
        - p[(2, 1)] * p[(0, 2)]
        - p[(2, 2)] * p[(2, 2)]
        + p[(0, 0)];
    let a15 = p[(0, 1)] * p[(2, 5)]
        + p[(0, 4)] * p[(2, 2)]
        + p[(0, 2)] * p[(1, 5)]
        + p[(0, 5)] * p[(1, 2)]
        - p[(2, 1)] * p[(0, 5)]
        - p[(2, 4)] * p[(0, 2)]
        - p[(2, 2)] * p[(2, 5)]
        - p[(2, 5)] * p[(2, 2)]
        + p[(0, 3)];
    let a16 = p[(0, 4)] * p[(2, 5)] + p[(0, 5)] * p[(1, 5)]
        - p[(2, 4)] * p[(0, 5)]
        - p[(2, 5)] * p[(2, 5)]
        + p[(0, 6)];
    let a17 = p[(0, 1)] * p[(2, 0)] + p[(0, 2)] * p[(1, 0)]
        - p[(2, 1)] * p[(0, 0)]
        - p[(2, 2)] * p[(2, 0)];
    let a18 = p[(0, 1)] * p[(2, 3)]
        + p[(0, 4)] * p[(2, 0)]
        + p[(0, 2)] * p[(1, 3)]
        + p[(0, 5)] * p[(1, 0)]
        - p[(2, 1)] * p[(0, 3)]
        - p[(2, 4)] * p[(0, 0)]
        - p[(2, 2)] * p[(2, 3)]
        - p[(2, 5)] * p[(2, 0)];
    let a19 = p[(0, 1)] * p[(2, 6)]
        + p[(0, 4)] * p[(2, 3)]
        + p[(0, 2)] * p[(1, 6)]
        + p[(0, 5)] * p[(1, 3)]
        - p[(2, 1)] * p[(0, 6)]
        - p[(2, 4)] * p[(0, 3)]
        - p[(2, 2)] * p[(2, 6)]
        - p[(2, 5)] * p[(2, 3)];
    let a110 = p[(0, 4)] * p[(2, 6)] + p[(0, 5)] * p[(1, 6)]
        - p[(2, 4)] * p[(0, 6)]
        - p[(2, 5)] * p[(2, 6)];

    let a21 = p[(2, 1)] * p[(2, 1)] + p[(2, 2)] * p[(1, 1)]
        - p[(1, 1)] * p[(0, 1)]
        - p[(1, 2)] * p[(2, 1)]
        - p[(1, 0)];
    let a22 = p[(2, 1)] * p[(2, 4)]
        + p[(2, 4)] * p[(2, 1)]
        + p[(2, 2)] * p[(1, 4)]
        + p[(2, 5)] * p[(1, 1)]
        - p[(1, 1)] * p[(0, 4)]
        - p[(1, 4)] * p[(0, 1)]
        - p[(1, 2)] * p[(2, 4)]
        - p[(1, 5)] * p[(2, 1)]
        - p[(1, 3)];
    let a23 = p[(2, 4)] * p[(2, 4)] + p[(2, 5)] * p[(1, 4)]
        - p[(1, 4)] * p[(0, 4)]
        - p[(1, 5)] * p[(2, 4)]
        - p[(1, 6)];
    let a24 = p[(2, 1)] * p[(2, 2)] + p[(2, 2)] * p[(1, 2)]
        - p[(1, 1)] * p[(0, 2)]
        - p[(1, 2)] * p[(2, 2)]
        + p[(2, 0)];
    let a25 = p[(2, 1)] * p[(2, 5)]
        + p[(2, 4)] * p[(2, 2)]
        + p[(2, 2)] * p[(1, 5)]
        + p[(2, 5)] * p[(1, 2)]
        - p[(1, 1)] * p[(0, 5)]
        - p[(1, 4)] * p[(0, 2)]
        - p[(1, 2)] * p[(2, 5)]
        - p[(1, 5)] * p[(2, 2)]
        + p[(2, 3)];
    let a26 = p[(2, 4)] * p[(2, 5)] + p[(2, 5)] * p[(1, 5)]
        - p[(1, 4)] * p[(0, 5)]
        - p[(1, 5)] * p[(2, 5)]
        + p[(2, 6)];
    let a27 = p[(2, 1)] * p[(2, 0)] + p[(2, 2)] * p[(1, 0)]
        - p[(1, 1)] * p[(0, 0)]
        - p[(1, 2)] * p[(2, 0)];
    let a28 = p[(2, 1)] * p[(2, 3)]
        + p[(2, 4)] * p[(2, 0)]
        + p[(2, 2)] * p[(1, 3)]
        + p[(2, 5)] * p[(1, 0)]
        - p[(1, 1)] * p[(0, 3)]
        - p[(1, 4)] * p[(0, 0)]
        - p[(1, 2)] * p[(2, 3)]
        - p[(1, 5)] * p[(2, 0)];
    let a29 = p[(2, 1)] * p[(2, 6)]
        + p[(2, 4)] * p[(2, 3)]
        + p[(2, 2)] * p[(1, 6)]
        + p[(2, 5)] * p[(1, 3)]
        - p[(1, 1)] * p[(0, 6)]
        - p[(1, 4)] * p[(0, 3)]
        - p[(1, 2)] * p[(2, 6)]
        - p[(1, 5)] * p[(2, 3)];
    let a210 = p[(2, 4)] * p[(2, 6)] + p[(2, 5)] * p[(1, 6)]
        - p[(1, 4)] * p[(0, 6)]
        - p[(1, 5)] * p[(2, 6)];

    let t2 = p[(2, 1)] * p[(2, 1)];
    let t3 = p[(2, 2)] * p[(2, 2)];
    let t4 = p[(0, 1)] * p[(1, 4)];
    let t5 = p[(0, 4)] * p[(1, 1)];
    let t6 = t4 + t5;
    let t7 = p[(0, 2)] * p[(1, 5)];
    let t8 = p[(0, 5)] * p[(1, 2)];
    let t9 = t7 + t8;
    let t10 = p[(0, 1)] * p[(1, 5)];
    let t11 = p[(0, 4)] * p[(1, 2)];
    let t12 = t10 + t11;
    let t13 = p[(0, 2)] * p[(1, 4)];
    let t14 = p[(0, 5)] * p[(1, 1)];
    let t15 = t13 + t14;
    let t16 = p[(2, 1)] * p[(2, 5)];
    let t17 = p[(2, 2)] * p[(2, 4)];
    let t18 = t16 + t17;
    let t19 = p[(2, 4)] * p[(2, 4)];
    let t20 = p[(2, 5)] * p[(2, 5)];

    let a31 = p[(0, 0)] * p[(1, 1)] + p[(0, 1)] * p[(1, 0)]
        - p[(2, 0)] * p[(2, 1)] * 2.0
        - p[(0, 1)] * t2
        - p[(1, 1)] * t3
        - p[(2, 2)] * t2 * 2.0
        + (p[(0, 1)] * p[(0, 1)]) * p[(1, 1)]
        + p[(0, 2)] * p[(1, 1)] * p[(1, 2)]
        + p[(0, 1)] * p[(1, 2)] * p[(2, 1)]
        + p[(0, 2)] * p[(1, 1)] * p[(2, 1)];
    let a32 = p[(0, 0)] * p[(1, 4)]
        + p[(0, 1)] * p[(1, 3)]
        + p[(0, 3)] * p[(1, 1)]
        + p[(0, 4)] * p[(1, 0)]
        - p[(2, 0)] * p[(2, 4)] * 2.0
        - p[(2, 1)] * p[(2, 3)] * 2.0
        - p[(0, 4)] * t2
        + p[(0, 1)] * t6
        - p[(1, 4)] * t3
        + p[(1, 1)] * t9
        + p[(2, 1)] * t12
        + p[(2, 1)] * t15
        - p[(2, 1)] * t18 * 2.0
        + p[(0, 1)] * p[(0, 4)] * p[(1, 1)]
        + p[(0, 2)] * p[(1, 2)] * p[(1, 4)]
        + p[(0, 1)] * p[(1, 2)] * p[(2, 4)]
        + p[(0, 2)] * p[(1, 1)] * p[(2, 4)]
        - p[(0, 1)] * p[(2, 1)] * p[(2, 4)] * 2.0
        - p[(1, 1)] * p[(2, 2)] * p[(2, 5)] * 2.0
        - p[(2, 1)] * p[(2, 2)] * p[(2, 4)] * 2.0;
    let a33 = p[(0, 1)] * p[(1, 6)]
        + p[(0, 3)] * p[(1, 4)]
        + p[(0, 4)] * p[(1, 3)]
        + p[(0, 6)] * p[(1, 1)]
        - p[(2, 1)] * p[(2, 6)] * 2.0
        - p[(2, 3)] * p[(2, 4)] * 2.0
        + p[(0, 4)] * t6
        - p[(0, 1)] * t19
        + p[(1, 4)] * t9
        - p[(1, 1)] * t20
        + p[(2, 4)] * t12
        + p[(2, 4)] * t15
        - p[(2, 4)] * t18 * 2.0
        + p[(0, 1)] * p[(0, 4)] * p[(1, 4)]
        + p[(0, 5)] * p[(1, 1)] * p[(1, 5)]
        + p[(0, 4)] * p[(1, 5)] * p[(2, 1)]
        + p[(0, 5)] * p[(1, 4)] * p[(2, 1)]
        - p[(0, 4)] * p[(2, 1)] * p[(2, 4)] * 2.0
        - p[(1, 4)] * p[(2, 2)] * p[(2, 5)] * 2.0
        - p[(2, 1)] * p[(2, 4)] * p[(2, 5)] * 2.0;
    let a34 = p[(0, 4)] * p[(1, 6)] + p[(0, 6)] * p[(1, 4)]
        - p[(2, 4)] * p[(2, 6)] * 2.0
        - p[(0, 4)] * t19
        - p[(1, 4)] * t20
        - p[(2, 5)] * t19 * 2.0
        + (p[(0, 4)] * p[(0, 4)]) * p[(1, 4)]
        + p[(0, 5)] * p[(1, 4)] * p[(1, 5)]
        + p[(0, 4)] * p[(1, 5)] * p[(2, 4)]
        + p[(0, 5)] * p[(1, 4)] * p[(2, 4)];
    let a35 = p[(0, 0)] * p[(1, 2)] + p[(0, 2)] * p[(1, 0)]
        - p[(2, 0)] * p[(2, 2)] * 2.0
        - p[(0, 2)] * t2
        - p[(1, 2)] * t3
        - p[(2, 1)] * t3 * 2.0
        + p[(0, 2)] * (p[(1, 2)] * p[(1, 2)])
        + p[(0, 1)] * p[(0, 2)] * p[(1, 1)]
        + p[(0, 1)] * p[(1, 2)] * p[(2, 2)]
        + p[(0, 2)] * p[(1, 1)] * p[(2, 2)];
    let a36 = p[(0, 0)] * p[(1, 5)]
        + p[(0, 2)] * p[(1, 3)]
        + p[(0, 3)] * p[(1, 2)]
        + p[(0, 5)] * p[(1, 0)]
        - p[(2, 0)] * p[(2, 5)] * 2.0
        - p[(2, 2)] * p[(2, 3)] * 2.0
        - p[(0, 5)] * t2
        + p[(0, 2)] * t6
        - p[(1, 5)] * t3
        + p[(1, 2)] * t9
        + p[(2, 2)] * t12
        + p[(2, 2)] * t15
        - p[(2, 2)] * t18 * 2.0
        + p[(0, 1)] * p[(0, 5)] * p[(1, 1)]
        + p[(0, 2)] * p[(1, 2)] * p[(1, 5)]
        + p[(0, 1)] * p[(1, 2)] * p[(2, 5)]
        + p[(0, 2)] * p[(1, 1)] * p[(2, 5)]
        - p[(0, 2)] * p[(2, 1)] * p[(2, 4)] * 2.0
        - p[(1, 2)] * p[(2, 2)] * p[(2, 5)] * 2.0
        - p[(2, 1)] * p[(2, 2)] * p[(2, 5)] * 2.0;
    let a37 = p[(0, 2)] * p[(1, 6)]
        + p[(0, 3)] * p[(1, 5)]
        + p[(0, 5)] * p[(1, 3)]
        + p[(0, 6)] * p[(1, 2)]
        - p[(2, 2)] * p[(2, 6)] * 2.0
        - p[(2, 3)] * p[(2, 5)] * 2.0
        + p[(0, 5)] * t6
        - p[(0, 2)] * t19
        + p[(1, 5)] * t9
        - p[(1, 2)] * t20
        + p[(2, 5)] * t12
        + p[(2, 5)] * t15
        - p[(2, 5)] * t18 * 2.0
        + p[(0, 2)] * p[(0, 4)] * p[(1, 4)]
        + p[(0, 5)] * p[(1, 2)] * p[(1, 5)]
        + p[(0, 4)] * p[(1, 5)] * p[(2, 2)]
        + p[(0, 5)] * p[(1, 4)] * p[(2, 2)]
        - p[(0, 5)] * p[(2, 1)] * p[(2, 4)] * 2.0
        - p[(1, 5)] * p[(2, 2)] * p[(2, 5)] * 2.0
        - p[(2, 2)] * p[(2, 4)] * p[(2, 5)] * 2.0;
    let a38 = p[(0, 5)] * p[(1, 6)] + p[(0, 6)] * p[(1, 5)]
        - p[(2, 5)] * p[(2, 6)] * 2.0
        - p[(0, 5)] * t19
        - p[(1, 5)] * t20
        - p[(2, 4)] * t20 * 2.0
        + p[(0, 5)] * (p[(1, 5)] * p[(1, 5)])
        + p[(0, 4)] * p[(0, 5)] * p[(1, 4)]
        + p[(0, 4)] * p[(1, 5)] * p[(2, 5)]
        + p[(0, 5)] * p[(1, 4)] * p[(2, 5)];
    let a39 = p[(0, 0)] * p[(1, 0)] - p[(0, 0)] * t2 - p[(1, 0)] * t3 - p[(2, 0)] * p[(2, 0)]
        + p[(0, 0)] * p[(0, 1)] * p[(1, 1)]
        + p[(0, 2)] * p[(1, 0)] * p[(1, 2)]
        + p[(0, 1)] * p[(1, 2)] * p[(2, 0)]
        + p[(0, 2)] * p[(1, 1)] * p[(2, 0)]
        - p[(2, 0)] * p[(2, 1)] * p[(2, 2)] * 2.0;
    let a310 = p[(0, 0)] * p[(1, 3)] + p[(0, 3)] * p[(1, 0)]
        - p[(2, 0)] * p[(2, 3)] * 2.0
        - p[(0, 3)] * t2
        + p[(0, 0)] * t6
        - p[(1, 3)] * t3
        + p[(1, 0)] * t9
        + p[(2, 0)] * t12
        + p[(2, 0)] * t15
        - p[(2, 0)] * t18 * 2.0
        + p[(0, 1)] * p[(0, 3)] * p[(1, 1)]
        + p[(0, 2)] * p[(1, 2)] * p[(1, 3)]
        + p[(0, 1)] * p[(1, 2)] * p[(2, 3)]
        + p[(0, 2)] * p[(1, 1)] * p[(2, 3)]
        - p[(0, 0)] * p[(2, 1)] * p[(2, 4)] * 2.0
        - p[(1, 0)] * p[(2, 2)] * p[(2, 5)] * 2.0
        - p[(2, 1)] * p[(2, 2)] * p[(2, 3)] * 2.0;
    let a311 = p[(0, 0)] * p[(1, 6)] + p[(0, 3)] * p[(1, 3)] + p[(0, 6)] * p[(1, 0)]
        - p[(2, 0)] * p[(2, 6)] * 2.0
        - p[(0, 6)] * t2
        + p[(0, 3)] * t6
        - p[(0, 0)] * t19
        - p[(1, 6)] * t3
        + p[(1, 3)] * t9
        - p[(1, 0)] * t20
        + p[(2, 3)] * t12
        + p[(2, 3)] * t15
        - p[(2, 3)] * t18 * 2.0
        - p[(2, 3)] * p[(2, 3)]
        + p[(0, 0)] * p[(0, 4)] * p[(1, 4)]
        + p[(0, 1)] * p[(0, 6)] * p[(1, 1)]
        + p[(0, 2)] * p[(1, 2)] * p[(1, 6)]
        + p[(0, 5)] * p[(1, 0)] * p[(1, 5)]
        + p[(0, 1)] * p[(1, 2)] * p[(2, 6)]
        + p[(0, 2)] * p[(1, 1)] * p[(2, 6)]
        + p[(0, 4)] * p[(1, 5)] * p[(2, 0)]
        + p[(0, 5)] * p[(1, 4)] * p[(2, 0)]
        - p[(0, 3)] * p[(2, 1)] * p[(2, 4)] * 2.0
        - p[(1, 3)] * p[(2, 2)] * p[(2, 5)] * 2.0
        - p[(2, 0)] * p[(2, 4)] * p[(2, 5)] * 2.0
        - p[(2, 1)] * p[(2, 2)] * p[(2, 6)] * 2.0;
    let a312 = p[(0, 3)] * p[(1, 6)] + p[(0, 6)] * p[(1, 3)] - p[(2, 3)] * p[(2, 6)] * 2.0
        + p[(0, 6)] * t6
        - p[(0, 3)] * t19
        + p[(1, 6)] * t9
        - p[(1, 3)] * t20
        + p[(2, 6)] * t12
        + p[(2, 6)] * t15
        - p[(2, 6)] * t18 * 2.0
        + p[(0, 3)] * p[(0, 4)] * p[(1, 4)]
        + p[(0, 5)] * p[(1, 3)] * p[(1, 5)]
        + p[(0, 4)] * p[(1, 5)] * p[(2, 3)]
        + p[(0, 5)] * p[(1, 4)] * p[(2, 3)]
        - p[(0, 6)] * p[(2, 1)] * p[(2, 4)] * 2.0
        - p[(1, 6)] * p[(2, 2)] * p[(2, 5)] * 2.0
        - p[(2, 3)] * p[(2, 4)] * p[(2, 5)] * 2.0;
    let a313 = p[(0, 6)] * p[(1, 6)] - p[(0, 6)] * t19 - p[(1, 6)] * t20 - p[(2, 6)] * p[(2, 6)]
        + p[(0, 4)] * p[(0, 6)] * p[(1, 4)]
        + p[(0, 5)] * p[(1, 5)] * p[(1, 6)]
        + p[(0, 4)] * p[(1, 5)] * p[(2, 6)]
        + p[(0, 5)] * p[(1, 4)] * p[(2, 6)]
        - p[(2, 4)] * p[(2, 5)] * p[(2, 6)] * 2.0;

    let mut c = [0.0f64; 9];
    c[8] = a14 * a27 * a31 - a17 * a24 * a31 - a11 * a27 * a35 + a17 * a21 * a35 + a11 * a24 * a39
        - a14 * a21 * a39;
    c[7] = a14 * a27 * a32 + a14 * a28 * a31 + a15 * a27 * a31
        - a17 * a24 * a32
        - a17 * a25 * a31
        - a18 * a24 * a31
        - a11 * a27 * a36
        - a11 * a28 * a35
        - a12 * a27 * a35
        + a17 * a21 * a36
        + a17 * a22 * a35
        + a18 * a21 * a35
        + a11 * a25 * a39
        + a12 * a24 * a39
        - a14 * a22 * a39
        - a15 * a21 * a39
        + a11 * a24 * a310
        - a14 * a21 * a310;
    c[6] = a14 * a27 * a33
        + a14 * a28 * a32
        + a14 * a29 * a31
        + a15 * a27 * a32
        + a15 * a28 * a31
        + a16 * a27 * a31
        - a17 * a24 * a33
        - a17 * a25 * a32
        - a17 * a26 * a31
        - a18 * a24 * a32
        - a18 * a25 * a31
        - a19 * a24 * a31
        - a11 * a27 * a37
        - a11 * a28 * a36
        - a11 * a29 * a35
        - a12 * a27 * a36
        - a12 * a28 * a35
        - a13 * a27 * a35
        + a17 * a21 * a37
        + a17 * a22 * a36
        + a17 * a23 * a35
        + a18 * a21 * a36
        + a18 * a22 * a35
        + a19 * a21 * a35
        + a11 * a26 * a39
        + a12 * a25 * a39
        + a13 * a24 * a39
        - a14 * a23 * a39
        - a15 * a22 * a39
        - a16 * a21 * a39
        + a11 * a24 * a311
        + a11 * a25 * a310
        + a12 * a24 * a310
        - a14 * a21 * a311
        - a14 * a22 * a310
        - a15 * a21 * a310;
    c[5] = a14 * a27 * a34
        + a14 * a28 * a33
        + a14 * a29 * a32
        + a15 * a27 * a33
        + a15 * a28 * a32
        + a15 * a29 * a31
        + a16 * a27 * a32
        + a16 * a28 * a31
        - a17 * a24 * a34
        - a17 * a25 * a33
        - a17 * a26 * a32
        - a18 * a24 * a33
        - a18 * a25 * a32
        - a18 * a26 * a31
        - a19 * a24 * a32
        - a19 * a25 * a31
        - a11 * a27 * a38
        - a11 * a28 * a37
        - a11 * a29 * a36
        - a12 * a27 * a37
        - a12 * a28 * a36
        - a12 * a29 * a35
        - a13 * a27 * a36
        - a13 * a28 * a35
        + a17 * a21 * a38
        + a17 * a22 * a37
        + a17 * a23 * a36
        + a18 * a21 * a37
        + a18 * a22 * a36
        + a18 * a23 * a35
        + a19 * a21 * a36
        + a19 * a22 * a35
        + a12 * a26 * a39
        + a13 * a25 * a39
        - a15 * a23 * a39
        - a16 * a22 * a39
        - a24 * a31 * a110
        + a21 * a35 * a110
        + a14 * a31 * a210
        - a11 * a35 * a210
        + a11 * a24 * a312
        + a11 * a25 * a311
        + a11 * a26 * a310
        + a12 * a24 * a311
        + a12 * a25 * a310
        + a13 * a24 * a310
        - a14 * a21 * a312
        - a14 * a22 * a311
        - a14 * a23 * a310
        - a15 * a21 * a311
        - a15 * a22 * a310
        - a16 * a21 * a310;
    c[4] = a14 * a28 * a34
        + a14 * a29 * a33
        + a15 * a27 * a34
        + a15 * a28 * a33
        + a15 * a29 * a32
        + a16 * a27 * a33
        + a16 * a28 * a32
        + a16 * a29 * a31
        - a17 * a25 * a34
        - a17 * a26 * a33
        - a18 * a24 * a34
        - a18 * a25 * a33
        - a18 * a26 * a32
        - a19 * a24 * a33
        - a19 * a25 * a32
        - a19 * a26 * a31
        - a11 * a28 * a38
        - a11 * a29 * a37
        - a12 * a27 * a38
        - a12 * a28 * a37
        - a12 * a29 * a36
        - a13 * a27 * a37
        - a13 * a28 * a36
        - a13 * a29 * a35
        + a17 * a22 * a38
        + a17 * a23 * a37
        + a18 * a21 * a38
        + a18 * a22 * a37
        + a18 * a23 * a36
        + a19 * a21 * a37
        + a19 * a22 * a36
        + a19 * a23 * a35
        + a13 * a26 * a39
        - a16 * a23 * a39
        - a24 * a32 * a110
        - a25 * a31 * a110
        + a21 * a36 * a110
        + a22 * a35 * a110
        + a14 * a32 * a210
        + a15 * a31 * a210
        - a11 * a36 * a210
        - a12 * a35 * a210
        + a11 * a24 * a313
        + a11 * a25 * a312
        + a11 * a26 * a311
        + a12 * a24 * a312
        + a12 * a25 * a311
        + a12 * a26 * a310
        + a13 * a24 * a311
        + a13 * a25 * a310
        - a14 * a21 * a313
        - a14 * a22 * a312
        - a14 * a23 * a311
        - a15 * a21 * a312
        - a15 * a22 * a311
        - a15 * a23 * a310
        - a16 * a21 * a311
        - a16 * a22 * a310;
    c[3] = a14 * a29 * a34
        + a15 * a28 * a34
        + a15 * a29 * a33
        + a16 * a27 * a34
        + a16 * a28 * a33
        + a16 * a29 * a32
        - a17 * a26 * a34
        - a18 * a25 * a34
        - a18 * a26 * a33
        - a19 * a24 * a34
        - a19 * a25 * a33
        - a19 * a26 * a32
        - a11 * a29 * a38
        - a12 * a28 * a38
        - a12 * a29 * a37
        - a13 * a27 * a38
        - a13 * a28 * a37
        - a13 * a29 * a36
        + a17 * a23 * a38
        + a18 * a22 * a38
        + a18 * a23 * a37
        + a19 * a21 * a38
        + a19 * a22 * a37
        + a19 * a23 * a36
        - a24 * a33 * a110
        - a25 * a32 * a110
        - a26 * a31 * a110
        + a21 * a37 * a110
        + a22 * a36 * a110
        + a23 * a35 * a110
        + a14 * a33 * a210
        + a15 * a32 * a210
        + a16 * a31 * a210
        - a11 * a37 * a210
        - a12 * a36 * a210
        - a13 * a35 * a210
        + a11 * a25 * a313
        + a11 * a26 * a312
        + a12 * a24 * a313
        + a12 * a25 * a312
        + a12 * a26 * a311
        + a13 * a24 * a312
        + a13 * a25 * a311
        + a13 * a26 * a310
        - a14 * a22 * a313
        - a14 * a23 * a312
        - a15 * a21 * a313
        - a15 * a22 * a312
        - a15 * a23 * a311
        - a16 * a21 * a312
        - a16 * a22 * a311
        - a16 * a23 * a310;
    c[2] = a15 * a29 * a34 + a16 * a28 * a34 + a16 * a29 * a33
        - a18 * a26 * a34
        - a19 * a25 * a34
        - a19 * a26 * a33
        - a12 * a29 * a38
        - a13 * a28 * a38
        - a13 * a29 * a37
        + a18 * a23 * a38
        + a19 * a22 * a38
        + a19 * a23 * a37
        - a24 * a34 * a110
        - a25 * a33 * a110
        - a26 * a32 * a110
        + a21 * a38 * a110
        + a22 * a37 * a110
        + a23 * a36 * a110
        + a14 * a34 * a210
        + a15 * a33 * a210
        + a16 * a32 * a210
        - a11 * a38 * a210
        - a12 * a37 * a210
        - a13 * a36 * a210
        + a11 * a26 * a313
        + a12 * a25 * a313
        + a12 * a26 * a312
        + a13 * a24 * a313
        + a13 * a25 * a312
        + a13 * a26 * a311
        - a14 * a23 * a313
        - a15 * a22 * a313
        - a15 * a23 * a312
        - a16 * a21 * a313
        - a16 * a22 * a312
        - a16 * a23 * a311;
    c[1] = a16 * a29 * a34 - a19 * a26 * a34 - a13 * a29 * a38 + a19 * a23 * a38
        - a25 * a34 * a110
        - a26 * a33 * a110
        + a22 * a38 * a110
        + a23 * a37 * a110
        + a15 * a34 * a210
        + a16 * a33 * a210
        - a12 * a38 * a210
        - a13 * a37 * a210
        + a12 * a26 * a313
        + a13 * a25 * a313
        + a13 * a26 * a312
        - a15 * a23 * a313
        - a16 * a22 * a313
        - a16 * a23 * a312;
    c[0] = -a26 * a34 * a110 + a23 * a38 * a110 + a16 * a34 * a210 - a13 * a38 * a210
        + a13 * a26 * a313
        - a16 * a23 * a313;

    let roots = bisect_sturm_8(&c, 1.0e-10);
    let mut solutions = Vec::new();
    for xs1 in roots {
        let xs2 = xs1 * xs1;
        let xs3 = xs1 * xs2;
        let xs4 = xs1 * xs3;
        let a = Mat3d::new(
            a11 * xs2 + a12 * xs1 + a13,
            a14 * xs2 + a15 * xs1 + a16,
            a17 * xs3 + a18 * xs2 + a19 * xs1 + a110,
            a21 * xs2 + a22 * xs1 + a23,
            a24 * xs2 + a25 * xs1 + a26,
            a27 * xs3 + a28 * xs2 + a29 * xs1 + a210,
            a31 * xs3 + a32 * xs2 + a33 * xs1 + a34,
            a35 * xs3 + a36 * xs2 + a37 * xs1 + a38,
            a39 * xs4 + a310 * xs3 + a311 * xs2 + a312 * xs1 + a313,
        );
        let denom_y = a[(0, 0)] * a[(1, 1)] - a[(1, 0)] * a[(0, 1)];
        let denom_z = a[(0, 1)] * a[(1, 0)] - a[(1, 1)] * a[(0, 0)];
        if denom_y.abs() <= 1.0e-12 || denom_z.abs() <= 1.0e-12 {
            continue;
        }
        let y = (a[(1, 2)] * a[(0, 1)] - a[(0, 2)] * a[(1, 1)]) / denom_y;
        let z = (a[(1, 2)] * a[(0, 0)] - a[(0, 2)] * a[(1, 0)]) / denom_z;
        let mut solution = Vec3d::new(xs1, y, z);
        if elim_var == 1 {
            solution.swap_rows(0, 1);
        } else if elim_var == 2 {
            solution.swap_rows(0, 2);
        }
        if solution.iter().all(|value| value.is_finite()) {
            solutions.push(solution);
        }
    }

    refine_3q3(coeffs, &mut solutions);
    solutions
}

fn refine_3q3(coeffs: Mat3x10d, solutions: &mut [Vec3d]) {
    for solution in solutions {
        let mut x = solution[0];
        let mut y = solution[1];
        let mut z = solution[2];
        for _ in 0..5 {
            let r = coeffs.column(0) * x * x
                + coeffs.column(1) * x * y
                + coeffs.column(2) * x * z
                + coeffs.column(3) * y * y
                + coeffs.column(4) * y * z
                + coeffs.column(5) * z * z
                + coeffs.column(6) * x
                + coeffs.column(7) * y
                + coeffs.column(8) * z
                + coeffs.column(9);
            if r.abs().max() < 1.0e-8 {
                break;
            }
            let mut j = Mat3d::zeros();
            j.set_column(
                0,
                &(coeffs.column(0) * (2.0 * x)
                    + coeffs.column(1) * y
                    + coeffs.column(2) * z
                    + coeffs.column(6)),
            );
            j.set_column(
                1,
                &(coeffs.column(1) * x
                    + coeffs.column(3) * (2.0 * y)
                    + coeffs.column(4) * z
                    + coeffs.column(7)),
            );
            j.set_column(
                2,
                &(coeffs.column(2) * x
                    + coeffs.column(4) * y
                    + coeffs.column(5) * (2.0 * z)
                    + coeffs.column(8)),
            );
            let Some(dx) = j.try_inverse().map(|inv| inv * r) else {
                break;
            };
            x -= dx[0];
            y -= dx[1];
            z -= dx[2];
        }
        *solution = Vec3d::new(x, y, z);
    }
}

fn bisect_sturm_8(coeffs: &[f64; 9], tol: f64) -> Vec<f64> {
    const N: usize = 8;
    if coeffs[N] == 0.0 {
        return Vec::new();
    }

    let mut fvec = [0.0f64; 2 * N + 1];
    fvec[..=N].copy_from_slice(coeffs);
    let c_inv = 1.0 / fvec[N];
    for value in fvec.iter_mut().take(N) {
        *value *= c_inv;
    }
    fvec[N] = 1.0;
    for i in 0..N - 1 {
        fvec[N + 1 + i] = fvec[i + 1] * ((i + 1) as f64 / N as f64);
    }
    fvec[2 * N] = 1.0;

    let svec = build_sturm_seq_8(&fvec);
    let r0 = 1.0 + fvec[..N].iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    let a = -r0;
    let b = r0;
    let sa = sturm_signchanges_8(&svec, a);
    let sb = sturm_signchanges_8(&svec, b);
    if sa - sb <= 0 {
        return Vec::new();
    }

    let mut roots = Vec::new();
    isolate_sturm_roots_8(&fvec, &svec, a, b, sa, sb, tol, 0, &mut roots);
    roots
}

fn build_sturm_seq_8(fvec: &[f64; 17]) -> [f64; 24] {
    const N: usize = 8;
    let mut f = [0.0f64; 3 * N];
    f[..2 * N + 1].copy_from_slice(fvec);
    let mut f1_start = 0usize;
    let mut f2_start = N + 1;
    let mut f3_start = f2_start + N;
    let mut svec = [0.0f64; 3 * N];

    for i in 0..N - 1 {
        let q1 = f[f1_start + N - i] * f[f2_start + N - 1 - i];
        let q0 = f[f1_start + N - 1 - i] * f[f2_start + N - 1 - i]
            - f[f1_start + N - i] * f[f2_start + N - 2 - i];

        f[f3_start] = f[f1_start] - q0 * f[f2_start];
        for j in 1..N - 1 - i {
            f[f3_start + j] = f[f1_start + j] - q1 * f[f2_start + j - 1] - q0 * f[f2_start + j];
        }
        let c = -f[f3_start + N - 2 - i].abs();
        if c.abs() <= 1.0e-30 {
            continue;
        }
        let ci = 1.0 / c;
        for j in 0..N - 1 - i {
            f[f3_start + j] *= ci;
        }

        std::mem::swap(&mut f1_start, &mut f2_start);
        std::mem::swap(&mut f2_start, &mut f3_start);

        svec[3 * i] = q0;
        svec[3 * i + 1] = q1;
        svec[3 * i + 2] = c;
    }

    svec[3 * N - 3] = f[f1_start];
    svec[3 * N - 2] = f[f1_start + 1];
    svec[3 * N - 1] = f[f2_start];
    svec
}

fn sturm_signchanges_8(svec: &[f64; 24], x: f64) -> i32 {
    const N: usize = 8;
    let mut f = [0.0f64; N + 1];
    f[N] = svec[3 * N - 1];
    f[N - 1] = svec[3 * N - 3] + x * svec[3 * N - 2];
    for i in (0..=N - 2).rev() {
        f[i] = (svec[3 * i] + x * svec[3 * i + 1]) * f[i + 1] + svec[3 * i + 2] * f[i + 2];
    }

    let mut count = 0i32;
    let mut prev_negative = f[0] < 0.0;
    for value in f.iter().take(N + 1).skip(1) {
        let negative = *value < 0.0;
        if prev_negative ^ negative {
            count += 1;
        }
        prev_negative = negative;
    }
    count
}

#[allow(clippy::too_many_arguments)]
fn isolate_sturm_roots_8(
    fvec: &[f64; 17],
    svec: &[f64; 24],
    a: f64,
    b: f64,
    sa: i32,
    sb: i32,
    tol: f64,
    depth: usize,
    roots: &mut Vec<f64>,
) {
    if depth > 300 {
        return;
    }
    if b - a < tol {
        roots.push(b);
        return;
    }

    let n_roots = sa - sb;
    if n_roots > 1 {
        let c = (a + b) * 0.5;
        let sc = sturm_signchanges_8(svec, c);
        isolate_sturm_roots_8(fvec, svec, a, c, sa, sc, tol, depth + 1, roots);
        isolate_sturm_roots_8(fvec, svec, c, b, sc, sb, tol, depth + 1, roots);
    } else if n_roots == 1 {
        if let Some(root) = ridders_newton_8(fvec, a, b, tol) {
            roots.push(root);
        }
    }
}

fn ridders_newton_8(fvec: &[f64; 17], mut a: f64, mut b: f64, tol: f64) -> Option<f64> {
    let mut fa = polyval_8(fvec, a);
    let mut fb = polyval_8(fvec, b);
    if !((fa < 0.0) ^ (fb < 0.0)) {
        return None;
    }

    for _ in 0..30 {
        if (a - b).abs() < 1.0e-3 {
            break;
        }
        let c = (a + b) * 0.5;
        let fc = polyval_8(fvec, c);
        let s_sq = fc * fc - fa * fb;
        if s_sq <= 0.0 {
            break;
        }
        let s = s_sq.sqrt();
        let d = if fa < fb {
            c + (a - c) * fc / s
        } else {
            c + (c - a) * fc / s
        };
        let fd = polyval_8(fvec, d);

        if if fd >= 0.0 { fc < 0.0 } else { fc > 0.0 } {
            a = c;
            fa = fc;
            b = d;
            fb = fd;
        } else if if fd >= 0.0 { fa < 0.0 } else { fa > 0.0 } {
            b = d;
            fb = fd;
        } else {
            a = d;
            fa = fd;
        }
    }

    let mut x = (a + b) * 0.5;
    for _ in 0..10 {
        let fx = polyval_8(fvec, x);
        if fx.abs() < tol {
            break;
        }
        let fpx = 8.0 * polyval_7_derivative_part(fvec, x);
        if fpx.abs() <= 1.0e-15 {
            break;
        }
        let dx = fx / fpx;
        x -= dx;
        if dx.abs() < tol {
            break;
        }
    }
    x.is_finite().then_some(x)
}

fn polyval_8(f: &[f64; 17], x: f64) -> f64 {
    let mut fx = x + f[7];
    for i in (0..=6).rev() {
        fx = x * fx + f[i];
    }
    fx
}

fn polyval_7_derivative_part(f: &[f64; 17], x: f64) -> f64 {
    let mut fx = x + f[15];
    for i in (9..=14).rev() {
        fx = x * fx + f[i];
    }
    fx
}

fn estimate_focal_pose_dlt(
    centered_pts: &[[f32; 2]],
    object_points: &[[f32; 3]],
) -> Option<FocalPoseModel> {
    let n = centered_pts.len().min(object_points.len());
    if n < 6 {
        return None;
    }

    let mut rows = Vec::with_capacity(n * 24);
    for i in 0..n {
        let x = object_points[i][0] as f64;
        let y = object_points[i][1] as f64;
        let z = object_points[i][2] as f64;
        let u = centered_pts[i][0] as f64;
        let v = centered_pts[i][1] as f64;
        rows.extend_from_slice(&[
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

    let a = DMatrix::<f64>::from_row_slice(n * 2, 12, &rows);
    let svd = a.svd(true, true);
    let v_t = svd.v_t?;
    let p_vec = v_t.row(11);
    let mut p = SMatrix::<f64, 3, 4>::zeros();
    for row in 0..3 {
        for col in 0..4 {
            p[(row, col)] = p_vec[row * 4 + col];
        }
    }

    if p.fixed_view::<3, 3>(0, 0).determinant() < 0.0 {
        p = -p;
    }

    let row2_norm = p.fixed_view::<1, 3>(2, 0).norm();
    if !row2_norm.is_finite() || row2_norm <= 1.0e-12 {
        return None;
    }
    p /= row2_norm;

    let fx = p.fixed_view::<1, 3>(0, 0).norm();
    let fy = p.fixed_view::<1, 3>(1, 0).norm();
    if !fx.is_finite() || !fy.is_finite() || fx <= 0.0 || fy <= 0.0 {
        return None;
    }

    let mut rotation = Mat3d::zeros();
    for col in 0..3 {
        rotation[(0, col)] = p[(0, col)] / fx;
        rotation[(1, col)] = p[(1, col)] / fy;
        rotation[(2, col)] = p[(2, col)];
    }
    let svd_r = rotation.svd(true, true);
    let u = svd_r.u?;
    let v_t = svd_r.v_t?;
    let mut correction = Mat3d::identity();
    if (u * v_t).determinant() < 0.0 {
        correction[(2, 2)] = -1.0;
    }
    rotation = u * correction * v_t;

    let focal = (fx + fy) * 0.5;
    let translation = Vec3d::new(p[(0, 3)] / focal, p[(1, 3)] / focal, p[(2, 3)]);
    if rotation.iter().any(|v| !v.is_finite()) || translation.iter().any(|v| !v.is_finite()) {
        return None;
    }

    Some(FocalPoseModel {
        pose: se3_from_f64_rt(&rotation, &translation),
        focal,
    })
}

fn project_focal_pose_model(model: FocalPoseModel, point: &[f32; 3]) -> Option<[f64; 2]> {
    if !model.focal.is_finite() || model.focal <= 0.0 {
        return None;
    }
    let p = model.pose.transform_point(point);
    let z = p[2] as f64;
    if z <= f64::EPSILON {
        return None;
    }
    Some([model.focal * p[0] as f64 / z, model.focal * p[1] as f64 / z])
}

fn focal_pose_model_cost(
    model: FocalPoseModel,
    centered_pts: &[[f32; 2]],
    object_points: &[[f32; 3]],
) -> Option<f64> {
    let mut cost = 0.0;
    let mut count = 0usize;
    for (obs, obj) in centered_pts.iter().zip(object_points.iter()) {
        let projected = project_focal_pose_model(model, obj)?;
        let dx = obs[0] as f64 - projected[0];
        let dy = obs[1] as f64 - projected[1];
        cost += dx * dx + dy * dy;
        count += 1;
    }
    (count > 0).then_some(cost / count as f64)
}

fn perturb_focal_pose_model(
    model: FocalPoseModel,
    dimension: usize,
    eps: f64,
) -> Option<FocalPoseModel> {
    let mut delta = SVector::<f64, 7>::zeros();
    delta[dimension] = eps;
    apply_focal_pose_delta(model, &delta, 1.0)
}

fn apply_focal_pose_delta(
    model: FocalPoseModel,
    delta: &SVector<f64, 7>,
    step: f64,
) -> Option<FocalPoseModel> {
    let mut twist = [0.0f32; 6];
    for i in 0..6 {
        twist[i] = (delta[i] * step) as f32;
    }
    let pose = SE3::exp(&twist).compose(&model.pose);
    let focal = model.focal + delta[6] * step;
    (focal.is_finite() && focal > 0.0).then_some(FocalPoseModel { pose, focal })
}

fn deduplicate_focal_models(models: Vec<FocalPoseModel>) -> Vec<FocalPoseModel> {
    let mut unique = Vec::<FocalPoseModel>::new();
    'models: for model in models {
        for existing in &unique {
            if (model.focal - existing.focal).abs() <= 1.0e-6 * model.focal.max(1.0)
                && pose_matrix_3x4_distance(model.pose, existing.pose) <= 1.0e-5
            {
                continue 'models;
            }
        }
        unique.push(model);
    }
    unique
}

fn pose_matrix_3x4_distance(a: SE3, b: SE3) -> f32 {
    let a = a.to_matrix();
    let b = b.to_matrix();
    let mut sum = 0.0f32;
    for row in 0..3 {
        for col in 0..4 {
            let diff = a[row][col] - b[row][col];
            sum += diff * diff;
        }
    }
    sum.sqrt()
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
        let mut rng = ColmapMt19937::new(n as u64 + (self.ransac_max_iterations as u64));
        let mut best_inliers = Vec::new();
        let mut best_e = None;
        let mut best_inlier_count = 0usize;
        let mut best_error = f32::INFINITY;

        for _ in 0..self.ransac_max_iterations {
            let sample = sample_unique_indices(&mut rng, n, 8);
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

pub fn compute_ransac_num_trials(
    num_inliers: usize,
    num_samples: usize,
    sample_size: usize,
    confidence: f32,
    num_trials_multiplier: f32,
) -> u32 {
    if num_samples < sample_size || num_inliers < sample_size {
        return u32::MAX;
    }
    let prob_failure = 1.0f64 - confidence as f64;
    if prob_failure <= 0.0 {
        return u32::MAX;
    }

    let mut prob_inlier = 1.0f64;
    for i in 0..sample_size {
        let denominator = num_samples - i;
        if denominator == 0 || num_inliers < i {
            return u32::MAX;
        }
        prob_inlier *= (num_inliers - i) as f64 / denominator as f64;
    }

    let prob_outlier = 1.0 - prob_inlier;
    if prob_outlier <= 0.0 {
        return 1;
    }
    if prob_outlier >= 1.0 {
        return u32::MAX;
    }

    let trials = (prob_failure.ln() / prob_outlier.ln() * num_trials_multiplier as f64).ceil();
    if !trials.is_finite() || trials >= u32::MAX as f64 {
        u32::MAX
    } else {
        trials.max(1.0) as u32
    }
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
