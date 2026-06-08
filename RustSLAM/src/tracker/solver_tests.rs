//! Tests for Geometric Solvers
//!
//! Tests PnP, Essential Matrix, Triangulation, and Sim3 solvers.

#[cfg(test)]
mod tests {
    use crate::core::SE3;
    use crate::tracker::solver::{
        EssentialSolver, PnPProblem, PnPSolver, Sim3Solver, Triangulator,
    };
    use glam::{Mat3, Vec3};

    // =========================================================================
    // PnP Solver Tests
    // =========================================================================

    #[test]
    fn test_pnp_solver_creation() {
        let solver = PnPSolver::new(500.0, 500.0, 320.0, 240.0);
        assert_eq!(solver.fx, 500.0);
        assert_eq!(solver.fy, 500.0);
        assert_eq!(solver.cx, 320.0);
        assert_eq!(solver.cy, 240.0);
    }

    #[test]
    fn test_pnp_problem() {
        let mut problem = PnPProblem::new();
        problem.add_correspondence([100.0, 100.0], [0.0, 0.0, 0.0]);
        problem.add_correspondence([200.0, 100.0], [1.0, 0.0, 0.0]);
        problem.add_correspondence([100.0, 200.0], [0.0, 1.0, 0.0]);
        problem.add_correspondence([200.0, 200.0], [1.0, 1.0, 0.0]);

        assert!(problem.is_solvable());
        assert_eq!(problem.image_points.len(), 4);
    }

    #[test]
    fn test_pnp_solve_simple() {
        let solver = PnPSolver::new(500.0, 500.0, 320.0, 240.0);

        let mut problem = PnPProblem::new();
        // 6 non-coplanar 3D points with correct projections for identity pose
        // u = fx * X/Z + cx, v = fy * Y/Z + cy
        problem.add_correspondence([320.0, 240.0], [0.0, 0.0, 5.0]);
        problem.add_correspondence([420.0, 240.0], [1.0, 0.0, 5.0]);
        problem.add_correspondence([320.0, 340.0], [0.0, 1.0, 5.0]);
        problem.add_correspondence([445.0, 365.0], [1.0, 1.0, 4.0]);
        problem.add_correspondence([236.67, 156.67], [-1.0, -1.0, 6.0]);
        problem.add_correspondence([403.33, 156.67], [0.5, -0.5, 3.0]);

        let result = solver.solve(&problem);

        // Should return a valid pose for consistent correspondences
        assert!(result.is_some());
        let (_pose, inliers) = result.unwrap();

        // Check that we have inliers
        assert!(!inliers.is_empty());
    }

    #[test]
    fn test_pnp_recovers_non_unit_translation_for_absolute_pose() {
        let solver = PnPSolver::new(500.0, 500.0, 320.0, 240.0);
        let pose = SE3::from_axis_angle(&[0.015, -0.01, 0.02], &[0.35, -0.15, 0.6]);

        let object_points = [
            [-0.8, -0.4, 4.5],
            [-0.1, -0.3, 4.8],
            [0.5, -0.2, 5.1],
            [0.9, -0.1, 5.4],
            [-0.7, 0.2, 4.7],
            [-0.2, 0.4, 5.0],
            [0.3, 0.3, 5.3],
            [0.8, 0.5, 5.6],
        ];

        let mut problem = PnPProblem::new();
        for point_world in object_points {
            let point_camera = pose.transform_point(&point_world);
            let pixel = [
                solver.fx * point_camera[0] / point_camera[2] + solver.cx,
                solver.fy * point_camera[1] / point_camera[2] + solver.cy,
            ];
            problem.add_correspondence(pixel, point_world);
        }

        let (estimated_pose, inliers) = solver.solve(&problem).expect("pnp pose");
        assert_eq!(
            inliers.iter().filter(|&&x| x).count(),
            problem.image_points.len()
        );

        let expected_t = pose.translation();
        let estimated_t = estimated_pose.translation();
        let err_t = ((estimated_t[0] - expected_t[0]).powi(2)
            + (estimated_t[1] - expected_t[1]).powi(2)
            + (estimated_t[2] - expected_t[2]).powi(2))
        .sqrt();
        assert!(
            err_t < 0.2,
            "expected translation {:?}, got {:?}, err={err_t}",
            expected_t,
            estimated_t
        );

        let expected_q = pose.quaternion();
        let estimated_q = estimated_pose.quaternion();
        let dot = (expected_q[0] * estimated_q[0]
            + expected_q[1] * estimated_q[1]
            + expected_q[2] * estimated_q[2]
            + expected_q[3] * estimated_q[3])
            .abs();
        assert!(
            dot > 0.99,
            "expected quaternion {:?}, got {:?}",
            expected_q,
            estimated_q
        );
    }

    #[test]
    fn test_pnp_recovers_dense_relocalization_geometry() {
        let solver = PnPSolver::new(500.0, 500.0, 320.0, 240.0);
        let pose = SE3::from_axis_angle(&[0.01, -0.015, 0.005], &[0.35, -0.1, 0.55]);
        let object_points: Vec<[f32; 3]> = (0..24)
            .map(|idx| {
                let x = ((idx % 6) as f32 - 2.5) * 0.35;
                let y = ((idx / 6) as f32 - 1.5) * 0.3;
                let z = 4.5 + (idx % 4) as f32 * 0.35;
                [x, y, z]
            })
            .collect();

        let mut problem = PnPProblem::new();
        for point_world in &object_points {
            let point_camera = pose.transform_point(point_world);
            let pixel = [
                solver.fx * point_camera[0] / point_camera[2] + solver.cx,
                solver.fy * point_camera[1] / point_camera[2] + solver.cy,
            ];
            problem.add_correspondence(pixel, *point_world);
        }

        let (estimated_pose, inliers) = solver.solve(&problem).expect("dense pnp pose");
        assert_eq!(
            inliers.iter().filter(|&&x| x).count(),
            problem.image_points.len()
        );

        let expected_t = pose.translation();
        let estimated_t = estimated_pose.translation();
        let err_t = ((estimated_t[0] - expected_t[0]).powi(2)
            + (estimated_t[1] - expected_t[1]).powi(2)
            + (estimated_t[2] - expected_t[2]).powi(2))
        .sqrt();
        assert!(
            err_t < 0.2,
            "expected translation {:?}, got {:?}, err={err_t}",
            expected_t,
            estimated_t
        );
    }

    // =========================================================================
    // Essential Matrix Solver Tests
    // =========================================================================

    #[test]
    fn test_essential_solver_creation() {
        let solver = EssentialSolver::new();
        assert_eq!(solver.ransac_threshold, 0.01);
        assert_eq!(solver.ransac_max_iterations, 200);
    }

    #[test]
    fn test_essential_matrix_from_matches() {
        let solver = EssentialSolver::new();

        // Create matched points (simulating two views) - need at least 8
        let pts1: Vec<[f32; 2]> = vec![
            [100.0, 100.0],
            [200.0, 100.0],
            [100.0, 200.0],
            [200.0, 200.0],
            [150.0, 150.0],
            [120.0, 180.0],
            [180.0, 120.0],
            [160.0, 160.0],
        ];

        let pts2: Vec<[f32; 2]> = vec![
            [110.0, 110.0], // slight translation
            [210.0, 110.0],
            [110.0, 210.0],
            [210.0, 210.0],
            [160.0, 160.0],
            [130.0, 190.0],
            [190.0, 130.0],
            [170.0, 170.0],
        ];

        let result = solver.compute(&[], &pts1, &pts2);

        // Should return an essential matrix
        assert!(result.is_some());
        let (E, _inliers) = result.unwrap();

        // E should be 3x3 (check using row/cols methods)
        let _ = E.row(0); // This is how we access rows in glam

        // Check rank-2 constraint (det(E) ≈ 0)
        let det = E.determinant();
        assert!(
            det.abs() < 0.1,
            "Essential matrix should have det ≈ 0, got {}",
            det
        );
    }

    #[test]
    fn test_essential_matrix_enforce_rank2() {
        let solver = EssentialSolver::new();

        let pts1: Vec<[f32; 2]> = vec![[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]];
        let pts2: Vec<[f32; 2]> = vec![[105.0, 105.0], [205.0, 205.0], [305.0, 305.0]];

        if let Some((E, _)) = solver.compute(&[], &pts1, &pts2) {
            // Enforce rank-2 constraint by SVD
            let _ = solver.enforce_rank2(E);
        }
    }

    // =========================================================================
    // Triangulation Tests
    // =========================================================================

    #[test]
    fn test_triangulator_creation() {
        let tri = Triangulator::new();
        assert!(tri.min_angle > 0.0);
        assert!(tri.min_dist > 0.0);
    }

    #[test]
    fn test_triangulate_simple() {
        let tri = Triangulator::new();

        // Two camera poses
        let pose1 = SE3::identity(); // First camera at origin
        let pose2 = SE3::from_axis_angle(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0]); // Second camera at (1, 0, 0)

        // Corresponding 2D points (projection of [0.5, 0, 1])
        let pts1: Vec<[f32; 2]> = vec![[320.0, 240.0]]; // (0.5, 1) * f + principal
        let pts2: Vec<[f32; 2]> = vec![[220.0, 240.0]]; // shifted due to camera translation

        let results = tri.triangulate(&pose1, &pose2, &pts1, &pts2);

        assert_eq!(results.len(), 1);

        // Check if triangulation produced valid 3D point
        if let Some(point) = results[0] {
            assert!(point[2] > 0.0, "Point should be in front of camera");
        }
    }

    #[test]
    fn test_triangulate_multiple_points() {
        let tri = Triangulator::new();

        let pose1 = SE3::identity();
        let pose2 = SE3::from_axis_angle(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0]);

        // Points at z=4 in world coordinates.
        // View 1 (P=[I|0]): project (x,y,z) → (x/z, y/z)
        // View 2 (P=[I|1,0,0]): project (x,y,z) → ((x+1)/z, y/z)
        let pts1: Vec<[f32; 2]> = vec![
            [0.0, 0.0],   // world (0, 0, 4)
            [0.3, 0.1],   // world (1.2, 0.4, 4)
            [-0.2, 0.15], // world (-0.8, 0.6, 4)
        ];
        let pts2: Vec<[f32; 2]> = vec![
            [0.25, 0.0],  // (0+1)/4 = 0.25
            [0.55, 0.1],  // (1.2+1)/4 = 0.55
            [0.05, 0.15], // (-0.8+1)/4 = 0.05
        ];

        let results = tri.triangulate(&pose1, &pose2, &pts1, &pts2);

        let valid_count = results.iter().filter(|p| p.is_some()).count();
        assert!(
            valid_count > 0,
            "Should have at least some valid triangulated points, got results: {:?}",
            results
        );
    }

    #[test]
    fn test_triangulation_check_angle() {
        let tri = Triangulator::new();

        // Cameras too close - should fail angle check
        let pose1 = SE3::identity();
        let pose2 = SE3::from_axis_angle(&[0.0, 0.0, 0.0], &[0.01, 0.0, 0.0]); // Very small baseline

        let pts1: Vec<[f32; 2]> = vec![[320.0, 240.0]];
        let pts2: Vec<[f32; 2]> = vec![[320.0, 240.0]];

        let results = tri.triangulate(&pose1, &pose2, &pts1, &pts2);

        // Should return None due to small angle
        assert!(results[0].is_none() || results.len() == 0);
    }

    // =========================================================================
    // Sim3 Solver Tests
    // =========================================================================

    #[test]
    fn test_sim3_solver_creation() {
        let solver = Sim3Solver::new(0.01);
        assert_eq!(solver.ransac_threshold, 0.01);
    }

    #[test]
    fn test_sim3_compute() {
        let solver = Sim3Solver::new(0.01);

        // 3D points in first view
        let pts1: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];

        // 3D points in second view (scaled by 2x, rotated, translated)
        let pts2: Vec<[f32; 3]> = vec![
            [2.0, 0.0, 2.0], // scale 2x
            [4.0, 0.0, 2.0],
            [2.0, 2.0, 2.0],
            [4.0, 2.0, 2.0],
        ];

        let result = solver.compute(&pts1, &pts2);

        // Should return similarity transform
        assert!(result.is_some());
        let (sim3, inliers) = result.unwrap();

        // Check scale is approximately 2
        let scale = sim3.0;
        assert!((scale - 2.0).abs() < 0.5, "Scale should be ~2");

        // Should have inliers
        assert!(!inliers.is_empty());
    }

    #[test]
    fn test_sim3_apply() {
        let solver = Sim3Solver::new(0.01);

        // Simple scale 2x transform
        let sim3 = solver.create_sim3(2.0, Vec3::ZERO, Mat3::IDENTITY);

        // Apply to a point
        let point = [1.0, 2.0, 3.0];
        let transformed = solver.apply_sim3(sim3, point);

        // Should be scaled by 2
        assert!((transformed[0] - 2.0).abs() < 0.001);
        assert!((transformed[1] - 4.0).abs() < 0.001);
        assert!((transformed[2] - 6.0).abs() < 0.001);
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================
}
