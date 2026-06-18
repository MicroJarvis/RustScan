//! Tests for Geometric Solvers
//!
//! Tests PnP, Essential Matrix, Triangulation, and Sim3 solvers.

#[cfg(test)]
mod tests {
    use crate::core::SE3;
    use crate::tracker::solver::{
        compute_ransac_num_trials, ColmapRng, EssentialSolver, PnPProblem, PnPSolver, Sim3Solver,
        Triangulator,
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
        assert_eq!(solver.ransac_confidence, 0.99);
        assert_eq!(solver.ransac_min_inlier_ratio, 0.0);
        assert_eq!(solver.ransac_dyn_num_trials_multiplier, 3.0);
        assert_eq!(solver.ransac_min_iterations, 0);
        assert_eq!(solver.ransac_random_seed, None);
    }

    #[test]
    fn test_ransac_num_trials_matches_colmap_formula_examples() {
        assert_eq!(compute_ransac_num_trials(1, 100, 3, 0.99, 1.0), u32::MAX);
        assert_eq!(compute_ransac_num_trials(10, 100, 3, 0.99, 1.0), 6204);
        assert_eq!(compute_ransac_num_trials(10, 100, 3, 0.999, 1.0), 9305);
        assert_eq!(compute_ransac_num_trials(10, 100, 3, 0.999, 2.0), 18610);
        assert_eq!(compute_ransac_num_trials(50, 100, 3, 0.99, 1.0), 36);
    }

    #[test]
    fn test_colmap_rng_matches_mt19937_reference_outputs() {
        let mut rng = ColmapRng::new(0);
        let outputs = [
            rng.next_u32(),
            rng.next_u32(),
            rng.next_u32(),
            rng.next_u32(),
            rng.next_u32(),
        ];
        assert_eq!(
            outputs,
            [
                2_357_136_044,
                2_546_248_239,
                3_071_714_933,
                3_626_093_760,
                2_588_848_963,
            ]
        );
    }

    #[test]
    fn test_epnp_matches_colmap_absolute_pose_examples() {
        let solver = PnPSolver::new(1.0, 1.0, 0.0, 0.0);
        let points3d = vec![
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [3.0, 1.0, 4.0],
            [3.0, 1.1, 4.0],
            [3.0, 1.2, 4.0],
            [3.0, 1.3, 4.0],
            [3.0, 1.4, 4.0],
            [2.0, 1.0, 7.0],
        ];

        for qx_idx in 0..5 {
            for tx_idx in 0..10 {
                let qx = qx_idx as f32 * 0.2;
                let tx = tx_idx as f32 * 0.1;
                let q_norm = (1.0 + qx * qx).sqrt();
                let expected = SE3::new(&[qx / q_norm, 0.0, 0.0, 1.0 / q_norm], &[tx, 0.0, 0.0]);

                let mut image_points = Vec::with_capacity(points3d.len());
                for point_world in &points3d {
                    let point_camera = expected.transform_point(point_world);
                    image_points.push([
                        point_camera[0] / point_camera[2],
                        point_camera[1] / point_camera[2],
                    ]);
                }

                let estimated = solver
                    .estimate_pose_epnp(&image_points, &points3d)
                    .expect("epnp pose");
                let matrix_error = pose_matrix_3x4_error(&expected, &estimated);
                assert!(
                    matrix_error < 2.0e-3,
                    "qx={qx}, tx={tx}, matrix_error={matrix_error}, expected={:?}, estimated={:?}",
                    expected.to_matrix(),
                    estimated.to_matrix()
                );
            }
        }
    }

    #[test]
    fn test_epnp_colmap_broken_solve_sign_case() {
        let solver = PnPSolver::new(1.0, 1.0, 0.0, 0.0);
        let image_points = vec![
            [-2.6783007931074532e-01, 5.3457197430746251e-01],
            [-4.2629907287470264e-01, 7.5623350319519789e-01],
            [-1.6767413005963930e-01, -1.3387172544910089e-01],
            [-5.6616329720373559e-02, 2.3621156497739373e-01],
            [-1.7721225948969935e-01, 2.3395366792735982e-02],
            [-5.1836259886632222e-02, -4.4380694271927049e-02],
            [-3.5897765845560037e-01, 1.6252721078589397e-01],
            [2.7057324473684058e-01, -1.4067450104631887e-01],
            [-2.5811166424334520e-01, 8.0167171300227366e-02],
            [2.0239567448222310e-02, -3.2845953375344145e-01],
            [4.2571014715170657e-01, -2.8321173570154773e-01],
            [-5.4597596412987237e-01, 9.1431935871671977e-02],
        ];
        let points3d = vec![
            [4.4276865308679305, -1.3384364366019632, -3.5997423085253892],
            [2.7278555252512309, -0.3815299618723123, -2.6558518399902824],
            [4.8548566083054894, -1.4756197433631739, -0.682749460224905],
            [3.152301352799845, -1.3377020437938025, -1.6443269301929087],
            [3.8551679771512073, -1.055770054588555, -1.1695994508851486],
            [5.957137315035381, -2.6120646101684555, -1.0841441206050342],
            [6.328708849935889, -1.1761274755817175, -2.5951879774151583],
            [2.300530599012125, -1.4019796626800123, -0.4448546445507232],
            [5.981685993458735, -1.4211814511691452, -2.028592388929345],
            [5.254334469066546, -2.3389255564264144, 0.4370817318552405],
            [3.218159924599169, -2.89066719884451, 0.2682571815006435],
            [4.4592895306946758, -0.00912352416415799, -1.655523711797087],
        ];

        let estimated = solver
            .estimate_pose_epnp(&image_points, &points3d)
            .expect("epnp pose");
        let mut reprojection_error = 0.0;
        for (image_point, point_world) in image_points.iter().zip(points3d.iter()) {
            let point_camera = estimated.transform_point(point_world);
            let projected = [
                point_camera[0] / point_camera[2],
                point_camera[1] / point_camera[2],
            ];
            reprojection_error += ((projected[0] - image_point[0]).powi(2)
                + (projected[1] - image_point[1]).powi(2))
            .sqrt();
        }

        assert!(
            reprojection_error < 0.2,
            "reprojection_error={reprojection_error}"
        );
    }

    #[test]
    fn test_pnp_estimated_focal_improves_unknown_focal_initialization() {
        let points3d = vec![
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [3.0, 1.0, 4.0],
            [3.0, 1.1, 4.0],
            [3.0, 1.2, 4.0],
            [3.0, 1.3, 4.0],
            [3.0, 1.4, 4.0],
            [2.0, 1.0, 7.0],
        ];

        for (qx, tx, focal) in [(0.0f32, 0.0f32, 4.5f32), (0.2, 0.3, 12.5)] {
            let q_norm = (1.0 + qx * qx).sqrt();
            let expected = SE3::new(&[qx / q_norm, 0.0, 0.0, 1.0 / q_norm], &[tx, 0.0, 0.0]);
            let mut problem = PnPProblem::new();
            for point_world in &points3d {
                let point_camera = expected.transform_point(point_world);
                problem.add_correspondence(
                    [
                        focal * point_camera[0] / point_camera[2],
                        focal * point_camera[1] / point_camera[2],
                    ],
                    *point_world,
                );
            }

            let initial_focal = 3.0f32;
            let mut solver = PnPSolver::new(initial_focal, initial_focal, 0.0, 0.0);
            solver.ransac_threshold = 2.0e-1;
            solver.ransac_max_iterations = 20;
            solver.ransac_random_seed = Some(7);
            let result = solver
                .solve_with_estimated_focal(&problem)
                .expect("estimated focal pose");

            assert!(
                (result.focal - focal).abs() < (initial_focal - focal).abs(),
                "qx={qx}, tx={tx}, focal={} expected={focal}, initial={initial_focal}",
                result.focal
            );
            assert!(result.inliers.iter().filter(|&&x| x).count() >= points3d.len() - 1);
            let mut reprojection_error = 0.0f32;
            for (image_point, point_world) in problem.image_points.iter().zip(points3d.iter()) {
                let point_camera = result.pose.transform_point(point_world);
                let projected = [
                    result.focal * point_camera[0] / point_camera[2],
                    result.focal * point_camera[1] / point_camera[2],
                ];
                reprojection_error += ((projected[0] - image_point[0]).powi(2)
                    + (projected[1] - image_point[1]).powi(2))
                .sqrt();
            }
            assert!(
                reprojection_error / points3d.len() as f32 <= solver.ransac_threshold,
                "qx={qx}, tx={tx}, focal={focal}, reprojection_error={reprojection_error}"
            );
        }
    }

    #[test]
    fn test_pnp_estimated_focal_accepts_four_point_minimal_samples() {
        let points3d = vec![
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [3.0, 1.0, 4.0],
            [2.0, 1.0, 7.0],
        ];
        let expected = SE3::new(
            &[
                0.2 / (1.0f32 + 0.2 * 0.2).sqrt(),
                0.0,
                0.0,
                1.0 / (1.0f32 + 0.2 * 0.2).sqrt(),
            ],
            &[0.3, 0.0, 0.0],
        );
        let focal = 8.0f32;
        let mut problem = PnPProblem::new();
        for point_world in &points3d {
            let point_camera = expected.transform_point(point_world);
            problem.add_correspondence(
                [
                    focal * point_camera[0] / point_camera[2],
                    focal * point_camera[1] / point_camera[2],
                ],
                *point_world,
            );
        }

        let mut solver = PnPSolver::new(3.0, 3.0, 0.0, 0.0);
        solver.ransac_threshold = 1.0e-1;
        solver.ransac_max_iterations = 1;
        solver.ransac_random_seed = Some(11);
        let result = solver
            .solve_with_estimated_focal(&problem)
            .expect("four-point estimated focal pose");

        assert_eq!(
            result.inliers.iter().filter(|&&x| x).count(),
            points3d.len()
        );
        let mean_error =
            mean_centered_pixel_error(&problem.image_points, &points3d, result.focal, result.pose);
        assert!(
            mean_error <= solver.ransac_threshold,
            "mean_error={mean_error}, focal={}",
            result.focal
        );
        assert!(
            (result.focal - focal).abs() < (3.0 - focal).abs(),
            "focal={} expected={focal}",
            result.focal
        );
    }

    #[test]
    fn test_pnp_estimated_focal_matches_colmap_p4pf_examples() {
        let points3d = vec![
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [3.0, 1.0, 4.0],
            [3.0, 1.1, 4.0],
            [3.0, 1.2, 4.0],
            [3.0, 1.3, 4.0],
            [3.0, 1.4, 4.0],
            [2.0, 1.0, 7.0],
        ];

        for (qx, tx, focal) in [(0.0f32, 0.0f32, 4.5f32), (0.2, 0.3, 12.5)] {
            let q_norm = (1.0 + qx * qx).sqrt();
            let expected = SE3::new(&[qx / q_norm, 0.0, 0.0, 1.0 / q_norm], &[tx, 0.0, 0.0]);
            let mut problem = PnPProblem::new();
            for point_world in &points3d {
                let point_camera = expected.transform_point(point_world);
                problem.add_correspondence(
                    [
                        focal * point_camera[0] / point_camera[2],
                        focal * point_camera[1] / point_camera[2],
                    ],
                    *point_world,
                );
            }

            let mut solver = PnPSolver::new(3.0, 3.0, 0.0, 0.0);
            solver.ransac_threshold = 1.0e-5;
            solver.ransac_max_iterations = 100;
            solver.ransac_random_seed = Some(3);
            let result = solver
                .solve_with_estimated_focal(&problem)
                .expect("p4pf-style estimated focal pose");

            assert!(
                (result.focal - focal).abs() < 1.0e-3,
                "focal={} expected={focal}",
                result.focal
            );
            let matrix_error = pose_matrix_3x4_error(&expected, &result.pose);
            assert!(
                matrix_error < 1.0e-3,
                "qx={qx}, tx={tx}, focal={focal}, matrix_error={matrix_error}"
            );
            let mean_error = mean_centered_pixel_error(
                &problem.image_points,
                &points3d,
                result.focal,
                result.pose,
            );
            assert!(
                mean_error < 1.0e-3,
                "qx={qx}, tx={tx}, focal={focal}, mean_error={mean_error}"
            );
        }
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

    #[test]
    fn test_pnp_threshold_respects_high_focal_pixel_error() {
        let mut solver = PnPSolver::new(3000.0, 3000.0, 768.0, 1024.0);
        solver.ransac_threshold = 8.0;
        solver.ransac_max_iterations = 5000;
        let pose = SE3::identity();
        let object_points: Vec<[f32; 3]> = (0..36)
            .map(|idx| {
                let x = ((idx % 6) as f32 - 2.5) * 0.25;
                let y = (((idx / 6) % 6) as f32 - 2.5) * 0.2;
                let z = 4.0 + (idx % 7) as f32 * 0.25;
                [x, y, z]
            })
            .collect();

        let mut problem = PnPProblem::new();
        for (idx, point_world) in object_points.iter().enumerate() {
            let point_camera = pose.transform_point(point_world);
            let mut pixel = [
                solver.fx * point_camera[0] / point_camera[2] + solver.cx,
                solver.fy * point_camera[1] / point_camera[2] + solver.cy,
            ];
            if idx == object_points.len() - 1 {
                pixel[0] += 20.0;
            }
            problem.add_correspondence(pixel, *point_world);
        }

        let (_estimated_pose, inliers) = solver.solve(&problem).expect("pnp pose");
        assert!(
            !inliers[object_points.len() - 1],
            "20px error must be outside an 8px high-focal PnP threshold"
        );
    }

    fn pose_matrix_3x4_error(expected: &SE3, estimated: &SE3) -> f32 {
        let expected = expected.to_matrix();
        let estimated = estimated.to_matrix();
        let mut sum = 0.0;
        for row in 0..3 {
            for col in 0..4 {
                let diff = expected[row][col] - estimated[row][col];
                sum += diff * diff;
            }
        }
        sum.sqrt()
    }

    fn mean_centered_pixel_error(
        image_points: &[[f32; 2]],
        points3d: &[[f32; 3]],
        focal: f32,
        pose: SE3,
    ) -> f32 {
        let mut total = 0.0;
        for (image_point, point_world) in image_points.iter().zip(points3d.iter()) {
            let point_camera = pose.transform_point(point_world);
            let projected = [
                focal * point_camera[0] / point_camera[2],
                focal * point_camera[1] / point_camera[2],
            ];
            total += ((projected[0] - image_point[0]).powi(2)
                + (projected[1] - image_point[1]).powi(2))
            .sqrt();
        }
        total / image_points.len().max(1) as f32
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

    #[test]
    fn test_essential_recover_pose_matches_known_relative_pose() {
        let solver = EssentialSolver {
            ransac_threshold: 1.0e-2,
            ransac_max_iterations: 256,
        };
        let triangulator = Triangulator::new();
        let truth = SE3::from_axis_angle(&[0.015, -0.02, 0.01], &[0.25, -0.04, 0.02]);
        let points = [
            [-0.8, -0.5, 4.0],
            [-0.3, -0.4, 4.8],
            [0.2, -0.35, 5.2],
            [0.7, -0.25, 4.5],
            [-0.6, 0.1, 5.4],
            [-0.1, 0.2, 4.2],
            [0.4, 0.25, 5.7],
            [0.9, 0.3, 4.9],
            [-0.5, 0.55, 6.1],
            [0.1, 0.6, 5.5],
            [0.6, 0.65, 4.6],
            [1.0, 0.7, 6.0],
        ];
        let mut pts1 = Vec::with_capacity(points.len());
        let mut pts2 = Vec::with_capacity(points.len());
        for point in points {
            let right = truth.transform_point(&point);
            pts1.push([point[0] / point[2], point[1] / point[2]]);
            pts2.push([right[0] / right[2], right[1] / right[2]]);
        }

        let truth_rotation = truth.rotation_matrix();
        let rotation = Mat3::from_cols(
            Vec3::new(
                truth_rotation[0][0],
                truth_rotation[1][0],
                truth_rotation[2][0],
            ),
            Vec3::new(
                truth_rotation[0][1],
                truth_rotation[1][1],
                truth_rotation[2][1],
            ),
            Vec3::new(
                truth_rotation[0][2],
                truth_rotation[1][2],
                truth_rotation[2][2],
            ),
        );
        let translation = Vec3::from(truth.translation());
        let skew_translation = Mat3::from_cols(
            Vec3::new(0.0, translation.z, -translation.y),
            Vec3::new(-translation.z, 0.0, translation.x),
            Vec3::new(translation.y, -translation.x, 0.0),
        );
        let true_essential = skew_translation * rotation;

        assert_recovered_pose_matches(
            &solver,
            &triangulator,
            &truth,
            true_essential,
            &pts1,
            &pts2,
            1.0e-3,
            1.0e-3,
        );

        let (essential, inliers) = solver
            .compute(&[], &pts1, &pts2)
            .expect("synthetic essential matrix");
        assert!(inliers.iter().filter(|&&value| value).count() >= 8);

        assert_recovered_pose_matches(
            &solver,
            &triangulator,
            &truth,
            essential,
            &pts1,
            &pts2,
            1.0e-3,
            1.0e-3,
        );
    }

    fn assert_recovered_pose_matches(
        solver: &EssentialSolver,
        triangulator: &Triangulator,
        truth: &SE3,
        essential: Mat3,
        pts1: &[[f32; 2]],
        pts2: &[[f32; 2]],
        max_rotation_error_rad: f32,
        max_translation_error_rad: f32,
    ) {
        let candidates = solver.recover_pose(essential);
        let best = candidates
            .iter()
            .max_by_key(|candidate| {
                triangulator
                    .triangulate(&SE3::identity(), candidate, pts1, pts2)
                    .iter()
                    .filter(|point| point.is_some())
                    .count()
            })
            .expect("pose candidate");

        let truth_rotation = truth.rotation_matrix();
        let best_rotation = best.rotation_matrix();
        let mut trace = 0.0f32;
        for row in 0..3 {
            for col in 0..3 {
                trace += truth_rotation[row][col] * best_rotation[row][col];
            }
        }
        let rotation_error = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0).acos();
        assert!(
            rotation_error < max_rotation_error_rad,
            "rotation error too large: {} rad",
            rotation_error
        );

        let truth_t = Vec3::from(truth.translation()).normalize();
        let best_t = Vec3::from(best.translation()).normalize();
        let translation_angle = truth_t.dot(best_t).clamp(-1.0, 1.0).acos();
        assert!(
            translation_angle < max_translation_error_rad,
            "translation direction error too large: {} rad",
            translation_angle
        );
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
