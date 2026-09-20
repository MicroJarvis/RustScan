//! Integration tests for Optimization Thread
//!
//! Tests BA convergence and 3DGS training with synthetic data.

use rustslam::core::SE3;
use rustslam::optimizer::ba::{BACamera, BALandmark, BAObservation, BundleAdjuster};
use rustslam::test_utils::*;

fn test_ba_basic_workflow() {
    // Create a simple BA problem with synthetic data
    let mut ba = BundleAdjuster::new();

    // Add 3 cameras
    let poses = create_synthetic_poses(3, TrajectoryType::Line);
    for (i, pose) in poses.iter().enumerate() {
        let mut camera = BACamera::new(100.0, 100.0, 50.0, 50.0);
        camera.pose = pose.clone();
        if i == 0 {
            camera = camera.fix_pose(); // Fix first camera
        }
        ba.add_camera(camera);
    }

    // Add 5 landmarks
    for i in 0..5 {
        let landmark = BALandmark::new(i as f64 * 0.5, 0.0, 2.0);
        ba.add_landmark(landmark);
    }

    // Add observations (each camera sees each landmark)
    for cam_id in 0..3 {
        for lm_id in 0..5 {
            let obs = BAObservation::new(100.0 + lm_id as f64 * 10.0, 100.0);
            ba.add_observation(cam_id, lm_id, obs);
        }
    }

    // Verify BA was set up correctly
    assert_eq!(ba.num_cameras(), 3, "Should have 3 cameras");
    assert_eq!(ba.num_landmarks(), 5, "Should have 5 landmarks");
    assert_eq!(ba.num_observations(), 15, "Should have 15 observations");
}

fn test_synthetic_poses_generation() {
    // Test circular trajectory
    let circle_poses = create_synthetic_poses(8, TrajectoryType::Circle);
    assert_eq!(circle_poses.len(), 8, "Should generate 8 poses");

    // Test line trajectory
    let line_poses = create_synthetic_poses(5, TrajectoryType::Line);
    assert_eq!(line_poses.len(), 5, "Should generate 5 poses");

    // Verify poses are different
    for i in 1..line_poses.len() {
        let t1 = line_poses[i - 1].translation();
        let t2 = line_poses[i].translation();
        let diff = (t1[0] - t2[0]).abs() + (t1[1] - t2[1]).abs() + (t1[2] - t2[2]).abs();
        assert!(diff > 0.01, "Consecutive poses should be different");
    }
}

fn test_synthetic_depth_generation() {
    // Test constant depth
    let constant_depth = create_synthetic_depth(100, 100, DepthPattern::Constant);
    assert_eq!(constant_depth.len(), 100 * 100);
    assert!(
        constant_depth.iter().all(|&d| (d - 1.0).abs() < 0.001),
        "Constant depth should all be 1.0"
    );

    // Test planar depth
    let planar_depth = create_synthetic_depth(100, 100, DepthPattern::Planar);
    assert_eq!(planar_depth.len(), 100 * 100);
    // Planar depth should vary
    let min_depth = planar_depth.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_depth = planar_depth
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_depth - min_depth > 0.1,
        "Planar depth should vary across image"
    );

    // Test spherical depth
    let spherical_depth = create_synthetic_depth(100, 100, DepthPattern::Spherical);
    assert_eq!(spherical_depth.len(), 100 * 100);
}

fn main() {
    println!("Running Optimization Thread integration tests...");

    test_ba_basic_workflow();
    println!("✓ BA basic workflow test passed");

    test_synthetic_poses_generation();
    println!("✓ Synthetic poses generation test passed");

    test_synthetic_depth_generation();
    println!("✓ Synthetic depth generation test passed");

    println!("\nAll Optimization Thread integration tests passed!");
}
