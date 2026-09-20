//! End-to-end SLAM to Mesh Example
//!
//! This example demonstrates the complete pipeline from:
//! 1. Loading TUM RGB-D dataset
//! 2. Visual Odometry for pose estimation
//! 3. Gaussian mapping for 3D reconstruction
//! 4. Mesh extraction and export

use nalgebra::Point3;
use std::fs;
use std::path::PathBuf;

use rustslam::core::SE3;
use rustslam::fusion::{GaussianMapper, MeshExtractionConfig, MeshExtractor, TsdfConfig};
use rustslam::io::{Dataset, DatasetConfig, TumRgbdDataset};
use rustslam::tracker::{VOState, VisualOdometry};

/// Main entry point
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RustSLAM End-to-End Example ===\n");

    if !cfg!(feature = "image") {
        eprintln!("This example requires the `image` feature to load dataset images.");
        eprintln!("Please run with: cargo run --example e2e_slam_to_mesh --features image");
        return Ok(());
    }

    // Note: This example uses pure Rust feature extraction (Harris/FAST)
    // OpenCV is optional but not required

    // Configuration
    let dataset_path = PathBuf::from("../artifacts/inputs/tum/rgbd_dataset_freiburg1_xyz");
    let output_dir = PathBuf::from("../artifacts/runs/test_output_tum");
    let max_frames = 50; // Process at most 50 frames for demo
    let keyframe_interval = 3; // Add keyframe every 3 frames

    // Create output directory
    fs::create_dir_all(&output_dir)?;

    // Step 1: Load TUM RGB-D dataset
    println!("[1/6] Loading TUM RGB-D dataset...");
    let config = DatasetConfig {
        root_path: dataset_path.clone(),
        load_depth: true,
        load_ground_truth: true,
        max_frames,
        stride: 1,
    };

    let dataset = match TumRgbdDataset::load(config) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error loading dataset: {}", e);
            eprintln!("Please download TUM RGB-D dataset from:");
            eprintln!("  https://vision.in.tum.de/data/datasets/rgbd-dataset");
            eprintln!("  and extract to: {}", dataset_path.display());
            return Ok(());
        }
    };

    println!("  Loaded {} frames", dataset.len());
    println!("  Camera: {:?}", dataset.camera());

    // Step 2: Initialize Visual Odometry (with default settings)
    println!("\n[2/6] Initializing Visual Odometry...");
    let camera = dataset.camera();

    // Create VO with default settings - it uses internal feature extraction
    let mut vo = VisualOdometry::new(camera.clone());

    // Step 3: Initialize Gaussian Mapper
    println!("\n[3/6] Initializing Gaussian Mapper...");
    let width = camera.width as usize;
    let height = camera.height as usize;
    let mut mapper = GaussianMapper::new(width, height);

    let mut trajectory: Vec<SE3> = Vec::new();
    let mut keyframe_poses: Vec<SE3> = Vec::new();
    let mut keyframe_indices: Vec<usize> = Vec::new();

    // Step 4: Process frames
    println!("\n[4/6] Processing frames...");
    let mut frame_count = 0;

    for frame_result in dataset.frames() {
        let frame = match frame_result {
            Ok(f) => f,
            Err(e) => {
                eprintln!("  Warning: Failed to load frame: {}", e);
                continue;
            }
        };

        // Run VO (convert RGB to grayscale)
        let gray = rgb_to_grayscale(&frame.color, frame.width as usize, frame.height as usize);
        let vo_result = vo.process_frame(&gray, frame.width, frame.height);
        let pose = vo_result.pose;

        // Store trajectory
        trajectory.push(pose.clone());

        // Print progress
        if frame_count % 10 == 0 {
            let state_str = match vo.state() {
                VOState::NotInitialized => "NotInitialized",
                VOState::Initializing => "Initializing",
                VOState::TrackingOk => "Tracking OK",
                VOState::TrackingLost => "Tracking Lost",
            };
            println!(
                "  Frame {}/{}: {} (inliers: {}/{})",
                frame_count + 1,
                dataset.len(),
                state_str,
                vo_result.num_inliers,
                vo_result.num_matches
            );
        }

        // Add keyframes to mapper
        if vo.state() == VOState::TrackingOk && frame.index % keyframe_interval == 0 {
            if let Some(depth) = &frame.depth {
                // Convert pose to rotation matrix and translation vector
                let rot = pose.rotation();
                let t = pose.translation();

                // Convert color to RGB format
                let color: Vec<[u8; 3]> =
                    frame.color.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();

                // Update mapper
                let result = mapper.update(
                    depth,
                    &color,
                    width,
                    height,
                    camera.focal.x,
                    camera.focal.y,
                    camera.principal.x,
                    camera.principal.y,
                    &rot,
                    &t,
                );

                keyframe_poses.push(pose.clone());
                keyframe_indices.push(frame.index);

                if frame_count % 20 == 0 {
                    println!(
                        "    Keyframe added: {} Gaussians (total: {})",
                        result.added, result.total_gaussians
                    );
                }
            }
        }

        frame_count += 1;
    }

    println!("\n  Processed {} frames", frame_count);
    println!("  Added {} keyframes", keyframe_poses.len());
    println!("  Total Gaussians: {}", mapper.num_gaussians());

    // Step 5: Extract mesh
    println!("\n[5/6] Extracting mesh...");

    // Get camera intrinsics
    let fx = camera.focal.x;
    let fy = camera.focal.y;
    let cx = camera.principal.x;
    let cy = camera.principal.y;

    // Create mesh extractor
    let voxel_size = 0.01;
    let tsdf_config = TsdfConfig {
        voxel_size,
        sdf_trunc: voxel_size * 4.0,
        min_bound: Point3::new(-1.0, -1.0, -1.0),
        max_bound: Point3::new(1.0, 1.0, 1.0),
        max_weight: 100.0,
        integration_weight: 1.0,
    };
    let mesh_extraction_config = MeshExtractionConfig {
        tsdf_config,
        min_cluster_size: 100,
        num_largest_clusters: 1,
        smooth_normals: true,
        normal_smoothing_iterations: 3,
    };
    let mut mesh_extractor = MeshExtractor::new(mesh_extraction_config);

    // Integrate depth from keyframes
    let mut integrated_frames = 0;
    for (i, pose) in keyframe_poses.iter().enumerate() {
        if let Ok(frame) = dataset.get_frame(keyframe_indices[i]) {
            if let Some(depth) = &frame.depth {
                let pose_mat4 = pose.to_homogeneous_matrix();

                let color: Vec<[u8; 3]> =
                    frame.color.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
                mesh_extractor.integrate_frame(
                    depth,
                    Some(&color),
                    width,
                    height,
                    [fx, fy, cx, cy],
                    &pose_mat4,
                );
                integrated_frames += 1;
            }
        }
    }

    println!("  Integrated {} keyframes", integrated_frames);

    let mesh = mesh_extractor.extract_with_postprocessing();

    println!(
        "  Extracted mesh: {} vertices, {} triangles",
        mesh.vertices.len(),
        mesh.triangles.len()
    );

    // Step 6: Export mesh
    println!("\n[6/6] Exporting mesh...");

    let mesh_output = output_dir.join("mesh_output.obj");

    // Write mesh as OBJ
    let mut obj_content = String::new();

    // Write vertices (mesh.vertices is Vec<MeshVertex>)
    for v in &mesh.vertices {
        obj_content.push_str(&format!(
            "v {} {} {}\n",
            v.position.x, v.position.y, v.position.z
        ));
    }

    // Write triangles (OBJ is 1-indexed) (mesh.triangles is Vec<MeshTriangle>)
    for t in &mesh.triangles {
        obj_content.push_str(&format!(
            "f {} {} {}\n",
            t.indices[0] as u32 + 1,
            t.indices[1] as u32 + 1,
            t.indices[2] as u32 + 1
        ));
    }

    fs::write(&mesh_output, obj_content)?;
    println!("  Mesh saved to: {}", mesh_output.display());

    // Save trajectory
    let traj_path = output_dir.join("trajectory.txt");
    let mut traj_content = String::from("# timestamp tx ty tz qx qy qz qw\n".to_string());
    for (i, pose) in trajectory.iter().enumerate() {
        let t = pose.translation();
        let q = pose.quaternion();
        traj_content.push_str(&format!(
            "{}.000000 {} {} {} {} {} {} {}\n",
            i as f64, t[0], t[1], t[2], q[0], q[1], q[2], q[3]
        ));
    }
    fs::write(&traj_path, traj_content)?;
    println!("  Trajectory saved to: {}", traj_path.display());

    println!("\n=== Done! ===");
    Ok(())
}

fn rgb_to_grayscale(rgb: &[u8], width: usize, height: usize) -> Vec<u8> {
    let expected = width.saturating_mul(height).saturating_mul(3);
    if rgb.len() != expected {
        return Vec::new();
    }

    let mut gray = Vec::with_capacity(width * height);
    for c in rgb.chunks(3) {
        let r = c[0] as u16;
        let g = c[1] as u16;
        let b = c[2] as u16;
        let y = (30 * r + 59 * g + 11 * b) / 100;
        gray.push(y as u8);
    }
    gray
}
