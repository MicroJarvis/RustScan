//! Example: Loading TUM RGB-D dataset
//!
//! This example demonstrates how to use the dataset loader to read
//! TUM RGB-D dataset frames.
//!
//! Usage:
//!   cargo run --example load_tum_dataset --features "image" -- /path/to/tum/dataset
//!
//! Example dataset: https://vision.in.tum.de/data/datasets/rgbd-dataset/download

use std::env;
use std::path::PathBuf;

use rustslam::io::{Dataset, DatasetConfig, TumRgbdDataset};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RustSLAM TUM RGB-D Dataset Loader ===\n");

    // Parse command line argument
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} /path/to/tum/dataset", args[0]);
        eprintln!("Example: {} /home/user/rgbd_dataset_freiburg1_xyz", args[0]);
        return Ok(());
    }

    let dataset_path = PathBuf::from(&args[1]);

    if !dataset_path.exists() {
        eprintln!(
            "Error: Dataset path '{}' does not exist",
            dataset_path.display()
        );
        return Ok(());
    }

    // Configure dataset loading
    let config = DatasetConfig {
        root_path: dataset_path.clone(),
        load_depth: true,
        load_ground_truth: true,
        max_frames: 100, // Limit to first 100 frames for demonstration
        stride: 5,       // Load every 5th frame
    };

    println!("Loading dataset from: {}", dataset_path.display());
    println!(
        "Configuration: load_depth={}, load_ground_truth={}, max_frames={}, stride={}",
        config.load_depth, config.load_ground_truth, config.max_frames, config.stride
    );

    // Load the dataset
    let dataset = TumRgbdDataset::load(config)?;

    // Print metadata
    let metadata = dataset.metadata();
    println!("\nDataset Metadata:");
    println!("  Name: {}", metadata.name);
    println!("  Sequence: {}", metadata.sequence);
    println!("  Total frames: {}", metadata.total_frames);
    println!("  Has depth: {}", metadata.has_depth);
    println!("  Has ground truth: {}", metadata.has_ground_truth);
    println!("  Frame rate: {:?} Hz", metadata.frame_rate);
    println!("  Notes: {}", metadata.notes);

    // Print camera information
    let camera = dataset.camera();
    println!("\nCamera Parameters:");
    println!("  Resolution: {}x{}", camera.width, camera.height);
    println!("  Focal length: ({}, {})", camera.focal.x, camera.focal.y);
    println!(
        "  Principal point: ({}, {})",
        camera.principal.x, camera.principal.y
    );
    println!("  Distortion: {:?}", camera.distortion);

    // Iterate through frames
    println!("\nIterating through frames (first 5):");
    let mut frame_count = 0;
    for (i, frame_result) in dataset.frames().enumerate() {
        if i >= 5 {
            break;
        }

        match frame_result {
            Ok(frame) => {
                println!("  Frame {}:", i);
                println!("    Index: {}", frame.index);
                println!("    Timestamp: {:.3} s", frame.timestamp);
                println!("    Dimensions: {}x{}", frame.width, frame.height);
                println!("    Color data: {} bytes", frame.color.len());
                println!(
                    "    Depth data: {}",
                    if frame.has_depth() {
                        format!("{} floats", frame.depth.as_ref().unwrap().len())
                    } else {
                        "None".to_string()
                    }
                );
                println!(
                    "    Ground truth pose: {}",
                    if frame.has_ground_truth() {
                        "Available"
                    } else {
                        "None"
                    }
                );

                // Example: Print first few depth values if available
                if let Some(depth) = &frame.depth {
                    if depth.len() >= 5 {
                        println!("    Depth samples: {:.3?} m", &depth[0..5]);
                    }
                }

                frame_count += 1;
            }
            Err(e) => {
                eprintln!("  Error loading frame {}: {}", i, e);
            }
        }
    }

    println!("\nSuccessfully loaded {} frames.", frame_count);
    println!("\n=== Dataset Loading Complete ===");

    // Example of using a specific frame
    if dataset.len() > 0 {
        println!("\nAccessing specific frames:");

        // Get first frame
        match dataset.get_frame(0) {
            Ok(frame) => {
                println!("  First frame: {}x{} image", frame.width, frame.height);
                if let Some(pose) = frame.ground_truth_pose {
                    let t = pose.translation();
                    println!(
                        "  Ground truth pose: translation=({:.2}, {:.2}, {:.2})",
                        t[0], t[1], t[2]
                    );
                }
            }
            Err(e) => eprintln!("  Error getting frame 0: {}", e),
        }

        // Get middle frame
        let middle_idx = dataset.len() / 2;
        if middle_idx > 0 {
            match dataset.get_frame(middle_idx) {
                Ok(frame) => {
                    println!(
                        "  Middle frame ({}): timestamp={:.3}s",
                        middle_idx, frame.timestamp
                    );
                }
                Err(e) => eprintln!("  Error getting frame {}: {}", middle_idx, e),
            }
        }
    }

    println!("\nDataset ready for SLAM processing!");
    println!("Next steps:");
    println!("  1. Pass frames to VisualOdometry for pose estimation");
    println!("  2. Use SparseDenseSlam for mapping");
    println!("  3. Train 3D Gaussians with the poses");
    println!("  4. Extract mesh with MeshExtractor");

    Ok(())
}
