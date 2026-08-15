//! Integration tests for RustViewer loaders with actual test data.

use image::{Rgb, RgbImage};
use rust_viewer::loader::{checkpoint, load_colmap_training_dataset, mesh};
use rust_viewer::renderer::scene::Scene;
use rustgs::ColmapConfig;
use rustsfm::colmap::export_colmap;
use rustsfm::types::{CameraModel, Point3D, Reconstruction, TrackObservation};
use rustslam::SE3;
use std::path::Path;

#[test]
fn loads_rustsfm_export_with_one_shared_camera() {
    let temp = tempfile::tempdir().unwrap();
    let source_images = temp.path().join("source-images");
    std::fs::create_dir_all(&source_images).unwrap();
    let first = source_images.join("first.png");
    let second = source_images.join("second.png");
    write_tiny_png(&first, [255, 0, 0]);
    write_tiny_png(&second, [0, 255, 0]);

    let camera = CameraModel::new_pinhole(2, 2, 2.0, 2.0, 1.0, 1.0);
    let reconstruction = Reconstruction {
        camera,
        cameras: vec![camera],
        camera_ids: vec![1],
        rigs: Vec::new(),
        frames: Vec::new(),
        image_names: vec!["first.png".to_string(), "second.png".to_string()],
        image_paths: vec![first, second],
        image_ids: vec![1, 2],
        image_camera_indices: vec![0, 0],
        image_frame_indices: vec![None, None],
        poses: vec![
            Some(SE3::identity()),
            Some(SE3::new(&[0.0, 0.0, 0.0, 1.0], &[1.0, 0.0, 0.0])),
        ],
        observations: vec![vec![Some(0)], vec![Some(0)]],
        keypoints: vec![
            vec![rustslam::KeyPoint::new(1.0, 1.0)],
            vec![rustslam::KeyPoint::new(1.0, 1.0)],
        ],
        point_ids: vec![1],
        points: vec![Point3D {
            xyz: [0.0, 0.0, 2.0],
            color: [255, 255, 255],
            error: 0.0,
            track: vec![
                TrackObservation {
                    image: 0,
                    feature: 0,
                },
                TrackObservation {
                    image: 1,
                    feature: 0,
                },
            ],
        }],
    };
    let output = temp.path().join("rustsfm-output");
    export_colmap(&output, &reconstruction, true).unwrap();

    let loaded = load_colmap_training_dataset(&output, &ColmapConfig::default())
        .expect("RustViewer should accept RustSFM's shared-camera COLMAP export");

    assert_eq!(loaded.summary.frame_count, 2);
    assert_eq!(loaded.summary.sparse_point_count, 1);
    assert_eq!(loaded.summary.intrinsics.width, 2);
    assert_eq!(loaded.summary.intrinsics.height, 2);
}

fn write_tiny_png(path: &Path, color: [u8; 3]) {
    let image = RgbImage::from_pixel(2, 2, Rgb(color));
    image.save(path).unwrap();
}

#[test]
#[ignore = "requires checkpoint file at ../RustSLAM/output/checkpoints/pipeline.json"]
fn test_load_checkpoint_from_output() {
    let checkpoint_path = Path::new("../RustSLAM/output/checkpoints/pipeline.json");
    if !checkpoint_path.exists() {
        eprintln!(
            "Skipping test: checkpoint file not found at {:?}",
            checkpoint_path
        );
        return;
    }

    let mut scene = Scene::default();
    let result = checkpoint::load_checkpoint(checkpoint_path, &mut scene);

    assert!(
        result.is_ok(),
        "Failed to load checkpoint: {:?}",
        result.err()
    );

    // Check that we loaded some data
    println!("Loaded {} keyframes", scene.trajectory.len());
    println!("Loaded {} map points", scene.map_points.len());

    assert!(scene.trajectory.len() > 0, "Should have loaded keyframes");
}

#[test]
#[ignore = "requires test file at ../test_data/middle/cube.obj"]
fn test_load_obj_cube() {
    let obj_path = Path::new("../test_data/middle/cube.obj");
    if !obj_path.exists() {
        eprintln!("Skipping test: cube.obj not found at {:?}", obj_path);
        return;
    }

    let mut scene = Scene::default();
    let result = mesh::load_mesh(obj_path, &mut scene);

    assert!(
        result.is_ok(),
        "Failed to load cube.obj: {:?}",
        result.err()
    );

    println!("Loaded {} vertices", scene.mesh_vertices.len());
    println!("Loaded {} indices", scene.mesh_indices.len());
    println!("Loaded {} edge indices", scene.mesh_edge_indices.len());

    assert!(scene.mesh_vertices.len() > 0, "Should have loaded vertices");
    assert!(scene.mesh_indices.len() > 0, "Should have loaded faces");
}

#[test]
#[ignore = "requires test file at ../test_data/large/FinalBaseMesh.obj"]
fn test_load_obj_finalbasemesh() {
    let obj_path = Path::new("../test_data/large/FinalBaseMesh.obj");
    if !obj_path.exists() {
        eprintln!(
            "Skipping test: FinalBaseMesh.obj not found at {:?}",
            obj_path
        );
        return;
    }

    let mut scene = Scene::default();
    let result = mesh::load_mesh(obj_path, &mut scene);

    assert!(
        result.is_ok(),
        "Failed to load FinalBaseMesh.obj: {:?}",
        result.err()
    );

    println!("Loaded {} vertices", scene.mesh_vertices.len());
    println!("Loaded {} indices", scene.mesh_indices.len());
    println!("Loaded {} edge indices", scene.mesh_edge_indices.len());

    assert!(scene.mesh_vertices.len() > 0, "Should have loaded vertices");
    assert!(scene.mesh_indices.len() > 0, "Should have loaded faces");

    // FinalBaseMesh.obj has 24461 vertices and 48918 faces
    assert!(
        scene.mesh_vertices.len() >= 24000,
        "Should have ~24k vertices"
    );
}
