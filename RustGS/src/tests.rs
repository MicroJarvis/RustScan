use super::*;
use tempfile::tempdir;

fn write_colmap_test_dataset(root: &Path) {
    let sparse = root.join("sparse").join("0");
    std::fs::create_dir_all(&sparse).unwrap();
    let images = root.join("images");
    std::fs::create_dir_all(&images).unwrap();

    std::fs::write(
        sparse.join("cameras.txt"),
        "# Camera list with one line of data per camera:\n1 PINHOLE 640 480 500 500 320 240\n",
    )
    .unwrap();
    std::fs::write(
        sparse.join("images.txt"),
        "# Image list with two lines of data per image:\n1 1.0 0.0 0.0 0.0 0.0 0.0 1.0 1 frame_0001.jpg\n",
    )
    .unwrap();
    std::fs::write(
        sparse.join("points3D.txt"),
        "# 3D point list with one line of data per point:\n1 0.0 0.0 1.0 128 128 128 0.1\n",
    )
    .unwrap();
    std::fs::write(images.join("frame_0001.jpg"), vec![0u8; 640 * 480 * 3]).unwrap();
}

#[test]
fn test_load_colmap_training_dataset_with_source_detects_colmap_directory() {
    let temp = tempdir().unwrap();
    write_colmap_test_dataset(temp.path());

    let (dataset, source) =
        load_colmap_training_dataset_with_source(temp.path(), &ColmapConfig::default()).unwrap();

    assert_eq!(source, TrainingInputKind::Colmap);
    assert_eq!(dataset.poses.len(), 1);
    assert_eq!(dataset.initial_points.len(), 1);
}

#[test]
fn test_load_colmap_training_dataset_rejects_non_colmap_directory() {
    let temp = tempdir().unwrap();
    std::fs::write(temp.path().join("frame.png"), []).unwrap();
    std::fs::write(
        temp.path().join("sparse_pc.ply"),
        "ply\n\
format ascii 1.0\n\
element vertex 1\n\
property float x\n\
property float y\n\
property float z\n\
property uchar red\n\
property uchar green\n\
property uchar blue\n\
end_header\n\
0.0 0.0 1.0 255 128 0\n",
    )
    .unwrap();
    std::fs::write(
        temp.path().join("transforms.json"),
        r#"{
            "fl_x": 500.0,
            "fl_y": 505.0,
            "cx": 320.0,
            "cy": 240.0,
            "w": 640,
            "h": 480,
            "frames": [
                {
                    "file_path": "frame.png",
                    "transform_matrix": [
                        [1.0, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0, 2.0],
                        [0.0, 0.0, 1.0, 3.0],
                        [0.0, 0.0, 0.0, 1.0]
                    ]
                }
            ]
        }"#,
    )
    .unwrap();

    let err = load_colmap_training_dataset_with_source(temp.path(), &ColmapConfig::default())
        .unwrap_err();

    assert!(
        err.to_string().contains("COLMAP"),
        "unexpected error: {err}"
    );
}
