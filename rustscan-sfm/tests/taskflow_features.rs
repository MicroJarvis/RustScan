use anyhow::Result;
use rustscan_sfm::colmap::ColmapCamera;
use rustscan_sfm::database::{ColmapDatabase, ColmapDatabaseCamera, ColmapDatabaseImage};
use std::process::Command;

#[test]
fn release_cli_image_dag_matches_default_stage_and_rejects_invalid_budget() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let images = directory.path().join("images");
    std::fs::create_dir(&images)?;
    image::GrayImage::from_fn(256, 256, |x, y| {
        image::Luma([if (x / 16 + y / 16) % 2 == 0 { 240 } else { 20 }])
    })
    .save(images.join("image.png"))?;
    let database = directory.path().join("database.db");
    let db = ColmapDatabase::open(&database)?;
    db.write_camera(
        &ColmapDatabaseCamera {
            camera: ColmapCamera {
                camera_id: 1,
                model_id: rustscan_sfm::types::COLMAP_PINHOLE,
                width: 256,
                height: 256,
                params: vec![200.0, 200.0, 128.0, 128.0],
            },
            has_prior_focal_length: true,
        },
        true,
    )?;
    db.write_image(
        &ColmapDatabaseImage {
            image_id: 1,
            name: "image.png".into(),
            camera_id: 1,
            frame_id: None,
        },
        true,
    )?;
    let command = || {
        let mut command = Command::new(env!("CARGO_BIN_EXE_rustsfm"));
        command
            .arg("extract-features")
            .arg("--database")
            .arg(&database)
            .arg("--images")
            .arg(&images)
            .args(["--max-features", "256"]);
        command
    };
    let default_stage = command().output()?;
    assert!(
        default_stage.status.success(),
        "{}",
        String::from_utf8_lossy(&default_stage.stderr)
    );
    let keypoints = db.read_keypoints(1)?;
    let descriptors = db.read_descriptors(1)?.data;
    assert!(!keypoints.is_empty());
    let report_path = directory.path().join("taskflow.json");
    let scheduled = command()
        .args([
            "--taskflow",
            "--taskflow-cpu-threads",
            "1",
            "--taskflow-memory-mib",
            "64",
            "--taskflow-image-memory-mib",
            "64",
        ])
        .arg("--output-json")
        .arg(&report_path)
        .output()?;
    assert!(
        scheduled.status.success(),
        "{}",
        String::from_utf8_lossy(&scheduled.stderr)
    );
    assert_eq!(db.read_keypoints(1)?, keypoints);
    assert_eq!(db.read_descriptors(1)?.data, descriptors);
    let report: serde_json::Value = serde_json::from_slice(&std::fs::read(report_path)?)?;
    assert_eq!(report["image_count"], 1);
    assert_eq!(report["total_keypoints"], keypoints.len());

    let invalid = command()
        .args(["--taskflow", "--taskflow-image-memory-mib", "0"])
        .output()?;
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("must be positive"));
    assert_eq!(db.read_keypoints(1)?, keypoints);
    assert_eq!(db.read_descriptors(1)?.data, descriptors);
    Ok(())
}
