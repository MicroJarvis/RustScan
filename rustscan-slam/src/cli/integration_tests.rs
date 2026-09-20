use super::*;
use std::path::PathBuf;
use std::time::Instant;
use tempfile::tempdir;

fn e2e_enabled() -> bool {
    match std::env::var("RUSTSCAN_E2E") {
        Ok(value) => matches!(value.as_str(), "1" | "true" | "yes"),
        Err(_) => false,
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => matches!(value.as_str(), "1" | "true" | "yes"),
        Err(_) => default,
    }
}

fn find_test_video() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let candidate = dir.join("test_data").join("video").join("sofa.MOV");
        if candidate.exists() {
            return Some(candidate);
        }
        if !dir.pop() {
            break;
        }
    }
    None
}

fn test_resolved_config(input: PathBuf) -> ResolvedConfig {
    ResolvedConfig {
        input,
        output: tempdir().unwrap().keep(),
        output_format: OutputFormat::Json,
        slam: SlamConfig::default(),
        video: VideoConfig {
            cache_capacity: 4,
            prefer_hardware: false,
        },
        mesh_voxel_size: None,
    }
}

fn test_slam_stage_output(with_depth: bool) -> SlamStageOutput {
    let keyframe = KeyframeSample {
        index: 0,
        timestamp: 0.0,
        width: 2,
        height: 1,
        color: vec![10, 20, 30, 40, 50, 60],
        depth: with_depth.then(|| vec![1.0, 2.0]),
        pose: SE3::identity(),
    };
    SlamStageOutput {
        keyframes: vec![keyframe],
        camera: Camera::new(525.0, 525.0, 0.5, 0.5, 2, 1),
        frame_count: 1,
        map_points: vec![rustscan_types::MapPointData::new(
            [0.0, 0.0, 1.0],
            Some([0.1, 0.2, 0.3]),
        )],
    }
}

#[test]
fn test_load_input_source_detects_kitti_dataset() {
    let dir = tempdir().unwrap();
    let image_dir = dir.path().join("image_0");
    std::fs::create_dir_all(&image_dir).unwrap();
    std::fs::write(
        dir.path().join("calib.txt"),
        "P0: 718.856 0.0 607.1928 0.0 0.0 718.856 185.2157 0.0 0.0 0.0 1.0 0.0\n",
    )
    .unwrap();
    std::fs::write(image_dir.join("000000.png"), b"dummy").unwrap();

    let mut resolved = test_resolved_config(dir.path().to_path_buf());
    resolved.slam.dataset.dataset_type = "kitti".to_string();

    let input = load_input_source(&resolved).unwrap();
    assert!(matches!(input, InputSource::Dataset(_)));
}

#[test]
fn test_load_input_source_detects_euroc_dataset() {
    let dir = tempdir().unwrap();
    let cam0 = dir.path().join("mav0").join("cam0");
    let data_dir = cam0.join("data");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::write(
        cam0.join("data.csv"),
        "#timestamp,filename\n1403636579763555584,1403636579763555584.png\n",
    )
    .unwrap();
    std::fs::write(data_dir.join("1403636579763555584.png"), b"dummy").unwrap();

    let mut resolved = test_resolved_config(dir.path().to_path_buf());
    resolved.slam.dataset.dataset_type = "euroc".to_string();

    let input = load_input_source(&resolved).unwrap();
    assert!(matches!(input, InputSource::Dataset(_)));
}

#[test]
fn bytes_to_mib_uses_mebibyte_units() {
    assert!((bytes_to_mib(32 * 1024 * 1024) - 32.0).abs() < f64::EPSILON);
}

#[test]
fn export_slam_output_for_rustgs_writes_json_export() {
    let output_dir = tempdir().unwrap();
    let slam = test_slam_stage_output(false);

    let path = export_slam_output_for_rustgs(&slam, output_dir.path()).unwrap();
    assert_eq!(path, output_dir.path().join("slam_output.json"));
    assert!(path.exists());

    let loaded = rustscan_types::SlamOutput::load(&path).unwrap();
    assert_eq!(loaded.num_poses(), 1);
    assert_eq!(loaded.num_points(), 1);
    assert_eq!(loaded.poses[0].frame_id, 0);
    assert!(loaded.poses[0].depth_path.is_none());
}

#[test]
fn save_slam_checkpoint_clears_downstream_stage_markers() {
    let output_dir = tempdir().unwrap();
    let slam = test_slam_stage_output(true);
    let existing = PipelineCheckpoint {
        gaussian_completed: true,
        mesh_completed: true,
        ..PipelineCheckpoint::default()
    };

    let saved = save_slam_checkpoint(output_dir.path(), &slam, Some(existing)).unwrap();

    assert!(saved.video_completed);
    assert!(saved.slam_completed);
    assert!(!saved.gaussian_completed);
    assert!(!saved.mesh_completed);
    assert!(saved.slam.as_ref().is_some());
    assert!(saved
        .slam
        .as_ref()
        .and_then(|slam| slam.keyframes.first())
        .and_then(|keyframe| keyframe.depth_path.as_ref())
        .is_some());
    assert_eq!(saved.slam.as_ref().unwrap().map_points.len(), 1);
}

#[test]
fn reset_post_slam_checkpoint_state_keeps_sparse_points() {
    let mut checkpoint = PipelineCheckpoint {
        gaussian_completed: true,
        mesh_completed: true,
        slam: Some(SlamCheckpoint {
            camera: CameraCheckpoint {
                fx: 525.0,
                fy: 525.0,
                cx: 0.5,
                cy: 0.5,
                width: 2,
                height: 1,
            },
            frame_count: 1,
            keyframes: Vec::new(),
            map_points: vec![MapPointCheckpoint {
                position: [0.0, 0.0, 1.0],
                color: Some([0.1, 0.2, 0.3]),
            }],
        }),
        ..PipelineCheckpoint::default()
    };

    reset_post_slam_checkpoint_state(&mut checkpoint);

    assert!(!checkpoint.gaussian_completed);
    assert!(!checkpoint.mesh_completed);
    assert_eq!(checkpoint.slam.as_ref().unwrap().map_points.len(), 1);
}

#[test]
fn test_end_to_end_pipeline_video_exports_slam_only_artifacts() {
    if !e2e_enabled() {
        eprintln!("Skipping E2E pipeline test (set RUSTSCAN_E2E=1 to enable)");
        return;
    }

    let video_path = match find_test_video() {
        Some(path) => path,
        None => {
            eprintln!("Skipping E2E pipeline test (missing artifacts/inputs/video/sofa.MOV)");
            return;
        }
    };

    let output_dir = tempdir().expect("create temp output dir");

    let mut slam = SlamConfig::default();
    slam.dataset.max_frames = env_usize("RUSTSCAN_E2E_MAX_FRAMES", 0);
    slam.dataset.stride = env_usize("RUSTSCAN_E2E_STRIDE", 1).max(1);
    slam.mapper.max_keyframes =
        env_usize("RUSTSCAN_E2E_MAX_KEYFRAMES", slam.mapper.max_keyframes).max(1);
    slam.mapper.keyframe_interval = env_usize(
        "RUSTSCAN_E2E_KEYFRAME_INTERVAL",
        slam.mapper.keyframe_interval,
    )
    .max(1);

    let resolved = ResolvedConfig {
        input: video_path,
        output: output_dir.path().to_path_buf(),
        output_format: OutputFormat::Json,
        slam,
        video: VideoConfig {
            cache_capacity: env_usize("RUSTSCAN_E2E_CACHE_CAPACITY", 50).max(1),
            prefer_hardware: env_bool("RUSTSCAN_E2E_PREFER_HW", false),
        },
        mesh_voxel_size: None,
    };

    let start = Instant::now();
    let report = run_pipeline(&resolved).expect("run SLAM-only pipeline");

    assert!(
        report.camera_count > 0,
        "expected at least one exported pose"
    );
    assert_eq!(
        report.slam_output_path,
        resolved.output.join("slam_output.json")
    );
    assert!(report.slam_output_path.exists());

    let loaded =
        rustscan_types::SlamOutput::load(&report.slam_output_path).expect("load slam export");
    assert_eq!(loaded.num_poses(), report.camera_count);
    assert!(loaded.num_poses() > 0);

    assert!(
        !resolved.output.join("mesh.obj").exists(),
        "mesh.obj should not be produced by RustSLAM anymore"
    );
    assert!(
        !resolved.output.join("mesh.ply").exists(),
        "mesh.ply should not be produced by RustSLAM anymore"
    );
    assert!(
        !resolved.output.join("checkpoints").join("rustgs").exists(),
        "RustGS training artifacts should not be created by RustSLAM"
    );

    let max_seconds = env_u64("RUSTSCAN_E2E_MAX_SECONDS", 30 * 60);
    assert!(
        start.elapsed().as_secs() <= max_seconds,
        "Processing time {}s exceeds threshold {}s",
        start.elapsed().as_secs(),
        max_seconds
    );
}
