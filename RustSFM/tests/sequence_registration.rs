use rustsfm::colmap::{
    export_colmap, read_colmap_sparse_files, read_colmap_sparse_model, write_colmap_sparse_binary,
    ColmapCamera,
};
use rustsfm::database::{
    ColmapDatabase, ColmapDatabaseCamera, ColmapDatabaseImage, ColmapDescriptors, ColmapKeypoint,
    COLMAP_FEATURE_SIFT,
};
use rustsfm::types::{CameraModel, Point3D, Reconstruction, TrackObservation, COLMAP_PINHOLE};
use rustsfm::{
    register_remaining_sequence_frames, require_complete_pose_coverage,
    run_keyframe_reconstruction, run_sequence_registration, FrameRegistrationDiagnostic,
    FrameRegistrationStatus, KeyframeReconstructionResult, MapperConfig, RegistrationRound,
    SequenceFrame, SequenceRegistrationConfig, SequenceRegistrationError, SequenceRegistrationPlan,
    SequenceRegistrationResult, SfmTaskContext, SfmTaskControl, MAX_DYNAMIC_SUPPORT_CANDIDATES,
    MAX_SEQUENCE_NEIGHBORS, MAX_SEQUENCE_PLAN_FRAMES, MAX_TIMESTAMP_PLATEAU,
    MAX_TOTAL_SUPPORT_ENTRIES,
};
use serde::{de::DeserializeOwned, Serialize};
use std::fmt::Debug;
use std::path::PathBuf;
use tempfile::tempdir;

const SYNTHETIC_KEYFRAME_INDICES: [usize; 4] = [0, 2, 4, 5];

fn synthetic_sequence_fixture(
    blank_frame: Option<usize>,
) -> anyhow::Result<(
    tempfile::TempDir,
    PathBuf,
    Vec<SequenceFrame>,
    KeyframeReconstructionResult,
    MapperConfig,
)> {
    let temp = tempdir()?;
    let source = temp.path().join("source");
    let output = temp.path().join("output");
    std::fs::create_dir_all(&source)?;
    std::fs::create_dir_all(output.join("Cache"))?;
    let frame_ids = [101, 202, 303, 404, 505, 606];
    let frames = frame_ids
        .iter()
        .enumerate()
        .map(|(index, &id)| {
            let path = source.join(format!("frame-{index:04}.png"));
            image::GrayImage::new(320, 240).save(&path).unwrap();
            SequenceFrame {
                id,
                image_path: path,
                timestamp_us: Some(index as i64 * 1_000),
            }
        })
        .collect::<Vec<_>>();
    let camera = CameraModel::new_pinhole(320, 240, 220.0, 220.0, 160.0, 120.0);
    let poses = (0..6)
        .map(|index| {
            rustslam::SE3::from_quat_translation(
                glam::Quat::from_rotation_y((index as f32 - 2.5) * 0.012),
                glam::Vec3::new(index as f32 * -0.11, (index % 2) as f32 * 0.01, 0.0),
            )
        })
        .collect::<Vec<_>>();
    let points = (0..64)
        .map(|index| {
            let column = (index % 8) as f32;
            let row = (index / 8) as f32;
            [
                -0.75 + column * 0.21,
                -0.55 + row * 0.16,
                3.0 + (index % 7) as f32 * 0.11,
            ]
        })
        .collect::<Vec<_>>();
    let projected = poses
        .iter()
        .map(|pose| {
            points
                .iter()
                .map(|point| {
                    let camera_point = pose.transform_point(point);
                    ColmapKeypoint::new(
                        camera.fx * camera_point[0] / camera_point[2] + camera.cx,
                        camera.fy * camera_point[1] / camera_point[2] + camera.cy,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let database_path = output.join("Cache/database.db");
    let database = ColmapDatabase::open(&database_path)?;
    database.write_camera(
        &ColmapDatabaseCamera {
            camera: ColmapCamera {
                camera_id: 1,
                model_id: COLMAP_PINHOLE,
                width: 320,
                height: 240,
                params: vec![220.0, 220.0, 160.0, 120.0],
            },
            has_prior_focal_length: true,
        },
        true,
    )?;
    let descriptor_rows = (0..points.len())
        .map(|feature| {
            let mut descriptor = [0u8; 128];
            for (offset, value) in descriptor.iter_mut().enumerate() {
                *value = ((feature * 37 + offset * 17 + feature * offset * 3) % 251) as u8;
            }
            descriptor
        })
        .collect::<Vec<_>>();
    for (index, frame) in frames.iter().enumerate() {
        database.write_image(
            &ColmapDatabaseImage {
                image_id: index as u32 + 1,
                name: frame
                    .image_path
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_owned(),
                camera_id: 1,
                frame_id: None,
            },
            true,
        )?;
        let keypoints = if blank_frame == Some(index) {
            Vec::new()
        } else {
            projected[index].clone()
        };
        database.write_keypoints(index as u32 + 1, &keypoints)?;
        let descriptor_data = if blank_frame == Some(index) {
            Vec::new()
        } else {
            descriptor_rows
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect()
        };
        database.write_descriptors(
            index as u32 + 1,
            &ColmapDescriptors::new(COLMAP_FEATURE_SIFT, keypoints.len(), 128, descriptor_data)?,
        )?;
    }
    drop(database);

    let keyframe_keypoints = SYNTHETIC_KEYFRAME_INDICES
        .iter()
        .map(|&index| {
            projected[index]
                .iter()
                .map(|keypoint| rustslam::KeyPoint::new(keypoint.x, keypoint.y))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut observations = vec![vec![None; points.len()]; SYNTHETIC_KEYFRAME_INDICES.len()];
    for image in 0..observations.len() {
        for point in 0..points.len() {
            observations[image][point] = Some(point);
        }
    }
    let sparse_points = points
        .iter()
        .enumerate()
        .map(|(point, &xyz)| Point3D {
            xyz,
            color: [point as u8, 10, 20],
            error: 0.0,
            track: (0..SYNTHETIC_KEYFRAME_INDICES.len())
                .map(|image| TrackObservation {
                    image,
                    feature: point,
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    let reconstruction = Reconstruction {
        camera,
        cameras: vec![camera],
        camera_ids: vec![1],
        rigs: Vec::new(),
        frames: Vec::new(),
        image_names: SYNTHETIC_KEYFRAME_INDICES
            .iter()
            .map(|&index| {
                frames[index]
                    .image_path
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_owned()
            })
            .collect(),
        image_paths: SYNTHETIC_KEYFRAME_INDICES
            .iter()
            .map(|&index| frames[index].image_path.clone())
            .collect(),
        image_ids: SYNTHETIC_KEYFRAME_INDICES
            .iter()
            .map(|&index| index as u32 + 1)
            .collect(),
        image_camera_indices: vec![0; SYNTHETIC_KEYFRAME_INDICES.len()],
        image_frame_indices: vec![None; SYNTHETIC_KEYFRAME_INDICES.len()],
        poses: SYNTHETIC_KEYFRAME_INDICES
            .iter()
            .map(|&index| Some(poses[index]))
            .collect(),
        observations,
        keypoints: keyframe_keypoints,
        point_ids: (0..points.len())
            .map(|index| index as u64 + 1_000)
            .collect(),
        points: sparse_points,
    };
    export_colmap(&output, &reconstruction, false)?;
    let sparse_model = output.join("sparse/0");
    let sparse_files = read_colmap_sparse_files(&sparse_model)?;
    write_colmap_sparse_binary(&sparse_model, &sparse_files)?;

    let keyframe_result = KeyframeReconstructionResult {
        imported_frames: frames.len(),
        keyframe_ids: SYNTHETIC_KEYFRAME_INDICES
            .iter()
            .map(|&index| frames[index].id)
            .collect(),
        registered_keyframes: SYNTHETIC_KEYFRAME_INDICES.len(),
        database: database_path,
        sparse_model,
    };
    let mut mapper_config = MapperConfig {
        fx: Some(220.0),
        fy: Some(220.0),
        cx: Some(160.0),
        cy: Some(120.0),
        min_matches: 8,
        min_inliers: 8,
        min_triangulated: 4,
        essential_threshold_px: 2.0,
        essential_iterations: 2_000,
        pnp_threshold_px: 2.0,
        pnp_iterations: 5_000,
        abs_pose_min_num_inliers: 8,
        abs_pose_min_inlier_ratio: 0.2,
        random_seed: 0,
        local_ba: false,
        global_ba: false,
        extract_colors: false,
        ..MapperConfig::default()
    };
    mapper_config.sift_matching.cpu_brute_force_matcher = true;
    mapper_config.sift_matching.use_gpu = false;
    Ok((temp, output, frames, keyframe_result, mapper_config))
}

fn synthetic_sequence_config() -> SequenceRegistrationConfig {
    SequenceRegistrationConfig {
        narrow_neighbors_each_side: 2,
        wide_neighbors_each_side: 4,
        min_inliers: 16,
        min_inlier_ratio: 0.5,
        max_reprojection_error: 2.0,
        use_gpu_pnp: false,
    }
}

fn assert_json_round_trip<T>(value: &T)
where
    T: Debug + DeserializeOwned + PartialEq + Serialize,
{
    let json = serde_json::to_string(value).unwrap();
    assert_eq!(&serde_json::from_str::<T>(&json).unwrap(), value);
}

#[test]
fn keyframe_reconstruction_result_round_trips_through_json() {
    let result = KeyframeReconstructionResult {
        imported_frames: 6,
        keyframe_ids: vec![101, 700, 42, u32::MAX],
        registered_keyframes: 4,
        database: PathBuf::from("output/Cache/database.db"),
        sparse_model: PathBuf::from("output/sparse/0"),
    };

    assert_json_round_trip(&result);
}

#[test]
fn task6_stage_api_is_public_and_uses_u32_keyframe_ids() {
    let _ = run_keyframe_reconstruction;
    let _ = register_remaining_sequence_frames;
    let _ = run_sequence_registration;
}

#[test]
fn strict_pose_coverage_reports_unresolved_frames() {
    let result = SequenceRegistrationResult {
        imported_frames: 2,
        registered_frames: 1,
        frame_ids: vec![101, 9001],
        diagnostics: vec![
            FrameRegistrationDiagnostic::new(101, FrameRegistrationStatus::Keyframe),
            FrameRegistrationDiagnostic::new(9001, FrameRegistrationStatus::Unresolved),
        ],
        sparse_model: PathBuf::from("sparse/0"),
    };

    let error = require_complete_pose_coverage(&result).unwrap_err();
    assert_eq!(error.to_string(), "1 frames could not be registered");
}

#[test]
fn keyframe_stage_rejects_duplicate_arbitrary_frame_ids_before_io() {
    let frames = vec![
        SequenceFrame {
            id: 77,
            image_path: PathBuf::from("missing-a.jpg"),
            timestamp_us: Some(0),
        },
        SequenceFrame {
            id: 77,
            image_path: PathBuf::from("missing-b.jpg"),
            timestamp_us: Some(1),
        },
    ];
    let output = tempdir().unwrap();
    let control = SfmTaskControl::new();
    let mut sink = |_| {};
    let mut task = SfmTaskContext::new(&control, &mut sink);

    let error = run_keyframe_reconstruction(
        &frames,
        &[77],
        &MapperConfig::default(),
        output.path(),
        &mut task,
    )
    .unwrap_err();

    assert!(error.to_string().contains("duplicate frame id 77"));
    assert!(!output.path().join("Cache").exists());
}

#[test]
fn keyframe_stage_rejects_unknown_u32_keyframe_id_before_io() {
    let frames = vec![SequenceFrame {
        id: u32::MAX,
        image_path: PathBuf::from("missing.jpg"),
        timestamp_us: Some(0),
    }];
    let output = tempdir().unwrap();
    let control = SfmTaskControl::new();
    let mut sink = |_| {};
    let mut task = SfmTaskContext::new(&control, &mut sink);

    let error = run_keyframe_reconstruction(
        &frames,
        &[42],
        &MapperConfig::default(),
        output.path(),
        &mut task,
    )
    .unwrap_err();

    assert!(error.to_string().contains("unknown keyframe id 42"));
    assert!(!output.path().join("Cache").exists());
}

#[test]
fn keyframe_stage_prepares_fixed_database_and_stable_keyframe_links() {
    let input = tempdir().unwrap();
    let output = tempdir().unwrap();
    let first = input.path().join("capture-A.png");
    let second = input.path().join("capture-Z.png");
    image::GrayImage::new(64, 64).save(&first).unwrap();
    image::GrayImage::new(64, 64).save(&second).unwrap();
    let frames = vec![
        SequenceFrame {
            id: 9001,
            image_path: first,
            timestamp_us: Some(0),
        },
        SequenceFrame {
            id: 42,
            image_path: second,
            timestamp_us: Some(1),
        },
    ];
    let control = SfmTaskControl::new();
    let mut events = Vec::new();
    let mut sink = |event| events.push(event);
    let mut task = SfmTaskContext::new(&control, &mut sink);

    let error = run_keyframe_reconstruction(
        &frames,
        &[9001, 42],
        &MapperConfig::default(),
        output.path(),
        &mut task,
    )
    .unwrap_err();

    assert!(!error.to_string().contains("not implemented"));
    assert!(output.path().join("Cache/database.db").is_file());
    assert!(output
        .path()
        .join("Cache/keyframes/capture-A.png")
        .is_file());
    assert!(output
        .path()
        .join("Cache/keyframes/capture-Z.png")
        .is_file());
}

#[test]
fn remaining_stage_rejects_mismatched_keyframe_artifacts_before_io() {
    let frames = vec![
        SequenceFrame {
            id: 101,
            image_path: PathBuf::from("missing-a.jpg"),
            timestamp_us: Some(0),
        },
        SequenceFrame {
            id: 9001,
            image_path: PathBuf::from("missing-b.jpg"),
            timestamp_us: Some(1),
        },
    ];
    let keyframe_result = KeyframeReconstructionResult {
        imported_frames: 2,
        keyframe_ids: vec![101],
        registered_keyframes: 1,
        database: PathBuf::from("missing.db"),
        sparse_model: PathBuf::from("missing-sparse"),
    };
    let output = tempdir().unwrap();
    let control = SfmTaskControl::new();
    let mut sink = |_| {};
    let mut task = SfmTaskContext::new(&control, &mut sink);
    let config = SequenceRegistrationConfig {
        use_gpu_pnp: false,
        ..Default::default()
    };

    let error = register_remaining_sequence_frames(
        &frames,
        &[9001],
        &keyframe_result,
        &MapperConfig::default(),
        &config,
        output.path(),
        &mut task,
    )
    .unwrap_err();

    assert!(error
        .to_string()
        .contains("keyframe artifacts do not match"));
    assert!(!output.path().join("registration.json").exists());
}

#[test]
fn complete_sequence_registers_all_six_arbitrary_frame_ids_on_cpu() -> anyhow::Result<()> {
    let (_temp, output, frames, keyframes, mapper_config) = synthetic_sequence_fixture(None)?;
    let control = SfmTaskControl::new();
    let mut events = Vec::new();
    let mut sink = |event| events.push(event);
    let mut task = SfmTaskContext::new(&control, &mut sink);

    let result = register_remaining_sequence_frames(
        &frames,
        &keyframes.keyframe_ids,
        &keyframes,
        &mapper_config,
        &synthetic_sequence_config(),
        &output,
        &mut task,
    )?;

    assert!(result.has_complete_coverage(), "{:#?}", result.diagnostics);
    assert_eq!(result.registered_frames, 6);
    assert_eq!(result.frame_ids, vec![101, 202, 303, 404, 505, 606]);
    assert_eq!(result.diagnostics.len(), 6);
    assert!(output.join("sparse/0/images.bin").is_file());
    assert!(output.join("registration.json").is_file());
    assert!(!output.join("registration.json.tmp").exists());
    let attempts = events
        .iter()
        .filter(|event| event.operation == rustsfm::SfmTaskOperation::RegisterFrameAttempt)
        .collect::<Vec<_>>();
    assert_eq!(attempts.len(), 2);
    assert!(attempts.iter().all(|event| {
        event.stage == rustsfm::SfmTaskStage::FullFrameRegistration
            && event.kind == rustsfm::SfmTaskEventKind::Progress
    }));
    assert!(events
        .windows(2)
        .all(|window| window[0].sequence < window[1].sequence));
    let merged = read_colmap_sparse_model(&result.sparse_model)?.reconstruction;
    assert_eq!(merged.poses.iter().flatten().count(), 6);
    assert_eq!(
        merged
            .image_ids
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        6
    );
    Ok(())
}

#[test]
fn blank_sequence_frame_returns_unresolved_incomplete_coverage() -> anyhow::Result<()> {
    let (_temp, output, frames, keyframes, mapper_config) = synthetic_sequence_fixture(Some(3))?;
    let control = SfmTaskControl::new();
    let mut sink = |_| {};
    let mut task = SfmTaskContext::new(&control, &mut sink);

    let result = register_remaining_sequence_frames(
        &frames,
        &keyframes.keyframe_ids,
        &keyframes,
        &mapper_config,
        &synthetic_sequence_config(),
        &output,
        &mut task,
    )?;

    assert!(!result.has_complete_coverage());
    assert_eq!(result.registered_frames, 5);
    let blank = result
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.frame_id == 404)
        .unwrap();
    assert_eq!(blank.status, FrameRegistrationStatus::Unresolved);
    assert_eq!(blank.attempts, 2);
    assert!(require_complete_pose_coverage(&result).is_err());
    assert!(output.join("registration.json").is_file());
    Ok(())
}

#[test]
fn pause_between_stages_does_not_repeat_or_modify_keyframe_work() -> anyhow::Result<()> {
    let (_temp, output, frames, keyframes, mapper_config) = synthetic_sequence_fixture(None)?;
    let database = ColmapDatabase::open_read_only(&keyframes.database)?;
    let keypoints_before = database.read_keypoints(1)?;
    let sparse_before = std::fs::read(output.join("sparse/0/images.bin"))?;
    drop(database);
    let control = SfmTaskControl::new();
    control.request_pause();
    let mut events = Vec::new();
    let mut sink = |event| events.push(event);
    let mut task = SfmTaskContext::new(&control, &mut sink);

    let error = register_remaining_sequence_frames(
        &frames,
        &keyframes.keyframe_ids,
        &keyframes,
        &mapper_config,
        &synthetic_sequence_config(),
        &output,
        &mut task,
    )
    .unwrap_err();

    assert_eq!(
        error.downcast_ref::<rustsfm::SfmTaskStop>(),
        Some(&rustsfm::SfmTaskStop::Paused)
    );
    let database = ColmapDatabase::open_read_only(&keyframes.database)?;
    assert_eq!(database.read_keypoints(1)?, keypoints_before);
    assert_eq!(
        std::fs::read(output.join("sparse/0/images.bin"))?,
        sparse_before
    );
    assert!(events.is_empty());
    Ok(())
}

#[test]
fn preseeded_keyframe_stage_and_remaining_stage_compose_to_complete_sequence() -> anyhow::Result<()>
{
    let (_temp, output, frames, old_keyframes, mut mapper_config) =
        synthetic_sequence_fixture(None)?;
    let database = ColmapDatabase::open(&old_keyframes.database)?;
    let pending_rows = [2u32, 4u32]
        .into_iter()
        .map(|image_id| {
            Ok((
                database.read_image(image_id)?.unwrap(),
                database.read_keypoints(image_id)?,
                database.read_descriptors(image_id)?,
            ))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    drop(database);
    let connection = rusqlite::Connection::open(&old_keyframes.database)?;
    connection.execute("DELETE FROM descriptors WHERE image_id IN (2, 4)", [])?;
    connection.execute("DELETE FROM keypoints WHERE image_id IN (2, 4)", [])?;
    connection.execute("DELETE FROM images WHERE image_id IN (2, 4)", [])?;
    drop(connection);
    std::fs::remove_dir_all(output.join("sparse"))?;

    mapper_config.multiple_models = false;
    mapper_config.copy_images = false;
    mapper_config.init_num_trials = 1;
    mapper_config.init_min_num_inliers = 16;
    mapper_config.init_min_tri_angle_deg = 0.5;
    mapper_config.abs_pose_min_num_inliers = 16;
    mapper_config.ignore_two_view_tracks = false;
    let control = SfmTaskControl::new();
    let mut events = Vec::new();
    let mut sink = |event| events.push(event);
    let mut task = SfmTaskContext::new(&control, &mut sink);
    let keyframe_ids = SYNTHETIC_KEYFRAME_INDICES
        .iter()
        .map(|&index| frames[index].id)
        .collect::<Vec<_>>();

    let keyframes =
        run_keyframe_reconstruction(&frames, &keyframe_ids, &mapper_config, &output, &mut task)?;
    drop(task);
    drop(sink);

    assert_eq!(keyframes.registered_keyframes, 4);
    assert_eq!(keyframes.database, output.join("Cache/database.db"));
    assert!(keyframes.sparse_model.join("images.bin").is_file());
    assert_eq!(
        events
            .iter()
            .filter(|event| event.operation == rustsfm::SfmTaskOperation::ExtractImage)
            .count(),
        0,
        "preseeded keyframe features must be reused"
    );
    let database = ColmapDatabase::open(&keyframes.database)?;
    for (image, keypoints, descriptors) in pending_rows {
        database.write_image(&image, true)?;
        database.write_keypoints(image.image_id, &keypoints)?;
        database.write_descriptors(image.image_id, &descriptors)?;
    }
    drop(database);

    let mut sink = |_| {};
    let mut task = SfmTaskContext::new(&control, &mut sink);

    let result = register_remaining_sequence_frames(
        &frames,
        &keyframe_ids,
        &keyframes,
        &mapper_config,
        &synthetic_sequence_config(),
        &output,
        &mut task,
    )?;

    assert!(result.has_complete_coverage(), "{:#?}", result.diagnostics);
    assert_eq!(result.registered_frames, 6);
    Ok(())
}

#[test]
fn temporal_sample_plan_uses_nearest_keyframes_in_deterministic_order() {
    let plan = SequenceRegistrationPlan::build(12, &[0, 3, 6, 9, 11], 2, 4).unwrap();

    assert_eq!(
        plan.attempts_for(4, RegistrationRound::Narrow),
        &[3, 6, 0, 9]
    );
    assert_eq!(
        plan.attempts_for(4, RegistrationRound::Wide),
        &[3, 6, 0, 9, 11]
    );
    assert_eq!(plan.pending_frames(), &[1, 2, 4, 5, 7, 8, 10]);
}

#[test]
fn temporal_plan_json_round_trip_rebuilds_equivalent_attempts() {
    let plan = SequenceRegistrationPlan::build(12, &[0, 3, 6, 9, 11], 2, 4).unwrap();

    let json = serde_json::to_string(&plan).unwrap();
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(value.as_object().unwrap().len(), 6);
    assert!(value.get("pending").is_none());
    assert!(value.get("narrow_support").is_none());
    assert!(value.get("wide_support").is_none());
    let restored: SequenceRegistrationPlan = serde_json::from_str(&json).unwrap();

    assert_eq!(restored, plan);
    assert_eq!(
        restored.attempts_for(4, RegistrationRound::Narrow),
        &[3, 6, 0, 9]
    );
    assert_eq!(
        restored.attempts_for(4, RegistrationRound::Wide),
        &[3, 6, 0, 9, 11]
    );
    assert_eq!(restored.pending_frames(), &[1, 2, 4, 5, 7, 8, 10]);
}

#[test]
fn temporal_plan_json_rejects_invalid_keyframe_inputs() {
    let invalid = serde_json::json!({
        "frame_count": 4,
        "keyframes": [0, 2, 1],
        "narrow_neighbors_each_side": 2,
        "wide_neighbors_each_side": 4,
        "frame_ids": [10, 20, 30, 40],
        "timestamps_us": null,
    });

    assert!(serde_json::from_value::<SequenceRegistrationPlan>(invalid).is_err());
}

#[test]
fn temporal_rounds_limit_each_side_then_sort_by_distance_and_frame_id() {
    let first = SequenceRegistrationPlan::build(10, &[0, 2, 7, 9], 1, 3).unwrap();
    let second = SequenceRegistrationPlan::build(10, &[0, 2, 7, 9], 1, 3).unwrap();

    assert_eq!(first.attempts_for(5, RegistrationRound::Narrow), &[7, 2]);
    assert_eq!(
        first.attempts_for(5, RegistrationRound::Wide),
        &[7, 2, 9, 0]
    );
    assert_eq!(first, second);
}

#[test]
fn temporal_later_round_can_add_only_explicit_registered_support() {
    let plan = SequenceRegistrationPlan::build(12, &[0, 3, 6, 9, 11], 2, 4).unwrap();

    assert_eq!(
        plan.attempts_for_with_support(4, RegistrationRound::Wide, &[]),
        vec![3, 6, 0, 9, 11]
    );
    assert_eq!(
        plan.attempts_for_with_support(4, RegistrationRound::Wide, &[7, 5, 2, 5, 4, usize::MAX],),
        vec![3, 5, 2, 6, 7, 0, 9]
    );
}

#[test]
fn temporal_sorted_dynamic_support_reuses_bounded_keyframe_attempts() {
    let plan = SequenceRegistrationPlan::build(12, &[0, 3, 6, 9, 11], 2, 4).unwrap();
    let registered_support = [2, 5, 7];
    let expected = vec![3, 5, 2, 6, 7, 0, 9];

    for _ in 0..1_000 {
        assert_eq!(
            plan.attempts_for_with_sorted_support(4, RegistrationRound::Wide, &registered_support,)
                .unwrap(),
            expected
        );
    }

    assert!(matches!(
        plan.attempts_for_with_sorted_support(4, RegistrationRound::Wide, &[2, 2]),
        Err(SequenceRegistrationError::DynamicSupportNotSortedUnique)
    ));
}

#[test]
fn temporal_sorted_dynamic_support_rejects_oversized_input() {
    let plan = SequenceRegistrationPlan::build(12, &[0, 3, 6, 9, 11], 2, 4).unwrap();
    let oversized = vec![0; MAX_DYNAMIC_SUPPORT_CANDIDATES + 1];

    assert!(matches!(
        plan.attempts_for_with_sorted_support(4, RegistrationRound::Wide, &oversized),
        Err(SequenceRegistrationError::DynamicSupportLimitExceeded {
            candidate_count,
            max_candidates: MAX_DYNAMIC_SUPPORT_CANDIDATES,
        }) if candidate_count == oversized.len()
    ));
    assert_eq!(
        plan.attempts_for_with_support(4, RegistrationRound::Wide, &oversized),
        plan.attempts_for(4, RegistrationRound::Wide)
    );
}

#[test]
fn temporal_dynamic_support_remains_bounded_for_large_merged_inputs() {
    let frame_count = 20_000;
    let frames: Vec<_> = (0..frame_count)
        .map(|frame| SequenceFrame {
            id: (frame_count - frame) as u32,
            image_path: PathBuf::new(),
            timestamp_us: Some((frame / 64) as i64),
        })
        .collect();
    let keyframes: Vec<_> = (0..frame_count).step_by(2).collect();
    let plan = SequenceRegistrationPlan::build_from_frames(&frames, &keyframes, 4, 8).unwrap();
    let registered_support: Vec<_> = (1..frame_count).step_by(2).collect();

    let support = plan
        .attempts_for_with_sorted_support(10_001, RegistrationRound::Narrow, &registered_support)
        .unwrap();

    assert!(support.len() <= 8);
    assert!(!support.contains(&10_001));
}

#[test]
fn temporal_frame_plan_orders_support_by_timestamp_distance_then_frame_id() {
    let frames = [
        SequenceFrame {
            id: 100,
            image_path: PathBuf::from("000.jpg"),
            timestamp_us: Some(0),
        },
        SequenceFrame {
            id: 90,
            image_path: PathBuf::from("001.jpg"),
            timestamp_us: Some(999),
        },
        SequenceFrame {
            id: 50,
            image_path: PathBuf::from("002.jpg"),
            timestamp_us: Some(1_000),
        },
        SequenceFrame {
            id: 10,
            image_path: PathBuf::from("003.jpg"),
            timestamp_us: Some(1_001),
        },
        SequenceFrame {
            id: 30,
            image_path: PathBuf::from("004.jpg"),
            timestamp_us: Some(2_000),
        },
    ];

    let plan = SequenceRegistrationPlan::build_from_frames(&frames, &[0, 1, 3, 4], 2, 4).unwrap();

    assert_eq!(
        plan.attempts_for(2, RegistrationRound::Narrow),
        &[3, 1, 4, 0]
    );

    let restored: SequenceRegistrationPlan =
        serde_json::from_str(&serde_json::to_string(&plan).unwrap()).unwrap();
    assert_eq!(restored, plan);
    assert_eq!(
        restored.attempts_for(2, RegistrationRound::Narrow),
        &[3, 1, 4, 0]
    );
}

#[test]
fn temporal_frame_plan_selects_each_side_by_timestamp_distance_and_frame_id() {
    let frames = [
        SequenceFrame {
            id: 1,
            image_path: PathBuf::from("000.jpg"),
            timestamp_us: Some(90),
        },
        SequenceFrame {
            id: 2,
            image_path: PathBuf::from("001.jpg"),
            timestamp_us: Some(90),
        },
        SequenceFrame {
            id: 50,
            image_path: PathBuf::from("002.jpg"),
            timestamp_us: Some(90),
        },
        SequenceFrame {
            id: 40,
            image_path: PathBuf::from("003.jpg"),
            timestamp_us: Some(90),
        },
        SequenceFrame {
            id: 30,
            image_path: PathBuf::from("004.jpg"),
            timestamp_us: Some(90),
        },
        SequenceFrame {
            id: 60,
            image_path: PathBuf::from("005.jpg"),
            timestamp_us: Some(100),
        },
        SequenceFrame {
            id: 3,
            image_path: PathBuf::from("006.jpg"),
            timestamp_us: Some(200),
        },
    ];
    let plan =
        SequenceRegistrationPlan::build_from_frames(&frames, &[0, 1, 2, 3, 4, 6], 2, 4).unwrap();

    assert_eq!(plan.attempts_for(5, RegistrationRound::Narrow), &[0, 1, 6]);
}

#[test]
fn temporal_frame_plan_rejects_unsorted_timestamps() {
    let frames = [
        SequenceFrame {
            id: 10,
            image_path: PathBuf::from("000.jpg"),
            timestamp_us: Some(0),
        },
        SequenceFrame {
            id: 20,
            image_path: PathBuf::from("001.jpg"),
            timestamp_us: Some(200),
        },
        SequenceFrame {
            id: 30,
            image_path: PathBuf::from("002.jpg"),
            timestamp_us: Some(100),
        },
    ];

    assert!(matches!(
        SequenceRegistrationPlan::build_from_frames(&frames, &[0, 2], 2, 4),
        Err(SequenceRegistrationError::UnsortedTimestamps {
            previous_frame: 1,
            current_frame: 2,
        })
    ));
}

#[test]
fn temporal_frame_plan_rejects_oversized_timestamp_plateau() {
    let frames: Vec<_> = (0..=MAX_TIMESTAMP_PLATEAU)
        .map(|frame| SequenceFrame {
            id: frame as u32,
            image_path: PathBuf::new(),
            timestamp_us: Some(42),
        })
        .collect();

    assert!(matches!(
        SequenceRegistrationPlan::build_from_frames(
            &frames,
            &[0, MAX_TIMESTAMP_PLATEAU],
            2,
            4,
        ),
        Err(SequenceRegistrationError::TimestampPlateauTooLarge {
            timestamp_us: 42,
            plateau_size,
            max_plateau_size: MAX_TIMESTAMP_PLATEAU,
        }) if plateau_size == frames.len()
    ));
}

#[test]
fn temporal_frame_plan_rejects_duplicate_frame_ids() {
    let frames = [
        SequenceFrame {
            id: 10,
            image_path: PathBuf::from("000.jpg"),
            timestamp_us: Some(0),
        },
        SequenceFrame {
            id: 10,
            image_path: PathBuf::from("001.jpg"),
            timestamp_us: Some(100),
        },
    ];

    assert!(matches!(
        SequenceRegistrationPlan::build_from_frames(&frames, &[0, 1], 2, 4),
        Err(SequenceRegistrationError::InvalidFrameIds {
            imported_frames: 2,
            frame_id_count: 2,
            duplicate_frame_ids,
        }) if duplicate_frame_ids == vec![10]
    ));
}

#[test]
fn temporal_plan_support_lists_only_contain_keyframes() {
    let plan = SequenceRegistrationPlan::build(8, &[0, 4, 7], 2, 4).unwrap();

    for frame in plan.pending_frames() {
        for round in [RegistrationRound::Narrow, RegistrationRound::Wide] {
            assert!(plan
                .attempts_for(*frame, round)
                .iter()
                .all(|support| [0, 4, 7].contains(support)));
        }
    }
}

#[test]
fn invalid_temporal_plan_rejects_empty_sequences_and_keyframes() {
    assert!(SequenceRegistrationPlan::build(0, &[], 2, 4).is_err());
    assert!(SequenceRegistrationPlan::build(4, &[], 2, 4).is_err());
}

#[test]
fn invalid_temporal_plan_rejects_duplicate_keyframes() {
    assert!(SequenceRegistrationPlan::build(4, &[0, 1, 1], 2, 4).is_err());
}

#[test]
fn invalid_temporal_plan_rejects_out_of_range_keyframes() {
    assert!(SequenceRegistrationPlan::build(4, &[0, 4], 2, 4).is_err());
}

#[test]
fn invalid_temporal_plan_rejects_unsorted_keyframes() {
    assert!(SequenceRegistrationPlan::build(4, &[0, 2, 1], 2, 4).is_err());
}

#[test]
fn invalid_temporal_plan_rejects_more_keyframes_than_frames() {
    assert!(matches!(
        SequenceRegistrationPlan::build(2, &[0, 1, 1], 2, 4),
        Err(SequenceRegistrationError::TooManyKeyframes {
            frame_count: 2,
            keyframe_count: 3,
        })
    ));
}

#[test]
fn invalid_temporal_plan_rejects_oversized_neighbor_rounds() {
    for (narrow, wide, round) in [
        (MAX_SEQUENCE_NEIGHBORS + 1, 4, RegistrationRound::Narrow),
        (2, MAX_SEQUENCE_NEIGHBORS + 1, RegistrationRound::Wide),
    ] {
        assert!(matches!(
            SequenceRegistrationPlan::build(4, &[0, 3], narrow, wide),
            Err(SequenceRegistrationError::SequenceNeighborLimitExceeded {
                round: rejected_round,
                requested,
                max_neighbors: MAX_SEQUENCE_NEIGHBORS,
            }) if rejected_round == round && requested == MAX_SEQUENCE_NEIGHBORS + 1
        ));
    }

    let json = serde_json::json!({
        "frame_count": 4,
        "keyframes": [0, 3],
        "narrow_neighbors_each_side": MAX_SEQUENCE_NEIGHBORS + 1,
        "wide_neighbors_each_side": 4,
        "frame_ids": [0, 1, 2, 3],
        "timestamps_us": null,
    });
    assert!(serde_json::from_value::<SequenceRegistrationPlan>(json).is_err());
}

#[test]
fn invalid_temporal_plan_rejects_total_support_budget_before_allocating() {
    let narrow = MAX_SEQUENCE_NEIGHBORS;
    let wide = MAX_SEQUENCE_NEIGHBORS;
    let entries_per_frame = 2 * (narrow + wide);
    let frame_count = MAX_TOTAL_SUPPORT_ENTRIES / entries_per_frame + 1;

    assert!(frame_count <= MAX_SEQUENCE_PLAN_FRAMES);
    assert!(matches!(
        SequenceRegistrationPlan::build(frame_count, &[0], narrow, wide),
        Err(SequenceRegistrationError::SequenceSupportBudgetExceeded {
            frame_count: rejected_frames,
            estimated_support_entries,
            max_support_entries: MAX_TOTAL_SUPPORT_ENTRIES,
        }) if rejected_frames == frame_count
            && estimated_support_entries
                == (frame_count as u128 * entries_per_frame as u128)
    ));

    let json = serde_json::json!({
        "frame_count": frame_count,
        "keyframes": [0],
        "narrow_neighbors_each_side": narrow,
        "wide_neighbors_each_side": wide,
        "frame_ids": (0..frame_count as u32).collect::<Vec<_>>(),
        "timestamps_us": null,
    });
    assert!(serde_json::from_value::<SequenceRegistrationPlan>(json).is_err());
}

#[cfg(target_pointer_width = "64")]
#[test]
fn invalid_temporal_plan_rejects_unrepresentable_frame_count_before_allocating() {
    for frame_count in [MAX_SEQUENCE_PLAN_FRAMES + 1, u32::MAX as usize + 1] {
        assert!(matches!(
            SequenceRegistrationPlan::build(frame_count, &[0], 2, 4),
            Err(SequenceRegistrationError::SequencePlanTooLarge {
                frame_count: rejected,
                max_frame_count: MAX_SEQUENCE_PLAN_FRAMES,
            }) if rejected == frame_count
        ));
    }

    let json = serde_json::json!({
        "frame_count": u32::MAX as usize + 1,
        "keyframes": [0],
        "narrow_neighbors_each_side": 2,
        "wide_neighbors_each_side": 4,
        "frame_ids": [0],
        "timestamps_us": null,
    });
    assert!(serde_json::from_value::<SequenceRegistrationPlan>(json).is_err());
}

#[test]
fn registration_status_identifies_pose_coverage() {
    assert!(FrameRegistrationStatus::Keyframe.is_registered());
    assert!(FrameRegistrationStatus::Registered.is_registered());
    assert!(!FrameRegistrationStatus::Unresolved.is_registered());
    assert!(!FrameRegistrationStatus::Excluded.is_registered());
}

#[test]
fn sequence_config_defaults_match_registration_policy() {
    let config = SequenceRegistrationConfig::default();

    assert_eq!(config.narrow_neighbors_each_side, 2);
    assert_eq!(config.wide_neighbors_each_side, 4);
    assert_eq!(config.min_inliers, 24);
    assert_eq!(config.min_inlier_ratio, 0.20);
    assert_eq!(config.max_reprojection_error, 4.0);
    assert!(config.use_gpu_pnp);
    assert_json_round_trip(&config);
}

#[test]
fn sequence_config_validation_rejects_non_finite_metrics() {
    let config = SequenceRegistrationConfig {
        min_inlier_ratio: f64::NAN,
        ..Default::default()
    };
    assert!(matches!(
        config.validate(),
        Err(SequenceRegistrationError::InvalidConfigMetric {
            field: "min_inlier_ratio"
        })
    ));

    let config = SequenceRegistrationConfig {
        max_reprojection_error: f64::INFINITY,
        ..Default::default()
    };
    assert!(matches!(
        config.validate(),
        Err(SequenceRegistrationError::InvalidConfigMetric {
            field: "max_reprojection_error"
        })
    ));
}

#[test]
fn sequence_frame_and_round_round_trip_through_json() {
    let frame = SequenceFrame {
        id: 42,
        image_path: PathBuf::from("images/000042.jpg"),
        timestamp_us: Some(1_234_567),
    };

    assert_json_round_trip(&frame);
    assert_json_round_trip(&RegistrationRound::Narrow);
    assert_json_round_trip(&RegistrationRound::Wide);
    assert_eq!(
        serde_json::to_value(RegistrationRound::Narrow).unwrap(),
        "narrow"
    );
}

#[test]
fn diagnostic_update_preserves_attempt_state_and_metrics_in_json() {
    let mut diagnostic = FrameRegistrationDiagnostic::new(4, FrameRegistrationStatus::Unresolved);

    diagnostic.record_attempt(
        FrameRegistrationStatus::Unresolved,
        vec![3, 6],
        18,
        0.36,
        Some(3.25),
        Some("narrow support was insufficient".to_owned()),
    );
    diagnostic.record_attempt(
        FrameRegistrationStatus::Registered,
        vec![3, 6, 0, 9, 11],
        31,
        0.62,
        Some(1.5),
        Some("registered in wide round".to_owned()),
    );

    assert_eq!(diagnostic.frame_id, 4);
    assert_eq!(diagnostic.status, FrameRegistrationStatus::Registered);
    assert_eq!(diagnostic.attempts, 2);
    assert_eq!(diagnostic.support_frame_ids, vec![3, 6, 0, 9, 11]);
    assert_eq!(diagnostic.inlier_count, 31);
    assert_eq!(diagnostic.inlier_ratio, 0.62);
    assert_eq!(diagnostic.mean_reprojection_error, Some(1.5));
    assert_eq!(
        diagnostic.message.as_deref(),
        Some("registered in wide round")
    );
    assert_json_round_trip(&diagnostic);
}

#[test]
fn sequence_result_round_trip_preserves_diagnostics() {
    let result = SequenceRegistrationResult {
        imported_frames: 2,
        registered_frames: 1,
        frame_ids: vec![0, 1],
        diagnostics: vec![
            FrameRegistrationDiagnostic::new(0, FrameRegistrationStatus::Keyframe),
            FrameRegistrationDiagnostic {
                frame_id: 1,
                status: FrameRegistrationStatus::Unresolved,
                attempts: 2,
                support_frame_ids: vec![0],
                inlier_count: 11,
                inlier_ratio: 0.15,
                mean_reprojection_error: Some(4.5),
                message: Some("inlier threshold not met".to_owned()),
            },
        ],
        sparse_model: PathBuf::from("sparse/0"),
    };

    assert_json_round_trip(&result);

    let mut json = serde_json::to_value(&result).unwrap();
    json.as_object_mut().unwrap().remove("frame_ids");
    assert!(serde_json::from_value::<SequenceRegistrationResult>(json).is_err());
}

#[test]
fn complete_coverage_accepts_keyframes_and_registered_frames() {
    let result = SequenceRegistrationResult {
        imported_frames: 2,
        registered_frames: 2,
        frame_ids: vec![10, 20],
        diagnostics: vec![
            FrameRegistrationDiagnostic::new(10, FrameRegistrationStatus::Keyframe),
            FrameRegistrationDiagnostic::new(20, FrameRegistrationStatus::Registered),
        ],
        sparse_model: PathBuf::from("sparse/0"),
    };

    assert!(result.has_complete_coverage());
    assert_eq!(result.validate_complete_coverage(), Ok(()));
}

#[test]
fn incomplete_frame_count_returns_an_explicit_error() {
    let result = SequenceRegistrationResult {
        imported_frames: 3,
        registered_frames: 2,
        frame_ids: vec![0, 1, 2],
        diagnostics: vec![
            FrameRegistrationDiagnostic::new(0, FrameRegistrationStatus::Keyframe),
            FrameRegistrationDiagnostic::new(1, FrameRegistrationStatus::Registered),
            FrameRegistrationDiagnostic::new(2, FrameRegistrationStatus::Unresolved),
        ],
        sparse_model: PathBuf::from("sparse/0"),
    };

    assert!(!result.has_complete_coverage());
    let error = result.validate_complete_coverage().unwrap_err();
    assert!(error.to_string().contains("2 of 3"));
    assert!(matches!(
        error,
        SequenceRegistrationError::IncompleteCoverage {
            imported_frames: 3,
            registered_frames: 2,
            unresolved_frame_ids,
        } if unresolved_frame_ids == vec![2]
    ));
}

#[test]
fn unresolved_diagnostic_fails_coverage_even_when_counts_match() {
    let result = SequenceRegistrationResult {
        imported_frames: 3,
        registered_frames: 3,
        frame_ids: vec![0, 1, 2],
        diagnostics: vec![
            FrameRegistrationDiagnostic::new(0, FrameRegistrationStatus::Keyframe),
            FrameRegistrationDiagnostic::new(1, FrameRegistrationStatus::Registered),
            FrameRegistrationDiagnostic::new(2, FrameRegistrationStatus::Unresolved),
        ],
        sparse_model: PathBuf::from("sparse/0"),
    };

    assert!(!result.has_complete_coverage());
    let error = result.validate_complete_coverage().unwrap_err();
    assert!(error.to_string().contains("frame 2"));
    assert!(matches!(
        error,
        SequenceRegistrationError::RegistrationStatusCountMismatch {
            registered_frames: 3,
            diagnostic_registered_frames: 2,
            unresolved_frame_ids,
        } if unresolved_frame_ids == vec![2]
    ));
}

#[test]
fn complete_counts_with_missing_diagnostic_fail_coverage() {
    let result = SequenceRegistrationResult {
        imported_frames: 3,
        registered_frames: 3,
        frame_ids: vec![0, 1, 2],
        diagnostics: vec![
            FrameRegistrationDiagnostic::new(0, FrameRegistrationStatus::Keyframe),
            FrameRegistrationDiagnostic::new(1, FrameRegistrationStatus::Registered),
        ],
        sparse_model: PathBuf::from("sparse/0"),
    };

    assert!(!result.has_complete_coverage());
    assert!(matches!(
        result.validate_complete_coverage(),
        Err(SequenceRegistrationError::DiagnosticCountMismatch {
            imported_frames: 3,
            diagnostic_count: 2,
        })
    ));
}

#[test]
fn complete_counts_with_duplicate_diagnostic_fail_coverage() {
    let result = SequenceRegistrationResult {
        imported_frames: 3,
        registered_frames: 3,
        frame_ids: vec![0, 1, 2],
        diagnostics: vec![
            FrameRegistrationDiagnostic::new(0, FrameRegistrationStatus::Keyframe),
            FrameRegistrationDiagnostic::new(1, FrameRegistrationStatus::Registered),
            FrameRegistrationDiagnostic::new(1, FrameRegistrationStatus::Registered),
        ],
        sparse_model: PathBuf::from("sparse/0"),
    };

    assert!(!result.has_complete_coverage());
    assert!(matches!(
        result.validate_complete_coverage(),
        Err(SequenceRegistrationError::InvalidDiagnostics {
            imported_frames: 3,
            diagnostic_count: 3,
            missing_frame_ids,
            duplicate_frame_ids,
            unexpected_frame_ids,
        }) if missing_frame_ids == vec![2]
            && duplicate_frame_ids == vec![1]
            && unexpected_frame_ids.is_empty()
    ));
}

#[test]
fn arbitrary_expected_frame_ids_reject_fabricated_contiguous_diagnostics() {
    let result = SequenceRegistrationResult {
        imported_frames: 2,
        registered_frames: 2,
        frame_ids: vec![10, 20],
        diagnostics: vec![
            FrameRegistrationDiagnostic::new(0, FrameRegistrationStatus::Keyframe),
            FrameRegistrationDiagnostic::new(1, FrameRegistrationStatus::Registered),
        ],
        sparse_model: PathBuf::from("sparse/0"),
    };

    assert!(matches!(
        result.validate_complete_coverage(),
        Err(SequenceRegistrationError::InvalidDiagnostics {
            imported_frames: 2,
            diagnostic_count: 2,
            missing_frame_ids,
            duplicate_frame_ids,
            unexpected_frame_ids,
        }) if missing_frame_ids == vec![10, 20]
            && duplicate_frame_ids.is_empty()
            && unexpected_frame_ids == vec![0, 1]
    ));
}

#[test]
fn duplicate_expected_frame_ids_fail_coverage() {
    let result = SequenceRegistrationResult {
        imported_frames: 2,
        registered_frames: 2,
        frame_ids: vec![10, 10],
        diagnostics: vec![
            FrameRegistrationDiagnostic::new(10, FrameRegistrationStatus::Keyframe),
            FrameRegistrationDiagnostic::new(20, FrameRegistrationStatus::Registered),
        ],
        sparse_model: PathBuf::from("sparse/0"),
    };

    assert!(matches!(
        result.validate_complete_coverage(),
        Err(SequenceRegistrationError::InvalidFrameIds {
            imported_frames: 2,
            frame_id_count: 2,
            duplicate_frame_ids,
        }) if duplicate_frame_ids == vec![10]
    ));
}

#[test]
fn expected_frame_id_count_must_match_imported_frames() {
    let result = SequenceRegistrationResult {
        imported_frames: 2,
        registered_frames: 2,
        frame_ids: vec![10],
        diagnostics: vec![
            FrameRegistrationDiagnostic::new(10, FrameRegistrationStatus::Keyframe),
            FrameRegistrationDiagnostic::new(20, FrameRegistrationStatus::Registered),
        ],
        sparse_model: PathBuf::from("sparse/0"),
    };

    assert!(matches!(
        result.validate_complete_coverage(),
        Err(SequenceRegistrationError::InvalidFrameIds {
            imported_frames: 2,
            frame_id_count: 1,
            duplicate_frame_ids,
        }) if duplicate_frame_ids.is_empty()
    ));
}

#[test]
fn empty_sequence_result_fails_coverage() {
    let result = SequenceRegistrationResult {
        imported_frames: 0,
        registered_frames: 0,
        frame_ids: Vec::new(),
        diagnostics: Vec::new(),
        sparse_model: PathBuf::from("sparse/0"),
    };

    assert_eq!(
        result.validate_complete_coverage(),
        Err(SequenceRegistrationError::EmptySequence)
    );
}

#[cfg(target_pointer_width = "64")]
#[test]
fn huge_sequence_result_rejects_diagnostic_count_without_allocating() {
    let result = SequenceRegistrationResult {
        imported_frames: usize::MAX,
        registered_frames: 0,
        frame_ids: Vec::new(),
        diagnostics: Vec::new(),
        sparse_model: PathBuf::from("sparse/0"),
    };

    assert!(matches!(
        result.validate_complete_coverage(),
        Err(SequenceRegistrationError::DiagnosticCountMismatch {
            imported_frames: usize::MAX,
            diagnostic_count: 0,
        })
    ));
}

#[test]
fn non_finite_diagnostic_metrics_fail_coverage() {
    let mut diagnostic = FrameRegistrationDiagnostic::new(10, FrameRegistrationStatus::Registered);
    diagnostic.inlier_ratio = f64::NAN;
    let result = SequenceRegistrationResult {
        imported_frames: 1,
        registered_frames: 1,
        frame_ids: vec![10],
        diagnostics: vec![diagnostic],
        sparse_model: PathBuf::from("sparse/0"),
    };

    assert!(matches!(
        result.validate_complete_coverage(),
        Err(SequenceRegistrationError::InvalidDiagnosticMetric {
            frame_id: 10,
            field: "inlier_ratio",
        })
    ));

    let mut diagnostic = FrameRegistrationDiagnostic::new(10, FrameRegistrationStatus::Registered);
    diagnostic.mean_reprojection_error = Some(f64::INFINITY);
    let result = SequenceRegistrationResult {
        imported_frames: 1,
        registered_frames: 1,
        frame_ids: vec![10],
        diagnostics: vec![diagnostic],
        sparse_model: PathBuf::from("sparse/0"),
    };
    assert!(matches!(
        result.validate_complete_coverage(),
        Err(SequenceRegistrationError::InvalidDiagnosticMetric {
            frame_id: 10,
            field: "mean_reprojection_error",
        })
    ));
}
