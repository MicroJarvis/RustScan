use std::path::PathBuf;

use rustgs::{
    evaluate_splats, evaluation_device, load_colmap_training_dataset, select_evaluation_frames,
    ColmapConfig, EvaluationDevice, SplatEvaluationConfig, SplatMetadata, TrainingConfig,
    TrainingOptions,
};

fn colmap_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../test_data/tum_freiburg1_xyz_colmap")
}

fn colmap_root_if_available() -> Option<PathBuf> {
    let root = colmap_root();
    root.exists().then_some(root)
}

#[test]
fn loads_workspace_colmap_directory_as_training_dataset() {
    let Some(root) = colmap_root_if_available() else {
        eprintln!(
            "skipping test: missing COLMAP fixture at {}",
            colmap_root().display()
        );
        return;
    };
    let dataset = load_colmap_training_dataset(
        &root,
        &ColmapConfig {
            max_frames: 90,
            frame_stride: 30,
            ..Default::default()
        },
    )
    .unwrap();

    assert!(dataset.poses.len() >= 3);
    assert!(!dataset.initial_points.is_empty());
}

#[test]
fn selects_stable_colmap_eval_subset_with_stride() {
    let Some(root) = colmap_root_if_available() else {
        eprintln!(
            "skipping test: missing COLMAP fixture at {}",
            colmap_root().display()
        );
        return;
    };
    let dataset = load_colmap_training_dataset(
        &root,
        &ColmapConfig {
            max_frames: 180,
            frame_stride: 1,
            ..Default::default()
        },
    )
    .unwrap();

    let selected = select_evaluation_frames(&dataset, 180, 30);
    assert!(!selected.poses.is_empty());
    assert!(selected.poses.len() <= dataset.poses.len());
}

#[cfg(feature = "gpu")]
#[test]
fn trains_directly_from_workspace_colmap_directory() {
    let Some(root) = colmap_root_if_available() else {
        eprintln!(
            "skipping test: missing COLMAP fixture at {}",
            colmap_root().display()
        );
        return;
    };
    if !rustgs::gpu_available() {
        eprintln!("skipping test: GPU unavailable in current environment");
        return;
    }
    let config = TrainingConfig {
        iterations: 1,
        initialization: rustgs::TrainingInitializationConfig {
            max_initial_gaussians: 10_000,
            ..rustgs::TrainingInitializationConfig::default()
        },
        ..TrainingConfig::default()
    };
    let dataset = load_colmap_training_dataset(
        &root,
        &ColmapConfig {
            max_frames: 90,
            frame_stride: 30,
            ..Default::default()
        },
    )
    .unwrap();

    let run = rustgs::train_splats(&dataset, &config, TrainingOptions::default()).unwrap();

    assert!(!run.splats.is_empty());
}

#[cfg(feature = "gpu")]
#[test]
fn colmap_training_smoke_produces_post_train_evaluation_summary() {
    let Some(root) = colmap_root_if_available() else {
        eprintln!(
            "skipping test: missing COLMAP fixture at {}",
            colmap_root().display()
        );
        return;
    };
    if !rustgs::gpu_available() {
        eprintln!("skipping test: GPU unavailable in current environment");
        return;
    }

    let colmap_config = ColmapConfig {
        max_frames: 90,
        frame_stride: 30,
        ..Default::default()
    };
    let dataset = load_colmap_training_dataset(&root, &colmap_config).unwrap();

    let config = TrainingConfig {
        iterations: 1,
        initialization: rustgs::TrainingInitializationConfig {
            max_initial_gaussians: 2_000,
            ..rustgs::TrainingInitializationConfig::default()
        },
        ..TrainingConfig::default()
    };
    let run = rustgs::train_splats(&dataset, &config, TrainingOptions::default()).unwrap();
    let metadata = SplatMetadata {
        iterations: config.iterations,
        final_loss: 0.0,
        gaussian_count: run.splats.len(),
        sh_degree: run.splats.sh_degree(),
    };
    let device = evaluation_device(EvaluationDevice::Gpu).unwrap();
    let evaluation = evaluate_splats(
        &dataset,
        &run.splats,
        &metadata,
        &SplatEvaluationConfig {
            render_scale: 0.25,
            raster_cov_blur: rustgs::DEFAULT_RASTER_COV_BLUR,
            frame_stride: 30,
            max_frames: 90,
            worst_frame_count: 2,
        },
        &device,
        None,
    )
    .unwrap();

    assert_eq!(evaluation.summary.device, EvaluationDevice::Gpu);
    assert!(evaluation.summary.frame_count > 0);
    assert_eq!(evaluation.summary.splat_count, run.splats.len());
    assert!(evaluation.summary.psnr_mean_db.is_finite());
    assert!(evaluation.summary.elapsed_seconds >= 0.0);
}
