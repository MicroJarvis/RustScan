use super::{
    ensure_sparse_initialization_points, filter_dataset_by_frame_ranges,
    filter_dataset_to_frame_ranges, oversample_dataset_frame_ranges, parse_frame_ranges,
    run_train_command,
};
use crate::TrainArgs;
use std::path::PathBuf;

#[test]
fn run_train_command_surfaces_missing_input_cleanly() {
    let args = TrainArgs {
        input: PathBuf::from("missing-dataset"),
        output: PathBuf::from("scene.ply"),
        train_config: None,
        train_preset: None,
        iterations: 1,
        max_initial_gaussians: 16,
        sampling_step: 0,
        max_frames: 0,
        frame_stride: 1,
        include_frame_ranges: None,
        exclude_frame_ranges: None,
        oversample_frame_ranges: None,
        oversample_frame_repeat: 1,
        frame_shuffle_seed: 0,
        render_scale: 0.5,
        raster_cov_blur: rustgs::DEFAULT_RASTER_COV_BLUR,
        raster_cov_blur_final: None,
        raster_cov_blur_final_after_epoch: None,
        litegs_sh_degree: 3,
        litegs_profile: rustgs::LiteGsTrainingProfile::Baseline,
        litegs_tile_size: rustgs::LiteGsTileSize::new(8, 16),
        litegs_sparse_grad: false,
        litegs_reg_weight: 0.0,
        litegs_enable_transmittance: false,
        litegs_enable_depth: false,
        litegs_densify_from: 3,
        litegs_densify_until: None,
        litegs_topology_freeze_after_epoch: None,
        litegs_growth_freeze_after_epoch: None,
        litegs_refine_every: 16,
        litegs_densification_interval: 100,
        litegs_growth_grad_threshold: 0.0002,
        litegs_split_score: rustgs::LiteGsSplitScoreMode::Baseline,
        litegs_split_grad_threshold: 0.0002,
        litegs_depth_scale_gamma: 0.37,
        litegs_growth_select_fraction: 0.2,
        litegs_growth_stop_iter: 15_000,
        litegs_opacity_decay: 0.004,
        litegs_scale_decay: 0.002,
        litegs_opacity_reset_interval: 3000,
        litegs_opacity_reset_mode: rustgs::LiteGsOpacityResetMode::Decay,
        litegs_prune_mode: rustgs::LiteGsPruneMode::Weight,
        litegs_prune_offset_epochs: 0,
        litegs_prune_min_age: 5,
        litegs_prune_invisible_epochs: 10,
        litegs_prune_opacity_threshold: 1.0 / 255.0,
        litegs_prune_visibility_dry_run: false,
        litegs_prune_visibility_threshold: 0.05,
        litegs_prune_high_opacity_threshold: 0.80,
        litegs_prune_until_epoch: None,
        litegs_target_primitives: 300_000,
        litegs_learnable_viewproj: false,
        litegs_lr_pose: 0.0001,
        litegs_prune_scale_threshold: 0.5,
        lr_position: 0.00016,
        lr_position_final: 0.0000016,
        lr_decay_iterations: 0,
        lr_scale: 0.005,
        lr_scale_final: 0.0,
        lr_rotation: 0.001,
        lr_rotation_final: 0.0,
        lr_opacity: 0.05,
        lr_opacity_final: 0.0,
        lr_color: 0.0025,
        lr_color_final: 0.0,
        loss_l1_weight: 0.8,
        loss_ssim_weight: 0.2,
        loss_gradient_weight: 0.0,
        loss_robust_delta: 0.0,
        loss_outlier_threshold: 0.0,
        loss_outlier_weight: 1.0,
        loss_dynamic_mask_threshold_low: 0.0,
        loss_dynamic_mask_threshold_high: 0.0,
        loss_dynamic_mask_min_weight: 1.0,
        loss_dynamic_mask_start_epoch: None,
        log_level: "error".to_string(),
        eval_after_train: false,
        eval_render_scale: 0.25,
        eval_raster_cov_blur: None,
        eval_max_frames: 180,
        eval_frame_stride: 30,
        eval_include_frame_ranges: None,
        eval_exclude_frame_ranges: None,
        eval_worst_frames: 5,
        eval_device: "metal".to_string(),
        eval_json: false,
        eval_crop_output_dir: None,
        eval_crop_frames: None,
        eval_crop_rect: None,
    };

    let err = run_train_command(args, super::TrainArgSources::default())
        .expect_err("missing input should fail");
    assert!(err.to_string().contains("COLMAP dataset directory"));
}

#[test]
fn training_requires_sparse_initialization_points() {
    let dataset =
        rustscan_types::TrainingDataset::new(rustgs::Intrinsics::new(1.0, 1.0, 0.0, 0.0, 1, 1));
    let err = ensure_sparse_initialization_points(
        &dataset,
        rustgs::TrainingInputKind::Colmap,
        PathBuf::from("test_data/colmap").as_path(),
    )
    .expect_err("dataset without sparse points should fail");

    assert!(
        err.to_string()
            .contains("training initialization now requires COLMAP sparse points"),
        "unexpected error: {err}"
    );
}

#[test]
fn parse_frame_ranges_accepts_singletons_and_inclusive_ranges() {
    let ranges = parse_frame_ranges(Some("76-93,155,200..202")).unwrap();

    assert_eq!(ranges.len(), 3);
    assert!(ranges[0].contains(76));
    assert!(ranges[0].contains(93));
    assert!(!ranges[0].contains(94));
    assert!(ranges[1].contains(155));
    assert!(ranges[2].contains(201));
}

#[test]
fn frame_range_filter_removes_matching_frame_ids() {
    let mut dataset =
        rustscan_types::TrainingDataset::new(rustgs::Intrinsics::new(1.0, 1.0, 0.0, 0.0, 1, 1));
    dataset.add_point([0.0, 0.0, 1.0], Some([1.0, 1.0, 1.0]));
    for frame_id in [75_u64, 76, 80, 93, 94] {
        dataset.add_pose(rustgs::ScenePose::new(
            frame_id,
            PathBuf::from(format!("frame_{frame_id:04}.png")),
            rustgs::SE3::identity(),
            frame_id as f64,
        ));
    }

    let ranges = parse_frame_ranges(Some("76-93")).unwrap();
    let filtered = filter_dataset_by_frame_ranges(dataset, &ranges, "test").unwrap();

    let kept: Vec<u64> = filtered.poses.iter().map(|pose| pose.frame_id).collect();
    assert_eq!(kept, vec![75, 94]);
    assert_eq!(filtered.initial_points.len(), 1);
}

#[test]
fn frame_range_include_keeps_matching_frame_ids() {
    let mut dataset =
        rustscan_types::TrainingDataset::new(rustgs::Intrinsics::new(1.0, 1.0, 0.0, 0.0, 1, 1));
    dataset.add_point([0.0, 0.0, 1.0], Some([1.0, 1.0, 1.0]));
    for frame_id in [0_u64, 30, 60, 90, 120, 150] {
        dataset.add_pose(rustgs::ScenePose::new(
            frame_id,
            PathBuf::from(format!("frame_{frame_id:04}.png")),
            rustgs::SE3::identity(),
            frame_id as f64,
        ));
    }

    let ranges = parse_frame_ranges(Some("0-60,150")).unwrap();
    let filtered = filter_dataset_to_frame_ranges(dataset, &ranges, "test").unwrap();

    let kept: Vec<u64> = filtered.poses.iter().map(|pose| pose.frame_id).collect();
    assert_eq!(kept, vec![0, 30, 60, 150]);
    assert_eq!(filtered.initial_points.len(), 1);
}

#[test]
fn frame_range_oversampling_repeats_matching_frame_ids() {
    let mut dataset =
        rustscan_types::TrainingDataset::new(rustgs::Intrinsics::new(1.0, 1.0, 0.0, 0.0, 1, 1));
    dataset.add_point([0.0, 0.0, 1.0], Some([1.0, 1.0, 1.0]));
    for frame_id in [0_u64, 30, 60, 90] {
        dataset.add_pose(rustgs::ScenePose::new(
            frame_id,
            PathBuf::from(format!("frame_{frame_id:04}.png")),
            rustgs::SE3::identity(),
            frame_id as f64,
        ));
    }

    let ranges = parse_frame_ranges(Some("30-60")).unwrap();
    let oversampled = oversample_dataset_frame_ranges(dataset, &ranges, 3, "test").unwrap();

    let kept: Vec<u64> = oversampled.poses.iter().map(|pose| pose.frame_id).collect();
    assert_eq!(kept, vec![0, 30, 60, 90, 30, 60, 30, 60]);
    assert_eq!(oversampled.initial_points.len(), 1);
}
