use crate::{
    train_command::{
        build_training_config, effective_train_args, effective_train_args_with_sources,
        evaluation_dataset_load_params, load_training_dataset_for_training,
        maybe_write_litegs_parity_report, maybe_write_litegs_parity_report_with_manifest_dir,
        TrainArgSources,
    },
    Cli, Commands, PruneSceneArgs, TrainArgs,
};
use clap::Parser;
use std::path::PathBuf;
use std::time::Duration;
use tempfile::tempdir;

fn parse_cli(args: &[&str]) -> Cli {
    Cli::try_parse_from(args).expect("cli args should parse")
}

fn parse_train_args(args: &[&str]) -> TrainArgs {
    let cli = parse_cli(args);
    let Commands::Train(args) = cli.command else {
        panic!("expected train command");
    };
    args
}

fn parse_train_args_with_sources(args: &[&str]) -> (TrainArgs, TrainArgSources) {
    let matches = <Cli as clap::CommandFactory>::command()
        .try_get_matches_from(args)
        .expect("cli args should parse");
    let sources = TrainArgSources::from_cli_matches(&matches);
    let cli =
        <Cli as clap::FromArgMatches>::from_arg_matches(&matches).expect("cli args should convert");
    let Commands::Train(args) = cli.command else {
        panic!("expected train command");
    };
    (args, sources)
}

fn parse_prune_scene_args(args: &[&str]) -> PruneSceneArgs {
    let cli = parse_cli(args);
    let Commands::PruneScene(args) = cli.command else {
        panic!("expected prune-scene command");
    };
    args
}

fn write_train_config(dir: &std::path::Path, contents: &str) -> PathBuf {
    let path = dir.join("rustgs-train.json");
    std::fs::write(&path, contents).unwrap();
    path
}

#[cfg(feature = "gpu")]
fn rgb_to_sh0_value(rgb: f32) -> f32 {
    (rgb - 0.5) / 0.282_094_8
}

#[cfg(feature = "gpu")]
fn opacity_to_logit(opacity: f32) -> f32 {
    let clamped = opacity.clamp(1e-6, 1.0 - 1e-6);
    (clamped / (1.0 - clamped)).ln()
}

#[cfg(feature = "gpu")]
fn test_splats() -> rustgs::HostSplats {
    rustgs::HostSplats::from_components(
        vec![0.0, 0.0, 0.0],
        vec![0.01f32.ln(), 0.01f32.ln(), 0.01f32.ln()],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0],
        vec![
            rgb_to_sh0_value(0.2),
            rgb_to_sh0_value(0.3),
            rgb_to_sh0_value(0.4),
        ],
        0,
    )
    .unwrap()
}

#[cfg(feature = "gpu")]
fn test_splats_metadata(splats: &rustgs::HostSplats) -> rustgs::SplatMetadata {
    rustgs::SplatMetadata {
        iterations: 1,
        final_loss: 0.0,
        gaussian_count: splats.len(),
        sh_degree: splats.sh_degree(),
    }
}

#[cfg(feature = "gpu")]
fn write_test_output_splats(path: &std::path::Path) -> rustgs::HostSplats {
    let splats = test_splats();
    rustgs::save_splats_ply(path, &splats, &test_splats_metadata(&splats)).unwrap();
    splats
}

#[cfg(feature = "gpu")]
fn convergence_fixture_input_path() -> Option<PathBuf> {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find_map(|path| {
            rustgs::resolve_litegs_parity_fixture_input_path(
                rustgs::DEFAULT_CONVERGENCE_FIXTURE_ID,
                path,
            )
        })
}

#[test]
fn train_command_parses_training_defaults() {
    let args = parse_train_args(&[
        "rustgs",
        "train",
        "--input",
        "scene.json",
        "--output",
        "scene.ply",
    ]);

    assert_eq!(args.render_scale, 0.5);
    assert_eq!(args.train_config, None);
    assert_eq!(args.train_preset, None);
    assert_eq!(args.include_frame_ranges, None);
    assert_eq!(args.exclude_frame_ranges, None);
    assert_eq!(args.oversample_frame_ranges, None);
    assert_eq!(args.oversample_frame_repeat, 1);
    assert_eq!(args.raster_cov_blur, rustgs::DEFAULT_RASTER_COV_BLUR);
    assert_eq!(args.raster_cov_blur_final, None);
    assert_eq!(args.raster_cov_blur_final_after_epoch, None);
    assert_eq!(args.litegs_sh_degree, 3);
    assert_eq!(args.litegs_profile, rustgs::LiteGsTrainingProfile::Baseline);
    assert_eq!(args.litegs_tile_size, rustgs::LiteGsTileSize::new(8, 16));
    assert!(!args.litegs_sparse_grad);
    assert_eq!(args.frame_shuffle_seed, 0);
    assert!(!args.eval_after_train);
    assert_eq!(args.eval_render_scale, 0.25);
    assert_eq!(args.eval_raster_cov_blur, None);
    assert_eq!(args.eval_max_frames, 180);
    assert_eq!(args.eval_frame_stride, 30);
    assert_eq!(args.eval_include_frame_ranges, None);
    assert_eq!(args.eval_exclude_frame_ranges, None);
    assert_eq!(args.eval_worst_frames, 5);
    assert_eq!(args.eval_device, "gpu");
    assert!(!args.eval_json);
    assert_eq!(args.eval_crop_output_dir, None);
    assert_eq!(args.eval_crop_frames, None);
    assert_eq!(args.eval_crop_rect, None);
    assert_eq!(args.loss_l1_weight, 0.8);
    assert_eq!(args.loss_ssim_weight, 0.2);
    assert_eq!(args.loss_gradient_weight, 0.0);
    assert_eq!(args.loss_robust_delta, 0.0);
    assert_eq!(args.loss_outlier_threshold, 0.0);
    assert_eq!(args.loss_outlier_weight, 1.0);
    assert_eq!(args.loss_dynamic_mask_threshold_low, 0.0);
    assert_eq!(args.loss_dynamic_mask_threshold_high, 0.0);
    assert_eq!(args.loss_dynamic_mask_min_weight, 1.0);
    assert_eq!(args.loss_dynamic_mask_start_epoch, None);
}

#[test]
fn train_command_parses_litegs_profile() {
    let args = parse_train_args(&[
        "rustgs",
        "train",
        "--input",
        "scene.json",
        "--output",
        "scene.ply",
        "--litegs-profile",
        "abs-pixel",
    ]);

    assert_eq!(args.litegs_profile, rustgs::LiteGsTrainingProfile::AbsPixel);
}

#[test]
fn train_command_parses_visibility_weight_prune_mode() {
    let args = parse_train_args(&[
        "rustgs",
        "train",
        "--input",
        "scene.json",
        "--output",
        "scene.ply",
        "--litegs-prune-mode",
        "visibility-weight",
    ]);
    let config = build_training_config(&args).unwrap();

    assert_eq!(
        config.litegs.pruning.prune_mode,
        rustgs::LiteGsPruneMode::VisibilityWeight
    );
}

#[test]
fn train_command_applies_dataset_train_preset() {
    let dir = tempdir().unwrap();
    let config_path = write_train_config(
        dir.path(),
        r#"{
  "presets": {
    "tum-prefix-compact": {
      "iterations": 8000,
      "lr_decay_iterations": 8000,
      "lr_scale_final": 0.0005,
      "lr_rotation_final": 0.0001,
      "lr_opacity_final": 0.005,
      "lr_color_final": 0.00025,
      "max_frames": 180,
      "frame_stride": 1,
      "eval_after_train": true,
      "eval_raster_cov_blur": 0.2,
      "raster_cov_blur": 0.3,
      "litegs_topology_freeze_after_epoch": 18,
      "litegs_growth_select_fraction": 0.14,
      "loss_l1_weight": 0.8,
      "loss_ssim_weight": 0.2
    }
  }
}"#,
    );
    let args = parse_train_args(&[
        "rustgs",
        "train",
        "--input",
        "scene.json",
        "--output",
        "scene.ply",
        "--train-config",
        config_path.to_str().unwrap(),
        "--train-preset",
        "tum-prefix-compact",
    ]);
    let args = effective_train_args(args);
    let config = build_training_config(&args).unwrap();

    assert_eq!(args.train_config, Some(config_path));
    assert_eq!(args.train_preset.as_deref(), Some("tum-prefix-compact"));
    assert_eq!(args.iterations, 8_000);
    assert_eq!(args.lr_decay_iterations, 8_000);
    assert_eq!(args.lr_scale_final, 0.0005);
    assert_eq!(args.lr_rotation_final, 0.0001);
    assert_eq!(args.lr_opacity_final, 0.005);
    assert_eq!(args.lr_color_final, 0.00025);
    assert_eq!(args.max_frames, 180);
    assert_eq!(args.frame_stride, 1);
    assert!(args.eval_after_train);
    assert_eq!(args.eval_raster_cov_blur, Some(0.2));
    assert_eq!(config.raster.raster_cov_blur, 0.3);
    assert_eq!(config.litegs.topology.topology_freeze_after_epoch, Some(18));
    assert_eq!(config.litegs.growth.growth_select_fraction, 0.14);
    assert_eq!(config.loss.loss_l1_weight, 0.8);
    assert_eq!(config.loss.loss_ssim_weight, 0.2);
}

#[test]
fn train_command_auto_loads_dataset_train_config_default_preset() {
    let dir = tempdir().unwrap();
    let config_path = write_train_config(
        dir.path(),
        r#"{
  "defaults": {
    "render_scale": 0.75
  },
  "default_preset": "efficient",
  "presets": {
    "efficient": {
      "iterations": 7000,
      "litegs_growth_select_fraction": 0.14,
      "loss_l1_weight": 0.9,
      "loss_ssim_weight": 0.1
    }
  }
}"#,
    );
    let args = parse_train_args(&[
        "rustgs",
        "train",
        "--input",
        dir.path().to_str().unwrap(),
        "--output",
        "scene.ply",
    ]);
    let args = effective_train_args(args);
    let config = build_training_config(&args).unwrap();

    assert_eq!(args.train_config, Some(config_path));
    assert_eq!(args.train_preset.as_deref(), Some("efficient"));
    assert_eq!(args.iterations, 7_000);
    assert_eq!(args.render_scale, 0.75);
    assert_eq!(config.litegs.growth.growth_select_fraction, 0.14);
    assert_eq!(config.loss.loss_l1_weight, 0.9);
    assert_eq!(config.loss.loss_ssim_weight, 0.1);
}

#[test]
fn train_command_applies_named_dataset_train_preset() {
    let dir = tempdir().unwrap();
    let config_path = write_train_config(
        dir.path(),
        r#"{
  "presets": {
    "quality": {
      "max_frames": 180,
      "litegs_profile": "abs_pixel",
      "litegs_growth_select_fraction": 0.25
    }
  }
}"#,
    );
    let args = parse_train_args(&[
        "rustgs",
        "train",
        "--input",
        "scene.json",
        "--output",
        "scene.ply",
        "--train-config",
        config_path.to_str().unwrap(),
        "--train-preset",
        "quality",
    ]);
    let args = effective_train_args(args);
    let config = build_training_config(&args).unwrap();

    assert_eq!(args.max_frames, 180);
    assert_eq!(
        config.litegs.features.training_profile,
        rustgs::LiteGsTrainingProfile::AbsPixel
    );
    assert_eq!(
        config.litegs.growth.split_score_mode,
        rustgs::LiteGsSplitScoreMode::AbsPixel
    );
    assert_eq!(config.litegs.growth.split_grad_threshold, 0.00001);
    assert_eq!(config.litegs.growth.growth_select_fraction, 0.25);
}

#[test]
fn train_command_applies_dataset_full_baseline_preset() {
    let dir = tempdir().unwrap();
    let config_path = write_train_config(
        dir.path(),
        r#"{
  "presets": {
    "full-baseline": {
      "max_frames": 0,
      "frame_stride": 1,
      "litegs_topology_freeze_after_epoch": 4,
      "litegs_growth_select_fraction": 0.25,
      "loss_l1_weight": 0.8,
      "loss_ssim_weight": 0.2
    }
  }
}"#,
    );
    let args = parse_train_args(&[
        "rustgs",
        "train",
        "--input",
        "scene.json",
        "--output",
        "scene.ply",
        "--train-config",
        config_path.to_str().unwrap(),
        "--train-preset",
        "full-baseline",
    ]);
    let args = effective_train_args(args);
    let config = build_training_config(&args).unwrap();

    assert_eq!(args.max_frames, 0);
    assert_eq!(args.frame_stride, 1);
    assert_eq!(config.litegs.topology.topology_freeze_after_epoch, Some(4));
    assert_eq!(config.litegs.growth.growth_select_fraction, 0.25);
    assert_eq!(config.loss.loss_l1_weight, 0.8);
    assert_eq!(config.loss.loss_ssim_weight, 0.2);
}

#[test]
fn train_command_applies_dataset_full_quality_preset() {
    let dir = tempdir().unwrap();
    let config_path = write_train_config(
        dir.path(),
        r#"{
  "presets": {
    "full-quality": {
      "max_frames": 0,
      "frame_stride": 1,
      "litegs_topology_freeze_after_epoch": 4,
      "litegs_profile": "abs-pixel",
      "loss_l1_weight": 0.8,
      "loss_ssim_weight": 0.2
    }
  }
}"#,
    );
    let args = parse_train_args(&[
        "rustgs",
        "train",
        "--input",
        "scene.json",
        "--output",
        "scene.ply",
        "--train-config",
        config_path.to_str().unwrap(),
        "--train-preset",
        "full-quality",
    ]);
    let args = effective_train_args(args);
    let config = build_training_config(&args).unwrap();

    assert_eq!(args.max_frames, 0);
    assert_eq!(args.frame_stride, 1);
    assert_eq!(config.litegs.topology.topology_freeze_after_epoch, Some(4));
    assert_eq!(
        config.litegs.features.training_profile,
        rustgs::LiteGsTrainingProfile::AbsPixel
    );
    assert_eq!(
        config.litegs.growth.split_score_mode,
        rustgs::LiteGsSplitScoreMode::AbsPixel
    );
    assert_eq!(config.litegs.growth.split_grad_threshold, 0.00001);
    assert_eq!(config.loss.loss_l1_weight, 0.8);
    assert_eq!(config.loss.loss_ssim_weight, 0.2);
}

#[test]
fn train_preset_keeps_explicit_cli_overrides() {
    let dir = tempdir().unwrap();
    let config_path = write_train_config(
        dir.path(),
        r#"{
  "presets": {
    "full-quality": {
      "iterations": 8000,
      "eval_raster_cov_blur": 0.2,
      "litegs_profile": "abs-pixel",
      "litegs_topology_freeze_after_epoch": 4
    }
  }
}"#,
    );
    let (args, sources) = parse_train_args_with_sources(&[
        "rustgs",
        "train",
        "--input",
        "scene.json",
        "--output",
        "scene.ply",
        "--train-config",
        config_path.to_str().unwrap(),
        "--train-preset",
        "full-quality",
        "--iterations",
        "1000",
        "--litegs-profile",
        "baseline",
        "--eval-raster-cov-blur",
        "0.3",
    ]);
    let args = effective_train_args_with_sources(args, &sources).unwrap();
    let config = build_training_config(&args).unwrap();

    assert_eq!(args.iterations, 1000);
    assert_eq!(args.eval_raster_cov_blur, Some(0.3));
    assert_eq!(
        config.litegs.features.training_profile,
        rustgs::LiteGsTrainingProfile::Baseline
    );
    assert_eq!(
        config.litegs.growth.split_score_mode,
        rustgs::LiteGsSplitScoreMode::Baseline
    );
    assert_eq!(config.litegs.topology.topology_freeze_after_epoch, Some(4));
}

#[test]
fn litegs_profile_applies_stable_experimental_split_defaults() {
    let args = parse_train_args(&[
        "rustgs",
        "train",
        "--input",
        "scene.json",
        "--output",
        "scene.ply",
        "--litegs-profile",
        "abs-pixel",
    ]);
    let config = build_training_config(&args).unwrap();

    assert_eq!(
        config.litegs.features.training_profile,
        rustgs::LiteGsTrainingProfile::AbsPixel
    );
    assert_eq!(
        config.litegs.growth.split_score_mode,
        rustgs::LiteGsSplitScoreMode::AbsPixel
    );
    assert_eq!(config.litegs.growth.split_grad_threshold, 0.00001);
}

#[test]
fn prune_scene_command_parses_cleanup_flags() {
    let args = parse_prune_scene_args(&[
        "rustgs",
        "prune-scene",
        "--input",
        "in.ply",
        "--output",
        "out.ply",
        "--min-opacity",
        "0.01",
        "--min-scale",
        "0.0001",
        "--max-scale",
        "0.2",
        "--max-distance-extent-multiplier",
        "2.5",
        "--dry-run",
        "--json",
    ]);

    assert_eq!(args.input, PathBuf::from("in.ply"));
    assert_eq!(args.output, PathBuf::from("out.ply"));
    assert_eq!(args.min_opacity, 0.01);
    assert_eq!(args.min_scale, 0.0001);
    assert_eq!(args.max_scale, 0.2);
    assert_eq!(args.max_distance_extent_multiplier, 2.5);
    assert!(args.dry_run);
    assert!(args.json);
}

#[cfg(feature = "gpu")]
#[test]
fn prune_splats_filters_low_opacity_and_far_outliers() {
    let splats = rustgs::HostSplats::from_components(
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 20.0, 0.0, 0.0],
        vec![0.01f32.ln(); 9],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        vec![
            opacity_to_logit(0.5),
            opacity_to_logit(0.001),
            opacity_to_logit(0.5),
        ],
        vec![
            rgb_to_sh0_value(0.2),
            rgb_to_sh0_value(0.3),
            rgb_to_sh0_value(0.4),
            rgb_to_sh0_value(0.5),
            rgb_to_sh0_value(0.6),
            rgb_to_sh0_value(0.7),
            rgb_to_sh0_value(0.8),
            rgb_to_sh0_value(0.7),
            rgb_to_sh0_value(0.6),
        ],
        0,
    )
    .unwrap();
    let args = PruneSceneArgs {
        input: PathBuf::from("in.ply"),
        output: PathBuf::from("out.ply"),
        min_opacity: 0.01,
        min_scale: 0.0,
        max_scale: 0.0,
        max_distance_from_center: 8.0,
        max_distance_extent_multiplier: 0.0,
        dry_run: false,
        json: false,
        log_level: "error".to_string(),
    };

    let (pruned, summary) = crate::prune_splats(&splats, &args).unwrap();

    assert_eq!(pruned.len(), 1);
    assert_eq!(summary.input_gaussians, 3);
    assert_eq!(summary.output_gaussians, 1);
    assert_eq!(summary.reasons.opacity, 1);
    assert_eq!(summary.reasons.distance, 1);
}

#[test]
fn train_command_rejects_removed_chunked_flag() {
    let err = Cli::try_parse_from([
        "rustgs",
        "train",
        "--input",
        "scene.json",
        "--output",
        "scene.ply",
        "--chunked-training",
    ])
    .expect_err("removed chunked flag should fail");

    assert!(err.to_string().contains("--chunked-training"));
}

#[test]
fn train_command_rejects_removed_chunk_size_flag() {
    let err = Cli::try_parse_from([
        "rustgs",
        "train",
        "--input",
        "scene.json",
        "--output",
        "scene.ply",
        "--metal-gaussian-chunk-size",
        "64",
    ])
    .expect_err("removed chunk-size flag should fail");

    assert!(err.to_string().contains("--metal-gaussian-chunk-size"));
}

#[test]
fn train_command_rejects_removed_litegs_mode_flag() {
    let err = Cli::try_parse_from([
        "rustgs",
        "train",
        "--input",
        "scene.json",
        "--output",
        "scene.ply",
        "--litegs-mode",
    ])
    .expect_err("litegs mode flag should fail because LiteGS is now mandatory");

    assert!(err.to_string().contains("--litegs-mode"));
}

#[test]
fn train_command_rejects_removed_prune_interval_flag() {
    let err = Cli::try_parse_from([
        "rustgs",
        "train",
        "--input",
        "scene.json",
        "--output",
        "scene.ply",
        "--prune-interval",
        "100",
    ])
    .expect_err("prune interval flag should fail because LiteGS refine cadence is mandatory");

    assert!(err.to_string().contains("--prune-interval"));
}

#[test]
fn train_command_rejects_removed_topology_warmup_flag() {
    let err = Cli::try_parse_from([
        "rustgs",
        "train",
        "--input",
        "scene.json",
        "--output",
        "scene.ply",
        "--topology-warmup",
        "100",
    ])
    .expect_err("topology warmup flag should fail because LiteGS topology controls are nested");

    assert!(err.to_string().contains("--topology-warmup"));
}

#[test]
fn train_command_rejects_removed_topology_log_interval_flag() {
    let err = Cli::try_parse_from([
        "rustgs",
        "train",
        "--input",
        "scene.json",
        "--output",
        "scene.ply",
        "--topology-log-interval",
        "500",
    ])
    .expect_err("topology log interval flag should fail because the legacy scheduler is removed");

    assert!(err.to_string().contains("--topology-log-interval"));
}

#[test]
fn train_command_parses_post_training_eval_flags() {
    let args = parse_train_args(&[
        "rustgs",
        "train",
        "--input",
        "scene.json",
        "--output",
        "scene.ply",
        "--include-frame-ranges",
        "0-179",
        "--oversample-frame-ranges",
        "0-179",
        "--oversample-frame-repeat",
        "3",
        "--eval-after-train",
        "--eval-render-scale",
        "0.125",
        "--eval-raster-cov-blur",
        "0.2",
        "--eval-max-frames",
        "60",
        "--eval-frame-stride",
        "10",
        "--eval-include-frame-ranges",
        "0-179,240-299",
        "--eval-exclude-frame-ranges",
        "76-93,155",
        "--eval-worst-frames",
        "3",
        "--eval-device",
        "gpu",
        "--eval-json",
        "--eval-crop-output-dir",
        "crops",
        "--eval-crop-frames",
        "0,90,120",
        "--eval-crop-rect",
        "16,12,64,48",
        "--loss-dynamic-mask-threshold-low",
        "0.12",
        "--loss-dynamic-mask-threshold-high",
        "0.32",
        "--loss-dynamic-mask-min-weight",
        "0.35",
        "--loss-dynamic-mask-start-epoch",
        "18",
    ]);

    assert!(args.eval_after_train);
    assert_eq!(args.include_frame_ranges.as_deref(), Some("0-179"));
    assert_eq!(args.oversample_frame_ranges.as_deref(), Some("0-179"));
    assert_eq!(args.oversample_frame_repeat, 3);
    assert_eq!(args.eval_render_scale, 0.125);
    assert_eq!(args.eval_raster_cov_blur, Some(0.2));
    assert_eq!(args.eval_max_frames, 60);
    assert_eq!(args.eval_frame_stride, 10);
    assert_eq!(
        args.eval_include_frame_ranges.as_deref(),
        Some("0-179,240-299")
    );
    assert_eq!(args.eval_exclude_frame_ranges.as_deref(), Some("76-93,155"));
    assert_eq!(args.eval_worst_frames, 3);
    assert_eq!(args.eval_device, "gpu");
    assert!(args.eval_json);
    assert_eq!(args.eval_crop_output_dir, Some(PathBuf::from("crops")));
    assert_eq!(args.eval_crop_frames.as_deref(), Some("0,90,120"));
    assert_eq!(args.eval_crop_rect.as_deref(), Some("16,12,64,48"));
    assert_eq!(args.loss_dynamic_mask_threshold_low, 0.12);
    assert_eq!(args.loss_dynamic_mask_threshold_high, 0.32);
    assert_eq!(args.loss_dynamic_mask_min_weight, 0.35);
    assert_eq!(args.loss_dynamic_mask_start_epoch, Some(18));
}

#[test]
fn post_training_eval_loads_prefix_without_double_applying_stride() {
    let args = parse_train_args(&[
        "rustgs",
        "train",
        "--input",
        "scene.json",
        "--output",
        "scene.ply",
        "--eval-after-train",
        "--eval-max-frames",
        "180",
        "--eval-frame-stride",
        "30",
    ]);

    let (dataset_max_frames, dataset_frame_stride) = evaluation_dataset_load_params(&args);
    assert_eq!(dataset_max_frames, 180);
    assert_eq!(dataset_frame_stride, 1);
}

#[test]
fn train_command_parses_litegs_flags_and_builds_nested_config() {
    let args = parse_train_args(&[
        "rustgs",
        "train",
        "--input",
        "scene.json",
        "--output",
        "scene.ply",
        "--frame-shuffle-seed",
        "42",
        "--litegs-sh-degree",
        "4",
        "--litegs-tile-size",
        "16x16",
        "--litegs-sparse-grad",
        "--litegs-reg-weight",
        "0.01",
        "--litegs-enable-transmittance",
        "--litegs-enable-depth",
        "--litegs-densify-from",
        "6",
        "--litegs-densify-until",
        "24",
        "--litegs-topology-freeze-after-epoch",
        "18",
        "--litegs-growth-freeze-after-epoch",
        "9",
        "--litegs-refine-every",
        "120",
        "--litegs-densification-interval",
        "8",
        "--litegs-growth-grad-threshold",
        "0.0003",
        "--litegs-growth-select-fraction",
        "0.35",
        "--litegs-growth-stop-iter",
        "2400",
        "--litegs-opacity-decay",
        "0.003",
        "--litegs-scale-decay",
        "0.0015",
        "--litegs-opacity-reset-interval",
        "12",
        "--litegs-opacity-reset-mode",
        "reset",
        "--litegs-prune-mode",
        "threshold",
        "--litegs-prune-opacity-threshold",
        "0.01",
        "--litegs-prune-visibility-dry-run",
        "--litegs-prune-visibility-threshold",
        "0.07",
        "--litegs-prune-high-opacity-threshold",
        "0.2",
        "--litegs-prune-until-epoch",
        "60",
        "--litegs-target-primitives",
        "200000",
        "--litegs-lr-pose",
        "0.0002",
        "--lr-decay-iterations",
        "10000",
        "--lr-scale-final",
        "0.0005",
        "--lr-rotation-final",
        "0.0001",
        "--lr-opacity-final",
        "0.005",
        "--lr-color-final",
        "0.00025",
    ]);
    let config = build_training_config(&args).unwrap();

    assert_eq!(config.litegs.rendering.sh_degree, 4);
    assert_eq!(
        config.litegs.rendering.tile_size,
        rustgs::LiteGsTileSize::new(16, 16)
    );
    assert!(config.litegs.features.sparse_grad);
    assert_eq!(config.litegs.features.reg_weight, 0.01);
    assert!(config.litegs.features.enable_transmittance);
    assert!(config.litegs.features.enable_depth);
    assert_eq!(config.litegs.topology.densify_from, 6);
    assert_eq!(config.litegs.topology.densify_until, Some(24));
    assert_eq!(config.litegs.topology.topology_freeze_after_epoch, Some(18));
    assert_eq!(config.litegs.topology.growth_freeze_after_epoch, Some(9));
    assert_eq!(config.litegs.topology.refine_every, 120);
    assert_eq!(config.litegs.topology.densification_interval, 8);
    assert_eq!(config.litegs.growth.growth_grad_threshold, 0.0003);
    assert_eq!(config.litegs.growth.growth_select_fraction, 0.35);
    assert_eq!(config.litegs.growth.growth_stop_iter, 2_400);
    assert_eq!(config.litegs.refine.opacity_decay, 0.003);
    assert_eq!(config.litegs.refine.scale_decay, 0.0015);
    assert_eq!(config.litegs.topology.opacity_reset_interval, 12);
    assert_eq!(
        config.litegs.topology.opacity_reset_mode,
        rustgs::LiteGsOpacityResetMode::Reset
    );
    assert_eq!(
        config.litegs.pruning.prune_mode,
        rustgs::LiteGsPruneMode::Threshold
    );
    assert_eq!(config.litegs.pruning.prune_offset_epochs, 0); // default value
    assert_eq!(config.litegs.pruning.prune_min_age, 5); // default value
    assert_eq!(config.litegs.pruning.prune_invisible_epochs, 10); // default value
    assert_eq!(config.litegs.pruning.prune_opacity_threshold, 0.01);
    assert!(config.litegs.pruning.prune_visibility_dry_run);
    assert_eq!(config.litegs.pruning.prune_visibility_threshold, 0.07);
    assert_eq!(config.litegs.pruning.prune_high_opacity_threshold, 0.2);
    assert_eq!(config.litegs.pruning.prune_until_epoch, Some(60));
    assert_eq!(config.litegs.topology.target_primitives, 200_000);
    assert!(!config.litegs.camera.learnable_viewproj);
    assert_eq!(config.litegs.camera.lr_pose, 0.0002);
    assert_eq!(config.data.frame_shuffle_seed, 42);
    assert_eq!(config.optimizer.lr_decay_iterations, Some(10_000));
    assert_eq!(config.optimizer.lr_scale_final, 0.0005);
    assert_eq!(config.optimizer.lr_rotation_final, 0.0001);
    assert_eq!(config.optimizer.lr_opacity_final, 0.005);
    assert_eq!(config.optimizer.lr_color_final, 0.00025);
}

#[test]
fn train_command_rejects_unimplemented_learnable_viewproj() {
    let args = parse_train_args(&[
        "rustgs",
        "train",
        "--input",
        "scene.json",
        "--output",
        "scene.ply",
        "--litegs-learnable-viewproj",
    ]);

    let err = build_training_config(&args)
        .expect_err("learnable viewproj should be rejected until trainer support exists");
    assert!(
        err.to_string().contains("not implemented"),
        "unexpected error: {err}"
    );
}

#[test]
fn train_command_builds_dynamic_mask_config() {
    let args = parse_train_args(&[
        "rustgs",
        "train",
        "--input",
        "scene.json",
        "--output",
        "scene.ply",
        "--loss-dynamic-mask-threshold-low",
        "0.12",
        "--loss-dynamic-mask-threshold-high",
        "0.32",
        "--loss-dynamic-mask-min-weight",
        "0.35",
    ]);
    let config = build_training_config(&args).unwrap();

    assert_eq!(config.loss.loss_dynamic_mask_threshold_low, 0.12);
    assert_eq!(config.loss.loss_dynamic_mask_threshold_high, 0.32);
    assert_eq!(config.loss.loss_dynamic_mask_min_weight, 0.35);
    assert_eq!(config.loss.loss_dynamic_mask_start_epoch, None);
}

#[test]
fn train_preset_preserves_dynamic_mask_override() {
    let dir = tempdir().unwrap();
    let config_path = write_train_config(
        dir.path(),
        r#"{
  "presets": {
    "compact": {
      "max_frames": 180,
      "litegs_growth_select_fraction": 0.14,
      "loss_dynamic_mask_threshold_low": 0.0,
      "loss_dynamic_mask_threshold_high": 0.0,
      "loss_dynamic_mask_min_weight": 1.0
    }
  }
}"#,
    );
    let (args, sources) = parse_train_args_with_sources(&[
        "rustgs",
        "train",
        "--input",
        "scene.json",
        "--output",
        "scene.ply",
        "--train-config",
        config_path.to_str().unwrap(),
        "--train-preset",
        "compact",
        "--loss-dynamic-mask-threshold-low",
        "0.12",
        "--loss-dynamic-mask-threshold-high",
        "0.32",
        "--loss-dynamic-mask-min-weight",
        "0.35",
    ]);
    let args = effective_train_args_with_sources(args, &sources).unwrap();
    let config = build_training_config(&args).unwrap();

    assert_eq!(args.max_frames, 180);
    assert_eq!(config.litegs.growth.growth_select_fraction, 0.14);
    assert_eq!(config.loss.loss_dynamic_mask_threshold_low, 0.12);
    assert_eq!(config.loss.loss_dynamic_mask_threshold_high, 0.32);
    assert_eq!(config.loss.loss_dynamic_mask_min_weight, 0.35);
}

#[test]
fn litegs_parity_report_is_written_next_to_output_scene() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("scene.ply");
    let input = std::path::Path::new("test_data/tum/rgbd_dataset_freiburg1_xyz");

    let splats = write_test_output_splats(&output);

    let mut dataset = rustgs::TrainingDataset::new(rustgs::Intrinsics::from_focal(500.0, 32, 32));
    dataset.add_point([0.0, 0.0, 0.0], None);
    dataset.add_point([1.0, 0.0, 0.0], None);
    let config = rustgs::TrainingConfig::default();

    maybe_write_litegs_parity_report(
        input,
        &output,
        &dataset,
        &splats,
        &config,
        None,
        Duration::from_millis(42),
        None,
    )
    .unwrap();

    let report_path = rustgs::default_parity_report_path(&output);
    assert!(report_path.exists());
    let report = rustgs::ParityHarnessReport::load_json(&report_path).unwrap();
    assert_eq!(report.fixture_id, rustgs::DEFAULT_CONVERGENCE_FIXTURE_ID);
    assert_eq!(report.topology.initialization_gaussians, Some(2));
    assert_eq!(report.topology.final_gaussians, Some(1));
    assert_eq!(report.topology.export_outputs, 1);
    assert_eq!(report.metrics.active_sh_degree, Some(3));
    assert_eq!(report.metrics.rotation_frozen, None);
    assert!(report.metrics.export_roundtrip_ok);
    assert_eq!(report.timing.training_ms, Some(42));
    assert_eq!(
        report.gate.as_ref().map(|gate| &gate.status),
        Some(&rustgs::ParityGateStatus::MissingReference)
    );
    assert!(report
        .notes
        .iter()
        .any(|note| note.contains("No checked-in LiteGS parity reference report was found")));
}

#[test]
fn litegs_parity_report_uses_runtime_telemetry_when_available() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("scene.ply");
    let input = std::path::Path::new("test_data/tum/rgbd_dataset_freiburg1_xyz");

    let splats = write_test_output_splats(&output);

    let mut dataset = rustgs::TrainingDataset::new(rustgs::Intrinsics::from_focal(500.0, 32, 32));
    dataset.add_point([0.0, 0.0, 0.0], None);
    let config = rustgs::TrainingConfig::default();
    let telemetry = rustgs::LiteGsTrainingTelemetry {
        loss_terms: rustgs::ParityLossTerms {
            l1: Some(0.1),
            ssim: Some(0.2),
            scale_regularization: Some(0.3),
            transmittance: Some(0.4),
            depth: None,
            total: Some(0.5),
        },
        loss_curve_samples: vec![rustgs::ParityLossCurveSample {
            iteration: 25,
            frame_idx: 1,
            l1: Some(0.1),
            ssim: Some(0.2),
            depth: None,
            total: Some(0.5),
            depth_valid_pixels: None,
        }],
        topology: rustgs::ParityTopologyMetrics {
            final_gaussians: Some(7),
            densify_events: 2,
            densify_added: 5,
            prune_events: 1,
            prune_removed: 3,
            opacity_reset_events: 4,
            ..Default::default()
        },
        active_sh_degree: Some(2),
        final_loss: Some(0.45),
        final_step_loss: Some(0.5),
        depth_valid_pixels: None,
        depth_grad_scale: None,
        rotation_frozen: true,
        learning_rates: rustgs::LiteGsOptimizerLrs::default(),
    };

    maybe_write_litegs_parity_report(
        input,
        &output,
        &dataset,
        &splats,
        &config,
        Some(&telemetry),
        Duration::from_millis(42),
        None,
    )
    .unwrap();

    let report_path = rustgs::default_parity_report_path(&output);
    let report = rustgs::ParityHarnessReport::load_json(&report_path).unwrap();
    assert_eq!(report.loss_terms.total, Some(0.5));
    assert_eq!(report.topology.densify_events, 2);
    assert_eq!(report.topology.densify_added, 5);
    assert_eq!(report.topology.prune_removed, 3);
    assert_eq!(report.metrics.active_sh_degree, Some(2));
    assert_eq!(report.metrics.depth_valid_pixels, None);
    assert_eq!(report.metrics.depth_grad_scale, None);
    assert_eq!(report.loss_curve_samples.len(), 1);
    assert_eq!(report.loss_curve_samples[0].iteration, 25);
    assert_eq!(report.metrics.rotation_frozen, Some(true));
    assert_eq!(report.topology.final_gaussians, Some(7));
    assert_eq!(
        report.gate.as_ref().map(|gate| &gate.status),
        Some(&rustgs::ParityGateStatus::MissingReference)
    );
}

#[test]
fn litegs_parity_report_records_final_psnr_from_evaluation_summary() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("scene.ply");
    let input = std::path::Path::new("test_data/tum/rgbd_dataset_freiburg1_xyz");

    let splats = write_test_output_splats(&output);

    let mut dataset = rustgs::TrainingDataset::new(rustgs::Intrinsics::from_focal(500.0, 32, 32));
    dataset.add_point([0.0, 0.0, 0.0], None);
    let config = rustgs::TrainingConfig::default();
    let evaluation_summary = rustgs::SplatEvaluationSummary {
        device: rustgs::EvaluationDevice::Gpu,
        render_scale: 0.25,
        raster_cov_blur: rustgs::DEFAULT_RASTER_COV_BLUR,
        render_width: 16,
        render_height: 16,
        frame_stride: 30,
        max_frames: 180,
        frame_count: 6,
        splat_iterations: 1,
        splat_count: 1,
        final_loss: 0.4,
        final_step_loss: Some(0.35),
        elapsed_seconds: 1.2,
        psnr_mean_db: 7.25,
        psnr_median_db: 7.10,
        psnr_min_db: 6.8,
        psnr_max_db: 7.6,
        psnr_std_db: 0.3,
        sharpness_grad_ratio_mean: 0.8,
        sharpness_lap_ratio_mean: 0.6,
        worst_frames: Vec::new(),
        crop_outputs: Vec::new(),
    };

    maybe_write_litegs_parity_report(
        input,
        &output,
        &dataset,
        &splats,
        &config,
        None,
        Duration::from_millis(42),
        Some(&evaluation_summary),
    )
    .unwrap();

    let report =
        rustgs::ParityHarnessReport::load_json(&rustgs::default_parity_report_path(&output))
            .unwrap();
    assert_eq!(report.metrics.final_psnr, Some(7.25));
    assert!(report
        .notes
        .iter()
        .any(|note| note.contains("mean PSNR 7.2500 dB")));
}

#[test]
fn litegs_parity_report_persists_depth_and_sparse_cluster_config() {
    let dir = tempdir().unwrap();
    let output = dir.path().join("scene.ply");
    let input = std::path::Path::new("test_data/tum/rgbd_dataset_freiburg1_xyz");

    let splats = write_test_output_splats(&output);

    let mut dataset = rustgs::TrainingDataset::new(rustgs::Intrinsics::from_focal(500.0, 32, 32));
    dataset.add_point([0.0, 0.0, 0.0], None);
    let config = rustgs::TrainingConfig {
        litegs: rustgs::LiteGsConfig {
            features: rustgs::LiteGsFeatureConfig {
                sparse_grad: true,
                enable_depth: true,
                ..rustgs::LiteGsFeatureConfig::default()
            },
            ..rustgs::LiteGsConfig::default()
        },
        ..rustgs::TrainingConfig::default()
    };
    let telemetry = rustgs::LiteGsTrainingTelemetry {
        loss_terms: rustgs::ParityLossTerms {
            l1: Some(0.1),
            ssim: Some(0.2),
            scale_regularization: Some(0.3),
            transmittance: Some(0.4),
            depth: Some(0.6),
            total: Some(0.7),
        },
        loss_curve_samples: vec![rustgs::ParityLossCurveSample {
            iteration: 50,
            frame_idx: 2,
            l1: Some(0.1),
            ssim: Some(0.2),
            depth: Some(0.6),
            total: Some(0.7),
            depth_valid_pixels: Some(256),
        }],
        topology: rustgs::ParityTopologyMetrics {
            final_gaussians: Some(3),
            ..Default::default()
        },
        active_sh_degree: Some(2),
        final_loss: Some(0.65),
        final_step_loss: Some(0.7),
        depth_valid_pixels: Some(256),
        depth_grad_scale: Some(0.1 / 256.0),
        rotation_frozen: true,
        learning_rates: rustgs::LiteGsOptimizerLrs::default(),
    };

    maybe_write_litegs_parity_report(
        input,
        &output,
        &dataset,
        &splats,
        &config,
        Some(&telemetry),
        Duration::from_millis(42),
        None,
    )
    .unwrap();

    let report_path = rustgs::default_parity_report_path(&output);
    let report = rustgs::ParityHarnessReport::load_json(&report_path).unwrap();
    assert!(report.litegs.features.sparse_grad);
    assert!(report.litegs.features.enable_depth);
    assert_eq!(report.loss_terms.depth, Some(0.6));
    assert_eq!(report.loss_terms.total, Some(0.7));
    assert_eq!(report.metrics.depth_valid_pixels, Some(256));
    assert_eq!(report.metrics.depth_grad_scale, Some(0.1 / 256.0));
    assert_eq!(report.loss_curve_samples.len(), 1);
    assert_eq!(report.loss_curve_samples[0].depth, Some(0.6));
    assert_eq!(
        report.gate.as_ref().map(|gate| &gate.status),
        Some(&rustgs::ParityGateStatus::MissingReference)
    );
}

#[test]
fn litegs_parity_report_populates_reference_comparison_from_workspace_fixture() {
    let dir = tempdir().unwrap();
    let workspace_root = dir.path().join("workspace");
    let manifest_dir = workspace_root.join("RustGS");
    std::fs::create_dir_all(&manifest_dir).unwrap();

    let reference_path =
        workspace_root.join("test_data/fixtures/litegs/colmap-small/parity-reference.json");
    std::fs::create_dir_all(reference_path.parent().unwrap()).unwrap();

    let mut reference_report = rustgs::ParityHarnessReport::new(
        rustgs::DEFAULT_CONVERGENCE_FIXTURE_ID,
        &rustgs::LiteGsConfig::default(),
    );
    reference_report.loss_curve_samples = vec![
        rustgs::ParityLossCurveSample {
            iteration: 0,
            frame_idx: 0,
            depth: Some(0.55),
            total: Some(1.1),
            depth_valid_pixels: Some(128),
            ..Default::default()
        },
        rustgs::ParityLossCurveSample {
            iteration: 25,
            frame_idx: 1,
            depth: Some(0.35),
            total: Some(0.8),
            depth_valid_pixels: Some(96),
            ..Default::default()
        },
    ];
    reference_report.topology.final_gaussians = Some(1);
    reference_report.save_json(&reference_path).unwrap();

    let output = dir.path().join("scene.ply");
    let input = std::path::Path::new("test_data/tum/rgbd_dataset_freiburg1_xyz");
    let splats = write_test_output_splats(&output);

    let mut dataset = rustgs::TrainingDataset::new(rustgs::Intrinsics::from_focal(500.0, 32, 32));
    dataset.add_point([0.0, 0.0, 0.0], None);
    let config = rustgs::TrainingConfig::default();
    let telemetry = rustgs::LiteGsTrainingTelemetry {
        loss_terms: rustgs::ParityLossTerms {
            depth: Some(0.4),
            total: Some(0.9),
            ..Default::default()
        },
        loss_curve_samples: vec![
            rustgs::ParityLossCurveSample {
                iteration: 0,
                frame_idx: 0,
                depth: Some(0.5),
                total: Some(1.0),
                depth_valid_pixels: Some(128),
                ..Default::default()
            },
            rustgs::ParityLossCurveSample {
                iteration: 25,
                frame_idx: 1,
                depth: Some(0.3),
                total: Some(0.7),
                depth_valid_pixels: Some(96),
                ..Default::default()
            },
        ],
        topology: rustgs::ParityTopologyMetrics {
            final_gaussians: Some(1),
            ..Default::default()
        },
        active_sh_degree: Some(3),
        final_loss: Some(0.75),
        final_step_loss: Some(0.7),
        depth_valid_pixels: Some(96),
        depth_grad_scale: Some(0.1 / 96.0),
        rotation_frozen: true,
        learning_rates: rustgs::LiteGsOptimizerLrs::default(),
    };

    maybe_write_litegs_parity_report_with_manifest_dir(
        input,
        &output,
        &dataset,
        &splats,
        &config,
        Some(&telemetry),
        Duration::from_millis(42),
        None,
        &manifest_dir,
    )
    .unwrap();

    let report_path = rustgs::default_parity_report_path(&output);
    let report = rustgs::ParityHarnessReport::load_json(&report_path).unwrap();
    let comparison = report
        .reference_comparison
        .expect("reference comparison should be populated");
    assert_eq!(comparison.compared_iterations, 2);
    assert_eq!(comparison.compared_depth_samples, 2);
    assert_eq!(comparison.compared_total_samples, 2);
    assert!((comparison.depth_mean_abs_delta.unwrap() - 0.05).abs() < 1e-6);
    assert!((comparison.depth_max_abs_delta.unwrap() - 0.05).abs() < 1e-6);
    assert!((comparison.total_mean_abs_delta.unwrap() - 0.1).abs() < 1e-6);
    assert!((comparison.total_max_abs_delta.unwrap() - 0.1).abs() < 1e-6);
    assert_eq!(report.metrics.gaussian_count_delta_ratio, Some(0.0));
    assert_eq!(
        report.gate.as_ref().map(|gate| &gate.status),
        Some(&rustgs::ParityGateStatus::Passed)
    );
    assert!(report
        .notes
        .iter()
        .any(|note| note.contains("Compared parity loss curve samples against reference report")));
}

#[cfg(feature = "gpu")]
#[test]
fn litegs_fixture_parity_regression_writes_report_from_real_training_run() {
    let Some(input) = convergence_fixture_input_path() else {
        eprintln!("skipping test: no LiteGS parity fixture was available in this workspace");
        return;
    };
    if !rustgs::gpu_available() {
        eprintln!("skipping test: GPU training is unavailable in this environment");
        return;
    }

    let max_frames = 90;
    let frame_stride = 30;
    let config = rustgs::TrainingConfig {
        iterations: 1,
        initialization: rustgs::TrainingInitializationConfig {
            max_initial_gaussians: 2048,
            ..rustgs::TrainingInitializationConfig::default()
        },
        raster: rustgs::TrainingRasterConfig {
            render_scale: 0.5,
            ..rustgs::TrainingRasterConfig::default()
        },
        litegs: rustgs::LiteGsConfig {
            features: rustgs::LiteGsFeatureConfig {
                sparse_grad: true,
                enable_depth: true,
                ..rustgs::LiteGsFeatureConfig::default()
            },
            ..rustgs::LiteGsConfig::default()
        },
        ..rustgs::TrainingConfig::default()
    };
    let output_dir = tempdir().unwrap();
    let output = output_dir.path().join("fixture-scene.ply");
    let Ok((dataset, source)) =
        load_training_dataset_for_training(&input, max_frames, frame_stride)
    else {
        eprintln!(
            "skipping test: could not load LiteGS convergence fixture at {:?}",
            input
        );
        return;
    };
    if dataset.initial_points.is_empty() {
        eprintln!(
            "skipping test: resolved {:?} as {} without sparse points; training now requires COLMAP sparse initialization",
            input, source
        );
        return;
    }

    let training_run =
        rustgs::train_splats(&dataset, &config, rustgs::TrainingOptions::default()).unwrap();
    let training_elapsed = training_run.report.elapsed;
    let final_loss = training_run.report.metadata_final_loss_or(0.0);
    let training_telemetry = training_run.report.telemetry.clone();
    let splats = training_run.splats;

    rustgs::save_splats_ply(
        &output,
        &splats,
        &rustgs::SplatMetadata {
            iterations: config.iterations,
            final_loss,
            gaussian_count: splats.len(),
            sh_degree: splats.sh_degree(),
        },
    )
    .unwrap();

    maybe_write_litegs_parity_report(
        &input,
        &output,
        &dataset,
        &splats,
        &config,
        training_telemetry.as_ref(),
        training_elapsed,
        None,
    )
    .unwrap();

    let report_path = rustgs::default_parity_report_path(&output);
    let report = rustgs::ParityHarnessReport::load_json(&report_path).unwrap();

    assert_eq!(report.fixture_id, rustgs::DEFAULT_CONVERGENCE_FIXTURE_ID);
    assert!(report.litegs.features.sparse_grad);
    assert!(report.litegs.features.enable_depth);
    assert_eq!(report.topology.export_outputs, 1);
    assert_eq!(report.topology.final_gaussians, Some(splats.len()));
    assert!(report.loss_terms.total.unwrap_or(0.0) > 0.0);
    assert!(report.loss_terms.depth.is_some());
    assert!(!report.loss_curve_samples.is_empty());
    assert!(report
        .loss_curve_samples
        .iter()
        .any(|sample| sample.depth.is_some()));
    assert!(report.metrics.depth_valid_pixels.unwrap_or(0) > 0);
    assert!(report.metrics.depth_grad_scale.unwrap_or(0.0) > 0.0);
    assert!(!report.metrics.had_nan);
    assert!(!report.metrics.had_oom);
    assert!(report.metrics.export_roundtrip_ok);
    assert!(report.timing.training_ms.unwrap_or(0) > 0);
    assert!(report
        .notes
        .iter()
        .all(|note| !note.contains("frame-based fallback")));
}
