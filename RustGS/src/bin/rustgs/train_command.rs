#![allow(clippy::too_many_arguments)]

use crate::TrainArgs;
use anyhow::{bail, Context};
use clap::parser::ValueSource;
use serde::{de, Deserialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;

#[cfg(feature = "gpu")]
pub(super) fn run_train_command(args: TrainArgs, sources: TrainArgSources) -> anyhow::Result<()> {
    let args = effective_train_args_with_sources(args, &sources)?;
    env_logger::Builder::new()
        .parse_filters(&args.log_level)
        .init();

    log::info!("Training 3DGS splats from {:?}", args.input);
    log::info!("Output: {:?}", args.output);
    log::info!("Iterations: {}", args.iterations);
    log::info!("Backend: wgpu");
    if args.sampling_step != 0 {
        log::warn!(
            "--sampling-step={} is ignored because training now initializes strictly from dataset sparse points",
            args.sampling_step
        );
    }
    if let Some(config_path) = &args.train_config {
        log::info!("Loaded RustGS train config: {}", config_path.display());
    }
    if let Some(preset) = &args.train_preset {
        log::info!("Applied RustGS train preset: {}", preset);
    }

    let (dataset, source) = load_training_dataset_for_training(
        &args.input,
        args.image_root.as_deref(),
        args.max_frames,
        args.frame_stride,
    )?;
    let included_training_ranges = parse_frame_ranges(args.include_frame_ranges.as_deref())?;
    let dataset = filter_dataset_to_frame_ranges(dataset, &included_training_ranges, "training")?;
    let excluded_training_ranges = parse_frame_ranges(args.exclude_frame_ranges.as_deref())?;
    let dataset = filter_dataset_by_frame_ranges(dataset, &excluded_training_ranges, "training")?;
    let oversample_training_ranges = parse_frame_ranges(args.oversample_frame_ranges.as_deref())?;
    let dataset = oversample_dataset_frame_ranges(
        dataset,
        &oversample_training_ranges,
        args.oversample_frame_repeat,
        "training",
    )?;
    log::info!(
        "Loaded {} poses, {} initialization points",
        dataset.poses.len(),
        dataset.initial_points.len()
    );
    ensure_sparse_initialization_points(&dataset, source, &args.input)?;

    let config = build_training_config(&args)?;
    log::info!("Frame shuffle seed: {}", config.data.frame_shuffle_seed);
    log_litegs_training_config(&config);

    let training_run = rustgs::train_splats(&dataset, &config, rustgs::TrainingOptions::default())?;
    let rustgs::TrainingRun {
        splats,
        report: training_report,
    } = training_run;
    let training_telemetry = training_report.telemetry.as_ref();

    log::info!(
        "CLI training summary | route=standard | final_gaussians={} | elapsed={:.2}s",
        training_report.gaussian_count,
        training_report.elapsed.as_secs_f64(),
    );
    log::info!("Trained {} Gaussians", splats.len());

    let metadata = rustgs::SplatMetadata {
        iterations: config.iterations,
        final_loss: training_report.metadata_final_loss_or(0.0),
        gaussian_count: splats.len(),
        sh_degree: splats.sh_degree(),
    };
    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create output directory {}", parent.display())
            })?;
        }
    }
    rustgs::save_splats(&args.output, &splats, &metadata)?;
    log::info!("Saved scene to {:?}", args.output);

    let evaluation_summary =
        maybe_evaluate_trained_splats(&args, &splats, &metadata, training_telemetry)?;

    if let Err(err) = maybe_write_litegs_parity_report(
        &args.input,
        &args.output,
        &dataset,
        &splats,
        &metadata,
        &config,
        training_telemetry,
        training_report.training_loop_elapsed,
        training_report.elapsed,
        evaluation_summary.as_ref(),
    ) {
        log::warn!("failed to persist LiteGS parity report: {err}");
    }

    Ok(())
}

#[cfg(not(feature = "gpu"))]
pub(super) fn run_train_command(args: TrainArgs, sources: TrainArgSources) -> anyhow::Result<()> {
    let args = effective_train_args_with_sources(args, &sources)?;
    env_logger::Builder::new()
        .parse_filters(&args.log_level)
        .init();
    log::error!("GPU feature is required for training. Rebuild with --features gpu");
    std::process::exit(1);
}

pub(super) fn effective_train_args_with_sources(
    mut args: TrainArgs,
    sources: &TrainArgSources,
) -> anyhow::Result<TrainArgs> {
    let config_path = resolve_train_config_path(&args);
    if let Some(config_path) = config_path {
        let config = TrainConfigFile::load(&config_path)?;
        config.apply_to_with_sources(&mut args, sources, &config_path)?;
        args.train_config = Some(config_path);
    } else if let Some(preset) = &args.train_preset {
        bail!(
            "RustGS train preset '{preset}' requires a training config file. Pass --train-config or create {}/{}",
            args.input.display(),
            DEFAULT_TRAIN_CONFIG_FILE
        );
    }
    Ok(args)
}

const DEFAULT_TRAIN_CONFIG_FILE: &str = "rustgs-train.json";

fn resolve_train_config_path(args: &TrainArgs) -> Option<PathBuf> {
    if let Some(path) = &args.train_config {
        return Some(path.clone());
    }
    if !args.input.is_dir() {
        return None;
    }

    let default_path = args.input.join(DEFAULT_TRAIN_CONFIG_FILE);
    if default_path.exists() {
        Some(default_path)
    } else {
        None
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TrainConfigFile {
    #[serde(default)]
    default_preset: Option<String>,
    #[serde(default)]
    defaults: TrainConfigOverrides,
    #[serde(default)]
    presets: BTreeMap<String, TrainConfigOverrides>,
}

impl TrainConfigFile {
    fn load(path: &Path) -> anyhow::Result<Self> {
        let contents = fs::read_to_string(path)
            .with_context(|| format!("failed to read RustGS train config {}", path.display()))?;
        serde_json::from_str(&contents)
            .with_context(|| format!("failed to parse RustGS train config {}", path.display()))
    }

    fn apply_to_with_sources(
        &self,
        args: &mut TrainArgs,
        sources: &TrainArgSources,
        path: &Path,
    ) -> anyhow::Result<()> {
        self.defaults.apply_to_with_sources(args, sources);

        let selected_preset = args
            .train_preset
            .clone()
            .or_else(|| self.default_preset.clone());
        if let Some(preset) = selected_preset {
            let overrides = self.presets.get(&preset).ok_or_else(|| {
                anyhow::anyhow!(
                    "RustGS train config {} does not define preset '{}'",
                    path.display(),
                    preset
                )
            })?;
            overrides.apply_to_with_sources(args, sources);
            args.train_preset = Some(preset);
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct TrainConfigOverrides {
    iterations: Option<usize>,
    max_initial_gaussians: Option<usize>,
    sampling_step: Option<usize>,
    init_point_scale_factor: Option<f32>,
    init_point_opacity: Option<f32>,
    init_vksplat_scale_estimator: Option<bool>,
    init_random_rotations: Option<bool>,
    init_rotation_seed: Option<u64>,
    max_frames: Option<usize>,
    frame_stride: Option<usize>,
    #[serde(default, deserialize_with = "deserialize_nullable_override")]
    include_frame_ranges: Option<Option<String>>,
    #[serde(default, deserialize_with = "deserialize_nullable_override")]
    exclude_frame_ranges: Option<Option<String>>,
    #[serde(default, deserialize_with = "deserialize_nullable_override")]
    oversample_frame_ranges: Option<Option<String>>,
    oversample_frame_repeat: Option<usize>,
    frame_shuffle_seed: Option<u64>,
    render_scale: Option<f32>,
    raster_cov_blur: Option<f32>,
    #[serde(default, deserialize_with = "deserialize_nullable_override")]
    raster_cov_blur_final: Option<Option<f32>>,
    #[serde(default, deserialize_with = "deserialize_nullable_override")]
    raster_cov_blur_final_after_epoch: Option<Option<usize>>,
    litegs_sh_degree: Option<usize>,
    #[serde(default, deserialize_with = "deserialize_optional_from_str")]
    litegs_profile: Option<rustgs::LiteGsTrainingProfile>,
    #[serde(default, deserialize_with = "deserialize_optional_litegs_tile_size")]
    litegs_tile_size: Option<rustgs::LiteGsTileSize>,
    litegs_sparse_grad: Option<bool>,
    litegs_reg_weight: Option<f32>,
    litegs_enable_transmittance: Option<bool>,
    litegs_enable_depth: Option<bool>,
    litegs_densify_from: Option<usize>,
    #[serde(default, deserialize_with = "deserialize_nullable_override")]
    litegs_densify_until: Option<Option<usize>>,
    #[serde(default, deserialize_with = "deserialize_nullable_override")]
    litegs_topology_freeze_after_epoch: Option<Option<usize>>,
    #[serde(default, deserialize_with = "deserialize_nullable_override")]
    litegs_growth_freeze_after_epoch: Option<Option<usize>>,
    litegs_densification_interval: Option<usize>,
    litegs_refine_every: Option<usize>,
    litegs_growth_grad_threshold: Option<f32>,
    #[serde(default, deserialize_with = "deserialize_optional_from_str")]
    litegs_split_score: Option<rustgs::LiteGsSplitScoreMode>,
    litegs_split_grad_threshold: Option<f32>,
    litegs_depth_scale_gamma: Option<f32>,
    litegs_growth_select_fraction: Option<f32>,
    litegs_growth_stop_iter: Option<usize>,
    litegs_opacity_decay: Option<f32>,
    litegs_scale_decay: Option<f32>,
    litegs_opacity_reset_interval: Option<usize>,
    #[serde(default, deserialize_with = "deserialize_optional_from_str")]
    litegs_opacity_reset_mode: Option<rustgs::LiteGsOpacityResetMode>,
    #[serde(default, deserialize_with = "deserialize_optional_from_str")]
    litegs_prune_mode: Option<rustgs::LiteGsPruneMode>,
    litegs_prune_offset_epochs: Option<usize>,
    litegs_prune_min_age: Option<usize>,
    litegs_prune_invisible_epochs: Option<usize>,
    litegs_prune_opacity_threshold: Option<f32>,
    litegs_prune_visibility_dry_run: Option<bool>,
    litegs_prune_visibility_threshold: Option<f32>,
    litegs_prune_high_opacity_threshold: Option<f32>,
    #[serde(default, deserialize_with = "deserialize_nullable_override")]
    litegs_prune_until_epoch: Option<Option<usize>>,
    litegs_target_primitives: Option<usize>,
    litegs_learnable_viewproj: Option<bool>,
    litegs_lr_pose: Option<f32>,
    litegs_prune_scale_threshold: Option<f32>,
    lr_position: Option<f32>,
    lr_position_final: Option<f32>,
    lr_decay_iterations: Option<usize>,
    lr_position_scene_scale: Option<bool>,
    lr_scale: Option<f32>,
    lr_scale_final: Option<f32>,
    lr_rotation: Option<f32>,
    lr_rotation_final: Option<f32>,
    lr_opacity: Option<f32>,
    lr_opacity_final: Option<f32>,
    lr_color: Option<f32>,
    lr_color_rest: Option<f32>,
    lr_color_final: Option<f32>,
    lr_color_rest_final: Option<f32>,
    loss_l1_weight: Option<f32>,
    loss_ssim_weight: Option<f32>,
    loss_gradient_weight: Option<f32>,
    loss_robust_delta: Option<f32>,
    loss_outlier_threshold: Option<f32>,
    loss_outlier_weight: Option<f32>,
    loss_dynamic_mask_threshold_low: Option<f32>,
    loss_dynamic_mask_threshold_high: Option<f32>,
    loss_dynamic_mask_min_weight: Option<f32>,
    #[serde(default, deserialize_with = "deserialize_nullable_override")]
    loss_dynamic_mask_start_epoch: Option<Option<usize>>,
    log_level: Option<String>,
    eval_after_train: Option<bool>,
    eval_render_scale: Option<f32>,
    #[serde(default, deserialize_with = "deserialize_nullable_override")]
    eval_raster_cov_blur: Option<Option<f32>>,
    eval_max_frames: Option<usize>,
    eval_frame_stride: Option<usize>,
    #[serde(default, deserialize_with = "deserialize_nullable_override")]
    eval_include_frame_ranges: Option<Option<String>>,
    #[serde(default, deserialize_with = "deserialize_nullable_override")]
    eval_exclude_frame_ranges: Option<Option<String>>,
    eval_worst_frames: Option<usize>,
    eval_device: Option<String>,
    eval_json: Option<bool>,
    #[serde(default, deserialize_with = "deserialize_nullable_override")]
    eval_crop_output_dir: Option<Option<PathBuf>>,
    #[serde(default, deserialize_with = "deserialize_nullable_override")]
    eval_crop_frames: Option<Option<String>>,
    #[serde(default, deserialize_with = "deserialize_nullable_override")]
    eval_crop_rect: Option<Option<String>>,
}

impl TrainConfigOverrides {
    fn apply_to_with_sources(&self, args: &mut TrainArgs, sources: &TrainArgSources) {
        apply_override(sources, "iterations", &mut args.iterations, self.iterations);
        apply_override(
            sources,
            "max_initial_gaussians",
            &mut args.max_initial_gaussians,
            self.max_initial_gaussians,
        );
        apply_override(
            sources,
            "sampling_step",
            &mut args.sampling_step,
            self.sampling_step,
        );
        apply_override(
            sources,
            "init_point_scale_factor",
            &mut args.init_point_scale_factor,
            self.init_point_scale_factor,
        );
        apply_override(
            sources,
            "init_point_opacity",
            &mut args.init_point_opacity,
            self.init_point_opacity,
        );
        apply_override(
            sources,
            "init_vksplat_scale_estimator",
            &mut args.init_vksplat_scale_estimator,
            self.init_vksplat_scale_estimator,
        );
        apply_override(
            sources,
            "init_random_rotations",
            &mut args.init_random_rotations,
            self.init_random_rotations,
        );
        apply_override(
            sources,
            "init_rotation_seed",
            &mut args.init_rotation_seed,
            self.init_rotation_seed,
        );
        apply_override(sources, "max_frames", &mut args.max_frames, self.max_frames);
        apply_override(
            sources,
            "frame_stride",
            &mut args.frame_stride,
            self.frame_stride,
        );
        apply_override(
            sources,
            "include_frame_ranges",
            &mut args.include_frame_ranges,
            self.include_frame_ranges.clone(),
        );
        apply_override(
            sources,
            "exclude_frame_ranges",
            &mut args.exclude_frame_ranges,
            self.exclude_frame_ranges.clone(),
        );
        apply_override(
            sources,
            "oversample_frame_ranges",
            &mut args.oversample_frame_ranges,
            self.oversample_frame_ranges.clone(),
        );
        apply_override(
            sources,
            "oversample_frame_repeat",
            &mut args.oversample_frame_repeat,
            self.oversample_frame_repeat,
        );
        apply_override(
            sources,
            "frame_shuffle_seed",
            &mut args.frame_shuffle_seed,
            self.frame_shuffle_seed,
        );
        apply_override(
            sources,
            "render_scale",
            &mut args.render_scale,
            self.render_scale,
        );
        apply_override(
            sources,
            "raster_cov_blur",
            &mut args.raster_cov_blur,
            self.raster_cov_blur,
        );
        apply_override(
            sources,
            "raster_cov_blur_final",
            &mut args.raster_cov_blur_final,
            self.raster_cov_blur_final,
        );
        apply_override(
            sources,
            "raster_cov_blur_final_after_epoch",
            &mut args.raster_cov_blur_final_after_epoch,
            self.raster_cov_blur_final_after_epoch,
        );
        apply_override(
            sources,
            "litegs_sh_degree",
            &mut args.litegs_sh_degree,
            self.litegs_sh_degree,
        );
        apply_override(
            sources,
            "litegs_profile",
            &mut args.litegs_profile,
            self.litegs_profile,
        );
        apply_override(
            sources,
            "litegs_tile_size",
            &mut args.litegs_tile_size,
            self.litegs_tile_size,
        );
        apply_override(
            sources,
            "litegs_sparse_grad",
            &mut args.litegs_sparse_grad,
            self.litegs_sparse_grad,
        );
        apply_override(
            sources,
            "litegs_reg_weight",
            &mut args.litegs_reg_weight,
            self.litegs_reg_weight,
        );
        apply_override(
            sources,
            "litegs_enable_transmittance",
            &mut args.litegs_enable_transmittance,
            self.litegs_enable_transmittance,
        );
        apply_override(
            sources,
            "litegs_enable_depth",
            &mut args.litegs_enable_depth,
            self.litegs_enable_depth,
        );
        apply_override(
            sources,
            "litegs_densify_from",
            &mut args.litegs_densify_from,
            self.litegs_densify_from,
        );
        apply_override(
            sources,
            "litegs_densify_until",
            &mut args.litegs_densify_until,
            self.litegs_densify_until,
        );
        apply_override(
            sources,
            "litegs_topology_freeze_after_epoch",
            &mut args.litegs_topology_freeze_after_epoch,
            self.litegs_topology_freeze_after_epoch,
        );
        apply_override(
            sources,
            "litegs_growth_freeze_after_epoch",
            &mut args.litegs_growth_freeze_after_epoch,
            self.litegs_growth_freeze_after_epoch,
        );
        apply_override(
            sources,
            "litegs_densification_interval",
            &mut args.litegs_densification_interval,
            self.litegs_densification_interval,
        );
        apply_override(
            sources,
            "litegs_refine_every",
            &mut args.litegs_refine_every,
            self.litegs_refine_every,
        );
        apply_override(
            sources,
            "litegs_growth_grad_threshold",
            &mut args.litegs_growth_grad_threshold,
            self.litegs_growth_grad_threshold,
        );
        apply_override(
            sources,
            "litegs_split_score",
            &mut args.litegs_split_score,
            self.litegs_split_score,
        );
        apply_override(
            sources,
            "litegs_split_grad_threshold",
            &mut args.litegs_split_grad_threshold,
            self.litegs_split_grad_threshold,
        );
        apply_override(
            sources,
            "litegs_depth_scale_gamma",
            &mut args.litegs_depth_scale_gamma,
            self.litegs_depth_scale_gamma,
        );
        apply_override(
            sources,
            "litegs_growth_select_fraction",
            &mut args.litegs_growth_select_fraction,
            self.litegs_growth_select_fraction,
        );
        apply_override(
            sources,
            "litegs_growth_stop_iter",
            &mut args.litegs_growth_stop_iter,
            self.litegs_growth_stop_iter,
        );
        apply_override(
            sources,
            "litegs_opacity_decay",
            &mut args.litegs_opacity_decay,
            self.litegs_opacity_decay,
        );
        apply_override(
            sources,
            "litegs_scale_decay",
            &mut args.litegs_scale_decay,
            self.litegs_scale_decay,
        );
        apply_override(
            sources,
            "litegs_opacity_reset_interval",
            &mut args.litegs_opacity_reset_interval,
            self.litegs_opacity_reset_interval,
        );
        apply_override(
            sources,
            "litegs_opacity_reset_mode",
            &mut args.litegs_opacity_reset_mode,
            self.litegs_opacity_reset_mode,
        );
        apply_override(
            sources,
            "litegs_prune_mode",
            &mut args.litegs_prune_mode,
            self.litegs_prune_mode,
        );
        apply_override(
            sources,
            "litegs_prune_offset_epochs",
            &mut args.litegs_prune_offset_epochs,
            self.litegs_prune_offset_epochs,
        );
        apply_override(
            sources,
            "litegs_prune_min_age",
            &mut args.litegs_prune_min_age,
            self.litegs_prune_min_age,
        );
        apply_override(
            sources,
            "litegs_prune_invisible_epochs",
            &mut args.litegs_prune_invisible_epochs,
            self.litegs_prune_invisible_epochs,
        );
        apply_override(
            sources,
            "litegs_prune_opacity_threshold",
            &mut args.litegs_prune_opacity_threshold,
            self.litegs_prune_opacity_threshold,
        );
        apply_override(
            sources,
            "litegs_prune_visibility_dry_run",
            &mut args.litegs_prune_visibility_dry_run,
            self.litegs_prune_visibility_dry_run,
        );
        apply_override(
            sources,
            "litegs_prune_visibility_threshold",
            &mut args.litegs_prune_visibility_threshold,
            self.litegs_prune_visibility_threshold,
        );
        apply_override(
            sources,
            "litegs_prune_high_opacity_threshold",
            &mut args.litegs_prune_high_opacity_threshold,
            self.litegs_prune_high_opacity_threshold,
        );
        apply_override(
            sources,
            "litegs_prune_until_epoch",
            &mut args.litegs_prune_until_epoch,
            self.litegs_prune_until_epoch,
        );
        apply_override(
            sources,
            "litegs_target_primitives",
            &mut args.litegs_target_primitives,
            self.litegs_target_primitives,
        );
        apply_override(
            sources,
            "litegs_learnable_viewproj",
            &mut args.litegs_learnable_viewproj,
            self.litegs_learnable_viewproj,
        );
        apply_override(
            sources,
            "litegs_lr_pose",
            &mut args.litegs_lr_pose,
            self.litegs_lr_pose,
        );
        apply_override(
            sources,
            "litegs_prune_scale_threshold",
            &mut args.litegs_prune_scale_threshold,
            self.litegs_prune_scale_threshold,
        );
        apply_override(
            sources,
            "lr_position",
            &mut args.lr_position,
            self.lr_position,
        );
        apply_override(
            sources,
            "lr_position_final",
            &mut args.lr_position_final,
            self.lr_position_final,
        );
        apply_override(
            sources,
            "lr_decay_iterations",
            &mut args.lr_decay_iterations,
            self.lr_decay_iterations,
        );
        apply_override(
            sources,
            "lr_position_scene_scale",
            &mut args.lr_position_scene_scale,
            self.lr_position_scene_scale,
        );
        apply_override(sources, "lr_scale", &mut args.lr_scale, self.lr_scale);
        apply_override(
            sources,
            "lr_scale_final",
            &mut args.lr_scale_final,
            self.lr_scale_final,
        );
        apply_override(
            sources,
            "lr_rotation",
            &mut args.lr_rotation,
            self.lr_rotation,
        );
        apply_override(
            sources,
            "lr_rotation_final",
            &mut args.lr_rotation_final,
            self.lr_rotation_final,
        );
        apply_override(sources, "lr_opacity", &mut args.lr_opacity, self.lr_opacity);
        apply_override(
            sources,
            "lr_opacity_final",
            &mut args.lr_opacity_final,
            self.lr_opacity_final,
        );
        apply_override(sources, "lr_color", &mut args.lr_color, self.lr_color);
        apply_override(
            sources,
            "lr_color_rest",
            &mut args.lr_color_rest,
            self.lr_color_rest,
        );
        apply_override(
            sources,
            "lr_color_final",
            &mut args.lr_color_final,
            self.lr_color_final,
        );
        apply_override(
            sources,
            "lr_color_rest_final",
            &mut args.lr_color_rest_final,
            self.lr_color_rest_final,
        );
        apply_override(
            sources,
            "loss_l1_weight",
            &mut args.loss_l1_weight,
            self.loss_l1_weight,
        );
        apply_override(
            sources,
            "loss_ssim_weight",
            &mut args.loss_ssim_weight,
            self.loss_ssim_weight,
        );
        apply_override(
            sources,
            "loss_gradient_weight",
            &mut args.loss_gradient_weight,
            self.loss_gradient_weight,
        );
        apply_override(
            sources,
            "loss_robust_delta",
            &mut args.loss_robust_delta,
            self.loss_robust_delta,
        );
        apply_override(
            sources,
            "loss_outlier_threshold",
            &mut args.loss_outlier_threshold,
            self.loss_outlier_threshold,
        );
        apply_override(
            sources,
            "loss_outlier_weight",
            &mut args.loss_outlier_weight,
            self.loss_outlier_weight,
        );
        apply_override(
            sources,
            "loss_dynamic_mask_threshold_low",
            &mut args.loss_dynamic_mask_threshold_low,
            self.loss_dynamic_mask_threshold_low,
        );
        apply_override(
            sources,
            "loss_dynamic_mask_threshold_high",
            &mut args.loss_dynamic_mask_threshold_high,
            self.loss_dynamic_mask_threshold_high,
        );
        apply_override(
            sources,
            "loss_dynamic_mask_min_weight",
            &mut args.loss_dynamic_mask_min_weight,
            self.loss_dynamic_mask_min_weight,
        );
        apply_override(
            sources,
            "loss_dynamic_mask_start_epoch",
            &mut args.loss_dynamic_mask_start_epoch,
            self.loss_dynamic_mask_start_epoch,
        );
        apply_override(
            sources,
            "log_level",
            &mut args.log_level,
            self.log_level.clone(),
        );
        apply_override(
            sources,
            "eval_after_train",
            &mut args.eval_after_train,
            self.eval_after_train,
        );
        apply_override(
            sources,
            "eval_render_scale",
            &mut args.eval_render_scale,
            self.eval_render_scale,
        );
        apply_override(
            sources,
            "eval_raster_cov_blur",
            &mut args.eval_raster_cov_blur,
            self.eval_raster_cov_blur,
        );
        apply_override(
            sources,
            "eval_max_frames",
            &mut args.eval_max_frames,
            self.eval_max_frames,
        );
        apply_override(
            sources,
            "eval_frame_stride",
            &mut args.eval_frame_stride,
            self.eval_frame_stride,
        );
        apply_override(
            sources,
            "eval_include_frame_ranges",
            &mut args.eval_include_frame_ranges,
            self.eval_include_frame_ranges.clone(),
        );
        apply_override(
            sources,
            "eval_exclude_frame_ranges",
            &mut args.eval_exclude_frame_ranges,
            self.eval_exclude_frame_ranges.clone(),
        );
        apply_override(
            sources,
            "eval_worst_frames",
            &mut args.eval_worst_frames,
            self.eval_worst_frames,
        );
        apply_override(
            sources,
            "eval_device",
            &mut args.eval_device,
            self.eval_device.clone(),
        );
        apply_override(sources, "eval_json", &mut args.eval_json, self.eval_json);
        apply_override(
            sources,
            "eval_crop_output_dir",
            &mut args.eval_crop_output_dir,
            self.eval_crop_output_dir.clone(),
        );
        apply_override(
            sources,
            "eval_crop_frames",
            &mut args.eval_crop_frames,
            self.eval_crop_frames.clone(),
        );
        apply_override(
            sources,
            "eval_crop_rect",
            &mut args.eval_crop_rect,
            self.eval_crop_rect.clone(),
        );
    }
}

fn apply_override<T>(sources: &TrainArgSources, id: &str, target: &mut T, value: Option<T>) {
    if !sources.is_command_line(id) {
        if let Some(value) = value {
            *target = value;
        }
    }
}

fn deserialize_optional_litegs_tile_size<'de, D>(
    deserializer: D,
) -> Result<Option<rustgs::LiteGsTileSize>, D::Error>
where
    D: de::Deserializer<'de>,
{
    let value = Option::<LiteGsTileSizeConfig>::deserialize(deserializer)?;
    value
        .map(|value| value.into_tile_size())
        .transpose()
        .map_err(de::Error::custom)
}

fn deserialize_optional_from_str<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: de::Deserializer<'de>,
    T: FromStr,
    T::Err: std::fmt::Display,
{
    let value = Option::<String>::deserialize(deserializer)?;
    value
        .map(|value| T::from_str(&value))
        .transpose()
        .map_err(de::Error::custom)
}

fn deserialize_nullable_override<'de, D, T>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
where
    D: de::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer).map(Some)
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum LiteGsTileSizeConfig {
    String(String),
    Object { width: usize, height: usize },
}

impl LiteGsTileSizeConfig {
    fn into_tile_size(self) -> Result<rustgs::LiteGsTileSize, String> {
        match self {
            Self::String(value) => rustgs::LiteGsTileSize::from_str(&value),
            Self::Object { width, height } => {
                if width == 0 || height == 0 {
                    Err("litegs_tile_size width and height must both be > 0".to_string())
                } else {
                    Ok(rustgs::LiteGsTileSize::new(width, height))
                }
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(super) struct TrainArgSources {
    command_line: BTreeSet<String>,
}

impl TrainArgSources {
    pub(super) fn from_cli_matches(matches: &clap::ArgMatches) -> Self {
        let Some(("train", train_matches)) = matches.subcommand() else {
            return Self::default();
        };
        Self::from_train_matches(train_matches)
    }

    fn from_train_matches(matches: &clap::ArgMatches) -> Self {
        let command_line = matches
            .ids()
            .filter(|id| matches.value_source(id.as_str()) == Some(ValueSource::CommandLine))
            .map(|id| id.as_str().to_string())
            .collect();
        Self { command_line }
    }

    pub(super) fn is_command_line(&self, id: &str) -> bool {
        self.command_line.contains(id)
    }
}

pub(super) fn load_training_dataset_for_training(
    input: &Path,
    image_root: Option<&Path>,
    max_frames: usize,
    frame_stride: usize,
) -> anyhow::Result<(rustscan_types::TrainingDataset, rustgs::TrainingInputKind)> {
    if !input.is_dir() && (max_frames > 0 || frame_stride > 1) {
        log::warn!(
            "--max-frames and --frame-stride only apply to dataset directories; ignoring them for {:?}",
            input
        );
    }

    let (dataset, source) = rustgs::load_colmap_training_dataset_with_source(
        input,
        &rustgs::ColmapConfig {
            max_frames,
            frame_stride,
            image_root: image_root.map(Path::to_path_buf),
            ..Default::default()
        },
    )?;

    log::info!(
        "Resolved {:?} as {} with {} poses",
        input,
        source,
        dataset.poses.len(),
    );

    Ok((dataset, source))
}

#[cfg(feature = "gpu")]
fn load_evaluation_dataset(
    input: &Path,
    image_root: Option<&Path>,
    max_frames: usize,
    frame_stride: usize,
) -> anyhow::Result<rustscan_types::TrainingDataset> {
    let (dataset, source) = rustgs::load_colmap_training_dataset_with_source(
        input,
        &rustgs::ColmapConfig {
            max_frames,
            frame_stride,
            image_root: image_root.map(Path::to_path_buf),
            ..Default::default()
        },
    )?;

    log::info!(
        "Resolved evaluation dataset {:?} as {} with {} poses",
        input,
        source,
        dataset.poses.len(),
    );

    Ok(dataset)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FrameIdRange {
    start: u64,
    end: u64,
}

impl FrameIdRange {
    fn contains(&self, frame_id: u64) -> bool {
        self.start <= frame_id && frame_id <= self.end
    }
}

fn parse_frame_ranges(value: Option<&str>) -> anyhow::Result<Vec<FrameIdRange>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let mut ranges = Vec::new();
    for raw_token in value.split(',') {
        let token = raw_token.trim();
        if token.is_empty() {
            continue;
        }
        let (start, end) = if let Some((start, end)) = token.split_once("..") {
            (start.trim(), end.trim())
        } else if let Some((start, end)) = token.split_once('-') {
            (start.trim(), end.trim())
        } else {
            (token, token)
        };
        if start.is_empty() || end.is_empty() {
            bail!("frame range '{token}' must be <frame_id> or <start>-<end>");
        }
        let start = start
            .parse::<u64>()
            .with_context(|| format!("invalid frame range start in '{token}'"))?;
        let end = end
            .parse::<u64>()
            .with_context(|| format!("invalid frame range end in '{token}'"))?;
        if start > end {
            bail!("frame range '{token}' has start greater than end");
        }
        ranges.push(FrameIdRange { start, end });
    }
    Ok(ranges)
}

fn filter_dataset_by_frame_ranges(
    dataset: rustscan_types::TrainingDataset,
    excluded_ranges: &[FrameIdRange],
    label: &str,
) -> anyhow::Result<rustscan_types::TrainingDataset> {
    if excluded_ranges.is_empty() {
        return Ok(dataset);
    }

    let original_pose_count = dataset.poses.len();
    let mut filtered = rustscan_types::TrainingDataset::new(dataset.intrinsics)
        .with_depth_scale(dataset.depth_scale);
    filtered.initial_points = dataset.initial_points.clone();
    for pose in dataset.poses {
        if excluded_ranges
            .iter()
            .any(|range| range.contains(pose.frame_id))
        {
            continue;
        }
        filtered.add_pose(pose);
    }

    let removed = original_pose_count.saturating_sub(filtered.poses.len());
    log::info!(
        "Applied {label} frame exclusion | removed={} | remaining={}",
        removed,
        filtered.poses.len()
    );
    if filtered.poses.is_empty() {
        bail!("{label} frame exclusion removed all frames");
    }

    Ok(filtered)
}

fn filter_dataset_to_frame_ranges(
    dataset: rustscan_types::TrainingDataset,
    included_ranges: &[FrameIdRange],
    label: &str,
) -> anyhow::Result<rustscan_types::TrainingDataset> {
    if included_ranges.is_empty() {
        return Ok(dataset);
    }

    let original_pose_count = dataset.poses.len();
    let mut filtered = rustscan_types::TrainingDataset::new(dataset.intrinsics)
        .with_depth_scale(dataset.depth_scale);
    filtered.initial_points = dataset.initial_points.clone();
    for pose in dataset.poses {
        if included_ranges
            .iter()
            .any(|range| range.contains(pose.frame_id))
        {
            filtered.add_pose(pose);
        }
    }

    log::info!(
        "Applied {label} frame include | kept={} | removed={}",
        filtered.poses.len(),
        original_pose_count.saturating_sub(filtered.poses.len())
    );
    if filtered.poses.is_empty() {
        bail!("{label} frame include selected no frames");
    }

    Ok(filtered)
}

fn oversample_dataset_frame_ranges(
    dataset: rustscan_types::TrainingDataset,
    oversample_ranges: &[FrameIdRange],
    repeat: usize,
    label: &str,
) -> anyhow::Result<rustscan_types::TrainingDataset> {
    if repeat == 0 {
        bail!("--oversample-frame-repeat must be >= 1");
    }
    if oversample_ranges.is_empty() || repeat == 1 {
        return Ok(dataset);
    }

    let matching_poses: Vec<_> = dataset
        .poses
        .iter()
        .filter(|pose| {
            oversample_ranges
                .iter()
                .any(|range| range.contains(pose.frame_id))
        })
        .cloned()
        .collect();
    if matching_poses.is_empty() {
        bail!("{label} frame oversampling selected no frames");
    }

    let original_pose_count = dataset.poses.len();
    let mut augmented = rustscan_types::TrainingDataset::new(dataset.intrinsics)
        .with_depth_scale(dataset.depth_scale);
    augmented.initial_points = dataset.initial_points.clone();
    for pose in dataset.poses {
        augmented.add_pose(pose);
    }
    for _ in 1..repeat {
        for pose in &matching_poses {
            augmented.add_pose(pose.clone());
        }
    }

    log::info!(
        "Applied {label} frame oversampling | matched={} | repeat={} | original={} | augmented={}",
        matching_poses.len(),
        repeat,
        original_pose_count,
        augmented.poses.len()
    );

    Ok(augmented)
}

#[cfg(feature = "gpu")]
pub(super) fn evaluation_dataset_load_params(args: &TrainArgs) -> (usize, usize) {
    // Keep the evaluation prefix trimming, but do not apply frame_stride here.
    // The actual evaluation subset selection should happen once inside evaluate_splats().
    (args.eval_max_frames, 1)
}

#[cfg(feature = "gpu")]
fn final_training_metrics_from_telemetry(
    training_telemetry: Option<&rustgs::LiteGsTrainingTelemetry>,
    metadata: &rustgs::SplatMetadata,
) -> Option<rustgs::FinalTrainingMetrics> {
    training_telemetry.map(|telemetry| rustgs::FinalTrainingMetrics {
        final_loss: telemetry.final_loss.unwrap_or(metadata.final_loss),
        final_step_loss: telemetry.final_step_loss.unwrap_or(metadata.final_loss),
    })
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EvalCropRect {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

#[cfg(feature = "gpu")]
fn parse_eval_crop_rect(
    value: Option<&str>,
    render_width: usize,
    render_height: usize,
) -> anyhow::Result<EvalCropRect> {
    let Some(value) = value else {
        return Ok(EvalCropRect {
            x: 0,
            y: 0,
            width: render_width,
            height: render_height,
        });
    };
    let parts = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    if parts.len() != 4 {
        bail!("--eval-crop-rect must be x,y,width,height in evaluation pixels");
    }
    let x = parts[0].parse::<usize>()?;
    let y = parts[1].parse::<usize>()?;
    let width = parts[2].parse::<usize>()?;
    let height = parts[3].parse::<usize>()?;
    if width == 0 || height == 0 {
        bail!("--eval-crop-rect width and height must be >= 1");
    }
    if x >= render_width
        || y >= render_height
        || x + width > render_width
        || y + height > render_height
    {
        bail!(
            "--eval-crop-rect {value} exceeds evaluation resolution {}x{}",
            render_width,
            render_height
        );
    }
    Ok(EvalCropRect {
        x,
        y,
        width,
        height,
    })
}

#[cfg(feature = "gpu")]
fn parse_eval_crop_frame_ids(value: Option<&str>) -> anyhow::Result<Option<BTreeSet<u64>>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let mut ids = BTreeSet::new();
    for token in value
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        ids.insert(token.parse::<u64>()?);
    }
    if ids.is_empty() {
        bail!("--eval-crop-frames must contain at least one frame id");
    }
    Ok(Some(ids))
}

#[cfg(feature = "gpu")]
fn crop_frame_indices(
    dataset: &rustscan_types::TrainingDataset,
    summary: &rustgs::SplatEvaluationSummary,
    requested_frame_ids: Option<&BTreeSet<u64>>,
) -> anyhow::Result<Vec<usize>> {
    if let Some(requested_frame_ids) = requested_frame_ids {
        let mut found = BTreeSet::new();
        let indices = dataset
            .poses
            .iter()
            .enumerate()
            .filter(|(_, pose)| requested_frame_ids.contains(&pose.frame_id))
            .map(|(idx, pose)| {
                found.insert(pose.frame_id);
                idx
            })
            .collect::<Vec<_>>();
        if found.len() != requested_frame_ids.len() {
            let missing = requested_frame_ids
                .difference(&found)
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(",");
            bail!("requested --eval-crop-frames were not in the evaluated frame subset: {missing}");
        }
        return Ok(indices);
    }

    let mut deduped = BTreeSet::new();
    for frame in &summary.worst_frames {
        if frame.dataset_index < dataset.poses.len() {
            deduped.insert(frame.dataset_index);
        }
    }
    Ok(deduped.into_iter().collect())
}

#[cfg(feature = "gpu")]
fn save_rgb_crop(
    path: &Path,
    data: &[f32],
    render_width: usize,
    rect: EvalCropRect,
) -> anyhow::Result<()> {
    let mut image = image::RgbImage::new(rect.width as u32, rect.height as u32);
    for crop_y in 0..rect.height {
        let src_y = rect.y + crop_y;
        for crop_x in 0..rect.width {
            let src_x = rect.x + crop_x;
            let base = (src_y * render_width + src_x) * 3;
            let pixel = [
                float_to_u8(data.get(base).copied().unwrap_or_default()),
                float_to_u8(data.get(base + 1).copied().unwrap_or_default()),
                float_to_u8(data.get(base + 2).copied().unwrap_or_default()),
            ];
            image.put_pixel(crop_x as u32, crop_y as u32, image::Rgb(pixel));
        }
    }
    image
        .save(path)
        .with_context(|| format!("failed to save evaluation crop {}", path.display()))
}

#[cfg(feature = "gpu")]
fn save_rgb_crop_strip(
    path: &Path,
    target: &[f32],
    rendered: &[f32],
    diff: &[f32],
    render_width: usize,
    rect: EvalCropRect,
) -> anyhow::Result<()> {
    let mut image = image::RgbImage::new((rect.width * 3) as u32, rect.height as u32);
    for crop_y in 0..rect.height {
        let src_y = rect.y + crop_y;
        for (panel_idx, data) in [target, rendered, diff].iter().enumerate() {
            for crop_x in 0..rect.width {
                let src_x = rect.x + crop_x;
                let src_base = (src_y * render_width + src_x) * 3;
                let dst_x = panel_idx * rect.width + crop_x;
                let pixel = [
                    float_to_u8(data.get(src_base).copied().unwrap_or_default()),
                    float_to_u8(data.get(src_base + 1).copied().unwrap_or_default()),
                    float_to_u8(data.get(src_base + 2).copied().unwrap_or_default()),
                ];
                image.put_pixel(dst_x as u32, crop_y as u32, image::Rgb(pixel));
            }
        }
    }
    image
        .save(path)
        .with_context(|| format!("failed to save evaluation crop strip {}", path.display()))
}

#[cfg(feature = "gpu")]
fn float_to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(feature = "gpu")]
fn diff_image(rendered: &[f32], target: &[f32]) -> Vec<f32> {
    rendered
        .iter()
        .zip(target.iter())
        .map(|(rendered, target)| ((rendered - target).abs() * 4.0).clamp(0.0, 1.0))
        .collect()
}

#[cfg(feature = "gpu")]
fn export_evaluation_crops(
    args: &TrainArgs,
    dataset: &rustscan_types::TrainingDataset,
    splats: &rustgs::HostSplats,
    device: &rustgs::EvaluationDevice,
    summary: &rustgs::SplatEvaluationSummary,
) -> anyhow::Result<Vec<PathBuf>> {
    let Some(output_dir) = args.eval_crop_output_dir.as_ref() else {
        return Ok(Vec::new());
    };
    fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create crop output dir {}", output_dir.display()))?;
    let selected =
        rustgs::select_evaluation_frames(dataset, args.eval_max_frames, args.eval_frame_stride);
    let requested_frame_ids = parse_eval_crop_frame_ids(args.eval_crop_frames.as_deref())?;
    let frame_indices = crop_frame_indices(&selected, summary, requested_frame_ids.as_ref())?;
    let rect = parse_eval_crop_rect(
        args.eval_crop_rect.as_deref(),
        summary.render_width,
        summary.render_height,
    )?;

    let runtime_splats =
        rustgs::runtime_from_splats(splats, device).map_err(anyhow::Error::from)?;
    let mut renderer = rustgs::SplatEvaluationRenderer::new(
        summary.render_width,
        summary.render_height,
        *device,
        summary.raster_cov_blur,
    )
    .map_err(anyhow::Error::from)?;
    let mut outputs = Vec::new();

    for idx in frame_indices {
        let pose = selected.poses.get(idx).with_context(|| {
            format!("evaluated frame index {idx} was not found for crop export")
        })?;
        let (target, rendered) = rustgs::render_evaluation_frame(
            &selected,
            pose,
            summary.render_width,
            summary.render_height,
            device,
            &runtime_splats,
            &mut renderer,
        )
        .map_err(anyhow::Error::from)?;
        let diff = diff_image(&rendered, &target);
        let base_name = format!("frame_{:06}_idx_{:04}", pose.frame_id, idx);
        for (kind, data) in [
            ("target", target.as_slice()),
            ("render", rendered.as_slice()),
            ("diff_x4", diff.as_slice()),
        ] {
            let path = output_dir.join(format!("{base_name}_{kind}.png"));
            save_rgb_crop(&path, data, summary.render_width, rect)?;
            outputs.push(path);
        }
        let strip_path = output_dir.join(format!("{base_name}_strip.png"));
        save_rgb_crop_strip(
            &strip_path,
            &target,
            &rendered,
            &diff,
            summary.render_width,
            rect,
        )?;
        outputs.push(strip_path);
    }

    Ok(outputs)
}

#[cfg(feature = "gpu")]
fn maybe_evaluate_trained_splats(
    args: &TrainArgs,
    splats: &rustgs::HostSplats,
    metadata: &rustgs::SplatMetadata,
    training_telemetry: Option<&rustgs::LiteGsTrainingTelemetry>,
) -> anyhow::Result<Option<rustgs::SplatEvaluationSummary>> {
    if !args.eval_after_train {
        return Ok(None);
    }
    if args.eval_frame_stride == 0 {
        bail!("--eval-frame-stride must be >= 1");
    }
    if !(0.0625..=1.0).contains(&args.eval_render_scale) {
        bail!("--eval-render-scale must be in [0.0625, 1.0]");
    }
    let eval_raster_cov_blur = args.eval_raster_cov_blur.unwrap_or(args.raster_cov_blur);
    if !eval_raster_cov_blur.is_finite() || eval_raster_cov_blur < 0.0 {
        bail!("--eval-raster-cov-blur must be finite and >= 0");
    }

    let eval_device = args
        .eval_device
        .parse::<rustgs::EvaluationDevice>()
        .map_err(anyhow::Error::msg)?;
    let device = rustgs::evaluation_device(eval_device).map_err(anyhow::Error::from)?;
    let (dataset_max_frames, dataset_frame_stride) = evaluation_dataset_load_params(args);
    let dataset = load_evaluation_dataset(
        &args.input,
        args.image_root.as_deref(),
        dataset_max_frames,
        dataset_frame_stride,
    )?;
    let included_eval_ranges = parse_frame_ranges(args.eval_include_frame_ranges.as_deref())?;
    let dataset = filter_dataset_to_frame_ranges(dataset, &included_eval_ranges, "evaluation")?;
    let excluded_eval_ranges = parse_frame_ranges(args.eval_exclude_frame_ranges.as_deref())?;
    let dataset = filter_dataset_by_frame_ranges(dataset, &excluded_eval_ranges, "evaluation")?;
    let mut evaluation = rustgs::evaluate_splats(
        &dataset,
        splats,
        metadata,
        &rustgs::SplatEvaluationConfig {
            render_scale: args.eval_render_scale,
            raster_cov_blur: eval_raster_cov_blur,
            frame_stride: args.eval_frame_stride,
            max_frames: args.eval_max_frames,
            worst_frame_count: args.eval_worst_frames,
        },
        &device,
        final_training_metrics_from_telemetry(training_telemetry, metadata),
    )
    .map_err(anyhow::Error::from)?;

    evaluation.summary.crop_outputs =
        export_evaluation_crops(args, &dataset, splats, &device, &evaluation.summary)?;
    log_splat_evaluation_summary(&evaluation.summary, args.eval_json)?;
    Ok(Some(evaluation.summary))
}

#[cfg(feature = "gpu")]
fn log_splat_evaluation_summary(
    summary: &rustgs::SplatEvaluationSummary,
    emit_json: bool,
) -> anyhow::Result<()> {
    log::info!(
        "Splat evaluation summary | device={} | render_scale={:.3} | raster_cov_blur={:.3} | resolution={}x{} | frames={} | final_loss={:.6} | final_step_loss={:?} | psnr_mean_db={:.4} | psnr_min_db={:.4} | psnr_max_db={:.4} | sharpness_grad_ratio_mean={:.4} | sharpness_lap_ratio_mean={:.4} | elapsed={:.2}s",
        summary.device,
        summary.render_scale,
        summary.raster_cov_blur,
        summary.render_width,
        summary.render_height,
        summary.frame_count,
        summary.final_loss,
        summary.final_step_loss,
        summary.psnr_mean_db,
        summary.psnr_min_db,
        summary.psnr_max_db,
        summary.sharpness_grad_ratio_mean,
        summary.sharpness_lap_ratio_mean,
        summary.elapsed_seconds,
    );
    for (rank, frame) in summary.worst_frames.iter().enumerate() {
        log::info!(
            "Worst evaluated frame | rank={} | dataset_index={} | frame_id={} | psnr_db={:.4} | sharpness_grad_ratio={:.4} | sharpness_lap_ratio={:.4} | image={}",
            rank + 1,
            frame.dataset_index,
            frame.frame_id,
            frame.psnr_db,
            frame.sharpness_grad_ratio,
            frame.sharpness_lap_ratio,
            frame.image_path.display()
        );
    }
    for path in &summary.crop_outputs {
        log::info!("Evaluation crop exported | path={}", path.display());
    }
    if emit_json {
        println!("{}", serde_json::to_string_pretty(summary)?);
    }
    Ok(())
}

pub(super) fn build_training_config(args: &TrainArgs) -> anyhow::Result<rustgs::TrainingConfig> {
    if args.litegs_target_primitives == 0 {
        bail!("--litegs-target-primitives must be >= 1");
    }
    if args.litegs_learnable_viewproj {
        bail!(
            "--litegs-learnable-viewproj is not implemented in the RustGS wgpu trainer yet; camera poses are fixed during training"
        );
    }

    let (split_score_mode, split_grad_threshold, depth_scale_gamma) =
        litegs_profile_overrides(args);
    let config = rustgs::TrainingConfig {
        iterations: args.iterations,
        initialization: rustgs::TrainingInitializationConfig {
            max_initial_gaussians: args.max_initial_gaussians,
            sampling_step: args.sampling_step,
            point_scale_factor: args.init_point_scale_factor,
            point_opacity: args.init_point_opacity,
            vksplat_scale_estimator: args.init_vksplat_scale_estimator,
            randomize_rotations: args.init_random_rotations,
            rotation_seed: args.init_rotation_seed,
            ..rustgs::TrainingInitializationConfig::default()
        },
        data: rustgs::TrainingDataConfig {
            frame_shuffle_seed: args.frame_shuffle_seed,
            ..rustgs::TrainingDataConfig::default()
        },
        raster: rustgs::TrainingRasterConfig {
            render_scale: args.render_scale,
            raster_cov_blur: args.raster_cov_blur,
            raster_cov_blur_final: args.raster_cov_blur_final,
            raster_cov_blur_final_after_epoch: args.raster_cov_blur_final_after_epoch,
        },
        optimizer: rustgs::TrainingOptimizerConfig {
            lr_position: args.lr_position,
            lr_pos_final: args.lr_position_final,
            lr_decay_iterations: (args.lr_decay_iterations > 0).then_some(args.lr_decay_iterations),
            lr_position_scene_scale: args.lr_position_scene_scale,
            lr_scale: args.lr_scale,
            lr_scale_final: args.lr_scale_final,
            lr_rotation: args.lr_rotation,
            lr_rotation_final: args.lr_rotation_final,
            lr_opacity: args.lr_opacity,
            lr_opacity_final: args.lr_opacity_final,
            lr_color: args.lr_color,
            lr_color_rest: args.lr_color_rest,
            lr_color_final: args.lr_color_final,
            lr_color_rest_final: args.lr_color_rest_final,
        },
        loss: rustgs::TrainingLossConfig {
            loss_l1_weight: args.loss_l1_weight,
            loss_ssim_weight: args.loss_ssim_weight,
            loss_gradient_weight: args.loss_gradient_weight,
            loss_robust_delta: args.loss_robust_delta,
            loss_outlier_threshold: args.loss_outlier_threshold,
            loss_outlier_weight: args.loss_outlier_weight,
            loss_dynamic_mask_threshold_low: args.loss_dynamic_mask_threshold_low,
            loss_dynamic_mask_threshold_high: args.loss_dynamic_mask_threshold_high,
            loss_dynamic_mask_min_weight: args.loss_dynamic_mask_min_weight,
            loss_dynamic_mask_start_epoch: args.loss_dynamic_mask_start_epoch,
        },
        litegs: rustgs::LiteGsConfig {
            rendering: rustgs::LiteGsRenderingConfig {
                sh_degree: args.litegs_sh_degree,
                tile_size: args.litegs_tile_size,
            },
            features: rustgs::LiteGsFeatureConfig {
                sparse_grad: args.litegs_sparse_grad,
                reg_weight: args.litegs_reg_weight,
                enable_transmittance: args.litegs_enable_transmittance,
                enable_depth: args.litegs_enable_depth,
                training_profile: args.litegs_profile,
            },
            topology: rustgs::LiteGsTopologyConfig {
                densify_from: args.litegs_densify_from,
                densify_until: args.litegs_densify_until,
                topology_freeze_after_epoch: args.litegs_topology_freeze_after_epoch,
                growth_freeze_after_epoch: args.litegs_growth_freeze_after_epoch,
                refine_every: args.litegs_refine_every,
                densification_interval: args.litegs_densification_interval,
                opacity_reset_interval: args.litegs_opacity_reset_interval,
                opacity_reset_mode: args.litegs_opacity_reset_mode,
                target_primitives: args.litegs_target_primitives,
            },
            growth: rustgs::LiteGsGrowthConfig {
                growth_grad_threshold: args.litegs_growth_grad_threshold,
                split_score_mode,
                split_grad_threshold,
                depth_scale_gamma,
                growth_select_fraction: args.litegs_growth_select_fraction,
                growth_stop_iter: args.litegs_growth_stop_iter,
            },
            refine: rustgs::LiteGsRefineConfig {
                opacity_decay: args.litegs_opacity_decay,
                scale_decay: args.litegs_scale_decay,
            },
            pruning: rustgs::LiteGsPruningConfig {
                prune_mode: args.litegs_prune_mode,
                prune_offset_epochs: args.litegs_prune_offset_epochs,
                prune_min_age: args.litegs_prune_min_age,
                prune_invisible_epochs: args.litegs_prune_invisible_epochs,
                prune_opacity_threshold: args.litegs_prune_opacity_threshold,
                prune_visibility_dry_run: args.litegs_prune_visibility_dry_run,
                prune_visibility_threshold: args.litegs_prune_visibility_threshold,
                prune_high_opacity_threshold: args.litegs_prune_high_opacity_threshold,
                prune_until_epoch: args.litegs_prune_until_epoch,
                prune_scale_threshold: args.litegs_prune_scale_threshold,
            },
            camera: rustgs::LiteGsCameraConfig {
                learnable_viewproj: args.litegs_learnable_viewproj,
                lr_pose: args.litegs_lr_pose,
            },
        },
        ..rustgs::TrainingConfig::default()
    };
    config.validate()?;

    Ok(config)
}

fn litegs_profile_overrides(args: &TrainArgs) -> (rustgs::LiteGsSplitScoreMode, f32, f32) {
    match args.litegs_profile {
        rustgs::LiteGsTrainingProfile::Baseline => (
            args.litegs_split_score,
            args.litegs_split_grad_threshold,
            args.litegs_depth_scale_gamma,
        ),
        rustgs::LiteGsTrainingProfile::AbsSplit => (
            rustgs::LiteGsSplitScoreMode::Abs,
            0.00001,
            args.litegs_depth_scale_gamma,
        ),
        rustgs::LiteGsTrainingProfile::AbsPixel => (
            rustgs::LiteGsSplitScoreMode::AbsPixel,
            0.00001,
            args.litegs_depth_scale_gamma,
        ),
        rustgs::LiteGsTrainingProfile::AbsPixelDepth => (
            rustgs::LiteGsSplitScoreMode::AbsPixelDepth,
            0.00001,
            args.litegs_depth_scale_gamma,
        ),
    }
}

fn log_litegs_training_config(config: &rustgs::TrainingConfig) {
    log::info!(
        "LiteGS profile config | profile={} | sh_degree={} | init(point_scale={:.3}, point_opacity={:.3}, vksplat_scale={}, random_rot={}, rotation_seed={}) | tile_size={} | sparse_grad={} | reg_weight={:.4} | enable_transmittance={} | enable_depth={} | learnable_viewproj={} | lr_pose={:.6} | densify_from={} | densify_until={:?} | topology_freeze_after_epoch={:?} | growth_freeze_after_epoch={:?} | refine_every={} | densification_interval={} | growth_grad_threshold={:.6} | split_score={} | split_grad_threshold={:.6} | depth_scale_gamma={:.3} | growth_select_fraction={:.3} | growth_stop_iter={} | opacity_decay={:.6} | scale_decay={:.6} | opacity_reset_interval={} | opacity_reset_mode={} | prune_mode={} | prune_opacity_threshold={:.6} | prune_visibility_dry_run={} | prune_visibility_threshold={:.3} | prune_high_opacity_threshold={:.3} | prune_until_epoch={:?} | target_primitives={} | lr_decay_iterations={:?} | lr_position_scene_scale={} | lr_final(scale={:.6}, rot={:.6}, opacity={:.6}, color={:.6}, color_rest={:.6}) | raster_cov_blur={:.3} | raster_cov_blur_final={:?} | raster_cov_blur_final_after_epoch={:?} | loss_weights(l1={:.3}, ssim={:.3}, gradient={:.3}, robust_delta={:.3}, outlier_threshold={:.3}, outlier_weight={:.3}, dynamic_mask_low={:.3}, dynamic_mask_high={:.3}, dynamic_mask_min_weight={:.3}, dynamic_mask_start_epoch={:?})",
        config.litegs.features.training_profile,
        config.litegs.rendering.sh_degree,
        config.initialization.point_scale_factor,
        config.initialization.point_opacity,
        config.initialization.vksplat_scale_estimator,
        config.initialization.randomize_rotations,
        config.initialization.rotation_seed,
        config.litegs.rendering.tile_size,
        config.litegs.features.sparse_grad,
        config.litegs.features.reg_weight,
        config.litegs.features.enable_transmittance,
        config.litegs.features.enable_depth,
        config.litegs.camera.learnable_viewproj,
        config.litegs.camera.lr_pose,
        config.litegs.topology.densify_from,
        config.litegs.topology.densify_until,
        config.litegs.topology.topology_freeze_after_epoch,
        config.litegs.topology.growth_freeze_after_epoch,
        config.litegs.topology.refine_every,
        config.litegs.topology.densification_interval,
        config.litegs.growth.growth_grad_threshold,
        config.litegs.growth.split_score_mode,
        config.litegs.growth.split_grad_threshold,
        config.litegs.growth.depth_scale_gamma,
        config.litegs.growth.growth_select_fraction,
        config.litegs.growth.growth_stop_iter,
        config.litegs.refine.opacity_decay,
        config.litegs.refine.scale_decay,
        config.litegs.topology.opacity_reset_interval,
        config.litegs.topology.opacity_reset_mode,
        config.litegs.pruning.prune_mode,
        config.litegs.pruning.prune_opacity_threshold,
        config.litegs.pruning.prune_visibility_dry_run,
        config.litegs.pruning.prune_visibility_threshold,
        config.litegs.pruning.prune_high_opacity_threshold,
        config.litegs.pruning.prune_until_epoch,
        config.litegs.topology.target_primitives,
        config.optimizer.lr_decay_iterations,
        config.optimizer.lr_position_scene_scale,
        config.optimizer.lr_scale_final,
        config.optimizer.lr_rotation_final,
        config.optimizer.lr_opacity_final,
        config.optimizer.lr_color_final,
        config.optimizer.lr_color_rest_final,
        config.raster.raster_cov_blur,
        config.raster.raster_cov_blur_final,
        config.raster.raster_cov_blur_final.and_then(|_| {
            config
                .raster
                .raster_cov_blur_final_after_epoch
                .or(config.litegs.topology.topology_freeze_after_epoch)
        }),
        config.loss.loss_l1_weight,
        config.loss.loss_ssim_weight,
        config.loss.loss_gradient_weight,
        config.loss.loss_robust_delta,
        config.loss.loss_outlier_threshold,
        config.loss.loss_outlier_weight,
        config.loss.loss_dynamic_mask_threshold_low,
        config.loss.loss_dynamic_mask_threshold_high,
        config.loss.loss_dynamic_mask_min_weight,
        config.loss.loss_dynamic_mask_start_epoch,
    );
}

fn ensure_sparse_initialization_points(
    dataset: &rustscan_types::TrainingDataset,
    source: rustgs::TrainingInputKind,
    input: &Path,
) -> anyhow::Result<()> {
    if !dataset.initial_points.is_empty() {
        return Ok(());
    }

    let source_hint =
        "the COLMAP input is missing sparse points3D output; make sure points3D.bin or points3D.txt exists";

    bail!(
        "training initialization now requires COLMAP sparse points, but {:?} ({}) provided none: {}",
        input,
        source,
        source_hint
    );
}

pub(super) fn maybe_write_litegs_parity_report(
    input: &Path,
    output: &Path,
    dataset: &rustscan_types::TrainingDataset,
    splats: &rustgs::HostSplats,
    metadata: &rustgs::SplatMetadata,
    config: &rustgs::TrainingConfig,
    training_telemetry: Option<&rustgs::LiteGsTrainingTelemetry>,
    training_loop_elapsed: Duration,
    total_training_elapsed: Duration,
    evaluation_summary: Option<&rustgs::SplatEvaluationSummary>,
) -> anyhow::Result<()> {
    maybe_write_litegs_parity_report_with_manifest_dir(
        input,
        output,
        dataset,
        splats,
        metadata,
        config,
        training_telemetry,
        training_loop_elapsed,
        total_training_elapsed,
        evaluation_summary,
        Path::new(env!("CARGO_MANIFEST_DIR")),
    )
}

pub(super) fn maybe_write_litegs_parity_report_with_manifest_dir(
    input: &Path,
    output: &Path,
    dataset: &rustscan_types::TrainingDataset,
    splats: &rustgs::HostSplats,
    metadata: &rustgs::SplatMetadata,
    config: &rustgs::TrainingConfig,
    training_telemetry: Option<&rustgs::LiteGsTrainingTelemetry>,
    training_loop_elapsed: Duration,
    total_training_elapsed: Duration,
    evaluation_summary: Option<&rustgs::SplatEvaluationSummary>,
    manifest_dir: &Path,
) -> anyhow::Result<()> {
    let report_path = rustgs::default_parity_report_path(output);
    let fixture_id = rustgs::parity_fixture_id_for_input_path(input);
    let mut report = rustgs::ParityHarnessReport::new(fixture_id, &config.litegs);

    report.topology.initialization_gaussians =
        inferred_initialization_gaussian_count(dataset, config);
    report.topology.final_gaussians = Some(splats.len());
    report.topology.export_outputs = 1;

    if let Some(telemetry) = training_telemetry {
        report.loss_terms = telemetry.loss_terms.clone();
        report.loss_curve_samples = telemetry.loss_curve_samples.clone();
        report.topology = telemetry.topology.clone();
        report.topology.initialization_gaussians = report
            .topology
            .initialization_gaussians
            .or_else(|| inferred_initialization_gaussian_count(dataset, config));
        report.topology.final_gaussians = report.topology.final_gaussians.or(Some(splats.len()));
        report.topology.export_outputs = 1;
        report.metrics.active_sh_degree = telemetry.active_sh_degree;
        report.metrics.depth_valid_pixels = telemetry.depth_valid_pixels;
        report.metrics.depth_grad_scale = telemetry.depth_grad_scale;
        report.metrics.rotation_frozen = Some(telemetry.rotation_frozen);
    } else {
        report.metrics.active_sh_degree = Some(config.litegs.rendering.sh_degree);
    }
    if let Some(summary) = evaluation_summary {
        report.metrics.final_psnr = Some(summary.psnr_mean_db);
        report.notes.push(format!(
            "Evaluation summary recorded with device={}, render_scale={:.3}, raster_cov_blur={:.3}, frame_stride={}, max_frames={}, frame_count={}, mean PSNR {:.4} dB, grad sharpness ratio {:.4}, and lap sharpness ratio {:.4}.",
            summary.device,
            summary.render_scale,
            summary.raster_cov_blur,
            summary.frame_stride,
            summary.max_frames,
            summary.frame_count,
            summary.psnr_mean_db,
            summary.sharpness_grad_ratio_mean,
            summary.sharpness_lap_ratio_mean,
        ));
    }
    report.metrics.had_nan = splats_have_non_finite(splats);
    report.metrics.had_oom = false;

    report.timing.training_ms = Some(training_loop_elapsed.as_millis() as u64);
    report.timing.total_wall_clock_ms = Some(total_training_elapsed.as_millis() as u64);
    report.timing.setup_ms = total_training_elapsed
        .checked_sub(training_loop_elapsed)
        .map(|elapsed| elapsed.as_millis() as u64);

    report.notes.push(
        "LiteGsMacV1 now evaluates the active SH degree for view-dependent color during wgpu training and can apply rotation-aware projection gradients when rotation learning is enabled."
            .to_string(),
    );
    report.notes.push(
        "Timing training_ms records only the wgpu training loop; total_wall_clock_ms also includes RustGS initialization, upload, and final readback."
            .to_string(),
    );
    if training_telemetry.is_none() {
        report.notes.push(
            "Wgpu training telemetry was unavailable for this run, so the parity report fell back to config-level LiteGS metadata."
                .to_string(),
        );
    }

    let (roundtrip_splats, roundtrip_metadata) = rustgs::load_splats(output)?;
    report.metrics.export_roundtrip_ok = matches!(
        rustgs::splat_artifact_fidelity(output)?,
        rustgs::SplatArtifactFidelity::Lossless
    ) && rustgs::verify_lossless_roundtrip(
        splats,
        metadata,
        &roundtrip_splats,
        &roundtrip_metadata,
    )
    .is_ok();
    if matches!(
        rustgs::splat_artifact_fidelity(output)?,
        rustgs::SplatArtifactFidelity::LossyLegacy
    ) {
        report.notes.push(
            "The legacy .splat export is intentionally lossy, so it cannot satisfy the lossless export round-trip gate."
                .to_string(),
        );
    }

    if let Some(reference_report_path) =
        resolve_parity_reference_report_path_from_manifest_dir(&report.fixture_id, manifest_dir)
    {
        match rustgs::ParityHarnessReport::load_json(&reference_report_path) {
            Ok(reference_report) => {
                report.metrics.litegs_reference_psnr = reference_report.metrics.final_psnr;
                report.metrics.gaussian_count_delta_ratio = gaussian_count_delta_ratio(
                    report.topology.final_gaussians,
                    reference_report.topology.final_gaussians,
                );
                report.reference_comparison = rustgs::compare_loss_curve_samples(
                    &report.loss_curve_samples,
                    &reference_report.loss_curve_samples,
                );
                report.notes.push(format!(
                    "Compared parity loss curve samples against reference report at {}.",
                    reference_report_path.display()
                ));
            }
            Err(err) => {
                log::warn!(
                    "failed to load LiteGS parity reference report {:?}: {}",
                    reference_report_path,
                    err
                );
            }
        }
    } else if report.fixture_id == rustgs::DEFAULT_CONVERGENCE_FIXTURE_ID {
        report.notes.push(
            "No checked-in LiteGS parity reference report was found for the convergence fixture, so gate evaluation is reference-blocked."
                .to_string(),
        );
    }

    report.gate = Some(report.evaluate_gate());
    report.save_json(&report_path)?;
    if let Some(gate) = report.gate.as_ref() {
        log::info!(
            "Saved LiteGS parity report to {:?} | gate_status={:?}",
            report_path,
            gate.status
        );
    } else {
        log::info!("Saved LiteGS parity report to {:?}", report_path);
    }
    Ok(())
}

fn resolve_parity_reference_report_path_from_manifest_dir(
    fixture_id: &str,
    manifest_dir: &Path,
) -> Option<PathBuf> {
    manifest_dir
        .ancestors()
        .find_map(|path| rustgs::resolve_litegs_parity_reference_report_path(fixture_id, path))
}

fn inferred_initialization_gaussian_count(
    dataset: &rustscan_types::TrainingDataset,
    _config: &rustgs::TrainingConfig,
) -> Option<usize> {
    let sparse_points = dataset.initial_points.len();
    if sparse_points == 0 {
        None
    } else {
        Some(sparse_points)
    }
}

fn gaussian_count_delta_ratio(current: Option<usize>, reference: Option<usize>) -> Option<f32> {
    match (current, reference) {
        (Some(current), Some(reference)) if reference > 0 => {
            Some(((current as f32) - (reference as f32)).abs() / reference as f32)
        }
        _ => None,
    }
}

#[cfg(feature = "gpu")]
fn splats_have_non_finite(splats: &rustgs::HostSplats) -> bool {
    let view = splats.as_view();
    view.positions.iter().any(|value| !value.is_finite())
        || view.log_scales.iter().any(|value| !value.is_finite())
        || view.rotations.iter().any(|value| !value.is_finite())
        || view.opacity_logits.iter().any(|value| !value.is_finite())
        || view.sh_coeffs.iter().any(|value| !value.is_finite())
}
