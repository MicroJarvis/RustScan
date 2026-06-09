//! RustGS CLI - 3D Gaussian Splatting Training
//!
//! Usage:
//!   rustgs train --input <colmap_dir> --output <scene.splat>
//!   rustgs render --input <scene.splat|scene.ply> --camera <pose.json> --output <image.png>

use anyhow::{bail, Context};
use clap::{CommandFactory, FromArgMatches};
#[cfg(feature = "gpu")]
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[path = "rustgs/train_command.rs"]
mod train_command;

#[derive(Debug, clap::Parser)]
#[command(name = "rustgs")]
#[command(about = "3D Gaussian Splatting Training", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Clone, clap::Args)]
struct TrainArgs {
    /// Path to a COLMAP dataset directory with sparse reconstruction points
    #[arg(short, long)]
    input: PathBuf,

    /// Output path for trained splats (.splat or .ply)
    #[arg(short, long)]
    output: PathBuf,

    /// Path to a JSON training config file. Defaults to <input>/rustgs-train.json when present.
    #[arg(long)]
    train_config: Option<PathBuf>,

    /// Training preset name loaded from the dataset training config file
    #[arg(long)]
    train_preset: Option<String>,

    /// Number of training iterations
    #[arg(long, default_value = "30000")]
    iterations: usize,

    /// Maximum number of Gaussians created during initialization
    #[arg(long, default_value = "100000")]
    max_initial_gaussians: usize,

    /// Deprecated: frame/depth initialization is disabled, so this flag is ignored
    #[arg(long, default_value = "0")]
    sampling_step: usize,

    /// Maximum number of input RGB frames to consider before applying --frame-stride (0 = all)
    #[arg(long, default_value = "0")]
    max_frames: usize,

    /// Keep every Nth frame within the considered prefix, e.g. 25 frames with stride 5 yields 5 training frames
    #[arg(long, default_value = "1")]
    frame_stride: usize,

    /// Comma-separated frame ids or inclusive ranges to include for training, e.g. 0-179,240-299
    #[arg(long)]
    include_frame_ranges: Option<String>,

    /// Comma-separated frame ids or inclusive ranges to exclude from training, e.g. 76-93,155
    #[arg(long)]
    exclude_frame_ranges: Option<String>,

    /// Comma-separated frame ids or inclusive ranges to repeat during training, e.g. 0-179
    #[arg(long)]
    oversample_frame_ranges: Option<String>,

    /// Total repeat count for --oversample-frame-ranges (1 = disabled)
    #[arg(long, default_value = "1")]
    oversample_frame_repeat: usize,

    /// Deterministic shuffle seed for training frame sampling (0 preserves dataset order)
    #[arg(long, default_value = "0")]
    frame_shuffle_seed: u64,

    /// Relative render scale used by training
    #[arg(long, default_value = "0.5")]
    render_scale: f32,

    /// Rasterizer 2D covariance blur floor. Lower is sharper but can alias.
    #[arg(long, default_value_t = rustgs::DEFAULT_RASTER_COV_BLUR)]
    raster_cov_blur: f32,

    /// Optional late-training covariance blur floor. Defaults to disabled.
    #[arg(long)]
    raster_cov_blur_final: Option<f32>,

    /// Epoch at which --raster-cov-blur-final becomes active. Defaults to topology freeze epoch.
    #[arg(long)]
    raster_cov_blur_final_after_epoch: Option<usize>,

    /// LiteGS SH degree
    #[arg(long, default_value = "3")]
    litegs_sh_degree: usize,

    /// LiteGS experiment profile: baseline, abs-split, abs-pixel, abs-pixel-depth
    #[arg(long, default_value_t = rustgs::LiteGsTrainingProfile::Baseline)]
    litegs_profile: rustgs::LiteGsTrainingProfile,

    /// LiteGS tile size, e.g. 8x16 or 8,16
    #[arg(long, default_value_t = rustgs::LiteGsTileSize::new(8, 16))]
    litegs_tile_size: rustgs::LiteGsTileSize,

    /// Enable LiteGS sparse-gradient optimizer semantics
    #[arg(long, default_value_t = false)]
    litegs_sparse_grad: bool,

    /// LiteGS scale regularization weight
    #[arg(long, default_value = "0.0")]
    litegs_reg_weight: f32,

    /// Enable LiteGS transmittance penalty
    #[arg(long, default_value_t = false)]
    litegs_enable_transmittance: bool,

    /// Enable LiteGS depth supervision path
    #[arg(long, default_value_t = false)]
    litegs_enable_depth: bool,

    /// LiteGS densify start epoch
    #[arg(long, default_value = "3")]
    litegs_densify_from: usize,

    /// LiteGS densify end epoch (omit to keep auto behavior)
    #[arg(long)]
    litegs_densify_until: Option<usize>,

    /// Freeze all LiteGS topology updates at or after this epoch
    #[arg(long)]
    litegs_topology_freeze_after_epoch: Option<usize>,

    /// Freeze LiteGS growth/densification at or after this epoch while allowing pruning to continue
    #[arg(long)]
    litegs_growth_freeze_after_epoch: Option<usize>,

    /// LiteGS densification interval
    #[arg(long, default_value = "5")]
    litegs_densification_interval: usize,

    /// LiteGS refine cadence in training iterations
    #[arg(long, default_value = "160")]
    litegs_refine_every: usize,

    /// LiteGS xy-gradient threshold for growth candidates
    #[arg(long, default_value = "0.00014")]
    litegs_growth_grad_threshold: f32,

    /// LiteGS split score mode for large Gaussians: baseline, abs, abs-pixel, abs-pixel-depth
    #[arg(long, default_value_t = rustgs::LiteGsSplitScoreMode::Baseline)]
    litegs_split_score: rustgs::LiteGsSplitScoreMode,

    /// LiteGS split score threshold used by --litegs-split-score abs
    #[arg(long, default_value = "0.00014")]
    litegs_split_grad_threshold: f32,

    /// Pixel-GS gamma for abs-pixel-depth near-camera split suppression
    #[arg(long, default_value = "0.37")]
    litegs_depth_scale_gamma: f32,

    /// LiteGS fraction of above-threshold candidates selected for extra growth
    #[arg(long, default_value = "0.25")]
    litegs_growth_select_fraction: f32,

    /// LiteGS iteration after which extra growth stops
    #[arg(long, default_value = "15000")]
    litegs_growth_stop_iter: usize,

    /// LiteGS opacity decay applied after each refine step
    #[arg(long, default_value = "0.0")]
    litegs_opacity_decay: f32,

    /// LiteGS scale decay applied after each refine step
    #[arg(long, default_value = "0.0")]
    litegs_scale_decay: f32,

    /// LiteGS opacity reset interval
    #[arg(long, default_value = "10")]
    litegs_opacity_reset_interval: usize,

    /// LiteGS opacity reset mode
    #[arg(long, default_value_t = rustgs::LiteGsOpacityResetMode::Decay)]
    litegs_opacity_reset_mode: rustgs::LiteGsOpacityResetMode,

    /// LiteGS prune mode
    #[arg(long, default_value_t = rustgs::LiteGsPruneMode::Weight)]
    litegs_prune_mode: rustgs::LiteGsPruneMode,

    /// LiteGS prune offset epochs - how many epochs to wait after densify before pruning
    #[arg(long, default_value = "0")]
    litegs_prune_offset_epochs: usize,

    /// LiteGS prune minimum age - minimum iterations before Gaussian is prune-eligible
    #[arg(long, default_value = "5")]
    litegs_prune_min_age: usize,

    /// LiteGS prune invisible epochs - consecutive invisibility required before pruning
    #[arg(long, default_value = "10")]
    litegs_prune_invisible_epochs: usize,

    /// LiteGS opacity threshold for pruning
    #[arg(long, default_value = "0.0039215689")]
    litegs_prune_opacity_threshold: f32,

    /// Report visibility-aware prune candidates without pruning them
    #[arg(long, default_value_t = false)]
    litegs_prune_visibility_dry_run: bool,

    /// Actual visibility ratio below which a Gaussian is considered low-visibility
    #[arg(long, default_value = "0.05")]
    litegs_prune_visibility_threshold: f32,

    /// Opacity ceiling for conservative visibility-prune dry-run candidates
    #[arg(long, default_value = "0.80")]
    litegs_prune_high_opacity_threshold: f32,

    /// Continue LiteGS pruning before this epoch even if growth is frozen
    #[arg(long)]
    litegs_prune_until_epoch: Option<usize>,

    /// LiteGS target primitive budget
    #[arg(long, default_value = "1000000")]
    litegs_target_primitives: usize,

    /// Enable LiteGS learnable camera extrinsics
    #[arg(long, default_value_t = false)]
    litegs_learnable_viewproj: bool,

    /// LiteGS pose learning rate
    #[arg(long, default_value = "0.0001")]
    litegs_lr_pose: f32,

    /// Prune Gaussians with max scale > threshold (0 disables scale pruning)
    #[arg(long, default_value = "0.5")]
    litegs_prune_scale_threshold: f32,

    /// Position learning rate (initial)
    #[arg(long, default_value = "0.00016")]
    lr_position: f32,

    /// Position learning rate (final) - exponential decay target
    #[arg(long, default_value = "0.0000016")]
    lr_position_final: f32,

    /// Learning-rate decay horizon in iterations (0 = use total iterations)
    #[arg(long, default_value = "0")]
    lr_decay_iterations: usize,

    /// Scale learning rate
    #[arg(long, default_value = "0.005")]
    lr_scale: f32,

    /// Scale learning rate final value (0 = keep scale LR constant)
    #[arg(long, default_value = "0")]
    lr_scale_final: f32,

    /// Rotation learning rate
    #[arg(long, default_value = "0.001")]
    lr_rotation: f32,

    /// Rotation learning rate final value (0 = keep rotation LR constant)
    #[arg(long, default_value = "0")]
    lr_rotation_final: f32,

    /// Opacity learning rate
    #[arg(long, default_value = "0.05")]
    lr_opacity: f32,

    /// Opacity learning rate final value (0 = keep opacity LR constant)
    #[arg(long, default_value = "0")]
    lr_opacity_final: f32,

    /// Color/SH learning rate
    #[arg(long, default_value = "0.0025")]
    lr_color: f32,

    /// Color/SH learning rate final value (0 = keep color LR constant)
    #[arg(long, default_value = "0")]
    lr_color_final: f32,

    /// L1 image reconstruction loss weight
    #[arg(long, default_value = "0.8")]
    loss_l1_weight: f32,

    /// D-SSIM image reconstruction loss weight
    #[arg(long, default_value = "0.2")]
    loss_ssim_weight: f32,

    /// Image-gradient reconstruction loss weight for edge/detail preservation
    #[arg(long, default_value = "0.0")]
    loss_gradient_weight: f32,

    /// Robust residual delta for saturating L1 reconstruction loss (0 = exact L1)
    #[arg(long, default_value = "0.0")]
    loss_robust_delta: f32,

    /// Soft outlier residual threshold for high-residual pixel downweighting (0 = disabled)
    #[arg(long, default_value = "0.0")]
    loss_outlier_threshold: f32,

    /// Gradient floor for high-residual pixels when --loss-outlier-threshold is enabled
    #[arg(long, default_value = "1.0")]
    loss_outlier_weight: f32,

    /// Low residual threshold for late-training dynamic/occlusion soft masking (0 = disabled)
    #[arg(long, default_value = "0.0")]
    loss_dynamic_mask_threshold_low: f32,

    /// High residual threshold for late-training dynamic/occlusion soft masking (0 = disabled)
    #[arg(long, default_value = "0.0")]
    loss_dynamic_mask_threshold_high: f32,

    /// Minimum L1 weight for high-residual dynamic/occlusion mask pixels
    #[arg(long, default_value = "1.0")]
    loss_dynamic_mask_min_weight: f32,

    /// Epoch when dynamic/occlusion masking starts; defaults to topology freeze epoch
    #[arg(long)]
    loss_dynamic_mask_start_epoch: Option<usize>,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Run a post-training scene evaluation pass and emit a structured summary
    #[arg(long, default_value_t = false)]
    eval_after_train: bool,

    /// Evaluation render scale used by the post-training scene evaluation
    #[arg(long, default_value = "0.25")]
    eval_render_scale: f32,

    /// Evaluation-only rasterizer covariance blur floor. Defaults to --raster-cov-blur.
    #[arg(long)]
    eval_raster_cov_blur: Option<f32>,

    /// Maximum number of frames considered during post-training evaluation (0 = all)
    #[arg(long, default_value = "180")]
    eval_max_frames: usize,

    /// Keep every Nth frame during post-training evaluation
    #[arg(long, default_value = "30")]
    eval_frame_stride: usize,

    /// Comma-separated frame ids or inclusive ranges to include during post-training evaluation
    #[arg(long)]
    eval_include_frame_ranges: Option<String>,

    /// Comma-separated frame ids or inclusive ranges to exclude from post-training evaluation
    #[arg(long)]
    eval_exclude_frame_ranges: Option<String>,

    /// Number of lowest-PSNR frames to keep in the evaluation summary
    #[arg(long, default_value = "5")]
    eval_worst_frames: usize,

    /// Evaluation device used by the post-training scene evaluation
    #[arg(long, default_value = "gpu")]
    eval_device: String,

    /// Print the post-training evaluation summary as JSON
    #[arg(long, default_value_t = false)]
    eval_json: bool,

    /// Directory for post-training target/render/diff crop exports
    #[arg(long)]
    eval_crop_output_dir: Option<PathBuf>,

    /// Comma-separated frame ids to export crops for; defaults to worst evaluated frames
    #[arg(long)]
    eval_crop_frames: Option<String>,

    /// Crop rectangle in evaluation pixels as x,y,width,height; defaults to full frame
    #[arg(long)]
    eval_crop_rect: Option<String>,
}

#[derive(Debug, Clone, clap::Args)]
struct RenderArgs {
    /// Path to scene .splat or PLY file
    #[arg(short, long)]
    input: PathBuf,

    /// Path to camera pose JSON file
    #[arg(short, long)]
    camera: PathBuf,

    /// Output image path
    #[arg(short, long)]
    output: PathBuf,

    /// Rasterizer 2D covariance blur floor. Lower is sharper but can alias.
    #[arg(long, default_value_t = rustgs::DEFAULT_RASTER_COV_BLUR)]
    raster_cov_blur: f32,
}

#[derive(Debug, Clone, clap::Args)]
struct PruneSceneArgs {
    /// Path to input scene .splat or PLY file
    #[arg(short, long)]
    input: PathBuf,

    /// Output path for pruned scene (.splat or .ply)
    #[arg(short, long)]
    output: PathBuf,

    /// Remove Gaussians with opacity below this value
    #[arg(long, default_value = "0.0")]
    min_opacity: f32,

    /// Remove Gaussians with any scale below this value (0 disables)
    #[arg(long, default_value = "0.0")]
    min_scale: f32,

    /// Remove Gaussians with any scale above this value (0 disables)
    #[arg(long, default_value = "0.0")]
    max_scale: f32,

    /// Remove Gaussians farther than this absolute distance from the scene center (0 disables)
    #[arg(long, default_value = "0.0")]
    max_distance_from_center: f32,

    /// Remove Gaussians farther than scene_extent * multiplier from the scene center (0 disables)
    #[arg(long, default_value = "0.0")]
    max_distance_extent_multiplier: f32,

    /// Print summary without writing output
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    /// Print pruning summary as JSON
    #[arg(long, default_value_t = false)]
    json: bool,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Debug, clap::Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Commands {
    /// Train a 3DGS scene from a COLMAP dataset directory
    Train(TrainArgs),

    /// Render a scene from a given viewpoint
    Render(RenderArgs),

    /// Remove low-quality splats from an existing scene .splat or PLY
    PruneScene(PruneSceneArgs),
}

fn main() -> anyhow::Result<()> {
    let matches = <Cli as CommandFactory>::command().get_matches();
    let sources = train_command::TrainArgSources::from_cli_matches(&matches);
    let cli = <Cli as FromArgMatches>::from_arg_matches(&matches)?;

    match cli.command {
        Commands::Train(args) => train_command::run_train_command(args, sources)?,
        Commands::Render(args) => run_render_command(args)?,
        Commands::PruneScene(args) => run_prune_scene_command(args)?,
    }

    Ok(())
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, Default, Serialize)]
struct PruneReasonCounts {
    invalid: usize,
    opacity: usize,
    min_scale: usize,
    max_scale: usize,
    distance: usize,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Serialize)]
struct PruneSceneSummary {
    input_gaussians: usize,
    output_gaussians: usize,
    removed_gaussians: usize,
    scene_center: [f32; 3],
    scene_extent: f32,
    max_distance_threshold: Option<f32>,
    min_opacity: f32,
    min_scale: Option<f32>,
    max_scale: Option<f32>,
    reasons: PruneReasonCounts,
}

#[cfg(feature = "gpu")]
fn run_prune_scene_command(args: PruneSceneArgs) -> anyhow::Result<()> {
    let _ = env_logger::Builder::new()
        .parse_filters(&args.log_level)
        .try_init();
    let (splats, mut metadata) = rustgs::load_splats(&args.input)?;
    let (pruned, summary) = prune_splats(&splats, &args)?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!(
            "Pruned scene: kept {} / {} Gaussians, removed {}",
            summary.output_gaussians, summary.input_gaussians, summary.removed_gaussians
        );
    }

    if args.dry_run {
        return Ok(());
    }

    if pruned.is_empty() {
        bail!("pruning removed every Gaussian; refusing to write empty scene");
    }
    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    metadata.gaussian_count = pruned.len();
    metadata.sh_degree = pruned.sh_degree();
    rustgs::save_splats(&args.output, &pruned, &metadata)?;
    log::info!("Saved pruned scene to {:?}", args.output);
    Ok(())
}

#[cfg(feature = "gpu")]
fn prune_splats(
    splats: &rustgs::HostSplats,
    args: &PruneSceneArgs,
) -> anyhow::Result<(rustgs::HostSplats, PruneSceneSummary)> {
    if !args.min_opacity.is_finite() || !(0.0..=1.0).contains(&args.min_opacity) {
        bail!("--min-opacity must be finite and in [0, 1]");
    }
    validate_non_negative("min-scale", args.min_scale)?;
    validate_non_negative("max-scale", args.max_scale)?;
    validate_non_negative("max-distance-from-center", args.max_distance_from_center)?;
    validate_non_negative(
        "max-distance-extent-multiplier",
        args.max_distance_extent_multiplier,
    )?;

    let view = splats.as_view();
    let row_count = splats.len();
    let sh_row_width = ((view.sh_degree + 1) * (view.sh_degree + 1)) * 3;
    let (center, extent) = scene_center_and_extent(&view);
    let max_distance_threshold = if args.max_distance_from_center > 0.0 {
        Some(args.max_distance_from_center)
    } else if args.max_distance_extent_multiplier > 0.0 {
        Some(extent * args.max_distance_extent_multiplier)
    } else {
        None
    };

    let mut kept_positions = Vec::with_capacity(view.positions.len());
    let mut kept_log_scales = Vec::with_capacity(view.log_scales.len());
    let mut kept_rotations = Vec::with_capacity(view.rotations.len());
    let mut kept_opacities = Vec::with_capacity(view.opacity_logits.len());
    let mut kept_sh_coeffs = Vec::with_capacity(view.sh_coeffs.len());
    let mut reasons = PruneReasonCounts::default();

    for idx in 0..row_count {
        let pos_base = idx * 3;
        let rot_base = idx * 4;
        let sh_base = idx * sh_row_width;
        let position = [
            view.positions[pos_base],
            view.positions[pos_base + 1],
            view.positions[pos_base + 2],
        ];
        let log_scale = [
            view.log_scales[pos_base],
            view.log_scales[pos_base + 1],
            view.log_scales[pos_base + 2],
        ];
        let rotation = [
            view.rotations[rot_base],
            view.rotations[rot_base + 1],
            view.rotations[rot_base + 2],
            view.rotations[rot_base + 3],
        ];
        let opacity_logit = view.opacity_logits[idx];
        let scale = log_scale.map(f32::exp);

        if position.iter().any(|value| !value.is_finite())
            || log_scale.iter().any(|value| !value.is_finite())
            || rotation.iter().any(|value| !value.is_finite())
            || !opacity_logit.is_finite()
            || scale.iter().any(|value| !value.is_finite())
        {
            reasons.invalid += 1;
            continue;
        }
        if sigmoid(opacity_logit) < args.min_opacity {
            reasons.opacity += 1;
            continue;
        }
        if args.min_scale > 0.0 && scale.iter().any(|value| *value < args.min_scale) {
            reasons.min_scale += 1;
            continue;
        }
        if args.max_scale > 0.0 && scale.iter().any(|value| *value > args.max_scale) {
            reasons.max_scale += 1;
            continue;
        }
        if let Some(max_distance) = max_distance_threshold {
            if distance(position, center) > max_distance {
                reasons.distance += 1;
                continue;
            }
        }

        kept_positions.extend_from_slice(&position);
        kept_log_scales.extend_from_slice(&log_scale);
        kept_rotations.extend_from_slice(&rotation);
        kept_opacities.push(opacity_logit);
        kept_sh_coeffs.extend_from_slice(&view.sh_coeffs[sh_base..sh_base + sh_row_width]);
    }

    let pruned = rustgs::HostSplats::from_components(
        kept_positions,
        kept_log_scales,
        kept_rotations,
        kept_opacities,
        kept_sh_coeffs,
        view.sh_degree,
    )?;
    let summary = PruneSceneSummary {
        input_gaussians: row_count,
        output_gaussians: pruned.len(),
        removed_gaussians: row_count.saturating_sub(pruned.len()),
        scene_center: center,
        scene_extent: extent,
        max_distance_threshold,
        min_opacity: args.min_opacity,
        min_scale: (args.min_scale > 0.0).then_some(args.min_scale),
        max_scale: (args.max_scale > 0.0).then_some(args.max_scale),
        reasons,
    };
    Ok((pruned, summary))
}

#[cfg(feature = "gpu")]
fn validate_non_negative(name: &str, value: f32) -> anyhow::Result<()> {
    if !value.is_finite() || value < 0.0 {
        bail!("--{name} must be finite and >= 0");
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(feature = "gpu")]
fn scene_center_and_extent(view: &rustgs::SplatView<'_>) -> ([f32; 3], f32) {
    let row_count = view.opacity_logits.len();
    if row_count == 0 {
        return ([0.0, 0.0, 0.0], 0.0);
    }
    let mut center = [0.0f32; 3];
    for idx in 0..row_count {
        let base = idx * 3;
        center[0] += view.positions[base];
        center[1] += view.positions[base + 1];
        center[2] += view.positions[base + 2];
    }
    let inv_count = 1.0 / row_count as f32;
    center[0] *= inv_count;
    center[1] *= inv_count;
    center[2] *= inv_count;

    let mut extent = 0.0f32;
    for idx in 0..row_count {
        let base = idx * 3;
        extent = extent.max(distance(
            [
                view.positions[base],
                view.positions[base + 1],
                view.positions[base + 2],
            ],
            center,
        ));
    }
    (center, extent.max(1e-6))
}

#[cfg(feature = "gpu")]
fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(not(feature = "gpu"))]
fn run_prune_scene_command(_args: PruneSceneArgs) -> anyhow::Result<()> {
    bail!("prune-scene requires the gpu feature");
}

#[cfg(feature = "gpu")]
fn run_render_command(args: RenderArgs) -> anyhow::Result<()> {
    if !args.raster_cov_blur.is_finite() || args.raster_cov_blur < 0.0 {
        bail!("--raster-cov-blur must be finite and >= 0");
    }
    log::info!("Rendering scene from {:?}", args.input);
    log::info!("Camera: {:?}", args.camera);
    log::info!("Output: {:?}", args.output);
    log::info!("Raster covariance blur: {:.3}", args.raster_cov_blur);

    let (splats, metadata) = rustgs::load_splats(&args.input)?;
    log::info!("Loaded {} Gaussians", splats.len());
    let _ = metadata;

    let camera = load_render_camera(&args.camera)?;
    let mut renderer = rustgs::SplatEvaluationRenderer::new(
        camera.intrinsics.width as usize,
        camera.intrinsics.height as usize,
        rustgs::EvaluationDevice::Gpu,
        args.raster_cov_blur,
    )
    .map_err(anyhow::Error::from)?;
    let rendered = renderer
        .render(&splats, &camera)
        .map_err(anyhow::Error::from)?;
    let color: Vec<u8> = rendered
        .into_iter()
        .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    let image = image::RgbImage::from_raw(camera.intrinsics.width, camera.intrinsics.height, color)
        .context("renderer produced an invalid RGB buffer")?;
    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    image.save(&args.output)?;
    log::info!(
        "Rendered {}x{} image to {:?}",
        camera.intrinsics.width,
        camera.intrinsics.height,
        args.output
    );
    Ok(())
}

#[cfg(feature = "gpu")]
#[derive(Debug, Deserialize)]
struct RenderCameraFile {
    intrinsics: rustgs::Intrinsics,
    pose: rustgs::SE3,
    #[serde(default)]
    pose_is_world_to_camera: bool,
}

#[cfg(feature = "gpu")]
fn load_render_camera(path: &std::path::Path) -> anyhow::Result<rustgs::GaussianCamera> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read camera JSON {}", path.display()))?;
    let camera: RenderCameraFile = serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse camera JSON {}", path.display()))?;
    let extrinsics = if camera.pose_is_world_to_camera {
        camera.pose
    } else {
        camera.pose.inverse()
    };
    Ok(rustgs::GaussianCamera::new(camera.intrinsics, extrinsics))
}

#[cfg(not(feature = "gpu"))]
fn run_render_command(_args: RenderArgs) -> anyhow::Result<()> {
    bail!("render requires the gpu feature");
}
