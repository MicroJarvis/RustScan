use anyhow::Result;
use clap::{Parser, Subcommand};
use rustsfm::{
    compare_colmap, compare_database_parity, run_reconstruction, FeatureType, MapperConfig,
};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "rustsfm")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Reconstruct(ReconstructArgs),
    Compare(CompareArgs),
    Parity(ParityArgs),
}

#[derive(Debug, Parser)]
struct ReconstructArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    reference: Option<PathBuf>,
    #[arg(long)]
    database: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    write_two_view_geometries: bool,
    #[arg(long)]
    max_images: Option<usize>,
    #[arg(long, default_value_t = false)]
    single_model: bool,
    #[arg(long, default_value = "50")]
    max_num_models: usize,
    #[arg(long, default_value = "20")]
    max_model_overlap: usize,
    #[arg(long, default_value = "10")]
    min_model_size: usize,
    #[arg(long)]
    snapshot_path: Option<PathBuf>,
    #[arg(long, default_value = "0")]
    snapshot_frames_freq: usize,
    #[arg(long, default_value_t = false)]
    no_extract_colors: bool,
    #[arg(long, default_value_t = false)]
    fix_existing_frames: bool,
    #[arg(long, default_value = "sift")]
    feature_type: FeatureType,
    #[arg(long, default_value = "8192")]
    max_features: usize,
    #[arg(long, default_value_t = false)]
    local_matching: bool,
    #[arg(long, default_value = "3")]
    local_window: usize,
    #[arg(long, default_value_t = false)]
    experimental_sequence_heuristics: bool,
    #[arg(long, default_value_t = false)]
    experimental_ring_closure: bool,
    #[arg(long, default_value_t = false)]
    experimental_structureless_pair_pose_fallback: bool,
    #[arg(long, default_value = "0.8")]
    match_ratio: f64,
    #[arg(long, default_value = "160")]
    max_hamming_distance: f32,
    #[arg(long, default_value = "15")]
    min_matches: usize,
    #[arg(long, default_value = "15")]
    min_inliers: usize,
    #[arg(long, default_value = "4")]
    min_triangulated: usize,
    #[arg(long, default_value = "100")]
    init_min_num_inliers: usize,
    #[arg(long, default_value = "16.0")]
    init_min_tri_angle_deg: f32,
    #[arg(long, default_value = "0.95")]
    init_max_forward_motion: f32,
    #[arg(long, default_value = "200")]
    init_num_trials: usize,
    #[arg(long, default_value = "2")]
    init_max_reg_trials: usize,
    #[arg(long, default_value = "2.0")]
    essential_threshold_px: f32,
    #[arg(long, default_value = "10000")]
    essential_iterations: u32,
    #[arg(long, default_value = "12.0")]
    pnp_threshold_px: f32,
    #[arg(long, default_value = "10000")]
    pnp_iterations: u32,
    #[arg(long, default_value = "30")]
    abs_pose_min_num_inliers: usize,
    #[arg(long, default_value = "0.25")]
    abs_pose_min_inlier_ratio: f32,
    #[arg(long, default_value = "-1")]
    random_seed: i32,
    #[arg(long, default_value = "3")]
    max_reg_trials: usize,
    #[arg(long, default_value_t = false)]
    no_local_ba: bool,
    #[arg(long, default_value = "6")]
    local_ba_num_images: usize,
    #[arg(long, default_value = "15")]
    local_ba_min_shared_points: usize,
    #[arg(long, default_value = "5")]
    local_ba_iterations: usize,
    #[arg(long, default_value_t = false)]
    no_global_ba: bool,
    #[arg(long, default_value = "8")]
    global_ba_iterations: usize,
    #[arg(long, default_value = "1.1")]
    global_ba_images_ratio: f32,
    #[arg(long, default_value = "1.1")]
    global_ba_points_ratio: f32,
    #[arg(long, default_value = "500")]
    global_ba_images_freq: usize,
    #[arg(long, default_value = "250000")]
    global_ba_points_freq: usize,
    #[arg(long, default_value = "5")]
    global_ba_max_refinements: usize,
    #[arg(long, default_value = "0.0005")]
    global_ba_max_refinement_change: f32,
    #[arg(long, default_value_t = false)]
    no_ba_refine_focal_length: bool,
    #[arg(long, default_value_t = false)]
    ba_refine_principal_point: bool,
    #[arg(long, default_value_t = false)]
    no_ba_refine_extra_params: bool,
    #[arg(long = "ba-constant-rig-id", value_delimiter = ',')]
    ba_constant_rig_ids: Vec<u32>,
    #[arg(long = "ba-constant-camera-id", value_delimiter = ',')]
    ba_constant_camera_ids: Vec<u32>,
    #[arg(long, default_value = "0.1")]
    min_focal_length_ratio: f64,
    #[arg(long, default_value = "10.0")]
    max_focal_length_ratio: f64,
    #[arg(long, default_value = "1.0")]
    max_extra_param: f64,
    #[arg(long, default_value = "8.0")]
    max_reprojection_error_px: f32,
    #[arg(long, default_value_t = false)]
    pose_graph: bool,
    #[arg(long)]
    fx: Option<f32>,
    #[arg(long)]
    fy: Option<f32>,
    #[arg(long)]
    cx: Option<f32>,
    #[arg(long)]
    cy: Option<f32>,
    #[arg(long)]
    threads: Option<usize>,
    #[arg(long, default_value_t = false)]
    no_copy_images: bool,
    #[arg(long)]
    summary_json: Option<PathBuf>,
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Debug, Parser)]
struct CompareArgs {
    #[arg(long)]
    reference: PathBuf,
    #[arg(long)]
    candidate: PathBuf,
    #[arg(long)]
    output_json: Option<PathBuf>,
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Debug, Parser)]
struct ParityArgs {
    #[arg(long)]
    database: PathBuf,
    #[arg(long)]
    images: Option<PathBuf>,
    #[arg(long = "image-name")]
    image_names: Vec<String>,
    #[arg(long, default_value = "15")]
    min_matches: usize,
    #[arg(long, default_value_t = false)]
    ignore_watermarks: bool,
    #[arg(long, default_value_t = false)]
    load_all_images: bool,
    #[arg(long, default_value_t = false)]
    convert_pose_priors_to_enu: bool,
    #[arg(long)]
    output_json: Option<PathBuf>,
    #[arg(long, default_value = "info")]
    log_level: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Reconstruct(args) => {
            env_logger::Builder::new()
                .parse_filters(&args.log_level)
                .init();
            let config = MapperConfig {
                input: args.input,
                output: args.output,
                reference: args.reference,
                database: args.database,
                write_two_view_geometries: args.write_two_view_geometries,
                max_images: args.max_images,
                multiple_models: !args.single_model,
                max_num_models: args.max_num_models,
                max_model_overlap: args.max_model_overlap,
                min_model_size: args.min_model_size,
                snapshot_path: args.snapshot_path,
                snapshot_frames_freq: args.snapshot_frames_freq,
                extract_colors: !args.no_extract_colors,
                fix_existing_frames: args.fix_existing_frames,
                feature_type: args.feature_type,
                max_features: args.max_features,
                match_ratio: args.match_ratio,
                max_hamming_distance: args.max_hamming_distance,
                local_matching: args.local_matching,
                local_window: args.local_window,
                experimental_sequence_heuristics: args.experimental_sequence_heuristics,
                experimental_ring_closure: args.experimental_ring_closure,
                experimental_structureless_pair_pose_fallback: args
                    .experimental_structureless_pair_pose_fallback,
                min_matches: args.min_matches,
                min_inliers: args.min_inliers,
                min_triangulated: args.min_triangulated,
                init_min_num_inliers: args.init_min_num_inliers,
                init_min_tri_angle_deg: args.init_min_tri_angle_deg,
                init_max_forward_motion: args.init_max_forward_motion,
                init_num_trials: args.init_num_trials,
                init_max_reg_trials: args.init_max_reg_trials,
                essential_threshold_px: args.essential_threshold_px,
                essential_iterations: args.essential_iterations,
                pnp_threshold_px: args.pnp_threshold_px,
                pnp_iterations: args.pnp_iterations,
                abs_pose_min_num_inliers: args.abs_pose_min_num_inliers,
                abs_pose_min_inlier_ratio: args.abs_pose_min_inlier_ratio,
                random_seed: args.random_seed,
                max_reg_trials: args.max_reg_trials,
                local_ba: !args.no_local_ba,
                local_ba_num_images: args.local_ba_num_images,
                local_ba_min_shared_points: args.local_ba_min_shared_points,
                local_ba_iterations: args.local_ba_iterations,
                global_ba: !args.no_global_ba,
                global_ba_iterations: args.global_ba_iterations,
                global_ba_images_ratio: args.global_ba_images_ratio,
                global_ba_points_ratio: args.global_ba_points_ratio,
                global_ba_images_freq: args.global_ba_images_freq,
                global_ba_points_freq: args.global_ba_points_freq,
                global_ba_max_refinements: args.global_ba_max_refinements,
                global_ba_max_refinement_change: args.global_ba_max_refinement_change,
                ba_refine_focal_length: !args.no_ba_refine_focal_length,
                ba_refine_principal_point: args.ba_refine_principal_point,
                ba_refine_extra_params: !args.no_ba_refine_extra_params,
                ba_constant_rig_ids: args.ba_constant_rig_ids,
                ba_constant_camera_ids: args.ba_constant_camera_ids,
                min_focal_length_ratio: args.min_focal_length_ratio,
                max_focal_length_ratio: args.max_focal_length_ratio,
                max_extra_param: args.max_extra_param,
                max_reprojection_error_px: args.max_reprojection_error_px,
                pose_graph: args.pose_graph,
                copy_images: !args.no_copy_images,
                threads: args.threads,
                fx: args.fx,
                fy: args.fy,
                cx: args.cx,
                cy: args.cy,
                ..Default::default()
            };
            let summary = run_reconstruction(&config)?;
            println!(
                "RustSFM completed: images={} registered={} points={} pairs={} elapsed_ms={:.2}",
                summary.images,
                summary.registered_images,
                summary.points,
                summary.pairs,
                summary.elapsed_ms
            );
            if let Some(path) = args.summary_json {
                std::fs::write(path, serde_json::to_string_pretty(&summary)?)?;
            }
        }
        Commands::Compare(args) => {
            env_logger::Builder::new()
                .parse_filters(&args.log_level)
                .init();
            let report = compare_colmap(&args.reference, &args.candidate)?;
            println!(
                "COLMAP pose comparison: common={} translation_rmse={:.6} rotation_rmse_deg={:.6} scale={:.6}",
                report.common_images,
                report.translation_error.rmse,
                report.rotation_error_deg.rmse,
                report.similarity_scale
            );
            if let Some(path) = args.output_json {
                std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
            }
        }
        Commands::Parity(args) => {
            env_logger::Builder::new()
                .parse_filters(&args.log_level)
                .init();
            let image_names = parity_image_names(args.images.as_ref(), &args.image_names)?;
            let report = compare_database_parity(
                &args.database,
                image_names,
                args.min_matches,
                args.ignore_watermarks,
                args.load_all_images,
                args.convert_pose_priors_to_enu,
            )?;
            println!(
                "COLMAP database parity: raw_images={} cache_images={} raw_frames={} cache_frames={} raw_frame_data={} cache_frame_data={} raw_pairs={} cache_pairs={} bridge_pairs={} bridge_matches={} differences={}",
                report.raw.images,
                report.cache.images,
                report.raw.frames,
                report.cache.frames,
                report.raw.frame_data,
                report.cache.frame_data,
                report.raw.two_view_pairs,
                report.cache.two_view_pairs,
                report.bridge.frame_pairs,
                report.bridge.matches,
                report.differences.len()
            );
            if !report.differences.is_empty() {
                for diff in &report.differences {
                    println!("difference {}: {}", diff.kind, diff.detail);
                }
            }
            if let Some(path) = args.output_json {
                std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
            }
        }
    }
    Ok(())
}

fn parity_image_names(images: Option<&PathBuf>, explicit_names: &[String]) -> Result<Vec<String>> {
    let mut names = explicit_names.to_vec();
    if let Some(root) = images {
        let mut from_dir = std::fs::read_dir(root)?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"))
                    .unwrap_or(false)
            })
            .filter_map(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.to_string())
            })
            .collect::<Vec<_>>();
        from_dir.sort();
        names.extend(from_dir);
    }
    names.sort();
    names.dedup();
    Ok(names)
}
