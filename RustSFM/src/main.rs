use anyhow::Result;
use clap::{Parser, Subcommand};
use rustsfm::{compare_colmap, run_reconstruction, FeatureType, MapperConfig};
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
    max_images: Option<usize>,
    #[arg(long, default_value = "sift")]
    feature_type: FeatureType,
    #[arg(long, default_value = "8192")]
    max_features: usize,
    #[arg(long, default_value = "3")]
    local_window: usize,
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
    #[arg(long, default_value = "2.0")]
    essential_threshold_px: f32,
    #[arg(long, default_value = "10000")]
    essential_iterations: u32,
    #[arg(long, default_value = "8.0")]
    pnp_threshold_px: f32,
    #[arg(long, default_value = "2000")]
    pnp_iterations: u32,
    #[arg(long, default_value = "8.0")]
    max_reprojection_error_px: f32,
    #[arg(long, default_value_t = false)]
    no_pose_graph: bool,
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
                max_images: args.max_images,
                feature_type: args.feature_type,
                max_features: args.max_features,
                match_ratio: args.match_ratio,
                max_hamming_distance: args.max_hamming_distance,
                local_window: args.local_window,
                min_matches: args.min_matches,
                min_inliers: args.min_inliers,
                min_triangulated: args.min_triangulated,
                essential_threshold_px: args.essential_threshold_px,
                essential_iterations: args.essential_iterations,
                pnp_threshold_px: args.pnp_threshold_px,
                pnp_iterations: args.pnp_iterations,
                max_reprojection_error_px: args.max_reprojection_error_px,
                pose_graph: !args.no_pose_graph,
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
    }
    Ok(())
}
