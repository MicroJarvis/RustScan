use std::path::PathBuf;

use image::{ImageBuffer, RgbImage};
use rustscan_gs::{
    evaluate_splats, evaluation_device, load_colmap_training_dataset, load_splats_ply,
    parse_frame_id_ranges, render_evaluation_frame, runtime_from_splats, ColmapConfig,
    EvaluationDevice, EvaluationFrameMetric, FrameSelection, FrameSelectionReport,
    FrameSelectionRequest, HostSplats, SplatEvaluationConfig, SplatEvaluationRenderer,
    SplatEvaluationSummary,
};

fn main() -> anyhow::Result<()> {
    let args = Args::parse()?;
    if args.frame_stride == 0 {
        anyhow::bail!("--frame-stride must be >= 1");
    }

    // Full load, then one canonical FrameSelection (include/exclude → max → stride).
    let full_dataset = load_colmap_training_dataset(
        &args.dataset,
        &ColmapConfig {
            max_frames: 0,
            frame_stride: 1,
            ..ColmapConfig::default()
        },
    )?;
    let selection = FrameSelection::select(
        &full_dataset,
        &FrameSelectionRequest {
            include_ranges: parse_frame_id_ranges(args.include_frame_ranges.as_deref())
                .map_err(anyhow::Error::msg)?,
            exclude_ranges: parse_frame_id_ranges(args.exclude_frame_ranges.as_deref())
                .map_err(anyhow::Error::msg)?,
            max_frames: args.max_frames,
            frame_stride: args.frame_stride,
            ..Default::default()
        },
    )
    .map_err(anyhow::Error::msg)?;
    if selection.dataset.poses.is_empty() {
        anyhow::bail!("frame selection resolved to zero frames");
    }
    let dataset = &selection.dataset;
    let (splats, metadata) = load_splats_ply(&args.scene)?;
    let eval_device = args
        .device
        .parse::<EvaluationDevice>()
        .map_err(anyhow::Error::msg)?;
    let device = evaluation_device(eval_device)?;
    let result = evaluate_splats(
        dataset,
        &splats,
        &metadata,
        &SplatEvaluationConfig {
            render_scale: args.render_scale,
            raster_cov_blur: args.raster_cov_blur,
            // Already selected once; do not re-filter.
            frame_stride: 1,
            max_frames: 0,
            worst_frame_count: summary_worst_count(args.export_worst_k),
        },
        &device,
        None,
    )?;

    if args.json {
        #[derive(serde::Serialize)]
        struct EvaluatePsnrJsonReport<'a> {
            summary: &'a SplatEvaluationSummary,
            selection: FrameSelectionReport,
        }
        let mut selection_report = FrameSelectionReport::from_in_view_selection(&selection, None);
        if args.include_frame_ranges.is_some() {
            selection_report.include_frame_ranges = args.include_frame_ranges.clone();
        }
        if args.exclude_frame_ranges.is_some() {
            selection_report.exclude_frame_ranges = args.exclude_frame_ranges.clone();
        }
        let report = EvaluatePsnrJsonReport {
            summary: &result.summary,
            selection: selection_report,
        };
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_human_summary(&args, &result.summary, &selection);
    }

    if args.export_worst_k > 0 {
        let export_dir = args
            .export_dir
            .clone()
            .unwrap_or_else(|| default_export_dir(&args.scene));
        export_worst_frames(
            dataset,
            &splats,
            &result.frame_metrics,
            args.export_worst_k,
            &device,
            &export_dir,
            args.render_scale,
            args.raster_cov_blur,
        )?;
        println!("worst_frames_export_dir={}", export_dir.display());
    }

    Ok(())
}

struct Args {
    scene: PathBuf,
    dataset: PathBuf,
    render_scale: f32,
    raster_cov_blur: f32,
    frame_stride: usize,
    max_frames: usize,
    device: String,
    export_worst_k: usize,
    export_dir: Option<PathBuf>,
    include_frame_ranges: Option<String>,
    exclude_frame_ranges: Option<String>,
    json: bool,
}

impl Args {
    fn parse() -> anyhow::Result<Self> {
        let mut scene = None;
        let mut dataset = None;
        let mut render_scale = 0.5f32;
        let mut raster_cov_blur = rustscan_gs::DEFAULT_RASTER_COV_BLUR;
        let mut frame_stride = 1usize;
        let mut max_frames = 0usize;
        let mut device = String::from("gpu");
        let mut export_worst_k = 0usize;
        let mut export_dir = None;
        let mut include_frame_ranges = None;
        let mut exclude_frame_ranges = None;
        let mut json = false;

        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--scene" => scene = Some(PathBuf::from(next_value(&mut it, "--scene")?)),
                "--dataset" => dataset = Some(PathBuf::from(next_value(&mut it, "--dataset")?)),
                "--render-scale" => {
                    render_scale = next_value(&mut it, "--render-scale")?.parse()?
                }
                "--raster-cov-blur" => {
                    raster_cov_blur = next_value(&mut it, "--raster-cov-blur")?.parse()?
                }
                "--frame-stride" => {
                    frame_stride = next_value(&mut it, "--frame-stride")?.parse()?
                }
                "--max-frames" => max_frames = next_value(&mut it, "--max-frames")?.parse()?,
                "--device" => device = next_value(&mut it, "--device")?,
                "--export-worst-k" => {
                    export_worst_k = next_value(&mut it, "--export-worst-k")?.parse()?
                }
                "--export-dir" => {
                    export_dir = Some(PathBuf::from(next_value(&mut it, "--export-dir")?))
                }
                "--include-frame-ranges" => {
                    include_frame_ranges = Some(next_value(&mut it, "--include-frame-ranges")?)
                }
                "--exclude-frame-ranges" => {
                    exclude_frame_ranges = Some(next_value(&mut it, "--exclude-frame-ranges")?)
                }
                "--json" => json = true,
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => {
                    return Err(anyhow::anyhow!("unrecognized argument: {other}"));
                }
            }
        }

        let scene = scene.ok_or_else(|| anyhow::anyhow!("missing --scene"))?;
        let dataset = dataset.ok_or_else(|| anyhow::anyhow!("missing --dataset"))?;
        if frame_stride == 0 {
            return Err(anyhow::anyhow!("--frame-stride must be >= 1"));
        }
        if !(0.0625..=1.0).contains(&render_scale) {
            return Err(anyhow::anyhow!("--render-scale must be in [0.0625, 1.0]"));
        }
        if !raster_cov_blur.is_finite() || raster_cov_blur < 0.0 {
            return Err(anyhow::anyhow!("--raster-cov-blur must be finite and >= 0"));
        }
        device
            .parse::<EvaluationDevice>()
            .map_err(anyhow::Error::msg)?;

        Ok(Self {
            scene,
            dataset,
            render_scale,
            raster_cov_blur,
            frame_stride,
            max_frames,
            device,
            export_worst_k,
            export_dir,
            include_frame_ranges,
            exclude_frame_ranges,
            json,
        })
    }
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> anyhow::Result<String> {
    args.next()
        .ok_or_else(|| anyhow::anyhow!("missing value for {flag}"))
}

fn print_help() {
    println!(
        "Usage: cargo run --manifest-path rustscan-gs/Cargo.toml --example evaluate_psnr -- \
  --scene <scene.ply> \
  --dataset <training_dataset.json> \
  [--render-scale 0.5] \
  [--raster-cov-blur 0.3] \
  [--frame-stride 1] \
  [--max-frames 0] \
  [--device gpu] \
  [--json] \
  [--export-worst-k 5] \
  [--export-dir artifacts/runs/psnr_review] \
  [--include-frame-ranges 0-179,240-299] \
  [--exclude-frame-ranges 76-93,155]"
    );
}

fn summary_worst_count(export_worst_k: usize) -> usize {
    if export_worst_k == 0 {
        5
    } else {
        export_worst_k
    }
}

fn print_human_summary(args: &Args, summary: &SplatEvaluationSummary, selection: &FrameSelection) {
    println!("scene={}", args.scene.display());
    println!("dataset={}", args.dataset.display());
    println!("device={}", summary.device);
    println!(
        "render_scale={:.3} raster_cov_blur={:.3} resolution={}x{} frames={} stride={} max_frames={} selection_fp={}",
        summary.render_scale,
        summary.raster_cov_blur,
        summary.render_width,
        summary.render_height,
        selection.stable_ids.len(),
        selection.frame_stride,
        selection.max_frames,
        selection.selection_fingerprint,
    );
    println!("stable_frame_ids={:?}", selection.stable_ids);
    println!(
        "splat_metadata iterations={} splat_count={} final_loss={} final_step_loss={}",
        summary.splat_iterations,
        summary.splat_count,
        summary.final_loss,
        summary
            .final_step_loss
            .map(|value| format!("{value}"))
            .unwrap_or_else(|| "null".to_string())
    );
    println!(
        "psnr_mean_db={:.4}\npsnr_median_db={:.4}\npsnr_min_db={:.4}\npsnr_max_db={:.4}\npsnr_std_db={:.4}\nelapsed_seconds={:.2}",
        summary.psnr_mean_db,
        summary.psnr_median_db,
        summary.psnr_min_db,
        summary.psnr_max_db,
        summary.psnr_std_db,
        summary.elapsed_seconds,
    );

    for (rank, metric) in summary.worst_frames.iter().enumerate() {
        println!(
            "worst_frame rank={} dataset_index={} frame_id={} psnr_db={:.4} image={}",
            rank + 1,
            metric.dataset_index,
            metric.frame_id,
            metric.psnr_db,
            metric.image_path.display()
        );
    }
}

fn default_export_dir(scene: &std::path::Path) -> PathBuf {
    let parent = scene
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = scene
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("scene");
    parent.join(format!("{stem}_psnr_review"))
}

#[allow(clippy::too_many_arguments)]
fn export_worst_frames(
    dataset: &rustscan_gs::TrainingDataset,
    splats: &HostSplats,
    frame_metrics: &[EvaluationFrameMetric],
    export_worst_k: usize,
    device: &EvaluationDevice,
    export_dir: &std::path::Path,
    render_scale: f32,
    raster_cov_blur: f32,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(export_dir)?;

    let (render_width, render_height) = rustscan_gs::scaled_dimensions(
        dataset.intrinsics.width as usize,
        dataset.intrinsics.height as usize,
        render_scale,
    );
    let trainable = runtime_from_splats(splats, device)?;
    let mut renderer =
        SplatEvaluationRenderer::new(render_width, render_height, *device, raster_cov_blur)?;
    let worst = rustscan_gs::worst_frame_metrics(frame_metrics, export_worst_k);

    let mut summary = String::from("rank\tdataset_index\tframe_id\tpsnr_db\timage_path\n");
    for (rank, metric) in worst.into_iter().enumerate() {
        let pose = &dataset.poses[metric.dataset_index];
        let (target, rendered) = render_evaluation_frame(
            dataset,
            pose,
            render_width,
            render_height,
            device,
            &trainable,
            &mut renderer,
        )?;
        let target_u8 = rgb_f32_to_u8(&target);
        let rendered_u8 = rgb_f32_to_u8(&rendered);
        let diff_u8 = diff_visualization(&target, &rendered);
        let strip_u8 = make_strip(
            &target_u8,
            &rendered_u8,
            &diff_u8,
            render_width,
            render_height,
        );

        let prefix = format!(
            "rank_{:02}_frame_{:04}_psnr_{:.2}",
            rank + 1,
            metric.frame_id,
            metric.psnr_db
        );
        save_rgb_png(
            export_dir.join(format!("{prefix}_gt.png")),
            render_width,
            render_height,
            &target_u8,
        )?;
        save_rgb_png(
            export_dir.join(format!("{prefix}_render.png")),
            render_width,
            render_height,
            &rendered_u8,
        )?;
        save_rgb_png(
            export_dir.join(format!("{prefix}_diff.png")),
            render_width,
            render_height,
            &diff_u8,
        )?;
        save_rgb_png(
            export_dir.join(format!("{prefix}_strip.png")),
            render_width * 3,
            render_height,
            &strip_u8,
        )?;
        summary.push_str(&format!(
            "{}\t{}\t{}\t{:.4}\t{}\n",
            rank + 1,
            metric.dataset_index,
            metric.frame_id,
            metric.psnr_db,
            metric.image_path.display()
        ));
    }

    std::fs::write(export_dir.join("summary.tsv"), summary)?;
    Ok(())
}

fn rgb_f32_to_u8(data: &[f32]) -> Vec<u8> {
    data.iter()
        .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect()
}

fn diff_visualization(target: &[f32], rendered: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(target.len());
    for (target_px, rendered_px) in target.chunks_exact(3).zip(rendered.chunks_exact(3)) {
        let abs_mean = ((target_px[0] - rendered_px[0]).abs()
            + (target_px[1] - rendered_px[1]).abs()
            + (target_px[2] - rendered_px[2]).abs())
            / 3.0;
        let heat = (abs_mean * 4.0).clamp(0.0, 1.0);
        out.push((heat * 255.0).round() as u8);
        out.push((heat * 64.0).round() as u8);
        out.push(0);
    }
    out
}

fn make_strip(left: &[u8], middle: &[u8], right: &[u8], width: usize, height: usize) -> Vec<u8> {
    let row_bytes = width * 3;
    let mut out = vec![0u8; width * 3 * height * 3];
    for y in 0..height {
        let src_start = y * row_bytes;
        let dst_start = y * row_bytes * 3;
        out[dst_start..dst_start + row_bytes]
            .copy_from_slice(&left[src_start..src_start + row_bytes]);
        out[dst_start + row_bytes..dst_start + row_bytes * 2]
            .copy_from_slice(&middle[src_start..src_start + row_bytes]);
        out[dst_start + row_bytes * 2..dst_start + row_bytes * 3]
            .copy_from_slice(&right[src_start..src_start + row_bytes]);
    }
    out
}

fn save_rgb_png(path: PathBuf, width: usize, height: usize, data: &[u8]) -> anyhow::Result<()> {
    let image: RgbImage = ImageBuffer::from_raw(width as u32, height as u32, data.to_vec())
        .ok_or_else(|| {
            anyhow::anyhow!("failed to construct image buffer for {}", path.display())
        })?;
    image.save(&path)?;
    Ok(())
}
