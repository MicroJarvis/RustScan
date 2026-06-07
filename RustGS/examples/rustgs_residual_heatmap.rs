use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use clap::Parser;
use image::{ImageBuffer, RgbImage};
use rustgs::{
    evaluation_device, load_colmap_training_dataset, load_splats_ply, render_evaluation_frame,
    runtime_from_splats, ColmapConfig, EvaluationDevice, SplatEvaluationRenderer, TrainingDataset,
};
use serde::Serialize;

const MIN_RENDER_SCALE: f32 = 0.0625;

#[derive(Debug, Parser)]
#[command(name = "rustgs_residual_heatmap")]
#[command(about = "Export residual heatmaps for RustGS dynamic/occlusion diagnostics")]
struct Args {
    /// Path to a trained RustGS scene PLY.
    #[arg(long)]
    scene: PathBuf,

    /// Path to the source COLMAP dataset.
    #[arg(long)]
    dataset: PathBuf,

    /// Output directory for heatmaps and reports.
    #[arg(long)]
    out: PathBuf,

    /// Comma-separated frame ids or inclusive ranges to diagnose.
    #[arg(long, default_value = "0,30,60,76-93,120,150")]
    frames: String,

    /// Evaluation render scale.
    #[arg(long, default_value = "0.25")]
    render_scale: f32,

    /// Evaluation rasterizer covariance blur floor.
    #[arg(long, default_value = "0.2")]
    raster_cov_blur: f32,

    /// Mean absolute RGB residual threshold used for connected-component stats.
    #[arg(long, default_value = "0.12")]
    residual_threshold: f32,

    /// Evaluation device. Currently cpu-compatible rendering is used.
    #[arg(long, default_value = "cpu")]
    device: String,
}

#[derive(Debug, Serialize)]
struct ResidualHeatmapReport {
    scene: PathBuf,
    dataset: PathBuf,
    frames: String,
    render_scale: f32,
    raster_cov_blur: f32,
    residual_threshold: f32,
    render_width: usize,
    render_height: usize,
    reports: Vec<FrameResidualReport>,
}

#[derive(Debug, Serialize)]
struct FrameResidualReport {
    dataset_index: usize,
    frame_id: u64,
    image_path: PathBuf,
    mean_abs: f32,
    p95_abs: f32,
    p99_abs: f32,
    max_abs: f32,
    high_residual_ratio: f32,
    connected_components: usize,
    max_component_pixels: usize,
    heatmap_path: PathBuf,
    strip_path: PathBuf,
}

#[derive(Debug, Clone, Copy)]
struct FrameIdRange {
    start: u64,
    end: u64,
}

impl FrameIdRange {
    fn contains(&self, frame_id: u64) -> bool {
        self.start <= frame_id && frame_id <= self.end
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    std::fs::create_dir_all(&args.out)?;

    let frame_ranges = parse_frame_ranges(&args.frames)?;
    let eval_device = args
        .device
        .parse::<EvaluationDevice>()
        .map_err(anyhow::Error::msg)?;
    let device = evaluation_device(eval_device)?;
    let mut dataset = load_colmap_training_dataset(
        &args.dataset,
        &ColmapConfig {
            max_frames: 0,
            frame_stride: 1,
            ..ColmapConfig::default()
        },
    )?;
    dataset = filter_dataset_to_frame_ranges(dataset, &frame_ranges)?;
    let (splats, _) = load_splats_ply(&args.scene)?;

    let (render_width, render_height) = rustgs::scaled_dimensions(
        dataset.intrinsics.width as usize,
        dataset.intrinsics.height as usize,
        args.render_scale,
    );
    let runtime_splats = runtime_from_splats(&splats, &device)?;
    let mut renderer =
        SplatEvaluationRenderer::new(render_width, render_height, device, args.raster_cov_blur)?;

    let mut reports = Vec::new();
    for (dataset_index, pose) in dataset.poses.iter().enumerate() {
        println!("diagnosing frame {}", pose.frame_id);
        let (target, rendered) = render_evaluation_frame(
            &dataset,
            pose,
            render_width,
            render_height,
            &device,
            &runtime_splats,
            &mut renderer,
        )?;
        let residual = mean_abs_residual(&target, &rendered);
        let stats = residual_stats(
            &residual,
            render_width,
            render_height,
            args.residual_threshold,
        );
        let heatmap = residual_heatmap(&residual, args.residual_threshold);
        let target_u8 = rgb_f32_to_u8(&target);
        let rendered_u8 = rgb_f32_to_u8(&rendered);
        let strip = make_strip(
            &target_u8,
            &rendered_u8,
            &heatmap,
            render_width,
            render_height,
        );

        let prefix = format!("frame_{:06}", pose.frame_id);
        let heatmap_path = args.out.join(format!("{prefix}_residual_heat.png"));
        let strip_path = args.out.join(format!("{prefix}_strip.png"));
        save_rgb_png(&heatmap_path, render_width, render_height, &heatmap)?;
        save_rgb_png(&strip_path, render_width * 3, render_height, &strip)?;
        reports.push(FrameResidualReport {
            dataset_index,
            frame_id: pose.frame_id,
            image_path: pose.image_path.clone(),
            mean_abs: stats.mean_abs,
            p95_abs: stats.p95_abs,
            p99_abs: stats.p99_abs,
            max_abs: stats.max_abs,
            high_residual_ratio: stats.high_residual_ratio,
            connected_components: stats.connected_components,
            max_component_pixels: stats.max_component_pixels,
            heatmap_path,
            strip_path,
        });
    }

    let report = ResidualHeatmapReport {
        scene: args.scene.clone(),
        dataset: args.dataset.clone(),
        frames: args.frames.clone(),
        render_scale: args.render_scale,
        raster_cov_blur: args.raster_cov_blur,
        residual_threshold: args.residual_threshold,
        render_width,
        render_height,
        reports,
    };
    let summary_json = args.out.join("summary.json");
    let summary_md = args.out.join("summary.md");
    std::fs::write(&summary_json, serde_json::to_vec_pretty(&report)?)?;
    std::fs::write(&summary_md, render_markdown(&report))?;
    println!("summary_json={}", summary_json.display());
    println!("summary_md={}", summary_md.display());

    Ok(())
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    if !(MIN_RENDER_SCALE..=1.0).contains(&args.render_scale) {
        anyhow::bail!("--render-scale must be in [{}, 1.0]", MIN_RENDER_SCALE);
    }
    if !args.raster_cov_blur.is_finite() || args.raster_cov_blur < 0.0 {
        anyhow::bail!("--raster-cov-blur must be finite and >= 0");
    }
    if !args.residual_threshold.is_finite() || args.residual_threshold <= 0.0 {
        anyhow::bail!("--residual-threshold must be finite and > 0");
    }
    args.device
        .parse::<EvaluationDevice>()
        .map_err(anyhow::Error::msg)?;
    Ok(())
}

fn parse_frame_ranges(value: &str) -> anyhow::Result<Vec<FrameIdRange>> {
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
            anyhow::bail!("frame range '{token}' must be <frame_id> or <start>-<end>");
        }
        let start = start
            .parse::<u64>()
            .map_err(|_| anyhow::anyhow!("invalid frame range start in '{token}'"))?;
        let end = end
            .parse::<u64>()
            .map_err(|_| anyhow::anyhow!("invalid frame range end in '{token}'"))?;
        if start > end {
            anyhow::bail!("frame range '{token}' has start greater than end");
        }
        ranges.push(FrameIdRange { start, end });
    }
    if ranges.is_empty() {
        anyhow::bail!("--frames must select at least one frame");
    }
    Ok(ranges)
}

fn filter_dataset_to_frame_ranges(
    dataset: TrainingDataset,
    included_ranges: &[FrameIdRange],
) -> anyhow::Result<TrainingDataset> {
    let selected_ids = dataset
        .poses
        .iter()
        .filter(|pose| {
            included_ranges
                .iter()
                .any(|range| range.contains(pose.frame_id))
        })
        .map(|pose| pose.frame_id)
        .collect::<BTreeSet<_>>();
    if selected_ids.is_empty() {
        anyhow::bail!("--frames selected no dataset frames");
    }

    let mut filtered =
        TrainingDataset::new(dataset.intrinsics).with_depth_scale(dataset.depth_scale);
    filtered.initial_points = dataset.initial_points.clone();
    for pose in dataset.poses {
        if selected_ids.contains(&pose.frame_id) {
            filtered.add_pose(pose);
        }
    }
    Ok(filtered)
}

fn mean_abs_residual(target: &[f32], rendered: &[f32]) -> Vec<f32> {
    target
        .chunks_exact(3)
        .zip(rendered.chunks_exact(3))
        .map(|(target_px, rendered_px)| {
            ((target_px[0] - rendered_px[0]).abs()
                + (target_px[1] - rendered_px[1]).abs()
                + (target_px[2] - rendered_px[2]).abs())
                / 3.0
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct ResidualStats {
    mean_abs: f32,
    p95_abs: f32,
    p99_abs: f32,
    max_abs: f32,
    high_residual_ratio: f32,
    connected_components: usize,
    max_component_pixels: usize,
}

fn residual_stats(residual: &[f32], width: usize, height: usize, threshold: f32) -> ResidualStats {
    let mut sorted = residual.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean_abs = residual.iter().sum::<f32>() / residual.len().max(1) as f32;
    let max_abs = sorted.last().copied().unwrap_or(0.0);
    let p95_abs = percentile(&sorted, 0.95);
    let p99_abs = percentile(&sorted, 0.99);
    let high_pixels = residual.iter().filter(|value| **value >= threshold).count();
    let (connected_components, max_component_pixels) =
        connected_components_above_threshold(residual, width, height, threshold);
    ResidualStats {
        mean_abs,
        p95_abs,
        p99_abs,
        max_abs,
        high_residual_ratio: high_pixels as f32 / residual.len().max(1) as f32,
        connected_components,
        max_component_pixels,
    }
}

fn percentile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f32 * q.clamp(0.0, 1.0)).round() as usize;
    sorted[idx]
}

fn connected_components_above_threshold(
    residual: &[f32],
    width: usize,
    height: usize,
    threshold: f32,
) -> (usize, usize) {
    let mut visited = vec![false; residual.len()];
    let mut component_count = 0usize;
    let mut max_component_pixels = 0usize;
    let mut stack = Vec::new();

    for idx in 0..residual.len() {
        if visited[idx] || residual[idx] < threshold {
            continue;
        }
        component_count += 1;
        visited[idx] = true;
        stack.push(idx);
        let mut pixels = 0usize;
        while let Some(current) = stack.pop() {
            pixels += 1;
            let x = current % width;
            let y = current / width;
            for next in neighbors4(x, y, width, height) {
                if !visited[next] && residual[next] >= threshold {
                    visited[next] = true;
                    stack.push(next);
                }
            }
        }
        max_component_pixels = max_component_pixels.max(pixels);
    }

    (component_count, max_component_pixels)
}

fn neighbors4(x: usize, y: usize, width: usize, height: usize) -> [usize; 4] {
    [
        if x > 0 {
            y * width + (x - 1)
        } else {
            y * width + x
        },
        if x + 1 < width {
            y * width + (x + 1)
        } else {
            y * width + x
        },
        if y > 0 {
            (y - 1) * width + x
        } else {
            y * width + x
        },
        if y + 1 < height {
            (y + 1) * width + x
        } else {
            y * width + x
        },
    ]
}

fn residual_heatmap(residual: &[f32], threshold: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(residual.len() * 3);
    for value in residual {
        let heat = (value / threshold).clamp(0.0, 1.0);
        out.push((heat * 255.0).round() as u8);
        out.push((heat * 64.0).round() as u8);
        out.push(0);
    }
    out
}

fn render_markdown(report: &ResidualHeatmapReport) -> String {
    let mut out = String::new();
    out.push_str("# RustGS residual heatmap diagnostics\n\n");
    out.push_str(&format!("scene: `{}`\n\n", report.scene.display()));
    out.push_str(&format!("dataset: `{}`\n\n", report.dataset.display()));
    out.push_str(&format!(
        "frames: `{}`; threshold: `{:.4}`\n\n",
        report.frames, report.residual_threshold
    ));
    out.push_str(
        "| Frame | Mean | P95 | P99 | Max | High ratio | Components | Max component | Strip |\n",
    );
    out.push_str("|---:|---:|---:|---:|---:|---:|---:|---:|---|\n");
    for frame in &report.reports {
        out.push_str(&format!(
            "| {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {} | {} | `{}` |\n",
            frame.frame_id,
            frame.mean_abs,
            frame.p95_abs,
            frame.p99_abs,
            frame.max_abs,
            frame.high_residual_ratio,
            frame.connected_components,
            frame.max_component_pixels,
            frame.strip_path.display()
        ));
    }
    out
}

fn rgb_f32_to_u8(data: &[f32]) -> Vec<u8> {
    data.iter()
        .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect()
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

fn save_rgb_png(path: &Path, width: usize, height: usize, data: &[u8]) -> anyhow::Result<()> {
    let image: RgbImage = ImageBuffer::from_raw(width as u32, height as u32, data.to_vec())
        .ok_or_else(|| {
            anyhow::anyhow!("failed to construct image buffer for {}", path.display())
        })?;
    image.save(path)?;
    Ok(())
}
