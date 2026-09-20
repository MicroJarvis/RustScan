use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::Parser;
use image::{ImageBuffer, RgbImage};
use rustgs::{
    evaluate_splats, evaluation_device, load_colmap_training_dataset, load_splats_ply,
    render_evaluation_frame, runtime_from_splats, select_evaluation_frames, ColmapConfig,
    EvaluationDevice, EvaluationFrameMetric, HostSplats, SplatEvaluationConfig,
    SplatEvaluationRenderer, SplatEvaluationSummary, TrainingDataset,
};
use serde::Serialize;

const MIN_RENDER_SCALE: f32 = 0.0625;

#[derive(Debug, Parser)]
#[command(name = "rustgs_eval_suite")]
#[command(about = "Run the RustGS TUM quality evaluation suite")]
struct Args {
    /// Path to a trained RustGS scene PLY.
    #[arg(long)]
    scene: PathBuf,

    /// Path to the source TUM/COLMAP dataset.
    #[arg(long)]
    dataset: PathBuf,

    /// Output directory for suite reports and optional crop strips.
    #[arg(long)]
    out: PathBuf,

    /// Evaluation render scale.
    #[arg(long, default_value = "0.25")]
    render_scale: f32,

    /// Evaluation rasterizer covariance blur floor.
    #[arg(long, default_value = "0.2")]
    raster_cov_blur: f32,

    /// Evaluation device.
    #[arg(long, default_value = "gpu")]
    device: String,

    /// Optional profile name stored in the generated reports.
    #[arg(long)]
    profile_hint: Option<String>,

    /// Optional regression gate profile.
    #[arg(long)]
    gate_profile: Option<GateProfile>,

    /// Exit with a non-zero status when the selected gate fails.
    #[arg(long, default_value_t = false)]
    fail_on_gate: bool,

    /// Export the K worst target/render/diff strips for each evaluation case.
    #[arg(long, default_value_t = 5)]
    export_worst_k: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GateProfile {
    Quality,
    Compact,
    Efficient,
}

impl GateProfile {
    fn as_str(self) -> &'static str {
        match self {
            Self::Quality => "tum-prefix-quality",
            Self::Compact => "tum-prefix-compact",
            Self::Efficient => "tum-prefix-efficient",
        }
    }
}

impl std::fmt::Display for GateProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for GateProfile {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "tum-prefix-quality" => Ok(Self::Quality),
            "tum-prefix-compact" => Ok(Self::Compact),
            "tum-prefix-efficient" => Ok(Self::Efficient),
            other => Err(format!(
                "unsupported gate profile '{other}'. Expected one of: tum-prefix-quality, tum-prefix-compact, tum-prefix-efficient"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct EvalCase {
    name: &'static str,
    title: &'static str,
    dataset_max_frames: usize,
    max_frames: usize,
    frame_stride: usize,
    include_frame_ranges: Option<&'static str>,
    exclude_frame_ranges: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct EvaluationCaseReport {
    name: String,
    title: String,
    max_frames: usize,
    frame_stride: usize,
    include_frame_ranges: Option<String>,
    exclude_frame_ranges: Option<String>,
    export_dir: Option<PathBuf>,
    summary: SplatEvaluationSummary,
}

#[derive(Debug, Serialize)]
struct SuiteReport {
    generated_unix_seconds: u64,
    scene: PathBuf,
    dataset: PathBuf,
    profile_hint: Option<String>,
    render_scale: f32,
    raster_cov_blur: f32,
    device: EvaluationDevice,
    cases: Vec<EvaluationCaseReport>,
    gate: Option<GateReport>,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum GateStatus {
    Passed,
    Failed,
}

#[derive(Debug, Serialize)]
struct GateCheck {
    name: String,
    case: String,
    metric: String,
    comparator: String,
    actual: f32,
    threshold: f32,
    status: GateStatus,
}

#[derive(Debug, Serialize)]
struct GateReport {
    profile: String,
    status: GateStatus,
    checks: Vec<GateCheck>,
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
    let eval_device = args
        .device
        .parse::<EvaluationDevice>()
        .map_err(anyhow::Error::msg)?;
    let device = evaluation_device(eval_device)?;
    let (splats, metadata) = load_splats_ply(&args.scene)?;

    let mut reports = Vec::new();
    for case in eval_cases() {
        println!("running case {}", case.name);
        let mut dataset = load_colmap_training_dataset(
            &args.dataset,
            &ColmapConfig {
                max_frames: case.dataset_max_frames,
                frame_stride: 1,
                ..ColmapConfig::default()
            },
        )?;
        let include_ranges = parse_frame_ranges(case.include_frame_ranges)?;
        dataset = filter_dataset_to_frame_ranges(dataset, &include_ranges, case.name)?;
        let exclude_ranges = parse_frame_ranges(case.exclude_frame_ranges)?;
        dataset = filter_dataset_by_frame_ranges(dataset, &exclude_ranges, case.name)?;

        let result = evaluate_splats(
            &dataset,
            &splats,
            &metadata,
            &SplatEvaluationConfig {
                render_scale: args.render_scale,
                raster_cov_blur: args.raster_cov_blur,
                max_frames: case.max_frames,
                frame_stride: case.frame_stride,
                worst_frame_count: args.export_worst_k.max(5),
            },
            &device,
            None,
        )?;
        let selected_dataset =
            select_evaluation_frames(&dataset, case.max_frames, case.frame_stride);
        let export_dir = if args.export_worst_k > 0 {
            let export_dir = args.out.join("crops").join(case.name);
            export_worst_frames(
                &selected_dataset,
                &splats,
                &result.frame_metrics,
                args.export_worst_k,
                &device,
                &export_dir,
                args.render_scale,
                args.raster_cov_blur,
            )?;
            Some(export_dir)
        } else {
            None
        };

        let mut summary = result.summary;
        if let Some(export_dir) = export_dir.as_ref() {
            summary.crop_outputs.push(export_dir.join("summary.tsv"));
        }
        reports.push(EvaluationCaseReport {
            name: case.name.to_string(),
            title: case.title.to_string(),
            max_frames: case.max_frames,
            frame_stride: case.frame_stride,
            include_frame_ranges: case.include_frame_ranges.map(str::to_string),
            exclude_frame_ranges: case.exclude_frame_ranges.map(str::to_string),
            export_dir,
            summary,
        });
    }

    let gate = args
        .gate_profile
        .map(|profile| evaluate_gate(profile, &reports));
    let report = SuiteReport {
        generated_unix_seconds: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        scene: args.scene.clone(),
        dataset: args.dataset.clone(),
        profile_hint: args.profile_hint.clone(),
        render_scale: args.render_scale,
        raster_cov_blur: args.raster_cov_blur,
        device: eval_device,
        cases: reports,
        gate,
    };

    let summary_json = args.out.join("summary.json");
    let summary_md = args.out.join("summary.md");
    std::fs::write(&summary_json, serde_json::to_vec_pretty(&report)?)?;
    std::fs::write(&summary_md, render_markdown(&report))?;
    println!("summary_json={}", summary_json.display());
    println!("summary_md={}", summary_md.display());

    if args.fail_on_gate
        && matches!(
            report.gate.as_ref().map(|gate| gate.status),
            Some(GateStatus::Failed)
        )
    {
        std::process::exit(2);
    }

    Ok(())
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    if !(MIN_RENDER_SCALE..=1.0).contains(&args.render_scale) {
        anyhow::bail!("--render-scale must be in [{}, 1.0]", MIN_RENDER_SCALE);
    }
    if !args.raster_cov_blur.is_finite() || args.raster_cov_blur < 0.0 {
        anyhow::bail!("--raster-cov-blur must be finite and >= 0");
    }
    args.device
        .parse::<EvaluationDevice>()
        .map_err(anyhow::Error::msg)?;
    Ok(())
}

fn eval_cases() -> [EvalCase; 4] {
    [
        EvalCase {
            name: "post_train_6_frame",
            title: "post-train 6-frame smoke",
            dataset_max_frames: 180,
            max_frames: 180,
            frame_stride: 30,
            include_frame_ranges: None,
            exclude_frame_ranges: None,
        },
        EvalCase {
            name: "full_180",
            title: "full 180-frame prefix",
            dataset_max_frames: 180,
            max_frames: 180,
            frame_stride: 1,
            include_frame_ranges: None,
            exclude_frame_ranges: None,
        },
        EvalCase {
            name: "static_162",
            title: "static 162-frame prefix",
            dataset_max_frames: 180,
            max_frames: 180,
            frame_stride: 1,
            include_frame_ranges: None,
            exclude_frame_ranges: Some("76-93"),
        },
        EvalCase {
            name: "full_trajectory_stride_4",
            title: "full trajectory stride-4",
            dataset_max_frames: 0,
            max_frames: 0,
            frame_stride: 4,
            include_frame_ranges: None,
            exclude_frame_ranges: None,
        },
    ]
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
    Ok(ranges)
}

fn filter_dataset_by_frame_ranges(
    dataset: TrainingDataset,
    excluded_ranges: &[FrameIdRange],
    label: &str,
) -> anyhow::Result<TrainingDataset> {
    if excluded_ranges.is_empty() {
        return Ok(dataset);
    }

    let mut filtered =
        TrainingDataset::new(dataset.intrinsics).with_depth_scale(dataset.depth_scale);
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
    if filtered.poses.is_empty() {
        anyhow::bail!("{label} excluded all frames");
    }
    Ok(filtered)
}

fn filter_dataset_to_frame_ranges(
    dataset: TrainingDataset,
    included_ranges: &[FrameIdRange],
    label: &str,
) -> anyhow::Result<TrainingDataset> {
    if included_ranges.is_empty() {
        return Ok(dataset);
    }

    let mut filtered =
        TrainingDataset::new(dataset.intrinsics).with_depth_scale(dataset.depth_scale);
    filtered.initial_points = dataset.initial_points.clone();
    for pose in dataset.poses {
        if included_ranges
            .iter()
            .any(|range| range.contains(pose.frame_id))
        {
            filtered.add_pose(pose);
        }
    }
    if filtered.poses.is_empty() {
        anyhow::bail!("{label} selected no frames");
    }
    Ok(filtered)
}

fn evaluate_gate(profile: GateProfile, reports: &[EvaluationCaseReport]) -> GateReport {
    let mut checks = Vec::new();
    let (full_threshold, static_threshold, splat_limit) = match profile {
        GateProfile::Quality => (23.05, 23.65, None),
        GateProfile::Compact => (22.95, 23.58, Some(55_000.0)),
        GateProfile::Efficient => (22.90, 23.50, Some(43_000.0)),
    };

    push_min_check(
        &mut checks,
        reports,
        "full_180",
        "psnr_mean_db",
        full_threshold,
        |summary| summary.psnr_mean_db,
    );
    push_min_check(
        &mut checks,
        reports,
        "static_162",
        "psnr_mean_db",
        static_threshold,
        |summary| summary.psnr_mean_db,
    );
    if let Some(limit) = splat_limit {
        push_max_check(
            &mut checks,
            reports,
            "full_180",
            "splat_count",
            limit,
            |summary| summary.splat_count as f32,
        );
    }

    let status = if checks
        .iter()
        .any(|check| check.status == GateStatus::Failed)
    {
        GateStatus::Failed
    } else {
        GateStatus::Passed
    };
    GateReport {
        profile: profile.to_string(),
        status,
        checks,
    }
}

fn push_min_check(
    checks: &mut Vec<GateCheck>,
    reports: &[EvaluationCaseReport],
    case_name: &str,
    metric: &str,
    threshold: f32,
    value: impl FnOnce(&SplatEvaluationSummary) -> f32,
) {
    let Some(report) = reports.iter().find(|report| report.name == case_name) else {
        checks.push(missing_check(case_name, metric, ">=", threshold));
        return;
    };
    let actual = value(&report.summary);
    checks.push(GateCheck {
        name: format!("{case_name}.{metric}"),
        case: case_name.to_string(),
        metric: metric.to_string(),
        comparator: ">=".to_string(),
        actual,
        threshold,
        status: if actual >= threshold {
            GateStatus::Passed
        } else {
            GateStatus::Failed
        },
    });
}

fn push_max_check(
    checks: &mut Vec<GateCheck>,
    reports: &[EvaluationCaseReport],
    case_name: &str,
    metric: &str,
    threshold: f32,
    value: impl FnOnce(&SplatEvaluationSummary) -> f32,
) {
    let Some(report) = reports.iter().find(|report| report.name == case_name) else {
        checks.push(missing_check(case_name, metric, "<=", threshold));
        return;
    };
    let actual = value(&report.summary);
    checks.push(GateCheck {
        name: format!("{case_name}.{metric}"),
        case: case_name.to_string(),
        metric: metric.to_string(),
        comparator: "<=".to_string(),
        actual,
        threshold,
        status: if actual <= threshold {
            GateStatus::Passed
        } else {
            GateStatus::Failed
        },
    });
}

fn missing_check(case_name: &str, metric: &str, comparator: &str, threshold: f32) -> GateCheck {
    GateCheck {
        name: format!("{case_name}.{metric}"),
        case: case_name.to_string(),
        metric: metric.to_string(),
        comparator: comparator.to_string(),
        actual: 0.0,
        threshold,
        status: GateStatus::Failed,
    }
}

fn render_markdown(report: &SuiteReport) -> String {
    let mut out = String::new();
    out.push_str("# RustGS evaluation suite\n\n");
    out.push_str(&format!("scene: `{}`\n\n", report.scene.display()));
    out.push_str(&format!("dataset: `{}`\n\n", report.dataset.display()));
    if let Some(profile_hint) = report.profile_hint.as_ref() {
        out.push_str(&format!("profile: `{profile_hint}`\n\n"));
    }
    out.push_str("| Case | Frames | PSNR mean | PSNR min | Grad ratio | Lap ratio | Splats | Worst frame |\n");
    out.push_str("|---|---:|---:|---:|---:|---:|---:|---|\n");
    for case in &report.cases {
        let summary = &case.summary;
        let worst = summary
            .worst_frames
            .first()
            .map(|frame| format!("{} ({:.4} dB)", frame.frame_id, frame.psnr_db))
            .unwrap_or_else(|| "-".to_string());
        out.push_str(&format!(
            "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {} | {} |\n",
            case.name,
            summary.frame_count,
            summary.psnr_mean_db,
            summary.psnr_min_db,
            summary.sharpness_grad_ratio_mean,
            summary.sharpness_lap_ratio_mean,
            summary.splat_count,
            worst
        ));
    }
    if let Some(gate) = report.gate.as_ref() {
        out.push_str("\n## Gate\n\n");
        out.push_str(&format!("profile: `{}`\n\n", gate.profile));
        out.push_str(&format!("status: `{:?}`\n\n", gate.status));
        out.push_str("| Check | Actual | Rule | Status |\n");
        out.push_str("|---|---:|---|---|\n");
        for check in &gate.checks {
            out.push_str(&format!(
                "| {} | {:.4} | {} {:.4} | {:?} |\n",
                check.name, check.actual, check.comparator, check.threshold, check.status
            ));
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn export_worst_frames(
    dataset: &TrainingDataset,
    splats: &HostSplats,
    frame_metrics: &[EvaluationFrameMetric],
    export_worst_k: usize,
    device: &EvaluationDevice,
    export_dir: &Path,
    render_scale: f32,
    raster_cov_blur: f32,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(export_dir)?;

    let (render_width, render_height) = rustgs::scaled_dimensions(
        dataset.intrinsics.width as usize,
        dataset.intrinsics.height as usize,
        render_scale,
    );
    let trainable = runtime_from_splats(splats, device)?;
    let mut renderer =
        SplatEvaluationRenderer::new(render_width, render_height, *device, raster_cov_blur)?;
    let worst = rustgs::worst_frame_metrics(frame_metrics, export_worst_k);

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
