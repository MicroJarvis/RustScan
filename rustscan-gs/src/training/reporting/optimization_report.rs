//! Comparable optimization experiment reports for RustGS training runs.
//!
//! CPU loop timings use `Instant`. GPU completion time and device VRAM are left
//! null unless a synchronized profiling boundary exists — never label unsynced
//! Instant intervals as GPU time.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct OptimizationReport {
    pub environment: OptimizationEnvironment,
    pub command: OptimizationCommand,
    pub train: OptimizationTrainMetrics,
    pub topology: OptimizationTopologyMetrics,
    pub memory: OptimizationMemoryMetrics,
    pub evaluation: Option<OptimizationEvaluationMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct OptimizationEnvironment {
    pub binary_version: Option<String>,
    pub git_revision: Option<String>,
    pub adapter_name: Option<String>,
    pub backend: Option<String>,
    pub driver: Option<String>,
    pub timestamp_query_available: Option<bool>,
    /// Present when adapter_name could not be probed at runtime.
    pub adapter_unavailable_reason: Option<String>,
    /// Present when driver could not be probed at runtime.
    pub driver_unavailable_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct OptimizationCommand {
    pub argv: Vec<String>,
    pub dataset_fingerprint: Option<String>,
    pub frame_shuffle_seed: Option<u64>,
    pub render_scale: Option<f32>,
    pub eval_render_scale: Option<f32>,
    pub eval_frame_ids: Vec<u32>,
    pub train_frame_ids: Vec<u32>,
    /// Pose count that actually entered the training loader after filtering.
    pub effective_max_frames: Option<usize>,
    pub eval_resolution: Option<[usize; 2]>,
    pub iterations: Option<usize>,
    pub max_frames: Option<usize>,
    pub loss_config_fingerprint: Option<String>,
    pub topology_config_fingerprint: Option<String>,
    pub sh_schedule_fingerprint: Option<String>,
    pub training_config_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct OptimizationTrainMetrics {
    pub wall_clock_seconds: Option<f64>,
    pub training_loop_seconds: Option<f64>,
    pub completed_iterations: Option<usize>,
    pub steps_per_second: Option<f64>,
    pub loop_duration_p50_ms: Option<f64>,
    pub loop_duration_p95_ms: Option<f64>,
    /// CPU submit-side loop samples only; not GPU completion time.
    pub loop_timing_kind: Option<String>,
    /// Aggregate GPU completion seconds from timestamp queries; null when unsupported.
    pub gpu_completion_seconds: Option<f64>,
    pub gpu_step_p50_ms: Option<f64>,
    pub gpu_step_p95_ms: Option<f64>,
    pub gpu_step_sample_count: Option<u64>,
    pub gpu_profiler_unsupported_reason: Option<String>,
    pub loss_readback_count: Option<usize>,
    pub count_readback_count: Option<usize>,
    pub status_readbacks: Option<usize>,
    pub status_readbacks_loss_cadence: Option<usize>,
    pub status_readbacks_topology: Option<usize>,
    pub status_readbacks_checkpoint: Option<usize>,
    pub status_readbacks_pause: Option<usize>,
    pub status_readbacks_cancel: Option<usize>,
    pub status_readbacks_training_end: Option<usize>,
    pub status_readbacks_forward_abort: Option<usize>,
    pub status_readbacks_step_disposition: Option<usize>,
    pub gpu_gate_optimizer_skips: Option<usize>,
    pub gpu_gate_backward_skips: Option<usize>,
    pub gpu_gate_topology_skips: Option<usize>,
    pub host_safety_point_aborts: Option<usize>,
    pub sort_dispatch_count_p50: Option<usize>,
    pub sort_dispatch_count_p95: Option<usize>,
    pub scan_dispatch_count_p50: Option<usize>,
    pub scan_dispatch_count_p95: Option<usize>,
    pub sort_workspace_bytes: Option<usize>,
    pub scan_workspace_bytes: Option<usize>,
    pub initial_gaussians: Option<usize>,
    pub final_gaussians: Option<usize>,
    pub final_loss: Option<f32>,
    pub forward_capacity: Option<crate::training::ForwardCapacityTelemetry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct OptimizationTopologyMetrics {
    pub scheduled_steps: Option<usize>,
    pub mutations: Option<usize>,
    pub skipped_no_eligible_candidates: Option<usize>,
    pub accumulator_resets: Option<usize>,
    pub snapshot_ms_p50: Option<f64>,
    pub plan_ms_p50: Option<f64>,
    pub apply_ms_p50: Option<f64>,
    pub snapshot_readback_bytes: Option<usize>,
    pub densify_added: Option<usize>,
    pub prune_removed: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct OptimizationMemoryMetrics {
    pub peak_rss_bytes: Option<u64>,
    pub peak_device_bytes: Option<u64>,
    pub peak_device_bytes_reason: Option<String>,
    pub estimated_buffer_bytes: Option<u64>,
    pub workspace_current_bytes: Option<u64>,
    pub workspace_peak_bytes: Option<u64>,
    pub workspace_growth_count: Option<u64>,
    pub fresh_step_allocations: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct OptimizationEvaluationMetrics {
    pub frame_count: Option<usize>,
    pub render_width: Option<usize>,
    pub render_height: Option<usize>,
    pub psnr_mean_db: Option<f32>,
    pub psnr_median_db: Option<f32>,
    pub psnr_min_db: Option<f32>,
    pub psnr_max_db: Option<f32>,
    pub worst_frame_ids: Vec<u32>,
    pub frames: Vec<OptimizationEvalFrame>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct OptimizationEvalFrame {
    pub frame_id: u32,
    pub psnr_db: f32,
    pub sharpness_grad_ratio: Option<f32>,
    pub sharpness_lap_ratio: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationCompareDecision {
    Compatible,
    Rejected { reasons: Vec<String> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationCompareResult {
    pub decision: OptimizationCompareDecision,
    pub deltas: Vec<OptimizationMetricDelta>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OptimizationMetricDelta {
    pub name: String,
    pub baseline: Option<f64>,
    pub candidate: Option<f64>,
    pub delta: Option<f64>,
}

pub fn duration_millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

pub fn percentile_f64(sorted: &[f64], percentile: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let percentile = percentile.clamp(0.0, 100.0);
    let rank = ((percentile / 100.0) * (sorted.len().saturating_sub(1) as f64)).round() as usize;
    sorted.get(rank).copied()
}

pub fn percentile_usize(values: &[usize], percentile: f64) -> Option<usize> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let percentile = percentile.clamp(0.0, 100.0);
    let rank = ((percentile / 100.0) * (sorted.len().saturating_sub(1) as f64)).round() as usize;
    sorted.get(rank).copied()
}

pub fn default_optimization_report_path(output: &Path) -> PathBuf {
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let stem = output
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("scene");
    parent.join(format!("{stem}.optimization.json"))
}

pub fn write_optimization_report(path: &Path, report: &OptimizationReport) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let json = serde_json::to_vec_pretty(report)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    let temp = path.with_extension("optimization.json.tmp");
    {
        let mut file = fs::File::create(&temp)?;
        file.write_all(&json)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
    }
    fs::rename(&temp, path)?;
    Ok(())
}

pub fn load_optimization_report(path: &Path) -> io::Result<OptimizationReport> {
    let bytes = fs::read(path)?;
    serde_json::from_slice(&bytes).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
}

pub fn compare_optimization_reports(
    baseline: &OptimizationReport,
    candidate: &OptimizationReport,
) -> OptimizationCompareResult {
    let mut reasons = Vec::new();
    compare_opt_eq(
        &mut reasons,
        "dataset_fingerprint",
        baseline.command.dataset_fingerprint.as_deref(),
        candidate.command.dataset_fingerprint.as_deref(),
    );
    compare_opt_eq(
        &mut reasons,
        "frame_shuffle_seed",
        baseline.command.frame_shuffle_seed,
        candidate.command.frame_shuffle_seed,
    );
    compare_opt_f32(
        &mut reasons,
        "render_scale",
        baseline.command.render_scale,
        candidate.command.render_scale,
    );
    compare_opt_f32(
        &mut reasons,
        "eval_render_scale",
        baseline.command.eval_render_scale,
        candidate.command.eval_render_scale,
    );
    compare_opt_eq(
        &mut reasons,
        "eval_resolution",
        baseline.command.eval_resolution,
        candidate.command.eval_resolution,
    );
    if baseline.command.eval_frame_ids != candidate.command.eval_frame_ids {
        reasons.push(format!(
            "eval_frame_ids mismatch: {:?} vs {:?}",
            baseline.command.eval_frame_ids, candidate.command.eval_frame_ids
        ));
    }
    if baseline.command.train_frame_ids != candidate.command.train_frame_ids {
        reasons.push(format!(
            "train_frame_ids mismatch: {:?} vs {:?}",
            baseline.command.train_frame_ids, candidate.command.train_frame_ids
        ));
    }
    compare_opt_eq(
        &mut reasons,
        "effective_max_frames",
        baseline.command.effective_max_frames,
        candidate.command.effective_max_frames,
    );
    compare_opt_eq(
        &mut reasons,
        "iterations",
        baseline.command.iterations,
        candidate.command.iterations,
    );
    compare_opt_eq(
        &mut reasons,
        "max_frames",
        baseline.command.max_frames,
        candidate.command.max_frames,
    );
    compare_opt_eq(
        &mut reasons,
        "loss_config_fingerprint",
        baseline.command.loss_config_fingerprint.as_deref(),
        candidate.command.loss_config_fingerprint.as_deref(),
    );
    compare_opt_eq(
        &mut reasons,
        "topology_config_fingerprint",
        baseline.command.topology_config_fingerprint.as_deref(),
        candidate.command.topology_config_fingerprint.as_deref(),
    );
    compare_opt_eq(
        &mut reasons,
        "sh_schedule_fingerprint",
        baseline.command.sh_schedule_fingerprint.as_deref(),
        candidate.command.sh_schedule_fingerprint.as_deref(),
    );
    compare_opt_eq(
        &mut reasons,
        "training_config_fingerprint",
        baseline.command.training_config_fingerprint.as_deref(),
        candidate.command.training_config_fingerprint.as_deref(),
    );

    let deltas = vec![
        metric_delta(
            "training_loop_seconds",
            baseline.train.training_loop_seconds,
            candidate.train.training_loop_seconds,
        ),
        metric_delta(
            "steps_per_second",
            baseline.train.steps_per_second,
            candidate.train.steps_per_second,
        ),
        metric_delta(
            "psnr_mean_db",
            baseline
                .evaluation
                .as_ref()
                .and_then(|evaluation| evaluation.psnr_mean_db)
                .map(f64::from),
            candidate
                .evaluation
                .as_ref()
                .and_then(|evaluation| evaluation.psnr_mean_db)
                .map(f64::from),
        ),
        metric_delta(
            "psnr_min_db",
            baseline
                .evaluation
                .as_ref()
                .and_then(|evaluation| evaluation.psnr_min_db)
                .map(f64::from),
            candidate
                .evaluation
                .as_ref()
                .and_then(|evaluation| evaluation.psnr_min_db)
                .map(f64::from),
        ),
        metric_delta(
            "final_gaussians",
            baseline.train.final_gaussians.map(|value| value as f64),
            candidate.train.final_gaussians.map(|value| value as f64),
        ),
    ];

    let decision = if reasons.is_empty() {
        OptimizationCompareDecision::Compatible
    } else {
        OptimizationCompareDecision::Rejected { reasons }
    };
    OptimizationCompareResult { decision, deltas }
}

fn compare_opt_eq<T: PartialEq + std::fmt::Debug>(
    reasons: &mut Vec<String>,
    name: &str,
    left: Option<T>,
    right: Option<T>,
) {
    if left != right {
        reasons.push(format!("{name} mismatch: {left:?} vs {right:?}"));
    }
}

fn compare_opt_f32(reasons: &mut Vec<String>, name: &str, left: Option<f32>, right: Option<f32>) {
    match (left, right) {
        (None, None) => {}
        (Some(a), Some(b)) if (a - b).abs() <= 1e-6 => {}
        _ => reasons.push(format!("{name} mismatch: {left:?} vs {right:?}")),
    }
}

fn metric_delta(
    name: &str,
    baseline: Option<f64>,
    candidate: Option<f64>,
) -> OptimizationMetricDelta {
    OptimizationMetricDelta {
        name: name.to_string(),
        baseline,
        candidate,
        delta: match (baseline, candidate) {
            (Some(a), Some(b)) => Some(b - a),
            _ => None,
        },
    }
}

pub fn current_peak_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        // SAFETY: getrusage with RUSAGE_SELF fills a local rusage struct.
        unsafe {
            let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
            if libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) == 0 {
                let usage = usage.assume_init();
                // macOS reports ru_maxrss in bytes.
                return Some(usage.ru_maxrss as u64);
            }
        }
        None
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// Assemble a comparable report from already-summarized training telemetry fields.
pub fn build_optimization_report(
    environment: OptimizationEnvironment,
    command: OptimizationCommand,
    train: OptimizationTrainMetrics,
    topology: OptimizationTopologyMetrics,
    memory: OptimizationMemoryMetrics,
    evaluation: Option<OptimizationEvaluationMetrics>,
) -> OptimizationReport {
    OptimizationReport {
        environment,
        command,
        train,
        topology,
        memory,
        evaluation,
    }
}

/// Canonical JSON fingerprint: sorted object keys, paths/log levels stripped.
pub fn canonical_config_fingerprint<T: Serialize>(value: &T) -> Result<String, String> {
    let raw = serde_json::to_value(value).map_err(|err| err.to_string())?;
    let scrubbed = scrub_fingerprint_value(raw);
    let canonical = canonicalize_json(scrubbed);
    let bytes = serde_json::to_vec(&canonical).map_err(|err| err.to_string())?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

/// Implicit SH schedule until Task 5.3 introduces an explicit config struct.
pub fn sh_schedule_fingerprint(max_degree: u32, increment_every: usize) -> Result<String, String> {
    #[derive(Serialize)]
    struct ShScheduleFingerprint {
        initial_degree: u32,
        increment_every: usize,
        max_degree: u32,
    }
    canonical_config_fingerprint(&ShScheduleFingerprint {
        initial_degree: 0,
        increment_every: increment_every.max(1),
        max_degree,
    })
}

fn scrub_fingerprint_value(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (key, child) in map {
                if should_omit_fingerprint_key(&key) {
                    continue;
                }
                out.insert(key, scrub_fingerprint_value(child));
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.into_iter().map(scrub_fingerprint_value).collect())
        }
        other => other,
    }
}

fn should_omit_fingerprint_key(key: &str) -> bool {
    let lower = key.to_ascii_lowercase();
    lower == "log_level"
        || lower == "argv"
        || lower.ends_with("_path")
        || lower.ends_with("_dir")
        || lower == "path"
        || lower == "input"
        || lower == "output"
}

fn canonicalize_json(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut keys = map.keys().cloned().collect::<Vec<_>>();
            keys.sort();
            let mut out = serde_json::Map::new();
            for key in keys {
                if let Some(child) = map.get(&key) {
                    out.insert(key, canonicalize_json(child.clone()));
                }
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.into_iter().map(canonicalize_json).collect())
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::ForwardCapacityTelemetry;

    fn sample_report() -> OptimizationReport {
        OptimizationReport {
            environment: OptimizationEnvironment {
                binary_version: Some("0.1.0".into()),
                git_revision: Some("deadbeef".into()),
                adapter_name: Some("Apple M5 Max".into()),
                backend: Some("Metal".into()),
                driver: None,
                timestamp_query_available: Some(false),
                adapter_unavailable_reason: None,
                driver_unavailable_reason: Some("driver_info_empty".into()),
            },
            command: OptimizationCommand {
                argv: vec!["rustgs".into(), "train".into()],
                dataset_fingerprint: Some("dataset-a".into()),
                frame_shuffle_seed: Some(0),
                render_scale: Some(0.25),
                eval_render_scale: Some(0.25),
                eval_frame_ids: vec![0, 1],
                train_frame_ids: vec![10, 11, 12],
                effective_max_frames: Some(12),
                eval_resolution: Some([160, 90]),
                iterations: Some(500),
                max_frames: Some(12),
                loss_config_fingerprint: Some("loss-fp".into()),
                topology_config_fingerprint: Some("topo-fp".into()),
                sh_schedule_fingerprint: Some("sh-fp".into()),
                training_config_fingerprint: Some("train-fp".into()),
            },
            train: OptimizationTrainMetrics {
                wall_clock_seconds: Some(12.0),
                training_loop_seconds: Some(10.0),
                completed_iterations: Some(500),
                steps_per_second: Some(50.0),
                loop_duration_p50_ms: Some(18.0),
                loop_duration_p95_ms: Some(22.0),
                loop_timing_kind: Some("cpu_submit_instant".into()),
                gpu_completion_seconds: None,
                gpu_step_p50_ms: None,
                gpu_step_p95_ms: None,
                gpu_step_sample_count: Some(0),
                gpu_profiler_unsupported_reason: Some("timestamp_query_unavailable".into()),
                loss_readback_count: Some(26),
                count_readback_count: Some(0),
                status_readbacks: Some(27),
                status_readbacks_loss_cadence: Some(26),
                status_readbacks_topology: Some(0),
                status_readbacks_checkpoint: Some(0),
                status_readbacks_pause: Some(0),
                status_readbacks_cancel: Some(0),
                status_readbacks_training_end: Some(1),
                status_readbacks_forward_abort: Some(0),
                status_readbacks_step_disposition: Some(0),
                gpu_gate_optimizer_skips: Some(0),
                gpu_gate_backward_skips: Some(0),
                gpu_gate_topology_skips: Some(0),
                host_safety_point_aborts: Some(0),
                sort_dispatch_count_p50: Some(8),
                sort_dispatch_count_p95: Some(8),
                scan_dispatch_count_p50: Some(3),
                scan_dispatch_count_p95: Some(3),
                sort_workspace_bytes: Some(1_024),
                scan_workspace_bytes: Some(512),
                initial_gaussians: Some(100),
                final_gaussians: Some(110),
                final_loss: Some(0.12),
                forward_capacity: Some(ForwardCapacityTelemetry {
                    logical_visible: 80,
                    logical_intersections: 1_000,
                    capacity: 8_000,
                    overflowed: false,
                }),
            },
            topology: OptimizationTopologyMetrics {
                scheduled_steps: Some(2),
                mutations: Some(1),
                skipped_no_eligible_candidates: Some(0),
                accumulator_resets: Some(1),
                snapshot_ms_p50: Some(1.5),
                plan_ms_p50: Some(0.4),
                apply_ms_p50: Some(0.8),
                snapshot_readback_bytes: Some(4_096),
                densify_added: Some(10),
                prune_removed: Some(0),
            },
            memory: OptimizationMemoryMetrics {
                peak_rss_bytes: Some(1_000_000),
                peak_device_bytes: None,
                peak_device_bytes_reason: Some("runtime_peak_device_bytes_unavailable".into()),
                estimated_buffer_bytes: Some(2_000_000),
                workspace_current_bytes: Some(512),
                workspace_peak_bytes: Some(512),
                workspace_growth_count: Some(1),
                fresh_step_allocations: Some(2),
            },
            evaluation: Some(OptimizationEvaluationMetrics {
                frame_count: Some(2),
                render_width: Some(160),
                render_height: Some(90),
                psnr_mean_db: Some(20.0),
                psnr_median_db: Some(20.1),
                psnr_min_db: Some(19.5),
                psnr_max_db: Some(20.5),
                worst_frame_ids: vec![1],
                frames: vec![
                    OptimizationEvalFrame {
                        frame_id: 0,
                        psnr_db: 20.5,
                        sharpness_grad_ratio: Some(0.2),
                        sharpness_lap_ratio: Some(0.1),
                    },
                    OptimizationEvalFrame {
                        frame_id: 1,
                        psnr_db: 19.5,
                        sharpness_grad_ratio: Some(0.3),
                        sharpness_lap_ratio: Some(0.2),
                    },
                ],
            }),
        }
    }

    #[test]
    fn optimization_report_round_trips_and_keeps_nulls() {
        let report = sample_report();
        let json = serde_json::to_string_pretty(&report).expect("serialize");
        assert!(json.contains("\"gpu_completion_seconds\": null"));
        assert!(json.contains("\"peak_device_bytes\": null"));
        assert!(json.contains("\"driver\": null"));
        assert!(json.contains("\"gpu_step_p50_ms\": null"));
        assert!(json.contains("timestamp_query_unavailable"));
        let decoded: OptimizationReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded, report);
        assert_eq!(decoded.evaluation.as_ref().unwrap().frames.len(), 2);
        assert_eq!(decoded.topology.mutations, Some(1));
    }

    #[test]
    fn comparator_rejects_mismatched_eval_fingerprint() {
        let baseline = sample_report();
        let mut candidate = sample_report();
        candidate.command.eval_frame_ids = vec![0, 2];
        candidate.train.steps_per_second = Some(60.0);
        let result = compare_optimization_reports(&baseline, &candidate);
        match result.decision {
            OptimizationCompareDecision::Rejected { reasons } => {
                assert!(reasons
                    .iter()
                    .any(|reason| reason.contains("eval_frame_ids")));
            }
            OptimizationCompareDecision::Compatible => panic!("expected rejection"),
        }
        assert!(result
            .deltas
            .iter()
            .any(|delta| delta.name == "steps_per_second" && delta.delta == Some(10.0)));
    }

    #[test]
    fn comparator_accepts_matching_inputs() {
        let baseline = sample_report();
        let mut candidate = sample_report();
        candidate.train.training_loop_seconds = Some(9.0);
        let result = compare_optimization_reports(&baseline, &candidate);
        assert_eq!(result.decision, OptimizationCompareDecision::Compatible);
    }

    #[test]
    fn comparator_rejects_each_identity_field_mismatch() {
        let cases: Vec<(&str, Box<dyn FnMut(&mut OptimizationCommand)>)> = vec![
            (
                "train_frame_ids",
                Box::new(|cmd| cmd.train_frame_ids = vec![99]),
            ),
            (
                "effective_max_frames",
                Box::new(|cmd| cmd.effective_max_frames = Some(99)),
            ),
            (
                "loss_config_fingerprint",
                Box::new(|cmd| cmd.loss_config_fingerprint = Some("x".into())),
            ),
            (
                "topology_config_fingerprint",
                Box::new(|cmd| cmd.topology_config_fingerprint = Some("x".into())),
            ),
            (
                "sh_schedule_fingerprint",
                Box::new(|cmd| cmd.sh_schedule_fingerprint = Some("x".into())),
            ),
            (
                "training_config_fingerprint",
                Box::new(|cmd| cmd.training_config_fingerprint = Some("x".into())),
            ),
            ("iterations", Box::new(|cmd| cmd.iterations = Some(1))),
            ("max_frames", Box::new(|cmd| cmd.max_frames = Some(1))),
            (
                "dataset_fingerprint",
                Box::new(|cmd| cmd.dataset_fingerprint = Some("other".into())),
            ),
            (
                "frame_shuffle_seed",
                Box::new(|cmd| cmd.frame_shuffle_seed = Some(9)),
            ),
            ("render_scale", Box::new(|cmd| cmd.render_scale = Some(0.5))),
            (
                "eval_render_scale",
                Box::new(|cmd| cmd.eval_render_scale = Some(0.5)),
            ),
            (
                "eval_resolution",
                Box::new(|cmd| cmd.eval_resolution = Some([1, 1])),
            ),
        ];

        for (needle, mut mutate) in cases {
            let baseline = sample_report();
            let mut candidate = sample_report();
            mutate(&mut candidate.command);
            let result = compare_optimization_reports(&baseline, &candidate);
            match result.decision {
                OptimizationCompareDecision::Rejected { reasons } => {
                    assert!(
                        reasons.iter().any(|reason| reason.contains(needle)),
                        "expected {needle} rejection, got {reasons:?}"
                    );
                }
                OptimizationCompareDecision::Compatible => {
                    panic!("expected rejection for {needle}")
                }
            }
        }
    }

    #[test]
    fn fingerprint_is_stable_and_ignores_paths_and_log_level() {
        #[derive(Serialize)]
        struct Sample {
            loss_l1_weight: f32,
            input_path: String,
            log_level: String,
            nested: Nested,
        }
        #[derive(Serialize)]
        struct Nested {
            output_dir: String,
            refine_every: usize,
        }
        let a = Sample {
            loss_l1_weight: 0.8,
            input_path: "/tmp/a".into(),
            log_level: "debug".into(),
            nested: Nested {
                output_dir: "/tmp/out-a".into(),
                refine_every: 100,
            },
        };
        let b = Sample {
            loss_l1_weight: 0.8,
            input_path: "/other/b".into(),
            log_level: "info".into(),
            nested: Nested {
                output_dir: "/other/out-b".into(),
                refine_every: 100,
            },
        };
        let fa = canonical_config_fingerprint(&a).expect("fp a");
        let fb = canonical_config_fingerprint(&b).expect("fp b");
        assert_eq!(fa, fb);
        assert_ne!(
            fa,
            canonical_config_fingerprint(&Sample {
                loss_l1_weight: 0.9,
                input_path: "/tmp/a".into(),
                log_level: "debug".into(),
                nested: Nested {
                    output_dir: "/tmp/out-a".into(),
                    refine_every: 100,
                },
            })
            .unwrap()
        );
    }

    #[test]
    fn sh_schedule_fingerprint_encodes_increment_and_max_degree() {
        let a = sh_schedule_fingerprint(3, 1000).unwrap();
        let b = sh_schedule_fingerprint(3, 1000).unwrap();
        let c = sh_schedule_fingerprint(2, 1000).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn default_report_path_sits_beside_output() {
        let path = default_optimization_report_path(Path::new("artifacts/runs/home-500.ply"));
        assert_eq!(
            path,
            PathBuf::from("artifacts/runs/home-500.optimization.json")
        );
    }
}
