//! Lost-solution attribution: CPU f64 has a model, GPU f32 has none.
//!
//! Pure measurement. For every trial this computes the f64 singular values of
//! the 5x9 constraint matrix, so the GPU nullspace gate can be evaluated in
//! exact-enough arithmetic and compared with the GPU's own rank decision, and
//! scores every CPU and GPU model on the pair's full observation set with the
//! pipeline's Sampson threshold, so "lost" can be weighed by model quality
//! rather than counted.
//!
//! Not RANSAC, not the 960-image pipeline, no CPU runtime fallback. The CPU f64
//! replay is an offline reference only.
#[allow(dead_code)]
#[path = "five_point_gpu_capacity_replay.rs"]
mod shared;

use anyhow::{ensure, Context, Result};
use clap::Parser;
use nalgebra::{DMatrix, Matrix3, Vector3};
use rayon::{prelude::*, ThreadPoolBuilder};
use rustscan_sfm::{
    database::ColmapDatabase,
    gpu::{FivePointStatus, FivePointTrialResult, WgpuContext, WgpuFivePointF32},
    types::CameraModel,
};
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf};

const INPUT: &str = "af07459d637b5b8c2b20c76d3ef103f058505fe0e8307578ee7e8750f3e230c5";
const GPU_SIGNATURE: &str = "1f6f3f8d88a17e4e1c42abfc85aad44a66c4b864091805594e086ca80bf9b339";
const CPU_SIGNATURE: &str = "9e6764b41602d0005710187d2558d046f1a843c5f873b9542dd7b4f6c29c563a";
/// `five_point_f32.wgsl` line 66: an eigenvalue of A^T A counts toward the rank
/// only if it exceeds this fraction of the trace.
const GPU_RANK_GATE: f64 = 1e-6;
/// Pipeline default `TwoViewGeometry.max_error` in pixels; converted per pair
/// through the cameras' mean focal length exactly as `geometry.rs` does.
const MAX_ERROR_PX: f64 = 4.0;
const TRIALS_PER_PAIR: usize = 512;

/// Row-major 5x9 constraint matrix, identical construction to both solvers.
fn constraint_matrix(left: &[Vector3<f64>; 5], right: &[Vector3<f64>; 5]) -> DMatrix<f64> {
    let mut rows = Vec::with_capacity(45);
    for (a, b) in left.iter().zip(right) {
        rows.extend_from_slice(&[
            b.x * a.x,
            b.x * a.y,
            b.x * a.z,
            b.y * a.x,
            b.y * a.y,
            b.y * a.z,
            b.z * a.x,
            b.z * a.y,
            b.z * a.z,
        ]);
    }
    DMatrix::from_row_slice(5, 9, &rows)
}

/// Singular values descending, plus the GPU gate quantity sigma_5^2 / sum sigma^2.
/// Trace(A^T A) = sum sigma^2 and the GPU's max-abs prescaling cancels in the
/// ratio, so this is the exact-arithmetic value of what the kernel compares.
fn spectrum(a: &DMatrix<f64>) -> ([f64; 5], f64) {
    let mut s: Vec<f64> = a
        .clone()
        .svd(false, false)
        .singular_values
        .iter()
        .copied()
        .collect();
    s.sort_by(|x, y| y.total_cmp(x));
    let sigma: [f64; 5] = std::array::from_fn(|i| s[i]);
    let total: f64 = sigma.iter().map(|v| v * v).sum();
    (sigma, sigma[4] * sigma[4] / total)
}

fn log_bucket(ratio: f64) -> &'static str {
    // NaN and zero both land here on purpose; NaN fails every comparison.
    if ratio.is_nan() || ratio <= 0.0 {
        "0 or non-finite"
    } else if ratio >= 1e-1 {
        ">= 1e-1"
    } else if ratio >= 1e-2 {
        "[1e-2, 1e-1)"
    } else if ratio >= 1e-3 {
        "[1e-3, 1e-2)"
    } else if ratio >= 1e-4 {
        "[1e-4, 1e-3)"
    } else if ratio >= 1e-5 {
        "[1e-5, 1e-4)"
    } else if ratio >= 1e-6 {
        "[1e-6, 1e-5)"
    } else if ratio >= 1e-7 {
        "[1e-7, 1e-6)"
    } else {
        "< 1e-7"
    }
}

/// Mirrors `two_view::squared_sampson_error`, which is crate-private.
fn squared_sampson(x1: &Vector3<f64>, x2: &Vector3<f64>, e: &Matrix3<f64>) -> f64 {
    let ex1 = e * x1;
    let etx2 = e.transpose() * x2;
    let num = x2.dot(&ex1);
    let denom = ex1.x * ex1.x + ex1.y * ex1.y + etx2.x * etx2.x + etx2.y * etx2.y;
    if denom <= 1.0e-24 {
        f64::INFINITY
    } else {
        num * num / denom
    }
}

fn inliers(obs: &shared::cpu::PairObservations, e: &Matrix3<f64>, threshold: f64) -> usize {
    let limit = threshold.max(1.0e-12).powi(2);
    obs.left
        .iter()
        .zip(&obs.right)
        .filter(|(l, r)| {
            let v = squared_sampson(l, r, e);
            v.is_finite() && v <= limit
        })
        .count()
}

fn best_inliers(
    obs: &shared::cpu::PairObservations,
    models: impl Iterator<Item = Matrix3<f64>>,
    threshold: f64,
) -> usize {
    models
        .map(|m| inliers(obs, &m, threshold))
        .max()
        .unwrap_or(0)
}

fn gpu_models(trial: &FivePointTrialResult) -> impl Iterator<Item = Matrix3<f64>> + '_ {
    trial
        .slots
        .iter()
        .filter_map(|s| s.essential)
        .map(|e| Matrix3::from_row_slice(&e.map(f64::from)))
}

fn stage(trial: &FivePointTrialResult) -> &'static str {
    match trial.upstream_status {
        FivePointStatus::RankDeficient => "nullspace_rank_deficient",
        FivePointStatus::NotConverged => "nullspace_not_converged",
        FivePointStatus::InvalidBasis => "nullspace_invalid_basis",
        FivePointStatus::SingularElimination => "algebra_singular",
        FivePointStatus::NonFinite => "algebra_nonfinite",
        FivePointStatus::Success if trial.polynomial_failed => "polynomial_failed",
        FivePointStatus::Success if trial.model_count == 0 => "roots_all_rejected",
        FivePointStatus::Success => "produced_models",
    }
}

fn quality_bucket(ratio: f64) -> &'static str {
    match ratio {
        r if r >= 1.0 => "== pair best",
        r if r >= 0.9 => "[0.9, 1.0)",
        r if r >= 0.5 => "[0.5, 0.9)",
        r if r >= 0.25 => "[0.25, 0.5)",
        _ => "< 0.25",
    }
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted[((sorted.len() - 1) as f64 * fraction).round() as usize]
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    database: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.output.exists(), "refusing to overwrite experiment");
    let (inputs, pairs, _, observations) =
        shared::cpu::load_with_observations(&shared::cpu::Args {
            database: args.database.clone(),
            pairs: 127,
            trials: 512,
            batch_size: 512,
        })?;
    ensure!(
        inputs.len() == 65024 && shared::cpu::input_digest(&inputs) == INPUT,
        "input provenance changed"
    );
    ensure!(
        observations.len() == pairs.len() && pairs.len() * TRIALS_PER_PAIR == inputs.len(),
        "pair / trial layout changed"
    );

    // Per-pair Sampson threshold, as geometry.rs derives it from the two cameras.
    let db = ColmapDatabase::open_read_only(&args.database)?;
    let images = db.read_all_images()?;
    let camera_for = |id: u64| -> Result<CameraModel> {
        let image = images
            .iter()
            .find(|i| u64::from(i.image_id) == id)
            .context("missing image")?;
        let c = db
            .read_camera(image.camera_id)?
            .context("missing camera")?
            .camera;
        CameraModel::from_colmap(c.model_id, c.width, c.height, &c.params)
            .context("unsupported camera")
    };
    let mut thresholds = Vec::with_capacity(pairs.len());
    for pair in &pairs {
        let ids = pair["image_ids"].as_array().context("image_ids")?;
        let c1 = camera_for(ids[0].as_u64().context("id")?)?;
        let c2 = camera_for(ids[1].as_u64().context("id")?)?;
        thresholds.push(
            0.5 * (c1.cam_from_img_threshold(MAX_ERROR_PX)
                + c2.cam_from_img_threshold(MAX_ERROR_PX)),
        );
    }

    // Both solvers, both signatures pinned to history.
    let context = WgpuContext::try_new()?;
    let device = format!("{:?}", context.capabilities());
    let gpu = WgpuFivePointF32::from_context(context)?;
    let gpu_results = shared::gpu_replay(&gpu, &inputs, inputs.len())?;
    let gpu_signature = shared::signature(&shared::gpu_bits(&gpu_results));
    ensure!(gpu_signature == GPU_SIGNATURE, "GPU signature changed");
    let pool = ThreadPoolBuilder::new().num_threads(8).build()?;
    let cpu_results = shared::cpu::replay_batches(&inputs, Some(&pool), inputs.len());
    let mut hasher = blake3::Hasher::new();
    for models in &shared::cpu::output_bits(&cpu_results) {
        hasher.update(&(models.len() as u64).to_le_bytes());
        for m in models {
            for w in m {
                hasher.update(&w.to_le_bytes());
            }
        }
    }
    let cpu_signature = hasher.finalize().to_hex().to_string();
    ensure!(cpu_signature == CPU_SIGNATURE, "CPU signature changed");

    // Per-trial spectrum and scores, in parallel; order is preserved by index.
    struct Row {
        sigma: [f64; 5],
        gate: f64,
        cpu_best: usize,
        gpu_best: usize,
    }
    let rows: Vec<Row> = pool.install(|| {
        (0..inputs.len())
            .into_par_iter()
            .map(|i| {
                let pair = i / TRIALS_PER_PAIR;
                let (sigma, gate) = spectrum(&constraint_matrix(&inputs[i].left, &inputs[i].right));
                let obs = &observations[pair];
                let t = thresholds[pair];
                Row {
                    sigma,
                    gate,
                    cpu_best: best_inliers(obs, cpu_results[i].iter().copied(), t),
                    gpu_best: best_inliers(obs, gpu_models(&gpu_results[i]), t),
                }
            })
            .collect()
    });

    // Pair-level best over all 512 CPU trials: the reference each trial's
    // quality is measured against, and what RANSAC would ultimately keep.
    let pair_cpu_best: Vec<usize> = (0..pairs.len())
        .map(|p| {
            rows[p * TRIALS_PER_PAIR..(p + 1) * TRIALS_PER_PAIR]
                .iter()
                .map(|r| r.cpu_best)
                .max()
                .unwrap_or(0)
        })
        .collect();
    let pair_gpu_best: Vec<usize> = (0..pairs.len())
        .map(|p| {
            rows[p * TRIALS_PER_PAIR..(p + 1) * TRIALS_PER_PAIR]
                .iter()
                .map(|r| r.gpu_best)
                .max()
                .unwrap_or(0)
        })
        .collect();

    // ---- Attribution over lost trials --------------------------------------
    let lost: Vec<usize> = (0..inputs.len())
        .filter(|&i| !cpu_results[i].is_empty() && gpu_results[i].model_count == 0)
        .collect();
    let kept: Vec<usize> = (0..inputs.len())
        .filter(|&i| !cpu_results[i].is_empty() && gpu_results[i].model_count > 0)
        .collect();

    let mut lost_stage: BTreeMap<&str, usize> = BTreeMap::new();
    let mut gpu_rank_when_deficient: BTreeMap<usize, usize> = BTreeMap::new();
    let mut gate_hist_lost_rd: BTreeMap<&str, usize> = BTreeMap::new();
    let mut gate_hist_kept: BTreeMap<&str, usize> = BTreeMap::new();
    let mut ratio_hist_lost_rd: BTreeMap<&str, usize> = BTreeMap::new();
    let mut ratio_hist_kept: BTreeMap<&str, usize> = BTreeMap::new();
    let mut rd_gate_below = 0usize; // H1: f64 gate already below 1e-6
    let mut rd_gate_above = 0usize; // H2: f64 gate above 1e-6 yet GPU rejected
    let mut rd_gate_above_values = Vec::new();
    let mut kept_gate_below = 0usize; // sanity: gate below yet GPU accepted
    for &i in &lost {
        let t = &gpu_results[i];
        *lost_stage.entry(stage(t)).or_default() += 1;
        if t.upstream_status == FivePointStatus::RankDeficient {
            *gpu_rank_when_deficient.entry(t.rank).or_default() += 1;
            *gate_hist_lost_rd
                .entry(log_bucket(rows[i].gate))
                .or_default() += 1;
            *ratio_hist_lost_rd
                .entry(log_bucket(rows[i].sigma[4] / rows[i].sigma[0]))
                .or_default() += 1;
            if rows[i].gate < GPU_RANK_GATE {
                rd_gate_below += 1;
            } else {
                rd_gate_above += 1;
                rd_gate_above_values.push(rows[i].gate);
            }
        }
    }
    for &i in &kept {
        *gate_hist_kept.entry(log_bucket(rows[i].gate)).or_default() += 1;
        *ratio_hist_kept
            .entry(log_bucket(rows[i].sigma[4] / rows[i].sigma[0]))
            .or_default() += 1;
        if rows[i].gate < GPU_RANK_GATE {
            kept_gate_below += 1;
        }
    }
    rd_gate_above_values.sort_by(f64::total_cmp);

    // Every RankDeficient trial, lost or not, against the f64 gate.
    let all_rd: Vec<usize> = (0..inputs.len())
        .filter(|&i| gpu_results[i].upstream_status == FivePointStatus::RankDeficient)
        .collect();
    let all_rd_gate_below = all_rd
        .iter()
        .filter(|&&i| rows[i].gate < GPU_RANK_GATE)
        .count();
    let all_gate_below = rows.iter().filter(|r| r.gate < GPU_RANK_GATE).count();

    // ---- Quality: are the lost models worth having? -----------------------
    let quality = |indices: &[usize]| -> serde_json::Value {
        let mut hist: BTreeMap<&str, usize> = BTreeMap::new();
        let mut ratios = Vec::new();
        let mut is_pair_best = 0usize;
        let mut sample_only = 0usize; // best model fits only its own 5 points
        for &i in indices {
            let pair = i / TRIALS_PER_PAIR;
            let best = pair_cpu_best[pair];
            let r = if best == 0 {
                0.0
            } else {
                rows[i].cpu_best as f64 / best as f64
            };
            *hist.entry(quality_bucket(r)).or_default() += 1;
            ratios.push(r);
            if rows[i].cpu_best == best && best > 0 {
                is_pair_best += 1;
            }
            if rows[i].cpu_best <= 5 {
                sample_only += 1;
            }
        }
        ratios.sort_by(f64::total_cmp);
        json!({"trials":indices.len(),"ratio_to_pair_best_histogram":hist,
            "p50":percentile(&ratios,0.5),"p90":percentile(&ratios,0.9),
            "mean":ratios.iter().sum::<f64>()/ratios.len().max(1) as f64,
            "trials_equal_to_pair_best":is_pair_best,
            "trials_fitting_only_their_own_sample":sample_only})
    };
    let lost_rd: Vec<usize> = lost
        .iter()
        .copied()
        .filter(|&i| gpu_results[i].upstream_status == FivePointStatus::RankDeficient)
        .collect();
    let lost_roots: Vec<usize> = lost
        .iter()
        .copied()
        .filter(|&i| stage(&gpu_results[i]) == "roots_all_rejected")
        .collect();

    // Pair-level: does the GPU still reach the pair's best CPU model?
    let mut pair_rows = Vec::new();
    let mut pairs_gpu_below_cpu = 0usize;
    let mut pairs_gpu_below_90 = 0usize;
    for p in 0..pairs.len() {
        let lost_here = lost.iter().filter(|&&i| i / TRIALS_PER_PAIR == p).count();
        let rd_here = all_rd.iter().filter(|&&i| i / TRIALS_PER_PAIR == p).count();
        if pair_gpu_best[p] < pair_cpu_best[p] {
            pairs_gpu_below_cpu += 1;
        }
        if (pair_gpu_best[p] as f64) < 0.9 * pair_cpu_best[p] as f64 {
            pairs_gpu_below_90 += 1;
        }
        // Median conditioning of this pair's samples, to see whether lost
        // trials cluster in intrinsically ill-conditioned pairs.
        let mut ratios: Vec<f64> = rows[p * TRIALS_PER_PAIR..(p + 1) * TRIALS_PER_PAIR]
            .iter()
            .map(|r| r.sigma[4] / r.sigma[0])
            .collect();
        ratios.sort_by(f64::total_cmp);
        pair_rows.push(json!({"pair_id":pairs[p]["pair_id"],
            "observations":observations[p].left.len(),
            "sampson_threshold":thresholds[p],
            "cpu_best_inliers":pair_cpu_best[p],"gpu_best_inliers":pair_gpu_best[p],
            "lost_trials":lost_here,"rank_deficient_trials":rd_here,
            "median_sigma5_over_sigma1":percentile(&ratios,0.5)}));
    }
    pair_rows.sort_by_key(|r| std::cmp::Reverse(r["lost_trials"].as_u64().unwrap_or(0)));

    // Slot composition and conditioning of the root-stage losses, against the
    // trials that reached roots and produced models.
    let mut roots_slots: BTreeMap<String, usize> = BTreeMap::new();
    let mut roots_ratio_hist: BTreeMap<&str, usize> = BTreeMap::new();
    let mut roots_at_cap = 0usize;
    for &i in &lost_roots {
        for s in &gpu_results[i].slots {
            *roots_slots.entry(format!("{:?}", s.status)).or_default() += 1;
        }
        *roots_ratio_hist
            .entry(log_bucket(rows[i].sigma[4] / rows[i].sigma[0]))
            .or_default() += 1;
        if gpu_results[i].root_iterations == 384 {
            roots_at_cap += 1;
        }
    }
    let produced: Vec<usize> = (0..inputs.len())
        .filter(|&i| gpu_results[i].model_count > 0)
        .collect();
    let produced_at_cap = produced
        .iter()
        .filter(|&&i| gpu_results[i].root_iterations == 384)
        .count();
    let cpu_real_roots_lost_roots: usize = lost_roots.iter().map(|&i| cpu_results[i].len()).sum();

    let report = json!({
        "schema":"five-point-lost-solutions-v1","source_commit":"3ab9c09",
        "database":args.database,"device":device,
        "input_digest":INPUT,"gpu_signature":gpu_signature,"cpu_signature":cpu_signature,
        "trials":inputs.len(),"pairs":pairs.len(),"max_error_px":MAX_ERROR_PX,
        "gpu_rank_gate":GPU_RANK_GATE,
        "counts":{"cpu_nonempty":cpu_results.iter().filter(|m|!m.is_empty()).count(),
            "gpu_nonempty":gpu_results.iter().filter(|t|t.model_count>0).count(),
            "lost":lost.len(),"kept_both":kept.len(),
            "gpu_rank_deficient_total":all_rd.len()},
        "lost_by_stage":lost_stage,
        "nullspace_gate":{
            "hypothesis_h1_threshold":"f64 sigma_5^2/sum sigma^2 already < 1e-6: the gate itself rejects, precision irrelevant",
            "hypothesis_h2_precision":"f64 gate >= 1e-6 but GPU still reported rank < 5: f32 A^T A lost it",
            "lost_rank_deficient":lost_rd.len(),
            "h1_f64_gate_below":rd_gate_below,"h2_f64_gate_above":rd_gate_above,
            "h2_gate_values_p50":percentile(&rd_gate_above_values,0.5),
            "h2_gate_values_max":rd_gate_above_values.last().copied(),
            "all_rank_deficient":all_rd.len(),"all_rank_deficient_f64_gate_below":all_rd_gate_below,
            "all_trials_f64_gate_below":all_gate_below,
            "kept_trials_with_f64_gate_below":kept_gate_below,
            "gpu_rank_reported_when_deficient":gpu_rank_when_deficient,
            "f64_gate_histogram_lost_rank_deficient":gate_hist_lost_rd,
            "f64_gate_histogram_kept":gate_hist_kept,
            "sigma5_over_sigma1_histogram_lost_rank_deficient":ratio_hist_lost_rd,
            "sigma5_over_sigma1_histogram_kept":ratio_hist_kept},
        "quality":{
            "definition":"best Sampson inlier count of the trial's CPU models over the pair's observations, divided by the best over all 512 CPU trials of that pair",
            "lost_rank_deficient":quality(&lost_rd),
            "lost_roots_all_rejected":quality(&lost_roots),
            "kept_both":quality(&kept),
            "pairs_where_gpu_best_below_cpu_best":pairs_gpu_below_cpu,
            "pairs_where_gpu_best_below_90pct_of_cpu_best":pairs_gpu_below_90},
        "roots_stage_lost":{"trials":lost_roots.len(),"slot_statuses":roots_slots,
            "sigma5_over_sigma1_histogram":roots_ratio_hist,
            "trials_at_iteration_cap":roots_at_cap,
            "produced_models_trials_at_iteration_cap":produced_at_cap,
            "produced_models_trials":produced.len(),
            "cpu_models_in_these_trials":cpu_real_roots_lost_roots},
        "pairs_by_lost_trials":pair_rows,
        "scope":"Independent offline measurement. No solver, shader or protected harness change. CPU f64 is a reference, not a runtime fallback. Sampson scoring mirrors two_view::squared_sampson_error on unit-norm rays with the per-pair threshold geometry.rs derives from TwoViewGeometry.max_error = 4 px; it is a quality proxy for candidate models, not a RANSAC run."});
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    eprintln!(
        "lost={} rank_deficient={} h1_below={} h2_above={} pairs_gpu_below_cpu={}",
        lost.len(),
        lost_rd.len(),
        rd_gate_below,
        rd_gate_above,
        pairs_gpu_below_cpu
    );
    eprintln!("wrote {}", args.output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_is_scale_invariant_and_matches_trace_identity() {
        let left = [
            Vector3::new(0.1, 0.2, 1.0).normalize(),
            Vector3::new(-0.3, 0.1, 1.0).normalize(),
            Vector3::new(0.2, -0.4, 1.0).normalize(),
            Vector3::new(-0.1, -0.2, 1.0).normalize(),
            Vector3::new(0.4, 0.3, 1.0).normalize(),
        ];
        let right = [
            Vector3::new(0.12, 0.19, 1.0).normalize(),
            Vector3::new(-0.28, 0.11, 1.0).normalize(),
            Vector3::new(0.21, -0.38, 1.0).normalize(),
            Vector3::new(-0.09, -0.21, 1.0).normalize(),
            Vector3::new(0.41, 0.29, 1.0).normalize(),
        ];
        let a = constraint_matrix(&left, &right);
        let (sigma, gate) = spectrum(&a);
        assert!(sigma.windows(2).all(|w| w[0] >= w[1]));
        let trace = (a.transpose() * &a).trace();
        let total: f64 = sigma.iter().map(|v| v * v).sum();
        assert!((trace - total).abs() < 1e-12 * trace);
        let (_, scaled) = spectrum(&(a * 37.0));
        assert!((gate - scaled).abs() < 1e-12);
    }

    #[test]
    fn duplicate_correspondence_drops_the_rank() {
        let v = |x: f64, y: f64| Vector3::new(x, y, 1.0).normalize();
        let left = [
            v(0.1, 0.2),
            v(-0.3, 0.1),
            v(0.2, -0.4),
            v(-0.1, -0.2),
            v(0.1, 0.2),
        ];
        let right = [
            v(0.12, 0.19),
            v(-0.28, 0.11),
            v(0.21, -0.38),
            v(-0.09, -0.21),
            v(0.12, 0.19),
        ];
        let (sigma, gate) = spectrum(&constraint_matrix(&left, &right));
        assert!(sigma[4] < 1e-12 * sigma[0]);
        assert!(gate < GPU_RANK_GATE);
    }

    #[test]
    fn sampson_inliers_accept_exact_epipolar_points() {
        let t = Vector3::new(1.0, 0.0, 0.0);
        let e = Matrix3::new(0.0, -t.z, t.y, t.z, 0.0, -t.x, -t.y, t.x, 0.0);
        let pts: Vec<Vector3<f64>> = (0..20)
            .map(|i| Vector3::new(0.05 * i as f64 - 0.5, 0.03 * i as f64 - 0.3, 3.0 + i as f64))
            .collect();
        let obs = shared::cpu::PairObservations {
            left: pts.iter().map(|p| p.normalize()).collect(),
            right: pts.iter().map(|p| (p + t).normalize()).collect(),
        };
        assert_eq!(inliers(&obs, &e, 1e-3), 20);
        assert_eq!(inliers(&obs, &Matrix3::identity(), 1e-3), 0);
    }

    #[test]
    fn buckets_cover_edges() {
        assert_eq!(log_bucket(1e-6), "[1e-6, 1e-5)");
        assert_eq!(log_bucket(9.99e-7), "[1e-7, 1e-6)");
        assert_eq!(log_bucket(0.0), "0 or non-finite");
        assert_eq!(quality_bucket(1.0), "== pair best");
        assert_eq!(quality_bucket(0.9), "[0.9, 1.0)");
        assert_eq!(quality_bucket(0.1), "< 0.25");
    }
}
