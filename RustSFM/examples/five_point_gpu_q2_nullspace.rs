//! Q2 nullspace gate verification.
//!
//! Acceptance is the σ₅/σ₁ bucket retain/reject table and Sampson quality of
//! rescued samples — not the raw lost-solution count. Model and diagnostic
//! signatures are expected to change; they are recorded as the new baseline.
//!
//! Not RANSAC, not the 960-image pipeline, no CPU runtime fallback.
#[allow(dead_code)]
#[path = "five_point_gpu_capacity_replay.rs"]
mod shared;

use anyhow::{ensure, Context, Result};
use clap::Parser;
use nalgebra::{DMatrix, Matrix3, Vector3};
use rayon::{prelude::*, ThreadPoolBuilder};
use rustsfm::{
    database::ColmapDatabase,
    gpu::{FivePointStatus, FivePointTrialResult, WgpuContext, WgpuFivePointF32},
    types::CameraModel,
};
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf};

const INPUT: &str = "af07459d637b5b8c2b20c76d3ef103f058505fe0e8307578ee7e8750f3e230c5";
/// Q1 model signature — must change under Q2 (numerical gate change).
const MODEL_SIGNATURE_Q1: &str = "030f43068d9e8f78961142915162f20bef4ecb909d49d247a4f766a2ab5412df";
/// Q1 diagnostic signature v2 — must change under Q2.
const DIAGNOSTIC_SIGNATURE_V2: &str =
    "3cd8091d6033f2fdb0b598c27483e67c4dd3ee1af1232ff4f1f5ee1da86117cb";
const CPU_SIGNATURE: &str = "9e6764b41602d0005710187d2558d046f1a843c5f873b9542dd7b4f6c29c563a";
const MAX_ERROR_PX: f64 = 4.0;
const TRIALS_PER_PAIR: usize = 512;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    database: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

fn constraint(left: &[Vector3<f64>; 5], right: &[Vector3<f64>; 5]) -> DMatrix<f64> {
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

fn sigma5_over_sigma1(a: &DMatrix<f64>) -> f64 {
    let mut s: Vec<f64> = a
        .clone()
        .svd(false, false)
        .singular_values
        .iter()
        .copied()
        .collect();
    s.sort_by(|x, y| y.total_cmp(x));
    if s[0] == 0.0 {
        0.0
    } else {
        s[4] / s[0]
    }
}

fn ratio_bucket(r: f64) -> &'static str {
    if r.is_nan() || r <= 0.0 {
        "<=0"
    } else if r >= 1e-1 {
        ">=1e-1"
    } else if r >= 1e-2 {
        "[1e-2,1e-1)"
    } else if r >= 1e-3 {
        "[1e-3,1e-2)"
    } else if r >= 1e-4 {
        "[1e-4,1e-3)"
    } else if r >= 1e-5 {
        "[1e-5,1e-4)"
    } else if r >= 1e-6 {
        "[1e-6,1e-5)"
    } else if r > 1e-7 {
        "(1e-7,1e-6]"
    } else {
        "<=1e-7"
    }
}

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

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted[((sorted.len() - 1) as f64 * fraction).round() as usize]
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

    let context = WgpuContext::try_new()?;
    let device = format!("{:?}", context.capabilities());
    let gpu = WgpuFivePointF32::from_context(context)?;
    let gpu_results = shared::gpu_replay(&gpu, &inputs, inputs.len())?;
    let model_sig = shared::model_signature(&gpu_results);
    let diagnostic_sig = shared::signature(&shared::gpu_bits(&gpu_results));
    ensure!(
        model_sig != MODEL_SIGNATURE_Q1,
        "model signature unexpectedly unchanged under Q2"
    );
    ensure!(
        diagnostic_sig != DIAGNOSTIC_SIGNATURE_V2,
        "diagnostic signature unexpectedly unchanged under Q2"
    );

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

    struct Row {
        ratio: f64,
        cpu_best: usize,
        gpu_best: usize,
    }
    let rows: Vec<Row> = pool.install(|| {
        (0..inputs.len())
            .into_par_iter()
            .map(|i| {
                let pair = i / TRIALS_PER_PAIR;
                let ratio = sigma5_over_sigma1(&constraint(&inputs[i].left, &inputs[i].right));
                let obs = &observations[pair];
                let t = thresholds[pair];
                Row {
                    ratio,
                    cpu_best: best_inliers(obs, cpu_results[i].iter().copied(), t),
                    gpu_best: best_inliers(obs, gpu_models(&gpu_results[i]), t),
                }
            })
            .collect()
    });

    let pair_cpu_best: Vec<usize> = (0..pairs.len())
        .map(|p| {
            rows[p * TRIALS_PER_PAIR..(p + 1) * TRIALS_PER_PAIR]
                .iter()
                .map(|r| r.cpu_best)
                .max()
                .unwrap_or(0)
        })
        .collect();

    #[derive(Default)]
    struct Bucket {
        total: usize,
        retained: usize,
        rank_deficient: usize,
        with_models: usize,
        rescued_models: usize,
    }
    let mut buckets: BTreeMap<&str, Bucket> = BTreeMap::new();
    let mut status_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut empty_gpu = 0usize;
    let mut cpu_has_gpu_empty = 0usize;
    let mut gpu_models_total = 0usize;
    let mut cpu_models_total = 0usize;
    let mut rescue_quality = Vec::new();
    let mut retained_true_degenerate = 0usize;
    let mut rejected_rescue_band = 0usize;
    let mut retained_rescue_band = 0usize;

    for (i, row) in rows.iter().enumerate() {
        let g = &gpu_results[i];
        let bucket = ratio_bucket(row.ratio);
        let entry = buckets.entry(bucket).or_default();
        entry.total += 1;
        let rd = g.upstream_status == FivePointStatus::RankDeficient;
        if rd {
            entry.rank_deficient += 1;
        } else {
            entry.retained += 1;
        }
        if g.model_count > 0 {
            entry.with_models += 1;
        }
        gpu_models_total += g.model_count;
        cpu_models_total += cpu_results[i].len();
        if g.model_count == 0 {
            empty_gpu += 1;
        }
        if !cpu_results[i].is_empty() && g.model_count == 0 {
            cpu_has_gpu_empty += 1;
        }
        *status_counts
            .entry(format!("{:?}", g.upstream_status))
            .or_default() += 1;

        let in_rescue = row.ratio >= 1e-5 && row.ratio < 1e-2;
        let true_deg = row.ratio > 0.0 && row.ratio <= 1e-7;
        if in_rescue {
            if rd {
                rejected_rescue_band += 1;
            } else {
                retained_rescue_band += 1;
                if g.model_count > 0 {
                    entry.rescued_models += 1;
                    let pair = i / TRIALS_PER_PAIR;
                    let denom = pair_cpu_best[pair].max(1) as f64;
                    rescue_quality.push(row.gpu_best as f64 / denom);
                }
            }
        }
        if true_deg && !rd {
            retained_true_degenerate += 1;
        }
    }

    rescue_quality.sort_by(|a, b| a.total_cmp(b));
    let bucket_table: BTreeMap<&str, serde_json::Value> = buckets
        .iter()
        .map(|(k, v)| {
            (
                *k,
                json!({
                    "total": v.total,
                    "retained": v.retained,
                    "rank_deficient": v.rank_deficient,
                    "with_models": v.with_models,
                    "retain_rate": if v.total == 0 { 0.0 } else { v.retained as f64 / v.total as f64 },
                }),
            )
        })
        .collect();

    let report = json!({
        "round": "Q2-nullspace",
        "device": device,
        "input_digest": INPUT,
        "model_signature_q1": MODEL_SIGNATURE_Q1,
        "model_signature_q2": model_sig,
        "diagnostic_signature_v2_q1": DIAGNOSTIC_SIGNATURE_V2,
        "diagnostic_signature_q2": diagnostic_sig,
        "cpu_signature": cpu_signature,
        "counts": {
            "trials": inputs.len(),
            "cpu_models": cpu_models_total,
            "gpu_models": gpu_models_total,
            "gpu_empty_trials": empty_gpu,
            "cpu_has_gpu_empty": cpu_has_gpu_empty,
            "upstream_status": status_counts,
        },
        "acceptance": {
            "criterion": "accept iff σ5/σ1 > 1e-5 from Jacobi on AAᵀ (5×5); basis still from AᵀA",
            "rescue_band": "[1e-5, 1e-2)",
            "rescue_band_retained": retained_rescue_band,
            "rescue_band_rejected": rejected_rescue_band,
            "true_degenerate_retained": retained_true_degenerate,
            "rescued_with_models_quality": {
                "n": rescue_quality.len(),
                "p50": percentile(&rescue_quality, 0.5),
                "mean": if rescue_quality.is_empty() {
                    f64::NAN
                } else {
                    rescue_quality.iter().sum::<f64>() / rescue_quality.len() as f64
                },
            },
        },
        "sigma5_over_sigma1_buckets": bucket_table,
    });

    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
