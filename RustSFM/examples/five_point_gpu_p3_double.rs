//! P3 Candidate B: bounded double-buffer on top of merge submit/wait.
//!
//! Compare wall time of `solve_essential_batches` (2 slots) against P3-A
//! merge-only sequential session solves. Gate on Q4 signatures.
//!
//! Retain only if wall time beats P3-A; fewer overlapping waits alone is not enough.
#[allow(dead_code)]
#[path = "five_point_gpu_capacity_replay.rs"]
mod shared;

use anyhow::{ensure, Result};
use clap::Parser;
use rustsfm::gpu::{WgpuContext, WgpuFivePointF32};
use serde_json::json;
use std::{path::PathBuf, time::Instant};

const INPUT: &str = "af07459d637b5b8c2b20c76d3ef103f058505fe0e8307578ee7e8750f3e230c5";
const MODEL_SIGNATURE_Q5b: &str = "fb5be154bb0fa742ab1b83e5c9e62269722678cab463bfbe3827457b7424e8f0";
const DIAGNOSTIC_SIGNATURE_Q5b: &str =
    "1412c6d5d2af9bb9ed23c8d55238af22deb2b259cfa16fa43a99fb21486b9a1a";
/// P3-A steady median for 127×512 (seconds), from p3-submit-merge-20260914.json.
const P3A_STEADY_MEDIAN: f64 = 1.344265917;
const BATCH: usize = 512;
const CALLS: usize = 127;
const WARMUP: usize = 2;
const MEASURE: usize = 5;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    database: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.total_cmp(b));
    xs[xs.len() / 2]
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.output.exists(), "refusing to overwrite experiment");
    let (inputs, _, _, _) = shared::cpu::load_with_observations(&shared::cpu::Args {
        database: args.database.clone(),
        pairs: 127,
        trials: 512,
        batch_size: 512,
    })?;
    ensure!(
        inputs.len() == 65024 && shared::cpu::input_digest(&inputs) == INPUT,
        "input provenance changed"
    );

    let context = WgpuContext::try_new()?;
    let device = format!("{:?}", context.capabilities());
    let gpu = WgpuFivePointF32::from_context(context)?;

    let chunks: Vec<_> = inputs.chunks(BATCH).collect();
    ensure!(chunks.len() == CALLS);
    let batches: Vec<(Vec<[f32; 3]>, Vec<[f32; 3]>)> = chunks
        .iter()
        .map(|chunk| {
            let left: Vec<[f32; 3]> = chunk
                .iter()
                .flat_map(|i| i.left.iter().map(|r| [r.x as f32, r.y as f32, r.z as f32]))
                .collect();
            let right: Vec<[f32; 3]> = chunk
                .iter()
                .flat_map(|i| i.right.iter().map(|r| [r.x as f32, r.y as f32, r.z as f32]))
                .collect();
            (left, right)
        })
        .collect();

    // Signature gate: merge path and double-buffer path both match Q4.
    let mut merge = gpu.session(BATCH)?;
    let mut merge_results = Vec::with_capacity(inputs.len());
    merge.reset_sync_counters();
    for (chunk_idx, (left, right)) in batches.iter().enumerate() {
        let mut part = merge.solve_essential(left, right)?;
        for r in &mut part {
            r.trial += chunk_idx * BATCH;
        }
        merge_results.extend(part);
    }
    ensure!(
        shared::model_signature(&merge_results) == MODEL_SIGNATURE_Q5b,
        "merge model signature mismatch"
    );
    ensure!(
        shared::signature(&shared::gpu_bits(&merge_results)) == DIAGNOSTIC_SIGNATURE_Q5b,
        "merge diagnostic signature mismatch"
    );
    ensure!(merge.submit_count() == CALLS as u64);
    ensure!(merge.wait_count() == CALLS as u64);

    let mut dbl = gpu.session(BATCH)?;
    dbl.enable_double_buffer()?;
    dbl.reset_sync_counters();
    let dbl_results = dbl.solve_essential_batches(&batches)?;
    ensure!(
        shared::model_signature(&dbl_results) == MODEL_SIGNATURE_Q5b,
        "double-buffer model signature mismatch"
    );
    ensure!(
        shared::signature(&shared::gpu_bits(&dbl_results)) == DIAGNOSTIC_SIGNATURE_Q5b,
        "double-buffer diagnostic signature mismatch"
    );
    ensure!(dbl.submit_count() == CALLS as u64);
    ensure!(dbl.wait_count() == CALLS as u64);

    let time_merge = |session: &mut rustsfm::gpu::FivePointSession<'_>| -> Result<f64> {
        let t0 = Instant::now();
        for (left, right) in &batches {
            let _ = session.solve_essential(left, right)?;
        }
        Ok(t0.elapsed().as_secs_f64())
    };
    let time_dbl = |session: &mut rustsfm::gpu::FivePointSession<'_>| -> Result<f64> {
        let t0 = Instant::now();
        let _ = session.solve_essential_batches(&batches)?;
        Ok(t0.elapsed().as_secs_f64())
    };

    let mut merge_sess = gpu.session(BATCH)?;
    for _ in 0..WARMUP {
        let _ = time_merge(&mut merge_sess)?;
    }
    let mut merge_samples = Vec::new();
    for _ in 0..MEASURE {
        merge_samples.push(time_merge(&mut merge_sess)?);
    }
    let merge_median = median(&mut merge_samples.clone());

    let mut dbl_sess = gpu.session(BATCH)?;
    dbl_sess.enable_double_buffer()?;
    for _ in 0..WARMUP {
        let _ = time_dbl(&mut dbl_sess)?;
    }
    dbl_sess.reset_sync_counters();
    let mut dbl_samples = Vec::new();
    for _ in 0..MEASURE {
        dbl_samples.push(time_dbl(&mut dbl_sess)?);
    }
    let submits = dbl_sess.submit_count();
    let waits = dbl_sess.wait_count();
    ensure!(submits == (MEASURE * CALLS) as u64);
    ensure!(waits == (MEASURE * CALLS) as u64);
    let dbl_median = median(&mut dbl_samples.clone());

    let vs_p3a = (P3A_STEADY_MEDIAN - dbl_median) / P3A_STEADY_MEDIAN;
    let vs_merge_same_run = (merge_median - dbl_median) / merge_median;
    // Stop rule: retain only if wall beats P3-A merge baseline.
    let retain = dbl_median < P3A_STEADY_MEDIAN;

    let report = json!({
        "round": "P3-double-buffer",
        "candidate": "bounded double-buffer (2 slots) on merge submit/wait",
        "device": device,
        "input_digest": INPUT,
        "model_signature_q5b": MODEL_SIGNATURE_Q5b,
        "diagnostic_signature_q5b": DIAGNOSTIC_SIGNATURE_Q5b,
        "signature_ok": true,
        "sync": {
            "calls_per_pass": CALLS,
            "submits_per_pass": CALLS,
            "waits_per_pass": CALLS,
            "max_inflight": 2,
        },
        "timing_seconds": {
            "batch": BATCH,
            "warmup": WARMUP,
            "measure": MEASURE,
            "p3a_steady_median_baseline": P3A_STEADY_MEDIAN,
            "merge_same_run_median": merge_median,
            "merge_same_run_samples": merge_samples,
            "double_buffer_median": dbl_median,
            "double_buffer_samples": dbl_samples,
            "relative_vs_p3a": vs_p3a,
            "relative_vs_merge_same_run": vs_merge_same_run,
        },
        "decision": if retain { "retain_candidate_b" } else { "rollback_candidate_b" },
        "retain": retain,
    });
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !retain {
        eprintln!(
            "P3 candidate B did not beat P3-A wall time ({dbl_median:.6} vs {P3A_STEADY_MEDIAN:.6}); keep double-buffer opt-in API but do not claim a win."
        );
    }
    Ok(())
}
