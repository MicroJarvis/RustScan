//! P3 submit/readback boundary verification.
//!
//! Candidate A: merge compute + staging copy into one submit/wait per session
//! solve. Gate on Q4 signatures and compare wall time + submit/wait counts
//! against the P2 baseline (2 syncs per call → 1).
//!
//! Not RANSAC, not the 960-image pipeline, no CPU runtime fallback.
#[allow(dead_code)]
#[path = "five_point_gpu_capacity_replay.rs"]
mod shared;

use anyhow::{ensure, Result};
use clap::Parser;
use rustscan_sfm::gpu::{WgpuContext, WgpuFivePointF32};
use serde_json::json;
use std::{path::PathBuf, time::Instant};

const INPUT: &str = "af07459d637b5b8c2b20c76d3ef103f058505fe0e8307578ee7e8750f3e230c5";
const MODEL_SIGNATURE_Q5b: &str =
    "fb5be154bb0fa742ab1b83e5c9e62269722678cab463bfbe3827457b7424e8f0";
const DIAGNOSTIC_SIGNATURE_Q5b: &str =
    "1412c6d5d2af9bb9ed23c8d55238af22deb2b259cfa16fa43a99fb21486b9a1a";
/// P2 steady median for 127×512 (seconds), from p2-session-final-20260914.json.
const P2_STEADY_MEDIAN: f64 = 1.774772709;
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

    let oneshot = shared::gpu_replay(&gpu, &inputs, inputs.len())?;
    ensure!(
        shared::model_signature(&oneshot) == MODEL_SIGNATURE_Q5b,
        "one-shot model signature drifted"
    );
    ensure!(
        shared::signature(&shared::gpu_bits(&oneshot)) == DIAGNOSTIC_SIGNATURE_Q5b,
        "one-shot diagnostic signature drifted"
    );

    let chunks: Vec<_> = inputs.chunks(BATCH).collect();
    ensure!(chunks.len() == CALLS);

    let mut session = gpu.session(BATCH)?;
    let mut session_results = Vec::with_capacity(inputs.len());
    session.reset_sync_counters();
    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        let left: Vec<[f32; 3]> = chunk
            .iter()
            .flat_map(|i| i.left.iter().map(|r| [r.x as f32, r.y as f32, r.z as f32]))
            .collect();
        let right: Vec<[f32; 3]> = chunk
            .iter()
            .flat_map(|i| i.right.iter().map(|r| [r.x as f32, r.y as f32, r.z as f32]))
            .collect();
        let mut part = session.solve_essential(&left, &right)?;
        for r in &mut part {
            r.trial += chunk_idx * BATCH;
        }
        session_results.extend(part);
    }
    ensure!(
        shared::model_signature(&session_results) == MODEL_SIGNATURE_Q5b,
        "session model signature mismatch"
    );
    ensure!(
        shared::signature(&shared::gpu_bits(&session_results)) == DIAGNOSTIC_SIGNATURE_Q5b,
        "session diagnostic signature mismatch"
    );
    ensure!(
        session.submit_count() == CALLS as u64,
        "expected {CALLS} submits, got {}",
        session.submit_count()
    );
    ensure!(
        session.wait_count() == CALLS as u64,
        "expected {CALLS} waits, got {}",
        session.wait_count()
    );

    let time_pass = |session: &mut rustscan_sfm::gpu::FivePointSession<'_>| -> Result<f64> {
        let t0 = Instant::now();
        for chunk in &chunks {
            let left: Vec<[f32; 3]> = chunk
                .iter()
                .flat_map(|i| i.left.iter().map(|r| [r.x as f32, r.y as f32, r.z as f32]))
                .collect();
            let right: Vec<[f32; 3]> = chunk
                .iter()
                .flat_map(|i| i.right.iter().map(|r| [r.x as f32, r.y as f32, r.z as f32]))
                .collect();
            let _ = session.solve_essential(&left, &right)?;
        }
        Ok(t0.elapsed().as_secs_f64())
    };

    let mut steady = gpu.session(BATCH)?;
    for _ in 0..WARMUP {
        let _ = time_pass(&mut steady)?;
    }
    steady.reset_sync_counters();
    let mut steady_samples = Vec::new();
    for _ in 0..MEASURE {
        steady_samples.push(time_pass(&mut steady)?);
    }
    let submits = steady.submit_count();
    let waits = steady.wait_count();
    ensure!(submits == (MEASURE * CALLS) as u64);
    ensure!(waits == (MEASURE * CALLS) as u64);

    let steady_median = median(&mut steady_samples);
    let vs_p2 = (P2_STEADY_MEDIAN - steady_median) / P2_STEADY_MEDIAN;
    // Stop rule: retain only if wall improves; call-count alone is not enough.
    let retain = steady_median < P2_STEADY_MEDIAN;

    let report = json!({
        "round": "P3-submit-merge",
        "candidate": "merge compute+copy into one submit/wait per solve",
        "device": device,
        "input_digest": INPUT,
        "model_signature_q5b": MODEL_SIGNATURE_Q5b,
        "diagnostic_signature_q5b": DIAGNOSTIC_SIGNATURE_Q5b,
        "signature_ok": true,
        "sync": {
            "calls_per_pass": CALLS,
            "submits_per_pass": CALLS,
            "waits_per_pass": CALLS,
            "p2_submits_per_pass": CALLS * 2,
            "p2_waits_per_pass": CALLS * 2,
            "submit_reduction": 0.5,
        },
        "timing_seconds": {
            "batch": BATCH,
            "warmup": WARMUP,
            "measure": MEASURE,
            "p2_steady_median_baseline": P2_STEADY_MEDIAN,
            "session_steady_median": steady_median,
            "session_steady_samples": steady_samples,
            "relative_vs_p2": vs_p2,
        },
        "decision": if retain { "retain_candidate_a" } else { "rollback_candidate_a" },
        "retain": retain,
        "double_buffer_next": retain,
    });
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !retain {
        eprintln!(
            "P3 candidate A did not beat P2 wall time ({steady_median:.6} vs {P2_STEADY_MEDIAN:.6}); per TODO stop rule, do not claim a win from submit count alone."
        );
    }
    Ok(())
}
