//! P2 session verification: signature match vs one-shot Q5b baseline, and
//! first-call vs steady-state timing with batch 512 (127 calls).
//!
//! Not RANSAC, not the 960-image pipeline, no CPU runtime fallback.
#[allow(dead_code)]
#[path = "five_point_gpu_capacity_replay.rs"]
mod shared;

use anyhow::{ensure, Result};
use clap::Parser;
use rayon::ThreadPoolBuilder;
use rustsfm::gpu::{WgpuContext, WgpuFivePointF32};
use serde_json::json;
use std::{path::PathBuf, time::Instant};

const INPUT: &str = "af07459d637b5b8c2b20c76d3ef103f058505fe0e8307578ee7e8750f3e230c5";
const MODEL_SIGNATURE_Q5b: &str = "fb5be154bb0fa742ab1b83e5c9e62269722678cab463bfbe3827457b7424e8f0";
const DIAGNOSTIC_SIGNATURE_Q5b: &str =
    "1412c6d5d2af9bb9ed23c8d55238af22deb2b259cfa16fa43a99fb21486b9a1a";
const CPU_SIGNATURE: &str = "9e6764b41602d0005710187d2558d046f1a843c5f873b9542dd7b4f6c29c563a";
const BATCH: usize = 512;
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

    // One-shot full batch: pins the Q4 arithmetic baseline under the new params
    // binding (capacity == count, so session vs one-shot must agree).
    let oneshot = shared::gpu_replay(&gpu, &inputs, inputs.len())?;
    let model_sig = shared::model_signature(&oneshot);
    let diagnostic_sig = shared::signature(&shared::gpu_bits(&oneshot));
    ensure!(
        model_sig == MODEL_SIGNATURE_Q5b,
        "one-shot model signature drifted: {model_sig}"
    );
    ensure!(
        diagnostic_sig == DIAGNOSTIC_SIGNATURE_Q5b,
        "one-shot diagnostic signature drifted: {diagnostic_sig}"
    );

    // Session: 127 × 512 with grow-once reuse; must match one-shot bit-for-bit.
    let mut session = gpu.session(BATCH)?;
    let mut session_results = Vec::with_capacity(inputs.len());
    for (chunk_idx, chunk) in inputs.chunks(BATCH).enumerate() {
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
    ensure!(session.capacity() == BATCH, "batch-512 session should not grow");

    // small→large→small signature check on a prefix (stale-state gate).
    let mut flex = gpu.session(1)?;
    let prefix_small = &inputs[..64];
    let prefix_large = &inputs[..2048];
    let run = |session: &mut rustsfm::gpu::FivePointSession<'_>,
               slice: &[shared::cpu::Input]|
     -> Result<String> {
        let left: Vec<[f32; 3]> = slice
            .iter()
            .flat_map(|i| i.left.iter().map(|r| [r.x as f32, r.y as f32, r.z as f32]))
            .collect();
        let right: Vec<[f32; 3]> = slice
            .iter()
            .flat_map(|i| i.right.iter().map(|r| [r.x as f32, r.y as f32, r.z as f32]))
            .collect();
        Ok(shared::model_signature(&session.solve_essential(&left, &right)?))
    };
    let small_a = run(&mut flex, prefix_small)?;
    let _large = run(&mut flex, prefix_large)?;
    let small_b = run(&mut flex, prefix_small)?;
    ensure!(
        small_a == small_b,
        "small→large→small model signature drifted"
    );
    ensure!(flex.capacity() >= 2048);

    // Timing: session construction, steady reuse, and one-shot allocation/call.
    let chunks: Vec<_> = inputs.chunks(BATCH).collect();
    ensure!(chunks.len() == 127);

    let time_session_pass = |session: &mut rustsfm::gpu::FivePointSession<'_>| -> Result<f64> {
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
    let time_oneshot_pass = || -> Result<f64> {
        let t0 = Instant::now();
        let _ = shared::gpu_replay(&gpu, &inputs, BATCH)?;
        Ok(t0.elapsed().as_secs_f64())
    };

    let mut create_samples = Vec::new();
    for _ in 0..MEASURE {
        let t0 = Instant::now();
        let _ = gpu.session(BATCH)?;
        create_samples.push(t0.elapsed().as_secs_f64());
    }

    let mut steady = gpu.session(BATCH)?;
    for _ in 0..WARMUP {
        let _ = time_session_pass(&mut steady)?;
    }
    let mut steady_samples = Vec::new();
    for _ in 0..MEASURE {
        steady_samples.push(time_session_pass(&mut steady)?);
    }

    let mut oneshot_samples = Vec::new();
    for _ in 0..WARMUP {
        let _ = time_oneshot_pass()?;
    }
    for _ in 0..MEASURE {
        oneshot_samples.push(time_oneshot_pass()?);
    }

    // Cold: create + first full 127-call pass (alloc + first use).
    let mut cold_samples = Vec::new();
    for _ in 0..MEASURE {
        let t0 = Instant::now();
        let mut cold = gpu.session(BATCH)?;
        let _ = time_session_pass(&mut cold)?;
        cold_samples.push(t0.elapsed().as_secs_f64());
    }

    let pool = ThreadPoolBuilder::new().num_threads(8).build()?;
    let cpu = shared::cpu::replay_batches(&inputs, Some(&pool), inputs.len());
    let mut hasher = blake3::Hasher::new();
    for models in &shared::cpu::output_bits(&cpu) {
        hasher.update(&(models.len() as u64).to_le_bytes());
        for m in models {
            for w in m {
                hasher.update(&w.to_le_bytes());
            }
        }
    }
    let cpu_signature = hasher.finalize().to_hex().to_string();
    ensure!(cpu_signature == CPU_SIGNATURE, "CPU signature changed");

    let report = json!({
        "round": "P2-session",
        "device": device,
        "input_digest": INPUT,
        "model_signature_q5b": MODEL_SIGNATURE_Q5b,
        "diagnostic_signature_q5b": DIAGNOSTIC_SIGNATURE_Q5b,
        "cpu_signature": cpu_signature,
        "session_capacity_after_512": BATCH,
        "flex_capacity_after_2048": flex.capacity(),
        "small_large_small_model_ok": true,
        "timing_seconds": {
            "batch": BATCH,
            "calls_per_pass": 127,
            "warmup": WARMUP,
            "measure": MEASURE,
            "session_create_512_median": median(&mut create_samples),
            "session_create_512_samples": create_samples,
            "session_cold_create_and_pass_median": median(&mut cold_samples),
            "session_cold_create_and_pass_samples": cold_samples,
            "session_steady_median": median(&mut steady_samples),
            "session_steady_samples": steady_samples,
            "oneshot_median": median(&mut oneshot_samples),
            "oneshot_samples": oneshot_samples,
        },
    });
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
