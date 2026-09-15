//! P1 layout evidence only; reuses the CPU-thread experiment's exact input loader.
#[allow(dead_code)]
#[path = "five_point_gpu_capacity_replay.rs"]
mod shared;
use anyhow::{ensure, Result};
use clap::Parser;
use rustsfm::gpu::{WgpuContext, WgpuFivePointF32, FIVE_POINT_PASS_NAMES};
use serde_json::json;
use std::{path::PathBuf, time::Instant};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    database: PathBuf,
    #[arg(long)]
    output: PathBuf,
}
fn rays(inputs: &[shared::cpu::Input]) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let convert = |left| {
        inputs
            .iter()
            .flat_map(move |i| {
                (if left { &i.left } else { &i.right })
                    .iter()
                    .map(|r| [r.x as f32, r.y as f32, r.z as f32])
            })
            .collect()
    };
    (convert(true), convert(false))
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.output.exists(), "refusing overwrite");
    let (inputs, _, _, observations) = shared::cpu::load_with_observations(&shared::cpu::Args {
        database: args.database,
        pairs: 127,
        trials: 512,
        batch_size: 512,
    })?;
    drop(observations);
    let digest = shared::cpu::input_digest(&inputs);
    ensure!(
        inputs.len() == 65024
            && digest == "af07459d637b5b8c2b20c76d3ef103f058505fe0e8307578ee7e8750f3e230c5",
        "input changed"
    );
    let context = WgpuContext::try_new()?;
    ensure!(
        !context.timestamp_queries_enabled(),
        "unprofiled context required"
    );
    let device = format!("{:?}", context.capabilities());
    let gpu = WgpuFivePointF32::from_context(context)?;
    let reference = shared::gpu_replay(&gpu, &inputs, inputs.len())?;
    ensure!(reference.len() == inputs.len(), "missing trials");
    let bits = shared::gpu_bits(&reference);
    let signature = shared::signature(&bits);
    ensure!(
        signature == "1f6f3f8d88a17e4e1c42abfc85aad44a66c4b864091805594e086ca80bf9b339",
        "historical signature changed: {signature}"
    );
    let mut boundaries = Vec::new();
    for n in [1, 31, 32, 33, 63, 65, 65023] {
        let result = shared::gpu_replay(&gpu, &inputs[..n], n)?;
        ensure!(
            result.len() == n && shared::gpu_bits(&result) == bits[..n],
            "full boundary {n}"
        );
        let (left, right) = rays(&inputs[..n]);
        let matrices = gpu.compute_constraint_matrices(&left, &right)?;
        let diagnostics = gpu.compute_nullspace_diagnostics(&matrices)?;
        let mut hash = blake3::Hasher::new();
        let mut bases = Vec::new();
        for d in &diagnostics {
            hash.update(&[d.status as u8]);
            for word in [d.sweeps, d.min_diagonal_ratio.to_bits(), d.rank] {
                hash.update(&word.to_le_bytes());
            }
            if let Some(b) = d.basis {
                for x in b {
                    hash.update(&x.to_bits().to_le_bytes());
                }
                bases.push(b);
            } else {
                bases.push([0.0; 36]);
            }
        }
        ensure!(diagnostics.len() == n, "diagnostic length");
        let algebra = gpu.compute_algebra(&bases)?;
        ensure!(algebra.len() == n, "algebra length");
        for a in algebra {
            hash.update(&[a.status as u8]);
            for x in a
                .elimination
                .iter()
                .chain(&a.solved)
                .chain(&a.determinant)
                .chain(&a.coefficients)
                .chain(std::iter::once(&a.minimum_relative_pivot))
            {
                hash.update(&x.to_bits().to_le_bytes());
            }
        }
        boundaries.push(json!({"count":n,"diagnostic_signature":hash.finalize().to_hex().to_string(),"full_signature":shared::signature(&shared::gpu_bits(&result))}));
    }
    let mut endtoend_ms = Vec::new();
    for iteration in 0..6 {
        let started = Instant::now();
        let result = shared::gpu_replay(&gpu, &inputs, inputs.len())?;
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        ensure!(
            result.len() == inputs.len()
                && shared::signature(&shared::gpu_bits(&result)) == signature,
            "timed signature"
        );
        if iteration >= 3 {
            endtoend_ms.push(ms);
        }
    }
    let context = WgpuContext::try_new_experimental_timestamps()?;
    ensure!(
        context.timestamp_queries_enabled(),
        "actual GPU timestamps required"
    );
    let profiled = WgpuFivePointF32::from_context(context)?;
    let (left, right) = rays(&inputs);
    let mut profiles = Vec::new();
    for iteration in 0..6 {
        let (result, profile) = profiled.solve_essential_profiled(&left, &right)?;
        ensure!(
            result.len() == inputs.len()
                && shared::signature(&shared::gpu_bits(&result)) == signature,
            "profile signature"
        );
        if iteration >= 3 {
            profiles.push(json!({"seconds":profile.gpu_pass_seconds,"ticks":profile.timestamp_ticks,"period_ns":profile.timestamp_period_ns}));
        }
    }
    let report = json!({"device":device,"input_digest":digest,"trials":inputs.len(),"signature":signature,"boundaries":boundaries,"endtoend_ms":endtoend_ms,"pass_names":FIVE_POINT_PASS_NAMES,"profiles":profiles});
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    eprintln!(
        "signature={signature} endtoend_ms={endtoend_ms:?}; wrote {}",
        args.output.display()
    );
    Ok(())
}
