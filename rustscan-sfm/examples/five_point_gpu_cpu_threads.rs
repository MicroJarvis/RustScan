//! Fixed-workset throughput only: GPU f32 is not numerically equivalent to CPU f64.
#[allow(dead_code)]
#[path = "five_point_gpu_capacity_replay.rs"]
mod shared;

use anyhow::{ensure, Result};
use clap::Parser;
use rayon::ThreadPoolBuilder;
use rustscan_sfm::gpu::{WgpuContext, WgpuFivePointF32};
use serde_json::json;
use shared::cpu;
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

const INPUT: &str = "af07459d637b5b8c2b20c76d3ef103f058505fe0e8307578ee7e8750f3e230c5";
const GPU_SIGNATURE: &str = "1f6f3f8d88a17e4e1c42abfc85aad44a66c4b864091805594e086ca80bf9b339";

#[derive(Parser)]
struct Args {
    #[arg(long)]
    database: PathBuf,
    #[arg(long, value_parser = clap::value_parser!(u32).range(4..=8))]
    cpu_threads: u32,
    #[arg(long)]
    output: PathBuf,
}

fn order(round: usize) -> [usize; 2] {
    if round % 2 == 0 {
        [0, 1]
    } else {
        [1, 0]
    }
}
fn median(values: &[f64]) -> f64 {
    assert!(!values.is_empty());
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
}
fn stable_win(cpu: &[f64], gpu: &[f64]) -> bool {
    assert_eq!(cpu.len(), gpu.len());
    !cpu.is_empty() && cpu.iter().zip(gpu).all(|(c, g)| g < c) && median(gpu) <= 0.95 * median(cpu)
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        matches!(args.cpu_threads, 4 | 8),
        "choose four or eight threads"
    );
    ensure!(!args.output.exists(), "refusing to overwrite experiment");
    let (inputs, pairs, _, observations) = cpu::load_with_observations(&cpu::Args {
        database: args.database.clone(),
        pairs: 127,
        trials: 512,
        batch_size: 512,
    })?;
    drop(observations);
    ensure!(
        inputs.len() == 65024 && cpu::input_digest(&inputs) == INPUT,
        "input provenance changed"
    );
    let init = Instant::now();
    let context = WgpuContext::try_new()?;
    let device = format!("{:?}", context.capabilities());
    ensure!(
        !context.timestamp_queries_enabled(),
        "profiling must be disabled"
    );
    let gpu = WgpuFivePointF32::from_context(context)?;
    let initialization_seconds = init.elapsed().as_secs_f64();
    let pool = ThreadPoolBuilder::new()
        .num_threads(args.cpu_threads as usize)
        .build()?;
    ensure!(
        pool.current_num_threads() == args.cpu_threads as usize,
        "pool size mismatch"
    );
    // Offline reference and quality counts, never used by the GPU execution path.
    let reference = cpu::replay_batches(&inputs, None, inputs.len());
    let reference_bits = cpu::output_bits(&reference);
    let mut cpu_hash = blake3::Hasher::new();
    for models in &reference_bits {
        cpu_hash.update(&(models.len() as u64).to_le_bytes());
        for model in models {
            for word in model {
                cpu_hash.update(&word.to_le_bytes());
            }
        }
    }
    let quality_gpu = shared::gpu_replay(&gpu, &inputs, inputs.len())?;
    ensure!(quality_gpu.len() == inputs.len(), "missing GPU trials");
    ensure!(
        shared::signature(&shared::gpu_bits(&quality_gpu)) == GPU_SIGNATURE,
        "GPU reference signature changed"
    );
    let quality = json!({
        "cpu_models":reference.iter().map(Vec::len).sum::<usize>(),
        "gpu_models":quality_gpu.iter().map(|r|r.model_count).sum::<usize>(),
        "gpu_empty_trials":quality_gpu.iter().filter(|r|r.model_count==0).count(),
        "count_mismatch_trials":reference.iter().zip(&quality_gpu).filter(|(c,g)|c.len()!=g.model_count).count(),
        "cpu_nonempty_gpu_empty_trials":reference.iter().zip(&quality_gpu).filter(|(c,g)|!c.is_empty()&&g.model_count==0).count(),
        "equal_quality":false
    });
    drop(quality_gpu);
    drop(reference);
    let mut reports = Vec::new();
    for batch in [32768, 65024] {
        let mut times = [Vec::new(), Vec::new()];
        let mut warmups = Vec::new();
        let mut rounds = Vec::new();
        for iteration in 0..11 {
            let warmup = iteration < 3;
            let round = if warmup { iteration } else { iteration - 3 };
            let execution_order = order(round);
            let mut seconds = [0.0; 2];
            for mode in execution_order {
                // Equal idle interval, outside timers, to limit recently active Rayon
                // workers affecting only the CPU-then-GPU order. Not an overlap test.
                std::thread::sleep(Duration::from_millis(100));
                let started = Instant::now();
                if mode == 0 {
                    let result = cpu::replay_batches(&inputs, Some(&pool), batch);
                    seconds[mode] = started.elapsed().as_secs_f64();
                    ensure!(
                        cpu::output_bits(&result) == reference_bits,
                        "CPU order/coefficients changed"
                    );
                } else {
                    let result = shared::gpu_replay(&gpu, &inputs, batch)?;
                    seconds[mode] = started.elapsed().as_secs_f64();
                    ensure!(result.len() == inputs.len(), "missing GPU trials");
                    ensure!(
                        shared::signature(&shared::gpu_bits(&result)) == GPU_SIGNATURE,
                        "GPU trial/slot signature changed"
                    );
                }
            }
            let record = json!({"round":round,"order":execution_order,"cpu_seconds":seconds[0],"gpu_seconds":seconds[1]});
            if warmup {
                warmups.push(record);
            } else {
                for mode in 0..2 {
                    times[mode].push(seconds[mode]);
                }
                eprintln!(
                    "threads={} batch={batch} round={round} cpu={:.3}ms gpu={:.3}ms",
                    args.cpu_threads,
                    seconds[0] * 1000.0,
                    seconds[1] * 1000.0
                );
                rounds.push(record);
            }
        }
        reports.push(json!({"batch":batch,"gpu_calls":inputs.len().div_ceil(batch),
            "warmups":warmups,"rounds":rounds,"cpu_median_ms":median(&times[0])*1000.0,
            "gpu_median_ms":median(&times[1])*1000.0,"cpu_over_gpu_ratio":median(&times[0])/median(&times[1]),
            "gpu_wins":times[0].iter().zip(&times[1]).filter(|(c,g)|g<c).count(),
            "gpu_stable_win":stable_win(&times[0],&times[1])}));
    }
    let report = json!({"schema":"five-point-cpu-thread-comparison-v1","source_commit":"3ab9c09",
        "database":args.database,"device":device,"cpu_threads":args.cpu_threads,
        "available_parallelism":std::thread::available_parallelism()?.get(),
        "input_digest":INPUT,"trials":inputs.len(),"pairs":pairs.len(),"unique_images":shared::unique_images(&pairs).len(),
        "gpu_signature":GPU_SIGNATURE,"cpu_signature":cpu_hash.finalize().to_hex().to_string(),
        "initialization_seconds":initialization_seconds,"quality":quality,"batches":reports,
        "warmup_rounds":3,"measured_rounds":8,"idle_before_each_path_ms":100,
        "stable_win_rule":"GPU faster in all eight paired rounds AND median GPU time at least 5% lower; local evidence, not statistical guarantee",
        "timing_contract":"CPU f64 solver and ordered collection, prebuilt dedicated Rayon pool. GPU includes f64-to-f32 conversion, buffers/uploads, all compute, waits, readback, decoding and collection. Excludes initialization, input loading/sampling, signatures/quality checks, 100ms idle, and final result destruction. No timestamps or kernel changes. Paths sequential, pool remains alive. No affinity/thermal control. Eight local rounds, one device. Not full RANSAC, not quality-equivalent throughput, not full 960-image processing."});
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    eprintln!("wrote {}", args.output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn balanced_orders_and_even_median() {
        assert_eq!((0..8).filter(|&i| order(i)[0] == 0).count(), 4);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    }
    #[test]
    fn win_gate_rejects_noisy_or_tiny_advantage() {
        assert!(stable_win(&[2.0; 8], &[1.0; 8]));
        assert!(!stable_win(&[2.0; 8], &[1.99; 8]));
        assert!(!stable_win(
            &[2.0; 8],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.1]
        ));
        assert!(!stable_win(&[], &[]));
    }
}
