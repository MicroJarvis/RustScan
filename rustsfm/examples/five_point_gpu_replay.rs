//! Independent fixed-prefix comparison, not RANSAC and not a 960-image pipeline.
#[allow(dead_code)]
#[path = "five_point_replay_probe.rs"]
mod cpu;

use anyhow::{ensure, Result};
use clap::Parser;
use nalgebra::Matrix3;
#[cfg(test)]
use nalgebra::Vector3;
use rayon::ThreadPoolBuilder;
use rustsfm::gpu::{FivePointTrialResult, WgpuContext, WgpuFivePointF32};
use serde_json::{json, Value};
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    database: PathBuf,
    #[arg(long,default_value_t=12,value_parser=clap::value_parser!(u32).range(1..=12))]
    pairs: u32,
    #[arg(long,default_value_t=512,value_parser=clap::value_parser!(u32).range(1..=512))]
    trials: u32,
    #[arg(long,default_value_t=3,value_parser=clap::value_parser!(u32).range(3..=8))]
    rounds: u32,
    #[arg(long)]
    output: Option<PathBuf>,
}

const BATCH_SIZES: [usize; 7] = [64, 128, 256, 512, 1024, 2048, 6144];

fn solve_calls(trials: usize, batch: usize) -> usize {
    trials.div_ceil(batch)
}

fn gpu_replay(
    gpu: &WgpuFivePointF32,
    inputs: &[cpu::Input],
    batch: usize,
) -> Result<Vec<FivePointTrialResult>> {
    let mut all = Vec::with_capacity(inputs.len());
    for (chunk, inputs) in inputs.chunks(batch).enumerate() {
        let left: Vec<[f32; 3]> = inputs
            .iter()
            .flat_map(|i| i.left.iter().map(|r| [r.x as f32, r.y as f32, r.z as f32]))
            .collect();
        let right: Vec<[f32; 3]> = inputs
            .iter()
            .flat_map(|i| i.right.iter().map(|r| [r.x as f32, r.y as f32, r.z as f32]))
            .collect();
        let mut results = gpu.solve_essential(&left, &right)?;
        for result in &mut results {
            result.trial += chunk * batch;
        }
        all.extend(results);
    }
    Ok(all)
}

fn gpu_bits(results: &[FivePointTrialResult]) -> Vec<Vec<u32>> {
    results
        .iter()
        .map(|r| {
            let mut words = vec![
                r.trial as u32,
                r.model_count as u32,
                r.unconverged as u32,
                r.root_iterations as u32,
                r.upstream_status as u32,
            ];
            for slot in &r.slots {
                words.extend([
                    slot.slot as u32,
                    slot.status as u32,
                    slot.root[0].to_bits(),
                    slot.root[1].to_bits(),
                ]);
                if let Some(e) = slot.essential {
                    words.extend(e.map(f32::to_bits));
                }
            }
            words
        })
        .collect()
}

fn summary(values: &[f64]) -> Value {
    let mut sorted: Vec<_> = values.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(f64::total_cmp);
    if sorted.is_empty() {
        return json!({"count":0});
    }
    let quantile = |p: f64| sorted[((sorted.len() - 1) as f64 * p).round() as usize];
    json!({"count":sorted.len(),"mean":sorted.iter().sum::<f64>()/sorted.len() as f64,"p50":quantile(0.5),"p90":quantile(0.9),"p99":quantile(0.99),"max":sorted.last()})
}
fn distance(a: &Matrix3<f64>, b: &Matrix3<f64>) -> f64 {
    let a = a.normalize();
    let b = b.normalize();
    (a - b).norm().min((a + b).norm())
}
fn nearest(source: &[Matrix3<f64>], target: &[Matrix3<f64>]) -> Vec<Option<(usize, f64)>> {
    source
        .iter()
        .map(|s| {
            target
                .iter()
                .enumerate()
                .map(|(i, t)| (i, distance(s, t)))
                .min_by(|a, b| a.1.total_cmp(&b.1))
        })
        .collect()
}
fn constraint_residual(e: &Matrix3<f64>, input: &cpu::Input) -> f64 {
    let mut sum = 0.0;
    let mut norm = 0.0;
    for (l, r) in input.left.iter().zip(&input.right) {
        sum += r.dot(&(e * l)).powi(2);
        norm += l.norm_squared() * r.norm_squared();
    }
    (sum / norm).sqrt() / e.norm()
}

// Standard Sampson squared distance on normalized image coordinates (z=1),
// not pixels or angular distance. Same CPU f64 diagnostic for both model sets.
const SAMPSON_SQUARED_THRESHOLD: f64 = 1e-6;
fn mask(e: &Matrix3<f64>, observations: &cpu::PairObservations) -> Vec<u64> {
    let mut mask = vec![0; observations.left.len().div_ceil(64)];
    for (i, (l, r)) in observations
        .left
        .iter()
        .zip(&observations.right)
        .enumerate()
    {
        if l.z.abs() < 1e-12 || r.z.abs() < 1e-12 {
            continue;
        }
        let l = l / l.z;
        let r = r / r.z;
        let el = e * l;
        let etr = e.transpose() * r;
        let denominator = el.x * el.x + el.y * el.y + etr.x * etr.x + etr.y * etr.y;
        if denominator > 1e-30 && r.dot(&el).powi(2) / denominator <= SAMPSON_SQUARED_THRESHOLD {
            mask[i / 64] |= 1u64 << (i % 64);
        }
    }
    mask
}
fn mask_comparison(a: &[u64], b: &[u64], observations: usize) -> Value {
    let mut different = 0u64;
    let mut intersection = 0u64;
    let mut union = 0u64;
    let mut ca = 0u64;
    let mut cb = 0u64;
    for (&a, &b) in a.iter().zip(b) {
        different += (a ^ b).count_ones() as u64;
        intersection += (a & b).count_ones() as u64;
        union += (a | b).count_ones() as u64;
        ca += a.count_ones() as u64;
        cb += b.count_ones() as u64;
    }
    json!({"source_support":ca,"target_support":cb,"hamming":different,"fraction":different as f64/observations as f64,"jaccard":if union==0 {1.0}else{intersection as f64/union as f64}})
}

fn compare(
    inputs: &[cpu::Input],
    cpu: &[Vec<Matrix3<f64>>],
    gpu: &[FivePointTrialResult],
    observations: &[cpu::PairObservations],
    trials: usize,
) -> Value {
    let mut records = Vec::new();
    let mut cpu_to_gpu = Vec::new();
    let mut gpu_to_cpu = Vec::new();
    let mut cr = Vec::new();
    let mut gr = Vec::new();
    let mut masks = Vec::new();
    let mut mismatched = 0;
    let mut missing_gpu = 0;
    let mut missing_cpu = 0;
    let mut upstream = std::collections::BTreeMap::<String, usize>::new();
    let mut algebra = std::collections::BTreeMap::<String, usize>::new();
    let mut filtered = [0usize; 6];
    let mut gpu_empty = 0;
    let mut cpu_total = 0;
    let mut gpu_total = 0;
    assert!(
        inputs.len() == cpu.len() && inputs.len() == gpu.len(),
        "replay compare length mismatch: inputs={}, cpu={}, gpu={}",
        inputs.len(),
        cpu.len(),
        gpu.len()
    );
    assert!(
        trials > 0 && inputs.len() % trials == 0 && inputs.len() / trials == observations.len(),
        "replay observation length mismatch: inputs={}, observations={}, trials_per_pair={trials}",
        inputs.len(),
        observations.len()
    );
    for (i, ((input, cpu), gpu)) in inputs.iter().zip(cpu).zip(gpu).enumerate() {
        assert_eq!(gpu.trial, i);
        let gm: Vec<_> = gpu
            .slots
            .iter()
            .filter_map(|s| {
                s.essential
                    .map(|e| Matrix3::from_row_slice(&e.map(f64::from)))
            })
            .collect();
        let slots: Vec<_> = gpu
            .slots
            .iter()
            .filter(|s| s.essential.is_some())
            .map(|s| s.slot)
            .collect();
        let cg = nearest(cpu, &gm);
        let gc = nearest(&gm, cpu);
        let c_res: Vec<_> = cpu.iter().map(|e| constraint_residual(e, input)).collect();
        let g_res: Vec<_> = gm.iter().map(|e| constraint_residual(e, input)).collect();
        cr.extend(&c_res);
        gr.extend(&g_res);
        for n in &cg {
            if let Some((_, d)) = n {
                cpu_to_gpu.push(*d);
            } else {
                missing_gpu += 1;
            }
        }
        for n in &gc {
            if let Some((_, d)) = n {
                gpu_to_cpu.push(*d);
            } else {
                missing_cpu += 1;
            }
        }
        mismatched += usize::from(cpu.len() != gm.len());
        gpu_empty += usize::from(gm.is_empty());
        cpu_total += cpu.len();
        gpu_total += gm.len();
        *upstream
            .entry(format!("{:?}", gpu.upstream_status))
            .or_default() += 1;
        *algebra
            .entry(format!("{:?}", gpu.algebra_status))
            .or_default() += 1;
        for (sum, n) in filtered.iter_mut().zip([
            gpu.unconverged,
            gpu.complex_filtered,
            gpu.duplicate_roots,
            gpu.recovery_rejected,
            gpu.duplicate_models,
            usize::from(gpu.polynomial_failed),
        ]) {
            *sum += n;
        }
        let obs = &observations[i / trials];
        let cmasks: Vec<_> = cpu.iter().map(|e| mask(e, obs)).collect();
        let gmasks: Vec<_> = gm.iter().map(|e| mask(e, obs)).collect();
        let cm: Vec<_> = cg
            .iter()
            .enumerate()
            .map(|(j, n)| n.map(|(k, _)| mask_comparison(&cmasks[j], &gmasks[k], obs.left.len())))
            .collect();
        let gm: Vec<_> = gc
            .iter()
            .enumerate()
            .map(|(j, n)| n.map(|(k, _)| mask_comparison(&gmasks[j], &cmasks[k], obs.left.len())))
            .collect();
        for v in cm.iter().chain(&gm).flatten() {
            masks.push(v["fraction"].as_f64().unwrap());
        }
        records.push(json!({"global_trial":i,"pair_index":i/trials,"trial":i%trials,"sample_indices":input.indices,
            "cpu_count":cpu.len(),"gpu_count":gpu.model_count,"accepted_gpu_slots":slots,
            "slot_statuses":gpu.slots.iter().map(|s|format!("{:?}",s.status)).collect::<Vec<_>>(),
            "roots":gpu.slots.iter().map(|s|s.root).collect::<Vec<_>>(),
            "upstream":format!("{:?}",gpu.upstream_status),"algebra":format!("{:?}",gpu.algebra_status),
            "degree":gpu.degree,"iterations":gpu.root_iterations,"unconverged":gpu.unconverged,"complex":gpu.complex_filtered,
            "duplicate_roots":gpu.duplicate_roots,"recovery_rejected":gpu.recovery_rejected,"duplicate_models":gpu.duplicate_models,
            "cpu_to_gpu":cg,"gpu_to_cpu":gc,"cpu_residual":c_res,"gpu_residual":g_res,
            "cpu_to_gpu_masks":cm,"gpu_to_cpu_masks":gm}));
    }
    json!({"summary":{"trials":inputs.len(),"cpu_models":cpu_total,"gpu_models":gpu_total,"count_mismatch_trials":mismatched,"gpu_empty_trials":gpu_empty,
        "cpu_models_without_gpu_counterpart":missing_gpu,"gpu_models_without_cpu_counterpart":missing_cpu,
        "cpu_to_gpu_distance":summary(&cpu_to_gpu),"gpu_to_cpu_distance":summary(&gpu_to_cpu),
        "cpu_constraint_residual":summary(&cr),"gpu_constraint_residual":summary(&gr),"mask_hamming_fraction":summary(&masks),
        "upstream_status":upstream,"algebra_status":algebra,
        "unconverged_roots":filtered[0],"complex_filtered":filtered[1],"duplicate_roots":filtered[2],"recovery_rejected":filtered[3],"duplicate_models":filtered[4],"polynomial_failure_trials":filtered[5]},
        "trials":records})
}

fn main() -> Result<()> {
    let args = Args::parse();
    let load_start = Instant::now();
    let (inputs, pairs, eligible, observations) = cpu::load_with_observations(&cpu::Args {
        database: args.database.clone(),
        pairs: args.pairs,
        trials: args.trials,
        batch_size: 512,
    })?;
    let loading = load_start.elapsed().as_secs_f64();
    eprintln!(
        "loaded {} trials, {} eligible pairs, observations={:?}",
        inputs.len(),
        eligible,
        observations
            .iter()
            .map(|o| o.left.len())
            .collect::<Vec<_>>()
    );
    let init_start = Instant::now();
    let context = WgpuContext::try_new()?;
    let device = format!("{:?}", context.capabilities());
    let gpu = WgpuFivePointF32::from_context(context)?;
    let initialization = init_start.elapsed().as_secs_f64();
    eprintln!("GPU initialization={initialization}s, {device}");
    let pool = ThreadPoolBuilder::new().num_threads(4).build()?;
    let mut reports = Vec::new();
    let mut baseline_gpu: Option<Vec<Vec<u32>>> = None;
    for batch in BATCH_SIZES {
        let t = Instant::now();
        let baseline = cpu::replay_batches(&inputs, None, batch);
        let serial_warmup = t.elapsed().as_secs_f64();
        let bits = cpu::output_bits(&baseline);
        let t = Instant::now();
        let parallel = cpu::replay_batches(&inputs, Some(&pool), batch);
        let parallel_warmup = t.elapsed().as_secs_f64();
        ensure!(
            cpu::output_bits(&parallel) == bits,
            "CPU serial/4-thread warmup differs"
        );
        let t = Instant::now();
        let gpu_baseline = gpu_replay(&gpu, &inputs, batch)?;
        let gpu_warmup = t.elapsed().as_secs_f64();
        let gbits = gpu_bits(&gpu_baseline);
        let batch_exact = baseline_gpu.as_ref().map(|previous| *previous == gbits);
        if baseline_gpu.is_none() {
            baseline_gpu = Some(gbits.clone());
        }
        eprintln!("batch {batch} warmup serial={serial_warmup}, threads4={parallel_warmup}, GPU={gpu_warmup}");
        let mut rounds = Vec::new();
        for round in 0..args.rounds {
            let order = match round % 3 {
                0 => [0, 1, 2],
                1 => [2, 0, 1],
                _ => [1, 2, 0],
            };
            let mut measurements = Vec::new();
            for path in order {
                let t = Instant::now();
                if path == 2 {
                    let output = gpu_replay(&gpu, &inputs, batch)?;
                    let seconds = t.elapsed().as_secs_f64();
                    let exact = gpu_bits(&output) == gbits;
                    measurements.push(json!({"path":"gpu_full_upload_readback","seconds":seconds,"repeat_exact":exact}));
                    eprintln!("batch={batch} round={round} GPU={seconds}s repeat_exact={exact}");
                } else {
                    let output = cpu::replay_batches(
                        &inputs,
                        if path == 0 { None } else { Some(&pool) },
                        batch,
                    );
                    let seconds = t.elapsed().as_secs_f64();
                    ensure!(
                        cpu::output_bits(&output) == bits,
                        "CPU measured output differs"
                    );
                    measurements.push(json!({"path":if path==0 {"cpu_serial"}else{"cpu_4threads"},"seconds":seconds,"repeat_exact":true}));
                    eprintln!("batch={batch} round={round} CPU path={path} {seconds}s");
                }
            }
            rounds.push(json!({"round":round,"order":order,"measurements":measurements}));
        }
        let mut comparison = compare(
            &inputs,
            &baseline,
            &gpu_baseline,
            &observations,
            args.trials as usize,
        );
        eprintln!("batch {batch} comparison: {}", comparison["summary"]);
        // Keep aggregate quality, not the large per-trial diagnostic payload.
        comparison.as_object_mut().unwrap().remove("trials");
        reports.push(json!({"batch":batch,"gpu_solve_calls_per_replay":solve_calls(inputs.len(), batch),"gpu_solve_calls_including_warmup":solve_calls(inputs.len(), batch) * (args.rounds as usize + 1),"warmup_seconds":{"serial":serial_warmup,"threads4":parallel_warmup,"gpu":gpu_warmup},"gpu_exact_across_batches":batch_exact,"rounds":rounds,"comparison":comparison}));
    }
    let report = json!({"benchmark":"five_point_full_gpu_fixed_sampler_PREFIX","database":args.database,"device":device,"gpu_initialization_seconds":initialization,
        "scope":"Independent candidate generation only. NOT full RANSAC, NOT the 960-image pipeline. No CPU runtime fallback in GPU path.",
        "timing_contract":"Pre-gathered f64 rays. CPU solver + ordered collection; GPU includes f64-to-f32 conversion, allocations, uploads, all kernels, waits, readback and decode. Initialization, DB I/O, sampler, warmup, comparisons, masks, JSON and destruction after timing excluded. One warmup per path per batch; at least three interleaved rounds. No speedup inferred without candidate-quality differences.",
        "profiling_contract":"Host wall-clock total per replay only; GPU solve calls counted from chunks. No per-stage or kernel timestamp timing.",
                "distance_contract":"Bidirectional per-trial nearest model, min(norm(E-F),norm(E+F)) after Frobenius normalization; no basis/slot alignment; empty counterpart explicitly counted, not zero distance.",
        "mask_contract":"CPU f64 diagnostic masks for both model sets, all valid raw pair observations (not verified inliers), normalized image coordinates z=1, squared Sampson <= 1e-6; nearest-model pairs in both directions. Not a production pixel threshold.",
        "loading_seconds":loading,"eligible_pairs":eligible,"pairs":pairs,"seed_per_pair":1,"trials_per_pair":args.trials,"input_fingerprint":cpu::input_digest(&inputs),"batches":reports});
    if let Some(path) = args.output {
        serde_json::to_writer(
            std::io::BufWriter::new(std::fs::File::create(path)?),
            &report,
        )?;
    } else {
        serde_json::to_writer(std::io::stdout().lock(), &report)?;
        println!();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn batch_sweep_covers_full_prefix_and_call_counts() {
        assert_eq!(BATCH_SIZES, [64, 128, 256, 512, 1024, 2048, 6144]);
        assert_eq!(
            BATCH_SIZES.map(|batch| solve_calls(6144, batch)),
            [96, 48, 24, 12, 6, 3, 1]
        );
        assert_eq!(solve_calls(65, 64), 2);
        assert_eq!(solve_calls(0, 64), 0);
        let inputs: Vec<_> = (0..6144).collect();
        for batch in BATCH_SIZES {
            let indices: Vec<_> = inputs
                .chunks(batch)
                .enumerate()
                .flat_map(|(chunk, values)| (0..values.len()).map(move |i| chunk * batch + i))
                .collect();
            assert_eq!(indices, inputs);
        }
    }

    #[test]
    fn sign_invariance_and_empty_counterpart() {
        let e = Matrix3::new(0., -1., 2., 1., 0., -3., -2., 3., 0.);
        assert!(distance(&e, &(-e * 7.0)) < 1e-14);
        assert_eq!(nearest(&[e], &[]), vec![None]);
    }
    #[test]
    fn mask_sign_invariance() {
        let obs = cpu::PairObservations {
            left: vec![Vector3::new(0.1, 0.2, 1.0)],
            right: vec![Vector3::new(0.3, 0.2, 1.0)],
        };
        let e = Matrix3::new(0., 0., 0., 0., 0., -1., 0., 1., 0.);
        assert_eq!(mask(&e, &obs), vec![1]);
        assert_eq!(mask(&e, &obs), mask(&(-e), &obs));
    }
}
