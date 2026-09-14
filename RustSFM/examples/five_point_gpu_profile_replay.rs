//! Diagnostic only: no production routing, threshold changes or CPU runtime fallback.
#[allow(dead_code)]
#[path = "five_point_replay_probe.rs"]
mod cpu;
#[allow(dead_code)]
#[path = "../src/geometry/five_point_generated.rs"]
mod generated;

use anyhow::{ensure, Result};
use clap::Parser;
use nalgebra::{DMatrix, Matrix3};
use rustsfm::gpu::{
    FivePointProfile, FivePointSlotStatus as Slot, FivePointStatus as Status, FivePointTrialResult,
    WgpuContext, WgpuFivePointF32, FIVE_POINT_PASS_NAMES,
};
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    database: PathBuf,
    #[arg(long)]
    output: PathBuf,
}
const INPUT_DIGEST: &str = "af07459d637b5b8c2b20c76d3ef103f058505fe0e8307578ee7e8750f3e230c5";
const PREVIOUS_SIGNATURE: &str = "1f6f3f8d88a17e4e1c42abfc85aad44a66c4b864091805594e086ca80bf9b339";
// Diagnostic association tolerances only; never passed to either solver.
const DISTANCES: [f64; 3] = [1e-3, 1e-2, 1e-1];

fn signature(results: &[FivePointTrialResult]) -> String {
    let mut hash = blake3::Hasher::new();
    hash.update(b"gpu-replay-bits-v1");
    for r in results {
        let mut words = vec![
            r.trial as u32,
            r.model_count as u32,
            r.unconverged as u32,
            r.root_iterations as u32,
            r.upstream_status as u32,
            r.algebra_status as u32,
            r.polynomial_failed as u32,
            r.degree as u32,
            r.complex_filtered as u32,
            r.duplicate_roots as u32,
            r.recovery_rejected as u32,
            r.duplicate_models as u32,
            r.dropped_leading as u32,
            r.rank as u32,
            r.jacobi_sweeps as u32,
        ];
        for s in &r.slots {
            words.extend([
                s.slot as u32,
                s.status as u32,
                s.root[0].to_bits(),
                s.root[1].to_bits(),
                s.root_backward_error.to_bits(),
                s.null_residual.to_bits(),
                s.constraint_residual.to_bits(),
                s.essential_residual.to_bits(),
                s.essential.is_some() as u32,
            ]);
            if let Some(e) = s.essential {
                words.extend(e.map(f32::to_bits));
            }
        }
        hash.update(&(words.len() as u64).to_le_bytes());
        for w in words {
            hash.update(&w.to_le_bytes());
        }
    }
    hash.finalize().to_hex().to_string()
}

fn rays(inputs: &[cpu::Input]) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
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
fn replay(
    gpu: &WgpuFivePointF32,
    inputs: &[cpu::Input],
    batch: usize,
    profile: bool,
) -> Result<(Vec<FivePointTrialResult>, Vec<FivePointProfile>, f64)> {
    let started = Instant::now();
    let mut all = Vec::with_capacity(inputs.len());
    let mut timings = Vec::new();
    for (chunk, inputs) in inputs.chunks(batch).enumerate() {
        let (left, right) = rays(inputs);
        let mut results = if profile {
            let (results, timing) = gpu.solve_essential_profiled(&left, &right)?;
            timings.push(timing);
            results
        } else {
            gpu.solve_essential(&left, &right)?
        };
        for r in &mut results {
            r.trial += chunk * batch;
        }
        all.extend(results);
    }
    let seconds = started.elapsed().as_secs_f64();
    Ok((all, timings, seconds))
}
fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
fn distance(a: &Matrix3<f64>, b: &Matrix3<f64>) -> f64 {
    let a = a.normalize();
    let b = b.normalize();
    (a - b).norm().min((a + b).norm())
}
fn missing(source: &[Matrix3<f64>], target: &[Matrix3<f64>]) -> [usize; 3] {
    let nearest: Vec<_> = source
        .iter()
        .map(|a| {
            target
                .iter()
                .map(|b| distance(a, b))
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    DISTANCES.map(|threshold| nearest.iter().filter(|&&d| d > threshold).count())
}
fn flags(r: &FivePointTrialResult) -> [bool; 6] {
    [
        r.upstream_status == Status::RankDeficient,
        r.algebra_status != Status::Success,
        r.polynomial_failed,
        r.unconverged > 0,
        r.recovery_rejected > 0,
        r.root_iterations == 384,
    ]
}
const FLAG_NAMES: [&str; 6] = [
    "rank_failed",
    "algebra_failed",
    "polynomial_failed",
    "not_converged_status",
    "recovery_rejected",
    "iteration_limit",
];

#[derive(Default, serde::Serialize)]
struct Counts {
    trials: usize,
    cpu_models: usize,
    gpu_models: usize,
    cpu_nonempty_gpu_empty_trials: usize,
    cpu_empty_gpu_nonempty_trials: usize,
    gpu_empty_trials: usize,
    count_mismatch_trials: usize,
    cpu_models_with_empty_gpu_set: usize,
    gpu_models_with_empty_cpu_set: usize,
    cpu_missing_by_distance: [usize; 3],
    gpu_missing_by_distance: [usize; 3],
    cpu_missing_trials_by_distance: [usize; 3],
    gpu_missing_trials_by_distance: [usize; 3],
}
impl Counts {
    fn add(&mut self, c: &[Matrix3<f64>], g: &[Matrix3<f64>], cm: [usize; 3], gm: [usize; 3]) {
        self.trials += 1;
        self.cpu_models += c.len();
        self.gpu_models += g.len();
        self.cpu_nonempty_gpu_empty_trials += usize::from(!c.is_empty() && g.is_empty());
        self.cpu_empty_gpu_nonempty_trials += usize::from(c.is_empty() && !g.is_empty());
        self.gpu_empty_trials += usize::from(g.is_empty());
        self.count_mismatch_trials += usize::from(c.len() != g.len());
        if g.is_empty() {
            self.cpu_models_with_empty_gpu_set += c.len();
        }
        if c.is_empty() {
            self.gpu_models_with_empty_cpu_set += g.len();
        }
        for i in 0..3 {
            self.cpu_missing_by_distance[i] += cm[i];
            self.gpu_missing_by_distance[i] += gm[i];
            self.cpu_missing_trials_by_distance[i] += usize::from(cm[i] > 0);
            self.gpu_missing_trials_by_distance[i] += usize::from(gm[i] > 0);
        }
    }
}
fn attribution(
    cpu: &[Vec<Matrix3<f64>>],
    gpu: &[FivePointTrialResult],
) -> Result<(Value, Vec<usize>)> {
    ensure!(cpu.len() == gpu.len(), "result count mismatch");
    let mut total = Counts::default();
    let mut marginal: BTreeMap<&str, Counts> = FLAG_NAMES
        .into_iter()
        .map(|name| (name, Counts::default()))
        .collect();
    let mut joint = BTreeMap::<u8, Counts>::new();
    let mut slots = BTreeMap::<String, usize>::new();
    let mut upstream = BTreeMap::<String, usize>::new();
    let mut algebra = BTreeMap::<String, usize>::new();
    let mut representatives = BTreeMap::<String, usize>::new();
    let mut degree_sum = 0;
    let mut root_trials = 0;
    for (i, (c, r)) in cpu.iter().zip(gpu).enumerate() {
        ensure!(r.trial == i, "trial identity mismatch");
        let g: Vec<_> = r
            .slots
            .iter()
            .filter_map(|s| {
                s.essential
                    .map(|e| Matrix3::from_row_slice(&e.map(f64::from)))
            })
            .collect();
        ensure!(r.model_count == g.len(), "model count mismatch");
        ensure!(
            r.unconverged
                == r.slots
                    .iter()
                    .filter(|s| s.status == Slot::NotConverged)
                    .count(),
            "root counter mismatch"
        );
        ensure!(
            r.recovery_rejected
                == r.slots
                    .iter()
                    .filter(|s| matches!(
                        s.status,
                        Slot::NullVectorFailure | Slot::InvalidEssential
                    ))
                    .count(),
            "recovery counter mismatch"
        );
        let cm = missing(c, &g);
        let gm = missing(&g, c);
        total.add(c, &g, cm, gm);
        let mut bits = 0u8;
        for (j, flag) in flags(r).into_iter().enumerate() {
            if flag {
                bits |= 1 << j;
                marginal
                    .entry(FLAG_NAMES[j])
                    .or_default()
                    .add(c, &g, cm, gm);
                representatives.entry(FLAG_NAMES[j].into()).or_insert(i);
            }
        }
        joint.entry(bits).or_default().add(c, &g, cm, gm);
        if cm[1] > 0 && bits == 0 {
            representatives
                .entry("missing_without_failure_flags".into())
                .or_insert(i);
        }
        if !c.is_empty() && g.is_empty() {
            representatives
                .entry("cpu_nonempty_gpu_empty".into())
                .or_insert(i);
        }
        *upstream
            .entry(format!("{:?}", r.upstream_status))
            .or_default() += 1;
        *algebra
            .entry(format!("{:?}", r.algebra_status))
            .or_default() += 1;
        for s in &r.slots {
            *slots.entry(format!("{:?}", s.status)).or_default() += 1;
        }
        degree_sum += r.degree;
        root_trials += usize::from(r.root_iterations > 0);
    }
    let ids = representatives
        .values()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    Ok((
        json!({"total":total,"marginal_overlapping_trial_groups":marginal,
        "joint_disjoint_trial_groups_by_bitmask":joint,"bit_order":FLAG_NAMES,
        "upstream_status_trials":upstream,"algebra_status_trials":algebra,
        "root_slot_status_counts_including_unused":slots,"allocated_slots":gpu.len()*10,
        "sum_reported_degree":degree_sum,"trials_entering_root_iterations":root_trials,
        "representative_first_trial_by_group":representatives,"distance_thresholds":DISTANCES,
        "association":"bidirectional nearest unit Frobenius sign-invariant distance; not bijective; missing includes empty target sets; diagnostic thresholds only",
        "warning":"Marginals overlap: never sum them. NotConverged includes real-axis residual/polish rejection, not only iteration failure. Associations are not causal attribution."}),
        ids,
    ))
}

fn relative_error(g: &[f32], c: &[f64]) -> f64 {
    let scale = c.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1e-30);
    g.iter()
        .zip(c)
        .map(|(&g, &c)| (f64::from(g) - c).abs() / scale)
        .fold(0.0, f64::max)
}
fn determinant(s: &DMatrix<f64>) -> [f64; 39] {
    let mut b = [0.0; 39];
    for col in 0..3 {
        for (offset, start, count) in [(0, 0, 3), (4, 3, 3), (8, 6, 4)] {
            for k in 0..count {
                b[13 * col + offset + k + 1] += s[(2 * col + 4, start + k)];
                b[13 * col + offset + k] -= s[(2 * col + 5, start + k)];
            }
        }
    }
    b
}
// Separate, untimed re-dispatches of a few selected trials. Never used for timing.
fn representative(
    gpu: &WgpuFivePointF32,
    input: &cpu::Input,
    id: usize,
    r: &FivePointTrialResult,
) -> Result<Value> {
    let (l, rr) = rays(std::slice::from_ref(input));
    let a = gpu.compute_constraint_matrices(&l, &rr)?[0];
    let d = gpu.compute_nullspace_diagnostics(&[a])?.remove(0);
    let a64 = DMatrix::from_row_slice(5, 9, &a.map(f64::from));
    let singular = a64.clone().svd(false, false).singular_values;
    let trace = a64.norm_squared();
    let eigen_ratios: Vec<_> = singular.iter().map(|s| s * s / trace).collect();
    let quantized = cpu::Input {
        indices: input.indices,
        left: input.left.map(|v| v.map(|x| f64::from(x as f32))),
        right: input.right.map(|v| v.map(|x| f64::from(x as f32))),
    };
    let original_cpu = cpu::replay_batches(std::slice::from_ref(input), None, 1);
    let quantized_cpu = cpu::replay_batches(&[quantized], None, 1);
    let mut same_basis = Value::Null;
    if let Some(basis) = d.basis {
        let g = gpu.compute_algebra(&[basis])?.remove(0);
        let expansion = generated::build_elimination_matrix(&basis.map(f64::from));
        let left = DMatrix::from_column_slice(10, 10, &expansion[..100]);
        let right = DMatrix::from_column_slice(10, 10, &expansion[100..]);
        let reference = left.lu().solve(&right).map(|solved| {
            let b = determinant(&solved);
            let coefficients = generated::determinant_coeffs(&b);
            json!({"solved_relative_max_error":relative_error(&g.solved,solved.as_slice()),
                "determinant_relative_max_error":relative_error(&g.determinant,&b),
                "coefficients_relative_max_error":relative_error(&g.coefficients,&coefficients)})
        });
        same_basis = json!({"gpu_status":format!("{:?}",g.status),
            "minimum_relative_pivot":g.minimum_relative_pivot,
            "expansion_relative_max_error":relative_error(&g.elimination,&expansion),
            "f64_partial_pivot_lu_reference":reference,
            "same_gpu_B_f64_polynomial_relative_max_error":relative_error(&g.coefficients,&generated::determinant_coeffs(&g.determinant.map(f64::from)))});
    }
    Ok(
        json!({"global_trial":id,"pair_index":id/512,"local_trial":id%512,
        "sample_indices":input.indices,"input_digest":cpu::input_digest(std::slice::from_ref(input)),
        "full_upstream":format!("{:?}",r.upstream_status),"stage_upstream":format!("{:?}",d.status),
        "gpu_rank":d.rank,"same_GPU_A_f64_singular_values":singular.as_slice(),
        "same_GPU_A_f64_squared_singular_over_trace":eigen_ratios,
        "cpu_original_models":original_cpu[0].len(),"cpu_f32_quantized_ray_models":quantized_cpu[0].len(),
        "gpu_models":r.model_count,"same_GPU_basis_algebra":same_basis,
        "limitation":"Selected first failures, not representative prevalence. f64 on GPU A isolates matrix rank arithmetic; quantized CPU uses its own basis. Same GPU basis algebra uses nalgebra partial-pivot LU, not a claim of CPU implementation bit parity. No basis is fabricated for rank failures; no recovery/root causal intervention."}),
    )
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.output.exists(), "refusing to overwrite report");
    let (inputs, pairs, eligible, _) = cpu::load_with_observations(&cpu::Args {
        database: args.database.clone(),
        pairs: 127,
        trials: 512,
        batch_size: 512,
    })?;
    let digest = cpu::input_digest(&inputs);
    ensure!(
        inputs.len() == 65024 && digest == INPUT_DIGEST,
        "real input provenance changed: {digest}"
    );
    let default = WgpuContext::try_new()?;
    ensure!(
        !default.timestamp_queries_enabled(),
        "production default unexpectedly requests timestamps"
    );
    let experimental = WgpuContext::try_new_experimental_timestamps()?;
    ensure!(
        default.capabilities().device_name == experimental.capabilities().device_name
            && default.backend() == experimental.backend(),
        "adapter mismatch"
    );
    let supported = experimental.timestamp_queries_enabled();
    let device = format!("{:?}", experimental.capabilities());
    let normal = WgpuFivePointF32::from_context(default)?;
    let diagnostic = WgpuFivePointF32::from_context(experimental)?;
    let mut baseline = None;
    let mut reports = Vec::new();
    for batch in [32768, 65024] {
        let mut warmups = Vec::new();
        let mut rounds = Vec::new();
        let mut seconds = [Vec::new(), Vec::new(), Vec::new()];
        for round in 0..6 {
            // Rotate default off / timestamp-enabled off / timestamp-enabled on.
            for step in 0..3 {
                let mode = (round + step) % 3;
                let gpu = if mode == 0 { &normal } else { &diagnostic };
                let (results, timings, elapsed) = replay(gpu, &inputs, batch, mode == 2)?;
                let sig = signature(&results);
                ensure!(
                    sig == PREVIOUS_SIGNATURE,
                    "GPU signature changed for batch {batch}, mode {mode}, round {round}: {sig}"
                );
                if baseline.is_none() {
                    baseline = Some(results);
                }
                let record = json!({"round":round%3,"mode":mode,"seconds":elapsed,"signature":sig,"chunks":timings});
                if round < 3 {
                    warmups.push(record);
                } else {
                    seconds[mode].push(elapsed);
                    rounds.push(record);
                }
            }
        }
        let medians = seconds.each_ref().map(|v| median(v));
        reports.push(
            json!({"batch":batch,"calls_per_workset":inputs.len().div_ceil(batch),
            "warmups":warmups,"rounds":rounds,"median_seconds":medians,
            "timestamp_context_off_vs_default_ratio":medians[1]/medians[0],
            "profile_on_vs_same_context_off_ratio":medians[2]/medians[1]}),
        );
        eprintln!(
            "batch {batch}: default/off/on median ms {:.3}/{:.3}/{:.3}",
            medians[0] * 1e3,
            medians[1] * 1e3,
            medians[2] * 1e3
        );
    }
    let baseline = baseline.unwrap();
    let cpu = cpu::replay_batches(&inputs, None, 65024);
    let (attribution, ids) = attribution(&cpu, &baseline)?;
    let representatives: Vec<_> = ids
        .into_iter()
        .map(|id| representative(&normal, &inputs[id], id, &baseline[id]))
        .collect::<Result<_>>()?;
    let report = json!({"schema":"five-point-profile-attribution-v1","database":args.database,
        "device":device,"timestamp_supported_and_enabled":supported,
        "measurement_method":if supported {"18 pass-boundary timestamp queries; separate resolve submission after completion of all 9 passes and result readback; one final timestamp readback (resolve/wait included in timestamp_readback_seconds)"} else {"host-only prepare/encode/submit/wait/readback/decode; no per-pass GPU times available; wait includes scheduling and GPU execution, cannot attribute stages"},
        "timing_contract":"Each replay covers all 65024 inputs. Outer time includes f64->f32 conversion, buffers, queries, encode/submit/wait/readback/decode and collection/destruction inside replay; excludes signatures, CPU diagnostics, initialization and DB loading. Host stage intervals disjoint; GPU intervals overlap host wait and must not be added to host intervals. GPU pass sum excludes inter-pass gaps and copy/resolve. No per-stage readback timing.",
        "modes":["default_context_profile_off","timestamp_context_profile_off","timestamp_context_profile_on"],
        "warmup_rounds_per_mode_per_batch":3,"measured_rounds_per_mode_per_batch":3,
        "gpu_pass_names":FIVE_POINT_PASS_NAMES,"input_fingerprint":digest,"pairs":pairs,
        "eligible_pairs":eligible,"trials":inputs.len(),"trials_per_pair":512,"seed_per_pair":1,
        "gpu_signature":signature(&baseline),"all_signatures_match_capacity_sweep":true,
        "batches":reports,"attribution":attribution,"selected_stage_diagnostics":representatives,
        "limitations":["One adapter, three measured rounds; no statistical performance guarantee or production equivalence.",
            "Equal pass-boundary timestamps are reported as null, never zero-cost GPU work. Raw ticks/period retained. Metal recovery samples showed this anomaly; driver/sampling cause not established. Other pass samples remain instrument observations, not certified exclusive hardware time.",
            "CPU f64 original rays versus GPU f32 quantized rays in full-workset association; representative quantized CPU comparison is separate.",
            "wgpu guarantees initialized buffer contents. No claim of previous uninitialized-buffer randomness.",
            "No solver, shader, threshold, matching/RANSAC changes; CPU calculations are offline diagnostics only."]});
    let bytes = serde_json::to_vec(&report)?;
    ensure!(
        bytes.len() < 100000,
        "diagnostic summary unexpectedly large"
    );
    std::fs::write(&args.output, &bytes)?;
    eprintln!("wrote {} bytes to {}", bytes.len(), args.output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn candidate_absence_and_sign_are_distinct() {
        let a = Matrix3::identity();
        assert_eq!(missing(&[a], &[-a]), [0; 3]);
        assert_eq!(missing(&[a], &[]), [1; 3]);
        assert_eq!(missing(&[], &[a]), [0; 3]);
        let mut c = Counts::default();
        c.add(&[a], &[], [1; 3], [0; 3]);
        assert_eq!(c.cpu_nonempty_gpu_empty_trials, 1);
        assert_eq!(c.cpu_models_with_empty_gpu_set, 1);
        assert_eq!(c.cpu_missing_trials_by_distance, [1; 3]);
    }
    #[test]
    fn actual_gpu_optional_profile_preserves_results() -> Result<()> {
        for timestamps in [false, true] {
            let context = if timestamps {
                WgpuContext::try_new_experimental_timestamps()?
            } else {
                WgpuContext::try_new()?
            };
            let enabled = context.timestamp_queries_enabled();
            if !timestamps {
                assert!(!enabled);
            }
            let gpu = WgpuFivePointF32::from_context(context)?;
            let left = [[1.0, 0.0, 0.0]; 10];
            let a = gpu.solve_essential(&left, &left)?;
            let (b, p) = gpu.solve_essential_profiled(&left, &left)?;
            assert_eq!(signature(&a), signature(&b));
            assert_eq!(p.gpu_pass_seconds.is_some(), enabled);
            if let Some(times) = p.gpu_pass_seconds {
                assert!(times.iter().flatten().all(|x| x.is_finite() && *x > 0.0));
                assert!(times.iter().flatten().sum::<f64>() > 0.0);
                let ticks = p.timestamp_ticks.unwrap();
                for i in 0..9 {
                    assert_eq!(times[i].is_none(), ticks[i * 2] == ticks[i * 2 + 1]);
                }
            }
            let (empty, _) = gpu.solve_essential_profiled(&[], &[])?;
            assert!(empty.is_empty());
            assert!(gpu
                .solve_essential_profiled(&[[f32::NAN; 3]; 5], &[[0.0; 3]; 5])
                .is_err());
            let cpu = vec![vec![Matrix3::identity(); 4]; 2];
            let (counts, _) = attribution(&cpu, &b)?;
            assert_eq!(counts["total"]["cpu_nonempty_gpu_empty_trials"], 2);
            assert_eq!(counts["total"]["cpu_models_with_empty_gpu_set"], 8);
            assert_eq!(
                counts["marginal_overlapping_trial_groups"]["rank_failed"]["trials"],
                2
            );
            assert_eq!(
                counts["marginal_overlapping_trial_groups"]["algebra_failed"]["trials"],
                2
            );
        }
        Ok(())
    }
}
