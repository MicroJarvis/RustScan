//! Q1 failure classification and `roots` SIMD-divergence attribution.
//!
//! Pure measurement round: no solver, shader or existing harness source is
//! changed. The divergence experiment only permutes the order of the input
//! trials and verifies that every trial's decoded result is unchanged under the
//! inverse permutation, so it isolates scheduling from arithmetic.
//!
//! Not RANSAC, not the 960-image pipeline, no CPU runtime fallback. The CPU f64
//! replay here is an offline reference for lost-solution attribution only.
#[allow(dead_code)]
#[path = "five_point_gpu_capacity_replay.rs"]
mod shared;

use anyhow::{ensure, Result};
use clap::Parser;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use rayon::ThreadPoolBuilder;
use rustscan_sfm::gpu::{
    FivePointSlotStatus, FivePointStatus, FivePointTrialResult, WgpuContext, WgpuFivePointF32,
    FIVE_POINT_PASS_NAMES,
};
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf};

const INPUT: &str = "af07459d637b5b8c2b20c76d3ef103f058505fe0e8307578ee7e8750f3e230c5";
const GPU_SIGNATURE: &str = "1f6f3f8d88a17e4e1c42abfc85aad44a66c4b864091805594e086ca80bf9b339";
const CPU_SIGNATURE: &str = "9e6764b41602d0005710187d2558d046f1a843c5f873b9542dd7b4f6c29c563a";
/// Index of `roots` in `FIVE_POINT_PASS_NAMES`.
const ROOTS_PASS: usize = 7;
/// The Aberth loop bound in `five_point_recovery.wgsl`.
const ITERATION_CAP: usize = 384;
/// `roots` is dispatched at `workgroup_size(32)`, one invocation per trial, so
/// 32 consecutive trials share one SIMD group and advance in lockstep.
const SIMD_WIDTH: usize = 32;
const WARMUPS: usize = 3;
const MEASURED: usize = 5;

fn bucket(iterations: usize) -> &'static str {
    match iterations {
        0 => "0 (never entered the loop)",
        1..=9 => "1-9",
        10..=19 => "10-19",
        20..=29 => "20-29",
        30..=49 => "30-49",
        50..=99 => "50-99",
        100..=199 => "100-199",
        200..=383 => "200-383",
        _ => "384 (cap)",
    }
}

fn percentile(sorted_ascending: &[usize], fraction: f64) -> usize {
    assert!(!sorted_ascending.is_empty());
    let index = ((sorted_ascending.len() - 1) as f64 * fraction).round() as usize;
    sorted_ascending[index]
}

/// Lockstep cost model for the Aberth loop only. Every lane of a SIMD group
/// executes until the slowest trial in that group stops, so the group costs its
/// maximum, not its mean. `sorted` is the floor reachable by grouping trials of
/// similar difficulty together; it is an oracle bound, not an implementable plan.
fn divergence(iterations: &[usize], width: usize) -> (u64, u64, u64) {
    let ideal: u64 = iterations.iter().map(|&i| i as u64).sum();
    let lockstep = |values: &[usize]| -> u64 {
        values
            .chunks(width)
            .map(|group| group.iter().copied().max().unwrap_or(0) as u64 * group.len() as u64)
            .sum()
    };
    let mut descending = iterations.to_vec();
    descending.sort_unstable_by(|a, b| b.cmp(a));
    (ideal, lockstep(iterations), lockstep(&descending))
}

/// Stage that ended the trial. `upstream_status` carries the nullspace status
/// when the nullspace failed and the algebra status otherwise; the two kernels
/// emit disjoint code sets (nullspace: NotConverged/RankDeficient/InvalidBasis;
/// algebra: SingularElimination/NonFinite), so the stage is unambiguous.
fn primary_category(trial: &FivePointTrialResult) -> &'static str {
    match trial.upstream_status {
        FivePointStatus::NotConverged => "1_nullspace_not_converged",
        FivePointStatus::RankDeficient => "2_nullspace_rank_deficient",
        FivePointStatus::InvalidBasis => "3_nullspace_invalid_basis",
        FivePointStatus::SingularElimination => "4_algebra_singular_elimination",
        FivePointStatus::NonFinite => "5_algebra_nonfinite",
        FivePointStatus::Success => {
            if trial.polynomial_failed {
                "6_polynomial_or_root_setup_rejected"
            } else if trial.model_count == 0 {
                "7_reached_roots_all_candidates_rejected"
            } else {
                "8_reached_roots_produced_models"
            }
        }
    }
}

fn rays(inputs: &[shared::cpu::Input], order: &[usize]) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let convert = |left: bool| {
        order
            .iter()
            .flat_map(|&i| {
                let input = &inputs[i];
                (if left { &input.left } else { &input.right })
                    .iter()
                    .map(|r| [r.x as f32, r.y as f32, r.z as f32])
            })
            .collect()
    };
    (convert(true), convert(false))
}

fn median(values: &[f64]) -> f64 {
    assert!(!values.is_empty());
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
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
    drop(observations);
    let digest = shared::cpu::input_digest(&inputs);
    ensure!(
        inputs.len() == 65024 && digest == INPUT,
        "input provenance changed"
    );
    let context = WgpuContext::try_new()?;
    let device = format!("{:?}", context.capabilities());
    let gpu = WgpuFivePointF32::from_context(context)?;
    let identity: Vec<usize> = (0..inputs.len()).collect();
    let baseline = shared::gpu_replay(&gpu, &inputs, inputs.len())?;
    let bits = shared::gpu_bits(&baseline);
    let signature = shared::signature(&bits);
    ensure!(
        signature == GPU_SIGNATURE,
        "historical signature changed: {signature}"
    );

    // ---- Iteration distribution -------------------------------------------
    let iterations: Vec<usize> = baseline.iter().map(|t| t.root_iterations).collect();
    let mut ascending = iterations.clone();
    ascending.sort_unstable();
    let mut histogram: BTreeMap<&str, usize> = BTreeMap::new();
    for &i in &iterations {
        *histogram.entry(bucket(i)).or_default() += 1;
    }
    let at_cap = iterations.iter().filter(|&&i| i == ITERATION_CAP).count();
    let entered = iterations.iter().filter(|&&i| i > 0).count();
    let total_iterations: u64 = iterations.iter().map(|&i| i as u64).sum();

    // ---- Lockstep divergence model ----------------------------------------
    let mut widths = Vec::new();
    for width in [8usize, 16, 32, 64] {
        let (ideal, lockstep, sorted) = divergence(&iterations, width);
        widths.push(json!({"width":width,"ideal_lane_iterations":ideal,
            "lockstep_lane_iterations":lockstep,
            "lockstep_over_ideal":lockstep as f64/ideal as f64,
            "sorted_lockstep_lane_iterations":sorted,
            "lockstep_over_sorted":lockstep as f64/sorted as f64}));
    }
    let (ideal32, lockstep32, sorted32) = divergence(&iterations, SIMD_WIDTH);
    let groups_with_cap = iterations
        .chunks(SIMD_WIDTH)
        .filter(|g| g.contains(&ITERATION_CAP))
        .count();
    let group_count = iterations.len().div_ceil(SIMD_WIDTH);

    // ---- Q1 classification -------------------------------------------------
    let mut categories: BTreeMap<&str, usize> = BTreeMap::new();
    for trial in &baseline {
        *categories.entry(primary_category(trial)).or_default() += 1;
    }
    let mut slot_statuses: BTreeMap<String, usize> = BTreeMap::new();
    for trial in &baseline {
        for slot in &trial.slots {
            *slot_statuses
                .entry(format!("{:?}", slot.status))
                .or_default() += 1;
        }
    }
    // A trial whose loop stopped before the cap left it through `all_done`,
    // which requires `done[i]` for every root of that trial. Path `!done[i]`
    // therefore cannot have fired, so any NotConverged slot in such a trial was
    // rejected by the unscaled final evaluation or by the real polish check --
    // not by exhausting iterations.
    let unconverged_trials = baseline.iter().filter(|t| t.unconverged > 0).count();
    let unconverged_below_cap = baseline
        .iter()
        .filter(|t| t.unconverged > 0 && t.root_iterations < ITERATION_CAP)
        .count();
    let unconverged_at_cap = baseline
        .iter()
        .filter(|t| t.unconverged > 0 && t.root_iterations == ITERATION_CAP)
        .count();
    let cap_with_models = baseline
        .iter()
        .filter(|t| t.root_iterations == ITERATION_CAP && t.model_count > 0)
        .count();
    let unconverged_slots_below_cap: usize = baseline
        .iter()
        .filter(|t| t.root_iterations < ITERATION_CAP)
        .map(|t| t.unconverged)
        .sum();
    let unconverged_slots_total: usize = baseline.iter().map(|t| t.unconverged).sum();
    // Three NotConverged paths, separated without touching the shader. The
    // first gate (`final_eval`) rejects before the root and backward error are
    // stored, so its slots keep the buffer's zero-initialized value; the later
    // two gates run after that store. Roots at exactly (0, 0) with zero
    // backward error are not otherwise reachable for these polynomials.
    let mut real_axis_gate = 0usize;
    let mut after_store_below_cap = 0usize;
    let mut after_store_at_cap = 0usize;
    let mut after_store_within_realness = 0usize;
    let mut accepted_slots = 0usize;
    let mut complex_slots = 0usize;
    for trial in &baseline {
        for slot in &trial.slots {
            match slot.status {
                FivePointSlotStatus::Accepted => accepted_slots += 1,
                FivePointSlotStatus::Complex => complex_slots += 1,
                FivePointSlotStatus::RealAxisRejected => real_axis_gate += 1,
                FivePointSlotStatus::RootNotDone | FivePointSlotStatus::PolishRejected => {
                    if trial.root_iterations < ITERATION_CAP {
                        after_store_below_cap += 1;
                    } else {
                        after_store_at_cap += 1;
                    }
                    if slot.root[1].abs() <= 2e-4 * (1.0 + slot.root[0].abs()) {
                        after_store_within_realness += 1;
                    }
                }
                _ => {}
            }
        }
    }
    let rejection_totals = json!({
        "unconverged_slots":unconverged_slots_total,
        "complex_filtered":baseline.iter().map(|t|t.complex_filtered).sum::<usize>(),
        "duplicate_roots":baseline.iter().map(|t|t.duplicate_roots).sum::<usize>(),
        "recovery_rejected":baseline.iter().map(|t|t.recovery_rejected).sum::<usize>(),
        "duplicate_models":baseline.iter().map(|t|t.duplicate_models).sum::<usize>(),
        "dropped_leading_trials":baseline.iter().filter(|t|t.dropped_leading>0).count(),
        "polynomial_failed_trials":baseline.iter().filter(|t|t.polynomial_failed).count(),
        "models":baseline.iter().map(|t|t.model_count).sum::<usize>(),
        "degree_sum":baseline.iter().map(|t|t.degree).sum::<usize>(),
    });

    // ---- CPU f64 reference: where the lost solutions go --------------------
    let pool = ThreadPoolBuilder::new().num_threads(8).build()?;
    let reference = shared::cpu::replay_batches(&inputs, Some(&pool), inputs.len());
    let mut cpu_hash = blake3::Hasher::new();
    for models in &shared::cpu::output_bits(&reference) {
        cpu_hash.update(&(models.len() as u64).to_le_bytes());
        for model in models {
            for word in model {
                cpu_hash.update(&word.to_le_bytes());
            }
        }
    }
    let cpu_signature = cpu_hash.finalize().to_hex().to_string();
    ensure!(
        cpu_signature == CPU_SIGNATURE,
        "CPU f64 reference signature changed: {cpu_signature}"
    );
    let mut lost: BTreeMap<&str, usize> = BTreeMap::new();
    let mut lost_iterations = Vec::new();
    for (cpu, trial) in reference.iter().zip(&baseline) {
        if !cpu.is_empty() && trial.model_count == 0 {
            *lost.entry(primary_category(trial)).or_default() += 1;
            lost_iterations.push(trial.root_iterations);
        }
    }
    lost_iterations.sort_unstable();
    let mut lost_histogram: BTreeMap<&str, usize> = BTreeMap::new();
    for &i in &lost_iterations {
        *lost_histogram.entry(bucket(i)).or_default() += 1;
    }

    // ---- Divergence experiment: reorder inputs only -----------------------
    let mut sorted_order = identity.clone();
    sorted_order.sort_by_key(|&i| (std::cmp::Reverse(iterations[i]), i));
    let mut shuffled_order = identity.clone();
    shuffled_order.shuffle(&mut StdRng::seed_from_u64(0x5f32));
    let timestamps = WgpuContext::try_new_experimental_timestamps()?;
    ensure!(
        timestamps.timestamp_queries_enabled(),
        "actual GPU timestamps required"
    );
    let profiled = WgpuFivePointF32::from_context(timestamps)?;
    let mut orders = Vec::new();
    for (name, order) in [
        ("identity", &identity),
        ("shuffled_control", &shuffled_order),
        ("sorted_by_iterations_desc", &sorted_order),
    ] {
        let (left, right) = rays(&inputs, order);
        let mut roots_ms = Vec::new();
        let mut recover_ms = Vec::new();
        let mut sum_ms = Vec::new();
        let mut verified = false;
        for iteration in 0..WARMUPS + MEASURED {
            let (results, profile) = profiled.solve_essential_profiled(&left, &right)?;
            ensure!(results.len() == inputs.len(), "missing trials for {name}");
            // Undo the permutation and restore the positional trial index, then
            // require the exact historical bits. Reordering must not change any
            // trial's own result.
            let mut restored = vec![None; results.len()];
            for (position, mut result) in results.into_iter().enumerate() {
                let original = order[position];
                result.trial = original;
                restored[original] = Some(result);
            }
            let restored: Vec<_> = restored
                .into_iter()
                .map(|r| r.expect("permutation is a bijection"))
                .collect();
            ensure!(
                shared::gpu_bits(&restored) == bits,
                "order {name} changed per-trial results"
            );
            verified = true;
            let passes = profile
                .gpu_pass_seconds
                .expect("timestamp context must report passes");
            if iteration >= WARMUPS {
                roots_ms.push(passes[ROOTS_PASS].unwrap_or(0.0) * 1000.0);
                recover_ms.push(passes[8].unwrap_or(0.0) * 1000.0);
                sum_ms.push(passes.iter().map(|p| p.unwrap_or(0.0)).sum::<f64>() * 1000.0);
            }
        }
        eprintln!(
            "order={name:<26} roots_median={:8.3}ms nine_sum_median={:8.3}ms",
            median(&roots_ms),
            median(&sum_ms)
        );
        orders.push(json!({"order":name,"per_trial_results_identical":verified,
            "roots_ms":{"median":median(&roots_ms),"samples":roots_ms},
            "recover_ms":{"median":median(&recover_ms),"samples":recover_ms},
            "nine_kernel_sum_ms":{"median":median(&sum_ms),"samples":sum_ms}}));
    }

    let report = json!({"schema":"five-point-roots-attribution-v1","source_commit":"3ab9c09",
        "database":args.database,"device":device,
        "input_digest":digest,"trials":inputs.len(),"pairs":pairs.len(),
        "unique_images":shared::unique_images(&pairs).len(),
        "gpu_signature":signature,"cpu_signature":cpu_signature,
        "iteration_cap":ITERATION_CAP,"simd_width":SIMD_WIDTH,
        "iterations":{"histogram":histogram,"entered_loop_trials":entered,
            "at_cap_trials":at_cap,"total_lane_iterations":total_iterations,
            "mean_over_entered":total_iterations as f64/entered.max(1) as f64,
            "p50":percentile(&ascending,0.5),"p90":percentile(&ascending,0.9),
            "p99":percentile(&ascending,0.99),"p999":percentile(&ascending,0.999),
            "max":ascending[ascending.len()-1]},
        "divergence_model":{"widths":widths,
            "simd32_ideal":ideal32,"simd32_lockstep":lockstep32,"simd32_sorted":sorted32,
            "simd32_waste_ratio":lockstep32 as f64/ideal32 as f64,
            "simd32_sorted_speedup_bound":lockstep32 as f64/sorted32 as f64,
            "groups":group_count,"groups_containing_a_capped_trial":groups_with_cap,
            "fraction_of_groups_capped":groups_with_cap as f64/group_count as f64},
        "q1_classification":{"primary_category":categories,"slot_status_totals":slot_statuses,
            "rejection_totals":rejection_totals,
            "unconverged_trials":unconverged_trials,
            "unconverged_trials_below_cap":unconverged_below_cap,
            "unconverged_trials_at_cap":unconverged_at_cap,
            "unconverged_slots_below_cap":unconverged_slots_below_cap,
            "trials_at_cap_still_producing_models":cap_with_models,
            "notconverged_split":{
                "real_axis_final_eval_gate":real_axis_gate,
                "after_store_below_cap_so_polish_only":after_store_below_cap,
                "after_store_at_cap_polish_or_done_flag":after_store_at_cap,
                "after_store_inside_realness_threshold":after_store_within_realness},
            "root_budget":{"degree_sum":baseline.iter().map(|t|t.degree).sum::<usize>(),
                "accepted":accepted_slots,"complex_status":complex_slots,
                "real_axis_gate":real_axis_gate,
                "accepted_plus_complex_plus_real_axis_gate":
                    accepted_slots+complex_slots+real_axis_gate}},
        "lost_solutions":{"definition":"CPU f64 has at least one model and GPU has none",
            "count":lost_iterations.len(),"primary_category":lost,
            "iteration_histogram":lost_histogram},
        "pass_names":FIVE_POINT_PASS_NAMES,"orders":orders,
        "divergence_contract":"The three runs differ only in the order of the input trials. Each run reverses the permutation, restores the positional trial index and requires the exact historical decoded bit signature, so any timing difference is scheduling, not arithmetic. The sorted order is an oracle built from a previous run's measured iteration counts; it bounds what any difficulty-aware grouping could recover and is not an implementable policy. Kernel intervals come from a timestamp-enabled context and overlap host wait; they are not additive with host phases.",
        "classification_contract":"Slot status NotConverged is emitted by three paths in five_point_recovery.wgsl: the final evaluation gate, the loop's own done[] flag, and the real Newton polish check. They are separated here without changing the shader, using two facts. First, the final evaluation gate rejects before the root and backward error are stored, so its slots still hold the zero-initialized buffer value, while the other two gates run after that store. Second, a trial whose loop stopped before the cap left through all_done, which requires done[i] for every root, so the done[] path cannot have fired for it. Note that the final evaluation gate evaluates the polynomial at the real part of the root alone, discarding the imaginary part, so a legitimately complex root fails it and is labelled NotConverged before ever reaching the dedicated realness test that would label it Complex. The at-cap bucket is defined by root_iterations == 384, which also captures the rare trial that converged exactly on the last iteration; that only makes the below-cap counts conservative.",
        "scope":"Independent GPU f32 candidate generation measurement only. No solver, shader or protected harness change. CPU f64 is an offline reference, not a runtime fallback. Quality is not equivalent and nothing here changes it."});
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    eprintln!("wrote {}", args.output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_and_percentiles_are_exact_at_the_edges() {
        assert_eq!(bucket(0), "0 (never entered the loop)");
        assert_eq!(bucket(1), "1-9");
        assert_eq!(bucket(9), "1-9");
        assert_eq!(bucket(10), "10-19");
        assert_eq!(bucket(383), "200-383");
        assert_eq!(bucket(ITERATION_CAP), "384 (cap)");
        // Nearest-rank on a 0-based index, so an odd length lands exactly.
        let ascending: Vec<usize> = (1..=101).collect();
        assert_eq!(percentile(&ascending, 0.5), 51);
        assert_eq!(percentile(&ascending, 0.0), 1);
        assert_eq!(percentile(&ascending, 1.0), 101);
        // Even length has no exact midpoint; the rank rounds up.
        assert_eq!(percentile(&(1..=100).collect::<Vec<_>>(), 0.5), 51);
    }

    #[test]
    fn lockstep_model_charges_each_group_its_maximum() {
        // One slow lane per group of four forces the whole group to its maximum.
        let iterations = [100, 1, 1, 1, 100, 1, 1, 1];
        let (ideal, lockstep, sorted) = divergence(&iterations, 4);
        assert_eq!(ideal, 206);
        assert_eq!(lockstep, 800);
        // Sorted: [100,100,1,1] and [1,1,1,1] cost 400 + 4.
        assert_eq!(sorted, 404);
        assert!(lockstep > sorted && sorted > ideal);
        // A homogeneous workload wastes nothing.
        let flat = [7usize; 8];
        let (ideal, lockstep, sorted) = divergence(&flat, 4);
        assert_eq!((ideal, lockstep, sorted), (56, 56, 56));
        // A partial trailing group is charged only its own lanes.
        let (_, lockstep, _) = divergence(&[5, 5, 5], 4);
        assert_eq!(lockstep, 15);
    }

    #[test]
    fn divergence_is_order_sensitive_but_ideal_is_not() {
        let clustered = [9, 9, 1, 1];
        let spread = [9, 1, 9, 1];
        assert_eq!(divergence(&clustered, 2).0, divergence(&spread, 2).0);
        assert_eq!(divergence(&clustered, 2).1, 20);
        assert_eq!(divergence(&spread, 2).1, 36);
    }

    /// The zero-initialized root is what distinguishes the pre-store gate, so
    /// pin the exact predicate the report relies on.
    #[test]
    fn notconverged_split_keys_on_the_unwritten_root() {
        let pre_store = ([0.0f32, 0.0], 0.0f32);
        let post_store = ([0.31f32, 0.94], 4e-7f32);
        let gate = |(root, backward): ([f32; 2], f32)| root == [0.0, 0.0] && backward == 0.0;
        assert!(gate(pre_store));
        assert!(!gate(post_store));
        // A stored root on the real axis still counts as stored.
        assert!(!gate(([0.5, 0.0], 1e-8)));
        // Realness uses the shader's relative form, not an absolute bound.
        let real = |root: [f32; 2]| root[1].abs() <= 2e-4 * (1.0 + root[0].abs());
        assert!(real([1000.0, 0.1]));
        assert!(!real([0.0, 0.1]));
    }

    #[test]
    fn permutation_reordering_is_a_bijection_over_the_workset() {
        let iterations = [5usize, 300, 5, 384, 1, 384];
        let mut order: Vec<usize> = (0..iterations.len()).collect();
        order.sort_by_key(|&i| (std::cmp::Reverse(iterations[i]), i));
        assert_eq!(order, vec![3, 5, 1, 0, 2, 4]);
        let mut seen = order.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..iterations.len()).collect::<Vec<_>>());
        // Ties keep ascending original index, so the order is deterministic.
        assert!(order.windows(2).all(|w| iterations[w[0]] > iterations[w[1]]
            || (iterations[w[0]] == iterations[w[1]] && w[0] < w[1])));
    }
}
