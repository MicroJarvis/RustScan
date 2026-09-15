//! Q1 diagnostic fix verification.
//!
//! Proves the three former `NotConverged` paths now have distinct slot status
//! codes, that candidate models are bit-identical to the pre-fix baseline
//! (model signature), records diagnostic signature v2, cross-checks the new
//! counts against the previous zero-root inference, and writes stratified
//! regression fixtures for the known lost-solution populations.
//!
//! Not RANSAC, not the 960-image pipeline, no CPU runtime fallback.
#[allow(dead_code)]
#[path = "five_point_gpu_capacity_replay.rs"]
mod shared;

use anyhow::{ensure, Result};
use clap::Parser;
use nalgebra::{DMatrix, Vector3};
use rayon::{prelude::*, ThreadPoolBuilder};
use rustsfm::gpu::{
    FivePointSlotStatus, FivePointStatus, FivePointTrialResult, WgpuContext, WgpuFivePointF32,
};
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf};

const INPUT: &str = "af07459d637b5b8c2b20c76d3ef103f058505fe0e8307578ee7e8750f3e230c5";
/// Pre-Q1 model signature: essentials + model_count only, captured before the
/// status-code change. Must remain identical.
const MODEL_SIGNATURE: &str = "030f43068d9e8f78961142915162f20bef4ecb909d49d247a4f766a2ab5412df";
/// Pre-Q1 full diagnostic signature (historical; expected to change).
const DIAGNOSTIC_SIGNATURE_V1: &str =
    "1f6f3f8d88a17e4e1c42abfc85aad44a66c4b864091805594e086ca80bf9b339";
const CPU_SIGNATURE: &str = "9e6764b41602d0005710187d2558d046f1a843c5f873b9542dd7b4f6c29c563a";
const ITERATION_CAP: usize = 384;
/// Zero-root inference from the previous measurement round.
const EXPECTED_REAL_AXIS: usize = 276_906;
const EXPECTED_POLISH_BELOW_CAP: usize = 130;
const EXPECTED_AFTER_STORE_AT_CAP: usize = 20_786;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    database: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    fixtures: PathBuf,
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
    } else if r >= 1e-3 {
        "[1e-3,1e-2)"
    } else if r >= 1e-4 {
        "[1e-4,1e-3)"
    } else if r >= 1e-5 {
        "[1e-5,1e-4)"
    } else if r <= 1e-7 {
        "<=1e-7"
    } else {
        "(1e-7,1e-5)"
    }
}

fn stage(t: &FivePointTrialResult) -> &'static str {
    match t.upstream_status {
        FivePointStatus::RankDeficient => "nullspace_rank_deficient",
        FivePointStatus::NotConverged => "nullspace_not_converged",
        FivePointStatus::InvalidBasis => "nullspace_invalid_basis",
        FivePointStatus::SingularElimination => "algebra_singular",
        FivePointStatus::NonFinite => "algebra_nonfinite",
        FivePointStatus::Success if t.model_count == 0 => "roots_all_rejected",
        FivePointStatus::Success => "produced_models",
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.output.exists(), "refusing to overwrite experiment");
    ensure!(!args.fixtures.exists(), "refusing to overwrite fixtures");
    let (inputs, _pairs, _, _) = shared::cpu::load_with_observations(&shared::cpu::Args {
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
    let results = shared::gpu_replay(&gpu, &inputs, inputs.len())?;
    let model_sig = shared::model_signature(&results);
    ensure!(
        model_sig == MODEL_SIGNATURE,
        "MODEL SIGNATURE CHANGED: {model_sig} (candidates must be bit-identical)"
    );
    let diagnostic_v2 = shared::signature(&shared::gpu_bits(&results));
    ensure!(
        diagnostic_v2 != DIAGNOSTIC_SIGNATURE_V1,
        "diagnostic signature unexpectedly unchanged"
    );

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
    ensure!(hasher.finalize().to_hex().to_string() == CPU_SIGNATURE);

    // ---- Status totals and consistency ------------------------------------
    let mut slot_totals: BTreeMap<String, usize> = BTreeMap::new();
    let mut real_axis = 0usize;
    let mut root_not_done = 0usize;
    let mut polish = 0usize;
    let mut polish_below_cap = 0usize;
    let mut after_store_at_cap = 0usize;
    let mut header_unconverged = 0usize;
    let mut split_mismatch = 0usize;
    for t in &results {
        header_unconverged += t.unconverged;
        let split = t.real_axis_rejected + t.root_not_done + t.polish_rejected;
        if split != t.unconverged {
            split_mismatch += 1;
        }
        real_axis += t.real_axis_rejected;
        root_not_done += t.root_not_done;
        polish += t.polish_rejected;
        if t.root_iterations < ITERATION_CAP {
            polish_below_cap += t.polish_rejected;
        }
        for s in &t.slots {
            *slot_totals.entry(format!("{:?}", s.status)).or_default() += 1;
            if matches!(
                s.status,
                FivePointSlotStatus::RootNotDone | FivePointSlotStatus::PolishRejected
            ) && t.root_iterations == ITERATION_CAP
            {
                after_store_at_cap += 1;
            }
        }
    }
    ensure!(
        split_mismatch == 0,
        "header unconverged disagrees with slot split"
    );
    ensure!(
        real_axis == EXPECTED_REAL_AXIS,
        "real_axis {real_axis} != inferred {EXPECTED_REAL_AXIS}"
    );
    ensure!(
        polish_below_cap == EXPECTED_POLISH_BELOW_CAP,
        "polish_below_cap {polish_below_cap} != inferred {EXPECTED_POLISH_BELOW_CAP}"
    );
    ensure!(
        after_store_at_cap == EXPECTED_AFTER_STORE_AT_CAP,
        "after_store_at_cap {after_store_at_cap} != inferred {EXPECTED_AFTER_STORE_AT_CAP}"
    );
    // The previous inference could not separate RootNotDone from PolishRejected
    // at the cap; their sum is what was known.
    ensure!(root_not_done + (polish - polish_below_cap) == EXPECTED_AFTER_STORE_AT_CAP);

    // ---- Lost-solution split by new codes ---------------------------------
    let lost: Vec<usize> = (0..inputs.len())
        .filter(|&i| !cpu[i].is_empty() && results[i].model_count == 0)
        .collect();
    let lost_roots: Vec<usize> = lost
        .iter()
        .copied()
        .filter(|&i| stage(&results[i]) == "roots_all_rejected")
        .collect();
    let lost_rd: Vec<usize> = lost
        .iter()
        .copied()
        .filter(|&i| results[i].upstream_status == FivePointStatus::RankDeficient)
        .collect();
    ensure!(lost.len() == 11967 && lost_rd.len() == 10028 && lost_roots.len() == 1938);

    let mut lost_roots_slots: BTreeMap<String, usize> = BTreeMap::new();
    let mut lost_roots_unconv: BTreeMap<&str, usize> = BTreeMap::new();
    for &i in &lost_roots {
        for s in &results[i].slots {
            *lost_roots_slots
                .entry(format!("{:?}", s.status))
                .or_default() += 1;
            if s.status.is_unconverged() {
                *lost_roots_unconv
                    .entry(match s.status {
                        FivePointSlotStatus::RealAxisRejected => "real_axis_rejected",
                        FivePointSlotStatus::RootNotDone => "root_not_done",
                        FivePointSlotStatus::PolishRejected => "polish_rejected",
                        _ => unreachable!(),
                    })
                    .or_default() += 1;
            }
        }
    }

    // ---- Stratified fixtures ----------------------------------------------
    let ratios: Vec<f64> = pool.install(|| {
        lost_rd
            .par_iter()
            .map(|&i| sigma5_over_sigma1(&constraint(&inputs[i].left, &inputs[i].right)))
            .collect()
    });
    let mut by_bucket: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (idx, &i) in lost_rd.iter().enumerate() {
        by_bucket
            .entry(ratio_bucket(ratios[idx]))
            .or_default()
            .push(i);
    }
    // Deterministic stratified sample: first / middle / last of each bucket,
    // capped so the fixture file stays small.
    let mut fixture_trials = Vec::new();
    for (bucket, ids) in &by_bucket {
        let n = ids.len();
        let picks = [0, n / 2, n.saturating_sub(1)];
        for &p in &picks {
            let i = ids[p];
            fixture_trials.push(json!({
                "kind":"nullspace_rank_deficient",
                "bucket":bucket,
                "trial":i,
                "pair":i/512,
                "sigma5_over_sigma1":ratios[ids.iter().position(|&x|x==i).unwrap()],
                "gpu_rank":results[i].rank,
                "indices":inputs[i].indices,
                "left":inputs[i].left.map(|r|[r.x,r.y,r.z]),
                "right":inputs[i].right.map(|r|[r.x,r.y,r.z]),
            }));
        }
    }
    // Sample root-stage losses by the new status codes.
    let root_picks: [(&str, usize); 5] = [
        (
            "real_axis_rejected",
            lost_roots
                .iter()
                .copied()
                .find(|&i| results[i].real_axis_rejected > 0)
                .unwrap_or(usize::MAX),
        ),
        (
            "root_not_done",
            lost_roots
                .iter()
                .copied()
                .find(|&i| results[i].root_not_done > 0)
                .unwrap_or(usize::MAX),
        ),
        (
            "polish_rejected",
            lost_roots
                .iter()
                .copied()
                .find(|&i| results[i].polish_rejected > 0)
                .unwrap_or(usize::MAX),
        ),
        (
            "invalid_essential",
            lost_roots
                .iter()
                .copied()
                .find(|&i| {
                    results[i]
                        .slots
                        .iter()
                        .any(|s| s.status == FivePointSlotStatus::InvalidEssential)
                })
                .unwrap_or(usize::MAX),
        ),
        (
            "null_vector_failure",
            lost_roots
                .iter()
                .copied()
                .find(|&i| {
                    results[i]
                        .slots
                        .iter()
                        .any(|s| s.status == FivePointSlotStatus::NullVectorFailure)
                })
                .unwrap_or(usize::MAX),
        ),
    ];
    for &(label, i) in &root_picks {
        if i == usize::MAX {
            continue;
        }
        fixture_trials.push(json!({
            "kind":"roots_all_rejected",
            "label":label,
            "trial":i,
            "pair":i/512,
            "root_iterations":results[i].root_iterations,
            "real_axis_rejected":results[i].real_axis_rejected,
            "root_not_done":results[i].root_not_done,
            "polish_rejected":results[i].polish_rejected,
            "slot_statuses":results[i].slots.iter().map(|s|format!("{:?}",s.status)).collect::<Vec<_>>(),
            "indices":inputs[i].indices,
            "left":inputs[i].left.map(|r|[r.x,r.y,r.z]),
            "right":inputs[i].right.map(|r|[r.x,r.y,r.z]),
        }));
    }

    let report = json!({
        "schema":"five-point-q1-diagnostics-v1",
        "source_commit":"3ab9c09",
        "database":args.database,
        "device":device,
        "input_digest":INPUT,
        "model_signature":model_sig,
        "model_signature_unchanged":true,
        "diagnostic_signature_v1_historical":DIAGNOSTIC_SIGNATURE_V1,
        "diagnostic_signature_v2":diagnostic_v2,
        "cpu_signature":CPU_SIGNATURE,
        "slot_status_totals":slot_totals,
        "unconverged_split":{
            "header_total":header_unconverged,
            "real_axis_rejected":real_axis,
            "root_not_done":root_not_done,
            "polish_rejected":polish,
            "polish_rejected_below_cap":polish_below_cap,
            "root_not_done_or_polish_at_cap":after_store_at_cap,
            "cross_check":{
                "real_axis_matches_inference":real_axis==EXPECTED_REAL_AXIS,
                "polish_below_cap_matches_inference":polish_below_cap==EXPECTED_POLISH_BELOW_CAP,
                "at_cap_sum_matches_inference":after_store_at_cap==EXPECTED_AFTER_STORE_AT_CAP
            }
        },
        "lost_solutions":{
            "total":lost.len(),
            "nullspace_rank_deficient":lost_rd.len(),
            "roots_all_rejected":lost_roots.len(),
            "roots_all_rejected_slot_statuses":lost_roots_slots,
            "roots_all_rejected_unconverged_split":lost_roots_unconv
        },
        "nullspace_fixture_buckets":by_bucket.iter().map(|(k,v)|json!({"bucket":k,"count":v.len()})).collect::<Vec<_>>(),
        "fixture_count":fixture_trials.len(),
        "status_codes":{
            "3.0":"RealAxisRejected",
            "7.0":"RootNotDone",
            "8.0":"PolishRejected",
            "note":"Header unconverged still sums all three; split is derived from slot statuses at decode."
        },
        "scope":"Q1 diagnostic fix only. Thresholds and rejection logic unchanged. Model signature must match the pre-fix capture."
    });
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    std::fs::write(
        &args.fixtures,
        serde_json::to_vec_pretty(&json!({
            "schema":"five-point-q1-fixtures-v1",
            "model_signature":model_sig,
            "diagnostic_signature_v2":diagnostic_v2,
            "trials":fixture_trials
        }))?,
    )?;
    eprintln!(
        "model_sig=ok diagnostic_v2={diagnostic_v2} real_axis={real_axis} root_not_done={root_not_done} polish={polish} polish_below_cap={polish_below_cap}"
    );
    eprintln!(
        "wrote {} and {}",
        args.output.display(),
        args.fixtures.display()
    );
    Ok(())
}
