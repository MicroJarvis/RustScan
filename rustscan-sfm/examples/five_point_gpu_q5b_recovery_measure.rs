//! Q5b offline measure: matched-root RecoveryFailed subtypes vs B quality.
//!
//! Reads existing Q4 verified stages.bin + bits.json + trials.json.
//! No GPU / no shader changes. Reconstructs same-basis f64 roots from the
//! staged GPU basis and greedily matches GPU slots (same algorithm as Q4).
#![allow(clippy::needless_range_loop)]

use anyhow::{ensure, Result};
use clap::Parser;
use rayon::prelude::*;
use rustscan_sfm::{
    five_point::{essential_reference_from_basis, EssentialBasis},
    gpu::FivePointSlotStatus,
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
};

const N_TRIALS: usize = 65_024;
const STAGE_FLOATS: usize = 386; // basis36 + A200 + solve100 + B39 + coeff11
const B_SAME_BASIS: usize = 4;
/// CPU pipeline realness criterion: |im| <= 1e-10.
const REAL_IM: f64 = 1.0e-10;
/// No-loss B p90 from Q5 premeasure (tol 1e-3).
const GOOD_B_P90: f64 = 1.4936871013940632e-5;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    stages: PathBuf,
    #[arg(long)]
    bits: PathBuf,
    #[arg(long)]
    trials: PathBuf,
    #[arg(long, default_value_t = 1e-3)]
    match_tol: f64,
    /// B threshold for good-B (default = no-loss p90).
    #[arg(long, default_value_t = GOOD_B_P90)]
    good_b_threshold: f64,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum RecoverySubtype {
    NullVectorFailure,
    InvalidEssential,
}

fn status_from_u32(v: u32) -> Option<FivePointSlotStatus> {
    Some(match v {
        0 => FivePointSlotStatus::Unused,
        1 => FivePointSlotStatus::Accepted,
        2 => FivePointSlotStatus::Complex,
        3 => FivePointSlotStatus::RealAxisRejected,
        4 => FivePointSlotStatus::RootNotDone,
        5 => FivePointSlotStatus::PolishRejected,
        6 => FivePointSlotStatus::DuplicateRoot,
        7 => FivePointSlotStatus::NullVectorFailure,
        8 => FivePointSlotStatus::InvalidEssential,
        9 => FivePointSlotStatus::DuplicateModel,
        10 => FivePointSlotStatus::RealRoot,
        _ => return None,
    })
}

fn parse_slots(words: &[u32]) -> Vec<(FivePointSlotStatus, [f32; 2])> {
    let mut i = 15usize;
    let mut slots = Vec::with_capacity(10);
    while i + 9 <= words.len() && slots.len() < 10 {
        let status = status_from_u32(words[i + 1]).unwrap_or(FivePointSlotStatus::Unused);
        let root0 = f32::from_bits(words[i + 2]);
        let root1 = f32::from_bits(words[i + 3]);
        let has_e = words[i + 8] != 0;
        i += 9;
        if has_e {
            i += 9;
        }
        slots.push((status, [root0, root1]));
    }
    slots
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !args.output.exists(),
        "refusing to overwrite {}",
        args.output.display()
    );
    ensure!(
        args.match_tol.is_finite() && args.match_tol > 0.0,
        "match_tol must be positive finite"
    );

    let bits: Vec<Vec<u32>> = serde_json::from_reader(BufReader::new(File::open(&args.bits)?))?;
    let trials: Vec<serde_json::Value> =
        serde_json::from_reader(BufReader::new(File::open(&args.trials)?))?;
    ensure!(bits.len() == N_TRIALS && trials.len() == N_TRIALS);

    let mut stage_bytes = Vec::new();
    File::open(&args.stages)?.read_to_end(&mut stage_bytes)?;
    let expected = N_TRIALS * STAGE_FLOATS * 4;
    ensure!(
        stage_bytes.len() == expected,
        "stages.bin size {} != expected {}",
        stage_bytes.len(),
        expected
    );

    #[derive(Clone, Default)]
    struct Acc {
        recovery_failed: usize,
        null_vec: usize,
        invalid_ess: usize,
        // cross-tab: subtype × B bucket
        xtab: BTreeMap<(RecoverySubtype, &'static str), usize>,
        // trial-level: any matched RecoveryFailed on good-B
        trials_any_rf_good_b: usize,
        trials_any_rf_bad_b: usize,
        trials_any_rf_nonfinite: usize,
        // absolute GPU slots (sanity vs premeasure)
        abs_null_slots: usize,
        abs_inv_slots: usize,
        class_counts: BTreeMap<&'static str, usize>,
        // quality proxy: matched RF on good-B that beats nothing here — just counts
        matched_rf_good_b_null: usize,
        matched_rf_good_b_inv: usize,
        matched_rf_bad_b_null: usize,
        matched_rf_bad_b_inv: usize,
        matched_rf_nonfinite_null: usize,
        matched_rf_nonfinite_inv: usize,
        // B stats on matched-RF trials
        b_matched_rf: Vec<f64>,
        b_matched_rf_null: Vec<f64>,
        b_matched_rf_inv: Vec<f64>,
        b_matched_rf_good: Vec<f64>,
    }

    let results: Vec<Acc> = (0..N_TRIALS)
        .into_par_iter()
        .map(|i| {
            let mut acc = Acc::default();
            let words = &bits[i];
            let slots = parse_slots(words);
            for (st, _) in &slots {
                match st {
                    FivePointSlotStatus::NullVectorFailure => acc.abs_null_slots += 1,
                    FivePointSlotStatus::InvalidEssential => acc.abs_inv_slots += 1,
                    _ => {}
                }
            }

            let b = trials[i]["stage_errors"][B_SAME_BASIS]
                .as_f64()
                .unwrap_or(f64::NAN);
            let b_lab = if !b.is_finite() {
                "nonfinite_B"
            } else if b <= args.good_b_threshold {
                "good_B"
            } else {
                "bad_B"
            };

            let off = i * STAGE_FLOATS * 4;
            let mut basis = [0.0f32; 36];
            let mut all_zero = true;
            for k in 0..36 {
                let bits = u32::from_le_bytes([
                    stage_bytes[off + k * 4],
                    stage_bytes[off + k * 4 + 1],
                    stage_bytes[off + k * 4 + 2],
                    stage_bytes[off + k * 4 + 3],
                ]);
                basis[k] = f32::from_bits(bits);
                if basis[k] != 0.0 {
                    all_zero = false;
                }
            }
            if all_zero {
                // Rank-deficient / no basis — no same-basis reference roots.
                return acc;
            }

            let eb = EssentialBasis::from_column_slice(&basis.map(f64::from));
            let reference = essential_reference_from_basis(&eb);
            let real_roots: Vec<f64> = reference
                .roots
                .iter()
                .filter(|r| r.im.abs() <= REAL_IM)
                .map(|r| r.re)
                .collect();

            let mut slot_used = [false; 10];
            let mut trial_has_rf = false;
            let mut trial_rf_null = false;
            let mut trial_rf_inv = false;

            for &z in &real_roots {
                let scale = z.abs().max(1.0);
                let mut best: Option<(usize, f64)> = None;
                for (s, (status, root)) in slots.iter().enumerate() {
                    if slot_used[s] || *status == FivePointSlotStatus::Unused {
                        continue;
                    }
                    let d = ((f64::from(root[0]) - z).powi(2) + f64::from(root[1]).powi(2)).sqrt()
                        / scale;
                    if best.map_or(true, |(_, bd)| d < bd) {
                        best = Some((s, d));
                    }
                }
                let class = match best {
                    Some((s, d)) if d <= args.match_tol => {
                        slot_used[s] = true;
                        match slots[s].0 {
                            FivePointSlotStatus::Accepted => "Covered",
                            FivePointSlotStatus::RealAxisRejected => "GateRealAxis",
                            FivePointSlotStatus::PolishRejected => "GatePolish",
                            FivePointSlotStatus::RootNotDone => "NotDone",
                            FivePointSlotStatus::NullVectorFailure => {
                                acc.recovery_failed += 1;
                                acc.null_vec += 1;
                                trial_has_rf = true;
                                trial_rf_null = true;
                                *acc.xtab
                                    .entry((RecoverySubtype::NullVectorFailure, b_lab))
                                    .or_default() += 1;
                                match b_lab {
                                    "good_B" => acc.matched_rf_good_b_null += 1,
                                    "bad_B" => acc.matched_rf_bad_b_null += 1,
                                    _ => acc.matched_rf_nonfinite_null += 1,
                                }
                                "RecoveryFailed"
                            }
                            FivePointSlotStatus::InvalidEssential => {
                                acc.recovery_failed += 1;
                                acc.invalid_ess += 1;
                                trial_has_rf = true;
                                trial_rf_inv = true;
                                *acc.xtab
                                    .entry((RecoverySubtype::InvalidEssential, b_lab))
                                    .or_default() += 1;
                                match b_lab {
                                    "good_B" => acc.matched_rf_good_b_inv += 1,
                                    "bad_B" => acc.matched_rf_bad_b_inv += 1,
                                    _ => acc.matched_rf_nonfinite_inv += 1,
                                }
                                "RecoveryFailed"
                            }
                            FivePointSlotStatus::Complex => "ComplexMisclassified",
                            FivePointSlotStatus::DuplicateRoot
                            | FivePointSlotStatus::DuplicateModel => "Duplicate",
                            FivePointSlotStatus::Unused | FivePointSlotStatus::RealRoot => {
                                "MissingNoSlot"
                            }
                        }
                    }
                    _ => "MissingNoSlot",
                };
                *acc.class_counts.entry(class).or_default() += 1;
            }

            if trial_has_rf {
                acc.b_matched_rf.push(b);
                if trial_rf_null {
                    acc.b_matched_rf_null.push(b);
                }
                if trial_rf_inv {
                    acc.b_matched_rf_inv.push(b);
                }
                match b_lab {
                    "good_B" => {
                        acc.trials_any_rf_good_b += 1;
                        acc.b_matched_rf_good.push(b);
                    }
                    "bad_B" => acc.trials_any_rf_bad_b += 1,
                    _ => acc.trials_any_rf_nonfinite += 1,
                }
            }
            acc
        })
        .collect();

    let mut total = Acc::default();
    for a in results {
        total.recovery_failed += a.recovery_failed;
        total.null_vec += a.null_vec;
        total.invalid_ess += a.invalid_ess;
        total.trials_any_rf_good_b += a.trials_any_rf_good_b;
        total.trials_any_rf_bad_b += a.trials_any_rf_bad_b;
        total.trials_any_rf_nonfinite += a.trials_any_rf_nonfinite;
        total.abs_null_slots += a.abs_null_slots;
        total.abs_inv_slots += a.abs_inv_slots;
        total.matched_rf_good_b_null += a.matched_rf_good_b_null;
        total.matched_rf_good_b_inv += a.matched_rf_good_b_inv;
        total.matched_rf_bad_b_null += a.matched_rf_bad_b_null;
        total.matched_rf_bad_b_inv += a.matched_rf_bad_b_inv;
        total.matched_rf_nonfinite_null += a.matched_rf_nonfinite_null;
        total.matched_rf_nonfinite_inv += a.matched_rf_nonfinite_inv;
        for (k, v) in a.xtab {
            *total.xtab.entry(k).or_default() += v;
        }
        for (k, v) in a.class_counts {
            *total.class_counts.entry(k).or_default() += v;
        }
        total.b_matched_rf.extend(a.b_matched_rf);
        total.b_matched_rf_null.extend(a.b_matched_rf_null);
        total.b_matched_rf_inv.extend(a.b_matched_rf_inv);
        total.b_matched_rf_good.extend(a.b_matched_rf_good);
    }

    fn pct(xs: &[f64], p: f64) -> Option<f64> {
        let mut v: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
        if v.is_empty() {
            return None;
        }
        v.sort_by(f64::total_cmp);
        Some(v[((v.len() - 1) as f64 * p).round() as usize])
    }
    fn summarize(xs: &[f64]) -> serde_json::Value {
        let finite: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
        json!({
            "n": finite.len(),
            "nonfinite": xs.len() - finite.len(),
            "p50": pct(&finite, 0.5),
            "p90": pct(&finite, 0.9),
            "p99": pct(&finite, 0.99),
        })
    }

    let good_b_rf = total.matched_rf_good_b_null + total.matched_rf_good_b_inv;
    let bad_b_rf = total.matched_rf_bad_b_null + total.matched_rf_bad_b_inv;
    let stop_recovery = good_b_rf < 500; // tiny residue → don't hack recovery

    let xtab_json: BTreeMap<String, usize> = total
        .xtab
        .into_iter()
        .map(|((sub, b), n)| {
            let s = match sub {
                RecoverySubtype::NullVectorFailure => "NullVectorFailure",
                RecoverySubtype::InvalidEssential => "InvalidEssential",
            };
            (format!("{s}×{b}"), n)
        })
        .collect();

    let out = json!({
        "round": "Q5b-offline-measure",
        "kind": "offline-only",
        "inputs": {
            "stages": args.stages,
            "bits": args.bits,
            "trials": args.trials,
            "match_tol": args.match_tol,
            "good_b_threshold": args.good_b_threshold,
            "good_b_threshold_note": "default = no-loss B p90 from Q5 premeasure tol1e-3",
            "n_trials": N_TRIALS,
        },
        "absolute_gpu_slots": {
            "NullVectorFailure": total.abs_null_slots,
            "InvalidEssential": total.abs_inv_slots,
        },
        "root_classes_of_f64_real_roots": total.class_counts,
        "matched_RecoveryFailed": {
            "total": total.recovery_failed,
            "NullVectorFailure": total.null_vec,
            "InvalidEssential": total.invalid_ess,
            "cross_tab_subtype_x_B": xtab_json,
            "good_B_total": good_b_rf,
            "bad_B_total": bad_b_rf,
            "good_B_NullVectorFailure": total.matched_rf_good_b_null,
            "good_B_InvalidEssential": total.matched_rf_good_b_inv,
            "bad_B_NullVectorFailure": total.matched_rf_bad_b_null,
            "bad_B_InvalidEssential": total.matched_rf_bad_b_inv,
            "nonfinite_B_NullVectorFailure": total.matched_rf_nonfinite_null,
            "nonfinite_B_InvalidEssential": total.matched_rf_nonfinite_inv,
        },
        "trials_with_any_matched_RecoveryFailed": {
            "good_B": total.trials_any_rf_good_b,
            "bad_B": total.trials_any_rf_bad_b,
            "nonfinite_B": total.trials_any_rf_nonfinite,
        },
        "B_gpu_vs_f64_same_basis_on_matched_RF_trials": {
            "any_subtype": summarize(&total.b_matched_rf),
            "with_NullVectorFailure": summarize(&total.b_matched_rf_null),
            "with_InvalidEssential": summarize(&total.b_matched_rf_inv),
            "good_B_subset": summarize(&total.b_matched_rf_good),
        },
        "decision": {
            "stop_recovery_hack": stop_recovery,
            "threshold_good_B_matched_RF": 500,
            "rationale": if stop_recovery {
                format!(
                    "good-B matched RecoveryFailed residue is {good_b_rf} (<500); \
                     prefer P4 or mixed-precision over 3×3 recovery hack"
                )
            } else {
                format!(
                    "good-B matched RecoveryFailed residue is {good_b_rf} (≥500); \
                     proceed to single-variable 3×3 recovery (do not loosen gates)"
                )
            },
        },
    });

    std::fs::write(&args.output, serde_json::to_vec_pretty(&out)?)?;
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
