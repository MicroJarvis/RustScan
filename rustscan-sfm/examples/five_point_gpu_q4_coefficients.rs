//! Recovery/roots-stage lost-model attribution (pure measurement).
//!
//! Key design: replay the f64 elimination → polynomial → roots → recovery
//! chain **on the GPU's own nullspace basis**, so basis differences are
//! separated from downstream f32 error. Every model in the CPU−GPU gap is
//! attributed to exactly one class:
//!
//!   basis/gate  — CPU's own basis yields models the GPU basis does not
//!   coeff       — GPU f32 polynomial coefficients drifted from same-basis f64
//!   root classes — per f64 real root: gate-rejected / not-done / recovery
//!                 failure / complex-misclassified / missing
//!
//! Not RANSAC, not the 960-image pipeline, no CPU runtime fallback. The f64
//! replay is an offline reference only. No shader or solver changes here.
#[path = "q4/attribution.rs"]
mod attribution;
#[allow(dead_code)]
#[path = "five_point_gpu_capacity_replay.rs"]
mod shared;

use anyhow::{ensure, Context, Result};
use clap::Parser;
use nalgebra::{Matrix3, Vector3};
use rayon::{prelude::*, ThreadPoolBuilder};
use rustscan_sfm::{
    database::ColmapDatabase,
    five_point::{essential_reference_from_basis, EssentialBasis, FivePointBasisReference},
    gpu::{
        FivePointSlotStatus, FivePointStatus, FivePointTrialResult, WgpuContext, WgpuFivePointF32,
    },
    types::CameraModel,
};
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf};

const INPUT: &str = "af07459d637b5b8c2b20c76d3ef103f058505fe0e8307578ee7e8750f3e230c5";
const MODEL_SIGNATURE_Q2: &str = "472124c8f1a9c6299d659e2f835dc11c682ff8a593757f0b145f73454f51d6ea";
const DIAGNOSTIC_SIGNATURE_Q2: &str =
    "3d04bb3edaef183049e07d6c3841e296f963da3b279bcbd99d878e1dc090c050";
const CPU_SIGNATURE: &str = "9e6764b41602d0005710187d2558d046f1a843c5f873b9542dd7b4f6c29c563a";
const MAX_ERROR_PX: f64 = 4.0;
const TRIALS_PER_PAIR: usize = 512;
const STAGE_CHUNK: usize = 8192;
/// CPU pipeline realness criterion (five_point.rs): |im| <= 1e-10.
const REAL_IM: f64 = 1.0e-10;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    database: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// Root pairing tolerance, relative: generous vs f32 root precision,
    /// narrow vs typical root spacing; distance distribution audits it.
    #[arg(long, default_value_t = 1.0e-3)]
    match_tol: f64,
    /// Baseline enforces historical Q2 signatures; candidate enforces repeat stability instead.
    #[arg(long)]
    candidate: bool,
    /// Exact baseline per-trial sidecar, for a fixed loss cohort comparison.
    #[arg(long)]
    baseline_trials: Option<PathBuf>,
}

fn squared_sampson(x1: &Vector3<f64>, x2: &Vector3<f64>, e: &Matrix3<f64>) -> f64 {
    let ex1 = e * x1;
    let etx2 = e.transpose() * x2;
    let num = x2.dot(&ex1);
    let denom = ex1.x * ex1.x + ex1.y * ex1.y + etx2.x * etx2.x + etx2.y * etx2.y;
    if denom <= 1.0e-24 {
        f64::INFINITY
    } else {
        num * num / denom
    }
}

fn inliers(obs: &shared::cpu::PairObservations, e: &Matrix3<f64>, threshold: f64) -> usize {
    let limit = threshold.max(1.0e-12).powi(2);
    obs.left
        .iter()
        .zip(&obs.right)
        .filter(|(l, r)| {
            let v = squared_sampson(l, r, e);
            v.is_finite() && v <= limit
        })
        .count()
}

fn gpu_models(trial: &FivePointTrialResult) -> impl Iterator<Item = Matrix3<f64>> + '_ {
    trial
        .slots
        .iter()
        .filter_map(|s| s.essential)
        .map(|e| Matrix3::from_row_slice(&e.map(f64::from)))
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted[((sorted.len() - 1) as f64 * fraction).round() as usize]
}

fn err_bucket(e: f64) -> &'static str {
    if !e.is_finite() {
        "nonfinite"
    } else if e >= 1e-1 {
        ">=1e-1"
    } else if e >= 1e-2 {
        "[1e-2,1e-1)"
    } else if e >= 1e-3 {
        "[1e-3,1e-2)"
    } else if e >= 1e-4 {
        "[1e-4,1e-3)"
    } else if e >= 1e-5 {
        "[1e-5,1e-4)"
    } else if e >= 1e-6 {
        "[1e-6,1e-5)"
    } else {
        "<1e-6"
    }
}

/// One f64-reference real root, attributed to a single downstream class.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
enum RootClass {
    Covered,
    GateRealAxis,
    GatePolish,
    NotDone,
    RecoveryFailed,
    ComplexMisclassified,
    Duplicate,
    MissingNoSlot,
}

fn classify(status: FivePointSlotStatus) -> RootClass {
    match status {
        FivePointSlotStatus::Accepted => RootClass::Covered,
        FivePointSlotStatus::RealAxisRejected => RootClass::GateRealAxis,
        FivePointSlotStatus::PolishRejected => RootClass::GatePolish,
        FivePointSlotStatus::RootNotDone => RootClass::NotDone,
        FivePointSlotStatus::NullVectorFailure | FivePointSlotStatus::InvalidEssential => {
            RootClass::RecoveryFailed
        }
        FivePointSlotStatus::Complex => RootClass::ComplexMisclassified,
        FivePointSlotStatus::DuplicateRoot | FivePointSlotStatus::DuplicateModel => {
            RootClass::Duplicate
        }
        FivePointSlotStatus::Unused | FivePointSlotStatus::RealRoot => RootClass::MissingNoSlot,
    }
}

struct TrialLedger {
    /// Same-basis f64 reference model count (0 when basis missing).
    ref_models: usize,
    /// Max-abs relative coefficient error vs same-basis f64 (NaN when unavailable).
    coeff_err: f64,
    /// Per-f64-real-root classes.
    classes: Vec<RootClass>,
    /// Match distances |gpu_root − f64_root| (relative) for audited pairs.
    match_dists: Vec<f64>,
    /// GPU accepted models whose root has no f64 real-root counterpart.
    spurious_gpu: usize,
    /// f64 reference models the GPU did not produce, with their class.
    missed: Vec<(RootClass, Matrix3<f64>)>,
}

fn attribute_trial(
    reference: &FivePointBasisReference,
    gpu: &FivePointTrialResult,
    match_tol: f64,
) -> TrialLedger {
    let real_roots: Vec<f64> = reference
        .roots
        .iter()
        .filter(|r| r.im.abs() <= REAL_IM)
        .map(|r| r.re)
        .collect();
    // Greedy nearest matching, each GPU slot used at most once. Slots with
    // Unused status carry no root information and are excluded.
    let mut slot_used = [false; 10];
    let mut classes = Vec::with_capacity(real_roots.len());
    let mut match_dists = Vec::new();
    let mut matched_slots: Vec<Option<usize>> = Vec::with_capacity(real_roots.len());
    for &z in &real_roots {
        let scale = z.abs().max(1.0);
        let mut best: Option<(usize, f64)> = None;
        for (s, slot) in gpu.slots.iter().enumerate() {
            if slot_used[s] || slot.status == FivePointSlotStatus::Unused {
                continue;
            }
            let d = ((f64::from(slot.root[0]) - z).powi(2) + f64::from(slot.root[1]).powi(2))
                .sqrt()
                / scale;
            if best.map_or(true, |(_, bd)| d < bd) {
                best = Some((s, d));
            }
        }
        match best {
            Some((s, d)) if d <= match_tol => {
                slot_used[s] = true;
                match_dists.push(d);
                classes.push(classify(gpu.slots[s].status));
                matched_slots.push(Some(s));
            }
            _ => {
                classes.push(RootClass::MissingNoSlot);
                matched_slots.push(None);
            }
        }
    }
    let spurious_gpu = gpu
        .slots
        .iter()
        .enumerate()
        .filter(|(s, slot)| slot.status == FivePointSlotStatus::Accepted && !slot_used[*s])
        .count();
    // A reference model is "missed" when its root's class is not Covered.
    let mut missed = Vec::new();
    for &(z, e) in &reference.models {
        if let Some(k) = real_roots.iter().position(|&r| r == z) {
            if classes[k] != RootClass::Covered && classes[k] != RootClass::Duplicate {
                missed.push((classes[k], e));
            }
        }
    }
    TrialLedger {
        ref_models: reference.models.len(),
        coeff_err: f64::NAN,
        classes,
        match_dists,
        spurious_gpu,
        missed,
    }
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
    ensure!(
        inputs.len() == 65024 && shared::cpu::input_digest(&inputs) == INPUT,
        "input provenance changed"
    );

    let db = ColmapDatabase::open_read_only(&args.database)?;
    let images = db.read_all_images()?;
    let camera_for = |id: u64| -> Result<CameraModel> {
        let image = images
            .iter()
            .find(|i| u64::from(i.image_id) == id)
            .context("missing image")?;
        let c = db
            .read_camera(image.camera_id)?
            .context("missing camera")?
            .camera;
        CameraModel::from_colmap(c.model_id, c.width, c.height, &c.params)
            .context("unsupported camera")
    };
    let mut thresholds = Vec::with_capacity(pairs.len());
    for pair in &pairs {
        let ids = pair["image_ids"].as_array().context("image_ids")?;
        let c1 = camera_for(ids[0].as_u64().context("id")?)?;
        let c2 = camera_for(ids[1].as_u64().context("id")?)?;
        thresholds.push(
            0.5 * (c1.cam_from_img_threshold(MAX_ERROR_PX)
                + c2.cam_from_img_threshold(MAX_ERROR_PX)),
        );
    }

    let context = WgpuContext::try_new()?;
    let device = format!("{:?}", context.capabilities());
    let gpu = WgpuFivePointF32::from_context(context)?;

    ensure!(
        args.match_tol.is_finite() && args.match_tol > 0.0,
        "invalid tolerance"
    );
    let gpu_results = shared::gpu_replay(&gpu, &inputs, inputs.len())?;
    let model_signature = shared::model_signature(&gpu_results);
    let diagnostic_signature = shared::signature(&shared::gpu_bits(&gpu_results));
    if !args.candidate {
        ensure!(
            model_signature == MODEL_SIGNATURE_Q2,
            "baseline model signature drifted"
        );
        ensure!(
            diagnostic_signature == DIAGNOSTIC_SIGNATURE_Q2,
            "baseline diagnostic signature drifted"
        );
    }
    let mut timings = BTreeMap::new();
    for batch in [512, inputs.len()] {
        let mut seconds = Vec::new();
        for _ in 0..3 {
            let start = std::time::Instant::now();
            let repeated = shared::gpu_replay(&gpu, &inputs, batch)?;
            seconds.push(start.elapsed().as_secs_f64());
            ensure!(
                shared::model_signature(&repeated) == model_signature,
                "unstable model signature batch {batch}"
            );
            ensure!(
                shared::signature(&shared::gpu_bits(&repeated)) == diagnostic_signature,
                "unstable diagnostics batch {batch}"
            );
        }
        timings.insert(batch.to_string(), seconds);
    }
    let bits_path = args.output.with_extension("bits.json");
    ensure!(!bits_path.exists(), "refusing to overwrite bits");
    std::fs::write(
        bits_path,
        serde_json::to_vec(&shared::gpu_bits(&gpu_results))?,
    )?;

    // Gate 2: CPU f64 replay unchanged (also validates the lib refactor).
    let pool = ThreadPoolBuilder::new().num_threads(8).build()?;
    let cpu_results = shared::cpu::replay_batches(&inputs, Some(&pool), inputs.len());
    let mut hasher = blake3::Hasher::new();
    for models in &shared::cpu::output_bits(&cpu_results) {
        hasher.update(&(models.len() as u64).to_le_bytes());
        for m in models {
            for w in m {
                hasher.update(&w.to_le_bytes());
            }
        }
    }
    ensure!(
        hasher.finalize().to_hex().to_string() == CPU_SIGNATURE,
        "CPU signature changed"
    );

    // Staged GPU replay: basis and f32 polynomial coefficients per trial.
    let mut bases: Vec<Option<[f32; 36]>> = Vec::with_capacity(inputs.len());
    let mut stage_errors = Vec::with_capacity(inputs.len());
    let stage_path = args.output.with_extension("stages.bin");
    let mut stage_file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(stage_path)?;
    use std::io::Write;
    let mut gpu_coeffs: Vec<Option<[f32; 11]>> = Vec::with_capacity(inputs.len());
    for chunk in inputs.chunks(STAGE_CHUNK) {
        let left: Vec<[f32; 3]> = chunk
            .iter()
            .flat_map(|i| i.left.iter().map(|r| [r.x as f32, r.y as f32, r.z as f32]))
            .collect();
        let right: Vec<[f32; 3]> = chunk
            .iter()
            .flat_map(|i| i.right.iter().map(|r| [r.x as f32, r.y as f32, r.z as f32]))
            .collect();
        let matrices = gpu.compute_constraint_matrices(&left, &right)?;
        let nullspace = gpu.compute_nullspace_diagnostics(&matrices)?;
        let with_basis: Vec<[f32; 36]> = nullspace.iter().filter_map(|n| n.basis).collect();
        let algebra = gpu.compute_algebra(&with_basis)?;
        let mut algebra_iter = algebra.into_iter();
        for n in &nullspace {
            match n.basis {
                Some(b) => {
                    let a = algebra_iter.next().context("algebra result missing")?;
                    stage_errors.push(attribution::measure(&b, &a));
                    // Fixed little-endian record: basis36, A200, solve100, B39, coeff11.
                    for x in b
                        .iter()
                        .chain(&a.elimination)
                        .chain(&a.solved)
                        .chain(&a.determinant)
                        .chain(&a.coefficients)
                    {
                        stage_file.write_all(&x.to_bits().to_le_bytes())?;
                    }
                    bases.push(Some(b));
                    gpu_coeffs
                        .push((a.status == FivePointStatus::Success).then_some(a.coefficients));
                }
                None => {
                    stage_errors.push([f64::NAN; 8]);
                    stage_file.write_all(&[0u8; 386 * 4])?;
                    bases.push(None);
                    gpu_coeffs.push(None);
                }
            }
        }
    }
    ensure!(bases.len() == inputs.len());
    // Consistency: staged basis availability must match full-pipeline nullspace
    // success. Roots copies non-success algebra status into results[ob]
    // (upstream), so SingularElimination/NonFinite there still imply a basis.
    for (i, r) in gpu_results.iter().enumerate() {
        let full_had_basis = r.upstream_status == FivePointStatus::Success
            || (r.algebra_status != FivePointStatus::Success
                && r.upstream_status == r.algebra_status);
        ensure!(
            bases[i].is_some() == full_had_basis,
            "staged/full nullspace disagreement at trial {i}"
        );
    }

    // Same-basis f64 reference + per-trial attribution.
    let ledgers: Vec<TrialLedger> = pool.install(|| {
        (0..inputs.len())
            .into_par_iter()
            .map(|i| match bases[i] {
                Some(b) => {
                    let basis = EssentialBasis::from_column_slice(&b.map(f64::from));
                    let reference = essential_reference_from_basis(&basis);
                    let mut ledger = attribute_trial(&reference, &gpu_results[i], args.match_tol);
                    if let (Some(rc), Some(gc)) = (reference.coefficients, gpu_coeffs[i]) {
                        let scale = rc.iter().fold(0.0f64, |m, &c| m.max(c.abs()));
                        if scale > 0.0 {
                            ledger.coeff_err = rc
                                .iter()
                                .zip(&gc)
                                .fold(0.0f64, |m, (&r, &g)| m.max((f64::from(g) - r).abs()))
                                / scale;
                        }
                    }
                    ledger
                }
                None => TrialLedger {
                    ref_models: 0,
                    coeff_err: f64::NAN,
                    classes: Vec::new(),
                    match_dists: Vec::new(),
                    spurious_gpu: 0,
                    missed: Vec::new(),
                },
            })
            .collect()
    });

    let trial_data: Vec<_> = ledgers
        .iter()
        .enumerate()
        .map(|(i, l)| {
            json!({
                "trial":i, "loss":!l.missed.is_empty(), "coefficient_error":l.coeff_err,
                "stage_errors":stage_errors[i], "gpu_models":gpu_results[i].model_count,
            })
        })
        .collect();
    let trial_path = args.output.with_extension("trials.json");
    ensure!(!trial_path.exists(), "refusing to overwrite trials");
    std::fs::write(&trial_path, serde_json::to_vec(&trial_data)?)?;
    let baseline: Vec<serde_json::Value> = if let Some(p) = &args.baseline_trials {
        serde_json::from_slice(&std::fs::read(p)?)?
    } else {
        trial_data.clone()
    };
    ensure!(
        baseline.len() == inputs.len(),
        "baseline trial count mismatch"
    );
    let stage_summary: BTreeMap<_,_> = attribution::NAMES.iter().enumerate().map(|(j,name)| {
        (*name,json!({
            "all":attribution::summary(stage_errors.iter().map(|e|e[j])),
            "fixed_baseline_loss":attribution::summary(stage_errors.iter().enumerate().filter(|(i,_)|baseline[*i]["loss"] == true).map(|(_,e)|e[j])),
        }))
    }).collect();
    let fixed_loss_error = attribution::summary(
        ledgers
            .iter()
            .enumerate()
            .filter(|(i, _)| baseline[*i]["loss"] == true)
            .map(|(_, l)| l.coeff_err),
    );

    // Ledger aggregation.
    let cpu_total: usize = cpu_results.iter().map(Vec::len).sum();
    let gpu_total: usize = gpu_results.iter().map(|r| r.model_count).sum();
    let ref_total: usize = ledgers.iter().map(|l| l.ref_models).sum();
    let rank_deficient: Vec<usize> = (0..inputs.len()).filter(|&i| bases[i].is_none()).collect();
    let cpu_models_in_rank_deficient: usize =
        rank_deficient.iter().map(|&i| cpu_results[i].len()).sum();

    let mut class_counts: BTreeMap<String, usize> = BTreeMap::new();
    for l in &ledgers {
        for c in &l.classes {
            *class_counts.entry(format!("{c:?}")).or_default() += 1;
        }
    }
    let spurious_total: usize = ledgers.iter().map(|l| l.spurious_gpu).sum();
    let missed_total: usize = ledgers.iter().map(|l| l.missed.len()).sum();

    let mut coeff_errs: Vec<f64> = ledgers
        .iter()
        .map(|l| l.coeff_err)
        .filter(|e| e.is_finite())
        .collect();
    coeff_errs.sort_by(|a, b| a.total_cmp(b));
    let mut coeff_buckets: BTreeMap<&'static str, usize> = BTreeMap::new();
    for &e in &coeff_errs {
        *coeff_buckets.entry(err_bucket(e)).or_default() += 1;
    }
    // Coefficient error conditioned on downstream loss.
    let mut err_with_loss: Vec<f64> = Vec::new();
    let mut err_without_loss: Vec<f64> = Vec::new();
    for l in &ledgers {
        if !l.coeff_err.is_finite() {
            continue;
        }
        if l.missed.is_empty() {
            err_without_loss.push(l.coeff_err);
        } else {
            err_with_loss.push(l.coeff_err);
        }
    }
    err_with_loss.sort_by(|a, b| a.total_cmp(b));
    err_without_loss.sort_by(|a, b| a.total_cmp(b));

    let mut dists: Vec<f64> = ledgers
        .iter()
        .flat_map(|l| l.match_dists.iter().copied())
        .collect();
    dists.sort_by(|a, b| a.total_cmp(b));

    // Quality weighting of missed reference models: inliers vs the trial's own
    // GPU best and vs the pair's GPU best (methodology from Q2 rescue-band).
    struct MissedScore {
        class: RootClass,
        rel_to_pair_best: f64,
        beats_trial_best: bool,
    }
    let pair_gpu_best: Vec<usize> = pool.install(|| {
        (0..pairs.len())
            .into_par_iter()
            .map(|p| {
                let obs = &observations[p];
                let t = thresholds[p];
                (p * TRIALS_PER_PAIR..(p + 1) * TRIALS_PER_PAIR)
                    .map(|i| {
                        gpu_models(&gpu_results[i])
                            .map(|m| inliers(obs, &m, t))
                            .max()
                            .unwrap_or(0)
                    })
                    .max()
                    .unwrap_or(0)
            })
            .collect()
    });
    // Absolute inlier counts and CPU-anchored ratios avoid a moving GPU denominator.
    let quality_rows: Vec<_> = pool.install(|| {
        (0..inputs.len())
            .into_par_iter()
            .map(|i| {
                let p = i / TRIALS_PER_PAIR;
                let cpu = cpu_results[i]
                    .iter()
                    .map(|m| inliers(&observations[p], m, thresholds[p]))
                    .max()
                    .unwrap_or(0);
                let gpu = gpu_models(&gpu_results[i])
                    .map(|m| inliers(&observations[p], &m, thresholds[p]))
                    .max()
                    .unwrap_or(0);
                [cpu, gpu]
            })
            .collect()
    });
    let pair_cpu_best: Vec<_> = quality_rows
        .chunks(TRIALS_PER_PAIR)
        .map(|p| p.iter().map(|r| r[0]).max().unwrap_or(0))
        .collect();
    let model_quality = json!({
        "gpu_trial_best_inliers":attribution::summary(quality_rows.iter().map(|r|r[1] as f64)),
        "cpu_trial_best_inliers":attribution::summary(quality_rows.iter().map(|r|r[0] as f64)),
        "gpu_trial_best_rel_cpu_pair_best":attribution::summary(quality_rows.iter().enumerate().map(|(i,r)|r[1] as f64/pair_cpu_best[i/TRIALS_PER_PAIR].max(1) as f64)),
        "gpu_beats_cpu_trial":quality_rows.iter().filter(|r|r[1]>r[0]).count(),
        "gpu_below_cpu_trial":quality_rows.iter().filter(|r|r[1]<r[0]).count(),
        "pair_gpu_best":pair_gpu_best,
        "pair_cpu_best":pair_cpu_best,
    });
    let quality_path = args.output.with_extension("quality.json");
    ensure!(!quality_path.exists(), "refusing to overwrite quality");
    std::fs::write(quality_path, serde_json::to_vec(&quality_rows)?)?;
    let missed_scores: Vec<MissedScore> = pool.install(|| {
        (0..inputs.len())
            .into_par_iter()
            .flat_map_iter(|i| {
                let pair = i / TRIALS_PER_PAIR;
                let obs = &observations[pair];
                let t = thresholds[pair];
                let trial_best = gpu_models(&gpu_results[i])
                    .map(|m| inliers(obs, &m, t))
                    .max()
                    .unwrap_or(0);
                let pair_best = pair_gpu_best[pair].max(1);
                ledgers[i]
                    .missed
                    .iter()
                    .map(move |&(class, e)| {
                        let n = inliers(obs, &e, t);
                        MissedScore {
                            class,
                            rel_to_pair_best: n as f64 / pair_best as f64,
                            beats_trial_best: n > trial_best,
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    });
    let mut missed_by_class: BTreeMap<String, (usize, Vec<f64>, usize)> = BTreeMap::new();
    for s in &missed_scores {
        let entry = missed_by_class
            .entry(format!("{:?}", s.class))
            .or_insert((0, Vec::new(), 0));
        entry.0 += 1;
        entry.1.push(s.rel_to_pair_best);
        entry.2 += usize::from(s.beats_trial_best);
    }
    let missed_quality: BTreeMap<String, serde_json::Value> = missed_by_class
        .into_iter()
        .map(|(k, (n, mut rel, beats))| {
            rel.sort_by(|a, b| a.total_cmp(b));
            (
                k,
                json!({
                    "n": n,
                    "rel_to_pair_best_p50": percentile(&rel, 0.5),
                    "rel_to_pair_best_p90": percentile(&rel, 0.9),
                    "beats_trial_best": beats,
                }),
            )
        })
        .collect();

    // The 2,899 CPU-has/GPU-empty trials specifically.
    let mut empty_class_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut empty_no_ref_models = 0usize;
    let mut empty_rank_deficient = 0usize;
    let mut cpu_has_gpu_empty = 0usize;
    for i in 0..inputs.len() {
        if cpu_results[i].is_empty() || gpu_results[i].model_count != 0 {
            continue;
        }
        cpu_has_gpu_empty += 1;
        if bases[i].is_none() {
            empty_rank_deficient += 1;
        } else if ledgers[i].ref_models == 0 {
            empty_no_ref_models += 1;
        } else {
            for c in &ledgers[i].classes {
                *empty_class_counts.entry(format!("{c:?}")).or_default() += 1;
            }
        }
    }

    let report = json!({
        "round": "Q4-coefficient-attribution",
        "kind": if args.candidate {"candidate"} else {"baseline"},
        "stage_attribution": stage_summary,
        "fixed_baseline_loss_coefficient_error": fixed_loss_error,
        "timing_full_diagnostic_replay_seconds_by_batch": timings,
        "signature_repeats": 6,
        "device": device,
        "input_digest": INPUT,
        "signatures": {
            "model": model_signature,
            "diagnostic": diagnostic_signature,
            "cpu_f64": CPU_SIGNATURE,
            "all_verified": true,
        },
        "counts": {
            "trials": inputs.len(),
            "cpu_models": cpu_total,
            "gpu_models": gpu_total,
            "f64_ref_models_on_gpu_basis": ref_total,
            "rank_deficient_trials": rank_deficient.len(),
            "cpu_models_in_rank_deficient_trials": cpu_models_in_rank_deficient,
            "cpu_has_gpu_empty": cpu_has_gpu_empty,
        },
        "model_gap_ledger": {
            "cpu_minus_gpu": cpu_total as i64 - gpu_total as i64,
            "basis_and_gate_component_cpu_minus_ref": cpu_total as i64 - ref_total as i64,
            "downstream_component_ref_minus_gpu": ref_total as i64 - gpu_total as i64,
            "missed_reference_models": missed_total,
            "spurious_gpu_models_no_f64_root": spurious_total,
        },
        "root_classes_of_f64_real_roots": class_counts,
        "root_match_distance": {
            "n": dists.len(),
            "p50": percentile(&dists, 0.5),
            "p99": percentile(&dists, 0.99),
            "max": dists.last().copied().unwrap_or(f64::NAN),
            "tolerance": args.match_tol,
        },
        "coefficient_error_vs_same_basis_f64": {
            "n": coeff_errs.len(),
            "buckets": coeff_buckets,
            "p50": percentile(&coeff_errs, 0.5),
            "p90": percentile(&coeff_errs, 0.9),
            "p99": percentile(&coeff_errs, 0.99),
            "trials_with_downstream_loss": {
                "n": err_with_loss.len(),
                "p50": percentile(&err_with_loss, 0.5),
                "p90": percentile(&err_with_loss, 0.9),
            },
            "trials_without_downstream_loss": {
                "n": err_without_loss.len(),
                "p50": percentile(&err_without_loss, 0.5),
                "p90": percentile(&err_without_loss, 0.9),
            },
        },
        "missed_model_quality": missed_quality,
        "model_quality": model_quality,
        "cpu_has_gpu_empty_breakdown": {
            "rank_deficient": empty_rank_deficient,
            "gpu_basis_f64_ref_also_empty": empty_no_ref_models,
            "root_classes": empty_class_counts,
        },
    });
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    println!("saved {}: {}", args.output.display(), report["counts"]);
    Ok(())
}
