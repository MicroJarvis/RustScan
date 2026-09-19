# Q5b — 3×3 recovery null-vector (2026-09-15)

## Decision

**Retain.** Single-variable improvement of `null3` in
`five_point_recovery.wgsl` (FMA Gram `BᵀB`, pick eigenvector by min `‖Bv‖`,
two damped inverse-iteration steps). Accept thresholds unchanged. Active quality
signatures switch from Q4 → Q5b.

## Offline characterization (no shader yet)

Tool: `RustSFM/examples/five_point_gpu_q5b_recovery_measure.rs` over Q4 verified
`stages.bin` + `bits.json` + `trials.json`. Good-B threshold = no-loss
`B_gpu_vs_f64_same_basis` p90 ≈ **1.494e-5** (Q5 premeasure).

| Metric (tol 1e-3) | Count |
|---|---:|
| Matched RecoveryFailed | **18,063** (= Q4 report) |
| → NullVectorFailure | 7,529 |
| → InvalidEssential | 10,534 |
| **good-B matched RF** | **6,759** |
| → good-B NullVector | 2,389 |
| → good-B InvalidEssential | 4,370 |
| bad-B matched RF | 11,304 |

Class table matches Q4 exactly. Absolute slots: NV 9,530 / IE 11,907.

Gate residuals on recovery-fail slots (good-B):

- **NullVectorFailure**: null residual p90 ≈ 1.5e-4 ≪ 2e-3 → almost all are
  `abs(nv.z) < 1e-6` (wrong/ill-conditioned homogeneous chart), not residual.
- **InvalidEssential**: constraint residual fine; **100%** fail
  `essential_residual > 2e-2` (p50 ≈ 0.097).

Stop rule (≥500 good-B matched RF): **do not stop** — proceed to 3×3 recovery.
Artifacts: `experiments/q5b-offline-measure-tol1e{3,2}-20260915.json`.

## Single variable

Only `null3` in `five_point_recovery.wgsl`:

1. FMA assembly of Gram `G = BᵀB`
2. After Jacobi, choose eigenvector with smallest `‖B v‖` (not only Jacobi `λ`)
3. Two damped inverse-iteration steps in the Jacobi eigenframe

Roots / algebra / gates / accept thresholds untouched. First probe with a
`BBᵀ` Gram bug collapsed models to 47 and was corrected before retain metrics.

## End-to-end (Q4 harness `--candidate`)

Device: Apple M5 Max / wgpu. Fixed loss cohort:
`experiments/q4-verified-tol1e3-20260915.trials.json`.

| Metric | Q4 | Q5b | Δ |
|---|---:|---:|---:|
| GPU models | 248,884 | **249,833** | **+949** |
| cpu_has_gpu_empty | 2,127 | 1,883 | −244 |
| RecoveryFailed tol 1e-3 | 18,063 | **17,475** | **−588** |
| Covered tol 1e-3 | 246,235 | 246,823 | +588 |
| RecoveryFailed tol 1e-2 | 20,119 | **19,295** | **−824** |
| Accepted slots | 248,884 | 249,833 | +949 |
| NullVectorFailure slots | 9,530 | 8,425 | −1,105 |
| InvalidEssential slots | 11,907 | 12,063 | +156 |

Good-B matched RF (same threshold): 6,759 → **6,647** (−112). Most of the
matched RF drop is on bad-B; NV improves, IE slightly worsens (trade).

## Quality (Sampson inliers, quality.json col1 = GPU best vs Q4)

| Trial GPU best vs Q4 | Count |
|---|---:|
| Better | 2,280 |
| Worse | 1,711 |
| Equal | 61,033 |
| **Net** | **+569** |

`gpu_trial_best_inliers` p50 98 → **101**, p90 1190 → 1208. No net regression.

## Signatures (retained)

| | Q4 (previous) | **Q5b (active)** |
|---|---|---|
| model | `26e825cf…c5e716` | `fb5be154bb0fa742ab1b83e5c9e62269722678cab463bfbe3827457b7424e8f0` |
| diagnostic | `d9e911ac…7f358a` | `1412c6d5d2af9bb9ed23c8d55238af22deb2b259cfa16fa43a99fb21486b9a1a` |
| CPU f64 | `9e6764b4…9c563a` | unchanged |
| digest | `af07459d…30c5` | unchanged |

Active harnesses (`p2_session`, `p3_submit`, `p3_double`,
`recovery_attribution`) gate on Q5b.

## Tests

`cargo test -p rustsfm --release --lib --no-default-features --features gpu-wgpu
--target-dir target five_point -- --nocapture --test-threads=1` — **11/11 pass**.
Log: `experiments/q5b-focused-tests.log`.

## Artifacts

- Offline: `experiments/q5b-offline-measure-tol1e{3,2}-20260915.json`
- Candidate: `experiments/q5b-candidate-tol1e{3,2}-20260915.*`
- Good-B remeasure on candidate: `experiments/q5b-candidate-goodb-measure-tol1e3-20260915.json`
- Pre-change shader copy: `experiments/q5b-recovery-shader-before.wgsl`
- Measure example: `RustSFM/examples/five_point_gpu_q5b_recovery_measure.rs`

## Follow-up

Good-B RF residue remains large (~6.6k); further recovery alone has limited
headroom without touching essential-cubic failures or mixed-precision B.
Next: **P4** under Q5b gate, or mixed-precision / f64 residual for LU/B.
Do not loosen `essential_residual` / `|z|` thresholds to chase IE counts.
