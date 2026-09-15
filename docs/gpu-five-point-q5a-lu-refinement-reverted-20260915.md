# Q5a LU/B f32 precision — one-step iterative refinement (2026-09-15)

## Decision

**Revert.** One-step iterative refinement of the 10×10 partial-pivot LU solve did
**not** improve `B_gpu_vs_f64_same_basis` on the fixed Q4 loss cohort; end-to-end
model counts and Sampson quality regressed. Shader restored to Q4 baseline; Q4
model/diagnostic signatures remain active.

## Single variable

**Technique:** one-step iterative refinement (Wilkinson-style) in
`five_point_algebra.wgsl`:

1. First `ge_solve10` on the augmented 10×20 matrix (unchanged arithmetic).
2. Early write-back preserves original A/b in `algebra_output[ob..ob+200]`.
3. Residual RHS per column: `r = b − A·x₀` using stored original A.
4. Second full partial-pivot GE + back-substitution on `[A | r]`.
5. `x = x₀ + dx` when the second solve succeeds; B assembled from refined `x`.

Extracted `ge_solve10()` helper to share LU/back-sub between passes. No changes to
`five_point_generated.wgsl`, roots, recovery, or nullspace gates.

## Device and inputs

- Apple M5 Max, wgpu
- DB: `flowers2_960_settlement_20260913/matching.db`
- Input digest: `af07459d637b5b8c2b20c76d3ef103f058505fe0e8307578ee7e8750f3e230c5`
- Harness: `five_point_gpu_q4_coefficients --candidate`
- Fixed loss cohort: `experiments/q4-verified-tol1e3-20260915.trials.json` `loss` flags
- Baseline: Q4 verified (`experiments/q4-verified-tol1e{3,2}-20260915.*`)

## Stage isolation (candidate run, tol 1e-3)

| Region | vs Q4 |
|---|---:|
| basis36 + A200 (bytes 0–235) | 65,024 / 65,024 identical |
| solve100 + B39 (bytes 236–535) | 64,403 / 65,024 changed |
| coeff11 (bytes 536–579) | 64,403 / 65,024 changed (downstream) |

Correct single-variable isolation: only LU/solve/B moved; upstream unchanged.

## B error (fixed baseline loss cohort, n=17,971)

| Metric | Q4 | Q5a candidate | Factor |
|---|---:|---:|---:|
| `B_gpu_vs_f64_same_basis` p90 | 4.447e-4 | 4.740e-4 | **0.94× (worse)** |
| `solve_gpu_vs_f64_same_basis` p90 | 4.451e-4 | 4.804e-4 | 0.93× (worse) |
| `polynomial_total` p90 | 1.281e-3 | 3.017e-3 | 0.42× (worse) |
| `polynomial_gpu_vs_f64_same_B` p90 | ~5.3e-8 | ~5.3e-8 | ~1× (unchanged) |

Within the fixed cohort: 8,847 trials B-improved, 9,124 B-worsened.

Q5 premeasure predicted B as the lever (~30× loss/no-loss ratio). Refinement in
pure f32 cannot compute residuals with enough extra precision; corrections often
overshoot, degrading both solve and downstream polynomial on the loss cohort.

## End-to-end counts (tol 1e-3)

| Metric | Q4 | Q5a candidate | Δ |
|---|---:|---:|---:|
| GPU models | 248,884 | 248,644 | −240 |
| MissingNoSlot | 10,094 | 16,862 | +6,768 |
| RecoveryFailed | 18,063 | 15,779 | −2,284 |
| NotDone | 10,359 | 9,610 | −749 |
| Covered | 246,235 | 242,495 | −3,740 |
| Spurious (no f64 root) | 2,649 | 6,149 | +3,500 |

RecoveryFailed/NotDone shifts are reclassification from worse roots, not evidence
recovery code improved.

## Signatures (candidate only — not retained)

| | Q4 (retained) | Q5a candidate |
|---|---|---|
| model | `26e825cf97002be4f7fc36b77ce4fed682445af37746504253444e0103c5e716` | `54303091737e7ee1b3b7a9a334dbc878c2769caeaccd2caf6e3d3d0cc08a1bb0` |
| diagnostic | `d9e911ac6b6e2be18f7e3b8b8615417efba6257f955ce2bbf21f943a1c7f358a` | `e195d0e7b8f3a7727064721108481008d2671ddc64c88192109d631388527dff` |
| CPU f64 | `9e6764b41602d0005710187d2558d046f1a843c5f873b9542dd7b4f6c29c563a` | unchanged |

## Quality (Sampson, tol 1e-2, fixed input)

| Trial GPU best vs Q4 | Count |
|---|---:|
| Better | 4,208 |
| Worse | 4,271 |
| Equal | 56,545 |

Net **worse** (was +4,917 better / −4,200 worse at Q4 retain).

## Performance (full diagnostic replay medians)

| Batch | Q4 | Q5a candidate | Ratio |
|---:|---:|---:|---:|
| 512 (127 calls) | 1.492 s | 1.573 s | 1.055× |
| 65,024 | 0.080 s | 0.080 s | ~1× |

Second GE pass adds ~5% wall time on batch-512 path; not a speedup claim.

## Tests

`cargo test -p rustsfm --release --lib --no-default-features --features gpu-wgpu
--target-dir target five_point -- --nocapture --test-threads=1` — **11/11 pass**
on candidate build; reverted shader matches Q4.

## Artifacts

- `experiments/q5a-candidate-tol1e3-20260915` (+ `.trials.json`, `.bits.json`,
  `.stages.bin`, `.quality.json`)
- `experiments/q5a-candidate-tol1e2-20260915` (+ sidecars)

## Follow-up

Next Q5a candidate (separate round): compensated summation in B assembly only, or
mixed-precision residual with device-supported f64 if available — **not** another
f32-only refinement pass on the same residual path.
