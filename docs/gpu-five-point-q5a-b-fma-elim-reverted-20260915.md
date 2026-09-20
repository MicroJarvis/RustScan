# Q5a-B — FMA elimination / back-sub (2026-09-15)

## Decision

**Revert.** FMA-compensated accumulation in the 10×10 elimination and
back-substitution did **not** materially improve
`B_gpu_vs_f64_same_basis` on the fixed Q4 loss cohort. Shader restored to Q4
baseline; Q4 model/diagnostic signatures remain active.

## Why this technique (vs “B assembly”)

Prior Q5a iterative refinement was reverted. Measured
`solve_gpu_vs_f64_same_basis` ≈ `B_gpu_vs_f64_same_basis`, and
`B_gpu_vs_f64_same_solve` p90 stays ~4e-8 — B entry wiring (`x−y`) is already
exact given `solved`. Compensating only B assembly cannot move the error.

This round therefore applied Q4-style FMA to the **LU→solved** producer inside
`five_point_algebra.wgsl` (still a single LU/B-path variable; polynomial FMA,
roots, recovery, and gates untouched).

## Single variable

In `five_point_algebra.wgsl` only:

1. Elimination update: `a[…] = fma(-factor, a[…], a[…])`
2. Back-substitution: `x = fma(-a[…], solved[…], x)`

Pivot selection and thresholds unchanged. Measurement-only harness tweak in
`five_point_gpu_q4_coefficients.rs`: treat algebra failure after a produced
basis as “had basis” when roots mirrors algebra status into `upstream`
(needed so FMA candidate trials with `SingularElimination` still attribute).

## Device and inputs

- Apple M5 Max, wgpu
- DB: `flowers2_960_settlement_20260913/matching.db`
- Input digest: `af07459d637b5b8c2b20c76d3ef103f058505fe0e8307578ee7e8750f3e230c5`
- Harness: `five_point_gpu_q4_coefficients --candidate --match-tol 0.001`
- Fixed loss cohort: `artifacts/evidence/q4-verified-tol1e3-20260915.trials.json`

## Stage isolation (tol 1e-3 stages.bin)

| Region (float indices) | vs Q4 |
|---|---:|
| basis36 + A200 (0–235) | 65,024 / 65,024 identical |
| solve100 + B39 (236–374) | 64,403 / 65,024 changed |
| coeff11 (375–385) | 64,403 / 65,024 changed (downstream) |

Upstream basis/A unchanged; only LU→solved/B (and downstream coeffs) moved.

## B / solve error (fixed baseline loss cohort)

| Metric | Q4 (IR report / same cohort) | Q5a-B FMA | Factor |
|---|---:|---:|---:|
| `B_gpu_vs_f64_same_basis` p90 | **4.447e-4** | **4.361e-4** | **~1.02×** (not material) |
| `solve_gpu_vs_f64_same_basis` p90 | 4.451e-4 | 4.340e-4 | ~1.03× |
| `B_gpu_vs_f64_same_solve` p90 | ~4.4e-8 | 4.39e-8 | ~1× |
| cohort n | 17,971 | 17,971 | — |

All-trials B p90: 6.278e-5 → 6.139e-5 (~2%). Acceptance required a **material**
drop; ~2% fails the stop condition.

## End-to-end counts (tol 1e-3)

| Metric | Q4 | Q5a-B FMA | Δ |
|---|---:|---:|---:|
| GPU models | 248,884 | 248,882 | −2 |
| cpu_has_gpu_empty | 2,127 | 2,097 | −30 |
| MissingNoSlot | 10,094 | 9,995 | −99 |
| RecoveryFailed | 18,063 | 18,169 | +106 |
| NotDone | 10,359 | 10,329 | −30 |
| Covered | 246,235 | 246,241 | +6 |

## Signatures (candidate only — not retained)

| | Q4 (retained) | Q5a-B candidate |
|---|---|---|
| model | `26e825cf97002be4f7fc36b77ce4fed682445af37746504253444e0103c5e716` | `593ac9bb74d9092bbc16de7142607468c344604f3811f185d5d082c9dc0483e1` |
| diagnostic | `d9e911ac6b6e2be18f7e3b8b8615417efba6257f955ce2bbf21f943a1c7f358a` | `b08202e49386b8e695975f777caeb50ce1aa856c9e08fc29bfd038ee11078117` |
| CPU f64 | `9e6764b41602d0005710187d2558d046f1a843c5f873b9542dd7b4f6c29c563a` | unchanged |

## Quality (Sampson inliers, quality.json column 1 vs Q4 verified)

| Trial GPU best vs Q4 | Count |
|---|---:|
| Better | 3,751 |
| Worse | 3,755 |
| Equal | 57,518 |

Pair-best over 127×512: **3 / 3 / 121**. Net flat-to-slightly-worse.

## Performance (full diagnostic replay)

| Batch | Q4 samples (med≈) | Q5a-B samples (med≈) |
|---:|---:|---:|
| 512 | 1.492 s | 1.557 s |
| 65,024 | 0.080 s | 0.079 s |

No speedup claim; batch-512 ~4% slower on this single run set.

## Tests

`cargo test -p rustsfm --release --lib --no-default-features --features gpu-wgpu
--target-dir target five_point -- --nocapture --test-threads=1` — **11/11 pass**
on the FMA candidate build. Algebra shader restored to Q4 afterward.

## Artifacts

- `artifacts/evidence/q5a-b-fma-tol1e3-20260915` (+ `.trials.json`, `.bits.json`,
  `.stages.bin`, `.quality.json`)
- `artifacts/evidence/q5a-b-fma-tol1e2-20260915` (+ sidecars)
- Focused test log: `artifacts/evidence/q5a-b-fma-focused-tests.log`

Harness consistency fix for algebra-mirrored `upstream` is retained (measurement
correctness; does not change Q4 solver output).

## Follow-up

f32 FMA on elim/back-sub is not enough headroom after Q4. Do **not** retry the
same FMA rewrite. Remaining LU/B options are heavier (e.g. mixed-precision /
device f64 residual if available, or a different factorization). Otherwise
queue **Q5b recovery** once B is accepted as stuck, or move to performance
line P4 under the existing Q4 quality gate.
