# Q3 recovery-stage lost-model attribution (2026-09-15, measurement only)

## Question

After Q2, the model gap is CPU 288,264 vs GPU 243,967 (−44,297), yet only
2,942 trials are fully empty on GPU. Where exactly do the missing models die,
and are the f32 polynomial coefficients trustworthy enough to blame the root
finder or the 2e-5 gates?

## Method

Key design: replay the **f64** elimination → determinant polynomial →
companion-matrix roots → recovery chain **on the GPU's own nullspace basis**
(readback via the staged `compute_nullspace_diagnostics` / `compute_algebra`
APIs). This separates basis differences from everything downstream, and gives
per-trial ground truth: f64 coefficients, f64 real roots, and f64 reference
models for exactly the polynomial the GPU was trying to solve.

Every f64 real root is matched to a GPU slot (greedy nearest, relative
tolerance, audited by the reported distance distribution) and attributed to
one class by that slot's Q1 status. Missed reference models are additionally
scored with Sampson inliers against the pair's GPU best (Q2 methodology).

- Lib change (behavior-preserving refactor, CPU f64 signature gate passed):
  `geometry::five_point::essential_reference_from_basis` exposes the
  post-nullspace f64 chain; `estimate_five_point_essential` now calls it.
- Harness: `rustsfm/examples/five_point_gpu_recovery_attribution.rs`
- Results: `artifacts/evidence/q3-recovery-attribution-20260915.json` (tol 1e-3),
  `artifacts/evidence/q3-recovery-attribution-tol1e2-20260915.json` (tol 1e-2
  sensitivity).
- Gates all passed: input digest, Q2 model + diagnostic signatures, CPU f64
  signature. No shader or solver changes. Device: Apple M5 Max.

## Ledger (65,024 trials)

| Component | Models |
| --- | ---: |
| CPU (own basis, f64) | 288,264 |
| f64 reference on GPU basis | 285,371 |
| GPU f32 | 243,967 |
| **basis + nullspace-gate component** (CPU − ref) | **2,893** |
| **downstream component** (ref − GPU) | **41,404** |

The nullspace/basis line is essentially settled by Q2: only 2,893 models
(6.5% of the gap), of which 2,476 sit in the 621 RankDeficient trials.
**93.5% of the gap is downstream of a correct basis.**

## Downstream attribution (tol 1e-3 / tol 1e-2)

Per f64 real root of the same-basis reference polynomial:

| Class | tol 1e-3 | tol 1e-2 |
| --- | ---: | ---: |
| Covered (GPU accepted same root) | 232,218 | 239,700 |
| **MissingNoSlot** (no GPU root nearby) | **37,345** | **21,467** |
| **RecoveryFailed** (root matched; NullVector/InvalidEssential) | **9,862** | **14,445** |
| NotDone (root matched; Aberth done[] unset) | 5,458 | 8,793 |
| GateRealAxis + GatePolish | 21 | 142 |
| ComplexMisclassified | 39 | 378 |
| Spurious GPU-accepted (no f64 root) | 11,749 | 4,267 |

Reading the two tolerances together: 16k of the "missing" roots at 1e-3 are
GPU roots displaced by 1e-3–1e-2 relative — root movement, not root absence.
Even at 1e-2, 21.5k roots have no GPU counterpart at all.

## The load-bearing finding: coefficients, not gates

- The **2e-5 real-axis / polish gates are irrelevant**: 21–142 roots total.
  Loosening them would recover nothing. Q1's `unconverged` counters do not
  identify the loss mechanism.
- Coefficient error (max-abs relative vs same-basis f64, n = 64,403):
  p50 = 4.2e-6, p90 = 8.4e-4, p99 = **11.3**; 2,010 trials ≥ 1e-1 (wrong
  coefficients), 4,159 ≥ 1e-2.
- Conditioned on downstream loss: trials **with** lost models have coeff
  error p50 = 8.6e-5, p90 = 0.084; trials **without** loss p50 = 2.1e-6,
  p90 = 2.9e-5. Three orders of magnitude apart at p90 — coefficient error
  is the discriminating variable for root displacement/absence.
- The missing models matter: rel-to-pair-best Sampson p50 ≈ 0.40, and
  15,427 of 37,345 MissingNoSlot models beat their own trial's GPU best.
  This is not duplicate/junk loss.

Secondary, real but smaller mechanisms (measured at matched roots, i.e. not
explained by coefficient error alone):

1. **RecoveryFailed** 9.9k–14.4k — correct root, then the 3×3 null-vector /
   essential construction fails in f32. Second target after coefficients.
2. **NotDone** 5.5k–8.8k — Aberth iteration cap at a correct root location.

The 2,899 CPU-has/GPU-empty trials decompose as: 616 RankDeficient (true
nullspace gate), 3 where even the f64 reference on the GPU basis is empty,
and the rest dominated by MissingNoSlot (7,854 roots) + RecoveryFailed (821)
+ NotDone (398) — same mechanism ranking as the global ledger.

## Conclusion for round B (single variable)

Per the TODO ordering rule ("coefficients must be trusted before studying
root finding or gates"), the next implementation round is:

**Improve f32 accuracy of the algebra stage** (10×10 elimination LU and/or
determinant-polynomial expansion in `five_point_algebra.wgsl`), e.g.
compensated summation / FMA two-prod in the coefficient accumulation, or
rescaling. Acceptance is a re-run of this harness: coefficient-error p90 for
the loss cohort must drop materially and MissingNoSlot must shrink, with new
model/diagnostic signatures and a re-established quality table. Roots
(NotDone) and recovery (NullVector/InvalidEssential) are follow-up rounds and
must not be mixed in.

## Residuals / caveats

- Root matching is greedy nearest with a relative tolerance; distances are
  reported (p50 2.0e-6, p99 7.6e-4 at tol 1e-3) and the two-tolerance runs
  bound the ambiguity. Class boundaries between MissingNoSlot / NotDone /
  RecoveryFailed shift with tolerance; the coefficient-error conclusion does
  not.
- `Duplicate` (428) counted as neither covered nor missed; the GPU dedups
  where f64 keeps near-identical roots — negligible at this scale.
- Spurious GPU models (11.7k at 1e-3) are mostly the same displacement story
  seen from the other side; at 1e-2 they drop to 4.3k.
