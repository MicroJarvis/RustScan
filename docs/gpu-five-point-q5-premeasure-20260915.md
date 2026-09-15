# Q5 premeasure — LU/B vs recovery (2026-09-15, offline only)

## Decision for next implementation round

**Recommend Q5a: improve f32 precision of the 10×10 LU solve / B construction.**

Do **not** start with recovery 3×3 or Aberth this round. Evidence below mirrors
the Q3 coefficient story: residual same-basis error after Q4 is concentrated
upstream of the polynomial, and it discriminates loss trials.

No GPU/shader changes in this round. Artifacts only.

## Inputs

Q4 verified sidecars (tol is offline matching only; GPU bits identical):

- `experiments/q4-verified-tol1e{3,2}-20260915.{trials,bits,json}`
- Script: `scripts/summarize_five_point_q5_premeasure.py`
- Outputs: `experiments/q5-premeasure-tol1e{3,2}-20260915.json`

Also completed in the same disposition session (not part of this measurement):

- Formal **retain Q4**; quality signatures switched to Q4.
- Active harnesses (`p2_session`, `p3_submit`, `p3_double`,
  `recovery_attribution`) now gate on Q4 model/diagnostic.
- P3 double-buffer re-run under Q4 shader:
  `experiments/p3-double-buffer-q4shader-20260915.json` — median **0.674 s**,
  signature OK.

## B error as discriminator

`stage_errors[4] = B_gpu_vs_f64_same_basis` (max-abs / scale):

| Cohort (tol 1e-3 trials) | n | p50 | p90 |
| --- | ---: | ---: | ---: |
| Loss trials | 17,971 | 1.34e-5 | **4.45e-4** |
| No-loss trials | 46,432 | 1.26e-6 | **1.49e-5** |
| **p90 loss / no-loss** | | | **29.8×** |

Tol 1e-2: same ratio **29.1×**. Trials that contain any
`NullVectorFailure` / `InvalidEssential` slot also show elevated B error
(p90 6.7e-4 vs 2.1e-5 without). Context: after Q4,
`polynomial_gpu_vs_f64_same_B` p90 is 4.4e-8 — the polynomial expansion is
essentially done; remaining total polynomial error tracks B.

## Recovery subtype (absolute GPU slots, not matched roots)

| Status | Slot count | Trials containing |
| --- | ---: | ---: |
| InvalidEssential | 11,907 | 8,826 |
| NullVectorFailure | 9,530 | 6,290 |
| either | — | 11,848 |

Both subtypes are large; InvalidEssential is slightly larger. They are
**not** the primary discriminator relative to B error. Matched-root
RecoveryFailed (18,063 at tol 1e-3 in the Q4 report) remains the largest
*end-to-end class*, but many of those failures sit on trials with elevated
B error — fixing B is the upstream lever.

`RealAxisRejected` dominates raw slot counts (~319k) but matched GateRealAxis
is still ~6 roots; most are unmatched complex/spurious slots (same as Q3/Q4).

## Recommendation rationale

1. Same pattern as Q3: loss cohort error ≫ no-loss cohort (here ~30× on B).
2. Q4 already removed same-B polynomial error; leftover same-basis error is
   LU/solve → B.
3. Recovery subtypes are mixed; jumping to 3×3 recovery would treat a
   symptom that may shrink when B improves (as MissingNoSlot shrank when
   coefficients improved).

**Q5a scope (next implementation round, single variable):** f32 accuracy of
the 10×10 elimination solve and/or B assembly in
`five_point_algebra.wgsl` / generated path. Candidates: iterative refinement
of the solve, compensated accumulation into B, or guarded pivots using the
existing `minimum_relative_pivot` diagnostic. Do not touch roots, recovery,
or gates.

**Acceptance:** re-run Q4 coefficient harness (or Q5 attribution):
`B_gpu_vs_f64_same_basis` p90 on the fixed loss cohort must drop materially;
expect MissingNoSlot and RecoveryFailed matched counts to move; new
model/diagnostic signatures and quality table required.

**Q5b (queued):** if after Q5a a large RecoveryFailed residue remains with
*good* B, then split NullVector vs InvalidEssential on matched roots and
fix the 3×3 path.

## Stop

Measurement complete. No implementation in this round.
