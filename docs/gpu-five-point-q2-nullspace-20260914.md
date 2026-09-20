# Q2 nullspace: σ-level gate via AAᵀ (2026-09-14)

## Decision

**Retain.** Rank gate now acts on singular values from the 5×5 Gram `AAᵀ`,
accept iff `σ₅/σ₁ > 1e-5`. Nullspace basis still comes from Jacobi on `AᵀA`
(algebra-compatible). Column-pivoted QR of `Aᵀ` remains 对照 only this round.

## Scope

Single variable: nullspace rank criterion (+ how σ is estimated for that
criterion). Roots / recovery / session reuse untouched.

## Criterion (written before the final gate constant)

1. Prescale `A` by max-abs entry.
2. Estimate `σ₁…σ₅` from Jacobi on `AAᵀ` (not from eigenvalues of `AᵀA`).
3. Accept iff `σ₅/σ₁ > 1e-5`.
4. On accept, take the four eigenvectors of `AᵀA` for the smallest
   eigenvalues as the basis; verify `A*N` and `NᵀN`.

Why not gate on `AᵀA` alone: f32 Jacobi on the 9×9 leaves a ~1e-5 noise floor
on numerical-null singular values, which overlaps the rescue band
`[1e-5, 1e-2)`. Hard degenerates (all-ones, five identical rows) then look
like mild rank-5. `AAᵀ` only has the five true singular values, so the gate
and the rescue band can share a floor at `1e-5`.

Why not QR this round: column-pivoted Householder QR of `Aᵀ` produced
`A*N≈0` bases but broke a synthetic algebra fixture (isolated polynomial
error ~1e-2 vs f64 five-point determinant identity). Parked as 对照.

`1e-6` on `AAᵀ` was measured first (`artifacts/evidence/q2-nullspace-20260914.json`);
outcomes were nearly identical to `1e-5`, which matches the bottom of the
written rescue band and is the retained constant
(`artifacts/evidence/q2-nullspace-gate1e5-20260914.json`).

## Provenance

- Worktree `.worktrees/gpu-five-point-f32`, source commit `3ab9c09` + Q1 + Q2.
- Device `Apple M5 Max`.
- Input digest unchanged:
  `af07459d637b5b8c2b20c76d3ef103f058505fe0e8307578ee7e8750f3e230c5`.
- CPU f64 signature unchanged:
  `9e6764b41602d0005710187d2558d046f1a843c5f873b9542dd7b4f6c29c563a`.
- Model signature Q1 → Q2:
  `030f4306…12df` → `472124c8f1a9c6299d659e2f835dc11c682ff8a593757f0b145f73454f51d6ea`.
- Diagnostic signature v2 → Q2:
  `3cd8091d…17cb` → `3d04bb3edaef183049e07d6c3841e296f963da3b279bcbd99d878e1dc090c050`.
- Harness: `rustsfm/examples/five_point_gpu_q2_nullspace.rs`.

## Acceptance table (f64 `σ₅/σ₁` buckets × GPU retain)

| bucket | total | retained | RankDeficient | retain rate | with models |
| --- | ---: | ---: | ---: | ---: | ---: |
| `>=1e-1` | 16 | 16 | 0 | 100% | 16 |
| `[1e-2,1e-1)` | 23,761 | 23,761 | 0 | 100% | 23,287 |
| `[1e-3,1e-2)` | 31,404 | 31,404 | 0 | 100% | 29,849 |
| `[1e-4,1e-3)` | 8,838 | 8,705 | 133 | 98.5% | 8,415 |
| `[1e-5,1e-4)` | 590 | 339 | 251 | 57.5% | 337 |
| `[1e-6,1e-5)` | 3 | 1 | 2 | 33% | 1 |
| `<=1e-7` | 197 | 88 | 109 | 44.7% | 88 |
| `<=0` | 215 | 89 | 126 | 41.4% | 89 |

Rescue band `[1e-5, 1e-2)`: **40,448 retained / 384 rejected** (99.1%).
Rescued trials that produced models: n=38,601, Sampson quality vs pair-best
p50=0.592, mean=0.589 (same quality class as the previous lost-nullspace
population).

## Headline counts vs Q1 baseline

| quantity | Q1 | Q2 | Δ |
| --- | ---: | ---: | ---: |
| GPU models | 209,600 | 243,967 | +34,367 |
| GPU empty trials | 12,012 | 2,942 | −9,070 |
| CPU-has / GPU-empty | 11,967 | 2,899 | −9,068 |
| `RankDeficient` | ~10,035 | 621 | −9,414 |

## Residual risk (documented, not a rollback trigger)

f64 `σ₅/σ₁ ≤ 1e-7` is **not** fully rejected: 88/197 still pass the f32
`AAᵀ` gate and produce models. CPU f64 has no rank test either, so these are
not a ground-truth failure by themselves, but they show f32 Gram inflation on
near-zero `σ₅`. Further tightening the constant does not remove them (1e-6 vs
1e-5 differed by only two trials). A later round that wants stricter
degenerate rejection needs a better σ estimator (QR/`R` diagonal or true SVD),
not another constant nudge.

`[1e-5,1e-4)` is only 57.5% retained: the hardest edge of the rescue band is
partial. The bulk of the historical loss (`[1e-4,1e-2)`) is essentially fully
rescued.

## Tests / gates

- `cargo test -p rustsfm --lib five_point` (release, `POSELIB_ROOT` +
  `--target-dir target`): pass, including mild rank-5 fixture at `σ₅/σ₁=3e-5`
  and hard degenerates (zeros / all-ones / five identical rows).
- Host field `min_diagonal_ratio` now means `σ₅/σ₁` from `AAᵀ`.

## Next

P2 persistent session. Q2 recovery / remaining lost roots stay later.
