# Lost solutions: CPU f64 has a model, GPU f32 has none (2026-09-14)

## Decision and scope

Measurement-only. No solver, shader or protected harness source changed; one new
harness `rustsfm/examples/five_point_gpu_lost_solutions.rs`. There is no
candidate to retain or roll back. This round names the cause of the 11,967 lost
trials with proof, and weighs them by model quality so the loss is a number
rather than a count.

Still independent offline candidate generation: not RANSAC, not the 960-image
pipeline. CPU f64 is an offline reference only.

Findings, up front:

1. **83.8% of the losses (10,028) are a threshold choice, not an f32 precision
   loss.** The GPU nullspace gate `eigenvalue > 1e-6 · trace(AᵀA)` is applied
   to σ², so it rejects any sample whose smallest singular value is below about
   1e-3 of the matrix norm. Evaluated in f64, that gate already rejects all
   10,028; there is not one trial where f32 arithmetic flipped the decision.
2. **Those samples are not junk.** Their CPU models score at least as well as
   the samples the GPU keeps (median 0.60 vs 0.51 of the pair's best inlier
   count). 96% of them are ordinary, mildly conditioned samples with
   σ₅/σ₁ ∈ [1e-5, 1e-2); only 4% are genuinely degenerate.
3. **At a fixed 512 trials per pair the pair-level outcome barely moves** —
   the GPU's best model trails the CPU's by more than 10% in only 2 of 127
   pairs, both already hopeless pairs — but the GPU discards 15.4% of all
   samples, concentrated in some pairs at up to 49.6%, which is a direct tax on
   effective sample rate for any RANSAC with early termination.
4. The remaining 16.2% (1,938) reach the root solver and lose every candidate,
   almost entirely at recovery. These are the *highest*-quality losses (median
   0.84 of pair best) and are a separate, downstream mechanism.

## Provenance

- Worktree `.worktrees/gpu-five-point-f32`, source commit `3ab9c09`.
- New file `rustsfm/examples/five_point_gpu_lost_solutions.rs`, sha256
  `1f5bfb1785336bd6cfaede7a3a3ab7da2e0c722d9c1242cb886b996b07713299`; binary
  `73a5108180fce60cd890ffa3f39deef145889772104d308c83a34ef597fc72a8`.
- Device `Apple M5 Max`. Fixed input unchanged: 127 pairs, 65,024 trials,
  digest `af07459d…c5`. GPU signature `1f6f3f8d…39` and CPU f64 signature
  `9e6764b4…3a` asserted equal to history before any analysis.
- `artifacts/evidence/lost-solutions-20260914.json` (first build) and
  `artifacts/evidence/lost-solutions-20260914-final.json` (final binary) agree on every
  attribution and quality field.
- Build requires `POSELIB_ROOT` pointing at the main checkout, as in the
  previous round; not on the measured path.

## What the two solvers actually do at the nullspace

CPU (`colmap_eigen.cpp:57-80`): `FullPivHouseholderQr` of the 9×5 Aᵀ and takes
the last four columns of Q. **There is no rank test.** It returns a basis for
every input, including a genuinely rank-deficient one, in which case four of the
five true null directions are chosen arbitrarily.

GPU (`five_point_f32.wgsl:33-68`): forms the 9×9 AᵀA in f32 after max-abs
prescaling, runs cyclic Jacobi, then

```
rank = #{ i : s[i][i] > 1e-6 · trace }        // line 66
if rank != 5 → status RankDeficient           // line 68
```

Since the eigenvalues of AᵀA are σᵢ² and the trace is Σσᵢ², this rejects when
σ₅² / Σσ² < 1e-6, i.e. roughly σ₅/‖A‖_F < 1e-3. The prescaling cancels in the
ratio, so the gate can be evaluated exactly on the f64 SVD of A and compared to
the GPU's own decision.

## Cause: the gate, not the arithmetic

Per trial the harness computes the f64 singular values of the 5×9 matrix and
`gate = σ₅²/Σσ²`.

| quantity | value |
| --- | --- |
| trials the GPU reports `RankDeficient` | 10,035 |
| trials with f64 `gate < 1e-6` | 10,036 |
| `RankDeficient` trials with f64 `gate < 1e-6` | **10,035 / 10,035** |
| lost `RankDeficient` trials with f64 `gate < 1e-6` (H1) | **10,028 / 10,028** |
| lost `RankDeficient` trials with f64 `gate ≥ 1e-6` (H2, precision) | **0** |
| kept trials with f64 `gate < 1e-6` | 1 |

The f64 gate reproduces the GPU's rank decision on 65,023 of 65,024 trials; the
single disagreement is a boundary case the f32 eigenvalue rounded across in the
*accepting* direction. So the GPU's Jacobi on f32 AᵀA is resolving this
threshold correctly. The threshold is the mechanism. Changing to a direct f32
method would not by itself recover anything unless the gate is also changed;
conversely, the gate cannot simply be lowered in the current AᵀA formulation
without first knowing where the f32 eigenvalue noise floor sits (see risks).

GPU-reported rank when deficient: 4 in 9,623 trials, 3 in 405.

### How ill-conditioned are the rejected samples?

σ₅/σ₁ of the 5×9 matrix, f64:

| σ₅/σ₁ | lost `RankDeficient` | kept (both solvers have models) |
| --- | --- | --- |
| ≥ 1e-1 | — | 16 |
| [1e-2, 1e-1) | — | 23,296 |
| [1e-3, 1e-2) | 192 | 29,698 |
| [1e-4, 1e-3) | **8,837** | — |
| [1e-5, 1e-4) | 589 | — |
| [1e-6, 1e-5) | 3 | — |
| < 1e-7 | 193 | — |
| σ₅ = 0 / non-finite | 214 | — |

The cut is sharp at σ₅/σ₁ ≈ 1e-3, a condition number of about a thousand. 9,621
of the rejected samples (95.9%) sit in [1e-5, 1e-2) — mild conditioning that a
direct QR or SVD resolves with margin even in f32, whose relative precision is
around 6e-8. Only 407 (4.1%) are at or below 1e-7: 214 have an exactly zero
σ₅ (a repeated correspondence in the sample — SIFT emits multiple keypoints at
one location with different orientations, and both can match) and 193 are
numerically degenerate. Those 407 are the ones a rank test *should* reject; the
CPU returns a model for them anyway because it has no test.

## The lost samples are not junk

Every CPU and GPU model was scored on the pair's full observation set by Sampson
inlier count, with the per-pair threshold `geometry.rs` derives from
`TwoViewGeometry.max_error = 4 px` through the two cameras' mean focal length.
Each trial's best model is divided by the best over all 512 CPU trials of its
pair, so 1.0 means "this sample alone reaches the pair's best model".

| population | trials | p50 | mean | p90 | ≥ 0.9 of pair best | < 0.25 | fits only its own 5 points |
| --- | --- | --- | --- | --- | --- | --- | --- |
| kept, both solvers have models | 53,010 | 0.507 | 0.546 | 0.990 | 20.5% | 19.8% | 271 |
| lost at nullspace (`RankDeficient`) | 10,028 | **0.603** | **0.595** | 0.959 | 16.1% | 11.0% | **9** |
| lost at roots (all candidates rejected) | 1,938 | **0.842** | **0.734** | 0.997 | **44.2%** | 8.5% | 7 |

The nullspace-rejected samples are, if anything, slightly better than average:
higher median, fewer in the bottom bucket, and only 9 of 10,028 produce a model
that fits nothing beyond its own five points. Hypothesis H3 — that the CPU is
producing models from degenerate samples RANSAC would never keep — is refuted
for 96% of these trials. It holds for the ~400 truly degenerate ones, which
the CPU should arguably also reject.

## What it costs at pair level

At 512 trials per pair, with no early termination:

| | pairs |
| --- | --- |
| GPU best inlier count below CPU best | 18 / 127 |
| GPU best below 90% of CPU best | **2 / 127** |

Of the 18, sixteen trail by one to three inliers out of hundreds or thousands
(e.g. 3,644 vs 3,645). The two substantive deficits are pairs with 15/86 and
21/119 inliers, which are wrong pairs regardless of solver. At this fixed budget
the final model is essentially unaffected.

The cost is in samples, not in the final model. The GPU discards 15.4% of all
trials at the nullspace. It is not uniform:

| pair | observations | lost / 512 | median σ₅/σ₁ of its samples | CPU best | GPU best |
| --- | --- | --- | --- | --- | --- |
| 1660004859914 | 459 | **254 (49.6%)** | 1.09e-3 | 366 | 365 |
| 1677184729097 | 526 | 227 | 1.36e-3 | 424 | 424 |
| 1389421920261 | 830 | 202 | 1.61e-3 | 749 | 749 |
| 992137445385 | 595 | 201 | 1.84e-3 | 486 | 486 |
| 1913407930374 | 511 | 195 | 2.06e-3 | 449 | 449 |

The pairs that lose most are those whose *typical* sample sits at the gate: a
median σ₅/σ₁ of 1e-3 to 2e-3 means half their samples fall on the wrong side.
Whether that comes from small baseline, narrow field of view or the scene's
depth structure is not established here; it is a property of the pair, not of
individual unlucky samples. For a RANSAC that terminates on confidence, halving
the effective sample rate on such a pair roughly doubles the trials needed to
reach the same confidence, and the reported quality comparison already shows
these are good samples being thrown away.

## The other 16%: losses at recovery

The 1,938 trials that reach the root solver and lose everything are a different
mechanism, and worth separating because they are the most valuable losses.

| | lost at roots | trials with models (kept 53,010 / produced 53,012) |
| --- | --- | --- |
| trials | 1,938 | 53,010 – 53,012 |
| σ₅/σ₁ in [1e-3, 1e-2) | 76.8% | 56.0% (of kept) |
| at the 384-iteration cap | 23.9% | 9.7% (of produced) |
| CPU models found in these trials | 7,628 (3.9 / trial) | — |

The CPU finds a normal ~4 real roots per trial here. The GPU's 19,380 slots in
these trials split as `NotConverged` 13,823, `InvalidEssential` 2,620,
`NullVectorFailure` 2,093, `Complex` 642, `DuplicateRoot` 3, `Unused` 199. Per
the previous round, `NotConverged` is 93% the real-axis gate acting as the
complex filter, so about six of the ten roots per trial are the expected
complex ones; the ~4 real roots are then rejected at recovery — 2.4 per trial by
`InvalidEssential` and `NullVectorFailure` combined — with the remainder
presumably real roots failing the real-axis gate's `2e-5` residual bound in
f32. That last split needs the Q1 diagnostic fix (a status code for the
real-axis gate) before it can be counted rather than inferred. These trials are
somewhat worse conditioned and more often hit the iteration cap than trials
that succeed, so the same upstream conditioning likely degrades the polynomial
they hand to recovery; but the rejection itself happens in recovery.

## Correctness gates

| gate | result |
| --- | --- |
| `cargo fmt --check` | pass |
| `cargo clippy --example five_point_gpu_lost_solutions --all-features` | pass, no warning in the new file |
| `cargo test --example five_point_gpu_lost_solutions` | pass, 12 tests |
| `git diff --check` | pass |
| GPU signature, CPU signature vs history | identical |
| first build vs final binary, all attribution and quality fields | identical |

Unit tests pin the analysis, not the measured values: the gate equals the trace
identity and is invariant to the GPU's prescaling; a repeated correspondence
drives σ₅ to zero and trips the gate; the Sampson inlier test accepts exact
epipolar points and rejects the identity matrix; bucket edges.

## What this means for the plan

- **Q2's first item is confirmed, and sharpened.** The TODO already lists
  "评估直接 f32 零空间算法（如 QR），避免 AᵀA 条件数恶化". The evidence
  says the squaring matters through the *gate*, not through lost precision: a
  direct method's value is that its natural rank test acts on σ rather than σ²,
  where a cut near 1e-6 of ‖A‖ would keep 96% of these samples and still reject
  the 407 genuinely degenerate ones. Any redesign must state its rank criterion
  up front and be judged against the σ₅/σ₁ table above, not against the raw
  lost count.
- **Do not simply lower the 1e-6 constant.** That is exactly the "放宽阈值让
  失败计数更好看" the TODO forbids, and it is unsafe on the merits: the f32
  eigenvalues of AᵀA carry a noise floor near ε·trace ≈ 6e-8·trace, and the
  kernel exports only `rank`, not the eigenvalues, so the distance between the
  floor and a lower gate is unmeasured. If a threshold change is ever tested,
  the smallest eigenvalue must be exported first so the change can be shown to
  stay above the floor.
- **The CPU reference is not a ground truth for degeneracy.** It has no rank
  test, so "CPU has a model" includes ~400 samples with an exactly or nearly
  rank-deficient constraint matrix. Quality gates comparing GPU to CPU should
  exclude or separately count trials with f64 σ₅/σ₁ below about 1e-6.
- **The recovery-stage losses are the next target after the nullspace**, and
  they depend on the Q1 diagnostic fix to be counted properly. They are 16% of
  the losses but the best models among them.
- **A fixture set falls out of this.** The 10,028 trials are real failure
  samples with known f64 spectra and known model quality; the TODO's "用真实失
  败样本建立回归 fixtures" can be drawn from them, stratified by the σ₅/σ₁
  buckets above, and the harness output already carries what is needed to
  select them.

## Unresolved risks

- Sampson inlier count at the pipeline's 4 px threshold is a proxy for
  candidate quality, not a RANSAC run. It says the lost models are as good as
  kept ones; it does not measure end-to-end pair verification or reconstruction.
- The per-pair loss concentration is described, not explained. Which geometric
  property drives a pair's typical σ₅/σ₁ toward 1e-3 was not investigated.
- The f32 eigenvalue noise floor is inferred from ε, not measured; only `rank`
  leaves the kernel.
- The recovery-stage split between `InvalidEssential` / `NullVectorFailure`
  and real roots failing the real-axis residual bound is inferred from counts
  and awaits the Q1 status code.
- Nothing here changed any output. Quality remains non-equivalent to CPU f64.

## Reproduction

```
POSELIB_ROOT=<main-checkout>/third_party/native/PoseLib \
  cargo build --release --target-dir target -p rustsfm \
  --example five_point_gpu_lost_solutions

./target/release/examples/five_point_gpu_lost_solutions \
  --database <fixed matching.db> \
  --output artifacts/evidence/lost-solutions-<date>.json
```

The run refuses to overwrite and asserts the input digest and both solver
signatures before analysis.
