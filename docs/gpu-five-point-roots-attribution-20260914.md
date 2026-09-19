# roots cost attribution and Q1 failure classification (2026-09-14)

## Decision and scope

Measurement-only round. No solver, shader or previously protected harness source
was changed, so there is no candidate to retain or roll back. The round answers
two questions that the P1 round left open:

1. Where does the `roots` kernel time go, now that it is the largest single cost
   at 33–36 ms of a 54–57 ms nine-kernel sum?
2. What do the failure counters actually mean, specifically the TODO's Q1
   constraint "不把所有 `NotConverged` 解释成达到迭代上限"?

Both are answered with the existing GPU f32 output. Still independent offline
candidate generation: not RANSAC, not the 960-image pipeline, and the CPU f64
replay here is an offline reference, never a runtime fallback.

Two findings change the roadmap and are stated up front:

- `roots` spends 24–27 ms of its 34–36 ms on SIMD lockstep divergence. Simply
  reordering the input trials by difficulty, changing no arithmetic, cuts the
  kernel from 33.7 ms to 9.3 ms and the nine-kernel sum from 54.1 ms to 29.3 ms.
- The `unconverged` counter does not measure convergence. 93.0% of all
  `NotConverged` slots are rejected by a gate that evaluates the polynomial at
  the real part of the root alone, so legitimately complex roots are labelled
  `NotConverged` before the dedicated realness test can label them `Complex`.
  Genuine polish rejection in trials that converged is 130 slots out of 297,822.

## Provenance

- Worktree `.worktrees/gpu-five-point-f32`, source commit `3ab9c09`, unchanged
  solver and shaders.
- New file only: `RustSFM/examples/five_point_gpu_roots_attribution.rs`,
  sha256 `1c838e511f4058471f1c40b4847717b30db519288a5c40f5d6b48c7c17a51a32`;
  release binary sha256
  `8aab2082f352a5a0aa27ac35a543459ba93a104971132ee23ba59343bd6597c5`.
- Device `Apple M5 Max`, wgpu backend, timestamp-enabled context for kernel
  intervals and a separate unprofiled context for the decoded reference.
- Fixed input unchanged from the B0/P1 rounds: 127 pairs, 231 unique images,
  65,024 trials, digest
  `af07459d637b5b8c2b20c76d3ef103f058505fe0e8307578ee7e8750f3e230c5`.
- GPU signature `1f6f3f8d…39` and CPU f64 signature `9e6764b4…3a` both asserted
  equal to their historical values, so this round is comparing the same objects
  as P1.
- Build note: this worktree's `third_party/PoseLib` is an empty submodule
  directory, so the build needs `POSELIB_ROOT` pointed at the main checkout.
  The poselib bridge is not on the GPU five-point path and cannot affect these
  numbers, but the flag is required to reproduce.

Artifacts, all three runs kept:

| file | source | purpose |
| --- | --- | --- |
| `experiments/roots-attribution-20260914.json` | first version | timings, before the `NotConverged` split existed |
| `experiments/roots-attribution-20260914b.json` | + `NotConverged` split | second timing sample |
| `experiments/roots-attribution-20260914-final.json` | + clippy `contains` rewrite | reported run, final binary |

Every distribution and classification field is bit-identical across all three
runs; only the kernel timings differ. That is the reproducibility evidence for
the classification numbers below.

## Method

### Divergence experiment

`roots` is dispatched at `workgroup_size(32)` with one invocation per trial, so
32 consecutive trials share a SIMD group and advance in lockstep. The Aberth
loop in `five_point_recovery.wgsl` runs up to 384 iterations and breaks on
`all_done`, which is the conjunction of `done[i]` over that trial's own roots.
A lane that finishes early is masked but the group keeps issuing, so the group
costs its maximum, not its mean.

The experiment tests this by permuting only the order of the input trials and
running three orders on the same context and the same single 65,024 call:

- `identity`, the historical order;
- `shuffled_control`, a seeded shuffle, to show the historical order is not
  special;
- `sorted_by_iterations_desc`, trials sorted by the iteration count measured in
  a previous run, which clusters difficulty and makes groups homogeneous.

Each run reverses its permutation, restores the positional trial index and
requires the exact historical decoded bit signature. All three passed, so any
timing difference is scheduling and not arithmetic.

The sorted order is an **oracle**: it needs per-trial iteration counts that are
only known after solving. It is not an implementable policy. It measures the
headroom that any difficulty-aware grouping could compete for.

### Separating the three `NotConverged` paths

`five_point_recovery.wgsl` writes slot status `3.0` (`NotConverged`) from three
places in the post-loop slot pass:

| line | gate | note |
| --- | --- | --- |
| 85 | `final_eval` at `vec2(root.x/radius, 0.0)` | evaluates at Re(z) only, discarding Im(z) |
| 88 | `!done[i]` | the loop's own convergence flag |
| 105 | real Newton polish residual | after polish |

They are separated without changing the shader, using two facts.

First, line 85 rejects with `continue` **before** the root and backward error are
stored at line 86, while lines 88 and 105 run after that store. So a
`NotConverged` slot whose root is exactly `(0, 0)` with zero backward error was
rejected at line 85, and one with a stored root was rejected at line 88 or 105.
This depends on the output buffer starting at zero, which after P1.2 is provided
by wgpu's device-side initialization.

Second, a trial whose `root_iterations` is below 384 left the loop through
`all_done`, which requires `done[i]` for every root, so line 88 cannot have
fired for it. For those trials a stored-root `NotConverged` is line 105 alone.

One edge is acknowledged: `root_iterations` is written every iteration, so a
trial that converged exactly on the last iteration also reads 384 and lands in
the at-cap bucket. That only makes the below-cap counts conservative.

## Iteration distribution

54,988 trials entered the loop; 10,036 never did, because they failed upstream.

| iterations | trials | share of entered |
| --- | --- | --- |
| 1–9 | 602 | 1.1% |
| 10–19 | 30,776 | 56.0% |
| 20–29 | 14,241 | 25.9% |
| 30–49 | 2,439 | 4.4% |
| 50–99 | 647 | 1.2% |
| 100–199 | 382 | 0.7% |
| 200–383 | 272 | 0.5% |
| 384 (cap) | 5,629 | 10.2% |

Median 17, p90 74, p99 384. Mean over entered trials 58.5, which is more than
three times the median — the mean is carried entirely by the capped tail.

This is the mechanism in one line: **8.7% of trials hit the cap, but they
contaminate 93.0% of the 2,032 SIMD groups.**

## Divergence model and measurement

Charging each group its maximum, for the loop only:

| SIMD width | ideal lane-iterations | lockstep | lockstep / ideal |
| --- | --- | --- | --- |
| 8 | 3,217,825 | 14,161,328 | 4.40× |
| 16 | 3,217,825 | 19,843,056 | 6.17× |
| 32 | 3,217,825 | 23,667,680 | **7.36×** |
| 64 | 3,217,825 | 24,889,920 | 7.74× |

Sorting brings the width-32 lockstep count down to 3,223,872, within 0.2% of
the ideal 3,217,825, so the model's predicted ceiling is 7.34×.

Measured, three runs, medians of five samples after three warmups:

| order | `roots` (ms) | `recover` (ms) | nine-kernel sum (ms) | per-trial bits |
| --- | --- | --- | --- | --- |
| identity | 35.95 / 36.18 / 33.75 | 6.52 / 6.76 / 6.28 | 56.65 / 57.06 / 54.07 | identical |
| shuffled control | 35.06 / 35.08 / 33.55 | 6.52 / 6.78 / 6.32 | 55.60 / 56.65 / 53.59 | identical |
| sorted by iterations | **9.42 / 9.54 / 9.33** | 6.45 / 6.74 / 6.35 | **29.94 / 30.61 / 29.35** | identical |

Readings:

- The hypothesis holds. `roots` improves 3.62×, 3.79× and 3.82× from reordering
  alone, saving 24.4–26.6 ms; the nine-kernel sum improves 1.84–1.89×.
- The shuffled control lands on top of identity, so the historical order is not
  accidentally good or bad — the waste is structural, not an artifact of the
  sampler's ordering.
- `recover` is the negative control. It is dispatched the same way but is not
  iterative, and within each run it moves at most 0.07 ms across the three
  orders, while drifting 0.5 ms between runs. So the reordering did not simply
  make the whole GPU faster, and the run-to-run drift is comfortably smaller
  than the 24 ms `roots` effect.
- Measured 3.6–3.8× against a modelled 7.34× ceiling. The gap is expected: the
  model covers only the iteration loop, while the kernel also pays per-trial
  fixed work (radius estimate, coefficient loads, the 10-slot post-loop pass)
  that reordering cannot remove. The sorted `roots` time of 9.3 ms is a
  reasonable estimate of that irreducible floor.

Caveat on what this buys end to end: kernel intervals come from a timestamp
context and overlap host wait, so they are not additive with the host phases
measured in P1. A 26 ms kernel saving is an upper bound on the end-to-end gain,
realisable only to the extent that host `wait` is kernel-bound. This was
measured at 65,024 trials in a single call. At batch 512 there are only 16 SIMD
groups and, per the P1 round, per-call latency dominates; nothing here applies
to that case.

## Q1: what the counters mean

Primary stage that ended each trial, mutually exclusive. `upstream_status`
carries the nullspace status when the nullspace failed and the algebra status
otherwise; the two kernels emit disjoint code sets, so the stage is unambiguous.

| primary category | trials | share |
| --- | --- | --- |
| nullspace `RankDeficient` | 10,035 | 15.43% |
| algebra `SingularElimination` | 1 | 0.00% |
| reached roots, every candidate rejected | 1,976 | 3.04% |
| reached roots, produced models | 53,012 | 81.53% |

`polynomial_failed` is zero for all 65,024 trials: the non-finite and
degenerate-coefficient guards in the root setup never fire on this input. The
nullspace is the only meaningful upstream failure, and it is always
`RankDeficient` — never `NotConverged` or `InvalidBasis`.

Root slots, 650,240 total = 65,024 × 10, and `degree` is 10 for essentially
every trial that entered the loop (549,479 roots over 54,988 trials):

| slot status | slots | of computed roots |
| --- | --- | --- |
| `Accepted` | 209,600 | 38.1% |
| `NotConverged` | 297,822 | 54.2% |
| `Complex` | 21,468 | 3.9% |
| `InvalidEssential` | 12,458 | 2.3% |
| `NullVectorFailure` | 7,733 | 1.4% |
| `DuplicateRoot` | 398 | 0.1% |
| `Unused` | 100,761 | — |

### The `NotConverged` label is mostly the complex-root filter

Splitting the 297,822 `NotConverged` slots by the method above:

| path | slots | share |
| --- | --- | --- |
| line 85, real-axis `final_eval` gate, root never stored | **276,906** | **93.0%** |
| line 88/105, at cap, polish or `done[]` | 20,786 | 7.0% |
| line 105, below cap, polish only | **130** | **0.04%** |

This settles the Q1 constraint quantitatively. Of the slots labelled
`NotConverged`, at most 7.0% can involve the iteration cap at all, and the
genuine "the loop converged but polish rejected it" case is 130 slots out of
297,822. Raising the 384 cap or improving Aberth convergence would address a
small minority of a label that is 93% something else.

What line 85 actually does: it evaluates the polynomial at `Re(z)` alone,
discarding the imaginary part. A root the loop legitimately converged to as a
complex number therefore fails it, and because line 85 precedes the dedicated
relative realness test at line 90, that root is recorded as `NotConverged`
rather than `Complex`. The root budget confirms this reading arithmetically:

```
accepted 209,600 + Complex 21,468 + real-axis gate 276,906 = 507,974  (92.4% of 549,479)
remainder 41,505 = InvalidEssential 12,458 + NullVectorFailure 7,733
                 + at-cap 20,786 + below-cap polish 130 + DuplicateRoot 398
```

Per trial that entered the loop: 3.81 accepted, and 5.43 rejected by the two
complex filters combined. For a degree-10 polynomial with roughly four real
roots, that is exactly the expected real/complex split. The real-axis gate is
the de facto complex filter: `complex_filtered` at 21,468 under-reports the
complex rejections by 13.9×, and `unconverged` at 297,822 over-reports the
convergence-related rejections by 14.2×.

A smaller item worth keeping: 9,307 of the stored-root `NotConverged` slots sit
*inside* the line-90 realness threshold, so they are real roots rejected by
polish or by `done[]`, not complex ones.

### Where the lost solutions actually come from

11,967 trials have at least one CPU f64 model and no GPU model. Their stage:

| primary category | trials | share of lost |
| --- | --- | --- |
| nullspace `RankDeficient` | **10,028** | **83.8%** |
| reached roots, every candidate rejected | 1,938 | 16.2% |
| algebra `SingularElimination` | 1 | 0.0% |
| of which never entered the loop | 10,029 | 83.8% |

This is the round's second roadmap change. The dominant f32 quality loss is not
in the root solver and not in recovery — 83.8% of lost solutions never reach the
polynomial at all, failing the f32 nullspace as rank deficient where f64
succeeds. Effort spent on root iteration, polish thresholds or recovery
tolerances cannot recover them.

## Correctness gates

| gate | result |
| --- | --- |
| `cargo fmt -p rustsfm -- --check` | pass |
| `cargo clippy … --example five_point_gpu_roots_attribution --all-features` | pass, no warning in the new file |
| `cargo test … --example five_point_gpu_roots_attribution` | pass, 13 tests |
| `cargo test -p rustsfm --lib five_point` (actual GPU) | pass, 8 tests |
| `git diff --check` | pass |
| GPU signature vs historical | identical |
| CPU f64 signature vs historical | identical |
| per-trial bits under all three orders | identical |
| classification fields across three runs | identical |

The four clippy warnings that remain under this example come from the shared
`five_point_gpu_capacity_replay.rs` included via `#[path]` and are pre-existing,
as recorded in the P1 round.

The new unit tests pin the analysis logic rather than the measured values: the
lockstep charge on a partial trailing group, order sensitivity of the model
against order invariance of the ideal count, the sorting permutation being a
deterministic bijection, and the zero-root predicate the `NotConverged` split
depends on.

## Consequences for the plan

The TODO currently has P2 (persistent GPU session) as the whole next round, on
the basis that per-call `wait` and `readback` dominate at batch 512. That
finding stands and P2's premise is unchanged. What this round changes is the
relative value of the remaining items.

1. **The roots divergence item should be promoted.** The TODO defers
   "协作式 roots 求解" until after P1–P3, pending a new profile. This is that
   profile, and it shows ~26 ms of a ~54 ms nine-kernel sum is recoverable
   scheduling waste at large batch — larger than anything P2 can reach in the
   kernels. It remains subordinate to P2 for the batch-512 case, which is
   latency-bound, so the ordering is defensible; but it should no longer be
   parked without a number attached.

2. **Q1's diagnostic fix is now specific and small.** Rather than a general
   reclassification, the concrete defect is that line 85 pre-empts line 90. The
   minimal change is to order the realness test before the real-axis evaluation,
   or to give the real-axis gate its own status code. Either is a diagnostic
   change that must not alter the candidate models; per the TODO this requires a
   versioned diagnostic signature with the model signature verified separately.

3. **Q2 should start at the nullspace, not the root solver.** 83.8% of lost
   solutions are f32 nullspace `RankDeficient`. That matches the TODO's existing
   "评估直接 f32 零空间算法（如 QR），避免 AᵀA 条件数恶化" and now has the
   evidence to make it Q2's first item rather than one of several.

4. **P2 has a newly discovered coupling to flag.** The `NotConverged` split
   above relies on the output buffer starting at zero so an unwritten root is
   distinguishable. P1.2 made that a device-side guarantee of freshly created
   buffers. Once P2 reuses buffers, this diagnostic silently stops working
   unless the `out` reset is explicit and verified. That strengthens the
   prerequisite the TODO already records for P2, and adds a reason to keep this
   harness runnable as a check on it.

## Unresolved risks

- The sorted order is an oracle and not a policy. The 3.6–3.8× figure is
  headroom, not a projected gain. Any real scheme must predict difficulty before
  solving, or restructure the kernel, and will pay overhead the oracle does not.
- Kernel intervals overlap host wait and are not additive with host phases, so
  the end-to-end share of a 26 ms kernel saving is not established here.
- Everything here is at 65,024 trials in one call. Batch 512 has 16 SIMD groups
  and different structure, and is untouched by this round.
- The line-85/88/105 split rests on reading the shader plus the zero-initialized
  buffer. It is a strong inference but not a direct instrumented count; a
  diagnostic counter under Q1 would confirm it independently.
- That complex roots are currently mislabelled does not by itself mean models
  are lost — they would very likely be rejected at line 90 anyway. The claim
  here is about the counters being misleading, not about quality. Whether the
  real-axis gate's `2e-5` relative bound also rejects roots that line 90 would
  have kept is not measured.
- The 11,967/1,938 root-stage losses are not attributed further than "every
  candidate rejected". Their internal breakdown needs the Q1 diagnostic change.
- Quality remains non-equivalent to CPU f64 and nothing in this round changed
  it.

## Reproduction

```
POSELIB_ROOT=<main-checkout>/third_party/PoseLib \
  cargo build --release --target-dir target -p rustsfm \
  --example five_point_gpu_roots_attribution

./target/release/examples/five_point_gpu_roots_attribution \
  --database <fixed matching.db> \
  --output experiments/roots-attribution-<date>.json
```

The harness refuses to overwrite an existing output and asserts the input
digest, the GPU signature and the CPU f64 signature before reporting, so a
provenance drift fails the run rather than producing a comparable-looking
number.
