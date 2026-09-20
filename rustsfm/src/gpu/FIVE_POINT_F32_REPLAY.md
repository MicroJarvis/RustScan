# Complete GPU f32 five-point: implementation and actual replay

## Scope and conclusion

The independent GPU path is complete through **essential-matrix candidate
recovery**, not just elimination stages. It does not call CPU numerical solvers,
production RANSAC, root fallbacks, or production dispatch. It is **not numerically
equivalent to the CPU f64 solver and is slower in this measured workload**.
Completing this experiment is not a recommendation to deploy it.

Implementation: `five_point_f32_complete.rs`, `shaders/five_point_recovery.wgsl`,
existing generated/stage shaders, and `WgpuFivePointF32::solve_essential`.
The new `examples/five_point_gpu_replay.rs` imports the existing independent CPU
probe as a module. The only changes to that probe expose its existing helpers and
a read-only loader variant retaining pair observations; selection, seed, sampler,
ray preparation, and CPU solver are shared rather than reimplemented. No library
geometry, production mapper, Cargo feature, or build.rs changes were made.

## Numerical policy and failure contract

All solver arithmetic below runs on the GPU in f32. Host code validates input,
allocates/binds buffers, dispatches, waits, reads, and decodes. `solve_essential`
uploads rays once and reads only final records; intermediate constraints, basis,
200-entry elimination, solve/B, coefficients, and roots stay in GPU buffers.
Each trial has ten stable slots, including filtered/nonconverged slots. APIs return
batch-relative trial IDs; replay adds the batch offset and stores pair/trial IDs.

### Roots

- Descending coefficients `z^10 .. z^0`. Leading coefficients at or below
  `1e-12 * max(abs(coefficients))` are dropped, with degree/drop count reported.
  Zero, nonfinite, and degree-zero polynomials fail explicitly. Degree reduction
  can lose roots at large magnitudes; it is not an algebraic equivalence claim.
- Complex **Aberth-Ehrlich** iteration, simultaneous/Jacobi updates, deterministic
  circle initialization with phase offset 0.37. Fujiwara radius scales `z=R*w`;
  coefficients are divided in steps to avoid computing `R^10` directly.
- **384 iterations maximum**. Complex division is scaled; large updates are capped.
  A root converges only when the last step in original coordinates is at most
  `2e-6 * (1+abs(z))`, the scaled polynomial backward error is at most `2e-6`,
  and the root is finite. Unconverged slots are reported and never recovered.
  Converged slots may still be used when other roots reach the limit.
- Realness: `abs(imag(z)) <= 2e-4 * (1+abs(real(z)))`, intentionally different
  from CPU f64's absolute `1e-10`. Up to eight real Newton polishing iterations;
  finite/capped steps only, and a step must not worsen relative backward error.
  Polished roots must satisfy backward error `<=2e-5`.
- Roots within `2e-3 * (1+abs(z))` of an earlier retained real root are reported
  as `DuplicateRoot` and excluded. This means an **unresolved close-root cluster**,
  not certified exact multiplicity. Distinct nearby roots can be merged.
  The conservative tolerance handles f32 repeated-root cancellation: the `(z-1)^2`
  fixture otherwise produced two apparent real roots about 0.001 apart.

`polynomial_failed` means invalid/constant polynomial, while `unconverged` and slot
statuses describe iteration failure. Upstream numerical failure, algebra status,
root iterations, degree, rank, and Jacobi sweeps are separate fields. A successful
upstream stage does not imply that all roots converged or that any E was accepted.

### 3×3 null vector and E recovery

- Evaluate the CPU reference's 13×3 B layout at each real z, with columns of
  degrees 3, 3, 4. Scale B by its largest absolute entry.
- Compute the smallest right singular vector via symmetric Jacobi on BᵀB,
  at most **24 sweeps**, convergence max off-diagonal `<2e-7` after scaling.
  Zero/nonfinite B or failure to converge is a null-vector failure. Require
  `norm(B*x)/norm(B) <=2e-3` and `abs(x[2]) >=1e-6`.
- Recover `E = N0*x0/x2 + N1*x1/x2 + N2*z + N3`, exactly the CPU chart; reject
  invalid norm, normalize to unit Frobenius norm, keep row-major coefficients.
- Require normalized sample `A*E` residual `<=5e-4` and essential cubic residual
  `norm(2 E Eᵀ E - E) <=2e-2`. No silent SVD projection or model replacement.
  Rejections are `NullVectorFailure` or `InvalidEssential`.
- Accepted E within sign-invariant distance `2e-4` of an earlier accepted E is
  `DuplicateModel`. No slot compaction. At most ten models per trial.

The existing AᵀA nullspace stage still has conservative numerical rank checks
(eigenvalue threshold `1e-6 * trace`). It rejects some exact-rank-five systems in
f32. The GPU does not retry them on CPU. A failed packed basis subsequently gives
singular elimination; these two counters describe the **same** failed trials and
must not be added. Full-path input rays must be finite with abs components `<=1e8`
to bound downstream residual arithmetic; replay supplies normalized unit rays.

## Focused actual GPU tests

```sh
cargo test -p rustsfm --lib --no-default-features --features gpu-wgpu five_point_f32_actual_gpu_stages -- --nocapture --test-threads=1
cargo test -p rustsfm --example five_point_gpu_replay --no-default-features --features gpu-wgpu
```

The first test actually initializes Apple M5 Max, compiles all shaders, executes
and reads back the full solver. A missing GPU is an error, not a skipped test.
It retains previous same-basis f64 stage checks and adds:

- known ten distinct real roots; `z²+1`; `(z-1)²`; zero polynomial;
  `(z-1)^10` explicitly reaching 384 iterations with nonconverged roots;
- full synthetic/random 16-trial recovery, unit E norm, independently evaluated
  cubic and A*E residuals, nearest CPU models, fixed trial/slot identity;
- failed trials mixed between valid trials, confirming no compaction or cross-trial
  corruption, zero batches and invalid input.

The replay tests reuse the CPU probe's sampler/order tests and add model-sign and
Sampson-mask invariance checks. These focused tests pass; they are not a broad
numerical certification. Existing unrelated warnings were left unchanged.

## Reproduce the actual benchmark

Run in `/Users/tfjiang/Projects/RustScan/.worktrees/gpu-five-point-f32`:

```sh
cargo build -p rustsfm --release --example five_point_gpu_replay --no-default-features --features gpu-wgpu
# Optional small debug workload; also performs warmups and three interleaved rounds.
target/release/examples/five_point_gpu_replay \
  --database /Users/tfjiang/Projects/RustScan/artifacts/runs/flowers2_960_settlement_20260913/matching.db \
  --pairs 2 --trials 16 --rounds 3 --output artifacts/runs/five_point_gpu_debug.json
# Final measured workload (both batch sizes are run by the example).
target/release/examples/five_point_gpu_replay \
  --database /Users/tfjiang/Projects/RustScan/artifacts/runs/flowers2_960_settlement_20260913/matching.db \
  --pairs 12 --trials 512 --rounds 3 --output artifacts/runs/five_point_gpu_replay_12x512_final.json
```

DB existence was verified: about 1.1 GB, opened **read-only**. 14,045 raw-match
pairs have at least five matches. Select midpoints of 12 equal contiguous strata
of sorted eligible pair IDs. Per pair, one stateful `ColmapRandomSampler`, seed 1,
512 prefix trials, never reset per trial. Only raw matches are loaded, not stored
verified inliers. Invalid keypoint/projection entries are filtered exactly as in
the existing probe; all resulting pair observations are retained for mask checks.

Final artifact: `artifacts/runs/five_point_gpu_replay_12x512_final.json` (22,494,838 bytes).
It contains every trial's sample indices, CPU/GPU counts, retained GPU slot IDs,
all root/slot statuses, directional nearest distances, constraint residuals, and
nearest-model mask comparisons, plus initialization/warmup/round timings.
The generated output directory is ignored by Git; this document records the
measured summary without adding a 22 MB benchmark artifact to source control.
Input fingerprint:
`3049eeb4981ff41bed23a9cab5adc59ce6c1e642d9b4360144d8aef2d221510c`.

### Timing contract

6144 pre-gathered f64 five-ray pairs per pass. CPU uses the unchanged full f64
solver serially or in a locally owned four-thread Rayon pool. GPU timing includes
f64→f32 conversion, host input collection, buffer allocation/upload, every kernel,
queue waits, final readback, decoding and result collection. There is no CPU
numerical completion. Outputs live until the timer stops. DB I/O, ray preparation,
sampling, pool/GPU initialization, warmup, output destruction after timing,
comparison/mask evaluation, and JSON writing are excluded from solve timing.
CPU uses original f64 rays, GPU quantizes to f32: comparison includes this difference.

One full warmup per path **per batch size**. Three interleaved measured rounds:
serial→4threads→GPU, GPU→serial→4threads, 4threads→GPU→serial.
CPU serial/4-thread output coefficients and order matched bit-for-bit throughout.
GPU trial/slot/root/model signatures repeated exactly and matched across batch
64/512; this is repeatability, not CPU equivalence.

### Final measurements: Apple M5 Max, 2026-09-13

Process-first GPU context/module/pipeline initialization: **0.1612025 s**, excluded
from timed solves. This is not a disk-cache-purged cold Metal compiler measurement.
The earlier small debug process measured 0.5474 s initialization.
Final build/test/replay was run with a **180 s command bound** and completed.

Times below are milliseconds for all 6144 trials, median of three rounds:

| Batch | CPU serial | CPU 4 threads | Complete GPU, upload/readback included |
|---|---:|---:|---:|
| 64 | 70.637 | 21.295 | 1062.966 |
| 512 | 63.501 | 19.460 | 144.572 |

Raw round times (ms):

| Batch/path | Round 0 | Round 1 | Round 2 |
|---|---:|---:|---:|
| 64 serial | 70.786 | 69.627 | 70.637 |
| 64 four-thread | 21.295 | 21.288 | 21.305 |
| 64 GPU | 1060.185 | 1067.110 | 1062.966 |
| 512 serial | 63.674 | 63.462 | 63.501 |
| 512 four-thread | 19.461 | 19.453 | 19.460 |
| 512 GPU | 144.955 | 142.844 | 144.572 |

Warmup seconds (serial / four-thread / GPU): batch64
`0.064570 / 0.022422 / 1.078005`; batch512
`0.063815 / 0.019688 / 0.145776`.
Batch512 GPU is about **2.28× slower than serial, 7.43× slower than four-thread**,
while producing fewer/different candidates. No equivalent-work speedup is claimed.

### Candidate counts, failures, differences

Identical comparison results for both batch sizes:

- CPU: **26,490** models. GPU: **20,219** models (76.33% of the CPU count, not recall).
- Counts differ in **2,136 / 6,144 trials (34.77%)**; equal in 4,008.
  GPU has more models in 15 trials and fewer in 2,121.
- GPU emits no model in **974 trials (15.85%)**; CPU emits none in five.
- **756** GPU numerical-rank failures, also reported as singular downstream
  elimination. **478** trials reach the 384-iteration root limit.
- Among attempted roots: **2,174 nonconverged**, **29,644 complex-filtered**,
  **30 duplicate/close roots**, **1,752 recovery rejects** (780 null-vector failures,
  972 invalid-essential failures), **0 duplicate models**, **0 invalid-polynomial
  trials**. Degree reduction: 55 trials to degree nine, three to degree eight.
- **3,982 CPU models** have no GPU counterpart because their trial has no GPU
  models. No GPU model has an empty CPU counterpart. These are counted separately,
  not inserted as zero distances and not included in the distance quantiles.

Distance is `min(||E-F||, ||E+F||)` after separate unit Frobenius normalization,
with nearest neighbors independently in **both directions within each trial**.
No common basis, root ordinal, or slot matching is assumed.

| Conditional model distance | Median | P90 | P99 | Maximum |
|---|---:|---:|---:|---:|
| CPU→GPU (22,508 models with counterpart) | 7.67e-5 | 0.2656 | 1.3918 | 1.4142 |
| GPU→CPU (20,219 models) | 5.24e-5 | 0.002935 | 0.2220 | 1.3461 |

Small medians do **not** conceal the substantial missing-model and long-tail
failures. In particular, a small constraint residual alone does not establish
correct essential recovery or complete root enumeration.

CPU original-ray normalized constraint residual: median `8.96e-18`, max `1.16e-16`.
GPU models evaluated independently on those original f64 rays: median `1.33e-7`,
P99 `3.78e-6`, max **`1.30e-5`**.

### Full-pair raw-observation Sampson masks

This is a CPU f64 **diagnostic** applied equally to both model sets, outside solve
timing. Project rays to normalized image coordinates (`z=1`), and threshold
squared Sampson distance at **`1e-6`**. This is not a pixel threshold and does not
claim production threshold parity. The same threshold is used across cameras.
For each directional nearest-model association, compare masks over **all valid
raw observations of that pair**, not only the five sampled rays or verified inliers.
No nearest counterpart is explicitly null; it is not a perfect-agreement mask.

42,727 directional mask comparisons: median Hamming fraction 0, P90 **0.00909**,
P99 **0.45415**, max **0.94872**, mean **0.01619**. The artifact also records supports,
Hamming counts and Jaccard per association. This has serious long-tail disagreement
and cannot justify a full-RANSAC equivalence claim.

### Per-pair counts (512 trials each)

| Pair ID | Valid raw observations | CPU models | GPU models | GPU-empty trials | Count-mismatch trials |
|---|---:|---:|---:|---:|---:|
| 79456894986 | 229 | 2236 | 1824 | 59 | 150 |
| 236223201320 | 79 | 2180 | 1948 | 34 | 89 |
| 392989507744 | 178 | 2238 | 1698 | 94 | 156 |
| 551903297537 | 3998 | 2246 | 871 | 181 | 442 |
| 708669603880 | 165 | 2236 | 1935 | 41 | 113 |
| 876173328464 | 110 | 2060 | 1780 | 57 | 104 |
| 1043677053008 | 155 | 2228 | 1928 | 46 | 103 |
| 1211180777552 | 175 | 2244 | 1935 | 48 | 105 |
| 1380831985665 | 3313 | 2154 | 888 | 199 | 427 |
| 1559073128457 | 257 | 2220 | 1817 | 66 | 150 |
| 1739461754960 | 110 | 2248 | 1989 | 34 | 105 |
| 1937030250504 | 307 | 2200 | 1606 | 115 | 192 |

## What is not established

No full pair RANSAC loop, adaptive stopping, scoring-based selection, local
optimization, pose decomposition, or 960-image reconstruction was run. This is a
fixed **sampler prefix**, not a trace of executed production RANSAC trials.
Other adapters, certified multiplicities/all-root completeness, f64-equivalent
rank decisions, and acceptable production model/mask agreement remain unproven.
The complete GPU pipeline and the requested replay experiment are implemented;
numerical parity and acceleration are explicitly **not achieved**.
