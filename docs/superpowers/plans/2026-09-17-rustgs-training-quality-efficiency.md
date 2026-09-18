# RustGS Training Quality and Efficiency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the RustGS wgpu training loop measurably faster while preserving or improving novel-view quality, with numerical parity gates for every hand-written differentiable GPU path.

**Architecture:** Keep the normal training path device-resident for forward counts, loss, sorting, scan, and topology selection whenever possible. Treat topology and rendering as separate ownership boundaries: topology may receive compact candidate records, while render and optimizer state remain on the GPU. Quality work begins with analytic-gradient verification; changes to densification, robustness, exposure, or filtering are experiment-gated configuration options until all scene gates pass.

**Tech Stack:** Rust 2021, Burn, burn-wgpu, CubeCL/WGSL, wgpu timestamp queries where supported, serde JSON, existing RustGS CLI and evaluation module.

**Spec:** `docs/RustGS-TODO-训练效果与效率优化-2026-09-17.md`

## Global Constraints

- Do not treat existing P0-01, P0-02, P1-01, P1-02, P1-03, P1-05, or P1-06 code as accepted: each requires the quality and performance gates below.
- Preserve a fixed baseline command, dataset manifest, frame order, seed, render scale, GPU/driver, and evaluation frame set for every candidate comparison.
- Every candidate report records wall-clock, median training-loop time, steps/s, peak VRAM, initial/final Gaussian count, intersection count, mean/median/minimum PSNR, worst frame id, gradient and Laplacian sharpness ratios.
- A candidate may not change a default unless it passes TUM full-view, Home smoke, and one external COLMAP scene. Its worst-frame PSNR must not decline by more than `0.2 dB`.
- A short 500-step run is a tuning filter, not default-selection evidence. Run 3k and 10k before 30k.
- Keep the CPU/reference path available while introducing fused GPU kernels or device-native topology work. Compare finite values and output ordering before deleting it.
- Overflow, non-finite loss, invalid indirect dispatch dimensions, and topology count mismatch are hard failures with an emitted JSON diagnostic; never silently clamp and continue.
- Add CLI/config fields additively, serialize them in checkpoints, validate them in `TrainingConfig::validate`, and retain present defaults.

---

## Files and Boundaries

| Area | Primary files | Responsibility |
| --- | --- | --- |
| Forward scheduling | `RustGS/src/training/forward/{mod.rs,dispatch.rs,sorting.rs,tile_mapping.rs}` | Device counts, bounded capacity, indirect dispatch, sort and tile-map sequencing. |
| GPU primitives | `RustGS/src/training/gpu_primitives/{device_radix.rs,prefix_sum.rs}` and `RustGS/src/training/shaders/{radix_histogram.wgsl,radix_scatter.wgsl,scan_block.wgsl,scan_add.wgsl}` | Stable key/value ordering and inclusive scan with reusable workspaces. |
| Training and reporting | `RustGS/src/training/engine/{trainer.rs,runtime.rs,loss.rs}`, `RustGS/src/training/reporting/metrics.rs`, `RustGS/src/bin/rustgs/train_command.rs` | Iteration policy, data selection, loss, telemetry, CLI JSON. |
| Topology | `RustGS/src/training/topology/{bridge.rs,apply.rs,mod.rs,schedule.rs}` and `RustGS/src/training/shaders/accumulate_topology_stats.wgsl` | Accumulators, candidate selection, mutation, optimizer continuity. |
| Differentiable renderer | `RustGS/src/training/shaders/{project_forward.wgsl,project_backwards.wgsl,helpers.wgsl,rasterize.wgsl}`, `RustGS/src/training/backward/{autodiff.rs,project_bwd.rs}` | Projection/raster VJP correctness and anti-aliasing. |
| Data and evaluation | `RustGS/src/training/data/{frame_loader.rs,frame_targets.rs,init_map.rs}`, `RustGS/src/training/evaluation/{core.rs,parity.rs}` | Frame representation/cache, initialization, deterministic evaluation/gates. |

## Standard Experiment Protocol

Use a unique output directory per candidate and place `run.json`, CLI log, final `.ply`, and evaluation JSON below it. First build one release binary and use that same binary for baseline and candidate:

```sh
cargo build -p rustgs --release --bin rustgs
mkdir -p output/rustgs-optimization/2026-09-17
```

The short TUM filter is:

```sh
target/release/rustgs train \
  --input output/rustgs_benchmark_pack/tum_freiburg1_xyz_colmap \
  --output output/rustgs-optimization/2026-09-17/<name>/tum-500.ply \
  --iterations 500 --render-scale 1.0 --frame-shuffle-seed 1592639710 \
  --eval-after-train --eval-render-scale 0.5 --eval-frame-stride 10 \
  --eval-worst-frames 10 --eval-json
```

Run the same command without candidate flags for `baseline`. For accepted short-run candidates, repeat at 3k, 10k, and 30k using an identical frame set. The Home smoke is the checked-in 12-frame invocation in `docs/RustGS-Home-Quality-2026-09-03.md`; the external scene uses the same command shape with a different COLMAP input. Do not compare logs from different render scale, frame selection, or seed.

### Task 1: Close the device-count and delayed-loss gates (P0-01, P0-02)

**Files:**
- Modify: `RustGS/src/training/forward/{dispatch.rs,mod.rs,sorting.rs,tile_mapping.rs}`
- Modify: `RustGS/src/training/engine/{trainer.rs,runtime.rs}`
- Modify: `RustGS/src/training/reporting/metrics.rs`
- Test: `RustGS/src/training/forward/dispatch.rs`, `RustGS/src/training/engine/trainer.rs`

**Interfaces:**
- Consumes: `CountPolicy::{Exact, Bounded}`, `ForwardDispatch`, `MAX_BOUNDED_INTERSECTIONS`.
- Produces: `ForwardCapacityTelemetry { logical_visible: u32, logical_intersections: u32, capacity: u32, overflowed: bool }` sampled only at existing loss/checkpoint cadence; `TrainingIterationMetrics::loop_duration` and `::loss_readback` fields.

- [x] **Step 1: Add failing host-only policy tests.** Assert `Exact` reads the two count tensors, `Bounded` does not register either tensor for readback on an ordinary iteration, and an overflow sample produces a `TrainingError::ForwardCapacityExceeded` containing logical count and capacity.
- [x] **Step 2: Run `cargo test -p rustgs forward::dispatch --no-default-features`; confirm the new assertions fail before changing implementation.** _(Host-only overflow/policy tests also live under `reporting::metrics` for `--no-default-features`.)_
- [x] **Step 3: Make `write_forward_dispatch` own all normal-path count decisions.** The shader writes `min(num_intersections, capacity)` into indirect arguments, sets `overflow[0]` when the logical count exceeds capacity, and leaves exact host sizing restricted to evaluation/viewport callers. Remove any `into_data_async` use reachable from `CountPolicy::Bounded` except the scheduled telemetry sample.
- [x] **Step 4: Make loss sampling explicit.** Add `fn should_read_loss(iteration, total_iterations, cadence, checkpoint_due, paused) -> bool`; call host scalar readback only when it returns true. Keep the device finite flag in the backward dependency chain and consume it at the same cadence.
- [ ] **Step 5: Run the focused tests and a 500-step TUM baseline/candidate pair.** Accept only when no normal iteration performs count/loss readback, no overflow occurs, final PSNR delta is within `0.05 dB`, and the candidate has a smaller median loop duration. _(Focused unit tests pass. TUM pack missing; ran Home COLMAP 500-step candidate at `output/profile_home/colmap60/0` instead — see TODO experiment record. Still need an old-binary baseline for PSNR/steps/s delta.)_
- [ ] **Step 6: Commit the isolated gate.** `git commit -m "perf(rustgs): keep training counts and loss on device"`

### Task 2: Verify and harden device radix sort and hierarchical scan (P1-05, P1-06)

**Files:**
- Modify: `RustGS/src/training/gpu_primitives/{device_radix.rs,prefix_sum.rs,mod.rs}`
- Modify: `RustGS/src/training/shaders/{radix_histogram.wgsl,radix_scatter.wgsl,scan_block.wgsl,scan_add.wgsl,fill_u32.wgsl}`
- Modify: `RustGS/src/training/forward/{sorting.rs,tile_mapping.rs}`
- Test: `RustGS/src/training/gpu_primitives/{device_radix.rs,prefix_sum.rs}`

**Interfaces:**
- Consumes: a key/value input of logical length `len <= workspace.capacity`.
- Produces: `DeviceRadixWorkspace::sort(keys, values, len)` with stable ascending unsigned key order and `PrefixSumWorkspace::inclusive_scan(input, len)`; both expose `dispatch_count(len)` for telemetry.

- [x] **Step 1: Add deterministic GPU fixture tests for lengths `0, 1, 17, 255, 256, 257, 4093`.** Compare keys and paired values to a stable CPU sort; include duplicate keys, signed-depth bit patterns transformed to sortable keys, tail values beyond `len`, all-zero scan input, and wrapping `i32::MAX` scan input.
- [x] **Step 2: Run `cargo test -p rustgs gpu_primitives --no-default-features`; verify the tests fail for a deliberately bypassed scatter/scan result.** _(GPU fixtures require `--features gpu`; host dispatch-count comparisons run without adapter.)_
- [x] **Step 3: Reuse ping-pong and histogram storage by capacity.** Allocate only when requested capacity grows, clear only logical ranges, and pad sort tails with the terminal sortable key so no tail element can enter the logical output.
- [x] **Step 4: Use block scan, recursively scan block totals, then uniform-add each block prefix.** Return immediately for `len == 0`, launch no kernel for `len == 1`, and reject a block-count recursion depth that cannot be represented by dispatch dimensions.
- [x] **Step 5: Instrument sort/scan dispatch count and workspace bytes in the training metrics.** Record per-iteration totals and report p50/p95 in final JSON. _(Fields added on telemetry; live per-iteration accumulation still pending Task 5 report wiring.)_
- [ ] **Step 6: Run primitive tests, then TUM 500/3k.** Require exact primitive parity, fewer dispatches than `bitonic_dispatch_count(len)` and `hillis_steele_dispatch_count(len)`, unchanged visible/intersection counts, and PSNR delta within `0.05 dB`.
- [ ] **Step 7: Commit.** `git commit -m "perf(rustgs): validate reusable radix and scan primitives"`

### Task 3: Validate topology semantics and preserve optimizer continuity (P1-01, P1-02, P1-03)

**Files:**
- Modify: `RustGS/src/training/{config.rs,engine/{trainer.rs,optimizer.rs},topology/{mod.rs,schedule.rs}}`
- Modify: `RustGS/src/training/shaders/accumulate_topology_stats.wgsl`
- Test: `RustGS/src/training/{engine/{trainer.rs,optimizer.rs},topology/{mod.rs,schedule.rs}}`

**Interfaces:**
- Consumes: `TopologyMutationPlan::origins()`, raster visibility bit, and the scheduled topology disposition.
- Produces: `TopologyStepTelemetry { scheduled_steps, mutations, opacity_resets, skipped_no_eligible_candidates, accumulator_resets }`; `WgpuAdam::remap_origins(&[Option<usize>], sh_coeffs, sh_channels, device)`.

- [x] **Step 1: Add a three-splat topology fixture.** Its rows are: long-term invisible/low-opacity, continuously visible/high-opacity, and a split source. Assert `Weight` uses actual raster visibility, only the intended low-contribution row is pruned, and a no-candidate step preserves all accumulators.
- [x] **Step 2: Add Adam remap tests with nonzero moments and `step = 7`.** After origins `[Some(1), None, Some(0)]`, surviving rows must gather their original first/second moments, the new row must be zero, and `step` remains seven.
- [x] **Step 3: Run the focused tests and confirm they fail if visibility is replaced with constant one, accumulators are unconditionally cleared, or optimizer reset is used.**
- [x] **Step 4: Define `Weight` as history visibility plus opacity threshold; keep `VisibilityWeight` as the stricter contribution-aware mode.** Serialize both meanings in config help text and checkpoint identity so resumed runs cannot silently alter pruning semantics. _(Semantics wired; config help/checkpoint identity text still light.)_
- [x] **Step 5: Consume `SkipNoEligibleCandidates` in the trainer.** Reset accumulators only after mutation or opacity reset; emit all disposition counters in training events and final metrics.
- [ ] **Step 6: Run TUM 3k and 10k plus Home 1500.** Require optimizer-state tests to pass, final intersection count not to rise, and worst-frame PSNR no lower than baseline by `0.2 dB`.
- [ ] **Step 7: Commit.** `git commit -m "fix(rustgs): preserve topology history and adam state"`

### Task 4: Replace full topology snapshots with compact candidate records (P1-04)

**Files:**
- Modify: `RustGS/src/training/topology/{bridge.rs,mod.rs,apply.rs}`
- Create: `RustGS/src/training/topology/candidates.rs`
- Modify: `RustGS/src/training/engine/{trainer.rs,optimizer.rs}`
- Test: `RustGS/src/training/topology/{bridge.rs,candidates.rs,apply.rs}`

**Interfaces:**
- Produces: `TopologyCandidateRecord { source_idx: u32, score: f32, flags: u32 }` and `TopologyCandidateSnapshot { records: Vec<TopologyCandidateRecord>, logical_count: usize, readback_bytes: usize }`.
- Produces: `apply_mutations_from_origins` that gathers survivor tensors and Adam moments on the device; host reads the source parameters only for selected split rows during the first increment.

- [ ] **Step 1: Add a host fixture that compares plan rows generated from today’s full snapshot and from compact candidate records.** It must yield identical survivor order, split parents, prune set, opacity reset flag, and seeded split offsets.
- [ ] **Step 2: Run the fixture before implementation; it must fail because no compact snapshot exists.**
- [ ] **Step 3: Add a GPU candidate kernel after accumulator update.** It computes eligibility flags and score, atomically appends `TopologyCandidateRecord`, clamps at candidate capacity, and writes an overflow flag. Read back only the compact records, counts, and overflow; retain full `HostSplats` only behind a temporary reference-mode switch.
- [ ] **Step 4: Gather existing tensor rows and Adam moments with device index tensors.** Build new split rows from selected parents; preserve deterministic RNG by deriving the same `topology_rng_seed(base_seed, iteration)` on host for only selected parent ids.
- [ ] **Step 5: Emit snapshot/planning/apply durations and bytes.** Reject any count or origin mismatch between parameters, optimizer moments, age vectors, and invisibility vectors.
- [ ] **Step 6: Compare reference and compact mode on deterministic 500-step TUM, then run 3k.** Require identical topology event fingerprints for the fixture, at least 30% lower topology-pause time at 30k-scale splat count, and no quality regression.
- [ ] **Step 7: Commit.** `git commit -m "perf(rustgs): compact topology candidate readback"`

### Task 5: Establish reproducible profiling and a single quality report

**Files:**
- Modify: `RustGS/src/training/{reporting/metrics.rs,engine/{trainer.rs,runtime.rs},evaluation/{core.rs,parity.rs}}`
- Modify: `RustGS/src/bin/rustgs/train_command.rs`
- Create: `docs/RustGS-Optimization-Experiment-Record.md`
- Test: `RustGS/src/training/{reporting/metrics.rs,evaluation/core.rs}`

**Interfaces:**
- Produces: JSON `OptimizationReport { environment, command, train, topology, memory, evaluation }` written beside `--eval-json` output.
- `train` includes elapsed and p50/p95 loop time; `evaluation` includes the existing mean/median/min/max PSNR, worst frames, sharpness ratios, and a per-frame list.

- [ ] **Step 1: Add JSON serialization tests using a synthetic report with two frames and one topology event.** Assert all mandatory fields survive round-trip and missing metric samples are represented as `null`, not zero.
- [ ] **Step 2: Record timing at iteration boundaries using `Instant`; use wgpu timestamp queries only when adapter features expose them.** Record adapter name, backend, driver, timestamp availability, and peak device allocation when the runtime exposes it; otherwise set that field to `null`.
- [ ] **Step 3: Add `--optimization-report <path>`; have `--eval-json` derive a sibling report path when the explicit path is absent.** Write atomically through a temporary path and rename after the final evaluation completes.
- [ ] **Step 4: Add a report comparator command or example that reads baseline/candidate JSON and rejects a mismatched dataset fingerprint, frame set, seed, render scale, or evaluation resolution.** It prints deltas for every global constraint metric.
- [ ] **Step 5: Run the unit tests and generate a baseline report with the standard 500-step TUM command.**
- [ ] **Step 6: Commit.** `git commit -m "feat(rustgs): report comparable training experiments"`

### Task 6: Reduce frame-cache, resize, and upload cost (P2-01, P2-02)

**Files:**
- Modify: `RustGS/src/training/{config.rs,data/{frame_loader.rs,frame_targets.rs},engine/{trainer.rs,runtime.rs}}`
- Modify: `RustGS/src/bin/rustgs/train_command.rs`
- Test: `RustGS/src/training/data/{frame_loader.rs,frame_targets.rs}`

**Interfaces:**
- Replace frame-count-only cache control with `TrainingDataConfig::frame_cache_byte_budget: usize`; retain `frame_cache_capacity` as a deprecated upper bound for one compatibility release.
- `DecodedFrame` stores RGB bytes plus optional depth source; `FrameTargetCache` owns only resized `Tensor<GsDiffBackend, 3>` values keyed by `(frame_id, width, height, color_space)`.

- [ ] **Step 1: Add cache tests with three synthetic frames of unequal byte sizes.** Assert LRU eviction respects byte budget, an in-use target is not evicted, and a requested target is recreated at exactly the requested dimensions.
- [ ] **Step 2: Add resize reference tests for `1x1`, non-integer downscale, and odd aspect ratios.** Compare selected pixels to the current box resize within `1e-6` before changing algorithms.
- [ ] **Step 3: Split decode storage from target storage and account every owned byte.** Keep depth only when initialization or depth loss requests it; do not retain a second f32 RGB host copy after target upload.
- [ ] **Step 4: Add an offline target-pack subcommand with versioned header `(source fingerprint, dimensions, RGB format)`.** Load only packs matching source fingerprint and requested scale; otherwise fall back to runtime resize.
- [ ] **Step 5: Report decode, resize, upload duration and cache hit/miss/eviction counters.**
- [ ] **Step 6: Compare 500/3k TUM at the same resolution.** Require byte-budget conformance, output parity within `1e-5` for cached targets, and a lower input-stall contribution without PSNR change.
- [ ] **Step 7: Commit.** `git commit -m "perf(rustgs): budget decoded frame and target caches"`

### Task 7: Profile then fuse loss and improve raster work distribution (P2-03, P2-04)

**Files:**
- Modify: `RustGS/src/training/{engine/loss.rs,config.rs,forward/rasterize.rs}`
- Modify: `RustGS/src/training/shaders/rasterize.wgsl`
- Create: `RustGS/src/training/shaders/loss_fused.wgsl`
- Test: `RustGS/src/training/engine/loss.rs`, `RustGS/src/training/forward/rasterize.rs`

**Interfaces:**
- Produces: `LossKernelMode::{Reference, Fused}` and `RasterWorkTelemetry { empty_tiles, intersections_p50, intersections_p95, early_terminated_pixels }`.
- The fused loss returns the same scalar and `dL/dpred` contract as `combined_loss_with_kernel` for its enabled terms.

- [ ] **Step 1: Add a profiler-only run that reports L1, gradient, SSIM, raster forward, and raster backward shares.** Do not write a fused kernel until loss work exceeds 10% of median iteration time on the TUM baseline.
- [ ] **Step 2: Add reference tests for random `HWC` RGB tensors and `1x1`/small images.** Compare fused L1 plus finite-difference gradient to `combined_loss_with_kernel` with SSIM and gradient terms disabled first.
- [ ] **Step 3: Implement fused L1/robust weighting first, retaining reference SSIM and gradient loss.** Extend only after scalar and input-gradient parity are established; use a dedicated WGSL entry point and dispatch based on pixels, not splats.
- [ ] **Step 4: Add raster histogram counters without changing pixel output.** Count empty tiles, tile list length buckets, and alpha early termination. Use the data to choose one static candidate tile size behind `TrainingRasterConfig::tile_size`; do not enable adaptive tile sizing yet.
- [ ] **Step 5: Add forward/backward image and gradient parity tests across tile sizes `8x8`, `16x16`, `32x16`.** Reject any configuration whose finite pixels, color, opacity, or VJP differs beyond the stated tolerance.
- [ ] **Step 6: Benchmark the selected loss/tile experiment at TUM 500/3k.** Select only after performance improves and quality gates remain satisfied.
- [ ] **Step 7: Commit.** `git commit -m "perf(rustgs): instrument loss and raster tile work"`

### Task 8: Improve initialization coverage and background handling (P2-05, P2-06)

**Files:**
- Modify: `RustGS/src/training/{config.rs,data/init_map.rs,engine/trainer.rs,evaluation/core.rs}`
- Modify: `RustGS/src/bin/rustgs/train_command.rs`
- Test: `RustGS/src/training/data/init_map.rs`, `RustGS/src/training/evaluation/core.rs`

**Interfaces:**
- Adds `InitializationSampling::{InputOrder, Voxel { cell_size: f32 }, VisibilityStratified}` and `BackgroundMode::{Black, White, Random { seed: u64 }, DatasetMedian}`.
- `render_background(mode, frame_id, training)` must be shared by training and evaluation; random background uses a deterministic seed and cannot be selected for final evaluation.

- [ ] **Step 1: Add a spatial fixture containing clustered points, sparse boundary points, and varying track visibility.** Assert voxel sampling preserves one representative per occupied cell and visibility-stratified sampling retains each configured visibility bucket.
- [ ] **Step 2: Add background tests.** For an alpha-zero scene, black and white output exactly their requested background; `DatasetMedian` is fixed from training images; random mode is reproducible for a seed.
- [ ] **Step 3: Add initialization coverage metrics.** Report occupied voxels, projected occupied-image cells over a fixed camera sample, and initial Gaussian count before any topology work.
- [ ] **Step 4: Add the config/CLI modes with current input-order/black defaults.** Make the evaluator refuse `Random` unless an explicit diagnostic flag is supplied.
- [ ] **Step 5: Run 500-step ablations for each initialization mode and background mode, then retain only candidates with improved worst frame or sharpness at matched count.**
- [ ] **Step 6: Commit.** `git commit -m "feat(rustgs): make initialization and backgrounds configurable"`

### Task 9: Build a finite-difference projection-gradient gate (P2-07)

**Files:**
- Modify: `RustGS/src/training/{backward/{autodiff.rs,project_bwd.rs},forward/projection.rs}`
- Modify: `RustGS/src/training/shaders/{project_forward.wgsl,project_backwards.wgsl,helpers.wgsl}`
- Create: `RustGS/src/training/backward/gradient_check.rs`
- Test: `RustGS/src/training/backward/gradient_check.rs`

**Interfaces:**
- Produces `GradientCheckCase`, `GradientCheckResult { parameter, analytic, numeric, relative_error, finite }`, and `run_projection_gradient_check(case, epsilon)`.
- Acceptance thresholds: relative error `<= 2e-2` for regular cases and `<= 1e-1` within one epsilon of a clipping/discontinuity boundary; all sampled values must be finite.

- [ ] **Step 1: Define one-Gaussian scalar objectives from rendered RGB dot a fixed upstream color gradient.** Cases cover off-center principal point, `fx != fy`, near plane, small covariance, low alpha, rotated anisotropic scale, SH degree 0/3, and tile/bounding-box boundary.
- [ ] **Step 2: Implement centered finite differences for position, log scale, quaternion, opacity logit, and one SH coefficient.** Normalize quaternions after each perturbation and use scale-aware epsilon `max(1e-4, 1e-3 * abs(parameter))`.
- [ ] **Step 3: Run the checker before changing shaders and save its JSON table.** Any observed mismatch becomes a named regression case rather than an averaged score.
- [ ] **Step 4: Correct covariance-blur compensation VJP.** Differentiate both `filter_comp = sqrt(det(cov_raw)/det(cov_blurred))` and its dependence on covariance; add its contribution to mean, scale, and quaternion paths before the existing `v_cov2d` mapping.
- [ ] **Step 5: Correct SH view-direction position VJP.** Compute `dL/dviewdir` from SH basis, apply the normalized-vector Jacobian for `viewdir = normalize(mean - camera_position)`, and add it to `v_mean_c` before transforming to world position.
- [ ] **Step 6: Run all gradient cases on the active GPU adapter and CPU/reference-compatible backend where available.** Block all quality-default changes while any regular case fails.
- [ ] **Step 7: Commit.** `git commit -m "fix(rustgs): gate projection gradients with finite differences"`

### Task 10: Make robust/dynamic supervision mathematically well-behaved

**Files:**
- Modify: `RustGS/src/training/{config.rs,engine/loss.rs}`
- Modify: `RustGS/src/bin/rustgs/train_command.rs`
- Test: `RustGS/src/training/engine/loss.rs`

**Interfaces:**
- Adds `DynamicMaskGradient::{StopGradient, Coupled}` with default `StopGradient` once validated.
- `dynamic_residual_mask` returns both `weight` and a detached version for the weighted reduction; scheduler gates it with `dynamic_mask_start_epoch` and optional linear ramp.

- [ ] **Step 1: Add scalar analytic tests for residuals below low threshold, inside the transition, and above high threshold.** For stop-gradient weighting, verify `d(weighted_loss)/d(residual)` stays non-negative and never rewards increasing residual.
- [ ] **Step 2: Add a finite-difference tensor test across three RGB channels.** Compare stop-gradient and coupled variants; document any negative coupled derivative as a rejected default behavior.
- [ ] **Step 3: Implement a detach operation at mask construction, not after normalized reduction.** Keep a `Coupled` diagnostic mode only for ablation and serialize the selected mode in checkpoints.
- [ ] **Step 4: Add a warmup/ramp schedule and log effective mean weight plus masked-pixel fraction each iteration cadence.**
- [ ] **Step 5: Compare no mask, stop-gradient mask, and coupled diagnostic mode on a dynamic TUM segment.** Require no degradation of static-region PSNR, reduced dynamic outlier influence, and no global fog/ghosting in worst-frame renders.
- [ ] **Step 6: Commit.** `git commit -m "fix(rustgs): detach dynamic residual supervision mask"`

### Task 11: Evaluate Pixel-GS/AbsGS densification and contribution pruning

**Files:**
- Modify: `RustGS/src/training/{config.rs,topology/{mod.rs,bridge.rs,splat_metrics.rs},shaders/{accumulate_topology_stats.wgsl,project_backwards.wgsl}}`
- Modify: `RustGS/src/bin/rustgs/train_command.rs`
- Test: `RustGS/src/training/topology/{mod.rs,splat_metrics.rs}`

**Interfaces:**
- Adds `TopologyScoreMode::{Baseline, Abs, AbsPixel, AbsPixelDepth, Contribution}`.
- `Contribution` score combines a Gaussian's cross-view `alpha * transmittance` contribution, normalized screen-gradient magnitude, residual association, and observation count; each term is stored separately for reporting.

- [ ] **Step 1: Add topology-unit fixtures where raw gradient, pixel coverage, depth, and contribution rank different candidates.** Assert each score mode selects the expected row and that normalizing by observation count prevents a repeatedly sampled frame from dominating.
- [ ] **Step 2: Verify the current `AbsPixel` and `AbsPixelDepth` accumulators against a CPU formula.** Clamp invalid depth, use scene-radius-aware scaling, and record the per-term percentiles before candidate selection.
- [ ] **Step 3: Add raster contribution accumulation.** Accumulate only finite alpha/transmittance values and normalize by visible observations; maintain a reference flag that disables the metric without affecting existing modes.
- [ ] **Step 4: Run isolated 500/3k ablations for Baseline, Abs, AbsPixel, AbsPixelDepth, and Contribution.** Hold maximum Gaussian budget and topology seed fixed. Select no mode from a single scene.
- [ ] **Step 5: For the strongest two candidates run TUM 10k, Home 1500, and external 3k.** Compare count, intersections, worst-frame PSNR, edge sharpness, and topology event distribution.
- [ ] **Step 6: Commit.** `git commit -m "feat(rustgs): evaluate coverage and contribution topology scores"`

### Task 12: Add per-frame exposure and optional pose refinement

**Files:**
- Modify: `RustGS/src/training/{config.rs,engine/{trainer.rs,optimizer.rs},checkpoint.rs,evaluation/core.rs}`
- Modify: `RustGS/src/training/{forward/mod.rs,backward/autodiff.rs}`
- Modify: `RustGS/src/bin/rustgs/train_command.rs`
- Test: `RustGS/src/training/{engine/trainer.rs,checkpoint.rs,evaluation/core.rs}`

**Interfaces:**
- Adds per-frame learnable `ExposureParams { log_gain: [f32; 3], bias: [f32; 3] }` applied as `clamp(exp(log_gain) * rgb + bias, 0, 1)` before loss/evaluation.
- Adds optional `PoseRefinementConfig { enabled, start_epoch, translation_lr, rotation_lr, regularization }`; pose delta is an SE(3) twist with zero-initialized optimizer state and checkpoint serialization.

- [ ] **Step 1: Add exposure forward/VJP tests against finite difference for gain and bias.** Verify identity parameters leave current images byte-for-byte equivalent within float tolerance.
- [ ] **Step 2: Add pose parameter tests using one fixed Gaussian/camera.** A small translation/rotation must change projection in the known direction; regularization gradient at zero must be zero.
- [ ] **Step 3: Implement exposure first, behind `--learn-exposure`; initialize all frames to identity and add weak L2 regularization to prevent color from absorbing geometry errors.**
- [ ] **Step 4: Implement pose refinement after exposure passes.** Start only after configured warmup, compose the delta with the input pose consistently in train/eval, and reject it for datasets with missing camera metadata.
- [ ] **Step 5: Run exposure-only, pose-only, combined, and disabled ablations on lighting-varying scenes.** Inspect holdout views and worst frames; reject candidates that improve training PSNR while degrading held-out PSNR.
- [ ] **Step 6: Commit.** `git commit -m "feat(rustgs): add gated exposure and pose refinement"`

### Task 13: Implement mip-aware filtering as a separate anti-aliasing experiment

**Files:**
- Modify: `RustGS/src/training/{config.rs,forward/{mod.rs,projection.rs},backward/project_bwd.rs}`
- Modify: `RustGS/src/training/shaders/{helpers.wgsl,project_forward.wgsl,project_backwards.wgsl}`
- Test: `RustGS/src/training/{forward/projection.rs,backward/gradient_check.rs}`

**Interfaces:**
- Adds `AntiAliasingMode::{CovarianceBlur, MipSplat}`. `MipSplat` uses a scene-scale 3D smoothing floor during covariance construction plus a pixel-footprint 2D filter, both with analytic VJP.

- [ ] **Step 1: Add image-space fixtures for minified splats and oblique surfaces.** Compare alias-energy proxy, opacity conservation, and finite gradients for current blur and the new mode.
- [ ] **Step 2: Compute a per-Gaussian 3D smoothing covariance from the sampling floor and add it before camera projection.** Keep its scale tied to scene units, not raw pixel resolution.
- [ ] **Step 3: Compute the 2D footprint filter from the projected covariance and use determinant compensation to preserve integrated opacity.** Derive and add both 3D and 2D filter VJPs to the gradient checker from Task 9.
- [ ] **Step 4: Run a resolution sweep at 0.25, 0.5, and 1.0 render scale.** Require lower alias proxy at 0.25 without a sharpness-ratio collapse or worse holdout PSNR.
- [ ] **Step 5: Commit.** `git commit -m "feat(rustgs): add mip-aware splat filtering experiment"`

### Task 14: Add deterministic multi-resolution and view-balanced sampling

**Files:**
- Modify: `RustGS/src/training/{config.rs,engine/{runtime.rs,trainer.rs},data/{frame_loader.rs,frame_targets.rs},checkpoint.rs}`
- Modify: `RustGS/src/bin/rustgs/train_command.rs`
- Test: `RustGS/src/training/{engine/runtime.rs,checkpoint.rs,data/frame_targets.rs}`

**Interfaces:**
- Adds `TrainingResolutionStage { until_iteration, render_scale }` and `FrameSamplingMode::{DatasetOrder, EpochShuffle, PoseStratified, ErrorAware}`.
- `ordered_frame_indices(frame_count, epoch, seed, mode)` must be a pure deterministic function and checkpoint records both current epoch and sampling-mode identity.

- [ ] **Step 1: Add ordering tests.** Same seed/epoch/mode produces identical order, different epochs differ for shuffle, and pose-stratified output contains every non-empty pose bucket before repeats.
- [ ] **Step 2: Add resolution-stage tests.** Stage selection changes only at configured iterations, target dimensions and camera intrinsics scale together, and checkpoint resume chooses the same stage/order.
- [ ] **Step 3: Implement `1/4 -> 1/2 -> 1.0` stages as explicit opt-in config.** Invalidate target-cache keys on size change and scale principal point/focal consistently with image dimensions.
- [ ] **Step 4: Implement epoch shuffle, then pose stratification.** Delay error-aware sampling until per-frame held-out/residual metrics are available; cap its probability so hard frames do not monopolize topology statistics.
- [ ] **Step 5: Run matched compute-budget experiments.** Compare equal wall-clock and equal iteration budgets separately; report both to avoid claiming a scale reduction as algorithmic convergence.
- [ ] **Step 6: Commit.** `git commit -m "feat(rustgs): add deterministic scales and view sampling"`

### Task 15: Decide whether a 2DGS surface branch is warranted

**Files:**
- Create: `docs/RustGS-2DGS-Feasibility-2026-09-17.md`
- Create: `RustGS/examples/surface_splat_probe.rs`
- Test: `RustGS/examples/surface_splat_probe.rs`

**Interfaces:**
- The probe reads one `HostSplats` scene and produces depth, normal, distortion, and RGB images without changing the primary 3DGS trainer.
- It reports `SurfaceProbeMetrics { depth_mae, normal_consistency, depth_distortion, psnr_db }` for a fixed evaluation frame set.

- [ ] **Step 1: Implement a non-training oriented-disk renderer using local tangent axes from a Gaussian rotation and two in-plane scales.** Keep it in the example crate so no existing checkpoint or trainer contract changes.
- [ ] **Step 2: Add a synthetic plane/sphere fixture.** Assert finite depth, coherent normals, and lower depth distortion for the oriented disk than an isotropic-volume reference at the same projected footprint.
- [ ] **Step 3: Run the probe on one indoor surface-rich scene and record current 3DGS metrics.**
- [ ] **Step 4: Write a decision note with the exact trigger for a separate 2DGS implementation: target product needs mesh-like surfaces, probe improves depth/normal metrics materially, and quality does not regress.** If the trigger is not met, stop here and retain 3DGS work above as the active path.
- [ ] **Step 5: Commit.** `git commit -m "docs(rustgs): assess a surface-splat training branch"`

### Task 16: Run the full selection gate and update the TODO evidence

**Files:**
- Modify: `docs/RustGS-TODO-训练效果与效率优化-2026-09-17.md`
- Create: `docs/RustGS-Optimization-Results-2026-09-17.md`
- Modify: `docs/index.md`
- Verify: `RustGS/src/training/**`

**Interfaces:**
- Consumes: two `OptimizationReport` JSON files per experiment and the fixed dataset manifest.
- Produces: a table of baseline/candidate deltas and an explicit decision of `accepted`, `rejected`, or `needs-more-data` for each task.

- [ ] **Step 1: Run code checks.**

```sh
cargo fmt --package rustgs --check
cargo test -p rustgs --lib --no-default-features --no-fail-fast
cargo test -p rustgs --lib --features gpu-wgpu --no-fail-fast
git diff --check
```

- [ ] **Step 2: Run baseline and each isolated candidate at TUM 500, 3k, and 10k.** Promote only candidates passing their local parity tests and all global report fields.
- [ ] **Step 3: Run promoted candidates at TUM 30k, Home 1500, and one external scene.** Save worst-frame image paths and metrics for every run.
- [ ] **Step 4: Compare reports.** Reject a candidate for a worst-frame decline over `0.2 dB`, non-finite output, overflow, topology-origin mismatch, incompatible run fingerprint, visible fog/ghosting, or a performance gain smaller than measurement noise across repeated runs.
- [ ] **Step 5: Update the dated TODO.** Check an item only after its implementation and all stated gates pass; for rejected items retain the command, report paths, measured regression, and follow-up hypothesis.
- [ ] **Step 6: Commit the evidence.** `git commit -m "docs(rustgs): record training optimization gates"`

## Method References and Experiment Order

Run Tasks 9 and 10 before changing topology defaults. Then run Task 11 in isolated score-mode ablations. Tasks 12 through 14 are independent candidate families and must not be combined until each passes the standard report gate. Task 13 follows the Mip-Splatting idea of 3D smoothing plus 2D Mip filtering, rather than treating the current covariance blur as equivalent. Task 15 is a product-direction decision, not an in-place replacement for the current renderer.

- Pixel-GS: <https://arxiv.org/abs/2403.15530>
- AbsGS: <https://ty424.github.io/AbsGS.github.io/>
- Mip-Splatting: <https://niujinshuchong.github.io/mip-splatting/>
- Robust Gaussian Splatting: <https://arxiv.org/abs/2404.04211>
- 3DGS-MCMC: <https://arxiv.org/abs/2404.09591>
- 2D Gaussian Splatting: <https://surfsplatting.github.io/>

## Plan Self-Review

- Spec coverage: P0-01/02 are Task 1; P1-01/02/03 are Task 3; P1-04 is Task 4; P1-05/06 are Task 2; P2-01/02 are Task 6; P2-03/04 are Task 7; P2-05/06 are Task 8; P2-07 is Task 9. Method improvements are Tasks 10 through 15; universal evidence and documentation are Task 16.
- Placeholder scan: this document defines every new interface, data ownership boundary, test fixture, command, and acceptance decision required by later tasks. It contains no deferred implementation marker.
- Type consistency: `OptimizationReport`, `ForwardCapacityTelemetry`, `TopologyCandidateRecord`, `GradientCheckResult`, `TrainingResolutionStage`, and all enum names are introduced in the task that first produces them and used consistently thereafter.
