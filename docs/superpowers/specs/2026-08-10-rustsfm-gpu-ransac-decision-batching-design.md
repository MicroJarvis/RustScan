# RustSFM GPU RANSAC Decision-Batching Design

## Goal

Reduce wgpu score-summary dispatch and readback waits by scoring up to 512 RANSAC trials per GPU
batch while preserving the exact model-selection semantics and raw SQLite output of the established
64-trial GPU RANSAC implementation.

The result is acceptable only when fixed-seed runs produce bit-identical selected-pair `matches`
and `two_view_geometries` rows. Equal aggregate counts, close floating-point values, or equivalent
poses are insufficient.

## Background

The failed 64-to-512 constant experiment coupled two independent concerns:

- the number of trials whose candidate models share one GPU score dispatch/readback; and
- the RANSAC boundary at which a newly reduced dynamic trial limit becomes effective.

On the first flowers2 pair, both variants read the same source snapshot and produced an identical
1,419-match `matches` row. The 512 variant nevertheless scored about five times as many Essential
and Fundamental candidates before its first dynamic-limit check and stored different E, F, inlier
data, qvec, and tvec. The optimization therefore needs separate physical and logical boundaries.

## Scope

This change affects only the generic-wgpu Essential, Fundamental, and Homography RANSAC loops in
`RustSFM/src/geometry/two_view.rs`, a read-only scorer-capacity query in
`RustSFM/src/gpu/scorer.rs`, and their focused tests and diagnostics.

It does not change:

- CPU RANSAC, PnP, thresholds, trial formulas, seeds, sampler algorithms, local optimization,
  fallback estimators, final refinement, pose recovery, or model classification;
- descriptor matching, image-pair generation or ordering, database transactions, progress events,
  pause/cancel behavior, or RustViewer orchestration;
- image, keyframe, or pair limits; or
- wgpu backend selection. RustSFM continues to request generic wgpu without forcing Metal or
  Vulkan.

## Approaches Considered

### 1. Decouple score batches from decision batches (selected)

Generate and score as many as 512 trials in one physical GPU batch, retain each candidate's source
trial, and consume returned summaries through the same 64-trial logical windows used by the
baseline. A dynamic limit updated in one logical window becomes effective before any candidate in
the next logical window can update the winner.

This keeps the expected dispatch/readback reduction while preserving the baseline decision
frontier. It requires careful trial metadata for five-point and seven-point solvers, which can emit
multiple models per trial.

### 2. Keep the first batch at 64 and grow later batches adaptively

This preserves the first decision checkpoint but does not solve the general problem: a later large
batch can still cross a newly reduced dynamic limit. Adding internal 64-trial checkpoints to those
larger batches reduces this approach to option 1 with extra policy complexity.

### 3. Restore 64 and optimize unrelated readbacks

This is the lowest-risk fallback and remains preferable to accepting changed reconstruction
results. It gives up the measured score-summary batching opportunity and does not address this
experiment's intended optimization.

Changing the parity contract to numerical tolerance is not an option.

## Constants And Policy

Replace the single chunk concept with two internal constants:

```rust
GPU_RANSAC_SCORE_BATCH_TRIALS = 512
GPU_RANSAC_DECISION_BATCH_TRIALS = 64
```

The score boundary limits candidate generation and one `score_two_view_models_profiled` call. The
decision boundary controls when `dynamic_max_trials` is re-evaluated. The existing inclusive
zero-based stop convention remains unchanged: the effective end is
`max(dynamic_max_trials, min_num_trials) + 1`, clamped to `max_num_trials`.

Internal tests may construct alternate policies such as `64/64` and `512/64`; production code uses
only the constants above. No CLI or project configuration is added.

## Candidate Representation

Physical generation first retains trial groups containing the absolute trial, sampled indices, and
all models emitted by that minimal-solver call. A trial that emits no model is retained as an empty
group so it still advances the logical cursor. Flattening a trial group produces GPU candidates
that each carry:

- zero-based absolute trial index;
- model index within that trial;
- number of models emitted by that trial; and
- the existing row-major `Matrix3<f64>` model.

The candidate vector remains in sampler/minimal-solver emission order. Essential five-point and
Fundamental seven-point models from one trial stay contiguous and are never split across a logical
decision boundary. Homography has at most one candidate per trial. The f32 GPU model vector and
returned summaries use the same candidate order. The physical batch separately records its
generated trial range so logical trial cursors still advance correctly when a minimal solver emits
no model for a sampled trial.

This metadata also supplies an internal winner trace. Trace logging is disabled by default and
records physical/decision batch boundaries, discarded suffix ranges, and accepted best updates. A
best-update record contains only existing CPU/GPU values: model family, trial, model index, raw
support, locally optimized support, and updated dynamic limit. It uses the existing
`log`/`env_logger` path at trace level rather than adding a CLI or project option, and it must not
issue another GPU dispatch, mask request, or readback.

## Execution Flow

For each Essential, Fundamental, or Homography stage:

1. Compute the effective end from the dynamic limit known at the start of the physical batch.
2. Generate ordered trials up to the smaller of the 512-trial score boundary and that known
   effective end. Preserve trial metadata for every emitted model.
3. Convert and score all eligible generated models with the existing prepared wgpu session and
   read back one summary array. If a device-capacity limit requires subdivision, split the retained
   groups into the largest possible ordered score slices without splitting a trial group or
   resampling.
4. Consume candidates in 64-trial logical windows. Within a window, preserve the current strict
   inlier/residual tie-break, selected-candidate mask readback, CPU local optimization, best update,
   and dynamic-limit update order.
5. After the complete logical window, recompute its next effective end. If the next trial is no
   longer eligible, stop and discard any already-scored suffix without requesting masks or running
   local optimization for it.
6. Otherwise consume the next logical window already present in the physical batch. When the
   scored candidates are exhausted, generate the next physical batch.
7. Preserve the existing sampler-exhaustion, no-model fallback, final refinement, and error paths.

The physical prefetch may perform extra candidate generation and score work, but extra candidates
are not eligible to change `best` once the 64-trial decision frontier closes.

Candidate f64-to-f32 conversion is speculative for trials beyond the current logical window. The
physical batch records conversion results by nominal 64-trial window. A nominal window containing
an invalid model is not submitted speculatively, while all complete valid windows before it may
still be coalesced into one score call.

When the frontier reaches the start of a deferred window, first compute the actual decision end
from the updated dynamic limit. If the invalid trial is before that end, return the error before
submitting or consuming any candidate from the window. If the invalid trial is at or beyond that
end, the invalid suffix is outside the baseline's actual partial window: score and consume only the
complete trial groups before the actual end, then stop at the frontier. This matches the baseline's
convert-before-score behavior even when a dynamic limit truncates a nominal 64-trial window.

Device capacity is not discovered by submitting an oversized batch or parsing an `anyhow` error
string. `WgpuModelScoringSession` exposes an internal read-only maximum model count derived from
the device's compute-workgroup, model-buffer, and summary-buffer limits. Retained trial groups are
partitioned against that count before submission, preferably at a 64-trial decision boundary and
always at a complete trial boundary. If one trial group alone exceeds capacity, the operation fails
only when its logical window becomes eligible. A submitted wgpu operation that returns a
device/internal error still fails immediately; it is not broadly retried as a smaller batch.

## Shared Random Stream

`RUSTSFM_COLMAP_SHARED_RANSAC_STREAM` makes sampler advancement externally observable by later
model families and image pairs. Generating a discarded suffix would change that global stream even
if its summaries never affected the current winner.

When the shared stream is active, the effective score batch is therefore forced to the 64-trial
decision batch. This preserves the current number and order of random draws. Owned deterministic
samplers use 512/64 because unused prefetch does not escape the current RANSAC stage.

If later work wants large shared-stream batches, it requires a separately designed clonable or
transactional sampler and is outside this change.

## Error Handling And Invariants

- Candidate, f32 model, and summary lengths must match for every submitted score slice; a mismatch
  remains an immediate scorer-contract error because it cannot be attributed safely to a suffix.
- A logical boundary must never split models emitted by one trial.
- A discarded scored suffix must never trigger an inlier-mask readback, CPU local optimization, or
  best-model update.
- A discarded suffix must not surface deferred model-conversion, candidate-summary validation, or
  sampler-exhaustion conditions that the 64-trial frontier would not reach.
- Physical score slices must be capped against the scorer's model-count, storage-buffer, and
  compute-workgroup limits. Capacity subdivision uses the same retained trial groups before
  submission rather than resampling or changing candidate order.
- GPU errors retain their current model-family context and fail the operation; they are not treated
  as a CPU fallback when GPU was explicitly requested.
- Sampler exhaustion processes the complete generated prefix and then terminates as it does today.
- Trace logging is opt-in, bounded to batch decisions and accepted best updates, and has no effect
  on results or GPU counters.

## Testing Strategy

Implementation follows RED/GREEN TDD.

### Pure boundary and ordering tests

- Verify independent 512 score and 64 decision ends, including min-trial, max-trial, overflow, and
  already-finished cases.
- Verify multi-model trials remain contiguous and candidate ordering is stable.
- Verify every model from a multi-model trial on the last eligible trial is consumed before the
  dynamic frontier is recomputed, even when its first model lowers the trial limit.
- Simulate a dynamic limit reduced inside the first logical window and prove trials beyond the next
  eligible boundary cannot be consumed even when their summaries were prefetched.
- Verify a discarded suffix performs zero mask/refinement/best-update actions.
- Verify an eligible window containing a non-finite model receives zero score submissions and
  returns the conversion error before consumption; verify a dynamic partial window ending before
  that model scores only its eligible prefix and ignores the suffix. An invalid summary remains a
  post-score ordered error: it is ignored in a discarded suffix and returned when consumed.
- Verify a physical-batch capacity subdivision reuses the same retained trials through ordered
  score slices, never splits one trial's models, and never resamples.
- Verify a single over-capacity trial group fails only if its 64-trial logical window is reached.
- Verify shared-stream mode selects an effective 64/64 policy.

### Geometry regression tests

- Parameterize internal GPU RANSAC helpers so tests can compare reference `64/64` with candidate
  `512/64` for deterministic Essential, Fundamental, and Homography fixtures.
- Require exact final model bits, support mask, inlier count, residual bits, RANSAC-success flag,
  and candidate order. Mask-call counts must match the 64/64 reference; on fixtures large enough to
  span multiple decision windows, score calls and their associated summary readbacks must decrease.
  Models scored may increase only because an already-scored suffix is discarded at a later logical
  boundary.
- Run each candidate twice to confirm determinism.
- If no compatible adapter is available, retain the repository's explicit skip behavior and rely on
  the required external flowers2 gate for real-GPU proof.

### Database and performance gates

1. Re-run the retained single-pair flowers2 case. Require the candidate fingerprint to equal the
   chunk-64 value
   `c8e440cd266b3609f75c90e5c90260f41af1b765a9e709e0202cf605ec7eb5f5`, including the 1,383-row
   geometry and exact F/E/H/qvec/tvec bytes.
2. Run 96 pairs for three repetitions with generic wgpu and fixed seed 0. Require every run to
   equal the chunk-64 fingerprint
   `5e05ca629b63c98ae63c95ce0f37fe49a43eb870760e598352c8f8ef3d84e8ed`, with
   `96 matched / 96 verified / 62,409 matches`.
3. Rerun a 64/64 reference adjacent to the candidate on the same adapter and source snapshot.
   Require fewer score-summary calls/readbacks and no bounded mean-runtime regression against that
   fresh reference. The pinned historical JSON remains a fingerprint/counter provenance check, not
   a runtime comparator. Report the measured speedup; do not hide a small or negative result behind
   timing noise.
4. Only after the bounded parity gate passes, run the sequential 2,890-pair comparison and require
   the full raw-output fingerprints and aggregate counts to match.

Any parity mismatch stops the experiment. Do not run the full benchmark, merge the branch, or
reinterpret a different model as equivalent.

## Integration And Cleanup

The work remains on `codex/ransac-chunk-512-experiment` in the existing isolated worktree. After
implementation, focused and library verification, real-GPU parity gates, and independent
specification/code-quality review all pass, synchronize with current `main`, rerun affected checks,
and merge locally. Only then remove the worktree and branch as previously requested.

If strict parity or the required performance behavior fails, preserve the evidence, do not merge,
and ask before deleting the unmerged experiment worktree or branch.
