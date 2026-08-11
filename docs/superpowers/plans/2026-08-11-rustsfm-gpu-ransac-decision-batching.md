# RustSFM GPU RANSAC Decision-Batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score up to 512 owned-sampler RANSAC trials per generic-wgpu readback while preserving the exact 64-trial decision frontier and raw SQLite output.

**Architecture:** A private batch policy separates physical score width from logical decision width. A shared generic runner retains trial groups, pre-scores ordered candidates, and advances a 64-trial cursor; Essential, Fundamental, and Homography provide only their sample solver and local-refinement closures. The scorer exposes a read-only model capacity so subdivision happens before GPU submission.

**Tech Stack:** Rust 2021, nalgebra, anyhow, wgpu 29, existing RustSFM model scorer/timing APIs, rusqlite diagnostic benchmarks, cargo test.

---

## File Map

- Modify: RustSFM/src/geometry/two_view.rs
  Own the score/decision policy, trial metadata, generic scheduling runner, E/F/H adapters, trace logging, and unit/GPU regression tests.
- Modify: RustSFM/src/gpu/scorer.rs
  Expose the maximum safe model count for one score submission and test its device-limit arithmetic.
- Modify: docs/superpowers/plans/2026-08-10-rustsfm-ransac-chunk-512-experiment.md
  Record RED/GREEN evidence, fingerprints, timings, and final gate outcome.

### Task 1: Separate Score And Decision Boundaries

**Files:**
- Modify: RustSFM/src/geometry/two_view.rs:2198 and the tests near 4782

- [ ] **Step 1: Write the failing policy and boundary tests**

Add these tests beside the existing GPU chunk-end test:

~~~rust
#[cfg(feature = "gpu-wgpu")]
#[test]
fn gpu_ransac_policy_decouples_score_and_decision_batches() {
    assert_eq!(
        gpu_ransac_batch_policy(false),
        GpuRansacBatchPolicy {
            score_trials: 512,
            decision_trials: 64,
        }
    );
}

#[cfg(feature = "gpu-wgpu")]
#[test]
fn gpu_ransac_policy_preserves_shared_stream_draws() {
    assert_eq!(
        gpu_ransac_batch_policy(true),
        GpuRansacBatchPolicy {
            score_trials: 64,
            decision_trials: 64,
        }
    );
}

#[cfg(feature = "gpu-wgpu")]
#[test]
fn gpu_ransac_batch_end_applies_dynamic_limits_at_selected_boundary() {
    assert_eq!(gpu_ransac_batch_end(0, 10_000, 10_000, 100, 512), 512);
    assert_eq!(gpu_ransac_batch_end(64, 10_000, 24, 100, 64), 101);
    assert_eq!(gpu_ransac_batch_end(96, 10_000, 24, 100, 64), 101);
    assert_eq!(gpu_ransac_batch_end(101, 10_000, 24, 100, 64), 101);
    assert_eq!(gpu_ransac_batch_end(9_980, 10_000, usize::MAX, 100, 512), 10_000);
}
~~~

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

~~~bash
cargo test -p rustsfm --features gpu-wgpu gpu_ransac_policy_ -- --nocapture
cargo test -p rustsfm --features gpu-wgpu gpu_ransac_batch_end_ -- --nocapture
~~~

Expected: compile failure because GpuRansacBatchPolicy, gpu_ransac_batch_policy, and gpu_ransac_batch_end do not exist.

- [ ] **Step 3: Implement the minimal policy and boundary helper**

Replace GPU_RANSAC_CHUNK_TRIALS and gpu_ransac_chunk_end with:

~~~rust
#[cfg(feature = "gpu-wgpu")]
const GPU_RANSAC_SCORE_BATCH_TRIALS: usize = 512;
#[cfg(feature = "gpu-wgpu")]
const GPU_RANSAC_DECISION_BATCH_TRIALS: usize = 64;

#[cfg(feature = "gpu-wgpu")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GpuRansacBatchPolicy {
    score_trials: usize,
    decision_trials: usize,
}

#[cfg(feature = "gpu-wgpu")]
fn gpu_ransac_batch_policy(shared_stream: bool) -> GpuRansacBatchPolicy {
    let score_trials = if shared_stream {
        GPU_RANSAC_DECISION_BATCH_TRIALS
    } else {
        GPU_RANSAC_SCORE_BATCH_TRIALS
    };
    GpuRansacBatchPolicy {
        score_trials,
        decision_trials: GPU_RANSAC_DECISION_BATCH_TRIALS,
    }
}

#[cfg(feature = "gpu-wgpu")]
fn gpu_ransac_batch_end(
    iteration: usize,
    max_num_trials: usize,
    dynamic_max_trials: usize,
    min_num_trials: usize,
    batch_trials: usize,
) -> usize {
    let effective_end = dynamic_max_trials
        .max(min_num_trials)
        .saturating_add(1)
        .min(max_num_trials);
    iteration
        .saturating_add(batch_trials.max(1))
        .min(effective_end)
}
~~~

Temporarily update the three existing loop call sites to pass GPU_RANSAC_SCORE_BATCH_TRIALS. Task 4 through Task 6 replace those loops with the shared runner.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the two commands from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 1**

~~~bash
git add RustSFM/src/geometry/two_view.rs
git commit -m "refactor(rustsfm): separate gpu ransac batch boundaries"
~~~

### Task 2: Add Trial Metadata And Capacity-Aware Score Slices

**Files:**
- Modify: RustSFM/src/geometry/two_view.rs near the batch helpers and tests
- Modify: RustSFM/src/gpu/scorer.rs in WgpuModelScoringSession and its tests

- [ ] **Step 1: Write failing trial-order and slice tests**

Use diagonal matrices whose first entry identifies the model:

~~~rust
#[cfg(feature = "gpu-wgpu")]
#[test]
fn gpu_ransac_candidates_keep_trial_and_model_ordinals() {
    let groups = vec![
        GpuRansacTrialGroup::new(8, vec![1, 2], vec![Matrix3::identity(), Matrix3::repeat(2.0)]),
        GpuRansacTrialGroup::new(9, vec![3, 4], Vec::new()),
        GpuRansacTrialGroup::new(10, vec![5, 6], vec![Matrix3::repeat(3.0)]),
    ];
    let candidates = gpu_ransac_candidates(&groups);
    let ordinals = candidates
        .iter()
        .map(|candidate| (candidate.trial, candidate.model_index, candidate.models_in_trial))
        .collect::<Vec<_>>();
    assert_eq!(ordinals, vec![(8, 0, 2), (8, 1, 2), (10, 0, 1)]);
    assert_eq!(groups[1].trial, 9);
}

#[cfg(feature = "gpu-wgpu")]
#[test]
fn gpu_ransac_score_slices_never_split_one_trial() {
    let model_counts = vec![(0, 3), (1, 2), (2, 4), (3, 1)];
    assert_eq!(gpu_ransac_score_slices(&model_counts, 5).unwrap(), vec![0..5, 5..10]);
    let error = gpu_ransac_score_slices(&[(7, 6)], 5).unwrap_err();
    assert!(error.to_string().contains("trial 7"));
}
~~~

In scorer.rs add a pure capacity test:

~~~rust
#[test]
fn two_view_model_capacity_uses_the_tightest_device_limit() {
    assert_eq!(two_view_model_capacity(100, 36 * 80, 8 * 90), 80);
    assert_eq!(two_view_model_capacity(70, u64::MAX, u64::MAX), 70);
    assert_eq!(two_view_model_capacity(100, 35, u64::MAX), 0);
}
~~~

- [ ] **Step 2: Verify RED**

Run:

~~~bash
cargo test -p rustsfm --features gpu-wgpu gpu_ransac_candidates_ -- --nocapture
cargo test -p rustsfm --features gpu-wgpu gpu_ransac_score_slices_ -- --nocapture
cargo test -p rustsfm --features gpu-wgpu two_view_model_capacity_ -- --nocapture
~~~

Expected: compile failures for the new structs and functions.

- [ ] **Step 3: Implement trial metadata and ordered slice planning**

Add private structures and helpers in two_view.rs:

~~~rust
#[cfg(feature = "gpu-wgpu")]
#[derive(Debug)]
struct GpuRansacTrialGroup {
    trial: usize,
    sample: Vec<usize>,
    models: Vec<Matrix3<f64>>,
}

#[cfg(feature = "gpu-wgpu")]
impl GpuRansacTrialGroup {
    fn new(trial: usize, sample: Vec<usize>, models: Vec<Matrix3<f64>>) -> Self {
        Self { trial, sample, models }
    }
}

#[cfg(feature = "gpu-wgpu")]
#[derive(Debug)]
struct GpuRansacCandidate<'a> {
    trial: usize,
    model_index: usize,
    models_in_trial: usize,
    sample: &'a [usize],
    model: &'a Matrix3<f64>,
}

#[cfg(feature = "gpu-wgpu")]
fn gpu_ransac_candidates(groups: &[GpuRansacTrialGroup]) -> Vec<GpuRansacCandidate<'_>> {
    groups
        .iter()
        .flat_map(|group| {
            let count = group.models.len();
            group.models.iter().enumerate().map(move |(model_index, model)| {
                GpuRansacCandidate {
                    trial: group.trial,
                    model_index,
                    models_in_trial: count,
                    sample: &group.sample,
                    model,
                }
            })
        })
        .collect()
}

#[cfg(feature = "gpu-wgpu")]
fn gpu_ransac_score_slices(
    model_counts: &[(usize, usize)],
    capacity: usize,
) -> anyhow::Result<Vec<std::ops::Range<usize>>> {
    let mut slices = Vec::new();
    let mut slice_start = 0usize;
    let mut slice_len = 0usize;
    for &(trial, count) in model_counts {
        if count > capacity {
            anyhow::bail!("GPU RANSAC trial {trial} emits {count} models, capacity is {capacity}");
        }
        if count > 0 && slice_len > 0 && slice_len.saturating_add(count) > capacity {
            slices.push(slice_start..slice_start + slice_len);
            slice_start += slice_len;
            slice_len = 0;
        }
        slice_len = slice_len.saturating_add(count);
    }
    if slice_len > 0 {
        slices.push(slice_start..slice_start + slice_len);
    }
    Ok(slices)
}
~~~

- [ ] **Step 4: Implement scorer capacity arithmetic**

Add in scorer.rs:

~~~rust
fn two_view_model_capacity(
    max_workgroups: u32,
    model_buffer_bytes: u64,
    summary_buffer_bytes: u64,
) -> usize {
    let dispatch = max_workgroups as usize;
    let models = model_buffer_bytes / std::mem::size_of::<[f32; 9]>() as u64;
    let summaries = summary_buffer_bytes / std::mem::size_of::<GpuModelSupport>() as u64;
    dispatch
        .min(usize::try_from(models).unwrap_or(usize::MAX))
        .min(usize::try_from(summaries).unwrap_or(usize::MAX))
}
~~~

Add this session method, deriving both byte limits from min(max_buffer_size, max_storage_buffer_binding_size):

~~~rust
pub(crate) fn max_two_view_models_per_score(&self) -> usize {
    let limits = self.scorer.context.device().limits();
    let storage_bytes = limits
        .max_buffer_size
        .min(u64::from(limits.max_storage_buffer_binding_size));
    two_view_model_capacity(
        limits.max_compute_workgroups_per_dimension,
        storage_bytes,
        storage_bytes,
    )
}
~~~

- [ ] **Step 5: Verify GREEN and unchanged scorer behavior**

Run the three commands from Step 2, then:

~~~bash
cargo test -p rustsfm --features gpu-wgpu wgpu_model_scorer_ -- --nocapture
~~~

Expected: all selected tests pass; adapter-dependent tests may use their existing explicit skip path.

- [ ] **Step 6: Commit Task 2**

~~~bash
git add RustSFM/src/geometry/two_view.rs RustSFM/src/gpu/scorer.rs
git commit -m "feat(rustsfm): retain gpu ransac trial batches"
~~~

### Task 3: Build The Shared Decision-Frontier Runner With A Scripted Scorer

**Files:**
- Modify: RustSFM/src/geometry/two_view.rs near the GPU helpers and tests

- [ ] **Step 1: Define a private scorer seam and write the scripted RED test**

Introduce this private scorer seam and implement it for WgpuModelScoringSession without changing the public API:

~~~rust
#[cfg(feature = "gpu-wgpu")]
trait GpuRansacScoring {
    fn max_models_per_score(&self) -> usize;
    fn score_models_profiled(
        &self,
        models: &[[f32; 9]],
        threshold: f32,
        kind: TwoViewModelKind,
    ) -> anyhow::Result<(Vec<GpuModelSupport>, crate::gpu::WgpuModelScorerTiming)>;
    fn inlier_mask_profiled(
        &self,
        model: &[f32; 9],
        threshold: f32,
        kind: TwoViewModelKind,
    ) -> anyhow::Result<(Vec<bool>, crate::gpu::WgpuModelScorerTiming)>;
}

#[cfg(feature = "gpu-wgpu")]
impl GpuRansacScoring for WgpuModelScoringSession<'_> {
    fn max_models_per_score(&self) -> usize {
        self.max_two_view_models_per_score()
    }

    fn score_models_profiled(
        &self,
        models: &[[f32; 9]],
        threshold: f32,
        kind: TwoViewModelKind,
    ) -> anyhow::Result<(Vec<GpuModelSupport>, crate::gpu::WgpuModelScorerTiming)> {
        self.score_two_view_models_profiled(models, threshold, kind)
    }

    fn inlier_mask_profiled(
        &self,
        model: &[f32; 9],
        threshold: f32,
        kind: TwoViewModelKind,
    ) -> anyhow::Result<(Vec<bool>, crate::gpu::WgpuModelScorerTiming)> {
        WgpuModelScoringSession::inlier_mask_profiled(self, model, threshold, kind)
    }
}
~~~

Create a ScriptedGpuRansacScorer in the test module with a Cell-based score/mask counter and a support_by_trial map. Its score method reads models[i][0] as an integer trial, returns the mapped support or a default two-inlier support, and its mask method returns exactly support.inliers true values followed by false values. Set max_models_per_score from a constructor argument so the same fake exercises capacity subdivision.

The primary test must run the same scripted sampler/generator with policies 64/64 and 512/64:

~~~rust
#[cfg(feature = "gpu-wgpu")]
#[test]
fn gpu_ransac_frontier_is_independent_of_score_batch_size() -> anyhow::Result<()> {
    let reference = run_scripted_gpu_ransac(GpuRansacBatchPolicy {
        score_trials: 64,
        decision_trials: 64,
    })?;
    let candidate = run_scripted_gpu_ransac(GpuRansacBatchPolicy {
        score_trials: 512,
        decision_trials: 64,
    })?;
    assert_eq!(candidate.best_trial, reference.best_trial);
    assert_eq!(candidate.best_trial, 100);
    assert_eq!(candidate.consumed_trials, reference.consumed_trials);
    assert!(!candidate.consumed_trials.contains(&101));
    assert_eq!(candidate.mask_calls, reference.mask_calls);
    assert_eq!(candidate.refinement_calls, reference.refinement_calls);
    assert!(candidate.score_calls < reference.score_calls);
    Ok(())
}
~~~

Also add focused RED tests named:

- gpu_ransac_empty_trials_advance_the_decision_cursor
- gpu_ransac_sampler_exhaustion_matches_the_legacy_prefix
- gpu_ransac_multi_model_boundary_trial_is_atomic
- gpu_ransac_discarded_suffix_requests_no_mask_or_refinement
- gpu_ransac_deferred_conversion_error_is_ignored_after_frontier_closes
- gpu_ransac_deferred_conversion_error_fails_when_frontier_reaches_it
- gpu_ransac_invalid_summary_is_ignored_after_frontier_closes
- gpu_ransac_invalid_summary_fails_when_frontier_reaches_it
- gpu_ransac_summary_count_mismatch_fails_immediately
- gpu_ransac_capacity_slices_do_not_resample_or_reorder_trials

Each test uses counters owned by the scripted sampler/scorer/refiner and asserts exact trial/model order, not only final support.

The shared-stream draw test uses an Rc<Cell<usize>> sampler counter. Run once with gpu_ransac_batch_policy(true), force a dynamic stop at trial 100, and assert the sampler was called exactly 101 times. Run the 64/64 reference from a fresh zero counter and assert the two next-draw indices are both 101. This tests the runner rather than only the policy value.

Name that test gpu_ransac_shared_stream_does_not_consume_speculative_samples. The remaining scripted tests use these exact fixtures and assertions:

- Empty trials: return no models for trials 0 through 62 and one model for trial 63; assert the consumed model log is [63] while the sampler log is 0 through 63.
- Sampler exhaustion: return a full sample for trials 0 through 36 and a short sample at trial 37; assert candidates from 0 through 36 are scored and consumed in order, no trial 37 model exists, and no later sample occurs.
- Multi-model boundary: make trial 100 emit two models. Let model 0 lower the dynamic limit below 100 and model 1 have strictly better support; assert both (100, 0) and (100, 1) are consumed in order, model 1 wins, and trial 101 is not consumed. This proves the frontier advances by trials only after the complete boundary group.
- Discarded suffix: use the primary 63/100/101 support script; assert trial 101 appears in the 512 score log but not in the consumed, mask, refinement, or best-update logs.
- Ignored conversion error: emit Matrix3::repeat(f64::MAX) at trial 101 with min_num_trials 100. Assert the nominal 64..128 window was not speculatively submitted, the actual 64..101 prefix is then scored on demand, the run succeeds with winner 100, and trial 101 appears in no score/mask/refinement/best-update log.
- Reached conversion error: use the same invalid trial 101 with min_num_trials 127; assert the run returns an error containing trial 101 with zero score calls/models and zero consumption for the entire 64..128 logical window.
- Ignored invalid summary: return residual_sum = f32::NAN for trial 101 with min_num_trials 100; assert winner 100 and no trial 101 mask, refinement, or best update.
- Reached invalid summary: use the same NaN summary with min_num_trials 127; assert candidates before trial 101 retain their baseline actions and the runner then returns the existing non-finite residual error at trial 101.
- Summary count mismatch: make the first score call return one fewer summary than models; assert an immediate family-context error and zero mask/refinement calls regardless of the current dynamic frontier.
- Capacity slicing: emit three models per trial for trials 0 through 3 with capacity 5; assert score-call trial groups are [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], consumed order is unchanged, and sampler calls remain exactly [0, 1, 2, 3].

- [ ] **Step 2: Verify RED**

Run:

~~~bash
cargo test -p rustsfm --features gpu-wgpu gpu_ransac_frontier_ -- --nocapture
cargo test -p rustsfm --features gpu-wgpu gpu_ransac_empty_trials_ -- --nocapture
cargo test -p rustsfm --features gpu-wgpu gpu_ransac_sampler_exhaustion_ -- --nocapture
cargo test -p rustsfm --features gpu-wgpu gpu_ransac_multi_model_boundary_ -- --nocapture
cargo test -p rustsfm --features gpu-wgpu gpu_ransac_discarded_suffix_ -- --nocapture
cargo test -p rustsfm --features gpu-wgpu gpu_ransac_deferred_conversion_ -- --nocapture
cargo test -p rustsfm --features gpu-wgpu gpu_ransac_invalid_summary_ -- --nocapture
cargo test -p rustsfm --features gpu-wgpu gpu_ransac_summary_count_ -- --nocapture
cargo test -p rustsfm --features gpu-wgpu gpu_ransac_capacity_slices_ -- --nocapture
cargo test -p rustsfm --features gpu-wgpu gpu_ransac_shared_stream_ -- --nocapture
~~~

Expected: compile failures because the runner and scorer seam do not exist.

- [ ] **Step 3: Implement the runner configuration and result**

Use these private types so all three model families share one scheduler:

~~~rust
#[cfg(feature = "gpu-wgpu")]
#[derive(Clone, Copy)]
struct GpuRansacRunConfig<'a> {
    family: &'static str,
    sample_size: usize,
    dynamic_support_observations: usize,
    observation_count: usize,
    threshold: f32,
    kind: TwoViewModelKind,
    options: &'a ColmapRansacOptions,
    policy: GpuRansacBatchPolicy,
}

#[cfg(feature = "gpu-wgpu")]
struct GpuRansacRunResult {
    best: Option<(Matrix3<f64>, ModelSupport)>,
    timing: WgpuRansacStageTiming,
}
~~~

The runner signature is:

~~~rust
fn run_gpu_ransac_batches<S, G, R>(
    scorer: &impl GpuRansacScoring,
    active_indices: &[usize],
    config: GpuRansacRunConfig<'_>,
    mut sample: S,
    mut generate_models: G,
    mut refine: R,
) -> anyhow::Result<GpuRansacRunResult>
where
    S: FnMut(usize) -> Vec<usize>,
    G: FnMut(&[usize]) -> Vec<Matrix3<f64>>,
    R: FnMut(Matrix3<f64>, ModelSupport) -> (Matrix3<f64>, ModelSupport),
~~~

Implement this exact order:

1. Generate GpuRansacTrialGroup values through the physical score end, including empty groups.
2. Stop generation after a short sample and retain the generated prefix plus exhaustion trial.
3. Convert candidates in order and partition them into nominal decision windows. Retain the first conversion/capacity failure with its trial; do not speculatively submit any candidate from its nominal window.
4. Submit only complete valid nominal windows before the deferred window using gpu_ransac_score_slices and append summaries in candidate order. Never split one trial group.
5. Advance decision_end with gpu_ransac_batch_end using config.policy.decision_trials.
6. Before consuming a deferred nominal window, compute the actual decision_end. If the error trial is less than decision_end, return it before any submission or consumption from that window. Otherwise score only complete groups with trial less than decision_end on demand, consume that partial prefix, and discard the invalid suffix.
7. Consume candidates with trial less than decision_end in order. Preserve gpu_summary_support, strict is_better_support, mask readback, gpu_masked_support, local refinement, dynamic_ransac_num_trials, and best update order.
8. Log physical boundaries, logical boundaries, discarded suffixes, and accepted best updates with log::trace; never call the scorer solely for logging.
9. Stop after the decision window containing sampler exhaustion. Ignore exhaustion in a suffix discarded by an earlier dynamic stop.

- [ ] **Step 4: Verify GREEN and deterministic counters**

Run all ten commands from Step 2. Expected: every scripted test passes and the 512/64 case has fewer score/readback calls with identical consumed candidate, mask, refinement, best, and dynamic-limit behavior.

- [ ] **Step 5: Commit Task 3**

~~~bash
git add RustSFM/src/geometry/two_view.rs
git commit -m "feat(rustsfm): add gpu ransac decision frontier"
~~~

### Task 4: Route Essential RANSAC Through The Shared Runner

**Files:**
- Modify: RustSFM/src/geometry/two_view.rs:1553

- [ ] **Step 1: Write the Essential 64/64 versus 512/64 RED regression**

Add estimate_essential_ransac_gpu_with_policy, preserving the current function as a wrapper that calls gpu_ransac_batch_policy(shared_stream). In the GPU test fixture force min_num_trials and max_num_trials to 640, use an owned sampler, and compare:

~~~rust
assert_matrix_bits_eq(reference.0, candidate.0);
assert_eq!(reference.1.inlier_mask, candidate.1.inlier_mask);
assert_eq!(reference.1.inliers, candidate.1.inliers);
assert_eq!(reference.1.residual_sum.to_bits(), candidate.1.residual_sum.to_bits());
assert_eq!(reference.2, candidate.2);
assert_eq!(reference_timing.scorer.mask_calls, candidate_timing.scorer.mask_calls);
assert!(candidate_timing.scorer.score_calls < reference_timing.scorer.score_calls);
~~~

Run the 512/64 call a second time with the same seed and inputs. Apply assert_matrix_bits_eq to the
two candidate matrices and require identical mask, inlier count, residual_sum.to_bits(), success,
score_calls, mask_calls, and models_scored.

Define and reuse this exact matrix helper for all three model families:

~~~rust
fn assert_matrix_bits_eq(left: Matrix3<f64>, right: Matrix3<f64>) {
    for row in 0..3 {
        for column in 0..3 {
            assert_eq!(
                left[(row, column)].to_bits(),
                right[(row, column)].to_bits(),
                "matrix mismatch at ({row}, {column})"
            );
        }
    }
}
~~~

Run:

~~~bash
cargo test -p rustsfm --features gpu-wgpu gpu_essential_decision_batching_is_bit_exact -- --nocapture
~~~

Expected RED: the policy-injected Essential helper is missing.

- [ ] **Step 2: Replace only the Essential chunk loop**

Create the sampler as today, then call run_gpu_ransac_batches with:

- family = "Essential"
- sample_size = 5 or 8
- dynamic_support_observations = active_indices.len(), preserving the current dynamic trial formula; num_observations remains only in the existing sampler-seed construction
- observation_count = pts1.len().min(pts2.len()), preserving the current full-mask expansion size
- kind = TwoViewModelKind::Sampson
- generator = five-point vector or lightweight eight-point option converted to a vector
- refiner = local_optimize_essential_support with all existing arguments

Keep session preparation, fallback eight-point estimate, final support recomputation, final refine_essential_support, and timing accumulation byte-for-byte equivalent outside the removed loop.

- [ ] **Step 3: Verify Essential GREEN and existing tests**

Run:

~~~bash
cargo test -p rustsfm --features gpu-wgpu gpu_essential_decision_batching_is_bit_exact -- --nocapture
cargo test -p rustsfm --features gpu-wgpu gpu_batched_two_view_preserves_cpu_geometric_support -- --nocapture
~~~

Expected: pass or the repository's explicit no-adapter skip; scripted Task 3 tests must still pass without an adapter.

- [ ] **Step 4: Commit Task 4**

~~~bash
git add RustSFM/src/geometry/two_view.rs
git commit -m "refactor(rustsfm): batch essential ransac decisions"
~~~

### Task 5: Route Fundamental RANSAC Through The Shared Runner

**Files:**
- Modify: RustSFM/src/geometry/two_view.rs:2482

- [ ] **Step 1: Write the Fundamental exact-parity RED test**

Add gpu_fundamental_decision_batching_is_bit_exact using the existing deterministic two-view fixture, a ColmapRansacOptions value with min_num_trials = max_num_trials = 640, random_seed 17, and shared_stream false. Call estimate_fundamental_ransac_gpu_with_policy twice with 64/64 and 512/64, unwrap both results, and assert:

~~~rust
assert_matrix_bits_eq(reference.0, candidate.0);
assert_eq!(reference.1.inlier_mask, candidate.1.inlier_mask);
assert_eq!(reference.1.inliers, candidate.1.inliers);
assert_eq!(reference.1.residual_sum.to_bits(), candidate.1.residual_sum.to_bits());
assert_eq!(reference.2, candidate.2);
assert_eq!(reference_timing.scorer.mask_calls, candidate_timing.scorer.mask_calls);
assert!(candidate_timing.scorer.score_calls < reference_timing.scorer.score_calls);
~~~

Repeat the 512/64 Fundamental call once and require exact matrix/support/success plus identical
score_calls, mask_calls, and models_scored between the two candidate runs.

Run:

~~~bash
cargo test -p rustsfm --features gpu-wgpu gpu_fundamental_decision_batching_is_bit_exact -- --nocapture
~~~

Expected RED: policy-injected Fundamental helper missing.

- [ ] **Step 2: Replace only the Fundamental chunk loop**

Call the shared runner with sample_size 7, dynamic_support_observations active_indices.len(), observation_count pts1.len().min(pts2.len()), Sampson scoring, estimate_fundamental_seven_point_indexed, and refine_fundamental_support using COLMAP_LORANSAC_LOCAL_TRIALS. Preserve fallback eight-point estimation and final refinement.

- [ ] **Step 3: Verify GREEN and multi-model ordering**

Run the test from Step 1 plus:

~~~bash
cargo test -p rustsfm --features gpu-wgpu gpu_ransac_candidates_keep_trial_and_model_ordinals -- --nocapture
~~~

Expected: pass or explicit no-adapter skip; the pure ordinal test always passes.

- [ ] **Step 4: Commit Task 5**

~~~bash
git add RustSFM/src/geometry/two_view.rs
git commit -m "refactor(rustsfm): batch fundamental ransac decisions"
~~~

### Task 6: Route Homography RANSAC Through The Shared Runner

**Files:**
- Modify: RustSFM/src/geometry/two_view.rs:3121

- [ ] **Step 1: Write the Homography exact-parity RED test**

Add gpu_homography_decision_batching_is_bit_exact using the existing translated-grid fixture, min_num_trials = max_num_trials = 640, random_seed 42, and shared_stream false. Call estimate_homography_ransac_gpu_with_policy with 64/64 and 512/64 and assert:

~~~rust
assert_matrix_bits_eq(reference.0, candidate.0);
assert_eq!(reference.1.inlier_mask, candidate.1.inlier_mask);
assert_eq!(reference.1.inliers, candidate.1.inliers);
assert_eq!(reference.1.residual_sum.to_bits(), candidate.1.residual_sum.to_bits());
assert_eq!(reference.2, candidate.2);
assert_eq!(reference_timing.scorer.mask_calls, candidate_timing.scorer.mask_calls);
assert!(candidate_timing.scorer.score_calls < reference_timing.scorer.score_calls);
~~~

Repeat the 512/64 Homography call once and require exact matrix/support/success plus identical
score_calls, mask_calls, and models_scored between the two candidate runs.

Run:

~~~bash
cargo test -p rustsfm --features gpu-wgpu gpu_homography_decision_batching_is_bit_exact -- --nocapture
~~~

Expected RED: policy-injected Homography helper missing.

- [ ] **Step 2: Replace only the Homography chunk loop**

Call the shared runner with sample_size 4, dynamic_support_observations active_indices.len(), observation_count pts1.len().min(pts2.len()), HomographyForward scoring, estimate_homography_dlt_indexed converted to a zero-or-one vector, and refine_homography_support using COLMAP_LORANSAC_LOCAL_TRIALS. Preserve fallback DLT and final refinement.

Remove the last legacy gpu_ransac_chunk_end call and its compatibility helper. Production E/F/H wrappers must all select gpu_ransac_batch_policy(shared_stream).

- [ ] **Step 3: Verify GREEN and all focused batching tests**

Run:

~~~bash
cargo test -p rustsfm --features gpu-wgpu gpu_homography_decision_batching_is_bit_exact -- --nocapture
cargo test -p rustsfm --features gpu-wgpu gpu_ransac_ -- --nocapture
cargo test -p rustsfm --features gpu-wgpu gpu_homography_ransac_recovers_grid_deterministically -- --nocapture
~~~

Expected: all pure tests pass; real-GPU tests pass or use explicit no-adapter skip.

- [ ] **Step 4: Commit Task 6**

~~~bash
git add RustSFM/src/geometry/two_view.rs
git commit -m "refactor(rustsfm): batch homography ransac decisions"
~~~

### Task 7: Verify Shared-Stream And Full RustSFM Regression Surface

**Files:**
- Modify: RustSFM/src/geometry/two_view.rs tests only if a regression is found
- Modify: docs/superpowers/plans/2026-08-10-rustsfm-ransac-chunk-512-experiment.md

- [ ] **Step 1: Run shared-stream draw parity**

Run the Task 3 Rc<Cell<usize>> counting-sampler test proving shared mode samples the same 101-trial prefix and leaves the same next-draw index as the 64/64 reference. Do not add an environment-variable mutation; the test calls gpu_ransac_batch_policy(true) directly.

Run:

~~~bash
cargo test -p rustsfm --features gpu-wgpu gpu_ransac_shared_stream_does_not_consume_speculative_samples -- --nocapture
cargo test -p rustsfm --features gpu-wgpu colmap_shared_ransac -- --nocapture
~~~

Expected: all selected tests pass.

- [ ] **Step 2: Run focused, library, feature, format, and diff checks**

Run:

~~~bash
cargo test -p rustsfm --features gpu-wgpu --lib
cargo check -p rustsfm --no-default-features
cargo check -p rustsfm --no-default-features --features gpu-wgpu
cargo fmt --all -- --check
git diff --check
~~~

Expected: exit 0 for each command. If unrelated platform prerequisites prevent the full library suite, record exact failing test names and prove every affected GPU RANSAC test passes separately; do not describe the full suite as passing.

- [ ] **Step 3: Record verification evidence and commit**

Append exact commands, pass/fail counts, adapter identity, and any explicit skips to the experiment plan.

~~~bash
git add docs/superpowers/plans/2026-08-10-rustsfm-ransac-chunk-512-experiment.md RustSFM/src/geometry/two_view.rs
git commit -m "test(rustsfm): verify gpu ransac decision batching"
~~~

### Task 8: Run The Single-Pair And 96x3 Strict SQLite Gates

**Files:**
- Modify: docs/superpowers/plans/2026-08-10-rustsfm-ransac-chunk-512-experiment.md
- Create outside git: unique /tmp artifact directories and JSON reports

- [ ] **Step 1: Reverify baseline provenance before benchmarking**

Recompute the pinned source database, chunk-64 binary, and baseline JSON hashes recorded in the experiment plan. Stop if any hash differs.

Run:

~~~bash
shasum -a 256 /tmp/rustsfm-gpu-ransac-chunk-64
shasum -a 256 /tmp/rustsfm-chunk64-diagnostics
shasum -a 256 /tmp/rustsfm-gpu-ransac-chunk-64-96x3.json
shasum -a 256 /Users/tfjiang/Projects/RustScan/test_data/flowers2/out9/Untitled.rustscanproject/Cache/.staging/keyframe_sfm-1/rustsfm/Cache/database.db
~~~

Require, in order:

~~~text
880c7d726f789621d641f89ad6dbae7c990137adb29cda6789dbda57d8792541
f441876c99db3d5903c2003e5c1edb0139611829c67032c2e622e345a025914b
4c56571897acfd0dd8d95079d26ad78cb0196fd5bce15fc28f1cf2b28cdea159
dcf79fa307a6294195a8e5db1cddb185bbc1baca2ee490061b89f2a5961a052c
~~~

- [ ] **Step 2: Build the release diagnostic binary**

Run:

~~~bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo build --release \
  -p rustsfm --no-default-features --features gpu-wgpu
~~~

Expected: release build succeeds without forcing a Metal or Vulkan backend.

- [ ] **Step 3: Run the retained single-pair gate**

Create a unique parent and run exactly:

~~~bash
RUSTSFM_PAIR1_ROOT=$(mktemp -d /tmp/rustsfm-decision-pair1.XXXXXX)
/Users/tfjiang/Projects/RustScan/target/release/rustsfm benchmark-match-pairs \
  --database /Users/tfjiang/Projects/RustScan/test_data/flowers2/out9/Untitled.rustscanproject/Cache/.staging/keyframe_sfm-1/rustsfm/Cache/database.db \
  --window 5 --pair-limit 1 --repetitions 1 --use-gpu --random-seed 0 \
  --artifacts-dir "$RUSTSFM_PAIR1_ROOT/artifacts" \
  --output-json "$RUSTSFM_PAIR1_ROOT/report.json"
jq -e '
  .pair_count == 1 and .repetitions == 1 and
  .runs[0].backend == "wgpu_match_and_score" and
  .runs[0].matched_pairs == 1 and .runs[0].verified_pairs == 1 and
  .runs[0].total_matches == 1419 and
  .runs[0].result_fingerprint == "c8e440cd266b3609f75c90e5c90260f41af1b765a9e709e0202cf605ec7eb5f5"
' "$RUSTSFM_PAIR1_ROOT/report.json"
~~~

Require fingerprint:

~~~text
c8e440cd266b3609f75c90e5c90260f41af1b765a9e709e0202cf605ec7eb5f5
~~~

Also require 1,419 raw matches, 1,383 geometry inliers, config 2, and exact matches/data/F/E/H/qvec/tvec bytes. On any mismatch, stop and preserve both databases.

- [ ] **Step 4: Run the 96-pair three-repeat gate**

Create another unique parent and run exactly:

~~~bash
RUSTSFM_96_ROOT=$(mktemp -d /tmp/rustsfm-decision-96x3.XXXXXX)
/tmp/rustsfm-chunk64-diagnostics benchmark-match-pairs \
  --database /Users/tfjiang/Projects/RustScan/test_data/flowers2/out9/Untitled.rustscanproject/Cache/.staging/keyframe_sfm-1/rustsfm/Cache/database.db \
  --window 5 --pair-limit 96 --repetitions 3 --use-gpu --random-seed 0 \
  --artifacts-dir "$RUSTSFM_96_ROOT/baseline-artifacts" \
  --output-json "$RUSTSFM_96_ROOT/baseline.json"
/Users/tfjiang/Projects/RustScan/target/release/rustsfm benchmark-match-pairs \
  --database /Users/tfjiang/Projects/RustScan/test_data/flowers2/out9/Untitled.rustscanproject/Cache/.staging/keyframe_sfm-1/rustsfm/Cache/database.db \
  --window 5 --pair-limit 96 --repetitions 3 --use-gpu --random-seed 0 \
  --artifacts-dir "$RUSTSFM_96_ROOT/candidate-artifacts" \
  --output-json "$RUSTSFM_96_ROOT/report.json"
jq -e --slurpfile fresh "$RUSTSFM_96_ROOT/baseline.json" '
  def score_calls:
    .timings.gpu_geometry_detail |
    [.essential.scorer.score_calls, .fundamental.scorer.score_calls,
     .homography.scorer.score_calls] | add;
  def readback_calls:
    .timings.gpu_geometry_detail |
    [.essential.scorer.readback_calls, .fundamental.scorer.readback_calls,
     .homography.scorer.readback_calls] | add;
  .pair_count == 96 and .repetitions == 3 and (.runs | length) == 3 and
  all(.runs[];
    .backend == "wgpu_match_and_score" and
    .matched_pairs == 96 and .verified_pairs == 96 and .total_matches == 62409 and
    .result_fingerprint == "5e05ca629b63c98ae63c95ce0f37fe49a43eb870760e598352c8f8ef3d84e8ed" and
    all(.. | numbers; (isnan | not) and (isinfinite | not))) and
  all($fresh[0].runs[];
    .result_fingerprint == "5e05ca629b63c98ae63c95ce0f37fe49a43eb870760e598352c8f8ef3d84e8ed") and
  (([.runs[].matching_seconds] | add / length) <=
   ([$fresh[0].runs[].matching_seconds] | add / length)) and
  (([.runs[] | score_calls] | add) < ([$fresh[0].runs[] | score_calls] | add)) and
  (([.runs[] | readback_calls] | add) < ([$fresh[0].runs[] | readback_calls] | add))
' "$RUSTSFM_96_ROOT/report.json"
~~~

Require every run fingerprint:

~~~text
5e05ca629b63c98ae63c95ce0f37fe49a43eb870760e598352c8f8ef3d84e8ed
~~~

Require 96 matched, 96 verified, and 62,409 total matches in every run. Compare score_calls,
readback_calls, models_scored, and mean runtime to the fresh adjacent 64/64 run on the same adapter.
Score/readback calls must decrease and bounded mean runtime must not regress. The pinned prior JSON
is used only to confirm the historical fingerprint and counter provenance, never as the runtime
comparator.

- [ ] **Step 5: Record and commit bounded gate evidence**

Append artifact paths, SHA-256 hashes, fingerprints, aggregate counts, scorer counters, raw durations, mean, and speedup to the experiment plan.

~~~bash
git add docs/superpowers/plans/2026-08-10-rustsfm-ransac-chunk-512-experiment.md
git commit -m "perf(rustsfm): verify bounded ransac batching parity"
~~~

### Task 9: Run The Sequential 2,890-Pair Gate

**Files:**
- Modify: docs/superpowers/plans/2026-08-10-rustsfm-ransac-chunk-512-experiment.md
- Create outside git: unique /tmp full-run artifacts and JSON report

- [ ] **Step 1: Confirm bounded gate prerequisites**

Read the Task 8 evidence from the committed plan. Do not start if either strict fingerprint or performance requirement is missing.

- [ ] **Step 2: Run the full sequential comparison**

There is no accepted chunk-64 2,890-pair artifact because the earlier 512 experiment stopped at its bounded mismatch. First create a fresh chunk-64 baseline with the preserved diagnostics binary, then run the candidate. The explicit 2,890 limit belongs only to this reproducible diagnostic gate; it does not change the unlimited benchmark default or any production image/keyframe/pair behavior:

~~~bash
RUSTSFM_FULL_ROOT=$(mktemp -d /tmp/rustsfm-decision-full.XXXXXX)
/tmp/rustsfm-chunk64-diagnostics benchmark-match-pairs \
  --database /Users/tfjiang/Projects/RustScan/test_data/flowers2/out9/Untitled.rustscanproject/Cache/.staging/keyframe_sfm-1/rustsfm/Cache/database.db \
  --window 5 --pair-limit 2890 --repetitions 1 --use-gpu --random-seed 0 \
  --artifacts-dir "$RUSTSFM_FULL_ROOT/baseline-artifacts" \
  --output-json "$RUSTSFM_FULL_ROOT/baseline.json"
/Users/tfjiang/Projects/RustScan/target/release/rustsfm benchmark-match-pairs \
  --database /Users/tfjiang/Projects/RustScan/test_data/flowers2/out9/Untitled.rustscanproject/Cache/.staging/keyframe_sfm-1/rustsfm/Cache/database.db \
  --window 5 --pair-limit 2890 --repetitions 1 --use-gpu --random-seed 0 \
  --artifacts-dir "$RUSTSFM_FULL_ROOT/candidate-artifacts" \
  --output-json "$RUSTSFM_FULL_ROOT/candidate.json"
jq -e --slurpfile baseline "$RUSTSFM_FULL_ROOT/baseline.json" '
  .pair_count == 2890 and .requested_pair_limit == 2890 and
  .runs[0].backend == "wgpu_match_and_score" and
  .runs[0].pair_count == 2890 and
  .runs[0].matched_pairs == $baseline[0].runs[0].matched_pairs and
  .runs[0].verified_pairs == $baseline[0].runs[0].verified_pairs and
  .runs[0].total_matches == $baseline[0].runs[0].total_matches and
  .runs[0].result_fingerprint == $baseline[0].runs[0].result_fingerprint
' "$RUSTSFM_FULL_ROOT/candidate.json"
~~~

The benchmark result_fingerprint hashes presence plus rows, cols, data, config, F, E, H, qvec, and tvec for every selected pair through ColmapDatabase::selected_pair_output_fingerprint. Fingerprint equality is therefore the byte-exact row gate; aggregate count equality alone is not sufficient.

- [ ] **Step 3: Record full evidence and commit**

Append command, adapter/backend, source/artifact hashes, full fingerprint, counts, scorer counters, elapsed time, and comparison result.

~~~bash
git add docs/superpowers/plans/2026-08-10-rustsfm-ransac-chunk-512-experiment.md
git commit -m "perf(rustsfm): verify full ransac batching parity"
~~~

### Task 10: Independent Review, Main Synchronization, Merge, And Cleanup

**Files:**
- Review every commit after a22a033
- No production edit unless review or synchronization exposes a defect

- [ ] **Step 1: Request specification compliance review**

Provide the design document, this plan, base SHA a22a033, final implementation SHA, and all retained benchmark evidence to a fresh read-only reviewer. Require explicit Critical/Important findings for decision frontier, multi-model trial ordering, shared RNG, deferred suffix handling, generic wgpu selection, absence of pair limits, and strict database parity.

- [ ] **Step 2: Fix and re-review every Critical or Important specification issue**

For each valid issue, first add a failing regression test, verify RED, implement the smallest correction, verify GREEN, commit, and return to Step 1 until approved.

- [ ] **Step 3: Request code-quality review**

Use a fresh reviewer after specification approval. Require review of overflow handling, candidate lifetimes/order, error context, score/mask counters, duplicated E/F/H logic, logging cost, test quality, and unrelated diff.

- [ ] **Step 4: Fix and re-review every Critical or Important quality issue**

Follow RED/GREEN for behavioral fixes. Repeat review until approved.

- [ ] **Step 5: Synchronize with main and rerun affected checks**

Fetch local main state, merge or rebase it into codex/ransac-chunk-512-experiment without discarding user changes, then rerun Task 7 focused checks and Task 8 single-pair/96x3 strict gates. A changed main invalidates old integration evidence.

Before the post-synchronization performance rerun, build a fresh 64/64 reference from the exact
synchronized HEAD in a detached temporary worktree. Change only
GPU_RANSAC_SCORE_BATCH_TRIALS from 512 to 64 there with apply_patch, build rustsfm release with
--no-default-features --features gpu-wgpu and an isolated CARGO_TARGET_DIR, and preserve the binary
as /tmp/rustsfm-decision-reference-after-sync. Record the synchronized HEAD, reference diff, and
binary SHA-256, then remove the detached reference worktree. Rerun Task 8 with this binary in place
of /tmp/rustsfm-chunk64-diagnostics. This makes the runtime comparator identical to the candidate
source except for physical score width.

- [ ] **Step 6: Merge only after every gate is green**

Merge codex/ransac-chunk-512-experiment into main locally. Verify main points to the integrated history and rerun git status plus the focused GPU RANSAC tests.

- [ ] **Step 7: Remove the linked worktree and merged branch**

Only after the merge and post-merge verification succeed, remove:

~~~text
/Users/tfjiang/Projects/RustScan/.worktrees/ransac-chunk-512
codex/ransac-chunk-512-experiment
~~~

If parity, performance, review, or synchronization fails, do not merge or clean up. Preserve the branch, worktree, databases, and reports and state the exact failed gate.
