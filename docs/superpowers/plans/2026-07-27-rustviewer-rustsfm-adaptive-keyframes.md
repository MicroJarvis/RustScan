# RustViewer RustSFM Adaptive Keyframes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make RustViewer ask RustSFM to select a dataset-dependent number of geometrically connected keyframes, reuse the selection database and sparse model for full-frame PnP, and start RustGS only after every imported frame has a pose.

**Architecture:** Add a pure adaptive-selection policy plus a RustSFM metric-acquisition adapter that extracts SIFT for the imported sequence and evaluates explicit anchor/candidate pairs through the existing wgpu-capable matcher. RustViewer persists the selected IDs, pair diagnostics, feature database, and keyframe sparse model as immutable stage artifacts; the PnP stage hydrates those artifacts and calls `register_remaining_sequence_frames`, avoiding the existing second keyframe reconstruction.

**Tech Stack:** Rust 2021, serde/serde_json, anyhow/thiserror, RustSFM SIFT and two-view geometry, wgpu, RustViewer project manifests and pipeline artifacts, Cargo tests.

---

## File Map

- Create `RustSFM/src/sequence_registration/adaptive_keyframes.rs`: public configuration/result/diagnostic types, deterministic pure policy, and runtime SIFT/two-view metric acquisition.
- Modify `RustSFM/src/sequence_registration.rs`: register/re-export the adaptive module and expose only the crate-private sequence helpers the adapter needs.
- Modify `RustSFM/src/lib.rs`: export the new controlled RustSFM API.
- Modify `RustSFM/src/task.rs`: add machine-readable keyframe-selection progress stage and operations.
- Create `RustSFM/tests/adaptive_keyframes.rs`: public API, policy, runtime cancellation, metric acquisition, and feature-reuse tests.
- Modify `RustSFM/tests/task_control.rs`: lock down the new task enum wire names and function signature.
- Modify `RustViewer/src/project/manifest.rs`: add the adaptive/all-images mode, serde defaults, thresholds, and validation.
- Modify `RustViewer/src/project/mod.rs`: export the new selection mode.
- Modify `RustViewer/tests/project_store.rs`: verify schema-v1 manifests without the new fields open in adaptive mode and invalid thresholds are rejected.
- Modify `RustViewer/src/pipeline/rustsfm_worker.rs`: call RustSFM selection, apply the 4,096/local-window-5 preset, persist reusable artifacts, and resume PnP from them.
- Modify the unit tests in `RustViewer/src/pipeline/rustsfm_worker.rs`: verify selection wiring, mapper preset, artifact discovery/hydration, and full-frame coverage behavior.

### Task 1: Pure RustSFM Adaptive Selection Policy

**Files:**
- Create: `RustSFM/src/sequence_registration/adaptive_keyframes.rs`
- Modify: `RustSFM/src/sequence_registration.rs:1-30`
- Modify: `RustSFM/src/lib.rs:1-20`
- Test: `RustSFM/tests/adaptive_keyframes.rs`

- [ ] **Step 1: Write failing public-policy tests**

Create `RustSFM/tests/adaptive_keyframes.rs` with helpers that construct exact pair evidence and tests for redundant sequences, connected transitions, bridge retention, boundary preservation, determinism, uniqueness, and validation:

```rust
use rustsfm::{
    select_adaptive_keyframes_from_metrics, AdaptiveKeyframePairMetrics,
    AdaptiveKeyframeSelectionConfig, AdaptiveKeyframeSelectionDecision,
    AdaptiveKeyframeSelectionError,
};

fn config() -> AdaptiveKeyframeSelectionConfig {
    AdaptiveKeyframeSelectionConfig {
        retention_feature_coverage: 0.35,
        min_inliers: 15,
        min_inlier_ratio: 0.20,
        min_triangulated: 4,
    }
}

fn metrics(anchor: u32, candidate: u32, coverage: f64, connected: bool) -> AdaptiveKeyframePairMetrics {
    AdaptiveKeyframePairMetrics {
        anchor_frame_id: anchor,
        candidate_frame_id: candidate,
        descriptor_matches: if connected { 100 } else { 10 },
        inliers: if connected { 50 } else { 2 },
        triangulated: if connected { 40 } else { 0 },
        inlier_ratio: if connected { 0.5 } else { 0.2 },
        feature_coverage: coverage,
    }
}

#[test]
fn redundant_sequence_selects_only_boundary_frames() {
    let evidence = [
        metrics(1, 2, 0.80, true),
        metrics(1, 3, 0.75, true),
        metrics(1, 4, 0.70, true),
    ];
    let result = select_adaptive_keyframes_from_metrics(&[1, 2, 3, 4], &evidence, &config()).unwrap();
    assert_eq!(result.selected_frame_ids, vec![1, 4]);
    assert_eq!(result.evaluated_pairs, 3);
}

#[test]
fn lower_coverage_connected_transition_advances_anchor() {
    let evidence = [
        metrics(1, 2, 0.80, true),
        metrics(1, 3, 0.25, true),
        metrics(3, 4, 0.25, true),
    ];
    let result = select_adaptive_keyframes_from_metrics(&[1, 2, 3, 4], &evidence, &config()).unwrap();
    assert_eq!(result.selected_frame_ids, vec![1, 3, 4]);
    assert!(result.diagnostics.iter().any(|diagnostic| {
        diagnostic.metrics.candidate_frame_id == 3
            && diagnostic.decision == AdaptiveKeyframeSelectionDecision::ConnectedTransition
    }));
}

#[test]
fn connectivity_loss_retains_last_connected_bridge_and_terminates() {
    let evidence = [
        metrics(1, 2, 0.80, true),
        metrics(1, 3, 0.10, false),
        metrics(2, 3, 0.25, true),
        metrics(3, 4, 0.80, true),
    ];
    let result = select_adaptive_keyframes_from_metrics(&[1, 2, 3, 4], &evidence, &config()).unwrap();
    assert_eq!(result.selected_frame_ids, vec![1, 2, 3, 4]);
    assert_eq!(result.selected_frame_ids.iter().copied().collect::<std::collections::BTreeSet<_>>().len(), 4);
    assert!(result.evaluated_pairs <= evidence.len());
}

#[test]
fn fast_change_selects_more_frames_than_redundant_input_of_same_length() {
    let redundant = (2..=6).map(|candidate| metrics(1, candidate, 0.8, true)).collect::<Vec<_>>();
    let changing = [
        metrics(1, 2, 0.2, true), metrics(2, 3, 0.2, true),
        metrics(3, 4, 0.2, true), metrics(4, 5, 0.2, true),
        metrics(5, 6, 0.2, true),
    ];
    let a = select_adaptive_keyframes_from_metrics(&[1, 2, 3, 4, 5, 6], &redundant, &config()).unwrap();
    let b = select_adaptive_keyframes_from_metrics(&[1, 2, 3, 4, 5, 6], &changing, &config()).unwrap();
    assert!(b.selected_frame_ids.len() > a.selected_frame_ids.len());
}

#[test]
fn policy_is_deterministic_ordered_unique_and_preserves_boundaries() {
    let evidence = [metrics(10, 20, 0.8, true), metrics(10, 30, 0.8, true)];
    let first = select_adaptive_keyframes_from_metrics(&[10, 20, 30], &evidence, &config()).unwrap();
    let second = select_adaptive_keyframes_from_metrics(&[10, 20, 30], &evidence, &config()).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.selected_frame_ids.first(), Some(&10));
    assert_eq!(first.selected_frame_ids.last(), Some(&30));
}

#[test]
fn policy_rejects_invalid_thresholds_insufficient_frames_and_missing_evidence() {
    let mut invalid = config();
    invalid.retention_feature_coverage = f64::NAN;
    assert!(matches!(invalid.validate(), Err(AdaptiveKeyframeSelectionError::InvalidConfigMetric { .. })));
    assert!(matches!(
        select_adaptive_keyframes_from_metrics(&[1], &[], &config()),
        Err(AdaptiveKeyframeSelectionError::InsufficientFrames { usable_frames: 1 })
    ));
    assert!(matches!(
        select_adaptive_keyframes_from_metrics(&[1, 2], &[], &config()),
        Err(AdaptiveKeyframeSelectionError::MissingPairEvidence { anchor_frame_id: 1, candidate_frame_id: 2 })
    ));
}
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm --test adaptive_keyframes --no-default-features
```

Expected: compilation fails because the adaptive types and functions are not exported.

- [ ] **Step 3: Implement the public types, validation, and deterministic policy**

In `RustSFM/src/sequence_registration/adaptive_keyframes.rs`, define serializable public types and a pure evidence lookup entry point:

```rust
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AdaptiveKeyframeSelectionConfig {
    pub retention_feature_coverage: f64,
    pub min_inliers: usize,
    pub min_inlier_ratio: f64,
    pub min_triangulated: usize,
}

impl Default for AdaptiveKeyframeSelectionConfig {
    fn default() -> Self {
        Self {
            retention_feature_coverage: 0.35,
            min_inliers: 15,
            min_inlier_ratio: 0.20,
            min_triangulated: 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AdaptiveKeyframePairMetrics {
    pub anchor_frame_id: u32,
    pub candidate_frame_id: u32,
    pub descriptor_matches: usize,
    pub inliers: usize,
    pub triangulated: usize,
    pub inlier_ratio: f64,
    pub feature_coverage: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdaptiveKeyframeSelectionDecision {
    Redundant,
    ConnectedTransition,
    ConnectivityBridge,
    ForcedProgress,
    Boundary,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AdaptiveKeyframePairDiagnostic {
    pub metrics: AdaptiveKeyframePairMetrics,
    pub decision: AdaptiveKeyframeSelectionDecision,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AdaptiveKeyframeSelectionResult {
    pub imported_frames: usize,
    pub usable_frames: usize,
    pub selected_frame_ids: Vec<u32>,
    pub config: AdaptiveKeyframeSelectionConfig,
    pub evaluated_pairs: usize,
    pub diagnostics: Vec<AdaptiveKeyframePairDiagnostic>,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum AdaptiveKeyframeSelectionError {
    #[error("invalid adaptive keyframe configuration field {field}")]
    InvalidConfigMetric { field: &'static str },
    #[error("adaptive keyframe selection requires at least two usable frames, found {usable_frames}")]
    InsufficientFrames { usable_frames: usize },
    #[error("adaptive keyframe selection contains duplicate frame id {frame_id}")]
    DuplicateFrameId { frame_id: u32 },
    #[error("missing pair evidence for anchor {anchor_frame_id} and candidate {candidate_frame_id}")]
    MissingPairEvidence { anchor_frame_id: u32, candidate_frame_id: u32 },
    #[error("pair evidence for {anchor_frame_id}-{candidate_frame_id} contains non-finite {field}")]
    NonFinitePairMetric { anchor_frame_id: u32, candidate_frame_id: u32, field: &'static str },
}
```

Implement `AdaptiveKeyframeSelectionConfig::validate`, a private callback-driven scan, and `select_adaptive_keyframes_from_metrics`. Validation accepts only finite `retention_feature_coverage` and `min_inlier_ratio` values in `(0.0, 1.0]`, and requires nonzero `min_inliers` and `min_triangulated`. The scan must:

1. select the first frame;
2. treat coverage at or above `retention_feature_coverage` as redundant regardless of low parallax/invalid geometry;
3. remember only geometrically connected redundant candidates as bridge candidates;
4. select a low-coverage candidate when `inliers`, `inlier_ratio`, and `triangulated` pass;
5. on connectivity loss, select the last connected bridge and retry the current candidate from the new anchor, or select the current candidate to guarantee forward progress;
6. append the last frame as `Boundary` if it was not selected by the scan;
7. reject duplicate IDs, missing evidence, non-finite ratios/coverage, and output that is not stable, ordered, and unique.

In `RustSFM/src/sequence_registration.rs` add:

```rust
mod adaptive_keyframes;
pub use adaptive_keyframes::{
    select_adaptive_keyframes_from_metrics, AdaptiveKeyframePairDiagnostic,
    AdaptiveKeyframePairMetrics, AdaptiveKeyframeSelectionConfig,
    AdaptiveKeyframeSelectionDecision, AdaptiveKeyframeSelectionError,
    AdaptiveKeyframeSelectionResult,
};
```

Re-export the same API from `RustSFM/src/lib.rs`.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm --test adaptive_keyframes --no-default-features
```

Expected: all pure-policy tests pass with zero failures.

- [ ] **Step 5: Commit Task 1**

```bash
git add RustSFM/src/sequence_registration/adaptive_keyframes.rs RustSFM/src/sequence_registration.rs RustSFM/src/lib.rs RustSFM/tests/adaptive_keyframes.rs
git commit -m "feat(rustsfm): add adaptive keyframe policy"
```

### Task 2: RustSFM SIFT And Two-View Metric Adapter

**Files:**
- Modify: `RustSFM/src/sequence_registration/adaptive_keyframes.rs`
- Modify: `RustSFM/src/sequence_registration.rs:296-447,1117-1158`
- Modify: `RustSFM/src/task.rs:1-45`
- Modify: `RustSFM/src/lib.rs:1-20`
- Modify: `RustSFM/tests/adaptive_keyframes.rs`
- Modify: `RustSFM/tests/task_control.rs:141-182,250-330`

- [ ] **Step 1: Add failing controlled-runtime and wire-format tests**

Extend `RustSFM/tests/task_control.rs` so JSON round trips include:

```rust
(SfmTaskStage::KeyframeSelection, "keyframe_selection")
(SfmTaskOperation::EvaluateKeyframePair, "evaluate_keyframe_pair")
(SfmTaskOperation::SelectKeyframe, "select_keyframe")
```

Add a compile-only signature assertion:

```rust
type ControlledAdaptiveSelectionApi =
    for<'frames, 'selection, 'mapper, 'output, 'context, 'task> fn(
        &'frames [SequenceFrame],
        &'selection AdaptiveKeyframeSelectionConfig,
        &'mapper MapperConfig,
        &'output Path,
        &'context mut SfmTaskContext<'task>,
    ) -> anyhow::Result<AdaptiveKeyframeSelectionResult>;
let _: ControlledAdaptiveSelectionApi = run_adaptive_keyframe_selection;
```

Extend `RustSFM/tests/adaptive_keyframes.rs` with a cancellation test that creates two valid fixture image paths, requests cancellation before the call, and asserts the returned error downcasts to `SfmTaskStop::Cancelled`. Add a small translated textured image sequence test using CPU SIFT (`use_gpu = false`) and permissive geometry thresholds; assert that the result preserves first/last IDs, reports at least one evaluated pair, records finite ratios, and creates `output/Cache/database.db`.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm --test task_control --test adaptive_keyframes --no-default-features
```

Expected: compilation fails because the new task variants and runtime entry point do not exist.

- [ ] **Step 3: Add selection progress variants**

In `RustSFM/src/task.rs`, add `KeyframeSelection` to `SfmTaskStage`, and add `EvaluateKeyframePair` and `SelectKeyframe` to `SfmTaskOperation`. Do not change or reinterpret existing wire names.

- [ ] **Step 4: Implement runtime acquisition using the existing RustSFM backends**

Expose these helpers as `pub(super)` in `RustSFM/src/sequence_registration.rs` so the child module can reuse the exact sequence database layout:

```rust
pub(super) fn link_or_copy_stable_image(...)
pub(super) fn import_database_images(...)
pub(super) fn database_image_ids_for_indices(...)
pub(super) fn sequence_match_options(...)
pub(super) fn database_features_exist(...)
```

In the adaptive module, implement:

```rust
pub fn run_adaptive_keyframe_selection(
    frames: &[SequenceFrame],
    config: &AdaptiveKeyframeSelectionConfig,
    mapper_config: &MapperConfig,
    output: &Path,
    task: &mut SfmTaskContext<'_>,
) -> anyhow::Result<AdaptiveKeyframeSelectionResult>
```

The implementation must perform these concrete operations in order:

1. validate the selection configuration, unique frame IDs/stable file names, at least two frames, and `FeatureType::Sift`;
2. checkpoint before filesystem/image work;
3. create `output/Cache/sequence`, hard-link or copy every image under its stable name, and import every frame into `output/Cache/database.db` with database ID `sequence_index + 1`;
4. clone `mapper_config.sift_extraction`, set `max_num_features = mapper_config.max_features`, and call `extract_selected_features_to_database_with_task` only for images missing keypoints/descriptors;
5. read `num_keypoints_for_image` for each database image, reject fewer than two usable images and missing feature evidence at either sequence boundary;
6. build one `ExplicitPairMatchingSession` from `sequence_match_options(mapper_config)` so wgpu resources are reused across all evaluated pairs;
7. before every pair, call `task.checkpoint()`, match exactly one explicit database-ID pair with `match_explicit_image_pairs_to_database_with_session`, and calculate:

```rust
let inlier_ratio = if pair.num_matches == 0 { 0.0 } else { pair.num_inliers as f64 / pair.num_matches as f64 };
let feature_coverage = pair.num_inliers as f64 / anchor_features.min(candidate_features) as f64;
```

8. feed the metrics into the pure callback-driven policy, retaining every evaluated diagnostic;
9. emit `KeyframeSelection/Begin`, one `EvaluateKeyframePair` progress event per pair with frame IDs, one `SelectKeyframe` event whenever selection advances, and `KeyframeSelection/Complete`; for selection events put the selected keyframe count in `completed`, the imported frame count in `total`, and the current anchor/candidate frame IDs in `pair`;
10. set `result.imported_frames = frames.len()` and `result.usable_frames` to the positive-feature count. The database remains at the documented `output/Cache/database.db` cache path and is not embedded as an absolute path in the serializable result.

Use the existing `mapper_config.sift_extraction.use_gpu` and `mapper_config.sift_matching.use_gpu` values unchanged. Do not inspect or force the concrete wgpu backend.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm --test task_control --test adaptive_keyframes --no-default-features
```

Expected: task wire/API tests, cancellation, and CPU metric-acquisition tests pass.

- [ ] **Step 6: Run the runtime test with the wgpu feature**

Run:

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm --features gpu-wgpu --test adaptive_keyframes
```

Expected: the runtime adapter test passes; an additional wgpu test may return early only when `WgpuContext::try_new_optional()` reports no compatible adapter.

- [ ] **Step 7: Commit Task 2**

```bash
git add RustSFM/src/sequence_registration/adaptive_keyframes.rs RustSFM/src/sequence_registration.rs RustSFM/src/task.rs RustSFM/src/lib.rs RustSFM/tests/adaptive_keyframes.rs RustSFM/tests/task_control.rs
git commit -m "feat(rustsfm): acquire adaptive keyframe metrics"
```

### Task 3: Prove Selection Features Are Reused By Reconstruction

**Files:**
- Modify: `RustSFM/tests/adaptive_keyframes.rs`
- Modify: `RustSFM/src/sequence_registration/adaptive_keyframes.rs` only if the test exposes a reuse defect
- Modify: `RustSFM/src/sequence_registration.rs` only if the test exposes a reuse defect

- [ ] **Step 1: Add a failing selection-to-reconstruction reuse test**

Using the same synthetic sequence fixture, run `run_adaptive_keyframe_selection`, discard the events accumulated during selection, then call `run_keyframe_reconstruction` with `selection.selected_frame_ids`, the same mapper config, and the same output directory. Assert:

```rust
assert_eq!(reconstruction.database, output.join("Cache/database.db"));
assert_eq!(reconstruction.keyframe_ids, selection.selected_frame_ids);
assert_eq!(
    reconstruction_events.iter()
        .filter(|event| event.operation == SfmTaskOperation::ExtractImage)
        .count(),
    0,
    "selected image features must be reused from adaptive selection",
);
```

Also set `mapper_config.matching_pair_strategy = MatchingPairStrategy::LocalWindow { window: 5 }` and assert the observed keyframe `MatchPairBatch` event count equals `generate_matching_pairs(selected_count, LocalWindow { window: 5 }).len()`, proving no quadratic pair expansion.

- [ ] **Step 2: Run the single test and verify RED when reuse is incomplete**

Run:

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm --features gpu-wgpu --test adaptive_keyframes selection_features_are_reused_by_keyframe_reconstruction -- --exact
```

Expected before any required correction: either PASS because the existing database-missing-feature guard is sufficient, or FAIL specifically because the adaptive adapter used a different database/image layout. A compile error or unrelated fixture failure must be corrected before proceeding.

- [ ] **Step 3: Make the minimum layout/reuse correction if RED**

Both APIs must use these exact shared paths:

```text
<output>/Cache/database.db
<output>/Cache/sequence/<stable image names>
<output>/Cache/keyframes/<selected stable image names>
```

Do not copy descriptors between databases and do not re-extract. `run_keyframe_reconstruction` must continue checking `database_features_exist` and skip extraction for selected rows already populated by adaptive selection.

- [ ] **Step 4: Re-run and verify GREEN**

Run:

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm --features gpu-wgpu --test adaptive_keyframes selection_features_are_reused_by_keyframe_reconstruction -- --exact
```

Expected: PASS and zero reconstruction-stage `ExtractImage` events.

- [ ] **Step 5: Commit Task 3**

```bash
git add RustSFM/tests/adaptive_keyframes.rs RustSFM/src/sequence_registration/adaptive_keyframes.rs RustSFM/src/sequence_registration.rs
git commit -m "test(rustsfm): verify adaptive feature reuse"
```

### Task 4: RustViewer Manifest Compatibility And Mapper Preset

**Files:**
- Modify: `RustViewer/src/project/manifest.rs:241-256,300-351,420-520`
- Modify: `RustViewer/src/project/mod.rs:1-25`
- Modify: `RustViewer/src/pipeline/rustsfm_worker.rs:196-218`
- Modify: `RustViewer/tests/project_store.rs`
- Modify: unit tests in `RustViewer/src/pipeline/rustsfm_worker.rs`

- [ ] **Step 1: Add failing legacy-manifest and preset tests**

In `RustViewer/tests/project_store.rs`, serialize a fresh manifest to `serde_json::Value`, remove `sfm_config.keyframe_selection` and `sfm_config.adaptive_keyframes`, deserialize it, and assert:

```rust
assert_eq!(restored.sfm_config.keyframe_selection, KeyframeSelectionMode::Adaptive);
assert_eq!(restored.sfm_config.adaptive_keyframes, AdaptiveKeyframeSelectionConfig::default());
assert!(restored.sfm_config.use_all_images); // legacy field remains readable
```

Add validation cases for NaN/out-of-range `retention_feature_coverage`, zero `min_inliers`, invalid `min_inlier_ratio`, and zero `min_triangulated`; each must yield `ProjectManifestValidationError::InvalidSfmConfig`.

In `rustsfm_worker.rs` tests extend `project_gpu_configuration_enables_all_rustsfm_gpu_paths`:

```rust
assert_eq!(mapper_config.max_features, 4096);
assert_eq!(
    mapper_config.matching_pair_strategy,
    rustsfm::MatchingPairStrategy::LocalWindow { window: 5 },
);
```

- [ ] **Step 2: Run focused RustViewer tests and verify RED**

Run:

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rust-viewer --test project_store
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rust-viewer project_gpu_configuration_enables_all_rustsfm_gpu_paths --lib
```

Expected: compilation fails because the mode/config fields and validation variant do not exist, then the preset assertion fails against 8,192/default sequential matching.

- [ ] **Step 3: Implement serde-compatible mode and validation**

Add to `manifest.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum KeyframeSelectionMode {
    #[default]
    Adaptive,
    AllImages,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SfmConfigSnapshot {
    #[serde(default)]
    pub keyframe_selection: KeyframeSelectionMode,
    #[serde(default)]
    pub adaptive_keyframes: rustsfm::AdaptiveKeyframeSelectionConfig,
    pub use_all_images: bool,
    pub use_gpu_sift: bool,
    pub use_gpu_matching: bool,
}
```

Set new-project defaults to adaptive. Keep `use_all_images` serialized/deserialized for compatibility but do not let it override `keyframe_selection`. Add:

```rust
#[error("invalid SFM config: {detail}")]
InvalidSfmConfig { detail: String },
```

and call `sfm_config.adaptive_keyframes.validate()` during manifest validation before stage work can begin. Re-export `KeyframeSelectionMode` from `project/mod.rs`.

In `mapper_config_for`, add exactly:

```rust
mapper_config.max_features = 4096;
mapper_config.matching_pair_strategy = rustsfm::MatchingPairStrategy::LocalWindow { window: 5 };
```

Do not set wgpu backend environment variables and do not add a Metal/Vulkan branch.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the two Step 2 commands again. Expected: both pass with zero failures.

- [ ] **Step 5: Commit Task 4**

```bash
git add RustViewer/src/project/manifest.rs RustViewer/src/project/mod.rs RustViewer/src/pipeline/rustsfm_worker.rs RustViewer/tests/project_store.rs
git commit -m "feat(rustviewer): default to adaptive keyframes"
```

### Task 5: RustViewer Keyframe Selection And Reusable Stage Artifacts

**Files:**
- Modify: `RustViewer/src/pipeline/rustsfm_worker.rs:14-116,220-288,325-434,436-617`

- [ ] **Step 1: Add failing selection-wiring and artifact tests**

Add serializable private stage payload types:

```rust
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct KeyframeStageResult {
    mode: KeyframeSelectionMode,
    imported_frames: usize,
    selected_keyframe_ids: Vec<u32>,
    registered_keyframes: usize,
    selection_config: Option<rustsfm::AdaptiveKeyframeSelectionConfig>,
    evaluated_pairs: usize,
    diagnostics: Vec<rustsfm::AdaptiveKeyframePairDiagnostic>,
}
```

Create a small private `resolve_keyframe_selection_with` helper whose injected closure has the same arguments/result as `run_adaptive_keyframe_selection`. Test that:

- `Adaptive` invokes the closure once and returns only its selected IDs even when every imported `ImportedFrame.is_keyframe` and legacy `use_all_images` are true;
- `AllImages` returns every imported frame ID and never invokes the closure;
- adaptive results with unknown, duplicate, unordered, or fewer than two selected IDs are rejected.

Add a `keyframe_stage_artifacts` test with a temporary RustSFM output tree. Assert returned relative names are exactly:

```text
keyframe-result.json
rustsfm/database.db
rustsfm/keyframe-sparse/0/cameras.txt
rustsfm/keyframe-sparse/0/images.txt
rustsfm/keyframe-sparse/0/points3D.txt
rustsfm/keyframe-sparse/0/cameras.bin
rustsfm/keyframe-sparse/0/images.bin
rustsfm/keyframe-sparse/0/points3D.bin
```

and that the JSON contains imported count, selected count/IDs, thresholds, evaluated pair count, and pair diagnostics.

- [ ] **Step 2: Run the worker unit tests and verify RED**

Run:

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rust-viewer rustsfm_worker::tests --lib
```

Expected: compilation fails because the stage payload, selection helper, and reusable artifact builder do not exist.

- [ ] **Step 3: Wire adaptive selection into `SfmWorker::run`**

Change `ImportedSequence` to contain only stable ordered frames; `load_imported_sequence` must validate and retain all imported images without deriving keyframes from `is_keyframe` or a stride.

Implement `resolve_keyframe_selection_with` and call it from `SfmWorker::run` with a closure that invokes:

```rust
rustsfm::run_adaptive_keyframe_selection(
    &sequence.frames,
    &request.manifest.sfm_config.adaptive_keyframes,
    &mapper_config,
    &output,
    &mut task,
)
```

For `AllImages`, use every stable ordered frame ID and record zero adaptive diagnostics. Then call `run_keyframe_reconstruction` with exactly the resolved IDs and the same `output`, mapper config, and task context.

Validate that the returned reconstruction IDs equal the resolved IDs. Build `KeyframeStageResult`, read the database/six sparse files into `PendingArtifact`s, and only then remove the temporary RustSFM output. Use `ArtifactValidation::Json` for the result and `ReadableFile` for database/sparse files.

Preserve existing pause/cancel cleanup and retryable `rustsfm_failed` behavior.

- [ ] **Step 4: Run worker tests and verify GREEN**

Run:

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rust-viewer rustsfm_worker::tests --lib
```

Expected: all worker unit tests pass, including proof that legacy all-image flags do not override adaptive mode.

- [ ] **Step 5: Commit Task 5**

```bash
git add RustViewer/src/pipeline/rustsfm_worker.rs
git commit -m "feat(rustviewer): persist adaptive keyframe artifacts"
```

### Task 6: Resume Full-Frame PnP Without Reconstructing Keyframes

**Files:**
- Modify: `RustViewer/src/pipeline/rustsfm_worker.rs:118-193,220-434,436-617`

- [ ] **Step 1: Add failing hydration and PnP-call-boundary tests**

Add unit tests that construct a fixture `StageRequest` whose `KeyframeSfm` record references the eight Task 5 artifacts. Test a private `hydrate_keyframe_result` helper and assert:

```rust
assert_eq!(hydrated.keyframe_ids, stage_result.selected_keyframe_ids);
assert_eq!(hydrated.database, output.join("Cache/database.db"));
assert_eq!(hydrated.sparse_model, output.join("Cache/keyframe-sparse/0"));
assert_eq!(fs::read(&hydrated.database).unwrap(), committed_database_bytes);
```

Add rejection tests for missing/duplicate artifact suffixes, unsafe paths, selected IDs not present in the imported sequence, and imported count mismatch.

Extract a small `run_remaining_registration_with` call boundary. Inject a fake function in the unit test and assert it receives all imported frames, the persisted selected IDs, and the hydrated `KeyframeReconstructionResult`; assert the keyframe reconstruction function is not part of this boundary.

Keep `incomplete_full_frame_registration_never_returns_success` as the final RustGS gate regression test.

- [ ] **Step 2: Run worker tests and verify RED**

Run:

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rust-viewer rustsfm_worker::tests --lib
```

Expected: compilation fails because hydration and remaining-registration helpers do not exist.

- [ ] **Step 3: Hydrate committed artifacts and call only remaining-frame registration**

Implement artifact lookup by unique suffix under the successful `KeyframeSfm` stage. Copy each committed regular file into its exact current-stage location:

```text
<output>/Cache/database.db
<output>/Cache/keyframe-sparse/0/{cameras,images,points3D}.{txt,bin}
```

Deserialize and validate `KeyframeStageResult`, then construct:

```rust
rustsfm::KeyframeReconstructionResult {
    imported_frames: stage_result.imported_frames,
    keyframe_ids: stage_result.selected_keyframe_ids.clone(),
    registered_keyframes: stage_result.registered_keyframes,
    database: output.join("Cache/database.db"),
    sparse_model: output.join("Cache/keyframe-sparse/0"),
}
```

Replace `run_sequence_registration` in `PnpWorker::run` with:

```rust
rustsfm::register_remaining_sequence_frames(
    &sequence.frames,
    &keyframes.keyframe_ids,
    &keyframes,
    &mapper_config,
    &registration_config,
    &output,
    &mut task,
)
```

The successful path must still require `result.has_complete_coverage()`, emit the final COLMAP artifacts, and return `IncompletePoseCoverage` otherwise. RustGS remains downstream of the successful `FullFramePnp` stage and needs no training-worker change.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rust-viewer rustsfm_worker::tests --lib
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rust-viewer --test pipeline_coordinator
```

Expected: worker and coordinator suites pass, with no path that starts training after incomplete PnP coverage.

- [ ] **Step 5: Commit Task 6**

```bash
git add RustViewer/src/pipeline/rustsfm_worker.rs
git commit -m "fix(rustviewer): reuse keyframe reconstruction for PnP"
```

### Task 7: Full Verification, Review, Merge, Cleanup, And Launch

**Files:**
- Review all files changed since `cc4154c`
- No new production files unless review finds a concrete defect

- [ ] **Step 1: Format and inspect the complete diff**

Run:

```bash
cargo fmt --all -- --check
git diff --check cc4154c..HEAD
git status --short
```

Expected: formatting/diff checks exit 0; status contains no uncommitted files before final review. If formatting fails, run `cargo fmt --all`, re-run targeted tests, and commit only the formatting changes.

- [ ] **Step 2: Run targeted RustSFM verification**

Run:

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rustsfm --features gpu-wgpu --test adaptive_keyframes --test task_control --test sequence_registration
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo check -p rustsfm --features gpu-wgpu
```

Expected: all tests pass and `cargo check` exits 0.

- [ ] **Step 3: Run complete RustViewer verification outside the macOS sandbox**

Run with macOS security-scope access:

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rust-viewer
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo check -p rust-viewer
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo build -p rust-viewer --release
```

Expected: the complete RustViewer suite reports zero failures, both checks/builds exit 0, and `/Users/tfjiang/Projects/RustScan/target/release/rust-viewer` exists.

- [ ] **Step 4: Perform final spec and code-quality review**

Review `cc4154c..HEAD` against the approved design. Confirm explicitly:

- there is no fixed selected-frame maximum, stride, ratio, or dataset-length cap;
- first/last boundaries, stable order, uniqueness, determinism, bridge retry, and termination are tested;
- metrics contain matches, inliers, triangulated count, ratio, and coverage;
- RustViewer defaults old manifests to adaptive mode while retaining legacy `use_all_images` readability;
- mapper defaults are 4,096 features and `LocalWindow { window: 5 }`;
- selection uses configured SIFT/matcher wgpu flags without forcing Vulkan or Metal;
- the keyframe database and sparse model cross the stage boundary and PnP does not call `run_sequence_registration`;
- all imported frames still flow to PnP and incomplete coverage blocks RustGS;
- async GPU readback and RustGS hyperparameters remain unchanged.

Fix every Critical/Important review finding with a failing regression test first, re-run affected suites, and commit the fix.

- [ ] **Step 5: Merge into main and verify the merged tree**

The user already selected local merge and cleanup. From `/Users/tfjiang/Projects/RustScan`, verify `main` has no conflicting user changes, switch to `main`, and merge `codex/rustviewer-adaptive-keyframes` without rebasing away the task commits. Then run:

```bash
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo test -p rust-viewer
CARGO_TARGET_DIR=/Users/tfjiang/Projects/RustScan/target cargo build -p rust-viewer --release
```

Expected: merged-tree tests and release build exit 0.

- [ ] **Step 6: Clean up the owned worktree and feature branch**

From `/Users/tfjiang/Projects/RustScan` after the merge and merged-tree verification succeed:

```bash
git worktree remove /Users/tfjiang/Projects/RustScan/.worktrees/rustviewer-adaptive-keyframes
git worktree prune
git branch -d codex/rustviewer-adaptive-keyframes
```

Expected: `git worktree list` no longer contains the adaptive worktree and `git branch --list codex/rustviewer-adaptive-keyframes` prints nothing.

- [ ] **Step 7: Launch the verified release RustViewer**

Run from `/Users/tfjiang/Projects/RustScan`:

```bash
target/release/rust-viewer
```

Expected: RustViewer remains running for the user to import a dataset and test the adaptive RustSFM -> full-frame PnP -> RustGS flow.
