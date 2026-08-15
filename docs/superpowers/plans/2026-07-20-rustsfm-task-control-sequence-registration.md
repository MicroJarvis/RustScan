# RustSFM Task Control and Sequence Registration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose RustSFM as a controllable in-process task API that reconstructs selected keyframes, registers every remaining sequence frame, emits typed progress, and validates complete pose coverage.

**Architecture:** Add an additive task contract beside the existing mapper callbacks, then thread it through extraction, matching, mapping, BA boundaries, and export. Build sequence registration as a public orchestration module that reuses the existing feature database, reference-model seeding, incremental mapper, and wgpu PnP scorer while recording deterministic per-frame attempts and diagnostics.

**Tech Stack:** Rust 2021, `anyhow`, `serde`, `rusqlite`, existing RustSFM SIFT/matching/mapper/COLMAP modules, optional wgpu SIFT and PnP scoring.

---

## File Map

- Create `RustSFM/src/task.rs`: public stage, operation, event, control, stop, warning, and sink types.
- Modify `RustSFM/src/lib.rs`: export the task API and new controlled entry points.
- Modify `RustSFM/src/feature/feature_extraction.rs`: add image-boundary progress and cooperative control.
- Modify `RustSFM/src/feature/feature_matching_db.rs`: add pair-batch progress and cooperative control.
- Modify `RustSFM/src/sfm/mapper/pipeline_types.rs`: adapt legacy mapper callbacks into task events.
- Modify `RustSFM/src/sfm/mapper.rs`: add the controlled reconstruction entry point and safe mapper/BA/export checkpoints.
- Create `RustSFM/src/sequence_registration.rs`: keyframe reconstruction, temporal retry rounds, diagnostics, pose coverage, and validated export.
- Create `RustSFM/tests/task_control.rs`: public contract and cancellation tests.
- Create `RustSFM/tests/sequence_registration.rs`: deterministic sequence planning, coverage gate, and small fixture tests.

### Task 1: Public Task Event and Cooperative Control Contract

**Files:**
- Create: `RustSFM/src/task.rs`
- Modify: `RustSFM/src/lib.rs`
- Test: `RustSFM/tests/task_control.rs`

- [ ] **Step 1: Write failing public-contract tests**

Create `RustSFM/tests/task_control.rs`:

```rust
use rustsfm::{
    SfmControlState, SfmTaskControl, SfmTaskEvent, SfmTaskEventKind, SfmTaskOperation,
    SfmTaskStage, SfmTaskStop,
};

#[test]
fn control_prioritizes_cancel_over_pause() {
    let control = SfmTaskControl::new();
    control.request_pause();
    assert_eq!(control.checkpoint(), Err(SfmTaskStop::Paused));
    control.request_cancel();
    assert_eq!(control.checkpoint(), Err(SfmTaskStop::Cancelled));
    assert_eq!(control.state(), SfmControlState::CancelRequested);
}

#[test]
fn event_progress_is_machine_readable() {
    let event = SfmTaskEvent {
        sequence: 7,
        elapsed_ms: 42,
        stage: SfmTaskStage::FeatureExtraction,
        operation: SfmTaskOperation::ExtractImage,
        kind: SfmTaskEventKind::Progress,
        completed: Some(3),
        total: Some(10),
        registered_images: None,
        sparse_points: None,
        image_id: Some(12),
        pair: None,
        message: None,
        issue: None,
    };
    let json = serde_json::to_value(event).unwrap();
    assert_eq!(json["sequence"], 7);
    assert_eq!(json["stage"], "feature_extraction");
    assert_eq!(json["operation"], "extract_image");
}
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
cargo test -p rustsfm --test task_control
```

Expected: FAIL because the task contract is not exported.

- [ ] **Step 3: Implement the task contract**

Create `RustSFM/src/task.rs` with these public shapes and implementations:

```rust
use serde::{Deserialize, Serialize};
use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc,
};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SfmTaskStage {
    FeatureExtraction,
    FeatureMatching,
    IncrementalMapping,
    BundleAdjustment,
    FullFrameRegistration,
    Export,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SfmTaskOperation {
    Begin,
    ExtractImage,
    MatchPairBatch,
    RegisterInitialPair,
    RegisterImage,
    LocalBundleAdjustment,
    GlobalBundleAdjustment,
    RegisterFrameAttempt,
    ValidateArtifacts,
    WriteArtifacts,
    Complete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SfmTaskEventKind {
    Started,
    Progress,
    Warning,
    Error,
    Completed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SfmTaskIssue {
    pub code: String,
    pub summary: String,
    pub detail: String,
    pub retryable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SfmTaskEvent {
    pub sequence: u64,
    pub elapsed_ms: u64,
    pub stage: SfmTaskStage,
    pub operation: SfmTaskOperation,
    pub kind: SfmTaskEventKind,
    pub completed: Option<usize>,
    pub total: Option<usize>,
    pub registered_images: Option<usize>,
    pub sparse_points: Option<usize>,
    pub image_id: Option<u32>,
    pub pair: Option<(u32, u32)>,
    pub message: Option<String>,
    pub issue: Option<SfmTaskIssue>,
}

pub trait SfmTaskEventSink {
    fn on_sfm_event(&mut self, event: SfmTaskEvent);
}

impl<F> SfmTaskEventSink for F
where
    F: FnMut(SfmTaskEvent),
{
    fn on_sfm_event(&mut self, event: SfmTaskEvent) {
        self(event);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SfmControlState {
    Running = 0,
    PauseRequested = 1,
    CancelRequested = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SfmTaskStop {
    #[error("SFM task paused at a safe boundary")]
    Paused,
    #[error("SFM task cancelled at a safe boundary")]
    Cancelled,
}

#[derive(Debug, Clone, Default)]
pub struct SfmTaskControl {
    state: Arc<AtomicU8>,
}

impl SfmTaskControl {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn request_pause(&self) {
        let _ = self.state.compare_exchange(
            SfmControlState::Running as u8,
            SfmControlState::PauseRequested as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    pub fn request_cancel(&self) {
        self.state
            .store(SfmControlState::CancelRequested as u8, Ordering::Release);
    }

    pub fn state(&self) -> SfmControlState {
        match self.state.load(Ordering::Acquire) {
            1 => SfmControlState::PauseRequested,
            2 => SfmControlState::CancelRequested,
            _ => SfmControlState::Running,
        }
    }

    pub fn checkpoint(&self) -> Result<(), SfmTaskStop> {
        match self.state() {
            SfmControlState::Running => Ok(()),
            SfmControlState::PauseRequested => Err(SfmTaskStop::Paused),
            SfmControlState::CancelRequested => Err(SfmTaskStop::Cancelled),
        }
    }
}

pub struct SfmTaskContext<'a> {
    pub control: &'a SfmTaskControl,
    sink: &'a mut dyn SfmTaskEventSink,
    started: Instant,
    next_sequence: u64,
}

impl<'a> SfmTaskContext<'a> {
    pub fn new(control: &'a SfmTaskControl, sink: &'a mut dyn SfmTaskEventSink) -> Self {
        Self { control, sink, started: Instant::now(), next_sequence: 0 }
    }

    pub fn emit(&mut self, mut event: SfmTaskEvent) {
        event.sequence = self.next_sequence;
        event.elapsed_ms = self.started.elapsed().as_millis() as u64;
        self.next_sequence = self.next_sequence.saturating_add(1);
        self.sink.on_sfm_event(event);
    }

    pub fn checkpoint(&self) -> Result<(), SfmTaskStop> {
        self.control.checkpoint()
    }
}
```

Add `pub mod task;` and re-export all public task types from `RustSFM/src/lib.rs`.

- [ ] **Step 4: Run the public-contract tests and verify GREEN**

Run:

```bash
cargo test -p rustsfm --test task_control
```

Expected: PASS with 2 tests.

- [ ] **Step 5: Commit Task 1**

```bash
git add RustSFM/src/task.rs RustSFM/src/lib.rs RustSFM/tests/task_control.rs
git commit -m "feat(rustsfm): add typed task control contract"
```

### Task 2: Controlled Feature Extraction

**Files:**
- Modify: `RustSFM/src/feature/feature_extraction.rs`
- Modify: `RustSFM/src/lib.rs`
- Test: `RustSFM/src/feature/feature_extraction.rs`

- [ ] **Step 1: Write a failing boundary-control test**

In the existing feature extraction test module, add a two-image database fixture and a fake extractor. The event sink requests pause after the first progress event:

```rust
#[test]
fn controlled_extraction_pauses_after_committed_image() -> Result<()> {
    let fixture = extraction_fixture(&["000.png", "001.png"])?;
    let control = SfmTaskControl::new();
    let mut events = Vec::new();
    let pause = control.clone();
    let mut sink = |event: SfmTaskEvent| {
        if event.operation == SfmTaskOperation::ExtractImage && event.completed == Some(1) {
            pause.request_pause();
        }
        events.push(event);
    };
    let mut task = SfmTaskContext::new(&control, &mut sink);

    let error = extract_features_to_database_with_extractor_and_task(
        &fixture.database,
        &fixture.images,
        &SiftExtractionOptions::default(),
        &FakeExtractor::default(),
        &mut task,
    )
    .unwrap_err();

    assert!(error.downcast_ref::<SfmTaskStop>().is_some());
    assert_eq!(fixture.database_handle().read_keypoint_counts()?.len(), 1);
    assert_eq!(events.last().unwrap().completed, Some(1));
    Ok(())
}
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
cargo test -p rustsfm --lib controlled_extraction_pauses_after_committed_image
```

Expected: FAIL because the controlled extraction function does not exist.

- [ ] **Step 3: Add the controlled entry point without changing the old API**

Keep `extract_features_to_database_with_extractor` as a wrapper. Move its loop into a new public function with `task: &mut SfmTaskContext<'_>`. Call `task.checkpoint()?` before decode and after both database upserts, then emit:

```rust
task.emit(SfmTaskEvent {
    sequence: 0,
    elapsed_ms: 0,
    stage: SfmTaskStage::FeatureExtraction,
    operation: SfmTaskOperation::ExtractImage,
    kind: SfmTaskEventKind::Progress,
    completed: Some(reports.len()),
    total: Some(image_count),
    registered_images: None,
    sparse_points: None,
    image_id: Some(image.image_id),
    pair: None,
    message: Some(image.name.clone()),
    issue: None,
});
task.checkpoint()?;
```

For compatibility, introduce a private `NoopSfmTaskEventSink` and a running control in the old wrapper. Re-export `extract_features_to_database_with_extractor_and_task` from `lib.rs`.

- [ ] **Step 4: Run extraction and public API tests**

Run:

```bash
cargo test -p rustsfm --lib feature_extraction
cargo test -p rustsfm --test task_control
```

Expected: PASS; a pause leaves exactly one complete image feature record and never a half-written keypoint/descriptor pair.

- [ ] **Step 5: Commit Task 2**

```bash
git add RustSFM/src/feature/feature_extraction.rs RustSFM/src/lib.rs
git commit -m "feat(rustsfm): report and control feature extraction"
```

### Task 3: Controlled Matching and Geometric Verification

**Files:**
- Modify: `RustSFM/src/feature/feature_matching_db.rs`
- Modify: `RustSFM/src/lib.rs`
- Test: `RustSFM/src/feature/feature_matching_db.rs`

- [ ] **Step 1: Write failing deterministic batch tests**

Add tests that set `task_pair_batch_size` to two, feed five valid pairs through the existing synthetic database fixture, and assert progress totals `[2, 4, 5]`. Add a second sink that requests cancel at `completed == 2` and assert only the first two pair rows are committed.

```rust
assert_eq!(
    progress.iter().map(|event| event.completed).collect::<Vec<_>>(),
    vec![Some(2), Some(4), Some(5)]
);
assert_eq!(cancel_error.downcast_ref::<SfmTaskStop>(), Some(&SfmTaskStop::Cancelled));
assert_eq!(read_committed_pair_count(&database)?, 2);
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
cargo test -p rustsfm --lib controlled_matching_
```

Expected: FAIL because matching has no task context or bounded pair batches.

- [ ] **Step 3: Extend options and implement batch checkpoints**

Add this field to `MatchFeaturesOptions` and its default:

```rust
pub task_pair_batch_size: usize,
// Default value:
task_pair_batch_size: 32,
```

Add `match_features_to_database_with_task(database_path, options, task)` and preserve the existing function as a no-op-task wrapper. Split `computed_match_pair_reports` into `matching_pair_indices(frames, options)`, `compute_pair_report_batch(frames, cameras, options, pairs)`, and `write_pair_report_batch(db, frames, image_id_by_index, options, reports)`. The existing-match route receives the equivalent `existing_match_inputs` and `verify_existing_match_batch` split. Iterate deterministic pair indices in chunks, commit one database transaction per chunk, and emit once per committed chunk:

```rust
for batch in pairs.chunks(options.task_pair_batch_size.max(1)) {
    task.checkpoint()?;
    let batch_reports = compute_pair_report_batch(frames, cameras, options, batch)?;
    let written = db.with_transaction(|| {
        write_pair_report_batch(
            &db,
            frames,
            &image_id_by_index,
            options,
            batch_reports,
        )
    })?;
    completed += batch.len();
    total_matches += written.total_matches;
    reports.extend(written.reports);
    let last_pair = batch.last().map(|&(left, right)| {
        (image_id_by_index[left].1, image_id_by_index[right].1)
    });
    task.emit(SfmTaskEvent {
        sequence: 0,
        elapsed_ms: 0,
        stage: SfmTaskStage::FeatureMatching,
        operation: SfmTaskOperation::MatchPairBatch,
        kind: SfmTaskEventKind::Progress,
        completed: Some(completed),
        total: Some(pairs.len()),
        registered_images: None,
        sparse_points: None,
        image_id: None,
        pair: last_pair,
        message: None,
        issue: None,
    });
    task.checkpoint()?;
}
```

- [ ] **Step 4: Run matching regression tests**

Run:

```bash
cargo test -p rustsfm --lib feature_matching_db
cargo test -p rustsfm --lib feature_matching
```

Expected: PASS, including exact geometric-verification parity from the existing tests.

- [ ] **Step 5: Commit Task 3**

```bash
git add RustSFM/src/feature/feature_matching_db.rs RustSFM/src/lib.rs
git commit -m "feat(rustsfm): control bounded matching batches"
```

### Task 4: Controlled Mapper, BA Boundaries, and Legacy Callback Adapter

**Files:**
- Modify: `RustSFM/src/sfm/mapper/pipeline_types.rs`
- Modify: `RustSFM/src/sfm/mapper.rs`
- Modify: `RustSFM/src/lib.rs`
- Test: `RustSFM/src/sfm/mapper.rs`
- Test: `RustSFM/tests/task_control.rs`

- [ ] **Step 1: Write failing callback adaptation and mapper-stop tests**

Test that `InitialImagePairReg`, `NextImageReg`, and `LastImageReg` become monotonically sequenced `IncrementalMapping` events with registered-image and point counts. In the mapper synthetic-seed fixture, request pause from the first `RegisterImage` event and assert the returned error is `SfmTaskStop::Paused` while the last valid reconstruction snapshot remains exportable.

```rust
assert_eq!(events[0].operation, SfmTaskOperation::RegisterInitialPair);
assert!(events.windows(2).all(|pair| pair[0].sequence < pair[1].sequence));
assert_eq!(events.last().unwrap().registered_images, Some(3));
assert!(error.downcast_ref::<SfmTaskStop>().is_some());
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
cargo test -p rustsfm --lib task_callback_adapter
cargo test -p rustsfm --lib controlled_mapper_pauses_
```

Expected: FAIL because no task adapter or controlled mapper entry point exists.

- [ ] **Step 3: Add the controlled mapper entry point**

Add:

```rust
pub fn run_reconstruction_with_task(
    config: &MapperConfig,
    task: &mut SfmTaskContext<'_>,
) -> Result<ReconstructionSummary> {
    let mut events = MapperEventBridge::Task(task);
    run_reconstruction_impl(config, &mut events)
}
```

Refactor the private implementation to accept `&mut MapperEventBridge<'_>`. `run_reconstruction` creates `MapperEventBridge::Silent`; `run_reconstruction_with_callbacks` creates `MapperEventBridge::Legacy(callback_sink)`; and the new API creates `MapperEventBridge::Task(task)`. This keeps one mutable owner and prevents aliasing. The bridge forwards legacy events unchanged and maps task callbacks as follows:

```rust
let operation = match event.callback {
    IncrementalPipelineCallback::InitialImagePairReg => SfmTaskOperation::RegisterInitialPair,
    IncrementalPipelineCallback::NextImageReg | IncrementalPipelineCallback::LastImageReg => {
        SfmTaskOperation::RegisterImage
    }
};
```

`MapperEventBridge` implements `callback(PipelineCallbackEvent)`, `emit_operation(stage: SfmTaskStage, operation: SfmTaskOperation, kind: SfmTaskEventKind)`, and `checkpoint()`. Silent and Legacy variants return `Ok(())` from `checkpoint`; the Task variant delegates to `SfmTaskContext`. Do not retain the old `Option<&mut dyn PipelineCallbackSink>` parameter after this refactor.

- [ ] **Step 4: Insert safe checkpoints around atomic mapper work**

At each registration attempt, call `checkpoint()` before selecting/correspondence building and after the attempt has either committed or rolled back. Around local/global BA use this exact ordering:

```rust
events.checkpoint()?;
events.emit_operation(SfmTaskStage::BundleAdjustment, operation, SfmTaskEventKind::Started);
let local_ba_report = refine_local_bundle_after_registration(
    frames,
    pairs,
    &mut reconstruction,
    choice.image,
    gauge_image,
    &tri_options,
    config,
    &local_registration_stats,
    &mut triangulation_state,
);
events.emit_operation(SfmTaskStage::BundleAdjustment, operation, SfmTaskEventKind::Completed);
events.checkpoint()?;
```

Use `SfmTaskOperation::LocalBundleAdjustment` around `refine_local_bundle_after_registration`, and `GlobalBundleAdjustment` around the scheduled and final calls to `refine_global_bundle_with_postprocessing`. Do not attempt to serialize a half-complete Ceres solve. A pause or cancel arriving during Ceres is observed immediately after the solver returns and before the next mutation phase. Add the same before/after boundary around sparse export and validation.

- [ ] **Step 5: Run mapper regression and task tests**

Run:

```bash
cargo test -p rustsfm --lib mapper::tests
cargo test -p rustsfm --test task_control
```

Expected: PASS; existing callback order remains unchanged and controlled execution stops only at valid reconstruction boundaries.

- [ ] **Step 6: Commit Task 4**

```bash
git add RustSFM/src/sfm/mapper/pipeline_types.rs RustSFM/src/sfm/mapper.rs RustSFM/src/lib.rs RustSFM/tests/task_control.rs
git commit -m "feat(rustsfm): expose controlled mapper progress"
```

### Task 5: Deterministic Full-Sequence Registration Plan and Diagnostics

**Files:**
- Create: `RustSFM/src/sequence_registration.rs`
- Modify: `RustSFM/src/lib.rs`
- Test: `RustSFM/tests/sequence_registration.rs`

- [ ] **Step 1: Write failing temporal-neighborhood tests**

Create tests for a 12-frame sequence with keyframes `[0, 3, 6, 9, 11]`. Require the narrow round to use the two nearest keyframes on each side, the wide round to use four, and successful non-keyframe neighbors to become support images only in later deterministic rounds.

```rust
let plan = SequenceRegistrationPlan::build(12, &[0, 3, 6, 9, 11], 2, 4).unwrap();
assert_eq!(plan.attempts_for(4, RegistrationRound::Narrow), &[3, 6, 0, 9]);
assert_eq!(plan.attempts_for(4, RegistrationRound::Wide), &[3, 6, 0, 9, 11]);
assert_eq!(plan.pending_frames(), &[1, 2, 4, 5, 7, 8, 10]);
```

Add JSON round-trip coverage for:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegistrationRound { Narrow, Wide }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrameRegistrationStatus { Keyframe, Registered, Unresolved, Excluded }

impl FrameRegistrationStatus {
    pub fn is_registered(self) -> bool {
        matches!(self, Self::Keyframe | Self::Registered)
    }
}

pub struct FrameRegistrationDiagnostic {
    pub frame_id: u32,
    pub status: FrameRegistrationStatus,
    pub attempts: usize,
    pub support_frame_ids: Vec<u32>,
    pub inlier_count: usize,
    pub inlier_ratio: f64,
    pub mean_reprojection_error: Option<f64>,
    pub message: Option<String>,
}
```

- [ ] **Step 2: Run sequence tests and verify RED**

Run:

```bash
cargo test -p rustsfm --test sequence_registration
```

Expected: FAIL because the sequence registration module is absent.

- [ ] **Step 3: Implement public configuration, plan, and result types**

Define:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SequenceFrame {
    pub id: u32,
    pub image_path: std::path::PathBuf,
    pub timestamp_us: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceRegistrationConfig {
    pub narrow_neighbors_each_side: usize,
    pub wide_neighbors_each_side: usize,
    pub min_inliers: usize,
    pub min_inlier_ratio: f64,
    pub max_reprojection_error: f64,
    pub use_gpu_pnp: bool,
}

impl Default for SequenceRegistrationConfig {
    fn default() -> Self {
        Self {
            narrow_neighbors_each_side: 2,
            wide_neighbors_each_side: 4,
            min_inliers: 24,
            min_inlier_ratio: 0.20,
            max_reprojection_error: 4.0,
            use_gpu_pnp: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceRegistrationResult {
    pub imported_frames: usize,
    pub registered_frames: usize,
    pub diagnostics: Vec<FrameRegistrationDiagnostic>,
    pub sparse_model: std::path::PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceRegistrationPlan {
    frame_count: usize,
    keyframes: Vec<usize>,
    pending: Vec<usize>,
    narrow_support: Vec<Vec<usize>>,
    wide_support: Vec<Vec<usize>>,
}

impl SequenceRegistrationResult {
    pub fn has_complete_coverage(&self) -> bool {
        self.imported_frames == self.registered_frames
            && self.diagnostics.iter().all(|item| item.status.is_registered())
    }
}
```

Implement neighborhood ordering by `(absolute timestamp/frame distance, frame_id)` and validate unique, in-range keyframes. Export these types from `lib.rs`.

- [ ] **Step 4: Run deterministic planning tests**

Run:

```bash
cargo test -p rustsfm --test sequence_registration temporal_
cargo test -p rustsfm --test sequence_registration diagnostic_
```

Expected: PASS for deterministic planning and serde round trips.

- [ ] **Step 5: Commit Task 5**

```bash
git add RustSFM/src/sequence_registration.rs RustSFM/src/lib.rs RustSFM/tests/sequence_registration.rs
git commit -m "feat(rustsfm): define full sequence registration"
```

### Task 6: Keyframe Reconstruction and Full-Frame PnP Runner

**Files:**
- Modify: `RustSFM/src/sequence_registration.rs`
- Modify: `RustSFM/src/sfm/mapper.rs`
- Test: `RustSFM/tests/sequence_registration.rs`

- [ ] **Step 1: Write a failing complete-coverage integration test**

Use the existing synthetic camera/image helpers to generate six overlapping images, select frames `0, 2, 4, 5` as keyframes, and run the sequence API into a temporary output root. Require all six poses and diagnostic records:

```rust
let result = run_sequence_registration(
    &fixture.frames,
    &[0, 2, 4, 5],
    &fixture.mapper_config,
    &SequenceRegistrationConfig { use_gpu_pnp: false, ..Default::default() },
    temp.path(),
    &mut task,
)?;
assert!(result.has_complete_coverage());
assert_eq!(result.registered_frames, 6);
assert_eq!(result.diagnostics.len(), 6);
assert!(temp.path().join("sparse/0/images.bin").is_file());
```

- [ ] **Step 2: Run the integration test and verify RED**

Run:

```bash
cargo test -p rustsfm --test sequence_registration complete_sequence_
```

Expected: FAIL because the runner is not implemented.

- [ ] **Step 3: Implement keyframe reconstruction and database reuse**

Add this stage result and entry point:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyframeReconstructionResult {
    pub imported_frames: usize,
    pub keyframe_ids: Vec<u32>,
    pub registered_keyframes: usize,
    pub database: PathBuf,
    pub sparse_model: PathBuf,
}
```

Implement `run_keyframe_reconstruction(frames: &[SequenceFrame], keyframe_ids: &[u32], mapper_config: &MapperConfig, output: &Path, task: &mut SfmTaskContext<'_>) -> Result<KeyframeReconstructionResult>` in four committed phases:

1. Create a keyframe-only input directory using hard links with copy fallback, keeping original stable filenames.
2. Run `run_reconstruction_with_task` using the caller's mapper settings and a database at `output/Cache/database.db`.
3. Validate the keyframe model has finite poses, at least two registered images, and non-empty finite points.
4. Preserve the database and sparse output before beginning full-frame registration.

The validation predicate must be explicit:

```rust
fn validate_sparse_reconstruction(model: &Reconstruction) -> Result<()> {
    if model.points.is_empty() {
        anyhow::bail!("keyframe reconstruction contains no sparse points");
    }
    if model.poses.iter().filter(|pose| pose.is_some()).count() < 2 {
        anyhow::bail!("keyframe reconstruction contains fewer than two registered images");
    }
    if model.points.iter().any(|point| point.xyz.iter().any(|v| !v.is_finite())) {
        anyhow::bail!("keyframe reconstruction contains non-finite points");
    }
    Ok(())
}
```

- [ ] **Step 4: Implement narrow and wide registration rounds**

Add `register_remaining_sequence_frames(frames, keyframes, keyframe_result, config, output, task) -> Result<SequenceRegistrationResult>`. It starts from the already committed keyframe sparse model and shared feature database; it must not rerun extraction, matching, mapping, or BA for keyframes. For each pending frame in timestamp order:

1. Ensure features exist in the shared database through the controlled extraction API.
2. Insert only the planned temporal pairs for the current round and run the controlled matcher.
3. Seed the current sparse model through `MapperConfig.reference`, set `fix_existing_frames = true`, `use_gpu_pnp` from `SequenceRegistrationConfig`, and constrain the registration attempt to the target plus support images.
4. Accept only a finite pose meeting the explicit inlier, ratio, and reprojection thresholds.
5. Commit an accepted pose and its observations before allowing that frame to support later rounds.

Use this acceptance gate:

```rust
fn accepts_registration(
    diagnostic: &FrameRegistrationDiagnostic,
    config: &SequenceRegistrationConfig,
) -> bool {
    diagnostic.inlier_count >= config.min_inliers
        && diagnostic.inlier_ratio >= config.min_inlier_ratio
        && diagnostic
            .mean_reprojection_error
            .is_some_and(|error| error.is_finite() && error <= config.max_reprojection_error)
}
```

Run `Narrow`, then `Wide`, and stop after the bounded two rounds. Emit one `RegisterFrameAttempt` event per attempt. A pause/cancel check occurs before each attempt and after an accepted model has been exported to the temporary round directory.

- [ ] **Step 5: Write diagnostics and enforce the coverage gate**

Serialize `registration.json` atomically via sibling `registration.json.tmp`, `File::sync_all`, and `rename`. Export the merged sparse model only after it validates. Return a normal `SequenceRegistrationResult` with failed diagnostics when coverage is incomplete; provide this strict helper for RustViewer:

```rust
pub fn require_complete_pose_coverage(result: &SequenceRegistrationResult) -> Result<()> {
    if result.has_complete_coverage() {
        Ok(())
    } else {
        let failed = result.imported_frames.saturating_sub(result.registered_frames);
        anyhow::bail!("{failed} frames could not be registered")
    }
}
```

Finally implement `run_sequence_registration` as the convenience composition of `run_keyframe_reconstruction` followed by `register_remaining_sequence_frames`. RustViewer uses the two stage-specific functions so pausing between Keyframe SFM and Full-frame PnP never repeats successful keyframe work.

- [ ] **Step 6: Run sequence and mapper regression tests**

Run:

```bash
cargo test -p rustsfm --test sequence_registration
cargo test -p rustsfm --lib mapper::tests
```

Expected: PASS; the small fixture produces all poses, while a deliberately blank frame returns `has_complete_coverage() == false` and remains explicitly unresolved.

- [ ] **Step 7: Commit Task 6**

```bash
git add RustSFM/src/sequence_registration.rs RustSFM/src/sfm/mapper.rs RustSFM/tests/sequence_registration.rs
git commit -m "feat(rustsfm): register complete image sequences"
```

### Task 7: Compatibility, Formatting, and Acceptance

**Files:**
- Modify: `RustSFM/src/lib.rs`
- Modify: `RustSFM/src/cli/commands.rs`
- Test: `RustSFM/tests/task_control.rs`
- Test: `RustSFM/tests/sequence_registration.rs`

- [ ] **Step 1: Keep CLI behavior on the additive API**

Route existing CLI reconstruction through the unchanged `run_reconstruction` wrapper. Add no mandatory flags and no changed defaults. Add a compile-time public API test that imports both legacy and new entry points.

- [ ] **Step 2: Run formatting and the complete RustSFM suite**

Run:

```bash
cargo fmt --all -- --check
cargo test -p rustsfm --all-features
cargo clippy -p rustsfm --all-targets --all-features -- -D warnings -A dead-code
```

Expected: all commands exit 0.

- [ ] **Step 3: Run a 96-frame flowers2 preflight**

Create the fixture first if absent, using the first 96 lexicographically sorted images without interval sampling:

```zsh
mkdir -p test_data/flowers2/preflight_96/images
images=(test_data/flowers2/images/*.(jpg|jpeg|png|JPG|JPEG|PNG)(N))
for source in ${images[1,96]}; do
  ln "$source" test_data/flowers2/preflight_96/images/${source:t}
done
```

Run the existing COLMAP-compatible RustSFM commands with wgpu SIFT, matching, and PnP, then load the generated model through RustGS:

```bash
cargo run --release -p rustsfm --bin rustsfm -- feature_extractor \
  --database_path test_data/flowers2/preflight_96/database.db \
  --image_path test_data/flowers2/preflight_96/images \
  --SiftExtraction.use_gpu 1
cargo run --release -p rustsfm --bin rustsfm -- sequential_matcher \
  --database_path test_data/flowers2/preflight_96/database.db \
  --SiftMatching.use_gpu 1
cargo run --release -p rustsfm --bin rustsfm -- mapper \
  --database_path test_data/flowers2/preflight_96/database.db \
  --image_path test_data/flowers2/preflight_96/images \
  --output_path test_data/flowers2/preflight_96/rustsfm_controlled \
  --Mapper.use_gpu_pnp 1 \
  --summary-json test_data/flowers2/preflight_96/rustsfm-summary.json
cargo run --release -p rustgs -- train \
  --input test_data/flowers2/preflight_96/rustsfm_controlled \
  --image-root test_data/flowers2/preflight_96/images \
  --output test_data/flowers2/preflight_96/rustgs-probe.ply \
  --iterations 1 \
  --max-initial-gaussians 1000
```

Expected: `rustsfm-summary.json` reports 96 registered images, RustGS loads the model and writes a finite one-iteration probe PLY, and no non-finite camera or point is reported.

- [ ] **Step 4: Commit Task 7**

```bash
git add RustSFM/src/lib.rs RustSFM/src/cli/commands.rs RustSFM/tests
git commit -m "test(rustsfm): verify controlled sequence workflow"
```
