# RustViewer RustSFM Project Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct an imported image-sequence project through RustSFM, persist its verified COLMAP artifacts, and only then enable the existing RustGS training path.

**Architecture:** The pipeline coordinator passes both the locked package root and the private stage workspace to workers. A concrete RustSFM worker reads only committed import metadata, writes all transient output into the provided workspace, reports typed task progress, and returns declared artifacts for the coordinator to validate and commit. `ViewerApp` owns a coordinator only through the SFM and full-frame-pose stages, refreshes its workbench summary from durable manifests, and loads the committed final COLMAP directory through the existing `load_colmap_training_dataset` path.

**Tech Stack:** Rust 2021, RustViewer project store and pipeline coordinator, RustSFM task API, RustGS COLMAP loader, eframe/egui.

---

### Task 1: Pass immutable package and workspace paths to pipeline workers

**Files:**
- Modify: `RustViewer/src/pipeline/worker.rs`
- Modify: `RustViewer/src/pipeline/coordinator.rs`
- Test: `RustViewer/tests/pipeline_coordinator.rs`

- [ ] **Step 1: Write a failing request-contract test**

```rust
#[test]
fn worker_request_includes_the_locked_project_root_and_stage_workspace() {
    let request = captured_request_after_starting(ProjectStage::KeyframeSfm);
    assert!(request.project_root.ends_with("fixture.rustscanproject"));
    assert!(request.workspace_path.starts_with(&request.project_root));
}
```

- [ ] **Step 2: Run the test**

Run: `cargo test -p rust-viewer --test pipeline_coordinator worker_request_includes_the_locked_project_root_and_stage_workspace`

Expected: FAIL because `StageRequest` has no `project_root` or `workspace_path`.

- [ ] **Step 3: Add the two paths and populate them at worker launch**

```rust
pub struct StageRequest {
    pub stage: ProjectStage,
    pub attempt: u32,
    pub project_root: PathBuf,
    pub workspace_path: PathBuf,
    pub manifest: ProjectManifest,
}

let request = StageRequest {
    stage,
    attempt: workspace.attempt(),
    project_root: self.store.root().to_path_buf(),
    workspace_path: workspace.path().to_path_buf(),
    manifest: self.store.manifest().clone(),
};
```

The worker must write only below `workspace_path`; `project_root` is read-only input.

- [ ] **Step 4: Run the contract test and coordinator suite**

Run: `cargo test -p rust-viewer --test pipeline_coordinator`

Expected: PASS.

### Task 2: Add a RustSFM worker with durable, declared artifacts

**Files:**
- Modify: `RustViewer/Cargo.toml`
- Create: `RustViewer/src/pipeline/rustsfm_worker.rs`
- Modify: `RustViewer/src/pipeline/mod.rs`
- Test: `RustViewer/src/pipeline/rustsfm_worker.rs`

- [ ] **Step 1: Write failing metadata and outcome tests**

```rust
#[test]
fn imported_frames_resolve_only_inside_the_committed_project_package() {
    let frames = load_imported_frames(&fixture_request()).unwrap();
    assert_eq!(frames.len(), 2);
    assert!(frames.iter().all(|frame| frame.image_path.is_file()));
}

#[test]
fn incomplete_full_frame_registration_never_returns_success() {
    let outcome = outcome_for_registration(2, 1, Vec::new());
    assert!(matches!(outcome, WorkerOutcome::Failed(_)));
}
```

- [ ] **Step 2: Run the targeted tests**

Run: `cargo test -p rust-viewer pipeline::rustsfm_worker::tests`

Expected: FAIL because the module and helpers do not exist.

- [ ] **Step 3: Implement the worker around RustSFM's public task API**

```rust
pub struct RustSfmWorker;

impl SfmWorker for RustSfmWorker { /* run_keyframe_reconstruction */ }
impl PnpWorker for RustSfmWorker { /* run_sequence_registration and coverage check */ }

let frames: Vec<rustsfm::SequenceFrame> = imported_frames
    .into_iter()
    .map(|frame| rustsfm::SequenceFrame {
        id: frame.id,
        image_path: request.project_root.join(frame.normalized_image),
        timestamp_us: frame.presentation_time_us,
    })
    .collect();
```

Use the committed `Cache/frames.json` import artifact named by the manifest. Reject absolute paths, parent components, missing files, duplicate IDs, and an empty keyframe selection. Forward every `SfmTaskEvent` through `WorkerEventSink::progress` as `PipelineProgressDetail::Sfm`.

Run RustSFM inside `request.workspace_path`. Copy only the final COLMAP sparse files, the normalized images required by the loader, and a JSON result summary into declared `PendingArtifact`s. For the full-frame stage, return `WorkerOutcome::Failed` unless `registered_frames == imported_frames`; do not manufacture a success result.

- [ ] **Step 4: Run targeted tests**

Run: `cargo test -p rust-viewer pipeline::rustsfm_worker::tests && cargo check -p rust-viewer`

Expected: PASS.

### Task 3: Drive only reconstruction stages from ViewerApp and load the committed COLMAP result

**Files:**
- Modify: `RustViewer/src/app.rs`
- Modify: `RustViewer/src/pipeline/events.rs`
- Test: `RustViewer/src/app.rs`

- [ ] **Step 1: Write a failing durable-result routing test**

```rust
#[test]
fn committed_full_frame_result_is_loaded_before_training_is_enabled() {
    let snapshot = snapshot_after_project_colmap_result(&fixture_colmap_root()).unwrap();
    assert!(snapshot.has_dataset);
    assert!(snapshot.primary_command().enabled);
}
```

- [ ] **Step 2: Run the test**

Run: `cargo test -p rust-viewer app::tests::committed_full_frame_result_is_loaded_before_training_is_enabled`

Expected: FAIL because app pipeline events do not load project COLMAP artifacts.

- [ ] **Step 3: Add bounded reconstruction execution and result event handling**

```rust
pub enum PipelineCommand {
    StartThrough { stage: ProjectStage },
    // existing pause, cancel, retry, restart, shutdown commands
}
```

`PipelineCoordinator` stops automatic execution after `FullFramePnp`; it must never invoke a training worker for project reconstruction. `ViewerApp` retains this coordinator after successful image import, drives it from `ui`, replaces `project_summary` from every `ManifestChanged`, and loads the committed full-frame COLMAP root only after the manifest reports `FullFramePnp: Succeeded`. Existing `TrainingManager` remains the only code path that starts RustGS.

- [ ] **Step 4: Run focused verification**

Run: `cargo test -p rust-viewer --lib app::tests::committed_full_frame_result_is_loaded_before_training_is_enabled && cargo test -p rust-viewer --test pipeline_coordinator`

Expected: PASS.

### Task 4: Verify the project-to-RustGS boundary

**Files:**
- Modify: `RustViewer/tests/media_import.rs`
- Modify: `RustViewer/README.md`

- [ ] **Step 1: Add a project-pipeline fixture assertion**

```rust
#[test]
fn image_project_full_frame_result_is_a_loadable_colmap_dataset() {
    let root = completed_image_project_fixture();
    let loaded = load_colmap_training_dataset(&root, &ColmapConfig::default()).unwrap();
    assert_eq!(loaded.summary.frame_count, 2);
}
```

- [ ] **Step 2: Run the integration test**

Run: `cargo test -p rust-viewer --test media_import image_project_full_frame_result_is_a_loadable_colmap_dataset`

Expected: PASS without a GPU adapter.

- [ ] **Step 3: Document delivered behavior only**

State that image sequences can be imported into a project, reconstructed through RustSFM, and handed to RustGS after complete pose coverage. Keep video explicitly blocked pending its dedicated import/PnP implementation.

- [ ] **Step 4: Run final RustViewer verification**

Run: `cargo test -p rust-viewer --lib && cargo test -p rust-viewer --test loader_integration_test && cargo test -p rust-viewer --test media_import && cargo test -p rust-viewer --test pipeline_coordinator && cargo check -p rust-viewer`

Expected: all non-ignored tests pass.
