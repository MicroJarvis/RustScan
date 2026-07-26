# RustViewer Media Pipeline Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make RustViewer's workbench own a persisted project and expose a truthful path from imported media through RustSFM reconstruction to RustGS training and its live wgpu render.

**Architecture:** Keep `ViewerApp` as the egui/wgpu owner. Add a small application-facing project session that converts native file-dialog selections into the existing `ProjectStore`, `media`, and `PipelineCoordinator` contracts; keep worker implementations independent of egui. Image sequences use RustSFM's exported task APIs and emit a COLMAP-compatible sparse result that is loaded through the existing `load_colmap_training_dataset` path. The video path may import and stage frames now, but it must remain blocked before automatic training until the full-frame PnP worker reports complete coverage.

**Tech Stack:** Rust 2021, eframe/egui 0.34, rfd native file dialogs, `ProjectStore`, existing media importers, RustSFM typed task APIs, RustGS `TrainingManager`, wgpu viewport bridge.

---

## Current Workbench Status

- [x] The approved command bar, stage rail, scene inspector, activity strip, and existing live wgpu viewport are composed by `ui::workbench`.
- [x] Direct COLMAP loading, RustGS start/cancel, live HostSplats, camera navigation, layer toggles, and fit remain wired through `ViewerApp`.
- [x] The COLMAP text fixture follows the required two-line `images.txt` record format, restoring the loader test suite.
- [x] Remove the obsolete secondary preview bridge and present a source-neutral empty state.
- [ ] Record real load/training activity entries instead of deriving a one-line placeholder from stage state.

### Task 1: Remove the obsolete secondary preview and make the empty state truthful

**Files:**
- Modify: `RustViewer/src/app.rs`
- Modify: `RustViewer/src/ui/viewport.rs`
- Test: `RustViewer/src/ui/viewport.rs`

- [x] **Step 1: Write the failing empty-state copy test**

```rust
#[test]
fn empty_state_copy_guides_users_to_a_project_or_colmap_workspace() {
    let copy = empty_state_copy();
    assert_eq!(heading, "No reconstruction loaded");
    assert_eq!(detail, "Load a COLMAP workspace from the left rail to begin.");
}
```

- [x] **Step 2: Run the targeted test to establish the failure**

Run: `cargo test -p rust-viewer ui::viewport::tests::empty_state_copy_guides_users_to_colmap_without_a_nonfunctional_action`

Expected: FAIL because `empty_state_copy` does not exist.

- [x] **Step 3: Implement the copy helper and remove unused preview state**

```rust
fn empty_state_copy() -> (&'static str, &'static str) {
    (
        "No reconstruction loaded",
        "Load a COLMAP workspace from the left rail to begin.",
    )
}
```

Remove `preview_camera`, `preview_bridge`, `preview_dirty`, `preview_texture`, `refresh_preview_texture_id`, and `draw_preview_panel` from `ViewerApp`. Preserve `viewport_bridge` and its existing dirty/resolution cache, because it owns the only visible RustGS renderer.

- [ ] **Step 4: Run focused validation**

Run: `cargo test -p rust-viewer ui::viewport::tests::empty_state_copy_guides_users_to_a_project_or_colmap_workspace && cargo check -p rust-viewer`

Expected: test PASS and no obsolete-preview dead-code warning.

### Task 2: Model workbench project stage data without fabricating progress

**Files:**
- Create: `RustViewer/src/project/session.rs`
- Modify: `RustViewer/src/project/mod.rs`
- Modify: `RustViewer/src/ui/workbench.rs`
- Test: `RustViewer/src/project/session.rs`

- [ ] **Step 1: Write failing stage-summary tests**

```rust
#[test]
fn video_import_blocks_training_until_full_frame_registration_succeeds() {
    let summary = ProjectSessionSummary::from_stage_records(
        SourceKind::Video,
        StageState::Succeeded,
        StageState::Succeeded,
        StageState::Ready,
        StageState::NotStarted,
    );
    assert_eq!(summary.training_state, ProjectStagePresentation::Waiting);
    assert_eq!(summary.training_detail, "Waiting for full-frame poses");
}
```

- [ ] **Step 2: Run the targeted test to establish the failure**

Run: `cargo test -p rust-viewer project::session::tests::video_import_blocks_training_until_full_frame_registration_succeeds`

Expected: FAIL because `ProjectSessionSummary` does not exist.

- [ ] **Step 3: Implement a presentation-only project summary**

```rust
pub struct ProjectSessionSummary {
    pub source_kind: SourceKind,
    pub imported_frames: Option<u64>,
    pub keyframe_sfm: ProjectStagePresentation,
    pub full_frame_pnp: ProjectStagePresentation,
    pub training_state: ProjectStagePresentation,
    pub training_detail: &'static str,
}
```

Construct it from `ProjectManifest::stage(ProjectStage)` records only. Extend `WorkbenchSnapshot` with `project: Option<ProjectSessionSummary>` and prefer that state over legacy direct-COLMAP state when it exists. Never mark any stage completed from a file-dialog result alone.

- [ ] **Step 4: Run targeted validation**

Run: `cargo test -p rust-viewer project::session::tests && cargo test -p rust-viewer ui::workbench::tests`

Expected: PASS.

### Task 3: Add a real image-sequence import to a project package

**Files:**
- Modify: `RustViewer/src/app.rs`
- Modify: `RustViewer/src/ui/panel.rs`
- Modify: `RustViewer/src/ui/workbench.rs`
- Test: `RustViewer/src/app.rs`

- [ ] **Step 1: Write a failing command-routing test**

```rust
#[test]
fn image_sequence_action_routes_to_project_import() {
    assert_eq!(panel_action_for_primary_import(SourceSelection::Images), PanelAction::ImportImages);
}
```

- [ ] **Step 2: Run the targeted test to establish the failure**

Run: `cargo test -p rust-viewer app::tests::image_sequence_action_routes_to_project_import`

Expected: FAIL because `ImportImages` is not an action.

- [ ] **Step 3: Implement the native dialog and background import**

```rust
enum AppCommand {
    ImageSequenceImported(Result<ProjectImportSummary, String>),
    // existing variants
}

fn spawn_image_sequence_import(&mut self) {
    self.ui_state.is_loading = true;
    self.ui_state.loading_message = Some("Importing image sequence...".to_owned());
    // pick_files -> save_file(.rustscanproject) -> ProjectStore::create -> import_image_sequence
}
```

Use `rfd::FileDialog::pick_files` for images and `save_file` for the package destination. Create the project with `ProjectStore::create`, call `import_image_sequence` on the background thread, then return the immutable summary to `ViewerApp`. Do not keep an unlocked source path or a `ProjectStore` across a UI frame.

- [ ] **Step 4: Run the targeted test and media integration suite**

Run: `cargo test -p rust-viewer app::tests::image_sequence_action_routes_to_project_import && cargo test -p rust-viewer --test media_import`

Expected: PASS.

### Task 4: Run RustSFM for imported image sequences and hand the verified sparse output to RustGS

**Files:**
- Modify: `RustViewer/Cargo.toml`
- Create: `RustViewer/src/pipeline/rustsfm_worker.rs`
- Modify: `RustViewer/src/pipeline/mod.rs`
- Modify: `RustViewer/src/app.rs`
- Test: `RustViewer/src/pipeline/rustsfm_worker.rs`

- [ ] **Step 1: Write failing worker-contract tests**

```rust
#[test]
fn image_sequence_sfm_worker_exports_a_colmap_dataset_only_after_success() {
    let result = run_image_sequence_sfm(&fixture_project(), WorkerControl::new(), &mut Vec::new());
    assert!(result.is_ok());
    assert!(result.unwrap().colmap_root.join("sparse/0/images.txt").is_file());
}
```

- [ ] **Step 2: Run the targeted test to establish the failure**

Run: `cargo test -p rust-viewer pipeline::rustsfm_worker::tests::image_sequence_sfm_worker_exports_a_colmap_dataset_only_after_success`

Expected: FAIL because `run_image_sequence_sfm` does not exist.

- [ ] **Step 3: Implement the worker against RustSFM's public task API**

```rust
let frames = imported_frames
    .iter()
    .map(|frame| rustsfm::SequenceFrame {
        id: frame.id,
        image_path: project_root.join(&frame.normalized_image),
        timestamp_us: frame.presentation_time_us,
    })
    .collect::<Vec<_>>();
let reconstruction = rustsfm::run_keyframe_reconstruction(&frames, &config, &mut task_context)?;
rustsfm::colmap::export_colmap_sparse_model(&output_root, &reconstruction.reconstruction)?;
```

Forward `SfmTaskEvent` values as `PipelineEvent::Progress`, map pause/cancel to the existing `WorkerOutcome`, and declare only validated output artifacts. `ViewerApp` loads the resulting output with `load_colmap_training_dataset`; the existing `TrainingManager` then remains the only RustGS execution path.

- [ ] **Step 4: Run cross-crate and workbench verification**

Run: `cargo test -p rustsfm --lib sequence_registration && cargo test -p rust-viewer pipeline::rustsfm_worker::tests && cargo test -p rust-viewer --lib && cargo check -p rust-viewer`

Expected: PASS.

### Task 5: Add video import and preserve the training gate

**Files:**
- Modify: `RustViewer/src/app.rs`
- Modify: `RustViewer/src/ui/workbench.rs`
- Test: `RustViewer/tests/media_import.rs`

- [ ] **Step 1: Write a failing video gate test**

```rust
#[test]
fn imported_video_cannot_enable_rustgs_before_pnp_coverage_is_complete() {
    let snapshot = video_project_snapshot(StageState::Succeeded, StageState::Ready);
    assert!(!snapshot.primary_command().enabled);
}
```

- [ ] **Step 2: Run the targeted test to establish the failure**

Run: `cargo test -p rust-viewer --test media_import imported_video_cannot_enable_rustgs_before_pnp_coverage_is_complete`

Expected: FAIL because video project snapshots are not exposed to the workbench.

- [ ] **Step 3: Implement import-only video selection**

Use `AvFoundationVideoDecoder::open_managed` with `import_video` on macOS, retain the generated project summary, and render an explicit `Waiting for full-frame poses` training state. Do not add a start-training action for that state until the PnP worker commits complete coverage.

- [ ] **Step 4: Run the media and crate verification**

Run: `cargo test -p rust-viewer --test media_import && cargo test -p rust-viewer --lib && cargo check -p rust-viewer`

Expected: PASS.

### Task 6: Native visual QA and documentation

**Files:**
- Modify: `RustViewer/README.md`
- Create: `design-qa.md`

- [ ] **Step 1: Capture source and rendered empty, imported, training, and blocked-video states**

Use the approved HTML workbench at `http://localhost:56818/` as the source visual and capture the native RustViewer states at 1280x800 and 1728x1117.

- [ ] **Step 2: Compare layout and control behavior**

Verify that the viewport remains dominant, labels do not clip, the inspector does not obscure the viewport, direct artifact loading remains available, and disabled video training never appears actionable.

- [ ] **Step 3: Update the README with only delivered behavior**

Document direct COLMAP, image-project import, RustSFM reconstruction, RustGS training, and live render only after each is functional. State the remaining video/PnP boundary explicitly until it is implemented.

- [ ] **Step 4: Run final verification**

Run: `cargo test -p rust-viewer --lib && cargo test -p rust-viewer --test loader_integration_test && cargo test -p rust-viewer --test media_import && cargo check -p rust-viewer && cargo build -p rust-viewer`

Expected: all commands exit 0; `design-qa.md` has `final result: passed`.
