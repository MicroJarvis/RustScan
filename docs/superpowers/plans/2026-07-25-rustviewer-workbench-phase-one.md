# RustViewer Workbench Phase One Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace RustViewer's current utility side panel with the approved reconstruction-workbench layout, wired to its existing COLMAP loading, RustGS training, and live Gaussian rendering paths.

**Architecture:** Keep `ViewerApp` as the owner of loading, training, renderer, and camera state. Add a focused `ui::workbench` presentation module that derives the four visible pipeline stages from `UiState` and the loaded scene, returns the existing `PanelAction` commands, and renders a compact activity feed from typed activity entries. No UI code will fabricate successful reconstruction results: a stage becomes complete only when the existing loader or training manager reports a valid result.

**Tech Stack:** Rust 2021, eframe/egui 0.34, existing RustGS `TrainingManager`, existing wgpu viewport bridge, unit tests in RustViewer.

---

### Task 1: Add workbench presentation state with deterministic status derivation

**Files:**
- Create: `RustViewer/src/ui/workbench.rs`
- Modify: `RustViewer/src/ui/mod.rs`
- Test: `RustViewer/src/ui/workbench.rs`

- [ ] **Step 1: Write failing tests for stage derivation**

```rust
#[test]
fn derives_import_pose_train_and_render_states_from_real_app_state() {
    let idle = WorkbenchSnapshot::default();
    assert_eq!(idle.stages()[0].state, PipelineStageState::Ready);
    assert_eq!(idle.stages()[1].state, PipelineStageState::Waiting);

    let loaded = WorkbenchSnapshot {
        dataset: Some(DatasetUiSummary::test_summary(87, 4_420)),
        training_state: TrainingSessionState::Idle,
        training_progress: TrainingProgress::default(),
        has_gaussian_snapshot: false,
        ..Default::default()
    };
    assert_eq!(loaded.stages()[1].state, PipelineStageState::Completed);
    assert_eq!(loaded.stages()[2].state, PipelineStageState::Ready);
}
```

- [ ] **Step 2: Run the targeted test to establish the failure**

Run: `cargo test -p rust-viewer ui::workbench::tests::derives_import_pose_train_and_render_states_from_real_app_state`

Expected: FAIL because `ui::workbench` and `WorkbenchSnapshot` do not exist.

- [ ] **Step 3: Implement only presentation types and stage derivation**

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStageState { Ready, Waiting, Running, Completed, Failed }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineStage { pub name: &'static str, pub detail: String, pub state: PipelineStageState }

#[derive(Debug, Clone, Default)]
pub struct WorkbenchSnapshot { /* dataset, training, splat and error fields */ }

impl WorkbenchSnapshot {
    pub fn stages(&self) -> [PipelineStage; 4] { /* import, pose, train, render */ }
}
```

Export the module from `ui/mod.rs`. Add a `DatasetUiSummary::test_summary` helper under `#[cfg(test)]` rather than constructing unrelated application state in tests.

- [ ] **Step 4: Run the targeted test to verify it passes**

Run: `cargo test -p rust-viewer ui::workbench::tests::derives_import_pose_train_and_render_states_from_real_app_state`

Expected: PASS.

### Task 2: Preserve observable workbench activity in app state

**Files:**
- Modify: `RustViewer/src/app.rs`
- Modify: `RustViewer/src/ui/workbench.rs`
- Test: `RustViewer/src/app.rs`

- [ ] **Step 1: Write failing activity-feed tests**

```rust
#[test]
fn activity_feed_keeps_the_newest_events_with_a_bounded_history() {
    let mut feed = ActivityFeed::default();
    for index in 0..(ACTIVITY_HISTORY_LIMIT + 1) {
        feed.push(ActivityLevel::Info, format!("event {index}"));
    }
    assert_eq!(feed.entries().len(), ACTIVITY_HISTORY_LIMIT);
    assert_eq!(feed.entries().last().unwrap().message, "event 1");
}
```

- [ ] **Step 2: Run the targeted test to establish the failure**

Run: `cargo test -p rust-viewer app::tests::activity_feed_keeps_the_newest_events_with_a_bounded_history`

Expected: FAIL because `ActivityFeed` is not defined.

- [ ] **Step 3: Add the bounded typed feed and record real events**

```rust
const ACTIVITY_HISTORY_LIMIT: usize = 40;

struct ActivityFeed { entries: VecDeque<ActivityEntry> }

impl ActivityFeed {
    fn push(&mut self, level: ActivityLevel, message: impl Into<String>) { /* push_front, truncate */ }
}
```

Add `activity_feed` to `ViewerApp`. Record only actual events from `handle_colmap_loaded`, loader failures, `start_training`, `stop_training`, and `poll_training_events`; use the existing error strings verbatim. Do not emit a fake RustSFM success because raw-media reconstruction is not implemented in this phase.

- [ ] **Step 4: Run the targeted test to verify it passes**

Run: `cargo test -p rust-viewer app::tests::activity_feed_keeps_the_newest_events_with_a_bounded_history`

Expected: PASS.

### Task 3: Render the approved command bar, stage rail, inspector, and activity strip

**Files:**
- Modify: `RustViewer/src/ui/theme.rs`
- Modify: `RustViewer/src/ui/workbench.rs`
- Test: `RustViewer/src/ui/workbench.rs`

- [ ] **Step 1: Write failing helper tests for metrics and primary command state**

```rust
#[test]
fn primary_command_requires_a_loaded_colmap_dataset() {
    assert!(!primary_command(&WorkbenchSnapshot::default()).enabled);
    assert!(primary_command(&loaded_idle_snapshot()).enabled);
}
```

- [ ] **Step 2: Run the targeted test to establish the failure**

Run: `cargo test -p rust-viewer ui::workbench::tests::primary_command_requires_a_loaded_colmap_dataset`

Expected: FAIL because `primary_command` is not defined.

- [ ] **Step 3: Implement the workbench draw functions**

```rust
pub fn draw_command_bar(ui: &mut egui::Ui, snapshot: &WorkbenchSnapshot);
pub fn draw_stage_rail(ui: &mut egui::Ui, snapshot: &WorkbenchSnapshot) -> Vec<PanelAction>;
pub fn draw_inspector(ui: &mut egui::Ui, state: &mut UiState, snapshot: &WorkbenchSnapshot) -> Vec<PanelAction>;
pub fn draw_activity_strip(ui: &mut egui::Ui, feed: &ActivityFeed, snapshot: &WorkbenchSnapshot);
```

Use fixed panel widths (180 px stage rail and 300 px inspector), thin separators, the existing dark-neutral palette, and compact labels. Add only semantic theme tokens required by the workbench. The training command must be disabled until a real COLMAP dataset is loaded; while training, expose the existing `StopTraining` action as `Cancel training` rather than calling it pause.

- [ ] **Step 4: Run the targeted test to verify it passes**

Run: `cargo test -p rust-viewer ui::workbench::tests::primary_command_requires_a_loaded_colmap_dataset`

Expected: PASS.

### Task 4: Wire the workbench around the existing interactive viewport

**Files:**
- Modify: `RustViewer/src/app.rs`
- Modify: `RustViewer/src/ui/viewport.rs`
- Test: `RustViewer/src/app.rs`

- [ ] **Step 1: Write a failing snapshot-construction test**

```rust
#[test]
fn workbench_snapshot_reports_live_training_and_render_data() {
    let snapshot = workbench_snapshot_for(
        Some(DatasetUiSummary::test_summary(12, 42)),
        TrainingSessionState::Training,
        TrainingProgress { gaussian_count: Some(64), ..Default::default() },
        true,
        None,
    );
    assert_eq!(snapshot.registered_camera_count(), 12);
    assert_eq!(snapshot.gaussian_count(), Some(64));
}
```

- [ ] **Step 2: Run the targeted test to establish the failure**

Run: `cargo test -p rust-viewer app::tests::workbench_snapshot_reports_live_training_and_render_data`

Expected: FAIL because `workbench_snapshot_for` is not defined.

- [ ] **Step 3: Replace the panel composition in `ViewerApp::ui`**

```rust
egui::TopBottomPanel::top("workbench_command_bar").exact_height(56.0).show_inside(ui, |ui| { /* bar */ });
egui::SidePanel::left("workbench_stage_rail").exact_width(180.0).show_inside(ui, |ui| { /* stages */ });
egui::SidePanel::right("workbench_inspector").exact_width(300.0).show_inside(ui, |ui| { /* inspector */ });
egui::TopBottomPanel::bottom("workbench_activity").exact_height(164.0).show_inside(ui, |ui| { /* feed */ });
egui::CentralPanel::default().show_inside(ui, |ui| { /* current interactive wgpu viewport */ });
```

Build `WorkbenchSnapshot` from the already-owned `loaded_colmap`, `TrainingManager`, `loaded_splats`, `UiState`, and scene counts. Keep all existing orbit, pan, zoom, fit, layer, picking, and direct asset-loading behavior. Move the useful viewport controls into a compact viewport toolbar, but do not introduce mock data.

- [ ] **Step 4: Run the focused test and crate check**

Run: `cargo test -p rust-viewer app::tests::workbench_snapshot_reports_live_training_and_render_data && cargo check -p rust-viewer`

Expected: test PASS and crate check PASS.

### Task 5: Verify user-facing behavior and protect the existing artifact workflow

**Files:**
- Modify: `RustViewer/README.md`
- Test: `RustViewer/tests/loader_integration_test.rs`

- [ ] **Step 1: Add regression coverage for direct COLMAP loading**

Use the existing text COLMAP fixture helper in `loader/colmap.rs` to assert that the UI-facing summary reports frame count, sparse point count, and resolution after successful load. Keep fixture data local and do not depend on a GPU adapter.

- [ ] **Step 2: Update the README with actual phase-one behavior**

Document that the workbench can load a valid COLMAP dataset, start or cancel RustGS training, and display live Gaussian snapshots. State explicitly that raw image-sequence/video import and RustSFM orchestration are the next implementation phase.

- [ ] **Step 3: Run verification**

Run: `cargo test -p rust-viewer --lib && cargo test -p rust-viewer --test loader_integration_test && cargo check -p rust-viewer`

Expected: all non-ignored tests pass and `cargo check` succeeds. Report any existing upstream test failure separately instead of weakening assertions.

### Phase Two Boundary

After phase one is accepted, implement the existing approved macOS project-pipeline specification in separate changes: project package persistence, image-sequence importer, macOS video decoder, RustSFM control/event API, full-frame PnP, pause/resume checkpoints, and final PLY/parity export. Those features require new durable domain modules and should not be simulated in the workbench UI.
