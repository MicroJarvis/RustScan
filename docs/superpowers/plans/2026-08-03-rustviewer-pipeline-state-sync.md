# RustViewer Pipeline State Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the pose-solve retry control whenever the durable project manifest is failed, even if the terminal pipeline event was dropped.

**Architecture:** `ViewerApp` will refresh its `ProjectSessionSummary` directly from the `PipelineCoordinator` store after each drive. The refreshed summary controls the workbench primary command, while queued events continue to feed activity text.

**Tech Stack:** Rust, eframe/egui, RustViewer project store, cargo test.

---

### Task 1: Cover durable failure synchronization

**Files:**
- Modify: `RustViewer/src/app.rs`

- [x] **Step 1: Write the failing test**

Create a test-only helper that updates a running app summary from a failed
`ProjectManifest`. Assert that `WorkbenchSnapshot::primary_command()` is enabled
and labelled `Retry pose solve`, and that `UiState::is_loading` is false.

- [x] **Step 2: Run test to verify it fails**

Run: `cargo test -p rust-viewer persisted_pose_failure_refreshes_retry_control -- --exact`

Expected: FAIL because no app-level durable-state refresh helper exists.

- [x] **Step 3: Implement the minimal synchronization helper**

Add one private `ViewerApp` method that derives the summary from the manifest,
records its stage activity, and clears loading plus records the stored error for
a persisted reconstruction failure. Call it after `PipelineCoordinator::drive_once`.

- [x] **Step 4: Run test to verify it passes**

Run: `cargo test -p rust-viewer persisted_pose_failure_refreshes_retry_control -- --exact`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add RustViewer/src/app.rs docs/superpowers/specs/2026-08-03-rustviewer-pipeline-state-sync-design.md docs/superpowers/plans/2026-08-03-rustviewer-pipeline-state-sync.md
git commit -m "fix(rustviewer): refresh pipeline failure state"
```

### Task 2: Verify the integration surface

**Files:**
- Modify: `RustViewer/src/app.rs`

- [x] **Step 1: Run focused state and coordinator tests**

Run: `cargo test -p rust-viewer persisted_pose_failure_refreshes_retry_control -- --exact`
Run: `cargo test -p rust-viewer --test pipeline_coordinator`

Expected: both commands pass.

- [x] **Step 2: Run full verification**

Run: `cargo test -p rust-viewer`
Run: `cargo build --release -p rust-viewer`
Run: `cargo fmt --all --check`
Run: `git diff --check`

Expected: all commands exit successfully.
