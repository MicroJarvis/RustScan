# RustViewer Project Open And Retry Implementation Plan

**Goal:** Reopen a RustScan project and restore its retryable pose-solve control.

**Architecture:** Route `.rustscanproject` selections through a dedicated asset kind. Reuse the existing project pipeline constructor and imported-project state setup so startup arguments and the dialog observe the same manifest-derived UI state.

### Task 1: Recognize project packages

- [ ] Add a failing unit test that asserts `startup_asset_kind(Path::new("Retry.rustscanproject"))` returns `AssetLoadKind::ProjectPackage`.
- [ ] Run `cargo test -p rust-viewer --lib startup_asset_kind_recognizes_rustscan_projects` and confirm it fails because the variant is absent.
- [ ] Add the variant and extension mapping.
- [ ] Rerun the focused test and confirm it passes.

### Task 2: Restore failed project state

- [ ] Add a failing test that creates a two-image managed project, marks `KeyframeSfm` retryable failed, opens it, and expects an enabled `Retry pose solve` primary command.
- [ ] Run `cargo test -p rust-viewer --lib opening_failed_project_restores_retry_pose_command` and confirm it fails because no project-open handler exists.
- [ ] Implement the handler using `new_project_pipeline`, clearing non-project state but sending no pipeline command.
- [ ] Rerun the focused test and confirm it passes.

### Task 3: Add a normal UI entry point

- [ ] Add `PanelAction::OpenProject` and the `Open RustScan Project` control in `RustViewer/src/ui/panel.rs`.
- [ ] Add the directory dialog and dispatch it through the shared project-package handler in `RustViewer/src/app.rs`.

### Task 4: Verify and integrate

- [ ] Run `cargo test -p rust-viewer`, `cargo fmt --all --check`, `git diff --check`, and `cargo build --release -p rust-viewer`.
- [ ] Commit, merge into `main`, remove the worktree and feature branch, then build the release binary from `main`.
