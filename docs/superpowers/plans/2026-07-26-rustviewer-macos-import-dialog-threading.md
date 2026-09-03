# RustViewer macOS Import Dialog Threading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make image-folder import complete on macOS so a saved image project reaches the `运行重建` state.

**Architecture:** `ViewerApp` opens the native rfd file and save dialogs synchronously on its eframe UI thread. A small `ImageImportRequest` value separates the completed dialog selection from background project creation. The existing worker thread only copies images and creates the project package, then returns the existing `AppCommand` result.

**Tech Stack:** Rust, eframe/egui, rfd native dialogs, existing RustViewer project store and media importer.

---

### Task 1: Separate Dialog Selection From Background Import

**Files:**
- Modify: `RustViewer/src/app.rs:60-85`
- Modify: `RustViewer/src/app.rs:617-684`
- Test: `RustViewer/src/app.rs:1843-1866`

- [ ] **Step 1: Write the failing test**

Add an `ImageImportRequest` test beside the current image-folder test. It must prove that project creation is only queued when both the image selection and the project destination exist:

```rust
#[test]
fn image_import_request_requires_selected_images_and_destination() {
    let paths = vec![PathBuf::from("frame-01.png"), PathBuf::from("frame-02.png")];
    let destination = PathBuf::from("Capture.rustscanproject");

    assert!(image_import_request(Some(paths.clone()), Some(destination.clone())).is_some());
    assert!(image_import_request(None, Some(destination.clone())).is_none());
    assert!(image_import_request(Some(paths), None).is_none());
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
cargo test -p rust-viewer app::tests::image_import_request_requires_selected_images_and_destination --lib --locked -- --exact
```

Expected: compilation failure because `image_import_request` does not exist.

- [ ] **Step 3: Implement the minimal request boundary and UI-thread dialogs**

Add this private request type and helper near `ProjectImportSummary`:

```rust
#[derive(Debug, Clone)]
struct ImageImportRequest {
    paths: Vec<PathBuf>,
    destination: PathBuf,
}

fn image_import_request(
    paths: Option<Vec<PathBuf>>,
    destination: Option<PathBuf>,
) -> Option<ImageImportRequest> {
    Some(ImageImportRequest {
        paths: paths?,
        destination: destination?,
    })
}
```

In `spawn_image_sequence_import`, call `rfd::FileDialog::pick_files`, `pick_folder`, and `save_file` before `std::thread::spawn`. Build the request with `image_import_request`; on `None`, send `AppCommand::ImageSequenceImportCancelled`. Keep only this work inside the spawned closure:

```rust
let result = create_image_sequence_project(request.paths, request.destination)
    .map_err(|error| error.to_string());
let _ = tx.send(AppCommand::ImageSequenceImported(result));
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
cargo test -p rust-viewer app::tests::image_import_request_requires_selected_images_and_destination --lib --locked -- --exact
```

Expected: PASS.

- [ ] **Step 5: Run focused import and UI regressions**

Run:

```bash
cargo test -p rust-viewer app::tests::image_folder_selection_collects_only_supported_image_files --lib --locked -- --exact
cargo test -p rust-viewer app::tests::imported_image_project_waits_for_explicit_reconstruction_start --lib --locked -- --exact
cargo fmt --all -- --check
```

Expected: all commands pass.

- [ ] **Step 6: Commit**

```bash
git add RustViewer/src/app.rs docs/superpowers/plans/2026-07-26-rustviewer-macos-import-dialog-threading.md
git commit -m "fix(viewer): open image dialogs on UI thread"
```

### Task 2: Verify The Release Application

**Files:**
- Modify: none
- Test: `RustViewer/src/app.rs`

- [ ] **Step 1: Run full RustViewer verification**

Run:

```bash
cargo test -p rust-viewer --all-targets --locked
cargo build -p rust-viewer --release --locked
git diff --check
```

Expected: RustViewer tests and release build pass; no diff whitespace errors.

- [ ] **Step 2: Launch release and manually verify the state transition**

Launch `target/release/rust-viewer`, choose a supported image folder, save a `.rustscanproject`, and confirm the activity list changes from `Choosing image folder` to imported frames and the left pipeline action changes to `运行重建`.
