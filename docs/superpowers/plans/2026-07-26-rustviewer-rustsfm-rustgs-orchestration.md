# RustViewer RustSFM-to-RustGS Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let RustViewer import a raw image directory, reconstruct it with RustSFM, and automatically begin RustGS training only after a valid COLMAP dataset is produced.

**Architecture:** A focused reconstruction module validates source images, creates collision-free run output, configures RustSFM image-only matching, and forwards callbacks as typed events. ViewerApp receives worker events and loads generated COLMAP output before starting TrainingManager.

**Tech Stack:** Rust 2021, eframe/egui, RustSFM, RustGS, std thread and mpsc, tempfile.

---

## File Structure

- `RustViewer/Cargo.toml`: direct RustSFM library dependency.
- `RustViewer/src/reconstruction.rs`: image validation, unique run directories, RustSFM configuration, callback forwarding, and runner seam.
- `RustViewer/src/lib.rs`: reconstruction module export.
- `RustViewer/src/ui/panel.rs`: source and reconstruction state, import and run commands.
- `RustViewer/src/app.rs`: asynchronous pipeline orchestration and auto-training gate.
- `RustViewer/README.md`: user workflow and runtime boundaries.

### Task 1: Add the RustSFM Runner Boundary

**Files:**

- Modify: `RustViewer/Cargo.toml:11-24`
- Modify: `RustViewer/src/lib.rs:1-8`
- Create: `RustViewer/src/reconstruction.rs`
- Test: `RustViewer/src/reconstruction.rs`

- [ ] **Step 1: Write the failing filesystem and configuration tests**

    #[test]
    fn image_source_requires_two_supported_images() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("only.jpg"), b"image").unwrap();
        let error = ImageSource::open(temp.path()).unwrap_err();
        assert!(error.to_string().contains("at least two JPEG or PNG images"));
    }

    #[test]
    fn run_directory_is_unique_and_stays_below_the_source() {
        let temp = tempfile::tempdir().unwrap();
        let first = create_run_directory(temp.path()).unwrap();
        let second = create_run_directory(temp.path()).unwrap();
        assert_ne!(first, second);
        assert!(first.starts_with(temp.path().join(".rustviewer/rustsfm")));
    }

    #[test]
    fn mapper_config_enables_local_matching_and_image_copying() {
        let input = PathBuf::from("/captures/set-a");
        let output = PathBuf::from("/captures/set-a/.rustviewer/rustsfm/run-1");
        let config = mapper_config_for(&input, &output);
        assert_eq!(config.input, input);
        assert_eq!(config.output, output);
        assert!(config.local_matching && config.copy_images);
        assert!(config.fx.is_none() && config.fy.is_none());
    }

- [ ] **Step 2: Run the test and verify RED**

Run: `cargo test -p rust-viewer reconstruction::tests`

Expected: compilation failure because `reconstruction`, `ImageSource`, `create_run_directory`, and `mapper_config_for` do not yet exist.

- [ ] **Step 3: Implement the minimal RustSFM runner module**

Add to `RustViewer/Cargo.toml`:

    anyhow = "1"
    rustsfm = { path = "../RustSFM", default-features = false, features = ["gpu-wgpu"] }

Add `pub mod reconstruction;` to `RustViewer/src/lib.rs`. Create the following types in `RustViewer/src/reconstruction.rs`:

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ImageSource { root: PathBuf, images: Vec<PathBuf> }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ReconstructionProgress {
        pub registered_images: usize,
        pub registered_frames: usize,
        pub points: usize,
        pub stage: &'static str,
    }

    #[derive(Debug, Clone)]
    pub struct ReconstructionOutput {
        pub output_dir: PathBuf,
        pub summary: rustsfm::ReconstructionSummary,
    }

    pub trait ReconstructionRunner: Send + 'static {
        fn run(
            &self,
            source: &ImageSource,
            output_dir: PathBuf,
            emit: &mut dyn FnMut(ReconstructionProgress),
        ) -> anyhow::Result<ReconstructionOutput>;
    }

    #[derive(Debug, Default, Clone, Copy)]
    pub struct RustSfmRunner;

`ImageSource::open` must sort direct `.jpg`, `.jpeg`, and `.png` children and reject fewer than two. `create_run_directory` must create a unique `run-<unix-nanos>-<counter>` directory under `<input>/.rustviewer/rustsfm/`, retrying `AlreadyExists` without deleting existing data. `mapper_config_for` must return `MapperConfig { input, output, local_matching: true, copy_images: true, ..Default::default() }`; do not set `fx`, `fy`, `cx`, or `cy`.

`RustSfmRunner::run` must create a local `PipelineCallbackSink` forwarding `registered_images`, `registered_frames`, `points`, and `callback.as_str()` through `emit`. It must call `rustsfm::run_reconstruction_with_callbacks` and return `ReconstructionOutput`; Task 4 adds the explicit completed-output validation before this runner returns.

- [ ] **Step 4: Run the runner tests and verify GREEN**

Run: `cargo test -p rust-viewer reconstruction::tests`

Expected: PASS without a real reconstruction or GPU adapter.

- [ ] **Step 5: Commit**

Run: `git add RustViewer/Cargo.toml RustViewer/src/lib.rs RustViewer/src/reconstruction.rs`

Run: `git commit -m "feat(viewer): add RustSFM reconstruction runner"`

### Task 2: Add Image Import and Run Controls

**Files:**

- Modify: `RustViewer/src/ui/panel.rs:8-90,117-145,278-390`
- Test: `RustViewer/src/ui/panel.rs`

- [ ] **Step 1: Write failing UI-state tests**

    #[test]
    fn reconstruction_can_run_only_with_source_and_idle_state() {
        let mut state = UiState::default();
        assert!(!state.can_run_reconstruction());
        state.image_source = Some(ImageSourceSummary {
            root_path: "/captures/chair".to_owned(), image_count: 24,
        });
        assert!(state.can_run_reconstruction());
        state.reconstruction_state = ReconstructionUiState::Running;
        assert!(!state.can_run_reconstruction());
    }

    #[test]
    fn reconstruction_labels_are_operator_facing() {
        assert_eq!(reconstruction_state_label(ReconstructionUiState::Ready), "ready");
        assert_eq!(reconstruction_state_label(ReconstructionUiState::Running), "solving poses");
        assert_eq!(reconstruction_state_label(ReconstructionUiState::Failed), "failed");
    }

- [ ] **Step 2: Run the tests and verify RED**

Run: `cargo test -p rust-viewer ui::panel::tests::reconstruction_`

Expected: compilation failure because the reconstruction UI types and helpers are absent.

- [ ] **Step 3: Add state and commands to the panel model**

Add `OpenImages` and `RunReconstruction` to `PanelAction`; retain every existing action. Add the following panel types:

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ImageSourceSummary {
        pub root_path: String,
        pub image_count: usize,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ReconstructionUiState {
        Idle,
        Ready,
        Running,
        Completed,
        Failed,
    }

    impl UiState {
        pub fn can_run_reconstruction(&self) -> bool {
            self.image_source.is_some()
                && !matches!(self.reconstruction_state, ReconstructionUiState::Running)
        }
    }

Add `image_source`, `reconstruction_state`, `reconstruction_registered_images`, `reconstruction_points`, and `reconstruction_error` to `UiState`, with empty or idle defaults. Add an `Import Images` action in file operations and a `Run Reconstruction` control in the training section. While reconstruction is running, show `solving poses` plus registered-image and sparse-point values, and disable both importing and a second run request. Do not alter manual `Load COLMAP` or `Start Training` behavior.

- [ ] **Step 4: Run the panel tests and verify GREEN**

Run: `cargo test -p rust-viewer ui::panel::tests::reconstruction_`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add RustViewer/src/ui/panel.rs`

Run: `git commit -m "feat(viewer): add image reconstruction controls"`

### Task 3: Orchestrate the Background Workflow and Auto-Start RustGS

**Files:**

- Modify: `RustViewer/src/app.rs:40-48,140-146,258-376,1190-1370`
- Test: `RustViewer/src/app.rs`

- [ ] **Step 1: Write failing state-transition tests**

    #[test]
    fn reconstruction_progress_updates_ui_state() {
        let mut state = UiState::default();
        apply_reconstruction_progress(&mut state, ReconstructionProgress {
            registered_images: 5,
            registered_frames: 5,
            points: 123,
            stage: "next_image_reg",
        });
        assert_eq!(state.reconstruction_state, ReconstructionUiState::Running);
        assert_eq!(state.reconstruction_registered_images, 5);
        assert_eq!(state.reconstruction_points, 123);
    }

    #[test]
    fn reconstruction_failure_never_requests_training() {
        let mut state = UiState::default();
        let start_training = apply_reconstruction_result(
            &mut state,
            Err("no verified image pairs".to_owned()),
        );
        assert!(!start_training);
        assert_eq!(state.reconstruction_state, ReconstructionUiState::Failed);
        assert_eq!(state.reconstruction_error.as_deref(), Some("no verified image pairs"));
    }

    #[test]
    fn successful_reconstruction_requests_training_only_after_completion() {
        let mut state = UiState::default();
        let start_training = apply_reconstruction_result(&mut state, Ok(()));
        assert!(start_training);
        assert_eq!(state.reconstruction_state, ReconstructionUiState::Completed);
        assert!(state.reconstruction_error.is_none());
    }

- [ ] **Step 2: Run the tests and verify RED**

Run: `cargo test -p rust-viewer app::tests::reconstruction_`

Expected: compilation failure because `apply_reconstruction_progress` and `apply_reconstruction_result` do not exist.

- [ ] **Step 3: Extend `AppCommand` and implement worker sequencing**

Extend `AppCommand` with these variants while retaining `LoadAsset` and `ColmapLoaded`:

    ImageSourceLoaded(Result<ImageSource, String>),
    ReconstructionProgress(ReconstructionProgress),
    ReconstructionFinished(Result<LoadedColmapDataset, String>),

`OpenImages` calls `spawn_image_import`, which uses `rfd::FileDialog::pick_folder`, validates the selected directory with `ImageSource::open`, and sends `ImageSourceLoaded`. `RunReconstruction` calls `spawn_reconstruction`, which executes this exact order on a background thread:

    let output = create_run_directory(source.root()).map_err(|error| error.to_string())?;
    let mut emit = |progress| {
        let _ = tx.send(AppCommand::ReconstructionProgress(progress));
    };
    RustSfmRunner.run(&source, output.clone(), &mut emit)
        .map_err(|error| error.to_string())?;
    load_colmap_training_dataset(&output, &ColmapConfig::default())
        .map_err(|error| error.to_string())

Store `image_source: Option<ImageSource>` on `ViewerApp`. On a successful `ImageSourceLoaded`, update the `ImageSourceSummary`, clear prior reconstruction error, and set the UI state to `Ready`. On a failed image import, set `Failed` with the returned error. Use these pure helpers for state changes:

    fn apply_reconstruction_progress(state: &mut UiState, progress: ReconstructionProgress) {
        state.reconstruction_state = ReconstructionUiState::Running;
        state.reconstruction_registered_images = progress.registered_images;
        state.reconstruction_points = progress.points;
    }

    fn apply_reconstruction_result(state: &mut UiState, result: Result<(), String>) -> bool {
        match result {
            Ok(()) => {
                state.reconstruction_state = ReconstructionUiState::Completed;
                state.reconstruction_error = None;
                true
            }
            Err(error) => {
                state.reconstruction_state = ReconstructionUiState::Failed;
                state.reconstruction_error = Some(error);
                false
            }
        }
    }

For `ReconstructionFinished(Ok(loaded))`, call `handle_colmap_loaded(Ok(loaded))`, then call `apply_reconstruction_result(&mut self.ui_state, Ok(()))`, then call `start_training()`. For an error, call `apply_reconstruction_result` with that error, clear loader status, and do not call `start_training`.

- [ ] **Step 4: Run orchestration and existing loader tests**

Run: `cargo test -p rust-viewer app::tests::reconstruction_ && cargo test -p rust-viewer loader::colmap::tests`

Expected: PASS. The success test gates the call to `start_training()` behind completed reconstruction; the failure test demonstrates that a failed RustSFM run cannot reach RustGS; loader tests preserve manual COLMAP support.

- [ ] **Step 5: Commit**

Run: `git add RustViewer/src/app.rs`

Run: `git commit -m "feat(viewer): orchestrate RustSFM before RustGS training"`

### Task 4: Validate Completed Output and Document the Workflow

**Files:**

- Modify: `RustViewer/src/reconstruction.rs`
- Modify: `RustViewer/README.md:5-55`
- Test: `RustViewer/src/reconstruction.rs`

- [ ] **Step 1: Write a failing completed-output test**

    #[test]
    fn completed_summary_requires_registered_images_and_sparse_points() {
        let empty = ReconstructionSummary {
            images: 12,
            registered_images: 0,
            points: 0,
            pairs: 0,
            models: 0,
            elapsed_ms: 1.0,
            debug_log: Vec::new(),
        };
        assert!(validate_completed_summary(&empty).is_err());
    }

- [ ] **Step 2: Run the test and verify RED**

Run: `cargo test -p rust-viewer reconstruction::tests::completed_summary_requires_registered_images_and_sparse_points`

Expected: compilation failure because `validate_completed_summary` is absent.

- [ ] **Step 3: Implement validation and document exact behavior**

Add this helper in `RustViewer/src/reconstruction.rs` and call it from `RustSfmRunner::run` before returning `ReconstructionOutput`:

    pub fn validate_completed_summary(
        summary: &rustsfm::ReconstructionSummary,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            summary.registered_images > 0,
            "RustSFM completed without registered images",
        );
        anyhow::ensure!(summary.points > 0, "RustSFM completed without sparse points");
        Ok(())
    }

Update `RustViewer/README.md` with this operator workflow:

    1. Select **Import Images** and choose a folder with at least two JPEG or PNG files.
    2. Select **Run Reconstruction**. RustViewer writes each result to
       `<input>/.rustviewer/rustsfm/<run-id>/` without overwriting older runs.
    3. After RustSFM exports a valid COLMAP sparse model, RustViewer automatically starts
       RustGS training and displays live Gaussian snapshots.

Document these explicit boundaries: RustSFM's automatic single-camera intrinsic estimate is used; macOS wgpu compute uses Metal rather than native Vulkan; this release cannot cancel RustSFM reconstruction, while existing RustGS training cancellation remains available.

- [ ] **Step 4: Run full verification**

Run: `cargo test -p rust-viewer --lib && cargo test -p rust-viewer --test loader_integration_test && cargo check -p rust-viewer`

Expected: all non-ignored library tests pass; fixture-dependent integration tests remain ignored; `cargo check` exits zero.

- [ ] **Step 5: Commit**

Run: `git add RustViewer/README.md RustViewer/src/reconstruction.rs`

Run: `git commit -m "docs(viewer): document image reconstruction workflow"`

## Plan Self-Review

- **Spec coverage:** Task 1 covers raw-image validation, non-overwriting output, automatic intrinsics, local matching, image copying, callback forwarding, and runner output. Task 2 provides explicit import/run controls and prevents concurrent reconstruction starts. Task 3 handles threaded execution, progress, COLMAP loading, success-gated RustGS auto-start, and terminal errors. Task 4 validates sparse output and documents user-visible limits.
- **No placeholders:** Every task names exact files, types, functions, commands, expected failure modes, and expected passing verification.
- **Type consistency:** `ImageSource`, `ReconstructionProgress`, `ReconstructionOutput`, `ReconstructionRunner`, `RustSfmRunner`, `ReconstructionUiState`, `ImageSourceSummary`, `apply_reconstruction_progress`, and `apply_reconstruction_result` use the same names throughout.
