# RustViewer Project and Media Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give RustViewer a persistent `.rustscanproject` model and background pipeline engine that imports image sequences or macOS video, runs RustSFM, enforces complete pose coverage, and starts resumable RustGS training.

**Architecture:** Keep all orchestration independent of egui. `ProjectStore` owns package validity and atomic manifests, `MediaImporter` produces stable normalized frame records, and `PipelineCoordinator` is the only stage-state authority. Compute workers implement narrow traits and communicate over bounded typed channels so fake workers can test pause, retry, invalidation, and recovery without GPU work.

**Tech Stack:** Rust 2021, serde/serde_json, uuid, blake3, crossbeam-channel, image, objc2 AVFoundation/CoreMedia/CoreVideo on macOS, RustSFM task APIs, RustGS checkpoint APIs.

---

## File Map

- Modify `RustViewer/Cargo.toml`: RustSFM, channel, identity, image, and macOS framework dependencies.
- Create `RustViewer/src/project/mod.rs`: project public API.
- Create `RustViewer/src/project/manifest.rs`: schema, stages, errors, artifacts, migration, and validation.
- Create `RustViewer/src/project/state.rs`: legal transitions, dependency readiness, and invalidation.
- Create `RustViewer/src/project/store.rs`: package creation/open/save, atomic file writes, staging directories, summaries, and lease recovery.
- Create `RustViewer/src/project/source.rs`: managed/reference source identity and macOS bookmark abstraction.
- Create `RustViewer/src/media/mod.rs`: importer API and shared frame metadata.
- Create `RustViewer/src/media/images.rs`: image-sequence normalization and thumbnail generation.
- Create `RustViewer/src/media/keyframes.rs`: deterministic keyframe selection.
- Create `RustViewer/src/media/video.rs`: platform-neutral video decoder trait.
- Create `RustViewer/src/media/avfoundation.rs`: macOS AVFoundation decoder.
- Create `RustViewer/src/pipeline/mod.rs`: coordinator public API.
- Create `RustViewer/src/pipeline/events.rs`: UI-facing typed events and commands.
- Create `RustViewer/src/pipeline/worker.rs`: stage worker traits and production adapters.
- Create `RustViewer/src/pipeline/coordinator.rs`: one-job state machine, control, persistence, invalidation, and recovery.
- Modify `RustViewer/src/training/session.rs`: implement RustGS runner adapter with pause/resume checkpoints.
- Modify `RustViewer/src/lib.rs`: export project, media, and pipeline modules.
- Create `RustViewer/tests/project_store.rs`: manifest/store/migration/lease coverage.
- Create `RustViewer/tests/media_import.rs`: image and video adapter contract coverage.
- Create `RustViewer/tests/pipeline_coordinator.rs`: fake-worker workflow and race coverage.

### Task 1: Manifest Schema and Stage State Machine

**Files:**
- Create: `RustViewer/src/project/mod.rs`
- Create: `RustViewer/src/project/manifest.rs`
- Create: `RustViewer/src/project/state.rs`
- Modify: `RustViewer/src/lib.rs`
- Test: `RustViewer/tests/project_store.rs`

- [ ] **Step 1: Write failing round-trip and transition tests**

Create tests that build a new manifest, serialize it, and require these legal and illegal transitions:

```rust
#[test]
fn manifest_round_trip_preserves_stage_records() {
    let mut manifest = ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));
    manifest.transition(ProjectStage::Import, StageState::Queued).unwrap();
    manifest.transition(ProjectStage::Import, StageState::Running).unwrap();
    let json = serde_json::to_string_pretty(&manifest).unwrap();
    assert_eq!(serde_json::from_str::<ProjectManifest>(&json).unwrap(), manifest);
}

#[test]
fn succeeded_sfm_becomes_stale_when_keyframes_change() {
    let mut manifest = succeeded_manifest_through(ProjectStage::Training);
    manifest.invalidate(ChangeKind::KeyframeSelection);
    assert_eq!(manifest.stage(ProjectStage::KeyframeSfm).state, StageState::Stale);
    assert_eq!(manifest.stage(ProjectStage::FullFramePnp).state, StageState::Stale);
    assert_eq!(manifest.stage(ProjectStage::Training).state, StageState::Stale);
    assert_eq!(manifest.stage(ProjectStage::Import).state, StageState::Succeeded);
}

#[test]
fn running_cannot_jump_directly_to_paused() {
    let mut manifest = ProjectManifest::new("Flowers", SourceSpec::managed_images("source-a"));
    manifest.transition(ProjectStage::Import, StageState::Queued).unwrap();
    manifest.transition(ProjectStage::Import, StageState::Running).unwrap();
    assert!(manifest.transition(ProjectStage::Import, StageState::Paused).is_err());
}
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cargo test -p rust-viewer --test project_store manifest_
cargo test -p rust-viewer --test project_store succeeded_sfm_
```

Expected: FAIL because project types do not exist.

- [ ] **Step 3: Implement the schema**

Use explicit serde names and no wall-clock-only state:

```rust
pub const PROJECT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectStage { Import, KeyframeSfm, FullFramePnp, Training, Complete }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageState {
    NotStarted, Ready, Queued, Running, PauseRequested, Paused,
    CancelRequested, Cancelled, Succeeded, Failed, Stale,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub relative_path: String,
    pub content_hash: String,
    pub byte_len: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageRecord {
    pub state: StageState,
    pub attempt: u32,
    pub completed: Option<u64>,
    pub total: Option<u64>,
    pub started_unix_ms: Option<u64>,
    pub updated_unix_ms: u64,
    pub artifacts: Vec<ArtifactRef>,
    pub error: Option<ProjectErrorRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuggestedAction { Retry, WiderSearch, RevealSource, OpenLog, StartFresh }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectErrorRecord {
    pub code: String,
    pub stage: ProjectStage,
    pub summary: String,
    pub detail: String,
    pub frame_id: Option<u32>,
    pub pair: Option<(u32, u32)>,
    pub retryable: bool,
    pub suggested_actions: Vec<SuggestedAction>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompatibilityRecord {
    pub rustsfm_artifact_version: u32,
    pub rustgs_checkpoint_version: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectLease {
    pub project_id: uuid::Uuid,
    pub stage: ProjectStage,
    pub attempt: u32,
    pub process_id: u32,
    pub started_unix_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectManifest {
    pub schema_version: u32,
    pub id: uuid::Uuid,
    pub display_name: String,
    pub created_unix_ms: u64,
    pub updated_unix_ms: u64,
    pub source: SourceSpec,
    pub import_config: ImportConfigSnapshot,
    pub sfm_config: SfmConfigSnapshot,
    pub pnp_config: PnpConfigSnapshot,
    pub training_config: rustgs::TrainingConfig,
    pub stages: BTreeMap<ProjectStage, StageRecord>,
    pub active_scene: Option<ArtifactRef>,
    pub final_scene: Option<ArtifactRef>,
    pub compatibility: CompatibilityRecord,
    pub lease: Option<ProjectLease>,
}
```

Define source and configuration records with these concrete fields:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceKind { ImageSequence, Video }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceOwnership { ManagedCopy, Referenced }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceSpec {
    pub kind: SourceKind,
    pub ownership: SourceOwnership,
    pub identity: String,
    pub display_paths: Vec<String>,
    pub bookmark: Option<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportConfigSnapshot {
    pub video_keyframes_per_second: f64,
    pub maximum_keyframe_gap_us: i64,
    pub thumbnail_long_edge: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SfmConfigSnapshot {
    pub use_all_images: bool,
    pub use_gpu_sift: bool,
    pub use_gpu_matching: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PnpConfigSnapshot {
    pub narrow_neighbors_each_side: usize,
    pub wide_neighbors_each_side: usize,
    pub min_inliers: usize,
    pub min_inlier_ratio: f64,
    pub max_reprojection_error: f64,
    pub use_gpu_pnp: bool,
}
```

Defaults are 3 keyframes/second, a 1,000,000 microsecond maximum gap, 256-pixel thumbnails, all images for image-sequence SFM, enabled wgpu SIFT/matching/PnP, 2/4 temporal neighbors, 24 PnP inliers, 0.20 inlier ratio, and 4.0 pixel reprojection error. Define `ProjectErrorRecord` with code, stage, summary, detail, optional frame/pair, retryability, and suggested actions. `SourceSpec::managed_images(identity)` is a test helper constructor with `ImageSequence`, `ManagedCopy`, empty paths, and no bookmark.

- [ ] **Step 4: Implement transitions and invalidation**

Encode legal transitions as a total match. `Succeeded -> Stale`, `Paused/Cancelled/Failed -> Queued`, and `Stale -> Ready` are legal; `Running -> Paused` is illegal unless it passes through `PauseRequested`; `Running -> Succeeded` is permitted only through `commit_stage_success`, which requires validated artifacts.

Implement invalidation with this exact dependency table:

```rust
let first = match change {
    ChangeKind::Source => ProjectStage::Import,
    ChangeKind::KeyframeSelection | ChangeKind::SfmConfig => ProjectStage::KeyframeSfm,
    ChangeKind::PnpConfig => ProjectStage::FullFramePnp,
    ChangeKind::TrainingConfig => ProjectStage::Training,
    ChangeKind::ViewerAppearance => return,
};
for stage in ProjectStage::ORDER.into_iter().skip_while(|stage| *stage != first) {
    if self.stage(stage).state == StageState::Succeeded {
        self.stage_mut(stage).state = StageState::Stale;
    } else if self.stage(stage).state != StageState::NotStarted {
        self.stage_mut(stage).state = StageState::Ready;
    }
}
```

- [ ] **Step 5: Run schema/state tests**

Run:

```bash
cargo test -p rust-viewer --test project_store
```

Expected: PASS for round trip, every legal transition, rejected jumps, dependency readiness, and the five invalidation categories.

- [ ] **Step 6: Commit Task 1**

```bash
git add RustViewer/src/project RustViewer/src/lib.rs RustViewer/tests/project_store.rs
git commit -m "feat(viewer): add persistent project state model"
```

### Task 2: Project Store, Atomic Artifacts, and Recovery Lease

**Files:**
- Modify: `RustViewer/Cargo.toml`
- Create: `RustViewer/src/project/store.rs`
- Test: `RustViewer/tests/project_store.rs`

- [ ] **Step 1: Add dependencies and write failing package tests**

Add:

```toml
uuid = { version = "1.24", features = ["serde", "v4"] }
blake3 = "1"
crossbeam-channel = "0.5"
image = "0.25"
rustsfm = { path = "../RustSFM" }
```

Test package creation, atomic manifest replacement, rejection of `../` artifact paths, and recovery from an active lease:

```rust
let store = ProjectStore::create(temp.path().join("Flowers.rustscanproject"), request)?;
assert!(store.root().join("project.json").is_file());
for directory in ["Sources", "Cache/frames", "Cache/thumbnails", "Reconstruction", "Training/checkpoints", "Logs"] {
    assert!(store.root().join(directory).is_dir());
}
store.begin_stage(ProjectStage::Import)?;
drop(store);
let reopened = ProjectStore::open(project_path)?;
assert_eq!(reopened.manifest().stage(ProjectStage::Import).state, StageState::Failed);
assert_eq!(reopened.manifest().stage(ProjectStage::Import).error.as_ref().unwrap().code, "interrupted");
```

- [ ] **Step 2: Run store tests and verify RED**

Run:

```bash
cargo test -p rust-viewer --test project_store project_store_
```

Expected: FAIL because `ProjectStore` does not exist.

- [ ] **Step 3: Implement creation/open/migration**

`ProjectStore::create` requires a `.rustscanproject` suffix, creates the exact package tree from the design, writes schema version 1, and fails if a non-empty destination exists. `open` canonicalizes the root, reads `project.json`, rejects future schema versions, runs explicit sequential migration functions for older versions, and validates every relative artifact remains under the canonical project root.

- [ ] **Step 4: Implement atomic JSON and stage directories**

Centralize writes:

```rust
pub fn write_json_atomic<T: Serialize>(&self, relative: &Path, value: &T) -> Result<(), ProjectStoreError> {
    let destination = self.resolve_relative(relative)?;
    let temporary = destination.with_extension("tmp");
    let bytes = serde_json::to_vec_pretty(value)?;
    let mut file = File::create(&temporary)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    fs::rename(&temporary, &destination)?;
    File::open(destination.parent().expect("project artifact parent"))?.sync_all()?;
    Ok(())
}
```

`stage_workspace(stage, attempt)` returns `Cache/.staging/{stage}-{attempt}`. `commit_stage_success` validates declared artifacts, hashes files with blake3, atomically moves the completed staging tree into its final location, updates `project.json`, and only then marks `Succeeded`.

Add `ProjectStore::list_summaries(library_root)`, `duplicate(destination)`, and `delete(confirmation_id)`. Summary reads include thumbnail, stage states, one-line status, and updated time without loading large artifacts. Duplicate creates a new UUID and atomically copies through a sibling temporary package; delete requires the exact project UUID and removes only the canonical package root. Reveal returns the canonical path for the UI's platform command.

Append every stage transition, structured warning/error, pause/cancel request, and artifact commit as one flushed JSON record in `Logs/events.jsonl`. A corrupt trailing line is preserved for diagnostics and ignored during recovery; `project.json` remains authoritative.

- [ ] **Step 5: Implement lease recovery**

`begin_stage` writes a lease containing project UUID, stage, attempt, process ID, and start time before the worker starts. Clean completion clears it. On open, a remaining lease converts only `Running`, `PauseRequested`, or `CancelRequested` to `Failed` with code `interrupted`, keeps all committed artifacts, moves abandoned staging output to `Logs/recovery/{stage}-{attempt}`, and requires user-confirmed retry.

- [ ] **Step 6: Run project store tests**

Run:

```bash
cargo test -p rust-viewer --test project_store
```

Expected: PASS for atomic replacement, traversal rejection, artifact hashes, stage commit ordering, future-version rejection, and interrupted recovery.

- [ ] **Step 7: Commit Task 2**

```bash
git add RustViewer/Cargo.toml Cargo.lock RustViewer/src/project/store.rs RustViewer/tests/project_store.rs
git commit -m "feat(viewer): persist atomic project packages"
```

### Task 3: Image Sequence Import and Stable Frame Identity

**Files:**
- Create: `RustViewer/src/media/mod.rs`
- Create: `RustViewer/src/media/images.rs`
- Create: `RustViewer/src/project/source.rs`
- Modify: `RustViewer/src/lib.rs`
- Test: `RustViewer/tests/media_import.rs`

- [ ] **Step 1: Write failing import tests**

Build three images named `frame10.png`, `frame2.png`, and `frame1.png`. Require natural ordering, managed copies, stable IDs, readable dimensions, thumbnails, and all-image keyframe selection:

```rust
let result = import_image_sequence(&request, &store, &mut sink)?;
assert_eq!(result.frames.iter().map(|frame| frame.source_name.as_str()).collect::<Vec<_>>(), ["frame1.png", "frame2.png", "frame10.png"]);
assert_eq!(result.frames.iter().map(|frame| frame.id).collect::<Vec<_>>(), [0, 1, 2]);
assert!(result.frames.iter().all(|frame| frame.is_keyframe));
assert!(result.frames.iter().all(|frame| store.resolve(&frame.normalized_image).unwrap().is_file()));
assert!(result.frames.iter().all(|frame| store.resolve(&frame.thumbnail).unwrap().is_file()));
```

- [ ] **Step 2: Run media tests and verify RED**

Run:

```bash
cargo test -p rust-viewer --test media_import image_sequence_
```

Expected: FAIL because media import APIs do not exist.

- [ ] **Step 3: Define frame and importer contracts**

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportedFrame {
    pub id: u32,
    pub source_name: String,
    pub presentation_time_us: Option<i64>,
    pub normalized_image: String,
    pub thumbnail: String,
    pub width: u32,
    pub height: u32,
    pub sharpness: f64,
    pub perceptual_hash: u64,
    pub is_keyframe: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaImportEvent {
    Started { total: Option<usize> },
    FrameCommitted { frame_id: u32, completed: usize, total: Option<usize> },
    Completed { frame_count: usize, keyframe_count: usize },
}

#[derive(Debug, thiserror::Error)]
pub enum MediaImportError {
    #[error("invalid media source: {0}")]
    InvalidSource(String),
    #[error("media decode failed: {0}")]
    Decode(String),
    #[error("media import I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("image conversion failed: {0}")]
    Image(#[from] image::ImageError),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportResult {
    pub source_identity: String,
    pub frames: Vec<ImportedFrame>,
    pub duration_us: Option<i64>,
}

pub trait MediaEventSink {
    fn on_media_event(&mut self, event: MediaImportEvent);
}
```

- [ ] **Step 4: Implement managed image import**

Accept jpg/jpeg/png/bmp/tif/tiff/webp, reject fewer than two readable images, natural-sort digit runs numerically, and assign IDs from zero. Decode with `image`, normalize orientation into lossless PNG at `Cache/frames/{id:08}.png`, and generate a 256-pixel longest-edge JPEG thumbnail at `Cache/thumbnails/{id:08}.jpg`. Compute source identity from sorted filename, byte length, modification nanoseconds, and first/last 64 KiB content hashes.

Use variance of a 3x3 Laplacian on a 320-pixel grayscale preview for sharpness. Use a 64-bit 8x8 DCT perceptual hash for near-duplicate comparisons. Every image-sequence frame sets `is_keyframe = true`.

Atomically write `Sources/source.json`, `Cache/frames.json`, and `Cache/keyframes.json` after all normalized images and thumbnails validate. These three files are the Import stage's declared artifacts.

- [ ] **Step 5: Implement reference-in-place source records**

Define `SourceOwnership::{ManagedCopy, Referenced}`. Managed copy is the default. Referenced image sequences store canonical paths plus identity; opening verifies both. Put macOS bookmark behavior behind `SourceBookmark` so non-macOS image projects compile with a plain canonical path and report a recoverable source error when moved.

- [ ] **Step 6: Run image import tests**

Run:

```bash
cargo test -p rust-viewer --test media_import image_sequence_
cargo test -p rust-viewer --test media_import missing_or_changed_
```

Expected: PASS; unreadable/missing files fail Import without partially committed frame metadata.

- [ ] **Step 7: Commit Task 3**

```bash
git add RustViewer/src/media RustViewer/src/project/source.rs RustViewer/src/lib.rs RustViewer/tests/media_import.rs
git commit -m "feat(viewer): import managed image sequences"
```

### Task 4: Deterministic Video Keyframe Selection

**Files:**
- Create: `RustViewer/src/media/keyframes.rs`
- Test: `RustViewer/tests/media_import.rs`

- [ ] **Step 1: Write failing selection tests**

Build 120 metadata-only frames at 30 fps. Require approximately 3 fps, force every one-second gap, prefer the sharper of near-duplicates, include first and last frames, and return identical IDs on repeated calls.

```rust
let selected = select_keyframes(&frames, KeyframeSelectionConfig::default())?;
assert_eq!(selected.first(), Some(&0));
assert_eq!(selected.last(), Some(&119));
assert!(selected.windows(2).all(|pair| frames[pair[1]].time_us - frames[pair[0]].time_us <= 1_000_000));
assert_eq!(selected, select_keyframes(&frames, KeyframeSelectionConfig::default())?);
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
cargo test -p rust-viewer --test media_import keyframe_
```

Expected: FAIL because selection is absent.

- [ ] **Step 3: Implement bounded-window selection**

Define:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyframeSelectionConfig {
    pub target_per_second: f64,
    pub max_gap_us: i64,
    pub duplicate_hamming_threshold: u32,
}

impl Default for KeyframeSelectionConfig {
    fn default() -> Self {
        Self { target_per_second: 3.0, max_gap_us: 1_000_000, duplicate_hamming_threshold: 6 }
    }
}
```

Divide the timeline into target-rate windows, select the highest tuple `(sharpness, reverse duplicate distance, reverse frame_id)` in each window, then fill any overlong gap with the sharpest candidate before the deadline. Sort and deduplicate final IDs.

- [ ] **Step 4: Run keyframe tests**

Run:

```bash
cargo test -p rust-viewer --test media_import keyframe_
```

Expected: PASS for determinism, sharpness preference, duplicate suppression, endpoints, and max gap.

- [ ] **Step 5: Commit Task 4**

```bash
git add RustViewer/src/media/keyframes.rs RustViewer/tests/media_import.rs
git commit -m "feat(viewer): select deterministic video keyframes"
```

### Task 5: macOS AVFoundation Video Adapter

**Files:**
- Modify: `RustViewer/Cargo.toml`
- Create: `RustViewer/src/media/video.rs`
- Create: `RustViewer/src/media/avfoundation.rs`
- Test: `RustViewer/tests/media_import.rs`

- [ ] **Step 1: Add target-specific dependencies**

Add:

```toml
[target.'cfg(target_os = "macos")'.dependencies]
objc2 = "0.6"
objc2-foundation = { version = "0.3.2", features = ["NSArray", "NSData", "NSError", "NSString", "NSURL"] }
objc2-av-foundation = { version = "0.3.2", features = ["AVAsset", "AVAssetReader", "AVAssetReaderOutput", "AVMediaFormat"] }
objc2-core-media = { version = "0.3.2", features = ["CMSampleBuffer", "CMTime"] }
objc2-core-video = { version = "0.3.2", features = ["CVPixelBuffer"] }
```

- [ ] **Step 2: Write a platform-neutral decoder contract test**

Use `FakeVideoDecoder` to emit five BGRA frames with presentation timestamps. Run `import_video` and assert all five normalized images exist while only selected frames have `is_keyframe`:

```rust
assert_eq!(result.frames.len(), 5);
assert!(result.frames.windows(2).all(|pair| pair[0].presentation_time_us < pair[1].presentation_time_us));
assert!(result.frames.iter().any(|frame| !frame.is_keyframe));
```

- [ ] **Step 3: Define the decoder boundary**

```rust
pub struct DecodedVideoFrame {
    pub presentation_time_us: i64,
    pub width: u32,
    pub height: u32,
    pub bgra: Vec<u8>,
    pub bytes_per_row: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoMetadata {
    pub duration_us: i64,
    pub width: u32,
    pub height: u32,
    pub nominal_fps: f64,
}

pub trait VideoDecoder: Send {
    fn metadata(&self) -> Result<VideoMetadata, MediaImportError>;
    fn next_frame(&mut self) -> Result<Option<DecodedVideoFrame>, MediaImportError>;
}
```

`import_video` converts BGRA to RGB, writes every frame, emits bounded progress, computes quality metadata, then applies the deterministic keyframe selector. It never drops a decoded frame.

- [ ] **Step 4: Implement AVFoundation decoding on macOS**

Open an `AVURLAsset`, select the first video track, configure `AVAssetReaderTrackOutput` for `kCVPixelFormatType_32BGRA`, and loop `copyNextSampleBuffer`. Lock each `CVPixelBuffer` read-only, copy row-strided bytes into owned memory, record `CMSampleBufferGetPresentationTimeStamp`, unlock, and release the sample before emitting the frame. Treat reader status failed/cancelled as structured errors; successful end-of-stream returns `None`.

Implement security-scoped bookmark creation/resolution with `NSURL` bookmark APIs. Balance every successful `startAccessingSecurityScopedResource` with a guard that calls `stopAccessingSecurityScopedResource` on drop.

- [ ] **Step 5: Run adapter tests**

Run:

```bash
cargo test -p rust-viewer --test media_import video_decoder_
cargo test -p rust-viewer --test media_import video_import_
```

Expected: PASS on every platform with the fake decoder. On macOS, an ignored test gated by `RUSTSCAN_VIDEO_FIXTURE` decodes the supplied MOV/MP4 and asserts non-empty, monotonic frames with valid dimensions.

- [ ] **Step 6: Commit Task 5**

```bash
git add RustViewer/Cargo.toml Cargo.lock RustViewer/src/media/video.rs RustViewer/src/media/avfoundation.rs RustViewer/tests/media_import.rs
git commit -m "feat(viewer): decode macOS video with AVFoundation"
```

### Task 6: Typed Pipeline Coordinator with Fake Workers

**Files:**
- Create: `RustViewer/src/pipeline/mod.rs`
- Create: `RustViewer/src/pipeline/events.rs`
- Create: `RustViewer/src/pipeline/worker.rs`
- Create: `RustViewer/src/pipeline/coordinator.rs`
- Modify: `RustViewer/src/lib.rs`
- Test: `RustViewer/tests/pipeline_coordinator.rs`

- [ ] **Step 1: Write failing workflow and race tests**

Use scripted fake workers to cover success, pause, cancel, failure/retry, PnP coverage failure, downstream invalidation, and interrupted recovery. The core success assertion is:

```rust
coordinator.send(PipelineCommand::StartAutomatic)?;
drive_until_idle(&mut coordinator);
assert_eq!(store.manifest().stage(ProjectStage::Complete).state, StageState::Succeeded);
assert_eq!(workers.start_order(), [ProjectStage::Import, ProjectStage::KeyframeSfm, ProjectStage::FullFramePnp, ProjectStage::Training]);
assert_eq!(workers.max_concurrent(), 1);
```

- [ ] **Step 2: Run coordinator tests and verify RED**

Run:

```bash
cargo test -p rust-viewer --test pipeline_coordinator
```

Expected: FAIL because the coordinator does not exist.

- [ ] **Step 3: Define typed commands, events, and worker outcome**

```rust
pub enum PipelineCommand {
    StartAutomatic,
    Pause,
    Cancel,
    Retry { stage: ProjectStage },
    RestartFrom { stage: ProjectStage },
    Shutdown { disposition: ShutdownDisposition },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownDisposition { PauseAndQuit, CancelAndQuit, KeepRunning }

#[derive(Debug, Clone, PartialEq)]
pub enum PipelineProgressDetail {
    Media { frame_id: Option<u32> },
    Sfm { operation: rustsfm::SfmTaskOperation, image_id: Option<u32>, pair: Option<(u32, u32)>, registered_images: Option<usize>, sparse_points: Option<usize> },
    Training { iteration: usize, loss: f32, gaussian_count: usize, elapsed_ms: u64 },
}

pub enum PipelineEvent {
    ManifestChanged(ProjectManifest),
    StageProgress { stage: ProjectStage, completed: Option<u64>, total: Option<u64>, detail: PipelineProgressDetail },
    SceneSnapshot(Arc<rustgs::HostSplats>),
    NeedsAttention { stage: ProjectStage, error: ProjectErrorRecord },
    Idle,
}

pub enum WorkerOutcome {
    Succeeded(Vec<PendingArtifact>),
    Paused(Vec<PendingArtifact>),
    Cancelled(Vec<PendingArtifact>),
    Failed(ProjectErrorRecord),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingArtifact {
    pub staged_path: PathBuf,
    pub final_relative_path: PathBuf,
    pub validation: ArtifactValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactValidation { ReadableFile, Json, ColmapSparse, RustGsCheckpoint, LosslessPly, ParityReport }
```

Define one trait per stage (`ImportWorker`, `SfmWorker`, `PnpWorker`, `TrainingWorker`) sharing `run(request, control, event_sink)`. Worker types have no egui dependency.

- [ ] **Step 4: Implement the single-job coordinator**

Use `crossbeam_channel::bounded(64)` for events and `bounded(8)` for commands. The coordinator determines the next ready dependency, writes the lease, transitions `Ready -> Queued -> Running`, spawns one named thread, and ignores additional Start commands while active. It persists throttled progress at most once per second, but forwards in-memory events immediately.

On outcome, commit/validate artifacts before `Succeeded`, preserve valid artifacts for pause/cancel, clear the lease, and schedule the next stage only for automatic mode. PnP success must additionally satisfy `registered_frames == imported_frames`; otherwise emit `NeedsAttention` and never start training.

- [ ] **Step 5: Implement pause/cancel/retry/restart semantics**

Pause immediately stores `PauseRequested`, calls the worker control token, and waits for `Paused` plus committed boundary artifacts. Cancel follows the same pattern with `CancelRequested`/`Cancelled`. Retry increments only the selected stage attempt. Restart first marks the selected and downstream stages Ready/Stale through `ChangeKind`, retains old final artifacts until replacements succeed, and requires a caller confirmation boolean.

- [ ] **Step 6: Run fake-worker tests**

Run:

```bash
cargo test -p rust-viewer --test pipeline_coordinator
```

Expected: PASS with one maximum concurrent worker, no direct Running-to-Paused transition, no training after incomplete PnP, and deterministic recovery.

- [ ] **Step 7: Commit Task 6**

```bash
git add RustViewer/src/pipeline RustViewer/src/lib.rs RustViewer/tests/pipeline_coordinator.rs
git commit -m "feat(viewer): coordinate persistent project stages"
```

### Task 7: Production RustSFM and RustGS Worker Adapters

**Files:**
- Modify: `RustViewer/src/pipeline/worker.rs`
- Modify: `RustViewer/src/training/session.rs`
- Test: `RustViewer/tests/pipeline_coordinator.rs`

- [ ] **Step 1: Write failing adapter translation tests**

Feed representative `SfmTaskEvent` and `TrainingEvent` values into adapters. Assert stage, progress, image/pair detail, loss, iteration, Gaussian count, and snapshot are preserved. Assert `SfmTaskStop::Paused` maps to `WorkerOutcome::Paused`, not failure.

- [ ] **Step 2: Run adapter tests and verify RED**

Run:

```bash
cargo test -p rust-viewer --test pipeline_coordinator sfm_adapter_
cargo test -p rust-viewer --test pipeline_coordinator training_adapter_
```

Expected: FAIL because production adapters are absent.

- [ ] **Step 3: Implement RustSFM workers**

`RustSfmWorker` builds `MapperConfig` from the manifest snapshot, points it at `Cache/frames`, uses every image for image-sequence projects and only `is_keyframe` frames for video, then invokes `run_keyframe_reconstruction`. It commits the reusable database and keyframe model under `Cache/database.db` and `Reconstruction/keyframes/0`. `RustPnpWorker` invokes `register_remaining_sequence_frames` with those committed inputs, writes `Reconstruction/registration.json`, `Reconstruction/poses.json`, and `Reconstruction/summary.json`, and declares `Reconstruction/sparse/0` only after validation. For image sequences, the PnP runner sees no pending frames and still validates/writes 100% coverage without rerunning SFM.

Forward pause/cancel commands into `SfmTaskControl`. Preserve database and last valid sparse artifacts on either stop outcome.

- [ ] **Step 4: Implement RustGS worker and training-session pause**

Load the validated COLMAP dataset, compute `TrainingIdentity` from dataset, reconstruction hash, and config, then call RustGS with snapshot cadence 100 and checkpoint cadence 1,000. The checkpoint sink writes `Training/checkpoints/iteration-{iteration:06}.rgscp` through RustGS atomic storage. A pause requests `TrainingControl::request_pause`; resume loads the newest compatible checkpoint and continues from its iteration.

Pass the `SharedWgpuContext` created from eframe's render state through `TrainingOptions::with_shared_wgpu_context`. The production worker owns a clone of that context, so training, checkpoint readback, live snapshots, and the existing preview bridge stay on the same Metal adapter/device/queue.

On completion, use the existing lossless PLY writer and parity validator. Declare both `Training/scene.ply` and `Training/scene.parity.json`; failure of either validation keeps Training failed and retains the previous active scene.

- [ ] **Step 5: Run adapter and coordinator tests**

Run:

```bash
cargo test -p rust-viewer --test pipeline_coordinator
cargo test -p rust-viewer training::session
```

Expected: PASS for typed event translation, pause/resume, snapshot forwarding, full coverage gate, and final artifact validation.

- [ ] **Step 6: Commit Task 7**

```bash
git add RustViewer/src/pipeline/worker.rs RustViewer/src/training/session.rs RustViewer/tests/pipeline_coordinator.rs
git commit -m "feat(viewer): connect SFM and GS project workers"
```

### Task 8: Engine Verification and Flowers2 Preflight

**Files:**
- Modify: `RustViewer/tests/project_store.rs`
- Modify: `RustViewer/tests/media_import.rs`
- Modify: `RustViewer/tests/pipeline_coordinator.rs`

- [ ] **Step 1: Run formatting and complete engine tests**

Run:

```bash
cargo fmt --all -- --check
cargo test -p rust-viewer
cargo clippy -p rust-viewer --all-targets -- -D warnings
```

Expected: all commands exit 0.

- [ ] **Step 2: Run a 96-frame image-sequence project**

Create a `.rustscanproject` from the first 96 lexicographically sorted flowers2 images, start automatic processing, and poll until idle. Expected manifest conditions:

```text
import=succeeded frames=96 keyframes=96
keyframe_sfm=succeeded registered=96
full_frame_pnp=succeeded registered=96/96
training=succeeded
complete=succeeded
```

Validate `Reconstruction/sparse/0`, `Reconstruction/poses.json`, `Training/scene.ply`, and `Training/scene.parity.json` through their production loaders.

- [ ] **Step 3: Test restart recovery at both compute stages**

Pause one preflight during SFM and one at RustGS iteration 1,000, close the coordinator, reopen the project, confirm resume, and finish. Expected: SFM reuses its database/sparse boundary; RustGS logs its first resumed iteration as 1,001; neither project restarts Import.

- [ ] **Step 4: Commit Task 8**

```bash
git add RustViewer/tests/project_store.rs RustViewer/tests/media_import.rs RustViewer/tests/pipeline_coordinator.rs
git commit -m "test(viewer): verify persistent project engine"
```
