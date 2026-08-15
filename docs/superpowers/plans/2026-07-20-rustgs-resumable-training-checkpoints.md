# RustGS Resumable Training Checkpoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make RustGS training pauseable and exactly resumable from versioned checkpoints containing Gaussian, optimizer, topology, schedule, and identity state.

**Architecture:** Define a backend-neutral serialized checkpoint model and atomic file store, then add explicit export/import methods at the optimizer and trainer ownership boundaries. Extend the existing training control/event API additively so the runtime can checkpoint after complete iterations, reject incompatible resumes, and continue at the next frame/iteration without resetting optimizer or topology state.

**Tech Stack:** Rust 2021, serde, bincode 1.3, blake3, Burn 0.20 git revision already pinned by the workspace, wgpu/Metal, existing RustGS event and training runtime.

---

## File Map

- Modify `RustGS/Cargo.toml`: add checkpoint encoding and stable hashing dependencies.
- Modify `RustGS/src/core/splats.rs`: make checkpoint payload equality testable.
- Create `RustGS/src/training/checkpoint.rs`: public schema, identity hashing, validation, and atomic storage.
- Modify `RustGS/src/training/mod.rs`: export checkpoint APIs.
- Modify `RustGS/src/lib.rs`: re-export checkpoint and resume APIs.
- Modify `RustGS/src/training/engine/optimizer.rs`: export and restore Adam tensor state.
- Modify `RustGS/src/training/engine/topology_accum.rs`: export and restore topology tensors.
- Modify `RustGS/src/training/engine/trainer.rs`: construct/restore full trainer checkpoints and continue from an iteration offset.
- Modify `RustGS/src/training/events.rs`: pause state, checkpoint cadence, checkpoint events, identity, and resume options.
- Modify `RustGS/src/training/engine/runtime.rs`: resume validation, periodic/pause checkpoints, and terminal paused result.
- Modify `RustGS/src/training/evaluation/core.rs`: expose the Burn device held by a shared eframe wgpu context to the training runtime.
- Modify `RustGS/src/bin/rustgs/train_command.rs`: optional CLI checkpoint/resume support without changing current defaults.
- Create `RustGS/tests/checkpoint_resume.rs`: schema, compatibility, atomic I/O, and resumed continuity tests.

### Task 1: Versioned Checkpoint Schema and Atomic Store

**Files:**
- Modify: `RustGS/Cargo.toml`
- Modify: `RustGS/src/core/splats.rs`
- Create: `RustGS/src/training/checkpoint.rs`
- Modify: `RustGS/src/training/mod.rs`
- Modify: `RustGS/src/lib.rs`
- Test: `RustGS/tests/checkpoint_resume.rs`

- [ ] **Step 1: Add dependencies and write failing round-trip tests**

Add:

```toml
bincode = "1.3"
blake3 = "1"
```

Create `RustGS/tests/checkpoint_resume.rs` with a minimal checkpoint generated through a test builder. Assert exact round trip and replacement of an existing file:

```rust
#[test]
fn checkpoint_store_round_trips_and_atomically_replaces() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("iteration-000010.rgscp");
    let first = TrainingCheckpoint::test_fixture(10, "dataset-a", "reconstruction-a");
    save_training_checkpoint(&path, &first).unwrap();
    assert_eq!(load_training_checkpoint(&path).unwrap(), first);

    let second = TrainingCheckpoint::test_fixture(20, "dataset-a", "reconstruction-a");
    save_training_checkpoint(&path, &second).unwrap();
    assert_eq!(load_training_checkpoint(&path).unwrap(), second);
    assert!(!path.with_extension("rgscp.tmp").exists());
}
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
cargo test -p rustgs --test checkpoint_resume checkpoint_store_
```

Expected: FAIL because checkpoint types and storage functions do not exist.

- [ ] **Step 3: Implement the public schema**

Define these serializable types in `checkpoint.rs`:

```rust
pub const TRAINING_CHECKPOINT_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingIdentity {
    pub dataset: String,
    pub reconstruction: String,
    pub config: String,
}

impl TrainingIdentity {
    pub fn from_inputs(dataset: &TrainingDataset, reconstruction: &str, config: &TrainingConfig) -> Result<Self, TrainingError> {
        let dataset_bytes = serde_json::to_vec(dataset)
            .map_err(|error| TrainingError::InvalidInput(error.to_string()))?;
        let mut resume_compatible_config = config.clone();
        resume_compatible_config.iterations = 0;
        let config_bytes = serde_json::to_vec(&resume_compatible_config)
            .map_err(|error| TrainingError::InvalidInput(error.to_string()))?;
        Ok(Self {
            dataset: blake3::hash(&dataset_bytes).to_hex().to_string(),
            reconstruction: reconstruction.to_owned(),
            config: blake3::hash(&config_bytes).to_hex().to_string(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorCheckpoint {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdamParameterCheckpoint {
    pub moment1: Option<TensorCheckpoint>,
    pub moment2: Option<TensorCheckpoint>,
    pub scaling: Option<TensorCheckpoint>,
    pub step: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdamCheckpoint {
    pub transforms: AdamParameterCheckpoint,
    pub sh_coeffs: AdamParameterCheckpoint,
    pub raw_opacities: AdamParameterCheckpoint,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopologyCheckpoint {
    pub grad_2d: TensorCheckpoint,
    pub screen_grad_2d: TensorCheckpoint,
    pub abs_grad_2d: TensorCheckpoint,
    pub abs_pixel_grad_2d: TensorCheckpoint,
    pub pixel_coverage: TensorCheckpoint,
    pub camera_depth: TensorCheckpoint,
    pub grad_color: TensorCheckpoint,
    pub num_observations: TensorCheckpoint,
    pub visible_observations: TensorCheckpoint,
    pub actual_visible_observations: TensorCheckpoint,
    pub splat_birth_iterations: Vec<usize>,
    pub splat_invisible_windows: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    pub version: u32,
    pub identity: TrainingIdentity,
    pub completed_iterations: usize,
    pub latest_loss: Option<f32>,
    pub splats: HostSplats,
    pub optimizer: AdamCheckpoint,
    pub topology: TopologyCheckpoint,
    pub frame_shuffle_seed: u64,
    pub active_sh_degree: usize,
}
```

Add `PartialEq` to the existing `HostSplats` derive so complete checkpoint round trips can be compared without weakening field privacy.

Add `TrainingCheckpoint::validate()` that checks version, identity fields, finite loss, `HostSplats::validate()`, every tensor's shape product against value count, finite tensor values, and all topology vector lengths against `splats.len()`.

- [ ] **Step 4: Implement atomic bincode storage**

Use a sibling temporary file and sync before rename:

```rust
pub fn save_training_checkpoint(path: &Path, checkpoint: &TrainingCheckpoint) -> Result<(), TrainingError> {
    checkpoint.validate()?;
    let bytes = bincode::serialize(checkpoint)
        .map_err(|error| TrainingError::TrainingFailed(format!("encode checkpoint: {error}")))?;
    let temp = path.with_extension("rgscp.tmp");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::File::create(&temp)?;
    std::io::Write::write_all(&mut file, &bytes)?;
    file.sync_all()?;
    std::fs::rename(&temp, path)?;
    if let Some(parent) = path.parent() {
        std::fs::File::open(parent)?.sync_all()?;
    }
    Ok(())
}

pub fn load_training_checkpoint(path: &Path) -> Result<TrainingCheckpoint, TrainingError> {
    let bytes = std::fs::read(path)?;
    let checkpoint: TrainingCheckpoint = bincode::deserialize(&bytes)
        .map_err(|error| TrainingError::InvalidInput(format!("decode checkpoint: {error}")))?;
    checkpoint.validate()?;
    Ok(checkpoint)
}
```

Keep I/O failures on the existing `TrainingError::Io(#[from] std::io::Error)` route. Encoding, decoding, schema, and identity failures use `TrainingFailed` or `InvalidInput` as shown above.

- [ ] **Step 5: Run schema and store tests**

Run:

```bash
cargo test -p rustgs --test checkpoint_resume checkpoint_
```

Expected: PASS for valid round trip and FAIL-as-expected assertions for truncated, non-finite, and wrong-version files.

- [ ] **Step 6: Commit Task 1**

```bash
git add RustGS/Cargo.toml Cargo.lock RustGS/src/core/splats.rs RustGS/src/training/checkpoint.rs RustGS/src/training/mod.rs RustGS/src/lib.rs RustGS/tests/checkpoint_resume.rs
git commit -m "feat(rustgs): add versioned training checkpoints"
```

### Task 2: Adam Optimizer Export and Restore

**Files:**
- Modify: `RustGS/src/training/engine/optimizer.rs`
- Test: `RustGS/src/training/engine/optimizer.rs`

- [ ] **Step 1: Write failing optimizer state tests**

Create a small device tensor, perform two optimizer steps, export the state, restore it into a fresh optimizer, and require the third step to match exactly within `1e-6`:

```rust
let checkpoint = optimizer.checkpoint().await.unwrap();
assert_eq!(checkpoint.transforms.step, 2);
let mut restored = AdamScaled::<GsBackendBase>::new(config.clone());
restored.restore(&checkpoint, &device).unwrap();
let expected = step_once(&mut optimizer, splats.clone(), grads.clone()).await;
let actual = step_once(&mut restored, splats, grads).await;
assert_abs_diff_eq!(expected, actual, epsilon = 1e-6);
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
cargo test -p rustgs --lib optimizer_checkpoint_
```

Expected: FAIL because `AdamScaled` cannot export or restore its private states.

- [ ] **Step 3: Implement generic tensor conversion helpers**

Add asynchronous `tensor_checkpoint` using `tensor.to_data_async().await`, and `restore_tensor` using `TensorData::new(values, Shape::from(shape))`. Reject a dimensionality mismatch before constructing the tensor.

```rust
async fn checkpoint_state<B: Backend, const D: usize>(state: &AdamState<B, D>) -> Result<AdamParameterCheckpoint, TrainingError> {
    Ok(AdamParameterCheckpoint {
        moment1: tensor_checkpoint_opt(state.moment1.clone()).await?,
        moment2: tensor_checkpoint_opt(state.moment2.clone()).await?,
        scaling: tensor_checkpoint_opt(state.scaling.clone()).await?,
        step: state.step,
    })
}
```

- [ ] **Step 4: Add `AdamScaled::checkpoint` and `restore`**

The restore method must verify that moment1/moment2 are both present or both absent and that each restored tensor has the same shape as its corresponding splat parameter. Scaling may be absent because it is rebuilt by the trainer's learning-rate update. Restore all three step counters exactly.

- [ ] **Step 5: Run optimizer tests**

Run:

```bash
cargo test -p rustgs --lib optimizer
```

Expected: PASS; uninterrupted and restored updates match within `1e-6`.

- [ ] **Step 6: Commit Task 2**

```bash
git add RustGS/src/training/engine/optimizer.rs
git commit -m "feat(rustgs): checkpoint Adam optimizer state"
```

### Task 3: Topology and Trainer State Export/Restore

**Files:**
- Modify: `RustGS/src/training/engine/topology_accum.rs`
- Modify: `RustGS/src/training/engine/trainer.rs`
- Test: `RustGS/src/training/engine/trainer.rs`

- [ ] **Step 1: Write failing trainer round-trip tests**

Construct a trainer with three splats, non-zero accumulator tensors, birth iterations `[0, 4, 8]`, invisible windows `[1, 2, 3]`, and a stepped optimizer. Export and restore, then assert every host vector and step matches.

```rust
let checkpoint = trainer.checkpoint(&device_splats, 12, Some(0.125)).await?;
let (mut restored_trainer, restored_splats) = WgpuTrainer::from_checkpoint(
    config,
    device,
    scene_scale,
    &checkpoint,
).await?;
assert_eq!(restored_splats.num_splats(), 3);
assert_eq!(restored_trainer.splat_birth_iterations, vec![0, 4, 8]);
assert_eq!(restored_trainer.splat_invisible_windows, vec![1, 2, 3]);
assert_eq!(restored_trainer.optimizer.checkpoint().await?, checkpoint.optimizer);
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
cargo test -p rustgs --lib trainer_checkpoint_
```

Expected: FAIL because trainer checkpoint methods do not exist.

- [ ] **Step 3: Implement topology tensor checkpoint helpers**

Add `TopologyAccumulatorSet::checkpoint().await` and `TopologyAccumulatorSet::from_checkpoint(checkpoint, device)`. All ten tensors are rank-one arrays with length equal to the current splat count. Centralize length validation so topology mutation and checkpoint restore share the same invariant.

- [ ] **Step 4: Implement full trainer checkpoint creation**

Add:

```rust
pub(crate) async fn checkpoint(
    &self,
    splats: &DeviceSplats<GsDiffBackend>,
    identity: TrainingIdentity,
    completed_iterations: usize,
    latest_loss: Option<f32>,
) -> Result<TrainingCheckpoint, TrainingError> {
    let host = device_splats_to_host(splats).await;
    Ok(TrainingCheckpoint {
        version: TRAINING_CHECKPOINT_VERSION,
        identity,
        completed_iterations,
        latest_loss,
        active_sh_degree: host.sh_degree(),
        frame_shuffle_seed: self.config.data.frame_shuffle_seed,
        splats: host,
        optimizer: self.optimizer.checkpoint().await?,
        topology: self.topology_checkpoint().await?,
    })
}
```

- [ ] **Step 5: Implement trainer restoration**

`WgpuTrainer::from_checkpoint` creates device splats from `checkpoint.splats`, initializes a trainer with matching dimensions, restores Adam and topology tensors, copies birth/invisibility vectors, and calls `update_optimizer_lrs(checkpoint.completed_iterations, sh_coeffs)` so the decay schedule continues. Reject mismatched splat/tensor lengths before allocating GPU buffers.

- [ ] **Step 6: Run trainer and topology tests**

Run:

```bash
cargo test -p rustgs --lib trainer_checkpoint_
cargo test -p rustgs --lib topology
```

Expected: PASS with exact state round trip and no regression in mutation planning.

- [ ] **Step 7: Commit Task 3**

```bash
git add RustGS/src/training/engine/topology_accum.rs RustGS/src/training/engine/trainer.rs
git commit -m "feat(rustgs): restore trainer topology state"
```

### Task 4: Pause, Periodic Checkpoints, and Resume Options

**Files:**
- Modify: `RustGS/src/training/events.rs`
- Modify: `RustGS/src/training/engine/trainer.rs`
- Modify: `RustGS/src/training/engine/runtime.rs`
- Test: `RustGS/src/training/engine/runtime.rs`
- Test: `RustGS/tests/checkpoint_resume.rs`

- [ ] **Step 1: Write failing control and cadence tests**

Require cancel to override pause, periodic policy at 1,000 iterations, and a pause checkpoint after iteration 7:

```rust
let control = TrainingControl::new(TrainingEventCadence {
    progress_every: 5,
    snapshot_every: Some(100),
});
let policy = TrainingCheckpointPolicy { every: Some(1_000) };
control.request_pause();
assert!(control.is_pause_requested());
control.request_cancel();
assert!(control.is_cancel_requested());
assert!(!control.is_pause_requested());
assert!(policy.should_checkpoint(1_000));
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
cargo test -p rustgs --lib training_control_
cargo test -p rustgs --test checkpoint_resume pause_
```

Expected: FAIL because control has only a cancellation boolean.

- [ ] **Step 3: Extend control and events additively**

Replace the internal boolean with an `AtomicU8` state while keeping `request_cancel`, `is_cancel_requested`, and the two-field `TrainingEventCadence` source-compatible. Add `request_pause`, `is_pause_requested`, and a separate checkpoint policy:

```rust
pub struct TrainingCheckpointReady {
    pub iteration: usize,
    pub reason: TrainingCheckpointReason,
    pub checkpoint: TrainingCheckpoint,
}

pub enum TrainingCheckpointReason { Periodic, Pause, Shutdown }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TrainingCheckpointPolicy {
    pub every: Option<usize>,
}

impl TrainingCheckpointPolicy {
    pub fn should_checkpoint(self, iteration: usize) -> bool {
        self.every.is_some_and(|every| iteration > 0 && iteration.is_multiple_of(every.max(1)))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingRunDisposition { Completed, Paused, Cancelled }
```

Add `TrainingEvent::CheckpointReady` and `TrainingEvent::RunPaused`. Keep the existing `cancelled` field in `TrainingRunReport` and add `disposition` so method-based callers retain their behavior and new callers can distinguish pause; update the workspace's internal report struct literals in the same task.

- [ ] **Step 4: Extend `TrainingOptions` with identity and resume state**

Add these fields to `TrainingOptions<'a>`:

```rust
pub identity: Option<TrainingIdentity>,
pub resume_checkpoint: Option<TrainingCheckpoint>,
pub checkpoint_policy: TrainingCheckpointPolicy,
pub shared_wgpu_context: Option<SharedWgpuContext>,
pub on_checkpoint: Option<Box<dyn FnMut(&TrainingCheckpointReady) -> Result<(), TrainingError> + 'a>>,
```

Add builder methods:

```rust
pub fn with_identity(mut self, identity: TrainingIdentity) -> Self {
    self.identity = Some(identity);
    self
}

pub fn with_resume_checkpoint(mut self, checkpoint: TrainingCheckpoint) -> Self {
    self.resume_checkpoint = Some(checkpoint);
    self
}

pub fn with_checkpoint_policy(mut self, policy: TrainingCheckpointPolicy) -> Self {
    self.checkpoint_policy = policy;
    self
}

pub fn with_shared_wgpu_context(mut self, context: SharedWgpuContext) -> Self {
    self.shared_wgpu_context = Some(context);
    self
}

pub fn with_checkpoint_sink<F>(mut self, sink: F) -> Self
where
    F: FnMut(&TrainingCheckpointReady) -> Result<(), TrainingError> + 'a,
{
    self.on_checkpoint = Some(Box::new(sink));
    self
}
```

Store the sink separately from the normal event sink so a failed disk commit fails the run instead of merely logging an event.

Expose a crate-visible `SharedWgpuContext::training_device()` that clones its Burn device. In `run_training`, use the supplied shared context's device when present and `GsDevice::default()` otherwise. Thread this device through initialization, warm-up, trainer construction, checkpoint restoration, and viewport snapshots so RustViewer training and preview use the eframe-owned adapter/device/queue rather than opening a second Metal device.

- [ ] **Step 5: Continue the trainer loop from the checkpoint offset**

Change `train_with_frame_loader` to accept `start_iteration`. Apply these exact index changes to the existing loop; the prefetch, target-cache, training step, topology, metric, and snapshot body remains between the shown unchanged context lines:

```diff
-for iteration in 0..num_iterations {
+for zero_based in start_iteration..num_iterations {
     if observer.should_cancel() {
         report.cancelled = true;
         break;
     }
-    let sample_idx = iteration % cameras.len();
+    let sample_idx = zero_based % cameras.len();
     let frame_idx = frame_order[sample_idx];
     frame_loader.prefetch_order_window(frame_order, sample_idx)?;
-    let iteration_idx = iteration + 1;
+    let iteration_idx = zero_based + 1;
```

After each complete iteration, if periodic cadence or pause is active, create the full checkpoint asynchronously, pass it to the checkpoint sink, then emit `CheckpointReady`. A pause sets disposition only after the sink returns `Ok(())`; cancel remains a non-resumable terminal request but retains the latest already committed checkpoint.

- [ ] **Step 6: Validate identity before GPU initialization**

Compare all identity fields and return messages in this exact form:

```text
checkpoint dataset does not match the current training dataset
checkpoint reconstruction does not match the current sparse reconstruction
checkpoint configuration does not match the current training configuration
```

Permit only `TrainingConfig.iterations` to increase on resume by computing the configuration hash from a clone with `iterations = 0`; reject lowering the target below `completed_iterations`.

- [ ] **Step 7: Run runtime and checkpoint tests**

Run:

```bash
cargo test -p rustgs --lib engine::runtime
cargo test -p rustgs --test checkpoint_resume
```

Expected: PASS; pausing at iteration 7 emits a committed checkpoint, resuming begins at iteration 8, and incompatible identities fail before a wgpu device is requested.

- [ ] **Step 8: Commit Task 4**

```bash
git add RustGS/src/training/events.rs RustGS/src/training/engine/trainer.rs RustGS/src/training/engine/runtime.rs RustGS/src/training/evaluation/core.rs RustGS/tests/checkpoint_resume.rs
git commit -m "feat(rustgs): pause and resume training runs"
```

### Task 5: Resumed Numerical Continuity

**Files:**
- Modify: `RustGS/tests/checkpoint_resume.rs`
- Modify: `RustGS/src/training/data/frame_loader.rs`

- [ ] **Step 1: Write a deterministic uninterrupted-versus-resumed test**

Train the existing tiny synthetic dataset for 12 iterations in one run. Train the same dataset for 7 iterations, capture its checkpoint, resume to 12, then compare packed splat components and report counters:

```rust
assert_eq!(resumed.report.completed_iterations, 12);
assert_eq!(resumed.splats.len(), uninterrupted.splats.len());
for (actual, expected) in resumed
    .splats.as_view().positions
    .iter()
    .zip(uninterrupted.splats.as_view().positions)
{
    assert!((actual - expected).abs() <= 1e-5);
}
assert_eq!(resumed.report.disposition, TrainingRunDisposition::Completed);
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
cargo test -p rustgs --test checkpoint_resume resumed_training_matches_
```

Expected: FAIL until frame ordering and every mutable trainer state continue from the stored iteration.

- [ ] **Step 3: Make frame ordering resume-stable**

Keep `ordered_frame_indices` deterministic from `frame_shuffle_seed`. Ensure prefetch receives `sample_idx = completed_iterations % frame_count` on the first resumed iteration and never reshuffles based on process-local state. Topology randomness already derives from `(frame_shuffle_seed, iteration)`; document this in the checkpoint field and keep no hidden mutable RNG.

- [ ] **Step 4: Run numerical continuity tests**

Run:

```bash
cargo test -p rustgs --test checkpoint_resume resumed_training_matches_ -- --nocapture
```

Expected: PASS within `1e-5` for all splat component arrays and exact equality for Gaussian count, SH degree, and completed iteration.

- [ ] **Step 5: Commit Task 5**

```bash
git add RustGS/src/training/data/frame_loader.rs RustGS/tests/checkpoint_resume.rs
git commit -m "test(rustgs): verify resumed training continuity"
```

### Task 6: CLI Compatibility and Full Verification

**Files:**
- Modify: `RustGS/src/bin/rustgs/train_command.rs`
- Test: `RustGS/tests/integration_test.rs`

- [ ] **Step 1: Add optional CLI checkpoint flags**

Add `--checkpoint-dir`, `--checkpoint-every` defaulting to 1,000 when a directory is supplied, and `--resume <file>`. Existing invocations without these flags retain current behavior. Save files as `iteration-{iteration:06}.rgscp` and retain the newest three periodic checkpoints plus any pause/shutdown checkpoint.

- [ ] **Step 2: Add a CLI parse regression test**

Parse the old minimal train command and assert every new field is `None`; parse a resume command and assert its path and cadence exactly.

- [ ] **Step 3: Run formatting and all RustGS checks**

Run:

```bash
cargo fmt --all -- --check
cargo test -p rustgs --all-features
cargo clippy -p rustgs --all-targets --all-features -- -D warnings
```

Expected: all commands exit 0.

- [ ] **Step 4: Run a real Metal pause/resume smoke test**

Train the 96-frame flowers2 preflight to iteration 1,100 with checkpoint cadence 1,000, resume that checkpoint to 1,200, and validate the final lossless PLY through the existing parity command. Expected: resume logs start at iteration 1,001, final report says 1,200, PLY loads successfully, and parity passes finite-value and round-trip checks.

- [ ] **Step 5: Commit Task 6**

```bash
git add RustGS/src/bin/rustgs/train_command.rs RustGS/tests/integration_test.rs
git commit -m "feat(rustgs): expose resumable CLI training"
```
