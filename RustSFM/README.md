# RustSFM

COLMAP-style incremental SfM implementation for RustScan.

## Default Taskflow pipeline scheduling

High-level feature extraction, database matching/geometric verification,
incremental/global reconstruction, adaptive keyframe selection, remaining-frame
registration and export now use Taskflow stage admission by default. This also
covers the CLI and RustViewer entry points; `--taskflow` is **not required** to
activate stage scheduling. Ceres inside these stages inherits the CPU allowance.
The sequence pipeline submits a real two-node dependency graph:

```text
keyframe reconstruction -> remaining registration + final BA/export
```

Nested synchronous stages reuse their parent's reservation instead of queueing
while holding it. Existing sequence final-only Global BA behavior is unchanged.
The default `SfmTaskflow::shared()` is process-wide: logical CPU count minus one
(at least one), a 2 GiB working-memory budget, and a 512 MiB scratch estimate per
large stage. Most stage entry points prefer four CPU threads; an explicit mapper
thread request is capped by the actual grant. Parallel numerical sections use
cached, grant-sized Rayon pools, not a reconfigured global Rayon pool. Borrowed
callbacks and native-call coordination stay on the original calling thread.

For explicit budgets, share one executor across contexts:

```rust,ignore
let runtime = std::sync::Arc::new(rustscan_taskflow::Runtime::new(config)?);
let executor = rustsfm::SfmTaskflow::new(runtime, 512 * 1024 * 1024)?;
let mut task = rustsfm::SfmTaskContext::new(&control, &mut event_sink)
    .with_taskflow(executor.clone());
rustsfm::run_reconstruction_with_task(&mapper_config, &mut task)?;
```

CPU-feature and Ceres adapters remain supported. A context combines their scratch
estimates using the maximum and requires the **same Runtime** for all bindings,
including `MapperConfig::ba_taskflow`. Mixed runtimes fail explicitly rather than
silently bypassing a budget. When nested in a reserved pipeline, feature
extraction uses the parent stage instead of submitting another per-image graph.

### Resource-control boundaries

- GPU stages share the conservative exclusive key `rustsfm/default-gpu` and
  retain it until tracked GPU submissions finish, including error/unwind paths.
  This is not per-device VRAM accounting or multi-GPU placement. A coarse stage
  can hold this key during intervening CPU work, trading utilization for safety.
- CPU/GPU backend selection still follows existing options. Neither automatic
  backend routing nor Taskflow's optional system-pressure monitor is enabled by
  default. Hosts may drive `Runtime::set_budget()` or opt into `system-monitor`.
  Budget changes affect new admissions, not already-running native calls.
- Memory figures are cooperative estimates, **not allocator/RSS limits**. Input
  preparation, resident models and long-lived outputs may be outside the ledger.
  Separate processes do not share the process-wide runtime.
- Low-level standalone numerical APIs remain callable directly. In particular,
  `try_refine_bundle_adjustment` with `taskflow: None` outside a stage calls Ceres
  directly. Raw GPU kernels are not separate DAG nodes.
- Do not call synchronous pipeline/admission APIs from the same Runtime's worker
  pool, or from a bounded Rayon numerical section: stage thread-local state is
  caller-local, not automatically propagated to all Rayon workers.
- Viewer pause/cancel and RustSFM now share the same control state, including
  while waiting for resources without progress events. Native solves are still
  cooperative, not forcibly interrupted.

This integration controls admission; it is not evidence of an end-to-end speedup.

## Optional image-only pair profiling

Set `RUSTSFM_PROFILE_PAIRS=1` at process startup and supply `--summary-json` to
`reconstruct --local-matching` to include `pair_profile_summary` and per-candidate
`pair_profile {JSON}` entries in `ReconstructionSummary.debug_log`. Profiling is
off by default and does not change matching options or resource budgets.

The records include image names, input descriptor counts, matches, returned
geometry inliers, acceptance before graph-level filters, matching time, geometry
time, combined guided matching/reverification time, and total pair time. When no
geometry is returned, `inliers=0` is a sentinel, not a count of rejected model
inliers. The uint8 SIFT backend additionally separates descriptor preparation,
index construction and forward/reverse search; `sift.uint8_backend=false` means
those nested counters are unavailable. Input descriptor counts do not represent
the reduced wide-descriptor subset used by special ring-bridge candidates.

Records are serialized on the caller after parallel work, not logged in search
or RANSAC loops. Their durations are **overlapping wall-clock spans, not CPU
time**: outer pair parallelism and inner descriptor parallelism share Rayon, so
suspended spans can include other tasks' execution. Do not add them to the stage
wall time or treat a single slow span as exclusive computation by that pair.
This diagnostic covers `build_pair_graph`, not database-reuse or GPU routes.

## Ceres BA admission

Standalone Ceres BA can opt into an adapter, while BA inside admitted pipelines
already inherits stage scheduling. Incremental local/global BA, final sequence
BA, generalized rig pose refinement and the global mapper propagate the adapter.

```rust,ignore
let runtime = std::sync::Arc::new(rustscan_taskflow::Runtime::new(config)?);
let features = rustsfm::CpuFeatureTaskflow::new(runtime.clone(), 512 * 1024 * 1024, 32)?;
let ba = rustsfm::CeresBaTaskflow::new(runtime.clone(), 2, 512 * 1024 * 1024)?;
let mut task = rustsfm::SfmTaskContext::new(&control, &mut event_sink)
    .with_cpu_feature_taskflow(&features)
    .with_ceres_ba_taskflow(&ba);
rustsfm::run_reconstruction_with_task(&mapper_config, &mut task)?;
// The same context bindings work with the sequence/keyframe entry points.
```

The per-image feature binding covers standalone database/selected-image
extraction. Image-only fallback extraction and matching inside a reconstruction
use the enclosing stage allowance. The BA binding can additionally cap each
solve below the parent stage's CPU grant.

For direct BA, use `ba.refine(frames, reconstruction, options)`, or set
`BundleAdjustmentOptions::taskflow` and call `ba::try_refine_bundle_adjustment`.
`MapperConfig::ba_taskflow` and `GlobalReconstructionOptions::ba_taskflow` are
also available for callers without `SfmTaskContext`. Controlled mapper/sequence
entry points bind the adapter to their own `SfmTaskControl`; standalone callers
can use `ba.clone().with_control(control.clone())`.

- BA requests a scalable CPU allowance with minimum one thread, capped by the
  adapter's per-BA maximum and any positive `options.num_threads`. The **actual
  grant**, never automatic all-core selection, is passed to Ceres.
- The existing 50,000-residual multithreading gate is unchanged. A cheap
  conservative bound avoids reserving multiple threads for clearly small BAs;
  filtering may still cause the solver to use fewer threads than reserved.
- The per-BA cap is a concurrency policy: requesting every CPU for the first BA
  can serialize a second BA. Grants do not rebalance/resize running solves.
- The calling thread retains borrowed frames/reconstruction and runs Ceres;
  the admitted dispatch worker parks until completion. No full-model copy or
  additional per-BA Rayon compute pool is created by the adapter.
- Scratch memory and CPU remain reserved through Ceres setup, solve, writeback,
  covariance computation and native cleanup. The supplied memory estimate is
  **not a hard allocator limit**; resident input/model storage and long-lived
  returned covariance data are not charged by this scratch-only adapter.
- Queued calls check pause/cancel approximately every 5ms. Running native Solve
  is not force-interrupted. A stop observed at the pre-writeback checkpoint
  discards Ceres's separate solution storage, leaving the reconstruction
  unchanged. A later stop, after writeback has begun, is handled at the next
  pipeline boundary; completed updates are not rolled back.
- The reservation is released on return/error/Rust unwind, not when Solve starts
  or cancellation is requested. This does not recover from native crashes or
  guarantee rollback of arbitrary panics during mutation. Do not call this
  synchronous adapter from a worker of the same Runtime: the coordinator must
  remain outside its dispatch pool.
- `BundleAdjustmentReport::solver_num_threads` records the limit passed to
  Ceres (not measured OS utilization). `scheduling` records the grant and queue
  time; existing BA execution timings exclude queue time.

`try_refine_bundle_adjustment` preserves admission/stop errors. The legacy
`refine_bundle_adjustment` Option API logs such errors and returns None; there is
no silent unscheduled fallback. Mapper paths retain their existing BA failure/
skip policy. Invalid capacities and queue-admission failures are not retried.

**Native math libraries are a separate concurrency boundary.** Ceres's thread
option does not universally cap BLAS/OpenMP/Accelerate workers. Configure those
libraries at process startup (before native initialization), rather than changing
global thread settings while other workflows are running. For example, on the
local macOS setup the demonstration was run with:

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  cargo run -p rustsfm --release --example taskflow_ba --offline
```

`taskflow_ba` runs two real Ceres problems plus CPU SIFT, with four global CPU
slots and at most two per BA. A local run granted **2 + 1 threads to BA and 1 to
SIFT**. Both 40,000-residual problems reduced their cost, SIFT returned features,
and reservations returned to zero. The example explicitly lowers the small-BA
threading gate to exercise scalable grants; production defaults are unchanged.
This is an admission/numerical demonstration, not a performance comparison or
proof that full reconstruction/UI stalls are resolved.

```sh
cargo test -p rustsfm --release --lib ba::taskflow --offline
cargo test -p rustsfm --release --lib taskflow_ceres_and_sift --offline
cargo test -p rustsfm --release --lib ba::ceres --offline
cargo test -p rustsfm --release --test sequence_registration --offline
cargo check -p rustsfm --release --no-default-features --all-targets --offline
```

## Optional per-image feature DAG

Standalone CPU database extraction can select a finer-grained per-image DAG
instead of the default coarse stage reservation. GPU extraction, matching and
Viewer reconstruction use stage scheduling; this option does not replace their
algorithms or claim a faster SIFT kernel.

```sh
# Use an already imported database and its matching image directory.
cargo run -p rustsfm --release --bin rustsfm -- extract-features \
  --database path/to/database.db --images path/to/images \
  --taskflow --taskflow-cpu-threads 4 \
  --taskflow-memory-mib 2048 --taskflow-image-memory-mib 512

# Self-contained comparison: generated temporary images/databases, no dataset needed.
cargo run -p rustsfm --release --example taskflow_features --offline
```

`--taskflow` selects the per-image DAG and conflicts with `--use-gpu`. Its CPU default is logical
CPU count minus one (at least one); I/O slots equal the configured CPU budget.
The per-image estimate must cover peak decode/SIFT scratch, conversion buffers
and output, including original-resolution decoding before SIFT rescaling. It is
**not an allocator limit or a proven upper bound**: increase it for large images
or covariant/DSP extraction, and ensure the total budget can admit at least one
image. The full estimate stays reserved until that image is committed/discarded.
I/O is conservatively reserved for the whole fused decode/extraction node.

Each window (32 images in the CLI) builds a static DAG:

```text
extract image 1 -> commit image 1 -> commit image 2 -> ...
extract image 2 -----------------> commit image 2
```

Extraction nodes run with one CPU grant each (the VLFeat build disables OpenMP);
parallelism is across images and workflows. Commit nodes reserve CPU/I/O and a
canonical-database-path exclusive key while the original calling thread performs
the transaction and progress callback. This keeps non-Send/borrowed event sinks
compatible and preserves alphabetical commit order and per-image atomicity.
Completed images can be written before the whole window finishes: if a later
extraction fails, earlier committed images remain, rather than rolling back the
window. Pause/cancel after a progress event prevents the next commit. Already
running native SIFT calls are drained before returning; they cannot be preempted
mid-image. Waiting graphs check the SFM control at approximately 5ms intervals.
Callback panics also cancel/drain the graph and release waiting writers.

For **multiple workflows in one process**, create one runtime and inject an
adapter into every relevant `SfmTaskContext`:

```rust,ignore
let runtime = std::sync::Arc::new(rustscan_taskflow::Runtime::new(config)?);
let features = rustsfm::CpuFeatureTaskflow::new(
    runtime.clone(), 512 * 1024 * 1024, 32,
)?;
let mut task = rustsfm::SfmTaskContext::new(&control, &mut event_sink)
    .with_cpu_feature_taskflow(&features);
rustsfm::extract_features_to_database_with_task(&database, &images, &options, &mut task)?;
```

Standalone selected-image extraction also supports the per-image DAG. Inside a
sequence/keyframe stage it inherits the parent reservation, using one image at a
time if the adapter supplied a per-image memory estimate. Custom borrowed
extractors and GPU extraction use stage admission. Do not call this synchronous
adapter from a worker of the same runtime: its coordinator must remain outside
the runtime's compute workers.
The host should own the runtime until all coordinators finish. Separate CLI
processes do not share a budget. SQLite work outside this adapter is not covered
by its exclusive key (SQLite's own locking still applies).

The host may drive `Runtime::set_budget()` or use Taskflow's optional
`system-monitor`; the CLI does not start a monitor automatically. Reducing
a budget affects new admission only. A zero/insufficient budget can leave graphs
waiting until recovery or cancellation. Queue admission errors are returned to
the caller, not retried silently.

### Validation and historical pilot results

The current `taskflow_features` example compares **stage vs per-image DAG** using
one shared four-CPU Runtime, two workflows and synthetic inputs. Both modes are
scheduled. Stage and per-image scratch estimates are printed separately; they
are not measurements of actual memory usage. Database parity is checked outside
the timed region. Do not compare its timings directly with the historical
unscheduled baseline below.

- Release tests cover exact default-stage/per-image-DAG database parity under a one-image
  memory budget, ordered callbacks on the caller thread, pause after one commit,
  cancellation while admission is paused, failed-transaction rollback, missing
  input, callback-panic cleanup, and two workflows sharing a runtime.
- Both default VLFeat and dependency-minimal CPU builds are tested. Existing
  feature-extraction and task-control API tests continue to pass.
- **Historical, before default stage integration:** on the local Apple M5 Max,
  the then-current example's two concurrent workflows × eight
  synthetic 512×512 images, with four shared CPU workers, took **1.219–1.246s**
  on the legacy path and **1.221–1.247s** through Taskflow across three warm
  measurements in alternating order. All keypoints and descriptor bytes matched;
  Taskflow reservations returned to zero. This small experiment shows comparable
  throughput, not a statistically established speedup or UI-latency improvement.

```sh
cargo test -p rustsfm --release --lib feature_extraction::tests --offline
cargo test -p rustsfm --release --no-default-features --lib feature_extraction::tests::taskflow --offline
cargo test -p rustsfm --release --bin rustsfm cli::tests --offline
cargo test -p rustsfm --release --test task_control --offline
cargo test -p rustsfm --release --test taskflow_features --offline
```

This crate intentionally does not call the external `colmap` executable. The
current implementation follows the COLMAP mapper shape: database/cache loading,
feature extraction, geometric verification, initial pair selection,
incremental PnP registration, track triangulation, and COLMAP text export.

By default the mapper now behaves like a COLMAP-style database-first pipeline:
it auto-discovers `database.db` next to the image root or input directory and
only falls back to local matching when `--local-matching` is set explicitly.
Two-view verification now preserves COLMAP-style geometry configs for
calibrated, uncalibrated, planar/panoramic, watermark, and multiple-model cases
while filtering ambiguous watermark/multiple pairs out of the default mapper
graph.
Rust-estimated verified pair geometry can be written back to the COLMAP
`two_view_geometries` table with `--write-two-view-geometries`; this is opt-in
so the default reconstruction path does not mutate the input database.
The local-matching fallback can also create a full COLMAP-style SQLite database
(cameras, images, keypoints, descriptors, matches, two-view geometries) with
`--local-matching --write-database [--database path/to/database.db]`.
Generalized rig relative/absolute pose now follows COLMAP's panoramic-rig
branches and uses PoseLib's GR6P/GP3P minimal solvers plus a COLMAP-derived
GR8P local refit bridge for non-panoramic rigs in default builds, with
BA-backed pose-only generalized absolute-pose refinement for rig frames and
COLMAP-style fallback to central PnP when a rig camera still needs focal-length
estimation. PoseLib v2.0.5 is pinned as the `third_party/PoseLib` submodule.
Bundle adjustment exclusively uses Ceres; the hand-written Native BA backend
has been removed. Initialize native dependencies and run the default solver
tests with:

```bash
git submodule update --init --recursive
cargo test -p rustsfm --release --lib
```

Existing clones can also run `./scripts/setup_rustsfm_deps.sh` to bootstrap
RustSFM's native dependencies. The dependency-minimal library still compiles,
but bundle adjustment is unavailable without `ceres-ba` and SIFT matching is
unavailable without `gpu-wgpu`. A missing GPU feature returns
`SIFT matching requires RustSFM to be compiled with the gpu-wgpu feature`;
it does not fall back to a CPU matcher or an empty result. `--no-default-features`
is a compile gate, not a full pipeline test. Matching and sequence tests that
need a real adapter must enable `gpu-wgpu` and `vlfeat-sift` explicitly:

```bash
cargo check -p rustsfm --release --no-default-features --all-targets
cargo test -p rustsfm --lib --features gpu-wgpu,vlfeat-sift -- --test-threads=1
cargo test -p rustsfm --test sequence_registration --features gpu-wgpu,vlfeat-sift -- --test-threads=1
```

The `real_colmap_sparse_*` parity tests require a compatible external
`test_data/flowers2_colmap` 24-image sparse fixture. That runtime tree is not
part of the default checkout path used by ordinary tests, so these cases stay
`#[ignore]`d. The pinned reference text lives under
`test_data/fixtures/flowers2_colmap_ref_text_20260630` (SHA-256 verified).
Provision it, then run the ignored suite explicitly:

```bash
./scripts/provision_flowers2_colmap_fixture.sh
cargo test -p rustsfm --lib -- --ignored
```

Opt-in CI: workflow_dispatch, or open a PR labeled `flowers2-parity` (job
`rustsfm-flowers2-parity-opt-in`).
Incremental registration is absolute-pose driven with COLMAP-style next-image
ranking methods, registration trial bookkeeping, inlier-ratio checks, and
pose-only reprojection refinement before accepting new images. Filtered or
previously failed registration units are retried from a lower-priority bucket
like COLMAP, and max-trial checks are applied over the full frame registration
unit.
Successful registrations now trigger a local bundle-adjustment pass over the
new image, its strongest shared-point neighbors selected with
triangulation-angle checks, and new or short-track local 3D points; local BA is
followed by track merge/completion, new-image completion, and track filtering.
Global bundle adjustment now runs after initialization, on COLMAP-style
registered-image/point growth triggers, and at finalization when the model has
changed since the last global pass; each pass fixes two registered images as the
global gauge, follows COLMAP's focal/principal-point/extra-parameter refinement
defaults, accepts `--ba-constant-camera-id`, and performs track
completion/merge/filter post-processing.
Triangulation applies angular/reprojection gates, re-estimates track geometry
after continuation/merge, and filters negative-depth, reprojection,
triangulation-angle, bogus-camera, and short-track outliers during the
incremental loop. After COLMAP's 20-registration-unit warm-up threshold, mapper
cleanup also deregisters full frames/images whose registered cameras are bogus
or whose frame has no remaining point3D observations, keeping the frame/rig and
per-camera registration counters in sync with the reconstruction state.
