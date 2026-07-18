# RustSFM wgpu Feature Pipeline Design

## Goal

Accelerate RustSFM on Apple Silicon without relying on parallel CPU feature extraction or
descriptor matching. The GPU path uses wgpu over Metal for SIFT, descriptor matching, and
batched RANSAC model scoring while preserving RustSFM's COLMAP-compatible database contract.

The CPU remains responsible for minimal geometric solvers, local model refinement, mapper
control flow, track management, triangulation decisions, and Ceres bundle adjustment.

## Baseline And Motivation

The full `flowers2` run contains 960 images and currently reports:

- SIFT extraction: 2815.8 seconds for 6,612,183 keypoints.
- Feature matching and geometric verification: 1899.2 seconds for 8,577 candidate pairs.
- Incremental mapper: 2664.2 seconds, including 173 local BA runs, 38 global BA runs, and
  734 failed registration attempts.
- Final reconstruction: 175 registered images and 118,126 points.

The extraction and matching workloads are regular and data-parallel. The mapper workload is
dominated by dynamic graph operations, repeated nonlinear optimization, and failure recovery.
This makes a heterogeneous pipeline preferable to a full mapper port.

## Scope

### Included

1. A persistent wgpu device and queue shared by all GPU feature stages in one command.
2. A GPU SIFT implementation covering Gaussian and DoG pyramids, extrema localization,
   orientation assignment, and 128-element descriptors.
3. COLMAP-compatible keypoint coordinates and quantized `u8[128]` descriptors.
4. A tiled GPU two-nearest-neighbor matcher with ratio, distance, and cross-check filtering.
5. Batched GPU residual scoring and inlier counting for Essential, Fundamental, Homography,
   and PnP RANSAC candidates.
6. Explicit CLI selection through COLMAP-compatible `SiftExtraction.use_gpu=1` and
   `SiftMatching.use_gpu=1`, plus equivalent native RustSFM options.
7. Stage-level timing, adapter diagnostics, correctness comparisons, and performance gates.

### Excluded

- GPU minimal-model solvers and nonlinear geometric refinement.
- GPU track graph construction or mapper scheduling.
- Replacing Ceres or its sparse Schur bundle-adjustment solver.
- Affine-shape estimation and domain-size pooling in the first GPU release.
- Silent fallback to CPU when GPU execution was explicitly requested.

Unsupported extraction modes fail before any database mutation with a precise error. The
existing CPU implementation remains available and behaviorally unchanged.

## Architecture

```text
image decode / grayscale preparation (serial CPU)
                       |
                       v
             persistent wgpu context
                       |
             +---------+----------+
             |                    |
             v                    v
       GPU SIFT extraction   descriptor cache
             |                    |
             +---------+----------+
                       v
              tiled GPU matching
                       |
                       v
       CPU minimal-model hypothesis generation
                       |
                       v
        batched GPU residual scoring / inliers
                       |
                       v
      CPU LO-RANSAC, refinement, DB persistence
                       |
                       v
          CPU mapper and Ceres bundle adjustment
```

### Cargo And Runtime Selection

RustSFM gains a `gpu-wgpu` Cargo feature. It is included in the default CLI build so the
COLMAP-compatible `use_gpu=1` switches work without a special binary, while CPU-only builds
can opt out with `--no-default-features` and an explicit CPU feature set.

GPU objects are created once per command, not once per image or pair. Adapter selection
prefers a high-performance adapter and records the backend, adapter name, limits, and
optional timestamp-query support. On macOS, wgpu selects Metal.

The public GPU services are separated by responsibility:

- `WgpuContext`: adapter, device, queue, shared staging buffers, and pipeline cache.
- `WgpuSiftExtractor`: grayscale image to `SiftFeatures`.
- `WgpuSiftMatcher`: descriptor pairs or pair batches to tentative matches.
- `WgpuModelScorer`: model/correspondence batches to support summaries and winning masks.

The services expose synchronous RustSFM-facing methods because the existing database and CLI
pipelines are synchronous. Internally they submit asynchronous GPU work and wait only at
stage boundaries where CPU results are required.

## GPU SIFT

### Input And Pyramid

The existing serial image decode, grayscale conversion, `max_image_size` handling, and
coordinate rescaling remain authoritative. The grayscale image is uploaded as normalized
`f32` data. `first_octave=-1` performs GPU upsampling and the required initial blur.

Each octave uses `octave_resolution + 3` Gaussian levels and
`octave_resolution + 2` DoG levels. Gaussian convolution is implemented as separable
horizontal and vertical compute passes with reusable ping-pong buffers. Octaves are streamed
through a reusable allocation instead of retaining every level for the whole image.

### Detection And Localization

A detection kernel compares every eligible DoG sample with its 26 spatial/scale neighbors.
Candidates are appended into a bounded storage buffer using atomics. A localization kernel
then performs the standard three-dimensional quadratic refinement, rejecting candidates by:

- configured peak/contrast threshold;
- configured edge-response threshold;
- octave and image borders;
- failure to converge within the fixed refinement budget.

An overflow flag is checked after submission. The host retries with a larger candidate
buffer up to a documented memory ceiling; exceeding the ceiling returns an error instead of
silently dropping keypoints.

### Orientation And Descriptor

Each localized keypoint receives a 36-bin orientation histogram. Smoothed local maxima above
the SIFT peak ratio create up to `max_num_orientations` oriented keypoints. Upright mode emits
one zero-angle keypoint.

Descriptor kernels build the standard 4x4x8 trilinearly interpolated histogram, normalize,
clip, and renormalize it. L2 and L1Root normalization follow the existing
`SiftDescriptorNormalization` contract. Quantization uses the current COLMAP conversion:
round `clamp(value, 0, 1) * 512` to `u8`.

Heavy numeric work remains on the GPU. The CPU performs only bounded readback, deterministic
feature ordering/truncation, Rust type construction, and database serialization. No Rayon
extraction is used on the GPU path.

## GPU Descriptor Matching

WGSL has no portable byte-vector arithmetic suitable for the existing descriptor layout, so
each `u8[128]` descriptor is packed into 32 `u32` words for upload. Kernels unpack four lanes
per word and accumulate squared integer L2 distance.

The matcher never materializes an `N x M` distance matrix. It scans target descriptors in
workgroup-sized tiles and retains only the best and second-best candidates for each query.
Filtering compares squared values for both maximum distance and Lowe ratio, avoiding square
roots. A reverse dispatch and a cross-check kernel produce mutual matches when requested.

Pair execution is GPU-batched and replaces the current nested Rayon path only when GPU
matching is selected. Descriptor uploads use an LRU cache bounded by a configurable memory
budget. Pair ordering and final match ordering are deterministic, including tie-breaking by
descriptor index. `max_num_matches` is applied after deterministic ranking.

## Batched RANSAC Scoring

CPU code continues to generate candidate Essential, Fundamental, Homography, and PnP models
with the existing seeded samplers and minimal solvers. Candidates are accumulated in bounded
chunks and sent to model-specific scoring kernels.

Each kernel evaluates all model/correspondence combinations, applies the existing residual
and threshold definition, and reduces:

- inlier count;
- truncated or summed residual used for support tie-breaking;
- model index.

Only compact support summaries are read back for every candidate. The full inlier mask is
generated and read back for the selected model, after which existing CPU local optimization,
cheirality tests, triangulation checks, and nonlinear refinement continue unchanged.

Adaptive RANSAC stopping is evaluated at chunk boundaries. Seeded candidate order is stable,
and ties select the earliest model, so runs are deterministic. Chunking may evaluate extra
hypotheses but may not select a hypothesis beyond the trial budget calculated at the previous
boundary.

GPU scoring must be integrated through a narrow scorer interface so the existing CPU scorer
remains the reference implementation and fallback when GPU use was not requested.

## Mapper Boundary

The mapper stays on CPU. In particular, Ceres bundle adjustment retains double-precision
residual evaluation and sparse Schur solving. Portable wgpu/WGSL on Apple GPU hardware does
not provide a suitable high-performance double-precision sparse solver, and replacing Ceres
with an f32 dense or custom sparse solver would create a substantial convergence risk.

The GPU work does not attempt to solve the observed 734 registration failures. Mapper timing
instrumentation will be retained so scheduling, repeated failed registrations, and BA
frequency can be addressed independently after feature-pipeline acceleration is measured.

## CLI And Compatibility

- `SiftExtraction.use_gpu=1` selects `WgpuSiftExtractor`.
- `SiftMatching.use_gpu=1` selects `WgpuSiftMatcher` and enables GPU model scoring by default.
- Native commands gain explicit GPU switches with the same semantics.
- `use_gpu=0` preserves the existing VLFeat/lowe-sift and CPU matcher paths.
- A missing adapter, missing required limits, device loss, unsupported SIFT option, shader
  failure, or bounded-buffer overflow returns contextual `anyhow::Error`.
- Explicit GPU selection never silently falls back to CPU.

Database writes retain the current failure-atomic behavior: all GPU work and output
validation for a unit of work complete before destructive replacement occurs.

## Testing Strategy

Implementation follows red-green-refactor. GPU-independent validation and packing logic use
normal unit tests. Compute kernels use tiny deterministic fixtures compared with scalar CPU
references. Hardware integration tests request a fallback adapter where available and report
a clear skip only when no compatible adapter exists.

### Kernel Tests

- Gaussian impulse and constant-image responses.
- DoG subtraction and 26-neighbor extrema classification.
- Localization acceptance, contrast rejection, and edge rejection.
- Orientation histogram peak selection and orientation duplication limits.
- Descriptor normalization, clipping, RootSIFT conversion, and `u8` quantization.
- Tiled best-two matching, ratio boundary, distance boundary, ties, and cross-check.
- Essential/Fundamental/Homography/PnP residuals and reduction tie-breaking.

### Integration Tests

- Empty, tiny, constant, portrait, landscape, and odd-sized images.
- CPU/GPU extraction comparison on deterministic textured fixtures.
- CPU/GPU tentative match and verified-geometry comparison.
- COLMAP SQLite keypoint/descriptor dimensions and byte layout.
- CLI parsing for GPU enabled, disabled, unavailable, and unsupported modes.
- Repeated extraction with one persistent context to detect resource leaks or recreation.

GPU floating-point results are not required to be byte-identical to VLFeat. Acceptance is
based on keypoint localization tolerance, descriptor similarity, mutual-match agreement, and
verified geometric support. Exact thresholds are recorded in the implementation plan after
a small fixture spike establishes realistic Metal and CPU reference variance.

## Performance And Quality Gates

Stage timings separate decode, upload, pyramid, detection, descriptor, matching, RANSAC
scoring, readback, and database IO. Timestamp queries are used when supported; wall-clock
timings remain available everywhere.

The GPU path is not declared production-ready unless:

1. It produces valid COLMAP-compatible data and passes all correctness tests.
2. Warmed GPU SIFT is materially faster than single-CPU extraction on representative images.
3. GPU matching plus scoring is materially faster than the existing CPU path on a
   representative pair batch.
4. A fixed `flowers2` subset preserves verified-pair and reconstruction quality within the
   tolerances established by the implementation plan.
5. The full `flowers2` benchmark completes without candidate overflow, device loss, silent
   fallback, or unbounded GPU memory growth.

The default CPU path remains the correctness reference until these gates pass. Enabling the
Cargo feature by default makes the GPU implementation available; it does not make runtime
GPU selection implicit.

## Delivery Sequence

1. Add timing boundaries and persistent wgpu infrastructure.
2. Implement and validate GPU pyramid, detection, orientation, and descriptors.
3. Integrate extraction into native and COLMAP-compatible CLI paths.
4. Implement tiled descriptor matching and deterministic filtering.
5. Implement batched model scorers and integrate them behind the scorer interface.
6. Run correctness comparisons and progressively larger `flowers2` benchmarks.
7. Address mapper scheduling and registration failures as a separate, measured change.
