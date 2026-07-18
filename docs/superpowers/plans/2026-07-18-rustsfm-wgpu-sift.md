# RustSFM wgpu SIFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a real Metal-backed wgpu SIFT extractor that produces RustSFM and COLMAP-compatible keypoints and descriptors without parallel CPU extraction.

**Architecture:** Add one persistent `WgpuContext`, stream one octave at a time through compute shaders, and read back only compact oriented keypoints plus descriptors. Keep serial image decode, deterministic result ordering, Rust object construction, and SQLite writes on CPU; preserve the current VLFeat path when GPU use is disabled.

**Tech Stack:** Rust 2021, wgpu 29, WGSL compute shaders, bytemuck, pollster, anyhow, lowe-sift descriptor types, COLMAP SQLite compatibility.

---

## Scope Boundary

This plan implements the first independently testable subsystem from the approved feature
pipeline design: GPU SIFT extraction. GPU descriptor matching and GPU RANSAC scoring receive
separate implementation plans after this plan establishes the shared wgpu context and output
contracts.

## File Map

- `RustSFM/Cargo.toml`: `gpu-wgpu` feature and optional GPU dependencies.
- `RustSFM/src/gpu/mod.rs`: public GPU capability and extractor exports.
- `RustSFM/src/gpu/context.rs`: adapter/device/queue creation and synchronous readback.
- `RustSFM/src/gpu/sift/mod.rs`: high-level octave execution and `SiftFeatures` construction.
- `RustSFM/src/gpu/sift/plan.rs`: validated octave dimensions, sigma schedule, and capacities.
- `RustSFM/src/gpu/sift/types.rs`: host/WGSL ABI structs and deterministic output conversion.
- `RustSFM/src/gpu/shaders/sift_pyramid.wgsl`: resize, Gaussian, downsample, and DoG kernels.
- `RustSFM/src/gpu/shaders/sift_detect.wgsl`: extrema detection and subpixel localization.
- `RustSFM/src/gpu/shaders/sift_descriptor.wgsl`: orientation assignment and 128-bin descriptors.
- `RustSFM/src/feature/sift.rs`: backend selection and GPU-compatible option validation.
- `RustSFM/src/feature/feature_extraction.rs`: one persistent extractor per database command.
- `RustSFM/src/cli/mod.rs`: native GPU extraction flags.
- `RustSFM/src/cli/commands.rs`: COLMAP/native CLI routing and error reporting.

### Task 1: Feature Gate And GPU Option Validation

**Files:**
- Modify: `RustSFM/Cargo.toml`
- Modify: `RustSFM/src/feature/sift.rs`
- Modify: `RustSFM/src/gpu/mod.rs`

- [ ] **Step 1: Write failing generic and backend validation tests**

Add these tests under `sift::tests` and `gpu::tests`:

```rust
#[cfg(feature = "gpu-wgpu")]
#[test]
fn generic_sift_options_allow_explicit_gpu_selection() {
    let options = SiftExtractionOptions {
        use_gpu: true,
        ..SiftExtractionOptions::default()
    };
    assert!(options.check().is_ok());
}

#[cfg(feature = "gpu-wgpu")]
#[test]
fn gpu_sift_rejects_covariant_modes_before_device_creation() {
    let options = SiftExtractionOptions {
        use_gpu: true,
        estimate_affine_shape: true,
        ..SiftExtractionOptions::default()
    };
    let error = validate_gpu_sift_options(&options).unwrap_err();
    assert!(error.to_string().contains("affine shape"));
}
```

- [ ] **Step 2: Run tests and verify the old placeholder rejects GPU use**

Run: `cargo test -p rustsfm --lib generic_sift_options_allow_explicit_gpu_selection --features gpu-wgpu -- --nocapture`

Run: `cargo test -p rustsfm --lib gpu_sift_rejects_covariant_modes_before_device_creation --features gpu-wgpu -- --nocapture`

Expected: FAIL because `gpu-wgpu` and `validate_gpu_sift_options` do not exist and generic
validation still rejects `use_gpu=true`.

- [ ] **Step 3: Add the feature and narrow backend validation**

Use this Cargo feature shape:

```toml
[features]
default = ["ceres-ba", "vlfeat-sift", "gpu-wgpu"]
gpu-wgpu = ["dep:bytemuck", "dep:pollster", "dep:wgpu"]

[dependencies]
bytemuck = { version = "1.25", features = ["derive"], optional = true }
pollster = { version = "0.3", optional = true }
wgpu = { version = "29.0.1", features = ["wgsl"], optional = true }
```

Remove the generic `use_gpu` rejection from `SiftExtractionOptions::check`. Add this function
to `gpu/mod.rs`:

```rust
pub fn validate_gpu_sift_options(options: &SiftExtractionOptions) -> Result<()> {
    if options.estimate_affine_shape {
        bail!("wgpu SIFT does not support affine shape estimation");
    }
    if options.domain_size_pooling {
        bail!("wgpu SIFT does not support domain-size pooling");
    }
    if options.force_covariant_extractor {
        bail!("wgpu SIFT does not support the covariant extractor");
    }
    if options.first_octave < -1 {
        bail!("wgpu SIFT first_octave must be >= -1");
    }
    Ok(())
}
```

- [ ] **Step 4: Run focused tests**

Run: `cargo test -p rustsfm --lib generic_sift_options_allow_explicit_gpu_selection --features gpu-wgpu -- --nocapture`

Run: `cargo test -p rustsfm --lib gpu_sift_rejects_covariant_modes_before_device_creation --features gpu-wgpu -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add RustSFM/Cargo.toml Cargo.lock RustSFM/src/feature/sift.rs RustSFM/src/gpu/mod.rs
git commit -m "feat(rustsfm): add wgpu SIFT feature gate"
```

### Task 2: Persistent wgpu Context And Readback

**Files:**
- Create: `RustSFM/src/gpu/context.rs`
- Modify: `RustSFM/src/gpu/mod.rs`

- [ ] **Step 1: Write a hardware-capability smoke test**

```rust
#[cfg(feature = "gpu-wgpu")]
#[test]
fn wgpu_context_reports_a_real_adapter_when_available() -> Result<()> {
    let Some(context) = WgpuContext::try_new_optional()? else {
        eprintln!("skipping GPU smoke test: no compatible adapter");
        return Ok(());
    };
    assert!(!context.capabilities().device_name.trim().is_empty());
    assert_eq!(context.capabilities().backend, GpuBackendKind::Wgpu);
    Ok(())
}
```

- [ ] **Step 2: Run test and verify it fails to compile**

Run: `cargo test -p rustsfm --lib wgpu_context_reports_a_real_adapter_when_available --features gpu-wgpu -- --nocapture`

Expected: FAIL because `WgpuContext` is undefined.

- [ ] **Step 3: Implement persistent context creation**

Define the context with shared handles and a synchronous wait helper:

```rust
#[derive(Debug)]
pub struct WgpuContext {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    capabilities: GpuSiftCapabilities,
}

impl WgpuContext {
    pub fn try_new() -> Result<Arc<Self>> {
        Self::try_new_optional()?.context("no compatible wgpu adapter is available")
    }

    pub fn try_new_optional() -> Result<Option<Arc<Self>>> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Option<Arc<Self>>> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            Ok(adapter) => adapter,
            Err(_) => return Ok(None),
        };
        let info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("rustsfm-wgpu-sift"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults()
                    .using_resolution(adapter.limits()),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await
            .context("failed to request the wgpu SIFT device")?;
        Ok(Some(Arc::new(Self {
            instance,
            adapter,
            device,
            queue,
            capabilities: GpuSiftCapabilities {
                backend: GpuBackendKind::Wgpu,
                device_name: info.name,
            },
        })))
    }

    pub(crate) fn wait(&self) -> Result<()> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("wgpu device wait failed")?;
        Ok(())
    }
}
```

Expose `device`, `queue`, and `capabilities` through read-only methods. Add a bounded
`read_buffer<T: Pod>` helper that copies into a `MAP_READ` staging buffer, maps it, waits,
copies into a `Vec<T>`, and unmaps before returning.

- [ ] **Step 4: Run smoke test on Metal**

Run: `cargo test -p rustsfm --lib wgpu_context_reports_a_real_adapter_when_available --features gpu-wgpu -- --nocapture`

Expected on the target Mac: PASS and a non-empty Apple GPU adapter name.

- [ ] **Step 5: Commit**

```bash
git add RustSFM/src/gpu/context.rs RustSFM/src/gpu/mod.rs
git commit -m "feat(rustsfm): initialize persistent wgpu context"
```

### Task 3: Octave Planning And ABI Types

**Files:**
- Create: `RustSFM/src/gpu/sift/mod.rs`
- Create: `RustSFM/src/gpu/sift/plan.rs`
- Create: `RustSFM/src/gpu/sift/types.rs`
- Modify: `RustSFM/src/gpu/mod.rs`

- [ ] **Step 1: Write deterministic octave-plan tests**

```rust
#[test]
fn octave_plan_matches_sift_level_and_sigma_schedule() {
    let options = SiftExtractionOptions {
        first_octave: -1,
        num_octaves: 4,
        octave_resolution: 3,
        ..SiftExtractionOptions::default()
    };
    let plan = SiftPlan::new(640, 480, &options).unwrap();
    assert_eq!(plan.octaves[0].dimensions(), (1280, 960));
    assert_eq!(plan.octaves[0].gaussian_levels, 6);
    assert_eq!(plan.octaves[0].dog_levels, 5);
    assert!((plan.sigma_step - 2.0f32.powf(1.0 / 3.0)).abs() < 1.0e-6);
    assert_eq!(plan.octaves[1].dimensions(), (640, 480));
}

#[test]
fn octave_plan_stops_before_images_become_too_small() {
    let plan = SiftPlan::new(33, 33, &SiftExtractionOptions::default()).unwrap();
    assert_eq!(plan.octaves.len(), 1);
}
```

- [ ] **Step 2: Run tests and verify missing plan types**

Run: `cargo test -p rustsfm --lib octave_plan_ --features gpu-wgpu -- --nocapture`

Expected: FAIL because `SiftPlan` is undefined.

- [ ] **Step 3: Implement validated plans and POD records**

Use checked pixel-count arithmetic and these stable ABI records:

```rust
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct SiftUniforms {
    pub width: u32,
    pub height: u32,
    pub level: u32,
    pub levels: u32,
    pub sigma: f32,
    pub peak_threshold: f32,
    pub edge_threshold: f32,
    pub octave_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuKeypoint {
    pub x: f32,
    pub y: f32,
    pub sigma: f32,
    pub response: f32,
    pub angle: f32,
    pub octave: i32,
    pub level: i32,
    pub valid: u32,
}

pub(crate) struct SiftPlan {
    pub first_octave: i32,
    pub sigma_step: f32,
    pub octaves: Vec<OctavePlan>,
    pub candidate_capacity: u32,
}
```

`SiftPlan::new` validates non-zero dimensions, limits every buffer to the device storage
binding ceiling, uses `octave_resolution + 3` Gaussian levels and
`octave_resolution + 2` DoG levels, and stops once either side is below 32 pixels.

- [ ] **Step 4: Run plan and layout tests**

Run: `cargo test -p rustsfm --lib octave_plan_ --features gpu-wgpu -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add RustSFM/src/gpu/mod.rs RustSFM/src/gpu/sift
git commit -m "feat(rustsfm): define GPU SIFT octave plan"
```

### Task 4: Gaussian And DoG Pyramid Kernels

**Files:**
- Create: `RustSFM/src/gpu/shaders/sift_pyramid.wgsl`
- Modify: `RustSFM/src/gpu/sift/mod.rs`
- Modify: `RustSFM/src/gpu/sift/types.rs`

- [ ] **Step 1: Write GPU pyramid reference tests**

```rust
#[test]
fn gpu_gaussian_preserves_a_constant_image() -> Result<()> {
    let Some(context) = WgpuContext::try_new_optional()? else { return Ok(()) };
    let pyramid = SiftPyramid::new(context)?;
    let input = vec![0.25f32; 17 * 13];
    let output = pyramid.gaussian_for_test(&input, 17, 13, 1.6)?;
    assert!(output.iter().all(|value| (value - 0.25).abs() < 2.0e-5));
    Ok(())
}

#[test]
fn gpu_dog_is_zero_for_equal_levels() -> Result<()> {
    let Some(context) = WgpuContext::try_new_optional()? else { return Ok(()) };
    let pyramid = SiftPyramid::new(context)?;
    let level = vec![0.75f32; 11 * 9];
    let dog = pyramid.dog_for_test(&level, &level, 11, 9)?;
    assert!(dog.iter().all(|value| value.abs() < 1.0e-7));
    Ok(())
}
```

- [ ] **Step 2: Run tests and verify missing pyramid implementation**

Run: `cargo test -p rustsfm --lib gpu_gaussian --features gpu-wgpu -- --nocapture`

Run: `cargo test -p rustsfm --lib gpu_dog --features gpu-wgpu -- --nocapture`

Expected: FAIL because `SiftPyramid` is undefined.

- [ ] **Step 3: Implement separable Gaussian, downsample, and DoG passes**

The WGSL module contains separate entry points and clamps convolution reads at image edges:

```wgsl
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> destination: array<f32>;
@group(0) @binding(2) var<uniform> params: PyramidParams;

fn sample_clamped(x: i32, y: i32) -> f32 {
    let sx = clamp(x, 0, i32(params.width) - 1);
    let sy = clamp(y, 0, i32(params.height) - 1);
    return source[u32(sy) * params.width + u32(sx)];
}

@compute @workgroup_size(16, 16)
fn dog_subtract(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) { return; }
    let pixel = id.y * params.width + id.x;
    destination[pixel] = source_b[pixel] - source[pixel];
}
```

Generate normalized Gaussian weights on the serial host, upload them to a storage buffer,
and use ping-pong buffers for horizontal and vertical passes. Build incremental blur with
`sqrt(sigma_next^2 - sigma_previous^2)`. Downsample Gaussian level
`octave_resolution` by reading every second pixel for the next octave base.

- [ ] **Step 4: Run pyramid tests**

Run: `cargo test -p rustsfm --lib gpu_gaussian --features gpu-wgpu -- --nocapture`

Run: `cargo test -p rustsfm --lib gpu_dog --features gpu-wgpu -- --nocapture`

Expected: PASS on Metal; clean skip with no adapter.

- [ ] **Step 5: Commit**

```bash
git add RustSFM/src/gpu/sift RustSFM/src/gpu/shaders/sift_pyramid.wgsl
git commit -m "feat(rustsfm): build SIFT pyramid on wgpu"
```

### Task 5: DoG Extrema Detection And Localization

**Files:**
- Create: `RustSFM/src/gpu/shaders/sift_detect.wgsl`
- Modify: `RustSFM/src/gpu/sift/mod.rs`
- Modify: `RustSFM/src/gpu/sift/types.rs`

- [ ] **Step 1: Write synthetic extrema tests**

```rust
#[test]
fn gpu_detector_finds_one_strict_scale_space_maximum() -> Result<()> {
    let Some(context) = WgpuContext::try_new_optional()? else { return Ok(()) };
    let mut dogs = vec![0.0f32; 5 * 9 * 9];
    dogs[(2 * 9 + 4) * 9 + 4] = 1.0;
    let points = detect_test_dog(context, &dogs, 9, 9, 5, 0.01, 10.0)?;
    assert_eq!(points.len(), 1);
    assert!((points[0].x - 4.0).abs() < 1.0e-4);
    assert!((points[0].y - 4.0).abs() < 1.0e-4);
    assert_eq!(points[0].level, 2);
    Ok(())
}

#[test]
fn gpu_detector_rejects_edge_like_response() -> Result<()> {
    let Some(context) = WgpuContext::try_new_optional()? else { return Ok(()) };
    let dogs = synthetic_edge_dog(11, 11, 5);
    assert!(detect_test_dog(context, &dogs, 11, 11, 5, 0.01, 10.0)?.is_empty());
    Ok(())
}
```

- [ ] **Step 2: Run tests and confirm detector is absent**

Run: `cargo test -p rustsfm --lib gpu_detector_ --features gpu-wgpu -- --nocapture`

Expected: FAIL because the detection path is undefined.

- [ ] **Step 3: Implement candidate append and fixed-budget localization**

The first kernel performs the 26-neighbor strict comparison and appends candidate indices:

```wgsl
if (abs(center) >= params.pre_threshold &&
    (strictly_greater_than_all_neighbors(center, x, y, level) ||
     strictly_less_than_all_neighbors(center, x, y, level))) {
    let slot = atomicAdd(&candidate_count, 1u);
    if (slot < params.capacity) {
        candidates[slot] = Candidate(x, y, level, 0u);
    } else {
        atomicStore(&overflow, 1u);
    }
}
```

The localization kernel performs up to five Taylor iterations, solves the symmetric 3x3
system analytically, moves when an offset exceeds 0.5, then applies interpolated contrast and
Hessian edge rejection using `(trace^2 / determinant) < ((r + 1)^2 / r)`. Invalid records
set `valid=0`; compaction occurs before readback.

Host code reads the count and overflow words first. It retries with doubled capacity until
the checked ceiling, then returns `GPU SIFT candidate buffer overflow`.

- [ ] **Step 4: Run detector tests**

Run: `cargo test -p rustsfm --lib gpu_detector_ --features gpu-wgpu -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add RustSFM/src/gpu/sift RustSFM/src/gpu/shaders/sift_detect.wgsl
git commit -m "feat(rustsfm): detect and localize SIFT extrema on wgpu"
```

### Task 6: Orientation Assignment

**Files:**
- Create: `RustSFM/src/gpu/shaders/sift_descriptor.wgsl`
- Modify: `RustSFM/src/gpu/sift/mod.rs`
- Modify: `RustSFM/src/gpu/sift/types.rs`

- [ ] **Step 1: Write orientation tests**

```rust
#[test]
fn gpu_orientation_of_horizontal_ramp_is_zero() -> Result<()> {
    let Some(context) = WgpuContext::try_new_optional()? else { return Ok(()) };
    let image = horizontal_ramp(41, 41);
    let keypoint = GpuKeypoint::for_test(20.0, 20.0, 2.0);
    let oriented = orientations_for_test(context, &image, 41, 41, keypoint, 2)?;
    assert_eq!(oriented.len(), 1);
    assert!(wrapped_angle_distance(oriented[0].angle, 0.0) < 0.08);
    Ok(())
}

#[test]
fn upright_mode_emits_exactly_one_zero_orientation() -> Result<()> {
    let Some(context) = WgpuContext::try_new_optional()? else { return Ok(()) };
    let oriented = upright_orientation_for_test(context, GpuKeypoint::for_test(8.0, 8.0, 1.6))?;
    assert_eq!(oriented.len(), 1);
    assert_eq!(oriented[0].angle, 0.0);
    Ok(())
}
```

- [ ] **Step 2: Run tests and verify orientation kernel is missing**

Run: `cargo test -p rustsfm --lib gpu_orientation --features gpu-wgpu -- --nocapture`

Run: `cargo test -p rustsfm --lib upright_mode_emits --features gpu-wgpu -- --nocapture`

Expected: FAIL.

- [ ] **Step 3: Implement 36-bin orientation histograms**

Assign one workgroup to each keypoint. Accumulate Gaussian-weighted central-difference
gradients into 36 bins, smooth the circular histogram six times, and interpolate peaks with:

```wgsl
let offset = 0.5 * (left - right) / max(left - 2.0 * center + right, -1.0e-12);
let angle = TAU * (f32(bin) + clamp(offset, -0.5, 0.5)) / 36.0;
```

Emit peaks at least `0.8 * max_peak` in descending peak order, capped by
`max_num_orientations`. Preserve deterministic bin-index tie-breaking. Upright mode bypasses
the histogram and emits one zero-angle record.

- [ ] **Step 4: Run orientation tests**

Run: `cargo test -p rustsfm --lib gpu_orientation --features gpu-wgpu -- --nocapture`

Run: `cargo test -p rustsfm --lib upright_mode_emits --features gpu-wgpu -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add RustSFM/src/gpu/sift RustSFM/src/gpu/shaders/sift_descriptor.wgsl
git commit -m "feat(rustsfm): assign SIFT orientations on wgpu"
```

### Task 7: 128-Element Descriptor And Normalization

**Files:**
- Modify: `RustSFM/src/gpu/shaders/sift_descriptor.wgsl`
- Modify: `RustSFM/src/gpu/sift/mod.rs`
- Modify: `RustSFM/src/gpu/sift/types.rs`

- [ ] **Step 1: Write descriptor contract tests**

```rust
#[test]
fn gpu_descriptor_is_finite_nonnegative_and_l2_normalized() -> Result<()> {
    let Some(context) = WgpuContext::try_new_optional()? else { return Ok(()) };
    let image = checkerboard_f32(65, 65, 4);
    let descriptor = descriptor_for_test(context, &image, 65, 65, 32.0, 32.0, 2.0, 0.0)?;
    assert!(descriptor.iter().all(|value| value.is_finite() && *value >= 0.0));
    let norm = descriptor.iter().map(|value| value * value).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 2.0e-4);
    Ok(())
}

#[test]
fn gpu_root_sift_quantization_matches_colmap_rule() {
    let mut values = [0.0f32; 128];
    values[0] = 0.25;
    values[1] = 0.5;
    let normalized = normalize_gpu_descriptor(values, SiftDescriptorNormalization::L1Root);
    let quantized = quantize_gpu_descriptor(&normalized);
    assert_eq!(quantized[0], (normalized[0] * 512.0).round() as u8);
    assert_eq!(quantized[1], 255);
}
```

- [ ] **Step 2: Run tests and verify descriptor path is incomplete**

Run: `cargo test -p rustsfm --lib gpu_descriptor --features gpu-wgpu -- --nocapture`

Run: `cargo test -p rustsfm --lib gpu_root_sift --features gpu-wgpu -- --nocapture`

Expected: FAIL.

- [ ] **Step 3: Implement 4x4x8 trilinear descriptors**

For each oriented keypoint, rotate samples into descriptor coordinates, apply the standard
descriptor Gaussian window, and trilinearly distribute magnitude into spatial and circular
orientation bins. Reduce the workgroup histogram, L2-normalize, clamp every element at 0.2,
and L2-normalize again.

Apply selected output normalization in a second kernel:

```wgsl
if (params.normalization == ROOT_SIFT) {
    let l1 = reduce_sum_128(max(histogram[lane], 0.0));
    histogram[lane] = sqrt(max(histogram[lane], 0.0) / max(l1, 1.0e-12));
    let l2 = sqrt(reduce_sum_128(histogram[lane] * histogram[lane]));
    histogram[lane] = histogram[lane] / max(l2, 1.0e-12);
}
```

Keep float descriptors for `lowe_sift::Descriptor::new` and quantize on the host with the
existing exact `round(clamp(value, 0, 1) * 512)` rule. This host loop is serial and bounded by
`max_num_features`; it is representation conversion, not feature extraction.

- [ ] **Step 4: Run descriptor tests**

Run: `cargo test -p rustsfm --lib gpu_descriptor --features gpu-wgpu -- --nocapture`

Run: `cargo test -p rustsfm --lib gpu_root_sift --features gpu-wgpu -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add RustSFM/src/gpu/sift RustSFM/src/gpu/shaders/sift_descriptor.wgsl
git commit -m "feat(rustsfm): compute SIFT descriptors on wgpu"
```

### Task 8: High-Level Extractor And COLMAP Output

**Files:**
- Modify: `RustSFM/src/gpu/mod.rs`
- Modify: `RustSFM/src/gpu/sift/mod.rs`
- Modify: `RustSFM/src/feature/sift.rs`

- [ ] **Step 1: Write end-to-end extractor contract tests**

```rust
#[test]
fn wgpu_sift_checkerboard_produces_aligned_colmap_outputs() -> Result<()> {
    let Some(context) = WgpuContext::try_new_optional()? else { return Ok(()) };
    let extractor = WgpuSiftExtractor::from_context(context)?;
    let gray = checkerboard_u8(256, 256, 16);
    let options = SiftExtractionOptions {
        use_gpu: true,
        max_num_features: 512,
        ..SiftExtractionOptions::default()
    };
    let features = extractor.extract_grayscale(&gray, 256, 256, &options)?;
    assert!(!features.keypoints.is_empty());
    assert!(features.keypoints.len() <= 512);
    assert_eq!(features.keypoints.len(), features.descriptors.len());
    assert_eq!(features.keypoints.len(), features.colmap_keypoints.len());
    assert_eq!(features.keypoints.len(), features.descriptors_u8.len());
    assert!(features.keypoints.iter().all(|point| {
        point.x().is_finite() && point.y().is_finite() && point.size > 0.0
    }));
    Ok(())
}

#[test]
fn wgpu_sift_constant_image_returns_no_features() -> Result<()> {
    let Some(context) = WgpuContext::try_new_optional()? else { return Ok(()) };
    let extractor = WgpuSiftExtractor::from_context(context)?;
    let features = extractor.extract_grayscale(
        &vec![127; 128 * 96],
        128,
        96,
        &SiftExtractionOptions { use_gpu: true, ..Default::default() },
    )?;
    assert!(features.keypoints.is_empty());
    Ok(())
}
```

- [ ] **Step 2: Run end-to-end tests and verify placeholder failure**

Run: `cargo test -p rustsfm --lib wgpu_sift_ --features gpu-wgpu -- --nocapture`

Expected: FAIL because `WgpuSiftExtractor` still returns the placeholder error.

- [ ] **Step 3: Implement octave streaming and output construction**

Replace the placeholder with:

```rust
pub struct WgpuSiftExtractor {
    context: Arc<WgpuContext>,
    pipelines: SiftPipelines,
    capabilities: GpuSiftCapabilities,
}

impl WgpuSiftExtractor {
    pub fn try_new() -> Result<Self> {
        Self::from_context(WgpuContext::try_new()?)
    }

    pub fn extract_grayscale(
        &self,
        gray: &[u8],
        width: u32,
        height: u32,
        options: &SiftExtractionOptions,
    ) -> Result<SiftFeatures> {
        validate_gpu_sift_options(options)?;
        validate_gray_buffer(gray, width, height)?;
        let plan = SiftPlan::new(width, height, options)?;
        let mut records = self.run_octaves(gray, &plan, options)?;
        records.sort_by(gpu_feature_order);
        records.truncate(options.max_num_features);
        Ok(records_to_sift_features(records, options.normalization))
    }
}
```

Sort by descending absolute octave scale, descending response, octave, level, y, x, and
orientation. Convert coordinates from octave space to the resized input image. Populate
`KeyPoint`, `ColmapKeypoint::from_scale_orientation`, `Descriptor`, and `u8[128]` in the same
order. Return empty aligned vectors for featureless images.

- [ ] **Step 4: Run all GPU SIFT tests**

Run: `cargo test -p rustsfm --lib gpu_ --features gpu-wgpu -- --nocapture`

Run: `cargo test -p rustsfm --lib wgpu_sift_ --features gpu-wgpu -- --nocapture`

Expected: PASS on Metal with no validation errors.

- [ ] **Step 5: Commit**

```bash
git add RustSFM/src/gpu RustSFM/src/feature/sift.rs
git commit -m "feat(rustsfm): expose complete wgpu SIFT extractor"
```

### Task 9: Persistent Database Extraction Integration

**Files:**
- Modify: `RustSFM/src/feature/feature_extraction.rs`
- Modify: `RustSFM/src/feature/sift.rs`

- [ ] **Step 1: Write backend routing and persistence tests**

```rust
#[test]
fn extraction_backend_name_reports_wgpu_for_gpu_options() {
    let options = SiftExtractionOptions { use_gpu: true, ..Default::default() };
    assert_eq!(sift_backend_name(&options), "wgpu");
}

#[test]
fn gpu_database_extraction_reuses_one_backend() -> Result<()> {
    let Some(context) = WgpuContext::try_new_optional()? else { return Ok(()) };
    let fixture = two_checkerboard_database_fixture()?;
    let extractor = WgpuSiftExtractor::from_context(context)?;
    let report = extract_features_to_database_with_extractor(
        &fixture.database,
        &fixture.images,
        &SiftExtractionOptions { use_gpu: true, ..Default::default() },
        &extractor,
    )?;
    assert_eq!(report.backend, "wgpu");
    assert_eq!(report.image_count, 2);
    assert!(report.total_keypoints > 0);
    Ok(())
}
```

- [ ] **Step 2: Run tests and verify current database loop always calls CPU extraction**

Run: `cargo test -p rustsfm --lib extraction_backend_name_reports_wgpu --features gpu-wgpu -- --nocapture`

Run: `cargo test -p rustsfm --lib gpu_database_extraction_reuses_one_backend --features gpu-wgpu -- --nocapture`

Expected: FAIL because the loop has no injected persistent backend.

- [ ] **Step 3: Add a backend enum created once per command**

```rust
enum SiftExtractionBackend {
    Cpu,
    #[cfg(feature = "gpu-wgpu")]
    Wgpu(WgpuSiftExtractor),
}

impl SiftExtractionBackend {
    fn from_options(options: &SiftExtractionOptions) -> Result<Self> {
        if options.use_gpu {
            #[cfg(feature = "gpu-wgpu")]
            return Ok(Self::Wgpu(WgpuSiftExtractor::try_new()?));
            #[cfg(not(feature = "gpu-wgpu"))]
            bail!("RustSFM was built without gpu-wgpu support");
        }
        Ok(Self::Cpu)
    }

    fn extract(&self, gray: &[u8], width: u32, height: u32, options: &SiftExtractionOptions)
        -> Result<SiftFeatures>
    {
        match self {
            Self::Cpu => extract_sift_from_grayscale_u8_cpu(gray, width, height, options),
            #[cfg(feature = "gpu-wgpu")]
            Self::Wgpu(extractor) => extractor.extract_grayscale(gray, width, height, options),
        }
    }
}
```

Create this enum once before iterating database images. Keep database persistence serial and
failure-atomic. Update `ExtractFeaturesReport.backend` from the selected backend rather than
compile-time CPU features.

- [ ] **Step 4: Run feature extraction tests**

Run: `cargo test -p rustsfm --lib feature_extraction::tests --features gpu-wgpu -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add RustSFM/src/feature/feature_extraction.rs RustSFM/src/feature/sift.rs
git commit -m "feat(rustsfm): persist wgpu extractor across images"
```

### Task 10: Native And COLMAP-Compatible CLI Routing

**Files:**
- Modify: `RustSFM/src/cli/mod.rs`
- Modify: `RustSFM/src/cli/commands.rs`

- [ ] **Step 1: Write CLI parsing tests**

```rust
#[test]
fn colmap_feature_extractor_accepts_gpu_one() {
    let cli = Cli::try_parse_from([
        "rustsfm", "feature_extractor",
        "--database_path", "database.db",
        "--image_path", "images",
        "--SiftExtraction.use_gpu", "1",
    ]).unwrap();
    let Commands::FeatureExtractor(args) = cli.command else { panic!("wrong command") };
    assert_eq!(args.use_gpu, Some(1));
}

#[test]
fn native_extract_features_parses_use_gpu() {
    let cli = Cli::try_parse_from([
        "rustsfm", "extract-features",
        "--database", "database.db",
        "--images", "images",
        "--use-gpu",
    ]).unwrap();
    let Commands::ExtractFeatures(args) = cli.command else { panic!("wrong command") };
    assert!(args.use_gpu);
}
```

- [ ] **Step 2: Run CLI tests and verify native flag/routing is missing**

Run: `cargo test -p rustsfm --bin rustsfm colmap_feature_extractor_accepts_gpu_one --features gpu-wgpu -- --nocapture`

Run: `cargo test -p rustsfm --bin rustsfm native_extract_features_parses_use_gpu --features gpu-wgpu -- --nocapture`

Expected: FAIL because native `--use-gpu` is absent and COLMAP command rejects GPU execution.

- [ ] **Step 3: Route CLI flags into extraction options**

Add `#[arg(long, default_value_t = false)] use_gpu: bool` to native extraction and benchmark
arguments. Remove the command-level `bail!` for COLMAP `use_gpu=1`, then assign:

```rust
let mut options = sift_extraction_from_args(
    args.max_num_features,
    estimate_affine_shape,
    domain_size_pooling,
    estimate_affine_shape || domain_size_pooling,
);
options.use_gpu = use_gpu.unwrap_or(false);
```

For native commands, assign `options.use_gpu = args.use_gpu`. Preserve `use_gpu=0` behavior.
When compiled without `gpu-wgpu`, return `RustSFM was built without gpu-wgpu support` before
opening the database for mutation.

- [ ] **Step 4: Run CLI and library tests**

Run: `cargo test -p rustsfm --bin rustsfm --features gpu-wgpu -- --nocapture`

Run: `cargo test -p rustsfm --lib sift::tests --features gpu-wgpu -- --nocapture`

Run: `cargo test -p rustsfm --lib feature_extraction::tests --features gpu-wgpu -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add RustSFM/src/cli/mod.rs RustSFM/src/cli/commands.rs
git commit -m "feat(rustsfm): enable wgpu SIFT from CLI"
```

### Task 11: Quality Comparison And flowers2 Performance Gate

**Files:**
- Modify: `RustSFM/src/feature/sift.rs`
- Modify: `RustSFM/src/gpu/sift/mod.rs`
- Create: `RustSFM/tests/wgpu_sift_quality.rs`

- [ ] **Step 1: Add a deterministic CPU/GPU quality comparison**

```rust
#[test]
fn gpu_sift_preserves_checkerboard_geometric_content() -> Result<()> {
    let Some(context) = WgpuContext::try_new_optional()? else { return Ok(()) };
    let gray = textured_transform_fixture(512, 384);
    let cpu_options = SiftExtractionOptions { max_num_features: 1024, ..Default::default() };
    let gpu_options = SiftExtractionOptions { use_gpu: true, ..cpu_options.clone() };
    let cpu = extract_sift_from_grayscale_u8(&gray, 512, 384, &cpu_options)?;
    let gpu = WgpuSiftExtractor::from_context(context)?
        .extract_grayscale(&gray, 512, 384, &gpu_options)?;
    assert!(gpu.keypoints.len() >= cpu.keypoints.len() / 3);
    assert!(gpu.keypoints.len() <= cpu.keypoints.len() * 3 + 1);
    assert!(nearest_keypoint_repeatability(&cpu, &gpu, 2.0, 0.5) >= 0.55);
    Ok(())
}
```

- [ ] **Step 2: Run comparison and record initial variance**

Run: `cargo test -p rustsfm --test wgpu_sift_quality --features gpu-wgpu -- --nocapture`

Expected before final tuning: FAIL with measured count/repeatability diagnostics.

- [ ] **Step 3: Correct coordinate, sigma, threshold, or descriptor discrepancies**

Use diagnostics to adjust only SIFT-semantic differences: first-octave coordinate scale,
incremental sigma, contrast scaling by octave resolution, edge Hessian ratio, orientation
window, descriptor sample radius, and RootSIFT normalization. Do not weaken the final
repeatability threshold to hide a systematic error.

- [ ] **Step 4: Run the complete GPU and CPU-compatible suite**

Run: `cargo test -p rustsfm --lib gpu_ --features gpu-wgpu -- --nocapture`

Run: `cargo test -p rustsfm --lib wgpu_ --features gpu-wgpu -- --nocapture`

Run: `cargo test -p rustsfm --test wgpu_sift_quality --features gpu-wgpu -- --nocapture`

Run: `cargo test -p rustsfm --lib sift::tests --features gpu-wgpu -- --nocapture`

Run: `cargo test -p rustsfm --lib feature_extraction::tests --features gpu-wgpu -- --nocapture`

Expected: PASS. The known real-COLMAP fixture tests remain excluded from the baseline because
the local `test_data/flowers2_colmap` 24-image fixture is absent.

- [ ] **Step 5: Benchmark a fixed flowers2 subset and the full extraction set**

Run a warmed 20-image subset first, then all 960 images with the same SIFT options on CPU and
GPU. Record adapter, image count, feature count, stage timings, wall time, peak candidate
capacity, and any device validation message in a JSON report under the ignored
`test_data/flowers2/rustsfm_wgpu_benchmark` directory.

Expected: GPU output is valid, no buffer overflow or device loss occurs, and warmed GPU SIFT
is materially faster than the existing single-CPU extraction path. If it is not faster, keep
runtime GPU selection opt-in and report the measured bottleneck before matcher work begins.

- [ ] **Step 6: Run formatting and static checks**

Run: `cargo fmt -p rustsfm -- --check`

Run: `cargo clippy -p rustsfm --all-targets --features gpu-wgpu -- -D warnings`

Expected: formatting passes. Any pre-existing dependency warning is recorded separately;
new RustSFM warnings are fixed.

- [ ] **Step 7: Commit**

```bash
git add RustSFM/src/feature/sift.rs RustSFM/src/gpu RustSFM/tests/wgpu_sift_quality.rs Cargo.lock
git commit -m "test(rustsfm): validate wgpu SIFT quality and performance"
```

## Completion Criteria

- `SiftExtraction.use_gpu=1` and native `--use-gpu` execute a real wgpu/Metal SIFT path.
- Gaussian/DoG, detection, localization, orientation, and descriptor math run on GPU.
- The GPU path contains no Rayon extraction or descriptor work.
- One device and pipeline set are reused for all images in a database command.
- Keypoint and descriptor vectors are aligned and COLMAP SQLite-compatible.
- Explicit GPU errors never silently fall back to CPU.
- CPU extraction remains behaviorally unchanged when GPU use is disabled.
- Correctness, adapter smoke, and quality tests pass on the target Mac.
- The flowers2 benchmark records a real speed comparison before proceeding to GPU matching.
