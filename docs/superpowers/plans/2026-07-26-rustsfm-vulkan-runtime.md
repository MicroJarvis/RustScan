# RustSFM Vulkan Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make RustViewer's RustSFM reconstruction require and use Vulkan GPU compute rather than compiling or selecting a CPU fallback.

**Architecture:** Add a `gpu-vulkan` RustSFM feature that enables the existing WGPU compute code and forces its private WGPU instance to request `Backends::VULKAN`. RustViewer depends on that feature and maps persisted project GPU options into RustSFM's SIFT extraction, matching, and PnP configuration. Vulkan adapter discovery failure remains an explicit stage failure.

**Tech Stack:** Rust, Cargo features, wgpu 29, MoltenVK portability on macOS, RustViewer pipeline workers.

---

### Task 1: Lock project GPU configuration propagation

**Files:**
- Modify: `RustViewer/src/pipeline/rustsfm_worker.rs`

- [ ] **Step 1: Write failing tests**

Add a test that prepares the existing fixture request with all persisted GPU flags enabled and asserts that the mapper and sequence registration configurations have `sift_extraction.use_gpu`, `sift_matching.use_gpu`, and `use_gpu_pnp` enabled. Add a second test that disables those flags and asserts the configuration disables the corresponding paths.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p rust-viewer --lib pipeline::rustsfm_worker::tests::project_gpu_configuration -- --exact`

Expected: compilation failure because the configuration helper does not exist.

- [ ] **Step 3: Implement configuration helpers**

Construct `MapperConfig` from `request.manifest.sfm_config` and `request.manifest.pnp_config`, then use it in both worker entry points. Construct `SequenceRegistrationConfig` with the persisted PnP GPU flag.

- [ ] **Step 4: Run the focused tests**

Run: `cargo test -p rust-viewer --lib pipeline::rustsfm_worker::tests::project_gpu_configuration -- --exact`

Expected: PASS.

### Task 2: Add a Vulkan-specific RustSFM feature and instance policy

**Files:**
- Modify: `RustSFM/Cargo.toml`
- Modify: `RustSFM/src/gpu/context.rs`
- Modify: `RustSFM/src/gpu/mod.rs`

- [ ] **Step 1: Write a failing feature-gated test**

Under `gpu-vulkan`, add a test that requires the WGPU context to report a Vulkan adapter rather than accepting any WGPU backend.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p rustsfm --no-default-features --features gpu-vulkan --lib wgpu_context_requires_vulkan_adapter -- --exact --nocapture`

Expected: compilation failure because the `gpu-vulkan` feature and Vulkan backend report do not exist.

- [ ] **Step 3: Implement the feature and policy**

Define `gpu-vulkan = ["gpu-wgpu", "wgpu/vulkan", "wgpu/vulkan-portability"]`. In `WgpuContext`, use an explicit `InstanceDescriptor` with `Backends::VULKAN` when `gpu-vulkan` is active; retain the existing environment-driven descriptor for `gpu-wgpu`. Record the adapter's actual backend in `GpuSiftCapabilities` and surface Vulkan in `GpuBackendKind`.

- [ ] **Step 4: Run the feature-gated test**

Run: `cargo test -p rustsfm --no-default-features --features gpu-vulkan --lib wgpu_context_requires_vulkan_adapter -- --exact --nocapture`

Expected: PASS on a host with a Vulkan adapter; otherwise, a test failure that names the missing adapter.

### Task 3: Link RustViewer to Vulkan-only RustSFM compute

**Files:**
- Modify: `RustViewer/Cargo.toml`

- [ ] **Step 1: Enable the Vulkan feature**

Change the RustSFM dependency from `default-features = false` to `default-features = false, features = ["gpu-vulkan"]`.

- [ ] **Step 2: Build the release target**

Run: `cargo build -p rust-viewer --release`

Expected: successful release build with the Vulkan RustSFM code path linked.

### Task 4: Validate the complete integration

**Files:**
- Verify only

- [ ] **Step 1: Format and test**

Run: `cargo fmt --check`, `cargo test -p rust-viewer`, and `cargo test -p rustsfm --no-default-features --features gpu-vulkan --lib`.

- [ ] **Step 2: Validate Vulkan availability**

Run the focused RustSFM Vulkan-context test and `vulkaninfo --summary`; record the selected backend and adapter.

- [ ] **Step 3: Review the diff**

Run: `git diff --check` and inspect only the files above before reporting results.
