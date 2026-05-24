# Phase 7: Verification and Testing

**Status**: Ready for Implementation  
**Dependencies**: Phase 6 (Wire + Delete Old Code) must be complete  
**Estimated Complexity**: Medium

## Overview

Verify the complete burn+wgpu migration by running compilation checks, unit tests, integration tests, and a small end-to-end training run. This phase ensures the new backend is functionally correct and ready for production use.

## Architecture

```
Verification Layers:
├─ Compilation (cargo check/build)
├─ Unit Tests (GPU primitives, topology, loss)
├─ Integration Tests (forward/backward pipeline)
├─ End-to-End Training (small dataset, 100 iterations)
└─ Parity Check (compare with reference results)
```

## Reference Implementation

- **Migration Plan**: `RustGS/docs/plans/2026-04-12-burn-wgpu-migration-plan.md` Section 10 (Phase 7)
- **Existing Tests**: `RustGS/src/training/metal/trainer/tests.rs` (reference patterns)
- **Parity Harness**: `RustGS/src/training/parity_harness.rs`

## Verification Tasks

### Part A: Compilation Verification

**Goal**: Ensure clean compilation with no warnings or errors.

**Commands**:
```bash
# 1. Check compilation
cargo check --features gpu

# 2. Build debug
cargo build --features gpu

# 3. Build release
cargo build --release --features gpu

# 4. Check all targets
cargo check --all-targets --features gpu
```

**Acceptance Criteria**:
- [ ] Zero compilation errors
- [ ] Zero clippy warnings (run `cargo clippy --features gpu`)
- [ ] No unused imports or dead code warnings
- [ ] All feature gates correct (`#[cfg(feature = "gpu")]`)

---

### Part B: Code Cleanup Verification

**Goal**: Confirm all Metal/candle code removed.

**Commands**:
```bash
# 1. No candle references
rg "candle_core|candle::" --type rust

# 2. No metal module references
rg "metal::" --type rust | grep -v "// " | grep -v "//"

# 3. No .metal shader files
find . -name "*.metal"

# 4. No diff module references
rg "crate::diff::" --type rust

# 5. Verify deleted directories
ls src/training/metal 2>/dev/null && echo "ERROR: metal/ still exists" || echo "OK: metal/ deleted"
ls src/diff 2>/dev/null && echo "ERROR: diff/ still exists" || echo "OK: diff/ deleted"
```

**Acceptance Criteria**:
- [ ] No candle_core or candle:: references
- [ ] No metal:: references (except in comments/docs)
- [ ] No .metal files
- [ ] No crate::diff:: references
- [ ] src/training/metal/ deleted
- [ ] src/diff/ deleted

---

### Part C: Unit Tests

**Goal**: Verify individual components work correctly.

#### C1: GPU Primitives Tests

**File**: `RustGS/src/training/wgpu/gpu_primitives/mod.rs` (add tests module)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::wgpu::GsDevice;
    use burn::prelude::*;
    use burn::tensor::Int;

    #[tokio::test]
    async fn test_radix_sort_u32() {
        let device = GsDevice::default();
        let input = Tensor::<GsBackend, 1, Int>::from_ints([5, 2, 8, 1, 9, 3], &device);
        let (sorted, _) = radix_sort_u32(input, 32, &device).await.unwrap();
        let result = sorted.into_data_async().await.to_vec::<i32>().unwrap();
        assert_eq!(result, vec![1, 2, 3, 5, 8, 9]);
    }

    #[tokio::test]
    async fn test_prefix_sum_u32() {
        let device = GsDevice::default();
        let input = Tensor::<GsBackend, 1, Int>::from_ints([1, 2, 3, 4], &device);
        let output = prefix_sum_u32(input, &device).await.unwrap();
        let result = output.into_data_async().await.to_vec::<i32>().unwrap();
        assert_eq!(result, vec![1, 3, 6, 10]);
    }
}
```

**Run**: `cargo test --features gpu gpu_primitives`

**Acceptance Criteria**:
- [ ] radix_sort_u32 test passes
- [ ] prefix_sum_u32 test passes
- [ ] No GPU errors or panics

#### C2: Loss Function Tests

**File**: `RustGS/src/training/wgpu/loss.rs` (add tests module)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::wgpu::GsDevice;
    use burn::prelude::*;

    #[test]
    fn test_ssim_loss_identical_images() {
        let device = GsDevice::default();
        let img = Tensor::<GsBackend, 3>::ones([64, 64, 3], &device);
        let loss = ssim_loss(img.clone(), img, &SsimConfig::default(), &device);
        let loss_val = loss.into_scalar();
        // SSIM of identical images should be 1.0, so loss (1 - SSIM) should be ~0
        assert!(loss_val < 0.01, "SSIM loss for identical images should be near 0, got {}", loss_val);
    }

    #[test]
    fn test_combined_loss() {
        let device = GsDevice::default();
        let pred = Tensor::<GsBackend, 3>::ones([32, 32, 3], &device);
        let target = Tensor::<GsBackend, 3>::zeros([32, 32, 3], &device);
        let loss = combined_loss(pred, target, 0.8, 0.2, &SsimConfig::default(), &device);
        let loss_val = loss.into_scalar();
        // Loss should be positive for different images
        assert!(loss_val > 0.5, "Combined loss should be significant for different images, got {}", loss_val);
    }
}
```

**Run**: `cargo test --features gpu loss`

**Acceptance Criteria**:
- [ ] SSIM loss test passes
- [ ] Combined loss test passes
- [ ] Loss values in expected range

#### C3: Optimizer Tests

**File**: `RustGS/src/training/wgpu/optimizer.rs` (add tests module)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::wgpu::GsDevice;
    use burn::prelude::*;

    #[test]
    fn test_adam_scaled_step() {
        let device = GsDevice::default();
        let config = AdamScaledConfig {
            lr: 0.01,
            ..Default::default()
        };
        let mut optimizer = AdamScaled::<GsBackend>::new(config);
        
        let param = Tensor::<GsBackend, 2>::ones([10, 10], &device);
        let grad = Tensor::<GsBackend, 2>::ones([10, 10], &device).mul_scalar(0.1);
        
        let mut state = AdamState::default();
        let updated = AdamScaled::step_tensor(&optimizer.config(), param.clone(), grad, &mut state);
        
        // Parameter should decrease (gradient descent)
        let param_mean = param.mean().into_scalar();
        let updated_mean = updated.mean().into_scalar();
        assert!(updated_mean < param_mean, "Adam step should decrease parameters");
    }
}
```

**Run**: `cargo test --features gpu optimizer`

**Acceptance Criteria**:
- [ ] Adam step test passes
- [ ] Parameters update correctly
- [ ] Bias correction applied

---

### Part D: Integration Tests

**Goal**: Verify end-to-end pipeline components.

#### D1: Forward Pipeline Test

**File**: `RustGS/src/training/wgpu/render/mod.rs` (add tests module)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::GaussianCamera;
    use crate::training::wgpu::{host_splats_to_device, GsDevice};
    use crate::training::HostSplats;
    use crate::{Intrinsics, SE3};

    #[tokio::test]
    async fn test_render_forward_single_gaussian() {
        let device = GsDevice::default();
        
        // Create minimal splat
        let positions = vec![0.0, 0.0, -5.0];
        let log_scales = vec![-2.0, -2.0, -2.0];
        let rotations = vec![1.0, 0.0, 0.0, 0.0]; // identity quat
        let opacity_logits = vec![2.0]; // high opacity
        let sh_coeffs = vec![0.5, 0.5, 0.5]; // gray
        
        let host_splats = HostSplats::from_components(
            positions, log_scales, rotations, opacity_logits, sh_coeffs, 0
        ).unwrap();
        
        let device_splats = host_splats_to_device(&host_splats, &device);
        
        // Create camera
        let intrinsics = Intrinsics {
            fx: 500.0, fy: 500.0, cx: 320.0, cy: 240.0,
            width: 640, height: 480,
        };
        let extrinsics = SE3::identity();
        let camera = GaussianCamera::new(intrinsics, extrinsics);
        
        // Render
        let output = render_forward(&device_splats, &camera, (640, 480), [0.0, 0.0, 0.0], &device).await;
        
        assert_eq!(output.num_visible, 1);
        assert!(output.num_intersections > 0);
        assert_eq!(output.out_img.dims(), [480, 640, 4]);
    }
}
```

**Run**: `cargo test --features gpu render_forward`

**Acceptance Criteria**:
- [ ] Forward render completes without errors
- [ ] Output dimensions correct
- [ ] Visible Gaussian detected

#### D2: Backward Pipeline Test

**File**: `RustGS/src/training/wgpu/render_bwd/mod.rs` (add tests module)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::GaussianCamera;
    use crate::training::wgpu::{host_splats_to_device, GsDevice, GsDiffBackend};
    use crate::training::HostSplats;
    use crate::{Intrinsics, SE3};
    use burn::prelude::*;

    #[tokio::test]
    async fn test_render_splats_autodiff() {
        let device = GsDevice::default();
        
        // Create minimal splat
        let positions = vec![0.0, 0.0, -5.0];
        let log_scales = vec![-2.0, -2.0, -2.0];
        let rotations = vec![1.0, 0.0, 0.0, 0.0];
        let opacity_logits = vec![2.0];
        let sh_coeffs = vec![0.5, 0.5, 0.5];
        
        let host_splats = HostSplats::from_components(
            positions, log_scales, rotations, opacity_logits, sh_coeffs, 0
        ).unwrap();
        
        let device_splats = host_splats_to_device::<GsDiffBackend>(&host_splats, &device);
        
        let intrinsics = Intrinsics {
            fx: 500.0, fy: 500.0, cx: 320.0, cy: 240.0,
            width: 640, height: 480,
        };
        let extrinsics = SE3::identity();
        let camera = GaussianCamera::new(intrinsics, extrinsics);
        
        // Render with autodiff
        let output = render_splats(&device_splats, &camera, (640, 480), [0.0, 0.0, 0.0]).await;
        
        // Compute loss and backward
        let loss = output.mean();
        let grads = loss.backward();
        
        // Check gradients exist
        let transforms_grad = device_splats.transforms.grad(&grads);
        assert!(transforms_grad.is_some(), "Transforms gradient should exist");
    }
}
```

**Run**: `cargo test --features gpu render_splats_autodiff`

**Acceptance Criteria**:
- [ ] Autodiff render completes
- [ ] Gradients computed successfully
- [ ] No NaN or Inf in gradients

---

### Part E: End-to-End Training Test

**Goal**: Run a complete training loop on a tiny dataset.

#### E1: Create Test Dataset

**File**: `RustGS/tests/integration_test.rs` (new file)

```rust
use rustgs::training::{train_splats, TrainingConfig, TrainingProfile};
use rustgs::{Intrinsics, SE3, TrainingDataset};
use glam::{Vec3, Quat};

#[test]
#[ignore] // Run with: cargo test --features gpu --ignored
fn test_wgpu_training_tiny_dataset() {
    // Create minimal dataset: 1 camera, 10 initial points
    let intrinsics = Intrinsics {
        fx: 500.0, fy: 500.0, cx: 320.0, cy: 240.0,
        width: 64, height: 64,
    };
    
    let pose = SE3::from_rotation_translation(
        &Quat::IDENTITY,
        &Vec3::new(0.0, 0.0, 0.0),
    );
    
    let initial_points = vec![
        Vec3::new(0.0, 0.0, -5.0),
        Vec3::new(1.0, 0.0, -5.0),
        Vec3::new(-1.0, 0.0, -5.0),
        Vec3::new(0.0, 1.0, -5.0),
        Vec3::new(0.0, -1.0, -5.0),
        Vec3::new(0.5, 0.5, -5.0),
        Vec3::new(-0.5, 0.5, -5.0),
        Vec3::new(0.5, -0.5, -5.0),
        Vec3::new(-0.5, -0.5, -5.0),
        Vec3::new(0.0, 0.0, -4.5),
    ];
    
    // Create dummy RGB image (64x64x3 = 12288 values)
    let image_data = vec![0.5f32; 64 * 64 * 3];
    
    let dataset = TrainingDataset {
        intrinsics: vec![intrinsics],
        poses: vec![pose],
        images: vec![image_data],
        initial_points,
    };
    
    let config = TrainingConfig {
        iterations: 100,
        training_profile: TrainingProfile::LegacyMetal, // Maps to wgpu
        ..Default::default()
    };
    
    // Run training
    let result = train_splats(&dataset, &config);
    
    assert!(result.is_ok(), "Training should complete without errors");
    let splats = result.unwrap();
    assert!(splats.num_splats() >= 10, "Should have at least initial splats");
    assert!(splats.num_splats() <= 1000, "Should not explode to too many splats");
}
```

**Run**: `cargo test --features gpu --ignored test_wgpu_training_tiny_dataset`

**Acceptance Criteria**:
- [ ] Training completes without panics
- [ ] Final splat count reasonable (10-1000)
- [ ] Loss decreases over iterations
- [ ] No GPU errors or timeouts

#### E2: Training Report Validation

Add to the test above:

```rust
use rustgs::training::train_splats_with_report;

#[test]
#[ignore]
fn test_wgpu_training_with_report() {
    // ... same dataset setup ...
    
    let result = train_splats_with_report(&dataset, &config);
    assert!(result.is_ok());
    
    let run = result.unwrap();
    assert!(run.report.final_loss.is_some(), "Should have final loss");
    assert!(run.report.gaussian_count > 0, "Should have Gaussians");
    assert!(run.report.elapsed.as_secs() < 300, "Should complete in reasonable time");
    
    println!("Training completed:");
    println!("  Final loss: {:?}", run.report.final_loss);
    println!("  Gaussian count: {}", run.report.gaussian_count);
    println!("  Elapsed: {:?}", run.report.elapsed);
}
```

**Acceptance Criteria**:
- [ ] TrainingRun structure populated correctly
- [ ] Report contains valid metrics
- [ ] Elapsed time reasonable

---

### Part F: Parity Check (Optional)

**Goal**: Compare wgpu results with reference Metal results (if available).

**Note**: This is optional since Metal backend is being removed. Only do this if you have saved reference results from before migration.

**File**: `RustGS/tests/parity_check.rs`

```rust
use rustgs::training::parity_harness::*;

#[test]
#[ignore]
fn test_wgpu_parity_with_reference() {
    // Load reference results (if saved before migration)
    let reference_path = "tests/fixtures/metal_reference_tiny.json";
    if !std::path::Path::new(reference_path).exists() {
        println!("Skipping parity check: no reference results");
        return;
    }
    
    // Run wgpu training
    // ... (same as E1) ...
    
    // Compare results
    // Check: final loss within 5% of reference
    // Check: splat count within 10% of reference
    // Check: PSNR within 1 dB of reference
}
```

**Acceptance Criteria** (if reference available):
- [ ] Final loss within 5% of Metal reference
- [ ] Splat count within 10% of Metal reference
- [ ] Visual quality comparable (PSNR within 1 dB)

---

## Summary Checklist

### Compilation
- [ ] `cargo check --features gpu` passes
- [ ] `cargo build --features gpu` passes
- [ ] `cargo build --release --features gpu` passes
- [ ] `cargo clippy --features gpu` no warnings

### Code Cleanup
- [ ] No candle references: `rg "candle_core|candle::"` empty
- [ ] No metal references: `rg "metal::"` empty (except comments)
- [ ] No .metal files: `find . -name "*.metal"` empty
- [ ] No diff module: `rg "crate::diff::"` empty
- [ ] Deleted: src/training/metal/
- [ ] Deleted: src/diff/

### Unit Tests
- [ ] GPU primitives tests pass
- [ ] Loss function tests pass
- [ ] Optimizer tests pass

### Integration Tests
- [ ] Forward pipeline test passes
- [ ] Backward pipeline test passes

### End-to-End
- [ ] Tiny dataset training completes
- [ ] Training report valid
- [ ] Loss decreases over iterations
- [ ] No GPU errors

### Documentation
- [ ] Update README.md (remove Metal references, add wgpu)
- [ ] Update CHANGELOG.md (document migration)
- [ ] Update any architecture docs

---

## Known Issues / Future Work

Document any known limitations or future improvements:

1. **Performance**: Initial wgpu performance may differ from Metal. Benchmark and optimize if needed.
2. **Telemetry**: TrainingRunReport.telemetry currently None. Add detailed telemetry in future.
3. **LR Scheduling**: Current implementation uses fixed LR. Add exponential decay scheduler.
4. **Topology Metrics**: Gradient accumulation logic may need tuning for optimal densification.
5. **WebGPU**: Test on WebGPU backend (browser) if needed.

---

## Codex Prompt

When ready to implement, use:

```
Implement Phase 7 (Verification and Testing) for RustGS burn+wgpu migration.

Reference:
- Task doc: RustGS/docs/tasks/phase7-verification.md

Tasks:
1. Run compilation verification (cargo check/build/clippy)
2. Run code cleanup verification (rg commands to check no candle/metal refs)
3. Add unit tests for GPU primitives, loss, optimizer
4. Add integration tests for forward/backward pipeline
5. Create end-to-end training test with tiny dataset (tests/integration_test.rs)
6. Run all tests and document results

Key requirements:
- All tests must pass
- No compilation warnings
- No candle/metal references remain
- End-to-end training completes successfully
- Document any issues or limitations

Verify with:
- cargo test --features gpu
- cargo test --features gpu --ignored (for integration tests)
- cargo clippy --features gpu
```
