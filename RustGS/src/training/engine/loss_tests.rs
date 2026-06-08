use super::{
    combined_loss, gradient_difference_loss, reconstruction_residual_loss, ssim_loss, SsimConfig,
};
use crate::training::engine::GsBackendBase;
use burn::prelude::Backend;
use burn::tensor::{ops::PadMode, s, Tensor, TensorData};

fn device() -> <GsBackendBase as burn::tensor::backend::Backend>::Device {
    Default::default()
}

fn separable_blur_reference<B: Backend>(
    tensor: Tensor<B, 4>,
    kernel: Tensor<B, 1>,
) -> Tensor<B, 4> {
    let horizontal = blur_width_reference(tensor, kernel.clone());
    blur_height_reference(horizontal, kernel)
}

fn blur_width_reference<B: Backend>(tensor: Tensor<B, 4>, kernel: Tensor<B, 1>) -> Tensor<B, 4> {
    let [_, _, _, width] = tensor.dims();
    let pad = kernel.dims()[0] / 2;
    let padded = tensor.pad([(0, 0), (pad, pad)], PadMode::Edge);
    let mut accum = padded
        .clone()
        .slice(s![.., .., .., 0..width])
        .mul(kernel.clone().slice(s![0]).unsqueeze());

    for index in 1..kernel.dims()[0] {
        let weight = kernel.clone().slice(s![index]).unsqueeze();
        let window = padded.clone().slice(s![.., .., .., index..index + width]);
        accum = accum + window.mul(weight);
    }

    accum
}

fn blur_height_reference<B: Backend>(tensor: Tensor<B, 4>, kernel: Tensor<B, 1>) -> Tensor<B, 4> {
    let [_, _, height, _] = tensor.dims();
    let pad = kernel.dims()[0] / 2;
    let padded = tensor.pad([(pad, pad), (0, 0)], PadMode::Edge);
    let mut accum = padded
        .clone()
        .slice(s![.., .., 0..height, ..])
        .mul(kernel.clone().slice(s![0]).unsqueeze());

    for index in 1..kernel.dims()[0] {
        let weight = kernel.clone().slice(s![index]).unsqueeze();
        let window = padded.clone().slice(s![.., .., index..index + height, ..]);
        accum = accum + window.mul(weight);
    }

    accum
}

#[tokio::test(flavor = "current_thread")]
async fn test_ssim_loss_identical_images() {
    let device = device();
    let image = Tensor::<GsBackendBase, 3>::ones([16, 16, 3], &device);

    let loss = ssim_loss(image.clone(), image, &SsimConfig::default(), &device)
        .into_scalar_async()
        .await
        .expect("ssim scalar");

    assert!(
        loss.abs() < 0.01,
        "expected near-zero SSIM loss, got {loss}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_ssim_loss_zero_images_is_finite() {
    let device = device();
    let image = Tensor::<GsBackendBase, 3>::zeros([16, 16, 3], &device);

    let loss = ssim_loss(image.clone(), image, &SsimConfig::default(), &device)
        .into_scalar_async()
        .await
        .expect("ssim scalar");

    assert!(loss.is_finite(), "expected finite SSIM loss, got {loss}");
}

#[tokio::test(flavor = "current_thread")]
async fn test_combined_loss() {
    let device = device();
    let pred = Tensor::<GsBackendBase, 3>::zeros([16, 16, 3], &device);
    let target = Tensor::<GsBackendBase, 3>::ones([16, 16, 3], &device);

    let loss = combined_loss(
        pred,
        target,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        &SsimConfig::default(),
        &device,
    )
    .into_scalar_async()
    .await
    .expect("combined loss scalar");

    assert!(
        loss > 0.5,
        "expected positive image mismatch loss, got {loss}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_robust_residual_loss_downweights_large_errors() {
    let device = device();
    let pred = Tensor::<GsBackendBase, 3>::zeros([1, 2, 3], &device);
    let target = Tensor::<GsBackendBase, 3>::from_data(
        TensorData::new(vec![0.1, 0.1, 0.1, 1.0, 1.0, 1.0], [1, 2, 3]),
        &device,
    );

    let exact =
        reconstruction_residual_loss(pred.clone(), target.clone(), 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
            .into_scalar_async()
            .await
            .expect("exact residual loss");
    let robust = reconstruction_residual_loss(pred, target, 0.1, 0.0, 1.0, 0.0, 0.0, 1.0)
        .into_scalar_async()
        .await
        .expect("robust residual loss");

    assert!(robust < exact, "robust loss should downweight outliers");
}

#[tokio::test(flavor = "current_thread")]
async fn test_soft_outlier_loss_preserves_gradient_floor() {
    let device = device();
    let pred = Tensor::<GsBackendBase, 3>::zeros([1, 2, 3], &device);
    let target = Tensor::<GsBackendBase, 3>::from_data(
        TensorData::new(vec![0.1, 0.1, 0.1, 1.0, 1.0, 1.0], [1, 2, 3]),
        &device,
    );

    let exact =
        reconstruction_residual_loss(pred.clone(), target.clone(), 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
            .into_scalar_async()
            .await
            .expect("exact residual loss");
    let weighted = reconstruction_residual_loss(pred, target, 0.0, 0.25, 0.25, 0.0, 0.0, 1.0)
        .into_scalar_async()
        .await
        .expect("soft outlier residual loss");

    assert!(
        weighted < exact,
        "soft outlier loss should downweight large residuals"
    );
    assert!(
        weighted > 0.0,
        "soft outlier loss should retain a gradient floor"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_dynamic_residual_mask_downweights_high_residual_pixels() {
    let device = device();
    let pred = Tensor::<GsBackendBase, 3>::zeros([1, 2, 3], &device);
    let target = Tensor::<GsBackendBase, 3>::from_data(
        TensorData::new(vec![0.1, 0.1, 0.1, 1.0, 1.0, 1.0], [1, 2, 3]),
        &device,
    );

    let exact =
        reconstruction_residual_loss(pred.clone(), target.clone(), 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
            .into_scalar_async()
            .await
            .expect("exact residual loss");
    let masked = reconstruction_residual_loss(pred, target, 0.0, 0.0, 1.0, 0.2, 0.8, 0.25)
        .into_scalar_async()
        .await
        .expect("masked residual loss");

    assert!(
        masked < exact,
        "dynamic residual mask should downweight high residual pixels"
    );
    assert!(
        masked > 0.0,
        "dynamic residual mask should retain a gradient floor"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_gradient_difference_loss_detects_edge_mismatch() {
    let device = device();
    let pred = Tensor::<GsBackendBase, 3>::zeros([4, 4, 3], &device);
    let mut target_values = vec![0.0f32; 4 * 4 * 3];
    for y in 0..4 {
        for x in 2..4 {
            let base = (y * 4 + x) * 3;
            target_values[base] = 1.0;
            target_values[base + 1] = 1.0;
            target_values[base + 2] = 1.0;
        }
    }
    let target =
        Tensor::<GsBackendBase, 3>::from_data(TensorData::new(target_values, [4, 4, 3]), &device);

    let loss = gradient_difference_loss(pred, target)
        .into_scalar_async()
        .await
        .expect("gradient loss scalar");

    assert!(loss > 0.0, "expected edge mismatch gradient loss");
}

#[tokio::test(flavor = "current_thread")]
async fn test_separable_blur_matches_reference() {
    let device = device();
    let tensor = Tensor::<GsBackendBase, 4>::from_data(
        TensorData::new((0..120).map(|v| v as f32 / 120.0).collect(), [1, 3, 5, 8]),
        &device,
    );
    let kernel = Tensor::<GsBackendBase, 1>::from_floats([0.25, 0.5, 0.25], &device);

    let reference = separable_blur_reference(tensor.clone(), kernel.clone())
        .into_data_async()
        .await
        .expect("reference blur");
    let optimized = super::separable_blur(tensor, kernel)
        .into_data_async()
        .await
        .expect("optimized blur");

    let reference = reference.as_slice::<f32>().expect("reference f32");
    let optimized = optimized.as_slice::<f32>().expect("optimized f32");

    for (idx, (lhs, rhs)) in reference.iter().zip(optimized.iter()).enumerate() {
        let delta = (lhs - rhs).abs();
        assert!(
            delta < 1e-4,
            "blur mismatch at index {idx}: {lhs} vs {rhs} (delta={delta})"
        );
    }
}
