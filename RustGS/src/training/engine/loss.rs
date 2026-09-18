use crate::training::config::DynamicMaskGradient;
use burn::prelude::*;
use burn::tensor::{
    module::conv2d,
    ops::{ConvOptions, PadMode},
    s,
};

#[derive(Debug, Clone)]
pub struct SsimConfig {
    pub window_size: usize,
    pub sigma: f64,
    pub k1: f64,
    pub k2: f64,
    pub data_range: f64,
}

impl Default for SsimConfig {
    fn default() -> Self {
        Self {
            window_size: 11,
            sigma: 1.5,
            k1: 0.01,
            k2: 0.03,
            data_range: 1.0,
        }
    }
}

pub fn ssim_loss_with_kernel<B: Backend>(
    pred: Tensor<B, 4>,
    target: Tensor<B, 4>,
    kernel: Tensor<B, 1>,
    config: &SsimConfig,
) -> Tensor<B, 1> {
    let pred = pred;
    let target = target;

    let mu_x = separable_blur(pred.clone(), kernel.clone());
    let mu_y = separable_blur(target.clone(), kernel.clone());

    let mu_x_sq = mu_x.clone().powi_scalar(2);
    let mu_y_sq = mu_y.clone().powi_scalar(2);
    let mu_xy = mu_x.clone() * mu_y.clone();

    let sigma_x_sq = separable_blur(pred.clone().powi_scalar(2), kernel.clone()) - mu_x_sq.clone();
    let sigma_y_sq =
        separable_blur(target.clone().powi_scalar(2), kernel.clone()) - mu_y_sq.clone();
    let sigma_xy = separable_blur(pred * target, kernel) - mu_xy.clone();

    let c1 = ((config.k1 * config.data_range).powi(2)) as f32;
    let c2 = ((config.k2 * config.data_range).powi(2)) as f32;

    let numerator = (mu_xy.mul_scalar(2.0) + c1) * (sigma_xy.mul_scalar(2.0) + c2);
    let denominator = (mu_x_sq + mu_y_sq + c1).clamp_min(1e-8_f32)
        * (sigma_x_sq + sigma_y_sq + c2).clamp_min(1e-8_f32);
    let ssim_map = numerator / denominator;

    let [_batch, _channels, height, width] = ssim_map.dims();
    let halo = config.window_size / 2;
    let ssim_mean = if height > halo * 2 && width > halo * 2 {
        ssim_map
            .slice(s![.., .., halo..height - halo, halo..width - halo])
            .mean()
    } else {
        ssim_map.mean()
    };

    ssim_mean.mul_scalar(-1.0).add_scalar(1.0).reshape([1])
}

#[allow(clippy::too_many_arguments)]
pub fn combined_loss_with_kernel<B: Backend>(
    pred: Tensor<B, 3>,
    target: Tensor<B, 3>,
    l1_weight: f64,
    ssim_weight: f64,
    gradient_weight: f64,
    robust_delta: f64,
    outlier_threshold: f64,
    outlier_weight: f64,
    dynamic_mask_threshold_low: f64,
    dynamic_mask_threshold_high: f64,
    dynamic_mask_min_weight: f64,
    dynamic_mask_gradient: DynamicMaskGradient,
    ssim_config: &SsimConfig,
    ssim_kernel: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let l1 = reconstruction_residual_loss(
        pred.clone(),
        target.clone(),
        robust_delta as f32,
        outlier_threshold as f32,
        outlier_weight as f32,
        dynamic_mask_threshold_low as f32,
        dynamic_mask_threshold_high as f32,
        dynamic_mask_min_weight as f32,
        dynamic_mask_gradient,
    );
    let gradient = if gradient_weight > 0.0 {
        gradient_difference_loss(pred.clone(), target.clone())
    } else {
        l1.clone().mul_scalar(0.0)
    };
    let mut total = l1.mul_scalar(l1_weight as f32) + gradient.mul_scalar(gradient_weight as f32);
    if ssim_weight > 0.0 {
        let ssim = ssim_loss_with_kernel(to_nchw(pred), to_nchw(target), ssim_kernel, ssim_config);
        total = total + ssim.mul_scalar(ssim_weight as f32);
    }
    total
}

#[allow(clippy::too_many_arguments)]
fn reconstruction_residual_loss<B: Backend>(
    pred: Tensor<B, 3>,
    target: Tensor<B, 3>,
    robust_delta: f32,
    outlier_threshold: f32,
    outlier_weight: f32,
    dynamic_mask_threshold_low: f32,
    dynamic_mask_threshold_high: f32,
    dynamic_mask_min_weight: f32,
    dynamic_mask_gradient: DynamicMaskGradient,
) -> Tensor<B, 1> {
    let abs_residual = (pred - target).abs();
    let loss = if robust_delta.is_finite() && robust_delta > 0.0 {
        // Saturating L1: behaves like L1 near zero but reduces the influence
        // of large residuals from dynamic objects, occlusion changes, or bad pixels.
        let delta = robust_delta.max(1e-6);
        abs_residual.clone().mul_scalar(delta) / (abs_residual + delta)
    } else if outlier_threshold.is_finite()
        && outlier_threshold > 0.0
        && outlier_weight.is_finite()
        && outlier_weight < 1.0
    {
        // Soft outlier weighting: preserves near-L1 behavior for small residuals
        // while retaining a configurable gradient floor for high-residual pixels.
        let threshold = outlier_threshold.max(1e-6);
        let floor = outlier_weight.clamp(0.0, 1.0);
        let adaptive_weight = abs_residual.clone().mul_scalar(0.0).add_scalar(floor)
            + (abs_residual
                .clone()
                .mul_scalar(0.0)
                .add_scalar(1.0 - floor)
                .mul_scalar(threshold)
                / (abs_residual.clone() + threshold));
        abs_residual * adaptive_weight
    } else {
        abs_residual
    };
    if dynamic_mask_threshold_high.is_finite()
        && dynamic_mask_threshold_low.is_finite()
        && dynamic_mask_threshold_high > dynamic_mask_threshold_low
        && dynamic_mask_min_weight.is_finite()
        && dynamic_mask_min_weight < 1.0
    {
        let weight = dynamic_residual_mask(
            loss.clone(),
            dynamic_mask_threshold_low,
            dynamic_mask_threshold_high,
            dynamic_mask_min_weight,
        );
        // Detach once at mask construction so both numerator and denominator see
        // the same stop-gradient weights (R08). Coupled mode keeps the old graph.
        let weight = match dynamic_mask_gradient {
            DynamicMaskGradient::StopGradient => weight.detach(),
            DynamicMaskGradient::Coupled => weight,
        };
        ((loss * weight.clone()).mean() / weight.mean().clamp_min(1e-6_f32)).reshape([1])
    } else {
        loss.mean().reshape([1])
    }
}

fn dynamic_residual_mask<B: Backend>(
    residual: Tensor<B, 3>,
    threshold_low: f32,
    threshold_high: f32,
    min_weight: f32,
) -> Tensor<B, 3> {
    let denom = (threshold_high - threshold_low).max(1e-6);
    let residual_mean = residual.mean_dim(2).repeat_dim(2, 3);
    residual_mean
        .mul_scalar(-1.0)
        .add_scalar(threshold_high)
        .div_scalar(denom)
        .clamp_min(min_weight.clamp(0.0, 1.0))
        .clamp_max(1.0)
}

fn gradient_difference_loss<B: Backend>(pred: Tensor<B, 3>, target: Tensor<B, 3>) -> Tensor<B, 1> {
    let [height, width, _channels] = pred.dims();
    debug_assert_eq!(pred.dims(), target.dims());
    let dx_pred =
        pred.clone().slice(s![.., 1..width, ..]) - pred.clone().slice(s![.., 0..width - 1, ..]);
    let dx_target =
        target.clone().slice(s![.., 1..width, ..]) - target.clone().slice(s![.., 0..width - 1, ..]);
    let dy_pred = pred.clone().slice(s![1..height, .., ..]) - pred.slice(s![0..height - 1, .., ..]);
    let dy_target =
        target.clone().slice(s![1..height, .., ..]) - target.slice(s![0..height - 1, .., ..]);
    let dx = (dx_pred - dx_target).abs().mean();
    let dy = (dy_pred - dy_target).abs().mean();
    (dx + dy).mul_scalar(0.5).reshape([1])
}

fn to_nchw<B: Backend>(tensor: Tensor<B, 3>) -> Tensor<B, 4> {
    tensor.unsqueeze_dim(0).swap_dims(1, 3).swap_dims(2, 3)
}

pub fn gaussian_kernel_1d<B: Backend>(config: &SsimConfig, device: &B::Device) -> Tensor<B, 1> {
    let radius = (config.window_size / 2) as isize;
    let mut values = Vec::with_capacity(config.window_size);
    let mut sum = 0.0f32;

    for offset in -radius..=radius {
        let value =
            (-((offset * offset) as f64) / (2.0 * config.sigma * config.sigma)).exp() as f32;
        values.push(value);
        sum += value;
    }

    for value in &mut values {
        *value /= sum.max(1e-8);
    }

    Tensor::<B, 1>::from_floats(values.as_slice(), device)
}

fn separable_blur<B: Backend>(tensor: Tensor<B, 4>, kernel: Tensor<B, 1>) -> Tensor<B, 4> {
    let kernel_size = kernel.dims()[0];
    let pad = kernel_size / 2;
    let [_n, channels, _height, _width] = tensor.dims();
    let horizontal_kernel = kernel
        .clone()
        .reshape([1, 1, 1, kernel_size])
        .repeat_dim(0, channels);
    let horizontal = conv2d(
        tensor.pad([(0, 0), (pad, pad)], PadMode::Constant(0.0)),
        horizontal_kernel,
        None,
        ConvOptions::new([1, 1], [0, 0], [1, 1], channels),
    );
    let vertical_kernel = kernel
        .reshape([1, 1, kernel_size, 1])
        .repeat_dim(0, channels);

    conv2d(
        horizontal.pad([(pad, pad), (0, 0)], PadMode::Constant(0.0)),
        vertical_kernel,
        None,
        ConvOptions::new([1, 1], [0, 0], [1, 1], channels),
    )
}

/// Host-only dynamic-mask math used by R08 gradient diagnostics.
pub mod dynamic_mask_host {
    pub fn mask_weight(residual: f32, low: f32, high: f32, min_weight: f32) -> f32 {
        let denom = (high - low).max(1e-6);
        ((high - residual) / denom).clamp(min_weight.clamp(0.0, 1.0), 1.0)
    }

    pub fn weighted_mean_loss(residuals: &[f32], weights: &[f32]) -> f32 {
        debug_assert_eq!(residuals.len(), weights.len());
        let num: f32 = residuals
            .iter()
            .zip(weights.iter())
            .map(|(residual, weight)| residual * weight)
            .sum();
        let den: f32 = weights.iter().sum::<f32>().max(1e-6);
        num / den
    }

    pub fn coupled_loss(residuals: &[f32], low: f32, high: f32, min_weight: f32) -> f32 {
        let weights: Vec<f32> = residuals
            .iter()
            .copied()
            .map(|residual| mask_weight(residual, low, high, min_weight))
            .collect();
        weighted_mean_loss(residuals, &weights)
    }

    pub fn detached_loss(residuals: &[f32], fixed_weights: &[f32]) -> f32 {
        weighted_mean_loss(residuals, fixed_weights)
    }
}

use burn::tensor::Int;
use burn_cubecl::cubecl::{prelude::KernelId, server::KernelArguments, CubeCount};
use burn_cubecl::{kernel::into_contiguous, BoolElement, CubeBackend, FloatElement, IntElement};
use burn_wgpu::{CubeDim, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};
use bytemuck::{Pod, Zeroable};

const MARK_NON_FINITE_SHADER: &str = include_str!("../shaders/mark_non_finite_loss.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MarkNonFiniteParams {
    iteration: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct MarkNonFiniteRaw;
impl MarkNonFiniteRaw {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(MARK_NON_FINITE_SHADER)
    }
}

#[derive(Debug)]
struct MarkNonFiniteKernel;
impl KernelSource for MarkNonFiniteKernel {
    fn source(&self) -> SourceTemplate {
        MarkNonFiniteRaw.source()
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

pub(crate) trait LossStatusBackend: Backend {
    fn mark_non_finite_loss(
        loss: Self::FloatTensorPrimitive,
        status: Self::IntTensorPrimitive,
        iteration: u32,
    );
}

impl<F, I, BT> LossStatusBackend for CubeBackend<WgpuRuntime, F, I, BT>
where
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn mark_non_finite_loss(
        loss: Self::FloatTensorPrimitive,
        status: Self::IntTensorPrimitive,
        iteration: u32,
    ) {
        let loss = into_contiguous(loss);
        let status = into_contiguous(status);
        let params = MarkNonFiniteParams {
            iteration,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_handle = loss.client.create_from_slice(bytemuck::bytes_of(&params));
        loss.client.launch(
            Box::new(SourceKernel::new(MarkNonFiniteKernel, CubeDim::new_1d(1))),
            CubeCount::Static(1, 1, 1),
            KernelArguments::new().with_buffers(vec![
                loss.handle.binding(),
                status.handle.binding(),
                params_handle.binding(),
            ]),
        );
        loss.client
            .flush()
            .expect("flush mark_non_finite_loss before status readback");
    }
}

#[cfg(test)]
mod tests {
    use super::dynamic_mask_host::{coupled_loss, detached_loss, mask_weight};
    use super::LossStatusBackend;
    use crate::training::engine::{
        DeviceTrainingStatus, GsBackendBase, GsDevice, STATUS_NON_FINITE_LOSS,
    };
    use burn::prelude::*;
    use burn::tensor::Int;

    #[test]
    fn coupled_dynamic_mask_can_reward_larger_residuals() {
        // Many static pixels plus one transitional pixel: raising the dynamic
        // residual lowers its weight enough that the normalized mean falls.
        let low = 0.05;
        let high = 0.20;
        let min_w = 0.1;
        let mut residuals = vec![0.01; 12];
        residuals.push(0.15);
        let base = coupled_loss(&residuals, low, high, min_w);
        *residuals.last_mut().unwrap() = 0.19;
        let increased = coupled_loss(&residuals, low, high, min_w);
        assert!(
            increased < base,
            "expected coupled mask to reward larger residual: base={base} increased={increased}"
        );
    }

    #[test]
    fn detached_dynamic_mask_keeps_non_negative_residual_derivative() {
        let low = 0.05;
        let high = 0.20;
        let min_w = 0.1;
        let residuals = vec![0.02, 0.08, 0.25];
        let fixed: Vec<f32> = residuals
            .iter()
            .copied()
            .map(|residual| mask_weight(residual, low, high, min_w))
            .collect();
        let eps = 1e-3;
        for idx in 0..residuals.len() {
            let mut plus = residuals.clone();
            let mut minus = residuals.clone();
            plus[idx] += eps;
            minus[idx] = (minus[idx] - eps).max(0.0);
            let d = (detached_loss(&plus, &fixed) - detached_loss(&minus, &fixed))
                / (plus[idx] - minus[idx]);
            assert!(
                d >= -1e-5,
                "detached mask derivative must not reward larger residual at {idx}: d={d}"
            );
        }
    }

    #[test]
    fn mask_weight_plateaus_outside_transition() {
        assert!((mask_weight(0.0, 0.05, 0.2, 0.1) - 1.0).abs() < 1e-6);
        assert!((mask_weight(0.3, 0.05, 0.2, 0.1) - 0.1).abs() < 1e-6);
    }

    async fn mark_and_read(
        value: f32,
        iteration: u32,
    ) -> crate::training::engine::TrainingStatusSnapshot {
        let device = GsDevice::default();
        let status = DeviceTrainingStatus::<GsBackendBase>::new(&device, 0);
        let loss = Tensor::<GsBackendBase, 1>::from_floats([value], &device);
        GsBackendBase::mark_non_finite_loss(
            loss.into_primitive().tensor(),
            status.buffer().clone().into_primitive(),
            iteration,
        );
        status.read().await.expect("status read")
    }

    #[tokio::test(flavor = "current_thread")]
    async fn finite_loss_does_not_set_non_finite_flag() {
        let snap = mark_and_read(0.125, 4).await;
        assert!(!snap.has_non_finite_loss(), "{snap:?}");
        assert_eq!(snap.flags & STATUS_NON_FINITE_LOSS, 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn nan_loss_sets_sticky_non_finite_flag() {
        let snap = mark_and_read(f32::NAN, 5).await;
        assert!(snap.has_non_finite_loss());
        assert_eq!(snap.first_invalid_iteration, 5);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn pos_inf_loss_sets_sticky_non_finite_flag() {
        let snap = mark_and_read(f32::INFINITY, 6).await;
        assert!(snap.has_non_finite_loss());
        assert_eq!(snap.first_invalid_iteration, 6);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn neg_inf_loss_sets_sticky_non_finite_flag() {
        let snap = mark_and_read(f32::NEG_INFINITY, 7).await;
        assert!(snap.has_non_finite_loss());
        assert_eq!(snap.first_invalid_iteration, 7);
    }
}
