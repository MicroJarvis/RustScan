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

fn reconstruction_residual_loss<B: Backend>(
    pred: Tensor<B, 3>,
    target: Tensor<B, 3>,
    robust_delta: f32,
    outlier_threshold: f32,
    outlier_weight: f32,
    dynamic_mask_threshold_low: f32,
    dynamic_mask_threshold_high: f32,
    dynamic_mask_min_weight: f32,
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
