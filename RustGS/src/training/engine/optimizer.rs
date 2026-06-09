use burn::module::Param;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use super::splats::DeviceSplats;

#[derive(Debug, Clone)]
pub struct AdamScaledConfig {
    pub lr: f64,
    pub betas: (f64, f64),
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for AdamScaledConfig {
    fn default() -> Self {
        Self {
            lr: 1.0,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdamState<B: Backend, const D: usize> {
    pub moment1: Option<Tensor<B, D>>,
    pub moment2: Option<Tensor<B, D>>,
    pub step: usize,
    pub scaling: Option<Tensor<B, D>>,
}

impl<B: Backend, const D: usize> Default for AdamState<B, D> {
    fn default() -> Self {
        Self {
            moment1: None,
            moment2: None,
            step: 0,
            scaling: None,
        }
    }
}

pub struct AdamScaled<B: Backend> {
    config: AdamScaledConfig,
    transforms: AdamState<B, 2>,
    sh_coeffs: AdamState<B, 3>,
    raw_opacities: AdamState<B, 1>,
}

impl<B: Backend> AdamScaled<B> {
    pub fn new(config: AdamScaledConfig) -> Self {
        Self {
            config,
            transforms: AdamState::default(),
            sh_coeffs: AdamState::default(),
            raw_opacities: AdamState::default(),
        }
    }

    pub fn reset(&mut self) {
        self.transforms = AdamState::default();
        self.sh_coeffs = AdamState::default();
        self.raw_opacities = AdamState::default();
    }

    pub fn set_transform_scaling(&mut self, scaling: Tensor<B, 2>) {
        self.transforms.scaling = Some(scaling);
    }

    pub fn set_sh_scaling(&mut self, scaling: Tensor<B, 3>) {
        self.sh_coeffs.scaling = Some(scaling);
    }

    pub fn set_opacity_scaling(&mut self, scaling: Tensor<B, 1>) {
        self.raw_opacities.scaling = Some(scaling);
    }

    pub fn step_device_splats<AD>(
        &mut self,
        splats: &mut DeviceSplats<AD>,
        transforms_grad: Tensor<B, 2>,
        sh_grad: Tensor<B, 3>,
        opacity_grad: Tensor<B, 1>,
    ) where
        AD: AutodiffBackend<InnerBackend = B>,
    {
        let new_transforms = Self::step_tensor(
            &self.config,
            splats.transforms.val().inner(),
            transforms_grad,
            &mut self.transforms,
        );
        let new_sh = Self::step_tensor(
            &self.config,
            splats.sh_coeffs.val().inner(),
            sh_grad,
            &mut self.sh_coeffs,
        );
        let new_opacity = Self::step_tensor(
            &self.config,
            splats.raw_opacities.val().inner(),
            opacity_grad,
            &mut self.raw_opacities,
        );

        splats.transforms = Param::initialized(
            splats.transforms.id,
            Tensor::<AD, 2>::from_inner(new_transforms).require_grad(),
        );
        splats.sh_coeffs = Param::initialized(
            splats.sh_coeffs.id,
            Tensor::<AD, 3>::from_inner(new_sh).require_grad(),
        );
        splats.raw_opacities = Param::initialized(
            splats.raw_opacities.id,
            Tensor::<AD, 1>::from_inner(new_opacity).require_grad(),
        );
    }

    fn step_tensor<const D: usize>(
        config: &AdamScaledConfig,
        param: Tensor<B, D>,
        mut grad: Tensor<B, D>,
        state: &mut AdamState<B, D>,
    ) -> Tensor<B, D> {
        if config.weight_decay != 0.0 {
            grad = grad + param.clone().mul_scalar(config.weight_decay as f32);
        }

        let beta1 = config.betas.0 as f32;
        let beta2 = config.betas.1 as f32;
        let one_minus_beta1 = 1.0 - beta1;
        let one_minus_beta2 = 1.0 - beta2;
        let grad_sq = grad.clone().powi_scalar(2);

        let moment1 = state.moment1.take().unwrap_or_else(|| param.zeros_like());
        let moment2 = state.moment2.take().unwrap_or_else(|| param.zeros_like());

        let moment1 = moment1.mul_scalar(beta1) + grad.clone().mul_scalar(one_minus_beta1);
        let moment2 = moment2.mul_scalar(beta2) + grad_sq.mul_scalar(one_minus_beta2);

        state.step = state.step.saturating_add(1);
        let step = state.step as i32;
        let bias_correction1 = 1.0 - beta1.powi(step);
        let bias_correction2 = 1.0 - beta2.powi(step);

        let moment1_hat = moment1.clone().div_scalar(bias_correction1);
        let moment2_hat = moment2.clone().div_scalar(bias_correction2);
        let update = moment1_hat / (moment2_hat.sqrt() + config.eps as f32);
        let scaled_lr = if let Some(scale) = &state.scaling {
            scale.clone() * config.lr as f32
        } else {
            update.ones_like().mul_scalar(config.lr as f32)
        };

        state.moment1 = Some(moment1);
        state.moment2 = Some(moment2);

        param - update * scaled_lr
    }
}
