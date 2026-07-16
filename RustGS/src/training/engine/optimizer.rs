use burn::module::Param;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::TensorPrimitive;
use burn_cubecl::cubecl::{prelude::KernelId, server::KernelArguments, CubeCount};
use burn_cubecl::{kernel::into_contiguous, BoolElement, CubeBackend, FloatElement, IntElement};
use burn_wgpu::{CubeDim, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};
use bytemuck::{Pod, Zeroable};

use super::splats::DeviceSplats;

const WORKGROUP_SIZE: u32 = 256;
const SHADER_SRC: &str = include_str!("../shaders/adam_update.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct AdamUpdateParams {
    len: u32,
    scale_len: u32,
    scale_inner_repeat: u32,
    step: u32,
    beta1: f32,
    beta2: f32,
    lr: f32,
    eps: f32,
    weight_decay: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct AdamUpdateRaw;

impl AdamUpdateRaw {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(SHADER_SRC)
    }
}

#[derive(Debug)]
struct AdamUpdateKernel;

impl KernelSource for AdamUpdateKernel {
    fn source(&self) -> SourceTemplate {
        AdamUpdateRaw.source()
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

pub(crate) struct AdamUpdatePrimitiveOutput<B: Backend> {
    param: B::FloatTensorPrimitive,
    moment1: B::FloatTensorPrimitive,
    moment2: B::FloatTensorPrimitive,
}

pub(crate) trait AdamUpdateBackend: Backend {
    fn adam_update_primitive(
        param: Self::FloatTensorPrimitive,
        grad: Self::FloatTensorPrimitive,
        moment1: Self::FloatTensorPrimitive,
        moment2: Self::FloatTensorPrimitive,
        scale: Self::FloatTensorPrimitive,
        params: AdamUpdateParams,
    ) -> AdamUpdatePrimitiveOutput<Self>;
}

impl<F, I, BT> AdamUpdateBackend for CubeBackend<WgpuRuntime, F, I, BT>
where
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn adam_update_primitive(
        param: Self::FloatTensorPrimitive,
        grad: Self::FloatTensorPrimitive,
        moment1: Self::FloatTensorPrimitive,
        moment2: Self::FloatTensorPrimitive,
        scale: Self::FloatTensorPrimitive,
        params: AdamUpdateParams,
    ) -> AdamUpdatePrimitiveOutput<Self> {
        let param = into_contiguous(param);
        let grad = into_contiguous(grad);
        let moment1 = into_contiguous(moment1);
        let moment2 = into_contiguous(moment2);
        let scale = into_contiguous(scale);

        if params.len > 0 {
            let params_handle = param.client.create_from_slice(bytemuck::bytes_of(&params));
            param.client.launch(
                Box::new(SourceKernel::new(
                    AdamUpdateKernel,
                    CubeDim::new_1d(WORKGROUP_SIZE),
                )),
                CubeCount::Static(params.len.div_ceil(WORKGROUP_SIZE), 1, 1),
                KernelArguments::new().with_buffers(vec![
                    param.handle.clone().binding(),
                    grad.handle.binding(),
                    moment1.handle.clone().binding(),
                    moment2.handle.clone().binding(),
                    scale.handle.binding(),
                    params_handle.binding(),
                ]),
            );
        }

        AdamUpdatePrimitiveOutput {
            param,
            moment1,
            moment2,
        }
    }
}

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
        Self::reset_state(&mut self.transforms);
        Self::reset_state(&mut self.sh_coeffs);
        Self::reset_state(&mut self.raw_opacities);
    }

    fn reset_state<const D: usize>(state: &mut AdamState<B, D>) {
        state.moment1 = None;
        state.moment2 = None;
        state.step = 0;
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

    fn step_tensor_burn<const D: usize>(
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

impl<B: AdamUpdateBackend> AdamScaled<B> {
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
            1,
        );
        let new_sh = Self::step_tensor(
            &self.config,
            splats.sh_coeffs.val().inner(),
            sh_grad,
            &mut self.sh_coeffs,
            3,
        );
        let new_opacity = Self::step_tensor(
            &self.config,
            splats.raw_opacities.val().inner(),
            opacity_grad,
            &mut self.raw_opacities,
            1,
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
        grad: Tensor<B, D>,
        state: &mut AdamState<B, D>,
        scale_inner_repeat: usize,
    ) -> Tensor<B, D> {
        let Some(scale) = state.scaling.clone() else {
            return Self::step_tensor_burn(config, param, grad, state);
        };

        let len = param.dims().iter().product::<usize>();
        let scale_len = scale.dims().iter().product::<usize>().max(1);
        let moment1 = state.moment1.take().unwrap_or_else(|| param.zeros_like());
        let moment2 = state.moment2.take().unwrap_or_else(|| param.zeros_like());

        state.step = state.step.saturating_add(1);
        let params = AdamUpdateParams {
            len: len as u32,
            scale_len: scale_len as u32,
            scale_inner_repeat: scale_inner_repeat.max(1) as u32,
            step: state.step as u32,
            beta1: config.betas.0 as f32,
            beta2: config.betas.1 as f32,
            lr: config.lr as f32,
            eps: config.eps as f32,
            weight_decay: config.weight_decay as f32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let output = B::adam_update_primitive(
            param.into_primitive().tensor(),
            grad.into_primitive().tensor(),
            moment1.into_primitive().tensor(),
            moment2.into_primitive().tensor(),
            scale.into_primitive().tensor(),
            params,
        );
        state.moment1 = Some(Tensor::from_primitive(TensorPrimitive::Float(
            output.moment1,
        )));
        state.moment2 = Some(Tensor::from_primitive(TensorPrimitive::Float(
            output.moment2,
        )));
        Tensor::from_primitive(TensorPrimitive::Float(output.param))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn optimizer_reset_preserves_per_parameter_scaling_contract() {
        let source = include_str!("optimizer.rs");
        let reset_state = source
            .split("fn reset_state")
            .nth(1)
            .and_then(|tail| tail.split("pub fn set_transform_scaling").next())
            .expect("reset_state body");
        assert!(!reset_state.contains("scaling = None"));
        assert!(!reset_state.contains("AdamState::default"));
        assert!(reset_state.contains("moment1 = None"));
        assert!(reset_state.contains("step = 0"));
    }
}
