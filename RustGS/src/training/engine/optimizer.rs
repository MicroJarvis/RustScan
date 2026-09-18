use burn::module::Param;
use burn::prelude::*;
use burn::tensor::{backend::AutodiffBackend, Int, TensorPrimitive};
use burn_cubecl::cubecl::{prelude::KernelId, server::KernelArguments, CubeCount};
use burn_cubecl::{kernel::into_contiguous, BoolElement, CubeBackend, FloatElement, IntElement};
use burn_wgpu::{CubeDim, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};
use bytemuck::{Pod, Zeroable};

use crate::training::{
    AdamCheckpoint, AdamParameterCheckpoint, TensorCheckpoint, MAX_TRAINING_ITERATIONS,
};
use crate::TrainingError;

use super::splats::DeviceSplats;

const WORKGROUP_SIZE: u32 = 256;
const SHADER_SRC: &str = include_str!("../shaders/adam_update.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct AdamUpdateParams {
    len: u32,
    scale_len: u32,
    scale_inner_repeat: u32,
    beta1: f32,
    beta2: f32,
    lr: f32,
    eps: f32,
    weight_decay: f32,
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

const PREPARE_SHADER_SRC: &str = include_str!("../shaders/prepare_optimizer_step.wgsl");

struct PrepareOptimizerRaw;
impl PrepareOptimizerRaw {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(PREPARE_SHADER_SRC)
    }
}

#[derive(Debug)]
struct PrepareOptimizerKernel;
impl KernelSource for PrepareOptimizerKernel {
    fn source(&self) -> SourceTemplate {
        PrepareOptimizerRaw.source()
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PrepareOptimizerParams {
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

pub(crate) struct AdamUpdatePrimitiveOutput<B: Backend> {
    param: B::FloatTensorPrimitive,
    moment1: B::FloatTensorPrimitive,
    moment2: B::FloatTensorPrimitive,
}

pub(crate) trait AdamUpdateBackend: Backend {
    fn prepare_optimizer_step(status: Self::IntTensorPrimitive);

    fn adam_update_primitive(
        param: Self::FloatTensorPrimitive,
        grad: Self::FloatTensorPrimitive,
        moment1: Self::FloatTensorPrimitive,
        moment2: Self::FloatTensorPrimitive,
        scale: Self::FloatTensorPrimitive,
        status: Self::IntTensorPrimitive,
        params: AdamUpdateParams,
    ) -> AdamUpdatePrimitiveOutput<Self>;
}

impl<F, I, BT> AdamUpdateBackend for CubeBackend<WgpuRuntime, F, I, BT>
where
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn prepare_optimizer_step(status: Self::IntTensorPrimitive) {
        let status = into_contiguous(status);
        let params = PrepareOptimizerParams {
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        let params_handle = status.client.create_from_slice(bytemuck::bytes_of(&params));
        status.client.launch(
            Box::new(SourceKernel::new(
                PrepareOptimizerKernel,
                CubeDim::new_1d(1),
            )),
            CubeCount::Static(1, 1, 1),
            KernelArguments::new()
                .with_buffers(vec![status.handle.binding(), params_handle.binding()]),
        );
        status.client.flush().expect("flush prepare_optimizer_step");
    }

    fn adam_update_primitive(
        param: Self::FloatTensorPrimitive,
        grad: Self::FloatTensorPrimitive,
        moment1: Self::FloatTensorPrimitive,
        moment2: Self::FloatTensorPrimitive,
        scale: Self::FloatTensorPrimitive,
        status: Self::IntTensorPrimitive,
        params: AdamUpdateParams,
    ) -> AdamUpdatePrimitiveOutput<Self> {
        let param = into_contiguous(param);
        let grad = into_contiguous(grad);
        let moment1 = into_contiguous(moment1);
        let moment2 = into_contiguous(moment2);
        let scale = into_contiguous(scale);
        let status = into_contiguous(status);

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
                    status.handle.binding(),
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

#[cfg_attr(not(test), allow(dead_code))]
pub(super) async fn tensor_checkpoint<B: Backend, const D: usize>(
    tensor: &Tensor<B, D>,
) -> Result<TensorCheckpoint, TrainingError> {
    let shape = tensor.dims().to_vec();
    let values = tensor
        .to_data_async()
        .await
        .map_err(|error| {
            TrainingError::TrainingFailed(format!("optimizer tensor readback failed: {error}"))
        })?
        .into_vec::<f32>()
        .map_err(|error| {
            TrainingError::TrainingFailed(format!("optimizer tensor conversion failed: {error}"))
        })?;
    Ok(TensorCheckpoint { shape, values })
}

#[cfg_attr(not(test), allow(dead_code))]
async fn checkpoint_state<B: Backend, const D: usize>(
    state: &AdamState<B, D>,
) -> Result<AdamParameterCheckpoint, TrainingError> {
    let moment1 = match &state.moment1 {
        Some(tensor) => Some(tensor_checkpoint(tensor).await?),
        None => None,
    };
    let moment2 = match &state.moment2 {
        Some(tensor) => Some(tensor_checkpoint(tensor).await?),
        None => None,
    };
    let scaling = match &state.scaling {
        Some(tensor) => Some(tensor_checkpoint(tensor).await?),
        None => None,
    };

    Ok(AdamParameterCheckpoint {
        moment1,
        moment2,
        scaling,
        step: state.step,
    })
}

#[cfg_attr(not(test), allow(dead_code))]
fn invalid_optimizer_checkpoint(message: impl Into<String>) -> TrainingError {
    TrainingError::InvalidInput(format!("invalid optimizer checkpoint: {}", message.into()))
}

#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn validate_tensor_checkpoint<const D: usize>(
    name: &str,
    checkpoint: &TensorCheckpoint,
    expected_shape: [usize; D],
) -> Result<(), TrainingError> {
    if checkpoint.shape.len() != D {
        return Err(invalid_optimizer_checkpoint(format!(
            "{name} expected rank {D}, got {}",
            checkpoint.shape.len()
        )));
    }
    if checkpoint.shape.as_slice() != expected_shape {
        return Err(invalid_optimizer_checkpoint(format!(
            "{name} expected shape {expected_shape:?}, got {:?}",
            checkpoint.shape
        )));
    }
    let expected_values = expected_shape
        .iter()
        .try_fold(1usize, |product, dimension| product.checked_mul(*dimension))
        .ok_or_else(|| invalid_optimizer_checkpoint(format!("{name} shape overflows usize")))?;
    if checkpoint.values.len() != expected_values {
        return Err(invalid_optimizer_checkpoint(format!(
            "{name} shape expects {expected_values} values, got {}",
            checkpoint.values.len()
        )));
    }
    if checkpoint.values.iter().any(|value| !value.is_finite()) {
        return Err(invalid_optimizer_checkpoint(format!(
            "{name} values must be finite"
        )));
    }

    Ok(())
}

#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn tensor_from_validated_checkpoint<B: Backend, const D: usize>(
    checkpoint: &TensorCheckpoint,
    device: &B::Device,
) -> Tensor<B, D> {
    Tensor::from_data(
        TensorData::new(
            checkpoint.values.clone(),
            Shape::from(checkpoint.shape.clone()),
        ),
        device,
    )
}

#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn restore_tensor<B: Backend, const D: usize>(
    name: &str,
    checkpoint: &TensorCheckpoint,
    expected_shape: [usize; D],
    device: &B::Device,
) -> Result<Tensor<B, D>, TrainingError> {
    validate_tensor_checkpoint(name, checkpoint, expected_shape)?;

    Ok(tensor_from_validated_checkpoint(checkpoint, device))
}

#[cfg_attr(not(test), allow(dead_code))]
fn restore_state<B: Backend, const D: usize>(
    name: &str,
    checkpoint: &AdamParameterCheckpoint,
    parameter_shape: [usize; D],
    scaling_shape: [usize; D],
    device: &B::Device,
) -> Result<AdamState<B, D>, TrainingError> {
    if checkpoint.moment1.is_some() != checkpoint.moment2.is_some() {
        return Err(invalid_optimizer_checkpoint(format!(
            "{name}.moment1 and moment2 must both be present or both be absent"
        )));
    }
    if checkpoint.step == 0 && checkpoint.moment1.is_some() {
        return Err(invalid_optimizer_checkpoint(format!(
            "{name} moments must be absent when step is zero"
        )));
    }
    if checkpoint.step > 0 && checkpoint.moment1.is_none() {
        return Err(invalid_optimizer_checkpoint(format!(
            "{name} moments must be present when step is non-zero"
        )));
    }

    let moment1 = checkpoint
        .moment1
        .as_ref()
        .map(|tensor| restore_tensor(&format!("{name}.moment1"), tensor, parameter_shape, device))
        .transpose()?;
    let moment2 = checkpoint
        .moment2
        .as_ref()
        .map(|tensor| restore_tensor(&format!("{name}.moment2"), tensor, parameter_shape, device))
        .transpose()?;
    let scaling = checkpoint
        .scaling
        .as_ref()
        .map(|tensor| restore_tensor(&format!("{name}.scaling"), tensor, scaling_shape, device))
        .transpose()?;

    Ok(AdamState {
        moment1,
        moment2,
        step: checkpoint.step,
        scaling,
    })
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

    #[cfg_attr(not(test), allow(dead_code))]
    pub async fn checkpoint(&self) -> Result<AdamCheckpoint, TrainingError> {
        Ok(AdamCheckpoint {
            transforms: checkpoint_state(&self.transforms).await?,
            sh_coeffs: checkpoint_state(&self.sh_coeffs).await?,
            raw_opacities: checkpoint_state(&self.raw_opacities).await?,
        })
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn restore<AD>(
        &mut self,
        checkpoint: &AdamCheckpoint,
        splats: &DeviceSplats<AD>,
        device: &B::Device,
    ) -> Result<(), TrainingError>
    where
        AD: AutodiffBackend<InnerBackend = B>,
    {
        if checkpoint.transforms.step != checkpoint.sh_coeffs.step
            || checkpoint.transforms.step != checkpoint.raw_opacities.step
        {
            return Err(invalid_optimizer_checkpoint(format!(
                "optimizer parameter steps must be equal, got transforms={}, sh_coeffs={}, raw_opacities={}",
                checkpoint.transforms.step,
                checkpoint.sh_coeffs.step,
                checkpoint.raw_opacities.step
            )));
        }
        if checkpoint.transforms.step > MAX_TRAINING_ITERATIONS {
            return Err(invalid_optimizer_checkpoint(format!(
                "optimizer step {} exceeds maximum safe step {MAX_TRAINING_ITERATIONS}",
                checkpoint.transforms.step
            )));
        }

        let transforms_shape = splats.transforms.val().dims();
        let sh_shape = splats.sh_coeffs.val().dims();
        let opacity_shape = splats.raw_opacities.val().dims();
        let num_splats = transforms_shape[0];

        if transforms_shape[1] != 10 {
            return Err(invalid_optimizer_checkpoint(format!(
                "splat transforms expected shape [N, 10], got {transforms_shape:?}"
            )));
        }
        if sh_shape[0] != num_splats || sh_shape[2] != 3 {
            return Err(invalid_optimizer_checkpoint(format!(
                "splat SH coefficients expected shape [N, K, 3], got {sh_shape:?}"
            )));
        }
        if opacity_shape[0] != num_splats {
            return Err(invalid_optimizer_checkpoint(format!(
                "splat raw opacities expected shape [N], got {opacity_shape:?}"
            )));
        }

        let transforms = restore_state(
            "transforms",
            &checkpoint.transforms,
            transforms_shape,
            [1, 10],
            device,
        )?;
        let sh_coeffs = restore_state(
            "sh_coeffs",
            &checkpoint.sh_coeffs,
            sh_shape,
            [1, sh_shape[1], 1],
            device,
        )?;
        let raw_opacities = restore_state(
            "raw_opacities",
            &checkpoint.raw_opacities,
            opacity_shape,
            [1],
            device,
        )?;

        self.transforms = transforms;
        self.sh_coeffs = sh_coeffs;
        self.raw_opacities = raw_opacities;
        Ok(())
    }

    fn reset_state<const D: usize>(state: &mut AdamState<B, D>) {
        state.moment1 = None;
        state.moment2 = None;
        state.step = 0;
    }

    /// Keep surviving rows' moments and Adam step. New rows start at zero.
    /// A full reset happens only when a moment's inner layout no longer matches.
    pub fn remap_origins(
        &mut self,
        origins: &[Option<usize>],
        sh_coeffs: usize,
        sh_channels: usize,
        device: &B::Device,
    ) {
        Self::remap_state(&mut self.transforms, origins, &[10], device);
        Self::remap_state(
            &mut self.sh_coeffs,
            origins,
            &[sh_coeffs, sh_channels],
            device,
        );
        Self::remap_state(&mut self.raw_opacities, origins, &[], device);
    }

    /// Opacity reset changes logits but must not rewind Adam step or wipe
    /// transform/SH moments. Only the opacity moment buffers are zeroed.
    pub fn clear_opacity_moments(&mut self) {
        if let Some(moment1) = self.raw_opacities.moment1.as_ref() {
            self.raw_opacities.moment1 = Some(moment1.zeros_like());
        }
        if let Some(moment2) = self.raw_opacities.moment2.as_ref() {
            self.raw_opacities.moment2 = Some(moment2.zeros_like());
        }
    }

    fn remap_state<const D: usize>(
        state: &mut AdamState<B, D>,
        origins: &[Option<usize>],
        inner_dims: &[usize],
        device: &B::Device,
    ) {
        let Some(moment1) = state.moment1.clone() else {
            return;
        };
        let dims = moment1.dims();
        let layout_matches = dims.len() == inner_dims.len() + 1
            && dims[1..].iter().copied().eq(inner_dims.iter().copied());
        if !layout_matches {
            Self::reset_state(state);
            return;
        }
        state.moment1 = Some(Self::remap_moment_rows(moment1, origins, device));
        if let Some(moment2) = state.moment2.clone() {
            state.moment2 = Some(Self::remap_moment_rows(moment2, origins, device));
        }
    }

    fn remap_moment_rows<const D: usize>(
        tensor: Tensor<B, D>,
        origins: &[Option<usize>],
        device: &B::Device,
    ) -> Tensor<B, D> {
        let dims = tensor.dims();
        let mut out_dims = dims;
        out_dims[0] = origins.len();
        if origins.is_empty() {
            return Tensor::zeros(out_dims, device);
        }
        let row_width = dims.iter().skip(1).copied().product::<usize>().max(1);
        let old_len = dims[0];
        let flat = tensor.reshape([old_len, row_width]);
        let mut index_values = Vec::with_capacity(origins.len() * row_width);
        let mut mask_values = Vec::with_capacity(origins.len() * row_width);
        for origin in origins {
            let (index, keep) = match origin {
                Some(source) if *source < old_len => (*source as i32, 1.0),
                _ => (0, 0.0),
            };
            index_values.extend(std::iter::repeat_n(index, row_width));
            mask_values.extend(std::iter::repeat_n(keep, row_width));
        }
        let indices = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(index_values, [origins.len(), row_width]),
            device,
        );
        let mask = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(mask_values, [origins.len(), row_width]),
            device,
        );
        flat.gather(0, indices).mul(mask).reshape(out_dims)
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
    /// Align host Adam step counters with device-committed steps (status word 4).
    /// Called from training safety-point status reads; normal steps stay readback-free.
    pub fn sync_committed_steps(&mut self, committed: usize) {
        self.transforms.step = committed;
        self.sh_coeffs.step = committed;
        self.raw_opacities.step = committed;
    }

    pub fn step_device_splats<AD>(
        &mut self,
        splats: &mut DeviceSplats<AD>,
        transforms_grad: Tensor<B, 2>,
        sh_grad: Tensor<B, 3>,
        opacity_grad: Tensor<B, 1>,
        status: Tensor<B, 1, Int>,
    ) where
        AD: AutodiffBackend<InnerBackend = B>,
    {
        B::prepare_optimizer_step(status.clone().into_primitive());
        // Optimistic host advance. Device word 4 is authoritative for bias
        // correction and for blocked gates; safety-point reads resync host.
        let next_step = self.transforms.step.saturating_add(1);
        self.transforms.step = next_step;
        self.sh_coeffs.step = next_step;
        self.raw_opacities.step = next_step;

        let new_transforms = Self::step_tensor(
            &self.config,
            splats.transforms.val().inner(),
            transforms_grad,
            &mut self.transforms,
            1,
            status.clone(),
        );
        let new_sh = Self::step_tensor(
            &self.config,
            splats.sh_coeffs.val().inner(),
            sh_grad,
            &mut self.sh_coeffs,
            3,
            status.clone(),
        );
        let new_opacity = Self::step_tensor(
            &self.config,
            splats.raw_opacities.val().inner(),
            opacity_grad,
            &mut self.raw_opacities,
            1,
            status,
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
        status: Tensor<B, 1, Int>,
    ) -> Tensor<B, D> {
        let device = param.device();
        let (scale, scale_inner_repeat) = match state.scaling.clone() {
            Some(scale) => {
                let flat_len = scale.dims().iter().product::<usize>().max(1);
                (scale.reshape([flat_len]), scale_inner_repeat.max(1))
            }
            None => (Tensor::<B, 1>::ones([1], &device), 1usize),
        };

        let len = param.dims().iter().product::<usize>();
        let scale_len = scale.dims()[0].max(1);
        let moment1 = state.moment1.take().unwrap_or_else(|| param.zeros_like());
        let moment2 = state.moment2.take().unwrap_or_else(|| param.zeros_like());

        let params = AdamUpdateParams {
            len: len as u32,
            scale_len: scale_len as u32,
            scale_inner_repeat: scale_inner_repeat as u32,
            beta1: config.betas.0 as f32,
            beta2: config.betas.1 as f32,
            lr: config.lr as f32,
            eps: config.eps as f32,
            weight_decay: config.weight_decay as f32,
        };
        let output = B::adam_update_primitive(
            param.into_primitive().tensor(),
            grad.into_primitive().tensor(),
            moment1.into_primitive().tensor(),
            moment2.into_primitive().tensor(),
            scale.into_primitive().tensor(),
            status.into_primitive(),
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
    use super::*;
    use crate::training::engine::{
        DeviceTrainingStatus, GsBackendBase, GsDiffBackend, TrainingStatusSnapshot,
        STATUS_NON_FINITE_LOSS,
    };

    fn test_splats(device: &<GsBackendBase as Backend>::Device) -> DeviceSplats<GsDiffBackend> {
        let transforms = Tensor::<GsDiffBackend, 2>::from_data(
            TensorData::new(
                (0..20).map(|value| value as f32 * 0.01).collect(),
                Shape::new([2, 10]),
            ),
            device,
        );
        let sh_coeffs = Tensor::<GsDiffBackend, 3>::from_data(
            TensorData::new(
                (0..24).map(|value| value as f32 * 0.005).collect(),
                Shape::new([2, 4, 3]),
            ),
            device,
        );
        let raw_opacities = Tensor::<GsDiffBackend, 1>::from_floats([0.1, -0.2], device);

        DeviceSplats {
            transforms: Param::from_tensor(transforms),
            sh_coeffs: Param::from_tensor(sh_coeffs),
            raw_opacities: Param::from_tensor(raw_opacities),
            sh_degree: 1,
        }
    }

    async fn copy_splats(
        splats: &DeviceSplats<GsDiffBackend>,
        device: &<GsBackendBase as Backend>::Device,
    ) -> DeviceSplats<GsDiffBackend> {
        DeviceSplats {
            transforms: Param::from_tensor(Tensor::from_data(
                splats
                    .transforms
                    .val()
                    .to_data_async()
                    .await
                    .expect("transforms readback"),
                device,
            )),
            sh_coeffs: Param::from_tensor(Tensor::from_data(
                splats
                    .sh_coeffs
                    .val()
                    .to_data_async()
                    .await
                    .expect("SH readback"),
                device,
            )),
            raw_opacities: Param::from_tensor(Tensor::from_data(
                splats
                    .raw_opacities
                    .val()
                    .to_data_async()
                    .await
                    .expect("opacity readback"),
                device,
            )),
            sh_degree: splats.sh_degree,
        }
    }

    async fn assert_tensor_close<B: Backend, const D: usize>(
        actual: Tensor<B, D>,
        expected: Tensor<B, D>,
    ) {
        let actual = actual
            .into_data_async()
            .await
            .expect("actual readback")
            .into_vec::<f32>()
            .expect("actual f32 data");
        let expected = expected
            .into_data_async()
            .await
            .expect("expected readback")
            .into_vec::<f32>()
            .expect("expected f32 data");
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "element {index}: expected {expected}, got {actual}"
            );
        }
    }

    fn optimizer_with_scaling(
        device: &<GsBackendBase as Backend>::Device,
    ) -> AdamScaled<GsBackendBase> {
        let mut optimizer = AdamScaled::new(AdamScaledConfig {
            lr: 0.01,
            ..AdamScaledConfig::default()
        });
        optimizer.set_transform_scaling(Tensor::ones([1, 10], device));
        optimizer.set_sh_scaling(Tensor::ones([1, 4, 1], device).mul_scalar(0.5));
        optimizer.set_opacity_scaling(Tensor::from_floats([0.25], device));
        optimizer
    }

    fn optimizer_step(
        optimizer: &mut AdamScaled<GsBackendBase>,
        splats: &mut DeviceSplats<GsDiffBackend>,
        status: &DeviceTrainingStatus<GsBackendBase>,
        device: &<GsBackendBase as Backend>::Device,
    ) {
        optimizer.step_device_splats(
            splats,
            Tensor::ones([2, 10], device).mul_scalar(0.1),
            Tensor::ones([2, 4, 3], device).mul_scalar(-0.2),
            Tensor::ones([2], device).mul_scalar(0.3),
            status.buffer().clone(),
        );
    }

    fn restore_error(
        checkpoint: &AdamCheckpoint,
        splats: &DeviceSplats<GsDiffBackend>,
        device: &<GsBackendBase as Backend>::Device,
    ) -> String {
        let mut optimizer = AdamScaled::<GsBackendBase>::new(AdamScaledConfig::default());
        optimizer
            .restore(checkpoint, splats, device)
            .expect_err("malformed checkpoint must be rejected")
            .to_string()
    }

    #[tokio::test(flavor = "current_thread")]
    async fn adam_mutation_gate_blocks_param_moment_and_step() {
        let device = <GsBackendBase as Backend>::Device::default();
        let mut splats = test_splats(&device);
        let mut optimizer = optimizer_with_scaling(&device);
        let mut status = DeviceTrainingStatus::<GsBackendBase>::new(&device, 0);

        // Build non-zero moments and step=7 on a healthy status buffer.
        for _ in 0..7 {
            optimizer_step(&mut optimizer, &mut splats, &status, &device);
        }
        assert_eq!(optimizer.transforms.step, 7);
        assert_eq!(
            status
                .read()
                .await
                .expect("status read")
                .committed_optimizer_steps,
            7
        );

        let before_ckpt = optimizer.checkpoint().await.expect("before checkpoint");
        let before_splats = copy_splats(&splats, &device).await;
        assert!(before_ckpt.transforms.moment1.is_some());
        assert!(before_ckpt.transforms.moment2.is_some());

        status.set_host_snapshot(TrainingStatusSnapshot {
            flags: STATUS_NON_FINITE_LOSS,
            first_invalid_iteration: 2,
            requested_intersections: 0,
            intersection_capacity: 0,
            committed_optimizer_steps: 7,
            mutation_gate: 1,
        });

        optimizer_step(&mut optimizer, &mut splats, &status, &device);

        let after_status = status.read().await.expect("status after blocked step");
        assert_eq!(after_status.committed_optimizer_steps, 7);
        assert_eq!(after_status.mutation_gate, 0);
        // Host may optimistically advance; device word 4 is authoritative.
        optimizer.sync_committed_steps(after_status.committed_optimizer_steps as usize);
        assert_eq!(optimizer.transforms.step, 7);
        assert_eq!(optimizer.sh_coeffs.step, 7);
        assert_eq!(optimizer.raw_opacities.step, 7);

        let after_ckpt = optimizer.checkpoint().await.expect("after checkpoint");
        assert_eq!(after_ckpt, before_ckpt);
        assert_tensor_close(splats.transforms.val(), before_splats.transforms.val()).await;
        assert_tensor_close(splats.sh_coeffs.val(), before_splats.sh_coeffs.val()).await;
        assert_tensor_close(
            splats.raw_opacities.val(),
            before_splats.raw_opacities.val(),
        )
        .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn optimizer_checkpoint_roundtrips_state_and_preserves_next_step() {
        let device = <GsBackendBase as Backend>::Device::default();
        let mut splats = test_splats(&device);
        let mut optimizer = optimizer_with_scaling(&device);
        let status = DeviceTrainingStatus::<GsBackendBase>::new(&device, 0);

        optimizer_step(&mut optimizer, &mut splats, &status, &device);
        optimizer_step(&mut optimizer, &mut splats, &status, &device);
        let checkpoint = optimizer.checkpoint().await.expect("export checkpoint");
        let mut resumed_splats = copy_splats(&splats, &device).await;

        assert_eq!(checkpoint.transforms.step, 2);
        assert_eq!(checkpoint.sh_coeffs.step, 2);
        assert_eq!(checkpoint.raw_opacities.step, 2);
        assert_eq!(
            checkpoint.transforms.moment1.as_ref().unwrap().shape,
            [2, 10]
        );
        assert_eq!(
            checkpoint.transforms.scaling.as_ref().unwrap().shape,
            [1, 10]
        );
        assert_eq!(
            checkpoint.sh_coeffs.moment1.as_ref().unwrap().shape,
            [2, 4, 3]
        );
        assert_eq!(
            checkpoint.sh_coeffs.scaling.as_ref().unwrap().shape,
            [1, 4, 1]
        );
        assert_eq!(
            checkpoint.raw_opacities.moment1.as_ref().unwrap().shape,
            [2]
        );
        assert_eq!(
            checkpoint.raw_opacities.scaling.as_ref().unwrap().shape,
            [1]
        );

        let mut restored = AdamScaled::new(AdamScaledConfig {
            lr: 0.01,
            ..AdamScaledConfig::default()
        });
        restored
            .restore(&checkpoint, &resumed_splats, &device)
            .expect("restore checkpoint");
        assert_eq!(
            restored.checkpoint().await.expect("re-export checkpoint"),
            checkpoint
        );

        optimizer_step(&mut optimizer, &mut splats, &status, &device);
        let restored_status = DeviceTrainingStatus::<GsBackendBase>::new(&device, 2);
        optimizer_step(
            &mut restored,
            &mut resumed_splats,
            &restored_status,
            &device,
        );
        assert_tensor_close(splats.transforms.val(), resumed_splats.transforms.val()).await;
        assert_tensor_close(splats.sh_coeffs.val(), resumed_splats.sh_coeffs.val()).await;
        assert_tensor_close(
            splats.raw_opacities.val(),
            resumed_splats.raw_opacities.val(),
        )
        .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn optimizer_checkpoint_restore_rejects_malformed_tensor_state() {
        let device = <GsBackendBase as Backend>::Device::default();
        let mut splats = test_splats(&device);
        let mut optimizer = optimizer_with_scaling(&device);
        let status = DeviceTrainingStatus::<GsBackendBase>::new(&device, 0);
        optimizer_step(&mut optimizer, &mut splats, &status, &device);
        let checkpoint = optimizer.checkpoint().await.expect("export checkpoint");

        let mut missing_pair = checkpoint.clone();
        missing_pair.transforms.moment2 = None;
        assert!(restore_error(&missing_pair, &splats, &device).contains("both be present"));

        let mut wrong_rank = checkpoint.clone();
        wrong_rank.transforms.moment1.as_mut().unwrap().shape = vec![2, 10, 1];
        assert!(restore_error(&wrong_rank, &splats, &device).contains("expected rank 2"));

        let mut wrong_parameter_shape = checkpoint.clone();
        wrong_parameter_shape
            .sh_coeffs
            .moment2
            .as_mut()
            .unwrap()
            .shape = vec![1, 8, 3];
        assert!(restore_error(&wrong_parameter_shape, &splats, &device)
            .contains("expected shape [2, 4, 3]"));

        let mut wrong_scaling_shape = checkpoint.clone();
        wrong_scaling_shape
            .raw_opacities
            .scaling
            .as_mut()
            .unwrap()
            .shape = vec![2];
        assert!(
            restore_error(&wrong_scaling_shape, &splats, &device).contains("expected shape [1]")
        );

        let mut non_finite = checkpoint;
        non_finite.raw_opacities.moment1.as_mut().unwrap().values[0] = f32::NAN;
        assert!(restore_error(&non_finite, &splats, &device).contains("must be finite"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn optimizer_checkpoint_restore_rejects_divergent_steps_without_mutation() {
        let device = <GsBackendBase as Backend>::Device::default();
        let mut splats = test_splats(&device);
        let mut optimizer = optimizer_with_scaling(&device);
        let status = DeviceTrainingStatus::<GsBackendBase>::new(&device, 0);
        optimizer_step(&mut optimizer, &mut splats, &status, &device);
        let mut checkpoint = optimizer.checkpoint().await.expect("export checkpoint");

        checkpoint.transforms.step = 3;
        checkpoint.sh_coeffs.step = 5;
        checkpoint.raw_opacities.step = 7;
        let mut restored = optimizer_with_scaling(&device);
        let before = restored
            .checkpoint()
            .await
            .expect("checkpoint before restore");
        let error = restored
            .restore(&checkpoint, &splats, &device)
            .expect_err("divergent steps must be rejected");
        assert!(matches!(
            error,
            TrainingError::InvalidInput(message) if message.contains("steps must be equal")
        ));
        assert_eq!(
            restored.checkpoint().await.expect("re-export checkpoint"),
            before
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn optimizer_checkpoint_restore_rejects_step_beyond_i32_boundary() {
        let device = <GsBackendBase as Backend>::Device::default();
        let mut splats = test_splats(&device);
        let mut optimizer = optimizer_with_scaling(&device);
        let status = DeviceTrainingStatus::<GsBackendBase>::new(&device, 0);
        optimizer_step(&mut optimizer, &mut splats, &status, &device);
        let mut checkpoint = optimizer.checkpoint().await.expect("export checkpoint");
        let unsafe_step = MAX_TRAINING_ITERATIONS + 1;
        checkpoint.transforms.step = unsafe_step;
        checkpoint.sh_coeffs.step = unsafe_step;
        checkpoint.raw_opacities.step = unsafe_step;

        let mut restored = AdamScaled::<GsBackendBase>::new(AdamScaledConfig::default());
        let error = restored
            .restore(&checkpoint, &splats, &device)
            .expect_err("step beyond the safe Adam boundary must be rejected");
        assert!(matches!(
            error,
            TrainingError::InvalidInput(message) if message.contains("maximum safe step")
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn optimizer_checkpoint_restore_accepts_missing_scaling_and_reset_state() {
        let device = <GsBackendBase as Backend>::Device::default();
        let mut splats = test_splats(&device);
        let mut optimizer = optimizer_with_scaling(&device);
        let status = DeviceTrainingStatus::<GsBackendBase>::new(&device, 0);
        optimizer_step(&mut optimizer, &mut splats, &status, &device);
        let mut checkpoint = optimizer.checkpoint().await.expect("export checkpoint");

        checkpoint.transforms.step = 3;
        checkpoint.sh_coeffs.step = 3;
        checkpoint.raw_opacities.step = 3;
        checkpoint.sh_coeffs.scaling = None;
        let mut restored = AdamScaled::<GsBackendBase>::new(AdamScaledConfig::default());
        restored
            .restore(&checkpoint, &splats, &device)
            .expect("restore equal steps and missing scaling");
        assert_eq!(
            restored.checkpoint().await.expect("re-export checkpoint"),
            checkpoint
        );

        optimizer.reset();
        let reset_checkpoint = optimizer
            .checkpoint()
            .await
            .expect("export reset checkpoint");
        assert_eq!(reset_checkpoint.transforms.step, 0);
        assert!(reset_checkpoint.transforms.moment1.is_none());
        assert!(reset_checkpoint.transforms.moment2.is_none());
        assert!(reset_checkpoint.transforms.scaling.is_some());
        assert_eq!(reset_checkpoint.sh_coeffs.step, 0);
        assert!(reset_checkpoint.sh_coeffs.moment1.is_none());
        assert!(reset_checkpoint.sh_coeffs.scaling.is_some());
        assert_eq!(reset_checkpoint.raw_opacities.step, 0);
        assert!(reset_checkpoint.raw_opacities.moment1.is_none());
        assert!(reset_checkpoint.raw_opacities.scaling.is_some());

        let mut restored_reset = AdamScaled::<GsBackendBase>::new(AdamScaledConfig::default());
        restored_reset
            .restore(&reset_checkpoint, &splats, &device)
            .expect("restore reset state");
        assert_eq!(
            restored_reset
                .checkpoint()
                .await
                .expect("re-export reset state"),
            reset_checkpoint
        );
    }

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

    #[tokio::test(flavor = "current_thread")]
    async fn optimizer_remap_keeps_surviving_moments_and_step() {
        let device = <GsBackendBase as Backend>::Device::default();
        let mut splats = test_splats(&device);
        // Grow to three rows so origins [Some(1), None, Some(0)] are expressible.
        let transforms = Tensor::<GsDiffBackend, 2>::from_data(
            TensorData::new(
                (0..30).map(|value| value as f32 * 0.01).collect(),
                Shape::new([3, 10]),
            ),
            &device,
        );
        let sh_coeffs = Tensor::<GsDiffBackend, 3>::from_data(
            TensorData::new(
                (0..36).map(|value| value as f32 * 0.005).collect(),
                Shape::new([3, 4, 3]),
            ),
            &device,
        );
        let raw_opacities = Tensor::<GsDiffBackend, 1>::from_floats([0.1, -0.2, 0.3], &device);
        splats.transforms = Param::from_tensor(transforms);
        splats.sh_coeffs = Param::from_tensor(sh_coeffs);
        splats.raw_opacities = Param::from_tensor(raw_opacities);

        let mut optimizer = optimizer_with_scaling(&device);
        let status = DeviceTrainingStatus::<GsBackendBase>::new(&device, 0);
        for _ in 0..7 {
            optimizer.step_device_splats(
                &mut splats,
                Tensor::ones([3, 10], &device).mul_scalar(0.1),
                Tensor::ones([3, 4, 3], &device).mul_scalar(-0.2),
                Tensor::ones([3], &device).mul_scalar(0.3),
                status.buffer().clone(),
            );
        }
        let before = optimizer
            .checkpoint()
            .await
            .expect("checkpoint before remap");
        assert_eq!(before.transforms.step, 7);
        assert_eq!(before.sh_coeffs.step, 7);
        assert_eq!(before.raw_opacities.step, 7);

        optimizer.remap_origins(&[Some(1), None, Some(0)], 4, 3, &device);
        let after = optimizer
            .checkpoint()
            .await
            .expect("checkpoint after remap");
        assert_eq!(after.transforms.step, 7);
        assert_eq!(after.sh_coeffs.step, 7);
        assert_eq!(after.raw_opacities.step, 7);

        fn assert_remapped_rows(
            before: &crate::training::AdamParameterCheckpoint,
            after: &crate::training::AdamParameterCheckpoint,
            row_width: usize,
        ) {
            for (label, before_moment, after_moment) in [
                (
                    "moment1",
                    before.moment1.as_ref().expect("before moment1"),
                    after.moment1.as_ref().expect("after moment1"),
                ),
                (
                    "moment2",
                    before.moment2.as_ref().expect("before moment2"),
                    after.moment2.as_ref().expect("after moment2"),
                ),
            ] {
                assert_eq!(
                    after_moment.values.len(),
                    3 * row_width,
                    "{label} remapped length"
                );
                assert_eq!(
                    &after_moment.values[..row_width],
                    &before_moment.values[row_width..2 * row_width],
                    "{label} row0 <- old row1"
                );
                assert!(
                    after_moment.values[row_width..2 * row_width]
                        .iter()
                        .all(|value| *value == 0.0),
                    "{label} new row must be zero"
                );
                assert_eq!(
                    &after_moment.values[2 * row_width..],
                    &before_moment.values[..row_width],
                    "{label} row2 <- old row0"
                );
            }
        }

        assert_remapped_rows(&before.transforms, &after.transforms, 10);
        assert_remapped_rows(&before.sh_coeffs, &after.sh_coeffs, 12);
        assert_remapped_rows(&before.raw_opacities, &after.raw_opacities, 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn optimizer_remap_checkpoint_restore_preserves_next_step_update() {
        let device = <GsBackendBase as Backend>::Device::default();
        let mut splats = test_splats(&device);
        let transforms = Tensor::<GsDiffBackend, 2>::from_data(
            TensorData::new(
                (0..30).map(|value| value as f32 * 0.01).collect(),
                Shape::new([3, 10]),
            ),
            &device,
        );
        let sh_coeffs = Tensor::<GsDiffBackend, 3>::from_data(
            TensorData::new(
                (0..36).map(|value| value as f32 * 0.005).collect(),
                Shape::new([3, 4, 3]),
            ),
            &device,
        );
        let raw_opacities = Tensor::<GsDiffBackend, 1>::from_floats([0.1, -0.2, 0.3], &device);
        splats.transforms = Param::from_tensor(transforms);
        splats.sh_coeffs = Param::from_tensor(sh_coeffs);
        splats.raw_opacities = Param::from_tensor(raw_opacities);

        let mut optimizer = optimizer_with_scaling(&device);
        let status = DeviceTrainingStatus::<GsBackendBase>::new(&device, 0);
        for _ in 0..7 {
            optimizer.step_device_splats(
                &mut splats,
                Tensor::ones([3, 10], &device).mul_scalar(0.1),
                Tensor::ones([3, 4, 3], &device).mul_scalar(-0.2),
                Tensor::ones([3], &device).mul_scalar(0.3),
                status.buffer().clone(),
            );
        }
        optimizer.remap_origins(&[Some(1), None, Some(0)], 4, 3, &device);

        // Resize live parameters to the remapped topology before checkpointing.
        let transforms = Tensor::<GsDiffBackend, 2>::from_data(
            TensorData::new(
                (0..30).map(|value| value as f32 * 0.02).collect(),
                Shape::new([3, 10]),
            ),
            &device,
        );
        let sh_coeffs = Tensor::<GsDiffBackend, 3>::from_data(
            TensorData::new(
                (0..36).map(|value| value as f32 * 0.01).collect(),
                Shape::new([3, 4, 3]),
            ),
            &device,
        );
        let raw_opacities = Tensor::<GsDiffBackend, 1>::from_floats([0.4, -0.1, 0.2], &device);
        splats.transforms = Param::from_tensor(transforms);
        splats.sh_coeffs = Param::from_tensor(sh_coeffs);
        splats.raw_opacities = Param::from_tensor(raw_opacities);

        let checkpoint = optimizer
            .checkpoint()
            .await
            .expect("checkpoint after remap");
        assert_eq!(checkpoint.transforms.step, 7);

        let mut live = AdamScaled::new(AdamScaledConfig {
            lr: 0.01,
            ..AdamScaledConfig::default()
        });
        live.restore(&checkpoint, &splats, &device)
            .expect("restore remapped optimizer");
        live.set_transform_scaling(Tensor::ones([1, 10], &device));
        live.set_sh_scaling(Tensor::ones([1, 4, 1], &device).mul_scalar(0.5));
        live.set_opacity_scaling(Tensor::from_floats([0.25], &device));

        let mut restored_splats = copy_splats(&splats, &device).await;
        let mut restored = AdamScaled::new(AdamScaledConfig {
            lr: 0.01,
            ..AdamScaledConfig::default()
        });
        restored
            .restore(&checkpoint, &restored_splats, &device)
            .expect("restore twin");
        restored.set_transform_scaling(Tensor::ones([1, 10], &device));
        restored.set_sh_scaling(Tensor::ones([1, 4, 1], &device).mul_scalar(0.5));
        restored.set_opacity_scaling(Tensor::from_floats([0.25], &device));

        let post_status = DeviceTrainingStatus::<GsBackendBase>::new(&device, 7);
        live.step_device_splats(
            &mut splats,
            Tensor::ones([3, 10], &device).mul_scalar(0.05),
            Tensor::ones([3, 4, 3], &device).mul_scalar(-0.05),
            Tensor::ones([3], &device).mul_scalar(0.05),
            post_status.buffer().clone(),
        );
        let restored_status = DeviceTrainingStatus::<GsBackendBase>::new(&device, 7);
        restored.step_device_splats(
            &mut restored_splats,
            Tensor::ones([3, 10], &device).mul_scalar(0.05),
            Tensor::ones([3, 4, 3], &device).mul_scalar(-0.05),
            Tensor::ones([3], &device).mul_scalar(0.05),
            restored_status.buffer().clone(),
        );

        let live_ckpt = live.checkpoint().await.expect("live after step");
        let restored_ckpt = restored.checkpoint().await.expect("restored after step");
        assert_eq!(live_ckpt.transforms.step, 8);
        assert_eq!(restored_ckpt.transforms.step, 8);
        assert_eq!(live_ckpt.sh_coeffs.step, 8);
        assert_eq!(restored_ckpt.raw_opacities.step, 8);
        assert_tensor_close(splats.transforms.val(), restored_splats.transforms.val()).await;
        assert_tensor_close(splats.sh_coeffs.val(), restored_splats.sh_coeffs.val()).await;
        assert_tensor_close(
            splats.raw_opacities.val(),
            restored_splats.raw_opacities.val(),
        )
        .await;
        // New row (index 1) moments must leave the zero-init path after one step.
        let m1 = live_ckpt
            .transforms
            .moment1
            .expect("transforms moment1 after step");
        assert!(
            m1.values[10..20].iter().any(|value| *value != 0.0),
            "new-row moment1 must update on the post-restore step"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn opacity_reset_clears_only_opacity_moments_and_keeps_step() {
        let device = <GsBackendBase as Backend>::Device::default();
        let mut splats = test_splats(&device);
        let mut optimizer = optimizer_with_scaling(&device);
        let status = DeviceTrainingStatus::<GsBackendBase>::new(&device, 0);
        for _ in 0..5 {
            optimizer_step(&mut optimizer, &mut splats, &status, &device);
        }
        let before = optimizer.checkpoint().await.expect("before opacity reset");
        assert_eq!(before.transforms.step, 5);
        assert!(before.transforms.moment1.is_some());
        assert!(before.raw_opacities.moment1.is_some());

        optimizer.clear_opacity_moments();
        let after = optimizer.checkpoint().await.expect("after opacity reset");
        assert_eq!(after.transforms.step, 5);
        assert_eq!(after.sh_coeffs.step, 5);
        assert_eq!(after.raw_opacities.step, 5);
        assert_eq!(after.transforms.moment1, before.transforms.moment1);
        assert_eq!(after.transforms.moment2, before.transforms.moment2);
        assert_eq!(after.sh_coeffs.moment1, before.sh_coeffs.moment1);
        assert_eq!(after.sh_coeffs.moment2, before.sh_coeffs.moment2);
        let opacity_m1 = after.raw_opacities.moment1.expect("opacity moment1");
        let opacity_m2 = after.raw_opacities.moment2.expect("opacity moment2");
        assert!(opacity_m1.values.iter().all(|value| *value == 0.0));
        assert!(opacity_m2.values.iter().all(|value| *value == 0.0));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn remap_mixed_prune_and_densify_keeps_survivor_moments_and_zeros_new_rows() {
        let device = <GsBackendBase as Backend>::Device::default();
        // Start with 5 rows so a non-contiguous prune + densify is expressible.
        let transforms = Tensor::<GsDiffBackend, 2>::from_data(
            TensorData::new(
                (0..50).map(|value| value as f32 * 0.01).collect(),
                Shape::new([5, 10]),
            ),
            &device,
        );
        let sh_coeffs = Tensor::<GsDiffBackend, 3>::from_data(
            TensorData::new(
                (0..60).map(|value| value as f32 * 0.005).collect(),
                Shape::new([5, 4, 3]),
            ),
            &device,
        );
        let raw_opacities =
            Tensor::<GsDiffBackend, 1>::from_floats([0.1, -0.2, 0.3, -0.4, 0.5], &device);
        let mut splats = DeviceSplats {
            transforms: Param::from_tensor(transforms),
            sh_coeffs: Param::from_tensor(sh_coeffs),
            raw_opacities: Param::from_tensor(raw_opacities),
            sh_degree: 1,
        };
        let mut optimizer = optimizer_with_scaling(&device);
        let status = DeviceTrainingStatus::<GsBackendBase>::new(&device, 0);
        for _ in 0..4 {
            optimizer.step_device_splats(
                &mut splats,
                Tensor::ones([5, 10], &device).mul_scalar(0.1),
                Tensor::ones([5, 4, 3], &device).mul_scalar(-0.2),
                Tensor::ones([5], &device).mul_scalar(0.3),
                status.buffer().clone(),
            );
        }
        let before = optimizer.checkpoint().await.expect("before remap");
        // Keep rows 4 and 1, insert two densified rows, drop 0/2/3.
        let origins = [Some(4), None, Some(1), None];
        optimizer.remap_origins(&origins, 4, 3, &device);
        let after = optimizer.checkpoint().await.expect("after remap");
        assert_eq!(after.transforms.step, 4);
        assert_eq!(after.sh_coeffs.step, 4);
        assert_eq!(after.raw_opacities.step, 4);

        let before_m1 = before.transforms.moment1.expect("before moment1");
        let after_m1 = after.transforms.moment1.expect("after moment1");
        assert_eq!(after_m1.shape, [4, 10]);
        assert_eq!(&after_m1.values[0..10], &before_m1.values[40..50]);
        assert!(after_m1.values[10..20].iter().all(|value| *value == 0.0));
        assert_eq!(&after_m1.values[20..30], &before_m1.values[10..20]);
        assert!(after_m1.values[30..40].iter().all(|value| *value == 0.0));
    }
}
