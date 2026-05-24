use burn::module::Param;
use burn::prelude::*;
use burn::tensor::{Shape, TensorData, Transaction};

use crate::core::HostSplats;

/// GPU-resident differentiable Gaussian splat set.
///
/// Packed layout:
/// - `transforms`: `[N, 10]` = position xyz + quaternion wxyz + log-scale xyz
/// - `sh_coeffs`: `[N, K, 3]`
/// - `raw_opacities`: `[N]`
pub struct DeviceSplats<B: Backend> {
    pub transforms: Param<Tensor<B, 2>>,
    pub sh_coeffs: Param<Tensor<B, 3>>,
    pub raw_opacities: Param<Tensor<B, 1>>,
    pub sh_degree: u32,
}

impl<B: Backend> DeviceSplats<B> {
    pub fn num_splats(&self) -> usize {
        self.transforms.val().dims()[0]
    }
}

pub fn host_splats_to_device<B: Backend>(hs: &HostSplats, device: &B::Device) -> DeviceSplats<B> {
    let num_splats = hs.len();
    let sh_degree = hs.sh_degree() as u32;
    let num_coeffs = ((sh_degree + 1) * (sh_degree + 1)) as usize;

    let mut transforms = Vec::with_capacity(num_splats * 10);
    for idx in 0..num_splats {
        transforms.extend_from_slice(&hs.position(idx));
        transforms.extend_from_slice(&hs.rotation(idx));
        transforms.extend_from_slice(&hs.log_scale(idx));
    }

    let transforms = Tensor::<B, 2>::from_data(
        TensorData::new(transforms, Shape::new([num_splats, 10])),
        device,
    );
    let sh_coeffs = Tensor::<B, 3>::from_data(
        TensorData::new(
            hs.as_view().sh_coeffs.to_vec(),
            Shape::new([num_splats, num_coeffs, 3]),
        ),
        device,
    );
    let raw_opacities = Tensor::<B, 1>::from_data(
        TensorData::new(
            hs.as_view().opacity_logits.to_vec(),
            Shape::new([num_splats]),
        ),
        device,
    );

    DeviceSplats {
        transforms: Param::from_tensor(transforms),
        sh_coeffs: Param::from_tensor(sh_coeffs),
        raw_opacities: Param::from_tensor(raw_opacities),
        sh_degree,
    }
}

pub async fn device_splats_to_host<B: Backend>(splats: &DeviceSplats<B>) -> HostSplats {
    let data = Transaction::default()
        .register(splats.transforms.val())
        .register(splats.sh_coeffs.val())
        .register(splats.raw_opacities.val())
        .execute_async()
        .await
        .expect("device splat readback");

    let transforms = data[0]
        .clone()
        .into_vec::<f32>()
        .expect("transforms readback");
    let sh_coeffs = data[1]
        .clone()
        .into_vec::<f32>()
        .expect("sh coeffs readback");
    let raw_opacities = data[2]
        .clone()
        .into_vec::<f32>()
        .expect("raw opacities readback");

    let num_splats = raw_opacities.len();
    let mut positions = Vec::with_capacity(num_splats * 3);
    let mut rotations = Vec::with_capacity(num_splats * 4);
    let mut log_scales = Vec::with_capacity(num_splats * 3);

    for idx in 0..num_splats {
        let base = idx * 10;
        positions.extend_from_slice(&transforms[base..base + 3]);
        rotations.extend_from_slice(&transforms[base + 3..base + 7]);
        log_scales.extend_from_slice(&transforms[base + 7..base + 10]);
    }

    HostSplats::from_components(
        positions,
        log_scales,
        rotations,
        raw_opacities,
        sh_coeffs,
        splats.sh_degree as usize,
    )
    .expect("valid host splats")
}
