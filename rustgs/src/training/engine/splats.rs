use burn::module::Param;
use burn::prelude::*;
use burn::tensor::{Shape, TensorData, Transaction};

use crate::core::HostSplats;
use crate::TrainingError;

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

pub async fn try_device_splats_to_host<B: Backend>(
    splats: &DeviceSplats<B>,
) -> Result<HostSplats, TrainingError> {
    let transforms_shape = splats.transforms.val().dims();
    let sh_shape = splats.sh_coeffs.val().dims();
    let opacity_shape = splats.raw_opacities.val().dims();
    let num_splats = transforms_shape[0];
    if transforms_shape[1] != 10 {
        return Err(invalid_device_splats(format!(
            "transforms expected shape [N, 10], got {transforms_shape:?}"
        )));
    }
    if sh_shape[0] != num_splats {
        return Err(invalid_device_splats(format!(
            "SH splat count {} does not match transforms splat count {num_splats}",
            sh_shape[0]
        )));
    }
    let sh_order = (splats.sh_degree as usize)
        .checked_add(1)
        .ok_or_else(|| invalid_device_splats("stored SH degree overflows usize"))?;
    let expected_sh_coeffs = sh_order
        .checked_mul(sh_order)
        .ok_or_else(|| invalid_device_splats("stored SH coefficient count overflows usize"))?;
    if sh_shape[1] != expected_sh_coeffs {
        return Err(invalid_device_splats(format!(
            "SH coefficient count {} does not match degree {} expected {expected_sh_coeffs}",
            sh_shape[1], splats.sh_degree
        )));
    }
    if sh_shape[2] != 3 {
        return Err(invalid_device_splats(format!(
            "SH channel count must be 3, got {}",
            sh_shape[2]
        )));
    }
    if opacity_shape[0] != num_splats {
        return Err(invalid_device_splats(format!(
            "opacity splat count {} does not match transforms splat count {num_splats}",
            opacity_shape[0]
        )));
    }

    let data = Transaction::default()
        .register(splats.transforms.val())
        .register(splats.sh_coeffs.val())
        .register(splats.raw_opacities.val())
        .execute_async()
        .await
        .map_err(|error| TrainingError::Gpu(format!("device splat readback failed: {error}")))?;
    let mut data = data.into_iter();
    let transforms = data
        .next()
        .ok_or_else(|| TrainingError::Gpu("missing transforms readback".to_string()))?
        .into_vec::<f32>()
        .map_err(|error| TrainingError::Gpu(format!("transforms conversion failed: {error}")))?;
    let sh_coeffs = data
        .next()
        .ok_or_else(|| TrainingError::Gpu("missing SH coefficients readback".to_string()))?
        .into_vec::<f32>()
        .map_err(|error| TrainingError::Gpu(format!("SH conversion failed: {error}")))?;
    let raw_opacities = data
        .next()
        .ok_or_else(|| TrainingError::Gpu("missing opacity readback".to_string()))?
        .into_vec::<f32>()
        .map_err(|error| TrainingError::Gpu(format!("opacity conversion failed: {error}")))?;

    let transforms_len = checked_splat_values(num_splats, 10, "transforms")?;
    let sh_rows = checked_splat_values(num_splats, expected_sh_coeffs, "SH coefficients")?;
    let sh_len = checked_splat_values(sh_rows, 3, "SH coefficients")?;
    validate_readback_len("transforms", transforms.len(), transforms_len)?;
    validate_readback_len("SH coefficients", sh_coeffs.len(), sh_len)?;
    validate_readback_len("raw opacities", raw_opacities.len(), num_splats)?;

    let mut positions = Vec::with_capacity(checked_splat_values(num_splats, 3, "positions")?);
    let mut rotations = Vec::with_capacity(checked_splat_values(num_splats, 4, "rotations")?);
    let mut log_scales = Vec::with_capacity(checked_splat_values(num_splats, 3, "log scales")?);

    for row in transforms.chunks_exact(10) {
        positions.extend_from_slice(&row[..3]);
        rotations.extend_from_slice(&row[3..7]);
        log_scales.extend_from_slice(&row[7..]);
    }

    HostSplats::from_components(
        positions,
        log_scales,
        rotations,
        raw_opacities,
        sh_coeffs,
        splats.sh_degree as usize,
    )
}

pub async fn device_splats_to_host<B: Backend>(splats: &DeviceSplats<B>) -> HostSplats {
    try_device_splats_to_host(splats)
        .await
        .expect("device splat readback")
}

fn checked_splat_values(rows: usize, row_width: usize, name: &str) -> Result<usize, TrainingError> {
    rows.checked_mul(row_width)
        .ok_or_else(|| invalid_device_splats(format!("{name} element count overflows usize")))
}

fn validate_readback_len(name: &str, actual: usize, expected: usize) -> Result<(), TrainingError> {
    if actual != expected {
        return Err(TrainingError::Gpu(format!(
            "{name} readback expected {expected} values, got {actual}"
        )));
    }
    Ok(())
}

fn invalid_device_splats(message: impl Into<String>) -> TrainingError {
    TrainingError::InvalidInput(format!("invalid device splats: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::engine::{GsBackendBase, GsDiffBackend};

    fn device_splats_with_shapes(
        device: &<GsBackendBase as Backend>::Device,
        transforms_shape: [usize; 2],
        sh_shape: [usize; 3],
        opacity_shape: [usize; 1],
    ) -> DeviceSplats<GsDiffBackend> {
        DeviceSplats {
            transforms: Param::from_tensor(Tensor::zeros(transforms_shape, device)),
            sh_coeffs: Param::from_tensor(Tensor::zeros(sh_shape, device)),
            raw_opacities: Param::from_tensor(Tensor::zeros(opacity_shape, device)),
            sh_degree: 1,
        }
    }

    async fn readback_error(splats: &DeviceSplats<GsDiffBackend>) -> String {
        try_device_splats_to_host(splats)
            .await
            .expect_err("malformed device splats must be rejected")
            .to_string()
    }

    #[tokio::test(flavor = "current_thread")]
    async fn try_device_splats_to_host_rejects_inconsistent_shapes_before_readback() {
        let device = <GsBackendBase as Backend>::Device::default();

        let wrong_transforms = device_splats_with_shapes(&device, [3, 9], [3, 4, 3], [3]);
        assert!(readback_error(&wrong_transforms).await.contains("[N, 10]"));

        let wrong_sh_count = device_splats_with_shapes(&device, [3, 10], [2, 4, 3], [3]);
        assert!(readback_error(&wrong_sh_count)
            .await
            .contains("SH splat count"));

        let wrong_sh_coeffs = device_splats_with_shapes(&device, [3, 10], [3, 3, 3], [3]);
        assert!(readback_error(&wrong_sh_coeffs)
            .await
            .contains("SH coefficient count"));

        let wrong_sh_channels = device_splats_with_shapes(&device, [3, 10], [3, 4, 2], [3]);
        assert!(readback_error(&wrong_sh_channels)
            .await
            .contains("SH channel count"));

        let wrong_opacity_count = device_splats_with_shapes(&device, [3, 10], [3, 4, 3], [2]);
        assert!(readback_error(&wrong_opacity_count)
            .await
            .contains("opacity splat count"));
    }
}
