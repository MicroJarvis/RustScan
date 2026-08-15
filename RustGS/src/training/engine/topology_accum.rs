use burn::prelude::*;
use burn::tensor::{TensorMetadata, TensorPrimitive};
use burn_cubecl::cubecl::{prelude::KernelId, server::KernelArguments, CubeCount};
use burn_cubecl::{kernel::into_contiguous, BoolElement, CubeBackend, FloatElement, IntElement};
use burn_wgpu::{CubeDim, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};
use bytemuck::{Pod, Zeroable};

use crate::training::TopologyCheckpoint;
use crate::TrainingError;

use super::optimizer::{
    tensor_checkpoint, tensor_from_validated_checkpoint, validate_tensor_checkpoint,
};

const WORKGROUP_SIZE: u32 = 256;
const SHADER_SRC: &str = include_str!("../shaders/accumulate_topology_stats.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AccumulateTopologyStatsParams {
    num_splats: u32,
    sh_coeffs: u32,
    use_actual_visibility: u32,
    collect_actual_visibility: u32,
}

struct AccumulateTopologyStatsRaw;

impl AccumulateTopologyStatsRaw {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(SHADER_SRC)
    }
}

#[derive(Debug)]
struct AccumulateTopologyStatsKernel;

impl KernelSource for AccumulateTopologyStatsKernel {
    fn source(&self) -> SourceTemplate {
        AccumulateTopologyStatsRaw.source()
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

pub(crate) struct TopologyAccumulatorSet<B: Backend> {
    pub(crate) grad_2d: Tensor<B, 1>,
    pub(crate) screen_grad_2d: Tensor<B, 1>,
    pub(crate) abs_grad_2d: Tensor<B, 1>,
    pub(crate) abs_pixel_grad_2d: Tensor<B, 1>,
    pub(crate) pixel_coverage: Tensor<B, 1>,
    pub(crate) camera_depth: Tensor<B, 1>,
    pub(crate) grad_color: Tensor<B, 1>,
    pub(crate) num_observations: Tensor<B, 1>,
    pub(crate) visible_observations: Tensor<B, 1>,
    pub(crate) actual_visible_observations: Tensor<B, 1>,
}

impl<B: Backend> TopologyAccumulatorSet<B> {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) async fn checkpoint(
        &self,
        splat_birth_iterations: &[usize],
        splat_invisible_windows: &[usize],
    ) -> Result<TopologyCheckpoint, TrainingError> {
        let checkpoint = TopologyCheckpoint {
            grad_2d: tensor_checkpoint(&self.grad_2d).await?,
            screen_grad_2d: tensor_checkpoint(&self.screen_grad_2d).await?,
            abs_grad_2d: tensor_checkpoint(&self.abs_grad_2d).await?,
            abs_pixel_grad_2d: tensor_checkpoint(&self.abs_pixel_grad_2d).await?,
            pixel_coverage: tensor_checkpoint(&self.pixel_coverage).await?,
            camera_depth: tensor_checkpoint(&self.camera_depth).await?,
            grad_color: tensor_checkpoint(&self.grad_color).await?,
            num_observations: tensor_checkpoint(&self.num_observations).await?,
            visible_observations: tensor_checkpoint(&self.visible_observations).await?,
            actual_visible_observations: tensor_checkpoint(&self.actual_visible_observations)
                .await?,
            splat_birth_iterations: splat_birth_iterations.to_vec(),
            splat_invisible_windows: splat_invisible_windows.to_vec(),
        };
        validate_topology_checkpoint(&checkpoint, self.grad_2d.dims()[0])?;
        Ok(checkpoint)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn from_checkpoint(
        checkpoint: &TopologyCheckpoint,
        splat_count: usize,
        device: &B::Device,
    ) -> Result<Self, TrainingError> {
        validate_topology_checkpoint(checkpoint, splat_count)?;

        Ok(Self {
            grad_2d: tensor_from_validated_checkpoint(&checkpoint.grad_2d, device),
            screen_grad_2d: tensor_from_validated_checkpoint(&checkpoint.screen_grad_2d, device),
            abs_grad_2d: tensor_from_validated_checkpoint(&checkpoint.abs_grad_2d, device),
            abs_pixel_grad_2d: tensor_from_validated_checkpoint(
                &checkpoint.abs_pixel_grad_2d,
                device,
            ),
            pixel_coverage: tensor_from_validated_checkpoint(&checkpoint.pixel_coverage, device),
            camera_depth: tensor_from_validated_checkpoint(&checkpoint.camera_depth, device),
            grad_color: tensor_from_validated_checkpoint(&checkpoint.grad_color, device),
            num_observations: tensor_from_validated_checkpoint(
                &checkpoint.num_observations,
                device,
            ),
            visible_observations: tensor_from_validated_checkpoint(
                &checkpoint.visible_observations,
                device,
            ),
            actual_visible_observations: tensor_from_validated_checkpoint(
                &checkpoint.actual_visible_observations,
                device,
            ),
        })
    }
}

fn validate_topology_checkpoint(
    checkpoint: &TopologyCheckpoint,
    splat_count: usize,
) -> Result<(), TrainingError> {
    let tensors = [
        ("topology.grad_2d", &checkpoint.grad_2d),
        ("topology.screen_grad_2d", &checkpoint.screen_grad_2d),
        ("topology.abs_grad_2d", &checkpoint.abs_grad_2d),
        ("topology.abs_pixel_grad_2d", &checkpoint.abs_pixel_grad_2d),
        ("topology.pixel_coverage", &checkpoint.pixel_coverage),
        ("topology.camera_depth", &checkpoint.camera_depth),
        ("topology.grad_color", &checkpoint.grad_color),
        ("topology.num_observations", &checkpoint.num_observations),
        (
            "topology.visible_observations",
            &checkpoint.visible_observations,
        ),
        (
            "topology.actual_visible_observations",
            &checkpoint.actual_visible_observations,
        ),
    ];
    for (name, tensor) in tensors {
        validate_tensor_checkpoint(name, tensor, [splat_count])?;
    }
    validate_topology_vector(
        "topology.splat_birth_iterations",
        checkpoint.splat_birth_iterations.len(),
        splat_count,
    )?;
    validate_topology_vector(
        "topology.splat_invisible_windows",
        checkpoint.splat_invisible_windows.len(),
        splat_count,
    )?;
    Ok(())
}

fn validate_topology_vector(
    name: &str,
    actual_len: usize,
    splat_count: usize,
) -> Result<(), TrainingError> {
    if actual_len != splat_count {
        return Err(TrainingError::InvalidInput(format!(
            "invalid topology checkpoint: {name} must contain {splat_count} values, got {actual_len}"
        )));
    }
    Ok(())
}

pub(crate) trait TopologyAccumBackend: Backend {
    #[allow(clippy::too_many_arguments)]
    fn accumulate_topology_stats_primitive(
        transforms_grad: Self::FloatTensorPrimitive,
        screen_grad_stats: Self::FloatTensorPrimitive,
        sh_grad: Self::FloatTensorPrimitive,
        visible: Self::FloatTensorPrimitive,
        grad_2d_accum: Self::FloatTensorPrimitive,
        screen_grad_2d_accum: Self::FloatTensorPrimitive,
        abs_grad_2d_accum: Self::FloatTensorPrimitive,
        abs_pixel_grad_2d_accum: Self::FloatTensorPrimitive,
        pixel_coverage_accum: Self::FloatTensorPrimitive,
        camera_depth_accum: Self::FloatTensorPrimitive,
        grad_color_accum: Self::FloatTensorPrimitive,
        num_observations: Self::FloatTensorPrimitive,
        visible_observations: Self::FloatTensorPrimitive,
        actual_visible_observations: Self::FloatTensorPrimitive,
        use_actual_visibility: bool,
        collect_actual_visibility: bool,
    ) -> TopologyAccumulatorSet<Self>;
}

impl<F, I, BT> TopologyAccumBackend for CubeBackend<WgpuRuntime, F, I, BT>
where
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn accumulate_topology_stats_primitive(
        transforms_grad: Self::FloatTensorPrimitive,
        screen_grad_stats: Self::FloatTensorPrimitive,
        sh_grad: Self::FloatTensorPrimitive,
        visible: Self::FloatTensorPrimitive,
        grad_2d_accum: Self::FloatTensorPrimitive,
        screen_grad_2d_accum: Self::FloatTensorPrimitive,
        abs_grad_2d_accum: Self::FloatTensorPrimitive,
        abs_pixel_grad_2d_accum: Self::FloatTensorPrimitive,
        pixel_coverage_accum: Self::FloatTensorPrimitive,
        camera_depth_accum: Self::FloatTensorPrimitive,
        grad_color_accum: Self::FloatTensorPrimitive,
        num_observations: Self::FloatTensorPrimitive,
        visible_observations: Self::FloatTensorPrimitive,
        actual_visible_observations: Self::FloatTensorPrimitive,
        use_actual_visibility: bool,
        collect_actual_visibility: bool,
    ) -> TopologyAccumulatorSet<Self> {
        let transforms_grad = into_contiguous(transforms_grad);
        let screen_grad_stats = into_contiguous(screen_grad_stats);
        let sh_grad = into_contiguous(sh_grad);
        let visible = into_contiguous(visible);
        let grad_2d_accum = into_contiguous(grad_2d_accum);
        let screen_grad_2d_accum = into_contiguous(screen_grad_2d_accum);
        let abs_grad_2d_accum = into_contiguous(abs_grad_2d_accum);
        let abs_pixel_grad_2d_accum = into_contiguous(abs_pixel_grad_2d_accum);
        let pixel_coverage_accum = into_contiguous(pixel_coverage_accum);
        let camera_depth_accum = into_contiguous(camera_depth_accum);
        let grad_color_accum = into_contiguous(grad_color_accum);
        let num_observations = into_contiguous(num_observations);
        let visible_observations = into_contiguous(visible_observations);
        let actual_visible_observations = into_contiguous(actual_visible_observations);

        let num_splats = transforms_grad.shape()[0];
        if num_splats > 0 {
            let sh_shape = sh_grad.shape();
            let sh_coeffs = sh_shape.get(1).copied().unwrap_or(0);
            let params = AccumulateTopologyStatsParams {
                num_splats: num_splats as u32,
                sh_coeffs: sh_coeffs as u32,
                use_actual_visibility: u32::from(use_actual_visibility),
                collect_actual_visibility: u32::from(collect_actual_visibility),
            };
            let params_handle = transforms_grad
                .client
                .create_from_slice(bytemuck::bytes_of(&params));
            transforms_grad.client.launch(
                Box::new(SourceKernel::new(
                    AccumulateTopologyStatsKernel,
                    CubeDim::new_1d(WORKGROUP_SIZE),
                )),
                CubeCount::Static((num_splats as u32).div_ceil(WORKGROUP_SIZE), 1, 1),
                KernelArguments::new().with_buffers(vec![
                    transforms_grad.handle.binding(),
                    screen_grad_stats.handle.binding(),
                    sh_grad.handle.binding(),
                    visible.handle.binding(),
                    grad_2d_accum.handle.clone().binding(),
                    screen_grad_2d_accum.handle.clone().binding(),
                    abs_grad_2d_accum.handle.clone().binding(),
                    abs_pixel_grad_2d_accum.handle.clone().binding(),
                    pixel_coverage_accum.handle.clone().binding(),
                    camera_depth_accum.handle.clone().binding(),
                    grad_color_accum.handle.clone().binding(),
                    num_observations.handle.clone().binding(),
                    visible_observations.handle.clone().binding(),
                    actual_visible_observations.handle.clone().binding(),
                    params_handle.binding(),
                ]),
            );
        }

        TopologyAccumulatorSet {
            grad_2d: Tensor::from_primitive(TensorPrimitive::Float(grad_2d_accum)),
            screen_grad_2d: Tensor::from_primitive(TensorPrimitive::Float(screen_grad_2d_accum)),
            abs_grad_2d: Tensor::from_primitive(TensorPrimitive::Float(abs_grad_2d_accum)),
            abs_pixel_grad_2d: Tensor::from_primitive(TensorPrimitive::Float(
                abs_pixel_grad_2d_accum,
            )),
            pixel_coverage: Tensor::from_primitive(TensorPrimitive::Float(pixel_coverage_accum)),
            camera_depth: Tensor::from_primitive(TensorPrimitive::Float(camera_depth_accum)),
            grad_color: Tensor::from_primitive(TensorPrimitive::Float(grad_color_accum)),
            num_observations: Tensor::from_primitive(TensorPrimitive::Float(num_observations)),
            visible_observations: Tensor::from_primitive(TensorPrimitive::Float(
                visible_observations,
            )),
            actual_visible_observations: Tensor::from_primitive(TensorPrimitive::Float(
                actual_visible_observations,
            )),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn accumulate_topology_stats<B: TopologyAccumBackend>(
    transforms_grad: Tensor<B, 2>,
    screen_grad_stats: Tensor<B, 2>,
    sh_grad: Tensor<B, 3>,
    visible: Tensor<B, 1>,
    accum: TopologyAccumulatorSet<B>,
    use_actual_visibility: bool,
    collect_actual_visibility: bool,
) -> TopologyAccumulatorSet<B> {
    B::accumulate_topology_stats_primitive(
        transforms_grad.into_primitive().tensor(),
        screen_grad_stats.into_primitive().tensor(),
        sh_grad.into_primitive().tensor(),
        visible.into_primitive().tensor(),
        accum.grad_2d.into_primitive().tensor(),
        accum.screen_grad_2d.into_primitive().tensor(),
        accum.abs_grad_2d.into_primitive().tensor(),
        accum.abs_pixel_grad_2d.into_primitive().tensor(),
        accum.pixel_coverage.into_primitive().tensor(),
        accum.camera_depth.into_primitive().tensor(),
        accum.grad_color.into_primitive().tensor(),
        accum.num_observations.into_primitive().tensor(),
        accum.visible_observations.into_primitive().tensor(),
        accum.actual_visible_observations.into_primitive().tensor(),
        use_actual_visibility,
        collect_actual_visibility,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::engine::GsBackendBase;
    use crate::training::TensorCheckpoint;

    fn accumulator_fixture(
        device: &<GsBackendBase as Backend>::Device,
    ) -> TopologyAccumulatorSet<GsBackendBase> {
        let tensor =
            |offset: f32| Tensor::from_floats([offset + 1.0, offset + 2.0, offset + 3.0], device);
        TopologyAccumulatorSet {
            grad_2d: tensor(0.0),
            screen_grad_2d: tensor(10.0),
            abs_grad_2d: tensor(20.0),
            abs_pixel_grad_2d: tensor(30.0),
            pixel_coverage: tensor(40.0),
            camera_depth: tensor(50.0),
            grad_color: tensor(60.0),
            num_observations: tensor(70.0),
            visible_observations: tensor(80.0),
            actual_visible_observations: tensor(90.0),
        }
    }

    fn checkpoint_tensor(offset: f32) -> TensorCheckpoint {
        TensorCheckpoint {
            shape: vec![3],
            values: vec![offset + 1.0, offset + 2.0, offset + 3.0],
        }
    }

    fn checkpoint_fixture() -> TopologyCheckpoint {
        TopologyCheckpoint {
            grad_2d: checkpoint_tensor(0.0),
            screen_grad_2d: checkpoint_tensor(10.0),
            abs_grad_2d: checkpoint_tensor(20.0),
            abs_pixel_grad_2d: checkpoint_tensor(30.0),
            pixel_coverage: checkpoint_tensor(40.0),
            camera_depth: checkpoint_tensor(50.0),
            grad_color: checkpoint_tensor(60.0),
            num_observations: checkpoint_tensor(70.0),
            visible_observations: checkpoint_tensor(80.0),
            actual_visible_observations: checkpoint_tensor(90.0),
            splat_birth_iterations: vec![0, 1, 2],
            splat_invisible_windows: vec![0, 1, 2],
        }
    }

    fn restore_error(
        checkpoint: &TopologyCheckpoint,
        device: &<GsBackendBase as Backend>::Device,
    ) -> String {
        match TopologyAccumulatorSet::<GsBackendBase>::from_checkpoint(checkpoint, 3, device) {
            Ok(_) => panic!("malformed topology checkpoint must be rejected"),
            Err(error) => error.to_string(),
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn topology_checkpoint_direct_export_preflights_tensors_and_vectors() {
        let device = <GsBackendBase as Backend>::Device::default();

        let mut wrong_count = accumulator_fixture(&device);
        wrong_count.screen_grad_2d = Tensor::zeros([2], &device);
        let error = wrong_count
            .checkpoint(&[0, 1, 2], &[0, 1, 2])
            .await
            .expect_err("non-uniform tensor counts must be rejected")
            .to_string();
        assert!(error.contains("screen_grad_2d"));

        let mut non_finite = accumulator_fixture(&device);
        non_finite.grad_color = Tensor::from_floats([1.0, f32::NAN, 3.0], &device);
        let error = non_finite
            .checkpoint(&[0, 1, 2], &[0, 1, 2])
            .await
            .expect_err("non-finite tensor values must be rejected")
            .to_string();
        assert!(error.contains("grad_color") && error.contains("finite"));

        let accumulators = accumulator_fixture(&device);
        let error = accumulators
            .checkpoint(&[0, 1], &[0, 1, 2])
            .await
            .expect_err("birth vector mismatch must be rejected")
            .to_string();
        assert!(error.contains("splat_birth_iterations"));
        let error = accumulators
            .checkpoint(&[0, 1, 2], &[0, 1])
            .await
            .expect_err("invisible vector mismatch must be rejected")
            .to_string();
        assert!(error.contains("splat_invisible_windows"));
    }

    #[test]
    fn topology_checkpoint_direct_restore_preflights_tensors_and_vectors() {
        let device = <GsBackendBase as Backend>::Device::default();

        let mut wrong_shape = checkpoint_fixture();
        wrong_shape.abs_grad_2d.shape = vec![1, 3];
        assert!(restore_error(&wrong_shape, &device).contains("abs_grad_2d"));

        let mut wrong_len = checkpoint_fixture();
        wrong_len.camera_depth.values.pop();
        assert!(restore_error(&wrong_len, &device).contains("camera_depth"));

        let mut non_finite = checkpoint_fixture();
        non_finite.visible_observations.values[1] = f32::INFINITY;
        let error = restore_error(&non_finite, &device);
        assert!(error.contains("visible_observations") && error.contains("finite"));

        let mut birth_mismatch = checkpoint_fixture();
        birth_mismatch.splat_birth_iterations.pop();
        assert!(restore_error(&birth_mismatch, &device).contains("splat_birth_iterations"));

        let mut invisible_mismatch = checkpoint_fixture();
        invisible_mismatch.splat_invisible_windows.pop();
        assert!(restore_error(&invisible_mismatch, &device).contains("splat_invisible_windows"));
    }
}
