use burn::prelude::*;
use burn::tensor::{TensorMetadata, TensorPrimitive};
use burn_cubecl::cubecl::{prelude::KernelId, server::KernelArguments, CubeCount};
use burn_cubecl::{kernel::into_contiguous, BoolElement, CubeBackend, FloatElement, IntElement};
use burn_wgpu::{CubeDim, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};
use bytemuck::{Pod, Zeroable};

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
