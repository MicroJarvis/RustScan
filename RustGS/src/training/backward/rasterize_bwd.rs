use burn::prelude::*;
use burn::tensor::{ops::FloatTensorOps, FloatDType, Int, Tensor, TensorPrimitive};
use burn_cubecl::cubecl::{prelude::KernelId, server::KernelArguments, CubeCount};
use burn_cubecl::{kernel::into_contiguous, BoolElement, CubeBackend, FloatElement, IntElement};
use burn_wgpu::{CubeDim, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};
use bytemuck::{Pod, Zeroable};

use crate::training::forward::{compose_shader, TILE_SIZE};

const SHADER_SRC: &str = include_str!("../shaders/rasterize_backwards.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RasterizeBwdUniforms {
    tile_bounds: [u32; 2],
    img_size: [u32; 2],
    background: [f32; 4],
}

struct RasterizeBwdRaw;

impl RasterizeBwdRaw {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(compose_shader("rasterize_backwards.wgsl", SHADER_SRC))
    }
}

#[derive(Debug)]
struct RasterizeBwdKernel;

impl KernelSource for RasterizeBwdKernel {
    fn source(&self) -> SourceTemplate {
        RasterizeBwdRaw.source()
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

pub(crate) trait RasterizeBwdBackend: Backend {
    #[allow(clippy::too_many_arguments)]
    fn rasterize_bwd_primitive(
        compact_gid_from_isect: Self::IntTensorPrimitive,
        tile_offsets: Self::IntTensorPrimitive,
        projected_splats: Self::FloatTensorPrimitive,
        out_img: Self::FloatTensorPrimitive,
        v_output: Self::FloatTensorPrimitive,
        num_visible: usize,
        img_size: (u32, u32),
        tile_bounds: (u32, u32),
        background: [f32; 3],
    ) -> RasterizeBwdPrimitiveOutput<Self>;
}

pub(crate) struct RasterizeBwdPrimitiveOutput<B: Backend> {
    pub(crate) v_splats: B::FloatTensorPrimitive,
    pub(crate) screen_grad_splats: B::FloatTensorPrimitive,
}

pub(crate) struct RasterizeBwdOutput<B: Backend> {
    pub(crate) v_splats: Tensor<B, 2>,
    pub(crate) screen_grad_splats: Tensor<B, 2>,
}

impl<F, I, BT> RasterizeBwdBackend for CubeBackend<WgpuRuntime, F, I, BT>
where
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn rasterize_bwd_primitive(
        compact_gid_from_isect: Self::IntTensorPrimitive,
        tile_offsets: Self::IntTensorPrimitive,
        projected_splats: Self::FloatTensorPrimitive,
        out_img: Self::FloatTensorPrimitive,
        v_output: Self::FloatTensorPrimitive,
        num_visible: usize,
        img_size: (u32, u32),
        tile_bounds: (u32, u32),
        background: [f32; 3],
    ) -> RasterizeBwdPrimitiveOutput<Self> {
        let compact_gid_from_isect = into_contiguous(compact_gid_from_isect);
        let tile_offsets = into_contiguous(tile_offsets);
        let projected_splats = into_contiguous(projected_splats);
        let out_img = into_contiguous(out_img);
        let v_output = into_contiguous(v_output);
        let device = projected_splats.device.clone();
        let client = projected_splats.client.clone();

        let v_splats = <Self as FloatTensorOps<Self>>::float_zeros(
            [num_visible, 10].into(),
            &device,
            FloatDType::F32,
        );
        let screen_grad_splats = <Self as FloatTensorOps<Self>>::float_zeros(
            [num_visible, 5].into(),
            &device,
            FloatDType::F32,
        );
        let num_tiles = tile_bounds.0 * tile_bounds.1;

        if num_visible > 0 && num_tiles > 0 {
            let uniforms = RasterizeBwdUniforms {
                tile_bounds: [tile_bounds.0, tile_bounds.1],
                img_size: [img_size.0, img_size.1],
                background: [background[0], background[1], background[2], 1.0],
            };
            let uniforms_handle = client.create_from_slice(bytemuck::bytes_of(&uniforms));

            // SAFETY: Shader indices are guarded and buffers are sized to all accessed ranges.
            unsafe {
                client.launch_unchecked(
                    Box::new(SourceKernel::new(
                        RasterizeBwdKernel,
                        CubeDim::new_1d(TILE_SIZE),
                    )),
                    CubeCount::Static(num_tiles, 1, 1),
                    KernelArguments::new().with_buffers(vec![
                        compact_gid_from_isect.handle.binding(),
                        tile_offsets.handle.binding(),
                        projected_splats.handle.binding(),
                        out_img.handle.binding(),
                        v_output.handle.binding(),
                        v_splats.handle.clone().binding(),
                        screen_grad_splats.handle.clone().binding(),
                        uniforms_handle.binding(),
                    ]),
                );
            }
        }

        RasterizeBwdPrimitiveOutput {
            v_splats,
            screen_grad_splats,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rasterize_bwd<B: RasterizeBwdBackend>(
    compact_gid_from_isect: Tensor<B, 1, Int>,
    tile_offsets: Tensor<B, 1, Int>,
    projected_splats: Tensor<B, 2>,
    out_img: Tensor<B, 3>,
    v_output: Tensor<B, 3>,
    num_visible: usize,
    img_size: (u32, u32),
    tile_bounds: (u32, u32),
    background: [f32; 3],
    _device: &B::Device,
) -> RasterizeBwdOutput<B> {
    let output = B::rasterize_bwd_primitive(
        compact_gid_from_isect.into_primitive(),
        tile_offsets.into_primitive(),
        projected_splats.into_primitive().tensor(),
        out_img.into_primitive().tensor(),
        v_output.into_primitive().tensor(),
        num_visible,
        img_size,
        tile_bounds,
        background,
    );
    RasterizeBwdOutput {
        v_splats: Tensor::from_primitive(TensorPrimitive::Float(output.v_splats)),
        screen_grad_splats: Tensor::from_primitive(TensorPrimitive::Float(
            output.screen_grad_splats,
        )),
    }
}
