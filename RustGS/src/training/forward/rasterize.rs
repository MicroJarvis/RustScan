#![allow(clippy::too_many_arguments)]

use burn::prelude::*;
use burn::tensor::{Int, Tensor, TensorPrimitive};
use burn_cubecl::cubecl::{prelude::KernelId, server::KernelArguments, CubeCount};
use burn_cubecl::{kernel::into_contiguous, BoolElement, CubeBackend, FloatElement, IntElement};
use burn_wgpu::{CubeDim, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};
use bytemuck::{Pod, Zeroable};

use super::{compose_shader, TILE_SIZE};

const SHADER_SRC: &str = include_str!("../shaders/rasterize.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RasterizeUniforms {
    tile_bounds: [u32; 2],
    img_size: [u32; 2],
    background: [f32; 4],
}

struct RasterizeRaw;

impl RasterizeRaw {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(compose_shader("rasterize.wgsl", SHADER_SRC))
    }
}

#[derive(Debug)]
struct RasterizeKernel;

impl KernelSource for RasterizeKernel {
    fn source(&self) -> SourceTemplate {
        RasterizeRaw.source()
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

pub(crate) struct RasterizeOutput<B: Backend> {
    pub out_img: Tensor<B, 3>,
    pub visible: Tensor<B, 1>,
    pub depth: Tensor<B, 2>,
}

pub(crate) trait RasterizeBackend: Backend {
    fn rasterize_primitive(
        compact_gid_from_isect: Self::IntTensorPrimitive,
        tile_offsets: Self::IntTensorPrimitive,
        projected_splats: Self::FloatTensorPrimitive,
        global_from_compact_gid: Self::IntTensorPrimitive,
        total_splats: usize,
        img_size: (u32, u32),
        tile_bounds: (u32, u32),
        background: [f32; 3],
    ) -> (
        Self::FloatTensorPrimitive,
        Self::FloatTensorPrimitive,
        Self::FloatTensorPrimitive,
    );
}

impl<F, I, BT> RasterizeBackend for CubeBackend<WgpuRuntime, F, I, BT>
where
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn rasterize_primitive(
        compact_gid_from_isect: Self::IntTensorPrimitive,
        tile_offsets: Self::IntTensorPrimitive,
        projected_splats: Self::FloatTensorPrimitive,
        global_from_compact_gid: Self::IntTensorPrimitive,
        total_splats: usize,
        img_size: (u32, u32),
        tile_bounds: (u32, u32),
        background: [f32; 3],
    ) -> (
        Self::FloatTensorPrimitive,
        Self::FloatTensorPrimitive,
        Self::FloatTensorPrimitive,
    ) {
        let compact_gid_from_isect = into_contiguous(compact_gid_from_isect);
        let tile_offsets = into_contiguous(tile_offsets);
        let projected_splats = into_contiguous(projected_splats);
        let global_from_compact_gid = into_contiguous(global_from_compact_gid);
        let device = projected_splats.device.clone();
        let client = projected_splats.client.clone();

        let out_img =
            Tensor::<Self, 3>::zeros([img_size.1 as usize, img_size.0 as usize, 4], &device);
        let visible = Tensor::<Self, 1>::zeros([total_splats], &device);
        let depth = Tensor::<Self, 2>::zeros([img_size.1 as usize, img_size.0 as usize], &device);

        let num_tiles = tile_bounds.0 * tile_bounds.1;
        if num_tiles > 0 {
            let uniforms = RasterizeUniforms {
                tile_bounds: [tile_bounds.0, tile_bounds.1],
                img_size: [img_size.0, img_size.1],
                background: [background[0], background[1], background[2], 1.0],
            };
            let uniforms_handle = client.create_from_slice(bytemuck::bytes_of(&uniforms));
            client.launch(
                Box::new(SourceKernel::new(
                    RasterizeKernel,
                    CubeDim::new_1d(TILE_SIZE),
                )),
                CubeCount::Static(num_tiles, 1, 1),
                KernelArguments::new().with_buffers(vec![
                    compact_gid_from_isect.handle.binding(),
                    tile_offsets.handle.binding(),
                    projected_splats.handle.binding(),
                    out_img.clone().into_primitive().tensor().handle.binding(),
                    global_from_compact_gid.handle.binding(),
                    visible.clone().into_primitive().tensor().handle.binding(),
                    depth.clone().into_primitive().tensor().handle.binding(),
                    uniforms_handle.binding(),
                ]),
            );
        }

        (
            out_img.into_primitive().tensor(),
            visible.into_primitive().tensor(),
            depth.into_primitive().tensor(),
        )
    }
}

pub(crate) fn rasterize<B: RasterizeBackend>(
    compact_gid_from_isect: &Tensor<B, 1, Int>,
    tile_offsets: &Tensor<B, 1, Int>,
    projected_splats: &Tensor<B, 2>,
    global_from_compact_gid: &Tensor<B, 1, Int>,
    total_splats: usize,
    img_size: (u32, u32),
    tile_bounds: (u32, u32),
    background: [f32; 3],
    _device: &B::Device,
) -> RasterizeOutput<B> {
    let (out_img, visible, depth) = B::rasterize_primitive(
        compact_gid_from_isect.clone().into_primitive(),
        tile_offsets.clone().into_primitive(),
        projected_splats.clone().into_primitive().tensor(),
        global_from_compact_gid.clone().into_primitive(),
        total_splats,
        img_size,
        tile_bounds,
        background,
    );

    RasterizeOutput {
        out_img: Tensor::from_primitive(TensorPrimitive::Float(out_img)),
        visible: Tensor::from_primitive(TensorPrimitive::Float(visible)),
        depth: Tensor::from_primitive(TensorPrimitive::Float(depth)),
    }
}
