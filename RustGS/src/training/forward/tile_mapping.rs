use burn::prelude::*;
use burn::tensor::{Int, Tensor};
use burn_cubecl::cubecl::{prelude::KernelId, server::KernelArguments, CubeCount};
use burn_cubecl::{kernel::into_contiguous, BoolElement, CubeBackend, FloatElement, IntElement};
use burn_wgpu::{CubeDim, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};
use bytemuck::{Pod, Zeroable};

use crate::training::gpu_primitives::prefix_sum::PrefixSumBackend;

use super::compose_shader;

const WORKGROUP_SIZE: u32 = 256;
const MAP_SHADER_SRC: &str = include_str!("../shaders/map_gaussian_to_intersects.wgsl");
const OFFSETS_SHADER_SRC: &str = include_str!("../shaders/get_tile_offsets.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct MGIUniforms {
    tile_bounds: [u32; 2],
    num_visible: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TileOffsetsUniforms {
    num_intersections: u32,
    num_tiles: u32,
    pad: [u32; 2],
}

struct MapGaussiansRaw;

impl MapGaussiansRaw {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(compose_shader(
            "map_gaussian_to_intersects.wgsl",
            MAP_SHADER_SRC,
        ))
    }
}

struct GetTileOffsetsRaw;

impl GetTileOffsetsRaw {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(compose_shader("get_tile_offsets.wgsl", OFFSETS_SHADER_SRC))
    }
}

#[derive(Debug)]
struct MapGaussiansKernel;

impl KernelSource for MapGaussiansKernel {
    fn source(&self) -> SourceTemplate {
        MapGaussiansRaw.source()
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

#[derive(Debug)]
struct GetTileOffsetsKernel;

impl KernelSource for GetTileOffsetsKernel {
    fn source(&self) -> SourceTemplate {
        GetTileOffsetsRaw.source()
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

pub(crate) struct TileMappingOutput<B: Backend> {
    pub tile_id_from_isect: Tensor<B, 1, Int>,
    pub compact_gid_from_isect: Tensor<B, 1, Int>,
}

pub(crate) trait TileMappingBackend: Backend {
    fn map_gaussians_to_intersects_primitive(
        projected: Self::FloatTensorPrimitive,
        cum_tiles_hit: Self::IntTensorPrimitive,
        logical_visible: Self::IntTensorPrimitive,
        num_intersections: usize,
        uniforms: MGIUniforms,
        dispatch: CubeCount,
    ) -> (Self::IntTensorPrimitive, Self::IntTensorPrimitive);

    fn get_tile_offsets_primitive(
        tile_id_from_isect: Self::IntTensorPrimitive,
        logical_intersections: Self::IntTensorPrimitive,
        num_intersections: usize,
        tile_bounds: (u32, u32),
        dispatch: CubeCount,
    ) -> Self::IntTensorPrimitive;
}

impl<F, I, BT> TileMappingBackend for CubeBackend<WgpuRuntime, F, I, BT>
where
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn map_gaussians_to_intersects_primitive(
        projected: Self::FloatTensorPrimitive,
        cum_tiles_hit: Self::IntTensorPrimitive,
        logical_visible: Self::IntTensorPrimitive,
        num_intersections: usize,
        uniforms: MGIUniforms,
        dispatch: CubeCount,
    ) -> (Self::IntTensorPrimitive, Self::IntTensorPrimitive) {
        let projected = into_contiguous(projected);
        let cum_tiles_hit = into_contiguous(cum_tiles_hit);
        let logical_visible = into_contiguous(logical_visible);
        let device = projected.device.clone();
        let client = projected.client.clone();

        let tile_id_from_isect = Tensor::<Self, 1, Int>::zeros([num_intersections], &device);
        let compact_gid_from_isect = Tensor::<Self, 1, Int>::zeros([num_intersections], &device);

        if !dispatch.is_empty() {
            let uniforms_handle = client.create_from_slice(bytemuck::bytes_of(&uniforms));
            client.launch(
                Box::new(SourceKernel::new(
                    MapGaussiansKernel,
                    CubeDim::new_1d(WORKGROUP_SIZE),
                )),
                dispatch,
                KernelArguments::new().with_buffers(vec![
                    projected.handle.binding(),
                    cum_tiles_hit.handle.binding(),
                    tile_id_from_isect.clone().into_primitive().handle.binding(),
                    compact_gid_from_isect
                        .clone()
                        .into_primitive()
                        .handle
                        .binding(),
                    uniforms_handle.binding(),
                    logical_visible.handle.binding(),
                ]),
            );
        }

        (
            tile_id_from_isect.into_primitive(),
            compact_gid_from_isect.into_primitive(),
        )
    }

    fn get_tile_offsets_primitive(
        tile_id_from_isect: Self::IntTensorPrimitive,
        logical_intersections: Self::IntTensorPrimitive,
        num_intersections: usize,
        tile_bounds: (u32, u32),
        dispatch: CubeCount,
    ) -> Self::IntTensorPrimitive {
        let tile_id_from_isect = into_contiguous(tile_id_from_isect);
        let logical_intersections = into_contiguous(logical_intersections);
        let device = tile_id_from_isect.device.clone();
        let client = tile_id_from_isect.client.clone();
        let num_tiles = (tile_bounds.0 * tile_bounds.1) as usize;
        let tile_offsets = Tensor::<Self, 1, Int>::zeros([2 * num_tiles], &device);

        if !dispatch.is_empty() {
            let uniforms = TileOffsetsUniforms {
                num_intersections: num_intersections as u32,
                num_tiles: tile_bounds.0 * tile_bounds.1,
                pad: [0, 0],
            };
            let uniforms_handle = client.create_from_slice(bytemuck::bytes_of(&uniforms));
            client.launch(
                Box::new(SourceKernel::new(
                    GetTileOffsetsKernel,
                    CubeDim::new_1d(WORKGROUP_SIZE),
                )),
                dispatch,
                KernelArguments::new().with_buffers(vec![
                    tile_id_from_isect.handle.binding(),
                    tile_offsets.clone().into_primitive().handle.binding(),
                    uniforms_handle.binding(),
                    logical_intersections.handle.binding(),
                ]),
            );
        }

        tile_offsets.into_primitive()
    }
}

pub(crate) fn tile_mapping<B: TileMappingBackend + PrefixSumBackend>(
    projected_splats: &Tensor<B, 2>,
    intersect_counts: Tensor<B, 1, Int>,
    logical_visible: &Tensor<B, 1, Int>,
    num_intersections: usize,
    _num_tiles: u32,
    tile_bounds: (u32, u32),
    map_dispatch: CubeCount,
    _device: &B::Device,
) -> TileMappingOutput<B> {
    let cum_tiles_hit = Tensor::<B, 1, Int>::from_primitive(
        B::prefix_sum_u32_primitive(intersect_counts.into_primitive()).expect("prefix sum"),
    );
    let (tile_id_from_isect, compact_gid_from_isect) = B::map_gaussians_to_intersects_primitive(
        projected_splats.clone().into_primitive().tensor(),
        cum_tiles_hit.into_primitive(),
        logical_visible.clone().into_primitive(),
        num_intersections,
        MGIUniforms {
            tile_bounds: [tile_bounds.0, tile_bounds.1],
            num_visible: projected_splats.dims()[0] as u32,
            pad: 0,
        },
        map_dispatch,
    );

    TileMappingOutput {
        tile_id_from_isect: Tensor::from_primitive(tile_id_from_isect),
        compact_gid_from_isect: Tensor::from_primitive(compact_gid_from_isect),
    }
}

pub(crate) fn get_tile_offsets<B: TileMappingBackend>(
    tile_id_from_isect: Tensor<B, 1, Int>,
    logical_intersections: &Tensor<B, 1, Int>,
    num_intersections: usize,
    tile_bounds: (u32, u32),
    dispatch: CubeCount,
    _device: &B::Device,
) -> Tensor<B, 1, Int> {
    Tensor::from_primitive(B::get_tile_offsets_primitive(
        tile_id_from_isect.into_primitive(),
        logical_intersections.clone().into_primitive(),
        num_intersections,
        tile_bounds,
        dispatch,
    ))
}
