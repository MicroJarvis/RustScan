use burn::prelude::*;
use burn::tensor::{DType, Int, Tensor};
use burn_cubecl::{kernel::into_contiguous, BoolElement, CubeBackend, FloatElement, IntElement};
use burn_wgpu::WgpuRuntime;

use crate::training::gpu_primitives::radix_sort::RadixSortBackend;

pub(crate) trait SortingBackend: Backend {
    fn reinterpret_f32_as_u32_primitive(
        tensor: Self::FloatTensorPrimitive,
    ) -> Self::IntTensorPrimitive;
}

impl<F, I, BT> SortingBackend for CubeBackend<WgpuRuntime, F, I, BT>
where
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn reinterpret_f32_as_u32_primitive(
        tensor: Self::FloatTensorPrimitive,
    ) -> Self::IntTensorPrimitive {
        let tensor = into_contiguous(tensor);
        burn_wgpu::CubeTensor::new(
            tensor.client.clone(),
            tensor.handle.clone(),
            (*tensor.meta).clone(),
            tensor.device.clone(),
            DType::U32,
        )
    }
}

pub(crate) fn sort_by_depth<B>(
    depths: Tensor<B, 1>,
    global_from_presort_gid: Tensor<B, 1, Int>,
    num_visible: usize,
    _device: &B::Device,
) -> Tensor<B, 1, Int>
where
    B: SortingBackend + RadixSortBackend,
{
    let depths = depths.slice_dim(0, 0..num_visible);
    let global_from_presort_gid = global_from_presort_gid.slice_dim(0, 0..num_visible);

    if num_visible <= 1 {
        return global_from_presort_gid;
    }

    // Canonicalize racey atomic compaction order by splat id, then stable-sort by
    // depth so equal-depth ties stay deterministic across Exact/Bounded launches.
    let depth_u32 = Tensor::<B, 1, Int>::from_primitive(B::reinterpret_f32_as_u32_primitive(
        depths.into_primitive().tensor(),
    ));
    let (sorted_gids, sorted_depth_u32) = B::radix_sort_by_key_u32_primitive(
        global_from_presort_gid.into_primitive(),
        depth_u32.into_primitive(),
    )
    .expect("depth pre-sort by splat id");

    let (_, sorted) =
        B::radix_sort_by_key_u32_primitive(sorted_depth_u32, sorted_gids).expect("depth sort");

    Tensor::from_primitive(sorted)
}

pub(crate) fn sort_by_depth_counted<B>(
    depths: Tensor<B, 1>,
    global_from_presort_gid: Tensor<B, 1, Int>,
    logical_visible: &Tensor<B, 1, Int>,
    dispatch: &Tensor<B, 1, Int>,
    value_fill: i32,
) -> Tensor<B, 1, Int>
where
    B: SortingBackend + RadixSortBackend,
{
    let depth_u32 = Tensor::<B, 1, Int>::from_primitive(B::reinterpret_f32_as_u32_primitive(
        depths.into_primitive().tensor(),
    ));
    let (sorted_gids, sorted_depth_u32) = B::radix_sort_counted_primitive(
        global_from_presort_gid.into_primitive(),
        depth_u32.into_primitive(),
        logical_visible.clone().into_primitive(),
        dispatch.clone().into_primitive(),
        0,
    )
    .expect("counted depth pre-sort by splat id");

    let (_, sorted) = B::radix_sort_counted_primitive(
        sorted_depth_u32,
        sorted_gids,
        logical_visible.clone().into_primitive(),
        dispatch.clone().into_primitive(),
        value_fill,
    )
    .expect("counted depth sort");
    Tensor::from_primitive(sorted)
}
