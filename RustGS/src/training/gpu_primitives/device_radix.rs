use std::cell::RefCell;

use burn::prelude::*;
use burn::tensor::{Int, TensorMetadata};
use burn_cubecl::cubecl::{prelude::KernelId, server::KernelArguments, CubeCount};
use burn_cubecl::{kernel::into_contiguous, BoolElement, CubeBackend, FloatElement, IntElement};
use burn_wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use bytemuck::{Pod, Zeroable};

use super::prefix_sum::PrefixSumBackend;

const WORKGROUP_SIZE: u32 = 256;
const RADIX_BINS: usize = 16;

struct HistogramRaw;
impl HistogramRaw {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("../shaders/radix_histogram.wgsl"))
    }
}

struct ScatterRaw;
impl ScatterRaw {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("../shaders/radix_scatter.wgsl"))
    }
}

#[derive(Debug)]
struct HistogramKernel;
impl KernelSource for HistogramKernel {
    fn source(&self) -> SourceTemplate {
        HistogramRaw.source()
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

#[derive(Debug)]
struct ScatterKernel;
impl KernelSource for ScatterKernel {
    fn source(&self) -> SourceTemplate {
        ScatterRaw.source()
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

struct FillRaw;
impl FillRaw {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("../shaders/fill_u32.wgsl"))
    }
}

#[derive(Debug)]
struct FillKernel;
impl KernelSource for FillKernel {
    fn source(&self) -> SourceTemplate {
        FillRaw.source()
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FillParams {
    len: u32,
    value: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RadixParams {
    shift: u32,
    num_blocks: u32,
    _pad0: u32,
    _pad1: u32,
}

pub(super) fn radix_sort_counted<F, I, BT>(
    keys: CubeTensor<WgpuRuntime>,
    values: CubeTensor<WgpuRuntime>,
    logical_count: CubeTensor<WgpuRuntime>,
    dispatch: CubeCount,
    value_fill: i32,
) -> Result<(CubeTensor<WgpuRuntime>, CubeTensor<WgpuRuntime>), String>
where
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    type B<F, I, BT> = CubeBackend<WgpuRuntime, F, I, BT>;

    let keys = into_contiguous(keys);
    let values = into_contiguous(values);
    let logical_count = into_contiguous(logical_count);
    if keys.shape()[0] != values.shape()[0] {
        return Err("radix_sort_counted expects matching key and value lengths".into());
    }
    let capacity = keys.shape()[0];
    if capacity == 0 {
        return Ok((keys, values));
    }

    let client = keys.client.clone();
    let num_blocks = capacity.div_ceil(WORKGROUP_SIZE as usize);
    let hist_len = RADIX_BINS * num_blocks;
    let cube_dim = CubeDim::new_1d(WORKGROUP_SIZE);
    let scratch = radix_scratch::<F, I, BT>(&keys, capacity, hist_len);
    let hist = Tensor::<B<F, I, BT>, 1, Int>::from_primitive(scratch.hist.clone());

    // Eight passes is even, so the sorted prefix lands back in the caller buffers.
    let mut src_is_input = true;
    for pass in 0..RADIX_PASSES {
        let shift = (pass * 4) as u32;
        fill_u32(&scratch.hist, scratch.hist_len as u32, 0);
        client
            .flush()
            .expect("flush histogram clear before radix histogram");
        let params = RadixParams {
            shift,
            num_blocks: num_blocks as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_handle = client.create_from_slice(bytemuck::bytes_of(&params));
        let (src_keys, src_values, dst_keys, dst_values) = if src_is_input {
            (
                keys.handle.clone(),
                values.handle.clone(),
                scratch.keys.handle.clone(),
                scratch.values.handle.clone(),
            )
        } else {
            (
                scratch.keys.handle.clone(),
                scratch.values.handle.clone(),
                keys.handle.clone(),
                values.handle.clone(),
            )
        };
        if !dispatch.is_empty() {
            client.launch(
                Box::new(SourceKernel::new(HistogramKernel, cube_dim)),
                dispatch.clone(),
                KernelArguments::new().with_buffers(vec![
                    src_keys.clone().binding(),
                    logical_count.handle.clone().binding(),
                    hist.clone().into_primitive().handle.binding(),
                    params_handle.clone().binding(),
                ]),
            );
        }

        let scanned = Tensor::<B<F, I, BT>, 1, Int>::from_primitive(
            <B<F, I, BT> as PrefixSumBackend>::prefix_sum_u32_primitive(
                hist.clone().into_primitive(),
            )
            .expect("radix histogram scan"),
        );
        if pass + 1 == RADIX_PASSES {
            let (fill_keys, fill_values) = if src_is_input {
                (&scratch.keys, &scratch.values)
            } else {
                (&keys, &values)
            };
            fill_u32(fill_keys, capacity as u32, -1i32 as u32);
            fill_u32(fill_values, capacity as u32, value_fill as u32);
            client
                .flush()
                .expect("flush radix tail fill before scatter");
        }
        if !dispatch.is_empty() {
            client.launch(
                Box::new(SourceKernel::new(ScatterKernel, cube_dim)),
                dispatch.clone(),
                KernelArguments::new().with_buffers(vec![
                    src_keys.binding(),
                    src_values.binding(),
                    logical_count.handle.clone().binding(),
                    hist.clone().into_primitive().handle.binding(),
                    scanned.into_primitive().handle.binding(),
                    dst_keys.binding(),
                    dst_values.binding(),
                    params_handle.binding(),
                ]),
            );
        }
        src_is_input = !src_is_input;
    }

    Ok((keys, values))
}

pub(crate) const RADIX_PASSES: usize = 8;

pub(crate) fn radix_sort_dispatch_count(len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let hist_len = RADIX_BINS * len.div_ceil(WORKGROUP_SIZE as usize);
    let scan = super::prefix_sum::prefix_sum_dispatch_count(hist_len);
    // Each pass: clear hist, histogram, scan, scatter. Last pass also fills keys and values.
    RADIX_PASSES * (3 + scan) + 2
}

/// Peak scratch for keys + values + histogram at the given capacity.
pub(crate) fn radix_sort_workspace_bytes(capacity: usize) -> usize {
    if capacity == 0 {
        return 0;
    }
    let hist_len = RADIX_BINS * capacity.div_ceil(WORKGROUP_SIZE as usize);
    (capacity * 2 + hist_len) * std::mem::size_of::<u32>()
}

pub(crate) fn bitonic_dispatch_count(len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let padded = len.next_power_of_two();
    let mut stages = 2usize;
    let mut k = 2usize;
    while k <= padded {
        let mut j = k / 2;
        while j > 0 {
            stages += 1;
            j >>= 1;
        }
        k <<= 1;
    }
    stages
}

#[cfg(test)]
mod tests {
    use super::{bitonic_dispatch_count, radix_sort_dispatch_count};

    #[test]
    fn radix_dispatch_count_stays_below_bitonic() {
        for len in [257, 4_096, 65_536, 1_000_000, 8_388_608] {
            let radix = radix_sort_dispatch_count(len);
            let bitonic = bitonic_dispatch_count(len);
            assert!(
                radix < bitonic,
                "len {len}: radix {radix} was not below bitonic {bitonic}"
            );
        }
    }
}

struct RadixScratch {
    capacity: usize,
    hist_len: usize,
    keys: CubeTensor<WgpuRuntime>,
    values: CubeTensor<WgpuRuntime>,
    hist: CubeTensor<WgpuRuntime>,
}

thread_local! {
    static RADIX_SCRATCH: RefCell<Option<RadixScratch>> = const { RefCell::new(None) };
}

fn radix_scratch<F, I, BT>(
    like: &CubeTensor<WgpuRuntime>,
    capacity: usize,
    hist_len: usize,
) -> RadixScratch
where
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    type B<F, I, BT> = CubeBackend<WgpuRuntime, F, I, BT>;
    RADIX_SCRATCH.with(|slot| {
        let mut slot = slot.borrow_mut();
        let reuse = slot.as_ref().is_some_and(|scratch| {
            scratch.capacity >= capacity
                && scratch.hist_len >= hist_len
                && scratch.keys.device == like.device
        });
        if !reuse {
            let device = like.device.clone();
            *slot = Some(RadixScratch {
                capacity,
                hist_len,
                keys: Tensor::<B<F, I, BT>, 1, Int>::zeros([capacity], &device).into_primitive(),
                values: Tensor::<B<F, I, BT>, 1, Int>::zeros([capacity], &device).into_primitive(),
                hist: Tensor::<B<F, I, BT>, 1, Int>::zeros([hist_len], &device).into_primitive(),
            });
        }
        slot.as_ref().expect("radix scratch").clone_parts()
    })
}

impl RadixScratch {
    fn clone_parts(&self) -> Self {
        Self {
            capacity: self.capacity,
            hist_len: self.hist_len,
            keys: self.keys.clone(),
            values: self.values.clone(),
            hist: self.hist.clone(),
        }
    }
}

fn fill_u32(target: &CubeTensor<WgpuRuntime>, len: u32, value: u32) {
    if len == 0 {
        return;
    }
    let params = FillParams {
        len,
        value,
        _pad0: 0,
        _pad1: 0,
    };
    let params_handle = target
        .client
        .create_from_slice(bytemuck::bytes_of(&params));
    target.client.launch(
        Box::new(SourceKernel::new(
            FillKernel,
            CubeDim::new_1d(WORKGROUP_SIZE),
        )),
        CubeCount::Static(len.div_ceil(WORKGROUP_SIZE), 1, 1),
        KernelArguments::new().with_buffers(vec![
            target.handle.clone().binding(),
            params_handle.binding(),
        ]),
    );
}
