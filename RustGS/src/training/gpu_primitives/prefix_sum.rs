use std::cell::RefCell;

use burn::tensor::{DType, Shape, TensorMetadata};
use burn_cubecl::cubecl::{prelude::KernelId, server::KernelArguments, CubeCount};
use burn_cubecl::{kernel::into_contiguous, BoolElement, CubeBackend, FloatElement, IntElement};
use burn_wgpu::{CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};
use bytemuck::{Pod, Zeroable};

const WORKGROUP_SIZE: u32 = 256;

struct ScanBlockRaw;
impl ScanBlockRaw {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("../shaders/scan_block.wgsl"))
    }
}

struct ScanAddRaw;
impl ScanAddRaw {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("../shaders/scan_add.wgsl"))
    }
}

#[derive(Debug)]
struct ScanBlockKernel;
impl KernelSource for ScanBlockKernel {
    fn source(&self) -> SourceTemplate {
        ScanBlockRaw.source()
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

#[derive(Debug)]
struct ScanAddKernel;
impl KernelSource for ScanAddKernel {
    fn source(&self) -> SourceTemplate {
        ScanAddRaw.source()
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScanParams {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub trait PrefixSumBackend: burn::tensor::backend::Backend {
    fn prefix_sum_u32_primitive(
        input: Self::IntTensorPrimitive,
    ) -> Result<Self::IntTensorPrimitive, String>;
}

/// Inclusive hierarchical scan launches. One block is a shared-memory scan;
/// larger inputs add a recursive block-sum scan and one uniform add.
pub(crate) fn prefix_sum_dispatch_count(len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let blocks = len.div_ceil(WORKGROUP_SIZE as usize);
    if blocks <= 1 {
        1
    } else {
        2 + prefix_sum_dispatch_count(blocks)
    }
}

pub(crate) fn hillis_steele_dispatch_count(len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    // Copy plus one kernel per doubling offset.
    1 + usize::BITS.saturating_sub(len.saturating_sub(1).leading_zeros()) as usize
}

struct ScanScratch {
    output: CubeTensor<WgpuRuntime>,
    block_sums: CubeTensor<WgpuRuntime>,
}

thread_local! {
    static SCAN_SCRATCH: RefCell<Vec<ScanScratch>> = const { RefCell::new(Vec::new()) };
}

fn empty_tensor(like: &CubeTensor<WgpuRuntime>, len: usize) -> CubeTensor<WgpuRuntime> {
    let shape = Shape::new([len.max(1)]);
    CubeTensor::new_contiguous(
        like.client.clone(),
        like.device.clone(),
        shape.clone(),
        like.client
            .empty(shape.num_elements() * core::mem::size_of::<u32>()),
        like.dtype(),
    )
}

fn scratch_output(
    like: &CubeTensor<WgpuRuntime>,
    len: usize,
    blocks: usize,
) -> (CubeTensor<WgpuRuntime>, CubeTensor<WgpuRuntime>) {
    SCAN_SCRATCH.with(|slot| {
        let mut slot = slot.borrow_mut();
        if let Some(scratch) = slot.iter().find(|scratch| {
            scratch.output.shape()[0] == len
                && scratch.block_sums.shape()[0] == blocks.max(1)
                && scratch.output.device == like.device
                && scratch.output.dtype() == like.dtype()
        }) {
            return (scratch.output.clone(), scratch.block_sums.clone());
        }
        if slot.len() >= 8 {
            slot.remove(0);
        }
        let scratch = ScanScratch {
            output: empty_tensor(like, len),
            block_sums: empty_tensor(like, blocks.max(1)),
        };
        let pair = (scratch.output.clone(), scratch.block_sums.clone());
        slot.push(scratch);
        pair
    })
}

impl<F, I, BT> PrefixSumBackend for CubeBackend<WgpuRuntime, F, I, BT>
where
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn prefix_sum_u32_primitive(
        input: Self::IntTensorPrimitive,
    ) -> Result<Self::IntTensorPrimitive, String> {
        let input = into_contiguous(input);
        if input.dtype() != DType::U32 && input.dtype() != DType::I32 {
            return Err(format!(
                "prefix_sum_u32 expects a 32-bit integer tensor, got {:?}",
                input.dtype()
            ));
        }
        inclusive_scan(input)
    }
}

fn inclusive_scan(input: CubeTensor<WgpuRuntime>) -> Result<CubeTensor<WgpuRuntime>, String> {
    let len = input.shape()[0];
    if len <= 1 {
        return Ok(input);
    }

    let client = input.client.clone();
    let blocks = len.div_ceil(WORKGROUP_SIZE as usize);
    let (output, block_sums) = scratch_output(&input, len, blocks);
    let params = ScanParams {
        len: len as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let params_handle = client.create_from_slice(bytemuck::bytes_of(&params));
    let cube_dim = CubeDim::new_1d(WORKGROUP_SIZE);
    client.launch(
        Box::new(SourceKernel::new(ScanBlockKernel, cube_dim)),
        CubeCount::Static(blocks as u32, 1, 1),
        KernelArguments::new().with_buffers(vec![
            input.handle.binding(),
            output.handle.clone().binding(),
            block_sums.handle.clone().binding(),
            params_handle.binding(),
        ]),
    );

    if blocks == 1 {
        return Ok(output);
    }

    let scanned_sums = inclusive_scan(block_sums.clone())?;
    client.launch(
        Box::new(SourceKernel::new(ScanAddKernel, cube_dim)),
        CubeCount::Static(blocks as u32, 1, 1),
        KernelArguments::new().with_buffers(vec![
            output.handle.clone().binding(),
            scanned_sums.handle.binding(),
            block_sums.handle.binding(),
            client
                .create_from_slice(bytemuck::bytes_of(&params))
                .binding(),
        ]),
    );
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::{hillis_steele_dispatch_count, prefix_sum_dispatch_count};

    #[test]
    fn hierarchical_scan_uses_fewer_dispatches_than_hillis_steele() {
        for len in [257, 4_096, 65_536, 1_000_000, 8_388_608] {
            let hierarchical = prefix_sum_dispatch_count(len);
            let hillis = hillis_steele_dispatch_count(len);
            assert!(
                hierarchical < hillis,
                "len {len}: hierarchical {hierarchical} was not below hillis {hillis}"
            );
            assert!(hierarchical <= 7, "len {len} launched {hierarchical}");
        }
        assert_eq!(prefix_sum_dispatch_count(0), 0);
        assert_eq!(prefix_sum_dispatch_count(1), 0);
        assert_eq!(prefix_sum_dispatch_count(256), 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn hierarchical_scan_matches_wrapping_cpu_reference() {
        use crate::training::engine::GsBackendBase;
        use burn::prelude::*;
        use burn::tensor::{Int, TensorData};
        use super::PrefixSumBackend;

        fn cpu_scan(values: &[i32]) -> Vec<i32> {
            let mut acc = 0u32;
            values
                .iter()
                .map(|value| {
                    acc = acc.wrapping_add(*value as u32);
                    acc as i32
                })
                .collect()
        }

        let device = <GsBackendBase as Backend>::Device::default();
        let cases: Vec<Vec<i32>> = vec![
            vec![],
            vec![7],
            vec![0, 0, 0, 0],
            vec![1, 2, 3, 4, 5],
            vec![i32::MAX, 1, 2, 3],
            (0..17).map(|index| (index * 3) as i32).collect(),
            (0..255).map(|index| (index * 5) as i32).collect(),
            (0..256).map(|index| (index * 7) as i32).collect(),
            (0..257).map(|index| (index * 11) as i32).collect(),
            (0..4093).map(|index| (index * 13) as i32).collect(),
            (0..1000)
                .map(|index| if index % 7 == 0 { 0 } else { 3 })
                .collect(),
        ];
        for values in cases {
            if values.is_empty() {
                let input = Tensor::<GsBackendBase, 1, Int>::zeros([0], &device);
                let scanned = GsBackendBase::prefix_sum_u32_primitive(input.into_primitive())
                    .expect("empty scan");
                let actual = Tensor::<GsBackendBase, 1, Int>::from_primitive(scanned)
                    .into_data_async()
                    .await
                    .expect("scan readback")
                    .into_vec::<i32>()
                    .expect("scan data");
                assert!(actual.is_empty());
                continue;
            }
            let expected = cpu_scan(&values);
            let input = Tensor::<GsBackendBase, 1, Int>::from_data(
                TensorData::new(values, [expected.len()]),
                &device,
            );
            let scanned = GsBackendBase::prefix_sum_u32_primitive(input.into_primitive())
                .expect("scan");
            let actual = Tensor::<GsBackendBase, 1, Int>::from_primitive(scanned)
                .into_data_async()
                .await
                .expect("scan readback")
                .into_vec::<i32>()
                .expect("scan data");
            assert_eq!(actual, expected);
        }
    }
}
