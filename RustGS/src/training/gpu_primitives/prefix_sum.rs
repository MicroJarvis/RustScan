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

/// Transient output + recursive block-sum buffers for an inclusive scan of `len`.
pub(crate) fn prefix_sum_workspace_bytes(len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let mut total = 0usize;
    let mut remaining = len;
    while remaining > 1 {
        let blocks = remaining.div_ceil(WORKGROUP_SIZE as usize);
        total = total.saturating_add(remaining.saturating_add(blocks.max(1)));
        if blocks <= 1 {
            break;
        }
        remaining = blocks;
    }
    total.saturating_mul(std::mem::size_of::<u32>())
}

pub(crate) fn hillis_steele_dispatch_count(len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    // Copy plus one kernel per doubling offset.
    1 + usize::BITS.saturating_sub(len.saturating_sub(1).leading_zeros()) as usize
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
    // Fresh output and block-sum buffers every call (R02):
    // - returned outputs must outlive later same-length scans
    // - recursive block scans must not overwrite a parent frame's totals
    let output = empty_tensor(&input, len);
    let block_sums = empty_tensor(&input, blocks.max(1));
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
    async fn consecutive_same_length_scans_keep_independent_results() {
        use crate::training::engine::GsBackendBase;
        use burn::prelude::*;
        use burn::tensor::{Int, TensorData};
        use super::PrefixSumBackend;

        let device = <GsBackendBase as Backend>::Device::default();
        let first_values = vec![1_i32, 2, 3];
        let second_values = vec![10_i32, 20, 30];
        let first_input = Tensor::<GsBackendBase, 1, Int>::from_data(
            TensorData::new(first_values.clone(), [3]),
            &device,
        );
        let second_input = Tensor::<GsBackendBase, 1, Int>::from_data(
            TensorData::new(second_values.clone(), [3]),
            &device,
        );

        let first_scanned = GsBackendBase::prefix_sum_u32_primitive(first_input.into_primitive())
            .expect("first scan");
        let second_scanned =
            GsBackendBase::prefix_sum_u32_primitive(second_input.into_primitive())
                .expect("second scan");

        // Read the first result AFTER the second scan has been submitted. The
        // historical bug overwrote the shared scratch so [1,3,6] became [10,30,60].
        let first_actual = Tensor::<GsBackendBase, 1, Int>::from_primitive(first_scanned)
            .into_data_async()
            .await
            .expect("first readback")
            .into_vec::<i32>()
            .expect("first data");
        let second_actual = Tensor::<GsBackendBase, 1, Int>::from_primitive(second_scanned)
            .into_data_async()
            .await
            .expect("second readback")
            .into_vec::<i32>()
            .expect("second data");

        assert_eq!(first_actual, vec![1, 3, 6]);
        assert_eq!(second_actual, vec![10, 30, 60]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn scan_results_survive_length_changes_and_reuse() {
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

        async fn scan_vec(device: &<GsBackendBase as Backend>::Device, values: &[i32]) -> Vec<i32> {
            let input = Tensor::<GsBackendBase, 1, Int>::from_data(
                TensorData::new(values.to_vec(), [values.len()]),
                device,
            );
            let scanned = GsBackendBase::prefix_sum_u32_primitive(input.into_primitive())
                .expect("scan");
            Tensor::<GsBackendBase, 1, Int>::from_primitive(scanned)
                .into_data_async()
                .await
                .expect("readback")
                .into_vec::<i32>()
                .expect("data")
        }

        let device = <GsBackendBase as Backend>::Device::default();
        let short = vec![1_i32, 2, 3];
        let long: Vec<i32> = (0..257).map(|index| (index % 5) as i32 + 1).collect();
        let short_again = vec![4_i32, 5, 6];

        let held_short = {
            let input = Tensor::<GsBackendBase, 1, Int>::from_data(
                TensorData::new(short.clone(), [3]),
                &device,
            );
            GsBackendBase::prefix_sum_u32_primitive(input.into_primitive()).expect("short scan")
        };
        let held_long = {
            let input = Tensor::<GsBackendBase, 1, Int>::from_data(
                TensorData::new(long.clone(), [long.len()]),
                &device,
            );
            GsBackendBase::prefix_sum_u32_primitive(input.into_primitive()).expect("long scan")
        };
        // Different length, then same length again while prior results are live.
        let _ = scan_vec(&device, &[9, 8, 7, 6]).await;
        let third = scan_vec(&device, &short_again).await;

        let short_actual = Tensor::<GsBackendBase, 1, Int>::from_primitive(held_short)
            .into_data_async()
            .await
            .expect("short readback")
            .into_vec::<i32>()
            .expect("short data");
        let long_actual = Tensor::<GsBackendBase, 1, Int>::from_primitive(held_long)
            .into_data_async()
            .await
            .expect("long readback")
            .into_vec::<i32>()
            .expect("long data");

        assert_eq!(short_actual, cpu_scan(&short));
        assert_eq!(long_actual, cpu_scan(&long));
        assert_eq!(third, cpu_scan(&short_again));
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
