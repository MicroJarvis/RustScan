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

    /// Training-path scan that reuses `workspace` scratch levels.
    fn prefix_sum_u32_with_workspace(
        workspace: &mut PrefixSumWorkspace,
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

/// Recursive block-sum scratch bytes for an inclusive scan of `len` (no output).
pub(crate) fn prefix_sum_workspace_bytes(len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let mut total = 0usize;
    let mut remaining = len;
    while remaining > 1 {
        let blocks = remaining.div_ceil(WORKGROUP_SIZE as usize);
        // Each recursive level keeps raw block totals and their inclusive prefix.
        total = total.saturating_add(blocks.max(1).saturating_mul(2));
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

/// Per-recursion-level scratch: raw block totals plus their inclusive prefix.
pub(crate) struct PrefixSumLevel {
    block_sums: CubeTensor<WgpuRuntime>,
    block_prefix: CubeTensor<WgpuRuntime>,
    blocks: usize,
}

/// Caller-owned hierarchical scan workspace reused across training steps.
pub(crate) struct PrefixSumWorkspace {
    capacity: usize,
    levels: Vec<PrefixSumLevel>,
    reserved_bytes: usize,
    growth_count: usize,
    step_fresh_allocations: usize,
}

impl Default for PrefixSumWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

impl PrefixSumWorkspace {
    pub(crate) fn new() -> Self {
        Self {
            capacity: 0,
            levels: Vec::new(),
            reserved_bytes: 0,
            growth_count: 0,
            step_fresh_allocations: 0,
        }
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(crate) fn reserved_bytes(&self) -> usize {
        self.reserved_bytes
    }

    pub(crate) fn growth_count(&self) -> usize {
        self.growth_count
    }

    pub(crate) fn step_fresh_allocations(&self) -> usize {
        self.step_fresh_allocations
    }

    /// Reset per-step fresh allocation counter (call at the start of a train step).
    pub(crate) fn begin_step(&mut self) {
        self.step_fresh_allocations = 0;
    }

    /// Inclusive scan into a caller-owned `output` using reserved level scratch.
    pub(crate) fn inclusive_scan_into(
        &mut self,
        input: CubeTensor<WgpuRuntime>,
        len: usize,
        output: CubeTensor<WgpuRuntime>,
    ) -> Result<CubeTensor<WgpuRuntime>, String> {
        if len <= 1 {
            return Ok(input);
        }
        if output.shape()[0] < len {
            return Err(format!(
                "prefix scan output len {} < requested {len}",
                output.shape()[0]
            ));
        }
        self.reserve(len, &input);
        self.scan_into_level(input, len, output, 0)
    }

    /// Grow scratch levels so an inclusive scan of `capacity` elements can run.
    pub(crate) fn reserve(&mut self, capacity: usize, like: &CubeTensor<WgpuRuntime>) {
        let capacity = capacity.max(1);
        let device_ok = self
            .levels
            .first()
            .is_none_or(|level| level.block_sums.device == like.device);
        if device_ok && self.capacity >= capacity {
            return;
        }

        let mut levels = Vec::new();
        let mut fresh = 0usize;
        let mut remaining = capacity;
        loop {
            let blocks = remaining.div_ceil(WORKGROUP_SIZE as usize).max(1);
            levels.push(PrefixSumLevel {
                block_sums: empty_tensor(like, blocks),
                block_prefix: empty_tensor(like, blocks),
                blocks,
            });
            fresh = fresh.saturating_add(2);
            if blocks <= 1 || remaining <= WORKGROUP_SIZE as usize {
                break;
            }
            remaining = blocks;
        }

        self.capacity = capacity;
        self.levels = levels;
        self.reserved_bytes = prefix_sum_workspace_bytes(capacity);
        self.growth_count = self.growth_count.saturating_add(1);
        self.step_fresh_allocations = self.step_fresh_allocations.saturating_add(fresh);
    }

    fn scan_into_level(
        &mut self,
        input: CubeTensor<WgpuRuntime>,
        len: usize,
        output: CubeTensor<WgpuRuntime>,
        level: usize,
    ) -> Result<CubeTensor<WgpuRuntime>, String> {
        if len <= 1 {
            return Ok(input);
        }
        let blocks = len.div_ceil(WORKGROUP_SIZE as usize);
        if level >= self.levels.len() {
            return Err(format!(
                "prefix scan workspace missing level {level} (have {})",
                self.levels.len()
            ));
        }
        if self.levels[level].blocks < blocks {
            return Err(format!(
                "prefix scan level {level} blocks {} < required {blocks}",
                self.levels[level].blocks
            ));
        }

        let client = input.client.clone();
        let params = ScanParams {
            len: len as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_handle = client.create_from_slice(bytemuck::bytes_of(&params));
        let cube_dim = CubeDim::new_1d(WORKGROUP_SIZE);
        let block_sums = self.levels[level].block_sums.clone();
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

        let block_prefix = self.levels[level].block_prefix.clone();
        let scanned_sums =
            self.scan_into_level(block_sums.clone(), blocks, block_prefix, level + 1)?;
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
}

/// Training-path scan that reuses reserved scratch in `workspace`.
/// Output storage is still allocated per call (P1.2 reuses it); levels are reused.
pub(crate) fn inclusive_scan_with_workspace(
    workspace: &mut PrefixSumWorkspace,
    input: CubeTensor<WgpuRuntime>,
) -> Result<CubeTensor<WgpuRuntime>, String> {
    let input = into_contiguous(input);
    if input.dtype() != DType::U32 && input.dtype() != DType::I32 {
        return Err(format!(
            "prefix_sum_u32 expects a 32-bit integer tensor, got {:?}",
            input.dtype()
        ));
    }
    let len = input.shape()[0];
    if len <= 1 {
        return Ok(input);
    }
    let output = empty_tensor(&input, len);
    workspace.inclusive_scan_into(input, len, output)
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
        let len = input.shape()[0];
        if len <= 1 {
            return Ok(input);
        }

        // Public / eval convenience path: independent allocation semantics.
        inclusive_scan_fresh(input)
    }

    fn prefix_sum_u32_with_workspace(
        workspace: &mut PrefixSumWorkspace,
        input: Self::IntTensorPrimitive,
    ) -> Result<Self::IntTensorPrimitive, String> {
        inclusive_scan_with_workspace(workspace, input)
    }
}

/// Public / eval path: every call allocates independent output + recursive scratch.
fn inclusive_scan_fresh(input: CubeTensor<WgpuRuntime>) -> Result<CubeTensor<WgpuRuntime>, String> {
    let len = input.shape()[0];
    if len <= 1 {
        return Ok(input);
    }

    let client = input.client.clone();
    let blocks = len.div_ceil(WORKGROUP_SIZE as usize);
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

    let scanned_sums = inclusive_scan_fresh(block_sums.clone())?;
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
    use super::{
        hillis_steele_dispatch_count, prefix_sum_dispatch_count, prefix_sum_workspace_bytes,
        PrefixSumBackend, PrefixSumWorkspace,
    };

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

        let device = <GsBackendBase as Backend>::Device::default();
        let first_input = Tensor::<GsBackendBase, 1, Int>::from_data(
            TensorData::new(vec![1_i32, 2, 3], [3]),
            &device,
        );
        let second_input = Tensor::<GsBackendBase, 1, Int>::from_data(
            TensorData::new(vec![10_i32, 20, 30], [3]),
            &device,
        );

        let first_scanned = GsBackendBase::prefix_sum_u32_primitive(first_input.into_primitive())
            .expect("first scan");
        let second_scanned =
            GsBackendBase::prefix_sum_u32_primitive(second_input.into_primitive())
                .expect("second scan");

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
            let scanned =
                GsBackendBase::prefix_sum_u32_primitive(input.into_primitive()).expect("scan");
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
            let scanned =
                GsBackendBase::prefix_sum_u32_primitive(input.into_primitive()).expect("scan");
            let actual = Tensor::<GsBackendBase, 1, Int>::from_primitive(scanned)
                .into_data_async()
                .await
                .expect("scan readback")
                .into_vec::<i32>()
                .expect("scan data");
            assert_eq!(actual, expected);
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn workspace_capacity_reuses_without_growth_then_grows() {
        use crate::training::engine::GsBackendBase;
        use burn::prelude::*;
        use burn::tensor::{Int, TensorData};

        let device = <GsBackendBase as Backend>::Device::default();
        let mut ws = PrefixSumWorkspace::new();
        assert_eq!(ws.growth_count(), 0);

        let scan = |ws: &mut PrefixSumWorkspace, values: Vec<i32>| {
            let len = values.len();
            let input =
                Tensor::<GsBackendBase, 1, Int>::from_data(TensorData::new(values, [len]), &device);
            GsBackendBase::prefix_sum_u32_with_workspace(ws, input.into_primitive()).expect("scan")
        };

        ws.begin_step();
        let _ = scan(&mut ws, vec![1, 2, 3, 4]);
        assert_eq!(ws.growth_count(), 1);
        assert!(ws.capacity() >= 4);
        let reserved_after_first = ws.reserved_bytes();
        assert_eq!(reserved_after_first, prefix_sum_workspace_bytes(4));
        let fresh_first = ws.step_fresh_allocations();
        assert!(fresh_first >= 2);

        ws.begin_step();
        let _ = scan(&mut ws, vec![5, 6, 7, 8]);
        assert_eq!(ws.growth_count(), 1, "same capacity must not grow");
        assert_eq!(ws.reserved_bytes(), reserved_after_first);
        assert_eq!(ws.step_fresh_allocations(), 0, "reuse must not fresh-allocate");

        ws.begin_step();
        let long: Vec<i32> = (0..300).map(|i| (i % 3) as i32 + 1).collect();
        let held = scan(&mut ws, long);
        assert_eq!(ws.growth_count(), 2, "larger capacity must grow once");
        assert!(ws.reserved_bytes() > reserved_after_first);
        assert!(ws.step_fresh_allocations() >= 2);

        ws.begin_step();
        let _ = scan(&mut ws, vec![1, 1, 1]);
        let held_vals = Tensor::<GsBackendBase, 1, Int>::from_primitive(held)
            .into_data_async()
            .await
            .expect("held readback")
            .into_vec::<i32>()
            .expect("held data");
        assert_eq!(held_vals.len(), 300);
        assert_eq!(held_vals[0], 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn workspace_scan_matches_fresh_path() {
        use crate::training::engine::GsBackendBase;
        use burn::prelude::*;
        use burn::tensor::{Int, TensorData};

        let device = <GsBackendBase as Backend>::Device::default();
        let values: Vec<i32> = (0..257).map(|i| (i % 7) as i32 + 1).collect();
        let input_fresh = Tensor::<GsBackendBase, 1, Int>::from_data(
            TensorData::new(values.clone(), [values.len()]),
            &device,
        );
        let input_ws = Tensor::<GsBackendBase, 1, Int>::from_data(
            TensorData::new(values.clone(), [values.len()]),
            &device,
        );

        let fresh = GsBackendBase::prefix_sum_u32_primitive(input_fresh.into_primitive())
            .expect("fresh");
        let mut ws = PrefixSumWorkspace::new();
        let reused =
            GsBackendBase::prefix_sum_u32_with_workspace(&mut ws, input_ws.into_primitive())
                .expect("ws");

        let fresh_vals = Tensor::<GsBackendBase, 1, Int>::from_primitive(fresh)
            .into_data_async()
            .await
            .expect("fresh read")
            .into_vec::<i32>()
            .expect("fresh data");
        let ws_vals = Tensor::<GsBackendBase, 1, Int>::from_primitive(reused)
            .into_data_async()
            .await
            .expect("ws read")
            .into_vec::<i32>()
            .expect("ws data");
        assert_eq!(fresh_vals, ws_vals);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn concurrent_owned_workspaces_do_not_interfere() {
        use crate::training::engine::GsBackendBase;
        use burn::prelude::*;
        use burn::tensor::{Int, TensorData};

        let device = <GsBackendBase as Backend>::Device::default();
        let left_values: Vec<i32> = (0..128).map(|i| (i % 5) as i32 + 1).collect();
        let right_values: Vec<i32> = (0..128).map(|i| (i % 7) as i32 + 2).collect();
        let expected_left = {
            let mut acc = 0i32;
            left_values
                .iter()
                .map(|v| {
                    acc = acc.wrapping_add(*v);
                    acc
                })
                .collect::<Vec<_>>()
        };
        let expected_right = {
            let mut acc = 0i32;
            right_values
                .iter()
                .map(|v| {
                    acc = acc.wrapping_add(*v);
                    acc
                })
                .collect::<Vec<_>>()
        };

        let mut left_ws = PrefixSumWorkspace::new();
        let mut right_ws = PrefixSumWorkspace::new();
        let left_input = Tensor::<GsBackendBase, 1, Int>::from_data(
            TensorData::new(left_values, [128]),
            &device,
        );
        let right_input = Tensor::<GsBackendBase, 1, Int>::from_data(
            TensorData::new(right_values, [128]),
            &device,
        );

        // Explicit ownership: each task holds its own &mut workspace. No TLS
        // pointer can be overwritten across awaits.
        let left = GsBackendBase::prefix_sum_u32_with_workspace(
            &mut left_ws,
            left_input.into_primitive(),
        )
        .expect("left scan");
        let right = GsBackendBase::prefix_sum_u32_with_workspace(
            &mut right_ws,
            right_input.into_primitive(),
        )
        .expect("right scan");

        let left_vals = Tensor::<GsBackendBase, 1, Int>::from_primitive(left)
            .into_data_async()
            .await
            .expect("left read")
            .into_vec::<i32>()
            .expect("left data");
        let right_vals = Tensor::<GsBackendBase, 1, Int>::from_primitive(right)
            .into_data_async()
            .await
            .expect("right read")
            .into_vec::<i32>()
            .expect("right data");
        assert_eq!(left_vals, expected_left);
        assert_eq!(right_vals, expected_right);
    }
}
