use burn::prelude::*;
use burn::tensor::Int;
use burn_cubecl::cubecl::{prelude::KernelId, server::KernelArguments, CubeCount};
use burn_cubecl::{kernel::into_contiguous, BoolElement, CubeBackend, FloatElement, IntElement};
use burn_wgpu::{CubeDim, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};
use bytemuck::{Pod, Zeroable};

pub(crate) const DISPATCH_WORKGROUP: u32 = 256;

/// Hard cap for the training path's unsynchronized intersection workspace.
/// TUM-scale images stay under this; larger scenes report overflow as a hard failure.
pub(crate) const MAX_BOUNDED_INTERSECTIONS: usize = 8_388_608;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CountPolicy {
    /// Read visible/intersection counts before sizing later kernels.
    Exact,
    /// Keep counts on device. Intersection writes are clamped to `intersection_capacity`.
    Bounded { intersection_capacity: usize },
}

impl CountPolicy {
    /// Host-only contract: Exact may schedule count readback; Bounded must not on
    /// the ordinary training iteration (overflow telemetry is sampled separately).
    pub(crate) fn allows_count_readback(&self) -> bool {
        matches!(self, Self::Exact)
    }
}

pub(crate) use crate::training::reporting::metrics::ForwardCapacityTelemetry;

pub(crate) fn hard_intersection_capacity(total_splats: usize, num_tiles: u32) -> usize {
    total_splats
        .saturating_mul(u32::from(num_tiles.max(1)) as usize)
        .max(1)
}

pub(crate) fn planned_intersection_capacity(total_splats: usize, num_tiles: u32) -> usize {
    hard_intersection_capacity(total_splats, num_tiles).min(MAX_BOUNDED_INTERSECTIONS)
}

pub(crate) struct ForwardDispatch<B: Backend> {
    pub logical_visible: Tensor<B, 1, Int>,
    pub logical_intersections: Tensor<B, 1, Int>,
    pub requested_intersections: Tensor<B, 1, Int>,
    pub visible_dispatch: Tensor<B, 1, Int>,
    pub intersection_dispatch: Tensor<B, 1, Int>,
    pub overflow: Tensor<B, 1, Int>,
    pub capacity: usize,
}

pub(crate) fn static_dispatch(count: usize) -> CubeCount {
    CubeCount::Static((count as u32).div_ceil(DISPATCH_WORKGROUP), 1, 1)
}

struct WriteDispatchRaw;

impl WriteDispatchRaw {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("../shaders/write_dispatch.wgsl"))
    }
}

#[derive(Debug)]
struct WriteDispatchKernel;

impl KernelSource for WriteDispatchKernel {
    fn source(&self) -> SourceTemplate {
        WriteDispatchRaw.source()
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WriteDispatchParams {
    intersection_capacity: u32,
    workgroup_size: u32,
    _pad0: u32,
    _pad1: u32,
}

pub(crate) trait WriteDispatchBackend: Backend {
    fn write_forward_dispatch(
        num_visible: Self::IntTensorPrimitive,
        num_intersections: Self::IntTensorPrimitive,
        intersection_capacity: usize,
    ) -> ForwardDispatch<Self>;

    fn indirect_dispatch(dispatch: &Tensor<Self, 1, Int>) -> CubeCount;
}

impl<F, I, BT> WriteDispatchBackend for CubeBackend<WgpuRuntime, F, I, BT>
where
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn write_forward_dispatch(
        num_visible: Self::IntTensorPrimitive,
        num_intersections: Self::IntTensorPrimitive,
        intersection_capacity: usize,
    ) -> ForwardDispatch<Self> {
        let num_visible = into_contiguous(num_visible);
        let num_intersections = into_contiguous(num_intersections);
        let device = num_visible.device.clone();
        let client = num_visible.client.clone();
        let logical_visible = Tensor::<Self, 1, Int>::zeros([1], &device);
        let logical_intersections = Tensor::<Self, 1, Int>::zeros([1], &device);
        let requested_intersections = Tensor::<Self, 1, Int>::zeros([1], &device);
        let visible_dispatch = Tensor::<Self, 1, Int>::zeros([3], &device);
        let intersection_dispatch = Tensor::<Self, 1, Int>::zeros([3], &device);
        let overflow = Tensor::<Self, 1, Int>::zeros([1], &device);
        let params = WriteDispatchParams {
            intersection_capacity: intersection_capacity as u32,
            workgroup_size: DISPATCH_WORKGROUP,
            _pad0: 0,
            _pad1: 0,
        };
        let params_handle = client.create_from_slice(bytemuck::bytes_of(&params));
        client.launch(
            Box::new(SourceKernel::new(
                WriteDispatchKernel,
                CubeDim::new_1d(1),
            )),
            CubeCount::Static(1, 1, 1),
            KernelArguments::new().with_buffers(vec![
                num_visible.handle.binding(),
                num_intersections.handle.binding(),
                logical_visible.clone().into_primitive().handle.binding(),
                logical_intersections
                    .clone()
                    .into_primitive()
                    .handle
                    .binding(),
                visible_dispatch.clone().into_primitive().handle.binding(),
                intersection_dispatch
                    .clone()
                    .into_primitive()
                    .handle
                    .binding(),
                overflow.clone().into_primitive().handle.binding(),
                requested_intersections
                    .clone()
                    .into_primitive()
                    .handle
                    .binding(),
                params_handle.binding(),
            ]),
        );
        // Indirect args must be visible to a later compute pass.
        client
            .flush()
            .expect("flush forward dispatch before indirect launches");

        ForwardDispatch {
            logical_visible,
            logical_intersections,
            requested_intersections,
            visible_dispatch,
            intersection_dispatch,
            overflow,
            capacity: intersection_capacity,
        }
    }

    fn indirect_dispatch(dispatch: &Tensor<Self, 1, Int>) -> CubeCount {
        CubeCount::Dynamic(dispatch.clone().into_primitive().handle.binding())
    }
}

pub(crate) fn host_count_tensor<B: Backend>(count: usize, device: &B::Device) -> Tensor<B, 1, Int> {
    Tensor::<B, 1, Int>::full([1], count as i32, device)
}

pub(crate) fn host_dispatch_tensor<B: Backend>(
    count: usize,
    device: &B::Device,
) -> Tensor<B, 1, Int> {
    let groups = (count as u32).div_ceil(DISPATCH_WORKGROUP) as i32;
    Tensor::<B, 1, Int>::from_ints([groups, 1, 1], device)
}

#[cfg(test)]
mod tests {
    use super::{
        hard_intersection_capacity, planned_intersection_capacity, CountPolicy,
        ForwardCapacityTelemetry, MAX_BOUNDED_INTERSECTIONS,
    };
    use crate::TrainingError;

    #[test]
    fn planned_capacity_uses_the_hard_bound_under_the_budget() {
        assert_eq!(hard_intersection_capacity(4, 3), 12);
        assert_eq!(planned_intersection_capacity(4, 3), 12);
    }

    #[test]
    fn planned_capacity_clamps_large_scenes() {
        let huge = planned_intersection_capacity(1_000_000, 4_096);
        assert_eq!(huge, MAX_BOUNDED_INTERSECTIONS);
    }

    #[test]
    fn zero_tiles_still_allocates_one_slot() {
        assert_eq!(hard_intersection_capacity(0, 0), 1);
        assert_eq!(planned_intersection_capacity(0, 0), 1);
    }

    #[test]
    fn exact_policy_allows_count_readback_and_bounded_does_not() {
        assert!(CountPolicy::Exact.allows_count_readback());
        assert!(!CountPolicy::Bounded {
            intersection_capacity: 1_024
        }
        .allows_count_readback());
    }

    #[test]
    fn overflow_sample_maps_to_forward_capacity_exceeded() {
        let telemetry = ForwardCapacityTelemetry {
            logical_visible: 128,
            logical_intersections: 9_000,
            capacity: 8_000,
            overflowed: true,
        };
        let err = TrainingError::ForwardCapacityExceeded {
            logical_intersections: telemetry.logical_intersections,
            capacity: telemetry.capacity,
        };
        let message = err.to_string();
        assert!(message.contains("9000"));
        assert!(message.contains("8000"));
        assert!(matches!(
            err,
            TrainingError::ForwardCapacityExceeded {
                logical_intersections: 9_000,
                capacity: 8_000
            }
        ));
    }
}
