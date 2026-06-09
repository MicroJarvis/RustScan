use burn::tensor::{DType, Shape, TensorMetadata};
use burn_cubecl::cubecl::{prelude::KernelId, server::KernelArguments, CubeCount};
use burn_cubecl::{kernel::into_contiguous, BoolElement, CubeBackend, FloatElement, IntElement};
use burn_wgpu::{CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};
use bytemuck::{Pod, Zeroable};

const WORKGROUP_SIZE: u32 = 256;
const MODE_INIT: u32 = 0;
const MODE_COMPARE: u32 = 1;
const MODE_COMPACT: u32 = 2;

struct RadixSortRaw;

impl RadixSortRaw {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("../shaders/radix_sort.wgsl"))
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SortParams {
    mode: u32,
    len: u32,
    padded_len: u32,
    stage_k: u32,
    stage_j: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[derive(Debug)]
struct RadixSortKernel;

impl KernelSource for RadixSortKernel {
    fn source(&self) -> SourceTemplate {
        RadixSortRaw
            .source()
            .register("workgroup_size", WORKGROUP_SIZE.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

pub trait RadixSortBackend: burn::tensor::backend::Backend {
    fn radix_sort_by_key_u32_primitive(
        keys: Self::IntTensorPrimitive,
        values: Self::IntTensorPrimitive,
    ) -> Result<(Self::IntTensorPrimitive, Self::IntTensorPrimitive), String>;
}

impl<F, I, BT> RadixSortBackend for CubeBackend<WgpuRuntime, F, I, BT>
where
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn radix_sort_by_key_u32_primitive(
        keys: Self::IntTensorPrimitive,
        values: Self::IntTensorPrimitive,
    ) -> Result<(Self::IntTensorPrimitive, Self::IntTensorPrimitive), String> {
        let keys = into_contiguous(keys);
        let values = into_contiguous(values);

        if keys.dtype() != DType::U32 && keys.dtype() != DType::I32 {
            return Err(format!(
                "radix_sort_by_key_u32 expects 32-bit integer keys, got {:?}",
                keys.dtype()
            ));
        }
        if values.dtype() != DType::U32 && values.dtype() != DType::I32 {
            return Err(format!(
                "radix_sort_by_key_u32 expects 32-bit integer values, got {:?}",
                values.dtype()
            ));
        }
        if keys.shape()[0] != values.shape()[0] {
            return Err(format!(
                "radix_sort_by_key_u32 expects matching lengths, got {} keys and {} values",
                keys.shape()[0],
                values.shape()[0]
            ));
        }
        if keys.device != values.device {
            return Err("radix_sort_by_key_u32 expects keys and values on the same device".into());
        }

        let len = keys.shape()[0];
        if len == 0 {
            return Ok((keys, values));
        }
        if len == 1 {
            return Ok((keys, values));
        }

        let padded_len = len.next_power_of_two();
        let client = keys.client.clone();
        let device = keys.device.clone();
        let keys_dtype = keys.dtype();
        let values_dtype = values.dtype();
        let padded_shape = Shape::new([padded_len]);
        let output_shape = Shape::new([len]);

        let work_keys = CubeTensor::new_contiguous(
            client.clone(),
            device.clone(),
            padded_shape.clone(),
            client.empty(padded_shape.num_elements() * core::mem::size_of::<u32>()),
            keys_dtype,
        );
        let work_values = CubeTensor::new_contiguous(
            client.clone(),
            device.clone(),
            padded_shape.clone(),
            client.empty(padded_shape.num_elements() * core::mem::size_of::<u32>()),
            values_dtype,
        );
        let work_indices = CubeTensor::new_contiguous(
            client.clone(),
            device.clone(),
            padded_shape,
            client.empty(padded_len * core::mem::size_of::<u32>()),
            DType::U32,
        );
        let output_keys = CubeTensor::new_contiguous(
            client.clone(),
            device.clone(),
            output_shape.clone(),
            client.empty(output_shape.num_elements() * core::mem::size_of::<u32>()),
            keys_dtype,
        );
        let output_values = CubeTensor::new_contiguous(
            client.clone(),
            device.clone(),
            output_shape,
            client.empty(len * core::mem::size_of::<u32>()),
            values_dtype,
        );

        let cube_dim = CubeDim::new_1d(WORKGROUP_SIZE);
        let init_count = CubeCount::Static((padded_len as u32).div_ceil(WORKGROUP_SIZE), 1, 1);

        let init_params = SortParams {
            mode: MODE_INIT,
            len: len as u32,
            padded_len: padded_len as u32,
            stage_k: 0,
            stage_j: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let init_handle = client.create_from_slice(bytemuck::bytes_of(&init_params));
        client.launch(
            Box::new(SourceKernel::new(RadixSortKernel, cube_dim)),
            init_count.clone(),
            KernelArguments::new().with_buffers(vec![
                keys.handle.clone().binding(),
                values.handle.clone().binding(),
                work_keys.handle.clone().binding(),
                work_values.handle.clone().binding(),
                work_indices.handle.clone().binding(),
                output_keys.handle.clone().binding(),
                output_values.handle.clone().binding(),
                init_handle.binding(),
            ]),
        );

        let mut k = 2usize;
        while k <= padded_len {
            let mut j = k / 2;
            while j > 0 {
                let params = SortParams {
                    mode: MODE_COMPARE,
                    len: len as u32,
                    padded_len: padded_len as u32,
                    stage_k: k as u32,
                    stage_j: j as u32,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                };
                let params_handle = client.create_from_slice(bytemuck::bytes_of(&params));
                client.launch(
                    Box::new(SourceKernel::new(RadixSortKernel, cube_dim)),
                    init_count.clone(),
                    KernelArguments::new().with_buffers(vec![
                        keys.handle.clone().binding(),
                        values.handle.clone().binding(),
                        work_keys.handle.clone().binding(),
                        work_values.handle.clone().binding(),
                        work_indices.handle.clone().binding(),
                        output_keys.handle.clone().binding(),
                        output_values.handle.clone().binding(),
                        params_handle.binding(),
                    ]),
                );
                j >>= 1;
            }
            k <<= 1;
        }

        let compact_params = SortParams {
            mode: MODE_COMPACT,
            len: len as u32,
            padded_len: padded_len as u32,
            stage_k: 0,
            stage_j: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let compact_handle = client.create_from_slice(bytemuck::bytes_of(&compact_params));
        let compact_count = CubeCount::Static((len as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        client.launch(
            Box::new(SourceKernel::new(RadixSortKernel, cube_dim)),
            compact_count,
            KernelArguments::new().with_buffers(vec![
                keys.handle.clone().binding(),
                values.handle.clone().binding(),
                work_keys.handle.clone().binding(),
                work_values.handle.clone().binding(),
                work_indices.handle.clone().binding(),
                output_keys.handle.clone().binding(),
                output_values.handle.clone().binding(),
                compact_handle.binding(),
            ]),
        );

        Ok((output_keys, output_values))
    }
}
