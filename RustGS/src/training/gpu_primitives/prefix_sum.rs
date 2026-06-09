use burn::tensor::{DType, Shape, TensorMetadata};
use burn_cubecl::cubecl::{prelude::KernelId, server::KernelArguments, CubeCount};
use burn_cubecl::{kernel::into_contiguous, BoolElement, CubeBackend, FloatElement, IntElement};
use burn_wgpu::{CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};
use bytemuck::{Pod, Zeroable};

const WORKGROUP_SIZE: u32 = 256;

struct PrefixSumRaw;

impl PrefixSumRaw {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("../shaders/prefix_sum.wgsl"))
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PrefixSumParams {
    len: u32,
    offset: u32,
    _pad0: u32,
    _pad1: u32,
}

#[derive(Debug)]
struct PrefixSumKernel;

impl KernelSource for PrefixSumKernel {
    fn source(&self) -> SourceTemplate {
        PrefixSumRaw
            .source()
            .register("workgroup_size", WORKGROUP_SIZE.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

pub trait PrefixSumBackend: burn::tensor::backend::Backend {
    fn prefix_sum_u32_primitive(
        input: Self::IntTensorPrimitive,
    ) -> Result<Self::IntTensorPrimitive, String>;
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

        let client = input.client.clone();
        let device = input.device.clone();
        let dtype = input.dtype();
        let shape = Shape::new([len]);

        let lhs_buffer = client.empty(shape.num_elements() * core::mem::size_of::<u32>());
        let rhs_buffer = client.empty(shape.num_elements() * core::mem::size_of::<u32>());

        let mut src = CubeTensor::new_contiguous(
            client.clone(),
            device.clone(),
            shape.clone(),
            lhs_buffer,
            dtype,
        );
        let mut dst =
            CubeTensor::new_contiguous(client.clone(), device.clone(), shape, rhs_buffer, dtype);

        let init_params = PrefixSumParams {
            len: len as u32,
            offset: 0,
            _pad0: 0,
            _pad1: 0,
        };
        let init_handle = client.create_from_slice(bytemuck::bytes_of(&init_params));
        let init_kernel = PrefixSumKernel;
        let init_count = CubeCount::Static((len as u32).div_ceil(WORKGROUP_SIZE), 1, 1);

        client.launch(
            Box::new(SourceKernel::new(
                init_kernel,
                CubeDim::new_1d(WORKGROUP_SIZE),
            )),
            init_count.clone(),
            KernelArguments::new().with_buffers(vec![
                input.handle.clone().binding(),
                src.handle.clone().binding(),
                init_handle.binding(),
            ]),
        );

        let mut offset = 1usize;
        while offset < len {
            let params = PrefixSumParams {
                len: len as u32,
                offset: offset as u32,
                _pad0: 0,
                _pad1: 0,
            };
            let params_handle = client.create_from_slice(bytemuck::bytes_of(&params));
            client.launch(
                Box::new(SourceKernel::new(
                    PrefixSumKernel,
                    CubeDim::new_1d(WORKGROUP_SIZE),
                )),
                init_count.clone(),
                KernelArguments::new().with_buffers(vec![
                    src.handle.clone().binding(),
                    dst.handle.clone().binding(),
                    params_handle.binding(),
                ]),
            );
            core::mem::swap(&mut src, &mut dst);
            offset <<= 1;
        }

        Ok(src)
    }
}
