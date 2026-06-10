use burn::prelude::*;
use burn::tensor::{Int, Tensor, TensorMetadata};
use burn_cubecl::cubecl::{prelude::KernelId, server::KernelArguments, CubeCount};
use burn_cubecl::{kernel::into_contiguous, BoolElement, CubeBackend, FloatElement, IntElement};
use burn_wgpu::{CubeDim, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};

use crate::core::GaussianCamera;
use crate::training::engine::DeviceSplats;

use super::{calc_tile_bounds, compose_shader, ProjectUniforms};

const WORKGROUP_SIZE: u32 = 256;
const SHADER_SRC: &str = include_str!("../shaders/project_forward.wgsl");

struct ProjectForwardRaw;

impl ProjectForwardRaw {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(compose_shader("project_forward.wgsl", SHADER_SRC))
    }
}

#[derive(Debug)]
struct ProjectForwardKernel;

impl KernelSource for ProjectForwardKernel {
    fn source(&self) -> SourceTemplate {
        ProjectForwardRaw.source()
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

pub(crate) struct ProjectForwardOutput<B: Backend> {
    pub global_from_presort_gid: Tensor<B, 1, Int>,
    pub depths: Tensor<B, 1>,
    pub intersect_counts: Tensor<B, 1, Int>,
    pub num_visible_buf: Tensor<B, 1, Int>,
    pub num_intersections_buf: Tensor<B, 1, Int>,
}

pub(crate) trait ProjectionBackend: Backend {
    fn project_forward_primitive(
        transforms: Self::FloatTensorPrimitive,
        raw_opacities: Self::FloatTensorPrimitive,
        uniforms: ProjectUniforms,
    ) -> ProjectForwardOutput<Self>;
}

impl<F, I, BT> ProjectionBackend for CubeBackend<WgpuRuntime, F, I, BT>
where
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn project_forward_primitive(
        transforms: Self::FloatTensorPrimitive,
        raw_opacities: Self::FloatTensorPrimitive,
        uniforms: ProjectUniforms,
    ) -> ProjectForwardOutput<Self> {
        let transforms = into_contiguous(transforms);
        let raw_opacities = into_contiguous(raw_opacities);
        let device = transforms.device.clone();
        let client = transforms.client.clone();
        let total_splats = transforms.shape()[0];

        let global_from_presort_gid = Tensor::<Self, 1, Int>::zeros([total_splats], &device);
        let depths = Tensor::<Self, 1>::zeros([total_splats], &device);
        let intersect_counts = Tensor::<Self, 1, Int>::zeros([total_splats], &device);
        let num_visible_buf = Tensor::<Self, 1, Int>::zeros([1], &device);
        let num_intersections_buf = Tensor::<Self, 1, Int>::zeros([1], &device);

        if total_splats > 0 {
            let uniforms_handle = client.create_from_slice(bytemuck::bytes_of(&uniforms));
            client.launch(
                Box::new(SourceKernel::new(
                    ProjectForwardKernel,
                    CubeDim::new_1d(WORKGROUP_SIZE),
                )),
                CubeCount::Static((total_splats as u32).div_ceil(WORKGROUP_SIZE), 1, 1),
                KernelArguments::new().with_buffers(vec![
                    transforms.handle.binding(),
                    raw_opacities.handle.binding(),
                    global_from_presort_gid
                        .clone()
                        .into_primitive()
                        .handle
                        .binding(),
                    depths.clone().into_primitive().tensor().handle.binding(),
                    num_visible_buf.clone().into_primitive().handle.binding(),
                    intersect_counts.clone().into_primitive().handle.binding(),
                    num_intersections_buf
                        .clone()
                        .into_primitive()
                        .handle
                        .binding(),
                    uniforms_handle.binding(),
                ]),
            );
        }

        ProjectForwardOutput {
            global_from_presort_gid,
            depths,
            intersect_counts,
            num_visible_buf,
            num_intersections_buf,
        }
    }
}

pub(crate) fn project_forward<B: ProjectionBackend>(
    splats: &DeviceSplats<B>,
    active_sh_degree: u32,
    camera: &GaussianCamera,
    img_size: (u32, u32),
    _device: &B::Device,
    cov_blur: f32,
) -> ProjectForwardOutput<B> {
    let tile_bounds = calc_tile_bounds(img_size);
    let uniforms = ProjectUniforms::new_with_cov_blur(
        camera,
        img_size,
        tile_bounds,
        active_sh_degree,
        splats.sh_degree,
        splats.num_splats() as u32,
        splats.num_splats() as u32,
        cov_blur,
    );

    B::project_forward_primitive(
        splats.transforms.val().into_primitive().tensor(),
        splats.raw_opacities.val().into_primitive().tensor(),
        uniforms,
    )
}
