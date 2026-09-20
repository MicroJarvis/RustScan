use burn::prelude::*;
use burn::tensor::{Tensor, TensorPrimitive};
use burn_cubecl::cubecl::{prelude::KernelId, server::KernelArguments, CubeCount};
use burn_cubecl::{kernel::into_contiguous, BoolElement, CubeBackend, FloatElement, IntElement};
use burn_wgpu::{CubeDim, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};

use crate::core::GaussianCamera;
use crate::training::engine::DeviceSplats;

use super::{calc_tile_bounds, compose_shader, ProjectUniforms};

const WORKGROUP_SIZE: u32 = 256;
const SHADER_SRC: &str = include_str!("../shaders/project_visible.wgsl");

struct ProjectVisibleRaw;

impl ProjectVisibleRaw {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(compose_shader("project_visible.wgsl", SHADER_SRC))
    }
}

#[derive(Debug)]
struct ProjectVisibleKernel;

impl KernelSource for ProjectVisibleKernel {
    fn source(&self) -> SourceTemplate {
        ProjectVisibleRaw.source()
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

pub(crate) trait ProjectVisibleBackend: Backend {
    fn project_visible_primitive(
        transforms: Self::FloatTensorPrimitive,
        sh_coeffs: Self::FloatTensorPrimitive,
        raw_opacities: Self::FloatTensorPrimitive,
        global_from_compact_gid: Self::IntTensorPrimitive,
        logical_visible: Self::IntTensorPrimitive,
        uniforms: ProjectUniforms,
        allocation: usize,
        dispatch: CubeCount,
    ) -> Self::FloatTensorPrimitive;
}

impl<F, I, BT> ProjectVisibleBackend for CubeBackend<WgpuRuntime, F, I, BT>
where
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn project_visible_primitive(
        transforms: Self::FloatTensorPrimitive,
        sh_coeffs: Self::FloatTensorPrimitive,
        raw_opacities: Self::FloatTensorPrimitive,
        global_from_compact_gid: Self::IntTensorPrimitive,
        logical_visible: Self::IntTensorPrimitive,
        uniforms: ProjectUniforms,
        allocation: usize,
        dispatch: CubeCount,
    ) -> Self::FloatTensorPrimitive {
        let transforms = into_contiguous(transforms);
        let sh_coeffs = into_contiguous(sh_coeffs);
        let raw_opacities = into_contiguous(raw_opacities);
        let global_from_compact_gid = into_contiguous(global_from_compact_gid);
        let logical_visible = into_contiguous(logical_visible);
        let device = transforms.device.clone();
        let client = transforms.client.clone();

        let projected = Tensor::<Self, 2>::zeros([allocation, 10], &device);

        if allocation > 0 && !dispatch.is_empty() {
            let uniforms_handle = client.create_from_slice(bytemuck::bytes_of(&uniforms));
            client.launch(
                Box::new(SourceKernel::new(
                    ProjectVisibleKernel,
                    CubeDim::new_1d(WORKGROUP_SIZE),
                )),
                dispatch,
                KernelArguments::new().with_buffers(vec![
                    transforms.handle.binding(),
                    sh_coeffs.handle.binding(),
                    raw_opacities.handle.binding(),
                    global_from_compact_gid.handle.binding(),
                    projected.clone().into_primitive().tensor().handle.binding(),
                    uniforms_handle.binding(),
                    logical_visible.handle.binding(),
                ]),
            );
        }

        projected.into_primitive().tensor()
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn project_visible<B: ProjectVisibleBackend>(
    splats: &DeviceSplats<B>,
    active_sh_degree: u32,
    global_from_compact_gid: &Tensor<B, 1, Int>,
    logical_visible: &Tensor<B, 1, Int>,
    allocation: usize,
    dispatch: CubeCount,
    camera: &GaussianCamera,
    img_size: (u32, u32),
    _device: &B::Device,
    cov_blur: f32,
) -> Tensor<B, 2> {
    let tile_bounds = calc_tile_bounds(img_size);
    let uniforms = ProjectUniforms::new_with_cov_blur(
        camera,
        img_size,
        tile_bounds,
        active_sh_degree,
        splats.sh_degree,
        splats.num_splats() as u32,
        allocation as u32,
        cov_blur,
    );

    Tensor::from_primitive(TensorPrimitive::Float(B::project_visible_primitive(
        splats.transforms.val().into_primitive().tensor(),
        splats.sh_coeffs.val().into_primitive().tensor(),
        splats.raw_opacities.val().into_primitive().tensor(),
        global_from_compact_gid.clone().into_primitive(),
        logical_visible.clone().into_primitive(),
        uniforms,
        allocation,
        dispatch,
    )))
}
