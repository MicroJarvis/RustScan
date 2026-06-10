use burn::module::Param;
use burn::prelude::*;
use burn::tensor::{s, Int, Tensor, TensorMetadata, TensorPrimitive};
use burn_cubecl::cubecl::{prelude::KernelId, server::KernelArguments, CubeCount};
use burn_cubecl::{kernel::into_contiguous, BoolElement, CubeBackend, FloatElement, IntElement};
use burn_wgpu::{CubeDim, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};

use crate::core::GaussianCamera;
use crate::sh::sh_coeff_count_for_degree;
use crate::training::engine::DeviceSplats;
use crate::training::forward::{calc_tile_bounds, compose_shader, ProjectUniforms};

const WORKGROUP_SIZE: u32 = 256;
const SHADER_SRC: &str = include_str!("../shaders/project_backwards.wgsl");

struct ProjectBwdRaw;

impl ProjectBwdRaw {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(compose_shader("project_backwards.wgsl", SHADER_SRC))
    }
}

#[derive(Debug)]
struct ProjectBwdKernel;

impl KernelSource for ProjectBwdKernel {
    fn source(&self) -> SourceTemplate {
        ProjectBwdRaw.source()
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

pub(crate) struct ProjectBwdOutput<B: Backend> {
    pub v_transforms: Tensor<B, 2>,
    pub v_sh_coeffs: Tensor<B, 3>,
    pub v_raw_opacities: Tensor<B, 1>,
    pub screen_grad_stats: Tensor<B, 2>,
}

pub(crate) trait ProjectBwdBackend: Backend {
    fn project_bwd_primitive(
        params: Self::FloatTensorPrimitive,
        global_from_compact_gid: Self::IntTensorPrimitive,
        v_splats: Self::FloatTensorPrimitive,
        screen_grad_splats: Self::FloatTensorPrimitive,
        uniforms: ProjectUniforms,
    ) -> ProjectBwdPrimitiveOutput<Self>;
}

pub(crate) struct ProjectBwdPrimitiveOutput<B: Backend> {
    pub(crate) v_params: B::FloatTensorPrimitive,
    pub(crate) v_sh_coeffs: B::FloatTensorPrimitive,
    pub(crate) screen_grad_stats: B::FloatTensorPrimitive,
}

impl<F, I, BT> ProjectBwdBackend for CubeBackend<WgpuRuntime, F, I, BT>
where
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn project_bwd_primitive(
        params: Self::FloatTensorPrimitive,
        global_from_compact_gid: Self::IntTensorPrimitive,
        v_splats: Self::FloatTensorPrimitive,
        screen_grad_splats: Self::FloatTensorPrimitive,
        uniforms: ProjectUniforms,
    ) -> ProjectBwdPrimitiveOutput<Self> {
        let params = into_contiguous(params);
        let global_from_compact_gid = into_contiguous(global_from_compact_gid);
        // NOTE: Keep the sparse rasterize gradients as-is.
        // Re-contiguizing here can rebuild from its original zero initializer and
        // drop out-of-band writes from the rasterize backward kernel.
        let v_splats = v_splats;
        let screen_grad_splats = screen_grad_splats;
        let device = params.device.clone();
        let client = params.client.clone();

        let num_splats = params.shape()[0];
        let num_coeffs = sh_coeff_count_for_degree(uniforms.storage_sh_degree as usize);

        let v_params = Tensor::<Self, 2>::zeros([num_splats, 11], &device);
        let v_sh_coeffs = Tensor::<Self, 3>::zeros([num_splats, num_coeffs, 3], &device);
        let screen_grad_stats = Tensor::<Self, 2>::zeros([num_splats, 7], &device);

        if uniforms.num_visible > 0 {
            let uniforms_handle = client.create_from_slice(bytemuck::bytes_of(&uniforms));
            client.launch(
                Box::new(SourceKernel::new(
                    ProjectBwdKernel,
                    CubeDim::new_1d(WORKGROUP_SIZE),
                )),
                CubeCount::Static(uniforms.num_visible.div_ceil(WORKGROUP_SIZE), 1, 1),
                KernelArguments::new().with_buffers(vec![
                    params.handle.binding(),
                    global_from_compact_gid.handle.binding(),
                    v_splats.handle.binding(),
                    v_params.clone().into_primitive().tensor().handle.binding(),
                    v_sh_coeffs
                        .clone()
                        .into_primitive()
                        .tensor()
                        .handle
                        .binding(),
                    uniforms_handle.binding(),
                    screen_grad_splats.handle.binding(),
                    screen_grad_stats
                        .clone()
                        .into_primitive()
                        .tensor()
                        .handle
                        .binding(),
                ]),
            );
        }

        ProjectBwdPrimitiveOutput {
            v_params: v_params.into_primitive().tensor(),
            v_sh_coeffs: v_sh_coeffs.into_primitive().tensor(),
            screen_grad_stats: screen_grad_stats.into_primitive().tensor(),
        }
    }
}

pub(crate) fn project_bwd<B: ProjectBwdBackend>(
    splats: &DeviceSplats<B>,
    active_sh_degree: u32,
    global_from_compact_gid: Tensor<B, 1, Int>,
    v_splats: Tensor<B, 2>,
    screen_grad_splats: Tensor<B, 2>,
    camera: &GaussianCamera,
    img_size: (u32, u32),
    num_visible: usize,
    _device: &B::Device,
    cov_blur: f32,
) -> ProjectBwdOutput<B> {
    let uniforms = ProjectUniforms::new_with_cov_blur(
        camera,
        img_size,
        calc_tile_bounds(img_size),
        active_sh_degree,
        splats.sh_degree,
        splats.num_splats() as u32,
        num_visible as u32,
        cov_blur,
    );
    let params = Tensor::cat(
        vec![
            splats.transforms.val(),
            splats.raw_opacities.val().unsqueeze_dim(1),
        ],
        1,
    );

    let bwd = B::project_bwd_primitive(
        params.into_primitive().tensor(),
        global_from_compact_gid.into_primitive(),
        v_splats.into_primitive().tensor(),
        screen_grad_splats.into_primitive().tensor(),
        uniforms,
    );
    let v_params = Tensor::from_primitive(TensorPrimitive::Float(bwd.v_params));
    let num_splats = splats.num_splats();

    ProjectBwdOutput {
        v_transforms: v_params.clone().slice(s![.., 0..10]),
        v_sh_coeffs: Tensor::from_primitive(TensorPrimitive::Float(bwd.v_sh_coeffs)),
        v_raw_opacities: v_params.slice(s![.., 10..11]).reshape([num_splats]),
        screen_grad_stats: Tensor::from_primitive(TensorPrimitive::Float(bwd.screen_grad_stats)),
    }
}

pub(crate) fn from_inner_splats<B: Backend>(
    transforms: Tensor<B, 2>,
    sh_coeffs: Tensor<B, 3>,
    raw_opacities: Tensor<B, 1>,
    sh_degree: u32,
) -> DeviceSplats<B> {
    DeviceSplats {
        transforms: Param::from_tensor(transforms),
        sh_coeffs: Param::from_tensor(sh_coeffs),
        raw_opacities: Param::from_tensor(raw_opacities),
        sh_degree,
    }
}
