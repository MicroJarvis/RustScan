use burn::backend::{
    autodiff::{
        checkpoint::{
            base::Checkpointer,
            strategy::{CheckpointStrategy, NoCheckpointing},
        },
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
    },
    Autodiff,
};
use burn::module::Param;
use burn::prelude::*;
use burn::tensor::{backend::AutodiffBackend as BurnAutodiffBackend, Tensor, TensorPrimitive};

use crate::core::GaussianCamera;
use crate::training::backward::{
    project_bwd::{from_inner_splats, project_bwd, ProjectBwdBackend},
    rasterize_bwd::{rasterize_bwd, RasterizeBwdBackend},
};
use crate::training::engine::{DeviceSplats, GsBackendBase, GsDiffBackend};
use crate::training::forward::{
    self, calc_tile_bounds, project_visible, projection, rasterize, sorting, tile_mapping,
};
use crate::training::gpu_primitives::{prefix_sum::PrefixSumBackend, radix_sort::RadixSortBackend};

trait RenderBackend:
    Backend
    + projection::ProjectionBackend
    + sorting::SortingBackend
    + project_visible::ProjectVisibleBackend
    + tile_mapping::TileMappingBackend
    + rasterize::RasterizeBackend
    + RasterizeBwdBackend
    + ProjectBwdBackend
    + PrefixSumBackend
    + RadixSortBackend
{
}

impl<T> RenderBackend for T where
    T: Backend
        + projection::ProjectionBackend
        + sorting::SortingBackend
        + project_visible::ProjectVisibleBackend
        + tile_mapping::TileMappingBackend
        + rasterize::RasterizeBackend
        + RasterizeBwdBackend
        + ProjectBwdBackend
        + PrefixSumBackend
        + RadixSortBackend
{
}

#[derive(Debug, Clone)]
pub(crate) struct RenderCheckpoint<B: Backend> {
    pub transforms: Tensor<B, 2>,
    pub sh_coeffs: Tensor<B, 3>,
    pub raw_opacities: Tensor<B, 1>,
    pub sh_degree: u32,
    pub active_sh_degree: u32,
    pub camera: GaussianCamera,
    pub img_size: (u32, u32),
    pub background: [f32; 3],
    pub cov_blur: f32,
    pub out_img: Tensor<B, 3>,
    pub projected_splats: Tensor<B, 2>,
    pub global_from_compact_gid: Tensor<B, 1, Int>,
    pub compact_gid_from_isect: Tensor<B, 1, Int>,
    pub tile_offsets: Tensor<B, 1, Int>,
    pub num_visible: usize,
}

pub(crate) struct RenderSplatsOutput<B: Backend> {
    pub image: Tensor<B, 3>,
    pub visible: Tensor<B, 1>,
    pub screen_grad_stats: Tensor<B, 2>,
}

#[derive(Debug)]
struct RenderBackward;

impl<B: RenderBackend> Backward<B, 4> for RenderBackward {
    type State = RenderCheckpoint<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 4>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let state = ops.state;
        let [transforms_node, sh_coeffs_node, raw_opacities_node, screen_grad_stats_node] =
            ops.parents;
        let device = state.transforms.device();
        let v_output =
            Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(grads.consume::<B>(&ops.node)));

        let raster_bwd = rasterize_bwd::<B>(
            state.compact_gid_from_isect,
            state.tile_offsets,
            state.projected_splats.clone(),
            state.out_img,
            v_output,
            state.num_visible,
            state.img_size,
            calc_tile_bounds(state.img_size),
            state.background,
            &device,
        );

        let splats = from_inner_splats(
            state.transforms,
            state.sh_coeffs,
            state.raw_opacities,
            state.sh_degree,
        );
        let bwd = project_bwd::<B>(
            &splats,
            state.active_sh_degree,
            state.global_from_compact_gid,
            raster_bwd.v_splats,
            raster_bwd.screen_grad_splats,
            &state.camera,
            state.img_size,
            state.num_visible,
            &device,
            state.cov_blur,
        );

        if let Some(node) = transforms_node {
            grads.register::<B>(node.id, bwd.v_transforms.into_primitive().tensor());
        }
        if let Some(node) = sh_coeffs_node {
            grads.register::<B>(node.id, bwd.v_sh_coeffs.into_primitive().tensor());
        }
        if let Some(node) = raw_opacities_node {
            grads.register::<B>(node.id, bwd.v_raw_opacities.into_primitive().tensor());
        }
        if let Some(node) = screen_grad_stats_node {
            grads.register::<B>(node.id, bwd.screen_grad_stats.into_primitive().tensor());
        }
    }
}

async fn render_splats_impl<B, C>(
    splats: &DeviceSplats<Autodiff<B, C>>,
    active_sh_degree: u32,
    camera: &GaussianCamera,
    img_size: (u32, u32),
    background: [f32; 3],
    cov_blur: f32,
) -> RenderSplatsOutput<Autodiff<B, C>>
where
    B: RenderBackend,
    C: CheckpointStrategy,
{
    type AD<B, C> = Autodiff<B, C>;

    let transforms = splats.transforms.val();
    let sh_coeffs = splats.sh_coeffs.val();
    let raw_opacities = splats.raw_opacities.val();

    let transforms_inner =
        <AD<B, C> as BurnAutodiffBackend>::inner(transforms.clone().into_primitive().tensor());
    let sh_coeffs_inner =
        <AD<B, C> as BurnAutodiffBackend>::inner(sh_coeffs.clone().into_primitive().tensor());
    let raw_opacities_inner =
        <AD<B, C> as BurnAutodiffBackend>::inner(raw_opacities.clone().into_primitive().tensor());

    let inner_splats = DeviceSplats {
        transforms: Param::from_tensor(Tensor::from_primitive(TensorPrimitive::Float(
            transforms_inner.clone(),
        ))),
        sh_coeffs: Param::from_tensor(Tensor::from_primitive(TensorPrimitive::Float(
            sh_coeffs_inner.clone(),
        ))),
        raw_opacities: Param::from_tensor(Tensor::from_primitive(TensorPrimitive::Float(
            raw_opacities_inner.clone(),
        ))),
        sh_degree: splats.sh_degree,
    };

    let device = inner_splats.transforms.val().device();
    let active_sh_degree = active_sh_degree.min(splats.sh_degree);
    let fwd_out = forward::render_forward_with_active_sh::<B>(
        &inner_splats,
        active_sh_degree,
        camera,
        img_size,
        background,
        &device,
        cov_blur,
    )
    .await;
    let visible = Tensor::<AD<B, C>, 1>::from_inner(fwd_out.visible.clone());
    let screen_grad_stats =
        Tensor::<AD<B, C>, 2>::from_inner(Tensor::<B, 2>::zeros([splats.num_splats(), 7], &device))
            .require_grad();

    let image = match RenderBackward
        .prepare::<C>([
            transforms.into_primitive().tensor().node,
            sh_coeffs.into_primitive().tensor().node,
            raw_opacities.into_primitive().tensor().node,
            screen_grad_stats.clone().into_primitive().tensor().node,
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => {
            let state = RenderCheckpoint {
                transforms: Tensor::from_primitive(TensorPrimitive::Float(transforms_inner)),
                sh_coeffs: Tensor::from_primitive(TensorPrimitive::Float(sh_coeffs_inner)),
                raw_opacities: Tensor::from_primitive(TensorPrimitive::Float(raw_opacities_inner)),
                sh_degree: splats.sh_degree,
                active_sh_degree,
                camera: camera.clone(),
                img_size,
                background,
                cov_blur,
                out_img: fwd_out.out_img.clone(),
                projected_splats: fwd_out.projected_splats,
                global_from_compact_gid: fwd_out.global_from_compact_gid,
                compact_gid_from_isect: fwd_out.compact_gid_from_isect,
                tile_offsets: fwd_out.tile_offsets,
                num_visible: fwd_out.num_visible,
            };

            Tensor::from_primitive(TensorPrimitive::Float(
                prep.finish(state, fwd_out.out_img.into_primitive().tensor()),
            ))
        }
        OpsKind::UnTracked(prep) => Tensor::from_primitive(TensorPrimitive::Float(
            prep.finish(fwd_out.out_img.into_primitive().tensor()),
        )),
    };

    RenderSplatsOutput {
        image,
        visible,
        screen_grad_stats,
    }
}

pub(crate) async fn render_splats_with_visibility_active_sh(
    splats: &DeviceSplats<GsDiffBackend>,
    active_sh_degree: u32,
    camera: &GaussianCamera,
    img_size: (u32, u32),
    background: [f32; 3],
    cov_blur: f32,
) -> RenderSplatsOutput<GsDiffBackend> {
    render_splats_impl::<GsBackendBase, NoCheckpointing>(
        splats,
        active_sh_degree,
        camera,
        img_size,
        background,
        cov_blur,
    )
    .await
}
