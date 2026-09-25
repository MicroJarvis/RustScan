//! Forward rendering pipeline.

use burn::prelude::*;
use burn::tensor::{Int, Tensor, TensorData};
use bytemuck::{Pod, Zeroable};
use naga_oil::compose::{ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderType};
use wgpu::naga;

use crate::core::GaussianCamera;
use crate::training::engine::DeviceSplats;
use crate::training::gpu_primitives::{
    prefix_sum::{PrefixSumBackend, PrefixSumWorkspace},
    radix_sort::RadixSortBackend,
};

pub mod dispatch;
pub mod parity;
pub mod project_visible;
pub mod projection;
pub mod rasterize;
pub mod sorting;
pub mod tile_mapping;

pub(crate) use dispatch::{hard_intersection_capacity, planned_intersection_capacity, CountPolicy};
pub(crate) use project_visible::project_visible;
pub(crate) use projection::project_forward;
pub(crate) use rasterize::rasterize;
pub(crate) use sorting::sort_by_depth;
pub(crate) use sorting::sort_by_depth_counted;
pub(crate) use tile_mapping::{get_tile_offsets, tile_mapping};

pub(crate) const TILE_WIDTH: u32 = 16;
pub(crate) const TILE_SIZE: u32 = TILE_WIDTH * TILE_WIDTH;
pub(crate) const HELPERS_SRC: &str = include_str!("../shaders/helpers.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ProjectUniforms {
    pub viewmat: [[f32; 4]; 4],
    pub focal: [f32; 2],
    pub img_size: [u32; 2],
    pub tile_bounds: [u32; 2],
    pub pixel_center: [f32; 2],
    pub camera_position: [f32; 4],
    pub sh_degree: u32,
    pub storage_sh_degree: u32,
    pub total_splats: u32,
    pub num_visible: u32,
    pub cov_blur: f32,
    pub _pad: [u32; 3],
}

impl ProjectUniforms {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_cov_blur(
        camera: &GaussianCamera,
        img_size: (u32, u32),
        tile_bounds: (u32, u32),
        sh_degree: u32,
        storage_sh_degree: u32,
        total_splats: u32,
        num_visible: u32,
        cov_blur: f32,
    ) -> Self {
        let sh_degree = sh_degree.min(storage_sh_degree);
        let camera_position = camera.position();
        Self {
            viewmat: crate::core::matrix4_to_column_major_array(camera.view_matrix()),
            focal: [camera.intrinsics.fx, camera.intrinsics.fy],
            img_size: [img_size.0, img_size.1],
            tile_bounds: [tile_bounds.0, tile_bounds.1],
            pixel_center: [camera.intrinsics.cx, camera.intrinsics.cy],
            camera_position: [camera_position.x, camera_position.y, camera_position.z, 0.0],
            sh_degree,
            storage_sh_degree,
            total_splats,
            num_visible,
            cov_blur: cov_blur.max(0.0),
            _pad: [0; 3],
        }
    }
}

pub(crate) fn calc_tile_bounds(img_size: (u32, u32)) -> (u32, u32) {
    (
        img_size.0.div_ceil(TILE_WIDTH),
        img_size.1.div_ceil(TILE_WIDTH),
    )
}

pub(crate) fn compose_shader(file_path: &str, source: &str) -> String {
    let mut composer = Composer::default();
    composer.capabilities = naga::valid::Capabilities::all();
    composer
        .add_composable_module(ComposableModuleDescriptor {
            source: HELPERS_SRC,
            file_path: "helpers.wgsl",
            ..Default::default()
        })
        .expect("helpers shader module");

    let module = composer
        .make_naga_module(NagaModuleDescriptor {
            source,
            file_path,
            shader_type: ShaderType::Wgsl,
            ..Default::default()
        })
        .expect("compose shader");

    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("validate shader");

    naga::back::wgsl::write_string(
        &module,
        &info,
        naga::back::wgsl::WriterFlags::EXPLICIT_TYPES,
    )
    .expect("serialize shader")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ProjectionCounts {
    pub visible: usize,
    pub intersections: usize,
}

fn synced_usize_from_int_data(data: &TensorData, index: usize, label: &str) -> usize {
    if let Ok(values) = data.as_slice::<i32>() {
        values[index].max(0) as usize
    } else if let Ok(values) = data.as_slice::<u32>() {
        values[index] as usize
    } else {
        panic!("{label}: expected i32/u32 tensor");
    }
}

async fn sync_projection_counts_from_gpu<B: Backend>(
    num_visible_buf: Tensor<B, 1, Int>,
    num_intersections_buf: Tensor<B, 1, Int>,
) -> ProjectionCounts {
    let counts = Tensor::<B, 1, Int>::cat(
        vec![
            num_visible_buf.reshape([1]),
            num_intersections_buf.reshape([1]),
        ],
        0,
    );
    let counts_data = counts
        .into_data_async()
        .await
        .expect("projection count readback");

    ProjectionCounts {
        visible: synced_usize_from_int_data(&counts_data, 0, "num_visible"),
        intersections: synced_usize_from_int_data(&counts_data, 1, "num_intersections"),
    }
}

pub(crate) struct RenderOutput<B: Backend> {
    pub out_img: Tensor<B, 3>,
    pub depth: Tensor<B, 2>,
    pub visible: Tensor<B, 1>,
    pub projected_splats: Tensor<B, 2>,
    pub global_from_compact_gid: Tensor<B, 1, Int>,
    pub tile_id_from_isect: Tensor<B, 1, Int>,
    pub compact_gid_from_isect: Tensor<B, 1, Int>,
    pub tile_offsets: Tensor<B, 1, Int>,
    pub logical_visible: Tensor<B, 1, Int>,
    pub logical_intersections: Tensor<B, 1, Int>,
    pub visible_dispatch: Tensor<B, 1, Int>,
    pub intersection_overflow: Tensor<B, 1, Int>,
    pub requested_intersections: Tensor<B, 1, Int>,
    pub intersection_capacity: usize,
    pub num_visible: usize,
}

pub(crate) async fn render_forward<B>(
    splats: &DeviceSplats<B>,
    camera: &GaussianCamera,
    img_size: (u32, u32),
    background: [f32; 3],
    device: &B::Device,
    cov_blur: f32,
) -> RenderOutput<B>
where
    B: projection::ProjectionBackend
        + sorting::SortingBackend
        + project_visible::ProjectVisibleBackend
        + tile_mapping::TileMappingBackend
        + rasterize::RasterizeBackend
        + PrefixSumBackend
        + RadixSortBackend
        + dispatch::WriteDispatchBackend,
{
    render_forward_with_active_sh(
        splats,
        splats.sh_degree,
        camera,
        img_size,
        background,
        device,
        cov_blur,
        CountPolicy::Exact,
        None,
        None,
    )
    .await
}

pub(crate) async fn render_forward_with_active_sh<B>(
    splats: &DeviceSplats<B>,
    active_sh_degree: u32,
    camera: &GaussianCamera,
    img_size: (u32, u32),
    background: [f32; 3],
    device: &B::Device,
    cov_blur: f32,
    count_policy: CountPolicy,
    training_status: Option<(u32, Tensor<B, 1, Int>)>,
    prefix_workspace: Option<&mut PrefixSumWorkspace>,
) -> RenderOutput<B>
where
    B: projection::ProjectionBackend
        + sorting::SortingBackend
        + project_visible::ProjectVisibleBackend
        + tile_mapping::TileMappingBackend
        + rasterize::RasterizeBackend
        + PrefixSumBackend
        + RadixSortBackend
        + dispatch::WriteDispatchBackend,
{
    let active_sh_degree = active_sh_degree.min(splats.sh_degree);
    let proj_out = project_forward(splats, active_sh_degree, camera, img_size, device, cov_blur);
    let projection::ProjectForwardOutput {
        global_from_presort_gid,
        depths,
        intersect_counts,
        num_visible_buf,
        num_intersections_buf,
    } = proj_out;
    let tile_bounds = calc_tile_bounds(img_size);
    let num_tiles = tile_bounds.0 * tile_bounds.1;
    let bounds = match count_policy {
        CountPolicy::Exact => {
            // Eval and viewport still read the exact counts. Training must not.
            let counts =
                sync_projection_counts_from_gpu(num_visible_buf, num_intersections_buf).await;
            ResolvedForwardBounds {
                logical_visible: dispatch::host_count_tensor(counts.visible, device),
                logical_intersections: dispatch::host_count_tensor(counts.intersections, device),
                requested_intersections: dispatch::host_count_tensor(counts.intersections, device),
                visible_dispatch: dispatch::host_dispatch_tensor(counts.visible, device),
                intersection_dispatch: dispatch::host_dispatch_tensor(counts.intersections, device),
                overflow: dispatch::host_count_tensor(0, device),
                capacity: counts.intersections.max(1),
                visible: counts.visible,
                intersections: counts.intersections,
                device_counted: false,
            }
        }
        CountPolicy::Bounded {
            intersection_capacity,
        } => {
            debug_assert!(!CountPolicy::Bounded {
                intersection_capacity
            }
            .allows_count_readback());
            let (iteration, status) = match training_status {
                Some((iteration, status)) => (iteration, status),
                None => {
                    // Evaluation / one-off bounded paths use an isolated dummy status.
                    let dummy = crate::training::engine::DeviceTrainingStatus::<B>::new(device, 0);
                    (0, dummy.buffer().clone())
                }
            };
            let prepared = B::write_forward_dispatch(
                num_visible_buf.into_primitive(),
                num_intersections_buf.into_primitive(),
                intersection_capacity,
                iteration,
                status.into_primitive(),
            );
            ResolvedForwardBounds {
                logical_visible: prepared.logical_visible,
                logical_intersections: prepared.logical_intersections,
                requested_intersections: prepared.requested_intersections,
                visible_dispatch: prepared.visible_dispatch,
                intersection_dispatch: prepared.intersection_dispatch,
                overflow: prepared.overflow,
                capacity: prepared.capacity,
                visible: splats.num_splats(),
                intersections: intersection_capacity,
                device_counted: true,
            }
        }
    };

    if !bounds.device_counted && bounds.visible == 0 {
        let empty_indices = Tensor::<B, 1, Int>::zeros([0], device);
        let projected_splats = Tensor::<B, 2>::zeros([0, 10], device);
        let tile_offsets = Tensor::<B, 1, Int>::zeros([2 * num_tiles as usize], device);
        let raster_out = rasterize(
            &empty_indices,
            &tile_offsets,
            &projected_splats,
            &empty_indices,
            splats.num_splats(),
            img_size,
            tile_bounds,
            background,
            device,
        );

        return RenderOutput {
            out_img: raster_out.out_img,
            depth: raster_out.depth,
            visible: raster_out.visible,
            projected_splats,
            global_from_compact_gid: empty_indices.clone(),
            tile_id_from_isect: empty_indices.clone(),
            compact_gid_from_isect: empty_indices,
            tile_offsets,
            logical_visible: bounds.logical_visible,
            logical_intersections: bounds.logical_intersections,
            visible_dispatch: bounds.visible_dispatch,
            intersection_overflow: bounds.overflow,
            requested_intersections: bounds.requested_intersections.clone(),
            intersection_capacity: bounds.capacity,
            num_visible: 0,
        };
    }

    let global_from_compact_gid = if bounds.device_counted {
        sort_by_depth_counted(
            depths,
            global_from_presort_gid,
            &bounds.logical_visible,
            &bounds.visible_dispatch,
            splats.num_splats() as i32,
        )
    } else {
        sort_by_depth(depths, global_from_presort_gid, bounds.visible, device)
    };

    let compact_intersect_counts = intersect_counts.gather(0, global_from_compact_gid.clone());
    let projected_allocation = if bounds.device_counted {
        splats.num_splats()
    } else {
        bounds.visible
    };
    let projected_splats = project_visible(
        splats,
        active_sh_degree,
        &global_from_compact_gid,
        &bounds.logical_visible,
        projected_allocation,
        bounds.visible_cube_count(),
        camera,
        img_size,
        device,
        cov_blur,
    );

    if !bounds.device_counted && bounds.intersections == 0 {
        let compact_gid_from_isect = Tensor::<B, 1, Int>::zeros([0], device);
        let tile_offsets = Tensor::<B, 1, Int>::zeros([2 * num_tiles as usize], device);
        let raster_out = rasterize(
            &compact_gid_from_isect,
            &tile_offsets,
            &projected_splats,
            &global_from_compact_gid,
            splats.num_splats(),
            img_size,
            tile_bounds,
            background,
            device,
        );

        return RenderOutput {
            out_img: raster_out.out_img,
            depth: raster_out.depth,
            visible: raster_out.visible,
            projected_splats,
            global_from_compact_gid,
            tile_id_from_isect: compact_gid_from_isect.clone(),
            compact_gid_from_isect,
            tile_offsets,
            logical_visible: bounds.logical_visible,
            logical_intersections: bounds.logical_intersections,
            visible_dispatch: bounds.visible_dispatch,
            intersection_overflow: bounds.overflow,
            requested_intersections: bounds.requested_intersections.clone(),
            intersection_capacity: bounds.capacity,
            num_visible: bounds.visible,
        };
    }

    let tile_out = tile_mapping(
        &projected_splats,
        compact_intersect_counts,
        &bounds.logical_visible,
        bounds.intersections,
        num_tiles,
        tile_bounds,
        bounds.visible_cube_count(),
        device,
        prefix_workspace,
    );
    let (tile_id_from_isect, compact_gid_from_isect) = if bounds.device_counted {
        let (keys, values) = B::radix_sort_counted_primitive(
            tile_out.tile_id_from_isect.into_primitive(),
            tile_out.compact_gid_from_isect.into_primitive(),
            bounds.logical_intersections.clone().into_primitive(),
            bounds.intersection_dispatch.clone().into_primitive(),
            0,
        )
        .expect("counted tile sort");
        (Tensor::from_primitive(keys), Tensor::from_primitive(values))
    } else if bounds.intersections > 1 && num_tiles > 1 {
        let (keys, values) = B::radix_sort_by_key_u32_primitive(
            tile_out.tile_id_from_isect.into_primitive(),
            tile_out.compact_gid_from_isect.into_primitive(),
        )
        .expect("tile sort");
        (Tensor::from_primitive(keys), Tensor::from_primitive(values))
    } else {
        (tile_out.tile_id_from_isect, tile_out.compact_gid_from_isect)
    };

    let tile_offsets = get_tile_offsets(
        tile_id_from_isect.clone(),
        &bounds.logical_intersections,
        bounds.intersections,
        tile_bounds,
        bounds.intersection_cube_count(),
        device,
    );

    let raster_out = rasterize(
        &compact_gid_from_isect,
        &tile_offsets,
        &projected_splats,
        &global_from_compact_gid,
        splats.num_splats(),
        img_size,
        tile_bounds,
        background,
        device,
    );

    RenderOutput {
        out_img: raster_out.out_img,
        depth: raster_out.depth,
        visible: raster_out.visible,
        projected_splats,
        global_from_compact_gid,
        tile_id_from_isect,
        compact_gid_from_isect,
        tile_offsets,
        logical_visible: bounds.logical_visible,
        logical_intersections: bounds.logical_intersections,
        visible_dispatch: bounds.visible_dispatch,
        intersection_overflow: bounds.overflow,
        requested_intersections: bounds.requested_intersections,
        intersection_capacity: bounds.capacity,
        num_visible: if bounds.device_counted {
            projected_allocation
        } else {
            bounds.visible
        },
    }
}

struct ResolvedForwardBounds<B: Backend> {
    logical_visible: Tensor<B, 1, Int>,
    logical_intersections: Tensor<B, 1, Int>,
    requested_intersections: Tensor<B, 1, Int>,
    visible_dispatch: Tensor<B, 1, Int>,
    intersection_dispatch: Tensor<B, 1, Int>,
    overflow: Tensor<B, 1, Int>,
    capacity: usize,
    visible: usize,
    intersections: usize,
    device_counted: bool,
}

impl<B: dispatch::WriteDispatchBackend> ResolvedForwardBounds<B> {
    fn visible_cube_count(&self) -> burn_cubecl::cubecl::CubeCount {
        if self.device_counted {
            B::indirect_dispatch(&self.visible_dispatch)
        } else {
            dispatch::static_dispatch(self.visible)
        }
    }

    fn intersection_cube_count(&self) -> burn_cubecl::cubecl::CubeCount {
        if self.device_counted {
            B::indirect_dispatch(&self.intersection_dispatch)
        } else {
            dispatch::static_dispatch(self.intersections)
        }
    }
}

#[cfg(test)]
mod bounded_overflow_gpu {
    use super::{render_forward_with_active_sh, CountPolicy};
    use crate::core::{GaussianCamera, HostSplats};
    use crate::training::engine::{
        host_splats_to_device, DeviceTrainingStatus, GsBackendBase, GsDevice, GsDiffBackend,
        WgpuTrainer,
    };
    use crate::training::gpu_primitives::{
        prefix_sum::PrefixSumBackend, radix_sort::RadixSortBackend,
    };
    use crate::{Intrinsics, TrainingConfig, TrainingError, SE3};
    use burn::prelude::*;
    use burn::tensor::{Int, Tensor, TensorData};

    const IMG_W: u32 = 64;
    const IMG_H: u32 = 16;
    const SENTINEL: u32 = 0xA5A5_A5A5;

    fn camera() -> GaussianCamera {
        GaussianCamera::new(
            Intrinsics::new(
                IMG_W as f32 * 0.5,
                IMG_H as f32 * 0.5,
                IMG_W as f32 * 0.5,
                IMG_H as f32 * 0.5,
                IMG_W,
                IMG_H,
            ),
            SE3::new(&[0.0, 0.0, 0.0, 1.0], &[0.0, 0.0, 0.0]),
        )
    }

    fn splat_at_tile(tile_x: u32, sh: [f32; 3]) -> HostSplats {
        let z = 2.0_f32;
        let fx = IMG_W as f32 * 0.5;
        let fy = IMG_H as f32 * 0.5;
        let cx = IMG_W as f32 * 0.5;
        let cy = IMG_H as f32 * 0.5;
        let px = tile_x as f32 * 16.0 + 8.0;
        let py = 8.0;
        HostSplats::from_components(
            vec![(px - cx) * z / fx, (py - cy) * z / fy, z],
            vec![-2.5, -2.5, -2.5],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![2.0],
            sh.to_vec(),
            0,
        )
        .expect("single tile splat")
    }

    fn two_visible_splats() -> HostSplats {
        let z = 2.0_f32;
        let fx = IMG_W as f32 * 0.5;
        let fy = IMG_H as f32 * 0.5;
        let cx = IMG_W as f32 * 0.5;
        let cy = IMG_H as f32 * 0.5;
        let mut positions = Vec::with_capacity(6);
        let mut sh = Vec::with_capacity(6);
        for (tile_x, color) in [(1_u32, [1.5_f32, -0.8, -0.8]), (2, [-0.8, -0.8, 1.5])] {
            let px = tile_x as f32 * 16.0 + 8.0;
            let py = 8.0;
            positions.extend_from_slice(&[(px - cx) * z / fx, (py - cy) * z / fy, z]);
            sh.extend_from_slice(&color);
        }
        HostSplats::from_components(
            positions,
            vec![-2.5; 6],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![2.0, 2.0],
            sh,
            0,
        )
        .expect("two visible splats")
    }

    async fn scalar_u32(tensor: &Tensor<GsBackendBase, 1, Int>) -> u32 {
        let data = tensor
            .clone()
            .into_data_async()
            .await
            .expect("scalar readback");
        scalar_from_data(&data)
    }

    fn scalar_from_data(data: &TensorData) -> u32 {
        if let Ok(values) = data.as_slice::<i32>() {
            values[0].max(0) as u32
        } else if let Ok(values) = data.as_slice::<u32>() {
            values[0]
        } else {
            panic!("expected i32/u32 scalar");
        }
    }

    async fn read_i32(tensor: &Tensor<GsBackendBase, 1, Int>) -> Vec<i32> {
        tensor
            .clone()
            .into_data_async()
            .await
            .expect("i32 readback")
            .into_vec::<i32>()
            .expect("i32 vector")
    }

    async fn read_f32<const D: usize>(tensor: &Tensor<GsBackendBase, D>) -> Vec<f32> {
        tensor
            .clone()
            .into_data_async()
            .await
            .expect("f32 readback")
            .into_vec::<f32>()
            .expect("f32 vector")
    }

    async fn sync_quiet(tensor: &Tensor<GsBackendBase, 1, Int>, what: &str) {
        let primitive = tensor.clone().into_primitive();
        primitive.client.sync().await.unwrap_or_else(|err| {
            panic!("{what}: wgpu device/internal error during sync: {err:?}")
        });
    }

    async fn render(
        device: &GsDevice,
        host: &HostSplats,
        policy: CountPolicy,
        status: &DeviceTrainingStatus<GsBackendBase>,
    ) -> super::RenderOutput<GsBackendBase> {
        let splats = host_splats_to_device::<GsBackendBase>(host, device);
        render_forward_with_active_sh(
            &splats,
            0,
            &camera(),
            (IMG_W, IMG_H),
            [0.0, 0.0, 0.0],
            device,
            0.0,
            policy,
            Some((1, status.buffer().clone())),
            None,
        )
        .await
    }

    async fn splat_params(
        splats: &crate::training::engine::DeviceSplats<GsDiffBackend>,
    ) -> Vec<f32> {
        let mut values = read_diff_f32(&splats.transforms.val()).await;
        values.extend(read_diff_f32(&splats.sh_coeffs.val()).await);
        values.extend(read_diff_f32(&splats.raw_opacities.val()).await);
        values
    }

    async fn read_diff_f32<const D: usize>(tensor: &Tensor<GsDiffBackend, D>) -> Vec<f32> {
        tensor
            .clone()
            .into_data_async()
            .await
            .expect("param readback")
            .into_vec::<f32>()
            .expect("param f32")
    }

    #[tokio::test(flavor = "current_thread")]
    async fn capacity_one_forward_runs_map_shader_and_returns_capacity_exceeded() {
        let device = GsDevice::default();
        let both = two_visible_splats();
        let first = splat_at_tile(1, [1.5, -0.8, -0.8]);
        let exact_status = DeviceTrainingStatus::<GsBackendBase>::new(&device, 0);
        let only_first = render(&device, &first, CountPolicy::Exact, &exact_status).await;
        let both_exact = render(&device, &both, CountPolicy::Exact, &exact_status).await;
        let only_first_requested = scalar_u32(&only_first.requested_intersections).await;
        let both_requested = scalar_u32(&both_exact.requested_intersections).await;
        assert_eq!(
            only_first_requested, 1,
            "fixture must keep the first gaussian to one intersection"
        );
        assert!(
            both_requested >= 2,
            "need two real intersections before clamping, got {both_requested}"
        );

        let status = DeviceTrainingStatus::<GsBackendBase>::new(&device, 0);
        let bounded = render(
            &device,
            &both,
            CountPolicy::Bounded {
                intersection_capacity: 1,
            },
            &status,
        )
        .await;
        sync_quiet(&bounded.tile_id_from_isect, "bounded forward").await;
        let snap = status.read().await.expect("sticky status readback");
        let err = snap
            .to_error()
            .expect("overflow status must become ForwardCapacityExceeded");
        match err {
            TrainingError::ForwardCapacityExceeded {
                logical_intersections,
                capacity,
                first_iteration,
            } => {
                assert_eq!(capacity, 1);
                assert_eq!(first_iteration, 1);
                assert!(
                    logical_intersections >= 2,
                    "device requested count must come from the shader, got {logical_intersections}"
                );
                assert_eq!(logical_intersections, snap.requested_intersections);
            }
            other => panic!("expected ForwardCapacityExceeded, got {other:?}"),
        }
        assert_eq!(snap.mutation_gate, 0);
        assert_eq!(
            bounded.tile_id_from_isect.dims(),
            [1],
            "bounded workspace must stay at capacity, not the requested count"
        );
        let bounded_tiles = read_i32(&bounded.tile_id_from_isect).await;
        let bounded_gids = read_i32(&bounded.compact_gid_from_isect).await;
        let first_tiles = read_i32(&only_first.tile_id_from_isect).await;
        let first_gids = read_i32(&only_first.compact_gid_from_isect).await;
        assert_eq!(bounded_tiles, first_tiles);
        assert_eq!(bounded_gids, first_gids);
        assert_ne!(
            bounded_tiles[0], 0,
            "map shader must store the in-capacity tile id"
        );

        let bounded_rgb = read_f32(&bounded.out_img).await;
        let first_rgb = read_f32(&only_first.out_img).await;
        let both_rgb = read_f32(&both_exact.out_img).await;
        assert_eq!(bounded_rgb.len(), first_rgb.len());
        let bounded_vs_first = max_abs_diff(&bounded_rgb, &first_rgb);
        let both_vs_first = max_abs_diff(&both_rgb, &first_rgb);
        assert!(
            bounded_vs_first < 1e-4,
            "sort/raster must consume only the in-capacity intersection, diff={bounded_vs_first}"
        );
        assert!(
            both_vs_first > 1e-3,
            "second gaussian must be visible on the exact path so the overflow case is real"
        );

        let mut config = TrainingConfig::default();
        config.optimizer.lr_pos_final = config.optimizer.lr_position;
        config.optimizer.lr_scale_final = config.optimizer.lr_scale;
        config.optimizer.lr_rotation_final = config.optimizer.lr_rotation;
        config.optimizer.lr_opacity_final = config.optimizer.lr_opacity;
        config.optimizer.lr_color_final = config.optimizer.lr_color;
        config.optimizer.lr_color_rest_final = config.optimizer.lr_color_rest;
        let mut trainer = WgpuTrainer::new(config, device.clone(), 2, 1, 1.0);
        trainer.force_intersection_capacity_for_test(1);
        let mut trained = host_splats_to_device::<GsDiffBackend>(&both, &device);
        let before = splat_params(&trained).await;
        let target =
            Tensor::<GsDiffBackend, 3>::zeros([IMG_H as usize, IMG_W as usize, 3], &device);
        let caller_err = trainer
            .train_step(
                &mut trained,
                &camera(),
                target,
                (IMG_W as usize, IMG_H as usize),
                1,
                1,
                false,
                true,
            )
            .await
            .expect_err("caller must observe ForwardCapacityExceeded after the forward");
        match caller_err {
            TrainingError::ForwardCapacityExceeded {
                logical_intersections,
                capacity,
                first_iteration,
            } => {
                assert_eq!(capacity, 1);
                assert_eq!(first_iteration, 1);
                assert!(logical_intersections >= 2);
            }
            other => panic!("expected ForwardCapacityExceeded from train_step, got {other:?}"),
        }
        let after = splat_params(&trained).await;
        assert_eq!(
            before, after,
            "overflow forward must not write persistent gaussian parameters"
        );
    }

    fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn map_shader_capacity_gate_skips_overflow_writes_under_validation() {
        // Platform-default backends (Metal/Vulkan/DX12/etc.); do not hardcode METAL.
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("wgpu adapter for validation-enabled map shader");
        let info = adapter.get_info();
        println!(
            "map-capacity-gate gpu name={} backend={:?} device_type={:?}",
            info.name, info.backend, info.device_type
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("wgpu device");

        let source = super::compose_shader(
            "map_gaussian_to_intersects.wgsl",
            include_str!("../shaders/map_gaussian_to_intersects.wgsl"),
        );
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("map_gaussian_to_intersects"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("map-capacity-gate"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let layout = pipeline.get_bind_group_layout(0);

        let capped = dispatch_map(&device, &queue, &pipeline, &layout, 1);
        assert_eq!(
            capped.tile_ids[0], 1,
            "in-capacity intersection must be stored"
        );
        assert_eq!(capped.compact_gids[0], 0);
        assert_eq!(
            capped.tile_ids[1], SENTINEL,
            "isect_id == capacity must not be written"
        );
        assert_eq!(capped.compact_gids[1], SENTINEL);

        let full = dispatch_map(&device, &queue, &pipeline, &layout, 2);
        assert_eq!(full.tile_ids, vec![1, 2]);
        assert_eq!(full.compact_gids, vec![0, 1]);

        let internal_error = internal.pop().await;
        let validation_error = validation.pop().await;
        assert!(
            internal_error.is_none(),
            "map shader device/internal error: {internal_error:?}"
        );
        assert!(
            validation_error.is_none(),
            "map shader validation error: {validation_error:?}"
        );
    }

    struct MapWords {
        tile_ids: Vec<u32>,
        compact_gids: Vec<u32>,
    }

    fn dispatch_map(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &wgpu::ComputePipeline,
        layout: &wgpu::BindGroupLayout,
        capacity: u32,
    ) -> MapWords {
        let projected: Vec<f32> = vec![
            24.0, 8.0, 100.0, 0.0, 100.0, 1.0, 0.0, 0.0, 1.0, 1.0, 40.0, 8.0, 100.0, 0.0, 100.0,
            0.0, 0.0, 1.0, 1.0, 1.0,
        ];
        let cum_tiles = [1_u32, 2];
        let logical_visible = [2_u32];
        let uniforms = [4_u32, 1, 2, capacity];
        let tile_ids = [SENTINEL, SENTINEL];
        let compact_gids = [SENTINEL, SENTINEL];

        let projected_buf = upload(device, queue, bytemuck::cast_slice(&projected));
        let cum_buf = upload(device, queue, bytemuck::cast_slice(&cum_tiles));
        let tile_buf = upload(device, queue, bytemuck::cast_slice(&tile_ids));
        let gid_buf = upload(device, queue, bytemuck::cast_slice(&compact_gids));
        let uniform_buf = upload(device, queue, bytemuck::cast_slice(&uniforms));
        let visible_buf = upload(device, queue, bytemuck::cast_slice(&logical_visible));
        let tile_read = readback_buffer(device, 8);
        let gid_read = readback_buffer(device, 8);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("map-capacity-bindings"),
            layout,
            entries: &[
                binding(0, &projected_buf),
                binding(1, &cum_buf),
                binding(2, &tile_buf),
                binding(3, &gid_buf),
                binding(4, &uniform_buf),
                binding(5, &visible_buf),
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("map-capacity"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("map-capacity"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&tile_buf, 0, &tile_read, 0, 8);
        encoder.copy_buffer_to_buffer(&gid_buf, 0, &gid_read, 0, 8);
        queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("map shader poll");
        MapWords {
            tile_ids: map_u32(device, &tile_read),
            compact_gids: map_u32(device, &gid_read),
        }
    }

    fn binding(index: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding: index,
            resource: buffer.as_entire_binding(),
        }
    }

    fn upload(device: &wgpu::Device, queue: &wgpu::Queue, contents: &[u8]) -> wgpu::Buffer {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("map-storage"),
            size: contents.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, contents);
        buffer
    }

    fn readback_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("map-readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn map_u32(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u32> {
        let slice = buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).expect("map callback");
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("map poll");
        receiver.recv().expect("map result").expect("map buffer");
        let words = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        buffer.unmap();
        words
    }

    #[tokio::test(flavor = "current_thread")]
    async fn scan_and_radix_boundary_lengths_255_256_257() {
        let device = <GsBackendBase as Backend>::Device::default();
        for len in [255_usize, 256, 257] {
            let input: Vec<i32> = (0..len).map(|index| ((index % 5) + 1) as i32).collect();
            let tensor = Tensor::<GsBackendBase, 1, Int>::from_data(
                TensorData::new(input.clone(), [len]),
                &device,
            );
            let scanned =
                GsBackendBase::prefix_sum_u32_primitive(tensor.into_primitive()).expect("scan");
            let scanned = Tensor::<GsBackendBase, 1, Int>::from_primitive(scanned);
            sync_quiet(&scanned, "scan boundary").await;
            let actual = read_i32(&scanned).await;
            assert_eq!(
                actual,
                cpu_inclusive_scan(&input),
                "scan len {len} including tail lane"
            );

            let keys: Vec<i32> = (0..len)
                .map(|index| ((len - 1 - index) * 3) as i32)
                .collect();
            let values: Vec<i32> = (0..len as i32).collect();
            let key_tensor = Tensor::<GsBackendBase, 1, Int>::from_data(
                TensorData::new(keys.clone(), [len]),
                &device,
            );
            let value_tensor = Tensor::<GsBackendBase, 1, Int>::from_data(
                TensorData::new(values.clone(), [len]),
                &device,
            );
            let (sorted_keys, sorted_values) = GsBackendBase::radix_sort_by_key_u32_primitive(
                key_tensor.into_primitive(),
                value_tensor.into_primitive(),
            )
            .expect("radix");
            let sorted_keys = Tensor::<GsBackendBase, 1, Int>::from_primitive(sorted_keys);
            let sorted_values = Tensor::<GsBackendBase, 1, Int>::from_primitive(sorted_values);
            sync_quiet(&sorted_keys, "radix boundary").await;
            let (expect_keys, expect_values) = cpu_unsigned_sort(&keys, &values);
            assert_eq!(
                read_i32(&sorted_keys).await,
                expect_keys,
                "radix keys len {len}"
            );
            assert_eq!(
                read_i32(&sorted_values).await,
                expect_values,
                "radix values len {len}"
            );
        }
    }

    fn cpu_inclusive_scan(values: &[i32]) -> Vec<i32> {
        let mut acc = 0_i32;
        values
            .iter()
            .map(|value| {
                acc += value;
                acc
            })
            .collect()
    }

    fn cpu_unsigned_sort(keys: &[i32], values: &[i32]) -> (Vec<i32>, Vec<i32>) {
        let mut pairs: Vec<(u32, i32, usize)> = keys
            .iter()
            .zip(values)
            .enumerate()
            .map(|(index, (key, value))| (*key as u32, *value, index))
            .collect();
        pairs.sort_by_key(|pair| (pair.0, pair.2));
        (
            pairs.iter().map(|pair| pair.0 as i32).collect(),
            pairs.iter().map(|pair| pair.1).collect(),
        )
    }
}
