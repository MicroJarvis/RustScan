use burn::prelude::*;
use burn::tensor::Transaction;

use crate::core::HostSplats;
use crate::training::engine::{device_splats_to_host, DeviceSplats};

use super::{plan_topology_from_host_snapshot, TopologyMutationPlan};

pub(crate) struct TopologySnapshot {
    pub splats: HostSplats,
    pub grad_2d_accum: Vec<f32>,
    pub screen_grad_2d_accum: Vec<f32>,
    pub abs_grad_2d_accum: Vec<f32>,
    pub abs_pixel_grad_2d_accum: Vec<f32>,
    pub pixel_coverage_accum: Vec<f32>,
    pub camera_depth_accum: Vec<f32>,
    pub grad_color_accum: Vec<f32>,
    pub num_observations: Vec<f32>,
    pub visible_observations: Vec<f32>,
    pub actual_visible_observations: Vec<f32>,
    pub splat_ages: Vec<usize>,
    pub invisible_windows: Vec<usize>,
}

pub(crate) async fn snapshot_for_topology<S: Backend, A: Backend>(
    splats: &DeviceSplats<S>,
    grad_2d_accum: &Tensor<A, 1>,
    screen_grad_2d_accum: &Tensor<A, 1>,
    abs_grad_2d_accum: &Tensor<A, 1>,
    abs_pixel_grad_2d_accum: &Tensor<A, 1>,
    pixel_coverage_accum: &Tensor<A, 1>,
    camera_depth_accum: &Tensor<A, 1>,
    grad_color_accum: &Tensor<A, 1>,
    num_observations: &Tensor<A, 1>,
    visible_observations: &Tensor<A, 1>,
    actual_visible_observations: Option<&Tensor<A, 1>>,
) -> TopologySnapshot {
    let splats_host = device_splats_to_host(splats).await;
    let transaction = Transaction::default()
        .register(grad_2d_accum.clone())
        .register(screen_grad_2d_accum.clone())
        .register(abs_grad_2d_accum.clone())
        .register(abs_pixel_grad_2d_accum.clone())
        .register(pixel_coverage_accum.clone())
        .register(camera_depth_accum.clone())
        .register(grad_color_accum.clone())
        .register(num_observations.clone())
        .register(visible_observations.clone());
    let data = if let Some(actual_visible_observations) = actual_visible_observations {
        transaction
            .register(actual_visible_observations.clone())
            .execute_async()
            .await
    } else {
        transaction.execute_async().await
    }
    .expect("topology accumulator readback");

    let grad_2d_accum = data[0]
        .clone()
        .into_vec::<f32>()
        .expect("topology grad_2d readback");
    let screen_grad_2d_accum = data[1]
        .clone()
        .into_vec::<f32>()
        .expect("topology screen_grad_2d readback");
    let abs_grad_2d_accum = data[2]
        .clone()
        .into_vec::<f32>()
        .expect("topology abs_grad_2d readback");
    let abs_pixel_grad_2d_accum = data[3]
        .clone()
        .into_vec::<f32>()
        .expect("topology abs_pixel_grad_2d readback");
    let pixel_coverage_accum = data[4]
        .clone()
        .into_vec::<f32>()
        .expect("topology pixel_coverage readback");
    let camera_depth_accum = data[5]
        .clone()
        .into_vec::<f32>()
        .expect("topology camera_depth readback");
    let grad_color_accum = data[6]
        .clone()
        .into_vec::<f32>()
        .expect("topology grad_color readback");
    let num_observations = data[7]
        .clone()
        .into_vec::<f32>()
        .expect("topology num_observations: expected f32 scalar tensor")
        .into_iter()
        .map(|value| {
            if value.is_finite() {
                value.max(0.0)
            } else {
                0.0
            }
        })
        .collect();
    let visible_observations = data[8]
        .clone()
        .into_vec::<f32>()
        .expect("topology visible_observations: expected f32 scalar tensor")
        .into_iter()
        .map(|value| {
            if value.is_finite() {
                value.max(0.0)
            } else {
                0.0
            }
        })
        .collect();
    let actual_visible_observations = if actual_visible_observations.is_some() {
        data[9]
            .clone()
            .into_vec::<f32>()
            .expect("topology actual_visible_observations: expected f32 scalar tensor")
            .into_iter()
            .map(|value| {
                if value.is_finite() {
                    value.max(0.0)
                } else {
                    0.0
                }
            })
            .collect()
    } else {
        vec![0.0; splats_host.len()]
    };

    TopologySnapshot {
        splat_ages: vec![0; splats_host.len()],
        invisible_windows: vec![0; splats_host.len()],
        splats: splats_host,
        grad_2d_accum,
        screen_grad_2d_accum,
        abs_grad_2d_accum,
        abs_pixel_grad_2d_accum,
        pixel_coverage_accum,
        camera_depth_accum,
        grad_color_accum,
        num_observations,
        visible_observations,
        actual_visible_observations,
    }
}

pub(crate) fn plan_mutations(
    snapshot: &TopologySnapshot,
    config: &crate::training::TrainingConfig,
    iteration: usize,
    frame_count: usize,
) -> TopologyMutationPlan {
    plan_topology_from_host_snapshot(
        config,
        &snapshot.splats,
        &snapshot.grad_2d_accum,
        &snapshot.screen_grad_2d_accum,
        &snapshot.abs_grad_2d_accum,
        &snapshot.abs_pixel_grad_2d_accum,
        &snapshot.pixel_coverage_accum,
        &snapshot.camera_depth_accum,
        &snapshot.grad_color_accum,
        &snapshot.num_observations,
        &snapshot.visible_observations,
        &snapshot.actual_visible_observations,
        &snapshot.splat_ages,
        &snapshot.invisible_windows,
        iteration,
        frame_count,
    )
}
