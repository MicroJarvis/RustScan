//! Exact vs bounded forward pipeline parity (Task 3.1).

use burn::module::Param;
use burn::prelude::*;
use burn::tensor::{s, Int};

use crate::core::{GaussianCamera, HostSplats};
use crate::training::backward;
use crate::training::config::DynamicMaskGradient;
use crate::training::engine::{
    combined_loss_with_kernel, gaussian_kernel_1d, host_splats_to_device, DeviceTrainingStatus,
    GsBackendBase, GsDevice, GsDiffBackend, SsimConfig, WgpuTrainer,
};
use crate::training::forward::{
    render_forward_with_active_sh, CountPolicy, RenderOutput, TILE_WIDTH,
};
use crate::{Intrinsics, TrainingConfig, SE3};

const FLOAT_TOL: f32 = 1e-5;
const COV_BLUR: f32 = 0.3;
const BACKGROUND: [f32; 3] = [0.0, 0.0, 0.0];

#[derive(Debug, Clone)]
struct ParityFixture {
    name: &'static str,
    host_splats: HostSplats,
    camera: GaussianCamera,
    img_size: (u32, u32),
    expect_intersections: Option<usize>,
    expect_visible: Option<usize>,
}

#[derive(Debug, Clone)]
struct HostForwardSnapshot {
    logical_visible: u32,
    logical_intersections: u32,
    requested_intersections: u32,
    overflow: u32,
    tile_ids: Vec<i32>,
    compact_gids: Vec<i32>,
    tile_offsets: Vec<i32>,
    rgb: Vec<f32>,
    depth: Vec<f32>,
    visible: Vec<f32>,
    grad_transforms: Option<Vec<f32>>,
    grad_sh: Option<Vec<f32>>,
    grad_opacity: Option<Vec<f32>>,
}

/// Public entry used by `tests/bounded_forward_parity.rs`.
pub async fn run_bounded_forward_parity_suite() -> Result<(), String> {
    let device = GsDevice::default();
    for fixture in build_fixtures() {
        run_sufficient_capacity_case(&device, &fixture).await?;
    }
    run_full_capacity_case(&device).await?;
    run_capacity_minus_one_no_mutation(&device).await?;
    Ok(())
}

fn build_fixtures() -> Vec<ParityFixture> {
    let camera = parity_camera(64, 64);
    vec![
        ParityFixture {
            name: "zero_visible",
            host_splats: single_splat([0.0, 0.0, -1.0], -2.0, 2.0),
            camera: camera.clone(),
            img_size: (64, 64),
            expect_intersections: Some(0),
            expect_visible: Some(0),
        },
        ParityFixture {
            name: "zero_intersections",
            // Tiny splat far outside the frame: projection rejects before visibility.
            // Still exercises the empty Exact/Bounded path pairing.
            host_splats: single_splat([80.0, 80.0, 2.0], -6.0, 4.0),
            camera: camera.clone(),
            img_size: (64, 64),
            expect_intersections: Some(0),
            expect_visible: Some(0),
        },
        ParityFixture {
            name: "one_intersection",
            host_splats: grid_tile_splats(1, 64, 64),
            camera: camera.clone(),
            img_size: (64, 64),
            expect_intersections: Some(1),
            expect_visible: Some(1),
        },
        ParityFixture {
            name: "intersections_255",
            host_splats: grid_tile_splats(255, 256, 256),
            camera: parity_camera(256, 256),
            img_size: (256, 256),
            expect_intersections: Some(255),
            expect_visible: Some(255),
        },
        ParityFixture {
            name: "intersections_256",
            host_splats: grid_tile_splats(256, 256, 256),
            camera: parity_camera(256, 256),
            img_size: (256, 256),
            expect_intersections: Some(256),
            expect_visible: Some(256),
        },
        ParityFixture {
            name: "intersections_257",
            host_splats: grid_tile_splats(257, 272, 272),
            camera: parity_camera(272, 272),
            img_size: (272, 272),
            expect_intersections: Some(257),
            expect_visible: Some(257),
        },
        ParityFixture {
            name: "partial_tile",
            host_splats: single_splat([-0.9, -0.9, 2.0], -3.5, 1.5),
            camera: camera.clone(),
            img_size: (64, 64),
            expect_intersections: None,
            expect_visible: Some(1),
        },
        ParityFixture {
            name: "duplicate_depth_key",
            host_splats: HostSplats::from_components(
                vec![0.0, 0.0, 2.0, 0.05, 0.0, 2.0],
                vec![-3.5, -3.5, -3.5, -3.5, -3.5, -3.5],
                vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                vec![1.5, 1.5],
                vec![0.4, 0.3, 0.2, 0.5, 0.4, 0.3],
                0,
            )
            .expect("duplicate-depth splats"),
            camera: camera.clone(),
            img_size: (64, 64),
            expect_intersections: None,
            expect_visible: Some(2),
        },
        ParityFixture {
            name: "low_alpha",
            // Opacity just above the 1/255 visibility floor.
            host_splats: single_splat([0.0, 0.0, 2.0], -3.5, -5.4),
            camera: camera.clone(),
            img_size: (64, 64),
            expect_intersections: None,
            expect_visible: None,
        },
        ParityFixture {
            name: "near_plane",
            host_splats: single_splat([0.0, 0.0, 0.011], -4.0, 2.0),
            camera: camera.clone(),
            img_size: (64, 64),
            expect_intersections: None,
            expect_visible: None,
        },
    ]
}

fn parity_camera(width: u32, height: u32) -> GaussianCamera {
    let fx = width as f32 * 0.5;
    let fy = height as f32 * 0.5;
    GaussianCamera::new(
        Intrinsics::new(
            fx,
            fy,
            width as f32 * 0.5,
            height as f32 * 0.5,
            width,
            height,
        ),
        SE3::new(&[0.0, 0.0, 0.0, 1.0], &[0.0, 0.0, 0.0]),
    )
}

fn single_splat(mean: [f32; 3], log_scale: f32, opacity_logit: f32) -> HostSplats {
    HostSplats::from_components(
        mean.to_vec(),
        vec![log_scale, log_scale, log_scale],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![opacity_logit],
        vec![0.5, 0.4, 0.3],
        0,
    )
    .expect("single splat")
}

/// One tiny splat per consecutive tile (row-major); each contributes 1 intersection.
fn grid_tile_splats(count: usize, width: u32, height: u32) -> HostSplats {
    let tiles_x = width.div_ceil(TILE_WIDTH);
    let tiles_y = height.div_ceil(TILE_WIDTH);
    let max_tiles = (tiles_x * tiles_y) as usize;
    assert!(
        count <= max_tiles,
        "need {count} tiles but image only has {max_tiles}"
    );

    let fx = width as f32 * 0.5;
    let fy = height as f32 * 0.5;
    let cx = width as f32 * 0.5;
    let cy = height as f32 * 0.5;
    let z = 2.0_f32;

    let mut positions = Vec::with_capacity(count * 3);
    let mut log_scales = Vec::with_capacity(count * 3);
    let mut rotations = Vec::with_capacity(count * 4);
    let mut opacities = Vec::with_capacity(count);
    let mut sh = Vec::with_capacity(count * 3);

    for idx in 0..count {
        let tx = (idx as u32) % tiles_x;
        let ty = (idx as u32) / tiles_x;
        let px = tx as f32 * TILE_WIDTH as f32 + TILE_WIDTH as f32 * 0.5;
        let py = ty as f32 * TILE_WIDTH as f32 + TILE_WIDTH as f32 * 0.5;
        let x = (px - cx) * z / fx;
        let y = (py - cy) * z / fy;
        positions.extend_from_slice(&[x, y, z]);
        log_scales.extend_from_slice(&[-4.5, -4.5, -4.5]);
        rotations.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        opacities.push(2.0);
        sh.extend_from_slice(&[0.4, 0.3, 0.2]);
    }

    HostSplats::from_components(positions, log_scales, rotations, opacities, sh, 0)
        .expect("grid tile splats")
}

async fn run_sufficient_capacity_case(
    device: &GsDevice,
    fixture: &ParityFixture,
) -> Result<(), String> {
    let exact = capture_forward(device, fixture, CountPolicy::Exact, true).await?;
    if let Some(expected) = fixture.expect_intersections {
        if exact.logical_intersections as usize != expected {
            return Err(format!(
                "{}: exact intersections {} != expected {expected}",
                fixture.name, exact.logical_intersections
            ));
        }
    }
    if let Some(expected) = fixture.expect_visible {
        if exact.logical_visible as usize != expected {
            return Err(format!(
                "{}: exact visible {} != expected {expected}",
                fixture.name, exact.logical_visible
            ));
        }
    }

    let capacity = exact.logical_intersections.max(1) as usize;
    let bounded = capture_forward(
        device,
        fixture,
        CountPolicy::Bounded {
            intersection_capacity: capacity,
        },
        true,
    )
    .await?;
    compare_snapshots(fixture.name, &exact, &bounded)
}

async fn run_full_capacity_case(device: &GsDevice) -> Result<(), String> {
    let fixture = ParityFixture {
        name: "exactly_full_capacity",
        host_splats: grid_tile_splats(16, 64, 64),
        camera: parity_camera(64, 64),
        img_size: (64, 64),
        expect_intersections: Some(16),
        expect_visible: Some(16),
    };
    let exact = capture_forward(device, &fixture, CountPolicy::Exact, true).await?;
    let capacity = exact.logical_intersections as usize;
    if capacity == 0 {
        return Err("exactly_full_capacity: expected non-zero intersections".into());
    }
    let bounded = capture_forward(
        device,
        &fixture,
        CountPolicy::Bounded {
            intersection_capacity: capacity,
        },
        true,
    )
    .await?;
    if bounded.overflow != 0 {
        return Err(format!(
            "exactly_full_capacity: capacity==requested must not overflow (got {})",
            bounded.overflow
        ));
    }
    compare_snapshots(fixture.name, &exact, &bounded)
}

async fn run_capacity_minus_one_no_mutation(device: &GsDevice) -> Result<(), String> {
    let mut config = TrainingConfig::default();
    config.iterations = 4;
    config.litegs.topology.refine_every = 10_000;
    config.litegs.topology.opacity_reset_interval = 10_000;
    // Keep Adam scaling stable across the healthy→overflow pair so Stage-1
    // continuity checks don't confuse LR schedule updates with mutations.
    config.optimizer.lr_pos_final = config.optimizer.lr_position;
    config.optimizer.lr_scale_final = config.optimizer.lr_scale;
    config.optimizer.lr_rotation_final = config.optimizer.lr_rotation;
    config.optimizer.lr_opacity_final = config.optimizer.lr_opacity;
    config.optimizer.lr_color_final = config.optimizer.lr_color;
    config.optimizer.lr_color_rest_final = config.optimizer.lr_color_rest;

    let host = grid_tile_splats(4, 32, 32);
    let mut splats = host_splats_to_device::<GsDiffBackend>(&host, device);
    let mut trainer = WgpuTrainer::new(config, device.clone(), host.len(), 1, 1.0);
    let camera = parity_camera(32, 32);
    let target = Tensor::<GsDiffBackend, 3>::full([32, 32, 3], 0.4, device);

    let probe = capture_forward(
        device,
        &ParityFixture {
            name: "capacity_minus_one_probe",
            host_splats: host,
            camera: camera.clone(),
            img_size: (32, 32),
            expect_intersections: None,
            expect_visible: None,
        },
        CountPolicy::Exact,
        false,
    )
    .await?;
    let requested = probe.logical_intersections as usize;
    if requested <= 1 {
        return Err(format!(
            "capacity_minus_one: need requested>1 for capacity-1 case, got {requested}"
        ));
    }

    trainer
        .assert_overflow_capacity_minus_one_mutates_nothing_for_test(
            &mut splats,
            &camera,
            target,
            (32, 32),
            requested - 1,
        )
        .await
        .map_err(|e| format!("capacity_minus_one (requested={requested}): {e}"))
}

async fn capture_forward(
    device: &GsDevice,
    fixture: &ParityFixture,
    policy: CountPolicy,
    with_grads: bool,
) -> Result<HostForwardSnapshot, String> {
    let base_splats = host_splats_to_device::<GsBackendBase>(&fixture.host_splats, device);
    // Exact ignores status; Bounded needs a live buffer for sticky overflow writes.
    let status = DeviceTrainingStatus::<GsBackendBase>::new(device, 0);
    let fwd = render_forward_with_active_sh(
        &base_splats,
        0,
        &fixture.camera,
        fixture.img_size,
        BACKGROUND,
        device,
        COV_BLUR,
        policy,
        Some((1, status.buffer().clone())),
        None,
    )
    .await;

    let mut snap = host_from_render(&fwd).await?;
    if with_grads && snap.logical_intersections > 0 && snap.overflow == 0 {
        let grads = capture_grads(device, fixture, policy).await?;
        snap.grad_transforms = Some(grads.0);
        snap.grad_sh = Some(grads.1);
        snap.grad_opacity = Some(grads.2);
    }
    Ok(snap)
}

async fn capture_grads(
    device: &GsDevice,
    fixture: &ParityFixture,
    policy: CountPolicy,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let mut splats = host_splats_to_device::<GsDiffBackend>(&fixture.host_splats, device);
    splats.transforms = Param::from_tensor(splats.transforms.val().require_grad());
    splats.sh_coeffs = Param::from_tensor(splats.sh_coeffs.val().require_grad());
    splats.raw_opacities = Param::from_tensor(splats.raw_opacities.val().require_grad());

    let status = DeviceTrainingStatus::<GsBackendBase>::new(device, 0);
    let rendered = backward::render_splats_with_count_policy(
        &splats,
        0,
        &fixture.camera,
        fixture.img_size,
        BACKGROUND,
        COV_BLUR,
        policy,
        Some((1, status.buffer().clone())),
    )
    .await;

    let target = Tensor::<GsDiffBackend, 3>::full(
        [
            fixture.img_size.1 as usize,
            fixture.img_size.0 as usize,
            3,
        ],
        0.25,
        device,
    );
    let pred = rendered.image.slice(s![.., .., 0..3]);
    let ssim_config = SsimConfig::default();
    let ssim_kernel = gaussian_kernel_1d::<GsDiffBackend>(&ssim_config, device);
    let loss = combined_loss_with_kernel(
        pred,
        target,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        DynamicMaskGradient::StopGradient,
        &ssim_config,
        ssim_kernel,
    );
    let mut grads = loss.backward();
    let gt = splats
        .transforms
        .grad_remove(&mut grads)
        .ok_or_else(|| format!("{}: missing transforms grad", fixture.name))?;
    let gs = splats
        .sh_coeffs
        .grad_remove(&mut grads)
        .ok_or_else(|| format!("{}: missing sh grad", fixture.name))?;
    let go = splats
        .raw_opacities
        .grad_remove(&mut grads)
        .ok_or_else(|| format!("{}: missing opacity grad", fixture.name))?;
    Ok((
        read_f32(&gt).await?,
        read_f32(&gs).await?,
        read_f32(&go).await?,
    ))
}

async fn host_from_render<B: Backend>(out: &RenderOutput<B>) -> Result<HostForwardSnapshot, String> {
    let logical_visible = read_u32_scalar(&out.logical_visible).await?;
    let logical_intersections = read_u32_scalar(&out.logical_intersections).await?;
    let requested = read_u32_scalar(&out.requested_intersections).await?;
    let overflow = read_u32_scalar(&out.intersection_overflow).await?;
    let n = logical_intersections as usize;

    Ok(HostForwardSnapshot {
        logical_visible,
        logical_intersections,
        requested_intersections: requested,
        overflow,
        tile_ids: truncate_prefix(read_i32(&out.tile_id_from_isect).await?, n),
        compact_gids: truncate_prefix(read_i32(&out.compact_gid_from_isect).await?, n),
        tile_offsets: read_i32(&out.tile_offsets).await?,
        rgb: read_f32(&out.out_img).await?,
        depth: read_f32(&out.depth).await?,
        visible: read_f32(&out.visible).await?,
        grad_transforms: None,
        grad_sh: None,
        grad_opacity: None,
    })
}

fn truncate_prefix(mut values: Vec<i32>, n: usize) -> Vec<i32> {
    if values.len() > n {
        values.truncate(n);
    }
    values
}

fn compare_snapshots(
    name: &str,
    exact: &HostForwardSnapshot,
    bounded: &HostForwardSnapshot,
) -> Result<(), String> {
    if exact.logical_visible != bounded.logical_visible {
        return Err(format!(
            "{name}: logical_visible exact={} bounded={}",
            exact.logical_visible, bounded.logical_visible
        ));
    }
    if exact.logical_intersections != bounded.logical_intersections {
        return Err(format!(
            "{name}: logical_intersections exact={} bounded={}",
            exact.logical_intersections, bounded.logical_intersections
        ));
    }
    if exact.requested_intersections != bounded.requested_intersections {
        return Err(format!(
            "{name}: requested_intersections exact={} bounded={}",
            exact.requested_intersections, bounded.requested_intersections
        ));
    }
    if bounded.overflow != 0 {
        return Err(format!(
            "{name}: bounded overflow unexpectedly set ({})",
            bounded.overflow
        ));
    }
    if exact.tile_ids != bounded.tile_ids {
        return Err(format!("{name}: tile sort keys diverge"));
    }
    if exact.compact_gids != bounded.compact_gids {
        return Err(format!("{name}: tile sort values diverge"));
    }
    if exact.tile_offsets != bounded.tile_offsets {
        return Err(format!("{name}: tile offsets diverge"));
    }
    assert_f32_close(name, "rgb", &exact.rgb, &bounded.rgb, FLOAT_TOL)?;
    assert_f32_close(name, "depth", &exact.depth, &bounded.depth, FLOAT_TOL)?;
    assert_f32_close(name, "visible", &exact.visible, &bounded.visible, FLOAT_TOL)?;

    if exact.logical_intersections > 0 {
        let (et, es, eo) = match (
            &exact.grad_transforms,
            &exact.grad_sh,
            &exact.grad_opacity,
        ) {
            (Some(t), Some(s), Some(o)) => (t, s, o),
            _ => return Err(format!("{name}: exact gradients missing")),
        };
        let (bt, bs, bo) = match (
            &bounded.grad_transforms,
            &bounded.grad_sh,
            &bounded.grad_opacity,
        ) {
            (Some(t), Some(s), Some(o)) => (t, s, o),
            _ => return Err(format!("{name}: bounded gradients missing")),
        };
        assert_f32_close(name, "grad_transforms", et, bt, FLOAT_TOL)?;
        assert_f32_close(name, "grad_sh", es, bs, FLOAT_TOL)?;
        assert_f32_close(name, "grad_opacity", eo, bo, FLOAT_TOL)?;
    }
    Ok(())
}

fn assert_f32_close(
    case: &str,
    label: &str,
    a: &[f32],
    b: &[f32],
    tol: f32,
) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!(
            "{case}/{label}: length {} vs {}",
            a.len(),
            b.len()
        ));
    }
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if !(x.is_finite() && y.is_finite()) {
            if x.is_nan() && y.is_nan() {
                continue;
            }
            return Err(format!("{case}/{label}[{i}]: non-finite {x} vs {y}"));
        }
        let diff = (x - y).abs();
        if diff > tol {
            return Err(format!(
                "{case}/{label}[{i}]: |{x} - {y}| = {diff} > {tol}"
            ));
        }
    }
    Ok(())
}

async fn read_f32<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> Result<Vec<f32>, String> {
    tensor
        .clone()
        .into_data_async()
        .await
        .map_err(|e| format!("f32 readback: {e}"))?
        .into_vec::<f32>()
        .map_err(|e| format!("f32 cast: {e:?}"))
}

async fn read_i32<B: Backend>(tensor: &Tensor<B, 1, Int>) -> Result<Vec<i32>, String> {
    let data = tensor
        .clone()
        .into_data_async()
        .await
        .map_err(|e| format!("i32 readback: {e}"))?;
    if let Ok(values) = data.clone().into_vec::<i32>() {
        Ok(values)
    } else if let Ok(values) = data.into_vec::<u32>() {
        Ok(values.into_iter().map(|v| v as i32).collect())
    } else {
        Err("expected i32/u32 int tensor".into())
    }
}

async fn read_u32_scalar<B: Backend>(tensor: &Tensor<B, 1, Int>) -> Result<u32, String> {
    let values = read_i32(tensor).await?;
    Ok(values.first().copied().unwrap_or(0).max(0) as u32)
}
