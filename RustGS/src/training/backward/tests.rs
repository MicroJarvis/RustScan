use super::autodiff::render_splats_with_visibility;
use crate::core::GaussianCamera;
use crate::core::HostSplats;
use crate::sh::rgb_to_sh0_value;
use crate::training::engine::{host_splats_to_device, GsDiffBackend};
use crate::{Intrinsics, SE3};

fn test_camera() -> GaussianCamera {
    GaussianCamera::new(Intrinsics::from_focal(500.0, 64, 64), SE3::identity())
}

fn test_splats() -> HostSplats {
    HostSplats::from_components(
        vec![0.0, 0.0, 2.0],
        vec![0.2f32.ln(), 0.2f32.ln(), 0.2f32.ln()],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0],
        [1.0, 0.5, 0.25].map(rgb_to_sh0_value).into(),
        0,
    )
    .expect("valid host splats")
}

#[tokio::test(flavor = "current_thread")]
async fn test_render_splats_autodiff() {
    let device = <GsDiffBackend as burn::tensor::backend::Backend>::Device::default();
    let splats = host_splats_to_device::<GsDiffBackend>(&test_splats(), &device);
    let rendered = render_splats_with_visibility(
        &splats,
        &test_camera(),
        (64, 64),
        [0.0, 0.0, 0.0],
        crate::training::DEFAULT_RASTER_COV_BLUR,
    )
    .await;
    let loss = rendered.image.mean();
    let mut grads = loss.backward();

    let transforms = splats
        .transforms
        .grad_remove(&mut grads)
        .expect("transforms grad missing");
    let sh = splats
        .sh_coeffs
        .grad_remove(&mut grads)
        .expect("sh grad missing");
    let opacity = splats
        .raw_opacities
        .grad_remove(&mut grads)
        .expect("opacity grad missing");
    let screen_grad_stats = rendered
        .screen_grad_stats
        .grad_remove(&mut grads)
        .expect("screen-space grad stats missing");

    let transforms_mean = transforms
        .abs()
        .mean()
        .into_scalar_async()
        .await
        .expect("transforms mean scalar");
    let sh_mean = sh
        .abs()
        .mean()
        .into_scalar_async()
        .await
        .expect("sh mean scalar");
    let opacity_mean = opacity
        .abs()
        .mean()
        .into_scalar_async()
        .await
        .expect("opacity mean scalar");
    let screen_stats_mean = screen_grad_stats
        .abs()
        .mean()
        .into_scalar_async()
        .await
        .expect("screen stats mean scalar");

    assert!(
        transforms_mean > 1e-9,
        "expected non-zero transform gradients, got {transforms_mean:e}"
    );
    assert!(
        sh_mean > 1e-9,
        "expected non-zero SH gradients, got {sh_mean:e}"
    );
    assert!(
        opacity_mean > 1e-9,
        "expected non-zero opacity gradients, got {opacity_mean:e}"
    );
    assert!(
        screen_stats_mean > 1e-9,
        "expected non-zero screen-space gradient stats, got {screen_stats_mean:e}"
    );
}
