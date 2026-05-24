use super::render_forward;
use crate::core::GaussianCamera;
use crate::core::HostSplats;
use crate::sh::rgb_to_sh0_value;
use crate::training::engine::{host_splats_to_device, GsBackendBase};
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
async fn test_render_forward_single_gaussian() {
    let device = <GsBackendBase as burn::tensor::backend::Backend>::Device::default();
    let splats = host_splats_to_device::<GsBackendBase>(&test_splats(), &device);
    let output = render_forward::<GsBackendBase>(
        &splats,
        &test_camera(),
        (64, 64),
        [0.0, 0.0, 0.0],
        &device,
        crate::training::DEFAULT_RASTER_COV_BLUR,
    )
    .await;

    assert_eq!(output.out_img.dims(), [64, 64, 4]);
}
