use super::*;
#[cfg(feature = "gpu")]
use crate::sh::rgb_to_sh0_value;
#[cfg(feature = "gpu")]
use crate::{Intrinsics, SE3};

#[cfg(feature = "gpu")]
fn single_rgb_splats(position: [f32; 3], scale: [f32; 3], color: [f32; 3]) -> HostSplats {
    HostSplats::from_components(
        position.into(),
        scale.map(f32::ln).into(),
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0],
        color.map(rgb_to_sh0_value).into(),
        0,
    )
    .unwrap()
}

#[test]
fn test_renderer_creation() {
    let renderer = GaussianRenderer::new(640, 480);
    assert_eq!(renderer.width, 640);
    assert_eq!(renderer.height, 480);
}

#[cfg(feature = "gpu")]
#[test]
fn test_render_empty_map() {
    let renderer = GaussianRenderer::new(64, 64);
    let splats = HostSplats::default();
    let intrinsics = Intrinsics::from_focal(500.0, 64, 64);
    let camera = GaussianCamera::new(intrinsics, SE3::identity());

    let output = renderer.render_splats(&splats, &camera).unwrap();
    assert_eq!(output.color.len(), 64 * 64 * 3);
    assert_eq!(output.depth.len(), 64 * 64);
}

#[cfg(feature = "gpu")]
#[test]
fn test_render_depth() {
    let renderer = GaussianRenderer::new(64, 64);
    let splats = single_rgb_splats(
        [0.0, 0.0, 1.0],
        [0.01, 0.01, 0.01],
        [1.0, 128.0 / 255.0, 64.0 / 255.0],
    );

    let intrinsics = Intrinsics::from_focal(500.0, 64, 64);
    let camera = GaussianCamera::new(intrinsics, SE3::identity());

    let depth = renderer.render_depth_splats(&splats, &camera).unwrap();

    // Should have some depth values
    assert!(depth.iter().any(|&d| d > 0.0));
}

#[cfg(feature = "gpu")]
#[test]
fn test_render_depth_and_color() {
    let renderer = GaussianRenderer::new(64, 64);
    let splats = single_rgb_splats(
        [0.0, 0.0, 1.0],
        [0.01, 0.01, 0.01],
        [1.0, 128.0 / 255.0, 64.0 / 255.0],
    );

    let intrinsics = Intrinsics::from_focal(500.0, 64, 64);
    let camera = GaussianCamera::new(intrinsics, SE3::identity());

    let (depth, color) = renderer
        .render_depth_and_color_splats(&splats, &camera)
        .unwrap();

    // Should have some depth values
    assert!(depth.iter().any(|&d| d > 0.0));
    // Should have some color values
    assert!(color.iter().any(|&c| c != [0, 0, 0]));
}

#[cfg(feature = "gpu")]
#[test]
fn test_render_splats() {
    let renderer = GaussianRenderer::new(64, 64).with_background(0.1, 0.2, 0.3);
    let splats = single_rgb_splats(
        [0.0, 0.0, 1.0],
        [0.01, 0.01, 0.01],
        [1.0, 128.0 / 255.0, 64.0 / 255.0],
    );
    let intrinsics = Intrinsics::from_focal(500.0, 64, 64);
    let camera = GaussianCamera::new(intrinsics, SE3::identity());

    let output = renderer.render_splats(&splats, &camera).unwrap();
    assert!(output.depth.iter().any(|&d| d > 0.0));
    assert!(output
        .color
        .iter()
        .any(|&channel| channel != 25 && channel != 51 && channel != 76));
}

#[cfg(feature = "gpu")]
#[test]
fn test_projection_uses_world_to_camera_extrinsics() {
    let renderer = GaussianRenderer::new(64, 64);
    let splats = single_rgb_splats([1.0, 0.0, 2.0], [0.01, 0.01, 0.01], [1.0, 1.0, 1.0]);
    let intrinsics = Intrinsics::from_focal(500.0, 64, 64);
    // World-to-camera: camera translated +1 on world X => t_cw = [-1, 0, 0].
    let camera = GaussianCamera::new(
        intrinsics,
        SE3::new(&[0.0, 0.0, 0.0, 1.0], &[-1.0, 0.0, 0.0]),
    );

    let projected = renderer.project_visible_splats(splats.as_view(), &camera);
    assert_eq!(projected.len(), 1);
    let (_gc, _depth, u, v, _radius) = projected[0];

    assert!(
        (u - 32.0).abs() < 1.0,
        "expected world-to-camera projection near principal point, got u={u}"
    );
    assert!(
        (v - 32.0).abs() < 1.0,
        "expected world-to-camera projection near principal point, got v={v}"
    );
}
