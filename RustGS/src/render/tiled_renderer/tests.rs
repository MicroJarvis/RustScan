use super::*;

#[cfg(feature = "gpu")]
use crate::sh::rgb_to_sh0_value;

#[cfg(feature = "gpu")]
fn opacity_to_logit(opacity: f32) -> f32 {
    let clamped = opacity.clamp(1e-6, 1.0 - 1e-6);
    (clamped / (1.0 - clamped)).ln()
}

#[cfg(feature = "gpu")]
fn single_rgb_splats(
    position: [f32; 3],
    scale: [f32; 3],
    rotation: [f32; 4],
    opacity: f32,
    color: [f32; 3],
) -> crate::core::HostSplats {
    crate::core::HostSplats::from_components(
        position.into(),
        scale.map(f32::ln).into(),
        rotation.into(),
        vec![opacity_to_logit(opacity)],
        color.map(rgb_to_sh0_value).into(),
        0,
    )
    .unwrap()
}

#[test]
fn test_tiled_renderer() {
    let renderer = TiledRenderer::new(640, 480);
    assert_eq!(renderer.num_tiles_x, 40);
    assert_eq!(renderer.num_tiles_y, 30);
}

#[cfg(feature = "gpu")]
#[test]
fn test_splat_projection() {
    let renderer = TiledRenderer::new(64, 64);
    let splats = single_rgb_splats(
        [0.0, 0.0, 1.0],
        [0.01, 0.01, 0.01],
        [1.0, 0.0, 0.0, 0.0],
        0.5,
        [1.0, 0.5, 0.25],
    );

    let rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    let projected = renderer.project_splats(
        splats.as_view(),
        500.0,
        500.0,
        32.0,
        32.0,
        &rotation,
        &[0.0, 0.0, 0.0],
    );

    assert!(!projected.is_empty());
    assert!(projected[0].depth > 0.0);
}

#[cfg(feature = "gpu")]
#[test]
fn test_render_splats_matches_project_then_render() {
    let renderer = TiledRenderer::new(64, 64);
    let splats = single_rgb_splats(
        [0.0, 0.0, 1.0],
        [0.01, 0.01, 0.01],
        [1.0, 0.0, 0.0, 0.0],
        0.5,
        [1.0, 0.5, 0.25],
    );
    let rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    let projected = renderer.project_splats(
        splats.as_view(),
        500.0,
        500.0,
        32.0,
        32.0,
        &rotation,
        &[0.0, 0.0, 0.0],
    );
    let projected_render = renderer.render_projected(projected);
    let splat_render = renderer.render_splats(
        splats.as_view(),
        500.0,
        500.0,
        32.0,
        32.0,
        &rotation,
        &[0.0, 0.0, 0.0],
    );

    assert_eq!(projected_render.color, splat_render.color);
    assert_eq!(projected_render.depth, splat_render.depth);
}

/// Verify that the full 2D covariance projection is correct.
///
/// For an axis-aligned Gaussian (identity rotation) at [0,0,z] the cross-term
/// cov_xy should be zero.  For a tilted Gaussian it should be non-zero.
#[cfg(feature = "gpu")]
#[test]
fn test_full_2d_covariance_identity_rotation() {
    let renderer = TiledRenderer::new(64, 64);
    let splats = single_rgb_splats(
        [0.0, 0.0, 2.0],
        [0.1, 0.05, 0.02],
        [1.0, 0.0, 0.0, 0.0], // identity
        0.8,
        [1.0, 0.0, 0.0],
    );

    let rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let projected = renderer.project_splats(
        splats.as_view(),
        500.0,
        500.0,
        0.0,
        0.0,
        &rotation,
        &[0.0, 0.0, 0.0],
    );

    assert!(!projected.is_empty());
    let p = &projected[0];
    // With identity rotation and centered projection, cov_xy should be ~0.
    assert!(
        p.cov_xy.abs() < 1e-4,
        "cov_xy should be ~0 for identity rotation at origin, got {}",
        p.cov_xy
    );
    // cov_xx and cov_yy must be positive.
    assert!(p.cov_xx > 0.0, "cov_xx must be positive, got {}", p.cov_xx);
    assert!(p.cov_yy > 0.0, "cov_yy must be positive, got {}", p.cov_yy);
}

#[cfg(feature = "gpu")]
#[test]
fn test_full_2d_covariance_rotated_gaussian() {
    let renderer = TiledRenderer::new(64, 64);

    // 45-degree rotation around Z axis: quaternion = [cos(π/8), 0, 0, sin(π/8)]
    let angle = std::f32::consts::PI / 4.0;
    let (sin_half, cos_half) = ((angle / 2.0).sin(), (angle / 2.0).cos());

    let splats = single_rgb_splats(
        [0.5, 0.5, 2.0],
        [0.3, 0.05, 0.01],
        [cos_half, 0.0, 0.0, sin_half], // 45° around Z
        0.8,
        [0.0, 1.0, 0.0],
    );

    let rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let projected = renderer.project_splats(
        splats.as_view(),
        500.0,
        500.0,
        0.0,
        0.0,
        &rotation,
        &[0.0, 0.0, 0.0],
    );

    assert!(!projected.is_empty());
    let p = &projected[0];
    // For a rotated elongated Gaussian with asymmetric scales, cov_xy != 0.
    assert!(
        p.cov_xy.abs() > 1e-6,
        "cov_xy should be non-zero for a rotated Gaussian, got {}",
        p.cov_xy
    );
}

#[test]
fn test_depth_sorting() {
    let mut gaussians = vec![
        ProjectedGaussian {
            x: 0.0,
            y: 0.0,
            depth: 2.0,
            cov_xx: 1.0,
            cov_xy: 0.0,
            cov_yy: 1.0,
            opacity: 0.5,
            color: [1.0, 0.0, 0.0],
            orig_idx: 0,
        },
        ProjectedGaussian {
            x: 0.0,
            y: 0.0,
            depth: 1.0,
            cov_xx: 1.0,
            cov_xy: 0.0,
            cov_yy: 1.0,
            opacity: 0.5,
            color: [0.0, 1.0, 0.0],
            orig_idx: 1,
        },
        ProjectedGaussian {
            x: 0.0,
            y: 0.0,
            depth: 3.0,
            cov_xx: 1.0,
            cov_xy: 0.0,
            cov_yy: 1.0,
            opacity: 0.5,
            color: [0.0, 0.0, 1.0],
            orig_idx: 2,
        },
    ];

    let renderer = TiledRenderer::new(64, 64);
    renderer.sort_by_depth(&mut gaussians);

    // Should be sorted front to back (near to far)
    assert_eq!(gaussians[0].depth, 1.0);
    assert_eq!(gaussians[1].depth, 2.0);
    assert_eq!(gaussians[2].depth, 3.0);
}
