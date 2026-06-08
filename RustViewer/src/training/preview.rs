//! Shared preview viewport helpers.

use crate::renderer::camera::ArcballCamera;
use eframe::egui::Vec2;
use glam::{Mat3, Quat};
use rustgs::{GaussianCamera, Intrinsics, SE3};

/// Integer preview target size used by GPU viewport paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreviewResolution {
    pub width: usize,
    pub height: usize,
}

impl PreviewResolution {
    pub fn new(width: usize, height: usize) -> Option<Self> {
        if width == 0 || height == 0 {
            return None;
        }
        Some(Self { width, height })
    }

    pub fn from_panel_size(size: Vec2) -> Option<Self> {
        if !size.x.is_finite() || !size.y.is_finite() {
            return None;
        }
        let width = size.x.floor().max(0.0) as usize;
        let height = size.y.floor().max(0.0) as usize;
        Self::new(width, height)
    }

    pub fn from_panel_size_scaled(size: Vec2, scale: f32) -> Option<Self> {
        if !size.x.is_finite()
            || !size.y.is_finite()
            || !scale.is_finite()
            || scale <= 0.0
            || size.x < 1.0
            || size.y < 1.0
        {
            return None;
        }
        let width = (size.x * scale).floor().max(1.0) as usize;
        let height = (size.y * scale).floor().max(1.0) as usize;
        Self::new(width, height)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PreviewRenderError {
    #[error("failed to initialize preview renderer: {0}")]
    RendererInit(String),
    #[error("failed to render preview frame: {0}")]
    RenderFailed(String),
}

/// Build a RustGS camera from current viewer arcball pose and dataset intrinsics.
///
/// The output camera keeps the dataset optics scaled to preview size and uses
/// the world-to-camera transform expected by RustGS projection/rasterization.
pub fn gaussian_camera_from_arcball(
    arcball: &ArcballCamera,
    dataset_intrinsics: Intrinsics,
    resolution: PreviewResolution,
) -> GaussianCamera {
    let sx = resolution.width as f32 / dataset_intrinsics.width.max(1) as f32;
    let sy = resolution.height as f32 / dataset_intrinsics.height.max(1) as f32;
    let scaled_intrinsics = Intrinsics::new(
        dataset_intrinsics.fx * sx,
        dataset_intrinsics.fy * sy,
        dataset_intrinsics.cx * sx,
        dataset_intrinsics.cy * sy,
        resolution.width as u32,
        resolution.height as u32,
    );
    GaussianCamera::new(scaled_intrinsics, arcball_pose_w2c(arcball))
}

/// Build a RustGS camera for a free-orbit viewer viewport.
///
/// Standalone splat files do not carry a dataset camera model, so this
/// synthesizes square-pixel intrinsics from the arcball vertical FOV and
/// viewport dimensions.
pub fn gaussian_camera_from_arcball_viewport(
    arcball: &ArcballCamera,
    resolution: PreviewResolution,
) -> GaussianCamera {
    let width = resolution.width as f32;
    let height = resolution.height as f32;
    let fy = 0.5 * height / (0.5 * arcball.fov_y).tan().max(1e-6);
    let fx = fy;
    let intrinsics = Intrinsics::new(
        fx,
        fy,
        width * 0.5,
        height * 0.5,
        resolution.width as u32,
        resolution.height as u32,
    );
    GaussianCamera::new(intrinsics, arcball_pose_w2c(arcball))
}

fn arcball_pose_w2c(arcball: &ArcballCamera) -> SE3 {
    let eye = arcball.eye();
    let forward = -arcball.backward();
    let right = arcball.right();
    let down = -arcball.up();
    let rotation = Mat3::from_cols(right, down, forward);
    let rotation = Quat::from_mat3(&rotation);

    SE3::from_quat_translation(rotation, eye).inverse()
}

#[cfg(test)]
mod tests {
    use super::{
        gaussian_camera_from_arcball, gaussian_camera_from_arcball_viewport, PreviewResolution,
    };
    use crate::renderer::camera::ArcballCamera;
    use eframe::egui::Vec2;
    use glam::Vec3;
    use rustgs::Intrinsics;

    #[test]
    fn gaussian_camera_from_arcball_scales_intrinsics_and_projects_target_to_center() {
        let arcball =
            ArcballCamera::from_angles(Vec3::ZERO, 5.0, 0.0, 0.0, 0.0, std::f32::consts::FRAC_PI_4);
        let dataset_intrinsics = Intrinsics::new(400.0, 300.0, 320.0, 240.0, 640, 480);
        let resolution = PreviewResolution::new(320, 240).unwrap();

        let camera = gaussian_camera_from_arcball(&arcball, dataset_intrinsics, resolution);
        assert!((camera.intrinsics.fx - 200.0).abs() < 1e-6);
        assert!((camera.intrinsics.fy - 150.0).abs() < 1e-6);
        assert_eq!(camera.intrinsics.width, 320);
        assert_eq!(camera.intrinsics.height, 240);

        let projected = camera
            .project([0.0, 0.0, 0.0])
            .expect("target should be visible");
        assert!((projected[0] - camera.intrinsics.cx).abs() < 1e-3);
        assert!((projected[1] - camera.intrinsics.cy).abs() < 1e-3);
    }

    #[test]
    fn gaussian_camera_from_arcball_matches_viewer_screen_axes() {
        let arcball =
            ArcballCamera::from_angles(Vec3::ZERO, 5.0, 0.0, 0.0, 0.0, std::f32::consts::FRAC_PI_4);
        let dataset_intrinsics = Intrinsics::new(400.0, 400.0, 320.0, 240.0, 640, 480);
        let resolution = PreviewResolution::new(640, 480).unwrap();

        let camera = gaussian_camera_from_arcball(&arcball, dataset_intrinsics, resolution);
        let right = camera.project([1.0, 0.0, 0.0]).unwrap();
        let up = camera.project([0.0, 1.0, 0.0]).unwrap();

        assert!(right[0] > camera.intrinsics.cx);
        assert!(up[1] < camera.intrinsics.cy);
    }

    #[test]
    fn gaussian_camera_from_arcball_viewport_uses_fov_intrinsics() {
        let arcball =
            ArcballCamera::from_angles(Vec3::ZERO, 5.0, 0.0, 0.0, 0.0, std::f32::consts::FRAC_PI_2);
        let resolution = PreviewResolution::new(800, 600).unwrap();

        let camera = gaussian_camera_from_arcball_viewport(&arcball, resolution);

        assert!((camera.intrinsics.fx - 300.0).abs() < 1e-4);
        assert!((camera.intrinsics.fy - 300.0).abs() < 1e-4);
        assert_eq!(camera.intrinsics.cx, 400.0);
        assert_eq!(camera.intrinsics.cy, 300.0);
        let projected = camera
            .project([0.0, 0.0, 0.0])
            .expect("target should be visible");
        assert!((projected[0] - 400.0).abs() < 1e-3);
        assert!((projected[1] - 300.0).abs() < 1e-3);
    }

    #[test]
    fn preview_resolution_scales_panel_size_for_interaction() {
        let resolution = PreviewResolution::from_panel_size_scaled(Vec2::new(801.0, 601.0), 0.5)
            .expect("scaled viewport");

        assert_eq!(resolution.width, 400);
        assert_eq!(resolution.height, 300);
        assert!(PreviewResolution::from_panel_size_scaled(Vec2::ZERO, 0.5).is_none());
    }
}
