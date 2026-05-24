//! Live preview bridge from RustViewer camera state to RustGS evaluation rendering.

use crate::renderer::camera::ArcballCamera;
use eframe::egui::{Color32, ColorImage, Vec2};
use glam::{Mat3, Quat};
use rustgs::{
    EvaluationDevice, GaussianCamera, HostSplats, Intrinsics, SplatEvaluationRenderer, SE3,
};

/// Integer preview target size used by renderer and texture upload paths.
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
}

/// Status result for each preview render request.
#[derive(Debug, Clone)]
pub enum PreviewRenderStatus {
    /// A new frame was rendered and converted to an egui-compatible image.
    Frame(ColorImage),
    /// No snapshot is available yet (or snapshot has zero splats).
    EmptySnapshot,
    /// Viewport size is currently invalid, typically during panel resize.
    InvalidViewport,
}

#[derive(Debug, thiserror::Error)]
pub enum PreviewRenderError {
    #[error("failed to initialize preview renderer: {0}")]
    RendererInit(String),
    #[error("failed to render preview frame: {0}")]
    RenderFailed(String),
    #[error("rendered preview buffer length {actual} does not match expected {expected}")]
    UnexpectedBufferLength { expected: usize, actual: usize },
}

/// Stateful bridge that keeps a cached RustGS renderer and rebuilds it on size changes.
pub struct LivePreviewBridge {
    device: EvaluationDevice,
    renderer: Option<SplatEvaluationRenderer>,
    renderer_resolution: Option<PreviewResolution>,
}

impl Default for LivePreviewBridge {
    fn default() -> Self {
        Self {
            device: EvaluationDevice::Cpu,
            renderer: None,
            renderer_resolution: None,
        }
    }
}

impl LivePreviewBridge {
    pub fn with_device(device: EvaluationDevice) -> Self {
        Self {
            device,
            renderer: None,
            renderer_resolution: None,
        }
    }

    pub fn render_from_arcball(
        &mut self,
        latest_splats: Option<&HostSplats>,
        arcball: &ArcballCamera,
        dataset_intrinsics: Intrinsics,
        panel_size: Vec2,
    ) -> Result<PreviewRenderStatus, PreviewRenderError> {
        let Some(resolution) = PreviewResolution::from_panel_size(panel_size) else {
            return Ok(PreviewRenderStatus::InvalidViewport);
        };

        let Some(splats) = latest_splats else {
            return Ok(PreviewRenderStatus::EmptySnapshot);
        };

        if splats.is_empty() {
            return Ok(PreviewRenderStatus::EmptySnapshot);
        }

        let camera = gaussian_camera_from_arcball(arcball, dataset_intrinsics, resolution);
        self.render_snapshot(splats, &camera, resolution)
    }

    pub fn render_snapshot(
        &mut self,
        splats: &HostSplats,
        camera: &GaussianCamera,
        resolution: PreviewResolution,
    ) -> Result<PreviewRenderStatus, PreviewRenderError> {
        if splats.is_empty() {
            return Ok(PreviewRenderStatus::EmptySnapshot);
        }

        let renderer = self.ensure_renderer(resolution)?;
        let rendered = renderer
            .render(splats, camera)
            .map_err(|err| PreviewRenderError::RenderFailed(err.to_string()))?;
        let image = rgb_f32_to_color_image(&rendered, resolution)?;
        Ok(PreviewRenderStatus::Frame(image))
    }

    fn ensure_renderer(
        &mut self,
        resolution: PreviewResolution,
    ) -> Result<&mut SplatEvaluationRenderer, PreviewRenderError> {
        let needs_rebuild = self.renderer_resolution != Some(resolution) || self.renderer.is_none();
        if needs_rebuild {
            self.renderer = Some(
                SplatEvaluationRenderer::new(
                    resolution.width,
                    resolution.height,
                    self.device,
                    rustgs::DEFAULT_RASTER_COV_BLUR,
                )
                .map_err(|err| PreviewRenderError::RendererInit(err.to_string()))?,
            );
            self.renderer_resolution = Some(resolution);
        }

        self.renderer
            .as_mut()
            .ok_or_else(|| PreviewRenderError::RendererInit("renderer cache missing".to_string()))
    }
}

/// Build a RustGS camera from current viewer arcball pose and dataset intrinsics.
///
/// The output camera keeps the dataset optics (scaled to preview size) and uses a
/// camera-to-world transform with +Z forward, matching RustGS CPU renderer conventions.
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
    GaussianCamera::new(scaled_intrinsics, arcball_pose_c2w(arcball))
}

fn arcball_pose_c2w(arcball: &ArcballCamera) -> SE3 {
    let eye = arcball.eye();
    let forward = -arcball.backward();
    let right = arcball.right();
    let down = -arcball.up();
    let rotation = Mat3::from_cols(right, down, forward);
    let rotation = Quat::from_mat3(&rotation);

    SE3::from_quat_translation(rotation, eye)
}

fn rgb_f32_to_color_image(
    rgb: &[f32],
    resolution: PreviewResolution,
) -> Result<ColorImage, PreviewRenderError> {
    let expected = resolution.width * resolution.height * 3;
    if rgb.len() != expected {
        return Err(PreviewRenderError::UnexpectedBufferLength {
            expected,
            actual: rgb.len(),
        });
    }

    let mut pixels = vec![Color32::BLACK; resolution.width * resolution.height];
    for (idx, chunk) in rgb.chunks_exact(3).enumerate() {
        let r = (chunk[0].clamp(0.0, 1.0) * 255.0).round() as u8;
        let g = (chunk[1].clamp(0.0, 1.0) * 255.0).round() as u8;
        let b = (chunk[2].clamp(0.0, 1.0) * 255.0).round() as u8;
        pixels[idx] = Color32::from_rgb(r, g, b);
    }

    Ok(ColorImage {
        size: [resolution.width, resolution.height],
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        gaussian_camera_from_arcball, LivePreviewBridge, PreviewRenderStatus, PreviewResolution,
    };
    use crate::renderer::camera::ArcballCamera;
    use eframe::egui::{Color32, Vec2};
    use glam::Vec3;
    use rustgs::{HostSplats, Intrinsics};

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
    fn render_from_arcball_returns_invalid_viewport_for_zero_panel_size() {
        let mut bridge = LivePreviewBridge::default();
        let arcball = ArcballCamera::default();
        let intrinsics = Intrinsics::from_focal(300.0, 640, 480);
        let status = bridge
            .render_from_arcball(None, &arcball, intrinsics, Vec2::new(0.0, 128.0))
            .expect("invalid viewport should not error");
        assert!(matches!(status, PreviewRenderStatus::InvalidViewport));
    }

    #[test]
    fn render_from_arcball_returns_empty_snapshot_when_input_missing() {
        let mut bridge = LivePreviewBridge::default();
        let arcball = ArcballCamera::default();
        let intrinsics = Intrinsics::from_focal(300.0, 640, 480);
        let status = bridge
            .render_from_arcball(None, &arcball, intrinsics, Vec2::new(128.0, 128.0))
            .expect("empty input should not error");
        assert!(matches!(status, PreviewRenderStatus::EmptySnapshot));
    }

    #[test]
    fn render_from_arcball_returns_frame_for_valid_snapshot() {
        let mut bridge = LivePreviewBridge::default();
        let arcball =
            ArcballCamera::from_angles(Vec3::ZERO, 5.0, 0.0, 0.0, 0.0, std::f32::consts::FRAC_PI_4);
        let intrinsics = Intrinsics::from_focal(300.0, 128, 128);
        let splats = HostSplats::from_components(
            vec![0.0, 0.0, 0.0],
            vec![0.01f32.ln(), 0.01f32.ln(), 0.01f32.ln()],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0],
            vec![1.0, 1.0, 1.0],
            0,
        )
        .expect("valid single-splat snapshot");

        let status = bridge
            .render_from_arcball(Some(&splats), &arcball, intrinsics, Vec2::new(64.0, 64.0))
            .expect("render should succeed");
        match status {
            PreviewRenderStatus::Frame(image) => {
                assert_eq!(image.size, [64, 64]);
                assert!(image.pixels.iter().any(|pixel| *pixel != Color32::BLACK));
            }
            other => panic!("expected frame status, got {other:?}"),
        }
    }
}
