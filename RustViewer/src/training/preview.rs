//! Live preview bridge from RustViewer camera state to RustGS evaluation rendering.

use crate::renderer::camera::ArcballCamera;
use eframe::egui::{Color32, ColorImage, Vec2};
use glam::{Mat3, Quat};
use rustgs::{
    EvaluationDevice, GaussianCamera, HostSplats, Intrinsics, SharedWgpuContext,
    SplatEvaluationRenderer, SE3,
};
use std::sync::{mpsc, Arc};
use std::thread;

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

struct AsyncPreviewRequest {
    id: u64,
    splats: Arc<HostSplats>,
    camera: GaussianCamera,
    resolution: PreviewResolution,
}

struct AsyncPreviewResult {
    id: u64,
    result: Result<PreviewRenderStatus, PreviewRenderError>,
}

/// Background preview renderer for interactive viewports.
///
/// Requests are deliberately lossy: when camera input outruns rendering, the
/// worker drains queued requests and renders only the newest camera.
pub struct AsyncPreviewBridge {
    request_tx: mpsc::Sender<AsyncPreviewRequest>,
    result_rx: mpsc::Receiver<AsyncPreviewResult>,
    next_request_id: u64,
    latest_requested_id: Option<u64>,
    latest_completed_id: Option<u64>,
    latest_requested_resolution: Option<PreviewResolution>,
    _worker: thread::JoinHandle<()>,
}

#[derive(Clone)]
enum PreviewRenderDevice {
    Evaluation(EvaluationDevice),
    SharedWgpu(SharedWgpuContext),
}

impl Default for AsyncPreviewBridge {
    fn default() -> Self {
        Self::with_device(EvaluationDevice::Gpu)
    }
}

impl AsyncPreviewBridge {
    pub fn with_device(device: EvaluationDevice) -> Self {
        Self::with_render_device(PreviewRenderDevice::Evaluation(device))
    }

    pub fn with_shared_wgpu_context(context: SharedWgpuContext) -> Self {
        Self::with_render_device(PreviewRenderDevice::SharedWgpu(context))
    }

    fn with_render_device(render_device: PreviewRenderDevice) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<AsyncPreviewRequest>();
        let (result_tx, result_rx) = mpsc::channel::<AsyncPreviewResult>();
        let worker = thread::spawn(move || {
            let mut bridge = match render_device {
                PreviewRenderDevice::Evaluation(device) => LivePreviewBridge::with_device(device),
                PreviewRenderDevice::SharedWgpu(context) => {
                    LivePreviewBridge::with_shared_wgpu_context(context)
                }
            };
            while let Ok(mut request) = request_rx.recv() {
                while let Ok(newer_request) = request_rx.try_recv() {
                    request = newer_request;
                }

                let result = bridge.render_snapshot(
                    request.splats.as_ref(),
                    &request.camera,
                    request.resolution,
                );
                if result_tx
                    .send(AsyncPreviewResult {
                        id: request.id,
                        result,
                    })
                    .is_err()
                {
                    break;
                }
            }
        });

        Self {
            request_tx,
            result_rx,
            next_request_id: 1,
            latest_requested_id: None,
            latest_completed_id: None,
            latest_requested_resolution: None,
            _worker: worker,
        }
    }

    pub fn request_render(
        &mut self,
        splats: Arc<HostSplats>,
        camera: GaussianCamera,
        resolution: PreviewResolution,
    ) -> Result<(), PreviewRenderError> {
        let id = self.next_request_id;
        self.next_request_id = self.next_request_id.saturating_add(1);
        self.request_tx
            .send(AsyncPreviewRequest {
                id,
                splats,
                camera,
                resolution,
            })
            .map_err(|err| {
                PreviewRenderError::RenderFailed(format!("preview worker stopped: {err}"))
            })?;
        self.latest_requested_id = Some(id);
        self.latest_requested_resolution = Some(resolution);
        Ok(())
    }

    pub fn poll_latest(&mut self) -> Option<Result<PreviewRenderStatus, PreviewRenderError>> {
        let mut latest = None;
        while let Ok(result) = self.result_rx.try_recv() {
            if self
                .latest_completed_id
                .is_none_or(|completed_id| result.id > completed_id)
            {
                latest = Some(result);
            }
        }

        latest.map(|result| {
            self.latest_completed_id = Some(result.id);
            result.result
        })
    }

    pub fn is_render_pending(&self) -> bool {
        self.latest_requested_id.is_some() && self.latest_completed_id != self.latest_requested_id
    }

    pub fn has_pending_for(&self, resolution: PreviewResolution) -> bool {
        self.is_render_pending() && self.latest_requested_resolution == Some(resolution)
    }

    pub fn clear_pending(&mut self) {
        self.latest_requested_id = None;
        self.latest_completed_id = None;
        self.latest_requested_resolution = None;
        while self.result_rx.try_recv().is_ok() {}
    }
}

/// Stateful bridge that keeps a cached RustGS renderer and rebuilds it on size changes.
pub struct LivePreviewBridge {
    device: EvaluationDevice,
    shared_wgpu_context: Option<SharedWgpuContext>,
    renderer: Option<SplatEvaluationRenderer>,
    renderer_resolution: Option<PreviewResolution>,
}

impl Default for LivePreviewBridge {
    fn default() -> Self {
        Self {
            device: EvaluationDevice::Gpu,
            shared_wgpu_context: None,
            renderer: None,
            renderer_resolution: None,
        }
    }
}

impl LivePreviewBridge {
    pub fn with_device(device: EvaluationDevice) -> Self {
        Self {
            device,
            shared_wgpu_context: None,
            renderer: None,
            renderer_resolution: None,
        }
    }

    pub fn with_shared_wgpu_context(context: SharedWgpuContext) -> Self {
        Self {
            device: EvaluationDevice::Gpu,
            shared_wgpu_context: Some(context),
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
            .render_rgba_f32(splats, camera)
            .map_err(|err| PreviewRenderError::RenderFailed(err.to_string()))?;
        let image = rgba_f32_to_color_image(&rendered, resolution)?;
        Ok(PreviewRenderStatus::Frame(image))
    }

    fn ensure_renderer(
        &mut self,
        resolution: PreviewResolution,
    ) -> Result<&mut SplatEvaluationRenderer, PreviewRenderError> {
        let needs_rebuild = self.renderer_resolution != Some(resolution) || self.renderer.is_none();
        if needs_rebuild {
            self.renderer = Some(if let Some(context) = self.shared_wgpu_context.clone() {
                SplatEvaluationRenderer::new_with_wgpu_context(
                    resolution.width,
                    resolution.height,
                    context,
                    rustgs::DEFAULT_RASTER_COV_BLUR,
                )
                .map_err(|err| PreviewRenderError::RendererInit(err.to_string()))?
            } else {
                SplatEvaluationRenderer::new(
                    resolution.width,
                    resolution.height,
                    self.device,
                    rustgs::DEFAULT_RASTER_COV_BLUR,
                )
                .map_err(|err| PreviewRenderError::RendererInit(err.to_string()))?
            });
            self.renderer_resolution = Some(resolution);
        }

        self.renderer
            .as_mut()
            .ok_or_else(|| PreviewRenderError::RendererInit("renderer cache missing".to_string()))
    }
}

/// Build a RustGS camera from current viewer arcball pose and dataset intrinsics.
///
/// The output camera keeps the dataset optics (scaled to preview size) and uses
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
/// Unlike the live training preview, standalone splat files do not carry a
/// dataset camera model, so this synthesizes square-pixel intrinsics from the
/// arcball vertical FOV and viewport dimensions.
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

fn rgba_f32_to_color_image(
    rgba: &[f32],
    resolution: PreviewResolution,
) -> Result<ColorImage, PreviewRenderError> {
    let expected = resolution.width * resolution.height * 4;
    if rgba.len() != expected {
        return Err(PreviewRenderError::UnexpectedBufferLength {
            expected,
            actual: rgba.len(),
        });
    }

    let mut pixels = vec![Color32::BLACK; resolution.width * resolution.height];
    for (idx, chunk) in rgba.chunks_exact(4).enumerate() {
        let r = (chunk[0].clamp(0.0, 1.0) * 255.0).round() as u8;
        let g = (chunk[1].clamp(0.0, 1.0) * 255.0).round() as u8;
        let b = (chunk[2].clamp(0.0, 1.0) * 255.0).round() as u8;
        pixels[idx] = Color32::from_rgb(r, g, b);
    }

    Ok(ColorImage::new(
        [resolution.width, resolution.height],
        pixels,
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        gaussian_camera_from_arcball, gaussian_camera_from_arcball_viewport, AsyncPreviewBridge,
        AsyncPreviewResult, LivePreviewBridge, PreviewRenderStatus, PreviewResolution,
    };
    use crate::renderer::camera::ArcballCamera;
    use eframe::egui::{Color32, Vec2};
    use glam::Vec3;
    use rustgs::{EvaluationDevice, HostSplats, Intrinsics};
    use std::sync::{mpsc, Arc};
    use std::time::{Duration, Instant};

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

    #[test]
    fn async_preview_bridge_delivers_requested_frame() {
        let mut bridge = AsyncPreviewBridge::with_device(EvaluationDevice::Cpu);
        let arcball =
            ArcballCamera::from_angles(Vec3::ZERO, 5.0, 0.0, 0.0, 0.0, std::f32::consts::FRAC_PI_4);
        let resolution = PreviewResolution::new(32, 32).unwrap();
        let camera = gaussian_camera_from_arcball_viewport(&arcball, resolution);
        let splats = Arc::new(
            HostSplats::from_components(
                vec![0.0, 0.0, 0.0],
                vec![0.5f32.ln(), 0.5f32.ln(), 0.5f32.ln()],
                vec![1.0, 0.0, 0.0, 0.0],
                vec![4.0],
                vec![1.0, 1.0, 1.0],
                0,
            )
            .expect("valid single-splat snapshot"),
        );

        bridge
            .request_render(splats, camera, resolution)
            .expect("request should enqueue");
        assert!(bridge.is_render_pending());

        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(result) = bridge.poll_latest() {
                match result.expect("render should succeed") {
                    PreviewRenderStatus::Frame(image) => {
                        assert_eq!(image.size, [32, 32]);
                        assert!(image.pixels.iter().any(|pixel| *pixel != Color32::BLACK));
                        assert!(!bridge.is_render_pending());
                        return;
                    }
                    other => panic!("expected frame status, got {other:?}"),
                }
            }

            assert!(
                Instant::now() < deadline,
                "async preview worker did not produce a frame"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn async_preview_bridge_delivers_stale_completed_frame_while_newer_request_is_pending() {
        let (request_tx, _request_rx) = mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        let worker = std::thread::spawn(|| {});
        let mut bridge = AsyncPreviewBridge {
            request_tx,
            result_rx,
            next_request_id: 3,
            latest_requested_id: Some(2),
            latest_completed_id: None,
            latest_requested_resolution: Some(PreviewResolution::new(32, 32).unwrap()),
            _worker: worker,
        };

        result_tx
            .send(AsyncPreviewResult {
                id: 1,
                result: Ok(PreviewRenderStatus::EmptySnapshot),
            })
            .expect("test result should enqueue");

        let result = bridge
            .poll_latest()
            .expect("stale completed frame should still be delivered")
            .expect("test result should be ok");
        assert!(matches!(result, PreviewRenderStatus::EmptySnapshot));
        assert_eq!(bridge.latest_completed_id, Some(1));
        assert!(
            bridge.is_render_pending(),
            "newer requested frame should remain pending"
        );
    }
}
