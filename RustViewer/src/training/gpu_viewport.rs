//! egui bridge for RustGS's wgpu-native 3DGS viewport renderer.

use crate::renderer::camera::ArcballCamera;
use crate::training::preview::{PreviewRenderError, PreviewResolution};
use eframe::egui::{TextureId, Vec2};
use eframe::{egui_wgpu, wgpu};
use rustgs::{
    HostSplats, SharedWgpuContext, WgpuViewportCamera, WgpuViewportRenderer, WgpuViewportResolution,
};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct GpuViewportBridge {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: WgpuViewportRenderer,
    texture_id: Option<TextureId>,
}

impl GpuViewportBridge {
    pub fn new(
        _context: SharedWgpuContext,
        device: wgpu::Device,
        queue: wgpu::Queue,
        render_state: &egui_wgpu::RenderState,
    ) -> Self {
        let mut bridge = Self {
            device,
            queue,
            renderer: WgpuViewportRenderer::new(),
            texture_id: None,
        };
        bridge.ensure_texture_id(render_state);
        bridge
    }

    pub fn render_texture_id(
        &mut self,
        render_state: Option<&egui_wgpu::RenderState>,
        splats: Arc<HostSplats>,
        camera: &ArcballCamera,
        panel_size: Vec2,
        scale: f32,
    ) -> Result<Option<TextureId>, PreviewRenderError> {
        if splats.is_empty() {
            return Ok(None);
        }
        let Some(resolution) = PreviewResolution::from_panel_size_scaled(panel_size, scale) else {
            return Ok(None);
        };
        let resolution = WgpuViewportResolution::new(resolution.width, resolution.height)
            .ok_or_else(|| PreviewRenderError::RenderFailed("invalid viewport size".to_string()))?;
        if let Some(render_state) = render_state {
            self.ensure_texture_id(render_state);
        }

        let camera = viewport_camera_from_arcball(camera, resolution);
        let texture_view = self
            .renderer
            .render(
                &self.device,
                &self.queue,
                splats.as_ref(),
                camera,
                resolution,
            )
            .ok_or_else(|| {
                PreviewRenderError::RenderFailed("gpu viewport returned no texture".to_string())
            })?;

        if let Some(render_state) = render_state {
            if let Some(texture_id) = self.texture_id {
                render_state
                    .renderer
                    .write()
                    .update_egui_texture_from_wgpu_texture(
                        &self.device,
                        texture_view,
                        wgpu::FilterMode::Linear,
                        texture_id,
                    );
            }
        }
        Ok(self.texture_id)
    }

    fn ensure_texture_id(&mut self, render_state: &egui_wgpu::RenderState) {
        if self.texture_id.is_some() {
            return;
        }
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("3dgs viewport placeholder texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let id = render_state.renderer.write().register_native_texture(
            &self.device,
            &view,
            wgpu::FilterMode::Linear,
        );
        self.texture_id = Some(id);
    }
}

pub fn viewport_render_scale(last_motion: Option<Instant>, idle_delay: Duration) -> f32 {
    let Some(last_motion) = last_motion else {
        return 1.0;
    };
    if Instant::now().saturating_duration_since(last_motion) < idle_delay {
        0.5
    } else {
        1.0
    }
}

fn viewport_camera_from_arcball(
    camera: &ArcballCamera,
    resolution: WgpuViewportResolution,
) -> WgpuViewportCamera {
    let aspect = resolution.width as f32 / resolution.height.max(1) as f32;
    let view = camera.view_matrix();
    let proj = camera.proj_matrix(aspect);
    let eye = camera.eye();
    WgpuViewportCamera::new(
        view.to_cols_array_2d(),
        proj.to_cols_array_2d(),
        (proj * view).to_cols_array_2d(),
        eye.to_array(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewport_scale_drops_only_while_recently_interacting() {
        assert_eq!(viewport_render_scale(None, Duration::from_millis(180)), 1.0);
        assert_eq!(
            viewport_render_scale(Some(Instant::now()), Duration::from_secs(60),),
            0.5
        );
        assert_eq!(
            viewport_render_scale(
                Some(Instant::now() - Duration::from_secs(60)),
                Duration::from_millis(180),
            ),
            1.0
        );
    }
}
