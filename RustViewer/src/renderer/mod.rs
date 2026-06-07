//! 3D scene renderer using eframe's wgpu integration.

pub mod camera;
pub mod pipelines;
pub mod scene;

use eframe::egui_wgpu;
use eframe::wgpu;

use camera::ArcballCamera;
use pipelines::{
    create_gaussian_pipeline, create_line_pipeline, create_mesh_pipeline, create_point_pipeline,
    create_wireframe_pipeline, expand_points_to_quads, project_gaussians_to_splats, PointVertex,
    GAUSSIAN_QUAD_INDICES, GAUSSIAN_QUAD_VERTICES,
};
use scene::{MeshGpuVertex, Scene};

/// Holds all GPU resources needed to render a scene.
pub struct SceneRenderer {
    point_pipeline: wgpu::RenderPipeline,
    gaussian_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    mesh_pipeline: wgpu::RenderPipeline,
    wireframe_pipeline: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    uniform_bg: wgpu::BindGroup,
    // Vertex/index buffers updated each frame
    point_vbuf: Option<wgpu::Buffer>,
    point_ibuf: Option<wgpu::Buffer>,
    point_count: u32,
    gauss_quad_vbuf: wgpu::Buffer,
    gauss_ibuf: wgpu::Buffer,
    gauss_instance_vbuf: Option<wgpu::Buffer>,
    gauss_instance_count: u32,
    line_vbuf: Option<wgpu::Buffer>,
    line_count: u32,
    mesh_vbuf: Option<wgpu::Buffer>,
    mesh_ibuf: Option<wgpu::Buffer>,
    mesh_index_count: u32,
    wireframe_ibuf: Option<wgpu::Buffer>,
    wireframe_index_count: u32,
    // Cached layer flags (set during prepare, used in paint)
    layer_trajectory: bool,
    layer_map_points: bool,
    layer_gaussians: bool,
    layer_mesh_solid: bool,
    layer_mesh_wireframe: bool,
}

impl SceneRenderer {
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniform_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(256),
                },
                count: None,
            }],
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniform_buf"),
            // 256 bytes: satisfies min_uniform_buffer_offset_alignment on all backends.
            // The view_proj matrix (64 bytes) is written at offset 0; the remaining padding
            // is never read by the shader.
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniform_bg"),
            layout: &uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let point_pipeline = create_point_pipeline(device, surface_format);
        let gaussian_pipeline = create_gaussian_pipeline(device, surface_format);
        let line_pipeline = create_line_pipeline(device, surface_format);
        let mesh_pipeline = create_mesh_pipeline(device, surface_format, &uniform_bgl);
        let wireframe_pipeline = create_wireframe_pipeline(device, surface_format, &uniform_bgl);
        let gauss_quad_vbuf =
            create_vertex_buffer(device, bytemuck::cast_slice(&GAUSSIAN_QUAD_VERTICES));
        let gauss_ibuf = create_index_buffer(device, bytemuck::cast_slice(&GAUSSIAN_QUAD_INDICES));

        Self {
            point_pipeline,
            gaussian_pipeline,
            line_pipeline,
            mesh_pipeline,
            wireframe_pipeline,
            uniform_buf,
            uniform_bg,
            point_vbuf: None,
            point_ibuf: None,
            point_count: 0,
            gauss_quad_vbuf,
            gauss_ibuf,
            gauss_instance_vbuf: None,
            gauss_instance_count: 0,
            line_vbuf: None,
            line_count: 0,
            mesh_vbuf: None,
            mesh_ibuf: None,
            mesh_index_count: 0,
            wireframe_ibuf: None,
            wireframe_index_count: 0,
            layer_trajectory: true,
            layer_map_points: true,
            layer_gaussians: true,
            layer_mesh_solid: false,
            layer_mesh_wireframe: true,
        }
    }

    /// Upload scene data to GPU buffers. Called each frame from prepare().
    pub fn update_buffers(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        camera: &ArcballCamera,
        viewport_size: [f32; 2],
    ) {
        self.update_buffers_with_options(device, queue, scene, camera, viewport_size, false);
    }

    /// Upload scene data to GPU buffers with optional layer overrides.
    pub fn update_buffers_with_options(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        camera: &ArcballCamera,
        viewport_size: [f32; 2],
        suppress_gaussians: bool,
    ) {
        // Cache layer flags for use in paint()
        self.layer_trajectory = scene.layers.trajectory;
        self.layer_map_points = scene.layers.map_points;
        self.layer_gaussians = scene.layers.gaussians && !suppress_gaussians;
        self.layer_mesh_solid = scene.layers.mesh_solid;
        self.layer_mesh_wireframe = scene.layers.mesh_wireframe;

        let aspect = viewport_size[0] / viewport_size[1].max(1.0);
        let vp = camera.view_proj(aspect);
        let vp_arr = vp.to_cols_array_2d();

        // Update uniform buffer (view_proj matrix)
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::cast_slice(&vp_arr));

        // Map points
        if scene.layers.map_points && !scene.map_points.is_empty() {
            let (verts, idxs) = expand_points_to_quads(
                &scene.map_points,
                &scene.map_point_colors,
                viewport_size[0],
                viewport_size[1],
                vp_arr,
            );
            self.point_count = idxs.len() as u32;
            self.point_vbuf = Some(create_vertex_buffer(device, bytemuck::cast_slice(&verts)));
            self.point_ibuf = Some(create_index_buffer(device, bytemuck::cast_slice(&idxs)));
        } else {
            self.point_count = 0;
        }

        // Gaussians
        if self.layer_gaussians && !scene.gaussians.is_empty() {
            let instances = project_gaussians_to_splats(&scene.gaussians, camera, viewport_size);
            self.gauss_instance_count = instances.len() as u32;
            self.gauss_instance_vbuf = (!instances.is_empty())
                .then(|| create_vertex_buffer(device, bytemuck::cast_slice(&instances)));
        } else {
            self.gauss_instance_count = 0;
            self.gauss_instance_vbuf = None;
        }

        // Trajectory (line list — also pre-project to NDC)
        if scene.layers.trajectory && scene.trajectory.len() >= 2 {
            let traj_color = [0.2f32, 0.6, 1.0];
            let mut line_verts: Vec<PointVertex> = Vec::new();
            for i in 0..scene.trajectory.len() - 1 {
                // Project both endpoints; skip the segment if either is behind the camera.
                let mut ndc_pair = [[0.0f32; 3]; 2];
                let mut valid = true;
                for (j, &pos) in [scene.trajectory[i], scene.trajectory[i + 1]]
                    .iter()
                    .enumerate()
                {
                    let p = glam::Vec4::new(pos[0], pos[1], pos[2], 1.0);
                    let clip = vp * p;
                    if clip.w <= 0.0 {
                        valid = false;
                        break;
                    }
                    ndc_pair[j] = [clip.x / clip.w, clip.y / clip.w, clip.z / clip.w];
                }
                if valid {
                    for ndc in &ndc_pair {
                        line_verts.push(PointVertex {
                            position: *ndc,
                            color: traj_color,
                        });
                    }
                }
            }
            self.line_count = line_verts.len() as u32;
            self.line_vbuf = Some(create_vertex_buffer(
                device,
                bytemuck::cast_slice(&line_verts),
            ));
        } else {
            self.line_count = 0;
        }

        // Mesh (uses uniform buffer for world-space transform)
        if (scene.layers.mesh_solid || scene.layers.mesh_wireframe)
            && !scene.mesh_vertices.is_empty()
        {
            self.mesh_index_count = scene.mesh_indices.len() as u32;
            self.wireframe_index_count = scene.mesh_edge_indices.len() as u32;
            self.mesh_vbuf = Some(create_vertex_buffer(
                device,
                bytemuck::cast_slice::<MeshGpuVertex, u8>(&scene.mesh_vertices),
            ));
            self.mesh_ibuf = Some(create_index_buffer(
                device,
                bytemuck::cast_slice(&scene.mesh_indices),
            ));
            self.wireframe_ibuf = Some(create_index_buffer(
                device,
                bytemuck::cast_slice(&scene.mesh_edge_indices),
            ));
        } else {
            self.mesh_index_count = 0;
            self.wireframe_index_count = 0;
        }
    }

    /// Issue draw calls using cached layer flags (called from paint() which can't access Scene).
    // TODO(F3): CPU MVP expansion for points is O(n) per frame — move to GPU compute post-MVP.
    // TODO(F4): All GPU buffers rebuilt every frame regardless of changes — add dirty-flag tracking post-MVP.
    pub fn render_with_layers(&self, rpass: &mut wgpu::RenderPass<'static>) {
        // Draw order: mesh first, then trajectory, then map points, then Gaussians

        if self.layer_mesh_solid {
            if let (Some(vbuf), Some(ibuf), count) =
                (&self.mesh_vbuf, &self.mesh_ibuf, self.mesh_index_count)
            {
                if count > 0 {
                    rpass.set_pipeline(&self.mesh_pipeline);
                    rpass.set_bind_group(0, &self.uniform_bg, &[]);
                    rpass.set_vertex_buffer(0, vbuf.slice(..));
                    rpass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    rpass.draw_indexed(0..count, 0, 0..1);
                }
            }
        }

        if self.layer_mesh_wireframe {
            if let (Some(vbuf), Some(ibuf), count) = (
                &self.mesh_vbuf,
                &self.wireframe_ibuf,
                self.wireframe_index_count,
            ) {
                if count > 0 {
                    rpass.set_pipeline(&self.wireframe_pipeline);
                    rpass.set_bind_group(0, &self.uniform_bg, &[]);
                    rpass.set_vertex_buffer(0, vbuf.slice(..));
                    rpass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    rpass.draw_indexed(0..count, 0, 0..1);
                }
            }
        }

        if self.layer_trajectory {
            if let (Some(vbuf), count) = (&self.line_vbuf, self.line_count) {
                if count > 0 {
                    rpass.set_pipeline(&self.line_pipeline);
                    rpass.set_vertex_buffer(0, vbuf.slice(..));
                    rpass.draw(0..count, 0..1);
                }
            }
        }

        if self.layer_map_points {
            if let (Some(vbuf), Some(ibuf), count) =
                (&self.point_vbuf, &self.point_ibuf, self.point_count)
            {
                if count > 0 {
                    rpass.set_pipeline(&self.point_pipeline);
                    rpass.set_vertex_buffer(0, vbuf.slice(..));
                    rpass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    rpass.draw_indexed(0..count, 0, 0..1);
                }
            }
        }

        if self.layer_gaussians {
            if let (Some(instance_vbuf), count) =
                (&self.gauss_instance_vbuf, self.gauss_instance_count)
            {
                if count > 0 {
                    rpass.set_pipeline(&self.gaussian_pipeline);
                    rpass.set_vertex_buffer(0, self.gauss_quad_vbuf.slice(..));
                    rpass.set_vertex_buffer(1, instance_vbuf.slice(..));
                    rpass.set_index_buffer(self.gauss_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    rpass.draw_indexed(0..GAUSSIAN_QUAD_INDICES.len() as u32, 0, 0..count);
                }
            }
        }
    }
}

/// Data passed to the wgpu callback each frame.
pub struct ViewerCallback {
    pub scene: std::sync::Arc<std::sync::Mutex<Scene>>,
    pub camera: ArcballCamera,
    pub viewport_size: [f32; 2],
    pub suppress_gaussians: bool,
    /// Surface format read from eframe at startup — used for lazy pipeline init.
    pub surface_format: wgpu::TextureFormat,
}

impl egui_wgpu::CallbackTrait for ViewerCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        // Lazy-init the renderer on first frame using the actual surface format
        // supplied by eframe (read from wgpu RenderState at app startup).
        if resources.get::<SceneRenderer>().is_none() {
            let renderer = SceneRenderer::new(device, queue, self.surface_format);
            resources.insert(renderer);
        }

        if let Some(renderer) = resources.get_mut::<SceneRenderer>() {
            if let Ok(scene) = self.scene.lock() {
                renderer.update_buffers_with_options(
                    device,
                    queue,
                    &scene,
                    &self.camera,
                    self.viewport_size,
                    self.suppress_gaussians,
                );
            }
        }

        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        if let Some(renderer) = resources.get::<SceneRenderer>() {
            // We need to pass layer info — clone it from scene snapshot taken during prepare
            // Since we can't lock scene here (would borrow from resources), use cached layer flags
            renderer.render_with_layers(render_pass);
        }
    }
}

fn create_vertex_buffer(device: &wgpu::Device, data: &[u8]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("vertex_buf"),
        contents: data,
        usage: wgpu::BufferUsages::VERTEX,
    })
}

fn create_index_buffer(device: &wgpu::Device, data: &[u8]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("index_buf"),
        contents: data,
        usage: wgpu::BufferUsages::INDEX,
    })
}
