//! Realtime viewport backed by RustGS's canonical Burn/CubeCL renderer.

use burn::prelude::Backend;
use burn::tensor::TensorPrimitive;
use burn_wgpu::{CubeTensor, WgpuResource, WgpuRuntime};
use wgpu::util::DeviceExt;

use crate::core::{GaussianCamera, HostSplats, HostSplatsCacheKey};
use crate::training::engine::{host_splats_to_device, DeviceSplats, GsBackendBase};
use crate::training::forward;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BurnViewportResolution {
    pub width: u32,
    pub height: u32,
}

impl BurnViewportResolution {
    pub fn new(width: usize, height: usize) -> Option<Self> {
        if width == 0 || height == 0 {
            return None;
        }
        Some(Self {
            width: u32::try_from(width).ok()?,
            height: u32::try_from(height).ok()?,
        })
    }
}

pub struct BurnViewportRenderer {
    device: <GsBackendBase as Backend>::Device,
    runtime: tokio::runtime::Runtime,
    cached_splats: Option<CachedSplats>,
    texture: Option<ViewportTexture>,
    converter: Option<TextureConverter>,
    raster_cov_blur: f32,
}

struct CachedSplats {
    key: HostSplatsCacheKey,
    splats: DeviceSplats<GsBackendBase>,
}

struct ViewportTexture {
    resolution: BurnViewportResolution,
    #[allow(dead_code)]
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

struct TextureConverter {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TextureConvertUniforms {
    width: u32,
    height: u32,
}

impl BurnViewportRenderer {
    pub fn new(context: crate::SharedWgpuContext) -> Result<Self, String> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|err| format!("failed to initialize viewport runtime: {err}"))?;
        Ok(Self {
            device: context.device(),
            runtime,
            cached_splats: None,
            texture: None,
            converter: None,
            raster_cov_blur: crate::DEFAULT_RASTER_COV_BLUR,
        })
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        splats: &HostSplats,
        camera: &GaussianCamera,
        resolution: BurnViewportResolution,
    ) -> Result<Option<&wgpu::TextureView>, String> {
        if splats.is_empty() {
            return Ok(None);
        }

        self.ensure_splats_uploaded(splats);
        self.ensure_texture(device, resolution);
        self.ensure_converter(device);

        let cached = self
            .cached_splats
            .as_ref()
            .ok_or_else(|| "viewport splat cache was not initialized".to_string())?;

        let img_size = (resolution.width, resolution.height);
        let rendered = self
            .runtime
            .block_on(forward::render_forward::<GsBackendBase>(
                &cached.splats,
                camera,
                img_size,
                [0.0, 0.0, 0.0],
                &self.device,
                self.raster_cov_blur,
            ));

        let out_tensor = match rendered.out_img.into_primitive() {
            TensorPrimitive::Float(tensor) => tensor,
            TensorPrimitive::QFloat(_) => {
                return Err("viewport renderer produced quantized output".to_string());
            }
        };
        copy_tensor_to_texture(
            device,
            queue,
            self.converter
                .as_mut()
                .ok_or_else(|| "viewport texture converter was not initialized".to_string())?,
            &out_tensor,
            self.texture
                .as_ref()
                .ok_or_else(|| "viewport texture was not initialized".to_string())?,
            resolution,
        )?;

        Ok(self.texture.as_ref().map(|texture| &texture.view))
    }

    fn ensure_splats_uploaded(&mut self, splats: &HostSplats) {
        let key = splats.cache_key();
        if self
            .cached_splats
            .as_ref()
            .map(|cached| cached.key == key)
            .unwrap_or(false)
        {
            return;
        }
        self.cached_splats = Some(CachedSplats {
            key,
            splats: host_splats_to_device::<GsBackendBase>(splats, &self.device),
        });
    }

    fn ensure_texture(&mut self, device: &wgpu::Device, resolution: BurnViewportResolution) {
        if self
            .texture
            .as_ref()
            .map(|texture| texture.resolution == resolution)
            .unwrap_or(false)
        {
            return;
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("burn viewport texture"),
            size: wgpu::Extent3d {
                width: resolution.width,
                height: resolution.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.texture = Some(ViewportTexture {
            resolution,
            texture,
            view,
        });
    }

    fn ensure_converter(&mut self, device: &wgpu::Device) {
        if self.converter.is_none() {
            self.converter = Some(TextureConverter::new(device));
        }
    }
}

impl TextureConverter {
    fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("burn viewport texture converter bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                            TextureConvertUniforms,
                        >() as u64),
                    },
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("burn viewport texture converter shader"),
            source: wgpu::ShaderSource::Wgsl(TEXTURE_CONVERT_WGSL.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("burn viewport texture converter layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("burn viewport texture converter pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("burn viewport texture converter uniforms"),
            contents: bytemuck::bytes_of(&TextureConvertUniforms {
                width: 1,
                height: 1,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            pipeline,
            bind_group_layout,
            uniform_buffer,
        }
    }
}

fn copy_tensor_to_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    converter: &mut TextureConverter,
    tensor: &CubeTensor<WgpuRuntime>,
    texture: &ViewportTexture,
    resolution: BurnViewportResolution,
) -> Result<(), String> {
    let resource = tensor
        .client
        .get_resource(tensor.handle.clone())
        .map_err(|err| format!("failed to access rendered tensor resource: {err}"))?;
    let WgpuResource {
        buffer,
        offset,
        size,
    } = resource.resource();

    let required_size = resolution.width as u64 * resolution.height as u64 * 4 * 4;
    if *size < required_size {
        return Err(format!(
            "rendered tensor resource is too small: {} bytes, expected at least {}",
            size, required_size
        ));
    }

    queue.write_buffer(
        &converter.uniform_buffer,
        0,
        bytemuck::bytes_of(&TextureConvertUniforms {
            width: resolution.width,
            height: resolution.height,
        }),
    );
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("burn viewport texture converter bind group"),
        layout: &converter.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer,
                    offset: *offset,
                    size: wgpu::BufferSize::new(required_size),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&texture.view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: converter.uniform_buffer.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("burn viewport texture converter encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("burn viewport texture converter pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&converter.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            resolution.width.div_ceil(16),
            resolution.height.div_ceil(16),
            1,
        );
    }
    queue.submit(Some(encoder.finish()));
    drop(resource);

    Ok(())
}

const TEXTURE_CONVERT_WGSL: &str = r#"
struct TextureConvertUniforms {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<storage, read> pixels: array<f32>;
@group(0) @binding(1) var out_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> uniforms: TextureConvertUniforms;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) {
        return;
    }
    let index = (gid.y * uniforms.width + gid.x) * 4u;
    let rgba = vec4<f32>(
        clamp(pixels[index], 0.0, 1.0),
        clamp(pixels[index + 1u], 0.0, 1.0),
        clamp(pixels[index + 2u], 0.0, 1.0),
        clamp(pixels[index + 3u], 0.0, 1.0),
    );
    textureStore(out_texture, vec2<i32>(i32(gid.x), i32(gid.y)), rgba);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sh::rgb_to_sh0_value;
    use crate::{Intrinsics, SE3};

    #[test]
    fn burn_viewport_renderer_renders_into_wgpu_texture() {
        let (context, device, queue) = test_shared_wgpu_context();
        let splats = test_splats();
        let camera = GaussianCamera::new(Intrinsics::from_focal(500.0, 32, 32), SE3::identity());
        let resolution = BurnViewportResolution::new(32, 32).expect("valid resolution");
        let mut renderer = BurnViewportRenderer::new(context).expect("viewport renderer");

        let view = renderer
            .render(&device, &queue, &splats, &camera, resolution)
            .expect("viewport render")
            .expect("texture view");

        let _ = view;
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("viewport command submission");
    }

    fn test_shared_wgpu_context() -> (crate::SharedWgpuContext, wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::from_build_config().with_env(),
            backend_options: wgpu::BackendOptions::from_env_or_default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        let adapter = runtime
            .block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
            .expect("wgpu adapter");
        let backend = adapter.get_info().backend;
        let (device, queue) = runtime
            .block_on(
                adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("rustgs viewport test device"),
                    required_features: adapter
                        .features()
                        .difference(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
                    required_limits: adapter.limits(),
                    experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    trace: wgpu::Trace::Off,
                }),
            )
            .expect("wgpu device");
        let context = crate::SharedWgpuContext::from_wgpu_parts(
            instance,
            adapter,
            device.clone(),
            queue.clone(),
            backend,
        );
        (context, device, queue)
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
        .expect("valid test splats")
    }
}
