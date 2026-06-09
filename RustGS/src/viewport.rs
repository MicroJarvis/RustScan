//! Realtime viewport backed by RustGS's canonical Burn/CubeCL renderer.

use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData, TensorPrimitive};
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
    depth: Option<ViewportDepthTensor>,
    converter: Option<TextureConverter>,
    raster_cov_blur: f32,
}

#[derive(Debug, Clone)]
pub struct BurnViewportDepth {
    pub resolution: BurnViewportResolution,
    pub values: Vec<f32>,
}

struct ViewportDepthTensor {
    resolution: BurnViewportResolution,
    tensor: Tensor<GsBackendBase, 2>,
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
            depth: None,
            converter: None,
            raster_cov_blur: crate::DEFAULT_RASTER_COV_BLUR,
        })
    }

    pub fn depth_at(&self, x: u32, y: u32) -> Option<f32> {
        let depth = self.read_depth().ok()??;
        let value = *depth.values.get(depth_index(depth.resolution, x, y)?)?;
        (value.is_finite() && value > 0.0).then_some(value)
    }

    pub fn depth_resolution(&self) -> Option<BurnViewportResolution> {
        self.depth.as_ref().map(|depth| depth.resolution)
    }

    pub fn read_depth(&self) -> Result<Option<BurnViewportDepth>, String> {
        let Some(depth) = self.depth.as_ref() else {
            return Ok(None);
        };
        let values = read_depth_values(&self.runtime, &depth.tensor, depth.resolution)?;
        Ok(Some(BurnViewportDepth {
            resolution: depth.resolution,
            values,
        }))
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
            self.depth = None;
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

        self.depth = Some(ViewportDepthTensor {
            resolution,
            tensor: rendered.depth,
        });

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

fn depth_index(resolution: BurnViewportResolution, x: u32, y: u32) -> Option<usize> {
    if x >= resolution.width || y >= resolution.height {
        return None;
    }
    (y as usize)
        .checked_mul(resolution.width as usize)?
        .checked_add(x as usize)
}

fn read_depth_values(
    runtime: &tokio::runtime::Runtime,
    depth: &Tensor<GsBackendBase, 2>,
    resolution: BurnViewportResolution,
) -> Result<Vec<f32>, String> {
    let data = runtime
        .block_on(depth.clone().into_data_async())
        .map_err(|err| format!("viewport depth readback failed: {err}"))?;
    let values = tensor_data_to_f32_vec(data)?;
    let expected = resolution.width as usize * resolution.height as usize;
    if values.len() != expected {
        return Err(format!(
            "viewport depth length {} does not match expected {}",
            values.len(),
            expected
        ));
    }
    Ok(values)
}

fn tensor_data_to_f32_vec(data: TensorData) -> Result<Vec<f32>, String> {
    data.into_vec::<f32>()
        .map_err(|err| format!("viewport depth tensor was not f32: {err}"))
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
