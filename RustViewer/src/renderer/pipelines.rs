//! GPU render pipelines using eframe's wgpu re-export.

use eframe::wgpu;
use glam::{Mat2, Mat3, Quat, Vec2, Vec3, Vec4};

use crate::renderer::camera::ArcballCamera;
use crate::renderer::scene::{GaussianSplat, MeshGpuVertex};

/// Point vertex: position + color (24 bytes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PointVertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

/// Static quad vertex used for instanced Gaussian splats.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GaussianQuadVertex {
    pub local: [f32; 2],
}

/// Per-instance data for one projected Gaussian splat.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GaussianInstance {
    pub center: [f32; 2],
    pub axis_u: [f32; 2],
    pub axis_v: [f32; 2],
    pub depth: f32,
    pub color: [f32; 3],
    pub opacity: f32,
}

const POINT_PIXEL_SIZE: f32 = 4.0;
const GAUSSIAN_SUPPORT_RADIUS: f32 = 3.0;

pub const GAUSSIAN_QUAD_VERTICES: [GaussianQuadVertex; 4] = [
    GaussianQuadVertex {
        local: [-GAUSSIAN_SUPPORT_RADIUS, -GAUSSIAN_SUPPORT_RADIUS],
    },
    GaussianQuadVertex {
        local: [GAUSSIAN_SUPPORT_RADIUS, -GAUSSIAN_SUPPORT_RADIUS],
    },
    GaussianQuadVertex {
        local: [GAUSSIAN_SUPPORT_RADIUS, GAUSSIAN_SUPPORT_RADIUS],
    },
    GaussianQuadVertex {
        local: [-GAUSSIAN_SUPPORT_RADIUS, GAUSSIAN_SUPPORT_RADIUS],
    },
];

pub const GAUSSIAN_QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];

/// Expand a list of points into small screen-space quads (2 triangles each).
/// Returns (vertices, indices).
pub fn expand_points_to_quads(
    points: &[[f32; 3]],
    colors: &[[f32; 3]],
    viewport_w: f32,
    viewport_h: f32,
    view_proj: [[f32; 4]; 4],
) -> (Vec<PointVertex>, Vec<u32>) {
    let vp = glam::Mat4::from_cols_array_2d(&view_proj);
    let ndc_w = POINT_PIXEL_SIZE / viewport_w;
    let ndc_h = POINT_PIXEL_SIZE / viewport_h;

    let mut verts: Vec<PointVertex> = Vec::with_capacity(points.len() * 4);
    let mut idxs: Vec<u32> = Vec::with_capacity(points.len() * 6);

    for (pos, col) in points.iter().zip(colors.iter()) {
        let p = Vec4::new(pos[0], pos[1], pos[2], 1.0);
        let clip = vp * p;
        if clip.w <= 0.0 {
            continue;
        }
        let ndcx = clip.x / clip.w;
        let ndcy = clip.y / clip.w;
        let ndcz = clip.z / clip.w;

        if ndcz < -1.0 || ndcz > 1.0 || ndcx < -2.0 || ndcx > 2.0 || ndcy < -2.0 || ndcy > 2.0 {
            continue;
        }

        let corners: [[f32; 3]; 4] = [
            [ndcx - ndc_w, ndcy - ndc_h, ndcz],
            [ndcx + ndc_w, ndcy - ndc_h, ndcz],
            [ndcx + ndc_w, ndcy + ndc_h, ndcz],
            [ndcx - ndc_w, ndcy + ndc_h, ndcz],
        ];

        let base = verts.len() as u32;
        for corner in &corners {
            verts.push(PointVertex {
                position: *corner,
                color: *col,
            });
        }
        idxs.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    (verts, idxs)
}

/// Project 3D Gaussians into screen-space elliptical splats for the current camera.
pub fn project_gaussians_to_splats(
    gaussians: &[GaussianSplat],
    camera: &ArcballCamera,
    viewport_size: [f32; 2],
) -> Vec<GaussianInstance> {
    let width = viewport_size[0].max(1.0);
    let height = viewport_size[1].max(1.0);
    let aspect = width / height;
    let view = camera.view_matrix();
    let proj = camera.proj_matrix(aspect);
    let view_proj = proj * view;
    let min_sigma_ndc = Vec2::new(2.0 / width, 2.0 / height) * 0.75;

    let mut projected = Vec::with_capacity(gaussians.len());

    for gaussian in gaussians {
        if gaussian.opacity <= 0.001 {
            continue;
        }

        let mean = Vec3::from_array(gaussian.position);
        let center_clip = view_proj * mean.extend(1.0);
        if center_clip.w <= 0.0 {
            continue;
        }

        let center_ndc = center_clip.truncate() / center_clip.w;
        if center_ndc.z < -1.0 || center_ndc.z > 1.0 {
            continue;
        }

        let view_pos = view * mean.extend(1.0);
        let view_depth = -view_pos.z;
        if view_depth <= 0.0 {
            continue;
        }

        let quat = decode_rotation(gaussian.rotation);
        let world_rot = Mat3::from_quat(quat);
        let world_basis = [
            world_rot * Vec3::X * gaussian.scale[0].abs().max(1e-4),
            world_rot * Vec3::Y * gaussian.scale[1].abs().max(1e-4),
            world_rot * Vec3::Z * gaussian.scale[2].abs().max(1e-4),
        ];

        let mut cov = Mat2::ZERO;
        for basis in &world_basis {
            if let Some(delta) = project_basis_delta(view_proj, mean, *basis) {
                cov += outer(delta, delta);
            }
        }

        if cov.determinant().abs() < 1e-12 {
            let fallback = min_sigma_ndc.max(Vec2::splat(1e-4));
            cov += Mat2::from_diagonal(fallback * fallback);
        }

        let (eig1, eig2, val1, val2) = decompose_covariance(cov);
        let sigma1 = val1.sqrt().max(min_sigma_ndc.x.max(min_sigma_ndc.y));
        let sigma2 = val2
            .sqrt()
            .max(min_sigma_ndc.x.min(min_sigma_ndc.y).max(1e-4));
        let axis_u = eig1 * sigma1;
        let axis_v = eig2 * sigma2;

        let support =
            axis_u.abs() * GAUSSIAN_SUPPORT_RADIUS + axis_v.abs() * GAUSSIAN_SUPPORT_RADIUS;
        if center_ndc.x + support.x < -1.2
            || center_ndc.x - support.x > 1.2
            || center_ndc.y + support.y < -1.2
            || center_ndc.y - support.y > 1.2
        {
            continue;
        }

        projected.push((
            view_depth,
            GaussianInstance {
                center: [center_ndc.x, center_ndc.y],
                axis_u: axis_u.to_array(),
                axis_v: axis_v.to_array(),
                depth: center_ndc.z,
                color: gaussian.color,
                opacity: gaussian.opacity.clamp(0.0, 1.0),
            },
        ));
    }

    projected.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    projected
        .into_iter()
        .map(|(_, instance)| instance)
        .collect()
}

fn decode_rotation(rotation: [f32; 4]) -> Quat {
    let quat = Quat::from_xyzw(rotation[1], rotation[2], rotation[3], rotation[0]);
    if quat.length_squared() <= 1e-12 {
        Quat::IDENTITY
    } else {
        quat.normalize()
    }
}

fn project_basis_delta(view_proj: glam::Mat4, mean: Vec3, basis: Vec3) -> Option<Vec2> {
    let plus = project_to_ndc(view_proj, mean + basis)?;
    let minus = project_to_ndc(view_proj, mean - basis)?;
    Some((plus - minus) * 0.5)
}

fn project_to_ndc(view_proj: glam::Mat4, point: Vec3) -> Option<Vec2> {
    let clip = view_proj * point.extend(1.0);
    (clip.w > 0.0).then_some(Vec2::new(clip.x / clip.w, clip.y / clip.w))
}

fn outer(a: Vec2, b: Vec2) -> Mat2 {
    Mat2::from_cols(a * b.x, a * b.y)
}

fn decompose_covariance(cov: Mat2) -> (Vec2, Vec2, f32, f32) {
    let xx = cov.x_axis.x;
    let xy = cov.y_axis.x;
    let yy = cov.y_axis.y;
    let trace = xx + yy;
    let det = (xx * yy - xy * xy).max(0.0);
    let disc = (trace * trace * 0.25 - det).max(0.0).sqrt();

    let lambda1 = (trace * 0.5 + disc).max(1e-10);
    let lambda2 = (trace * 0.5 - disc).max(1e-10);

    let eig1 = if xy.abs() > 1e-8 {
        Vec2::new(lambda1 - yy, xy).normalize_or_zero()
    } else if xx >= yy {
        Vec2::X
    } else {
        Vec2::Y
    };
    let eig1 = if eig1.length_squared() <= 1e-12 {
        Vec2::X
    } else {
        eig1
    };
    let eig2 = Vec2::new(-eig1.y, eig1.x);

    (eig1, eig2, lambda1, lambda2)
}

/// Create the wgpu render pipeline for point quads (NDC pre-transformed).
pub fn create_point_pipeline(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("point_shader"),
        source: wgpu::ShaderSource::Wgsl(POINT_WGSL.into()),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("point_pipeline_layout"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("point_pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<PointVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 12,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                ],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Create the wgpu render pipeline for instanced Gaussian splats.
pub fn create_gaussian_pipeline(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gaussian_shader"),
        source: wgpu::ShaderSource::Wgsl(GAUSSIAN_WGSL.into()),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gaussian_pipeline_layout"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("gaussian_pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GaussianQuadVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x2,
                    }],
                },
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GaussianInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: 8,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: 16,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: 24,
                            shader_location: 4,
                            format: wgpu::VertexFormat::Float32,
                        },
                        wgpu::VertexAttribute {
                            offset: 28,
                            shader_location: 5,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 40,
                            shader_location: 6,
                            format: wgpu::VertexFormat::Float32,
                        },
                    ],
                },
            ],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Trajectory vertex layout is the same as PointVertex but uses LineList.
pub fn create_line_pipeline(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("line_shader"),
        source: wgpu::ShaderSource::Wgsl(POINT_WGSL.into()),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("line_pipeline_layout"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("line_pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<PointVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 12,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                ],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Mesh pipeline for solid rendering.
pub fn create_mesh_pipeline(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
    uniform_bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mesh_shader"),
        source: wgpu::ShaderSource::Wgsl(MESH_WGSL.into()),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("mesh_pipeline_layout"),
        bind_group_layouts: &[Some(uniform_bgl)],
        immediate_size: 0,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("mesh_pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<MeshGpuVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 12,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 24,
                        shader_location: 2,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                ],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Wireframe pipeline for mesh edges.
pub fn create_wireframe_pipeline(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
    uniform_bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wireframe_shader"),
        source: wgpu::ShaderSource::Wgsl(WIREFRAME_WGSL.into()),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("wireframe_pipeline_layout"),
        bind_group_layouts: &[Some(uniform_bgl)],
        immediate_size: 0,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("wireframe_pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<MeshGpuVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 12,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 24,
                        shader_location: 2,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                ],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

const POINT_WGSL: &str = r#"
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#;

const GAUSSIAN_WGSL: &str = r#"
struct VertexInput {
    @location(0) local: vec2<f32>,
    @location(1) center: vec2<f32>,
    @location(2) axis_u: vec2<f32>,
    @location(3) axis_v: vec2<f32>,
    @location(4) depth: f32,
    @location(5) color: vec3<f32>,
    @location(6) opacity: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) opacity: f32,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let offset = in.axis_u * in.local.x + in.axis_v * in.local.y;
    out.clip_position = vec4<f32>(in.center + offset, in.depth, 1.0);
    out.local = in.local;
    out.color = in.color;
    out.opacity = in.opacity;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let r2 = dot(in.local, in.local);
    if (r2 > 9.0) {
        discard;
    }

    let alpha = in.opacity * exp(-0.5 * r2);
    if (alpha < 0.002) {
        discard;
    }

    return vec4<f32>(in.color, alpha);
}
"#;

const MESH_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    out.normal = normalize(in.normal);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.5));
    let lambert = max(abs(dot(in.normal, light_dir)), 0.4);
    return vec4<f32>(in.color * (0.4 + 0.6 * lambert), 1.0);
}
"#;

const WIREFRAME_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {
    return uniforms.view_proj * vec4<f32>(in.position, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.2, 0.6, 1.0, 1.0);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::camera::ArcballCamera;
    use crate::renderer::scene::GaussianSplat;

    #[test]
    fn projects_visible_gaussian_to_splat_instance() {
        let camera = ArcballCamera::default();
        let gaussians = vec![GaussianSplat {
            position: [0.0, 0.0, 0.0],
            scale: [0.05, 0.02, 0.01],
            rotation: [1.0, 0.0, 0.0, 0.0],
            opacity: 0.8,
            color: [1.0, 0.5, 0.25],
        }];

        let instances = project_gaussians_to_splats(&gaussians, &camera, [1280.0, 720.0]);
        assert_eq!(instances.len(), 1);

        let instance = instances[0];
        assert!(instance.opacity > 0.0);
        assert!(Vec2::from_array(instance.axis_u).length() > 0.0);
        assert!(Vec2::from_array(instance.axis_v).length() > 0.0);
    }

    #[test]
    fn sorts_farther_gaussian_first_for_alpha_blending() {
        let camera = ArcballCamera::default();
        let gaussians = vec![
            GaussianSplat {
                position: [0.0, 0.0, 0.0],
                scale: [0.05, 0.05, 0.05],
                rotation: [1.0, 0.0, 0.0, 0.0],
                opacity: 0.8,
                color: [1.0, 0.0, 0.0],
            },
            GaussianSplat {
                position: [0.0, 0.0, -2.0],
                scale: [0.05, 0.05, 0.05],
                rotation: [1.0, 0.0, 0.0, 0.0],
                opacity: 0.8,
                color: [0.0, 1.0, 0.0],
            },
        ];

        let instances = project_gaussians_to_splats(&gaussians, &camera, [1280.0, 720.0]);
        assert_eq!(instances.len(), 2);
        assert_eq!(instances[0].color, [0.0, 1.0, 0.0]);
        assert_eq!(instances[1].color, [1.0, 0.0, 0.0]);
    }
}
