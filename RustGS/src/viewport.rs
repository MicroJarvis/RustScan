//! wgpu-native realtime 3DGS viewport renderer.

use crate::HostSplats;
use wgpu::util::DeviceExt;

const MAX_SH_COEFFS: usize = 16;
const SORT_WORKGROUP_SIZE: u32 = 256;
const PROJECTED_SPLAT_STRIDE: u64 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgpuViewportResolution {
    pub width: usize,
    pub height: usize,
}

impl WgpuViewportResolution {
    pub fn new(width: usize, height: usize) -> Option<Self> {
        if width == 0 || height == 0 {
            return None;
        }
        Some(Self { width, height })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WgpuViewportCamera {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub view_proj: [[f32; 4]; 4],
    pub position: [f32; 3],
}

impl WgpuViewportCamera {
    pub fn new(
        view: [[f32; 4]; 4],
        proj: [[f32; 4]; 4],
        view_proj: [[f32; 4]; 4],
        position: [f32; 3],
    ) -> Self {
        Self {
            view,
            proj,
            view_proj,
            position,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSplat {
    position: [f32; 3],
    opacity_logit: f32,
    log_scale: [f32; 3],
    sh_degree: u32,
    rotation: [f32; 4],
    sh_coeffs: [[f32; 4]; MAX_SH_COEFFS],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ProjectUniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    view_proj: [[f32; 4]; 4],
    camera_position: [f32; 4],
    viewport_size: [f32; 2],
    splat_count: u32,
    padded_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SortUniforms {
    len: u32,
    padded_len: u32,
    stage_k: u32,
    stage_j: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SplatQuadVertex {
    local: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawUniforms {
    viewport_size: [f32; 2],
    _pad: [u32; 2],
}

const QUAD_VERTICES: [SplatQuadVertex; 4] = [
    SplatQuadVertex {
        local: [-1.0, -1.0],
    },
    SplatQuadVertex { local: [1.0, -1.0] },
    SplatQuadVertex { local: [1.0, 1.0] },
    SplatQuadVertex { local: [-1.0, 1.0] },
];
const QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];

pub struct WgpuViewportRenderer {
    splat_cache: Option<WgpuSplatCache>,
    renderer: Option<WgpuSplatRenderer>,
}

impl WgpuViewportRenderer {
    pub fn new() -> Self {
        Self {
            splat_cache: None,
            renderer: None,
        }
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        splats: &HostSplats,
        camera: WgpuViewportCamera,
        resolution: WgpuViewportResolution,
    ) -> Option<&wgpu::TextureView> {
        if splats.is_empty() {
            return None;
        }
        self.ensure_splats_uploaded(device, splats);
        let padded_count = self.splat_cache.as_ref()?.padded_count();
        self.ensure_renderer(device, resolution, padded_count);
        let cache = self.splat_cache.as_ref()?;
        let renderer = self.renderer.as_mut()?;
        renderer.render(device, queue, cache, camera, resolution);
        Some(renderer.texture_view())
    }

    fn ensure_splats_uploaded(&mut self, device: &wgpu::Device, splats: &HostSplats) {
        if self
            .splat_cache
            .as_ref()
            .is_some_and(|cache| cache.matches(splats))
        {
            return;
        }

        self.splat_cache = Some(WgpuSplatCache::new(device, splats));
        self.renderer = None;
    }

    fn ensure_renderer(
        &mut self,
        device: &wgpu::Device,
        resolution: WgpuViewportResolution,
        padded_count: u32,
    ) {
        if self
            .renderer
            .as_ref()
            .is_some_and(|renderer| renderer.matches(resolution, padded_count))
        {
            return;
        }
        self.renderer = Some(WgpuSplatRenderer::new(device, resolution, padded_count));
    }
}

impl Default for WgpuViewportRenderer {
    fn default() -> Self {
        Self::new()
    }
}

const PROJECT_WGSL: &str = r#"
const MAX_SH_COEFFS: u32 = 16u;
const SH_C0: f32 = 0.28209479177387814;
const INVALID_KEY: u32 = 0xffffffffu;

struct GpuSplat {
    position: vec3<f32>,
    opacity_logit: f32,
    log_scale: vec3<f32>,
    sh_degree: u32,
    rotation: vec4<f32>,
    sh_coeffs: array<vec4<f32>, 16>,
}

struct ProjectUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    camera_position: vec4<f32>,
    viewport_size: vec2<f32>,
    splat_count: u32,
    padded_count: u32,
}

struct ProjectedSplat {
    center: vec2<f32>,
    axis_u: vec2<f32>,
    axis_v: vec2<f32>,
    color: vec4<f32>,
    clip_z: f32,
    depth: f32,
    valid: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> splats: array<GpuSplat>;
@group(0) @binding(1) var<storage, read_write> projected: array<ProjectedSplat>;
@group(0) @binding(2) var<storage, read_write> sort_keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> sort_values: array<u32>;
@group(0) @binding(4) var<uniform> uniforms: ProjectUniforms;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn quat_to_mat3(q_raw: vec4<f32>) -> mat3x3<f32> {
    var q = q_raw;
    let len2 = dot(q, q);
    if (len2 <= 1e-12) {
        return mat3x3<f32>(
            vec3<f32>(1.0, 0.0, 0.0),
            vec3<f32>(0.0, 1.0, 0.0),
            vec3<f32>(0.0, 0.0, 1.0)
        );
    }
    q *= inverseSqrt(len2);
    let w = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w)),
        vec3<f32>(2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w)),
        vec3<f32>(2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y))
    );
}

fn project_ndc(pos: vec3<f32>) -> vec3<f32> {
    let clip = uniforms.view_proj * vec4<f32>(pos, 1.0);
    return clip.xyz / max(clip.w, 1e-8);
}

fn sh_color(splat: GpuSplat, viewdir: vec3<f32>) -> vec3<f32> {
    var color = SH_C0 * splat.sh_coeffs[0].xyz;
    if (splat.sh_degree >= 1u) {
        let x = viewdir.x;
        let y = viewdir.y;
        let z = viewdir.z;
        color += 0.48860251190292 * (
            -y * splat.sh_coeffs[1].xyz +
             z * splat.sh_coeffs[2].xyz +
            -x * splat.sh_coeffs[3].xyz
        );
        if (splat.sh_degree >= 2u) {
            let z2 = z * z;
            let fTmp0B = -1.092548430592079 * z;
            let fTmp1A = 0.5462742152960395;
            let fC1 = x * x - y * y;
            let fS1 = 2.0 * x * y;
            let pSH6 = 0.9461746957575601 * z2 - 0.3153915652525201;
            let pSH7 = fTmp0B * x;
            let pSH5 = fTmp0B * y;
            let pSH8 = fTmp1A * fC1;
            let pSH4 = fTmp1A * fS1;
            color += pSH4 * splat.sh_coeffs[4].xyz +
                pSH5 * splat.sh_coeffs[5].xyz +
                pSH6 * splat.sh_coeffs[6].xyz +
                pSH7 * splat.sh_coeffs[7].xyz +
                pSH8 * splat.sh_coeffs[8].xyz;
            if (splat.sh_degree >= 3u) {
                let fTmp0C = -2.285228997322329 * z2 + 0.4570457994644658;
                let fTmp1B = 1.445305721320277 * z;
                let fTmp2A = -0.5900435899266435;
                let fC2 = x * fC1 - y * fS1;
                let fS2 = x * fS1 + y * fC1;
                let pSH12 = z * (1.865881662950577 * z2 - 1.119528997770346);
                let pSH13 = fTmp0C * x;
                let pSH11 = fTmp0C * y;
                let pSH14 = fTmp1B * fC1;
                let pSH10 = fTmp1B * fS1;
                let pSH15 = fTmp2A * fC2;
                let pSH9 = fTmp2A * fS2;
                color += pSH9 * splat.sh_coeffs[9].xyz +
                    pSH10 * splat.sh_coeffs[10].xyz +
                    pSH11 * splat.sh_coeffs[11].xyz +
                    pSH12 * splat.sh_coeffs[12].xyz +
                    pSH13 * splat.sh_coeffs[13].xyz +
                    pSH14 * splat.sh_coeffs[14].xyz +
                    pSH15 * splat.sh_coeffs[15].xyz;
            }
        }
    }
    return max(color + vec3<f32>(0.5), vec3<f32>(0.0));
}

fn make_invalid() -> ProjectedSplat {
    return ProjectedSplat(
        vec2<f32>(0.0),
        vec2<f32>(0.0),
        vec2<f32>(0.0),
        vec4<f32>(0.0),
        0.0,
        0.0,
        0u,
        0u
    );
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.padded_count) {
        return;
    }
    sort_values[idx] = idx;

    if (idx >= uniforms.splat_count) {
        projected[idx] = make_invalid();
        sort_keys[idx] = INVALID_KEY;
        return;
    }

    let splat = splats[idx];
    let opacity = sigmoid(splat.opacity_logit);
    let mean = splat.position;
    let clip = uniforms.view_proj * vec4<f32>(mean, 1.0);
    let view_pos = uniforms.view * vec4<f32>(mean, 1.0);
    let depth = -view_pos.z;
    if (clip.w <= 0.0 || depth <= 0.0 || opacity < (1.0 / 255.0)) {
        projected[idx] = make_invalid();
        sort_keys[idx] = INVALID_KEY;
        return;
    }

    let center = clip.xyz / clip.w;
    if (center.z < -1.0 || center.z > 1.0 || center.x < -2.0 || center.x > 2.0 || center.y < -2.0 || center.y > 2.0) {
        projected[idx] = make_invalid();
        sort_keys[idx] = INVALID_KEY;
        return;
    }

    let rot = quat_to_mat3(splat.rotation);
    let scale = exp(splat.log_scale);
    var cov = mat2x2<f32>(vec2<f32>(0.0), vec2<f32>(0.0));
    for (var axis = 0u; axis < 3u; axis += 1u) {
        let basis = rot[axis] * scale[axis];
        let plus = project_ndc(mean + basis);
        let minus = project_ndc(mean - basis);
        let delta = (plus.xy - minus.xy) * 0.5;
        cov += mat2x2<f32>(
            vec2<f32>(delta.x * delta.x, delta.x * delta.y),
            vec2<f32>(delta.x * delta.y, delta.y * delta.y)
        );
    }

    let min_sigma = vec2<f32>(2.0 / max(uniforms.viewport_size.x, 1.0), 2.0 / max(uniforms.viewport_size.y, 1.0)) * 0.75;
    cov[0][0] += min_sigma.x * min_sigma.x;
    cov[1][1] += min_sigma.y * min_sigma.y;

    let xx = cov[0][0];
    let xy = cov[0][1];
    let yy = cov[1][1];
    let trace = xx + yy;
    let det = max(xx * yy - xy * xy, 0.0);
    let disc = sqrt(max(trace * trace * 0.25 - det, 0.0));
    let lambda1 = max(trace * 0.5 + disc, 1e-10);
    let lambda2 = max(trace * 0.5 - disc, 1e-10);
    var eig1 = vec2<f32>(1.0, 0.0);
    if (abs(xy) > 1e-8) {
        eig1 = normalize(vec2<f32>(lambda1 - yy, xy));
    } else if (yy > xx) {
        eig1 = vec2<f32>(0.0, 1.0);
    }
    let eig2 = vec2<f32>(-eig1.y, eig1.x);
    let sigma1 = sqrt(lambda1);
    let sigma2 = sqrt(lambda2);
    let support = (abs(eig1 * sigma1) + abs(eig2 * sigma2)) * 3.0;
    if (center.x + support.x < -1.2 || center.x - support.x > 1.2 || center.y + support.y < -1.2 || center.y - support.y > 1.2) {
        projected[idx] = make_invalid();
        sort_keys[idx] = INVALID_KEY;
        return;
    }

    let dir = mean - uniforms.camera_position.xyz;
    let viewdir = dir * inverseSqrt(max(dot(dir, dir), 1e-12));
    projected[idx] = ProjectedSplat(
        center.xy,
        eig1 * sigma1 * 3.0,
        eig2 * sigma2 * 3.0,
        vec4<f32>(sh_color(splat, viewdir), opacity),
        center.z,
        depth,
        1u,
        0u
    );

    let normalized_depth = clamp(depth / 1000.0, 0.0, 1.0);
    sort_keys[idx] = u32((1.0 - normalized_depth) * 4294967294.0);
}
"#;

const SORT_WGSL: &str = r#"
struct SortUniforms {
    len: u32,
    padded_len: u32,
    stage_k: u32,
    stage_j: u32,
}

@group(0) @binding(0) var<storage, read_write> keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> values: array<u32>;
@group(0) @binding(2) var<uniform> uniforms: SortUniforms;

fn should_swap(lhs_key: u32, lhs_value: u32, rhs_key: u32, rhs_value: u32) -> bool {
    return (lhs_key > rhs_key) || ((lhs_key == rhs_key) && (lhs_value > rhs_value));
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.padded_len) {
        return;
    }
    let partner = idx ^ uniforms.stage_j;
    if (partner <= idx || partner >= uniforms.padded_len) {
        return;
    }

    let ascending = (idx & uniforms.stage_k) == 0u;
    let lhs_key = keys[idx];
    let rhs_key = keys[partner];
    let lhs_value = values[idx];
    let rhs_value = values[partner];
    let swap_ascending = should_swap(lhs_key, lhs_value, rhs_key, rhs_value);
    let swap = select(!swap_ascending, swap_ascending, ascending);
    if (swap) {
        keys[idx] = rhs_key;
        values[idx] = rhs_value;
        keys[partner] = lhs_key;
        values[partner] = lhs_value;
    }
}
"#;

const DRAW_WGSL: &str = r#"
struct ProjectedSplat {
    center: vec2<f32>,
    axis_u: vec2<f32>,
    axis_v: vec2<f32>,
    color: vec4<f32>,
    clip_z: f32,
    depth: f32,
    valid: u32,
    pad: u32,
}

struct DrawUniforms {
    viewport_size: vec2<f32>,
    pad: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> projected: array<ProjectedSplat>;
@group(0) @binding(1) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(2) var<uniform> uniforms: DrawUniforms;

struct VertexInput {
    @location(0) local: vec2<f32>,
    @builtin(instance_index) instance_index: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) color: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let splat_index = sorted_indices[in.instance_index];
    let splat = projected[splat_index];
    var out: VertexOutput;
    if (splat.valid == 0u) {
        out.clip_position = vec4<f32>(2.0, 2.0, 0.0, 1.0);
        out.local = vec2<f32>(4.0, 4.0);
        out.color = vec4<f32>(0.0);
        return out;
    }
    let offset = splat.axis_u * in.local.x + splat.axis_v * in.local.y;
    out.clip_position = vec4<f32>(splat.center + offset, splat.clip_z, 1.0);
    out.local = in.local * 3.0;
    out.color = splat.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let r2 = dot(in.local, in.local);
    if (r2 > 9.0) {
        discard;
    }
    let alpha = in.color.a * exp(-0.5 * r2);
    if (alpha < 0.002) {
        discard;
    }
    return vec4<f32>(in.color.rgb, alpha);
}
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SplatCacheKey {
    len: usize,
    sh_degree: usize,
    positions: usize,
    log_scales: usize,
    rotations: usize,
    opacity_logits: usize,
    sh_coeffs: usize,
}

impl SplatCacheKey {
    fn from_splats(splats: &HostSplats) -> Self {
        let view = splats.as_view();
        Self {
            len: splats.len(),
            sh_degree: splats.sh_degree(),
            positions: view.positions.as_ptr() as usize,
            log_scales: view.log_scales.as_ptr() as usize,
            rotations: view.rotations.as_ptr() as usize,
            opacity_logits: view.opacity_logits.as_ptr() as usize,
            sh_coeffs: view.sh_coeffs.as_ptr() as usize,
        }
    }
}

struct WgpuSplatCache {
    key: SplatCacheKey,
    splat_buffer: wgpu::Buffer,
    splat_count: u32,
    padded_count: u32,
}

impl WgpuSplatCache {
    fn new(device: &wgpu::Device, splats: &HostSplats) -> Self {
        let key = SplatCacheKey::from_splats(splats);
        let view = splats.as_view();
        let row_width = ((splats.sh_degree() + 1) * (splats.sh_degree() + 1)) * 3;
        let mut packed = Vec::with_capacity(splats.len());
        for idx in 0..splats.len() {
            let position = splats.position(idx);
            let log_scale = splats.log_scale(idx);
            let rotation = splats.rotation(idx);
            let opacity_logit = splats.opacity_logit(idx);
            let mut sh_coeffs = [[0.0f32; 4]; MAX_SH_COEFFS];
            let base = idx * row_width;
            let coeff_count =
                ((splats.sh_degree() + 1) * (splats.sh_degree() + 1)).min(MAX_SH_COEFFS);
            for coeff in 0..coeff_count {
                let src = base + coeff * 3;
                sh_coeffs[coeff] = [
                    view.sh_coeffs[src],
                    view.sh_coeffs[src + 1],
                    view.sh_coeffs[src + 2],
                    0.0,
                ];
            }
            packed.push(GpuSplat {
                position,
                opacity_logit,
                log_scale,
                sh_degree: splats.sh_degree().min(3) as u32,
                rotation,
                sh_coeffs,
            });
        }
        let splat_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("3dgs viewer splat buffer"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let padded_count = splats
            .len()
            .max(1)
            .next_power_of_two()
            .min(u32::MAX as usize) as u32;
        Self {
            key,
            splat_buffer,
            splat_count: splats.len().min(u32::MAX as usize) as u32,
            padded_count,
        }
    }

    fn matches(&self, splats: &HostSplats) -> bool {
        self.key == SplatCacheKey::from_splats(splats)
    }

    fn padded_count(&self) -> u32 {
        self.padded_count
    }
}

struct WgpuSplatRenderer {
    resolution: WgpuViewportResolution,
    padded_count: u32,
    project_pipeline: wgpu::ComputePipeline,
    sort_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    project_bgl: wgpu::BindGroupLayout,
    draw_bgl: wgpu::BindGroupLayout,
    projected_buffer: wgpu::Buffer,
    sort_keys: wgpu::Buffer,
    sort_values: wgpu::Buffer,
    sort_passes: Vec<SortPass>,
    project_uniform: wgpu::Buffer,
    draw_uniform: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    _texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
}

struct SortPass {
    _uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    dispatch_count: u32,
}

impl WgpuSplatRenderer {
    fn new(device: &wgpu::Device, resolution: WgpuViewportResolution, padded_count: u32) -> Self {
        let project_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("3dgs project bgl"),
            entries: &[
                storage_entry(0, true, wgpu::ShaderStages::COMPUTE),
                storage_entry(1, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(2, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(3, false, wgpu::ShaderStages::COMPUTE),
                uniform_entry(4, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let sort_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("3dgs sort bgl"),
            entries: &[
                storage_entry(0, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(1, false, wgpu::ShaderStages::COMPUTE),
                uniform_entry(2, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let draw_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("3dgs draw bgl"),
            entries: &[
                storage_entry(0, true, wgpu::ShaderStages::VERTEX),
                storage_entry(1, true, wgpu::ShaderStages::VERTEX),
                uniform_entry(2, wgpu::ShaderStages::VERTEX),
            ],
        });

        let project_pipeline =
            create_compute_pipeline(device, "3dgs project", PROJECT_WGSL, &project_bgl);
        let sort_pipeline = create_compute_pipeline(device, "3dgs sort", SORT_WGSL, &sort_bgl);
        let render_pipeline = create_render_pipeline(device, &draw_bgl);

        let count = padded_count.max(1) as u64;
        let projected_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("3dgs projected splat buffer"),
            size: PROJECTED_SPLAT_STRIDE * count,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let sort_keys = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("3dgs sort keys"),
            size: 4 * count,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let sort_values = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("3dgs sort values"),
            size: 4 * count,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let project_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("3dgs project uniforms"),
            size: std::mem::size_of::<ProjectUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let draw_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("3dgs draw uniforms"),
            contents: bytemuck::bytes_of(&DrawUniforms {
                viewport_size: [resolution.width as f32, resolution.height as f32],
                _pad: [0, 0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("3dgs quad vertices"),
            contents: bytemuck::cast_slice(&QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("3dgs quad indices"),
            contents: bytemuck::cast_slice(&QUAD_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("3dgs viewport texture"),
            size: wgpu::Extent3d {
                width: resolution.width as u32,
                height: resolution.height as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sort_passes =
            create_sort_passes(device, &sort_bgl, &sort_keys, &sort_values, padded_count);

        Self {
            resolution,
            padded_count,
            project_pipeline,
            sort_pipeline,
            render_pipeline,
            project_bgl,
            draw_bgl,
            projected_buffer,
            sort_keys,
            sort_values,
            sort_passes,
            project_uniform,
            draw_uniform,
            vertex_buffer,
            index_buffer,
            _texture: texture,
            texture_view,
        }
    }

    fn matches(&self, resolution: WgpuViewportResolution, padded_count: u32) -> bool {
        self.resolution == resolution && self.padded_count == padded_count
    }

    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cache: &WgpuSplatCache,
        camera: WgpuViewportCamera,
        resolution: WgpuViewportResolution,
    ) {
        let uniforms = ProjectUniforms {
            view: camera.view,
            proj: camera.proj,
            view_proj: camera.view_proj,
            camera_position: [
                camera.position[0],
                camera.position[1],
                camera.position[2],
                0.0,
            ],
            viewport_size: [resolution.width as f32, resolution.height as f32],
            splat_count: cache.splat_count,
            padded_count: cache.padded_count,
        };
        queue.write_buffer(&self.project_uniform, 0, bytemuck::bytes_of(&uniforms));
        queue.write_buffer(
            &self.draw_uniform,
            0,
            bytemuck::bytes_of(&DrawUniforms {
                viewport_size: [resolution.width as f32, resolution.height as f32],
                _pad: [0, 0],
            }),
        );

        let project_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("3dgs project bind group"),
            layout: &self.project_bgl,
            entries: &[
                bind_buffer(0, &cache.splat_buffer),
                bind_buffer(1, &self.projected_buffer),
                bind_buffer(2, &self.sort_keys),
                bind_buffer(3, &self.sort_values),
                bind_buffer(4, &self.project_uniform),
            ],
        });
        let draw_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("3dgs draw bind group"),
            layout: &self.draw_bgl,
            entries: &[
                bind_buffer(0, &self.projected_buffer),
                bind_buffer(1, &self.sort_values),
                bind_buffer(2, &self.draw_uniform),
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("3dgs viewport encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("3dgs project pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.project_pipeline);
            pass.set_bind_group(0, &project_bg, &[]);
            pass.dispatch_workgroups(cache.padded_count.div_ceil(256), 1, 1);
        }

        for sort_pass in &self.sort_passes {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("3dgs sort pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sort_pipeline);
            pass.set_bind_group(0, &sort_pass.bind_group, &[]);
            pass.dispatch_workgroups(sort_pass.dispatch_count, 1, 1);
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("3dgs viewport render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.texture_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &draw_bg, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..QUAD_INDICES.len() as u32, 0, 0..cache.padded_count);
        }

        queue.submit([encoder.finish()]);
    }

    fn texture_view(&self) -> &wgpu::TextureView {
        &self.texture_view
    }
}

fn create_sort_passes(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sort_keys: &wgpu::Buffer,
    sort_values: &wgpu::Buffer,
    padded_count: u32,
) -> Vec<SortPass> {
    let mut passes = Vec::new();
    let mut k = 2u32;
    while k <= padded_count {
        let mut j = k / 2;
        while j > 0 {
            let params = SortUniforms {
                len: padded_count,
                padded_len: padded_count,
                stage_k: k,
                stage_j: j,
            };
            let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("3dgs sort uniforms"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("3dgs sort bind group"),
                layout,
                entries: &[
                    bind_buffer(0, sort_keys),
                    bind_buffer(1, sort_values),
                    bind_buffer(2, &uniform),
                ],
            });
            passes.push(SortPass {
                _uniform: uniform,
                bind_group,
                dispatch_count: padded_count.div_ceil(SORT_WORKGROUP_SIZE),
            });
            j >>= 1;
        }
        k <<= 1;
    }
    passes
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(bgl)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn create_render_pipeline(
    device: &wgpu::Device,
    draw_bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("3dgs draw shader"),
        source: wgpu::ShaderSource::Wgsl(DRAW_WGSL.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("3dgs draw pipeline layout"),
        bind_group_layouts: &[Some(draw_bgl)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("3dgs draw pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<SplatQuadVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                }],
            }],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
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

fn storage_entry(
    binding: u32,
    read_only: bool,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bind_buffer(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_splat_storage_layout_matches_wgsl() {
        assert_eq!(std::mem::size_of::<GpuSplat>(), 304);
        assert_eq!(std::mem::align_of::<GpuSplat>(), 4);
        assert_eq!(
            std::mem::size_of::<ProjectUniforms>() % wgpu::COPY_BUFFER_ALIGNMENT as usize,
            0
        );
        assert_eq!(PROJECTED_SPLAT_STRIDE, 64);
    }
}
