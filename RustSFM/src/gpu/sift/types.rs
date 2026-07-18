use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub(crate) struct SiftUniforms {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) level: u32,
    pub(crate) levels: u32,
    pub(crate) sigma: f32,
    pub(crate) peak_threshold: f32,
    pub(crate) edge_threshold: f32,
    pub(crate) octave_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub(crate) struct GpuKeypoint {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) sigma: f32,
    pub(crate) response: f32,
    pub(crate) angle: f32,
    pub(crate) octave: i32,
    pub(crate) level: i32,
    pub(crate) valid: u32,
}
