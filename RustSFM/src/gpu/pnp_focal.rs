use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuPnpFocalModel {
    pub(crate) row0: [f32; 4],
    pub(crate) row1: [f32; 4],
    pub(crate) row2: [f32; 4],
    pub(crate) log_focal_and_padding: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuPnpFocalResult {
    pub(crate) selected_model: u32,
    pub(crate) inliers: u32,
    pub(crate) valid: u32,
    pub(crate) pad: u32,
    pub(crate) residual_sum: f32,
    pub(crate) focal: f32,
    pub(crate) pad1: f32,
    pub(crate) pad2: f32,
}

pub(crate) const PNP_FOCAL_SHADER: &str = include_str!("shaders/pnp_focal.wgsl");
