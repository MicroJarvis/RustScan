//! Backend type aliases for burn+wgpu

use burn::backend::Autodiff;
use burn::prelude::Backend;
use burn_cubecl::CubeBackend;
use burn_wgpu::WgpuRuntime;

/// Base wgpu compute backend
pub type GsBackendBase = CubeBackend<WgpuRuntime, f32, i32, u32>;

/// Differentiable backend for the custom render pipeline.
///
/// The Phase 4 render kernels currently target the plain cube backend directly,
/// so the autodiff path uses `GsBackendBase` rather than the fusion wrapper.
pub type GsDiffBackend = Autodiff<GsBackendBase>;

/// Device type
pub type GsDevice = <GsDiffBackend as Backend>::Device;
