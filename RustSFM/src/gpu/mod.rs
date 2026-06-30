//! GPU backends for COLMAP-parity feature extraction and matching.
//!
//! Platform strategy: **wgpu** only (Vulkan / Metal / DX12). CUDA and SiftGPU are
//! intentionally out of scope.

use crate::sift::{SiftExtractionOptions, SiftFeatures};
use anyhow::{bail, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackendKind {
    Wgpu,
}

#[derive(Debug, Clone)]
pub struct GpuSiftCapabilities {
    pub backend: GpuBackendKind,
    pub device_name: String,
}

pub trait GpuSiftExtractor {
    fn capabilities(&self) -> &GpuSiftCapabilities;

    fn extract_sift(
        &self,
        rgb: &[u8],
        width: u32,
        height: u32,
        options: &SiftExtractionOptions,
    ) -> Result<SiftFeatures>;
}

/// Placeholder wgpu backend. Phase 2 will replace this with a real implementation.
#[derive(Debug, Default)]
pub struct WgpuSiftExtractor;

impl WgpuSiftExtractor {
    pub fn try_new() -> Result<Self> {
        bail!("wgpu SIFT backend is not implemented yet; use the default VLFeat CPU extractor")
    }
}

impl GpuSiftExtractor for WgpuSiftExtractor {
    fn capabilities(&self) -> &GpuSiftCapabilities {
        static CAPS: std::sync::OnceLock<GpuSiftCapabilities> = std::sync::OnceLock::new();
        CAPS.get_or_init(|| GpuSiftCapabilities {
            backend: GpuBackendKind::Wgpu,
            device_name: "uninitialized".to_string(),
        })
    }

    fn extract_sift(
        &self,
        _rgb: &[u8],
        _width: u32,
        _height: u32,
        _options: &SiftExtractionOptions,
    ) -> Result<SiftFeatures> {
        bail!("wgpu SIFT extraction is not implemented yet")
    }
}
