//! Retired: CPU `SiftDescriptorIndex` matching was removed.
//! SIFT descriptor matching always uses `WgpuSiftMatcher` (`gpu-wgpu`).
use anyhow::{bail, Result};

fn main() -> Result<()> {
    bail!(
        "sift_search_bench retired: CPU SiftDescriptorIndex matching was removed; \
         use WgpuSiftMatcher / `cargo test -p rustscan-sfm gpu_matching --features gpu-wgpu` instead"
    )
}
