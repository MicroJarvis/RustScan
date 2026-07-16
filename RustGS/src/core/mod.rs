//! Core shared data structures used by the canonical splat architecture.

mod camera;
pub(crate) mod splats;

pub use camera::{viewmat_from_pose, GaussianCamera};
#[cfg(feature = "gpu")]
pub(crate) use splats::HostSplatsCacheKey;
pub use splats::{HostSplats, SplatView};
