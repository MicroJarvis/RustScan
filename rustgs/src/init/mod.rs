//! Gaussian initialization module.
//!
//! Provides utilities to create initial Gaussians from point clouds.

pub mod initialization;

pub use initialization::GaussianInitConfig;

#[cfg(feature = "gpu")]
pub use initialization::initialize_host_splats_from_points;
