#[cfg(feature = "gpu")]
pub(crate) mod frame_loader;
#[cfg(feature = "gpu")]
pub(crate) mod frame_targets;
#[cfg(feature = "gpu")]
pub(crate) mod init_map;

#[cfg(all(test, feature = "gpu"))]
#[path = "init_map/tests.rs"]
mod init_map_tests;
