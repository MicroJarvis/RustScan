//! Dense Fusion Module
//!
//! This module provides dense reconstruction utilities that remain local to
//! RustSLAM. Offline 3DGS training now lives in the `rustgs` crate.
//!
//! Architecture:
//! - gaussian.rs: Core 3D Gaussian data structures
//! - renderer.rs: Basic forward rendering
//! - tiled_renderer.rs: Complete tiled rasterization
//! - slam_integrator.rs: Sparse-Dense SLAM integration
//! - tracker.rs: Gaussian-based tracking
//! - mapper.rs: Incremental Gaussian mapping
//! - tsdf_volume.rs: TSDF volume for mesh extraction
//! - marching_cubes.rs: Marching cubes algorithm
//! - mesh_extractor.rs: High-level mesh extraction API

pub mod gaussian;
pub mod mapper;
pub mod marching_cubes;
pub mod mesh_extractor;
pub mod mesh_io;
pub mod mesh_metadata;
pub mod renderer;
pub mod scene_io;
pub mod slam_integrator;
pub mod tiled_renderer;
pub mod tracker;
pub mod tsdf_volume;
pub use gaussian::{Gaussian3D, GaussianCamera, GaussianMap, GaussianState};
pub use mapper::{GaussianMapper, MapperConfig, MapperUpdateResult};
pub use marching_cubes::{MarchingCubes, Mesh, MeshTriangle, MeshVertex};
pub use mesh_extractor::{MeshExtractionConfig, MeshExtractor};
pub use mesh_io::{export_mesh, save_mesh_obj, save_mesh_ply, MeshIoError};
pub use mesh_metadata::{
    export_mesh_metadata, save_mesh_metadata, BoundingBox, MeshMetadata, MeshMetadataError,
    MeshTimings, TsdfMetadata,
};
pub use renderer::{GaussianRenderer, RenderOutput};
pub use scene_io::{load_scene_ply, save_scene_ply, SceneIoError, SceneMetadata};
pub use slam_integrator::{KeyFrame, SlamConfig, SlamIntegrator, SlamOutput, SparseDenseSlam};
pub use tiled_renderer::{
    densify, prune, Gaussian, ProjectedGaussian, RenderBuffer, TiledRenderer,
};
pub use tracker::{GaussianTracker, TrackingResult};
pub use tsdf_volume::{TsdfConfig, TsdfVolume, Voxel};
