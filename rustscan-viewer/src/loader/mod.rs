//! File loaders for SLAM results.

pub mod checkpoint;
pub mod colmap;
pub mod gaussian;
pub mod mesh;

pub use checkpoint::LoadError;
pub use colmap::{
    load_colmap_into_scene, load_colmap_training_dataset, map_training_dataset_to_scene,
    ColmapDatasetSummary, LoadedColmapDataset,
};
