//! IO module for scene files and checkpoints.
//!
//! - `scene_io`: PLY scene export/import
//! - `colmap_dataset`: COLMAP dataset loading

pub mod colmap_dataset;
pub mod scene_io;

#[cfg(feature = "gpu")]
use crate::core::HostSplats;
use crate::TrainingError;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Simple training checkpoint used by the trainer's incremental save.
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    /// Current iteration
    pub iteration: usize,
    /// Current loss
    pub loss: f32,
    /// Host-side splat artifact captured at the checkpoint boundary.
    pub splats: HostSplats,
}

#[cfg(feature = "gpu")]
impl Default for TrainingCheckpoint {
    fn default() -> Self {
        Self {
            iteration: 0,
            loss: 0.0,
            splats: HostSplats::default(),
        }
    }
}

#[cfg(feature = "gpu")]
impl TrainingCheckpoint {
    /// Create a new checkpoint.
    pub fn new() -> Self {
        Self::default()
    }

    /// Save checkpoint to file.
    pub fn save(&self, path: &Path) -> Result<(), TrainingError> {
        let serialized = serde_json::to_vec_pretty(self).map_err(|err| {
            TrainingError::TrainingFailed(format!("failed to serialize checkpoint: {err}"))
        })?;
        std::fs::write(path, serialized)?;
        Ok(())
    }

    /// Load checkpoint from file.
    pub fn load(path: &Path) -> Result<Self, TrainingError> {
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(|err| {
            TrainingError::TrainingFailed(format!(
                "failed to deserialize checkpoint {}: {err}",
                path.display()
            ))
        })
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests;
