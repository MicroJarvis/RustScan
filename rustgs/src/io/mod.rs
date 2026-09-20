//! IO module for scene files and checkpoints.
//!
//! - `scene_io`: PLY scene export/import
//! - `colmap_dataset`: COLMAP dataset loading

pub mod colmap_dataset;
pub mod scene_io;

use crate::core::HostSplats;
use crate::TrainingError;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Legacy JSON snapshot containing no optimizer or topology state.
///
/// This artifact can recover its splats, but cannot resume training exactly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegacyTrainingCheckpoint {
    /// Current iteration
    pub iteration: usize,
    /// Current loss
    pub loss: f32,
    /// Host-side splat artifact captured at the checkpoint boundary.
    pub splats: HostSplats,
}

impl Default for LegacyTrainingCheckpoint {
    fn default() -> Self {
        Self {
            iteration: 0,
            loss: 0.0,
            splats: HostSplats::default(),
        }
    }
}

impl LegacyTrainingCheckpoint {
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
        load_legacy_training_checkpoint(path)
    }

    /// Recover the splat snapshot from this non-resumable legacy artifact.
    pub fn into_splats(self) -> HostSplats {
        self.splats
    }
}

/// Load the pre-versioned JSON checkpoint format.
pub fn load_legacy_training_checkpoint(
    path: &Path,
) -> Result<LegacyTrainingCheckpoint, TrainingError> {
    let bytes = std::fs::read(path)?;
    serde_json::from_slice(&bytes).map_err(|err| {
        TrainingError::TrainingFailed(format!(
            "failed to deserialize checkpoint {}: {err}",
            path.display()
        ))
    })
}

/// Compatibility alias for the old `rustgs::io::TrainingCheckpoint` path.
#[deprecated(
    note = "use rustgs::LegacyTrainingCheckpoint for JSON snapshots or rustgs::TrainingCheckpoint for resumable checkpoints"
)]
pub type TrainingCheckpoint = LegacyTrainingCheckpoint;
