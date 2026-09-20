//! Pipeline checkpointing utilities for cross-stage resume.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

const PIPELINE_CHECKPOINT_VERSION: u32 = 1;

#[derive(Debug, Error)]
pub enum PipelineCheckpointError {
    #[error("failed to create checkpoint directory {path}: {source}")]
    CreateDir {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to read pipeline checkpoint {path}: {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to write pipeline checkpoint {path}: {source}")]
    Write {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to parse pipeline checkpoint {path}: {source}")]
    Parse {
        path: String,
        source: serde_json::Error,
    },
    #[error("failed to serialize pipeline checkpoint {path}: {source}")]
    Serialize {
        path: String,
        source: serde_json::Error,
    },
    #[error("unsupported pipeline checkpoint version {found}")]
    Version { found: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineCheckpoint {
    pub version: u32,
    pub video_completed: bool,
    pub slam_completed: bool,
    pub gaussian_completed: bool,
    pub mesh_completed: bool,
    pub slam: Option<SlamCheckpoint>,
}

impl Default for PipelineCheckpoint {
    fn default() -> Self {
        Self {
            version: PIPELINE_CHECKPOINT_VERSION,
            video_completed: false,
            slam_completed: false,
            gaussian_completed: false,
            mesh_completed: false,
            slam: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlamCheckpoint {
    pub camera: CameraCheckpoint,
    pub frame_count: usize,
    pub keyframes: Vec<KeyframeCheckpoint>,
    #[serde(default)]
    pub map_points: Vec<MapPointCheckpoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapPointCheckpoint {
    pub position: [f32; 3],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<[f32; 3]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraCheckpoint {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyframeCheckpoint {
    pub index: usize,
    pub timestamp: f64,
    pub width: u32,
    pub height: u32,
    pub pose: PoseCheckpoint,
    pub color_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoseCheckpoint {
    pub rotation: [[f32; 3]; 3],
    pub translation: [f32; 3],
}

pub fn checkpoint_dir(output_dir: &Path) -> PathBuf {
    output_dir.join("checkpoints")
}

pub fn slam_frames_dir(output_dir: &Path) -> PathBuf {
    checkpoint_dir(output_dir).join("slam")
}

pub fn pipeline_checkpoint_path(output_dir: &Path) -> PathBuf {
    checkpoint_dir(output_dir).join("pipeline.json")
}

pub fn load_pipeline_checkpoint(
    output_dir: &Path,
) -> Result<Option<PipelineCheckpoint>, PipelineCheckpointError> {
    let path = pipeline_checkpoint_path(output_dir);
    if !path.exists() {
        return Ok(None);
    }

    let file = File::open(&path).map_err(|source| PipelineCheckpointError::Read {
        path: path.display().to_string(),
        source,
    })?;
    let reader = BufReader::new(file);
    let checkpoint: PipelineCheckpoint =
        serde_json::from_reader(reader).map_err(|source| PipelineCheckpointError::Parse {
            path: path.display().to_string(),
            source,
        })?;

    if checkpoint.version != PIPELINE_CHECKPOINT_VERSION {
        return Err(PipelineCheckpointError::Version {
            found: checkpoint.version,
        });
    }

    Ok(Some(checkpoint))
}

pub fn save_pipeline_checkpoint(
    output_dir: &Path,
    checkpoint: &PipelineCheckpoint,
) -> Result<(), PipelineCheckpointError> {
    let dir = checkpoint_dir(output_dir);
    if !dir.exists() {
        fs::create_dir_all(&dir).map_err(|source| PipelineCheckpointError::CreateDir {
            path: dir.display().to_string(),
            source,
        })?;
    }

    let path = pipeline_checkpoint_path(output_dir);
    let tmp_path = path.with_extension("json.tmp");
    let file = File::create(&tmp_path).map_err(|source| PipelineCheckpointError::Write {
        path: path.display().to_string(),
        source,
    })?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, checkpoint).map_err(|source| {
        PipelineCheckpointError::Serialize {
            path: path.display().to_string(),
            source,
        }
    })?;
    writer
        .flush()
        .map_err(|source| PipelineCheckpointError::Write {
            path: path.display().to_string(),
            source,
        })?;
    drop(writer);

    fs::rename(&tmp_path, &path).map_err(|source| PipelineCheckpointError::Write {
        path: path.display().to_string(),
        source,
    })?;

    Ok(())
}
