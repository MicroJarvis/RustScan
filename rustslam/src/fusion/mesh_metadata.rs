//! Mesh metadata export utilities (JSON).

use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;

use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MeshMetadataError {
    #[error("failed to create output directory {path}: {source}")]
    CreateDir {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to write metadata {path}: {source}")]
    Write {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to serialize metadata {path}: {source}")]
    Serialize {
        path: String,
        source: serde_json::Error,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct MeshMetadata {
    pub vertex_count: usize,
    pub triangle_count: usize,
    pub bounding_box: BoundingBox,
    pub isolated_triangle_percentage: f32,
    pub tsdf: TsdfMetadata,
    pub viewpoint_count: usize,
    pub timings_ms: MeshTimings,
}

#[derive(Debug, Clone, Serialize)]
pub struct BoundingBox {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

#[derive(Debug, Clone, Serialize)]
pub struct TsdfMetadata {
    pub voxel_size: f32,
    pub truncation_distance: f32,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct MeshTimings {
    pub tsdf_fusion_ms: u64,
    pub marching_cubes_ms: u64,
    pub post_process_ms: u64,
}

pub fn save_mesh_metadata(path: &Path, metadata: &MeshMetadata) -> Result<(), MeshMetadataError> {
    let file = File::create(path).map_err(|source| MeshMetadataError::Write {
        path: path.display().to_string(),
        source,
    })?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, metadata).map_err(|source| {
        MeshMetadataError::Serialize {
            path: path.display().to_string(),
            source,
        }
    })?;
    Ok(())
}

pub fn export_mesh_metadata(
    output_dir: &Path,
    metadata: &MeshMetadata,
) -> Result<std::path::PathBuf, MeshMetadataError> {
    if !output_dir.exists() {
        fs::create_dir_all(output_dir).map_err(|source| MeshMetadataError::CreateDir {
            path: output_dir.display().to_string(),
            source,
        })?;
    }

    let path = output_dir.join("mesh_metadata.json");
    save_mesh_metadata(&path, metadata)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_export_metadata() {
        let dir = tempdir().unwrap();
        let metadata = MeshMetadata {
            vertex_count: 3,
            triangle_count: 1,
            bounding_box: BoundingBox {
                min: [0.0, 0.0, 0.0],
                max: [1.0, 1.0, 0.0],
            },
            isolated_triangle_percentage: 0.0,
            tsdf: TsdfMetadata {
                voxel_size: 0.01,
                truncation_distance: 0.03,
            },
            viewpoint_count: 10,
            timings_ms: MeshTimings {
                tsdf_fusion_ms: 12,
                marching_cubes_ms: 5,
                post_process_ms: 2,
            },
        };

        let path = export_mesh_metadata(dir.path(), &metadata).unwrap();
        let payload = std::fs::read_to_string(path).unwrap();
        assert!(payload.contains("\"vertex_count\": 3"));
        assert!(payload.contains("\"mesh_metadata\"") == false);
    }
}
