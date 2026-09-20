//! 3DGS scene export and import utilities (PLY).
//!
//! Stores Gaussian parameters in an ASCII PLY with metadata comments.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use thiserror::Error;

use crate::fusion::tiled_renderer::Gaussian;

#[derive(Debug, Clone)]
pub struct SceneMetadata {
    pub iterations: usize,
    pub final_loss: f32,
    pub gaussian_count: usize,
}

impl Default for SceneMetadata {
    fn default() -> Self {
        Self {
            iterations: 0,
            final_loss: 0.0,
            gaussian_count: 0,
        }
    }
}

#[derive(Debug, Error)]
pub enum SceneIoError {
    #[error("failed to write scene {path}: {source}")]
    Write {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to read scene {path}: {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },
    #[error("invalid scene format: {message}")]
    InvalidFormat { message: String },
    #[error("parse error: {0}")]
    Parse(String),
}

pub fn save_scene_ply(
    path: &Path,
    gaussians: &[Gaussian],
    metadata: &SceneMetadata,
) -> Result<(), SceneIoError> {
    let file = File::create(path).map_err(|source| SceneIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "ply").map_err(|source| SceneIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "format ascii 1.0").map_err(|source| SceneIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "comment rustslam_scene").map_err(|source| SceneIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "comment iterations {}", metadata.iterations).map_err(|source| {
        SceneIoError::Write {
            path: path.display().to_string(),
            source,
        }
    })?;
    writeln!(writer, "comment final_loss {}", metadata.final_loss).map_err(|source| {
        SceneIoError::Write {
            path: path.display().to_string(),
            source,
        }
    })?;
    writeln!(writer, "comment gaussian_count {}", metadata.gaussian_count).map_err(|source| {
        SceneIoError::Write {
            path: path.display().to_string(),
            source,
        }
    })?;

    writeln!(writer, "element vertex {}", gaussians.len()).map_err(|source| {
        SceneIoError::Write {
            path: path.display().to_string(),
            source,
        }
    })?;
    writeln!(writer, "property float x").map_err(|source| SceneIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "property float y").map_err(|source| SceneIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "property float z").map_err(|source| SceneIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "property float scale_x").map_err(|source| SceneIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "property float scale_y").map_err(|source| SceneIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "property float scale_z").map_err(|source| SceneIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "property float rot_w").map_err(|source| SceneIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "property float rot_x").map_err(|source| SceneIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "property float rot_y").map_err(|source| SceneIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "property float rot_z").map_err(|source| SceneIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "property float opacity").map_err(|source| SceneIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "property float color_r").map_err(|source| SceneIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "property float color_g").map_err(|source| SceneIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "property float color_b").map_err(|source| SceneIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "end_header").map_err(|source| SceneIoError::Write {
        path: path.display().to_string(),
        source,
    })?;

    for g in gaussians {
        writeln!(
            writer,
            "{} {} {} {} {} {} {} {} {} {} {} {} {} {}",
            g.position[0],
            g.position[1],
            g.position[2],
            g.scale[0],
            g.scale[1],
            g.scale[2],
            g.rotation[0],
            g.rotation[1],
            g.rotation[2],
            g.rotation[3],
            g.opacity,
            g.color[0],
            g.color[1],
            g.color[2],
        )
        .map_err(|source| SceneIoError::Write {
            path: path.display().to_string(),
            source,
        })?;
    }

    Ok(())
}

pub fn load_scene_ply(path: &Path) -> Result<(Vec<Gaussian>, SceneMetadata), SceneIoError> {
    let file = File::open(path).map_err(|source| SceneIoError::Read {
        path: path.display().to_string(),
        source,
    })?;
    let reader = BufReader::new(file);

    let mut gaussians = Vec::new();
    let mut metadata = SceneMetadata::default();
    let mut in_header = true;
    let mut expected_vertices: Option<usize> = None;

    for line in reader.lines() {
        let line = line.map_err(|source| SceneIoError::Read {
            path: path.display().to_string(),
            source,
        })?;
        let trimmed = line.trim();

        if in_header {
            if trimmed.starts_with("comment iterations ") {
                metadata.iterations = trimmed["comment iterations ".len()..]
                    .trim()
                    .parse()
                    .unwrap_or(0);
            } else if trimmed.starts_with("comment final_loss ") {
                metadata.final_loss = trimmed["comment final_loss ".len()..]
                    .trim()
                    .parse()
                    .unwrap_or(0.0);
            } else if trimmed.starts_with("comment gaussian_count ") {
                metadata.gaussian_count = trimmed["comment gaussian_count ".len()..]
                    .trim()
                    .parse()
                    .unwrap_or(0);
            } else if trimmed.starts_with("element vertex ") {
                expected_vertices = trimmed["element vertex ".len()..]
                    .trim()
                    .parse::<usize>()
                    .ok();
            } else if trimmed == "end_header" {
                in_header = false;
            }
            continue;
        }

        if trimmed.is_empty() {
            continue;
        }

        let values: Vec<f32> = trimmed
            .split_whitespace()
            .map(|v| v.parse::<f32>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| SceneIoError::Parse(err.to_string()))?;

        if values.len() < 14 {
            return Err(SceneIoError::InvalidFormat {
                message: format!("expected 14 values per vertex, got {}", values.len()),
            });
        }

        let gaussian = Gaussian::new(
            [values[0], values[1], values[2]],
            [values[3], values[4], values[5]],
            [values[6], values[7], values[8], values[9]],
            values[10],
            [values[11], values[12], values[13]],
        );
        gaussians.push(gaussian);
    }

    if let Some(expected) = expected_vertices {
        if gaussians.len() != expected {
            return Err(SceneIoError::InvalidFormat {
                message: format!(
                    "vertex count mismatch: header {}, parsed {}",
                    expected,
                    gaussians.len()
                ),
            });
        }
    }

    if metadata.gaussian_count == 0 {
        metadata.gaussian_count = gaussians.len();
    }

    Ok((gaussians, metadata))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_scene_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("scene.ply");

        let gaussians = vec![
            Gaussian::new(
                [0.0, 0.0, 1.0],
                [0.1, 0.1, 0.1],
                [1.0, 0.0, 0.0, 0.0],
                0.5,
                [0.2, 0.3, 0.4],
            ),
            Gaussian::new(
                [1.0, 0.0, 2.0],
                [0.2, 0.2, 0.2],
                [1.0, 0.0, 0.0, 0.0],
                0.6,
                [0.5, 0.4, 0.3],
            ),
        ];

        let metadata = SceneMetadata {
            iterations: 3000,
            final_loss: 0.42,
            gaussian_count: gaussians.len(),
        };

        save_scene_ply(&path, &gaussians, &metadata).unwrap();

        let (loaded, loaded_meta) = load_scene_ply(&path).unwrap();
        assert_eq!(loaded.len(), gaussians.len());
        assert_eq!(loaded_meta.iterations, 3000);
        assert!((loaded_meta.final_loss - 0.42).abs() < 1e-6);
    }
}
