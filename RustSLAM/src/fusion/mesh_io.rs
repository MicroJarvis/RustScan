//! Mesh export utilities (OBJ and PLY).

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use thiserror::Error;

use crate::fusion::marching_cubes::Mesh;

#[derive(Debug, Error)]
pub enum MeshIoError {
    #[error("failed to create output directory {path}: {source}")]
    CreateDir {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to write mesh {path}: {source}")]
    Write {
        path: String,
        source: std::io::Error,
    },
}

pub fn save_mesh_obj(path: &Path, mesh: &Mesh) -> Result<(), MeshIoError> {
    let file = File::create(path).map_err(|source| MeshIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "# rustslam mesh").map_err(|source| MeshIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "# vertices {}", mesh.vertices.len()).map_err(|source| {
        MeshIoError::Write {
            path: path.display().to_string(),
            source,
        }
    })?;
    writeln!(writer, "# triangles {}", mesh.triangles.len()).map_err(|source| {
        MeshIoError::Write {
            path: path.display().to_string(),
            source,
        }
    })?;

    for v in &mesh.vertices {
        let r = (v.color[0].clamp(0.0, 1.0) * 255.0).round() / 255.0;
        let g = (v.color[1].clamp(0.0, 1.0) * 255.0).round() / 255.0;
        let b = (v.color[2].clamp(0.0, 1.0) * 255.0).round() / 255.0;
        writeln!(
            writer,
            "v {} {} {} {} {} {}",
            v.position.x, v.position.y, v.position.z, r, g, b
        )
        .map_err(|source| MeshIoError::Write {
            path: path.display().to_string(),
            source,
        })?;
    }

    for v in &mesh.vertices {
        writeln!(writer, "vn {} {} {}", v.normal.x, v.normal.y, v.normal.z).map_err(|source| {
            MeshIoError::Write {
                path: path.display().to_string(),
                source,
            }
        })?;
    }

    for tri in &mesh.triangles {
        let a = tri.indices[0] + 1;
        let b = tri.indices[1] + 1;
        let c = tri.indices[2] + 1;
        writeln!(writer, "f {a}//{a} {b}//{b} {c}//{c}").map_err(|source| MeshIoError::Write {
            path: path.display().to_string(),
            source,
        })?;
    }

    Ok(())
}

pub fn save_mesh_ply(path: &Path, mesh: &Mesh) -> Result<(), MeshIoError> {
    let file = File::create(path).map_err(|source| MeshIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "ply").map_err(|source| MeshIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "format ascii 1.0").map_err(|source| MeshIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "comment rustslam_mesh").map_err(|source| MeshIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "element vertex {}", mesh.vertices.len()).map_err(|source| {
        MeshIoError::Write {
            path: path.display().to_string(),
            source,
        }
    })?;
    writeln!(writer, "property float x").map_err(|source| MeshIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "property float y").map_err(|source| MeshIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "property float z").map_err(|source| MeshIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "property float nx").map_err(|source| MeshIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "property float ny").map_err(|source| MeshIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "property float nz").map_err(|source| MeshIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "property uchar red").map_err(|source| MeshIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "property uchar green").map_err(|source| MeshIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "property uchar blue").map_err(|source| MeshIoError::Write {
        path: path.display().to_string(),
        source,
    })?;
    writeln!(writer, "element face {}", mesh.triangles.len()).map_err(|source| {
        MeshIoError::Write {
            path: path.display().to_string(),
            source,
        }
    })?;
    writeln!(writer, "property list uchar int vertex_indices").map_err(|source| {
        MeshIoError::Write {
            path: path.display().to_string(),
            source,
        }
    })?;
    writeln!(writer, "end_header").map_err(|source| MeshIoError::Write {
        path: path.display().to_string(),
        source,
    })?;

    for v in &mesh.vertices {
        let r = (v.color[0].clamp(0.0, 1.0) * 255.0).round() as u8;
        let g = (v.color[1].clamp(0.0, 1.0) * 255.0).round() as u8;
        let b = (v.color[2].clamp(0.0, 1.0) * 255.0).round() as u8;
        writeln!(
            writer,
            "{} {} {} {} {} {} {} {} {}",
            v.position.x, v.position.y, v.position.z, v.normal.x, v.normal.y, v.normal.z, r, g, b
        )
        .map_err(|source| MeshIoError::Write {
            path: path.display().to_string(),
            source,
        })?;
    }

    for tri in &mesh.triangles {
        writeln!(
            writer,
            "3 {} {} {}",
            tri.indices[0], tri.indices[1], tri.indices[2]
        )
        .map_err(|source| MeshIoError::Write {
            path: path.display().to_string(),
            source,
        })?;
    }

    Ok(())
}

pub fn export_mesh(
    output_dir: &Path,
    mesh: &Mesh,
) -> Result<(std::path::PathBuf, std::path::PathBuf), MeshIoError> {
    if !output_dir.exists() {
        fs::create_dir_all(output_dir).map_err(|source| MeshIoError::CreateDir {
            path: output_dir.display().to_string(),
            source,
        })?;
    }

    let obj_path = output_dir.join("mesh.obj");
    let ply_path = output_dir.join("mesh.ply");

    save_mesh_obj(&obj_path, mesh)?;
    save_mesh_ply(&ply_path, mesh)?;

    Ok((obj_path, ply_path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::marching_cubes::{Mesh, MeshTriangle, MeshVertex};
    use nalgebra::{Point3, Vector3};
    use tempfile::tempdir;

    #[test]
    fn test_export_mesh_files() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(MeshVertex {
            position: Point3::new(0.0, 0.0, 0.0),
            normal: Vector3::z(),
            color: [1.0, 0.0, 0.0],
        });
        mesh.vertices.push(MeshVertex {
            position: Point3::new(1.0, 0.0, 0.0),
            normal: Vector3::z(),
            color: [0.0, 1.0, 0.0],
        });
        mesh.vertices.push(MeshVertex {
            position: Point3::new(0.0, 1.0, 0.0),
            normal: Vector3::z(),
            color: [0.0, 0.0, 1.0],
        });
        mesh.triangles.push(MeshTriangle { indices: [0, 1, 2] });

        let dir = tempdir().unwrap();
        let (obj_path, ply_path) = export_mesh(dir.path(), &mesh).unwrap();

        let obj = std::fs::read_to_string(obj_path).unwrap();
        let ply = std::fs::read_to_string(ply_path).unwrap();

        assert!(obj.contains("v 0"));
        assert!(obj.contains("vn"));
        assert!(obj.contains("f"));
        assert!(ply.contains("element vertex 3"));
        assert!(ply.contains("element face 1"));
    }
}
