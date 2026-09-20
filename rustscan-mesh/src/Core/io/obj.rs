//! OBJ File Format Support
//!
//! Wavefront OBJ format is one of the most widely supported mesh formats.
//!
//! Format specification:
//! - v x y z          (vertex position)
//! - vn nx ny nz      (vertex normal)
//! - vt u v           (texture coordinate)
//! - f v1 v2 v3       (face with vertex indices)
//! - f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 (face with all attributes)

use crate::RustMesh;
use crate::{Point3, Vec2, Vec3};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// Write mesh to OBJ file
pub fn write_obj(mesh: &RustMesh, path: impl AsRef<Path>) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(writer, "# RustMesh OBJ Export")?;
    writeln!(writer, "# Vertices: {}", mesh.n_vertices())?;
    writeln!(writer, "# Faces: {}", mesh.n_faces())?;
    writeln!(writer)?;

    // Write vertices
    for v_idx in 0..mesh.n_vertices() {
        if let Some(point) = mesh.point_by_index(v_idx) {
            writeln!(writer, "v {} {} {}", point.x, point.y, point.z)?;
        }
    }

    writeln!(writer)?;

    // Write normals if available
    let has_normals = mesh.has_vertex_normals();
    if has_normals {
        for v_idx in 0..mesh.n_vertices() {
            if let Some(normal) = mesh.vertex_normal_by_index(v_idx) {
                writeln!(writer, "vn {} {} {}", normal.x, normal.y, normal.z)?;
            }
        }
        writeln!(writer)?;
    }

    // Write texture coordinates if available
    let has_texcoords = mesh.has_vertex_texcoords();
    if has_texcoords {
        for v_idx in 0..mesh.n_vertices() {
            if let Some(tc) = mesh.vertex_texcoord_by_index(v_idx) {
                writeln!(writer, "vt {} {}", tc.x, tc.y)?;
            }
        }
        writeln!(writer)?;
    }

    // Write faces
    for f_idx in 0..mesh.n_faces() {
        let face_handle = crate::FaceHandle::new(f_idx as u32);
        let vertices = mesh.face_vertices_vec(face_handle);

        if vertices.is_empty() {
            continue;
        }

        write!(writer, "f")?;

        for vh in vertices {
            let v = vh.idx_usize() + 1; // OBJ indices start at 1

            if has_normals && has_texcoords {
                // Format: f v/vt/vn
                write!(writer, " {}/{}/{}", v, v, v)?;
            } else if has_normals {
                // Format: f v//vn
                write!(writer, " {}//{}", v, v)?;
            } else if has_texcoords {
                // Format: f v/vt
                write!(writer, " {}/{}", v, v)?;
            } else {
                // Format: f v
                write!(writer, " {}", v)?;
            }
        }

        writeln!(writer)?;
    }

    writer.flush()?;
    Ok(())
}

/// Read mesh from OBJ file
pub fn read_obj(path: impl AsRef<Path>) -> io::Result<RustMesh> {
    use std::io::{BufRead, BufReader};

    let path = path.as_ref();
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut mesh = RustMesh::new();
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut texcoords = Vec::new();

    // First pass: collect vertices
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "v" if parts.len() >= 4 => {
                let x = parts[1].parse::<f32>().unwrap_or(0.0);
                let y = parts[2].parse::<f32>().unwrap_or(0.0);
                let z = parts[3].parse::<f32>().unwrap_or(0.0);
                positions.push(Point3::new(x, y, z));
            }
            "vn" if parts.len() >= 4 => {
                let nx = parts[1].parse::<f32>().unwrap_or(0.0);
                let ny = parts[2].parse::<f32>().unwrap_or(0.0);
                let nz = parts[3].parse::<f32>().unwrap_or(0.0);
                normals.push(Vec3::new(nx, ny, nz));
            }
            "vt" if parts.len() >= 3 => {
                let u = parts[1].parse::<f32>().unwrap_or(0.0);
                let v = parts[2].parse::<f32>().unwrap_or(0.0);
                texcoords.push(Vec2::new(u, v));
            }
            _ => {}
        }
    }

    // Add vertices to mesh
    for pos in &positions {
        mesh.add_vertex(*pos);
    }

    // Request attributes if available (actual assignment happens in face pass)
    if !normals.is_empty() {
        mesh.request_vertex_normals();
    }

    if !texcoords.is_empty() {
        mesh.request_vertex_texcoords();
    }

    // Second pass: read faces and apply per-face-vertex attributes
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() || parts[0] != "f" {
            continue;
        }

        // Parse face indices - handle v/vt/vn format
        let mut face_verts = Vec::new();

        for i in 1..parts.len() {
            let indices: Vec<&str> = parts[i].split('/').collect();

            // Parse vertex index (required)
            let v_idx = match indices[0].parse::<usize>() {
                Ok(idx) if idx > 0 && idx <= mesh.n_vertices() => idx - 1,
                _ => continue,
            };
            let vh = crate::VertexHandle::new(v_idx as u32);
            face_verts.push(vh);

            // Apply texture coordinate from face definition if available
            if indices.len() > 1 && !indices[1].is_empty() {
                if let Ok(vt_idx) = indices[1].parse::<usize>() {
                    let vt_idx = vt_idx - 1;
                    if vt_idx < texcoords.len() && mesh.has_vertex_texcoords() {
                        mesh.set_vertex_texcoord_by_index(v_idx, texcoords[vt_idx]);
                    }
                }
            }

            // Apply normal from face definition if available
            if indices.len() > 2 && !indices[2].is_empty() {
                if let Ok(vn_idx) = indices[2].parse::<usize>() {
                    let vn_idx = vn_idx - 1;
                    if vn_idx < normals.len() && mesh.has_vertex_normals() {
                        mesh.set_vertex_normal_by_index(v_idx, normals[vn_idx]);
                    }
                }
            }
        }

        if face_verts.len() >= 3 {
            mesh.add_face(&face_verts);
        }
    }

    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obj_roundtrip() {
        use std::fs;

        let mut mesh = RustMesh::new();
        let v0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(Point3::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(Point3::new(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]);

        let path = "/tmp/test_mesh.obj";
        write_obj(&mesh, path).unwrap();

        let loaded = read_obj(path).unwrap();
        assert_eq!(loaded.n_vertices(), 3);
        assert_eq!(loaded.n_faces(), 1);

        fs::remove_file(path).ok();
    }
}
