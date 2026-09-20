//! OFF File Format Support
//!
//! Object File Format - simple format for polygonal meshes.

use crate::Point3;
use crate::RustMesh;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Write mesh to OFF file
pub fn write_off(mesh: &RustMesh, path: impl AsRef<Path>) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let mut faces = Vec::new();
    let mut remap = vec![None; mesh.n_vertices()];
    let mut compact_vertices = Vec::new();

    for idx in 0..mesh.n_faces() {
        let vertices = mesh.face_vertices_vec(crate::FaceHandle::new(idx as u32));
        if vertices.len() < 3 {
            continue;
        }

        let mut face = Vec::with_capacity(vertices.len());
        for vh in vertices {
            let original_idx = vh.idx_usize();
            let compact_idx = if let Some(existing) = remap[original_idx] {
                existing
            } else {
                let point = mesh.point_by_index(original_idx).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("missing vertex position at index {original_idx}"),
                    )
                })?;
                let next_idx = compact_vertices.len();
                compact_vertices.push(point);
                remap[original_idx] = Some(next_idx);
                next_idx
            };
            face.push(compact_idx);
        }
        faces.push(face);
    }

    if compact_vertices.is_empty() {
        for idx in 0..mesh.n_vertices() {
            let point = mesh.point_by_index(idx).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("missing vertex position at index {idx}"),
                )
            })?;
            compact_vertices.push(point);
        }
    }

    writeln!(writer, "OFF")?;
    writeln!(writer, "{} {} 0", compact_vertices.len(), faces.len())?;

    for point in compact_vertices {
        writeln!(writer, "{} {} {}", point.x, point.y, point.z)?;
    }

    for face in faces {
        write!(writer, "{}", face.len())?;
        for idx in face {
            write!(writer, " {idx}")?;
        }
        writeln!(writer)?;
    }

    writer.flush()?;
    Ok(())
}

/// Read mesh from OFF file
pub fn read_off(path: impl AsRef<Path>) -> io::Result<RustMesh> {
    read_off_impl(path.as_ref(), OffReadMode::Standard)
}

/// Read OFF through the OpenMesh-style parity face insertion path.
///
/// This is intended for parity/debug workflows and preserves the anchors chosen
/// by `add_face_openmesh_parity()` instead of normalizing them afterwards.
pub fn read_off_openmesh_parity(path: impl AsRef<Path>) -> io::Result<RustMesh> {
    read_off_impl(path.as_ref(), OffReadMode::OpenMeshParity)
}

#[derive(Clone, Copy)]
enum OffReadMode {
    Standard,
    OpenMeshParity,
}

fn read_off_impl(path: &Path, mode: OffReadMode) -> io::Result<RustMesh> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let header = next_data_line(&mut lines)?
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "missing OFF header"))?;
    if header != "OFF" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported OFF header: {header}"),
        ));
    }

    let counts = next_data_line(&mut lines)?
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "missing OFF counts line"))?;
    let count_parts: Vec<_> = counts.split_whitespace().collect();
    if count_parts.len() < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "OFF counts line must contain vertex and face counts",
        ));
    }

    let n_vertices = parse_usize(count_parts[0], "vertex count")?;
    let n_faces = parse_usize(count_parts[1], "face count")?;

    let mut mesh = RustMesh::new();
    for vertex_idx in 0..n_vertices {
        let line = next_data_line(&mut lines)?.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("missing OFF vertex line {vertex_idx}"),
            )
        })?;
        let parts: Vec<_> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("vertex line {vertex_idx} has fewer than 3 coordinates"),
            ));
        }
        let x = parse_f32(parts[0], "vertex x")?;
        let y = parse_f32(parts[1], "vertex y")?;
        let z = parse_f32(parts[2], "vertex z")?;
        mesh.add_vertex(Point3::new(x, y, z));
    }

    for face_idx in 0..n_faces {
        let line = next_data_line(&mut lines)?.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("missing OFF face line {face_idx}"),
            )
        })?;
        let parts: Vec<_> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        let n_face_vertices = parse_usize(parts[0], "face vertex count")?;
        if n_face_vertices < 3 || parts.len() < n_face_vertices + 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("face line {face_idx} is malformed"),
            ));
        }

        let mut face = Vec::with_capacity(n_face_vertices);
        for token in &parts[1..=n_face_vertices] {
            let idx = parse_usize(token, "face vertex index")?;
            if idx >= mesh.n_vertices() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("face index {idx} out of bounds"),
                ));
            }
            face.push(crate::VertexHandle::from_usize(idx));
        }

        let added = match mode {
            OffReadMode::Standard => mesh.add_face(&face),
            OffReadMode::OpenMeshParity => mesh.add_face_openmesh_parity(&face),
        };

        added.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to add face {face_idx} from OFF input"),
            )
        })?;
    }

    if matches!(mode, OffReadMode::Standard) {
        mesh.normalize_boundary_halfedge_handles();
    }

    Ok(mesh)
}

fn next_data_line(
    lines: &mut impl Iterator<Item = io::Result<String>>,
) -> io::Result<Option<String>> {
    for line in lines {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        return Ok(Some(trimmed.to_string()));
    }
    Ok(None)
}

fn parse_usize(token: &str, label: &str) -> io::Result<usize> {
    token.parse::<usize>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("failed to parse {label}: {token}"),
        )
    })
}

fn parse_f32(token: &str, label: &str) -> io::Result<f32> {
    token.parse::<f32>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("failed to parse {label}: {token}"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_off_roundtrip() {
        let mut mesh = RustMesh::new();
        let v0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(Point3::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(Point3::new(1.0, 1.0, 0.0));
        let v3 = mesh.add_vertex(Point3::new(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2, v3]).unwrap();

        let path = "/tmp/test_mesh.off";
        write_off(&mesh, path).unwrap();

        let loaded = read_off(path).unwrap();
        assert_eq!(loaded.n_vertices(), 4);
        assert_eq!(loaded.n_faces(), 1);
        assert_eq!(loaded.face_vertices_vec(crate::FaceHandle::new(0)).len(), 4);

        fs::remove_file(path).ok();
    }
}
