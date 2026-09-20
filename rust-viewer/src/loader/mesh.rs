//! Loaders for mesh files (OBJ and ASCII PLY).
//!
//! Parses the exact formats produced by RustSLAM's mesh_io module.

use std::collections::HashSet;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::loader::checkpoint::LoadError;
use crate::renderer::scene::{MeshGpuVertex, Scene};

/// Load an OBJ mesh (RustSLAM format: `v x y z r g b`, `vn nx ny nz`, `f A//A B//B C//C`).
/// Returns (vertices, triangle_indices, edge_indices).
///
/// # Example
/// ```no_run
/// use std::path::Path;
/// use rust_viewer::loader::mesh::load_obj;
///
/// let (vertices, indices, edges) = load_obj(Path::new("mesh.obj")).unwrap();
/// println!("Loaded {} vertices", vertices.len());
/// ```
pub fn load_obj(path: &Path) -> Result<(Vec<MeshGpuVertex>, Vec<u32>, Vec<u32>), LoadError> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut colors: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut faces: Vec<[usize; 3]> = Vec::new(); // each: [v_idx, n_idx] × 3, stored flat
    let mut face_normals: Vec<[usize; 3]> = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }

        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("v") => {
                let vals: Result<Vec<f32>, _> = parts.map(|s| s.parse::<f32>()).collect();
                let vals =
                    vals.map_err(|_| LoadError::ObjParse(format!("bad vertex line: {line}")))?;
                if vals.len() >= 3 {
                    positions.push([vals[0], vals[1], vals[2]]);
                    // Support both with and without color
                    let r = vals.get(3).copied().unwrap_or(0.7); // Default gray
                    let g = vals.get(4).copied().unwrap_or(0.7);
                    let b = vals.get(5).copied().unwrap_or(0.7);
                    colors.push([r, g, b]);
                }
            }
            Some("vn") => {
                let vals: Result<Vec<f32>, _> = parts.map(|s| s.parse::<f32>()).collect();
                let vals =
                    vals.map_err(|_| LoadError::ObjParse(format!("bad normal line: {line}")))?;
                if vals.len() >= 3 {
                    normals.push([vals[0], vals[1], vals[2]]);
                }
            }
            Some("f") => {
                // Support multiple formats:
                // - A//A B//B C//C (RustSLAM format: vertex//normal)
                // - A/B/C D/E/F G/H/I (standard format: vertex/texture/normal)
                // - A B C (simple format: vertex only)
                let corners: Vec<&str> = parts.collect();
                if corners.len() >= 3 {
                    let mut vidx = [0usize; 3];
                    let mut nidx = [0usize; 3];
                    for (i, corner) in corners[..3].iter().enumerate() {
                        if corner.contains("//") {
                            // Format: v//vn
                            let mut split = corner.split("//");
                            let v = split
                                .next()
                                .and_then(|s| s.parse::<usize>().ok())
                                .ok_or_else(|| {
                                    LoadError::ObjParse(format!("bad face vertex: {corner}"))
                                })?;
                            let n = split
                                .next()
                                .and_then(|s| s.parse::<usize>().ok())
                                .unwrap_or(1);
                            vidx[i] = v.saturating_sub(1);
                            nidx[i] = n.saturating_sub(1);
                        } else if corner.contains('/') {
                            // Format: v/vt/vn or v/vt
                            let parts_vec: Vec<&str> = corner.split('/').collect();
                            let v = parts_vec
                                .get(0)
                                .and_then(|s| s.parse::<usize>().ok())
                                .ok_or_else(|| {
                                    LoadError::ObjParse(format!("bad face vertex: {corner}"))
                                })?;
                            let n = parts_vec
                                .get(2)
                                .and_then(|s| s.parse::<usize>().ok())
                                .unwrap_or(1);
                            vidx[i] = v.saturating_sub(1);
                            nidx[i] = n.saturating_sub(1);
                        } else {
                            // Format: v (vertex only)
                            let v = corner.parse::<usize>().map_err(|_| {
                                LoadError::ObjParse(format!("bad face vertex: {corner}"))
                            })?;
                            vidx[i] = v.saturating_sub(1);
                            nidx[i] = 0; // Use first normal or default
                        }
                    }
                    faces.push(vidx);
                    face_normals.push(nidx);
                }
            }
            _ => {}
        }
    }

    // Build flat vertex + index buffers (one MeshGpuVertex per face corner)
    // Also track which original vertex each expanded vertex came from
    let mut vertices: Vec<MeshGpuVertex> = Vec::with_capacity(faces.len() * 3);
    let mut indices: Vec<u32> = Vec::with_capacity(faces.len() * 3);
    let mut expanded_to_orig: Vec<usize> = Vec::with_capacity(faces.len() * 3);

    for face in faces.iter() {
        // Get the three vertex positions
        let p0 = *positions.get(face[0]).unwrap_or(&[0.0; 3]);
        let p1 = *positions.get(face[1]).unwrap_or(&[0.0; 3]);
        let p2 = *positions.get(face[2]).unwrap_or(&[0.0; 3]);

        // Compute face normal from vertex positions using cross product
        // This ensures the normal is consistent with the vertex winding order
        let edge1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let edge2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

        // Cross product: edge1 × edge2
        let mut computed_normal = [
            edge1[1] * edge2[2] - edge1[2] * edge2[1],
            edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0],
        ];

        // Normalize the computed normal
        let length = (computed_normal[0] * computed_normal[0]
            + computed_normal[1] * computed_normal[1]
            + computed_normal[2] * computed_normal[2])
            .sqrt();
        if length > 1e-6 {
            computed_normal[0] /= length;
            computed_normal[1] /= length;
            computed_normal[2] /= length;
        } else {
            computed_normal = [0.0, 1.0, 0.0]; // Default up vector
        }

        // Use the computed normal for all three vertices of this face
        for i in 0..3 {
            let vi = face[i];
            let pos = *positions.get(vi).unwrap_or(&[0.0; 3]);
            let col = *colors.get(vi).unwrap_or(&[0.7; 3]);
            indices.push(vertices.len() as u32);
            vertices.push(MeshGpuVertex {
                position: pos,
                normal: computed_normal, // Use computed normal instead of file normal
                color: col,
            });
            expanded_to_orig.push(vi);
        }
    }

    // Extract unique edges based on original vertex indices
    let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
    for face in faces.iter() {
        let i0 = face[0];
        let i1 = face[1];
        let i2 = face[2];
        edge_set.insert((i0.min(i1), i0.max(i1)));
        edge_set.insert((i1.min(i2), i1.max(i2)));
        edge_set.insert((i2.min(i0), i2.max(i0)));
    }

    // Build a map from original vertex index to one expanded vertex index
    let mut orig_to_expanded: std::collections::HashMap<usize, u32> =
        std::collections::HashMap::new();
    for (expanded_idx, &orig_idx) in expanded_to_orig.iter().enumerate() {
        orig_to_expanded
            .entry(orig_idx)
            .or_insert(expanded_idx as u32);
    }

    // Convert unique edges to expanded vertex indices
    let edge_indices: Vec<u32> = edge_set
        .into_iter()
        .filter_map(|(a, b)| {
            let va = orig_to_expanded.get(&a)?;
            let vb = orig_to_expanded.get(&b)?;
            Some([*va, *vb])
        })
        .flatten()
        .collect();

    Ok((vertices, indices, edge_indices))
}

/// Load an ASCII PLY mesh (RustSLAM format).
/// Returns (vertices, triangle_indices, edge_indices).
pub fn load_ply(path: &Path) -> Result<(Vec<MeshGpuVertex>, Vec<u32>, Vec<u32>), LoadError> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let mut vertex_count = 0usize;
    let mut face_count = 0usize;
    let mut in_header = true;

    // Parse header
    while in_header {
        let line = lines
            .next()
            .ok_or_else(|| LoadError::PlyParse("unexpected end of header".into()))??;
        let line = line.trim().to_string();
        if line == "end_header" {
            in_header = false;
        } else if line.starts_with("element vertex") {
            vertex_count = line
                .split_whitespace()
                .nth(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
        } else if line.starts_with("element face") {
            face_count = line
                .split_whitespace()
                .nth(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
        }
    }

    // Read vertices: x y z nx ny nz r g b
    let mut vertices: Vec<MeshGpuVertex> = Vec::with_capacity(vertex_count);
    for _ in 0..vertex_count {
        let line = lines
            .next()
            .ok_or_else(|| LoadError::PlyParse("unexpected end of vertex data".into()))??;
        let vals: Vec<f32> = line
            .split_whitespace()
            .map(|s| s.parse::<f32>().unwrap_or(0.0))
            .collect();
        if vals.len() < 9 {
            return Err(LoadError::PlyParse(format!(
                "vertex line too short: {}",
                line
            )));
        }
        vertices.push(MeshGpuVertex {
            position: [vals[0], vals[1], vals[2]],
            normal: [vals[3], vals[4], vals[5]],
            color: [vals[6] / 255.0, vals[7] / 255.0, vals[8] / 255.0],
        });
    }

    // Read faces: 3 i j k
    let mut indices: Vec<u32> = Vec::with_capacity(face_count * 3);
    let mut edge_set: HashSet<(u32, u32)> = HashSet::new();
    for _ in 0..face_count {
        let line = lines
            .next()
            .ok_or_else(|| LoadError::PlyParse("unexpected end of face data".into()))??;
        let vals: Vec<u32> = line
            .split_whitespace()
            .skip(1) // skip vertex count (always 3)
            .map(|s| s.parse::<u32>().unwrap_or(0))
            .collect();
        if vals.len() < 3 {
            return Err(LoadError::PlyParse(format!(
                "face line too short: {}",
                line
            )));
        }
        indices.extend_from_slice(&vals[..3]);

        // Extract edges for wireframe
        let i0 = vals[0];
        let i1 = vals[1];
        let i2 = vals[2];
        edge_set.insert((i0.min(i1), i0.max(i1)));
        edge_set.insert((i1.min(i2), i1.max(i2)));
        edge_set.insert((i2.min(i0), i2.max(i0)));
    }

    let edge_indices: Vec<u32> = edge_set.into_iter().flat_map(|(a, b)| [a, b]).collect();

    Ok((vertices, indices, edge_indices))
}

/// Load a mesh file (OBJ or PLY) and merge into the scene.
///
/// # Example
/// ```no_run
/// use std::path::Path;
/// use rust_viewer::loader::mesh::load_mesh;
/// use rust_viewer::renderer::scene::Scene;
///
/// let mut scene = Scene::default();
/// load_mesh(Path::new("mesh.obj"), &mut scene).unwrap();
/// println!("Loaded {} vertices", scene.mesh_vertices.len());
/// ```
pub fn load_mesh(path: &Path, scene: &mut Scene) -> Result<(), LoadError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let (verts, idxs, edge_idxs) = match ext.as_str() {
        "obj" => load_obj(path)?,
        "ply" => load_ply(path)?,
        other => {
            return Err(LoadError::UnsupportedFormat(format!(
                "unsupported mesh format: .{other}"
            )));
        }
    };

    for v in &verts {
        scene.bounds.extend(v.position);
    }
    scene.mesh_vertices = verts;
    scene.mesh_indices = idxs;
    scene.mesh_edge_indices = edge_idxs;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_load_obj_triangle() {
        let obj = "# rustslam mesh\n\
                   v 0.0 0.0 0.0 0.8 0.2 0.2\n\
                   v 1.0 0.0 0.0 0.2 0.8 0.2\n\
                   v 0.0 1.0 0.0 0.2 0.2 0.8\n\
                   vn 0.0 0.0 1.0\n\
                   vn 0.0 0.0 1.0\n\
                   vn 0.0 0.0 1.0\n\
                   f 1//1 2//2 3//3\n";

        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        tmpfile.write_all(obj.as_bytes()).unwrap();
        let path = tmpfile.path().to_path_buf();

        let (verts, idxs, edge_idxs) = load_obj(&path).unwrap();
        assert_eq!(verts.len(), 3, "should have 3 vertices");
        assert_eq!(idxs, vec![0, 1, 2], "indices should be 0,1,2");
        assert_eq!(edge_idxs.len(), 6, "should have 3 edges (6 indices)"); // 3 edges = 6 indices
        let c = verts[0].color;
        assert!(
            (c[0] - 0.8).abs() < 1e-3,
            "red component mismatch: {}",
            c[0]
        );
    }

    #[test]
    fn test_load_ply_ascii() {
        let ply = "ply\n\
                   format ascii 1.0\n\
                   comment rustslam_mesh\n\
                   element vertex 3\n\
                   property float x\n\
                   property float y\n\
                   property float z\n\
                   property float nx\n\
                   property float ny\n\
                   property float nz\n\
                   property uchar red\n\
                   property uchar green\n\
                   property uchar blue\n\
                   element face 1\n\
                   property list uchar int vertex_indices\n\
                   end_header\n\
                   0.0 0.0 0.0 0.0 0.0 1.0 204 51 51\n\
                   1.0 0.0 0.0 0.0 0.0 1.0 51 204 51\n\
                   0.0 1.0 0.0 0.0 0.0 1.0 51 51 204\n\
                   3 0 1 2\n";

        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        tmpfile.write_all(ply.as_bytes()).unwrap();
        let path = tmpfile.path().to_path_buf();

        let (verts, idxs, edge_idxs) = load_ply(&path).unwrap();
        assert_eq!(verts.len(), 3);
        assert_eq!(idxs, vec![0, 1, 2]);
        assert_eq!(edge_idxs.len(), 6, "should have 3 edges (6 indices)");
        let c = verts[0].color;
        assert!((c[0] - 204.0 / 255.0).abs() < 1e-3, "red: {}", c[0]);
        assert!((c[1] - 51.0 / 255.0).abs() < 1e-3, "green: {}", c[1]);
    }
}
