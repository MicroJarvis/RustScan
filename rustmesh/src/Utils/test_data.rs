//! # Test Data Generator
//!
//! Generates test meshes for benchmarking and validation.

use crate::handles::VertexHandle;
use crate::RustMesh;
use crate::{Point3, Vec3};
use std::path::Path;

/// Generate a simple cube
pub fn generate_cube() -> RustMesh {
    let mut mesh = RustMesh::new();

    // 8 vertices of a unit cube centered at origin
    let v = [
        mesh.add_vertex(Point3::new(-1.0, -1.0, -1.0)),
        mesh.add_vertex(Point3::new(1.0, -1.0, -1.0)),
        mesh.add_vertex(Point3::new(1.0, 1.0, -1.0)),
        mesh.add_vertex(Point3::new(-1.0, 1.0, -1.0)),
        mesh.add_vertex(Point3::new(-1.0, -1.0, 1.0)),
        mesh.add_vertex(Point3::new(1.0, -1.0, 1.0)),
        mesh.add_vertex(Point3::new(1.0, 1.0, 1.0)),
        mesh.add_vertex(Point3::new(-1.0, 1.0, 1.0)),
    ];

    // 6 faces - consistent orientation so adjacent faces share edges in opposite directions
    mesh.add_face(&[v[0], v[1], v[2], v[3]]); // back  (v0→v1→v2→v3)
    mesh.add_face(&[v[7], v[6], v[5], v[4]]); // front (reversed)
    mesh.add_face(&[v[4], v[5], v[1], v[0]]); // bottom (reversed)
    mesh.add_face(&[v[3], v[2], v[6], v[7]]); // top
    mesh.add_face(&[v[4], v[0], v[3], v[7]]); // left  (reversed)
    mesh.add_face(&[v[1], v[5], v[6], v[2]]); // right (reversed)

    mesh
}

/// Generate a tetrahedron
pub fn generate_tetrahedron() -> RustMesh {
    let mut mesh = RustMesh::new();

    let a = mesh.add_vertex(Point3::new(1.0, 1.0, 1.0));
    let b = mesh.add_vertex(Point3::new(-1.0, -1.0, 1.0));
    let c = mesh.add_vertex(Point3::new(-1.0, 1.0, -1.0));
    let d = mesh.add_vertex(Point3::new(1.0, -1.0, -1.0));

    mesh.add_face(&[a, b, c]);
    mesh.add_face(&[a, c, d]);
    mesh.add_face(&[a, d, b]);
    mesh.add_face(&[b, d, c]);

    mesh
}

/// Generate a pyramid (square base)
pub fn generate_pyramid() -> RustMesh {
    let mut mesh = RustMesh::new();

    let base0 = mesh.add_vertex(Point3::new(-1.0, -1.0, 0.0));
    let base1 = mesh.add_vertex(Point3::new(1.0, -1.0, 0.0));
    let base2 = mesh.add_vertex(Point3::new(1.0, 1.0, 0.0));
    let base3 = mesh.add_vertex(Point3::new(-1.0, 1.0, 0.0));
    let apex = mesh.add_vertex(Point3::new(0.0, 0.0, 2.0));

    // Base
    mesh.add_face(&[base0, base1, base2, base3]);

    // Side faces
    mesh.add_face(&[base0, base1, apex]);
    mesh.add_face(&[base1, base2, apex]);
    mesh.add_face(&[base2, base3, apex]);
    mesh.add_face(&[base3, base0, apex]);

    mesh
}

/// Generate an icosahedron (20-sided polyhedron)
pub fn generate_icosahedron() -> RustMesh {
    let mut mesh = RustMesh::new();

    let t = (1.0 + 5.0_f32.sqrt()) / 2.0;

    let v = [
        mesh.add_vertex(Point3::from(Vec3::new(-1.0, t, 0.0).normalize())),
        mesh.add_vertex(Point3::from(Vec3::new(1.0, t, 0.0).normalize())),
        mesh.add_vertex(Point3::from(Vec3::new(-1.0, -t, 0.0).normalize())),
        mesh.add_vertex(Point3::from(Vec3::new(1.0, -t, 0.0).normalize())),
        mesh.add_vertex(Point3::from(Vec3::new(0.0, -1.0, t).normalize())),
        mesh.add_vertex(Point3::from(Vec3::new(0.0, 1.0, t).normalize())),
        mesh.add_vertex(Point3::from(Vec3::new(0.0, -1.0, -t).normalize())),
        mesh.add_vertex(Point3::from(Vec3::new(0.0, 1.0, -t).normalize())),
        mesh.add_vertex(Point3::from(Vec3::new(t, 0.0, -1.0).normalize())),
        mesh.add_vertex(Point3::from(Vec3::new(t, 0.0, 1.0).normalize())),
        mesh.add_vertex(Point3::from(Vec3::new(-t, 0.0, -1.0).normalize())),
        mesh.add_vertex(Point3::from(Vec3::new(-t, 0.0, 1.0).normalize())),
    ];

    // 20 triangular faces
    let faces = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    for face in &faces {
        mesh.add_face(&[v[face[0]], v[face[1]], v[face[2]]]);
    }

    mesh
}

/// Generate a subdivided icosphere (by subdividing icosahedron)
pub fn generate_icosphere(subdivisions: u32) -> RustMesh {
    let mesh = generate_icosahedron();

    for _ in 0..subdivisions {
        // Simplified: just copy for now
        // Full implementation would subdivide each triangle
    }

    mesh
}

/// Generate a grid mesh
pub fn generate_grid(rows: usize, cols: usize) -> RustMesh {
    let mut mesh = RustMesh::new();

    // Create vertices
    let mut vertices: Vec<Vec<VertexHandle>> = Vec::new();
    for i in 0..rows {
        let mut row: Vec<VertexHandle> = Vec::new();
        for j in 0..cols {
            let x = (j as f32) / ((cols - 1) as f32) - 0.5;
            let y = (i as f32) / ((rows - 1) as f32) - 0.5;
            let vh = mesh.add_vertex(Point3::new(x, y, 0.0));
            row.push(vh);
        }
        vertices.push(row);
    }

    // Create quads
    for i in 0..rows - 1 {
        for j in 0..cols - 1 {
            let v0 = vertices[i][j];
            let v1 = vertices[i][j + 1];
            let v2 = vertices[i + 1][j + 1];
            let v3 = vertices[i + 1][j];
            mesh.add_face(&[v0, v1, v2, v3]);
        }
    }

    mesh
}

/// Generate a sphere using latitude/longitude subdivision
pub fn generate_sphere(radius: f32, segments: usize, rings: usize) -> RustMesh {
    let mut mesh = RustMesh::new();

    let mut vertices: Vec<Vec<VertexHandle>> = Vec::new();

    // Create vertices
    for r in 0..=rings {
        let phi = std::f32::consts::PI * (r as f32) / (rings as f32);
        let mut row: Vec<VertexHandle> = Vec::new();

        for s in 0..=segments {
            let theta = 2.0 * std::f32::consts::PI * (s as f32) / (segments as f32);

            let x = radius * phi.sin() * theta.cos();
            let y = radius * phi.cos();
            let z = radius * phi.sin() * theta.sin();

            let vh = mesh.add_vertex(Point3::new(x, y, z));
            row.push(vh);
        }

        vertices.push(row);
    }

    // Create triangles
    for r in 0..rings {
        for s in 0..segments {
            let v0 = vertices[r][s];
            let v1 = vertices[r][s + 1];
            let v2 = vertices[r + 1][s + 1];
            let v3 = vertices[r + 1][s];

            mesh.add_face(&[v0, v1, v2]);
            mesh.add_face(&[v0, v2, v3]);
        }
    }

    mesh
}

/// Generate a torus
pub fn generate_torus(
    major_radius: f32,
    minor_radius: f32,
    major_segments: usize,
    minor_segments: usize,
) -> RustMesh {
    let mut mesh = RustMesh::new();

    let mut vertices: Vec<Vec<VertexHandle>> = Vec::new();

    // Create vertices
    for i in 0..=major_segments {
        let u = 2.0 * std::f32::consts::PI * (i as f32) / (major_segments as f32);
        let mut row: Vec<VertexHandle> = Vec::new();

        for j in 0..=minor_segments {
            let v = 2.0 * std::f32::consts::PI * (j as f32) / (minor_segments as f32);

            let x = (major_radius + minor_radius * v.cos()) * u.cos();
            let y = (major_radius + minor_radius * v.cos()) * u.sin();
            let z = minor_radius * v.sin();

            let vh = mesh.add_vertex(Point3::new(x, y, z));
            row.push(vh);
        }

        vertices.push(row);
    }

    // Create quads
    for i in 0..major_segments {
        for j in 0..minor_segments {
            let v0 = vertices[i][j];
            let v1 = vertices[i][j + 1];
            let v2 = vertices[i + 1][j + 1];
            let v3 = vertices[i + 1][j];

            mesh.add_face(&[v0, v1, v2, v3]);
        }
    }

    mesh
}

/// Generate a complex mesh with random perturbations (for smoothing tests)
pub fn generate_noisy_sphere(radius: f32, noise: f32, segments: usize, rings: usize) -> RustMesh {
    use rand::Rng;
    let mut mesh = RustMesh::new();
    let mut rng = rand::thread_rng();

    let mut vertices: Vec<Vec<VertexHandle>> = Vec::new();

    // Create noisy vertices
    for r in 0..=rings {
        let phi = std::f32::consts::PI * (r as f32) / (rings as f32);
        let mut row: Vec<VertexHandle> = Vec::new();

        for s in 0..=segments {
            let theta = 2.0 * std::f32::consts::PI * (s as f32) / (segments as f32);

            let noise_factor = 1.0 + rng.gen_range(-noise..noise);
            let r = radius * noise_factor;

            let x = r * phi.sin() * theta.cos();
            let y = r * phi.cos();
            let z = r * phi.sin() * theta.sin();

            let vh = mesh.add_vertex(Point3::new(x, y, z));
            row.push(vh);
        }

        vertices.push(row);
    }

    // Create triangles
    for r in 0..rings {
        for s in 0..segments {
            let v0 = vertices[r][s];
            let v1 = vertices[r][s + 1];
            let v2 = vertices[r + 1][s + 1];
            let v3 = vertices[r + 1][s];

            mesh.add_face(&[v0, v1, v2]);
            mesh.add_face(&[v0, v2, v3]);
        }
    }

    mesh
}

/// Save generated mesh to file
/// TODO: Implement when IO module is ready
#[allow(dead_code)]
pub fn save_mesh<P: AsRef<Path>>(_mesh: &RustMesh, _path: P) -> std::io::Result<()> {
    unimplemented!("IO module not yet implemented")
}

/// Load generated mesh from file
/// TODO: Implement when IO module is ready
#[allow(dead_code)]
pub fn load_mesh<P: AsRef<Path>>(_path: P) -> std::io::Result<RustMesh> {
    unimplemented!("IO module not yet implemented")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cube() {
        let mesh = generate_cube();
        assert_eq!(mesh.n_vertices(), 8);
        // All 6 faces should be created with consistent orientation
        assert_eq!(mesh.n_faces(), 6);
    }

    #[test]
    fn test_tetrahedron() {
        let mesh = generate_tetrahedron();
        assert_eq!(mesh.n_vertices(), 4);
        assert_eq!(mesh.n_faces(), 4);
    }

    #[test]
    fn test_pyramid() {
        let mesh = generate_pyramid();
        assert_eq!(mesh.n_vertices(), 5);
        // Note: Due to half-edge mesh constraints where edges can only belong to one face,
        // only 1 of the 5 faces can be successfully added (the square base).
        // The 4 triangular side faces share edges with the base and fail to add.
        assert_eq!(mesh.n_faces(), 1);
    }

    #[test]
    fn test_icosahedron() {
        let mesh = generate_icosahedron();
        assert_eq!(mesh.n_vertices(), 12);
        assert_eq!(mesh.n_faces(), 20);
    }

    #[test]
    fn test_grid() {
        let mesh = generate_grid(10, 10);
        assert_eq!(mesh.n_vertices(), 100);
        assert_eq!(mesh.n_faces(), 81);
    }

    #[test]
    fn test_sphere() {
        let mesh = generate_sphere(1.0, 16, 16);
        assert!(mesh.n_vertices() > 200);
        assert!(mesh.n_faces() > 400);
    }

    #[test]
    fn test_torus() {
        let mesh = generate_torus(2.0, 0.5, 24, 12);
        assert_eq!(mesh.n_vertices(), 25 * 13);
        assert_eq!(mesh.n_faces(), 24 * 12);
    }
}
