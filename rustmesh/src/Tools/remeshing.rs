//! # Isotropic Remeshing
//!
//! Implementation of the isotropic remeshing algorithm by Botsch & Kobbelt.
//!
//! The algorithm produces a mesh with approximately uniform edge lengths
//! while preserving the shape and topology of the original mesh.
//!
//! ## Algorithm Overview
//!
//! 1. **Split long edges**: Edges longer than 4/3 * target are split
//! 2. **Collapse short edges**: Edges shorter than 4/5 * target are collapsed
//! 3. **Flip edges for valence**: Flip edges to improve vertex valence
//! 4. **Tangential smoothing**: Smooth vertices while preserving shape
//!
//! ## References
//!
//! - Botsch, M., & Kobbelt, L. (2004). "A Remeshing Approach to Multiresolution Modeling".
//!   Symposium on Geometry Processing.

use crate::{Vec3, Point3};
use crate::handles::{EdgeHandle, VertexHandle};
use crate::smoother::{tangential_smooth, SmootherConfig};
use crate::RustMesh;

/// Edge length statistics for a mesh
#[derive(Debug, Clone)]
pub struct EdgeLengthStats {
    /// Minimum edge length
    pub min_length: f32,
    /// Maximum edge length
    pub max_length: f32,
    /// Mean edge length
    pub mean_length: f32,
    /// Number of edges
    pub n_edges: usize,
}

fn active_edge_endpoints(mesh: &RustMesh, eh: EdgeHandle) -> Option<(VertexHandle, VertexHandle)> {
    if mesh.is_edge_deleted(eh) {
        return None;
    }

    let h0 = mesh.edge_halfedge_handle(eh, 0);
    let v0 = mesh.from_vertex_handle(h0);
    let v1 = mesh.to_vertex_handle(h0);

    if !v0.is_valid()
        || !v1.is_valid()
        || v0 == v1
        || mesh.is_vertex_deleted(v0)
        || mesh.is_vertex_deleted(v1)
    {
        return None;
    }

    Some((v0, v1))
}

/// Compute edge length statistics for a mesh
pub fn edge_length_statistics(mesh: &RustMesh) -> EdgeLengthStats {
    let mut min_length = f32::MAX;
    let mut max_length = 0.0f32;
    let mut total_length = 0.0f32;
    let mut counted_edges = 0usize;

    for i in 0..mesh.n_edges() {
        let eh = EdgeHandle::new(i as u32);
        if let Some((v0, v1)) = active_edge_endpoints(mesh, eh) {
            let (Some(p0), Some(p1)) = (mesh.point(v0), mesh.point(v1)) else {
                continue;
            };
            let length = (p1 - p0).norm();
            if !length.is_finite() || length <= f32::EPSILON {
                continue;
            }

            min_length = min_length.min(length);
            max_length = max_length.max(length);
            total_length += length;
            counted_edges += 1;
        }
    }

    if counted_edges == 0 {
        return EdgeLengthStats {
            min_length: 0.0,
            max_length: 0.0,
            mean_length: 0.0,
            n_edges: 0,
        };
    }

    EdgeLengthStats {
        min_length,
        max_length,
        mean_length: total_length / counted_edges as f32,
        n_edges: counted_edges,
    }
}

/// Split all edges longer than a threshold
///
/// For each edge longer than `max_length`, insert a new vertex at the midpoint
/// and split the adjacent faces.
///
/// # Returns
/// Number of edges split
pub fn split_long_edges(mesh: &mut RustMesh, max_length: f32) -> usize {
    mesh.garbage_collection();

    let mut edges_to_split: Vec<(VertexHandle, VertexHandle, Vec3)> = Vec::new();

    for i in 0..mesh.n_edges() {
        let eh = EdgeHandle::new(i as u32);
        if let Some((v0, v1)) = active_edge_endpoints(mesh, eh) {
            let (Some(p0), Some(p1)) = (mesh.point(v0), mesh.point(v1)) else {
                continue;
            };
            let length = (p1 - p0).norm();
            if length > max_length {
                edges_to_split.push((v0, v1, (p0 + p1) * 0.5));
            }
        }
    }

    if edges_to_split.is_empty() {
        return 0;
    }

    let mut split_count = 0usize;

    for (v0, v1, midpoint) in edges_to_split {
        let Some(eh) = mesh.find_edge_between(v0, v1) else {
            continue;
        };

        if mesh.split_edge(eh, midpoint).is_ok() {
            split_count += 1;
        }
    }

    split_count
}

/// Collapse all edges shorter than a threshold
///
/// For each edge shorter than `min_length`, collapse it to one endpoint.
///
/// # Returns
/// Number of edges collapsed
pub fn collapse_short_edges(mesh: &mut RustMesh, min_length: f32) -> usize {
    let mut collapse_count = 0;

    // Collect edges to collapse
    let mut edges_to_collapse: Vec<EdgeHandle> = Vec::new();

    for i in 0..mesh.n_edges() {
        let eh = EdgeHandle::new(i as u32);
        if mesh.is_edge_deleted(eh) {
            continue;
        }

        let h0 = mesh.edge_halfedge_handle(eh, 0);
        let v0 = mesh.from_vertex_handle(h0);
        let v1 = mesh.to_vertex_handle(h0);

        if let (Some(p0), Some(p1)) = (mesh.point(v0), mesh.point(v1)) {
            let length = (p1 - p0).norm();
            if length < min_length {
                edges_to_collapse.push(eh);
            }
        }
    }

    // Collapse each edge
    for eh in edges_to_collapse {
        let h0 = mesh.edge_halfedge_handle(eh, 0);

        // Check if collapse is legal
        if mesh.is_collapse_ok(h0) {
            // Move the vertex to midpoint for smoother result
            let v0 = mesh.from_vertex_handle(h0);
            let v1 = mesh.to_vertex_handle(h0);

            if let (Some(p0), Some(p1)) = (mesh.point(v0), mesh.point(v1)) {
                let midpoint = (p0 + p1) * 0.5;
                mesh.set_point(v0, midpoint);
            }

            // Collapse the edge
            if mesh.collapse(h0).is_ok() {
                collapse_count += 1;
            }
        }
    }

    if collapse_count > 0 {
        mesh.garbage_collection();
    }

    collapse_count
}

/// Get the valence (number of incident edges) of a vertex
pub fn vertex_valence(mesh: &RustMesh, vh: VertexHandle) -> usize {
    if !vh.is_valid() || mesh.is_vertex_deleted(vh) {
        return 0;
    }

    let mut count = 0usize;

    for i in 0..mesh.n_edges() {
        let eh = EdgeHandle::new(i as u32);
        if let Some((v0, v1)) = active_edge_endpoints(mesh, eh) {
            if v0 == vh || v1 == vh {
                count += 1;
            }
        }
    }

    count
}

/// Compute optimal valence for a vertex
///
/// For interior vertices: 6
/// For boundary vertices: 4
fn optimal_valence(mesh: &RustMesh, vh: VertexHandle) -> usize {
    if is_boundary_vertex(mesh, vh) {
        4
    } else {
        6
    }
}

/// Check if a vertex is on the boundary
fn is_boundary_vertex(mesh: &RustMesh, vh: VertexHandle) -> bool {
    if !vh.is_valid() || mesh.is_vertex_deleted(vh) {
        return false;
    }

    for i in 0..mesh.n_edges() {
        let eh = EdgeHandle::new(i as u32);
        if let Some((v0, v1)) = active_edge_endpoints(mesh, eh) {
            if v0 == vh || v1 == vh {
                let h0 = mesh.edge_halfedge_handle(eh, 0);
                let h1 = mesh.edge_halfedge_handle(eh, 1);
                if mesh.is_boundary(h0) || mesh.is_boundary(h1) {
                    return true;
                }
            }
        }
    }
    false
}

fn edge_flip_vertices(
    mesh: &RustMesh,
    eh: EdgeHandle,
) -> Option<(VertexHandle, VertexHandle, VertexHandle, VertexHandle)> {
    let h0 = mesh.edge_halfedge_handle(eh, 0);
    if mesh.is_boundary(h0) {
        return None;
    }

    let h1 = mesh.opposite_halfedge_handle(h0);
    if mesh.is_boundary(h1) {
        return None;
    }

    let v0 = mesh.from_vertex_handle(h0);
    let v1 = mesh.to_vertex_handle(h0);
    let v2 = mesh.to_vertex_handle(mesh.next_halfedge_handle(h0));
    let v3 = mesh.to_vertex_handle(mesh.next_halfedge_handle(h1));

    if !v0.is_valid()
        || !v1.is_valid()
        || !v2.is_valid()
        || !v3.is_valid()
        || mesh.is_vertex_deleted(v0)
        || mesh.is_vertex_deleted(v1)
        || mesh.is_vertex_deleted(v2)
        || mesh.is_vertex_deleted(v3)
    {
        return None;
    }

    Some((v0, v1, v2, v3))
}

/// Flip edges to improve vertex valence
///
/// For each edge, check if flipping would improve the valence of its endpoints.
/// An edge is flipped if it reduces the deviation from optimal valence.
///
/// # Returns
/// Number of edges flipped
pub fn flip_edges_for_valence(mesh: &mut RustMesh) -> usize {
    let mut flip_count = 0;

    // Collect potential flips
    let mut edges_to_check: Vec<EdgeHandle> = Vec::new();
    let mut valences = vec![0usize; mesh.n_vertices()];
    let mut boundary_vertices = vec![false; mesh.n_vertices()];

    for i in 0..mesh.n_edges() {
        let eh = EdgeHandle::new(i as u32);
        if let Some((v0, v1)) = active_edge_endpoints(mesh, eh) {
            valences[v0.idx_usize()] += 1;
            valences[v1.idx_usize()] += 1;

            let h0 = mesh.edge_halfedge_handle(eh, 0);
            let h1 = mesh.edge_halfedge_handle(eh, 1);
            if mesh.is_boundary(h0) || mesh.is_boundary(h1) {
                boundary_vertices[v0.idx_usize()] = true;
                boundary_vertices[v1.idx_usize()] = true;
            }

            edges_to_check.push(eh);
        }
    }

    for eh in edges_to_check {
        let Some((v0, v1, v2, v3)) = edge_flip_vertices(mesh, eh) else {
            continue;
        };

        if should_flip_for_valence(mesh, eh, &valences, &boundary_vertices) {
            if mesh.flip_edge(eh).is_ok() {
                valences[v0.idx_usize()] = valences[v0.idx_usize()].saturating_sub(1);
                valences[v1.idx_usize()] = valences[v1.idx_usize()].saturating_sub(1);
                valences[v2.idx_usize()] += 1;
                valences[v3.idx_usize()] += 1;
                flip_count += 1;
            }
        }
    }

    flip_count
}

/// Check if flipping an edge would improve valence
fn should_flip_for_valence(
    mesh: &RustMesh,
    eh: EdgeHandle,
    valences: &[usize],
    boundary_vertices: &[bool],
) -> bool {
    let Some((v0, v1, v2, v3)) = edge_flip_vertices(mesh, eh) else {
        return false;
    };

    let val0 = valences[v0.idx_usize()];
    let val1 = valences[v1.idx_usize()];
    let val2 = valences[v2.idx_usize()];
    let val3 = valences[v3.idx_usize()];

    if val0 == 0 || val1 == 0 || val2 == 0 || val3 == 0 {
        return false;
    }

    let opt0 = if boundary_vertices[v0.idx_usize()] {
        4
    } else {
        6
    };
    let opt1 = if boundary_vertices[v1.idx_usize()] {
        4
    } else {
        6
    };
    let opt2 = if boundary_vertices[v2.idx_usize()] {
        4
    } else {
        6
    };
    let opt3 = if boundary_vertices[v3.idx_usize()] {
        4
    } else {
        6
    };

    // Current deviation
    let current_deviation = (val0 as i32 - opt0 as i32).abs()
        + (val1 as i32 - opt1 as i32).abs()
        + (val2 as i32 - opt2 as i32).abs()
        + (val3 as i32 - opt3 as i32).abs();

    // After flip: v0 and v1 lose one incident edge, v2 and v3 gain one
    let new_val0 = val0.saturating_sub(1);
    let new_val1 = val1.saturating_sub(1);
    let new_val2 = val2 + 1;
    let new_val3 = val3 + 1;

    let new_deviation = (new_val0 as i32 - opt0 as i32).abs()
        + (new_val1 as i32 - opt1 as i32).abs()
        + (new_val2 as i32 - opt2 as i32).abs()
        + (new_val3 as i32 - opt3 as i32).abs();

    new_deviation < current_deviation
}

/// Isotropic remeshing configuration
#[derive(Debug, Clone)]
pub struct RemeshingConfig {
    /// Target edge length
    pub target_edge_length: f32,
    /// Number of iterations
    pub iterations: usize,
    /// Enable edge splitting
    pub enable_split: bool,
    /// Enable edge collapse
    pub enable_collapse: bool,
    /// Enable edge flipping
    pub enable_flip: bool,
    /// Enable tangential smoothing
    pub enable_smooth: bool,
}

impl Default for RemeshingConfig {
    fn default() -> Self {
        Self {
            target_edge_length: 0.1,
            iterations: 10,
            enable_split: true,
            enable_collapse: true,
            enable_flip: true,
            enable_smooth: true,
        }
    }
}

/// Result of isotropic remeshing
#[derive(Debug, Clone)]
pub struct RemeshingResult {
    /// Number of edges split
    pub split_count: usize,
    /// Number of edges collapsed
    pub collapse_count: usize,
    /// Number of edges flipped
    pub flip_count: usize,
    /// Final edge length statistics
    pub final_stats: EdgeLengthStats,
}

/// Perform isotropic remeshing
///
/// This implements the Botsch & Kobbelt algorithm for isotropic remeshing:
/// 1. Split edges longer than 4/3 * target
/// 2. Collapse edges shorter than 4/5 * target
/// 3. Flip edges to improve valence
/// 4. Apply tangential smoothing
///
/// # Arguments
/// * `mesh` - The mesh to remesh (modified in place)
/// * `config` - Remeshing configuration
///
/// # Returns
/// Statistics about the remeshing operation
pub fn isotropic_remesh(mesh: &mut RustMesh, config: RemeshingConfig) -> RemeshingResult {
    let target = config.target_edge_length;
    let max_length = target * 4.0 / 3.0;
    let min_length = target * 4.0 / 5.0;

    let mut total_split = 0;
    let mut total_collapse = 0;
    let mut total_flip = 0;

    for _iter in 0..config.iterations {
        // Step 1: Split long edges
        if config.enable_split {
            let split = split_long_edges(mesh, max_length);
            total_split += split;
        }

        // Step 2: Collapse short edges
        if config.enable_collapse {
            let collapse = collapse_short_edges(mesh, min_length);
            total_collapse += collapse;
        }

        // Step 3: Flip edges for valence
        if config.enable_flip {
            let flip = flip_edges_for_valence(mesh);
            total_flip += flip;
        }

        // Step 4: Tangential smoothing
        if config.enable_smooth {
            let smooth_config = SmootherConfig {
                iterations: 1,
                strength: 0.5,
                uniform: false,
                fixed_boundary: true,
            };
            tangential_smooth(mesh, smooth_config);
        }
    }

    // Run garbage collection to clean up deleted elements
    mesh.garbage_collection();

    let final_stats = edge_length_statistics(mesh);

    RemeshingResult {
        split_count: total_split,
        collapse_count: total_collapse,
        flip_count: total_flip,
        final_stats,
    }
}

/// Equalize edge lengths through iterative remeshing
///
/// This is a simplified version that focuses on achieving uniform edge lengths
/// without changing the mesh topology significantly.
pub fn equalize_edge_lengths(mesh: &mut RustMesh, target_length: f32, iterations: usize) {
    let config = RemeshingConfig {
        target_edge_length: target_length,
        iterations,
        enable_split: true,
        enable_collapse: true,
        enable_flip: true,
        enable_smooth: true,
    };

    isotropic_remesh(mesh, config);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate_sphere;

    fn make_two_triangle_patch(width: f32, height: f32) -> RustMesh {
        let mut mesh = RustMesh::new();
        let v0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(Point3::new(width, 0.0, 0.0));
        let v2 = mesh.add_vertex(Point3::new(width, height, 0.0));
        let v3 = mesh.add_vertex(Point3::new(0.0, height, 0.0));
        mesh.add_face(&[v0, v1, v2]);
        mesh.add_face(&[v0, v2, v3]);
        mesh
    }

    #[test]
    fn test_edge_length_statistics() {
        let mesh = generate_sphere(1.0, 16, 16);
        let stats = edge_length_statistics(&mesh);

        println!("Edge length stats: {:?}", stats);

        assert!(stats.n_edges > 0);
        assert!(stats.min_length > 0.0);
        assert!(stats.max_length >= stats.min_length);
        assert!(stats.mean_length >= stats.min_length);
        assert!(stats.mean_length <= stats.max_length);
    }

    #[test]
    fn test_vertex_valence() {
        let mesh = generate_sphere(1.0, 16, 16);

        // For a sphere mesh, interior vertices should have valence 6 (approximately)
        // and vertices at poles might have different valence
        let mut valences: Vec<usize> = Vec::new();

        for vh in mesh.vertices() {
            valences.push(vertex_valence(&mesh, vh));
        }

        println!(
            "Valence distribution: {:?}",
            valences.iter().sum::<usize>() / valences.len()
        );

        // Most vertices should have valence around 4-8
        let avg_valence: f32 = valences.iter().sum::<usize>() as f32 / valences.len() as f32;
        assert!(avg_valence > 0.0 && avg_valence < 20.0);
    }

    #[test]
    fn test_split_long_edges_splits_shared_interior_edge_into_four_triangles() {
        let mut mesh = make_two_triangle_patch(2.2, 1.0);
        let original_stats = edge_length_statistics(&mesh);

        let split = split_long_edges(&mut mesh, 2.25);

        assert_eq!(split, 1);
        assert!(original_stats.max_length > 2.25);
        assert!(mesh.validate().is_ok());
        assert_eq!(mesh.n_faces(), 4);
        assert_eq!(mesh.n_active_faces(), 4);
        let final_stats = edge_length_statistics(&mesh);
        assert!(final_stats.max_length <= 2.25 + 1e-5);

        for fh in mesh.faces() {
            let verts = mesh.face_vertices_vec(fh);
            assert_eq!(verts.len(), 3);
            assert_ne!(verts[0], verts[1]);
            assert_ne!(verts[1], verts[2]);
            assert_ne!(verts[2], verts[0]);
        }
    }

    #[test]
    fn test_isotropic_remesh_split_only_keeps_shared_split_path_valid() {
        let mut mesh = make_two_triangle_patch(3.0, 1.0);

        let config = RemeshingConfig {
            target_edge_length: 1.5,
            iterations: 1,
            enable_split: true,
            enable_collapse: false,
            enable_flip: false,
            enable_smooth: false,
        };

        let result = isotropic_remesh(&mut mesh, config);

        assert_eq!(result.split_count, 3);
        assert_eq!(result.collapse_count, 0);
        assert_eq!(result.flip_count, 0);
        assert!(mesh.validate().is_ok());
        assert_eq!(mesh.n_faces(), mesh.n_active_faces());
        assert!(result.final_stats.max_length <= 2.0 + 1e-5);
    }

    #[test]
    fn test_isotropic_remesh_sphere() {
        let mut mesh = generate_sphere(1.0, 8, 8);

        let original_stats = edge_length_statistics(&mesh);
        println!(
            "Original edge lengths: min={}, max={}, mean={}",
            original_stats.min_length, original_stats.max_length, original_stats.mean_length
        );

        let config = RemeshingConfig {
            target_edge_length: 0.3,
            iterations: 5,
            enable_split: true,
            enable_collapse: true,
            enable_flip: true,
            enable_smooth: true,
        };

        let result = isotropic_remesh(&mut mesh, config);

        println!(
            "Remeshing result: split={}, collapse={}, flip={}",
            result.split_count, result.collapse_count, result.flip_count
        );
        println!(
            "Final edge lengths: min={}, max={}, mean={}",
            result.final_stats.min_length,
            result.final_stats.max_length,
            result.final_stats.mean_length
        );

        // Mesh should still have faces
        assert!(mesh.n_active_faces() > 0);
        assert!(result.split_count + result.collapse_count + result.flip_count > 0);
        assert!(mesh.validate().is_ok());
        assert_eq!(mesh.n_faces(), mesh.n_active_faces());
        assert!(result.final_stats.n_edges > 0);
    }

    #[test]
    fn test_flip_edge_basic() {
        let mut mesh = RustMesh::new();

        // Create two triangles sharing an edge
        let v0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(Point3::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(Point3::new(0.5, 1.0, 0.0));
        let v3 = mesh.add_vertex(Point3::new(0.5, -1.0, 0.0));

        mesh.add_face(&[v0, v1, v2]);
        mesh.add_face(&[v0, v3, v1]);

        // Get the edge between v0 and v1
        let eh = find_edge(&mesh, v0, v1).expect("Edge should exist");

        // Flip the edge
        let result = mesh.flip_edge(eh);
        assert!(result.is_ok(), "Edge flip should succeed");

        // After flip, the edge should now connect v2 and v3
        // Verify by checking valence
        assert_eq!(vertex_valence(&mesh, v0), 2);
        assert_eq!(vertex_valence(&mesh, v1), 2);
        assert_eq!(vertex_valence(&mesh, v2), 3);
        assert_eq!(vertex_valence(&mesh, v3), 3);
    }

    /// Helper function to find an edge between two vertices
    fn find_edge(mesh: &RustMesh, v0: VertexHandle, v1: VertexHandle) -> Option<EdgeHandle> {
        for i in 0..mesh.n_edges() {
            let eh = EdgeHandle::new(i as u32);
            if let Some((from, to)) = active_edge_endpoints(mesh, eh) {
                if (from == v0 && to == v1) || (from == v1 && to == v0) {
                    return Some(eh);
                }
            }
        }
        None
    }

    #[test]
    fn test_flip_boundary_edge_fails() {
        let mut mesh = RustMesh::new();

        // Create a single triangle (boundary edges)
        let v0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(Point3::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(Point3::new(0.5, 1.0, 0.0));

        mesh.add_face(&[v0, v1, v2]);

        // Get the edge between v0 and v1 (this is a boundary edge)
        let eh = find_edge(&mesh, v0, v1).expect("Edge should exist");

        // Flipping a boundary edge should fail
        let result = mesh.flip_edge(eh);
        assert!(result.is_err(), "Flipping boundary edge should fail");
    }

    #[test]
    fn test_optimal_valence() {
        let mut mesh = RustMesh::new();

        let v0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(Point3::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(Point3::new(0.5, 1.0, 0.0));

        mesh.add_face(&[v0, v1, v2]);

        // All vertices are boundary vertices in a single triangle
        assert_eq!(optimal_valence(&mesh, v0), 4);
        assert_eq!(optimal_valence(&mesh, v1), 4);
        assert_eq!(optimal_valence(&mesh, v2), 4);
    }
}
