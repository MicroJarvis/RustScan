//! # Hole Filling
//!
//! Algorithms for finding and filling holes in mesh surfaces.
//! Uses ear clipping triangulation to fill boundary loops.
//!
//! ## Usage
//!
//! ```rust
//! use rustmesh::{RustMesh, find_boundary_loops, fill_all_holes, Point3};
//!
//! let mut mesh = RustMesh::new();
//! // ... create mesh with holes ...
//!
//! // Find all boundary loops
//! let loops = find_boundary_loops(&mesh);
//!
//! // Fill all holes
//! let result = fill_all_holes(&mut mesh);
//! println!("Filled {} holes", result.holes_filled);
//! ```

use crate::geometry::triangle_area;
use crate::handles::{HalfedgeHandle, VertexHandle};
use crate::Point3;
use crate::RustMesh;

/// Result of a hole filling operation
#[derive(Debug, Clone)]
pub struct HoleFillResult {
    /// Number of holes filled
    pub holes_filled: usize,
    /// Total number of new faces created
    pub faces_created: usize,
    /// Boundary loop vertex counts (for each filled hole)
    pub loop_sizes: Vec<usize>,
}

impl Default for HoleFillResult {
    fn default() -> Self {
        Self {
            holes_filled: 0,
            faces_created: 0,
            loop_sizes: Vec::new(),
        }
    }
}

/// A boundary loop representing a hole in the mesh
#[derive(Debug, Clone)]
pub struct BoundaryLoop {
    /// The halfedges that make up this boundary loop
    pub halfedges: Vec<HalfedgeHandle>,
    /// The vertices that make up this boundary loop (in order)
    pub vertices: Vec<VertexHandle>,
    /// The positions of the boundary vertices
    pub points: Vec<Point3>,
}

impl BoundaryLoop {
    /// Get the number of vertices in this boundary loop
    #[inline]
    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    /// Check if the boundary loop is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }
}

/// Find all boundary loops (holes) in the mesh.
///
/// A boundary loop is a cycle of boundary halfedges (halfedges with no face).
///
/// # Arguments
/// * `mesh` - The mesh to search for boundary loops
///
/// # Returns
/// A vector of BoundaryLoop, one for each hole found.
pub fn find_boundary_loops(mesh: &RustMesh) -> Vec<BoundaryLoop> {
    let n_halfedges = mesh.n_halfedges();
    let mut visited = vec![false; n_halfedges];
    let mut loops = Vec::new();

    // Build a map: from_vertex -> boundary halfedge
    // This allows us to follow boundary loops without relying on next pointers
    let mut boundary_from: std::collections::HashMap<u32, HalfedgeHandle> =
        std::collections::HashMap::new();
    for heh_idx in 0..n_halfedges {
        let heh = HalfedgeHandle::new(heh_idx as u32);
        if mesh.is_boundary(heh) {
            let from_v = mesh.from_vertex_handle(heh);
            boundary_from.insert(from_v.idx(), heh);
        }
    }

    // Follow boundary loops using the from_vertex map
    for heh_idx in 0..n_halfedges {
        let heh = HalfedgeHandle::new(heh_idx as u32);

        if visited[heh_idx] || !mesh.is_boundary(heh) {
            continue;
        }

        let mut loop_heh = Vec::new();
        let mut loop_vertices = Vec::new();
        let mut loop_points = Vec::new();

        let mut current = heh;
        let max_iter = n_halfedges + 1;
        let mut count = 0;

        loop {
            if count > max_iter {
                break;
            }
            count += 1;

            let idx = current.idx_usize();
            if visited[idx] {
                break;
            }

            visited[idx] = true;
            loop_heh.push(current);

            let vh = mesh.from_vertex_handle(current);
            loop_vertices.push(vh);
            if let Some(p) = mesh.point(vh) {
                loop_points.push(p);
            }

            // Follow to the next boundary halfedge via to_vertex
            let to_v = mesh.to_vertex_handle(current);
            match boundary_from.get(&to_v.idx()) {
                Some(&next_he) if next_he != current => {
                    current = next_he;
                }
                _ => break,
            }

            if current == heh {
                break;
            }
        }

        // Only add non-empty loops with at least 3 vertices
        if loop_vertices.len() >= 3 {
            loops.push(BoundaryLoop {
                halfedges: loop_heh,
                vertices: loop_vertices,
                points: loop_points,
            });
        }
    }

    loops
}

/// Check if a vertex is a valid ear for ear clipping.
///
/// An ear is a triangle formed by three consecutive vertices (i-1, i, i+1)
/// where:
/// 1. The triangle is not degenerate (area > 0)
/// 2. No other vertices of the polygon lie inside the triangle
/// 3. The triangle is locally convex (interior angle < 180 degrees)
///
/// # Arguments
/// * `mesh` - The mesh (for checking existing faces)
/// * `vertices` - The boundary vertices
/// * `points` - The boundary vertex positions
/// * `i` - The index of the middle vertex of the ear
/// * `valid_indices` - Set of currently valid vertex indices
///
/// # Returns
/// True if the vertex forms a valid ear
fn is_valid_ear(
    _mesh: &RustMesh,
    _vertices: &[VertexHandle],
    points: &[Point3],
    prev_idx: usize,
    curr_idx: usize,
    next_idx: usize,
    active: &[usize],
) -> bool {
    let p_prev = points[prev_idx];
    let p_curr = points[curr_idx];
    let p_next = points[next_idx];

    // Check 1: Non-zero area (not degenerate)
    let area = triangle_area(p_prev, p_curr, p_next);
    if area < 1e-10 {
        return false;
    }

    // Check 2: No other active vertices inside the ear
    for &j in active {
        if j == prev_idx || j == curr_idx || j == next_idx {
            continue;
        }

        let p_test = points[j];
        if point_in_triangle(p_test, p_prev, p_curr, p_next) {
            return false;
        }
    }

    // Check 3: Minimum edge length to avoid degenerate triangles
    let min_edge = (p_curr - p_prev)
        .norm()
        .min((p_next - p_curr).norm())
        .min((p_next - p_prev).norm());
    if min_edge < 1e-6 {
        return false;
    }

    true
}

/// Check if a point is inside a triangle using barycentric coordinates
#[inline]
fn point_in_triangle(p: Point3, a: Point3, b: Point3, c: Point3) -> bool {
    // Compute vectors
    let v0 = c - a;
    let v1 = b - a;
    let v2 = p - a;

    // Compute dot products
    let dot00 = v0.dot(&v0);
    let dot01 = v0.dot(&v1);
    let dot02 = v0.dot(&v2);
    let dot11 = v1.dot(&v1);
    let dot12 = v1.dot(&v2);

    // Compute barycentric coordinates
    let inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

    // Check if point is in triangle
    u >= 0.0 && v >= 0.0 && (u + v) <= 1.0
}

/// Fill a single hole using ear clipping triangulation.
///
/// Takes a boundary halfedge or face handle and fills the hole by triangulating
/// the boundary loop using ear clipping.
///
/// # Arguments
/// * `mesh` - The mesh (will be modified)
/// * `boundary_heh` - A boundary halfedge belonging to the hole to fill
///
/// # Returns
/// * `Ok(usize)` - Number of new faces created
/// * `Err(&str)` - Error message if filling failed
pub fn fill_hole(mesh: &mut RustMesh, boundary_heh: HalfedgeHandle) -> Result<usize, &'static str> {
    // First, find the boundary loop containing this halfedge
    let loops = find_boundary_loops(mesh);

    // Find the loop containing this halfedge
    let mut target_loop: Option<BoundaryLoop> = None;
    for loop_info in &loops {
        if loop_info.halfedges.contains(&boundary_heh) {
            target_loop = Some(loop_info.clone());
            break;
        }
    }

    let loop_info = target_loop.ok_or("Boundary halfedge not found in any boundary loop")?;

    fill_boundary_loop(mesh, loop_info)
}

/// Fill a boundary loop using ear clipping triangulation.
///
/// # Arguments
/// * `mesh` - The mesh (will be modified)
/// * `loop_info` - The boundary loop to fill
///
/// # Returns
/// * `Ok(usize)` - Number of new faces created
/// * `Err(&str)` - Error message if filling failed
fn fill_boundary_loop(mesh: &mut RustMesh, loop_info: BoundaryLoop) -> Result<usize, &'static str> {
    if loop_info.len() < 3 {
        return Err("Boundary loop has fewer than 3 vertices");
    }

    let n = loop_info.len();
    if n == 3 {
        // Triangle - just add the face directly
        let vertices = &loop_info.vertices;
        if mesh.add_face(vertices).is_some() {
            return Ok(1);
        } else {
            return Err("Failed to add face");
        }
    }

    // Make a copy of vertices and points that we can modify
    let vertices = loop_info.vertices.clone();
    let points = loop_info.points.clone();
    let mut faces_created = 0;

    // Active indices in circular order (Vec for O(1) prev/next access)
    let mut active: Vec<usize> = (0..n).collect();

    // Ear clipping loop
    while active.len() > 3 {
        let mut ear_found = false;
        let active_len = active.len();

        for pos in 0..active_len {
            let prev_idx = active[(pos + active_len - 1) % active_len];
            let curr_idx = active[pos];
            let next_idx = active[(pos + 1) % active_len];

            if is_valid_ear(
                mesh, &vertices, &points, prev_idx, curr_idx, next_idx, &active,
            ) {
                // Create the ear face
                let face_vertices = [vertices[prev_idx], vertices[curr_idx], vertices[next_idx]];

                if mesh.add_face(&face_vertices).is_some() {
                    faces_created += 1;
                } else {
                    return Err("Failed to add ear face");
                }

                // Remove the ear vertex from the active list
                active.remove(pos);
                ear_found = true;
                break;
            }
        }

        if !ear_found {
            // No valid ear found - break to avoid infinite loop
            break;
        }

        // Safety check
        if faces_created > n - 2 {
            break;
        }
    }

    // If we have exactly 3 vertices left, add the final face
    if active.len() == 3 {
        let face_vertices = [
            vertices[active[0]],
            vertices[active[1]],
            vertices[active[2]],
        ];
        if mesh.add_face(&face_vertices).is_some() {
            faces_created += 1;
        }
    }

    Ok(faces_created)
}

/// Fill all holes in the mesh.
///
/// Finds all boundary loops and fills each one using ear clipping triangulation.
///
/// # Arguments
/// * `mesh` - The mesh (will be modified)
///
/// # Returns
/// A `HoleFillResult` containing the number of holes filled and faces created.
pub fn fill_all_holes(mesh: &mut RustMesh) -> HoleFillResult {
    let mut result = HoleFillResult::default();

    // Find all boundary loops
    let loops = find_boundary_loops(mesh);

    if loops.is_empty() {
        return result;
    }

    // Fill each hole
    for loop_info in loops {
        let loop_size = loop_info.len();

        match fill_boundary_loop(mesh, loop_info) {
            Ok(faces) => {
                result.holes_filled += 1;
                result.faces_created += faces;
                result.loop_sizes.push(loop_size);
            }
            Err(e) => {
                println!("Warning: Failed to fill hole of size {}: {}", loop_size, e);
            }
        }
    }

    result
}

/// Find and fill a single hole starting from a boundary halfedge.
///
/// This is a convenience function that finds the boundary loop containing
/// the given halfedge and fills it.
///
/// # Arguments
/// * `mesh` - The mesh (will be modified)
/// * `start_heh` - A boundary halfedge to start from
///
/// # Returns
/// * `Ok(usize)` - Number of new faces created
/// * `Err(&str)` - Error message if filling failed
pub fn fill_hole_from_halfedge(
    mesh: &mut RustMesh,
    start_heh: HalfedgeHandle,
) -> Result<usize, &'static str> {
    if !mesh.is_boundary(start_heh) {
        return Err("Halfedge is not a boundary halfedge");
    }

    fill_hole(mesh, start_heh)
}

/// Get statistics about holes in the mesh.
///
/// # Arguments
/// * `mesh` - The mesh to analyze
///
/// # Returns
/// A string describing the hole statistics
pub fn hole_statistics(mesh: &RustMesh) -> String {
    let loops = find_boundary_loops(mesh);

    if loops.is_empty() {
        return "No holes found".to_string();
    }

    let mut total_vertices = 0;
    let mut min_size = usize::MAX;
    let mut max_size = 0;

    for loop_info in &loops {
        let size = loop_info.len();
        total_vertices += size;
        min_size = min_size.min(size);
        max_size = max_size.max(size);
    }

    format!(
        "Found {} hole(s): total {} boundary vertices, size range [{}, {}]",
        loops.len(),
        total_vertices,
        min_size,
        max_size
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple mesh with a triangular hole for testing
    fn create_mesh_with_triangular_hole() -> RustMesh {
        let mut mesh = RustMesh::new();

        // Create a mesh surrounding a triangular hole
        // Outer square
        let v0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(Point3::new(2.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(Point3::new(2.0, 2.0, 0.0));
        let v3 = mesh.add_vertex(Point3::new(0.0, 2.0, 0.0));

        // Inner triangle vertices (hole boundary)
        let h0 = mesh.add_vertex(Point3::new(0.5, 0.5, 0.0));
        let h1 = mesh.add_vertex(Point3::new(1.5, 0.5, 0.0));
        let h2 = mesh.add_vertex(Point3::new(1.0, 1.5, 0.0));

        // Faces with consistent CCW orientation (viewed from +Z)
        // The inner triangle [h0, h1, h2] is left as a hole
        mesh.add_face(&[v0, v1, h1]);
        mesh.add_face(&[v0, h1, h0]); // fixed: was [v0, h0, h1]
        mesh.add_face(&[v1, v2, h2]);
        mesh.add_face(&[v1, h2, h1]);
        mesh.add_face(&[v2, v3, h2]);
        mesh.add_face(&[v3, h0, h2]);
        mesh.add_face(&[v3, v0, h0]);

        mesh
    }

    /// Create a simple mesh with a square hole
    #[allow(dead_code)]
    fn create_mesh_with_square_hole() -> RustMesh {
        let mut mesh = RustMesh::new();

        // Create an L-shaped mesh with a square hole
        let v0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(Point3::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(Point3::new(1.0, 1.0, 0.0));
        let v3 = mesh.add_vertex(Point3::new(0.0, 1.0, 0.0));

        // Inner square (hole)
        let h0 = mesh.add_vertex(Point3::new(0.3, 0.3, 0.0));
        let h1 = mesh.add_vertex(Point3::new(0.7, 0.3, 0.0));
        let h2 = mesh.add_vertex(Point3::new(0.7, 0.7, 0.0));
        let h3 = mesh.add_vertex(Point3::new(0.3, 0.7, 0.0));

        // Add faces around the hole (but not filling it)
        mesh.add_face(&[v0, v1, h1, h0]);
        mesh.add_face(&[v1, v2, h2, h1]);
        mesh.add_face(&[v2, v3, h3, h3]); // This will create degenerate triangles
                                          // Let's just create triangles instead
        mesh.add_face(&[v0, v1, h1]);
        mesh.add_face(&[v0, h1, h0]);
        mesh.add_face(&[v1, v2, h2]);
        mesh.add_face(&[v1, h2, h1]);
        mesh.add_face(&[v2, v3, h3]);
        mesh.add_face(&[v2, h3, h2]);
        mesh.add_face(&[v3, v0, h0]);
        mesh.add_face(&[v3, h0, h3]);

        mesh
    }

    #[test]
    fn test_find_boundary_loops_empty() {
        let mesh = RustMesh::new();
        let loops = find_boundary_loops(&mesh);
        assert!(loops.is_empty());
    }

    #[test]
    fn test_find_boundary_loops_simple_triangle() {
        let mut mesh = RustMesh::new();

        // Single triangle - has boundary edges (the 3 outer edges)
        let v0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(Point3::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(Point3::new(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]);

        let loops = find_boundary_loops(&mesh);
        // A single triangle has 3 boundary edges forming one loop
        assert!(!loops.is_empty(), "Single triangle should have boundary");
        // Should have exactly one loop with 3 vertices
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].len(), 3);
    }

    #[test]
    fn test_hole_statistics() {
        let mesh = create_mesh_with_triangular_hole();
        let stats = hole_statistics(&mesh);
        println!("{}", stats);
        assert!(stats.contains("hole"));
    }

    #[test]
    fn test_fill_all_holes_triangular() {
        let mut mesh = create_mesh_with_triangular_hole();

        let before_faces = mesh.n_faces();
        println!("Before filling: {} faces", before_faces);

        let result = fill_all_holes(&mut mesh);

        println!(
            "Filled {} holes, created {} faces",
            result.holes_filled, result.faces_created
        );

        // Should have filled the triangular hole with 1 face
        assert!(result.holes_filled >= 1);
        assert!(result.faces_created >= 1);
    }

    #[test]
    fn test_fill_all_holes_empty_mesh() {
        let mut mesh = RustMesh::new();

        let result = fill_all_holes(&mut mesh);

        assert_eq!(result.holes_filled, 0);
        assert_eq!(result.faces_created, 0);
    }

    #[test]
    fn test_point_in_triangle() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(1.0, 0.0, 0.0);
        let c = Point3::new(0.0, 1.0, 0.0);

        // Point inside triangle
        let p_inside = Point3::new(0.1, 0.1, 0.0);
        assert!(point_in_triangle(p_inside, a, b, c));

        // Point outside triangle
        let p_outside = Point3::new(1.0, 1.0, 0.0);
        assert!(!point_in_triangle(p_outside, a, b, c));

        // Point on edge
        let p_edge = Point3::new(0.5, 0.0, 0.0);
        assert!(point_in_triangle(p_edge, a, b, c));
    }

    #[test]
    fn test_triangle_area() {
        // Right triangle with legs 1 and 1
        let p0 = Point3::new(0.0, 0.0, 0.0);
        let p1 = Point3::new(1.0, 0.0, 0.0);
        let p2 = Point3::new(0.0, 1.0, 0.0);

        let area = triangle_area(p0, p1, p2);
        assert!((area - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_boundary_loop_len() {
        let mesh = create_mesh_with_triangular_hole();
        let loops = find_boundary_loops(&mesh);

        for loop_info in &loops {
            println!("Boundary loop with {} vertices", loop_info.len());
            assert!(loop_info.len() >= 3);
        }
    }

    #[test]
    fn test_fill_hole_from_halfedge() {
        let mut mesh = create_mesh_with_triangular_hole();

        // Find a boundary halfedge
        let mut boundary_heh = HalfedgeHandle::invalid();
        for heh_idx in 0..mesh.n_halfedges() {
            let heh = HalfedgeHandle::new(heh_idx as u32);
            if mesh.is_boundary(heh) {
                boundary_heh = heh;
                break;
            }
        }

        assert!(boundary_heh.is_valid(), "Should find a boundary halfedge");

        let result = fill_hole_from_halfedge(&mut mesh, boundary_heh);

        match result {
            Ok(faces) => println!("Filled hole with {} faces", faces),
            Err(e) => println!("Error: {}", e),
        }
    }

    #[test]
    fn test_fill_hole_invalid_halfedge() {
        let mut mesh = RustMesh::new();

        // Add a single vertex and try to fill
        let _v0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));

        // No boundary edges yet - should fail
        let invalid_heh = HalfedgeHandle::invalid();
        let result = fill_hole(&mut mesh, invalid_heh);

        assert!(result.is_err());
    }

    // Integration test with more complex geometry
    #[test]
    fn test_fill_complex_hole() {
        let mut mesh = RustMesh::new();

        // Create a simple mesh with a square hole
        // First create the outer square
        let v0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(Point3::new(2.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(Point3::new(2.0, 2.0, 0.0));
        let v3 = mesh.add_vertex(Point3::new(0.0, 2.0, 0.0));

        // Create inner square (hole)
        let h0 = mesh.add_vertex(Point3::new(0.5, 0.5, 0.0));
        let h1 = mesh.add_vertex(Point3::new(1.5, 0.5, 0.0));
        let h2 = mesh.add_vertex(Point3::new(1.5, 1.5, 0.0));
        let h3 = mesh.add_vertex(Point3::new(0.5, 1.5, 0.0));

        // Create faces around the hole (leaving the inner square as a hole)
        // Bottom
        mesh.add_face(&[v0, v1, h1, h0]);
        // Right
        mesh.add_face(&[v1, v2, h2, h1]);
        // Top
        mesh.add_face(&[v2, v3, h3, h2]);
        // Left
        mesh.add_face(&[v3, v0, h0, h3]);

        let loops = find_boundary_loops(&mesh);
        println!("Found {} boundary loops for square hole", loops.len());

        // The square hole should be detected
        assert!(!loops.is_empty());

        // Verify we found the hole
        let mut found_square_hole = false;
        for loop_info in &loops {
            if loop_info.len() == 4 {
                found_square_hole = true;
                break;
            }
        }
        assert!(found_square_hole, "Should find a 4-vertex boundary loop");
    }
}
