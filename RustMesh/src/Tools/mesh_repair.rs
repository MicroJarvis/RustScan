//! # Mesh Repair
//!
//! Mesh repair utilities for cleaning up malformed meshes.
//!
//! - Remove duplicate vertices
//! - Remove degenerate (zero-area) faces
//! - Fix face winding order
//! - Merge close vertices

use crate::connectivity::RustMesh;
use crate::geometry::{triangle_area, triangle_normal};
use crate::handles::{FaceHandle, VertexHandle};
use crate::{Point3, Vec3};
use std::collections::HashMap;

/// Error type for mesh repair operations
#[derive(Debug, Clone)]
pub struct MeshRepairError(pub String);

impl std::fmt::Display for MeshRepairError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MeshRepairError: {}", self.0)
    }
}

impl std::error::Error for MeshRepairError {}

impl From<String> for MeshRepairError {
    fn from(s: String) -> Self {
        MeshRepairError(s)
    }
}

impl From<&str> for MeshRepairError {
    fn from(s: &str) -> Self {
        MeshRepairError(s.to_string())
    }
}

// ============================================================================
// Remove Duplicates
// ============================================================================

/// Merge duplicate vertices based on position.
/// Uses a hash map to find vertices at the same position and merges them.
///
/// Returns the number of vertices that were merged (removed).
pub fn remove_duplicates(mesh: &mut RustMesh) -> Result<usize, MeshRepairError> {
    if mesh.n_vertices() == 0 {
        return Ok(0);
    }

    // Map from quantized position to first vertex handle at that position
    // Using i64 to store quantized coordinates (f32 * 1000000 as i64 to avoid Eq issue)
    let mut position_map: HashMap<(i64, i64, i64), VertexHandle> = HashMap::new();

    // First pass: collect all duplicate pairs (vertex to merge -> target vertex)
    let mut duplicates: Vec<(VertexHandle, VertexHandle)> = Vec::new();

    // Collect vertex data first
    let vertices: Vec<_> = mesh.vertices().collect();

    for vh in vertices {
        if let Some(point) = mesh.point(vh) {
            // Quantize to avoid floating point equality issues
            let key = (
                (point.x * 1_000_000.0) as i64,
                (point.y * 1_000_000.0) as i64,
                (point.z * 1_000_000.0) as i64,
            );

            if let Some(first_vh) = position_map.get(&key) {
                // This is a duplicate - record for later processing
                duplicates.push((vh, *first_vh));
            } else {
                // First occurrence - store it
                position_map.insert(key, vh);
            }
        }
    }

    let merged_count = duplicates.len();

    // Second pass: process duplicates (now we can mutably borrow mesh)
    // Keep a reference for later use
    let duplicates_ref = &duplicates;
    for (old_vh, new_vh) in duplicates.iter().cloned() {
        remap_vertex_in_faces(mesh, old_vh, new_vh);
    }

    // Compact vertices if we have duplicates
    if merged_count > 0 {
        // Build keep_vertex array
        let mut keep_vertex: Vec<bool> = vec![true; mesh.n_vertices()];
        for (old_vh, _) in duplicates_ref {
            keep_vertex[old_vh.idx_usize()] = false;
        }
        compact_vertices(mesh, &keep_vertex)?;
    }

    Ok(merged_count)
}

/// Remap all faces using old_vh to use new_vh instead
fn remap_vertex_in_faces(mesh: &mut RustMesh, old_vh: VertexHandle, new_vh: VertexHandle) {
    // Avoid using vertex_faces circulator here because broken connectivity can
    // cause extremely long iterations. Scan all faces by index instead.
    let face_handles: Vec<_> = mesh.faces().collect();
    for fh in face_handles {
        let verts = mesh.face_vertices_vec(fh);

        if !verts.iter().any(|&v| v == old_vh) {
            continue;
        }

        let face_verts: Vec<VertexHandle> = verts
            .into_iter()
            .map(|v| if v == old_vh { new_vh } else { v })
            .collect();

        // Remove the old face and add a new one with remapped vertices.
        mesh.delete_face(fh);
        let _ = mesh.add_face(&face_verts);
    }
}

/// Compact vertices, removing those marked as false
fn compact_vertices(mesh: &mut RustMesh, keep_vertex: &[bool]) -> Result<(), MeshRepairError> {
    // Build remapping: old index -> new index
    let mut remap: Vec<Option<usize>> = vec![None; keep_vertex.len()];
    let mut new_count = 0;

    for (i, &keep) in keep_vertex.iter().enumerate() {
        if keep {
            remap[i] = Some(new_count);
            new_count += 1;
        }
    }

    // For now, we just clear and rebuild the mesh
    // A more efficient approach would be to update in-place
    let old_n_vertices = mesh.n_vertices();
    let old_n_faces = mesh.n_faces();

    // IMPORTANT: Save face data BEFORE clearing mesh
    let mut old_face_data: Vec<Vec<VertexHandle>> = Vec::new();
    for i in 0..old_n_faces {
        let fh = FaceHandle::new(i as u32);
        let verts = mesh.face_vertices_vec(fh);
        let valid_verts: Vec<_> = verts.into_iter().filter(|vh| vh.is_valid()).collect();
        if !valid_verts.is_empty() {
            old_face_data.push(valid_verts);
        }
    }

    // Store old geometry
    let mut old_points: Vec<(VertexHandle, Point3)> = Vec::new();
    for i in 0..old_n_vertices {
        if keep_vertex[i] {
            let vh = VertexHandle::new(i as u32);
            if let Some(p) = mesh.point(vh) {
                old_points.push((vh, p));
            }
        }
    }

    // Clear mesh
    mesh.clear();

    // Rebuild with unique vertices
    let mut new_vh_map: HashMap<usize, VertexHandle> = HashMap::new();
    for (old_vh, point) in old_points {
        let new_vh = mesh.add_vertex(point);
        new_vh_map.insert(old_vh.idx_usize(), new_vh);
    }

    // Rebuild faces using SAVED data (not from mesh after clear)
    for face_verts in old_face_data {
        let mut new_face_verts: Vec<VertexHandle> = Vec::new();
        for vh in &face_verts {
            if let Some(&new_vh) = new_vh_map.get(&vh.idx_usize()) {
                new_face_verts.push(new_vh);
            }
        }

        // Only add if we have at least 3 vertices with no duplicates
        if new_face_verts.len() >= 3 {
            // Deduplicate while preserving order
            let mut seen = Vec::new();
            let mut deduped = Vec::new();
            for vh in new_face_verts {
                if !seen.contains(&vh) {
                    seen.push(vh);
                    deduped.push(vh);
                }
            }
            if deduped.len() >= 3 {
                mesh.add_face(&deduped);
            }
        }
    }

    Ok(())
}

// ============================================================================
// Remove Degenerate Faces
// ============================================================================

/// Remove faces with zero area (degenerate triangles).
/// For triangular faces, checks if all three vertices are collinear or coincident.
/// For polygonal faces, checks if the face has area close to zero.
///
/// Returns the number of degenerate faces removed.
pub fn remove_degenerate_faces(mesh: &mut RustMesh) -> Result<usize, MeshRepairError> {
    if mesh.n_faces() == 0 {
        return Ok(0);
    }

    let mut faces_to_remove: Vec<FaceHandle> = Vec::new();

    // Check each face for degeneracy
    for fh in mesh.faces() {
        let verts = mesh.face_vertices_vec(fh);
        let face_verts: Vec<_> = verts;

        if is_face_degenerate(mesh, &face_verts) {
            faces_to_remove.push(fh);
        }
    }

    // Remove degenerate faces
    for fh in &faces_to_remove {
        delete_face(mesh, *fh);
    }

    Ok(faces_to_remove.len())
}

/// Check if a face is degenerate (zero area)
fn is_face_degenerate(mesh: &RustMesh, verts: &[VertexHandle]) -> bool {
    if verts.len() < 3 {
        return true;
    }

    // For triangles, check using cross product
    if verts.len() == 3 {
        let p0 = match mesh.point(verts[0]) {
            Some(p) => p,
            None => return true,
        };
        let p1 = match mesh.point(verts[1]) {
            Some(p) => p,
            None => return true,
        };
        let p2 = match mesh.point(verts[2]) {
            Some(p) => p,
            None => return true,
        };

        // Check if area is close to zero
        let area = triangle_area(p0, p1, p2);
        return area < 1e-10;
    }

    // For polygons, compute area using triangulation from centroid
    // A face is degenerate if all vertices are collinear or within tolerance
    let mut total_area = 0.0f32;
    let centroid = compute_polygon_centroid(mesh, verts);

    for i in 0..verts.len() {
        let j = (i + 1) % verts.len();

        let pi = match mesh.point(verts[i]) {
            Some(p) => p,
            None => return true,
        };
        let pj = match mesh.point(verts[j]) {
            Some(p) => p,
            None => return true,
        };

        // Triangle from centroid
        let tri_area = triangle_area(centroid, pi, pj);
        total_area += tri_area;
    }

    total_area < 1e-10
}

/// Compute centroid of a polygon
fn compute_polygon_centroid(mesh: &RustMesh, verts: &[VertexHandle]) -> Point3 {
    if verts.is_empty() {
        return Point3::origin();
    }

    let mut sum = Vec3::zeros();
    let mut count = 0;

    for vh in verts {
        if let Some(p) = mesh.point(*vh) {
            sum += p.coords;
            count += 1;
        }
    }

    if count > 0 {
        Point3::from(sum / count as f32)
    } else {
        Point3::origin()
    }
}

/// Delete a face from the mesh
fn delete_face(mesh: &mut RustMesh, fh: FaceHandle) {
    mesh.delete_face(fh);
}

// ============================================================================
// Fix Winding Order
// ============================================================================

/// Ensure consistent face winding order across the mesh.
/// Uses the first face's normal as reference and flips faces that don't match.
///
/// Returns the number of faces that were flipped.
pub fn fix_winding_order(mesh: &mut RustMesh) -> Result<usize, MeshRepairError> {
    if mesh.n_faces() == 0 {
        return Ok(0);
    }

    // Get reference normal from first face
    let first_fh = FaceHandle::new(0);
    let reference_normal = match compute_face_normal(mesh, first_fh) {
        Some(n) => n,
        None => return Err("Cannot compute normal for first face".into()),
    };

    let mut flipped_count = 0;

    // Collect faces first to avoid borrow issues
    let faces: Vec<_> = mesh.faces().collect();

    // Check each face against reference
    for fh in faces {
        if let Some(current_normal) = compute_face_normal(mesh, fh) {
            // If normals point in opposite directions, flip the face
            if current_normal.dot(&reference_normal) < 0.0 {
                flip_face_winding(mesh, fh)?;
                flipped_count += 1;
            }
        }
    }

    Ok(flipped_count)
}

/// Compute the normal of a face
fn compute_face_normal(mesh: &RustMesh, fh: FaceHandle) -> Option<Vec3> {
    let verts = mesh.face_vertices_vec(fh);

    if verts.len() < 3 {
        return None;
    }

    let p0 = mesh.point(verts[0])?;
    let p1 = mesh.point(verts[1])?;
    let p2 = mesh.point(verts[2])?;

    Some(triangle_normal(p0, p1, p2))
}

/// Flip the winding order of a face (reverse vertex order)
fn flip_face_winding(mesh: &mut RustMesh, fh: FaceHandle) -> Result<(), MeshRepairError> {
    let mut face_verts = mesh.face_vertices_vec(fh);

    if face_verts.len() < 3 {
        return Err("Face has fewer than 3 vertices".into());
    }

    // Reverse the vertex order (but keep first vertex to maintain manifold)
    face_verts.reverse();

    // Remove old face and add new one
    // Note: In a full implementation, we'd modify in-place
    delete_face(mesh, fh);

    if face_verts.len() >= 3 {
        mesh.add_face(&face_verts);
    }

    Ok(())
}

// ============================================================================
// Merge Close Vertices
// ============================================================================

/// Merge vertices that are within the given threshold distance of each other.
///
/// # Arguments
/// * `mesh` - The mesh to repair
/// * `threshold` - Maximum distance between vertices to merge (default: 1e-5)
///
/// Returns the number of vertex pairs merged.
pub fn merge_close_vertices(mesh: &mut RustMesh, threshold: f32) -> Result<usize, MeshRepairError> {
    if mesh.n_vertices() < 2 {
        return Ok(0);
    }

    let threshold_sq = threshold * threshold;
    let mut merged_count = 0;
    let n_vertices = mesh.n_vertices();

    // Track which vertices are still valid
    let mut is_active: Vec<bool> = vec![true; n_vertices];

    // For each vertex, find and merge close neighbors
    for i in 0..n_vertices {
        if !is_active[i] {
            continue;
        }

        let vh_i = VertexHandle::new(i as u32);
        let p_i = match mesh.point(vh_i) {
            Some(p) => p,
            None => continue,
        };

        // Find all vertices close to this one
        for j in (i + 1)..n_vertices {
            if !is_active[j] {
                continue;
            }

            let vh_j = VertexHandle::new(j as u32);
            let p_j = match mesh.point(vh_j) {
                Some(p) => p,
                None => continue,
            };

            let dist_sq = (p_i - p_j).norm_squared();

            if dist_sq < threshold_sq {
                // Merge j into i
                remap_vertex_in_faces(mesh, vh_j, vh_i);
                is_active[j] = false;
                merged_count += 1;
            }
        }
    }

    // Compact if we merged anything
    if merged_count > 0 {
        compact_vertices(mesh, &is_active)?;
    }

    Ok(merged_count)
}

// ============================================================================
// Combined Repair Function
// ============================================================================

/// Perform all mesh repair operations in a sensible order.
///
/// 1. Merge close vertices (coalesce duplicates)
/// 2. Remove degenerate faces
/// 3. Fix winding order
///
/// # Arguments
/// * `mesh` - The mesh to repair
/// * `merge_threshold` - Distance threshold for merging vertices
pub fn repair_mesh(
    mesh: &mut RustMesh,
    merge_threshold: f32,
) -> Result<RepairStats, MeshRepairError> {
    let mut stats = RepairStats::default();

    // Step 1: Merge close vertices
    stats.vertices_merged = merge_close_vertices(mesh, merge_threshold)?;

    // Step 2: Remove duplicate vertices (exact positions)
    stats.duplicates_removed = remove_duplicates(mesh)?;

    // Step 3: Remove degenerate faces
    stats.degenerate_faces_removed = remove_degenerate_faces(mesh)?;

    // Step 4: Fix winding order
    stats.faces_flipped = fix_winding_order(mesh)?;

    Ok(stats)
}

/// Statistics from mesh repair operations
#[derive(Debug, Clone, Default)]
pub struct RepairStats {
    /// Number of vertex pairs merged
    pub vertices_merged: usize,
    /// Number of duplicate vertices removed
    pub duplicates_removed: usize,
    /// Number of degenerate faces removed
    pub degenerate_faces_removed: usize,
    /// Number of faces flipped to fix winding
    pub faces_flipped: usize,
}

impl std::fmt::Display for RepairStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RepairStats {{ vertices_merged: {}, duplicates_removed: {}, degenerate_faces_removed: {}, faces_flipped: {} }}",
            self.vertices_merged,
            self.duplicates_removed,
            self.degenerate_faces_removed,
            self.faces_flipped
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_mesh() -> RustMesh {
        let mut mesh = RustMesh::new();

        // Create a simple quad
        let v0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(Point3::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(Point3::new(1.0, 1.0, 0.0));
        let v3 = mesh.add_vertex(Point3::new(0.0, 1.0, 0.0));

        // Add two triangles to form a quad
        mesh.add_face(&[v0, v1, v2]);
        mesh.add_face(&[v0, v2, v3]);

        mesh
    }

    #[test]
    fn test_remove_degenerate_faces() {
        let mut mesh = create_test_mesh();
        let _initial_faces = mesh.n_faces();

        // Add a degenerate triangle (all same point)
        let v0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0)); // duplicate
        let v2 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0)); // duplicate
        mesh.add_face(&[v0, v1, v2]);

        // Add another degenerate (collinear)
        let v3 = mesh.add_vertex(Point3::new(2.0, 0.0, 0.0));
        let v4 = mesh.add_vertex(Point3::new(3.0, 0.0, 0.0));
        let v5 = mesh.add_vertex(Point3::new(4.0, 0.0, 0.0));
        mesh.add_face(&[v3, v4, v5]);

        let removed = remove_degenerate_faces(&mut mesh).unwrap();
        assert!(
            removed >= 2,
            "Should remove at least 2 degenerate faces, got {}",
            removed
        );
    }

    #[test]
    fn test_fix_winding_order() {
        let mut mesh = create_test_mesh();

        // Add a face with reversed winding
        let v0 = mesh.add_vertex(Point3::new(2.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(Point3::new(3.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(Point3::new(2.5, 1.0, 0.0));
        mesh.add_face(&[v0, v2, v1]); // Opposite normal from the reference face

        let flipped = fix_winding_order(&mut mesh).unwrap();
        // At least one face should be flipped
        assert!(flipped > 0);
    }

    #[test]
    fn test_merge_close_vertices() {
        let mut mesh = create_test_mesh();
        let initial_verts = mesh.n_vertices();

        // Add vertices very close together
        let v0 = mesh.add_vertex(Point3::new(0.001, 0.0, 0.0)); // very close to (0,0,0)
        let _v1 = mesh.add_vertex(Point3::new(0.0001, 0.0, 0.0)); // even closer
        mesh.add_face(&[VertexHandle::new(0), v0, VertexHandle::new(2)]);
        mesh.add_face(&[VertexHandle::new(0), VertexHandle::new(1), v0]);

        let merged = merge_close_vertices(&mut mesh, 0.01).unwrap();
        assert!(merged > 0, "Should merge some close vertices");
        assert!(mesh.n_vertices() <= initial_verts + 1);
    }

    #[test]
    fn test_merge_close_vertices_no_hang_on_face_scan() {
        let mut mesh = create_test_mesh();

        // Add two very close vertices and a face referencing one of them.
        let v0 = mesh.add_vertex(Point3::new(0.001, 0.0, 0.0));
        let _v1 = mesh.add_vertex(Point3::new(0.0009, 0.0, 0.0));
        mesh.add_face(&[VertexHandle::new(0), v0, VertexHandle::new(1)]);

        // This should complete quickly even if connectivity is imperfect.
        let merged = merge_close_vertices(&mut mesh, 0.01).unwrap();
        assert!(merged >= 1, "Expected at least one merge");
    }

    #[test]
    fn test_repair_stats_display() {
        let stats = RepairStats {
            vertices_merged: 5,
            duplicates_removed: 3,
            degenerate_faces_removed: 2,
            faces_flipped: 1,
        };

        let display = format!("{}", stats);
        assert!(display.contains("vertices_merged: 5"));
        assert!(display.contains("duplicates_removed: 3"));
    }

    #[test]
    fn test_empty_mesh() {
        let mut mesh = RustMesh::new();

        assert_eq!(remove_duplicates(&mut mesh).unwrap(), 0);
        assert_eq!(remove_degenerate_faces(&mut mesh).unwrap(), 0);
        assert_eq!(fix_winding_order(&mut mesh).unwrap(), 0);
        assert_eq!(merge_close_vertices(&mut mesh, 0.001).unwrap(), 0);
    }
}
