//! # PolyConnectivity
//!
//! Polygonal mesh connectivity implementation.
//! Provides iteration and circulation over mesh elements.

use crate::handles::{EdgeHandle, FaceHandle, HalfedgeHandle, VertexHandle};
use crate::soa_kernel::SoAKernel;
use crate::{Point3, Vec2, Vec3, Vec4};
use std::collections::{HashMap, VecDeque};

// ============================================================================
// High-Performance Index Iterators (no Handle overhead)
// ============================================================================

/// Vertex index iterator - Returns usize instead of VertexHandle
/// This avoids Handle creation overhead
#[derive(Debug)]
pub struct VertexIndexIter {
    current: usize,
    end: usize,
}

impl VertexIndexIter {
    #[inline]
    pub fn new(n_vertices: usize) -> Self {
        Self {
            current: 0,
            end: n_vertices,
        }
    }
}

impl Iterator for VertexIndexIter {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.end {
            let idx = self.current;
            self.current += 1;
            Some(idx)
        } else {
            None
        }
    }
}

/// Face index iterator - Returns usize instead of FaceHandle
#[derive(Debug)]
pub struct FaceIndexIter {
    current: usize,
    end: usize,
}

impl FaceIndexIter {
    #[inline]
    pub fn new(n_faces: usize) -> Self {
        Self {
            current: 0,
            end: n_faces,
        }
    }
}

impl Iterator for FaceIndexIter {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.end {
            let idx = self.current;
            self.current += 1;
            Some(idx)
        } else {
            None
        }
    }
}

pub struct RustMesh {
    kernel: SoAKernel,
}

impl Clone for RustMesh {
    fn clone(&self) -> Self {
        Self {
            kernel: self.kernel.clone(),
        }
    }
}

impl std::fmt::Debug for RustMesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RustMesh")
            .field("n_vertices", &self.kernel.n_vertices())
            .field("n_edges", &self.kernel.n_edges())
            .field("n_faces", &self.kernel.n_faces())
            .field("n_halfedges", &self.kernel.n_halfedges())
            .finish()
    }
}

impl RustMesh {
    /// Create a new empty mesh
    #[inline]
    pub fn new() -> Self {
        Self {
            kernel: SoAKernel::new(),
        }
    }

    /// Clear the mesh
    #[inline]
    pub fn clear(&mut self) {
        self.kernel.clear();
    }

    // --- Vertex operations ---

    /// Add a vertex at the given position
    #[inline]
    pub fn add_vertex(&mut self, point: Point3) -> VertexHandle {
        self.kernel.add_vertex(point)
    }

    /// Get the number of vertices
    #[inline]
    pub fn n_vertices(&self) -> usize {
        self.kernel.n_vertices()
    }

    /// Get the number of active vertices referenced by at least one face.
    pub fn active_vertex_count(&self) -> usize {
        let mut used = vec![false; self.n_vertices()];

        for fh in self.faces() {
            if self.face_halfedge_handle(fh).is_none() {
                continue;
            }

            for vh in self.face_vertices_vec(fh) {
                let idx = vh.idx_usize();
                if idx < used.len() {
                    used[idx] = true;
                }
            }
        }

        let active = used.into_iter().filter(|used| *used).count();
        if active == 0 {
            self.vertices()
                .filter(|&vh| self.halfedge_handle(vh).is_some())
                .count()
        } else {
            active
        }
    }

    /// Get vertex position by handle
    #[inline]
    pub fn point(&self, vh: VertexHandle) -> Option<Point3> {
        self.kernel.point(vh.idx_usize())
    }

    /// Get vertex position by index (for internal use)
    #[inline]
    pub unsafe fn point_unchecked(&self, idx: usize) -> Point3 {
        self.kernel.point_unchecked(idx)
    }

    /// Set vertex position
    #[inline]
    pub fn set_point(&mut self, vh: VertexHandle, point: Point3) {
        self.kernel.set_point(vh.idx_usize(), point);
    }

    // --- SIMD-friendly access ---

    /// Get x coordinates slice
    #[inline]
    pub fn x(&self) -> &[f32] {
        self.kernel.x_slice()
    }

    /// Get y coordinates slice
    #[inline]
    pub fn y(&self) -> &[f32] {
        self.kernel.y_slice()
    }

    /// Get z coordinates slice
    #[inline]
    pub fn z(&self) -> &[f32] {
        self.kernel.z_slice()
    }

    /// Get x pointer for SIMD
    #[inline]
    pub fn x_ptr(&self) -> *const f32 {
        self.kernel.x_ptr()
    }

    /// Get y pointer for SIMD
    #[inline]
    pub fn y_ptr(&self) -> *const f32 {
        self.kernel.y_ptr()
    }

    /// Get z pointer for SIMD
    #[inline]
    pub fn z_ptr(&self) -> *const f32 {
        self.kernel.z_ptr()
    }

    // --- Vertex iteration ---

    /// Get an iterator over all vertex indices (fastest)
    #[inline]
    pub fn vertex_indices(&self) -> VertexIndexIter {
        VertexIndexIter::new(self.n_vertices())
    }

    /// Iterate over all vertex handles
    #[inline]
    pub fn vertices(&self) -> impl Iterator<Item = VertexHandle> + '_ {
        (0..self.n_vertices()).map(|i| VertexHandle::from_usize(i))
    }

    // --- Edge operations ---

    /// Add an edge between two vertices
    #[inline]
    pub fn add_edge(&mut self, v0: VertexHandle, v1: VertexHandle) -> HalfedgeHandle {
        self.kernel.add_edge(v0, v1)
    }

    /// Get the number of edges
    #[inline]
    pub fn n_edges(&self) -> usize {
        self.kernel.n_edges()
    }

    // --- Face operations ---

    /// Add a face from a list of vertex handles
    pub fn add_face(&mut self, vertices: &[VertexHandle]) -> Option<FaceHandle> {
        if vertices.len() < 3 {
            return None;
        }

        let n = vertices.len();
        let mut halfedges: Vec<HalfedgeHandle> = Vec::with_capacity(n);

        // First: create all edges and track them
        for i in 0..n {
            let start = vertices[i];
            let end = vertices[(i + 1) % n];

            // Create the halfedge from start to end
            let he = self.add_edge(start, end);
            // The halfedge used by this face must be free (no face assigned yet)
            if self.kernel.face_handle(he).is_some() {
                return None;
            }
            halfedges.push(he);
        }

        // Second: set next/prev pointers for ALL halfedges in this face
        // Each halfedge points to the next one in the face cycle
        for i in 0..n {
            let curr = halfedges[i];
            let next_in_face = halfedges[(i + 1) % n];

            self.kernel.set_next_halfedge_handle(curr, next_in_face);
            self.kernel.set_prev_halfedge_handle(next_in_face, curr);
        }

        let fh = self.kernel.add_face(Some(halfedges[0]));

        // Set face handle for all halfedges
        for &he in &halfedges {
            self.kernel.set_face_handle(he, fh);
        }

        // Set vertex halfedge handles - use outgoing halfedge from each vertex
        for (i, &vh) in vertices.iter().enumerate() {
            // halfedges[i] goes FROM vertices[i] TO vertices[(i+1)%n]
            // This is the outgoing halfedge from vh, which is what circulators expect
            self.kernel.set_halfedge_handle(vh, halfedges[i]);
        }

        Some(fh)
    }

    /// Compatibility wrapper for older OpenMesh parity import code paths.
    ///
    /// The current implementation keeps parity-specific anchor behavior by
    /// skipping any post-insert boundary normalization. Callers that need the
    /// normalized vertex anchors should invoke
    /// `normalize_boundary_halfedge_handles()` explicitly afterwards.
    pub fn add_face_openmesh_parity(&mut self, vertices: &[VertexHandle]) -> Option<FaceHandle> {
        let fh = self.add_face(vertices)?;
        let anchor = self
            .face_halfedge_handle(fh)
            .map(|heh| self.prev_halfedge_handle(heh));
        self.set_face_halfedge_handle(fh, anchor);
        Some(fh)
    }

    /// Get the number of faces (includes deleted faces)
    #[inline]
    pub fn n_faces(&self) -> usize {
        self.kernel.n_faces()
    }

    /// Get the number of active (non-deleted) faces
    #[inline]
    pub fn n_active_faces(&self) -> usize {
        self.kernel.n_active_faces()
    }

    /// Get the number of halfedges
    #[inline]
    pub fn n_halfedges(&self) -> usize {
        self.kernel.n_halfedges()
    }

    // --- Face iteration ---

    /// Get an iterator over all face indices
    #[inline]
    pub fn face_indices(&self) -> FaceIndexIter {
        FaceIndexIter::new(self.n_faces())
    }

    /// Iterate over all face handles
    #[inline]
    pub fn faces(&self) -> impl Iterator<Item = FaceHandle> + '_ {
        (0..self.n_faces()).map(|i| FaceHandle::from_usize(i))
    }

    // --- Connectivity queries ---

    /// Get the halfedge handle from a vertex
    #[inline]
    pub fn halfedge_handle(&self, vh: VertexHandle) -> Option<HalfedgeHandle> {
        self.kernel.halfedge_handle(vh)
    }

    /// Get the edge handle from a halfedge
    #[inline]
    pub fn edge_handle(&self, heh: HalfedgeHandle) -> EdgeHandle {
        self.kernel.edge_handle(heh)
    }

    /// Get a halfedge from an edge (0 or 1)
    #[inline]
    pub fn edge_halfedge_handle(&self, eh: EdgeHandle, idx: usize) -> HalfedgeHandle {
        self.kernel.edge_halfedge_handle(eh, idx)
    }

    /// Get the face handle from a halfedge
    #[inline]
    pub fn face_handle(&self, heh: HalfedgeHandle) -> Option<FaceHandle> {
        self.kernel.face_handle(heh)
    }

    #[inline]
    pub fn is_vertex_deleted(&self, vh: VertexHandle) -> bool {
        self.kernel.is_vertex_deleted(vh)
    }

    #[inline]
    pub fn is_halfedge_deleted(&self, heh: HalfedgeHandle) -> bool {
        self.kernel.is_halfedge_deleted(heh)
    }

    #[inline]
    pub fn is_edge_deleted(&self, eh: EdgeHandle) -> bool {
        self.kernel.is_edge_deleted(eh)
    }

    #[inline]
    pub fn is_face_deleted(&self, fh: FaceHandle) -> bool {
        self.kernel.is_face_deleted(fh)
    }

    /// Check if a halfedge is a boundary
    #[inline]
    pub fn is_boundary(&self, heh: HalfedgeHandle) -> bool {
        self.kernel.is_boundary(heh)
    }

    /// Validate halfedge structure integrity
    /// Returns Ok if valid, Err with message if issues found
    pub fn validate(&self) -> Result<(), String> {
        let n_vertices = self.n_vertices();
        let n_edges = self.n_edges();
        let n_faces = self.n_faces();
        let n_halfedges = self.n_halfedges();

        // Euler formula check: V - E + F = 2 for closed manifold
        // For meshes with boundary: V - E + F = 1 + B (B = boundary components)
        let euler = n_vertices as i32 - n_edges as i32 + n_faces as i32;
        println!(
            "Euler characteristic: {} (V={}, E={}, F={})",
            euler, n_vertices, n_edges, n_faces
        );

        // Check: halfedges should be 2 * edges
        if n_halfedges != 2 * n_edges {
            return Err(format!(
                "Halfedge count mismatch: {} != 2 * {}",
                n_halfedges, n_edges
            ));
        }

        // Check each vertex has valid halfedge
        for vh in 0..n_vertices {
            let vh = VertexHandle::new(vh as u32);
            if let Some(heh) = self.halfedge_handle(vh) {
                if !heh.is_valid() {
                    return Err(format!("Vertex {:?} has invalid halfedge", vh));
                }
            }
        }

        // Check halfedge cycles (prevent infinite loops)
        for fh in 0..n_faces {
            let fh = FaceHandle::new(fh as u32);
            if let Some(start_heh) = self.face_halfedge_handle(fh) {
                let mut count = 0;
                let mut current = start_heh;
                loop {
                    count += 1;
                    if count > 64 {
                        return Err(format!("Face {:?} has >64 halfedges - cycle broken!", fh));
                    }
                    current = self.next_halfedge_handle(current);
                    if current == start_heh || !current.is_valid() {
                        break;
                    }
                }
                if count < 3 {
                    return Err(format!("Face {:?} has {} halfedges - too few!", fh, count));
                }
            }
        }

        // Check vertex rings (prevent infinite loops)
        for vh in 0..n_vertices {
            let vh = VertexHandle::new(vh as u32);
            if let Some(start_heh) = self.halfedge_handle(vh) {
                let mut count = 0;
                let mut current = start_heh;
                loop {
                    count += 1;
                    if count > 64 {
                        return Err(format!("Vertex {:?} has >64 halfedges - cycle broken!", vh));
                    }
                    // Move to next halfedge around vertex
                    let opposite = self.opposite_halfedge_handle(current);
                    current = self.next_halfedge_handle(opposite);
                    if current == start_heh || !current.is_valid() {
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    /// Get the to-vertex of a halfedge
    #[inline]
    pub fn to_vertex_handle(&self, heh: HalfedgeHandle) -> VertexHandle {
        self.kernel.to_vertex_handle(heh)
    }

    /// Get the from-vertex of a halfedge
    #[inline]
    pub fn from_vertex_handle(&self, heh: HalfedgeHandle) -> VertexHandle {
        if !heh.is_valid() {
            return VertexHandle::invalid();
        }

        let opp = heh.opposite();
        if opp.idx_usize() >= self.n_halfedges() {
            return VertexHandle::invalid();
        }

        self.kernel.to_vertex_handle(opp)
    }

    /// Get the opposite halfedge (across the edge)
    #[inline]
    pub fn opposite_halfedge_handle(&self, heh: HalfedgeHandle) -> HalfedgeHandle {
        if !heh.is_valid() {
            return heh;
        }

        let opp = heh.opposite();
        if opp.idx_usize() < self.n_halfedges() {
            opp
        } else {
            heh
        }
    }

    /// Get the next halfedge in the cycle
    #[inline]
    pub fn next_halfedge_handle(&self, heh: HalfedgeHandle) -> HalfedgeHandle {
        self.kernel.next_halfedge_handle(heh).unwrap_or(heh)
    }

    /// Get the next outgoing halfedge around the same source vertex.
    ///
    /// This matches the traversal pattern used elsewhere in the crate:
    /// `next(opposite(heh))`. If the computed halfedge no longer originates
    /// from the same vertex, keep the current halfedge so callers can stop
    /// without jumping into a different one-ring.
    #[inline]
    pub fn next_outgoing_halfedge(&self, heh: HalfedgeHandle) -> HalfedgeHandle {
        let from = self.from_vertex_handle(heh);
        let next = self.next_halfedge_handle(self.opposite_halfedge_handle(heh));
        if self.from_vertex_handle(next) == from {
            next
        } else {
            heh
        }
    }

    /// Get the previous halfedge in the cycle
    #[inline]
    pub fn prev_halfedge_handle(&self, heh: HalfedgeHandle) -> HalfedgeHandle {
        self.kernel.prev_halfedge_handle(heh).unwrap_or(heh)
    }

    /// Get the halfedge handle associated with a face
    #[inline]
    pub fn face_halfedge_handle(&self, fh: FaceHandle) -> Option<HalfedgeHandle> {
        self.kernel.face_halfedge_handle(fh)
    }

    #[inline]
    fn set_face_halfedge_handle(&mut self, fh: FaceHandle, heh: Option<HalfedgeHandle>) {
        self.kernel.set_face_halfedge_handle(fh, heh);
    }

    #[inline]
    fn clear_face_handle(&mut self, heh: HalfedgeHandle) {
        self.kernel.clear_face_handle(heh);
    }

    #[inline]
    fn clear_halfedge_handle(&mut self, vh: VertexHandle) {
        self.kernel.clear_halfedge_handle(vh);
    }

    // --- SIMD-optimized operations ---

    /// Compute bounding box (optimized)
    #[inline]
    pub fn bounding_box(&self) -> (f32, f32, f32, f32, f32, f32) {
        self.kernel.bounding_box()
    }

    /// Compute bounding box using NEON SIMD
    #[inline]
    pub unsafe fn bounding_box_simd(&self) -> (f32, f32, f32, f32, f32, f32) {
        self.kernel.bounding_box_simd()
    }

    /// Compute centroid (optimized)
    #[inline]
    pub fn centroid(&self) -> (f32, f32, f32) {
        self.kernel.centroid()
    }

    /// Compute centroid using NEON SIMD
    #[inline]
    pub unsafe fn centroid_simd(&self) -> (f32, f32, f32) {
        self.kernel.centroid_simd()
    }

    /// Compute vertex sum using NEON SIMD
    #[inline]
    pub unsafe fn vertex_sum_simd(&self) -> (f32, f32, f32) {
        self.kernel.vertex_sum_simd()
    }

    // =========================================================================
    // Edge Collapse (Halfedge Collapse)
    // =========================================================================

    /// Check if an edge collapse is legal
    /// Returns true if the halfedge can be collapsed without creating topological issues.
    ///
    /// Checks:
    /// 1. For triangle faces: link condition (1-ring intersection)
    /// 2. For polygon faces: no duplicate edges after collapse
    /// 3. Boundary consistency
    pub fn is_collapse_ok(&self, heh: HalfedgeHandle) -> bool {
        if !heh.is_valid() {
            return false;
        }
        if self.is_halfedge_deleted(heh) || self.is_edge_deleted(self.edge_handle(heh)) {
            return false;
        }

        let v0 = self.from_vertex_handle(heh); // Vertex to be removed
        let v1 = self.to_vertex_handle(heh); // Remaining vertex
        if !v0.is_valid()
            || !v1.is_valid()
            || self.is_vertex_deleted(v0)
            || self.is_vertex_deleted(v1)
        {
            return false;
        }

        // Both vertices must have valid halfedges
        if self.halfedge_handle(v0).is_none() || self.halfedge_handle(v1).is_none() {
            return false;
        }

        let heh_opp = self.opposite_halfedge_handle(heh);

        // Get adjacent faces
        let fh_left = self.face_handle(heh);
        let fh_right = self.face_handle(heh_opp);

        // At least one face must exist
        if fh_left.is_none() && fh_right.is_none() {
            return false;
        }

        // Match OpenMesh TriConnectivity: collapsing through a boundary wedge
        // would create invalid local boundary topology.
        if fh_left.is_some() {
            let h1 = self.next_halfedge_handle(heh);
            let h2 = self.next_halfedge_handle(h1);
            if self.is_boundary(self.opposite_halfedge_handle(h1))
                && self.is_boundary(self.opposite_halfedge_handle(h2))
            {
                return false;
            }
        }

        if fh_right.is_some() {
            let h1 = self.next_halfedge_handle(heh_opp);
            let h2 = self.next_halfedge_handle(h1);
            if self.is_boundary(self.opposite_halfedge_handle(h1))
                && self.is_boundary(self.opposite_halfedge_handle(h2))
            {
                return false;
            }
        }

        // Check for duplicate edges: after collapse, v0's neighbors become v1's neighbors.
        // If any neighbor of v0 is already a neighbor of v1 (and not in an adjacent face),
        // collapsing would create a duplicate edge.
        //
        // Vertices that are allowed to be shared are those in the faces adjacent to the edge.
        let mut allowed_shared: Vec<VertexHandle> = Vec::new();
        if let Some(fh) = fh_left {
            let h1 = self.next_halfedge_handle(heh);
            let h2 = self.next_halfedge_handle(h1);
            if self.next_halfedge_handle(h2) == heh {
                let vh = self.to_vertex_handle(h1);
                if vh != v0 && vh != v1 {
                    allowed_shared.push(vh);
                }
            } else {
                for vh in self.face_vertices_vec(fh) {
                    if vh != v0 && vh != v1 && !allowed_shared.contains(&vh) {
                        allowed_shared.push(vh);
                    }
                }
            }
        }
        if let Some(fh) = fh_right {
            let h1 = self.next_halfedge_handle(heh_opp);
            let h2 = self.next_halfedge_handle(h1);
            if self.next_halfedge_handle(h2) == heh_opp {
                let vh = self.to_vertex_handle(h1);
                if vh != v0 && vh != v1 && !allowed_shared.contains(&vh) {
                    allowed_shared.push(vh);
                }
            } else {
                for vh in self.face_vertices_vec(fh) {
                    if vh != v0 && vh != v1 && !allowed_shared.contains(&vh) {
                        allowed_shared.push(vh);
                    }
                }
            }
        }

        let neighbors_v1 = self.collect_vertex_neighbors_fast(v1);

        if let Some(start_heh) = self.halfedge_handle(v0) {
            let mut current_heh = start_heh;
            let max_iterations = self.n_halfedges().max(1);

            for _ in 0..max_iterations {
                let neighbor = self.to_vertex_handle(current_heh);
                if neighbor != v1
                    && !allowed_shared.contains(&neighbor)
                    && neighbors_v1.contains(&neighbor)
                {
                    return false;
                }

                let next_heh = self.next_outgoing_halfedge(current_heh);
                if !next_heh.is_valid() || next_heh == current_heh || next_heh == start_heh {
                    break;
                }

                current_heh = next_heh;
            }
        }

        // For triangle faces: check that the opposite vertices are distinct
        if fh_left.is_some() && fh_right.is_some() {
            let vl = self.to_vertex_handle(self.next_halfedge_handle(heh));
            let vr = self.to_vertex_handle(self.next_halfedge_handle(heh_opp));
            if vl == vr {
                return false;
            }
        }

        // Boundary check: both boundary vertices can only collapse along a boundary edge
        let v0_boundary = self.is_boundary_vertex(v0);
        let v1_boundary = self.is_boundary_vertex(v1);
        let edge_boundary = fh_left.is_none() || fh_right.is_none();

        if v0_boundary && v1_boundary && !edge_boundary {
            return false;
        }

        // Match OpenMesh PolyConnectivity: prevent collapses that would
        // degenerate a backside polygon sharing both adjacent opposite edges.
        if let Some(fh) = fh_left {
            if self.face_vertices_vec(fh).len() == 3 {
                let one = self.opposite_halfedge_handle(self.next_halfedge_handle(heh));
                let two = self.opposite_halfedge_handle(
                    self.next_halfedge_handle(self.next_halfedge_handle(heh)),
                );
                if let (Some(face_one), Some(face_two)) =
                    (self.face_handle(one), self.face_handle(two))
                {
                    if face_one == face_two && self.face_vertices_vec(face_one).len() != 3 {
                        return false;
                    }
                }
            }
        }

        if let Some(fh) = fh_right {
            if self.face_vertices_vec(fh).len() == 3 {
                let one = self.opposite_halfedge_handle(self.next_halfedge_handle(heh_opp));
                let two = self.opposite_halfedge_handle(
                    self.next_halfedge_handle(self.next_halfedge_handle(heh_opp)),
                );
                if let (Some(face_one), Some(face_two)) =
                    (self.face_handle(one), self.face_handle(two))
                {
                    if face_one == face_two && self.face_vertices_vec(face_one).len() != 3 {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Check if a vertex is on the boundary (has at least one boundary halfedge)
    pub fn is_boundary_vertex(&self, vh: VertexHandle) -> bool {
        if !vh.is_valid() || self.is_vertex_deleted(vh) {
            return false;
        }

        if let Some(start_heh) = self.halfedge_handle(vh) {
            let mut current = start_heh;
            let max_iterations = self.n_halfedges().max(1);

            for _ in 0..max_iterations {
                // Check both the outgoing halfedge and its opposite
                if self.is_boundary(current) {
                    return true;
                }
                let opp = self.opposite_halfedge_handle(current);
                if self.is_boundary(opp) {
                    return true;
                }

                let next = self.next_outgoing_halfedge(current);
                if !next.is_valid() || next == current || next == start_heh {
                    break;
                }

                current = next;
            }
        }
        false
    }

    #[inline]
    pub fn vertex_face_count_at_least(&self, vh: VertexHandle, min_faces: usize) -> bool {
        if min_faces == 0 {
            return true;
        }
        if !vh.is_valid() || self.is_vertex_deleted(vh) {
            return false;
        }

        let Some(start_heh) = self.halfedge_handle(vh) else {
            return false;
        };

        let mut current_heh = start_heh;
        let mut faces_seen = 0usize;
        let max_iterations = self.n_halfedges().max(1);

        for _ in 0..max_iterations {
            if self.face_handle(current_heh).is_some() {
                faces_seen += 1;
                if faces_seen >= min_faces {
                    return true;
                }
            }

            let next_heh = self.next_outgoing_halfedge(current_heh);
            if !next_heh.is_valid() || next_heh == current_heh || next_heh == start_heh {
                break;
            }

            current_heh = next_heh;
        }

        false
    }

    fn collect_vertex_neighbors_fast(&self, vh: VertexHandle) -> Vec<VertexHandle> {
        if !vh.is_valid() || self.is_vertex_deleted(vh) {
            return Vec::new();
        }

        let Some(start_heh) = self.halfedge_handle(vh) else {
            return Vec::new();
        };

        let mut neighbors = Vec::new();
        let mut current_heh = start_heh;
        let max_iterations = self.n_halfedges().max(1);

        for _ in 0..max_iterations {
            neighbors.push(self.to_vertex_handle(current_heh));

            let next_heh = self.next_outgoing_halfedge(current_heh);
            if !next_heh.is_valid() || next_heh == current_heh || next_heh == start_heh {
                break;
            }

            current_heh = next_heh;
        }

        neighbors
    }

    /// Collapse a halfedge: move v0 to v1 and remove v0 and adjacent faces
    /// Returns Ok if successful, Err with message if failed
    pub fn collapse(&mut self, heh: HalfedgeHandle) -> Result<(), &'static str> {
        if !self.is_collapse_ok(heh) {
            return Err("Collapse not legal");
        }

        let h0 = heh;
        let h1 = self.next_halfedge_handle(h0);
        let o0 = self.opposite_halfedge_handle(h0);
        let o1 = self.next_halfedge_handle(o0);

        self.collapse_edge_local(h0);

        if !self.is_halfedge_deleted(h1)
            && !self.is_edge_deleted(self.edge_handle(h1))
            && self.next_halfedge_handle(self.next_halfedge_handle(h1)) == h1
        {
            self.collapse_loop_local(self.next_halfedge_handle(h1));
        }
        if !self.is_halfedge_deleted(o1)
            && !self.is_edge_deleted(self.edge_handle(o1))
            && self.next_halfedge_handle(self.next_halfedge_handle(o1)) == o1
        {
            self.collapse_loop_local(o1);
        }

        Ok(())
    }

    fn collapse_edge_local(&mut self, heh: HalfedgeHandle) {
        let h = heh;
        let hn = self.next_halfedge_handle(h);
        let hp = self.prev_halfedge_handle(h);

        let o = self.opposite_halfedge_handle(h);
        let on = self.next_halfedge_handle(o);
        let op = self.prev_halfedge_handle(o);

        let fh = self.face_handle(h);
        let fo = self.face_handle(o);

        let vh = self.to_vertex_handle(h);
        let vo = self.to_vertex_handle(o);

        for heh_idx in 0..self.n_halfedges() {
            let incoming = HalfedgeHandle::new(heh_idx as u32);
            if self.is_halfedge_deleted(incoming)
                || self.is_edge_deleted(self.edge_handle(incoming))
            {
                continue;
            }
            if self.to_vertex_handle(incoming) == vo {
                self.kernel.set_halfedge_to_vertex(incoming, vh);
            }
        }

        self.kernel.set_next_halfedge_handle(hp, hn);
        self.kernel.set_next_halfedge_handle(op, on);

        if let Some(fh) = fh {
            self.set_face_halfedge_handle(fh, Some(hn));
        }
        if let Some(fo) = fo {
            self.set_face_halfedge_handle(fo, Some(on));
        }

        if self.halfedge_handle(vh) == Some(o) {
            self.kernel.set_halfedge_handle(vh, hn);
        }
        self.adjust_outgoing_halfedge(vh);
        self.clear_halfedge_handle(vo);

        self.kernel.mark_edge_deleted(self.edge_handle(h), true);
        self.kernel.mark_vertex_deleted(vo, true);
        self.kernel.mark_halfedge_deleted(h, true);
        self.kernel.mark_halfedge_deleted(o, true);
    }

    fn collapse_loop_local(&mut self, heh: HalfedgeHandle) {
        if !heh.is_valid()
            || self.is_halfedge_deleted(heh)
            || self.is_edge_deleted(self.edge_handle(heh))
        {
            return;
        }

        let h0 = heh;
        let h1 = self.next_halfedge_handle(h0);

        let o0 = self.opposite_halfedge_handle(h0);
        let o1 = self.opposite_halfedge_handle(h1);

        let v0 = self.to_vertex_handle(h0);
        let v1 = self.to_vertex_handle(h1);

        let fh = self.face_handle(h0);
        let fo = self.face_handle(o0);

        if self.next_halfedge_handle(h1) != h0 || h1 == o0 {
            return;
        }

        self.kernel
            .set_next_halfedge_handle(h1, self.next_halfedge_handle(o0));
        self.kernel
            .set_next_halfedge_handle(self.prev_halfedge_handle(o0), h1);

        match fo {
            Some(fo) => self.kernel.set_face_handle(h1, fo),
            None => self.clear_face_handle(h1),
        }

        self.kernel.set_halfedge_handle(v0, h1);
        self.adjust_outgoing_halfedge(v0);
        self.kernel.set_halfedge_handle(v1, o1);
        self.adjust_outgoing_halfedge(v1);

        if let Some(fo) = fo {
            if self.face_halfedge_handle(fo) == Some(o0) {
                self.set_face_halfedge_handle(fo, Some(h1));
            }
        }

        if let Some(fh) = fh {
            self.set_face_halfedge_handle(fh, None);
            self.kernel.mark_face_deleted(fh, true);
        }
        self.kernel.mark_edge_deleted(self.edge_handle(h0), true);
        self.kernel.mark_halfedge_deleted(h0, true);
        self.kernel.mark_halfedge_deleted(o0, true);
    }

    fn adjust_outgoing_halfedge(&mut self, vh: VertexHandle) {
        if !vh.is_valid() || self.is_vertex_deleted(vh) {
            self.clear_halfedge_handle(vh);
            return;
        }

        let mut first = None;
        let mut boundary = None;

        for heh_idx in 0..self.n_halfedges() {
            let heh = HalfedgeHandle::new(heh_idx as u32);
            if self.is_halfedge_deleted(heh) || self.is_edge_deleted(self.edge_handle(heh)) {
                continue;
            }
            if self.from_vertex_handle(heh) != vh {
                continue;
            }

            if first.is_none() {
                first = Some(heh);
            }
            if self.is_boundary(heh) {
                boundary = Some(heh);
                break;
            }
        }

        match boundary.or(first) {
            Some(heh) => self.kernel.set_halfedge_handle(vh, heh),
            None => self.clear_halfedge_handle(vh),
        }
    }

    /// Normalize vertex anchor halfedges so boundary vertices prefer a boundary
    /// outgoing halfedge, matching the legacy import contract used by IO code.
    pub fn normalize_boundary_halfedge_handles(&mut self) {
        let vertices: Vec<VertexHandle> = self.vertices().collect();
        for vh in vertices {
            self.adjust_outgoing_halfedge(vh);
        }
    }

    /// Redirect all halfedges that reference from_vertex to reference to_vertex
    #[allow(dead_code)]
    fn redirect_halfedges(
        &mut self,
        from_vertex: VertexHandle,
        to_vertex: VertexHandle,
    ) -> Result<(), &'static str> {
        // Get all halfedges and update those that reference from_vertex
        let n_halfedges = self.n_halfedges();

        for heh_idx in 0..n_halfedges {
            let heh = HalfedgeHandle::new(heh_idx as u32);

            // Check if this halfedge's to_vertex is from_vertex
            let to_vh = self.to_vertex_handle(heh);
            if to_vh == from_vertex {
                // Update the to_vertex to to_vertex
                // This requires modifying the halfedge data directly
                self.kernel.set_halfedge_to_vertex(heh, to_vertex);
            }
        }

        Ok(())
    }

    /// Delete a face from the mesh
    pub fn delete_face(&mut self, fh: FaceHandle) {
        // Delete the face in the kernel
        // This clears the face_handle references in all halfedges of this face
        // but preserves the halfedge connectivity (next/prev pointers)
        // so that vertex circulators can still traverse around vertices
        self.kernel.delete_face(fh);
    }

    /// Get all halfedges of a face
    #[allow(dead_code)]
    fn get_face_halfedges(&self, fh: FaceHandle) -> Vec<HalfedgeHandle> {
        let mut result = Vec::new();
        if let Some(start_heh) = self.kernel.face(fh).and_then(|f| f.halfedge_handle) {
            let mut current = start_heh;
            loop {
                result.push(current);
                let next = self.next_halfedge_handle(current);
                if !next.is_valid() || next == start_heh {
                    break;
                }
                current = next;
            }
        }
        result
    }

    /// Delete a vertex from the mesh
    pub fn delete_vertex(&mut self, vh: VertexHandle) {
        self.kernel.delete_vertex(vh);
    }

    /// Delete an edge from the mesh
    pub fn delete_edge(&mut self, eh: EdgeHandle) {
        self.kernel.delete_edge(eh);
    }

    /// Rebuild the mesh from currently active faces and vertices.
    ///
    /// This mirrors OpenMesh's `garbage_collection()` at a higher level:
    /// deleted elements are discarded and a clean connectivity structure is rebuilt.
    pub fn garbage_collection(&mut self) {
        let old_vertex_count = self.n_vertices();
        if old_vertex_count == 0 {
            return;
        }

        let preserve_normals = self.has_vertex_normals();
        let preserve_colors = self.has_vertex_colors();
        let preserve_texcoords = self.has_vertex_texcoords();

        let mut used_vertices = vec![false; old_vertex_count];
        let mut face_loops: Vec<Vec<VertexHandle>> = Vec::new();

        for fh in self.faces() {
            let vertices = self.sanitized_face_vertices(fh);
            if vertices.len() < 3 {
                continue;
            }

            for vh in &vertices {
                let idx = vh.idx_usize();
                if idx < used_vertices.len() {
                    used_vertices[idx] = true;
                }
            }
            face_loops.push(vertices);
        }

        if face_loops.is_empty() {
            self.clear();
            return;
        }

        let mut remap: Vec<Option<usize>> = vec![None; old_vertex_count];
        let mut positions: Vec<Point3> = Vec::new();
        let mut normals: Vec<Vec3> = Vec::new();
        let mut colors: Vec<Vec4> = Vec::new();
        let mut texcoords: Vec<Vec2> = Vec::new();

        for (idx, &used) in used_vertices.iter().enumerate() {
            if !used {
                continue;
            }

            let Some(point) = self.point_by_index(idx) else {
                continue;
            };

            remap[idx] = Some(positions.len());
            positions.push(point);

            if preserve_normals {
                normals.push(self.vertex_normal_by_index(idx).unwrap_or(Vec3::zeros()));
            }
            if preserve_colors {
                colors.push(
                    self.vertex_color_by_index(idx)
                        .unwrap_or(Vec4::new(1.0, 1.0, 1.0, 1.0)),
                );
            }
            if preserve_texcoords {
                texcoords.push(self.vertex_texcoord_by_index(idx).unwrap_or(Vec2::zeros()));
            }
        }

        let mut rebuilt = RustMesh::new();
        for point in &positions {
            rebuilt.add_vertex(*point);
        }

        if preserve_normals {
            rebuilt.request_vertex_normals();
            for (idx, normal) in normals.iter().enumerate() {
                rebuilt.set_vertex_normal_by_index(idx, *normal);
            }
        }
        if preserve_colors {
            rebuilt.request_vertex_colors();
            for (idx, color) in colors.iter().enumerate() {
                rebuilt.set_vertex_color_by_index(idx, *color);
            }
        }
        if preserve_texcoords {
            rebuilt.request_vertex_texcoords();
            for (idx, texcoord) in texcoords.iter().enumerate() {
                rebuilt.set_vertex_texcoord_by_index(idx, *texcoord);
            }
        }

        let remapped_faces: Vec<Vec<VertexHandle>> = face_loops
            .into_iter()
            .map(|face| {
                dedupe_face_vertices(
                    face.into_iter()
                        .filter_map(|vh| remap.get(vh.idx_usize()).and_then(|idx| *idx))
                        .map(VertexHandle::from_usize)
                        .collect(),
                )
            })
            .filter(|face: &Vec<VertexHandle>| face.len() >= 3)
            .collect();

        if remapped_faces.iter().all(|face| face.len() == 3) {
            rebuild_oriented_triangles(&mut rebuilt, &remapped_faces);
        } else {
            rebuild_polygon_faces(&mut rebuilt, &remapped_faces);
        }

        *self = rebuilt;
    }

    #[allow(dead_code)]
    fn rebuild_preserving_vertex_indices(&mut self, faces: &[Vec<VertexHandle>]) {
        let old_vertex_count = self.n_vertices();
        let preserve_normals = self.has_vertex_normals();
        let preserve_colors = self.has_vertex_colors();
        let preserve_texcoords = self.has_vertex_texcoords();

        let positions: Vec<Point3> = (0..old_vertex_count)
            .map(|idx| self.point_by_index(idx).unwrap_or(Point3::origin()))
            .collect();
        let normals: Vec<Vec3> = if preserve_normals {
            (0..old_vertex_count)
                .map(|idx| self.vertex_normal_by_index(idx).unwrap_or(Vec3::zeros()))
                .collect()
        } else {
            Vec::new()
        };
        let colors: Vec<Vec4> = if preserve_colors {
            (0..old_vertex_count)
                .map(|idx| {
                    self.vertex_color_by_index(idx)
                        .unwrap_or(Vec4::new(1.0, 1.0, 1.0, 1.0))
                })
                .collect()
        } else {
            Vec::new()
        };
        let texcoords: Vec<Vec2> = if preserve_texcoords {
            (0..old_vertex_count)
                .map(|idx| self.vertex_texcoord_by_index(idx).unwrap_or(Vec2::zeros()))
                .collect()
        } else {
            Vec::new()
        };

        let mut rebuilt = RustMesh::new();
        for point in positions {
            rebuilt.add_vertex(point);
        }

        if preserve_normals {
            rebuilt.request_vertex_normals();
            for (idx, normal) in normals.iter().enumerate() {
                rebuilt.set_vertex_normal_by_index(idx, *normal);
            }
        }
        if preserve_colors {
            rebuilt.request_vertex_colors();
            for (idx, color) in colors.iter().enumerate() {
                rebuilt.set_vertex_color_by_index(idx, *color);
            }
        }
        if preserve_texcoords {
            rebuilt.request_vertex_texcoords();
            for (idx, texcoord) in texcoords.iter().enumerate() {
                rebuilt.set_vertex_texcoord_by_index(idx, *texcoord);
            }
        }

        rebuild_polygon_faces(&mut rebuilt, faces);

        *self = rebuilt;
    }

    // =========================================================================
    // Vertex attributes
    // =========================================================================

    /// Request vertex normals
    pub fn request_vertex_normals(&mut self) {
        self.kernel.request_vertex_normals();
    }

    /// Check if vertex normals are available
    pub fn has_vertex_normals(&self) -> bool {
        self.kernel.has_vertex_normals()
    }

    /// Get vertex normal
    pub fn normal(&self, vh: VertexHandle) -> Option<Vec3> {
        self.kernel.vertex_normal(vh)
    }

    /// Set vertex normal
    pub fn set_normal(&mut self, vh: VertexHandle, n: Vec3) {
        self.kernel.set_vertex_normal(vh, n);
    }

    /// Request vertex colors
    pub fn request_vertex_colors(&mut self) {
        self.kernel.request_vertex_colors();
    }

    /// Check if vertex colors are available
    pub fn has_vertex_colors(&self) -> bool {
        self.kernel.has_vertex_colors()
    }

    /// Get vertex color
    pub fn color(&self, vh: VertexHandle) -> Option<Vec4> {
        self.kernel.vertex_color(vh)
    }

    /// Set vertex color
    pub fn set_color(&mut self, vh: VertexHandle, c: Vec4) {
        self.kernel.set_vertex_color(vh, c);
    }

    /// Request vertex texture coordinates
    pub fn request_vertex_texcoords(&mut self) {
        self.kernel.request_vertex_texcoords();
    }

    /// Check if vertex texcoords are available
    pub fn has_vertex_texcoords(&self) -> bool {
        self.kernel.has_vertex_texcoords()
    }

    /// Get vertex texcoord
    pub fn texcoord(&self, vh: VertexHandle) -> Option<Vec2> {
        self.kernel.vertex_texcoord(vh)
    }

    /// Set vertex texcoord
    pub fn set_texcoord(&mut self, vh: VertexHandle, t: Vec2) {
        self.kernel.set_vertex_texcoord(vh, t);
    }

    // =========================================================================
    // Face attributes
    // =========================================================================

    /// Request face normals
    pub fn request_face_normals(&mut self) {
        self.kernel.request_face_normals();
    }

    /// Check if face normals are available
    pub fn has_face_normals(&self) -> bool {
        self.kernel.has_face_normals()
    }

    /// Get face normal
    pub fn f_normal(&self, fh: FaceHandle) -> Option<Vec3> {
        self.kernel.face_normal(fh)
    }

    /// Set face normal
    pub fn set_f_normal(&mut self, fh: FaceHandle, n: Vec3) {
        self.kernel.set_face_normal(fh, n);
    }

    /// Request face colors
    pub fn request_face_colors(&mut self) {
        self.kernel.request_face_colors();
    }

    /// Check if face colors are available
    pub fn has_face_colors(&self) -> bool {
        self.kernel.has_face_colors()
    }

    /// Get face color
    pub fn f_color(&self, fh: FaceHandle) -> Option<Vec4> {
        self.kernel.face_color(fh)
    }

    /// Set face color
    pub fn set_f_color(&mut self, fh: FaceHandle, c: Vec4) {
        self.kernel.set_face_color(fh, c);
    }

    // =========================================================================
    // Halfedge attributes
    // =========================================================================

    /// Request halfedge normals
    pub fn request_halfedge_normals(&mut self) {
        self.kernel.request_halfedge_normals();
    }

    /// Check if halfedge normals are available
    pub fn has_halfedge_normals(&self) -> bool {
        self.kernel.has_halfedge_normals()
    }

    /// Get halfedge normal
    pub fn h_normal(&self, heh: HalfedgeHandle) -> Option<Vec3> {
        self.kernel.halfedge_normal(heh)
    }

    /// Set halfedge normal
    pub fn set_h_normal(&mut self, heh: HalfedgeHandle, n: Vec3) {
        self.kernel.set_halfedge_normal(heh, n);
    }

    /// Request halfedge colors
    pub fn request_halfedge_colors(&mut self) {
        self.kernel.request_halfedge_colors();
    }

    /// Check if halfedge colors are available
    pub fn has_halfedge_colors(&self) -> bool {
        self.kernel.has_halfedge_colors()
    }

    /// Get halfedge color
    pub fn h_color(&self, heh: HalfedgeHandle) -> Option<Vec4> {
        self.kernel.halfedge_color(heh)
    }

    /// Set halfedge color
    pub fn set_h_color(&mut self, heh: HalfedgeHandle, c: Vec4) {
        self.kernel.set_halfedge_color(heh, c);
    }

    /// Request halfedge texture coordinates
    pub fn request_halfedge_texcoords(&mut self) {
        self.kernel.request_halfedge_texcoords();
    }

    /// Check if halfedge texcoords are available
    pub fn has_halfedge_texcoords(&self) -> bool {
        self.kernel.has_halfedge_texcoords()
    }

    /// Get halfedge texcoord
    pub fn h_texcoord(&self, heh: HalfedgeHandle) -> Option<Vec2> {
        self.kernel.halfedge_texcoord(heh)
    }

    /// Set halfedge texcoord
    pub fn set_h_texcoord(&mut self, heh: HalfedgeHandle, t: Vec2) {
        self.kernel.set_halfedge_texcoord(heh, t);
    }

    // =========================================================================
    // Edge attributes
    // =========================================================================

    /// Request edge colors
    pub fn request_edge_colors(&mut self) {
        self.kernel.request_edge_colors();
    }

    /// Check if edge colors are available
    pub fn has_edge_colors(&self) -> bool {
        self.kernel.has_edge_colors()
    }

    /// Get edge color
    pub fn e_color(&self, eh: EdgeHandle) -> Option<Vec4> {
        self.kernel.edge_color(eh)
    }

    /// Set edge color
    pub fn set_e_color(&mut self, eh: EdgeHandle, c: Vec4) {
        self.kernel.set_edge_color(eh, c);
    }

    // =========================================================================
    // IO helper methods (index-based access for export)
    // =========================================================================

    /// Get vertex position by index (for IO operations)
    pub fn point_by_index(&self, idx: usize) -> Option<Point3> {
        self.kernel.point(idx)
    }

    /// Get vertex normal by index (for IO operations)
    pub fn vertex_normal_by_index(&self, idx: usize) -> Option<Vec3> {
        if idx < self.n_vertices() {
            self.kernel.vertex_normal(VertexHandle::from_usize(idx))
        } else {
            None
        }
    }

    /// Get vertex color by index (for IO operations)
    pub fn vertex_color_by_index(&self, idx: usize) -> Option<Vec4> {
        if idx < self.n_vertices() {
            self.kernel.vertex_color(VertexHandle::from_usize(idx))
        } else {
            None
        }
    }

    /// Get vertex texcoord by index (for IO operations)
    pub fn vertex_texcoord_by_index(&self, idx: usize) -> Option<Vec2> {
        if idx < self.n_vertices() {
            self.kernel.vertex_texcoord(VertexHandle::from_usize(idx))
        } else {
            None
        }
    }

    /// Set vertex normal by index (for IO operations)
    pub fn set_vertex_normal_by_index(&mut self, idx: usize, normal: Vec3) {
        if idx < self.n_vertices() {
            self.kernel
                .set_vertex_normal(VertexHandle::from_usize(idx), normal);
        }
    }

    /// Set vertex color by index (for IO operations)
    pub fn set_vertex_color_by_index(&mut self, idx: usize, color: Vec4) {
        if idx < self.n_vertices() {
            self.kernel
                .set_vertex_color(VertexHandle::from_usize(idx), color);
        }
    }

    /// Set vertex texcoord by index (for IO operations)
    pub fn set_vertex_texcoord_by_index(&mut self, idx: usize, texcoord: Vec2) {
        if idx < self.n_vertices() {
            self.kernel
                .set_vertex_texcoord(VertexHandle::from_usize(idx), texcoord);
        }
    }

    /// Get all vertices of a face (for IO operations - returns Vec)
    pub fn face_vertices_vec(&self, fh: FaceHandle) -> Vec<VertexHandle> {
        let mut vertices = Vec::new();

        // Get the first halfedge of the face
        if let Some(first_heh) = self.kernel.face_halfedge_handle(fh) {
            let mut current_heh = first_heh;
            let mut steps = 0usize;
            let max_steps = self.n_halfedges().max(1);

            loop {
                if steps >= max_steps || !current_heh.is_valid() {
                    break;
                }
                steps += 1;

                // Get the target vertex of this halfedge
                let to_vh = self.kernel.to_vertex_handle(current_heh);
                if !to_vh.is_valid() {
                    break;
                }
                vertices.push(to_vh);

                // Move to next halfedge
                if let Some(next_heh) = self.kernel.next_halfedge_handle(current_heh) {
                    if !next_heh.is_valid() || next_heh == current_heh {
                        break;
                    }
                    current_heh = next_heh;
                    if current_heh == first_heh {
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        vertices
    }

    fn sanitized_face_vertices(&self, fh: FaceHandle) -> Vec<VertexHandle> {
        dedupe_face_vertices(self.face_vertices_vec(fh))
    }

    // =========================================================================
    // Conversion from RustSLAM mesh
    // =========================================================================

    /// Create RustMesh from simple triangle mesh (e.g., from RustSLAM marching cubes)
    ///
    /// Accepts:
    /// - vertices: list of positions
    /// - triangles: list of (v0, v1, v2) index triplets
    /// - normals: optional per-vertex normals
    /// - colors: optional per-vertex colors (RGB as [f32; 3])
    pub fn from_triangle_mesh(
        vertices: &[Point3],
        triangles: &[[usize; 3]],
        normals: Option<&[Vec3]>,
        colors: Option<&[[f32; 3]]>,
    ) -> Self {
        let mut mesh = RustMesh::new();

        // Add vertices
        for pos in vertices {
            mesh.add_vertex(*pos);
        }

        // Add normals if provided
        if let Some(norms) = normals {
            mesh.request_vertex_normals();
            for (i, normal) in norms.iter().enumerate() {
                if i < mesh.n_vertices() {
                    mesh.set_vertex_normal_by_index(i, *normal);
                }
            }
        }

        // Add colors if provided
        if let Some(cols) = colors {
            mesh.request_vertex_colors();
            for (i, color) in cols.iter().enumerate() {
                if i < mesh.n_vertices() {
                    // Convert RGB to RGBA (add alpha = 1.0)
                    let rgba = Vec4::new(color[0], color[1], color[2], 1.0);
                    mesh.set_vertex_color_by_index(i, rgba);
                }
            }
        }

        // Add faces
        for tri in triangles {
            let v0 = VertexHandle::from_usize(tri[0]);
            let v1 = VertexHandle::from_usize(tri[1]);
            let v2 = VertexHandle::from_usize(tri[2]);
            mesh.add_face(&[v0, v1, v2]);
        }

        mesh
    }
}

fn dedupe_face_vertices(vertices: Vec<VertexHandle>) -> Vec<VertexHandle> {
    if vertices.is_empty() {
        return vertices;
    }

    let mut deduped: Vec<VertexHandle> = Vec::with_capacity(vertices.len());
    for vh in vertices {
        if deduped.last().copied() != Some(vh) {
            deduped.push(vh);
        }
    }

    while deduped.len() >= 2 && deduped.first() == deduped.last() {
        deduped.pop();
    }

    let mut unique = Vec::with_capacity(deduped.len());
    for vh in deduped {
        if !unique.contains(&vh) {
            unique.push(vh);
        }
    }
    unique
}

#[allow(dead_code)]
fn canonical_face_key(vertices: &[VertexHandle]) -> Vec<usize> {
    let ids: Vec<usize> = vertices.iter().map(|vh| vh.idx_usize()).collect();
    if ids.is_empty() {
        return ids;
    }

    let mut best: Option<Vec<usize>> = None;
    for base in [ids.clone(), ids.iter().copied().rev().collect()] {
        for shift in 0..base.len() {
            let rotated: Vec<usize> = base
                .iter()
                .cycle()
                .skip(shift)
                .take(base.len())
                .copied()
                .collect();

            if best.as_ref().is_none_or(|current| rotated < *current) {
                best = Some(rotated);
            }
        }
    }

    best.unwrap_or_default()
}

fn try_add_face_with_rotations(mesh: &mut RustMesh, vertices: &[VertexHandle]) -> bool {
    if vertices.len() < 3 {
        return false;
    }

    let len = vertices.len();
    let mut candidate = vertices.to_vec();
    for _ in 0..len {
        if mesh.add_face(&candidate).is_some() {
            return true;
        }
        candidate.rotate_left(1);
    }

    candidate = vertices.iter().copied().rev().collect();
    for _ in 0..len {
        if mesh.add_face(&candidate).is_some() {
            return true;
        }
        candidate.rotate_left(1);
    }

    false
}

fn rebuild_polygon_faces(mesh: &mut RustMesh, faces: &[Vec<VertexHandle>]) {
    for face in faces {
        let _ = try_add_face_with_rotations(mesh, face);
    }
}

fn rebuild_oriented_triangles(mesh: &mut RustMesh, faces: &[Vec<VertexHandle>]) {
    let triangles: Vec<[usize; 3]> = faces
        .iter()
        .map(|face| {
            [
                face[0].idx_usize(),
                face[1].idx_usize(),
                face[2].idx_usize(),
            ]
        })
        .collect();

    let flips = orient_triangle_faces(&triangles);
    for (triangle, flip) in triangles.iter().zip(flips.iter()) {
        let mut face = vec![
            VertexHandle::from_usize(triangle[0]),
            VertexHandle::from_usize(triangle[1]),
            VertexHandle::from_usize(triangle[2]),
        ];
        if *flip {
            face.swap(1, 2);
        }
        let _ = try_add_face_with_rotations(mesh, &face);
    }
}

fn orient_triangle_faces(triangles: &[[usize; 3]]) -> Vec<bool> {
    #[derive(Clone, Copy)]
    struct EdgeUse {
        triangle_idx: usize,
        sign: i8,
    }

    let mut edge_map: HashMap<(usize, usize), Vec<EdgeUse>> = HashMap::new();
    for (triangle_idx, triangle) in triangles.iter().enumerate() {
        for (from, to) in [
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ] {
            let key = if from < to { (from, to) } else { (to, from) };
            let sign = if from < to { 1 } else { -1 };
            edge_map
                .entry(key)
                .or_default()
                .push(EdgeUse { triangle_idx, sign });
        }
    }

    let mut flips: Vec<Option<bool>> = vec![None; triangles.len()];
    for start in 0..triangles.len() {
        if flips[start].is_some() {
            continue;
        }

        flips[start] = Some(false);
        let mut queue = VecDeque::new();
        queue.push_back(start);

        while let Some(current_idx) = queue.pop_front() {
            let current_flip = flips[current_idx].unwrap_or(false);
            let triangle = triangles[current_idx];

            for (from, to) in [
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ] {
                let key = if from < to { (from, to) } else { (to, from) };
                let current_sign = if from < to { 1 } else { -1 };

                let Some(uses) = edge_map.get(&key) else {
                    continue;
                };
                for other in uses {
                    if other.triangle_idx == current_idx {
                        continue;
                    }

                    let should_flip_other = current_flip ^ (current_sign == other.sign);
                    match flips[other.triangle_idx] {
                        Some(existing) if existing != should_flip_other => {}
                        Some(_) => {}
                        None => {
                            flips[other.triangle_idx] = Some(should_flip_other);
                            queue.push_back(other.triangle_idx);
                        }
                    }
                }
            }
        }
    }

    flips
        .into_iter()
        .map(|flip| flip.unwrap_or(false))
        .collect()
}

#[cfg(test)]
mod tests_soa {
    use super::*;

    #[test]
    fn test_soa_mesh() {
        let mut mesh = RustMesh::new();

        // Add vertices
        let v0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(Point3::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(Point3::new(0.0, 1.0, 0.0));

        // Add face
        let face = mesh.add_face(&[v0, v1, v2]);
        assert!(face.is_some());

        // Check counts
        assert_eq!(mesh.n_vertices(), 3);
        assert_eq!(mesh.n_faces(), 1);

        // Check vertex access
        assert_eq!(mesh.point(v0), Some(Point3::new(0.0, 0.0, 0.0)));
        assert_eq!(mesh.point(v1), Some(Point3::new(1.0, 0.0, 0.0)));
        assert_eq!(mesh.point(v2), Some(Point3::new(0.0, 1.0, 0.0)));

        // Check SIMD pointers
        assert!(!mesh.x_ptr().is_null());
        assert!(!mesh.y_ptr().is_null());
        assert!(!mesh.z_ptr().is_null());
    }

    #[test]
    fn test_soa_bounding_box() {
        let mut mesh = RustMesh::new();

        // Add vertices of a cube
        mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        mesh.add_vertex(Point3::new(1.0, 0.0, 0.0));
        mesh.add_vertex(Point3::new(1.0, 1.0, 0.0));
        mesh.add_vertex(Point3::new(0.0, 1.0, 0.0));
        mesh.add_vertex(Point3::new(0.0, 0.0, 1.0));
        mesh.add_vertex(Point3::new(1.0, 0.0, 1.0));
        mesh.add_vertex(Point3::new(1.0, 1.0, 1.0));
        mesh.add_vertex(Point3::new(0.0, 1.0, 1.0));

        let (min_x, max_x, min_y, max_y, min_z, max_z) = mesh.bounding_box();

        assert_eq!(min_x, 0.0);
        assert_eq!(max_x, 1.0);
        assert_eq!(min_y, 0.0);
        assert_eq!(max_y, 1.0);
        assert_eq!(min_z, 0.0);
        assert_eq!(max_z, 1.0);
    }

    #[test]
    fn test_soa_centroid() {
        let mut mesh = RustMesh::new();

        // Add vertices
        mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        mesh.add_vertex(Point3::new(2.0, 0.0, 0.0));
        mesh.add_vertex(Point3::new(0.0, 2.0, 0.0));

        let (cx, cy, cz) = mesh.centroid();

        // Centroid of (0,0,0), (2,0,0), (0,2,0) = (2/3, 2/3, 0)
        assert!((cx - 0.667).abs() < 0.001);
        assert!((cy - 0.667).abs() < 0.001);
        assert_eq!(cz, 0.0);
    }

    #[test]
    fn test_vertex_attributes() {
        let mut mesh = RustMesh::new();

        // Add vertices
        let v0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(Point3::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(Point3::new(0.0, 1.0, 0.0));

        // Request and set vertex normals
        mesh.request_vertex_normals();
        assert!(mesh.has_vertex_normals());

        mesh.set_normal(v0, Vec3::new(0.0, 0.0, 1.0));
        mesh.set_normal(v1, Vec3::new(0.0, 0.0, 1.0));
        mesh.set_normal(v2, Vec3::new(0.0, 0.0, 1.0));

        assert_eq!(mesh.normal(v0), Some(Vec3::new(0.0, 0.0, 1.0)));

        // Request and set vertex colors
        mesh.request_vertex_colors();
        assert!(mesh.has_vertex_colors());

        mesh.set_color(v0, Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(mesh.color(v0), Some(Vec4::new(1.0, 0.0, 0.0, 1.0)));

        // Request and set vertex texcoords
        mesh.request_vertex_texcoords();
        assert!(mesh.has_vertex_texcoords());

        mesh.set_texcoord(v0, Vec2::new(0.0, 0.0));
        mesh.set_texcoord(v1, Vec2::new(1.0, 0.0));
        mesh.set_texcoord(v2, Vec2::new(0.0, 1.0));

        assert_eq!(mesh.texcoord(v0), Some(Vec2::new(0.0, 0.0)));
    }

    #[test]
    fn test_face_attributes() {
        let mut mesh = RustMesh::new();

        // Add vertices and face
        let v0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(Point3::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(Point3::new(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]);

        // Request face normals
        mesh.request_face_normals();
        assert!(mesh.has_face_normals());

        mesh.set_f_normal(FaceHandle::new(0), Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(
            mesh.f_normal(FaceHandle::new(0)),
            Some(Vec3::new(0.0, 0.0, 1.0))
        );

        // Request face colors
        mesh.request_face_colors();
        assert!(mesh.has_face_colors());

        mesh.set_f_color(FaceHandle::new(0), Vec4::new(0.5, 0.5, 0.5, 1.0));
        assert_eq!(
            mesh.f_color(FaceHandle::new(0)),
            Some(Vec4::new(0.5, 0.5, 0.5, 1.0))
        );
    }
}

// RustMesh is now the primary mesh type (previously PolyMeshSoA)
