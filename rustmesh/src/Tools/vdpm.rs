// ============================================================================
// VDPM - Vertex-Based Progressive Meshes
// Based on the progressive mesh paper by Hugues Hoppe
// ============================================================================

use crate::{FaceHandle, HalfedgeHandle, Point3, QuadricT, RustMesh, VertexHandle};
use std::collections::VecDeque;

/// Record of a single edge collapse operation
/// This is stored to enable later refinement (vertex split)
#[derive(Debug, Clone)]
pub struct CollapseRecord {
    /// The halfedge that was collapsed (points from v_kept to v_removed)
    pub halfedge: HalfedgeHandle,
    /// The vertex that was removed
    pub v_removed: VertexHandle,
    /// The vertex that was kept
    pub v_kept: VertexHandle,
    /// The original position of v_kept before the collapse
    pub original_v_kept_position: Point3,
    /// The original position of v_removed before the collapse
    pub original_v_removed_position: Point3,
    /// The new position after collapse (where v_kept moved to)
    pub new_position: Point3,
    /// The error/priority of this collapse
    pub error: f32,
    /// Exact mesh state before the collapse, used for deterministic replay.
    pub pre_collapse_mesh: Box<RustMesh>,

    // === Topology information for vertex split ===
    /// The two faces that were removed by the collapse (if any)
    pub removed_faces: [Option<FaceHandle>; 2],
    /// The vertices adjacent to the collapsed edge (fan vertices)
    /// For a typical edge collapse, there are 2-4 fan vertices
    pub fan_vertices: Vec<VertexHandle>,
    /// Whether the collapsed edge was on the boundary
    pub is_boundary: bool,
}

/// Topology information needed for vertex split
#[derive(Debug, Clone)]
pub struct SplitTopologyInfo {
    /// The vertex to split (the merged vertex)
    pub v_split: VertexHandle,
    /// The new vertex to create
    pub v_new: VertexHandle,
    /// Position for the new vertex
    pub new_position: Point3,
    /// Position for the split vertex
    pub split_position: Point3,
    /// The fan vertices that will be separated
    pub left_fan: Vec<VertexHandle>,
    pub right_fan: Vec<VertexHandle>,
}

/// Progressive Mesh representation
/// Stores the original mesh and a sequence of collapse records
/// for forward simplification and reverse refinement
#[derive(Debug)]
pub struct ProgressiveMesh {
    /// The original mesh (cloned at creation)
    pub original: RustMesh,
    /// The current mesh state (starts as clone of original)
    pub current: RustMesh,
    /// Stack of collapse records (for simplification - pop to undo)
    pub collapse_stack: VecDeque<CollapseRecord>,
    /// Quadric-based error metrics for each vertex
    vertex_quadrics: Vec<QuadricT>,
}

impl ProgressiveMesh {
    /// Create a new progressive mesh from an existing mesh
    pub fn new(mesh: &RustMesh) -> Self {
        let original = mesh.clone();
        let current = mesh.clone();

        // Pre-compute vertex quadrics for collapse prioritization
        let vertex_quadrics = Self::compute_vertex_quadrics(&current);

        Self {
            original,
            current,
            collapse_stack: VecDeque::new(),
            vertex_quadrics,
        }
    }

    /// Compute vertex quadrics from face planes
    fn compute_vertex_quadrics(mesh: &RustMesh) -> Vec<QuadricT> {
        let n_vertices = mesh.n_vertices();
        let mut quadrics: Vec<QuadricT> = vec![QuadricT::zero(); n_vertices];

        // Accumulate face quadrics onto vertices
        for fh in mesh.faces() {
            let heh = match mesh.face_halfedge_handle(fh) {
                Some(heh) => heh,
                None => continue,
            };

            // Get face vertices
            let mut vertices = Vec::new();
            let mut current = heh;
            loop {
                let vh = mesh.to_vertex_handle(current);
                vertices.push(vh);
                current = mesh.next_halfedge_handle(current);
                if current == heh || vertices.len() >= 64 {
                    break;
                }
            }

            if vertices.len() < 3 {
                continue;
            }

            // Get vertex positions
            let p0 = match mesh.point(vertices[0]) {
                Some(p) => p,
                None => continue,
            };
            let p1 = match mesh.point(vertices[1]) {
                Some(p) => p,
                None => continue,
            };
            let p2 = match mesh.point(vertices[2]) {
                Some(p) => p,
                None => continue,
            };

            // Compute face normal and center
            let edge1 = p1 - p0;
            let edge2 = p2 - p0;
            let normal = edge1.cross(&edge2);
            let len = normal.norm();
            if len < 1e-10 {
                continue;
            }
            let normal = normal / len;
            let center = Point3::from((p0.coords + p1.coords + p2.coords) / 3.0);

            // Create face quadric
            let face_quadric = QuadricT::from_face(normal, center);

            // Accumulate onto vertices
            for vh in &vertices {
                let idx = vh.idx_usize();
                if idx < n_vertices {
                    quadrics[idx].add_assign_values(face_quadric);
                }
            }
        }

        quadrics
    }

    /// Get the optimal collapse position using quadrics
    fn compute_optimal_position(&self, v0: VertexHandle, v1: VertexHandle) -> Point3 {
        let idx0 = v0.idx_usize();
        let idx1 = v1.idx_usize();

        if idx0 >= self.vertex_quadrics.len() || idx1 >= self.vertex_quadrics.len() {
            return self.current.point(v1).unwrap_or(Point3::origin());
        }

        // Combine quadrics
        let combined = self.vertex_quadrics[idx0].add_values(self.vertex_quadrics[idx1]);
        let (optimal, _error) = combined.optimize();

        optimal
    }

    /// Compute the error of collapsing v0 onto v1
    fn compute_collapse_error(&self, v0: VertexHandle, v1: VertexHandle) -> f32 {
        let idx0 = v0.idx_usize();
        let idx1 = v1.idx_usize();

        if idx0 >= self.vertex_quadrics.len() || idx1 >= self.vertex_quadrics.len() {
            return f32::MAX;
        }

        let combined = self.vertex_quadrics[idx0].add_values(self.vertex_quadrics[idx1]);
        let (_optimal, error) = combined.optimize();

        error
    }

    /// Find the best collapse candidate (halfedge with lowest error)
    fn find_best_collapse(
        &self,
    ) -> Option<(HalfedgeHandle, VertexHandle, VertexHandle, Point3, f32)> {
        let n_halfedges = self.current.n_halfedges();

        let mut best_heh = None;
        let mut best_error = f32::MAX;
        let mut best_v0 = VertexHandle::invalid();
        let mut best_v1 = VertexHandle::invalid();
        let mut best_pos = Point3::origin();

        // Sample halfedges for performance (not all need checking)
        let step = if n_halfedges > 1000 {
            n_halfedges / 500
        } else {
            1
        };

        for heh_idx in (0..n_halfedges).step_by(step) {
            let heh = HalfedgeHandle::new(heh_idx as u32);

            // Check if collapse is legal
            if !self.current.is_collapse_ok(heh) {
                continue;
            }

            let v0 = self.current.to_vertex_handle(heh); // to be removed
            let v1 = self.current.from_vertex_handle(heh); // to be kept

            // Skip boundary edges
            let fh_left = self.current.face_handle(heh);
            let fh_right = self.current.face_handle(heh.opposite());
            if fh_left.is_none() || fh_right.is_none() {
                continue;
            }

            let error = self.compute_collapse_error(v0, v1);

            if error < best_error {
                best_error = error;
                best_heh = Some(heh);
                best_v0 = v0;
                best_v1 = v1;
                best_pos = self.compute_optimal_position(v0, v1);
            }
        }

        if let Some(heh) = best_heh {
            Some((heh, best_v0, best_v1, best_pos, best_error))
        } else {
            None
        }
    }

    /// Get the current number of faces
    pub fn n_faces(&self) -> usize {
        self.current.n_faces()
    }

    /// Get the current number of valid (non-deleted) faces
    pub fn n_valid_faces(&self) -> usize {
        self.current
            .faces()
            .filter(|fh| self.current.face_halfedge_handle(*fh).is_some())
            .count()
    }

    /// Get the original number of faces
    pub fn original_n_faces(&self) -> usize {
        self.original.n_faces()
    }

    /// Get the current number of vertices
    pub fn n_vertices(&self) -> usize {
        self.current.n_vertices()
    }

    /// Get the number of collapse operations performed
    pub fn n_collapses(&self) -> usize {
        self.collapse_stack.len()
    }

    /// Get a reference to the current mesh
    pub fn mesh(&self) -> &RustMesh {
        &self.current
    }

    /// Get a mutable reference to the current mesh
    pub fn mesh_mut(&mut self) -> &mut RustMesh {
        &mut self.current
    }

    fn lod_target_faces(&self, level: f32) -> usize {
        let original_faces = self.original_n_faces();
        if original_faces == 0 {
            return 0;
        }

        let clamped = level.clamp(0.0, 1.0);
        if clamped <= 0.0 {
            return 0;
        }
        if clamped >= 1.0 {
            return original_faces;
        }

        ((original_faces as f32) * clamped).round() as usize
    }

    /// Reposition the progressive mesh to a normalized level-of-detail.
    ///
    /// `level = 0.0` requests the maximally simplified mesh reachable by the
    /// current collapse legality checks, while `level = 1.0` restores the
    /// original mesh. Intermediate values map to a face-budget ratio.
    ///
    /// Exact refine records now exist, but this API still resets and replays
    /// from `original` until incremental current-state navigation is wired up.
    pub fn get_lod(&mut self, level: f32) -> &RustMesh {
        let target_faces = self.lod_target_faces(level);

        reset(self);
        if target_faces < self.original_n_faces() {
            simplify(self, target_faces);
        }

        &self.current
    }
}

/// Create a progressive mesh from a standard mesh
pub fn create_progressive_mesh(mesh: &RustMesh) -> ProgressiveMesh {
    ProgressiveMesh::new(mesh)
}

/// Simplify the progressive mesh to approximately the target number of faces
/// Returns the actual number of collapses performed
pub fn simplify(pm: &mut ProgressiveMesh, target_faces: usize) -> usize {
    let mut collapses = 0;

    while pm.n_valid_faces() > target_faces {
        // Find best collapse
        let collapse = match pm.find_best_collapse() {
            Some(c) => c,
            None => break,
        };

        let (heh, v_removed, v_kept, new_pos, error) = collapse;

        // Get original positions before collapse
        let original_v_kept_pos = pm.current.point(v_kept).unwrap_or(Point3::origin());
        let original_v_removed_pos = pm.current.point(v_removed).unwrap_or(Point3::origin());
        let pre_collapse_mesh = Box::new(pm.current.clone());

        // Collect topology info before collapse
        let (removed_faces, fan_vertices, is_boundary) =
            collect_collapse_topology(&pm.current, heh, v_kept, v_removed);

        // Perform the collapse
        if let Err(e) = pm.current.collapse(heh) {
            eprintln!("Collapse failed: {}", e);
            break;
        }

        // Update v_kept's position to optimal
        pm.current.set_point(v_kept, new_pos);

        // Record the collapse for later refinement
        let record = CollapseRecord {
            halfedge: heh,
            v_removed,
            v_kept,
            original_v_kept_position: original_v_kept_pos,
            original_v_removed_position: original_v_removed_pos,
            new_position: new_pos,
            error,
            pre_collapse_mesh,
            removed_faces,
            fan_vertices,
            is_boundary,
        };

        pm.collapse_stack.push_back(record);

        // Update quadrics: combine v_removed's quadric into v_kept
        let idx_removed = v_removed.idx_usize();
        let idx_kept = v_kept.idx_usize();
        if idx_removed < pm.vertex_quadrics.len() && idx_kept < pm.vertex_quadrics.len() {
            let q_removed = pm.vertex_quadrics[idx_removed];
            pm.vertex_quadrics[idx_kept].add_assign_values(q_removed);
        }

        collapses += 1;
    }

    collapses
}

fn valid_face_count(mesh: &RustMesh) -> usize {
    mesh.faces()
        .filter(|&fh| mesh.face_halfedge_handle(fh).is_some())
        .count()
}

/// Collect topology information before a collapse operation
fn collect_collapse_topology(
    mesh: &RustMesh,
    heh: HalfedgeHandle,
    v_kept: VertexHandle,
    v_removed: VertexHandle,
) -> ([Option<FaceHandle>; 2], Vec<VertexHandle>, bool) {
    let mut removed_faces: [Option<FaceHandle>; 2] = [None, None];
    let mut fan_vertices = Vec::new();
    // Get the opposite halfedge
    let opp = mesh.opposite_halfedge_handle(heh);

    // Check if the edge is on boundary
    let left_face = mesh.face_handle(heh);
    let right_face = mesh.face_handle(opp);
    let is_boundary = left_face.is_none() || right_face.is_none();

    // Store removed faces
    removed_faces[0] = left_face;
    removed_faces[1] = right_face;

    // Collect fan vertices (vertices around the collapsed edge)
    // These are the vertices that will form the new faces after split

    // Get neighbors of v_kept (excluding v_removed)
    if let Some(vv) = mesh.vertex_vertices(v_kept) {
        for v in vv {
            if v != v_removed {
                fan_vertices.push(v);
            }
        }
    }

    // Get neighbors of v_removed (excluding v_kept)
    if let Some(vv) = mesh.vertex_vertices(v_removed) {
        for v in vv {
            if v != v_kept && !fan_vertices.contains(&v) {
                fan_vertices.push(v);
            }
        }
    }

    (removed_faces, fan_vertices, is_boundary)
}

/// Perform a vertex split operation (inverse of edge collapse)
///
/// This restores the exact pre-collapse mesh snapshot recorded for the split.
///
/// Returns the restored vertex handle on success.
pub fn vertex_split(mesh: &mut RustMesh, record: &CollapseRecord) -> Result<VertexHandle, String> {
    *mesh = (*record.pre_collapse_mesh).clone();
    Ok(record.v_removed)
}

/// Refine the progressive mesh to approximately the target number of faces
/// Returns the actual number of refinement operations performed
pub fn refine(pm: &mut ProgressiveMesh, target_faces: usize) -> usize {
    let mut refinements = 0;

    while let Some(next_faces) = pm
        .collapse_stack
        .back()
        .map(|record| valid_face_count(&record.pre_collapse_mesh))
    {
        let current_faces = pm.n_valid_faces();
        let should_refine = current_faces < target_faces
            || (current_faces == target_faces && next_faces == target_faces);
        if !should_refine {
            break;
        }

        if !refine_one(pm) {
            break;
        }

        refinements += 1;
    }

    if refinements > 0 {
        pm.vertex_quadrics = ProgressiveMesh::compute_vertex_quadrics(&pm.current);
    }

    refinements
}

fn refine_one(pm: &mut ProgressiveMesh) -> bool {
    let record = match pm.collapse_stack.pop_back() {
        Some(record) => record,
        None => return false,
    };

    match vertex_split(&mut pm.current, &record) {
        Ok(_) => true,
        Err(error) => {
            eprintln!("Vertex split failed: {}", error);
            pm.collapse_stack.push_back(record);
            false
        }
    }
}

/// Get the current mesh from a progressive mesh
pub fn get_mesh(pm: &ProgressiveMesh) -> &RustMesh {
    &pm.current
}

/// Get the original mesh (unmodified)
pub fn get_original_mesh(pm: &ProgressiveMesh) -> &RustMesh {
    &pm.original
}

/// Reset the progressive mesh to its original state
pub fn reset(pm: &mut ProgressiveMesh) {
    pm.current = pm.original.clone();
    pm.collapse_stack.clear();
    pm.vertex_quadrics = ProgressiveMesh::compute_vertex_quadrics(&pm.current);
}

/// Get simplification progress (0.0 = original, 1.0 = fully simplified)
pub fn simplification_progress(pm: &ProgressiveMesh) -> f32 {
    let original = pm.original_n_faces() as f32;
    let current = pm.n_valid_faces() as f32;

    if original <= 0.0 {
        return 0.0;
    }

    let progress = 1.0 - (current / original);
    progress.max(0.0).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate_cube;
    use crate::generate_sphere;

    fn canonical_face(face: &[VertexHandle]) -> Vec<usize> {
        let mut vertices: Vec<usize> = face.iter().map(|vh| vh.idx_usize()).collect();
        vertices.sort_unstable();
        vertices
    }

    fn mesh_signature(mesh: &RustMesh) -> (Vec<(usize, [f32; 3])>, Vec<Vec<usize>>) {
        let mut vertices = Vec::new();
        for idx in 0..mesh.n_vertices() {
            let vh = VertexHandle::from_usize(idx);
            if mesh.is_vertex_deleted(vh) {
                continue;
            }
            if let Some(point) = mesh.point(vh) {
                vertices.push((idx, point.into()));
            }
        }

        let mut faces: Vec<Vec<usize>> = mesh
            .faces()
            .filter(|&fh| !mesh.is_face_deleted(fh))
            .map(|fh| canonical_face(&mesh.face_vertices_vec(fh)))
            .collect();
        faces.sort_unstable();

        (vertices, faces)
    }

    fn assert_non_decreasing(counts: &[usize], label: &str) {
        for window in counts.windows(2) {
            assert!(
                window[0] <= window[1],
                "{label} should be non-decreasing, got {:?}",
                counts
            );
        }
    }

    fn assert_non_increasing(counts: &[usize], label: &str) {
        for window in counts.windows(2) {
            assert!(
                window[0] >= window[1],
                "{label} should be non-increasing, got {:?}",
                counts
            );
        }
    }

    #[test]
    fn test_create_progressive_mesh() {
        let mesh = generate_cube();
        let pm = create_progressive_mesh(&mesh);

        assert_eq!(pm.n_faces(), pm.original_n_faces());
        assert_eq!(pm.n_collapses(), 0);
    }

    #[test]
    fn test_simplify_cube() {
        let mesh = generate_cube();
        let original_faces = mesh.n_faces();

        let mut pm = create_progressive_mesh(&mesh);

        // Simplify to half the faces
        let target = original_faces / 2;
        let collapses = simplify(&mut pm, target);

        println!("Original faces: {}", original_faces);
        println!("Current faces (valid): {}", pm.n_valid_faces());
        println!("Collapses performed: {}", collapses);

        // With proper link condition, we may not reach the exact target
        // but some simplification should occur
        assert!(pm.n_valid_faces() < original_faces);
        assert!(collapses > 0);
    }

    #[test]
    fn test_simplify_sphere() {
        let mesh = generate_sphere(1.0, 12, 12);
        let original_faces = mesh.n_faces();

        println!("Sphere has {} faces", original_faces);

        let mut pm = create_progressive_mesh(&mesh);

        // Simplify to 25% of original
        let target = original_faces / 4;
        let collapses = simplify(&mut pm, target);

        println!("Simplified to {} valid faces", pm.n_valid_faces());
        println!("Collapses: {}", collapses);

        // With proper link condition, exact target may not be reachable
        // but significant simplification should occur
        assert!(pm.n_valid_faces() < original_faces);
        assert!(collapses > 0);
    }

    #[test]
    fn test_simplify_then_refine() {
        let mesh = generate_cube();
        let expected_signature = mesh_signature(&mesh);
        let original_faces = mesh.n_faces();

        let mut pm = create_progressive_mesh(&mesh);

        // Simplify
        let target = original_faces / 2;
        let _ = simplify(&mut pm, target);

        let simplified_faces = pm.n_valid_faces();
        println!(
            "Simplified from {} to {} faces",
            original_faces, simplified_faces
        );

        // Refine back
        let _ = refine(&mut pm, original_faces);

        println!("Refined to {} faces", pm.n_valid_faces());

        assert!(pm.n_valid_faces() >= simplified_faces);
        assert_eq!(mesh_signature(&pm.current), expected_signature);
    }

    #[test]
    fn test_vertex_split_restores_pre_collapse_snapshot_exactly() {
        let mesh = generate_cube();
        let mut pm = create_progressive_mesh(&mesh);

        let collapses = simplify(&mut pm, mesh.n_faces() / 2);
        assert!(collapses > 0);

        let record = pm.collapse_stack.back().unwrap().clone();
        let expected_signature = mesh_signature(&record.pre_collapse_mesh);

        let restored = vertex_split(&mut pm.current, &record).unwrap();

        assert_eq!(restored, record.v_removed);
        assert_eq!(mesh_signature(&pm.current), expected_signature);
    }

    #[test]
    fn test_refine_restores_last_recorded_snapshot_exactly() {
        let mesh = generate_cube();
        let mut pm = create_progressive_mesh(&mesh);

        let collapses = simplify(&mut pm, mesh.n_faces() / 2);
        assert!(collapses > 0);

        let expected_signature =
            mesh_signature(&pm.collapse_stack.back().unwrap().pre_collapse_mesh);
        let remaining_before = pm.collapse_stack.len();

        assert!(refine_one(&mut pm));

        assert_eq!(pm.collapse_stack.len(), remaining_before - 1);
        assert_eq!(mesh_signature(&pm.current), expected_signature);
    }

    #[test]
    fn test_refine_does_not_depend_on_original_mesh_replay() {
        let mesh = generate_cube();
        let original_signature = mesh_signature(&mesh);
        let original_faces = mesh.n_faces();
        let mut pm = create_progressive_mesh(&mesh);

        let collapses = simplify(&mut pm, original_faces / 2);
        assert!(collapses > 0);

        pm.original = RustMesh::new();
        let refinements = refine(&mut pm, original_faces);

        assert!(refinements > 0);
        assert_eq!(mesh_signature(&pm.current), original_signature);
    }

    #[test]
    fn test_refine_replays_exact_snapshots_in_lifo_order() {
        let mesh = generate_cube();
        let mut pm = create_progressive_mesh(&mesh);

        let collapses = simplify(&mut pm, mesh.n_faces() / 2);
        assert!(collapses >= 2);

        let expected_signatures: Vec<_> = pm
            .collapse_stack
            .iter()
            .rev()
            .map(|record| mesh_signature(&record.pre_collapse_mesh))
            .collect();

        for expected_signature in expected_signatures {
            assert!(refine_one(&mut pm));
            assert_eq!(mesh_signature(&pm.current), expected_signature);
        }

        assert!(pm.collapse_stack.is_empty());
    }

    #[test]
    fn test_reset() {
        let mesh = generate_cube();
        let original_faces = mesh.n_faces();

        let mut pm = create_progressive_mesh(&mesh);

        // Simplify
        let _ = simplify(&mut pm, original_faces / 2);

        // Reset
        reset(&mut pm);

        assert_eq!(pm.n_valid_faces(), original_faces);
        assert_eq!(pm.n_collapses(), 0);
    }

    #[test]
    fn test_progress() {
        let mesh = generate_cube();
        let mut pm = create_progressive_mesh(&mesh);

        let initial_progress = simplification_progress(&pm);
        assert_eq!(initial_progress, 0.0);

        // Simplify
        let _ = simplify(&mut pm, mesh.n_faces() / 2);

        let progress = simplification_progress(&pm);
        println!("Progress after simplification: {:.2}", progress);

        assert!(progress > 0.0);
        assert!(progress < 1.0);
    }

    #[test]
    fn test_get_mesh() {
        let mesh = generate_cube();
        let pm = create_progressive_mesh(&mesh);

        let current = get_mesh(&pm);
        assert_eq!(current.n_faces(), mesh.n_faces());
    }

    #[test]
    fn test_get_lod_clamps_to_extremes() {
        let mesh = generate_sphere(1.0, 12, 12);
        let original_faces = mesh.n_faces();
        let mut pm = create_progressive_mesh(&mesh);

        let _ = pm.get_lod(-0.5);
        let simplified_faces = pm.n_valid_faces();
        assert!(simplified_faces < original_faces);

        let _ = pm.get_lod(1.5);
        assert_eq!(pm.n_valid_faces(), original_faces);
        assert_eq!(pm.n_collapses(), 0);
    }

    #[test]
    fn test_get_lod_midpoint_targets_half_face_budget() {
        let mesh = generate_sphere(1.0, 12, 12);
        let original_faces = mesh.n_faces();
        let target_faces = ((original_faces as f32) * 0.5).round() as usize;
        let tolerance = (original_faces / 8).max(4);

        let mut pm = create_progressive_mesh(&mesh);

        let _ = pm.get_lod(0.0);
        let fully_simplified_faces = pm.n_valid_faces();
        assert!(fully_simplified_faces < original_faces);

        let _ = pm.get_lod(0.5);
        let midpoint_faces = pm.n_valid_faces();

        assert!(midpoint_faces > fully_simplified_faces);
        assert!(midpoint_faces < original_faces);
        assert!(
            midpoint_faces.abs_diff(target_faces) <= tolerance,
            "midpoint faces {} should be within {} of target {}",
            midpoint_faces,
            tolerance,
            target_faces
        );
    }

    #[test]
    fn test_get_lod_face_counts_are_monotonic_for_increasing_levels() {
        let mesh = generate_sphere(1.0, 12, 12);
        let levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let mut pm = create_progressive_mesh(&mesh);

        let counts: Vec<usize> = levels
            .into_iter()
            .map(|level| {
                let _ = pm.get_lod(level);
                pm.n_valid_faces()
            })
            .collect();

        assert_non_decreasing(&counts, "increasing LOD levels");
        assert_eq!(*counts.last().unwrap(), mesh.n_faces());
    }

    #[test]
    fn test_get_lod_bidirectional_scrub_keeps_face_counts_ordered() {
        let mesh = generate_sphere(1.0, 12, 12);
        let levels = [1.0, 0.75, 0.5, 0.25, 0.0, 0.25, 0.5, 0.75, 1.0];
        let mut pm = create_progressive_mesh(&mesh);

        let counts: Vec<usize> = levels
            .into_iter()
            .map(|level| {
                let _ = pm.get_lod(level);
                pm.n_valid_faces()
            })
            .collect();

        assert_non_increasing(&counts[..5], "downward LOD sweep");
        assert_non_decreasing(&counts[4..], "upward LOD sweep");
        assert_eq!(counts[0], mesh.n_faces());
        assert_eq!(*counts.last().unwrap(), mesh.n_faces());
    }

    #[test]
    fn test_get_lod_repeated_level_is_deterministic_after_scrubbing() {
        let mesh = generate_sphere(1.0, 12, 12);
        let mut pm = create_progressive_mesh(&mesh);

        let _ = pm.get_lod(0.5);
        let midpoint_signature = mesh_signature(&pm.current);
        let midpoint_faces = pm.n_valid_faces();

        let _ = pm.get_lod(0.0);
        let _ = pm.get_lod(1.0);
        let _ = pm.get_lod(0.5);

        assert_eq!(pm.n_valid_faces(), midpoint_faces);
        assert_eq!(mesh_signature(&pm.current), midpoint_signature);
    }
}
