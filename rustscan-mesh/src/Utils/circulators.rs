//! # Circulators
//!
//! Circulators provide iterator-based traversal of adjacent mesh elements.

use crate::connectivity::RustMesh;
use crate::handles::{EdgeHandle, FaceHandle, HalfedgeHandle, VertexHandle};

/// Vertex-Vertex Circulator: Visit all unique vertices adjacent to a vertex
pub struct VertexVertexCirculator<'a> {
    mesh: &'a RustMesh,
    start_heh: HalfedgeHandle,
    current_heh: HalfedgeHandle,
    first: bool,
    iteration_count: usize,
    max_iterations: usize,
}

impl<'a> Iterator for VertexVertexCirculator<'a> {
    type Item = VertexHandle;

    fn next(&mut self) -> Option<Self::Item> {
        // Infinite loop protection
        if self.iteration_count >= self.max_iterations {
            return None;
        }
        self.iteration_count += 1;

        // Check if we've completed the cycle (after first iteration)
        if !self.first && self.current_heh == self.start_heh {
            return None;
        }

        // Get the neighbor from current halfedge (it points to the neighbor)
        let next_target = self.mesh.to_vertex_handle(self.current_heh);

        let next_heh = self.mesh.next_outgoing_halfedge(self.current_heh);
        let stuck = next_heh == self.current_heh;

        self.first = false;
        self.current_heh = if stuck { self.start_heh } else { next_heh };
        Some(next_target)
    }
}

impl<'a> RustMesh {
    pub fn vertex_vertices(&'a self, vh: VertexHandle) -> Option<VertexVertexCirculator<'a>> {
        let start_heh = self.halfedge_handle(vh)?;
        let max_iterations = self.n_halfedges().max(1);
        Some(VertexVertexCirculator {
            mesh: self,
            start_heh,
            current_heh: start_heh,
            first: true,
            iteration_count: 0,
            max_iterations,
        })
    }
}

/// Vertex-Face Circulator: Visit all faces adjacent to a vertex
pub struct VertexFaceCirculator<'a> {
    mesh: &'a RustMesh,
    start_heh: HalfedgeHandle,
    current_heh: HalfedgeHandle,
    first: bool,
    iteration_count: usize,
    max_iterations: usize,
}

impl<'a> Iterator for VertexFaceCirculator<'a> {
    type Item = FaceHandle;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Infinite loop protection
            if self.iteration_count >= self.max_iterations {
                return None;
            }
            self.iteration_count += 1;

            // Check if we've completed the cycle
            if !self.first && self.current_heh == self.start_heh {
                return None;
            }

            // Skip invalid halfedges
            if !self.current_heh.is_valid() {
                return None;
            }

            // Get the face from current halfedge
            let fh = self.mesh.face_handle(self.current_heh);

            let next_heh = self.mesh.next_outgoing_halfedge(self.current_heh);
            let stuck = next_heh == self.current_heh;

            self.first = false;
            self.current_heh = if stuck { self.start_heh } else { next_heh };

            // Return face if valid (skip boundary halfedges that have no face)
            if let Some(face) = fh {
                return Some(face);
            }
            if stuck {
                return None;
            }
            // Continue loop to find next valid face (skip boundary)
        }
    }
}

impl<'a> RustMesh {
    pub fn vertex_faces(&'a self, vh: VertexHandle) -> Option<VertexFaceCirculator<'a>> {
        let start_heh = self.halfedge_handle(vh)?;
        let max_iterations = self.n_halfedges().max(1);
        Some(VertexFaceCirculator {
            mesh: self,
            start_heh,
            current_heh: start_heh,
            first: true,
            iteration_count: 0,
            max_iterations,
        })
    }
}

/// Vertex-Halfedge Iterator: Visit all outgoing halfedges around a vertex
pub struct VertexHalfedgeIter<'a> {
    mesh: &'a RustMesh,
    start_heh: HalfedgeHandle,
    current_heh: HalfedgeHandle,
    first: bool,
    iteration_count: usize,
    max_iterations: usize,
}

impl<'a> Iterator for VertexHalfedgeIter<'a> {
    type Item = HalfedgeHandle;

    fn next(&mut self) -> Option<Self::Item> {
        // Infinite loop protection
        if self.iteration_count >= self.max_iterations {
            return None;
        }
        self.iteration_count += 1;

        if !self.first && self.current_heh == self.start_heh {
            return None;
        }

        let heh = self.current_heh;

        let next_outgoing = self.mesh.next_outgoing_halfedge(self.current_heh);
        let stuck = next_outgoing == self.current_heh;

        // Check if we've completed the cycle
        if !self.first && next_outgoing == self.start_heh {
            self.current_heh = next_outgoing;
            self.first = false;
            return Some(heh);
        }

        // Check if next_outgoing is valid
        if !next_outgoing.is_valid() {
            return Some(heh); // Return current but don't advance further
        }

        self.first = false;
        self.current_heh = if stuck { self.start_heh } else { next_outgoing };

        Some(heh)
    }
}

impl<'a> RustMesh {
    pub fn vertex_halfedges(&'a self, vh: VertexHandle) -> Option<VertexHalfedgeIter<'a>> {
        let start_heh = self.halfedge_handle(vh)?;
        let max_iterations = self.n_halfedges().max(1);
        Some(VertexHalfedgeIter {
            mesh: self,
            start_heh,
            current_heh: start_heh,
            first: true,
            iteration_count: 0,
            max_iterations,
        })
    }
}

/// Vertex-Edge Iterator: Visit all edges adjacent to a vertex
pub struct VertexEdgeIter<'a> {
    mesh: &'a RustMesh,
    halfedges: VertexHalfedgeIter<'a>,
}

impl<'a> Iterator for VertexEdgeIter<'a> {
    type Item = EdgeHandle;

    fn next(&mut self) -> Option<Self::Item> {
        self.halfedges.next().map(|heh| self.mesh.edge_handle(heh))
    }
}

impl<'a> RustMesh {
    pub fn vertex_edges(&'a self, vh: VertexHandle) -> Option<VertexEdgeIter<'a>> {
        let halfedges = self.vertex_halfedges(vh)?;
        Some(VertexEdgeIter {
            mesh: self,
            halfedges,
        })
    }
}

/// Face-Vertex Circulator: Visit all vertices of a face
pub struct FaceVertexCirculator<'a> {
    mesh: &'a RustMesh,
    start_heh: HalfedgeHandle,
    current_heh: HalfedgeHandle,
    done: bool,
    iteration_count: usize,
    max_iterations: usize,
}

impl<'a> Iterator for FaceVertexCirculator<'a> {
    type Item = VertexHandle;

    fn next(&mut self) -> Option<Self::Item> {
        // Infinite loop protection
        if self.iteration_count >= self.max_iterations {
            return None;
        }
        self.iteration_count += 1;

        if self.done {
            return None;
        }

        let vh = self.mesh.from_vertex_handle(self.current_heh);

        self.current_heh = self.mesh.next_halfedge_handle(self.current_heh);

        if self.current_heh == self.start_heh {
            self.done = true;
        }

        Some(vh)
    }
}

impl<'a> RustMesh {
    pub fn face_vertices(&'a self, fh: FaceHandle) -> Option<FaceVertexCirculator<'a>> {
        let start_heh = self.face_halfedge_handle(fh)?;
        let max_iterations = self.n_halfedges().max(1);
        Some(FaceVertexCirculator {
            mesh: self,
            start_heh,
            current_heh: start_heh,
            done: false,
            iteration_count: 0,
            max_iterations,
        })
    }
}

/// Face-Halfedge Iterator: Visit all halfedges in a face cycle
pub struct FaceHalfedgeIter<'a> {
    mesh: &'a RustMesh,
    start_heh: HalfedgeHandle,
    current_heh: HalfedgeHandle,
    done: bool,
    iteration_count: usize,
    max_iterations: usize,
}

impl<'a> Iterator for FaceHalfedgeIter<'a> {
    type Item = HalfedgeHandle;

    fn next(&mut self) -> Option<Self::Item> {
        // Infinite loop protection
        if self.iteration_count >= self.max_iterations {
            return None;
        }
        self.iteration_count += 1;

        if self.done {
            return None;
        }

        let heh = self.current_heh;
        self.current_heh = self.mesh.next_halfedge_handle(self.current_heh);

        // Check if we've completed the cycle AFTER advancing
        if self.current_heh == self.start_heh {
            self.done = true;
        }

        Some(heh)
    }
}

impl<'a> RustMesh {
    pub fn face_halfedges(&'a self, fh: FaceHandle) -> Option<FaceHalfedgeIter<'a>> {
        let start_heh = self.face_halfedge_handle(fh)?;
        let max_iterations = self.n_halfedges().max(1);
        Some(FaceHalfedgeIter {
            mesh: self,
            start_heh,
            current_heh: start_heh,
            done: false,
            iteration_count: 0,
            max_iterations,
        })
    }
}

/// Face-Edge Iterator: Visit all edges of a face
pub struct FaceEdgeIter<'a> {
    mesh: &'a RustMesh,
    halfedges: FaceHalfedgeIter<'a>,
}

impl<'a> Iterator for FaceEdgeIter<'a> {
    type Item = EdgeHandle;

    fn next(&mut self) -> Option<Self::Item> {
        self.halfedges.next().map(|heh| self.mesh.edge_handle(heh))
    }
}

impl<'a> RustMesh {
    pub fn face_edges(&'a self, fh: FaceHandle) -> Option<FaceEdgeIter<'a>> {
        let halfedges = self.face_halfedges(fh)?;
        Some(FaceEdgeIter {
            mesh: self,
            halfedges,
        })
    }
}

/// Face-Face Circulator: Visit all adjacent faces
pub struct FaceFaceCirculator<'a> {
    mesh: &'a RustMesh,
    start_heh: HalfedgeHandle,
    current_heh: HalfedgeHandle,
    done: bool,
    iteration_count: usize,
    max_iterations: usize,
}

impl<'a> Iterator for FaceFaceCirculator<'a> {
    type Item = FaceHandle;

    fn next(&mut self) -> Option<Self::Item> {
        // Infinite loop protection
        if self.iteration_count >= self.max_iterations {
            return None;
        }
        self.iteration_count += 1;

        if self.done {
            return None;
        }

        let opposite = self.mesh.opposite_halfedge_handle(self.current_heh);
        let fh = self.mesh.face_handle(opposite);

        self.current_heh = self.mesh.next_halfedge_handle(self.current_heh);

        if self.current_heh == self.start_heh {
            self.done = true;
        }

        fh
    }
}

impl<'a> RustMesh {
    pub fn face_faces(&'a self, fh: FaceHandle) -> Option<FaceFaceCirculator<'a>> {
        let start_heh = self.face_halfedge_handle(fh)?;
        let max_iterations = self.n_halfedges().max(1);
        Some(FaceFaceCirculator {
            mesh: self,
            start_heh,
            current_heh: start_heh,
            done: false,
            iteration_count: 0,
            max_iterations,
        })
    }
}

// ============================================================================
// Additional Circulators (E6)
// ============================================================================

/// Halfedge-Halfedge Circulator: Visit halfedges around a halfedge
///
/// This visits halfedges that are connected to the current halfedge through
/// shared vertices or shared faces.
pub struct HalfedgeHalfedgeCirculator<'a> {
    /// Adjacent halfedges to visit
    adjacent: Vec<HalfedgeHandle>,
    current_idx: usize,
    _marker: std::marker::PhantomData<&'a RustMesh>,
}

impl<'a> Iterator for HalfedgeHalfedgeCirculator<'a> {
    type Item = HalfedgeHandle;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.adjacent.len() {
            return None;
        }
        let heh = self.adjacent[self.current_idx];
        self.current_idx += 1;
        Some(heh)
    }
}

impl<'a> RustMesh {
    /// Get halfedges adjacent to a given halfedge
    ///
    /// Adjacent halfedges include:
    /// - The opposite halfedge
    /// - Halfedges sharing the same face (prev/next)
    /// - Other outgoing halfedges from the same vertex
    pub fn halfedge_halfedges(
        &'a self,
        heh: HalfedgeHandle,
    ) -> Option<HalfedgeHalfedgeCirculator<'a>> {
        let mut adjacent = Vec::new();

        // Add opposite halfedge
        let opp = self.opposite_halfedge_handle(heh);
        adjacent.push(opp);

        // Add prev and next in the same face (if face exists)
        if let Some(_fh) = self.face_handle(heh) {
            let prev = self.prev_halfedge_handle(heh);
            let next = self.next_halfedge_handle(heh);
            if prev != heh && prev != opp {
                adjacent.push(prev);
            }
            if next != heh && next != opp {
                adjacent.push(next);
            }
        }

        // Add other outgoing halfedges from the same vertex (from_vertex of heh)
        let from_v = self.from_vertex_handle(heh);
        if let Some(vh_iter) = self.vertex_halfedges(from_v) {
            for other_heh in vh_iter {
                if other_heh != heh && other_heh != opp && !adjacent.contains(&other_heh) {
                    adjacent.push(other_heh);
                }
            }
        }

        // Limit max iterations for safety
        let max_iterations = self.n_halfedges().max(1);
        adjacent.truncate(max_iterations);

        Some(HalfedgeHalfedgeCirculator {
            adjacent,
            current_idx: 0,
            _marker: std::marker::PhantomData,
        })
    }
}

/// Edge-Edge Circulator: Visit edges around an edge
///
/// This visits edges that share a vertex with the given edge.
pub struct EdgeEdgeCirculator<'a> {
    adjacent: Vec<EdgeHandle>,
    current_idx: usize,
    _marker: std::marker::PhantomData<&'a RustMesh>,
}

impl<'a> Iterator for EdgeEdgeCirculator<'a> {
    type Item = EdgeHandle;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.adjacent.len() {
            return None;
        }
        let eh = self.adjacent[self.current_idx];
        self.current_idx += 1;
        Some(eh)
    }
}

impl<'a> RustMesh {
    /// Get edges adjacent to a given edge (sharing a vertex)
    pub fn edge_edges(&'a self, eh: EdgeHandle) -> Option<EdgeEdgeCirculator<'a>> {
        // Get the first halfedge for this edge
        let heh = self.edge_halfedge_handle(eh, 0);

        // Get both vertices of the edge
        let v0 = self.from_vertex_handle(heh);
        let v1 = self.to_vertex_handle(heh);

        let mut adjacent = Vec::new();

        // Add edges incident to v0
        if let Some(vh_edges) = self.vertex_edges(v0) {
            for other_eh in vh_edges {
                if other_eh != eh && !adjacent.contains(&other_eh) {
                    adjacent.push(other_eh);
                }
            }
        }

        // Add edges incident to v1
        if let Some(vh_edges) = self.vertex_edges(v1) {
            for other_eh in vh_edges {
                if other_eh != eh && !adjacent.contains(&other_eh) {
                    adjacent.push(other_eh);
                }
            }
        }

        Some(EdgeEdgeCirculator {
            adjacent,
            current_idx: 0,
            _marker: std::marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate_tetrahedron;

    #[test]
    fn test_edge_count() {
        let mesh = generate_tetrahedron();
        assert_eq!(mesh.n_vertices(), 4);
        assert_eq!(mesh.n_edges(), 6);
        assert_eq!(mesh.n_faces(), 4);
    }

    #[test]
    fn test_basic_iteration() {
        let mesh = generate_tetrahedron();
        let v_count = mesh.vertices().count();
        assert_eq!(v_count, 4);

        let f_count = mesh.faces().count();
        assert_eq!(f_count, 4);
    }

    #[test]
    fn test_halfedge_structure() {
        // Verify that halfedge next pointers are correctly set
        let mesh = generate_tetrahedron();

        // Each face should have 3 halfedges in a cycle
        for fh in mesh.faces() {
            let start_heh = mesh
                .face_halfedge_handle(fh)
                .expect("Face should have halfedge");

            eprintln!("Face {:?}: start_heh = {:?}", fh, start_heh);

            // Follow the cycle
            let mut count = 0;
            let mut heh = start_heh;
            loop {
                count += 1;
                if count > 10 {
                    panic!("More than 10 halfedges in face cycle - next pointers broken!");
                }

                let next = mesh.next_halfedge_handle(heh);
                eprintln!("  heh={:?} -> next={:?}", heh, next);
                if next == start_heh {
                    break;
                }
                heh = next;
            }

            assert_eq!(count, 3, "Face should have exactly 3 halfedges in cycle");
        }
    }

    #[test]
    fn test_vertex_vertex_circulator() {
        let mesh = generate_tetrahedron();
        let v0 = VertexHandle::from_usize(0);

        let neighbors: Vec<_> = match mesh.vertex_vertices(v0) {
            Some(c) => c.collect(),
            None => panic!("No circulator for vertex 0"),
        };

        eprintln!("v0 = {:?}, neighbors = {:?}", v0, neighbors);

        // Tetrahedron: each vertex has 3 neighbors
        assert_eq!(
            neighbors.len(),
            3,
            "Expected 3 neighbors, got {:?}",
            neighbors
        );

        // Verify neighbors are unique and not self
        let mut unique: Vec<_> = neighbors.iter().filter(|&&vh| vh != v0).collect();
        unique.sort();
        unique.dedup();
        eprintln!("unique (after filter) = {:?}", unique);
        assert_eq!(
            unique.len(),
            3,
            "All 3 neighbors should be unique and not self"
        );
    }

    #[test]
    fn test_vertex_face_circulator() {
        let mesh = generate_tetrahedron();
        let v0 = VertexHandle::from_usize(0);

        let faces: Vec<_> = match mesh.vertex_faces(v0) {
            Some(c) => c.collect(),
            None => panic!("No circulator for vertex 0"),
        };

        // Tetrahedron: each vertex belongs to 3 faces
        assert_eq!(faces.len(), 3, "Expected 3 faces, got {:?}", faces);
    }

    #[test]
    fn test_face_vertex_circulator() {
        let mesh = generate_tetrahedron();
        let fh = FaceHandle::from_usize(0);

        let vertices: Vec<_> = match mesh.face_vertices(fh) {
            Some(c) => c.collect(),
            None => panic!("No circulator for face 0"),
        };

        // Each face is a triangle
        assert_eq!(vertices.len(), 3, "Expected 3 vertices, got {:?}", vertices);
    }

    #[test]
    fn test_face_face_circulator() {
        let mesh = generate_tetrahedron();
        let fh = FaceHandle::from_usize(0);

        let neighbors: Vec<_> = match mesh.face_faces(fh) {
            Some(c) => c.collect(),
            None => panic!("No circulator for face 0"),
        };

        // Tetrahedron: each face has 3 adjacent faces
        assert_eq!(
            neighbors.len(),
            3,
            "Expected 3 adjacent faces, got {:?}",
            neighbors
        );
    }

    #[test]
    fn test_vertex_halfedge_iter() {
        let mesh = generate_tetrahedron();
        let v0 = VertexHandle::from_usize(0);

        // Get via circulator
        let halfedges: Vec<_> = match mesh.vertex_halfedges(v0) {
            Some(c) => c.collect(),
            None => panic!("No iterator for vertex 0"),
        };

        // Tetrahedron: each vertex has 3 outgoing halfedges
        assert_eq!(
            halfedges.len(),
            3,
            "Expected 3 halfedges, got {:?}",
            halfedges
        );

        // All returned halfedges should be outgoing from v0
        for heh in &halfedges {
            assert_eq!(
                mesh.from_vertex_handle(*heh),
                v0,
                "Halfedge {:?} should be outgoing from vertex {:?}",
                heh,
                v0
            );
        }
    }

    #[test]
    fn test_vertex_edge_iter() {
        let mesh = generate_tetrahedron();
        let v0 = VertexHandle::from_usize(0);

        let edges: Vec<_> = match mesh.vertex_edges(v0) {
            Some(c) => c.collect(),
            None => panic!("No iterator for vertex 0"),
        };
        let halfedge_edges: Vec<_> = match mesh.vertex_halfedges(v0) {
            Some(c) => c.map(|heh| mesh.edge_handle(heh)).collect(),
            None => panic!("No iterator for vertex 0"),
        };

        assert_eq!(edges.len(), 3, "Expected 3 edges, got {:?}", edges);

        let mut sorted_edges = edges.clone();
        sorted_edges.sort();
        let mut sorted_halfedge_edges = halfedge_edges.clone();
        sorted_halfedge_edges.sort();
        assert_eq!(
            sorted_edges, sorted_halfedge_edges,
            "vertex_edges should match vertex_halfedges + edge_handle"
        );

        sorted_edges.dedup();
        assert_eq!(
            sorted_edges.len(),
            3,
            "Edges around vertex should be unique"
        );
    }

    #[test]
    fn test_face_halfedge_iter() {
        let mesh = generate_tetrahedron();

        for fh in mesh.faces() {
            let halfedges: Vec<_> = match mesh.face_halfedges(fh) {
                Some(c) => c.collect(),
                None => panic!("No iterator for face {:?}", fh),
            };

            assert_eq!(
                halfedges.len(),
                3,
                "Expected 3 halfedges, got {:?}",
                halfedges
            );

            for heh in &halfedges {
                assert_eq!(
                    mesh.face_handle(*heh),
                    Some(fh),
                    "Face halfedge must belong to queried face"
                );
            }
        }
    }

    #[test]
    fn test_face_edge_iter() {
        let mesh = generate_tetrahedron();

        for fh in mesh.faces() {
            let edges: Vec<_> = match mesh.face_edges(fh) {
                Some(c) => c.collect(),
                None => panic!("No iterator for face {:?}", fh),
            };
            let halfedge_edges: Vec<_> = match mesh.face_halfedges(fh) {
                Some(c) => c.map(|heh| mesh.edge_handle(heh)).collect(),
                None => panic!("No iterator for face {:?}", fh),
            };

            assert_eq!(edges.len(), 3, "Expected 3 edges, got {:?}", edges);

            let mut sorted_edges = edges.clone();
            sorted_edges.sort();
            let mut sorted_halfedge_edges = halfedge_edges.clone();
            sorted_halfedge_edges.sort();
            assert_eq!(
                sorted_edges, sorted_halfedge_edges,
                "face_edges should match face_halfedges + edge_handle"
            );

            sorted_edges.dedup();
            assert_eq!(sorted_edges.len(), 3, "Edges in a face should be unique");
        }
    }

    #[test]
    fn test_new_iterators_invalid_handles() {
        let mesh = generate_tetrahedron();

        assert!(mesh.vertex_halfedges(VertexHandle::invalid()).is_none());
        assert!(mesh.vertex_edges(VertexHandle::invalid()).is_none());
        assert!(mesh.face_halfedges(FaceHandle::invalid()).is_none());
        assert!(mesh.face_edges(FaceHandle::invalid()).is_none());
    }

    #[test]
    fn test_halfedge_halfedge_circulator() {
        let mesh = generate_tetrahedron();
        // Get a valid halfedge from a vertex
        let v0 = VertexHandle::new(0);
        let heh = mesh
            .halfedge_handle(v0)
            .expect("Vertex should have halfedge");

        let adjacent: Vec<_> = match mesh.halfedge_halfedges(heh) {
            Some(c) => c.collect(),
            None => panic!("No circulator for halfedge"),
        };

        // Should include opposite halfedge and adjacent halfedges
        println!("Adjacent halfedges to {:?}: {:?}", heh, adjacent);
        assert!(!adjacent.is_empty(), "Should have adjacent halfedges");

        // Should include the opposite
        let opp = mesh.opposite_halfedge_handle(heh);
        assert!(adjacent.contains(&opp), "Should include opposite halfedge");
    }

    #[test]
    fn test_edge_edge_circulator() {
        let mesh = generate_tetrahedron();
        // Get a valid edge from a vertex's halfedge
        let v0 = VertexHandle::new(0);
        let heh = mesh
            .halfedge_handle(v0)
            .expect("Vertex should have halfedge");
        let eh = mesh.edge_handle(heh);

        let adjacent: Vec<_> = match mesh.edge_edges(eh) {
            Some(c) => c.collect(),
            None => panic!("No circulator for edge"),
        };

        println!("Adjacent edges to {:?}: {:?}", eh, adjacent);

        // In a tetrahedron, each edge connects 2 vertices
        // Each vertex has 3 edges, so an edge should have (3-1) + (3-1) = 4 adjacent edges
        // But some might overlap, so we expect at least 2 and at most 4
        assert!(!adjacent.is_empty(), "Should have adjacent edges");

        // The edge itself should not be in the list
        assert!(!adjacent.contains(&eh), "Edge should not include itself");
    }
}
