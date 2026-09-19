//! # Mesh Items
//!
//! Core mesh data structures: Vertex, Halfedge, Edge, Face.
//! These represent the fundamental elements stored in the mesh.

use crate::handles::{EdgeHandle, FaceHandle, HalfedgeHandle, VertexHandle};
use crate::Point3;

/// A vertex in the mesh
#[derive(Debug, Clone, PartialEq)]
pub struct Vertex {
    /// Position of the vertex in 3D space
    pub point: Point3,
    /// Handle to one of the outgoing halfedges
    pub halfedge_handle: Option<HalfedgeHandle>,
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            point: Point3::origin(),
            halfedge_handle: None,
        }
    }
}

impl Vertex {
    /// Create a new vertex at the given position
    pub fn new(point: Point3) -> Self {
        Self {
            point,
            halfedge_handle: None,
        }
    }

    /// Check if the vertex is isolated (no connected edges)
    pub fn is_isolated(&self) -> bool {
        self.halfedge_handle.is_none()
    }
}

/// A halfedge in the mesh (directional edge)
/// Each edge consists of two halfedges in opposite directions
#[derive(Debug, Clone, PartialEq)]
pub struct Halfedge {
    /// The vertex this halfedge points to
    pub vertex_handle: VertexHandle,
    /// The face this halfedge borders (None if boundary)
    pub face_handle: Option<FaceHandle>,
    /// The next halfedge in the face boundary
    pub next_halfedge_handle: Option<HalfedgeHandle>,
    /// The previous halfedge in the face boundary
    pub prev_halfedge_handle: Option<HalfedgeHandle>,
    /// The opposite halfedge (same edge, opposite direction)
    pub opposite_halfedge_handle: Option<HalfedgeHandle>,
    /// The edge index this halfedge belongs to
    pub edge_idx: u32,
}

impl Default for Halfedge {
    #[inline]
    fn default() -> Self {
        Self {
            vertex_handle: VertexHandle::default(),
            face_handle: None,
            next_halfedge_handle: None,
            prev_halfedge_handle: None,
            opposite_halfedge_handle: None,
            edge_idx: u32::MAX,
        }
    }
}

impl Halfedge {
    /// Create a new halfedge pointing to the given vertex
    pub fn new(heh: HalfedgeHandle, to_vertex: VertexHandle) -> Self {
        Self {
            vertex_handle: to_vertex,
            face_handle: None,
            next_halfedge_handle: None,
            prev_halfedge_handle: None,
            opposite_halfedge_handle: None,
            edge_idx: heh.idx() / 2, // Each edge has 2 halfedges: (2n, 2n+1)
        }
    }

    /// Set the opposite halfedge handle
    pub fn set_opposite_halfedge(&mut self, heh: HalfedgeHandle) {
        self.opposite_halfedge_handle = Some(heh);
    }

    /// Set the next halfedge handle
    pub fn set_next_halfedge_handle(&mut self, heh: HalfedgeHandle) {
        self.next_halfedge_handle = Some(heh);
    }

    /// Set the face handle
    pub fn set_face_handle(&mut self, fh: FaceHandle) {
        self.face_handle = Some(fh);
    }

    /// Get the vertex handle this halfedge points to
    pub fn to_vertex_handle(&self) -> VertexHandle {
        self.vertex_handle
    }

    /// Set the to vertex handle
    pub fn set_to_vertex_handle(&mut self, vh: VertexHandle) {
        self.vertex_handle = vh;
    }

    /// Get the from vertex handle (opposite halfedge's to_vertex)
    /// Note: This requires mesh context to access the opposite halfedge's data.
    /// Use Connectivity::from_vertex_handle() instead for proper implementation.
    pub fn from_vertex_handle(&self) -> VertexHandle {
        // This is a placeholder - actual implementation requires mesh context
        // Use kernel/connectivity to find the opposite halfedge and get its to_vertex
        // For now, this returns invalid handle and callers should use
        // mesh.from_vertex_handle(halfedge) instead
        VertexHandle::invalid()
    }

    /// Get the opposite halfedge handle
    pub fn opposite_halfedge_handle(&self) -> Option<HalfedgeHandle> {
        self.opposite_halfedge_handle
    }

    /// Get the face handle
    pub fn face_handle(&self) -> Option<FaceHandle> {
        self.face_handle
    }

    /// Get the next halfedge handle
    pub fn next_halfedge_handle(&self) -> Option<HalfedgeHandle> {
        self.next_halfedge_handle
    }

    /// Get the prev halfedge handle
    pub fn prev_halfedge_handle(&self) -> Option<HalfedgeHandle> {
        self.prev_halfedge_handle
    }

    /// Get the edge handle
    pub fn edge_handle(&self) -> Option<EdgeHandle> {
        if self.edge_idx != u32::MAX {
            Some(EdgeHandle::new(self.edge_idx))
        } else {
            None
        }
    }

    /// Check if this is a boundary halfedge
    pub fn is_boundary(&self) -> bool {
        self.face_handle.is_none()
    }
}

/// An edge in the mesh (undirected connection between two vertices)
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    /// The two halfedges belonging to this edge
    pub halfedges: [HalfedgeHandle; 2],
}

impl Default for Edge {
    fn default() -> Self {
        Self {
            halfedges: [HalfedgeHandle::default(), HalfedgeHandle::default()],
        }
    }
}

impl Edge {
    /// Create a new edge with the given halfedge handles
    pub fn new(halfedge0: HalfedgeHandle, halfedge1: HalfedgeHandle) -> Self {
        Self {
            halfedges: [halfedge0, halfedge1],
        }
    }

    /// Get the halfedge at the given index (0 or 1)
    pub fn halfedge(&self, idx: usize) -> HalfedgeHandle {
        assert!(idx < 2, "Halfedge index must be 0 or 1");
        self.halfedges[idx]
    }

    /// Get the halfedge handle (alias for halfedge(0))
    pub fn halfedge_handle(&self, idx: usize) -> HalfedgeHandle {
        self.halfedge(idx)
    }
}

/// A face in the mesh (polygon)
#[derive(Debug, Clone, PartialEq)]
pub struct Face {
    /// Handle to one of the halfedges bordering this face
    pub halfedge_handle: Option<HalfedgeHandle>,
}

impl Default for Face {
    fn default() -> Self {
        Self {
            halfedge_handle: None,
        }
    }
}

impl Face {
    /// Create a new face with the given halfedge handle
    pub fn new(halfedge_handle: Option<HalfedgeHandle>) -> Self {
        Self { halfedge_handle }
    }

    /// Get the halfedge handle
    pub fn halfedge_handle(&self) -> Option<HalfedgeHandle> {
        self.halfedge_handle
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handles::HalfedgeHandle;

    #[test]
    fn test_vertex_creation() {
        let v = Vertex::new(Point3::new(1.0, 2.0, 3.0));
        assert_eq!(v.point, Point3::new(1.0, 2.0, 3.0));
        assert!(v.is_isolated());
    }

    #[test]
    fn test_edge_creation() {
        let he0 = HalfedgeHandle::new(0);
        let he1 = HalfedgeHandle::new(1);
        let e = Edge::new(he0, he1);
        assert_eq!(e.halfedge(0), he0);
        assert_eq!(e.halfedge(1), he1);
    }

    #[test]
    fn test_face_creation() {
        let he = HalfedgeHandle::new(5);
        let f = Face::new(Some(he));
        assert_eq!(f.halfedge_handle, Some(he));
    }
}
