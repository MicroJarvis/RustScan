//! # Handles
//!
//! Handle types for mesh entities (Vertex, Edge, Halfedge, Face).
//! Handles are lightweight references to mesh elements using integer indices.

use std::fmt;

/// Base handle type for all mesh entities
/// Using u32 for better Rust/usize interoperability (no i32 conversion overhead)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BaseHandle {
    idx: u32,
}

impl BaseHandle {
    /// Create a new handle with the given index (default: u32::MAX = invalid)
    #[inline]
    pub fn new(idx: u32) -> Self {
        Self { idx }
    }

    /// Create from usize
    #[inline]
    pub fn from_usize(idx: usize) -> Self {
        Self { idx: idx as u32 }
    }

    /// Get the underlying index
    #[inline]
    pub fn idx(&self) -> u32 {
        self.idx
    }

    /// Get as usize (for indexing)
    #[inline]
    pub fn idx_usize(&self) -> usize {
        self.idx as usize
    }

    /// Check if the handle is valid (index != MAX)
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.idx != u32::MAX
    }

    /// Invalidate the handle
    #[inline]
    pub fn invalidate(&mut self) {
        self.idx = u32::MAX;
    }

    /// Reset to invalid state
    #[inline]
    pub fn reset(&mut self) {
        self.invalidate();
    }
}

impl Default for BaseHandle {
    #[inline]
    fn default() -> Self {
        Self::new(u32::MAX)
    }
}

impl fmt::Display for BaseHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.idx)
    }
}

/// Handle referencing a vertex entity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VertexHandle(BaseHandle);

impl VertexHandle {
    /// Create a new vertex handle
    #[inline]
    pub fn new(idx: u32) -> Self {
        Self(BaseHandle::new(idx))
    }

    /// Create from usize
    #[inline]
    pub fn from_usize(idx: usize) -> Self {
        Self(BaseHandle::from_usize(idx))
    }

    /// Get an invalid vertex handle
    #[inline]
    pub fn invalid() -> Self {
        Self::new(u32::MAX)
    }

    /// Get the underlying index
    #[inline]
    pub fn idx(&self) -> u32 {
        self.0.idx()
    }

    /// Get as usize (for indexing)
    #[inline]
    pub fn idx_usize(&self) -> usize {
        self.0.idx_usize()
    }

    /// Check if valid
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }

    /// Invalidate the handle
    #[inline]
    pub fn invalidate(&mut self) {
        self.0.invalidate();
    }
}

impl From<u32> for VertexHandle {
    #[inline]
    fn from(idx: u32) -> Self {
        Self::new(idx)
    }
}

impl From<usize> for VertexHandle {
    #[inline]
    fn from(idx: usize) -> Self {
        Self::new(idx as u32)
    }
}

impl Default for VertexHandle {
    #[inline]
    fn default() -> Self {
        Self::invalid()
    }
}

/// Handle referencing a halfedge entity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HalfedgeHandle(BaseHandle);

impl HalfedgeHandle {
    /// Create a new halfedge handle
    #[inline]
    pub fn new(idx: u32) -> Self {
        Self(BaseHandle::new(idx))
    }

    /// Get an invalid halfedge handle
    #[inline]
    pub fn invalid() -> Self {
        Self::new(u32::MAX)
    }

    /// Get the underlying index
    #[inline]
    pub fn idx(&self) -> u32 {
        self.0.idx()
    }

    /// Get as usize (for indexing)
    #[inline]
    pub fn idx_usize(&self) -> usize {
        self.0.idx_usize()
    }

    /// Check if valid
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }

    /// Invalidate the handle
    #[inline]
    pub fn invalidate(&mut self) {
        self.0.invalidate();
    }

    /// Get the opposite halfedge index (xor 1)
    #[inline]
    pub fn opposite(&self) -> Self {
        Self::new(self.idx() ^ 1)
    }
}

impl Default for HalfedgeHandle {
    #[inline]
    fn default() -> Self {
        Self(BaseHandle::default())
    }
}

impl From<u32> for HalfedgeHandle {
    #[inline]
    fn from(idx: u32) -> Self {
        Self::new(idx)
    }
}

impl From<usize> for HalfedgeHandle {
    #[inline]
    fn from(idx: usize) -> Self {
        Self::new(idx as u32)
    }
}

/// Handle referencing an edge entity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EdgeHandle(BaseHandle);

impl EdgeHandle {
    /// Create a new edge handle
    #[inline]
    pub fn new(idx: u32) -> Self {
        Self(BaseHandle::new(idx))
    }

    /// Get an invalid edge handle
    #[inline]
    pub fn invalid() -> Self {
        Self::new(u32::MAX)
    }

    /// Get the underlying index
    #[inline]
    pub fn idx(&self) -> u32 {
        self.0.idx()
    }

    /// Get as usize (for indexing)
    #[inline]
    pub fn idx_usize(&self) -> usize {
        self.0.idx_usize()
    }

    /// Check if valid
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }

    /// Invalidate the handle
    #[inline]
    pub fn invalidate(&mut self) {
        self.0.invalidate();
    }
}

impl Default for EdgeHandle {
    #[inline]
    fn default() -> Self {
        Self(BaseHandle::default())
    }
}

impl From<u32> for EdgeHandle {
    #[inline]
    fn from(idx: u32) -> Self {
        Self::new(idx)
    }
}

impl From<usize> for EdgeHandle {
    #[inline]
    fn from(idx: usize) -> Self {
        Self::new(idx as u32)
    }
}

/// Handle referencing a face entity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FaceHandle(BaseHandle);

impl FaceHandle {
    /// Create a new face handle
    #[inline]
    pub fn new(idx: u32) -> Self {
        Self(BaseHandle::new(idx))
    }

    /// Create from usize
    #[inline]
    pub fn from_usize(idx: usize) -> Self {
        Self(BaseHandle::from_usize(idx))
    }

    /// Get an invalid face handle
    #[inline]
    pub fn invalid() -> Self {
        Self::new(u32::MAX)
    }

    /// Get the underlying index
    #[inline]
    pub fn idx(&self) -> u32 {
        self.0.idx()
    }

    /// Get as usize (for indexing)
    #[inline]
    pub fn idx_usize(&self) -> usize {
        self.0.idx_usize()
    }

    /// Check if valid
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }

    /// Invalidate the handle
    #[inline]
    pub fn invalidate(&mut self) {
        self.0.invalidate();
    }
}

impl Default for FaceHandle {
    #[inline]
    fn default() -> Self {
        Self(BaseHandle::default())
    }
}

impl From<u32> for FaceHandle {
    #[inline]
    fn from(idx: u32) -> Self {
        Self::new(idx)
    }
}

impl From<usize> for FaceHandle {
    #[inline]
    fn from(idx: usize) -> Self {
        Self::new(idx as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_validity() {
        let valid = VertexHandle::new(0);
        let invalid = VertexHandle::default();

        assert!(valid.is_valid());
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_halfedge_opposite() {
        let he = HalfedgeHandle::new(5);
        assert_eq!(he.opposite().idx(), 4);

        let he2 = HalfedgeHandle::new(4);
        assert_eq!(he2.opposite().idx(), 5);
    }
}
