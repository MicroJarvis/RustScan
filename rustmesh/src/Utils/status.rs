//! # Mesh Status Flags
//!
//! Status flags for mesh elements (deleted, selected, locked, etc.)
//! Based on OpenMesh's StatusSet concept.

use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

/// Status flags for mesh elements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StatusFlags(u32);

impl StatusFlags {
    /// Create empty status flags
    pub fn new() -> Self {
        Self(0)
    }

    /// Create with all flags set
    pub fn all() -> Self {
        Self(0xFFFF_FFFF)
    }

    /// Check if deleted
    #[inline]
    pub fn is_deleted(&self) -> bool {
        (self.0 & 1) != 0
    }

    /// Set deleted flag
    #[inline]
    pub fn set_deleted(&mut self, deleted: bool) {
        if deleted {
            self.0 |= 1;
        } else {
            self.0 &= !1;
        }
    }

    /// Check if selected
    #[inline]
    pub fn is_selected(&self) -> bool {
        (self.0 & 2) != 0
    }

    /// Set selected flag
    #[inline]
    pub fn set_selected(&mut self, selected: bool) {
        if selected {
            self.0 |= 2;
        } else {
            self.0 &= !2;
        }
    }

    /// Check if locked
    #[inline]
    pub fn is_locked(&self) -> bool {
        (self.0 & 4) != 0
    }

    /// Set locked flag
    #[inline]
    pub fn set_locked(&mut self, locked: bool) {
        if locked {
            self.0 |= 4;
        } else {
            self.0 &= !4;
        }
    }

    /// Check if hidden
    #[inline]
    pub fn is_hidden(&self) -> bool {
        (self.0 & 8) != 0
    }

    /// Set hidden flag
    #[inline]
    pub fn set_hidden(&mut self, hidden: bool) {
        if hidden {
            self.0 |= 8;
        } else {
            self.0 &= !8;
        }
    }

    /// Check if boundary
    #[inline]
    pub fn is_boundary(&self) -> bool {
        (self.0 & 16) != 0
    }

    /// Set boundary flag
    #[inline]
    pub fn set_boundary(&mut self, boundary: bool) {
        if boundary {
            self.0 |= 16;
        } else {
            self.0 &= !16;
        }
    }

    /// Check if feature
    #[inline]
    pub fn is_feature(&self) -> bool {
        (self.0 & 32) != 0
    }

    /// Set feature flag
    #[inline]
    pub fn set_feature(&mut self, feature: bool) {
        if feature {
            self.0 |= 32;
        } else {
            self.0 &= !32;
        }
    }

    /// Check if tagged
    #[inline]
    pub fn is_tagged(&self) -> bool {
        (self.0 & 64) != 0
    }

    /// Set tagged flag
    #[inline]
    pub fn set_tagged(&mut self, tagged: bool) {
        if tagged {
            self.0 |= 64;
        } else {
            self.0 &= !64;
        }
    }

    /// Check if fixed
    #[inline]
    pub fn is_fixed(&self) -> bool {
        (self.0 & 256) != 0
    }

    /// Set fixed flag
    #[inline]
    pub fn set_fixed(&mut self, fixed: bool) {
        if fixed {
            self.0 |= 256;
        } else {
            self.0 &= !256;
        }
    }

    /// Check if constrained
    #[inline]
    pub fn is_constrained(&self) -> bool {
        (self.0 & 512) != 0
    }

    /// Set constrained flag
    #[inline]
    pub fn set_constrained(&mut self, constrained: bool) {
        if constrained {
            self.0 |= 512;
        } else {
            self.0 &= !512;
        }
    }

    /// Check if visited
    #[inline]
    pub fn is_visited(&self) -> bool {
        (self.0 & 128) != 0
    }

    /// Set visited flag
    #[inline]
    pub fn set_visited(&mut self, visited: bool) {
        if visited {
            self.0 |= 128;
        } else {
            self.0 &= !128;
        }
    }

    /// Get raw bits
    #[inline]
    pub fn bits(&self) -> u32 {
        self.0
    }

    /// Set raw bits
    #[inline]
    pub fn set_bits(&mut self, bits: u32) {
        self.0 = bits;
    }

    /// Check if any flag is set
    #[inline]
    pub fn is_any(&self) -> bool {
        self.0 != 0
    }

    /// Check if all flags are clear
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    /// Reset all flags
    #[inline]
    pub fn clear(&mut self) {
        self.0 = 0;
    }
}

impl BitAnd for StatusFlags {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for StatusFlags {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl BitOr for StatusFlags {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for StatusFlags {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitXor for StatusFlags {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}

impl BitXorAssign for StatusFlags {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl Not for StatusFlags {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}

/// Status set for managing mesh element states
#[derive(Debug, Clone)]
pub struct StatusSet {
    deleted: Vec<bool>,
    selected: Vec<bool>,
    locked: Vec<bool>,
    hidden: Vec<bool>,
    boundary: Vec<bool>,
    feature: Vec<bool>,
    tagged: Vec<bool>,
    visited: Vec<bool>,
    fixed: Vec<bool>,
    constrained: Vec<bool>,
}

impl StatusSet {
    /// Create a new empty status set
    pub fn new() -> Self {
        Self {
            deleted: Vec::new(),
            selected: Vec::new(),
            locked: Vec::new(),
            hidden: Vec::new(),
            boundary: Vec::new(),
            feature: Vec::new(),
            tagged: Vec::new(),
            visited: Vec::new(),
            fixed: Vec::new(),
            constrained: Vec::new(),
        }
    }

    /// Reserve capacity
    pub fn reserve(&mut self, n: usize) {
        self.deleted.reserve(n);
        self.selected.reserve(n);
        self.locked.reserve(n);
        self.hidden.reserve(n);
        self.boundary.reserve(n);
        self.feature.reserve(n);
        self.tagged.reserve(n);
        self.visited.reserve(n);
        self.fixed.reserve(n);
        self.constrained.reserve(n);
    }

    /// Add a new element
    pub fn add(&mut self) {
        self.deleted.push(false);
        self.selected.push(false);
        self.locked.push(false);
        self.hidden.push(false);
        self.boundary.push(false);
        self.feature.push(false);
        self.tagged.push(false);
        self.visited.push(false);
        self.fixed.push(false);
        self.constrained.push(false);
    }

    /// Clear all flags at index
    #[inline]
    pub fn clear_flags(&mut self, idx: usize) {
        self.deleted[idx] = false;
        self.selected[idx] = false;
        self.locked[idx] = false;
        self.hidden[idx] = false;
        self.boundary[idx] = false;
        self.feature[idx] = false;
        self.tagged[idx] = false;
        self.visited[idx] = false;
        self.fixed[idx] = false;
        self.constrained[idx] = false;
    }

    /// Get StatusFlags for element
    #[inline]
    pub fn get(&self, idx: usize) -> StatusFlags {
        let mut flags = StatusFlags::new();
        if self.deleted[idx] {
            flags.set_deleted(true);
        }
        if self.selected[idx] {
            flags.set_selected(true);
        }
        if self.locked[idx] {
            flags.set_locked(true);
        }
        if self.hidden[idx] {
            flags.set_hidden(true);
        }
        if self.boundary[idx] {
            flags.set_boundary(true);
        }
        if self.feature[idx] {
            flags.set_feature(true);
        }
        if self.tagged[idx] {
            flags.set_tagged(true);
        }
        if self.visited[idx] {
            flags.set_visited(true);
        }
        if self.fixed[idx] {
            flags.set_fixed(true);
        }
        if self.constrained[idx] {
            flags.set_constrained(true);
        }
        flags
    }

    /// Set StatusFlags for element
    #[inline]
    pub fn set(&mut self, idx: usize, flags: StatusFlags) {
        self.deleted[idx] = flags.is_deleted();
        self.selected[idx] = flags.is_selected();
        self.locked[idx] = flags.is_locked();
        self.hidden[idx] = flags.is_hidden();
        self.boundary[idx] = flags.is_boundary();
        self.feature[idx] = flags.is_feature();
        self.tagged[idx] = flags.is_tagged();
        self.visited[idx] = flags.is_visited();
        self.fixed[idx] = flags.is_fixed();
        self.constrained[idx] = flags.is_constrained();
    }

    #[inline]
    pub fn set_deleted(&mut self, idx: usize, deleted: bool) {
        self.deleted[idx] = deleted;
    }
    #[inline]
    pub fn is_deleted(&self, idx: usize) -> bool {
        self.deleted[idx]
    }
    #[inline]
    pub fn set_selected(&mut self, idx: usize, selected: bool) {
        self.selected[idx] = selected;
    }
    #[inline]
    pub fn is_selected(&self, idx: usize) -> bool {
        self.selected[idx]
    }
    #[inline]
    pub fn set_locked(&mut self, idx: usize, locked: bool) {
        self.locked[idx] = locked;
    }
    #[inline]
    pub fn is_locked(&self, idx: usize) -> bool {
        self.locked[idx]
    }
    #[inline]
    pub fn set_hidden(&mut self, idx: usize, hidden: bool) {
        self.hidden[idx] = hidden;
    }
    #[inline]
    pub fn is_hidden(&self, idx: usize) -> bool {
        self.hidden[idx]
    }
    #[inline]
    pub fn set_boundary(&mut self, idx: usize, boundary: bool) {
        self.boundary[idx] = boundary;
    }
    #[inline]
    pub fn is_boundary(&self, idx: usize) -> bool {
        self.boundary[idx]
    }
    #[inline]
    pub fn set_feature(&mut self, idx: usize, feature: bool) {
        self.feature[idx] = feature;
    }
    #[inline]
    pub fn is_feature(&self, idx: usize) -> bool {
        self.feature[idx]
    }
    #[inline]
    pub fn set_tagged(&mut self, idx: usize, tagged: bool) {
        self.tagged[idx] = tagged;
    }
    #[inline]
    pub fn is_tagged(&self, idx: usize) -> bool {
        self.tagged[idx]
    }
    #[inline]
    pub fn set_visited(&mut self, idx: usize, visited: bool) {
        self.visited[idx] = visited;
    }
    #[inline]
    pub fn is_visited(&self, idx: usize) -> bool {
        self.visited[idx]
    }
    #[inline]
    pub fn set_fixed(&mut self, idx: usize, fixed: bool) {
        self.fixed[idx] = fixed;
    }
    #[inline]
    pub fn is_fixed(&self, idx: usize) -> bool {
        self.fixed[idx]
    }
    #[inline]
    pub fn set_constrained(&mut self, idx: usize, constrained: bool) {
        self.constrained[idx] = constrained;
    }
    #[inline]
    pub fn is_constrained(&self, idx: usize) -> bool {
        self.constrained[idx]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.deleted.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.deleted.is_empty()
    }
    #[inline]
    pub fn clear(&mut self) {
        self.deleted.clear();
        self.selected.clear();
        self.locked.clear();
        self.hidden.clear();
        self.boundary.clear();
        self.feature.clear();
        self.tagged.clear();
        self.visited.clear();
        self.fixed.clear();
        self.constrained.clear();
    }
}

impl Default for StatusSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_flags() {
        let mut flags = StatusFlags::new();
        assert!(!flags.is_deleted());
        flags.set_deleted(true);
        assert!(flags.is_deleted());
    }

    #[test]
    fn test_status_flags_multiple() {
        let mut flags = StatusFlags::new();
        flags.set_deleted(true);
        flags.set_selected(true);
        flags.set_locked(true);
        assert!(flags.is_deleted());
        assert!(flags.is_selected());
        assert!(flags.is_locked());
    }

    #[test]
    fn test_status_flags_bitwise() {
        let mut flags1 = StatusFlags::new();
        flags1.set_deleted(true);
        flags1.set_selected(true);

        let mut flags2 = StatusFlags::new();
        flags2.set_deleted(true);
        flags2.set_locked(true);

        let combined = flags1 | flags2;
        assert!(combined.is_deleted());
        assert!(combined.is_selected());
        assert!(combined.is_locked());
    }

    #[test]
    fn test_status_set() {
        let mut status = StatusSet::new();
        status.add();
        status.add();
        status.add();

        status.set_selected(0, true);
        status.set_locked(1, true);
        status.set_deleted(2, true);

        assert!(status.is_selected(0));
        assert!(status.is_locked(1));
        assert!(status.is_deleted(2));
    }
}
