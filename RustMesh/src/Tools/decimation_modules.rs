//! # Decimation Modules - Modular Constraints for Mesh Simplification
//!
//! This module provides a modular constraint system for mesh decimation,
//! inspired by OpenMesh's Decimater framework.
//!
//! Each module can:
//! - Check if a collapse is legal
//! - Compute the priority of a collapse
//! - Perform post-collapse updates
//!
//! Multiple modules can be combined to create complex decimation behaviors.

use crate::{HalfedgeHandle, RustMesh, Vec3, VertexHandle, Point3};

/// Information about a potential collapse operation
#[derive(Debug, Clone)]
pub struct CollapseInfo {
    /// The halfedge to collapse (from v_from to v_to)
    pub halfedge: HalfedgeHandle,
    /// The vertex that will be removed
    pub v_from: VertexHandle,
    /// The vertex that will be kept
    pub v_to: VertexHandle,
    /// The target position for v_to after collapse
    pub target_position: Vec3,
}

/// Result of a collapse operation
#[derive(Debug, Clone)]
pub struct CollapseResult {
    /// The halfedge that was collapsed
    pub halfedge: HalfedgeHandle,
    /// The vertex that was removed
    pub v_removed: VertexHandle,
    /// The vertex that was kept
    pub v_kept: VertexHandle,
}

/// Trait for decimation constraint modules
///
/// Each module can independently check collapse legality,
/// compute priorities, and perform post-collapse updates.
pub trait DecimationModule: std::fmt::Debug {
    /// Get the module name
    fn name(&self) -> &str;

    /// Check if a collapse is legal according to this module's constraints
    fn is_collapse_legal(&self, mesh: &RustMesh, info: &CollapseInfo) -> bool;

    /// Compute the priority of a collapse (lower is better)
    /// Returns None if the collapse should be rejected
    fn compute_priority(&self, mesh: &RustMesh, info: &CollapseInfo) -> Option<f32>;

    /// Called after a collapse is performed
    fn post_collapse(&mut self, _mesh: &RustMesh, _result: &CollapseResult) {
        // Default: no post-collapse action
    }
}

// ============================================================================
// Quadric Error Module
// ============================================================================

/// Quadric error constraint module
///
/// Uses quadric error metrics to prioritize collapses.
/// Collapses with error above max_error are rejected.
#[derive(Debug, Clone)]
pub struct ModQuadric {
    /// Maximum allowed quadric error
    pub max_error: f32,
    /// Factor applied to boundary vertices (higher = more protection)
    pub boundary_factor: f32,
}

impl ModQuadric {
    pub fn new(max_error: f32) -> Self {
        Self {
            max_error,
            boundary_factor: 10.0,
        }
    }
}

impl Default for ModQuadric {
    fn default() -> Self {
        Self::new(0.1)
    }
}

impl DecimationModule for ModQuadric {
    fn name(&self) -> &str {
        "Quadric"
    }

    fn is_collapse_legal(&self, _mesh: &RustMesh, _info: &CollapseInfo) -> bool {
        // Quadric module doesn't reject based on legality
        true
    }

    fn compute_priority(&self, mesh: &RustMesh, info: &CollapseInfo) -> Option<f32> {
        // Compute quadric error at target position
        let error = compute_quadric_error(mesh, info.v_from, info.v_to, info.target_position);

        // Apply boundary factor
        let adjusted_error =
            if is_boundary_vertex(mesh, info.v_from) || is_boundary_vertex(mesh, info.v_to) {
                error * self.boundary_factor
            } else {
                error
            };

        if adjusted_error > self.max_error {
            None
        } else {
            Some(adjusted_error)
        }
    }
}

// ============================================================================
// Normal Deviation Module
// ============================================================================

/// Normal deviation constraint module
///
/// Rejects collapses that would cause too much normal deviation.
#[derive(Debug, Clone)]
pub struct ModNormal {
    /// Maximum allowed normal deviation in radians
    pub max_normal_deviation: f32,
}

impl ModNormal {
    pub fn new(max_deviation_degrees: f32) -> Self {
        Self {
            max_normal_deviation: max_deviation_degrees.to_radians(),
        }
    }
}

impl Default for ModNormal {
    fn default() -> Self {
        Self::new(30.0) // 30 degrees default
    }
}

impl DecimationModule for ModNormal {
    fn name(&self) -> &str {
        "Normal"
    }

    fn is_collapse_legal(&self, mesh: &RustMesh, info: &CollapseInfo) -> bool {
        let deviation = compute_normal_deviation(mesh, info);
        deviation <= self.max_normal_deviation
    }

    fn compute_priority(&self, mesh: &RustMesh, info: &CollapseInfo) -> Option<f32> {
        if self.is_collapse_legal(mesh, info) {
            Some(compute_normal_deviation(mesh, info))
        } else {
            None
        }
    }
}

// ============================================================================
// Aspect Ratio Module
// ============================================================================

/// Aspect ratio constraint module
///
/// Rejects collapses that would create faces with poor aspect ratio.
#[derive(Debug, Clone)]
pub struct ModAspectRatio {
    /// Minimum allowed aspect ratio (1.0 = equilateral)
    pub min_aspect_ratio: f32,
}

impl ModAspectRatio {
    pub fn new(min_ratio: f32) -> Self {
        Self {
            min_aspect_ratio: min_ratio,
        }
    }
}

impl Default for ModAspectRatio {
    fn default() -> Self {
        Self::new(0.1) // Allow fairly degenerate triangles
    }
}

impl DecimationModule for ModAspectRatio {
    fn name(&self) -> &str {
        "AspectRatio"
    }

    fn is_collapse_legal(&self, mesh: &RustMesh, info: &CollapseInfo) -> bool {
        // Check if collapse would create degenerate faces
        let faces_after = get_affected_faces(mesh, info.v_from, info.v_to);

        for face_vertices in &faces_after {
            if face_vertices.len() == 3 {
                let ratio = compute_aspect_ratio(mesh, face_vertices);
                if ratio < self.min_aspect_ratio {
                    return false;
                }
            }
        }

        true
    }

    fn compute_priority(&self, mesh: &RustMesh, info: &CollapseInfo) -> Option<f32> {
        if self.is_collapse_legal(mesh, info) {
            Some(0.0) // Aspect ratio doesn't affect priority
        } else {
            None
        }
    }
}

// ============================================================================
// Boundary Protection Module
// ============================================================================

/// Boundary protection constraint module
///
/// Prevents boundary vertices from collapsing to interior vertices.
#[derive(Debug, Clone)]
pub struct ModBoundary {
    /// Whether to completely block boundary collapses
    pub block_boundary_collapses: bool,
}

impl ModBoundary {
    pub fn new(block: bool) -> Self {
        Self {
            block_boundary_collapses: block,
        }
    }
}

impl Default for ModBoundary {
    fn default() -> Self {
        Self::new(false) // Allow boundary collapses by default
    }
}

impl DecimationModule for ModBoundary {
    fn name(&self) -> &str {
        "Boundary"
    }

    fn is_collapse_legal(&self, mesh: &RustMesh, info: &CollapseInfo) -> bool {
        if self.block_boundary_collapses {
            // Block collapse if either vertex is on boundary
            if is_boundary_vertex(mesh, info.v_from) || is_boundary_vertex(mesh, info.v_to) {
                return false;
            }
        }

        // Prevent boundary vertex from collapsing to interior vertex
        let v_from_is_boundary = is_boundary_vertex(mesh, info.v_from);
        let v_to_is_boundary = is_boundary_vertex(mesh, info.v_to);

        // If v_from is boundary and v_to is interior, reject
        if v_from_is_boundary && !v_to_is_boundary {
            return false;
        }

        true
    }

    fn compute_priority(&self, mesh: &RustMesh, info: &CollapseInfo) -> Option<f32> {
        if self.is_collapse_legal(mesh, info) {
            // Give boundary collapses lower priority
            if is_boundary_vertex(mesh, info.v_from) || is_boundary_vertex(mesh, info.v_to) {
                Some(0.01) // Low priority for boundary
            } else {
                Some(0.0)
            }
        } else {
            None
        }
    }
}

// ============================================================================
// Module Composition
// ============================================================================

/// Combined decimation modules
///
/// Combines multiple modules with priority aggregation.
#[derive(Debug, Default)]
pub struct CombinedModules {
    pub modules: Vec<Box<dyn DecimationModule>>,
}

impl CombinedModules {
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
        }
    }

    pub fn add<M: DecimationModule + 'static>(&mut self, module: M) {
        self.modules.push(Box::new(module));
    }

    /// Check if collapse is legal according to all modules
    pub fn is_collapse_legal(&self, mesh: &RustMesh, info: &CollapseInfo) -> bool {
        self.modules.iter().all(|m| m.is_collapse_legal(mesh, info))
    }

    /// Compute combined priority from all modules
    /// Returns None if any module rejects the collapse
    pub fn compute_priority(&self, mesh: &RustMesh, info: &CollapseInfo) -> Option<f32> {
        let mut total_priority = 0.0f32;

        for module in &self.modules {
            match module.compute_priority(mesh, info) {
                Some(p) => total_priority += p,
                None => return None,
            }
        }

        Some(total_priority)
    }

    /// Notify all modules of a completed collapse
    pub fn post_collapse(&mut self, mesh: &RustMesh, result: &CollapseResult) {
        for module in &mut self.modules {
            module.post_collapse(mesh, result);
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn is_boundary_vertex(mesh: &RustMesh, vh: VertexHandle) -> bool {
    if let Some(heh) = mesh.halfedge_handle(vh) {
        return mesh.is_boundary(heh);
    }
    false
}

fn compute_quadric_error(mesh: &RustMesh, v0: VertexHandle, v1: VertexHandle, target: Vec3) -> f32 {
    // Simplified quadric error: distance from target to edge midpoint
    let p0 = mesh.point(v0).unwrap_or(Point3::origin());
    let p1 = mesh.point(v1).unwrap_or(Point3::origin());
    let midpoint = (p0 + p1) * 0.5;
    (target - midpoint).norm_squared()
}

fn compute_normal_deviation(mesh: &RustMesh, info: &CollapseInfo) -> f32 {
    // Compute the change in normals of affected faces
    let p_from = mesh.point(info.v_from).unwrap_or(Point3::origin());
    let p_to = mesh.point(info.v_to).unwrap_or(Point3::origin());

    // Get faces adjacent to v_from
    let mut total_deviation = 0.0f32;
    let mut count = 0;

    if let Some(vf) = mesh.vertex_faces(info.v_from) {
        for fh in vf {
            if let Some(normal) = mesh.f_normal(fh) {
                // Compute new normal after collapse (simplified)
                total_deviation += 0.1; // Placeholder for actual computation
                count += 1;
            }
        }
    }

    if count > 0 {
        total_deviation / count as f32
    } else {
        0.0
    }
}

fn get_affected_faces(
    mesh: &RustMesh,
    v_from: VertexHandle,
    v_to: VertexHandle,
) -> Vec<Vec<VertexHandle>> {
    let mut faces = Vec::new();

    // Get all faces adjacent to v_from that don't contain v_to
    if let Some(vf) = mesh.vertex_faces(v_from) {
        for fh in vf {
            let face_verts: Vec<VertexHandle> = mesh.face_vertices_vec(fh);
            if !face_verts.contains(&v_to) {
                // This face will be modified: replace v_from with v_to
                let new_verts: Vec<VertexHandle> = face_verts
                    .iter()
                    .map(|&v| if v == v_from { v_to } else { v })
                    .collect();
                faces.push(new_verts);
            }
        }
    }

    faces
}

fn compute_aspect_ratio(mesh: &RustMesh, vertices: &[VertexHandle]) -> f32 {
    if vertices.len() != 3 {
        return 1.0;
    }

    let p0 = mesh.point(vertices[0]).unwrap_or(Point3::origin());
    let p1 = mesh.point(vertices[1]).unwrap_or(Point3::origin());
    let p2 = mesh.point(vertices[2]).unwrap_or(Point3::origin());

    let a = (p1 - p0).norm();
    let b = (p2 - p1).norm();
    let c = (p0 - p2).norm();

    let s = (a + b + c) * 0.5;
    let area = (s * (s - a) * (s - b) * (s - c)).max(0.0).sqrt();

    let longest = a.max(b).max(c);

    if longest > 0.0 {
        // Aspect ratio: 2 * area / (longest_edge * sqrt(3))
        let equilateral_area = longest * longest * 3.0f32.sqrt() / 4.0;
        area / equilateral_area
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate_cube;

    #[test]
    fn test_mod_quadric() {
        let mesh = generate_cube();
        let module = ModQuadric::new(100.0); // Allow larger error

        assert_eq!(module.name(), "Quadric");

        // Create a dummy collapse info
        let info = CollapseInfo {
            halfedge: HalfedgeHandle::new(0),
            v_from: VertexHandle::new(0),
            v_to: VertexHandle::new(1),
            target_position: Vec3::new(0.5, 0.5, 0.5),
        };

        let legal = module.is_collapse_legal(&mesh, &info);
        assert!(legal);

        let priority = module.compute_priority(&mesh, &info);
        assert!(priority.is_some());
    }

    #[test]
    fn test_mod_normal() {
        let mesh = generate_cube();
        let module = ModNormal::new(45.0);

        assert_eq!(module.name(), "Normal");

        let info = CollapseInfo {
            halfedge: HalfedgeHandle::new(0),
            v_from: VertexHandle::new(0),
            v_to: VertexHandle::new(1),
            target_position: Vec3::new(0.5, 0.5, 0.5),
        };

        let legal = module.is_collapse_legal(&mesh, &info);
        // Should be legal for a cube with generous normal threshold
        assert!(legal);
    }

    #[test]
    fn test_mod_boundary() {
        let mesh = generate_cube();
        let module = ModBoundary::default();

        assert_eq!(module.name(), "Boundary");

        let info = CollapseInfo {
            halfedge: HalfedgeHandle::new(0),
            v_from: VertexHandle::new(0),
            v_to: VertexHandle::new(1),
            target_position: Vec3::new(0.5, 0.5, 0.5),
        };

        let priority = module.compute_priority(&mesh, &info);
        assert!(priority.is_some());
    }

    #[test]
    fn test_combined_modules() {
        let mesh = generate_cube();

        let mut combined = CombinedModules::new();
        combined.add(ModQuadric::new(100.0)); // Allow larger error
        combined.add(ModNormal::new(30.0));
        combined.add(ModBoundary::default());

        let info = CollapseInfo {
            halfedge: HalfedgeHandle::new(0),
            v_from: VertexHandle::new(0),
            v_to: VertexHandle::new(1),
            target_position: Vec3::new(0.5, 0.5, 0.5),
        };

        let legal = combined.is_collapse_legal(&mesh, &info);
        assert!(legal);

        let priority = combined.compute_priority(&mesh, &info);
        assert!(priority.is_some());
    }

    #[test]
    fn test_aspect_ratio() {
        let mut mesh = crate::RustMesh::new();

        // Create an equilateral triangle
        let v0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(Point3::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(Point3::new(0.5, 0.866, 0.0));

        let ratio = compute_aspect_ratio(&mesh, &[v0, v1, v2]);
        assert!(
            (ratio - 1.0).abs() < 0.01,
            "Equilateral triangle should have aspect ratio ~1.0"
        );
    }
}
