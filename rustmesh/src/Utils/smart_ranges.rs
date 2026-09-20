//! # SmartRanges - Chainable iteration with aggregation
//!
//! Provides convenient methods for mesh element iteration and aggregation.

use crate::handles::{FaceHandle, VertexHandle};
use crate::RustMesh;
use crate::{Point3, Vec3};

/// Smart vertex range with aggregation methods
pub struct VertexRange<'a> {
    mesh: &'a RustMesh,
}

impl<'a> VertexRange<'a> {
    pub fn new(mesh: &'a RustMesh) -> Self {
        Self { mesh }
    }

    /// Get iterator over vertices
    pub fn iter(&self) -> impl Iterator<Item = VertexHandle> + '_ {
        self.mesh.vertices()
    }

    /// Count vertices
    pub fn count(&self) -> usize {
        self.mesh.n_vertices()
    }

    /// Sum of positions
    pub fn sum_positions(&self) -> (f32, f32, f32) {
        let mut sum = (0.0f32, 0.0f32, 0.0f32);
        for vh in self.mesh.vertices() {
            if let Some(p) = self.mesh.point(vh) {
                sum.0 += p.x;
                sum.1 += p.y;
                sum.2 += p.z;
            }
        }
        sum
    }

    /// Average position
    pub fn average_position(&self) -> Option<(f32, f32, f32)> {
        let count = self.mesh.n_vertices();
        if count == 0 {
            return None;
        }
        let (x, y, z) = self.sum_positions();
        Some((x / count as f32, y / count as f32, z / count as f32))
    }

    /// Bounding box
    pub fn bounding_box(&self) -> Option<((f32, f32, f32), (f32, f32, f32))> {
        let mut min = (f32::MAX, f32::MAX, f32::MAX);
        let mut max = (f32::MIN, f32::MIN, f32::MIN);
        let mut has_point = false;

        for vh in self.mesh.vertices() {
            if let Some(p) = self.mesh.point(vh) {
                has_point = true;
                min.0 = min.0.min(p.x);
                min.1 = min.1.min(p.y);
                min.2 = min.2.min(p.z);
                max.0 = max.0.max(p.x);
                max.1 = max.1.max(p.y);
                max.2 = max.2.max(p.z);
            }
        }

        if has_point {
            Some((min, max))
        } else {
            None
        }
    }
}

/// Smart face range with aggregation methods
pub struct FaceRange<'a> {
    mesh: &'a RustMesh,
}

impl<'a> FaceRange<'a> {
    pub fn new(mesh: &'a RustMesh) -> Self {
        Self { mesh }
    }

    /// Get iterator over faces
    pub fn iter(&self) -> impl Iterator<Item = FaceHandle> + '_ {
        self.mesh.faces()
    }

    /// Count faces
    pub fn count(&self) -> usize {
        self.mesh.n_faces()
    }

    /// Compute face centroids
    pub fn centroids(&self) -> Vec<(f32, f32, f32)> {
        let mut centroids = Vec::new();

        for fh in self.mesh.faces() {
            // Use circulator to get face vertices
            let mut points = Vec::new();
            if let Some(verts) = self.mesh.face_vertices(fh) {
                for vh in verts {
                    if let Some(p) = self.mesh.point(vh) {
                        points.push(p);
                    }
                }
            }

            if points.len() >= 3 {
                let center = Point3::from(
                    points.iter().fold(Vec3::zeros(), |acc, p| acc + p.coords)
                        / points.len() as f32,
                );
                centroids.push((center.x, center.y, center.z));
            }
        }

        centroids
    }
}

/// Extension trait for RustMesh to provide smart ranges
pub trait SmartMesh {
    fn vertex_range(&self) -> VertexRange<'_>;
    fn face_range(&self) -> FaceRange<'_>;
}

impl SmartMesh for RustMesh {
    fn vertex_range(&self) -> VertexRange<'_> {
        VertexRange::new(self)
    }

    fn face_range(&self) -> FaceRange<'_> {
        FaceRange::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate_cube;

    #[test]
    fn test_vertex_range() {
        let mesh = generate_cube();

        // Use SmartMesh trait
        let range = mesh.vertex_range();

        assert_eq!(range.count(), 8);

        let (x, y, z) = range.sum_positions();
        println!("Sum: ({}, {}, {})", x, y, z);

        let avg = range.average_position().unwrap();
        println!("Average: ({}, {}, {})", avg.0, avg.1, avg.2);

        let bb = range.bounding_box().unwrap();
        println!("Bounding box: {:?} to {:?}", bb.0, bb.1);
    }

    #[test]
    fn test_face_range() {
        let mesh = generate_cube();

        // Use SmartMesh trait
        let range = mesh.face_range();

        assert_eq!(range.count(), 6);

        let centroids = range.centroids();
        println!("Face centroids: {}", centroids.len());
    }
}
