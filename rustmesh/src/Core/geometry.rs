//! # Geometry
//!
//! Geometric utilities for mesh operations.

use crate::Point3 as Point;
use crate::{Point3, Vec3};
pub type Vector = Vec3;
pub type Normal = Vec3;

/// Calculate the centroid of a triangle
#[inline]
pub fn triangle_centroid(p0: Point, p1: Point, p2: Point) -> Point {
    Point3::from((p0.coords + p1.coords + p2.coords) / 3.0)
}

/// Calculate the area of a triangle using cross product
#[inline]
pub fn triangle_area(p0: Point, p1: Point, p2: Point) -> f32 {
    (p1 - p0).cross(&(p2 - p0)).norm() * 0.5
}

/// Calculate the normal of a triangle
#[inline]
pub fn triangle_normal(p0: Point, p1: Point, p2: Point) -> Vector {
    (p1 - p0).cross(&(p2 - p0)).normalize()
}

/// Check if a 2D point is inside a triangle (barycentric technique)
#[inline]
pub fn point_in_triangle_2d(p: Point, a: Point, b: Point, c: Point) -> bool {
    let v0 = c - a;
    let v1 = b - a;
    let v2 = p - a;

    let dot00 = v0.dot(&v0);
    let dot01 = v0.dot(&v1);
    let dot02 = v0.dot(&v2);
    let dot11 = v1.dot(&v1);
    let dot12 = v1.dot(&v2);

    let inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

    u >= 0.0 && v >= 0.0 && (u + v) <= 1.0
}

/// Calculate the bounding box of a point set
#[inline]
pub fn bounding_box(points: &[Point]) -> (Point, Point) {
    if points.is_empty() {
        return (Point3::origin(), Point3::origin());
    }

    let mut min = points[0];
    let mut max = points[0];

    for &p in points {
        min = Point3::from(min.coords.inf(&p.coords));
        max = Point3::from(max.coords.sup(&p.coords));
    }

    (min, max)
}

/// Calculate the center of mass of a point set
#[inline]
pub fn center_of_mass(points: &[Point]) -> Point {
    if points.is_empty() {
        return Point3::origin();
    }

    let sum = points.iter().fold(Vec3::zeros(), |acc, p| acc + p.coords);
    Point3::from(sum / points.len() as f32)
}

/// Calculate the squared distance between two points
#[inline]
pub fn squared_distance(p0: Point, p1: Point) -> f32 {
    (p0 - p1).norm_squared()
}

/// Calculate the distance between two points
#[inline]
pub fn distance(p0: Point, p1: Point) -> f32 {
    (p0 - p1).norm()
}

/// Normalize a vector
#[inline]
pub fn normalize(v: Vector) -> Vector {
    v.normalize()
}

/// Calculate the dot product of two vectors
#[inline]
pub fn dot(v0: Vector, v1: Vector) -> f32 {
    v0.dot(&v1)
}

/// Calculate the cross product of two vectors
#[inline]
pub fn cross(v0: Vector, v1: Vector) -> Vector {
    v0.cross(&v1)
}

/// Calculate the length of a vector
#[inline]
pub fn length(v: Vector) -> f32 {
    v.norm()
}

/// Calculate the squared length of a vector
#[inline]
pub fn squared_length(v: Vector) -> f32 {
    v.norm_squared()
}

// ============================================================================
// Triangle Angle Functions (E2-S1)
// ============================================================================

/// Calculate the angle at vertex p0 in a triangle
///
/// Returns angle in radians
#[inline]
pub fn triangle_angle_at(p0: Point, p1: Point, p2: Point) -> f32 {
    let v01 = p1 - p0;
    let v02 = p2 - p0;

    let len01 = v01.norm();
    let len02 = v02.norm();

    if len01 < 1e-10 || len02 < 1e-10 {
        return 0.0;
    }

    let cos_angle = (v01.dot(&v02) / (len01 * len02)).clamp(-1.0, 1.0);
    cos_angle.acos()
}

/// Calculate all three angles of a triangle
///
/// Returns (angle_at_p0, angle_at_p1, angle_at_p2) in radians
#[inline]
pub fn triangle_angles(p0: Point, p1: Point, p2: Point) -> (f32, f32, f32) {
    let a0 = triangle_angle_at(p0, p1, p2);
    let a1 = triangle_angle_at(p1, p2, p0);
    let a2 = triangle_angle_at(p2, p0, p1);
    (a0, a1, a2)
}

/// Calculate the cotangent of an angle (used in Laplacian smoothing)
///
/// cot(θ) = cos(θ) / sin(θ) = adjacent / opposite
#[inline]
pub fn cotangent(angle: f32) -> f32 {
    let sin_val = angle.sin();
    if sin_val.abs() < 1e-10 {
        return 0.0; // Avoid division by zero for very small angles
    }
    angle.cos() / sin_val
}

/// Calculate the cotangent weight for an edge in a mesh
///
/// For an edge between vertices v0 and v1, the cotangent weight is:
/// w = (cot(α) + cot(β)) / 2
/// where α and β are the angles opposite to the edge in the two adjacent triangles
///
/// This requires the positions of the two opposite vertices in the adjacent triangles
#[inline]
pub fn cotangent_weight(
    v0: Point,
    v1: Point,
    opposite0: Point, // opposite vertex in first triangle (v0, v1, opposite0)
    opposite1: Point, // opposite vertex in second triangle (v1, v0, opposite1)
) -> f32 {
    // In triangle (v0, v1, opposite0), the angle at opposite0 is α
    let alpha = triangle_angle_at(opposite0, v0, v1);
    let cot_alpha = cotangent(alpha);

    // In triangle (v1, v0, opposite1), the angle at opposite1 is β
    let beta = triangle_angle_at(opposite1, v1, v0);
    let cot_beta = cotangent(beta);

    (cot_alpha + cot_beta) * 0.5
}

/// Calculate the quality of a triangle (aspect ratio)
///
/// Returns a value between 0 (degenerate) and 1 (equilateral)
#[inline]
pub fn triangle_quality(p0: Point, p1: Point, p2: Point) -> f32 {
    let a = distance(p0, p1);
    let b = distance(p1, p2);
    let c = distance(p2, p0);

    let s = (a + b + c) * 0.5; // semi-perimeter
    let area = triangle_area(p0, p1, p2);

    if s < 1e-10 {
        return 0.0;
    }

    // Quality = 4 * sqrt(3) * area / (a^2 + b^2 + c^2)
    // For equilateral triangle, this equals 1
    let sum_sq = a * a + b * b + c * c;
    if sum_sq < 1e-10 {
        return 0.0;
    }

    (4.0 * (3.0_f32).sqrt() * area) / sum_sq
}

/// Calculate the circumradius of a triangle
#[inline]
pub fn triangle_circumradius(p0: Point, p1: Point, p2: Point) -> f32 {
    let a = distance(p0, p1);
    let b = distance(p1, p2);
    let c = distance(p2, p0);

    let area = triangle_area(p0, p1, p2);
    if area < 1e-10 {
        return 0.0;
    }

    // R = (a * b * c) / (4 * area)
    (a * b * c) / (4.0 * area)
}

/// Calculate the inradius of a triangle
#[inline]
pub fn triangle_inradius(p0: Point, p1: Point, p2: Point) -> f32 {
    let a = distance(p0, p1);
    let b = distance(p1, p2);
    let c = distance(p2, p0);

    let s = (a + b + c) * 0.5; // semi-perimeter
    let area = triangle_area(p0, p1, p2);

    if s < 1e-10 {
        return 0.0;
    }

    // r = area / s
    area / s
}

/// Calculate the dihedral angle between two triangles sharing an edge
///
/// The edge is defined by v0-v1, with opposite vertices o0 and o1
/// Returns angle in radians
#[inline]
pub fn dihedral_angle(
    v0: Point,
    v1: Point,
    o0: Point, // opposite vertex in first triangle
    o1: Point, // opposite vertex in second triangle
) -> f32 {
    let n0 = triangle_normal(o0, v0, v1);
    let n1 = triangle_normal(o1, v1, v0);

    let cos_angle = n0.dot(&n1).clamp(-1.0, 1.0);
    cos_angle.acos()
}

// ============================================================================
// Bounding Sphere (E8-S1)
// ============================================================================

/// Calculate a bounding sphere for a point set using Ritter's algorithm
///
/// This is an approximation that runs in O(n) time.
/// Returns (center, radius)
pub fn bounding_sphere(points: &[Point]) -> (Point, f32) {
    if points.is_empty() {
        return (Point3::origin(), 0.0);
    }

    if points.len() == 1 {
        return (points[0], 0.0);
    }

    // Step 1: Find the min and max points along each axis
    let mut min_x = points[0];
    let mut max_x = points[0];
    let mut min_y = points[0];
    let mut max_y = points[0];
    let mut min_z = points[0];
    let mut max_z = points[0];

    for &p in points {
        if p.x < min_x.x {
            min_x = p;
        }
        if p.x > max_x.x {
            max_x = p;
        }
        if p.y < min_y.y {
            min_y = p;
        }
        if p.y > max_y.y {
            max_y = p;
        }
        if p.z < min_z.z {
            min_z = p;
        }
        if p.z > max_z.z {
            max_z = p;
        }
    }

    // Step 2: Find the pair with the largest distance
    let dist_x = distance(min_x, max_x);
    let dist_y = distance(min_y, max_y);
    let dist_z = distance(min_z, max_z);

    let (p1, p2) = if dist_x >= dist_y && dist_x >= dist_z {
        (min_x, max_x)
    } else if dist_y >= dist_z {
        (min_y, max_y)
    } else {
        (min_z, max_z)
    };

    // Step 3: Initial sphere from the two points
    let mut center = Point3::from((p1.coords + p2.coords) * 0.5);
    let mut radius = distance(p1, p2) * 0.5;

    // Step 4: Expand sphere to include all points
    for &p in points {
        let d = distance(p, center);
        if d > radius {
            // Expand the sphere to include this point
            let direction = (p - center)
                .try_normalize(f32::EPSILON)
                .unwrap_or_else(Vec3::zeros);
            let new_radius = (radius + d) * 0.5;
            center = center + direction * (new_radius - radius);
            radius = new_radius;
        }
    }

    (center, radius)
}

/// Calculate the minimum enclosing sphere using an iterative refinement
///
/// This is more accurate than bounding_sphere but slower.
pub fn minimum_enclosing_sphere(points: &[Point], iterations: usize) -> (Point, f32) {
    let (mut center, mut radius) = bounding_sphere(points);

    // Iteratively refine the sphere
    for _ in 0..iterations {
        // Find the point farthest from the center
        let mut farthest_dist = radius;
        let mut farthest_point = center;

        for &p in points {
            let d = distance(p, center);
            if d > farthest_dist {
                farthest_dist = d;
                farthest_point = p;
            }
        }

        // Move center towards the farthest point slightly
        if farthest_dist > radius {
            let direction = (farthest_point - center)
                .try_normalize(f32::EPSILON)
                .unwrap_or_else(Vec3::zeros);
            center = center + direction * (farthest_dist - radius) * 0.5;
            radius = (radius + farthest_dist) * 0.5;
        }
    }

    (center, radius)
}

// ============================================================================
// Curvature Estimation Functions (E8-S2, E8-S3, E8-S4)
// ============================================================================

/// Calculate the Voronoi area for a vertex in a mesh
///
/// The Voronoi area is the area of the region around a vertex that is
/// closer to that vertex than to any other vertex.
///
/// For a vertex with neighboring triangles, this is computed using
/// the circumcenters of the triangles.
pub fn voronoi_area(p_center: Point, neighbors: &[Point]) -> f32 {
    if neighbors.len() < 3 {
        return 0.0;
    }

    let n = neighbors.len();
    let mut total_area = 0.0;

    // For each triangle (p_center, neighbors[i], neighbors[(i+1)%n])
    for i in 0..n {
        let p1 = neighbors[i];
        let p2 = neighbors[(i + 1) % n];

        // Calculate the angles at p1 and p2
        let angle1 = triangle_angle_at(p1, p_center, p2);
        let angle2 = triangle_angle_at(p2, p_center, p1);

        // For obtuse triangles, use mixed Voronoi area
        let angle_center = triangle_angle_at(p_center, p1, p2);

        if angle_center >= std::f32::consts::PI / 2.0 {
            // Obtuse angle at center: use half the triangle area
            total_area += triangle_area(p_center, p1, p2) * 0.5;
        } else if angle1 >= std::f32::consts::PI / 2.0 || angle2 >= std::f32::consts::PI / 2.0 {
            // Obtuse angle elsewhere: use half the triangle area
            total_area += triangle_area(p_center, p1, p2) * 0.5;
        } else {
            // Acute triangle: use proper Voronoi area
            // Area = (|p_center - p1|^2 * cot(angle2) + |p_center - p2|^2 * cot(angle1)) / 8
            let d1_sq = (p_center - p1).norm_squared();
            let d2_sq = (p_center - p2).norm_squared();

            let cot1 = cotangent(angle1);
            let cot2 = cotangent(angle2);

            total_area += (d1_sq * cot2 + d2_sq * cot1) / 8.0;
        }
    }

    total_area.max(1e-10) // Avoid division by zero
}

/// Estimate the Gaussian curvature at a vertex
///
/// Gaussian curvature K = (2π - Σθi) / Ai
/// where θi are the angles at the vertex in each incident triangle,
/// and Ai is the Voronoi area.
pub fn gaussian_curvature(p_center: Point, neighbors: &[Point]) -> f32 {
    if neighbors.len() < 3 {
        return 0.0;
    }

    let mut angle_sum = 0.0;

    // Sum up angles at the center vertex
    let n = neighbors.len();
    for i in 0..n {
        let p1 = neighbors[i];
        let p2 = neighbors[(i + 1) % n];
        angle_sum += triangle_angle_at(p_center, p1, p2);
    }

    let angle_defect = 2.0 * std::f32::consts::PI - angle_sum;
    let v_area = voronoi_area(p_center, neighbors);

    angle_defect / v_area
}

/// Estimate the mean curvature at a vertex using the Laplace-Beltrami operator
///
/// Mean curvature H = ||Δp|| / 2
/// where Δp is the Laplace-Beltrami of the position.
pub fn mean_curvature(p_center: Point, neighbors: &[Point]) -> f32 {
    if neighbors.len() < 3 {
        return 0.0;
    }

    // Compute the Laplace-Beltrami using cotangent weights
    let mut laplacian = Vector::zeros();
    let mut total_weight = 0.0;

    let n = neighbors.len();
    for i in 0..n {
        let p1 = neighbors[i];
        let p2 = neighbors[(i + 1) % n];
        let p0 = neighbors[(i + n - 1) % n];

        // Compute cotangent weights for the edge (p_center, p1)
        // α is the angle at p2 in triangle (p_center, p1, p2)
        // β is the angle at p0 in triangle (p_center, p0, p1)
        let alpha = triangle_angle_at(p2, p_center, p1);
        let beta = triangle_angle_at(p0, p_center, p1);

        let cot_alpha = cotangent(alpha);
        let cot_beta = cotangent(beta);

        let weight = (cot_alpha + cot_beta) * 0.5;
        total_weight += weight;

        laplacian += (p1 - p_center) * weight;
    }

    if total_weight > 1e-10 {
        laplacian /= total_weight;
    }

    // Mean curvature = half the magnitude of the Laplace-Beltrami
    laplacian.norm() * 0.5
}

/// Calculate the two principal curvatures from Gaussian and mean curvatures
///
/// k1, k2 = H ± sqrt(H² - K)
/// Returns (k1, k2) where k1 >= k2
pub fn principal_curvatures(gaussian: f32, mean: f32) -> (f32, f32) {
    let discriminant = mean * mean - gaussian;

    if discriminant < 0.0 {
        // Complex curvatures (shouldn't happen for valid surfaces)
        return (mean, mean);
    }

    let sqrt_disc = discriminant.sqrt();
    let k1 = mean + sqrt_disc;
    let k2 = mean - sqrt_disc;

    (k1, k2)
}

/// Calculate the curvature directions (shape index and curvedness)
///
/// Shape index: S = -2/π * arctan((k1 + k2) / (k1 - k2))
/// Ranges from +1 (cup/concave) to -1 (cap/convex)
///
/// Curvedness: C = sqrt((k1² + k2²) / 2)
pub fn curvature_descriptors(k1: f32, k2: f32) -> (f32, f32) {
    let shape_index = if (k1 - k2).abs() < 1e-10 {
        let mean_sign = k1 + k2;
        if mean_sign.abs() < 1e-10 {
            0.0
        } else if mean_sign < 0.0 {
            1.0
        } else {
            -1.0
        }
    } else {
        -2.0 / std::f32::consts::PI * ((k1 + k2) / (k1 - k2)).atan()
    };

    let curvedness = ((k1 * k1 + k2 * k2) * 0.5).sqrt();

    (shape_index, curvedness)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, Vector3};

    #[test]
    fn point_and_vector_have_distinct_semantics() {
        let p0 = Point3::new(1.0, 2.0, 3.0);
        let p1 = Point3::new(3.0, 5.0, 7.0);
        let delta: Vector3<f32> = p1 - p0;
        assert_eq!(delta, Vector3::new(2.0, 3.0, 4.0));
        assert_eq!(p0 + delta, p1);
    }

    #[test]
    fn centroid_returns_a_point() {
        let c = triangle_centroid(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(3.0, 0.0, 0.0),
            Point3::new(0.0, 3.0, 0.0),
        );
        assert_eq!(c, Point3::new(1.0, 1.0, 0.0));
    }

    #[test]
    fn point_translation_includes_translation_vector_does_not() {
        // Non-origin translation: transforming a point includes translation;
        // transforming a direction/normal does not.
        let translation = Vector3::new(10.0, -4.0, 7.0);
        let rotation = nalgebra::Rotation3::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.2, 0.5, 0.8)),
            0.7,
        );
        let transform = nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::from(translation),
            nalgebra::UnitQuaternion::from(rotation),
        );

        let point = Point3::new(1.0, 2.0, 3.0);
        let direction = Vector3::new(0.0, 1.0, 0.0);

        let transformed_point = transform * point;
        let rotated_only = Point3::from(rotation * point.coords);
        assert!((transformed_point - (rotated_only + translation)).norm() < 1e-5);

        let transformed_direction = transform * direction;
        assert!((transformed_direction - (rotation * direction)).norm() < 1e-5);
        assert!((transformed_direction - direction).norm() > 0.1);
    }

    #[test]
    fn test_triangle_area() {
        let p0 = Point3::new(0.0, 0.0, 0.0);
        let p1 = Point3::new(1.0, 0.0, 0.0);
        let p2 = Point3::new(0.0, 1.0, 0.0);

        // Right triangle with legs of length 1
        assert!((triangle_area(p0, p1, p2) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_triangle_normal() {
        let p0 = Point3::new(0.0, 0.0, 0.0);
        let p1 = Point3::new(1.0, 0.0, 0.0);
        let p2 = Point3::new(0.0, 1.0, 0.0);

        let normal = triangle_normal(p0, p1, p2);
        assert!((normal.norm() - 1.0).abs() < 1e-6);
        // Should point in +Z direction
        assert!(normal.z > 0.0);
    }

    #[test]
    fn test_bounding_box() {
        let points = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 2.0, 3.0),
            Point3::new(-1.0, 5.0, -2.0),
        ];

        let (min, max) = bounding_box(&points);

        assert_eq!(min, Point3::new(-1.0, 0.0, -2.0));
        assert_eq!(max, Point3::new(1.0, 5.0, 3.0));
    }

    #[test]
    fn test_triangle_angles() {
        // Equilateral triangle - all angles should be 60 degrees (π/3)
        let p0 = Point3::new(0.0, 0.0, 0.0);
        let p1 = Point3::new(1.0, 0.0, 0.0);
        let p2 = Point3::new(0.5, (3.0_f32).sqrt() / 2.0, 0.0);

        let (a0, a1, a2) = triangle_angles(p0, p1, p2);

        let expected = std::f32::consts::PI / 3.0;
        assert!((a0 - expected).abs() < 0.01);
        assert!((a1 - expected).abs() < 0.01);
        assert!((a2 - expected).abs() < 0.01);

        // Angles should sum to π
        assert!((a0 + a1 + a2 - std::f32::consts::PI).abs() < 0.01);
    }

    #[test]
    fn test_triangle_angle_right() {
        // Right triangle with 45-45-90 angles
        let p0 = Point3::new(0.0, 0.0, 0.0);
        let p1 = Point3::new(1.0, 0.0, 0.0);
        let p2 = Point3::new(0.0, 1.0, 0.0);

        let (a0, a1, a2) = triangle_angles(p0, p1, p2);

        // Angle at p0 should be 90 degrees (π/2)
        assert!((a0 - std::f32::consts::PI / 2.0).abs() < 0.01);

        // Angles at p1 and p2 should be 45 degrees (π/4)
        assert!((a1 - std::f32::consts::PI / 4.0).abs() < 0.01);
        assert!((a2 - std::f32::consts::PI / 4.0).abs() < 0.01);
    }

    #[test]
    fn test_cotangent() {
        // cot(π/4) = 1
        assert!((cotangent(std::f32::consts::PI / 4.0) - 1.0).abs() < 0.01);

        // cot(π/3) = 1/sqrt(3)
        let expected = 1.0 / (3.0_f32).sqrt();
        assert!((cotangent(std::f32::consts::PI / 3.0) - expected).abs() < 0.01);
    }

    #[test]
    fn test_cotangent_weight() {
        // Equilateral triangle: cot(60°) = 1/sqrt(3)
        let v0 = Point3::new(0.0, 0.0, 0.0);
        let v1 = Point3::new(1.0, 0.0, 0.0);
        let o0 = Point3::new(0.5, (3.0_f32).sqrt() / 2.0, 0.0);
        let o1 = Point3::new(0.5, -(3.0_f32).sqrt() / 2.0, 0.0);

        let weight = cotangent_weight(v0, v1, o0, o1);

        // For equilateral triangles, cot(60°) ≈ 0.577
        let expected_cot = 1.0 / (3.0_f32).sqrt();
        let expected = expected_cot; // (cot + cot) / 2 = cot for symmetric case

        assert!((weight - expected).abs() < 0.1);
    }

    #[test]
    fn test_triangle_quality() {
        // Equilateral triangle should have quality close to 1
        let p0 = Point3::new(0.0, 0.0, 0.0);
        let p1 = Point3::new(1.0, 0.0, 0.0);
        let p2 = Point3::new(0.5, (3.0_f32).sqrt() / 2.0, 0.0);

        let quality = triangle_quality(p0, p1, p2);
        assert!(quality > 0.99);

        // Right triangle should have lower quality
        let r0 = Point3::new(0.0, 0.0, 0.0);
        let r1 = Point3::new(1.0, 0.0, 0.0);
        let r2 = Point3::new(0.0, 1.0, 0.0);

        let right_quality = triangle_quality(r0, r1, r2);
        assert!(right_quality < quality);
        assert!(right_quality > 0.8); // Still reasonable quality
    }

    #[test]
    fn test_dihedral_angle() {
        // Two coplanar triangles
        let v0 = Point3::new(0.0, 0.0, 0.0);
        let v1 = Point3::new(1.0, 0.0, 0.0);
        let o0 = Point3::new(0.5, 1.0, 0.0);
        let o1 = Point3::new(0.5, -1.0, 0.0);

        let angle = dihedral_angle(v0, v1, o0, o1);

        // Coplanar triangles should have dihedral angle of 0
        assert!(angle.abs() < 0.01);

        // Folded triangles (90 degrees)
        let f0 = Point3::new(0.0, 0.0, 0.0);
        let f1 = Point3::new(1.0, 0.0, 0.0);
        let fo0 = Point3::new(0.5, 1.0, 0.0);
        let fo1 = Point3::new(0.5, 0.0, 1.0);

        let folded_angle = dihedral_angle(f0, f1, fo0, fo1);
        assert!((folded_angle - std::f32::consts::PI / 2.0).abs() < 0.1);
    }

    #[test]
    fn test_bounding_sphere() {
        let points = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ];

        let (center, radius) = bounding_sphere(&points);

        // All points should be inside the sphere
        for &p in &points {
            let d = distance(p, center);
            assert!(d <= radius + 0.01, "Point {:?} outside sphere", p);
        }

        println!("Bounding sphere: center={:?}, radius={}", center, radius);
    }

    #[test]
    fn test_voronoi_area() {
        // Equilateral triangle fan around center
        let center = Point3::new(0.0, 0.0, 0.0);
        let r = 1.0;
        let neighbors: Vec<Point> = (0..6)
            .map(|i| {
                let angle = i as f32 * std::f32::consts::PI / 3.0;
                Point3::new(r * angle.cos(), r * angle.sin(), 0.0)
            })
            .collect();

        let area = voronoi_area(center, &neighbors);
        println!("Voronoi area for regular hexagon vertex: {}", area);

        // For a regular 1-ring equilateral fan, the mixed Voronoi cell is a
        // regular hexagon of circumradius r / sqrt(3), with area sqrt(3) / 2 * r^2.
        let expected = 0.5 * 3.0_f32.sqrt() * r * r;
        assert!((area - expected).abs() < 0.1 * expected);
    }

    #[test]
    fn test_gaussian_curvature() {
        // For a flat region, Gaussian curvature should be ~0
        let center = Point3::new(0.0, 0.0, 0.0);
        let neighbors = [
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(-1.0, 1.0, 0.0),
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(-1.0, -1.0, 0.0),
            Point3::new(0.0, -1.0, 0.0),
            Point3::new(1.0, -1.0, 0.0),
        ];

        let k = gaussian_curvature(center, &neighbors);
        println!("Gaussian curvature for flat region: {}", k);
        assert!(
            k.abs() < 0.1,
            "Flat region should have near-zero Gaussian curvature"
        );
    }

    #[test]
    fn test_mean_curvature() {
        // For a flat region, mean curvature should be ~0
        let center = Point3::new(0.0, 0.0, 0.0);
        let neighbors = [
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(0.0, -1.0, 0.0),
        ];

        let h = mean_curvature(center, &neighbors);
        println!("Mean curvature for flat region: {}", h);
        assert!(
            h.abs() < 0.1,
            "Flat region should have near-zero mean curvature"
        );
    }

    #[test]
    fn test_principal_curvatures() {
        // Test for a saddle point (K < 0)
        let k1 = 1.0;
        let k2 = -1.0;
        let gaussian = k1 * k2;
        let mean = (k1 + k2) * 0.5;

        let (pk1, pk2) = principal_curvatures(gaussian, mean);
        println!("Principal curvatures: k1={}, k2={}", pk1, pk2);

        // For saddle point: K = -1, H = 0, k1 = 1, k2 = -1
        assert!((pk1 - 1.0).abs() < 0.01);
        assert!((pk2 + 1.0).abs() < 0.01);
    }

    #[test]
    fn test_curvature_descriptors() {
        // Saddle point
        let (shape, curved) = curvature_descriptors(1.0, -1.0);
        println!("Saddle: shape_index={}, curvedness={}", shape, curved);
        assert!(shape.abs() < 0.01, "Saddle should have shape index near 0");

        // Cup (concave)
        let (shape_cup, _) = curvature_descriptors(-1.0, -1.0);
        assert!(shape_cup > 0.5, "Cup should have positive shape index");

        // Cap (convex)
        let (shape_cap, _) = curvature_descriptors(1.0, 1.0);
        assert!(shape_cap < -0.5, "Cap should have negative shape index");
    }
}
