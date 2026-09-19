//! # Smoother - Mesh smoothing algorithms
//!
//! Provides Laplacian and Jacobi smoothing for mesh optimization.

use crate::geometry::{cotangent, triangle_angle_at};
use crate::handles::VertexHandle;
use crate::RustMesh;
use crate::{Point3, Vec3};

/// Smoothing configuration
#[derive(Debug, Clone)]
pub struct SmootherConfig {
    /// Number of smoothing iterations
    pub iterations: usize,
    /// Smoothing strength (0.0 - 1.0)
    pub strength: f32,
    /// Use uniform weight (false = use cotangent weights)
    pub uniform: bool,
    /// Boundary vertices are fixed
    pub fixed_boundary: bool,
}

impl Default for SmootherConfig {
    fn default() -> Self {
        Self {
            iterations: 10,
            strength: 0.5,
            uniform: true,
            fixed_boundary: true,
        }
    }
}

/// Laplace smoothing result
#[derive(Debug)]
pub struct SmoothResult {
    pub iterations: usize,
    pub max_displacement: f32,
}

/// Compute Laplacian of a vertex (uniform weights)
#[allow(dead_code)]
fn compute_laplacian_uniform(mesh: &RustMesh, vh: VertexHandle) -> Option<Vec3> {
    let current = mesh.point(vh)?;
    let (neighbor_sum, neighbor_count) = accumulate_neighbor_positions(mesh, vh, None);

    if neighbor_count == 0 {
        return None;
    }

    let avg = neighbor_sum / neighbor_count as f32;
    Some(avg - current.coords)
}

fn accumulate_neighbor_positions(
    mesh: &RustMesh,
    vh: VertexHandle,
    cached_positions: Option<&[Point3]>,
) -> (Vec3, usize) {
    let mut sum = Vec3::zeros();
    let mut count = 0usize;

    if let Some(vv) = mesh.vertex_vertices(vh) {
        for neighbor in vv {
            let position = cached_positions
                .and_then(|positions| positions.get(neighbor.idx_usize()).copied())
                .or_else(|| mesh.point(neighbor));

            if let Some(point) = position {
                sum += point.coords;
                count += 1;
            }
        }
    }

    (sum, count)
}

/// Check if vertex is on boundary
fn is_boundary_vertex(mesh: &RustMesh, vh: VertexHandle) -> bool {
    if let Some(heh) = mesh.halfedge_handle(vh) {
        return mesh.is_boundary(heh);
    }
    false
}

/// Laplace smoothing (explicit)
///
/// Moves each vertex towards the average of its neighbors.
///
/// Formula: p' = p + lambda * (p_avg - p)
///
/// where lambda is the smoothing strength (0-1)
pub fn laplace_smooth(mesh: &mut RustMesh, config: SmootherConfig) -> SmoothResult {
    let n_vertices = mesh.n_vertices();
    if n_vertices == 0 {
        return SmoothResult {
            iterations: 0,
            max_displacement: 0.0,
        };
    }

    // Cache mesh topology information once, mirroring OpenMesh's property-based examples.
    let vhs: Vec<VertexHandle> = mesh.vertices().collect();
    let boundary_mask: Vec<bool> = if config.fixed_boundary {
        vhs.iter().map(|&vh| is_boundary_vertex(mesh, vh)).collect()
    } else {
        vec![false; n_vertices]
    };
    let mut current_positions = vec![Point3::origin(); n_vertices];
    let mut next_positions = vec![Point3::origin(); n_vertices];
    let mut max_displacement = 0.0f32;

    for _ in 0..config.iterations {
        for (i, &vh) in vhs.iter().enumerate() {
            let point = mesh.point(vh).unwrap_or(Point3::origin());
            current_positions[i] = point;
            next_positions[i] = point;
        }

        for (i, &vh) in vhs.iter().enumerate() {
            if boundary_mask[i] {
                continue;
            }

            let (neighbor_sum, neighbor_count) =
                accumulate_neighbor_positions(mesh, vh, Some(&current_positions));
            if neighbor_count == 0 {
                continue;
            }

            let avg = neighbor_sum / neighbor_count as f32;
            let displacement = (avg - current_positions[i].coords) * config.strength;
            max_displacement = max_displacement.max(displacement.norm());
            next_positions[i] = current_positions[i] + displacement;
        }

        for (i, &vh) in vhs.iter().enumerate() {
            if next_positions[i] != current_positions[i] {
                mesh.set_point(vh, next_positions[i]);
            }
        }
    }

    SmoothResult {
        iterations: config.iterations,
        max_displacement,
    }
}

/// Tangential smoothing
///
/// Similar to Laplacian but constrains movement to be tangential
/// to preserve volume better.
pub fn tangential_smooth(mesh: &mut RustMesh, config: SmootherConfig) -> SmoothResult {
    let n_vertices = mesh.n_vertices();
    if n_vertices == 0 {
        return SmoothResult {
            iterations: 0,
            max_displacement: 0.0,
        };
    }

    let vhs: Vec<VertexHandle> = mesh.vertices().collect();
    let boundary_mask: Vec<bool> = if config.fixed_boundary {
        vhs.iter().map(|&vh| is_boundary_vertex(mesh, vh)).collect()
    } else {
        vec![false; n_vertices]
    };
    let mut current_positions = vec![Point3::origin(); n_vertices];
    let mut next_positions = vec![Point3::origin(); n_vertices];
    let mut max_displacement = 0.0f32;

    for _ in 0..config.iterations {
        let mut centroid_sum = Vec3::zeros();
        for (i, &vh) in vhs.iter().enumerate() {
            let point = mesh.point(vh).unwrap_or(Point3::origin());
            current_positions[i] = point;
            next_positions[i] = point;
            centroid_sum += point.coords;
        }
        let centroid = Point3::from(centroid_sum / n_vertices as f32);

        for (i, &vh) in vhs.iter().enumerate() {
            if boundary_mask[i] {
                continue;
            }

            let (neighbor_sum, neighbor_count) =
                accumulate_neighbor_positions(mesh, vh, Some(&current_positions));
            if neighbor_count == 0 {
                continue;
            }

            let laplacian = neighbor_sum / neighbor_count as f32 - current_positions[i].coords;
            let normal = (current_positions[i] - centroid)
                .try_normalize(f32::EPSILON)
                .unwrap_or_else(Vec3::zeros);
            let tangential = laplacian - normal * laplacian.dot(&normal);
            let displacement = tangential * config.strength;
            max_displacement = max_displacement.max(displacement.norm());
            next_positions[i] = current_positions[i] + displacement;
        }

        for (i, &vh) in vhs.iter().enumerate() {
            if next_positions[i] != current_positions[i] {
                mesh.set_point(vh, next_positions[i]);
            }
        }
    }

    SmoothResult {
        iterations: config.iterations,
        max_displacement,
    }
}

/// Compute cotangent weights for Laplacian smoothing (more accurate)
///
/// Note: This is computationally expensive but more accurate for
/// non-uniform meshes.
pub fn cotangent_weight_laplacian(mesh: &RustMesh, vh: VertexHandle) -> Option<Vec3> {
    let current = mesh.point(vh)?;

    // Get all outgoing halfedges from this vertex
    let halfedges: Vec<_> = mesh.vertex_halfedges(vh)?.collect();
    if halfedges.is_empty() {
        return None;
    }

    let mut laplacian = Vec3::zeros();
    let mut total_weight = 0.0f32;

    for heh in halfedges {
        // Get neighbor vertex
        let neighbor = mesh.to_vertex_handle(heh);
        let neighbor_pos = mesh.point(neighbor)?;

        // Get the opposite halfedge
        let opp = heh.opposite();

        // Find the two opposite vertices in the adjacent triangles
        // For the triangle on the heh side
        let next_heh = mesh.next_halfedge_handle(heh);
        let opp0 = mesh.to_vertex_handle(next_heh);
        let opp0_pos = mesh.point(opp0)?;

        // For the triangle on the opposite side
        let mut cot_sum = 0.0;

        // Compute cotangent for the first triangle
        let p0 = current;
        let p1 = neighbor_pos;
        let p2 = opp0_pos;

        // Angle at opp0 in triangle (vh, neighbor, opp0)
        let alpha = triangle_angle_at(p2, p0, p1);
        if alpha > 1e-6 && alpha < std::f32::consts::PI - 1e-6 {
            cot_sum += cotangent(alpha);
        }

        // For the second triangle (on the opposite halfedge side)
        if let Some(_opp_face) = mesh.face_handle(opp) {
            // This halfedge has a face, so there's an opposite triangle
            let next_opp = mesh.next_halfedge_handle(opp);
            let opp1 = mesh.to_vertex_handle(next_opp);
            if let Some(opp1_pos) = mesh.point(opp1) {
                // Angle at opp1 in triangle (neighbor, vh, opp1)
                let beta = triangle_angle_at(opp1_pos, p1, p0);
                if beta > 1e-6 && beta < std::f32::consts::PI - 1e-6 {
                    cot_sum += cotangent(beta);
                }
            }
        }

        // Clamp negative weights to zero (for obtuse triangles)
        let weight = cot_sum.max(0.0);

        laplacian += (neighbor_pos - current) * weight;
        total_weight += weight;
    }

    if total_weight > 1e-10 {
        Some(laplacian / total_weight)
    } else {
        None
    }
}

/// Laplace smoothing with cotangent weights (more accurate)
///
/// Uses cotangent weights for better preservation of mesh features
pub fn cotangent_smooth(mesh: &mut RustMesh, config: SmootherConfig) -> SmoothResult {
    let n_vertices = mesh.n_vertices();
    if n_vertices == 0 {
        return SmoothResult {
            iterations: 0,
            max_displacement: 0.0,
        };
    }

    let vhs: Vec<VertexHandle> = mesh.vertices().collect();
    let boundary_mask: Vec<bool> = if config.fixed_boundary {
        vhs.iter().map(|&vh| is_boundary_vertex(mesh, vh)).collect()
    } else {
        vec![false; n_vertices]
    };

    let mut current_positions = vec![Point3::origin(); n_vertices];
    let mut next_positions = vec![Point3::origin(); n_vertices];
    let mut max_displacement = 0.0f32;

    for _ in 0..config.iterations {
        // Cache current positions
        for (i, &vh) in vhs.iter().enumerate() {
            let point = mesh.point(vh).unwrap_or(Point3::origin());
            current_positions[i] = point;
            next_positions[i] = point;
        }

        // Compute cotangent-weighted Laplacian
        for (i, &vh) in vhs.iter().enumerate() {
            if boundary_mask[i] {
                continue;
            }

            if let Some(laplacian) = cotangent_weight_laplacian(mesh, vh) {
                let displacement = laplacian * config.strength;
                max_displacement = max_displacement.max(displacement.norm());
                next_positions[i] = current_positions[i] + displacement;
            }
        }

        // Apply new positions
        for (i, &vh) in vhs.iter().enumerate() {
            if next_positions[i] != current_positions[i] {
                mesh.set_point(vh, next_positions[i]);
            }
        }
    }

    SmoothResult {
        iterations: config.iterations,
        max_displacement,
    }
}

/// Estimate mean curvature at a vertex using cotangent weights
///
/// The mean curvature is approximated by the length of the cotangent-weighted Laplacian.
/// Higher values indicate sharper features (edges, corners).
pub fn estimate_mean_curvature(mesh: &RustMesh, vh: VertexHandle) -> Option<f32> {
    let laplacian = cotangent_weight_laplacian(mesh, vh)?;
    Some(laplacian.norm())
}

/// Compute curvature for all vertices
///
/// Returns a vector of curvature values (0.0 for boundary or invalid vertices)
pub fn compute_curvatures(mesh: &RustMesh) -> Vec<f32> {
    let n_vertices = mesh.n_vertices();
    let mut curvatures = vec![0.0f32; n_vertices];

    for vh in mesh.vertices() {
        let idx = vh.idx_usize();
        if idx < n_vertices {
            curvatures[idx] = estimate_mean_curvature(mesh, vh).unwrap_or(0.0);
        }
    }

    curvatures
}

/// Adaptive smoothing configuration
#[derive(Debug, Clone)]
pub struct AdaptiveSmootherConfig {
    /// Number of smoothing iterations
    pub iterations: usize,
    /// Minimum smoothing strength (for high curvature areas)
    pub min_strength: f32,
    /// Maximum smoothing strength (for low curvature areas)
    pub max_strength: f32,
    /// Curvature threshold for minimum strength
    pub curvature_high: f32,
    /// Curvature threshold for maximum strength
    pub curvature_low: f32,
    /// Boundary vertices are fixed
    pub fixed_boundary: bool,
}

impl Default for AdaptiveSmootherConfig {
    fn default() -> Self {
        Self {
            iterations: 10,
            min_strength: 0.1,
            max_strength: 0.8,
            curvature_high: 0.5,
            curvature_low: 0.01,
            fixed_boundary: true,
        }
    }
}

/// Adaptive Laplace smoothing
///
/// Adjusts smoothing strength based on local curvature:
/// - High curvature areas (sharp features) get less smoothing to preserve edges
/// - Low curvature areas (flat regions) get more smoothing to reduce noise
///
/// The strength interpolation formula:
///   strength = min_strength + (max_strength - min_strength) * (1 - t)
///   where t = clamp((curvature - curvature_low) / (curvature_high - curvature_low), 0, 1)
pub fn adaptive_smooth(mesh: &mut RustMesh, config: AdaptiveSmootherConfig) -> SmoothResult {
    let n_vertices = mesh.n_vertices();
    if n_vertices == 0 {
        return SmoothResult {
            iterations: 0,
            max_displacement: 0.0,
        };
    }

    let vhs: Vec<VertexHandle> = mesh.vertices().collect();
    let boundary_mask: Vec<bool> = if config.fixed_boundary {
        vhs.iter().map(|&vh| is_boundary_vertex(mesh, vh)).collect()
    } else {
        vec![false; n_vertices]
    };

    let mut current_positions = vec![Point3::origin(); n_vertices];
    let mut next_positions = vec![Point3::origin(); n_vertices];
    let mut max_displacement = 0.0f32;

    // Precompute curvature range
    let curvature_range = config.curvature_high - config.curvature_low;
    let strength_range = config.max_strength - config.min_strength;

    for _ in 0..config.iterations {
        // Cache current positions
        for (i, &vh) in vhs.iter().enumerate() {
            let point = mesh.point(vh).unwrap_or(Point3::origin());
            current_positions[i] = point;
            next_positions[i] = point;
        }

        // Compute adaptive smoothing
        for (i, &vh) in vhs.iter().enumerate() {
            if boundary_mask[i] {
                continue;
            }

            // Compute curvature and Laplacian
            let curvature = estimate_mean_curvature(mesh, vh).unwrap_or(0.0);
            let laplacian = match cotangent_weight_laplacian(mesh, vh) {
                Some(l) => l,
                None => continue,
            };

            // Compute adaptive strength
            let t = if curvature_range > 1e-10 {
                ((curvature - config.curvature_low) / curvature_range).clamp(0.0, 1.0)
            } else {
                0.5
            };

            // Higher curvature -> lower strength (t close to 1)
            // Lower curvature -> higher strength (t close to 0)
            let strength = config.min_strength + strength_range * (1.0 - t);

            let displacement = laplacian * strength;
            max_displacement = max_displacement.max(displacement.norm());
            next_positions[i] = current_positions[i] + displacement;
        }

        // Apply new positions
        for (i, &vh) in vhs.iter().enumerate() {
            if next_positions[i] != current_positions[i] {
                mesh.set_point(vh, next_positions[i]);
            }
        }
    }

    SmoothResult {
        iterations: config.iterations,
        max_displacement,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate_cube;
    use crate::generate_sphere;

    #[test]
    fn test_laplace_smooth() {
        let mut mesh = generate_cube();
        let original_count = mesh.n_vertices();

        let config = SmootherConfig {
            iterations: 2,
            strength: 0.5,
            uniform: true,
            fixed_boundary: true,
        };

        let result = laplace_smooth(&mut mesh, config);

        println!(
            "Laplace smooth: {} iterations, {} vertices",
            result.iterations,
            mesh.n_vertices()
        );

        // Count should not change
        assert_eq!(mesh.n_vertices(), original_count);
    }

    #[test]
    fn test_tangential_smooth() {
        let mut mesh = generate_cube();
        let original_count = mesh.n_vertices();

        let config = SmootherConfig {
            iterations: 2,
            strength: 0.5,
            uniform: true,
            fixed_boundary: true,
        };

        let result = tangential_smooth(&mut mesh, config);

        println!(
            "Tangential smooth: {} iterations, {} vertices",
            result.iterations,
            mesh.n_vertices()
        );

        assert_eq!(mesh.n_vertices(), original_count);
    }

    #[test]
    fn test_cotangent_smooth() {
        // Use a sphere which is a closed mesh (no boundary issues)
        let mut mesh = generate_sphere(1.0, 16, 16);
        let original_count = mesh.n_vertices();

        let config = SmootherConfig {
            iterations: 2,
            strength: 0.5,
            uniform: false,
            fixed_boundary: false,
        };

        let result = cotangent_smooth(&mut mesh, config);

        println!(
            "Cotangent smooth: {} iterations, {} vertices, max displacement: {}",
            result.iterations,
            mesh.n_vertices(),
            result.max_displacement
        );

        // Count should not change
        assert_eq!(mesh.n_vertices(), original_count);
    }

    #[test]
    fn test_cotangent_weight_laplacian() {
        // Create a simple triangle mesh
        let mut mesh = RustMesh::new();
        let v0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(Point3::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(Point3::new(0.0, 1.0, 0.0));
        let v3 = mesh.add_vertex(Point3::new(1.0, 1.0, 0.0));

        mesh.add_face(&[v0, v1, v2]);
        mesh.add_face(&[v1, v3, v2]);

        // Compute Laplacian for v1 (interior vertex with 4 neighbors)
        let laplacian = cotangent_weight_laplacian(&mesh, v1);

        println!("Cotangent Laplacian for v1: {:?}", laplacian);

        // The Laplacian should exist
        assert!(laplacian.is_some());
    }

    #[test]
    fn test_estimate_mean_curvature() {
        let mut mesh = RustMesh::new();
        let v0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(Point3::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(Point3::new(0.0, 1.0, 0.0));
        let v3 = mesh.add_vertex(Point3::new(1.0, 1.0, 0.0));

        mesh.add_face(&[v0, v1, v2]);
        mesh.add_face(&[v1, v3, v2]);

        // Interior vertex should have finite curvature
        let curvature = estimate_mean_curvature(&mesh, v1);
        println!("Mean curvature at v1: {:?}", curvature);
        assert!(curvature.is_some());
    }

    #[test]
    fn test_compute_curvatures() {
        let mesh = generate_sphere(1.0, 16, 16);
        let curvatures = compute_curvatures(&mesh);

        // All curvature values should be non-negative
        for &c in &curvatures {
            assert!(c >= 0.0);
        }

        // For a sphere, curvature should be relatively uniform
        let non_zero: Vec<f32> = curvatures.iter().copied().filter(|&c| c > 0.0).collect();
        if !non_zero.is_empty() {
            let mean: f32 = non_zero.iter().sum::<f32>() / non_zero.len() as f32;
            println!("Mean curvature on sphere: {}", mean);
        }
    }

    #[test]
    fn test_adaptive_smooth() {
        let mut mesh = generate_sphere(1.0, 16, 16);
        let original_count = mesh.n_vertices();

        let config = AdaptiveSmootherConfig {
            iterations: 3,
            min_strength: 0.1,
            max_strength: 0.5,
            curvature_high: 0.3,
            curvature_low: 0.05,
            fixed_boundary: false,
        };

        let result = adaptive_smooth(&mut mesh, config);

        println!(
            "Adaptive smooth: {} iterations, {} vertices, max displacement: {}",
            result.iterations,
            mesh.n_vertices(),
            result.max_displacement
        );

        // Count should not change
        assert_eq!(mesh.n_vertices(), original_count);
    }

    #[test]
    fn test_adaptive_smooth_preserves_features() {
        // Create a mesh with sharp features (a cube)
        let mut mesh = generate_cube();

        // Store original edge positions
        let original_positions: Vec<_> = mesh
            .vertices()
            .map(|vh| mesh.point(vh).unwrap_or(Point3::origin()))
            .collect();

        let config = AdaptiveSmootherConfig {
            iterations: 5,
            min_strength: 0.01, // Very low for high curvature
            max_strength: 0.3,
            curvature_high: 0.5,
            curvature_low: 0.01,
            fixed_boundary: true,
        };

        let _ = adaptive_smooth(&mut mesh, config);

        // Cube corners (high curvature) should be well preserved
        let new_positions: Vec<_> = mesh
            .vertices()
            .map(|vh| mesh.point(vh).unwrap_or(Point3::origin()))
            .collect();

        // Check that vertices haven't moved too much
        for (orig, new) in original_positions.iter().zip(new_positions.iter()) {
            let displacement = (*new - *orig).norm();
            // Adaptive smooth with fixed boundary should preserve corners
            assert!(
                displacement < 0.1,
                "Vertex moved too much: {}",
                displacement
            );
        }
    }
}
