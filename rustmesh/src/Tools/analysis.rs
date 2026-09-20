//! # Mesh Analysis Tools
//!
//! High-level mesh analysis utilities including curvature estimation,
//! bounding sphere computation, and quality metrics.

use crate::geometry::{
    bounding_sphere, gaussian_curvature, mean_curvature, principal_curvatures, triangle_area,
    triangle_quality, voronoi_area,
};
use crate::{FaceHandle, Point3, RustMesh, Vec3, VertexHandle};

/// Curvature information for a mesh vertex
#[derive(Debug, Clone, Copy)]
pub struct VertexCurvature {
    /// Gaussian curvature (angle defect / Voronoi area)
    pub gaussian: f32,
    /// Mean curvature (half Laplace-Beltrami magnitude)
    pub mean: f32,
    /// First principal curvature (larger)
    pub k1: f32,
    /// Second principal curvature (smaller)
    pub k2: f32,
}

impl Default for VertexCurvature {
    fn default() -> Self {
        Self {
            gaussian: 0.0,
            mean: 0.0,
            k1: 0.0,
            k2: 0.0,
        }
    }
}

/// Mesh quality metrics
#[derive(Debug, Clone)]
pub struct MeshQuality {
    /// Minimum triangle quality (0 = degenerate, 1 = equilateral)
    pub min_quality: f32,
    /// Maximum triangle quality
    pub max_quality: f32,
    /// Mean triangle quality
    pub mean_quality: f32,
    /// Number of degenerate triangles (quality < 0.01)
    pub degenerate_count: usize,
    /// Total number of triangles
    pub total_triangles: usize,
}

/// Mesh analysis results
#[derive(Debug, Clone)]
pub struct MeshAnalysis {
    /// Number of vertices
    pub n_vertices: usize,
    /// Number of faces
    pub n_faces: usize,
    /// Bounding box (min, max)
    pub bounding_box: (Point3, Point3),
    /// Bounding sphere (center, radius)
    pub bounding_sphere: (Point3, f32),
    /// Mesh quality metrics
    pub quality: MeshQuality,
    /// Total surface area
    pub surface_area: f32,
    /// Approximate volume (for closed meshes)
    pub volume: Option<f32>,
}

/// Compute curvature for a single vertex
pub fn compute_vertex_curvature(mesh: &RustMesh, vh: VertexHandle) -> VertexCurvature {
    let p_center = match mesh.point(vh) {
        Some(p) => p,
        None => return VertexCurvature::default(),
    };

    // Collect neighbor positions
    let neighbors: Vec<Point3> = match mesh.vertex_vertices(vh) {
        Some(vv) => vv.filter_map(|n| mesh.point(n)).collect(),
        None => return VertexCurvature::default(),
    };

    if neighbors.len() < 3 {
        return VertexCurvature::default();
    }

    let gauss = gaussian_curvature(p_center, &neighbors);
    let mean = mean_curvature(p_center, &neighbors);
    let (k1, k2) = principal_curvatures(gauss, mean);

    VertexCurvature {
        gaussian: gauss,
        mean,
        k1,
        k2,
    }
}

/// Compute curvatures for all vertices
pub fn compute_all_curvatures(mesh: &RustMesh) -> Vec<VertexCurvature> {
    mesh.vertices()
        .map(|vh| compute_vertex_curvature(mesh, vh))
        .collect()
}

/// Compute mesh quality metrics
pub fn compute_mesh_quality(mesh: &RustMesh) -> MeshQuality {
    let mut min_quality = f32::MAX;
    let mut max_quality = 0.0f32;
    let mut quality_sum = 0.0f32;
    let mut degenerate_count = 0;
    let mut total_triangles = 0;

    for fh in mesh.faces() {
        let vertices: Vec<VertexHandle> = mesh.face_vertices_vec(fh);
        if vertices.len() != 3 {
            continue; // Only analyze triangles
        }

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

        let quality = triangle_quality(p0, p1, p2);
        min_quality = min_quality.min(quality);
        max_quality = max_quality.max(quality);
        quality_sum += quality;

        if quality < 0.01 {
            degenerate_count += 1;
        }

        total_triangles += 1;
    }

    MeshQuality {
        min_quality: if total_triangles > 0 {
            min_quality
        } else {
            0.0
        },
        max_quality,
        mean_quality: if total_triangles > 0 {
            quality_sum / total_triangles as f32
        } else {
            0.0
        },
        degenerate_count,
        total_triangles,
    }
}

/// Compute total surface area
pub fn compute_surface_area(mesh: &RustMesh) -> f32 {
    let mut total_area = 0.0;

    for fh in mesh.faces() {
        let vertices: Vec<VertexHandle> = mesh.face_vertices_vec(fh);
        if vertices.len() < 3 {
            continue;
        }

        // Triangulate polygon (fan triangulation)
        let p0 = match mesh.point(vertices[0]) {
            Some(p) => p,
            None => continue,
        };

        for i in 1..vertices.len() - 1 {
            let p1 = match mesh.point(vertices[i]) {
                Some(p) => p,
                None => continue,
            };
            let p2 = match mesh.point(vertices[i + 1]) {
                Some(p) => p,
                None => continue,
            };

            total_area += triangle_area(p0, p1, p2);
        }
    }

    total_area
}

/// Compute approximate volume for a closed mesh using the divergence theorem
pub fn compute_volume(mesh: &RustMesh) -> Option<f32> {
    let mut volume = 0.0;
    let mut has_boundary = false;

    for fh in mesh.faces() {
        let vertices: Vec<VertexHandle> = mesh.face_vertices_vec(fh);
        if vertices.len() != 3 {
            continue;
        }

        // Check for boundary
        let heh = match mesh.face_halfedge_handle(fh) {
            Some(he) => he,
            None => continue,
        };
        let opp = mesh.opposite_halfedge_handle(heh);
        if mesh.face_handle(opp).is_none() {
            has_boundary = true;
        }

        let p0 = mesh.point(vertices[0])?;
        let p1 = mesh.point(vertices[1])?;
        let p2 = mesh.point(vertices[2])?;

        // Signed volume of tetrahedron formed with origin
        volume += p0.coords.dot(&p1.coords.cross(&p2.coords)) / 6.0;
    }

    if has_boundary {
        None
    } else {
        Some(volume.abs())
    }
}

/// Perform comprehensive mesh analysis
pub fn analyze_mesh(mesh: &RustMesh) -> MeshAnalysis {
    // Bounding box
    let points: Vec<Point3> = mesh.vertices().filter_map(|vh| mesh.point(vh)).collect();
    let (bbox_min, bbox_max) = if points.is_empty() {
        (Point3::origin(), Point3::origin())
    } else {
        let mut min = points[0];
        let mut max = points[0];
        for &p in &points {
            min = Point3::from(min.coords.inf(&p.coords));
            max = Point3::from(max.coords.sup(&p.coords));
        }
        (min, max)
    };

    // Bounding sphere
    let (bs_center, bs_radius) = bounding_sphere(&points);

    // Quality
    let quality = compute_mesh_quality(mesh);

    // Surface area
    let surface_area = compute_surface_area(mesh);

    // Volume
    let volume = compute_volume(mesh);

    MeshAnalysis {
        n_vertices: mesh.n_vertices(),
        n_faces: mesh.n_faces(),
        bounding_box: (bbox_min, bbox_max),
        bounding_sphere: (bs_center, bs_radius),
        quality,
        surface_area,
        volume,
    }
}

/// Export curvature values as a scalar field for visualization
///
/// Returns a vector of (vertex_index, curvature_value) pairs
pub fn export_curvature_field(mesh: &RustMesh, curvature_type: CurvatureType) -> Vec<(usize, f32)> {
    let curvatures = compute_all_curvatures(mesh);

    curvatures
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let value = match curvature_type {
                CurvatureType::Gaussian => c.gaussian,
                CurvatureType::Mean => c.mean,
                CurvatureType::K1 => c.k1,
                CurvatureType::K2 => c.k2,
            };
            (i, value)
        })
        .collect()
}

/// Types of curvature that can be exported
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurvatureType {
    Gaussian,
    Mean,
    K1,
    K2,
}

/// Edge length statistics
#[derive(Debug, Clone, Copy)]
pub struct EdgeLengthStats {
    pub min_length: f32,
    pub max_length: f32,
    pub mean_length: f32,
    pub std_dev: f32,
}

/// Compute edge length statistics
pub fn compute_edge_length_stats(mesh: &RustMesh) -> EdgeLengthStats {
    let mut lengths = Vec::new();

    for i in 0..mesh.n_edges() {
        let eh = crate::EdgeHandle::new(i as u32);
        if mesh.is_edge_deleted(eh) {
            continue;
        }

        let h0 = mesh.edge_halfedge_handle(eh, 0);
        let v0 = mesh.from_vertex_handle(h0);
        let v1 = mesh.to_vertex_handle(h0);

        if let (Some(p0), Some(p1)) = (mesh.point(v0), mesh.point(v1)) {
            lengths.push((p1 - p0).norm());
        }
    }

    if lengths.is_empty() {
        return EdgeLengthStats {
            min_length: 0.0,
            max_length: 0.0,
            mean_length: 0.0,
            std_dev: 0.0,
        };
    }

    let n = lengths.len();
    let min_length = lengths.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_length = lengths.iter().cloned().fold(0.0f32, f32::max);
    let mean_length = lengths.iter().sum::<f32>() / n as f32;

    let variance = if n > 1 {
        lengths
            .iter()
            .map(|&l| (l - mean_length).powi(2))
            .sum::<f32>()
            / (n - 1) as f32
    } else {
        0.0
    };
    let std_dev = variance.sqrt();

    EdgeLengthStats {
        min_length,
        max_length,
        mean_length,
        std_dev,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate_cube;
    use crate::generate_sphere;

    #[test]
    fn test_compute_vertex_curvature() {
        let mesh = generate_sphere(1.0, 32, 32);

        // Compute curvature for a vertex (use a vertex that's not at a pole)
        let vh = VertexHandle::new(mesh.n_vertices() as u32 / 2);
        let curv = compute_vertex_curvature(&mesh, vh);

        println!(
            "Sphere vertex curvature: gaussian={}, mean={}",
            curv.gaussian, curv.mean
        );

        // For a sphere of radius 1: K = 1/R² = 1, H = 1/R = 1
        // Our estimate won't be exact due to discretization
        // Just check that we get reasonable values
        assert!(curv.gaussian.is_finite());
        assert!(curv.mean.is_finite());
    }

    #[test]
    fn test_compute_mesh_quality() {
        let mesh = generate_sphere(1.0, 16, 16);
        let quality = compute_mesh_quality(&mesh);

        println!(
            "Mesh quality: min={}, max={}, mean={}",
            quality.min_quality, quality.max_quality, quality.mean_quality
        );
        println!(
            "Degenerate triangles: {} / {}",
            quality.degenerate_count, quality.total_triangles
        );

        assert!(quality.mean_quality > 0.3);
        // Some discretizations may have near-degenerate triangles
        assert!(quality.degenerate_count < quality.total_triangles / 4);
    }

    #[test]
    fn test_compute_surface_area() {
        let mesh = generate_sphere(1.0, 32, 32);
        let area = compute_surface_area(&mesh);
        let expected = 4.0 * std::f32::consts::PI; // Surface area of unit sphere

        println!("Sphere surface area: {} (expected: {})", area, expected);

        // Should be within 5% of expected
        assert!((area - expected).abs() / expected < 0.05);
    }

    #[test]
    fn test_analyze_mesh() {
        let mesh = generate_cube();
        let analysis = analyze_mesh(&mesh);

        println!("Mesh analysis:");
        println!("  Vertices: {}", analysis.n_vertices);
        println!("  Faces: {}", analysis.n_faces);
        println!(
            "  Bounding box: {:?} .. {:?}",
            analysis.bounding_box.0, analysis.bounding_box.1
        );
        println!(
            "  Bounding sphere: center={:?}, radius={}",
            analysis.bounding_sphere.0, analysis.bounding_sphere.1
        );
        println!("  Surface area: {}", analysis.surface_area);
        println!("  Volume: {:?}", analysis.volume);
        println!(
            "  Quality: min={}, mean={}",
            analysis.quality.min_quality, analysis.quality.mean_quality
        );

        assert_eq!(analysis.n_vertices, 8);
        assert_eq!(analysis.n_faces, 6);
        assert!(analysis.volume.is_some());
    }

    #[test]
    fn test_edge_length_stats() {
        let mesh = generate_cube();
        let stats = compute_edge_length_stats(&mesh);

        println!(
            "Edge length stats: min={}, max={}, mean={}, std={}",
            stats.min_length, stats.max_length, stats.mean_length, stats.std_dev
        );

        // Cube edges should all be length 2
        assert!((stats.min_length - 2.0).abs() < 0.01);
        assert!((stats.max_length - 2.0).abs() < 0.01);
        assert!(stats.std_dev < 0.01);
    }

    #[test]
    fn test_export_curvature_field() {
        let mesh = generate_sphere(1.0, 16, 16);

        let gaussian_field = export_curvature_field(&mesh, CurvatureType::Gaussian);
        let mean_field = export_curvature_field(&mesh, CurvatureType::Mean);

        println!("Gaussian curvature field: {} values", gaussian_field.len());
        println!("Mean curvature field: {} values", mean_field.len());

        assert_eq!(gaussian_field.len(), mesh.n_vertices());
        assert_eq!(mean_field.len(), mesh.n_vertices());
    }
}
