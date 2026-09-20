//! Integration tests for Marching Cubes mesh extraction
//!
//! Tests complete workflows from TSDF volume to mesh extraction.

use nalgebra::Point3;
use rustscan_slam::fusion::marching_cubes::extract_mesh_from_tsdf;
use rustscan_slam::fusion::tsdf_volume::{TsdfConfig, TsdfVolume};
use rustscan_slam::test_utils::*;

fn test_sphere_mesh_extraction() {
    // Create sphere TSDF
    let center = Point3::origin();
    let radius = 1.0;
    let voxel_size = 0.05;

    let tsdf = create_sphere_tsdf(center, radius, voxel_size);

    // Extract mesh
    let mesh = extract_mesh_from_tsdf(&tsdf);

    // Verify mesh is approximately spherical
    assert!(
        mesh.num_vertices() > 100,
        "Sphere should have many vertices, got {}",
        mesh.num_vertices()
    );
    assert!(
        mesh.num_triangles() > 100,
        "Sphere should have many triangles, got {}",
        mesh.num_triangles()
    );

    // Verify all vertices are roughly on sphere surface
    let tolerance = 0.15; // Allow tolerance due to voxelization
    for vertex in &mesh.vertices {
        let dist_from_center = vertex.position.coords.norm();
        assert!(
            (dist_from_center - radius).abs() < tolerance,
            "Vertex at {:?} is {} from center (expected ~{})",
            vertex.position,
            dist_from_center,
            radius
        );
    }

    // Verify normals point outward
    for vertex in &mesh.vertices {
        let expected_normal = vertex.position.coords.normalize();
        let dot = vertex.normal.dot(&expected_normal);
        assert!(
            dot > 0.5,
            "Normal {:?} should point outward from center (dot product: {})",
            vertex.normal,
            dot
        );
    }
}

fn test_cube_mesh_extraction() {
    // Create cube TSDF
    let center = Point3::origin();
    let size = 1.0;
    let voxel_size = 0.05;

    let tsdf = create_cube_tsdf(center, size, voxel_size);

    // Extract mesh
    let mesh = extract_mesh_from_tsdf(&tsdf);

    // Verify mesh has vertices and triangles
    assert!(
        mesh.num_vertices() > 8,
        "Cube should have more than 8 vertices (corners), got {}",
        mesh.num_vertices()
    );
    assert!(
        mesh.num_triangles() > 12,
        "Cube should have more than 12 triangles (2 per face), got {}",
        mesh.num_triangles()
    );

    // Verify vertices are within cube bounds
    let half_size = size / 2.0;
    let tolerance = 0.15;
    for vertex in &mesh.vertices {
        let pos = vertex.position;
        assert!(
            pos.x.abs() <= half_size + tolerance,
            "Vertex X {} exceeds cube bounds",
            pos.x
        );
        assert!(
            pos.y.abs() <= half_size + tolerance,
            "Vertex Y {} exceeds cube bounds",
            pos.y
        );
        assert!(
            pos.z.abs() <= half_size + tolerance,
            "Vertex Z {} exceeds cube bounds",
            pos.z
        );
    }
}

fn test_mesh_properties() {
    // Create a simple sphere
    let tsdf = create_sphere_tsdf(Point3::origin(), 0.5, 0.05);
    let mesh = extract_mesh_from_tsdf(&tsdf);

    // Verify mesh is not empty
    assert!(mesh.num_vertices() > 0, "Mesh should have vertices");
    assert!(mesh.num_triangles() > 0, "Mesh should have triangles");

    // Verify all triangle indices are valid
    for (i, triangle) in mesh.triangles.iter().enumerate() {
        for &idx in &triangle.indices {
            assert!(
                idx < mesh.num_vertices(),
                "Triangle {} has invalid vertex index {} (max: {})",
                i,
                idx,
                mesh.num_vertices() - 1
            );
        }
    }

    // Verify all vertices have valid normals (unit length)
    for (i, vertex) in mesh.vertices.iter().enumerate() {
        let normal_length = vertex.normal.norm();
        assert!(
            (normal_length - 1.0).abs() < 0.1,
            "Vertex {} has non-unit normal (length: {})",
            i,
            normal_length
        );
    }

    // Verify all vertices have valid colors (0-1 range)
    for (i, vertex) in mesh.vertices.iter().enumerate() {
        for (j, &color) in vertex.color.iter().enumerate() {
            assert!(
                color >= 0.0 && color <= 1.0,
                "Vertex {} color[{}] = {} is out of range [0, 1]",
                i,
                j,
                color
            );
        }
    }
}

fn main() {
    println!("Running Marching Cubes integration tests...");

    test_sphere_mesh_extraction();
    println!("✓ Sphere mesh extraction test passed");

    test_cube_mesh_extraction();
    println!("✓ Cube mesh extraction test passed");

    test_mesh_properties();
    println!("✓ Mesh properties test passed");

    println!("\nAll Marching Cubes integration tests passed!");
}
