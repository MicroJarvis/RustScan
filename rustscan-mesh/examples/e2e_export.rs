//! Example: 3DGS → Mesh → Export
//!
//! Demonstrates the full end-to-end pipeline:
//! 1. Extract mesh from 3D Gaussian Splatting (via TSDF + Marching Cubes)
//! 2. Convert to RustMesh
//! 3. Export to OBJ/PLY files

use rustscan_mesh::{Point3, RustMesh, Vec3};
use std::path::Path;

// This example shows how to use RustSLAM's mesh extraction with RustMesh export
fn main() -> std::io::Result<()> {
    println!("=== RustScan E2E Pipeline Example ===\n");

    // NOTE: This is a conceptual example showing the API
    // In a real application, you would:
    // 1. Run SLAM to get camera poses
    // 2. Train 3DGS from those poses
    // 3. Render depth maps from Gaussians
    // 4. Fuse into TSDF volume
    // 5. Extract mesh with Marching Cubes
    // 6. Convert to RustMesh
    // 7. Export

    // For demonstration, create a simple test mesh
    println!("Creating test mesh...");
    let vertices = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.5, 1.0, 0.0),
        Point3::new(0.5, 0.5, 1.0),
    ];

    let triangles = vec![
        [0, 1, 2], // Base triangle
        [0, 1, 3], // Side 1
        [1, 2, 3], // Side 2
        [2, 0, 3], // Side 3
    ];

    let normals = vec![
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(-1.0, -1.0, -1.0).normalize(),
    ];

    let colors = vec![
        [1.0, 0.0, 0.0], // Red
        [0.0, 1.0, 0.0], // Green
        [0.0, 0.0, 1.0], // Blue
        [1.0, 1.0, 0.0], // Yellow
    ];

    // Convert to RustMesh
    println!("Converting to RustMesh...");
    let mesh = rustscan_mesh::RustMesh::from_triangle_mesh(
        &vertices,
        &triangles,
        Some(&normals),
        Some(&colors),
    );

    println!("Mesh stats:");
    println!("  Vertices: {}", mesh.n_vertices());
    println!("  Faces: {}", mesh.n_faces());
    println!("  Has normals: {}", mesh.has_vertex_normals());
    println!("  Has colors: {}", mesh.has_vertex_colors());
    println!();

    // Export to OBJ
    let obj_path = "/tmp/e2e_output.obj";
    println!("Exporting to OBJ: {}", obj_path);
    rustscan_mesh::io::write_obj(&mesh, obj_path)?;
    println!("  ✓ OBJ export complete");

    // Export to PLY (ASCII)
    let ply_ascii_path = "/tmp/e2e_output_ascii.ply";
    println!("Exporting to PLY (ASCII): {}", ply_ascii_path);
    rustscan_mesh::io::write_ply(&mesh, ply_ascii_path, rustscan_mesh::io::PlyFormat::Ascii)?;
    println!("  ✓ PLY ASCII export complete");

    // Export to PLY (Binary)
    let ply_binary_path = "/tmp/e2e_output_binary.ply";
    println!("Exporting to PLY (Binary): {}", ply_binary_path);
    rustscan_mesh::io::write_ply(
        &mesh,
        ply_binary_path,
        rustscan_mesh::io::PlyFormat::BinaryLittleEndian,
    )?;
    println!("  ✓ PLY Binary export complete");

    println!("\n=== Export Complete ===");
    println!("Files created:");
    println!("  - {}", obj_path);
    println!("  - {}", ply_ascii_path);
    println!("  - {}", ply_binary_path);

    // Verify by reading back
    println!("\nVerifying OBJ roundtrip...");
    let loaded = rustscan_mesh::io::read_obj(obj_path)?;
    println!("  Loaded mesh:");
    println!("    Vertices: {}", loaded.n_vertices());
    println!("    Faces: {}", loaded.n_faces());

    if loaded.n_vertices() == mesh.n_vertices() && loaded.n_faces() == mesh.n_faces() {
        println!("  ✓ Roundtrip verification passed!");
    } else {
        println!("  ✗ Roundtrip verification failed!");
    }

    println!("\n=== Integration Example for RustSLAM ===");
    println!("To integrate with RustSLAM 3DGS mesh extraction:");
    println!(
        "
    // In RustSLAM:
    use rustscan_slam::fusion::{{MeshExtractor, MeshExtractionConfig}};

    // 1. Extract mesh from Gaussians
    let mut extractor = MeshExtractor::centered(Vec3::zeros(), 2.0, 0.01);
    extractor.integrate_from_gaussians(|idx| depth[idx], ...);
    let slam_mesh = extractor.extract_with_postprocessing();

    // 2. Convert to RustMesh format
    let vertices: Vec<Vec3> = slam_mesh.vertices.iter()
        .map(|v| v.position)
        .collect();
    let triangles: Vec<[usize; 3]> = slam_mesh.triangles.iter()
        .map(|t| t.indices)
        .collect();
    let normals: Vec<Vec3> = slam_mesh.vertices.iter()
        .map(|v| v.normal)
        .collect();
    let colors: Vec<[f32; 3]> = slam_mesh.vertices.iter()
        .map(|v| v.color)
        .collect();

    let mesh = RustMesh::from_triangle_mesh(
        &vertices,
        &triangles,
        Some(&normals),
        Some(&colors),
    );

    // 3. Export
    rustscan_mesh::io::write_obj(&mesh, \"output.obj\")?;
    rustscan_mesh::io::write_ply(&mesh, \"output.ply\", PlyFormat::Ascii)?;
    "
    );

    Ok(())
}
