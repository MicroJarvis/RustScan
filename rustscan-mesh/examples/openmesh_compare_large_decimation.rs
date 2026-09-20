//! Large-scale decimation benchmark using subdivided sphere
//! Uses Loop subdivision for proper triangulation

mod openmesh_compare_common;

use openmesh_compare_common::print_header;
use rustscan_mesh::{loop_subdivide, write_off, Decimater, Point3, RustMesh, Vec3};
use std::time::Instant;

/// Create a subdivided sphere for large-scale testing using Loop subdivision
fn create_subdivided_sphere(iterations: usize) -> RustMesh {
    let mut mesh = create_octahedron();

    // Apply Loop subdivision multiple times (better than sqrt3 for triangulation)
    for _ in 0..iterations {
        loop_subdivide(&mut mesh).expect("subdivision should succeed");
    }

    mesh
}

/// Create base octahedron (V=6, F=8)
fn create_octahedron() -> RustMesh {
    let mut mesh = RustMesh::new();

    // Add 6 vertices
    let v0 = mesh.add_vertex(Point3::new(0.0, 1.0, 0.0)); // top
    let v1 = mesh.add_vertex(Point3::new(0.0, -1.0, 0.0)); // bottom
    let v2 = mesh.add_vertex(Point3::new(1.0, 0.0, 0.0)); // right
    let v3 = mesh.add_vertex(Point3::new(-1.0, 0.0, 0.0)); // left
    let v4 = mesh.add_vertex(Point3::new(0.0, 0.0, 1.0)); // front
    let v5 = mesh.add_vertex(Point3::new(0.0, 0.0, -1.0)); // back

    // Add 8 triangular faces (top half)
    mesh.add_face(&[v0, v4, v2]);
    mesh.add_face(&[v0, v2, v5]);
    mesh.add_face(&[v0, v5, v3]);
    mesh.add_face(&[v0, v3, v4]);
    // Bottom half
    mesh.add_face(&[v1, v2, v4]);
    mesh.add_face(&[v1, v5, v2]);
    mesh.add_face(&[v1, v3, v5]);
    mesh.add_face(&[v1, v4, v3]);

    mesh
}

fn raw_face_diagnostics(mesh: &rustscan_mesh::RustMesh) -> (usize, usize, usize) {
    let mut active_faces = 0usize;
    let mut degenerate_faces = 0usize;
    let mut edge_use: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();

    for fh in mesh.faces() {
        let vertices = mesh.face_vertices_vec(fh);
        if vertices.len() < 3 {
            continue;
        }
        active_faces += 1;
        if vertices.len() != 3 {
            degenerate_faces += 1;
            continue;
        }

        let ids = [
            vertices[0].idx_usize(),
            vertices[1].idx_usize(),
            vertices[2].idx_usize(),
        ];
        if ids[0] == ids[1] || ids[1] == ids[2] || ids[2] == ids[0] {
            degenerate_faces += 1;
        }

        for (a, b) in [(ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])] {
            let key = if a < b { (a, b) } else { (b, a) };
            *edge_use.entry(key).or_insert(0) += 1;
        }
    }

    let non_manifold_edges = edge_use.values().filter(|&&count| count > 2).count();
    (active_faces, degenerate_faces, non_manifold_edges)
}

fn main() {
    print_header("Large-Scale Decimation Benchmark (Subdivided Sphere)");

    // Generate subdivided sphere: 4 iterations of Loop subdivision
    let iterations = 4;
    println!("Generating sphere with {} Loop subdivisions...", iterations);

    let start_gen = Instant::now();
    let mesh = create_subdivided_sphere(iterations);
    let gen_time = start_gen.elapsed();

    let initial_v = mesh.n_vertices();
    let initial_f = mesh.n_faces();
    println!(
        "Mesh generation time: {:.3} ms",
        gen_time.as_secs_f64() * 1000.0
    );
    println!("Initial mesh: V={}, F={}", initial_v, initial_f);

    // Target: decimate to 50% of original vertices
    let target_vertices = initial_v / 2;
    println!("Target vertices: {} (50% reduction)", target_vertices);

    // ============================================================
    // RustMesh Decimation
    // ============================================================
    let mut mesh_rm = mesh.clone();
    let start_rm = Instant::now();
    let mut decimater = Decimater::new(&mut mesh_rm);
    let collapsed_rm = decimater.decimate_to(target_vertices);
    let boundary_rm = decimater.boundary_collapses();
    let interior_rm = decimater.interior_collapses();
    drop(decimater);

    let (raw_faces, raw_degenerate, raw_non_manifold) = raw_face_diagnostics(&mesh_rm);
    mesh_rm.garbage_collection();

    let elapsed_rm = start_rm.elapsed();
    let final_v_rm = mesh_rm.n_vertices();
    let final_f_rm = mesh_rm.n_faces();
    let final_e_rm = mesh_rm.n_edges();

    println!("\n============================================================");
    println!("RustMesh Results");
    println!("============================================================");
    println!(
        "Decimation time: {:.3} ms",
        elapsed_rm.as_secs_f64() * 1000.0
    );
    println!("Collapsed: {}", collapsed_rm);
    println!("Boundary collapses: {}", boundary_rm);
    println!("Interior collapses: {}", interior_rm);
    println!("Final V: {}", final_v_rm);
    println!("Final E: {}", final_e_rm);
    println!("Final F: {}", final_f_rm);
    println!(
        "Raw diagnostics: active_faces={}, degenerate={}, non_manifold_edges={}",
        raw_faces, raw_degenerate, raw_non_manifold
    );

    // Write RustMesh output
    let rust_output = std::env::temp_dir().join("rustmesh-large-decimated.off");
    write_off(&mesh_rm, &rust_output).expect("write RustMesh output");
    println!("Output: {}", rust_output.display());

    // ============================================================
    // Validation
    // ============================================================
    println!("\n============================================================");
    println!("Validation");
    println!("============================================================");

    let v_close = final_v_rm <= target_vertices + 5 && final_v_rm >= target_vertices - 5;
    let topology_ok = raw_degenerate == 0 && raw_non_manifold == 0;

    if v_close && topology_ok {
        println!("✅ Decimation successful!");
        println!("   - Vertex count within target range");
        println!("   - No degenerate faces");
        println!("   - No non-manifold edges");
    } else {
        if !v_close {
            println!(
                "❌ Vertex count out of range: {} (target: {})",
                final_v_rm, target_vertices
            );
        }
        if !topology_ok {
            println!(
                "❌ Topology issues: degenerate={}, non_manifold={}",
                raw_degenerate, raw_non_manifold
            );
        }
    }

    // Performance summary
    println!("\n============================================================");
    println!("Performance Summary");
    println!("============================================================");
    let collapses_per_ms = collapsed_rm as f64 / elapsed_rm.as_secs_f64() * 1000.0;
    println!("Collapses/ms: {:.1}", collapses_per_ms);
    println!(
        "Vertices/ms: {:.1}",
        (initial_v - final_v_rm) as f64 / elapsed_rm.as_secs_f64() * 1000.0
    );

    // Memory estimate
    println!(
        "\nMesh size: {:.2} MB vertices, {:.2} MB faces",
        initial_v as f64 * 12.0 / 1e6, // 3 floats per vertex
        initial_f as f64 * 12.0 / 1e6  // 3 indices per face
    );
}
