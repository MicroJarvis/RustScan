use rustmesh::{decimate_mesh, read_obj, write_obj};
use std::time::Instant;

fn main() {
    let path = "../artifacts/inputs/middle/cube.obj";

    println!("=== RustMesh Test (without validate) ===");

    // Load
    let start = Instant::now();
    let mut mesh = read_obj(path).expect("Failed to load");
    let load_time = start.elapsed();

    println!(
        "Load: {:?} - V={}, F={}",
        load_time,
        mesh.n_vertices(),
        mesh.n_faces()
    );

    // Decimation
    println!("\n[Decimation 50%]");
    let target = mesh.n_vertices() / 2;
    let start = Instant::now();
    let collapsed = decimate_mesh(&mut mesh, target, 1000.0);
    let decimate_time = start.elapsed();

    // Count valid
    let valid_verts = mesh
        .vertices()
        .filter(|vh| mesh.halfedge_handle(*vh).is_some())
        .count();
    let valid_faces = mesh
        .faces()
        .filter(|fh| mesh.face_halfedge_handle(*fh).is_some())
        .count();

    println!("Decimate: {:?} - collapsed={}", decimate_time, collapsed);
    println!("  V: {} -> {}", mesh.n_vertices(), valid_verts);
    println!("  F: {} -> {}", mesh.n_faces(), valid_faces);

    // Write result
    write_obj(&mesh, "/tmp/cube_decimated.obj").expect("Failed to write");
    println!("\nSaved to /tmp/cube_decimated.obj");
}
