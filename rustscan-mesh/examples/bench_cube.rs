use rustscan_mesh::{decimate_mesh, read_obj};
use std::time::Instant;

fn main() {
    let path = "../artifacts/inputs/middle/cube.obj";

    println!("=== RustMesh Loading ===");
    let start = Instant::now();
    let mut mesh = read_obj(path).expect("Failed to load OBJ");
    let load_time = start.elapsed();

    let n_verts = mesh.n_vertices();
    let n_faces = mesh.n_faces();

    println!("Load time: {:?}", load_time);
    println!("Original: V={}, F={}", n_verts, n_faces);

    // Validate
    println!("\n[Validate]");
    match mesh.validate() {
        Ok(()) => println!("✅ Valid"),
        Err(e) => println!("❌ {}", e),
    }

    // Decimation to 50%
    println!("\n[Decimation to 50%]");
    let target = n_verts / 2;
    let start = Instant::now();
    let collapsed = decimate_mesh(&mut mesh, target, 1000.0);
    let time = start.elapsed();

    let valid_verts = mesh
        .vertices()
        .filter(|vh| mesh.halfedge_handle(*vh).is_some())
        .count();
    let valid_faces = mesh
        .faces()
        .filter(|fh| mesh.face_halfedge_handle(*fh).is_some())
        .count();

    println!("Collapsed: {} edges", collapsed);
    println!("Time: {:?}", time);
    println!("Vertices: {} -> {}", n_verts, valid_verts);
    println!("Faces: {} -> {}", n_faces, valid_faces);
}
