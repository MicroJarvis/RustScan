use rustmesh::{decimate_mesh, read_obj};
use std::time::Instant;

fn main() {
    let path = "../artifacts/inputs/large/FinalBaseMesh.obj";

    println!("Loading mesh...");
    let mut mesh = read_obj(path).expect("Failed to load OBJ");
    let n_verts = mesh.n_vertices();
    let n_faces = mesh.n_faces();

    println!("Original: V={}, F={}", n_verts, n_faces);

    // Test decimation to 50%
    println!("\n[Decimation to 50%]");
    let target = n_verts / 2;
    let start = Instant::now();
    let collapsed = decimate_mesh(&mut mesh, target, 10.0);
    let time = start.elapsed();

    // Count valid vertices
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
    println!(
        "Vertices: {} -> {} (target: {})",
        n_verts, valid_verts, target
    );
    println!("Faces: {} -> {}", n_faces, valid_faces);
}
