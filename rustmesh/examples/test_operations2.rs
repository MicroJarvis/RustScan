use rustmesh::{decimate_mesh, read_obj};
use std::time::Instant;

fn main() {
    let path = "../artifacts/inputs/large/FinalBaseMesh.obj";

    println!("Loading mesh...");
    let mut mesh = read_obj(path).expect("Failed to load OBJ");
    let n_verts = mesh.n_vertices();
    let n_faces = mesh.n_faces();

    println!("Original: V={}, F={}", n_verts, n_faces);

    // Test decimation - target 50%
    println!("\n[Decimation Test - target 50%]");
    let target = n_verts / 2;
    let start = Instant::now();
    let collapsed = decimate_mesh(&mut mesh, target, 0.5);
    let decimate_time = start.elapsed();

    // Count valid faces (non-None halfedge_handle)
    let mut valid_faces = 0;
    for fh in mesh.faces() {
        if mesh.face_halfedge_handle(fh).is_some() {
            valid_faces += 1;
        }
    }

    // Count valid vertices (have valid halfedge)
    let mut valid_verts = 0;
    for vh in mesh.vertices() {
        if mesh.halfedge_handle(vh).is_some() {
            valid_verts += 1;
        }
    }

    println!("Collapsed: {} edges", collapsed);
    println!("Time: {:?}", decimate_time);
    println!(
        "Valid vertices: {} (total: {})",
        valid_verts,
        mesh.n_vertices()
    );
    println!("Valid faces: {} (total: {})", valid_faces, mesh.n_faces());
    println!("Expected vertices: {}", n_verts - collapsed);
}
