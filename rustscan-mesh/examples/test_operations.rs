use rustscan_mesh::{decimate_mesh, read_obj};
use std::time::Instant;

fn main() {
    let path = "../artifacts/inputs/large/FinalBaseMesh.obj";

    println!("Loading mesh...");
    let mut mesh = read_obj(path).expect("Failed to load OBJ");
    let n_verts = mesh.n_vertices();
    let n_faces = mesh.n_faces();

    println!("Original: V={}, F={}", n_verts, n_faces);

    // Test decimation
    println!("\n[Decimation Test]");
    let target = n_verts / 2;
    let start = Instant::now();
    let collapsed = decimate_mesh(&mut mesh, target, 0.5);
    let decimate_time = start.elapsed();

    println!("Collapsed: {}", collapsed);
    println!("Time: {:?}", decimate_time);
    println!("After: V={}, F={}", mesh.n_vertices(), mesh.n_faces());
}
