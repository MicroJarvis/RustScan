use rustscan_mesh::{read_obj, HalfedgeHandle, RustMesh};
use std::time::Instant;

fn main() {
    let path = "../artifacts/inputs/large/FinalBaseMesh.obj";

    println!("Loading mesh...");
    let mut mesh = read_obj(path).expect("Failed to load OBJ");
    let n_verts = mesh.n_vertices();
    let n_faces = mesh.n_faces();

    println!("Original: V={}, F={}", n_verts, n_faces);

    // Simple decimation without quadrics - just collapse edges sequentially
    println!("\n[Simple Decimation - Sequential]");
    let target = n_verts / 2;
    let collapse_count = n_verts - target;

    let start = Instant::now();
    let mut collapsed = 0;

    // Try to collapse edges in order
    'outer: for heh_idx in 0..mesh.n_halfedges() {
        if collapsed >= collapse_count {
            break;
        }

        let heh = HalfedgeHandle::new(heh_idx as u32);
        if mesh.is_collapse_ok(heh) {
            if mesh.collapse(heh).is_ok() {
                collapsed += 1;
            }
        }
    }

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
    println!("Vertices: {} -> {}", n_verts, valid_verts);
    println!("Faces: {} -> {}", n_faces, valid_faces);
}
