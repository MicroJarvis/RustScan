use rustmesh::{decimate_mesh, read_obj, HalfedgeHandle, RustMesh};
use std::time::Instant;

fn main() {
    let path = "../artifacts/inputs/large/FinalBaseMesh.obj";

    println!("Loading mesh...");
    let mut mesh = read_obj(path).expect("Failed to load OBJ");

    // Check how many edges are collapsible
    let mut collapsible = 0;
    for heh_idx in 0..mesh.n_halfedges() {
        let heh = HalfedgeHandle::new(heh_idx as u32);
        if mesh.is_collapse_ok(heh) {
            collapsible += 1;
        }
    }
    println!("Total halfedges: {}", mesh.n_halfedges());
    println!("Collapsible edges: {}", collapsible);

    // Try decimation with higher limit
    println!("\n[Decimation - no limit]");
    let start = Instant::now();
    let collapsed = decimate_mesh(&mut mesh, 1000, 10.0); // Allow up to 1000 collapses, high error
    let decimate_time = start.elapsed();

    // Count valid faces
    let mut valid_faces = 0;
    for fh in mesh.faces() {
        if mesh.face_halfedge_handle(fh).is_some() {
            valid_faces += 1;
        }
    }

    println!("Collapsed: {} edges", collapsed);
    println!("Time: {:?}", decimate_time);
    println!("Valid faces: {} (was 48918)", valid_faces);
}
