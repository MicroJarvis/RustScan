use rustmesh::{read_obj, HalfedgeHandle};

fn main() {
    let mesh = read_obj("../artifacts/inputs/large/FinalBaseMesh.obj").unwrap();

    println!("=== Boundary Test (RustMesh) ===");
    let mut boundary = 0;
    let mut sample = 0;
    for heh_idx in 0..mesh.n_halfedges() {
        if sample >= 100 {
            break;
        }
        let heh = HalfedgeHandle::new(heh_idx as u32);
        if mesh.is_boundary(heh) {
            boundary += 1;
        }
        sample += 1;
    }
    println!("Boundary halfedges (sample 100): {}", boundary);
}
