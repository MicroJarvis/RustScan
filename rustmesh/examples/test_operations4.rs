use rustmesh::{read_obj, HalfedgeHandle, RustMesh};
use std::time::Instant;

fn main() {
    let path = "../artifacts/inputs/large/FinalBaseMesh.obj";

    println!("Loading mesh...");
    let mut mesh = read_obj(path).expect("Failed to load OBJ");

    // Manually try to collapse a few edges
    println!("\n[Manual Collapse Test]");
    let mut collapsed = 0;

    for heh_idx in 0..1000 {
        let heh = HalfedgeHandle::new(heh_idx as u32);

        if mesh.is_collapse_ok(heh) {
            match mesh.collapse(heh) {
                Ok(()) => {
                    collapsed += 1;
                    println!(
                        "Collapsed edge {} - V:{}, F:{}",
                        heh_idx,
                        mesh.n_vertices(),
                        mesh.n_faces()
                    );
                }
                Err(e) => {
                    // println!("Failed to collapse {}: {}", heh_idx, e);
                }
            }
        }

        if collapsed >= 10 {
            break;
        }
    }

    println!("\nTotal collapsed: {}", collapsed);
}
