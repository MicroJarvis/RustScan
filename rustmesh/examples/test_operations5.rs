use rustmesh::{read_obj, HalfedgeHandle, RustMesh};

fn main() {
    let path = "../artifacts/inputs/large/FinalBaseMesh.obj";

    println!("Loading mesh...");
    let mut mesh = read_obj(path).expect("Failed to load OBJ");

    // Count valid items
    let orig_valid_verts = mesh
        .vertices()
        .filter(|vh| mesh.halfedge_handle(*vh).is_some())
        .count();
    let orig_valid_faces = mesh
        .faces()
        .filter(|fh| mesh.face_halfedge_handle(*fh).is_some())
        .count();

    println!(
        "Original: {} valid vertices, {} valid faces",
        orig_valid_verts, orig_valid_faces
    );

    // Manually collapse 100 edges
    println!("\n[Collapsing 100 edges]");
    let mut collapsed = 0;

    for heh_idx in 0..mesh.n_halfedges() {
        let heh = HalfedgeHandle::new(heh_idx as u32);

        if mesh.is_collapse_ok(heh) {
            if mesh.collapse(heh).is_ok() {
                collapsed += 1;
            }
        }

        if collapsed >= 100 {
            break;
        }
    }

    // Count valid items after collapse
    let new_valid_verts = mesh
        .vertices()
        .filter(|vh| mesh.halfedge_handle(*vh).is_some())
        .count();
    let new_valid_faces = mesh
        .faces()
        .filter(|fh| mesh.face_halfedge_handle(*fh).is_some())
        .count();

    println!("\nResults:");
    println!("  Collapsed: {} edges", collapsed);
    println!(
        "  Valid vertices: {} -> {} (expected -{})",
        orig_valid_verts, new_valid_verts, collapsed
    );
    println!(
        "  Valid faces: {} -> {} (expected -{}*2)",
        orig_valid_faces, new_valid_faces, collapsed
    );

    if new_valid_verts == orig_valid_verts - collapsed {
        println!("\n✅ Decimation is working correctly!");
    } else {
        println!("\n❌ Something is wrong");
    }
}
