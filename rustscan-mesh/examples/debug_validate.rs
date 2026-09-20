use rustscan_mesh::{read_obj, HalfedgeHandle, RustMesh, VertexHandle};

fn main() {
    let path = "../artifacts/inputs/middle/cube.obj";
    let mesh = read_obj(path).expect("Failed to load");

    println!("=== Debugging vertex 0 ===");
    let vh = VertexHandle::new(0);

    // Get vertex halfedge
    let start_heh = match mesh.halfedge_handle(vh) {
        Some(h) => h,
        None => {
            println!("Vertex 0 has no halfedge!");
            return;
        }
    };

    println!("Start halfedge: {:?}", start_heh);

    // Walk around the vertex
    let mut current = start_heh;
    let mut count = 0;
    loop {
        count += 1;
        if count > 20 {
            println!("Too many iterations!");
            break;
        }

        println!(
            "  Step {}: heh={:?}, to_vertex={:?}",
            count,
            current,
            mesh.to_vertex_handle(current)
        );

        // Go to opposite halfedge
        let opposite = mesh.opposite_halfedge_handle(current);
        println!("    opposite={:?}", opposite);

        // Go to next halfedge around the vertex
        let next = mesh.next_halfedge_handle(opposite);
        println!("    next={:?}", next);

        if next == current || !next.is_valid() {
            println!("  Breaking - next is invalid or same as current");
            break;
        }

        if next == start_heh {
            println!("  Completed loop!");
            break;
        }

        current = next;
    }

    println!("Total steps: {}", count);
}
