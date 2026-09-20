use rustscan_mesh::{decimate_mesh, read_obj, HalfedgeHandle, Point3, QuadricT, RustMesh};

fn main() {
    let path = "../artifacts/inputs/large/FinalBaseMesh.obj";

    println!("Loading mesh...");
    let mesh = read_obj(path).expect("Failed to load OBJ");

    // Test quadric on a few edges
    println!("\n[Testing Quadric Collapse Priority]");
    let mut quadrics: Vec<Option<QuadricT>> = vec![None; mesh.n_vertices()];

    // Initialize quadrics
    for q in &mut quadrics {
        *q = Some(QuadricT::zero());
    }

    // Compute face quadrics
    for fh in mesh.faces() {
        let heh = match mesh.face_halfedge_handle(fh) {
            Some(h) => h,
            None => continue,
        };

        let mut verts = Vec::new();
        let mut current = heh;
        loop {
            let vh = mesh.to_vertex_handle(current);
            verts.push(vh);
            current = mesh.next_halfedge_handle(current);
            if current == heh || verts.len() > 10 {
                break;
            }
        }

        if verts.len() >= 3 {
            let p0 = mesh.point(verts[0]).unwrap();
            let p1 = mesh.point(verts[1]).unwrap();
            let p2 = mesh.point(verts[2]).unwrap();

            let edge1 = p1 - p0;
            let edge2 = p2 - p0;
            let normal = edge1.cross(&edge2).normalize();
            let center = Point3::from((p0.coords + p1.coords + p2.coords) / 3.0);

            let q = QuadricT::from_face(normal, center);

            for vh in &verts {
                let idx = vh.idx_usize();
                if let Some(ref mut vq) = quadrics[idx] {
                    vq.add_assign_values(q);
                }
            }
        }
    }

    // Check a few edges
    println!("Testing first 10 edges:");
    for heh_idx in 0..10 {
        let heh = HalfedgeHandle::new(heh_idx as u32);
        let to_vh = mesh.to_vertex_handle(heh);
        let from_vh = mesh.from_vertex_handle(heh);

        let idx0 = from_vh.idx_usize();
        let idx1 = to_vh.idx_usize();

        let q0 = match quadrics.get(idx0) {
            Some(Some(q)) => q,
            _ => continue,
        };
        let q1 = match quadrics.get(idx1) {
            Some(Some(q)) => q,
            _ => continue,
        };

        let combined = q0.add_values(*q1);
        let (opt, error) = combined.optimize();

        println!(
            "  Edge {}: v{}->v{}: error={:.4}, opt={:?}",
            heh_idx,
            from_vh.idx(),
            to_vh.idx(),
            error,
            opt
        );
    }
}
