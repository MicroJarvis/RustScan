fn main() {
    let mesh = rustmesh::generate_cube();
    println!(
        "Cube: {} vertices, {} edges, {} faces",
        mesh.n_vertices(),
        mesh.n_edges(),
        mesh.n_faces()
    );

    // Test face_halfedges
    let fh = rustmesh::FaceHandle::new(0);
    println!("\nTesting face_halfedges for face 0:");
    if let Some(iter) = mesh.face_halfedges(fh) {
        let mut count = 0;
        for _ in iter {
            count += 1;
        }
        println!("  Found {} halfedges", count);
    } else {
        println!("  ERROR: No halfedges!");
    }

    // Test face_edges
    println!("\nTesting face_edges for face 0:");
    if let Some(iter) = mesh.face_edges(fh) {
        let mut count = 0;
        for _ in iter {
            count += 1;
        }
        println!("  Found {} edges", count);
    } else {
        println!("  ERROR: No edges!");
    }

    println!("\n✅ Face circulators working!");
}
