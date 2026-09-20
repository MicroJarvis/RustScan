fn main() {
    let mesh = rustmesh::generate_cube();
    println!(
        "Cube: {} vertices, {} edges, {} faces",
        mesh.n_vertices(),
        mesh.n_edges(),
        mesh.n_faces()
    );

    // Test vertex_halfedges
    let v0 = rustmesh::VertexHandle::new(0);
    println!("\nTesting vertex_halfedges for v0:");
    if let Some(iter) = mesh.vertex_halfedges(v0) {
        let mut count = 0;
        for _ in iter {
            count += 1;
        }
        println!("  Found {} halfedges", count);
    } else {
        println!("  ERROR: No halfedges!");
    }

    // Test vertex_edges
    println!("\nTesting vertex_edges for v0:");
    if let Some(iter) = mesh.vertex_edges(v0) {
        let mut count = 0;
        for _ in iter {
            count += 1;
        }
        println!("  Found {} edges", count);
    } else {
        println!("  ERROR: No edges!");
    }

    println!("\n✅ Done!");
}
