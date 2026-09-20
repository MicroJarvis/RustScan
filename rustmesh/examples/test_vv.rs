fn main() {
    let mesh = rustmesh::generate_cube();
    let v0 = rustmesh::VertexHandle::new(0);

    println!("Testing vertex_vertices for v0");
    let mut count = 0;
    if let Some(iter) = mesh.vertex_vertices(v0) {
        for _ in iter {
            count += 1;
            if count > 10 {
                break;
            }
        }
    }
    println!("Found {} neighbors", count);
}
