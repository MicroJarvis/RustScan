fn main() {
    let mesh = rustscan_mesh::generate_cube();
    let v0 = rustscan_mesh::VertexHandle::new(0);

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
