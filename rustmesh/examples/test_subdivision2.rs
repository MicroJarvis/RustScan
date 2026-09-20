use rustmesh::generate_cube;

fn main() {
    let mesh = generate_cube();
    println!("Cube V={}, F={}", mesh.n_vertices(), mesh.n_faces());
}
