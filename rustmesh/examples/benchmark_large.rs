use rustmesh::read_obj;
use std::time::Instant;

fn main() {
    let path = "../artifacts/inputs/large/FinalBaseMesh.obj";

    println!("=== Loading large OBJ file with RustMesh ===");
    println!("File: {}", path);

    let start = Instant::now();
    let mesh = read_obj(path).expect("Failed to load OBJ");
    let load_time = start.elapsed();

    println!("\n=== RustMesh Statistics ===");
    println!("Vertices: {}", mesh.n_vertices());
    println!("Edges: {}", mesh.n_edges());
    println!("Faces: {}", mesh.n_faces());
    println!("Halfedges: {}", mesh.n_halfedges());
    println!("Load time: {:?}", load_time);

    println!("\n=== Testing validate() ===");
    match mesh.validate() {
        Ok(()) => println!("✅ Mesh is valid"),
        Err(e) => println!("❌ {}", e),
    }

    println!("\n=== Done ===");
}
