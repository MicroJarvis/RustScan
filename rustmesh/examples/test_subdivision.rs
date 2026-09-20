use rustmesh::{generate_cube, loop_subdivide};

fn main() {
    println!("Creating cube...");
    let mut mesh = generate_cube();

    println!("Original: V={}, F={}", mesh.n_vertices(), mesh.n_faces());

    println!("Running loop_subdivide...");
    let result = loop_subdivide(&mut mesh);

    match result {
        Ok(stats) => {
            println!("Success!");
            println!("After: V={}, F={}", mesh.n_vertices(), mesh.n_faces());
        }
        Err(e) => {
            println!("Error: {:?}", e);
        }
    }
}
