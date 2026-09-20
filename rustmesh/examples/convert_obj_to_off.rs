use rustmesh::read_obj;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: convert_obj_to_off <input.obj> <output.off>");
        return;
    }

    let input = &args[1];
    let output = &args[2];

    println!("Loading {}...", input);
    let mesh = read_obj(input).expect("Failed to load OBJ");

    println!("Vertices: {}, Faces: {}", mesh.n_vertices(), mesh.n_faces());

    // Write as OFF
    use rustmesh::write_off;
    write_off(&mesh, output).expect("Failed to write OFF");
    println!("Saved to {}", output);
}
