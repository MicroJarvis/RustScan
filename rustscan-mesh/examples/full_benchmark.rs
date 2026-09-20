use rustscan_mesh::{read_obj, read_off, write_obj, write_off};
use std::time::Instant;

fn main() {
    let path = "../artifacts/inputs/large/FinalBaseMesh.obj";

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║          RustMesh Large File Benchmark                      ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    // Test 1: Load OBJ
    println!("\n[1] Load OBJ");
    let start = Instant::now();
    let mesh = read_obj(path).expect("Failed to load OBJ");
    let load_obj = start.elapsed();
    println!("    Time: {:?}", load_obj);
    println!(
        "    V:{} E:{} F:{}",
        mesh.n_vertices(),
        mesh.n_edges(),
        mesh.n_faces()
    );

    // Test 2: Validate
    println!("\n[2] Validate");
    match mesh.validate() {
        Ok(()) => println!("    ✅ Valid"),
        Err(e) => println!("    ❌ {}", e),
    }

    // Test 3: Write OBJ
    println!("\n[3] Write OBJ");
    let start = Instant::now();
    write_obj(&mesh, "/tmp/rustmesh_out.obj").expect("Failed to write OBJ");
    let write_obj_time = start.elapsed();
    println!("    Time: {:?}", write_obj_time);

    // Test 4: Write OFF
    println!("\n[4] Write OFF");
    let start = Instant::now();
    write_off(&mesh, "/tmp/rustmesh_out.off").expect("Failed to write OFF");
    let write_off_time = start.elapsed();
    println!("    Time: {:?}", write_off_time);

    // Test 5: Read OFF (round trip)
    println!("\n[5] Read OFF (round trip)");
    let start = Instant::now();
    let _mesh2 = read_off("/tmp/rustmesh_out.off").expect("Failed to read OFF");
    let read_off_time = start.elapsed();
    println!("    Time: {:?}", read_off_time);

    // Test 6: STL Read
    println!("\n[6] STL Read (if available)");
    // Skip if no STL file

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║                    Summary                                  ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!("Total vertices: {}", mesh.n_vertices());
    println!("Total edges: {}", mesh.n_edges());
    println!("Total faces: {}", mesh.n_faces());
}
