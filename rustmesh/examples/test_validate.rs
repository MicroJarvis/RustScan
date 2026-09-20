fn main() {
    use rustmesh::{generate_cube, generate_sphere, generate_tetrahedron};

    // Test tetrahedron
    let mesh = generate_tetrahedron();
    println!("\n=== Tetrahedron ===");
    match mesh.validate() {
        Ok(()) => println!("✅ Valid!"),
        Err(e) => println!("❌ {}", e),
    }

    // Test cube
    let mesh = generate_cube();
    println!("\n=== Cube ===");
    match mesh.validate() {
        Ok(()) => println!("✅ Valid!"),
        Err(e) => println!("❌ {}", e),
    }

    // Test sphere
    let mesh = generate_sphere(1.0, 8, 8);
    println!("\n=== Sphere (8x8) ===");
    match mesh.validate() {
        Ok(()) => println!("✅ Valid!"),
        Err(e) => println!("❌ {}", e),
    }
}
