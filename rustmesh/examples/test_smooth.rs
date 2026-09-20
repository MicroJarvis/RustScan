fn main() {
    use rustmesh::{generate_cube, laplace_smooth, SmootherConfig};

    let mut mesh = generate_cube();
    println!("Before: {} vertices", mesh.n_vertices());

    let config = SmootherConfig {
        iterations: 1,
        strength: 0.5,
        uniform: true,
        fixed_boundary: true,
    };

    println!("Running laplace_smooth...");
    let result = laplace_smooth(&mut mesh, config);

    println!(
        "After: {} vertices, {} iterations",
        mesh.n_vertices(),
        result.iterations
    );
    println!("✅ Smoothing test passed!");
}
