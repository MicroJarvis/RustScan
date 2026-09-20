// RustMesh Performance Benchmark
use rustmesh::{generate_sphere, RustMesh};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    println!("=== RustMesh Performance Benchmark ===");
    println!();

    let mesh = generate_sphere(1.0, 64, 64);
    let n_vertices = mesh.n_vertices();

    println!("Mesh: {} vertices, {} faces", n_vertices, mesh.n_faces());
    println!();

    // Warm up
    for _ in 0..100000 {
        unsafe {
            let _ = mesh.vertex_sum_simd();
        }
    }

    // Benchmark
    let iterations = 500_000;
    let start = Instant::now();
    for _ in 0..iterations {
        unsafe {
            let s = mesh.vertex_sum_simd();
            black_box(s);
        }
    }
    let simd_ns = start.elapsed().as_nanos() as f64 / iterations as f64;
    let per_vertex = simd_ns / n_vertices as f64;

    println!("=== Results ===");
    println!("SIMD (RustMesh): {:.3} ns/batch", simd_ns);
    println!("Per vertex: {:.3} ns", per_vertex);
    println!();
    println!("OpenMesh: 0.27 ns/vertex");

    if per_vertex < 0.27 {
        let speedup = (1.0 - per_vertex / 0.27) * 100.0;
        println!();
        println!("✅ {:.1}% faster!", speedup);
    } else {
        let slower = per_vertex / 0.27;
        println!();
        println!("⚠️  {:.1}x slower", slower);
    }
}
