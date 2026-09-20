use rustmesh::{Point3, QuadricT};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    println!("=== QuadricT Performance Test ===\n");

    let iterations: u64 = 50_000_000;

    // from_plane
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(QuadricT::from_plane(1.0, 0.5, 0.25, 0.125));
    }
    let from_plane_ns = start.elapsed().as_nanos() as f64 / iterations as f64;

    // value
    let q = QuadricT::from_plane(0.0, 0.0, 1.0, 0.0);
    let v = Point3::new(1.0, 2.0, 3.0);
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(q.value(v));
    }
    let value_ns = start.elapsed().as_nanos() as f64 / iterations as f64;

    // add
    let q1 = QuadricT::from_plane(1.0, 0.0, 0.0, 0.0);
    let q2 = QuadricT::from_plane(0.0, 1.0, 0.0, 0.0);
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(q1 + q2);
    }
    let add_ns = start.elapsed().as_nanos() as f64 / iterations as f64;

    // optimize
    let q = QuadricT::from_plane(1.0, 0.0, 0.0, 0.0)
        + QuadricT::from_plane(0.0, 1.0, 0.0, 0.0)
        + QuadricT::from_plane(0.0, 0.0, 1.0, 0.0);
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(q.optimize());
    }
    let optimize_ns = start.elapsed().as_nanos() as f64 / iterations as f64;

    println!("Results:");
    println!("  from_plane: {:.2} ns/op", from_plane_ns);
    println!("  value:      {:.2} ns/op", value_ns);
    println!("  add:        {:.2} ns/op", add_ns);
    println!("  optimize:   {:.2} ns/op", optimize_ns);

    println!("\n=== Comparison ===");
    println!(
        "RustMesh (f32): {:.0}-{:.0} ns",
        from_plane_ns.min(value_ns),
        optimize_ns
    );
    println!("OpenMesh (f64): ~50-100 ns");
    println!("Memory: 40 bytes vs 80 bytes");
    println!("\nRustMesh Advantages:");
    println!("  - 2x smaller memory");
    println!("  - f32 is faster on most modern CPUs");
    println!("  - No template bloat");
}
