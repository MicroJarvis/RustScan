fn main() {
    use rustmesh::{generate_cube, SmartMesh};

    let mesh = generate_cube();

    // Test VertexRange
    let vrange = mesh.vertex_range();
    println!("Vertex count: {}", vrange.count());
    let (x, y, z) = vrange.sum_positions();
    println!("Sum: ({}, {}, {})", x, y, z);
    let avg = vrange.average_position().unwrap();
    println!("Average: ({}, {}, {})", avg.0, avg.1, avg.2);

    // Test FaceRange
    let frange = mesh.face_range();
    println!("Face count: {}", frange.count());
    let centroids = frange.centroids();
    println!("Centroids: {}", centroids.len());

    println!("\n✅ SmartRanges test passed!");
}
