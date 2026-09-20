//! QuadricT Mesh Decimation Test
use rustscan_mesh::{
    generate_cube, generate_sphere, generate_torus, Point3, QuadricT, RustMesh, Vec3,
};

fn main() {
    println!("=== QuadricT Mesh Decimation Test ===\n");

    println!("1. Cube (8 vertices, 6 faces)");
    test_mesh(&generate_cube(), 8, 6);

    println!("\n2. Sphere 16x16");
    test_mesh(&generate_sphere(1.0, 16, 16), 272, 544);

    println!("\n3. Sphere 32x32");
    test_mesh(&generate_sphere(1.0, 32, 32), 1056, 2112);

    println!("\n4. Torus");
    test_mesh(&generate_torus(2.0, 0.5, 24, 12), 325, 288);

    println!("\n=== All tests completed! ===");
}

fn test_mesh(mesh: &RustMesh, expected_v: usize, expected_f: usize) {
    let n_v = mesh.n_vertices();
    let n_f = mesh.n_faces();
    println!("  Input: {} vertices, {} faces", n_v, n_f);

    let mut total_error = 0.0f32;
    let mut optimized = 0;
    let mut face_count = 0;

    for fh in mesh.faces() {
        if let Some(verts) = mesh.face_vertices(fh) {
            let vs: Vec<_> = verts.collect();
            if vs.len() >= 3 {
                let p0 = mesh.point(vs[0]).unwrap();
                let p1 = mesh.point(vs[1]).unwrap();
                let p2 = mesh.point(vs[2]).unwrap();
                let edge1 = p1 - p0;
                let edge2 = p2 - p0;
                let normal = edge1.cross(&edge2).normalize();
                let center = Point3::from((p0.coords + p1.coords + p2.coords) / 3.0);
                let q = QuadricT::from_face(normal, center);

                let (opt, err) = q.optimize();
                total_error += err;
                optimized += 1;
                face_count += 1;
            }
        }
    }

    println!("  Faces processed: {}", face_count);
    println!("  Quadrics built: {}", optimized);
    println!("  Avg face error: {:.6}", total_error / optimized as f32);
}
