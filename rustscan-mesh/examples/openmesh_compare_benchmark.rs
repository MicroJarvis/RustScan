mod openmesh_compare_common;

use openmesh_compare_common::{
    bench_ns_per_iter, black_box_value, measure, mesh_digest, openmesh_benchmark_binary,
    parse_openmesh_tri_benchmark, print_duration_compare, print_header, print_mesh_digest,
    reference_root, run_capture,
};
use rustscan_mesh::{triangle_area, Point3, RustMesh, Vec3};
use std::time::Duration;

fn build_tetrahedron() -> RustMesh {
    let mut mesh = RustMesh::new();
    let v0 = mesh.add_vertex(Point3::new(-1.0, -1.0, -1.0));
    let v1 = mesh.add_vertex(Point3::new(1.0, -1.0, -1.0));
    let v2 = mesh.add_vertex(Point3::new(1.0, 1.0, -1.0));
    let v3 = mesh.add_vertex(Point3::new(-1.0, 1.0, -1.0));

    mesh.add_face(&[v0, v1, v2]).unwrap();
    mesh.add_face(&[v0, v2, v3]).unwrap();
    mesh.add_face(&[v0, v3, v1]).unwrap();
    mesh.add_face(&[v1, v3, v2]).unwrap();
    mesh
}

fn build_benchmark_sphere() -> RustMesh {
    rustscan_mesh::generate_sphere(1.0, 32, 32)
}

fn measure_triangle_area(mesh: &RustMesh) -> (usize, f32) {
    let mut iterations = 0usize;
    let mut accumulated = 0.0f32;

    for _ in 0..1000 {
        for fh in mesh.faces() {
            let Some(mut fv) = mesh.face_vertices(fh) else {
                continue;
            };
            let Some(v0) = fv.next() else { continue };
            let Some(v1) = fv.next() else { continue };
            let Some(v2) = fv.next() else { continue };

            let p0 = mesh.point(v0).unwrap_or(Point3::origin());
            let p1 = mesh.point(v1).unwrap_or(Point3::origin());
            let p2 = mesh.point(v2).unwrap_or(Point3::origin());
            accumulated += triangle_area(p0, p1, p2);
            iterations += 1;
        }
    }

    (iterations, black_box_value(accumulated))
}

fn main() {
    print_header("RustMesh vs OpenMesh TriMesh Benchmark");
    println!("Reference source: Mirror/OpenMesh-11.0.0/build/OpenMeshBenchmark.cc");

    let openmesh_reference = if openmesh_benchmark_binary().exists() {
        let mut command = std::process::Command::new(openmesh_benchmark_binary());
        command.current_dir(reference_root());
        run_capture(&mut command).ok().map(|output| {
            println!("\nOpenMesh benchmark output:");
            println!("{output}");
            parse_openmesh_tri_benchmark(&output)
        })
    } else {
        None
    };

    print_header("Case 1: Tetrahedron Build");
    let (tet_time, tet_faces) = measure(|| {
        let mut faces = 0usize;
        for _ in 0..1000 {
            let mesh = build_tetrahedron();
            faces += mesh.n_faces();
        }
        faces
    });
    println!("RustMesh tetrahedra built: {}", tet_faces / 4);
    print_duration_compare(
        "Build 1000 tetrahedra",
        tet_time,
        openmesh_reference
            .and_then(|reference| reference.tetrahedron_us)
            .map(|value| Duration::from_secs_f64(value / 1_000_000.0)),
    );

    print_header("Case 2: Mesh Traversal");
    let sphere = build_benchmark_sphere();
    let digest = mesh_digest(&sphere);
    print_mesh_digest("RustMesh sphere", digest);

    let vertex_count = sphere.vertices().count();
    let face_count = sphere.faces().count();
    let vertex_traversal_ns = bench_ns_per_iter(1_000, || {
        let mut visited = 0usize;
        for _ in sphere.vertices() {
            visited += 1;
        }
        black_box_value(visited);
    });
    let face_traversal_ns = bench_ns_per_iter(1_000, || {
        let mut visited = 0usize;
        for _ in sphere.faces() {
            visited += 1;
        }
        black_box_value(visited);
    });
    println!("Traversal counts: vertices={vertex_count}, faces={face_count}");
    match openmesh_reference.and_then(|reference| reference.vertex_traversal_ns) {
        Some(openmesh_ns) => println!(
            "Vertex traversal: RustMesh={vertex_traversal_ns:.3} ns, OpenMesh={openmesh_ns:.3} ns, OpenMesh/rustmesh={:.2}x",
            openmesh_ns / vertex_traversal_ns
        ),
        None => println!("Vertex traversal: RustMesh={vertex_traversal_ns:.3} ns"),
    }
    match openmesh_reference.and_then(|reference| reference.face_traversal_ns) {
        Some(openmesh_ns) => {
            if face_traversal_ns > 0.0 {
                println!(
                    "Face traversal: RustMesh={face_traversal_ns:.3} ns, OpenMesh={openmesh_ns:.3} ns, OpenMesh/rustmesh={:.2}x",
                    openmesh_ns / face_traversal_ns
                );
            } else {
                println!(
                    "Face traversal: RustMesh={face_traversal_ns:.3} ns, OpenMesh={openmesh_ns:.3} ns"
                );
            }
        }
        None => println!("Face traversal: RustMesh={face_traversal_ns:.3} ns"),
    }

    print_header("Case 3: Add 1000 Triangles");
    let (add_time, triangle_count) = measure(|| {
        let mut total = 0usize;
        for _ in 0..1000 {
            let mut mesh = RustMesh::new();
            for j in 0..1000 {
                let v0 = mesh.add_vertex(Point3::new(j as f32, 0.0, 0.0));
                let v1 = mesh.add_vertex(Point3::new(j as f32 + 1.0, 0.0, 0.0));
                let v2 = mesh.add_vertex(Point3::new(j as f32 + 1.0, 1.0, 0.0));
                if mesh.add_face(&[v0, v1, v2]).is_some() {
                    total += 1;
                }
            }
        }
        total
    });
    println!("RustMesh triangles added: {triangle_count}");
    print_duration_compare(
        "Add 1000 triangles x 1000",
        add_time,
        openmesh_reference
            .and_then(|reference| reference.add_triangles_us)
            .map(|value| Duration::from_secs_f64(value / 1_000_000.0)),
    );

    print_header("Case 4: Triangle Area");
    let (area_time, (area_iterations, accumulated_area)) =
        measure(|| measure_triangle_area(&sphere));
    println!(
        "Triangle area iterations: {area_iterations}, accumulated area checksum: {:.6}",
        accumulated_area
    );
    print_duration_compare(
        "Triangle area workload",
        area_time,
        openmesh_reference
            .and_then(|reference| reference.triangle_area_us)
            .map(|value| Duration::from_secs_f64(value / 1_000_000.0)),
    );
}
