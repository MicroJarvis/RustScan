mod openmesh_compare_common;

use openmesh_compare_common::{
    active_faces, active_vertices, bench_ns_per_iter, cleanup_paths, load_mesh, measure,
    mesh_digest, print_header, print_mesh_digest, write_temp_off,
};
use rustscan_mesh::{
    generate_cube, generate_sphere, read_obj, write_off, FaceHandle, Point3, RustMesh, Vec3,
    VertexHandle,
};
use std::path::Path;

fn build_tutorial08_cube() -> RustMesh {
    let mut mesh = RustMesh::new();

    let v = [
        mesh.add_vertex(Point3::new(-1.0, -1.0, 1.0)),
        mesh.add_vertex(Point3::new(1.0, -1.0, 1.0)),
        mesh.add_vertex(Point3::new(1.0, 1.0, 1.0)),
        mesh.add_vertex(Point3::new(-1.0, 1.0, 1.0)),
        mesh.add_vertex(Point3::new(-1.0, -1.0, -1.0)),
        mesh.add_vertex(Point3::new(1.0, -1.0, -1.0)),
        mesh.add_vertex(Point3::new(1.0, 1.0, -1.0)),
        mesh.add_vertex(Point3::new(-1.0, 1.0, -1.0)),
    ];

    mesh.add_face(&[v[0], v[1], v[2], v[3]]).unwrap();
    mesh.add_face(&[v[7], v[6], v[5], v[4]]).unwrap();
    mesh.add_face(&[v[1], v[0], v[4], v[5]]).unwrap();
    mesh.add_face(&[v[2], v[1], v[5], v[6]]).unwrap();
    mesh.add_face(&[v[3], v[2], v[6], v[7]]).unwrap();
    mesh.add_face(&[v[0], v[3], v[7], v[4]]).unwrap();

    mesh
}

fn main() {
    print_header("RustMesh Example Parity with OpenMesh Tutorials");
    println!("Reference sources:");
    println!("  - Mirror/OpenMesh-11.0.0/src/OpenMesh/Examples/Tutorial01/cube.cc");
    println!("  - Mirror/OpenMesh-11.0.0/Doc/Examples/iterators.cc");
    println!("  - Mirror/OpenMesh-11.0.0/Doc/Examples/circulators.cc");
    println!("  - Mirror/OpenMesh-11.0.0/Doc/Examples/mesh_io.cc");
    println!("  - Mirror/OpenMesh-11.0.0/src/OpenMesh/Examples/Tutorial08/delete_geometry.cc");

    print_header("Tutorial01: Cube Build + OFF Roundtrip");
    let (build_time, cube) = measure(generate_cube);
    let cube_digest = mesh_digest(&cube);
    print_mesh_digest("Generated cube", cube_digest);
    println!(
        "OpenMesh tutorial expectation: V=8, F=6. RustMesh: V={}, F={}",
        cube_digest.vertices, cube_digest.faces
    );

    let off_path = write_temp_off(&cube, "tutorial01-cube").expect("write OFF");
    let (roundtrip_time, roundtrip) = measure(|| load_mesh(&off_path).expect("read OFF"));
    print_mesh_digest("Roundtrip cube", mesh_digest(&roundtrip));
    println!(
        "Timings: build={:.3} ms, off_roundtrip={:.3} ms",
        build_time.as_secs_f64() * 1_000.0,
        roundtrip_time.as_secs_f64() * 1_000.0
    );
    cleanup_paths(&[off_path.as_path()]);

    print_header("Iterators: Handle vs Index Traversal");
    let sphere = generate_sphere(1.0, 32, 32);
    println!(
        "Sphere stats: active_vertices={}, active_faces={}",
        active_vertices(&sphere),
        active_faces(&sphere)
    );

    let handle_vertex_ns = bench_ns_per_iter(1_000, || {
        let mut acc = 0usize;
        for vh in sphere.vertices() {
            acc += vh.idx_usize();
        }
        openmesh_compare_common::black_box_value(acc);
    });
    let index_vertex_ns = bench_ns_per_iter(1_000, || {
        let mut acc = 0usize;
        for idx in sphere.vertex_indices() {
            acc += idx;
        }
        openmesh_compare_common::black_box_value(acc);
    });
    println!(
        "Vertex iteration result parity: handle_sum={}, index_sum={}",
        sphere.vertices().map(|vh| vh.idx_usize()).sum::<usize>(),
        sphere.vertex_indices().sum::<usize>()
    );
    println!(
        "Vertex traversal timing: OpenMesh-style handles={handle_vertex_ns:.2} ns, RustMesh index path={index_vertex_ns:.2} ns"
    );

    let handle_face_ns = bench_ns_per_iter(1_000, || {
        let mut acc = 0usize;
        for fh in sphere.faces() {
            acc += fh.idx_usize();
        }
        openmesh_compare_common::black_box_value(acc);
    });
    let index_face_ns = bench_ns_per_iter(1_000, || {
        let mut acc = 0usize;
        for idx in sphere.face_indices() {
            acc += idx;
        }
        openmesh_compare_common::black_box_value(acc);
    });
    println!(
        "Face iteration result parity: handle_sum={}, index_sum={}",
        sphere.faces().map(|fh| fh.idx_usize()).sum::<usize>(),
        sphere.face_indices().sum::<usize>()
    );
    println!(
        "Face traversal timing: OpenMesh-style handles={handle_face_ns:.2} ns, RustMesh index path={index_face_ns:.2} ns"
    );

    print_header("Circulators: Vertex Neighbors");
    let cube = generate_cube();
    let center = VertexHandle::new(0);
    let neighbors: Vec<_> = cube
        .vertex_vertices(center)
        .expect("cube vertex neighbors")
        .map(|vh| vh.idx_usize())
        .collect();
    println!("Vertex 0 neighbors (OpenMesh vv_iter equivalent): {neighbors:?}");
    let vv_ns = bench_ns_per_iter(10_000, || {
        let mut count = 0usize;
        if let Some(vv) = cube.vertex_vertices(center) {
            for _ in vv {
                count += 1;
            }
        }
        openmesh_compare_common::black_box_value(count);
    });
    println!("Neighbor count: {}, timing: {vv_ns:.2} ns", neighbors.len());

    print_header("Mesh I/O: OBJ -> OFF");
    let input_obj = Path::new("../artifacts/inputs/middle/cube.obj");
    let (read_time, obj_mesh) = measure(|| read_obj(input_obj).expect("read OBJ"));
    let output_off = Path::new("/tmp/rustmesh-openmesh-compare-obj-to-off.off");
    let (write_time, ()) = measure(|| write_off(&obj_mesh, output_off).expect("write OFF"));
    let converted = load_mesh(output_off).expect("read converted OFF");
    print_mesh_digest("Converted mesh", mesh_digest(&converted));
    println!(
        "Timings: read_obj={:.3} ms, write_off={:.3} ms",
        read_time.as_secs_f64() * 1_000.0,
        write_time.as_secs_f64() * 1_000.0
    );
    cleanup_paths(&[output_off]);

    print_header("Tutorial08: Delete Geometry + Garbage Collection");
    let (delete_time, mut delete_mesh) = measure(|| {
        let mut mesh = build_tutorial08_cube();
        for face_idx in [0_u32, 2, 3, 4, 5] {
            mesh.delete_face(FaceHandle::new(face_idx));
        }
        for vertex_idx in [0_u32, 1, 2, 3] {
            mesh.delete_vertex(VertexHandle::new(vertex_idx));
        }
        mesh
    });
    println!(
        "OpenMesh tutorial expectation: keep only face [7, 6, 5, 4], so active V=4 and active F=1."
    );
    println!(
        "RustMesh before GC: active_vertices={}, active_faces={}",
        active_vertices(&delete_mesh),
        active_faces(&delete_mesh)
    );
    print_mesh_digest("RustMesh marked-deleted digest", mesh_digest(&delete_mesh));
    let (gc_time, ()) = measure(|| delete_mesh.garbage_collection());
    let delete_digest = mesh_digest(&delete_mesh);
    print_mesh_digest("RustMesh post-GC digest", delete_digest);
    println!(
        "OpenMesh post-GC expectation: V=4, F=1, centroid=(0.0, 0.0, -1.0). RustMesh: V={}, F={}, centroid=({:.4}, {:.4}, {:.4})",
        delete_digest.vertices,
        delete_digest.faces,
        delete_digest.centroid.x,
        delete_digest.centroid.y,
        delete_digest.centroid.z
    );
    println!(
        "Timings: mark_deleted={:.3} ms, garbage_collection={:.3} ms",
        delete_time.as_secs_f64() * 1_000.0,
        gc_time.as_secs_f64() * 1_000.0
    );
}
