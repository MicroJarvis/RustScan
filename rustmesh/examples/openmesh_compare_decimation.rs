mod openmesh_compare_common;

use openmesh_compare_common::{
    cleanup_paths, load_mesh, measure, mesh_digest, off_digest, openmesh_tool,
    print_duration_compare, print_header, print_mesh_digest, write_temp_off,
};
use rustmesh::{generate_sphere, read_off, write_off, Decimater};
use std::collections::HashMap;

fn raw_face_diagnostics(mesh: &rustmesh::RustMesh) -> (usize, usize, usize) {
    let mut active_faces = 0usize;
    let mut degenerate_faces = 0usize;
    let mut edge_use: HashMap<(usize, usize), usize> = HashMap::new();

    for fh in mesh.faces() {
        let vertices = mesh.face_vertices_vec(fh);
        if vertices.len() < 3 {
            continue;
        }
        active_faces += 1;
        if vertices.len() != 3 {
            degenerate_faces += 1;
            continue;
        }

        let ids = [
            vertices[0].idx_usize(),
            vertices[1].idx_usize(),
            vertices[2].idx_usize(),
        ];
        if ids[0] == ids[1] || ids[1] == ids[2] || ids[2] == ids[0] {
            degenerate_faces += 1;
        }

        for (a, b) in [(ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])] {
            let key = if a < b { (a, b) } else { (b, a) };
            *edge_use.entry(key).or_insert(0) += 1;
        }
    }

    let non_manifold_edges = edge_use.values().filter(|&&count| count > 2).count();
    (active_faces, degenerate_faces, non_manifold_edges)
}

fn main() {
    print_header("RustMesh Decimation vs OpenMesh Decimater");
    println!("Reference sources:");
    println!("  - Mirror/OpenMesh-11.0.0/Doc/Examples/decimater.cc");
    println!("  - Mirror/OpenMesh-11.0.0/src/OpenMesh/Apps/Decimating/decimater.cc");

    let input = generate_sphere(1.0, 10, 10);
    let input_path = write_temp_off(&input, "decimate-input").expect("write input OFF");
    let rust_output = input_path.with_file_name("rustmesh-decimate-output.off");
    let openmesh_output = input_path.with_file_name("openmesh-decimate-output.off");

    print_mesh_digest("Input mesh", mesh_digest(&input));
    let target_vertices = input.n_vertices() / 2;

    let (rust_time, rust_digest) = measure(|| {
        let mut mesh = read_off(&input_path).expect("read OFF for RustMesh");
        let mut decimater = Decimater::new(&mut mesh);
        let collapsed = decimater.decimate_to(target_vertices);
        let boundary_collapses = decimater.boundary_collapses();
        let interior_collapses = decimater.interior_collapses();
        let faces_removed_estimate = decimater.faces_removed_estimate();
        drop(decimater);
        let (raw_faces, raw_degenerate, raw_non_manifold_edges) = raw_face_diagnostics(&mesh);
        println!(
            "RustMesh raw decimation: active_faces={raw_faces}, degenerate_faces={raw_degenerate}, non_manifold_edges={raw_non_manifold_edges}"
        );
        mesh.garbage_collection();
        write_off(&mesh, &rust_output).expect("write RustMesh OFF");
        println!("RustMesh collapsed edges: {collapsed}");
        println!(
            "RustMesh collapse mix: boundary={}, interior={}, estimated_faces_removed={}",
            boundary_collapses, interior_collapses, faces_removed_estimate
        );
        mesh_digest(&mesh)
    });
    print_mesh_digest("RustMesh output", rust_digest);
    match load_mesh(&rust_output) {
        Ok(roundtrip_mesh) => {
            print_mesh_digest("RustMesh roundtrip output", mesh_digest(&roundtrip_mesh));
        }
        Err(err) => {
            println!("RustMesh roundtrip warning: failed to reload serialized output: {err}");
        }
    }

    let (openmesh_time, openmesh_digest) = measure(|| {
        let output = openmesh_tool("commandlineDecimater")
            .arg("-i")
            .arg(&input_path)
            .arg("-o")
            .arg(&openmesh_output)
            .arg("-n")
            .arg("0.5")
            .arg("-M")
            .arg("Q")
            .output()
            .expect("run OpenMesh decimater");

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stdout.trim().is_empty() {
            println!("OpenMesh stdout:\n{stdout}");
        }
        if !stderr.trim().is_empty() {
            println!("OpenMesh stderr:\n{stderr}");
        }
        assert!(output.status.success(), "OpenMesh decimater failed");

        off_digest(&openmesh_output).expect("parse OpenMesh OFF output")
    });
    print_mesh_digest("OpenMesh output", openmesh_digest);

    print_duration_compare("Decimation pipeline", rust_time, Some(openmesh_time));
    cleanup_paths(&[
        input_path.as_path(),
        rust_output.as_path(),
        openmesh_output.as_path(),
    ]);
}
