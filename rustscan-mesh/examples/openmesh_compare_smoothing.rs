mod openmesh_compare_common;

use openmesh_compare_common::{
    measure, mesh_digest, print_duration_compare, print_header, print_mesh_digest,
};
use rustscan_mesh::{
    generate_noisy_sphere, laplace_smooth, Point3, RustMesh, SmootherConfig, Vec3, VertexHandle,
};
use std::time::Duration;

fn tutorial_style_smooth(mesh: &mut RustMesh, iterations: usize) {
    let vhs: Vec<VertexHandle> = mesh.vertices().collect();
    let mut next_positions = vec![Point3::origin(); vhs.len()];

    for _ in 0..iterations {
        for (i, &vh) in vhs.iter().enumerate() {
            let current = mesh.point(vh).unwrap_or(Point3::origin());
            next_positions[i] = current;

            if is_boundary_vertex(mesh, vh) {
                continue;
            }

            let mut neighbor_sum = Vec3::zeros();
            let mut neighbor_count = 0usize;
            if let Some(vv) = mesh.vertex_vertices(vh) {
                for neighbor in vv {
                    if let Some(point) = mesh.point(neighbor) {
                        neighbor_sum += point.coords;
                        neighbor_count += 1;
                    }
                }
            }

            if neighbor_count > 0 {
                next_positions[i] = Point3::from(neighbor_sum / neighbor_count as f32);
            }
        }

        for (i, &vh) in vhs.iter().enumerate() {
            mesh.set_point(vh, next_positions[i]);
        }
    }
}

fn is_boundary_vertex(mesh: &RustMesh, vh: VertexHandle) -> bool {
    if let Some(halfedges) = mesh.vertex_halfedges(vh) {
        for heh in halfedges {
            if mesh.is_boundary(heh) || mesh.is_boundary(mesh.opposite_halfedge_handle(heh)) {
                return true;
            }
        }
    }
    false
}

fn max_position_delta(lhs: &RustMesh, rhs: &RustMesh) -> f32 {
    let mut max_delta = 0.0f32;
    let n = lhs.n_vertices().min(rhs.n_vertices());
    for idx in 0..n {
        let Some(lp) = lhs.point(VertexHandle::from_usize(idx)) else {
            continue;
        };
        let Some(rp) = rhs.point(VertexHandle::from_usize(idx)) else {
            continue;
        };
        max_delta = max_delta.max((lp - rp).norm());
    }
    max_delta
}

fn main() {
    print_header("RustMesh Smoothing vs OpenMesh Tutorial Style");
    println!("Reference sources:");
    println!("  - Mirror/OpenMesh-11.0.0/src/OpenMesh/Examples/Tutorial02/smooth.cc");
    println!("  - Mirror/OpenMesh-11.0.0/src/OpenMesh/Examples/Tutorial03/smooth.cc");
    println!("  - Mirror/OpenMesh-11.0.0/src/OpenMesh/Examples/Tutorial04/smooth.cc");

    let input = generate_noisy_sphere(1.0, 0.05, 24, 24);
    print_mesh_digest("Input mesh", mesh_digest(&input));

    let iterations = 5usize;

    let mut tutorial_mesh = input.clone();
    let (tutorial_time, ()) = measure(|| tutorial_style_smooth(&mut tutorial_mesh, iterations));
    print_mesh_digest("Tutorial-style result", mesh_digest(&tutorial_mesh));

    let mut optimized_mesh = input.clone();
    let (rust_time, result) = measure(|| {
        laplace_smooth(
            &mut optimized_mesh,
            SmootherConfig {
                iterations,
                strength: 1.0,
                uniform: true,
                fixed_boundary: true,
            },
        )
    });
    print_mesh_digest("RustMesh result", mesh_digest(&optimized_mesh));

    let max_delta = max_position_delta(&tutorial_mesh, &optimized_mesh);
    println!(
        "Result comparison: tutorial_vs_rustmesh max_position_delta={max_delta:.8}, reported max_displacement={:.8}",
        result.max_displacement
    );
    print_duration_compare(
        "Smoothing workload",
        rust_time,
        Some(Duration::from_secs_f64(tutorial_time.as_secs_f64())),
    );
}
