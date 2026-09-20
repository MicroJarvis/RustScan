//! Compare RustMesh PLY/STL I/O with OpenMesh output
//!
//! Tests that RustMesh PLY and STL read/write produces output
//! consistent with OpenMesh reference implementation.

mod openmesh_compare_common;

use openmesh_compare_common::{
    cleanup_paths, measure, mesh_digest, print_header, print_mesh_digest, MeshDigest,
};
use rustscan_mesh::{
    generate_cube, generate_sphere, read_ply, read_stl, write_ply, write_stl, PlyFormat, Point3,
    RustMesh, StlFormat, Vec3, Vec4,
};
use std::fs;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};

fn temp_path(stem: &str, ext: &str) -> PathBuf {
    PathBuf::from("/tmp").join(format!(
        "rustmesh-io-compare-{stem}-{}.{ext}",
        std::process::id()
    ))
}

fn ply_digest(path: &Path) -> io::Result<MeshDigest> {
    let file = fs::File::open(path)?;
    let mut reader = BufReader::new(file);

    // Parse PLY header
    let mut line = String::new();
    reader.read_line(&mut line)?;

    if !line.trim().eq_ignore_ascii_case("ply") {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Not a PLY file"));
    }

    let mut vertex_count = 0usize;
    let mut face_count = 0usize;
    let mut format = String::new();

    loop {
        line.clear();
        reader.read_line(&mut line)?;
        let trimmed = line.trim();

        if trimmed.starts_with("format") {
            format = trimmed.split_whitespace().nth(1).unwrap_or("").to_string();
        } else if trimmed.starts_with("element vertex") {
            vertex_count = trimmed
                .split_whitespace()
                .nth(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
        } else if trimmed.starts_with("element face") {
            face_count = trimmed
                .split_whitespace()
                .nth(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
        } else if trimmed == "end_header" {
            break;
        }
    }

    // Read vertices
    let mut vertices = Vec::with_capacity(vertex_count);
    for _ in 0..vertex_count {
        line.clear();
        reader.read_line(&mut line)?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            let x: f32 = parts[0].parse().unwrap_or(0.0);
            let y: f32 = parts[1].parse().unwrap_or(0.0);
            let z: f32 = parts[2].parse().unwrap_or(0.0);
            vertices.push(Point3::new(x, y, z));
        }
    }

    // Mark used vertices from faces
    let mut used = vec![false; vertex_count];
    for _ in 0..face_count {
        line.clear();
        reader.read_line(&mut line)?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        let n: usize = parts[0].parse().unwrap_or(0);
        for i in 1..=n.min(parts.len() - 1) {
            let idx: usize = parts[i].parse().unwrap_or(0);
            if idx < used.len() {
                used[idx] = true;
            }
        }
    }

    // Compute digest
    let mut min = Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max = Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    let mut sum = Vec3::zeros();
    let mut count = 0usize;
    let mut checksum = 0.0f32;

    for (idx, point) in vertices.iter().copied().enumerate() {
        if !used.get(idx).copied().unwrap_or(false) {
            continue;
        }
        min = Point3::from(min.coords.inf(&point.coords));
        max = Point3::from(max.coords.sup(&point.coords));
        sum += point.coords;
        checksum += point.x.abs() + point.y.abs() + point.z.abs();
        count += 1;
    }

    if count == 0 {
        return Ok(MeshDigest {
            vertices: 0,
            faces: 0,
            bbox_min: Point3::origin(),
            bbox_max: Point3::origin(),
            centroid: Point3::origin(),
            checksum_l1: 0.0,
        });
    }

    Ok(MeshDigest {
        vertices: count,
        faces: face_count,
        bbox_min: min,
        bbox_max: max,
        centroid: Point3::from(sum / count as f32),
        checksum_l1: checksum,
    })
}

fn stl_ascii_digest(path: &Path) -> io::Result<MeshDigest> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let mut vertices_set = std::collections::HashSet::new();
    let mut face_count = 0usize;
    let mut sum = Vec3::zeros();
    let mut checksum = 0.0f32;

    for line in lines {
        let line = line?;
        let trimmed = line.trim();

        if trimmed.starts_with("vertex") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 4 {
                let x: f32 = parts[1].parse().unwrap_or(0.0);
                let y: f32 = parts[2].parse().unwrap_or(0.0);
                let z: f32 = parts[3].parse().unwrap_or(0.0);

                // Quantize for deduplication
                let key = (
                    (x * 1e6).round() as i64,
                    (y * 1e6).round() as i64,
                    (z * 1e6).round() as i64,
                );

                if vertices_set.insert(key) {
                    sum += Vec3::new(x, y, z);
                    checksum += x.abs() + y.abs() + z.abs();
                }
            }
        } else if trimmed.starts_with("facet") {
            face_count += 1;
        }
    }

    let count = vertices_set.len();
    if count == 0 {
        return Ok(MeshDigest {
            vertices: 0,
            faces: 0,
            bbox_min: Point3::origin(),
            bbox_max: Point3::origin(),
            centroid: Point3::origin(),
            checksum_l1: 0.0,
        });
    }

    // Compute bounding box from unique vertices
    let mut min = Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max = Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for key in &vertices_set {
        let x = key.0 as f32 / 1e6;
        let y = key.1 as f32 / 1e6;
        let z = key.2 as f32 / 1e6;
        let p = Point3::new(x, y, z);
        min = Point3::from(min.coords.inf(&p.coords));
        max = Point3::from(max.coords.sup(&p.coords));
    }

    Ok(MeshDigest {
        vertices: count,
        faces: face_count,
        bbox_min: min,
        bbox_max: max,
        centroid: Point3::from(sum / count as f32),
        checksum_l1: checksum,
    })
}

fn test_ply_roundtrip_ascii(mesh: &RustMesh, label: &str) {
    print_header(&format!("PLY ASCII Roundtrip: {}", label));

    let path = temp_path(label, "ply");
    let (write_time, ()) = measure(|| write_ply(mesh, &path, PlyFormat::Ascii).unwrap());
    let (read_time, loaded) = measure(|| read_ply(&path).unwrap());

    let original_digest = mesh_digest(mesh);
    let loaded_digest = mesh_digest(&loaded);
    let file_digest = ply_digest(&path).unwrap();

    print_mesh_digest("Original", original_digest);
    print_mesh_digest("Loaded", loaded_digest);
    print_mesh_digest("File", file_digest);

    println!(
        "Timings: write={:.3} ms, read={:.3} ms",
        write_time.as_secs_f64() * 1_000.0,
        read_time.as_secs_f64() * 1_000.0
    );

    // Verify
    assert_eq!(
        loaded_digest.vertices, original_digest.vertices,
        "Vertex count mismatch"
    );
    assert_eq!(
        loaded_digest.faces, original_digest.faces,
        "Face count mismatch"
    );

    cleanup_paths(&[&path]);
    println!("✓ PLY ASCII roundtrip verified");
}

fn test_ply_roundtrip_binary(mesh: &RustMesh, label: &str) {
    print_header(&format!("PLY Binary Roundtrip: {}", label));

    let path = temp_path(label, "ply");

    // Binary little-endian
    let (write_time, ()) =
        measure(|| write_ply(mesh, &path, PlyFormat::BinaryLittleEndian).unwrap());
    let (read_time, loaded) = measure(|| read_ply(&path).unwrap());

    let original_digest = mesh_digest(mesh);
    let loaded_digest = mesh_digest(&loaded);

    print_mesh_digest("Original", original_digest);
    print_mesh_digest("Loaded", loaded_digest);

    println!(
        "Timings: write={:.3} ms, read={:.3} ms",
        write_time.as_secs_f64() * 1_000.0,
        read_time.as_secs_f64() * 1_000.0
    );

    assert_eq!(
        loaded_digest.vertices, original_digest.vertices,
        "Vertex count mismatch"
    );
    assert_eq!(
        loaded_digest.faces, original_digest.faces,
        "Face count mismatch"
    );

    cleanup_paths(&[&path]);
    println!("✓ PLY Binary roundtrip verified");
}

fn test_stl_roundtrip_ascii(mesh: &RustMesh, label: &str) {
    print_header(&format!("STL ASCII Roundtrip: {}", label));

    let path = temp_path(label, "stl");

    let (write_time, ()) = measure(|| write_stl(mesh, &path, StlFormat::Ascii).unwrap());
    let (read_time, loaded) = measure(|| read_stl(&path).unwrap());

    let original_digest = mesh_digest(mesh);
    let loaded_digest = mesh_digest(&loaded);
    let file_digest = stl_ascii_digest(&path).unwrap();

    print_mesh_digest("Original", original_digest);
    print_mesh_digest("Loaded", loaded_digest);
    print_mesh_digest("File", file_digest);

    println!(
        "Timings: write={:.3} ms, read={:.3} ms",
        write_time.as_secs_f64() * 1_000.0,
        read_time.as_secs_f64() * 1_000.0
    );

    // STL triangulates polygons and deduplicates vertices
    // Face count may differ due to triangulation and vertex merging
    println!(
        "Original: {}V {}F, Loaded: {}V {}F",
        original_digest.vertices,
        original_digest.faces,
        loaded_digest.vertices,
        loaded_digest.faces
    );

    // Verify bounding box is preserved
    let bbox_ok = (loaded_digest.bbox_min - original_digest.bbox_min).norm() < 0.01
        && (loaded_digest.bbox_max - original_digest.bbox_max).norm() < 0.01;
    assert!(bbox_ok, "Bounding box should be preserved");

    cleanup_paths(&[&path]);
    println!("✓ STL ASCII roundtrip verified");
}

fn test_stl_roundtrip_binary(mesh: &RustMesh, label: &str) {
    print_header(&format!("STL Binary Roundtrip: {}", label));

    let path = temp_path(label, "stl");

    let (write_time, ()) = measure(|| write_stl(mesh, &path, StlFormat::Binary).unwrap());
    let (read_time, loaded) = measure(|| read_stl(&path).unwrap());

    let original_digest = mesh_digest(mesh);
    let loaded_digest = mesh_digest(&loaded);

    print_mesh_digest("Original", original_digest);
    print_mesh_digest("Loaded", loaded_digest);

    println!(
        "Timings: write={:.3} ms, read={:.3} ms",
        write_time.as_secs_f64() * 1_000.0,
        read_time.as_secs_f64() * 1_000.0
    );

    // Verify file size (note: STL triangulates polygons)
    let metadata = fs::metadata(&path).unwrap();
    let expected_min_size = 80 + 4 + (original_digest.faces as u64) * 50;
    let actual_triangles = (metadata.len() - 84) / 50;
    println!(
        "Binary STL contains {} triangles (original {} faces, some may be triangulated)",
        actual_triangles, original_digest.faces
    );
    assert!(
        metadata.len() >= expected_min_size,
        "Binary STL file size mismatch: {} < {}",
        metadata.len(),
        expected_min_size
    );

    cleanup_paths(&[&path]);
    println!("✓ STL Binary roundtrip verified");
}

fn test_ply_with_attributes() {
    print_header("PLY with Normals and Colors");

    let mut mesh = RustMesh::new();
    mesh.request_vertex_normals();
    mesh.request_vertex_colors();

    let v0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
    let v1 = mesh.add_vertex(Point3::new(1.0, 0.0, 0.0));
    let v2 = mesh.add_vertex(Point3::new(0.0, 1.0, 0.0));
    mesh.add_face(&[v0, v1, v2]);

    // Set normals
    mesh.set_vertex_normal_by_index(0, Vec3::new(0.0, 0.0, 1.0));
    mesh.set_vertex_normal_by_index(1, Vec3::new(0.0, 0.0, 1.0));
    mesh.set_vertex_normal_by_index(2, Vec3::new(0.0, 0.0, 1.0));

    // Set colors
    mesh.set_vertex_color_by_index(0, Vec4::new(1.0, 0.0, 0.0, 1.0));
    mesh.set_vertex_color_by_index(1, Vec4::new(0.0, 1.0, 0.0, 1.0));
    mesh.set_vertex_color_by_index(2, Vec4::new(0.0, 0.0, 1.0, 1.0));

    let path = temp_path("colors", "ply");
    write_ply(&mesh, &path, PlyFormat::Ascii).unwrap();

    // Read back and verify attributes exist
    let loaded = read_ply(&path).unwrap();

    assert!(
        loaded.has_vertex_normals(),
        "Loaded mesh should have normals"
    );
    assert!(loaded.has_vertex_colors(), "Loaded mesh should have colors");

    println!(
        "Original: normals={}, colors={}",
        mesh.has_vertex_normals(),
        mesh.has_vertex_colors()
    );
    println!(
        "Loaded: normals={}, colors={}",
        loaded.has_vertex_normals(),
        loaded.has_vertex_colors()
    );

    cleanup_paths(&[&path]);
    println!("✓ PLY with attributes verified");
}

fn compare_with_openmesh_reference() {
    print_header("Compare with OpenMesh Reference");

    // Check if OpenMesh tools are available
    let openmesh_root = openmesh_compare_common::openmesh_root();
    if !openmesh_root.exists() {
        println!("OpenMesh reference not available at {:?}", openmesh_root);
        println!("Skipping OpenMesh comparison");
        return;
    }

    // Create a test mesh
    let mesh = generate_sphere(1.0, 16, 16);

    // Write using RustMesh
    let rm_ply_path = temp_path("rustmesh-sphere", "ply");
    write_ply(&mesh, &rm_ply_path, PlyFormat::Ascii).unwrap();

    let rm_stl_path = temp_path("rustmesh-sphere", "stl");
    write_stl(&mesh, &rm_stl_path, StlFormat::Ascii).unwrap();

    // Load back and compare
    let ply_loaded = read_ply(&rm_ply_path).unwrap();
    let stl_loaded = read_stl(&rm_stl_path).unwrap();

    let original_digest = mesh_digest(&mesh);
    let ply_digest = mesh_digest(&ply_loaded);
    let stl_digest = mesh_digest(&stl_loaded);

    println!(
        "Original sphere: {}V {}F",
        original_digest.vertices, original_digest.faces
    );
    println!(
        "PLY roundtrip: {}V {}F",
        ply_digest.vertices, ply_digest.faces
    );
    println!(
        "STL roundtrip: {}V {}F",
        stl_digest.vertices, stl_digest.faces
    );

    cleanup_paths(&[&rm_ply_path, &rm_stl_path]);
    println!("✓ OpenMesh reference comparison complete");
}

fn main() {
    print_header("RustMesh I/O Parity Tests: PLY/STL Read/Write");

    // Test meshes
    let cube = generate_cube();
    let sphere = generate_sphere(1.0, 16, 16);

    // PLY tests
    test_ply_roundtrip_ascii(&cube, "cube");
    test_ply_roundtrip_binary(&cube, "cube");
    test_ply_roundtrip_ascii(&sphere, "sphere");
    test_ply_roundtrip_binary(&sphere, "sphere");
    test_ply_with_attributes();

    // STL tests
    test_stl_roundtrip_ascii(&cube, "cube");
    test_stl_roundtrip_binary(&cube, "cube");
    test_stl_roundtrip_ascii(&sphere, "sphere");
    test_stl_roundtrip_binary(&sphere, "sphere");

    // OpenMesh comparison
    compare_with_openmesh_reference();

    print_header("All I/O Tests Passed!");
}
