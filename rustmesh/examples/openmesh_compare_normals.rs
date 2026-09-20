mod openmesh_compare_common;

use openmesh_compare_common::{
    measure, mesh_digest, openmesh_root, print_duration_compare, print_header, print_mesh_digest,
};
use rustmesh::{generate_sphere, write_off, RustMesh, Vec3, VertexHandle};
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

const ITERATIONS: usize = 200;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FaceMode {
    TimedRefresh,
    PrecomputedOnce,
}

impl FaceMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::TimedRefresh => "timed_refresh",
            Self::PrecomputedOnce => "precomputed_once",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "timed_refresh" => Some(Self::TimedRefresh),
            "precomputed_once" => Some(Self::PrecomputedOnce),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VertexMode {
    None,
    AreaWeighted,
    FaceAverage,
}

impl VertexMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::AreaWeighted => "area_weighted",
            Self::FaceAverage => "face_average",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "none" => Some(Self::None),
            "area_weighted" => Some(Self::AreaWeighted),
            "face_average" => Some(Self::FaceAverage),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NormalContract {
    face_mode: FaceMode,
    vertex_mode: VertexMode,
}

#[derive(Debug, Clone, Copy)]
struct NormalBenchResult {
    contract: NormalContract,
    elapsed: Duration,
    face_checksum: f64,
    vertex_checksum: f64,
    faces: usize,
    vertices: usize,
}

fn main() {
    print_header("RustMesh Normal Update vs OpenMesh");
    println!("Reference sources:");
    println!("  - Mirror/OpenMesh-11.0.0/src/OpenMesh/Core/Mesh/PolyMeshT_impl.hh");
    println!("  - rustmesh/src/Core/connectivity.rs");
    println!(
        "Workload: same OFF input, {ITERATIONS} iterations each of update_face_normals / update_vertex_normals / update_normals"
    );
    println!(
        "Note: vertex-normal timing precomputes face normals once before the timed loop to match OpenMesh's documented contract."
    );
    println!(
        "Semantics: RustMesh default vertex normals are area-weighted; OpenMesh defaults to equal-weight face averaging."
    );
    println!(
        "Contract keys: face_mode={{timed_refresh|precomputed_once}} vertex_mode={{none|area_weighted|face_average}}"
    );

    let input = generate_sphere(1.0, 64, 64);
    print_mesh_digest("Input mesh", mesh_digest(&input));

    let input_path = PathBuf::from("/tmp").join("rustmesh-openmesh-normals-input.off");
    if let Err(err) = write_off(&input, &input_path) {
        eprintln!("failed to write temporary OFF input: {err}");
        return;
    }

    let rust_cases = run_rust_cases(&input);
    let openmesh_cases = run_openmesh_cases(&input_path).ok();

    for (name, rust_result) in &rust_cases {
        print_header(name);
        print_result("rustmesh", rust_result);

        match openmesh_cases.as_ref().and_then(|cases| cases.get(name)) {
            Some(openmesh_result) => {
                print_result("OpenMesh", openmesh_result);
                println!(
                    "Checksum deltas: face={:.6}, vertex={:.6}",
                    (rust_result.face_checksum - openmesh_result.face_checksum).abs(),
                    (rust_result.vertex_checksum - openmesh_result.vertex_checksum).abs()
                );
                print_duration_compare(name, rust_result.elapsed, Some(openmesh_result.elapsed));
            }
            None => {
                print_duration_compare(name, rust_result.elapsed, None);
            }
        }
    }

    if openmesh_cases.is_none() {
        println!("\nOpenMesh normal benchmark compilation failed in this environment.");
    }
}

fn print_result(label: &str, result: &NormalBenchResult) {
    println!(
        "{label}: V={} F={} face_checksum={:.6} vertex_checksum={:.6}",
        result.vertices, result.faces, result.face_checksum, result.vertex_checksum
    );
    println!(
        "  Contract: face_mode={} vertex_mode={}",
        result.contract.face_mode.as_str(),
        result.contract.vertex_mode.as_str()
    );
}

fn run_rust_cases(input: &RustMesh) -> BTreeMap<String, NormalBenchResult> {
    let mut cases = BTreeMap::new();
    cases.insert("update_face_normals".into(), bench_rust_face_normals(input));
    cases.insert(
        "update_vertex_normals".into(),
        bench_rust_vertex_normals(input),
    );
    cases.insert("update_normals".into(), bench_rust_full_normals(input));
    cases
}

fn bench_rust_face_normals(input: &RustMesh) -> NormalBenchResult {
    let mut mesh = input.clone();
    mesh.request_face_normals();

    let (elapsed, ()) = measure(|| {
        for _ in 0..ITERATIONS {
            update_face_normals(&mut mesh);
        }
    });

    NormalBenchResult {
        contract: NormalContract {
            face_mode: FaceMode::TimedRefresh,
            vertex_mode: VertexMode::None,
        },
        elapsed,
        face_checksum: face_normal_checksum(&mesh),
        vertex_checksum: 0.0,
        faces: mesh.n_faces(),
        vertices: mesh.n_vertices(),
    }
}

fn bench_rust_vertex_normals(input: &RustMesh) -> NormalBenchResult {
    let mut mesh = input.clone();
    mesh.request_face_normals();
    mesh.request_vertex_normals();
    update_face_normals(&mut mesh);

    let (elapsed, ()) = measure(|| {
        for _ in 0..ITERATIONS {
            update_vertex_normals_area_weighted(&mut mesh);
        }
    });

    NormalBenchResult {
        contract: NormalContract {
            face_mode: FaceMode::PrecomputedOnce,
            vertex_mode: VertexMode::AreaWeighted,
        },
        elapsed,
        face_checksum: face_normal_checksum(&mesh),
        vertex_checksum: vertex_normal_checksum(&mesh),
        faces: mesh.n_faces(),
        vertices: mesh.n_vertices(),
    }
}

fn bench_rust_full_normals(input: &RustMesh) -> NormalBenchResult {
    let mut mesh = input.clone();
    mesh.request_face_normals();
    mesh.request_vertex_normals();

    let (elapsed, ()) = measure(|| {
        for _ in 0..ITERATIONS {
            update_normals(&mut mesh);
        }
    });

    NormalBenchResult {
        contract: NormalContract {
            face_mode: FaceMode::TimedRefresh,
            vertex_mode: VertexMode::AreaWeighted,
        },
        elapsed,
        face_checksum: face_normal_checksum(&mesh),
        vertex_checksum: vertex_normal_checksum(&mesh),
        faces: mesh.n_faces(),
        vertices: mesh.n_vertices(),
    }
}

fn face_normal_checksum(mesh: &RustMesh) -> f64 {
    if !mesh.has_face_normals() {
        return 0.0;
    }

    mesh.faces()
        .filter_map(|fh| mesh.f_normal(fh))
        .map(normal_l1)
        .sum()
}

fn vertex_normal_checksum(mesh: &RustMesh) -> f64 {
    if !mesh.has_vertex_normals() {
        return 0.0;
    }

    (0..mesh.n_vertices())
        .filter_map(|idx| mesh.vertex_normal_by_index(idx))
        .map(normal_l1)
        .sum()
}

fn normal_l1(normal: Vec3) -> f64 {
    (normal.x.abs() + normal.y.abs() + normal.z.abs()) as f64
}

fn update_normals(mesh: &mut RustMesh) {
    update_face_normals(mesh);
    update_vertex_normals_area_weighted(mesh);
}

fn update_face_normals(mesh: &mut RustMesh) {
    if !mesh.has_face_normals() {
        mesh.request_face_normals();
    }
    let faces = mesh.faces().collect::<Vec<_>>();
    for fh in faces {
        if let Some(normal) = compute_face_normal(mesh, fh) {
            mesh.set_f_normal(fh, normal);
        }
    }
}

fn update_vertex_normals_area_weighted(mesh: &mut RustMesh) {
    if !mesh.has_vertex_normals() {
        mesh.request_vertex_normals();
    }
    let mut normal_sums = vec![Vec3::zeros(); mesh.n_vertices()];
    for fh in mesh.faces() {
        let verts = mesh.face_vertices_vec(fh);
        if verts.len() < 3 {
            continue;
        }
        let Some((normal, area)) = compute_face_normal_and_area(mesh, &verts) else {
            continue;
        };
        for vh in verts {
            normal_sums[vh.idx_usize()] += normal * area;
        }
    }
    for (idx, normal) in normal_sums.into_iter().enumerate() {
        if normal.norm_squared() > 0.0 {
            mesh.set_normal(VertexHandle::from_usize(idx), normal.normalize());
        }
    }
}

fn compute_face_normal(mesh: &RustMesh, fh: rustmesh::FaceHandle) -> Option<Vec3> {
    let verts = mesh.face_vertices_vec(fh);
    compute_face_normal_and_area(mesh, &verts).map(|(normal, _)| normal)
}

fn compute_face_normal_and_area(mesh: &RustMesh, verts: &[VertexHandle]) -> Option<(Vec3, f32)> {
    let p0 = mesh.point(*verts.first()?)?;
    let mut area_sum = 0.0f32;
    let mut normal_sum = Vec3::zeros();
    for tri in 1..verts.len().saturating_sub(1) {
        let p1 = mesh.point(verts[tri])?;
        let p2 = mesh.point(verts[tri + 1])?;
        let area_normal = (p1 - p0).cross(&(p2 - p0));
        area_sum += area_normal.norm() * 0.5;
        normal_sum += area_normal;
    }
    if normal_sum.norm_squared() > 0.0 {
        Some((normal_sum.normalize(), area_sum))
    } else {
        None
    }
}

fn run_openmesh_cases(input_path: &PathBuf) -> io::Result<BTreeMap<String, NormalBenchResult>> {
    let cpp_path = PathBuf::from("/tmp").join("rustmesh-openmesh-normal-bench.cpp");
    let bin_path = PathBuf::from("/tmp").join("rustmesh-openmesh-normal-bench");
    fs::write(&cpp_path, openmesh_cpp_source())?;

    let openmesh_root = openmesh_root();
    let include_src = openmesh_root.join("src");
    let include_build = openmesh_root.join("build/src");
    let lib_dir = openmesh_root.join("build/Build/lib");

    let compile_output = Command::new("c++")
        .arg("-O3")
        .arg("-std=c++17")
        .arg("-I")
        .arg(&include_src)
        .arg("-I")
        .arg(&include_build)
        .arg(&cpp_path)
        .arg("-L")
        .arg(&lib_dir)
        .arg("-lOpenMeshTools")
        .arg("-lOpenMeshCore")
        .arg("-o")
        .arg(&bin_path)
        .output()?;
    if !compile_output.status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!(
                "failed to compile OpenMesh normal benchmark:\n{}\n{}",
                String::from_utf8_lossy(&compile_output.stdout),
                String::from_utf8_lossy(&compile_output.stderr)
            ),
        ));
    }

    let output = Command::new(&bin_path)
        .arg(input_path)
        .arg(ITERATIONS.to_string())
        .env("DYLD_LIBRARY_PATH", &lib_dir)
        .output()?;
    if !output.status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }

    parse_openmesh_cases(&String::from_utf8_lossy(&output.stdout))
}

fn parse_openmesh_cases(stdout: &str) -> io::Result<BTreeMap<String, NormalBenchResult>> {
    let mut cases = BTreeMap::new();

    for line in stdout.lines() {
        let Some(rest) = line.strip_prefix("CASE ") else {
            continue;
        };
        let (name, result) = parse_openmesh_case(rest)?;
        cases.insert(name, result);
    }

    if cases.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "failed to parse OpenMesh normal benchmark output",
        ));
    }

    Ok(cases)
}

fn parse_kv_fields(line: &str) -> Vec<(&str, &str)> {
    line.split_whitespace()
        .filter_map(|field| field.split_once('='))
        .collect()
}

fn parse_openmesh_case(rest: &str) -> io::Result<(String, NormalBenchResult)> {
    let mut name = None;
    let mut elapsed_ns = None;
    let mut face_checksum = 0.0;
    let mut vertex_checksum = 0.0;
    let mut faces = None;
    let mut vertices = None;
    let mut face_mode = None;
    let mut vertex_mode = None;

    for (key, value) in parse_kv_fields(rest) {
        match key {
            "name" => name = Some(value.to_string()),
            "elapsed_ns" => elapsed_ns = value.parse::<f64>().ok(),
            "face_checksum" => face_checksum = value.parse::<f64>().unwrap_or(0.0),
            "vertex_checksum" => vertex_checksum = value.parse::<f64>().unwrap_or(0.0),
            "faces" => faces = value.parse::<usize>().ok(),
            "vertices" => vertices = value.parse::<usize>().ok(),
            "face_mode" => face_mode = FaceMode::parse(value),
            "vertex_mode" => vertex_mode = VertexMode::parse(value),
            _ => {}
        }
    }

    let name = name.ok_or_else(|| invalid_case_output(rest, "missing name"))?;
    let elapsed_ns = elapsed_ns.ok_or_else(|| invalid_case_output(rest, "missing elapsed_ns"))?;
    let faces = faces.ok_or_else(|| invalid_case_output(rest, "missing faces"))?;
    let vertices = vertices.ok_or_else(|| invalid_case_output(rest, "missing vertices"))?;
    let face_mode = face_mode.ok_or_else(|| invalid_case_output(rest, "missing face_mode"))?;
    let vertex_mode =
        vertex_mode.ok_or_else(|| invalid_case_output(rest, "missing vertex_mode"))?;

    Ok((
        name,
        NormalBenchResult {
            contract: NormalContract {
                face_mode,
                vertex_mode,
            },
            elapsed: Duration::from_secs_f64(elapsed_ns / 1_000_000_000.0),
            face_checksum,
            vertex_checksum,
            faces,
            vertices,
        },
    ))
}

fn invalid_case_output(rest: &str, detail: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::Other,
        format!("invalid OpenMesh normal benchmark output ({detail}): {rest}"),
    )
}

fn openmesh_cpp_source() -> &'static str {
    r#"#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

struct Traits : public OpenMesh::DefaultTraits {
  VertexAttributes(OpenMesh::Attributes::Normal);
  FaceAttributes(OpenMesh::Attributes::Normal);
};

using Mesh = OpenMesh::TriMesh_ArrayKernelT<Traits>;

double face_checksum(const Mesh& mesh) {
  double sum = 0.0;
  for (auto f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
    const auto& normal = mesh.normal(*f_it);
    sum += std::abs(normal[0]) + std::abs(normal[1]) + std::abs(normal[2]);
  }
  return sum;
}

double vertex_checksum(const Mesh& mesh) {
  double sum = 0.0;
  for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
    const auto& normal = mesh.normal(*v_it);
    sum += std::abs(normal[0]) + std::abs(normal[1]) + std::abs(normal[2]);
  }
  return sum;
}

template <typename Fn>
double measure_ns(Fn&& fn, std::size_t iterations) {
  const auto start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < iterations; ++i) {
    fn();
  }
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::nano>(end - start).count();
}

void print_case(
    const char* name,
    const Mesh& mesh,
    double elapsed_ns,
    bool include_face,
    bool include_vertex,
    const char* face_mode,
    const char* vertex_mode) {
  std::cout << "CASE"
            << " name=" << name
            << " elapsed_ns=" << elapsed_ns
            << " faces=" << mesh.n_faces()
            << " vertices=" << mesh.n_vertices()
            << " face_checksum=" << (include_face ? face_checksum(mesh) : 0.0)
            << " vertex_checksum=" << (include_vertex ? vertex_checksum(mesh) : 0.0)
            << " face_mode=" << face_mode
            << " vertex_mode=" << vertex_mode
            << '\n';
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: normal_bench <input.off> <iterations>\n";
    return 1;
  }

  const std::string input_path = argv[1];
  const auto iterations = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));

  Mesh base;
  if (!OpenMesh::IO::read_mesh(base, input_path)) {
    std::cerr << "failed to read mesh: " << input_path << '\n';
    return 1;
  }

  std::cout << std::fixed << std::setprecision(6);

  {
    Mesh mesh = base;
    mesh.request_face_normals();
    const double elapsed_ns = measure_ns([&mesh]() { mesh.update_face_normals(); }, iterations);
    print_case("update_face_normals", mesh, elapsed_ns, true, false, "timed_refresh", "none");
  }

  {
    Mesh mesh = base;
    mesh.request_face_normals();
    mesh.request_vertex_normals();
    mesh.update_face_normals();
    const double elapsed_ns = measure_ns([&mesh]() { mesh.update_vertex_normals(); }, iterations);
    print_case(
        "update_vertex_normals",
        mesh,
        elapsed_ns,
        true,
        true,
        "precomputed_once",
        "face_average");
  }

  {
    Mesh mesh = base;
    mesh.request_face_normals();
    mesh.request_vertex_normals();
    const double elapsed_ns = measure_ns([&mesh]() { mesh.update_normals(); }, iterations);
    print_case("update_normals", mesh, elapsed_ns, true, true, "timed_refresh", "face_average");
  }

  return 0;
}
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_openmesh_cases_requires_explicit_contract_fields() {
        let output = "CASE name=update_normals elapsed_ns=100.0 faces=10 vertices=8 face_checksum=1.0 vertex_checksum=2.0";
        let error = parse_openmesh_cases(output).unwrap_err();
        assert!(error.to_string().contains("face_mode"));
    }

    #[test]
    fn parse_openmesh_cases_preserves_contract_metadata() {
        let output = concat!(
            "CASE name=update_vertex_normals elapsed_ns=100.0 faces=10 vertices=8 ",
            "face_checksum=1.0 vertex_checksum=2.0 face_mode=precomputed_once ",
            "vertex_mode=face_average\n",
        );

        let cases = parse_openmesh_cases(output).unwrap();
        let result = cases.get("update_vertex_normals").unwrap();

        assert_eq!(result.contract.face_mode, FaceMode::PrecomputedOnce);
        assert_eq!(result.contract.vertex_mode, VertexMode::FaceAverage);
        assert_eq!(result.faces, 10);
        assert_eq!(result.vertices, 8);
    }
}
