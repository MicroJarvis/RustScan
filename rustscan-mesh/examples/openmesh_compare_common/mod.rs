use rustscan_mesh::{read_obj, read_off, write_off, FaceHandle, Point3, RustMesh, Vec3};
use std::fs;
use std::hint::black_box;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
pub struct MeshDigest {
    pub vertices: usize,
    pub faces: usize,
    pub bbox_min: Point3,
    pub bbox_max: Point3,
    pub centroid: Point3,
    pub checksum_l1: f32,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct OpenMeshTriBenchmark {
    pub tetrahedron_us: Option<f64>,
    pub vertex_traversal_ns: Option<f64>,
    pub face_traversal_ns: Option<f64>,
    pub add_triangles_us: Option<f64>,
    pub triangle_area_us: Option<f64>,
}

pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("RustMesh crate should live under repo root")
        .to_path_buf()
}

pub fn reference_root() -> PathBuf {
    for ancestor in repo_root().ancestors() {
        if ancestor.join("Mirror/OpenMesh-11.0.0").exists() {
            return ancestor.to_path_buf();
        }
    }
    repo_root()
}

pub fn openmesh_root() -> PathBuf {
    reference_root().join("Mirror/OpenMesh-11.0.0")
}

pub fn openmesh_benchmark_binary() -> PathBuf {
    openmesh_root().join("build/OpenMeshBenchmark")
}

pub fn openmesh_tool(binary_name: &str) -> Command {
    let mut command = Command::new(openmesh_root().join(format!("build/Build/bin/{binary_name}")));
    command.env("DYLD_LIBRARY_PATH", openmesh_root().join("build/Build/lib"));
    command.current_dir(reference_root());
    command
}

pub fn run_capture(command: &mut Command) -> io::Result<String> {
    let output = command.output()?;
    let mut combined = String::new();
    combined.push_str(&String::from_utf8_lossy(&output.stdout));
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    if output.status.success() {
        Ok(combined)
    } else {
        Err(io::Error::new(
            io::ErrorKind::Other,
            format!("command failed: {combined}"),
        ))
    }
}

pub fn print_header(title: &str) {
    println!("\n============================================================");
    println!("{title}");
    println!("============================================================");
}

pub fn print_mesh_digest(label: &str, digest: MeshDigest) {
    println!(
        "{label}: V={} F={} bbox=({:.4},{:.4},{:.4})..({:.4},{:.4},{:.4}) centroid=({:.4},{:.4},{:.4}) checksum_l1={:.4}",
        digest.vertices,
        digest.faces,
        digest.bbox_min.x,
        digest.bbox_min.y,
        digest.bbox_min.z,
        digest.bbox_max.x,
        digest.bbox_max.y,
        digest.bbox_max.z,
        digest.centroid.x,
        digest.centroid.y,
        digest.centroid.z,
        digest.checksum_l1,
    );
}

pub fn print_duration_compare(label: &str, rust_time: Duration, openmesh_time: Option<Duration>) {
    let rust_secs = rust_time.as_secs_f64();
    match openmesh_time {
        Some(openmesh_time) => {
            let openmesh_secs = openmesh_time.as_secs_f64();
            let ratio = if rust_secs > 0.0 {
                openmesh_secs / rust_secs
            } else {
                0.0
            };
            println!(
                "{label}: RustMesh={}, OpenMesh={}, OpenMesh/rustmesh={ratio:.2}x",
                format_duration(rust_time),
                format_duration(openmesh_time),
            );
        }
        None => println!("{label}: RustMesh={}", format_duration(rust_time)),
    }
}

pub fn format_duration(duration: Duration) -> String {
    let nanos = duration.as_nanos();
    if nanos < 1_000 {
        format!("{nanos} ns")
    } else if nanos < 1_000_000 {
        format!("{:.3} us", nanos as f64 / 1_000.0)
    } else {
        format!("{:.3} ms", nanos as f64 / 1_000_000.0)
    }
}

pub fn active_faces(mesh: &RustMesh) -> usize {
    mesh.faces()
        .filter(|&fh| mesh.face_halfedge_handle(fh).is_some())
        .count()
}

pub fn active_vertices(mesh: &RustMesh) -> usize {
    let mut used = vec![false; mesh.n_vertices()];

    for face_idx in 0..mesh.n_faces() {
        for vh in mesh.face_vertices_vec(FaceHandle::new(face_idx as u32)) {
            let idx = vh.idx_usize();
            if idx < used.len() {
                used[idx] = true;
            }
        }
    }

    let active = used.into_iter().filter(|used| *used).count();
    if active == 0 {
        mesh.vertices()
            .filter(|&vh| mesh.halfedge_handle(vh).is_some())
            .count()
    } else {
        active
    }
}

pub fn mesh_digest(mesh: &RustMesh) -> MeshDigest {
    let mut min = Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max = Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    let mut sum = Vec3::zeros();
    let mut count = 0usize;
    let mut checksum = 0.0f32;
    let mut used = vec![false; mesh.n_vertices()];

    for face_idx in 0..mesh.n_faces() {
        for vh in mesh.face_vertices_vec(FaceHandle::new(face_idx as u32)) {
            let idx = vh.idx_usize();
            if idx < used.len() {
                used[idx] = true;
            }
        }
    }

    let has_face_usage = used.iter().any(|used| *used);
    for vh in mesh.vertices() {
        let idx = vh.idx_usize();
        let is_active = if has_face_usage {
            used.get(idx).copied().unwrap_or(false)
        } else {
            mesh.halfedge_handle(vh).is_some()
        };

        if !is_active {
            continue;
        }

        if let Some(point) = mesh.point(vh) {
            min = Point3::from(min.coords.inf(&point.coords));
            max = Point3::from(max.coords.sup(&point.coords));
            sum += point.coords;
            checksum += point.x.abs() + point.y.abs() + point.z.abs();
            count += 1;
        }
    }

    if count == 0 {
        return MeshDigest {
            vertices: 0,
            faces: 0,
            bbox_min: Point3::origin(),
            bbox_max: Point3::origin(),
            centroid: Point3::origin(),
            checksum_l1: 0.0,
        };
    }

    MeshDigest {
        vertices: count,
        faces: active_faces(mesh),
        bbox_min: min,
        bbox_max: max,
        centroid: Point3::from(sum / count as f32),
        checksum_l1: checksum,
    }
}

pub fn off_digest(path: &Path) -> io::Result<MeshDigest> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let header = next_data_line(&mut lines)?;
    if header != "OFF" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported OFF header: {header}"),
        ));
    }

    let counts_line = next_data_line(&mut lines)?;
    let mut counts = counts_line.split_whitespace();
    let vertex_count = parse_usize(counts.next(), "vertex count")?;
    let face_count = parse_usize(counts.next(), "face count")?;

    let mut vertices = Vec::with_capacity(vertex_count);
    for _ in 0..vertex_count {
        let line = next_data_line(&mut lines)?;
        let mut parts = line.split_whitespace();
        let x = parse_f32(parts.next(), "vertex x")?;
        let y = parse_f32(parts.next(), "vertex y")?;
        let z = parse_f32(parts.next(), "vertex z")?;
        vertices.push(Point3::new(x, y, z));
    }

    let mut used = vec![false; vertex_count];
    for _ in 0..face_count {
        let line = next_data_line(&mut lines)?;
        let mut parts = line.split_whitespace();
        let n = parse_usize(parts.next(), "face vertex count")?;
        for _ in 0..n {
            let idx = parse_usize(parts.next(), "face index")?;
            if idx < used.len() {
                used[idx] = true;
            }
        }
    }

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
            faces: face_count,
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

fn next_data_line(lines: &mut impl Iterator<Item = io::Result<String>>) -> io::Result<String> {
    for line in lines {
        let line = line?;
        let content = line.split('#').next().unwrap_or("").trim();
        if !content.is_empty() {
            return Ok(content.to_string());
        }
    }

    Err(io::Error::new(
        io::ErrorKind::UnexpectedEof,
        "unexpected end of OFF file",
    ))
}

fn parse_usize(token: Option<&str>, label: &str) -> io::Result<usize> {
    token
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("missing {label}")))?
        .parse()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, format!("invalid {label}")))
}

fn parse_f32(token: Option<&str>, label: &str) -> io::Result<f32> {
    token
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("missing {label}")))?
        .parse()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, format!("invalid {label}")))
}

pub fn load_mesh(path: &Path) -> io::Result<RustMesh> {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("obj") => read_obj(path),
        Some("off") => read_off(path),
        Some(other) => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unsupported mesh extension: {other}"),
        )),
        None => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "mesh path is missing an extension",
        )),
    }
}

pub fn write_temp_off(mesh: &RustMesh, stem: &str) -> io::Result<PathBuf> {
    let path = PathBuf::from("/tmp").join(format!(
        "rustmesh-openmesh-compare-{stem}-{}.off",
        std::process::id()
    ));
    write_off(mesh, &path)?;
    Ok(path)
}

pub fn cleanup_paths(paths: &[&Path]) {
    for path in paths {
        fs::remove_file(path).ok();
    }
}

pub fn measure<T>(mut f: impl FnMut() -> T) -> (Duration, T) {
    let start = Instant::now();
    let result = f();
    (start.elapsed(), result)
}

pub fn bench_ns_per_iter(iterations: usize, mut f: impl FnMut()) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed().as_nanos() as f64;
    elapsed / iterations as f64
}

pub fn black_box_value<T>(value: T) -> T {
    black_box(value)
}

pub fn parse_openmesh_tri_benchmark(output: &str) -> OpenMeshTriBenchmark {
    let mut parsed = OpenMeshTriBenchmark::default();

    for line in output.lines() {
        if line.contains("OpenMesh 四面体") {
            parsed.tetrahedron_us = parse_number_after_colon(line);
        } else if line.contains("顶点遍历") {
            parsed.vertex_traversal_ns = parse_number_after_colon(line);
        } else if line.contains("面遍历") {
            parsed.face_traversal_ns = parse_number_after_colon(line);
        } else if line.contains("OpenMesh 添加 1000 个三角形") {
            parsed.add_triangles_us = parse_number_after_colon(line);
        } else if line.contains("OpenMesh 三角形面积") {
            parsed.triangle_area_us = parse_number_after_colon(line);
        }
    }

    parsed
}

pub fn parse_number_after_colon(line: &str) -> Option<f64> {
    let numeric = line.split(':').nth(1)?.trim();
    let token = numeric.split_whitespace().next()?;
    token.parse::<f64>().ok()
}
