use rustmesh::{RustMesh, Vec3, Vec4};
mod openmesh_compare_common;

use openmesh_compare_common::{openmesh_root, print_header};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

const ITERATIONS: usize = 2_000_000;

fn main() {
    print_header("RustMesh vs OpenMesh VectorT Cases");
    println!("Reference sources:");
    println!("  - Mirror/OpenMesh-11.0.0/src/Benchmark/VectorT.cpp");
    println!("  - Mirror/OpenMesh-11.0.0/src/Benchmark/VectorT_new.cpp");
    println!("This example benchmarks Rust `nalgebra` against OpenMesh `VectorT` using the same case shapes.");

    let rust_cases = run_rust_cases();
    let openmesh_cases = run_openmesh_cases().ok();

    println!("\nCase results (ns/iter):");
    for (name, rust_ns) in &rust_cases {
        match openmesh_cases
            .as_ref()
            .and_then(|cases| cases.get(name))
            .copied()
        {
            Some(openmesh_ns) => {
                println!(
                    "{name}: rustmesh/nalgebra={rust_ns:.3}, OpenMesh={openmesh_ns:.3}, OpenMesh/rustmesh={:.2}x",
                    openmesh_ns / rust_ns
                );
            }
            None => println!("{name}: rustmesh/nalgebra={rust_ns:.3}"),
        }
    }

    if openmesh_cases.is_none() {
        println!("\nOpenMesh VectorT benchmark compilation failed in this environment.");
    }
}

fn run_rust_cases() -> BTreeMap<String, f64> {
    let mut cases = BTreeMap::new();

    cases.insert("Vec3f_add_compare".into(), bench_vec3_add_compare());
    cases.insert("Vec3f_cross_product".into(), bench_vec3_cross_product());
    cases.insert("Vec3f_scalar_product".into(), bench_vec3_scalar_product());
    cases.insert("Vec3f_norm".into(), bench_vec3_norm());
    cases.insert("Vec3f_times_scalar".into(), bench_vec3_times_scalar());

    cases.insert("Vec4f_add_compare".into(), bench_vec4_add_compare());
    cases.insert(
        "Vec4f_add_compare_glam_eq".into(),
        bench_vec4_add_compare_glam_eq(),
    );
    cases.insert("Vec4f_scalar_product".into(), bench_vec4_scalar_product());
    cases.insert("Vec4f_norm".into(), bench_vec4_norm());
    cases.insert("Vec4f_times_scalar".into(), bench_vec4_times_scalar());

    cases
}

fn run_openmesh_cases() -> std::io::Result<BTreeMap<String, f64>> {
    let cpp_path = PathBuf::from("/tmp").join("rustmesh-openmesh-vector-bench.cpp");
    let bin_path = PathBuf::from("/tmp").join("rustmesh-openmesh-vector-bench");
    fs::write(&cpp_path, openmesh_cpp_source())?;

    let openmesh_root = openmesh_root();
    let status = Command::new("c++")
        .arg("-O3")
        .arg("-std=c++17")
        .arg("-I")
        .arg(openmesh_root.join("src"))
        .arg("-I")
        .arg(openmesh_root.join("build/src"))
        .arg(&cpp_path)
        .arg("-o")
        .arg(&bin_path)
        .status()?;
    if !status.success() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "failed to compile OpenMesh vector benchmark",
        ));
    }

    let output = Command::new(&bin_path).output()?;
    if !output.status.success() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut cases = BTreeMap::new();
    for line in stdout.lines() {
        if let Some((name, value)) = line.split_once(':') {
            if let Ok(value) = value.trim().parse::<f64>() {
                cases.insert(name.trim().to_string(), value);
            }
        }
    }
    Ok(cases)
}

fn bench_vec3_add_compare() -> f64 {
    let mut v1 = Vec3::zeros();
    let mut v2 = Vec3::repeat(1000.0);
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        v1 += Vec3::new(1.1, 1.2, 1.3);
        v2 -= Vec3::new(1.1, 1.2, 1.3);
        if v1 == v2 {
            v1 -= v2;
            v2 += v1;
        }
    }
    std::hint::black_box(v1.norm() + v2.norm());
    start.elapsed().as_nanos() as f64 / ITERATIONS as f64
}

fn bench_vec3_cross_product() -> f64 {
    let mut v1 = Vec3::zeros();
    let mut v2 = Vec3::repeat(1000.0);
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        v1 += Vec3::new(1.1, 1.2, 1.3);
        v2 -= Vec3::new(1.1, 1.2, 1.3);
        v1 = v1.cross(&v2);
    }
    std::hint::black_box(v1.norm() + v2.norm());
    start.elapsed().as_nanos() as f64 / ITERATIONS as f64
}

fn bench_vec3_scalar_product() -> f64 {
    let mut v1 = Vec3::zeros();
    let mut v2 = Vec3::repeat(1000.0);
    let mut acc = 0.0f32;
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        v1 += Vec3::new(1.1, 1.2, 1.3);
        v2 -= Vec3::new(1.1, 1.2, 1.3);
        acc += v1.dot(&v2);
    }
    std::hint::black_box(acc);
    start.elapsed().as_nanos() as f64 / ITERATIONS as f64
}

fn bench_vec3_norm() -> f64 {
    let mut v1 = Vec3::zeros();
    let mut acc = 0.0f32;
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        v1 += Vec3::new(1.1, 1.2, 1.3);
        acc += v1.norm();
    }
    std::hint::black_box(acc);
    start.elapsed().as_nanos() as f64 / ITERATIONS as f64
}

fn bench_vec3_times_scalar() -> f64 {
    let mut v1 = Vec3::repeat(1.0);
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        v1 += Vec3::new(1.1, 1.2, 1.3);
        v1 *= 1.0 / v1.x;
        v1 *= v1.y;
    }
    std::hint::black_box(v1.norm());
    start.elapsed().as_nanos() as f64 / ITERATIONS as f64
}

fn bench_vec4_add_compare() -> f64 {
    let mut v1 = Vec4::zeros();
    let mut v2 = Vec4::repeat(1000.0);
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        v1 += Vec4::new(1.1, 1.2, 1.3, 1.4);
        v2 -= Vec4::new(1.1, 1.2, 1.3, 1.4);
        if vec4_eq_components(v1, v2) {
            v1 -= v2;
            v2 += v1;
        }
    }
    std::hint::black_box(v1.norm() + v2.norm());
    start.elapsed().as_nanos() as f64 / ITERATIONS as f64
}

fn bench_vec4_add_compare_glam_eq() -> f64 {
    let mut v1 = Vec4::zeros();
    let mut v2 = Vec4::repeat(1000.0);
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        v1 += Vec4::new(1.1, 1.2, 1.3, 1.4);
        v2 -= Vec4::new(1.1, 1.2, 1.3, 1.4);
        if v1 == v2 {
            v1 -= v2;
            v2 += v1;
        }
    }
    std::hint::black_box(v1.norm() + v2.norm());
    start.elapsed().as_nanos() as f64 / ITERATIONS as f64
}

#[inline(always)]
fn vec4_eq_components(lhs: Vec4, rhs: Vec4) -> bool {
    lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w
}

fn bench_vec4_scalar_product() -> f64 {
    let mut v1 = Vec4::zeros();
    let mut v2 = Vec4::repeat(1000.0);
    let mut acc = 0.0f32;
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        v1 += Vec4::new(1.1, 1.2, 1.3, 1.4);
        v2 -= Vec4::new(1.1, 1.2, 1.3, 1.4);
        acc += v1.dot(&v2);
    }
    std::hint::black_box(acc);
    start.elapsed().as_nanos() as f64 / ITERATIONS as f64
}

fn bench_vec4_norm() -> f64 {
    let mut v1 = Vec4::zeros();
    let mut acc = 0.0f32;
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        v1 += Vec4::new(1.1, 1.2, 1.3, 1.4);
        acc += v1.norm();
    }
    std::hint::black_box(acc);
    start.elapsed().as_nanos() as f64 / ITERATIONS as f64
}

fn bench_vec4_times_scalar() -> f64 {
    let mut v1 = Vec4::repeat(1.0);
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        v1 += Vec4::new(1.1, 1.2, 1.3, 1.4);
        v1 *= 1.0 / v1.x;
        v1 *= v1.y;
    }
    std::hint::black_box(v1.norm());
    start.elapsed().as_nanos() as f64 / ITERATIONS as f64
}

fn openmesh_cpp_source() -> &'static str {
    r#"
#include <chrono>
#include <iostream>
#include <OpenMesh/Core/Geometry/VectorT.hh>

template<class Vec>
static inline Vec testVec();

template<>
inline OpenMesh::Vec3f testVec<OpenMesh::Vec3f>() {
    return OpenMesh::Vec3f(1.1f, 1.2f, 1.3f);
}

template<>
inline OpenMesh::Vec4f testVec<OpenMesh::Vec4f>() {
    return OpenMesh::Vec4f(1.1f, 1.2f, 1.3f, 1.4f);
}

template<class Fn>
double bench(Fn&& fn) {
    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 2000000.0;
}

int main() {
    std::cout << "Vec3f_add_compare:" << bench([]{
        OpenMesh::Vec3f v1(0.0f), v2(1000.0f);
        for (size_t i = 0; i < 2000000; ++i) {
            v1 += testVec<OpenMesh::Vec3f>();
            v2 -= testVec<OpenMesh::Vec3f>();
            if (v1 == v2) { v1 -= v2; v2 += v1; }
        }
        volatile float dummy = v1.norm() + v2.norm();
        (void)dummy;
    }) << "\n";
    std::cout << "Vec3f_cross_product:" << bench([]{
        OpenMesh::Vec3f v1(0.0f), v2(1000.0f);
        for (size_t i = 0; i < 2000000; ++i) {
            v1 += testVec<OpenMesh::Vec3f>();
            v2 -= testVec<OpenMesh::Vec3f>();
            v1 = v1 % v2;
        }
        volatile float dummy = v1.norm() + v2.norm();
        (void)dummy;
    }) << "\n";
    std::cout << "Vec3f_scalar_product:" << bench([]{
        OpenMesh::Vec3f v1(0.0f), v2(1000.0f);
        volatile float acc = 0.0f;
        for (size_t i = 0; i < 2000000; ++i) {
            v1 += testVec<OpenMesh::Vec3f>();
            v2 -= testVec<OpenMesh::Vec3f>();
            acc += (v1 | v2);
        }
        (void)acc;
    }) << "\n";
    std::cout << "Vec3f_norm:" << bench([]{
        OpenMesh::Vec3f v1(0.0f);
        volatile float acc = 0.0f;
        for (size_t i = 0; i < 2000000; ++i) {
            v1 += testVec<OpenMesh::Vec3f>();
            acc += v1.norm();
        }
        (void)acc;
    }) << "\n";
    std::cout << "Vec3f_times_scalar:" << bench([]{
        OpenMesh::Vec3f v1(1.0f);
        for (size_t i = 0; i < 2000000; ++i) {
            v1 += testVec<OpenMesh::Vec3f>();
            v1 *= 1.0f / v1[0];
            v1 *= v1[1];
        }
        volatile float dummy = v1.norm();
        (void)dummy;
    }) << "\n";

    std::cout << "Vec4f_add_compare:" << bench([]{
        OpenMesh::Vec4f v1(0.0f), v2(1000.0f);
        for (size_t i = 0; i < 2000000; ++i) {
            v1 += testVec<OpenMesh::Vec4f>();
            v2 -= testVec<OpenMesh::Vec4f>();
            if (v1 == v2) { v1 -= v2; v2 += v1; }
        }
        volatile float dummy = v1.norm() + v2.norm();
        (void)dummy;
    }) << "\n";
    std::cout << "Vec4f_scalar_product:" << bench([]{
        OpenMesh::Vec4f v1(0.0f), v2(1000.0f);
        volatile float acc = 0.0f;
        for (size_t i = 0; i < 2000000; ++i) {
            v1 += testVec<OpenMesh::Vec4f>();
            v2 -= testVec<OpenMesh::Vec4f>();
            acc += (v1 | v2);
        }
        (void)acc;
    }) << "\n";
    std::cout << "Vec4f_norm:" << bench([]{
        OpenMesh::Vec4f v1(0.0f);
        volatile float acc = 0.0f;
        for (size_t i = 0; i < 2000000; ++i) {
            v1 += testVec<OpenMesh::Vec4f>();
            acc += v1.norm();
        }
        (void)acc;
    }) << "\n";
    std::cout << "Vec4f_times_scalar:" << bench([]{
        OpenMesh::Vec4f v1(1.0f);
        for (size_t i = 0; i < 2000000; ++i) {
            v1 += testVec<OpenMesh::Vec4f>();
            v1 *= 1.0f / v1[0];
            v1 *= v1[1];
        }
        volatile float dummy = v1.norm();
        (void)dummy;
    }) << "\n";
    return 0;
}
"#
}
