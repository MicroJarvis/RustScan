use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=POSELIB_ROOT");
    println!("cargo:rerun-if-env-changed=EIGEN3_INCLUDE_DIR");

    if env::var_os("CARGO_FEATURE_POSELIB").is_none() {
        return;
    }

    let poselib_root = env::var_os("POSELIB_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("third_party/PoseLib"));
    if !poselib_root
        .join("PoseLib/solvers/gen_relpose_6pt.cc")
        .exists()
    {
        panic!(
            "RustSFM poselib feature requires PoseLib v2.0.5 source. Set POSELIB_ROOT=/path/to/PoseLib-2.0.5 or place it at third_party/PoseLib"
        );
    }

    let eigen_include = eigen_include_dir().unwrap_or_else(|| {
        panic!(
            "RustSFM poselib feature requires Eigen3 headers. Set EIGEN3_INCLUDE_DIR or install eigen3 so pkg-config can find it"
        )
    });

    println!("cargo:rerun-if-changed=src/poselib_bridge.cpp");
    println!(
        "cargo:rerun-if-changed={}",
        poselib_root
            .join("PoseLib/solvers/gen_relpose_6pt.cc")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        poselib_root.join("PoseLib/solvers/gp3p.cc").display()
    );

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .warnings(false)
        .include(&poselib_root)
        .include(&eigen_include)
        .file("src/poselib_bridge.cpp")
        .file(poselib_root.join("PoseLib/solvers/gen_relpose_6pt.cc"))
        .file(poselib_root.join("PoseLib/solvers/gp3p.cc"))
        .file(poselib_root.join("PoseLib/solvers/p3p.cc"))
        .file(poselib_root.join("PoseLib/misc/re3q3.cc"))
        .file(poselib_root.join("PoseLib/misc/univariate.cc"))
        .file(poselib_root.join("PoseLib/misc/essential.cc"));

    if cfg!(target_os = "macos") {
        build.flag_if_supported("-stdlib=libc++");
    }

    build.compile("rustsfm_poselib_bridge");
}

fn eigen_include_dir() -> Option<PathBuf> {
    if let Some(path) = env::var_os("EIGEN3_INCLUDE_DIR").map(PathBuf::from) {
        if path.join("Eigen/Core").exists() {
            return Some(path);
        }
    }

    pkg_config_include_dir("eigen3").or_else(|| {
        [
            "/opt/homebrew/include/eigen3",
            "/usr/local/include/eigen3",
            "/usr/include/eigen3",
        ]
        .iter()
        .map(Path::new)
        .find(|path| path.join("Eigen/Core").exists())
        .map(Path::to_path_buf)
    })
}

fn pkg_config_include_dir(package: &str) -> Option<PathBuf> {
    let output = std::process::Command::new("pkg-config")
        .args(["--cflags-only-I", package])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    stdout
        .split_whitespace()
        .filter_map(|flag| flag.strip_prefix("-I"))
        .map(PathBuf::from)
        .find(|path| path.join("Eigen/Core").exists())
}
