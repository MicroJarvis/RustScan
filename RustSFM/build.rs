use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rustc-check-cfg=cfg(colmap_freeimage)");
    println!("cargo:rustc-check-cfg=cfg(colmap_eigen)");
    println!("cargo:rerun-if-env-changed=POSELIB_ROOT");
    println!("cargo:rerun-if-env-changed=EIGEN3_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=VLFEAT_ROOT");

    if env::var_os("CARGO_FEATURE_POSELIB").is_some() {
        build_poselib_bridge();
    }

    if env::var_os("CARGO_FEATURE_VLFEAT_SIFT").is_some() {
        build_vlfeat_sift();
    }

    build_colmap_eigen();
    build_colmap_image();
}

fn build_vlfeat_sift() {
    let vlfeat_root = resolve_vlfeat_root().unwrap_or_else(|| {
        panic!(
            "RustSFM vlfeat-sift feature requires VLFeat source. Run scripts/setup_vlfeat.sh, set VLFEAT_ROOT=/path/to/VLFeat, or place it at third_party/vlfeat"
        )
    });

    println!("cargo:rerun-if-changed=src/native/vlfeat_sift.c");
    println!("cargo:rerun-if-changed=src/native/vlfeat_sift.h");
    println!(
        "cargo:rerun-if-changed={}",
        vlfeat_root.join("sift.c").display()
    );

    let mut build = cc::Build::new();
    build
        .warnings(false)
        .include(&vlfeat_root)
        .file("src/native/vlfeat_sift.c");

    let arch = target_arch();
    for source in vlfeat_sift_sources(&arch) {
        build.file(vlfeat_root.join(source));
    }
    if arch == "x86_64" || arch == "x86" {
        build.flag_if_supported("-msse2");
        if env::var("CARGO_CFG_TARGET_FEATURE")
            .map(|features| features.contains("avx"))
            .unwrap_or(false)
        {
            build.flag_if_supported("-mavx");
        } else {
            build.define("VL_DISABLE_AVX", None);
        }
    } else {
        build.define("VL_DISABLE_AVX", None);
        build.define("VL_DISABLE_SSE2", None);
    }
    build.define("VL_DISABLE_OPENMP", None);

    build.compile("rustsfm_vlfeat_sift");
}

fn target_arch() -> String {
    env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default()
}

fn target_os() -> String {
    env::var("CARGO_CFG_TARGET_OS").unwrap_or_default()
}

fn vlfeat_sift_sources(target_arch: &str) -> Vec<&'static str> {
    let mut sources = vec![
        "generic.c",
        "host.c",
        "mathop.c",
        "imopv.c",
        "sift.c",
        "scalespace.c",
        "stringop.c",
        "random.c",
        "covdet.c",
    ];
    if target_arch == "x86_64" || target_arch == "x86" {
        sources.push("mathop_sse2.c");
        sources.push("imopv_sse2.c");
    }
    sources
}

fn build_poselib_bridge() {
    let poselib_root = resolve_poselib_root().unwrap_or_else(|| {
        panic!(
            "RustSFM poselib feature requires PoseLib v2.0.5 source. Set POSELIB_ROOT=/path/to/PoseLib-2.0.5, place it at RustSFM/third_party/PoseLib, or at the workspace root third_party/PoseLib"
        )
    });

    let eigen_include = eigen_include_dir().unwrap_or_else(|| {
        panic!(
            "RustSFM poselib feature requires Eigen3 headers. Set EIGEN3_INCLUDE_DIR or install eigen3 so pkg-config can find it"
        )
    });

    println!("cargo:rerun-if-changed=src/native/poselib_bridge.cpp");
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
        .file("src/native/poselib_bridge.cpp")
        .file(poselib_root.join("PoseLib/solvers/gen_relpose_6pt.cc"))
        .file(poselib_root.join("PoseLib/solvers/gp3p.cc"))
        .file(poselib_root.join("PoseLib/solvers/p3p.cc"))
        .file(poselib_root.join("PoseLib/misc/re3q3.cc"))
        .file(poselib_root.join("PoseLib/misc/univariate.cc"))
        .file(poselib_root.join("PoseLib/misc/essential.cc"));

    if target_os() == "macos" {
        build.flag_if_supported("-stdlib=libc++");
    }

    build.compile("rustsfm_poselib_bridge");
}

fn resolve_vlfeat_root() -> Option<PathBuf> {
    let marker = Path::new("sift.c");
    let candidates = env::var_os("VLFEAT_ROOT")
        .map(PathBuf::from)
        .into_iter()
        .chain([
            PathBuf::from("third_party/vlfeat"),
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("third_party/vlfeat"),
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("third_party/vlfeat"),
        ]);
    candidates
        .map(|path| path.canonicalize().unwrap_or(path))
        .find(|path| path.join(marker).exists())
}

fn resolve_poselib_root() -> Option<PathBuf> {
    let marker = Path::new("PoseLib/solvers/gen_relpose_6pt.cc");
    let candidates = env::var_os("POSELIB_ROOT")
        .map(PathBuf::from)
        .into_iter()
        .chain([
            PathBuf::from("third_party/PoseLib"),
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("third_party/PoseLib"),
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("third_party/PoseLib"),
        ]);
    candidates
        .map(|path| path.canonicalize().unwrap_or(path))
        .find(|path| path.join(marker).exists())
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

fn build_colmap_eigen() {
    let Some(eigen_include) = eigen_include_dir() else {
        println!(
            "cargo:warning=Eigen3 not found; COLMAP Eigen numerical bridge disabled (set EIGEN3_INCLUDE_DIR or install eigen3)"
        );
        return;
    };

    println!("cargo:rerun-if-changed=src/native/colmap_eigen.cpp");
    println!("cargo:rerun-if-changed=src/native/colmap_eigen.h");
    println!("cargo:rustc-cfg=colmap_eigen");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .warnings(false)
        .include(&eigen_include)
        .file("src/native/colmap_eigen.cpp");

    if target_os() == "macos" {
        build.flag_if_supported("-stdlib=libc++");
    }

    build.compile("rustsfm_colmap_eigen");
}

fn build_colmap_image() {
    let Some(root) = resolve_freeimage_root() else {
        println!(
            "cargo:warning=FreeImage not found; COLMAP JPEG parity image loading disabled (install freeimage or set FREEIMAGE_ROOT)"
        );
        return;
    };

    println!("cargo:rerun-if-changed=src/native/colmap_image.c");
    println!("cargo:rerun-if-changed=src/native/colmap_image.h");
    println!("cargo:rustc-cfg=colmap_freeimage");

    cc::Build::new()
        .file("src/native/colmap_image.c")
        .include(root.join("include"))
        .compile("rustsfm_colmap_image");

    println!(
        "cargo:rustc-link-search=native={}",
        root.join("lib").display()
    );
    println!("cargo:rustc-link-lib=dylib=freeimage");
}

fn resolve_freeimage_root() -> Option<PathBuf> {
    if let Ok(root) = env::var("FREEIMAGE_ROOT") {
        let path = PathBuf::from(root);
        if path.join("include/FreeImage.h").exists() {
            return Some(path);
        }
    }

    for candidate in [
        "/opt/homebrew/opt/freeimage",
        "/usr/local/opt/freeimage",
        "/usr",
    ] {
        let path = PathBuf::from(candidate);
        if path.join("include/FreeImage.h").exists() {
            return Some(path);
        }
    }

    pkg_config_prefix("freeimage")
}

fn pkg_config_prefix(package: &str) -> Option<PathBuf> {
    let output = std::process::Command::new("pkg-config")
        .args(["--variable=prefix", package])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let prefix = String::from_utf8(output.stdout).ok()?;
    let path = PathBuf::from(prefix.trim());
    if path.join("include/FreeImage.h").exists() {
        Some(path)
    } else {
        None
    }
}
