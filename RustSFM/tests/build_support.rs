#[allow(dead_code)]
#[path = "../src/build_support.rs"]
mod build_support;

use std::path::Path;

const MANIFEST_PROBE_ENV: &str = "RUSTSFM_BUILD_SUPPORT_MANIFEST_PROBE";

#[test]
fn source_root_candidates_follow_the_current_manifest_directory() {
    let manifest = Path::new("/repo/.worktrees/runtime-fix/RustSFM");
    let candidates =
        build_support::source_root_candidates(manifest, Path::new("third_party/PoseLib"));

    assert_eq!(candidates[0], Path::new("third_party/PoseLib"));
    assert_eq!(
        candidates[1],
        Path::new("/repo/.worktrees/runtime-fix/RustSFM/third_party/PoseLib")
    );
    assert_eq!(
        candidates[2],
        Path::new("/repo/.worktrees/runtime-fix/RustSFM/../third_party/PoseLib")
    );
    assert!(candidates
        .iter()
        .any(|path| path.to_string_lossy().contains(".worktrees/")));
}

#[test]
fn production_build_script_uses_shared_runtime_manifest_lookup() {
    let build_script = include_str!("../build.rs");

    assert!(build_script.contains("build_support::manifest_dir_from_env()"));
    assert!(!build_script.contains("env!(\"CARGO_MANIFEST_DIR\")"));
}

#[test]
fn manifest_dir_from_env_reads_the_child_process_environment() {
    let runtime_manifest =
        std::env::temp_dir().join(format!("rustsfm-runtime-manifest-{}", std::process::id()));
    assert_ne!(
        runtime_manifest,
        Path::new(env!("CARGO_MANIFEST_DIR")),
        "runtime probe must differ from the compile-time manifest directory"
    );

    let output = std::process::Command::new(std::env::current_exe().expect("current test binary"))
        .arg("--exact")
        .arg("manifest_dir_from_env_child_process_probe")
        .arg("--ignored")
        .arg("--nocapture")
        .env("CARGO_MANIFEST_DIR", &runtime_manifest)
        .env(MANIFEST_PROBE_ENV, &runtime_manifest)
        .output()
        .expect("run isolated manifest-directory probe");

    assert!(
        output.status.success(),
        "child probe failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore = "launched by manifest_dir_from_env_reads_the_child_process_environment"]
fn manifest_dir_from_env_child_process_probe() {
    let expected = std::env::var_os(MANIFEST_PROBE_ENV)
        .expect("manifest probe must be launched by the parent test");

    assert_eq!(
        build_support::manifest_dir_from_env(),
        std::path::PathBuf::from(expected)
    );
}
