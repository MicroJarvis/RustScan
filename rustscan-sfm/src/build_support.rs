use std::path::{Path, PathBuf};

pub fn manifest_dir_from_env() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn source_root_candidates(manifest_dir: &Path, relative_root: &Path) -> Vec<PathBuf> {
    vec![
        relative_root.to_path_buf(),
        manifest_dir.join(relative_root),
        manifest_dir.join("..").join(relative_root),
    ]
}
