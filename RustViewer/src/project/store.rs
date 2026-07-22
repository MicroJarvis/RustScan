use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};

use fs2::FileExt;
use thiserror::Error;
use uuid::Uuid;

use super::{
    ArtifactRef, ChangeKind, ImportConfigSnapshot, PnpConfigSnapshot, ProjectManifest,
    ProjectManifestValidationError, SfmConfigSnapshot, SourceSpec, PROJECT_SCHEMA_VERSION,
};

const PACKAGE_EXTENSION: &str = "rustscanproject";
const MANIFEST_NAME: &str = "project.json";
const LOCK_NAME: &str = "project.lock";
const PACKAGE_DIRECTORIES: [&str; 12] = [
    "Sources",
    "Sources/managed",
    "Cache",
    "Cache/frames",
    "Cache/thumbnails",
    "Cache/.staging",
    "Reconstruction",
    "Training",
    "Training/checkpoints",
    "Artifacts",
    "Logs",
    "Logs/recovery",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectCreateRequest {
    pub display_name: String,
    pub source: SourceSpec,
}

impl ProjectCreateRequest {
    pub fn new(display_name: impl Into<String>, source: SourceSpec) -> Self {
        Self {
            display_name: display_name.into(),
            source,
        }
    }
}

#[derive(Debug, Error)]
pub enum ProjectStoreError {
    #[error("project package path must end in .rustscanproject: {path:?}")]
    InvalidPackageSuffix { path: PathBuf },
    #[error("project destination exists and is not an empty directory: {path:?}")]
    DestinationNotEmpty { path: PathBuf },
    #[error("project package root must not be a symbolic link: {path:?}")]
    SymlinkPackageRoot { path: PathBuf },
    #[error("project package is already open for writing: {path:?}")]
    AlreadyOpen { path: PathBuf },
    #[error("project manifest is missing a valid schema_version")]
    InvalidSchemaVersion,
    #[error("project schema version {found} is newer than supported version {supported}")]
    FutureSchemaVersion { found: u32, supported: u32 },
    #[error("no declared project migration is available from schema {from} to {to}")]
    MigrationUnavailable { from: u32, to: u32 },
    #[error("project manifest JSON is malformed at {path:?}: {source}")]
    MalformedJson {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("project manifest could not be serialized: {0}")]
    ManifestSerialization(#[source] serde_json::Error),
    #[error("project manifest identity changed from {expected} to {found}")]
    ProjectIdentityMismatch { expected: Uuid, found: Uuid },
    #[error("committed artifact is missing: {path:?}")]
    ArtifactMissing { path: PathBuf },
    #[error("committed artifact path contains a symbolic link: {path:?}")]
    ArtifactSymlink { path: PathBuf },
    #[error("committed artifact is not a regular file: {path:?}")]
    ArtifactNotRegularFile { path: PathBuf },
    #[error("committed artifact escapes the project package: {path:?}")]
    ArtifactPathEscapesPackage { path: PathBuf },
    #[error("artifact {path:?} has byte length {found}, expected {expected}")]
    ArtifactLengthMismatch {
        path: PathBuf,
        expected: u64,
        found: u64,
    },
    #[error("artifact {path:?} has BLAKE3 hash {found}, expected {expected}")]
    ArtifactHashMismatch {
        path: PathBuf,
        expected: String,
        found: String,
    },
    #[error(transparent)]
    InvalidManifest(#[from] ProjectManifestValidationError),
    #[error(transparent)]
    Io(#[from] io::Error),
}

#[derive(Debug)]
pub struct ProjectStore {
    root: PathBuf,
    manifest: ProjectManifest,
    _lock_file: File,
}

impl ProjectStore {
    pub fn create(
        path: impl AsRef<Path>,
        request: ProjectCreateRequest,
    ) -> Result<Self, ProjectStoreError> {
        let path = path.as_ref();
        require_package_suffix(path)?;
        let manifest = ProjectManifest::new(request.display_name, request.source);
        manifest.validate()?;

        let created_root = if path.exists() {
            let metadata = fs::symlink_metadata(path)?;
            if !metadata.is_dir() || fs::read_dir(path)?.next().transpose()?.is_some() {
                return Err(ProjectStoreError::DestinationNotEmpty {
                    path: path.to_path_buf(),
                });
            }
            false
        } else {
            fs::create_dir(path)?;
            true
        };

        let mut cleanup = InitializationCleanup::new(path.to_path_buf(), created_root);
        let root = fs::canonicalize(path)?;
        let lock_file = create_and_lock(&root, Some(&mut cleanup))?;
        for relative in PACKAGE_DIRECTORIES {
            cleanup.create_directory(&root.join(relative))?;
        }
        write_manifest_bootstrap(&root, &manifest)?;
        cleanup.disarm();

        Ok(Self {
            root,
            manifest,
            _lock_file: lock_file,
        })
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self, ProjectStoreError> {
        let path = path.as_ref();
        require_package_suffix(path)?;
        if fs::symlink_metadata(path)?.file_type().is_symlink() {
            return Err(ProjectStoreError::SymlinkPackageRoot {
                path: path.to_path_buf(),
            });
        }
        let root = fs::canonicalize(path)?;
        let lock_file = create_and_lock(&root, None)?;
        let manifest_path = root.join(MANIFEST_NAME);
        let bytes = fs::read(&manifest_path)?;
        let value: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|source| ProjectStoreError::MalformedJson {
                path: manifest_path.clone(),
                source,
            })?;
        let value = migrate_manifest(value)?;
        let manifest: ProjectManifest =
            serde_json::from_value(value).map_err(|source| ProjectStoreError::MalformedJson {
                path: manifest_path,
                source,
            })?;
        manifest.validate()?;

        let store = Self {
            root,
            manifest,
            _lock_file: lock_file,
        };
        store.validate_committed_artifacts()?;
        Ok(store)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn manifest(&self) -> &ProjectManifest {
        &self.manifest
    }

    pub fn update_source(&mut self, source: SourceSpec) -> Result<(), ProjectStoreError> {
        if self.manifest.source == source {
            return Ok(());
        }
        self.update_manifest(|manifest| {
            manifest.source = source;
            manifest.invalidate(ChangeKind::Source);
        })
    }

    pub fn update_import_config(
        &mut self,
        config: ImportConfigSnapshot,
    ) -> Result<(), ProjectStoreError> {
        if self.manifest.import_config == config {
            return Ok(());
        }
        self.update_manifest(|manifest| {
            manifest.import_config = config;
            manifest.invalidate(ChangeKind::ImportConfig);
        })
    }

    pub fn update_sfm_config(
        &mut self,
        config: SfmConfigSnapshot,
    ) -> Result<(), ProjectStoreError> {
        if self.manifest.sfm_config == config {
            return Ok(());
        }
        self.update_manifest(|manifest| {
            manifest.sfm_config = config;
            manifest.invalidate(ChangeKind::SfmConfig);
        })
    }

    pub fn update_pnp_config(
        &mut self,
        config: PnpConfigSnapshot,
    ) -> Result<(), ProjectStoreError> {
        if self.manifest.pnp_config == config {
            return Ok(());
        }
        self.update_manifest(|manifest| {
            manifest.pnp_config = config;
            manifest.invalidate(ChangeKind::PnpConfig);
        })
    }

    pub fn update_training_config(
        &mut self,
        config: rustgs::TrainingConfig,
    ) -> Result<(), ProjectStoreError> {
        if self.manifest.training_config == config {
            return Ok(());
        }
        self.update_manifest(|manifest| {
            manifest.training_config = config;
            manifest.invalidate(ChangeKind::TrainingConfig);
        })
    }

    fn update_manifest(
        &mut self,
        update: impl FnOnce(&mut ProjectManifest),
    ) -> Result<(), ProjectStoreError> {
        let mut manifest = self.manifest.clone();
        update(&mut manifest);
        self.write_manifest_atomic(&manifest)?;
        self.manifest = manifest;
        Ok(())
    }

    fn write_manifest_atomic(&self, manifest: &ProjectManifest) -> Result<(), ProjectStoreError> {
        manifest.validate()?;
        if manifest.id != self.manifest.id {
            return Err(ProjectStoreError::ProjectIdentityMismatch {
                expected: self.manifest.id,
                found: manifest.id,
            });
        }
        let bytes = serde_json::to_vec_pretty(manifest)
            .map_err(ProjectStoreError::ManifestSerialization)?;
        write_bytes_atomic(&self.root.join(MANIFEST_NAME), &bytes)?;
        Ok(())
    }

    fn validate_committed_artifacts(&self) -> Result<(), ProjectStoreError> {
        let mut seen = BTreeSet::new();
        for stage in super::ProjectStage::ORDER {
            for artifact in self.manifest.try_stage(stage)?.artifacts() {
                if seen.insert((
                    artifact.relative_path.clone(),
                    artifact.content_hash.clone(),
                    artifact.byte_len,
                )) {
                    self.validate_committed_artifact(artifact)?;
                }
            }
        }
        for artifact in [
            self.manifest.active_scene.as_ref(),
            self.manifest.final_scene.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            if seen.insert((
                artifact.relative_path.clone(),
                artifact.content_hash.clone(),
                artifact.byte_len,
            )) {
                self.validate_committed_artifact(artifact)?;
            }
        }
        Ok(())
    }

    fn validate_committed_artifact(&self, artifact: &ArtifactRef) -> Result<(), ProjectStoreError> {
        let relative = Path::new(&artifact.relative_path);
        let mut current = self.root.clone();
        for component in relative.components() {
            let Component::Normal(part) = component else {
                return Err(ProjectStoreError::ArtifactPathEscapesPackage {
                    path: relative.to_path_buf(),
                });
            };
            current.push(part);
            let metadata = match fs::symlink_metadata(&current) {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    return Err(ProjectStoreError::ArtifactMissing {
                        path: relative.to_path_buf(),
                    });
                }
                Err(error) => return Err(error.into()),
            };
            if metadata.file_type().is_symlink() {
                return Err(ProjectStoreError::ArtifactSymlink {
                    path: relative.to_path_buf(),
                });
            }
        }

        let canonical = fs::canonicalize(&current)?;
        if !canonical.starts_with(&self.root) {
            return Err(ProjectStoreError::ArtifactPathEscapesPackage {
                path: relative.to_path_buf(),
            });
        }
        if !fs::symlink_metadata(&current)?.is_file() {
            return Err(ProjectStoreError::ArtifactNotRegularFile {
                path: relative.to_path_buf(),
            });
        }
        let (found_len, found_hash) = hash_reader(File::open(&current)?)?;
        if found_len != artifact.byte_len {
            return Err(ProjectStoreError::ArtifactLengthMismatch {
                path: relative.to_path_buf(),
                expected: artifact.byte_len,
                found: found_len,
            });
        }
        let found_hash = found_hash.to_hex().to_string();
        if found_hash != artifact.content_hash {
            return Err(ProjectStoreError::ArtifactHashMismatch {
                path: relative.to_path_buf(),
                expected: artifact.content_hash.clone(),
                found: found_hash,
            });
        }
        Ok(())
    }
}

fn create_and_lock(
    root: &Path,
    cleanup: Option<&mut InitializationCleanup>,
) -> Result<File, ProjectStoreError> {
    let lock_path = root.join(LOCK_NAME);
    let (file, created) = match OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&lock_path)
    {
        Ok(file) => (file, true),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => (
            OpenOptions::new().read(true).write(true).open(&lock_path)?,
            false,
        ),
        Err(error) => return Err(error.into()),
    };
    if created {
        if let Some(cleanup) = cleanup {
            cleanup.created.push(lock_path.clone());
        }
    }
    match file.try_lock_exclusive() {
        Ok(()) => Ok(file),
        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
            Err(ProjectStoreError::AlreadyOpen { path: lock_path })
        }
        Err(error) => Err(error.into()),
    }
}

struct InitializationCleanup {
    root: PathBuf,
    remove_root: bool,
    created: Vec<PathBuf>,
    armed: bool,
}

impl InitializationCleanup {
    fn new(root: PathBuf, remove_root: bool) -> Self {
        Self {
            root,
            remove_root,
            created: Vec::new(),
            armed: true,
        }
    }

    fn create_directory(&mut self, path: &Path) -> io::Result<()> {
        if !path.exists() {
            fs::create_dir(path)?;
            self.created.push(path.to_path_buf());
        }
        Ok(())
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for InitializationCleanup {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        if self.remove_root {
            let _ = fs::remove_dir_all(&self.root);
            return;
        }
        for path in self.created.iter().rev() {
            match fs::symlink_metadata(path) {
                Ok(metadata) if metadata.is_dir() => {
                    let _ = fs::remove_dir(path);
                }
                Ok(_) => {
                    let _ = fs::remove_file(path);
                }
                Err(_) => {}
            }
        }
    }
}

fn require_package_suffix(path: &Path) -> Result<(), ProjectStoreError> {
    if path.extension() != Some(OsStr::new(PACKAGE_EXTENSION)) {
        return Err(ProjectStoreError::InvalidPackageSuffix {
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

fn schema_version(value: &serde_json::Value) -> Result<u32, ProjectStoreError> {
    value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .and_then(|version| u32::try_from(version).ok())
        .ok_or(ProjectStoreError::InvalidSchemaVersion)
}

fn migrate_manifest(mut value: serde_json::Value) -> Result<serde_json::Value, ProjectStoreError> {
    loop {
        let version = schema_version(&value)?;
        if version == PROJECT_SCHEMA_VERSION {
            return Ok(value);
        }
        if version > PROJECT_SCHEMA_VERSION {
            return Err(ProjectStoreError::FutureSchemaVersion {
                found: version,
                supported: PROJECT_SCHEMA_VERSION,
            });
        }
        value = migrate_one_version(value, version)?;
    }
}

fn migrate_one_version(
    _value: serde_json::Value,
    from: u32,
) -> Result<serde_json::Value, ProjectStoreError> {
    // Schema v1 is the first published format, so no v0 data contract exists to migrate.
    Err(ProjectStoreError::MigrationUnavailable {
        from,
        to: PROJECT_SCHEMA_VERSION,
    })
}

fn write_manifest_bootstrap(
    root: &Path,
    manifest: &ProjectManifest,
) -> Result<(), ProjectStoreError> {
    manifest.validate()?;
    let bytes =
        serde_json::to_vec_pretty(manifest).map_err(ProjectStoreError::ManifestSerialization)?;
    write_bytes_atomic(&root.join(MANIFEST_NAME), &bytes)?;
    Ok(())
}

fn write_bytes_atomic(destination: &Path, bytes: &[u8]) -> io::Result<()> {
    let parent = destination.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "project manifest has no parent directory",
        )
    })?;
    let name = destination
        .file_name()
        .unwrap_or_else(|| OsStr::new("artifact"))
        .to_string_lossy();
    let temporary = parent.join(format!(".{name}.{}.tmp", Uuid::new_v4()));
    let result = (|| {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, destination)?;
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn hash_reader(mut reader: impl Read) -> io::Result<(u64, blake3::Hash)> {
    let mut hasher = blake3::Hasher::new();
    let mut total = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
        total = total.checked_add(count as u64).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "artifact length overflow")
        })?;
    }
    Ok((total, hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_manifest_writer_rejects_project_identity_changes() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("Flowers.rustscanproject");
        let store = ProjectStore::create(
            &path,
            ProjectCreateRequest::new("Flowers", SourceSpec::managed_images("source-a")),
        )
        .unwrap();
        let before = fs::read(path.join(MANIFEST_NAME)).unwrap();
        let mut changed = store.manifest().clone();
        changed.id = Uuid::new_v4();

        assert!(matches!(
            store.write_manifest_atomic(&changed),
            Err(ProjectStoreError::ProjectIdentityMismatch { .. })
        ));
        assert_eq!(fs::read(path.join(MANIFEST_NAME)).unwrap(), before);
    }

    #[test]
    fn initialization_cleanup_removes_a_partial_root_it_created() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("Partial.rustscanproject");
        fs::create_dir(&root).unwrap();
        let mut cleanup = InitializationCleanup::new(root.clone(), true);
        cleanup.create_directory(&root.join("Sources")).unwrap();
        fs::write(root.join(LOCK_NAME), b"").unwrap();
        drop(cleanup);

        assert!(!root.exists());
    }
}
