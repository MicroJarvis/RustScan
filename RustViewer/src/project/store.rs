use std::collections::BTreeSet;
use std::ffi::OsStr;
#[cfg(any(not(unix), test))]
use std::fs::OpenOptions;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};

use fs2::FileExt;
#[cfg(unix)]
use rustix::fs::{self as rustix_fs, FileType, Mode, OFlags};
#[cfg(unix)]
use rustix::io::Errno;
use thiserror::Error;
use uuid::Uuid;

use super::{
    ArtifactRef, ChangeKind, ImportConfigSnapshot, PnpConfigSnapshot, ProjectManifest,
    ProjectManifestValidationError, ProjectStage, SfmConfigSnapshot, SourceSpec,
    PROJECT_SCHEMA_VERSION,
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
    #[error("cannot apply {change:?} while {stage:?} is active")]
    StageActive {
        stage: ProjectStage,
        change: ChangeKind,
    },
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectStoreWarning {
    ParentDirectorySyncFailed {
        path: PathBuf,
        error_kind: io::ErrorKind,
        detail: String,
    },
}

#[derive(Debug)]
pub struct ProjectStore {
    root: PathBuf,
    manifest: ProjectManifest,
    warnings: Vec<ProjectStoreWarning>,
    root_directory: File,
    _lock_file: File,
}

impl ProjectStore {
    pub fn create(
        path: impl AsRef<Path>,
        request: ProjectCreateRequest,
    ) -> Result<Self, ProjectStoreError> {
        Self::create_with_hooks(path, request, |_| Ok(()), || {}, || {})
    }

    #[cfg(test)]
    fn create_with_parent_sync(
        path: impl AsRef<Path>,
        request: ProjectCreateRequest,
        sync_parent: impl FnMut(&Path) -> io::Result<()>,
    ) -> Result<Self, ProjectStoreError> {
        Self::create_with_hooks(path, request, sync_parent, || {}, || {})
    }

    #[cfg(test)]
    fn create_with_initialization_hook(
        path: impl AsRef<Path>,
        request: ProjectCreateRequest,
        after_lock: impl FnOnce(),
    ) -> Result<Self, ProjectStoreError> {
        Self::create_with_hooks(path, request, |_| Ok(()), || {}, after_lock)
    }

    #[cfg(test)]
    fn create_with_parent_open_hook(
        path: impl AsRef<Path>,
        request: ProjectCreateRequest,
        after_parent_open: impl FnOnce(),
    ) -> Result<Self, ProjectStoreError> {
        Self::create_with_hooks(path, request, |_| Ok(()), after_parent_open, || {})
    }

    fn create_with_hooks(
        path: impl AsRef<Path>,
        request: ProjectCreateRequest,
        mut sync_hook: impl FnMut(&Path) -> io::Result<()>,
        after_parent_open: impl FnOnce(),
        after_lock: impl FnOnce(),
    ) -> Result<Self, ProjectStoreError> {
        let path = path.as_ref();
        require_package_suffix(path)?;
        let manifest = ProjectManifest::new(request.display_name, request.source);
        manifest.validate()?;

        let package_parent_input = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let package_parent = fs::canonicalize(package_parent_input)?;
        let root_name = path.file_name().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "project package has no directory name",
            )
        })?;
        let root = package_parent.join(root_name);
        let package_parent_directory = open_package_directory(&package_parent)?;
        after_parent_open();
        let (root_directory, created_root) =
            open_or_create_package_root(&package_parent_directory, root_name, &root)?;
        if !is_bootstrap_destination_in_directory(&root_directory)? {
            return Err(ProjectStoreError::DestinationNotEmpty { path: root });
        }
        let mut cleanup = InitializationCleanup::new(root.clone(), created_root);
        let lock_file = match create_and_lock(&root, &root_directory, Some(&mut cleanup)) {
            Ok(lock_file) => lock_file,
            Err(error) => {
                // Without ownership of the persistent lock, cleanup could race its holder.
                cleanup.disarm();
                return Err(error);
            }
        };
        let mut initialization = InitializationGuard::new(cleanup, lock_file, root_directory);
        after_lock();
        if !is_bootstrap_destination_in_directory(initialization.root_directory())? {
            return Err(ProjectStoreError::DestinationNotEmpty { path: root });
        }
        let mut created_directory_parents = BTreeSet::new();
        for relative in PACKAGE_DIRECTORIES {
            let relative = Path::new(relative);
            if initialization.create_directory(relative)? {
                created_directory_parents.insert(
                    relative
                        .parent()
                        .expect("package scaffold directory has a parent")
                        .to_path_buf(),
                );
            }
        }
        let mut warnings = write_manifest_bootstrap_with_parent_sync(
            &root,
            initialization.root_directory(),
            &manifest,
            |parent| sync_hook(parent),
        )?
        .into_iter()
        .collect::<Vec<_>>();
        // The manifest rename sync already covers all new entries directly under the root.
        created_directory_parents.remove(Path::new(""));
        for relative_parent in created_directory_parents {
            let label = root.join(&relative_parent);
            if let Some(warning) = sync_directory_at_with_hook(
                initialization.root_directory(),
                &relative_parent,
                &label,
                &mut sync_hook,
            ) {
                warnings.push(warning);
            }
        }
        if created_root {
            if let Some(warning) = sync_open_directory_with_hook(
                &package_parent_directory,
                &package_parent,
                &mut sync_hook,
            ) {
                warnings.push(warning);
            }
        }
        let (lock_file, root_directory) = initialization.finish();

        Ok(Self {
            root,
            manifest,
            warnings,
            root_directory,
            _lock_file: lock_file,
        })
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self, ProjectStoreError> {
        Self::open_with_validation_hook(path, || {})
    }

    fn open_with_validation_hook(
        path: impl AsRef<Path>,
        before_artifact_validation: impl FnOnce(),
    ) -> Result<Self, ProjectStoreError> {
        let path = path.as_ref();
        require_package_suffix(path)?;
        if fs::symlink_metadata(path)?.file_type().is_symlink() {
            return Err(ProjectStoreError::SymlinkPackageRoot {
                path: path.to_path_buf(),
            });
        }
        let root = fs::canonicalize(path)?;
        let root_directory = open_package_directory(&root)?;
        let lock_file = create_and_lock(&root, &root_directory, None)?;
        let manifest_path = root.join(MANIFEST_NAME);
        let bytes = read_manifest_from_directory(&root_directory, &root)?;
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
            warnings: Vec::new(),
            root_directory,
            _lock_file: lock_file,
        };
        before_artifact_validation();
        store.validate_committed_artifacts()?;
        Ok(store)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn manifest(&self) -> &ProjectManifest {
        &self.manifest
    }

    pub fn warnings(&self) -> &[ProjectStoreWarning] {
        &self.warnings
    }

    pub fn take_warnings(&mut self) -> Vec<ProjectStoreWarning> {
        std::mem::take(&mut self.warnings)
    }

    pub fn update_source(&mut self, source: SourceSpec) -> Result<(), ProjectStoreError> {
        if self.manifest.source == source {
            return Ok(());
        }
        self.ensure_change_does_not_invalidate_active_stage(ChangeKind::Source)?;
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
        self.ensure_change_does_not_invalidate_active_stage(ChangeKind::ImportConfig)?;
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
        self.ensure_change_does_not_invalidate_active_stage(ChangeKind::SfmConfig)?;
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
        self.ensure_change_does_not_invalidate_active_stage(ChangeKind::PnpConfig)?;
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
        self.ensure_change_does_not_invalidate_active_stage(ChangeKind::TrainingConfig)?;
        self.update_manifest(|manifest| {
            manifest.training_config = config;
            manifest.invalidate(ChangeKind::TrainingConfig);
        })
    }

    fn update_manifest(
        &mut self,
        update: impl FnOnce(&mut ProjectManifest),
    ) -> Result<(), ProjectStoreError> {
        self.update_manifest_with_parent_sync(update, |_| Ok(()))
    }

    fn ensure_change_does_not_invalidate_active_stage(
        &self,
        change: ChangeKind,
    ) -> Result<(), ProjectStoreError> {
        if let Some(lease) = self.manifest.lease() {
            if change.invalidates(lease.stage) {
                return Err(ProjectStoreError::StageActive {
                    stage: lease.stage,
                    change,
                });
            }
        }
        Ok(())
    }

    fn update_manifest_with_parent_sync(
        &mut self,
        update: impl FnOnce(&mut ProjectManifest),
        sync_parent: impl FnOnce(&Path) -> io::Result<()>,
    ) -> Result<(), ProjectStoreError> {
        let mut manifest = self.manifest.clone();
        update(&mut manifest);
        let warning = self.write_manifest_atomic_with_parent_sync(&manifest, sync_parent)?;
        self.manifest = manifest;
        self.warnings.extend(warning);
        Ok(())
    }

    fn write_manifest_atomic_with_parent_sync(
        &self,
        manifest: &ProjectManifest,
        sync_parent: impl FnOnce(&Path) -> io::Result<()>,
    ) -> Result<Option<ProjectStoreWarning>, ProjectStoreError> {
        manifest.validate()?;
        if manifest.id() != self.manifest.id() {
            return Err(ProjectStoreError::ProjectIdentityMismatch {
                expected: self.manifest.id(),
                found: manifest.id(),
            });
        }
        let bytes = serde_json::to_vec_pretty(manifest)
            .map_err(ProjectStoreError::ManifestSerialization)?;
        write_bytes_atomic_in_directory_with_hooks(
            &self.root_directory,
            &self.root,
            MANIFEST_NAME,
            &bytes,
            sync_parent,
            |_| Ok(()),
        )
        .map_err(ProjectStoreError::Io)
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
        for component in relative.components() {
            let Component::Normal(_) = component else {
                return Err(ProjectStoreError::ArtifactPathEscapesPackage {
                    path: relative.to_path_buf(),
                });
            };
        }
        let file = open_artifact_file(&self.root_directory, &self.root, relative)?;
        let (found_len, found_hash) = hash_reader(file)?;
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
    root_directory: &File,
    cleanup: Option<&mut InitializationCleanup>,
) -> Result<File, ProjectStoreError> {
    let lock_path = root.join(LOCK_NAME);
    #[cfg(unix)]
    let (file, created) = match rustix_fs::openat(
        root_directory,
        LOCK_NAME,
        OFlags::RDWR | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::RUSR | Mode::WUSR,
    ) {
        Ok(file) => (File::from(file), true),
        Err(error) if error == Errno::EXIST => (
            File::from(
                rustix_fs::openat(
                    root_directory,
                    LOCK_NAME,
                    OFlags::RDWR | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                    Mode::empty(),
                )
                .map_err(io::Error::from)?,
            ),
            false,
        ),
        Err(error) => return Err(io::Error::from(error).into()),
    };
    #[cfg(not(unix))]
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
    match file.try_lock_exclusive() {
        Ok(()) => {
            if created {
                if let Some(cleanup) = cleanup {
                    cleanup.record_created_file(PathBuf::from(LOCK_NAME));
                }
            }
            Ok(file)
        }
        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
            Err(ProjectStoreError::AlreadyOpen { path: lock_path })
        }
        Err(error) => Err(error.into()),
    }
}

#[derive(Debug)]
enum CreatedEntry {
    File(PathBuf),
    Directory(PathBuf),
}

struct InitializationCleanup {
    root: PathBuf,
    remove_root: bool,
    created: Vec<CreatedEntry>,
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

    fn create_directory(&mut self, root_directory: &File, relative: &Path) -> io::Result<bool> {
        #[cfg(unix)]
        {
            let components = relative
                .components()
                .map(|component| match component {
                    Component::Normal(part) => Ok(part),
                    _ => Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "package scaffold path is not relative",
                    )),
                })
                .collect::<io::Result<Vec<_>>>()?;
            let (leaf, parents) = components.split_last().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "empty scaffold path")
            })?;
            let mut parent_directory = root_directory.try_clone()?;
            for parent in parents {
                let opened = rustix_fs::openat(
                    &parent_directory,
                    *parent,
                    OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                    Mode::empty(),
                )
                .map_err(io::Error::from)?;
                parent_directory = File::from(opened);
            }
            rustix_fs::mkdirat(
                &parent_directory,
                *leaf,
                Mode::RWXU | Mode::RGRP | Mode::XGRP | Mode::ROTH | Mode::XOTH,
            )
            .map_err(io::Error::from)?;
        }
        #[cfg(not(unix))]
        fs::create_dir(self.root.join(relative))?;
        self.created
            .push(CreatedEntry::Directory(relative.to_path_buf()));
        Ok(true)
    }

    fn record_created_file(&mut self, relative: PathBuf) {
        self.created.push(CreatedEntry::File(relative));
    }

    fn cleanup_with_directory(&mut self, root_directory: &File, before_cleanup: impl FnOnce()) {
        if !self.armed {
            return;
        }
        before_cleanup();

        for entry in self.created.iter().rev() {
            if matches!(entry, CreatedEntry::File(relative) if relative == Path::new(LOCK_NAME)) {
                continue;
            }
            #[cfg(unix)]
            {
                let (relative, flags) = match entry {
                    CreatedEntry::File(relative) => (relative, rustix_fs::AtFlags::empty()),
                    CreatedEntry::Directory(relative) => (relative, rustix_fs::AtFlags::REMOVEDIR),
                };
                let _ = rustix_fs::unlinkat(root_directory, relative, flags);
            }
            #[cfg(not(unix))]
            {
                let path = match entry {
                    CreatedEntry::File(relative) | CreatedEntry::Directory(relative) => {
                        self.root.join(relative)
                    }
                };
                match entry {
                    CreatedEntry::File(_) => {
                        let _ = fs::remove_file(path);
                    }
                    CreatedEntry::Directory(_) => {
                        let _ = fs::remove_dir(path);
                    }
                }
            }
        }

        self.disarm();
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
        if self.created.iter().any(|entry| {
            matches!(entry, CreatedEntry::File(relative) if relative == Path::new(LOCK_NAME))
        }) {
            // A recorded persistent lock may only be removed by InitializationGuard while held.
            return;
        }
        if self.remove_root {
            let _ = fs::remove_dir_all(&self.root);
            return;
        }
        for entry in self.created.iter().rev() {
            let path = match entry {
                CreatedEntry::File(relative) | CreatedEntry::Directory(relative) => {
                    self.root.join(relative)
                }
            };
            match entry {
                CreatedEntry::File(_) => {
                    let _ = fs::remove_file(path);
                }
                CreatedEntry::Directory(_) => {
                    let _ = fs::remove_dir(path);
                }
            }
        }
    }
}

struct InitializationGuard {
    cleanup: InitializationCleanup,
    lock_file: Option<File>,
    root_directory: Option<File>,
}

impl InitializationGuard {
    fn new(cleanup: InitializationCleanup, lock_file: File, root_directory: File) -> Self {
        Self {
            cleanup,
            lock_file: Some(lock_file),
            root_directory: Some(root_directory),
        }
    }

    fn root_directory(&self) -> &File {
        self.root_directory
            .as_ref()
            .expect("initialization guard owns the project directory")
    }

    fn create_directory(&mut self, relative: &Path) -> io::Result<bool> {
        let root_directory = self
            .root_directory
            .as_ref()
            .expect("initialization guard owns the project directory");
        self.cleanup.create_directory(root_directory, relative)
    }

    fn finish(mut self) -> (File, File) {
        self.cleanup.disarm();
        let lock_file = self
            .lock_file
            .take()
            .expect("initialization guard owns the project lock");
        let root_directory = self
            .root_directory
            .take()
            .expect("initialization guard owns the project directory");
        (lock_file, root_directory)
    }

    #[cfg(test)]
    fn cleanup_with_hook(mut self, before_cleanup: impl FnOnce()) {
        let root_directory = self
            .root_directory
            .as_ref()
            .expect("initialization guard owns the project directory");
        self.cleanup
            .cleanup_with_directory(root_directory, before_cleanup);
    }
}

impl Drop for InitializationGuard {
    fn drop(&mut self) {
        if let Some(root_directory) = self.root_directory.as_ref() {
            self.cleanup.cleanup_with_directory(root_directory, || {});
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

#[cfg(not(unix))]
fn is_bootstrap_destination(path: &Path) -> io::Result<bool> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if entry.file_name() != OsStr::new(LOCK_NAME)
            || !fs::symlink_metadata(entry.path())?.is_file()
        {
            return Ok(false);
        }
    }
    Ok(true)
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

fn write_manifest_bootstrap_with_parent_sync(
    root: &Path,
    root_directory: &File,
    manifest: &ProjectManifest,
    sync_parent: impl FnOnce(&Path) -> io::Result<()>,
) -> Result<Option<ProjectStoreWarning>, ProjectStoreError> {
    manifest.validate()?;
    let bytes =
        serde_json::to_vec_pretty(manifest).map_err(ProjectStoreError::ManifestSerialization)?;
    write_bytes_atomic_in_directory_with_hooks(
        root_directory,
        root,
        MANIFEST_NAME,
        &bytes,
        sync_parent,
        |_| Ok(()),
    )
    .map_err(ProjectStoreError::Io)
}

fn write_bytes_atomic_in_directory_with_hooks(
    root_directory: &File,
    root: &Path,
    destination_name: &str,
    bytes: &[u8],
    sync_parent: impl FnOnce(&Path) -> io::Result<()>,
    before_rename: impl FnOnce(&Path) -> io::Result<()>,
) -> io::Result<Option<ProjectStoreWarning>> {
    let temporary_name = format!(".{destination_name}.{}.tmp", Uuid::new_v4());
    let temporary = root.join(&temporary_name);
    let before_rename = (|| {
        #[cfg(unix)]
        let mut file = File::from(
            rustix_fs::openat(
                root_directory,
                temporary_name.as_str(),
                OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::RUSR | Mode::WUSR,
            )
            .map_err(io::Error::from)?,
        );
        #[cfg(not(unix))]
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        before_rename(&temporary)?;
        #[cfg(unix)]
        rustix_fs::renameat(
            root_directory,
            temporary_name.as_str(),
            root_directory,
            destination_name,
        )
        .map_err(io::Error::from)?;
        #[cfg(not(unix))]
        fs::rename(&temporary, root.join(destination_name))?;
        Ok(())
    })();
    if before_rename.is_err() {
        #[cfg(unix)]
        let _ = rustix_fs::unlinkat(
            root_directory,
            temporary_name.as_str(),
            rustix_fs::AtFlags::empty(),
        );
        #[cfg(not(unix))]
        let _ = fs::remove_file(&temporary);
        return before_rename.map(|()| None);
    }
    // Rename switches the manifest authority. A later durability failure must not invite a retry.
    let descriptor_error = root_directory.sync_all().err();
    let hook_error = sync_parent(root).err();
    Ok(descriptor_error.or(hook_error).map(|error| {
        ProjectStoreWarning::ParentDirectorySyncFailed {
            path: root.to_path_buf(),
            error_kind: error.kind(),
            detail: error.to_string(),
        }
    }))
}

fn sync_open_directory_with_hook(
    directory: &File,
    label: &Path,
    sync_hook: &mut impl FnMut(&Path) -> io::Result<()>,
) -> Option<ProjectStoreWarning> {
    let descriptor_error = directory.sync_all().err();
    let hook_error = sync_hook(label).err();
    descriptor_error
        .or(hook_error)
        .map(|error| ProjectStoreWarning::ParentDirectorySyncFailed {
            path: label.to_path_buf(),
            error_kind: error.kind(),
            detail: error.to_string(),
        })
}

fn sync_directory_at_with_hook(
    root_directory: &File,
    relative: &Path,
    label: &Path,
    sync_hook: &mut impl FnMut(&Path) -> io::Result<()>,
) -> Option<ProjectStoreWarning> {
    let descriptor_result = open_directory_at(root_directory, relative, label)
        .and_then(|directory| directory.sync_all());
    let hook_error = sync_hook(label).err();
    descriptor_result.err().or(hook_error).map(|error| {
        ProjectStoreWarning::ParentDirectorySyncFailed {
            path: label.to_path_buf(),
            error_kind: error.kind(),
            detail: error.to_string(),
        }
    })
}

#[cfg(test)]
fn sync_parent_directory(parent: &Path) -> io::Result<()> {
    File::open(parent)?.sync_all()
}

#[cfg(unix)]
fn open_package_directory(root: &Path) -> io::Result<File> {
    rustix_fs::open(
        root,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map(File::from)
    .map_err(io::Error::from)
}

#[cfg(unix)]
fn open_directory_at(root_directory: &File, relative: &Path, _label: &Path) -> io::Result<File> {
    let mut directory = root_directory.try_clone()?;
    for component in relative.components() {
        let Component::Normal(component) = component else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "directory path contains unsafe components",
            ));
        };
        directory = File::from(
            rustix_fs::openat(
                &directory,
                component,
                OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .map_err(io::Error::from)?,
        );
    }
    Ok(directory)
}

#[cfg(not(unix))]
fn open_directory_at(_root_directory: &File, _relative: &Path, label: &Path) -> io::Result<File> {
    File::open(label)
}

#[cfg(unix)]
fn open_or_create_package_root(
    package_parent_directory: &File,
    name: &OsStr,
    root_label: &Path,
) -> Result<(File, bool), ProjectStoreError> {
    match rustix_fs::statat(
        package_parent_directory,
        name,
        rustix_fs::AtFlags::SYMLINK_NOFOLLOW,
    ) {
        Ok(metadata) => {
            if !FileType::from_raw_mode(metadata.st_mode).is_dir() {
                return Err(ProjectStoreError::DestinationNotEmpty {
                    path: root_label.to_path_buf(),
                });
            }
            Ok((
                open_directory_at(package_parent_directory, Path::new(name), root_label)?,
                false,
            ))
        }
        Err(error) if error == Errno::NOENT => {
            rustix_fs::mkdirat(
                package_parent_directory,
                name,
                Mode::RWXU | Mode::RGRP | Mode::XGRP | Mode::ROTH | Mode::XOTH,
            )
            .map_err(io::Error::from)?;
            Ok((
                open_directory_at(package_parent_directory, Path::new(name), root_label)?,
                true,
            ))
        }
        Err(error) => Err(io::Error::from(error).into()),
    }
}

#[cfg(not(unix))]
fn open_or_create_package_root(
    _package_parent_directory: &File,
    name: &OsStr,
    root_label: &Path,
) -> Result<(File, bool), ProjectStoreError> {
    if root_label.exists() {
        let metadata = fs::symlink_metadata(root_label)?;
        if !metadata.is_dir() || !is_bootstrap_destination(root_label)? {
            return Err(ProjectStoreError::DestinationNotEmpty {
                path: root_label.to_path_buf(),
            });
        }
        Ok((File::open(root_label)?, false))
    } else {
        let parent = root_label.parent().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "project package has no parent")
        })?;
        fs::create_dir(parent.join(name))?;
        Ok((File::open(root_label)?, true))
    }
}

#[cfg(unix)]
fn is_bootstrap_destination_in_directory(root_directory: &File) -> io::Result<bool> {
    let directory = rustix_fs::Dir::read_from(root_directory).map_err(io::Error::from)?;
    for entry in directory {
        let entry = entry.map_err(io::Error::from)?;
        let name = entry.file_name().to_bytes();
        if name == b"." || name == b".." {
            continue;
        }
        if name != LOCK_NAME.as_bytes() {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(not(unix))]
fn is_bootstrap_destination_in_directory(_root_directory: &File) -> io::Result<bool> {
    Ok(true)
}

#[cfg(not(unix))]
fn open_package_directory(root: &Path) -> io::Result<File> {
    File::open(root)
}

#[cfg(unix)]
fn read_manifest_from_directory(root_directory: &File, _root: &Path) -> io::Result<Vec<u8>> {
    let opened = rustix_fs::openat(
        root_directory,
        MANIFEST_NAME,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(io::Error::from)?;
    let metadata = rustix_fs::fstat(&opened).map_err(io::Error::from)?;
    if !FileType::from_raw_mode(metadata.st_mode).is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "project manifest is not a regular file",
        ));
    }
    let mut bytes = Vec::new();
    File::from(opened).read_to_end(&mut bytes)?;
    Ok(bytes)
}

#[cfg(not(unix))]
fn read_manifest_from_directory(_root_directory: &File, root: &Path) -> io::Result<Vec<u8>> {
    fs::read(root.join(MANIFEST_NAME))
}

#[cfg(unix)]
fn open_artifact_file(
    root_directory: &File,
    _root: &Path,
    relative: &Path,
) -> Result<File, ProjectStoreError> {
    let mut components = relative.components().map(|component| match component {
        Component::Normal(part) => Ok(part),
        _ => Err(ProjectStoreError::ArtifactPathEscapesPackage {
            path: relative.to_path_buf(),
        }),
    });
    let mut directory = root_directory.try_clone()?;
    let Some(mut component) = components.next().transpose()? else {
        return Err(ProjectStoreError::ArtifactPathEscapesPackage {
            path: relative.to_path_buf(),
        });
    };

    for next in components {
        let next = next?;
        let opened = rustix_fs::openat(
            &directory,
            component,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(|error| artifact_open_error(&directory, component, relative, error))?;
        directory = File::from(opened);
        component = next;
    }

    let opened = rustix_fs::openat(
        &directory,
        component,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|error| artifact_open_error(&directory, component, relative, error))?;
    let metadata = rustix_fs::fstat(&opened).map_err(io::Error::from)?;
    if !FileType::from_raw_mode(metadata.st_mode).is_file() {
        return Err(ProjectStoreError::ArtifactNotRegularFile {
            path: relative.to_path_buf(),
        });
    }
    Ok(File::from(opened))
}

#[cfg(unix)]
fn artifact_open_error(
    directory: &File,
    component: &OsStr,
    relative: &Path,
    error: Errno,
) -> ProjectStoreError {
    if error == Errno::NOENT {
        return ProjectStoreError::ArtifactMissing {
            path: relative.to_path_buf(),
        };
    }
    if error == Errno::LOOP
        || rustix_fs::statat(directory, component, rustix_fs::AtFlags::SYMLINK_NOFOLLOW)
            .is_ok_and(|metadata| FileType::from_raw_mode(metadata.st_mode).is_symlink())
    {
        return ProjectStoreError::ArtifactSymlink {
            path: relative.to_path_buf(),
        };
    }
    if error == Errno::NOTDIR {
        return ProjectStoreError::ArtifactNotRegularFile {
            path: relative.to_path_buf(),
        };
    }
    ProjectStoreError::Io(error.into())
}

#[cfg(not(unix))]
fn open_artifact_file(
    _root_directory: &File,
    root: &Path,
    relative: &Path,
) -> Result<File, ProjectStoreError> {
    let mut current = root.to_path_buf();
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
    if !canonical.starts_with(root) {
        return Err(ProjectStoreError::ArtifactPathEscapesPackage {
            path: relative.to_path_buf(),
        });
    }
    if !fs::symlink_metadata(&current)?.is_file() {
        return Err(ProjectStoreError::ArtifactNotRegularFile {
            path: relative.to_path_buf(),
        });
    }
    Ok(File::open(current)?)
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
    use std::cell::RefCell;
    use std::rc::Rc;

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
            store.write_manifest_atomic_with_parent_sync(&changed, sync_parent_directory),
            Err(ProjectStoreError::ProjectIdentityMismatch { .. })
        ));
        assert_eq!(fs::read(path.join(MANIFEST_NAME)).unwrap(), before);
    }

    #[test]
    fn initialization_cleanup_removes_a_partial_root_it_created() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("Partial.rustscanproject");
        fs::create_dir(&root).unwrap();
        let root = fs::canonicalize(root).unwrap();
        let root_directory = open_package_directory(&root).unwrap();
        let mut cleanup = InitializationCleanup::new(root.clone(), true);
        cleanup
            .create_directory(&root_directory, Path::new("Sources"))
            .unwrap();
        drop(cleanup);

        assert!(!root.exists());
    }

    #[test]
    fn post_rename_sync_failure_commits_manifest_and_records_warning() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("Flowers.rustscanproject");
        let mut store = ProjectStore::create(
            &path,
            ProjectCreateRequest::new("Flowers", SourceSpec::managed_images("source-a")),
        )
        .unwrap();
        let mut expected = store.manifest().import_config.clone();
        expected.video_keyframes_per_second = 4.0;

        store
            .update_manifest_with_parent_sync(
                |manifest| {
                    manifest.import_config = expected.clone();
                    manifest.invalidate(ChangeKind::ImportConfig);
                },
                |_| Err(io::Error::other("injected parent sync failure")),
            )
            .unwrap();

        assert_eq!(store.manifest().import_config, expected);
        let persisted: ProjectManifest =
            serde_json::from_slice(&fs::read(path.join(MANIFEST_NAME)).unwrap()).unwrap();
        assert_eq!(persisted.import_config, expected);
        assert_eq!(
            store.warnings(),
            &[ProjectStoreWarning::ParentDirectorySyncFailed {
                path: store.root().to_path_buf(),
                error_kind: io::ErrorKind::Other,
                detail: "injected parent sync failure".to_owned(),
            }]
        );
        assert_eq!(store.take_warnings().len(), 1);
        assert!(store.warnings().is_empty());
    }

    #[test]
    fn bootstrap_post_rename_sync_failure_returns_a_committed_store() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("Bootstrap.rustscanproject");
        fs::create_dir(&root).unwrap();

        let store = ProjectStore::create_with_parent_sync(
            &root,
            ProjectCreateRequest::new("Bootstrap", SourceSpec::managed_images("source-a")),
            |_| Err(io::Error::other("injected bootstrap sync failure")),
        )
        .unwrap();

        assert!(!store.warnings().is_empty());
        assert!(store.warnings().iter().all(|warning| matches!(
            warning,
            ProjectStoreWarning::ParentDirectorySyncFailed { .. }
        )));
        assert!(matches!(
            ProjectStore::open(&root),
            Err(ProjectStoreError::AlreadyOpen { .. })
        ));
        let persisted: ProjectManifest =
            serde_json::from_slice(&fs::read(root.join(MANIFEST_NAME)).unwrap()).unwrap();
        assert_eq!(persisted.id(), store.manifest().id());
        assert!(root.join("Sources").is_dir());
    }

    #[test]
    fn bootstrap_syncs_created_directory_parents_and_new_package_parent() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("Durable.rustscanproject");
        let synced = Rc::new(RefCell::new(Vec::new()));
        let observed = Rc::clone(&synced);

        ProjectStore::create_with_parent_sync(
            &root,
            ProjectCreateRequest::new("Durable", SourceSpec::managed_images("source-a")),
            move |path| {
                observed.borrow_mut().push(path.to_path_buf());
                Ok(())
            },
        )
        .unwrap();

        let canonical_root = fs::canonicalize(&root).unwrap();
        let canonical_parent = canonical_root.parent().unwrap().to_path_buf();
        let synced = synced.borrow();
        for expected in [
            canonical_root.clone(),
            canonical_root.join("Sources"),
            canonical_root.join("Cache"),
            canonical_root.join("Training"),
            canonical_root.join("Logs"),
            canonical_parent,
        ] {
            assert!(
                synced.iter().any(|path| path == &expected),
                "did not sync {expected:?}"
            );
        }
    }

    #[test]
    fn bootstrap_directory_sync_failures_after_commit_are_warnings() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("Warning.rustscanproject");
        let package_parent = fs::canonicalize(temp.path()).unwrap();

        let store = ProjectStore::create_with_parent_sync(
            &root,
            ProjectCreateRequest::new("Warning", SourceSpec::managed_images("source-a")),
            |path| {
                if path == package_parent {
                    Err(io::Error::other("injected package-parent sync failure"))
                } else {
                    Ok(())
                }
            },
        )
        .unwrap();

        assert!(root.join(MANIFEST_NAME).is_file());
        assert!(store.warnings().iter().any(|warning| matches!(
            warning,
            ProjectStoreWarning::ParentDirectorySyncFailed { path, detail, .. }
                if path == &package_parent && detail == "injected package-parent sync failure"
        )));
    }

    #[test]
    fn pre_rename_failure_after_temp_creation_removes_the_temp_file() {
        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join(MANIFEST_NAME);
        let root_directory = open_package_directory(temp.path()).unwrap();

        let result = write_bytes_atomic_in_directory_with_hooks(
            &root_directory,
            temp.path(),
            MANIFEST_NAME,
            b"replacement",
            |_| Ok(()),
            |temporary| {
                assert!(temporary.is_file());
                Err(io::Error::other("injected pre-rename failure"))
            },
        );

        assert!(result.is_err());
        assert!(!destination.exists());
        assert!(fs::read_dir(temp.path()).unwrap().next().is_none());
    }

    #[cfg(unix)]
    #[test]
    fn initialization_cleanup_preserves_the_persistent_lock_inode_for_the_next_writer() {
        use std::os::unix::fs::MetadataExt;

        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("Failed.rustscanproject");
        fs::create_dir(&root).unwrap();
        let canonical_root = fs::canonicalize(&root).unwrap();
        let mut cleanup = InitializationCleanup::new(canonical_root.clone(), true);
        let root_directory = open_package_directory(&canonical_root).unwrap();
        let lock_file =
            create_and_lock(&canonical_root, &root_directory, Some(&mut cleanup)).unwrap();
        let mut guard = InitializationGuard::new(cleanup, lock_file, root_directory);
        guard.create_directory(Path::new("Sources")).unwrap();
        let lock_path = root.join(LOCK_NAME);
        let observer = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&lock_path)
            .unwrap();
        let old_metadata = observer.metadata().unwrap();

        guard.cleanup_with_hook(|| {});

        assert!(lock_path.is_file());
        assert!(!root.join("Sources").exists());
        let store = ProjectStore::create(
            &root,
            ProjectCreateRequest::new("Recovered", SourceSpec::managed_images("source-a")),
        )
        .unwrap();
        let writer_metadata = store._lock_file.metadata().unwrap();
        let path_metadata = fs::metadata(&lock_path).unwrap();
        assert_eq!(
            (old_metadata.dev(), old_metadata.ino()),
            (writer_metadata.dev(), writer_metadata.ino())
        );
        assert_eq!(
            (old_metadata.dev(), old_metadata.ino()),
            (path_metadata.dev(), path_metadata.ino())
        );
        assert!(matches!(
            observer.try_lock_exclusive(),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock
        ));
    }

    #[cfg(unix)]
    #[test]
    fn scaffold_creation_rejects_a_replaced_ancestor_symlink() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("Package.rustscanproject");
        let external = temp.path().join("outside");
        fs::create_dir(&root).unwrap();
        fs::create_dir(&external).unwrap();
        let root_directory = open_package_directory(&root).unwrap();
        let mut cleanup = InitializationCleanup::new(root.clone(), false);
        cleanup
            .create_directory(&root_directory, Path::new("Sources"))
            .unwrap();
        fs::remove_dir(root.join("Sources")).unwrap();
        symlink(&external, root.join("Sources")).unwrap();

        assert!(cleanup
            .create_directory(&root_directory, Path::new("Sources/managed"))
            .is_err());
        assert!(!external.join("managed").exists());
    }

    #[cfg(unix)]
    #[test]
    fn create_scaffold_and_manifest_stay_bound_to_the_locked_package_inode() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("Original.rustscanproject");
        let moved = temp.path().join("Moved.rustscanproject");

        let store = ProjectStore::create_with_initialization_hook(
            &root,
            ProjectCreateRequest::new("Original", SourceSpec::managed_images("source-a")),
            || {
                fs::rename(&root, &moved).unwrap();
                fs::create_dir(&root).unwrap();
            },
        )
        .unwrap();

        assert!(moved.join(MANIFEST_NAME).is_file());
        assert!(moved.join("Sources/managed").is_dir());
        assert!(moved.join("Cache/frames").is_dir());
        assert!(fs::read_dir(&root).unwrap().next().is_none());
        assert_eq!(store.manifest().display_name, "Original");
    }

    #[cfg(unix)]
    #[test]
    fn create_binds_the_package_to_the_parent_directory_inode_before_opening_root() {
        let temp = tempfile::tempdir().unwrap();
        let parent = temp.path().join("Library");
        let moved_parent = temp.path().join("MovedLibrary");
        let root = parent.join("Original.rustscanproject");
        fs::create_dir(&parent).unwrap();

        let store = ProjectStore::create_with_parent_open_hook(
            &root,
            ProjectCreateRequest::new("Original", SourceSpec::managed_images("source-a")),
            || {
                assert!(!root.exists());
                fs::rename(&parent, &moved_parent).unwrap();
                fs::create_dir(&parent).unwrap();
                fs::create_dir(parent.join("Original.rustscanproject")).unwrap();
            },
        )
        .unwrap();

        assert!(moved_parent
            .join("Original.rustscanproject/project.json")
            .is_file());
        assert!(fs::read_dir(parent.join("Original.rustscanproject"))
            .unwrap()
            .next()
            .is_none());
        assert_eq!(store.manifest().display_name, "Original");
    }

    #[cfg(unix)]
    #[test]
    fn typed_manifest_update_stays_bound_to_the_locked_package_inode() {
        use std::os::unix::fs::MetadataExt;

        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("Original.rustscanproject");
        let moved = temp.path().join("Moved.rustscanproject");
        let mut store = ProjectStore::create(
            &root,
            ProjectCreateRequest::new("Original", SourceSpec::managed_images("source-a")),
        )
        .unwrap();
        fs::rename(&root, &moved).unwrap();
        fs::create_dir(&root).unwrap();
        let replacement_manifest = b"replacement package must remain untouched";
        fs::write(root.join(MANIFEST_NAME), replacement_manifest).unwrap();

        let mut config = store.manifest().import_config.clone();
        config.video_keyframes_per_second += 1.0;
        store.update_import_config(config.clone()).unwrap();

        assert_eq!(
            fs::read(root.join(MANIFEST_NAME)).unwrap(),
            replacement_manifest
        );
        let persisted: ProjectManifest =
            serde_json::from_slice(&fs::read(moved.join(MANIFEST_NAME)).unwrap()).unwrap();
        assert_eq!(persisted.import_config, config);
        assert_eq!(persisted.id(), store.manifest().id());
        assert_eq!(
            fs::metadata(moved.join(LOCK_NAME)).unwrap().ino(),
            store._lock_file.metadata().unwrap().ino()
        );
    }

    #[cfg(unix)]
    #[test]
    fn open_validates_artifacts_from_the_locked_package_directory_inode() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("Original.rustscanproject");
        let moved = temp.path().join("Moved.rustscanproject");
        let store = ProjectStore::create(
            &root,
            ProjectCreateRequest::new("Original", SourceSpec::managed_images("source-a")),
        )
        .unwrap();
        let relative = "Artifacts/import/attempt-00000001/result.bin";
        let original = b"original";
        fs::create_dir_all(root.join(relative).parent().unwrap()).unwrap();
        fs::write(root.join(relative), original).unwrap();
        let mut manifest = store.manifest().clone();
        manifest.stage_mut(ProjectStage::Import).artifacts = vec![ArtifactRef {
            relative_path: relative.to_owned(),
            content_hash: blake3::hash(original).to_hex().to_string(),
            byte_len: original.len() as u64,
        }];
        store
            .write_manifest_atomic_with_parent_sync(&manifest, sync_parent_directory)
            .unwrap();
        drop(store);

        let opened = ProjectStore::open_with_validation_hook(&root, || {
            fs::rename(&root, &moved).unwrap();
            fs::create_dir(&root).unwrap();
            fs::create_dir_all(root.join(relative).parent().unwrap()).unwrap();
            fs::write(root.join(relative), b"replaced").unwrap();
        });

        let store = opened.expect("validation must stay bound to the original package inode");
        assert_eq!(store.manifest().id(), manifest.id());
        assert_eq!(
            fs::read(moved.join(relative)).unwrap(),
            original,
            "the original package artifact changed"
        );
    }
}
