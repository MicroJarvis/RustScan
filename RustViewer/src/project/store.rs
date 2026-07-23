use std::collections::BTreeSet;
use std::ffi::OsStr;
#[cfg(all(
    test,
    any(
        target_vendor = "apple",
        target_os = "android",
        target_os = "linux",
        target_os = "redox",
    )
))]
use std::fs::OpenOptions;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};

#[cfg(unix)]
use fs2::FileExt;
#[cfg(unix)]
use rustix::fs::{self as rustix_fs, FileType, Mode, OFlags};
#[cfg(unix)]
use rustix::io::Errno;
use thiserror::Error;
use uuid::Uuid;

use super::state::ValidatedArtifacts;
use super::{
    artifacts, events, library, source, ArtifactRef, ChangeKind, ImportConfigSnapshot,
    PnpConfigSnapshot, ProjectErrorRecord, ProjectLease, ProjectLibraryError, ProjectManifest,
    ProjectManifestValidationError, ProjectStage, ProjectStateError, SfmConfigSnapshot, SourceKind,
    SourceOwnership, SourceSpec, StageState, SuggestedAction, PROJECT_SCHEMA_VERSION,
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

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommitFailpoint {
    None,
    AfterWorkspaceSync,
    AfterAttemptRename,
    BeforeManifestWrite,
}

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
    #[error(
        "project storage requires Unix descriptor-relative filesystem operations and native flock"
    )]
    UnsupportedPlatform,
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
    #[error("project already has an active lease for {stage:?}")]
    StageLeaseActive { stage: ProjectStage },
    #[error("project lease belongs to {found:?}, not {expected:?}")]
    StageLeaseMismatch {
        expected: ProjectStage,
        found: ProjectStage,
    },
    #[error("stage workspace {found_stage:?} attempt {found_attempt} does not match {expected_stage:?} attempt {expected_attempt}")]
    WorkspaceLeaseMismatch {
        expected_stage: ProjectStage,
        expected_attempt: u32,
        found_stage: ProjectStage,
        found_attempt: u32,
    },
    #[error("artifact commit failed: {detail}")]
    ArtifactCommit { detail: String },
    #[error("project stage state update failed: {0}")]
    State(#[from] ProjectStateError),
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
    EventAppendFailed {
        path: PathBuf,
        error_kind: io::ErrorKind,
        detail: String,
    },
    ReferencedSourceUnavailable {
        path: PathBuf,
        error_kind: io::ErrorKind,
        detail: String,
    },
    ReferencedSourceChanged {
        expected: String,
        found: String,
    },
}

#[derive(Debug)]
pub struct ProjectStore {
    root: PathBuf,
    manifest: ProjectManifest,
    warnings: Vec<ProjectStoreWarning>,
    root_directory: File,
    _lock_file: File,
    // Must drop last so the package-inode writer lock outlives all other file handles.
    _writer_directory_lock: File,
}

#[cfg_attr(not(test), allow(dead_code))]
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
        require_supported_platform()?;
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
        let writer_directory_lock = initialization.root_directory().try_clone()?;
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
            _writer_directory_lock: writer_directory_lock,
        })
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self, ProjectStoreError> {
        Self::open_with_hooks(path, || {}, || {})
    }

    #[cfg(test)]
    fn open_with_validation_hook(
        path: impl AsRef<Path>,
        before_artifact_validation: impl FnOnce(),
    ) -> Result<Self, ProjectStoreError> {
        Self::open_with_hooks(path, || {}, before_artifact_validation)
    }

    #[cfg(test)]
    fn open_with_parent_open_hook(
        path: impl AsRef<Path>,
        after_parent_open: impl FnOnce(),
    ) -> Result<Self, ProjectStoreError> {
        Self::open_with_hooks(path, after_parent_open, || {})
    }

    fn open_with_hooks(
        path: impl AsRef<Path>,
        after_parent_open: impl FnOnce(),
        before_artifact_validation: impl FnOnce(),
    ) -> Result<Self, ProjectStoreError> {
        require_supported_platform()?;
        let path = path.as_ref();
        require_package_suffix(path)?;
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
        let root_directory =
            open_existing_package_root(&package_parent_directory, root_name, &root)?;
        let lock_file = create_and_lock(&root, &root_directory, None)?;
        let writer_directory_lock = root_directory.try_clone()?;
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

        let mut store = Self {
            root,
            manifest,
            warnings: Vec::new(),
            root_directory,
            _lock_file: lock_file,
            _writer_directory_lock: writer_directory_lock,
        };
        store.recover_interrupted_stage()?;
        before_artifact_validation();
        if let Some(warning) = store.referenced_source_warning() {
            store.warnings.push(warning);
        }
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

    pub fn duplicate(
        &self,
        destination: impl AsRef<Path>,
    ) -> Result<super::ProjectSummary, ProjectLibraryError> {
        library::duplicate_package(
            &self.root_directory,
            &self.root,
            &self.manifest,
            destination,
        )
    }

    pub fn reveal_path(&self) -> Result<PathBuf, ProjectLibraryError> {
        library::reveal_path(&self.root)
    }

    pub fn delete(self, confirmation_id: Uuid) -> Result<(), ProjectLibraryError> {
        self.delete_with_before_deletion(confirmation_id, || {})
    }

    fn delete_with_before_deletion(
        self,
        confirmation_id: Uuid,
        before_deletion: impl FnOnce(),
    ) -> Result<(), ProjectLibraryError> {
        if self.manifest.id() != confirmation_id {
            return Err(ProjectLibraryError::DeleteConfirmationMismatch {
                expected: self.manifest.id(),
                provided: confirmation_id,
            });
        }
        let target = library::prepare_delete(&self.root_directory, &self.root)?;
        target.release_root_directory_lock()?;
        drop(self);
        before_deletion();
        library::delete_prepared(target)
    }

    pub fn take_warnings(&mut self) -> Vec<ProjectStoreWarning> {
        std::mem::take(&mut self.warnings)
    }

    pub fn update_source(&mut self, source: SourceSpec) -> Result<(), ProjectStoreError> {
        require_supported_platform()?;
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
        require_supported_platform()?;
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
        require_supported_platform()?;
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
        require_supported_platform()?;
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
        require_supported_platform()?;
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
        require_supported_platform()?;
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

    fn referenced_source_warning(&self) -> Option<ProjectStoreWarning> {
        let source = &self.manifest.source;
        if source.kind != SourceKind::ImageSequence
            || source.ownership != SourceOwnership::Referenced
        {
            return None;
        }
        match source.bookmark.as_deref() {
            Some(bytes) => match source::SourceBookmark::decode(bytes)
                .map_err(source::SourceBookmarkError::from)
                .and_then(|bookmark| {
                    bookmark.with_resolved_paths(|paths| {
                        self.referenced_source_warning_for_paths(paths)
                    })
                }) {
                Ok(warning) => warning,
                Err(error) => Some(ProjectStoreWarning::ReferencedSourceUnavailable {
                    path: self.root.join("project.json"),
                    error_kind: io::ErrorKind::InvalidData,
                    detail: format!("referenced source bookmark is invalid: {error}"),
                }),
            },
            None => self.referenced_source_warning_for_paths(&source.display_paths),
        }
    }

    fn referenced_source_warning_for_paths(
        &self,
        display_paths: &[String],
    ) -> Option<ProjectStoreWarning> {
        let mut entries = Vec::with_capacity(display_paths.len());
        for display_path in display_paths {
            let requested = PathBuf::from(display_path);
            let canonical = match fs::canonicalize(&requested) {
                Ok(path) => path,
                Err(error) => {
                    return Some(ProjectStoreWarning::ReferencedSourceUnavailable {
                        path: requested,
                        error_kind: error.kind(),
                        detail: error.to_string(),
                    });
                }
            };
            let metadata = match fs::metadata(&canonical) {
                Ok(metadata) => metadata,
                Err(error) => {
                    return Some(ProjectStoreWarning::ReferencedSourceUnavailable {
                        path: canonical,
                        error_kind: error.kind(),
                        detail: error.to_string(),
                    });
                }
            };
            if !metadata.is_file() {
                return Some(ProjectStoreWarning::ReferencedSourceUnavailable {
                    path: canonical,
                    error_kind: io::ErrorKind::InvalidInput,
                    detail: "source is not a regular file".to_owned(),
                });
            }
            let name = match canonical.file_name().and_then(OsStr::to_str) {
                Some(name) => name.to_owned(),
                None => {
                    return Some(ProjectStoreWarning::ReferencedSourceUnavailable {
                        path: canonical,
                        error_kind: io::ErrorKind::InvalidData,
                        detail: "source name is not valid UTF-8".to_owned(),
                    });
                }
            };
            entries.push((name, canonical));
        }
        let found = match source::image_sequence_identity(&entries) {
            Ok(identity) => identity,
            Err(error) => {
                return Some(ProjectStoreWarning::ReferencedSourceUnavailable {
                    path: self.root.clone(),
                    error_kind: error.kind(),
                    detail: error.to_string(),
                });
            }
        };
        if found != self.manifest.source.identity {
            return Some(ProjectStoreWarning::ReferencedSourceChanged {
                expected: self.manifest.source.identity.clone(),
                found,
            });
        }
        None
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

    pub(crate) fn begin_stage(
        &mut self,
        stage: ProjectStage,
    ) -> Result<artifacts::StageWorkspace, ProjectStoreError> {
        self.begin_stage_with_workspace_creator(stage, artifacts::create_workspace)
    }

    pub(crate) fn restart_from_stage(
        &mut self,
        stage: ProjectStage,
    ) -> Result<(), ProjectStoreError> {
        if let Some(lease) = self.manifest.lease() {
            return Err(ProjectStoreError::StageLeaseActive { stage: lease.stage });
        }
        let change = match stage {
            ProjectStage::Import => super::ChangeKind::Source,
            ProjectStage::KeyframeSfm => super::ChangeKind::SfmConfig,
            ProjectStage::FullFramePnp => super::ChangeKind::PnpConfig,
            ProjectStage::Training => super::ChangeKind::TrainingConfig,
            ProjectStage::Complete => return Ok(()),
        };
        let mut next = self.manifest.clone();
        next.invalidate(change);
        let attempt = next.stage(stage).attempt();
        self.persist_manifest(&next, Some(("restart_requested", stage, attempt)))
    }

    fn begin_stage_with_workspace_creator(
        &mut self,
        stage: ProjectStage,
        create_workspace: impl FnOnce(
            &File,
            &Path,
            ProjectStage,
            u32,
        ) -> Result<
            artifacts::StageWorkspace,
            artifacts::ArtifactCommitError,
        >,
    ) -> Result<artifacts::StageWorkspace, ProjectStoreError> {
        require_supported_platform()?;
        if let Some(lease) = self.manifest.lease() {
            return Err(ProjectStoreError::StageLeaseActive { stage: lease.stage });
        }

        let mut next = self.manifest.clone();
        match next.stage(stage).state() {
            StageState::Stale => next.transition(stage, StageState::Ready)?,
            StageState::Ready | StageState::Paused | StageState::Cancelled | StageState::Failed => {
            }
            _ => {
                return Err(ProjectStoreError::State(
                    ProjectStateError::IllegalTransition {
                        stage,
                        from: next.stage(stage).state(),
                        to: StageState::Running,
                    },
                ));
            }
        }
        next.transition(stage, StageState::Queued)?;
        next.transition(stage, StageState::Running)?;
        let attempt = next.stage(stage).attempt();
        next.lease = Some(ProjectLease {
            project_id: next.id(),
            stage,
            attempt,
            process_id: std::process::id(),
            started_unix_ms: super::manifest::unix_time_ms(),
        });
        next.validate()?;

        self.persist_manifest(&next, Some(("began", stage, attempt)))?;
        match create_workspace(&self.root_directory, &self.root, stage, attempt) {
            Ok(workspace) => Ok(workspace),
            Err(error) => {
                let workspace_may_exist = matches!(
                    &error,
                    artifacts::ArtifactCommitError::WorkspaceCreationUncertain { .. }
                );
                let creation_error = Self::artifact_commit_error(error);
                if workspace_may_exist {
                    return Err(creation_error);
                }
                let mut failed = self.manifest.clone();
                failed.transition(stage, StageState::Failed)?;
                failed.stage_mut(stage).error = Some(ProjectErrorRecord {
                    code: "workspace_create_failed".to_owned(),
                    stage,
                    summary: "Stage workspace creation failed".to_owned(),
                    detail: creation_error.to_string(),
                    frame_id: None,
                    pair: None,
                    retryable: true,
                    suggested_actions: vec![SuggestedAction::Retry],
                });
                failed.lease = None;
                self.persist_manifest(&failed, Some(("failed", stage, attempt)))?;
                Err(creation_error)
            }
        }
    }

    pub(crate) fn request_stage_pause(
        &mut self,
        stage: ProjectStage,
    ) -> Result<(), ProjectStoreError> {
        self.transition_active_stage(stage, StageState::PauseRequested, "pause_requested", false)
    }

    pub(crate) fn record_stage_progress(
        &mut self,
        stage: ProjectStage,
        completed: Option<u64>,
        total: Option<u64>,
    ) -> Result<(), ProjectStoreError> {
        let lease = self.require_stage_lease(stage)?.clone();
        let mut next = self.manifest.clone();
        let now = super::manifest::unix_time_ms();
        let record = next.stage_mut(stage);
        record.completed = completed;
        record.total = total;
        record.updated_unix_ms = now;
        next.updated_unix_ms = now;
        self.persist_manifest(&next, Some(("progress", stage, lease.attempt)))
    }

    pub(crate) fn request_stage_cancel(
        &mut self,
        stage: ProjectStage,
    ) -> Result<(), ProjectStoreError> {
        self.transition_active_stage(
            stage,
            StageState::CancelRequested,
            "cancel_requested",
            false,
        )
    }

    pub(crate) fn mark_stage_paused(
        &mut self,
        stage: ProjectStage,
    ) -> Result<(), ProjectStoreError> {
        self.transition_active_stage(stage, StageState::Paused, "paused", true)
    }

    pub(crate) fn mark_stage_cancelled(
        &mut self,
        stage: ProjectStage,
    ) -> Result<(), ProjectStoreError> {
        self.transition_active_stage(stage, StageState::Cancelled, "cancelled", true)
    }

    pub(crate) fn mark_stage_failed(
        &mut self,
        stage: ProjectStage,
        error: ProjectErrorRecord,
    ) -> Result<(), ProjectStoreError> {
        if error.stage != stage {
            return Err(ProjectStoreError::State(
                ProjectStateError::IllegalTransition {
                    stage,
                    from: self.manifest.stage(stage).state(),
                    to: StageState::Failed,
                },
            ));
        }
        let lease = self.require_stage_lease(stage)?.clone();
        self.recover_active_stage_content(&lease)?;
        let mut next = self.manifest.clone();
        next.transition(stage, StageState::Failed)?;
        next.stage_mut(stage).error = Some(error);
        next.lease = None;
        self.persist_manifest(&next, Some(("failed", stage, lease.attempt)))
    }

    pub(crate) fn mark_stage_preflight_failed(
        &mut self,
        stage: ProjectStage,
        error: ProjectErrorRecord,
    ) -> Result<(), ProjectStoreError> {
        if error.stage != stage {
            return Err(ProjectStoreError::State(
                ProjectStateError::IllegalTransition {
                    stage,
                    from: self.manifest.stage(stage).state(),
                    to: StageState::Failed,
                },
            ));
        }
        if let Some(lease) = self.manifest.lease() {
            return Err(ProjectStoreError::StageLeaseActive { stage: lease.stage });
        }

        let mut next = self.manifest.clone();
        match next.stage(stage).state() {
            StageState::Ready => next.transition(stage, StageState::Failed)?,
            StageState::Stale => {
                next.transition(stage, StageState::Ready)?;
                next.transition(stage, StageState::Failed)?;
            }
            StageState::Failed => {}
            StageState::Succeeded => return Ok(()),
            from => {
                return Err(ProjectStoreError::State(
                    ProjectStateError::IllegalTransition {
                        stage,
                        from,
                        to: StageState::Failed,
                    },
                ));
            }
        }
        let attempt = next.stage(stage).attempt();
        next.stage_mut(stage).error = Some(error);
        self.persist_manifest(&next, Some(("preflight_failed", stage, attempt)))
    }

    pub(crate) fn validate_stage_payloads(
        &self,
        workspace: &artifacts::StageWorkspace,
        payloads: &[artifacts::StagedArtifact],
    ) -> Result<(), ProjectStoreError> {
        artifacts::validate_and_sync_workspace_payloads(&self.root_directory, workspace, payloads)
            .map_err(Self::artifact_commit_error)
    }

    pub(crate) fn commit_stage_success(
        &mut self,
        workspace: &artifacts::StageWorkspace,
        declarations: &[artifacts::StagedArtifact],
        strict: bool,
    ) -> Result<(), ProjectStoreError> {
        self.commit_stage_success_with_hooks(workspace, declarations, strict, |_| Ok(()), || Ok(()))
    }

    #[cfg(test)]
    fn commit_stage_success_with_failpoint(
        &mut self,
        workspace: &artifacts::StageWorkspace,
        declarations: &[artifacts::StagedArtifact],
        strict: bool,
        failpoint: CommitFailpoint,
    ) -> Result<(), ProjectStoreError> {
        self.commit_stage_success_with_hooks(
            workspace,
            declarations,
            strict,
            |phase| {
                let inject = matches!(
                    (failpoint, phase),
                    (
                        CommitFailpoint::AfterWorkspaceSync,
                        artifacts::CommitPhase::AfterWorkspaceSync
                    ) | (
                        CommitFailpoint::AfterAttemptRename,
                        artifacts::CommitPhase::AfterAttemptRename
                    )
                );
                if inject {
                    Err(artifacts::ArtifactCommitError::Io(io::Error::other(
                        "injected artifact commit failure",
                    )))
                } else {
                    Ok(())
                }
            },
            || {
                if failpoint == CommitFailpoint::BeforeManifestWrite {
                    Err(artifacts::ArtifactCommitError::Io(io::Error::other(
                        "injected manifest commit failure",
                    )))
                } else {
                    Ok(())
                }
            },
        )
    }

    fn commit_stage_success_with_hooks(
        &mut self,
        workspace: &artifacts::StageWorkspace,
        declarations: &[artifacts::StagedArtifact],
        strict: bool,
        phase_hook: impl FnMut(artifacts::CommitPhase) -> Result<(), artifacts::ArtifactCommitError>,
        before_manifest: impl FnOnce() -> Result<(), artifacts::ArtifactCommitError>,
    ) -> Result<(), ProjectStoreError> {
        let lease = self.require_stage_lease(workspace.stage())?.clone();
        if lease.attempt != workspace.attempt() {
            return Err(ProjectStoreError::WorkspaceLeaseMismatch {
                expected_stage: lease.stage,
                expected_attempt: lease.attempt,
                found_stage: workspace.stage(),
                found_attempt: workspace.attempt(),
            });
        }
        let artifacts = artifacts::commit_workspace(
            &self.root_directory,
            workspace,
            declarations,
            strict,
            phase_hook,
        )
        .map_err(Self::artifact_commit_error)?;
        before_manifest().map_err(Self::artifact_commit_error)?;

        let mut next = self.manifest.clone();
        let validated = ValidatedArtifacts::try_new(artifacts).map_err(|error| {
            ProjectStoreError::ArtifactCommit {
                detail: error.to_string(),
            }
        })?;
        next.commit_stage_success(lease.stage, validated)?;
        next.lease = None;
        self.persist_manifest(&next, Some(("succeeded", lease.stage, lease.attempt)))
    }

    fn transition_active_stage(
        &mut self,
        stage: ProjectStage,
        target: StageState,
        event: &'static str,
        clear_lease: bool,
    ) -> Result<(), ProjectStoreError> {
        let lease = self.require_stage_lease(stage)?.clone();
        let mut next = self.manifest.clone();
        next.transition(stage, target)?;
        if clear_lease {
            next.lease = None;
        }
        self.persist_manifest(&next, Some((event, stage, lease.attempt)))
    }

    fn require_stage_lease(&self, stage: ProjectStage) -> Result<&ProjectLease, ProjectStoreError> {
        let lease = self
            .manifest
            .lease()
            .ok_or(ProjectStoreError::StageLeaseMismatch {
                expected: stage,
                found: stage,
            })?;
        if lease.stage != stage {
            return Err(ProjectStoreError::StageLeaseMismatch {
                expected: stage,
                found: lease.stage,
            });
        }
        Ok(lease)
    }

    fn persist_manifest(
        &mut self,
        manifest: &ProjectManifest,
        event: Option<(&'static str, ProjectStage, u32)>,
    ) -> Result<(), ProjectStoreError> {
        let warning = self.write_manifest_atomic_with_parent_sync(manifest, |_| Ok(()))?;
        self.manifest = manifest.clone();
        self.warnings.extend(warning);
        if let Some((kind, stage, attempt)) = event {
            self.append_event_nonfatal(kind, stage, attempt);
        }
        Ok(())
    }

    fn artifact_commit_error(error: artifacts::ArtifactCommitError) -> ProjectStoreError {
        ProjectStoreError::ArtifactCommit {
            detail: error.to_string(),
        }
    }

    pub(crate) fn recover_interrupted_stage(&mut self) -> Result<(), ProjectStoreError> {
        let Some(lease) = self.manifest.lease().cloned() else {
            return Ok(());
        };
        self.recover_active_stage_content(&lease)?;

        let mut recovered = self.manifest.clone();
        recovered.transition(lease.stage, StageState::Failed)?;
        recovered.stage_mut(lease.stage).error = Some(ProjectErrorRecord {
            code: "interrupted".to_owned(),
            stage: lease.stage,
            summary: "Stage interrupted while the project was closed".to_owned(),
            detail: "The unfinished workspace was moved to Logs/recovery.".to_owned(),
            frame_id: None,
            pair: None,
            retryable: true,
            suggested_actions: vec![SuggestedAction::OpenLog, SuggestedAction::Retry],
        });
        recovered.lease = None;
        let warning = self.write_manifest_atomic_with_parent_sync(&recovered, |_| Ok(()))?;
        self.manifest = recovered;
        self.warnings.extend(warning);
        self.append_event_nonfatal("recovered_interrupted", lease.stage, lease.attempt);
        Ok(())
    }

    fn recover_active_stage_content(&self, lease: &ProjectLease) -> Result<(), ProjectStoreError> {
        let referenced_artifacts = self
            .manifest_artifacts()
            .map(|artifact| PathBuf::from(&artifact.relative_path))
            .collect::<BTreeSet<_>>();
        artifacts::recover_interrupted_attempts(
            &self.root_directory,
            lease.stage,
            lease.attempt,
            &referenced_artifacts,
        )?;
        Ok(())
    }

    fn manifest_artifacts(&self) -> impl Iterator<Item = &ArtifactRef> {
        ProjectStage::ORDER
            .into_iter()
            .flat_map(|stage| self.manifest.stage(stage).artifacts().iter())
            .chain(self.manifest.active_scene.iter())
            .chain(self.manifest.final_scene.iter())
    }

    fn append_event_nonfatal(&mut self, kind: &'static str, stage: ProjectStage, attempt: u32) {
        let event =
            events::ProjectEvent::new(kind, stage, attempt, super::manifest::unix_time_ms());
        if let Err(error) = events::append(&self.root_directory, &event) {
            self.warnings.push(ProjectStoreWarning::EventAppendFailed {
                path: self.root.join("Logs/events.jsonl"),
                error_kind: error.kind(),
                detail: error.to_string(),
            });
        }
    }
}

fn require_supported_platform() -> Result<(), ProjectStoreError> {
    platform_support(cfg!(any(
        target_vendor = "apple",
        target_os = "android",
        target_os = "linux",
        target_os = "redox",
    )))
}

fn platform_support(supported: bool) -> Result<(), ProjectStoreError> {
    if supported {
        Ok(())
    } else {
        Err(ProjectStoreError::UnsupportedPlatform)
    }
}

#[cfg(unix)]
fn create_and_lock(
    root: &Path,
    root_directory: &File,
    cleanup: Option<&mut InitializationCleanup>,
) -> Result<File, ProjectStoreError> {
    let lock_path = root.join(LOCK_NAME);
    // The stable package inode is the outer writer lock. A lock-file inode alone can be
    // unlinked and replaced while still locked, admitting a second writer on the replacement.
    match root_directory.try_lock_exclusive() {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
            return Err(ProjectStoreError::AlreadyOpen { path: lock_path });
        }
        Err(error) => return Err(error.into()),
    }
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
                    OFlags::RDWR | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
                    Mode::empty(),
                )
                .map_err(io::Error::from)?,
            ),
            false,
        ),
        Err(error) => return Err(io::Error::from(error).into()),
    };
    if !file.metadata()?.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "project lock is not a regular file",
        )
        .into());
    }
    match file.try_lock_exclusive() {
        Ok(()) => {
            let opened_metadata = rustix_fs::fstat(&file).map_err(io::Error::from)?;
            let entry_metadata = rustix_fs::statat(
                root_directory,
                LOCK_NAME,
                rustix_fs::AtFlags::SYMLINK_NOFOLLOW,
            )
            .map_err(io::Error::from)?;
            if opened_metadata.st_dev != entry_metadata.st_dev
                || opened_metadata.st_ino != entry_metadata.st_ino
            {
                return Err(ProjectStoreError::AlreadyOpen { path: lock_path });
            }
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

#[cfg(not(unix))]
fn create_and_lock(
    _root: &Path,
    _root_directory: &File,
    _cleanup: Option<&mut InitializationCleanup>,
) -> Result<File, ProjectStoreError> {
    Err(ProjectStoreError::UnsupportedPlatform)
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
        {
            let _ = (root_directory, relative);
            return Err(unsupported_platform_io());
        }
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
                let _ = (root_directory, entry);
            }
        }

        self.disarm();
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

#[cfg(unix)]
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

#[cfg(unix)]
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
    let mut temporary_created = false;
    let before_rename = (|| {
        let mut file = File::from(
            rustix_fs::openat(
                root_directory,
                temporary_name.as_str(),
                OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::RUSR | Mode::WUSR,
            )
            .map_err(io::Error::from)?,
        );
        temporary_created = true;
        file.write_all(bytes)?;
        file.sync_all()?;
        before_rename(&temporary)?;
        rustix_fs::renameat(
            root_directory,
            temporary_name.as_str(),
            root_directory,
            destination_name,
        )
        .map_err(io::Error::from)?;
        Ok(())
    })();
    if let Err(operation_error) = before_rename {
        if !temporary_created {
            return Err(operation_error);
        }
        let cleanup_result = rustix_fs::unlinkat(
            root_directory,
            temporary_name.as_str(),
            rustix_fs::AtFlags::empty(),
        )
        .map_err(io::Error::from);
        if let Err(cleanup_error) = cleanup_result {
            return Err(io::Error::new(
                cleanup_error.kind(),
                format!(
                    "pre-rename operation failed: {operation_error}; temporary file cleanup failed: {cleanup_error}"
                ),
            ));
        }
        return Err(operation_error);
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

#[cfg(not(unix))]
fn write_bytes_atomic_in_directory_with_hooks(
    _root_directory: &File,
    _root: &Path,
    _destination_name: &str,
    _bytes: &[u8],
    _sync_parent: impl FnOnce(&Path) -> io::Result<()>,
    _before_rename: impl FnOnce(&Path) -> io::Result<()>,
) -> io::Result<Option<ProjectStoreWarning>> {
    Err(unsupported_platform_io())
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
fn open_directory_at(_root_directory: &File, _relative: &Path, _label: &Path) -> io::Result<File> {
    Err(unsupported_platform_io())
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

#[cfg(unix)]
fn open_existing_package_root(
    package_parent_directory: &File,
    name: &OsStr,
    root_label: &Path,
) -> Result<File, ProjectStoreError> {
    let metadata = rustix_fs::statat(
        package_parent_directory,
        name,
        rustix_fs::AtFlags::SYMLINK_NOFOLLOW,
    )
    .map_err(io::Error::from)?;
    let file_type = FileType::from_raw_mode(metadata.st_mode);
    if file_type.is_symlink() {
        return Err(ProjectStoreError::SymlinkPackageRoot {
            path: root_label.to_path_buf(),
        });
    }
    if !file_type.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotADirectory,
            "project package root is not a directory",
        )
        .into());
    }
    Ok(open_directory_at(
        package_parent_directory,
        Path::new(name),
        root_label,
    )?)
}

#[cfg(not(unix))]
fn open_or_create_package_root(
    _package_parent_directory: &File,
    _name: &OsStr,
    _root_label: &Path,
) -> Result<(File, bool), ProjectStoreError> {
    Err(ProjectStoreError::UnsupportedPlatform)
}

#[cfg(not(unix))]
fn open_existing_package_root(
    _package_parent_directory: &File,
    _name: &OsStr,
    _root_label: &Path,
) -> Result<File, ProjectStoreError> {
    Err(ProjectStoreError::UnsupportedPlatform)
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
    Err(unsupported_platform_io())
}

#[cfg(not(unix))]
fn open_package_directory(_root: &Path) -> io::Result<File> {
    Err(unsupported_platform_io())
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
fn read_manifest_from_directory(_root_directory: &File, _root: &Path) -> io::Result<Vec<u8>> {
    Err(unsupported_platform_io())
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
    _root: &Path,
    _relative: &Path,
) -> Result<File, ProjectStoreError> {
    Err(ProjectStoreError::UnsupportedPlatform)
}

#[cfg(not(unix))]
fn unsupported_platform_io() -> io::Error {
    io::Error::new(
        io::ErrorKind::Unsupported,
        ProjectStoreError::UnsupportedPlatform,
    )
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

#[cfg(all(
    test,
    any(
        target_vendor = "apple",
        target_os = "android",
        target_os = "linux",
        target_os = "redox",
    )
))]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::{BTreeSet, VecDeque};
    use std::process::Command;
    use std::rc::Rc;

    #[test]
    fn create_and_lock_rejects_a_non_regular_existing_lock_file() {
        let temp = tempfile::tempdir().unwrap();
        let lock_path = temp.path().join(LOCK_NAME);
        assert!(Command::new("mkfifo")
            .arg(&lock_path)
            .status()
            .unwrap()
            .success());
        let root_directory = open_package_directory(temp.path()).unwrap();

        let result = create_and_lock(temp.path(), &root_directory, None);

        assert!(matches!(
            result,
            Err(ProjectStoreError::Io(error)) if error.kind() == io::ErrorKind::InvalidData
        ));
    }

    #[cfg(any(
        target_vendor = "apple",
        target_os = "android",
        target_os = "linux",
        target_os = "redox",
    ))]
    #[test]
    fn writer_directory_lock_survives_operational_directory_handle_drop() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("Locked.rustscanproject");
        let store = ProjectStore::create(
            &root,
            ProjectCreateRequest::new("Locked", SourceSpec::managed_images("source-a")),
        )
        .unwrap();
        let lock_path = root.join(LOCK_NAME);
        fs::remove_file(&lock_path).unwrap();
        fs::write(&lock_path, b"").unwrap();

        let ProjectStore {
            root_directory,
            _lock_file,
            ..
        } = store;
        drop(root_directory);

        assert!(matches!(
            ProjectStore::open(&root),
            Err(ProjectStoreError::AlreadyOpen { .. })
        ));
        drop(_lock_file);
    }

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
    fn pre_rename_temp_cleanup_failure_is_reported_with_the_original_error() {
        let temp = tempfile::tempdir().unwrap();
        let root_directory = open_package_directory(temp.path()).unwrap();
        let moved_temporary = temp.path().join("moved-project.tmp");

        let error = write_bytes_atomic_in_directory_with_hooks(
            &root_directory,
            temp.path(),
            MANIFEST_NAME,
            b"replacement",
            |_| Ok(()),
            |temporary| {
                fs::rename(temporary, &moved_temporary).unwrap();
                Err(io::Error::other("injected pre-rename failure"))
            },
        )
        .unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        let detail = error.to_string();
        assert!(detail.contains("injected pre-rename failure"));
        assert!(detail.contains("temporary file cleanup failed"));
        assert!(moved_temporary.is_file());
    }

    #[cfg(unix)]
    #[test]
    fn temp_creation_failure_does_not_report_a_cleanup_failure() {
        let temp = tempfile::tempdir().unwrap();
        let root_directory = open_package_directory(temp.path()).unwrap();

        let error = write_bytes_atomic_in_directory_with_hooks(
            &root_directory,
            temp.path(),
            "missing/project.json",
            b"replacement",
            |_| Ok(()),
            |_| panic!("the hook must not run when temp creation fails"),
        )
        .unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert!(!error.to_string().contains("temporary file cleanup failed"));
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
    fn open_binds_the_package_to_the_parent_directory_inode_before_opening_root() {
        let temp = tempfile::tempdir().unwrap();
        let parent = temp.path().join("Library");
        let moved_parent = temp.path().join("MovedLibrary");
        let root = parent.join("Original.rustscanproject");
        fs::create_dir(&parent).unwrap();
        let created = ProjectStore::create(
            &root,
            ProjectCreateRequest::new("Original", SourceSpec::managed_images("source-a")),
        )
        .unwrap();
        drop(created);

        let store = ProjectStore::open_with_parent_open_hook(&root, || {
            fs::rename(&parent, &moved_parent).unwrap();
            fs::create_dir(&parent).unwrap();
            fs::create_dir(parent.join("Original.rustscanproject")).unwrap();
        })
        .unwrap();

        assert_eq!(store.manifest().display_name, "Original");
        assert!(moved_parent
            .join("Original.rustscanproject/project.json")
            .is_file());
        assert!(fs::read_dir(parent.join("Original.rustscanproject"))
            .unwrap()
            .next()
            .is_none());
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
        let import = manifest.stage_mut(ProjectStage::Import);
        import.attempt = 1;
        import.artifacts = vec![ArtifactRef {
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

    #[test]
    fn commit_failpoints_preserve_the_prior_attempt_for_recovery() {
        let _ = CommitFailpoint::None;
        for failpoint in [
            CommitFailpoint::AfterWorkspaceSync,
            CommitFailpoint::AfterAttemptRename,
            CommitFailpoint::BeforeManifestWrite,
        ] {
            let temp = tempfile::tempdir().unwrap();
            let root = temp.path().join("Failpoint.rustscanproject");
            let mut store = ProjectStore::create(
                &root,
                ProjectCreateRequest::new("Failpoint", SourceSpec::managed_images("source-a")),
            )
            .unwrap();
            let first = store.begin_stage(ProjectStage::Import).unwrap();
            fs::create_dir_all(first.path().join("Sources")).unwrap();
            fs::write(
                first.path().join("Sources/source.json"),
                br#"{"version":1}"#,
            )
            .unwrap();
            let declaration = artifacts::StagedArtifact::new(
                "Sources/source.json",
                artifacts::ArtifactValidationKind::Json,
            )
            .unwrap();
            store
                .commit_stage_success(&first, std::slice::from_ref(&declaration), true)
                .unwrap();
            let prior = store
                .manifest()
                .try_stage(ProjectStage::Import)
                .unwrap()
                .artifacts()
                .to_vec();
            let prior_path = root.join(&prior[0].relative_path);
            let prior_bytes = fs::read(&prior_path).unwrap();

            let mut config = store.manifest().import_config.clone();
            config.video_keyframes_per_second += 1.0;
            store.update_import_config(config).unwrap();
            let second = store.begin_stage(ProjectStage::Import).unwrap();
            fs::create_dir_all(second.path().join("Sources")).unwrap();
            fs::write(
                second.path().join("Sources/source.json"),
                br#"{"version":2}"#,
            )
            .unwrap();

            assert!(store
                .commit_stage_success_with_failpoint(
                    &second,
                    std::slice::from_ref(&declaration),
                    true,
                    failpoint,
                )
                .is_err());
            drop(store);

            let reopened = ProjectStore::open(&root).unwrap();
            assert_eq!(
                reopened
                    .manifest()
                    .try_stage(ProjectStage::Import)
                    .unwrap()
                    .artifacts(),
                prior
            );
            assert_eq!(fs::read(prior_path).unwrap(), prior_bytes);
            assert_eq!(
                reopened
                    .manifest()
                    .try_stage(ProjectStage::Import)
                    .unwrap()
                    .state(),
                StageState::Failed
            );
            assert!(fs::read_dir(root.join("Logs/recovery"))
                .unwrap()
                .next()
                .is_some());
        }
    }

    #[test]
    fn stage_controls_persist_authoritative_state_before_events() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("Controls.rustscanproject");
        let mut store = ProjectStore::create(
            &root,
            ProjectCreateRequest::new("Controls", SourceSpec::managed_images("source-a")),
        )
        .unwrap();

        store.begin_stage(ProjectStage::Import).unwrap();
        assert!(store.manifest().lease().is_some());
        let mut config = store.manifest().import_config.clone();
        config.video_keyframes_per_second += 1.0;
        assert!(matches!(
            store.update_import_config(config),
            Err(ProjectStoreError::StageActive {
                stage: ProjectStage::Import,
                change: ChangeKind::ImportConfig,
            })
        ));
        store.request_stage_pause(ProjectStage::Import).unwrap();
        assert_eq!(
            store.manifest().stage(ProjectStage::Import).state(),
            StageState::PauseRequested
        );
        store.mark_stage_paused(ProjectStage::Import).unwrap();
        assert_eq!(store.manifest().lease(), None);

        store.begin_stage(ProjectStage::Import).unwrap();
        store.request_stage_cancel(ProjectStage::Import).unwrap();
        store.mark_stage_cancelled(ProjectStage::Import).unwrap();
        assert_eq!(store.manifest().lease(), None);

        store.begin_stage(ProjectStage::Import).unwrap();
        store
            .mark_stage_failed(
                ProjectStage::Import,
                ProjectErrorRecord {
                    code: "worker_failed".to_owned(),
                    stage: ProjectStage::Import,
                    summary: "Worker failed".to_owned(),
                    detail: "Injected failure".to_owned(),
                    frame_id: None,
                    pair: None,
                    retryable: true,
                    suggested_actions: vec![SuggestedAction::Retry],
                },
            )
            .unwrap();
        assert_eq!(
            store.manifest().stage(ProjectStage::Import).state(),
            StageState::Failed
        );
        assert_eq!(store.manifest().lease(), None);
    }

    #[test]
    fn project_library_refuses_duplicate_while_a_stage_lease_is_active() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("Active.rustscanproject");
        let mut store = ProjectStore::create(
            &source,
            ProjectCreateRequest::new("Active", SourceSpec::managed_images("source-a")),
        )
        .unwrap();
        store.begin_stage(ProjectStage::Import).unwrap();

        assert!(matches!(
            store.duplicate(temp.path().join("Duplicate.rustscanproject")),
            Err(ProjectLibraryError::ActiveLease)
        ));
    }

    #[test]
    fn project_library_delete_preserves_a_package_replaced_after_lock_release() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("Delete.rustscanproject");
        let moved = temp.path().join("Original.rustscanproject");
        let store = ProjectStore::create(
            &root,
            ProjectCreateRequest::new("Delete", SourceSpec::managed_images("source-a")),
        )
        .unwrap();
        let id = store.manifest().id();
        let replacement = root.join("replacement-must-survive");

        let error = store
            .delete_with_before_deletion(id, || {
                let reopened = ProjectStore::open(&root)
                    .expect("all writer locks must be released before deletion");
                drop(reopened);
                fs::rename(&root, &moved).unwrap();
                fs::create_dir(&root).unwrap();
                fs::write(&replacement, b"replacement").unwrap();
            })
            .unwrap_err();

        assert!(error.to_string().contains("changed"));
        assert_eq!(fs::read(&replacement).unwrap(), b"replacement");
        assert!(moved.join(MANIFEST_NAME).is_file());
    }

    #[test]
    fn begin_stage_records_workspace_creation_failure_after_durable_lease() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("WorkspaceFailure.rustscanproject");
        let mut store = ProjectStore::create(
            &root,
            ProjectCreateRequest::new("Workspace Failure", SourceSpec::managed_images("source-a")),
        )
        .unwrap();

        let staging = root.join("Cache/.staging");
        fs::remove_dir(&staging).unwrap();
        fs::write(&staging, b"not a directory").unwrap();

        assert!(matches!(
            store.begin_stage(ProjectStage::Import),
            Err(ProjectStoreError::ArtifactCommit { .. })
        ));
        assert_eq!(
            store.manifest().stage(ProjectStage::Import).state(),
            StageState::Failed
        );
        assert_eq!(store.manifest().lease(), None);
        assert_eq!(
            store
                .manifest()
                .stage(ProjectStage::Import)
                .error()
                .unwrap()
                .code,
            "workspace_create_failed"
        );
        drop(store);

        let reopened = ProjectStore::open(&root).unwrap();
        assert_eq!(
            reopened.manifest().stage(ProjectStage::Import).state(),
            StageState::Failed
        );
        assert_eq!(reopened.manifest().lease(), None);
        drop(reopened);

        fs::remove_file(&staging).unwrap();
        fs::create_dir(&staging).unwrap();
        let mut retry = ProjectStore::open(&root).unwrap();
        let workspace = retry.begin_stage(ProjectStage::Import).unwrap();
        assert_eq!(workspace.attempt(), 2);
    }

    #[test]
    fn begin_stage_recovers_an_uncertain_workspace_creation_before_retrying() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("UncertainWorkspace.rustscanproject");
        let mut store = ProjectStore::create(
            &root,
            ProjectCreateRequest::new(
                "Uncertain Workspace",
                SourceSpec::managed_images("source-a"),
            ),
        )
        .unwrap();

        assert!(matches!(
            store.begin_stage_with_workspace_creator(
                ProjectStage::Import,
                |root_directory, root, stage, attempt| {
                    artifacts::create_workspace(root_directory, root, stage, attempt)?;
                    Err(artifacts::ArtifactCommitError::WorkspaceCreationUncertain {
                        source: io::Error::other("injected staging sync failure"),
                    })
                }
            ),
            Err(ProjectStoreError::ArtifactCommit { .. })
        ));
        assert_eq!(
            store.manifest().stage(ProjectStage::Import).state(),
            StageState::Running
        );
        assert!(store.manifest().lease().is_some());
        drop(store);

        let recovered = ProjectStore::open(&root).unwrap();
        assert_eq!(
            recovered.manifest().stage(ProjectStage::Import).state(),
            StageState::Failed
        );
        assert_eq!(recovered.manifest().lease(), None);
        assert!(fs::read_dir(root.join("Logs/recovery"))
            .unwrap()
            .filter_map(Result::ok)
            .any(|entry| entry.path().is_dir()));
        drop(recovered);

        let mut retry = ProjectStore::open(&root).unwrap();
        let workspace = retry.begin_stage(ProjectStage::Import).unwrap();
        assert_eq!(workspace.attempt(), 2);
    }

    #[test]
    fn stage_commit_streams_declared_files_into_one_immutable_attempt() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("Artifacts.rustscanproject");
        let mut store = ProjectStore::create(
            &root,
            ProjectCreateRequest::new("Artifacts", SourceSpec::managed_images("source-a")),
        )
        .unwrap();
        let workspace = store.begin_stage(ProjectStage::Import).unwrap();
        let payload = workspace.path().join("Sources/source.json");
        fs::create_dir_all(payload.parent().unwrap()).unwrap();
        fs::write(&payload, b"{not valid json").unwrap();
        let declaration = artifacts::StagedArtifact::new(
            "Sources/source.json",
            artifacts::ArtifactValidationKind::Json,
        )
        .unwrap();

        assert!(store
            .commit_stage_success(&workspace, std::slice::from_ref(&declaration), true)
            .is_err());
        fs::write(&payload, br#"{"source":"valid"}"#).unwrap();
        fs::write(workspace.path().join("undeclared.bin"), b"undeclared").unwrap();
        assert!(store
            .commit_stage_success(&workspace, std::slice::from_ref(&declaration), true)
            .is_err());
        fs::remove_file(workspace.path().join("undeclared.bin")).unwrap();
        assert!(store
            .commit_stage_success(
                &workspace,
                &[declaration.clone(), declaration.clone()],
                true,
            )
            .is_err());
        assert!(artifacts::StagedArtifact::new(
            "../outside.json",
            artifacts::ArtifactValidationKind::Json,
        )
        .is_err());

        store
            .commit_stage_success(&workspace, std::slice::from_ref(&declaration), true)
            .unwrap();
        let artifacts = store.manifest().stage(ProjectStage::Import).artifacts();
        assert_eq!(artifacts.len(), 1);
        assert_eq!(
            artifacts[0].relative_path,
            "Artifacts/import/attempt-00000001/Sources/source.json"
        );
        assert_eq!(
            fs::read(root.join(&artifacts[0].relative_path)).unwrap(),
            br#"{"source":"valid"}"#
        );
        assert!(!workspace.path().exists());
    }

    #[test]
    fn stage_commit_rehashes_the_final_attempt_after_payload_replacement() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("FinalValidation.rustscanproject");
        let mut store = ProjectStore::create(
            &root,
            ProjectCreateRequest::new("Final Validation", SourceSpec::managed_images("source-a")),
        )
        .unwrap();
        let workspace = store.begin_stage(ProjectStage::Import).unwrap();
        let payload = workspace.path().join("Sources/source.json");
        fs::create_dir_all(payload.parent().unwrap()).unwrap();
        fs::write(&payload, br#"{"source":"initial"}"#).unwrap();
        let declaration = artifacts::StagedArtifact::new(
            "Sources/source.json",
            artifacts::ArtifactValidationKind::Json,
        )
        .unwrap();
        let replacement = br#"{"source":"replaced after validation"}"#;
        let replacement_path = payload.clone();

        store
            .commit_stage_success_with_hooks(
                &workspace,
                std::slice::from_ref(&declaration),
                true,
                move |phase| {
                    if phase == artifacts::CommitPhase::AfterWorkspaceSync {
                        fs::remove_file(&replacement_path).unwrap();
                        fs::write(&replacement_path, replacement).unwrap();
                    }
                    Ok(())
                },
                || Ok(()),
            )
            .unwrap();

        let artifact = &store.manifest().stage(ProjectStage::Import).artifacts()[0];
        assert_eq!(artifact.byte_len, replacement.len() as u64);
        assert_eq!(
            artifact.content_hash,
            blake3::hash(replacement).to_hex().to_string()
        );
        assert_eq!(
            fs::read(root.join(&artifact.relative_path)).unwrap(),
            replacement
        );
    }

    #[test]
    fn artifact_commit_syncs_payload_ancestors_and_both_rename_parents() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("SyncCoverage.rustscanproject");
        let mut store = ProjectStore::create(
            &root,
            ProjectCreateRequest::new("Sync Coverage", SourceSpec::managed_images("source-a")),
        )
        .unwrap();
        let workspace = store.begin_stage(ProjectStage::Import).unwrap();
        let payload = workspace.path().join("Sources/nested/source.json");
        fs::create_dir_all(payload.parent().unwrap()).unwrap();
        fs::write(&payload, br#"{"source":"nested"}"#).unwrap();
        let declaration = artifacts::StagedArtifact::new(
            "Sources/nested/source.json",
            artifacts::ArtifactValidationKind::Json,
        )
        .unwrap();
        let synced = Rc::new(RefCell::new(Vec::new()));
        let observed = Rc::clone(&synced);

        artifacts::commit_workspace_with_test_sync_hook(
            &store.root_directory,
            &workspace,
            std::slice::from_ref(&declaration),
            true,
            |_| Ok(()),
            move |path| observed.borrow_mut().push(path.to_path_buf()),
        )
        .unwrap();

        let synced = synced.borrow();
        for expected in [
            PathBuf::from("Cache/.staging/import-1/Sources/nested/source.json"),
            PathBuf::from("Cache/.staging/import-1/Sources/nested"),
            PathBuf::from("Cache/.staging/import-1/Sources"),
            PathBuf::from("Cache/.staging/import-1"),
            PathBuf::from("Artifacts/import/attempt-00000001/Sources/nested/source.json"),
            PathBuf::from("Artifacts/import/attempt-00000001/Sources/nested"),
            PathBuf::from("Artifacts/import/attempt-00000001/Sources"),
            PathBuf::from("Artifacts/import/attempt-00000001"),
        ] {
            assert!(synced.contains(&expected), "missing sync for {expected:?}");
        }
        let staging_parent = synced
            .iter()
            .position(|path| path == Path::new("Cache/.staging"))
            .unwrap();
        let attempt_parent = synced
            .iter()
            .position(|path| path == Path::new("Artifacts/import"))
            .unwrap();
        assert!(staging_parent < attempt_parent);
    }

    #[test]
    fn recovery_retries_collision_without_replacing_and_syncs_both_parents() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("RecoveryCollision.rustscanproject");
        let store = ProjectStore::create(
            &root,
            ProjectCreateRequest::new("Recovery Collision", SourceSpec::managed_images("source-a")),
        )
        .unwrap();
        let abandoned = root.join("Cache/.staging/import-2/Sources/source.json");
        fs::create_dir_all(abandoned.parent().unwrap()).unwrap();
        fs::write(&abandoned, br#"{"source":"partial"}"#).unwrap();
        let collision = root.join("Logs/recovery/collision");
        fs::create_dir(&collision).unwrap();
        fs::write(collision.join("sentinel"), b"must survive").unwrap();
        let names = Rc::new(RefCell::new(VecDeque::from([
            "collision".to_owned(),
            "available".to_owned(),
        ])));
        let next_name = Rc::clone(&names);
        let synced = Rc::new(RefCell::new(Vec::new()));
        let observed = Rc::clone(&synced);

        artifacts::recover_interrupted_attempts_with_test_hooks(
            &store.root_directory,
            ProjectStage::Import,
            2,
            &BTreeSet::new(),
            move || next_name.borrow_mut().pop_front().unwrap(),
            move |path| observed.borrow_mut().push(path.to_path_buf()),
        )
        .unwrap();

        assert_eq!(
            fs::read(collision.join("sentinel")).unwrap(),
            b"must survive"
        );
        assert_eq!(
            fs::read(root.join("Logs/recovery/available/Sources/source.json")).unwrap(),
            br#"{"source":"partial"}"#
        );
        assert!(!abandoned.exists());
        assert_eq!(
            *synced.borrow(),
            vec![
                PathBuf::from("Cache/.staging"),
                PathBuf::from("Logs/recovery")
            ]
        );
    }

    #[test]
    fn event_append_failure_is_a_warning_after_stage_begin_is_persisted() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("EventWarning.rustscanproject");
        let mut store = ProjectStore::create(
            &root,
            ProjectCreateRequest::new("Event Warning", SourceSpec::managed_images("source-a")),
        )
        .unwrap();
        fs::create_dir(root.join("Logs/events.jsonl")).unwrap();

        store.begin_stage(ProjectStage::Import).unwrap();

        assert_eq!(
            store.manifest().stage(ProjectStage::Import).state(),
            StageState::Running
        );
        assert!(store
            .warnings()
            .iter()
            .any(|warning| matches!(warning, ProjectStoreWarning::EventAppendFailed { .. })));
    }
}

#[cfg(all(
    test,
    not(any(
        target_vendor = "apple",
        target_os = "android",
        target_os = "linux",
        target_os = "redox",
    ))
))]
mod unsupported_platform_tests {
    use super::*;

    #[test]
    fn project_store_rejects_platforms_without_descriptor_relative_io() {
        assert!(matches!(
            platform_support(false),
            Err(ProjectStoreError::UnsupportedPlatform)
        ));
    }

    #[test]
    fn public_create_reports_the_unsupported_platform() {
        let result = ProjectStore::create(
            "Unsupported.rustscanproject",
            ProjectCreateRequest::new("Unsupported", SourceSpec::managed_images("source-a")),
        );

        assert!(matches!(
            result,
            Err(ProjectStoreError::UnsupportedPlatform)
        ));
    }
}
