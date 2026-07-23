use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};

use rustix::fs::{self as rustix_fs, FileType, Mode, OFlags};
use rustix::io::Errno;
use thiserror::Error;
use uuid::Uuid;

use super::artifacts;
use super::{
    ArtifactRef, ProjectManifest, ProjectManifestValidationError, ProjectStage, StageState,
    PROJECT_SCHEMA_VERSION,
};

const PACKAGE_EXTENSION: &str = "rustscanproject";
const MANIFEST_NAME: &str = "project.json";
const LOCK_NAME: &str = "project.lock";
const DELETE_TOMBSTONE_PREFIX: &str = ".rustscan-delete-";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectSummaryStatus {
    Ready,
    Active { stage: ProjectStage },
    Failed,
    Complete,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectSummary {
    pub id: Uuid,
    pub display_name: String,
    pub root: PathBuf,
    pub updated_unix_ms: u64,
    pub stages: BTreeMap<ProjectStage, StageState>,
    pub thumbnail: Option<ArtifactRef>,
    pub status: ProjectSummaryStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectSummaryEntry {
    Project(ProjectSummary),
    Invalid { root: PathBuf, error: String },
}

#[derive(Debug, Error)]
pub enum ProjectLibraryError {
    #[error("project library operations require native descriptor-relative no-replace support")]
    UnsupportedPlatform,
    #[error("project package path must end in .rustscanproject: {path:?}")]
    InvalidPackageSuffix { path: PathBuf },
    #[error("project package root is not a regular directory: {path:?}")]
    InvalidPackageRoot { path: PathBuf },
    #[error("project package root must not be a symbolic link: {path:?}")]
    SymlinkPackageRoot { path: PathBuf },
    #[error("duplicate destination must not be the source package or a descendant: {path:?}")]
    UnsafeDestination { path: PathBuf },
    #[error("duplicate destination exists and is not empty: {path:?}")]
    DestinationNotEmpty { path: PathBuf },
    #[error("source package has an active stage lease")]
    ActiveLease,
    #[error("delete confirmation id {provided} does not match project id {expected}")]
    DeleteConfirmationMismatch { expected: Uuid, provided: Uuid },
    #[error("project package changed before deletion: {path:?}")]
    DeleteTargetChanged { path: PathBuf },
    #[error(
        "project deletion cleanup failed; recoverable tombstone is at {tombstone:?}: {source}"
    )]
    DeleteCleanupFailed {
        tombstone: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("project deletion completed but parent directory sync failed: {source}")]
    DeleteCompletedButUnsynced {
        #[source]
        source: io::Error,
    },
    #[error("delete recovery path is not a tombstone directory: {path:?}")]
    InvalidDeleteTombstone { path: PathBuf },
    #[error("project manifest is malformed: {0}")]
    InvalidManifest(String),
    #[error("copied artifact does not match its manifest reference: {path:?}")]
    CopiedArtifactMismatch { path: PathBuf },
    #[error(transparent)]
    Io(#[from] io::Error),
}

pub fn list_summaries(
    library_root: impl AsRef<Path>,
) -> Result<Vec<ProjectSummaryEntry>, ProjectLibraryError> {
    require_supported_platform()?;
    let library_root_label = library_root.as_ref();
    let library_root = fs::canonicalize(library_root_label)?;
    let library_directory = open_directory_path(&library_root)?;
    let mut entries = Vec::new();
    for entry in rustix_fs::Dir::read_from(&library_directory).map_err(io::Error::from)? {
        let entry = entry.map_err(io::Error::from)?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == "." || name == ".." {
            continue;
        }
        if parse_delete_tombstone_name(&name).is_some() {
            continue;
        }
        let root = library_root_label.join(&name);
        if root.extension() != Some(OsStr::new(PACKAGE_EXTENSION)) {
            continue;
        }
        entries.push(read_summary(&library_directory, &name, root));
    }
    entries.sort_by(compare_summary_entries);
    Ok(entries)
}

pub(crate) fn duplicate_package(
    source_directory: &File,
    source_root: &Path,
    source_manifest: &ProjectManifest,
    destination: impl AsRef<Path>,
) -> Result<ProjectSummary, ProjectLibraryError> {
    require_supported_platform()?;
    source_manifest.validate().map_err(manifest_error)?;
    if source_manifest.lease().is_some() {
        return Err(ProjectLibraryError::ActiveLease);
    }
    let requested_destination = destination.as_ref().to_path_buf();
    let source_root = canonical_package_root(source_root)?;
    let (destination_parent, destination_name, destination_root, destination_state) =
        validate_duplicate_destination(&source_root, &requested_destination)?;

    let temporary_name = format!(".{}-duplicate-{}", destination_name, Uuid::new_v4());
    rustix_fs::mkdirat(&destination_parent, temporary_name.as_str(), Mode::RWXU)
        .map_err(io::Error::from)?;
    destination_parent.sync_all()?;
    let temporary_root = File::from(
        rustix_fs::openat(
            &destination_parent,
            temporary_name.as_str(),
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(io::Error::from)?,
    );
    let temporary_path = destination_root
        .parent()
        .expect("duplicate destination has a parent")
        .join(&temporary_name);

    let result = (|| {
        copy_non_artifact_payloads(source_directory, &temporary_root)?;
        let mut manifest = source_manifest.clone();
        manifest.id = Uuid::new_v4();
        manifest.lease = None;
        manifest.updated_unix_ms = super::manifest::unix_time_ms();
        manifest.validate().map_err(manifest_error)?;
        copy_manifest_artifacts(source_directory, &temporary_root, &manifest)?;
        validate_copied_artifacts(&temporary_root, &manifest)?;
        write_new_file_at(
            &temporary_root,
            Path::new(MANIFEST_NAME),
            &serde_json::to_vec_pretty(&manifest)
                .map_err(|error| ProjectLibraryError::InvalidManifest(error.to_string()))?,
        )?;
        let event = serde_json::json!({
            "kind": "duplicated_from",
            "source_project_id": source_manifest.id().to_string(),
            "unix_ms": super::manifest::unix_time_ms(),
        });
        write_new_file_at(
            &temporary_root,
            Path::new("Logs/events.jsonl"),
            format!("{}\n", serde_json::to_string(&event).unwrap()).as_bytes(),
        )?;
        temporary_root.sync_all()?;

        if destination_state == DestinationState::EmptyDirectory {
            ensure_empty_destination(&destination_parent, &destination_name)?;
            rustix_fs::unlinkat(
                &destination_parent,
                &destination_name,
                rustix_fs::AtFlags::REMOVEDIR,
            )
            .map_err(io::Error::from)?;
            destination_parent.sync_all()?;
        }
        artifacts::rename_no_replace(
            &destination_parent,
            temporary_name.as_str(),
            &destination_parent,
            &destination_name,
        )
        .map_err(io::Error::from)?;
        destination_parent.sync_all()?;
        Ok(summary_from_manifest(&manifest, requested_destination))
    })();

    if result.is_err() {
        let _ = fs::remove_dir_all(&temporary_path);
    }
    result
}

pub(crate) struct DeleteTarget {
    parent: File,
    name: String,
    root: File,
    root_device: u64,
    root_inode: u64,
    root_path: PathBuf,
}

pub(crate) fn prepare_delete(
    root_directory: &File,
    root: &Path,
) -> Result<DeleteTarget, ProjectLibraryError> {
    require_supported_platform()?;
    let canonical = canonical_package_root(root)?;
    let parent_path =
        canonical
            .parent()
            .ok_or_else(|| ProjectLibraryError::InvalidPackageRoot {
                path: canonical.clone(),
            })?;
    let name = canonical
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| ProjectLibraryError::InvalidPackageRoot {
            path: canonical.clone(),
        })?
        .to_owned();
    let root = root_directory.try_clone()?;
    let metadata = rustix_fs::fstat(&root).map_err(io::Error::from)?;
    Ok(DeleteTarget {
        parent: open_directory_path(parent_path)?,
        name,
        root,
        root_device: metadata.st_dev as u64,
        root_inode: metadata.st_ino as u64,
        root_path: canonical,
    })
}

impl DeleteTarget {
    pub(crate) fn release_root_directory_lock(&self) -> Result<(), ProjectLibraryError> {
        self.root.unlock()?;
        Ok(())
    }
}

pub(crate) fn delete_prepared(target: DeleteTarget) -> Result<(), ProjectLibraryError> {
    delete_prepared_with_after_removal(target, &mut || Ok(()))
}

pub fn cleanup_delete_tombstone(tombstone: impl AsRef<Path>) -> Result<(), ProjectLibraryError> {
    require_supported_platform()?;
    let requested = tombstone.as_ref();
    let metadata = fs::symlink_metadata(requested)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(ProjectLibraryError::InvalidDeleteTombstone {
            path: requested.to_path_buf(),
        });
    }
    let tombstone = fs::canonicalize(requested)?;
    let parent_path =
        tombstone
            .parent()
            .ok_or_else(|| ProjectLibraryError::InvalidDeleteTombstone {
                path: tombstone.clone(),
            })?;
    let name = tombstone
        .file_name()
        .and_then(OsStr::to_str)
        .filter(|name| parse_delete_tombstone_name(name).is_some())
        .ok_or_else(|| ProjectLibraryError::InvalidDeleteTombstone {
            path: tombstone.clone(),
        })?;
    let parent = open_directory_path(parent_path)?;
    let directory = open_directory_at(&parent, Path::new(name))?;
    remove_directory_contents(&directory, &mut || Ok(()))?;
    rustix_fs::unlinkat(&parent, name, rustix_fs::AtFlags::REMOVEDIR).map_err(io::Error::from)?;
    parent.sync_all()?;
    Ok(())
}

fn delete_prepared_with_after_removal(
    target: DeleteTarget,
    after_removal: &mut impl FnMut() -> io::Result<()>,
) -> Result<(), ProjectLibraryError> {
    delete_prepared_with_after_removal_and_parent_sync(target, after_removal, &mut |parent| {
        parent.sync_all()
    })
}

fn delete_prepared_with_after_removal_and_parent_sync(
    target: DeleteTarget,
    after_removal: &mut impl FnMut() -> io::Result<()>,
    sync_parent: &mut impl FnMut(&File) -> io::Result<()>,
) -> Result<(), ProjectLibraryError> {
    ensure_delete_target_identity(&target)?;
    let (tombstone_name, tombstone) = move_delete_target_to_tombstone(&target)?;
    let cleanup = (|| {
        sync_parent(&target.parent)?;
        remove_directory_contents(&target.root, after_removal)?;
        rustix_fs::unlinkat(
            &target.parent,
            tombstone_name.as_str(),
            rustix_fs::AtFlags::REMOVEDIR,
        )
        .map_err(io::Error::from)?;
        Ok(())
    })();
    if let Err(source) = cleanup {
        return Err(ProjectLibraryError::DeleteCleanupFailed { tombstone, source });
    }
    if let Err(source) = sync_parent(&target.parent) {
        return Err(ProjectLibraryError::DeleteCompletedButUnsynced { source });
    }
    Ok(())
}

fn parse_delete_tombstone_name(name: &str) -> Option<Uuid> {
    name.strip_prefix(DELETE_TOMBSTONE_PREFIX)
        .and_then(|id| Uuid::parse_str(id).ok())
}

fn move_delete_target_to_tombstone(
    target: &DeleteTarget,
) -> Result<(String, PathBuf), ProjectLibraryError> {
    for _ in 0..16 {
        let name = format!("{DELETE_TOMBSTONE_PREFIX}{}", Uuid::new_v4());
        match artifacts::rename_no_replace(&target.parent, &target.name, &target.parent, &name) {
            Ok(()) => return Ok((name.clone(), target.root_path.with_file_name(name))),
            Err(Errno::EXIST) => continue,
            Err(error) => return Err(io::Error::from(error).into()),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "unable to allocate collision-free deletion tombstone",
    )
    .into())
}

#[cfg(test)]
fn delete_prepared_with_test_failure_after_removal(
    target: DeleteTarget,
    fail_after: usize,
) -> Result<(), ProjectLibraryError> {
    let mut removals = 0;
    delete_prepared_with_after_removal(target, &mut || {
        removals += 1;
        if removals >= fail_after {
            Err(io::Error::other("injected deletion cleanup failure"))
        } else {
            Ok(())
        }
    })
}

#[cfg(test)]
fn delete_prepared_with_test_failure_after_final_sync(
    target: DeleteTarget,
) -> Result<(), ProjectLibraryError> {
    let mut syncs = 0;
    delete_prepared_with_after_removal_and_parent_sync(target, &mut || Ok(()), &mut |_| {
        syncs += 1;
        if syncs == 2 {
            Err(io::Error::other("injected final parent sync failure"))
        } else {
            Ok(())
        }
    })
}

pub(crate) fn reveal_path(root: &Path) -> Result<PathBuf, ProjectLibraryError> {
    require_supported_platform()?;
    canonical_package_root(root)
}

fn read_summary(parent: &File, name: &str, root: PathBuf) -> ProjectSummaryEntry {
    let result = (|| {
        let metadata = rustix_fs::statat(parent, name, rustix_fs::AtFlags::SYMLINK_NOFOLLOW)
            .map_err(io::Error::from)?;
        let file_type = FileType::from_raw_mode(metadata.st_mode);
        if file_type.is_symlink() {
            return Err(ProjectLibraryError::SymlinkPackageRoot { path: root.clone() });
        }
        if !file_type.is_dir() {
            return Err(ProjectLibraryError::InvalidPackageRoot { path: root.clone() });
        }
        let directory = open_directory_at(parent, Path::new(name))?;
        let value: serde_json::Value =
            serde_json::from_slice(&read_regular_file_at(&directory, Path::new(MANIFEST_NAME))?)
                .map_err(|error| ProjectLibraryError::InvalidManifest(error.to_string()))?;
        let version = value
            .get("schema_version")
            .and_then(serde_json::Value::as_u64)
            .and_then(|version| u32::try_from(version).ok())
            .ok_or_else(|| {
                ProjectLibraryError::InvalidManifest("missing schema_version".to_owned())
            })?;
        if version > PROJECT_SCHEMA_VERSION {
            return Err(ProjectLibraryError::InvalidManifest(format!(
                "future schema version {version}"
            )));
        }
        let manifest: ProjectManifest = serde_json::from_value(value)
            .map_err(|error| ProjectLibraryError::InvalidManifest(error.to_string()))?;
        manifest.validate().map_err(manifest_error)?;
        Ok(summary_from_manifest(&manifest, root.clone()))
    })();
    match result {
        Ok(summary) => ProjectSummaryEntry::Project(summary),
        Err(error) => ProjectSummaryEntry::Invalid {
            root,
            error: error.to_string(),
        },
    }
}

fn summary_from_manifest(manifest: &ProjectManifest, root: PathBuf) -> ProjectSummary {
    let stages = ProjectStage::ORDER
        .into_iter()
        .map(|stage| (stage, manifest.stage(stage).state()))
        .collect::<BTreeMap<_, _>>();
    let status = if let Some(lease) = manifest.lease() {
        ProjectSummaryStatus::Active { stage: lease.stage }
    } else if manifest.stage(ProjectStage::Complete).state() == StageState::Succeeded {
        ProjectSummaryStatus::Complete
    } else if ProjectStage::ORDER
        .into_iter()
        .any(|stage| manifest.stage(stage).state() == StageState::Failed)
    {
        ProjectSummaryStatus::Failed
    } else {
        ProjectSummaryStatus::Ready
    };
    ProjectSummary {
        id: manifest.id(),
        display_name: manifest.display_name.clone(),
        root,
        updated_unix_ms: manifest.updated_unix_ms,
        stages,
        thumbnail: manifest
            .active_scene
            .clone()
            .or(manifest.final_scene.clone()),
        status,
    }
}

fn compare_summary_entries(left: &ProjectSummaryEntry, right: &ProjectSummaryEntry) -> Ordering {
    match (left, right) {
        (ProjectSummaryEntry::Project(left), ProjectSummaryEntry::Project(right)) => right
            .updated_unix_ms
            .cmp(&left.updated_unix_ms)
            .then_with(|| {
                left.display_name
                    .to_lowercase()
                    .cmp(&right.display_name.to_lowercase())
            })
            .then_with(|| left.id.cmp(&right.id)),
        (ProjectSummaryEntry::Project(_), ProjectSummaryEntry::Invalid { .. }) => Ordering::Less,
        (ProjectSummaryEntry::Invalid { .. }, ProjectSummaryEntry::Project(_)) => Ordering::Greater,
        (
            ProjectSummaryEntry::Invalid { root: left, .. },
            ProjectSummaryEntry::Invalid { root: right, .. },
        ) => left.cmp(right),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DestinationState {
    Missing,
    EmptyDirectory,
}

fn validate_duplicate_destination(
    source_root: &Path,
    destination: &Path,
) -> Result<(File, String, PathBuf, DestinationState), ProjectLibraryError> {
    require_package_suffix(destination)?;
    let parent = destination
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent = fs::canonicalize(parent)?;
    let name = destination
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| ProjectLibraryError::InvalidPackageRoot {
            path: destination.to_path_buf(),
        })?
        .to_owned();
    let destination_root = parent.join(&name);
    if destination_root == source_root || destination_root.starts_with(source_root) {
        return Err(ProjectLibraryError::UnsafeDestination {
            path: destination_root,
        });
    }
    let parent_directory = open_directory_path(&parent)?;
    let state = match rustix_fs::statat(
        &parent_directory,
        &name,
        rustix_fs::AtFlags::SYMLINK_NOFOLLOW,
    ) {
        Ok(metadata) => {
            let file_type = FileType::from_raw_mode(metadata.st_mode);
            if file_type.is_symlink() {
                return Err(ProjectLibraryError::SymlinkPackageRoot {
                    path: destination_root,
                });
            }
            if !file_type.is_dir() {
                return Err(ProjectLibraryError::DestinationNotEmpty {
                    path: destination_root,
                });
            }
            ensure_empty_destination(&parent_directory, &name)?;
            DestinationState::EmptyDirectory
        }
        Err(error) if error == Errno::NOENT => DestinationState::Missing,
        Err(error) => return Err(io::Error::from(error).into()),
    };
    Ok((parent_directory, name, destination_root, state))
}

fn ensure_empty_destination(parent: &File, name: &str) -> Result<(), ProjectLibraryError> {
    let directory = File::from(
        rustix_fs::openat(
            parent,
            name,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(io::Error::from)?,
    );
    for entry in rustix_fs::Dir::read_from(&directory).map_err(io::Error::from)? {
        let entry = entry.map_err(io::Error::from)?;
        let name = entry.file_name().to_string_lossy();
        if name != "." && name != ".." {
            return Err(ProjectLibraryError::DestinationNotEmpty {
                path: PathBuf::from(name.as_ref()),
            });
        }
    }
    Ok(())
}

fn canonical_package_root(root: &Path) -> Result<PathBuf, ProjectLibraryError> {
    require_package_suffix(root)?;
    let metadata = fs::symlink_metadata(root)?;
    if metadata.file_type().is_symlink() {
        return Err(ProjectLibraryError::SymlinkPackageRoot {
            path: root.to_path_buf(),
        });
    }
    if !metadata.is_dir() {
        return Err(ProjectLibraryError::InvalidPackageRoot {
            path: root.to_path_buf(),
        });
    }
    let canonical = fs::canonicalize(root)?;
    require_package_suffix(&canonical)?;
    Ok(canonical)
}

fn require_package_suffix(path: &Path) -> Result<(), ProjectLibraryError> {
    if path.extension() != Some(OsStr::new(PACKAGE_EXTENSION)) {
        return Err(ProjectLibraryError::InvalidPackageSuffix {
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

fn copy_non_artifact_payloads(
    source: &File,
    destination: &File,
) -> Result<(), ProjectLibraryError> {
    copy_tree(source, destination, Path::new(""))
}

fn copy_tree(source: &File, destination: &File, prefix: &Path) -> Result<(), ProjectLibraryError> {
    for entry in rustix_fs::Dir::read_from(source).map_err(io::Error::from)? {
        let entry = entry.map_err(io::Error::from)?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == "." || name == ".." {
            continue;
        }
        let relative = prefix.join(&name);
        if should_skip_payload(&relative) {
            continue;
        }
        let metadata =
            rustix_fs::statat(source, name.as_str(), rustix_fs::AtFlags::SYMLINK_NOFOLLOW)
                .map_err(io::Error::from)?;
        let file_type = FileType::from_raw_mode(metadata.st_mode);
        if file_type.is_symlink() {
            return Err(ProjectLibraryError::SymlinkPackageRoot { path: relative });
        }
        if file_type.is_dir() {
            ensure_directory_at(destination, &relative)?;
            let child = open_directory_at(source, Path::new(&name))?;
            copy_tree(&child, destination, &relative)?;
        } else if file_type.is_file() {
            copy_regular_file(source, destination, &relative)?;
        } else {
            return Err(ProjectLibraryError::InvalidPackageRoot { path: relative });
        }
    }
    Ok(())
}

fn should_skip_payload(relative: &Path) -> bool {
    relative == Path::new(MANIFEST_NAME)
        || relative == Path::new(LOCK_NAME)
        || relative == Path::new("Logs")
        || relative == Path::new("Artifacts")
        || relative == Path::new("Cache/.staging")
}

fn copy_manifest_artifacts(
    source: &File,
    destination: &File,
    manifest: &ProjectManifest,
) -> Result<(), ProjectLibraryError> {
    let mut references = BTreeSet::new();
    for stage in ProjectStage::ORDER {
        references.extend(
            manifest
                .stage(stage)
                .artifacts()
                .iter()
                .map(|artifact| PathBuf::from(&artifact.relative_path)),
        );
    }
    references.extend(
        [
            manifest.active_scene.as_ref(),
            manifest.final_scene.as_ref(),
        ]
        .into_iter()
        .flatten()
        .map(|artifact| PathBuf::from(&artifact.relative_path)),
    );
    for reference in references {
        copy_regular_file(source, destination, &reference)?;
    }
    Ok(())
}

fn validate_copied_artifacts(
    root: &File,
    manifest: &ProjectManifest,
) -> Result<(), ProjectLibraryError> {
    let mut seen = BTreeSet::new();
    for stage in ProjectStage::ORDER {
        for artifact in manifest.stage(stage).artifacts() {
            validate_copied_artifact(root, artifact, &mut seen)?;
        }
    }
    for artifact in [
        manifest.active_scene.as_ref(),
        manifest.final_scene.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        validate_copied_artifact(root, artifact, &mut seen)?;
    }
    Ok(())
}

fn validate_copied_artifact(
    root: &File,
    artifact: &ArtifactRef,
    seen: &mut BTreeSet<(String, String, u64)>,
) -> Result<(), ProjectLibraryError> {
    if !seen.insert((
        artifact.relative_path.clone(),
        artifact.content_hash.clone(),
        artifact.byte_len,
    )) {
        return Ok(());
    }
    let file = open_regular_file_at(root, Path::new(&artifact.relative_path))?;
    let (length, hash) = hash_reader(file)?;
    if length != artifact.byte_len || hash.to_hex().as_str() != artifact.content_hash {
        return Err(ProjectLibraryError::CopiedArtifactMismatch {
            path: PathBuf::from(&artifact.relative_path),
        });
    }
    Ok(())
}

fn copy_regular_file(
    source_root: &File,
    destination_root: &File,
    relative: &Path,
) -> Result<(), ProjectLibraryError> {
    let mut source = open_regular_file_at(source_root, relative)?;
    let parent = relative.parent().unwrap_or_else(|| Path::new(""));
    let destination_parent = ensure_directory_at(destination_root, parent)?;
    let name = relative
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| ProjectLibraryError::InvalidPackageRoot {
            path: relative.to_path_buf(),
        })?;
    let opened = rustix_fs::openat(
        &destination_parent,
        name,
        OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::RUSR | Mode::WUSR,
    )
    .map_err(io::Error::from)?;
    let mut destination = File::from(opened);
    io::copy(&mut source, &mut destination)?;
    destination.sync_all()?;
    destination_parent.sync_all()?;
    Ok(())
}

fn write_new_file_at(
    root: &File,
    relative: &Path,
    bytes: &[u8],
) -> Result<(), ProjectLibraryError> {
    let parent = relative.parent().unwrap_or_else(|| Path::new(""));
    let directory = ensure_directory_at(root, parent)?;
    let name = relative
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| ProjectLibraryError::InvalidPackageRoot {
            path: relative.to_path_buf(),
        })?;
    let opened = rustix_fs::openat(
        &directory,
        name,
        OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::RUSR | Mode::WUSR,
    )
    .map_err(io::Error::from)?;
    let mut file = File::from(opened);
    file.write_all(bytes)?;
    file.sync_all()?;
    directory.sync_all()?;
    Ok(())
}

fn ensure_directory_at(root: &File, relative: &Path) -> Result<File, ProjectLibraryError> {
    let mut directory = root.try_clone()?;
    for component in relative.components() {
        let Component::Normal(component) = component else {
            return Err(ProjectLibraryError::InvalidPackageRoot {
                path: relative.to_path_buf(),
            });
        };
        match rustix_fs::statat(&directory, component, rustix_fs::AtFlags::SYMLINK_NOFOLLOW) {
            Ok(metadata) if FileType::from_raw_mode(metadata.st_mode).is_dir() => {}
            Ok(_) => {
                return Err(ProjectLibraryError::InvalidPackageRoot {
                    path: relative.to_path_buf(),
                })
            }
            Err(error) if error == Errno::NOENT => {
                rustix_fs::mkdirat(&directory, component, Mode::RWXU).map_err(io::Error::from)?;
                directory.sync_all()?;
            }
            Err(error) => return Err(io::Error::from(error).into()),
        }
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

fn open_directory_path(path: &Path) -> Result<File, ProjectLibraryError> {
    if !path.is_absolute() {
        return Err(ProjectLibraryError::InvalidPackageRoot {
            path: path.to_path_buf(),
        });
    }
    let mut directory = File::from(
        rustix_fs::open(
            Path::new("/"),
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(io::Error::from)?,
    );
    for component in path.components() {
        match component {
            Component::RootDir => {}
            Component::Normal(component) => {
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
            _ => {
                return Err(ProjectLibraryError::InvalidPackageRoot {
                    path: path.to_path_buf(),
                })
            }
        }
    }
    Ok(directory)
}

fn open_directory_at(root: &File, relative: &Path) -> Result<File, ProjectLibraryError> {
    let mut directory = root.try_clone()?;
    for component in relative.components() {
        let Component::Normal(component) = component else {
            return Err(ProjectLibraryError::InvalidPackageRoot {
                path: relative.to_path_buf(),
            });
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

fn read_regular_file_at(root: &File, relative: &Path) -> Result<Vec<u8>, ProjectLibraryError> {
    let mut file = open_regular_file_at(root, relative)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn open_regular_file_at(root: &File, relative: &Path) -> Result<File, ProjectLibraryError> {
    let mut components = relative.components();
    let Some(Component::Normal(mut component)) = components.next() else {
        return Err(ProjectLibraryError::InvalidPackageRoot {
            path: relative.to_path_buf(),
        });
    };
    let mut directory = root.try_clone()?;
    for next in components {
        let Component::Normal(next) = next else {
            return Err(ProjectLibraryError::InvalidPackageRoot {
                path: relative.to_path_buf(),
            });
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
        component = next;
    }
    let opened = rustix_fs::openat(
        &directory,
        component,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(io::Error::from)?;
    if !FileType::from_raw_mode(rustix_fs::fstat(&opened).map_err(io::Error::from)?.st_mode)
        .is_file()
    {
        return Err(ProjectLibraryError::InvalidPackageRoot {
            path: relative.to_path_buf(),
        });
    }
    Ok(File::from(opened))
}

fn ensure_delete_target_identity(target: &DeleteTarget) -> Result<(), ProjectLibraryError> {
    let metadata = rustix_fs::statat(
        &target.parent,
        &target.name,
        rustix_fs::AtFlags::SYMLINK_NOFOLLOW,
    );
    match metadata {
        Ok(metadata)
            if metadata.st_dev as u64 == target.root_device
                && metadata.st_ino as u64 == target.root_inode =>
        {
            Ok(())
        }
        Ok(_) | Err(Errno::NOENT) => Err(ProjectLibraryError::DeleteTargetChanged {
            path: target.root_path.clone(),
        }),
        Err(error) => Err(io::Error::from(error).into()),
    }
}

fn remove_directory_tree_at(
    parent: &File,
    name: &str,
    after_removal: &mut impl FnMut() -> io::Result<()>,
) -> io::Result<()> {
    let directory = File::from(
        rustix_fs::openat(
            parent,
            name,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(io::Error::from)?,
    );
    remove_directory_contents(&directory, after_removal)?;
    rustix_fs::unlinkat(parent, name, rustix_fs::AtFlags::REMOVEDIR).map_err(io::Error::from)?;
    after_removal()?;
    Ok(())
}

fn remove_directory_contents(
    directory: &File,
    after_removal: &mut impl FnMut() -> io::Result<()>,
) -> io::Result<()> {
    for entry in rustix_fs::Dir::read_from(directory).map_err(io::Error::from)? {
        let entry = entry.map_err(io::Error::from)?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == "." || name == ".." {
            continue;
        }
        let metadata = rustix_fs::statat(
            directory,
            name.as_str(),
            rustix_fs::AtFlags::SYMLINK_NOFOLLOW,
        )
        .map_err(io::Error::from)?;
        if FileType::from_raw_mode(metadata.st_mode).is_dir() {
            remove_directory_tree_at(directory, &name, after_removal)?;
        } else {
            rustix_fs::unlinkat(directory, name.as_str(), rustix_fs::AtFlags::empty())
                .map_err(io::Error::from)?;
            after_removal()?;
        }
    }
    directory.sync_all()?;
    Ok(())
}

fn hash_reader(mut reader: impl Read) -> io::Result<(u64, blake3::Hash)> {
    let mut hasher = blake3::Hasher::new();
    let mut length = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
        length = length.checked_add(count as u64).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "artifact length overflow")
        })?;
    }
    Ok((length, hasher.finalize()))
}

fn manifest_error(error: ProjectManifestValidationError) -> ProjectLibraryError {
    ProjectLibraryError::InvalidManifest(error.to_string())
}

fn require_supported_platform() -> Result<(), ProjectLibraryError> {
    if cfg!(any(
        target_vendor = "apple",
        target_os = "android",
        target_os = "linux",
        target_os = "redox",
    )) {
        Ok(())
    } else {
        Err(ProjectLibraryError::UnsupportedPlatform)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interrupted_delete_hides_the_package_and_leaves_a_recoverable_tombstone() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("Delete.rustscanproject");
        fs::create_dir(&root).unwrap();
        fs::write(root.join("first-child"), b"first").unwrap();
        fs::write(root.join("second-child"), b"second").unwrap();
        let root = fs::canonicalize(root).unwrap();
        let root_directory = open_directory_path(&root).unwrap();
        let target = prepare_delete(&root_directory, &root).unwrap();

        let error = delete_prepared_with_test_failure_after_removal(target, 1).unwrap_err();
        let ProjectLibraryError::DeleteCleanupFailed { tombstone, source } = error else {
            panic!("expected a recoverable delete cleanup error");
        };

        assert_eq!(source.kind(), io::ErrorKind::Other);
        assert!(!root.exists());
        assert!(tombstone.is_dir());
        assert!(tombstone
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with(".rustscan-delete-"));
        assert!(
            !tombstone.join("first-child").exists() || !tombstone.join("second-child").exists(),
            "the injected failure must occur after a child was removed"
        );
        assert!(list_summaries(temp.path()).unwrap().is_empty());

        cleanup_delete_tombstone(&tombstone).unwrap();
        assert!(!tombstone.exists());
    }

    #[test]
    fn cleanup_delete_tombstone_rejects_a_prefix_match_without_a_uuid_and_preserves_it() {
        let temp = tempfile::tempdir().unwrap();
        let tombstone = temp.path().join(".rustscan-delete-notes");
        fs::create_dir(&tombstone).unwrap();
        fs::write(tombstone.join("payload"), b"keep").unwrap();

        let error = cleanup_delete_tombstone(&tombstone).unwrap_err();

        assert!(matches!(
            error,
            ProjectLibraryError::InvalidDeleteTombstone { .. }
        ));
        assert!(tombstone.is_dir());
        assert_eq!(fs::read(tombstone.join("payload")).unwrap(), b"keep");
    }

    #[test]
    fn cleanup_delete_tombstone_removes_a_prefix_and_uuid_marker() {
        let temp = tempfile::tempdir().unwrap();
        let tombstone = temp
            .path()
            .join(format!("{DELETE_TOMBSTONE_PREFIX}{}", Uuid::new_v4()));
        fs::create_dir(&tombstone).unwrap();
        fs::write(tombstone.join("payload"), b"delete").unwrap();

        cleanup_delete_tombstone(&tombstone).unwrap();

        assert!(!tombstone.exists());
    }

    #[test]
    fn final_delete_sync_failure_reports_completed_but_unsynced_without_a_tombstone() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("Delete.rustscanproject");
        fs::create_dir(&root).unwrap();
        fs::write(root.join("payload"), b"delete").unwrap();
        let root = fs::canonicalize(root).unwrap();
        let root_directory = open_directory_path(&root).unwrap();
        let target = prepare_delete(&root_directory, &root).unwrap();

        let error = delete_prepared_with_test_failure_after_final_sync(target).unwrap_err();

        let ProjectLibraryError::DeleteCompletedButUnsynced { source } = error else {
            panic!("expected a completed-but-unsynced delete error");
        };
        assert_eq!(source.kind(), io::ErrorKind::Other);
        assert!(!root.exists());
        assert!(fs::read_dir(temp.path()).unwrap().all(|entry| !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(DELETE_TOMBSTONE_PREFIX)));
    }

    #[cfg(unix)]
    #[test]
    fn open_directory_path_rejects_a_symlinked_ancestor() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("target");
        let symlinked = temp.path().join("symlinked");
        fs::create_dir(&target).unwrap();
        symlink(&target, &symlinked).unwrap();

        assert!(open_directory_path(&symlinked).is_err());
    }
}
