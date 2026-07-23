#![cfg_attr(not(test), allow(dead_code))]

use std::collections::BTreeSet;
use std::fs::File;
use std::io::{self, Read, Seek};
use std::path::{Path, PathBuf};

use rustix::fs::{self as rustix_fs, FileType, Mode, OFlags};
use rustix::io::Errno;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use super::{ArtifactRef, ProjectStage};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ArtifactValidationKind {
    ReadableFile,
    Json,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StagedArtifact {
    payload_path: PathBuf,
    kind: ArtifactValidationKind,
}

impl StagedArtifact {
    pub(crate) fn new(
        payload_path: impl AsRef<Path>,
        kind: ArtifactValidationKind,
    ) -> Result<Self, ArtifactCommitError> {
        let payload_path = normalize_relative_path(payload_path.as_ref())?;
        Ok(Self { payload_path, kind })
    }

    fn payload_path(&self) -> &Path {
        &self.payload_path
    }
}

#[derive(Debug, Clone)]
pub(crate) struct StageWorkspace {
    stage: ProjectStage,
    attempt: u32,
    relative: PathBuf,
    path: PathBuf,
}

impl StageWorkspace {
    pub(crate) fn stage(&self) -> ProjectStage {
        self.stage
    }

    pub(crate) fn attempt(&self) -> u32 {
        self.attempt
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommitPhase {
    AfterWorkspaceSync,
    AfterAttemptRename,
}

#[derive(Debug, Error)]
pub(crate) enum ArtifactCommitError {
    #[error("a stage attempt must declare at least one artifact")]
    EmptyDeclaration,
    #[error("artifact payload path is not a safe package-relative UTF-8 path: {path:?}")]
    UnsafePayloadPath { path: PathBuf },
    #[error("artifact payload path is declared more than once: {path:?}")]
    DuplicatePayloadPath { path: PathBuf },
    #[error("strict artifact validation found undeclared payload file: {path:?}")]
    UndeclaredPayload { path: PathBuf },
    #[error("artifact payload is not a regular file: {path:?}")]
    NotRegularFile { path: PathBuf },
    #[error("artifact payload contains a symbolic link: {path:?}")]
    Symlink { path: PathBuf },
    #[error("artifact JSON payload is malformed: {path:?}: {source}")]
    MalformedJson {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("attempt directory already exists: {path:?}")]
    AttemptAlreadyExists { path: PathBuf },
    #[error("workspace directory may exist after its parent-directory sync failed: {source}")]
    WorkspaceCreationUncertain {
        #[source]
        source: io::Error,
    },
    #[error(transparent)]
    Io(#[from] io::Error),
}

pub(crate) fn stage_name(stage: ProjectStage) -> &'static str {
    match stage {
        ProjectStage::Import => "import",
        ProjectStage::KeyframeSfm => "keyframe_sfm",
        ProjectStage::FullFramePnp => "full_frame_pnp",
        ProjectStage::Training => "training",
        ProjectStage::Complete => "complete",
    }
}

pub(crate) fn staging_relative(stage: ProjectStage, attempt: u32) -> PathBuf {
    PathBuf::from("Cache/.staging").join(format!("{}-{attempt}", stage_name(stage)))
}

pub(crate) fn attempt_relative(stage: ProjectStage, attempt: u32) -> PathBuf {
    PathBuf::from("Artifacts")
        .join(stage_name(stage))
        .join(format!("attempt-{attempt:08}"))
}

pub(crate) fn recover_interrupted_attempts(
    root_directory: &File,
    stage: ProjectStage,
    attempt: u32,
    referenced_artifacts: &BTreeSet<PathBuf>,
) -> io::Result<()> {
    recover_interrupted_attempts_with_hooks(
        root_directory,
        stage,
        attempt,
        referenced_artifacts,
        || {
            format!(
                "interrupted-{}-{attempt}-{}",
                stage_name(stage),
                Uuid::new_v4()
            )
        },
        |_| {},
    )
}

#[cfg(test)]
pub(crate) fn recover_interrupted_attempts_with_test_hooks(
    root_directory: &File,
    stage: ProjectStage,
    attempt: u32,
    referenced_artifacts: &BTreeSet<PathBuf>,
    recovery_name: impl FnMut() -> String,
    sync_hook: impl FnMut(&Path),
) -> io::Result<()> {
    recover_interrupted_attempts_with_hooks(
        root_directory,
        stage,
        attempt,
        referenced_artifacts,
        recovery_name,
        sync_hook,
    )
}

fn recover_interrupted_attempts_with_hooks(
    root_directory: &File,
    stage: ProjectStage,
    attempt: u32,
    referenced_artifacts: &BTreeSet<PathBuf>,
    mut recovery_name: impl FnMut() -> String,
    mut sync_hook: impl FnMut(&Path),
) -> io::Result<()> {
    let staging = staging_relative(stage, attempt);
    if !references_content(referenced_artifacts, &staging) {
        move_to_recovery_if_present(
            root_directory,
            staging.parent().expect("staging workspace has a parent"),
            staging
                .file_name()
                .expect("staging workspace has a name")
                .to_str()
                .expect("staging workspace name is ASCII"),
            stage,
            attempt,
            &mut recovery_name,
            &mut sync_hook,
        )?;
    }
    let committed_attempt = attempt_relative(stage, attempt);
    if !references_content(referenced_artifacts, &committed_attempt) {
        move_to_recovery_if_present(
            root_directory,
            committed_attempt
                .parent()
                .expect("committed attempt has a parent"),
            committed_attempt
                .file_name()
                .expect("committed attempt has a name")
                .to_str()
                .expect("committed attempt name is ASCII"),
            stage,
            attempt,
            &mut recovery_name,
            &mut sync_hook,
        )?;
    }
    Ok(())
}

fn references_content(references: &BTreeSet<PathBuf>, directory: &Path) -> bool {
    references
        .iter()
        .any(|reference| reference == directory || reference.starts_with(directory))
}

fn move_to_recovery_if_present(
    root_directory: &File,
    source_parent: &Path,
    source_name: &str,
    stage: ProjectStage,
    attempt: u32,
    recovery_name: &mut impl FnMut() -> String,
    sync_hook: &mut impl FnMut(&Path),
) -> io::Result<()> {
    let source_parent_directory = match open_directory(root_directory, source_parent) {
        Ok(directory) => directory,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    match rustix_fs::statat(
        &source_parent_directory,
        source_name,
        rustix_fs::AtFlags::SYMLINK_NOFOLLOW,
    ) {
        Ok(metadata) => {
            if !FileType::from_raw_mode(metadata.st_mode).is_dir() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "interrupted project workspace is not a directory",
                ));
            }
        }
        Err(error) if error == Errno::NOENT => return Ok(()),
        Err(error) => return Err(io::Error::from(error)),
    }

    let recovery_directory = open_directory(root_directory, Path::new("Logs/recovery"))?;
    for _ in 0..16 {
        let destination_name = recovery_name();
        match rename_no_replace(
            &source_parent_directory,
            source_name,
            &recovery_directory,
            destination_name.as_str(),
        ) {
            Ok(()) => {
                sync_directory(&source_parent_directory, source_parent, sync_hook)?;
                sync_directory(&recovery_directory, Path::new("Logs/recovery"), sync_hook)?;
                return Ok(());
            }
            Err(error) if error == Errno::EXIST => continue,
            Err(error) => return Err(io::Error::from(error)),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        format!(
            "unable to allocate collision-free recovery directory for {} attempt {attempt}",
            stage_name(stage)
        ),
    ))
}

fn open_directory(root_directory: &File, relative: &Path) -> io::Result<File> {
    let mut directory = root_directory.try_clone()?;
    for component in relative.components() {
        let std::path::Component::Normal(component) = component else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "project directory path must be relative",
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

pub(crate) fn create_workspace(
    root_directory: &File,
    root: &Path,
    stage: ProjectStage,
    attempt: u32,
) -> Result<StageWorkspace, ArtifactCommitError> {
    let staging_parent = open_directory(root_directory, Path::new("Cache/.staging"))?;
    let name = format!("{}-{attempt}", stage_name(stage));
    rustix_fs::mkdirat(&staging_parent, name.as_str(), Mode::RWXU).map_err(io::Error::from)?;
    if let Err(source) = staging_parent.sync_all() {
        return Err(ArtifactCommitError::WorkspaceCreationUncertain { source });
    }
    let relative = staging_relative(stage, attempt);
    Ok(StageWorkspace {
        stage,
        attempt,
        path: root.join(&relative),
        relative,
    })
}

pub(crate) fn commit_workspace(
    root_directory: &File,
    workspace: &StageWorkspace,
    declarations: &[StagedArtifact],
    strict: bool,
    phase_hook: impl FnMut(CommitPhase) -> Result<(), ArtifactCommitError>,
) -> Result<Vec<ArtifactRef>, ArtifactCommitError> {
    commit_workspace_with_hooks(
        root_directory,
        workspace,
        declarations,
        strict,
        phase_hook,
        |_| {},
    )
}

#[cfg(test)]
pub(crate) fn commit_workspace_with_test_sync_hook(
    root_directory: &File,
    workspace: &StageWorkspace,
    declarations: &[StagedArtifact],
    strict: bool,
    phase_hook: impl FnMut(CommitPhase) -> Result<(), ArtifactCommitError>,
    sync_hook: impl FnMut(&Path),
) -> Result<Vec<ArtifactRef>, ArtifactCommitError> {
    commit_workspace_with_hooks(
        root_directory,
        workspace,
        declarations,
        strict,
        phase_hook,
        sync_hook,
    )
}

fn commit_workspace_with_hooks(
    root_directory: &File,
    workspace: &StageWorkspace,
    declarations: &[StagedArtifact],
    strict: bool,
    mut phase_hook: impl FnMut(CommitPhase) -> Result<(), ArtifactCommitError>,
    mut sync_hook: impl FnMut(&Path),
) -> Result<Vec<ArtifactRef>, ArtifactCommitError> {
    if declarations.is_empty() {
        return Err(ArtifactCommitError::EmptyDeclaration);
    }
    let mut declared = BTreeSet::new();
    for declaration in declarations {
        if !declared.insert(declaration.payload_path.clone()) {
            return Err(ArtifactCommitError::DuplicatePayloadPath {
                path: declaration.payload_path.clone(),
            });
        }
    }

    let workspace_directory = open_directory(root_directory, &workspace.relative)?;
    validate_workspace_payloads(
        &workspace_directory,
        &workspace.relative,
        declarations,
        &declared,
        strict,
        &mut sync_hook,
    )?;
    phase_hook(CommitPhase::AfterWorkspaceSync)?;

    let attempt_parent = ensure_directory(
        root_directory,
        &PathBuf::from("Artifacts").join(stage_name(workspace.stage)),
    )?;
    let attempt_name = format!("attempt-{:08}", workspace.attempt);
    match rustix_fs::statat(
        &attempt_parent,
        attempt_name.as_str(),
        rustix_fs::AtFlags::SYMLINK_NOFOLLOW,
    ) {
        Ok(_) => {
            return Err(ArtifactCommitError::AttemptAlreadyExists {
                path: attempt_relative(workspace.stage, workspace.attempt),
            });
        }
        Err(error) if error == Errno::NOENT => {}
        Err(error) => return Err(io::Error::from(error).into()),
    }
    let staging_parent = open_directory(
        root_directory,
        workspace
            .relative
            .parent()
            .expect("staging workspace has a parent"),
    )?;
    let workspace_name = workspace
        .relative
        .file_name()
        .expect("staging workspace has a name");
    rename_no_replace(
        &staging_parent,
        workspace_name,
        &attempt_parent,
        attempt_name.as_str(),
    )
    .map_err(io::Error::from)?;
    sync_directory(
        &staging_parent,
        workspace
            .relative
            .parent()
            .expect("staging workspace has a parent"),
        &mut sync_hook,
    )?;
    sync_directory(
        &attempt_parent,
        attempt_relative(workspace.stage, workspace.attempt)
            .parent()
            .expect("attempt directory has a parent"),
        &mut sync_hook,
    )?;
    phase_hook(CommitPhase::AfterAttemptRename)?;

    let attempt_root = attempt_relative(workspace.stage, workspace.attempt);
    let attempt_directory = open_directory(root_directory, &attempt_root)?;
    let validated = validate_workspace_payloads(
        &attempt_directory,
        &attempt_root,
        declarations,
        &declared,
        strict,
        &mut sync_hook,
    )?;
    Ok(validated
        .into_iter()
        .map(|(payload_path, byte_len, content_hash)| ArtifactRef {
            relative_path: attempt_root
                .join(payload_path)
                .to_string_lossy()
                .replace('\\', "/"),
            content_hash,
            byte_len,
        })
        .collect())
}

fn validate_workspace_payloads(
    workspace_directory: &File,
    workspace_relative: &Path,
    declarations: &[StagedArtifact],
    declared: &BTreeSet<PathBuf>,
    strict: bool,
    sync_hook: &mut impl FnMut(&Path),
) -> Result<Vec<(PathBuf, u64, String)>, ArtifactCommitError> {
    let mut validated = Vec::with_capacity(declarations.len());
    for declaration in declarations {
        let (mut file, containing_directories) = open_payload_file(
            workspace_directory,
            workspace_relative,
            declaration.payload_path(),
        )?;
        if declaration.kind == ArtifactValidationKind::Json {
            serde_json::from_reader::<_, serde_json::Value>(&mut file).map_err(|source| {
                ArtifactCommitError::MalformedJson {
                    path: declaration.payload_path.clone(),
                    source,
                }
            })?;
            file.rewind()?;
        }
        let (byte_len, content_hash) = hash_reader(&mut file)?;
        sync_file(
            &file,
            &workspace_relative.join(declaration.payload_path()),
            sync_hook,
        )?;
        for (directory, relative) in containing_directories.into_iter().rev() {
            sync_directory(&directory, &relative, sync_hook)?;
        }
        validated.push((
            declaration.payload_path.clone(),
            byte_len,
            content_hash.to_hex().to_string(),
        ));
    }
    if strict {
        let mut discovered = BTreeSet::new();
        collect_regular_files(workspace_directory, Path::new(""), &mut discovered)?;
        if let Some(path) = discovered.into_iter().find(|path| !declared.contains(path)) {
            return Err(ArtifactCommitError::UndeclaredPayload { path });
        }
    }
    sync_directory(workspace_directory, workspace_relative, sync_hook)?;
    Ok(validated)
}

fn sync_file(
    file: &File,
    relative: &Path,
    sync_hook: &mut impl FnMut(&Path),
) -> Result<(), ArtifactCommitError> {
    file.sync_all()?;
    sync_hook(relative);
    Ok(())
}

fn sync_directory(
    directory: &File,
    relative: &Path,
    sync_hook: &mut impl FnMut(&Path),
) -> io::Result<()> {
    directory.sync_all()?;
    sync_hook(relative);
    Ok(())
}

fn normalize_relative_path(path: &Path) -> Result<PathBuf, ArtifactCommitError> {
    if path.as_os_str().is_empty() || path.is_absolute() || path.to_str().is_none() {
        return Err(ArtifactCommitError::UnsafePayloadPath {
            path: path.to_path_buf(),
        });
    }
    let normalized = path.to_path_buf();
    if normalized.to_string_lossy().contains('\\')
        || normalized
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(ArtifactCommitError::UnsafePayloadPath { path: normalized });
    }
    Ok(normalized)
}

fn open_payload_file(
    workspace_directory: &File,
    workspace_relative: &Path,
    relative: &Path,
) -> Result<(File, Vec<(File, PathBuf)>), ArtifactCommitError> {
    let relative = normalize_relative_path(relative)?;
    let mut components = relative.components();
    let Some(std::path::Component::Normal(mut component)) = components.next() else {
        return Err(ArtifactCommitError::UnsafePayloadPath { path: relative });
    };
    let mut directory = workspace_directory.try_clone()?;
    let mut directory_relative = PathBuf::new();
    let mut containing_directories = vec![(
        workspace_directory.try_clone()?,
        workspace_relative.to_path_buf(),
    )];
    for next in components {
        let std::path::Component::Normal(next) = next else {
            return Err(ArtifactCommitError::UnsafePayloadPath { path: relative });
        };
        let opened = rustix_fs::openat(
            &directory,
            component,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(io::Error::from)?;
        directory = File::from(opened);
        directory_relative.push(component);
        containing_directories.push((
            directory.try_clone()?,
            workspace_relative.join(&directory_relative),
        ));
        component = next;
    }
    let opened = rustix_fs::openat(
        &directory,
        component,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(io::Error::from)?;
    let metadata = rustix_fs::fstat(&opened).map_err(io::Error::from)?;
    let file_type = FileType::from_raw_mode(metadata.st_mode);
    if file_type.is_symlink() {
        return Err(ArtifactCommitError::Symlink { path: relative });
    }
    if !file_type.is_file() {
        return Err(ArtifactCommitError::NotRegularFile { path: relative });
    }
    Ok((File::from(opened), containing_directories))
}

fn collect_regular_files(
    directory: &File,
    prefix: &Path,
    discovered: &mut BTreeSet<PathBuf>,
) -> Result<(), ArtifactCommitError> {
    let entries = rustix_fs::Dir::read_from(directory).map_err(io::Error::from)?;
    for entry in entries {
        let entry = entry.map_err(io::Error::from)?;
        let name = entry.file_name().to_string_lossy();
        if name == "." || name == ".." {
            continue;
        }
        let relative = prefix.join(name.as_ref());
        let metadata = rustix_fs::statat(
            directory,
            entry.file_name(),
            rustix_fs::AtFlags::SYMLINK_NOFOLLOW,
        )
        .map_err(io::Error::from)?;
        let file_type = FileType::from_raw_mode(metadata.st_mode);
        if file_type.is_symlink() {
            return Err(ArtifactCommitError::Symlink { path: relative });
        }
        if file_type.is_dir() {
            let child = File::from(
                rustix_fs::openat(
                    directory,
                    entry.file_name(),
                    OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                    Mode::empty(),
                )
                .map_err(io::Error::from)?,
            );
            collect_regular_files(&child, &relative, discovered)?;
        } else if file_type.is_file() {
            discovered.insert(relative);
        } else {
            return Err(ArtifactCommitError::NotRegularFile { path: relative });
        }
    }
    Ok(())
}

fn ensure_directory(root_directory: &File, relative: &Path) -> io::Result<File> {
    let mut directory = root_directory.try_clone()?;
    for component in relative.components() {
        let std::path::Component::Normal(component) = component else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "project directory path must be relative",
            ));
        };
        match rustix_fs::statat(&directory, component, rustix_fs::AtFlags::SYMLINK_NOFOLLOW) {
            Ok(metadata) => {
                if !FileType::from_raw_mode(metadata.st_mode).is_dir() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "project artifact directory is not a directory",
                    ));
                }
            }
            Err(error) if error == Errno::NOENT => {
                rustix_fs::mkdirat(&directory, component, Mode::RWXU).map_err(io::Error::from)?;
                directory.sync_all()?;
            }
            Err(error) => return Err(io::Error::from(error)),
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

#[cfg(any(
    target_vendor = "apple",
    target_os = "android",
    target_os = "linux",
    target_os = "redox"
))]
pub(crate) fn rename_no_replace(
    old_directory: &File,
    old_name: impl rustix::path::Arg,
    new_directory: &File,
    new_name: impl rustix::path::Arg,
) -> rustix::io::Result<()> {
    rustix_fs::renameat_with(
        old_directory,
        old_name,
        new_directory,
        new_name,
        rustix_fs::RenameFlags::NOREPLACE,
    )
}

#[cfg(not(any(
    target_vendor = "apple",
    target_os = "android",
    target_os = "linux",
    target_os = "redox"
)))]
pub(crate) fn rename_no_replace(
    _old_directory: &File,
    _old_name: impl rustix::path::Arg,
    _new_directory: &File,
    _new_name: impl rustix::path::Arg,
) -> rustix::io::Result<()> {
    Err(Errno::NOSYS)
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
