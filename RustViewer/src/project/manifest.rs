use std::collections::BTreeMap;
use std::path::{Component, Path};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

pub const PROJECT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectStage {
    Import,
    KeyframeSfm,
    FullFramePnp,
    Training,
    Complete,
}

impl ProjectStage {
    pub const ORDER: [Self; 5] = [
        Self::Import,
        Self::KeyframeSfm,
        Self::FullFramePnp,
        Self::Training,
        Self::Complete,
    ];

    pub(crate) fn artifact_directory(self) -> &'static str {
        match self {
            Self::Import => "import",
            Self::KeyframeSfm => "keyframe_sfm",
            Self::FullFramePnp => "full_frame_pnp",
            Self::Training => "training",
            Self::Complete => "complete",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageState {
    NotStarted,
    Ready,
    Queued,
    Running,
    PauseRequested,
    Paused,
    CancelRequested,
    Cancelled,
    Succeeded,
    Failed,
    Stale,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub relative_path: String,
    pub content_hash: String,
    pub byte_len: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ArtifactValidationError {
    #[error("a successful stage must commit at least one artifact")]
    Empty,
    #[error("artifact path must be a non-empty relative path without traversal: {0:?}")]
    InvalidRelativePath(String),
    #[error("artifact {relative_path:?} must have a canonical 64-character lowercase hex hash")]
    InvalidContentHash {
        relative_path: String,
        content_hash: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageRecord {
    pub(crate) state: StageState,
    pub(crate) attempt: u32,
    pub(crate) completed: Option<u64>,
    pub(crate) total: Option<u64>,
    pub(crate) started_unix_ms: Option<u64>,
    pub(crate) updated_unix_ms: u64,
    pub(crate) artifacts: Vec<ArtifactRef>,
    pub(crate) error: Option<ProjectErrorRecord>,
}

impl StageRecord {
    pub fn state(&self) -> StageState {
        self.state
    }

    pub fn attempt(&self) -> u32 {
        self.attempt
    }

    pub fn completed(&self) -> Option<u64> {
        self.completed
    }

    pub fn total(&self) -> Option<u64> {
        self.total
    }

    pub fn started_unix_ms(&self) -> Option<u64> {
        self.started_unix_ms
    }

    pub fn updated_unix_ms(&self) -> u64 {
        self.updated_unix_ms
    }

    pub fn artifacts(&self) -> &[ArtifactRef] {
        &self.artifacts
    }

    pub fn error(&self) -> Option<&ProjectErrorRecord> {
        self.error.as_ref()
    }

    pub(crate) fn new(state: StageState, now_unix_ms: u64) -> Self {
        Self {
            state,
            attempt: 0,
            completed: None,
            total: None,
            started_unix_ms: None,
            updated_unix_ms: now_unix_ms,
            artifacts: Vec::new(),
            error: None,
        }
    }

    pub(crate) fn reset_transient_work(&mut self) {
        self.completed = None;
        self.total = None;
        self.started_unix_ms = None;
        self.error = None;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SuggestedAction {
    Retry,
    WiderSearch,
    RevealSource,
    OpenLog,
    StartFresh,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectErrorRecord {
    pub code: String,
    pub stage: ProjectStage,
    pub summary: String,
    pub detail: String,
    pub frame_id: Option<u32>,
    pub pair: Option<(u32, u32)>,
    pub retryable: bool,
    pub suggested_actions: Vec<SuggestedAction>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompatibilityRecord {
    pub rustsfm_artifact_version: u32,
    pub rustgs_checkpoint_version: u32,
}

impl Default for CompatibilityRecord {
    fn default() -> Self {
        Self {
            rustsfm_artifact_version: 1,
            rustgs_checkpoint_version: rustgs::TRAINING_CHECKPOINT_VERSION,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectLease {
    pub project_id: Uuid,
    pub stage: ProjectStage,
    pub attempt: u32,
    pub process_id: u32,
    pub started_unix_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    ImageSequence,
    Video,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceOwnership {
    ManagedCopy,
    Referenced,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceSpec {
    pub kind: SourceKind,
    pub ownership: SourceOwnership,
    pub identity: String,
    pub display_paths: Vec<String>,
    pub bookmark: Option<Vec<u8>>,
}

impl SourceSpec {
    pub fn managed_images(identity: impl Into<String>) -> Self {
        Self {
            kind: SourceKind::ImageSequence,
            ownership: SourceOwnership::ManagedCopy,
            identity: identity.into(),
            display_paths: Vec::new(),
            bookmark: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportConfigSnapshot {
    pub video_keyframes_per_second: f64,
    pub maximum_keyframe_gap_us: i64,
    pub thumbnail_long_edge: u32,
}

impl Default for ImportConfigSnapshot {
    fn default() -> Self {
        Self {
            video_keyframes_per_second: 3.0,
            maximum_keyframe_gap_us: 1_000_000,
            thumbnail_long_edge: 256,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SfmConfigSnapshot {
    #[serde(default)]
    pub keyframe_selection: KeyframeSelectionMode,
    #[serde(default)]
    pub adaptive_keyframes: rustsfm::AdaptiveKeyframeSelectionConfig,
    pub use_all_images: bool,
    pub use_gpu_sift: bool,
    pub use_gpu_matching: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum KeyframeSelectionMode {
    #[default]
    Adaptive,
    AllImages,
}

fn default_use_gpu_sift() -> bool {
    // The current wgpu SIFT path synchronizes multiple readbacks per image.
    // On macOS, the CPU VLFeat backend now extracts images in parallel and is
    // both faster and more feature-complete for the supported workflow.
    !cfg!(target_os = "macos")
}

impl Default for SfmConfigSnapshot {
    fn default() -> Self {
        Self {
            keyframe_selection: KeyframeSelectionMode::default(),
            adaptive_keyframes: rustsfm::AdaptiveKeyframeSelectionConfig::default(),
            use_all_images: true,
            use_gpu_sift: default_use_gpu_sift(),
            use_gpu_matching: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PnpConfigSnapshot {
    pub narrow_neighbors_each_side: usize,
    pub wide_neighbors_each_side: usize,
    pub min_inliers: usize,
    pub min_inlier_ratio: f64,
    pub max_reprojection_error: f64,
    pub use_gpu_pnp: bool,
}

impl Default for PnpConfigSnapshot {
    fn default() -> Self {
        Self {
            narrow_neighbors_each_side: 2,
            wide_neighbors_each_side: 4,
            min_inliers: 24,
            min_inlier_ratio: 0.20,
            max_reprojection_error: 4.0,
            use_gpu_pnp: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectManifest {
    pub(crate) schema_version: u32,
    pub(crate) id: Uuid,
    pub display_name: String,
    pub created_unix_ms: u64,
    pub updated_unix_ms: u64,
    pub source: SourceSpec,
    pub import_config: ImportConfigSnapshot,
    pub sfm_config: SfmConfigSnapshot,
    pub pnp_config: PnpConfigSnapshot,
    pub training_config: rustgs::TrainingConfig,
    pub(crate) stages: BTreeMap<ProjectStage, StageRecord>,
    pub active_scene: Option<ArtifactRef>,
    pub final_scene: Option<ArtifactRef>,
    pub compatibility: CompatibilityRecord,
    pub(crate) lease: Option<ProjectLease>,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ProjectManifestValidationError {
    #[error("unsupported project schema version {found}; expected {expected}")]
    UnsupportedSchemaVersion { expected: u32, found: u32 },
    #[error("project display name must not be empty")]
    EmptyDisplayName,
    #[error("project source identity must not be empty")]
    EmptySourceIdentity,
    #[error("project manifest is missing the {stage:?} stage")]
    MissingStage { stage: ProjectStage },
    #[error("{stage:?} progress is invalid: completed {completed} exceeds total {total}")]
    InvalidProgress {
        stage: ProjectStage,
        completed: u64,
        total: u64,
    },
    #[error("succeeded stage {stage:?} must declare at least one artifact")]
    SucceededStageWithoutArtifacts { stage: ProjectStage },
    #[error("invalid artifact at {location}: {source}")]
    InvalidArtifact {
        location: String,
        #[source]
        source: ArtifactValidationError,
    },
    #[error("invalid import config field {field}")]
    InvalidImportConfig { field: &'static str },
    #[error("invalid SfM config: {detail}")]
    InvalidSfmConfig { detail: String },
    #[error("invalid PnP config field {field}")]
    InvalidPnpConfig { field: &'static str },
    #[error("invalid training config: {detail}")]
    InvalidTrainingConfig { detail: String },
    #[error("lease project id {found} does not match manifest project id {expected}")]
    LeaseProjectMismatch { expected: Uuid, found: Uuid },
    #[error("lease for {stage:?} must have a nonzero attempt")]
    LeaseAttemptZero { stage: ProjectStage },
    #[error("lease attempt {found} for {stage:?} does not match stage attempt {expected}")]
    LeaseAttemptMismatch {
        stage: ProjectStage,
        expected: u32,
        found: u32,
    },
    #[error("project manifest contains multiple active stages: {stages:?}")]
    MultipleActiveStages { stages: Vec<ProjectStage> },
    #[error("active stage {stage:?} is missing its project lease")]
    ActiveStageWithoutLease { stage: ProjectStage },
    #[error("lease exists for {stage:?}, but no project stage is active")]
    LeaseWithoutActiveStage { stage: ProjectStage },
    #[error("lease targets {lease_stage:?}, but the active stage is {active_stage:?}")]
    LeaseActiveStageMismatch {
        lease_stage: ProjectStage,
        active_stage: ProjectStage,
    },
}

impl ProjectManifest {
    pub fn new(display_name: impl Into<String>, source: SourceSpec) -> Self {
        Self::new_at(display_name.into(), source, Uuid::new_v4(), unix_time_ms())
    }

    fn new_at(display_name: String, source: SourceSpec, id: Uuid, now_unix_ms: u64) -> Self {
        let stages = ProjectStage::ORDER
            .into_iter()
            .map(|stage| {
                let state = if stage == ProjectStage::Import {
                    StageState::Ready
                } else {
                    StageState::NotStarted
                };
                (stage, StageRecord::new(state, now_unix_ms))
            })
            .collect();

        Self {
            schema_version: PROJECT_SCHEMA_VERSION,
            id,
            display_name,
            created_unix_ms: now_unix_ms,
            updated_unix_ms: now_unix_ms,
            source,
            import_config: ImportConfigSnapshot::default(),
            sfm_config: SfmConfigSnapshot::default(),
            pnp_config: PnpConfigSnapshot::default(),
            training_config: rustgs::TrainingConfig::default(),
            stages,
            active_scene: None,
            final_scene: None,
            compatibility: CompatibilityRecord::default(),
            lease: None,
        }
    }

    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }

    pub fn id(&self) -> Uuid {
        self.id
    }

    pub fn lease(&self) -> Option<&ProjectLease> {
        self.lease.as_ref()
    }

    pub fn try_stage(
        &self,
        stage: ProjectStage,
    ) -> Result<&StageRecord, ProjectManifestValidationError> {
        self.stages
            .get(&stage)
            .ok_or(ProjectManifestValidationError::MissingStage { stage })
    }

    pub(crate) fn stage(&self, stage: ProjectStage) -> &StageRecord {
        self.try_stage(stage)
            .expect("ProjectManifest must be validated before state-machine use")
    }

    pub(crate) fn stage_mut(&mut self, stage: ProjectStage) -> &mut StageRecord {
        self.stages
            .get_mut(&stage)
            .expect("ProjectManifest must be validated before state-machine use")
    }

    /// Validates untrusted persisted data before it is used by the state machine.
    pub fn validate(&self) -> Result<(), ProjectManifestValidationError> {
        if self.schema_version != PROJECT_SCHEMA_VERSION {
            return Err(ProjectManifestValidationError::UnsupportedSchemaVersion {
                expected: PROJECT_SCHEMA_VERSION,
                found: self.schema_version,
            });
        }
        if self.display_name.trim().is_empty() {
            return Err(ProjectManifestValidationError::EmptyDisplayName);
        }
        if self.source.identity.trim().is_empty() {
            return Err(ProjectManifestValidationError::EmptySourceIdentity);
        }
        self.validate_configs()?;

        let mut active_stages = Vec::new();
        for stage in ProjectStage::ORDER {
            let record = self.try_stage(stage)?;
            if matches!(
                record.state,
                StageState::Running | StageState::PauseRequested | StageState::CancelRequested
            ) {
                active_stages.push(stage);
            }
            if let (Some(completed), Some(total)) = (record.completed, record.total) {
                if completed > total {
                    return Err(ProjectManifestValidationError::InvalidProgress {
                        stage,
                        completed,
                        total,
                    });
                }
            }
            if record.state == StageState::Succeeded && record.artifacts.is_empty() {
                return Err(
                    ProjectManifestValidationError::SucceededStageWithoutArtifacts { stage },
                );
            }
            for (index, artifact) in record.artifacts.iter().enumerate() {
                validate_stage_artifact_ref(stage, record.attempt, artifact).map_err(|source| {
                    ProjectManifestValidationError::InvalidArtifact {
                        location: format!("stage {stage:?} artifact {index}"),
                        source,
                    }
                })?;
            }
        }

        for (location, artifact) in [
            ("active_scene", self.active_scene.as_ref()),
            ("final_scene", self.final_scene.as_ref()),
        ] {
            if let Some(artifact) = artifact {
                validate_artifact_ref(artifact).map_err(|source| {
                    ProjectManifestValidationError::InvalidArtifact {
                        location: location.to_owned(),
                        source,
                    }
                })?;
            }
        }

        if active_stages.len() > 1 {
            return Err(ProjectManifestValidationError::MultipleActiveStages {
                stages: active_stages,
            });
        }
        match (active_stages.first().copied(), self.lease.as_ref()) {
            (None, None) => {}
            (Some(stage), None) => {
                return Err(ProjectManifestValidationError::ActiveStageWithoutLease { stage });
            }
            (None, Some(lease)) => {
                return Err(ProjectManifestValidationError::LeaseWithoutActiveStage {
                    stage: lease.stage,
                });
            }
            (Some(active_stage), Some(lease)) => {
                if lease.project_id != self.id {
                    return Err(ProjectManifestValidationError::LeaseProjectMismatch {
                        expected: self.id,
                        found: lease.project_id,
                    });
                }
                if lease.stage != active_stage {
                    return Err(ProjectManifestValidationError::LeaseActiveStageMismatch {
                        lease_stage: lease.stage,
                        active_stage,
                    });
                }
                if lease.attempt == 0 {
                    return Err(ProjectManifestValidationError::LeaseAttemptZero {
                        stage: lease.stage,
                    });
                }
                let record = self.try_stage(lease.stage)?;
                if lease.attempt != record.attempt {
                    return Err(ProjectManifestValidationError::LeaseAttemptMismatch {
                        stage: lease.stage,
                        expected: record.attempt,
                        found: lease.attempt,
                    });
                }
            }
        }

        Ok(())
    }

    fn validate_configs(&self) -> Result<(), ProjectManifestValidationError> {
        if !self.import_config.video_keyframes_per_second.is_finite()
            || self.import_config.video_keyframes_per_second <= 0.0
        {
            return Err(ProjectManifestValidationError::InvalidImportConfig {
                field: "video_keyframes_per_second",
            });
        }
        if self.import_config.maximum_keyframe_gap_us <= 0 {
            return Err(ProjectManifestValidationError::InvalidImportConfig {
                field: "maximum_keyframe_gap_us",
            });
        }
        if self.import_config.thumbnail_long_edge == 0 {
            return Err(ProjectManifestValidationError::InvalidImportConfig {
                field: "thumbnail_long_edge",
            });
        }

        self.sfm_config
            .adaptive_keyframes
            .validate()
            .map_err(|error| ProjectManifestValidationError::InvalidSfmConfig {
                detail: error.to_string(),
            })?;

        if self.pnp_config.narrow_neighbors_each_side == 0 {
            return Err(ProjectManifestValidationError::InvalidPnpConfig {
                field: "narrow_neighbors_each_side",
            });
        }
        if self.pnp_config.wide_neighbors_each_side < self.pnp_config.narrow_neighbors_each_side {
            return Err(ProjectManifestValidationError::InvalidPnpConfig {
                field: "wide_neighbors_each_side",
            });
        }
        if self.pnp_config.min_inliers == 0 {
            return Err(ProjectManifestValidationError::InvalidPnpConfig {
                field: "min_inliers",
            });
        }
        if !self.pnp_config.min_inlier_ratio.is_finite()
            || !(0.0..=1.0).contains(&self.pnp_config.min_inlier_ratio)
            || self.pnp_config.min_inlier_ratio == 0.0
        {
            return Err(ProjectManifestValidationError::InvalidPnpConfig {
                field: "min_inlier_ratio",
            });
        }
        if !self.pnp_config.max_reprojection_error.is_finite()
            || self.pnp_config.max_reprojection_error <= 0.0
        {
            return Err(ProjectManifestValidationError::InvalidPnpConfig {
                field: "max_reprojection_error",
            });
        }
        self.training_config.validate().map_err(|error| {
            ProjectManifestValidationError::InvalidTrainingConfig {
                detail: error.to_string(),
            }
        })?;
        Ok(())
    }
}

pub(crate) fn validate_artifact_ref(artifact: &ArtifactRef) -> Result<(), ArtifactValidationError> {
    let path = Path::new(&artifact.relative_path);
    if artifact.relative_path.trim().is_empty()
        || artifact.relative_path.contains('\\')
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(ArtifactValidationError::InvalidRelativePath(
            artifact.relative_path.clone(),
        ));
    }
    let valid_hash = artifact.content_hash.len() == 64
        && artifact
            .content_hash
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if !valid_hash {
        return Err(ArtifactValidationError::InvalidContentHash {
            relative_path: artifact.relative_path.clone(),
            content_hash: artifact.content_hash.clone(),
        });
    }
    parse_immutable_artifact_path(path, &artifact.relative_path)?;
    Ok(())
}

fn validate_stage_artifact_ref(
    stage: ProjectStage,
    current_attempt: u32,
    artifact: &ArtifactRef,
) -> Result<(), ArtifactValidationError> {
    validate_artifact_ref(artifact)?;
    let (artifact_stage, artifact_attempt) =
        parse_immutable_artifact_path(Path::new(&artifact.relative_path), &artifact.relative_path)?;
    if artifact_stage != stage || artifact_attempt > current_attempt {
        return Err(ArtifactValidationError::InvalidRelativePath(
            artifact.relative_path.clone(),
        ));
    }
    Ok(())
}

fn parse_immutable_artifact_path(
    path: &Path,
    original: &str,
) -> Result<(ProjectStage, u32), ArtifactValidationError> {
    let mut components = path.components();
    let Some(Component::Normal(root)) = components.next() else {
        return Err(ArtifactValidationError::InvalidRelativePath(
            original.to_owned(),
        ));
    };
    if root != "Artifacts" {
        return Err(ArtifactValidationError::InvalidRelativePath(
            original.to_owned(),
        ));
    }
    let Some(Component::Normal(stage_name)) = components.next() else {
        return Err(ArtifactValidationError::InvalidRelativePath(
            original.to_owned(),
        ));
    };
    let Some(stage) = ProjectStage::ORDER
        .into_iter()
        .find(|stage| stage.artifact_directory() == stage_name)
    else {
        return Err(ArtifactValidationError::InvalidRelativePath(
            original.to_owned(),
        ));
    };
    let Some(Component::Normal(attempt_name)) = components.next() else {
        return Err(ArtifactValidationError::InvalidRelativePath(
            original.to_owned(),
        ));
    };
    let Some(attempt) = attempt_name
        .to_str()
        .and_then(|name| name.strip_prefix("attempt-"))
        .filter(|digits| digits.len() == 8 && digits.bytes().all(|digit| digit.is_ascii_digit()))
        .and_then(|digits| digits.parse::<u32>().ok())
        .filter(|attempt| *attempt > 0)
    else {
        return Err(ArtifactValidationError::InvalidRelativePath(
            original.to_owned(),
        ));
    };
    if components.next().is_none() {
        return Err(ArtifactValidationError::InvalidRelativePath(
            original.to_owned(),
        ));
    }
    Ok((stage, attempt))
}

pub(crate) fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}
