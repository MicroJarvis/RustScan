use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageRecord {
    pub state: StageState,
    pub attempt: u32,
    pub completed: Option<u64>,
    pub total: Option<u64>,
    pub started_unix_ms: Option<u64>,
    pub updated_unix_ms: u64,
    pub artifacts: Vec<ArtifactRef>,
    pub error: Option<ProjectErrorRecord>,
}

impl StageRecord {
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
            rustgs_checkpoint_version: 1,
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
    pub use_all_images: bool,
    pub use_gpu_sift: bool,
    pub use_gpu_matching: bool,
}

impl Default for SfmConfigSnapshot {
    fn default() -> Self {
        Self {
            use_all_images: true,
            use_gpu_sift: true,
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
    pub schema_version: u32,
    pub id: Uuid,
    pub display_name: String,
    pub created_unix_ms: u64,
    pub updated_unix_ms: u64,
    pub source: SourceSpec,
    pub import_config: ImportConfigSnapshot,
    pub sfm_config: SfmConfigSnapshot,
    pub pnp_config: PnpConfigSnapshot,
    pub training_config: rustgs::TrainingConfig,
    pub stages: BTreeMap<ProjectStage, StageRecord>,
    pub active_scene: Option<ArtifactRef>,
    pub final_scene: Option<ArtifactRef>,
    pub compatibility: CompatibilityRecord,
    pub lease: Option<ProjectLease>,
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
}

pub(crate) fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}
