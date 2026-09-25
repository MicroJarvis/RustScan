use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use bincode::Options;
use serde::de::{self, SeqAccess, Visitor};
use serde::ser::{SerializeSeq, SerializeStruct};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tempfile::{NamedTempFile, TempPath};

use crate::{HostSplats, TrainingConfig, TrainingDataset, TrainingError};

use super::config::MAX_TRAINING_ITERATIONS;

pub const TRAINING_CHECKPOINT_VERSION: u32 = 3;
pub const TRAINING_CHECKPOINT_VERSION_V2: u32 = 2;
pub const TRAINING_CHECKPOINT_VERSION_V1: u32 = 1;
pub const TRAINING_CHECKPOINT_MAGIC: [u8; 8] = *b"RGSCPBIN";
pub const TRAINING_CHECKPOINT_FORMAT_VERSION: u32 = 1;
pub const MAX_TRAINING_CHECKPOINT_BYTES: u64 = 4 * 1024 * 1024 * 1024;
pub const MAX_TRAINING_IDENTITY_BYTES: usize = 4 * 1024;
pub const MAX_TRAINING_CHECKPOINT_SPLATS: usize = 1_000_000;
pub const MAX_TRAINING_CHECKPOINT_TENSOR_RANK: usize = 4;
pub const MAX_TRAINING_CHECKPOINT_TENSOR_ELEMENTS: usize = 1_000_000_000;

const TRAINING_CHECKPOINT_ENVELOPE_BYTES: u64 =
    TRAINING_CHECKPOINT_MAGIC.len() as u64 + size_of::<u32>() as u64;
static CHECKPOINT_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingIdentity {
    pub dataset: String,
    pub reconstruction: String,
    pub config: String,
}

impl Serialize for TrainingIdentity {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("TrainingIdentity", 3)?;
        state.serialize_field("dataset", &IdentityBytes(&self.dataset))?;
        state.serialize_field("reconstruction", &IdentityBytes(&self.reconstruction))?;
        state.serialize_field("config", &IdentityBytes(&self.config))?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for TrainingIdentity {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WireIdentity {
            dataset: IdentityString,
            reconstruction: IdentityString,
            config: IdentityString,
        }

        let identity = WireIdentity::deserialize(deserializer)?;
        Ok(Self {
            dataset: identity.dataset.0,
            reconstruction: identity.reconstruction.0,
            config: identity.config.0,
        })
    }
}

struct IdentityBytes<'a>(&'a str);

impl Serialize for IdentityBytes<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let bytes = self.0.as_bytes();
        if bytes.len() > MAX_TRAINING_IDENTITY_BYTES {
            return Err(serde::ser::Error::custom(format!(
                "identity field exceeds maximum length {MAX_TRAINING_IDENTITY_BYTES}"
            )));
        }
        let mut sequence = serializer.serialize_seq(Some(bytes.len()))?;
        for byte in bytes {
            sequence.serialize_element(byte)?;
        }
        sequence.end()
    }
}

struct IdentityString(String);

impl<'de> Deserialize<'de> for IdentityString {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer
            .deserialize_seq(IdentityStringVisitor)
            .map(IdentityString)
    }
}

struct IdentityStringVisitor;

impl<'de> Visitor<'de> for IdentityStringVisitor {
    type Value = String;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("an identity field encoded as a bounded UTF-8 byte sequence")
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let declared_len = sequence
            .size_hint()
            .ok_or_else(|| de::Error::custom("identity field must declare its length"))?;
        if declared_len > MAX_TRAINING_IDENTITY_BYTES {
            return Err(de::Error::custom(format!(
                "identity field exceeds maximum length {MAX_TRAINING_IDENTITY_BYTES}"
            )));
        }

        let mut bytes = Vec::with_capacity(declared_len);
        for index in 0..declared_len {
            let byte = sequence
                .next_element()?
                .ok_or_else(|| de::Error::invalid_length(index, &self))?;
            bytes.push(byte);
        }
        String::from_utf8(bytes)
            .map_err(|_| de::Error::custom("identity field must be valid UTF-8"))
    }
}

impl TrainingIdentity {
    pub fn from_inputs<P: AsRef<Path>>(
        dataset: &TrainingDataset,
        reconstruction: P,
        config: &TrainingConfig,
    ) -> Result<Self, TrainingError> {
        Ok(Self {
            dataset: hash_training_dataset(dataset)?,
            reconstruction: hash_reconstruction_path(reconstruction.as_ref())?,
            config: hash_training_config(config)?,
        })
    }

    pub fn from_canonical_content(
        dataset: &TrainingDataset,
        reconstruction_content: &[u8],
        config: &TrainingConfig,
    ) -> Result<Self, TrainingError> {
        Ok(Self {
            dataset: hash_training_dataset(dataset)?,
            reconstruction: blake3::hash(reconstruction_content).to_hex().to_string(),
            config: hash_training_config(config)?,
        })
    }

    pub(crate) fn validate_dataset_and_config(
        &self,
        dataset: &TrainingDataset,
        config: &TrainingConfig,
    ) -> Result<(), TrainingError> {
        match_checkpoint_dataset_identity(&self.dataset, dataset)?;
        if self.config != hash_training_config(config)? {
            return Err(TrainingError::InvalidInput(
                "training identity configuration does not match the current training configuration"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

/// How a checkpoint dataset hash relates to the live training dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetIdentityMatch {
    /// Digest matches the C5 stable-`image_id` encoding.
    Current,
    /// Digest matches the pre-C5 gap-free enumerated encoding (`0..n-1` with
    /// COLMAP `image_id` in `timestamp`). Only accepted when the live dataset
    /// has unique stable IDs (no oversample duplicates); gapped or filtered
    /// pre-C5 IDs cannot be reconstructed from the final pose list alone.
    PreC5GapFreeLegacy,
}

/// Compare a checkpoint dataset hash against the live dataset.
///
/// Used by both [`TrainingIdentity::validate_dataset_and_config`] and the real
/// resume entry [`crate::training::engine::runtime`] so compatibility cannot
/// diverge between helper validation and `prepare_resume_runtime`.
pub fn match_checkpoint_dataset_identity(
    checkpoint_dataset_hash: &str,
    dataset: &TrainingDataset,
) -> Result<DatasetIdentityMatch, TrainingError> {
    let current = hash_training_dataset(dataset)?;
    if checkpoint_dataset_hash == current {
        return Ok(DatasetIdentityMatch::Current);
    }

    if dataset_has_duplicate_stable_ids(dataset) {
        return Err(TrainingError::InvalidInput(format!(
            "checkpoint dataset does not match the current training dataset: \
             pre-C5 identity migration refuses oversampled/duplicate stable IDs \
             (stable-image_id hash={current}, checkpoint hash={checkpoint_dataset_hash}). \
             Re-train after the C5 frame-identity change."
        )));
    }

    let Some(legacy) = try_hash_training_dataset_pre_c5_gap_free(dataset)? else {
        return Err(TrainingError::InvalidInput(format!(
            "checkpoint dataset does not match the current training dataset \
             (stable-image_id hash={current}, checkpoint hash={checkpoint_dataset_hash}). \
             Pre-C5 gapped/filtered frame IDs cannot be reconstructed from the final \
             dataset alone; re-train or supply a checkpoint saved with stable image IDs."
        )));
    };

    if checkpoint_dataset_hash == legacy {
        log::warn!(
            "accepted pre-C5 training-dataset identity (gap-free enumerated frame_id + image_id timestamp); \
             re-save the checkpoint to persist the stable-image_id identity"
        );
        return Ok(DatasetIdentityMatch::PreC5GapFreeLegacy);
    }

    if current == legacy {
        // Dataset shape makes the two encodings identical; keep the historical
        // short rejection text for unchanged callers/tests.
        return Err(TrainingError::InvalidInput(
            "checkpoint dataset does not match the current training dataset".to_string(),
        ));
    }

    Err(TrainingError::InvalidInput(format!(
        "checkpoint dataset does not match the current training dataset \
         (stable-image_id hash={current}, pre-C5 gap-free enumerated hash={legacy}, \
         checkpoint hash={checkpoint_dataset_hash}). Gapped pre-C5 IDs (missing images, \
         exclude filters) cannot be reconstructed by re-enumerating the final dataset; \
         re-train or migrate with a stable-image_id checkpoint."
    )))
}

/// Frozen pre-C5 COLMAP loader ID assignment.
///
/// Candidates are the post-`take(max_frames)` / `step_by(stride)` COLMAP image
/// list in loader order. Enumeration includes missing files; skips leave gaps
/// (e.g. exists=[true,false,true] → old frame IDs `[0, 2]`).
#[must_use]
pub fn assign_pre_c5_enumerated_ids(candidates: &[(u64, bool)]) -> Vec<(u64, u64)> {
    candidates
        .iter()
        .enumerate()
        .filter_map(|(frame_idx, &(stable_id, exists))| {
            exists.then_some((frame_idx as u64, stable_id))
        })
        .collect()
}

/// Hash a dataset using explicit pre-C5 `(enumerated frame_id, image_id timestamp)` rows.
///
/// `old_frame_ids` must align 1:1 with `dataset.poses`. Intended for frozen-fixture
/// tests and for callers that still have the original COLMAP candidate list.
pub fn hash_training_dataset_with_pre_c5_frame_ids(
    dataset: &TrainingDataset,
    old_frame_ids: &[u64],
) -> Result<String, TrainingError> {
    if old_frame_ids.len() != dataset.poses.len() {
        return Err(TrainingError::InvalidInput(format!(
            "pre-C5 frame_id mapping length {} does not match dataset pose count {}",
            old_frame_ids.len(),
            dataset.poses.len()
        )));
    }
    let poses = dataset
        .poses
        .iter()
        .zip(old_frame_ids.iter())
        .map(|(pose, &old_frame_id)| {
            Ok(CanonicalTrainingPose {
                frame_id: old_frame_id,
                image_content: hash_file_content(&pose.image_path)?,
                depth_content: pose
                    .depth_path
                    .as_deref()
                    .map(hash_file_content)
                    .transpose()?,
                pose: &pose.pose,
                timestamp: pose.frame_id as f64,
            })
        })
        .collect::<Result<Vec<_>, TrainingError>>()?;
    hash_canonical_dataset(dataset, poses)
}

fn dataset_has_duplicate_stable_ids(dataset: &TrainingDataset) -> bool {
    let mut seen = std::collections::HashSet::new();
    dataset.poses.iter().any(|pose| !seen.insert(pose.frame_id))
}

/// Gap-free pre-C5 digest: only valid when old IDs were exactly `0..n-1` in pose order.
fn try_hash_training_dataset_pre_c5_gap_free(
    dataset: &TrainingDataset,
) -> Result<Option<String>, TrainingError> {
    if dataset_has_duplicate_stable_ids(dataset) {
        return Ok(None);
    }
    let old_ids: Vec<u64> = (0..dataset.poses.len() as u64).collect();
    Ok(Some(hash_training_dataset_with_pre_c5_frame_ids(
        dataset, &old_ids,
    )?))
}

#[derive(Serialize)]
struct CanonicalTrainingDataset<'a> {
    intrinsics: &'a crate::Intrinsics,
    depth_scale: f32,
    poses: Vec<CanonicalTrainingPose<'a>>,
    initial_points: &'a [([f32; 3], Option<[f32; 3]>)],
}

#[derive(Serialize)]
struct CanonicalTrainingPose<'a> {
    frame_id: u64,
    image_content: [u8; 32],
    depth_content: Option<[u8; 32]>,
    pose: &'a crate::SE3,
    timestamp: f64,
}

fn hash_training_dataset(dataset: &TrainingDataset) -> Result<String, TrainingError> {
    let poses = dataset
        .poses
        .iter()
        .map(|pose| {
            Ok(CanonicalTrainingPose {
                frame_id: pose.frame_id,
                image_content: hash_file_content(&pose.image_path)?,
                depth_content: pose
                    .depth_path
                    .as_deref()
                    .map(hash_file_content)
                    .transpose()?,
                pose: &pose.pose,
                timestamp: pose.timestamp,
            })
        })
        .collect::<Result<Vec<_>, TrainingError>>()?;
    hash_canonical_dataset(dataset, poses)
}

fn hash_canonical_dataset(
    dataset: &TrainingDataset,
    poses: Vec<CanonicalTrainingPose<'_>>,
) -> Result<String, TrainingError> {
    let canonical = CanonicalTrainingDataset {
        intrinsics: &dataset.intrinsics,
        depth_scale: dataset.depth_scale,
        poses,
        initial_points: &dataset.initial_points,
    };
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| TrainingError::InvalidInput(error.to_string()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn hash_training_config(config: &TrainingConfig) -> Result<String, TrainingError> {
    // Historical (701d051) continuity fingerprint: clone config, zero iterations,
    // then `serde_json::to_vec` the **struct** (declaration order). Profiler is a
    // measurement-only field added later — omit it via a continuity-shaped view
    // that matches the pre-profiler TrainingConfig layout. Do **not** route through
    // `serde_json::Value` (BTreeMap key order ≠ struct field order).
    let bytes = continuity_fingerprint_bytes(config)?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

/// Bytes hashed for training-config continuity (iterations=0, no profiler).
///
/// Layout mirrors `TrainingConfig` at commit `701d051` (pre-profiler).
fn continuity_fingerprint_bytes(config: &TrainingConfig) -> Result<Vec<u8>, TrainingError> {
    #[derive(Serialize)]
    struct ContinuityFingerprintConfig<'a> {
        backend: &'a super::config::TrainingBackend,
        iterations: usize,
        #[serde(flatten)]
        optimizer: &'a super::config::TrainingOptimizerConfig,
        #[serde(flatten)]
        loss: &'a super::config::TrainingLossConfig,
        #[serde(flatten)]
        initialization: &'a super::config::TrainingInitializationConfig,
        #[serde(flatten)]
        data: &'a super::config::TrainingDataConfig,
        #[serde(flatten)]
        raster: &'a super::config::TrainingRasterConfig,
        litegs: &'a super::config::LiteGsConfig,
    }

    let fingerprint = ContinuityFingerprintConfig {
        backend: &config.backend,
        iterations: 0,
        optimizer: &config.optimizer,
        loss: &config.loss,
        initialization: &config.initialization,
        data: &config.data,
        raster: &config.raster,
        litegs: &config.litegs,
    };
    serde_json::to_vec(&fingerprint).map_err(|error| TrainingError::InvalidInput(error.to_string()))
}

fn hash_reconstruction_path(path: &Path) -> Result<String, TrainingError> {
    let canonical = fs::canonicalize(path)?;
    let metadata = fs::metadata(&canonical)?;
    if metadata.is_file() {
        return Ok(blake3::Hash::from_bytes(hash_file_content(&canonical)?)
            .to_hex()
            .to_string());
    }
    if !metadata.is_dir() {
        return Err(TrainingError::InvalidInput(format!(
            "reconstruction input {} is neither a file nor a directory",
            path.display()
        )));
    }

    let mut files = Vec::new();
    collect_reconstruction_files(&canonical, &canonical, &mut files)?;
    files.sort_by(|left, right| left.0.cmp(&right.0));
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"RustGS reconstruction directory\0");
    for (relative, file_path) in files {
        update_length_prefixed(&mut hasher, relative.as_bytes());
        hasher.update(&hash_file_content(&file_path)?);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn collect_reconstruction_files(
    root: &Path,
    directory: &Path,
    files: &mut Vec<(String, PathBuf)>,
) -> Result<(), TrainingError> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        if file_type.is_dir() {
            collect_reconstruction_files(root, &path, files)?;
        } else if file_type.is_file() {
            let relative = path.strip_prefix(root).map_err(|error| {
                TrainingError::InvalidInput(format!(
                    "cannot canonicalize reconstruction path {}: {error}",
                    path.display()
                ))
            })?;
            let canonical_relative = relative
                .components()
                .map(|component| component.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/");
            files.push((canonical_relative, path));
        } else {
            return Err(TrainingError::InvalidInput(format!(
                "unsupported reconstruction entry {}",
                path.display()
            )));
        }
    }
    Ok(())
}

fn hash_file_content(path: &Path) -> Result<[u8; 32], TrainingError> {
    let mut file = File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(*hasher.finalize().as_bytes())
}

fn update_length_prefixed(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorCheckpoint {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdamParameterCheckpoint {
    pub moment1: Option<TensorCheckpoint>,
    pub moment2: Option<TensorCheckpoint>,
    pub scaling: Option<TensorCheckpoint>,
    pub step: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointMigration {
    None,
    V1BaselineReset,
    /// Loaded a v2 checkpoint that predates selection metadata; fields are absent.
    V2SelectionMetaAbsent,
}

/// Frame-selection metadata persisted beside training state.
///
/// Distinguishes canonical selection IDs from the oversampled/shuffled loader
/// order. Missing on v1/v2 files — never invent values when loading legacy bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct CheckpointFrameSelectionMeta {
    /// `in-view` / `holdout` when evaluation split is known; `None` if unset.
    pub eval_split_kind: Option<String>,
    /// Canonical train FrameSelection stable IDs (full u64).
    pub train_stable_ids: Vec<u64>,
    /// Canonical eval FrameSelection stable IDs (full u64); empty when unused.
    pub eval_stable_ids: Vec<u64>,
    /// Pose order that entered the training loader after oversample/shuffle.
    pub train_loader_frame_ids: Vec<u64>,
    pub manifest_fingerprint: Option<String>,
    pub selection_fingerprint: Option<String>,
    pub eval_selection_fingerprint: Option<String>,
}

impl CheckpointFrameSelectionMeta {
    /// True when this record carries any selection fingerprint or non-empty ID list.
    #[must_use]
    pub fn is_populated(&self) -> bool {
        self.eval_split_kind.is_some()
            || !self.train_stable_ids.is_empty()
            || !self.eval_stable_ids.is_empty()
            || !self.train_loader_frame_ids.is_empty()
            || self.manifest_fingerprint.is_some()
            || self.selection_fingerprint.is_some()
            || self.eval_selection_fingerprint.is_some()
    }
}

/// Resolve selection metadata for a training run / resume.
///
/// Rules:
/// - both absent (v1/v2 or never recorded) → remain absent (do not invent)
/// - checkpoint absent, caller provides → use caller
/// - checkpoint present, caller absent → validate against dataset then inherit
/// - both present → must match; otherwise reject
pub fn resolve_checkpoint_selection(
    checkpoint_selection: Option<&CheckpointFrameSelectionMeta>,
    provided_selection: Option<&CheckpointFrameSelectionMeta>,
    dataset: &TrainingDataset,
) -> Result<Option<CheckpointFrameSelectionMeta>, TrainingError> {
    let resolved = match (checkpoint_selection, provided_selection) {
        (None, None) => None,
        (None, Some(provided)) => Some(provided.clone()),
        (Some(checkpoint), None) => Some(checkpoint.clone()),
        (Some(checkpoint), Some(provided)) if checkpoint == provided => Some(checkpoint.clone()),
        (Some(checkpoint), Some(provided)) => {
            return Err(TrainingError::InvalidInput(format!(
                "checkpoint frame-selection metadata does not match the current training selection \
                 (checkpoint selection_fingerprint={:?}, current selection_fingerprint={:?})",
                checkpoint.selection_fingerprint, provided.selection_fingerprint
            )));
        }
    };

    if let Some(meta) = resolved.as_ref() {
        validate_selection_meta_against_dataset(meta, dataset)?;
    }
    Ok(resolved)
}

/// Reject resume when both sides recorded selection metadata and they disagree.
pub fn validate_checkpoint_selection_consistency(
    checkpoint: Option<&CheckpointFrameSelectionMeta>,
    expected: Option<&CheckpointFrameSelectionMeta>,
) -> Result<(), TrainingError> {
    match (checkpoint, expected) {
        (None, _) | (_, None) => Ok(()),
        (Some(left), Some(right)) if left == right => Ok(()),
        (Some(left), Some(right)) => Err(TrainingError::InvalidInput(format!(
            "checkpoint frame-selection metadata does not match the current training selection \
             (checkpoint selection_fingerprint={:?}, current selection_fingerprint={:?})",
            left.selection_fingerprint, right.selection_fingerprint
        ))),
    }
}

fn validate_selection_meta_against_dataset(
    meta: &CheckpointFrameSelectionMeta,
    dataset: &TrainingDataset,
) -> Result<(), TrainingError> {
    let available: std::collections::HashSet<u64> =
        dataset.poses.iter().map(|pose| pose.frame_id).collect();
    for (label, ids) in [
        ("train_stable_ids", meta.train_stable_ids.as_slice()),
        ("eval_stable_ids", meta.eval_stable_ids.as_slice()),
        (
            "train_loader_frame_ids",
            meta.train_loader_frame_ids.as_slice(),
        ),
    ] {
        for &id in ids {
            if !available.contains(&id) {
                return Err(TrainingError::InvalidInput(format!(
                    "checkpoint frame-selection metadata cannot be verified against the current \
                     training dataset: {label} contains unknown stable id {id}"
                )));
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdamCheckpoint {
    pub transforms: AdamParameterCheckpoint,
    pub sh_coeffs: AdamParameterCheckpoint,
    pub raw_opacities: AdamParameterCheckpoint,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopologyCheckpoint {
    pub grad_2d: TensorCheckpoint,
    pub screen_grad_2d: TensorCheckpoint,
    pub abs_grad_2d: TensorCheckpoint,
    pub abs_pixel_grad_2d: TensorCheckpoint,
    pub pixel_coverage: TensorCheckpoint,
    pub camera_depth: TensorCheckpoint,
    pub grad_color: TensorCheckpoint,
    pub num_observations: TensorCheckpoint,
    pub visible_observations: TensorCheckpoint,
    pub actual_visible_observations: TensorCheckpoint,
    pub splat_birth_iterations: Vec<usize>,
    pub splat_invisible_windows: Vec<usize>,
    pub visibility_window_baseline: Vec<f32>,
    pub actual_visibility_window_baseline: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct TopologyCheckpointV1 {
    grad_2d: TensorCheckpoint,
    screen_grad_2d: TensorCheckpoint,
    abs_grad_2d: TensorCheckpoint,
    abs_pixel_grad_2d: TensorCheckpoint,
    pixel_coverage: TensorCheckpoint,
    camera_depth: TensorCheckpoint,
    grad_color: TensorCheckpoint,
    num_observations: TensorCheckpoint,
    visible_observations: TensorCheckpoint,
    actual_visible_observations: TensorCheckpoint,
    splat_birth_iterations: Vec<usize>,
    splat_invisible_windows: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct TrainingCheckpointV1 {
    version: u32,
    identity: TrainingIdentity,
    completed_iterations: usize,
    latest_loss: Option<f32>,
    splats: HostSplats,
    optimizer: AdamCheckpoint,
    topology: TopologyCheckpointV1,
    frame_shuffle_seed: u64,
    active_sh_degree: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct TrainingCheckpointV2 {
    version: u32,
    identity: TrainingIdentity,
    completed_iterations: usize,
    latest_loss: Option<f32>,
    splats: HostSplats,
    optimizer: AdamCheckpoint,
    topology: TopologyCheckpoint,
    frame_shuffle_seed: u64,
    active_sh_degree: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    pub version: u32,
    pub identity: TrainingIdentity,
    pub completed_iterations: usize,
    pub latest_loss: Option<f32>,
    pub splats: HostSplats,
    pub optimizer: AdamCheckpoint,
    pub topology: TopologyCheckpoint,
    pub frame_shuffle_seed: u64,
    pub active_sh_degree: usize,
    /// Present on v3+ saves that recorded selection; `None` means absent (v1/v2
    /// migration or intentionally omitted) — never fabricate IDs/fingerprints.
    pub selection: Option<CheckpointFrameSelectionMeta>,
}

impl TrainingCheckpoint {
    pub fn validate(&self) -> Result<(), TrainingError> {
        if self.version != TRAINING_CHECKPOINT_VERSION {
            return Err(invalid_checkpoint(format!(
                "checkpoint version {} is unsupported; expected {TRAINING_CHECKPOINT_VERSION}",
                self.version
            )));
        }
        validate_identity_field("dataset", &self.identity.dataset)?;
        validate_identity_field("reconstruction", &self.identity.reconstruction)?;
        validate_identity_field("config", &self.identity.config)?;
        if self.latest_loss.is_some_and(|loss| !loss.is_finite()) {
            return Err(invalid_checkpoint("latest loss must be finite"));
        }
        if self.completed_iterations > MAX_TRAINING_ITERATIONS {
            return Err(invalid_checkpoint(format!(
                "completed iterations {} exceeds maximum safe step {MAX_TRAINING_ITERATIONS}",
                self.completed_iterations
            )));
        }
        if self.splats.len() > MAX_TRAINING_CHECKPOINT_SPLATS {
            return Err(invalid_checkpoint(format!(
                "splat count exceeds maximum {MAX_TRAINING_CHECKPOINT_SPLATS}"
            )));
        }
        if self.splats.sh_degree() > 3 {
            return Err(invalid_checkpoint(format!(
                "stored SH degree exceeds supported maximum 3: got {}",
                self.splats.sh_degree()
            )));
        }
        self.splats.validate().map_err(|error| {
            invalid_checkpoint(format!("checkpoint splats are invalid: {error}"))
        })?;
        if self.active_sh_degree > self.splats.sh_degree() {
            return Err(invalid_checkpoint(format!(
                "active SH degree {} exceeds stored SH degree {}",
                self.active_sh_degree,
                self.splats.sh_degree()
            )));
        }

        let splat_count = self.splats.len();
        let sh_coeff_count = self.splats.sh_coeffs_row_width() / 3;
        validate_adam_parameter(
            "optimizer.transforms",
            &self.optimizer.transforms,
            &[splat_count, 10],
            &[1, 10],
            self.completed_iterations,
        )?;
        validate_adam_parameter(
            "optimizer.sh_coeffs",
            &self.optimizer.sh_coeffs,
            &[splat_count, sh_coeff_count, 3],
            &[1, sh_coeff_count.max(1), 1],
            self.completed_iterations,
        )?;
        validate_adam_parameter(
            "optimizer.raw_opacities",
            &self.optimizer.raw_opacities,
            &[splat_count],
            &[1],
            self.completed_iterations,
        )?;
        if self.optimizer.transforms.step != self.optimizer.sh_coeffs.step
            || self.optimizer.transforms.step != self.optimizer.raw_opacities.step
        {
            return Err(invalid_checkpoint(
                "optimizer parameter steps must be equal",
            ));
        }

        validate_topology_tensor("topology.grad_2d", &self.topology.grad_2d, splat_count)?;
        validate_topology_tensor(
            "topology.screen_grad_2d",
            &self.topology.screen_grad_2d,
            splat_count,
        )?;
        validate_topology_tensor(
            "topology.abs_grad_2d",
            &self.topology.abs_grad_2d,
            splat_count,
        )?;
        validate_topology_tensor(
            "topology.abs_pixel_grad_2d",
            &self.topology.abs_pixel_grad_2d,
            splat_count,
        )?;
        validate_topology_tensor(
            "topology.pixel_coverage",
            &self.topology.pixel_coverage,
            splat_count,
        )?;
        validate_topology_tensor(
            "topology.camera_depth",
            &self.topology.camera_depth,
            splat_count,
        )?;
        validate_topology_tensor(
            "topology.grad_color",
            &self.topology.grad_color,
            splat_count,
        )?;
        validate_topology_tensor(
            "topology.num_observations",
            &self.topology.num_observations,
            splat_count,
        )?;
        validate_topology_tensor(
            "topology.visible_observations",
            &self.topology.visible_observations,
            splat_count,
        )?;
        validate_topology_tensor(
            "topology.actual_visible_observations",
            &self.topology.actual_visible_observations,
            splat_count,
        )?;
        validate_topology_vector(
            "topology.splat_birth_iterations",
            self.topology.splat_birth_iterations.len(),
            splat_count,
        )?;
        validate_topology_vector(
            "topology.splat_invisible_windows",
            self.topology.splat_invisible_windows.len(),
            splat_count,
        )?;
        validate_visibility_baseline(
            "topology.visibility_window_baseline",
            &self.topology.visibility_window_baseline,
            splat_count,
        )?;
        validate_visibility_baseline(
            "topology.actual_visibility_window_baseline",
            &self.topology.actual_visibility_window_baseline,
            splat_count,
        )?;
        if self
            .topology
            .splat_birth_iterations
            .iter()
            .any(|&iteration| iteration > self.completed_iterations)
        {
            return Err(invalid_checkpoint(
                "topology.splat_birth_iterations contains future iteration",
            ));
        }
        if self
            .topology
            .splat_invisible_windows
            .iter()
            .any(|&window| window > self.completed_iterations)
        {
            return Err(invalid_checkpoint(
                "topology.splat_invisible_windows exceeds completed iterations",
            ));
        }
        Ok(())
    }
}

fn validate_visibility_baseline(
    label: &str,
    values: &[f32],
    splat_count: usize,
) -> Result<(), TrainingError> {
    if values.len() != splat_count {
        return Err(invalid_checkpoint(format!(
            "{label} length {} does not match splat count {splat_count}",
            values.len()
        )));
    }
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(invalid_checkpoint(format!(
                "{label}[{index}] must be finite"
            )));
        }
        if *value < 0.0 {
            return Err(invalid_checkpoint(format!(
                "{label}[{index}] must be non-negative"
            )));
        }
    }
    Ok(())
}

fn migrate_topology_v1(topology: TopologyCheckpointV1) -> TopologyCheckpoint {
    let visibility_window_baseline =
        visibility_window_baseline_from_cumulative_values(&topology.visible_observations.values);
    let actual_visibility_window_baseline = visibility_window_baseline_from_cumulative_values(
        &topology.actual_visible_observations.values,
    );
    TopologyCheckpoint {
        grad_2d: topology.grad_2d,
        screen_grad_2d: topology.screen_grad_2d,
        abs_grad_2d: topology.abs_grad_2d,
        abs_pixel_grad_2d: topology.abs_pixel_grad_2d,
        pixel_coverage: topology.pixel_coverage,
        camera_depth: topology.camera_depth,
        grad_color: topology.grad_color,
        num_observations: topology.num_observations,
        visible_observations: topology.visible_observations,
        actual_visible_observations: topology.actual_visible_observations,
        splat_birth_iterations: topology.splat_birth_iterations,
        splat_invisible_windows: topology.splat_invisible_windows,
        visibility_window_baseline,
        actual_visibility_window_baseline,
    }
}

fn visibility_window_baseline_from_cumulative_values(cumulative: &[f32]) -> Vec<f32> {
    cumulative
        .iter()
        .map(|value| {
            if value.is_finite() {
                (*value).max(0.0)
            } else {
                0.0
            }
        })
        .collect()
}

fn migrate_checkpoint_v2(v2: TrainingCheckpointV2) -> (TrainingCheckpoint, CheckpointMigration) {
    (
        TrainingCheckpoint {
            version: TRAINING_CHECKPOINT_VERSION,
            identity: v2.identity,
            completed_iterations: v2.completed_iterations,
            latest_loss: v2.latest_loss,
            splats: v2.splats,
            optimizer: v2.optimizer,
            topology: v2.topology,
            frame_shuffle_seed: v2.frame_shuffle_seed,
            active_sh_degree: v2.active_sh_degree,
            selection: None,
        },
        CheckpointMigration::V2SelectionMetaAbsent,
    )
}

fn migrate_checkpoint_v1(v1: TrainingCheckpointV1) -> (TrainingCheckpoint, CheckpointMigration) {
    let (mut checkpoint, _) = migrate_checkpoint_v2(TrainingCheckpointV2 {
        version: TRAINING_CHECKPOINT_VERSION_V2,
        identity: v1.identity,
        completed_iterations: v1.completed_iterations,
        latest_loss: v1.latest_loss,
        splats: v1.splats,
        optimizer: v1.optimizer,
        topology: migrate_topology_v1(v1.topology),
        frame_shuffle_seed: v1.frame_shuffle_seed,
        active_sh_degree: v1.active_sh_degree,
    });
    checkpoint.version = TRAINING_CHECKPOINT_VERSION;
    (checkpoint, CheckpointMigration::V1BaselineReset)
}

pub fn save_training_checkpoint(
    path: &Path,
    checkpoint: &TrainingCheckpoint,
) -> Result<(), TrainingError> {
    checkpoint.validate()?;
    let parent = checkpoint_parent(path);
    fs::create_dir_all(parent)?;

    let temp = create_unique_checkpoint_temp(path, parent)?;
    write_and_commit_checkpoint(path, parent, temp, checkpoint)
}

pub fn load_training_checkpoint(path: &Path) -> Result<TrainingCheckpoint, TrainingError> {
    Ok(load_training_checkpoint_with_migration(path)?.0)
}

pub fn load_training_checkpoint_with_migration(
    path: &Path,
) -> Result<(TrainingCheckpoint, CheckpointMigration), TrainingError> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    if file_len > MAX_TRAINING_CHECKPOINT_BYTES {
        return Err(TrainingError::InvalidInput(format!(
            "checkpoint file size {file_len} exceeds maximum size {MAX_TRAINING_CHECKPOINT_BYTES}"
        )));
    }
    if file_len < TRAINING_CHECKPOINT_ENVELOPE_BYTES {
        return Err(TrainingError::InvalidInput(
            "decode checkpoint header: file is truncated".to_string(),
        ));
    }

    let mut reader = BufReader::new(file);
    let mut magic = [0u8; TRAINING_CHECKPOINT_MAGIC.len()];
    reader.read_exact(&mut magic)?;
    if magic != TRAINING_CHECKPOINT_MAGIC {
        return Err(TrainingError::InvalidInput(
            "checkpoint magic does not match the RustGS checkpoint format".to_string(),
        ));
    }
    let mut format_version = [0u8; size_of::<u32>()];
    reader.read_exact(&mut format_version)?;
    let format_version = u32::from_le_bytes(format_version);
    if format_version != TRAINING_CHECKPOINT_FORMAT_VERSION {
        return Err(TrainingError::InvalidInput(format!(
            "checkpoint format version {format_version} is unsupported; expected {TRAINING_CHECKPOINT_FORMAT_VERSION}"
        )));
    }

    let mut payload = Vec::new();
    reader.read_to_end(&mut payload)?;
    let (checkpoint, migration) = decode_training_checkpoint_payload(&payload)?;
    checkpoint.validate()?;
    Ok((checkpoint, migration))
}

fn decode_training_checkpoint_payload(
    payload: &[u8],
) -> Result<(TrainingCheckpoint, CheckpointMigration), TrainingError> {
    if let Ok(checkpoint) = checkpoint_bincode_options().deserialize::<TrainingCheckpoint>(payload)
    {
        if checkpoint.version == TRAINING_CHECKPOINT_VERSION {
            return Ok((checkpoint, CheckpointMigration::None));
        }
        return Err(invalid_checkpoint(format!(
            "checkpoint version {} is unsupported for v3 layout; expected {TRAINING_CHECKPOINT_VERSION}",
            checkpoint.version
        )));
    }

    if let Ok(v2) = checkpoint_bincode_options().deserialize::<TrainingCheckpointV2>(payload) {
        if v2.version == TRAINING_CHECKPOINT_VERSION_V2 {
            return Ok(migrate_checkpoint_v2(v2));
        }
        if v2.version == TRAINING_CHECKPOINT_VERSION_V1 {
            return Err(invalid_checkpoint(
                "checkpoint version 1 payload was decoded as v2 layout; file is corrupt",
            ));
        }
        return Err(invalid_checkpoint(format!(
            "checkpoint version {} is unsupported for v2 layout; expected {TRAINING_CHECKPOINT_VERSION_V2}",
            v2.version
        )));
    }

    let v1: TrainingCheckpointV1 = checkpoint_bincode_options()
        .deserialize(payload)
        .map_err(|error| decode_checkpoint_error(*error))?;
    if v1.version != TRAINING_CHECKPOINT_VERSION_V1 {
        return Err(invalid_checkpoint(format!(
            "checkpoint version {} is unsupported; expected {TRAINING_CHECKPOINT_VERSION_V1} for v1 layout",
            v1.version
        )));
    }
    Ok(migrate_checkpoint_v1(v1))
}

fn validate_identity_field(name: &str, value: &str) -> Result<(), TrainingError> {
    if value.trim().is_empty() {
        return Err(invalid_checkpoint(format!(
            "identity {name} must not be empty"
        )));
    }
    if value.len() > MAX_TRAINING_IDENTITY_BYTES {
        return Err(invalid_checkpoint(format!(
            "identity {name} exceeds maximum length {MAX_TRAINING_IDENTITY_BYTES}"
        )));
    }
    Ok(())
}

fn validate_adam_parameter(
    name: &str,
    parameter: &AdamParameterCheckpoint,
    parameter_shape: &[usize],
    scaling_shape: &[usize],
    completed_iterations: usize,
) -> Result<(), TrainingError> {
    if parameter.step > completed_iterations {
        return Err(invalid_checkpoint(format!(
            "{name}.step must not exceed completed iterations {completed_iterations}, got {}",
            parameter.step
        )));
    }
    if parameter.moment1.is_some() != parameter.moment2.is_some() {
        return Err(invalid_checkpoint(format!(
            "{name}.moment1 and moment2 must both be present or both be absent"
        )));
    }
    if parameter.step == 0 && parameter.moment1.is_some() {
        return Err(invalid_checkpoint(format!(
            "{name} moments must be absent when step is zero"
        )));
    }
    if parameter.step > 0 && parameter.moment1.is_none() {
        return Err(invalid_checkpoint(format!(
            "{name} moments must be present when step is non-zero"
        )));
    }
    if let Some(moment1) = &parameter.moment1 {
        validate_tensor_shape(&format!("{name}.moment1"), moment1, parameter_shape)?;
    }
    if let Some(moment2) = &parameter.moment2 {
        validate_tensor_shape(&format!("{name}.moment2"), moment2, parameter_shape)?;
    }
    if let Some(scaling) = &parameter.scaling {
        validate_tensor_shape(&format!("{name}.scaling"), scaling, scaling_shape)?;
    }
    Ok(())
}

fn validate_tensor_shape(
    name: &str,
    tensor: &TensorCheckpoint,
    expected_shape: &[usize],
) -> Result<(), TrainingError> {
    validate_tensor(name, tensor)?;
    if tensor.shape != expected_shape {
        return Err(invalid_checkpoint(format!(
            "{name} expected shape {expected_shape:?}, got {:?}",
            tensor.shape
        )));
    }
    Ok(())
}

fn validate_topology_tensor(
    name: &str,
    tensor: &TensorCheckpoint,
    splat_count: usize,
) -> Result<(), TrainingError> {
    validate_tensor(name, tensor)?;
    if tensor.shape.as_slice() != [splat_count] {
        return Err(invalid_checkpoint(format!(
            "{name} must have shape [{splat_count}], got {:?}",
            tensor.shape
        )));
    }
    Ok(())
}

fn validate_tensor(name: &str, tensor: &TensorCheckpoint) -> Result<(), TrainingError> {
    if tensor.shape.len() > MAX_TRAINING_CHECKPOINT_TENSOR_RANK {
        return Err(invalid_checkpoint(format!(
            "{name} tensor rank {} exceeds maximum {MAX_TRAINING_CHECKPOINT_TENSOR_RANK}",
            tensor.shape.len()
        )));
    }
    let expected = tensor
        .shape
        .iter()
        .try_fold(1usize, |product, &dimension| product.checked_mul(dimension));
    let Some(expected) = expected else {
        return Err(invalid_checkpoint(format!(
            "{name} tensor shape overflows usize"
        )));
    };
    if expected > MAX_TRAINING_CHECKPOINT_TENSOR_ELEMENTS {
        return Err(invalid_checkpoint(format!(
            "{name} tensor element count {expected} exceeds maximum {MAX_TRAINING_CHECKPOINT_TENSOR_ELEMENTS}"
        )));
    }
    if tensor.values.len() != expected {
        return Err(invalid_checkpoint(format!(
            "{name} tensor shape expects {expected} values, got {}",
            tensor.values.len()
        )));
    }
    if tensor.values.iter().any(|value| !value.is_finite()) {
        return Err(invalid_checkpoint(format!(
            "{name} tensor values must be finite"
        )));
    }
    Ok(())
}

fn validate_topology_vector(
    name: &str,
    actual: usize,
    splat_count: usize,
) -> Result<(), TrainingError> {
    if actual != splat_count {
        return Err(invalid_checkpoint(format!(
            "{name} must contain {splat_count} values, got {actual}"
        )));
    }
    Ok(())
}

fn invalid_checkpoint(message: impl Into<String>) -> TrainingError {
    TrainingError::InvalidInput(format!("invalid training checkpoint: {}", message.into()))
}

fn checkpoint_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn write_and_commit_checkpoint(
    path: &Path,
    parent: &Path,
    mut temp: NamedTempFile,
    checkpoint: &TrainingCheckpoint,
) -> Result<(), TrainingError> {
    let file = temp.as_file_mut();
    file.write_all(&TRAINING_CHECKPOINT_MAGIC)?;
    file.write_all(&TRAINING_CHECKPOINT_FORMAT_VERSION.to_le_bytes())?;
    checkpoint_bincode_options()
        .serialize_into(&mut *file, checkpoint)
        .map_err(|error| encode_checkpoint_error(*error))?;
    file.sync_all()?;
    temp.persist(path)
        .map_err(|error| TrainingError::Io(error.error))?;
    sync_parent_directory(parent)?;
    Ok(())
}

fn create_unique_checkpoint_temp(path: &Path, parent: &Path) -> std::io::Result<NamedTempFile> {
    let target_name = path.file_name().unwrap_or_else(|| OsStr::new("checkpoint"));
    loop {
        let sequence = CHECKPOINT_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let mut temp_name = target_name.to_os_string();
        temp_name.push(format!(".{}.{}.tmp", std::process::id(), sequence));
        let temp_path = parent.join(temp_name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
        {
            Ok(file) => {
                return Ok(NamedTempFile::from_parts(
                    file,
                    TempPath::from_path(temp_path),
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
}

#[cfg(unix)]
fn sync_parent_directory(parent: &Path) -> std::io::Result<()> {
    File::open(parent)?.sync_all()
}

#[cfg(not(unix))]
fn sync_parent_directory(_parent: &Path) -> std::io::Result<()> {
    Ok(())
}

fn checkpoint_bincode_options() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(MAX_TRAINING_CHECKPOINT_BYTES - TRAINING_CHECKPOINT_ENVELOPE_BYTES)
        .reject_trailing_bytes()
}

fn encode_checkpoint_error(error: bincode::ErrorKind) -> TrainingError {
    match error {
        bincode::ErrorKind::Io(error) => TrainingError::Io(error),
        error => TrainingError::TrainingFailed(format!("encode checkpoint: {error}")),
    }
}

fn decode_checkpoint_error(error: bincode::ErrorKind) -> TrainingError {
    match error {
        bincode::ErrorKind::Io(error) if error.kind() != std::io::ErrorKind::UnexpectedEof => {
            TrainingError::Io(error)
        }
        error => {
            let message = error.to_string();
            if message.contains("bytes remaining") || message.contains("trailing") {
                TrainingError::InvalidInput(
                    "decode checkpoint: trailing bytes are not allowed".to_string(),
                )
            } else {
                TrainingError::InvalidInput(format!("decode checkpoint: {message}"))
            }
        }
    }
}

#[cfg(test)]
mod fingerprint_tests {
    use super::{continuity_fingerprint_bytes, hash_training_config};
    use crate::TrainingConfig;

    /// Continuity hash of `TrainingConfig::default()` under the historical
    /// (701d051) algorithm: struct-order JSON with `iterations=0` and no profiler.
    /// Generated by `continuity_fingerprint_bytes` (isomorphic to pre-profiler
    /// `TrainingConfig` serialization). Do **not** regenerate via `serde_json::Value`.
    const HISTORICAL_DEFAULT_CONTINUITY_HASH: &str =
        "5fc39cedfcf31e652d71b8a43b63118d8b9709eb320248cf4fe72269341a0135";

    #[test]
    fn continuity_fingerprint_matches_historical_struct_order_fixture() {
        let baseline = TrainingConfig::default();
        let bytes = continuity_fingerprint_bytes(&baseline).expect("bytes");
        let hash = hash_training_config(&baseline).expect("baseline hash");
        assert_eq!(
            hash, HISTORICAL_DEFAULT_CONTINUITY_HASH,
            "default continuity hash must match frozen historical fixture"
        );
        let text = String::from_utf8(bytes.clone()).expect("utf8");
        assert!(
            text.starts_with(r#"{"backend":"#),
            "historical bytes must use struct field order, got prefix {}",
            &text[..text.len().min(40)]
        );

        // Value+remove(profiler) is NOT a valid historical oracle — it must differ.
        let mut value_order = serde_json::to_value(&baseline).expect("value");
        let object = value_order.as_object_mut().expect("object");
        object.insert("iterations".to_string(), serde_json::json!(0));
        object.remove("profiler");
        let value_bytes = serde_json::to_vec(&value_order).expect("value bytes");
        let value_hash = blake3::hash(&value_bytes).to_hex().to_string();
        assert_ne!(
            value_hash, hash,
            "Value-key-order serialization must not be treated as the historical fixture"
        );
        assert_ne!(value_bytes, bytes);

        let mut toggled = baseline.clone();
        toggled.profiler.enabled = false;
        toggled.profiler.gpu_timing_enabled = true;
        toggled.profiler.gpu_sample_every = 99;
        assert_eq!(
            hash_training_config(&toggled).expect("profiler toggle"),
            HISTORICAL_DEFAULT_CONTINUITY_HASH,
            "profiler-only changes must remain restorable"
        );

        let mut changed_loss = baseline.clone();
        changed_loss.loss.loss_l1_weight *= 2.0;
        assert_ne!(
            hash_training_config(&changed_loss).expect("loss hash"),
            HISTORICAL_DEFAULT_CONTINUITY_HASH,
            "real training parameter changes must still mismatch"
        );

        let mut changed_opt = baseline.clone();
        changed_opt.optimizer.lr_position *= 2.0;
        assert_ne!(
            hash_training_config(&changed_opt).expect("optimizer hash"),
            HISTORICAL_DEFAULT_CONTINUITY_HASH,
            "optimizer changes must still mismatch"
        );
    }
}

#[cfg(test)]
mod identity_migration_tests {
    use super::{
        assign_pre_c5_enumerated_ids, hash_training_config, hash_training_dataset,
        hash_training_dataset_with_pre_c5_frame_ids, match_checkpoint_dataset_identity,
        try_hash_training_dataset_pre_c5_gap_free, DatasetIdentityMatch, TrainingIdentity,
    };
    use crate::TrainingConfig;
    use rustscan_types::{Intrinsics, ScenePose, TrainingDataset, SE3};
    use std::io::Write;
    use tempfile::tempdir;

    fn dataset_with_stable_ids(ids: &[u64]) -> (tempfile::TempDir, TrainingDataset) {
        let dir = tempdir().unwrap();
        let mut dataset = TrainingDataset::new(Intrinsics::new(1.0, 1.0, 0.0, 0.0, 8, 8));
        for &id in ids {
            let path = dir.path().join(format!("frame_{id}.png"));
            let mut file = std::fs::File::create(&path).unwrap();
            write!(file, "image-{id}").unwrap();
            dataset.add_pose(ScenePose::new(id, path, SE3::identity(), 0.0));
        }
        (dir, dataset)
    }

    #[test]
    fn pre_c5_gap_free_identity_is_accepted() {
        let (_dir, dataset) = dataset_with_stable_ids(&[11, 22, 33]);
        let config = TrainingConfig::default();
        let current = hash_training_dataset(&dataset).unwrap();
        let legacy = try_hash_training_dataset_pre_c5_gap_free(&dataset)
            .unwrap()
            .expect("gap-free legacy hash");
        assert_ne!(
            current, legacy,
            "stable-id and pre-C5 enumerated hashes must differ"
        );

        let identity = TrainingIdentity {
            dataset: legacy.clone(),
            reconstruction: "recon".into(),
            config: hash_training_config(&config).unwrap(),
        };
        identity
            .validate_dataset_and_config(&dataset, &config)
            .expect("pre-C5 gap-free identity must be accepted");
        assert_eq!(
            match_checkpoint_dataset_identity(&legacy, &dataset).unwrap(),
            DatasetIdentityMatch::PreC5GapFreeLegacy
        );

        let wrong = TrainingIdentity {
            dataset: "deadbeef".into(),
            reconstruction: "recon".into(),
            config: identity.config.clone(),
        };
        let err = wrong
            .validate_dataset_and_config(&dataset, &config)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("C5")
                || err.contains("stable-image_id")
                || err.contains("pre-C5")
                || err.contains("Gapped"),
            "rejection must explain the frame-identity change: {err}"
        );
    }

    #[test]
    fn missing_image_gaps_are_not_reconstructed_by_reenumerate() {
        // Frozen old loader: candidates [(11,true),(22,false),(33,true)] → IDs [0,2].
        let assigned = assign_pre_c5_enumerated_ids(&[(11, true), (22, false), (33, true)]);
        assert_eq!(assigned, vec![(0, 11), (2, 33)]);

        let (_dir, dataset) = dataset_with_stable_ids(&[11, 33]);
        let old_frame_ids: Vec<u64> = assigned.iter().map(|(id, _)| *id).collect();
        let expected_legacy =
            hash_training_dataset_with_pre_c5_frame_ids(&dataset, &old_frame_ids).unwrap();
        let gap_free = try_hash_training_dataset_pre_c5_gap_free(&dataset)
            .unwrap()
            .expect("gap-free helper still produces a digest");
        assert_ne!(
            expected_legacy, gap_free,
            "re-enumerate [0,1] must not equal frozen gapped [0,2]"
        );

        let err = match_checkpoint_dataset_identity(&expected_legacy, &dataset)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("Gapped") || err.contains("reconstructed") || err.contains("re-enumerat"),
            "resume must refuse unreliable gapped migration: {err}"
        );
    }

    #[test]
    fn oversampled_duplicates_refuse_pre_c5_migration() {
        let dir = tempdir().unwrap();
        let mut dataset = TrainingDataset::new(Intrinsics::new(1.0, 1.0, 0.0, 0.0, 8, 8));
        for &id in &[7u64, 7u64] {
            let path = dir
                .path()
                .join(format!("frame_{id}_{}.png", dataset.poses.len()));
            let mut file = std::fs::File::create(&path).unwrap();
            write!(file, "image-{id}").unwrap();
            dataset.add_pose(ScenePose::new(id, path, SE3::identity(), 0.0));
        }
        // Old oversample kept the same enumerated frame_id on duplicates; gap-free
        // re-enumerate cannot prove that layout from the final list alone.
        let forged = hash_training_dataset_with_pre_c5_frame_ids(&dataset, &[0, 0]).unwrap();
        let err = match_checkpoint_dataset_identity(&forged, &dataset)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("oversampled") || err.contains("duplicate"),
            "oversample must refuse silent migration: {err}"
        );
    }
}
