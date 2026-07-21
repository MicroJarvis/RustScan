use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use bincode::Options;
use serde::{Deserialize, Serialize};
use tempfile::{NamedTempFile, TempPath};

use crate::{HostSplats, TrainingConfig, TrainingDataset, TrainingError};

pub const TRAINING_CHECKPOINT_VERSION: u32 = 1;
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingIdentity {
    pub dataset: String,
    pub reconstruction: String,
    pub config: String,
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
    let mut resume_compatible_config = config.clone();
    resume_compatible_config.iterations = 0;
    let bytes = serde_json::to_vec(&resume_compatible_config)
        .map_err(|error| TrainingError::InvalidInput(error.to_string()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
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
        if self.splats.len() > MAX_TRAINING_CHECKPOINT_SPLATS {
            return Err(invalid_checkpoint(format!(
                "splat count exceeds maximum {MAX_TRAINING_CHECKPOINT_SPLATS}"
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

    let checkpoint: TrainingCheckpoint = checkpoint_bincode_options()
        .deserialize_from(&mut reader)
        .map_err(|error| decode_checkpoint_error(*error))?;
    let mut trailing = [0u8; 1];
    if reader.read(&mut trailing)? != 0 {
        return Err(TrainingError::InvalidInput(
            "decode checkpoint: trailing bytes are not allowed".to_string(),
        ));
    }
    checkpoint.validate()?;
    Ok(checkpoint)
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
        error => TrainingError::InvalidInput(format!("decode checkpoint: {error}")),
    }
}
