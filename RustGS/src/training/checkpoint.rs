use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{HostSplats, TrainingConfig, TrainingDataset, TrainingError};

pub const TRAINING_CHECKPOINT_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingIdentity {
    pub dataset: String,
    pub reconstruction: String,
    pub config: String,
}

impl TrainingIdentity {
    pub fn from_inputs(
        dataset: &TrainingDataset,
        reconstruction: &str,
        config: &TrainingConfig,
    ) -> Result<Self, TrainingError> {
        let dataset_bytes = serde_json::to_vec(dataset)
            .map_err(|error| TrainingError::InvalidInput(error.to_string()))?;
        let mut resume_compatible_config = config.clone();
        resume_compatible_config.iterations = 0;
        let config_bytes = serde_json::to_vec(&resume_compatible_config)
            .map_err(|error| TrainingError::InvalidInput(error.to_string()))?;
        Ok(Self {
            dataset: blake3::hash(&dataset_bytes).to_hex().to_string(),
            reconstruction: reconstruction.to_owned(),
            config: blake3::hash(&config_bytes).to_hex().to_string(),
        })
    }
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
        self.splats.validate().map_err(|error| {
            invalid_checkpoint(format!("checkpoint splats are invalid: {error}"))
        })?;

        validate_adam_parameter("optimizer.transforms", &self.optimizer.transforms)?;
        validate_adam_parameter("optimizer.sh_coeffs", &self.optimizer.sh_coeffs)?;
        validate_adam_parameter("optimizer.raw_opacities", &self.optimizer.raw_opacities)?;

        let splat_count = self.splats.len();
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
        Ok(())
    }
}

pub fn save_training_checkpoint(
    path: &Path,
    checkpoint: &TrainingCheckpoint,
) -> Result<(), TrainingError> {
    checkpoint.validate()?;
    let bytes = bincode::serialize(checkpoint)
        .map_err(|error| TrainingError::TrainingFailed(format!("encode checkpoint: {error}")))?;
    let parent = checkpoint_parent(path);
    fs::create_dir_all(parent)?;

    let temp = checkpoint_temp_path(path);
    let result = write_and_commit_checkpoint(path, parent, &temp, &bytes);
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result.map_err(TrainingError::Io)
}

pub fn load_training_checkpoint(path: &Path) -> Result<TrainingCheckpoint, TrainingError> {
    let bytes = fs::read(path)?;
    let checkpoint: TrainingCheckpoint = bincode::deserialize(&bytes)
        .map_err(|error| TrainingError::InvalidInput(format!("decode checkpoint: {error}")))?;
    checkpoint.validate()?;
    Ok(checkpoint)
}

fn validate_identity_field(name: &str, value: &str) -> Result<(), TrainingError> {
    if value.trim().is_empty() {
        return Err(invalid_checkpoint(format!(
            "identity {name} must not be empty"
        )));
    }
    Ok(())
}

fn validate_adam_parameter(
    name: &str,
    parameter: &AdamParameterCheckpoint,
) -> Result<(), TrainingError> {
    for (field, tensor) in [
        ("moment1", parameter.moment1.as_ref()),
        ("moment2", parameter.moment2.as_ref()),
        ("scaling", parameter.scaling.as_ref()),
    ] {
        if let Some(tensor) = tensor {
            validate_tensor(&format!("{name}.{field}"), tensor)?;
        }
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
    let expected = tensor
        .shape
        .iter()
        .try_fold(1usize, |product, &dimension| product.checked_mul(dimension));
    let Some(expected) = expected else {
        return Err(invalid_checkpoint(format!(
            "{name} tensor shape overflows usize"
        )));
    };
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

fn checkpoint_temp_path(path: &Path) -> PathBuf {
    path.with_extension("rgscp.tmp")
}

fn write_and_commit_checkpoint(
    path: &Path,
    parent: &Path,
    temp: &Path,
    bytes: &[u8],
) -> std::io::Result<()> {
    let mut file = File::create(temp)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(temp, path)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}
