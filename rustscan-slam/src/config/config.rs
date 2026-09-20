//! Main configuration structures for RustSLAM

use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

use super::params::*;

/// Configuration loading errors
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Failed to read config file: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Failed to parse YAML: {0}")]
    YamlError(#[from] serde_yaml::Error),
    #[error("Failed to parse TOML: {0}")]
    TomlError(#[from] toml::de::Error),
    #[error("Failed to serialize TOML: {0}")]
    TomlSerializeError(#[from] toml::ser::Error),
    #[error("Unsupported config format: {0}")]
    UnsupportedFormat(String),
    #[error("Invalid configuration:\n{0}")]
    Validation(String),
}

/// Main SLAM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlamConfig {
    /// Camera configuration
    pub camera: CameraConfig,
    /// Tracker parameters
    pub tracker: TrackerParams,
    /// Mapper parameters
    pub mapper: MapperParams,
    /// Optimizer parameters
    pub optimizer: OptimizerParams,
    /// Loop closing parameters
    pub loop_closing: LoopClosingParams,
    /// Dataset parameters
    pub dataset: DatasetParams,
    /// Viewer parameters
    pub viewer: ViewerParams,
}

impl Default for SlamConfig {
    fn default() -> Self {
        Self {
            camera: CameraConfig::default(),
            tracker: TrackerParams::default(),
            mapper: MapperParams::default(),
            optimizer: OptimizerParams::default(),
            loop_closing: LoopClosingParams::default(),
            dataset: DatasetParams::default(),
            viewer: ViewerParams::default(),
        }
    }
}

/// Camera configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraConfig {
    /// Camera width
    pub width: u32,
    /// Camera height
    pub height: u32,
    /// Focal length x
    pub fx: f32,
    /// Focal length y
    pub fy: f32,
    /// Principal point x
    pub cx: f32,
    /// Principal point y
    pub cy: f32,
    /// Camera model
    pub model: String,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            width: 640,
            height: 480,
            fx: 525.0,
            fy: 525.0,
            cx: 319.5,
            cy: 239.5,
            model: "pinhole".to_string(),
        }
    }
}

impl CameraConfig {
    pub fn validate(&self) -> Vec<String> {
        let mut errs = Vec::new();
        if self.width == 0 {
            errs.push("camera.width must be > 0".into());
        }
        if self.height == 0 {
            errs.push("camera.height must be > 0".into());
        }
        if self.fx <= 0.0 {
            errs.push("camera.fx must be > 0".into());
        }
        if self.fy <= 0.0 {
            errs.push("camera.fy must be > 0".into());
        }
        if self.cx <= 0.0 {
            errs.push("camera.cx must be > 0".into());
        }
        if self.cy <= 0.0 {
            errs.push("camera.cy must be > 0".into());
        }
        errs
    }

    /// Convert to core Camera struct
    pub fn to_camera(&self) -> crate::core::Camera {
        crate::core::Camera::new(self.fx, self.fy, self.cx, self.cy, self.width, self.height)
    }
}

/// Configuration loader supporting YAML and TOML
pub struct ConfigLoader;

impl ConfigLoader {
    /// Load configuration from file (with validation)
    pub fn load<P: AsRef<Path>>(path: P) -> Result<SlamConfig, ConfigError> {
        let path = path.as_ref();
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        let config = match extension.to_lowercase().as_str() {
            "yaml" | "yml" => Self::load_yaml_raw(path)?,
            "toml" => Self::load_toml_raw(path)?,
            _ => return Err(ConfigError::UnsupportedFormat(extension.to_string())),
        };
        config.validate()?;
        Ok(config)
    }

    /// Load from YAML without validation (internal)
    fn load_yaml_raw<P: AsRef<Path>>(path: P) -> Result<SlamConfig, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: SlamConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Load from TOML without validation (internal)
    fn load_toml_raw<P: AsRef<Path>>(path: P) -> Result<SlamConfig, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: SlamConfig = toml::from_str(&content)?;
        Ok(config)
    }

    /// Load configuration from YAML file (with validation)
    pub fn load_yaml<P: AsRef<Path>>(path: P) -> Result<SlamConfig, ConfigError> {
        let config = Self::load_yaml_raw(path)?;
        config.validate()?;
        Ok(config)
    }

    /// Load configuration from TOML file (with validation)
    pub fn load_toml<P: AsRef<Path>>(path: P) -> Result<SlamConfig, ConfigError> {
        let config = Self::load_toml_raw(path)?;
        config.validate()?;
        Ok(config)
    }

    /// Save configuration to YAML file
    pub fn save_yaml<P: AsRef<Path>>(config: &SlamConfig, path: P) -> Result<(), ConfigError> {
        let content = serde_yaml::to_string(config)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Save configuration to TOML file
    pub fn save_toml<P: AsRef<Path>>(config: &SlamConfig, path: P) -> Result<(), ConfigError> {
        let content = toml::to_string(config)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

impl SlamConfig {
    /// Validate all configuration parameters.
    /// Returns `Ok(())` if valid, or `Err(ConfigError::Validation)` with all issues listed.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let mut errs = Vec::new();
        errs.extend(self.camera.validate());
        errs.extend(self.tracker.validate());
        errs.extend(self.mapper.validate());
        errs.extend(self.optimizer.validate());
        errs.extend(self.loop_closing.validate());
        errs.extend(self.dataset.validate());
        errs.extend(self.viewer.validate());
        if errs.is_empty() {
            Ok(())
        } else {
            Err(ConfigError::Validation(errs.join("\n")))
        }
    }

    /// Create default configuration for TUM dataset
    pub fn tum_rgbd() -> Self {
        Self {
            camera: CameraConfig {
                width: 640,
                height: 480,
                fx: 525.0,
                fy: 525.0,
                cx: 319.5,
                cy: 239.5,
                model: "pinhole".to_string(),
            },
            tracker: TrackerParams {
                max_features: 2000,
                min_features: 500,
                pyramid_levels: 8,
                patch_size: 31,
                ..Default::default()
            },
            mapper: MapperParams::default(),
            optimizer: OptimizerParams::default(),
            loop_closing: LoopClosingParams::default(),
            dataset: DatasetParams {
                dataset_type: "tum".to_string(),
                ..Default::default()
            },
            viewer: ViewerParams::default(),
        }
    }

    /// Create default configuration for KITTI dataset
    pub fn kitti() -> Self {
        Self {
            camera: CameraConfig {
                width: 1241,
                height: 376,
                fx: 718.856,
                fy: 718.856,
                cx: 607.1928,
                cy: 185.2157,
                model: "pinhole".to_string(),
            },
            tracker: TrackerParams {
                max_features: 2000,
                min_features: 500,
                pyramid_levels: 4,
                patch_size: 31,
                ..Default::default()
            },
            mapper: MapperParams::default(),
            optimizer: OptimizerParams::default(),
            loop_closing: LoopClosingParams::default(),
            dataset: DatasetParams {
                dataset_type: "kitti".to_string(),
                ..Default::default()
            },
            viewer: ViewerParams::default(),
        }
    }

    /// Create default configuration for EuRoC dataset
    pub fn euroc() -> Self {
        Self {
            camera: CameraConfig {
                width: 752,
                height: 480,
                fx: 458.654,
                fy: 457.296,
                cx: 367.215,
                cy: 248.375,
                model: "pinhole".to_string(),
            },
            tracker: TrackerParams {
                max_features: 1200,
                min_features: 400,
                pyramid_levels: 8,
                patch_size: 31,
                ..Default::default()
            },
            mapper: MapperParams::default(),
            optimizer: OptimizerParams::default(),
            loop_closing: LoopClosingParams::default(),
            dataset: DatasetParams {
                dataset_type: "euroc".to_string(),
                ..Default::default()
            },
            viewer: ViewerParams::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_config_default() {
        let config = SlamConfig::default();
        assert_eq!(config.camera.width, 640);
    }

    #[test]
    fn test_config_tum() {
        let config = SlamConfig::tum_rgbd();
        assert_eq!(config.dataset.dataset_type, "tum");
    }

    #[test]
    fn test_config_kitti() {
        let config = SlamConfig::kitti();
        assert_eq!(config.dataset.dataset_type, "kitti");
    }

    #[test]
    fn test_config_euroc() {
        let config = SlamConfig::euroc();
        assert_eq!(config.dataset.dataset_type, "euroc");
    }

    #[test]
    fn test_load_yaml() {
        // Test by loading a full config from default
        let config = SlamConfig::default();
        let yaml = serde_yaml::to_string(&config).unwrap();
        let loaded: SlamConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(loaded.camera.width, 640);
    }

    #[test]
    fn test_save_load_yaml() {
        let config = SlamConfig::tum_rgbd();
        let temp_file = NamedTempFile::new().unwrap();

        ConfigLoader::save_yaml(&config, temp_file.path()).unwrap();
        let loaded = ConfigLoader::load_yaml(temp_file.path()).unwrap();

        assert_eq!(loaded.camera.width, config.camera.width);
    }

    #[test]
    fn test_save_load_toml() {
        let config = SlamConfig::kitti();
        let temp_file = NamedTempFile::new().unwrap();

        ConfigLoader::save_toml(&config, temp_file.path()).unwrap();
        let loaded = ConfigLoader::load_toml(temp_file.path()).unwrap();

        assert_eq!(loaded.camera.fx, config.camera.fx);
    }

    #[test]
    fn test_default_config_validates() {
        SlamConfig::default().validate().unwrap();
    }

    #[test]
    fn test_preset_configs_validate() {
        SlamConfig::tum_rgbd().validate().unwrap();
        SlamConfig::kitti().validate().unwrap();
        SlamConfig::euroc().validate().unwrap();
    }

    #[test]
    fn test_invalid_camera_rejected() {
        let mut config = SlamConfig::default();
        config.camera.fx = 0.0;
        config.camera.height = 0;
        let err = config.validate().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("camera.fx"));
        assert!(msg.contains("camera.height"));
    }

    #[test]
    fn test_invalid_toml_rejected_at_load() {
        let temp_file = NamedTempFile::new().unwrap();
        // Write a config with invalid tracker params
        let mut config = SlamConfig::default();
        config.tracker.max_features = 10;
        config.tracker.min_features = 100; // max < min
        ConfigLoader::save_toml(&config, temp_file.path()).unwrap();

        let result = ConfigLoader::load_toml(temp_file.path());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("max_features"));
    }

    #[test]
    fn test_multiple_errors_collected() {
        let mut config = SlamConfig::default();
        config.camera.fx = -1.0;
        config.tracker.scale_factor = 0.5;
        config.mapper.keyframe_interval = 0;
        let err = config.validate().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("camera.fx"));
        assert!(msg.contains("scale_factor"));
        assert!(msg.contains("keyframe_interval"));
    }
}
