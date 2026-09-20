//! Configuration module for RustSLAM
//!
//! This module provides configuration management for all SLAM components.

pub mod config;
pub mod params;

pub use config::{CameraConfig, ConfigError, ConfigLoader, SlamConfig};
pub use params::{
    DatasetParams, FeatureType, GaussianSplattingParams, LoopClosingParams, MapperParams,
    OptimizerParams, TrackerParams, TsdfParams, ValidationErrors, ViewerParams,
};
