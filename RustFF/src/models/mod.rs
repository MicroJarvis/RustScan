//! Shared data structures and error types

use nalgebra::Matrix4;
use thiserror::Error;

/// Inference configuration
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Input image size (square, e.g. 512)
    pub image_size: u32,
    /// Maximum number of frames in memory bank
    pub max_memory_frames: usize,
    /// Keyframe sampling interval (process every N-th frame)
    pub kf_every: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            image_size: 512,
            max_memory_frames: 200,
            kf_every: 1,
        }
    }
}

/// Result of processing a single frame
#[derive(Debug, Clone)]
pub struct PointmapResult {
    /// Camera pose in world coordinates [4x4]
    pub pose: Matrix4<f32>,
    /// Dense 3D pointmap [H x W x 3], in world coordinates
    pub points: Vec<[f32; 3]>,
    /// Per-pixel confidence [H x W]
    pub confidence: Vec<f32>,
    /// Pointmap dimensions (width, height)
    pub width: u32,
    pub height: u32,
}

impl PointmapResult {
    /// Extract depth map from pointmap (Z channel in camera frame)
    pub fn depth_map(&self) -> Vec<f32> {
        self.points.iter().map(|p| p[2]).collect()
    }
}

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("ONNX model loading failed: {0}")]
    ModelLoad(String),

    #[error("Inference failed: {0}")]
    Inference(String),

    #[error("Unsupported ONNX op: {0}")]
    UnsupportedOp(String),

    #[cfg(feature = "onnx-ort")]
    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),

    #[error("Image preprocessing error: {0}")]
    ImagePreprocess(String),
}
