//! Spann3R inference pipeline
//!
//! Implements the Spann3R forward pass using ONNX models:
//! 1. Encode each frame with CroCo ViT encoder
//! 2. Maintain an explicit spatial memory bank
//! 3. Decode pointmaps using the DUSt3R-style decoder
//! 4. Extract camera pose from pointmap via Procrustes alignment

use crate::inference::procrustes;
use crate::models::{InferenceConfig, InferenceError, PointmapResult};
use glam::Mat4;
use image::DynamicImage;

#[cfg(feature = "onnx-ort")]
use log::info;

#[cfg(feature = "onnx-ort")]
use ort::{
    session::{builder::SessionBuilder, Session},
    value::Tensor,
};

/// Spann3R inference engine
///
/// Processes video frames sequentially, maintaining a spatial memory bank
/// for globally-consistent pose and depth estimation.
pub struct Spann3RInference {
    config: InferenceConfig,
    memory_bank: Vec<MemoryEntry>,
    frame_count: usize,

    #[cfg(feature = "onnx-ort")]
    encoder_session: Session,

    #[cfg(feature = "onnx-ort")]
    #[allow(dead_code)]
    decoder_session: Session,
}

struct MemoryEntry {
    /// Frame index
    #[allow(dead_code)]
    frame_idx: usize,
    /// Encoded feature tokens
    features: Vec<f32>,
    /// Global feature (avg-pooled)
    #[allow(dead_code)]
    global_feature: Vec<f32>,
}

impl Spann3RInference {
    /// Create inference engine from ONNX model files
    #[cfg(feature = "onnx-ort")]
    pub fn from_onnx(encoder_path: &str, decoder_path: &str) -> Result<Self, InferenceError> {
        Self::from_onnx_with_config(encoder_path, decoder_path, InferenceConfig::default())
    }

    /// Create inference engine with custom config
    #[cfg(feature = "onnx-ort")]
    pub fn from_onnx_with_config(
        encoder_path: &str,
        decoder_path: &str,
        config: InferenceConfig,
    ) -> Result<Self, InferenceError> {
        info!("Loading Spann3R encoder from: {}", encoder_path);
        let encoder_session = SessionBuilder::new()?.commit_from_file(encoder_path)?;

        info!("Loading Spann3R decoder from: {}", decoder_path);
        let decoder_session = SessionBuilder::new()?.commit_from_file(decoder_path)?;

        Ok(Self {
            config,
            memory_bank: Vec::new(),
            frame_count: 0,
            encoder_session,
            decoder_session,
        })
    }

    /// Process a single frame
    ///
    /// Returns the estimated camera pose and dense pointmap.
    pub fn process_frame(
        &mut self,
        image: &DynamicImage,
    ) -> Result<PointmapResult, InferenceError> {
        let img_size = self.config.image_size;
        let preprocessed = self.preprocess_image(image, img_size)?;

        // 1. Encode current frame
        let features = self.encode_frame(&preprocessed)?;

        // 2. Update memory bank
        self.update_memory(features.clone());

        // 3. Decode pointmap with memory context
        let (pointmap, confidence) = self.decode_pointmap(&features)?;

        // 4. Extract pose from pointmap
        let pose = self.extract_pose(&pointmap);

        self.frame_count += 1;

        Ok(PointmapResult {
            pose,
            points: pointmap,
            confidence,
            width: img_size,
            height: img_size,
        })
    }

    /// Preprocess image: resize + normalize
    fn preprocess_image(
        &self,
        image: &DynamicImage,
        size: u32,
    ) -> Result<Vec<f32>, InferenceError> {
        let resized = image.resize_exact(size, size, image::imageops::FilterType::Lanczos3);
        let rgb = resized.to_rgb8();
        let (w, h) = rgb.dimensions();

        // ImageNet normalization (same as CroCo/DUSt3R)
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];

        let mut tensor = Vec::with_capacity(3 * w as usize * h as usize);

        // CHW format, normalized
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let pixel = rgb.get_pixel(x, y);
                    let val = pixel[c] as f32 / 255.0;
                    tensor.push((val - mean[c]) / std[c]);
                }
            }
        }

        Ok(tensor)
    }

    /// Encode frame features using the encoder ONNX model
    #[cfg(feature = "onnx-ort")]
    fn encode_frame(&mut self, preprocessed: &[f32]) -> Result<Vec<f32>, InferenceError> {
        let img_size = self.config.image_size as i64;

        // Create input tensor [1, 3, H, W]
        let input = Tensor::<f32>::from_array((
            [1usize, 3, img_size as usize, img_size as usize],
            preprocessed.to_vec().into_boxed_slice(),
        ))?;

        let outputs = self.encoder_session.run(ort::inputs![input])?;
        let (_, output) = outputs[0].try_extract_tensor::<f32>()?;

        Ok(output.to_vec())
    }

    /// Encode frame - stub for non-ORT backends
    #[cfg(not(feature = "onnx-ort"))]
    fn encode_frame(&mut self, _preprocessed: &[f32]) -> Result<Vec<f32>, InferenceError> {
        Err(InferenceError::Inference(
            "No ONNX backend enabled. Enable 'onnx-ort' feature.".to_string(),
        ))
    }

    /// Add frame features to memory bank
    fn update_memory(&mut self, features: Vec<f32>) {
        let global = if !features.is_empty() {
            let dim = features.len();
            vec![features.iter().sum::<f32>() / dim as f32; 1]
        } else {
            vec![]
        };

        self.memory_bank.push(MemoryEntry {
            frame_idx: self.frame_count,
            features,
            global_feature: global,
        });

        // Evict old frames if exceeding limit
        if self.memory_bank.len() > self.config.max_memory_frames {
            self.memory_bank.remove(0);
        }
    }

    /// Decode pointmap using decoder ONNX model
    #[cfg(feature = "onnx-ort")]
    fn decode_pointmap(
        &self,
        _current_features: &[f32],
    ) -> Result<(Vec<[f32; 3]>, Vec<f32>), InferenceError> {
        Err(InferenceError::Inference(
            "Spann3R decoder ONNX execution is not implemented yet".to_string(),
        ))
    }

    #[cfg(not(feature = "onnx-ort"))]
    fn decode_pointmap(
        &self,
        _current_features: &[f32],
    ) -> Result<(Vec<[f32; 3]>, Vec<f32>), InferenceError> {
        Err(InferenceError::Inference(
            "No ONNX backend enabled. Enable 'onnx-ort' feature.".to_string(),
        ))
    }

    /// Extract camera pose from pointmap via weighted Procrustes alignment
    fn extract_pose(&self, pointmap: &[[f32; 3]]) -> Mat4 {
        if pointmap.is_empty() {
            return Mat4::IDENTITY;
        }

        // Generate confidence from pointmap (non-zero points are valid)
        let confidence: Vec<f32> = pointmap
            .iter()
            .map(|p| {
                if p[0] == 0.0 && p[1] == 0.0 && p[2] == 0.0 {
                    0.0
                } else {
                    1.0 / (p[2].abs() + 1.0) // closer points get higher confidence
                }
            })
            .collect();

        procrustes::pointmap_to_pose(pointmap, &confidence)
    }

    /// Get current memory bank size
    pub fn memory_size(&self) -> usize {
        self.memory_bank.len()
    }

    /// Get total processed frame count
    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    /// Reset memory bank (e.g. for new sequence)
    pub fn reset(&mut self) {
        self.memory_bank.clear();
        self.frame_count = 0;
    }
}

#[cfg(all(test, feature = "onnx-ort"))]
mod ort_api_tests {
    use super::*;

    #[test]
    fn ort_rc12_encoder_constructor_api_is_linked() {
        // This is intentionally compile-only: model fixtures and ONNX Runtime
        // initialization are external dependencies, but this catches an API
        // regression in the feature-gated constructor without loading a dylib.
        let _constructor: fn(
            &str,
            &str,
            InferenceConfig,
        ) -> Result<Spann3RInference, InferenceError> = Spann3RInference::from_onnx_with_config;
        let _session_builder = Session::builder;
    }
}
