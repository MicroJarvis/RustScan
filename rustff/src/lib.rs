//! RustFF - FeedForward neural network inference for 3D reconstruction
//!
//! Provides pose estimation and dense depth prediction from image sequences
//! using feedforward models like Spann3R, without traditional feature matching
//! or bundle adjustment optimization.
//!
//! ## Architecture
//!
//! ```text
//! Image Sequence
//!     ↓ ONNX Encoder (CroCo ViT)
//!     ↓ Spatial Memory Bank
//!     ↓ ONNX Decoder (Pointmap)
//! Pose [4x4] + Dense Pointmap [H×W×3]
//! ```
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use rustff::Spann3RInference;
//! use rustff::models::InferenceConfig;
//!
//! # fn main() -> anyhow::Result<()> {
//! let config = InferenceConfig::default();
//! // Enable 'onnx-ort' feature for actual inference:
//! // let mut model = Spann3RInference::from_onnx("encoder.onnx", "decoder.onnx")?;
//! # Ok(())
//! # }
//! ```

pub mod inference;
pub mod models;

pub use inference::Spann3RInference;
pub use models::{InferenceConfig, InferenceError, PointmapResult};
