//! ONNX model inference backends

pub mod procrustes;
pub mod spann3r;

pub use procrustes::{pointmap_to_pose, weighted_procrustes};
pub use spann3r::Spann3RInference;
