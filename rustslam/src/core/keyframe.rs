//! KeyFrame representation

use crate::core::{Frame, FrameFeatures, SE3};

/// A keyframe is a special frame used for mapping
#[derive(Debug, Clone)]
pub struct KeyFrame {
    /// The underlying frame
    pub frame: Frame,
    /// Features extracted from this keyframe
    pub features: FrameFeatures,
    /// Connected keyframes and their weights
    pub connected_keyframes: Vec<(u64, u32)>, // (keyframe_id, weight)
    /// BoW vector for loop closing
    pub bow_vector: Option<Vec<f32>>,
}

impl KeyFrame {
    /// Create a new keyframe from a frame
    pub fn new(frame: Frame, features: FrameFeatures) -> Self {
        Self {
            frame,
            features,
            connected_keyframes: Vec::new(),
            bow_vector: None,
        }
    }

    /// Get the ID
    pub fn id(&self) -> u64 {
        self.frame.id
    }

    /// Get the pose
    pub fn pose(&self) -> Option<SE3> {
        self.frame.pose
    }

    /// Get number of features
    pub fn num_features(&self) -> usize {
        self.features.len()
    }

    /// Set BoW vector
    pub fn set_bow(&mut self, bow: Vec<f32>) {
        self.bow_vector = Some(bow);
    }
}
