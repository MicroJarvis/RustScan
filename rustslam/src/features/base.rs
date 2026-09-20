//! Base trait for feature extractors

use thiserror::Error;

/// ORB descriptor size in bytes.
pub const ORB_DESCRIPTOR_SIZE: usize = 32;

#[derive(Error, Debug)]
pub enum FeatureError {
    #[error("OpenCV error: {0}")]
    OpenCV(String),

    #[error("Image error: {0}")]
    Image(String),

    #[error("No features found")]
    NoFeatures,
}

/// A 2D keypoint
#[derive(Debug, Clone)]
pub struct KeyPoint {
    /// Pixel coordinates
    pub pt: (f32, f32),
    /// Size of the keypoint
    pub size: f32,
    /// Angle in radians
    pub angle: f32,
    /// Response (strength)
    pub response: f32,
    /// Octave (scale level)
    pub octave: i32,
}

impl KeyPoint {
    /// Create a new keypoint
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            pt: (x, y),
            size: 1.0,
            angle: -1.0,
            response: 0.0,
            octave: 0,
        }
    }

    /// Get x coordinate
    pub fn x(&self) -> f32 {
        self.pt.0
    }

    /// Get y coordinate
    pub fn y(&self) -> f32 {
        self.pt.1
    }
}

/// Feature descriptors
#[derive(Debug, Clone)]
pub struct Descriptors {
    /// Raw descriptor data
    pub data: Vec<u8>,
    /// Descriptor size in bytes (e.g., 32 for ORB)
    pub size: usize,
    /// Number of descriptors
    pub count: usize,
}

impl Descriptors {
    /// Create empty descriptors
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            size: 0,
            count: 0,
        }
    }

    /// Create with capacity
    pub fn with_capacity(count: usize, size: usize) -> Self {
        Self {
            data: vec![0; count * size],
            size,
            count,
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get descriptor at index
    pub fn get(&self, idx: usize) -> Option<&[u8]> {
        if idx < self.count {
            let start = idx * self.size;
            let end = start + self.size;
            Some(&self.data[start..end])
        } else {
            None
        }
    }
}

impl Default for Descriptors {
    fn default() -> Self {
        Self::new()
    }
}

/// A feature match
#[derive(Debug, Clone)]
pub struct Match {
    /// Query descriptor index
    pub query_idx: u32,
    /// Train descriptor index
    pub train_idx: u32,
    /// Distance
    pub distance: f32,
}

/// Trait for feature extractors
pub trait FeatureExtractor {
    /// Detect keypoints and compute descriptors
    fn detect_and_compute(
        &mut self,
        image: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(Vec<KeyPoint>, Descriptors), FeatureError>;

    /// Detect keypoints only
    fn detect(
        &mut self,
        image: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<KeyPoint>, FeatureError>;

    /// Get number of features to extract
    fn num_features(&self) -> usize;

    /// Set number of features
    fn set_num_features(&mut self, num: usize);
}

/// Trait for feature matchers
pub trait FeatureMatcher {
    /// Match two sets of descriptors
    fn match_descriptors(
        &self,
        query: &Descriptors,
        train: &Descriptors,
    ) -> Result<Vec<Match>, FeatureError>;
}
