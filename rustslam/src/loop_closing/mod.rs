//! Loop closing module
//!
//! Provides loop detection using Bag of Words vocabulary.

pub mod closing;
pub mod database;
pub mod detector;
pub mod optimized_detector;
pub mod relocalization;
pub mod vocabulary;

pub use closing::LoopClosing;
pub use database::{KeyFrameDatabase, KeyFrameEntry};
pub use detector::{LoopCandidate, LoopDetectionResult, LoopDetector};
pub use optimized_detector::{
    DescriptorDistance, GeometricVerifier, InvertedIndex, OptimizedLoopDetectorConfig,
};
pub use relocalization::{RelocalizationResult, Relocalizer};
pub use vocabulary::{hamming_distance, kmeans, Vocabulary, Word};
