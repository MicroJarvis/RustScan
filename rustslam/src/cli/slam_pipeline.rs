//! SLAM processing pipeline helpers for the RustScan CLI.
#![allow(dead_code)]

use crate::config::SlamConfig;
use crate::core::{Frame, FrameFeatures};
use crate::features::base::{FeatureError, ORB_DESCRIPTOR_SIZE};
use crate::features::{
    Descriptors, FastDetector, FastParams, FeatureExtractor, FeatureMatcher, HammingMatcher,
    HarrisDetector, HarrisParams, Match, OrbExtractor,
};

/// 256 pre-computed BRIEF point pairs `(x1, y1, x2, y2)` within a 31×31 patch.
const NUM_BRIEF_TESTS: usize = 256;
const BRIEF_PAIRS: [(i8, i8, i8, i8); NUM_BRIEF_TESTS] = generate_brief_pairs();

const fn generate_brief_pairs() -> [(i8, i8, i8, i8); NUM_BRIEF_TESTS] {
    let mut pairs = [(0i8, 0i8, 0i8, 0i8); NUM_BRIEF_TESTS];
    let mut state: u32 = 0x12345678;
    let mut i = 0;
    while i < NUM_BRIEF_TESTS {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let x1 = ((state >> 16) % 31) as i8 - 15;
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let y1 = ((state >> 16) % 31) as i8 - 15;
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let x2 = ((state >> 16) % 31) as i8 - 15;
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let y2 = ((state >> 16) % 31) as i8 - 15;
        pairs[i] = (x1, y1, x2, y2);
        i += 1;
    }
    pairs
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureType {
    Orb,
    Harris,
    Fast,
}

impl FeatureType {
    pub fn from_env() -> Option<Self> {
        let raw = std::env::var("RUSTSCAN_FEATURE_TYPE").ok()?;
        match raw.to_ascii_lowercase().as_str() {
            "orb" => Some(Self::Orb),
            "harris" => Some(Self::Harris),
            "fast" => Some(Self::Fast),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SlamPipelineConfig {
    pub feature_type: FeatureType,
    pub max_features: usize,
}

impl SlamPipelineConfig {
    pub fn from_slam_config(config: &SlamConfig) -> Self {
        let feature_type = FeatureType::from_env().unwrap_or(FeatureType::Orb);
        Self {
            feature_type,
            max_features: config.tracker.max_features,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FeatureFrame {
    pub frame: Frame,
    pub keypoints: Vec<[f32; 2]>,
    pub descriptors: Descriptors,
}

impl FeatureFrame {
    pub fn to_frame_features(&self) -> FrameFeatures {
        FrameFeatures {
            keypoints: self.keypoints.clone(),
            descriptors: self.descriptors.data.clone(),
            map_points: vec![None; self.keypoints.len()],
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SlamPipelineError {
    #[error("feature extraction failed: {0}")]
    FeatureExtraction(#[from] FeatureError),
    #[error("feature matching failed: {0}")]
    FeatureMatching(FeatureError),
}

pub struct FeatureExtractorEngine {
    feature_type: FeatureType,
    orb: OrbExtractor,
    harris: HarrisDetector,
    fast: FastDetector,
    max_features: usize,
}

impl FeatureExtractorEngine {
    pub fn new(config: &SlamPipelineConfig) -> Self {
        Self {
            feature_type: config.feature_type,
            orb: OrbExtractor::new(config.max_features),
            harris: HarrisDetector::new(HarrisParams::default()),
            fast: FastDetector::new(FastParams::default()),
            max_features: config.max_features,
        }
    }

    pub fn extract(
        &mut self,
        frame: Frame,
        rgb: &[u8],
        width: u32,
        height: u32,
    ) -> Result<FeatureFrame, SlamPipelineError> {
        let gray = rgb_to_grayscale(rgb, width, height);
        match self.feature_type {
            FeatureType::Orb => {
                let (keypoints, descriptors) = self.orb.detect_and_compute(&gray, width, height)?;
                Ok(FeatureFrame {
                    frame,
                    keypoints: keypoints.iter().map(|kp| [kp.x(), kp.y()]).collect(),
                    descriptors,
                })
            }
            FeatureType::Harris => {
                let mut keypoints = self.harris.detect(&gray, width, height);
                keypoints.truncate(self.max_features);
                Ok(FeatureFrame {
                    frame,
                    keypoints: keypoints.iter().map(|kp| [kp.x, kp.y]).collect(),
                    descriptors: build_patch_descriptors(&gray, width, height, &keypoints),
                })
            }
            FeatureType::Fast => {
                let mut keypoints = self.fast.detect(&gray, width, height);
                keypoints.truncate(self.max_features);
                Ok(FeatureFrame {
                    frame,
                    keypoints: keypoints.iter().map(|kp| [kp.x, kp.y]).collect(),
                    descriptors: build_patch_descriptors(&gray, width, height, &keypoints),
                })
            }
        }
    }
}

pub struct FeatureMatcherEngine {
    matcher: HammingMatcher,
}

impl FeatureMatcherEngine {
    pub fn new(ratio_threshold: f64) -> Self {
        Self {
            matcher: HammingMatcher::new(2).with_ratio_threshold(ratio_threshold),
        }
    }

    pub fn match_frames(
        &mut self,
        current: &FeatureFrame,
        previous: &FeatureFrame,
    ) -> Result<Vec<Match>, SlamPipelineError> {
        if current.descriptors.is_empty() || previous.descriptors.is_empty() {
            return Ok(Vec::new());
        }

        let mut matches = self
            .matcher
            .match_descriptors(&current.descriptors, &previous.descriptors)
            .map_err(|e| SlamPipelineError::FeatureMatching(e))?;

        matches.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(matches)
    }
}

fn rgb_to_grayscale(rgb: &[u8], width: u32, height: u32) -> Vec<u8> {
    let expected = (width as usize) * (height as usize) * 3;
    if rgb.len() < expected {
        return Vec::new();
    }
    let mut gray = Vec::with_capacity((width as usize) * (height as usize));
    for chunk in rgb.chunks_exact(3) {
        let r = chunk[0] as f32;
        let g = chunk[1] as f32;
        let b = chunk[2] as f32;
        let luma = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
        gray.push(luma);
    }
    gray
}

fn build_patch_descriptors(
    gray: &[u8],
    width: u32,
    height: u32,
    keypoints: &[crate::features::pure_rust::KeyPoint],
) -> Descriptors {
    let mut descriptors = Descriptors::with_capacity(keypoints.len(), ORB_DESCRIPTOR_SIZE);
    if keypoints.is_empty() || gray.is_empty() {
        return descriptors;
    }
    let w = width as i32;
    let h = height as i32;
    for (kp_idx, kp) in keypoints.iter().enumerate() {
        let cx = kp.x.round() as i32;
        let cy = kp.y.round() as i32;
        let use_rot = kp.angle.is_finite() && kp.angle >= 0.0;
        let (sin_a, cos_a) = if use_rot {
            kp.angle.sin_cos()
        } else {
            (0.0f32, 1.0f32)
        };
        let base = kp_idx * ORB_DESCRIPTOR_SIZE;
        for (i, &(x1, y1, x2, y2)) in BRIEF_PAIRS.iter().enumerate() {
            let (rx1, ry1) = if use_rot {
                (
                    (x1 as f32 * cos_a - y1 as f32 * sin_a).round() as i32,
                    (x1 as f32 * sin_a + y1 as f32 * cos_a).round() as i32,
                )
            } else {
                (x1 as i32, y1 as i32)
            };
            let (rx2, ry2) = if use_rot {
                (
                    (x2 as f32 * cos_a - y2 as f32 * sin_a).round() as i32,
                    (x2 as f32 * sin_a + y2 as f32 * cos_a).round() as i32,
                )
            } else {
                (x2 as i32, y2 as i32)
            };
            let i1 = if cx + rx1 >= 0 && cx + rx1 < w && cy + ry1 >= 0 && cy + ry1 < h {
                gray[((cy + ry1) * w + (cx + rx1)) as usize]
            } else {
                0
            };
            let i2 = if cx + rx2 >= 0 && cx + rx2 < w && cy + ry2 >= 0 && cy + ry2 < h {
                gray[((cy + ry2) * w + (cx + rx2)) as usize]
            } else {
                0
            };
            if i1 > i2 {
                descriptors.data[base + i / 8] |= 1 << (i % 8);
            }
        }
    }
    descriptors
}
