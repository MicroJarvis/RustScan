//! Depth Fusion
//!
//! This module provides depth fusion capabilities for combining multiple
//! depth sources (e.g., from stereo matching, RGB-D, depth from motion).

use std::collections::HashMap;

/// Configuration for depth fusion
#[derive(Debug, Clone)]
pub struct DepthFusionConfig {
    /// Maximum number of observations per pixel
    pub max_observations: usize,
    /// Confidence threshold for depth
    pub confidence_threshold: f32,
    /// Depth consistency check tolerance
    pub consistency_tolerance: f32,
}

impl Default for DepthFusionConfig {
    fn default() -> Self {
        Self {
            max_observations: 10,
            confidence_threshold: 0.5,
            consistency_tolerance: 0.1,
        }
    }
}

/// Depth observation from a single source
#[derive(Debug, Clone)]
pub struct DepthObservation {
    /// Depth value in meters
    pub depth: f32,
    /// Confidence score [0, 1]
    pub confidence: f32,
    /// Source identifier
    pub source_id: usize,
}

/// Depth fusion for combining multiple depth sources
pub struct DepthFusion {
    config: DepthFusionConfig,
    observations: HashMap<usize, Vec<DepthObservation>>, // pixel_index -> observations
}

impl DepthFusion {
    /// Create a new depth fusion instance
    pub fn new(config: DepthFusionConfig) -> Self {
        Self {
            config,
            observations: HashMap::new(),
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(DepthFusionConfig::default())
    }

    /// Add a depth observation for a pixel
    pub fn add_observation(&mut self, pixel_index: usize, observation: DepthObservation) {
        let observations = self
            .observations
            .entry(pixel_index)
            .or_insert_with(Vec::new);

        if observations.len() < self.config.max_observations {
            observations.push(observation);
        }
    }

    /// Add depth map from a source
    pub fn add_depth_map(
        &mut self,
        source_id: usize,
        depth: &[f32],
        confidence: Option<&[f32]>,
        width: usize,
        height: usize,
    ) {
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let d = depth[idx];

                if d > 0.0 {
                    let conf = confidence.map(|c| c[idx]).unwrap_or(1.0);

                    if conf >= self.config.confidence_threshold {
                        self.add_observation(
                            idx,
                            DepthObservation {
                                depth: d,
                                confidence: conf,
                                source_id,
                            },
                        );
                    }
                }
            }
        }
    }

    /// Fuse observations to get best depth estimate
    pub fn fuse(&self, pixel_index: usize) -> Option<f32> {
        let observations = self.observations.get(&pixel_index)?;

        if observations.is_empty() {
            return None;
        }

        // Simple fusion: weighted average by confidence
        let mut sum_weighted_depth = 0.0;
        let mut sum_weights = 0.0;

        for obs in observations {
            let weight = obs.confidence;
            sum_weighted_depth += obs.depth * weight;
            sum_weights += weight;
        }

        if sum_weights > 0.0 {
            Some(sum_weighted_depth / sum_weights)
        } else {
            None
        }
    }

    /// Fuse all pixels and return depth map
    pub fn fuse_all(&self, width: usize, height: usize) -> Vec<f32> {
        (0..width * height)
            .map(|idx| self.fuse(idx).unwrap_or(0.0))
            .collect()
    }

    /// Get number of observations for a pixel
    pub fn num_observations(&self, pixel_index: usize) -> usize {
        self.observations
            .get(&pixel_index)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Clear all observations
    pub fn clear(&mut self) {
        self.observations.clear();
    }

    /// Get total number of pixels with observations
    pub fn num_pixels(&self) -> usize {
        self.observations.len()
    }
}

/// Temporal depth fusion (for depth from motion)
pub struct TemporalDepthFusion {
    _config: DepthFusionConfig,
    prev_depth: Option<Vec<f32>>,
    alpha: f32, // Blending factor
}

impl TemporalDepthFusion {
    /// Create a new temporal depth fusion
    pub fn new(config: DepthFusionConfig, alpha: f32) -> Self {
        Self {
            _config: config,
            prev_depth: None,
            alpha: alpha.clamp(0.0, 1.0),
        }
    }

    /// Update with new depth frame
    pub fn update(&mut self, new_depth: &[f32]) -> Vec<f32> {
        let fused = match &self.prev_depth {
            Some(prev) => {
                // Blend current and previous depth
                new_depth
                    .iter()
                    .zip(prev.iter())
                    .map(|(&curr, &prev)| {
                        if curr > 0.0 && prev > 0.0 {
                            self.alpha * curr + (1.0 - self.alpha) * prev
                        } else if curr > 0.0 {
                            curr
                        } else if prev > 0.0 {
                            prev
                        } else {
                            0.0
                        }
                    })
                    .collect()
            }
            None => new_depth.to_vec(),
        };

        self.prev_depth = Some(fused.clone());
        fused
    }

    /// Reset the fusion state
    pub fn reset(&mut self) {
        self.prev_depth = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depth_fusion_creation() {
        let fusion = DepthFusion::default();
        assert_eq!(fusion.num_pixels(), 0);
    }

    #[test]
    fn test_add_observation() {
        let mut fusion = DepthFusion::default();

        fusion.add_observation(
            0,
            DepthObservation {
                depth: 1.0,
                confidence: 0.9,
                source_id: 0,
            },
        );

        assert_eq!(fusion.num_observations(0), 1);
    }

    #[test]
    fn test_fuse() {
        let mut fusion = DepthFusion::default();

        fusion.add_observation(
            0,
            DepthObservation {
                depth: 1.0,
                confidence: 0.5,
                source_id: 0,
            },
        );
        fusion.add_observation(
            0,
            DepthObservation {
                depth: 2.0,
                confidence: 0.5,
                source_id: 1,
            },
        );

        let fused = fusion.fuse(0);
        assert!(fused.is_some());
        // Weighted average: (1.0 * 0.5 + 2.0 * 0.5) / (0.5 + 0.5) = 1.5
        assert!((fused.unwrap() - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_fuse_all() {
        let mut fusion = DepthFusion::default();

        fusion.add_observation(
            0,
            DepthObservation {
                depth: 1.0,
                confidence: 1.0,
                source_id: 0,
            },
        );
        fusion.add_observation(
            1,
            DepthObservation {
                depth: 2.0,
                confidence: 1.0,
                source_id: 0,
            },
        );

        let result = fusion.fuse_all(2, 1);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);
    }

    #[test]
    fn test_temporal_fusion() {
        let config = DepthFusionConfig::default();
        let mut fusion = TemporalDepthFusion::new(config, 0.5);

        let depth1 = vec![1.0, 2.0, 3.0];
        let result1 = fusion.update(&depth1);
        assert_eq!(result1, depth1);

        let depth2 = vec![2.0, 4.0, 6.0];
        let result2 = fusion.update(&depth2);
        // Should be blended: 0.5 * new + 0.5 * prev
        assert!((result2[0] - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_depth_fusion_config_default() {
        let config = DepthFusionConfig::default();
        assert_eq!(config.max_observations, 10);
        assert_eq!(config.confidence_threshold, 0.5);
    }
}
