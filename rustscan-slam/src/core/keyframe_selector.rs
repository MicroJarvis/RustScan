//! Keyframe Selection Strategy
//!
//! This module implements intelligent keyframe selection to reduce redundancy
//! while ensuring tracking quality.

use crate::core::SE3;

/// Configuration for keyframe selection
#[derive(Debug, Clone)]
pub struct KeyframeSelectorConfig {
    /// Translation threshold (meters) - insert keyframe if moved more than this
    pub translation_threshold: f32,
    /// Rotation threshold (degrees) - insert keyframe if rotated more than this
    pub rotation_threshold: f32,
    /// Minimum time interval between keyframes (seconds)
    pub min_time_interval: f64,
    /// Minimum number of tracked features to maintain
    pub min_tracked_features: usize,
    /// Maximum number of keyframes to keep in memory
    pub max_keyframes: usize,
    /// Whether to use covisibility for selection
    pub use_covisibility: bool,
    /// Covisibility threshold (0-1)
    pub covisibility_threshold: f32,
}

impl Default for KeyframeSelectorConfig {
    fn default() -> Self {
        Self {
            translation_threshold: 0.1, // 10cm
            rotation_threshold: 10.0,   // 10 degrees
            min_time_interval: 0.5,     // 0.5 seconds
            min_tracked_features: 50,
            max_keyframes: 100,
            use_covisibility: true,
            covisibility_threshold: 0.6,
        }
    }
}

/// Keyframe selection decision
#[derive(Debug, Clone, PartialEq)]
pub enum KeyframeDecision {
    /// Should insert a new keyframe
    Insert,
    /// Should not insert (tracking is good)
    Skip,
    /// Should force insert (tracking is poor)
    ForceInsert,
}

/// Keyframe selector for intelligent keyframe selection
pub struct KeyframeSelector {
    /// Last keyframe pose
    last_kf_pose: Option<SE3>,
    /// Last keyframe timestamp
    last_kf_timestamp: Option<f64>,
    /// Last number of tracked features
    last_kf_tracked_features: Option<usize>,
    /// Configuration
    config: KeyframeSelectorConfig,
    /// Number of keyframes inserted
    num_keyframes: usize,
    /// Tracking history for decision making
    tracking_history: Vec<TrackingStatus>,
}

/// Tracking status at a frame
#[derive(Debug, Clone)]
pub struct TrackingStatus {
    /// Number of tracked features
    pub num_tracked: usize,
    /// Number of total features
    pub num_total: usize,
    /// Timestamp
    pub timestamp: f64,
    /// Tracking quality (0-1)
    pub quality: f32,
}

impl KeyframeSelector {
    /// Create a new keyframe selector
    pub fn new(config: KeyframeSelectorConfig) -> Self {
        Self {
            last_kf_pose: None,
            last_kf_timestamp: None,
            last_kf_tracked_features: None,
            config,
            num_keyframes: 0,
            tracking_history: Vec::new(),
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(KeyframeSelectorConfig::default())
    }

    /// Decide whether to insert a keyframe
    ///
    /// # Arguments
    /// * `current_pose` - Current camera pose
    /// * `tracked_features` - Number of currently tracked features
    /// * `total_features` - Total number of features
    /// * `timestamp` - Current timestamp
    ///
    /// # Returns
    /// KeyframeDecision indicating whether to insert a keyframe
    pub fn should_insert(
        &mut self,
        current_pose: &SE3,
        tracked_features: usize,
        total_features: usize,
        timestamp: f64,
    ) -> KeyframeDecision {
        // Add to tracking history
        let quality = if total_features > 0 {
            tracked_features as f32 / total_features as f32
        } else {
            0.0
        };

        self.tracking_history.push(TrackingStatus {
            num_tracked: tracked_features,
            num_total: total_features,
            timestamp,
            quality,
        });

        // Keep history limited
        if self.tracking_history.len() > 30 {
            self.tracking_history.remove(0);
        }

        // Check if we have enough features
        if tracked_features < self.config.min_tracked_features {
            self.num_keyframes += 1;
            self.last_kf_pose = Some(current_pose.clone());
            self.last_kf_timestamp = Some(timestamp);
            self.last_kf_tracked_features = Some(tracked_features);
            return KeyframeDecision::ForceInsert;
        }

        // Check minimum time interval
        if let Some(last_ts) = self.last_kf_timestamp {
            if timestamp - last_ts < self.config.min_time_interval {
                return KeyframeDecision::Skip;
            }
        }

        // Check if first keyframe (no previous pose)
        if self.last_kf_pose.is_none() {
            self.num_keyframes += 1;
            self.last_kf_pose = Some(current_pose.clone());
            self.last_kf_timestamp = Some(timestamp);
            self.last_kf_tracked_features = Some(tracked_features);
            return KeyframeDecision::Insert;
        }

        // Check translation distance
        if let Some(last_pose) = &self.last_kf_pose {
            let inverse_last = last_pose.inverse();
            let delta = inverse_last.compose(current_pose);
            let t = delta.translation();
            let trans_dist = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();

            if trans_dist > self.config.translation_threshold {
                self.num_keyframes += 1;
                self.last_kf_pose = Some(current_pose.clone());
                self.last_kf_timestamp = Some(timestamp);
                self.last_kf_tracked_features = Some(tracked_features);
                return KeyframeDecision::Insert;
            }
        }

        // Check rotation angle
        if let Some(last_pose) = &self.last_kf_pose {
            let inverse_last = last_pose.inverse();
            let delta = inverse_last.compose(current_pose);
            let rot_mat = delta.rotation();
            // Compute rotation angle from matrix
            let angle = ((rot_mat[0][0] + rot_mat[1][1] + rot_mat[2][2] - 1.0) / 2.0).acos();

            let angle_degrees = angle.to_degrees();
            if angle_degrees > self.config.rotation_threshold {
                self.num_keyframes += 1;
                self.last_kf_pose = Some(current_pose.clone());
                self.last_kf_timestamp = Some(timestamp);
                self.last_kf_tracked_features = Some(tracked_features);
                return KeyframeDecision::Insert;
            }
        }

        // Check tracking quality degradation
        if let Some(last_tracked) = self.last_kf_tracked_features {
            let tracked_ratio = tracked_features as f32 / last_tracked as f32;
            if tracked_ratio < self.config.covisibility_threshold
                && tracked_features < self.config.min_tracked_features * 2
            {
                self.num_keyframes += 1;
                self.last_kf_pose = Some(current_pose.clone());
                self.last_kf_timestamp = Some(timestamp);
                self.last_kf_tracked_features = Some(tracked_features);
                return KeyframeDecision::Insert;
            }
        }

        // Update last keyframe state even when skipping
        self.last_kf_pose = Some(current_pose.clone());
        self.last_kf_timestamp = Some(timestamp);
        self.last_kf_tracked_features = Some(tracked_features);

        KeyframeDecision::Skip
    }

    /// Get the last keyframe pose
    pub fn last_keyframe_pose(&self) -> Option<&SE3> {
        self.last_kf_pose.as_ref()
    }

    /// Get number of keyframes inserted
    pub fn num_keyframes(&self) -> usize {
        self.num_keyframes
    }

    /// Get tracking quality history
    pub fn recent_tracking_quality(&self, n: usize) -> Vec<f32> {
        self.tracking_history
            .iter()
            .rev()
            .take(n)
            .map(|s| s.quality)
            .collect()
    }

    /// Get average tracking quality
    pub fn average_tracking_quality(&self) -> f32 {
        if self.tracking_history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.tracking_history.iter().map(|s| s.quality).sum();
        sum / self.tracking_history.len() as f32
    }

    /// Reset the selector
    pub fn reset(&mut self) {
        self.last_kf_pose = None;
        self.last_kf_timestamp = None;
        self.last_kf_tracked_features = None;
        self.num_keyframes = 0;
        self.tracking_history.clear();
    }

    /// Set the last keyframe manually (e.g., for initialization)
    pub fn set_last_keyframe(&mut self, pose: SE3, timestamp: f64, tracked_features: usize) {
        self.last_kf_pose = Some(pose);
        self.last_kf_timestamp = Some(timestamp);
        self.last_kf_tracked_features = Some(tracked_features);
    }
}

/// Keyframe culling - remove redundant keyframes
pub struct KeyframeCulling {
    /// Minimum angle between keyframes (degrees)
    min_angle: f32,
    /// Minimum distance between keyframes (meters)
    min_distance: f32,
    /// Maximum keyframes to keep
    max_keyframes: usize,
}

impl KeyframeCulling {
    /// Create a new keyframe culler
    pub fn new(min_angle: f32, min_distance: f32, max_keyframes: usize) -> Self {
        Self {
            min_angle,
            min_distance,
            max_keyframes,
        }
    }

    /// Decide which keyframes to remove
    ///
    /// # Arguments
    /// * `keyframes` - Vector of (id, pose) for keyframes
    ///
    /// # Returns
    /// Vector of keyframe IDs to remove
    pub fn compute_redundant_keyframes(&self, keyframes: &[(u64, SE3)]) -> Vec<u64> {
        let n = keyframes.len();
        if n <= 1 {
            return Vec::new();
        }

        let mut to_remove = Vec::new();

        // Check each keyframe (skip first and last)
        for i in 1..n - 1 {
            let prev = &keyframes[i - 1].1;
            let curr = &keyframes[i].1;
            let next = &keyframes[i + 1].1;

            // Check angle with neighbors
            let delta1 = prev.inverse().compose(curr);
            let delta2 = curr.inverse().compose(next);

            // Compute rotation angle from matrix
            let rot1 = delta1.rotation();
            let rot2 = delta2.rotation();
            let angle1 = ((rot1[0][0] + rot1[1][1] + rot1[2][2] - 1.0) / 2.0)
                .acos()
                .to_degrees();
            let angle2 = ((rot2[0][0] + rot2[1][1] + rot2[2][2] - 1.0) / 2.0)
                .acos()
                .to_degrees();

            // Check distance with neighbors
            let t1 = curr.translation();
            let t0 = prev.translation();
            let t2 = next.translation();
            let dist1 =
                ((t1[0] - t0[0]).powi(2) + (t1[1] - t0[1]).powi(2) + (t1[2] - t0[2]).powi(2))
                    .sqrt();
            let dist2 =
                ((t2[0] - t1[0]).powi(2) + (t2[1] - t1[1]).powi(2) + (t2[2] - t1[2]).powi(2))
                    .sqrt();

            // If both angles and distances are small, this keyframe is redundant
            if angle1 < self.min_angle
                && angle2 < self.min_angle
                && dist1 < self.min_distance
                && dist2 < self.min_distance
            {
                to_remove.push(keyframes[i].0);
            }
        }

        // Limit removals
        if keyframes.len() - to_remove.len() > self.max_keyframes {
            // Remove from the middle
            to_remove.truncate(keyframes.len() - self.max_keyframes);
        }

        to_remove
    }
}

impl Default for KeyframeCulling {
    fn default() -> Self {
        Self::new(5.0, 0.05, 100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyframe_selector_config_default() {
        let config = KeyframeSelectorConfig::default();
        assert_eq!(config.translation_threshold, 0.1);
        assert_eq!(config.rotation_threshold, 10.0);
        assert_eq!(config.min_tracked_features, 50);
    }

    #[test]
    fn test_keyframe_selector_creation() {
        let selector = KeyframeSelector::default();
        assert_eq!(selector.num_keyframes(), 0);
    }

    #[test]
    fn test_should_insert_first_frame() {
        let mut selector = KeyframeSelector::default();
        let pose = SE3::identity();

        let decision = selector.should_insert(&pose, 100, 200, 0.0);
        assert_eq!(decision, KeyframeDecision::Insert);
    }

    #[test]
    fn test_should_insert_translation_threshold() {
        let mut selector = KeyframeSelector::new(KeyframeSelectorConfig {
            translation_threshold: 0.2,
            ..Default::default()
        });

        let pose1 = SE3::identity();
        selector.should_insert(&pose1, 100, 200, 0.0);

        // Move 30cm - should trigger
        let pose2 = SE3::new(&[0.0, 0.0, 0.0, 1.0], &[0.3, 0.0, 0.0]);
        let decision = selector.should_insert(&pose2, 100, 200, 1.0);
        assert_eq!(decision, KeyframeDecision::Insert);
    }

    #[test]
    fn test_should_insert_min_time_interval() {
        let mut selector = KeyframeSelector::new(KeyframeSelectorConfig {
            min_time_interval: 1.0,
            ..Default::default()
        });

        let pose = SE3::identity();
        selector.should_insert(&pose, 100, 200, 0.0);

        // Try after 0.5 seconds - should skip (min interval is 1.0)
        let decision = selector.should_insert(&pose, 100, 200, 0.5);
        assert_eq!(decision, KeyframeDecision::Skip);
    }

    #[test]
    fn test_should_insert_low_features() {
        let mut selector = KeyframeSelector::default();
        let pose = SE3::identity();

        // Very few features - force insert
        let decision = selector.should_insert(&pose, 10, 200, 0.0);
        assert_eq!(decision, KeyframeDecision::ForceInsert);
    }

    #[test]
    fn test_keyframe_culling() {
        let culler = KeyframeCulling::default();
        let keyframes = vec![
            (0, SE3::identity()),
            (1, SE3::new(&[0.0, 0.0, 0.0, 1.0], &[0.01, 0.0, 0.0])), // Very close
            (2, SE3::new(&[0.0, 0.0, 0.0, 1.0], &[0.02, 0.0, 0.0])), // Very close
            (3, SE3::identity()),
        ];

        let to_remove = culler.compute_redundant_keyframes(&keyframes);
        assert!(!to_remove.is_empty());
    }

    #[test]
    fn test_reset() {
        let mut selector = KeyframeSelector::default();
        let pose = SE3::identity();
        selector.should_insert(&pose, 100, 200, 0.0);

        assert_eq!(selector.num_keyframes(), 1);

        selector.reset();
        assert_eq!(selector.num_keyframes(), 0);
    }

    #[test]
    fn test_tracking_quality() {
        let mut selector = KeyframeSelector::default();

        selector.should_insert(&SE3::identity(), 100, 200, 0.0);
        selector.should_insert(&SE3::identity(), 80, 200, 0.6);
        selector.should_insert(&SE3::identity(), 60, 200, 1.2);

        let recent = selector.recent_tracking_quality(2);
        assert_eq!(recent.len(), 2);

        let avg = selector.average_tracking_quality();
        assert!(avg > 0.0 && avg <= 1.0);
    }
}
