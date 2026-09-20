//! Loop closing detector implementation
//!
//! This module implements loop detection based on Bag of Words (BoW).

use crate::core::{Map, SE3};
use crate::features::base::ORB_DESCRIPTOR_SIZE;
use nalgebra::{Matrix3, Vector3};
use std::collections::HashMap;

/// A loop candidate detected between frames
#[derive(Debug, Clone)]
pub struct LoopCandidate {
    /// Keyframe ID that forms a loop
    pub keyframe_id: u64,
    /// BoW similarity score
    pub score: f32,
    /// Timestamp of the keyframe
    pub timestamp: f64,
}

/// Result of loop detection
#[derive(Debug, Clone)]
pub struct LoopDetectionResult {
    /// Whether a loop was detected
    pub loop_detected: bool,
    /// The loop candidate (if detected)
    pub candidate: Option<LoopCandidate>,
    /// Similarity transform (Sim3) if computed
    pub sim3: Option<Sim3>,
    /// Number of matched features
    pub num_matches: usize,
    /// Number of inliers
    pub num_inliers: usize,
}

impl LoopDetectionResult {
    /// Create a negative result
    pub fn no_loop() -> Self {
        Self {
            loop_detected: false,
            candidate: None,
            sim3: None,
            num_matches: 0,
            num_inliers: 0,
        }
    }
}

/// Similarity transform (Sim3) representation
/// Sim3 = s * R * t + t
#[derive(Debug, Clone)]
pub struct Sim3 {
    /// Rotation matrix (3x3)
    pub rotation: [[f32; 3]; 3],
    /// Translation vector
    pub translation: [f32; 3],
    /// Scale factor
    pub scale: f32,
}

impl Sim3 {
    /// Create identity Sim3
    pub fn identity() -> Self {
        Self {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
            scale: 1.0,
        }
    }

    /// Convert to SE3 (assuming scale = 1)
    pub fn to_se3(&self) -> SE3 {
        SE3::from_rotation_translation(&self.rotation, &self.translation)
    }
}

/// Loop detector using Bag of Words
pub struct LoopDetector {
    /// Minimum score to consider as loop candidate
    min_loop_score: f32,
    /// Minimum number of matches to accept loop
    min_matches: usize,
    /// Minimum distance (in frames) between current and candidate
    min_distance: u64,
    /// Maximum time gap (seconds) for loop detection
    _max_time_gap: f64,
    /// Confidence threshold for consistency check
    consistency_threshold: usize,
    /// Recent loop candidates for consistency check
    recent_loops: Vec<u64>,
    /// Maximum recent loops to store
    max_recent_loops: usize,
}

impl LoopDetector {
    /// Create a new loop detector
    pub fn new() -> Self {
        Self {
            min_loop_score: 0.05,
            min_matches: 20,
            min_distance: 30,
            _max_time_gap: 60.0,
            consistency_threshold: 3,
            recent_loops: Vec::new(),
            max_recent_loops: 20,
        }
    }

    /// Get minimum loop score
    pub fn min_loop_score(&self) -> f32 {
        self.min_loop_score
    }

    /// Set minimum loop score
    pub fn set_min_loop_score(&mut self, score: f32) {
        self.min_loop_score = score;
    }

    /// Set minimum distance between frames
    pub fn set_min_distance_between_frames(&mut self, distance: u64) {
        self.min_distance = distance;
    }

    /// Set minimum matches
    pub fn set_min_matches(&mut self, matches: usize) {
        self.min_matches = matches;
    }

    /// Get minimum matches
    pub fn min_matches(&self) -> usize {
        self.min_matches
    }

    /// Compute BoW similarity between two descriptors
    ///
    /// This is a simplified BoW implementation using
    /// descriptor clustering for vocabulary
    fn compute_bow_similarity(&self, query: &[u8], train: &[u8]) -> f32 {
        if query.is_empty() || train.is_empty() {
            return 0.0;
        }

        if query.len() < ORB_DESCRIPTOR_SIZE || train.len() < ORB_DESCRIPTOR_SIZE {
            return 0.0;
        }

        let mut q_tf: HashMap<u32, f32> = HashMap::new();
        let mut t_tf: HashMap<u32, f32> = HashMap::new();
        let mut q_total = 0.0f32;
        let mut t_total = 0.0f32;

        for desc in query.chunks_exact(ORB_DESCRIPTOR_SIZE) {
            let word = descriptor_to_word(desc);
            *q_tf.entry(word).or_insert(0.0) += 1.0;
            q_total += 1.0;
        }
        for desc in train.chunks_exact(ORB_DESCRIPTOR_SIZE) {
            let word = descriptor_to_word(desc);
            *t_tf.entry(word).or_insert(0.0) += 1.0;
            t_total += 1.0;
        }
        if q_total == 0.0 || t_total == 0.0 {
            return 0.0;
        }

        for v in q_tf.values_mut() {
            *v /= q_total;
        }
        for v in t_tf.values_mut() {
            *v /= t_total;
        }

        let mut dot = 0.0f32;
        let mut q_norm = 0.0f32;
        let mut t_norm = 0.0f32;

        for (&word, &q_tf_word) in &q_tf {
            let mut df = 0.0f32;
            if q_tf.contains_key(&word) {
                df += 1.0;
            }
            if t_tf.contains_key(&word) {
                df += 1.0;
            }
            let idf = ((1.0 + 2.0) / (1.0 + df)).ln() + 1.0;
            let qw = q_tf_word * idf;
            let tw = t_tf.get(&word).copied().unwrap_or(0.0) * idf;
            dot += qw * tw;
            q_norm += qw * qw;
        }

        for (&word, &t_tf_word) in &t_tf {
            let mut df = 0.0f32;
            if q_tf.contains_key(&word) {
                df += 1.0;
            }
            if t_tf.contains_key(&word) {
                df += 1.0;
            }
            let idf = ((1.0 + 2.0) / (1.0 + df)).ln() + 1.0;
            let tw = t_tf_word * idf;
            t_norm += tw * tw;
        }

        if q_norm <= 1e-8 || t_norm <= 1e-8 {
            0.0
        } else {
            (dot / (q_norm.sqrt() * t_norm.sqrt())).clamp(0.0, 1.0)
        }
    }

    /// Compute loop candidates for a given frame
    ///
    /// # Arguments
    /// * `map` - The map containing keyframes
    /// * `current_frame_id` - ID of the current frame
    /// * `descriptors` - Descriptors of the current frame
    ///
    /// # Returns
    /// Vector of loop candidates sorted by score
    pub fn compute_loop_candidates(
        &self,
        map: &Map,
        current_frame_id: u64,
        descriptors: &[u8],
    ) -> Vec<LoopCandidate> {
        let mut candidates = Vec::new();

        // Get all keyframes from map
        for keyframe in map.keyframes() {
            let kf_id = keyframe.id();

            // Skip if too close to current frame
            if current_frame_id.saturating_sub(kf_id) < self.min_distance {
                continue;
            }

            // Skip if this is the first keyframe
            if kf_id == 0 {
                continue;
            }

            // Compute BoW similarity
            let score = self.compute_bow_similarity(descriptors, &keyframe.features.descriptors);

            if score > self.min_loop_score {
                candidates.push(LoopCandidate {
                    keyframe_id: kf_id,
                    score,
                    timestamp: keyframe.frame.timestamp,
                });
            }
        }

        // Sort by score descending
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates
    }

    /// Check loop consistency using temporal filtering
    ///
    /// This ensures that loop candidates are consistent over time
    /// to avoid false positives from transient matches
    pub fn compute_loop_consistency(
        &self,
        candidates: &[LoopCandidate],
        current_frame_id: u64,
    ) -> Vec<LoopCandidate> {
        let mut filtered: Vec<LoopCandidate> = candidates
            .iter()
            .filter(|candidate| {
                let distance = current_frame_id.saturating_sub(candidate.keyframe_id);
                distance >= self.min_distance
            })
            .cloned()
            .collect();

        if filtered.is_empty() {
            return filtered;
        }

        filtered.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Bootstrap: when no confirmed loops exist yet, keep only the strongest candidate.
        if self.recent_loops.is_empty() {
            filtered.truncate(1);
            return filtered;
        }

        let warmup = self.recent_loops.len() < self.consistency_threshold;
        let neighbor_window = (self.min_distance / 2).max(5);
        let strong_score = (self.min_loop_score * 1.5).min(0.9);

        filtered
            .into_iter()
            .filter(|candidate| {
                let near_recent = self
                    .recent_loops
                    .iter()
                    .any(|&recent_id| recent_id.abs_diff(candidate.keyframe_id) <= neighbor_window);

                near_recent || (warmup && candidate.score >= strong_score)
            })
            .collect()
    }

    /// Add a confirmed loop to recent history
    pub fn add_confirmed_loop(&mut self, keyframe_id: u64) {
        self.recent_loops.push(keyframe_id);
        if self.recent_loops.len() > self.max_recent_loops {
            self.recent_loops.remove(0);
        }
    }

    /// Clear recent loop history
    pub fn clear_history(&mut self) {
        self.recent_loops.clear();
    }
}

impl Default for LoopDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute similarity transform (Sim3) between two point clouds
///
/// This implements the method to compute Sim3 from matched 3D points:
/// - Find rotation, translation, and scale that best aligns two point sets
///
/// # Arguments
/// * `rotation` - Initial rotation matrix (3x3)
/// * `translation` - Initial translation vector
/// * `scale` - Initial scale
///
/// # Returns
/// Option containing the Sim3 transform if computation succeeded
pub fn compute_similarity_transform(
    rotation: &[[f32; 3]; 3],
    translation: &[f32; 3],
    scale: f32,
) -> Option<Sim3> {
    // Simplified implementation: validate and return the input as Sim3
    // A full implementation would use SVD to find optimal transformation

    // Validate rotation matrix (should be orthogonal)
    let det = rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0]);

    if (det - 1.0).abs() > 0.1 {
        // Not a valid rotation matrix
        return None;
    }

    // Scale should be positive
    if scale <= 0.0 {
        return None;
    }

    Some(Sim3 {
        rotation: *rotation,
        translation: *translation,
        scale,
    })
}

/// Compute Sim3 from matched 3D points between two keyframes
///
/// This is a more complete implementation that takes actual point matches
pub fn compute_sim3_from_matches(points1: &[[f32; 3]], points2: &[[f32; 3]]) -> Option<Sim3> {
    if points1.len() != points2.len() || points1.len() < 3 {
        return None;
    }

    // Compute centroids
    let mut centroid1 = [0.0f32; 3];
    let mut centroid2 = [0.0f32; 3];

    for i in 0..points1.len() {
        for j in 0..3 {
            centroid1[j] += points1[i][j];
            centroid2[j] += points2[i][j];
        }
    }

    let n = points1.len() as f32;
    for j in 0..3 {
        centroid1[j] /= n;
        centroid2[j] /= n;
    }

    // Umeyama alignment: estimate Sim3 with SVD
    let mut sigma = Matrix3::<f32>::zeros();
    let mut var1 = 0.0f32;
    let n = points1.len() as f32;

    for i in 0..points1.len() {
        let p1 = Vector3::new(
            points1[i][0] - centroid1[0],
            points1[i][1] - centroid1[1],
            points1[i][2] - centroid1[2],
        );
        let p2 = Vector3::new(
            points2[i][0] - centroid2[0],
            points2[i][1] - centroid2[1],
            points2[i][2] - centroid2[2],
        );
        sigma += p2 * p1.transpose();
        var1 += p1.dot(&p1);
    }

    sigma /= n;
    var1 /= n;
    if var1 < 1e-8 {
        return None;
    }

    let svd = sigma.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let mut s = Matrix3::<f32>::identity();
    if (u * v_t).determinant() < 0.0 {
        s[(2, 2)] = -1.0;
    }

    let r = u * s * v_t;
    let s_mat = Matrix3::from_diagonal(&svd.singular_values);
    let scale = (s_mat * s).trace() / var1;
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }

    let c1 = Vector3::new(centroid1[0], centroid1[1], centroid1[2]);
    let c2 = Vector3::new(centroid2[0], centroid2[1], centroid2[2]);
    let t = c2 - scale * (r * c1);

    let rotation = [
        [r[(0, 0)], r[(0, 1)], r[(0, 2)]],
        [r[(1, 0)], r[(1, 1)], r[(1, 2)]],
        [r[(2, 0)], r[(2, 1)], r[(2, 2)]],
    ];

    Some(Sim3 {
        rotation,
        translation: [t[0], t[1], t[2]],
        scale,
    })
}

/// Calculate Hamming distance between two descriptors
#[allow(dead_code)]
fn hamming_distance(a: &[u8], b: &[u8]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x ^ *y).count_ones() as f32)
        .sum()
}

fn descriptor_to_word(desc: &[u8]) -> u32 {
    // FNV-1a hash on prefix bytes for coarse visual-word assignment.
    let mut hash: u32 = 2166136261;
    for &b in desc.iter().take(8) {
        hash ^= b as u32;
        hash = hash.wrapping_mul(16777619);
    }
    hash & 0x0FFF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loop_detector_default() {
        let detector = LoopDetector::new();
        assert!(detector.min_loop_score() > 0.0);
    }

    #[test]
    fn test_sim3_identity() {
        let sim3 = Sim3::identity();
        assert!((sim3.scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sim3_to_se3() {
        let sim3 = Sim3::identity();
        let _se3 = sim3.to_se3();
    }

    #[test]
    fn test_compute_similarity_transform_identity() {
        let result = compute_similarity_transform(
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &[0.0, 0.0, 0.0],
            1.0,
        );
        assert!(result.is_some());
    }

    #[test]
    fn test_compute_similarity_transform_invalid_rotation() {
        let result = compute_similarity_transform(
            &[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], // Invalid rotation
            &[0.0, 0.0, 0.0],
            1.0,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_compute_sim3_from_matches() {
        // Same points should give identity transform
        let points = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let result = compute_sim3_from_matches(&points, &points);

        assert!(result.is_some());
        let sim3 = result.unwrap();
        assert!((sim3.scale - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_compute_sim3_from_matches_different() {
        // Translated points
        let points1 = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let points2 = vec![[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [1.0, 2.0, 0.0]];

        let result = compute_sim3_from_matches(&points1, &points2);
        assert!(result.is_some());
    }

    #[test]
    fn test_compute_sim3_from_matches_empty() {
        let points1: Vec<[f32; 3]> = vec![];
        let points2: Vec<[f32; 3]> = vec![];

        let result = compute_sim3_from_matches(&points1, &points2);
        assert!(result.is_none());
    }

    #[test]
    fn test_loop_consistency_bootstrap_keeps_strongest() {
        let detector = LoopDetector::new();
        let candidates = vec![
            LoopCandidate {
                keyframe_id: 10,
                score: 0.9,
                timestamp: 1.0,
            },
            LoopCandidate {
                keyframe_id: 20,
                score: 0.7,
                timestamp: 2.0,
            },
            LoopCandidate {
                keyframe_id: 30,
                score: 0.6,
                timestamp: 3.0,
            },
        ];

        let consistent = detector.compute_loop_consistency(&candidates, 100);
        assert_eq!(consistent.len(), 1);
        assert_eq!(consistent[0].keyframe_id, 10);
    }

    #[test]
    fn test_loop_consistency_prefers_recent_neighbors() {
        let mut detector = LoopDetector::new();
        detector.add_confirmed_loop(25);

        let candidates = vec![
            LoopCandidate {
                keyframe_id: 24,
                score: 0.06,
                timestamp: 1.0,
            },
            LoopCandidate {
                keyframe_id: 80,
                score: 0.06,
                timestamp: 2.0,
            },
        ];

        let consistent = detector.compute_loop_consistency(&candidates, 200);
        assert_eq!(consistent.len(), 1);
        assert_eq!(consistent[0].keyframe_id, 24);
    }

    #[test]
    fn test_loop_consistency_warmup_allows_strong_outlier() {
        let mut detector = LoopDetector::new();
        detector.add_confirmed_loop(25);

        let candidates = vec![LoopCandidate {
            keyframe_id: 120,
            score: 0.3,
            timestamp: 1.0,
        }];

        let consistent = detector.compute_loop_consistency(&candidates, 300);
        assert_eq!(consistent.len(), 1);
        assert_eq!(consistent[0].keyframe_id, 120);
    }

    #[test]
    fn test_loop_consistency_after_warmup_requires_history_neighbor() {
        let mut detector = LoopDetector::new();
        detector.add_confirmed_loop(10);
        detector.add_confirmed_loop(40);
        detector.add_confirmed_loop(70);

        let candidates = vec![
            LoopCandidate {
                keyframe_id: 72,
                score: 0.1,
                timestamp: 1.0,
            },
            LoopCandidate {
                keyframe_id: 130,
                score: 0.9,
                timestamp: 2.0,
            },
        ];

        let consistent = detector.compute_loop_consistency(&candidates, 250);
        assert_eq!(consistent.len(), 1);
        assert_eq!(consistent[0].keyframe_id, 72);
    }
}
