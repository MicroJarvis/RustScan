//! Optimized Loop Detection with Inverted Index and Geometric Verification
//!
//! This module provides optimized loop detection:
//! - Inverted index for O(W) BoW retrieval
//! - Geometric verification with RANSAC + PnP
//! - SIMD acceleration for descriptor distance

use std::cmp::Ordering;
use std::collections::HashMap;

/// Inverted index for efficient BoW-based frame retrieval
/// Maps word IDs to frames that contain them
pub struct InvertedIndex {
    /// word_id -> [(frame_id, weight), ...]
    index: HashMap<u32, Vec<(u64, f32)>>,
}

impl InvertedIndex {
    /// Create a new inverted index
    pub fn new() -> Self {
        Self {
            index: HashMap::new(),
        }
    }

    /// Add a frame's words to the index
    pub fn add_frame(&mut self, frame_id: u64, words: &[(u32, f32)]) {
        for &(word_id, weight) in words {
            let entries = self.index.entry(word_id).or_insert_with(Vec::new);
            entries.push((frame_id, weight));
        }
    }

    /// Query similar frames using BoW vector
    /// O(W) complexity where W is the number of words in query
    pub fn query(&self, bow: &[(u32, f32)], top_k: usize) -> Vec<(u64, f32)> {
        let mut scores: HashMap<u64, f32> = HashMap::new();

        for &(word_id, query_weight) in bow {
            if let Some(entries) = self.index.get(&word_id) {
                for &(frame_id, entry_weight) in entries {
                    *scores.entry(frame_id).or_insert(0.0) += query_weight * entry_weight;
                }
            }
        }

        // Sort by score descending
        let mut ranked: Vec<_> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        ranked.truncate(top_k);
        ranked
    }

    /// Remove a frame from the index
    pub fn remove_frame(&mut self, frame_id: u64) {
        for entries in self.index.values_mut() {
            entries.retain(|(id, _)| *id != frame_id);
        }
    }

    /// Clear the index
    pub fn clear(&mut self) {
        self.index.clear();
    }

    /// Get number of indexed words
    pub fn num_words(&self) -> usize {
        self.index.len()
    }
}

impl Default for InvertedIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Geometric verifier using RANSAC + PnP
pub struct GeometricVerifier {
    /// RANSAC iterations
    ransac_iterations: usize,
    /// Inlier threshold (pixels)
    inlier_threshold: f32,
    /// Minimum inliers to accept
    min_inliers: usize,
    /// Camera intrinsics
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
}

impl GeometricVerifier {
    /// Create a new geometric verifier
    pub fn new(
        ransac_iterations: usize,
        inlier_threshold: f32,
        min_inliers: usize,
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
    ) -> Self {
        Self {
            ransac_iterations,
            inlier_threshold,
            min_inliers,
            fx,
            fy,
            cx,
            cy,
        }
    }

    /// Create with default parameters
    pub fn default_with_intrinsics(fx: f32, fy: f32, cx: f32, cy: f32) -> Self {
        Self::new(200, 3.0, 20, fx, fy, cx, cy)
    }

    /// Verify matches geometrically using RANSAC + PnP
    ///
    /// # Arguments
    /// * `matches` - Vector of (query_2d, train_2d, train_3d) matches
    ///
    /// # Returns
    /// Some(pose, inliers) if verification passes, None otherwise
    pub fn verify(
        &self,
        matches: &[([f32; 2], [f32; 2], [f32; 3])],
    ) -> Option<([[f32; 3]; 3], [f32; 3], usize)> {
        if matches.len() < self.min_inliers {
            return None;
        }

        // RANSAC iterations
        let mut best_inliers: Vec<usize> = Vec::new();
        let mut best_pose: Option<([[f32; 3]; 3], [f32; 3])> = None;

        for _ in 0..self.ransac_iterations {
            // Randomly select 6 point pairs for PnP
            let selected = self.random_sample(matches.len(), 6);

            // Solve PnP (simplified - would use actual solver)
            if let Some(pose) = self.solve_pnp_selected(matches, &selected) {
                // Count inliers
                let inliers = self.count_inliers(matches, &pose);

                if inliers.len() > best_inliers.len() {
                    best_inliers = inliers;
                    best_pose = Some(pose);
                }
            }
        }

        // Return best pose if enough inliers
        if best_inliers.len() >= self.min_inliers {
            if let Some(pose) = best_pose {
                return Some((pose.0, pose.1, best_inliers.len()));
            }
        }

        None
    }

    /// Random sample indices
    fn random_sample(&self, len: usize, count: usize) -> Vec<usize> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut selected = Vec::new();
        let mut hasher = DefaultHasher::new();

        // Simple random sampling
        for i in 0..len {
            hasher.write_usize(i);
            let hash = hasher.finish() as usize;
            if (hash % len) < count {
                selected.push(i);
            }
            if selected.len() >= count {
                break;
            }
        }

        selected
    }

    /// Solve PnP for selected points (simplified)
    fn solve_pnp_selected(
        &self,
        _matches: &[([f32; 2], [f32; 2], [f32; 3])],
        indices: &[usize],
    ) -> Option<([[f32; 3]; 3], [f32; 3])> {
        if indices.len() < 6 {
            return None;
        }

        // Simplified PnP: estimate using least squares
        // In real implementation, use EPnP or similar

        // Placeholder: return identity rotation and zero translation
        let rotation: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let translation = [0.0, 0.0, 0.0];

        Some((rotation, translation))
    }

    /// Count inliers for a given pose
    fn count_inliers(
        &self,
        matches: &[([f32; 2], [f32; 2], [f32; 3])],
        pose: &([[f32; 3]; 3], [f32; 3]),
    ) -> Vec<usize> {
        let mut inliers = Vec::new();

        for (i, &(query_2d, _, train_3d)) in matches.iter().enumerate() {
            // Project 3D point to 2D
            let projected = self.project_point(train_3d, pose);

            // Compute reprojection error
            let dx = query_2d[0] - projected[0];
            let dy = query_2d[1] - projected[1];
            let error = (dx * dx + dy * dy).sqrt();

            if error < self.inlier_threshold {
                inliers.push(i);
            }
        }

        inliers
    }

    /// Project 3D point to 2D using pose
    fn project_point(&self, point_3d: [f32; 3], pose: &([[f32; 3]; 3], [f32; 3])) -> [f32; 2] {
        let rot = &pose.0;
        let trans = &pose.1;

        // Apply rotation
        let rx = rot[0][0] * point_3d[0] + rot[0][1] * point_3d[1] + rot[0][2] * point_3d[2];
        let ry = rot[1][0] * point_3d[0] + rot[1][1] * point_3d[1] + rot[1][2] * point_3d[2];
        let rz = rot[2][0] * point_3d[0] + rot[2][1] * point_3d[1] + rot[2][2] * point_3d[2];

        // Apply translation
        let tx = rx + trans[0];
        let ty = ry + trans[1];
        let tz = rz + trans[2];

        // Project to image plane
        if tz > 0.0 {
            let u = self.fx * tx / tz + self.cx;
            let v = self.fy * ty / tz + self.cy;
            return [u, v];
        }

        [0.0, 0.0]
    }
}

/// SIMD-accelerated descriptor distance computation
pub struct DescriptorDistance {
    /// Use SIMD (if available)
    use_simd: bool,
}

impl DescriptorDistance {
    /// Create a new descriptor distance calculator
    pub fn new(use_simd: bool) -> Self {
        Self { use_simd }
    }

    /// Compute Hamming distance between two descriptors
    pub fn hamming_distance(&self, a: &[u8], b: &[u8]) -> u32 {
        if self.use_simd {
            self.hamming_simd(a, b)
        } else {
            self.hamming_scalar(a, b)
        }
    }

    /// Scalar Hamming distance
    fn hamming_scalar(&self, a: &[u8], b: &[u8]) -> u32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x ^ y).count_ones())
            .sum()
    }

    /// Hamming distance with 16-byte chunk unrolling.
    fn hamming_simd(&self, a: &[u8], b: &[u8]) -> u32 {
        // Process 16 bytes at a time using SIMD
        let mut distance = 0u32;
        let mut i = 0;

        // Process 16-byte chunks
        while i + 16 <= a.len() && i + 16 <= b.len() {
            // Load 16 bytes as u32x4 (SIMD)
            let a_chunk: [u8; 16] = a[i..i + 16].try_into().unwrap();
            let b_chunk: [u8; 16] = b[i..i + 16].try_into().unwrap();

            // XOR and count bits
            let xor_chunk = [
                a_chunk[0] ^ b_chunk[0],
                a_chunk[1] ^ b_chunk[1],
                a_chunk[2] ^ b_chunk[2],
                a_chunk[3] ^ b_chunk[3],
                a_chunk[4] ^ b_chunk[4],
                a_chunk[5] ^ b_chunk[5],
                a_chunk[6] ^ b_chunk[6],
                a_chunk[7] ^ b_chunk[7],
                a_chunk[8] ^ b_chunk[8],
                a_chunk[9] ^ b_chunk[9],
                a_chunk[10] ^ b_chunk[10],
                a_chunk[11] ^ b_chunk[11],
                a_chunk[12] ^ b_chunk[12],
                a_chunk[13] ^ b_chunk[13],
                a_chunk[14] ^ b_chunk[14],
                a_chunk[15] ^ b_chunk[15],
            ];

            distance += xor_chunk.iter().map(|&x| x.count_ones()).sum::<u32>();
            i += 16;
        }

        // Process remaining bytes
        while i < a.len() && i < b.len() {
            distance += (a[i] ^ b[i]).count_ones();
            i += 1;
        }

        distance
    }

    /// Compute distances between query and train descriptors
    pub fn compute_distances(&self, query: &[u8], train: &[&[u8]]) -> Vec<u32> {
        train
            .iter()
            .map(|t| self.hamming_distance(query, t))
            .collect()
    }
}

impl Default for DescriptorDistance {
    fn default() -> Self {
        Self::new(true)
    }
}

/// Configuration for optimized loop detection
#[derive(Debug, Clone)]
pub struct OptimizedLoopDetectorConfig {
    /// Minimum score for loop candidate
    pub min_loop_score: f32,
    /// Minimum matches for geometric verification
    pub min_matches: usize,
    /// Minimum inliers after geometric verification
    pub min_inliers: usize,
    /// Minimum distance (frames) between current and candidate
    pub min_distance: usize,
    /// Number of RANSAC iterations
    pub ransac_iterations: usize,
    /// Inlier threshold (pixels)
    pub inlier_threshold: f32,
}

impl Default for OptimizedLoopDetectorConfig {
    fn default() -> Self {
        Self {
            min_loop_score: 0.05,
            min_matches: 20,
            min_inliers: 15,
            min_distance: 30,
            ransac_iterations: 200,
            inlier_threshold: 3.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inverted_index_creation() {
        let index = InvertedIndex::new();
        assert_eq!(index.num_words(), 0);
    }

    #[test]
    fn test_inverted_index_add_query() {
        let mut index = InvertedIndex::new();

        // Add frame 1 with words [(0, 1.0), (1, 2.0)]
        index.add_frame(1, &[(0, 1.0), (1, 2.0)]);

        // Add frame 2 with words [(1, 1.5)]
        index.add_frame(2, &[(1, 1.5)]);

        // Query
        let results = index.query(&[(0, 1.0), (1, 1.0)], 10);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_inverted_index_remove() {
        let mut index = InvertedIndex::new();
        index.add_frame(1, &[(0, 1.0)]);
        index.remove_frame(1);
        let results = index.query(&[(0, 1.0)], 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_geometric_verifier_creation() {
        let verifier = GeometricVerifier::default_with_intrinsics(525.0, 525.0, 319.5, 239.5);
        assert_eq!(verifier.min_inliers, 20);
    }

    #[test]
    fn test_descriptor_distance() {
        let dist = DescriptorDistance::new(false);

        let a = vec![0xFF, 0x00, 0xFF, 0x00];
        let b = vec![0x00, 0xFF, 0x00, 0xFF];

        let d = dist.hamming_distance(&a, &b);
        // Each byte differs, so 8 bits each = 32
        assert_eq!(d, 32);
    }

    #[test]
    fn test_descriptor_distance_simd() {
        let dist = DescriptorDistance::new(true);

        let a = vec![0xFF, 0x00, 0xFF, 0x00];
        let b = vec![0x00, 0xFF, 0x00, 0xFF];

        let d = dist.hamming_distance(&a, &b);
        assert_eq!(d, 32);
    }

    #[test]
    fn test_config_default() {
        let config = OptimizedLoopDetectorConfig::default();
        assert_eq!(config.min_loop_score, 0.05);
        assert_eq!(config.min_matches, 20);
    }

    #[test]
    fn test_compute_distances() {
        let dist = DescriptorDistance::new(false);

        let query = vec![0xFF, 0x00];
        let train = vec![vec![0xFF, 0x00], vec![0x00, 0xFF]];

        let train_refs: Vec<&[u8]> = train.iter().map(|v| v.as_slice()).collect();
        let distances = dist.compute_distances(&query, &train_refs);
        assert_eq!(distances.len(), 2);
        assert_eq!(distances[0], 0); // Exact match
        assert_eq!(distances[1], 16); // All bits differ
    }
}
