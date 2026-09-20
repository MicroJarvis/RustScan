//! KNN Matcher using kiddo KD-Tree for efficient feature matching
//!
//! This module provides a KD-Tree based K-Nearest Neighbors matcher
//! optimized for ORB-sized feature descriptors.
//!
//! Supports both Hamming distance (for binary descriptors like ORB/BRIEF)
//! and Squared Euclidean distance (for float descriptors like SIFT/SURF).

use crate::features::base::{
    Descriptors, FeatureError, FeatureMatcher, Match, ORB_DESCRIPTOR_SIZE,
};
use kiddo::KdTree;
use kiddo::SquaredEuclidean;
use std::collections::HashMap;

/// Distance metric for descriptor matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Hamming distance — correct for binary descriptors (ORB, BRIEF).
    Hamming,
    /// Squared Euclidean distance — correct for float descriptors (SIFT, SURF).
    SquaredEuclidean,
}

impl Default for DistanceMetric {
    fn default() -> Self {
        DistanceMetric::Hamming
    }
}

/// KNN Matcher using KD-Tree for efficient nearest neighbor search
///
/// Optimized for ORB descriptors (fixed-length byte vectors).
/// Defaults to Hamming distance for binary descriptors.
pub struct KnnMatcher {
    /// KD-Tree storing the training descriptors
    tree: KdTree<f64, ORB_DESCRIPTOR_SIZE>,
    /// Number of neighbors to search for
    k: usize,
    /// Whether the tree has been built
    built: bool,
    /// Lowe's ratio threshold (if enabled)
    ratio_threshold: Option<f64>,
    /// Distance metric to use
    metric: DistanceMetric,
}

impl KnnMatcher {
    /// Create a new KnnMatcher with specified number of neighbors
    ///
    /// # Arguments
    /// * `k` - Number of nearest neighbors to find
    ///
    /// # Example
    /// ```ignore
    /// let matcher = KnnMatcher::new(2);
    /// ```
    pub fn new(k: usize) -> Self {
        Self {
            tree: KdTree::new(),
            k,
            built: false,
            ratio_threshold: None,
            metric: DistanceMetric::default(),
        }
    }

    /// Enable Lowe's ratio test for matching.
    pub fn with_ratio_threshold(mut self, ratio_threshold: f64) -> Self {
        self.ratio_threshold = Some(ratio_threshold);
        self
    }

    /// Set the distance metric.
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Build the KD-Tree from training descriptors
    ///
    /// # Arguments
    /// * `descriptors` - Vector of descriptors, each descriptor is ORB-sized.
    ///
    /// # Example
    /// ```ignore
    /// let mut matcher = KnnMatcher::new(2);
    /// let descriptors: Vec<[f64; ORB_DESCRIPTOR_SIZE]> = vec![...];
    /// matcher.build_tree(&descriptors);
    /// ```
    pub fn build_tree(&mut self, descriptors: &[[f64; ORB_DESCRIPTOR_SIZE]]) {
        self.tree = KdTree::new();
        for (idx, descriptor) in descriptors.iter().enumerate() {
            self.tree.add(descriptor, idx as u64);
        }
        self.built = true;
    }

    /// Perform KNN matching for a single query descriptor
    ///
    /// # Arguments
    /// * `query` - Query descriptor (ORB-sized).
    ///
    /// # Returns
    /// Vector of (distance, index) pairs for the k nearest neighbors
    pub fn knn_match(&self, query: &[f64; ORB_DESCRIPTOR_SIZE]) -> Vec<(f64, u64)> {
        if !self.built {
            return Vec::new();
        }

        let results = self.tree.nearest_n::<SquaredEuclidean>(query, self.k);
        results.into_iter().map(|r| (r.distance, r.item)).collect()
    }

    /// Perform KNN matching for multiple query descriptors
    ///
    /// # Arguments
    /// * `queries` - Vector of query descriptors
    ///
    /// # Returns
    /// Vector of vectors, each containing (distance, index) pairs
    pub fn knn_match_batch(&self, queries: &[[f64; ORB_DESCRIPTOR_SIZE]]) -> Vec<Vec<(f64, u64)>> {
        queries.iter().map(|q| self.knn_match(q)).collect()
    }

    /// Match with Lowe's ratio test for filtering ambiguous matches
    ///
    /// The ratio test filters matches where the best match is significantly
    /// better than the second best match. This helps eliminate ambiguous matches.
    ///
    /// # Arguments
    /// * `query` - Query descriptor (ORB-sized).
    /// * `ratio_threshold` - Lowe's ratio test threshold (typically 0.7-0.8)
    ///
    /// # Returns
    /// Vector of (distance, index) pairs that pass the ratio test
    pub fn match_with_ratio(
        &self,
        query: &[f64; ORB_DESCRIPTOR_SIZE],
        ratio_threshold: f64,
    ) -> Vec<(f64, u64)> {
        let matches = self.knn_match(query);

        if matches.len() < 2 {
            return matches;
        }

        let best_distance = matches[0].0;
        let second_best_distance = matches[1].0;

        // Apply Lowe's ratio test
        if best_distance < ratio_threshold * second_best_distance {
            vec![matches[0].clone()]
        } else {
            Vec::new()
        }
    }

    /// Match multiple queries with ratio test
    ///
    /// # Arguments
    /// * `queries` - Vector of query descriptors
    /// * `ratio_threshold` - Lowe's ratio test threshold
    ///
    /// # Returns
    /// HashMap mapping query index to matched (distance, index) pairs
    pub fn match_batch_with_ratio(
        &self,
        queries: &[[f64; ORB_DESCRIPTOR_SIZE]],
        ratio_threshold: f64,
    ) -> HashMap<usize, Vec<(f64, u64)>> {
        let mut results = HashMap::new();
        for (idx, query) in queries.iter().enumerate() {
            let matches = self.match_with_ratio(query, ratio_threshold);
            if !matches.is_empty() {
                results.insert(idx, matches);
            }
        }
        results
    }

    /// Check if the KD-Tree has been built
    pub fn is_built(&self) -> bool {
        self.built
    }

    /// Get the number of neighbors (k)
    pub fn k(&self) -> usize {
        self.k
    }

    /// Get the number of descriptors in the tree
    pub fn size(&self) -> u64 {
        self.tree.size()
    }

    /// Convert Descriptors to f64 array format for KD-Tree matching
    fn convert_descriptors(descriptors: &Descriptors) -> Vec<[f64; ORB_DESCRIPTOR_SIZE]> {
        if descriptors.size != ORB_DESCRIPTOR_SIZE {
            return Vec::new();
        }
        descriptors
            .data
            .chunks(ORB_DESCRIPTOR_SIZE)
            .map(|chunk| {
                let mut arr = [0.0; ORB_DESCRIPTOR_SIZE];
                for (i, &byte) in chunk.iter().enumerate() {
                    arr[i] = byte as f64;
                }
                arr
            })
            .collect()
    }
}

impl FeatureMatcher for KnnMatcher {
    fn match_descriptors(
        &self,
        query: &Descriptors,
        train: &Descriptors,
    ) -> Result<Vec<Match>, FeatureError> {
        if query.is_empty() || train.is_empty() {
            return Ok(Vec::new());
        }

        match self.metric {
            DistanceMetric::Hamming => self.match_hamming(query, train),
            DistanceMetric::SquaredEuclidean => self.match_euclidean(query, train),
        }
    }
}

impl KnnMatcher {
    /// Hamming distance matching — brute-force on raw binary bytes.
    fn match_hamming(
        &self,
        query: &Descriptors,
        train: &Descriptors,
    ) -> Result<Vec<Match>, FeatureError> {
        if query.size != train.size || query.size == 0 {
            return Ok(Vec::new());
        }

        let ratio_threshold = self.ratio_threshold;
        let mut matches = Vec::new();

        for q_idx in 0..query.count {
            let q_desc = match query.get(q_idx) {
                Some(d) => d,
                None => continue,
            };

            let mut best: Option<(u32, usize)> = None;
            let mut second: Option<(u32, usize)> = None;

            for t_idx in 0..train.count {
                if let Some(t_desc) = train.get(t_idx) {
                    let dist = hamming_distance(q_desc, t_desc);
                    if best.map(|b| dist < b.0).unwrap_or(true) {
                        second = best;
                        best = Some((dist, t_idx));
                    } else if second.map(|s| dist < s.0).unwrap_or(true) {
                        second = Some((dist, t_idx));
                    }
                }
            }

            let Some(best) = best else {
                continue;
            };

            if let Some(ratio) = ratio_threshold {
                if let Some(second) = second {
                    if (best.0 as f64) < ratio * (second.0 as f64) {
                        matches.push(Match {
                            query_idx: q_idx as u32,
                            train_idx: best.1 as u32,
                            distance: best.0 as f32,
                        });
                    }
                } else {
                    matches.push(Match {
                        query_idx: q_idx as u32,
                        train_idx: best.1 as u32,
                        distance: best.0 as f32,
                    });
                }
            } else {
                matches.push(Match {
                    query_idx: q_idx as u32,
                    train_idx: best.1 as u32,
                    distance: best.0 as f32,
                });
            }
        }

        Ok(matches)
    }

    /// Squared Euclidean matching via KD-Tree — for float descriptors.
    fn match_euclidean(
        &self,
        query: &Descriptors,
        train: &Descriptors,
    ) -> Result<Vec<Match>, FeatureError> {
        let query_arrays = Self::convert_descriptors(query);
        let train_arrays = Self::convert_descriptors(train);

        let mut tree = KdTree::<f64, ORB_DESCRIPTOR_SIZE>::new();
        for (idx, desc) in train_arrays.iter().enumerate() {
            tree.add(desc, idx as u64);
        }

        let ratio_threshold = self.ratio_threshold;
        let mut matches = Vec::new();
        for (q_idx, q_desc) in query_arrays.iter().enumerate() {
            let results = tree.nearest_n::<SquaredEuclidean>(q_desc, self.k);

            if results.is_empty() {
                continue;
            }

            if let Some(ratio) = ratio_threshold {
                if results.len() >= 2 {
                    let best = &results[0];
                    let second = &results[1];
                    if best.distance < ratio * second.distance {
                        matches.push(Match {
                            query_idx: q_idx as u32,
                            train_idx: best.item as u32,
                            distance: best.distance as f32,
                        });
                    }
                } else {
                    let best = &results[0];
                    matches.push(Match {
                        query_idx: q_idx as u32,
                        train_idx: best.item as u32,
                        distance: best.distance as f32,
                    });
                }
            } else {
                let best = &results[0];
                matches.push(Match {
                    query_idx: q_idx as u32,
                    train_idx: best.item as u32,
                    distance: best.distance as f32,
                });
            }
        }

        Ok(matches)
    }
}

/// Compute Hamming distance between two binary descriptors using popcount.
fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to generate test descriptors
    fn generate_test_descriptors(count: usize, seed: usize) -> Vec<[f64; ORB_DESCRIPTOR_SIZE]> {
        (0..count)
            .map(|i| {
                let base = (i + seed) as f64;
                let mut desc = [0.0; ORB_DESCRIPTOR_SIZE];
                for j in 0..ORB_DESCRIPTOR_SIZE {
                    desc[j] = (base + j as f64 * 0.1).sin().abs();
                }
                desc
            })
            .collect()
    }

    #[test]
    fn test_new_matcher() {
        let matcher = KnnMatcher::new(2);
        assert_eq!(matcher.k(), 2);
        assert!(!matcher.is_built());
        assert_eq!(matcher.metric, DistanceMetric::Hamming);
    }

    #[test]
    fn test_build_tree() {
        let mut matcher = KnnMatcher::new(2);
        let descriptors = generate_test_descriptors(10, 42);

        matcher.build_tree(&descriptors);

        assert!(matcher.is_built());
        assert_eq!(matcher.size(), 10);
    }

    #[test]
    fn test_knn_match_single() {
        let mut matcher = KnnMatcher::new(3);
        let descriptors = generate_test_descriptors(10, 0);

        matcher.build_tree(&descriptors);

        // Query with the first descriptor - should find itself as nearest
        let query = descriptors[0];
        let matches = matcher.knn_match(&query);

        assert_eq!(matches.len(), 3);
        // First match should be the query itself (distance = 0)
        assert_eq!(matches[0].1, 0); // index 0
        assert!((matches[0].0).abs() < 1e-10);
    }

    #[test]
    fn test_knn_match_batch() {
        let mut matcher = KnnMatcher::new(2);
        let descriptors = generate_test_descriptors(20, 100);

        matcher.build_tree(&descriptors);

        let queries: Vec<[f64; ORB_DESCRIPTOR_SIZE]> =
            vec![descriptors[0], descriptors[5], descriptors[10]];
        let results = matcher.knn_match_batch(&queries);

        assert_eq!(results.len(), 3);
        for (i, matches) in results.iter().enumerate() {
            assert_eq!(matches.len(), 2);
            // Each query should find itself as the best match
            assert!((matches[0].0).abs() < 1e-10);
            assert_eq!(matches[0].1, (i * 5) as u64); // indices 0, 5, 10
        }
    }

    #[test]
    fn test_match_with_ratio() {
        let mut matcher = KnnMatcher::new(2);
        let descriptors = generate_test_descriptors(10, 0);

        matcher.build_tree(&descriptors);

        // Query with first descriptor
        let query = descriptors[0];
        let matches = matcher.match_with_ratio(&query, 0.75);

        // Should return match since best distance << second best
        assert!(!matches.is_empty());
        assert_eq!(matches[0].1, 0);
    }

    #[test]
    fn test_match_with_ratio_threshold() {
        let mut matcher = KnnMatcher::new(2);

        // Create descriptors where second match is too close to best
        let mut desc1 = [0.0; ORB_DESCRIPTOR_SIZE];
        let mut desc2 = [0.1; ORB_DESCRIPTOR_SIZE];
        let desc3 = [1.0; ORB_DESCRIPTOR_SIZE]; // Far

        for i in 0..ORB_DESCRIPTOR_SIZE {
            desc1[i] = (i as f64).sin();
            desc2[i] = desc1[i] + 0.01; // Very small difference
        }

        let descriptors = vec![desc1.clone(), desc2.clone(), desc3.clone()];
        matcher.build_tree(&descriptors);

        // Query with first descriptor
        let matches = matcher.match_with_ratio(&desc1, 0.75);

        let distance_1_2: f64 = desc1
            .iter()
            .zip(desc2.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt();
        let distance_1_1 = 0.0f64;

        let ratio = distance_1_1 / distance_1_2;
        if ratio > 0.75 {
            assert!(
                matches.is_empty(),
                "Expected no match due to ratio test, but got matches"
            );
        } else {
            assert!(!matches.is_empty(), "Expected match but got none");
        }
    }

    #[test]
    fn test_match_descriptors_hamming() {
        // Default metric is Hamming — test that match_descriptors uses it
        let matcher = KnnMatcher::new(2).with_ratio_threshold(0.8);

        // Create two distinct binary descriptors
        let mut query = Descriptors::with_capacity(1, ORB_DESCRIPTOR_SIZE);
        query.data.fill(0b1010_1010);

        let mut train = Descriptors::with_capacity(2, ORB_DESCRIPTOR_SIZE);
        // First train descriptor: identical to query (distance 0)
        for i in 0..ORB_DESCRIPTOR_SIZE {
            train.data[i] = 0b1010_1010;
        }
        // Second train descriptor: very different (high distance)
        for i in ORB_DESCRIPTOR_SIZE..(2 * ORB_DESCRIPTOR_SIZE) {
            train.data[i] = 0b0101_0101;
        }

        let matches = matcher.match_descriptors(&query, &train).unwrap();
        // Best=0 (identical), second=256 (all bits differ). Ratio 0/256 < 0.8 → match accepted.
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].train_idx, 0);
        assert_eq!(matches[0].distance, 0.0);
    }

    #[test]
    fn test_match_descriptors_hamming_rejects_ambiguous() {
        let matcher = KnnMatcher::new(2).with_ratio_threshold(0.8);

        // Both train descriptors identical → ratio test should reject
        let mut query = Descriptors::with_capacity(1, ORB_DESCRIPTOR_SIZE);
        query.data.fill(10);

        let mut train = Descriptors::with_capacity(2, ORB_DESCRIPTOR_SIZE);
        train.data.fill(10);

        let matches = matcher.match_descriptors(&query, &train).unwrap();
        assert!(
            matches.is_empty(),
            "Expected ratio test to reject ambiguous match"
        );
    }

    #[test]
    fn test_match_descriptors_euclidean() {
        let matcher = KnnMatcher::new(2)
            .with_metric(DistanceMetric::SquaredEuclidean)
            .with_ratio_threshold(0.8);

        let mut query = Descriptors::with_capacity(1, ORB_DESCRIPTOR_SIZE);
        query.data.fill(10);

        let mut train = Descriptors::with_capacity(2, ORB_DESCRIPTOR_SIZE);
        train.data.fill(10);

        let matches = matcher.match_descriptors(&query, &train).unwrap();
        assert!(
            matches.is_empty(),
            "Expected ratio test to reject ambiguous match"
        );
    }

    #[test]
    fn test_match_batch_with_ratio() {
        let mut matcher = KnnMatcher::new(2);
        let descriptors = generate_test_descriptors(15, 0);

        matcher.build_tree(&descriptors);

        let queries: Vec<[f64; ORB_DESCRIPTOR_SIZE]> =
            vec![descriptors[0], descriptors[3], descriptors[7]];
        let results = matcher.match_batch_with_ratio(&queries, 0.75);

        // Should have matches for all 3 queries
        assert_eq!(results.len(), 3);
        assert!(results.contains_key(&0));
        assert!(results.contains_key(&1));
        assert!(results.contains_key(&2));
    }

    #[test]
    fn test_empty_tree_match() {
        let matcher = KnnMatcher::new(2);
        let query = [0.0; ORB_DESCRIPTOR_SIZE];

        let matches = matcher.knn_match(&query);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_distance_ordering() {
        let mut matcher = KnnMatcher::new(3);

        // Create descriptors at known distances
        let mut center = [0.0; ORB_DESCRIPTOR_SIZE];
        for i in 0..ORB_DESCRIPTOR_SIZE {
            center[i] = (i as f64 * 0.5).sin();
        }

        let mut near = center;
        near[0] += 0.1; // Small perturbation

        let mut far = center;
        far[0] += 10.0; // Large perturbation

        let descriptors = vec![far, near, center];
        matcher.build_tree(&descriptors);

        let matches = matcher.knn_match(&center);

        // Should be ordered by distance
        assert!(matches[0].0 <= matches[1].0);
        assert!(matches[1].0 <= matches[2].0);
    }
}
