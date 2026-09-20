//! Keyframe Database for loop closing
//!
//! Provides efficient indexing and retrieval of keyframes for loop detection.

use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

/// A keyframe entry in the database
#[derive(Debug, Clone)]
pub struct KeyFrameEntry {
    /// Keyframe ID
    pub id: u64,
    /// Word IDs present in this keyframe
    pub word_ids: Vec<u32>,
    /// BoW vector (word_id -> weight)
    pub bow_vector: HashMap<u32, f32>,
    /// Timestamp
    pub timestamp: f64,
    /// Connected keyframe IDs (covisibility graph)
    pub connected_ids: Vec<u64>,
}

/// Keyframe Database for loop detection
pub struct KeyFrameDatabase {
    /// All keyframe entries
    entries: RwLock<HashMap<u64, KeyFrameEntry>>,
    /// Inverted index: word_id -> keyframe IDs
    inverted_index: RwLock<HashMap<u32, HashSet<u64>>>,
    /// Number of words in vocabulary
    _num_words: u32,
}

impl KeyFrameDatabase {
    /// Create a new database
    pub fn new(num_words: u32) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            inverted_index: RwLock::new(HashMap::new()),
            _num_words: num_words,
        }
    }

    /// Add a keyframe to the database
    pub fn add_keyframe(&self, entry: KeyFrameEntry) {
        let mut entries = self.entries.write().unwrap_or_else(|e| e.into_inner());
        let mut inverted = self
            .inverted_index
            .write()
            .unwrap_or_else(|e| e.into_inner());

        // Add to entries
        entries.insert(entry.id, entry.clone());

        // Update inverted index
        for word_id in &entry.word_ids {
            inverted
                .entry(*word_id)
                .or_insert_with(HashSet::new)
                .insert(entry.id);
        }
    }

    /// Remove a keyframe from the database
    pub fn remove_keyframe(&self, keyframe_id: u64) {
        let mut entries = self.entries.write().unwrap_or_else(|e| e.into_inner());
        let mut inverted = self
            .inverted_index
            .write()
            .unwrap_or_else(|e| e.into_inner());

        if let Some(entry) = entries.remove(&keyframe_id) {
            // Remove from inverted index
            for word_id in &entry.word_ids {
                if let Some(ids) = inverted.get_mut(word_id) {
                    ids.remove(&keyframe_id);
                }
            }
        }
    }

    /// Get a keyframe entry
    pub fn get_keyframe(&self, keyframe_id: u64) -> Option<KeyFrameEntry> {
        let entries = self.entries.read().unwrap_or_else(|e| e.into_inner());
        entries.get(&keyframe_id).cloned()
    }

    /// Find candidate keyframes that share words with the query
    ///
    /// # Arguments
    /// * `word_ids` - Word IDs from the query keyframe
    /// * `min_shared_words` - Minimum number of shared words
    ///
    /// # Returns
    /// Vector of (keyframe_id, shared_word_count) sorted by count
    pub fn get_candidates(&self, word_ids: &[u32], min_shared_words: usize) -> Vec<(u64, usize)> {
        let inverted = self
            .inverted_index
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let _entries = self.entries.read().unwrap_or_else(|e| e.into_inner());

        // Count shared words for each keyframe
        let mut shared_counts: HashMap<u64, usize> = HashMap::new();

        for word_id in word_ids {
            if let Some(keyframe_ids) = inverted.get(word_id) {
                for &kf_id in keyframe_ids {
                    *shared_counts.entry(kf_id).or_insert(0) += 1;
                }
            }
        }

        // Filter by minimum shared words and sort
        let mut candidates: Vec<(u64, usize)> = shared_counts
            .into_iter()
            .filter(|(_, count)| *count >= min_shared_words)
            .collect();

        candidates.sort_by(|a, b| b.1.cmp(&a.1));

        candidates
    }

    /// Get keyframes connected to a given keyframe (covisibility)
    pub fn get_connected(&self, keyframe_id: u64) -> Vec<u64> {
        let entries = self.entries.read().unwrap_or_else(|e| e.into_inner());
        entries
            .get(&keyframe_id)
            .map(|e| e.connected_ids.clone())
            .unwrap_or_default()
    }

    /// Detect loops by finding keyframes with high word overlap
    ///
    /// # Arguments
    /// * `word_ids` - Word IDs from current keyframe
    /// * `current_id` - Current keyframe ID (to exclude)
    /// * `min_shared` - Minimum shared words
    /// * `min_score` - Minimum similarity score
    ///
    /// # Returns
    /// Vector of (keyframe_id, score)
    pub fn detect_loop(
        &self,
        word_ids: &[u32],
        bow_vector: &HashMap<u32, f32>,
        current_id: u64,
        min_shared: usize,
        min_score: f32,
    ) -> Vec<(u64, f32)> {
        // Get candidates with shared words
        let candidates = self.get_candidates(word_ids, min_shared);

        let entries = self.entries.read().unwrap_or_else(|e| e.into_inner());
        let mut results = Vec::new();

        for (kf_id, _shared_count) in candidates {
            // Skip current keyframe and immediate neighbors
            if kf_id == current_id {
                continue;
            }

            // Skip if connected (too recent)
            if let Some(current) = entries.get(&current_id) {
                if current.connected_ids.contains(&kf_id) {
                    continue;
                }
            }

            // Compute similarity
            if let Some(candidate) = entries.get(&kf_id) {
                let score = compute_bow_similarity(bow_vector, &candidate.bow_vector);

                if score > min_score {
                    results.push((kf_id, score));
                }
            }
        }

        // Sort by score
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        results
    }

    /// Clear the database
    pub fn clear(&self) {
        let mut entries = self.entries.write().unwrap_or_else(|e| e.into_inner());
        let mut inverted = self
            .inverted_index
            .write()
            .unwrap_or_else(|e| e.into_inner());
        entries.clear();
        inverted.clear();
    }

    /// Get number of keyframes
    pub fn num_keyframes(&self) -> usize {
        let entries = self.entries.read().unwrap_or_else(|e| e.into_inner());
        entries.len()
    }
}

impl Default for KeyFrameDatabase {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Compute similarity between two BoW vectors
fn compute_bow_similarity(bow1: &HashMap<u32, f32>, bow2: &HashMap<u32, f32>) -> f32 {
    if bow1.is_empty() || bow2.is_empty() {
        return 0.0;
    }

    let mut score = 0.0;

    for (word_id, weight1) in bow1 {
        if let Some(weight2) = bow2.get(word_id) {
            score += weight1 * weight2;
        }
    }

    // L2 normalize
    let norm1: f32 = bow1.values().map(|w| w * w).sum::<f32>().sqrt();
    let norm2: f32 = bow2.values().map(|w| w * w).sum::<f32>().sqrt();

    if norm1 > 0.0 && norm2 > 0.0 {
        score / (norm1 * norm2)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_creation() {
        let db = KeyFrameDatabase::new(100);
        assert_eq!(db.num_keyframes(), 0);
    }

    #[test]
    fn test_add_keyframe() {
        let db = KeyFrameDatabase::new(100);

        let entry = KeyFrameEntry {
            id: 1,
            word_ids: vec![1, 2, 3],
            bow_vector: HashMap::new(),
            timestamp: 0.0,
            connected_ids: vec![],
        };

        db.add_keyframe(entry);
        assert_eq!(db.num_keyframes(), 1);
    }

    #[test]
    fn test_get_candidates() {
        let db = KeyFrameDatabase::new(100);

        // Add keyframe 1
        let entry1 = KeyFrameEntry {
            id: 1,
            word_ids: vec![1, 2, 3],
            bow_vector: HashMap::new(),
            timestamp: 0.0,
            connected_ids: vec![],
        };
        db.add_keyframe(entry1);

        // Add keyframe 2
        let entry2 = KeyFrameEntry {
            id: 2,
            word_ids: vec![2, 3, 4],
            bow_vector: HashMap::new(),
            timestamp: 0.0,
            connected_ids: vec![],
        };
        db.add_keyframe(entry2);

        // Query with word 2 and 3
        let candidates = db.get_candidates(&[2, 3], 1);

        assert!(candidates.len() >= 1);
    }

    #[test]
    fn test_remove_keyframe() {
        let db = KeyFrameDatabase::new(100);

        let entry = KeyFrameEntry {
            id: 1,
            word_ids: vec![1, 2, 3],
            bow_vector: HashMap::new(),
            timestamp: 0.0,
            connected_ids: vec![],
        };

        db.add_keyframe(entry);
        assert_eq!(db.num_keyframes(), 1);

        db.remove_keyframe(1);
        assert_eq!(db.num_keyframes(), 0);
    }

    #[test]
    fn test_clear() {
        let db = KeyFrameDatabase::new(100);

        let entry = KeyFrameEntry {
            id: 1,
            word_ids: vec![1, 2, 3],
            bow_vector: HashMap::new(),
            timestamp: 0.0,
            connected_ids: vec![],
        };

        db.add_keyframe(entry);
        db.clear();

        assert_eq!(db.num_keyframes(), 0);
    }
}
