//! Bag of Words (BoW) vocabulary for loop closing
//!
//! This module implements a visual vocabulary for efficient loop detection.
//! Based on ORB-SLAM3's BoW approach.

use crate::features::base::Descriptors;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// A visual word (cluster center)
#[derive(Debug, Clone)]
pub struct Word {
    /// Word ID
    pub id: u32,
    /// Word weight (IDF)
    pub weight: f32,
    /// Cluster center descriptor
    pub descriptor: Vec<u8>,
}

/// A node in the vocabulary tree
#[derive(Debug, Clone)]
pub struct Node {
    /// Node ID
    pub id: u32,
    /// Word ID (if leaf node)
    pub word_id: Option<u32>,
    /// Parent node ID
    pub parent_id: Option<u32>,
    /// Children node IDs
    pub children: Vec<u32>,
    /// Descriptor (cluster center)
    pub descriptor: Vec<u8>,
    /// Accumulated weight from children
    pub weight: f32,
}

/// Bag of Words vocabulary
pub struct Vocabulary {
    /// Vocabulary words
    words: Vec<Word>,
    /// Vocabulary tree nodes
    nodes: HashMap<u32, Node>,
    /// Number of images used for training
    num_images: u32,
    /// Number of words in vocabulary
    pub k: u32,
    /// Branching factor
    pub branching: u32,
    /// Depth of the tree
    pub depth: u32,
}

impl Vocabulary {
    /// Create a new empty vocabulary
    pub fn new(branching: u32, depth: u32) -> Self {
        Self {
            words: Vec::new(),
            nodes: HashMap::new(),
            num_images: 0,
            k: 0,
            branching,
            depth,
        }
    }

    /// Get number of words
    pub fn num_words(&self) -> usize {
        self.words.len()
    }

    /// Get word by ID
    pub fn get_word(&self, word_id: u32) -> Option<&Word> {
        self.words.get(word_id as usize)
    }

    /// Transform a descriptor to word IDs and weights
    ///
    /// Returns: (word_ids, word_weights)
    pub fn transform(&self, descriptors: &Descriptors) -> (Vec<u32>, Vec<f32>) {
        let mut word_ids = Vec::with_capacity(descriptors.count as usize);
        let mut word_weights = Vec::with_capacity(descriptors.count as usize);

        for i in 0..descriptors.count {
            if let Some(desc) = descriptors.get(i) {
                let word_id = self.find_word(desc);
                let weight = self.words[word_id as usize].weight;
                word_ids.push(word_id);
                word_weights.push(weight);
            }
        }

        (word_ids, word_weights)
    }

    /// Transform to BoW vector (word_id -> weight)
    pub fn transform_bow(&self, descriptors: &Descriptors) -> HashMap<u32, f32> {
        let mut bow = HashMap::new();

        for i in 0..descriptors.count {
            if let Some(desc) = descriptors.get(i) {
                let word_id = self.find_word(desc);
                let weight = self.words[word_id as usize].weight;
                *bow.entry(word_id).or_insert(0.0) += weight;
            }
        }

        bow
    }

    /// Find the word ID for a descriptor (approximate)
    fn find_word(&self, descriptor: &[u8]) -> u32 {
        // Start from root and descend the tree
        let mut node_id = 0u32; // Root is always 0

        for _ in 0..self.depth {
            let node = &self.nodes[&node_id];

            if node.children.is_empty() {
                break;
            }

            // Find best child
            let mut best_child = node.children[0];
            let mut best_dist = u32::MAX;

            for &child_id in &node.children {
                if let Some(child) = self.nodes.get(&child_id) {
                    let dist = hamming_distance(descriptor, &child.descriptor);
                    if dist < best_dist {
                        best_dist = dist;
                        best_child = child_id;
                    }
                }
            }

            node_id = best_child;
        }

        // Return word ID at leaf
        self.nodes
            .get(&node_id)
            .and_then(|n| n.word_id)
            .unwrap_or(0)
    }

    /// Compute similarity between two BoW vectors
    pub fn similarity(&self, bow1: &HashMap<u32, f32>, bow2: &HashMap<u32, f32>) -> f32 {
        if bow1.is_empty() || bow2.is_empty() {
            return 0.0;
        }

        let mut score = 0.0;

        // Find common words
        for (word_id, weight1) in bow1 {
            if let Some(weight2) = bow2.get(word_id) {
                score += weight1 * weight2;
            }
        }

        // Normalize
        let norm1: f32 = bow1.values().map(|w| w * w).sum::<f32>().sqrt();
        let norm2: f32 = bow2.values().map(|w| w * w).sum::<f32>().sqrt();

        if norm1 > 0.0 && norm2 > 0.0 {
            score / (norm1 * norm2)
        } else {
            0.0
        }
    }

    /// Save vocabulary to file
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut file = File::create(path)?;

        // Header
        writeln!(file, "Vocabulary {}", self.branching)?;
        writeln!(file, "Depth {}", self.depth)?;
        writeln!(file, "NumImages {}", self.num_images)?;
        writeln!(file, "NumWords {}", self.words.len())?;

        // Words
        for word in &self.words {
            writeln!(
                file,
                "W {} {:.6} {:?}",
                word.id, word.weight, word.descriptor
            )?;
        }

        // Nodes
        for (id, node) in &self.nodes {
            let word_id = node.word_id.map(|w| w.to_string()).unwrap_or_default();
            writeln!(
                file,
                "N {} {} {:?} {:.6}",
                id, word_id, node.descriptor, node.weight
            )?;
        }

        Ok(())
    }

    /// Load vocabulary from file
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut branching = 0u32;
        let mut depth = 0u32;
        let mut num_images = 0u32;
        let mut num_words = 0usize;
        let mut words = Vec::new();
        let nodes = HashMap::new();

        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.splitn(3, ' ').collect();

            match parts[0] {
                "Vocabulary" => branching = parts[1].parse().unwrap_or(0),
                "Depth" => depth = parts[1].parse().unwrap_or(0),
                "NumImages" => num_images = parts[1].parse().unwrap_or(0),
                "NumWords" => num_words = parts[1].parse().unwrap_or(0),
                "W" => {
                    let id: u32 = parts[1].parse().unwrap_or(0);
                    let weight: f32 = parts[2].parse().unwrap_or(0.0);
                    // Parse descriptor...
                    words.push(Word {
                        id,
                        weight,
                        descriptor: Vec::new(),
                    });
                }
                "N" => {
                    // Parse node...
                }
                _ => {}
            }
        }

        Ok(Self {
            words,
            nodes,
            num_images,
            k: num_words as u32,
            branching,
            depth,
        })
    }
}

/// Compute Hamming distance between two descriptors
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x ^ *y).count_ones())
        .sum()
}

/// K-means clustering for vocabulary creation
pub fn kmeans(descriptors: &[Vec<u8>], k: usize, max_iterations: usize) -> Vec<Vec<u8>> {
    if descriptors.is_empty() || k == 0 {
        return Vec::new();
    }

    // Initialize centers randomly
    let mut centers: Vec<Vec<u8>> = descriptors.iter().take(k).cloned().collect();

    // If not enough descriptors, pad
    while centers.len() < k {
        if let Some(d) = descriptors.first() {
            centers.push(d.clone());
        } else {
            break;
        }
    }

    for _ in 0..max_iterations {
        // Assign each descriptor to nearest center
        let mut clusters: Vec<Vec<&[u8]>> = vec![Vec::new(); k];

        for desc in descriptors {
            let mut best_idx = 0;
            let mut best_dist = u32::MAX;

            for (idx, center) in centers.iter().enumerate() {
                let dist = hamming_distance(desc, center);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = idx;
                }
            }

            clusters[best_idx].push(desc);
        }

        // Update centers
        let mut converged = true;

        for (idx, cluster) in clusters.iter().enumerate() {
            if cluster.is_empty() {
                continue;
            }

            // Compute new center (majority vote for each bit)
            let mut new_center = vec![0u8; centers[idx].len()];

            for bit_idx in 0..new_center.len() {
                let mut ones = 0;
                for desc in cluster {
                    if bit_idx < desc.len() {
                        ones += desc[bit_idx] as usize;
                    }
                }
                // Majority vote
                new_center[bit_idx] = if ones > cluster.len() / 2 { 255 } else { 0 };
            }

            if new_center != centers[idx] {
                converged = false;
            }
            centers[idx] = new_center;
        }

        if converged {
            break;
        }
    }

    centers
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocabulary_creation() {
        let vocab = Vocabulary::new(10, 6);
        assert_eq!(vocab.num_words(), 0);
    }

    #[test]
    fn test_hamming_distance() {
        let a = vec![0b11110000, 0b00001111];
        let b = vec![0b11111111, 0b00000000];
        assert_eq!(hamming_distance(&a, &b), 8);
    }

    #[test]
    fn test_similarity_empty() {
        let vocab = Vocabulary::new(10, 6);
        let bow1: HashMap<u32, f32> = HashMap::new();
        let bow2 = HashMap::new();
        assert_eq!(vocab.similarity(&bow1, &bow2), 0.0);
    }
}
