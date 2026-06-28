//! Image retrieval via a classical vocabulary tree (Nister-Stewenius).
//!
//! COLMAP's modern `retrieval::VisualIndex` uses a FAISS-backed inverted index
//! with Hamming embedding and a vote-and-verify spatial step. FAISS is a heavy
//! C++ dependency that is out of scope for this Rust port (the GPU strategy is
//! `wgpu`-only, and FAISS IVF acceleration is tracked separately). This module
//! provides the functional, pure-Rust core of vocabulary-tree retrieval that
//! COLMAP exposes through `vocab_tree_matcher` pairing:
//!
//! 1. A hierarchical k-means vocabulary tree trained on SIFT descriptors.
//! 2. Descriptor quantization to leaf visual words.
//! 3. A TF-IDF inverted index over indexed images.
//! 4. Cosine (L2-normalized TF-IDF) Bag-of-Words query scoring that returns a
//!    ranked list of the most similar images.
//!
//! Determinism: clustering uses the shared COLMAP-compatible MT19937
//! (`rustslam::ColmapMt19937`) so a fixed `random_seed` yields a reproducible
//! tree, matching the project-wide deterministic-RNG policy.

use std::collections::HashMap;
use std::path::Path;

use rustslam::ColmapMt19937;
use serde::{Deserialize, Serialize};

/// Default SIFT descriptor dimensionality.
pub const SIFT_DESC_DIM: usize = 128;

/// Options controlling vocabulary-tree construction.
///
/// `branching_factor` and `num_levels` together bound the number of leaf
/// visual words at `branching_factor^num_levels`. `num_visual_words` is an
/// upper bound; the realized number of words can be smaller when branches run
/// out of training descriptors (matching COLMAP's "actual number of visual
/// words might be less" note on `BuildOptions::num_visual_words`).
#[derive(Debug, Clone, Copy)]
pub struct VocabTreeBuildOptions {
    /// Number of child clusters per internal node (k in hierarchical k-means).
    pub branching_factor: usize,
    /// Maximum tree depth (number of clustering levels).
    pub num_levels: usize,
    /// Upper bound on the number of leaf visual words.
    pub num_visual_words: usize,
    /// Lloyd's k-means iterations per node.
    pub num_iterations: usize,
    /// Redo clustering this many times per node, keep the lowest-inertia run.
    pub num_rounds: usize,
    /// Deterministic clustering seed (>= 0). Negative selects a fixed default.
    pub random_seed: i64,
}

impl Default for VocabTreeBuildOptions {
    fn default() -> Self {
        // COLMAP `BuildOptions` defaults aim at 256*256 visual words with 100
        // iterations and 3 rounds. A branching factor of 10 over 5 levels
        // reaches 100k leaves; the realized count is capped by training data.
        Self {
            branching_factor: 10,
            num_levels: 5,
            num_visual_words: 256 * 256,
            num_iterations: 100,
            num_rounds: 3,
            random_seed: 0,
        }
    }
}

impl VocabTreeBuildOptions {
    fn seed(&self) -> u64 {
        if self.random_seed < 0 {
            0
        } else {
            self.random_seed as u64
        }
    }
}

/// A node in the vocabulary tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Node {
    /// Centroid of this node within its parent (root centroid is unused).
    centroid: Vec<f32>,
    /// Indices of child nodes (empty for a leaf).
    children: Vec<usize>,
    /// Assigned visual word id, present only for leaves.
    word_id: Option<usize>,
}

/// A trained hierarchical k-means vocabulary tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabTree {
    nodes: Vec<Node>,
    dim: usize,
    num_words: usize,
}

impl VocabTree {
    /// Descriptor dimensionality the tree was trained with.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Serialize the trained tree to `path` as JSON for reuse across runs
    /// (COLMAP persists its vocabulary tree to disk via `Write`/`Read`).
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_vec(self).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    /// Load a vocabulary tree previously written by [`VocabTree::save`].
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(std::io::Error::other)
    }

    /// Total number of leaf visual words.
    pub fn num_visual_words(&self) -> usize {
        self.num_words
    }

    /// Train a vocabulary tree from L2-normalized `descriptors`, supplied as a
    /// row-major flat buffer of `count = descriptors.len() / dim` rows.
    pub fn build(options: &VocabTreeBuildOptions, descriptors: &[f32], dim: usize) -> Option<Self> {
        if dim == 0 || descriptors.is_empty() || descriptors.len() % dim != 0 {
            return None;
        }
        if options.branching_factor < 2 || options.num_levels == 0 {
            return None;
        }
        let count = descriptors.len() / dim;
        let mut points: Vec<Vec<f32>> = Vec::with_capacity(count);
        for i in 0..count {
            points.push(l2_normalized(&descriptors[i * dim..(i + 1) * dim]));
        }

        let mut tree = VocabTree {
            nodes: Vec::new(),
            dim,
            num_words: 0,
        };
        let mut rng = ColmapMt19937::new(options.seed());
        let all_indices: Vec<usize> = (0..count).collect();
        let root_centroid = vec![0.0f32; dim];
        tree.build_node(&all_indices, &points, 1, options, &mut rng, root_centroid);
        Some(tree)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_node(
        &mut self,
        indices: &[usize],
        points: &[Vec<f32>],
        level: usize,
        options: &VocabTreeBuildOptions,
        rng: &mut ColmapMt19937,
        centroid: Vec<f32>,
    ) -> usize {
        let node_idx = self.nodes.len();
        self.nodes.push(Node {
            centroid,
            children: Vec::new(),
            word_id: None,
        });

        let make_leaf = level >= options.num_levels
            || indices.len() <= options.branching_factor
            || self.num_words >= options.num_visual_words;
        if make_leaf {
            let word_id = self.num_words;
            self.num_words += 1;
            self.nodes[node_idx].word_id = Some(word_id);
            return node_idx;
        }

        let k = options.branching_factor.min(indices.len());
        let (labels, centroids) = kmeans(indices, points, k, self.dim, options, rng);

        // Group point indices by assigned cluster.
        let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); centroids.len()];
        for (pos, &point_idx) in indices.iter().enumerate() {
            clusters[labels[pos]].push(point_idx);
        }

        let mut children = Vec::new();
        for (cluster_idx, cluster_indices) in clusters.into_iter().enumerate() {
            if cluster_indices.is_empty() {
                continue;
            }
            let child = self.build_node(
                &cluster_indices,
                points,
                level + 1,
                options,
                rng,
                centroids[cluster_idx].clone(),
            );
            children.push(child);
        }

        // Degenerate split (all points collapsed into one child): make a leaf
        // so the tree still terminates with a usable visual word.
        if children.len() <= 1 {
            // Roll back any single child that became its own subtree-leaf: keep
            // it, but if no children at all, turn this node into a leaf.
            if children.is_empty() {
                let word_id = self.num_words;
                self.num_words += 1;
                self.nodes[node_idx].word_id = Some(word_id);
                return node_idx;
            }
        }
        self.nodes[node_idx].children = children;
        node_idx
    }

    /// Quantize one descriptor (length `dim`, will be L2-normalized) to a leaf
    /// visual word id. Returns `None` only for an empty/degenerate tree.
    pub fn quantize(&self, descriptor: &[f32]) -> Option<usize> {
        if descriptor.len() != self.dim || self.nodes.is_empty() {
            return None;
        }
        let query = l2_normalized(descriptor);
        let mut node_idx = 0usize;
        loop {
            let node = &self.nodes[node_idx];
            if let Some(word_id) = node.word_id {
                return Some(word_id);
            }
            if node.children.is_empty() {
                return None;
            }
            let mut best_child = node.children[0];
            let mut best_dist = f32::MAX;
            for &child_idx in &node.children {
                let dist = squared_distance(&query, &self.nodes[child_idx].centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_child = child_idx;
                }
            }
            node_idx = best_child;
        }
    }
}

/// One retrieved image and its similarity score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageScore {
    pub image_id: i32,
    pub score: f32,
}

/// A TF-IDF inverted index over a trained vocabulary tree.
pub struct VisualIndex {
    tree: VocabTree,
    /// Raw per-image term frequencies (word id -> count).
    images: Vec<(i32, HashMap<usize, f32>)>,
    /// Document frequency per visual word.
    doc_freq: Vec<usize>,
    /// IDF weight per visual word (filled by `prepare`).
    idf: Vec<f32>,
    /// L2-normalized TF-IDF vectors per image (filled by `prepare`).
    prepared: Vec<(i32, HashMap<usize, f32>)>,
}

impl VisualIndex {
    pub fn new(tree: VocabTree) -> Self {
        let num_words = tree.num_visual_words();
        Self {
            tree,
            images: Vec::new(),
            doc_freq: vec![0; num_words],
            idf: Vec::new(),
            prepared: Vec::new(),
        }
    }

    pub fn tree(&self) -> &VocabTree {
        &self.tree
    }

    pub fn num_images(&self) -> usize {
        self.images.len()
    }

    pub fn is_image_indexed(&self, image_id: i32) -> bool {
        self.images.iter().any(|(id, _)| *id == image_id)
    }

    /// Quantize image descriptors into a sparse term-frequency vector.
    fn term_frequencies(&self, descriptors: &[f32], dim: usize) -> HashMap<usize, f32> {
        let mut tf: HashMap<usize, f32> = HashMap::new();
        if dim != self.tree.dim() || dim == 0 || descriptors.len() % dim != 0 {
            return tf;
        }
        let count = descriptors.len() / dim;
        for i in 0..count {
            if let Some(word) = self.tree.quantize(&descriptors[i * dim..(i + 1) * dim]) {
                *tf.entry(word).or_insert(0.0) += 1.0;
            }
        }
        tf
    }

    /// Add an image's descriptors to the index. Must be called before
    /// [`VisualIndex::prepare`]. Re-adding the same `image_id` is rejected.
    pub fn add_image(&mut self, image_id: i32, descriptors: &[f32], dim: usize) -> bool {
        if self.is_image_indexed(image_id) {
            return false;
        }
        let tf = self.term_frequencies(descriptors, dim);
        if tf.is_empty() {
            return false;
        }
        for &word in tf.keys() {
            self.doc_freq[word] += 1;
        }
        self.images.push((image_id, tf));
        self.prepared.clear();
        true
    }

    /// Compute IDF weights and normalized per-image TF-IDF vectors. Call once
    /// after all images are added and before querying.
    pub fn prepare(&mut self) {
        let num_images = self.images.len();
        let num_words = self.tree.num_visual_words();
        self.idf = vec![0.0; num_words];
        for word in 0..num_words {
            let df = self.doc_freq[word];
            if df > 0 {
                self.idf[word] = ((num_images as f32) / (df as f32)).ln().max(0.0);
            }
        }
        self.prepared = self
            .images
            .iter()
            .map(|(image_id, tf)| (*image_id, self.weighted_normalized(tf)))
            .collect();
    }

    /// Build an L2-normalized TF-IDF vector from raw term frequencies.
    fn weighted_normalized(&self, tf: &HashMap<usize, f32>) -> HashMap<usize, f32> {
        let mut vec: HashMap<usize, f32> = HashMap::with_capacity(tf.len());
        let mut norm_sq = 0.0f32;
        for (&word, &freq) in tf {
            let w = freq * self.idf.get(word).copied().unwrap_or(0.0);
            if w != 0.0 {
                vec.insert(word, w);
                norm_sq += w * w;
            }
        }
        if norm_sq > 0.0 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            for w in vec.values_mut() {
                *w *= inv_norm;
            }
        }
        vec
    }

    /// Query the index with an image's descriptors and return up to
    /// `max_num_images` most similar indexed images, ranked by descending
    /// cosine similarity. A non-positive `max_num_images` returns all images.
    pub fn query(&self, descriptors: &[f32], dim: usize, max_num_images: i64) -> Vec<ImageScore> {
        if self.prepared.is_empty() {
            return Vec::new();
        }
        let tf = self.term_frequencies(descriptors, dim);
        if tf.is_empty() {
            return Vec::new();
        }
        let query_vec = self.weighted_normalized(&tf);
        if query_vec.is_empty() {
            return Vec::new();
        }

        let mut scores: Vec<ImageScore> = self
            .prepared
            .iter()
            .map(|(image_id, doc_vec)| {
                // Iterate the smaller vector for the sparse dot product.
                let (small, large) = if query_vec.len() <= doc_vec.len() {
                    (&query_vec, doc_vec)
                } else {
                    (doc_vec, &query_vec)
                };
                let mut score = 0.0f32;
                for (&word, &w) in small {
                    if let Some(&other) = large.get(&word) {
                        score += w * other;
                    }
                }
                ImageScore {
                    image_id: *image_id,
                    score,
                }
            })
            .collect();

        scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.image_id.cmp(&b.image_id))
        });

        if max_num_images > 0 && (max_num_images as usize) < scores.len() {
            scores.truncate(max_num_images as usize);
        }
        scores
    }
}

/// Convert COLMAP `u8` SIFT descriptors (row-major, `dim` columns) to a
/// row-major `f32` buffer. The vocabulary tree L2-normalizes internally.
pub fn descriptors_u8_to_f32(data: &[u8]) -> Vec<f32> {
    data.iter().map(|&b| b as f32).collect()
}

/// Options for vocabulary-tree candidate-pair generation, mirroring COLMAP's
/// `VocabTreeMatchingOptions`.
#[derive(Debug, Clone, Copy)]
pub struct VocabTreePairOptions {
    /// Number of most similar images to retrieve per query image
    /// (COLMAP `num_images`, default 100).
    pub num_images: usize,
}

impl Default for VocabTreePairOptions {
    fn default() -> Self {
        Self { num_images: 100 }
    }
}

/// Generate candidate image pairs from a prepared [`VisualIndex`].
///
/// For every `(image_id, descriptors)` query, the top `num_images` most similar
/// indexed images are retrieved and emitted as unordered `(min, max)` image-id
/// pairs (self-matches excluded, duplicates removed), matching COLMAP's
/// `VocabTreePairGenerator` behavior.
pub fn generate_vocab_tree_pairs(
    index: &VisualIndex,
    queries: &[(i32, &[f32])],
    dim: usize,
    options: &VocabTreePairOptions,
) -> Vec<(i32, i32)> {
    use std::collections::BTreeSet;
    let mut pairs: BTreeSet<(i32, i32)> = BTreeSet::new();
    // Retrieve one extra so that dropping the self-match still leaves
    // `num_images` candidates available.
    let retrieve = options.num_images.saturating_add(1) as i64;
    for &(query_id, descriptors) in queries {
        for result in index.query(descriptors, dim, retrieve) {
            if result.image_id == query_id {
                continue;
            }
            let pair = if query_id < result.image_id {
                (query_id, result.image_id)
            } else {
                (result.image_id, query_id)
            };
            pairs.insert(pair);
        }
    }
    pairs.into_iter().collect()
}

/// End-to-end vocabulary-tree pairing: train a tree on the union of all image
/// descriptors, index every image, and return candidate `(min, max)` image-id
/// pairs. This is the high-level entry equivalent to running COLMAP's
/// `vocab_tree_matcher` pair generation on an in-memory descriptor set.
pub fn build_vocab_tree_pairs(
    images: &[(i32, Vec<f32>)],
    dim: usize,
    build_options: &VocabTreeBuildOptions,
    pair_options: &VocabTreePairOptions,
) -> Vec<(i32, i32)> {
    if images.len() < 2 || dim == 0 {
        return Vec::new();
    }
    let mut training: Vec<f32> = Vec::new();
    for (_, descriptors) in images {
        if descriptors.len() % dim == 0 {
            training.extend_from_slice(descriptors);
        }
    }
    let Some(tree) = VocabTree::build(build_options, &training, dim) else {
        return Vec::new();
    };
    let mut index = VisualIndex::new(tree);
    for (image_id, descriptors) in images {
        index.add_image(*image_id, descriptors, dim);
    }
    index.prepare();
    let queries: Vec<(i32, &[f32])> = images
        .iter()
        .map(|(id, descriptors)| (*id, descriptors.as_slice()))
        .collect();
    generate_vocab_tree_pairs(&index, &queries, dim, pair_options)
}

fn l2_normalized(descriptor: &[f32]) -> Vec<f32> {
    let norm_sq: f32 = descriptor.iter().map(|&v| v * v).sum();
    if norm_sq > 0.0 {
        let inv_norm = 1.0 / norm_sq.sqrt();
        descriptor.iter().map(|&v| v * inv_norm).collect()
    } else {
        descriptor.to_vec()
    }
}

#[inline]
fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Seeded k-means (k-means++ init + Lloyd iterations) over a subset of points.
///
/// Returns `(labels, centroids)` where `labels[i]` is the cluster of
/// `points[indices[i]]` and `centroids[c]` is the L2-normalized centroid of
/// cluster `c`. Repeats `num_rounds` times and keeps the lowest-inertia run.
fn kmeans(
    indices: &[usize],
    points: &[Vec<f32>],
    k: usize,
    dim: usize,
    options: &VocabTreeBuildOptions,
    rng: &mut ColmapMt19937,
) -> (Vec<usize>, Vec<Vec<f32>>) {
    let n = indices.len();
    let k = k.min(n).max(1);

    let mut best_labels: Vec<usize> = vec![0; n];
    let mut best_centroids: Vec<Vec<f32>> = Vec::new();
    let mut best_inertia = f32::MAX;

    for _ in 0..options.num_rounds.max(1) {
        let mut centroids = kmeans_pp_init(indices, points, k, rng);
        let mut labels = vec![0usize; n];

        for _ in 0..options.num_iterations.max(1) {
            // Assignment step.
            let mut changed = false;
            for (pos, &point_idx) in indices.iter().enumerate() {
                let point = &points[point_idx];
                let mut best_c = 0usize;
                let mut best_d = f32::MAX;
                for (c, centroid) in centroids.iter().enumerate() {
                    let d = squared_distance(point, centroid);
                    if d < best_d {
                        best_d = d;
                        best_c = c;
                    }
                }
                if labels[pos] != best_c {
                    labels[pos] = best_c;
                    changed = true;
                }
            }

            // Update step.
            let mut sums = vec![vec![0.0f32; dim]; centroids.len()];
            let mut counts = vec![0usize; centroids.len()];
            for (pos, &point_idx) in indices.iter().enumerate() {
                let c = labels[pos];
                counts[c] += 1;
                let point = &points[point_idx];
                for d in 0..dim {
                    sums[c][d] += point[d];
                }
            }
            for c in 0..centroids.len() {
                if counts[c] > 0 {
                    let inv = 1.0 / counts[c] as f32;
                    for d in 0..dim {
                        sums[c][d] *= inv;
                    }
                    centroids[c] = l2_normalized(&sums[c]);
                }
                // Empty cluster: keep its previous centroid.
            }

            if !changed {
                break;
            }
        }

        // Inertia of this run.
        let mut inertia = 0.0f32;
        for (pos, &point_idx) in indices.iter().enumerate() {
            inertia += squared_distance(&points[point_idx], &centroids[labels[pos]]);
        }
        if inertia < best_inertia {
            best_inertia = inertia;
            best_labels = labels;
            best_centroids = centroids;
        }
    }

    (best_labels, best_centroids)
}

/// k-means++ seeding using the COLMAP MT19937 RNG.
fn kmeans_pp_init(
    indices: &[usize],
    points: &[Vec<f32>],
    k: usize,
    rng: &mut ColmapMt19937,
) -> Vec<Vec<f32>> {
    let n = indices.len();
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);

    let first = rng.uniform_usize(0, n - 1);
    centroids.push(points[indices[first]].clone());

    let mut dist_sq = vec![f32::MAX; n];
    while centroids.len() < k {
        let last = centroids.last().expect("non-empty");
        let mut total = 0.0f64;
        for (pos, &point_idx) in indices.iter().enumerate() {
            let d = squared_distance(&points[point_idx], last);
            if d < dist_sq[pos] {
                dist_sq[pos] = d;
            }
            total += dist_sq[pos] as f64;
        }
        if total <= 0.0 {
            // All remaining points coincide with chosen centers; pad with a
            // deterministic pick so we still return k centroids.
            let pick = rng.uniform_usize(0, n - 1);
            centroids.push(points[indices[pick]].clone());
            continue;
        }
        // Sample proportional to squared distance using a uniform draw.
        let threshold = (rng.uniform_u32(0, u32::MAX) as f64 / u32::MAX as f64) * total;
        let mut acc = 0.0f64;
        let mut chosen = indices[n - 1];
        for (pos, &point_idx) in indices.iter().enumerate() {
            acc += dist_sq[pos] as f64;
            if acc >= threshold {
                chosen = point_idx;
                break;
            }
        }
        centroids.push(points[chosen].clone());
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build four well-separated descriptor blobs in a low dimension.
    fn make_blobs(dim: usize, per_blob: usize) -> (Vec<f32>, Vec<usize>) {
        let centers = [
            (0usize, 10.0f32),
            (1, 10.0),
            (2, 10.0),
            (3, 10.0),
        ];
        let mut data = Vec::new();
        let mut labels = Vec::new();
        for (label, (axis, mag)) in centers.iter().enumerate() {
            for j in 0..per_blob {
                let mut d = vec![0.1f32; dim];
                d[*axis] = *mag + (j as f32) * 0.01;
                data.extend_from_slice(&d);
                labels.push(label);
            }
        }
        (data, labels)
    }

    #[test]
    fn build_quantizes_same_blob_to_same_word() {
        let dim = 8;
        let (data, labels) = make_blobs(dim, 12);
        let options = VocabTreeBuildOptions {
            branching_factor: 4,
            num_levels: 3,
            num_visual_words: 64,
            num_iterations: 25,
            num_rounds: 3,
            random_seed: 0,
        };
        let tree = VocabTree::build(&options, &data, dim).expect("tree builds");
        assert!(tree.num_visual_words() >= 4);

        let count = data.len() / dim;
        let mut word_of: Vec<usize> = Vec::new();
        for i in 0..count {
            word_of.push(tree.quantize(&data[i * dim..(i + 1) * dim]).expect("word"));
        }
        // A tight blob may be subdivided across several leaf words, but words
        // must never be shared across different blobs: the visual vocabularies
        // of distinct blobs are disjoint.
        let mut blob_word_sets: Vec<std::collections::HashSet<usize>> = vec![Default::default(); 4];
        for (&word, &label) in word_of.iter().zip(labels.iter()) {
            blob_word_sets[label].insert(word);
        }
        for a in 0..4 {
            for b in (a + 1)..4 {
                assert!(
                    blob_word_sets[a].is_disjoint(&blob_word_sets[b]),
                    "blobs {a} and {b} share words: {:?} vs {:?}",
                    blob_word_sets[a],
                    blob_word_sets[b]
                );
            }
        }
    }

    #[test]
    fn save_load_roundtrips_quantization() {
        let dim = 8;
        let (data, _) = make_blobs(dim, 12);
        let options = VocabTreeBuildOptions {
            branching_factor: 4,
            num_levels: 3,
            num_visual_words: 64,
            num_iterations: 25,
            num_rounds: 3,
            random_seed: 0,
        };
        let tree = VocabTree::build(&options, &data, dim).expect("tree builds");

        let mut path = std::env::temp_dir();
        path.push(format!("rustsfm_vocab_tree_{}.json", std::process::id()));
        tree.save(&path).expect("save tree");
        let loaded = VocabTree::load(&path).expect("load tree");
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.dim(), tree.dim());
        assert_eq!(loaded.num_visual_words(), tree.num_visual_words());
        let count = data.len() / dim;
        for i in 0..count {
            let d = &data[i * dim..(i + 1) * dim];
            assert_eq!(loaded.quantize(d), tree.quantize(d));
        }
    }

    #[test]
    fn build_is_deterministic_for_fixed_seed() {
        let dim = 8;
        let (data, _) = make_blobs(dim, 10);
        let options = VocabTreeBuildOptions {
            branching_factor: 3,
            num_levels: 3,
            num_visual_words: 64,
            num_iterations: 20,
            num_rounds: 2,
            random_seed: 7,
        };
        let tree_a = VocabTree::build(&options, &data, dim).unwrap();
        let tree_b = VocabTree::build(&options, &data, dim).unwrap();
        assert_eq!(tree_a.num_visual_words(), tree_b.num_visual_words());
        let count = data.len() / dim;
        for i in 0..count {
            let d = &data[i * dim..(i + 1) * dim];
            assert_eq!(tree_a.quantize(d), tree_b.quantize(d));
        }
    }

    #[test]
    fn query_ranks_self_and_similar_images_highest() {
        let dim = 8;
        let (data, _) = make_blobs(dim, 12);
        let options = VocabTreeBuildOptions {
            branching_factor: 4,
            num_levels: 3,
            num_visual_words: 64,
            num_iterations: 25,
            num_rounds: 3,
            random_seed: 0,
        };
        let tree = VocabTree::build(&options, &data, dim).unwrap();

        // Image 1: blobs 0 + 1. Image 2: blobs 2 + 3. Image 3: blobs 0 + 1
        // (same content as image 1, so it should be the closest match).
        let per_blob = 12;
        let blob = |b: usize| -> Vec<f32> {
            let start = b * per_blob * dim;
            data[start..start + per_blob * dim].to_vec()
        };
        let mut img1 = blob(0);
        img1.extend(blob(1));
        let mut img2 = blob(2);
        img2.extend(blob(3));
        let mut img3 = blob(0);
        img3.extend(blob(1));

        let mut index = VisualIndex::new(tree);
        assert!(index.add_image(1, &img1, dim));
        assert!(index.add_image(2, &img2, dim));
        assert!(index.add_image(3, &img3, dim));
        assert!(!index.add_image(1, &img1, dim), "duplicate id rejected");
        index.prepare();

        let results = index.query(&img1, dim, -1);
        assert_eq!(results.len(), 3);
        // The identical-content images (1 and 3) must rank above image 2.
        let top_two: std::collections::HashSet<i32> =
            results.iter().take(2).map(|s| s.image_id).collect();
        assert!(top_two.contains(&1) && top_two.contains(&3), "{results:?}");
        assert!(results[0].score >= results[2].score);
        assert_eq!(results.last().unwrap().image_id, 2, "{results:?}");
    }

    #[test]
    fn query_before_prepare_returns_empty() {
        let dim = 8;
        let (data, _) = make_blobs(dim, 6);
        let options = VocabTreeBuildOptions {
            branching_factor: 3,
            num_levels: 2,
            num_visual_words: 32,
            num_iterations: 10,
            num_rounds: 1,
            random_seed: 0,
        };
        let tree = VocabTree::build(&options, &data, dim).unwrap();
        let mut index = VisualIndex::new(tree);
        index.add_image(1, &data, dim);
        // No prepare() call yet.
        assert!(index.query(&data, dim, -1).is_empty());
    }

    #[test]
    fn build_rejects_invalid_input() {
        let options = VocabTreeBuildOptions::default();
        assert!(VocabTree::build(&options, &[], 128).is_none());
        assert!(VocabTree::build(&options, &[1.0, 2.0, 3.0], 0).is_none());
        // Length not a multiple of dim.
        assert!(VocabTree::build(&options, &[1.0, 2.0, 3.0], 2).is_none());
    }

    #[test]
    fn max_num_images_truncates_results() {
        let dim = 8;
        let (data, _) = make_blobs(dim, 12);
        let options = VocabTreeBuildOptions {
            branching_factor: 4,
            num_levels: 3,
            num_visual_words: 64,
            num_iterations: 25,
            num_rounds: 2,
            random_seed: 0,
        };
        let tree = VocabTree::build(&options, &data, dim).unwrap();
        let mut index = VisualIndex::new(tree);
        let per_blob = 12;
        let blob = |b: usize| data[b * per_blob * dim..(b + 1) * per_blob * dim].to_vec();
        index.add_image(1, &blob(0), dim);
        index.add_image(2, &blob(1), dim);
        index.add_image(3, &blob(2), dim);
        index.prepare();
        let results = index.query(&blob(0), dim, 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn descriptors_u8_to_f32_preserves_values() {
        let u8s = [0u8, 1, 255, 128];
        let f = descriptors_u8_to_f32(&u8s);
        assert_eq!(f, vec![0.0, 1.0, 255.0, 128.0]);
    }

    #[test]
    fn build_vocab_tree_pairs_links_images_with_shared_content() {
        let dim = 8;
        let (data, _) = make_blobs(dim, 12);
        let per_blob = 12;
        let blob = |b: usize| data[b * per_blob * dim..(b + 1) * per_blob * dim].to_vec();

        // Images 1 and 3 share blobs {0,1}; image 2 is blobs {2,3}; image 4 is
        // blobs {0,1} again. With num_images=1, each query links to its single
        // most-similar image, so the shared-content images pair up.
        let mut img1 = blob(0);
        img1.extend(blob(1));
        let img2 = {
            let mut v = blob(2);
            v.extend(blob(3));
            v
        };
        let mut img4 = blob(0);
        img4.extend(blob(1));

        let images = vec![(1, img1), (2, img2), (4, img4)];
        let build_options = VocabTreeBuildOptions {
            branching_factor: 4,
            num_levels: 3,
            num_visual_words: 64,
            num_iterations: 25,
            num_rounds: 2,
            random_seed: 0,
        };
        let pair_options = VocabTreePairOptions { num_images: 1 };
        let pairs = build_vocab_tree_pairs(&images, dim, &build_options, &pair_options);

        // The identical-content images 1 and 4 must be paired; no self pairs.
        assert!(pairs.contains(&(1, 4)), "{pairs:?}");
        assert!(pairs.iter().all(|(a, b)| a < b), "{pairs:?}");
    }

    #[test]
    fn generate_vocab_tree_pairs_excludes_self_and_dedups() {
        let dim = 8;
        let (data, _) = make_blobs(dim, 12);
        let per_blob = 12;
        let blob = |b: usize| data[b * per_blob * dim..(b + 1) * per_blob * dim].to_vec();
        let options = VocabTreeBuildOptions {
            branching_factor: 4,
            num_levels: 3,
            num_visual_words: 64,
            num_iterations: 25,
            num_rounds: 2,
            random_seed: 0,
        };
        let tree = VocabTree::build(&options, &data, dim).unwrap();
        let mut index = VisualIndex::new(tree);
        let img_a = blob(0);
        let img_b = blob(1);
        index.add_image(10, &img_a, dim);
        index.add_image(20, &img_b, dim);
        index.prepare();

        let queries = vec![(10, img_a.as_slice()), (20, img_b.as_slice())];
        let pairs = generate_vocab_tree_pairs(&index, &queries, dim, &VocabTreePairOptions::default());
        // Two images -> exactly one unordered pair, no self entries.
        assert_eq!(pairs, vec![(10, 20)]);
    }

    #[test]
    fn build_vocab_tree_pairs_handles_too_few_images() {
        let dim = 8;
        let (data, _) = make_blobs(dim, 4);
        let images = vec![(1, data)];
        let pairs = build_vocab_tree_pairs(
            &images,
            dim,
            &VocabTreeBuildOptions::default(),
            &VocabTreePairOptions::default(),
        );
        assert!(pairs.is_empty());
    }
}
