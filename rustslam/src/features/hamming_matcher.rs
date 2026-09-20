//! Hamming matcher for binary descriptors (e.g., ORB).

use crate::features::base::{Descriptors, FeatureError, FeatureMatcher, Match};
use std::collections::{HashMap, HashSet};

/// Hamming matcher using bounded multi-probe candidate search.
pub struct HammingMatcher {
    k: usize,
    ratio_threshold: Option<f64>,
}

impl HammingMatcher {
    pub fn new(k: usize) -> Self {
        Self {
            k: k.max(1),
            ratio_threshold: None,
        }
    }

    pub fn with_ratio_threshold(mut self, ratio_threshold: f64) -> Self {
        self.ratio_threshold = Some(ratio_threshold);
        self
    }

    fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
        static LUT: [u8; 256] = build_popcount_lut();
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| LUT[(x ^ y) as usize] as u32)
            .sum()
    }

    fn bucket_key(desc: &[u8]) -> u16 {
        let b0 = desc.get(0).copied().unwrap_or(0) as u16;
        let b1 = desc.get(1).copied().unwrap_or(0) as u16;
        (b0 << 8) | b1
    }
}

impl FeatureMatcher for HammingMatcher {
    fn match_descriptors(
        &self,
        query: &Descriptors,
        train: &Descriptors,
    ) -> Result<Vec<Match>, FeatureError> {
        if query.is_empty() || train.is_empty() {
            return Ok(Vec::new());
        }
        if query.size != train.size || query.size == 0 {
            return Ok(Vec::new());
        }

        let ratio_threshold = self.ratio_threshold;
        let mut matches = Vec::new();

        let mut buckets_16: HashMap<u16, Vec<usize>> = HashMap::new();
        let mut buckets_8: HashMap<u8, Vec<usize>> = HashMap::new();
        for t_idx in 0..train.count {
            if let Some(desc) = train.get(t_idx) {
                buckets_16
                    .entry(Self::bucket_key(desc))
                    .or_default()
                    .push(t_idx);
                buckets_8.entry(desc[0]).or_default().push(t_idx);
            }
        }

        for q_idx in 0..query.count {
            let q_desc = query.get(q_idx).unwrap_or(&[]);
            if q_desc.is_empty() {
                continue;
            }

            let key = Self::bucket_key(q_desc);
            let key8 = q_desc[0];
            let min_candidates = self.k.saturating_mul(16).max(32);
            let max_candidates = min_candidates
                .saturating_mul(8)
                .min(train.count.max(min_candidates));
            let mut candidate_indices: Vec<usize> = Vec::new();
            let mut seen: HashSet<usize> = HashSet::with_capacity(max_candidates.saturating_mul(2));

            if let Some(primary) = buckets_16.get(&key) {
                add_group_candidates(primary, &mut candidate_indices, &mut seen, max_candidates);
            }

            if candidate_indices.len() < min_candidates {
                for bit in 0..16 {
                    let neighbor = key ^ (1u16 << bit);
                    if let Some(group) = buckets_16.get(&neighbor) {
                        add_group_candidates(
                            group,
                            &mut candidate_indices,
                            &mut seen,
                            max_candidates,
                        );
                    }
                    if candidate_indices.len() >= min_candidates
                        || candidate_indices.len() >= max_candidates
                    {
                        break;
                    }
                }
            }

            if candidate_indices.len() < min_candidates {
                if let Some(group) = buckets_8.get(&key8) {
                    add_group_candidates(group, &mut candidate_indices, &mut seen, max_candidates);
                }
            }

            if candidate_indices.len() < min_candidates {
                for bit in 0..8 {
                    let neighbor = key8 ^ (1u8 << bit);
                    if let Some(group) = buckets_8.get(&neighbor) {
                        add_group_candidates(
                            group,
                            &mut candidate_indices,
                            &mut seen,
                            max_candidates,
                        );
                    }
                    if candidate_indices.len() >= min_candidates
                        || candidate_indices.len() >= max_candidates
                    {
                        break;
                    }
                }
            }

            if candidate_indices.len() < self.k.max(2) && train.count > 0 {
                let target = min_candidates.min(train.count);
                let mut probe = (key as usize) % train.count;
                let step = 131usize;
                for _ in 0..target.saturating_mul(2) {
                    if seen.insert(probe) {
                        candidate_indices.push(probe);
                        if candidate_indices.len() >= target
                            || candidate_indices.len() >= max_candidates
                        {
                            break;
                        }
                    }
                    probe = (probe + step) % train.count;
                }
            }

            let mut best: Option<(u32, usize)> = None;
            let mut second: Option<(u32, usize)> = None;
            for &t_idx in &candidate_indices {
                if let Some(t_desc) = train.get(t_idx) {
                    let dist = Self::hamming_distance(q_desc, t_desc);
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
}

const fn build_popcount_lut() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut i = 0u16;
    while i < 256 {
        table[i as usize] = i.count_ones() as u8;
        i += 1;
    }
    table
}

fn add_group_candidates(
    group: &[usize],
    candidate_indices: &mut Vec<usize>,
    seen: &mut HashSet<usize>,
    max_candidates: usize,
) {
    for &idx in group {
        if candidate_indices.len() >= max_candidates {
            break;
        }
        if seen.insert(idx) {
            candidate_indices.push(idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::base::ORB_DESCRIPTOR_SIZE;

    #[test]
    fn test_hamming_matcher_ratio() {
        let matcher = HammingMatcher::new(2).with_ratio_threshold(0.8);

        let mut query = Descriptors::with_capacity(1, ORB_DESCRIPTOR_SIZE);
        query.data.fill(0b1010_1010);

        let mut train = Descriptors::with_capacity(2, ORB_DESCRIPTOR_SIZE);
        train.data.fill(0b1010_1010);

        let matches = matcher.match_descriptors(&query, &train).unwrap();
        assert!(matches.is_empty());
    }
}
