//! COLMAP-style image pair generation for local/feature matching.

use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchingPairStrategy {
    /// Match every unordered image pair.
    Exhaustive,
    /// Match each image against nearby neighbors in sorted order.
    Sequential {
        overlap: usize,
        quadratic_overlap: bool,
        loop_detection: bool,
        loop_detection_period: usize,
    },
    /// Match each image against the next `window` images only.
    LocalWindow { window: usize },
    /// Match each image against its most visually similar images, found via a
    /// vocabulary tree (COLMAP `vocab_tree_matcher`). Pair generation is
    /// descriptor-aware and handled in the matching path, not by
    /// [`generate_matching_pairs`], which has no descriptor access.
    VocabTree { num_images: usize },
}

impl Default for MatchingPairStrategy {
    fn default() -> Self {
        Self::Sequential {
            overlap: 10,
            quadratic_overlap: true,
            loop_detection: false,
            loop_detection_period: 10,
        }
    }
}

impl MatchingPairStrategy {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "exhaustive" => Some(Self::Exhaustive),
            "sequential" => Some(Self::default()),
            "local-window" | "local_window" | "localwindow" => {
                Some(Self::LocalWindow { window: 3 })
            }
            "vocab-tree" | "vocab_tree" | "vocabtree" => {
                Some(Self::VocabTree { num_images: 100 })
            }
            _ => None,
        }
    }
}

pub fn generate_matching_pairs(
    frame_count: usize,
    strategy: MatchingPairStrategy,
) -> Vec<(usize, usize)> {
    if frame_count < 2 {
        return Vec::new();
    }
    let mut pairs = BTreeSet::new();
    match strategy {
        MatchingPairStrategy::Exhaustive => {
            for left in 0..frame_count {
                for right in left + 1..frame_count {
                    pairs.insert((left, right));
                }
            }
        }
        MatchingPairStrategy::Sequential {
            overlap,
            quadratic_overlap,
            loop_detection,
            loop_detection_period,
        } => {
            let overlap = overlap.max(1);
            for left in 0..frame_count {
                for offset in 1..=overlap {
                    let right = left + offset;
                    if right < frame_count {
                        pairs.insert((left, right));
                    }
                }
                if quadratic_overlap {
                    let mut offset = overlap;
                    while left + offset < frame_count {
                        pairs.insert((left, left + offset));
                        offset = offset.saturating_mul(2);
                        if offset == 0 {
                            break;
                        }
                    }
                }
                if loop_detection && loop_detection_period > 0 {
                    let right = left + loop_detection_period;
                    if right < frame_count {
                        pairs.insert((left, right));
                    }
                }
            }
        }
        MatchingPairStrategy::LocalWindow { window } => {
            if window == 0 {
                return Vec::new();
            }
            for offset in 1..=window.min(frame_count - 1) {
                for left in 0..frame_count - offset {
                    pairs.insert((left, left + offset));
                }
            }
        }
        // Vocabulary-tree pairing needs descriptors and is produced by the
        // descriptor-aware matching path (see `feature_matching_db`).
        MatchingPairStrategy::VocabTree { .. } => return Vec::new(),
    }
    pairs.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhaustive_generates_all_unordered_pairs() {
        assert_eq!(
            generate_matching_pairs(4, MatchingPairStrategy::Exhaustive),
            vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        );
    }

    #[test]
    fn from_name_parses_vocab_tree_aliases() {
        for alias in ["vocab-tree", "vocab_tree", "VocabTree"] {
            assert_eq!(
                MatchingPairStrategy::from_name(alias),
                Some(MatchingPairStrategy::VocabTree { num_images: 100 })
            );
        }
    }

    #[test]
    fn vocab_tree_yields_no_index_only_pairs() {
        // Vocab-tree pairing is descriptor-aware; the index-only generator must
        // defer by returning no pairs.
        assert!(
            generate_matching_pairs(8, MatchingPairStrategy::VocabTree { num_images: 4 })
                .is_empty()
        );
    }

    #[test]
    fn sequential_overlap_matches_colmap_neighbor_window() {
        let pairs = generate_matching_pairs(
            5,
            MatchingPairStrategy::Sequential {
                overlap: 2,
                quadratic_overlap: false,
                loop_detection: false,
                loop_detection_period: 10,
            },
        );
        assert_eq!(
            pairs,
            vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
        );
    }

    #[test]
    fn sequential_quadratic_overlap_adds_exponential_offsets() {
        let pairs = generate_matching_pairs(
            20,
            MatchingPairStrategy::Sequential {
                overlap: 5,
                quadratic_overlap: true,
                loop_detection: false,
                loop_detection_period: 10,
            },
        );
        assert!(pairs.contains(&(0, 5)));
        assert!(pairs.contains(&(0, 10)));
        assert!(!pairs.contains(&(0, 15)));
    }

    #[test]
    fn sequential_overlap_two_quadratic_matches_flowers2_colmap_pair_count() {
        let pairs = generate_matching_pairs(
            24,
            MatchingPairStrategy::Sequential {
                overlap: 2,
                quadratic_overlap: true,
                loop_detection: false,
                loop_detection_period: 10,
            },
        );
        assert_eq!(pairs.len(), 89);
    }

    #[test]
    fn sequential_loop_detection_adds_periodic_pair() {
        let pairs = generate_matching_pairs(
            12,
            MatchingPairStrategy::Sequential {
                overlap: 1,
                quadratic_overlap: false,
                loop_detection: true,
                loop_detection_period: 10,
            },
        );
        assert!(pairs.contains(&(0, 1)));
        assert!(pairs.contains(&(0, 10)));
    }

    #[test]
    fn local_window_matches_legacy_local_pair_candidates() {
        assert_eq!(
            generate_matching_pairs(5, MatchingPairStrategy::LocalWindow { window: 2 }),
            vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
        );
    }
}
