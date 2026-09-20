//! Track establishment for a GLOMAP-style global mapper.
//!
//! After feature matching and two-view verification, the view graph contains
//! pairwise feature correspondences. *Track establishment* fuses those pairwise
//! links into multi-view feature tracks: a track is a maximal set of feature
//! observations across images that are all transitively matched and that
//! therefore (should) correspond to a single 3D point. This mirrors GLOMAP's
//! `TrackEstablisher` and COLMAP's connected-component track building over the
//! correspondence graph.
//!
//! Algorithm:
//! 1. Union-find over feature nodes `(image, feature)`, unioning every inlier
//!    correspondence.
//! 2. Group nodes by connected component into candidate tracks.
//! 3. Reject *inconsistent* tracks — those that observe the same image more than
//!    once (two distinct features in one image cannot be the same 3D point).
//! 4. Apply `min_track_length` / `max_track_length` filtering.
//!
//! Determinism: feature node ids are assigned in first-encounter order while
//! scanning matches; tracks and their observations are sorted before returning,
//! so a fixed input always yields identical output.

use std::collections::HashMap;

/// A single feature observation: feature `feature` in image `image`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FeatureNode {
    pub image: usize,
    pub feature: usize,
}

impl FeatureNode {
    pub fn new(image: usize, feature: usize) -> Self {
        Self { image, feature }
    }
}

/// Inlier feature correspondences between an ordered image pair.
#[derive(Debug, Clone)]
pub struct PairwiseMatches {
    pub image_i: usize,
    pub image_j: usize,
    /// `(feature_in_i, feature_in_j)` inlier correspondences.
    pub matches: Vec<(usize, usize)>,
}

/// A multi-view feature track: observations sorted by `(image, feature)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Track {
    pub observations: Vec<FeatureNode>,
}

impl Track {
    /// Number of observations (track length).
    pub fn len(&self) -> usize {
        self.observations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }
}

/// Options for [`establish_tracks`].
#[derive(Debug, Clone, Copy)]
pub struct TrackEstablishmentOptions {
    /// Minimum observations a track must have to be kept.
    pub min_track_length: usize,
    /// Maximum track length to keep (`0` means unlimited). Overly long tracks are
    /// usually the result of erroneous merges and are commonly capped.
    pub max_track_length: usize,
}

impl Default for TrackEstablishmentOptions {
    fn default() -> Self {
        Self {
            min_track_length: 2,
            max_track_length: 0,
        }
    }
}

/// Statistics describing a track-establishment run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TrackEstablishmentStats {
    /// Connected components found before any filtering.
    pub num_components: usize,
    /// Components rejected because an image appeared more than once.
    pub num_inconsistent: usize,
    /// Components rejected by `min_track_length` / `max_track_length`.
    pub num_filtered_by_length: usize,
    /// Tracks returned.
    pub num_tracks: usize,
    /// Total observations across the returned tracks.
    pub num_observations: usize,
}

/// Establish multi-view tracks from pairwise inlier matches.
///
/// Returns the kept tracks (sorted deterministically) and run statistics.
pub fn establish_tracks(
    matches: &[PairwiseMatches],
    options: &TrackEstablishmentOptions,
) -> (Vec<Track>, TrackEstablishmentStats) {
    let mut node_ids: HashMap<FeatureNode, usize> = HashMap::new();
    let mut nodes: Vec<FeatureNode> = Vec::new();
    let mut uf = UnionFind::new();

    let intern = |node: FeatureNode,
                  node_ids: &mut HashMap<FeatureNode, usize>,
                  nodes: &mut Vec<FeatureNode>,
                  uf: &mut UnionFind|
     -> usize {
        if let Some(&id) = node_ids.get(&node) {
            id
        } else {
            let id = nodes.len();
            node_ids.insert(node, id);
            nodes.push(node);
            uf.push();
            id
        }
    };

    for pair in matches {
        for &(fi, fj) in &pair.matches {
            let a = intern(
                FeatureNode::new(pair.image_i, fi),
                &mut node_ids,
                &mut nodes,
                &mut uf,
            );
            let b = intern(
                FeatureNode::new(pair.image_j, fj),
                &mut node_ids,
                &mut nodes,
                &mut uf,
            );
            uf.union(a, b);
        }
    }

    // Group nodes by connected-component root.
    let mut components: HashMap<usize, Vec<FeatureNode>> = HashMap::new();
    for (id, node) in nodes.iter().enumerate() {
        components.entry(uf.find(id)).or_default().push(*node);
    }

    let mut stats = TrackEstablishmentStats {
        num_components: components.len(),
        ..Default::default()
    };

    let mut tracks: Vec<Track> = Vec::new();
    for (_, mut observations) in components {
        observations.sort();

        // Consistency: reject tracks that observe an image more than once.
        let mut consistent = true;
        for window in observations.windows(2) {
            if window[0].image == window[1].image {
                consistent = false;
                break;
            }
        }
        if !consistent {
            stats.num_inconsistent += 1;
            continue;
        }

        let len = observations.len();
        if len < options.min_track_length
            || (options.max_track_length > 0 && len > options.max_track_length)
        {
            stats.num_filtered_by_length += 1;
            continue;
        }

        tracks.push(Track { observations });
    }

    // Deterministic output order: by first observation, then length.
    tracks.sort_by(|a, b| {
        a.observations[0]
            .cmp(&b.observations[0])
            .then_with(|| a.observations.len().cmp(&b.observations.len()))
    });

    stats.num_tracks = tracks.len();
    stats.num_observations = tracks.iter().map(Track::len).sum();
    (tracks, stats)
}

/// Minimal union-find with path compression and union by size.
struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new() -> Self {
        Self {
            parent: Vec::new(),
            size: Vec::new(),
        }
    }

    fn push(&mut self) {
        let id = self.parent.len();
        self.parent.push(id);
        self.size.push(1);
    }

    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut cur = x;
        while self.parent[cur] != root {
            let next = self.parent[cur];
            self.parent[cur] = root;
            cur = next;
        }
        root
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        let (big, small) = if self.size[ra] >= self.size[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small] = big;
        self.size[big] += self.size[small];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(i: usize, j: usize, matches: &[(usize, usize)]) -> PairwiseMatches {
        PairwiseMatches {
            image_i: i,
            image_j: j,
            matches: matches.to_vec(),
        }
    }

    #[test]
    fn merges_transitive_matches_into_one_track() {
        // (0,a)-(1,b) and (1,b)-(2,c) -> single length-3 track.
        let matches = vec![pair(0, 1, &[(10, 20)]), pair(1, 2, &[(20, 30)])];
        let (tracks, stats) = establish_tracks(&matches, &TrackEstablishmentOptions::default());
        assert_eq!(tracks.len(), 1);
        assert_eq!(
            tracks[0].observations,
            vec![
                FeatureNode::new(0, 10),
                FeatureNode::new(1, 20),
                FeatureNode::new(2, 30),
            ]
        );
        assert_eq!(stats.num_components, 1);
        assert_eq!(stats.num_tracks, 1);
        assert_eq!(stats.num_observations, 3);
    }

    #[test]
    fn separates_independent_tracks() {
        let matches = vec![pair(0, 1, &[(0, 0), (1, 1)]), pair(1, 2, &[(0, 0)])];
        let (tracks, stats) = establish_tracks(&matches, &TrackEstablishmentOptions::default());
        // Track A: (0,0)-(1,0)-(2,0); Track B: (0,1)-(1,1).
        assert_eq!(tracks.len(), 2);
        assert_eq!(stats.num_components, 2);
        assert_eq!(tracks[0].observations[0], FeatureNode::new(0, 0));
        assert_eq!(tracks[0].len(), 3);
        assert_eq!(tracks[1].observations[0], FeatureNode::new(0, 1));
        assert_eq!(tracks[1].len(), 2);
    }

    #[test]
    fn rejects_tracks_with_same_image_conflict() {
        // Two features in image 0 both match the same feature in image 1, which
        // merges them into one component observing image 0 twice -> inconsistent.
        let matches = vec![pair(0, 1, &[(5, 7)]), pair(0, 1, &[(6, 7)])];
        let (tracks, stats) = establish_tracks(&matches, &TrackEstablishmentOptions::default());
        assert!(tracks.is_empty());
        assert_eq!(stats.num_components, 1);
        assert_eq!(stats.num_inconsistent, 1);
        assert_eq!(stats.num_tracks, 0);
    }

    #[test]
    fn applies_min_and_max_length_filters() {
        let matches = vec![
            // Length-3 track.
            pair(0, 1, &[(0, 0)]),
            pair(1, 2, &[(0, 0)]),
            // Length-2 track.
            pair(0, 1, &[(9, 9)]),
        ];
        let min3 = TrackEstablishmentOptions {
            min_track_length: 3,
            max_track_length: 0,
        };
        let (tracks, stats) = establish_tracks(&matches, &min3);
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].len(), 3);
        assert_eq!(stats.num_filtered_by_length, 1);

        let max2 = TrackEstablishmentOptions {
            min_track_length: 2,
            max_track_length: 2,
        };
        let (tracks, _) = establish_tracks(&matches, &max2);
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].len(), 2);
    }

    #[test]
    fn is_deterministic_across_runs() {
        let matches = vec![
            pair(2, 5, &[(3, 4)]),
            pair(0, 1, &[(7, 8)]),
            pair(1, 5, &[(8, 4)]),
            pair(0, 2, &[(7, 3)]),
        ];
        let a = establish_tracks(&matches, &TrackEstablishmentOptions::default());
        let b = establish_tracks(&matches, &TrackEstablishmentOptions::default());
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
    }

    #[test]
    fn empty_input_yields_no_tracks() {
        let (tracks, stats) = establish_tracks(&[], &TrackEstablishmentOptions::default());
        assert!(tracks.is_empty());
        assert_eq!(stats, TrackEstablishmentStats::default());
    }
}
