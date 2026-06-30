//! Connected-component splitting of the two-view (covisibility) graph.
//!
//! GLOMAP reconstructs each connected cluster of the view graph independently
//! when the graph is disconnected. This module finds those clusters and provides
//! helpers to remap pairs / frames to component-local indices before running the
//! global mapper on each component.

use crate::types::{ImageFrame, PairGeometry};
use std::collections::HashMap;

/// Options for splitting the view graph into independent reconstruction models.
#[derive(Debug, Clone, Copy)]
pub struct ViewGraphComponentSplittingOptions {
    /// Split the calibrated view graph into connected components and reconstruct
    /// each qualifying component separately.
    pub enabled: bool,
    /// Minimum number of views required to keep a component.
    pub min_component_size: usize,
    /// Maximum number of components to reconstruct (`0` = unlimited). Components
    /// are processed in descending size order.
    pub max_components: usize,
}

impl Default for ViewGraphComponentSplittingOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            min_component_size: 2,
            max_components: 0,
        }
    }
}

/// Statistics from connected-component splitting.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ViewGraphComponentSplittingStats {
    /// Connected components found in the calibrated view graph.
    pub num_components: usize,
    /// Components that passed `min_component_size` and were selected for
    /// reconstruction.
    pub num_selected: usize,
    /// View counts per selected component (descending size order).
    pub selected_component_sizes: Vec<usize>,
    /// Reconstructions successfully produced.
    pub num_reconstructed: usize,
}

/// A connected cluster of views in the covisibility graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewGraphComponent {
    /// Original view indices in ascending order.
    pub views: Vec<usize>,
}

/// Find connected components in the view graph defined by `pairs`.
///
/// Only views incident to at least one non-empty pair edge are included. Each
/// component is returned with views sorted ascending; components are sorted by
/// descending size (then by smallest view index).
pub fn find_view_graph_components(
    num_views: usize,
    pairs: &[PairGeometry],
) -> Vec<ViewGraphComponent> {
    if num_views == 0 {
        return Vec::new();
    }

    let mut uf = UnionFind::with_capacity(num_views);
    for pair in pairs {
        if pair.pose_graph_only || pair.inlier_matches.is_empty() {
            continue;
        }
        if pair.left < num_views && pair.right < num_views && pair.left != pair.right {
            uf.union(pair.left, pair.right);
        }
    }

    let mut grouped: HashMap<usize, Vec<usize>> = HashMap::new();
    for view in 0..num_views {
        if !uf.is_active(view) {
            continue;
        }
        grouped.entry(uf.find(view)).or_default().push(view);
    }

    let mut components: Vec<ViewGraphComponent> = grouped
        .into_values()
        .map(|mut views| {
            views.sort_unstable();
            ViewGraphComponent { views }
        })
        .collect();

    components.sort_by(|a, b| {
        b.views
            .len()
            .cmp(&a.views.len())
            .then_with(|| a.views[0].cmp(&b.views[0]))
    });
    components
}

/// Select components that meet splitting policy.
pub fn select_view_graph_components(
    components: Vec<ViewGraphComponent>,
    options: &ViewGraphComponentSplittingOptions,
) -> (Vec<ViewGraphComponent>, ViewGraphComponentSplittingStats) {
    let num_components = components.len();
    let min_size = options.min_component_size.max(2);
    let mut selected: Vec<ViewGraphComponent> = components
        .into_iter()
        .filter(|component| component.views.len() >= min_size)
        .collect();

    if options.max_components > 0 && selected.len() > options.max_components {
        selected.truncate(options.max_components);
    }

    let stats = ViewGraphComponentSplittingStats {
        num_components,
        num_selected: selected.len(),
        selected_component_sizes: selected.iter().map(|c| c.views.len()).collect(),
        num_reconstructed: 0,
    };
    (selected, stats)
}

/// Resolve which components to reconstruct.
pub fn components_for_reconstruction(
    num_views: usize,
    pairs: &[PairGeometry],
    options: &ViewGraphComponentSplittingOptions,
) -> (Vec<ViewGraphComponent>, ViewGraphComponentSplittingStats) {
    if !options.enabled {
        let stats = ViewGraphComponentSplittingStats {
            num_components: 1,
            num_selected: 1,
            selected_component_sizes: vec![num_views],
            num_reconstructed: 0,
        };
        return (
            vec![ViewGraphComponent {
                views: (0..num_views).collect(),
            }],
            stats,
        );
    }

    let components = find_view_graph_components(num_views, pairs);
    select_view_graph_components(components, options)
}

/// Build a local `old_view -> new_view` map for a component.
pub fn component_view_map(component: &ViewGraphComponent) -> HashMap<usize, usize> {
    component
        .views
        .iter()
        .enumerate()
        .map(|(new_view, &old_view)| (old_view, new_view))
        .collect()
}

/// Remap pair endpoints to component-local view indices.
pub fn remap_pairs_for_component(
    pairs: &[PairGeometry],
    component: &ViewGraphComponent,
) -> Vec<PairGeometry> {
    let view_map = component_view_map(component);
    pairs
        .iter()
        .filter_map(|pair| {
            let left = *view_map.get(&pair.left)?;
            let right = *view_map.get(&pair.right)?;
            Some(PairGeometry {
                left,
                right,
                ..pair.clone()
            })
        })
        .collect()
}

/// Extract frames for a component and assign component-local `ImageFrame::id`s.
pub fn subset_frames_for_component(
    frames: &[ImageFrame],
    component: &ViewGraphComponent,
) -> Vec<ImageFrame> {
    component
        .views
        .iter()
        .enumerate()
        .filter_map(|(new_id, &old_id)| {
            frames.get(old_id).map(|frame| {
                let mut subset = frame.clone();
                subset.id = new_id;
                subset
            })
        })
        .collect()
}

struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
    active: Vec<bool>,
}

impl UnionFind {
    fn with_capacity(num_views: usize) -> Self {
        Self {
            parent: (0..num_views).collect(),
            size: vec![1; num_views],
            active: vec![false; num_views],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        self.active[a] = true;
        self.active[b] = true;
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.size[ra] < self.size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        self.size[ra] += self.size[rb];
    }

    fn is_active(&self, x: usize) -> bool {
        self.active[x]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::COLMAP_TWO_VIEW_CALIBRATED;
    use rustslam::{Match, SE3};

    fn edge(left: usize, right: usize) -> PairGeometry {
        PairGeometry {
            left,
            right,
            two_view_config: COLMAP_TWO_VIEW_CALIBRATED,
            f_matrix: None,
            e_matrix: None,
            h_matrix: None,
            qvec: None,
            tvec: None,
            matches: Vec::new(),
            inlier_matches: vec![Match {
                query_idx: 0,
                train_idx: 0,
                distance: 0.0,
            }],
            relative_pose: SE3::identity(),
            inliers: 1,
            triangulated: 1,
            mean_reprojection_error_px: 0.5,
            rotation_deg: 0.0,
            median_triangulation_angle_deg: 5.0,
            pose_graph_only: false,
        }
    }

    #[test]
    fn finds_two_disconnected_components() {
        let pairs = vec![edge(0, 1), edge(1, 2), edge(3, 4)];
        let components = find_view_graph_components(5, &pairs);
        assert_eq!(components.len(), 2);
        assert_eq!(components[0].views, vec![0, 1, 2]);
        assert_eq!(components[1].views, vec![3, 4]);
    }

    #[test]
    fn select_respects_min_size_and_max_components() {
        let components = find_view_graph_components(
            6,
            &[
                edge(0, 1),
                edge(1, 2),
                edge(3, 4),
                edge(5, 5), // ignored self-edge
            ],
        );
        let options = ViewGraphComponentSplittingOptions {
            enabled: true,
            min_component_size: 3,
            max_components: 1,
        };
        let (selected, stats) = select_view_graph_components(components, &options);
        assert_eq!(stats.num_components, 2);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].views, vec![0, 1, 2]);
    }

    #[test]
    fn remap_pairs_and_frames_to_local_indices() {
        let component = ViewGraphComponent {
            views: vec![2, 5, 7],
        };
        let pairs = vec![edge(2, 5), edge(5, 7), edge(0, 1)];
        let remapped = remap_pairs_for_component(&pairs, &component);
        assert_eq!(remapped.len(), 2);
        assert_eq!(remapped[0].left, 0);
        assert_eq!(remapped[0].right, 1);
        assert_eq!(remapped[1].left, 1);
        assert_eq!(remapped[1].right, 2);

        let frames = (0..8)
            .map(|id| ImageFrame {
                id,
                name: format!("img_{id}.jpg"),
                path: std::path::PathBuf::from(format!("img_{id}.jpg")),
                width: 640,
                height: 480,
                keypoints: Vec::new(),
                descriptors: rustslam::Descriptors::new(),
                sift: crate::sift::SiftFeatures::default(),
                wide_descriptors: crate::wide::WideDescriptors {
                    data: Vec::new(),
                    dim: 0,
                    count: 0,
                },
                strong_feature_indices: Vec::new(),
                colors: Vec::new(),
            })
            .collect::<Vec<_>>();
        let subset = subset_frames_for_component(&frames, &component);
        assert_eq!(subset.len(), 3);
        assert_eq!(subset[0].id, 0);
        assert_eq!(subset[0].name, "img_2.jpg");
        assert_eq!(subset[2].id, 2);
        assert_eq!(subset[2].name, "img_7.jpg");
    }
}
