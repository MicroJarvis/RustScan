//! Global rotation averaging, the foundational stage of a COLMAP/GLOMAP-style
//! global mapper.
//!
//! Given a view graph of pairwise *relative* rotations `R_ij` (mapping camera
//! `i`'s frame into camera `j`'s frame), this estimates a single consistent set
//! of *global* rotations `R_i` (world-to-camera) that best explains all
//! relative measurements. This mirrors GLOMAP's `RotationEstimator`:
//!
//! 1. Initialize the global rotations from a maximum spanning tree of the view
//!    graph (highest-weight edges first), so the starting point is already
//!    chained-consistent on a spanning subgraph.
//! 2. Refine with iteratively reweighted least squares (IRLS) Gauss-Newton on
//!    the `so(3)` tangent space, using a Huber weight so that outlier edges are
//!    progressively down-weighted (GLOMAP uses an `L1`-then-IRLS schedule; the
//!    Huber IRLS here is the robust core of that schedule).
//!
//! Conventions:
//! - A global rotation `R_i` maps world coordinates into camera `i`.
//! - A relative measurement `M_ij` maps camera `i` coordinates into camera `j`,
//!   i.e. `M_ij == R_j * R_i^T` (this is exactly COLMAP's two-view relative
//!   rotation, [`crate::geometry::pose_rotation`] of `PairGeometry::relative_pose`).
//!
//! Gauge: the solution is only determined up to a global left rotation, so view
//! `0` (the spanning-tree root) is held fixed during refinement.
//!
//! Determinism: there is no randomness. The maximum spanning tree breaks weight
//! ties by `(i, j)` index order, so a fixed view graph always yields the same
//! result.

use crate::geometry::pose_rotation;
use crate::types::PairGeometry;
use glam::Quat;
use nalgebra::{DMatrix, DVector, Matrix3, Quaternion, Rotation3, UnitQuaternion, Vector3};

/// A single relative-rotation measurement between two views.
#[derive(Debug, Clone, Copy)]
pub struct RelativeRotation {
    /// Source view index `i`.
    pub i: usize,
    /// Destination view index `j`.
    pub j: usize,
    /// Relative rotation mapping camera `i` into camera `j` (`R_j R_i^T`).
    pub rotation: Quat,
    /// Non-negative edge weight (e.g. inlier count); larger is more trusted.
    pub weight: f64,
}

/// Options controlling [`estimate_global_rotations`].
#[derive(Debug, Clone, Copy)]
pub struct RotationAveragingOptions {
    /// Maximum number of IRLS Gauss-Newton iterations.
    pub max_num_iterations: usize,
    /// Stop once the largest per-view tangent update drops below this angle.
    pub convergence_deg: f64,
    /// Huber robustification threshold on the per-edge residual angle. Residuals
    /// above this are down-weighted by `threshold / residual`.
    pub huber_threshold_deg: f64,
    /// Initialize from the maximum spanning tree (recommended). When false, all
    /// rotations start at identity.
    pub use_spanning_tree_init: bool,
}

impl Default for RotationAveragingOptions {
    fn default() -> Self {
        Self {
            max_num_iterations: 100,
            convergence_deg: 1.0e-3,
            huber_threshold_deg: 5.0,
            use_spanning_tree_init: true,
        }
    }
}

/// Result of global rotation averaging.
#[derive(Debug, Clone)]
pub struct RotationAveragingResult {
    /// Estimated global rotations (world-to-camera), one per view.
    pub global_rotations: Vec<Quat>,
    /// Number of refinement iterations actually performed.
    pub num_iterations: usize,
    /// Largest per-view tangent update (degrees) on the final iteration.
    pub final_max_update_deg: f64,
    /// Mean per-edge residual angle (degrees) after refinement.
    pub mean_residual_deg: f64,
    /// Views reachable from the spanning-tree root through the view graph. Views
    /// outside this set keep their initial (identity) rotation and are not
    /// constrained by the data.
    pub connected: Vec<bool>,
}

/// Build relative-rotation edges from COLMAP-style [`PairGeometry`] pairs.
///
/// Pairs flagged `pose_graph_only` or without inliers are skipped. The edge
/// weight is the inlier count, matching the typical view-graph confidence.
pub fn relative_rotations_from_pairs(pairs: &[PairGeometry]) -> Vec<RelativeRotation> {
    pairs
        .iter()
        .filter(|pair| !pair.pose_graph_only && pair.inliers > 0 && pair.left != pair.right)
        .map(|pair| RelativeRotation {
            i: pair.left,
            j: pair.right,
            rotation: pose_rotation(pair.relative_pose),
            weight: pair.inliers as f64,
        })
        .collect()
}

/// Estimate global rotations from a view graph of relative rotations.
///
/// Returns `None` only when `num_views == 0`.
pub fn estimate_global_rotations(
    num_views: usize,
    edges: &[RelativeRotation],
    options: &RotationAveragingOptions,
) -> Option<RotationAveragingResult> {
    if num_views == 0 {
        return None;
    }

    let clean: Vec<(usize, usize, Rotation3<f64>, f64)> = edges
        .iter()
        .filter(|e| e.i < num_views && e.j < num_views && e.i != e.j && e.weight > 0.0)
        .map(|e| (e.i, e.j, quat_to_rotation(e.rotation), e.weight))
        .collect();

    let mut rotations = vec![Rotation3::<f64>::identity(); num_views];
    let connected = if options.use_spanning_tree_init {
        initialize_from_spanning_tree(num_views, &clean, &mut rotations)
    } else {
        let mut c = vec![false; num_views];
        c[0] = true;
        c
    };

    let huber = options.huber_threshold_deg.to_radians().max(1.0e-6);
    let mut num_iterations = 0usize;
    let mut final_max_update_deg = 0.0f64;
    for _ in 0..options.max_num_iterations {
        num_iterations += 1;
        let Some(updates) = gauss_newton_step(num_views, &clean, &rotations, huber) else {
            break;
        };
        let mut max_update = 0.0f64;
        for (idx, update) in updates.iter().enumerate().skip(1) {
            let norm = update.norm();
            max_update = max_update.max(norm);
            if norm > 0.0 {
                rotations[idx] = Rotation3::from_scaled_axis(*update) * rotations[idx];
            }
        }
        final_max_update_deg = max_update.to_degrees();
        if final_max_update_deg < options.convergence_deg {
            break;
        }
    }

    let mean_residual_deg = mean_residual_deg(&clean, &rotations);
    Some(RotationAveragingResult {
        global_rotations: rotations.iter().map(rotation_to_quat).collect(),
        num_iterations,
        final_max_update_deg,
        mean_residual_deg,
        connected,
    })
}

/// Maximum-spanning-tree initialization. Returns the per-view reachability mask.
fn initialize_from_spanning_tree(
    num_views: usize,
    edges: &[(usize, usize, Rotation3<f64>, f64)],
    rotations: &mut [Rotation3<f64>],
) -> Vec<bool> {
    let mut order: Vec<usize> = (0..edges.len()).collect();
    order.sort_by(|&a, &b| {
        edges[b]
            .3
            .partial_cmp(&edges[a].3)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(edges[a].0.cmp(&edges[b].0))
            .then(edges[a].1.cmp(&edges[b].1))
    });

    let mut parent: Vec<usize> = (0..num_views).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        let mut root = x;
        while parent[root] != root {
            root = parent[root];
        }
        let mut cur = x;
        while parent[cur] != root {
            let next = parent[cur];
            parent[cur] = root;
            cur = next;
        }
        root
    }

    // Spanning-tree adjacency: (neighbor, measurement, measurement_is_forward).
    let mut adjacency: Vec<Vec<(usize, Rotation3<f64>, bool)>> = vec![Vec::new(); num_views];
    for &idx in &order {
        let (i, j, m, _) = edges[idx];
        let ri = find(&mut parent, i);
        let rj = find(&mut parent, j);
        if ri == rj {
            continue;
        }
        parent[ri] = rj;
        adjacency[i].push((j, m, true));
        adjacency[j].push((i, m, false));
    }

    // BFS from view 0, propagating global rotations along tree edges.
    let mut connected = vec![false; num_views];
    let mut queue = std::collections::VecDeque::new();
    rotations[0] = Rotation3::identity();
    connected[0] = true;
    queue.push_back(0usize);
    while let Some(node) = queue.pop_front() {
        for &(neighbor, measurement, forward) in &adjacency[node] {
            if connected[neighbor] {
                continue;
            }
            // measurement M maps cam_i -> cam_j, M = R_j R_i^T.
            rotations[neighbor] = if forward {
                // node == i, neighbor == j: R_j = M R_i.
                measurement * rotations[node]
            } else {
                // node == j, neighbor == i: R_i = M^T R_j.
                measurement.inverse() * rotations[node]
            };
            connected[neighbor] = true;
            queue.push_back(neighbor);
        }
    }
    connected
}

/// One IRLS Gauss-Newton step. Returns per-view left-tangent updates (view 0
/// fixed to zero as the gauge). Linearizes `v_j - P v_i = Log(M P^T)` with
/// `P = R_j R_i^T`, weighting each edge by `weight * huber(residual)`.
fn gauss_newton_step(
    num_views: usize,
    edges: &[(usize, usize, Rotation3<f64>, f64)],
    rotations: &[Rotation3<f64>],
    huber: f64,
) -> Option<Vec<Vector3<f64>>> {
    if num_views < 2 {
        return Some(vec![Vector3::zeros(); num_views]);
    }
    let dim = (num_views - 1) * 3;
    let mut h = DMatrix::<f64>::zeros(dim, dim);
    let mut g = DVector::<f64>::zeros(dim);
    let identity = Matrix3::<f64>::identity();
    let mut used_edges = 0usize;

    for &(i, j, m, weight) in edges {
        let p = rotations[j] * rotations[i].inverse();
        let residual_rot = m * p.inverse();
        let delta = residual_rot.scaled_axis();
        let angle = delta.norm();
        let robust = if angle <= huber { 1.0 } else { huber / angle };
        let w = weight * robust;
        if w <= 0.0 {
            continue;
        }
        used_edges += 1;

        // Jacobian blocks: J_j = +I, J_i = -P (3x3). Accumulate Jᵀ W J and Jᵀ W r.
        let p_mat = *p.matrix();
        let neg_p = -p_mat;
        if j != 0 {
            accumulate_block(&mut h, j, j, &(identity * w));
            accumulate_g(&mut g, j, &(delta * w));
        }
        if i != 0 {
            // (-P)ᵀ (-P) = Pᵀ P = I.
            accumulate_block(&mut h, i, i, &(identity * w));
            accumulate_g(&mut g, i, &(neg_p.transpose() * delta * w));
        }
        if i != 0 && j != 0 {
            // Off-diagonal: J_jᵀ W J_i = I·w·(-P) and its transpose.
            accumulate_block(&mut h, j, i, &(neg_p * w));
            accumulate_block(&mut h, i, j, &(neg_p.transpose() * w));
        }
    }

    if used_edges == 0 {
        return None;
    }
    for d in 0..dim {
        h[(d, d)] += 1.0e-9;
    }

    let solution = match h.clone().cholesky() {
        Some(chol) => chol.solve(&g),
        None => h.lu().solve(&g)?,
    };
    if !solution.iter().all(|v| v.is_finite()) {
        return None;
    }

    let mut updates = vec![Vector3::zeros(); num_views];
    for idx in 1..num_views {
        let base = (idx - 1) * 3;
        updates[idx] = Vector3::new(solution[base], solution[base + 1], solution[base + 2]);
    }
    Some(updates)
}

fn accumulate_block(h: &mut DMatrix<f64>, view_row: usize, view_col: usize, block: &Matrix3<f64>) {
    let r = (view_row - 1) * 3;
    let c = (view_col - 1) * 3;
    for a in 0..3 {
        for b in 0..3 {
            h[(r + a, c + b)] += block[(a, b)];
        }
    }
}

fn accumulate_g(g: &mut DVector<f64>, view: usize, block: &Vector3<f64>) {
    let r = (view - 1) * 3;
    for a in 0..3 {
        g[r + a] += block[a];
    }
}

fn mean_residual_deg(
    edges: &[(usize, usize, Rotation3<f64>, f64)],
    rotations: &[Rotation3<f64>],
) -> f64 {
    if edges.is_empty() {
        return 0.0;
    }
    let mut total = 0.0f64;
    for &(i, j, m, _) in edges {
        let p = rotations[j] * rotations[i].inverse();
        let residual = m * p.inverse();
        // Clamp before acos: near identity the trace can drift just above 3.
        let cos = ((residual.matrix().trace() - 1.0) / 2.0).clamp(-1.0, 1.0);
        total += cos.acos();
    }
    (total / edges.len() as f64).to_degrees()
}

fn quat_to_rotation(q: Quat) -> Rotation3<f64> {
    let q = q.normalize();
    let unit = UnitQuaternion::from_quaternion(Quaternion::new(
        q.w as f64,
        q.x as f64,
        q.y as f64,
        q.z as f64,
    ));
    unit.to_rotation_matrix()
}

fn rotation_to_quat(r: &Rotation3<f64>) -> Quat {
    let q = UnitQuaternion::from_rotation_matrix(r).into_inner();
    Quat::from_xyzw(q.i as f32, q.j as f32, q.k as f32, q.w as f32).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustslam::ColmapMt19937;

    fn unit(rng: &mut ColmapMt19937) -> f32 {
        rng.next_u32() as f32 / u32::MAX as f32
    }

    fn random_quat(rng: &mut ColmapMt19937) -> Quat {
        // Uniform-ish random rotation via random axis-angle.
        let axis = glam::Vec3::new(
            unit(rng) - 0.5,
            unit(rng) - 0.5,
            unit(rng) - 0.5,
        )
        .normalize_or_zero();
        let axis = if axis.length_squared() < 1.0e-6 {
            glam::Vec3::X
        } else {
            axis
        };
        let angle = unit(rng) * std::f32::consts::PI;
        Quat::from_axis_angle(axis, angle)
    }

    fn relative(i: usize, j: usize, gt: &[Quat], weight: f64) -> RelativeRotation {
        // M_ij = R_j R_i^T.
        RelativeRotation {
            i,
            j,
            rotation: (gt[j] * gt[i].inverse()).normalize(),
            weight,
        }
    }

    fn angle_between_deg(a: Quat, b: Quat) -> f64 {
        let r = (a * b.inverse()).normalize();
        (2.0 * r.w.abs().clamp(-1.0, 1.0).acos() as f64).to_degrees()
    }

    #[test]
    fn recovers_global_rotations_from_clean_chain() {
        let mut rng = ColmapMt19937::new(7);
        let n = 8;
        let mut gt = vec![Quat::IDENTITY];
        for _ in 1..n {
            gt.push(random_quat(&mut rng));
        }
        // Chain edges plus a few skip edges, all noise-free.
        let mut edges = Vec::new();
        for i in 0..n - 1 {
            edges.push(relative(i, i + 1, &gt, 100.0));
        }
        edges.push(relative(0, 3, &gt, 50.0));
        edges.push(relative(2, 6, &gt, 50.0));

        let result =
            estimate_global_rotations(n, &edges, &RotationAveragingOptions::default()).unwrap();
        assert!(result.connected.iter().all(|&c| c));
        for idx in 0..n {
            let err = angle_between_deg(result.global_rotations[idx], gt[idx]);
            assert!(err < 0.05, "view {idx} error {err} deg too large");
        }
        assert!(
            result.mean_residual_deg < 0.05,
            "mean residual {} deg too large",
            result.mean_residual_deg
        );
    }

    #[test]
    fn converges_under_small_noise() {
        let mut rng = ColmapMt19937::new(42);
        let n = 12;
        let mut gt = vec![Quat::IDENTITY];
        for _ in 1..n {
            gt.push(random_quat(&mut rng));
        }
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if j - i > 3 {
                    continue;
                }
                let mut e = relative(i, j, &gt, 80.0);
                // Inject ~<1 deg noise.
                let noise_axis = glam::Vec3::new(
                    unit(&mut rng) - 0.5,
                    unit(&mut rng) - 0.5,
                    unit(&mut rng) - 0.5,
                )
                .normalize_or_zero();
                let noise = Quat::from_axis_angle(noise_axis, 0.01 * unit(&mut rng));
                e.rotation = (noise * e.rotation).normalize();
                edges.push(e);
            }
        }
        let result =
            estimate_global_rotations(n, &edges, &RotationAveragingOptions::default()).unwrap();
        for idx in 0..n {
            let err = angle_between_deg(result.global_rotations[idx], gt[idx]);
            assert!(err < 2.0, "view {idx} error {err} deg too large under noise");
        }
    }

    #[test]
    fn is_robust_to_a_single_outlier_edge() {
        let mut rng = ColmapMt19937::new(123);
        let n = 8;
        let mut gt = vec![Quat::IDENTITY];
        for _ in 1..n {
            gt.push(random_quat(&mut rng));
        }
        let mut edges = Vec::new();
        for i in 0..n - 1 {
            edges.push(relative(i, i + 1, &gt, 100.0));
        }
        // Redundant clean edges so the outlier is outvoted.
        for i in 0..n - 2 {
            edges.push(relative(i, i + 2, &gt, 100.0));
        }
        // A grossly wrong measurement on one edge.
        edges.push(RelativeRotation {
            i: 2,
            j: 3,
            rotation: random_quat(&mut rng),
            weight: 100.0,
        });

        let result =
            estimate_global_rotations(n, &edges, &RotationAveragingOptions::default()).unwrap();
        for idx in 0..n {
            let err = angle_between_deg(result.global_rotations[idx], gt[idx]);
            assert!(err < 3.0, "view {idx} error {err} deg; outlier not suppressed");
        }
    }

    #[test]
    fn marks_disconnected_views() {
        let gt = vec![Quat::IDENTITY; 4];
        // Only connect views 0-1; views 2,3 are isolated.
        let edges = vec![relative(0, 1, &gt, 10.0)];
        let result =
            estimate_global_rotations(4, &edges, &RotationAveragingOptions::default()).unwrap();
        assert!(result.connected[0]);
        assert!(result.connected[1]);
        assert!(!result.connected[2]);
        assert!(!result.connected[3]);
    }

    #[test]
    fn empty_graph_returns_identities() {
        let result =
            estimate_global_rotations(3, &[], &RotationAveragingOptions::default()).unwrap();
        assert_eq!(result.global_rotations.len(), 3);
        for q in result.global_rotations {
            assert!(angle_between_deg(q, Quat::IDENTITY) < 1.0e-6);
        }
    }
}
