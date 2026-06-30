//! Global positioning (translation averaging), the second stage of a
//! COLMAP/GLOMAP-style global mapper.
//!
//! Once global rotations are known (see [`crate::rotation_averaging`]), this
//! recovers camera *centers* `c_i` in a common world frame from the pairwise
//! relative-translation directions of the view graph. It follows GLOMAP's
//! `GlobalPositioner` family (BATA — "Bundle-Adjusted Translation Averaging"),
//! solving for camera centers and per-edge positive depths that best satisfy
//!
//! ```text
//! minimize  sum_ij  w_ij * || (c_j - c_i) - d_ij * dhat_ij ||^2
//! ```
//!
//! where `dhat_ij` is the unit world-frame baseline direction implied by the
//! relative pose and `d_ij >= 0` is an auxiliary per-edge scale. The problem is
//! solved by alternating:
//!
//! 1. **Centers**: with depths `d_ij` fixed, the residual is linear in the
//!    centers and decouples per axis into a weighted graph-Laplacian solve
//!    (view `0` is held at the origin as the translation gauge).
//! 2. **Depths**: with centers fixed, each `d_ij = max(0, dhat_ij · (c_j - c_i))`.
//! 3. **Robust reweighting**: a Huber weight on the per-edge residual
//!    progressively down-weights outlier edges (IRLS).
//! 4. **Scale gauge**: the configuration is rescaled to unit RMS center radius,
//!    fixing the otherwise-free global scale.
//!
//! World-frame baseline direction: for a relative pose mapping camera `i` into
//! camera `j` as `x_j = R_ij x_i + t_ij`, the camera centers satisfy
//! `c_j - c_i = -R_j^T t_ij`, where `R_j` is the *global* rotation (world→cam)
//! of view `j`. Hence `dhat_ij = normalize(-(R_j^T t_ij))`.
//!
//! Determinism: no randomness; tie handling and iteration order are fixed.

use crate::types::PairGeometry;
use glam::{Quat, Vec3};
use nalgebra::DMatrix;

/// A single relative-translation measurement between two views.
#[derive(Debug, Clone, Copy)]
pub struct RelativeTranslation {
    /// Source view index `i`.
    pub i: usize,
    /// Destination view index `j`.
    pub j: usize,
    /// Relative-pose translation `t_ij` (`x_j = R_ij x_i + t_ij`); scale-free.
    pub translation: Vec3,
    /// Non-negative edge weight (e.g. inlier count); larger is more trusted.
    pub weight: f64,
}

/// Options controlling [`estimate_global_positions`].
#[derive(Debug, Clone, Copy)]
pub struct GlobalPositioningOptions {
    /// Maximum number of alternating (center/depth) iterations.
    pub max_num_iterations: usize,
    /// Stop once the largest per-view center change (in normalized units) drops
    /// below this value.
    pub convergence: f64,
    /// Huber robustification threshold on the per-edge residual norm (normalized
    /// units). Residuals above this are down-weighted by `threshold / residual`.
    pub huber_threshold: f64,
}

impl Default for GlobalPositioningOptions {
    fn default() -> Self {
        Self {
            max_num_iterations: 100,
            convergence: 1.0e-5,
            huber_threshold: 0.1,
        }
    }
}

/// Result of global positioning.
#[derive(Debug, Clone)]
pub struct GlobalPositioningResult {
    /// Estimated camera centers in world coordinates (view `0` at the origin,
    /// configuration scaled to unit RMS radius).
    pub centers: Vec<Vec3>,
    /// Number of alternating iterations actually performed.
    pub num_iterations: usize,
    /// Mean per-edge residual norm (normalized units) after the final iteration.
    pub mean_residual: f64,
    /// Views reachable from view `0` through the view graph. Unreachable views
    /// keep the origin and are not constrained by the data.
    pub connected: Vec<bool>,
}

/// Build relative-translation edges from COLMAP-style [`PairGeometry`] pairs.
///
/// Pairs flagged `pose_graph_only`, with no inliers, or with a degenerate
/// translation are skipped. The edge weight is the inlier count.
pub fn relative_translations_from_pairs(pairs: &[PairGeometry]) -> Vec<RelativeTranslation> {
    pairs
        .iter()
        .filter(|pair| !pair.pose_graph_only && pair.inliers > 0 && pair.left != pair.right)
        .filter_map(|pair| {
            let t = Vec3::from_array(pair.relative_pose.translation());
            (t.length_squared() > 1.0e-12).then_some(RelativeTranslation {
                i: pair.left,
                j: pair.right,
                translation: t,
                weight: pair.inliers as f64,
            })
        })
        .collect()
}

/// Estimate global camera centers from global rotations and relative
/// translations.
///
/// `global_rotations[k]` is the world→camera rotation of view `k` (typically the
/// output of [`crate::rotation_averaging::estimate_global_rotations`]). Returns
/// `None` if there are fewer than two views or no usable edges.
pub fn estimate_global_positions(
    global_rotations: &[Quat],
    edges: &[RelativeTranslation],
    options: &GlobalPositioningOptions,
) -> Option<GlobalPositioningResult> {
    let num_views = global_rotations.len();
    if num_views < 2 {
        return None;
    }

    // Precompute world-frame unit baseline directions dhat_ij = -(R_j^T t_ij).
    let mut clean: Vec<(usize, usize, Vec3, f64)> = Vec::with_capacity(edges.len());
    for e in edges {
        if e.i >= num_views || e.j >= num_views || e.i == e.j || e.weight <= 0.0 {
            continue;
        }
        let dir = -(global_rotations[e.j].inverse() * e.translation);
        if let Some(unit) = dir.try_normalize() {
            clean.push((e.i, e.j, unit, e.weight));
        }
    }
    if clean.is_empty() {
        return None;
    }

    let connected = connectivity(num_views, &clean);

    let mut centers = vec![Vec3::ZERO; num_views];
    let mut depths = vec![1.0f64; clean.len()];
    let mut num_iterations = 0usize;

    for _ in 0..options.max_num_iterations {
        num_iterations += 1;

        // Robust weights from the current residuals.
        let weights: Vec<f64> = clean
            .iter()
            .zip(depths.iter())
            .map(|(&(i, j, dir, w), &d)| {
                let residual = (centers[j] - centers[i]) - dir * d as f32;
                let norm = residual.length() as f64;
                let robust = if norm <= options.huber_threshold {
                    1.0
                } else {
                    options.huber_threshold / norm
                };
                w * robust
            })
            .collect();

        let Some(new_centers) = solve_centers(num_views, &clean, &depths, &weights) else {
            break;
        };

        // Update depths via projection onto the (positive) baseline direction.
        for (edge_idx, &(i, j, dir, _)) in clean.iter().enumerate() {
            let projection = dir.dot(new_centers[j] - new_centers[i]) as f64;
            depths[edge_idx] = projection.max(0.0);
        }

        let mut scaled = new_centers;
        normalize_scale(&mut scaled, &mut depths);

        let max_change = scaled
            .iter()
            .zip(centers.iter())
            .map(|(a, b)| (*a - *b).length() as f64)
            .fold(0.0f64, f64::max);
        centers = scaled;
        if max_change < options.convergence {
            break;
        }
    }

    let mean_residual = mean_residual(&clean, &centers, &depths);
    Some(GlobalPositioningResult {
        centers,
        num_iterations,
        mean_residual,
        connected,
    })
}

/// Solve for centers (view 0 fixed at origin) with fixed depths and weights.
/// Each axis decouples into the same weighted graph-Laplacian system.
fn solve_centers(
    num_views: usize,
    edges: &[(usize, usize, Vec3, f64)],
    depths: &[f64],
    weights: &[f64],
) -> Option<Vec<Vec3>> {
    let dim = num_views - 1;
    let mut laplacian = DMatrix::<f64>::zeros(dim, dim);
    let mut rhs = DMatrix::<f64>::zeros(dim, 3);

    for (edge_idx, &(i, j, dir, _)) in edges.iter().enumerate() {
        let w = weights[edge_idx];
        if w <= 0.0 {
            continue;
        }
        let b = dir * depths[edge_idx] as f32; // target c_j - c_i
        let bi = [b.x as f64, b.y as f64, b.z as f64];
        let ii = i.checked_sub(1);
        let jj = j.checked_sub(1);
        if let Some(ii) = ii {
            laplacian[(ii, ii)] += w;
        }
        if let Some(jj) = jj {
            laplacian[(jj, jj)] += w;
        }
        if let (Some(ii), Some(jj)) = (ii, jj) {
            laplacian[(ii, jj)] -= w;
            laplacian[(jj, ii)] -= w;
        }
        for axis in 0..3 {
            // residual = (c_j - c_i) - b; normal equations add w * (±1) * b.
            if let Some(jj) = jj {
                rhs[(jj, axis)] += w * bi[axis];
            }
            if let Some(ii) = ii {
                rhs[(ii, axis)] -= w * bi[axis];
            }
        }
    }

    for d in 0..dim {
        laplacian[(d, d)] += 1.0e-9;
    }

    let solution = match laplacian.clone().cholesky() {
        Some(chol) => chol.solve(&rhs),
        None => laplacian.lu().solve(&rhs)?,
    };
    if !solution.iter().all(|v| v.is_finite()) {
        return None;
    }

    let mut centers = vec![Vec3::ZERO; num_views];
    for idx in 1..num_views {
        let row = idx - 1;
        centers[idx] = Vec3::new(
            solution[(row, 0)] as f32,
            solution[(row, 1)] as f32,
            solution[(row, 2)] as f32,
        );
    }
    Some(centers)
}

/// Scale the configuration to unit RMS center radius (and rescale depths in
/// lockstep so the objective is unchanged). Keeps view 0 at the origin.
fn normalize_scale(centers: &mut [Vec3], depths: &mut [f64]) {
    let sum_sq: f64 = centers
        .iter()
        .map(|c| c.length_squared() as f64)
        .sum::<f64>();
    let rms = (sum_sq / centers.len().max(1) as f64).sqrt();
    if !rms.is_finite() || rms < 1.0e-12 {
        return;
    }
    let scale = (1.0 / rms) as f32;
    for c in centers.iter_mut() {
        *c *= scale;
    }
    let scale_d = 1.0 / rms;
    for d in depths.iter_mut() {
        *d *= scale_d;
    }
}

fn mean_residual(edges: &[(usize, usize, Vec3, f64)], centers: &[Vec3], depths: &[f64]) -> f64 {
    if edges.is_empty() {
        return 0.0;
    }
    let mut total = 0.0f64;
    for (edge_idx, &(i, j, dir, _)) in edges.iter().enumerate() {
        let residual = (centers[j] - centers[i]) - dir * depths[edge_idx] as f32;
        total += residual.length() as f64;
    }
    total / edges.len() as f64
}

fn connectivity(num_views: usize, edges: &[(usize, usize, Vec3, f64)]) -> Vec<bool> {
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); num_views];
    for &(i, j, _, _) in edges {
        adjacency[i].push(j);
        adjacency[j].push(i);
    }
    let mut connected = vec![false; num_views];
    let mut queue = std::collections::VecDeque::new();
    connected[0] = true;
    queue.push_back(0usize);
    while let Some(node) = queue.pop_front() {
        for &neighbor in &adjacency[node] {
            if !connected[neighbor] {
                connected[neighbor] = true;
                queue.push_back(neighbor);
            }
        }
    }
    connected
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustslam::ColmapMt19937;

    fn unit(rng: &mut ColmapMt19937) -> f32 {
        rng.next_u32() as f32 / u32::MAX as f32
    }

    fn random_vec(rng: &mut ColmapMt19937) -> Vec3 {
        Vec3::new(
            unit(rng) * 2.0 - 1.0,
            unit(rng) * 2.0 - 1.0,
            unit(rng) * 2.0 - 1.0,
        )
    }

    fn random_quat(rng: &mut ColmapMt19937) -> Quat {
        let axis = Vec3::new(unit(rng) - 0.5, unit(rng) - 0.5, unit(rng) - 0.5).normalize_or_zero();
        let axis = if axis.length_squared() < 1.0e-6 {
            Vec3::X
        } else {
            axis
        };
        Quat::from_axis_angle(axis, unit(rng) * std::f32::consts::PI)
    }

    /// Synthesize the relative translation for an edge: t_ij = -R_j (c_j - c_i).
    fn synth_edge(
        i: usize,
        j: usize,
        centers: &[Vec3],
        rotations: &[Quat],
        weight: f64,
    ) -> RelativeTranslation {
        let t = -(rotations[j] * (centers[j] - centers[i]));
        RelativeTranslation {
            i,
            j,
            translation: t,
            weight,
        }
    }

    /// Best-fit isotropic scale aligning `est` to `gt` (both share origin at 0).
    fn align_scale(est: &[Vec3], gt: &[Vec3]) -> f32 {
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for (e, g) in est.iter().zip(gt.iter()) {
            num += e.dot(*g);
            den += e.dot(*e);
        }
        if den < 1.0e-12 {
            1.0
        } else {
            num / den
        }
    }

    #[test]
    fn recovers_centers_from_clean_directions_identity_rotations() {
        let mut rng = ColmapMt19937::new(11);
        let n = 8;
        let rotations = vec![Quat::IDENTITY; n];
        let mut gt = vec![Vec3::ZERO];
        for _ in 1..n {
            gt.push(random_vec(&mut rng));
        }
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if j - i > 3 {
                    continue;
                }
                edges.push(synth_edge(i, j, &gt, &rotations, 100.0));
            }
        }
        let result =
            estimate_global_positions(&rotations, &edges, &GlobalPositioningOptions::default())
                .unwrap();
        assert!(result.connected.iter().all(|&c| c));
        let scale = align_scale(&result.centers, &gt);
        for idx in 0..n {
            let err = (result.centers[idx] * scale - gt[idx]).length();
            assert!(err < 1.0e-2, "view {idx} center error {err} too large");
        }
    }

    #[test]
    fn recovers_centers_with_nonidentity_rotations() {
        let mut rng = ColmapMt19937::new(99);
        let n = 10;
        let mut rotations = vec![Quat::IDENTITY];
        let mut gt = vec![Vec3::ZERO];
        for _ in 1..n {
            rotations.push(random_quat(&mut rng));
            gt.push(random_vec(&mut rng));
        }
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if j - i > 4 {
                    continue;
                }
                edges.push(synth_edge(i, j, &gt, &rotations, 80.0));
            }
        }
        let result =
            estimate_global_positions(&rotations, &edges, &GlobalPositioningOptions::default())
                .unwrap();
        let scale = align_scale(&result.centers, &gt);
        for idx in 0..n {
            let err = (result.centers[idx] * scale - gt[idx]).length();
            assert!(err < 1.0e-2, "view {idx} center error {err} too large");
        }
        assert!(result.mean_residual < 1.0e-2);
    }

    #[test]
    fn is_robust_to_an_outlier_translation() {
        let mut rng = ColmapMt19937::new(5);
        let n = 8;
        let rotations = vec![Quat::IDENTITY; n];
        let mut gt = vec![Vec3::ZERO];
        for _ in 1..n {
            gt.push(random_vec(&mut rng));
        }
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if j - i > 3 {
                    continue;
                }
                edges.push(synth_edge(i, j, &gt, &rotations, 100.0));
            }
        }
        // Corrupt one edge with a random direction.
        edges.push(RelativeTranslation {
            i: 1,
            j: 4,
            translation: random_vec(&mut rng),
            weight: 100.0,
        });
        let result =
            estimate_global_positions(&rotations, &edges, &GlobalPositioningOptions::default())
                .unwrap();
        let scale = align_scale(&result.centers, &gt);
        for idx in 0..n {
            let err = (result.centers[idx] * scale - gt[idx]).length();
            assert!(
                err < 0.1,
                "view {idx} center error {err}; outlier not suppressed"
            );
        }
    }

    #[test]
    fn marks_disconnected_views() {
        let rotations = vec![Quat::IDENTITY; 4];
        let gt = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
        let edges = vec![synth_edge(0, 1, &gt, &rotations, 10.0)];
        let result =
            estimate_global_positions(&rotations, &edges, &GlobalPositioningOptions::default())
                .unwrap();
        assert!(result.connected[0]);
        assert!(result.connected[1]);
        assert!(!result.connected[2]);
        assert!(!result.connected[3]);
    }

    #[test]
    fn rejects_too_few_views() {
        assert!(estimate_global_positions(
            &[Quat::IDENTITY],
            &[],
            &GlobalPositioningOptions::default()
        )
        .is_none());
    }
}
