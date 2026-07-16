//! Splat initialization from point clouds.
//!
//! Provides utilities to create initial splats for 3DGS training
//! from sparse point clouds (e.g., SLAM map points).
//! Uses KdTree for nearest-neighbor scale computation.

#[cfg(feature = "gpu")]
use glam::Vec3;
#[cfg(feature = "gpu")]
use kiddo::{KdTree, SquaredEuclidean};
#[cfg(feature = "gpu")]
use rand::{rngs::StdRng, Rng, SeedableRng};

#[cfg(feature = "gpu")]
use crate::core::HostSplats;
#[cfg(feature = "gpu")]
use crate::sh::{rgb_to_sh0_value, sh_coeff_count_for_degree};
#[cfg(feature = "gpu")]
use crate::TrainingError;

/// Configuration for Gaussian initialization from point clouds.
#[derive(Debug, Clone)]
pub struct GaussianInitConfig {
    /// Minimum scale (meters).
    pub min_scale: f32,
    /// Absolute maximum scale (meters) after scene-derived clamping.
    pub max_scale: f32,
    /// Scale factor applied to the average of the two nearest-neighbor distances.
    pub scale_factor: f32,
    /// Default color when point color is unavailable (RGB, 0-1).
    pub default_color: [f32; 3],
    /// Default opacity for initialized Gaussians.
    pub opacity: f32,
    /// Use deterministic random unit quaternions instead of identity rotations.
    pub randomize_rotations: bool,
    /// Seed used when `randomize_rotations` is enabled.
    pub rotation_seed: u64,
    /// Match VkSplat/Nerfstudio nearest-neighbor scale estimation.
    pub vksplat_scale_estimator: bool,
}

impl Default for GaussianInitConfig {
    fn default() -> Self {
        Self {
            min_scale: 1e-3,
            max_scale: f32::MAX,
            scale_factor: 0.5,
            default_color: [0.5, 0.5, 0.5],
            opacity: 0.5,
            randomize_rotations: false,
            rotation_seed: 42,
            vksplat_scale_estimator: false,
        }
    }
}

/// Initialize runtime splats directly on device from a point cloud.
/// Initialize host-side splats from a point cloud without materializing AoS gaussians.
#[cfg(feature = "gpu")]
pub fn initialize_host_splats_from_points(
    points: &[([f32; 3], Option<[f32; 3]>)],
    config: &GaussianInitConfig,
    sh_degree: usize,
) -> Result<HostSplats, TrainingError> {
    if points.is_empty() {
        return HostSplats::from_components(
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            sh_degree,
        );
    }

    let positions_vec3: Vec<Vec3> = points
        .iter()
        .map(|(p, _)| Vec3::new(p[0], p[1], p[2]))
        .collect();
    let scales = compute_scales(&positions_vec3, config);

    let row_count = points.len();
    let sh_row_width = sh_coeff_count_for_degree(sh_degree) * 3;
    let mut positions = Vec::with_capacity(row_count * 3);
    let mut log_scales = Vec::with_capacity(row_count * 3);
    let mut rotations = Vec::with_capacity(row_count * 4);
    let mut opacity_logits = Vec::with_capacity(row_count);
    let mut sh_coeffs = vec![0.0; row_count * sh_row_width];
    let mut rotation_rng = config
        .randomize_rotations
        .then(|| StdRng::seed_from_u64(config.rotation_seed));

    for (idx, ((position, color), scale)) in points.iter().zip(scales.iter()).enumerate() {
        let rgb = color.unwrap_or(config.default_color);
        positions.extend_from_slice(position);
        log_scales.extend_from_slice(&[scale.ln(), scale.ln(), scale.ln()]);
        rotations.extend_from_slice(&next_rotation(&mut rotation_rng));
        opacity_logits.push(opacity_to_logit(config.opacity));

        let sh_base = idx * sh_row_width;
        sh_coeffs[sh_base..sh_base + 3].copy_from_slice(&rgb.map(rgb_to_sh0_value));
    }

    HostSplats::from_components(
        positions,
        log_scales,
        rotations,
        opacity_logits,
        sh_coeffs,
        sh_degree,
    )
}

#[cfg(feature = "gpu")]
fn compute_scales(points: &[Vec3], config: &GaussianInitConfig) -> Vec<f32> {
    if points.len() < 3 {
        return vec![1.0; points.len()];
    }

    let mut tree: KdTree<f32, 3> = KdTree::new();
    for (idx, pos) in points.iter().enumerate() {
        tree.add(&[pos.x, pos.y, pos.z], idx as u64);
    }

    let scene_max_scale = brush_scene_max_scale(points).clamp(config.min_scale, config.max_scale);
    let mut scales = Vec::with_capacity(points.len());
    for (idx, pos) in points.iter().enumerate() {
        let query = [pos.x, pos.y, pos.z];
        let neighbor_count = if config.vksplat_scale_estimator { 4 } else { 3 };
        let neighbors =
            tree.nearest_n::<SquaredEuclidean>(&query, points.len().min(neighbor_count));

        let mut nearest = [None, None, None];
        let mut count = 0usize;
        for neighbor in neighbors {
            if neighbor.item as usize == idx {
                continue;
            }
            nearest[count] = Some(neighbor.distance.sqrt());
            count += 1;
            let needed = if config.vksplat_scale_estimator { 3 } else { 2 };
            if count == needed {
                break;
            }
        }

        let scale = if config.vksplat_scale_estimator {
            if let [Some(first), Some(second), Some(third)] = nearest {
                let mean_dist = ((first * first + second * second + third * third) / 3.0).sqrt();
                (mean_dist * config.scale_factor).clamp(config.min_scale, config.max_scale)
            } else {
                1.0
            }
        } else if let [Some(first), Some(second), _] = nearest {
            let avg_neighbor_distance = (first + second) * 0.5;
            (avg_neighbor_distance * config.scale_factor).clamp(config.min_scale, scene_max_scale)
        } else {
            1.0
        };
        scales.push(scale);
    }

    scales
}

#[cfg(feature = "gpu")]
fn brush_scene_max_scale(points: &[Vec3]) -> f32 {
    let bounds = percentile_bounds(points, 0.75);
    let mut extents = [
        (bounds.1.x - bounds.0.x) * 0.5,
        (bounds.1.y - bounds.0.y) * 0.5,
        (bounds.1.z - bounds.0.z) * 0.5,
    ];
    extents.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_size = (extents[1] * 2.0).max(0.01);
    median_size * 0.1
}

#[cfg(feature = "gpu")]
fn percentile_bounds(points: &[Vec3], percentile: f32) -> (Vec3, Vec3) {
    let mut xs = Vec::with_capacity(points.len());
    let mut ys = Vec::with_capacity(points.len());
    let mut zs = Vec::with_capacity(points.len());
    for pos in points {
        xs.push(pos.x);
        ys.push(pos.y);
        zs.push(pos.z);
    }

    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    zs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let len = xs.len().max(1);
    let lower_idx = (((1.0 - percentile) * 0.5) * len as f32) as usize;
    let upper_idx = (len - 1).min((((1.0 + percentile) * 0.5) * len as f32) as usize);

    (
        Vec3::new(xs[lower_idx], ys[lower_idx], zs[lower_idx]),
        Vec3::new(xs[upper_idx], ys[upper_idx], zs[upper_idx]),
    )
}

#[cfg(feature = "gpu")]
fn opacity_to_logit(opacity: f32) -> f32 {
    let clamped = opacity.clamp(1e-6, 1.0 - 1e-6);
    (clamped / (1.0 - clamped)).ln()
}

#[cfg(feature = "gpu")]
fn next_rotation(rotation_rng: &mut Option<StdRng>) -> [f32; 4] {
    let Some(rng) = rotation_rng else {
        return [1.0, 0.0, 0.0, 0.0];
    };

    loop {
        let rotation = [
            standard_normal(rng),
            standard_normal(rng),
            standard_normal(rng),
            standard_normal(rng),
        ];
        let norm_sq = rotation.iter().map(|value| value * value).sum::<f32>();
        if norm_sq > 1e-12 {
            let inv_norm = norm_sq.sqrt().recip();
            return rotation.map(|value| value * inv_norm);
        }
    }
}

#[cfg(feature = "gpu")]
fn standard_normal(rng: &mut StdRng) -> f32 {
    let u1 = rng.gen::<f32>().clamp(f32::MIN_POSITIVE, 1.0);
    let u2 = rng.gen::<f32>();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}
