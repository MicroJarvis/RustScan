use std::collections::BTreeMap;

use crate::core::HostSplats;
use crate::init::{initialize_host_splats_from_points, GaussianInitConfig};
use crate::{TrainingConfig, TrainingDataset, TrainingError};

pub(crate) fn build_initial_splats(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<HostSplats, TrainingError> {
    if dataset.initial_points.is_empty() {
        return Err(TrainingError::InvalidInput(
            "training now requires COLMAP sparse points for initialization; no initial_points were found in the dataset".to_string(),
        ));
    }

    let sh_degree = config.litegs.rendering.sh_degree;
    let init_config = gaussian_init_config_for_training(config);
    let budget = dataset
        .initial_points
        .len()
        .min(config.initialization.max_initial_gaussians)
        .min(config.litegs.topology.target_primitives);
    let selected = select_initial_points(
        &dataset.initial_points,
        budget,
        config.initialization.init_sampling_seed,
        config.initialization.init_voxel_cell_size,
    );
    let splats = initialize_host_splats_from_points(&selected, &init_config, sh_degree)?;

    splats
        .validate()
        .map_err(|err| TrainingError::TrainingFailed(err.to_string()))?;
    Ok(splats)
}

/// Deterministic truncation: when the budget is large enough keep input order;
/// otherwise keep one representative per occupied voxel (first index wins) and
/// fill remaining slots with leftover points in input order.
pub(crate) fn select_initial_points(
    points: &[( [f32; 3], Option<[f32; 3]> )],
    budget: usize,
    seed: u64,
    cell_size: f32,
) -> Vec<([f32; 3], Option<[f32; 3]>)> {
    let _ = seed; // reserved for future stratified jitter; voxel choice is index-stable.
    if budget == 0 || points.is_empty() {
        return Vec::new();
    }
    if points.len() <= budget {
        return points.to_vec();
    }
    let cell = if cell_size.is_finite() && cell_size > 0.0 {
        cell_size
    } else {
        // Auto cell from axis-aligned extents so clustered scenes keep boundary coverage.
        auto_voxel_cell_size(points, budget)
    };

    let mut voxel_first: BTreeMap<(i32, i32, i32), usize> = BTreeMap::new();
    for (idx, (position, _)) in points.iter().enumerate() {
        let key = voxel_key(*position, cell);
        voxel_first.entry(key).or_insert(idx);
    }

    let mut selected_indices = Vec::with_capacity(budget);
    let mut used = vec![false; points.len()];
    for idx in voxel_first.values().copied() {
        if selected_indices.len() >= budget {
            break;
        }
        selected_indices.push(idx);
        used[idx] = true;
    }
    // Fill remaining budget with unused points in original order for stability.
    for (idx, taken) in used.iter().enumerate() {
        if selected_indices.len() >= budget {
            break;
        }
        if !*taken {
            selected_indices.push(idx);
        }
    }
    selected_indices.sort_unstable();
    selected_indices
        .into_iter()
        .map(|idx| points[idx].clone())
        .collect()
}

fn voxel_key(position: [f32; 3], cell: f32) -> (i32, i32, i32) {
    let inv = 1.0 / cell;
    (
        (position[0] * inv).floor() as i32,
        (position[1] * inv).floor() as i32,
        (position[2] * inv).floor() as i32,
    )
}

fn auto_voxel_cell_size(points: &[([f32; 3], Option<[f32; 3]>)], budget: usize) -> f32 {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for (position, _) in points {
        for axis in 0..3 {
            min[axis] = min[axis].min(position[axis]);
            max[axis] = max[axis].max(position[axis]);
        }
    }
    let extent = ((max[0] - min[0]).max(0.0) * (max[1] - min[1]).max(0.0) * (max[2] - min[2]).max(0.0))
        .cbrt()
        .max(1e-3);
    let target_cells = budget.max(1) as f32;
    (extent / target_cells.cbrt()).max(1e-3)
}

pub(super) fn gaussian_init_config_for_training(config: &TrainingConfig) -> GaussianInitConfig {
    GaussianInitConfig {
        scale_factor: config.initialization.point_scale_factor,
        opacity: config.initialization.point_opacity,
        vksplat_scale_estimator: config.initialization.vksplat_scale_estimator,
        randomize_rotations: config.initialization.randomize_rotations,
        rotation_seed: config.initialization.rotation_seed,
        ..GaussianInitConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Intrinsics;

    #[test]
    fn sparse_initialization_honors_max_initial_gaussians() {
        let mut dataset = TrainingDataset::new(Intrinsics::default());
        for idx in 0..8 {
            dataset.add_point([idx as f32, 0.0, 1.0], None);
        }
        let mut config = TrainingConfig::default();
        config.initialization.max_initial_gaussians = 3;

        let splats = build_initial_splats(&dataset, &config).unwrap();
        assert_eq!(splats.len(), 3);
    }

    #[test]
    fn voxel_sampling_keeps_boundary_when_truncating_clusters() {
        // Dense cluster near origin plus sparse boundary points.
        let mut points = Vec::new();
        for idx in 0..50 {
            points.push(([0.01 * idx as f32, 0.0, 0.0], Some([1.0, 0.0, 0.0])));
        }
        points.push(([10.0, 0.0, 0.0], Some([0.0, 1.0, 0.0])));
        points.push(([-10.0, 0.0, 0.0], Some([0.0, 0.0, 1.0])));
        let selected = select_initial_points(&points, 10, 0, 1.0);
        assert_eq!(selected.len(), 10);
        let has_pos = selected.iter().any(|(p, _)| (p[0] - 10.0).abs() < 1e-5);
        let has_neg = selected.iter().any(|(p, _)| (p[0] + 10.0).abs() < 1e-5);
        assert!(has_pos && has_neg, "voxel sampling must retain sparse boundary points");
    }

    #[test]
    fn voxel_sampling_is_deterministic() {
        let points: Vec<_> = (0..40)
            .map(|idx| ([idx as f32, (idx % 5) as f32, 0.0], None))
            .collect();
        let a = select_initial_points(&points, 12, 7, 2.0);
        let b = select_initial_points(&points, 12, 7, 2.0);
        assert_eq!(a, b);
    }
}
