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
    let initial_count = dataset
        .initial_points
        .len()
        .min(config.initialization.max_initial_gaussians)
        .min(config.litegs.topology.target_primitives);
    let splats = initialize_host_splats_from_points(
        &dataset.initial_points[..initial_count],
        &init_config,
        sh_degree,
    )?;

    splats
        .validate()
        .map_err(|err| TrainingError::TrainingFailed(err.to_string()))?;
    Ok(splats)
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
}
