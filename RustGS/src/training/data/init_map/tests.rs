use super::init_map::{build_initial_splats, gaussian_init_config_for_training};
use crate::{Intrinsics, TrainingConfig, TrainingDataset, TrainingInitializationConfig};

#[test]
fn training_profiles_share_brush_sparse_point_defaults() {
    let init = gaussian_init_config_for_training();
    assert_eq!(init.min_scale, 1e-3);
    assert_eq!(init.scale_factor, 0.5);
    assert_eq!(init.opacity, 0.5);
    assert_eq!(init.max_scale, f32::MAX);
}

#[test]
fn build_initial_splats_requires_sparse_points() {
    let dataset = TrainingDataset::new(Intrinsics::new(1.0, 1.0, 0.0, 0.0, 2, 1));
    let err = build_initial_splats(&dataset, &crate::TrainingConfig::default()).unwrap_err();
    assert!(
        err.to_string()
            .contains("COLMAP sparse points for initialization"),
        "unexpected error: {err}"
    );
}

#[test]
fn build_initial_splats_keeps_all_sparse_points() {
    let mut dataset = TrainingDataset::new(Intrinsics::new(1.0, 1.0, 0.0, 0.0, 2, 1));
    dataset.add_point([0.0, 0.0, 0.0], None);
    dataset.add_point([1.0, 0.0, 0.0], None);
    dataset.add_point([0.0, 1.0, 0.0], None);

    let config = TrainingConfig {
        initialization: TrainingInitializationConfig {
            max_initial_gaussians: 1,
            ..Default::default()
        },
        ..Default::default()
    };

    let splats = build_initial_splats(&dataset, &config).unwrap();
    assert_eq!(splats.len(), dataset.initial_points.len());
}
