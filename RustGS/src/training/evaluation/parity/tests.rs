use super::{
    compare_loss_curve_samples, default_litegs_parity_fixtures, default_parity_report_path,
    parity_fixture_id_for_input_path, resolve_litegs_parity_fixture_input_path,
    resolve_litegs_parity_reference_report_path, ParityGateStatus, ParityHarnessReport,
    ParityReferenceComparison, ParityThresholds, DEFAULT_CONVERGENCE_FIXTURE_ID,
    DEFAULT_TINY_FIXTURE_ID,
};
use crate::training::reporting::metrics::ParityLossCurveSample;
use crate::LiteGsConfig;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

#[test]
fn default_fixture_registry_contains_tiny_and_convergence_targets() {
    let fixtures = default_litegs_parity_fixtures();
    assert_eq!(fixtures.len(), 2);
    assert_eq!(fixtures[0].id, DEFAULT_TINY_FIXTURE_ID);
    assert_eq!(fixtures[1].id, DEFAULT_CONVERGENCE_FIXTURE_ID);
    assert_eq!(
        fixtures[1].input_path.as_ref().unwrap().to_string_lossy(),
        "test_data/tum_freiburg1_xyz_colmap"
    );
    assert!(fixtures[1].bootstrap_input_path.is_none());
}

#[test]
fn parity_thresholds_match_epic_17_targets() {
    let thresholds = ParityThresholds::default();
    assert!(thresholds.require_no_nans);
    assert!(thresholds.require_no_oom);
    assert!(thresholds.require_deterministic_export_roundtrip);
    assert_eq!(thresholds.psnr_delta_db_tolerance, None);
    assert_eq!(thresholds.gaussian_count_delta_ratio_tolerance, Some(0.10));
    assert_eq!(thresholds.depth_mean_abs_delta_tolerance, None);
    assert_eq!(thresholds.depth_max_abs_delta_tolerance, None);
    assert_eq!(thresholds.total_mean_abs_delta_tolerance, None);
    assert_eq!(thresholds.total_max_abs_delta_tolerance, None);
}

#[test]
fn parity_report_round_trips_through_json() {
    let tempdir = tempdir().unwrap();
    let path = tempdir.path().join("parity/report.json");
    let mut report = ParityHarnessReport::new(DEFAULT_TINY_FIXTURE_ID, &LiteGsConfig::default());
    report.metrics.final_psnr = Some(28.4);
    report.loss_curve_samples.push(ParityLossCurveSample {
        iteration: 4,
        frame_idx: 1,
        depth: Some(0.2),
        total: Some(0.5),
        depth_valid_pixels: Some(64),
        ..Default::default()
    });
    report.topology.initialization_gaussians = Some(1024);
    report.notes.push("roundtrip".to_string());

    report.save_json(&path).unwrap();
    let loaded = ParityHarnessReport::load_json(&path).unwrap();

    assert_eq!(loaded, report);
}

#[test]
fn parity_fixture_id_matches_known_bootstrap_fixture_and_external_fallback() {
    assert_eq!(
        parity_fixture_id_for_input_path(Path::new("test_data/tum_freiburg1_xyz_colmap")),
        DEFAULT_CONVERGENCE_FIXTURE_ID
    );
    assert_eq!(
        parity_fixture_id_for_input_path(Path::new("/tmp/my_scene")),
        "external::my_scene"
    );
}

#[test]
fn default_parity_report_path_uses_output_stem() {
    let path = default_parity_report_path(Path::new("/tmp/scene.ply"));
    assert_eq!(path, PathBuf::from("/tmp/scene.parity.json"));
}

#[test]
fn resolve_litegs_fixture_input_prefers_canonical_fixture_and_falls_back_to_bootstrap() {
    let tempdir = tempdir().unwrap();
    let workspace_root = tempdir.path();
    assert!(resolve_litegs_parity_fixture_input_path(
        DEFAULT_CONVERGENCE_FIXTURE_ID,
        workspace_root
    )
    .is_none());

    let canonical = workspace_root.join("test_data/tum_freiburg1_xyz_colmap");
    std::fs::create_dir_all(&canonical).unwrap();

    let resolved_canonical =
        resolve_litegs_parity_fixture_input_path(DEFAULT_CONVERGENCE_FIXTURE_ID, workspace_root)
            .unwrap();
    assert_eq!(resolved_canonical, canonical);
}

#[test]
fn resolve_litegs_reference_report_path_returns_existing_reference_only() {
    let tempdir = tempdir().unwrap();
    let workspace_root = tempdir.path();
    assert!(resolve_litegs_parity_reference_report_path(
        DEFAULT_CONVERGENCE_FIXTURE_ID,
        workspace_root,
    )
    .is_none());

    let reference =
        workspace_root.join("test_data/fixtures/litegs/colmap-small/parity-reference.json");
    std::fs::create_dir_all(reference.parent().unwrap()).unwrap();
    std::fs::write(&reference, "{}").unwrap();

    let resolved =
        resolve_litegs_parity_reference_report_path(DEFAULT_CONVERGENCE_FIXTURE_ID, workspace_root)
            .unwrap();
    assert_eq!(resolved, reference);
}

#[test]
fn compare_loss_curve_samples_matches_iterations_and_reports_deltas() {
    let current = vec![
        ParityLossCurveSample {
            iteration: 0,
            depth: Some(0.6),
            total: Some(1.0),
            ..Default::default()
        },
        ParityLossCurveSample {
            iteration: 25,
            depth: Some(0.3),
            total: Some(0.7),
            ..Default::default()
        },
    ];
    let reference = vec![
        ParityLossCurveSample {
            iteration: 0,
            depth: Some(0.5),
            total: Some(0.8),
            ..Default::default()
        },
        ParityLossCurveSample {
            iteration: 25,
            depth: Some(0.4),
            total: Some(0.6),
            ..Default::default()
        },
        ParityLossCurveSample {
            iteration: 50,
            depth: Some(0.2),
            total: Some(0.5),
            ..Default::default()
        },
    ];

    let comparison = compare_loss_curve_samples(&current, &reference).unwrap();
    assert_eq!(comparison.compared_iterations, 2);
    assert_eq!(comparison.compared_depth_samples, 2);
    assert_eq!(comparison.compared_total_samples, 2);
    assert!(
        (comparison.depth_mean_abs_delta.unwrap() - 0.1).abs() < 1e-6,
        "{comparison:?}"
    );
    assert!(
        (comparison.depth_max_abs_delta.unwrap() - 0.1).abs() < 1e-6,
        "{comparison:?}"
    );
    assert!(
        (comparison.total_mean_abs_delta.unwrap() - 0.15).abs() < 1e-6,
        "{comparison:?}"
    );
    assert!(
        (comparison.total_max_abs_delta.unwrap() - 0.2).abs() < 1e-6,
        "{comparison:?}"
    );
}

#[test]
fn parity_gate_reports_missing_reference_for_convergence_fixture() {
    let mut report =
        ParityHarnessReport::new(DEFAULT_CONVERGENCE_FIXTURE_ID, &LiteGsConfig::default());
    report.metrics.export_roundtrip_ok = true;

    let evaluation = report.evaluate_gate();
    assert_eq!(evaluation.status, ParityGateStatus::MissingReference);
    assert!(evaluation
        .checks
        .iter()
        .any(|check| check.name == "reference_report"));
}

#[test]
fn parity_gate_fails_when_gaussian_count_delta_exceeds_tolerance() {
    let mut report =
        ParityHarnessReport::new(DEFAULT_CONVERGENCE_FIXTURE_ID, &LiteGsConfig::default());
    report.metrics.export_roundtrip_ok = true;
    report.metrics.gaussian_count_delta_ratio = Some(0.25);

    let evaluation = report.evaluate_gate();
    assert_eq!(evaluation.status, ParityGateStatus::Failed);
    assert!(evaluation.checks.iter().any(|check| {
        check.name == "gaussian_count_delta_ratio"
            && check.status == super::ParityCheckStatus::Failed
    }));
}

#[test]
fn parity_gate_passes_when_reference_backed_checks_are_within_tolerance() {
    let mut report =
        ParityHarnessReport::new(DEFAULT_CONVERGENCE_FIXTURE_ID, &LiteGsConfig::default());
    report.metrics.export_roundtrip_ok = true;
    report.metrics.gaussian_count_delta_ratio = Some(0.05);
    report.reference_comparison = Some(ParityReferenceComparison {
        compared_iterations: 2,
        compared_depth_samples: 2,
        compared_total_samples: 2,
        depth_mean_abs_delta: Some(0.02),
        depth_max_abs_delta: Some(0.05),
        total_mean_abs_delta: Some(0.03),
        total_max_abs_delta: Some(0.05),
    });
    report.thresholds.depth_mean_abs_delta_tolerance = Some(0.05);
    report.thresholds.depth_max_abs_delta_tolerance = Some(0.10);

    let evaluation = report.evaluate_gate();
    assert_eq!(evaluation.status, ParityGateStatus::Passed);
}
