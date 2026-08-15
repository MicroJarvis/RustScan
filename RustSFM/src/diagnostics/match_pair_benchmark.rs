use crate::database::ColmapDatabase;
use crate::feature_matching::{generate_matching_pairs, MatchingPairStrategy};
use crate::feature_matching_db::{
    match_explicit_image_pairs_to_database_with_session, ExplicitPairMatchingSession,
};
use crate::{MatchFeaturesOptions, MatchFeaturesTimingReport, SfmTaskContext, SfmTaskControl};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn copy_benchmark_snapshot_for_run(
    snapshot: &Path,
    work_dir: &Path,
    run_index: usize,
) -> Result<PathBuf> {
    let database = work_dir.join(format!("run-{}.db", run_index + 1));
    std::fs::copy(snapshot, &database)
        .with_context(|| format!("copy benchmark snapshot for run {}", run_index + 1))?;
    Ok(database)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchPairBenchmarkRun {
    pub run_index: usize,
    pub backend: String,
    pub pair_count: usize,
    pub matched_pairs: usize,
    pub verified_pairs: usize,
    pub total_matches: usize,
    pub matching_seconds: f64,
    pub timings: MatchFeaturesTimingReport,
    #[serde(default)]
    pub result_fingerprint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchPairBenchmarkReport {
    pub source_database: PathBuf,
    pub copied_database_bytes: u64,
    pub database_copy_seconds: f64,
    pub window: usize,
    pub requested_pair_limit: Option<usize>,
    pub pair_count: usize,
    pub repetitions: usize,
    pub session_initialization_seconds: f64,
    pub runs: Vec<MatchPairBenchmarkRun>,
}

pub(crate) fn select_local_window_image_pairs(
    image_ids: &[u32],
    window: usize,
    pair_limit: Option<usize>,
) -> Result<Vec<(u32, u32)>> {
    if window == 0 {
        bail!("match-pair benchmark window must be greater than zero");
    }
    if pair_limit == Some(0) {
        bail!("match-pair benchmark pair limit must be greater than zero");
    }
    let mut pairs = generate_matching_pairs(
        image_ids.len(),
        MatchingPairStrategy::LocalWindow { window },
    )
    .into_iter()
    .map(|(left, right)| (image_ids[left], image_ids[right]))
    .collect::<Vec<_>>();
    if let Some(limit) = pair_limit {
        pairs.truncate(limit);
    }
    Ok(pairs)
}

pub fn benchmark_match_pairs(
    source_database: &Path,
    window: usize,
    pair_limit: Option<usize>,
    repetitions: usize,
    options: &MatchFeaturesOptions,
) -> Result<MatchPairBenchmarkReport> {
    validate_benchmark_controls(window, pair_limit, repetitions)?;
    let work_dir = tempfile::tempdir().context("create match-pair benchmark directory")?;
    benchmark_match_pairs_in_directory(
        source_database,
        window,
        pair_limit,
        repetitions,
        options,
        work_dir.path(),
    )
}

pub fn benchmark_match_pairs_with_artifacts(
    source_database: &Path,
    window: usize,
    pair_limit: Option<usize>,
    repetitions: usize,
    options: &MatchFeaturesOptions,
    artifacts_dir: &Path,
) -> Result<MatchPairBenchmarkReport> {
    validate_benchmark_controls(window, pair_limit, repetitions)?;
    std::fs::create_dir(artifacts_dir).with_context(|| {
        format!(
            "create match-pair benchmark artifacts directory {} (path must not already exist)",
            artifacts_dir.display()
        )
    })?;
    benchmark_match_pairs_in_directory(
        source_database,
        window,
        pair_limit,
        repetitions,
        options,
        artifacts_dir,
    )
}

fn validate_benchmark_controls(
    window: usize,
    pair_limit: Option<usize>,
    repetitions: usize,
) -> Result<()> {
    if window == 0 {
        bail!("match-pair benchmark window must be greater than zero");
    }
    if pair_limit == Some(0) {
        bail!("match-pair benchmark pair limit must be greater than zero");
    }
    if repetitions == 0 {
        bail!("match-pair benchmark repetitions must be greater than zero");
    }
    Ok(())
}

fn benchmark_match_pairs_in_directory(
    source_database: &Path,
    window: usize,
    pair_limit: Option<usize>,
    repetitions: usize,
    options: &MatchFeaturesOptions,
    work_dir: &Path,
) -> Result<MatchPairBenchmarkReport> {
    let source_snapshot = work_dir.join("source-snapshot.db");
    let copy_started = Instant::now();
    let source = ColmapDatabase::open_read_only(source_database)
        .with_context(|| format!("open benchmark source {}", source_database.display()))?;
    let copied_database_bytes = source.backup_to(&source_snapshot)?;
    drop(source);
    let database_copy_seconds = copy_started.elapsed().as_secs_f64();

    let snapshot = ColmapDatabase::open_read_only(&source_snapshot)
        .context("open completed benchmark source snapshot")?;
    let mut images = snapshot.read_all_images()?;
    drop(snapshot);
    images.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| left.image_id.cmp(&right.image_id))
    });
    let image_ids = images
        .into_iter()
        .map(|image| image.image_id)
        .collect::<Vec<_>>();
    let image_pairs = select_local_window_image_pairs(&image_ids, window, pair_limit)?;
    if image_pairs.is_empty() {
        bail!("match-pair benchmark requires at least one generated image pair");
    }

    let mut run_options = options.clone();
    run_options.clear_existing = true;
    run_options.use_existing_matches = false;
    let session = ExplicitPairMatchingSession::new(&run_options)?;
    let session_initialization_seconds = session.initialization_seconds();
    let mut runs = Vec::with_capacity(repetitions);
    for run_index in 0..repetitions {
        let working_database =
            copy_benchmark_snapshot_for_run(&source_snapshot, work_dir, run_index)?;
        let control = SfmTaskControl::new();
        let mut sink = |_event| {};
        let mut task = SfmTaskContext::new(&control, &mut sink);
        let report = match_explicit_image_pairs_to_database_with_session(
            &working_database,
            &image_pairs,
            &run_options,
            &session,
            &mut task,
        )?;
        let result_fingerprint = ColmapDatabase::open_read_only(&working_database)?
            .selected_pair_output_fingerprint(&image_pairs)
            .with_context(|| format!("fingerprint benchmark run {} outputs", run_index + 1))?;
        let mut timings = report.timings;
        timings.backend_initialization_seconds = 0.0;
        timings.finish(report.matching_seconds);
        runs.push(MatchPairBenchmarkRun {
            run_index: run_index + 1,
            backend: report.backend,
            pair_count: report.pair_count,
            matched_pairs: report.matched_pairs,
            verified_pairs: report.verified_pairs,
            total_matches: report.total_matches,
            matching_seconds: report.matching_seconds,
            timings,
            result_fingerprint,
        });
    }

    Ok(MatchPairBenchmarkReport {
        source_database: source_database.to_path_buf(),
        copied_database_bytes,
        database_copy_seconds,
        window,
        requested_pair_limit: pair_limit,
        pair_count: image_pairs.len(),
        repetitions,
        session_initialization_seconds,
        runs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colmap::ColmapCamera;
    use crate::correspondence_graph::FeatureMatch;
    use crate::database::{
        ColmapDatabase, ColmapDatabaseCamera, ColmapDatabaseImage, ColmapDescriptors,
        ColmapTwoViewGeometry,
    };
    use crate::types::COLMAP_PINHOLE;

    fn write_empty_feature_database(path: &std::path::Path) {
        let db = ColmapDatabase::open(path).unwrap();
        db.write_camera(
            &ColmapDatabaseCamera {
                camera: ColmapCamera {
                    camera_id: 1,
                    model_id: COLMAP_PINHOLE,
                    width: 100,
                    height: 100,
                    params: vec![50.0, 50.0, 50.0, 50.0],
                },
                has_prior_focal_length: true,
            },
            true,
        )
        .unwrap();
        for image_id in 1..=3 {
            db.write_image(
                &ColmapDatabaseImage {
                    image_id,
                    name: format!("frame_{image_id:04}.jpg"),
                    camera_id: 1,
                    frame_id: None,
                },
                true,
            )
            .unwrap();
            db.write_keypoints(image_id, &[]).unwrap();
            db.write_descriptors(
                image_id,
                &ColmapDescriptors {
                    feature_type: 0,
                    rows: 0,
                    cols: 128,
                    data: Vec::new(),
                },
            )
            .unwrap();
        }
    }

    #[test]
    fn match_pair_benchmark_limit_is_explicit_and_unlimited_by_default() {
        let image_ids = (1..=600).collect::<Vec<_>>();

        let limited = select_local_window_image_pairs(&image_ids, 5, Some(96)).unwrap();
        let unlimited = select_local_window_image_pairs(&image_ids, 5, None).unwrap();

        assert_eq!(limited.len(), 96);
        assert!(unlimited.len() > 2_890);
        assert_eq!(limited, unlimited[..96]);
    }

    #[test]
    fn match_pair_benchmark_rejects_zero_controls_before_io() {
        let missing = std::path::Path::new("missing.db");
        let options = crate::MatchFeaturesOptions::default();

        assert!(benchmark_match_pairs(missing, 0, Some(1), 1, &options).is_err());
        assert!(benchmark_match_pairs(missing, 5, Some(0), 1, &options).is_err());
        assert!(benchmark_match_pairs(missing, 5, Some(1), 0, &options).is_err());
    }

    #[test]
    fn match_pair_benchmark_repeats_without_modifying_source_database() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("database.db");
        write_empty_feature_database(&source);
        let original_matches = vec![FeatureMatch::new(3, 7)];
        let original_geometry = ColmapTwoViewGeometry {
            config: 2,
            inlier_matches: original_matches.clone(),
            e_matrix: Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            ..ColmapTwoViewGeometry::default()
        };
        let database = ColmapDatabase::open(&source).unwrap();
        database.write_matches(1, 2, &original_matches).unwrap();
        database
            .write_two_view_geometry(1, 2, &original_geometry)
            .unwrap();
        drop(database);
        let before = std::fs::read(&source).unwrap();

        let report =
            benchmark_match_pairs(&source, 1, None, 2, &crate::MatchFeaturesOptions::default())
                .unwrap();

        assert_eq!(report.pair_count, 2);
        assert_eq!(report.repetitions, 2);
        assert_eq!(report.runs.len(), 2);
        assert!(report
            .runs
            .iter()
            .all(|run| run.timings.backend_initialization_seconds == 0.0));
        assert!(report
            .runs
            .iter()
            .all(|run| run.result_fingerprint.len() == 64));
        assert_eq!(
            report
                .runs
                .iter()
                .map(|run| run.result_fingerprint.as_str())
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            1
        );
        assert_eq!(std::fs::read(&source).unwrap(), before);
        let database = ColmapDatabase::open_read_only(&source).unwrap();
        assert_eq!(database.read_matches(1, 2).unwrap(), original_matches);
        assert_eq!(
            database.read_two_view_geometry(1, 2).unwrap(),
            original_geometry
        );
    }

    #[test]
    fn match_pair_benchmark_retains_independent_artifact_databases() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("database.db");
        let artifacts = dir.path().join("benchmark-artifacts");
        write_empty_feature_database(&source);
        let marker = rusqlite::Connection::open(&source).unwrap();
        marker
            .execute_batch(
                "CREATE TABLE benchmark_marker(value INTEGER NOT NULL);
                 INSERT INTO benchmark_marker(value) VALUES(42);",
            )
            .unwrap();
        drop(marker);
        let before = std::fs::read(&source).unwrap();

        let report = benchmark_match_pairs_with_artifacts(
            &source,
            1,
            Some(1),
            2,
            &crate::MatchFeaturesOptions::default(),
            &artifacts,
        )
        .unwrap();

        assert_eq!(report.runs.len(), 2);
        let snapshot = artifacts.join("source-snapshot.db");
        let run1 = artifacts.join("run-1.db");
        let run2 = artifacts.join("run-2.db");
        for path in [&snapshot, &run1, &run2] {
            assert!(
                path.is_file(),
                "missing retained database {}",
                path.display()
            );
        }
        assert_eq!(std::fs::read(&source).unwrap(), before);

        let first_run = rusqlite::Connection::open(&run1).unwrap();
        first_run
            .execute("UPDATE benchmark_marker SET value = 99", [])
            .unwrap();
        drop(first_run);
        let marker_value = |path: &Path| {
            rusqlite::Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
                .unwrap()
                .query_row("SELECT value FROM benchmark_marker", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap()
        };
        assert_eq!(marker_value(&run1), 99);
        for path in [&source, &snapshot, &run2] {
            assert_eq!(
                marker_value(path),
                42,
                "database changed: {}",
                path.display()
            );
        }
    }

    #[test]
    fn match_pair_benchmark_rejects_existing_artifacts_path() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("database.db");
        write_empty_feature_database(&source);
        let existing_dir = dir.path().join("existing-dir");
        std::fs::create_dir(&existing_dir).unwrap();
        let existing_file = dir.path().join("existing-file");
        std::fs::write(&existing_file, b"do not overwrite").unwrap();

        for artifacts in [&existing_dir, &existing_file] {
            let result = benchmark_match_pairs_with_artifacts(
                &source,
                1,
                Some(1),
                1,
                &crate::MatchFeaturesOptions::default(),
                artifacts,
            );
            assert!(
                result.is_err(),
                "accepted existing artifacts path {}",
                artifacts.display()
            );
        }
        assert!(existing_dir.read_dir().unwrap().next().is_none());
        assert_eq!(std::fs::read(existing_file).unwrap(), b"do not overwrite");
    }

    #[test]
    fn match_pair_benchmark_run_deserializes_without_fingerprint() {
        let legacy_run = serde_json::json!({
            "run_index": 1,
            "backend": "cpu",
            "pair_count": 2,
            "matched_pairs": 2,
            "verified_pairs": 1,
            "total_matches": 16,
            "matching_seconds": 0.25,
            "timings": MatchFeaturesTimingReport::default(),
        });

        let run: MatchPairBenchmarkRun = serde_json::from_value(legacy_run).unwrap();

        assert_eq!(run.result_fingerprint, "");
    }

    #[test]
    fn match_pair_benchmark_snapshots_source_with_uncheckpointed_wal() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("database.db");
        write_empty_feature_database(&source);
        let writer = rusqlite::Connection::open(&source).unwrap();
        writer.pragma_update(None, "journal_mode", "WAL").unwrap();
        writer
            .execute_batch(
                "CREATE TABLE benchmark_marker(value INTEGER NOT NULL);
                 INSERT INTO benchmark_marker(value) VALUES(42);",
            )
            .unwrap();
        assert!(dir.path().join("database.db-wal").metadata().unwrap().len() > 0);
        let before = std::fs::read(&source).unwrap();

        let report = benchmark_match_pairs(
            &source,
            1,
            Some(1),
            1,
            &crate::MatchFeaturesOptions::default(),
        )
        .unwrap();

        assert_eq!(report.pair_count, 1);
        assert_eq!(report.runs.len(), 1);
        assert_eq!(std::fs::read(&source).unwrap(), before);
        drop(writer);
    }

    #[test]
    fn match_pair_benchmark_run_databases_start_from_independent_snapshots() {
        let dir = tempfile::tempdir().unwrap();
        let snapshot = dir.path().join("snapshot.db");
        write_empty_feature_database(&snapshot);
        let matches = vec![FeatureMatch::new(3, 7)];
        let geometry = ColmapTwoViewGeometry {
            config: 2,
            inlier_matches: matches.clone(),
            ..ColmapTwoViewGeometry::default()
        };
        let database = ColmapDatabase::open(&snapshot).unwrap();
        database.write_matches(1, 2, &matches).unwrap();
        database.write_two_view_geometry(1, 2, &geometry).unwrap();
        drop(database);

        let run0 = copy_benchmark_snapshot_for_run(&snapshot, dir.path(), 0).unwrap();
        let run1 = copy_benchmark_snapshot_for_run(&snapshot, dir.path(), 1).unwrap();
        let database = ColmapDatabase::open(&run0).unwrap();
        database.clear_matches().unwrap();
        database.clear_two_view_geometries().unwrap();
        drop(database);

        assert!(ColmapDatabase::open_read_only(&run0)
            .unwrap()
            .read_matches(1, 2)
            .unwrap()
            .is_empty());
        for path in [&snapshot, &run1] {
            let database = ColmapDatabase::open_read_only(path).unwrap();
            assert_eq!(database.read_matches(1, 2).unwrap(), matches);
            assert_eq!(database.read_two_view_geometry(1, 2).unwrap(), geometry);
        }
    }
}
