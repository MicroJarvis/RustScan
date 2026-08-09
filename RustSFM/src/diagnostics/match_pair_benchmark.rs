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

fn sqlite_sidecar_path(database: &Path, suffix: &str) -> PathBuf {
    let mut path = database.as_os_str().to_os_string();
    path.push(suffix);
    PathBuf::from(path)
}

fn reject_uncheckpointed_wal(database: &Path) -> Result<()> {
    let wal = sqlite_sidecar_path(database, "-wal");
    if wal
        .metadata()
        .map(|metadata| metadata.len() > 0)
        .unwrap_or(false)
    {
        bail!(
            "benchmark source has a non-empty SQLite WAL at {}; checkpoint it before benchmarking",
            wal.display()
        );
    }
    Ok(())
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
    if window == 0 {
        bail!("match-pair benchmark window must be greater than zero");
    }
    if pair_limit == Some(0) {
        bail!("match-pair benchmark pair limit must be greater than zero");
    }
    if repetitions == 0 {
        bail!("match-pair benchmark repetitions must be greater than zero");
    }
    reject_uncheckpointed_wal(source_database)?;

    let source = ColmapDatabase::open_read_only(source_database)
        .with_context(|| format!("open benchmark source {}", source_database.display()))?;
    let mut images = source.read_all_images()?;
    drop(source);
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

    let work_dir = tempfile::tempdir().context("create match-pair benchmark directory")?;
    let working_database = work_dir.path().join("database.db");
    let copy_started = Instant::now();
    let copied_database_bytes = std::fs::copy(source_database, &working_database)
        .with_context(|| format!("copy benchmark database {}", source_database.display()))?;
    let database_copy_seconds = copy_started.elapsed().as_secs_f64();

    let mut run_options = options.clone();
    run_options.clear_existing = true;
    run_options.use_existing_matches = false;
    let session = ExplicitPairMatchingSession::new(&run_options)?;
    let session_initialization_seconds = session.initialization_seconds();
    let mut runs = Vec::with_capacity(repetitions);
    for run_index in 0..repetitions {
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
    use crate::database::{
        ColmapDatabase, ColmapDatabaseCamera, ColmapDatabaseImage, ColmapDescriptors,
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
        assert_eq!(std::fs::read(&source).unwrap(), before);
    }

    #[test]
    fn match_pair_benchmark_rejects_source_with_uncheckpointed_wal() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("database.db");
        write_empty_feature_database(&source);
        std::fs::write(dir.path().join("database.db-wal"), b"pending").unwrap();

        let error = benchmark_match_pairs(
            &source,
            1,
            Some(1),
            1,
            &crate::MatchFeaturesOptions::default(),
        )
        .unwrap_err();

        assert!(error.to_string().contains("non-empty SQLite WAL"));
    }
}
