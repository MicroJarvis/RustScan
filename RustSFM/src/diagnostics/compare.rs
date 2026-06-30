use crate::colmap::{
    camera_center, read_colmap_images, read_colmap_points3d, read_colmap_poses,
    world_to_camera_rotation, ColmapPose,
};
use crate::correspondence_graph::{pair_id_to_image_pair, ImageId};
use crate::database::{
    ColmapDatabase, ColmapTwoViewGeometry, COLMAP_TWO_VIEW_CALIBRATED,
    COLMAP_TWO_VIEW_CALIBRATED_RIG, COLMAP_TWO_VIEW_DEGENERATE, COLMAP_TWO_VIEW_MULTIPLE,
    COLMAP_TWO_VIEW_PANORAMIC, COLMAP_TWO_VIEW_PLANAR, COLMAP_TWO_VIEW_PLANAR_OR_PANORAMIC,
    COLMAP_TWO_VIEW_UNCALIBRATED, COLMAP_TWO_VIEW_UNDEFINED, COLMAP_TWO_VIEW_WATERMARK,
};
use anyhow::{bail, Context, Result};
use nalgebra::{
    Matrix3, Matrix4, Quaternion, Rotation3, SymmetricEigen, UnitQuaternion, Vector3, Vector4,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ErrorStats {
    pub mean: f64,
    pub rmse: f64,
    pub max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerImageError {
    pub image_name: String,
    pub translation_error: f64,
    pub rotation_error_deg: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerAdjacentError {
    pub left_image_name: String,
    pub right_image_name: String,
    pub relative_rotation_error_deg: f64,
    pub relative_translation_angle_deg: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareReport {
    pub common_images: usize,
    pub similarity_scale: f64,
    pub translation_error: ErrorStats,
    pub rotation_error_deg: ErrorStats,
    pub adjacent_relative_rotation_error_deg: ErrorStats,
    pub adjacent_relative_translation_angle_deg: ErrorStats,
    pub per_image: Vec<PerImageError>,
    pub per_adjacent: Vec<PerAdjacentError>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompareStage {
    Features,
    Matches,
    TwoView,
    Registration,
    Tracks,
    Ba,
}

impl CompareStage {
    pub const ALL: [CompareStage; 6] = [
        CompareStage::Features,
        CompareStage::Matches,
        CompareStage::TwoView,
        CompareStage::Registration,
        CompareStage::Tracks,
        CompareStage::Ba,
    ];
}

pub fn parse_compare_stages(stages: &[String]) -> Result<Vec<CompareStage>> {
    if stages.is_empty() || stages.iter().any(|stage| stage.eq_ignore_ascii_case("all")) {
        return Ok(CompareStage::ALL.to_vec());
    }
    let mut parsed = Vec::with_capacity(stages.len());
    for stage in stages {
        let value = match stage.to_ascii_lowercase().as_str() {
            "features" | "feature" => CompareStage::Features,
            "matches" | "match" => CompareStage::Matches,
            "twoview" | "two_view" | "two-view" | "two_view_geometry" => CompareStage::TwoView,
            "registration" | "register" => CompareStage::Registration,
            "tracks" | "track" | "points3d" | "points" => CompareStage::Tracks,
            "ba" | "poses" | "pose" => CompareStage::Ba,
            other => bail!("unknown compare stage: {other}"),
        };
        if !parsed.contains(&value) {
            parsed.push(value);
        }
    }
    Ok(parsed)
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CountDiffStats {
    pub mean_abs_diff: f64,
    pub max_abs_diff: usize,
    pub pct_exact: f64,
    pub pct_within_1: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerImageKeypointDiff {
    pub image_name: String,
    pub reference_keypoints: usize,
    pub candidate_keypoints: usize,
    pub abs_diff: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeaturesCompareReport {
    pub common_images: usize,
    pub reference_only_images: usize,
    pub candidate_only_images: usize,
    pub reference_total_keypoints: usize,
    pub candidate_total_keypoints: usize,
    pub keypoint_count_diff: CountDiffStats,
    pub per_image: Vec<PerImageKeypointDiff>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchesCompareReport {
    pub reference_pairs: usize,
    pub candidate_pairs: usize,
    pub common_pairs: usize,
    pub reference_only_pairs: usize,
    pub candidate_only_pairs: usize,
    pub reference_total_matches: usize,
    pub candidate_total_matches: usize,
    pub common_pair_match_count_diff: CountDiffStats,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConfigHistogram {
    pub counts: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoViewConfigMismatch {
    pub image_name1: String,
    pub image_name2: String,
    pub reference_config: String,
    pub candidate_config: String,
    pub reference_inliers: usize,
    pub candidate_inliers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoViewCompareReport {
    pub reference_pairs: usize,
    pub candidate_pairs: usize,
    pub common_pairs: usize,
    pub config_agreement_rate: f64,
    pub reference_config_histogram: ConfigHistogram,
    pub candidate_config_histogram: ConfigHistogram,
    pub config_confusion: BTreeMap<String, BTreeMap<String, usize>>,
    pub config_mismatches: Vec<TwoViewConfigMismatch>,
    pub inlier_count_diff: CountDiffStats,
    pub inlier_set_overlap: OverlapRateStats,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OverlapRateStats {
    pub mean_rate: f64,
    pub min_rate: f64,
    pub pct_exact: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistrationCompareReport {
    pub reference_registered: usize,
    pub candidate_registered: usize,
    pub common_registered: usize,
    pub reference_only: Vec<String>,
    pub candidate_only: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrackLengthHistogram {
    pub length_2: usize,
    pub length_3_4: usize,
    pub length_5_9: usize,
    pub length_10_plus: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracksCompareReport {
    pub reference_points: usize,
    pub candidate_points: usize,
    pub reference_mean_track_length: f64,
    pub candidate_mean_track_length: f64,
    pub reference_mean_reprojection_error: f64,
    pub candidate_mean_reprojection_error: f64,
    pub reference_total_observations: usize,
    pub candidate_total_observations: usize,
    pub reference_track_length_histogram: TrackLengthHistogram,
    pub candidate_track_length_histogram: TrackLengthHistogram,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareStagesReport {
    pub reference_sparse: PathBuf,
    pub candidate_sparse: PathBuf,
    pub reference_database: Option<PathBuf>,
    pub candidate_database: Option<PathBuf>,
    pub stages: Vec<CompareStage>,
    pub features: Option<FeaturesCompareReport>,
    pub matches: Option<MatchesCompareReport>,
    pub twoview: Option<TwoViewCompareReport>,
    pub registration: Option<RegistrationCompareReport>,
    pub tracks: Option<TracksCompareReport>,
    pub ba: Option<CompareReport>,
}

pub fn compare_colmap_stages(
    reference_sparse: &Path,
    candidate_sparse: &Path,
    reference_database: Option<&Path>,
    candidate_database: Option<&Path>,
    stages: &[CompareStage],
) -> Result<CompareStagesReport> {
    let needs_database = stages.iter().any(|stage| {
        matches!(
            stage,
            CompareStage::Features | CompareStage::Matches | CompareStage::TwoView
        )
    });
    let resolved_reference_database = if needs_database {
        Some(resolve_compare_database_path(
            reference_sparse,
            reference_database,
        )?)
    } else {
        reference_database.map(Path::to_path_buf)
    };
    let resolved_candidate_database = if needs_database {
        Some(resolve_compare_database_path(
            candidate_sparse,
            candidate_database,
        )?)
    } else {
        candidate_database.map(Path::to_path_buf)
    };

    let mut report = CompareStagesReport {
        reference_sparse: reference_sparse.to_path_buf(),
        candidate_sparse: candidate_sparse.to_path_buf(),
        reference_database: resolved_reference_database.clone(),
        candidate_database: resolved_candidate_database.clone(),
        stages: stages.to_vec(),
        features: None,
        matches: None,
        twoview: None,
        registration: None,
        tracks: None,
        ba: None,
    };

    for stage in stages {
        match stage {
            CompareStage::Features => {
                let reference_db = resolved_reference_database.as_ref().with_context(|| {
                    format!(
                        "features stage requires a database for reference sparse model {}",
                        reference_sparse.display()
                    )
                })?;
                let candidate_db = resolved_candidate_database.as_ref().with_context(|| {
                    format!(
                        "features stage requires a database for candidate sparse model {}",
                        candidate_sparse.display()
                    )
                })?;
                report.features = Some(compare_features(reference_db, candidate_db)?);
            }
            CompareStage::Matches => {
                let reference_db = resolved_reference_database.as_ref().with_context(|| {
                    format!(
                        "matches stage requires a database for reference sparse model {}",
                        reference_sparse.display()
                    )
                })?;
                let candidate_db = resolved_candidate_database.as_ref().with_context(|| {
                    format!(
                        "matches stage requires a database for candidate sparse model {}",
                        candidate_sparse.display()
                    )
                })?;
                report.matches = Some(compare_matches(reference_db, candidate_db)?);
            }
            CompareStage::TwoView => {
                let reference_db = resolved_reference_database.as_ref().with_context(|| {
                    format!(
                        "twoview stage requires a database for reference sparse model {}",
                        reference_sparse.display()
                    )
                })?;
                let candidate_db = resolved_candidate_database.as_ref().with_context(|| {
                    format!(
                        "twoview stage requires a database for candidate sparse model {}",
                        candidate_sparse.display()
                    )
                })?;
                report.twoview = Some(compare_two_view(reference_db, candidate_db)?);
            }
            CompareStage::Registration => {
                report.registration =
                    Some(compare_registration(reference_sparse, candidate_sparse)?);
            }
            CompareStage::Tracks => {
                report.tracks = Some(compare_tracks(reference_sparse, candidate_sparse)?);
            }
            CompareStage::Ba => {
                report.ba = Some(compare_colmap(reference_sparse, candidate_sparse)?);
            }
        }
    }
    Ok(report)
}

pub fn resolve_compare_database_path(
    sparse_root: &Path,
    explicit: Option<&Path>,
) -> Result<PathBuf> {
    if let Some(path) = explicit {
        if !path.exists() {
            bail!("database path does not exist: {}", path.display());
        }
        return Ok(path.to_path_buf());
    }
    for candidate in default_compare_database_candidates(sparse_root) {
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    bail!(
        "could not locate database.db for sparse model {}; pass --reference-database / --candidate-database",
        sparse_root.display()
    )
}

fn default_compare_database_candidates(sparse_root: &Path) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let sparse = resolve_sparse_search_root(sparse_root);
    push_unique_path(&mut candidates, sparse.join("database.db"));
    push_unique_path(&mut candidates, sparse.join("images").join("database.db"));
    if let Some(parent) = sparse.parent() {
        push_unique_path(&mut candidates, parent.join("database.db"));
        push_unique_path(&mut candidates, parent.join("images").join("database.db"));
    }
    candidates
}

fn resolve_sparse_search_root(sparse_root: &Path) -> PathBuf {
    let file_name = sparse_root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if file_name == "0" || file_name == "text" || file_name == "binary" {
        sparse_root
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| sparse_root.to_path_buf())
    } else {
        sparse_root.to_path_buf()
    }
}

fn push_unique_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.iter().any(|existing| existing == &path) {
        paths.push(path);
    }
}

fn compare_features(reference_db: &Path, candidate_db: &Path) -> Result<FeaturesCompareReport> {
    let reference = ColmapDatabase::open(reference_db)?;
    let candidate = ColmapDatabase::open(candidate_db)?;
    let reference_counts = keypoint_counts_by_name(&reference)?;
    let candidate_counts = keypoint_counts_by_name(&candidate)?;
    compare_feature_counts(&reference_counts, &candidate_counts)
}

fn compare_matches(reference_db: &Path, candidate_db: &Path) -> Result<MatchesCompareReport> {
    let reference = ColmapDatabase::open(reference_db)?;
    let candidate = ColmapDatabase::open(candidate_db)?;
    let reference_counts = raw_match_counts_by_name(&reference)?;
    let candidate_counts = raw_match_counts_by_name(&candidate)?;
    compare_pair_count_maps(&reference_counts, &candidate_counts)
}

fn compare_two_view(reference_db: &Path, candidate_db: &Path) -> Result<TwoViewCompareReport> {
    let reference = ColmapDatabase::open(reference_db)?;
    let candidate = ColmapDatabase::open(candidate_db)?;
    let reference_geometries = two_view_by_name(&reference)?;
    let candidate_geometries = two_view_by_name(&candidate)?;
    let reference_names = image_id_to_name_map(&reference)?;
    let candidate_names = image_id_to_name_map(&candidate)?;
    compare_two_view_maps(
        &reference_geometries,
        &candidate_geometries,
        &reference_names,
        &candidate_names,
    )
}

fn compare_registration(
    reference_sparse: &Path,
    candidate_sparse: &Path,
) -> Result<RegistrationCompareReport> {
    let reference_names = registered_image_names(reference_sparse)?;
    let candidate_names = registered_image_names(candidate_sparse)?;
    let reference_set = reference_names
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let candidate_set = candidate_names
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let mut reference_only = reference_set
        .difference(&candidate_set)
        .cloned()
        .collect::<Vec<_>>();
    let mut candidate_only = candidate_set
        .difference(&reference_set)
        .cloned()
        .collect::<Vec<_>>();
    reference_only.sort_unstable();
    candidate_only.sort_unstable();
    Ok(RegistrationCompareReport {
        reference_registered: reference_names.len(),
        candidate_registered: candidate_names.len(),
        common_registered: reference_set.intersection(&candidate_set).count(),
        reference_only,
        candidate_only,
    })
}

fn compare_tracks(reference_sparse: &Path, candidate_sparse: &Path) -> Result<TracksCompareReport> {
    let reference = track_summary(reference_sparse)?;
    let candidate = track_summary(candidate_sparse)?;
    Ok(TracksCompareReport {
        reference_points: reference.num_points,
        candidate_points: candidate.num_points,
        reference_mean_track_length: reference.mean_track_length,
        candidate_mean_track_length: candidate.mean_track_length,
        reference_mean_reprojection_error: reference.mean_reprojection_error,
        candidate_mean_reprojection_error: candidate.mean_reprojection_error,
        reference_total_observations: reference.total_observations,
        candidate_total_observations: candidate.total_observations,
        reference_track_length_histogram: reference.track_length_histogram,
        candidate_track_length_histogram: candidate.track_length_histogram,
    })
}

fn registered_image_names(sparse_root: &Path) -> Result<Vec<String>> {
    let mut names = read_colmap_images(sparse_root)?
        .into_iter()
        .map(|image| image.name)
        .collect::<Vec<_>>();
    names.sort_unstable();
    Ok(names)
}

struct TrackSummary {
    num_points: usize,
    mean_track_length: f64,
    mean_reprojection_error: f64,
    total_observations: usize,
    track_length_histogram: TrackLengthHistogram,
}

fn track_length_histogram(track_lengths: &[usize]) -> TrackLengthHistogram {
    let mut histogram = TrackLengthHistogram::default();
    for length in track_lengths {
        match length {
            2 => histogram.length_2 += 1,
            3..=4 => histogram.length_3_4 += 1,
            5..=9 => histogram.length_5_9 += 1,
            _ => histogram.length_10_plus += 1,
        }
    }
    histogram
}

fn track_summary(sparse_root: &Path) -> Result<TrackSummary> {
    let points = read_colmap_points3d(sparse_root)?;
    let num_points = points.len();
    if num_points == 0 {
        return Ok(TrackSummary {
            num_points: 0,
            mean_track_length: 0.0,
            mean_reprojection_error: 0.0,
            total_observations: 0,
            track_length_histogram: TrackLengthHistogram::default(),
        });
    }
    let track_lengths = points
        .iter()
        .map(|point| point.track.len())
        .collect::<Vec<_>>();
    let total_observations = track_lengths.iter().sum::<usize>();
    let mean_track_length = total_observations as f64 / num_points as f64;
    let mean_reprojection_error =
        points.iter().map(|point| point.error).sum::<f64>() / num_points as f64;
    Ok(TrackSummary {
        num_points,
        mean_track_length,
        mean_reprojection_error,
        total_observations,
        track_length_histogram: track_length_histogram(&track_lengths),
    })
}

fn keypoint_counts_by_name(db: &ColmapDatabase) -> Result<HashMap<String, usize>> {
    let id_to_name = image_id_to_name_map(db)?;
    db.read_keypoint_counts()?
        .into_iter()
        .map(|(image_id, count)| {
            let name = id_to_name
                .get(&image_id)
                .with_context(|| format!("missing image name for image_id={image_id}"))?
                .clone();
            Ok((name, count))
        })
        .collect()
}

fn raw_match_counts_by_name(db: &ColmapDatabase) -> Result<HashMap<PairNameKey, usize>> {
    let id_to_name = image_id_to_name_map(db)?;
    let mut counts = HashMap::new();
    for (pair_id, matches) in db.read_all_matches()? {
        let (image_id1, image_id2) =
            pair_id_to_image_pair(pair_id).map_err(|err| anyhow::anyhow!("{err:?}"))?;
        let name1 = id_to_name
            .get(&image_id1)
            .with_context(|| format!("missing image name for image_id={image_id1}"))?;
        let name2 = id_to_name
            .get(&image_id2)
            .with_context(|| format!("missing image name for image_id={image_id2}"))?;
        counts.insert(canonical_pair_key(name1, name2), matches.len());
    }
    Ok(counts)
}

fn two_view_by_name(
    db: &ColmapDatabase,
) -> Result<HashMap<PairNameKey, (ColmapTwoViewGeometry, ImageId, ImageId)>> {
    let id_to_name = image_id_to_name_map(db)?;
    let mut geometries = HashMap::new();
    for (pair_id, geometry) in db.read_two_view_geometries()? {
        let (image_id1, image_id2) =
            pair_id_to_image_pair(pair_id).map_err(|err| anyhow::anyhow!("{err:?}"))?;
        let name1 = id_to_name
            .get(&image_id1)
            .with_context(|| format!("missing image name for image_id={image_id1}"))?;
        let name2 = id_to_name
            .get(&image_id2)
            .with_context(|| format!("missing image name for image_id={image_id2}"))?;
        geometries.insert(
            canonical_pair_key(name1, name2),
            (geometry, image_id1, image_id2),
        );
    }
    Ok(geometries)
}

fn canonical_inlier_match_set(
    geometry: &ColmapTwoViewGeometry,
    name_key: &PairNameKey,
    image_id1: ImageId,
    image_id2: ImageId,
    id_to_name: &HashMap<ImageId, String>,
) -> std::collections::HashSet<(u32, u32)> {
    let lo_id = image_id1.min(image_id2);
    let hi_id = image_id1.max(image_id2);
    let lo_name = id_to_name
        .get(&lo_id)
        .expect("canonical inlier set requires image names");
    let mut set = std::collections::HashSet::new();
    for m in &geometry.inlier_matches {
        let (first_idx, second_idx) = if lo_name == &name_key.0 {
            (m.point2d_idx1, m.point2d_idx2)
        } else {
            (m.point2d_idx2, m.point2d_idx1)
        };
        set.insert((first_idx, second_idx));
    }
    let _ = hi_id;
    set
}

fn image_id_to_name_map(db: &ColmapDatabase) -> Result<HashMap<ImageId, String>> {
    db.read_all_images()?
        .into_iter()
        .map(|image| Ok((image.image_id, image.name)))
        .collect()
}

type PairNameKey = (String, String);

fn canonical_pair_key(name1: &str, name2: &str) -> PairNameKey {
    if name1 <= name2 {
        (name1.to_string(), name2.to_string())
    } else {
        (name2.to_string(), name1.to_string())
    }
}

pub fn compare_feature_counts(
    reference: &HashMap<String, usize>,
    candidate: &HashMap<String, usize>,
) -> Result<FeaturesCompareReport> {
    let reference_names = reference
        .keys()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let candidate_names = candidate
        .keys()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let common_names = reference_names
        .intersection(&candidate_names)
        .cloned()
        .collect::<Vec<_>>();
    let mut per_image = Vec::with_capacity(common_names.len());
    let mut diffs = Vec::with_capacity(common_names.len());
    let mut reference_total_keypoints = 0usize;
    let mut candidate_total_keypoints = 0usize;
    for name in &common_names {
        let reference_keypoints = *reference.get(name).unwrap_or(&0);
        let candidate_keypoints = *candidate.get(name).unwrap_or(&0);
        reference_total_keypoints += reference_keypoints;
        candidate_total_keypoints += candidate_keypoints;
        let abs_diff = reference_keypoints.abs_diff(candidate_keypoints);
        diffs.push(abs_diff);
        per_image.push(PerImageKeypointDiff {
            image_name: name.clone(),
            reference_keypoints,
            candidate_keypoints,
            abs_diff,
        });
    }
    per_image.sort_unstable_by(|left, right| right.abs_diff.cmp(&left.abs_diff));
    Ok(FeaturesCompareReport {
        common_images: common_names.len(),
        reference_only_images: reference_names.difference(&candidate_names).count(),
        candidate_only_images: candidate_names.difference(&reference_names).count(),
        reference_total_keypoints,
        candidate_total_keypoints,
        keypoint_count_diff: count_diff_stats(&diffs),
        per_image,
    })
}

fn compare_pair_count_maps(
    reference: &HashMap<PairNameKey, usize>,
    candidate: &HashMap<PairNameKey, usize>,
) -> Result<MatchesCompareReport> {
    let reference_keys = reference
        .keys()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let candidate_keys = candidate
        .keys()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let common_keys = reference_keys
        .intersection(&candidate_keys)
        .cloned()
        .collect::<Vec<_>>();
    let mut diffs = Vec::with_capacity(common_keys.len());
    let mut reference_total_matches = 0usize;
    let mut candidate_total_matches = 0usize;
    for key in &common_keys {
        let reference_count = *reference.get(key).unwrap_or(&0);
        let candidate_count = *candidate.get(key).unwrap_or(&0);
        reference_total_matches += reference_count;
        candidate_total_matches += candidate_count;
        diffs.push(reference_count.abs_diff(candidate_count));
    }
    Ok(MatchesCompareReport {
        reference_pairs: reference.len(),
        candidate_pairs: candidate.len(),
        common_pairs: common_keys.len(),
        reference_only_pairs: reference_keys.difference(&candidate_keys).count(),
        candidate_only_pairs: candidate_keys.difference(&reference_keys).count(),
        reference_total_matches,
        candidate_total_matches,
        common_pair_match_count_diff: count_diff_stats(&diffs),
    })
}

fn compare_two_view_maps(
    reference: &HashMap<PairNameKey, (ColmapTwoViewGeometry, ImageId, ImageId)>,
    candidate: &HashMap<PairNameKey, (ColmapTwoViewGeometry, ImageId, ImageId)>,
    reference_names: &HashMap<ImageId, String>,
    candidate_names: &HashMap<ImageId, String>,
) -> Result<TwoViewCompareReport> {
    let reference_keys = reference
        .keys()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let candidate_keys = candidate
        .keys()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let common_keys = reference_keys
        .intersection(&candidate_keys)
        .cloned()
        .collect::<Vec<_>>();
    let mut config_matches = 0usize;
    let mut inlier_diffs = Vec::with_capacity(common_keys.len());
    let mut inlier_overlap_rates = Vec::with_capacity(common_keys.len());
    let mut config_confusion: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    let mut config_mismatches = Vec::new();
    for key in &common_keys {
        let (reference_geometry, reference_id1, reference_id2) =
            reference.get(key).expect("common key");
        let (candidate_geometry, candidate_id1, candidate_id2) =
            candidate.get(key).expect("common key");
        let reference_config = two_view_config_name(reference_geometry.config).to_string();
        let candidate_config = two_view_config_name(candidate_geometry.config).to_string();
        *config_confusion
            .entry(reference_config.clone())
            .or_default()
            .entry(candidate_config.clone())
            .or_insert(0) += 1;
        if reference_geometry.config == candidate_geometry.config {
            config_matches += 1;
        } else {
            config_mismatches.push(TwoViewConfigMismatch {
                image_name1: key.0.clone(),
                image_name2: key.1.clone(),
                reference_config,
                candidate_config,
                reference_inliers: reference_geometry.inlier_matches.len(),
                candidate_inliers: candidate_geometry.inlier_matches.len(),
            });
        }
        inlier_diffs.push(
            reference_geometry
                .inlier_matches
                .len()
                .abs_diff(candidate_geometry.inlier_matches.len()),
        );
        let reference_set = canonical_inlier_match_set(
            reference_geometry,
            key,
            *reference_id1,
            *reference_id2,
            reference_names,
        );
        let candidate_set = canonical_inlier_match_set(
            candidate_geometry,
            key,
            *candidate_id1,
            *candidate_id2,
            candidate_names,
        );
        let overlap = if reference_set.is_empty() {
            0.0
        } else {
            reference_set.intersection(&candidate_set).count() as f64 / reference_set.len() as f64
        };
        inlier_overlap_rates.push(overlap);
    }
    config_mismatches.sort_by(|left, right| {
        right
            .reference_inliers
            .abs_diff(right.candidate_inliers)
            .cmp(&left.reference_inliers.abs_diff(left.candidate_inliers))
    });
    config_mismatches.truncate(32);
    Ok(TwoViewCompareReport {
        reference_pairs: reference.len(),
        candidate_pairs: candidate.len(),
        common_pairs: common_keys.len(),
        config_agreement_rate: if common_keys.is_empty() {
            0.0
        } else {
            config_matches as f64 / common_keys.len() as f64
        },
        reference_config_histogram: config_histogram(
            reference.values().map(|(geometry, _, _)| geometry),
        ),
        candidate_config_histogram: config_histogram(
            candidate.values().map(|(geometry, _, _)| geometry),
        ),
        config_confusion,
        config_mismatches,
        inlier_count_diff: count_diff_stats(&inlier_diffs),
        inlier_set_overlap: overlap_rate_stats(&inlier_overlap_rates),
    })
}

fn config_histogram<'a>(
    geometries: impl Iterator<Item = &'a ColmapTwoViewGeometry>,
) -> ConfigHistogram {
    let mut counts = BTreeMap::new();
    for geometry in geometries {
        *counts
            .entry(two_view_config_name(geometry.config).to_string())
            .or_insert(0) += 1;
    }
    ConfigHistogram { counts }
}

fn two_view_config_name(config: i32) -> &'static str {
    match config {
        COLMAP_TWO_VIEW_UNDEFINED => "undefined",
        COLMAP_TWO_VIEW_DEGENERATE => "degenerate",
        COLMAP_TWO_VIEW_CALIBRATED => "calibrated",
        COLMAP_TWO_VIEW_UNCALIBRATED => "uncalibrated",
        COLMAP_TWO_VIEW_PLANAR => "planar",
        COLMAP_TWO_VIEW_PANORAMIC => "panoramic",
        COLMAP_TWO_VIEW_PLANAR_OR_PANORAMIC => "planar_or_panoramic",
        COLMAP_TWO_VIEW_WATERMARK => "watermark",
        COLMAP_TWO_VIEW_MULTIPLE => "multiple",
        COLMAP_TWO_VIEW_CALIBRATED_RIG => "calibrated_rig",
        other => {
            if other < 0 {
                "invalid"
            } else {
                "unknown"
            }
        }
    }
}

fn count_diff_stats(diffs: &[usize]) -> CountDiffStats {
    if diffs.is_empty() {
        return CountDiffStats::default();
    }
    let mean_abs_diff = diffs.iter().sum::<usize>() as f64 / diffs.len() as f64;
    let max_abs_diff = *diffs.iter().max().unwrap_or(&0);
    let exact = diffs.iter().filter(|diff| **diff == 0).count();
    let within_1 = diffs.iter().filter(|diff| **diff <= 1).count();
    let n = diffs.len() as f64;
    CountDiffStats {
        mean_abs_diff,
        max_abs_diff,
        pct_exact: exact as f64 / n,
        pct_within_1: within_1 as f64 / n,
    }
}

fn overlap_rate_stats(rates: &[f64]) -> OverlapRateStats {
    if rates.is_empty() {
        return OverlapRateStats::default();
    }
    let mean_rate = rates.iter().sum::<f64>() / rates.len() as f64;
    let min_rate = rates
        .iter()
        .copied()
        .fold(f64::INFINITY, |current, rate| current.min(rate));
    let exact = rates.iter().filter(|rate| **rate >= 1.0 - 1.0e-9).count();
    OverlapRateStats {
        mean_rate,
        min_rate,
        pct_exact: exact as f64 / rates.len() as f64,
    }
}

pub fn compare_colmap(reference: &Path, candidate: &Path) -> Result<CompareReport> {
    let ref_poses = read_colmap_poses(reference)?;
    let cand_poses = read_colmap_poses(candidate)?;
    let ref_by_name = ref_poses
        .iter()
        .map(|p| (p.name.as_str(), p))
        .collect::<HashMap<_, _>>();
    let cand_by_name = cand_poses
        .iter()
        .map(|p| (p.name.as_str(), p))
        .collect::<HashMap<_, _>>();
    let mut names = ref_by_name
        .keys()
        .filter(|name| cand_by_name.contains_key(**name))
        .copied()
        .collect::<Vec<_>>();
    names.sort_unstable();
    if names.len() < 3 {
        bail!("need at least 3 common images for pose comparison");
    }
    let ref_centers = names
        .iter()
        .map(|name| camera_center(ref_by_name[name]))
        .collect::<Vec<_>>();
    let cand_centers = names
        .iter()
        .map(|name| camera_center(cand_by_name[name]))
        .collect::<Vec<_>>();
    let sim = estimate_similarity(&cand_centers, &ref_centers)?;
    let orientation_alignment =
        estimate_orientation_alignment(&names, &ref_by_name, &cand_by_name)?;

    let mut trans_errors = Vec::new();
    let mut rot_errors = Vec::new();
    let mut per_image = Vec::new();
    for name in &names {
        let r = ref_by_name[name];
        let c = cand_by_name[name];
        let aligned_c = sim.scale * sim.rotation * camera_center(c) + sim.translation;
        let translation_error = (aligned_c - camera_center(r)).norm();
        // Camera centers and orientations have independent gauges in a sparse reconstruction.
        // Use a center Sim3 for translation and a quaternion average for the orientation gauge.
        let aligned_rwc = world_to_camera_rotation(c) * orientation_alignment.transpose();
        let rotation_error_deg =
            rotation_angle_deg(world_to_camera_rotation(r).transpose() * aligned_rwc);
        trans_errors.push(translation_error);
        rot_errors.push(rotation_error_deg);
        per_image.push(PerImageError {
            image_name: name.to_string(),
            translation_error,
            rotation_error_deg,
        });
    }
    let mut rel_rot_errors = Vec::new();
    let mut rel_trans_angle_errors = Vec::new();
    let mut per_adjacent = Vec::new();
    for pair in names.windows(2) {
        let r1 = ref_by_name[pair[0]];
        let r2 = ref_by_name[pair[1]];
        let c1 = cand_by_name[pair[0]];
        let c2 = cand_by_name[pair[1]];
        let ref_rel = relative_pose_parts(r1, r2);
        let cand_rel = relative_pose_parts(c1, c2);
        let relative_rotation_error_deg =
            rotation_angle_deg(ref_rel.rotation.transpose() * cand_rel.rotation);
        rel_rot_errors.push(relative_rotation_error_deg);
        let mut relative_translation_angle_deg = None;
        if let (Some(rt), Some(ct)) = (
            ref_rel.translation.try_normalize(1.0e-12),
            cand_rel.translation.try_normalize(1.0e-12),
        ) {
            let angle = rt.dot(&ct).clamp(-1.0, 1.0).acos().to_degrees();
            rel_trans_angle_errors.push(angle);
            relative_translation_angle_deg = Some(angle);
        }
        per_adjacent.push(PerAdjacentError {
            left_image_name: pair[0].to_string(),
            right_image_name: pair[1].to_string(),
            relative_rotation_error_deg,
            relative_translation_angle_deg,
        });
    }
    Ok(CompareReport {
        common_images: per_image.len(),
        similarity_scale: sim.scale,
        translation_error: stats(&trans_errors),
        rotation_error_deg: stats(&rot_errors),
        adjacent_relative_rotation_error_deg: stats(&rel_rot_errors),
        adjacent_relative_translation_angle_deg: stats(&rel_trans_angle_errors),
        per_image,
        per_adjacent,
    })
}

fn estimate_orientation_alignment(
    names: &[&str],
    ref_by_name: &HashMap<&str, &ColmapPose>,
    cand_by_name: &HashMap<&str, &ColmapPose>,
) -> Result<Matrix3<f64>> {
    let mut accum = Matrix4::<f64>::zeros();
    for name in names {
        let ref_r = world_to_camera_rotation(ref_by_name[name]);
        let cand_r = world_to_camera_rotation(cand_by_name[name]);
        // Candidate and reference reconstructions may differ by a global world rotation G:
        // Rcw_ref ~= Rcw_cand * G^T, so each image votes for G ~= Rcw_ref^T * Rcw_cand.
        let gauge = ref_r.transpose() * cand_r;
        let quat = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(gauge))
            .into_inner();
        let mut v = Vector4::new(quat.w, quat.i, quat.j, quat.k);
        if v[0] < 0.0 {
            v = -v;
        }
        accum += v * v.transpose();
    }
    let eig = SymmetricEigen::new(accum);
    let mut best = 0usize;
    for idx in 1..4 {
        if eig.eigenvalues[idx] > eig.eigenvalues[best] {
            best = idx;
        }
    }
    let q = eig.eigenvectors.column(best);
    let quat = Quaternion::new(q[0], q[1], q[2], q[3]).normalize();
    Ok(UnitQuaternion::from_quaternion(quat)
        .to_rotation_matrix()
        .into_inner())
}

struct RelativePoseParts {
    rotation: Matrix3<f64>,
    translation: Vector3<f64>,
}

fn relative_pose_parts(left: &ColmapPose, right: &ColmapPose) -> RelativePoseParts {
    let left_r = world_to_camera_rotation(left);
    let right_r = world_to_camera_rotation(right);
    let left_t = Vector3::new(left.tvec[0], left.tvec[1], left.tvec[2]);
    let right_t = Vector3::new(right.tvec[0], right.tvec[1], right.tvec[2]);
    let rotation = right_r * left_r.transpose();
    let translation = right_t - rotation * left_t;
    RelativePoseParts {
        rotation,
        translation,
    }
}

struct Similarity {
    scale: f64,
    rotation: Matrix3<f64>,
    translation: Vector3<f64>,
}

fn estimate_similarity(
    candidate: &[Vector3<f64>],
    reference: &[Vector3<f64>],
) -> Result<Similarity> {
    let n = candidate.len();
    if n != reference.len() || n < 3 {
        bail!("invalid similarity inputs");
    }
    let n_f = n as f64;
    let cm = candidate.iter().fold(Vector3::zeros(), |a, p| a + p) / n_f;
    let rm = reference.iter().fold(Vector3::zeros(), |a, p| a + p) / n_f;
    let mut cov = Matrix3::zeros();
    let mut var = 0.0;
    for (c, r) in candidate.iter().zip(reference.iter()) {
        let dc = c - cm;
        let dr = r - rm;
        cov += dr * dc.transpose();
        var += dc.norm_squared();
    }
    cov /= n_f;
    var /= n_f;
    let svd = cov.svd(true, true);
    let u = svd.u.context("missing U")?;
    let vt = svd.v_t.context("missing Vt")?;
    let mut d = Matrix3::identity();
    if (u * vt).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    let rotation = u * d * vt;
    let scale = (svd.singular_values[0] * d[(0, 0)]
        + svd.singular_values[1]
        + svd.singular_values[2] * d[(2, 2)])
        / var.max(1.0e-12);
    let translation = rm - scale * rotation * cm;
    Ok(Similarity {
        scale,
        rotation,
        translation,
    })
}

fn rotation_angle_deg(delta: Matrix3<f64>) -> f64 {
    ((delta.trace() - 1.0) * 0.5)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

fn stats(values: &[f64]) -> ErrorStats {
    if values.is_empty() {
        return ErrorStats::default();
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let rmse = (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt();
    let max = values.iter().copied().fold(0.0, f64::max);
    ErrorStats { mean, rmse, max }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colmap::ColmapCamera;
    use crate::correspondence_graph::{image_pair_to_pair_id, FeatureMatch};
    use crate::database::{
        ColmapDatabaseCamera, ColmapDatabaseImage, ColmapKeypoint, COLMAP_TWO_VIEW_CALIBRATED,
    };
    use crate::types::COLMAP_PINHOLE;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn parse_compare_stages_supports_all_and_aliases() {
        let stages = parse_compare_stages(&["all".to_string()]).unwrap();
        assert_eq!(stages, CompareStage::ALL.to_vec());
        let stages = parse_compare_stages(&[
            "features".to_string(),
            "match".to_string(),
            "two-view".to_string(),
            "register".to_string(),
            "points".to_string(),
            "pose".to_string(),
        ])
        .unwrap();
        assert_eq!(stages, CompareStage::ALL.to_vec());
    }

    #[test]
    fn compare_registration_reports_reference_only_images() {
        let reference = tempdir().unwrap();
        let candidate = tempdir().unwrap();
        write_sparse_images_txt(
            reference.path(),
            &[("img_a.jpg", 1), ("img_b.jpg", 2), ("img_c.jpg", 3)],
        );
        write_sparse_images_txt(candidate.path(), &[("img_a.jpg", 1), ("img_b.jpg", 2)]);
        let report = compare_registration(reference.path(), candidate.path()).unwrap();
        assert_eq!(report.reference_registered, 3);
        assert_eq!(report.candidate_registered, 2);
        assert_eq!(report.common_registered, 2);
        assert_eq!(report.reference_only, vec!["img_c.jpg".to_string()]);
        assert!(report.candidate_only.is_empty());
    }

    #[test]
    fn compare_features_uses_image_names_not_ids() {
        let reference_dir = tempdir().unwrap();
        let candidate_dir = tempdir().unwrap();
        let reference_db_path = reference_dir.path().join("database.db");
        let candidate_db_path = candidate_dir.path().join("database.db");
        populate_feature_database(&reference_db_path, 1, "left.jpg", 10);
        populate_feature_database(&candidate_db_path, 99, "left.jpg", 10);
        populate_feature_database(&reference_db_path, 2, "right.jpg", 8);
        populate_feature_database(&candidate_db_path, 7, "right.jpg", 6);
        let report = compare_features(&reference_db_path, &candidate_db_path).unwrap();
        assert_eq!(report.common_images, 2);
        assert_eq!(report.reference_total_keypoints, 18);
        assert_eq!(report.candidate_total_keypoints, 16);
        assert_eq!(report.keypoint_count_diff.max_abs_diff, 2);
        assert_eq!(report.keypoint_count_diff.pct_exact, 0.5);
    }

    #[test]
    fn compare_colmap_stages_runs_sparse_only_stages() {
        let root = Path::new("../test_data/flowers2_colmap/sparse/text");
        if !root.join("images.txt").exists() {
            return;
        }
        let report = compare_colmap_stages(
            root,
            root,
            None,
            None,
            &[
                CompareStage::Registration,
                CompareStage::Tracks,
                CompareStage::Ba,
            ],
        )
        .unwrap();
        let registration = report.registration.expect("registration");
        assert_eq!(
            registration.reference_registered,
            registration.candidate_registered
        );
        assert!(registration.reference_only.is_empty());
        let tracks = report.tracks.expect("tracks");
        assert_eq!(tracks.reference_points, tracks.candidate_points);
        let ba = report.ba.expect("ba");
        assert!(ba.rotation_error_deg.rmse < 1.0e-5);
        assert!(ba.translation_error.rmse < 1.0e-5);
    }

    fn write_sparse_images_txt(root: &Path, images: &[(&str, u32)]) {
        fs::create_dir_all(root).unwrap();
        let mut contents = String::from(
            "# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n",
        );
        for (name, image_id) in images {
            contents.push_str(&format!("{image_id} 1 0 0 0 0 0 0 1 {name}\n\n"));
        }
        fs::write(root.join("images.txt"), contents).unwrap();
        fs::write(
            root.join("cameras.txt"),
            "# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n1 PINHOLE 10 10 1 1 5 5\n",
        )
        .unwrap();
        fs::write(
            root.join("points3D.txt"),
            "# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n",
        )
        .unwrap();
    }

    fn populate_feature_database(path: &Path, image_id: u32, name: &str, num_keypoints: usize) {
        let db = ColmapDatabase::open(path).unwrap();
        if db.read_all_cameras().unwrap().is_empty() {
            db.write_camera(
                &ColmapDatabaseCamera {
                    camera: ColmapCamera {
                        camera_id: 1,
                        model_id: COLMAP_PINHOLE,
                        width: 10,
                        height: 10,
                        params: vec![1.0, 1.0, 5.0, 5.0],
                    },
                    has_prior_focal_length: false,
                },
                true,
            )
            .unwrap();
        }
        db.write_image(
            &ColmapDatabaseImage {
                image_id,
                name: name.to_string(),
                camera_id: 1,
                frame_id: None,
            },
            true,
        )
        .unwrap();
        let keypoints = (0..num_keypoints)
            .map(|idx| ColmapKeypoint::new(idx as f32, idx as f32))
            .collect::<Vec<_>>();
        db.write_keypoints(image_id, &keypoints).unwrap();
        db.close().unwrap();
    }

    #[test]
    fn compare_matches_and_twoview_use_canonical_pair_names() {
        let reference_dir = tempdir().unwrap();
        let candidate_dir = tempdir().unwrap();
        let reference_db_path = reference_dir.path().join("database.db");
        let candidate_db_path = candidate_dir.path().join("database.db");
        populate_match_database(&reference_db_path, 1, "a.jpg", 2, "b.jpg", 5, 4);
        populate_match_database(&candidate_db_path, 9, "b.jpg", 8, "a.jpg", 5, 3);
        let matches = compare_matches(&reference_db_path, &candidate_db_path).unwrap();
        assert_eq!(matches.common_pairs, 1);
        assert_eq!(matches.common_pair_match_count_diff.max_abs_diff, 0);
        let twoview = compare_two_view(&reference_db_path, &candidate_db_path).unwrap();
        assert_eq!(twoview.common_pairs, 1);
        assert_eq!(twoview.config_agreement_rate, 1.0);
        assert_eq!(twoview.inlier_count_diff.max_abs_diff, 1);
    }

    fn populate_match_database(
        path: &Path,
        image_id1: u32,
        name1: &str,
        image_id2: u32,
        name2: &str,
        raw_matches: usize,
        inliers: usize,
    ) {
        populate_feature_database(path, image_id1, name1, 4);
        let db = ColmapDatabase::open(path).unwrap();
        db.write_image(
            &ColmapDatabaseImage {
                image_id: image_id2,
                name: name2.to_string(),
                camera_id: 1,
                frame_id: None,
            },
            true,
        )
        .unwrap();
        let keypoints = (0..4)
            .map(|idx| ColmapKeypoint::new(idx as f32, idx as f32))
            .collect::<Vec<_>>();
        db.write_keypoints(image_id2, &keypoints).unwrap();
        let matches = (0..raw_matches)
            .map(|idx| FeatureMatch {
                point2d_idx1: (idx % 4) as u32,
                point2d_idx2: (idx % 4) as u32,
            })
            .collect::<Vec<_>>();
        db.write_matches(image_id1, image_id2, &matches).unwrap();
        let geometry = ColmapTwoViewGeometry {
            config: COLMAP_TWO_VIEW_CALIBRATED,
            inlier_matches: matches.into_iter().take(inliers).collect(),
            ..Default::default()
        };
        db.write_two_view_geometry(image_id1, image_id2, &geometry)
            .unwrap();
        db.close().unwrap();
    }

    #[test]
    fn canonical_pair_key_is_order_invariant() {
        assert_eq!(
            canonical_pair_key("a.jpg", "b.jpg"),
            canonical_pair_key("b.jpg", "a.jpg")
        );
        assert_eq!(
            image_pair_to_pair_id(1, 2).unwrap(),
            image_pair_to_pair_id(2, 1).unwrap()
        );
    }
}
