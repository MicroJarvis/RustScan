use crate::correspondence_graph::{pair_id_to_image_pair, ImageId};
use crate::database::{
    is_colmap_two_view_geometry_with_inliers, ColmapDatabase, DatabaseCacheOptions,
    COLMAP_TWO_VIEW_CALIBRATED, COLMAP_TWO_VIEW_CALIBRATED_RIG, COLMAP_TWO_VIEW_DEGENERATE,
    COLMAP_TWO_VIEW_MULTIPLE, COLMAP_TWO_VIEW_PANORAMIC, COLMAP_TWO_VIEW_PLANAR,
    COLMAP_TWO_VIEW_PLANAR_OR_PANORAMIC, COLMAP_TWO_VIEW_UNCALIBRATED, COLMAP_TWO_VIEW_UNDEFINED,
    COLMAP_TWO_VIEW_WATERMARK,
};
use crate::mapper::database_pair_matches_for_frames;
use crate::types::ImageFrame;
use anyhow::Result;
use rustslam::{Descriptors, KeyPoint};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityReport {
    pub database: PathBuf,
    pub requested_images: Vec<String>,
    pub raw: DatabaseLayerStats,
    pub cache: DatabaseLayerStats,
    pub bridge: BridgeStats,
    pub initial_pair_input: InitialPairInputReport,
    pub config_histogram: Vec<TwoViewConfigCount>,
    pub differences: Vec<ParityDifference>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct DatabaseLayerStats {
    pub cameras: usize,
    pub images: usize,
    pub frames: usize,
    pub rigs: usize,
    pub frame_data: usize,
    pub images_with_frame: usize,
    pub images_without_frame: usize,
    pub pose_priors: usize,
    pub keypoint_rows: usize,
    pub keypoints: usize,
    pub raw_match_pairs: usize,
    pub raw_matches: usize,
    pub two_view_pairs: usize,
    pub verified_two_view_pairs: usize,
    pub inlier_matches: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct BridgeStats {
    pub frame_pairs: usize,
    pub matches: usize,
    pub missing_requested_images: Vec<String>,
    pub unmatched_cache_pairs: Vec<PairName>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct InitialPairInputReport {
    pub min_num_inliers: usize,
    pub total_candidates: usize,
    pub eligible_candidates: usize,
    pub selected: Option<InitialPairCandidate>,
    pub top_candidates: Vec<InitialPairCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InitialPairCandidate {
    pub rank: usize,
    pub left: String,
    pub right: String,
    pub left_image_id: ImageId,
    pub right_image_id: ImageId,
    pub config: i32,
    pub config_name: String,
    pub inlier_matches: usize,
    pub left_total_inliers: usize,
    pub right_total_inliers: usize,
    pub score: f32,
    pub eligible: bool,
    pub rejection_reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TwoViewConfigCount {
    pub config: i32,
    pub name: String,
    pub pairs: usize,
    pub inlier_matches: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PairName {
    pub left: String,
    pub right: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParityDifference {
    pub kind: String,
    pub detail: String,
}

pub fn compare_database_parity(
    database: &Path,
    image_names: impl IntoIterator<Item = String>,
    min_num_matches: usize,
    ignore_watermarks: bool,
    load_all_images: bool,
    convert_pose_priors_to_enu: bool,
) -> Result<ParityReport> {
    let requested_images = image_names.into_iter().collect::<BTreeSet<_>>();
    let db = ColmapDatabase::open_read_only(database)?;
    let raw = raw_database_stats(&db)?;
    let all_images = db.read_all_images()?;
    let keypoint_counts = db
        .read_keypoint_counts()?
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let cache = db.load_cache(&DatabaseCacheOptions {
        min_num_matches,
        ignore_watermarks,
        image_names: requested_images.clone(),
        load_all_images,
        convert_pose_priors_to_enu,
        ..DatabaseCacheOptions::default()
    })?;
    let cache_stats = cache_stats(&cache, &db)?;
    let frames = cache_frames(&cache.images, &keypoint_counts);
    let bridge_pairs = database_pair_matches_for_frames(&frames, &cache)?;
    let bridge = bridge_stats(
        &requested_images,
        &all_images,
        &cache,
        &frames,
        &bridge_pairs,
    )?;
    let initial_pair_input = initial_pair_input_report(&cache, 100)?;
    let config_histogram = two_view_config_histogram(&db)?;
    let differences = parity_differences(&requested_images, &raw, &cache_stats, &bridge);

    Ok(ParityReport {
        database: database.to_path_buf(),
        requested_images: requested_images.into_iter().collect(),
        raw,
        cache: cache_stats,
        bridge,
        initial_pair_input,
        config_histogram,
        differences,
    })
}

fn raw_database_stats(db: &ColmapDatabase) -> Result<DatabaseLayerStats> {
    let keypoint_counts = db.read_keypoint_counts()?;
    let raw_matches = db.read_all_matches()?;
    let two_view_geometries = db.read_two_view_geometries()?;
    let images = db.read_all_images()?;
    let frames = db.read_all_frames()?;
    Ok(DatabaseLayerStats {
        cameras: db.read_all_cameras()?.len(),
        images: images.len(),
        frames: frames.len(),
        rigs: db.read_all_rigs()?.len(),
        frame_data: frames.iter().map(|frame| frame.data_ids.len()).sum(),
        images_with_frame: images
            .iter()
            .filter(|image| image.frame_id.is_some())
            .count(),
        images_without_frame: images
            .iter()
            .filter(|image| image.frame_id.is_none())
            .count(),
        pose_priors: db.read_all_pose_priors()?.len(),
        keypoint_rows: keypoint_counts.len(),
        keypoints: keypoint_counts.iter().map(|(_, count)| *count).sum(),
        raw_match_pairs: raw_matches.len(),
        raw_matches: raw_matches.iter().map(|(_, matches)| matches.len()).sum(),
        two_view_pairs: two_view_geometries.len(),
        verified_two_view_pairs: two_view_geometries
            .iter()
            .filter(|(_, geometry)| {
                is_colmap_two_view_geometry_with_inliers(geometry.config)
                    && !geometry.inlier_matches.is_empty()
            })
            .count(),
        inlier_matches: two_view_geometries
            .iter()
            .map(|(_, geometry)| geometry.inlier_matches.len())
            .sum(),
    })
}

fn cache_stats(
    cache: &crate::database::DatabaseCache,
    db: &ColmapDatabase,
) -> Result<DatabaseLayerStats> {
    let mut keypoints = 0usize;
    for image_id in cache.images.keys() {
        keypoints += db.read_keypoints(*image_id)?.len();
    }
    let pair_matches = cache.correspondence_graph.num_matches_between_all_images();
    Ok(DatabaseLayerStats {
        cameras: cache.cameras.len(),
        images: cache.images.len(),
        frames: cache.frames.len(),
        rigs: cache.rigs.len(),
        frame_data: cache
            .frames
            .values()
            .map(|frame| frame.data_ids.len())
            .sum(),
        images_with_frame: cache
            .images
            .values()
            .filter(|image| image.frame_id.is_some())
            .count(),
        images_without_frame: cache
            .images
            .values()
            .filter(|image| image.frame_id.is_none())
            .count(),
        pose_priors: cache.pose_priors.len(),
        keypoint_rows: cache.images.len(),
        keypoints,
        raw_match_pairs: 0,
        raw_matches: 0,
        two_view_pairs: pair_matches.len(),
        verified_two_view_pairs: pair_matches.len(),
        inlier_matches: pair_matches.values().map(|&count| count as usize).sum(),
    })
}

fn cache_frames(
    images: &BTreeMap<ImageId, crate::database::ColmapDatabaseImage>,
    keypoint_counts: &BTreeMap<ImageId, usize>,
) -> Vec<ImageFrame> {
    images
        .values()
        .enumerate()
        .map(|(id, image)| {
            let keypoints =
                vec![KeyPoint::new(0.0, 0.0); *keypoint_counts.get(&image.image_id).unwrap_or(&0)];
            ImageFrame {
                id,
                name: image.name.clone(),
                path: PathBuf::from(&image.name),
                width: 0,
                height: 0,
                keypoints,
                descriptors: Descriptors::new(),
                sift: Default::default(),
                wide_descriptors: crate::wide::WideDescriptors {
                    data: Vec::new(),
                    dim: 0,
                    count: 0,
                },
                strong_feature_indices: Vec::new(),
                colors: Vec::new(),
            }
        })
        .collect()
}

fn bridge_stats(
    requested_images: &BTreeSet<String>,
    all_images: &[crate::database::ColmapDatabaseImage],
    cache: &crate::database::DatabaseCache,
    frames: &[ImageFrame],
    bridge_pairs: &[crate::mapper::DatabasePairMatches],
) -> Result<BridgeStats> {
    let all_image_names = all_images
        .iter()
        .map(|image| image.name.as_str())
        .collect::<BTreeSet<_>>();
    let cache_image_names = cache
        .images
        .values()
        .map(|image| image.name.as_str())
        .collect::<BTreeSet<_>>();
    let missing_requested_images = requested_images
        .iter()
        .filter(|name| !all_image_names.contains(name.as_str()))
        .cloned()
        .collect::<Vec<_>>();

    let bridged = bridge_pairs
        .iter()
        .map(|pair| ordered_names(&frames[pair.left].name, &frames[pair.right].name))
        .collect::<BTreeSet<_>>();
    let mut unmatched_cache_pairs = Vec::new();
    for pair_id in cache.correspondence_graph.image_pairs() {
        let (image_id1, image_id2) =
            pair_id_to_image_pair(pair_id).map_err(|err| anyhow::anyhow!("{err:?}"))?;
        let Some(image1) = cache.images.get(&image_id1) else {
            continue;
        };
        let Some(image2) = cache.images.get(&image_id2) else {
            continue;
        };
        if !cache_image_names.contains(image1.name.as_str())
            || !cache_image_names.contains(image2.name.as_str())
        {
            continue;
        }
        let names = ordered_names(&image1.name, &image2.name);
        if !bridged.contains(&names) {
            unmatched_cache_pairs.push(PairName {
                left: names.0,
                right: names.1,
            });
        }
    }

    Ok(BridgeStats {
        frame_pairs: bridge_pairs.len(),
        matches: bridge_pairs
            .iter()
            .map(|pair| pair.matches.len())
            .sum::<usize>(),
        missing_requested_images,
        unmatched_cache_pairs,
    })
}

fn two_view_config_histogram(db: &ColmapDatabase) -> Result<Vec<TwoViewConfigCount>> {
    let mut counts = BTreeMap::<i32, TwoViewConfigCount>::new();
    for (_, geometry) in db.read_two_view_geometries()? {
        let entry = counts
            .entry(geometry.config)
            .or_insert_with(|| TwoViewConfigCount {
                config: geometry.config,
                name: two_view_config_name(geometry.config).to_string(),
                pairs: 0,
                inlier_matches: 0,
            });
        entry.pairs += 1;
        entry.inlier_matches += geometry.inlier_matches.len();
    }
    Ok(counts.into_values().collect())
}

fn initial_pair_input_report(
    cache: &crate::database::DatabaseCache,
    min_num_inliers: usize,
) -> Result<InitialPairInputReport> {
    let pair_matches = cache.correspondence_graph.num_matches_between_all_images();
    let mut image_inliers = BTreeMap::<ImageId, usize>::new();
    for (&pair_id, &matches) in &pair_matches {
        let (image_id1, image_id2) =
            pair_id_to_image_pair(pair_id).map_err(|err| anyhow::anyhow!("{err:?}"))?;
        *image_inliers.entry(image_id1).or_default() += matches as usize;
        *image_inliers.entry(image_id2).or_default() += matches as usize;
    }

    let mut candidates = Vec::new();
    for (&pair_id, &matches) in &pair_matches {
        let (image_id1, image_id2) =
            pair_id_to_image_pair(pair_id).map_err(|err| anyhow::anyhow!("{err:?}"))?;
        let Some(image1) = cache.images.get(&image_id1) else {
            continue;
        };
        let Some(image2) = cache.images.get(&image_id2) else {
            continue;
        };
        let geometry = cache
            .correspondence_graph
            .extract_two_view_geometry(image_id1, image_id2, false)
            .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        let inlier_matches = matches as usize;
        let mut rejection_reasons = Vec::new();
        if inlier_matches < min_num_inliers {
            rejection_reasons.push(format!("inliers_lt_{min_num_inliers}"));
        }
        if !is_initial_pair_usable_config(geometry.config) {
            rejection_reasons.push(format!(
                "config_{}",
                two_view_config_name(geometry.config).to_ascii_lowercase()
            ));
        }
        let left_total_inliers = *image_inliers.get(&image_id1).unwrap_or(&0);
        let right_total_inliers = *image_inliers.get(&image_id2).unwrap_or(&0);
        let score =
            initial_pair_input_score(inlier_matches, left_total_inliers, right_total_inliers);
        candidates.push(InitialPairCandidate {
            rank: 0,
            left: image1.name.clone(),
            right: image2.name.clone(),
            left_image_id: image_id1,
            right_image_id: image_id2,
            config: geometry.config,
            config_name: two_view_config_name(geometry.config).to_string(),
            inlier_matches,
            left_total_inliers,
            right_total_inliers,
            score,
            eligible: rejection_reasons.is_empty(),
            rejection_reasons,
        });
    }
    candidates.sort_by(|a, b| {
        b.eligible
            .cmp(&a.eligible)
            .then_with(|| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.left_image_id.cmp(&b.left_image_id))
            .then_with(|| a.right_image_id.cmp(&b.right_image_id))
    });
    for (rank, candidate) in candidates.iter_mut().enumerate() {
        candidate.rank = rank + 1;
    }
    let selected = candidates
        .iter()
        .find(|candidate| candidate.eligible)
        .cloned();
    let eligible_candidates = candidates
        .iter()
        .filter(|candidate| candidate.eligible)
        .count();
    let total_candidates = candidates.len();
    candidates.truncate(50);
    Ok(InitialPairInputReport {
        min_num_inliers,
        total_candidates,
        eligible_candidates,
        selected,
        top_candidates: candidates,
    })
}

fn is_initial_pair_usable_config(config: i32) -> bool {
    matches!(
        config,
        COLMAP_TWO_VIEW_CALIBRATED
            | COLMAP_TWO_VIEW_UNCALIBRATED
            | COLMAP_TWO_VIEW_PLANAR
            | COLMAP_TWO_VIEW_PANORAMIC
            | COLMAP_TWO_VIEW_PLANAR_OR_PANORAMIC
            | COLMAP_TWO_VIEW_CALIBRATED_RIG
    )
}

fn initial_pair_input_score(
    pair_inliers: usize,
    left_total_inliers: usize,
    right_total_inliers: usize,
) -> f32 {
    (left_total_inliers as f32).sqrt() * 20.0
        + (right_total_inliers as f32).sqrt() * 20.0
        + pair_inliers as f32 * 10.0
}

fn parity_differences(
    requested_images: &BTreeSet<String>,
    raw: &DatabaseLayerStats,
    cache: &DatabaseLayerStats,
    bridge: &BridgeStats,
) -> Vec<ParityDifference> {
    let mut out = Vec::new();
    if raw.keypoint_rows != raw.images {
        out.push(ParityDifference {
            kind: "missing_keypoint_rows".to_string(),
            detail: format!(
                "database has {} images but {} keypoint rows",
                raw.images, raw.keypoint_rows
            ),
        });
    }
    if !requested_images.is_empty() && bridge.missing_requested_images.is_empty() {
        if cache.images < requested_images.len() {
            out.push(ParityDifference {
                kind: "filtered_requested_images".to_string(),
                detail: format!(
                    "cache kept {} images from {} requested names",
                    cache.images,
                    requested_images.len()
                ),
            });
        }
    }
    if !bridge.missing_requested_images.is_empty() {
        out.push(ParityDifference {
            kind: "missing_requested_images".to_string(),
            detail: bridge.missing_requested_images.join(", "),
        });
    }
    if bridge.frame_pairs != cache.two_view_pairs {
        out.push(ParityDifference {
            kind: "bridge_pair_count_mismatch".to_string(),
            detail: format!(
                "bridge produced {} frame pairs from {} cache graph pairs",
                bridge.frame_pairs, cache.two_view_pairs
            ),
        });
    }
    if bridge.matches != cache.inlier_matches {
        out.push(ParityDifference {
            kind: "bridge_match_count_mismatch".to_string(),
            detail: format!(
                "bridge produced {} matches from {} cache inlier matches",
                bridge.matches, cache.inlier_matches
            ),
        });
    }
    if !bridge.unmatched_cache_pairs.is_empty() {
        out.push(ParityDifference {
            kind: "unmatched_cache_pairs".to_string(),
            detail: format!(
                "{} cache pairs did not map to frame pairs",
                bridge.unmatched_cache_pairs.len()
            ),
        });
    }
    out
}

fn ordered_names(left: &str, right: &str) -> (String, String) {
    if left <= right {
        (left.to_string(), right.to_string())
    } else {
        (right.to_string(), left.to_string())
    }
}

fn two_view_config_name(config: i32) -> &'static str {
    match config {
        COLMAP_TWO_VIEW_UNDEFINED => "UNDEFINED",
        COLMAP_TWO_VIEW_DEGENERATE => "DEGENERATE",
        COLMAP_TWO_VIEW_CALIBRATED => "CALIBRATED",
        COLMAP_TWO_VIEW_UNCALIBRATED => "UNCALIBRATED",
        COLMAP_TWO_VIEW_PLANAR => "PLANAR",
        COLMAP_TWO_VIEW_PANORAMIC => "PANORAMIC",
        COLMAP_TWO_VIEW_PLANAR_OR_PANORAMIC => "PLANAR_OR_PANORAMIC",
        COLMAP_TWO_VIEW_WATERMARK => "WATERMARK",
        COLMAP_TWO_VIEW_MULTIPLE => "MULTIPLE",
        COLMAP_TWO_VIEW_CALIBRATED_RIG => "CALIBRATED_RIG",
        _ => "UNKNOWN",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colmap::ColmapCamera;
    use crate::correspondence_graph::FeatureMatch;
    use crate::database::{
        ColmapDatabaseCamera, ColmapDatabaseImage, ColmapTwoViewGeometry,
        COLMAP_TWO_VIEW_CALIBRATED, COLMAP_TWO_VIEW_DEGENERATE, COLMAP_TWO_VIEW_WATERMARK,
    };
    use crate::types::COLMAP_PINHOLE;

    #[test]
    fn parity_report_summarizes_cache_and_bridge_layers() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("database.db");
        let db = ColmapDatabase::open(&path)?;
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
        )?;
        for (image_id, name) in [(1, "a.jpg"), (2, "b.jpg"), (3, "c.jpg")] {
            db.write_image(
                &ColmapDatabaseImage {
                    image_id,
                    name: name.to_string(),
                    camera_id: 1,
                    frame_id: None,
                },
                true,
            )?;
            db.write_keypoints(
                image_id,
                &[
                    crate::database::ColmapKeypoint::new(0.0, 0.0),
                    crate::database::ColmapKeypoint::new(1.0, 1.0),
                    crate::database::ColmapKeypoint::new(2.0, 2.0),
                ],
            )?;
        }
        db.write_two_view_geometry(
            1,
            2,
            &ColmapTwoViewGeometry {
                config: COLMAP_TWO_VIEW_CALIBRATED,
                inlier_matches: vec![FeatureMatch::new(0, 0), FeatureMatch::new(1, 1)],
                ..Default::default()
            },
        )?;
        db.write_two_view_geometry(
            2,
            3,
            &ColmapTwoViewGeometry {
                config: COLMAP_TWO_VIEW_WATERMARK,
                inlier_matches: vec![FeatureMatch::new(0, 0), FeatureMatch::new(1, 1)],
                ..Default::default()
            },
        )?;
        db.write_two_view_geometry(
            1,
            3,
            &ColmapTwoViewGeometry {
                config: COLMAP_TWO_VIEW_DEGENERATE,
                inlier_matches: vec![FeatureMatch::new(0, 0), FeatureMatch::new(1, 1)],
                ..Default::default()
            },
        )?;

        let report = compare_database_parity(&path, Vec::<String>::new(), 1, true, false, false)?;
        assert_eq!(report.raw.images, 3);
        assert_eq!(report.raw.frames, 0);
        assert_eq!(report.raw.frame_data, 0);
        assert_eq!(report.raw.images_with_frame, 0);
        assert_eq!(report.raw.images_without_frame, 3);
        assert_eq!(report.raw.two_view_pairs, 3);
        assert_eq!(report.raw.verified_two_view_pairs, 2);
        assert_eq!(report.cache.images, 3);
        assert_eq!(report.cache.frames, 3);
        assert_eq!(report.cache.frame_data, 3);
        assert_eq!(report.cache.images_with_frame, 3);
        assert_eq!(report.cache.images_without_frame, 0);
        assert_eq!(report.cache.two_view_pairs, 2);
        assert_eq!(report.cache.inlier_matches, 4);
        assert_eq!(report.bridge.frame_pairs, 2);
        assert_eq!(report.bridge.matches, 4);
        assert_eq!(report.initial_pair_input.total_candidates, 2);
        assert_eq!(report.initial_pair_input.eligible_candidates, 0);
        assert!(report.initial_pair_input.selected.is_none());
        let rejection_reasons = report
            .initial_pair_input
            .top_candidates
            .iter()
            .map(|candidate| candidate.rejection_reasons.clone())
            .collect::<Vec<_>>();
        assert!(rejection_reasons.contains(&vec!["inliers_lt_100".to_string()]));
        assert!(rejection_reasons.contains(&vec![
            "inliers_lt_100".to_string(),
            "config_degenerate".to_string()
        ]));
        assert!(report
            .config_histogram
            .iter()
            .any(|entry| entry.config == COLMAP_TWO_VIEW_WATERMARK && entry.pairs == 1));
        assert!(report.differences.is_empty());
        Ok(())
    }

    #[test]
    fn parity_report_notes_missing_requested_images() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("database.db");
        let db = ColmapDatabase::open(&path)?;
        let report = compare_database_parity(
            &path,
            vec!["missing.jpg".to_string()],
            1,
            false,
            false,
            false,
        )?;
        assert_eq!(
            report.bridge.missing_requested_images,
            vec!["missing.jpg".to_string()]
        );
        assert!(report
            .differences
            .iter()
            .any(|diff| diff.kind == "missing_requested_images"));
        drop(db);
        Ok(())
    }

    #[test]
    fn initial_pair_input_prefers_high_inlier_verified_pair() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("database.db");
        let db = ColmapDatabase::open(&path)?;
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
        )?;
        for (image_id, name) in [(1, "a.jpg"), (2, "b.jpg"), (3, "c.jpg")] {
            db.write_image(
                &ColmapDatabaseImage {
                    image_id,
                    name: name.to_string(),
                    camera_id: 1,
                    frame_id: None,
                },
                true,
            )?;
            let keypoints = (0..140)
                .map(|idx| crate::database::ColmapKeypoint::new(idx as f32, idx as f32))
                .collect::<Vec<_>>();
            db.write_keypoints(image_id, &keypoints)?;
        }
        db.write_two_view_geometry(
            1,
            2,
            &ColmapTwoViewGeometry {
                config: COLMAP_TWO_VIEW_CALIBRATED,
                inlier_matches: (0..120).map(|idx| FeatureMatch::new(idx, idx)).collect(),
                ..Default::default()
            },
        )?;
        db.write_two_view_geometry(
            2,
            3,
            &ColmapTwoViewGeometry {
                config: COLMAP_TWO_VIEW_WATERMARK,
                inlier_matches: (0..130).map(|idx| FeatureMatch::new(idx, idx)).collect(),
                ..Default::default()
            },
        )?;

        let report = compare_database_parity(&path, Vec::<String>::new(), 1, false, false, false)?;
        let selected = report
            .initial_pair_input
            .selected
            .as_ref()
            .expect("selected candidate");
        assert_eq!(
            (selected.left.as_str(), selected.right.as_str()),
            ("a.jpg", "b.jpg")
        );
        assert_eq!(selected.inlier_matches, 120);
        assert_eq!(report.initial_pair_input.eligible_candidates, 1);
        assert!(report
            .initial_pair_input
            .top_candidates
            .iter()
            .any(|candidate| candidate.config == COLMAP_TWO_VIEW_WATERMARK
                && candidate
                    .rejection_reasons
                    .contains(&"config_watermark".to_string())));
        Ok(())
    }
}
