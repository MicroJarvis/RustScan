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
    pub config_histogram: Vec<TwoViewConfigCount>,
    pub differences: Vec<ParityDifference>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct DatabaseLayerStats {
    pub cameras: usize,
    pub images: usize,
    pub frames: usize,
    pub rigs: usize,
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
) -> Result<ParityReport> {
    let requested_images = image_names.into_iter().collect::<BTreeSet<_>>();
    let db = ColmapDatabase::open(database)?;
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
    let config_histogram = two_view_config_histogram(&db)?;
    let differences = parity_differences(&requested_images, &raw, &cache_stats, &bridge);

    Ok(ParityReport {
        database: database.to_path_buf(),
        requested_images: requested_images.into_iter().collect(),
        raw,
        cache: cache_stats,
        bridge,
        config_histogram,
        differences,
    })
}

fn raw_database_stats(db: &ColmapDatabase) -> Result<DatabaseLayerStats> {
    let keypoint_counts = db.read_keypoint_counts()?;
    let raw_matches = db.read_all_matches()?;
    let two_view_geometries = db.read_two_view_geometries()?;
    Ok(DatabaseLayerStats {
        cameras: db.read_all_cameras()?.len(),
        images: db.read_all_images()?.len(),
        frames: db.read_all_frames()?.len(),
        rigs: db.read_all_rigs()?.len(),
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

        let report = compare_database_parity(&path, Vec::<String>::new(), 1, true, false)?;
        assert_eq!(report.raw.images, 3);
        assert_eq!(report.raw.two_view_pairs, 3);
        assert_eq!(report.raw.verified_two_view_pairs, 2);
        assert_eq!(report.cache.images, 2);
        assert_eq!(report.cache.two_view_pairs, 1);
        assert_eq!(report.bridge.frame_pairs, 1);
        assert_eq!(report.bridge.matches, 2);
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
        let report =
            compare_database_parity(&path, vec!["missing.jpg".to_string()], 1, false, false)?;
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
}
