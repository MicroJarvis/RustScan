use anyhow::{Context, Result};
use clap::Parser;
use rustsfm::correspondence_graph::{pair_id_to_image_pair, FeatureMatch, ImageId, ImagePairId};
use rustsfm::database::{
    ColmapDatabase, ColmapTwoViewGeometry, COLMAP_TWO_VIEW_CALIBRATED,
    COLMAP_TWO_VIEW_CALIBRATED_RIG, COLMAP_TWO_VIEW_DEGENERATE, COLMAP_TWO_VIEW_MULTIPLE,
    COLMAP_TWO_VIEW_PANORAMIC, COLMAP_TWO_VIEW_PLANAR, COLMAP_TWO_VIEW_PLANAR_OR_PANORAMIC,
    COLMAP_TWO_VIEW_UNCALIBRATED, COLMAP_TWO_VIEW_UNDEFINED, COLMAP_TWO_VIEW_WATERMARK,
};
use serde::Serialize;
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    reference_db: PathBuf,
    #[arg(long)]
    candidate_db: PathBuf,
    #[arg(long, default_value_t = 32)]
    limit: usize,
    #[arg(long, default_value_t = 16)]
    sample_limit: usize,
    #[arg(long)]
    include_exact: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
struct MatchKey {
    point2d_idx1: u32,
    point2d_idx2: u32,
}

#[derive(Debug, Serialize)]
struct PairDiff {
    image_name1: String,
    image_name2: String,
    reference_config: String,
    candidate_config: String,
    reference_inliers: usize,
    candidate_inliers: usize,
    inlier_delta: isize,
    intersection: usize,
    union: usize,
    jaccard: f64,
    reference_overlap: f64,
    candidate_overlap: f64,
    reference_only_count: usize,
    candidate_only_count: usize,
    reference_only_sample: Vec<MatchKey>,
    candidate_only_sample: Vec<MatchKey>,
}

#[derive(Debug, Serialize)]
struct DiffReport {
    reference_db: PathBuf,
    candidate_db: PathBuf,
    reference_pairs: usize,
    candidate_pairs: usize,
    common_pairs: usize,
    config_mismatch_count: usize,
    inlier_mismatch_count: usize,
    mask_mismatch_count: usize,
    max_abs_inlier_delta: usize,
    total_inlier_delta: isize,
    pair_diffs: Vec<PairDiff>,
}

type PairNameKey = (String, String);
type GeometryByName = HashMap<PairNameKey, (ColmapTwoViewGeometry, ImageId, ImageId)>;

fn main() -> Result<()> {
    let args = Args::parse();
    let report = build_report(&args)?;
    serde_json::to_writer_pretty(std::io::stdout(), &report)?;
    println!();
    Ok(())
}

fn build_report(args: &Args) -> Result<DiffReport> {
    let reference = ColmapDatabase::open(&args.reference_db)
        .with_context(|| format!("open reference db {}", args.reference_db.display()))?;
    let candidate = ColmapDatabase::open(&args.candidate_db)
        .with_context(|| format!("open candidate db {}", args.candidate_db.display()))?;
    let reference_names = image_id_to_name_map(&reference)?;
    let candidate_names = image_id_to_name_map(&candidate)?;
    let reference_geometries = two_view_by_name(&reference, &reference_names)?;
    let candidate_geometries = two_view_by_name(&candidate, &candidate_names)?;

    let reference_keys = reference_geometries
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    let candidate_keys = candidate_geometries
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    let common_keys = reference_keys
        .intersection(&candidate_keys)
        .cloned()
        .collect::<Vec<_>>();

    let mut config_mismatch_count = 0usize;
    let mut inlier_mismatch_count = 0usize;
    let mut mask_mismatch_count = 0usize;
    let mut max_abs_inlier_delta = 0usize;
    let mut total_inlier_delta = 0isize;
    let mut pair_diffs = Vec::new();

    for key in &common_keys {
        let (reference_geometry, reference_id1, reference_id2) =
            reference_geometries.get(key).expect("common key");
        let (candidate_geometry, candidate_id1, candidate_id2) =
            candidate_geometries.get(key).expect("common key");

        let reference_set = canonical_inlier_match_set(
            reference_geometry,
            key,
            *reference_id1,
            *reference_id2,
            &reference_names,
        );
        let candidate_set = canonical_inlier_match_set(
            candidate_geometry,
            key,
            *candidate_id1,
            *candidate_id2,
            &candidate_names,
        );
        let intersection = reference_set.intersection(&candidate_set).count();
        let union = reference_set.union(&candidate_set).count();
        let reference_only = reference_set
            .difference(&candidate_set)
            .cloned()
            .collect::<Vec<_>>();
        let candidate_only = candidate_set
            .difference(&reference_set)
            .cloned()
            .collect::<Vec<_>>();

        let config_mismatch = reference_geometry.config != candidate_geometry.config;
        let inlier_delta = candidate_geometry.inlier_matches.len() as isize
            - reference_geometry.inlier_matches.len() as isize;
        let inlier_mismatch = inlier_delta != 0;
        let mask_mismatch = !reference_only.is_empty() || !candidate_only.is_empty();
        if config_mismatch {
            config_mismatch_count += 1;
        }
        if inlier_mismatch {
            inlier_mismatch_count += 1;
            total_inlier_delta += inlier_delta;
            max_abs_inlier_delta = max_abs_inlier_delta.max(inlier_delta.unsigned_abs());
        }
        if mask_mismatch {
            mask_mismatch_count += 1;
        }
        if args.include_exact || config_mismatch || inlier_mismatch || mask_mismatch {
            pair_diffs.push(PairDiff {
                image_name1: key.0.clone(),
                image_name2: key.1.clone(),
                reference_config: two_view_config_name(reference_geometry.config).to_string(),
                candidate_config: two_view_config_name(candidate_geometry.config).to_string(),
                reference_inliers: reference_geometry.inlier_matches.len(),
                candidate_inliers: candidate_geometry.inlier_matches.len(),
                inlier_delta,
                intersection,
                union,
                jaccard: ratio(intersection, union),
                reference_overlap: ratio(intersection, reference_set.len()),
                candidate_overlap: ratio(intersection, candidate_set.len()),
                reference_only_count: reference_only.len(),
                candidate_only_count: candidate_only.len(),
                reference_only_sample: sample_matches(reference_only, args.sample_limit),
                candidate_only_sample: sample_matches(candidate_only, args.sample_limit),
            });
        }
    }

    pair_diffs.sort_by(|left, right| {
        let left_score = left.reference_only_count + left.candidate_only_count;
        let right_score = right.reference_only_count + right.candidate_only_count;
        right_score
            .cmp(&left_score)
            .then_with(|| right.inlier_delta.abs().cmp(&left.inlier_delta.abs()))
            .then_with(|| left.image_name1.cmp(&right.image_name1))
            .then_with(|| left.image_name2.cmp(&right.image_name2))
    });
    pair_diffs.truncate(args.limit);

    Ok(DiffReport {
        reference_db: args.reference_db.clone(),
        candidate_db: args.candidate_db.clone(),
        reference_pairs: reference_geometries.len(),
        candidate_pairs: candidate_geometries.len(),
        common_pairs: common_keys.len(),
        config_mismatch_count,
        inlier_mismatch_count,
        mask_mismatch_count,
        max_abs_inlier_delta,
        total_inlier_delta,
        pair_diffs,
    })
}

fn image_id_to_name_map(db: &ColmapDatabase) -> Result<HashMap<ImageId, String>> {
    db.read_all_images()?
        .into_iter()
        .map(|image| Ok((image.image_id, image.name)))
        .collect()
}

fn two_view_by_name(
    db: &ColmapDatabase,
    id_to_name: &HashMap<ImageId, String>,
) -> Result<GeometryByName> {
    let mut geometries = HashMap::new();
    for (pair_id, geometry) in db.read_two_view_geometries()? {
        let (image_id1, image_id2) = image_pair_from_pair_id(pair_id)?;
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

fn image_pair_from_pair_id(pair_id: ImagePairId) -> Result<(ImageId, ImageId)> {
    pair_id_to_image_pair(pair_id).map_err(|err| anyhow::anyhow!("{err:?}"))
}

fn canonical_pair_key(name1: &str, name2: &str) -> PairNameKey {
    if name1 <= name2 {
        (name1.to_string(), name2.to_string())
    } else {
        (name2.to_string(), name1.to_string())
    }
}

fn canonical_inlier_match_set(
    geometry: &ColmapTwoViewGeometry,
    name_key: &PairNameKey,
    image_id1: ImageId,
    image_id2: ImageId,
    id_to_name: &HashMap<ImageId, String>,
) -> BTreeSet<MatchKey> {
    let lo_id = image_id1.min(image_id2);
    let lo_name = id_to_name
        .get(&lo_id)
        .expect("canonical inlier set requires image names");
    geometry
        .inlier_matches
        .iter()
        .map(|m| canonical_match_key(m, lo_name == &name_key.0))
        .collect()
}

fn canonical_match_key(match_: &FeatureMatch, stored_order_matches_key: bool) -> MatchKey {
    let (point2d_idx1, point2d_idx2) = if stored_order_matches_key {
        (match_.point2d_idx1, match_.point2d_idx2)
    } else {
        (match_.point2d_idx2, match_.point2d_idx1)
    };
    MatchKey {
        point2d_idx1,
        point2d_idx2,
    }
}

fn sample_matches(mut matches: Vec<MatchKey>, sample_limit: usize) -> Vec<MatchKey> {
    matches.sort_unstable();
    matches.truncate(sample_limit);
    matches
}

fn ratio(numer: usize, denom: usize) -> f64 {
    if denom == 0 {
        0.0
    } else {
        numer as f64 / denom as f64
    }
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
