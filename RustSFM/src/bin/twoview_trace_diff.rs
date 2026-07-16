use anyhow::{bail, Context, Result};
use clap::Parser;
use nalgebra::{Matrix3, Vector3};
use rustsfm::correspondence_graph::{pair_id_to_image_pair, FeatureMatch, ImageId, ImagePairId};
use rustsfm::database::{
    ColmapDatabase, ColmapKeypoint, ColmapTwoViewGeometry, COLMAP_TWO_VIEW_CALIBRATED,
    COLMAP_TWO_VIEW_CALIBRATED_RIG, COLMAP_TWO_VIEW_DEGENERATE, COLMAP_TWO_VIEW_MULTIPLE,
    COLMAP_TWO_VIEW_PANORAMIC, COLMAP_TWO_VIEW_PLANAR, COLMAP_TWO_VIEW_PLANAR_OR_PANORAMIC,
    COLMAP_TWO_VIEW_UNCALIBRATED, COLMAP_TWO_VIEW_UNDEFINED, COLMAP_TWO_VIEW_WATERMARK,
};
use rustsfm::types::CameraModel;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::path::PathBuf;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    trace_json: PathBuf,
    #[arg(long)]
    candidate_db: PathBuf,
    #[arg(long, default_value_t = 32)]
    limit: usize,
    #[arg(long, default_value_t = 16)]
    sample_limit: usize,
    #[arg(long)]
    include_exact: bool,
    #[arg(long, default_value_t = 4.0)]
    max_error_px: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
struct MatchKey {
    point2d_idx1: u32,
    point2d_idx2: u32,
}

#[derive(Debug, Deserialize)]
struct TraceRoot {
    events: Option<Vec<TraceEvent>>,
    verifier_trace: Option<NestedTrace>,
}

#[derive(Debug, Deserialize)]
struct NestedTrace {
    events: Vec<TraceEvent>,
}

#[derive(Debug, Deserialize)]
struct TraceEvent {
    left_image: String,
    right_image: String,
    num_inliers: usize,
    two_view_config: i32,
    #[serde(default)]
    has_model_details: bool,
    #[serde(default)]
    e_success: bool,
    #[serde(default)]
    f_success: bool,
    #[serde(default)]
    h_success: bool,
    #[serde(default)]
    e_inliers: usize,
    #[serde(default)]
    f_inliers: usize,
    #[serde(default)]
    h_inliers: usize,
    #[serde(default)]
    selected_source: String,
    #[serde(default)]
    has_e_model: bool,
    #[serde(default)]
    has_f_model: bool,
    #[serde(default)]
    has_h_model: bool,
    #[serde(default)]
    e_matrix: Option<[f64; 9]>,
    #[serde(default)]
    f_matrix: Option<[f64; 9]>,
    #[serde(default)]
    h_matrix: Option<[f64; 9]>,
    #[serde(default)]
    inlier_matches: Option<Vec<[u32; 2]>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ResidualModelKind {
    Essential,
    Fundamental,
    Homography,
}

#[derive(Debug, Clone, Serialize)]
struct MatchResidualSample {
    point2d_idx1: u32,
    point2d_idx2: u32,
    trace_selected_residual: Option<f64>,
    candidate_selected_residual: Option<f64>,
    trace_essential_residual: Option<f64>,
    trace_fundamental_residual: Option<f64>,
    trace_homography_residual: Option<f64>,
    candidate_essential_residual: Option<f64>,
    candidate_fundamental_residual: Option<f64>,
    candidate_homography_residual: Option<f64>,
    trace_margin: Option<f64>,
    candidate_margin: Option<f64>,
    trace_inlier_under_selected_model: Option<bool>,
    candidate_inlier_under_selected_model: Option<bool>,
    trace_selected_model: Option<ResidualModelKind>,
    candidate_selected_model: Option<ResidualModelKind>,
}

#[derive(Debug, Serialize)]
struct PairDiff {
    image_name1: String,
    image_name2: String,
    trace_config: String,
    candidate_config: String,
    trace_selected_source: Option<String>,
    trace_e_success: Option<bool>,
    trace_f_success: Option<bool>,
    trace_h_success: Option<bool>,
    trace_e_inliers: Option<usize>,
    trace_f_inliers: Option<usize>,
    trace_h_inliers: Option<usize>,
    trace_inliers: usize,
    candidate_inliers: usize,
    inlier_delta: isize,
    intersection: usize,
    union: usize,
    jaccard: f64,
    trace_overlap: f64,
    candidate_overlap: f64,
    trace_only_count: usize,
    candidate_only_count: usize,
    trace_only_sample: Vec<MatchKey>,
    candidate_only_sample: Vec<MatchKey>,
    trace_only_residual_sample: Vec<MatchResidualSample>,
    candidate_only_residual_sample: Vec<MatchResidualSample>,
}

#[derive(Debug, Serialize)]
struct DiffReport {
    trace_json: PathBuf,
    candidate_db: PathBuf,
    trace_pairs: usize,
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
type CameraIdByImageId = HashMap<ImageId, u32>;

fn main() -> Result<()> {
    let args = Args::parse();
    let report = build_report(&args)?;
    serde_json::to_writer_pretty(std::io::stdout(), &report)?;
    println!();
    Ok(())
}

fn build_report(args: &Args) -> Result<DiffReport> {
    let trace_events = read_trace_events(&args.trace_json)?;
    let trace_by_name = trace_events_by_name(&trace_events)?;
    let candidate = ColmapDatabase::open_read_only(&args.candidate_db)
        .with_context(|| format!("open candidate db {}", args.candidate_db.display()))?;
    let candidate_names = image_id_to_name_map(&candidate)?;
    let candidate_ids_by_name = image_name_to_id_map(&candidate_names);
    let candidate_camera_ids = image_id_to_camera_id_map(&candidate)?;
    let candidate_geometries = two_view_by_name(&candidate, &candidate_names)?;

    let trace_keys = trace_by_name.keys().cloned().collect::<BTreeSet<_>>();
    let candidate_keys = candidate_geometries
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    let common_keys = trace_keys
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
        let trace = trace_by_name.get(key).expect("common key");
        let (candidate_geometry, candidate_id1, candidate_id2) =
            candidate_geometries.get(key).expect("common key");
        let candidate_set = canonical_inlier_match_set(candidate_geometry);
        let intersection = trace.inlier_set.intersection(&candidate_set).count();
        let union = trace.inlier_set.union(&candidate_set).count();
        let trace_only = trace
            .inlier_set
            .difference(&candidate_set)
            .cloned()
            .collect::<Vec<_>>();
        let candidate_only = candidate_set
            .difference(&trace.inlier_set)
            .cloned()
            .collect::<Vec<_>>();

        let config_mismatch = trace.config != candidate_geometry.config;
        let inlier_delta =
            candidate_geometry.inlier_matches.len() as isize - trace.inlier_count as isize;
        let inlier_mismatch = inlier_delta != 0;
        let mask_mismatch = !trace_only.is_empty() || !candidate_only.is_empty();
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
            let residual_context = PairResidualContext::from_database(
                &candidate,
                key,
                *candidate_id1,
                *candidate_id2,
                &candidate_ids_by_name,
                &candidate_camera_ids,
                args.max_error_px,
            )?;
            let trace_selected_model = trace
                .model_details
                .as_ref()
                .and_then(|details| residual_model_kind_from_source(&details.selected_source));
            let candidate_selected_model = infer_candidate_selected_model(
                candidate_geometry,
                &candidate_set,
                &residual_context,
            );
            pair_diffs.push(PairDiff {
                image_name1: key.0.clone(),
                image_name2: key.1.clone(),
                trace_config: two_view_config_name(trace.config).to_string(),
                candidate_config: two_view_config_name(candidate_geometry.config).to_string(),
                trace_selected_source: trace
                    .model_details
                    .as_ref()
                    .map(|details| details.selected_source.clone()),
                trace_e_success: trace
                    .model_details
                    .as_ref()
                    .map(|details| details.e_success),
                trace_f_success: trace
                    .model_details
                    .as_ref()
                    .map(|details| details.f_success),
                trace_h_success: trace
                    .model_details
                    .as_ref()
                    .map(|details| details.h_success),
                trace_e_inliers: trace
                    .model_details
                    .as_ref()
                    .map(|details| details.e_inliers),
                trace_f_inliers: trace
                    .model_details
                    .as_ref()
                    .map(|details| details.f_inliers),
                trace_h_inliers: trace
                    .model_details
                    .as_ref()
                    .map(|details| details.h_inliers),
                trace_inliers: trace.inlier_count,
                candidate_inliers: candidate_geometry.inlier_matches.len(),
                inlier_delta,
                intersection,
                union,
                jaccard: ratio(intersection, union),
                trace_overlap: ratio(intersection, trace.inlier_set.len()),
                candidate_overlap: ratio(intersection, candidate_set.len()),
                trace_only_count: trace_only.len(),
                candidate_only_count: candidate_only.len(),
                trace_only_sample: sample_matches(trace_only.clone(), args.sample_limit),
                candidate_only_sample: sample_matches(candidate_only.clone(), args.sample_limit),
                trace_only_residual_sample: sample_residuals(
                    trace_only,
                    args.sample_limit,
                    trace.model_details.as_ref(),
                    candidate_geometry,
                    trace_selected_model,
                    candidate_selected_model,
                    &residual_context,
                ),
                candidate_only_residual_sample: sample_residuals(
                    candidate_only,
                    args.sample_limit,
                    trace.model_details.as_ref(),
                    candidate_geometry,
                    trace_selected_model,
                    candidate_selected_model,
                    &residual_context,
                ),
            });
        }
    }

    pair_diffs.sort_by(|left, right| {
        let left_score = left.trace_only_count + left.candidate_only_count;
        let right_score = right.trace_only_count + right.candidate_only_count;
        right_score
            .cmp(&left_score)
            .then_with(|| right.inlier_delta.abs().cmp(&left.inlier_delta.abs()))
            .then_with(|| left.image_name1.cmp(&right.image_name1))
            .then_with(|| left.image_name2.cmp(&right.image_name2))
    });
    pair_diffs.truncate(args.limit);

    Ok(DiffReport {
        trace_json: args.trace_json.clone(),
        candidate_db: args.candidate_db.clone(),
        trace_pairs: trace_by_name.len(),
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

#[derive(Debug)]
struct TraceGeometry {
    config: i32,
    inlier_count: usize,
    inlier_set: BTreeSet<MatchKey>,
    model_details: Option<TraceModelDetails>,
}

#[derive(Debug)]
struct TraceModelDetails {
    e_success: bool,
    f_success: bool,
    h_success: bool,
    e_inliers: usize,
    f_inliers: usize,
    h_inliers: usize,
    selected_source: String,
    e_matrix: Option<[f64; 9]>,
    f_matrix: Option<[f64; 9]>,
    h_matrix: Option<[f64; 9]>,
}

fn read_trace_events(path: &PathBuf) -> Result<Vec<TraceEvent>> {
    let file = File::open(path).with_context(|| format!("open trace JSON {}", path.display()))?;
    let root: TraceRoot = serde_json::from_reader(file)
        .with_context(|| format!("parse trace JSON {}", path.display()))?;
    if let Some(events) = root.events {
        Ok(events)
    } else if let Some(trace) = root.verifier_trace {
        Ok(trace.events)
    } else {
        bail!("trace JSON has neither events nor verifier_trace.events");
    }
}

fn trace_events_by_name(events: &[TraceEvent]) -> Result<HashMap<PairNameKey, TraceGeometry>> {
    let mut out = HashMap::new();
    for event in events {
        let matches = event.inlier_matches.as_ref().with_context(|| {
            format!(
                "trace event {} / {} has no inlier_matches; rerun colmap_verifier_trace with --include-inlier-matches",
                event.left_image, event.right_image
            )
        })?;
        if matches.len() != event.num_inliers {
            bail!(
                "trace event {} / {} has num_inliers={} but {} inlier_matches",
                event.left_image,
                event.right_image,
                event.num_inliers,
                matches.len()
            );
        }
        let key = canonical_pair_key(&event.left_image, &event.right_image);
        let stored_order_matches_key = event.left_image <= event.right_image;
        let inlier_set = matches
            .iter()
            .map(|m| {
                if stored_order_matches_key {
                    MatchKey {
                        point2d_idx1: m[0],
                        point2d_idx2: m[1],
                    }
                } else {
                    MatchKey {
                        point2d_idx1: m[1],
                        point2d_idx2: m[0],
                    }
                }
            })
            .collect::<BTreeSet<_>>();
        out.insert(
            key,
            TraceGeometry {
                config: event.two_view_config,
                inlier_count: event.num_inliers,
                inlier_set,
                model_details: event.has_model_details.then(|| TraceModelDetails {
                    e_success: event.e_success,
                    f_success: event.f_success,
                    h_success: event.h_success,
                    e_inliers: event.e_inliers,
                    f_inliers: event.f_inliers,
                    h_inliers: event.h_inliers,
                    selected_source: event.selected_source.clone(),
                    e_matrix: canonical_trace_epipolar_matrix(
                        event.has_e_model.then_some(event.e_matrix).flatten(),
                        stored_order_matches_key,
                    ),
                    f_matrix: canonical_trace_epipolar_matrix(
                        event.has_f_model.then_some(event.f_matrix).flatten(),
                        stored_order_matches_key,
                    ),
                    h_matrix: canonical_trace_homography_matrix(
                        event.has_h_model.then_some(event.h_matrix).flatten(),
                        stored_order_matches_key,
                    ),
                }),
            },
        );
    }
    Ok(out)
}

fn canonical_trace_epipolar_matrix(
    matrix: Option<[f64; 9]>,
    already_canonical: bool,
) -> Option<[f64; 9]> {
    if already_canonical {
        matrix
    } else {
        matrix.map(|matrix| row_array_from_matrix3(Matrix3::from_row_slice(&matrix).transpose()))
    }
}

fn canonical_trace_homography_matrix(
    matrix: Option<[f64; 9]>,
    already_canonical: bool,
) -> Option<[f64; 9]> {
    if already_canonical {
        matrix
    } else {
        let matrix = Matrix3::from_row_slice(&matrix?);
        matrix.try_inverse().map(row_array_from_matrix3)
    }
}

fn row_array_from_matrix3(matrix: Matrix3<f64>) -> [f64; 9] {
    [
        matrix[(0, 0)],
        matrix[(0, 1)],
        matrix[(0, 2)],
        matrix[(1, 0)],
        matrix[(1, 1)],
        matrix[(1, 2)],
        matrix[(2, 0)],
        matrix[(2, 1)],
        matrix[(2, 2)],
    ]
}

fn image_id_to_name_map(db: &ColmapDatabase) -> Result<HashMap<ImageId, String>> {
    db.read_all_images()?
        .into_iter()
        .map(|image| Ok((image.image_id, image.name)))
        .collect()
}

fn image_name_to_id_map(id_to_name: &HashMap<ImageId, String>) -> HashMap<String, ImageId> {
    id_to_name
        .iter()
        .map(|(&image_id, name)| (name.clone(), image_id))
        .collect()
}

fn image_id_to_camera_id_map(db: &ColmapDatabase) -> Result<CameraIdByImageId> {
    db.read_all_images()?
        .into_iter()
        .map(|image| Ok((image.image_id, image.camera_id)))
        .collect()
}

fn two_view_by_name(
    db: &ColmapDatabase,
    id_to_name: &HashMap<ImageId, String>,
) -> Result<GeometryByName> {
    let mut geometries = HashMap::new();
    for (pair_id, _) in db.read_two_view_geometries()? {
        let (image_id1, image_id2) = image_pair_from_pair_id(pair_id)?;
        let name1 = id_to_name
            .get(&image_id1)
            .with_context(|| format!("missing image name for image_id={image_id1}"))?;
        let name2 = id_to_name
            .get(&image_id2)
            .with_context(|| format!("missing image name for image_id={image_id2}"))?;
        let (canonical_name1, canonical_name2, canonical_id1, canonical_id2) = if name1 <= name2 {
            (name1.clone(), name2.clone(), image_id1, image_id2)
        } else {
            (name2.clone(), name1.clone(), image_id2, image_id1)
        };
        let geometry = db.read_two_view_geometry(canonical_id1, canonical_id2)?;
        geometries.insert(
            (canonical_name1, canonical_name2),
            (geometry, canonical_id1, canonical_id2),
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

fn canonical_inlier_match_set(geometry: &ColmapTwoViewGeometry) -> BTreeSet<MatchKey> {
    geometry
        .inlier_matches
        .iter()
        .map(|m| canonical_match_key(m))
        .collect()
}

fn canonical_match_key(match_: &FeatureMatch) -> MatchKey {
    MatchKey {
        point2d_idx1: match_.point2d_idx1,
        point2d_idx2: match_.point2d_idx2,
    }
}

fn sample_matches(mut matches: Vec<MatchKey>, sample_limit: usize) -> Vec<MatchKey> {
    matches.sort_unstable();
    matches.truncate(sample_limit);
    matches
}

#[derive(Debug)]
struct PairResidualContext {
    keypoints1: Vec<ColmapKeypoint>,
    keypoints2: Vec<ColmapKeypoint>,
    matches: Vec<MatchKey>,
    camera1: CameraModel,
    camera2: CameraModel,
    essential_threshold: f64,
    pixel_threshold: f64,
}

impl PairResidualContext {
    fn from_database(
        db: &ColmapDatabase,
        key: &PairNameKey,
        image_id1: ImageId,
        image_id2: ImageId,
        id_by_name: &HashMap<String, ImageId>,
        camera_id_by_image_id: &CameraIdByImageId,
        max_error_px: f64,
    ) -> Result<Self> {
        let canonical_id1 = *id_by_name
            .get(&key.0)
            .with_context(|| format!("missing image_id for {}", key.0))?;
        let canonical_id2 = *id_by_name
            .get(&key.1)
            .with_context(|| format!("missing image_id for {}", key.1))?;
        if canonical_id1 != image_id1 || canonical_id2 != image_id2 {
            bail!(
                "internal pair direction mismatch for {} / {}: geometry ids=({}, {}) name ids=({}, {})",
                key.0,
                key.1,
                image_id1,
                image_id2,
                canonical_id1,
                canonical_id2
            );
        }
        let camera1 = load_camera_model(db, canonical_id1, camera_id_by_image_id)?;
        let camera2 = load_camera_model(db, canonical_id2, camera_id_by_image_id)?;
        let essential_threshold = 0.5
            * (camera1.cam_from_img_threshold(max_error_px)
                + camera2.cam_from_img_threshold(max_error_px));
        Ok(Self {
            keypoints1: db.read_keypoints(canonical_id1)?,
            keypoints2: db.read_keypoints(canonical_id2)?,
            matches: db
                .read_matches(canonical_id1, canonical_id2)?
                .iter()
                .map(canonical_match_key)
                .collect(),
            camera1,
            camera2,
            essential_threshold,
            pixel_threshold: max_error_px,
        })
    }

    fn residual(&self, key: &MatchKey, model: ResidualModelKind, matrix: [f64; 9]) -> Option<f64> {
        let kp1 = self.keypoints1.get(key.point2d_idx1 as usize)?;
        let kp2 = self.keypoints2.get(key.point2d_idx2 as usize)?;
        let matrix = Matrix3::from_row_slice(&matrix);
        match model {
            ResidualModelKind::Essential => {
                let ray1 = self.camera1.cam_ray_from_img(kp1.x as f64, kp1.y as f64)?;
                let ray2 = self.camera2.cam_ray_from_img(kp2.x as f64, kp2.y as f64)?;
                let x1 = vector3_from_ray(ray1);
                let x2 = vector3_from_ray(ray2);
                let residual = squared_sampson_error(&x1, &x2, &matrix);
                residual.is_finite().then_some(residual)
            }
            ResidualModelKind::Fundamental => {
                let x1 = Vector3::new(kp1.x as f64, kp1.y as f64, 1.0);
                let x2 = Vector3::new(kp2.x as f64, kp2.y as f64, 1.0);
                let residual = squared_sampson_error(&x1, &x2, &matrix);
                residual.is_finite().then_some(residual)
            }
            ResidualModelKind::Homography => {
                let x1 = Vector3::new(kp1.x as f64, kp1.y as f64, 1.0);
                let x2 = Vector3::new(kp2.x as f64, kp2.y as f64, 1.0);
                let residual = homography_forward_error(&x1, &x2, &matrix);
                residual.is_finite().then_some(residual)
            }
        }
    }

    fn threshold_sq(&self, model: ResidualModelKind) -> f64 {
        match model {
            ResidualModelKind::Essential => self.essential_threshold.max(1.0e-12).powi(2),
            ResidualModelKind::Fundamental | ResidualModelKind::Homography => {
                self.pixel_threshold.max(1.0e-12).powi(2)
            }
        }
    }
}

fn load_camera_model(
    db: &ColmapDatabase,
    image_id: ImageId,
    camera_id_by_image_id: &CameraIdByImageId,
) -> Result<CameraModel> {
    let camera_id = camera_id_by_image_id
        .get(&image_id)
        .with_context(|| format!("missing camera_id for image_id={image_id}"))?;
    let camera = db
        .read_camera(*camera_id)?
        .with_context(|| format!("missing camera_id={camera_id}"))?;
    CameraModel::from_colmap(
        camera.camera.model_id,
        camera.camera.width,
        camera.camera.height,
        &camera.camera.params,
    )
    .with_context(|| format!("unsupported camera_id={camera_id}"))
}

fn sample_residuals(
    matches: Vec<MatchKey>,
    sample_limit: usize,
    trace_details: Option<&TraceModelDetails>,
    candidate_geometry: &ColmapTwoViewGeometry,
    trace_selected_model: Option<ResidualModelKind>,
    candidate_selected_model: Option<ResidualModelKind>,
    context: &PairResidualContext,
) -> Vec<MatchResidualSample> {
    sample_matches(matches, sample_limit)
        .into_iter()
        .map(|key| {
            residual_sample_for_match(
                key,
                trace_details,
                candidate_geometry,
                trace_selected_model,
                candidate_selected_model,
                context,
            )
        })
        .collect()
}

fn residual_sample_for_match(
    key: MatchKey,
    trace_details: Option<&TraceModelDetails>,
    candidate_geometry: &ColmapTwoViewGeometry,
    trace_selected_model: Option<ResidualModelKind>,
    candidate_selected_model: Option<ResidualModelKind>,
    context: &PairResidualContext,
) -> MatchResidualSample {
    let trace_essential_residual = trace_details
        .and_then(|details| details.e_matrix)
        .and_then(|matrix| context.residual(&key, ResidualModelKind::Essential, matrix));
    let trace_fundamental_residual = trace_details
        .and_then(|details| details.f_matrix)
        .and_then(|matrix| context.residual(&key, ResidualModelKind::Fundamental, matrix));
    let trace_homography_residual = trace_details
        .and_then(|details| details.h_matrix)
        .and_then(|matrix| context.residual(&key, ResidualModelKind::Homography, matrix));
    let candidate_essential_residual = candidate_geometry
        .e_matrix
        .and_then(|matrix| context.residual(&key, ResidualModelKind::Essential, matrix));
    let candidate_fundamental_residual = candidate_geometry
        .f_matrix
        .and_then(|matrix| context.residual(&key, ResidualModelKind::Fundamental, matrix));
    let candidate_homography_residual = candidate_geometry
        .h_matrix
        .and_then(|matrix| context.residual(&key, ResidualModelKind::Homography, matrix));

    let trace_selected_residual = residual_by_kind(
        trace_selected_model,
        trace_essential_residual,
        trace_fundamental_residual,
        trace_homography_residual,
    );
    let candidate_selected_residual = residual_by_kind(
        candidate_selected_model,
        candidate_essential_residual,
        candidate_fundamental_residual,
        candidate_homography_residual,
    );
    let trace_margin = residual_margin(context, trace_selected_model, trace_selected_residual);
    let candidate_margin = residual_margin(
        context,
        candidate_selected_model,
        candidate_selected_residual,
    );

    MatchResidualSample {
        point2d_idx1: key.point2d_idx1,
        point2d_idx2: key.point2d_idx2,
        trace_selected_residual,
        candidate_selected_residual,
        trace_essential_residual,
        trace_fundamental_residual,
        trace_homography_residual,
        candidate_essential_residual,
        candidate_fundamental_residual,
        candidate_homography_residual,
        trace_margin,
        candidate_margin,
        trace_inlier_under_selected_model: trace_margin.map(|margin| margin <= 0.0),
        candidate_inlier_under_selected_model: candidate_margin.map(|margin| margin <= 0.0),
        trace_selected_model,
        candidate_selected_model,
    }
}

fn residual_by_kind(
    kind: Option<ResidualModelKind>,
    essential: Option<f64>,
    fundamental: Option<f64>,
    homography: Option<f64>,
) -> Option<f64> {
    match kind? {
        ResidualModelKind::Essential => essential,
        ResidualModelKind::Fundamental => fundamental,
        ResidualModelKind::Homography => homography,
    }
}

fn residual_margin(
    context: &PairResidualContext,
    kind: Option<ResidualModelKind>,
    residual: Option<f64>,
) -> Option<f64> {
    let kind = kind?;
    residual.map(|residual| residual - context.threshold_sq(kind))
}

fn residual_model_kind_from_source(source: &str) -> Option<ResidualModelKind> {
    match source {
        "essential" => Some(ResidualModelKind::Essential),
        "fundamental" => Some(ResidualModelKind::Fundamental),
        "homography" => Some(ResidualModelKind::Homography),
        _ => None,
    }
}

fn infer_candidate_selected_model(
    geometry: &ColmapTwoViewGeometry,
    inlier_set: &BTreeSet<MatchKey>,
    context: &PairResidualContext,
) -> Option<ResidualModelKind> {
    [
        (ResidualModelKind::Essential, geometry.e_matrix),
        (ResidualModelKind::Fundamental, geometry.f_matrix),
        (ResidualModelKind::Homography, geometry.h_matrix),
    ]
    .into_iter()
    .filter_map(|(kind, matrix)| {
        let matrix = matrix?;
        let predicted = predicted_inlier_set(kind, matrix, context);
        let intersection = predicted.intersection(inlier_set).count();
        let union = predicted.union(inlier_set).count();
        let jaccard = ratio(intersection, union);
        Some((kind, jaccard, intersection, predicted.len()))
    })
    .max_by(|left, right| {
        left.1
            .partial_cmp(&right.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.2.cmp(&right.2))
            .then_with(|| left.3.cmp(&right.3))
    })
    .map(|(kind, _, _, _)| kind)
}

fn predicted_inlier_set(
    kind: ResidualModelKind,
    matrix: [f64; 9],
    context: &PairResidualContext,
) -> BTreeSet<MatchKey> {
    let threshold_sq = context.threshold_sq(kind);
    let mut set = BTreeSet::new();
    for key in &context.matches {
        let Some(residual) = context.residual(key, kind, matrix) else {
            continue;
        };
        if residual <= threshold_sq {
            set.insert(key.clone());
        }
    }
    set
}

fn vector3_from_ray(ray: [f64; 3]) -> Vector3<f64> {
    let ray = Vector3::new(ray[0], ray[1], ray[2]);
    let norm = ray.norm();
    if norm > 1.0e-12 && norm.is_finite() {
        ray / norm
    } else {
        Vector3::new(0.0, 0.0, 1.0)
    }
}

fn squared_sampson_error(x1: &Vector3<f64>, x2: &Vector3<f64>, matrix: &Matrix3<f64>) -> f64 {
    let mx1 = matrix * x1;
    let mtx2 = matrix.transpose() * x2;
    let num = x2.dot(&(matrix * x1));
    let denom = mx1.x * mx1.x + mx1.y * mx1.y + mtx2.x * mtx2.x + mtx2.y * mtx2.y;
    if denom <= 1.0e-24 {
        f64::INFINITY
    } else {
        num * num / denom
    }
}

fn homography_forward_error(
    x1: &Vector3<f64>,
    x2: &Vector3<f64>,
    homography: &Matrix3<f64>,
) -> f64 {
    let projected = homography * x1;
    if projected.z.abs() <= 1.0e-12 || !projected.z.is_finite() {
        return f64::INFINITY;
    }
    let px = projected.x / projected.z;
    let py = projected.y / projected.z;
    let x2x = x2.x / x2.z;
    let x2y = x2.y / x2.z;
    (px - x2x).powi(2) + (py - x2y).powi(2)
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
