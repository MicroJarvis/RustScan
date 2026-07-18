use crate::colmap_image::load_colmap_grayscale_u8;
use crate::correspondence_graph::{pair_id_to_image_pair, FeatureMatch};
use crate::database::{
    ColmapDatabase, ColmapDatabaseImage, ColmapDescriptors, ColmapKeypoint, ColmapTwoViewGeometry,
};
use crate::feature_matching::{generate_matching_pairs, MatchingPairStrategy};
use crate::geometry::{estimate_pair_geometry_with_options_and_cameras, PairEstimationOptions};
#[cfg(feature = "gpu-wgpu")]
use crate::gpu::WgpuSiftMatcher;
use crate::mapper::pair_geometry_to_colmap_two_view_geometry;
use crate::sift::{match_sift_with_options, SiftFeatures, SiftMatchingOptions};
use crate::two_view::{
    diagnose_calibrated_two_view_with_observations_rays_and_cameras,
    diagnose_stored_two_view_models, TwoViewDiagnostics, TwoViewOptions,
    TwoViewStoredModelDiagnostics,
};
use crate::types::{CameraModel, ImageFrame, PairGeometry};
use anyhow::{bail, Context, Result};
use lowe_sift::Descriptor;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Instant;

const COLMAP_EXISTING_MATCH_BATCH_SIZE: usize = 1000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchFeaturesPairReport {
    pub left_image: String,
    pub right_image: String,
    pub num_matches: usize,
    pub num_inliers: usize,
    pub triangulated: usize,
    pub two_view_config: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchFeaturesReport {
    pub database: PathBuf,
    pub pair_count: usize,
    pub matched_pairs: usize,
    pub verified_pairs: usize,
    pub total_matches: usize,
    pub matching_seconds: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verifier_trace: Option<MatchFeaturesVerifierTrace>,
    pub pairs: Vec<MatchFeaturesPairReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchFeaturesVerifierTrace {
    pub mode: String,
    pub worker_count: usize,
    pub events: Vec<MatchFeaturesVerifierEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchFeaturesVerifierEvent {
    pub worker_id: usize,
    pub dequeue_order: usize,
    pub complete_order: usize,
    pub left_index: usize,
    pub right_index: usize,
    pub left_image: String,
    pub right_image: String,
    pub num_matches: usize,
    pub num_inliers: usize,
    pub triangulated: usize,
    pub two_view_config: i32,
}

#[derive(Debug, Clone)]
pub struct MatchFeaturesOptions {
    pub pair_strategy: MatchingPairStrategy,
    pub sift_matching: SiftMatchingOptions,
    pub essential_threshold_px: f32,
    pub essential_iterations: u32,
    pub min_inliers: usize,
    pub min_triangulated: usize,
    pub min_num_matches: usize,
    pub random_seed: i32,
    pub clear_existing: bool,
    pub use_existing_matches: bool,
    pub existing_match_batch_size: usize,
}

#[derive(Debug, Clone)]
pub struct DebugTwoViewOptions {
    pub essential_threshold_px: f32,
    pub essential_iterations: u32,
    pub min_inliers: usize,
    pub min_triangulated: usize,
    pub random_seed: i32,
}

impl Default for DebugTwoViewOptions {
    fn default() -> Self {
        let match_options = MatchFeaturesOptions::default();
        Self {
            essential_threshold_px: match_options.essential_threshold_px,
            essential_iterations: match_options.essential_iterations,
            min_inliers: match_options.min_inliers,
            min_triangulated: match_options.min_triangulated,
            random_seed: match_options.random_seed,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugTwoViewReport {
    pub left_image: String,
    pub right_image: String,
    pub num_matches: usize,
    pub left_index: usize,
    pub right_index: usize,
    pub sampler_seed: u64,
    pub diagnostics: Option<TwoViewDiagnostics>,
    pub stored_models: Option<TwoViewStoredModelDiagnostics>,
    pub estimate_config: Option<i32>,
    pub estimate_inliers: Option<usize>,
    pub estimate_triangulated: Option<usize>,
    pub estimate_mean_reprojection_error_px: Option<f32>,
    pub estimate_rotation_deg: Option<f32>,
    pub estimate_median_triangulation_angle_deg: Option<f32>,
}

impl Default for MatchFeaturesOptions {
    fn default() -> Self {
        Self {
            pair_strategy: MatchingPairStrategy::default(),
            sift_matching: SiftMatchingOptions::default(),
            essential_threshold_px: 4.0,
            essential_iterations: 10_000,
            min_inliers: 15,
            min_triangulated: 4,
            min_num_matches: 15,
            random_seed: -1,
            clear_existing: true,
            use_existing_matches: false,
            existing_match_batch_size: COLMAP_EXISTING_MATCH_BATCH_SIZE,
        }
    }
}

pub fn match_features_to_database(
    database_path: &Path,
    options: &MatchFeaturesOptions,
) -> Result<MatchFeaturesReport> {
    options.sift_matching.check()?;
    let started = Instant::now();
    let db = ColmapDatabase::open(database_path)?;
    let mut images = db.read_all_images()?;
    if images.len() < 2 {
        bail!("database needs at least two images for matching");
    }
    images.sort_by(|left, right| left.name.cmp(&right.name));

    let (frames, cameras) =
        load_database_frames_and_cameras(&db, &images, !options.use_existing_matches)?;

    let image_id_by_index = images
        .iter()
        .enumerate()
        .map(|(idx, image)| (idx, image.image_id))
        .collect::<Vec<_>>();

    let pair_batch = if options.use_existing_matches {
        existing_match_pair_reports(&db, &frames, &cameras, &images, options)?
    } else {
        computed_match_pair_reports(&frames, &cameras, options)?
    };
    let PairReportBatch {
        pair_count,
        reports: pair_reports,
        verifier_trace,
    } = pair_batch;

    let (mut reports, total_matches) = db.with_transaction(|| {
        if options.clear_existing && !options.use_existing_matches {
            db.clear_matches()?;
            db.clear_two_view_geometries()?;
        } else if options.use_existing_matches {
            db.clear_two_view_geometries()?;
        }

        let mut reports = Vec::with_capacity(pair_reports.len());
        let mut total_matches = 0usize;
        for (left, right, matches, geometry) in pair_reports {
            let left_image_id = image_id_by_index[left].1;
            let right_image_id = image_id_by_index[right].1;
            let feature_matches = matches
                .iter()
                .map(|match_| FeatureMatch {
                    point2d_idx1: match_.query_idx,
                    point2d_idx2: match_.train_idx,
                })
                .collect::<Vec<_>>();
            if !options.use_existing_matches {
                if db.exists_matches(left_image_id, right_image_id)? {
                    db.delete_matches(left_image_id, right_image_id)?;
                }
                db.write_matches(left_image_id, right_image_id, &feature_matches)?;
            }
            total_matches += matches.len();

            let (inliers, triangulated, config) = if let Some(geometry) = geometry.as_ref() {
                let colmap_geometry = pair_geometry_to_colmap_two_view_geometry(geometry);
                if db.exists_two_view_geometry(left_image_id, right_image_id)? {
                    db.update_two_view_geometry(left_image_id, right_image_id, &colmap_geometry)?;
                } else {
                    db.write_two_view_geometry(left_image_id, right_image_id, &colmap_geometry)?;
                }
                (
                    geometry.inliers,
                    geometry.triangulated,
                    geometry.two_view_config,
                )
            } else if options.use_existing_matches {
                let colmap_geometry = ColmapTwoViewGeometry::default();
                if db.exists_two_view_geometry(left_image_id, right_image_id)? {
                    db.update_two_view_geometry(left_image_id, right_image_id, &colmap_geometry)?;
                } else {
                    db.write_two_view_geometry(left_image_id, right_image_id, &colmap_geometry)?;
                }
                (0, 0, colmap_geometry.config)
            } else {
                if db.exists_two_view_geometry(left_image_id, right_image_id)? {
                    db.delete_two_view_geometry(left_image_id, right_image_id)?;
                }
                (0, 0, -1)
            };

            reports.push(MatchFeaturesPairReport {
                left_image: frames[left].name.clone(),
                right_image: frames[right].name.clone(),
                num_matches: matches.len(),
                num_inliers: inliers,
                triangulated,
                two_view_config: config,
            });
        }
        Ok((reports, total_matches))
    })?;
    reports.sort_by(|left, right| {
        left.left_image
            .cmp(&right.left_image)
            .then_with(|| left.right_image.cmp(&right.right_image))
    });

    Ok(MatchFeaturesReport {
        database: database_path.to_path_buf(),
        pair_count,
        matched_pairs: reports.len(),
        verified_pairs: reports
            .iter()
            .filter(|pair| pair.num_inliers >= options.min_inliers)
            .count(),
        total_matches,
        matching_seconds: started.elapsed().as_secs_f64(),
        verifier_trace,
        pairs: reports,
    })
}

pub fn debug_two_view_database_pair(
    database_path: &Path,
    left_image: &str,
    right_image: &str,
    options: &DebugTwoViewOptions,
) -> Result<DebugTwoViewReport> {
    let db = ColmapDatabase::open_read_only(database_path)?;
    let left = db
        .read_image_with_name(left_image)?
        .with_context(|| format!("missing image {left_image}"))?;
    let right = db
        .read_image_with_name(right_image)?
        .with_context(|| format!("missing image {right_image}"))?;
    let mut all_images = db.read_all_images()?;
    all_images.sort_by(|left, right| left.name.cmp(&right.name));
    let left_index = all_images
        .iter()
        .position(|image| image.image_id == left.image_id)
        .with_context(|| format!("missing sorted image index for {left_image}"))?;
    let right_index = all_images
        .iter()
        .position(|image| image.image_id == right.image_id)
        .with_context(|| format!("missing sorted image index for {right_image}"))?;
    let sampler_seed = pair_sampler_seed(left_index, right_index);
    let images = vec![left.clone(), right.clone()];
    let (frames, cameras) = load_database_frames_and_cameras(&db, &images, false)?;
    let matches = db
        .read_matches(left.image_id, right.image_id)?
        .into_iter()
        .map(Into::into)
        .collect::<Vec<rustslam::Match>>();

    let prepared = prepare_pair_observations(
        &frames[0],
        &frames[1],
        &matches,
        cameras[0],
        cameras[1],
        options.min_inliers.max(8),
    );
    let two_view_options = debug_two_view_options(cameras[0], cameras[1], options, sampler_seed);
    let diagnostics = prepared.as_ref().and_then(|prepared| {
        diagnose_calibrated_two_view_with_observations_rays_and_cameras(
            &prepared.norm_left,
            &prepared.norm_right,
            &prepared.obs_left_px,
            &prepared.obs_right_px,
            Some(&prepared.ray_left),
            Some(&prepared.ray_right),
            cameras[0],
            cameras[1],
            &two_view_options,
        )
    });
    let stored_models = if let Some(prepared) = prepared.as_ref() {
        if db.exists_two_view_geometry(left.image_id, right.image_id)? {
            let stored = db.read_two_view_geometry(left.image_id, right.image_id)?;
            Some(diagnose_stored_two_view_models(
                &prepared.obs_left_px,
                &prepared.obs_right_px,
                &prepared.ray_left,
                &prepared.ray_right,
                stored.e_matrix,
                stored.f_matrix,
                stored.h_matrix,
                two_view_options.ransac_threshold,
                two_view_options.ransac_max_error_px,
            ))
        } else {
            None
        }
    } else {
        None
    };
    let estimate = estimate_pair_geometry_with_options_and_cameras(
        left_index,
        right_index,
        &frames[0],
        &frames[1],
        &matches,
        cameras[0],
        cameras[1],
        options.essential_threshold_px,
        options.essential_iterations,
        options.min_inliers,
        options.min_triangulated,
        PairEstimationOptions {
            max_pose_matches: 0,
            refine_sampson: false,
            ransac_random_seed: options.random_seed,
            expand_dense_inliers: false,
            ..PairEstimationOptions::default()
        },
    );

    Ok(DebugTwoViewReport {
        left_image: left_image.to_string(),
        right_image: right_image.to_string(),
        num_matches: matches.len(),
        left_index,
        right_index,
        sampler_seed,
        diagnostics,
        stored_models,
        estimate_config: estimate.as_ref().map(|estimate| estimate.two_view_config),
        estimate_inliers: estimate.as_ref().map(|estimate| estimate.inliers),
        estimate_triangulated: estimate.as_ref().map(|estimate| estimate.triangulated),
        estimate_mean_reprojection_error_px: estimate
            .as_ref()
            .map(|estimate| estimate.mean_reprojection_error_px),
        estimate_rotation_deg: estimate.as_ref().map(|estimate| estimate.rotation_deg),
        estimate_median_triangulation_angle_deg: estimate
            .as_ref()
            .map(|estimate| estimate.median_triangulation_angle_deg),
    })
}

struct PreparedPairObservations {
    norm_left: Vec<[f32; 2]>,
    norm_right: Vec<[f32; 2]>,
    obs_left_px: Vec<[f32; 2]>,
    obs_right_px: Vec<[f32; 2]>,
    ray_left: Vec<[f64; 3]>,
    ray_right: Vec<[f64; 3]>,
}

fn prepare_pair_observations(
    left: &ImageFrame,
    right: &ImageFrame,
    matches: &[rustslam::Match],
    left_camera: CameraModel,
    right_camera: CameraModel,
    min_required: usize,
) -> Option<PreparedPairObservations> {
    if matches.len() < min_required {
        return None;
    }
    let mut norm_left = Vec::with_capacity(matches.len());
    let mut norm_right = Vec::with_capacity(matches.len());
    let mut obs_left_px = Vec::with_capacity(matches.len());
    let mut obs_right_px = Vec::with_capacity(matches.len());
    let mut ray_left = Vec::with_capacity(matches.len());
    let mut ray_right = Vec::with_capacity(matches.len());
    for m in matches {
        let li = m.query_idx as usize;
        let ri = m.train_idx as usize;
        if li >= left.keypoints.len() || ri >= right.keypoints.len() {
            continue;
        }
        let lk = &left.keypoints[li];
        let rk = &right.keypoints[ri];
        let Some(left_xy) = left_camera.cam_from_img_f32(lk.x(), lk.y()) else {
            continue;
        };
        let Some(right_xy) = right_camera.cam_from_img_f32(rk.x(), rk.y()) else {
            continue;
        };
        let Some(left_ray) = left_camera.cam_ray_from_img(lk.x() as f64, lk.y() as f64) else {
            continue;
        };
        let Some(right_ray) = right_camera.cam_ray_from_img(rk.x() as f64, rk.y() as f64) else {
            continue;
        };
        norm_left.push(left_xy);
        norm_right.push(right_xy);
        obs_left_px.push([lk.x(), lk.y()]);
        obs_right_px.push([rk.x(), rk.y()]);
        ray_left.push(left_ray);
        ray_right.push(right_ray);
    }
    if norm_left.len() < min_required {
        return None;
    }

    Some(PreparedPairObservations {
        norm_left,
        norm_right,
        obs_left_px,
        obs_right_px,
        ray_left,
        ray_right,
    })
}

fn debug_two_view_options(
    left_camera: CameraModel,
    right_camera: CameraModel,
    options: &DebugTwoViewOptions,
    sampler_seed: u64,
) -> TwoViewOptions {
    TwoViewOptions {
        ransac_max_error_px: options.essential_threshold_px as f64,
        ransac_threshold: 0.5
            * (left_camera.cam_from_img_threshold(options.essential_threshold_px as f64)
                + right_camera.cam_from_img_threshold(options.essential_threshold_px as f64)),
        ransac_min_inlier_ratio: 0.25,
        ransac_min_iterations: 100,
        ransac_max_iterations: options.essential_iterations,
        ransac_random_seed: options.random_seed,
        random_seed: sampler_seed,
        loransac_num_lo_steps: 0,
        min_inliers: options.min_inliers,
        min_inlier_ratio: 0.0,
        min_triangulated: options.min_triangulated,
        min_e_f_inlier_ratio: 0.95,
        max_h_inlier_ratio: 0.8,
        force_h_use: false,
        multiple_models: false,
        multiple_ignore_watermark: true,
        detect_watermark: true,
        watermark_min_inlier_ratio: 0.7,
        watermark_border_size: 0.1,
        watermark_detection_max_error_px: 4.0,
        filter_stationary_matches: false,
        stationary_matches_max_error_px: 4.0,
        use_hartley_refinement: true,
        use_five_point: true,
    }
}

fn pair_sampler_seed(left_idx: usize, right_idx: usize) -> u64 {
    ((left_idx as u64) << 32) ^ right_idx as u64 ^ 0x243f_6a88_85a3_08d3
}

type PairReportInput = (usize, usize, Vec<rustslam::Match>, Option<PairGeometry>);

struct PairReportBatch {
    pair_count: usize,
    reports: Vec<PairReportInput>,
    verifier_trace: Option<MatchFeaturesVerifierTrace>,
}

fn load_database_frames_and_cameras(
    db: &ColmapDatabase,
    images: &[ColmapDatabaseImage],
    load_sift_descriptors: bool,
) -> Result<(Vec<ImageFrame>, Vec<CameraModel>)> {
    let mut frames = Vec::with_capacity(images.len());
    let mut cameras = Vec::with_capacity(images.len());
    for (idx, image) in images.iter().enumerate() {
        let db_camera = db
            .read_camera(image.camera_id)?
            .with_context(|| format!("missing camera_id={}", image.camera_id))?;
        let camera = CameraModel::from_colmap(
            db_camera.camera.model_id,
            db_camera.camera.width,
            db_camera.camera.height,
            &db_camera.camera.params,
        )
        .with_context(|| format!("unsupported camera_id={}", image.camera_id))?;
        let keypoints = db.read_keypoints(image.image_id)?;
        let sift = if load_sift_descriptors {
            let descriptors = db.read_descriptors(image.image_id)?;
            sift_features_from_database(&keypoints, &descriptors)?
        } else {
            SiftFeatures {
                keypoints: keypoints.iter().map(|kp| kp.to_keypoint()).collect(),
                colmap_keypoints: keypoints.clone(),
                ..SiftFeatures::default()
            }
        };
        let rust_keypoints = keypoints
            .iter()
            .map(|kp| kp.to_keypoint())
            .collect::<Vec<_>>();
        frames.push(ImageFrame {
            id: idx,
            name: image.name.clone(),
            path: PathBuf::from(&image.name),
            width: db_camera.camera.width,
            height: db_camera.camera.height,
            keypoints: rust_keypoints,
            descriptors: rustslam::Descriptors::new(),
            sift,
            wide_descriptors: crate::wide::WideDescriptors {
                data: Vec::new(),
                dim: 0,
                count: 0,
            },
            strong_feature_indices: Vec::new(),
            colors: Vec::new(),
        });
        cameras.push(camera);
    }
    Ok((frames, cameras))
}

fn computed_match_pair_reports(
    frames: &[ImageFrame],
    cameras: &[CameraModel],
    options: &MatchFeaturesOptions,
) -> Result<PairReportBatch> {
    let pairs = match options.pair_strategy {
        MatchingPairStrategy::VocabTree { num_images } => {
            vocab_tree_pairs_from_frames(frames, num_images, options.random_seed)
        }
        strategy => generate_matching_pairs(frames.len(), strategy),
    };
    let pair_count = pairs.len();
    #[cfg(feature = "gpu-wgpu")]
    let gpu_matcher = if options.sift_matching.use_gpu {
        Some(WgpuSiftMatcher::try_new()?)
    } else {
        None
    };
    #[cfg(not(feature = "gpu-wgpu"))]
    if options.sift_matching.use_gpu {
        bail!("RustSFM was built without gpu-wgpu support");
    }

    #[cfg(feature = "gpu-wgpu")]
    let reports = if let Some(matcher) = gpu_matcher.as_ref() {
        let mut reports = Vec::with_capacity(pairs.len());
        for &(left, right) in &pairs {
            let matches = matcher.match_descriptors(
                &frames[left].sift.descriptors_u8,
                &frames[right].sift.descriptors_u8,
                &options.sift_matching,
            )?;
            if let Some(report) =
                estimate_existing_or_computed_pair(left, right, matches, frames, cameras, options)
            {
                reports.push(report);
            }
        }
        reports
    } else {
        pairs
            .par_iter()
            .filter_map(|&(left, right)| {
                let matches = match_sift_with_options(
                    &frames[left].sift,
                    &frames[right].sift,
                    &options.sift_matching,
                );
                estimate_existing_or_computed_pair(left, right, matches, frames, cameras, options)
            })
            .collect()
    };
    #[cfg(not(feature = "gpu-wgpu"))]
    let reports = pairs
        .par_iter()
        .filter_map(|&(left, right)| {
            let matches = match_sift_with_options(
                &frames[left].sift,
                &frames[right].sift,
                &options.sift_matching,
            );
            estimate_existing_or_computed_pair(left, right, matches, frames, cameras, options)
        })
        .collect();
    Ok(PairReportBatch {
        pair_count,
        reports,
        verifier_trace: None,
    })
}

fn existing_match_pair_reports(
    db: &ColmapDatabase,
    frames: &[ImageFrame],
    cameras: &[CameraModel],
    images: &[ColmapDatabaseImage],
    options: &MatchFeaturesOptions,
) -> Result<PairReportBatch> {
    let image_index_by_id = images
        .iter()
        .enumerate()
        .map(|(idx, image)| (image.image_id, idx))
        .collect::<HashMap<_, _>>();
    let mut pairs = Vec::new();
    for (pair_id, _) in db.read_num_matches()? {
        let (image_id1, image_id2) =
            pair_id_to_image_pair(pair_id).map_err(|err| anyhow::anyhow!("{err:?}"))?;
        let Some(&left) = image_index_by_id.get(&image_id1) else {
            continue;
        };
        let Some(&right) = image_index_by_id.get(&image_id2) else {
            continue;
        };
        let matches = db
            .read_matches(image_id1, image_id2)?
            .into_iter()
            .map(Into::into)
            .collect::<Vec<_>>();
        pairs.push((left, right, matches));
    }
    let pair_count = pairs.len();
    let (reports, verifier_trace) = if colmap_fifo_verifier_enabled(options) {
        colmap_fifo_existing_match_pair_reports(pairs, frames, cameras, options)?
    } else {
        let reports = pairs
            .into_par_iter()
            .filter_map(|(left, right, matches)| {
                estimate_existing_or_computed_pair(left, right, matches, frames, cameras, options)
            })
            .collect();
        (reports, None)
    };
    Ok(PairReportBatch {
        pair_count,
        reports,
        verifier_trace,
    })
}

type ExistingMatchPairInput = (usize, usize, Vec<rustslam::Match>);

fn colmap_fifo_verifier_enabled(options: &MatchFeaturesOptions) -> bool {
    options.use_existing_matches
        && options.random_seed < 0
        && std::env::var_os("RUSTSFM_COLMAP_FIFO_VERIFIER").is_some()
        && std::env::var_os("RUSTSFM_COLMAP_SHARED_RANSAC_STREAM").is_some()
}

fn colmap_fifo_verifier_threads() -> usize {
    let requested = std::env::var("RUSTSFM_COLMAP_FIFO_VERIFIER_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| {
            thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
        });
    requested
}

fn colmap_fifo_existing_match_pair_reports(
    pairs: Vec<ExistingMatchPairInput>,
    frames: &[ImageFrame],
    cameras: &[CameraModel],
    options: &MatchFeaturesOptions,
) -> Result<(Vec<PairReportInput>, Option<MatchFeaturesVerifierTrace>)> {
    if let Some(trace_path) = colmap_fifo_replay_trace_path() {
        return colmap_replay_existing_match_pair_reports(
            pairs,
            frames,
            cameras,
            options,
            &trace_path,
        );
    }

    if pairs.is_empty() {
        return Ok((
            Vec::new(),
            if std::env::var_os("RUSTSFM_COLMAP_FIFO_VERIFIER_TRACE").is_some() {
                Some(MatchFeaturesVerifierTrace {
                    mode: "colmap_fifo_shared_ransac_stream".to_string(),
                    worker_count: 0,
                    events: Vec::new(),
                })
            } else {
                None
            },
        ));
    }

    let input_queue = Arc::new(ColmapFifoVerifierQueue::new());
    let output_queue = Arc::new(ColmapFifoVerifierOutputQueue::new());
    let worker_count = colmap_fifo_verifier_threads();
    let trace_enabled = std::env::var_os("RUSTSFM_COLMAP_FIFO_VERIFIER_TRACE").is_some();
    let batch_size = options.existing_match_batch_size.max(2);
    let mut worker_results = Vec::<ColmapFifoWorkerResult>::with_capacity(pairs.len());

    thread::scope(|scope| {
        for worker_id in 0..worker_count {
            let input_queue = Arc::clone(&input_queue);
            let output_queue = Arc::clone(&output_queue);
            scope.spawn(move || loop {
                let Some((dequeue_order, (left, right, matches))) = input_queue.pop() else {
                    return;
                };
                let report = estimate_existing_or_computed_pair(
                    left, right, matches, frames, cameras, options,
                );
                output_queue.push(worker_id, dequeue_order, report, frames, trace_enabled);
            });
        }

        let mut batch = Vec::with_capacity(batch_size);
        for pair in pairs {
            batch.push(pair);
            if batch.len() == batch_size {
                for pair in batch.drain(..) {
                    input_queue.push(pair);
                }
                for _ in 0..batch_size {
                    if let Some(result) = output_queue.pop() {
                        worker_results.push(result);
                    }
                }
            }
        }
        if !batch.is_empty() {
            let batch_len = batch.len();
            for pair in batch.drain(..) {
                input_queue.push(pair);
            }
            for _ in 0..batch_len {
                if let Some(result) = output_queue.pop() {
                    worker_results.push(result);
                }
            }
        }
        input_queue.stop();
    });

    let mut reports = Vec::new();
    let mut events = Vec::new();
    for mut result in worker_results {
        if let Some(report) = result.report.take() {
            reports.push(report);
        }
        if let Some(event) = result.event.take() {
            events.push(event);
        }
    }

    Ok((
        reports,
        if trace_enabled {
            Some(MatchFeaturesVerifierTrace {
                mode: "colmap_fifo_shared_ransac_stream".to_string(),
                worker_count,
                events,
            })
        } else {
            None
        },
    ))
}

fn colmap_fifo_replay_trace_path() -> Option<PathBuf> {
    std::env::var_os("RUSTSFM_COLMAP_FIFO_VERIFIER_REPLAY_TRACE").map(PathBuf::from)
}

#[derive(Debug, Clone, Deserialize)]
struct ColmapVerifierReplayRoot {
    worker_count: Option<usize>,
    events: Option<Vec<ColmapVerifierReplayEvent>>,
    verifier_trace: Option<ColmapVerifierReplayNested>,
}

#[derive(Debug, Clone, Deserialize)]
struct ColmapVerifierReplayNested {
    worker_count: Option<usize>,
    events: Vec<ColmapVerifierReplayEvent>,
}

#[derive(Debug, Clone, Deserialize)]
struct ColmapVerifierReplayEvent {
    worker_id: usize,
    dequeue_order: usize,
    complete_order: usize,
    left_index: usize,
    right_index: usize,
    left_image: String,
    right_image: String,
}

#[derive(Debug, Clone)]
struct ColmapVerifierReplaySchedule {
    worker_count: usize,
    events: Vec<ColmapVerifierReplayEvent>,
}

fn load_colmap_fifo_replay_schedule(path: &Path) -> Result<ColmapVerifierReplaySchedule> {
    let file = File::open(path)
        .with_context(|| format!("open COLMAP verifier replay trace {}", path.display()))?;
    let root: ColmapVerifierReplayRoot = serde_json::from_reader(file)
        .with_context(|| format!("parse COLMAP verifier replay trace {}", path.display()))?;
    let (worker_count, events) = if let Some(events) = root.events {
        (root.worker_count, events)
    } else if let Some(trace) = root.verifier_trace {
        (trace.worker_count, trace.events)
    } else {
        bail!(
            "COLMAP verifier replay trace {} has neither top-level events nor verifier_trace.events",
            path.display()
        );
    };
    if events.is_empty() {
        bail!(
            "COLMAP verifier replay trace {} has no events",
            path.display()
        );
    }
    let required_workers = events
        .iter()
        .map(|event| event.worker_id + 1)
        .max()
        .unwrap_or(0);
    let worker_count = worker_count
        .unwrap_or(required_workers)
        .max(required_workers);
    if worker_count == 0 {
        bail!(
            "COLMAP verifier replay trace {} has invalid worker_count=0",
            path.display()
        );
    }
    Ok(ColmapVerifierReplaySchedule {
        worker_count,
        events,
    })
}

#[derive(Debug)]
struct ColmapReplayWorkerJob {
    event: ColmapVerifierReplayEvent,
    pair: ExistingMatchPairInput,
}

fn colmap_replay_existing_match_pair_reports(
    pairs: Vec<ExistingMatchPairInput>,
    frames: &[ImageFrame],
    cameras: &[CameraModel],
    options: &MatchFeaturesOptions,
    trace_path: &Path,
) -> Result<(Vec<PairReportInput>, Option<MatchFeaturesVerifierTrace>)> {
    let schedule = load_colmap_fifo_replay_schedule(trace_path)?;
    if schedule.events.len() != pairs.len() {
        bail!(
            "COLMAP verifier replay trace has {} events, but database has {} existing-match pairs",
            schedule.events.len(),
            pairs.len()
        );
    }

    let mut pairs_by_index = HashMap::<(usize, usize), ExistingMatchPairInput>::new();
    for pair in pairs {
        let key = (pair.0, pair.1);
        if pairs_by_index.insert(key, pair).is_some() {
            bail!(
                "duplicate existing-match pair in database for frame indices {}-{}",
                key.0,
                key.1
            );
        }
    }

    let mut events = schedule.events;
    events.sort_by_key(|event| event.dequeue_order);
    let mut worker_jobs = (0..schedule.worker_count)
        .map(|_| Vec::<ColmapReplayWorkerJob>::new())
        .collect::<Vec<_>>();
    for event in events {
        let key = (event.left_index, event.right_index);
        let pair = pairs_by_index.remove(&key).with_context(|| {
            format!(
                "COLMAP verifier replay event references missing pair {}-{} ({} / {})",
                event.left_index, event.right_index, event.left_image, event.right_image
            )
        })?;
        worker_jobs[event.worker_id].push(ColmapReplayWorkerJob { event, pair });
    }
    if !pairs_by_index.is_empty() {
        let mut missing = pairs_by_index.keys().copied().collect::<Vec<_>>();
        missing.sort_unstable();
        bail!(
            "COLMAP verifier replay trace is missing {} database pairs, first missing frame indices {:?}",
            missing.len(),
            missing.first()
        );
    }

    let worker_results = Arc::new(Mutex::new(Vec::<ColmapFifoWorkerResult>::new()));
    thread::scope(|scope| {
        for (worker_id, jobs) in worker_jobs.into_iter().enumerate() {
            let worker_results = Arc::clone(&worker_results);
            scope.spawn(move || {
                for job in jobs {
                    let report = estimate_existing_or_computed_pair(
                        job.pair.0, job.pair.1, job.pair.2, frames, cameras, options,
                    );
                    let event = report.as_ref().map(|report| {
                        verifier_event_from_report(
                            worker_id,
                            job.event.dequeue_order,
                            job.event.complete_order,
                            report,
                            frames,
                        )
                    });
                    if let Ok(mut results) = worker_results.lock() {
                        results.push(ColmapFifoWorkerResult { report, event });
                    }
                }
            });
        }
    });

    let mut worker_results = Arc::try_unwrap(worker_results)
        .map_err(|_| anyhow::anyhow!("COLMAP replay worker results still shared"))?
        .into_inner()
        .map_err(|_| anyhow::anyhow!("COLMAP replay worker results lock poisoned"))?;
    worker_results.sort_by_key(|result| {
        result
            .event
            .as_ref()
            .map(|event| event.complete_order)
            .unwrap_or(usize::MAX)
    });

    let mut reports = Vec::new();
    let mut trace_events = Vec::new();
    for mut result in worker_results {
        if let Some(report) = result.report.take() {
            reports.push(report);
        }
        if let Some(event) = result.event.take() {
            trace_events.push(event);
        }
    }
    Ok((
        reports,
        Some(MatchFeaturesVerifierTrace {
            mode: "colmap_fifo_shared_ransac_stream_replay".to_string(),
            worker_count: schedule.worker_count,
            events: trace_events,
        }),
    ))
}

#[derive(Debug)]
struct ColmapFifoWorkerResult {
    report: Option<PairReportInput>,
    event: Option<MatchFeaturesVerifierEvent>,
}

struct ColmapFifoVerifierOutputQueue {
    state: Mutex<ColmapFifoVerifierOutputQueueState>,
    condvar: Condvar,
}

struct ColmapFifoVerifierOutputQueueState {
    next_completion_order: usize,
    jobs: VecDeque<ColmapFifoWorkerResult>,
}

impl ColmapFifoVerifierOutputQueue {
    fn new() -> Self {
        Self {
            state: Mutex::new(ColmapFifoVerifierOutputQueueState {
                next_completion_order: 0,
                jobs: VecDeque::new(),
            }),
            condvar: Condvar::new(),
        }
    }

    fn push(
        &self,
        worker_id: usize,
        dequeue_order: usize,
        report: Option<PairReportInput>,
        frames: &[ImageFrame],
        trace_enabled: bool,
    ) {
        if let Ok(mut state) = self.state.lock() {
            let completion_order = state.next_completion_order;
            state.next_completion_order += 1;
            let event = if trace_enabled {
                report.as_ref().map(|report| {
                    verifier_event_from_report(
                        worker_id,
                        dequeue_order,
                        completion_order,
                        report,
                        frames,
                    )
                })
            } else {
                None
            };
            state
                .jobs
                .push_back(ColmapFifoWorkerResult { report, event });
            self.condvar.notify_one();
        }
    }

    fn pop(&self) -> Option<ColmapFifoWorkerResult> {
        let mut state = self.state.lock().ok()?;
        loop {
            if let Some(job) = state.jobs.pop_front() {
                return Some(job);
            }
            state = self.condvar.wait(state).ok()?;
        }
    }
}

fn verifier_event_from_report(
    worker_id: usize,
    dequeue_order: usize,
    complete_order: usize,
    report: &PairReportInput,
    frames: &[ImageFrame],
) -> MatchFeaturesVerifierEvent {
    let (left, right, matches, geometry) = report;
    let (num_inliers, triangulated, two_view_config) = geometry
        .as_ref()
        .map(|geometry| {
            (
                geometry.inliers,
                geometry.triangulated,
                geometry.two_view_config,
            )
        })
        .unwrap_or((0, 0, -1));
    MatchFeaturesVerifierEvent {
        worker_id,
        dequeue_order,
        complete_order,
        left_index: *left,
        right_index: *right,
        left_image: frames[*left].name.clone(),
        right_image: frames[*right].name.clone(),
        num_matches: matches.len(),
        num_inliers,
        triangulated,
        two_view_config,
    }
}

struct ColmapFifoVerifierQueue {
    state: Mutex<ColmapFifoVerifierQueueState>,
    condvar: Condvar,
}

struct ColmapFifoVerifierQueueState {
    stopped: bool,
    next_dequeue_order: usize,
    jobs: VecDeque<ExistingMatchPairInput>,
}

impl ColmapFifoVerifierQueue {
    fn new() -> Self {
        Self {
            state: Mutex::new(ColmapFifoVerifierQueueState {
                stopped: false,
                next_dequeue_order: 0,
                jobs: VecDeque::new(),
            }),
            condvar: Condvar::new(),
        }
    }

    fn push(&self, job: ExistingMatchPairInput) {
        if let Ok(mut state) = self.state.lock() {
            if !state.stopped {
                state.jobs.push_back(job);
                self.condvar.notify_one();
            }
        }
    }

    fn pop(&self) -> Option<(usize, ExistingMatchPairInput)> {
        let mut state = self.state.lock().ok()?;
        loop {
            if let Some(job) = state.jobs.pop_front() {
                let order = state.next_dequeue_order;
                state.next_dequeue_order += 1;
                return Some((order, job));
            }
            if state.stopped {
                return None;
            }
            state = self.condvar.wait(state).ok()?;
        }
    }

    fn stop(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.stopped = true;
        }
        self.condvar.notify_all();
    }
}

fn estimate_existing_or_computed_pair(
    left: usize,
    right: usize,
    matches: Vec<rustslam::Match>,
    frames: &[ImageFrame],
    cameras: &[CameraModel],
    options: &MatchFeaturesOptions,
) -> Option<PairReportInput> {
    let min_matches_for_estimation = if options.use_existing_matches {
        options.min_inliers
    } else {
        options.min_num_matches
    };
    if matches.len() < min_matches_for_estimation {
        if options.use_existing_matches {
            return Some((left, right, matches, None));
        }
        return None;
    }
    let geometry = estimate_pair_geometry_with_options_and_cameras(
        left,
        right,
        &frames[left],
        &frames[right],
        &matches,
        cameras[left],
        cameras[right],
        options.essential_threshold_px,
        options.essential_iterations,
        options.min_inliers,
        options.min_triangulated,
        PairEstimationOptions {
            max_pose_matches: 0,
            refine_sampson: !options.use_existing_matches,
            ransac_random_seed: options.random_seed,
            expand_dense_inliers: !options.use_existing_matches,
            ..PairEstimationOptions::default()
        },
    );
    Some((left, right, matches, geometry))
}

/// Build vocabulary-tree candidate pairs from the in-memory frame descriptors.
///
/// Frames are keyed by their slice index, so the returned `(i32, i32)` image-id
/// pairs map directly back to `(left, right)` frame indices. The vocabulary
/// tree is trained on the union of all frame SIFT descriptors.
pub fn vocab_tree_pairs_from_frames(
    frames: &[ImageFrame],
    num_images: usize,
    random_seed: i32,
) -> Vec<(usize, usize)> {
    let dim = lowe_sift::DESCRIPTOR_LEN;
    let images: Vec<(i32, Vec<f32>)> = frames
        .iter()
        .enumerate()
        .map(|(idx, frame)| {
            let mut descriptors = Vec::with_capacity(frame.sift.descriptors_u8.len() * dim);
            for desc in &frame.sift.descriptors_u8 {
                descriptors.extend(desc.iter().map(|&b| b as f32));
            }
            (idx as i32, descriptors)
        })
        .collect();

    let build_options = crate::retrieval::VocabTreeBuildOptions {
        random_seed: if random_seed < 0 {
            0
        } else {
            random_seed as i64
        },
        ..crate::retrieval::VocabTreeBuildOptions::default()
    };
    let pair_options = crate::retrieval::VocabTreePairOptions {
        num_images: num_images.max(1),
    };
    crate::retrieval::build_vocab_tree_pairs(&images, dim, &build_options, &pair_options)
        .into_iter()
        .filter_map(|(a, b)| {
            let (a, b) = (a as usize, b as usize);
            (a < frames.len() && b < frames.len()).then_some((a.min(b), a.max(b)))
        })
        .collect()
}

fn sift_features_from_database(
    keypoints: &[ColmapKeypoint],
    descriptors: &ColmapDescriptors,
) -> Result<SiftFeatures> {
    const DESCRIPTOR_LEN: usize = 128;
    if descriptors.cols != DESCRIPTOR_LEN {
        bail!(
            "expected SIFT descriptors with {} columns, got {}",
            DESCRIPTOR_LEN,
            descriptors.cols
        );
    }
    if keypoints.len() != descriptors.rows {
        bail!(
            "keypoint/descriptor row mismatch: {} vs {}",
            keypoints.len(),
            descriptors.rows
        );
    }

    let mut descriptors_u8 = Vec::with_capacity(descriptors.rows);
    let mut float_descriptors = Vec::with_capacity(descriptors.rows);
    for row in 0..descriptors.rows {
        let start = row * DESCRIPTOR_LEN;
        let end = start + DESCRIPTOR_LEN;
        let mut values = [0u8; DESCRIPTOR_LEN];
        values.copy_from_slice(&descriptors.data[start..end]);
        let mut floats = [0f32; DESCRIPTOR_LEN];
        for (slot, value) in floats.iter_mut().zip(values.iter()) {
            *slot = *value as f32 / 512.0;
        }
        descriptors_u8.push(values);
        float_descriptors.push(Descriptor::new(floats));
    }

    Ok(SiftFeatures {
        keypoints: keypoints.iter().map(|kp| kp.to_keypoint()).collect(),
        descriptors: float_descriptors,
        colmap_keypoints: keypoints.to_vec(),
        descriptors_u8,
    })
}

pub fn load_rgb_image_for_frame(path: &Path) -> Result<(Vec<u8>, u32, u32)> {
    use image::ImageReader;

    let decoded = ImageReader::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?
        .decode()
        .with_context(|| format!("failed to decode {}", path.display()))?
        .to_rgb8();
    let (width, height) = decoded.dimensions();
    Ok((decoded.into_raw(), width, height))
}

pub fn load_grayscale_image_for_frame(path: &Path) -> Result<(Vec<u8>, u32, u32)> {
    if cfg!(colmap_freeimage) {
        let gray = load_colmap_grayscale_u8(path)?;
        return Ok((gray.data, gray.width, gray.height));
    }
    let rgb = load_rgb_image_for_frame(path)?;
    crate::sift::prepare_grayscale_for_extraction(
        &crate::sift::rgb_to_colmap_gray_u8(&rgb.0, rgb.1, rgb.2)?,
        rgb.1,
        rgb.2,
        crate::sift::SiftExtractionOptions::default().max_image_size,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colmap::ColmapCamera;
    use crate::database::{ColmapDatabaseCamera, ColmapDatabaseImage};
    use tempfile::tempdir;

    #[test]
    fn sift_features_from_database_roundtrips_rows() -> Result<()> {
        let keypoints = vec![ColmapKeypoint::new(1.0, 2.0), ColmapKeypoint::new(3.0, 4.0)];
        let descriptors = ColmapDescriptors::new(
            crate::database::COLMAP_FEATURE_SIFT,
            keypoints.len(),
            128,
            (0..keypoints.len() * 128).map(|v| v as u8).collect(),
        )?;
        let features = sift_features_from_database(&keypoints, &descriptors)?;
        assert_eq!(features.keypoints.len(), 2);
        assert_eq!(features.descriptors_u8.len(), 2);
        Ok(())
    }

    #[test]
    fn database_frame_loader_skips_descriptors_for_existing_matches() -> Result<()> {
        let dir = tempdir()?;
        let db_path = dir.path().join("database.db");
        let db = ColmapDatabase::open(&db_path)?;
        db.write_camera(
            &ColmapDatabaseCamera {
                camera: ColmapCamera {
                    camera_id: 1,
                    model_id: crate::types::COLMAP_PINHOLE,
                    width: 100,
                    height: 100,
                    params: vec![50.0, 50.0, 50.0, 50.0],
                },
                has_prior_focal_length: true,
            },
            true,
        )?;
        for (image_id, name) in [(1, "left.jpg"), (2, "right.jpg")] {
            db.write_image(
                &ColmapDatabaseImage {
                    image_id,
                    name: name.to_string(),
                    camera_id: 1,
                    frame_id: None,
                },
                true,
            )?;
        }

        let left_keypoints = vec![
            ColmapKeypoint::new(42.0, 44.0),
            ColmapKeypoint::new(50.0, 45.0),
            ColmapKeypoint::new(58.0, 46.0),
        ];
        let right_keypoints = vec![
            ColmapKeypoint::new(36.0, 44.0),
            ColmapKeypoint::new(44.0, 45.0),
            ColmapKeypoint::new(52.0, 46.0),
        ];
        db.write_keypoints(1, &left_keypoints)?;
        db.write_keypoints(2, &right_keypoints)?;

        let images = db.read_all_images()?;
        let (frames, cameras) = load_database_frames_and_cameras(&db, &images, false)?;

        assert_eq!(frames.len(), 2);
        assert_eq!(cameras.len(), 2);
        assert_eq!(frames[0].keypoints.len(), 3);
        assert_eq!(frames[0].sift.keypoints.len(), 3);
        assert!(frames[0].sift.descriptors.is_empty());
        assert!(frames[0].sift.descriptors_u8.is_empty());
        Ok(())
    }

    #[test]
    fn matching_failure_preserves_existing_database_results() -> Result<()> {
        let dir = tempdir()?;
        let db_path = dir.path().join("database.db");
        let db = ColmapDatabase::open(&db_path)?;
        db.write_camera(
            &ColmapDatabaseCamera {
                camera: ColmapCamera {
                    camera_id: 1,
                    model_id: crate::types::COLMAP_PINHOLE,
                    width: 100,
                    height: 100,
                    params: vec![50.0, 50.0, 50.0, 50.0],
                },
                has_prior_focal_length: true,
            },
            true,
        )?;
        db.write_image(
            &ColmapDatabaseImage {
                image_id: 1,
                name: "only.jpg".to_string(),
                camera_id: 1,
                frame_id: None,
            },
            true,
        )?;
        let original_matches = vec![FeatureMatch::new(3, 7)];
        db.write_matches(1, 2, &original_matches)?;
        let original_geometry = ColmapTwoViewGeometry {
            inlier_matches: original_matches.clone(),
            config: 2,
            ..ColmapTwoViewGeometry::default()
        };
        db.write_two_view_geometry(1, 2, &original_geometry)?;
        drop(db);

        let error = match_features_to_database(&db_path, &MatchFeaturesOptions::default())
            .expect_err("one-image database must fail validation");
        assert!(error.to_string().contains("at least two images"));

        let db = ColmapDatabase::open(&db_path)?;
        assert_eq!(db.read_matches(1, 2)?, original_matches);
        assert_eq!(db.read_two_view_geometry(1, 2)?, original_geometry);
        Ok(())
    }

    #[test]
    fn existing_match_verifier_uses_min_inliers_as_estimation_gate() {
        let mut options = MatchFeaturesOptions {
            use_existing_matches: true,
            min_num_matches: 15,
            min_inliers: 17,
            ..MatchFeaturesOptions::default()
        };
        let matches = (0..16)
            .map(|idx| rustslam::Match {
                query_idx: idx,
                train_idx: idx,
                distance: 0.0,
            })
            .collect::<Vec<_>>();

        let report = estimate_existing_or_computed_pair(0, 1, matches.clone(), &[], &[], &options)
            .expect("existing-match verifier keeps failed pairs for default geometry rows");
        assert_eq!(report.2.len(), matches.len());
        assert!(report.3.is_none());

        options.use_existing_matches = false;
        options.min_num_matches = 17;
        let skipped = estimate_existing_or_computed_pair(0, 1, matches, &[], &[], &options);
        assert!(skipped.is_none());
    }

    #[test]
    fn colmap_fifo_replay_schedule_loads_top_level_trace() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("trace.json");
        std::fs::write(
            &path,
            r#"{
              "worker_count": 2,
              "events": [
                {
                  "worker_id": 1,
                  "dequeue_order": 4,
                  "complete_order": 7,
                  "left_index": 3,
                  "right_index": 9,
                  "left_image": "left.jpg",
                  "right_image": "right.jpg"
                }
              ]
            }"#,
        )?;

        let schedule = load_colmap_fifo_replay_schedule(&path)?;

        assert_eq!(schedule.worker_count, 2);
        assert_eq!(schedule.events.len(), 1);
        assert_eq!(schedule.events[0].worker_id, 1);
        assert_eq!(schedule.events[0].dequeue_order, 4);
        assert_eq!(schedule.events[0].complete_order, 7);
        assert_eq!(schedule.events[0].left_index, 3);
        assert_eq!(schedule.events[0].right_index, 9);
        Ok(())
    }

    #[test]
    fn colmap_fifo_replay_schedule_loads_nested_rustsfm_trace() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("trace.json");
        std::fs::write(
            &path,
            r#"{
              "verifier_trace": {
                "worker_count": 1,
                "events": [
                  {
                    "worker_id": 0,
                    "dequeue_order": 0,
                    "complete_order": 0,
                    "left_index": 1,
                    "right_index": 2,
                    "left_image": "a.jpg",
                    "right_image": "b.jpg"
                  }
                ]
              }
            }"#,
        )?;

        let schedule = load_colmap_fifo_replay_schedule(&path)?;

        assert_eq!(schedule.worker_count, 1);
        assert_eq!(schedule.events.len(), 1);
        assert_eq!(schedule.events[0].left_image, "a.jpg");
        assert_eq!(schedule.events[0].right_image, "b.jpg");
        Ok(())
    }
}
