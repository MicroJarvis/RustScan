use crate::colmap_image::load_colmap_grayscale_u8;
use crate::correspondence_graph::{pair_id_to_image_pair, FeatureMatch};
use crate::database::{
    ColmapDatabase, ColmapDatabaseImage, ColmapDescriptors, ColmapKeypoint, ColmapTwoViewGeometry,
};
use crate::feature_matching::{generate_matching_pairs, MatchingPairStrategy};
#[cfg(feature = "gpu-wgpu")]
use crate::geometry::estimate_pair_geometry_with_options_and_cameras_gpu;
use crate::geometry::{estimate_pair_geometry_with_options_and_cameras, PairEstimationOptions};
#[cfg(feature = "gpu-wgpu")]
use crate::gpu::{WgpuContext, WgpuModelScorer, WgpuSiftMatcher};
use crate::mapper::pair_geometry_to_colmap_two_view_geometry;
use crate::sift::{match_sift_with_options, SiftFeatures, SiftMatchingOptions};
use crate::task::{SfmTaskContext, SfmTaskEvent, SfmTaskEventKind, SfmTaskOperation, SfmTaskStage};
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
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Condvar, Mutex,
};
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
    pub backend: String,
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
    pub task_pair_batch_size: usize,
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
            task_pair_batch_size: 32,
        }
    }
}

fn matching_backend_name(options: &SiftMatchingOptions) -> &'static str {
    if options.use_gpu {
        "wgpu_match_and_score"
    } else {
        "cpu_match_and_score"
    }
}

pub fn match_features_to_database(
    database_path: &Path,
    options: &MatchFeaturesOptions,
) -> Result<MatchFeaturesReport> {
    let control = crate::task::SfmTaskControl::new();
    let mut sink = |_event: SfmTaskEvent| {};
    let mut task = SfmTaskContext::new(&control, &mut sink);
    match_features_to_database_with_task(database_path, options, &mut task)
}

pub fn match_features_to_database_with_task(
    database_path: &Path,
    options: &MatchFeaturesOptions,
    task: &mut SfmTaskContext<'_>,
) -> Result<MatchFeaturesReport> {
    options.sift_matching.check()?;
    task.checkpoint()?;
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

    let input_pairs = if options.use_existing_matches {
        existing_match_inputs(&db, &images)?
    } else {
        computed_match_inputs(frames.len(), &frames, options)
            .into_iter()
            .map(|(left, right)| MatchPairInput::Computed { left, right })
            .collect::<Vec<_>>()
    };
    let pair_count = input_pairs.len();
    let batch_size = options.task_pair_batch_size.max(1);

    #[cfg(feature = "gpu-wgpu")]
    let computed_backend = if !options.use_existing_matches && options.sift_matching.use_gpu {
        Some(ComputedGpuBackend::new()?)
    } else {
        None
    };
    #[cfg(feature = "gpu-wgpu")]
    let existing_backend = if options.use_existing_matches && options.sift_matching.use_gpu {
        Some(ExistingGpuBackend::new()?)
    } else {
        None
    };
    #[cfg(not(feature = "gpu-wgpu"))]
    if options.sift_matching.use_gpu {
        bail!("RustSFM was built without gpu-wgpu support");
    }

    let fifo_enabled = options.use_existing_matches
        && !options.sift_matching.use_gpu
        && colmap_fifo_verifier_enabled(options);

    let mut reports = Vec::new();
    let mut total_matches = 0usize;
    let mut completed = 0usize;
    let mut did_clear = false;
    let verifier_trace = if fifo_enabled {
        let fifo_pairs = input_pairs
            .iter()
            .map(|pair| match pair {
                MatchPairInput::Existing {
                    left,
                    right,
                    matches,
                } => (*left, *right, matches.clone()),
                MatchPairInput::Computed { .. } => unreachable!(),
            })
            .collect::<Vec<_>>();
        run_controlled_colmap_fifo_batches(
            fifo_pairs,
            &frames,
            &cameras,
            options,
            batch_size,
            pair_count,
            &db,
            &image_id_by_index,
            task,
            &mut reports,
            &mut total_matches,
            &mut completed,
            &mut did_clear,
        )?
    } else {
        for batch in input_pairs.chunks(batch_size) {
            task.checkpoint()?;
            let pair_reports = if options.use_existing_matches {
                existing_match_pair_reports_for_inputs(
                    batch,
                    &frames,
                    &cameras,
                    options,
                    #[cfg(feature = "gpu-wgpu")]
                    existing_backend.as_ref(),
                )?
            } else {
                computed_match_pair_reports_for_inputs(
                    batch,
                    &frames,
                    &cameras,
                    options,
                    #[cfg(feature = "gpu-wgpu")]
                    computed_backend.as_ref(),
                )?
            };
            let input_indices = batch
                .iter()
                .map(MatchPairInput::indices)
                .collect::<Vec<_>>();
            commit_and_emit_pair_batch(
                &db,
                &frames,
                &image_id_by_index,
                options,
                pair_reports,
                &input_indices,
                pair_count,
                task,
                &mut did_clear,
                &mut reports,
                &mut total_matches,
                &mut completed,
            )?;
        }
        None
    };
    if input_pairs.is_empty() {
        task.checkpoint()?;
        db.with_transaction(|| {
            if options.clear_existing && !options.use_existing_matches {
                db.clear_matches()?;
                db.clear_two_view_geometries()?;
            } else if options.use_existing_matches {
                db.clear_two_view_geometries()?;
            }
            Ok(())
        })?;
    }
    reports.sort_by(|left, right| {
        left.left_image
            .cmp(&right.left_image)
            .then_with(|| left.right_image.cmp(&right.right_image))
    });

    Ok(MatchFeaturesReport {
        database: database_path.to_path_buf(),
        backend: matching_backend_name(&options.sift_matching).to_string(),
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

pub(crate) fn match_explicit_image_pairs_to_database_with_task(
    database_path: &Path,
    image_pairs: &[(u32, u32)],
    options: &MatchFeaturesOptions,
    task: &mut SfmTaskContext<'_>,
) -> Result<MatchFeaturesReport> {
    options.sift_matching.check()?;
    if options.use_existing_matches {
        bail!("explicit image-pair matching requires computed matches");
    }
    task.checkpoint()?;
    let started = Instant::now();
    let db = ColmapDatabase::open(database_path)?;
    let mut images = db.read_all_images()?;
    images.sort_by(|left, right| left.name.cmp(&right.name));
    let (frames, cameras) = load_database_frames_and_cameras(&db, &images, true)?;
    let index_by_image_id = images
        .iter()
        .enumerate()
        .map(|(index, image)| (image.image_id, index))
        .collect::<HashMap<_, _>>();
    let image_id_by_index = images
        .iter()
        .enumerate()
        .map(|(index, image)| (index, image.image_id))
        .collect::<Vec<_>>();
    let mut seen = std::collections::BTreeSet::new();
    let mut inputs = Vec::with_capacity(image_pairs.len());
    for &(left_id, right_id) in image_pairs {
        if left_id == right_id {
            bail!("explicit image pair repeats image_id={left_id}");
        }
        let key = (left_id.min(right_id), left_id.max(right_id));
        if !seen.insert(key) {
            bail!("duplicate explicit image pair {}-{}", key.0, key.1);
        }
        let left = index_by_image_id
            .get(&left_id)
            .copied()
            .with_context(|| format!("explicit pair references missing image_id={left_id}"))?;
        let right = index_by_image_id
            .get(&right_id)
            .copied()
            .with_context(|| format!("explicit pair references missing image_id={right_id}"))?;
        inputs.push(MatchPairInput::Computed { left, right });
    }

    #[cfg(feature = "gpu-wgpu")]
    let computed_backend = if options.sift_matching.use_gpu {
        Some(ComputedGpuBackend::new()?)
    } else {
        None
    };
    #[cfg(not(feature = "gpu-wgpu"))]
    if options.sift_matching.use_gpu {
        bail!("RustSFM was built without gpu-wgpu support");
    }

    let pair_count = inputs.len();
    let mut reports = Vec::new();
    let mut total_matches = 0usize;
    let mut completed = 0usize;
    let mut did_clear = false;
    for batch in inputs.chunks(options.task_pair_batch_size.max(1)) {
        task.checkpoint()?;
        let pair_reports = computed_match_pair_reports_for_inputs(
            batch,
            &frames,
            &cameras,
            options,
            #[cfg(feature = "gpu-wgpu")]
            computed_backend.as_ref(),
        )?;
        let indices = batch
            .iter()
            .map(MatchPairInput::indices)
            .collect::<Vec<_>>();
        commit_and_emit_pair_batch(
            &db,
            &frames,
            &image_id_by_index,
            options,
            pair_reports,
            &indices,
            pair_count,
            task,
            &mut did_clear,
            &mut reports,
            &mut total_matches,
            &mut completed,
        )?;
    }
    reports.sort_by(|left, right| {
        left.left_image
            .cmp(&right.left_image)
            .then_with(|| left.right_image.cmp(&right.right_image))
    });
    Ok(MatchFeaturesReport {
        database: database_path.to_path_buf(),
        backend: matching_backend_name(&options.sift_matching).to_owned(),
        pair_count,
        matched_pairs: reports.len(),
        verified_pairs: reports
            .iter()
            .filter(|pair| pair.num_inliers >= options.min_inliers)
            .count(),
        total_matches,
        matching_seconds: started.elapsed().as_secs_f64(),
        verifier_trace: None,
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

#[derive(Debug, Clone)]
enum MatchPairInput {
    Computed {
        left: usize,
        right: usize,
    },
    Existing {
        left: usize,
        right: usize,
        matches: Vec<rustslam::Match>,
    },
}

impl MatchPairInput {
    fn indices(&self) -> (usize, usize) {
        match self {
            Self::Computed { left, right } | Self::Existing { left, right, .. } => (*left, *right),
        }
    }
}

fn computed_match_inputs(
    frame_count: usize,
    frames: &[ImageFrame],
    options: &MatchFeaturesOptions,
) -> Vec<(usize, usize)> {
    match options.pair_strategy {
        MatchingPairStrategy::VocabTree { num_images } => {
            vocab_tree_pairs_from_frames(frames, num_images, options.random_seed)
        }
        strategy => generate_matching_pairs(frame_count, strategy),
    }
}

fn existing_match_inputs(
    db: &ColmapDatabase,
    images: &[ColmapDatabaseImage],
) -> Result<Vec<MatchPairInput>> {
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
        pairs.push(MatchPairInput::Existing {
            left,
            right,
            matches,
        });
    }
    pairs.sort_by_key(|pair| {
        let (left, right) = pair.indices();
        (left.min(right), left.max(right), left, right)
    });
    Ok(pairs)
}

fn persist_pair_reports(
    db: &ColmapDatabase,
    frames: &[ImageFrame],
    image_id_by_index: &[(usize, u32)],
    options: &MatchFeaturesOptions,
    pair_reports: Vec<PairReportInput>,
) -> Result<(Vec<MatchFeaturesPairReport>, usize)> {
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
}

#[allow(clippy::too_many_arguments)]
fn commit_and_emit_pair_batch(
    db: &ColmapDatabase,
    frames: &[ImageFrame],
    image_id_by_index: &[(usize, u32)],
    options: &MatchFeaturesOptions,
    pair_reports: Vec<PairReportInput>,
    input_indices: &[(usize, usize)],
    pair_count: usize,
    task: &mut SfmTaskContext<'_>,
    did_clear: &mut bool,
    reports: &mut Vec<MatchFeaturesPairReport>,
    total_matches: &mut usize,
    completed: &mut usize,
) -> Result<()> {
    let (batch_reports, batch_matches) = db.with_transaction(|| {
        if !*did_clear {
            if options.clear_existing && !options.use_existing_matches {
                db.clear_matches()?;
                db.clear_two_view_geometries()?;
            } else if options.use_existing_matches {
                db.clear_two_view_geometries()?;
            }
        }
        persist_pair_reports(db, frames, image_id_by_index, options, pair_reports)
    })?;
    *did_clear = true;
    *total_matches += batch_matches;
    reports.extend(batch_reports);
    *completed += input_indices.len();
    let last_pair = input_indices
        .last()
        .map(|&(left, right)| (image_id_by_index[left].1, image_id_by_index[right].1));
    task.emit(SfmTaskEvent {
        sequence: 0,
        elapsed_ms: 0,
        stage: SfmTaskStage::FeatureMatching,
        operation: SfmTaskOperation::MatchPairBatch,
        kind: SfmTaskEventKind::Progress,
        completed: Some(*completed),
        total: Some(pair_count),
        registered_images: None,
        sparse_points: None,
        image_id: None,
        pair: last_pair,
        message: None,
        issue: None,
    });
    task.checkpoint()?;
    Ok(())
}

#[cfg(feature = "gpu-wgpu")]
struct ComputedGpuBackend {
    matcher: WgpuSiftMatcher,
    scorer: WgpuModelScorer,
}

#[cfg(feature = "gpu-wgpu")]
impl ComputedGpuBackend {
    fn new() -> Result<Self> {
        let context = WgpuContext::try_new()?;
        Ok(Self {
            matcher: WgpuSiftMatcher::from_context(context.clone())?,
            scorer: WgpuModelScorer::from_context(context)?,
        })
    }
}

#[cfg(feature = "gpu-wgpu")]
struct ExistingGpuBackend {
    scorer: WgpuModelScorer,
}

#[cfg(feature = "gpu-wgpu")]
impl ExistingGpuBackend {
    fn new() -> Result<Self> {
        Ok(Self {
            scorer: WgpuModelScorer::try_new()?,
        })
    }
}

fn computed_match_pair_reports_for_inputs(
    batch: &[MatchPairInput],
    frames: &[ImageFrame],
    cameras: &[CameraModel],
    options: &MatchFeaturesOptions,
    #[cfg(feature = "gpu-wgpu")] gpu_backend: Option<&ComputedGpuBackend>,
) -> Result<Vec<PairReportInput>> {
    let pairs = batch.iter().map(|pair| pair.indices()).collect::<Vec<_>>();
    #[cfg(feature = "gpu-wgpu")]
    if let Some(backend) = gpu_backend {
        let mut reports = Vec::with_capacity(pairs.len());
        for &(left, right) in &pairs {
            let matches = backend.matcher.match_descriptors(
                &frames[left].sift.descriptors_u8,
                &frames[right].sift.descriptors_u8,
                &options.sift_matching,
            )?;
            if let Some(report) = estimate_existing_or_computed_pair_gpu(
                &backend.scorer,
                left,
                right,
                matches,
                frames,
                cameras,
                options,
            )? {
                reports.push(report);
            }
        }
        return Ok(reports);
    }
    Ok(pairs
        .par_iter()
        .filter_map(|&(left, right)| {
            let matches = match_sift_with_options(
                &frames[left].sift,
                &frames[right].sift,
                &options.sift_matching,
            );
            estimate_existing_or_computed_pair(left, right, matches, frames, cameras, options)
        })
        .collect())
}

fn existing_match_pair_reports_for_inputs(
    batch: &[MatchPairInput],
    frames: &[ImageFrame],
    cameras: &[CameraModel],
    options: &MatchFeaturesOptions,
    #[cfg(feature = "gpu-wgpu")] gpu_backend: Option<&ExistingGpuBackend>,
) -> Result<Vec<PairReportInput>> {
    #[cfg(feature = "gpu-wgpu")]
    if let Some(backend) = gpu_backend {
        let mut reports = Vec::with_capacity(batch.len());
        for pair in batch {
            let MatchPairInput::Existing {
                left,
                right,
                matches,
            } = pair
            else {
                unreachable!()
            };
            if let Some(report) = estimate_existing_or_computed_pair_gpu(
                &backend.scorer,
                *left,
                *right,
                matches.clone(),
                frames,
                cameras,
                options,
            )? {
                reports.push(report);
            }
        }
        return Ok(reports);
    }
    #[cfg(not(feature = "gpu-wgpu"))]
    if options.sift_matching.use_gpu {
        bail!("RustSFM was built without gpu-wgpu support");
    }
    Ok(batch
        .par_iter()
        .filter_map(|pair| {
            let MatchPairInput::Existing {
                left,
                right,
                matches,
            } = pair
            else {
                unreachable!()
            };
            estimate_existing_or_computed_pair(
                *left,
                *right,
                matches.clone(),
                frames,
                cameras,
                options,
            )
        })
        .collect())
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

#[allow(clippy::too_many_arguments)]
fn commit_ready_fifo_prefixes(
    input_indices: &[(usize, usize)],
    task_batch_size: usize,
    pending_reports: &mut HashMap<(usize, usize), PairReportInput>,
    next_commit_start: &mut usize,
    pair_count: usize,
    db: &ColmapDatabase,
    frames: &[ImageFrame],
    image_id_by_index: &[(usize, u32)],
    options: &MatchFeaturesOptions,
    task: &mut SfmTaskContext<'_>,
    did_clear: &mut bool,
    reports: &mut Vec<MatchFeaturesPairReport>,
    total_matches: &mut usize,
    completed: &mut usize,
) -> Result<()> {
    while *next_commit_start < input_indices.len() {
        let end = (*next_commit_start + task_batch_size).min(input_indices.len());
        let batch = &input_indices[*next_commit_start..end];
        if !batch
            .iter()
            .all(|indices| pending_reports.contains_key(indices))
        {
            break;
        }
        task.checkpoint()?;
        let pair_reports = batch
            .iter()
            .map(|indices| {
                pending_reports
                    .remove(indices)
                    .expect("ready FIFO prefix contains every report")
            })
            .collect::<Vec<_>>();
        commit_and_emit_pair_batch(
            db,
            frames,
            image_id_by_index,
            options,
            pair_reports,
            batch,
            pair_count,
            task,
            did_clear,
            reports,
            total_matches,
            completed,
        )?;
        *next_commit_start = end;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_controlled_colmap_fifo_batches(
    pairs: Vec<ExistingMatchPairInput>,
    frames: &[ImageFrame],
    cameras: &[CameraModel],
    options: &MatchFeaturesOptions,
    task_batch_size: usize,
    pair_count: usize,
    db: &ColmapDatabase,
    image_id_by_index: &[(usize, u32)],
    task: &mut SfmTaskContext<'_>,
    reports: &mut Vec<MatchFeaturesPairReport>,
    total_matches: &mut usize,
    completed: &mut usize,
    did_clear: &mut bool,
) -> Result<Option<MatchFeaturesVerifierTrace>> {
    if let Some(trace_path) = colmap_fifo_replay_trace_path() {
        return run_controlled_colmap_replay_batches(
            pairs,
            frames,
            cameras,
            options,
            task_batch_size,
            pair_count,
            db,
            image_id_by_index,
            task,
            reports,
            total_matches,
            completed,
            did_clear,
            &trace_path,
        );
    }

    task.checkpoint()?;
    let trace_enabled = std::env::var_os("RUSTSFM_COLMAP_FIFO_VERIFIER_TRACE").is_some();
    if pairs.is_empty() {
        return Ok(trace_enabled.then(|| MatchFeaturesVerifierTrace {
            mode: "colmap_fifo_shared_ransac_stream".to_string(),
            worker_count: 0,
            events: Vec::new(),
        }));
    }

    let input_indices = pairs
        .iter()
        .map(|pair| (pair.0, pair.1))
        .collect::<Vec<_>>();
    let input_queue = Arc::new(ColmapFifoVerifierQueue::new());
    let output_queue = Arc::new(ColmapFifoVerifierOutputQueue::new());
    let worker_count = colmap_fifo_verifier_threads();
    let mut pending_reports = HashMap::new();
    let mut next_commit_start = 0usize;
    let mut events = Vec::new();

    thread::scope(|scope| -> Result<()> {
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

        let (result_sender, result_receiver) = std::sync::mpsc::channel();
        let scheduler_input_queue = Arc::clone(&input_queue);
        let scheduler_output_queue = Arc::clone(&output_queue);
        scope.spawn(move || {
            'operations: for operation in colmap_fifo_dispatch_operations(
                pairs.len(),
                options.existing_match_batch_size,
                options.task_pair_batch_size,
            ) {
                match operation {
                    ColmapFifoDispatchOperation::Enqueue { start, end } => {
                        for pair in pairs[start..end].iter().cloned() {
                            scheduler_input_queue.push(pair);
                        }
                    }
                    ColmapFifoDispatchOperation::Drain { count } => {
                        for _ in 0..count {
                            let Some(result) = scheduler_output_queue.pop() else {
                                break 'operations;
                            };
                            if result_sender.send(result).is_err() {
                                break 'operations;
                            }
                        }
                    }
                }
            }
            scheduler_input_queue.stop();
        });

        let result = (|| -> Result<()> {
            for _ in 0..input_indices.len() {
                let mut worker_result = result_receiver
                    .recv()
                    .context("COLMAP FIFO verifier scheduler stopped unexpectedly")?;
                let report = worker_result
                    .report
                    .take()
                    .context("COLMAP FIFO verifier omitted an existing-match report")?;
                let key = (report.0, report.1);
                if pending_reports.insert(key, report).is_some() {
                    bail!(
                        "COLMAP FIFO verifier returned duplicate pair {}-{}",
                        key.0,
                        key.1
                    );
                }
                if let Some(event) = worker_result.event.take() {
                    events.push(event);
                }
                commit_ready_fifo_prefixes(
                    &input_indices,
                    task_batch_size,
                    &mut pending_reports,
                    &mut next_commit_start,
                    pair_count,
                    db,
                    frames,
                    image_id_by_index,
                    options,
                    task,
                    did_clear,
                    reports,
                    total_matches,
                    completed,
                )?;
            }
            if next_commit_start != input_indices.len() {
                bail!(
                    "COLMAP FIFO verifier completed {} pairs but committed only {}",
                    input_indices.len(),
                    next_commit_start
                );
            }
            Ok(())
        })();
        if result.is_err() {
            input_queue.cancel();
            output_queue.stop();
        }
        result
    })?;

    Ok(trace_enabled.then(|| MatchFeaturesVerifierTrace {
        mode: "colmap_fifo_shared_ransac_stream".to_string(),
        worker_count,
        events,
    }))
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColmapFifoDispatchOperation {
    Enqueue { start: usize, end: usize },
    Drain { count: usize },
}

fn colmap_fifo_dispatch_operations(
    pair_count: usize,
    existing_match_batch_size: usize,
    _task_pair_batch_size: usize,
) -> Vec<ColmapFifoDispatchOperation> {
    let window_size = existing_match_batch_size.max(2);
    let mut operations = Vec::new();
    for start in (0..pair_count).step_by(window_size) {
        let end = (start + window_size).min(pair_count);
        operations.push(ColmapFifoDispatchOperation::Enqueue { start, end });
        operations.push(ColmapFifoDispatchOperation::Drain { count: end - start });
    }
    operations
}

#[cfg(test)]
fn colmap_fifo_dispatch_windows(
    pair_count: usize,
    existing_match_batch_size: usize,
) -> Vec<(usize, usize)> {
    colmap_fifo_dispatch_operations(pair_count, existing_match_batch_size, usize::MAX)
        .into_iter()
        .filter_map(|operation| match operation {
            ColmapFifoDispatchOperation::Enqueue { start, end } => Some((start, end)),
            ColmapFifoDispatchOperation::Drain { .. } => None,
        })
        .collect()
}

type ColmapReplayAssignment = (usize, usize, usize, usize, usize);

fn colmap_fifo_replay_dispatch_batches(
    pair_count: usize,
    _task_pair_batch_size: usize,
) -> Vec<(usize, usize)> {
    vec![(0, pair_count)]
}

fn colmap_fifo_replay_assignments(
    pairs: &[ExistingMatchPairInput],
    schedule: &ColmapVerifierReplaySchedule,
) -> Result<Vec<Vec<ColmapReplayAssignment>>> {
    if schedule.events.len() != pairs.len() {
        bail!(
            "COLMAP verifier replay trace has {} events, but database has {} existing-match pairs",
            schedule.events.len(),
            pairs.len()
        );
    }
    let pair_keys = pairs
        .iter()
        .map(|pair| (pair.0, pair.1))
        .collect::<std::collections::HashSet<_>>();
    if pair_keys.len() != pairs.len() {
        bail!("duplicate existing-match pair in replay input");
    }
    let mut seen = std::collections::HashSet::new();
    let mut events = schedule.events.clone();
    events.sort_by_key(|event| event.dequeue_order);
    let mut assignments = (0..schedule.worker_count)
        .map(|_| Vec::<ColmapReplayAssignment>::new())
        .collect::<Vec<_>>();
    for event in events {
        let key = (event.left_index, event.right_index);
        if !pair_keys.contains(&key) || !seen.insert(key) {
            bail!(
                "COLMAP verifier replay event does not map one-to-one to input pair {}-{}",
                key.0,
                key.1
            );
        }
        assignments[event.worker_id].push((
            event.worker_id,
            event.dequeue_order,
            event.complete_order,
            event.left_index,
            event.right_index,
        ));
    }
    if seen.len() != pairs.len() {
        bail!("COLMAP verifier replay trace is missing input pairs");
    }
    Ok(assignments)
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

#[allow(clippy::too_many_arguments)]
fn run_controlled_colmap_replay_batches(
    pairs: Vec<ExistingMatchPairInput>,
    frames: &[ImageFrame],
    cameras: &[CameraModel],
    options: &MatchFeaturesOptions,
    task_batch_size: usize,
    pair_count: usize,
    db: &ColmapDatabase,
    image_id_by_index: &[(usize, u32)],
    task: &mut SfmTaskContext<'_>,
    reports: &mut Vec<MatchFeaturesPairReport>,
    total_matches: &mut usize,
    completed: &mut usize,
    did_clear: &mut bool,
    trace_path: &Path,
) -> Result<Option<MatchFeaturesVerifierTrace>> {
    let schedule = load_colmap_fifo_replay_schedule(trace_path)?;
    if schedule.events.len() != pairs.len() {
        bail!(
            "COLMAP verifier replay trace has {} events, but database has {} existing-match pairs",
            schedule.events.len(),
            pairs.len()
        );
    }

    task.checkpoint()?;
    let input_indices = pairs
        .iter()
        .map(|pair| (pair.0, pair.1))
        .collect::<Vec<_>>();
    let input_pairs = pairs.clone();
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

    let worker_count = schedule.worker_count;
    let assignments = colmap_fifo_replay_assignments(&input_pairs, &schedule)?;
    let mut events_by_pair = schedule
        .events
        .into_iter()
        .map(|event| ((event.left_index, event.right_index), event))
        .collect::<HashMap<_, _>>();
    let mut worker_jobs = (0..worker_count)
        .map(|_| Vec::<ColmapReplayWorkerJob>::new())
        .collect::<Vec<_>>();
    for (batch_start, batch_end) in
        colmap_fifo_replay_dispatch_batches(input_indices.len(), task_batch_size)
    {
        for worker_assignments in &assignments {
            for (_, dequeue_order, complete_order, left, right) in worker_assignments {
                if *dequeue_order < batch_start || *dequeue_order >= batch_end {
                    continue;
                }
                let key = (*left, *right);
                let mut event = events_by_pair.remove(&key).with_context(|| {
                    let names = events_by_pair
                        .get(&key)
                        .map(|event| (event.left_image.as_str(), event.right_image.as_str()));
                    format!(
                        "COLMAP verifier replay event references missing pair {}-{} ({:?})",
                        left, right, names
                    )
                })?;
                event.dequeue_order = *dequeue_order;
                event.complete_order = *complete_order;
                let pair = pairs_by_index.remove(&key).with_context(|| {
                    format!(
                        "COLMAP verifier replay event references missing pair {}-{}",
                        left, right
                    )
                })?;
                worker_jobs[event.worker_id].push(ColmapReplayWorkerJob { event, pair });
            }
        }
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

    let stop = Arc::new(AtomicBool::new(false));
    let mut pending_reports = HashMap::new();
    let mut next_commit_start = 0usize;
    let mut trace_events = Vec::new();
    thread::scope(|scope| -> Result<()> {
        let (result_sender, result_receiver) = std::sync::mpsc::channel();
        for (worker_id, jobs) in worker_jobs.into_iter().enumerate() {
            let result_sender = result_sender.clone();
            let stop = Arc::clone(&stop);
            scope.spawn(move || {
                for job in jobs {
                    if stop.load(Ordering::SeqCst) {
                        return;
                    }
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
                    if result_sender
                        .send(ColmapFifoWorkerResult { report, event })
                        .is_err()
                    {
                        return;
                    }
                }
            });
        }
        drop(result_sender);

        let result = (|| -> Result<()> {
            for _ in 0..input_indices.len() {
                let mut worker_result = result_receiver
                    .recv()
                    .context("COLMAP replay verifier workers stopped unexpectedly")?;
                let report = worker_result
                    .report
                    .take()
                    .context("COLMAP replay verifier omitted an existing-match report")?;
                let key = (report.0, report.1);
                if pending_reports.insert(key, report).is_some() {
                    bail!(
                        "COLMAP replay verifier returned duplicate pair {}-{}",
                        key.0,
                        key.1
                    );
                }
                if let Some(event) = worker_result.event.take() {
                    trace_events.push(event);
                }
                commit_ready_fifo_prefixes(
                    &input_indices,
                    task_batch_size,
                    &mut pending_reports,
                    &mut next_commit_start,
                    pair_count,
                    db,
                    frames,
                    image_id_by_index,
                    options,
                    task,
                    did_clear,
                    reports,
                    total_matches,
                    completed,
                )?;
            }
            if next_commit_start != input_indices.len() {
                bail!(
                    "COLMAP replay verifier completed {} pairs but committed only {}",
                    input_indices.len(),
                    next_commit_start
                );
            }
            Ok(())
        })();
        if result.is_err() {
            stop.store(true, Ordering::SeqCst);
        }
        result
    })?;
    trace_events.sort_by_key(|event| event.complete_order);

    Ok(Some(MatchFeaturesVerifierTrace {
        mode: "colmap_fifo_shared_ransac_stream_replay".to_string(),
        worker_count,
        events: trace_events,
    }))
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
    stopped: bool,
    next_completion_order: usize,
    jobs: VecDeque<ColmapFifoWorkerResult>,
}

impl ColmapFifoVerifierOutputQueue {
    fn new() -> Self {
        Self {
            state: Mutex::new(ColmapFifoVerifierOutputQueueState {
                stopped: false,
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
            if state.stopped {
                return;
            }
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
            if state.stopped {
                return None;
            }
            state = self.condvar.wait(state).ok()?;
        }
    }

    fn stop(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.stopped = true;
            state.jobs.clear();
        }
        self.condvar.notify_all();
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

    fn cancel(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.stopped = true;
            state.jobs.clear();
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

#[cfg(feature = "gpu-wgpu")]
fn estimate_existing_or_computed_pair_gpu(
    scorer: &WgpuModelScorer,
    left: usize,
    right: usize,
    matches: Vec<rustslam::Match>,
    frames: &[ImageFrame],
    cameras: &[CameraModel],
    options: &MatchFeaturesOptions,
) -> Result<Option<PairReportInput>> {
    let min_matches_for_estimation = if options.use_existing_matches {
        options.min_inliers
    } else {
        options.min_num_matches
    };
    if matches.len() < min_matches_for_estimation {
        return Ok(if options.use_existing_matches {
            Some((left, right, matches, None))
        } else {
            None
        });
    }
    let geometry = estimate_pair_geometry_with_options_and_cameras_gpu(
        scorer,
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
    )?;
    Ok(Some((left, right, matches, geometry)))
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
    use std::sync::Mutex as StdMutex;
    use tempfile::tempdir;

    static MATCHING_ENV_LOCK: StdMutex<()> = StdMutex::new(());

    #[test]
    fn match_features_task_batch_default_is_32() {
        assert_eq!(MatchFeaturesOptions::default().task_pair_batch_size, 32);
    }

    #[test]
    fn fifo_dispatch_windows_are_independent_of_task_commit_batch_size() {
        let expected = vec![
            ColmapFifoDispatchOperation::Enqueue { start: 0, end: 7 },
            ColmapFifoDispatchOperation::Drain { count: 7 },
            ColmapFifoDispatchOperation::Enqueue { start: 7, end: 14 },
            ColmapFifoDispatchOperation::Drain { count: 7 },
            ColmapFifoDispatchOperation::Enqueue { start: 14, end: 21 },
            ColmapFifoDispatchOperation::Drain { count: 7 },
            ColmapFifoDispatchOperation::Enqueue { start: 21, end: 24 },
            ColmapFifoDispatchOperation::Drain { count: 3 },
        ];
        for task_pair_batch_size in [1, 2, 32] {
            assert_eq!(
                colmap_fifo_dispatch_operations(24, 7, task_pair_batch_size),
                expected,
                "task batch size {task_pair_batch_size} changed legacy FIFO dispatch"
            );
        }
        assert_eq!(
            colmap_fifo_dispatch_windows(24, 7),
            vec![(0, 7), (7, 14), (14, 21), (21, 24)]
        );
    }

    #[test]
    fn fifo_replay_assignments_preserve_every_trace_event() -> Result<()> {
        let schedule = ColmapVerifierReplaySchedule {
            worker_count: 2,
            events: vec![
                replay_test_event(1, 4, 2, 0, 1),
                replay_test_event(0, 0, 3, 1, 2),
                replay_test_event(1, 2, 1, 0, 2),
                replay_test_event(0, 3, 0, 0, 3),
            ],
        };
        let pairs = vec![
            (0, 3, Vec::new()),
            (0, 1, Vec::new()),
            (0, 2, Vec::new()),
            (1, 2, Vec::new()),
        ];

        let assignments = colmap_fifo_replay_assignments(&pairs, &schedule)?;

        assert_eq!(
            assignments,
            vec![
                vec![(0, 0, 3, 1, 2), (0, 3, 0, 0, 3)],
                vec![(1, 2, 1, 0, 2), (1, 4, 2, 0, 1)],
            ]
        );
        Ok(())
    }

    #[test]
    fn fifo_replay_dispatch_is_global_across_task_commit_batches() {
        for task_pair_batch_size in [1, 2, 32] {
            assert_eq!(
                colmap_fifo_replay_dispatch_batches(24, task_pair_batch_size),
                vec![(0, 24)],
                "task commit batch {task_pair_batch_size} split replay dispatch"
            );
        }
    }

    #[test]
    fn existing_match_inputs_are_sorted_by_normalized_frame_indices() -> Result<()> {
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
        for (image_id, name) in [(1, "z.jpg"), (2, "a.jpg"), (3, "y.jpg"), (4, "b.jpg")] {
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
        for (left, right) in [(1, 2), (1, 3), (2, 4), (3, 4)] {
            db.write_matches(left, right, &[FeatureMatch::new(0, 0)])?;
        }
        let mut images = db.read_all_images()?;
        images.sort_by(|left, right| left.name.cmp(&right.name));

        let pairs = existing_match_inputs(&db, &images)?;

        assert_eq!(
            pairs
                .iter()
                .map(MatchPairInput::indices)
                .collect::<Vec<_>>(),
            vec![(0, 1), (3, 0), (2, 1), (3, 2)]
        );
        Ok(())
    }

    #[test]
    fn controlled_matching_honors_pre_requested_cancellation() -> Result<()> {
        use crate::task::{SfmTaskContext, SfmTaskControl, SfmTaskEvent, SfmTaskStop};

        let (_dir, db_path, _) = controlled_matching_fixture()?;
        let original_geometry = ColmapTwoViewGeometry {
            config: 5,
            ..ColmapTwoViewGeometry::default()
        };
        let db = ColmapDatabase::open(&db_path)?;
        db.write_two_view_geometry(1, 2, &original_geometry)?;
        drop(db);
        let control = SfmTaskControl::new();
        control.request_cancel();
        let mut events = Vec::<SfmTaskEvent>::new();
        let mut sink = |event: SfmTaskEvent| events.push(event);
        let mut task = SfmTaskContext::new(&control, &mut sink);
        let error = match_features_to_database_with_task(
            &db_path,
            &controlled_matching_options(2),
            &mut task,
        )
        .expect_err("pre-requested cancellation");
        assert_eq!(
            error.downcast_ref::<SfmTaskStop>(),
            Some(&SfmTaskStop::Cancelled)
        );
        assert!(events.is_empty());
        let db = ColmapDatabase::open_read_only(&db_path)?;
        assert_eq!(db.read_two_view_geometry(1, 2)?, original_geometry);
        assert_eq!(db.read_num_matches()?.len(), 5);
        Ok(())
    }

    #[test]
    fn controlled_matching_reports_bounded_pair_progress() -> Result<()> {
        use crate::task::{
            SfmTaskContext, SfmTaskControl, SfmTaskEvent, SfmTaskEventKind, SfmTaskOperation,
            SfmTaskStage,
        };

        let (_dir, db_path, input_pairs) = controlled_matching_fixture()?;
        let control = SfmTaskControl::new();
        let mut events = Vec::<SfmTaskEvent>::new();
        let mut sink = |event: SfmTaskEvent| events.push(event);
        let mut task = SfmTaskContext::new(&control, &mut sink);
        let report = match_features_to_database_with_task(
            &db_path,
            &controlled_matching_options(2),
            &mut task,
        )?;

        assert_eq!(report.pair_count, 5);
        assert_eq!(report.matched_pairs, 5);
        assert_eq!(
            events
                .iter()
                .map(|event| event.completed)
                .collect::<Vec<_>>(),
            vec![Some(2), Some(4), Some(5)]
        );
        assert!(events.iter().all(|event| event.total == Some(5)));
        assert_eq!(
            events
                .iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert!(events.iter().all(|event| {
            event.stage == SfmTaskStage::FeatureMatching
                && event.operation == SfmTaskOperation::MatchPairBatch
                && event.kind == SfmTaskEventKind::Progress
        }));
        assert_eq!(
            events.iter().map(|event| event.pair).collect::<Vec<_>>(),
            vec![
                Some(input_pairs[1]),
                Some(input_pairs[3]),
                Some(input_pairs[4])
            ]
        );
        Ok(())
    }

    #[test]
    fn controlled_computed_matching_cancel_keeps_exactly_first_batch() -> Result<()> {
        use crate::task::{SfmTaskContext, SfmTaskControl, SfmTaskEvent, SfmTaskStop};

        let (_dir, db_path, input_pairs) = controlled_computed_matching_fixture()?;
        let control = SfmTaskControl::new();
        let sink_control = control.clone();
        let mut events = Vec::<SfmTaskEvent>::new();
        let mut sink = |event: SfmTaskEvent| {
            if event.completed == Some(2) {
                sink_control.request_cancel();
            }
            events.push(event);
        };
        let mut task = SfmTaskContext::new(&control, &mut sink);
        let error = match_features_to_database_with_task(
            &db_path,
            &controlled_computed_matching_options(2),
            &mut task,
        )
        .expect_err("cancel requested after first computed pair batch");
        assert_eq!(
            error.downcast_ref::<SfmTaskStop>(),
            Some(&SfmTaskStop::Cancelled)
        );
        assert_eq!(events.len(), 1);

        let db = ColmapDatabase::open_read_only(&db_path)?;
        for (index, &(left, right)) in input_pairs.iter().enumerate() {
            assert_eq!(
                db.exists_matches(left, right)?,
                index < 2,
                "unexpected computed match row for pair {left}-{right}"
            );
        }
        Ok(())
    }

    #[test]
    fn controlled_computed_matching_uses_bounded_progress_batches() -> Result<()> {
        use crate::task::{SfmTaskContext, SfmTaskControl, SfmTaskEvent};

        let (_dir, db_path, input_pairs) = controlled_computed_matching_fixture()?;
        let control = SfmTaskControl::new();
        let mut events = Vec::<SfmTaskEvent>::new();
        let mut sink = |event: SfmTaskEvent| events.push(event);
        let mut task = SfmTaskContext::new(&control, &mut sink);
        let report = match_features_to_database_with_task(
            &db_path,
            &controlled_computed_matching_options(2),
            &mut task,
        )?;

        assert_eq!(report.pair_count, input_pairs.len());
        assert_eq!(report.matched_pairs, input_pairs.len());
        assert_eq!(
            events
                .iter()
                .map(|event| event.completed)
                .collect::<Vec<_>>(),
            vec![Some(2), Some(4), Some(6)]
        );
        assert_eq!(
            events.iter().map(|event| event.pair).collect::<Vec<_>>(),
            vec![
                Some(input_pairs[1]),
                Some(input_pairs[3]),
                Some(input_pairs[5])
            ]
        );
        Ok(())
    }

    #[test]
    fn controlled_explicit_matching_commits_only_requested_database_pairs() -> Result<()> {
        use crate::task::{SfmTaskContext, SfmTaskControl, SfmTaskEvent};

        let (_dir, db_path, input_pairs) = controlled_computed_matching_fixture()?;
        let requested = vec![input_pairs[1], input_pairs[4]];
        let control = SfmTaskControl::new();
        let mut events = Vec::<SfmTaskEvent>::new();
        let mut sink = |event: SfmTaskEvent| events.push(event);
        let mut task = SfmTaskContext::new(&control, &mut sink);
        let report = match_explicit_image_pairs_to_database_with_task(
            &db_path,
            &requested,
            &controlled_computed_matching_options(1),
            &mut task,
        )?;

        assert_eq!(report.pair_count, requested.len());
        assert_eq!(events.len(), requested.len());
        let db = ColmapDatabase::open_read_only(&db_path)?;
        for pair in input_pairs {
            assert_eq!(
                db.exists_matches(pair.0, pair.1)?,
                requested.contains(&pair)
            );
        }
        Ok(())
    }

    #[test]
    fn controlled_fifo_trace_is_independent_of_task_commit_batch_size() -> Result<()> {
        let _env_guard = MATCHING_ENV_LOCK.lock().expect("matching env lock");
        let (_dir, db_path, input_pairs) = controlled_fifo_geometry_fixture()?;
        let _env = MatchingEnvGuard::fifo_trace(2);

        let expected_dispatch = colmap_fifo_dispatch_operations(input_pairs.len(), 1000, 2);
        for task_pair_batch_size in [1, 2, 32] {
            assert_eq!(
                colmap_fifo_dispatch_operations(input_pairs.len(), 1000, task_pair_batch_size),
                expected_dispatch,
                "task commit batch {task_pair_batch_size} changed live scheduler operations"
            );
        }
        let report = match_features_to_database(&db_path, &controlled_fifo_geometry_options(2))?;
        let trace = report.verifier_trace.as_ref().expect("live FIFO trace");
        assert_eq!(trace.worker_count, 2);
        assert_eq!(trace.events.len(), input_pairs.len());
        assert!(trace.events.iter().all(|event| event.num_matches == 24));
        assert!(trace.events.iter().any(|event| event.num_inliers >= 15));
        for &(left, right) in &input_pairs {
            assert!(
                ColmapDatabase::open_read_only(&db_path)?.exists_two_view_geometry(left, right)?
            );
        }
        Ok(())
    }

    #[test]
    fn controlled_fifo_replay_is_independent_of_task_commit_batch_size() -> Result<()> {
        let _env_guard = MATCHING_ENV_LOCK.lock().expect("matching env lock");
        let trace_dir = tempdir()?;
        let trace_path = trace_dir.path().join("trace.json");
        std::fs::write(
            &trace_path,
            r#"{
              "worker_count": 2,
              "events": [
                {"worker_id":0,"dequeue_order":0,"complete_order":3,"left_index":0,"right_index":1,"left_image":"image-1.jpg","right_image":"image-2.jpg"},
                {"worker_id":1,"dequeue_order":1,"complete_order":0,"left_index":0,"right_index":2,"left_image":"image-1.jpg","right_image":"image-3.jpg"},
                {"worker_id":0,"dequeue_order":2,"complete_order":4,"left_index":0,"right_index":3,"left_image":"image-1.jpg","right_image":"image-4.jpg"},
                {"worker_id":1,"dequeue_order":3,"complete_order":1,"left_index":1,"right_index":2,"left_image":"image-2.jpg","right_image":"image-3.jpg"},
                {"worker_id":0,"dequeue_order":4,"complete_order":5,"left_index":1,"right_index":3,"left_image":"image-2.jpg","right_image":"image-4.jpg"},
                {"worker_id":1,"dequeue_order":5,"complete_order":2,"left_index":2,"right_index":3,"left_image":"image-3.jpg","right_image":"image-4.jpg"}
              ]
            }"#,
        )?;
        let (_small_dir, small_db_path, input_pairs) = controlled_fifo_geometry_fixture()?;
        let (_large_dir, large_db_path, _) = controlled_fifo_geometry_fixture()?;
        let _env = MatchingEnvGuard::fifo_replay(&trace_path);

        let small_options = controlled_fifo_geometry_options(1);
        let mut large_options = small_options.clone();
        large_options.task_pair_batch_size = 32;
        let small = match_features_to_database(&small_db_path, &small_options)?;
        let large = match_features_to_database(&large_db_path, &large_options)?;

        assert_eq!(
            serde_json::to_value(&small.pairs)?,
            serde_json::to_value(&large.pairs)?
        );
        assert_eq!(
            serde_json::to_value(&small.verifier_trace)?,
            serde_json::to_value(&large.verifier_trace)?
        );
        let trace = small.verifier_trace.as_ref().expect("replay trace");
        assert_eq!(
            trace
                .events
                .iter()
                .map(|event| {
                    (
                        event.worker_id,
                        event.dequeue_order,
                        event.complete_order,
                        event.left_index,
                        event.right_index,
                    )
                })
                .collect::<Vec<_>>(),
            vec![
                (1, 1, 0, 0, 2),
                (1, 3, 1, 1, 2),
                (1, 5, 2, 2, 3),
                (0, 0, 3, 0, 1),
                (0, 2, 4, 0, 3),
                (0, 4, 5, 1, 3),
            ]
        );
        for &(left, right) in &input_pairs {
            let small_db = ColmapDatabase::open_read_only(&small_db_path)?;
            let large_db = ColmapDatabase::open_read_only(&large_db_path)?;
            assert_eq!(
                small_db.read_two_view_geometry(left, right)?,
                large_db.read_two_view_geometry(left, right)?
            );
        }
        Ok(())
    }

    #[test]
    fn controlled_live_fifo_cancel_commits_only_first_prefix() -> Result<()> {
        use crate::task::{SfmTaskContext, SfmTaskControl, SfmTaskEvent, SfmTaskStop};

        let _env_guard = MATCHING_ENV_LOCK.lock().expect("matching env lock");
        let (_dir, db_path, input_pairs) = controlled_fifo_geometry_fixture()?;
        let _env = MatchingEnvGuard::fifo_trace(2);
        let control = SfmTaskControl::new();
        let sink_control = control.clone();
        let mut events = Vec::<SfmTaskEvent>::new();
        let mut sink = |event: SfmTaskEvent| {
            if event.completed == Some(2) {
                sink_control.request_cancel();
            }
            events.push(event);
        };
        let mut task = SfmTaskContext::new(&control, &mut sink);
        let error = match_features_to_database_with_task(
            &db_path,
            &controlled_fifo_geometry_options(2),
            &mut task,
        )
        .expect_err("live FIFO cancellation after first committed prefix");
        assert_eq!(
            error.downcast_ref::<SfmTaskStop>(),
            Some(&SfmTaskStop::Cancelled)
        );
        assert_eq!(events.len(), 1);
        let db = ColmapDatabase::open_read_only(&db_path)?;
        for (index, &(left, right)) in input_pairs.iter().enumerate() {
            assert_eq!(
                db.exists_two_view_geometry(left, right)?,
                index < 2,
                "unexpected live FIFO geometry after cancellation for {left}-{right}"
            );
        }
        Ok(())
    }

    #[test]
    fn controlled_replay_fifo_cancel_commits_only_first_prefix() -> Result<()> {
        use crate::task::{SfmTaskContext, SfmTaskControl, SfmTaskEvent, SfmTaskStop};

        let _env_guard = MATCHING_ENV_LOCK.lock().expect("matching env lock");
        let trace_dir = tempdir()?;
        let trace_path = trace_dir.path().join("trace.json");
        write_fifo_replay_trace(&trace_path)?;
        let (_dir, db_path, input_pairs) = controlled_fifo_geometry_fixture()?;
        let _env = MatchingEnvGuard::fifo_replay(&trace_path);
        let control = SfmTaskControl::new();
        let sink_control = control.clone();
        let mut events = Vec::<SfmTaskEvent>::new();
        let mut sink = |event: SfmTaskEvent| {
            if event.completed == Some(2) {
                sink_control.request_cancel();
            }
            events.push(event);
        };
        let mut task = SfmTaskContext::new(&control, &mut sink);
        let error = match_features_to_database_with_task(
            &db_path,
            &controlled_fifo_geometry_options(2),
            &mut task,
        )
        .expect_err("replay FIFO cancellation after first committed prefix");
        assert_eq!(
            error.downcast_ref::<SfmTaskStop>(),
            Some(&SfmTaskStop::Cancelled)
        );
        assert_eq!(events.len(), 1);
        let db = ColmapDatabase::open_read_only(&db_path)?;
        for (index, &(left, right)) in input_pairs.iter().enumerate() {
            assert_eq!(
                db.exists_two_view_geometry(left, right)?,
                index < 2,
                "unexpected replay FIFO geometry after cancellation for {left}-{right}"
            );
        }
        Ok(())
    }

    #[test]
    fn controlled_matching_pause_keeps_only_committed_pair_prefix() -> Result<()> {
        use crate::task::{SfmTaskContext, SfmTaskControl, SfmTaskEvent, SfmTaskStop};

        let (_dir, db_path, input_pairs) = controlled_matching_fixture()?;
        let control = SfmTaskControl::new();
        let sink_control = control.clone();
        let mut events = Vec::<SfmTaskEvent>::new();
        let mut sink = |event: SfmTaskEvent| {
            if event.completed == Some(2) {
                sink_control.request_pause();
            }
            events.push(event);
        };
        let mut task = SfmTaskContext::new(&control, &mut sink);
        let error = match_features_to_database_with_task(
            &db_path,
            &controlled_matching_options(2),
            &mut task,
        )
        .expect_err("pause requested after first pair batch");
        assert_eq!(
            error.downcast_ref::<SfmTaskStop>(),
            Some(&SfmTaskStop::Paused)
        );
        assert_eq!(events.len(), 1);

        let db = ColmapDatabase::open_read_only(&db_path)?;
        for (index, &(left, right)) in input_pairs.iter().enumerate() {
            assert_eq!(
                db.exists_two_view_geometry(left, right)?,
                index < 2,
                "unexpected committed geometry for pair {left}-{right}"
            );
        }
        Ok(())
    }

    #[test]
    fn controlled_matching_rolls_back_failed_batch_and_keeps_prior_batch() -> Result<()> {
        use crate::correspondence_graph::image_pair_to_pair_id;
        use crate::task::{SfmTaskContext, SfmTaskControl, SfmTaskEvent};
        use rusqlite::Connection;

        let (_dir, db_path, input_pairs) = controlled_matching_fixture()?;
        let failed_pair = input_pairs[3];
        let failed_pair_id = image_pair_to_pair_id(failed_pair.0, failed_pair.1)
            .map_err(|error| anyhow::anyhow!("{error:?}"))?;
        let trigger_connection = Connection::open(&db_path)?;
        trigger_connection.execute_batch(&format!(
            "CREATE TRIGGER fail_second_geometry_in_batch
             BEFORE INSERT ON two_view_geometries
             WHEN NEW.pair_id = {failed_pair_id}
             BEGIN
                 SELECT RAISE(ABORT, 'second geometry write failed');
             END;"
        ))?;
        drop(trigger_connection);

        let control = SfmTaskControl::new();
        let mut sink = |_event: SfmTaskEvent| {};
        let mut task = SfmTaskContext::new(&control, &mut sink);
        let error = match_features_to_database_with_task(
            &db_path,
            &controlled_matching_options(2),
            &mut task,
        )
        .expect_err("second geometry write in the second batch must fail");
        assert!(error.to_string().contains("second geometry write failed"));

        let db = ColmapDatabase::open_read_only(&db_path)?;
        for (index, &(left, right)) in input_pairs.iter().enumerate() {
            assert_eq!(
                db.exists_two_view_geometry(left, right)?,
                index < 2,
                "unexpected geometry after failed batch for pair {left}-{right}"
            );
        }
        Ok(())
    }

    #[test]
    fn controlled_matching_zero_batch_size_behaves_as_one() -> Result<()> {
        use crate::task::{SfmTaskContext, SfmTaskControl, SfmTaskEvent};

        let (_dir, db_path, _) = controlled_matching_fixture()?;
        let control = SfmTaskControl::new();
        let mut events = Vec::<SfmTaskEvent>::new();
        let mut sink = |event: SfmTaskEvent| events.push(event);
        let mut task = SfmTaskContext::new(&control, &mut sink);
        match_features_to_database_with_task(&db_path, &controlled_matching_options(0), &mut task)?;

        assert_eq!(
            events
                .iter()
                .map(|event| event.completed)
                .collect::<Vec<_>>(),
            vec![Some(1), Some(2), Some(3), Some(4), Some(5)]
        );
        Ok(())
    }

    fn controlled_matching_options(task_pair_batch_size: usize) -> MatchFeaturesOptions {
        MatchFeaturesOptions {
            use_existing_matches: true,
            min_inliers: 2,
            min_num_matches: 2,
            random_seed: 0,
            task_pair_batch_size,
            ..MatchFeaturesOptions::default()
        }
    }

    fn replay_test_event(
        worker_id: usize,
        dequeue_order: usize,
        complete_order: usize,
        left_index: usize,
        right_index: usize,
    ) -> ColmapVerifierReplayEvent {
        ColmapVerifierReplayEvent {
            worker_id,
            dequeue_order,
            complete_order,
            left_index,
            right_index,
            left_image: format!("image-{left_index}.jpg"),
            right_image: format!("image-{right_index}.jpg"),
        }
    }

    fn write_fifo_replay_trace(path: &Path) -> Result<()> {
        std::fs::write(
            path,
            r#"{
              "worker_count": 2,
              "events": [
                {"worker_id":0,"dequeue_order":0,"complete_order":3,"left_index":0,"right_index":1,"left_image":"image-1.jpg","right_image":"image-2.jpg"},
                {"worker_id":1,"dequeue_order":1,"complete_order":0,"left_index":0,"right_index":2,"left_image":"image-1.jpg","right_image":"image-3.jpg"},
                {"worker_id":0,"dequeue_order":2,"complete_order":4,"left_index":0,"right_index":3,"left_image":"image-1.jpg","right_image":"image-4.jpg"},
                {"worker_id":1,"dequeue_order":3,"complete_order":1,"left_index":1,"right_index":2,"left_image":"image-2.jpg","right_image":"image-3.jpg"},
                {"worker_id":0,"dequeue_order":4,"complete_order":5,"left_index":1,"right_index":3,"left_image":"image-2.jpg","right_image":"image-4.jpg"},
                {"worker_id":1,"dequeue_order":5,"complete_order":2,"left_index":2,"right_index":3,"left_image":"image-3.jpg","right_image":"image-4.jpg"}
              ]
            }"#,
        )?;
        Ok(())
    }

    fn controlled_computed_matching_options(task_pair_batch_size: usize) -> MatchFeaturesOptions {
        MatchFeaturesOptions {
            pair_strategy: MatchingPairStrategy::Exhaustive,
            sift_matching: SiftMatchingOptions {
                cpu_brute_force_matcher: true,
                ..SiftMatchingOptions::default()
            },
            min_num_matches: 1,
            min_inliers: 8,
            random_seed: 0,
            task_pair_batch_size,
            ..MatchFeaturesOptions::default()
        }
    }

    fn controlled_fifo_geometry_options(task_pair_batch_size: usize) -> MatchFeaturesOptions {
        MatchFeaturesOptions {
            use_existing_matches: true,
            essential_threshold_px: 2.0,
            essential_iterations: 256,
            min_inliers: 15,
            min_triangulated: 8,
            min_num_matches: 15,
            random_seed: -1,
            task_pair_batch_size,
            ..MatchFeaturesOptions::default()
        }
    }

    fn controlled_matching_fixture() -> Result<(tempfile::TempDir, PathBuf, Vec<(u32, u32)>)> {
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
        for (image_id, name) in [(1, "a.jpg"), (2, "b.jpg"), (3, "c.jpg"), (4, "d.jpg")] {
            db.write_image(
                &ColmapDatabaseImage {
                    image_id,
                    name: name.to_string(),
                    camera_id: 1,
                    frame_id: None,
                },
                true,
            )?;
            db.write_keypoints(image_id, &[ColmapKeypoint::new(50.0, 50.0)])?;
        }
        let input_pairs = vec![(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)];
        for &(left, right) in &input_pairs {
            db.write_matches(left, right, &[FeatureMatch::new(0, 0)])?;
        }
        drop(db);
        Ok((dir, db_path, input_pairs))
    }

    fn controlled_computed_matching_fixture(
    ) -> Result<(tempfile::TempDir, PathBuf, Vec<(u32, u32)>)> {
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
        let mut descriptor_data = vec![0u8; 128];
        descriptor_data.extend([255u8; 128]);
        let descriptors = ColmapDescriptors::new(
            crate::database::COLMAP_FEATURE_SIFT,
            2,
            128,
            descriptor_data,
        )?;
        for (image_id, name) in [(1, "a.jpg"), (2, "b.jpg"), (3, "c.jpg"), (4, "d.jpg")] {
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
                    ColmapKeypoint::new(40.0, 40.0),
                    ColmapKeypoint::new(60.0, 60.0),
                ],
            )?;
            db.write_descriptors(image_id, &descriptors)?;
        }
        drop(db);
        Ok((
            dir,
            db_path,
            vec![(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
        ))
    }

    fn controlled_fifo_geometry_fixture() -> Result<(tempfile::TempDir, PathBuf, Vec<(u32, u32)>)> {
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
                    params: vec![80.0, 80.0, 50.0, 50.0],
                },
                has_prior_focal_length: true,
            },
            true,
        )?;
        for image_id in 1..=4 {
            db.write_image(
                &ColmapDatabaseImage {
                    image_id,
                    name: format!("image-{image_id}.jpg"),
                    camera_id: 1,
                    frame_id: None,
                },
                true,
            )?;
        }
        let transforms = [
            (
                nalgebra::Rotation3::identity().into_inner(),
                nalgebra::Vector3::new(0.0, 0.0, 0.0),
            ),
            (
                nalgebra::Rotation3::from_euler_angles(0.03, -0.04, 0.02).into_inner(),
                nalgebra::Vector3::new(0.2, -0.03, 0.05),
            ),
            (
                nalgebra::Rotation3::from_euler_angles(-0.02, 0.05, -0.03).into_inner(),
                nalgebra::Vector3::new(-0.16, 0.08, 0.03),
            ),
            (
                nalgebra::Rotation3::from_euler_angles(0.04, 0.01, 0.05).into_inner(),
                nalgebra::Vector3::new(0.12, 0.14, 0.08),
            ),
        ];
        let points = (0..24usize)
            .map(|index| {
                nalgebra::Vector3::new(
                    (index % 6) as f64 * 0.25 - 0.6,
                    (index / 6) as f64 * 0.22 - 0.35,
                    3.0 + (index % 5) as f64 * 0.35,
                )
            })
            .collect::<Vec<_>>();
        for (image_index, (rotation, translation)) in transforms.iter().enumerate() {
            let keypoints = points
                .iter()
                .map(|point| {
                    let transformed = rotation * point + translation;
                    ColmapKeypoint::new(
                        (80.0 * transformed.x / transformed.z + 50.0) as f32,
                        (80.0 * transformed.y / transformed.z + 50.0) as f32,
                    )
                })
                .collect::<Vec<_>>();
            db.write_keypoints((image_index + 1) as u32, &keypoints)?;
        }
        let pair_matches = (0..24usize)
            .map(|index| FeatureMatch::new(index as u32, index as u32))
            .collect::<Vec<_>>();
        let input_pairs = (1..=4)
            .flat_map(|left| ((left + 1)..=4).map(move |right| (left, right)))
            .collect::<Vec<_>>();
        for &(left, right) in &input_pairs {
            db.write_matches(left, right, &pair_matches)?;
        }
        drop(db);
        Ok((dir, db_path, input_pairs))
    }

    struct MatchingEnvGuard;

    impl MatchingEnvGuard {
        fn fifo_trace(worker_count: usize) -> Self {
            std::env::set_var("RUSTSFM_COLMAP_FIFO_VERIFIER", "1");
            std::env::set_var("RUSTSFM_COLMAP_SHARED_RANSAC_STREAM", "1");
            std::env::set_var("RUSTSFM_COLMAP_FIFO_VERIFIER_TRACE", "1");
            std::env::set_var(
                "RUSTSFM_COLMAP_FIFO_VERIFIER_THREADS",
                worker_count.to_string(),
            );
            std::env::remove_var("RUSTSFM_COLMAP_FIFO_VERIFIER_REPLAY_TRACE");
            Self
        }

        fn fifo_replay(trace_path: &Path) -> Self {
            let guard = Self::fifo_trace(2);
            std::env::set_var("RUSTSFM_COLMAP_FIFO_VERIFIER_REPLAY_TRACE", trace_path);
            guard
        }
    }

    impl Drop for MatchingEnvGuard {
        fn drop(&mut self) {
            for name in [
                "RUSTSFM_COLMAP_FIFO_VERIFIER",
                "RUSTSFM_COLMAP_SHARED_RANSAC_STREAM",
                "RUSTSFM_COLMAP_FIFO_VERIFIER_TRACE",
                "RUSTSFM_COLMAP_FIFO_VERIFIER_THREADS",
                "RUSTSFM_COLMAP_FIFO_VERIFIER_REPLAY_TRACE",
            ] {
                std::env::remove_var(name);
            }
        }
    }

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

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn gpu_matching_routes_model_scoring() {
        let options = SiftMatchingOptions {
            use_gpu: true,
            ..SiftMatchingOptions::default()
        };
        assert_eq!(matching_backend_name(&options), "wgpu_match_and_score");
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn gpu_matching_persists_gpu_scored_two_view_geometry() -> Result<()> {
        if crate::gpu::WgpuContext::try_new_optional()?.is_none() {
            eprintln!("skipping GPU matching database test: no compatible adapter");
            return Ok(());
        }
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
                    params: vec![80.0, 80.0, 50.0, 50.0],
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
        let rotation = nalgebra::Rotation3::from_euler_angles(0.03, -0.04, 0.02).into_inner();
        let translation = nalgebra::Vector3::new(0.2, -0.03, 0.05);
        let mut left_keypoints = Vec::new();
        let mut right_keypoints = Vec::new();
        let mut descriptor_data = Vec::new();
        for index in 0..24usize {
            let point = nalgebra::Vector3::new(
                (index % 6) as f64 * 0.25 - 0.6,
                (index / 6) as f64 * 0.22 - 0.35,
                3.0 + (index % 5) as f64 * 0.35,
            );
            left_keypoints.push(ColmapKeypoint::new(
                (80.0 * point.x / point.z + 50.0) as f32,
                (80.0 * point.y / point.z + 50.0) as f32,
            ));
            let transformed = rotation * point + translation;
            right_keypoints.push(ColmapKeypoint::new(
                (80.0 * transformed.x / transformed.z + 50.0) as f32,
                (80.0 * transformed.y / transformed.z + 50.0) as f32,
            ));
            descriptor_data.extend(
                (0..128usize).map(|lane| ((index * 17 + lane * 3 + index * lane) % 256) as u8),
            );
        }
        db.write_keypoints(1, &left_keypoints)?;
        db.write_keypoints(2, &right_keypoints)?;
        let descriptors = ColmapDescriptors::new(
            crate::database::COLMAP_FEATURE_SIFT,
            24,
            128,
            descriptor_data,
        )?;
        db.write_descriptors(1, &descriptors)?;
        db.write_descriptors(2, &descriptors)?;
        drop(db);

        let report = match_features_to_database(
            &db_path,
            &MatchFeaturesOptions {
                pair_strategy: MatchingPairStrategy::Exhaustive,
                sift_matching: SiftMatchingOptions {
                    use_gpu: true,
                    max_num_matches: 128,
                    ..SiftMatchingOptions::default()
                },
                essential_threshold_px: 2.0,
                essential_iterations: 256,
                min_inliers: 15,
                min_triangulated: 8,
                min_num_matches: 15,
                random_seed: 17,
                ..MatchFeaturesOptions::default()
            },
        )?;
        assert_eq!(report.backend, "wgpu_match_and_score");
        assert_eq!(report.pair_count, 1);
        assert_eq!(report.pairs.len(), 1);
        assert!(report.pairs[0].num_inliers >= 15);
        let db = ColmapDatabase::open_read_only(&db_path)?;
        assert!(db.read_two_view_geometry(1, 2)?.inlier_matches.len() >= 15);
        drop(db);

        let existing_report = match_features_to_database(
            &db_path,
            &MatchFeaturesOptions {
                sift_matching: SiftMatchingOptions {
                    use_gpu: true,
                    ..SiftMatchingOptions::default()
                },
                essential_threshold_px: 2.0,
                essential_iterations: 256,
                min_inliers: 15,
                min_triangulated: 8,
                random_seed: 17,
                use_existing_matches: true,
                ..MatchFeaturesOptions::default()
            },
        )?;
        assert_eq!(existing_report.backend, "wgpu_match_and_score");
        assert_eq!(existing_report.pairs.len(), 1);
        assert!(existing_report.pairs[0].num_inliers >= 15);
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
