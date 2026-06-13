use crate::colmap::{
    export_colmap, read_camera_model, read_colmap_cameras, read_colmap_poses,
    world_to_camera_rotation,
};
use crate::correspondence_graph::ImagePairId;
use crate::database::{ColmapDatabase, ColmapTwoViewGeometry, DatabaseCache, DatabaseCacheOptions};
use crate::geometry::{
    camera_center, estimate_pair_geometry_with_cameras,
    estimate_pair_geometry_with_options_and_cameras, mean_pair_reprojection_error_with_cameras,
    pose_from_rotation_center, pose_rotation, pose_with_flipped_translation, relative_rotation_deg,
    PairEstimationOptions,
};
use crate::pose_graph::initialize_pose_graph;
use crate::sift::{
    extract_sift_features_with_options, match_sift_with_options, SiftExtractionOptions,
    SiftMatchingOptions,
};
use crate::types::{
    CameraModel, ImageFrame, PairGeometry, Point3D, Reconstruction, TrackObservation,
};
use crate::wide::{
    build_wide_descriptors, match_wide_mutual, match_wide_mutual_indices, rgb_to_gray,
};
use anyhow::{bail, Context, Result};
use image::ImageReader;
use nalgebra::{DMatrix, DVector, Matrix3, Matrix3x4, SMatrix, SVector, Vector3};
use rayon::prelude::*;
use rustslam::features::HammingMatcher;
use rustslam::tracker::{PnPProblem, PnPSolver};
use rustslam::{FeatureExtractor, FeatureMatcher, OrbExtractor, SE3};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct DatabasePairMatches {
    pub left: usize,
    pub right: usize,
    pub matches: Vec<rustslam::Match>,
}

#[derive(Debug, Clone)]
struct MapperDatabaseInput {
    cache: DatabaseCache,
    keypoints_by_name: HashMap<String, Vec<rustslam::KeyPoint>>,
    two_view_geometries: HashMap<ImagePairId, ColmapTwoViewGeometry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureType {
    Orb,
    Sift,
}

impl std::str::FromStr for FeatureType {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "orb" => Ok(Self::Orb),
            "sift" => Ok(Self::Sift),
            _ => bail!("unsupported feature type '{value}', expected orb or sift"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MapperConfig {
    pub input: PathBuf,
    pub output: PathBuf,
    pub reference: Option<PathBuf>,
    pub database: Option<PathBuf>,
    pub max_images: Option<usize>,
    pub feature_type: FeatureType,
    pub max_features: usize,
    pub match_ratio: f64,
    pub sift_extraction: SiftExtractionOptions,
    pub sift_matching: SiftMatchingOptions,
    pub max_hamming_distance: f32,
    pub local_window: usize,
    pub min_matches: usize,
    pub min_inliers: usize,
    pub min_triangulated: usize,
    pub essential_threshold_px: f32,
    pub essential_iterations: u32,
    pub pnp_threshold_px: f32,
    pub pnp_iterations: u32,
    pub max_reprojection_error_px: f32,
    pub pose_graph: bool,
    pub copy_images: bool,
    pub threads: Option<usize>,
    pub fx: Option<f32>,
    pub fy: Option<f32>,
    pub cx: Option<f32>,
    pub cy: Option<f32>,
}

impl Default for MapperConfig {
    fn default() -> Self {
        Self {
            input: PathBuf::new(),
            output: PathBuf::new(),
            reference: None,
            database: None,
            max_images: None,
            feature_type: FeatureType::Sift,
            max_features: 8192,
            match_ratio: 0.8,
            sift_extraction: SiftExtractionOptions::default(),
            sift_matching: SiftMatchingOptions::default(),
            max_hamming_distance: 160.0,
            local_window: 3,
            min_matches: 15,
            min_inliers: 15,
            min_triangulated: 4,
            essential_threshold_px: 2.0,
            essential_iterations: 10000,
            pnp_threshold_px: 8.0,
            pnp_iterations: 2000,
            max_reprojection_error_px: 8.0,
            pose_graph: true,
            copy_images: true,
            threads: None,
            fx: None,
            fy: None,
            cx: None,
            cy: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconstructionSummary {
    pub images: usize,
    pub registered_images: usize,
    pub points: usize,
    pub pairs: usize,
    pub elapsed_ms: f64,
    pub debug_log: Vec<String>,
}

pub fn run_reconstruction(config: &MapperConfig) -> Result<ReconstructionSummary> {
    if let Some(threads) = config.threads {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.max(1))
            .build_global();
    }
    let start = Instant::now();
    let paths = collect_images(&config.input, config.max_images)?;
    if paths.len() < 2 {
        bail!("need at least two images");
    }
    let mut reference_camera_setup = config
        .reference
        .as_ref()
        .and_then(|reference| reference_camera_setup(reference, &paths).ok());
    let mut camera = if let Some(setup) = &reference_camera_setup {
        setup
            .cameras
            .first()
            .copied()
            .unwrap_or_else(|| fallback_camera(&paths[0]))
    } else if let Some(reference) = &config.reference {
        read_camera_model(reference).unwrap_or_else(|_| fallback_camera(&paths[0]))
    } else {
        fallback_camera(&paths[0])
    };
    if let Some(fx) = config.fx {
        camera.set_fx(fx);
    }
    if let Some(fy) = config.fy {
        camera.set_fy(fy);
    }
    if let Some(cx) = config.cx {
        camera.set_cx(cx);
    }
    if let Some(cy) = config.cy {
        camera.set_cy(cy);
    }
    if let Some(setup) = &mut reference_camera_setup {
        for setup_camera in &mut setup.cameras {
            if let Some(fx) = config.fx {
                setup_camera.set_fx(fx);
            }
            if let Some(fy) = config.fy {
                setup_camera.set_fy(fy);
            }
            if let Some(cx) = config.cx {
                setup_camera.set_cx(cx);
            }
            if let Some(cy) = config.cy {
                setup_camera.set_cy(cy);
            }
        }
        if let Some(first) = setup.cameras.first().copied() {
            camera = first;
        }
    }

    let frames_start = Instant::now();
    let mut sift_extraction = config.sift_extraction.clone();
    sift_extraction.max_num_features = config.max_features;
    let mut sift_matching = config.sift_matching.clone();
    sift_matching.max_ratio = config.match_ratio as f32;
    let mut frames = extract_frames(
        &paths,
        config.max_features,
        config.feature_type,
        &sift_extraction,
    )?;
    let mapper_database =
        load_mapper_database(config.database.as_deref(), &frames, config.min_matches)?;
    if let Some(database) = mapper_database.as_ref() {
        apply_database_keypoints(&mut frames, &database.keypoints_by_name);
        if reference_camera_setup.is_none() {
            reference_camera_setup = database_camera_setup(&database.cache, &paths).ok();
            if let Some(setup) = &mut reference_camera_setup {
                for setup_camera in &mut setup.cameras {
                    if let Some(fx) = config.fx {
                        setup_camera.set_fx(fx);
                    }
                    if let Some(fy) = config.fy {
                        setup_camera.set_fy(fy);
                    }
                    if let Some(cx) = config.cx {
                        setup_camera.set_cx(cx);
                    }
                    if let Some(cy) = config.cy {
                        setup_camera.set_cy(cy);
                    }
                }
                if let Some(first) = setup.cameras.first().copied() {
                    camera = first;
                }
            }
        }
    }
    let frames_elapsed_ms = frames_start.elapsed().as_secs_f64() * 1000.0;
    if reference_camera_setup.is_none() {
        camera.width = frames[0].width;
        camera.height = frames[0].height;
    }
    let pair_start = Instant::now();
    let pairs = if let Some(database) = mapper_database.as_ref() {
        estimate_database_pair_geometries(
            &frames,
            &database.cache,
            &database.two_view_geometries,
            camera,
            reference_camera_setup.as_ref(),
            config,
        )?
    } else {
        build_pair_graph(
            &frames,
            camera,
            reference_camera_setup.as_ref(),
            config,
            &sift_matching,
        )?
    };
    let pair_elapsed_ms = pair_start.elapsed().as_secs_f64() * 1000.0;
    if pairs.is_empty() {
        bail!("no verified image pairs");
    }
    let mut debug_log = Vec::new();
    debug_log.push(format!(
        "timing_extract_ms={:.2} timing_pairs_ms={:.2}",
        frames_elapsed_ms, pair_elapsed_ms
    ));
    debug_log.extend(pair_connectivity_summary(&pairs, &frames));
    if let Some(reference) = &config.reference {
        debug_log.extend(pair_reference_error_summary(&pairs, &frames, reference));
    }
    let incremental_start = Instant::now();
    let (mut reconstruction, incremental_log) = incremental_map(
        &frames,
        camera,
        reference_camera_setup.as_ref(),
        &pairs,
        config,
    )?;
    let incremental_elapsed_ms = incremental_start.elapsed().as_secs_f64() * 1000.0;
    debug_log.push(format!("timing_incremental_ms={incremental_elapsed_ms:.2}"));
    debug_log.extend(incremental_log);
    debug_log.extend(pair_quality_summary(&pairs));
    if config.pose_graph && reconstruction.poses.iter().all(|p| p.is_some()) {
        let pose_graph_start = Instant::now();
        let graph_poses = initialize_pose_graph(frames.len(), &pairs, &reconstruction.poses);
        reconstruction.poses = graph_poses.into_iter().map(Some).collect();
        reconstruction.observations = frames
            .iter()
            .map(|f| vec![None; f.keypoints.len()])
            .collect();
        reconstruction.points.clear();
        reconstruction.point_ids.clear();
        let mapping_pairs = pairs
            .iter()
            .filter(|pair| !pair.pose_graph_only)
            .cloned()
            .collect::<Vec<_>>();
        rebuild_tracks_from_pair_graph(&frames, &mapping_pairs, &mut reconstruction, config);
        if std::env::var_os("RUSTSFM_FILTER_TRACKS").is_some() {
            filter_reprojection_tracks(&frames, &mut reconstruction, config);
        }
        if std::env::var_os("RUSTSFM_SKIP_POSE_REFINE").is_none() {
            refine_registered_poses_pose_only(&frames, &mapping_pairs, &mut reconstruction, config);
        }
        if std::env::var_os("RUSTSFM_EXPERIMENTAL_BA").is_some() {
            if let Some(report) = crate::ba::refine_bundle_adjustment(
                &frames,
                &mut reconstruction,
                crate::ba::BundleAdjustmentOptions {
                    iterations: global_ba_iterations(),
                    huber_delta_px: global_ba_huber_delta_px(),
                    max_observation_error_px: global_ba_max_observation_error_px(config),
                },
            ) {
                debug_log.push(format!(
                    "schur_ba iterations={} observations={} cost={:.6}->{:.6}",
                    report.iterations, report.observations, report.initial_cost, report.final_cost
                ));
            }
            if std::env::var_os("RUSTSFM_FILTER_TRACKS").is_some() {
                let removed = filter_reprojection_tracks(&frames, &mut reconstruction, config);
                debug_log.push(format!("track_filter removed_observations={removed}"));
            }
        }
        debug_log.push(format!(
            "timing_pose_graph_ms={:.2}",
            pose_graph_start.elapsed().as_secs_f64() * 1000.0
        ));
        debug_log.push("pose_graph_refinement enabled".to_string());
    }
    let registered_images = reconstruction.poses.iter().filter(|p| p.is_some()).count();
    export_colmap(&config.output, &reconstruction, config.copy_images)?;
    Ok(ReconstructionSummary {
        images: frames.len(),
        registered_images,
        points: reconstruction.points.len(),
        pairs: pairs.len(),
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        debug_log,
    })
}

fn collect_images(input: &Path, max_images: Option<usize>) -> Result<Vec<PathBuf>> {
    let mut paths = fs::read_dir(input)
        .with_context(|| format!("failed to read {}", input.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.extension()
                .and_then(|e| e.to_str())
                .map(|e| matches!(e.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    paths.sort();
    if let Some(max) = max_images {
        paths.truncate(max);
    }
    Ok(paths)
}

#[derive(Debug, Clone)]
struct ReferenceCameraSetup {
    cameras: Vec<CameraModel>,
    camera_ids: Vec<u32>,
    image_ids: Vec<u32>,
    image_camera_indices: Vec<usize>,
}

fn reference_camera_setup(
    reference: &Path,
    image_paths: &[PathBuf],
) -> Result<ReferenceCameraSetup> {
    let cameras_with_ids = read_colmap_cameras(reference)?;
    if cameras_with_ids.is_empty() {
        bail!("reference model has no cameras");
    }
    let camera_ids = cameras_with_ids
        .iter()
        .map(|(camera_id, _)| *camera_id)
        .collect::<Vec<_>>();
    let cameras = cameras_with_ids
        .iter()
        .map(|(_, camera)| *camera)
        .collect::<Vec<_>>();
    let camera_index_by_id = camera_ids
        .iter()
        .enumerate()
        .map(|(idx, &camera_id)| (camera_id, idx))
        .collect::<HashMap<_, _>>();
    let poses = read_colmap_poses(reference)?;
    let pose_by_name = poses
        .iter()
        .map(|pose| (pose.name.as_str(), pose))
        .collect::<HashMap<_, _>>();

    let mut image_ids = Vec::with_capacity(image_paths.len());
    let mut image_camera_indices = Vec::with_capacity(image_paths.len());
    for (idx, path) in image_paths.iter().enumerate() {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if let Some(pose) = pose_by_name.get(name) {
            image_ids.push(pose.image_id);
            image_camera_indices.push(*camera_index_by_id.get(&pose.camera_id).unwrap_or(&0));
        } else {
            image_ids.push(idx as u32 + 1);
            image_camera_indices.push(0);
        }
    }

    Ok(ReferenceCameraSetup {
        cameras,
        camera_ids,
        image_ids,
        image_camera_indices,
    })
}

fn load_mapper_database(
    database: Option<&Path>,
    frames: &[ImageFrame],
    min_num_matches: usize,
) -> Result<Option<MapperDatabaseInput>> {
    let Some(database) = database else {
        return Ok(None);
    };
    let image_names = frames
        .iter()
        .map(|frame| frame.name.clone())
        .collect::<BTreeSet<_>>();
    let db = ColmapDatabase::open(database)?;
    let cache = db.load_cache(&DatabaseCacheOptions {
        min_num_matches,
        ignore_watermarks: false,
        image_names,
        load_all_images: false,
    })?;
    let mut keypoints_by_name = HashMap::new();
    for image in cache.images.values() {
        let keypoints = db
            .read_keypoints(image.image_id)?
            .into_iter()
            .map(|kp| kp.to_keypoint())
            .collect::<Vec<_>>();
        keypoints_by_name.insert(image.name.clone(), keypoints);
    }
    let two_view_geometries = db
        .read_two_view_geometries()?
        .into_iter()
        .collect::<HashMap<_, _>>();
    Ok(Some(MapperDatabaseInput {
        cache,
        keypoints_by_name,
        two_view_geometries,
    }))
}

fn database_camera_setup(
    cache: &DatabaseCache,
    image_paths: &[PathBuf],
) -> Result<ReferenceCameraSetup> {
    if cache.cameras.is_empty() {
        bail!("database cache has no cameras");
    }
    let mut camera_ids = Vec::with_capacity(cache.cameras.len());
    let mut cameras = Vec::with_capacity(cache.cameras.len());
    for (&camera_id, db_camera) in &cache.cameras {
        let camera = CameraModel::from_colmap(
            db_camera.camera.model_id,
            db_camera.camera.width,
            db_camera.camera.height,
            &db_camera.camera.params,
        )
        .with_context(|| format!("unsupported database camera_id={camera_id}"))?;
        camera_ids.push(camera_id);
        cameras.push(camera);
    }
    let camera_index_by_id = camera_ids
        .iter()
        .enumerate()
        .map(|(idx, &camera_id)| (camera_id, idx))
        .collect::<HashMap<_, _>>();
    let image_by_name = cache
        .images
        .values()
        .map(|image| (image.name.as_str(), image))
        .collect::<HashMap<_, _>>();

    let mut image_ids = Vec::with_capacity(image_paths.len());
    let mut image_camera_indices = Vec::with_capacity(image_paths.len());
    for (idx, path) in image_paths.iter().enumerate() {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if let Some(image) = image_by_name.get(name) {
            image_ids.push(image.image_id);
            image_camera_indices.push(*camera_index_by_id.get(&image.camera_id).unwrap_or(&0));
        } else {
            image_ids.push(idx as u32 + 1);
            image_camera_indices.push(0);
        }
    }

    Ok(ReferenceCameraSetup {
        cameras,
        camera_ids,
        image_ids,
        image_camera_indices,
    })
}

fn apply_database_keypoints(
    frames: &mut [ImageFrame],
    keypoints_by_name: &HashMap<String, Vec<rustslam::KeyPoint>>,
) {
    for frame in frames {
        let Some(keypoints) = keypoints_by_name.get(frame.name.as_str()) else {
            continue;
        };
        if !keypoints.is_empty() {
            frame.keypoints = keypoints.clone();
            frame.descriptors = rustslam::Descriptors::new();
            frame.sift = crate::sift::SiftFeatures::default();
            frame.wide_descriptors = crate::wide::WideDescriptors {
                data: Vec::new(),
                dim: 0,
                count: 0,
            };
            frame.strong_feature_indices = Vec::new();
            frame.colors = sample_keypoint_colors(frame);
        }
    }
}

fn sample_keypoint_colors(frame: &ImageFrame) -> Vec<[u8; 3]> {
    let Ok(reader) = ImageReader::open(&frame.path) else {
        return vec![[0, 0, 0]; frame.keypoints.len()];
    };
    let Ok(image) = reader.decode() else {
        return vec![[0, 0, 0]; frame.keypoints.len()];
    };
    let image = image.to_rgb8();
    let width = image.width().max(1);
    let height = image.height().max(1);
    frame
        .keypoints
        .iter()
        .map(|kp| {
            let x = kp.x().round().clamp(0.0, (width - 1) as f32) as u32;
            let y = kp.y().round().clamp(0.0, (height - 1) as f32) as u32;
            image.get_pixel(x, y).0
        })
        .collect()
}

fn fallback_camera(first_image: &Path) -> CameraModel {
    let image = ImageReader::open(first_image)
        .ok()
        .and_then(|r| r.decode().ok());
    let (width, height) = image
        .as_ref()
        .map(|i| (i.width(), i.height()))
        .unwrap_or((1536, 2048));
    let focal = width.max(height) as f32 * 1.2;
    CameraModel::new_pinhole(
        width,
        height,
        focal,
        focal,
        width as f32 * 0.5,
        height as f32 * 0.5,
    )
}

fn global_ba_iterations() -> usize {
    std::env::var("RUSTSFM_BA_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(8)
}

fn global_ba_huber_delta_px() -> f64 {
    std::env::var("RUSTSFM_BA_HUBER_PX")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(4.0)
}

fn global_ba_max_observation_error_px(config: &MapperConfig) -> f64 {
    std::env::var("RUSTSFM_BA_MAX_OBS_ERROR_PX")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(config.max_reprojection_error_px as f64 * 2.0)
}

fn extract_frames(
    paths: &[PathBuf],
    max_features: usize,
    feature_type: FeatureType,
    sift_options: &SiftExtractionOptions,
) -> Result<Vec<ImageFrame>> {
    paths
        .par_iter()
        .enumerate()
        .map(|(id, path)| -> Result<ImageFrame> {
            let image = ImageReader::open(path)
                .with_context(|| format!("failed to open {}", path.display()))?
                .decode()
                .with_context(|| format!("failed to decode {}", path.display()))?
                .to_rgb8();
            let (width, height) = image.dimensions();
            let (keypoints, descriptors, sift) = match feature_type {
                FeatureType::Orb => {
                    let mut extractor = OrbExtractor::new(max_features);
                    let (keypoints, descriptors) = extractor
                        .detect_and_compute(image.as_raw(), width, height)
                        .map_err(|e| anyhow::anyhow!("feature extraction failed: {e}"))?;
                    (keypoints, descriptors, Default::default())
                }
                FeatureType::Sift => {
                    let sift = extract_sift_features_with_options(
                        image.as_raw(),
                        width,
                        height,
                        sift_options,
                    )?;
                    (sift.keypoints.clone(), rustslam::Descriptors::new(), sift)
                }
            };
            let gray = rgb_to_gray(image.as_raw(), width, height);
            let wide_descriptors = build_wide_descriptors(&gray, width, height, &keypoints);
            let strong_feature_indices = strong_feature_indices(&keypoints, 1024);
            let colors = keypoints
                .iter()
                .map(|kp| {
                    let x = kp.x().round().clamp(0.0, (width - 1) as f32) as u32;
                    let y = kp.y().round().clamp(0.0, (height - 1) as f32) as u32;
                    let p = image.get_pixel(x, y).0;
                    [p[0], p[1], p[2]]
                })
                .collect();
            Ok(ImageFrame {
                id,
                name: path.file_name().unwrap().to_string_lossy().to_string(),
                path: path.clone(),
                width,
                height,
                keypoints,
                descriptors,
                sift,
                wide_descriptors,
                strong_feature_indices,
                colors,
            })
        })
        .collect()
}

fn strong_feature_indices(keypoints: &[rustslam::KeyPoint], limit: usize) -> Vec<usize> {
    let mut indices = (0..keypoints.len()).collect::<Vec<_>>();
    indices.sort_by(|&a, &b| {
        keypoints[b]
            .response
            .partial_cmp(&keypoints[a].response)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices.truncate(limit.min(indices.len()));
    indices
}

fn build_pair_graph(
    frames: &[ImageFrame],
    camera: CameraModel,
    reference_camera_setup: Option<&ReferenceCameraSetup>,
    config: &MapperConfig,
    sift_matching: &SiftMatchingOptions,
) -> Result<Vec<PairGeometry>> {
    let matcher = HammingMatcher::new(2).with_ratio_threshold(config.match_ratio);
    let mut candidates = Vec::new();
    for offset in 1..=config.local_window.min(frames.len() - 1) {
        for left in 0..frames.len() - offset {
            candidates.push((left, left + offset));
        }
    }
    add_segment_bridge_candidates(frames.len(), &mut candidates);
    let mut pairs = candidates
        .par_iter()
        .filter_map(|&(left, right)| {
            let left_camera = setup_camera_for_image(reference_camera_setup, left, camera);
            let right_camera = setup_camera_for_image(reference_camera_setup, right, camera);
            estimate_candidate_pair(
                left,
                right,
                frames,
                left_camera,
                right_camera,
                config,
                Some(&matcher),
                sift_matching,
            )
        })
        .collect::<Vec<_>>();
    if std::env::var_os("RUSTSFM_RING_CLOSURE").is_some() {
        let mut closure_pairs = Vec::new();
        for (left, right) in intra_segment_ring_candidates(frames.len(), 192) {
            let left_camera = setup_camera_for_image(reference_camera_setup, left, camera);
            let right_camera = setup_camera_for_image(reference_camera_setup, right, camera);
            if let Some(pair) = estimate_candidate_pair(
                left,
                right,
                frames,
                left_camera,
                right_camera,
                config,
                None,
                sift_matching,
            ) {
                closure_pairs.push(pair);
            }
        }
        retain_best_ring_closures(&mut closure_pairs);
        pairs.extend(closure_pairs);
    }
    enforce_adjacent_translation_continuity(&mut pairs);
    regularize_low_parallax_adjacent_translations(&mut pairs);
    filter_translation_outlier_pairs(&mut pairs);
    Ok(pairs)
}

pub fn database_pair_matches_for_frames(
    frames: &[ImageFrame],
    cache: &DatabaseCache,
) -> Result<Vec<DatabasePairMatches>> {
    let frame_by_name = frames
        .iter()
        .enumerate()
        .map(|(idx, frame)| (frame.name.as_str(), idx))
        .collect::<HashMap<_, _>>();
    let mut out = Vec::new();
    for pair_id in cache.correspondence_graph.image_pairs() {
        let (image_id1, image_id2) = crate::correspondence_graph::pair_id_to_image_pair(pair_id)
            .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        let Some(image1) = cache.images.get(&image_id1) else {
            continue;
        };
        let Some(image2) = cache.images.get(&image_id2) else {
            continue;
        };
        let Some(&frame1) = frame_by_name.get(image1.name.as_str()) else {
            continue;
        };
        let Some(&frame2) = frame_by_name.get(image2.name.as_str()) else {
            continue;
        };
        let (left, right) = if frame1 <= frame2 {
            (frame1, frame2)
        } else {
            (frame2, frame1)
        };
        let (left_image_id, right_image_id) = if frame1 <= frame2 {
            (image_id1, image_id2)
        } else {
            (image_id2, image_id1)
        };
        let matches = cache
            .correspondence_graph
            .extract_matches_between_images(left_image_id, right_image_id)
            .map_err(|err| anyhow::anyhow!("{err:?}"))?
            .into_iter()
            .map(Into::into)
            .collect();
        out.push(DatabasePairMatches {
            left,
            right,
            matches,
        });
    }
    out.sort_by_key(|pair| (pair.left, pair.right));
    Ok(out)
}

#[allow(dead_code)]
fn estimate_database_pair_geometries(
    frames: &[ImageFrame],
    cache: &DatabaseCache,
    stored_geometries: &HashMap<ImagePairId, ColmapTwoViewGeometry>,
    camera: CameraModel,
    reference_camera_setup: Option<&ReferenceCameraSetup>,
    config: &MapperConfig,
) -> Result<Vec<PairGeometry>> {
    let pair_matches = database_pair_matches_for_frames(frames, cache)?;
    let pairs = pair_matches
        .par_iter()
        .filter_map(|pair| {
            let left_camera = setup_camera_for_image(reference_camera_setup, pair.left, camera);
            let right_camera = setup_camera_for_image(reference_camera_setup, pair.right, camera);
            database_pair_geometry_from_stored_pose(
                pair,
                frames,
                cache,
                stored_geometries,
                left_camera,
                right_camera,
                config,
            )
            .or_else(|| {
                estimate_pair_geometry_with_cameras(
                    pair.left,
                    pair.right,
                    &frames[pair.left],
                    &frames[pair.right],
                    &pair.matches,
                    left_camera,
                    right_camera,
                    config.essential_threshold_px,
                    config.essential_iterations,
                    config.min_inliers,
                    config.min_triangulated,
                )
            })
            .filter(keep_verified_pair)
        })
        .collect::<Vec<_>>();
    Ok(pairs)
}

fn database_pair_geometry_from_stored_pose(
    pair: &DatabasePairMatches,
    frames: &[ImageFrame],
    cache: &DatabaseCache,
    stored_geometries: &HashMap<ImagePairId, ColmapTwoViewGeometry>,
    left_camera: CameraModel,
    right_camera: CameraModel,
    config: &MapperConfig,
) -> Option<PairGeometry> {
    let left_name = frames.get(pair.left)?.name.as_str();
    let right_name = frames.get(pair.right)?.name.as_str();
    let image_by_name = cache
        .images
        .values()
        .map(|image| (image.name.as_str(), image.image_id))
        .collect::<HashMap<_, _>>();
    let left_image_id = *image_by_name.get(left_name)?;
    let right_image_id = *image_by_name.get(right_name)?;
    if left_image_id > right_image_id {
        return None;
    }
    let pair_id =
        crate::correspondence_graph::image_pair_to_pair_id(left_image_id, right_image_id).ok()?;
    let geometry = stored_geometries.get(&pair_id)?;
    let pose = stored_two_view_pose(geometry)?;
    let metrics = stored_pose_pair_metrics(
        pair,
        &frames[pair.left],
        &frames[pair.right],
        pose,
        left_camera,
        right_camera,
        config.max_reprojection_error_px,
    )?;
    if metrics.triangulated < config.min_triangulated
        || metrics.inlier_matches.len() < config.min_inliers
    {
        return None;
    }
    Some(PairGeometry {
        left: pair.left,
        right: pair.right,
        matches: pair.matches.clone(),
        inlier_matches: metrics.inlier_matches,
        relative_pose: pose,
        inliers: metrics.inliers,
        triangulated: metrics.triangulated,
        mean_reprojection_error_px: metrics.mean_reprojection_error_px,
        rotation_deg: relative_rotation_deg(pose, SE3::identity()),
        median_triangulation_angle_deg: metrics.median_triangulation_angle_deg,
        pose_graph_only: false,
    })
}

fn stored_two_view_pose(geometry: &ColmapTwoViewGeometry) -> Option<SE3> {
    let q = geometry.qvec?;
    let t = geometry.tvec?;
    let rotation = glam::Quat::from_xyzw(q[1] as f32, q[2] as f32, q[3] as f32, q[0] as f32);
    if !rotation.is_finite() {
        return None;
    }
    let rotation = rotation.normalize();
    let translation = glam::Vec3::new(t[0] as f32, t[1] as f32, t[2] as f32);
    let translation = translation.try_normalize()?;
    Some(SE3::from_quat_translation(rotation, translation))
}

struct StoredPosePairMetrics {
    inlier_matches: Vec<rustslam::Match>,
    inliers: usize,
    triangulated: usize,
    mean_reprojection_error_px: f32,
    median_triangulation_angle_deg: f32,
}

fn stored_pose_pair_metrics(
    pair: &DatabasePairMatches,
    left: &ImageFrame,
    right: &ImageFrame,
    pose: SE3,
    left_camera: CameraModel,
    right_camera: CameraModel,
    max_reprojection_error_px: f32,
) -> Option<StoredPosePairMetrics> {
    let mut inlier_matches = Vec::new();
    let mut reproj_sum = 0.0f32;
    let mut triangulation_angles = Vec::new();
    for m in &pair.matches {
        let li = m.query_idx as usize;
        let ri = m.train_idx as usize;
        if li >= left.keypoints.len() || ri >= right.keypoints.len() {
            continue;
        }
        let lk = &left.keypoints[li];
        let rk = &right.keypoints[ri];
        let left_xy = left_camera.normalize(lk.x(), lk.y());
        let right_xy = right_camera.normalize(rk.x(), rk.y());
        let Some(xyz) =
            crate::two_view::triangulate_world_point(SE3::identity(), pose, left_xy, right_xy)
        else {
            continue;
        };
        let err = mean_pair_reprojection_error_with_cameras(
            xyz,
            SE3::identity(),
            pose,
            [lk.x(), lk.y()],
            [rk.x(), rk.y()],
            left_camera,
            right_camera,
        );
        if !err.is_finite() || err > max_reprojection_error_px {
            continue;
        }
        if let Some(angle) = pair_triangulation_angle_deg(SE3::identity(), pose, xyz) {
            triangulation_angles.push(angle);
        }
        reproj_sum += err;
        inlier_matches.push(m.clone());
    }
    let inliers = inlier_matches.len();
    if inliers == 0 {
        return None;
    }
    Some(StoredPosePairMetrics {
        inlier_matches,
        inliers,
        triangulated: triangulation_angles.len(),
        mean_reprojection_error_px: reproj_sum / inliers as f32,
        median_triangulation_angle_deg: median_f32(&mut triangulation_angles),
    })
}

fn pair_triangulation_angle_deg(left_pose: SE3, right_pose: SE3, point: [f32; 3]) -> Option<f32> {
    let c1 = camera_center(left_pose);
    let c2 = camera_center(right_pose);
    let p = glam::Vec3::from_array(point);
    let v1 = (p - c1).try_normalize()?;
    let v2 = (p - c2).try_normalize()?;
    Some(v1.dot(v2).abs().clamp(-1.0, 1.0).acos().to_degrees())
}

fn median_f32(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values[values.len() / 2]
}

fn setup_camera_for_image(
    setup: Option<&ReferenceCameraSetup>,
    image: usize,
    fallback: CameraModel,
) -> CameraModel {
    setup
        .and_then(|setup| {
            setup
                .image_camera_indices
                .get(image)
                .and_then(|&camera_idx| setup.cameras.get(camera_idx))
        })
        .copied()
        .unwrap_or(fallback)
}

fn estimate_candidate_pair(
    left: usize,
    right: usize,
    frames: &[ImageFrame],
    left_camera: CameraModel,
    right_camera: CameraModel,
    config: &MapperConfig,
    matcher: Option<&HammingMatcher>,
    sift_matching: &SiftMatchingOptions,
) -> Option<PairGeometry> {
    let matches = if config.feature_type == FeatureType::Sift {
        if is_ring_bridge_candidate(left, right) {
            let left_strong = limited_indices(&frames[left].strong_feature_indices, 1024);
            let right_strong = limited_indices(&frames[right].strong_feature_indices, 1024);
            match_wide_mutual_indices(
                &frames[left].wide_descriptors,
                &frames[right].wide_descriptors,
                left_strong,
                right_strong,
                0.9,
                0.85,
            )
        } else {
            match_sift_with_options(&frames[left].sift, &frames[right].sift, sift_matching)
        }
    } else if is_ring_bridge_candidate(left, right) {
        let loose = HammingMatcher::new(2).with_ratio_threshold(0.92);
        let mut matches = loose
            .match_descriptors(&frames[left].descriptors, &frames[right].descriptors)
            .ok()?
            .into_iter()
            .filter(|m| m.distance <= config.max_hamming_distance.max(180.0))
            .collect::<Vec<_>>();
        let wide = match_wide_mutual(
            &frames[left].wide_descriptors,
            &frames[right].wide_descriptors,
            0.9,
            0.85,
        );
        merge_matches(&mut matches, wide);
        matches
    } else {
        let matcher = matcher?;
        mutual_matches(matcher, &frames[left], &frames[right])
            .ok()?
            .into_iter()
            .filter(|m| m.distance <= config.max_hamming_distance)
            .collect::<Vec<_>>()
    };
    let mut pair = if is_ring_bridge_candidate(left, right) {
        estimate_pair_geometry_with_options_and_cameras(
            left,
            right,
            &frames[left],
            &frames[right],
            &matches,
            left_camera,
            right_camera,
            config.essential_threshold_px,
            config.essential_iterations.min(200),
            config.min_inliers,
            config.min_triangulated,
            PairEstimationOptions {
                max_pose_matches: 128,
                use_hartley_refinement: false,
                use_five_point: false,
                refine_sampson: false,
            },
        )
    } else {
        estimate_pair_geometry_with_cameras(
            left,
            right,
            &frames[left],
            &frames[right],
            &matches,
            left_camera,
            right_camera,
            config.essential_threshold_px,
            config.essential_iterations,
            config.min_inliers,
            config.min_triangulated,
        )
    }?;
    if is_ring_bridge_candidate(left, right) {
        pair.pose_graph_only = true;
    }
    keep_verified_pair(&pair).then_some(pair)
}

fn limited_indices(indices: &[usize], limit: usize) -> &[usize] {
    &indices[..indices.len().min(limit)]
}

fn keep_verified_pair(pair: &PairGeometry) -> bool {
    let offset = pair.right.abs_diff(pair.left);
    if offset <= 1 || is_ring_bridge_candidate(pair.left, pair.right) {
        if is_ring_bridge_candidate(pair.left, pair.right) {
            return pair.inliers >= 40
                && pair.mean_reprojection_error_px <= 1.0
                && pair.median_triangulation_angle_deg >= 0.75
                && pair.rotation_deg <= 8.0;
        }
        return pair.inliers >= 15 && pair.mean_reprojection_error_px <= 8.0;
    }
    let min_inliers = 40 + offset.saturating_sub(2) * 12;
    pair.inliers >= min_inliers
        && pair.mean_reprojection_error_px <= 1.8
        && pair.rotation_deg <= 20.0
}

fn retain_best_ring_closures(pairs: &mut Vec<PairGeometry>) {
    pairs.retain(|pair| is_ring_bridge_candidate(pair.left, pair.right));
    pairs.sort_by(|a, b| {
        ring_closure_score(b)
            .partial_cmp(&ring_closure_score(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let max_per_segment = std::env::var("RUSTSFM_RING_CLOSURE_COUNT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(2);
    let mut kept = Vec::new();
    let mut per_segment = HashMap::<usize, usize>::new();
    for pair in pairs.drain(..) {
        let segment = pair.left / 192;
        let count = per_segment.entry(segment).or_insert(0);
        if *count >= max_per_segment {
            continue;
        }
        *count += 1;
        kept.push(pair);
    }
    *pairs = kept;
}

fn ring_closure_score(pair: &PairGeometry) -> f32 {
    let reproj = pair.mean_reprojection_error_px.max(0.1);
    let tri = pair.median_triangulation_angle_deg.max(0.05);
    pair.inliers as f32 * tri.sqrt() / reproj
}

fn is_ring_bridge_candidate(left: usize, right: usize) -> bool {
    const PERIOD: usize = 192;
    if left >= right {
        return false;
    }
    let delta = right - left;
    if delta < PERIOD.saturating_sub(4) || delta > PERIOD + 4 {
        return false;
    }
    left % PERIOD <= 3 || left % PERIOD >= PERIOD.saturating_sub(4)
}

fn add_segment_bridge_candidates(frame_count: usize, candidates: &mut Vec<(usize, usize)>) {
    const PERIOD: usize = 192;
    if frame_count <= PERIOD {
        return;
    }
    let segments = frame_count.div_ceil(PERIOD);
    for segment in 1..segments {
        let right_start = segment * PERIOD;
        if right_start >= frame_count {
            continue;
        }
        let left_start = (segment - 1) * PERIOD;
        for left_offset in 0..3 {
            let left = left_start + left_offset;
            if left < frame_count {
                candidates.push((left, right_start));
                if right_start + 1 < frame_count {
                    candidates.push((left, right_start + 1));
                }
            }
        }
        let left_end = right_start.saturating_sub(1);
        for back in 0..3 {
            let left = left_end.saturating_sub(back);
            if left < right_start {
                candidates.push((left, right_start));
                if right_start + 1 < frame_count {
                    candidates.push((left, right_start + 1));
                }
            }
        }
    }
    candidates.sort_unstable();
    candidates.dedup();
}

fn intra_segment_ring_candidates(frame_count: usize, period: usize) -> Vec<(usize, usize)> {
    let mut candidates = Vec::new();
    if frame_count < period {
        return candidates;
    }
    let segments = frame_count / period;
    for segment in 0..segments {
        let start = segment * period;
        let end = start + period - 1;
        if end >= frame_count {
            continue;
        }
        let seam_window = 3usize;
        for left_offset in 0..seam_window {
            for right_back in 0..seam_window {
                if left_offset + right_back + 1 > seam_window {
                    continue;
                }
                let left = start + left_offset;
                let right = end.saturating_sub(right_back);
                if left < right {
                    candidates.push((left, right));
                }
            }
        }
    }
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

fn pair_quality_summary(pairs: &[PairGeometry]) -> Vec<String> {
    if pairs.is_empty() {
        return Vec::new();
    }
    let mean_reproj = pairs
        .iter()
        .map(|p| p.mean_reprojection_error_px)
        .sum::<f32>()
        / pairs.len() as f32;
    let mean_inliers = pairs.iter().map(|p| p.inliers).sum::<usize>() as f32 / pairs.len() as f32;
    let high_error = pairs
        .iter()
        .filter(|p| p.mean_reprojection_error_px > 4.0 || p.rotation_deg > 25.0)
        .count();
    let mean_triangulation_angle = pairs
        .iter()
        .map(|p| p.median_triangulation_angle_deg)
        .sum::<f32>()
        / pairs.len() as f32;
    vec![format!(
        "pair_quality mean_inliers={:.1} mean_reproj={:.3} mean_tri_angle={:.3}deg high_error_pairs={}/{}",
        mean_inliers,
        mean_reproj,
        mean_triangulation_angle,
        high_error,
        pairs.len()
    )]
}

fn pair_connectivity_summary(pairs: &[PairGeometry], frames: &[ImageFrame]) -> Vec<String> {
    let mut degree = vec![0usize; frames.len()];
    let mut first_edges = Vec::new();
    for pair in pairs {
        degree[pair.left] += 1;
        degree[pair.right] += 1;
        if pair.left < 6 || pair.right < 6 {
            first_edges.push(format!(
                "{}->{}(in={},tri={},err={:.2})",
                pair.left + 1,
                pair.right + 1,
                pair.inliers,
                pair.triangulated,
                pair.mean_reprojection_error_px
            ));
        }
    }
    let isolated = degree.iter().filter(|&&d| d == 0).count();
    let min_degree = degree.iter().copied().min().unwrap_or(0);
    let max_degree = degree.iter().copied().max().unwrap_or(0);
    let mean_degree = if degree.is_empty() {
        0.0
    } else {
        degree.iter().sum::<usize>() as f32 / degree.len() as f32
    };
    first_edges.truncate(18);
    vec![
        format!(
            "pair_connectivity isolated={} min_degree={} mean_degree={:.2} max_degree={}",
            isolated, min_degree, mean_degree, max_degree
        ),
        format!("pair_first_edges {}", first_edges.join(" ")),
    ]
}

fn pair_reference_error_summary(
    pairs: &[PairGeometry],
    frames: &[ImageFrame],
    reference: &Path,
) -> Vec<String> {
    let Ok(poses) = read_colmap_poses(reference) else {
        return Vec::new();
    };
    let by_name = poses
        .iter()
        .map(|pose| (pose.name.as_str(), pose))
        .collect::<HashMap<_, _>>();
    let mut rot_errors = Vec::new();
    let mut trans_errors = Vec::new();
    let mut worst_pairs = Vec::<(f64, String)>::new();
    for pair in pairs {
        let Some(left_ref) = by_name.get(frames[pair.left].name.as_str()) else {
            continue;
        };
        let Some(right_ref) = by_name.get(frames[pair.right].name.as_str()) else {
            continue;
        };
        let ref_rel = reference_relative_pose(left_ref, right_ref);
        let cand_rel = rust_relative_pose_parts(pair.relative_pose);
        let rot_error = rotation_angle_deg_na(ref_rel.0.transpose() * cand_rel.0);
        let trans_error = if let (Some(a), Some(b)) = (
            ref_rel.1.try_normalize(1.0e-12),
            cand_rel.1.try_normalize(1.0e-12),
        ) {
            a.dot(&b).clamp(-1.0, 1.0).acos().to_degrees()
        } else {
            f64::INFINITY
        };
        if trans_error.is_finite() {
            trans_errors.push(trans_error);
        }
        rot_errors.push(rot_error);
        let score = rot_error + trans_error.min(180.0);
        let label = format!(
            "{}->{} rot={:.4}deg trans={:.4}deg inliers={} reproj={:.3} tri_angle={:.4}deg",
            frames[pair.left].name,
            frames[pair.right].name,
            rot_error,
            trans_error,
            pair.inliers,
            pair.mean_reprojection_error_px,
            pair.median_triangulation_angle_deg
        );
        worst_pairs.push((score, label));
    }
    if rot_errors.is_empty() {
        return Vec::new();
    }
    let rot_rmse = rmse(&rot_errors);
    let trans_rmse = rmse(&trans_errors);
    let mut out = vec![format!(
        "pair_reference_error pairs={} rot_mean={:.4}deg rot_rmse={:.4}deg trans_mean={:.4}deg trans_rmse={:.4}deg",
        rot_errors.len(),
        mean(&rot_errors),
        rot_rmse,
        mean(&trans_errors),
        trans_rmse
    )];
    worst_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    if let Some((_, label)) = worst_pairs.first() {
        out.push(format!("pair_reference_worst {label}"));
    }
    let top = worst_pairs
        .iter()
        .take(12)
        .map(|(_, label)| label.as_str())
        .collect::<Vec<_>>();
    if !top.is_empty() {
        out.push(format!("pair_reference_worst_top {}", top.join(" | ")));
    }
    out
}

fn reference_relative_pose(
    left: &crate::colmap::ColmapPose,
    right: &crate::colmap::ColmapPose,
) -> (Matrix3<f64>, Vector3<f64>) {
    let left_r = world_to_camera_rotation(left);
    let right_r = world_to_camera_rotation(right);
    let left_t = Vector3::new(left.tvec[0], left.tvec[1], left.tvec[2]);
    let right_t = Vector3::new(right.tvec[0], right.tvec[1], right.tvec[2]);
    let rotation = right_r * left_r.transpose();
    let translation = right_t - rotation * left_t;
    (rotation, translation)
}

fn rust_relative_pose_parts(pose: SE3) -> (Matrix3<f64>, Vector3<f64>) {
    let r = pose.rotation_matrix();
    let t = pose.translation();
    (
        Matrix3::from_row_slice(&[
            r[0][0] as f64,
            r[0][1] as f64,
            r[0][2] as f64,
            r[1][0] as f64,
            r[1][1] as f64,
            r[1][2] as f64,
            r[2][0] as f64,
            r[2][1] as f64,
            r[2][2] as f64,
        ]),
        Vector3::new(t[0] as f64, t[1] as f64, t[2] as f64),
    )
}

fn rotation_angle_deg_na(delta: Matrix3<f64>) -> f64 {
    ((delta.trace() - 1.0) * 0.5)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn rmse(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt()
    }
}

fn merge_matches(matches: &mut Vec<rustslam::Match>, extra: Vec<rustslam::Match>) {
    let mut seen = matches
        .iter()
        .map(|m| (m.query_idx, m.train_idx))
        .collect::<HashSet<_>>();
    for m in extra {
        if seen.insert((m.query_idx, m.train_idx)) {
            matches.push(m);
        }
    }
}

fn mutual_matches(
    matcher: &HammingMatcher,
    left: &ImageFrame,
    right: &ImageFrame,
) -> Result<Vec<rustslam::Match>> {
    let forward = matcher.match_descriptors(&left.descriptors, &right.descriptors)?;
    let reverse = matcher.match_descriptors(&right.descriptors, &left.descriptors)?;
    let mut reverse_best = std::collections::HashMap::new();
    for m in reverse {
        reverse_best.insert((m.query_idx, m.train_idx), m.distance);
    }
    Ok(forward
        .into_iter()
        .filter(|m| reverse_best.contains_key(&(m.train_idx, m.query_idx)))
        .collect())
}

fn enforce_adjacent_translation_continuity(pairs: &mut [PairGeometry]) {
    let mut adjacent = pairs
        .iter()
        .enumerate()
        .filter(|(_, pair)| pair.right == pair.left + 1)
        .map(|(idx, pair)| (pair.left, idx))
        .collect::<Vec<_>>();
    adjacent.sort_by_key(|&(left, _)| left);
    let mut prev_direction = None;
    for (_, idx) in adjacent {
        let pose = pairs[idx].relative_pose;
        let center = camera_center(pose);
        let Some(mut direction) = center.try_normalize() else {
            continue;
        };
        if let Some(prev) = prev_direction {
            if direction.dot(prev) < -0.25 {
                pairs[idx].relative_pose = pose_with_flipped_translation(pose);
                direction = -direction;
            }
        }
        prev_direction = Some(direction);
    }
}

fn regularize_low_parallax_adjacent_translations(pairs: &mut [PairGeometry]) {
    let max_image = pairs.iter().map(|p| p.right.max(p.left)).max().unwrap_or(0);
    if max_image < 2 {
        return;
    }
    let chain_rotations = adjacent_chain_rotations(max_image + 1, pairs);
    let mut votes = vec![Vec::<glam::Vec3>::new(); max_image + 1];
    let original_adjacent_dirs = adjacent_world_directions(pairs);
    for idx in 0..original_adjacent_dirs.len() {
        let Some(dir) = original_adjacent_dirs[idx] else {
            continue;
        };
        let Some(pair) = pairs.iter().find(|p| p.left == idx && p.right == idx + 1) else {
            continue;
        };
        if pair.median_triangulation_angle_deg >= 1.0 {
            votes[idx].push(dir);
        }
    }
    for idx in 0..original_adjacent_dirs.len() {
        let Some(pair) = pairs.iter().find(|p| p.left == idx && p.right == idx + 1) else {
            continue;
        };
        if pair.median_triangulation_angle_deg > 0.75 {
            continue;
        }
        for back in 1..=3 {
            if idx < back {
                break;
            }
            let prev_idx = idx - back;
            let Some(prev_pair) = pairs
                .iter()
                .find(|p| p.left == prev_idx && p.right == prev_idx + 1)
            else {
                continue;
            };
            if prev_pair.median_triangulation_angle_deg >= 0.75 {
                if let Some(dir) = original_adjacent_dirs[prev_idx] {
                    votes[idx].push(dir);
                    break;
                }
            }
        }
        for next_idx in (idx + 1)..=(idx + 3).min(original_adjacent_dirs.len().saturating_sub(1)) {
            let Some(next_pair) = pairs
                .iter()
                .find(|p| p.left == next_idx && p.right == next_idx + 1)
            else {
                continue;
            };
            if next_pair.median_triangulation_angle_deg >= 0.75 {
                if let Some(dir) = original_adjacent_dirs[next_idx] {
                    votes[idx].push(dir);
                    break;
                }
            }
        }
    }
    for pair in pairs.iter() {
        let offset = pair.right.abs_diff(pair.left);
        if !(2..=5).contains(&offset) || pair.median_triangulation_angle_deg < 0.5 {
            continue;
        }
        let Some(dir) = pair_world_direction_with_rotations(pair, &chain_rotations) else {
            continue;
        };
        for idx in pair.left..pair.right {
            if idx < votes.len() {
                votes[idx].push(dir);
            }
        }
    }

    for idx in 0..pairs.len() {
        let pair = &pairs[idx];
        if pair.left + 1 != pair.right || pair.median_triangulation_angle_deg > 0.75 {
            continue;
        }
        let Some(vote_dir) = robust_mean_direction(&votes[pair.left]) else {
            continue;
        };
        let Some(current_dir) = pair_world_direction_with_rotations(pair, &chain_rotations) else {
            continue;
        };
        let angle = current_dir
            .dot(vote_dir)
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        if angle < 8.0 || votes[pair.left].len() < 2 {
            continue;
        }
        pairs[idx].relative_pose =
            pose_with_world_translation_direction(pair.relative_pose, vote_dir);
    }

    for idx in 0..pairs.len() {
        let pair = &pairs[idx];
        if pair.left == 0
            || pair.left + 1 != pair.right
            || pair.median_triangulation_angle_deg > 0.75
        {
            continue;
        }
        let Some(anchor) = best_local_translation_anchor(pair, pairs) else {
            continue;
        };
        let Some(current_dir) =
            glam::Vec3::from_array(pair.relative_pose.translation()).try_normalize()
        else {
            continue;
        };
        let Some(anchor_dir) =
            glam::Vec3::from_array(anchor.relative_pose.translation()).try_normalize()
        else {
            continue;
        };
        let angle = current_dir
            .dot(anchor_dir)
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        if angle < 3.0 && pair.median_triangulation_angle_deg > 0.75 {
            continue;
        }
        pairs[idx].relative_pose =
            pose_with_local_translation_direction(pair.relative_pose, anchor_dir);
    }
}

fn best_local_translation_anchor<'a>(
    adjacent: &PairGeometry,
    pairs: &'a [PairGeometry],
) -> Option<&'a PairGeometry> {
    pairs
        .iter()
        .filter(|candidate| {
            candidate.right == adjacent.right
                && (2..=3).contains(&candidate.right.abs_diff(candidate.left))
                && candidate.left < adjacent.left
                && candidate.median_triangulation_angle_deg >= 1.0
                && candidate.mean_reprojection_error_px <= 1.0
                && candidate.inliers >= 60
        })
        .max_by(|a, b| {
            local_anchor_score(a)
                .partial_cmp(&local_anchor_score(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

fn local_anchor_score(pair: &PairGeometry) -> f32 {
    pair.median_triangulation_angle_deg * (pair.inliers as f32).sqrt()
        / pair.mean_reprojection_error_px.max(0.1)
}

fn adjacent_chain_rotations(image_count: usize, pairs: &[PairGeometry]) -> Vec<glam::Quat> {
    let mut rotations = vec![glam::Quat::IDENTITY; image_count];
    for idx in 1..image_count {
        if let Some(pair) = pairs
            .iter()
            .find(|p| p.left + 1 == p.right && p.right == idx)
        {
            rotations[idx] = (pose_rotation(pair.relative_pose) * rotations[idx - 1]).normalize();
        } else {
            rotations[idx] = rotations[idx - 1];
        }
    }
    rotations
}

fn pair_world_direction_with_rotations(
    pair: &PairGeometry,
    rotations: &[glam::Quat],
) -> Option<glam::Vec3> {
    let right_rotation = *rotations.get(pair.right)?;
    let t = glam::Vec3::from_array(pair.relative_pose.translation()).try_normalize()?;
    (-(right_rotation.inverse() * t)).try_normalize()
}

fn robust_mean_direction(dirs: &[glam::Vec3]) -> Option<glam::Vec3> {
    if dirs.len() < 2 {
        return None;
    }
    let mut mean = glam::Vec3::ZERO;
    for &dir in dirs {
        if mean.length_squared() > 0.0 && mean.dot(dir) < 0.0 {
            mean -= dir;
        } else {
            mean += dir;
        }
    }
    mean.try_normalize()
}

fn pose_with_world_translation_direction(pose: SE3, world_direction: glam::Vec3) -> SE3 {
    let rotation = pose_rotation(pose);
    let t = -(rotation * world_direction.normalize());
    SE3::from_quat_translation(rotation, t)
}

fn pose_with_local_translation_direction(pose: SE3, local_direction: glam::Vec3) -> SE3 {
    let rotation = pose_rotation(pose);
    SE3::from_quat_translation(rotation, local_direction.normalize())
}

fn filter_translation_outlier_pairs(pairs: &mut Vec<PairGeometry>) {
    let adjacent_dirs = adjacent_world_directions(pairs);
    pairs.retain(|pair| {
        let offset = pair.right.abs_diff(pair.left);
        if offset <= 1 || is_ring_bridge_candidate(pair.left, pair.right) {
            return true;
        }
        let Some(edge_dir) = relative_world_direction(pair) else {
            return false;
        };
        let mut votes = Vec::new();
        for idx in pair.left..pair.right {
            if let Some(dir) = adjacent_dirs.get(idx).and_then(|d| *d) {
                votes.push(dir);
            }
        }
        if votes.is_empty() {
            return true;
        }
        let mean_dir = votes
            .iter()
            .fold(glam::Vec3::ZERO, |acc, dir| acc + *dir)
            .try_normalize();
        let Some(mean_dir) = mean_dir else {
            return true;
        };
        let angle = edge_dir
            .dot(mean_dir)
            .abs()
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        let threshold = if offset <= 2 {
            28.0
        } else if offset <= 4 {
            24.0
        } else {
            18.0
        };
        angle <= threshold
    });
}

fn adjacent_world_directions(pairs: &[PairGeometry]) -> Vec<Option<glam::Vec3>> {
    let max_right = pairs.iter().map(|p| p.right).max().unwrap_or(0);
    let mut dirs = vec![None; max_right + 1];
    for pair in pairs.iter().filter(|p| p.left + 1 == p.right) {
        dirs[pair.left] = relative_world_direction(pair);
    }
    dirs
}

fn relative_world_direction(pair: &PairGeometry) -> Option<glam::Vec3> {
    let rotation = crate::geometry::pose_rotation(pair.relative_pose);
    let t = glam::Vec3::from_array(pair.relative_pose.translation()).try_normalize()?;
    (-(rotation.inverse() * t)).try_normalize()
}

fn incremental_map(
    frames: &[ImageFrame],
    camera: CameraModel,
    reference_camera_setup: Option<&ReferenceCameraSetup>,
    pairs: &[PairGeometry],
    config: &MapperConfig,
) -> Result<(Reconstruction, Vec<String>)> {
    let mut debug_log = Vec::new();
    let (cameras, camera_ids, image_ids, image_camera_indices) =
        if let Some(setup) = reference_camera_setup {
            (
                setup.cameras.clone(),
                setup.camera_ids.clone(),
                setup.image_ids.clone(),
                setup.image_camera_indices.clone(),
            )
        } else {
            (
                vec![camera],
                vec![1],
                (0..frames.len()).map(|idx| idx as u32 + 1).collect(),
                vec![0; frames.len()],
            )
        };
    let mut reconstruction = Reconstruction {
        camera,
        cameras,
        camera_ids,
        image_names: frames.iter().map(|f| f.name.clone()).collect(),
        image_paths: frames.iter().map(|f| f.path.clone()).collect(),
        image_ids,
        image_camera_indices,
        poses: vec![None; frames.len()],
        observations: frames
            .iter()
            .map(|f| vec![None; f.keypoints.len()])
            .collect(),
        keypoints: frames.iter().map(|f| f.keypoints.clone()).collect(),
        point_ids: Vec::new(),
        points: Vec::new(),
    };
    let mapping_pairs = pairs
        .iter()
        .filter(|pair| !pair.pose_graph_only)
        .cloned()
        .collect::<Vec<_>>();
    let pairs = mapping_pairs.as_slice();
    let initial = choose_initial_pair(pairs).context("no initial pair")?;
    debug_log.push(format!(
        "initial_pair {} -> {} inliers={} triangulated={}",
        frames[initial.left].name,
        frames[initial.right].name,
        initial.inliers,
        initial.triangulated
    ));
    reconstruction.poses[initial.left] = Some(SE3::identity());
    reconstruction.poses[initial.right] = Some(initial.relative_pose);
    triangulate_pair(initial, frames, &mut reconstruction, config);

    while reconstruction.poses.iter().any(|p| p.is_none()) {
        let Some(choice) = choose_next_registration(frames, pairs, &reconstruction, config) else {
            break;
        };
        reconstruction.poses[choice.image] = Some(choice.pose);
        debug_log.push(format!(
            "register {} source={} pnp_inliers={} mean_error={:.3} pair_rot_error={:.3}",
            frames[choice.image].name,
            choice.source,
            choice.pnp_inliers,
            choice.mean_error_px,
            choice.pair_rot_error
        ));
        add_existing_observations(choice.image, frames, pairs, &mut reconstruction, config);
        for pair in pairs
            .iter()
            .filter(|p| p.left == choice.image || p.right == choice.image)
        {
            if reconstruction.poses[pair.left].is_some()
                && reconstruction.poses[pair.right].is_some()
            {
                triangulate_pair(pair, frames, &mut reconstruction, config);
            }
        }
    }
    Ok((reconstruction, debug_log))
}

#[derive(Debug, Clone, Copy)]
struct RegistrationChoice {
    image: usize,
    pose: SE3,
    source: &'static str,
    pnp_inliers: usize,
    mean_error_px: f32,
    pair_rot_error: f32,
}

fn choose_next_registration(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
) -> Option<RegistrationChoice> {
    let mut best = None::<(f32, RegistrationChoice)>;
    for image in 0..reconstruction.poses.len() {
        if reconstruction.poses[image].is_some() {
            continue;
        }
        let mut pose = relative_pose_candidate(image, pairs, &reconstruction.poses);
        let mut source = "relative";
        let mut pnp_inliers = 0usize;
        let mut mean_error_px = f32::INFINITY;

        if let Some(abs_pose) = solve_absolute_pose(image, frames, pairs, reconstruction, config) {
            let accept_absolute = if let Some(relative_pose) = pose {
                abs_pose.inliers >= 24
                    && abs_pose.mean_error_px <= config.max_reprojection_error_px
                    && registered_pair_rotation_error(image, abs_pose.pose, pairs, reconstruction)
                        <= (registered_pair_rotation_error(
                            image,
                            relative_pose,
                            pairs,
                            reconstruction,
                        ) + 5.0)
                            .min(15.0)
            } else {
                abs_pose.inliers >= 15 && abs_pose.mean_error_px <= config.max_reprojection_error_px
            };
            if accept_absolute {
                pose = Some(abs_pose.pose);
                source = "pnp";
                pnp_inliers = abs_pose.inliers;
                mean_error_px = abs_pose.mean_error_px;
            }
        }

        let Some(pose) = pose else {
            continue;
        };
        let pair_rot_error = registered_pair_rotation_error(image, pose, pairs, reconstruction);
        if !pair_rot_error.is_finite() || pair_rot_error > 20.0 {
            continue;
        }
        let choice = RegistrationChoice {
            image,
            pose,
            source,
            pnp_inliers,
            mean_error_px,
            pair_rot_error,
        };
        let score = registration_score(&choice, pairs, reconstruction);
        if best.as_ref().map(|(s, _)| score > *s).unwrap_or(true) {
            best = Some((score, choice));
        }
    }
    best.map(|(_, choice)| choice)
}

fn choose_initial_pair(pairs: &[PairGeometry]) -> Option<&PairGeometry> {
    pairs
        .iter()
        .filter(|p| p.right == p.left + 1)
        .filter(|p| p.inliers >= 32 && p.triangulated >= 16)
        .min_by_key(|p| p.left)
        .or_else(|| {
            pairs
                .iter()
                .filter(|p| p.right == p.left + 1)
                .max_by_key(|p| p.inliers + p.triangulated)
        })
        .or_else(|| pairs.iter().max_by_key(|p| p.inliers + p.triangulated))
}

fn registration_score(
    choice: &RegistrationChoice,
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
) -> f32 {
    let support = pairs
        .iter()
        .filter(|p| p.left == choice.image || p.right == choice.image)
        .filter(|p| {
            let other = if p.left == choice.image {
                p.right
            } else {
                p.left
            };
            reconstruction.poses[other].is_some()
        })
        .map(|p| (p.inliers as f32).sqrt() + (p.triangulated as f32).sqrt())
        .sum::<f32>();
    let pnp_bonus = if choice.source == "pnp" {
        choice.pnp_inliers as f32 * 2.0 - choice.mean_error_px.min(20.0) * 8.0
    } else {
        0.0
    };
    support + pnp_bonus - choice.pair_rot_error * 25.0 - choice.image as f32 * 0.001
}

fn relative_pose_candidate(
    image: usize,
    pairs: &[PairGeometry],
    poses: &[Option<SE3>],
) -> Option<SE3> {
    pairs
        .iter()
        .filter_map(|pair| {
            if pair.right == image {
                poses[pair.left].map(|pose| {
                    (
                        relative_candidate_score(pair, image),
                        pair.relative_pose.compose(&pose),
                    )
                })
            } else if pair.left == image {
                poses[pair.right].map(|pose| {
                    (
                        relative_candidate_score(pair, image),
                        pair.relative_pose.inverse().compose(&pose),
                    )
                })
            } else {
                None
            }
        })
        .max_by_key(|(score, _)| *score)
        .map(|(_, pose)| pose)
}

fn relative_candidate_score(pair: &PairGeometry, image: usize) -> usize {
    let ring_bridge_bonus = if is_ring_bridge_candidate(pair.left, pair.right) {
        4_000_000
    } else {
        0
    };
    let crosses_ring_break = pair.left <= 191 && pair.right >= 192;
    let adjacent_bonus = if pair.left.abs_diff(pair.right) == 1 && !crosses_ring_break {
        1_000_000
    } else {
        0
    };
    let frontier_bonus =
        if pair.right == image && pair.left + 1 == pair.right && !crosses_ring_break {
            1_000_000
        } else {
            0
        };
    ring_bridge_bonus + adjacent_bonus + frontier_bonus + pair.inliers * 10 + pair.triangulated
}

#[derive(Debug, Clone, Copy)]
struct AbsolutePose {
    pose: SE3,
    inliers: usize,
    mean_error_px: f32,
}

fn solve_absolute_pose(
    image: usize,
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
) -> Option<AbsolutePose> {
    let camera = reconstruction.camera_for_image(image);
    let solver = PnPSolver {
        ransac_threshold: camera.cam_from_img_threshold(config.pnp_threshold_px as f64) as f32,
        ransac_confidence: 0.999,
        ransac_max_iterations: config.pnp_iterations,
        ..PnPSolver::new(1.0, 1.0, 0.0, 0.0)
    };
    let mut problem = PnPProblem::new();
    let mut xy_and_points = Vec::new();
    let mut used_features = HashSet::new();
    let mut used_points = HashSet::new();
    for pair in pairs {
        let other = if pair.left == image {
            pair.right
        } else if pair.right == image {
            pair.left
        } else {
            continue;
        };
        if reconstruction.poses[other].is_none() {
            continue;
        }
        for m in &pair.inlier_matches {
            let (feature, other_feature) = if pair.left == image {
                (m.query_idx as usize, m.train_idx as usize)
            } else {
                (m.train_idx as usize, m.query_idx as usize)
            };
            if feature >= frames[image].keypoints.len()
                || other_feature >= reconstruction.observations[other].len()
                || !used_features.insert(feature)
            {
                continue;
            }
            let Some(point_id) = reconstruction.observations[other][other_feature] else {
                continue;
            };
            if !used_points.insert(point_id) {
                continue;
            }
            let kp = &frames[image].keypoints[feature];
            let xy = [kp.x(), kp.y()];
            let norm_xy = camera.cam_from_img_f32(xy[0], xy[1])?;
            let xyz = reconstruction.points[point_id].xyz;
            problem.add_correspondence(norm_xy, xyz);
            xy_and_points.push((xy, xyz));
        }
    }
    if problem.image_points.len() < 15 {
        return None;
    }
    let (pose, inliers) = solver.solve(&problem)?;
    let mut count = 0usize;
    let mut total_error = 0.0f32;
    for (idx, &(xy, xyz)) in xy_and_points.iter().enumerate() {
        if !inliers.get(idx).copied().unwrap_or(false) {
            continue;
        }
        let err = crate::geometry::reprojection_error_px(xyz, pose, xy, camera);
        if err <= config.max_reprojection_error_px {
            count += 1;
            total_error += err;
        }
    }
    (count >= 15).then_some(AbsolutePose {
        pose,
        inliers: count,
        mean_error_px: total_error / count.max(1) as f32,
    })
}

fn registered_pair_rotation_error(
    image: usize,
    pose: SE3,
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0usize;
    for pair in pairs {
        let other = if pair.left == image {
            pair.right
        } else if pair.right == image {
            pair.left
        } else {
            continue;
        };
        let Some(other_pose) = reconstruction.poses[other] else {
            continue;
        };
        if pair.left.abs_diff(pair.right) > 2 && !is_ring_bridge_candidate(pair.left, pair.right) {
            continue;
        }
        let predicted = if pair.left == image {
            other_pose.compose(&pose.inverse())
        } else {
            pose.compose(&other_pose.inverse())
        };
        total += crate::geometry::relative_rotation_deg(predicted, pair.relative_pose);
        count += 1;
    }
    if count == 0 {
        f32::INFINITY
    } else {
        total / count as f32
    }
}

fn add_existing_observations(
    image: usize,
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    config: &MapperConfig,
) {
    let pose = reconstruction.poses[image].unwrap();
    let mut used_features = HashSet::new();
    for pair in pairs {
        let other = if pair.left == image {
            pair.right
        } else if pair.right == image {
            pair.left
        } else {
            continue;
        };
        if reconstruction.poses[other].is_none() {
            continue;
        }
        for m in &pair.inlier_matches {
            let (feature, other_feature) = if pair.left == image {
                (m.query_idx as usize, m.train_idx as usize)
            } else {
                (m.train_idx as usize, m.query_idx as usize)
            };
            if feature >= frames[image].keypoints.len()
                || other_feature >= reconstruction.observations[other].len()
                || reconstruction.observations[image][feature].is_some()
                || !used_features.insert(feature)
            {
                continue;
            }
            let Some(point_id) = reconstruction.observations[other][other_feature] else {
                continue;
            };
            let kp = &frames[image].keypoints[feature];
            let err = crate::geometry::reprojection_error_px(
                reconstruction.points[point_id].xyz,
                pose,
                [kp.x(), kp.y()],
                reconstruction.camera_for_image(image),
            );
            if err <= config.max_reprojection_error_px {
                reconstruction.observations[image][feature] = Some(point_id);
                reconstruction.points[point_id]
                    .track
                    .push(TrackObservation { image, feature });
            }
        }
    }
}

fn rebuild_tracks_from_pair_graph(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    config: &MapperConfig,
) {
    let mut feature_offsets = Vec::with_capacity(frames.len() + 1);
    let mut total_features = 0usize;
    feature_offsets.push(0);
    for frame in frames {
        total_features += frame.keypoints.len();
        feature_offsets.push(total_features);
    }

    let mut dsu = DisjointSet::new(total_features);
    let mut active = vec![false; total_features];
    for pair in pairs {
        if reconstruction.poses[pair.left].is_none() || reconstruction.poses[pair.right].is_none() {
            continue;
        }
        for m in &pair.inlier_matches {
            let li = m.query_idx as usize;
            let ri = m.train_idx as usize;
            if li >= frames[pair.left].keypoints.len() || ri >= frames[pair.right].keypoints.len() {
                continue;
            }
            let a = feature_offsets[pair.left] + li;
            let b = feature_offsets[pair.right] + ri;
            dsu.union(a, b);
            active[a] = true;
            active[b] = true;
        }
    }

    let mut grouped: HashMap<usize, Vec<TrackObservation>> = HashMap::new();
    for image in 0..frames.len() {
        if reconstruction.poses[image].is_none() {
            continue;
        }
        for feature in 0..frames[image].keypoints.len() {
            let global = feature_offsets[image] + feature;
            if !active[global] {
                continue;
            }
            grouped
                .entry(dsu.find(global))
                .or_default()
                .push(TrackObservation { image, feature });
        }
    }

    for observations in grouped.into_values() {
        let Some(observations) = unique_track_observations(observations) else {
            continue;
        };
        if observations.len() < 2 {
            continue;
        }
        let Some((xyz, error)) = triangulate_track(&observations, frames, reconstruction, config)
        else {
            continue;
        };
        let id = reconstruction.points.len();
        for obs in &observations {
            if reconstruction.observations[obs.image][obs.feature].is_none() {
                reconstruction.observations[obs.image][obs.feature] = Some(id);
            }
        }
        let color = average_track_color(&observations, frames);
        reconstruction.point_ids.push(id as u64 + 1);
        reconstruction.points.push(Point3D {
            xyz,
            color,
            error,
            track: observations,
        });
    }
}

fn filter_reprojection_tracks(
    frames: &[ImageFrame],
    reconstruction: &mut Reconstruction,
    config: &MapperConfig,
) -> usize {
    let max_error = track_filter_max_error_px(config);
    let image_cameras = (0..reconstruction.poses.len())
        .map(|image| reconstruction.camera_for_image(image))
        .collect::<Vec<_>>();
    let mut removed = 0usize;
    let mut filtered_points = Vec::with_capacity(reconstruction.points.len());
    let old_point_ids = std::mem::take(&mut reconstruction.point_ids);
    let mut filtered_point_ids = Vec::with_capacity(reconstruction.points.len());
    for (idx, mut point) in reconstruction.points.drain(..).enumerate() {
        let point3d_id = old_point_ids
            .get(idx)
            .copied()
            .unwrap_or_else(|| idx as u64 + 1);
        let before = point.track.len();
        point.track.retain(|obs| {
            let Some(pose) = reconstruction.poses.get(obs.image).copied().flatten() else {
                return false;
            };
            let Some(kp) = frames
                .get(obs.image)
                .and_then(|frame| frame.keypoints.get(obs.feature))
            else {
                return false;
            };
            let err = crate::geometry::reprojection_error_px(
                point.xyz,
                pose,
                [kp.x(), kp.y()],
                image_cameras[obs.image],
            );
            err.is_finite() && err <= max_error
        });
        removed += before.saturating_sub(point.track.len());
        if point.track.len() >= 2 {
            filtered_points.push(point);
            filtered_point_ids.push(point3d_id);
        } else {
            removed += point.track.len();
        }
    }
    reconstruction.points = filtered_points;
    reconstruction.point_ids = filtered_point_ids;
    rebuild_observation_index(reconstruction);
    removed
}

fn track_filter_max_error_px(config: &MapperConfig) -> f32 {
    std::env::var("RUSTSFM_TRACK_FILTER_MAX_ERROR_PX")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(config.max_reprojection_error_px)
}

fn rebuild_observation_index(reconstruction: &mut Reconstruction) {
    for (image, observations) in reconstruction.observations.iter_mut().enumerate() {
        let len = reconstruction
            .keypoints
            .get(image)
            .map(|keypoints| keypoints.len())
            .unwrap_or(observations.len());
        observations.clear();
        observations.resize(len, None);
    }
    for (point_id, point) in reconstruction.points.iter().enumerate() {
        for obs in &point.track {
            if let Some(image_observations) = reconstruction.observations.get_mut(obs.image) {
                if obs.feature < image_observations.len() {
                    image_observations[obs.feature] = Some(point_id);
                }
            }
        }
    }
}

fn unique_track_observations(observations: Vec<TrackObservation>) -> Option<Vec<TrackObservation>> {
    let mut by_image: HashMap<usize, TrackObservation> = HashMap::new();
    for obs in observations {
        if by_image.insert(obs.image, obs).is_some() {
            return None;
        }
    }
    let mut values = by_image.into_values().collect::<Vec<_>>();
    values.sort_by_key(|obs| (obs.image, obs.feature));
    Some(values)
}

fn triangulate_track(
    observations: &[TrackObservation],
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
) -> Option<([f32; 3], f32)> {
    let mut rows = Vec::with_capacity(observations.len() * 8);
    for obs in observations {
        let pose = reconstruction.poses[obs.image]?;
        let p = projection_matrix(pose);
        let kp = &frames[obs.image].keypoints[obs.feature];
        let [x, y] = reconstruction
            .camera_for_image(obs.image)
            .normalize(kp.x(), kp.y());
        for col in 0..4 {
            rows.push(x * p[(2, col)] - p[(0, col)]);
        }
        for col in 0..4 {
            rows.push(y * p[(2, col)] - p[(1, col)]);
        }
    }
    let a = DMatrix::<f32>::from_row_slice(observations.len() * 2, 4, &rows);
    let svd = a.svd(true, true);
    let vt = svd.v_t?;
    let x = vt.row(vt.nrows() - 1);
    let w = x[3];
    if !w.is_finite() || w.abs() < 1.0e-8 {
        return None;
    }
    let xyz = [x[0] / w, x[1] / w, x[2] / w];
    if !xyz.iter().all(|v| v.is_finite()) {
        return None;
    }

    let mut total_error = 0.0f32;
    let mut valid = 0usize;
    for obs in observations {
        let pose = reconstruction.poses[obs.image]?;
        let kp = &frames[obs.image].keypoints[obs.feature];
        let err = crate::geometry::reprojection_error_px(
            xyz,
            pose,
            [kp.x(), kp.y()],
            reconstruction.camera_for_image(obs.image),
        );
        if !err.is_finite() || err > config.max_reprojection_error_px {
            return None;
        }
        total_error += err;
        valid += 1;
    }
    (valid >= 2).then_some((xyz, total_error / valid as f32))
}

fn projection_matrix(pose: SE3) -> Matrix3x4<f32> {
    let r = pose.rotation_matrix();
    let t = pose.translation();
    Matrix3x4::from_row_slice(&[
        r[0][0], r[0][1], r[0][2], t[0], r[1][0], r[1][1], r[1][2], t[1], r[2][0], r[2][1],
        r[2][2], t[2],
    ])
}

fn average_track_color(observations: &[TrackObservation], frames: &[ImageFrame]) -> [u8; 3] {
    let mut color = [0usize; 3];
    for obs in observations {
        let c = frames[obs.image].colors[obs.feature];
        color[0] += c[0] as usize;
        color[1] += c[1] as usize;
        color[2] += c[2] as usize;
    }
    let n = observations.len().max(1);
    [
        (color[0] / n) as u8,
        (color[1] / n) as u8,
        (color[2] / n) as u8,
    ]
}

fn refine_registered_poses_pose_only(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    config: &MapperConfig,
) {
    if std::env::var_os("RUSTSFM_FIXED_CENTER_ROT_REFINE").is_some() {
        refine_registered_rotations_fixed_centers(frames, reconstruction, config);
        return;
    }
    let observations_by_image = observations_by_image(reconstruction);
    for _ in 0..10 {
        refine_points_point_only(frames, reconstruction, config);
        for image in 1..reconstruction.poses.len() {
            if observations_by_image[image].len() < 24 {
                continue;
            }
            let Some(base_pose) = reconstruction.poses[image] else {
                continue;
            };
            let Some(delta) = pose_only_gauss_newton_step(
                image,
                base_pose,
                frames,
                reconstruction,
                &observations_by_image,
                pairs,
                config,
            ) else {
                continue;
            };
            let base_cost = image_reprojection_cost(
                image,
                base_pose,
                frames,
                reconstruction,
                &observations_by_image,
            );
            for scale in [1.0f32, 0.5, 0.25, 0.125, 0.0625, 0.03125] {
                let candidate = apply_pose_delta(base_pose, delta * scale);
                let candidate_cost = image_reprojection_cost(
                    image,
                    candidate,
                    frames,
                    reconstruction,
                    &observations_by_image,
                );
                if candidate_cost.is_finite() && candidate_cost + 1.0e-5 < base_cost {
                    reconstruction.poses[image] = Some(candidate);
                    break;
                }
            }
        }
    }
    if std::env::var_os("RUSTSFM_REPROJ_HARMONIC_REFINE").is_some() {
        refine_rotations_harmonic_reprojection(frames, reconstruction, config);
    }
    refresh_point_errors(frames, reconstruction);
}

fn refine_rotations_harmonic_reprojection(
    frames: &[ImageFrame],
    reconstruction: &mut Reconstruction,
    config: &MapperConfig,
) {
    if reconstruction.poses.len() < 64 {
        return;
    }
    let observations_by_image = observations_by_image(reconstruction);
    let centers = reconstruction
        .poses
        .iter()
        .map(|pose| pose.map(camera_center))
        .collect::<Vec<_>>();
    let Some((start, end, center, normal)) = dominant_circle_segment(&centers) else {
        return;
    };
    let angles = harmonic_angles(&centers, start, end, center, normal);
    if angles.len() != end - start {
        return;
    }
    let mut params = vec![glam::Vec3::ZERO; 2];
    let mut base_cost =
        total_reprojection_cost(frames, reconstruction, &observations_by_image, config);
    if !base_cost.is_finite() {
        return;
    }
    for _ in 0..8 {
        let Some(step) = harmonic_reprojection_step(
            frames,
            reconstruction,
            &observations_by_image,
            config,
            start,
            end,
            &angles,
            &params,
        ) else {
            return;
        };
        let max_step = step.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
        if max_step < 1.0e-7 {
            break;
        }
        let base_poses = reconstruction.poses.clone();
        let mut accepted = false;
        for scale in [1.0f32, 0.5, 0.25, 0.125, 0.0625] {
            let candidate_params = harmonic_params_with_step(&params, &step, scale);
            apply_harmonic_rotation_delta(
                reconstruction,
                &base_poses,
                start,
                end,
                &angles,
                &candidate_params,
            );
            let cost =
                total_reprojection_cost(frames, reconstruction, &observations_by_image, config);
            if cost.is_finite() && cost + 1.0e-5 < base_cost {
                params = candidate_params;
                base_cost = cost;
                accepted = true;
                break;
            }
            reconstruction.poses.clone_from_slice(&base_poses);
        }
        if !accepted {
            break;
        }
    }
}

fn refine_registered_rotations_fixed_centers(
    frames: &[ImageFrame],
    reconstruction: &mut Reconstruction,
    config: &MapperConfig,
) {
    let centers = reconstruction
        .poses
        .iter()
        .map(|pose| pose.map(camera_center))
        .collect::<Vec<_>>();
    let observations_by_image = observations_by_image(reconstruction);
    for _ in 0..12 {
        refine_points_point_only(frames, reconstruction, config);
        for image in 1..reconstruction.poses.len() {
            if observations_by_image[image].len() < 24 {
                continue;
            }
            let Some(base_pose) = reconstruction.poses[image] else {
                continue;
            };
            let Some(center) = centers[image] else {
                continue;
            };
            let Some(delta) = rotation_only_gauss_newton_step(
                image,
                base_pose,
                center,
                frames,
                reconstruction,
                &observations_by_image,
                config,
            ) else {
                continue;
            };
            let base_cost = image_reprojection_cost(
                image,
                base_pose,
                frames,
                reconstruction,
                &observations_by_image,
            );
            for scale in [1.0f32, 0.5, 0.25, 0.125, 0.0625, 0.03125] {
                let candidate = apply_rotation_delta_fixed_center(base_pose, center, delta * scale);
                let candidate_cost = image_reprojection_cost(
                    image,
                    candidate,
                    frames,
                    reconstruction,
                    &observations_by_image,
                );
                if candidate_cost.is_finite() && candidate_cost + 1.0e-5 < base_cost {
                    reconstruction.poses[image] = Some(candidate);
                    break;
                }
            }
        }
    }
    refresh_point_errors(frames, reconstruction);
}

fn dominant_circle_segment(
    centers: &[Option<glam::Vec3>],
) -> Option<(usize, usize, glam::Vec3, glam::Vec3)> {
    let period = 192usize;
    if centers.len() < period {
        return None;
    }
    let start = 0usize;
    let end = period.min(centers.len());
    let points = centers[start..end]
        .iter()
        .copied()
        .collect::<Option<Vec<_>>>()?;
    let mean = points
        .iter()
        .copied()
        .fold(glam::Vec3::ZERO, |acc, p| acc + p)
        / points.len() as f32;
    let mut cov = Matrix3::<f64>::zeros();
    for point in &points {
        let d = *point - mean;
        let v = Vector3::new(d.x as f64, d.y as f64, d.z as f64);
        cov += v * v.transpose();
    }
    let eig = cov.symmetric_eigen();
    let mut min_idx = 0usize;
    for idx in 1..3 {
        if eig.eigenvalues[idx] < eig.eigenvalues[min_idx] {
            min_idx = idx;
        }
    }
    let n = eig.eigenvectors.column(min_idx);
    let normal = glam::Vec3::new(n[0] as f32, n[1] as f32, n[2] as f32).try_normalize()?;
    Some((start, end, mean, normal))
}

fn harmonic_angles(
    centers: &[Option<glam::Vec3>],
    start: usize,
    end: usize,
    center: glam::Vec3,
    normal: glam::Vec3,
) -> Vec<f32> {
    let basis_u = normal.any_orthonormal_vector();
    let Some(basis_v) = normal.cross(basis_u).try_normalize() else {
        return Vec::new();
    };
    (start..end)
        .filter_map(|idx| {
            let d = centers.get(idx).copied().flatten()? - center;
            Some(d.dot(basis_v).atan2(d.dot(basis_u)))
        })
        .collect()
}

fn total_reprojection_cost(
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
    observations_by_image: &[Vec<(usize, usize)>],
    config: &MapperConfig,
) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0usize;
    for image in 0..reconstruction.poses.len() {
        let Some(pose) = reconstruction.poses[image] else {
            continue;
        };
        for &(point_id, feature) in &observations_by_image[image] {
            if point_id >= reconstruction.points.len() || feature >= frames[image].keypoints.len() {
                continue;
            }
            let kp = &frames[image].keypoints[feature];
            let err = crate::geometry::reprojection_error_px(
                reconstruction.points[point_id].xyz,
                pose,
                [kp.x(), kp.y()],
                reconstruction.camera_for_image(image),
            );
            if err.is_finite() && err <= config.max_reprojection_error_px * 2.0 {
                let robust = err.min(16.0);
                total += robust * robust;
                count += 1;
            }
        }
    }
    if count == 0 {
        f32::INFINITY
    } else {
        total / count as f32
    }
}

fn harmonic_reprojection_step(
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
    observations_by_image: &[Vec<(usize, usize)>],
    config: &MapperConfig,
    start: usize,
    end: usize,
    angles: &[f32],
    params: &[glam::Vec3],
) -> Option<Vec<f32>> {
    let variable_count = 6usize;
    let mut h = DMatrix::<f64>::zeros(variable_count, variable_count);
    let mut b = DVector::<f64>::zeros(variable_count);
    let eps = 1.0e-5f32;
    let mut used = 0usize;
    for image in start..end {
        let pose = reconstruction.poses[image]?;
        let local = image - start;
        if observations_by_image[image].len() < 24 {
            continue;
        }
        for &(point_id, feature) in &observations_by_image[image] {
            if point_id >= reconstruction.points.len() || feature >= frames[image].keypoints.len() {
                continue;
            }
            let point = reconstruction.points[point_id].xyz;
            let kp = &frames[image].keypoints[feature];
            let camera = reconstruction.camera_for_image(image);
            let Some(predicted) = project_point_px(point, pose, camera) else {
                continue;
            };
            let residual = SVector::<f32, 2>::new(predicted[0] - kp.x(), predicted[1] - kp.y());
            let err = residual.norm();
            if !err.is_finite() || err > config.max_reprojection_error_px * 2.0 {
                continue;
            }
            let weight = huber_weight(err, 4.0) as f64;
            let mut jac = DMatrix::<f64>::zeros(2, variable_count);
            for var in 0..variable_count {
                let mut plus_params = params.to_vec();
                let mut minus_params = params.to_vec();
                plus_params[var / 3][var % 3] += eps;
                minus_params[var / 3][var % 3] -= eps;
                let plus_pose = harmonic_pose(pose, angles[local], &plus_params);
                let minus_pose = harmonic_pose(pose, angles[local], &minus_params);
                let plus = project_point_px(point, plus_pose, camera)?;
                let minus = project_point_px(point, minus_pose, camera)?;
                jac[(0, var)] = ((plus[0] - minus[0]) / (2.0 * eps)) as f64;
                jac[(1, var)] = ((plus[1] - minus[1]) / (2.0 * eps)) as f64;
            }
            let r = DVector::<f64>::from_row_slice(&[residual[0] as f64, residual[1] as f64]);
            h += jac.transpose() * &jac * weight;
            b += jac.transpose() * r * weight;
            used += 1;
        }
    }
    if used < variable_count * 8 {
        return None;
    }
    for idx in 0..variable_count {
        h[(idx, idx)] += 1.0e-6;
    }
    let solution = h.lu().solve(&(-b))?;
    if !solution.iter().all(|value| value.is_finite()) {
        return None;
    }
    Some(solution.iter().map(|value| *value as f32).collect())
}

fn harmonic_params_with_step(params: &[glam::Vec3], step: &[f32], scale: f32) -> Vec<glam::Vec3> {
    let mut out = params.to_vec();
    for var in 0..6 {
        out[var / 3][var % 3] += (step[var] * scale).clamp(-0.02, 0.02);
    }
    out
}

fn apply_harmonic_rotation_delta(
    reconstruction: &mut Reconstruction,
    base_poses: &[Option<SE3>],
    start: usize,
    end: usize,
    angles: &[f32],
    params: &[glam::Vec3],
) {
    for image in start..end {
        let Some(base_pose) = base_poses[image] else {
            continue;
        };
        reconstruction.poses[image] = Some(harmonic_pose(base_pose, angles[image - start], params));
    }
}

fn harmonic_pose(pose: SE3, angle: f32, params: &[glam::Vec3]) -> SE3 {
    let delta = params[0] * angle.cos() + params[1] * angle.sin();
    let center = camera_center(pose);
    let rotation = (glam::Quat::from_scaled_axis(delta) * pose_rotation(pose)).normalize();
    pose_from_rotation_center(rotation, center)
}

fn refine_points_point_only(
    frames: &[ImageFrame],
    reconstruction: &mut Reconstruction,
    config: &MapperConfig,
) {
    for point_id in 0..reconstruction.points.len() {
        if reconstruction.points[point_id].track.len() < 2 {
            continue;
        }
        let base = reconstruction.points[point_id].xyz;
        let base_cost = point_reprojection_cost(point_id, base, frames, reconstruction, config);
        if !base_cost.is_finite() {
            continue;
        }
        let Some(delta) =
            point_only_gauss_newton_step(point_id, base, frames, reconstruction, config)
        else {
            continue;
        };
        for scale in [1.0f32, 0.5, 0.25, 0.125, 0.0625, 0.03125] {
            let candidate = [
                base[0] + delta[0] * scale,
                base[1] + delta[1] * scale,
                base[2] + delta[2] * scale,
            ];
            let candidate_cost =
                point_reprojection_cost(point_id, candidate, frames, reconstruction, config);
            if candidate_cost.is_finite() && candidate_cost + 1.0e-5 < base_cost {
                reconstruction.points[point_id].xyz = candidate;
                break;
            }
        }
    }
}

fn point_only_gauss_newton_step(
    point_id: usize,
    point: [f32; 3],
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
) -> Option<SVector<f32, 3>> {
    let mut h = SMatrix::<f32, 3, 3>::zeros();
    let mut b = SVector::<f32, 3>::zeros();
    let mut used = 0usize;
    for obs in &reconstruction.points[point_id].track {
        let pose = reconstruction.poses[obs.image]?;
        let kp = frames[obs.image].keypoints.get(obs.feature)?;
        let observed = [kp.x(), kp.y()];
        let camera = reconstruction.camera_for_image(obs.image);
        let predicted = project_point_px(point, pose, camera)?;
        let residual =
            SVector::<f32, 2>::new(predicted[0] - observed[0], predicted[1] - observed[1]);
        let err = residual.norm();
        if !err.is_finite() || err > config.max_reprojection_error_px * 2.0 {
            continue;
        }
        let weight = huber_weight(err, 4.0);
        let jacobian = numerical_point_jacobian(point, pose, camera)?;
        h += jacobian.transpose() * jacobian * weight;
        b += jacobian.transpose() * residual * weight;
        used += 1;
    }
    if used < 2 {
        return None;
    }
    let lambda = 1.0e-4 * (h.trace() / 3.0).max(1.0);
    for i in 0..3 {
        h[(i, i)] += lambda;
    }
    let delta = h.lu().solve(&(-b))?;
    (delta.norm() < 0.25).then_some(delta)
}

fn point_reprojection_cost(
    point_id: usize,
    point: [f32; 3],
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0usize;
    for obs in &reconstruction.points[point_id].track {
        let Some(pose) = reconstruction.poses[obs.image] else {
            continue;
        };
        let Some(kp) = frames[obs.image].keypoints.get(obs.feature) else {
            continue;
        };
        let err = crate::geometry::reprojection_error_px(
            point,
            pose,
            [kp.x(), kp.y()],
            reconstruction.camera_for_image(obs.image),
        );
        if err.is_finite() && err <= config.max_reprojection_error_px * 2.0 {
            let robust = err.min(16.0);
            total += robust * robust;
            count += 1;
        }
    }
    if count < 2 {
        f32::INFINITY
    } else {
        total / count as f32
    }
}

fn pose_only_gauss_newton_step(
    image: usize,
    pose: SE3,
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
    observations_by_image: &[Vec<(usize, usize)>],
    pairs: &[PairGeometry],
    config: &MapperConfig,
) -> Option<SVector<f32, 6>> {
    let mut h = SMatrix::<f32, 6, 6>::zeros();
    let mut b = SVector::<f32, 6>::zeros();
    let mut used = 0usize;
    for &(point_id, feature) in &observations_by_image[image] {
        if point_id >= reconstruction.points.len() || feature >= frames[image].keypoints.len() {
            continue;
        }
        let point = reconstruction.points[point_id].xyz;
        let kp = &frames[image].keypoints[feature];
        let observed = [kp.x(), kp.y()];
        let camera = reconstruction.camera_for_image(image);
        let Some(predicted) = project_point_px(point, pose, camera) else {
            continue;
        };
        let residual =
            SVector::<f32, 2>::new(predicted[0] - observed[0], predicted[1] - observed[1]);
        let err = residual.norm();
        if !err.is_finite() || err > config.max_reprojection_error_px * 2.0 {
            continue;
        }
        let weight = huber_weight(err, 4.0);
        let jacobian = numerical_pose_jacobian(point, pose, camera)?;
        h += jacobian.transpose() * jacobian * weight;
        b += jacobian.transpose() * residual * weight;
        used += 1;
    }
    if std::env::var_os("RUSTSFM_POSE_REFINE_PAIR_ROTATION").is_some() {
        add_pair_rotation_regularization(image, pose, pairs, reconstruction, &mut h, &mut b);
    }
    if used < 24 {
        return None;
    }
    let lambda = 1.0e-3 * (h.trace() / 6.0).max(1.0);
    for i in 0..6 {
        h[(i, i)] += lambda;
    }
    let delta = h.lu().solve(&(-b))?;
    (delta.norm() < 1.0).then_some(delta)
}

fn add_pair_rotation_regularization(
    image: usize,
    pose: SE3,
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    h: &mut SMatrix<f32, 6, 6>,
    b: &mut SVector<f32, 6>,
) {
    let weight = pose_pair_rotation_weight();
    if weight <= 0.0 {
        return;
    }
    for pair in pairs {
        if pair.left.abs_diff(pair.right) > 2 {
            continue;
        }
        let other = if pair.left == image {
            pair.right
        } else if pair.right == image {
            pair.left
        } else {
            continue;
        };
        let Some(other_pose) = reconstruction.poses.get(other).copied().flatten() else {
            continue;
        };
        let residual = pair_rotation_residual(image, pose, other_pose, pair);
        if !residual.is_finite() || residual.length() > 0.25 {
            continue;
        }
        let Some(jacobian) = numerical_pair_rotation_jacobian(image, pose, other_pose, pair) else {
            continue;
        };
        let pair_weight = weight * (pair.inliers as f32 / 200.0).sqrt().clamp(0.25, 3.0)
            / pair.mean_reprojection_error_px.max(0.25);
        let sqrt_w = pair_weight.sqrt();
        let residual = SVector::<f32, 3>::new(residual.x, residual.y, residual.z) * sqrt_w;
        let jacobian = jacobian * sqrt_w;
        *h += jacobian.transpose() * jacobian;
        *b += jacobian.transpose() * residual;
    }
}

fn pair_rotation_residual(
    image: usize,
    pose: SE3,
    other_pose: SE3,
    pair: &PairGeometry,
) -> glam::Vec3 {
    let predicted = if pair.left == image {
        other_pose.compose(&pose.inverse())
    } else {
        pose.compose(&other_pose.inverse())
    };
    let observed = pair.relative_pose;
    rotation_residual_vector(predicted, observed)
}

fn numerical_pair_rotation_jacobian(
    image: usize,
    pose: SE3,
    other_pose: SE3,
    pair: &PairGeometry,
) -> Option<SMatrix<f32, 3, 6>> {
    let mut jacobian = SMatrix::<f32, 3, 6>::zeros();
    let eps = [1.0e-5, 1.0e-5, 1.0e-5, 1.0e-4, 1.0e-4, 1.0e-4];
    for axis in 0..6 {
        let mut plus = SVector::<f32, 6>::zeros();
        plus[axis] = eps[axis];
        let mut minus = SVector::<f32, 6>::zeros();
        minus[axis] = -eps[axis];
        let r_plus = pair_rotation_residual(image, apply_pose_delta(pose, plus), other_pose, pair);
        let r_minus =
            pair_rotation_residual(image, apply_pose_delta(pose, minus), other_pose, pair);
        if !r_plus.is_finite() || !r_minus.is_finite() {
            return None;
        }
        let d = (r_plus - r_minus) / (2.0 * eps[axis]);
        jacobian[(0, axis)] = d.x;
        jacobian[(1, axis)] = d.y;
        jacobian[(2, axis)] = d.z;
    }
    Some(jacobian)
}

fn rotation_residual_vector(predicted: SE3, observed: SE3) -> glam::Vec3 {
    let delta = (pose_rotation(observed) * pose_rotation(predicted).inverse()).normalize();
    quat_log_local(delta)
}

fn quat_log_local(q: glam::Quat) -> glam::Vec3 {
    let mut q = q.normalize();
    if q.w < 0.0 {
        q = -q;
    }
    let w = q.w.clamp(-1.0, 1.0);
    let angle = 2.0 * w.acos();
    let sin_half = (1.0 - w * w).sqrt();
    if sin_half < 1.0e-6 || angle.abs() < 1.0e-6 {
        glam::Vec3::ZERO
    } else {
        glam::Vec3::new(q.x, q.y, q.z) * (angle / sin_half)
    }
}

fn pose_pair_rotation_weight() -> f32 {
    std::env::var("RUSTSFM_POSE_PAIR_ROTATION_WEIGHT")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(5.0)
}

fn rotation_only_gauss_newton_step(
    image: usize,
    pose: SE3,
    center: glam::Vec3,
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
    observations_by_image: &[Vec<(usize, usize)>],
    config: &MapperConfig,
) -> Option<SVector<f32, 3>> {
    let mut h = SMatrix::<f32, 3, 3>::zeros();
    let mut b = SVector::<f32, 3>::zeros();
    let mut used = 0usize;
    for &(point_id, feature) in &observations_by_image[image] {
        if point_id >= reconstruction.points.len() || feature >= frames[image].keypoints.len() {
            continue;
        }
        let point = reconstruction.points[point_id].xyz;
        let kp = &frames[image].keypoints[feature];
        let observed = [kp.x(), kp.y()];
        let camera = reconstruction.camera_for_image(image);
        let Some(predicted) = project_point_px(point, pose, camera) else {
            continue;
        };
        let residual =
            SVector::<f32, 2>::new(predicted[0] - observed[0], predicted[1] - observed[1]);
        let err = residual.norm();
        if !err.is_finite() || err > config.max_reprojection_error_px * 2.0 {
            continue;
        }
        let weight = huber_weight(err, 4.0);
        let jacobian = numerical_rotation_jacobian_fixed_center(point, pose, center, camera)?;
        h += jacobian.transpose() * jacobian * weight;
        b += jacobian.transpose() * residual * weight;
        used += 1;
    }
    if used < 24 {
        return None;
    }
    let lambda = 1.0e-3 * (h.trace() / 3.0).max(1.0);
    for i in 0..3 {
        h[(i, i)] += lambda;
    }
    let delta = h.lu().solve(&(-b))?;
    (delta.norm() < 0.25).then_some(delta)
}

fn observations_by_image(reconstruction: &Reconstruction) -> Vec<Vec<(usize, usize)>> {
    let mut by_image = vec![Vec::new(); reconstruction.poses.len()];
    for (point_id, point) in reconstruction.points.iter().enumerate() {
        for obs in &point.track {
            if obs.image < by_image.len() {
                by_image[obs.image].push((point_id, obs.feature));
            }
        }
    }
    by_image
}

fn image_reprojection_cost(
    image: usize,
    pose: SE3,
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
    observations_by_image: &[Vec<(usize, usize)>],
) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0usize;
    for &(point_id, feature) in &observations_by_image[image] {
        if point_id >= reconstruction.points.len() || feature >= frames[image].keypoints.len() {
            continue;
        }
        let kp = &frames[image].keypoints[feature];
        let err = crate::geometry::reprojection_error_px(
            reconstruction.points[point_id].xyz,
            pose,
            [kp.x(), kp.y()],
            reconstruction.camera_for_image(image),
        );
        if err.is_finite() {
            let robust = err.min(16.0);
            total += robust * robust;
            count += 1;
        }
    }
    if count == 0 {
        f32::INFINITY
    } else {
        total / count as f32
    }
}

fn apply_pose_delta(pose: SE3, delta: SVector<f32, 6>) -> SE3 {
    let tangent = [delta[0], delta[1], delta[2], delta[3], delta[4], delta[5]];
    SE3::exp(&tangent).compose(&pose)
}

fn apply_rotation_delta_fixed_center(pose: SE3, center: glam::Vec3, delta: SVector<f32, 3>) -> SE3 {
    let rotation = (glam::Quat::from_scaled_axis(glam::Vec3::new(delta[0], delta[1], delta[2]))
        * pose_rotation(pose))
    .normalize();
    pose_from_rotation_center(rotation, center)
}

fn numerical_pose_jacobian(
    point: [f32; 3],
    pose: SE3,
    camera: CameraModel,
) -> Option<SMatrix<f32, 2, 6>> {
    let mut jacobian = SMatrix::<f32, 2, 6>::zeros();
    let eps = [1.0e-5, 1.0e-5, 1.0e-5, 1.0e-4, 1.0e-4, 1.0e-4];
    for axis in 0..6 {
        let mut plus = SVector::<f32, 6>::zeros();
        plus[axis] = eps[axis];
        let mut minus = SVector::<f32, 6>::zeros();
        minus[axis] = -eps[axis];
        let p_plus = project_point_px(point, apply_pose_delta(pose, plus), camera)?;
        let p_minus = project_point_px(point, apply_pose_delta(pose, minus), camera)?;
        jacobian[(0, axis)] = (p_plus[0] - p_minus[0]) / (2.0 * eps[axis]);
        jacobian[(1, axis)] = (p_plus[1] - p_minus[1]) / (2.0 * eps[axis]);
    }
    Some(jacobian)
}

fn numerical_rotation_jacobian_fixed_center(
    point: [f32; 3],
    pose: SE3,
    center: glam::Vec3,
    camera: CameraModel,
) -> Option<SMatrix<f32, 2, 3>> {
    let mut jacobian = SMatrix::<f32, 2, 3>::zeros();
    let eps = 1.0e-5;
    for axis in 0..3 {
        let mut plus = SVector::<f32, 3>::zeros();
        plus[axis] = eps;
        let mut minus = SVector::<f32, 3>::zeros();
        minus[axis] = -eps;
        let p_plus = project_point_px(
            point,
            apply_rotation_delta_fixed_center(pose, center, plus),
            camera,
        )?;
        let p_minus = project_point_px(
            point,
            apply_rotation_delta_fixed_center(pose, center, minus),
            camera,
        )?;
        jacobian[(0, axis)] = (p_plus[0] - p_minus[0]) / (2.0 * eps);
        jacobian[(1, axis)] = (p_plus[1] - p_minus[1]) / (2.0 * eps);
    }
    Some(jacobian)
}

fn numerical_point_jacobian(
    point: [f32; 3],
    pose: SE3,
    camera: CameraModel,
) -> Option<SMatrix<f32, 2, 3>> {
    let mut jacobian = SMatrix::<f32, 2, 3>::zeros();
    let eps = 1.0e-4;
    for axis in 0..3 {
        let mut plus = point;
        let mut minus = point;
        plus[axis] += eps;
        minus[axis] -= eps;
        let p_plus = project_point_px(plus, pose, camera)?;
        let p_minus = project_point_px(minus, pose, camera)?;
        jacobian[(0, axis)] = (p_plus[0] - p_minus[0]) / (2.0 * eps);
        jacobian[(1, axis)] = (p_plus[1] - p_minus[1]) / (2.0 * eps);
    }
    Some(jacobian)
}

fn project_point_px(point: [f32; 3], pose: SE3, camera: CameraModel) -> Option<[f32; 2]> {
    let p = pose.transform_point(&point);
    camera.img_from_cam_f32(p[0], p[1], p[2])
}

fn huber_weight(err: f32, delta: f32) -> f32 {
    if err <= delta {
        1.0
    } else {
        delta / err.max(1.0e-6)
    }
}

fn refresh_point_errors(frames: &[ImageFrame], reconstruction: &mut Reconstruction) {
    let image_cameras = (0..reconstruction.poses.len())
        .map(|image| reconstruction.camera_for_image(image))
        .collect::<Vec<_>>();
    for point in &mut reconstruction.points {
        let mut total = 0.0f32;
        let mut count = 0usize;
        for obs in &point.track {
            let Some(pose) = reconstruction.poses[obs.image] else {
                continue;
            };
            let kp = &frames[obs.image].keypoints[obs.feature];
            let err = crate::geometry::reprojection_error_px(
                point.xyz,
                pose,
                [kp.x(), kp.y()],
                image_cameras[obs.image],
            );
            if err.is_finite() {
                total += err;
                count += 1;
            }
        }
        if count > 0 {
            point.error = total / count as f32;
        }
    }
}

struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        if self.rank[ra] == self.rank[rb] {
            self.rank[ra] += 1;
        }
    }
}

fn triangulate_pair(
    pair: &PairGeometry,
    frames: &[ImageFrame],
    reconstruction: &mut Reconstruction,
    config: &MapperConfig,
) {
    let Some(left_pose) = reconstruction.poses[pair.left] else {
        return;
    };
    let Some(right_pose) = reconstruction.poses[pair.right] else {
        return;
    };
    let mut norm_left = Vec::new();
    let mut norm_right = Vec::new();
    let mut matches = Vec::new();
    for m in &pair.inlier_matches {
        let li = m.query_idx as usize;
        let ri = m.train_idx as usize;
        if li >= frames[pair.left].keypoints.len()
            || ri >= frames[pair.right].keypoints.len()
            || reconstruction.observations[pair.left][li].is_some()
            || reconstruction.observations[pair.right][ri].is_some()
        {
            continue;
        }
        let lk = &frames[pair.left].keypoints[li];
        let rk = &frames[pair.right].keypoints[ri];
        norm_left.push(
            reconstruction
                .camera_for_image(pair.left)
                .normalize(lk.x(), lk.y()),
        );
        norm_right.push(
            reconstruction
                .camera_for_image(pair.right)
                .normalize(rk.x(), rk.y()),
        );
        matches.push(m.clone());
    }
    for (idx, (&left_xy, &right_xy)) in norm_left.iter().zip(norm_right.iter()).enumerate() {
        let Some(xyz) =
            crate::two_view::triangulate_world_point(left_pose, right_pose, left_xy, right_xy)
        else {
            continue;
        };
        let m = &matches[idx];
        let li = m.query_idx as usize;
        let ri = m.train_idx as usize;
        let lk = &frames[pair.left].keypoints[li];
        let rk = &frames[pair.right].keypoints[ri];
        let err = mean_pair_reprojection_error_with_cameras(
            xyz,
            left_pose,
            right_pose,
            [lk.x(), lk.y()],
            [rk.x(), rk.y()],
            reconstruction.camera_for_image(pair.left),
            reconstruction.camera_for_image(pair.right),
        );
        if !err.is_finite() || err > config.max_reprojection_error_px {
            continue;
        }
        let id = reconstruction.points.len();
        reconstruction.observations[pair.left][li] = Some(id);
        reconstruction.observations[pair.right][ri] = Some(id);
        reconstruction.point_ids.push(id as u64 + 1);
        reconstruction.points.push(Point3D {
            xyz,
            color: frames[pair.left].colors[li],
            error: err,
            track: vec![
                TrackObservation {
                    image: pair.left,
                    feature: li,
                },
                TrackObservation {
                    image: pair.right,
                    feature: ri,
                },
            ],
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::correspondence_graph::FeatureMatch;
    use crate::database::{
        ColmapDatabase, ColmapDatabaseCamera, ColmapDatabaseImage, ColmapKeypoint,
        ColmapTwoViewGeometry, DatabaseCacheOptions,
    };
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn reference_camera_setup_maps_images_to_shared_cameras() -> Result<()> {
        let dir = tempdir()?;
        let sparse = dir.path().join("sparse/0");
        fs::create_dir_all(&sparse)?;
        fs::write(
            sparse.join("cameras.txt"),
            "11 PINHOLE 640 480 500 501 320 240\n42 SIMPLE_RADIAL 800 600 700 401 299 0\n",
        )?;
        fs::write(
            sparse.join("images.txt"),
            "7 1 0 0 0 0 0 0 11 a.jpg\n\n8 1 0 0 0 0 0 0 42 b.jpg\n\n",
        )?;
        let image_paths = vec![PathBuf::from("a.jpg"), PathBuf::from("b.jpg")];

        let setup = reference_camera_setup(dir.path(), &image_paths)?;

        assert_eq!(setup.camera_ids, vec![11, 42]);
        assert_eq!(setup.image_ids, vec![7, 8]);
        assert_eq!(setup.image_camera_indices, vec![0, 1]);
        assert_eq!(setup.cameras[1].model_name(), "SIMPLE_RADIAL");
        Ok(())
    }

    #[test]
    fn database_pair_matches_map_colmap_image_names_to_frames() -> Result<()> {
        let dir = tempdir()?;
        let db = ColmapDatabase::open(dir.path().join("database.db"))?;
        db.write_camera(
            &ColmapDatabaseCamera {
                camera: crate::colmap::ColmapCamera {
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
        for (image_id, name) in [(10, "b.jpg"), (20, "a.jpg")] {
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
                &[ColmapKeypoint::new(0.0, 0.0), ColmapKeypoint::new(1.0, 1.0)],
            )?;
        }
        db.write_two_view_geometry(
            10,
            20,
            &ColmapTwoViewGeometry {
                config: 2,
                inlier_matches: vec![FeatureMatch::new(1, 0), FeatureMatch::new(0, 1)],
                ..ColmapTwoViewGeometry::default()
            },
        )?;
        let cache = db.load_cache(&DatabaseCacheOptions::default())?;
        let frames = vec![minimal_frame(0, "a.jpg"), minimal_frame(1, "b.jpg")];

        let pairs = database_pair_matches_for_frames(&frames, &cache)?;

        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].left, 0);
        assert_eq!(pairs[0].right, 1);
        assert_eq!(pairs[0].matches.len(), 2);
        assert_eq!(pairs[0].matches[0].query_idx, 0);
        assert_eq!(pairs[0].matches[0].train_idx, 1);
        Ok(())
    }

    #[test]
    fn mapper_database_input_loads_keypoints_and_camera_ownership() -> Result<()> {
        let dir = tempdir()?;
        let db_path = dir.path().join("database.db");
        let db = ColmapDatabase::open(&db_path)?;
        db.write_camera(
            &ColmapDatabaseCamera {
                camera: crate::colmap::ColmapCamera {
                    camera_id: 5,
                    model_id: crate::types::COLMAP_PINHOLE,
                    width: 200,
                    height: 150,
                    params: vec![80.0, 81.0, 100.0, 75.0],
                },
                has_prior_focal_length: true,
            },
            true,
        )?;
        for (image_id, name, keypoints) in [
            (
                11,
                "left.jpg",
                vec![
                    ColmapKeypoint::new(10.0, 20.0),
                    ColmapKeypoint::new(30.0, 40.0),
                ],
            ),
            (
                12,
                "right.jpg",
                vec![
                    ColmapKeypoint::new(50.0, 60.0),
                    ColmapKeypoint::new(70.0, 80.0),
                ],
            ),
        ] {
            db.write_image(
                &ColmapDatabaseImage {
                    image_id,
                    name: name.to_string(),
                    camera_id: 5,
                    frame_id: None,
                },
                true,
            )?;
            db.write_keypoints(image_id, &keypoints)?;
        }
        db.write_two_view_geometry(
            11,
            12,
            &ColmapTwoViewGeometry {
                config: 2,
                inlier_matches: vec![FeatureMatch::new(0, 1), FeatureMatch::new(1, 0)],
                ..ColmapTwoViewGeometry::default()
            },
        )?;
        let mut frames = vec![minimal_frame(0, "left.jpg"), minimal_frame(1, "right.jpg")];

        let database = load_mapper_database(Some(&db_path), &frames, 0)?.expect("database input");
        apply_database_keypoints(&mut frames, &database.keypoints_by_name);
        let setup = database_camera_setup(
            &database.cache,
            &[frames[0].path.clone(), frames[1].path.clone()],
        )?;

        assert_eq!(frames[0].keypoints[0].x(), 10.0);
        assert_eq!(frames[0].keypoints[1].y(), 40.0);
        assert_eq!(frames[1].keypoints[0].x(), 50.0);
        assert_eq!(setup.camera_ids, vec![5]);
        assert_eq!(setup.image_ids, vec![11, 12]);
        assert_eq!(setup.image_camera_indices, vec![0, 0]);
        assert_eq!(setup.cameras[0].fx, 80.0);
        Ok(())
    }

    #[test]
    fn database_pair_geometry_uses_stored_two_view_pose() -> Result<()> {
        let dir = tempdir()?;
        let db_path = dir.path().join("database.db");
        let db = ColmapDatabase::open(&db_path)?;
        db.write_camera(
            &ColmapDatabaseCamera {
                camera: crate::colmap::ColmapCamera {
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
        let left_keypoints = vec![
            ColmapKeypoint::new(45.0, 45.0),
            ColmapKeypoint::new(55.0, 45.0),
            ColmapKeypoint::new(45.0, 55.0),
            ColmapKeypoint::new(55.0, 55.0),
        ];
        let right_keypoints = vec![
            ColmapKeypoint::new(40.0, 45.0),
            ColmapKeypoint::new(50.0, 45.0),
            ColmapKeypoint::new(40.0, 55.0),
            ColmapKeypoint::new(50.0, 55.0),
        ];
        for (image_id, name, keypoints) in [
            (1, "left.jpg", left_keypoints),
            (2, "right.jpg", right_keypoints),
        ] {
            db.write_image(
                &ColmapDatabaseImage {
                    image_id,
                    name: name.to_string(),
                    camera_id: 1,
                    frame_id: None,
                },
                true,
            )?;
            db.write_keypoints(image_id, &keypoints)?;
        }
        db.write_two_view_geometry(
            1,
            2,
            &ColmapTwoViewGeometry {
                config: 2,
                inlier_matches: (0..4).map(|idx| FeatureMatch::new(idx, idx)).collect(),
                qvec: Some([1.0, 0.0, 0.0, 0.0]),
                tvec: Some([-1.0, 0.0, 0.0]),
                ..ColmapTwoViewGeometry::default()
            },
        )?;
        let mut frames = vec![minimal_frame(0, "left.jpg"), minimal_frame(1, "right.jpg")];
        let database = load_mapper_database(Some(&db_path), &frames, 0)?.expect("database input");
        apply_database_keypoints(&mut frames, &database.keypoints_by_name);
        let pair_matches = database_pair_matches_for_frames(&frames, &database.cache)?;
        let camera = CameraModel::from_colmap(
            crate::types::COLMAP_PINHOLE,
            100,
            100,
            &[50.0, 50.0, 50.0, 50.0],
        )
        .unwrap();

        let pair = database_pair_geometry_from_stored_pose(
            &pair_matches[0],
            &frames,
            &database.cache,
            &database.two_view_geometries,
            camera,
            camera,
            &MapperConfig {
                min_inliers: 4,
                min_triangulated: 4,
                max_reprojection_error_px: 1.0,
                ..MapperConfig::default()
            },
        )
        .expect("stored pose pair");

        assert_eq!(pair.left, 0);
        assert_eq!(pair.right, 1);
        assert_eq!(pair.inliers, 4);
        assert_eq!(pair.triangulated, 4);
        assert!(pair.mean_reprojection_error_px < 1.0e-4);
        assert_eq!(pair.relative_pose.translation(), [-1.0, 0.0, 0.0]);
        Ok(())
    }

    fn minimal_frame(id: usize, name: &str) -> ImageFrame {
        ImageFrame {
            id,
            name: name.to_string(),
            path: PathBuf::from(name),
            width: 100,
            height: 100,
            keypoints: vec![
                rustslam::KeyPoint::new(0.0, 0.0),
                rustslam::KeyPoint::new(1.0, 1.0),
            ],
            descriptors: rustslam::Descriptors::new(),
            sift: crate::sift::SiftFeatures::default(),
            wide_descriptors: crate::wide::WideDescriptors {
                data: Vec::new(),
                dim: 0,
                count: 0,
            },
            strong_feature_indices: Vec::new(),
            colors: vec![[0, 0, 0], [0, 0, 0]],
        }
    }
}
