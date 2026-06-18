use crate::colmap::{
    export_colmap, read_camera_model, read_colmap_cameras, read_colmap_poses,
    read_colmap_sparse_model, world_to_camera_rotation, ColmapDataId, ColmapRig, ColmapRigSensor,
    ColmapRigid3, ColmapSensorId, ColmapSensorType,
};
use crate::correspondence_graph::ImagePairId;
use crate::database::{ColmapDatabase, ColmapTwoViewGeometry, DatabaseCache, DatabaseCacheOptions};
use crate::generalized_pose::{
    estimate_structureless_absolute_pose, GeneralizedPoseError,
    StructureLessAbsolutePoseEstimationOptions, StructureLessAbsolutePoseProblem,
};
use crate::geometry::{
    camera_center, estimate_pair_geometry_with_cameras,
    estimate_pair_geometry_with_options_and_cameras, mean_pair_reprojection_error_with_cameras,
    pose_from_rotation_center, pose_rotation, pose_with_flipped_translation, relative_rotation_deg,
    PairEstimationOptions,
};
use crate::incremental_triangulator::{
    IncrementalTriangulator, IncrementalTriangulatorOptions, TriangulationReport,
};
use crate::observation_manager::ObservationManager;
use crate::pose_graph::initialize_pose_graph;
use crate::sift::{
    extract_sift_features_with_options, match_sift_with_options, SiftExtractionOptions,
    SiftMatchingOptions,
};
use crate::types::{
    colmap_camera_model_extra_idxs, colmap_camera_model_focal_idxs,
    colmap_camera_model_principal_point_idxs, CameraModel, DataId, Frame, ImageFrame, PairGeometry,
    Point3D, Reconstruction, Rig, RigSensor, Rigid3, SensorId, SensorType, TrackObservation,
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
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
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

#[derive(Debug, Clone, Default)]
struct RegistrationStats {
    num_reg_frames_per_rig: HashMap<u32, usize>,
    num_reg_images_per_camera: HashMap<u32, usize>,
    num_registrations: Vec<usize>,
    num_total_reg_images: usize,
    num_shared_reg_images: usize,
}

impl RegistrationStats {
    fn from_reconstruction(reconstruction: &Reconstruction) -> Self {
        let mut stats = Self {
            num_registrations: vec![0; reconstruction.poses.len()],
            ..Self::default()
        };
        let mut registered_frames = HashSet::new();
        for image in 0..reconstruction.poses.len() {
            if reconstruction.poses[image].is_none() {
                continue;
            }
            if let Some(frame_idx) = reconstruction.frame_index_for_image(image) {
                if registered_frames.insert(frame_idx) {
                    stats.register_frame_event(reconstruction, frame_idx);
                }
            } else {
                stats.register_image_event(reconstruction, image);
            }
        }
        stats
    }

    fn register_image_event(&mut self, reconstruction: &Reconstruction, image: usize) {
        if image >= reconstruction.poses.len() {
            return;
        }
        let camera_id = reconstruction.camera_id_for_image(image);
        *self.num_reg_images_per_camera.entry(camera_id).or_default() += 1;
        if image >= self.num_registrations.len() {
            self.num_registrations.resize(image + 1, 0);
        }
        self.num_registrations[image] += 1;
        if self.num_registrations[image] == 1 {
            self.num_total_reg_images += 1;
        } else {
            self.num_shared_reg_images += 1;
        }
    }

    #[allow(dead_code)]
    fn deregister_image_event(&mut self, reconstruction: &Reconstruction, image: usize) {
        if image >= reconstruction.poses.len() {
            return;
        }
        let camera_id = reconstruction.camera_id_for_image(image);
        if let Some(count) = self.num_reg_images_per_camera.get_mut(&camera_id) {
            *count = count.saturating_sub(1);
        }
        if let Some(registrations) = self.num_registrations.get_mut(image) {
            if *registrations > 0 {
                *registrations -= 1;
                if *registrations == 0 {
                    self.num_total_reg_images = self.num_total_reg_images.saturating_sub(1);
                } else {
                    self.num_shared_reg_images = self.num_shared_reg_images.saturating_sub(1);
                }
            }
        }
    }

    fn register_frame_for_image_event(&mut self, reconstruction: &Reconstruction, image: usize) {
        if let Some(frame_idx) = reconstruction.frame_index_for_image(image) {
            self.register_frame_event(reconstruction, frame_idx);
        } else {
            self.register_image_event(reconstruction, image);
        }
    }

    #[allow(dead_code)]
    fn deregister_frame_for_image_event(&mut self, reconstruction: &Reconstruction, image: usize) {
        if let Some(frame_idx) = reconstruction.frame_index_for_image(image) {
            self.deregister_frame_event(reconstruction, frame_idx);
        } else {
            self.deregister_image_event(reconstruction, image);
        }
    }

    fn register_frame_event(&mut self, reconstruction: &Reconstruction, frame_idx: usize) {
        let Some(frame) = reconstruction.frames.get(frame_idx) else {
            return;
        };
        *self.num_reg_frames_per_rig.entry(frame.rig_id).or_default() += 1;
        for image in reconstruction.image_indices_for_frame_index(frame_idx) {
            self.register_image_event(reconstruction, image);
        }
    }

    #[allow(dead_code)]
    fn deregister_frame_event(&mut self, reconstruction: &Reconstruction, frame_idx: usize) {
        let Some(frame) = reconstruction.frames.get(frame_idx) else {
            return;
        };
        if let Some(count) = self.num_reg_frames_per_rig.get_mut(&frame.rig_id) {
            *count = count.saturating_sub(1);
        }
        for image in reconstruction.image_indices_for_frame_index(frame_idx) {
            self.deregister_image_event(reconstruction, image);
        }
    }

    fn registered_images_with_camera_id(&self, camera_id: u32) -> usize {
        self.num_reg_images_per_camera
            .get(&camera_id)
            .copied()
            .unwrap_or(0)
    }
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
    pub local_matching: bool,
    pub local_window: usize,
    pub experimental_sequence_heuristics: bool,
    pub experimental_ring_closure: bool,
    pub experimental_structureless_pair_pose_fallback: bool,
    pub min_matches: usize,
    pub min_inliers: usize,
    pub min_triangulated: usize,
    pub init_min_num_inliers: usize,
    pub init_min_tri_angle_deg: f32,
    pub init_max_forward_motion: f32,
    pub essential_threshold_px: f32,
    pub essential_iterations: u32,
    pub pnp_threshold_px: f32,
    pub pnp_iterations: u32,
    pub abs_pose_min_num_inliers: usize,
    pub abs_pose_min_inlier_ratio: f32,
    pub random_seed: i32,
    pub max_reg_trials: usize,
    pub local_ba: bool,
    pub local_ba_num_images: usize,
    pub local_ba_min_shared_points: usize,
    pub local_ba_iterations: usize,
    pub global_ba: bool,
    pub global_ba_iterations: usize,
    pub global_ba_images_ratio: f32,
    pub global_ba_points_ratio: f32,
    pub global_ba_images_freq: usize,
    pub global_ba_points_freq: usize,
    pub global_ba_max_refinements: usize,
    pub global_ba_max_refinement_change: f32,
    pub ba_refine_focal_length: bool,
    pub ba_refine_principal_point: bool,
    pub ba_refine_extra_params: bool,
    pub ba_constant_rig_ids: Vec<u32>,
    pub ba_constant_camera_ids: Vec<u32>,
    pub min_focal_length_ratio: f64,
    pub max_focal_length_ratio: f64,
    pub max_extra_param: f64,
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
            local_matching: false,
            local_window: 0,
            experimental_sequence_heuristics: false,
            experimental_ring_closure: false,
            experimental_structureless_pair_pose_fallback: false,
            min_matches: 15,
            min_inliers: 15,
            min_triangulated: 4,
            init_min_num_inliers: 100,
            init_min_tri_angle_deg: 16.0,
            init_max_forward_motion: 0.95,
            essential_threshold_px: 2.0,
            essential_iterations: 10000,
            pnp_threshold_px: 12.0,
            pnp_iterations: 10000,
            abs_pose_min_num_inliers: 30,
            abs_pose_min_inlier_ratio: 0.25,
            random_seed: -1,
            max_reg_trials: 3,
            local_ba: true,
            local_ba_num_images: 6,
            local_ba_min_shared_points: 15,
            local_ba_iterations: 5,
            global_ba: true,
            global_ba_iterations: 8,
            global_ba_images_ratio: 1.1,
            global_ba_points_ratio: 1.1,
            global_ba_images_freq: 500,
            global_ba_points_freq: 250_000,
            global_ba_max_refinements: 5,
            global_ba_max_refinement_change: 0.0005,
            ba_refine_focal_length: true,
            ba_refine_principal_point: false,
            ba_refine_extra_params: true,
            ba_constant_rig_ids: Vec::new(),
            ba_constant_camera_ids: Vec::new(),
            min_focal_length_ratio: 0.1,
            max_focal_length_ratio: 10.0,
            max_extra_param: 1.0,
            max_reprojection_error_px: 8.0,
            pose_graph: false,
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
    let database_path = resolve_mapper_database_path(config)?;
    let mapper_database =
        load_mapper_database(database_path.as_deref(), &frames, config.min_matches)?;
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
    if reference_camera_setup.is_none() && mapper_database.is_none() {
        reference_camera_setup = Some(local_image_camera_setup(&frames, config));
        if let Some(first) = reference_camera_setup
            .as_ref()
            .and_then(|setup| setup.cameras.first())
            .copied()
        {
            camera = first;
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
        if !config.local_matching {
            bail!(
                "COLMAP-style reconstruction requires a database. Provide --database, place database.db next to the image root, or pass --local-matching for the experimental image-only fallback."
            );
        }
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
    if let Some(path) = database_path.as_ref() {
        debug_log.push(format!("database_path={}", path.display()));
    } else {
        debug_log.push("database_path=<none> local_matching_fallback=true".to_string());
    }
    debug_log.extend(pair_connectivity_summary(&pairs, &frames));
    debug_log.extend(pair_config_summary(&pairs));
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
            filter_reprojection_tracks(&frames, &mapping_pairs, &mut reconstruction, config);
        }
        if std::env::var_os("RUSTSFM_SKIP_POSE_REFINE").is_none() {
            refine_registered_poses_pose_only(&frames, &mapping_pairs, &mut reconstruction, config);
        }
        if std::env::var_os("RUSTSFM_EXPERIMENTAL_BA").is_some() {
            let ba_iterations = global_ba_iterations(config);
            let mut ba_options = mapper_ba_options(
                config,
                &reconstruction,
                ba_iterations,
                None,
                Vec::new(),
                None,
                None,
            );
            ba_options.gauge = crate::ba::BundleAdjustmentGauge::TwoCamsFromWorld;
            if let Some(report) =
                refine_bundle_adjustment_checked(&frames, &mut reconstruction, config, ba_options)
            {
                debug_log.push(format!(
                    "schur_ba termination={:?} reason={:?} iterations={}/{} observations={} residuals={} cost={:.6}->{:.6}",
                    report.termination_type,
                    report.termination_reason,
                    report.iterations,
                    report.attempted_iterations,
                    report.observations,
                    report.residuals,
                    report.initial_cost,
                    report.final_cost
                ));
            }
            if std::env::var_os("RUSTSFM_FILTER_TRACKS").is_some() {
                let removed = filter_reprojection_tracks(
                    &frames,
                    &mapping_pairs,
                    &mut reconstruction,
                    config,
                );
                debug_log.push(format!("track_filter removed_observations={removed}"));
            }
        }
        debug_log.push(format!(
            "timing_pose_graph_ms={:.2}",
            pose_graph_start.elapsed().as_secs_f64() * 1000.0
        ));
        debug_log.push("pose_graph_refinement enabled".to_string());
    }
    sync_registered_frame_poses_from_images(&mut reconstruction);
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

fn resolve_mapper_database_path(config: &MapperConfig) -> Result<Option<PathBuf>> {
    if let Some(database) = &config.database {
        if !database.exists() {
            bail!("database path does not exist: {}", database.display());
        }
        return Ok(Some(database.clone()));
    }

    for candidate in default_database_candidates(&config.input) {
        if candidate.exists() {
            return Ok(Some(candidate));
        }
    }

    Ok(None)
}

fn default_database_candidates(input: &Path) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    push_unique_path(&mut candidates, input.join("database.db"));
    if let Some(parent) = input.parent() {
        push_unique_path(&mut candidates, parent.join("database.db"));
    }
    candidates
}

fn push_unique_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.iter().any(|existing| existing == &path) {
        paths.push(path);
    }
}

#[derive(Debug, Clone)]
struct ReferenceCameraSetup {
    cameras: Vec<CameraModel>,
    camera_ids: Vec<u32>,
    camera_has_prior_focal_length: Vec<bool>,
    rigs: Vec<Rig>,
    frames: Vec<Frame>,
    image_ids: Vec<u32>,
    image_camera_indices: Vec<usize>,
    image_frame_indices: Vec<Option<usize>>,
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
    let sparse_model = read_colmap_sparse_model(reference).ok();
    let rigs = sparse_model
        .as_ref()
        .map(|model| model.reconstruction.rigs.clone())
        .unwrap_or_default();
    let frames = sparse_model
        .as_ref()
        .map(|model| model.reconstruction.frames.clone())
        .unwrap_or_default();
    let image_frame_by_id = sparse_model
        .as_ref()
        .map(|model| {
            model
                .reconstruction
                .image_ids
                .iter()
                .copied()
                .zip(model.reconstruction.image_frame_indices.iter().copied())
                .collect::<HashMap<_, _>>()
        })
        .unwrap_or_default();

    let mut image_ids = Vec::with_capacity(image_paths.len());
    let mut image_camera_indices = Vec::with_capacity(image_paths.len());
    let mut image_frame_indices = Vec::with_capacity(image_paths.len());
    for (idx, path) in image_paths.iter().enumerate() {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if let Some(pose) = pose_by_name.get(name) {
            image_ids.push(pose.image_id);
            image_camera_indices.push(*camera_index_by_id.get(&pose.camera_id).unwrap_or(&0));
            image_frame_indices.push(*image_frame_by_id.get(&pose.image_id).unwrap_or(&None));
        } else {
            image_ids.push(idx as u32 + 1);
            image_camera_indices.push(0);
            image_frame_indices.push(None);
        }
    }

    Ok(ReferenceCameraSetup {
        cameras,
        camera_ids,
        camera_has_prior_focal_length: vec![true; cameras_with_ids.len()],
        rigs,
        frames,
        image_ids,
        image_camera_indices,
        image_frame_indices,
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
    let mut camera_has_prior_focal_length = Vec::with_capacity(cache.cameras.len());
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
        camera_has_prior_focal_length.push(db_camera.has_prior_focal_length);
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
    let rigs = cache.rigs.values().map(rig_from_colmap).collect::<Vec<_>>();
    let frames = cache
        .frames
        .values()
        .map(database_frame_to_frame)
        .collect::<Vec<_>>();
    let frame_index_by_id = frames
        .iter()
        .enumerate()
        .map(|(idx, frame)| (frame.frame_id, idx))
        .collect::<HashMap<_, _>>();

    let mut image_ids = Vec::with_capacity(image_paths.len());
    let mut image_camera_indices = Vec::with_capacity(image_paths.len());
    let mut image_frame_indices = Vec::with_capacity(image_paths.len());
    for (idx, path) in image_paths.iter().enumerate() {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if let Some(image) = image_by_name.get(name) {
            image_ids.push(image.image_id);
            image_camera_indices.push(*camera_index_by_id.get(&image.camera_id).unwrap_or(&0));
            image_frame_indices.push(
                image
                    .frame_id
                    .and_then(|frame_id| frame_index_by_id.get(&frame_id).copied()),
            );
        } else {
            image_ids.push(idx as u32 + 1);
            image_camera_indices.push(0);
            image_frame_indices.push(None);
        }
    }

    Ok(ReferenceCameraSetup {
        cameras,
        camera_ids,
        camera_has_prior_focal_length,
        rigs,
        frames,
        image_ids,
        image_camera_indices,
        image_frame_indices,
    })
}

fn local_image_camera_setup(frames: &[ImageFrame], config: &MapperConfig) -> ReferenceCameraSetup {
    let mut cameras = Vec::with_capacity(frames.len());
    let mut camera_ids = Vec::with_capacity(frames.len());
    let mut image_ids = Vec::with_capacity(frames.len());
    let mut image_camera_indices = Vec::with_capacity(frames.len());
    for (idx, frame) in frames.iter().enumerate() {
        let focal = frame.width.max(frame.height) as f32 * 1.2;
        let mut camera = CameraModel::new_pinhole(
            frame.width,
            frame.height,
            focal,
            focal,
            frame.width as f32 * 0.5,
            frame.height as f32 * 0.5,
        );
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
        cameras.push(camera);
        camera_ids.push(idx as u32 + 1);
        image_ids.push(idx as u32 + 1);
        image_camera_indices.push(idx);
    }
    ReferenceCameraSetup {
        cameras,
        camera_ids,
        camera_has_prior_focal_length: vec![true; frames.len()],
        rigs: Vec::new(),
        frames: Vec::new(),
        image_ids,
        image_camera_indices,
        image_frame_indices: vec![None; frames.len()],
    }
}

fn rig_from_colmap(rig: &ColmapRig) -> Rig {
    Rig {
        rig_id: rig.rig_id,
        ref_sensor_id: rig.ref_sensor_id.as_ref().map(sensor_id_from_colmap),
        sensors: rig.sensors.iter().map(rig_sensor_from_colmap).collect(),
    }
}

fn rig_sensor_from_colmap(sensor: &ColmapRigSensor) -> RigSensor {
    RigSensor {
        sensor_id: sensor_id_from_colmap(&sensor.sensor_id),
        sensor_from_rig: sensor.sensor_from_rig.as_ref().map(rigid3_from_colmap),
    }
}

fn database_frame_to_frame(frame: &crate::database::ColmapDatabaseFrame) -> Frame {
    Frame {
        frame_id: frame.frame_id,
        rig_id: frame.rig_id,
        rig_from_world: Rigid3 {
            qvec: [1.0, 0.0, 0.0, 0.0],
            tvec: [0.0, 0.0, 0.0],
        },
        data_ids: frame.data_ids.iter().map(data_id_from_colmap).collect(),
    }
}

fn sensor_id_from_colmap(sensor_id: &ColmapSensorId) -> SensorId {
    SensorId {
        sensor_type: sensor_type_from_colmap(&sensor_id.sensor_type),
        sensor_id: sensor_id.sensor_id,
    }
}

fn sensor_type_from_colmap(sensor_type: &ColmapSensorType) -> SensorType {
    match sensor_type {
        ColmapSensorType::Invalid => SensorType::Invalid,
        ColmapSensorType::Camera => SensorType::Camera,
        ColmapSensorType::Imu => SensorType::Imu,
        ColmapSensorType::Other(value) => SensorType::Other(value.clone()),
    }
}

fn rigid3_from_colmap(rigid: &ColmapRigid3) -> Rigid3 {
    Rigid3 {
        qvec: rigid.qvec,
        tvec: rigid.tvec,
    }
}

fn data_id_from_colmap(data_id: &ColmapDataId) -> DataId {
    DataId {
        sensor_id: sensor_id_from_colmap(&data_id.sensor_id),
        data_id: data_id.data_id,
    }
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

fn global_ba_iterations(config: &MapperConfig) -> usize {
    std::env::var("RUSTSFM_BA_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(config.global_ba_iterations)
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

fn mapper_ba_options(
    config: &MapperConfig,
    reconstruction: &Reconstruction,
    iterations: usize,
    variable_images: Option<Vec<usize>>,
    constant_images: Vec<usize>,
    point_ids: Option<Vec<usize>>,
    constant_point_ids: Option<Vec<usize>>,
) -> crate::ba::BundleAdjustmentOptions {
    let mut options = crate::ba::BundleAdjustmentOptions {
        iterations,
        huber_delta_px: global_ba_huber_delta_px(),
        max_observation_error_px: global_ba_max_observation_error_px(config),
        variable_images,
        constant_images,
        variable_cameras: None,
        constant_cameras: ba_constant_camera_indices(config, reconstruction),
        constant_rigs: ba_constant_rig_ids(config, reconstruction),
        constant_sensor_from_rig: ba_constant_sensor_from_rig_ids(config, reconstruction),
        refine_focal_length: config.ba_refine_focal_length,
        refine_principal_point: config.ba_refine_principal_point,
        refine_extra_params: config.ba_refine_extra_params,
        point_ids,
        constant_point_ids,
        ..crate::ba::BundleAdjustmentOptions::default()
    };
    apply_colmap_global_ba_solver_options(&mut options);
    options
}

fn apply_colmap_global_ba_solver_options(options: &mut crate::ba::BundleAdjustmentOptions) {
    options.gradient_tolerance = 1.0;
    options.parameter_tolerance = 0.0;
    options.max_linear_solver_iterations = 100;
}

fn apply_colmap_local_ba_solver_options(options: &mut crate::ba::BundleAdjustmentOptions) {
    options.gradient_tolerance = 10.0;
    options.parameter_tolerance = 0.0;
    options.max_linear_solver_iterations = 100;
}

fn mapper_local_ba_options(
    config: &MapperConfig,
    reconstruction: &Reconstruction,
    registration_stats: &RegistrationStats,
    iterations: usize,
    variable_images: Vec<usize>,
    constant_images: Vec<usize>,
    point_ids: Option<Vec<usize>>,
    constant_point_ids: Option<Vec<usize>>,
) -> crate::ba::BundleAdjustmentOptions {
    let variable_images = expand_images_to_registration_frames(reconstruction, &variable_images);
    let constant_images = expand_images_to_registration_frames(reconstruction, &constant_images);
    let local_constant_cameras = local_ba_constant_camera_indices(
        config,
        reconstruction,
        registration_stats,
        &variable_images,
    );
    let local_constant_sensors = local_ba_constant_sensor_from_rig_ids(
        config,
        reconstruction,
        registration_stats,
        &variable_images,
    );
    let mut options = mapper_ba_options(
        config,
        reconstruction,
        iterations,
        Some(variable_images),
        constant_images,
        point_ids,
        constant_point_ids,
    );
    options.constant_cameras = local_constant_cameras;
    options.constant_sensor_from_rig = local_constant_sensors;
    options.gauge = crate::ba::BundleAdjustmentGauge::ThreePoints;
    apply_colmap_local_ba_solver_options(&mut options);
    options
}

fn refine_bundle_adjustment_checked(
    frames: &[ImageFrame],
    reconstruction: &mut Reconstruction,
    config: &MapperConfig,
    options: crate::ba::BundleAdjustmentOptions,
) -> Option<crate::ba::BundleAdjustmentReport> {
    if !bogus_registered_camera_indices(reconstruction, config).is_empty() {
        return None;
    }
    let base_camera = reconstruction.camera;
    let base_cameras = reconstruction.cameras.clone();
    let base_poses = reconstruction.poses.clone();
    let base_points = reconstruction
        .points
        .iter()
        .map(|point| point.xyz)
        .collect::<Vec<_>>();

    let report = crate::ba::refine_bundle_adjustment(frames, reconstruction, options);
    if report
        .as_ref()
        .map_or(true, |report| !report.is_solution_usable())
        || !bogus_registered_camera_indices(reconstruction, config).is_empty()
    {
        restore_ba_state(
            reconstruction,
            base_camera,
            &base_cameras,
            &base_poses,
            &base_points,
        );
        return None;
    }
    sync_registered_frame_poses_from_images(reconstruction);
    report
}

fn sync_registered_frame_poses_from_images(reconstruction: &mut Reconstruction) {
    for frame_idx in 0..reconstruction.frames.len() {
        let Some((rig_from_world, image_poses)) =
            frame_consistent_poses_from_registered_images(reconstruction, frame_idx)
        else {
            continue;
        };
        reconstruction.frames[frame_idx].rig_from_world = Rigid3::from_se3(rig_from_world);
        for (image, pose) in image_poses {
            if let Some(slot) = reconstruction.poses.get_mut(image) {
                *slot = Some(pose);
            }
        }
    }
}

fn frame_consistent_poses_from_registered_images(
    reconstruction: &Reconstruction,
    frame_idx: usize,
) -> Option<(SE3, Vec<(usize, SE3)>)> {
    let frame = reconstruction.frames.get(frame_idx)?;
    let registered_image = reconstruction
        .image_indices_for_frame_index(frame_idx)
        .into_iter()
        .find(|&image| reconstruction.poses.get(image).copied().flatten().is_some())?;
    let selected_pose = reconstruction.poses[registered_image]?;
    let selected_sensor_id =
        reconstruction.frame_sensor_id_for_image(frame_idx, registered_image)?;
    let selected_sensor_from_rig = reconstruction
        .sensor_from_rig(frame.rig_id, selected_sensor_id)
        .unwrap_or_else(SE3::identity);
    let rig_from_world = selected_sensor_from_rig.inverse().compose(&selected_pose);
    let image_poses = reconstruction
        .image_indices_for_frame_index(frame_idx)
        .into_iter()
        .filter_map(|image| {
            let sensor_id = reconstruction.frame_sensor_id_for_image(frame_idx, image)?;
            let sensor_from_rig = reconstruction
                .sensor_from_rig(frame.rig_id, sensor_id)
                .unwrap_or_else(SE3::identity);
            Some((image, sensor_from_rig.compose(&rig_from_world)))
        })
        .collect::<Vec<_>>();
    (!image_poses.is_empty()).then_some((rig_from_world, image_poses))
}

fn restore_ba_state(
    reconstruction: &mut Reconstruction,
    camera: CameraModel,
    cameras: &[CameraModel],
    poses: &[Option<SE3>],
    points: &[[f32; 3]],
) {
    reconstruction.camera = camera;
    reconstruction.cameras.clone_from_slice(cameras);
    reconstruction.poses.clone_from_slice(poses);
    for (point, xyz) in reconstruction.points.iter_mut().zip(points.iter()) {
        point.xyz = *xyz;
    }
}

fn ba_constant_camera_indices(
    config: &MapperConfig,
    reconstruction: &Reconstruction,
) -> Vec<usize> {
    if config.ba_constant_camera_ids.is_empty() {
        return Vec::new();
    }
    let constant_ids = config
        .ba_constant_camera_ids
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    reconstruction
        .camera_ids
        .iter()
        .enumerate()
        .filter_map(|(idx, camera_id)| constant_ids.contains(camera_id).then_some(idx))
        .collect()
}

fn ba_constant_rig_ids(config: &MapperConfig, reconstruction: &Reconstruction) -> Vec<u32> {
    if config.ba_constant_rig_ids.is_empty() {
        return Vec::new();
    }
    let available = reconstruction
        .rigs
        .iter()
        .map(|rig| rig.rig_id)
        .collect::<HashSet<_>>();
    let mut rig_ids = config
        .ba_constant_rig_ids
        .iter()
        .copied()
        .filter(|rig_id| available.contains(rig_id))
        .collect::<Vec<_>>();
    rig_ids.sort_unstable();
    rig_ids.dedup();
    rig_ids
}

fn ba_constant_sensor_from_rig_ids(
    config: &MapperConfig,
    reconstruction: &Reconstruction,
) -> Vec<SensorId> {
    constant_sensor_from_rig_ids_for_rigs(
        reconstruction,
        ba_constant_rig_ids(config, reconstruction).into_iter(),
    )
}

fn local_ba_constant_sensor_from_rig_ids(
    config: &MapperConfig,
    reconstruction: &Reconstruction,
    registration_stats: &RegistrationStats,
    variable_images: &[usize],
) -> Vec<SensorId> {
    let constant_rigs = config.ba_constant_rig_ids.iter().copied();
    let partial_rigs =
        local_ba_partial_rig_ids(reconstruction, registration_stats, variable_images);
    constant_sensor_from_rig_ids_for_rigs(reconstruction, constant_rigs.chain(partial_rigs))
}

fn local_ba_partial_rig_ids(
    reconstruction: &Reconstruction,
    registration_stats: &RegistrationStats,
    variable_images: &[usize],
) -> Vec<u32> {
    let mut local_frame_indices = HashSet::new();
    let mut local_frames_per_rig = HashMap::<u32, usize>::new();
    for &image in variable_images {
        let Some(frame_idx) = reconstruction.frame_index_for_image(image) else {
            continue;
        };
        if !local_frame_indices.insert(frame_idx) {
            continue;
        }
        let Some(frame) = reconstruction.frames.get(frame_idx) else {
            continue;
        };
        *local_frames_per_rig.entry(frame.rig_id).or_default() += 1;
    }
    let mut partial_rigs = local_frames_per_rig
        .into_iter()
        .filter_map(|(rig_id, local_count)| {
            let registered_count = registration_stats
                .num_reg_frames_per_rig
                .get(&rig_id)
                .copied()
                .unwrap_or(0);
            (local_count < registered_count).then_some(rig_id)
        })
        .collect::<Vec<_>>();
    partial_rigs.sort_unstable();
    partial_rigs
}

fn constant_sensor_from_rig_ids_for_rigs(
    reconstruction: &Reconstruction,
    rig_ids: impl IntoIterator<Item = u32>,
) -> Vec<SensorId> {
    let rig_ids = rig_ids.into_iter().collect::<HashSet<_>>();
    let mut sensor_ids = reconstruction
        .rigs
        .iter()
        .filter(|rig| rig_ids.contains(&rig.rig_id))
        .flat_map(|rig| {
            rig.sensors
                .iter()
                .filter(|sensor| {
                    rig.ref_sensor_id
                        .as_ref()
                        .is_none_or(|ref_sensor_id| ref_sensor_id != &sensor.sensor_id)
                })
                .map(|sensor| sensor.sensor_id.clone())
        })
        .collect::<Vec<_>>();
    sensor_ids.sort_unstable();
    sensor_ids.dedup();
    sensor_ids
}

fn local_ba_constant_camera_indices(
    config: &MapperConfig,
    reconstruction: &Reconstruction,
    registration_stats: &RegistrationStats,
    variable_images: &[usize],
) -> Vec<usize> {
    let mut constant = ba_constant_camera_indices(config, reconstruction)
        .into_iter()
        .collect::<BTreeSet<_>>();
    let local_camera_counts = variable_images
        .iter()
        .filter_map(|&image| {
            let camera_idx = reconstruction.image_camera_indices.get(image).copied()?;
            let camera_id = reconstruction
                .camera_ids
                .get(camera_idx)
                .copied()
                .unwrap_or(1);
            Some((camera_idx, camera_id))
        })
        .fold(
            HashMap::<usize, (u32, usize)>::new(),
            |mut counts, (idx, id)| {
                counts
                    .entry(idx)
                    .and_modify(|(_, count)| *count += 1)
                    .or_insert((id, 1));
                counts
            },
        );
    for (camera_idx, (camera_id, local_count)) in local_camera_counts {
        if local_count < registration_stats.registered_images_with_camera_id(camera_id) {
            constant.insert(camera_idx);
        }
    }
    constant.into_iter().collect()
}

fn expand_images_to_registration_frames(
    reconstruction: &Reconstruction,
    images: &[usize],
) -> Vec<usize> {
    let mut expanded = Vec::new();
    for &image in images {
        expanded.extend(reconstruction.image_indices_for_registration_unit(image));
    }
    expanded.sort_unstable();
    expanded.dedup();
    expanded
}

fn registered_frame_count(reconstruction: &Reconstruction) -> usize {
    let mut frame_indices = HashSet::new();
    let mut trivial_images = 0usize;
    for (image, pose) in reconstruction.poses.iter().enumerate() {
        if pose.is_none() {
            continue;
        }
        if let Some(frame_idx) = reconstruction.frame_index_for_image(image) {
            frame_indices.insert(frame_idx);
        } else {
            trivial_images += 1;
        }
    }
    frame_indices.len() + trivial_images
}

fn bogus_registered_camera_indices(
    reconstruction: &Reconstruction,
    config: &MapperConfig,
) -> Vec<usize> {
    let mut indices = BTreeSet::new();
    for (image, pose) in reconstruction.poses.iter().enumerate() {
        if pose.is_none() {
            continue;
        }
        let camera = reconstruction.camera_for_image(image);
        if camera_has_bogus_params(camera, config) {
            let camera_idx = reconstruction
                .image_camera_indices
                .get(image)
                .copied()
                .unwrap_or(0);
            indices.insert(camera_idx);
        }
    }
    indices.into_iter().collect()
}

fn apply_image_camera(reconstruction: &mut Reconstruction, image: usize, camera: CameraModel) {
    if let Some(&camera_idx) = reconstruction.image_camera_indices.get(image) {
        if let Some(slot) = reconstruction.cameras.get_mut(camera_idx) {
            *slot = camera;
            if camera_idx == 0 {
                reconstruction.camera = camera;
            }
            return;
        }
    }
    reconstruction.camera = camera;
    if let Some(first) = reconstruction.cameras.first_mut() {
        *first = camera;
    }
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
    let candidates = local_pair_candidates(
        frames.len(),
        config.local_window,
        config.experimental_sequence_heuristics,
    );
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
    if config.experimental_ring_closure || std::env::var_os("RUSTSFM_RING_CLOSURE").is_some() {
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
    if config.experimental_sequence_heuristics {
        enforce_adjacent_translation_continuity(&mut pairs);
        regularize_low_parallax_adjacent_translations(&mut pairs);
        filter_translation_outlier_pairs(&mut pairs);
    }
    Ok(pairs)
}

fn local_pair_candidates(
    frame_count: usize,
    local_window: usize,
    experimental_sequence_heuristics: bool,
) -> Vec<(usize, usize)> {
    let mut candidates = Vec::new();
    if frame_count < 2 || local_window == 0 {
        return candidates;
    }
    for offset in 1..=local_window.min(frame_count - 1) {
        for left in 0..frame_count - offset {
            candidates.push((left, left + offset));
        }
    }
    if experimental_sequence_heuristics {
        add_segment_bridge_candidates(frame_count, &mut candidates);
    }
    candidates
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
        two_view_config: geometry.config,
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
        let Some(left_xy) = left_camera.cam_from_img_f32(lk.x(), lk.y()) else {
            continue;
        };
        let Some(right_xy) = right_camera.cam_from_img_f32(rk.x(), rk.y()) else {
            continue;
        };
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
    if matches!(
        pair.two_view_config,
        crate::database::COLMAP_TWO_VIEW_UNDEFINED
            | crate::database::COLMAP_TWO_VIEW_DEGENERATE
            | crate::database::COLMAP_TWO_VIEW_WATERMARK
            | crate::database::COLMAP_TWO_VIEW_MULTIPLE
    ) {
        return false;
    }
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

fn pair_config_summary(pairs: &[PairGeometry]) -> Vec<String> {
    if pairs.is_empty() {
        return Vec::new();
    }
    let mut counts = pairs
        .iter()
        .fold(HashMap::<i32, usize>::new(), |mut counts, pair| {
            *counts.entry(pair.two_view_config).or_default() += 1;
            counts
        });
    let mut configs = counts.keys().copied().collect::<Vec<_>>();
    configs.sort_unstable();
    let parts = configs
        .into_iter()
        .map(|config| {
            let count = counts.remove(&config).unwrap_or(0);
            format!("{}={}", colmap_two_view_config_name(config), count)
        })
        .collect::<Vec<_>>();
    vec![format!("pair_config {}", parts.join(" "))]
}

fn colmap_two_view_config_name(config: i32) -> &'static str {
    match config {
        crate::database::COLMAP_TWO_VIEW_UNDEFINED => "UNDEFINED",
        crate::database::COLMAP_TWO_VIEW_DEGENERATE => "DEGENERATE",
        crate::database::COLMAP_TWO_VIEW_CALIBRATED => "CALIBRATED",
        crate::database::COLMAP_TWO_VIEW_UNCALIBRATED => "UNCALIBRATED",
        crate::database::COLMAP_TWO_VIEW_PLANAR => "PLANAR",
        crate::database::COLMAP_TWO_VIEW_PANORAMIC => "PANORAMIC",
        crate::database::COLMAP_TWO_VIEW_PLANAR_OR_PANORAMIC => "PLANAR_OR_PANORAMIC",
        crate::database::COLMAP_TWO_VIEW_WATERMARK => "WATERMARK",
        crate::database::COLMAP_TWO_VIEW_MULTIPLE => "MULTIPLE",
        crate::database::COLMAP_TWO_VIEW_CALIBRATED_RIG => "CALIBRATED_RIG",
        _ => "UNKNOWN",
    }
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
    let (
        cameras,
        camera_ids,
        camera_has_prior_focal_length,
        rigs,
        rig_frames,
        image_ids,
        image_camera_indices,
        image_frame_indices,
    ) = if let Some(setup) = reference_camera_setup {
        (
            setup.cameras.clone(),
            setup.camera_ids.clone(),
            setup.camera_has_prior_focal_length.clone(),
            setup.rigs.clone(),
            setup.frames.clone(),
            setup.image_ids.clone(),
            setup.image_camera_indices.clone(),
            setup.image_frame_indices.clone(),
        )
    } else {
        (
            vec![camera],
            vec![1],
            vec![true],
            Vec::new(),
            Vec::new(),
            (0..frames.len()).map(|idx| idx as u32 + 1).collect(),
            vec![0; frames.len()],
            vec![None; frames.len()],
        )
    };
    let camera_priors = cameras.clone();
    let mut reconstruction = Reconstruction {
        camera,
        cameras,
        camera_ids,
        rigs,
        frames: rig_frames,
        image_names: frames.iter().map(|f| f.name.clone()).collect(),
        image_paths: frames.iter().map(|f| f.path.clone()).collect(),
        image_ids,
        image_camera_indices,
        image_frame_indices,
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
    let initial = choose_initial_pair(pairs, &reconstruction, config).context("no initial pair")?;
    debug_log.push(format!(
        "initial_pair {} -> {} inliers={} triangulated={}",
        frames[initial.left].name,
        frames[initial.right].name,
        initial.inliers,
        initial.triangulated
    ));
    {
        let mut observation_manager = ObservationManager::new(frames, pairs, &reconstruction);
        observation_manager.register_image(
            frames,
            pairs,
            &mut reconstruction,
            initial.left,
            SE3::identity(),
        );
        observation_manager.register_image(
            frames,
            pairs,
            &mut reconstruction,
            initial.right,
            initial.relative_pose,
        );
    }
    let mut registration_stats = RegistrationStats::from_reconstruction(&reconstruction);
    let gauge_image = initial.left;
    let tri_options =
        IncrementalTriangulatorOptions::from_mapper_threshold(config.max_reprojection_error_px);
    {
        let mut triangulator = IncrementalTriangulator::new(frames, pairs, &mut reconstruction);
        triangulator.triangulate_image(&tri_options, initial.left);
        triangulator.triangulate_image(&tri_options, initial.right);
        let modified = triangulator.get_modified_points3d().clone();
        triangulator.complete_tracks(&tri_options, &modified);
        let modified = triangulator.get_modified_points3d().clone();
        triangulator.merge_tracks(&tri_options, &modified);
        triangulator.retriangulate(&tri_options);
    }
    filter_reprojection_tracks(frames, pairs, &mut reconstruction, config);
    let mut global_ba_schedule = GlobalBaSchedule::new(&reconstruction);
    if refine_global_bundle_with_postprocessing(
        frames,
        pairs,
        &mut reconstruction,
        &tri_options,
        config,
        "initial",
        &mut debug_log,
    ) {
        global_ba_schedule.mark(&reconstruction);
    }

    let mut reg_trials = vec![0usize; frames.len()];
    while reconstruction.poses.iter().any(|p| p.is_none()) {
        let Some(choice) = choose_next_registration(
            frames,
            pairs,
            &reconstruction,
            &reg_trials,
            config,
            &camera_priors,
            &camera_has_prior_focal_length,
            &registration_stats,
        ) else {
            break;
        };
        let registration_snapshot = reconstruction.clone();
        let registration_log = format!(
            "register {} source={} pnp_inliers={} inlier_ratio={:.3} visible_points={} visible_ratio={:.3} mean_error={:.3} pair_rot_error={:.3}",
            frames[choice.image].name,
            choice.source,
            choice.pnp_inliers,
            choice.inlier_ratio,
            choice.visible_points,
            choice.visible_points_ratio,
            choice.mean_error_px,
            choice.pair_rot_error
        );
        apply_image_camera(&mut reconstruction, choice.image, choice.camera);
        {
            let mut observation_manager = ObservationManager::new(frames, pairs, &reconstruction);
            observation_manager.register_image(
                frames,
                pairs,
                &mut reconstruction,
                choice.image,
                choice.pose,
            );
        }
        let structureless_track_report = if !choice.structureless_inliers.is_empty() {
            continue_or_triangulate_structureless_tracks(
                frames,
                pairs,
                &mut reconstruction,
                &choice.structureless_inliers,
                &tri_options,
                config,
            )
        } else {
            TriangulationReport::default()
        };
        {
            let mut triangulator = IncrementalTriangulator::new(frames, pairs, &mut reconstruction);
            triangulator.triangulate_image(&tri_options, choice.image);
            let modified = triangulator.get_modified_points3d().clone();
            triangulator.complete_tracks(&tri_options, &modified);
            let modified = triangulator.get_modified_points3d().clone();
            triangulator.merge_tracks(&tri_options, &modified);
            triangulator.retriangulate(&tri_options);
        }
        filter_reprojection_tracks(frames, pairs, &mut reconstruction, config);
        let local_ba_required =
            local_bundle_refinement_required(&reconstruction, choice.image, gauge_image, config);
        let local_ba_report = refine_local_bundle_after_registration(
            frames,
            pairs,
            &mut reconstruction,
            choice.image,
            gauge_image,
            &tri_options,
            config,
            &registration_stats,
        );
        let mut local_ba_filter_removed = 0usize;
        if local_ba_report.is_some() {
            local_ba_filter_removed =
                filter_reprojection_tracks(frames, pairs, &mut reconstruction, config);
        }
        let rollback_reason = registration_rollback_reason(
            &reconstruction,
            choice.image,
            local_ba_required,
            local_ba_report.is_some(),
            config,
        );
        if let Some(reason) = rollback_reason {
            reconstruction = registration_snapshot;
            reg_trials[choice.image] += 1;
            debug_log.push(format!(
                "registration_rollback {} reason={reason}",
                frames[choice.image].name
            ));
            continue;
        }
        registration_stats.register_frame_for_image_event(&reconstruction, choice.image);
        reset_registration_unit_trials(&reconstruction, choice.image, &mut reg_trials);
        debug_log.push(registration_log);
        if structureless_track_report.total_observations() > 0 {
            debug_log.push(format!(
                "structureless_tracks image={} created={} continued={}",
                frames[choice.image].name,
                structureless_track_report.created_points,
                structureless_track_report.continued_observations
            ));
        }
        if let Some(report) = local_ba_report {
            debug_log.push(format!(
                "local_ba image={} local_images={} variable_images={} points={} observations={} residuals={} cost={:.6}->{:.6} iterations={}/{} termination={:?} reason={:?} merged={} completed={} image_completed={}",
                frames[choice.image].name,
                report.local_images,
                report.variable_images,
                report.points,
                report.report.observations,
                report.report.residuals,
                report.report.initial_cost,
                report.report.final_cost,
                report.report.iterations,
                report.report.attempted_iterations,
                report.report.termination_type,
                report.report.termination_reason,
                report.merged_observations,
                report.completed_observations,
                report.completed_image_observations
            ));
            if local_ba_filter_removed > 0 {
                debug_log.push(format!(
                    "local_ba_filter removed_observations={local_ba_filter_removed}"
                ));
            }
        }
        if should_run_global_ba(&global_ba_schedule, &reconstruction, config) {
            if refine_global_bundle_with_postprocessing(
                frames,
                pairs,
                &mut reconstruction,
                &tri_options,
                config,
                "scheduled",
                &mut debug_log,
            ) {
                global_ba_schedule.mark(&reconstruction);
            }
        }
        mark_unregistered_images_with_no_absolute_pose(
            frames,
            pairs,
            &reconstruction,
            &mut reg_trials,
            config,
            &camera_priors,
            &camera_has_prior_focal_length,
            &registration_stats,
        );
    }
    if should_run_final_global_ba(&global_ba_schedule, &reconstruction, config) {
        refine_global_bundle_with_postprocessing(
            frames,
            pairs,
            &mut reconstruction,
            &tri_options,
            config,
            "final",
            &mut debug_log,
        );
    }
    sync_registered_frame_poses_from_images(&mut reconstruction);
    Ok((reconstruction, debug_log))
}

#[derive(Debug, Clone)]
struct RegistrationChoice {
    image: usize,
    pose: SE3,
    camera: CameraModel,
    source: &'static str,
    pnp_inliers: usize,
    inlier_ratio: f32,
    visible_points: usize,
    visible_points_ratio: f32,
    mean_error_px: f32,
    pair_rot_error: f32,
    structureless_inliers: Vec<StructurelessInlier>,
}

fn choose_next_registration(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    reg_trials: &[usize],
    config: &MapperConfig,
    camera_priors: &[CameraModel],
    camera_has_prior_focal_length: &[bool],
    registration_stats: &RegistrationStats,
) -> Option<RegistrationChoice> {
    let mut best = None::<(f32, RegistrationChoice)>;
    let obs_manager = ObservationManager::new(frames, pairs, reconstruction);
    for image in 0..reconstruction.poses.len() {
        if registration_unit_is_registered(reconstruction, image) {
            continue;
        }
        if reg_trials.get(image).copied().unwrap_or(0) >= config.max_reg_trials {
            continue;
        }
        let (abs_pose, source) = if let Some(abs_pose) = solve_absolute_pose(
            image,
            frames,
            pairs,
            reconstruction,
            config,
            camera_priors,
            camera_has_prior_focal_length,
            registration_stats,
        ) {
            (abs_pose, "pnp")
        } else {
            let Some(abs_pose) = solve_structureless_absolute_pose(
                image,
                frames,
                pairs,
                reconstruction,
                config,
                &obs_manager,
                camera_priors,
                registration_stats,
            ) else {
                continue;
            };
            (abs_pose, "structureless")
        };
        let pair_rot_error =
            registered_pair_rotation_error(image, abs_pose.pose, pairs, reconstruction);
        if !pair_rot_error.is_finite() || pair_rot_error > absolute_pose_pair_rotation_limit_deg() {
            continue;
        }
        let visible_points = obs_manager.num_visible_points3d(image);
        let num_observations = obs_manager.num_observations(image).max(1);
        let visible_points_ratio = visible_points as f32 / num_observations as f32;
        let choice = RegistrationChoice {
            image,
            pose: abs_pose.pose,
            camera: abs_pose.camera,
            source,
            pnp_inliers: abs_pose.inliers,
            inlier_ratio: abs_pose.inlier_ratio,
            visible_points,
            visible_points_ratio,
            mean_error_px: abs_pose.mean_error_px,
            pair_rot_error,
            structureless_inliers: abs_pose.structureless_inliers,
        };
        let score = registration_score(&choice, &obs_manager);
        if best.as_ref().map(|(s, _)| score > *s).unwrap_or(true) {
            best = Some((score, choice));
        }
    }
    best.map(|(_, choice)| choice)
}

fn mark_unregistered_images_with_no_absolute_pose(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    reg_trials: &mut [usize],
    config: &MapperConfig,
    camera_priors: &[CameraModel],
    camera_has_prior_focal_length: &[bool],
    registration_stats: &RegistrationStats,
) {
    let obs_manager = ObservationManager::new(frames, pairs, reconstruction);
    for image in 0..reconstruction.poses.len() {
        if registration_unit_is_registered(reconstruction, image)
            || reg_trials.get(image).copied().unwrap_or(0) >= config.max_reg_trials
        {
            continue;
        }
        let has_pose = solve_absolute_pose(
            image,
            frames,
            pairs,
            reconstruction,
            config,
            camera_priors,
            camera_has_prior_focal_length,
            registration_stats,
        )
        .or_else(|| {
            solve_structureless_absolute_pose(
                image,
                frames,
                pairs,
                reconstruction,
                config,
                &obs_manager,
                camera_priors,
                registration_stats,
            )
        })
        .map(|pose| {
            let pair_rot_error =
                registered_pair_rotation_error(image, pose.pose, pairs, reconstruction);
            pair_rot_error.is_finite() && pair_rot_error <= absolute_pose_pair_rotation_limit_deg()
        })
        .unwrap_or(false);
        if !has_pose {
            reg_trials[image] += 1;
        }
    }
}

fn registration_unit_is_registered(reconstruction: &Reconstruction, image: usize) -> bool {
    reconstruction
        .image_indices_for_registration_unit(image)
        .iter()
        .any(|&frame_image| {
            reconstruction
                .poses
                .get(frame_image)
                .copied()
                .flatten()
                .is_some()
        })
}

fn reset_registration_unit_trials(
    reconstruction: &Reconstruction,
    image: usize,
    reg_trials: &mut [usize],
) {
    for frame_image in reconstruction.image_indices_for_registration_unit(image) {
        if let Some(trials) = reg_trials.get_mut(frame_image) {
            *trials = 0;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LocalBundleReport {
    report: crate::ba::BundleAdjustmentReport,
    variable_images: usize,
    local_images: usize,
    points: usize,
    merged_observations: usize,
    completed_observations: usize,
    completed_image_observations: usize,
}

#[derive(Debug, Clone, Copy)]
struct GlobalBaSchedule {
    prev_registered_images: usize,
    prev_registered_frames: usize,
    prev_points: usize,
}

impl GlobalBaSchedule {
    fn new(reconstruction: &Reconstruction) -> Self {
        Self {
            prev_registered_images: registered_image_count(reconstruction),
            prev_registered_frames: registered_frame_count(reconstruction),
            prev_points: reconstruction.points.len(),
        }
    }

    fn mark(&mut self, reconstruction: &Reconstruction) {
        self.prev_registered_images = registered_image_count(reconstruction);
        self.prev_registered_frames = registered_frame_count(reconstruction);
        self.prev_points = reconstruction.points.len();
    }
}

fn refine_global_bundle_with_postprocessing(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    tri_options: &IncrementalTriangulatorOptions,
    config: &MapperConfig,
    reason: &str,
    debug_log: &mut Vec<String>,
) -> bool {
    if !global_ba_enabled(config)
        || registered_image_count(reconstruction) < 2
        || reconstruction.points.is_empty()
    {
        return false;
    }

    let (pre_completed, pre_merged, retriangulated) = {
        let mut triangulator = IncrementalTriangulator::new(frames, pairs, reconstruction);
        let completed = triangulator.complete_all_tracks(tri_options);
        let merged = triangulator.merge_all_tracks(tri_options);
        let retriangulated = triangulator.retriangulate(tri_options);
        (completed, merged, retriangulated)
    };
    if pre_completed > 0 || pre_merged > 0 || retriangulated > 0 {
        debug_log.push(format!(
            "global_ba_prepare reason={reason} completed={} merged={} retriangulated={}",
            pre_completed, pre_merged, retriangulated
        ));
    }

    let mut attempted = false;
    for round in 0..config.global_ba_max_refinements {
        let observations_before = reconstruction_num_observations(reconstruction);
        if observations_before == 0 {
            break;
        }
        let gauge_images = global_ba_gauge_images(reconstruction);
        if gauge_images.is_empty() {
            break;
        }
        attempted = true;
        let mut ba_options = mapper_ba_options(
            config,
            reconstruction,
            global_ba_iterations(config),
            None,
            Vec::new(),
            None,
            None,
        );
        ba_options.gauge = crate::ba::BundleAdjustmentGauge::TwoCamsFromWorld;
        let Some(report) =
            refine_bundle_adjustment_checked(frames, reconstruction, config, ba_options)
        else {
            debug_log.push(format!(
                "global_ba reason={reason} round={} skipped gauge_images={:?} observations={}",
                round + 1,
                gauge_images,
                observations_before
            ));
            break;
        };

        let (completed, merged) = {
            let mut triangulator = IncrementalTriangulator::new(frames, pairs, reconstruction);
            let completed = triangulator.complete_all_tracks(tri_options);
            let merged = triangulator.merge_all_tracks(tri_options);
            (completed, merged)
        };
        let filtered = filter_reprojection_tracks(frames, pairs, reconstruction, config);
        let changed = (completed + merged + filtered) as f32 / observations_before.max(1) as f32;
        debug_log.push(format!(
            "global_ba reason={reason} round={} size={} gauge_images={:?} observations={} residuals={} cost={:.6}->{:.6} iterations={}/{} termination={:?} termination_reason={:?} completed={} merged={} filtered={} changed={:.6}",
            round + 1,
            global_ba_size_tag(reconstruction, config),
            gauge_images,
            report.observations,
            report.residuals,
            report.initial_cost,
            report.final_cost,
            report.iterations,
            report.attempted_iterations,
            report.termination_type,
            report.termination_reason,
            completed,
            merged,
            filtered,
            changed
        ));
        if changed <= config.global_ba_max_refinement_change {
            break;
        }
    }
    attempted
}

fn should_run_global_ba(
    schedule: &GlobalBaSchedule,
    reconstruction: &Reconstruction,
    config: &MapperConfig,
) -> bool {
    if !global_ba_enabled(config)
        || registered_image_count(reconstruction) < 2
        || reconstruction.points.is_empty()
    {
        return false;
    }
    let registered = registered_frame_count(reconstruction);
    let points = reconstruction.points.len();
    let image_freq_hit = config.global_ba_images_freq > 0
        && registered
            >= schedule
                .prev_registered_frames
                .saturating_add(config.global_ba_images_freq);
    let point_freq_hit = config.global_ba_points_freq > 0
        && points
            >= schedule
                .prev_points
                .saturating_add(config.global_ba_points_freq);
    let image_ratio_hit = config.global_ba_images_ratio > 1.0
        && schedule.prev_registered_frames > 0
        && registered as f32
            >= schedule.prev_registered_frames as f32 * config.global_ba_images_ratio;
    let point_ratio_hit = config.global_ba_points_ratio > 1.0
        && schedule.prev_points > 0
        && points as f32 >= schedule.prev_points as f32 * config.global_ba_points_ratio;
    image_freq_hit || point_freq_hit || image_ratio_hit || point_ratio_hit
}

fn should_run_final_global_ba(
    schedule: &GlobalBaSchedule,
    reconstruction: &Reconstruction,
    config: &MapperConfig,
) -> bool {
    global_ba_enabled(config)
        && registered_image_count(reconstruction) >= 2
        && !reconstruction.points.is_empty()
        && (registered_frame_count(reconstruction) != schedule.prev_registered_frames
            || reconstruction.points.len() != schedule.prev_points)
}

fn global_ba_enabled(config: &MapperConfig) -> bool {
    config.global_ba && global_ba_iterations(config) > 0 && config.global_ba_max_refinements > 0
}

fn global_ba_gauge_images(reconstruction: &Reconstruction) -> Vec<usize> {
    let mut gauge_images = Vec::new();
    let mut frame_indices = HashSet::new();
    for (image, pose) in reconstruction.poses.iter().enumerate() {
        if pose.is_none() {
            continue;
        }
        if let Some(frame_idx) = reconstruction.frame_index_for_image(image) {
            if frame_indices.insert(frame_idx) {
                gauge_images.push(image);
            }
        } else {
            gauge_images.push(image);
        }
        if gauge_images.len() >= 2 {
            break;
        }
    }
    gauge_images
}

fn global_ba_size_tag(reconstruction: &Reconstruction, config: &MapperConfig) -> &'static str {
    if (config.global_ba_images_freq > 0
        && registered_frame_count(reconstruction) >= config.global_ba_images_freq)
        || (config.global_ba_points_freq > 0
            && reconstruction.points.len() >= config.global_ba_points_freq)
    {
        "large"
    } else {
        "small"
    }
}

fn registered_image_count(reconstruction: &Reconstruction) -> usize {
    reconstruction
        .poses
        .iter()
        .filter(|pose| pose.is_some())
        .count()
}

fn reconstruction_num_observations(reconstruction: &Reconstruction) -> usize {
    reconstruction
        .points
        .iter()
        .map(|point| point.track.len())
        .sum()
}

fn refine_local_bundle_after_registration(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    registered_image: usize,
    gauge_image: usize,
    tri_options: &IncrementalTriangulatorOptions,
    config: &MapperConfig,
    registration_stats: &RegistrationStats,
) -> Option<LocalBundleReport> {
    if !config.local_ba || config.local_ba_iterations == 0 {
        return None;
    }
    let local_bundle = select_local_bundle(
        reconstruction,
        registered_image,
        gauge_image,
        config.local_ba_num_images,
        config.local_ba_min_shared_points,
    )?;
    let ba_options = mapper_local_ba_options(
        config,
        reconstruction,
        registration_stats,
        config.local_ba_iterations,
        local_bundle.variable_images.clone(),
        vec![gauge_image],
        Some(local_bundle.point_ids.clone()),
        Some(local_bundle.constant_point_ids.clone()),
    );
    let report = refine_bundle_adjustment_checked(frames, reconstruction, config, ba_options)?;
    let stable_point_ids = local_bundle.stable_point_ids.clone();
    let mut post_ba_point_ids =
        point_indices_for_stable_point_ids(reconstruction, &stable_point_ids);
    let (merged_observations, modified_after_merge) = {
        let mut triangulator = IncrementalTriangulator::new(frames, pairs, reconstruction);
        let merged = triangulator.merge_tracks(tri_options, &post_ba_point_ids);
        let modified = triangulator.get_modified_points3d().clone();
        (merged, modified)
    };
    post_ba_point_ids = point_indices_for_stable_point_ids(reconstruction, &stable_point_ids);
    post_ba_point_ids.extend(modified_after_merge);
    let (completed_observations, completed_image_observations) = {
        let mut triangulator = IncrementalTriangulator::new(frames, pairs, reconstruction);
        let completed = triangulator.complete_tracks(tri_options, &post_ba_point_ids);
        let image_report = triangulator.triangulate_image(tri_options, registered_image);
        (completed, image_report.total_observations())
    };
    let variable_image_count =
        expand_images_to_registration_frames(reconstruction, &local_bundle.variable_images).len();
    Some(LocalBundleReport {
        report,
        variable_images: variable_image_count,
        local_images: local_bundle.local_images.len(),
        points: local_bundle.point_ids.len(),
        merged_observations,
        completed_observations,
        completed_image_observations,
    })
}

fn local_bundle_refinement_required(
    reconstruction: &Reconstruction,
    registered_image: usize,
    gauge_image: usize,
    config: &MapperConfig,
) -> bool {
    if !config.local_ba || config.local_ba_iterations == 0 {
        return false;
    }
    select_local_bundle(
        reconstruction,
        registered_image,
        gauge_image,
        config.local_ba_num_images,
        config.local_ba_min_shared_points,
    )
    .is_some()
}

fn registration_state_has_bogus_camera(
    reconstruction: &Reconstruction,
    image: usize,
    config: &MapperConfig,
) -> bool {
    reconstruction.poses.get(image).copied().flatten().is_some()
        && camera_has_bogus_params(reconstruction.camera_for_image(image), config)
}

fn registration_rollback_reason(
    reconstruction: &Reconstruction,
    image: usize,
    local_ba_required: bool,
    local_ba_succeeded: bool,
    config: &MapperConfig,
) -> Option<&'static str> {
    if local_ba_required && !local_ba_succeeded {
        Some("local_ba_failed")
    } else if registration_state_has_bogus_camera(reconstruction, image, config) {
        Some("bogus_camera")
    } else {
        None
    }
}

#[derive(Debug, Clone)]
struct LocalBundleSelection {
    variable_images: Vec<usize>,
    point_ids: Vec<usize>,
    constant_point_ids: Vec<usize>,
    stable_point_ids: Vec<u64>,
    local_images: Vec<usize>,
}

fn select_local_bundle(
    reconstruction: &Reconstruction,
    registered_image: usize,
    gauge_image: usize,
    max_num_images: usize,
    min_shared_points: usize,
) -> Option<LocalBundleSelection> {
    if registered_image >= reconstruction.poses.len()
        || reconstruction.poses[registered_image].is_none()
        || max_num_images == 0
    {
        return None;
    }

    let query_point_ids = reconstruction
        .points
        .iter()
        .enumerate()
        .filter_map(|(point_id, point)| {
            point
                .track
                .iter()
                .any(|obs| obs.image == registered_image)
                .then_some(point_id)
        })
        .collect::<Vec<_>>();
    let mut query_point_ids = query_point_ids;
    query_point_ids.sort_unstable();
    query_point_ids.dedup();
    if query_point_ids.len() < min_shared_points.max(2) {
        return None;
    }

    let mut shared = HashMap::<usize, Vec<usize>>::new();
    for &point_id in &query_point_ids {
        let Some(point) = reconstruction.points.get(point_id) else {
            continue;
        };
        for obs in &point.track {
            if obs.image == registered_image
                || obs.image == gauge_image
                || reconstruction
                    .poses
                    .get(obs.image)
                    .copied()
                    .flatten()
                    .is_none()
            {
                continue;
            }
            shared.entry(obs.image).or_default().push(point_id);
        }
    }

    let mut variable_images = vec![registered_image];
    let mut local_images = Vec::new();
    let mut neighbors = shared.into_iter().collect::<Vec<_>>();
    neighbors.sort_by(|(image_a, points_a), (image_b, points_b)| {
        points_b
            .len()
            .cmp(&points_a.len())
            .then_with(|| image_a.cmp(image_b))
    });
    let desired_neighbors = max_num_images.saturating_sub(1).min(neighbors.len());
    let mut used_neighbors = vec![false; neighbors.len()];
    if desired_neighbors > 0 {
        for (angle_divisor, min_shared_ratio) in local_bundle_selection_thresholds() {
            let min_angle = local_ba_min_tri_angle_deg() / angle_divisor;
            let min_shared = ((query_point_ids.len() as f32) * min_shared_ratio)
                .ceil()
                .max(min_shared_points as f32) as usize;
            for idx in 0..neighbors.len() {
                if used_neighbors[idx] {
                    continue;
                }
                let (image, shared_points) = &neighbors[idx];
                if shared_points.len() < min_shared {
                    break;
                }
                if shared_track_triangulation_percentile_deg(
                    reconstruction,
                    registered_image,
                    *image,
                    shared_points,
                    75.0,
                )
                .is_some_and(|angle| angle >= min_angle)
                {
                    local_images.push(*image);
                    variable_images.push(*image);
                    used_neighbors[idx] = true;
                    if local_images.len() >= desired_neighbors {
                        break;
                    }
                }
            }
            if local_images.len() >= desired_neighbors {
                break;
            }
        }
        if local_images.len() < desired_neighbors {
            for idx in 0..neighbors.len() {
                if used_neighbors[idx] {
                    continue;
                }
                let (image, _) = &neighbors[idx];
                local_images.push(*image);
                variable_images.push(*image);
                used_neighbors[idx] = true;
                if local_images.len() >= desired_neighbors {
                    break;
                }
            }
        }
    }
    variable_images.sort_unstable();
    variable_images.dedup();
    if local_images.is_empty() || (variable_images.len() == 1 && query_point_ids.len() < 6) {
        return None;
    }

    let mut point_ids = query_point_ids
        .iter()
        .copied()
        .filter(|&point_id| {
            reconstruction
                .points
                .get(point_id)
                .map(|point| {
                    point.error <= 0.0 || point.track.len() <= local_ba_max_variable_track_length()
                })
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    if point_ids.len() < min_shared_points.max(2) {
        point_ids = query_point_ids.clone();
    }
    let variable_point_set = point_ids.iter().copied().collect::<HashSet<_>>();
    let mut constant_point_ids = query_point_ids
        .iter()
        .copied()
        .filter(|point_id| !variable_point_set.contains(point_id))
        .collect::<Vec<_>>();
    constant_point_ids.sort_unstable();
    let stable_point_ids = point_ids
        .iter()
        .chain(constant_point_ids.iter())
        .map(|&point_id| reconstruction_point3d_id(reconstruction, point_id))
        .collect::<Vec<_>>();

    Some(LocalBundleSelection {
        variable_images,
        point_ids,
        constant_point_ids,
        stable_point_ids,
        local_images,
    })
}

fn point_indices_for_stable_point_ids(
    reconstruction: &Reconstruction,
    stable_point_ids: &[u64],
) -> HashSet<usize> {
    let ids = stable_point_ids.iter().copied().collect::<HashSet<_>>();
    reconstruction
        .points
        .iter()
        .enumerate()
        .filter_map(|(point_id, _)| {
            ids.contains(&reconstruction_point3d_id(reconstruction, point_id))
                .then_some(point_id)
        })
        .collect()
}

fn reconstruction_point3d_id(reconstruction: &Reconstruction, point_id: usize) -> u64 {
    reconstruction
        .point_ids
        .get(point_id)
        .copied()
        .unwrap_or(point_id as u64 + 1)
}

fn local_bundle_selection_thresholds() -> [(f32, f32); 8] {
    [
        (1.0, 0.6),
        (1.5, 0.6),
        (2.0, 0.5),
        (2.5, 0.4),
        (3.0, 0.3),
        (4.0, 0.2),
        (5.0, 0.1),
        (6.0, 0.1),
    ]
}

fn local_ba_min_tri_angle_deg() -> f32 {
    std::env::var("RUSTSFM_LOCAL_BA_MIN_TRI_ANGLE_DEG")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(6.0)
}

fn local_ba_max_variable_track_length() -> usize {
    std::env::var("RUSTSFM_LOCAL_BA_MAX_VARIABLE_TRACK_LENGTH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 2)
        .unwrap_or(15)
}

fn shared_track_triangulation_percentile_deg(
    reconstruction: &Reconstruction,
    left_image: usize,
    right_image: usize,
    point_ids: &[usize],
    percentile: f32,
) -> Option<f32> {
    let left_pose = reconstruction.poses.get(left_image).copied().flatten()?;
    let right_pose = reconstruction.poses.get(right_image).copied().flatten()?;
    let mut angles = point_ids
        .iter()
        .filter_map(|&point_id| {
            let point = reconstruction.points.get(point_id)?;
            pair_triangulation_angle_deg(left_pose, right_pose, point.xyz)
        })
        .filter(|angle| angle.is_finite())
        .collect::<Vec<_>>();
    percentile_f32(&mut angles, percentile)
}

fn percentile_f32(values: &mut [f32], percentile: f32) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p = percentile.clamp(0.0, 100.0) / 100.0;
    let idx = ((values.len() - 1) as f32 * p).round() as usize;
    values.get(idx).copied()
}

fn absolute_pose_pair_rotation_limit_deg() -> f32 {
    std::env::var("RUSTSFM_ABS_POSE_PAIR_ROTATION_LIMIT_DEG")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(20.0)
}

fn choose_initial_pair<'a>(
    pairs: &'a [PairGeometry],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
) -> Option<&'a PairGeometry> {
    let image_correspondences = image_correspondence_counts(pairs);
    pairs
        .iter()
        .filter(|p| is_colmap_style_initial_pair(p, reconstruction, config))
        .max_by(|a, b| compare_initial_pairs(a, b, &image_correspondences))
}

fn is_colmap_style_initial_pair(
    pair: &PairGeometry,
    reconstruction: &Reconstruction,
    config: &MapperConfig,
) -> bool {
    pair.inliers >= config.init_min_num_inliers
        && pair.median_triangulation_angle_deg > config.init_min_tri_angle_deg
        && pair.triangulated >= config.min_triangulated
        && initial_pair_forward_motion(pair) < config.init_max_forward_motion
        && !same_registration_frame(reconstruction, pair.left, pair.right)
}

fn same_registration_frame(reconstruction: &Reconstruction, left: usize, right: usize) -> bool {
    left != right
        && reconstruction
            .frame_index_for_image(left)
            .zip(reconstruction.frame_index_for_image(right))
            .is_some_and(|(left_frame, right_frame)| left_frame == right_frame)
}

fn initial_pair_forward_motion(pair: &PairGeometry) -> f32 {
    let translation = glam::Vec3::from_array(pair.relative_pose.translation());
    let norm = translation.length();
    if norm <= f32::EPSILON || !norm.is_finite() {
        return f32::INFINITY;
    }
    (translation.z / norm).abs()
}

fn image_correspondence_counts(pairs: &[PairGeometry]) -> HashMap<usize, usize> {
    let mut counts = HashMap::new();
    for pair in pairs {
        *counts.entry(pair.left).or_default() += pair.inliers;
        *counts.entry(pair.right).or_default() += pair.inliers;
    }
    counts
}

fn compare_initial_pairs(
    a: &PairGeometry,
    b: &PairGeometry,
    image_correspondences: &HashMap<usize, usize>,
) -> std::cmp::Ordering {
    initial_pair_score(a, image_correspondences)
        .partial_cmp(&initial_pair_score(b, image_correspondences))
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| b.left.cmp(&a.left))
        .then_with(|| b.right.cmp(&a.right))
}

fn initial_pair_score(pair: &PairGeometry, image_correspondences: &HashMap<usize, usize>) -> f32 {
    let left_corrs = *image_correspondences.get(&pair.left).unwrap_or(&0) as f32;
    let right_corrs = *image_correspondences.get(&pair.right).unwrap_or(&0) as f32;
    let pair_corrs = pair.inliers as f32;
    let tri_angle = pair.median_triangulation_angle_deg.max(0.0);
    let triangulated = pair.triangulated as f32;
    let forward_penalty = initial_pair_forward_motion(pair).min(1.0) * 500.0;
    left_corrs.sqrt() * 20.0
        + right_corrs.sqrt() * 20.0
        + pair_corrs * 10.0
        + triangulated * 3.0
        + tri_angle * 30.0
        - forward_penalty
}

fn registration_score(choice: &RegistrationChoice, obs_manager: &ObservationManager) -> f32 {
    let visible_points = obs_manager.num_visible_points3d(choice.image) as f32;
    let visible_corrs = obs_manager.num_visible_correspondences(choice.image) as f32;
    let visibility_score = obs_manager.point3d_visibility_score(choice.image) as f32;
    let support = visible_points * 1000.0
        + choice.pnp_inliers as f32 * 200.0
        + choice.visible_points_ratio * 500.0
        + visible_corrs.sqrt() * 25.0
        + visibility_score as f32 * 3.0;
    support
        - choice.mean_error_px.min(20.0) * 20.0
        - choice.pair_rot_error.min(45.0) * 10.0
        - choice.image as f32 * 0.001
}

#[derive(Debug, Clone)]
struct AbsolutePose {
    pose: SE3,
    camera: CameraModel,
    inliers: usize,
    inlier_ratio: f32,
    mean_error_px: f32,
    structureless_inliers: Vec<StructurelessInlier>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct StructurelessInlier {
    image: usize,
    feature: usize,
    other: usize,
    other_feature: usize,
}

#[derive(Debug, Clone)]
struct ColmapStructurelessProblem {
    query_points2d: Vec<[f64; 2]>,
    world_points2d: Vec<[f64; 2]>,
    world_camera_idxs: Vec<usize>,
    world_cams_from_world: Vec<SE3>,
    world_cameras: Vec<CameraModel>,
    correspondences: Vec<StructurelessInlier>,
}

#[derive(Debug, Clone)]
struct StructurelessPairConstraint {
    other: usize,
    image_is_left: bool,
    other_pose: SE3,
    relative_pose: SE3,
    candidate_pose: SE3,
    line_origin: glam::Vec3,
    line_direction: glam::Vec3,
    inliers: usize,
    mean_error_px: f32,
    matches: Vec<StructurelessInlier>,
}

fn solve_structureless_absolute_pose(
    image: usize,
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
    obs_manager: &ObservationManager,
    camera_priors: &[CameraModel],
    registration_stats: &RegistrationStats,
) -> Option<AbsolutePose> {
    if let Some(abs_pose) = solve_colmap_structureless_absolute_pose(
        image,
        frames,
        pairs,
        reconstruction,
        config,
        obs_manager,
        camera_priors,
        registration_stats,
    ) {
        return Some(abs_pose);
    }
    if !config.experimental_structureless_pair_pose_fallback {
        return None;
    }
    solve_experimental_structureless_pair_pose_fallback(
        image,
        frames,
        pairs,
        reconstruction,
        config,
        obs_manager,
        camera_priors,
        registration_stats,
    )
}

fn solve_colmap_structureless_absolute_pose(
    image: usize,
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
    obs_manager: &ObservationManager,
    camera_priors: &[CameraModel],
    registration_stats: &RegistrationStats,
) -> Option<AbsolutePose> {
    if registered_image_count(reconstruction) < 2 {
        return None;
    }
    let camera = registration_camera_for_image(
        image,
        reconstruction,
        config,
        camera_priors,
        registration_stats,
    );
    if camera_has_bogus_params(camera, config) {
        return None;
    }
    let min_num_inliers = structureless_min_num_inliers(config);
    if obs_manager.num_visible_correspondences(image) < min_num_inliers {
        return None;
    }

    let problem =
        collect_colmap_structureless_problem(image, frames, pairs, reconstruction, config);
    if problem.world_points2d.len() < min_num_inliers {
        return None;
    }

    let mut options = StructureLessAbsolutePoseEstimationOptions::default();
    options.ransac_options.max_error = 0.5 * config.pnp_threshold_px as f64;
    options.ransac_options.min_inlier_ratio = config.abs_pose_min_inlier_ratio as f64;
    options.ransac_options.random_seed = config.random_seed;
    options.ransac_options.num_threads =
        config.threads.map(|threads| threads as isize).unwrap_or(1);

    let estimate = match estimate_structureless_absolute_pose(
        &options,
        StructureLessAbsolutePoseProblem {
            query_points2d: &problem.query_points2d,
            world_points2d: &problem.world_points2d,
            world_camera_idxs: &problem.world_camera_idxs,
            world_cams_from_world: &problem.world_cams_from_world,
            world_cameras: &problem.world_cameras,
            query_camera: camera,
        },
    ) {
        Ok(Some(estimate)) => estimate,
        Ok(None) => return None,
        Err(GeneralizedPoseError::MissingGeneralizedRelativePoseSolver) => {
            log::debug!(
                "COLMAP structure-less registration skipped: GR6P/GR8P solver is not ported yet"
            );
            return None;
        }
        Err(err) => {
            log::debug!("COLMAP structure-less registration skipped: {err}");
            return None;
        }
    };

    if estimate.num_inliers < min_num_inliers {
        return None;
    }
    let structureless_inliers = estimate
        .inlier_mask
        .iter()
        .zip(problem.correspondences.iter())
        .filter_map(|(&is_inlier, &correspondence)| is_inlier.then_some(correspondence))
        .collect::<Vec<_>>();
    if structureless_inliers.len() < min_num_inliers {
        return None;
    }

    Some(AbsolutePose {
        pose: estimate.query_cam_from_world,
        camera,
        inliers: structureless_inliers.len(),
        inlier_ratio: structureless_inliers.len() as f32 / problem.world_points2d.len() as f32,
        mean_error_px: 0.0,
        structureless_inliers,
    })
}

fn collect_colmap_structureless_problem(
    image: usize,
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
) -> ColmapStructurelessProblem {
    let mut problem = ColmapStructurelessProblem {
        query_points2d: Vec::new(),
        world_points2d: Vec::new(),
        world_camera_idxs: Vec::new(),
        world_cams_from_world: Vec::new(),
        world_cameras: Vec::new(),
        correspondences: Vec::new(),
    };
    let mut world_image_to_camera_idx = HashMap::new();

    for pair in pairs {
        let Some((other, image_is_left)) = structureless_pair_side(image, pair) else {
            continue;
        };
        let Some(other_pose) = reconstruction.poses.get(other).copied().flatten() else {
            continue;
        };
        let other_camera = reconstruction.camera_for_image(other);
        if camera_has_bogus_params(other_camera, config) {
            continue;
        }

        let world_camera_idx = if let Some(&camera_idx) = world_image_to_camera_idx.get(&other) {
            camera_idx
        } else {
            let camera_idx = problem.world_cameras.len();
            world_image_to_camera_idx.insert(other, camera_idx);
            problem.world_cams_from_world.push(other_pose);
            problem.world_cameras.push(other_camera);
            camera_idx
        };

        for m in &pair.inlier_matches {
            let (feature, other_feature) = if image_is_left {
                (m.query_idx as usize, m.train_idx as usize)
            } else {
                (m.train_idx as usize, m.query_idx as usize)
            };
            let Some(query_kp) = frames
                .get(image)
                .and_then(|frame| frame.keypoints.get(feature))
            else {
                continue;
            };
            let Some(world_kp) = frames
                .get(other)
                .and_then(|frame| frame.keypoints.get(other_feature))
            else {
                continue;
            };
            problem
                .query_points2d
                .push([query_kp.x() as f64, query_kp.y() as f64]);
            problem
                .world_points2d
                .push([world_kp.x() as f64, world_kp.y() as f64]);
            problem.world_camera_idxs.push(world_camera_idx);
            problem.correspondences.push(StructurelessInlier {
                image,
                feature,
                other,
                other_feature,
            });
        }
    }

    problem
}

fn structureless_pair_side(image: usize, pair: &PairGeometry) -> Option<(usize, bool)> {
    if pair.left == image {
        Some((pair.right, true))
    } else if pair.right == image {
        Some((pair.left, false))
    } else {
        None
    }
}

fn solve_experimental_structureless_pair_pose_fallback(
    image: usize,
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
    obs_manager: &ObservationManager,
    camera_priors: &[CameraModel],
    registration_stats: &RegistrationStats,
) -> Option<AbsolutePose> {
    if registered_image_count(reconstruction) < 2 {
        return None;
    }
    let camera = registration_camera_for_image(
        image,
        reconstruction,
        config,
        camera_priors,
        registration_stats,
    );
    if camera_has_bogus_params(camera, config) {
        return None;
    }
    let min_num_inliers = structureless_min_num_inliers(config);
    let visible_corrs = obs_manager.num_visible_correspondences(image);
    if visible_corrs < min_num_inliers {
        return None;
    }

    let constraints =
        collect_structureless_pair_constraints(image, frames, pairs, reconstruction, config);
    let valid_corrs = constraints.iter().map(|c| c.inliers).sum::<usize>();
    if valid_corrs < min_num_inliers || distinct_structureless_neighbors(&constraints) < 2 {
        return None;
    }

    let mut best = None::<(f32, AbsolutePose)>;
    for hypothesis in &constraints {
        let rotation = pose_rotation(hypothesis.candidate_pose);
        let compatible = compatible_structureless_constraints(rotation, &constraints);
        let inliers = compatible.iter().map(|c| c.inliers).sum::<usize>();
        if inliers < min_num_inliers || distinct_structureless_neighbors(&compatible) < 2 {
            continue;
        }
        let inlier_ratio = inliers as f32 / valid_corrs.max(1) as f32;
        if inlier_ratio < config.abs_pose_min_inlier_ratio {
            continue;
        }
        let pair_rot_error = weighted_structureless_rotation_error(rotation, &compatible)?;
        if pair_rot_error > absolute_pose_pair_rotation_limit_deg() {
            continue;
        }
        let center = estimate_structureless_camera_center(&compatible)
            .unwrap_or_else(|| camera_center(hypothesis.candidate_pose));
        let mut pose = pose_from_rotation_center(rotation, center);
        let mut structureless_inliers = structureless_inliers_from_pose(
            pose,
            camera,
            &compatible,
            frames,
            reconstruction,
            config,
        );
        if let Some((refined_pose, refined_inliers)) = refine_structureless_pose_sampson(
            pose,
            camera,
            &compatible,
            &structureless_inliers,
            frames,
            reconstruction,
            config,
        ) {
            pose = refined_pose;
            structureless_inliers = refined_inliers;
        }
        let inliers = structureless_inliers.len();
        if inliers < min_num_inliers
            || distinct_structureless_inlier_neighbors(&structureless_inliers) < 2
        {
            continue;
        }
        let inlier_ratio = inliers as f32 / valid_corrs.max(1) as f32;
        if inlier_ratio < config.abs_pose_min_inlier_ratio {
            continue;
        }
        let mean_error_px = weighted_structureless_mean_error(&compatible);
        let score = inliers as f32 * 100.0 + inlier_ratio * 500.0
            - pair_rot_error.min(45.0) * 50.0
            - mean_error_px.min(20.0) * 10.0
            - image as f32 * 0.001;
        let absolute_pose = AbsolutePose {
            pose,
            camera,
            inliers,
            inlier_ratio,
            mean_error_px,
            structureless_inliers,
        };
        if best.as_ref().map(|(s, _)| score > *s).unwrap_or(true) {
            best = Some((score, absolute_pose));
        }
    }

    best.map(|(_, pose)| pose)
}

fn structureless_min_num_inliers(config: &MapperConfig) -> usize {
    2 * config.abs_pose_min_num_inliers
}

fn collect_structureless_pair_constraints(
    image: usize,
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
) -> Vec<StructurelessPairConstraint> {
    let mut constraints = Vec::new();
    for pair in pairs {
        if pair.pose_graph_only {
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
        if camera_has_bogus_params(reconstruction.camera_for_image(other), config) {
            continue;
        }
        let matches = valid_structureless_matches(image, other, frames, pair);
        let inliers = matches.len();
        if inliers == 0 {
            continue;
        }
        let Some(candidate_pose) = structureless_pose_from_pair(image, pair, other_pose) else {
            continue;
        };
        let line_origin = camera_center(other_pose);
        let candidate_center = camera_center(candidate_pose);
        let Some(line_direction) = (candidate_center - line_origin).try_normalize() else {
            continue;
        };
        constraints.push(StructurelessPairConstraint {
            other,
            image_is_left: pair.left == image,
            other_pose,
            relative_pose: pair.relative_pose,
            candidate_pose,
            line_origin,
            line_direction,
            inliers,
            mean_error_px: if pair.mean_reprojection_error_px.is_finite() {
                pair.mean_reprojection_error_px
            } else {
                0.0
            },
            matches,
        });
    }
    constraints
}

fn structureless_pose_from_pair(image: usize, pair: &PairGeometry, other_pose: SE3) -> Option<SE3> {
    if pair.left == image {
        Some(pair.relative_pose.inverse().compose(&other_pose))
    } else if pair.right == image {
        Some(pair.relative_pose.compose(&other_pose))
    } else {
        None
    }
}

fn valid_structureless_matches(
    image: usize,
    other: usize,
    frames: &[ImageFrame],
    pair: &PairGeometry,
) -> Vec<StructurelessInlier> {
    let Some(image_frame) = frames.get(image) else {
        return Vec::new();
    };
    let Some(other_frame) = frames.get(other) else {
        return Vec::new();
    };
    pair.inlier_matches
        .iter()
        .filter_map(|m| {
            let (feature, other_feature) = if pair.left == image {
                (m.query_idx as usize, m.train_idx as usize)
            } else {
                (m.train_idx as usize, m.query_idx as usize)
            };
            (feature < image_frame.keypoints.len() && other_feature < other_frame.keypoints.len())
                .then_some(StructurelessInlier {
                    image,
                    feature,
                    other,
                    other_feature,
                })
        })
        .collect()
}

fn compatible_structureless_constraints(
    rotation: glam::Quat,
    constraints: &[StructurelessPairConstraint],
) -> Vec<StructurelessPairConstraint> {
    constraints
        .iter()
        .cloned()
        .filter(|constraint| {
            structureless_pair_rotation_error(rotation, constraint)
                .is_some_and(|error| error <= absolute_pose_pair_rotation_limit_deg())
        })
        .collect()
}

fn structureless_inliers_from_pose(
    pose: SE3,
    camera: CameraModel,
    constraints: &[StructurelessPairConstraint],
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
) -> Vec<StructurelessInlier> {
    let max_residual = structureless_sampson_threshold(camera, config);
    let mut inliers = constraints
        .iter()
        .flat_map(|constraint| {
            constraint.matches.iter().copied().filter(move |inlier| {
                structureless_sampson_residual_for_inlier(
                    pose,
                    camera,
                    *inlier,
                    frames,
                    reconstruction,
                )
                .is_some_and(|residual| residual <= max_residual)
            })
        })
        .collect::<Vec<_>>();
    inliers.sort_unstable();
    inliers.dedup();
    inliers
}

fn refine_structureless_pose_sampson(
    initial_pose: SE3,
    camera: CameraModel,
    constraints: &[StructurelessPairConstraint],
    initial_inliers: &[StructurelessInlier],
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
) -> Option<(SE3, Vec<StructurelessInlier>)> {
    if initial_inliers.len() < structureless_min_num_inliers(config) {
        return None;
    }
    let mut pose = initial_pose;
    let mut inliers = initial_inliers.to_vec();
    let mut best_eval =
        evaluate_structureless_pose_sampson(pose, camera, &inliers, frames, reconstruction)?;
    let center = camera_center(initial_pose);
    let mut best_pair_rot_error =
        weighted_structureless_rotation_error(pose_rotation(pose), constraints)?;
    let mut improved = false;
    for _ in 0..6 {
        let delta = structureless_rotation_sampson_step(
            pose,
            camera,
            center,
            &inliers,
            frames,
            reconstruction,
        )?;
        if delta.norm() < 1.0e-7 {
            break;
        }
        let mut accepted = false;
        for scale in [1.0f32, 0.5, 0.25, 0.125, 0.0625] {
            let candidate_pose = apply_rotation_delta_fixed_center(pose, center, delta * scale);
            let Some(candidate_pair_rot_error) =
                weighted_structureless_rotation_error(pose_rotation(candidate_pose), constraints)
            else {
                continue;
            };
            if candidate_pair_rot_error > best_pair_rot_error + 0.05 {
                continue;
            }
            let candidate_inliers = structureless_inliers_from_pose(
                candidate_pose,
                camera,
                constraints,
                frames,
                reconstruction,
                config,
            );
            if candidate_inliers.len() + 2 < inliers.len()
                || candidate_inliers.len() < structureless_min_num_inliers(config)
            {
                continue;
            }
            let Some(candidate_eval) = evaluate_structureless_pose_sampson(
                candidate_pose,
                camera,
                &candidate_inliers,
                frames,
                reconstruction,
            ) else {
                continue;
            };
            if candidate_eval + 1.0e-8 < best_eval {
                pose = candidate_pose;
                inliers = candidate_inliers;
                best_eval = candidate_eval;
                best_pair_rot_error = candidate_pair_rot_error;
                improved = true;
                accepted = true;
                break;
            }
        }
        if !accepted {
            break;
        }
    }
    improved.then_some((pose, inliers))
}

fn evaluate_structureless_pose_sampson(
    pose: SE3,
    camera: CameraModel,
    inliers: &[StructurelessInlier],
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
) -> Option<f32> {
    let mut total = 0.0f32;
    let mut count = 0usize;
    for &inlier in inliers {
        let residual = structureless_sampson_residual_for_inlier(
            pose,
            camera,
            inlier,
            frames,
            reconstruction,
        )?;
        if !residual.is_finite() {
            return None;
        }
        let robust = residual.min(1.0e-2);
        total += robust * robust;
        count += 1;
    }
    (count > 0).then_some(total / count as f32)
}

fn structureless_rotation_sampson_step(
    pose: SE3,
    camera: CameraModel,
    center: glam::Vec3,
    inliers: &[StructurelessInlier],
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
) -> Option<SVector<f32, 3>> {
    let mut h = SMatrix::<f32, 3, 3>::zeros();
    let mut b = SVector::<f32, 3>::zeros();
    let mut used = 0usize;
    for &inlier in inliers {
        let residual = structureless_signed_sampson_residual_for_inlier(
            pose,
            camera,
            inlier,
            frames,
            reconstruction,
        )?;
        if !residual.is_finite() {
            continue;
        }
        let jacobian = numerical_structureless_rotation_sampson_jacobian(
            pose,
            camera,
            center,
            inlier,
            frames,
            reconstruction,
        )?;
        let weight = huber_weight(residual.abs(), structureless_sampson_huber_delta());
        h += jacobian.transpose() * jacobian * weight;
        b += jacobian.transpose() * SVector::<f32, 1>::new(residual) * weight;
        used += 1;
    }
    if used < 12 {
        return None;
    }
    let lambda = 1.0e-3 * (h.trace() / 3.0).max(1.0e-8);
    for i in 0..3 {
        h[(i, i)] += lambda;
    }
    let delta = h.lu().solve(&(-b))?;
    (delta.norm() < 0.05).then_some(delta)
}

fn structureless_sampson_huber_delta() -> f32 {
    2.0e-3
}

fn numerical_structureless_rotation_sampson_jacobian(
    pose: SE3,
    camera: CameraModel,
    center: glam::Vec3,
    inlier: StructurelessInlier,
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
) -> Option<SMatrix<f32, 1, 3>> {
    let mut jacobian = SMatrix::<f32, 1, 3>::zeros();
    let eps = [1.0e-5, 1.0e-5, 1.0e-5];
    for axis in 0..3 {
        let mut plus = SVector::<f32, 3>::zeros();
        plus[axis] = eps[axis];
        let mut minus = SVector::<f32, 3>::zeros();
        minus[axis] = -eps[axis];
        let r_plus = structureless_signed_sampson_residual_for_inlier(
            apply_rotation_delta_fixed_center(pose, center, plus),
            camera,
            inlier,
            frames,
            reconstruction,
        )?;
        let r_minus = structureless_signed_sampson_residual_for_inlier(
            apply_rotation_delta_fixed_center(pose, center, minus),
            camera,
            inlier,
            frames,
            reconstruction,
        )?;
        if !r_plus.is_finite() || !r_minus.is_finite() {
            return None;
        }
        jacobian[(0, axis)] = (r_plus - r_minus) / (2.0 * eps[axis]);
    }
    Some(jacobian)
}

fn structureless_sampson_threshold(camera: CameraModel, config: &MapperConfig) -> f32 {
    camera.cam_from_img_threshold((0.5 * config.pnp_threshold_px) as f64) as f32
}

fn structureless_sampson_residual_for_inlier(
    pose: SE3,
    camera: CameraModel,
    inlier: StructurelessInlier,
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
) -> Option<f32> {
    structureless_signed_sampson_residual_for_inlier(pose, camera, inlier, frames, reconstruction)
        .map(|residual| residual.abs())
}

fn structureless_signed_sampson_residual_for_inlier(
    pose: SE3,
    camera: CameraModel,
    inlier: StructurelessInlier,
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
) -> Option<f32> {
    let other_pose = reconstruction.poses.get(inlier.other).copied().flatten()?;
    let other_camera = reconstruction.camera_for_image(inlier.other);
    let query_kp = frames.get(inlier.image)?.keypoints.get(inlier.feature)?;
    let other_kp = frames
        .get(inlier.other)?
        .keypoints
        .get(inlier.other_feature)?;
    let query_xy = camera.cam_from_img_f32(query_kp.x(), query_kp.y())?;
    let other_xy = other_camera.cam_from_img_f32(other_kp.x(), other_kp.y())?;
    let relative_pose = other_pose.compose(&pose.inverse());
    let residual = sampson_residual_normalized(relative_pose, query_xy, other_xy);
    residual.is_finite().then_some(residual)
}

fn sampson_residual_normalized(pose: SE3, p1: [f32; 2], p2: [f32; 2]) -> f32 {
    let r = pose.rotation_matrix();
    let t = pose.translation();
    let rotation = Matrix3::from_row_slice(&[
        r[0][0], r[0][1], r[0][2], r[1][0], r[1][1], r[1][2], r[2][0], r[2][1], r[2][2],
    ]);
    let translation = Vector3::new(t[0], t[1], t[2]);
    let essential = skew_matrix(translation) * rotation;
    let x1 = Vector3::new(p1[0], p1[1], 1.0);
    let x2 = Vector3::new(p2[0], p2[1], 1.0);
    let ex1 = essential * x1;
    let etx2 = essential.transpose() * x2;
    let numerator = x2.dot(&(essential * x1));
    let denom = (ex1[0] * ex1[0] + ex1[1] * ex1[1] + etx2[0] * etx2[0] + etx2[1] * etx2[1]).sqrt();
    if denom <= 1.0e-12 {
        f32::INFINITY
    } else {
        numerator / denom
    }
}

fn skew_matrix(t: Vector3<f32>) -> Matrix3<f32> {
    Matrix3::new(0.0, -t[2], t[1], t[2], 0.0, -t[0], -t[1], t[0], 0.0)
}

fn distinct_structureless_inlier_neighbors(inliers: &[StructurelessInlier]) -> usize {
    inliers
        .iter()
        .map(|inlier| inlier.other)
        .collect::<BTreeSet<_>>()
        .len()
}

fn weighted_structureless_rotation_error(
    rotation: glam::Quat,
    constraints: &[StructurelessPairConstraint],
) -> Option<f32> {
    let mut total = 0.0f32;
    let mut weight = 0usize;
    for constraint in constraints {
        let error = structureless_pair_rotation_error(rotation, constraint)?;
        total += error * constraint.inliers as f32;
        weight += constraint.inliers;
    }
    (weight > 0).then_some(total / weight as f32)
}

fn structureless_pair_rotation_error(
    rotation: glam::Quat,
    constraint: &StructurelessPairConstraint,
) -> Option<f32> {
    let pose = SE3::from_quat_translation(rotation, glam::Vec3::ZERO);
    let predicted = if constraint
        .other_pose
        .translation()
        .iter()
        .any(|v| !v.is_finite())
    {
        return None;
    } else if constraint
        .relative_pose
        .translation()
        .iter()
        .any(|v| !v.is_finite())
    {
        return None;
    } else {
        // The temporary translation is zero; relative_rotation_deg ignores it.
        if constraint.image_is_left {
            constraint.other_pose.compose(&pose.inverse())
        } else {
            pose.compose(&constraint.other_pose.inverse())
        }
    };
    Some(crate::geometry::relative_rotation_deg(
        predicted,
        constraint.relative_pose,
    ))
}

fn estimate_structureless_camera_center(
    constraints: &[StructurelessPairConstraint],
) -> Option<glam::Vec3> {
    if constraints.len() < 2 {
        return None;
    }
    let mut a = Matrix3::<f32>::zeros();
    let mut b = Vector3::<f32>::zeros();
    let identity = Matrix3::<f32>::identity();
    for constraint in constraints {
        let d = Vector3::new(
            constraint.line_direction.x,
            constraint.line_direction.y,
            constraint.line_direction.z,
        );
        let origin = Vector3::new(
            constraint.line_origin.x,
            constraint.line_origin.y,
            constraint.line_origin.z,
        );
        let projector = identity - d * d.transpose();
        let weight = constraint.inliers.max(1) as f32;
        a += projector * weight;
        b += projector * origin * weight;
    }
    let center = a.lu().solve(&b)?;
    let center = glam::Vec3::new(center.x, center.y, center.z);
    center.is_finite().then_some(center)
}

fn weighted_structureless_mean_error(constraints: &[StructurelessPairConstraint]) -> f32 {
    let mut total = 0.0f32;
    let mut weight = 0usize;
    for constraint in constraints {
        if constraint.mean_error_px.is_finite() {
            total += constraint.mean_error_px * constraint.inliers as f32;
            weight += constraint.inliers;
        }
    }
    if weight == 0 {
        0.0
    } else {
        total / weight as f32
    }
}

fn distinct_structureless_neighbors(constraints: &[StructurelessPairConstraint]) -> usize {
    constraints
        .iter()
        .map(|constraint| constraint.other)
        .collect::<BTreeSet<_>>()
        .len()
}

#[derive(Debug, Clone, Copy)]
struct AbsolutePoseObservation {
    feature: usize,
    xy: [f32; 2],
    xyz: [f32; 3],
}

fn solve_absolute_pose(
    image: usize,
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
    camera_priors: &[CameraModel],
    camera_has_prior_focal_length: &[bool],
    registration_stats: &RegistrationStats,
) -> Option<AbsolutePose> {
    let camera = registration_camera_for_image(
        image,
        reconstruction,
        config,
        camera_priors,
        registration_stats,
    );
    if camera_has_bogus_params(camera, config) {
        return None;
    }
    let mut pose_observations = Vec::new();
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
        if camera_has_bogus_params(reconstruction.camera_for_image(other), config) {
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
            let xyz = reconstruction.points[point_id].xyz;
            pose_observations.push(AbsolutePoseObservation { feature, xy, xyz });
        }
    }
    let num_correspondences = pose_observations.len();
    if num_correspondences < config.abs_pose_min_num_inliers.max(4) {
        return None;
    }
    let estimate_focal = absolute_pose_estimate_focal_length_enabled(
        image,
        camera,
        reconstruction,
        config,
        camera_has_prior_focal_length,
        registration_stats,
    );
    let (pose, inliers, camera) = solve_absolute_pose_with_camera_hypotheses(
        &pose_observations,
        camera,
        estimate_focal,
        config,
    )?;
    let initial_eval =
        evaluate_absolute_pose(pose, &pose_observations, Some(&inliers), camera, config)?;
    if !accept_absolute_pose_eval(initial_eval, num_correspondences, config) {
        return None;
    }
    let refinement_observations = inlier_absolute_pose_observations(&pose_observations, &inliers);
    if refinement_observations.len() < config.abs_pose_min_num_inliers {
        return None;
    }
    let (pose, camera) = refine_absolute_pose_reprojection(
        pose,
        image,
        frames,
        reconstruction,
        &refinement_observations,
        camera,
        absolute_pose_refine_camera_params_enabled(
            image,
            camera,
            reconstruction,
            config,
            registration_stats,
        ),
        config,
    )?;
    let final_eval = evaluate_absolute_pose(pose, &pose_observations, None, camera, config)?;
    if !accept_absolute_pose_eval(final_eval, num_correspondences, config) {
        return None;
    }
    Some(AbsolutePose {
        pose,
        camera,
        inliers: final_eval.inliers,
        inlier_ratio: final_eval.inliers as f32 / num_correspondences.max(1) as f32,
        mean_error_px: final_eval.mean_error_px,
        structureless_inliers: Vec::new(),
    })
}

fn inlier_absolute_pose_observations(
    observations: &[AbsolutePoseObservation],
    inlier_mask: &[bool],
) -> Vec<AbsolutePoseObservation> {
    observations
        .iter()
        .enumerate()
        .filter_map(|(idx, observation)| {
            inlier_mask
                .get(idx)
                .copied()
                .unwrap_or(false)
                .then_some(*observation)
        })
        .collect()
}

fn solve_absolute_pose_with_camera_hypotheses(
    observations: &[AbsolutePoseObservation],
    camera: CameraModel,
    estimate_focal: bool,
    config: &MapperConfig,
) -> Option<(SE3, Vec<bool>, CameraModel)> {
    if estimate_focal {
        return solve_absolute_pose_with_focal_estimation(observations, camera, config);
    }
    let mut best = None::<(AbsolutePoseEval, SE3, Vec<bool>, CameraModel)>;
    let Some((pose, inliers)) = solve_absolute_pose_for_camera(observations, camera, config) else {
        return None;
    };
    let Some(eval) = evaluate_absolute_pose(pose, observations, Some(&inliers), camera, config)
    else {
        return None;
    };
    if accept_absolute_pose_eval(eval, observations.len(), config) {
        best = Some((eval, pose, inliers, camera));
    }
    best.map(|(_, pose, inliers, camera)| (pose, inliers, camera))
}

fn solve_absolute_pose_with_focal_estimation(
    observations: &[AbsolutePoseObservation],
    initial_camera: CameraModel,
    config: &MapperConfig,
) -> Option<(SE3, Vec<bool>, CameraModel)> {
    let Some(focal_idxs) = colmap_camera_model_focal_idxs(initial_camera.model_id) else {
        return solve_absolute_pose_for_camera(observations, initial_camera, config)
            .map(|(pose, inliers)| (pose, inliers, initial_camera));
    };
    if focal_idxs.is_empty() {
        return solve_absolute_pose_for_camera(observations, initial_camera, config)
            .map(|(pose, inliers)| (pose, inliers, initial_camera));
    }

    let initial_focal = average_camera_focal(initial_camera) as f32;
    let solver = PnPSolver {
        ransac_threshold: config.pnp_threshold_px,
        ransac_confidence: 0.99999,
        ransac_min_inlier_ratio: config.abs_pose_min_inlier_ratio,
        ransac_min_iterations: 100,
        ransac_max_iterations: config.pnp_iterations,
        ransac_random_seed: Some(absolute_pose_ransac_seed(config)),
        ..PnPSolver::new(
            initial_focal,
            initial_focal,
            initial_camera.cx,
            initial_camera.cy,
        )
    };
    let mut problem = PnPProblem::new();
    for observation in observations {
        problem.add_correspondence(observation.xy, observation.xyz);
    }
    let result = solver.solve_with_estimated_focal(&problem)?;
    let mut camera = initial_camera;
    for &idx in focal_idxs {
        if idx >= camera.num_params {
            return None;
        }
        camera.params[idx] = result.focal as f64;
    }
    camera.sync_intrinsics_from_params();
    if camera_has_bogus_params(camera, config) {
        return None;
    }
    Some((result.pose, result.inliers, camera))
}

fn solve_absolute_pose_for_camera(
    observations: &[AbsolutePoseObservation],
    camera: CameraModel,
    config: &MapperConfig,
) -> Option<(SE3, Vec<bool>)> {
    let solver = PnPSolver {
        ransac_threshold: camera.cam_from_img_threshold(config.pnp_threshold_px as f64) as f32,
        ransac_confidence: 0.99999,
        ransac_min_inlier_ratio: config.abs_pose_min_inlier_ratio,
        ransac_min_iterations: 100,
        ransac_max_iterations: config.pnp_iterations,
        ransac_random_seed: Some(absolute_pose_ransac_seed(config)),
        ..PnPSolver::new(1.0, 1.0, 0.0, 0.0)
    };
    let mut problem = PnPProblem::new();
    for observation in observations {
        let norm_xy = camera.cam_from_img_f32(observation.xy[0], observation.xy[1])?;
        problem.add_correspondence(norm_xy, observation.xyz);
    }
    solver.solve(&problem)
}

fn absolute_pose_ransac_seed(config: &MapperConfig) -> u64 {
    if config.random_seed >= 0 {
        config.random_seed as u64
    } else {
        next_absolute_pose_ransac_seed()
    }
}

fn next_absolute_pose_ransac_seed() -> u64 {
    static ABSOLUTE_POSE_RANSAC_SEED: AtomicU64 = AtomicU64::new(1);
    ABSOLUTE_POSE_RANSAC_SEED.fetch_add(1, Ordering::Relaxed)
}

fn average_camera_focal(camera: CameraModel) -> f64 {
    if let Some(focal_idxs) = colmap_camera_model_focal_idxs(camera.model_id) {
        let mut sum = 0.0;
        let mut count = 0usize;
        for &idx in focal_idxs {
            if idx < camera.num_params {
                sum += camera.params[idx];
                count += 1;
            }
        }
        if count > 0 && sum.is_finite() {
            return (sum / count as f64).max(1.0);
        }
    }
    camera.fx.max(camera.fy).max(1.0) as f64
}

fn absolute_pose_estimate_focal_length_enabled(
    image: usize,
    camera: CameraModel,
    reconstruction: &Reconstruction,
    config: &MapperConfig,
    camera_has_prior_focal_length: &[bool],
    registration_stats: &RegistrationStats,
) -> bool {
    if !config.ba_refine_focal_length {
        return false;
    }
    let camera_idx = reconstruction
        .image_camera_indices
        .get(image)
        .copied()
        .unwrap_or(0);
    let camera_id = reconstruction
        .camera_ids
        .get(camera_idx)
        .copied()
        .unwrap_or(1);
    if config.ba_constant_camera_ids.contains(&camera_id) {
        return false;
    }
    if camera_has_prior_focal_length
        .get(camera_idx)
        .copied()
        .unwrap_or(true)
    {
        return false;
    }
    registration_stats.registered_images_with_camera_id(camera_id) == 0
        || camera_has_bogus_params(camera, config)
}

#[derive(Debug, Clone, Copy)]
struct AbsolutePoseEval {
    inliers: usize,
    mean_error_px: f32,
}

fn evaluate_absolute_pose(
    pose: SE3,
    observations: &[AbsolutePoseObservation],
    inlier_mask: Option<&[bool]>,
    camera: CameraModel,
    config: &MapperConfig,
) -> Option<AbsolutePoseEval> {
    let mut count = 0usize;
    let mut total_error = 0.0f32;
    for (idx, observation) in observations.iter().enumerate() {
        if inlier_mask
            .and_then(|mask| mask.get(idx))
            .copied()
            .is_some_and(|is_inlier| !is_inlier)
        {
            continue;
        }
        let err =
            crate::geometry::reprojection_error_px(observation.xyz, pose, observation.xy, camera);
        if err.is_finite() && err <= config.pnp_threshold_px {
            count += 1;
            total_error += err;
        }
    }
    (count > 0).then_some(AbsolutePoseEval {
        inliers: count,
        mean_error_px: total_error / count as f32,
    })
}

fn accept_absolute_pose_eval(
    eval: AbsolutePoseEval,
    _num_correspondences: usize,
    config: &MapperConfig,
) -> bool {
    eval.inliers >= config.abs_pose_min_num_inliers && eval.mean_error_px <= config.pnp_threshold_px
}

fn refine_absolute_pose_reprojection(
    initial_pose: SE3,
    image: usize,
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
    observations: &[AbsolutePoseObservation],
    camera: CameraModel,
    refine_camera_params: bool,
    config: &MapperConfig,
) -> Option<(SE3, CameraModel)> {
    if observations.len() < config.abs_pose_min_num_inliers {
        return None;
    }
    let mut pose = initial_pose;
    let mut best_eval = evaluate_absolute_pose(pose, observations, None, camera, config)?;
    let scratch_reconstruction =
        absolute_pose_scratch_reconstruction(reconstruction, image, pose, observations);
    let observations_by_image = observations_by_image(&scratch_reconstruction);
    for _ in 0..8 {
        let Some(delta) = pose_only_gauss_newton_step(
            image,
            pose,
            frames,
            &scratch_reconstruction,
            &observations_by_image,
            &[],
            config,
        ) else {
            break;
        };
        let mut accepted = false;
        for scale in [1.0f32, 0.5, 0.25, 0.125, 0.0625] {
            let candidate = apply_pose_delta(pose, delta * scale);
            let Some(candidate_eval) =
                evaluate_absolute_pose(candidate, observations, None, camera, config)
            else {
                continue;
            };
            if candidate_eval.inliers >= best_eval.inliers.saturating_sub(1)
                && candidate_eval.mean_error_px + 1.0e-4 < best_eval.mean_error_px
            {
                pose = candidate;
                best_eval = candidate_eval;
                accepted = true;
                break;
            }
        }
        if !accepted || delta.norm() < 1.0e-6 {
            break;
        }
    }
    let camera = if refine_camera_params {
        let camera = refine_absolute_pose_camera_params(pose, observations, camera, config)?;
        if camera_has_bogus_params(camera, config) {
            return None;
        }
        camera
    } else {
        camera
    };
    Some((pose, camera))
}

fn absolute_pose_refine_camera_params_enabled(
    image: usize,
    camera: CameraModel,
    reconstruction: &Reconstruction,
    config: &MapperConfig,
    registration_stats: &RegistrationStats,
) -> bool {
    let camera_idx = reconstruction
        .image_camera_indices
        .get(image)
        .copied()
        .unwrap_or(0);
    let camera_id = reconstruction
        .camera_ids
        .get(camera_idx)
        .copied()
        .unwrap_or(1);
    if config.ba_constant_camera_ids.contains(&camera_id) {
        return false;
    }
    let registered_images_with_camera =
        registration_stats.registered_images_with_camera_id(camera_id);
    if registered_images_with_camera > 0 && !camera_has_bogus_params(camera, config) {
        return false;
    }
    config.ba_refine_focal_length
        || config.ba_refine_principal_point
        || config.ba_refine_extra_params
}

fn refine_absolute_pose_camera_params(
    pose: SE3,
    observations: &[AbsolutePoseObservation],
    initial_camera: CameraModel,
    config: &MapperConfig,
) -> Option<CameraModel> {
    let params = absolute_pose_camera_params(initial_camera, config);
    if params.is_empty() || observations.len() < config.abs_pose_min_num_inliers {
        return Some(initial_camera);
    }
    let mut camera = initial_camera;
    let mut best_cost = absolute_pose_camera_cost(pose, observations, camera)?;
    let mut damping = 1.0e-3;
    for _ in 0..6 {
        let mut h = DMatrix::<f64>::zeros(params.len(), params.len());
        let mut g = DVector::<f64>::zeros(params.len());
        for observation in observations {
            let residual = absolute_pose_residual(pose, camera, observation)?;
            let jacobians = params
                .iter()
                .map(|&param| {
                    numerical_absolute_pose_camera_jacobian(pose, camera, observation, param)
                })
                .collect::<Option<Vec<_>>>()?;
            for (i, j_i) in jacobians.iter().enumerate() {
                g[i] += j_i[0] * residual[0] + j_i[1] * residual[1];
                for (j, j_j) in jacobians.iter().enumerate() {
                    h[(i, j)] += j_i[0] * j_j[0] + j_i[1] * j_j[1];
                }
            }
        }
        for idx in 0..params.len() {
            h[(idx, idx)] += damping;
        }
        let Some(delta) = h.lu().solve(&(-g)) else {
            damping *= 10.0;
            continue;
        };
        if !delta.iter().all(|value| value.is_finite()) || delta.norm() > 100.0 {
            damping *= 10.0;
            continue;
        }
        let mut accepted = false;
        for step in [1.0, 0.5, 0.25, 0.125, 0.0625] {
            let mut candidate = camera;
            for (idx, &param) in params.iter().enumerate() {
                candidate.params[param] += delta[idx] * step;
            }
            candidate.sync_intrinsics_from_params();
            if camera_has_bogus_params(candidate, config) {
                continue;
            }
            let Some(cost) = absolute_pose_camera_cost(pose, observations, candidate) else {
                continue;
            };
            if cost + 1.0e-8 < best_cost {
                camera = candidate;
                best_cost = cost;
                damping = (damping * 0.5).max(1.0e-8);
                accepted = true;
                break;
            }
        }
        if !accepted {
            damping *= 4.0;
        }
        if delta.norm() < 1.0e-8 {
            break;
        }
    }
    Some(camera)
}

fn absolute_pose_camera_params(camera: CameraModel, config: &MapperConfig) -> Vec<usize> {
    let mut params = Vec::new();
    if config.ba_refine_focal_length {
        if let Some(idxs) = colmap_camera_model_focal_idxs(camera.model_id) {
            params.extend(idxs.iter().copied());
        }
    }
    if config.ba_refine_principal_point {
        if let Some([idx_x, idx_y]) = colmap_camera_model_principal_point_idxs(camera.model_id) {
            params.extend([idx_x, idx_y]);
        }
    }
    if config.ba_refine_extra_params {
        if let Some(idxs) = colmap_camera_model_extra_idxs(camera.model_id) {
            params.extend(idxs.iter().copied());
        }
    }
    params.retain(|&idx| idx < camera.num_params);
    params.sort_unstable();
    params.dedup();
    params
}

fn absolute_pose_camera_cost(
    pose: SE3,
    observations: &[AbsolutePoseObservation],
    camera: CameraModel,
) -> Option<f64> {
    let mut cost = 0.0;
    let mut count = 0usize;
    for observation in observations {
        let residual = absolute_pose_residual(pose, camera, observation)?;
        cost += residual[0] * residual[0] + residual[1] * residual[1];
        count += 1;
    }
    (count > 0).then_some(cost / count as f64)
}

fn absolute_pose_residual(
    pose: SE3,
    camera: CameraModel,
    observation: &AbsolutePoseObservation,
) -> Option<[f64; 2]> {
    let p = pose.transform_point(&observation.xyz);
    let predicted = camera.img_from_cam(p[0] as f64, p[1] as f64, p[2] as f64)?;
    Some([
        predicted[0] - observation.xy[0] as f64,
        predicted[1] - observation.xy[1] as f64,
    ])
}

fn numerical_absolute_pose_camera_jacobian(
    pose: SE3,
    camera: CameraModel,
    observation: &AbsolutePoseObservation,
    param: usize,
) -> Option<[f64; 2]> {
    if param >= camera.num_params {
        return None;
    }
    let eps = camera.params[param].abs().max(1.0) * 1.0e-6;
    let mut plus = camera;
    let mut minus = camera;
    plus.params[param] += eps;
    minus.params[param] -= eps;
    plus.sync_intrinsics_from_params();
    minus.sync_intrinsics_from_params();
    let r_plus = absolute_pose_residual(pose, plus, observation)?;
    let r_minus = absolute_pose_residual(pose, minus, observation)?;
    Some([
        (r_plus[0] - r_minus[0]) / (2.0 * eps),
        (r_plus[1] - r_minus[1]) / (2.0 * eps),
    ])
}

fn absolute_pose_scratch_reconstruction(
    base: &Reconstruction,
    image: usize,
    pose: SE3,
    observations: &[AbsolutePoseObservation],
) -> Reconstruction {
    let mut reconstruction = base.clone();
    if image < reconstruction.poses.len() {
        reconstruction.poses[image] = Some(pose);
    }
    reconstruction.points.clear();
    reconstruction.point_ids.clear();
    reconstruction.observations = reconstruction
        .keypoints
        .iter()
        .map(|keypoints| vec![None; keypoints.len()])
        .collect();
    for (point_id, observation) in observations.iter().enumerate() {
        let feature = observation.feature;
        if feature >= reconstruction.observations[image].len() {
            continue;
        }
        if reconstruction.observations[image][feature].is_some() {
            continue;
        }
        reconstruction.observations[image][feature] = Some(point_id);
        reconstruction.point_ids.push(point_id as u64 + 1);
        reconstruction.points.push(Point3D {
            xyz: observation.xyz,
            color: [0, 0, 0],
            error: 0.0,
            track: vec![TrackObservation { image, feature }],
        });
    }
    reconstruction
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

fn continue_or_triangulate_structureless_tracks(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    inliers: &[StructurelessInlier],
    tri_options: &IncrementalTriangulatorOptions,
    config: &MapperConfig,
) -> TriangulationReport {
    let mut report = TriangulationReport::default();
    let mut by_query_feature = BTreeMap::<(usize, usize), Vec<StructurelessInlier>>::new();
    for inlier in inliers {
        if !structureless_inlier_is_registered_and_valid(*inlier, frames, reconstruction) {
            continue;
        }
        by_query_feature
            .entry((inlier.image, inlier.feature))
            .or_default()
            .push(*inlier);
    }

    let mut observation_manager = ObservationManager::new(frames, pairs, reconstruction);
    for ((image, feature), mut group) in by_query_feature {
        if reconstruction.observations[image][feature].is_some() {
            continue;
        }
        group.sort_unstable();
        group.dedup();

        if let Some(point_id) = first_continueable_structureless_point(
            image,
            feature,
            &group,
            frames,
            reconstruction,
            config,
        ) {
            if observation_manager.add_observation(
                frames,
                pairs,
                reconstruction,
                point_id,
                TrackObservation { image, feature },
            ) {
                if let Some((xyz, error)) = triangulate_track(
                    &reconstruction.points[point_id].track.clone(),
                    frames,
                    reconstruction,
                    config,
                ) {
                    reconstruction.points[point_id].xyz = xyz;
                    reconstruction.points[point_id].error = error;
                }
                observation_manager.mark_point3d_modified(point_id);
                report.continued_observations += 1;
            }
            continue;
        }

        let mut observations = Vec::with_capacity(group.len() + 1);
        observations.push(TrackObservation { image, feature });
        observations.extend(group.iter().map(|inlier| TrackObservation {
            image: inlier.other,
            feature: inlier.other_feature,
        }));
        let Some(observations) = unique_track_observations(observations) else {
            continue;
        };
        if observations.len() < 2
            || observations.iter().any(|obs| {
                reconstruction
                    .observations
                    .get(obs.image)
                    .and_then(|features| features.get(obs.feature))
                    .copied()
                    .flatten()
                    .is_some()
            })
        {
            continue;
        }
        let Some((xyz, error)) = triangulate_track(&observations, frames, reconstruction, config)
        else {
            continue;
        };
        if !track_has_positive_depth(xyz, &observations, reconstruction)
            || !track_has_min_triangulation_angle(
                xyz,
                &observations,
                reconstruction,
                tri_options.min_angle_deg,
            )
        {
            continue;
        }
        if observation_manager
            .add_point3d(
                frames,
                pairs,
                reconstruction,
                Point3D {
                    xyz,
                    color: average_track_color(&observations, frames),
                    error,
                    track: observations,
                },
            )
            .is_some()
        {
            report.created_points += 1;
        }
    }
    report
}

fn structureless_inlier_is_registered_and_valid(
    inlier: StructurelessInlier,
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
) -> bool {
    reconstruction
        .poses
        .get(inlier.image)
        .copied()
        .flatten()
        .is_some()
        && reconstruction
            .poses
            .get(inlier.other)
            .copied()
            .flatten()
            .is_some()
        && frames
            .get(inlier.image)
            .is_some_and(|frame| inlier.feature < frame.keypoints.len())
        && frames
            .get(inlier.other)
            .is_some_and(|frame| inlier.other_feature < frame.keypoints.len())
        && reconstruction
            .observations
            .get(inlier.image)
            .is_some_and(|features| inlier.feature < features.len())
        && reconstruction
            .observations
            .get(inlier.other)
            .is_some_and(|features| inlier.other_feature < features.len())
}

fn first_continueable_structureless_point(
    image: usize,
    feature: usize,
    group: &[StructurelessInlier],
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
) -> Option<usize> {
    let pose = reconstruction.poses[image]?;
    let keypoint = frames.get(image)?.keypoints.get(feature)?;
    group.iter().find_map(|inlier| {
        let point_id = reconstruction.observations[inlier.other][inlier.other_feature]?;
        let point = reconstruction.points.get(point_id)?;
        if point.track.iter().any(|obs| obs.image == image) {
            return None;
        }
        let error = crate::geometry::reprojection_error_px(
            point.xyz,
            pose,
            [keypoint.x(), keypoint.y()],
            reconstruction.camera_for_image(image),
        );
        if !error.is_finite() || error > config.max_reprojection_error_px {
            return None;
        }
        let mut track = point.track.clone();
        track.push(TrackObservation { image, feature });
        if !track_has_positive_depth(point.xyz, &track, reconstruction)
            || !track_has_min_triangulation_angle(
                point.xyz,
                &track,
                reconstruction,
                track_filter_min_tri_angle_deg(config),
            )
        {
            return None;
        }
        Some(point_id)
    })
}

#[allow(dead_code)]
fn add_existing_observations(
    image: usize,
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    config: &MapperConfig,
) {
    let pose = reconstruction.poses[image].unwrap();
    let mut used_features = HashSet::new();
    let mut observation_manager = ObservationManager::new(frames, pairs, reconstruction);
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
                observation_manager.add_observation(
                    frames,
                    pairs,
                    reconstruction,
                    point_id,
                    TrackObservation { image, feature },
                );
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

    let mut observation_manager = ObservationManager::new(frames, pairs, reconstruction);
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
        let color = average_track_color(&observations, frames);
        observation_manager.add_point3d(
            frames,
            pairs,
            reconstruction,
            Point3D {
                xyz,
                color,
                error,
                track: observations,
            },
        );
    }
}

fn filter_reprojection_tracks(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    config: &MapperConfig,
) -> usize {
    filter_reprojection_tracks_with_policy(
        frames,
        pairs,
        reconstruction,
        config,
        track_filter_max_error_px(config),
        track_filter_min_tri_angle_deg(config),
        track_filter_min_track_length(),
    )
}

fn filter_reprojection_tracks_with_policy(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    config: &MapperConfig,
    max_error: f32,
    min_tri_angle: f32,
    min_track_length: usize,
) -> usize {
    let image_cameras = (0..reconstruction.poses.len())
        .map(|image| reconstruction.camera_for_image(image))
        .collect::<Vec<_>>();
    let mut removed = 0usize;
    let mut observation_manager = ObservationManager::new(frames, pairs, reconstruction);
    let mut point_id = 0usize;
    while point_id < reconstruction.points.len() {
        let point_xyz = reconstruction.points[point_id].xyz;
        let track = reconstruction.points[point_id].track.clone();
        let observations_to_delete = track
            .iter()
            .filter(|obs| {
                let Some(pose) = reconstruction.poses.get(obs.image).copied().flatten() else {
                    return true;
                };
                let Some(kp) = frames
                    .get(obs.image)
                    .and_then(|frame| frame.keypoints.get(obs.feature))
                else {
                    return true;
                };
                if !point_has_positive_depth(point_xyz, pose) {
                    return true;
                }
                let Some(camera) = image_cameras.get(obs.image).copied() else {
                    return true;
                };
                if camera_has_bogus_params(camera, config) {
                    return true;
                }
                let err = crate::geometry::reprojection_error_px(
                    point_xyz,
                    pose,
                    [kp.x(), kp.y()],
                    camera,
                );
                !err.is_finite() || err > max_error
            })
            .cloned()
            .collect::<Vec<_>>();

        if observations_to_delete.len() >= track.len().saturating_sub(1) {
            removed += reconstruction.points[point_id].track.len();
            observation_manager.delete_point3d(frames, pairs, reconstruction, point_id);
            continue;
        }

        for obs in observations_to_delete {
            if observation_manager.delete_observation(
                frames,
                pairs,
                reconstruction,
                obs.image,
                obs.feature,
            ) {
                removed += 1;
            }
        }

        if point_id >= reconstruction.points.len() {
            continue;
        }
        let track = reconstruction.points[point_id].track.clone();
        if track.len() < min_track_length {
            removed += track.len();
            observation_manager.delete_point3d(frames, pairs, reconstruction, point_id);
            continue;
        }
        if !track_has_min_triangulation_angle(
            reconstruction.points[point_id].xyz,
            &track,
            reconstruction,
            min_tri_angle,
        ) {
            removed += track.len();
            observation_manager.delete_point3d(frames, pairs, reconstruction, point_id);
            continue;
        }
        if let Some((xyz, error)) = triangulate_track(&track, frames, reconstruction, config)
            .filter(|(xyz, _)| {
                track_has_positive_depth(*xyz, &track, reconstruction)
                    && track_has_min_triangulation_angle(
                        *xyz,
                        &track,
                        reconstruction,
                        min_tri_angle,
                    )
            })
        {
            reconstruction.points[point_id].xyz = xyz;
            reconstruction.points[point_id].error = error;
        } else if let Some(error) = mean_track_reprojection_error(
            reconstruction.points[point_id].xyz,
            &track,
            frames,
            reconstruction,
        ) {
            reconstruction.points[point_id].error = error;
        } else {
            removed += track.len();
            observation_manager.delete_point3d(frames, pairs, reconstruction, point_id);
            continue;
        }
        point_id += 1;
    }
    removed
}

fn track_filter_max_error_px(config: &MapperConfig) -> f32 {
    std::env::var("RUSTSFM_TRACK_FILTER_MAX_ERROR_PX")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(config.max_reprojection_error_px)
}

fn track_filter_min_tri_angle_deg(config: &MapperConfig) -> f32 {
    std::env::var("RUSTSFM_TRACK_FILTER_MIN_TRI_ANGLE_DEG")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(
            IncrementalTriangulatorOptions::from_mapper_threshold(config.max_reprojection_error_px)
                .min_angle_deg,
        )
}

fn track_filter_min_track_length() -> usize {
    std::env::var("RUSTSFM_TRACK_FILTER_MIN_TRACK_LENGTH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 2)
        .unwrap_or(2)
}

fn camera_has_bogus_params(camera: CameraModel, config: &MapperConfig) -> bool {
    camera.has_bogus_params(
        config.min_focal_length_ratio,
        config.max_focal_length_ratio,
        config.max_extra_param,
    )
}

fn registration_camera_for_image(
    image: usize,
    reconstruction: &Reconstruction,
    config: &MapperConfig,
    camera_priors: &[CameraModel],
    registration_stats: &RegistrationStats,
) -> CameraModel {
    let camera = reconstruction.camera_for_image(image);
    let Some(camera_idx) = reconstruction.image_camera_indices.get(image).copied() else {
        return camera;
    };
    let camera_id = reconstruction
        .camera_ids
        .get(camera_idx)
        .copied()
        .unwrap_or(1);
    if registration_stats.registered_images_with_camera_id(camera_id) == 0 {
        if let Some(prior) = healthy_camera_prior(camera_idx, config, camera_priors) {
            return prior;
        }
    }
    if !camera_has_bogus_params(camera, config) {
        return camera;
    }
    healthy_camera_prior(camera_idx, config, camera_priors).unwrap_or(camera)
}

fn healthy_camera_prior(
    camera_idx: usize,
    config: &MapperConfig,
    camera_priors: &[CameraModel],
) -> Option<CameraModel> {
    let prior = camera_priors.get(camera_idx).copied()?;
    (!camera_has_bogus_params(prior, config)).then_some(prior)
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
            .cam_from_img_f32(kp.x(), kp.y())?;
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

fn mean_track_reprojection_error(
    xyz: [f32; 3],
    observations: &[TrackObservation],
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
) -> Option<f32> {
    let mut total_error = 0.0f32;
    let mut valid = 0usize;
    for obs in observations {
        let pose = reconstruction.poses[obs.image]?;
        let kp = frames.get(obs.image)?.keypoints.get(obs.feature)?;
        let err = crate::geometry::reprojection_error_px(
            xyz,
            pose,
            [kp.x(), kp.y()],
            reconstruction.camera_for_image(obs.image),
        );
        if !err.is_finite() {
            return None;
        }
        total_error += err;
        valid += 1;
    }
    (valid > 0).then_some(total_error / valid as f32)
}

fn point_has_positive_depth(point: [f32; 3], pose: SE3) -> bool {
    let cam_point = pose.transform_point(&point);
    cam_point[2].is_finite() && cam_point[2] > 1.0e-12
}

fn track_has_positive_depth(
    point: [f32; 3],
    observations: &[TrackObservation],
    reconstruction: &Reconstruction,
) -> bool {
    observations.iter().all(|obs| {
        reconstruction
            .poses
            .get(obs.image)
            .copied()
            .flatten()
            .map(|pose| point_has_positive_depth(point, pose))
            .unwrap_or(false)
    })
}

fn track_has_min_triangulation_angle(
    point: [f32; 3],
    observations: &[TrackObservation],
    reconstruction: &Reconstruction,
    min_angle_deg: f32,
) -> bool {
    if min_angle_deg <= 0.0 {
        return observations.len() >= 2;
    }
    let mut best = 0.0f32;
    for i in 0..observations.len() {
        for j in 0..i {
            let Some(pose_i) = reconstruction.poses[observations[i].image] else {
                continue;
            };
            let Some(pose_j) = reconstruction.poses[observations[j].image] else {
                continue;
            };
            if let Some(angle) = pair_triangulation_angle_deg(pose_i, pose_j, point) {
                best = best.max(angle);
            }
        }
    }
    best >= min_angle_deg
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

#[allow(dead_code)]
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
    let pairs = std::slice::from_ref(pair);
    let mut observation_manager = ObservationManager::new(frames, pairs, reconstruction);
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
        let Some(left_xy) = reconstruction
            .camera_for_image(pair.left)
            .cam_from_img_f32(lk.x(), lk.y())
        else {
            continue;
        };
        let Some(right_xy) = reconstruction
            .camera_for_image(pair.right)
            .cam_from_img_f32(rk.x(), rk.y())
        else {
            continue;
        };
        norm_left.push(left_xy);
        norm_right.push(right_xy);
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
        observation_manager.add_point3d(
            frames,
            pairs,
            reconstruction,
            Point3D {
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
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colmap::{
        ColmapDataId, ColmapRig, ColmapRigSensor, ColmapSensorId, ColmapSensorType,
    };
    use crate::correspondence_graph::FeatureMatch;
    use crate::database::{
        ColmapDatabase, ColmapDatabaseCamera, ColmapDatabaseFrame, ColmapDatabaseImage,
        ColmapKeypoint, ColmapTwoViewGeometry, DatabaseCacheOptions,
    };
    use std::fs;
    use tempfile::tempdir;

    fn experimental_structureless_config() -> MapperConfig {
        MapperConfig {
            experimental_structureless_pair_pose_fallback: true,
            ..MapperConfig::default()
        }
    }

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
    fn reference_camera_setup_preserves_frame_ownership() -> Result<()> {
        let dir = tempdir()?;
        let sparse = dir.path().join("sparse/0");
        fs::create_dir_all(&sparse)?;
        fs::write(
            sparse.join("cameras.txt"),
            "11 PINHOLE 640 480 500 501 320 240\n",
        )?;
        fs::write(sparse.join("images.txt"), "7 1 0 0 0 0 0 0 11 a.jpg\n\n")?;
        fs::write(sparse.join("points3D.txt"), "# points\n")?;
        fs::write(sparse.join("rigs.txt"), "3 1 CAMERA 11\n")?;
        fs::write(
            sparse.join("frames.txt"),
            "9 3 1 0 0 0 0 0 0 1 CAMERA 11 7\n",
        )?;

        let setup = reference_camera_setup(dir.path(), &[PathBuf::from("a.jpg")])?;

        assert_eq!(setup.rigs.len(), 1);
        assert_eq!(setup.frames.len(), 1);
        assert_eq!(setup.frames[0].frame_id, 9);
        assert_eq!(setup.image_frame_indices, vec![Some(0)]);
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
        let cache = db.load_cache(&DatabaseCacheOptions {
            load_all_images: true,
            ..DatabaseCacheOptions::default()
        })?;
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
    fn database_camera_setup_preserves_frame_ownership() -> Result<()> {
        let dir = tempdir()?;
        let db = ColmapDatabase::open(dir.path().join("database.db"))?;
        db.write_camera(
            &ColmapDatabaseCamera {
                camera: crate::colmap::ColmapCamera {
                    camera_id: 11,
                    model_id: crate::types::COLMAP_PINHOLE,
                    width: 100,
                    height: 80,
                    params: vec![50.0, 51.0, 50.0, 40.0],
                },
                has_prior_focal_length: true,
            },
            true,
        )?;
        db.write_image(
            &ColmapDatabaseImage {
                image_id: 7,
                name: "a.jpg".to_string(),
                camera_id: 11,
                frame_id: None,
            },
            true,
        )?;
        db.write_rig(
            &ColmapRig {
                rig_id: 3,
                ref_sensor_id: Some(ColmapSensorId {
                    sensor_type: ColmapSensorType::Camera,
                    sensor_id: 11,
                }),
                sensors: vec![ColmapRigSensor {
                    sensor_id: ColmapSensorId {
                        sensor_type: ColmapSensorType::Camera,
                        sensor_id: 11,
                    },
                    sensor_from_rig: None,
                }],
            },
            true,
        )?;
        db.write_frame(
            &ColmapDatabaseFrame {
                frame_id: 9,
                rig_id: 3,
                data_ids: vec![ColmapDataId {
                    sensor_id: ColmapSensorId {
                        sensor_type: ColmapSensorType::Camera,
                        sensor_id: 11,
                    },
                    data_id: 7,
                }],
            },
            true,
        )?;
        db.write_keypoints(7, &[ColmapKeypoint::new(0.0, 0.0)])?;
        let cache = db.load_cache(&DatabaseCacheOptions {
            min_num_matches: 0,
            load_all_images: true,
            ..DatabaseCacheOptions::default()
        })?;

        let setup = database_camera_setup(&cache, &[PathBuf::from("a.jpg")])?;

        assert_eq!(setup.rigs.len(), 1);
        assert_eq!(setup.frames.len(), 1);
        assert_eq!(setup.frames[0].frame_id, 9);
        assert_eq!(setup.image_frame_indices, vec![Some(0)]);
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
        assert_eq!(setup.camera_has_prior_focal_length, vec![true]);
        assert_eq!(setup.cameras[0].fx, 80.0);
        Ok(())
    }

    #[test]
    fn database_camera_setup_preserves_prior_focal_flags() -> Result<()> {
        let dir = tempdir()?;
        let db_path = dir.path().join("database.db");
        let db = ColmapDatabase::open(&db_path)?;
        for (camera_id, prior) in [(3, true), (4, false)] {
            db.write_camera(
                &ColmapDatabaseCamera {
                    camera: crate::colmap::ColmapCamera {
                        camera_id,
                        model_id: crate::types::COLMAP_PINHOLE,
                        width: 100,
                        height: 100,
                        params: vec![50.0, 50.0, 50.0, 50.0],
                    },
                    has_prior_focal_length: prior,
                },
                true,
            )?;
        }
        for (image_id, name, camera_id) in [(10, "a.jpg", 3), (11, "b.jpg", 4)] {
            db.write_image(
                &ColmapDatabaseImage {
                    image_id,
                    name: name.to_string(),
                    camera_id,
                    frame_id: None,
                },
                true,
            )?;
        }
        let cache = db.load_cache(&DatabaseCacheOptions {
            load_all_images: true,
            ..DatabaseCacheOptions::default()
        })?;
        let setup =
            database_camera_setup(&cache, &[PathBuf::from("a.jpg"), PathBuf::from("b.jpg")])?;

        assert_eq!(setup.camera_ids, vec![3, 4]);
        assert_eq!(setup.camera_has_prior_focal_length, vec![true, false]);
        assert_eq!(setup.image_camera_indices, vec![0, 1]);
        Ok(())
    }

    #[test]
    fn resolves_default_database_next_to_input_or_parent() -> Result<()> {
        let dir = tempdir()?;
        let images_dir = dir.path().join("images");
        fs::create_dir_all(&images_dir)?;
        let root_db = dir.path().join("database.db");
        fs::write(&root_db, [])?;

        let resolved = resolve_mapper_database_path(&MapperConfig {
            input: images_dir.clone(),
            ..MapperConfig::default()
        })?;
        assert_eq!(resolved.as_deref(), Some(root_db.as_path()));

        let image_db = images_dir.join("database.db");
        fs::write(&image_db, [])?;
        let resolved = resolve_mapper_database_path(&MapperConfig {
            input: images_dir,
            ..MapperConfig::default()
        })?;
        assert_eq!(resolved.as_deref(), Some(image_db.as_path()));
        Ok(())
    }

    #[test]
    fn explicit_missing_database_is_an_error() {
        let dir = tempdir().unwrap();
        let err = resolve_mapper_database_path(&MapperConfig {
            input: dir.path().join("images"),
            database: Some(dir.path().join("missing.db")),
            ..MapperConfig::default()
        })
        .unwrap_err();

        assert!(err.to_string().contains("database path does not exist"));
    }

    #[test]
    fn default_database_candidates_are_deduplicated_for_relative_input() {
        let candidates = default_database_candidates(Path::new("images"));

        assert_eq!(
            candidates,
            vec![
                PathBuf::from("images/database.db"),
                PathBuf::from("database.db")
            ]
        );
    }

    #[test]
    fn local_matching_requires_explicit_opt_in() {
        assert!(!MapperConfig::default().local_matching);
        assert_eq!(MapperConfig::default().local_window, 0);
        assert!(!MapperConfig::default().pose_graph);
        assert!(!MapperConfig::default().experimental_sequence_heuristics);
    }

    #[test]
    fn mapper_ba_defaults_match_colmap_intrinsic_refinement_policy() {
        let frames = vec![minimal_frame(0, "a.jpg")];
        let reconstruction = test_reconstruction(&frames);

        let options = mapper_ba_options(
            &MapperConfig::default(),
            &reconstruction,
            3,
            Some(vec![0]),
            Vec::new(),
            None,
            None,
        );

        assert!(options.refine_focal_length);
        assert!(!options.refine_principal_point);
        assert!(options.refine_extra_params);
        assert!(options.constant_cameras.is_empty());
        assert_eq!(options.gradient_tolerance, 1.0);
        assert_eq!(options.parameter_tolerance, 0.0);
        assert_eq!(options.max_linear_solver_iterations, 100);
    }

    #[test]
    fn local_ba_options_expand_images_to_frames_and_fix_partial_shared_cameras() {
        let frames = vec![
            minimal_frame(0, "outside_shared_camera.jpg"),
            minimal_frame(1, "registered_ref.jpg"),
            minimal_frame(2, "registered_aux.jpg"),
            minimal_frame(3, "local_neighbor.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.cameras = vec![
            CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0),
            CameraModel::new_pinhole(100, 100, 60.0, 60.0, 50.0, 50.0),
        ];
        reconstruction.camera_ids = vec![11, 12];
        reconstruction.image_camera_indices = vec![0, 0, 1, 1];
        for image in 0..4 {
            reconstruction.poses[image] = Some(SE3::identity());
        }
        reconstruction.frames = vec![Frame {
            frame_id: 9,
            rig_id: 3,
            rig_from_world: Rigid3::identity(),
            data_ids: Vec::new(),
        }];
        reconstruction.image_frame_indices[1] = Some(0);
        reconstruction.image_frame_indices[2] = Some(0);
        let stats = registration_stats(&reconstruction);

        let options = mapper_local_ba_options(
            &MapperConfig::default(),
            &reconstruction,
            &stats,
            5,
            vec![1, 3],
            vec![],
            None,
            None,
        );

        assert_eq!(options.variable_images, Some(vec![1, 2, 3]));
        assert_eq!(options.constant_images, Vec::<usize>::new());
        assert_eq!(options.constant_cameras, vec![0]);
        assert_eq!(options.gauge, crate::ba::BundleAdjustmentGauge::ThreePoints);
        assert_eq!(options.gradient_tolerance, 10.0);
        assert_eq!(options.parameter_tolerance, 0.0);
        assert_eq!(options.max_linear_solver_iterations, 100);
    }

    #[test]
    fn local_ba_options_fix_sensor_from_rig_when_rig_is_partially_covered() {
        let frames = vec![
            minimal_frame(0, "rig_ref_a.jpg"),
            minimal_frame(1, "rig_aux_a.jpg"),
            minimal_frame(2, "rig_ref_b.jpg"),
            minimal_frame(3, "rig_aux_b.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        let ref_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 11,
        };
        let aux_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 12,
        };
        reconstruction.rigs = vec![Rig {
            rig_id: 3,
            ref_sensor_id: Some(ref_sensor.clone()),
            sensors: vec![
                RigSensor {
                    sensor_id: ref_sensor.clone(),
                    sensor_from_rig: None,
                },
                RigSensor {
                    sensor_id: aux_sensor.clone(),
                    sensor_from_rig: Some(Rigid3 {
                        qvec: [1.0, 0.0, 0.0, 0.0],
                        tvec: [1.0, 0.0, 0.0],
                    }),
                },
            ],
        }];
        reconstruction.frames = vec![
            Frame {
                frame_id: 9,
                rig_id: 3,
                rig_from_world: Rigid3::identity(),
                data_ids: Vec::new(),
            },
            Frame {
                frame_id: 10,
                rig_id: 3,
                rig_from_world: Rigid3::identity(),
                data_ids: Vec::new(),
            },
        ];
        reconstruction.image_frame_indices = vec![Some(0), Some(0), Some(1), Some(1)];
        for image in 0..frames.len() {
            reconstruction.poses[image] = Some(SE3::identity());
        }
        let stats = registration_stats(&reconstruction);

        let partial_options = mapper_local_ba_options(
            &MapperConfig::default(),
            &reconstruction,
            &stats,
            5,
            vec![0],
            vec![],
            None,
            None,
        );
        assert_eq!(partial_options.variable_images, Some(vec![0, 1]));
        assert_eq!(
            partial_options.constant_sensor_from_rig,
            vec![aux_sensor.clone()]
        );

        let full_options = mapper_local_ba_options(
            &MapperConfig::default(),
            &reconstruction,
            &stats,
            5,
            vec![0, 2],
            vec![],
            None,
            None,
        );
        assert_eq!(full_options.variable_images, Some(vec![0, 1, 2, 3]));
        assert!(full_options.constant_sensor_from_rig.is_empty());
    }

    #[test]
    fn mapper_ba_constant_rig_ids_fix_non_ref_sensors() {
        let frames = vec![
            minimal_frame(0, "rig_ref.jpg"),
            minimal_frame(1, "rig_aux.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        let ref_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 11,
        };
        let aux_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 12,
        };
        reconstruction.rigs = vec![Rig {
            rig_id: 3,
            ref_sensor_id: Some(ref_sensor.clone()),
            sensors: vec![
                RigSensor {
                    sensor_id: ref_sensor,
                    sensor_from_rig: None,
                },
                RigSensor {
                    sensor_id: aux_sensor.clone(),
                    sensor_from_rig: Some(Rigid3 {
                        qvec: [1.0, 0.0, 0.0, 0.0],
                        tvec: [1.0, 0.0, 0.0],
                    }),
                },
            ],
        }];
        let config = MapperConfig {
            ba_constant_rig_ids: vec![99, 3, 3],
            ..MapperConfig::default()
        };

        let options = mapper_ba_options(&config, &reconstruction, 3, None, vec![], None, None);

        assert_eq!(options.constant_rigs, vec![3]);
        assert_eq!(options.constant_sensor_from_rig, vec![aux_sensor]);
    }

    #[test]
    fn mapper_absolute_pose_defaults_match_colmap_thresholds() {
        let config = MapperConfig::default();

        assert_eq!(config.pnp_threshold_px, 12.0);
        assert_eq!(config.pnp_iterations, 10000);
        assert_eq!(config.abs_pose_min_num_inliers, 30);
        assert_eq!(config.abs_pose_min_inlier_ratio, 0.25);
        assert_eq!(config.random_seed, -1);
        let default_seed_a = absolute_pose_ransac_seed(&config);
        let default_seed_b = absolute_pose_ransac_seed(&config);
        assert_ne!(default_seed_a, default_seed_b);
        assert_eq!(
            absolute_pose_ransac_seed(&MapperConfig {
                random_seed: 42,
                ..MapperConfig::default()
            }),
            42
        );
    }

    #[test]
    fn registration_rollback_reason_rejects_failed_refinement_or_bogus_camera() {
        let frames = vec![minimal_frame(0, "registered.jpg")];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        let config = MapperConfig::default();

        assert_eq!(
            registration_rollback_reason(&reconstruction, 0, true, false, &config),
            Some("local_ba_failed")
        );
        assert_eq!(
            registration_rollback_reason(&reconstruction, 0, true, true, &config),
            None
        );

        reconstruction.cameras[0].params[0] = 1.0;
        reconstruction.cameras[0].sync_intrinsics_from_params();
        assert_eq!(
            registration_rollback_reason(&reconstruction, 0, false, false, &config),
            Some("bogus_camera")
        );
    }

    #[test]
    fn absolute_pose_camera_refinement_follows_colmap_shared_camera_schedule() {
        let frames = vec![
            minimal_frame(0, "registered.jpg"),
            minimal_frame(1, "candidate.jpg"),
            minimal_frame(2, "other_camera.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.cameras = vec![
            CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0),
            CameraModel::new_pinhole(100, 100, 60.0, 60.0, 50.0, 50.0),
        ];
        reconstruction.camera_ids = vec![11, 42];
        reconstruction.image_camera_indices = vec![0, 0, 1];
        reconstruction.poses[0] = Some(SE3::identity());

        assert!(!absolute_pose_refine_camera_params_enabled(
            1,
            reconstruction.camera_for_image(1),
            &reconstruction,
            &MapperConfig::default(),
            &registration_stats(&reconstruction),
        ));
        assert!(absolute_pose_refine_camera_params_enabled(
            2,
            reconstruction.camera_for_image(2),
            &reconstruction,
            &MapperConfig::default(),
            &registration_stats(&reconstruction),
        ));

        let constant_config = MapperConfig {
            ba_constant_camera_ids: vec![42],
            ..MapperConfig::default()
        };
        assert!(!absolute_pose_refine_camera_params_enabled(
            2,
            reconstruction.camera_for_image(2),
            &reconstruction,
            &constant_config,
            &registration_stats(&reconstruction),
        ));

        reconstruction.cameras[0].params[0] = 1.0;
        reconstruction.cameras[0].sync_intrinsics_from_params();
        assert!(absolute_pose_refine_camera_params_enabled(
            1,
            reconstruction.camera_for_image(1),
            &reconstruction,
            &MapperConfig::default(),
            &registration_stats(&reconstruction),
        ));
    }

    #[test]
    fn registration_stats_track_frame_rig_camera_and_shared_image_events() {
        let frames = vec![
            minimal_frame(0, "seed.jpg"),
            minimal_frame(1, "rig_ref.jpg"),
            minimal_frame(2, "rig_aux.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.cameras = vec![
            CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0),
            CameraModel::new_pinhole(100, 100, 60.0, 60.0, 50.0, 50.0),
        ];
        reconstruction.camera_ids = vec![11, 12];
        reconstruction.image_camera_indices = vec![0, 0, 1];
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.rigs = vec![Rig {
            rig_id: 3,
            ref_sensor_id: Some(SensorId {
                sensor_type: SensorType::Camera,
                sensor_id: 11,
            }),
            sensors: Vec::new(),
        }];
        reconstruction.frames = vec![Frame {
            frame_id: 9,
            rig_id: 3,
            rig_from_world: Rigid3::identity(),
            data_ids: Vec::new(),
        }];
        reconstruction.image_frame_indices[1] = Some(0);
        reconstruction.image_frame_indices[2] = Some(0);

        let mut stats = RegistrationStats::from_reconstruction(&reconstruction);
        assert_eq!(stats.num_total_reg_images, 1);
        assert_eq!(stats.registered_images_with_camera_id(11), 1);

        reconstruction.poses[1] = Some(SE3::identity());
        reconstruction.poses[2] = Some(SE3::identity());
        stats.register_frame_for_image_event(&reconstruction, 1);

        assert_eq!(stats.num_reg_frames_per_rig.get(&3), Some(&1));
        assert_eq!(stats.registered_images_with_camera_id(11), 2);
        assert_eq!(stats.registered_images_with_camera_id(12), 1);
        assert_eq!(stats.num_total_reg_images, 3);
        assert_eq!(stats.num_shared_reg_images, 0);

        stats.deregister_frame_for_image_event(&reconstruction, 2);

        assert_eq!(stats.num_reg_frames_per_rig.get(&3), Some(&0));
        assert_eq!(stats.registered_images_with_camera_id(11), 1);
        assert_eq!(stats.registered_images_with_camera_id(12), 0);
        assert_eq!(stats.num_total_reg_images, 1);
    }

    #[test]
    fn sync_registered_frame_poses_updates_exported_rig_from_world_after_pose_changes() {
        let frames = vec![
            minimal_frame(0, "rig_ref.jpg"),
            minimal_frame(1, "rig_aux.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        let ref_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 11,
        };
        let aux_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 12,
        };
        reconstruction.rigs = vec![Rig {
            rig_id: 3,
            ref_sensor_id: Some(ref_sensor.clone()),
            sensors: vec![
                RigSensor {
                    sensor_id: ref_sensor.clone(),
                    sensor_from_rig: None,
                },
                RigSensor {
                    sensor_id: aux_sensor.clone(),
                    sensor_from_rig: Some(Rigid3 {
                        qvec: [1.0, 0.0, 0.0, 0.0],
                        tvec: [1.0, 0.0, 0.0],
                    }),
                },
            ],
        }];
        reconstruction.frames = vec![Frame {
            frame_id: 9,
            rig_id: 3,
            rig_from_world: Rigid3::identity(),
            data_ids: vec![
                DataId {
                    sensor_id: ref_sensor,
                    data_id: reconstruction.image_id(0) as u64,
                },
                DataId {
                    sensor_id: aux_sensor,
                    data_id: reconstruction.image_id(1) as u64,
                },
            ],
        }];
        reconstruction.image_frame_indices = vec![Some(0), Some(0)];
        reconstruction.poses[0] = Some(SE3::from_quat_translation(
            glam::Quat::IDENTITY,
            glam::Vec3::new(5.0, 0.0, 0.0),
        ));
        reconstruction.poses[1] = Some(SE3::from_quat_translation(
            glam::Quat::IDENTITY,
            glam::Vec3::new(99.0, 0.0, 0.0),
        ));

        sync_registered_frame_poses_from_images(&mut reconstruction);

        assert_eq!(
            reconstruction.frames[0].rig_from_world.tvec,
            [5.0, 0.0, 0.0]
        );
        assert_eq!(
            reconstruction.poses[0].unwrap().translation(),
            [5.0, 0.0, 0.0]
        );
        assert_eq!(
            reconstruction.poses[1].unwrap().translation(),
            [6.0, 0.0, 0.0]
        );
    }

    #[test]
    fn registration_camera_resets_unused_shared_camera_to_prior() {
        let frames = vec![
            minimal_frame(0, "registered.jpg"),
            minimal_frame(1, "candidate.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        let prior = CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0);
        let drifted = CameraModel::new_pinhole(100, 100, 70.0, 70.0, 50.0, 50.0);
        reconstruction.cameras = vec![prior, drifted];
        reconstruction.camera_ids = vec![11, 42];
        reconstruction.image_camera_indices = vec![0, 1];
        reconstruction.poses[0] = Some(SE3::identity());
        let priors = vec![prior, prior];

        let camera = registration_camera_for_image(
            1,
            &reconstruction,
            &MapperConfig::default(),
            &priors,
            &registration_stats(&reconstruction),
        );

        assert_eq!(camera.fx, prior.fx);
        assert_eq!(camera.fy, prior.fy);
    }

    #[test]
    fn registration_camera_keeps_registered_healthy_shared_camera() {
        let frames = vec![
            minimal_frame(0, "registered.jpg"),
            minimal_frame(1, "candidate.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        let prior = CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0);
        let refined = CameraModel::new_pinhole(100, 100, 65.0, 65.0, 50.0, 50.0);
        reconstruction.cameras = vec![refined];
        reconstruction.camera_ids = vec![11];
        reconstruction.image_camera_indices = vec![0, 0];
        reconstruction.poses[0] = Some(SE3::identity());
        let priors = vec![prior];

        let camera = registration_camera_for_image(
            1,
            &reconstruction,
            &MapperConfig::default(),
            &priors,
            &registration_stats(&reconstruction),
        );

        assert_eq!(camera.fx, refined.fx);
        assert_eq!(camera.fy, refined.fy);
    }

    #[test]
    fn absolute_pose_focal_estimation_follows_prior_focal_schedule() {
        let frames = vec![
            minimal_frame(0, "registered.jpg"),
            minimal_frame(1, "candidate.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.cameras = vec![
            CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0),
            CameraModel::new_pinhole(100, 100, 40.0, 40.0, 50.0, 50.0),
        ];
        reconstruction.camera_ids = vec![1, 2];
        reconstruction.image_camera_indices = vec![0, 1];
        reconstruction.poses[0] = Some(SE3::identity());
        let camera = reconstruction.camera_for_image(1);

        assert!(absolute_pose_estimate_focal_length_enabled(
            1,
            camera,
            &reconstruction,
            &MapperConfig::default(),
            &[true, false],
            &registration_stats(&reconstruction),
        ));
        assert!(!absolute_pose_estimate_focal_length_enabled(
            1,
            camera,
            &reconstruction,
            &MapperConfig::default(),
            &[true, true],
            &registration_stats(&reconstruction),
        ));
    }

    #[test]
    fn absolute_pose_unknown_focal_estimation_updates_camera() {
        let true_camera = CameraModel::new_pinhole(100, 100, 60.0, 60.0, 50.0, 50.0);
        let initial_camera = CameraModel::new_pinhole(100, 100, 40.0, 40.0, 50.0, 50.0);
        let pose = SE3::from_axis_angle(&[0.01, -0.015, 0.005], &[0.15, -0.05, 0.35]);
        let points: Vec<[f32; 3]> = (0..36)
            .map(|idx| {
                let x = ((idx % 6) as f32 - 2.5) * 0.25;
                let y = (((idx / 6) % 6) as f32 - 2.5) * 0.22;
                let z = 3.0 + (idx % 5) as f32 * 0.25;
                [x, y, z]
            })
            .collect();
        let observations = points
            .iter()
            .enumerate()
            .map(|(feature, &xyz)| {
                let p = pose.transform_point(&xyz);
                let xy = true_camera
                    .img_from_cam(p[0] as f64, p[1] as f64, p[2] as f64)
                    .unwrap();
                AbsolutePoseObservation {
                    feature,
                    xy: [xy[0] as f32, xy[1] as f32],
                    xyz,
                }
            })
            .collect::<Vec<_>>();
        let config = MapperConfig {
            abs_pose_min_num_inliers: 20,
            pnp_iterations: 10000,
            ..MapperConfig::default()
        };

        let (_estimated_pose, inliers, estimated_camera) =
            solve_absolute_pose_with_camera_hypotheses(
                &observations,
                initial_camera,
                true,
                &config,
            )
            .expect("absolute pose with focal estimation");

        assert!(inliers.iter().filter(|&&x| x).count() >= 20);
        assert!(
            (estimated_camera.fx as f64 - true_camera.fx as f64).abs()
                < (initial_camera.fx as f64 - true_camera.fx as f64).abs(),
            "estimated_fx={} true_fx={}",
            estimated_camera.fx,
            true_camera.fx
        );
    }

    #[test]
    fn mapper_ba_constant_camera_ids_map_to_camera_indices() {
        let frames = vec![minimal_frame(0, "a.jpg"), minimal_frame(1, "b.jpg")];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.cameras = vec![
            CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0),
            CameraModel::new_pinhole(100, 100, 55.0, 55.0, 50.0, 50.0),
        ];
        reconstruction.camera_ids = vec![11, 42];
        reconstruction.image_camera_indices = vec![0, 1];
        let config = MapperConfig {
            ba_constant_camera_ids: vec![42, 99],
            ..MapperConfig::default()
        };

        let options = mapper_ba_options(&config, &reconstruction, 3, None, Vec::new(), None, None);

        assert_eq!(options.constant_cameras, vec![1]);
    }

    #[test]
    fn checked_bundle_adjustment_rejects_bogus_registered_camera() {
        let mut frames = vec![minimal_frame(0, "a.jpg"), minimal_frame(1, "b.jpg")];
        frames[0].keypoints = vec![
            rustslam::KeyPoint::new(50.0, 50.0),
            rustslam::KeyPoint::new(55.0, 50.0),
            rustslam::KeyPoint::new(50.0, 55.0),
            rustslam::KeyPoint::new(55.0, 55.0),
        ];
        frames[1].keypoints = vec![
            rustslam::KeyPoint::new(75.0, 50.0),
            rustslam::KeyPoint::new(80.0, 50.0),
            rustslam::KeyPoint::new(75.0, 55.0),
            rustslam::KeyPoint::new(80.0, 55.0),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::from_quat_translation(
            glam::Quat::IDENTITY,
            glam::Vec3::new(1.0, 0.0, 0.0),
        ));
        for (idx, xyz) in [
            [0.0, 0.0, 2.0],
            [0.2, 0.0, 2.0],
            [0.0, 0.2, 2.0],
            [0.2, 0.2, 2.0],
        ]
        .into_iter()
        .enumerate()
        {
            reconstruction.observations[0][idx] = Some(idx);
            reconstruction.observations[1][idx] = Some(idx);
            reconstruction.points.push(Point3D {
                xyz,
                color: [0, 0, 0],
                error: 0.0,
                track: vec![
                    TrackObservation {
                        image: 0,
                        feature: idx,
                    },
                    TrackObservation {
                        image: 1,
                        feature: idx,
                    },
                ],
            });
            reconstruction.point_ids.push(idx as u64 + 1);
        }
        let original_camera = reconstruction.cameras[0];
        let report = refine_bundle_adjustment_checked(
            &frames,
            &mut reconstruction,
            &MapperConfig {
                max_focal_length_ratio: 0.4,
                ..MapperConfig::default()
            },
            crate::ba::BundleAdjustmentOptions {
                iterations: 2,
                huber_delta_px: 4.0,
                max_observation_error_px: 50.0,
                variable_images: Some(vec![1]),
                constant_images: vec![0],
                point_ids: Some((0..4).collect()),
                ..crate::ba::BundleAdjustmentOptions::default()
            },
        );

        assert!(report.is_none());
        assert_eq!(reconstruction.cameras[0].fx, original_camera.fx);
        assert_eq!(
            reconstruction.poses[1].unwrap().translation(),
            [1.0, 0.0, 0.0]
        );
    }

    #[test]
    fn absolute_pose_skips_correspondences_from_bogus_provider_cameras() {
        let mut frames = vec![
            minimal_frame(0, "provider.jpg"),
            minimal_frame(1, "candidate.jpg"),
        ];
        let good_camera = CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0);
        let provider_pose = SE3::identity();
        let candidate_pose = SE3::from_quat_translation(
            glam::Quat::from_rotation_y(0.08),
            glam::Vec3::new(-0.15, 0.02, 0.05),
        );
        let points = (0..30)
            .map(|idx| {
                let col = (idx % 6) as f32;
                let row = (idx / 6) as f32;
                [
                    -0.45 + col * 0.18,
                    -0.35 + row * 0.2,
                    3.0 + idx as f32 * 0.03,
                ]
            })
            .collect::<Vec<_>>();
        frames[0].keypoints = points
            .iter()
            .map(|&point| {
                let p = provider_pose.transform_point(&point);
                let xy = good_camera
                    .img_from_cam(p[0] as f64, p[1] as f64, p[2] as f64)
                    .unwrap();
                rustslam::KeyPoint::new(xy[0] as f32, xy[1] as f32)
            })
            .collect();
        frames[1].keypoints = points
            .iter()
            .map(|&point| {
                let p = candidate_pose.transform_point(&point);
                let xy = good_camera
                    .img_from_cam(p[0] as f64, p[1] as f64, p[2] as f64)
                    .unwrap();
                rustslam::KeyPoint::new(xy[0] as f32, xy[1] as f32)
            })
            .collect();
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.cameras = vec![
            CameraModel::new_pinhole(100, 100, 1.0, 1.0, 50.0, 50.0),
            good_camera,
        ];
        reconstruction.camera_ids = vec![1, 2];
        reconstruction.image_camera_indices = vec![0, 1];
        reconstruction.poses[0] = Some(provider_pose);
        for (idx, &xyz) in points.iter().enumerate() {
            reconstruction.observations[0][idx] = Some(idx);
            reconstruction.points.push(Point3D {
                xyz,
                color: [0, 0, 0],
                error: 0.0,
                track: vec![TrackObservation {
                    image: 0,
                    feature: idx,
                }],
            });
            reconstruction.point_ids.push(idx as u64 + 1);
        }
        let matches = (0..points.len() as u32)
            .map(|idx| (idx, idx))
            .collect::<Vec<_>>();
        let pair = pair_with_inliers(0, 1, &matches);

        assert!(solve_absolute_pose(
            1,
            &frames,
            &[pair],
            &reconstruction,
            &MapperConfig::default(),
            &camera_priors(&reconstruction),
            &camera_prior_focal_flags(&reconstruction, true),
            &registration_stats(&reconstruction),
        )
        .is_none());
    }

    #[test]
    fn absolute_pose_rejects_bogus_camera_before_pnp() {
        let frames = vec![
            minimal_frame(0, "registered.jpg"),
            minimal_frame(1, "candidate.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.cameras = vec![
            CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0),
            CameraModel::new_pinhole(100, 100, 1.0, 1.0, 50.0, 50.0),
        ];
        reconstruction.camera_ids = vec![1, 2];
        reconstruction.image_camera_indices = vec![0, 1];
        reconstruction.poses[0] = Some(SE3::identity());

        assert!(solve_absolute_pose(
            1,
            &frames,
            &[],
            &reconstruction,
            &MapperConfig::default(),
            &camera_priors(&reconstruction),
            &camera_prior_focal_flags(&reconstruction, true),
            &registration_stats(&reconstruction),
        )
        .is_none());
    }

    #[test]
    fn absolute_pose_resets_bogus_candidate_camera_from_prior() {
        let mut frames = vec![
            minimal_frame(0, "provider.jpg"),
            minimal_frame(1, "candidate.jpg"),
        ];
        let good_camera = CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0);
        let provider_pose = SE3::identity();
        let candidate_pose = SE3::from_quat_translation(
            glam::Quat::from_rotation_y(0.06),
            glam::Vec3::new(-0.2, 0.03, 0.02),
        );
        let points = (0..32)
            .map(|idx| {
                let col = (idx % 8) as f32;
                let row = (idx / 8) as f32;
                [-0.5 + col * 0.15, -0.3 + row * 0.2, 3.0 + idx as f32 * 0.02]
            })
            .collect::<Vec<_>>();
        frames[0].keypoints = points
            .iter()
            .map(|&point| {
                let p = provider_pose.transform_point(&point);
                let xy = good_camera
                    .img_from_cam(p[0] as f64, p[1] as f64, p[2] as f64)
                    .unwrap();
                rustslam::KeyPoint::new(xy[0] as f32, xy[1] as f32)
            })
            .collect();
        frames[1].keypoints = points
            .iter()
            .map(|&point| {
                let p = candidate_pose.transform_point(&point);
                let xy = good_camera
                    .img_from_cam(p[0] as f64, p[1] as f64, p[2] as f64)
                    .unwrap();
                rustslam::KeyPoint::new(xy[0] as f32, xy[1] as f32)
            })
            .collect();
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.cameras = vec![
            good_camera,
            CameraModel::new_pinhole(100, 100, 1.0, 1.0, 50.0, 50.0),
        ];
        reconstruction.camera_ids = vec![1, 2];
        reconstruction.image_camera_indices = vec![0, 1];
        reconstruction.poses[0] = Some(provider_pose);
        for (idx, &xyz) in points.iter().enumerate() {
            reconstruction.observations[0][idx] = Some(idx);
            reconstruction.points.push(Point3D {
                xyz,
                color: [0, 0, 0],
                error: 0.0,
                track: vec![TrackObservation {
                    image: 0,
                    feature: idx,
                }],
            });
            reconstruction.point_ids.push(idx as u64 + 1);
        }
        let matches = (0..points.len() as u32)
            .map(|idx| (idx, idx))
            .collect::<Vec<_>>();
        let pair = pair_with_inliers(0, 1, &matches);
        let camera_priors = vec![good_camera, good_camera];
        let config = MapperConfig {
            abs_pose_min_num_inliers: 24,
            ..MapperConfig::default()
        };

        let choice = choose_next_registration(
            &frames,
            &[pair],
            &reconstruction,
            &[0; 2],
            &config,
            &camera_priors,
            &[true, false],
            &registration_stats(&reconstruction),
        )
        .unwrap();

        assert_eq!(choice.source, "pnp");
        assert!(
            (choice.camera.fx - good_camera.fx).abs() < 1.0e-3,
            "fx={} expected={}",
            choice.camera.fx,
            good_camera.fx
        );
        assert!(
            (choice.camera.fy - good_camera.fy).abs() < 1.0e-3,
            "fy={} expected={}",
            choice.camera.fy,
            good_camera.fy
        );
        assert!(crate::geometry::relative_rotation_deg(choice.pose, candidate_pose) < 5.0);
    }

    #[test]
    fn absolute_pose_refinement_uses_ransac_inliers_only() {
        let pose = SE3::identity();
        let camera = CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0);
        let good = AbsolutePoseObservation {
            feature: 0,
            xy: [50.0, 50.0],
            xyz: [0.0, 0.0, 3.0],
        };
        let outlier = AbsolutePoseObservation {
            feature: 1,
            xy: [95.0, 95.0],
            xyz: [0.0, 0.0, 3.0],
        };
        let observations = vec![good, outlier];
        let inliers = inlier_absolute_pose_observations(&observations, &[true, false]);

        assert_eq!(inliers.len(), 1);
        assert_eq!(inliers[0].feature, 0);
        assert!(
            evaluate_absolute_pose(
                pose,
                &inliers,
                None,
                camera,
                &MapperConfig {
                    abs_pose_min_num_inliers: 1,
                    ..MapperConfig::default()
                },
            )
            .unwrap()
            .mean_error_px
                < 1.0e-6
        );
    }

    #[test]
    fn structureless_tracks_continue_existing_point_after_registration() {
        let camera = CameraModel::new_pinhole(200, 160, 80.0, 80.0, 100.0, 80.0);
        let poses = [
            SE3::identity(),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(0.04),
                glam::Vec3::new(-0.25, 0.0, 0.0),
            ),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(-0.03),
                glam::Vec3::new(0.35, 0.0, 0.0),
            ),
        ];
        let point = [0.0, 0.0, 3.0];
        let mut frames = vec![
            minimal_frame(0, "registered_a.jpg"),
            minimal_frame(1, "candidate.jpg"),
            minimal_frame(2, "registered_b.jpg"),
        ];
        for (image, pose) in poses.iter().copied().enumerate() {
            frames[image].width = camera.width;
            frames[image].height = camera.height;
            frames[image].keypoints = vec![project_test_point(camera, pose, point)];
            frames[image].colors = vec![[image as u8, 0, 0]];
        }
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.camera = camera;
        reconstruction.cameras = vec![camera];
        reconstruction.poses[0] = Some(poses[0]);
        reconstruction.poses[1] = Some(poses[1]);
        reconstruction.poses[2] = Some(poses[2]);
        reconstruction.observations[0][0] = Some(0);
        reconstruction.observations[2][0] = Some(0);
        reconstruction.point_ids.push(1);
        reconstruction.points.push(Point3D {
            xyz: point,
            color: [0, 0, 0],
            error: 0.0,
            track: vec![
                TrackObservation {
                    image: 0,
                    feature: 0,
                },
                TrackObservation {
                    image: 2,
                    feature: 0,
                },
            ],
        });
        let inliers = vec![
            StructurelessInlier {
                image: 1,
                feature: 0,
                other: 0,
                other_feature: 0,
            },
            StructurelessInlier {
                image: 1,
                feature: 0,
                other: 2,
                other_feature: 0,
            },
        ];

        let report = continue_or_triangulate_structureless_tracks(
            &frames,
            &[],
            &mut reconstruction,
            &inliers,
            &IncrementalTriangulatorOptions::from_mapper_threshold(4.0),
            &MapperConfig {
                max_reprojection_error_px: 4.0,
                ..MapperConfig::default()
            },
        );

        assert_eq!(report.continued_observations, 1);
        assert_eq!(report.created_points, 0);
        assert_eq!(reconstruction.observations[1][0], Some(0));
        assert!(reconstruction.points[0]
            .track
            .iter()
            .any(|obs| obs.image == 1 && obs.feature == 0));
    }

    #[test]
    fn structureless_tracks_triangulate_new_point_from_inlier_group() {
        let camera = CameraModel::new_pinhole(200, 160, 80.0, 80.0, 100.0, 80.0);
        let poses = [
            SE3::identity(),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(0.04),
                glam::Vec3::new(-0.25, 0.0, 0.0),
            ),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(-0.03),
                glam::Vec3::new(0.35, 0.0, 0.0),
            ),
        ];
        let point = [0.05, -0.02, 3.2];
        let mut frames = vec![
            minimal_frame(0, "registered_a.jpg"),
            minimal_frame(1, "candidate.jpg"),
            minimal_frame(2, "registered_b.jpg"),
        ];
        for (image, pose) in poses.iter().copied().enumerate() {
            frames[image].width = camera.width;
            frames[image].height = camera.height;
            frames[image].keypoints = vec![project_test_point(camera, pose, point)];
            frames[image].colors = vec![[image as u8, 0, 0]];
        }
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.camera = camera;
        reconstruction.cameras = vec![camera];
        reconstruction.poses[0] = Some(poses[0]);
        reconstruction.poses[1] = Some(poses[1]);
        reconstruction.poses[2] = Some(poses[2]);
        let inliers = vec![
            StructurelessInlier {
                image: 1,
                feature: 0,
                other: 0,
                other_feature: 0,
            },
            StructurelessInlier {
                image: 1,
                feature: 0,
                other: 2,
                other_feature: 0,
            },
        ];

        let report = continue_or_triangulate_structureless_tracks(
            &frames,
            &[],
            &mut reconstruction,
            &inliers,
            &IncrementalTriangulatorOptions::from_mapper_threshold(4.0),
            &MapperConfig {
                max_reprojection_error_px: 4.0,
                ..MapperConfig::default()
            },
        );

        assert_eq!(report.created_points, 1);
        assert_eq!(report.continued_observations, 0);
        assert_eq!(reconstruction.points.len(), 1);
        assert_eq!(reconstruction.points[0].track.len(), 3);
        assert_eq!(reconstruction.observations[1][0], Some(0));
        let xyz = glam::Vec3::from_array(reconstruction.points[0].xyz);
        assert!((xyz - glam::Vec3::from_array(point)).length() < 1.0e-3);
    }

    #[test]
    fn default_structureless_path_does_not_use_pair_pose_fallback() {
        let frames = structureless_frames(4);
        let poses = structureless_world_poses();
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses[0] = Some(poses[0]);
        reconstruction.poses[2] = Some(poses[2]);
        let pairs = vec![
            structureless_pair_from_poses(1, 0, poses[1], poses[0], 30),
            structureless_pair_from_poses(1, 2, poses[1], poses[2], 30),
        ];
        let config = MapperConfig {
            abs_pose_min_num_inliers: 20,
            ..MapperConfig::default()
        };

        let choice = choose_next_registration(
            &frames,
            &pairs,
            &reconstruction,
            &[0; 4],
            &config,
            &camera_priors(&reconstruction),
            &camera_prior_focal_flags(&reconstruction, true),
            &registration_stats(&reconstruction),
        );

        assert!(choice.is_none());
    }

    #[test]
    fn structureless_fallback_registers_candidate_without_points3d() {
        let frames = structureless_frames(4);
        let poses = structureless_world_poses();
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses[0] = Some(poses[0]);
        reconstruction.poses[2] = Some(poses[2]);
        let pairs = vec![
            structureless_pair_from_poses(1, 0, poses[1], poses[0], 30),
            structureless_pair_from_poses(1, 2, poses[1], poses[2], 30),
        ];
        let config = MapperConfig {
            abs_pose_min_num_inliers: 20,
            ..experimental_structureless_config()
        };

        let choice = choose_next_registration(
            &frames,
            &pairs,
            &reconstruction,
            &[0; 4],
            &config,
            &camera_priors(&reconstruction),
            &camera_prior_focal_flags(&reconstruction, true),
            &registration_stats(&reconstruction),
        )
        .unwrap();

        assert_eq!(choice.image, 1);
        assert_eq!(choice.source, "structureless");
        assert_eq!(choice.pnp_inliers, 60);
        let actual_t = glam::Vec3::from_array(choice.pose.translation());
        let expected_t = glam::Vec3::from_array(poses[1].translation());
        assert!((actual_t - expected_t).length() < 1.0e-3);
        assert!(crate::geometry::relative_rotation_deg(choice.pose, poses[1]) < 1.0e-4);
    }

    #[test]
    fn structureless_fallback_filters_matches_by_final_pose_sampson_error() {
        let camera = CameraModel::new_pinhole(200, 160, 80.0, 80.0, 100.0, 80.0);
        let poses = [
            SE3::identity(),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(0.04),
                glam::Vec3::new(-0.25, 0.0, 0.0),
            ),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(-0.03),
                glam::Vec3::new(0.35, 0.0, 0.0),
            ),
        ];
        let points = (0..48)
            .map(|idx| {
                let col = (idx % 8) as f32;
                let row = (idx / 8) as f32;
                [
                    -0.45 + col * 0.13,
                    -0.3 + row * 0.12,
                    3.0 + idx as f32 * 0.01,
                ]
            })
            .collect::<Vec<_>>();
        let mut frames = vec![
            minimal_frame(0, "registered_a.jpg"),
            minimal_frame(1, "candidate.jpg"),
            minimal_frame(2, "registered_b.jpg"),
        ];
        for (image, pose) in poses.iter().copied().enumerate() {
            frames[image].width = camera.width;
            frames[image].height = camera.height;
            frames[image].keypoints = points
                .iter()
                .map(|&point| project_test_point(camera, pose, point))
                .collect();
            frames[image].colors = vec![[image as u8, 0, 0]; points.len()];
        }
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.camera = camera;
        reconstruction.cameras = vec![camera];
        reconstruction.poses[0] = Some(poses[0]);
        reconstruction.poses[2] = Some(poses[2]);
        let mut pair_a = structureless_pair_from_poses(1, 0, poses[1], poses[0], 24);
        let mut pair_b = structureless_pair_from_poses(1, 2, poses[1], poses[2], 24);
        pair_a.inlier_matches[23].train_idx = 0;
        pair_b.inlier_matches[23].train_idx = 0;
        let config = MapperConfig {
            abs_pose_min_num_inliers: 20,
            pnp_threshold_px: 4.0,
            ..experimental_structureless_config()
        };

        let choice = choose_next_registration(
            &frames,
            &[pair_a, pair_b],
            &reconstruction,
            &[0; 3],
            &config,
            &camera_priors(&reconstruction),
            &camera_prior_focal_flags(&reconstruction, true),
            &registration_stats(&reconstruction),
        )
        .unwrap();

        assert_eq!(choice.source, "structureless");
        assert_eq!(choice.pnp_inliers, 46);
        assert_eq!(choice.structureless_inliers.len(), 46);
        assert!(!choice
            .structureless_inliers
            .iter()
            .any(|inlier| inlier.feature == 23));
    }

    #[test]
    fn structureless_sampson_refinement_improves_perturbed_rotation() {
        let camera = CameraModel::new_pinhole(200, 160, 80.0, 80.0, 100.0, 80.0);
        let poses = [
            SE3::identity(),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(0.04),
                glam::Vec3::new(-0.25, 0.0, 0.0),
            ),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(-0.03),
                glam::Vec3::new(0.35, 0.0, 0.0),
            ),
        ];
        let points = (0..64)
            .map(|idx| {
                let col = (idx % 8) as f32;
                let row = (idx / 8) as f32;
                [
                    -0.45 + col * 0.13,
                    -0.35 + row * 0.1,
                    3.0 + idx as f32 * 0.01,
                ]
            })
            .collect::<Vec<_>>();
        let mut frames = vec![
            minimal_frame(0, "registered_a.jpg"),
            minimal_frame(1, "candidate.jpg"),
            minimal_frame(2, "registered_b.jpg"),
        ];
        for (image, pose) in poses.iter().copied().enumerate() {
            frames[image].width = camera.width;
            frames[image].height = camera.height;
            frames[image].keypoints = points
                .iter()
                .map(|&point| project_test_point(camera, pose, point))
                .collect();
            frames[image].colors = vec![[image as u8, 0, 0]; points.len()];
        }
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.camera = camera;
        reconstruction.cameras = vec![camera];
        reconstruction.poses[0] = Some(poses[0]);
        reconstruction.poses[2] = Some(poses[2]);
        let pairs = vec![
            structureless_pair_from_poses(1, 0, poses[1], poses[0], points.len()),
            structureless_pair_from_poses(1, 2, poses[1], poses[2], points.len()),
        ];
        let config = MapperConfig {
            abs_pose_min_num_inliers: 20,
            pnp_threshold_px: 4.0,
            ..MapperConfig::default()
        };
        let constraints =
            collect_structureless_pair_constraints(1, &frames, &pairs, &reconstruction, &config);
        let center = camera_center(poses[1]);
        let initial_pose = pose_from_rotation_center(
            (glam::Quat::from_rotation_x(0.015) * pose_rotation(poses[1])).normalize(),
            center,
        );
        let initial_inliers = structureless_inliers_from_pose(
            initial_pose,
            camera,
            &constraints,
            &frames,
            &reconstruction,
            &config,
        );
        let initial_cost = evaluate_structureless_pose_sampson(
            initial_pose,
            camera,
            &initial_inliers,
            &frames,
            &reconstruction,
        )
        .unwrap();

        let (refined_pose, refined_inliers) = refine_structureless_pose_sampson(
            initial_pose,
            camera,
            &constraints,
            &initial_inliers,
            &frames,
            &reconstruction,
            &config,
        )
        .unwrap();
        let refined_cost = evaluate_structureless_pose_sampson(
            refined_pose,
            camera,
            &refined_inliers,
            &frames,
            &reconstruction,
        )
        .unwrap();

        assert!(refined_cost < initial_cost);
        assert!(
            crate::geometry::relative_rotation_deg(refined_pose, poses[1])
                < crate::geometry::relative_rotation_deg(initial_pose, poses[1])
        );
    }

    #[test]
    fn structureless_fallback_requires_colmap_visible_correspondence_threshold() {
        let frames = structureless_frames(3);
        let poses = structureless_world_poses();
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses[0] = Some(poses[0]);
        reconstruction.poses[2] = Some(poses[2]);
        let pairs = vec![
            structureless_pair_from_poses(1, 0, poses[1], poses[0], 19),
            structureless_pair_from_poses(1, 2, poses[1], poses[2], 20),
        ];
        let config = MapperConfig {
            abs_pose_min_num_inliers: 20,
            ..experimental_structureless_config()
        };

        let choice = choose_next_registration(
            &frames,
            &pairs,
            &reconstruction,
            &[0; 3],
            &config,
            &camera_priors(&reconstruction),
            &camera_prior_focal_flags(&reconstruction, true),
            &registration_stats(&reconstruction),
        );

        assert!(choice.is_none());
    }

    #[test]
    fn structureless_fallback_requires_two_registered_neighbors() {
        let frames = structureless_frames(3);
        let poses = structureless_world_poses();
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses[0] = Some(poses[0]);
        let pairs = vec![structureless_pair_from_poses(1, 0, poses[1], poses[0], 40)];
        let config = MapperConfig {
            abs_pose_min_num_inliers: 20,
            ..experimental_structureless_config()
        };

        let choice = choose_next_registration(
            &frames,
            &pairs,
            &reconstruction,
            &[0; 3],
            &config,
            &camera_priors(&reconstruction),
            &camera_prior_focal_flags(&reconstruction, true),
            &registration_stats(&reconstruction),
        );

        assert!(choice.is_none());
    }

    #[test]
    fn structureless_fallback_skips_bogus_registered_neighbor_camera() {
        let frames = structureless_frames(4);
        let poses = structureless_world_poses();
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.cameras = vec![
            CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0),
            CameraModel::new_pinhole(100, 100, 1.0, 1.0, 50.0, 50.0),
        ];
        reconstruction.camera_ids = vec![1, 2];
        reconstruction.image_camera_indices = vec![0, 0, 1, 0];
        reconstruction.poses[0] = Some(poses[0]);
        reconstruction.poses[2] = Some(poses[2]);
        reconstruction.poses[3] = Some(poses[3]);
        let pairs = vec![
            structureless_pair_from_poses(1, 0, poses[1], poses[0], 20),
            structureless_pair_from_poses(1, 2, poses[1], poses[2], 20),
            structureless_pair_from_poses(1, 3, poses[1], poses[3], 20),
        ];
        let config = MapperConfig {
            abs_pose_min_num_inliers: 20,
            ..experimental_structureless_config()
        };

        let choice = choose_next_registration(
            &frames,
            &pairs,
            &reconstruction,
            &[0; 4],
            &config,
            &camera_priors(&reconstruction),
            &camera_prior_focal_flags(&reconstruction, true),
            &registration_stats(&reconstruction),
        )
        .unwrap();

        assert_eq!(choice.source, "structureless");
        assert_eq!(choice.pnp_inliers, 40);
        assert!(crate::geometry::relative_rotation_deg(choice.pose, poses[1]) < 1.0e-4);
    }

    #[test]
    fn mark_unregistered_images_keeps_structureless_candidate_retryable() {
        let frames = structureless_frames(3);
        let poses = structureless_world_poses();
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses[0] = Some(poses[0]);
        reconstruction.poses[2] = Some(poses[2]);
        let pairs = vec![
            structureless_pair_from_poses(1, 0, poses[1], poses[0], 20),
            structureless_pair_from_poses(1, 2, poses[1], poses[2], 20),
        ];
        let config = MapperConfig {
            abs_pose_min_num_inliers: 20,
            ..experimental_structureless_config()
        };
        let mut reg_trials = vec![0; 3];

        mark_unregistered_images_with_no_absolute_pose(
            &frames,
            &pairs,
            &reconstruction,
            &mut reg_trials,
            &config,
            &camera_priors(&reconstruction),
            &camera_prior_focal_flags(&reconstruction, true),
            &registration_stats(&reconstruction),
        );

        assert_eq!(reg_trials[1], 0);
    }

    #[test]
    fn absolute_pose_camera_refinement_improves_focal_length() {
        let true_camera = CameraModel::new_pinhole(100, 100, 60.0, 60.0, 50.0, 50.0);
        let initial_camera = CameraModel::new_pinhole(100, 100, 45.0, 45.0, 50.0, 50.0);
        let pose = SE3::identity();
        let points: [[f32; 3]; 8] = [
            [-0.5, -0.4, 3.0],
            [-0.1, -0.3, 3.2],
            [0.3, -0.35, 3.1],
            [-0.4, 0.0, 3.3],
            [0.0, 0.0, 3.0],
            [0.4, 0.1, 3.4],
            [-0.25, 0.45, 3.5],
            [0.2, 0.4, 3.2],
        ];
        let observations = points
            .iter()
            .enumerate()
            .map(|(feature, &xyz)| {
                let p = pose.transform_point(&xyz);
                let xy = true_camera
                    .img_from_cam(p[0] as f64, p[1] as f64, p[2] as f64)
                    .unwrap();
                AbsolutePoseObservation {
                    feature,
                    xy: [xy[0] as f32, xy[1] as f32],
                    xyz,
                }
            })
            .collect::<Vec<_>>();
        let initial_cost = absolute_pose_camera_cost(pose, &observations, initial_camera).unwrap();

        let refined = refine_absolute_pose_camera_params(
            pose,
            &observations,
            initial_camera,
            &MapperConfig {
                abs_pose_min_num_inliers: 4,
                ..MapperConfig::default()
            },
        )
        .unwrap();
        let refined_cost = absolute_pose_camera_cost(pose, &observations, refined).unwrap();

        assert!(refined_cost < initial_cost);
        assert!(
            (60.0 - refined.fx as f64).abs() < (60.0 - initial_camera.fx as f64).abs(),
            "refined_fx={}",
            refined.fx
        );
    }

    #[test]
    fn global_ba_schedule_triggers_on_image_or_point_growth() {
        let frames = vec![
            minimal_frame(0, "a.jpg"),
            minimal_frame(1, "b.jpg"),
            minimal_frame(2, "c.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::identity());
        reconstruction.points.push(Point3D {
            xyz: [0.0, 0.0, 2.0],
            color: [0, 0, 0],
            error: 0.0,
            track: vec![
                TrackObservation {
                    image: 0,
                    feature: 0,
                },
                TrackObservation {
                    image: 1,
                    feature: 0,
                },
            ],
        });
        reconstruction.point_ids.push(1);
        let mut schedule = GlobalBaSchedule::new(&reconstruction);
        let config = MapperConfig {
            global_ba_images_freq: 999,
            global_ba_points_freq: 999,
            global_ba_images_ratio: 1.1,
            global_ba_points_ratio: 1.1,
            ..MapperConfig::default()
        };

        assert!(!should_run_global_ba(&schedule, &reconstruction, &config));

        reconstruction.poses[2] = Some(SE3::identity());
        reconstruction.points.push(Point3D {
            xyz: [0.1, 0.0, 2.0],
            color: [0, 0, 0],
            error: 0.0,
            track: vec![
                TrackObservation {
                    image: 0,
                    feature: 1,
                },
                TrackObservation {
                    image: 2,
                    feature: 1,
                },
            ],
        });
        reconstruction.point_ids.push(2);

        assert!(should_run_global_ba(&schedule, &reconstruction, &config));

        schedule.mark(&reconstruction);
        assert!(!should_run_global_ba(&schedule, &reconstruction, &config));
    }

    #[test]
    fn global_ba_schedule_counts_registered_frames_not_frame_images() {
        let frames = vec![
            minimal_frame(0, "rig_ref.jpg"),
            minimal_frame(1, "rig_aux.jpg"),
            minimal_frame(2, "new_frame.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.frames = vec![Frame {
            frame_id: 9,
            rig_id: 3,
            rig_from_world: Rigid3::identity(),
            data_ids: Vec::new(),
        }];
        reconstruction.image_frame_indices[0] = Some(0);
        reconstruction.image_frame_indices[1] = Some(0);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::identity());
        reconstruction.points.push(Point3D {
            xyz: [0.0, 0.0, 2.0],
            color: [0, 0, 0],
            error: 0.0,
            track: vec![
                TrackObservation {
                    image: 0,
                    feature: 0,
                },
                TrackObservation {
                    image: 1,
                    feature: 0,
                },
            ],
        });
        reconstruction.point_ids.push(1);
        let schedule = GlobalBaSchedule::new(&reconstruction);
        let config = MapperConfig {
            global_ba_images_freq: 1,
            global_ba_points_freq: 999,
            global_ba_images_ratio: 10.0,
            global_ba_points_ratio: 10.0,
            ..MapperConfig::default()
        };

        assert!(!should_run_global_ba(&schedule, &reconstruction, &config));

        reconstruction.poses[2] = Some(SE3::identity());
        assert!(should_run_global_ba(&schedule, &reconstruction, &config));
    }

    #[test]
    fn global_ba_gauge_images_use_registered_images_not_zero_index() {
        let frames = vec![
            minimal_frame(0, "a.jpg"),
            minimal_frame(1, "b.jpg"),
            minimal_frame(2, "c.jpg"),
            minimal_frame(3, "d.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses[2] = Some(SE3::identity());
        reconstruction.poses[3] = Some(SE3::identity());

        assert_eq!(global_ba_gauge_images(&reconstruction), vec![2, 3]);
    }

    #[test]
    fn global_ba_gauge_images_use_distinct_registration_units() {
        let frames = vec![
            minimal_frame(0, "rig_ref.jpg"),
            minimal_frame(1, "rig_aux.jpg"),
            minimal_frame(2, "next_frame.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.frames = vec![Frame {
            frame_id: 9,
            rig_id: 3,
            rig_from_world: Rigid3::identity(),
            data_ids: Vec::new(),
        }];
        reconstruction.image_frame_indices[0] = Some(0);
        reconstruction.image_frame_indices[1] = Some(0);
        for image in 0..3 {
            reconstruction.poses[image] = Some(SE3::identity());
        }

        assert_eq!(global_ba_gauge_images(&reconstruction), vec![0, 2]);
    }

    #[test]
    fn local_image_fallback_uses_per_image_camera_ownership() {
        let mut frames = vec![minimal_frame(0, "left.jpg"), minimal_frame(1, "right.jpg")];
        frames[0].width = 120;
        frames[0].height = 80;
        frames[1].width = 200;
        frames[1].height = 150;

        let setup = local_image_camera_setup(
            &frames,
            &MapperConfig {
                fx: Some(90.0),
                ..MapperConfig::default()
            },
        );

        assert_eq!(setup.camera_ids, vec![1, 2]);
        assert_eq!(setup.image_ids, vec![1, 2]);
        assert_eq!(setup.image_camera_indices, vec![0, 1]);
        assert_eq!(setup.cameras[0].width, 120);
        assert_eq!(setup.cameras[1].height, 150);
        assert_eq!(setup.cameras[0].fx, 90.0);
        assert_eq!(setup.cameras[1].cx, 100.0);
    }

    #[test]
    fn local_pair_candidates_do_not_include_segment_bridges_by_default() {
        let candidates = local_pair_candidates(194, 3, false);

        assert!(candidates.contains(&(0, 1)));
        assert!(candidates.contains(&(0, 3)));
        assert!(!candidates.contains(&(0, 192)));
        assert!(!candidates.contains(&(188, 192)));
    }

    #[test]
    fn segment_bridge_candidates_require_experimental_sequence_heuristics() {
        let candidates = local_pair_candidates(194, 3, true);

        assert!(candidates.contains(&(0, 192)));
        assert!(candidates.contains(&(191, 192)));
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
        assert_eq!(
            pair.two_view_config,
            crate::database::COLMAP_TWO_VIEW_CALIBRATED
        );
        assert_eq!(pair.inliers, 4);
        assert_eq!(pair.triangulated, 4);
        assert!(pair.mean_reprojection_error_px < 1.0e-4);
        assert_eq!(pair.relative_pose.translation(), [-1.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn initial_pair_prefers_strong_non_adjacent_colmap_style_candidate() {
        let weak_adjacent = test_pair(0, 1, 120, 60, 20.0, [1.0, 0.0, 0.0]);
        let strong_non_adjacent = test_pair(0, 3, 220, 140, 25.0, [1.0, 0.1, 0.0]);
        let pairs = vec![weak_adjacent, strong_non_adjacent];
        let frames = structureless_frames(4);
        let reconstruction = test_reconstruction(&frames);

        let chosen =
            choose_initial_pair(&pairs, &reconstruction, &MapperConfig::default()).unwrap();

        assert_eq!((chosen.left, chosen.right), (0, 3));
    }

    #[test]
    fn initial_pair_rejects_forward_motion_and_low_triangulation_in_strict_pass() {
        let forward_motion = test_pair(0, 1, 300, 200, 30.0, [0.0, 0.0, 1.0]);
        let low_angle = test_pair(0, 2, 260, 180, 2.0, [1.0, 0.0, 0.0]);
        let stable = test_pair(1, 3, 140, 90, 18.0, [1.0, 0.0, 0.1]);
        let pairs = vec![forward_motion, low_angle, stable];
        let frames = structureless_frames(4);
        let reconstruction = test_reconstruction(&frames);

        let chosen =
            choose_initial_pair(&pairs, &reconstruction, &MapperConfig::default()).unwrap();

        assert_eq!((chosen.left, chosen.right), (1, 3));
    }

    #[test]
    fn initial_pair_returns_none_when_no_pair_passes_colmap_checks() {
        let forward_motion = test_pair(0, 1, 300, 200, 30.0, [0.0, 0.0, 1.0]);
        let low_angle = test_pair(0, 2, 260, 180, 2.0, [1.0, 0.0, 0.0]);
        let low_inliers = test_pair(1, 3, 50, 90, 18.0, [1.0, 0.0, 0.1]);
        let pairs = vec![forward_motion, low_angle, low_inliers];
        let frames = structureless_frames(4);
        let reconstruction = test_reconstruction(&frames);

        assert!(choose_initial_pair(&pairs, &reconstruction, &MapperConfig::default()).is_none());
    }

    #[test]
    fn initial_pair_skips_images_from_same_frame() {
        let same_frame = test_pair(0, 1, 500, 450, 25.0, [1.0, 0.0, 0.0]);
        let cross_frame = test_pair(1, 2, 180, 140, 18.0, [1.0, 0.0, 0.0]);
        let pairs = vec![same_frame, cross_frame];
        let frames = structureless_frames(3);
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.frames = vec![Frame {
            frame_id: 9,
            rig_id: 3,
            rig_from_world: Rigid3::identity(),
            data_ids: Vec::new(),
        }];
        reconstruction.image_frame_indices[0] = Some(0);
        reconstruction.image_frame_indices[1] = Some(0);

        let chosen =
            choose_initial_pair(&pairs, &reconstruction, &MapperConfig::default()).unwrap();

        assert_eq!((chosen.left, chosen.right), (1, 2));
    }

    #[test]
    fn registration_score_prefers_more_visible_points3d() {
        let frames = vec![
            minimal_frame(0, "seed.jpg"),
            minimal_frame(1, "weak.jpg"),
            minimal_frame(2, "strong.jpg"),
        ];
        let pairs = vec![
            pair_with_inliers(0, 1, &[(0, 0)]),
            pair_with_inliers(0, 2, &[(0, 0), (1, 1)]),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.observations[0][0] = Some(0);
        reconstruction.observations[0][1] = Some(1);
        reconstruction.points.push(Point3D {
            xyz: [0.0, 0.0, 1.0],
            color: [0, 0, 0],
            error: 0.0,
            track: vec![TrackObservation {
                image: 0,
                feature: 0,
            }],
        });
        reconstruction.points.push(Point3D {
            xyz: [0.1, 0.0, 1.0],
            color: [0, 0, 0],
            error: 0.0,
            track: vec![TrackObservation {
                image: 0,
                feature: 1,
            }],
        });
        let manager = ObservationManager::new(&frames, &pairs, &reconstruction);
        let weak = RegistrationChoice {
            image: 1,
            pose: SE3::identity(),
            camera: CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0),
            source: "pnp",
            pnp_inliers: 0,
            inlier_ratio: 0.0,
            visible_points: 1,
            visible_points_ratio: 0.5,
            mean_error_px: f32::INFINITY,
            pair_rot_error: 0.0,
            structureless_inliers: Vec::new(),
        };
        let strong = RegistrationChoice {
            image: 2,
            pose: SE3::identity(),
            camera: CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0),
            source: "pnp",
            pnp_inliers: 0,
            inlier_ratio: 0.0,
            visible_points: 2,
            visible_points_ratio: 1.0,
            mean_error_px: f32::INFINITY,
            pair_rot_error: 0.0,
            structureless_inliers: Vec::new(),
        };

        assert!(registration_score(&strong, &manager) > registration_score(&weak, &manager));
    }

    #[test]
    fn local_bundle_selection_uses_shared_points_and_excludes_gauge_image() {
        let frames = vec![
            minimal_frame(0, "gauge.jpg"),
            minimal_frame(1, "registered.jpg"),
            minimal_frame(2, "neighbor.jpg"),
            minimal_frame(3, "weak.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        for image in 0..4 {
            reconstruction.poses[image] = Some(SE3::identity());
        }
        reconstruction.poses[2] = Some(SE3::from_quat_translation(
            glam::Quat::IDENTITY,
            glam::Vec3::new(1.0, 0.0, 0.0),
        ));
        reconstruction.poses[3] = Some(SE3::from_quat_translation(
            glam::Quat::IDENTITY,
            glam::Vec3::new(0.01, 0.0, 0.0),
        ));
        for point_id in 0..4 {
            reconstruction.points.push(Point3D {
                xyz: [point_id as f32 * 0.1, 0.0, 2.0],
                color: [0, 0, 0],
                error: 0.0,
                track: vec![
                    TrackObservation {
                        image: 1,
                        feature: 0,
                    },
                    TrackObservation {
                        image: 2,
                        feature: 0,
                    },
                    TrackObservation {
                        image: 0,
                        feature: 0,
                    },
                ],
            });
            reconstruction.point_ids.push(point_id as u64 + 1);
        }
        reconstruction.points.push(Point3D {
            xyz: [1.0, 0.0, 2.0],
            color: [0, 0, 0],
            error: 0.0,
            track: vec![
                TrackObservation {
                    image: 1,
                    feature: 1,
                },
                TrackObservation {
                    image: 3,
                    feature: 1,
                },
            ],
        });
        reconstruction.point_ids.push(5);
        reconstruction.points.push(Point3D {
            xyz: [0.5, 0.0, 2.0],
            color: [0, 0, 0],
            error: 1.0,
            track: (0..16)
                .map(|idx| TrackObservation {
                    image: if idx == 0 { 1 } else { 2 },
                    feature: 1,
                })
                .collect(),
        });
        reconstruction.point_ids.push(6);

        let selection = select_local_bundle(&reconstruction, 1, 0, 2, 2).unwrap();

        assert_eq!(selection.variable_images, vec![1, 2]);
        assert_eq!(selection.local_images, vec![2]);
        assert_eq!(selection.point_ids, vec![0, 1, 2, 3, 4]);
        assert_eq!(selection.constant_point_ids, vec![5]);
        assert_eq!(selection.stable_point_ids, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn track_filter_removes_negative_depth_observation_and_keeps_valid_track() {
        let mut frames = vec![
            minimal_frame(0, "front.jpg"),
            minimal_frame(1, "behind.jpg"),
            minimal_frame(2, "side.jpg"),
        ];
        frames[0].keypoints[0] = rustslam::KeyPoint::new(50.0, 50.0);
        frames[1].keypoints[0] = rustslam::KeyPoint::new(50.0, 50.0);
        frames[2].keypoints[0] = rustslam::KeyPoint::new(75.0, 50.0);
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::from_quat_translation(
            glam::Quat::IDENTITY,
            glam::Vec3::new(0.0, 0.0, -4.0),
        ));
        reconstruction.poses[2] = Some(SE3::from_quat_translation(
            glam::Quat::IDENTITY,
            glam::Vec3::new(1.0, 0.0, 0.0),
        ));
        reconstruction.observations[0][0] = Some(0);
        reconstruction.observations[1][0] = Some(0);
        reconstruction.observations[2][0] = Some(0);
        reconstruction.point_ids.push(7);
        reconstruction.points.push(Point3D {
            xyz: [0.0, 0.0, 2.0],
            color: [0, 0, 0],
            error: 0.0,
            track: vec![
                TrackObservation {
                    image: 0,
                    feature: 0,
                },
                TrackObservation {
                    image: 1,
                    feature: 0,
                },
                TrackObservation {
                    image: 2,
                    feature: 0,
                },
            ],
        });

        let removed = filter_reprojection_tracks(
            &frames,
            &[],
            &mut reconstruction,
            &MapperConfig {
                max_reprojection_error_px: 1.0,
                ..MapperConfig::default()
            },
        );

        assert_eq!(removed, 1);
        assert_eq!(reconstruction.points.len(), 1);
        assert_eq!(
            reconstruction.points[0]
                .track
                .iter()
                .map(|obs| obs.image)
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert_eq!(reconstruction.observations[1][0], None);
        assert_eq!(reconstruction.point_ids, vec![7]);
    }

    #[test]
    fn track_filter_removes_tracks_observed_by_bogus_camera() {
        let mut frames = vec![minimal_frame(0, "good.jpg"), minimal_frame(1, "bogus.jpg")];
        frames[0].keypoints[0] = rustslam::KeyPoint::new(50.0, 50.0);
        frames[1].keypoints[0] = rustslam::KeyPoint::new(75.0, 50.0);
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.cameras = vec![
            CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0),
            CameraModel::new_pinhole(100, 100, 1.0, 1.0, 50.0, 50.0),
        ];
        reconstruction.camera_ids = vec![1, 2];
        reconstruction.image_camera_indices = vec![0, 1];
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::from_quat_translation(
            glam::Quat::IDENTITY,
            glam::Vec3::new(1.0, 0.0, 0.0),
        ));
        reconstruction.observations[0][0] = Some(0);
        reconstruction.observations[1][0] = Some(0);
        reconstruction.point_ids.push(13);
        reconstruction.points.push(Point3D {
            xyz: [0.0, 0.0, 2.0],
            color: [0, 0, 0],
            error: 0.0,
            track: vec![
                TrackObservation {
                    image: 0,
                    feature: 0,
                },
                TrackObservation {
                    image: 1,
                    feature: 0,
                },
            ],
        });

        let removed =
            filter_reprojection_tracks(&frames, &[], &mut reconstruction, &MapperConfig::default());

        assert_eq!(removed, 2);
        assert!(reconstruction.points.is_empty());
        assert_eq!(reconstruction.observations[0][0], None);
        assert_eq!(reconstruction.observations[1][0], None);
    }

    #[test]
    fn track_filter_removes_small_triangulation_angle_track() {
        let mut frames = vec![minimal_frame(0, "a.jpg"), minimal_frame(1, "b.jpg")];
        frames[0].keypoints[0] = rustslam::KeyPoint::new(50.0, 50.0);
        frames[1].keypoints[0] = rustslam::KeyPoint::new(50.0025, 50.0);
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::from_quat_translation(
            glam::Quat::IDENTITY,
            glam::Vec3::new(0.001, 0.0, 0.0),
        ));
        reconstruction.observations[0][0] = Some(0);
        reconstruction.observations[1][0] = Some(0);
        reconstruction.point_ids.push(9);
        reconstruction.points.push(Point3D {
            xyz: [0.0, 0.0, 20.0],
            color: [0, 0, 0],
            error: 0.0,
            track: vec![
                TrackObservation {
                    image: 0,
                    feature: 0,
                },
                TrackObservation {
                    image: 1,
                    feature: 0,
                },
            ],
        });

        let removed = filter_reprojection_tracks(
            &frames,
            &[],
            &mut reconstruction,
            &MapperConfig {
                max_reprojection_error_px: 1.0,
                ..MapperConfig::default()
            },
        );

        assert_eq!(removed, 2);
        assert!(reconstruction.points.is_empty());
        assert_eq!(reconstruction.observations[0][0], None);
        assert_eq!(reconstruction.observations[1][0], None);
    }

    #[test]
    fn track_filter_prunes_short_tracks_when_min_length_is_raised() {
        let mut frames = vec![minimal_frame(0, "a.jpg"), minimal_frame(1, "b.jpg")];
        frames[0].keypoints[0] = rustslam::KeyPoint::new(50.0, 50.0);
        frames[1].keypoints[0] = rustslam::KeyPoint::new(75.0, 50.0);
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::from_quat_translation(
            glam::Quat::IDENTITY,
            glam::Vec3::new(1.0, 0.0, 0.0),
        ));
        reconstruction.observations[0][0] = Some(0);
        reconstruction.observations[1][0] = Some(0);
        reconstruction.point_ids.push(11);
        reconstruction.points.push(Point3D {
            xyz: [0.0, 0.0, 2.0],
            color: [0, 0, 0],
            error: 0.0,
            track: vec![
                TrackObservation {
                    image: 0,
                    feature: 0,
                },
                TrackObservation {
                    image: 1,
                    feature: 0,
                },
            ],
        });

        let removed = filter_reprojection_tracks_with_policy(
            &frames,
            &[],
            &mut reconstruction,
            &MapperConfig {
                max_reprojection_error_px: 1.0,
                ..MapperConfig::default()
            },
            1.0,
            0.1,
            3,
        );

        assert_eq!(removed, 2);
        assert!(reconstruction.points.is_empty());
    }

    fn test_pair(
        left: usize,
        right: usize,
        inliers: usize,
        triangulated: usize,
        median_triangulation_angle_deg: f32,
        translation: [f32; 3],
    ) -> PairGeometry {
        PairGeometry {
            left,
            right,
            two_view_config: crate::database::COLMAP_TWO_VIEW_CALIBRATED,
            matches: Vec::new(),
            inlier_matches: Vec::new(),
            relative_pose: SE3::from_quat_translation(
                glam::Quat::IDENTITY,
                glam::Vec3::from_array(translation),
            ),
            inliers,
            triangulated,
            mean_reprojection_error_px: 1.0,
            rotation_deg: 0.0,
            median_triangulation_angle_deg,
            pose_graph_only: false,
        }
    }

    fn pair_with_inliers(left: usize, right: usize, matches: &[(u32, u32)]) -> PairGeometry {
        let mut pair = test_pair(left, right, matches.len(), 0, 1.0, [1.0, 0.0, 0.0]);
        pair.inlier_matches = matches
            .iter()
            .map(|&(query_idx, train_idx)| rustslam::Match {
                query_idx,
                train_idx,
                distance: 0.0,
            })
            .collect();
        pair
    }

    fn structureless_frames(count: usize) -> Vec<ImageFrame> {
        (0..count)
            .map(|idx| {
                let mut frame = minimal_frame(idx, &format!("image_{idx}.jpg"));
                frame.keypoints = (0..64)
                    .map(|feature| {
                        let col = (feature % 8) as f32;
                        let row = (feature / 8) as f32;
                        rustslam::KeyPoint::new(10.0 + col * 8.0, 12.0 + row * 7.0)
                    })
                    .collect();
                frame.colors = vec![[0, 0, 0]; frame.keypoints.len()];
                frame
            })
            .collect()
    }

    fn structureless_world_poses() -> [SE3; 4] {
        [
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(0.02),
                glam::Vec3::new(0.0, 0.0, 0.0),
            ),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(0.11),
                glam::Vec3::new(-0.45, 0.02, 0.04),
            ),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(-0.05),
                glam::Vec3::new(0.75, -0.01, 0.02),
            ),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(0.04),
                glam::Vec3::new(1.4, 0.03, -0.03),
            ),
        ]
    }

    fn structureless_pair_from_poses(
        left: usize,
        right: usize,
        left_pose: SE3,
        right_pose: SE3,
        inliers: usize,
    ) -> PairGeometry {
        let matches = (0..inliers as u32)
            .map(|idx| rustslam::Match {
                query_idx: idx,
                train_idx: idx,
                distance: 0.0,
            })
            .collect::<Vec<_>>();
        PairGeometry {
            left,
            right,
            two_view_config: crate::database::COLMAP_TWO_VIEW_CALIBRATED,
            matches: matches.clone(),
            inlier_matches: matches,
            relative_pose: right_pose.compose(&left_pose.inverse()),
            inliers,
            triangulated: 0,
            mean_reprojection_error_px: 0.25,
            rotation_deg: 0.0,
            median_triangulation_angle_deg: 2.0,
            pose_graph_only: false,
        }
    }

    fn camera_priors(reconstruction: &Reconstruction) -> Vec<CameraModel> {
        reconstruction.cameras.clone()
    }

    fn camera_prior_focal_flags(reconstruction: &Reconstruction, value: bool) -> Vec<bool> {
        vec![value; reconstruction.cameras.len()]
    }

    fn registration_stats(reconstruction: &Reconstruction) -> RegistrationStats {
        RegistrationStats::from_reconstruction(reconstruction)
    }

    fn test_reconstruction(frames: &[ImageFrame]) -> Reconstruction {
        Reconstruction {
            camera: CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0),
            cameras: vec![CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0)],
            camera_ids: vec![1],
            rigs: Vec::new(),
            frames: Vec::new(),
            image_names: frames.iter().map(|frame| frame.name.clone()).collect(),
            image_paths: frames.iter().map(|frame| frame.path.clone()).collect(),
            image_ids: (0..frames.len()).map(|idx| idx as u32 + 1).collect(),
            image_camera_indices: vec![0; frames.len()],
            image_frame_indices: vec![None; frames.len()],
            poses: vec![None; frames.len()],
            observations: frames
                .iter()
                .map(|frame| vec![None; frame.keypoints.len()])
                .collect(),
            keypoints: frames.iter().map(|frame| frame.keypoints.clone()).collect(),
            point_ids: Vec::new(),
            points: Vec::new(),
        }
    }

    fn project_test_point(camera: CameraModel, pose: SE3, point: [f32; 3]) -> rustslam::KeyPoint {
        let p = pose.transform_point(&point);
        let xy = camera
            .img_from_cam(p[0] as f64, p[1] as f64, p[2] as f64)
            .unwrap();
        rustslam::KeyPoint::new(xy[0] as f32, xy[1] as f32)
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
