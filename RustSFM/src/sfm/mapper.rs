use crate::colmap::{
    export_colmap, export_colmap_sparse_snapshot, export_colmap_with_sparse_index,
    read_colmap_cameras, read_colmap_sparse_model, ColmapDataId, ColmapRig, ColmapRigSensor,
    ColmapRigid3, ColmapSensorId, ColmapSensorType,
};
use crate::correspondence_graph::{image_pair_to_pair_id, CorrespondenceGraph, ImagePairId};
use crate::database::{ColmapDatabase, ColmapTwoViewGeometry, DatabaseCache, DatabaseCacheOptions};
use crate::feature_matching::{generate_matching_pairs, MatchingPairStrategy};
use crate::generalized_pose::{
    estimate_generalized_absolute_pose, estimate_structureless_absolute_pose,
    GeneralizedAbsolutePoseEstimationOptions, GeneralizedAbsolutePoseProblem, GeneralizedPoseError,
    StructureLessAbsolutePoseEstimationOptions, StructureLessAbsolutePoseProblem,
};
use crate::geometry::{
    camera_center, estimate_pair_geometry_with_options_and_cameras,
    mean_pair_reprojection_error_with_cameras, pose_from_rotation_center, pose_rotation,
    pose_with_flipped_translation, relative_rotation_deg, PairEstimationOptions,
};
use crate::global_mapper::{
    run_global_reconstructions, GlobalMapperOptions, GlobalReconstructionOptions,
    GlobalRefinementOptions,
};
use crate::incremental_triangulator::{
    IncrementalTriangulator, IncrementalTriangulatorOptions, IncrementalTriangulatorState,
    TriangulationReport,
};
use crate::joint_global_positioning::JointGlobalPositioningOptions;
use crate::observation_manager::ObservationManager;
use crate::pose_graph::initialize_pose_graph;
use crate::rotation_averaging::RotationAveragingOptions;
use crate::sift::{match_sift_guided_with_options, match_sift_with_options, SiftMatchingOptions};
use crate::task::{SfmTaskContext, SfmTaskEventKind, SfmTaskOperation, SfmTaskStage, SfmTaskStop};
use crate::track_establishment::TrackEstablishmentOptions;
use crate::track_triangulation::TrackTriangulationOptions;
use crate::types::{
    colmap_camera_model_extra_idxs, colmap_camera_model_focal_idxs,
    colmap_camera_model_principal_point_idxs, CameraModel, DataId, Frame, ImageFrame, PairGeometry,
    Point3D, Reconstruction, Rig, RigSensor, Rigid3, SensorId, SensorType, TrackObservation,
};
use crate::view_graph_calibration::ViewGraphCalibrationOptions;
use crate::view_graph_splitting::ViewGraphComponentSplittingOptions;
use crate::wide::{match_wide_mutual, match_wide_mutual_indices};
use anyhow::{bail, Context, Result};
use image::ImageReader;
use nalgebra::{DMatrix, DVector, Matrix3, Matrix3x4, SMatrix, SVector, Vector3};
use rayon::prelude::*;
use rustslam::features::HammingMatcher;
use rustslam::tracker::{PnPModelScorer, PnPProblem, PnPSolver};
use rustslam::{FeatureMatcher, SE3};
use std::borrow::Cow;
use std::cmp::Ordering as CmpOrdering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::Instant;

#[path = "mapper/bundle_adjustment.rs"]
mod bundle_adjustment;
#[path = "mapper/config.rs"]
mod config;
#[path = "mapper/database_io.rs"]
mod database_io;
#[path = "mapper/diagnostics.rs"]
mod diagnostics;
#[path = "mapper/image_features.rs"]
mod image_features;
#[path = "mapper/pipeline_types.rs"]
mod pipeline_types;
#[path = "mapper/reconstruction_input.rs"]
mod reconstruction_input;
#[path = "mapper/state.rs"]
mod state;

use bundle_adjustment::*;
pub use config::{FeatureType, ImageSelectionMethod, MapperConfig, ReconstructionSummary};
use database_io::{populate_local_matching_database, write_pair_geometries_to_database};
use diagnostics::{
    pair_config_summary, pair_connectivity_summary, pair_quality_summary,
    pair_reference_error_summary, pair_two_view_metadata_summary,
};
use image_features::extract_frames;
use pipeline_types::MapperEventBridge;
pub use pipeline_types::{
    IncrementalPipelineCallback, IncrementalPipelineMapResult, IncrementalPipelineResult,
    IncrementalPipelineStatus, PipelineCallbackEvent, PipelineCallbackSink,
};
use reconstruction_input::{
    apply_color_extraction_policy, collect_images, database_camera_setup, database_frames,
    fallback_camera, load_mapper_database_for_paths, local_image_camera_setup,
    resolve_mapper_database_path, sample_keypoint_colors, sensor_id_from_colmap,
    setup_for_reconstruction_attempt,
};
#[cfg(test)]
use reconstruction_input::{
    apply_database_keypoints, default_database_candidates, load_mapper_database,
};
pub use reconstruction_input::{reference_camera_setup, ReconstructionSeed, ReferenceCameraSetup};
pub use state::InitialPairFailure;
use state::{IncrementalMapperSession, InitialPairSelectionState, RegistrationStats};

pub(crate) type DynPnPModelScorer = dyn PnPModelScorer<Error = anyhow::Error>;

#[derive(Debug)]
struct GpuPnpMapperError(String);

impl std::fmt::Display for GpuPnpMapperError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for GpuPnpMapperError {}

fn gpu_pnp_mapper_error(message: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(GpuPnpMapperError(message.into()))
}

/// Keeps fixed-intrinsics GPU PnP fail-fast when the backend could not initialize,
/// while allowing the unknown-focal route to run its own GPU-to-CPU fallback.
#[cfg(feature = "gpu-wgpu")]
struct UnavailableGpuPnpModelScorer {
    error: String,
}

#[cfg(feature = "gpu-wgpu")]
impl PnPModelScorer for UnavailableGpuPnpModelScorer {
    type Error = anyhow::Error;

    fn prepare(
        &mut self,
        _normalized_points: &[[f32; 2]],
        _object_points: &[[f32; 3]],
        _threshold: f32,
    ) -> Result<(), Self::Error> {
        Err(gpu_pnp_mapper_error(format!(
            "failed to initialize gpu pnp scorer: {}",
            self.error
        )))
    }

    fn score_models(
        &mut self,
        _models: &[SE3],
    ) -> Result<Vec<rustslam::tracker::PnPModelSupport>, Self::Error> {
        unreachable!("GPU PnP initialization failure must be reported by prepare")
    }

    fn inlier_mask(&mut self, _model: &SE3) -> Result<Vec<bool>, Self::Error> {
        unreachable!("GPU PnP initialization failure must be reported by prepare")
    }
}

#[cfg(feature = "gpu-wgpu")]
#[derive(Debug)]
struct GpuPnPFocalFallbackError {
    gpu_error: Option<anyhow::Error>,
}

#[cfg(feature = "gpu-wgpu")]
impl std::fmt::Display for GpuPnPFocalFallbackError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("CPU PnP-focal fallback could not solve")?;
        if let Some(error) = &self.gpu_error {
            write!(formatter, "; GPU PnP-focal error: {error:#}")?;
        }
        Ok(())
    }
}

#[cfg(feature = "gpu-wgpu")]
impl std::error::Error for GpuPnPFocalFallbackError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.gpu_error
            .as_ref()
            .map(|error| error.as_ref() as &(dyn std::error::Error + 'static))
    }
}

#[derive(Debug, Clone)]
pub struct DatabasePairMatches {
    pub left: usize,
    pub right: usize,
    pub matches: Vec<rustslam::Match>,
}

#[derive(Debug, Clone)]
pub(crate) struct SingleTargetRegistrationCandidate {
    pub reconstruction: Reconstruction,
    pub inlier_count: usize,
    pub inlier_ratio: f64,
    pub mean_reprojection_error: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct SingleTargetRegistrationAttempt {
    pub candidate: Option<SingleTargetRegistrationCandidate>,
    pub debug_log: Vec<String>,
}

pub(crate) fn register_single_target_from_seed(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    setup: &ReferenceCameraSetup,
    target_image: usize,
    config: &MapperConfig,
    pnp_scorer: Option<&mut DynPnPModelScorer>,
) -> Result<SingleTargetRegistrationAttempt> {
    if config.reference.is_none() {
        bail!("single-target registration requires a reference model");
    }
    if !config.fix_existing_frames {
        bail!("single-target registration requires fix_existing_frames=true");
    }
    if target_image >= frames.len() {
        bail!("target image index {target_image} is out of range");
    }
    let camera = setup
        .cameras
        .first()
        .copied()
        .with_context(|| "reference setup has no cameras")?;
    let mut reconstruction = Reconstruction {
        camera,
        cameras: setup.cameras.clone(),
        camera_ids: setup.camera_ids.clone(),
        rigs: setup.rigs.clone(),
        frames: setup.frames.clone(),
        image_names: frames.iter().map(|frame| frame.name.clone()).collect(),
        image_paths: frames.iter().map(|frame| frame.path.clone()).collect(),
        image_ids: setup.image_ids.clone(),
        image_camera_indices: setup.image_camera_indices.clone(),
        image_frame_indices: setup.image_frame_indices.clone(),
        poses: vec![None; frames.len()],
        observations: frames
            .iter()
            .map(|frame| vec![None; frame.keypoints.len()])
            .collect(),
        keypoints: frames.iter().map(|frame| frame.keypoints.clone()).collect(),
        point_ids: Vec::new(),
        points: Vec::new(),
    };
    let seed = setup
        .seed_reconstruction
        .clone()
        .with_context(|| "reference model did not provide a sparse reconstruction seed")?;
    apply_reconstruction_seed(&mut reconstruction, seed, frames);
    if reconstruction.poses[target_image].is_some() {
        bail!("target image is already registered in the reference model");
    }
    for pair in pairs {
        if pair.left != target_image && pair.right != target_image {
            bail!("single-target pair set contains a pair that omits the target");
        }
        let support = if pair.left == target_image {
            pair.right
        } else {
            pair.left
        };
        if reconstruction
            .poses
            .get(support)
            .and_then(|pose| *pose)
            .is_none()
        {
            bail!("single-target pair references unregistered support image {support}");
        }
    }

    let mut registration_stats = RegistrationStats::from_reconstruction(&reconstruction);
    registration_stats.set_existing_registration_units_from_reconstruction(&reconstruction);
    let mut observation_manager = ObservationManager::new(frames, pairs, &reconstruction);
    let graph = observation_manager.correspondence_graph();
    let mut telemetry = IncrementalRegistrationTelemetry::default();
    let Some(absolute_pose) = solve_absolute_pose_with_pnp_scorer(
        target_image,
        frames,
        pairs,
        &reconstruction,
        config,
        &setup.cameras,
        &setup.camera_has_prior_focal_length,
        &registration_stats,
        graph,
        pnp_scorer,
        &mut telemetry,
    )?
    else {
        return Ok(SingleTargetRegistrationAttempt {
            candidate: None,
            debug_log: vec![telemetry.format_log()],
        });
    };
    if absolute_pose
        .pose
        .translation()
        .iter()
        .chain(absolute_pose.pose.quaternion().iter())
        .any(|value| !value.is_finite())
        || !absolute_pose.inlier_ratio.is_finite()
        || !absolute_pose.mean_error_px.is_finite()
    {
        return Ok(SingleTargetRegistrationAttempt {
            candidate: None,
            debug_log: vec![telemetry.format_log()],
        });
    }
    apply_image_camera(&mut reconstruction, target_image, absolute_pose.camera);
    observation_manager.register_image(
        frames,
        pairs,
        &mut reconstruction,
        target_image,
        absolute_pose.pose,
    );
    for inlier in &absolute_pose.point_inliers {
        observation_manager.add_observation(
            frames,
            pairs,
            &mut reconstruction,
            inlier.point_id,
            TrackObservation {
                image: target_image,
                feature: inlier.feature,
            },
        );
    }
    Ok(SingleTargetRegistrationAttempt {
        candidate: Some(SingleTargetRegistrationCandidate {
            reconstruction,
            inlier_count: absolute_pose.inliers,
            inlier_ratio: f64::from(absolute_pose.inlier_ratio),
            mean_reprojection_error: f64::from(absolute_pose.mean_error_px),
        }),
        debug_log: vec![telemetry.format_log()],
    })
}

pub(crate) fn register_single_target_from_database_with_pnp_scorer(
    input: &Path,
    database: &Path,
    reference: &Path,
    target_name: &str,
    support_names: &[String],
    config: &MapperConfig,
    pnp_scorer: Option<&mut DynPnPModelScorer>,
) -> Result<SingleTargetRegistrationAttempt> {
    let reference_model = read_colmap_sparse_model(reference)?;
    let registered_names = reference_model
        .reconstruction
        .image_names
        .iter()
        .zip(&reference_model.reconstruction.poses)
        .filter_map(|(name, pose)| pose.is_some().then_some(name.clone()))
        .collect::<BTreeSet<_>>();
    for support in support_names {
        if !registered_names.contains(support) {
            bail!("support image '{support}' is not registered in the reference model");
        }
    }
    if registered_names.contains(target_name) {
        bail!("target image '{target_name}' is already registered");
    }
    let mut names = registered_names.into_iter().collect::<Vec<_>>();
    names.push(target_name.to_owned());
    let paths = names
        .iter()
        .map(|name| input.join(name))
        .collect::<Vec<_>>();
    for path in &paths {
        if !path.is_file() {
            bail!("missing registration input image {}", path.display());
        }
    }
    let database_input =
        load_mapper_database_for_paths(Some(database), &paths, config.min_matches)?
            .with_context(|| "single-target registration database was not loaded")?;
    let frames = database_frames(&paths, &database_input)?;
    let mut setup = reference_camera_setup(reference, &paths)?;
    let database_setup = database_camera_setup(&database_input.cache, &paths)?;
    let target_image = names.len() - 1;
    let database_camera_index = database_setup.image_camera_indices[target_image];
    let database_camera_id = database_setup.camera_ids[database_camera_index];
    let target_camera_index = if let Some(index) = setup
        .camera_ids
        .iter()
        .position(|&camera_id| camera_id == database_camera_id)
    {
        index
    } else {
        setup.camera_ids.push(database_camera_id);
        setup
            .cameras
            .push(database_setup.cameras[database_camera_index]);
        setup
            .camera_has_prior_focal_length
            .push(database_setup.camera_has_prior_focal_length[database_camera_index]);
        setup.cameras.len() - 1
    };
    setup.image_ids[target_image] = database_setup.image_ids[target_image];
    setup.image_camera_indices[target_image] = target_camera_index;
    setup.image_frame_indices[target_image] = database_setup.image_frame_indices[target_image];
    let camera = setup
        .cameras
        .first()
        .copied()
        .with_context(|| "reference setup has no cameras")?;
    let all_pairs = estimate_database_pair_geometries(
        &frames,
        &database_input.cache,
        &database_input.two_view_geometries,
        camera,
        Some(&setup),
        config,
    )?;
    if frames[target_image].name != target_name {
        bail!("target image '{target_name}' is missing from database frames");
    }
    let support_names = support_names
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let pairs = all_pairs
        .into_iter()
        .filter(|pair| {
            let other = if pair.left == target_image {
                Some(pair.right)
            } else if pair.right == target_image {
                Some(pair.left)
            } else {
                None
            };
            other.is_some_and(|other| support_names.contains(frames[other].name.as_str()))
        })
        .collect::<Vec<_>>();
    if pairs.is_empty() {
        return Ok(SingleTargetRegistrationAttempt {
            candidate: None,
            debug_log: Vec::new(),
        });
    }
    let mut target_config = config.clone();
    target_config.reference = Some(reference.to_path_buf());
    target_config.database = Some(database.to_path_buf());
    target_config.fix_existing_frames = true;
    target_config.local_ba = false;
    target_config.global_ba = false;
    validate_gpu_pnp_config(&target_config, false)?;
    register_single_target_from_seed(
        &frames,
        &pairs,
        &setup,
        target_image,
        &target_config,
        pnp_scorer,
    )
}

pub fn run_reconstruction(config: &MapperConfig) -> Result<ReconstructionSummary> {
    let mut events = MapperEventBridge::Silent;
    run_reconstruction_impl(config, &mut events)
}

fn validate_gpu_pnp_config(config: &MapperConfig, has_global_mapper: bool) -> Result<()> {
    if !config.use_gpu_pnp {
        return Ok(());
    }
    if has_global_mapper {
        bail!("gpu pnp is only supported by the incremental mapper, not the global mapper");
    }
    #[cfg(not(feature = "gpu-wgpu"))]
    bail!("gpu pnp requires RustSFM to be compiled with the gpu-wgpu feature");
    #[cfg(feature = "gpu-wgpu")]
    Ok(())
}

fn record_gpu_pnp_route_fallback(
    config: &MapperConfig,
    telemetry: &mut IncrementalRegistrationTelemetry,
    reason: &str,
) {
    // The GPU PnP pipeline only covers structure-based central PnP-focal.
    // Generalized-rig and structureless routes are always solved by the
    // existing CPU solvers, so GPU PnP users fall back instead of failing.
    #[cfg(feature = "gpu-wgpu")]
    if config.use_gpu_pnp {
        telemetry.record_gpu_pnp_focal_fallback(reason);
    }
    #[cfg(not(feature = "gpu-wgpu"))]
    let _ = (config, telemetry, reason);
}

pub(crate) fn create_gpu_pnp_scorer(
    config: &MapperConfig,
) -> Result<Option<Box<DynPnPModelScorer>>> {
    if !config.use_gpu_pnp {
        return Ok(None);
    }
    #[cfg(feature = "gpu-wgpu")]
    {
        let scorer = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::gpu::WgpuPnpModelScorer::try_new()
        }));
        match scorer {
            Ok(Ok(scorer)) => Ok(Some(Box::new(scorer))),
            Ok(Err(error)) => Ok(Some(Box::new(UnavailableGpuPnpModelScorer {
                error: format!("{error:#}"),
            }))),
            Err(panic) => Ok(Some(Box::new(UnavailableGpuPnpModelScorer {
                error: format!(
                    "GPU PnP scorer panicked during initialization: {}",
                    panic_message(panic)
                ),
            }))),
        }
    }
    #[cfg(not(feature = "gpu-wgpu"))]
    {
        bail!("gpu pnp requires RustSFM to be compiled with the gpu-wgpu feature")
    }
}

pub fn run_reconstruction_with_callbacks(
    config: &MapperConfig,
    callback_sink: Option<&mut dyn PipelineCallbackSink>,
) -> Result<ReconstructionSummary> {
    let mut events = match callback_sink {
        Some(sink) => MapperEventBridge::Legacy(sink),
        None => MapperEventBridge::Silent,
    };
    run_reconstruction_impl(config, &mut events)
}

pub fn run_reconstruction_with_task(
    config: &MapperConfig,
    task: &mut SfmTaskContext<'_>,
) -> Result<ReconstructionSummary> {
    let mut events = MapperEventBridge::Task(task);
    run_reconstruction_impl(config, &mut events)
}

fn run_reconstruction_impl(
    config: &MapperConfig,
    events: &mut MapperEventBridge<'_, '_>,
) -> Result<ReconstructionSummary> {
    events.checkpoint()?;
    let mut runtime_config = config.clone();
    let config = &mut runtime_config;
    validate_gpu_pnp_config(config, config.global_mapper)?;
    let mut pnp_scorer = create_gpu_pnp_scorer(config)?;
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
        .map(|reference| {
            reference_camera_setup(reference, &paths).with_context(|| {
                format!(
                    "failed to configure reference model {}",
                    reference.display()
                )
            })
        })
        .transpose()?;
    let mut camera = if let Some(setup) = &reference_camera_setup {
        setup
            .cameras
            .first()
            .copied()
            .unwrap_or_else(|| fallback_camera(&paths[0]))
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
    let database_path = resolve_mapper_database_path(config)?;
    let mapper_database = if database_path.as_ref().is_some_and(|path| path.exists()) {
        load_mapper_database_for_paths(database_path.as_deref(), &paths, config.min_matches)?
    } else {
        None
    };
    let mut frames = if let Some(database) = mapper_database.as_ref() {
        database_frames(&paths, database)?
    } else {
        extract_frames(
            &paths,
            config.max_features,
            config.feature_type,
            &sift_extraction,
        )?
    };
    if let Some(database) = mapper_database.as_ref() {
        config.pose_priors = database.cache.pose_priors.clone();
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
        reference_camera_setup = Some(local_image_camera_setup(&frames, config)?);
        if let Some(first) = reference_camera_setup
            .as_ref()
            .and_then(|setup| setup.cameras.first())
            .copied()
        {
            camera = first;
        }
    }
    apply_color_extraction_policy(&mut frames, config.extract_colors);
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
    debug_log.push(format!("extract_colors={}", config.extract_colors));
    if let Some(path) = database_path.as_ref() {
        debug_log.push(format!("database_path={}", path.display()));
        debug_log.push(format!(
            "ignore_database_two_view_poses={}",
            config.ignore_database_two_view_poses
        ));
    } else {
        debug_log.push("database_path=<none> local_matching_fallback=true".to_string());
    }
    debug_log.extend(pair_connectivity_summary(&pairs, &frames));
    debug_log.extend(pair_config_summary(&pairs));
    debug_log.extend(pair_two_view_metadata_summary(&pairs));
    if config.write_two_view_geometries {
        if let Some(path) = database_path.as_ref() {
            let written = write_pair_geometries_to_database(path, &frames, &pairs)?;
            debug_log.push(format!("database_two_view_geometries_written={written}"));
        } else {
            debug_log
                .push("database_two_view_geometries_written=0 database_path=<none>".to_string());
        }
    }
    if config.write_database && config.local_matching {
        if let Some(path) = database_path.as_ref() {
            let setup = reference_camera_setup.as_ref().with_context(|| {
                "local matching database write requires per-image camera setup".to_string()
            })?;
            let written = populate_local_matching_database(
                path,
                &frames,
                setup,
                &pairs,
                config.feature_type,
            )?;
            debug_log.push(format!("database_populated_entries={written}"));
        } else {
            debug_log.push("database_populated_entries=0 database_path=<none>".to_string());
        }
    }
    if let Some(reference) = &config.reference {
        debug_log.extend(pair_reference_error_summary(&pairs, &frames, reference));
    }
    let incremental_start = Instant::now();
    let (mut reconstructions, mut reconstruction, pipeline_debug) = if config.global_mapper {
        let mapping_pairs = pairs
            .iter()
            .filter(|pair| !pair.pose_graph_only)
            .cloned()
            .collect::<Vec<_>>();
        let global_options = global_reconstruction_options_from_config(config);
        let results = run_global_reconstructions(&frames, &mapping_pairs, camera, &global_options)
            .context("global mapper failed to produce a reconstruction")?;
        let mut pipeline_debug = vec!["pipeline=global_mapper".to_string()];
        pipeline_debug.push(format!(
            "global_mapper components found={} selected={} reconstructed={} sizes={:?}",
            results.component_splitting.num_components,
            results.component_splitting.num_selected,
            results.component_splitting.num_reconstructed,
            results.component_splitting.selected_component_sizes,
        ));
        for (model_index, result) in results.reconstructions.iter().enumerate() {
            pipeline_debug.push(format!(
                    "global_mapper model={} component_views={:?} registered={} tracks={} observations={} triangulated={} joint={} ba_rounds={} created={} image_completed={} completed={} merged={} retriangulated={} filtered={} global_ba={}",
                    model_index,
                    result.component_views,
                    result.mapper.num_registered,
                    result.track_stats.num_tracks,
                    result.track_stats.num_observations,
                    result.triangulation_stats.num_triangulated,
                    result.used_joint_positioning,
                    result.refinement_rounds,
                    result.structure_refinement.created_points,
                    result.structure_refinement.image_completed_observations,
                    result.structure_refinement.completed_observations,
                    result.structure_refinement.merged_tracks,
                    result.structure_refinement.retriangulated_points,
                    result.structure_refinement.filtered_observations,
                    result.global_ba_success,
                ));
            pipeline_debug.push(format!(
                "global_mapper model={} rotation_residual_deg={:.4} position_residual={:.4}",
                model_index,
                result.mapper.mean_rotation_residual_deg,
                result.mapper.mean_position_residual,
            ));
        }
        if let Some(result) = results.reconstructions.first() {
            pipeline_debug.push(format!(
                    "global_mapper view_graph pairs={}/{} matches={}/{} rotation_filtered={} intrinsics_refined={}",
                    result.view_graph_calibration.pairs_out,
                    result.view_graph_calibration.pairs_in,
                    result.view_graph_calibration.matches_out,
                    result.view_graph_calibration.matches_in,
                    result.view_graph_calibration.rotation_filtered_pairs,
                    result.view_graph_calibration.intrinsics_refined,
                ));
        }
        let reconstructions = results
            .reconstructions
            .into_iter()
            .map(|result| result.reconstruction)
            .collect::<Vec<_>>();
        let reconstruction = reconstructions
            .first()
            .cloned()
            .context("global mapper kept no reconstruction")?;
        (reconstructions, reconstruction, pipeline_debug)
    } else {
        let pipeline_result = incremental_pipeline_map_with_pnp_scorer_and_events(
            &frames,
            camera,
            reference_camera_setup.as_ref(),
            &pairs,
            config,
            events,
            pnp_scorer.as_deref_mut(),
        )?;
        let reconstruction = pipeline_result
            .reconstructions
            .first()
            .cloned()
            .context("incremental pipeline kept no reconstruction")?;
        (
            pipeline_result.reconstructions,
            reconstruction,
            pipeline_result.debug_log,
        )
    };
    events.checkpoint()?;
    let incremental_elapsed_ms = incremental_start.elapsed().as_secs_f64() * 1000.0;
    debug_log.push(format!("timing_incremental_ms={incremental_elapsed_ms:.2}"));
    debug_log.extend(pipeline_debug);
    debug_log.extend(pair_quality_summary(&pairs));
    if config.pose_graph
        && !config.global_mapper
        && reconstruction.poses.iter().all(|p| p.is_some())
    {
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
            if let Ok(report) =
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
    if let Some(first) = reconstructions.first_mut() {
        *first = reconstruction.clone();
    }
    sort_reconstructions_for_colmap_output(&mut reconstructions);
    reconstruction = reconstructions
        .first()
        .cloned()
        .context("pipeline kept no reconstruction after output ordering")?;
    let registered_images = reconstruction.poses.iter().filter(|p| p.is_some()).count();
    events.checkpoint()?;
    events.emit_operation(
        SfmTaskStage::Export,
        SfmTaskOperation::ValidateArtifacts,
        SfmTaskEventKind::Started,
    );
    if events.is_task() {
        for (model_index, model) in reconstructions.iter().enumerate() {
            validate_reconstruction_for_export(model).with_context(|| {
                format!("invalid reconstruction model {model_index} before export")
            })?;
        }
    }
    events.emit_operation(
        SfmTaskStage::Export,
        SfmTaskOperation::ValidateArtifacts,
        SfmTaskEventKind::Completed,
    );
    events.checkpoint()?;
    events.emit_operation(
        SfmTaskStage::Export,
        SfmTaskOperation::WriteArtifacts,
        SfmTaskEventKind::Started,
    );
    if reconstructions.len() <= 1 {
        export_colmap(&config.output, &reconstruction, config.copy_images)?;
    } else {
        for (idx, model) in reconstructions.iter().enumerate() {
            export_colmap_with_sparse_index(&config.output, model, config.copy_images, idx)?;
        }
    }
    events.emit_operation(
        SfmTaskStage::Export,
        SfmTaskOperation::WriteArtifacts,
        SfmTaskEventKind::Completed,
    );
    events.checkpoint()?;
    Ok(ReconstructionSummary {
        images: frames.len(),
        registered_images,
        points: reconstruction.points.len(),
        pairs: pairs.len(),
        models: reconstructions.len(),
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        debug_log,
    })
}

fn validate_reconstruction_for_export(reconstruction: &Reconstruction) -> Result<()> {
    let num_images = reconstruction.image_names.len();
    if reconstruction.poses.len() != num_images
        || reconstruction.observations.len() != num_images
        || reconstruction.keypoints.len() != num_images
    {
        bail!(
            "image metadata lengths differ: names={} poses={} observations={} keypoints={}",
            num_images,
            reconstruction.poses.len(),
            reconstruction.observations.len(),
            reconstruction.keypoints.len()
        );
    }
    if reconstruction.point_ids.len() != reconstruction.points.len() {
        bail!(
            "point id count {} differs from sparse point count {}",
            reconstruction.point_ids.len(),
            reconstruction.points.len()
        );
    }
    for (image, observations) in reconstruction.observations.iter().enumerate() {
        if observations.len() != reconstruction.keypoints[image].len() {
            bail!("observation count differs from keypoint count for image {image}");
        }
        if observations
            .iter()
            .flatten()
            .any(|point| *point >= reconstruction.points.len())
        {
            bail!("observation references a missing sparse point for image {image}");
        }
    }
    Ok(())
}

fn sort_reconstructions_for_colmap_output(reconstructions: &mut [Reconstruction]) {
    reconstructions.sort_by(|left, right| {
        registered_image_count(right)
            .cmp(&registered_image_count(left))
            .then_with(|| right.points.len().cmp(&left.points.len()))
    });
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

fn reject_bad_initial_pair_after_registration(
    reconstruction: &Reconstruction,
    config: &MapperConfig,
) -> std::result::Result<(), InitialPairFailure> {
    if registered_frame_count(reconstruction) == 0 || reconstruction.points.is_empty() {
        return Err(InitialPairFailure::BadInitialPair);
    }
    if reconstruction.points.len() < config.abs_pose_min_num_inliers {
        return Err(InitialPairFailure::BadInitialPair);
    }
    Ok(())
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

fn reconstruction_with_reset_frame_cameras(
    reconstruction: &Reconstruction,
    image: usize,
    config: &MapperConfig,
    camera_priors: &[CameraModel],
) -> Reconstruction {
    let mut snapshot = reconstruction.clone();
    reset_bogus_frame_cameras_from_priors(&mut snapshot, image, config, camera_priors);
    snapshot
}

fn reset_bogus_frame_cameras_from_priors(
    reconstruction: &mut Reconstruction,
    image: usize,
    config: &MapperConfig,
    camera_priors: &[CameraModel],
) {
    let mut camera_indices = reconstruction
        .image_indices_for_registration_unit(image)
        .iter()
        .filter_map(|&frame_image| {
            reconstruction
                .image_camera_indices
                .get(frame_image)
                .copied()
        })
        .collect::<Vec<_>>();
    camera_indices.sort_unstable();
    camera_indices.dedup();
    for camera_idx in camera_indices {
        let Some(camera) = reconstruction.cameras.get(camera_idx).copied() else {
            continue;
        };
        if !camera_has_bogus_params(camera, config) {
            continue;
        }
        let Some(prior) = healthy_camera_prior(camera_idx, config, camera_priors) else {
            continue;
        };
        if let Some(slot) = reconstruction.cameras.get_mut(camera_idx) {
            *slot = prior;
        }
        if camera_idx == 0 {
            reconstruction.camera = prior;
        }
    }
}

fn build_pair_graph(
    frames: &[ImageFrame],
    camera: CameraModel,
    reference_camera_setup: Option<&ReferenceCameraSetup>,
    config: &MapperConfig,
    sift_matching: &SiftMatchingOptions,
) -> Result<Vec<PairGeometry>> {
    let matcher = HammingMatcher::new(2).with_ratio_threshold(config.match_ratio);
    let mut candidates = match config.matching_pair_strategy {
        MatchingPairStrategy::VocabTree { num_images } => {
            crate::feature_matching_db::vocab_tree_pairs_from_frames(
                frames,
                num_images,
                config.random_seed,
            )
        }
        strategy => generate_matching_pairs(frames.len(), strategy),
    };
    if config.experimental_sequence_heuristics {
        add_segment_bridge_candidates(frames.len(), &mut candidates);
    }
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
    let mut candidates = generate_matching_pairs(
        frame_count,
        MatchingPairStrategy::LocalWindow {
            window: local_window,
        },
    );
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
            let stored_pair = (!config.ignore_database_two_view_poses)
                .then(|| {
                    database_pair_geometry_from_stored_pose(
                        pair,
                        frames,
                        cache,
                        stored_geometries,
                        left_camera,
                        right_camera,
                        config,
                    )
                })
                .flatten();
            stored_pair
                .filter(|pair| keep_pair_for_mapping(pair, config))
                .or_else(|| {
                    let stored_geometry =
                        stored_database_geometry_for_pair(pair, frames, cache, stored_geometries);
                    let mut estimated = estimate_pair_geometry_with_options_and_cameras(
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
                        PairEstimationOptions {
                            ransac_random_seed: config.random_seed,
                            ..PairEstimationOptions::default()
                        },
                    )?;
                    if let Some(geometry) = stored_geometry {
                        estimated.two_view_config = geometry.config;
                        estimated.f_matrix = geometry.f_matrix.or(estimated.f_matrix);
                        estimated.e_matrix = geometry.e_matrix.or(estimated.e_matrix);
                        estimated.h_matrix = geometry.h_matrix.or(estimated.h_matrix);
                        keep_pair_for_mapping(&estimated, config).then_some(estimated)
                    } else {
                        keep_pair_for_mapping(&estimated, config).then_some(estimated)
                    }
                })
        })
        .collect::<Vec<_>>();
    Ok(pairs)
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

pub(crate) fn pair_geometry_to_colmap_two_view_geometry(
    pair: &PairGeometry,
) -> ColmapTwoViewGeometry {
    ColmapTwoViewGeometry {
        config: pair.two_view_config,
        inlier_matches: pair
            .inlier_matches
            .iter()
            .map(|match_| crate::correspondence_graph::FeatureMatch {
                point2d_idx1: match_.query_idx,
                point2d_idx2: match_.train_idx,
            })
            .collect(),
        f_matrix: pair.f_matrix,
        e_matrix: pair.e_matrix,
        h_matrix: pair.h_matrix,
        qvec: pair.qvec,
        tvec: pair.tvec,
    }
}

struct StoredPosePairMetrics {
    inlier_matches: Vec<rustslam::Match>,
    inliers: usize,
    triangulated: usize,
    mean_reprojection_error_px: f32,
    median_triangulation_angle_deg: f32,
}

fn stored_pose_pair_metrics(
    inlier_matches: &[rustslam::Match],
    left: &ImageFrame,
    right: &ImageFrame,
    pose: SE3,
    left_camera: CameraModel,
    right_camera: CameraModel,
) -> StoredPosePairMetrics {
    let mut reproj_sum = 0.0f32;
    let mut reproj_count = 0usize;
    let mut triangulation_angles = Vec::new();
    for m in inlier_matches {
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
        if err.is_finite() {
            reproj_sum += err;
            reproj_count += 1;
        }
        if let Some(angle) = pair_triangulation_angle_deg(SE3::identity(), pose, xyz) {
            triangulation_angles.push(angle);
        }
    }
    let inliers = inlier_matches.len();
    StoredPosePairMetrics {
        inlier_matches: inlier_matches.to_vec(),
        inliers,
        triangulated: triangulation_angles.len(),
        mean_reprojection_error_px: if reproj_count > 0 {
            reproj_sum / reproj_count as f32
        } else {
            0.0
        },
        median_triangulation_angle_deg: median_f32(&mut triangulation_angles),
    }
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

fn database_image_ids_for_pair(
    pair: &DatabasePairMatches,
    frames: &[ImageFrame],
    cache: &DatabaseCache,
) -> Option<(u32, u32)> {
    let left_name = frames.get(pair.left)?.name.as_str();
    let right_name = frames.get(pair.right)?.name.as_str();
    let image_by_name = cache
        .images
        .values()
        .map(|image| (image.name.as_str(), image.image_id))
        .collect::<HashMap<_, _>>();
    let left_image_id = *image_by_name.get(left_name)?;
    let right_image_id = *image_by_name.get(right_name)?;
    Some((left_image_id, right_image_id))
}

fn stored_database_geometry_for_pair(
    pair: &DatabasePairMatches,
    frames: &[ImageFrame],
    cache: &DatabaseCache,
    stored_geometries: &HashMap<ImagePairId, ColmapTwoViewGeometry>,
) -> Option<ColmapTwoViewGeometry> {
    let (left_image_id, right_image_id) = database_image_ids_for_pair(pair, frames, cache)?;
    let pair_id =
        crate::correspondence_graph::image_pair_to_pair_id(left_image_id, right_image_id).ok()?;
    let mut geometry = stored_geometries.get(&pair_id)?.clone();
    if crate::correspondence_graph::should_swap_image_pair(left_image_id, right_image_id) {
        geometry.invert();
    }
    is_usable_stored_database_geometry(&geometry).then_some(geometry)
}

fn is_usable_stored_database_geometry(geometry: &ColmapTwoViewGeometry) -> bool {
    if geometry.inlier_matches.is_empty() {
        return false;
    }
    !matches!(
        geometry.config,
        crate::database::COLMAP_TWO_VIEW_UNDEFINED
            | crate::database::COLMAP_TWO_VIEW_DEGENERATE
            | crate::database::COLMAP_TWO_VIEW_WATERMARK
            | crate::database::COLMAP_TWO_VIEW_MULTIPLE
    )
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
                ransac_random_seed: config.random_seed,
                expand_dense_inliers: false,
            },
        )
    } else {
        estimate_pair_geometry_with_options_and_cameras(
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
            PairEstimationOptions {
                ransac_random_seed: config.random_seed,
                ..PairEstimationOptions::default()
            },
        )
    }?;
    if config.feature_type == FeatureType::Sift
        && sift_matching.guided_matching
        && !is_ring_bridge_candidate(left, right)
    {
        if let Some(f_matrix) = pair.f_matrix {
            let guided = match_sift_guided_with_options(
                &frames[left].sift,
                &frames[right].sift,
                &f_matrix,
                sift_matching,
            );
            if guided.len() >= config.min_matches {
                if let Some(refined) = estimate_pair_geometry_with_options_and_cameras(
                    left,
                    right,
                    &frames[left],
                    &frames[right],
                    &guided,
                    left_camera,
                    right_camera,
                    config.essential_threshold_px,
                    config.essential_iterations,
                    config.min_inliers,
                    config.min_triangulated,
                    PairEstimationOptions {
                        ransac_random_seed: config.random_seed,
                        ..PairEstimationOptions::default()
                    },
                ) {
                    if refined.inliers >= pair.inliers {
                        pair = refined;
                    }
                }
            }
        }
    }
    if is_ring_bridge_candidate(left, right) {
        pair.pose_graph_only = true;
    }
    keep_pair_for_mapping(&pair, config).then_some(pair)
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
    let geometry = stored_database_geometry_for_pair(pair, frames, cache, stored_geometries)?;
    let inlier_matches = geometry
        .inlier_matches
        .iter()
        .filter(|match_| {
            (match_.point2d_idx1 as usize) < frames[pair.left].keypoints.len()
                && (match_.point2d_idx2 as usize) < frames[pair.right].keypoints.len()
        })
        .map(|match_| rustslam::Match {
            query_idx: match_.point2d_idx1,
            train_idx: match_.point2d_idx2,
            distance: 0.0,
        })
        .collect::<Vec<_>>();
    if inlier_matches.len() < config.min_inliers {
        return None;
    }
    let pose = stored_two_view_pose(&geometry)?;
    let metrics = stored_pose_pair_metrics(
        &inlier_matches,
        &frames[pair.left],
        &frames[pair.right],
        pose,
        left_camera,
        right_camera,
    );
    Some(PairGeometry {
        left: pair.left,
        right: pair.right,
        two_view_config: geometry.config,
        f_matrix: geometry.f_matrix,
        e_matrix: geometry.e_matrix,
        h_matrix: geometry.h_matrix,
        qvec: geometry.qvec,
        tvec: geometry.tvec,
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

fn limited_indices(indices: &[usize], limit: usize) -> &[usize] {
    &indices[..indices.len().min(limit)]
}

fn keep_pair_for_mapping(pair: &PairGeometry, config: &MapperConfig) -> bool {
    if matches!(
        pair.two_view_config,
        crate::database::COLMAP_TWO_VIEW_UNDEFINED
            | crate::database::COLMAP_TWO_VIEW_DEGENERATE
            | crate::database::COLMAP_TWO_VIEW_WATERMARK
            | crate::database::COLMAP_TWO_VIEW_MULTIPLE
    ) {
        return false;
    }
    if !pair.mean_reprojection_error_px.is_finite()
        || pair.inliers < config.min_inliers
        || pair.triangulated < config.min_triangulated
    {
        return false;
    }

    let offset = pair.right.abs_diff(pair.left);
    if is_ring_bridge_candidate(pair.left, pair.right) {
        return pair.inliers >= config.min_inliers.max(40)
            && pair.mean_reprojection_error_px <= config.max_reprojection_error_px.min(1.0)
            && pair.median_triangulation_angle_deg >= 0.75;
    }
    if offset <= 1 {
        return pair.mean_reprojection_error_px <= config.max_reprojection_error_px;
    }
    pair.inliers >= config.min_inliers.max(40)
        && pair.mean_reprojection_error_px <= config.max_reprojection_error_px.min(1.8)
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

#[allow(dead_code)]
fn incremental_map(
    frames: &[ImageFrame],
    camera: CameraModel,
    reference_camera_setup: Option<&ReferenceCameraSetup>,
    pairs: &[PairGeometry],
    config: &MapperConfig,
) -> Result<(Reconstruction, Vec<String>)> {
    let mut session = IncrementalMapperSession::default();
    incremental_map_with_session(
        frames,
        camera,
        reference_camera_setup,
        pairs,
        config,
        &mut session,
    )
}

#[derive(Debug, Clone)]
struct PipelineSnapshotState {
    previous_registered_frames: usize,
    next_index: usize,
}

impl PipelineSnapshotState {
    fn new(reconstruction: &Reconstruction) -> Self {
        Self {
            previous_registered_frames: registered_frame_count(reconstruction),
            next_index: 0,
        }
    }
}

fn maybe_write_pipeline_snapshot(
    reconstruction: &Reconstruction,
    config: &MapperConfig,
    snapshot_state: &mut PipelineSnapshotState,
    debug_log: &mut Vec<String>,
    events: &mut MapperEventBridge<'_, '_>,
) -> Result<()> {
    if config.snapshot_frames_freq == 0 {
        return Ok(());
    }
    let Some(snapshot_path) = config.snapshot_path.as_ref() else {
        return Ok(());
    };
    let registered_frames = registered_frame_count(reconstruction);
    if registered_frames < snapshot_state.previous_registered_frames + config.snapshot_frames_freq {
        return Ok(());
    }
    snapshot_state.previous_registered_frames = registered_frames;
    snapshot_state.next_index += 1;
    let path = snapshot_path.join(format!("{:010}", snapshot_state.next_index));
    events.checkpoint()?;
    events.emit_operation(
        SfmTaskStage::Export,
        SfmTaskOperation::WriteArtifacts,
        SfmTaskEventKind::Started,
    );
    export_colmap_sparse_snapshot(&path, reconstruction)?;
    events.emit_operation(
        SfmTaskStage::Export,
        SfmTaskOperation::WriteArtifacts,
        SfmTaskEventKind::Completed,
    );
    events.checkpoint()?;
    debug_log.push(format!(
        "pipeline_snapshot path={} registered_frames={registered_frames}",
        path.display()
    ));
    Ok(())
}

fn incremental_pipeline_map(
    frames: &[ImageFrame],
    camera: CameraModel,
    reference_camera_setup: Option<&ReferenceCameraSetup>,
    pairs: &[PairGeometry],
    config: &MapperConfig,
) -> Result<IncrementalPipelineMapResult> {
    let mut events = MapperEventBridge::Silent;
    incremental_pipeline_map_with_pnp_scorer_and_events(
        frames,
        camera,
        reference_camera_setup,
        pairs,
        config,
        &mut events,
        None,
    )
}

fn incremental_pipeline_map_with_pnp_scorer_and_events(
    frames: &[ImageFrame],
    camera: CameraModel,
    reference_camera_setup: Option<&ReferenceCameraSetup>,
    pairs: &[PairGeometry],
    config: &MapperConfig,
    events: &mut MapperEventBridge<'_, '_>,
    pnp_scorer: Option<&mut DynPnPModelScorer>,
) -> Result<IncrementalPipelineMapResult> {
    let mut pnp_scorer = pnp_scorer;
    let mut session = IncrementalMapperSession::default();
    let mut reconstructions = Vec::new();
    let mut debug_log = Vec::new();
    let max_num_models = if config.multiple_models {
        config.max_num_models.max(1)
    } else {
        1
    };
    let min_model_size = config.min_model_size.min(frames.len() / 2);
    let mut last_error = None;

    'stages: for stage_config in initialization_stage_configs(config) {
        if stage_config.stage != InitializationRelaxationStage::Strict {
            session.reset_initialization_stats();
        }
        for trial in 0..stage_config.config.init_num_trials.max(1) {
            events.checkpoint()?;
            if reconstructions.len() >= max_num_models
                || (config.multiple_models
                    && session.num_total_registered_images() >= frames.len().saturating_sub(1))
            {
                break 'stages;
            }

            let attempt_uses_seed =
                stage_config.stage == InitializationRelaxationStage::Strict && trial == 0;
            let attempt_setup =
                setup_for_reconstruction_attempt(reference_camera_setup, attempt_uses_seed);
            let model_index = reconstructions.len();
            match incremental_map_single_attempt_with_pnp_scorer(
                frames,
                camera,
                attempt_setup.as_ref(),
                pairs,
                &stage_config.config,
                &mut session,
                model_index,
                events,
                &mut pnp_scorer,
            ) {
                Ok((reconstruction, mut attempt_log)) => {
                    let registered_images = registered_image_count(&reconstruction);
                    let registered_frames = registered_frame_count(&reconstruction);
                    let points = reconstruction.points.len();
                    let keep = registered_images > 0
                        && (!config.multiple_models
                            || reconstructions.is_empty()
                            || registered_images >= min_model_size);
                    debug_log.push(format!(
                        "initialization_attempt stage={} trial={} init_min_num_inliers={} init_min_tri_angle_deg={:.6}",
                        initialization_stage_name(stage_config.stage),
                        trial,
                        stage_config.config.init_min_num_inliers,
                        stage_config.config.init_min_tri_angle_deg
                    ));
                    debug_log.push(format!(
                        "pipeline_submodel index={model_index} status={} registered_images={} points={} total_registered_images={} shared_registered_images={}",
                        if keep { "kept" } else { "discarded_insufficient_size" },
                        registered_images,
                        points,
                        session.num_total_registered_images(),
                        session.num_shared_registered_image_events()
                    ));
                    debug_log.append(&mut attempt_log);

                    if keep {
                        session.end_reconstruction(&reconstruction, false);
                        reconstructions.push(reconstruction);
                    } else {
                        session.end_reconstruction(&reconstruction, true);
                    }
                    push_pipeline_callback(
                        &mut debug_log,
                        events,
                        PipelineCallbackEvent {
                            callback: IncrementalPipelineCallback::LastImageReg,
                            model_index,
                            registered_images,
                            registered_frames,
                            points,
                        },
                    );
                    events.checkpoint()?;

                    if !config.multiple_models
                        || session.num_shared_registered_image_events() >= config.max_model_overlap
                    {
                        break 'stages;
                    }
                }
                Err(err) => {
                    if err.downcast_ref::<GpuPnpMapperError>().is_some()
                        || err.downcast_ref::<SfmTaskStop>().is_some()
                        || {
                            #[cfg(feature = "gpu-wgpu")]
                            {
                                err.downcast_ref::<GpuPnPFocalFallbackError>().is_some()
                            }
                            #[cfg(not(feature = "gpu-wgpu"))]
                            {
                                false
                            }
                        }
                    {
                        return Err(err);
                    }
                    events.checkpoint()?;
                    let message = err.to_string();
                    let initial_failure = err.downcast_ref::<InitialPairFailure>().copied();
                    debug_log.push(format!(
                        "initialization_attempt_failed stage={} trial={} error={message}",
                        initialization_stage_name(stage_config.stage),
                        trial,
                    ));
                    let no_initial_pair =
                        initial_failure == Some(InitialPairFailure::NoInitialPair);
                    last_error = Some(err);
                    if no_initial_pair {
                        break;
                    }
                }
            }
        }
    }

    if reconstructions.is_empty() {
        return Err(last_error
            .unwrap_or_else(|| anyhow::anyhow!("no reconstruction models kept"))
            .context("no reconstruction models kept"));
    }
    Ok(IncrementalPipelineMapResult {
        reconstructions,
        debug_log,
    })
}

pub fn run_incremental_pipeline(
    frames: &[ImageFrame],
    camera: CameraModel,
    reference_camera_setup: Option<&ReferenceCameraSetup>,
    pairs: &[PairGeometry],
    config: &MapperConfig,
    callback_sink: Option<&mut dyn PipelineCallbackSink>,
) -> IncrementalPipelineResult {
    let mut events = match callback_sink {
        Some(sink) => MapperEventBridge::Legacy(sink),
        None => MapperEventBridge::Silent,
    };
    match incremental_pipeline_map_with_pnp_scorer_and_events(
        frames,
        camera,
        reference_camera_setup,
        pairs,
        config,
        &mut events,
        None,
    ) {
        Ok(result) => IncrementalPipelineResult {
            status: IncrementalPipelineStatus::Success,
            reconstructions: result.reconstructions,
            debug_log: result.debug_log,
        },
        Err(err) => {
            let mut status = IncrementalPipelineStatus::NoModelsKept;
            let mut current: &(dyn std::error::Error + 'static) = err.as_ref();
            loop {
                if let Some(failure) = current.downcast_ref::<InitialPairFailure>() {
                    status = match failure {
                        InitialPairFailure::NoInitialPair => {
                            IncrementalPipelineStatus::NoInitialPair
                        }
                        InitialPairFailure::BadInitialPair => {
                            IncrementalPipelineStatus::BadInitialPair
                        }
                    };
                    break;
                }
                match current.source() {
                    Some(source) => current = source,
                    None => break,
                }
            }
            IncrementalPipelineResult {
                status,
                reconstructions: Vec::new(),
                debug_log: vec![err.to_string()],
            }
        }
    }
}

fn triangulate_registration_unit(
    triangulator: &mut IncrementalTriangulator<'_>,
    tri_options: &IncrementalTriangulatorOptions,
    unit_images: &[usize],
) {
    for &frame_image in unit_images {
        triangulator.triangulate_image(tri_options, frame_image);
    }
}

fn apply_probe_cameras_to_registration_unit(
    reconstruction: &mut Reconstruction,
    probe: &Reconstruction,
    image: usize,
) {
    for frame_image in reconstruction.image_indices_for_registration_unit(image) {
        let Some(camera_idx) = reconstruction
            .image_camera_indices
            .get(frame_image)
            .copied()
        else {
            continue;
        };
        let Some(camera) = probe.cameras.get(camera_idx).copied() else {
            continue;
        };
        if let Some(slot) = reconstruction.cameras.get_mut(camera_idx) {
            *slot = camera;
        }
        if camera_idx == 0 {
            reconstruction.camera = camera;
        }
    }
}

fn register_and_triangulate_initial_image_pair(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    triangulation_state: &mut IncrementalTriangulatorState,
    tri_options: &IncrementalTriangulatorOptions,
    init_min_tri_angle_deg: f32,
    initial: &PairGeometry,
) {
    let mut initial_tri_options = *tri_options;
    initial_tri_options.min_angle_deg = initial_pair_create_min_angle_deg(init_min_tri_angle_deg);
    triangulation_state
        .observation_manager_mut()
        .register_image(frames, pairs, reconstruction, initial.left, SE3::identity());
    triangulation_state
        .observation_manager_mut()
        .register_image(
            frames,
            pairs,
            reconstruction,
            initial.right,
            initial.relative_pose,
        );
    let left_unit = reconstruction.image_indices_for_registration_unit(initial.left);
    let right_unit = reconstruction.image_indices_for_registration_unit(initial.right);
    let mut triangulator =
        IncrementalTriangulator::new(frames, pairs, reconstruction, triangulation_state);
    triangulate_registration_unit(&mut triangulator, &initial_tri_options, &left_unit);
    triangulate_registration_unit(&mut triangulator, &initial_tri_options, &right_unit);
}

fn initial_pair_create_min_angle_deg(init_min_tri_angle_deg: f32) -> f32 {
    const INITIAL_CREATE_MIN_ANGLE_COMPENSATION_DEG: f32 = 0.325;
    init_min_tri_angle_deg + INITIAL_CREATE_MIN_ANGLE_COMPENSATION_DEG
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InitializationRelaxationStage {
    Strict,
    MinInliersRelaxed(usize),
    MinTriAngleRelaxed(usize),
}

#[derive(Debug, Clone)]
struct InitializationStageConfig {
    stage: InitializationRelaxationStage,
    config: MapperConfig,
}

fn initialization_stage_configs(config: &MapperConfig) -> Vec<InitializationStageConfig> {
    let mut stages = Vec::new();
    let mut relaxed_config = config.clone();
    stages.push(InitializationStageConfig {
        stage: InitializationRelaxationStage::Strict,
        config: relaxed_config.clone(),
    });
    for relaxation in 0..2 {
        relaxed_config.init_min_num_inliers = (relaxed_config.init_min_num_inliers / 2).max(1);
        stages.push(InitializationStageConfig {
            stage: InitializationRelaxationStage::MinInliersRelaxed(relaxation + 1),
            config: relaxed_config.clone(),
        });
        relaxed_config.init_min_tri_angle_deg *= 0.5;
        stages.push(InitializationStageConfig {
            stage: InitializationRelaxationStage::MinTriAngleRelaxed(relaxation + 1),
            config: relaxed_config.clone(),
        });
    }
    stages
}

fn initialization_stage_name(stage: InitializationRelaxationStage) -> &'static str {
    match stage {
        InitializationRelaxationStage::Strict => "strict",
        InitializationRelaxationStage::MinInliersRelaxed(_) => "relaxed_min_inliers",
        InitializationRelaxationStage::MinTriAngleRelaxed(_) => "relaxed_min_tri_angle",
    }
}

fn push_pipeline_callback(
    debug_log: &mut Vec<String>,
    events: &mut MapperEventBridge<'_, '_>,
    event: PipelineCallbackEvent,
) {
    debug_log.push(format!("callback {}", event.callback.as_str()));
    events.callback(event);
}

fn incremental_map_with_session(
    frames: &[ImageFrame],
    camera: CameraModel,
    reference_camera_setup: Option<&ReferenceCameraSetup>,
    pairs: &[PairGeometry],
    config: &MapperConfig,
    session: &mut IncrementalMapperSession,
) -> Result<(Reconstruction, Vec<String>)> {
    let mut debug_log = Vec::new();
    let mut last_error = None;
    let mut events = MapperEventBridge::Silent;
    for stage_config in initialization_stage_configs(config) {
        if stage_config.stage != InitializationRelaxationStage::Strict {
            session.reset_initialization_stats();
        }
        for trial in 0..stage_config.config.init_num_trials.max(1) {
            let attempt_uses_seed =
                stage_config.stage == InitializationRelaxationStage::Strict && trial == 0;
            let attempt_setup =
                setup_for_reconstruction_attempt(reference_camera_setup, attempt_uses_seed);
            match incremental_map_single_attempt(
                frames,
                camera,
                attempt_setup.as_ref(),
                pairs,
                &stage_config.config,
                session,
                0,
                &mut events,
            ) {
                Ok((reconstruction, mut attempt_log)) => {
                    debug_log.push(format!(
                        "initialization_attempt stage={} trial={} init_min_num_inliers={} init_min_tri_angle_deg={:.6}",
                        initialization_stage_name(stage_config.stage),
                        trial,
                        stage_config.config.init_min_num_inliers,
                        stage_config.config.init_min_tri_angle_deg
                    ));
                    debug_log.append(&mut attempt_log);
                    return Ok((reconstruction, debug_log));
                }
                Err(err) => {
                    let message = err.to_string();
                    let initial_failure = err.downcast_ref::<InitialPairFailure>().copied();
                    debug_log.push(format!(
                        "initialization_attempt_failed stage={} trial={} error={message}",
                        initialization_stage_name(stage_config.stage),
                        trial,
                    ));
                    let no_initial_pair =
                        initial_failure == Some(InitialPairFailure::NoInitialPair);
                    last_error = Some(err);
                    if no_initial_pair {
                        break;
                    }
                }
            }
        }
    }
    Err(last_error
        .unwrap_or_else(|| anyhow::anyhow!("no initialization attempts configured"))
        .context("no initial pair after initialization trials and relaxations"))
}

fn incremental_map_single_attempt(
    frames: &[ImageFrame],
    camera: CameraModel,
    reference_camera_setup: Option<&ReferenceCameraSetup>,
    pairs: &[PairGeometry],
    config: &MapperConfig,
    session: &mut IncrementalMapperSession,
    model_index: usize,
    events: &mut MapperEventBridge<'_, '_>,
) -> Result<(Reconstruction, Vec<String>)> {
    let mut pnp_scorer = None;
    incremental_map_single_attempt_with_pnp_scorer(
        frames,
        camera,
        reference_camera_setup,
        pairs,
        config,
        session,
        model_index,
        events,
        &mut pnp_scorer,
    )
}

#[allow(clippy::too_many_arguments)]
fn incremental_map_single_attempt_with_pnp_scorer(
    frames: &[ImageFrame],
    camera: CameraModel,
    reference_camera_setup: Option<&ReferenceCameraSetup>,
    pairs: &[PairGeometry],
    config: &MapperConfig,
    session: &mut IncrementalMapperSession,
    model_index: usize,
    events: &mut MapperEventBridge<'_, '_>,
    pnp_scorer: &mut Option<&mut DynPnPModelScorer>,
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
        seed_reconstruction,
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
            setup.seed_reconstruction.clone(),
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
            None,
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
    if let Some(seed) = seed_reconstruction {
        apply_reconstruction_seed(&mut reconstruction, seed, frames);
    }
    session.begin_reconstruction(&reconstruction);
    let mapping_pairs = pairs
        .iter()
        .filter(|pair| !pair.pose_graph_only)
        .cloned()
        .collect::<Vec<_>>();
    let pairs = mapping_pairs.as_slice();
    let mut registration_stats = RegistrationStats::from_reconstruction(&reconstruction);
    if config.fix_existing_frames {
        registration_stats.set_existing_registration_units_from_reconstruction(&reconstruction);
    }
    let tri_options = mapper_triangulator_options(config);
    let mut triangulation_state = IncrementalTriangulatorState::new(frames, pairs, &reconstruction);
    let mut initial_color_images = Vec::new();
    events.checkpoint()?;
    let gauge_image = if let Some(image) = reconstruction.poses.iter().position(Option::is_some) {
        debug_log.push(format!(
            "continue_reconstruction registered_images={} points={}",
            registered_image_count(&reconstruction),
            reconstruction.points.len()
        ));
        image
    } else {
        let mut initial_pair_state = session.initial_pair_selection_state(&reconstruction);
        let initial = choose_initial_pair(
            pairs,
            &reconstruction,
            config,
            &camera_has_prior_focal_length,
            &mut initial_pair_state,
        )
        .ok_or(InitialPairFailure::NoInitialPair)?;
        initial_pair_state.register_initial_pair(&reconstruction, initial.left, initial.right);
        session.commit_initial_pair_selection_state(&reconstruction, &initial_pair_state);
        debug_log.push(format!(
            "initial_pair {} -> {} inliers={} triangulated={}",
            frames[initial.left].name,
            frames[initial.right].name,
            initial.inliers,
            initial.triangulated
        ));
        {
            register_and_triangulate_initial_image_pair(
                frames,
                pairs,
                &mut reconstruction,
                &mut triangulation_state,
                &tri_options,
                config.init_min_tri_angle_deg,
                &initial,
            );
        }
        registration_stats = RegistrationStats::from_reconstruction(&reconstruction);
        registration_stats.set_initial_pair_selection_state(initial_pair_state);
        initial_color_images = vec![initial.left, initial.right];
        initial.left
    };
    let mut global_ba_schedule = GlobalBaSchedule::new(&reconstruction);
    let mut filtered_units = HashSet::<RegistrationUnitKey>::new();
    let initial_pair_registered = !initial_color_images.is_empty();
    let initial_global_ba_ran = if initial_pair_registered {
        refine_initial_global_bundle(
            frames,
            pairs,
            &mut reconstruction,
            config,
            &mut debug_log,
            Some(&mut registration_stats),
            Some(&mut filtered_units),
            &mut triangulation_state,
        )
    } else {
        refine_global_bundle_with_postprocessing(
            frames,
            pairs,
            &mut reconstruction,
            &tri_options,
            config,
            "initial",
            incremental_global_ba_normalizes_reconstruction(config),
            &mut debug_log,
            Some(&mut registration_stats),
            Some(&mut filtered_units),
            &mut triangulation_state,
        )
    };
    if initial_global_ba_ran {
        global_ba_schedule.mark(&reconstruction);
    }
    if initial_pair_registered {
        if !initial_global_ba_ran {
            filter_reprojection_tracks_with_state(
                frames,
                pairs,
                &mut reconstruction,
                config,
                &mut triangulation_state,
            );
        }
        reject_bad_initial_pair_after_registration(&reconstruction, config)?;
    }
    for image in initial_color_images.iter().copied() {
        let color_report =
            extract_colors_for_registration_unit(frames, &mut reconstruction, image, config);
        if color_report.images > 0 {
            debug_log.push(format!(
                "extract_colors image={} images={} updated_points={}",
                frames[image].name, color_report.images, color_report.updated_points
            ));
        }
    }
    if initial_pair_registered {
        push_pipeline_callback(
            &mut debug_log,
            events,
            PipelineCallbackEvent {
                callback: IncrementalPipelineCallback::InitialImagePairReg,
                model_index,
                registered_images: registered_image_count(&reconstruction),
                registered_frames: registered_frame_count(&reconstruction),
                points: reconstruction.points.len(),
            },
        );
    }
    events.checkpoint()?;

    let mut snapshot_state = PipelineSnapshotState::new(&reconstruction);
    let mut retry_state = RegistrationRetryState::new(frames.len());
    let mut telemetry = IncrementalRegistrationTelemetry::default();
    let mut fallback_available = true;
    while reconstruction.poses.iter().any(|p| p.is_none()) {
        events.checkpoint()?;
        let NextRegistrationSelection {
            choice,
            failed_attempts,
        } = choose_next_registration_with_failures_and_pnp_scorer(
            frames,
            pairs,
            &reconstruction,
            &retry_state,
            RegistrationPass::Normal,
            &filtered_units,
            config,
            &camera_priors,
            &camera_has_prior_focal_length,
            &registration_stats,
            triangulation_state.observation_manager(),
            &mut telemetry,
            pnp_scorer,
        )?;
        events.checkpoint()?;
        let normal_attempted_candidates = !failed_attempts.is_empty();
        for (failed_image, mode) in failed_attempts {
            let support = registration_unit_support(
                &reconstruction,
                failed_image,
                triangulation_state.observation_manager(),
                mode,
            );
            retry_state.record_failure(&reconstruction, failed_image, mode, support);
            debug_log.push(format!(
                "registration_attempt_failed {} mode={:?}",
                frames[failed_image].name, mode
            ));
        }
        let choice = if let Some(choice) = choice {
            choice
        } else if normal_attempted_candidates {
            continue;
        } else {
            if !fallback_available {
                break;
            }
            fallback_available = false;
            telemetry.fallback_epochs += 1;
            let NextRegistrationSelection {
                choice,
                failed_attempts,
            } = choose_next_registration_with_failures_and_pnp_scorer(
                frames,
                pairs,
                &reconstruction,
                &retry_state,
                RegistrationPass::ExhaustiveFallback,
                &filtered_units,
                config,
                &camera_priors,
                &camera_has_prior_focal_length,
                &registration_stats,
                triangulation_state.observation_manager(),
                &mut telemetry,
                pnp_scorer,
            )?;
            events.checkpoint()?;
            for (failed_image, mode) in failed_attempts {
                let support = registration_unit_support(
                    &reconstruction,
                    failed_image,
                    triangulation_state.observation_manager(),
                    mode,
                );
                retry_state.record_failure(&reconstruction, failed_image, mode, support);
                debug_log.push(format!(
                    "registration_attempt_failed {} mode={:?} fallback=true",
                    frames[failed_image].name, mode
                ));
            }
            let Some(choice) = choice else {
                break;
            };
            choice
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
        let registration_probe = reconstruction_with_reset_frame_cameras(
            &reconstruction,
            choice.image,
            config,
            &camera_priors,
        );
        apply_probe_cameras_to_registration_unit(
            &mut reconstruction,
            &registration_probe,
            choice.image,
        );
        reset_bogus_frame_cameras_from_priors(
            &mut reconstruction,
            choice.image,
            config,
            &camera_priors,
        );
        let observation_update_start = Instant::now();
        {
            triangulation_state
                .observation_manager_mut()
                .register_image(
                    frames,
                    pairs,
                    &mut reconstruction,
                    choice.image,
                    choice.pose,
                );
            for &(frame_image, frame_pose) in &choice.frame_image_poses {
                if let Some(slot) = reconstruction.poses.get_mut(frame_image) {
                    *slot = Some(frame_pose);
                }
            }
            if let Some(frame_idx) = reconstruction.frame_index_for_image(choice.image) {
                if let Some((rig_from_world, image_poses)) =
                    frame_consistent_poses_from_registered_images(&reconstruction, frame_idx)
                {
                    if let Some(frame) = reconstruction.frames.get_mut(frame_idx) {
                        frame.rig_from_world = Rigid3::from_se3(rig_from_world);
                    }
                    for (frame_image, frame_pose) in image_poses {
                        if let Some(slot) = reconstruction.poses.get_mut(frame_image) {
                            *slot = Some(frame_pose);
                        }
                    }
                }
            }
            for inlier in &choice.generalized_inliers {
                triangulation_state
                    .observation_manager_mut()
                    .add_observation(
                        frames,
                        pairs,
                        &mut reconstruction,
                        inlier.point_id,
                        TrackObservation {
                            image: inlier.image,
                            feature: inlier.feature,
                        },
                    );
            }
        }
        telemetry.observation_update_ms +=
            observation_update_start.elapsed().as_secs_f64() * 1000.0;
        let triangulation_start = Instant::now();
        let structureless_track_report = if !choice.structureless_inliers.is_empty() {
            continue_or_triangulate_structureless_tracks(
                frames,
                pairs,
                &mut reconstruction,
                &choice.structureless_inliers,
                &tri_options,
                config,
                triangulation_state.observation_manager_mut(),
            )
        } else {
            TriangulationReport::default()
        };
        {
            let unit_images = reconstruction.image_indices_for_registration_unit(choice.image);
            let mut triangulator = IncrementalTriangulator::new(
                frames,
                pairs,
                &mut reconstruction,
                &mut triangulation_state,
            );
            triangulate_registration_unit(&mut triangulator, &tri_options, &unit_images);
        }
        {
            let mut triangulator = IncrementalTriangulator::new(
                frames,
                pairs,
                &mut reconstruction,
                &mut triangulation_state,
            );
            let modified = triangulator.get_modified_points3d().clone();
            triangulator.complete_tracks(&tri_options, &modified);
            let modified = triangulator.get_modified_points3d().clone();
            triangulator.merge_tracks(&tri_options, &modified);
            triangulator.retriangulate(&tri_options);
        }
        telemetry.triangulation_ms += triangulation_start.elapsed().as_secs_f64() * 1000.0;
        filter_modified_reprojection_tracks_with_state(
            frames,
            pairs,
            &mut reconstruction,
            config,
            &mut triangulation_state,
        );
        let mut local_registration_stats = registration_stats.clone();
        local_registration_stats.register_frame_for_image_event(&reconstruction, choice.image);
        let local_ba_required =
            local_bundle_refinement_required(&reconstruction, choice.image, gauge_image, config);
        events.checkpoint()?;
        events.emit_operation(
            SfmTaskStage::BundleAdjustment,
            SfmTaskOperation::LocalBundleAdjustment,
            SfmTaskEventKind::Started,
        );
        let local_ba_report = refine_local_bundle_after_registration(
            frames,
            pairs,
            &mut reconstruction,
            choice.image,
            gauge_image,
            &tri_options,
            config,
            &local_registration_stats,
            &mut triangulation_state,
        );
        events.emit_operation(
            SfmTaskStage::BundleAdjustment,
            SfmTaskOperation::LocalBundleAdjustment,
            SfmTaskEventKind::Completed,
        );
        events.checkpoint()?;
        let rollback_reason = registration_rollback_reason(
            &reconstruction,
            choice.image,
            local_ba_required,
            local_ba_report.is_some(),
            config,
        );
        if let Some(reason) = rollback_reason {
            reconstruction = registration_snapshot;
            triangulation_state.sync_after_reconstruction_rollback(frames, pairs, &reconstruction);
            let mode = registration_mode_for_choice(&choice);
            let support = registration_unit_support(
                &reconstruction,
                choice.image,
                triangulation_state.observation_manager(),
                mode,
            );
            retry_state.record_failure(&reconstruction, choice.image, mode, support);
            debug_log.push(format!(
                "registration_rollback {} reason={reason}",
                frames[choice.image].name
            ));
            events.checkpoint()?;
            continue;
        }
        registration_stats.register_frame_for_image_event(&reconstruction, choice.image);
        retry_state.record_success(
            &reconstruction,
            choice.image,
            registration_mode_for_choice(&choice),
        );
        fallback_available = true;
        filtered_units.remove(&registration_unit_key(&reconstruction, choice.image));
        let filtered_frames = filter_registered_frames(
            frames,
            pairs,
            &mut reconstruction,
            config,
            &mut registration_stats,
            Some(&mut filtered_units),
            &mut triangulation_state,
        );
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
                "local_ba image={} refinements={} local_images={} variable_images={} points={} observations={} residuals={} cost={:.6}->{:.6} iterations={}/{} termination={:?} reason={:?} merged={} completed={} image_completed={} filtered={} changed={:.6} solver={:?} preconditioner={:?} sparse_backend={:?} setup_ms={:.2} solve_ms={:.2} postprocess_ms={:.2} ba_elapsed_ms={:.2}",
                frames[choice.image].name,
                report.refinements,
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
                report.completed_image_observations,
                report.filtered_observations,
                report.changed_observation_ratio,
                report.report.linear_solver,
                report.report.preconditioner,
                report.report.sparse_backend,
                report.report.setup_ms,
                report.report.solve_ms,
                report.report.postprocess_ms,
                report.report.elapsed_ms
            ));
        }
        if filtered_frames > 0 {
            debug_log.push(format!("filtered_frames count={filtered_frames}"));
        }
        if should_run_global_ba(&global_ba_schedule, &reconstruction, config) {
            events.checkpoint()?;
            events.emit_operation(
                SfmTaskStage::BundleAdjustment,
                SfmTaskOperation::GlobalBundleAdjustment,
                SfmTaskEventKind::Started,
            );
            let global_ba_ran = refine_global_bundle_with_postprocessing(
                frames,
                pairs,
                &mut reconstruction,
                &tri_options,
                config,
                "scheduled",
                incremental_global_ba_normalizes_reconstruction(config),
                &mut debug_log,
                Some(&mut registration_stats),
                Some(&mut filtered_units),
                &mut triangulation_state,
            );
            events.emit_operation(
                SfmTaskStage::BundleAdjustment,
                SfmTaskOperation::GlobalBundleAdjustment,
                SfmTaskEventKind::Completed,
            );
            events.checkpoint()?;
            if global_ba_ran {
                global_ba_schedule.mark(&reconstruction);
            }
        }
        let color_report =
            extract_colors_for_registration_unit(frames, &mut reconstruction, choice.image, config);
        if color_report.images > 0 {
            debug_log.push(format!(
                "extract_colors image={} images={} updated_points={}",
                frames[choice.image].name, color_report.images, color_report.updated_points
            ));
        }
        maybe_write_pipeline_snapshot(
            &reconstruction,
            config,
            &mut snapshot_state,
            &mut debug_log,
            events,
        )?;
        push_pipeline_callback(
            &mut debug_log,
            events,
            PipelineCallbackEvent {
                callback: IncrementalPipelineCallback::NextImageReg,
                model_index,
                registered_images: registered_image_count(&reconstruction),
                registered_frames: registered_frame_count(&reconstruction),
                points: reconstruction.points.len(),
            },
        );
        events.checkpoint()?;
    }
    if should_run_final_global_ba(&global_ba_schedule, &reconstruction, config) {
        events.checkpoint()?;
        events.emit_operation(
            SfmTaskStage::BundleAdjustment,
            SfmTaskOperation::GlobalBundleAdjustment,
            SfmTaskEventKind::Started,
        );
        refine_global_bundle_with_postprocessing(
            frames,
            pairs,
            &mut reconstruction,
            &tri_options,
            config,
            "final",
            final_global_ba_normalizes_reconstruction(config),
            &mut debug_log,
            Some(&mut registration_stats),
            Some(&mut filtered_units),
            &mut triangulation_state,
        );
        events.emit_operation(
            SfmTaskStage::BundleAdjustment,
            SfmTaskOperation::GlobalBundleAdjustment,
            SfmTaskEventKind::Completed,
        );
        events.checkpoint()?;
    }
    let final_color_report =
        extract_colors_for_all_registered_images(frames, &mut reconstruction, config);
    if final_color_report.images > 0 {
        debug_log.push(format!(
            "extract_colors_all images={} updated_points={}",
            final_color_report.images, final_color_report.updated_points
        ));
    }
    sync_registered_frame_poses_from_images(&mut reconstruction);
    debug_log.push(
        triangulation_state
            .observation_manager()
            .sparse_maintenance_log(),
    );
    debug_log.push(telemetry.format_log());
    Ok((reconstruction, debug_log))
}

fn apply_reconstruction_seed(
    reconstruction: &mut Reconstruction,
    seed: ReconstructionSeed,
    frames: &[ImageFrame],
) {
    for (image, pose) in seed.poses.into_iter().enumerate() {
        if let Some(slot) = reconstruction.poses.get_mut(image) {
            *slot = pose;
        }
    }
    for (image, seed_observations) in seed.observations.into_iter().enumerate() {
        let Some(observations) = reconstruction.observations.get_mut(image) else {
            continue;
        };
        let len = frames
            .get(image)
            .map(|frame| frame.keypoints.len())
            .unwrap_or(observations.len());
        observations.clear();
        observations.resize(len, None);
        for (feature, point_id) in seed_observations.into_iter().enumerate().take(len) {
            *observations.get_mut(feature).expect("feature in range") = point_id;
        }
    }
    reconstruction.point_ids = seed.point_ids;
    reconstruction.points = seed.points;
    for point in &mut reconstruction.points {
        point.track.retain(|obs| {
            frames
                .get(obs.image)
                .is_some_and(|frame| obs.feature < frame.keypoints.len())
        });
    }
    for (point_idx, point) in reconstruction.points.iter().enumerate() {
        for obs in &point.track {
            if let Some(slot) = reconstruction
                .observations
                .get_mut(obs.image)
                .and_then(|image_observations| image_observations.get_mut(obs.feature))
            {
                *slot = Some(point_idx);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ColorExtractionReport {
    images: usize,
    updated_points: usize,
}

fn extract_colors_for_registration_unit(
    frames: &[ImageFrame],
    reconstruction: &mut Reconstruction,
    image: usize,
    config: &MapperConfig,
) -> ColorExtractionReport {
    if !config.extract_colors {
        return ColorExtractionReport::default();
    }
    let mut report = ColorExtractionReport::default();
    for frame_image in reconstruction.image_indices_for_registration_unit(image) {
        report.images += 1;
        report.updated_points += extract_colors_for_image(frames, reconstruction, frame_image);
    }
    report
}

fn extract_colors_for_image(
    frames: &[ImageFrame],
    reconstruction: &mut Reconstruction,
    image: usize,
) -> usize {
    let Some(frame) = frames.get(image) else {
        return 0;
    };
    let Some(observations) = reconstruction.observations.get(image) else {
        return 0;
    };
    let colors = colors_for_frame(frame);
    let mut updates = Vec::new();
    for (feature, point_id) in observations.iter().copied().enumerate() {
        let Some(point_id) = point_id else {
            continue;
        };
        let Some(point) = reconstruction.points.get(point_id) else {
            continue;
        };
        if point.color != [0, 0, 0] {
            continue;
        }
        let Some(color) = colors.get(feature).copied() else {
            continue;
        };
        updates.push((point_id, color));
    }

    let mut updated = 0usize;
    for (point_id, color) in updates {
        let Some(point) = reconstruction.points.get_mut(point_id) else {
            continue;
        };
        if point.color == [0, 0, 0] {
            point.color = color;
            updated += 1;
        }
    }
    updated
}

fn colors_for_frame(frame: &ImageFrame) -> Cow<'_, [[u8; 3]]> {
    if frame.colors.len() == frame.keypoints.len() {
        Cow::Borrowed(frame.colors.as_slice())
    } else {
        Cow::Owned(sample_keypoint_colors(frame))
    }
}

fn extract_colors_for_all_registered_images(
    frames: &[ImageFrame],
    reconstruction: &mut Reconstruction,
    config: &MapperConfig,
) -> ColorExtractionReport {
    if !config.extract_colors {
        return ColorExtractionReport::default();
    }
    let images = reconstruction
        .poses
        .iter()
        .filter(|pose| pose.is_some())
        .count();
    let colors_by_image = frames
        .iter()
        .enumerate()
        .map(|(image, frame)| {
            reconstruction
                .poses
                .get(image)
                .is_some_and(|pose| pose.is_some())
                .then(|| colors_for_frame(frame))
        })
        .collect::<Vec<_>>();
    let mut updated_points = 0usize;
    for point in &mut reconstruction.points {
        let mut sum = [0usize; 3];
        let mut count = 0usize;
        for obs in &point.track {
            if reconstruction
                .poses
                .get(obs.image)
                .copied()
                .flatten()
                .is_none()
            {
                continue;
            }
            let Some(color) = colors_by_image
                .get(obs.image)
                .and_then(Option::as_ref)
                .and_then(|colors| colors.get(obs.feature))
                .copied()
            else {
                continue;
            };
            sum[0] += color[0] as usize;
            sum[1] += color[1] as usize;
            sum[2] += color[2] as usize;
            count += 1;
        }
        let color = if count == 0 {
            [0, 0, 0]
        } else {
            [
                ((sum[0] as f32 / count as f32).round() as u8),
                ((sum[1] as f32 / count as f32).round() as u8),
                ((sum[2] as f32 / count as f32).round() as u8),
            ]
        };
        if point.color != color {
            point.color = color;
            updated_points += 1;
        }
    }
    ColorExtractionReport {
        images,
        updated_points,
    }
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
    frame_image_poses: Vec<(usize, SE3)>,
    generalized_inliers: Vec<GeneralizedFrameInlier>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NextImageRegistrationMode {
    StructureBased,
    StructureLess,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegistrationPass {
    Normal,
    ExhaustiveFallback,
}

#[derive(Debug, Clone, Copy, Default)]
struct RegistrationModeAttempt {
    trials: usize,
    last_support: usize,
}

#[derive(Debug, Clone)]
struct RegistrationRetryState {
    structure_based: Vec<RegistrationModeAttempt>,
    structureless: Vec<RegistrationModeAttempt>,
}

impl RegistrationRetryState {
    fn new(num_images: usize) -> Self {
        Self {
            structure_based: vec![RegistrationModeAttempt::default(); num_images],
            structureless: vec![RegistrationModeAttempt::default(); num_images],
        }
    }

    #[cfg(test)]
    fn from_trial_vectors(
        structure_based_trials: &[usize],
        structureless_trials: &[usize],
    ) -> Self {
        let num_images = structure_based_trials.len().max(structureless_trials.len());
        let mut state = Self::new(num_images);
        for (attempt, &trials) in state.structure_based.iter_mut().zip(structure_based_trials) {
            attempt.trials = trials;
            if trials > 0 {
                attempt.last_support = usize::MAX;
            }
        }
        for (attempt, &trials) in state.structureless.iter_mut().zip(structureless_trials) {
            attempt.trials = trials;
            if trials > 0 {
                attempt.last_support = usize::MAX;
            }
        }
        state
    }

    fn attempts(&self, mode: NextImageRegistrationMode) -> &[RegistrationModeAttempt] {
        match mode {
            NextImageRegistrationMode::StructureBased => &self.structure_based,
            NextImageRegistrationMode::StructureLess => &self.structureless,
        }
    }

    fn attempts_mut(&mut self, mode: NextImageRegistrationMode) -> &mut [RegistrationModeAttempt] {
        match mode {
            NextImageRegistrationMode::StructureBased => &mut self.structure_based,
            NextImageRegistrationMode::StructureLess => &mut self.structureless,
        }
    }

    fn unit_attempt(
        &self,
        reconstruction: &Reconstruction,
        image: usize,
        mode: NextImageRegistrationMode,
    ) -> RegistrationModeAttempt {
        let attempts = self.attempts(mode);
        reconstruction
            .image_indices_for_registration_unit(image)
            .into_iter()
            .filter_map(|frame_image| attempts.get(frame_image).copied())
            .max_by_key(|attempt| attempt.trials)
            .unwrap_or_else(|| attempts.get(image).copied().unwrap_or_default())
    }

    fn num_trials(
        &self,
        reconstruction: &Reconstruction,
        image: usize,
        mode: NextImageRegistrationMode,
        support: usize,
    ) -> usize {
        let attempt = self.unit_attempt(reconstruction, image, mode);
        if support > attempt.last_support {
            0
        } else {
            attempt.trials
        }
    }

    fn is_eligible(
        &self,
        reconstruction: &Reconstruction,
        image: usize,
        mode: NextImageRegistrationMode,
        support: usize,
        max_trials: usize,
        pass: RegistrationPass,
    ) -> bool {
        pass == RegistrationPass::ExhaustiveFallback
            || self.num_trials(reconstruction, image, mode, support) < max_trials
    }

    fn record_failure(
        &mut self,
        reconstruction: &Reconstruction,
        image: usize,
        mode: NextImageRegistrationMode,
        support: usize,
    ) {
        let trials = self
            .num_trials(reconstruction, image, mode, support)
            .saturating_add(1);
        let attempt = RegistrationModeAttempt {
            trials,
            last_support: support,
        };
        let attempts = self.attempts_mut(mode);
        for frame_image in reconstruction.image_indices_for_registration_unit(image) {
            if let Some(slot) = attempts.get_mut(frame_image) {
                *slot = attempt;
            }
        }
    }

    fn record_success(
        &mut self,
        reconstruction: &Reconstruction,
        image: usize,
        mode: NextImageRegistrationMode,
    ) {
        let attempts = self.attempts_mut(mode);
        for frame_image in reconstruction.image_indices_for_registration_unit(image) {
            if let Some(slot) = attempts.get_mut(frame_image) {
                *slot = RegistrationModeAttempt::default();
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
struct IncrementalRegistrationTelemetry {
    candidate_units: usize,
    skipped_unchanged: usize,
    structure_based_attempts: usize,
    structureless_attempts: usize,
    structureless_estimates: usize,
    structureless_accepted: usize,
    structureless_solver_ms: f64,
    fallback_epochs: usize,
    collect_observations_ms: f64,
    pose_solve_refine_ms: f64,
    observation_update_ms: f64,
    triangulation_ms: f64,
    gpu_pnp_focal_fallbacks: Vec<String>,
}

impl IncrementalRegistrationTelemetry {
    #[cfg(feature = "gpu-wgpu")]
    fn record_gpu_pnp_focal_fallback(&mut self, reason: impl std::fmt::Display) {
        let reason = reason.to_string();
        if !self.gpu_pnp_focal_fallbacks.contains(&reason) {
            self.gpu_pnp_focal_fallbacks.push(reason);
        }
    }

    fn format_log(&self) -> String {
        format!(
            "incremental_registration candidate_units={} skipped_unchanged={} structure_based_attempts={} structureless_attempts={} structureless_estimates={} structureless_accepted={} structureless_solver_ms={:.2} fallback_epochs={} collect_observations_ms={:.2} pose_solve_refine_ms={:.2} observation_update_ms={:.2} triangulation_ms={:.2} gpu_pnp_focal_fallback={}",
            self.candidate_units,
            self.skipped_unchanged,
            self.structure_based_attempts,
            self.structureless_attempts,
            self.structureless_estimates,
            self.structureless_accepted,
            self.structureless_solver_ms,
            self.fallback_epochs,
            self.collect_observations_ms,
            self.pose_solve_refine_ms,
            self.observation_update_ms,
            self.triangulation_ms,
            self.gpu_pnp_focal_fallbacks.join("; "),
        )
    }
}

#[derive(Debug, Clone)]
struct NextRegistrationSelection {
    choice: Option<RegistrationChoice>,
    failed_attempts: Vec<(usize, NextImageRegistrationMode)>,
}

#[cfg(test)]
fn choose_next_registration(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    reg_trials: &[usize],
    structureless_reg_trials: &[usize],
    filtered_units: &HashSet<RegistrationUnitKey>,
    config: &MapperConfig,
    camera_priors: &[CameraModel],
    camera_has_prior_focal_length: &[bool],
    registration_stats: &RegistrationStats,
) -> Option<RegistrationChoice> {
    let obs_manager = ObservationManager::new(frames, pairs, reconstruction);
    choose_next_registration_with_failures(
        frames,
        pairs,
        reconstruction,
        reg_trials,
        structureless_reg_trials,
        filtered_units,
        config,
        camera_priors,
        camera_has_prior_focal_length,
        registration_stats,
        &obs_manager,
    )
    .choice
}

#[cfg(test)]
fn choose_next_registration_with_failures(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    reg_trials: &[usize],
    structureless_reg_trials: &[usize],
    filtered_units: &HashSet<RegistrationUnitKey>,
    config: &MapperConfig,
    camera_priors: &[CameraModel],
    camera_has_prior_focal_length: &[bool],
    registration_stats: &RegistrationStats,
    obs_manager: &ObservationManager,
) -> NextRegistrationSelection {
    let mut pnp_scorer = None;
    let retry_state =
        RegistrationRetryState::from_trial_vectors(reg_trials, structureless_reg_trials);
    let mut telemetry = IncrementalRegistrationTelemetry::default();
    choose_next_registration_with_failures_and_pnp_scorer(
        frames,
        pairs,
        reconstruction,
        &retry_state,
        RegistrationPass::Normal,
        filtered_units,
        config,
        camera_priors,
        camera_has_prior_focal_length,
        registration_stats,
        obs_manager,
        &mut telemetry,
        &mut pnp_scorer,
    )
    .expect("CPU absolute pose routes are infallible")
}

#[allow(clippy::too_many_arguments)]
fn choose_next_registration_with_failures_and_pnp_scorer(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    retry_state: &RegistrationRetryState,
    pass: RegistrationPass,
    filtered_units: &HashSet<RegistrationUnitKey>,
    config: &MapperConfig,
    camera_priors: &[CameraModel],
    camera_has_prior_focal_length: &[bool],
    registration_stats: &RegistrationStats,
    obs_manager: &ObservationManager,
    telemetry: &mut IncrementalRegistrationTelemetry,
    pnp_scorer: &mut Option<&mut DynPnPModelScorer>,
) -> Result<NextRegistrationSelection> {
    let correspondence_graph = obs_manager.correspondence_graph();
    let mut failed_attempts = Vec::new();
    for mode in next_registration_modes(config) {
        let next_images = find_next_registration_images_with_retry_state(
            reconstruction,
            retry_state,
            filtered_units,
            config,
            obs_manager,
            mode,
            pass,
            telemetry,
        );
        telemetry.candidate_units += next_images.len();
        for image in next_images {
            match mode {
                NextImageRegistrationMode::StructureBased => {
                    telemetry.structure_based_attempts += 1;
                }
                NextImageRegistrationMode::StructureLess => {
                    telemetry.structureless_attempts += 1;
                }
            }
            if let Some(choice) = registration_choice_for_image_with_pnp_scorer(
                image,
                frames,
                pairs,
                reconstruction,
                config,
                camera_priors,
                camera_has_prior_focal_length,
                registration_stats,
                obs_manager,
                correspondence_graph,
                mode,
                telemetry,
                pnp_scorer,
            )? {
                return Ok(NextRegistrationSelection {
                    choice: Some(choice),
                    failed_attempts,
                });
            } else {
                failed_attempts.push((image, mode));
            }
        }
    }
    Ok(NextRegistrationSelection {
        choice: None,
        failed_attempts,
    })
}

fn next_registration_modes(config: &MapperConfig) -> Vec<NextImageRegistrationMode> {
    let mut modes = vec![NextImageRegistrationMode::StructureBased];
    if structureless_registration_enabled(config) {
        modes.push(NextImageRegistrationMode::StructureLess);
    }
    modes
}

fn structureless_registration_enabled(_config: &MapperConfig) -> bool {
    // COLMAP always runs a structure-less registration bucket after structure-based.
    true
}

fn registration_unit_num_visible_points3d(
    reconstruction: &Reconstruction,
    image: usize,
    obs_manager: &ObservationManager,
) -> usize {
    reconstruction
        .image_indices_for_registration_unit(image)
        .iter()
        .map(|&frame_image| obs_manager.num_visible_points3d(frame_image))
        .sum()
}

fn registration_unit_num_visible_correspondences(
    reconstruction: &Reconstruction,
    image: usize,
    obs_manager: &ObservationManager,
) -> usize {
    reconstruction
        .image_indices_for_registration_unit(image)
        .iter()
        .map(|&frame_image| obs_manager.num_visible_correspondences(frame_image))
        .sum()
}

fn registration_unit_support(
    reconstruction: &Reconstruction,
    image: usize,
    obs_manager: &ObservationManager,
    mode: NextImageRegistrationMode,
) -> usize {
    match mode {
        NextImageRegistrationMode::StructureBased => {
            registration_unit_num_visible_points3d(reconstruction, image, obs_manager)
        }
        NextImageRegistrationMode::StructureLess => {
            registration_unit_num_visible_correspondences(reconstruction, image, obs_manager)
        }
    }
}

#[cfg(test)]
fn find_next_registration_images(
    reconstruction: &Reconstruction,
    reg_trials: &[usize],
    structureless_reg_trials: &[usize],
    filtered_units: &HashSet<RegistrationUnitKey>,
    config: &MapperConfig,
    obs_manager: &ObservationManager,
    mode: NextImageRegistrationMode,
) -> Vec<usize> {
    let retry_state =
        RegistrationRetryState::from_trial_vectors(reg_trials, structureless_reg_trials);
    let mut telemetry = IncrementalRegistrationTelemetry::default();
    find_next_registration_images_with_retry_state(
        reconstruction,
        &retry_state,
        filtered_units,
        config,
        obs_manager,
        mode,
        RegistrationPass::Normal,
        &mut telemetry,
    )
}

fn find_next_registration_images_with_retry_state(
    reconstruction: &Reconstruction,
    retry_state: &RegistrationRetryState,
    filtered_units: &HashSet<RegistrationUnitKey>,
    config: &MapperConfig,
    obs_manager: &ObservationManager,
    mode: NextImageRegistrationMode,
    pass: RegistrationPass,
    telemetry: &mut IncrementalRegistrationTelemetry,
) -> Vec<usize> {
    let mut image_ranks = Vec::<(usize, f32)>::new();
    let mut other_image_ranks = Vec::<(usize, f32)>::new();
    let mut seen_units = HashSet::new();
    for image in 0..reconstruction.poses.len() {
        let unit_key = registration_unit_key(reconstruction, image);
        if !seen_units.insert(unit_key) {
            continue;
        }
        if registration_unit_is_registered(reconstruction, image) {
            continue;
        }
        let support = registration_unit_support(reconstruction, image, obs_manager, mode);
        let min_support = match mode {
            NextImageRegistrationMode::StructureBased => config.abs_pose_min_num_inliers,
            NextImageRegistrationMode::StructureLess => structureless_min_num_inliers(config),
        };
        if support < min_support {
            continue;
        }
        if !retry_state.is_eligible(
            reconstruction,
            image,
            mode,
            support,
            config.max_reg_trials,
            pass,
        ) {
            telemetry.skipped_unchanged += 1;
            continue;
        }
        let num_trials = retry_state.num_trials(reconstruction, image, mode, support);
        let rank = match mode {
            NextImageRegistrationMode::StructureBased => {
                next_image_rank(reconstruction, image, obs_manager, config)
            }
            NextImageRegistrationMode::StructureLess => support as f32,
        };
        if filtered_units.contains(&registration_unit_key(reconstruction, image)) || num_trials > 0
        {
            other_image_ranks.push((image, rank));
        } else {
            image_ranks.push((image, rank));
        }
    }

    let mut ranked_images = Vec::new();
    sort_and_append_next_images(image_ranks, &mut ranked_images);
    sort_and_append_next_images(other_image_ranks, &mut ranked_images);
    ranked_images
}

fn sort_and_append_next_images(mut image_ranks: Vec<(usize, f32)>, ranked_images: &mut Vec<usize>) {
    image_ranks.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked_images.reserve(image_ranks.len());
    ranked_images.extend(image_ranks.into_iter().map(|(image, _)| image));
}

#[allow(clippy::too_many_arguments)]
fn registration_choice_for_image_with_pnp_scorer(
    image: usize,
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
    camera_priors: &[CameraModel],
    camera_has_prior_focal_length: &[bool],
    registration_stats: &RegistrationStats,
    obs_manager: &ObservationManager,
    correspondence_graph: Option<&CorrespondenceGraph>,
    mode: NextImageRegistrationMode,
    telemetry: &mut IncrementalRegistrationTelemetry,
    pnp_scorer: &mut Option<&mut DynPnPModelScorer>,
) -> Result<Option<RegistrationChoice>> {
    let registration_reconstruction =
        reconstruction_with_reset_frame_cameras(reconstruction, image, config, camera_priors);
    let structureless_estimates_before = telemetry.structureless_estimates;
    let (abs_pose, source) = match mode {
        NextImageRegistrationMode::StructureBased => {
            if generalized_frame_registration_applicable(
                image,
                &registration_reconstruction,
                config,
                obs_manager,
                camera_has_prior_focal_length,
                registration_stats,
            ) {
                record_gpu_pnp_route_fallback(
                    config,
                    telemetry,
                    "generalized rig absolute pose is solved on the CPU",
                );
                let Some(abs_pose) = solve_generalized_frame_absolute_pose(
                    image,
                    frames,
                    pairs,
                    &registration_reconstruction,
                    config,
                    obs_manager,
                    camera_has_prior_focal_length,
                    registration_stats,
                    correspondence_graph,
                ) else {
                    return Ok(None);
                };
                (abs_pose, "generalized_frame")
            } else if let Some(abs_pose) = solve_absolute_pose_with_pnp_scorer(
                image,
                frames,
                pairs,
                &registration_reconstruction,
                config,
                camera_priors,
                camera_has_prior_focal_length,
                registration_stats,
                correspondence_graph,
                pnp_scorer.as_deref_mut(),
                telemetry,
            )? {
                (abs_pose, "pnp")
            } else {
                return Ok(None);
            }
        }
        NextImageRegistrationMode::StructureLess => {
            record_gpu_pnp_route_fallback(
                config,
                telemetry,
                "structureless absolute pose is solved on the CPU",
            );
            let Some(abs_pose) = solve_structureless_absolute_pose(
                image,
                frames,
                pairs,
                &registration_reconstruction,
                config,
                obs_manager,
                camera_priors,
                registration_stats,
                correspondence_graph,
                telemetry,
            ) else {
                return Ok(None);
            };
            (abs_pose, "structureless")
        }
    };
    let pair_rot_error =
        registered_pair_rotation_error(image, abs_pose.pose, pairs, reconstruction);
    if mode == NextImageRegistrationMode::StructureLess
        && (!pair_rot_error.is_finite() || pair_rot_error > absolute_pose_pair_rotation_limit_deg())
    {
        return Ok(None);
    }
    if mode == NextImageRegistrationMode::StructureLess
        && telemetry.structureless_estimates > structureless_estimates_before
    {
        telemetry.structureless_accepted += 1;
    }
    let visible_points = obs_manager.num_visible_points3d(image);
    let num_observations = obs_manager.num_observations(image).max(1);
    let visible_points_ratio = visible_points as f32 / num_observations as f32;
    Ok(Some(RegistrationChoice {
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
        frame_image_poses: abs_pose.frame_image_poses,
        generalized_inliers: abs_pose.generalized_inliers,
    }))
}

fn next_image_rank(
    reconstruction: &Reconstruction,
    image: usize,
    obs_manager: &ObservationManager,
    config: &MapperConfig,
) -> f32 {
    let unit_images = reconstruction.image_indices_for_registration_unit(image);
    match config.image_selection_method {
        ImageSelectionMethod::MaxVisiblePointsNum => unit_images
            .iter()
            .map(|&frame_image| obs_manager.num_visible_points3d(frame_image))
            .sum::<usize>() as f32,
        ImageSelectionMethod::MaxVisiblePointsRatio => {
            let visible = unit_images
                .iter()
                .map(|&frame_image| obs_manager.num_visible_points3d(frame_image))
                .sum::<usize>() as f32;
            let observations = unit_images
                .iter()
                .map(|&frame_image| obs_manager.num_observations(frame_image))
                .sum::<usize>()
                .max(1) as f32;
            visible / observations
        }
        ImageSelectionMethod::MinUncertainty => unit_images
            .iter()
            .map(|&frame_image| obs_manager.point3d_visibility_score(frame_image))
            .max()
            .unwrap_or(0) as f32,
    }
}

fn registration_mode_for_choice(choice: &RegistrationChoice) -> NextImageRegistrationMode {
    if choice.source == "structureless" {
        NextImageRegistrationMode::StructureLess
    } else {
        NextImageRegistrationMode::StructureBased
    }
}

#[cfg(test)]
fn registration_rank(
    choice: &RegistrationChoice,
    reconstruction: &Reconstruction,
    obs_manager: &ObservationManager,
    config: &MapperConfig,
) -> f32 {
    if choice.source == "structureless" {
        return registration_unit_num_visible_correspondences(
            reconstruction,
            choice.image,
            obs_manager,
        ) as f32;
    }
    next_image_rank(reconstruction, choice.image, obs_manager, config)
}

#[cfg(test)]
fn mark_unregistered_images_with_no_absolute_pose(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    reg_trials: &mut [usize],
    config: &MapperConfig,
    camera_priors: &[CameraModel],
    camera_has_prior_focal_length: &[bool],
    registration_stats: &RegistrationStats,
    obs_manager: &ObservationManager,
) {
    let mut pnp_scorer = None;
    mark_unregistered_images_with_no_absolute_pose_and_pnp_scorer(
        frames,
        pairs,
        reconstruction,
        reg_trials,
        config,
        camera_priors,
        camera_has_prior_focal_length,
        registration_stats,
        obs_manager,
        &mut pnp_scorer,
    )
    .expect("CPU absolute pose routes are infallible");
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn mark_unregistered_images_with_no_absolute_pose_and_pnp_scorer(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    reg_trials: &mut [usize],
    config: &MapperConfig,
    camera_priors: &[CameraModel],
    camera_has_prior_focal_length: &[bool],
    registration_stats: &RegistrationStats,
    obs_manager: &ObservationManager,
    pnp_scorer: &mut Option<&mut DynPnPModelScorer>,
) -> Result<()> {
    let mut marked_units = HashSet::new();
    let mut telemetry = IncrementalRegistrationTelemetry::default();
    for image in 0..reconstruction.poses.len() {
        if registration_unit_is_registered(reconstruction, image)
            || registration_unit_num_trials(reconstruction, image, reg_trials)
                >= config.max_reg_trials
        {
            continue;
        }
        let unit = registration_unit_key(reconstruction, image);
        if !marked_units.insert(unit) {
            continue;
        }
        let structure_based_pose = if generalized_frame_registration_applicable(
            image,
            reconstruction,
            config,
            obs_manager,
            camera_has_prior_focal_length,
            registration_stats,
        ) {
            record_gpu_pnp_route_fallback(
                config,
                &mut telemetry,
                "generalized rig absolute pose is solved on the CPU",
            );
            solve_generalized_frame_absolute_pose(
                image,
                frames,
                pairs,
                reconstruction,
                config,
                obs_manager,
                camera_has_prior_focal_length,
                registration_stats,
                obs_manager.correspondence_graph(),
            )
        } else {
            solve_absolute_pose_with_pnp_scorer(
                image,
                frames,
                pairs,
                reconstruction,
                config,
                camera_priors,
                camera_has_prior_focal_length,
                registration_stats,
                obs_manager.correspondence_graph(),
                pnp_scorer.as_deref_mut(),
                &mut telemetry,
            )?
        };
        let pose = if structure_based_pose.is_some() {
            structure_based_pose
        } else {
            record_gpu_pnp_route_fallback(
                config,
                &mut telemetry,
                "structureless absolute pose is solved on the CPU",
            );
            solve_structureless_absolute_pose(
                image,
                frames,
                pairs,
                reconstruction,
                config,
                obs_manager,
                camera_priors,
                registration_stats,
                obs_manager.correspondence_graph(),
                &mut telemetry,
            )
        };
        let has_pose = pose
            .map(|pose| {
                let pair_rot_error =
                    registered_pair_rotation_error(image, pose.pose, pairs, reconstruction);
                pair_rot_error.is_finite()
                    && pair_rot_error <= absolute_pose_pair_rotation_limit_deg()
            })
            .unwrap_or(false);
        if !has_pose {
            increment_registration_unit_trials(reconstruction, image, reg_trials);
        }
    }
    Ok(())
}

#[cfg(test)]
fn mark_unregistered_images_with_no_absolute_pose_for_test(
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
    mark_unregistered_images_with_no_absolute_pose(
        frames,
        pairs,
        reconstruction,
        reg_trials,
        config,
        camera_priors,
        camera_has_prior_focal_length,
        registration_stats,
        &obs_manager,
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RegistrationUnitKey {
    Image(usize),
    Frame(usize),
}

fn registration_unit_key(reconstruction: &Reconstruction, image: usize) -> RegistrationUnitKey {
    reconstruction
        .frame_index_for_image(image)
        .map(RegistrationUnitKey::Frame)
        .unwrap_or(RegistrationUnitKey::Image(image))
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

#[cfg(test)]
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

#[cfg(test)]
fn increment_registration_unit_trials(
    reconstruction: &Reconstruction,
    image: usize,
    reg_trials: &mut [usize],
) {
    for frame_image in reconstruction.image_indices_for_registration_unit(image) {
        if let Some(trials) = reg_trials.get_mut(frame_image) {
            *trials += 1;
        }
    }
}

#[cfg(test)]
fn registration_unit_num_trials(
    reconstruction: &Reconstruction,
    image: usize,
    reg_trials: &[usize],
) -> usize {
    reconstruction
        .image_indices_for_registration_unit(image)
        .into_iter()
        .filter_map(|frame_image| reg_trials.get(frame_image).copied())
        .max()
        .unwrap_or_else(|| reg_trials.get(image).copied().unwrap_or(0))
}

#[cfg(test)]
fn registration_unit_num_trials_for_mode(
    reconstruction: &Reconstruction,
    image: usize,
    reg_trials: &[usize],
    structureless_reg_trials: &[usize],
    mode: NextImageRegistrationMode,
) -> usize {
    match mode {
        NextImageRegistrationMode::StructureBased => {
            registration_unit_num_trials(reconstruction, image, reg_trials)
        }
        NextImageRegistrationMode::StructureLess => {
            registration_unit_num_trials(reconstruction, image, structureless_reg_trials)
        }
    }
}

#[cfg(test)]
fn increment_registration_unit_trials_for_mode(
    reconstruction: &Reconstruction,
    image: usize,
    mode: NextImageRegistrationMode,
    reg_trials: &mut [usize],
    structureless_reg_trials: &mut [usize],
) {
    match mode {
        NextImageRegistrationMode::StructureBased => {
            increment_registration_unit_trials(reconstruction, image, reg_trials)
        }
        NextImageRegistrationMode::StructureLess => {
            increment_registration_unit_trials(reconstruction, image, structureless_reg_trials)
        }
    }
}

/// COLMAP retries unregistered images as the reconstruction grows. Reset trial
/// counters after each successful registration so tail frames are not excluded
/// because they were probed too early with insufficient 2D-3D visibility.
#[cfg(test)]
fn reset_unregistered_registration_trials(
    reconstruction: &Reconstruction,
    reg_trials: &mut [usize],
    structureless_reg_trials: &mut [usize],
) {
    for image in 0..reconstruction.poses.len() {
        if registration_unit_is_registered(reconstruction, image) {
            continue;
        }
        reset_registration_unit_trials(reconstruction, image, reg_trials);
        reset_registration_unit_trials(reconstruction, image, structureless_reg_trials);
    }
}

#[derive(Debug, Clone)]
struct LocalBundleReport {
    report: crate::ba::BundleAdjustmentReport,
    refinements: usize,
    variable_images: usize,
    local_images: usize,
    points: usize,
    merged_observations: usize,
    completed_observations: usize,
    completed_image_observations: usize,
    filtered_observations: usize,
    changed_observation_ratio: f32,
}

#[derive(Debug, Clone, Copy)]
struct GlobalBaSchedule {
    prev_registered_images: usize,
    prev_registered_frames: usize,
    prev_points: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ReconstructionNormalizationTransform {
    scale: f32,
    translation: glam::Vec3,
}

#[derive(Debug, Clone, Copy)]
struct RedundantPoint3DInfo {
    point_id: usize,
    stable_point_id: u64,
    gain: f64,
}

impl PartialEq for RedundantPoint3DInfo {
    fn eq(&self, other: &Self) -> bool {
        self.point_id == other.point_id
            && self.stable_point_id == other.stable_point_id
            && self.gain.total_cmp(&other.gain).is_eq()
    }
}

impl Eq for RedundantPoint3DInfo {}

impl PartialOrd for RedundantPoint3DInfo {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for RedundantPoint3DInfo {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        self.gain
            .total_cmp(&other.gain)
            .then_with(|| self.stable_point_id.cmp(&other.stable_point_id))
    }
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

fn refine_initial_global_bundle(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    config: &MapperConfig,
    debug_log: &mut Vec<String>,
    mut registration_stats: Option<&mut RegistrationStats>,
    mut filtered_units: Option<&mut HashSet<RegistrationUnitKey>>,
    triangulation_state: &mut IncrementalTriangulatorState,
) -> bool {
    if !global_ba_enabled(config)
        || registered_image_count(reconstruction) < 2
        || reconstruction.points.is_empty()
    {
        return false;
    }

    let observations_before = reconstruction_num_observations(reconstruction);
    if observations_before == 0 {
        return false;
    }
    let gauge_images = global_ba_gauge_images(reconstruction);
    if gauge_images.is_empty() {
        return false;
    }
    let mut ba_options = mapper_global_ba_options(
        config,
        reconstruction,
        global_ba_iterations_for_reconstruction(config, reconstruction),
        None,
        if config.fix_existing_frames {
            registration_stats
                .as_deref()
                .map(|stats| stats.existing_registered_images(reconstruction))
                .unwrap_or_default()
        } else {
            Vec::new()
        },
        None,
        None,
    );
    let uses_prior_position = global_ba_uses_prior_position(&ba_options, reconstruction);
    ba_options.gauge = if uses_prior_position {
        crate::ba::BundleAdjustmentGauge::None
    } else {
        crate::ba::BundleAdjustmentGauge::TwoCamsFromWorld
    };
    if uses_prior_position {
        let Some(transform) = align_reconstruction_to_pose_priors(
            reconstruction,
            &ba_options.pose_priors,
            config.random_seed,
        ) else {
            debug_log.push(format!(
                "global_ba reason=initial round=1 skipped skip_reason=prior_alignment_failed gauge_images={:?} observations={}",
                gauge_images, observations_before
            ));
            return false;
        };
        debug_log.push(format!(
            "global_ba_align_priors reason=initial round=1 scale={:.6} rotation=({:.4},{:.4},{:.4},{:.4}) translation=({:.6},{:.6},{:.6})",
            transform.scale,
            glam::Quat::from_mat3(&transform.rotation).x,
            glam::Quat::from_mat3(&transform.rotation).y,
            glam::Quat::from_mat3(&transform.rotation).z,
            glam::Quat::from_mat3(&transform.rotation).w,
            transform.translation.x,
            transform.translation.y,
            transform.translation.z
        ));
    }
    let report = match refine_bundle_adjustment_checked(frames, reconstruction, config, ba_options)
    {
        Ok(report) => report,
        Err(skip_reason) => {
            debug_log.push(format!(
                "global_ba reason=initial round=1 skipped skip_reason={} gauge_images={:?} observations={}",
                skip_reason, gauge_images, observations_before
            ));
            return false;
        }
    };
    let normalization =
        if incremental_global_ba_normalizes_reconstruction(config) && !uses_prior_position {
            normalize_reconstruction_colmap(reconstruction, false, 10.0, 0.1, 0.9, true)
        } else {
            None
        };
    let filtered = filter_reprojection_tracks_with_state(
        frames,
        pairs,
        reconstruction,
        config,
        triangulation_state,
    );
    let filtered_frames = if let Some(stats) = registration_stats.as_deref_mut() {
        filter_registered_frames(
            frames,
            pairs,
            reconstruction,
            config,
            stats,
            filtered_units.as_deref_mut(),
            triangulation_state,
        )
    } else {
        0
    };
    let changed = filtered as f32 / observations_before.max(1) as f32;
    debug_log.push(format!(
        "global_ba reason=initial round=1 size={} gauge_images={:?} observations={} residuals={} cost={:.6}->{:.6} iterations={}/{} termination={:?} termination_reason={:?} completed=0 merged=0 filtered={} filtered_frames={} changed={:.6} solver={:?} preconditioner={:?} sparse_backend={:?} setup_ms={:.2} solve_ms={:.2} postprocess_ms={:.2} ba_elapsed_ms={:.2}",
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
        filtered,
        filtered_frames,
        changed,
        report.linear_solver,
        report.preconditioner,
        report.sparse_backend,
        report.setup_ms,
        report.solve_ms,
        report.postprocess_ms,
        report.elapsed_ms
    ));
    if let Some(transform) = normalization {
        debug_log.push(format!(
            "global_ba_normalize reason=initial round=1 scale={:.6} translation=({:.6},{:.6},{:.6})",
            transform.scale,
            transform.translation.x,
            transform.translation.y,
            transform.translation.z
        ));
    }
    true
}

fn refine_global_bundle_with_postprocessing(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    tri_options: &IncrementalTriangulatorOptions,
    config: &MapperConfig,
    reason: &str,
    normalize_reconstruction: bool,
    debug_log: &mut Vec<String>,
    mut registration_stats: Option<&mut RegistrationStats>,
    mut filtered_units: Option<&mut HashSet<RegistrationUnitKey>>,
    triangulation_state: &mut IncrementalTriangulatorState,
) -> bool {
    if !global_ba_enabled(config)
        || registered_image_count(reconstruction) < 2
        || reconstruction.points.is_empty()
    {
        return false;
    }

    let (pre_completed, pre_merged, retriangulated) = {
        let mut triangulator =
            IncrementalTriangulator::new(frames, pairs, reconstruction, triangulation_state);
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
    for round in 0..global_ba_max_refinements_for_reason(config, reason) {
        let observations_before = reconstruction_num_observations(reconstruction);
        if observations_before == 0 {
            break;
        }
        let gauge_images = global_ba_gauge_images(reconstruction);
        if gauge_images.is_empty() {
            break;
        }
        attempted = true;
        let redundant_point_ids = global_ba_redundant_point_ids(config, reconstruction);
        if let Some(redundant_point_ids) = redundant_point_ids.as_ref() {
            debug_log.push(format!(
                "global_ba_redundant_points reason={reason} round={} ignored={} points={}",
                round + 1,
                redundant_point_ids.len(),
                reconstruction.points.len()
            ));
        }
        let non_redundant_point_ids = redundant_point_ids.as_ref().map(|redundant_point_ids| {
            non_redundant_point_ids(reconstruction.points.len(), redundant_point_ids)
        });
        let mut ba_options = mapper_global_ba_options(
            config,
            reconstruction,
            global_ba_iterations_for_reason(config, reconstruction, reason),
            None,
            if config.fix_existing_frames {
                registration_stats
                    .as_deref()
                    .map(|stats| stats.existing_registered_images(reconstruction))
                    .unwrap_or_default()
            } else {
                Vec::new()
            },
            non_redundant_point_ids,
            None,
        );
        let uses_prior_position = global_ba_uses_prior_position(&ba_options, reconstruction);
        ba_options.gauge = if uses_prior_position {
            crate::ba::BundleAdjustmentGauge::None
        } else {
            crate::ba::BundleAdjustmentGauge::TwoCamsFromWorld
        };
        if uses_prior_position && round == 0 {
            let Some(transform) = align_reconstruction_to_pose_priors(
                reconstruction,
                &ba_options.pose_priors,
                config.random_seed,
            ) else {
                debug_log.push(format!(
                    "global_ba reason={reason} round=1 skipped skip_reason=prior_alignment_failed gauge_images={:?} observations={}",
                    gauge_images, observations_before
                ));
                break;
            };
            debug_log.push(format!(
                "global_ba_align_priors reason={reason} round=1 scale={:.6} rotation=({:.4},{:.4},{:.4},{:.4}) translation=({:.6},{:.6},{:.6})",
                transform.scale,
                glam::Quat::from_mat3(&transform.rotation).x,
                glam::Quat::from_mat3(&transform.rotation).y,
                glam::Quat::from_mat3(&transform.rotation).z,
                glam::Quat::from_mat3(&transform.rotation).w,
                transform.translation.x,
                transform.translation.y,
                transform.translation.z
            ));
        }
        let report = match refine_bundle_adjustment_checked(
            frames,
            reconstruction,
            config,
            ba_options,
        ) {
            Ok(report) => report,
            Err(skip_reason) => {
                debug_log.push(format!(
                    "global_ba reason={reason} round={} skipped skip_reason={} gauge_images={:?} observations={}",
                    round + 1,
                    skip_reason,
                    gauge_images,
                    observations_before
                ));
                break;
            }
        };
        if let Some(redundant_point_ids) = redundant_point_ids {
            let redundant_ba_options =
                redundant_point_global_ba_options(config, reconstruction, redundant_point_ids);
            let redundant_report = match refine_bundle_adjustment_checked(
                frames,
                reconstruction,
                config,
                redundant_ba_options,
            ) {
                Ok(report) => report,
                Err(skip_reason) => {
                    debug_log.push(format!(
                        "global_ba_redundant_points reason={reason} round={} skipped skip_reason={}",
                        round + 1,
                        skip_reason
                    ));
                    break;
                }
            };
            debug_log.push(format!(
                "global_ba_redundant_points reason={reason} round={} residuals={} cost={:.6}->{:.6} iterations={}/{} termination={:?} termination_reason={:?} solver={:?} preconditioner={:?} sparse_backend={:?} setup_ms={:.2} solve_ms={:.2} postprocess_ms={:.2} ba_elapsed_ms={:.2}",
                round + 1,
                redundant_report.residuals,
                redundant_report.initial_cost,
                redundant_report.final_cost,
                redundant_report.iterations,
                redundant_report.attempted_iterations,
                redundant_report.termination_type,
                redundant_report.termination_reason,
                redundant_report.linear_solver,
                redundant_report.preconditioner,
                redundant_report.sparse_backend,
                redundant_report.setup_ms,
                redundant_report.solve_ms,
                redundant_report.postprocess_ms,
                redundant_report.elapsed_ms
            ));
        }
        let normalization = if normalize_reconstruction && !uses_prior_position {
            normalize_reconstruction_colmap(reconstruction, false, 10.0, 0.1, 0.9, true)
        } else {
            None
        };

        let (completed, merged) = {
            let mut triangulator =
                IncrementalTriangulator::new(frames, pairs, reconstruction, triangulation_state);
            let completed = triangulator.complete_all_tracks(tri_options);
            let merged = triangulator.merge_all_tracks(tri_options);
            (completed, merged)
        };
        let filtered = filter_reprojection_tracks_with_state(
            frames,
            pairs,
            reconstruction,
            config,
            triangulation_state,
        );
        let filtered_frames = if let Some(stats) = registration_stats.as_deref_mut() {
            filter_registered_frames(
                frames,
                pairs,
                reconstruction,
                config,
                stats,
                filtered_units.as_deref_mut(),
                triangulation_state,
            )
        } else {
            0
        };
        let changed = (completed + merged + filtered) as f32 / observations_before.max(1) as f32;
        debug_log.push(format!(
            "global_ba reason={reason} round={} size={} gauge_images={:?} observations={} residuals={} cost={:.6}->{:.6} iterations={}/{} termination={:?} termination_reason={:?} completed={} merged={} filtered={} filtered_frames={} changed={:.6} solver={:?} preconditioner={:?} sparse_backend={:?} setup_ms={:.2} solve_ms={:.2} postprocess_ms={:.2} ba_elapsed_ms={:.2}",
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
            filtered_frames,
            changed,
            report.linear_solver,
            report.preconditioner,
            report.sparse_backend,
            report.setup_ms,
            report.solve_ms,
            report.postprocess_ms,
            report.elapsed_ms
        ));
        if let Some(transform) = normalization {
            debug_log.push(format!(
                "global_ba_normalize reason={reason} round={} scale={:.6} translation=({:.6},{:.6},{:.6})",
                round + 1,
                transform.scale,
                transform.translation.x,
                transform.translation.y,
                transform.translation.z
            ));
        }
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
    // COLMAP's CheckRunGlobalRefinement applies the ratio thresholds with no
    // minimum-growth gate: the first frame past the ratio threshold triggers.
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

// Scheduled BA recurs on model-growth thresholds, so repeated full refinements
// have diminishing value before the next trigger. Keep the full configured
// budget for initial/final quality closure and cap only intermediate passes.
fn global_ba_max_refinements_for_reason(config: &MapperConfig, reason: &str) -> usize {
    const MAX_SCHEDULED_REFINEMENTS: usize = 2;
    if reason == "scheduled" {
        config
            .global_ba_max_refinements
            .min(MAX_SCHEDULED_REFINEMENTS)
    } else {
        config.global_ba_max_refinements
    }
}

fn global_ba_iterations_for_reason(
    config: &MapperConfig,
    reconstruction: &Reconstruction,
    reason: &str,
) -> usize {
    const MAX_SCHEDULED_ITERATIONS: usize = 20;
    let configured = global_ba_iterations_for_reconstruction(config, reconstruction);
    if reason == "scheduled" {
        configured.min(MAX_SCHEDULED_ITERATIONS)
    } else {
        configured
    }
}

fn incremental_global_ba_normalizes_reconstruction(_config: &MapperConfig) -> bool {
    // COLMAP disables this only for prior-position BA and final-all handoff paths.
    true
}

fn final_global_ba_normalizes_reconstruction(_config: &MapperConfig) -> bool {
    false
}

fn global_ba_uses_prior_position(
    options: &crate::ba::BundleAdjustmentOptions,
    _reconstruction: &Reconstruction,
) -> bool {
    // COLMAP's AdjustGlobalBundle switches to the pose-prior adjuster when at
    // least three registered images carry valid pose priors.
    options.pose_priors.len() >= 3
}

fn global_ba_redundant_point_ids(
    config: &MapperConfig,
    reconstruction: &Reconstruction,
) -> Option<Vec<usize>> {
    const MIN_NUM_REG_FRAMES_FOR_FAST_BA: usize = 10;
    if !config.global_ba_ignore_redundant_points3d
        || registered_frame_count(reconstruction) < MIN_NUM_REG_FRAMES_FOR_FAST_BA
    {
        return None;
    }
    Some(find_redundant_points3d_colmap(
        config.global_ba_ignore_redundant_points3d_min_coverage_gain,
        reconstruction,
    ))
}

fn non_redundant_point_ids(num_points: usize, redundant_point_ids: &[usize]) -> Vec<usize> {
    let redundant = redundant_point_ids.iter().copied().collect::<HashSet<_>>();
    (0..num_points)
        .filter(|point_id| !redundant.contains(point_id))
        .collect()
}

fn redundant_point_global_ba_options(
    config: &MapperConfig,
    reconstruction: &Reconstruction,
    redundant_point_ids: Vec<usize>,
) -> crate::ba::BundleAdjustmentOptions {
    let mut options = mapper_global_ba_options(
        config,
        reconstruction,
        global_ba_iterations_for_reconstruction(config, reconstruction),
        Some(Vec::new()),
        registered_image_indices(reconstruction),
        Some(redundant_point_ids),
        None,
    );
    options.variable_images = Some(Vec::new());
    options.constant_cameras = all_camera_indices(reconstruction);
    options.constant_rigs = reconstruction.rigs.iter().map(|rig| rig.rig_id).collect();
    options.constant_sensor_from_rig = all_non_ref_sensor_from_rig_ids(reconstruction);
    options.refine_focal_length = false;
    options.refine_principal_point = false;
    options.refine_extra_params = false;
    options.gauge = crate::ba::BundleAdjustmentGauge::Default;
    options
}

fn registered_image_indices(reconstruction: &Reconstruction) -> Vec<usize> {
    reconstruction
        .poses
        .iter()
        .enumerate()
        .filter_map(|(image, pose)| pose.is_some().then_some(image))
        .collect()
}

fn all_camera_indices(reconstruction: &Reconstruction) -> Vec<usize> {
    (0..reconstruction.cameras.len().max(1)).collect()
}

fn all_non_ref_sensor_from_rig_ids(reconstruction: &Reconstruction) -> Vec<SensorId> {
    let mut sensor_ids = Vec::new();
    for rig in &reconstruction.rigs {
        for sensor in &rig.sensors {
            if rig
                .ref_sensor_id
                .as_ref()
                .is_some_and(|ref_sensor_id| ref_sensor_id == &sensor.sensor_id)
            {
                continue;
            }
            sensor_ids.push(sensor.sensor_id.clone());
        }
    }
    sensor_ids.sort();
    sensor_ids.dedup();
    sensor_ids
}

fn find_redundant_points3d_colmap(
    min_coverage_gain: f64,
    reconstruction: &Reconstruction,
) -> Vec<usize> {
    const NUM_IMAGE_TILES_PER_DIM: usize = 8;
    const NUM_IMAGE_TILES: usize = NUM_IMAGE_TILES_PER_DIM * NUM_IMAGE_TILES_PER_DIM;

    if reconstruction.points.is_empty() {
        return Vec::new();
    }

    let image_tile_idxs = compute_image_tile_idxs_colmap(NUM_IMAGE_TILES_PER_DIM, reconstruction);
    let mut selected_points_per_image_tile =
        vec![[0usize; NUM_IMAGE_TILES]; reconstruction.poses.len()];
    let mut priority_queue = BinaryHeap::new();
    for (point_id, point) in reconstruction.points.iter().enumerate() {
        priority_queue.push(RedundantPoint3DInfo {
            point_id,
            stable_point_id: reconstruction_point3d_id(reconstruction, point_id),
            gain: compute_coverage_gain_colmap(
                point,
                &selected_points_per_image_tile,
                &image_tile_idxs,
            ),
        });
    }

    let mut selected_point_ids = HashSet::with_capacity(reconstruction.points.len());
    while let Some(mut point_info) = priority_queue.pop() {
        if point_info.gain <= min_coverage_gain {
            break;
        }
        let Some(point) = reconstruction.points.get(point_info.point_id) else {
            continue;
        };
        let updated_gain =
            compute_coverage_gain_colmap(point, &selected_points_per_image_tile, &image_tile_idxs);
        if updated_gain < point_info.gain {
            point_info.gain = updated_gain;
            priority_queue.push(point_info);
            continue;
        }
        for obs in &point.track {
            let Some(tile_idx) = image_tile_idxs
                .get(obs.image)
                .and_then(|tile_idxs| tile_idxs.get(obs.feature))
                .copied()
            else {
                continue;
            };
            if let Some(counts) = selected_points_per_image_tile.get_mut(obs.image) {
                counts[tile_idx] += 1;
            }
        }
        selected_point_ids.insert(point_info.point_id);
    }

    reconstruction
        .points
        .iter()
        .enumerate()
        .filter_map(|(point_id, _)| (!selected_point_ids.contains(&point_id)).then_some(point_id))
        .collect()
}

fn compute_image_tile_idxs_colmap(
    num_tiles_per_dim: usize,
    reconstruction: &Reconstruction,
) -> Vec<Vec<usize>> {
    reconstruction
        .keypoints
        .iter()
        .enumerate()
        .map(|(image, keypoints)| {
            let camera = reconstruction.camera_for_image(image);
            keypoints
                .iter()
                .map(|keypoint| {
                    let tile_idx_x =
                        image_tile_idx_colmap(num_tiles_per_dim, keypoint.x(), camera.width);
                    let tile_idx_y =
                        image_tile_idx_colmap(num_tiles_per_dim, keypoint.y(), camera.height);
                    tile_idx_x * num_tiles_per_dim + tile_idx_y
                })
                .collect()
        })
        .collect()
}

fn image_tile_idx_colmap(num_tiles_per_dim: usize, coord: f32, extent: u32) -> usize {
    if num_tiles_per_dim == 0 {
        return 0;
    }
    let high = num_tiles_per_dim - 1;
    if extent == 0 || !coord.is_finite() {
        return 0;
    }
    ((num_tiles_per_dim as f32 * coord / extent as f32) as isize).clamp(0, high as isize) as usize
}

fn compute_coverage_gain_colmap(
    point: &Point3D,
    selected_points_per_image_tile: &[[usize; 64]],
    image_tile_idxs: &[Vec<usize>],
) -> f64 {
    let mut gain = 0.0;
    for obs in &point.track {
        let Some(tile_idx) = image_tile_idxs
            .get(obs.image)
            .and_then(|tile_idxs| tile_idxs.get(obs.feature))
            .copied()
        else {
            continue;
        };
        let Some(tile_counts) = selected_points_per_image_tile.get(obs.image) else {
            continue;
        };
        let n = 1 + tile_counts[tile_idx];
        gain += 1.0 / (n as f64).sqrt() - 1.0 / ((1 + n) as f64).sqrt();
    }
    gain
}

fn normalize_reconstruction_colmap(
    reconstruction: &mut Reconstruction,
    fixed_scale: bool,
    extent: f32,
    min_percentile: f32,
    max_percentile: f32,
    use_images: bool,
) -> Option<ReconstructionNormalizationTransform> {
    if !extent.is_finite()
        || extent <= 0.0
        || !min_percentile.is_finite()
        || !max_percentile.is_finite()
        || !(0.0..=1.0).contains(&min_percentile)
        || !(0.0..=1.0).contains(&max_percentile)
        || min_percentile > max_percentile
    {
        return None;
    }
    if (use_images && registered_frame_count(reconstruction) < 2)
        || (!use_images && reconstruction.points.len() < 2)
    {
        return None;
    }
    let coords = if use_images {
        registered_reconstruction_camera_centers(reconstruction)
    } else {
        reconstruction
            .points
            .iter()
            .map(|point| glam::Vec3::from_array(point.xyz))
            .collect::<Vec<_>>()
    };
    if coords.len() < 2 {
        return None;
    }
    let (min, max, centroid) =
        robust_bbox_and_centroid_colmap(coords, min_percentile, max_percentile)?;
    let old_extent = (max - min).length();
    let scale = if fixed_scale || old_extent <= f32::EPSILON {
        1.0
    } else {
        extent / old_extent
    };
    let transform = ReconstructionNormalizationTransform {
        scale,
        translation: -scale * centroid,
    };
    transform_reconstruction_colmap(reconstruction, transform);
    Some(transform)
}

fn registered_reconstruction_camera_centers(reconstruction: &Reconstruction) -> Vec<glam::Vec3> {
    reconstruction
        .poses
        .iter()
        .filter_map(|pose| pose.map(camera_center))
        .collect()
}

fn robust_bbox_and_centroid_colmap(
    coords: Vec<glam::Vec3>,
    min_percentile: f32,
    max_percentile: f32,
) -> Option<(glam::Vec3, glam::Vec3, glam::Vec3)> {
    if coords.is_empty() || min_percentile > max_percentile {
        return None;
    }
    let end_idx = coords.len() - 1;
    let min_idx = ((min_percentile * end_idx as f32).floor() as usize).min(end_idx);
    let max_idx = ((max_percentile * end_idx as f32).ceil() as usize).min(end_idx);
    let mut coords_x = coords.iter().map(|coord| coord.x).collect::<Vec<_>>();
    let mut coords_y = coords.iter().map(|coord| coord.y).collect::<Vec<_>>();
    let mut coords_z = coords.iter().map(|coord| coord.z).collect::<Vec<_>>();
    coords_x.sort_by(|a, b| a.total_cmp(b));
    coords_y.sort_by(|a, b| a.total_cmp(b));
    coords_z.sort_by(|a, b| a.total_cmp(b));
    let bbox_min = glam::Vec3::new(coords_x[min_idx], coords_y[min_idx], coords_z[min_idx]);
    let bbox_max = glam::Vec3::new(coords_x[max_idx], coords_y[max_idx], coords_z[max_idx]);
    let normalization = 1.0 / (max_idx - min_idx + 1) as f32;
    let mut centroid = glam::Vec3::ZERO;
    for idx in min_idx..=max_idx {
        centroid += normalization * glam::Vec3::new(coords_x[idx], coords_y[idx], coords_z[idx]);
    }
    Some((bbox_min, bbox_max, centroid))
}

fn transform_reconstruction_colmap(
    reconstruction: &mut Reconstruction,
    transform: ReconstructionNormalizationTransform,
) {
    for rig in &mut reconstruction.rigs {
        for sensor in &mut rig.sensors {
            if rig
                .ref_sensor_id
                .as_ref()
                .is_some_and(|ref_sensor_id| ref_sensor_id == &sensor.sensor_id)
            {
                continue;
            }
            if let Some(sensor_from_rig) = sensor.sensor_from_rig.as_mut() {
                for value in &mut sensor_from_rig.tvec {
                    *value *= transform.scale as f64;
                }
            }
        }
    }
    for frame_idx in 0..reconstruction.frames.len() {
        if !frame_has_registered_image(reconstruction, frame_idx) {
            continue;
        }
        let rig_from_world = reconstruction.frames[frame_idx].rig_from_world.to_se3();
        reconstruction.frames[frame_idx].rig_from_world = Rigid3::from_se3(
            transform_camera_world_pose_colmap(rig_from_world, transform),
        );
    }
    for image in 0..reconstruction.poses.len() {
        if reconstruction.frame_index_for_image(image).is_some() {
            continue;
        }
        if let Some(pose) = reconstruction.poses[image] {
            reconstruction.poses[image] = Some(transform_camera_world_pose_colmap(pose, transform));
        }
    }
    for point in &mut reconstruction.points {
        point.xyz = (transform.scale * glam::Vec3::from_array(point.xyz) + transform.translation)
            .to_array();
    }
    sync_registered_image_poses_from_frames(reconstruction);
}

fn frame_has_registered_image(reconstruction: &Reconstruction, frame_idx: usize) -> bool {
    reconstruction
        .image_indices_for_frame_index(frame_idx)
        .into_iter()
        .any(|image| reconstruction.poses.get(image).copied().flatten().is_some())
}

fn transform_camera_world_pose_colmap(
    pose: SE3,
    transform: ReconstructionNormalizationTransform,
) -> SE3 {
    let center = camera_center(pose);
    let transformed_center = transform.scale * center + transform.translation;
    pose_from_rotation_center(pose_rotation(pose), transformed_center)
}

fn sync_registered_image_poses_from_frames(reconstruction: &mut Reconstruction) {
    for frame_idx in 0..reconstruction.frames.len() {
        let rig_from_world = reconstruction.frames[frame_idx].rig_from_world.to_se3();
        let rig_id = reconstruction.frames[frame_idx].rig_id;
        for image in reconstruction.image_indices_for_frame_index(frame_idx) {
            if reconstruction.poses.get(image).copied().flatten().is_none() {
                continue;
            }
            let pose = reconstruction
                .frame_sensor_id_for_image(frame_idx, image)
                .and_then(|sensor_id| reconstruction.sensor_from_rig(rig_id, sensor_id))
                .map(|sensor_from_rig| sensor_from_rig.compose(&rig_from_world))
                .unwrap_or(rig_from_world);
            if let Some(slot) = reconstruction.poses.get_mut(image) {
                *slot = Some(pose);
            }
        }
    }
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

fn filter_registered_frames(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    config: &MapperConfig,
    registration_stats: &mut RegistrationStats,
    mut filtered_units: Option<&mut HashSet<RegistrationUnitKey>>,
    triangulation_state: &mut IncrementalTriangulatorState,
) -> usize {
    const MIN_NUM_REGISTERED_FRAMES_FOR_FILTERING: usize = 20;
    if registered_frame_count(reconstruction) < MIN_NUM_REGISTERED_FRAMES_FOR_FILTERING {
        return 0;
    }

    let mut seen_units = HashSet::new();
    let mut candidates = Vec::new();
    for image in 0..reconstruction.poses.len() {
        if !registration_unit_is_registered(reconstruction, image) {
            continue;
        }
        let unit = registration_unit_key(reconstruction, image);
        if !seen_units.insert(unit)
            || !registered_unit_should_be_filtered(reconstruction, image, config)
            || (config.fix_existing_frames
                && registration_stats.is_existing_registration_unit(reconstruction, image))
        {
            continue;
        }
        candidates.push(image);
    }

    let mut filtered = 0usize;
    let observation_manager = triangulation_state.observation_manager_mut();
    for image in candidates {
        if !registration_unit_is_registered(reconstruction, image) {
            continue;
        }
        let unit = registration_unit_key(reconstruction, image);
        if observation_manager.deregister_frame_for_image(frames, pairs, reconstruction, image) {
            registration_stats.deregister_frame_for_image_event(reconstruction, image);
            if let Some(filtered_units) = filtered_units.as_deref_mut() {
                filtered_units.insert(unit);
            }
            filtered += 1;
        }
    }
    filtered
}

fn registered_unit_should_be_filtered(
    reconstruction: &Reconstruction,
    image: usize,
    config: &MapperConfig,
) -> bool {
    let mut num_point3d_observations = 0usize;
    for frame_image in reconstruction.image_indices_for_registration_unit(image) {
        if reconstruction
            .poses
            .get(frame_image)
            .copied()
            .flatten()
            .is_none()
        {
            continue;
        }
        if camera_has_bogus_params(reconstruction.camera_for_image(frame_image), config) {
            return true;
        }
        num_point3d_observations += reconstruction
            .observations
            .get(frame_image)
            .map(|observations| {
                observations
                    .iter()
                    .filter(|point_id| point_id.is_some())
                    .count()
            })
            .unwrap_or(0);
    }
    num_point3d_observations < 1
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
    triangulation_state: &mut IncrementalTriangulatorState,
) -> Option<LocalBundleReport> {
    if !local_ba_enabled(config) {
        return None;
    }

    let mut last_report: Option<LocalBundleReport> = None;
    for round in 0..config.local_ba_max_refinements {
        let Some(mut report) = refine_local_bundle_round(
            frames,
            pairs,
            reconstruction,
            registered_image,
            gauge_image,
            tri_options,
            config,
            registration_stats,
            triangulation_state,
            mapper_local_ba_refinement_loss_function(round),
        ) else {
            break;
        };
        report.refinements = round + 1;
        let changed = report.changed_observation_ratio;
        if let Some(accumulated) = last_report.as_mut() {
            accumulated.report = report.report;
            accumulated.refinements = report.refinements;
            accumulated.variable_images = report.variable_images;
            accumulated.local_images = report.local_images;
            accumulated.points = report.points;
            accumulated.merged_observations += report.merged_observations;
            accumulated.completed_observations += report.completed_observations;
            accumulated.completed_image_observations += report.completed_image_observations;
            accumulated.filtered_observations += report.filtered_observations;
            accumulated.changed_observation_ratio = changed;
        } else {
            last_report = Some(report);
        }
        if changed < config.local_ba_max_refinement_change {
            break;
        }
    }
    last_report
}

fn local_bundle_refinement_required(
    reconstruction: &Reconstruction,
    registered_image: usize,
    gauge_image: usize,
    config: &MapperConfig,
) -> bool {
    if !local_ba_enabled(config) {
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

fn local_ba_enabled(config: &MapperConfig) -> bool {
    config.local_ba && config.local_ba_iterations > 0 && config.local_ba_max_refinements > 0
}

fn refine_local_bundle_round(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    registered_image: usize,
    gauge_image: usize,
    tri_options: &IncrementalTriangulatorOptions,
    config: &MapperConfig,
    registration_stats: &RegistrationStats,
    triangulation_state: &mut IncrementalTriangulatorState,
    loss_function: crate::ba::BundleAdjustmentLoss,
) -> Option<LocalBundleReport> {
    let local_bundle = select_local_bundle(
        reconstruction,
        registered_image,
        gauge_image,
        config.local_ba_num_images,
        config.local_ba_min_shared_points,
    )?;
    let mut ba_options = mapper_local_ba_options(
        config,
        reconstruction,
        registration_stats,
        config.local_ba_iterations,
        local_bundle.variable_images.clone(),
        vec![gauge_image],
        Some(local_bundle.point_ids.clone()),
        Some(local_bundle.constant_point_ids.clone()),
    );
    ba_options.loss_function = loss_function;
    let report =
        refine_bundle_adjustment_checked(frames, reconstruction, config, ba_options).ok()?;
    let post_ba_point_ids =
        point_indices_for_stable_point_ids(reconstruction, &local_bundle.stable_point_ids);
    for &point_id in &post_ba_point_ids {
        triangulation_state
            .observation_manager_mut()
            .mark_point3d_modified(point_id);
    }
    let (merged_observations, completed_observations, completed_image_observations) = {
        let mut triangulator =
            IncrementalTriangulator::new(frames, pairs, reconstruction, triangulation_state);
        let modified = triangulator.get_modified_points3d().clone();
        let merged = triangulator.merge_tracks(tri_options, &modified);
        let modified = triangulator.get_modified_points3d().clone();
        let completed = triangulator.complete_tracks(tri_options, &modified);
        let complete_report = triangulator.complete_image(tri_options, registered_image);
        (merged, completed, complete_report.total_observations())
    };
    let filtered_observations = filter_modified_reprojection_tracks_with_state(
        frames,
        pairs,
        reconstruction,
        config,
        triangulation_state,
    );
    let changed_observation_ratio = local_ba_refinement_change_ratio(
        report.observations,
        merged_observations,
        completed_observations + completed_image_observations,
        filtered_observations,
    );
    let variable_image_count =
        expand_images_to_registration_frames(reconstruction, &local_bundle.variable_images).len();
    Some(LocalBundleReport {
        report,
        refinements: 1,
        variable_images: variable_image_count,
        local_images: local_bundle.local_images.len(),
        points: local_bundle.point_ids.len(),
        merged_observations,
        completed_observations,
        completed_image_observations,
        filtered_observations,
        changed_observation_ratio,
    })
}

fn local_ba_refinement_change_ratio(
    adjusted_observations: usize,
    merged_observations: usize,
    completed_observations: usize,
    filtered_observations: usize,
) -> f32 {
    if adjusted_observations == 0 {
        0.0
    } else {
        (merged_observations + completed_observations + filtered_observations) as f32
            / adjusted_observations as f32
    }
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

fn initial_pair_probe_num_threads(config: &MapperConfig) -> usize {
    config.threads.filter(|&threads| threads > 1).unwrap_or(1)
}

fn choose_initial_pair(
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
    camera_has_prior_focal_length: &[bool],
    selection_state: &mut InitialPairSelectionState,
) -> Option<PairGeometry> {
    if initial_pair_probe_num_threads(config) > 1 {
        choose_initial_pair_parallel(
            pairs,
            reconstruction,
            config,
            camera_has_prior_focal_length,
            selection_state,
        )
    } else {
        choose_initial_pair_sequential(
            pairs,
            reconstruction,
            config,
            camera_has_prior_focal_length,
            selection_state,
        )
    }
}

fn choose_initial_pair_sequential(
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
    camera_has_prior_focal_length: &[bool],
    selection_state: &mut InitialPairSelectionState,
) -> Option<PairGeometry> {
    let image_correspondences = image_correspondence_counts(pairs);
    for image_id1 in sorted_initial_image_ids(
        reconstruction,
        &image_correspondences,
        camera_has_prior_focal_length,
        selection_state,
        config,
    ) {
        if let Some(pair) = probe_initial_pairs_for_first_image(
            pairs,
            reconstruction,
            config,
            camera_has_prior_focal_length,
            selection_state,
            image_id1,
        ) {
            return Some(pair);
        }
    }
    None
}

fn choose_initial_pair_parallel(
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
    camera_has_prior_focal_length: &[bool],
    selection_state: &mut InitialPairSelectionState,
) -> Option<PairGeometry> {
    let candidates = initial_pair_candidates_in_colmap_order(
        pairs,
        reconstruction,
        config,
        camera_has_prior_focal_length,
        selection_state,
    );
    let results = candidates
        .par_iter()
        .map(|candidate| {
            probe_initial_pair_candidate(pairs, reconstruction, config, candidate)
                .map(|pair| (candidate.pair_id, pair))
        })
        .collect::<Vec<_>>();

    for (candidate, result) in candidates.iter().zip(results) {
        selection_state.init_image_pairs.insert(candidate.pair_id);
        if let Some((_, pair)) = result {
            return Some(pair);
        }
    }
    None
}

fn probe_initial_pairs_for_first_image(
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
    camera_has_prior_focal_length: &[bool],
    selection_state: &mut InitialPairSelectionState,
    image_id1: usize,
) -> Option<PairGeometry> {
    for image_id2 in sorted_second_initial_image_ids(
        pairs,
        reconstruction,
        image_id1,
        camera_has_prior_focal_length,
        selection_state,
        config,
    ) {
        if !selection_state.mark_initial_pair_tried(reconstruction, image_id1, image_id2) {
            continue;
        }
        let Some(pair) = oriented_initial_pair(pairs, image_id1, image_id2) else {
            continue;
        };
        let Some(pair) = initial_pair_geometry_for_gate(&pair, reconstruction, config) else {
            continue;
        };
        if is_colmap_style_initial_pair(&pair, reconstruction, config) {
            return Some(pair);
        }
    }
    None
}

#[derive(Debug, Clone, Copy)]
struct InitialPairCandidate {
    image_id1: usize,
    image_id2: usize,
    pair_id: ImagePairId,
}

fn initial_pair_candidates_in_colmap_order(
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
    camera_has_prior_focal_length: &[bool],
    selection_state: &InitialPairSelectionState,
) -> Vec<InitialPairCandidate> {
    let image_correspondences = image_correspondence_counts(pairs);
    let image_ids1 = sorted_initial_image_ids(
        reconstruction,
        &image_correspondences,
        camera_has_prior_focal_length,
        selection_state,
        config,
    );
    let mut candidates = Vec::new();
    let mut tried_pairs = selection_state.init_image_pairs.clone();
    for image_id1 in image_ids1 {
        for image_id2 in sorted_second_initial_image_ids(
            pairs,
            reconstruction,
            image_id1,
            camera_has_prior_focal_length,
            selection_state,
            config,
        ) {
            let Some(pair_id) = initial_image_pair_id(reconstruction, image_id1, image_id2) else {
                continue;
            };
            if !tried_pairs.insert(pair_id) {
                continue;
            }
            candidates.push(InitialPairCandidate {
                image_id1,
                image_id2,
                pair_id,
            });
        }
    }
    candidates
}

fn probe_initial_pair_candidate(
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
    candidate: &InitialPairCandidate,
) -> Option<PairGeometry> {
    let pair = oriented_initial_pair(pairs, candidate.image_id1, candidate.image_id2)?;
    let pair = initial_pair_geometry_for_gate(&pair, reconstruction, config)?;
    is_colmap_style_initial_pair(&pair, reconstruction, config).then_some(pair)
}

fn initial_pair_geometry_for_gate(
    pair: &PairGeometry,
    reconstruction: &Reconstruction,
    config: &MapperConfig,
) -> Option<PairGeometry> {
    if should_reestimate_initial_pair_geometry(pair) {
        reestimate_initial_pair_geometry(pair, reconstruction, config)
    } else {
        Some(pair.clone())
    }
}

fn should_reestimate_initial_pair_geometry(pair: &PairGeometry) -> bool {
    let disabled = std::env::var("RUSTSFM_REESTIMATE_INITIAL_PAIR")
        .ok()
        .is_some_and(|value| value == "0" || value.eq_ignore_ascii_case("false"));
    !disabled
        && (pair.e_matrix.is_some()
            || pair.f_matrix.is_some()
            || pair.h_matrix.is_some()
            || pair.qvec.is_some()
            || pair.tvec.is_some())
}

fn reestimate_initial_pair_geometry(
    pair: &PairGeometry,
    reconstruction: &Reconstruction,
    config: &MapperConfig,
) -> Option<PairGeometry> {
    let matches = &pair.inlier_matches;
    if matches.is_empty() {
        return None;
    }
    let left_frame = initial_pair_frame_from_reconstruction(reconstruction, pair.left)?;
    let right_frame = initial_pair_frame_from_reconstruction(reconstruction, pair.right)?;
    estimate_pair_geometry_with_options_and_cameras(
        pair.left,
        pair.right,
        &left_frame,
        &right_frame,
        matches,
        reconstruction.camera_for_image(pair.left),
        reconstruction.camera_for_image(pair.right),
        config.essential_threshold_px,
        config.essential_iterations,
        config.init_min_num_inliers,
        config.min_triangulated,
        PairEstimationOptions {
            max_pose_matches: 0,
            refine_sampson: false,
            ransac_random_seed: config.random_seed,
            expand_dense_inliers: false,
            ..PairEstimationOptions::default()
        },
    )
}

fn initial_pair_frame_from_reconstruction(
    reconstruction: &Reconstruction,
    image: usize,
) -> Option<ImageFrame> {
    let keypoints = reconstruction.keypoints.get(image)?.clone();
    let camera = reconstruction.camera_for_image(image);
    Some(ImageFrame {
        id: image,
        name: reconstruction
            .image_names
            .get(image)
            .cloned()
            .unwrap_or_else(|| format!("image_{image}.jpg")),
        path: reconstruction
            .image_paths
            .get(image)
            .cloned()
            .unwrap_or_else(|| PathBuf::from(format!("image_{image}.jpg"))),
        width: camera.width,
        height: camera.height,
        keypoints,
        descriptors: rustslam::Descriptors::new(),
        sift: crate::sift::SiftFeatures::default(),
        wide_descriptors: crate::wide::WideDescriptors {
            data: Vec::new(),
            dim: 0,
            count: 0,
        },
        strong_feature_indices: Vec::new(),
        colors: vec![[0, 0, 0]; reconstruction.keypoints.get(image)?.len()],
    })
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
        && generalized_initial_pair_gate(pair, reconstruction)
}

fn same_registration_frame(reconstruction: &Reconstruction, left: usize, right: usize) -> bool {
    left != right
        && reconstruction
            .frame_index_for_image(left)
            .zip(reconstruction.frame_index_for_image(right))
            .is_some_and(|(left_frame, right_frame)| left_frame == right_frame)
}

fn generalized_initial_pair_gate(pair: &PairGeometry, reconstruction: &Reconstruction) -> bool {
    if non_trivial_registration_rig(reconstruction, pair.left)
        || non_trivial_registration_rig(reconstruction, pair.right)
    {
        pair.two_view_config == crate::database::COLMAP_TWO_VIEW_CALIBRATED_RIG
    } else {
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InitialImageInfo {
    image: usize,
    prior_focal_length: bool,
    num_correspondences: usize,
}

fn sorted_initial_image_ids(
    reconstruction: &Reconstruction,
    image_correspondences: &HashMap<usize, usize>,
    camera_has_prior_focal_length: &[bool],
    selection_state: &InitialPairSelectionState,
    config: &MapperConfig,
) -> Vec<usize> {
    let mut image_infos = image_correspondences
        .iter()
        .filter_map(|(&image, &num_correspondences)| {
            (num_correspondences > 0
                && image < reconstruction.poses.len()
                && selection_state.first_image_available_for_initialization(image, config))
            .then_some(InitialImageInfo {
                image,
                prior_focal_length: image_has_prior_focal_length(
                    reconstruction,
                    image,
                    camera_has_prior_focal_length,
                ),
                num_correspondences,
            })
        })
        .collect::<Vec<_>>();
    image_infos.sort_by(compare_initial_image_infos);
    image_infos.into_iter().map(|info| info.image).collect()
}

fn sorted_second_initial_image_ids(
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    first_image: usize,
    camera_has_prior_focal_length: &[bool],
    selection_state: &InitialPairSelectionState,
    config: &MapperConfig,
) -> Vec<usize> {
    let mut counts = HashMap::<usize, usize>::new();
    for pair in pairs {
        if pair.left == first_image {
            *counts.entry(pair.right).or_default() += pair.inliers;
        } else if pair.right == first_image {
            *counts.entry(pair.left).or_default() += pair.inliers;
        }
    }
    let mut image_infos = counts
        .into_iter()
        .filter_map(|(image, num_correspondences)| {
            (num_correspondences >= config.init_min_num_inliers
                && image < reconstruction.poses.len()
                && selection_state.image_not_registered_in_other_reconstruction(image))
            .then_some(InitialImageInfo {
                image,
                prior_focal_length: image_has_prior_focal_length(
                    reconstruction,
                    image,
                    camera_has_prior_focal_length,
                ),
                num_correspondences,
            })
        })
        .collect::<Vec<_>>();
    image_infos.sort_by(compare_initial_image_infos);
    image_infos.into_iter().map(|info| info.image).collect()
}

fn compare_initial_image_infos(a: &InitialImageInfo, b: &InitialImageInfo) -> std::cmp::Ordering {
    b.prior_focal_length
        .cmp(&a.prior_focal_length)
        .then_with(|| b.num_correspondences.cmp(&a.num_correspondences))
        .then_with(|| a.image.cmp(&b.image))
}

fn image_has_prior_focal_length(
    reconstruction: &Reconstruction,
    image: usize,
    camera_has_prior_focal_length: &[bool],
) -> bool {
    let camera_idx = reconstruction
        .image_camera_indices
        .get(image)
        .copied()
        .unwrap_or(0);
    camera_has_prior_focal_length
        .get(camera_idx)
        .copied()
        .unwrap_or(true)
}

fn initial_image_pair_id(
    reconstruction: &Reconstruction,
    left: usize,
    right: usize,
) -> Option<ImagePairId> {
    image_pair_to_pair_id(
        reconstruction.image_id(left),
        reconstruction.image_id(right),
    )
    .ok()
}

fn oriented_initial_pair(
    pairs: &[PairGeometry],
    left: usize,
    right: usize,
) -> Option<PairGeometry> {
    pairs.iter().find_map(|pair| {
        if pair.left == left && pair.right == right {
            Some(pair.clone())
        } else if pair.left == right && pair.right == left {
            Some(invert_pair_geometry(pair))
        } else {
            None
        }
    })
}

fn invert_pair_geometry(pair: &PairGeometry) -> PairGeometry {
    let mut inverted = pair.clone();
    std::mem::swap(&mut inverted.left, &mut inverted.right);
    swap_rustslam_matches(&mut inverted.matches);
    swap_rustslam_matches(&mut inverted.inlier_matches);
    inverted.relative_pose = pair.relative_pose.inverse();
    inverted.f_matrix = pair.f_matrix.map(transpose3);
    inverted.e_matrix = pair.e_matrix.map(transpose3);
    inverted.h_matrix = pair.h_matrix.and_then(invert_matrix3);
    if let (Some(qvec), Some(tvec)) = (pair.qvec, pair.tvec) {
        let rotation = glam::DQuat::from_xyzw(qvec[1], qvec[2], qvec[3], qvec[0]).normalize();
        let translation = glam::DVec3::from_array(tvec);
        let inverse_rotation = rotation.inverse();
        let inverse_translation = -(inverse_rotation * translation);
        inverted.qvec = Some([
            inverse_rotation.w,
            inverse_rotation.x,
            inverse_rotation.y,
            inverse_rotation.z,
        ]);
        inverted.tvec = Some(inverse_translation.to_array());
    }
    inverted
}

fn swap_rustslam_matches(matches: &mut [rustslam::Match]) {
    for match_ in matches {
        std::mem::swap(&mut match_.query_idx, &mut match_.train_idx);
    }
}

fn transpose3(matrix: [f64; 9]) -> [f64; 9] {
    [
        matrix[0], matrix[3], matrix[6], matrix[1], matrix[4], matrix[7], matrix[2], matrix[5],
        matrix[8],
    ]
}

fn invert_matrix3(matrix: [f64; 9]) -> Option<[f64; 9]> {
    let m = Matrix3::<f64>::from_row_slice(&matrix);
    let inv = m.try_inverse()?;
    Some([
        inv[(0, 0)],
        inv[(0, 1)],
        inv[(0, 2)],
        inv[(1, 0)],
        inv[(1, 1)],
        inv[(1, 2)],
        inv[(2, 0)],
        inv[(2, 1)],
        inv[(2, 2)],
    ])
}

fn non_trivial_registration_rig(reconstruction: &Reconstruction, image: usize) -> bool {
    let Some(frame_idx) = reconstruction.frame_index_for_image(image) else {
        return false;
    };
    let Some(frame) = reconstruction.frames.get(frame_idx) else {
        return false;
    };
    reconstruction
        .rigs
        .iter()
        .find(|rig| rig.rig_id == frame.rig_id)
        .is_some_and(|rig| rig.sensors.len() > 1)
}

fn initial_pair_forward_motion(pair: &PairGeometry) -> f32 {
    if let Some(tvec) = pair.tvec {
        let tz = tvec[2].abs() as f32;
        if tz.is_finite() {
            return tz;
        }
    }
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
    point_inliers: Vec<AbsolutePosePointInlier>,
    structureless_inliers: Vec<StructurelessInlier>,
    frame_image_poses: Vec<(usize, SE3)>,
    generalized_inliers: Vec<GeneralizedFrameInlier>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct AbsolutePosePointInlier {
    feature: usize,
    point_id: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct StructurelessInlier {
    image: usize,
    feature: usize,
    other: usize,
    other_feature: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct GeneralizedFrameInlier {
    image: usize,
    feature: usize,
    point_id: usize,
}

#[derive(Debug, Clone)]
struct GeneralizedFrameAbsolutePoseProblem {
    points2d: Vec<[f64; 2]>,
    points3d: Vec<[f64; 3]>,
    camera_idxs: Vec<usize>,
    cams_from_rig: Vec<SE3>,
    cameras: Vec<CameraModel>,
    correspondences: Vec<GeneralizedFrameInlier>,
}

#[derive(Debug, Clone)]
struct GeneralizedFrameRefinement {
    frame_image_poses: Vec<(usize, SE3)>,
    inliers: Vec<GeneralizedFrameInlier>,
    mean_error_px: f32,
}

fn generalized_frame_registration_applicable(
    image: usize,
    reconstruction: &Reconstruction,
    config: &MapperConfig,
    obs_manager: &ObservationManager,
    camera_has_prior_focal_length: &[bool],
    registration_stats: &RegistrationStats,
) -> bool {
    if !non_trivial_registration_rig(reconstruction, image) {
        return false;
    }
    let Some(frame_idx) = reconstruction.frame_index_for_image(image) else {
        return false;
    };
    let frame_images = reconstruction.image_indices_for_frame_index(frame_idx);
    if frame_images.len() < 2 {
        return false;
    }
    for &frame_image in &frame_images {
        let camera = reconstruction.camera_for_image(frame_image);
        if camera_has_bogus_params(camera, config) {
            return false;
        }
        if !frame_image_camera_has_good_focal_length(
            frame_image,
            reconstruction,
            camera_has_prior_focal_length,
            registration_stats,
        ) {
            return false;
        }
    }
    let visible_points = frame_images
        .iter()
        .map(|&frame_image| obs_manager.num_visible_points3d(frame_image))
        .sum::<usize>();
    visible_points >= config.abs_pose_min_num_inliers
}

fn solve_generalized_frame_absolute_pose(
    image: usize,
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
    obs_manager: &ObservationManager,
    camera_has_prior_focal_length: &[bool],
    registration_stats: &RegistrationStats,
    graph: Option<&CorrespondenceGraph>,
) -> Option<AbsolutePose> {
    if !generalized_frame_registration_applicable(
        image,
        reconstruction,
        config,
        obs_manager,
        camera_has_prior_focal_length,
        registration_stats,
    ) {
        return None;
    }
    let frame_idx = reconstruction.frame_index_for_image(image)?;

    let problem = collect_generalized_frame_absolute_pose_problem(
        frame_idx,
        frames,
        pairs,
        reconstruction,
        config,
        graph,
    )?;
    if problem.points2d.len() < config.abs_pose_min_num_inliers {
        return None;
    }

    let mut options = GeneralizedAbsolutePoseEstimationOptions::default();
    options.ransac_options.max_error = config.pnp_threshold_px as f64;
    options.ransac_options.min_inlier_ratio = config.abs_pose_min_inlier_ratio as f64;
    options.ransac_options.random_seed = config.random_seed;
    options.ransac_options.num_threads =
        config.threads.map(|threads| threads as isize).unwrap_or(-1);

    let estimate = match estimate_generalized_absolute_pose(
        &options,
        GeneralizedAbsolutePoseProblem {
            points2d: &problem.points2d,
            points3d: &problem.points3d,
            camera_idxs: &problem.camera_idxs,
            cams_from_rig: &problem.cams_from_rig,
            cameras: &problem.cameras,
        },
    ) {
        Ok(Some(estimate)) => estimate,
        Ok(None) => return None,
        Err(GeneralizedPoseError::MissingGeneralizedRelativePoseSolver) => {
            log::debug!(
                "COLMAP generalized frame registration skipped: GP3P solver is not enabled"
            );
            return None;
        }
        Err(err) => {
            log::debug!("COLMAP generalized frame registration skipped: {err}");
            return None;
        }
    };

    if estimate.num_unique_inliers < config.abs_pose_min_num_inliers {
        return None;
    }
    let refinement = refine_generalized_frame_absolute_pose(
        frame_idx,
        frames,
        reconstruction,
        config,
        &problem,
        estimate.rig_from_world,
        &estimate.inlier_mask,
    )?;
    let unique_inliers = unique_generalized_frame_point_count(&refinement.inliers);
    if unique_inliers < config.abs_pose_min_num_inliers
        || refinement.inliers.len() < config.abs_pose_min_num_inliers
    {
        return None;
    }
    let frame_image_poses = refinement.frame_image_poses;
    let selected_pose = frame_image_poses
        .iter()
        .find_map(|&(frame_image, pose)| (frame_image == image).then_some(pose))?;
    let camera = reconstruction.camera_for_image(image);

    Some(AbsolutePose {
        pose: selected_pose,
        camera,
        inliers: unique_inliers,
        inlier_ratio: unique_inliers as f32 / problem.points2d.len().max(1) as f32,
        mean_error_px: refinement.mean_error_px,
        point_inliers: Vec::new(),
        structureless_inliers: Vec::new(),
        frame_image_poses,
        generalized_inliers: refinement.inliers,
    })
}

fn refine_generalized_frame_absolute_pose(
    frame_idx: usize,
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
    problem: &GeneralizedFrameAbsolutePoseProblem,
    rig_from_world: SE3,
    inlier_mask: &[bool],
) -> Option<GeneralizedFrameRefinement> {
    if inlier_mask.len() != problem.correspondences.len() {
        return None;
    }
    let mut initial_inliers = inlier_mask
        .iter()
        .zip(problem.correspondences.iter())
        .filter_map(|(&is_inlier, &correspondence)| is_inlier.then_some(correspondence))
        .collect::<Vec<_>>();
    initial_inliers.sort_unstable();
    initial_inliers.dedup();
    if unique_generalized_frame_point_count(&initial_inliers) < config.abs_pose_min_num_inliers {
        return None;
    }

    let mut scratch = generalized_frame_refinement_reconstruction(
        frame_idx,
        reconstruction,
        problem,
        rig_from_world,
        inlier_mask,
    )?;
    let variable_images = scratch.image_indices_for_frame_index(frame_idx);
    if variable_images.is_empty() {
        return None;
    }
    let constant_points = (0..scratch.points.len()).collect::<Vec<_>>();
    let mut options = crate::ba::BundleAdjustmentOptions {
        iterations: 100,
        gradient_tolerance: 1.0,
        function_tolerance: 0.0,
        parameter_tolerance: 0.0,
        max_linear_solver_iterations: 100,
        max_observation_error_px: f64::INFINITY,
        loss_function: crate::ba::BundleAdjustmentLoss::Huber { scale: 1.0 },
        variable_images: Some(variable_images),
        constant_cameras: (0..scratch.cameras.len()).collect(),
        point_ids: Some(constant_points.clone()),
        constant_point_ids: Some(constant_points),
        allow_single_observation_points: true,
        ..crate::ba::BundleAdjustmentOptions::default()
    };
    options.constant_sensor_from_rig = scratch
        .rigs
        .iter()
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
        .collect();

    let report = crate::ba::refine_bundle_adjustment(frames, &mut scratch, options)?;
    if !report.is_solution_usable() {
        return None;
    }
    let refined_rig_from_world = scratch.frames.get(frame_idx)?.rig_from_world.to_se3();
    if unique_generalized_frame_point_count(&initial_inliers) < config.abs_pose_min_num_inliers {
        return None;
    }
    let frame_image_poses =
        frame_image_poses_from_rig_pose(reconstruction, frame_idx, refined_rig_from_world)?;
    let mean_error_px = generalized_frame_mean_error_px(
        refined_rig_from_world,
        reconstruction,
        problem,
        &initial_inliers,
    );
    Some(GeneralizedFrameRefinement {
        frame_image_poses,
        inliers: initial_inliers,
        mean_error_px,
    })
}

fn frame_image_camera_has_good_focal_length(
    image: usize,
    reconstruction: &Reconstruction,
    camera_has_prior_focal_length: &[bool],
    registration_stats: &RegistrationStats,
) -> bool {
    let camera_idx = reconstruction
        .image_camera_indices
        .get(image)
        .copied()
        .unwrap_or(0);
    if camera_has_prior_focal_length
        .get(camera_idx)
        .copied()
        .unwrap_or(true)
    {
        return true;
    }
    let camera_id = reconstruction
        .camera_ids
        .get(camera_idx)
        .copied()
        .unwrap_or(1);
    registration_stats.registered_images_with_camera_id(camera_id) > 0
}

fn generalized_frame_refinement_reconstruction(
    frame_idx: usize,
    reconstruction: &Reconstruction,
    problem: &GeneralizedFrameAbsolutePoseProblem,
    rig_from_world: SE3,
    inlier_mask: &[bool],
) -> Option<Reconstruction> {
    let mut scratch = reconstruction.clone();
    scratch.points.clear();
    scratch.point_ids.clear();
    scratch.observations = scratch
        .keypoints
        .iter()
        .map(|keypoints| vec![None; keypoints.len()])
        .collect();
    let frame_image_poses =
        frame_image_poses_from_rig_pose(reconstruction, frame_idx, rig_from_world)?;
    for (image, pose) in frame_image_poses {
        if let Some(slot) = scratch.poses.get_mut(image) {
            *slot = Some(pose);
        }
    }
    if let Some(frame) = scratch.frames.get_mut(frame_idx) {
        frame.rig_from_world = Rigid3::from_se3(rig_from_world);
    }

    let mut point_idx_by_source = HashMap::<usize, usize>::new();
    for (&is_inlier, correspondence) in inlier_mask.iter().zip(problem.correspondences.iter()) {
        if !is_inlier {
            continue;
        }
        let Some(point_idx) = point_idx_by_source
            .get(&correspondence.point_id)
            .copied()
            .or_else(|| {
                let source_point = reconstruction.points.get(correspondence.point_id)?;
                let point_idx = scratch.points.len();
                scratch.points.push(Point3D {
                    xyz: source_point.xyz,
                    color: source_point.color,
                    error: source_point.error,
                    track: Vec::new(),
                });
                scratch
                    .point_ids
                    .push(reconstruction.point3d_id(correspondence.point_id));
                point_idx_by_source.insert(correspondence.point_id, point_idx);
                Some(point_idx)
            })
        else {
            continue;
        };
        if correspondence.image >= scratch.observations.len()
            || correspondence.feature >= scratch.observations[correspondence.image].len()
        {
            continue;
        }
        if scratch.observations[correspondence.image][correspondence.feature].is_some() {
            continue;
        }
        scratch.observations[correspondence.image][correspondence.feature] = Some(point_idx);
        scratch.points[point_idx].track.push(TrackObservation {
            image: correspondence.image,
            feature: correspondence.feature,
        });
    }
    (!scratch.points.is_empty()).then_some(scratch)
}

#[cfg(all(test, feature = "poselib"))]
fn generalized_frame_inliers_for_pose(
    rig_from_world: SE3,
    reconstruction: &Reconstruction,
    problem: &GeneralizedFrameAbsolutePoseProblem,
    max_error_px: f64,
) -> Vec<GeneralizedFrameInlier> {
    let mut inliers = Vec::new();
    for (idx, correspondence) in problem.correspondences.iter().enumerate() {
        let Some(&camera_idx) = problem.camera_idxs.get(idx) else {
            continue;
        };
        let Some(&sensor_from_rig) = problem.cams_from_rig.get(camera_idx) else {
            continue;
        };
        let Some(&point3d) = problem.points3d.get(idx) else {
            continue;
        };
        let Some(&point2d) = problem.points2d.get(idx) else {
            continue;
        };
        let Some(&camera) = problem.cameras.get(camera_idx) else {
            continue;
        };
        let cam_from_world = sensor_from_rig.compose(&rig_from_world);
        let point = [point3d[0] as f32, point3d[1] as f32, point3d[2] as f32];
        let Some(predicted) = project_point_px(point, cam_from_world, camera) else {
            continue;
        };
        let error = ((predicted[0] as f64 - point2d[0]).powi(2)
            + (predicted[1] as f64 - point2d[1]).powi(2))
        .sqrt();
        if error.is_finite() && error <= max_error_px {
            inliers.push(*correspondence);
        }
    }
    inliers.sort_unstable();
    inliers.dedup();
    inliers.retain(|inlier| {
        reconstruction
            .points
            .get(inlier.point_id)
            .is_some_and(|point| point.xyz.iter().all(|value| value.is_finite()))
    });
    inliers
}

fn generalized_frame_mean_error_px(
    rig_from_world: SE3,
    _reconstruction: &Reconstruction,
    problem: &GeneralizedFrameAbsolutePoseProblem,
    inliers: &[GeneralizedFrameInlier],
) -> f32 {
    if inliers.is_empty() {
        return 0.0;
    }
    let inlier_set = inliers
        .iter()
        .map(|inlier| (inlier.image, inlier.feature, inlier.point_id))
        .collect::<HashSet<_>>();
    let mut sum = 0.0;
    let mut count = 0usize;
    for (idx, correspondence) in problem.correspondences.iter().enumerate() {
        if !inlier_set.contains(&(
            correspondence.image,
            correspondence.feature,
            correspondence.point_id,
        )) {
            continue;
        }
        let Some(&camera_idx) = problem.camera_idxs.get(idx) else {
            continue;
        };
        let Some(&sensor_from_rig) = problem.cams_from_rig.get(camera_idx) else {
            continue;
        };
        let Some(&point3d) = problem.points3d.get(idx) else {
            continue;
        };
        let Some(&point2d) = problem.points2d.get(idx) else {
            continue;
        };
        let Some(&camera) = problem.cameras.get(camera_idx) else {
            continue;
        };
        let cam_from_world = sensor_from_rig.compose(&rig_from_world);
        let point = [point3d[0] as f32, point3d[1] as f32, point3d[2] as f32];
        let Some(predicted) = project_point_px(point, cam_from_world, camera) else {
            continue;
        };
        let error = ((predicted[0] as f64 - point2d[0]).powi(2)
            + (predicted[1] as f64 - point2d[1]).powi(2))
        .sqrt();
        if error.is_finite() {
            sum += error;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        (sum / count as f64) as f32
    }
}

fn unique_generalized_frame_point_count(inliers: &[GeneralizedFrameInlier]) -> usize {
    inliers
        .iter()
        .map(|inlier| inlier.point_id)
        .collect::<HashSet<_>>()
        .len()
}

fn collect_generalized_frame_absolute_pose_problem(
    frame_idx: usize,
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
    graph: Option<&CorrespondenceGraph>,
) -> Option<GeneralizedFrameAbsolutePoseProblem> {
    let frame = reconstruction.frames.get(frame_idx)?;
    let frame_images = reconstruction.image_indices_for_frame_index(frame_idx);
    if frame_images.is_empty() {
        return None;
    }

    let mut local_camera_idx_by_image = HashMap::new();
    let mut cams_from_rig = Vec::new();
    let mut cameras = Vec::new();
    for &frame_image in &frame_images {
        let sensor_id = reconstruction.frame_sensor_id_for_image(frame_idx, frame_image)?;
        let sensor_from_rig = reconstruction
            .sensor_from_rig(frame.rig_id, sensor_id)
            .unwrap_or_else(SE3::identity);
        local_camera_idx_by_image.insert(frame_image, cameras.len());
        cams_from_rig.push(sensor_from_rig);
        cameras.push(reconstruction.camera_for_image(frame_image));
    }

    let mut points2d = Vec::new();
    let mut points3d = Vec::new();
    let mut camera_idxs = Vec::new();
    let mut correspondences = Vec::new();
    let mut seen = HashSet::<(usize, usize, usize)>::new();

    if let Some(graph) = graph {
        for &query_image in &frame_images {
            let Some(&camera_idx) = local_camera_idx_by_image.get(&query_image) else {
                continue;
            };
            let num_features = frames
                .get(query_image)
                .map(|frame| frame.keypoints.len())
                .unwrap_or(0);
            for feature in 0..num_features {
                let Ok(corrs) = graph.find_correspondences(query_image as u32, feature as u32)
                else {
                    continue;
                };
                for corr in corrs {
                    let other_image = corr.image_id as usize;
                    let other_feature = corr.point2d_idx as usize;
                    if reconstruction
                        .poses
                        .get(other_image)
                        .copied()
                        .flatten()
                        .is_none()
                    {
                        continue;
                    }
                    if camera_has_bogus_params(reconstruction.camera_for_image(other_image), config)
                    {
                        continue;
                    }
                    let Some(point_id) = reconstruction
                        .observations
                        .get(other_image)
                        .and_then(|obs| obs.get(other_feature))
                        .copied()
                        .flatten()
                    else {
                        continue;
                    };
                    if !seen.insert((query_image, feature, point_id)) {
                        continue;
                    }
                    let Some(kp) = frames
                        .get(query_image)
                        .and_then(|frame| frame.keypoints.get(feature))
                    else {
                        continue;
                    };
                    let Some(point) = reconstruction.points.get(point_id) else {
                        continue;
                    };
                    points2d.push([kp.x() as f64, kp.y() as f64]);
                    points3d.push([
                        point.xyz[0] as f64,
                        point.xyz[1] as f64,
                        point.xyz[2] as f64,
                    ]);
                    camera_idxs.push(camera_idx);
                    correspondences.push(GeneralizedFrameInlier {
                        image: query_image,
                        feature,
                        point_id,
                    });
                }
            }
        }
    } else {
        let frame_image_set = frame_images.iter().copied().collect::<HashSet<_>>();
        for pair in pairs {
            let (query_image, other_image, image_is_left) = match (
                frame_image_set.contains(&pair.left),
                frame_image_set.contains(&pair.right),
            ) {
                (true, false) => (pair.left, pair.right, true),
                (false, true) => (pair.right, pair.left, false),
                _ => continue,
            };
            if reconstruction
                .poses
                .get(other_image)
                .copied()
                .flatten()
                .is_none()
            {
                continue;
            }
            if camera_has_bogus_params(reconstruction.camera_for_image(other_image), config) {
                continue;
            }
            let Some(&camera_idx) = local_camera_idx_by_image.get(&query_image) else {
                continue;
            };

            for m in &pair.inlier_matches {
                let (feature, other_feature) = if image_is_left {
                    (m.query_idx as usize, m.train_idx as usize)
                } else {
                    (m.train_idx as usize, m.query_idx as usize)
                };
                let Some(point_id) = reconstruction
                    .observations
                    .get(other_image)
                    .and_then(|obs| obs.get(other_feature))
                    .copied()
                    .flatten()
                else {
                    continue;
                };
                if !seen.insert((query_image, feature, point_id)) {
                    continue;
                }
                let Some(kp) = frames
                    .get(query_image)
                    .and_then(|frame| frame.keypoints.get(feature))
                else {
                    continue;
                };
                let Some(point) = reconstruction.points.get(point_id) else {
                    continue;
                };
                points2d.push([kp.x() as f64, kp.y() as f64]);
                points3d.push([
                    point.xyz[0] as f64,
                    point.xyz[1] as f64,
                    point.xyz[2] as f64,
                ]);
                camera_idxs.push(camera_idx);
                correspondences.push(GeneralizedFrameInlier {
                    image: query_image,
                    feature,
                    point_id,
                });
            }
        }
    }

    Some(GeneralizedFrameAbsolutePoseProblem {
        points2d,
        points3d,
        camera_idxs,
        cams_from_rig,
        cameras,
        correspondences,
    })
}

fn frame_image_poses_from_rig_pose(
    reconstruction: &Reconstruction,
    frame_idx: usize,
    rig_from_world: SE3,
) -> Option<Vec<(usize, SE3)>> {
    let frame = reconstruction.frames.get(frame_idx)?;
    let image_poses = reconstruction
        .image_indices_for_frame_index(frame_idx)
        .into_iter()
        .filter_map(|frame_image| {
            let sensor_id = reconstruction.frame_sensor_id_for_image(frame_idx, frame_image)?;
            let sensor_from_rig = reconstruction
                .sensor_from_rig(frame.rig_id, sensor_id)
                .unwrap_or_else(SE3::identity);
            Some((frame_image, sensor_from_rig.compose(&rig_from_world)))
        })
        .collect::<Vec<_>>();
    (!image_poses.is_empty()).then_some(image_poses)
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
    graph: Option<&CorrespondenceGraph>,
    telemetry: &mut IncrementalRegistrationTelemetry,
) -> Option<AbsolutePose> {
    if config.experimental_structureless_pair_pose_fallback {
        if let Some(abs_pose) = solve_experimental_structureless_pair_pose_fallback(
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
    }
    solve_colmap_structureless_absolute_pose(
        image,
        frames,
        pairs,
        reconstruction,
        config,
        obs_manager,
        camera_priors,
        registration_stats,
        graph,
        telemetry,
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
    graph: Option<&CorrespondenceGraph>,
    telemetry: &mut IncrementalRegistrationTelemetry,
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
        collect_colmap_structureless_problem(image, frames, pairs, reconstruction, config, graph);
    if problem.world_points2d.len() < min_num_inliers {
        return None;
    }

    let mut options = StructureLessAbsolutePoseEstimationOptions::default();
    options.ransac_options.max_error = 0.5 * config.pnp_threshold_px as f64;
    options.ransac_options.min_inlier_ratio = config.abs_pose_min_inlier_ratio as f64;
    options.ransac_options.random_seed = config.random_seed;
    options.ransac_options.num_threads =
        config.threads.map(|threads| threads as isize).unwrap_or(1);

    let solve_started = Instant::now();
    let estimate_result = estimate_structureless_absolute_pose(
        &options,
        StructureLessAbsolutePoseProblem {
            query_points2d: &problem.query_points2d,
            world_points2d: &problem.world_points2d,
            world_camera_idxs: &problem.world_camera_idxs,
            world_cams_from_world: &problem.world_cams_from_world,
            world_cameras: &problem.world_cameras,
            query_camera: camera,
        },
    );
    telemetry.structureless_solver_ms += solve_started.elapsed().as_secs_f64() * 1_000.0;
    let estimate = match estimate_result {
        Ok(Some(estimate)) => {
            telemetry.structureless_estimates += 1;
            estimate
        }
        Ok(None) => return None,
        Err(err @ GeneralizedPoseError::MissingGeneralizedRelativePoseSolver) => {
            log::debug!("COLMAP structure-less registration skipped: {err}");
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
        point_inliers: Vec::new(),
        structureless_inliers,
        frame_image_poses: Vec::new(),
        generalized_inliers: Vec::new(),
    })
}

fn collect_colmap_structureless_problem(
    image: usize,
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
    graph: Option<&CorrespondenceGraph>,
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

    let mut append_correspondence =
        |other: usize,
         feature: usize,
         other_feature: usize,
         problem: &mut ColmapStructurelessProblem| {
            let Some(other_pose) = reconstruction.poses.get(other).copied().flatten() else {
                return;
            };
            let other_camera = reconstruction.camera_for_image(other);
            if camera_has_bogus_params(other_camera, config) {
                return;
            }
            let Some(query_kp) = frames
                .get(image)
                .and_then(|frame| frame.keypoints.get(feature))
            else {
                return;
            };
            let Some(world_kp) = frames
                .get(other)
                .and_then(|frame| frame.keypoints.get(other_feature))
            else {
                return;
            };
            let world_camera_idx = if let Some(&camera_idx) = world_image_to_camera_idx.get(&other)
            {
                camera_idx
            } else {
                let camera_idx = problem.world_cameras.len();
                world_image_to_camera_idx.insert(other, camera_idx);
                problem.world_cams_from_world.push(other_pose);
                problem.world_cameras.push(other_camera);
                camera_idx
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
        };

    if let Some(graph) = graph {
        let num_features = frames
            .get(image)
            .map(|frame| frame.keypoints.len())
            .unwrap_or(0);
        for feature in 0..num_features {
            let Ok(corrs) = graph.find_correspondences(image as u32, feature as u32) else {
                continue;
            };
            for corr in corrs {
                append_correspondence(
                    corr.image_id as usize,
                    feature,
                    corr.point2d_idx as usize,
                    &mut problem,
                );
            }
        }
    } else {
        for pair in pairs {
            let Some((other, image_is_left)) = structureless_pair_side(image, pair) else {
                continue;
            };
            for m in &pair.inlier_matches {
                let (feature, other_feature) = if image_is_left {
                    (m.query_idx as usize, m.train_idx as usize)
                } else {
                    (m.train_idx as usize, m.query_idx as usize)
                };
                append_correspondence(other, feature, other_feature, &mut problem);
            }
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
            point_inliers: Vec::new(),
            structureless_inliers,
            frame_image_poses: Vec::new(),
            generalized_inliers: Vec::new(),
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
    point_id: usize,
    xy: [f32; 2],
    xyz: [f32; 3],
}

fn collect_absolute_pose_observations(
    image: usize,
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
    graph: Option<&CorrespondenceGraph>,
) -> Vec<AbsolutePoseObservation> {
    if let Some(graph) = graph {
        collect_absolute_pose_observations_from_graph(image, frames, reconstruction, config, graph)
    } else {
        collect_absolute_pose_observations_from_pairs(image, frames, pairs, reconstruction, config)
    }
}

fn collect_absolute_pose_observations_from_graph(
    image: usize,
    frames: &[ImageFrame],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
    graph: &CorrespondenceGraph,
) -> Vec<AbsolutePoseObservation> {
    let mut pose_observations = Vec::new();
    let mut used_features = HashSet::new();
    let mut used_points = HashSet::new();
    let num_features = frames
        .get(image)
        .map(|frame| frame.keypoints.len())
        .unwrap_or(0);
    for feature in 0..num_features {
        let Ok(corrs) = graph.find_correspondences(image as u32, feature as u32) else {
            continue;
        };
        for corr in corrs {
            let other = corr.image_id as usize;
            let other_feature = corr.point2d_idx as usize;
            if reconstruction.poses.get(other).copied().flatten().is_none() {
                continue;
            }
            if camera_has_bogus_params(reconstruction.camera_for_image(other), config) {
                continue;
            }
            let Some(point_id) = reconstruction
                .observations
                .get(other)
                .and_then(|obs| obs.get(other_feature))
                .copied()
                .flatten()
            else {
                continue;
            };
            if !used_features.insert(feature) || !used_points.insert(point_id) {
                continue;
            }
            let kp = &frames[image].keypoints[feature];
            pose_observations.push(AbsolutePoseObservation {
                feature,
                point_id,
                xy: [kp.x(), kp.y()],
                xyz: reconstruction.points[point_id].xyz,
            });
            break;
        }
    }
    pose_observations
}

fn collect_absolute_pose_observations_from_pairs(
    image: usize,
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
) -> Vec<AbsolutePoseObservation> {
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
            pose_observations.push(AbsolutePoseObservation {
                feature,
                point_id,
                xy: [kp.x(), kp.y()],
                xyz: reconstruction.points[point_id].xyz,
            });
        }
    }
    pose_observations
}

#[cfg(test)]
fn solve_absolute_pose(
    image: usize,
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
    camera_priors: &[CameraModel],
    camera_has_prior_focal_length: &[bool],
    registration_stats: &RegistrationStats,
    graph: Option<&CorrespondenceGraph>,
) -> Option<AbsolutePose> {
    let mut telemetry = IncrementalRegistrationTelemetry::default();
    solve_absolute_pose_with_pnp_scorer(
        image,
        frames,
        pairs,
        reconstruction,
        config,
        camera_priors,
        camera_has_prior_focal_length,
        registration_stats,
        graph,
        None,
        &mut telemetry,
    )
    .expect("CPU absolute pose route is infallible")
}

#[allow(clippy::too_many_arguments)]
fn solve_absolute_pose_with_pnp_scorer(
    image: usize,
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &Reconstruction,
    config: &MapperConfig,
    camera_priors: &[CameraModel],
    camera_has_prior_focal_length: &[bool],
    registration_stats: &RegistrationStats,
    graph: Option<&CorrespondenceGraph>,
    pnp_scorer: Option<&mut DynPnPModelScorer>,
    telemetry: &mut IncrementalRegistrationTelemetry,
) -> Result<Option<AbsolutePose>> {
    let camera = registration_camera_for_image(
        image,
        reconstruction,
        config,
        camera_priors,
        registration_stats,
    );
    if camera_has_bogus_params(camera, config) {
        return Ok(None);
    }
    let collect_observations_start = Instant::now();
    let pose_observations =
        collect_absolute_pose_observations(image, frames, pairs, reconstruction, config, graph);
    telemetry.collect_observations_ms +=
        collect_observations_start.elapsed().as_secs_f64() * 1000.0;
    let pose_solve_refine_start = Instant::now();
    let result = (|| -> Result<Option<AbsolutePose>> {
        let num_correspondences = pose_observations.len();
        if num_correspondences < config.abs_pose_min_num_inliers.max(4) {
            return Ok(None);
        }
        let estimate_focal = absolute_pose_estimate_focal_length_enabled(
            image,
            camera,
            reconstruction,
            config,
            camera_has_prior_focal_length,
            registration_stats,
        );
        let Some((pose, inliers, camera)) =
            solve_absolute_pose_with_camera_hypotheses_and_pnp_scorer(
                &pose_observations,
                camera,
                estimate_focal,
                config,
                pnp_scorer,
                telemetry,
            )?
        else {
            return Ok(None);
        };
        let Some(initial_eval) =
            evaluate_absolute_pose(pose, &pose_observations, Some(&inliers), camera, config)
        else {
            return Ok(None);
        };
        if !accept_absolute_pose_eval(initial_eval, num_correspondences, config) {
            return Ok(None);
        }
        let refinement_observations =
            inlier_absolute_pose_observations(&pose_observations, &inliers);
        if refinement_observations.len() < config.abs_pose_min_num_inliers {
            return Ok(None);
        }
        let Some((pose, camera)) = refine_absolute_pose_reprojection(
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
        ) else {
            return Ok(None);
        };
        let Some(final_eval) =
            evaluate_absolute_pose(pose, &pose_observations, None, camera, config)
        else {
            return Ok(None);
        };
        if !accept_absolute_pose_eval(final_eval, num_correspondences, config) {
            return Ok(None);
        }
        let point_inliers = final_absolute_pose_point_inliers(
            pose,
            &pose_observations,
            camera,
            config.pnp_threshold_px,
        );
        debug_assert_eq!(point_inliers.len(), final_eval.inliers);
        Ok(Some(AbsolutePose {
            pose,
            camera,
            inliers: final_eval.inliers,
            inlier_ratio: final_eval.inliers as f32 / num_correspondences.max(1) as f32,
            mean_error_px: final_eval.mean_error_px,
            point_inliers,
            structureless_inliers: Vec::new(),
            frame_image_poses: Vec::new(),
            generalized_inliers: Vec::new(),
        }))
    })();
    telemetry.pose_solve_refine_ms += pose_solve_refine_start.elapsed().as_secs_f64() * 1000.0;
    result
}

fn final_absolute_pose_point_inliers(
    pose: SE3,
    observations: &[AbsolutePoseObservation],
    camera: CameraModel,
    threshold_px: f32,
) -> Vec<AbsolutePosePointInlier> {
    observations
        .iter()
        .filter_map(|observation| {
            let error = crate::geometry::reprojection_error_px(
                observation.xyz,
                pose,
                observation.xy,
                camera,
            );
            (error.is_finite() && error <= threshold_px).then_some(AbsolutePosePointInlier {
                feature: observation.feature,
                point_id: observation.point_id,
            })
        })
        .collect()
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

#[cfg(test)]
fn solve_absolute_pose_with_camera_hypotheses(
    observations: &[AbsolutePoseObservation],
    camera: CameraModel,
    estimate_focal: bool,
    config: &MapperConfig,
) -> Option<(SE3, Vec<bool>, CameraModel)> {
    let mut telemetry = IncrementalRegistrationTelemetry::default();
    solve_absolute_pose_with_camera_hypotheses_and_pnp_scorer(
        observations,
        camera,
        estimate_focal,
        config,
        None,
        &mut telemetry,
    )
    .expect("CPU absolute pose route is infallible")
}

fn solve_absolute_pose_with_camera_hypotheses_and_pnp_scorer(
    observations: &[AbsolutePoseObservation],
    camera: CameraModel,
    estimate_focal: bool,
    config: &MapperConfig,
    pnp_scorer: Option<&mut DynPnPModelScorer>,
    telemetry: &mut IncrementalRegistrationTelemetry,
) -> Result<Option<(SE3, Vec<bool>, CameraModel)>> {
    #[cfg(not(feature = "gpu-wgpu"))]
    let _ = telemetry;
    if estimate_focal {
        if config.use_gpu_pnp {
            #[cfg(feature = "gpu-wgpu")]
            {
                return solve_absolute_pose_with_gpu_focal_dispatch(
                    observations,
                    camera,
                    config,
                    telemetry,
                    solve_absolute_pose_with_gpu_focal_estimation,
                );
            }
        }
        return Ok(solve_absolute_pose_with_focal_estimation(
            observations,
            camera,
            config,
        ));
    }
    let mut best = None::<(AbsolutePoseEval, SE3, Vec<bool>, CameraModel)>;
    let Some((pose, inliers)) =
        solve_absolute_pose_for_camera_with_pnp_scorer(observations, camera, config, pnp_scorer)?
    else {
        return Ok(None);
    };
    let Some(eval) = evaluate_absolute_pose(pose, observations, Some(&inliers), camera, config)
    else {
        return Ok(None);
    };
    if accept_absolute_pose_eval(eval, observations.len(), config) {
        best = Some((eval, pose, inliers, camera));
    }
    Ok(best.map(|(_, pose, inliers, camera)| (pose, inliers, camera)))
}

#[cfg(feature = "gpu-wgpu")]
fn fallback_to_cpu_focal_estimation(
    observations: &[AbsolutePoseObservation],
    camera: CameraModel,
    config: &MapperConfig,
    telemetry: &mut IncrementalRegistrationTelemetry,
    reason: impl std::fmt::Display,
    gpu_error: Option<anyhow::Error>,
) -> Result<Option<(SE3, Vec<bool>, CameraModel)>> {
    telemetry.record_gpu_pnp_focal_fallback(reason);
    match solve_absolute_pose_with_focal_estimation(observations, camera, config) {
        Some(result) => Ok(Some(result)),
        None if gpu_error.is_none() => Ok(None),
        None => Err(anyhow::Error::new(GpuPnPFocalFallbackError { gpu_error })),
    }
}

#[cfg(feature = "gpu-wgpu")]
fn is_task_cancellation_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<crate::task::SfmTaskStop>()
            .is_some_and(|stop| matches!(stop, crate::task::SfmTaskStop::Cancelled))
    })
}

#[cfg(feature = "gpu-wgpu")]
fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = panic.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = panic.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

#[cfg(feature = "gpu-wgpu")]
fn solve_absolute_pose_with_gpu_focal_dispatch<F>(
    observations: &[AbsolutePoseObservation],
    camera: CameraModel,
    config: &MapperConfig,
    telemetry: &mut IncrementalRegistrationTelemetry,
    dispatch: F,
) -> Result<Option<(SE3, Vec<bool>, CameraModel)>>
where
    F: FnOnce(
        &[AbsolutePoseObservation],
        CameraModel,
        &MapperConfig,
    ) -> Result<Option<(SE3, Vec<bool>, CameraModel)>>,
{
    let gpu_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        dispatch(observations, camera, config)
    }));
    match gpu_result {
        Ok(Ok(Some(result))) => Ok(Some(result)),
        Ok(Ok(None)) => fallback_to_cpu_focal_estimation(
            observations,
            camera,
            config,
            telemetry,
            "gpu pnp-focal returned no valid solution; using CPU PnP-f",
            None,
        ),
        Ok(Err(error)) if is_task_cancellation_error(&error) => Err(error),
        Ok(Err(error)) => fallback_to_cpu_focal_estimation(
            observations,
            camera,
            config,
            telemetry,
            format!("gpu pnp-focal failed: {error:#}; using CPU PnP-f"),
            Some(error),
        ),
        Err(panic) => {
            let panic = panic_message(panic);
            fallback_to_cpu_focal_estimation(
                observations,
                camera,
                config,
                telemetry,
                format!("gpu pnp-focal panicked: {panic}; using CPU PnP-f"),
                Some(gpu_pnp_mapper_error(format!(
                    "gpu pnp-focal panicked: {panic}"
                ))),
            )
        }
    }
}

#[cfg(feature = "gpu-wgpu")]
fn solve_absolute_pose_with_gpu_focal_estimation(
    observations: &[AbsolutePoseObservation],
    initial_camera: CameraModel,
    config: &MapperConfig,
) -> Result<Option<(SE3, Vec<bool>, CameraModel)>> {
    let Some(focal_idxs) = colmap_camera_model_focal_idxs(initial_camera.model_id) else {
        return Ok(None);
    };
    if focal_idxs.is_empty() {
        return Ok(None);
    }
    let max_dimension = initial_camera.width.max(initial_camera.height).max(1) as f32;
    let min_focal = max_dimension * config.min_focal_length_ratio as f32;
    let max_focal = max_dimension * config.max_focal_length_ratio as f32;
    let centered_points = observations
        .iter()
        .map(|observation| {
            [
                observation.xy[0] - initial_camera.cx,
                observation.xy[1] - initial_camera.cy,
            ]
        })
        .collect::<Vec<_>>();
    let object_points = observations
        .iter()
        .map(|observation| observation.xyz)
        .collect::<Vec<_>>();
    let context = crate::gpu::WgpuContext::try_new()?;
    let solver = crate::gpu::WgpuPnPFocalSolver::from_context(context)?;
    let Some(solution) = solver.solve(
        &centered_points,
        &object_points,
        config.pnp_threshold_px,
        absolute_pose_ransac_seed(config) as u32,
        config.pnp_iterations as usize,
        min_focal,
        max_focal,
    )?
    else {
        return Ok(None);
    };
    if solution.inlier_mask.len() != observations.len()
        || solution.inliers
            != solution
                .inlier_mask
                .iter()
                .filter(|&&inlier| inlier)
                .count()
    {
        bail!("gpu pnp-focal returned an invalid inlier mask");
    }
    let mut camera = initial_camera;
    for &idx in focal_idxs {
        if idx >= camera.num_params {
            bail!("gpu pnp-focal focal parameter index {idx} is out of range");
        }
        camera.params[idx] = solution.focal as f64;
    }
    camera.sync_intrinsics_from_params();
    if camera_has_bogus_params(camera, config) {
        bail!("gpu pnp-focal returned invalid camera parameters");
    }
    Ok(Some((solution.pose, solution.inlier_mask, camera)))
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
    solve_absolute_pose_for_camera_with_pnp_scorer(observations, camera, config, None)
        .expect("CPU PnP scorer is infallible")
}

fn solve_absolute_pose_for_camera_with_pnp_scorer(
    observations: &[AbsolutePoseObservation],
    camera: CameraModel,
    config: &MapperConfig,
    pnp_scorer: Option<&mut DynPnPModelScorer>,
) -> Result<Option<(SE3, Vec<bool>)>> {
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
        let Some(norm_xy) = camera.cam_from_img_f32(observation.xy[0], observation.xy[1]) else {
            return Ok(None);
        };
        problem.add_correspondence(norm_xy, observation.xyz);
    }
    if let Some(scorer) = pnp_scorer {
        solver
            .solve_with_model_scorer(&problem, scorer)
            .map_err(|error| {
                gpu_pnp_mapper_error(format!("gpu pnp absolute pose scoring failed: {error:#}"))
            })
    } else {
        Ok(solver.solve(&problem))
    }
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
    ABSOLUTE_POSE_RANSAC_SEED.fetch_add(1, AtomicOrdering::Relaxed)
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
    observation_manager: &mut ObservationManager,
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
                    color: [0, 0, 0],
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
        observation_manager.add_point3d(
            frames,
            pairs,
            reconstruction,
            Point3D {
                xyz,
                color: [0, 0, 0],
                error,
                track: observations,
            },
        );
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct TrackFilterReport {
    removed_observations: usize,
    examined_points: usize,
    examined_observations: usize,
}

#[derive(Debug, Clone, Copy)]
enum TrackFilterScope<'a> {
    Full,
    Subset(&'a HashSet<usize>),
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
        None,
        track_filter_max_error_px(config),
        track_filter_min_tri_angle_deg(config),
        track_filter_min_track_length(),
    )
}

fn filter_reprojection_tracks_with_state(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    config: &MapperConfig,
    triangulation_state: &mut IncrementalTriangulatorState,
) -> usize {
    let max_error = track_filter_max_error_px(config);
    let min_tri_angle = track_filter_min_tri_angle_deg(config);
    let min_track_length = track_filter_min_track_length();
    let removed = filter_reprojection_tracks_with_policy(
        frames,
        pairs,
        reconstruction,
        config,
        Some(triangulation_state.observation_manager_mut()),
        max_error,
        min_tri_angle,
        min_track_length,
    );
    let _ = triangulation_state
        .observation_manager_mut()
        .take_modified_point3d_ids();
    removed
}

fn filter_reprojection_tracks_subset_with_state(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    config: &MapperConfig,
    triangulation_state: &mut IncrementalTriangulatorState,
    point_ids: &HashSet<usize>,
) -> usize {
    filter_reprojection_tracks_with_policy_in_scope(
        frames,
        pairs,
        reconstruction,
        config,
        Some(triangulation_state.observation_manager_mut()),
        TrackFilterScope::Subset(point_ids),
        track_filter_max_error_px(config),
        track_filter_min_tri_angle_deg(config),
        track_filter_min_track_length(),
    )
}

fn filter_modified_reprojection_tracks_with_state(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    config: &MapperConfig,
    triangulation_state: &mut IncrementalTriangulatorState,
) -> usize {
    let point_ids = triangulation_state
        .observation_manager()
        .modified_point3d_ids()
        .clone();
    let removed = filter_reprojection_tracks_subset_with_state(
        frames,
        pairs,
        reconstruction,
        config,
        triangulation_state,
        &point_ids,
    );
    let _ = triangulation_state
        .observation_manager_mut()
        .take_modified_point3d_ids();
    removed
}

fn filter_reprojection_tracks_with_policy(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    config: &MapperConfig,
    observation_manager: Option<&mut ObservationManager>,
    max_error: f32,
    min_tri_angle: f32,
    min_track_length: usize,
) -> usize {
    filter_reprojection_tracks_with_policy_in_scope(
        frames,
        pairs,
        reconstruction,
        config,
        observation_manager,
        TrackFilterScope::Full,
        max_error,
        min_tri_angle,
        min_track_length,
    )
}

fn filter_reprojection_tracks_with_policy_in_scope(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    config: &MapperConfig,
    observation_manager: Option<&mut ObservationManager>,
    scope: TrackFilterScope<'_>,
    max_error: f32,
    min_tri_angle: f32,
    min_track_length: usize,
) -> usize {
    let started = Instant::now();
    let image_cameras = (0..reconstruction.poses.len())
        .map(|image| reconstruction.camera_for_image(image))
        .collect::<Vec<_>>();
    let mut temporary_manager;
    let observation_manager = if let Some(observation_manager) = observation_manager {
        observation_manager
    } else {
        temporary_manager = ObservationManager::new(frames, pairs, reconstruction);
        &mut temporary_manager
    };
    let mut report = TrackFilterReport::default();
    match scope {
        TrackFilterScope::Full => {
            let mut point_id = 0usize;
            while point_id < reconstruction.points.len() {
                let (removed, examined, deleted) = filter_reprojection_point(
                    frames,
                    pairs,
                    reconstruction,
                    config,
                    observation_manager,
                    &image_cameras,
                    point_id,
                    max_error,
                    min_tri_angle,
                    min_track_length,
                );
                report.removed_observations += removed;
                report.examined_points += 1;
                report.examined_observations += examined;
                if !deleted {
                    point_id += 1;
                }
            }
        }
        TrackFilterScope::Subset(point_ids) => {
            let mut point_ids = point_ids
                .iter()
                .copied()
                .filter(|&point_id| point_id < reconstruction.points.len())
                .collect::<Vec<_>>();
            point_ids.sort_unstable_by(|left, right| right.cmp(left));
            for point_id in point_ids {
                if point_id >= reconstruction.points.len() {
                    continue;
                }
                let (removed, examined, _) = filter_reprojection_point(
                    frames,
                    pairs,
                    reconstruction,
                    config,
                    observation_manager,
                    &image_cameras,
                    point_id,
                    max_error,
                    min_tri_angle,
                    min_track_length,
                );
                report.removed_observations += removed;
                report.examined_points += 1;
                report.examined_observations += examined;
            }
        }
    }
    observation_manager.record_filter(
        matches!(scope, TrackFilterScope::Subset(_)),
        report.examined_points,
        report.examined_observations,
        report.removed_observations,
        started.elapsed().as_secs_f64() * 1_000.0,
    );
    report.removed_observations
}

fn filter_reprojection_point(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    config: &MapperConfig,
    observation_manager: &mut ObservationManager,
    image_cameras: &[CameraModel],
    point_id: usize,
    max_error: f32,
    min_tri_angle: f32,
    min_track_length: usize,
) -> (usize, usize, bool) {
    if point_id >= reconstruction.points.len() {
        return (0, 0, false);
    }
    let point_xyz = reconstruction.points[point_id].xyz;
    let track = reconstruction.points[point_id].track.clone();
    let examined = track.len();
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
            let err =
                crate::geometry::reprojection_error_px(point_xyz, pose, [kp.x(), kp.y()], camera);
            !err.is_finite() || err > max_error
        })
        .cloned()
        .collect::<Vec<_>>();

    if observations_to_delete.len() >= track.len().saturating_sub(1) {
        let removed = track.len();
        let deleted = observation_manager.delete_point3d(frames, pairs, reconstruction, point_id);
        return (removed, examined, deleted);
    }

    let mut removed = 0usize;
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
        return (removed, examined, true);
    }
    let track = reconstruction.points[point_id].track.clone();
    if track.len() < min_track_length {
        removed += track.len();
        let deleted = observation_manager.delete_point3d(frames, pairs, reconstruction, point_id);
        return (removed, examined, deleted);
    }
    if !track_has_min_triangulation_angle(
        reconstruction.points[point_id].xyz,
        &track,
        reconstruction,
        min_tri_angle,
    ) {
        removed += track.len();
        let deleted = observation_manager.delete_point3d(frames, pairs, reconstruction, point_id);
        return (removed, examined, deleted);
    }
    if let Some(error) = mean_track_reprojection_error(
        reconstruction.points[point_id].xyz,
        &track,
        frames,
        reconstruction,
    ) {
        reconstruction.points[point_id].error = error;
        (removed, examined, false)
    } else {
        removed += track.len();
        let deleted = observation_manager.delete_point3d(frames, pairs, reconstruction, point_id);
        (removed, examined, deleted)
    }
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
        read_colmap_sparse_model, write_colmap_sparse_text, ColmapCamera, ColmapDataId,
        ColmapFrame, ColmapImage, ColmapPoint2D, ColmapPoint3D, ColmapRig, ColmapRigSensor,
        ColmapSensorId, ColmapSensorType, ColmapSparseFiles, ColmapTrackElement,
    };
    use crate::correspondence_graph::FeatureMatch;
    use crate::database::{
        ColmapDatabase, ColmapDatabaseCamera, ColmapDatabaseFrame, ColmapDatabaseImage,
        ColmapKeypoint, ColmapPosePrior, ColmapPosePriorCoordinateSystem, ColmapTwoViewGeometry,
        DatabaseCacheOptions,
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
    fn mapper_gpu_pnp_is_disabled_by_default() {
        assert!(!MapperConfig::default().use_gpu_pnp);
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn mapper_gpu_pnp_unsupported_routes_fall_back_to_cpu_with_telemetry() {
        let config = MapperConfig {
            use_gpu_pnp: true,
            ..MapperConfig::default()
        };
        let mut telemetry = IncrementalRegistrationTelemetry::default();
        record_gpu_pnp_route_fallback(&config, &mut telemetry, "generalized rig route");
        record_gpu_pnp_route_fallback(&config, &mut telemetry, "structureless route");
        record_gpu_pnp_route_fallback(&config, &mut telemetry, "structureless route");
        assert_eq!(telemetry.gpu_pnp_focal_fallbacks.len(), 2);
        assert!(telemetry.gpu_pnp_focal_fallbacks[0].contains("generalized"));
        assert!(telemetry.gpu_pnp_focal_fallbacks[1].contains("structureless"));

        let mut disabled_telemetry = IncrementalRegistrationTelemetry::default();
        record_gpu_pnp_route_fallback(
            &MapperConfig::default(),
            &mut disabled_telemetry,
            "structureless route",
        );
        assert!(disabled_telemetry.gpu_pnp_focal_fallbacks.is_empty());
    }

    #[test]
    fn mapper_gpu_pnp_rejects_global_mapper() {
        let config = MapperConfig {
            use_gpu_pnp: true,
            global_mapper: true,
            ..MapperConfig::default()
        };
        let error = validate_gpu_pnp_config(&config, true).expect_err("global route must reject");
        assert!(error.to_string().contains("global mapper"));
    }

    #[cfg(not(feature = "gpu-wgpu"))]
    #[test]
    fn mapper_gpu_pnp_requires_compiled_backend() {
        let config = MapperConfig {
            use_gpu_pnp: true,
            ..MapperConfig::default()
        };
        let error =
            validate_gpu_pnp_config(&config, false).expect_err("missing backend must reject");
        assert!(error.to_string().contains("gpu-wgpu"));
    }

    struct FailingMapperPnpScorer;

    impl PnPModelScorer for FailingMapperPnpScorer {
        type Error = anyhow::Error;

        fn prepare(
            &mut self,
            _normalized_points: &[[f32; 2]],
            _object_points: &[[f32; 3]],
            _threshold: f32,
        ) -> Result<(), Self::Error> {
            bail!("simulated device loss")
        }

        fn score_models(
            &mut self,
            _models: &[SE3],
        ) -> Result<Vec<rustslam::tracker::PnPModelSupport>, Self::Error> {
            unreachable!("prepare must fail")
        }

        fn inlier_mask(&mut self, _model: &SE3) -> Result<Vec<bool>, Self::Error> {
            unreachable!("prepare must fail")
        }
    }

    #[test]
    fn mapper_gpu_pnp_scorer_errors_keep_gpu_marker() {
        let camera = CameraModel::new_pinhole(640, 480, 700.0, 700.0, 320.0, 240.0);
        let observations = [
            AbsolutePoseObservation {
                feature: 0,
                point_id: 0,
                xy: [320.0, 240.0],
                xyz: [0.0, 0.0, 4.0],
            },
            AbsolutePoseObservation {
                feature: 1,
                point_id: 1,
                xy: [390.0, 240.0],
                xyz: [0.4, 0.0, 4.0],
            },
            AbsolutePoseObservation {
                feature: 2,
                point_id: 2,
                xy: [320.0, 310.0],
                xyz: [0.0, 0.4, 4.0],
            },
            AbsolutePoseObservation {
                feature: 3,
                point_id: 3,
                xy: [376.0, 296.0],
                xyz: [0.4, 0.4, 5.0],
            },
        ];
        let mut scorer = FailingMapperPnpScorer;
        let error = solve_absolute_pose_for_camera_with_pnp_scorer(
            &observations,
            camera,
            &MapperConfig::default(),
            Some(&mut scorer),
        )
        .expect_err("GPU scorer error must propagate");

        assert!(error.to_string().contains("simulated device loss"));
        assert!(error.downcast_ref::<GpuPnpMapperError>().is_some());
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn mapper_gpu_pnp_focal_dispatch_failure_falls_back_to_cpu() -> Result<()> {
        let camera = CameraModel::new_pinhole(640, 480, 450.0, 450.0, 320.0, 240.0);
        let expected_pose = SE3::from_axis_angle(&[0.015, -0.01, 0.02], &[0.3, -0.12, 0.55]);
        let expected_focal = 700.0;
        let observations = (0..32)
            .map(|index| {
                let xyz = [
                    ((index % 8) as f32 - 3.5) * 0.22,
                    ((index / 8) as f32 - 1.5) * 0.27,
                    4.2 + (index % 5) as f32 * 0.3,
                ];
                let point = expected_pose.transform_point(&xyz);
                AbsolutePoseObservation {
                    feature: index,
                    point_id: index,
                    xy: [
                        expected_focal * point[0] / point[2] + camera.cx,
                        expected_focal * point[1] / point[2] + camera.cy,
                    ],
                    xyz,
                }
            })
            .collect::<Vec<_>>();
        let config = MapperConfig {
            use_gpu_pnp: true,
            ba_refine_focal_length: true,
            pnp_threshold_px: 2.0,
            pnp_iterations: 512,
            abs_pose_min_num_inliers: 4,
            abs_pose_min_inlier_ratio: 0.0,
            random_seed: 7,
            ..MapperConfig::default()
        };
        let cpu = solve_absolute_pose_with_focal_estimation(&observations, camera, &config)
            .expect("CPU unknown-focal PnP");
        let mut telemetry = IncrementalRegistrationTelemetry::default();
        let gpu = solve_absolute_pose_with_gpu_focal_dispatch(
            &observations,
            camera,
            &config,
            &mut telemetry,
            |_observations, _camera, _config| bail!("simulated focal GPU dispatch failure"),
        )?
        .expect("CPU PnP-fallback result");

        assert!(gpu.1.iter().filter(|&&inlier| inlier).count() >= 24);
        assert!((gpu.2.fx - expected_focal).abs() / expected_focal < 0.05);
        assert!((gpu.2.fx - cpu.2.fx).abs() / cpu.2.fx < 0.05);
        assert!(!camera_has_bogus_params(gpu.2, &config));
        assert_eq!(telemetry.gpu_pnp_focal_fallbacks.len(), 1);
        assert!(telemetry.gpu_pnp_focal_fallbacks[0].contains("simulated focal GPU dispatch"));
        assert!(telemetry
            .format_log()
            .contains("gpu_pnp_focal_fallback=gpu pnp-focal"));
        Ok(())
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn mapper_gpu_pnp_focal_fallback_reports_cpu_unsolved() -> Result<()> {
        let camera = CameraModel::new_pinhole(640, 480, 450.0, 450.0, 320.0, 240.0);
        let config = MapperConfig::default();
        let mut telemetry = IncrementalRegistrationTelemetry::default();

        let gpu_error = anyhow::Error::msg("simulated GPU device loss")
            .context("simulated GPU dispatch failure");
        let error = fallback_to_cpu_focal_estimation(
            &[],
            camera,
            &config,
            &mut telemetry,
            "simulated focal GPU dispatch failure",
            Some(gpu_error),
        )
        .expect_err("an unsolved CPU PnP-focal fallback must be reported");

        assert!(error
            .to_string()
            .contains("CPU PnP-focal fallback could not solve"));
        assert!(error.to_string().contains("simulated GPU dispatch failure"));
        let fallback = error
            .downcast_ref::<GpuPnPFocalFallbackError>()
            .expect("CPU fallback error type");
        let source = std::error::Error::source(fallback).expect("GPU error source");
        assert!(source
            .to_string()
            .contains("simulated GPU dispatch failure"));
        assert!(source
            .source()
            .expect("GPU error root cause")
            .to_string()
            .contains("simulated GPU device loss"));
        assert_eq!(telemetry.gpu_pnp_focal_fallbacks.len(), 1);
        Ok(())
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn mapper_gpu_pnp_focal_no_solution_preserves_cpu_none() -> Result<()> {
        let camera = CameraModel::new_pinhole(640, 480, 450.0, 450.0, 320.0, 240.0);
        let config = MapperConfig::default();
        let mut telemetry = IncrementalRegistrationTelemetry::default();

        let result = fallback_to_cpu_focal_estimation(
            &[],
            camera,
            &config,
            &mut telemetry,
            "gpu pnp-focal returned no valid solution; using CPU PnP-f",
            None,
        )?;

        assert!(result.is_none());
        assert_eq!(telemetry.gpu_pnp_focal_fallbacks.len(), 1);
        Ok(())
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn mapper_gpu_pnp_focal_real_solver_maps_focal_camera_and_mask() -> Result<()> {
        let Some(_) = crate::gpu::WgpuContext::try_new_optional()? else {
            eprintln!("skipping mapper GPU PnP-focal test: no compatible adapter");
            return Ok(());
        };
        let camera = CameraModel::new_pinhole(640, 480, 450.0, 450.0, 320.0, 240.0);
        let expected_pose = SE3::from_axis_angle(&[0.015, -0.01, 0.02], &[0.3, -0.12, 0.55]);
        let expected_focal = 700.0;
        let observations = (0..32)
            .map(|index| {
                let xyz = [
                    ((index % 8) as f32 - 3.5) * 0.22,
                    ((index / 8) as f32 - 1.5) * 0.27,
                    4.2 + (index % 5) as f32 * 0.3,
                ];
                let point = expected_pose.transform_point(&xyz);
                AbsolutePoseObservation {
                    feature: index,
                    point_id: index,
                    xy: [
                        expected_focal * point[0] / point[2] + camera.cx,
                        expected_focal * point[1] / point[2] + camera.cy,
                    ],
                    xyz,
                }
            })
            .collect::<Vec<_>>();
        let config = MapperConfig {
            use_gpu_pnp: true,
            ba_refine_focal_length: true,
            pnp_threshold_px: 2.0,
            pnp_iterations: 512,
            abs_pose_min_num_inliers: 4,
            abs_pose_min_inlier_ratio: 0.0,
            random_seed: 7,
            ..MapperConfig::default()
        };

        let solved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            solve_absolute_pose_with_gpu_focal_estimation(&observations, camera, &config)
        }));
        let (_, mask, solved_camera) = match solved {
            Ok(Ok(Some(solution))) => solution,
            Ok(Ok(None)) => {
                eprintln!("skipping mapper GPU PnP-focal test: adapter returned no solution");
                return Ok(());
            }
            Ok(Err(error)) => {
                if crate::gpu::is_known_macos_agx_pipeline_failure(&error) {
                    eprintln!("skipping mapper GPU PnP-focal test: {error:#}");
                    return Ok(());
                }
                return Err(error);
            }
            Err(panic) => {
                let message = panic_message(panic);
                let error = anyhow::anyhow!(message.clone());
                if crate::gpu::is_known_macos_agx_pipeline_failure(&error) {
                    eprintln!("skipping mapper GPU PnP-focal test: {message}");
                    return Ok(());
                }
                std::panic::panic_any(message);
            }
        };

        assert_eq!(mask.len(), observations.len());
        assert!(mask.iter().all(|&inlier| inlier));
        assert!((solved_camera.fx - expected_focal).abs() / expected_focal < 0.05);
        assert_eq!(solved_camera.params[0], solved_camera.fx as f64);
        assert_eq!(solved_camera.params[1], solved_camera.fy as f64);
        assert!(!camera_has_bogus_params(solved_camera, &config));
        Ok(())
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn mapper_gpu_pnp_matches_known_intrinsics_cpu_mask() -> Result<()> {
        let Some(context) = crate::gpu::WgpuContext::try_new_optional()? else {
            eprintln!("skipping mapper GPU PnP test: no compatible adapter");
            return Ok(());
        };
        let camera = CameraModel::new_pinhole(640, 480, 700.0, 700.0, 320.0, 240.0);
        let expected_pose = SE3::from_axis_angle(&[0.015, -0.01, 0.02], &[0.3, -0.12, 0.55]);
        let observations = (0..32)
            .map(|index| {
                let xyz = [
                    ((index % 8) as f32 - 3.5) * 0.22,
                    ((index / 8) as f32 - 1.5) * 0.27,
                    4.2 + (index % 5) as f32 * 0.3,
                ];
                let point = expected_pose.transform_point(&xyz);
                let mut xy = [
                    camera.fx * point[0] / point[2] + camera.cx,
                    camera.fy * point[1] / point[2] + camera.cy,
                ];
                if index >= 26 {
                    xy[0] += 80.0;
                    xy[1] -= 60.0;
                }
                AbsolutePoseObservation {
                    feature: index,
                    point_id: index,
                    xy,
                    xyz,
                }
            })
            .collect::<Vec<_>>();
        let config = MapperConfig {
            use_gpu_pnp: true,
            pnp_threshold_px: 2.0,
            pnp_iterations: 512,
            abs_pose_min_num_inliers: 4,
            abs_pose_min_inlier_ratio: 0.0,
            random_seed: 7,
            ..MapperConfig::default()
        };
        let cpu = solve_absolute_pose_for_camera(&observations, camera, &config)
            .expect("CPU known-intrinsics PnP");
        let mut scorer = crate::gpu::WgpuPnpModelScorer::from_context(context)?;
        let mut telemetry = IncrementalRegistrationTelemetry::default();
        let gpu = solve_absolute_pose_with_camera_hypotheses_and_pnp_scorer(
            &observations,
            camera,
            false,
            &config,
            Some(&mut scorer),
            &mut telemetry,
        )?
        .expect("GPU known-intrinsics PnP");
        let cpu_inliers = cpu.1.iter().filter(|&&value| value).count();
        let gpu_inliers = gpu.1.iter().filter(|&&value| value).count();
        assert_eq!(gpu.1, cpu.1);
        assert!(gpu_inliers * 10 >= cpu_inliers * 9);
        Ok(())
    }

    fn assert_vec3_near(actual: glam::Vec3, expected: glam::Vec3, tolerance: f32) {
        assert!(
            actual.abs_diff_eq(expected, tolerance),
            "expected {expected:?}, got {actual:?}"
        );
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
    fn reference_camera_setup_propagates_malformed_sparse_model() -> Result<()> {
        let dir = tempdir()?;
        let sparse = dir.path().join("sparse/0");
        fs::create_dir_all(&sparse)?;
        fs::write(
            sparse.join("cameras.txt"),
            "1 PINHOLE 640 480 500 500 320 240\n",
        )?;
        fs::write(sparse.join("images.txt"), "1 1 0 0 0 0 0 0 1 a.jpg\n\n")?;
        fs::write(sparse.join("points3D.txt"), "1 malformed\n")?;

        assert!(reference_camera_setup(dir.path(), &[PathBuf::from("a.jpg")]).is_err());
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
    fn reference_camera_setup_seeds_existing_reconstruction_like_colmap() -> Result<()> {
        let dir = tempdir()?;
        let sparse = dir.path().join("sparse/0");
        fs::create_dir_all(&sparse)?;
        fs::write(
            sparse.join("cameras.txt"),
            "1 PINHOLE 100 100 50 50 50 50\n",
        )?;
        fs::write(
            sparse.join("images.txt"),
            concat!(
                "1 1 0 0 0 0 0 0 1 a.jpg\n",
                "50 50 7\n",
                "2 1 0 0 0 -1 0 0 1 b.jpg\n",
                "75 50 7\n"
            ),
        )?;
        fs::write(
            sparse.join("points3D.txt"),
            "7 0 0 2 12 34 56 0.5 1 0 2 0\n",
        )?;
        let setup = reference_camera_setup(
            dir.path(),
            &[
                PathBuf::from("a.jpg"),
                PathBuf::from("b.jpg"),
                PathBuf::from("c.jpg"),
            ],
        )?;
        let seed = setup.seed_reconstruction.expect("seed reconstruction");

        assert_eq!(seed.poses.iter().filter(|pose| pose.is_some()).count(), 2);
        assert_eq!(seed.point_ids, vec![7]);
        assert_eq!(seed.observations[0][0], Some(0));
        assert_eq!(seed.observations[1][0], Some(0));
        assert_eq!(seed.points[0].track.len(), 2);
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
    fn colmap_output_order_prefers_largest_reconstruction() {
        let frames = (0..4)
            .map(|idx| minimal_frame(idx, &format!("image_{idx}.jpg")))
            .collect::<Vec<_>>();
        let mut small = test_reconstruction(&frames);
        small.poses[0] = Some(SE3::identity());
        small.poses[1] = Some(SE3::identity());
        small.points = vec![
            Point3D {
                xyz: [0.0, 0.0, 1.0],
                color: [0, 0, 0],
                error: 0.0,
                track: Vec::new(),
            };
            20
        ];
        let mut large = test_reconstruction(&frames);
        large.poses[0] = Some(SE3::identity());
        large.poses[1] = Some(SE3::identity());
        large.poses[2] = Some(SE3::identity());
        large.points = vec![Point3D {
            xyz: [0.0, 0.0, 1.0],
            color: [0, 0, 0],
            error: 0.0,
            track: Vec::new(),
        }];

        let mut reconstructions = vec![small, large];

        sort_reconstructions_for_colmap_output(&mut reconstructions);

        assert_eq!(registered_image_count(&reconstructions[0]), 3);
        assert_eq!(registered_image_count(&reconstructions[1]), 2);
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
    fn database_frame_fast_path_does_not_extract_descriptors_or_colors() -> Result<()> {
        let dir = tempdir()?;
        let db_path = dir.path().join("database.db");
        let paths = [dir.path().join("left.png"), dir.path().join("right.png")];
        for path in &paths {
            image::RgbImage::from_pixel(16, 12, image::Rgb([12, 34, 56])).save(path)?;
        }

        let db = ColmapDatabase::open(&db_path)?;
        db.write_camera(
            &ColmapDatabaseCamera {
                camera: crate::colmap::ColmapCamera {
                    camera_id: 5,
                    model_id: crate::types::COLMAP_PINHOLE,
                    width: 16,
                    height: 12,
                    params: vec![10.0, 10.0, 8.0, 6.0],
                },
                has_prior_focal_length: true,
            },
            true,
        )?;
        for (image_id, name) in [(11, "left.png"), (12, "right.png")] {
            db.write_image(
                &ColmapDatabaseImage {
                    image_id,
                    name: name.to_string(),
                    camera_id: 5,
                    frame_id: None,
                },
                true,
            )?;
            db.write_keypoints(
                image_id,
                &[
                    ColmapKeypoint::new(1.0, 2.0),
                    ColmapKeypoint::new(3.0, 4.0),
                    ColmapKeypoint::new(5.0, 6.0),
                    ColmapKeypoint::new(7.0, 8.0),
                ],
            )?;
        }
        db.write_two_view_geometry(
            11,
            12,
            &ColmapTwoViewGeometry {
                config: 2,
                inlier_matches: (0..4)
                    .map(|index| FeatureMatch::new(index, index))
                    .collect(),
                ..ColmapTwoViewGeometry::default()
            },
        )?;
        let lookup_frames = paths
            .iter()
            .enumerate()
            .map(|(id, path)| {
                let mut frame = minimal_frame(id, path.file_name().unwrap().to_str().unwrap());
                frame.path = path.clone();
                frame
            })
            .collect::<Vec<_>>();
        let database =
            load_mapper_database(Some(&db_path), &lookup_frames, 0)?.expect("database input");

        let mut frames = reconstruction_input::database_frames(&paths, &database)?;
        apply_color_extraction_policy(&mut frames, true);

        assert_eq!(frames.len(), 2);
        assert_eq!((frames[0].width, frames[0].height), (16, 12));
        assert_eq!(frames[0].keypoints.len(), 4);
        assert_eq!(frames[0].keypoints[2].x(), 5.0);
        assert!(frames[0].descriptors.is_empty());
        assert!(frames[0].sift.descriptors_u8.is_empty());
        assert!(frames[0].wide_descriptors.data.is_empty());
        assert!(frames[0].strong_feature_indices.is_empty());
        assert!(frames[0].colors.is_empty());
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
    fn disabling_database_discovery_ignores_input_and_parent_databases() -> Result<()> {
        let dir = tempdir()?;
        let images_dir = dir.path().join("images");
        fs::create_dir_all(&images_dir)?;
        fs::write(dir.path().join("database.db"), [])?;
        fs::write(images_dir.join("database.db"), [])?;

        let resolved = resolve_mapper_database_path(&MapperConfig {
            input: images_dir,
            discover_database: false,
            ..MapperConfig::default()
        })?;

        assert_eq!(resolved, None);
        Ok(())
    }

    #[test]
    fn explicit_database_is_used_when_discovery_is_disabled() -> Result<()> {
        let dir = tempdir()?;
        let images_dir = dir.path().join("images");
        fs::create_dir_all(&images_dir)?;
        fs::write(images_dir.join("database.db"), [])?;
        let explicit = dir.path().join("explicit.db");
        fs::write(&explicit, [])?;

        let resolved = resolve_mapper_database_path(&MapperConfig {
            input: images_dir,
            database: Some(explicit.clone()),
            discover_database: false,
            ..MapperConfig::default()
        })?;

        assert_eq!(resolved.as_deref(), Some(explicit.as_path()));
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
    fn registration_retry_state_suppresses_unchanged_support_until_fallback() {
        let frames = vec![
            minimal_frame(0, "registered.jpg"),
            minimal_frame(1, "candidate.jpg"),
        ];
        let reconstruction = test_reconstruction(&frames);
        let mut state = RegistrationRetryState::new(frames.len());

        for _ in 0..3 {
            state.record_failure(
                &reconstruction,
                1,
                NextImageRegistrationMode::StructureBased,
                30,
            );
        }

        assert!(!state.is_eligible(
            &reconstruction,
            1,
            NextImageRegistrationMode::StructureBased,
            30,
            3,
            RegistrationPass::Normal,
        ));
        assert!(state.is_eligible(
            &reconstruction,
            1,
            NextImageRegistrationMode::StructureBased,
            30,
            3,
            RegistrationPass::ExhaustiveFallback,
        ));
        assert!(state.is_eligible(
            &reconstruction,
            1,
            NextImageRegistrationMode::StructureBased,
            31,
            3,
            RegistrationPass::Normal,
        ));
        assert_eq!(
            state.num_trials(
                &reconstruction,
                1,
                NextImageRegistrationMode::StructureBased,
                31,
            ),
            0
        );
    }

    #[test]
    fn registration_retry_state_keeps_modes_independent() {
        let frames = vec![
            minimal_frame(0, "registered.jpg"),
            minimal_frame(1, "candidate.jpg"),
        ];
        let reconstruction = test_reconstruction(&frames);
        let mut state = RegistrationRetryState::new(frames.len());
        for _ in 0..3 {
            state.record_failure(
                &reconstruction,
                1,
                NextImageRegistrationMode::StructureBased,
                30,
            );
        }

        assert!(!state.is_eligible(
            &reconstruction,
            1,
            NextImageRegistrationMode::StructureBased,
            30,
            3,
            RegistrationPass::Normal,
        ));
        assert!(state.is_eligible(
            &reconstruction,
            1,
            NextImageRegistrationMode::StructureLess,
            30,
            3,
            RegistrationPass::Normal,
        ));
    }

    #[test]
    fn registration_retry_state_shares_rig_sibling_attempts() {
        let frames = vec![
            minimal_frame(0, "registered.jpg"),
            minimal_frame(1, "rig_ref.jpg"),
            minimal_frame(2, "rig_aux.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        attach_two_image_rig_frame(&mut reconstruction, 1, 2);
        let mut state = RegistrationRetryState::new(frames.len());
        for _ in 0..3 {
            state.record_failure(
                &reconstruction,
                1,
                NextImageRegistrationMode::StructureBased,
                30,
            );
        }

        assert!(!state.is_eligible(
            &reconstruction,
            2,
            NextImageRegistrationMode::StructureBased,
            30,
            3,
            RegistrationPass::Normal,
        ));
        assert_eq!(
            state.num_trials(
                &reconstruction,
                2,
                NextImageRegistrationMode::StructureBased,
                30,
            ),
            3
        );
    }

    #[test]
    fn incremental_registration_telemetry_reports_hot_path_stages() {
        let telemetry = IncrementalRegistrationTelemetry {
            candidate_units: 7,
            skipped_unchanged: 3,
            structure_based_attempts: 4,
            structureless_attempts: 2,
            structureless_estimates: 2,
            structureless_accepted: 1,
            structureless_solver_ms: 6.25,
            fallback_epochs: 1,
            collect_observations_ms: 1.25,
            pose_solve_refine_ms: 2.5,
            observation_update_ms: 3.75,
            triangulation_ms: 4.5,
            gpu_pnp_focal_fallbacks: Vec::new(),
        };

        let line = telemetry.format_log();

        for key in [
            "candidate_units=7",
            "skipped_unchanged=3",
            "structure_based_attempts=4",
            "structureless_attempts=2",
            "structureless_estimates=2",
            "structureless_accepted=1",
            "structureless_solver_ms=6.25",
            "fallback_epochs=1",
            "collect_observations_ms=1.25",
            "pose_solve_refine_ms=2.50",
            "observation_update_ms=3.75",
            "triangulation_ms=4.50",
        ] {
            assert!(line.contains(key), "missing {key}: {line}");
        }
    }

    #[test]
    fn reset_unregistered_registration_trials_clears_stale_tail_frame_exclusions() {
        let frames = vec![
            minimal_frame(0, "a.jpg"),
            minimal_frame(1, "b.jpg"),
            minimal_frame(2, "c.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        let mut reg_trials = vec![3, 2, 3];
        let mut structureless_reg_trials = vec![1, 2, 2];

        reset_unregistered_registration_trials(
            &reconstruction,
            &mut reg_trials,
            &mut structureless_reg_trials,
        );

        assert_eq!(reg_trials, vec![3, 0, 0]);
        assert_eq!(structureless_reg_trials, vec![1, 0, 0]);
    }

    #[test]
    fn local_matching_requires_explicit_opt_in() {
        assert!(!MapperConfig::default().local_matching);
        assert_eq!(MapperConfig::default().local_window, 0);
        assert!(!MapperConfig::default().pose_graph);
        assert!(!MapperConfig::default().experimental_sequence_heuristics);
        assert!(MapperConfig::default().extract_colors);
        assert!(!MapperConfig::default().fix_existing_frames);
    }

    #[test]
    fn color_extraction_policy_controls_frame_color_sampling_like_colmap() {
        let mut frames = vec![minimal_frame(0, "a.jpg"), minimal_frame(1, "b.jpg")];
        frames[0].colors = vec![[10, 20, 30], [40, 50, 60]];
        frames[1].colors = vec![[70, 80, 90], [100, 110, 120]];

        apply_color_extraction_policy(&mut frames, false);
        assert_eq!(frames[0].colors, vec![[0, 0, 0], [0, 0, 0]]);
        assert_eq!(frames[1].colors, vec![[0, 0, 0], [0, 0, 0]]);

        frames[0].colors = vec![[1, 2, 3], [4, 5, 6]];
        apply_color_extraction_policy(&mut frames, true);
        assert_eq!(frames[0].colors, vec![[1, 2, 3], [4, 5, 6]]);
    }

    #[test]
    fn per_registration_color_extraction_only_updates_black_points_like_colmap() {
        let mut frames = vec![minimal_frame(0, "a.jpg"), minimal_frame(1, "b.jpg")];
        frames[0].colors = vec![[10, 20, 30], [40, 50, 60]];
        frames[1].colors = vec![[70, 80, 90], [100, 110, 120]];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses = vec![Some(SE3::identity()), Some(SE3::identity())];
        reconstruction.observations[0][0] = Some(0);
        reconstruction.observations[0][1] = Some(1);
        reconstruction.observations[1][0] = Some(0);
        reconstruction.point_ids = vec![1, 2];
        reconstruction.points = vec![
            Point3D {
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
            },
            Point3D {
                xyz: [0.1, 0.0, 2.0],
                color: [9, 9, 9],
                error: 0.0,
                track: vec![TrackObservation {
                    image: 0,
                    feature: 1,
                }],
            },
        ];

        let report = extract_colors_for_registration_unit(
            &frames,
            &mut reconstruction,
            0,
            &MapperConfig::default(),
        );

        assert_eq!(
            report,
            ColorExtractionReport {
                images: 1,
                updated_points: 1,
            }
        );
        assert_eq!(reconstruction.points[0].color, [10, 20, 30]);
        assert_eq!(reconstruction.points[1].color, [9, 9, 9]);

        let report = extract_colors_for_registration_unit(
            &frames,
            &mut reconstruction,
            1,
            &MapperConfig {
                extract_colors: false,
                ..MapperConfig::default()
            },
        );
        assert_eq!(report, ColorExtractionReport::default());
        assert_eq!(reconstruction.points[0].color, [10, 20, 30]);
    }

    #[test]
    fn extract_colors_for_image_reads_empty_database_frame_lazily() -> Result<()> {
        let dir = tempdir()?;
        let image_path = dir.path().join("registered.png");
        let mut image = image::RgbImage::from_pixel(2, 2, image::Rgb([1, 2, 3]));
        image.put_pixel(1, 1, image::Rgb([40, 50, 60]));
        image.save(&image_path)?;

        let mut frames = vec![minimal_frame(0, "registered.png")];
        frames[0].path = image_path;
        frames[0].width = 2;
        frames[0].height = 2;
        frames[0].colors.clear();
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.observations[0][1] = Some(0);
        reconstruction.point_ids = vec![1];
        reconstruction.points = vec![Point3D {
            xyz: [0.0, 0.0, 2.0],
            color: [0, 0, 0],
            error: 0.0,
            track: vec![TrackObservation {
                image: 0,
                feature: 1,
            }],
        }];

        let updated = extract_colors_for_image(&frames, &mut reconstruction, 0);

        assert_eq!(updated, 1);
        assert_eq!(reconstruction.points[0].color, [40, 50, 60]);
        Ok(())
    }

    #[test]
    fn final_all_image_color_extraction_averages_registered_track_colors_like_colmap() {
        let mut frames = vec![minimal_frame(0, "a.jpg"), minimal_frame(1, "b.jpg")];
        frames[0].colors = vec![[10, 20, 30], [40, 50, 60]];
        frames[1].colors = vec![[70, 80, 90], [100, 110, 120]];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses = vec![Some(SE3::identity()), Some(SE3::identity())];
        reconstruction.point_ids = vec![1, 2];
        reconstruction.points = vec![
            Point3D {
                xyz: [0.0, 0.0, 2.0],
                color: [255, 0, 0],
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
            },
            Point3D {
                xyz: [0.1, 0.0, 2.0],
                color: [9, 9, 9],
                error: 0.0,
                track: vec![TrackObservation {
                    image: 1,
                    feature: 1,
                }],
            },
        ];

        let report = extract_colors_for_all_registered_images(
            &frames,
            &mut reconstruction,
            &MapperConfig::default(),
        );

        assert_eq!(
            report,
            ColorExtractionReport {
                images: 2,
                updated_points: 2,
            }
        );
        assert_eq!(reconstruction.points[0].color, [40, 50, 60]);
        assert_eq!(reconstruction.points[1].color, [100, 110, 120]);

        let report = extract_colors_for_all_registered_images(
            &frames,
            &mut reconstruction,
            &MapperConfig {
                extract_colors: false,
                ..MapperConfig::default()
            },
        );
        assert_eq!(report, ColorExtractionReport::default());
    }

    #[test]
    fn mapper_ba_defaults_match_colmap_intrinsic_refinement_policy() {
        let frames = vec![minimal_frame(0, "a.jpg")];
        let reconstruction = test_reconstruction(&frames);

        let config = MapperConfig::default();
        assert_eq!(config.local_ba_iterations, 25);
        assert_eq!(config.local_ba_max_refinements, 2);
        assert_eq!(config.local_ba_max_refinement_change, 0.001);
        assert_eq!(config.global_ba_iterations, 50);
        assert_eq!(config.global_ba_images_ratio, 1.5);
        assert_eq!(config.global_ba_points_ratio, 1.5);

        let options = mapper_ba_options(
            &config,
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
        assert_eq!(options.num_threads, -1);

        let preferred = mapper_ba_options(
            &MapperConfig {
                ba_linear_solver: crate::ba::BundleAdjustmentLinearSolverPreference::IterativeSchur,
                ba_sparse_backend: crate::ba::BundleAdjustmentSparseLinearAlgebra::AccelerateSparse,
                ..MapperConfig::default()
            },
            &reconstruction,
            3,
            Some(vec![0]),
            Vec::new(),
            None,
            None,
        );
        assert_eq!(
            preferred.linear_solver,
            crate::ba::BundleAdjustmentLinearSolverPreference::IterativeSchur
        );
        assert_eq!(
            preferred.sparse_linear_algebra,
            crate::ba::BundleAdjustmentSparseLinearAlgebra::AccelerateSparse
        );

        let threaded_options = mapper_ba_options(
            &MapperConfig {
                threads: Some(4),
                ..MapperConfig::default()
            },
            &reconstruction,
            3,
            Some(vec![0]),
            Vec::new(),
            None,
            None,
        );
        assert_eq!(threaded_options.num_threads, 4);
    }

    #[test]
    fn global_ba_iteration_budget_does_not_grow_with_observations() {
        let frames = vec![minimal_frame(0, "a.jpg"), minimal_frame(1, "b.jpg")];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.points = vec![
            Point3D {
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
            };
            10_001
        ];
        let config = MapperConfig {
            global_ba_iterations: 50,
            ..MapperConfig::default()
        };

        assert_eq!(
            global_ba_iterations_for_reconstruction(&config, &reconstruction),
            50
        );
    }

    #[test]
    fn mapper_triangulator_options_keep_colmap_combination_sampler_serial() {
        let options = mapper_triangulator_options(&MapperConfig::default());
        assert_eq!(options.random_seed, -1);
        assert_eq!(options.num_threads, 1);

        let threaded = mapper_triangulator_options(&MapperConfig {
            random_seed: 7,
            threads: Some(4),
            ..MapperConfig::default()
        });
        assert_eq!(threaded.random_seed, 7);
        assert_eq!(threaded.num_threads, 1);
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
        assert_eq!(options.num_threads, -1);
    }

    #[test]
    fn global_reconstruction_options_follow_mapper_config() {
        let config = MapperConfig {
            random_seed: 42,
            global_ba: false,
            global_ba_iterations: 17,
            init_min_tri_angle_deg: 3.5,
            max_reprojection_error_px: 6.0,
            ..MapperConfig::default()
        };
        let options = global_reconstruction_options_from_config(&config);
        assert!(!options.run_global_ba);
        assert_eq!(options.global_ba_iterations, 17);
        assert!(options.use_joint_positioning);
        assert_eq!(
            options.refinement.max_refinements,
            config.global_ba_max_refinements
        );
        assert_eq!(
            options.triangulation.min_triangulation_angle_deg,
            TrackTriangulationOptions::default().min_triangulation_angle_deg
        );
        assert_eq!(
            options.triangulation.max_reprojection_error_px,
            TrackTriangulationOptions::default().max_reprojection_error_px
        );
        assert!(options.incremental_triangulation.ignore_two_view_tracks);
    }

    #[test]
    fn global_ba_options_keep_configured_iteration_budget_for_small_reconstructions() {
        let frames = (0..10)
            .map(|idx| minimal_frame(idx, &format!("image_{idx}.jpg")))
            .collect::<Vec<_>>();
        let mut reconstruction = test_reconstruction(&frames);
        for image in 0..9 {
            reconstruction.poses[image] = Some(SE3::identity());
        }
        let config = MapperConfig {
            global_ba_iterations: 50,
            ..MapperConfig::default()
        };

        let small_options = mapper_global_ba_options(
            &config,
            &reconstruction,
            config.global_ba_iterations,
            None,
            vec![],
            None,
            None,
        );
        assert_eq!(registered_frame_count(&reconstruction), 9);
        assert_eq!(small_options.iterations, 50);
        assert_eq!(small_options.function_tolerance, 0.0);
        assert_eq!(small_options.gradient_tolerance, 0.1);
        assert_eq!(small_options.parameter_tolerance, 0.0);
        assert_eq!(small_options.max_linear_solver_iterations, 200);

        reconstruction.poses[9] = Some(SE3::identity());
        let large_options = mapper_global_ba_options(
            &config,
            &reconstruction,
            config.global_ba_iterations,
            None,
            vec![],
            None,
            None,
        );
        assert_eq!(registered_frame_count(&reconstruction), 10);
        assert_eq!(large_options.iterations, 50);
        assert_eq!(large_options.gradient_tolerance, 1.0);
        assert_eq!(large_options.max_linear_solver_iterations, 100);
    }

    #[test]
    fn global_ba_redundant_points_gate_matches_colmap_min_frames() {
        let frames = (0..10)
            .map(|idx| minimal_frame(idx, &format!("image_{idx}.jpg")))
            .collect::<Vec<_>>();
        let mut reconstruction = test_reconstruction(&frames);
        for image in 0..9 {
            reconstruction.poses[image] = Some(SE3::identity());
        }
        let config = MapperConfig {
            global_ba_ignore_redundant_points3d: true,
            ..MapperConfig::default()
        };

        assert!(global_ba_redundant_point_ids(&config, &reconstruction).is_none());

        reconstruction.poses[9] = Some(SE3::identity());
        assert_eq!(
            global_ba_redundant_point_ids(&config, &reconstruction),
            Some(Vec::new())
        );
    }

    #[test]
    fn redundant_point_global_ba_options_fix_all_non_point_parameters() {
        let frames = vec![
            minimal_frame(0, "rig_ref.jpg"),
            minimal_frame(1, "rig_aux.jpg"),
            minimal_frame(2, "plain.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.cameras = vec![
            CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0),
            CameraModel::new_pinhole(100, 100, 60.0, 60.0, 50.0, 50.0),
        ];
        reconstruction.camera_ids = vec![11, 12];
        reconstruction.image_camera_indices = vec![0, 1, 1];
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
                    sensor_from_rig: Some(Rigid3::identity()),
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
                    sensor_id: aux_sensor.clone(),
                    data_id: reconstruction.image_id(1) as u64,
                },
            ],
        }];
        reconstruction.image_frame_indices[0] = Some(0);
        reconstruction.image_frame_indices[1] = Some(0);
        for image in 0..3 {
            reconstruction.poses[image] = Some(SE3::identity());
        }

        let options = redundant_point_global_ba_options(
            &MapperConfig::default(),
            &reconstruction,
            vec![2, 4],
        );

        assert_eq!(options.variable_images, Some(Vec::new()));
        assert_eq!(options.constant_images, vec![0, 1, 2]);
        assert_eq!(options.point_ids, Some(vec![2, 4]));
        assert_eq!(options.constant_cameras, vec![0, 1]);
        assert_eq!(options.constant_rigs, vec![3]);
        assert_eq!(options.constant_sensor_from_rig, vec![aux_sensor]);
        assert!(!options.refine_focal_length);
        assert!(!options.refine_principal_point);
        assert!(!options.refine_extra_params);
        assert_eq!(options.gauge, crate::ba::BundleAdjustmentGauge::Default);
    }

    #[test]
    fn redundant_points3d_empty_matches_colmap() {
        let reconstruction = test_reconstruction(&[]);
        assert!(find_redundant_points3d_colmap(0.0, &reconstruction).is_empty());
    }

    #[test]
    fn redundant_points3d_threshold_monotonically_prunes_like_colmap() {
        let mut frames = (0..5)
            .map(|idx| minimal_frame(idx, &format!("image_{idx}.jpg")))
            .collect::<Vec<_>>();
        for frame in &mut frames {
            frame.keypoints = (0..100)
                .map(|idx| rustslam::KeyPoint::new(10.0 + idx as f32 * 0.01, 10.0))
                .collect();
            frame.colors = vec![[0, 0, 0]; frame.keypoints.len()];
        }
        let mut reconstruction = test_reconstruction(&frames);
        for point_id in 0..100 {
            let track = (0..frames.len())
                .map(|image| TrackObservation {
                    image,
                    feature: point_id,
                })
                .collect::<Vec<_>>();
            add_test_point3d(&mut reconstruction, point_id as u64 + 1, track);
        }

        assert!(find_redundant_points3d_colmap(0.0, &reconstruction).is_empty());
        let mut prev_redundant = 0;
        for min_coverage_gain in [0.1, 0.4, 0.7, 10.0] {
            let redundant = find_redundant_points3d_colmap(min_coverage_gain, &reconstruction);
            assert!(redundant.len() > prev_redundant);
            prev_redundant = redundant.len();
        }
        assert_eq!(prev_redundant, reconstruction.points.len());
    }

    #[test]
    fn redundant_points3d_ties_use_colmap_point3d_id_order() {
        let mut frames = vec![minimal_frame(0, "image.jpg")];
        frames[0].keypoints = vec![rustslam::KeyPoint::new(10.0, 10.0); 3];
        frames[0].colors = vec![[0, 0, 0]; frames[0].keypoints.len()];
        let mut reconstruction = test_reconstruction(&frames);
        add_test_point3d(
            &mut reconstruction,
            10,
            vec![TrackObservation {
                image: 0,
                feature: 0,
            }],
        );
        add_test_point3d(
            &mut reconstruction,
            20,
            vec![TrackObservation {
                image: 0,
                feature: 1,
            }],
        );
        add_test_point3d(
            &mut reconstruction,
            15,
            vec![TrackObservation {
                image: 0,
                feature: 2,
            }],
        );

        let redundant = find_redundant_points3d_colmap(0.2, &reconstruction);

        assert_eq!(redundant, vec![0, 2]);
    }

    #[test]
    fn image_tile_idxs_match_colmap_clamp_semantics() {
        let mut frames = vec![minimal_frame(0, "image.jpg")];
        frames[0].keypoints = vec![
            rustslam::KeyPoint::new(-50.0, -50.0),
            rustslam::KeyPoint::new(0.0, 0.0),
            rustslam::KeyPoint::new(12.5, 12.5),
            rustslam::KeyPoint::new(99.9, 99.9),
            rustslam::KeyPoint::new(100.0, 100.0),
        ];
        frames[0].colors = vec![[0, 0, 0]; frames[0].keypoints.len()];
        let reconstruction = test_reconstruction(&frames);

        let tile_idxs = compute_image_tile_idxs_colmap(8, &reconstruction);

        assert_eq!(tile_idxs[0], vec![0, 0, 9, 63, 63]);
    }

    #[test]
    fn normalization_uses_colmap_robust_bbox_and_transforms_points_and_poses() {
        let coords = vec![
            glam::Vec3::new(2.0, 3.0, 4.0),
            glam::Vec3::new(-1.0, -2.0, -3.0),
            glam::Vec3::new(5.0, 5.0, 5.0),
            glam::Vec3::new(100.0, 100.0, 100.0),
            glam::Vec3::new(-100.0, -100.0, -100.0),
        ];
        let (bbox_min, bbox_max, centroid) =
            robust_bbox_and_centroid_colmap(coords, 0.3, 0.7).unwrap();
        assert_vec3_near(bbox_min, glam::Vec3::new(-1.0, -2.0, -3.0), 1.0e-6);
        assert_vec3_near(bbox_max, glam::Vec3::new(5.0, 5.0, 5.0), 1.0e-6);
        assert_vec3_near(centroid, glam::Vec3::new(2.0, 2.0, 2.0), 1.0e-6);

        let frames = vec![
            minimal_frame(0, "a.jpg"),
            minimal_frame(1, "b.jpg"),
            minimal_frame(2, "c.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        let rotation = glam::Quat::from_rotation_y(0.2);
        for (image, z) in [-20.0, -10.0, 0.0].into_iter().enumerate() {
            reconstruction.poses[image] = Some(pose_from_rotation_center(
                rotation,
                glam::Vec3::new(0.0, 0.0, z),
            ));
            reconstruction.points.push(Point3D {
                xyz: [0.0, 0.0, z],
                color: [0, 0, 0],
                error: 0.0,
                track: Vec::new(),
            });
            reconstruction.point_ids.push(image as u64 + 1);
        }

        let transform =
            normalize_reconstruction_colmap(&mut reconstruction, true, 10.0, 0.0, 1.0, false)
                .unwrap();
        assert_eq!(transform.scale, 1.0);
        assert_vec3_near(
            transform.translation,
            glam::Vec3::new(0.0, 0.0, 10.0),
            1.0e-6,
        );
        assert_vec3_near(
            camera_center(reconstruction.poses[0].unwrap()),
            glam::Vec3::new(0.0, 0.0, -10.0),
            1.0e-5,
        );
        assert_vec3_near(
            camera_center(reconstruction.poses[1].unwrap()),
            glam::Vec3::ZERO,
            1.0e-5,
        );
        assert_vec3_near(
            camera_center(reconstruction.poses[2].unwrap()),
            glam::Vec3::new(0.0, 0.0, 10.0),
            1.0e-5,
        );
        assert_vec3_near(
            glam::Vec3::from_array(reconstruction.points[0].xyz),
            glam::Vec3::new(0.0, 0.0, -10.0),
            1.0e-6,
        );
        assert!(pose_rotation(reconstruction.poses[0].unwrap()).abs_diff_eq(rotation, 1.0e-6));

        let mut reconstruction = test_reconstruction(&frames);
        for (image, z) in [-20.0, -10.0, 0.0].into_iter().enumerate() {
            reconstruction.poses[image] = Some(pose_from_rotation_center(
                glam::Quat::IDENTITY,
                glam::Vec3::new(0.0, 0.0, z),
            ));
            reconstruction.points.push(Point3D {
                xyz: [0.0, 0.0, z],
                color: [0, 0, 0],
                error: 0.0,
                track: Vec::new(),
            });
            reconstruction.point_ids.push(image as u64 + 1);
        }
        let transform =
            normalize_reconstruction_colmap(&mut reconstruction, false, 10.0, 0.0, 1.0, false)
                .unwrap();
        assert!((transform.scale - 0.5).abs() < 1.0e-6);
        assert_vec3_near(
            transform.translation,
            glam::Vec3::new(0.0, 0.0, 5.0),
            1.0e-6,
        );
        assert_vec3_near(
            camera_center(reconstruction.poses[0].unwrap()),
            glam::Vec3::new(0.0, 0.0, -5.0),
            1.0e-6,
        );
        assert_vec3_near(
            glam::Vec3::from_array(reconstruction.points[2].xyz),
            glam::Vec3::new(0.0, 0.0, 5.0),
            1.0e-6,
        );

        let mut reconstruction = test_reconstruction(&frames);
        for (image, z) in [-20.0, -10.0, 0.0].into_iter().enumerate() {
            reconstruction.poses[image] = Some(pose_from_rotation_center(
                glam::Quat::IDENTITY,
                glam::Vec3::new(0.0, 0.0, z),
            ));
        }
        let transform =
            normalize_reconstruction_colmap(&mut reconstruction, false, 10.0, 0.0, 1.0, true)
                .unwrap();
        assert!((transform.scale - 0.5).abs() < 1.0e-6);
        assert_vec3_near(
            transform.translation,
            glam::Vec3::new(0.0, 0.0, 5.0),
            1.0e-6,
        );
        assert_vec3_near(
            camera_center(reconstruction.poses[2].unwrap()),
            glam::Vec3::new(0.0, 0.0, 5.0),
            1.0e-6,
        );
    }

    #[test]
    fn normalization_scales_rig_sensors_and_registered_frames_like_colmap() {
        let frames = vec![
            minimal_frame(0, "rig_ref.jpg"),
            minimal_frame(1, "rig_aux.jpg"),
            minimal_frame(2, "unregistered.jpg"),
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
                    sensor_from_rig: Some(Rigid3 {
                        qvec: [1.0, 0.0, 0.0, 0.0],
                        tvec: [7.0, 0.0, 0.0],
                    }),
                },
                RigSensor {
                    sensor_id: aux_sensor.clone(),
                    sensor_from_rig: Some(Rigid3 {
                        qvec: [1.0, 0.0, 0.0, 0.0],
                        tvec: [2.0, 0.0, 0.0],
                    }),
                },
            ],
        }];
        reconstruction.frames = vec![
            Frame {
                frame_id: 9,
                rig_id: 3,
                rig_from_world: Rigid3::from_se3(pose_from_rotation_center(
                    glam::Quat::IDENTITY,
                    glam::Vec3::new(0.0, 0.0, -20.0),
                )),
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
            },
            Frame {
                frame_id: 10,
                rig_id: 3,
                rig_from_world: Rigid3 {
                    qvec: [1.0, 0.0, 0.0, 0.0],
                    tvec: [3.0, 4.0, 5.0],
                },
                data_ids: Vec::new(),
            },
        ];
        reconstruction.image_frame_indices = vec![Some(0), Some(0), Some(1)];
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::identity());
        sync_registered_image_poses_from_frames(&mut reconstruction);
        let unregistered_frame_before = reconstruction.frames[1].rig_from_world.clone();
        reconstruction.points.push(Point3D {
            xyz: [0.0, 0.0, -20.0],
            color: [0, 0, 0],
            error: 0.0,
            track: Vec::new(),
        });
        reconstruction.points.push(Point3D {
            xyz: [0.0, 0.0, 0.0],
            color: [0, 0, 0],
            error: 0.0,
            track: Vec::new(),
        });

        assert!(normalize_reconstruction_colmap(
            &mut reconstruction.clone(),
            false,
            10.0,
            0.0,
            1.0,
            true
        )
        .is_none());

        let transform =
            normalize_reconstruction_colmap(&mut reconstruction, false, 10.0, 0.0, 1.0, false)
                .unwrap();

        assert!((transform.scale - 0.5).abs() < 1.0e-6);
        assert_eq!(
            reconstruction.rigs[0].sensors[0]
                .sensor_from_rig
                .as_ref()
                .unwrap()
                .tvec,
            [7.0, 0.0, 0.0]
        );
        assert_eq!(
            reconstruction.rigs[0].sensors[1]
                .sensor_from_rig
                .as_ref()
                .unwrap()
                .tvec,
            [1.0, 0.0, 0.0]
        );
        assert_eq!(
            reconstruction.frames[1].rig_from_world,
            unregistered_frame_before
        );
        assert_vec3_near(
            camera_center(reconstruction.frames[0].rig_from_world.to_se3()),
            glam::Vec3::new(0.0, 0.0, -5.0),
            1.0e-6,
        );
        assert_vec3_near(
            camera_center(reconstruction.poses[0].unwrap()),
            glam::Vec3::new(0.0, 0.0, -5.0),
            1.0e-6,
        );
        assert_vec3_near(
            camera_center(reconstruction.poses[1].unwrap()),
            glam::Vec3::new(-1.0, 0.0, -5.0),
            1.0e-6,
        );
        assert!(reconstruction.poses[2].is_none());
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
    fn mapper_global_ba_options_map_database_pose_priors_to_registered_images() {
        let frames = vec![
            minimal_frame(0, "registered_a.jpg"),
            minimal_frame(1, "registered_b.jpg"),
            minimal_frame(2, "unregistered.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.camera_ids = vec![11];
        reconstruction.image_ids = vec![101, 102, 103];
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::identity());
        let covariance = [1.0, 0.1, 0.0, 0.1, 2.0, 0.0, 0.0, 0.0, 3.0];
        let config = MapperConfig {
            pose_priors: vec![
                test_pose_prior(1, 11, 101, [1.0, 2.0, 3.0], covariance),
                test_pose_prior(2, 11, 102, [4.0, 5.0, 6.0], [0.0; 9]),
                test_pose_prior(3, 11, 103, [7.0, 8.0, 9.0], [0.0; 9]),
            ],
            ..MapperConfig::default()
        };

        let options =
            mapper_global_ba_options(&config, &reconstruction, 5, None, vec![1], None, None);

        assert_eq!(options.pose_priors.len(), 1);
        assert_eq!(options.pose_priors[0].image, 0);
        assert_eq!(options.pose_priors[0].position, [1.0, 2.0, 3.0]);
        assert_eq!(options.pose_priors[0].position_covariance, covariance);
    }

    #[test]
    fn mapper_local_ba_options_filter_pose_priors_to_variable_images() {
        let frames = vec![
            minimal_frame(0, "registered_a.jpg"),
            minimal_frame(1, "registered_b.jpg"),
            minimal_frame(2, "registered_c.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.camera_ids = vec![11];
        reconstruction.image_ids = vec![101, 102, 103];
        for image in 0..3 {
            reconstruction.poses[image] = Some(SE3::identity());
        }
        let stats = registration_stats(&reconstruction);
        let config = MapperConfig {
            pose_priors: vec![
                test_pose_prior(1, 11, 101, [1.0, 0.0, 0.0], [0.0; 9]),
                test_pose_prior(2, 11, 102, [2.0, 0.0, 0.0], [0.0; 9]),
                test_pose_prior(3, 11, 103, [3.0, 0.0, 0.0], [0.0; 9]),
            ],
            ..MapperConfig::default()
        };

        let options = mapper_local_ba_options(
            &config,
            &reconstruction,
            &stats,
            5,
            vec![0, 2],
            vec![2],
            None,
            None,
        );

        assert_eq!(options.variable_images, Some(vec![0, 2]));
        assert_eq!(options.constant_images, vec![2]);
        assert_eq!(
            options
                .pose_priors
                .iter()
                .map(|prior| prior.image)
                .collect::<Vec<_>>(),
            vec![0]
        );
    }

    #[test]
    fn redundant_point_global_ba_options_do_not_inject_pose_priors() {
        let frames = vec![minimal_frame(0, "a.jpg"), minimal_frame(1, "b.jpg")];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.camera_ids = vec![11];
        reconstruction.image_ids = vec![101, 102];
        for image in 0..2 {
            reconstruction.poses[image] = Some(SE3::identity());
        }
        let config = MapperConfig {
            pose_priors: vec![test_pose_prior(1, 11, 101, [1.0, 2.0, 3.0], [0.0; 9])],
            ..MapperConfig::default()
        };

        let options = redundant_point_global_ba_options(&config, &reconstruction, vec![0]);

        assert_eq!(options.variable_images, Some(Vec::new()));
        assert!(options.pose_priors.is_empty());
    }

    #[test]
    fn mapper_pose_priors_match_rig_frame_data_ids() {
        let frames = vec![
            minimal_frame(0, "rig_ref.jpg"),
            minimal_frame(1, "rig_aux.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.cameras = vec![
            CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0),
            CameraModel::new_pinhole(100, 100, 60.0, 60.0, 50.0, 50.0),
        ];
        reconstruction.camera_ids = vec![11, 12];
        reconstruction.image_camera_indices = vec![0, 1];
        reconstruction.image_ids = vec![101, 102];
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::identity());
        reconstruction.rigs = vec![Rig {
            rig_id: 3,
            ref_sensor_id: Some(SensorId {
                sensor_type: SensorType::Camera,
                sensor_id: 11,
            }),
            sensors: vec![
                RigSensor {
                    sensor_id: SensorId {
                        sensor_type: SensorType::Camera,
                        sensor_id: 11,
                    },
                    sensor_from_rig: None,
                },
                RigSensor {
                    sensor_id: SensorId {
                        sensor_type: SensorType::Camera,
                        sensor_id: 12,
                    },
                    sensor_from_rig: Some(Rigid3::identity()),
                },
            ],
        }];
        reconstruction.frames = vec![Frame {
            frame_id: 9,
            rig_id: 3,
            rig_from_world: Rigid3::identity(),
            data_ids: vec![
                DataId {
                    sensor_id: SensorId {
                        sensor_type: SensorType::Camera,
                        sensor_id: 11,
                    },
                    data_id: 101,
                },
                DataId {
                    sensor_id: SensorId {
                        sensor_type: SensorType::Camera,
                        sensor_id: 12,
                    },
                    data_id: 102,
                },
            ],
        }];
        reconstruction.image_frame_indices = vec![Some(0), Some(0)];
        let config = MapperConfig {
            pose_priors: vec![test_pose_prior(1, 12, 102, [2.0, 3.0, 4.0], [0.0; 9])],
            ..MapperConfig::default()
        };

        let options = mapper_ba_options(
            &config,
            &reconstruction,
            5,
            Some(vec![0, 1]),
            Vec::new(),
            None,
            None,
        );

        assert_eq!(options.pose_priors.len(), 1);
        assert_eq!(options.pose_priors[0].image, 1);
        assert_eq!(options.pose_priors[0].position, [2.0, 3.0, 4.0]);
    }

    #[test]
    fn mapper_ba_defaults_match_colmap_incremental_pipeline_losses() {
        // Only assert when the environment overrides are unset so the test
        // stays deterministic.
        if std::env::var_os("RUSTSFM_BA_LOSS").is_some()
            || std::env::var_os("RUSTSFM_BA_LOSS_SCALE").is_some()
        {
            return;
        }
        let frames = vec![minimal_frame(0, "registered.jpg")];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        let stats = registration_stats(&reconstruction);
        let config = MapperConfig::default();

        let global_options =
            mapper_ba_options(&config, &reconstruction, 3, None, vec![], None, None);
        assert_eq!(
            global_options.loss_function,
            crate::ba::BundleAdjustmentLoss::Trivial
        );

        let local_options = mapper_local_ba_options(
            &config,
            &reconstruction,
            &stats,
            3,
            vec![0],
            vec![],
            None,
            None,
        );
        assert_eq!(
            local_options.loss_function,
            crate::ba::BundleAdjustmentLoss::Trivial
        );
        assert_eq!(
            mapper_local_ba_refinement_loss_function(0),
            crate::ba::BundleAdjustmentLoss::Trivial
        );
        assert_eq!(
            mapper_local_ba_refinement_loss_function(1),
            crate::ba::BundleAdjustmentLoss::Trivial
        );
    }

    #[test]
    fn local_ba_refinement_change_ratio_matches_colmap_formula() {
        assert_eq!(local_ba_refinement_change_ratio(0, 1, 2, 3), 0.0);
        assert_eq!(local_ba_refinement_change_ratio(100, 7, 11, 2), 0.2);
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
    fn local_ba_failure_rollback_preserves_triangulator_state_and_refreshes_stats() {
        let mut frames = vec![
            minimal_frame(0, "seed.jpg"),
            minimal_frame(1, "gauge.jpg"),
            minimal_frame(2, "candidate.jpg"),
        ];
        frames[0].keypoints = vec![
            rustslam::KeyPoint::new(50.0, 50.0),
            rustslam::KeyPoint::new(55.0, 50.0),
        ];
        frames[2].keypoints = vec![
            rustslam::KeyPoint::new(55.0, 50.0),
            rustslam::KeyPoint::new(60.0, 50.0),
        ];
        let pairs = vec![pair_with_inliers(0, 2, &[(0, 0), (1, 1)])];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::identity());
        let registration_snapshot = reconstruction.clone();
        let mut triangulation_state =
            IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);

        reconstruction.poses[2] = Some(SE3::from_quat_translation(
            glam::Quat::IDENTITY,
            glam::Vec3::new(1.0, 0.0, 0.0),
        ));
        let tri_options = IncrementalTriangulatorOptions {
            re_min_ratio: 0.5,
            re_max_trials: 1,
            min_angle_deg: 0.1,
            merge_max_reproj_error_px: 10.0,
            ignore_two_view_tracks: false,
            ..IncrementalTriangulatorOptions::default()
        };
        {
            let mut triangulator = IncrementalTriangulator::new(
                &frames,
                &pairs,
                &mut reconstruction,
                &mut triangulation_state,
            );
            assert_eq!(triangulator.retriangulate(&tri_options), 4);
        }
        assert_eq!(reconstruction.points.len(), 2);
        assert_eq!(
            triangulation_state
                .retriangulation_trials()
                .get(&(0, 2))
                .copied(),
            Some(1)
        );
        assert_eq!(
            triangulation_state
                .observation_manager()
                .num_observations(2),
            2
        );

        assert_eq!(
            registration_rollback_reason(&reconstruction, 2, true, false, &MapperConfig::default()),
            Some("local_ba_failed")
        );
        reconstruction = registration_snapshot;
        triangulation_state.sync_after_reconstruction_rollback(&frames, &pairs, &reconstruction);

        assert!(reconstruction.points.is_empty());
        assert!(reconstruction.poses[2].is_none());
        assert_eq!(
            triangulation_state
                .retriangulation_trials()
                .get(&(0, 2))
                .copied(),
            Some(1)
        );
        assert_eq!(
            triangulation_state
                .observation_manager()
                .num_visible_points3d(2),
            0
        );
        assert_eq!(
            triangulation_state
                .observation_manager()
                .num_correspondences_have_point3d(2, 0),
            0
        );
        assert_observation_manager_matches_fresh(
            &frames,
            &pairs,
            &reconstruction,
            &triangulation_state,
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
    fn local_ba_camera_fixing_counts_new_registration_like_colmap() {
        let frames = vec![
            minimal_frame(0, "registered_a.jpg"),
            minimal_frame(1, "registered_b.jpg"),
            minimal_frame(2, "candidate.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.cameras = vec![CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0)];
        reconstruction.camera_ids = vec![11];
        reconstruction.image_camera_indices = vec![0, 0, 0];
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::identity());

        let before_registration = registration_stats(&reconstruction);
        reconstruction.poses[2] = Some(SE3::identity());
        let variable_images = vec![0, 2];

        let stale_options = mapper_local_ba_options(
            &MapperConfig::default(),
            &reconstruction,
            &before_registration,
            5,
            variable_images.clone(),
            vec![],
            None,
            None,
        );
        assert!(
            stale_options.constant_cameras.is_empty(),
            "pre-registration stats miss the just-registered image and can leave shared camera intrinsics variable"
        );

        let mut colmap_stats = before_registration.clone();
        colmap_stats.register_frame_for_image_event(&reconstruction, 2);
        let colmap_options = mapper_local_ba_options(
            &MapperConfig::default(),
            &reconstruction,
            &colmap_stats,
            5,
            variable_images,
            vec![],
            None,
            None,
        );
        assert_eq!(colmap_options.constant_cameras, vec![0]);
    }

    #[test]
    fn fix_existing_frames_marks_seeded_registration_units_constant_for_local_ba() {
        let frames = vec![
            minimal_frame(0, "existing.jpg"),
            minimal_frame(1, "new.jpg"),
            minimal_frame(2, "neighbor.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        for image in 0..3 {
            reconstruction.poses[image] = Some(SE3::identity());
        }
        let mut stats = registration_stats(&reconstruction);
        let mut begin_reconstruction = reconstruction.clone();
        begin_reconstruction.poses[1] = None;
        begin_reconstruction.poses[2] = None;
        stats.set_existing_registration_units_from_reconstruction(&begin_reconstruction);
        let config = MapperConfig {
            fix_existing_frames: true,
            ..MapperConfig::default()
        };

        let options = mapper_local_ba_options(
            &config,
            &reconstruction,
            &stats,
            5,
            vec![0, 1, 2],
            Vec::new(),
            None,
            None,
        );

        assert_eq!(options.constant_images, vec![0]);
    }

    #[test]
    fn fix_existing_frames_marks_seeded_rig_frame_constant_for_local_ba() {
        let frames = vec![
            minimal_frame(0, "rig_ref.jpg"),
            minimal_frame(1, "rig_aux.jpg"),
            minimal_frame(2, "new.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        for image in 0..3 {
            reconstruction.poses[image] = Some(SE3::identity());
        }
        attach_two_image_rig_frame(&mut reconstruction, 0, 1);
        let mut stats = registration_stats(&reconstruction);
        let mut begin_reconstruction = reconstruction.clone();
        begin_reconstruction.poses[2] = None;
        stats.set_existing_registration_units_from_reconstruction(&begin_reconstruction);
        let config = MapperConfig {
            fix_existing_frames: true,
            ..MapperConfig::default()
        };

        let options = mapper_local_ba_options(
            &config,
            &reconstruction,
            &stats,
            5,
            vec![0, 1, 2],
            Vec::new(),
            None,
            None,
        );

        assert_eq!(options.constant_images, vec![0, 1]);
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
    fn filter_registered_frames_noops_below_colmap_min_frame_threshold() {
        let frames = (0..19)
            .map(|idx| minimal_frame(idx, &format!("image_{idx}.jpg")))
            .collect::<Vec<_>>();
        let mut reconstruction = test_reconstruction(&frames);
        for pose in &mut reconstruction.poses {
            *pose = Some(SE3::identity());
        }
        let mut stats = RegistrationStats::from_reconstruction(&reconstruction);
        let mut tri_state = IncrementalTriangulatorState::new(&frames, &[], &reconstruction);

        let filtered = filter_registered_frames(
            &frames,
            &[],
            &mut reconstruction,
            &MapperConfig::default(),
            &mut stats,
            None,
            &mut tri_state,
        );

        assert_eq!(filtered, 0);
        assert_eq!(registered_frame_count(&reconstruction), 19);
        assert!(reconstruction.poses.iter().all(|pose| pose.is_some()));
        assert_eq!(stats.num_total_reg_images, 19);
    }

    #[test]
    fn filter_registered_frames_deregisters_full_frame_and_rolls_back_stats() {
        let frames = (0..21)
            .map(|idx| minimal_frame(idx, &format!("image_{idx}.jpg")))
            .collect::<Vec<_>>();
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.cameras = vec![
            CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0),
            CameraModel::new_pinhole(100, 100, 60.0, 60.0, 50.0, 50.0),
        ];
        reconstruction.camera_ids = vec![11, 12];
        reconstruction.image_camera_indices = vec![0; frames.len()];
        reconstruction.image_camera_indices[1] = 1;
        attach_two_image_rig_frame(&mut reconstruction, 0, 1);
        for pose in &mut reconstruction.poses {
            *pose = Some(SE3::identity());
        }
        for image in 2..frames.len() {
            let point_id = reconstruction.points.len();
            reconstruction.observations[image][0] = Some(point_id);
            reconstruction.point_ids.push(point_id as u64 + 1);
            reconstruction.points.push(Point3D {
                xyz: [image as f32, 0.0, 5.0],
                color: [0, 0, 0],
                error: 0.0,
                track: vec![TrackObservation { image, feature: 0 }],
            });
        }
        let mut stats = RegistrationStats::from_reconstruction(&reconstruction);
        assert_eq!(registered_frame_count(&reconstruction), 20);
        assert_eq!(stats.num_reg_frames_per_rig.get(&99), Some(&1));
        assert_eq!(stats.registered_images_with_camera_id(11), 20);
        assert_eq!(stats.registered_images_with_camera_id(12), 1);
        assert_eq!(stats.num_total_reg_images, 21);

        let mut filtered_units = HashSet::new();
        let mut tri_state = IncrementalTriangulatorState::new(&frames, &[], &reconstruction);
        let filtered = filter_registered_frames(
            &frames,
            &[],
            &mut reconstruction,
            &MapperConfig::default(),
            &mut stats,
            Some(&mut filtered_units),
            &mut tri_state,
        );

        assert_eq!(filtered, 1);
        assert!(filtered_units.contains(&RegistrationUnitKey::Frame(0)));
        assert_eq!(registered_frame_count(&reconstruction), 19);
        assert!(reconstruction.poses[0].is_none());
        assert!(reconstruction.poses[1].is_none());
        assert!(reconstruction.poses[2..].iter().all(|pose| pose.is_some()));
        assert_eq!(stats.num_reg_frames_per_rig.get(&99), Some(&0));
        assert_eq!(stats.registered_images_with_camera_id(11), 19);
        assert_eq!(stats.registered_images_with_camera_id(12), 0);
        assert_eq!(stats.num_total_reg_images, 19);
        assert_eq!(reconstruction.points.len(), 19);
        assert!(reconstruction.observations[0]
            .iter()
            .all(|obs| obs.is_none()));
        assert!(reconstruction.observations[1]
            .iter()
            .all(|obs| obs.is_none()));
    }

    #[test]
    fn filter_registered_frames_preserves_triangulator_trial_state() {
        let mut frames = (0..21)
            .map(|idx| minimal_frame(idx, &format!("image_{idx}.jpg")))
            .collect::<Vec<_>>();
        frames[2].keypoints = vec![
            rustslam::KeyPoint::new(50.0, 50.0),
            rustslam::KeyPoint::new(55.0, 50.0),
        ];
        frames[3].keypoints = vec![
            rustslam::KeyPoint::new(55.0, 50.0),
            rustslam::KeyPoint::new(60.0, 50.0),
        ];
        let pairs = vec![pair_with_inliers(2, 3, &[(0, 0), (1, 1)])];
        let mut reconstruction = test_reconstruction(&frames);
        for pose in &mut reconstruction.poses {
            *pose = Some(SE3::identity());
        }
        reconstruction.poses[3] = Some(SE3::from_quat_translation(
            glam::Quat::IDENTITY,
            glam::Vec3::new(1.0, 0.0, 0.0),
        ));
        for image in 4..frames.len() {
            let point_id = reconstruction.points.len();
            reconstruction.observations[image][0] = Some(point_id);
            reconstruction.point_ids.push(point_id as u64 + 1);
            reconstruction.points.push(Point3D {
                xyz: [image as f32, 0.0, 5.0],
                color: [0, 0, 0],
                error: 0.0,
                track: vec![TrackObservation { image, feature: 0 }],
            });
        }
        let mut tri_state = IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        {
            let mut triangulator =
                IncrementalTriangulator::new(&frames, &pairs, &mut reconstruction, &mut tri_state);
            assert_eq!(
                triangulator.retriangulate(&IncrementalTriangulatorOptions {
                    re_min_ratio: 0.5,
                    re_max_trials: 1,
                    min_angle_deg: 0.1,
                    merge_max_reproj_error_px: 10.0,
                    ignore_two_view_tracks: false,
                    ..IncrementalTriangulatorOptions::default()
                }),
                4
            );
        }
        assert_eq!(
            tri_state.retriangulation_trials().get(&(2, 3)).copied(),
            Some(1)
        );
        let mut stats = RegistrationStats::from_reconstruction(&reconstruction);
        let mut filtered_units = HashSet::new();

        let filtered = filter_registered_frames(
            &frames,
            &pairs,
            &mut reconstruction,
            &MapperConfig::default(),
            &mut stats,
            Some(&mut filtered_units),
            &mut tri_state,
        );

        assert_eq!(filtered, 2);
        assert!(filtered_units.contains(&RegistrationUnitKey::Image(0)));
        assert!(filtered_units.contains(&RegistrationUnitKey::Image(1)));
        assert_eq!(
            tri_state.retriangulation_trials().get(&(2, 3)).copied(),
            Some(1)
        );
        assert_observation_manager_matches_fresh(&frames, &pairs, &reconstruction, &tri_state);
        assert_eq!(registered_frame_count(&reconstruction), 19);
    }

    #[test]
    fn fix_existing_frames_protects_seeded_units_from_registered_frame_filtering() {
        let frames = (0..20)
            .map(|idx| minimal_frame(idx, &format!("image_{idx}.jpg")))
            .collect::<Vec<_>>();
        let mut reconstruction = test_reconstruction(&frames);
        for image in 0..20 {
            reconstruction.poses[image] = Some(SE3::identity());
        }
        let mut stats = registration_stats(&reconstruction);
        let mut begin_reconstruction = reconstruction.clone();
        for image in 1..20 {
            begin_reconstruction.poses[image] = None;
        }
        stats.set_existing_registration_units_from_reconstruction(&begin_reconstruction);
        let mut filtered_units = HashSet::new();
        let config = MapperConfig {
            fix_existing_frames: true,
            ..MapperConfig::default()
        };
        let mut tri_state = IncrementalTriangulatorState::new(&frames, &[], &reconstruction);

        let filtered = filter_registered_frames(
            &frames,
            &[],
            &mut reconstruction,
            &config,
            &mut stats,
            Some(&mut filtered_units),
            &mut tri_state,
        );

        assert_eq!(filtered, 19);
        assert!(reconstruction.poses[0].is_some());
        assert!(reconstruction.poses[1..].iter().all(Option::is_none));
        assert!(!filtered_units.contains(&RegistrationUnitKey::Image(0)));
    }

    #[test]
    fn fix_existing_frames_protects_seeded_rig_frame_from_registered_frame_filtering() {
        let frames = (0..21)
            .map(|idx| minimal_frame(idx, &format!("image_{idx}.jpg")))
            .collect::<Vec<_>>();
        let mut reconstruction = test_reconstruction(&frames);
        attach_two_image_rig_frame(&mut reconstruction, 0, 1);
        for image in 0..21 {
            reconstruction.poses[image] = Some(SE3::identity());
        }
        let mut stats = registration_stats(&reconstruction);
        let mut begin_reconstruction = reconstruction.clone();
        for image in 2..21 {
            begin_reconstruction.poses[image] = None;
        }
        stats.set_existing_registration_units_from_reconstruction(&begin_reconstruction);
        let mut filtered_units = HashSet::new();
        let config = MapperConfig {
            fix_existing_frames: true,
            ..MapperConfig::default()
        };
        let mut tri_state = IncrementalTriangulatorState::new(&frames, &[], &reconstruction);

        let filtered = filter_registered_frames(
            &frames,
            &[],
            &mut reconstruction,
            &config,
            &mut stats,
            Some(&mut filtered_units),
            &mut tri_state,
        );

        assert_eq!(filtered, 19);
        assert!(filtered_units.contains(&RegistrationUnitKey::Image(2)));
        assert!(!filtered_units.contains(&RegistrationUnitKey::Frame(0)));
        assert!(reconstruction.poses[0].is_some());
        assert!(reconstruction.poses[1].is_some());
        assert!(reconstruction.poses[2..].iter().all(Option::is_none));
        assert_eq!(registered_frame_count(&reconstruction), 1);
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
    fn generalized_frame_requires_known_or_registered_focal_lengths() {
        let frames = vec![
            minimal_frame(0, "rig_ref.jpg"),
            minimal_frame(1, "rig_aux.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.cameras = vec![
            CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0),
            CameraModel::new_pinhole(100, 100, 45.0, 45.0, 50.0, 50.0),
        ];
        reconstruction.camera_ids = vec![11, 12];
        reconstruction.image_camera_indices = vec![0, 1];
        let stats = registration_stats(&reconstruction);

        assert!(frame_image_camera_has_good_focal_length(
            0,
            &reconstruction,
            &[true, false],
            &stats,
        ));
        assert!(!frame_image_camera_has_good_focal_length(
            1,
            &reconstruction,
            &[true, false],
            &stats,
        ));

        reconstruction.poses[1] = Some(SE3::identity());
        let stats = registration_stats(&reconstruction);
        assert!(frame_image_camera_has_good_focal_length(
            1,
            &reconstruction,
            &[true, false],
            &stats,
        ));
    }

    #[test]
    fn registration_resets_bogus_sibling_frame_cameras_from_priors() {
        let frames = vec![
            minimal_frame(0, "rig_ref.jpg"),
            minimal_frame(1, "rig_aux.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        let good = CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0);
        let bogus = CameraModel::new_pinhole(100, 100, 1.0e6, 1.0e6, 50.0, 50.0);
        let ref_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 11,
        };
        let aux_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 12,
        };
        reconstruction.cameras = vec![good, bogus];
        reconstruction.camera_ids = vec![11, 12];
        reconstruction.image_camera_indices = vec![0, 1];
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
                    sensor_from_rig: Some(Rigid3::identity()),
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
        let camera_priors = vec![good, good];

        reset_bogus_frame_cameras_from_priors(
            &mut reconstruction,
            0,
            &MapperConfig::default(),
            &camera_priors,
        );

        assert_eq!(reconstruction.cameras[0].fx, good.fx);
        assert_eq!(reconstruction.cameras[1].fx, good.fx);
    }

    #[test]
    fn non_trivial_rig_with_unknown_focal_uses_central_registration_first() {
        let true_camera = CameraModel::new_pinhole(120, 100, 60.0, 60.0, 60.0, 50.0);
        let initial_camera = CameraModel::new_pinhole(120, 100, 40.0, 40.0, 60.0, 50.0);
        let provider_pose = SE3::identity();
        let candidate_pose = SE3::from_quat_translation(
            glam::Quat::from_rotation_y(0.04),
            glam::Vec3::new(-0.18, 0.02, 0.04),
        );
        let ref_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 31,
        };
        let aux_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 32,
        };
        let points = (0..36)
            .map(|idx| {
                let col = (idx % 6) as f32;
                let row = (idx / 6) as f32;
                [
                    -0.45 + col * 0.18,
                    -0.3 + row * 0.15,
                    3.2 + idx as f32 * 0.02,
                ]
            })
            .collect::<Vec<_>>();
        let mut frames = vec![
            minimal_frame(0, "provider.jpg"),
            minimal_frame(1, "rig_ref.jpg"),
            minimal_frame(2, "rig_aux.jpg"),
        ];
        for frame in &mut frames {
            frame.width = true_camera.width;
            frame.height = true_camera.height;
            frame.keypoints.clear();
            frame.colors.clear();
        }
        for &point in &points {
            frames[0]
                .keypoints
                .push(project_test_point(true_camera, provider_pose, point));
            frames[1]
                .keypoints
                .push(project_test_point(true_camera, candidate_pose, point));
        }
        frames[2].keypoints = vec![rustslam::KeyPoint::new(10.0, 10.0)];
        for frame in &mut frames {
            frame.colors = vec![[0, 0, 0]; frame.keypoints.len()];
        }

        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.camera = true_camera;
        reconstruction.cameras = vec![true_camera, initial_camera, initial_camera];
        reconstruction.camera_ids = vec![1, 31, 32];
        reconstruction.image_camera_indices = vec![0, 1, 2];
        reconstruction.rigs = vec![Rig {
            rig_id: 5,
            ref_sensor_id: Some(ref_sensor.clone()),
            sensors: vec![
                RigSensor {
                    sensor_id: ref_sensor.clone(),
                    sensor_from_rig: None,
                },
                RigSensor {
                    sensor_id: aux_sensor.clone(),
                    sensor_from_rig: Some(Rigid3::identity()),
                },
            ],
        }];
        reconstruction.frames = vec![Frame {
            frame_id: 13,
            rig_id: 5,
            rig_from_world: Rigid3::identity(),
            data_ids: vec![
                DataId {
                    sensor_id: ref_sensor,
                    data_id: reconstruction.image_id(1) as u64,
                },
                DataId {
                    sensor_id: aux_sensor,
                    data_id: reconstruction.image_id(2) as u64,
                },
            ],
        }];
        reconstruction.image_frame_indices = vec![None, Some(0), Some(0)];
        reconstruction.poses[0] = Some(provider_pose);
        for (idx, &point) in points.iter().enumerate() {
            reconstruction.observations[0][idx] = Some(idx);
            reconstruction.points.push(Point3D {
                xyz: point,
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
        let config = MapperConfig {
            abs_pose_min_num_inliers: 24,
            pnp_iterations: 512,
            random_seed: 7,
            ..MapperConfig::default()
        };

        let choice = choose_next_registration(
            &frames,
            &[pair],
            &reconstruction,
            &[0; 3],
            &[0; 3],
            &HashSet::new(),
            &config,
            &camera_priors(&reconstruction),
            &[true, false, false],
            &registration_stats(&reconstruction),
        )
        .expect("central fallback registration");

        assert_eq!(choice.source, "pnp");
        assert!(choice.frame_image_poses.is_empty());
        assert!(choice.inlier_ratio > 0.5);
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
                    point_id: feature,
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
                loss_function: crate::ba::BundleAdjustmentLoss::Huber { scale: 4.0 },
                max_observation_error_px: 50.0,
                variable_images: Some(vec![1]),
                constant_images: vec![0],
                point_ids: Some((0..4).collect()),
                ..crate::ba::BundleAdjustmentOptions::default()
            },
        );

        assert!(report.is_err());
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
            None,
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
            None,
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
            &[0; 2],
            &HashSet::new(),
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
    fn high_quality_pnp_is_not_rejected_by_pair_rotation_diagnostic() {
        let mut frames = vec![
            minimal_frame(0, "provider.jpg"),
            minimal_frame(1, "candidate.jpg"),
        ];
        let camera = CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0);
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
            .map(|&point| project_test_point(camera, provider_pose, point))
            .collect();
        frames[1].keypoints = points
            .iter()
            .map(|&point| project_test_point(camera, candidate_pose, point))
            .collect();

        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.cameras = vec![camera];
        reconstruction.camera_ids = vec![1];
        reconstruction.image_camera_indices = vec![0, 0];
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
        let mut pair = pair_with_inliers(0, 1, &matches);
        pair.relative_pose =
            SE3::from_quat_translation(glam::Quat::from_rotation_y(0.8), glam::Vec3::X);
        let config = MapperConfig {
            abs_pose_min_num_inliers: 24,
            random_seed: 0,
            ..MapperConfig::default()
        };

        let choice = choose_next_registration(
            &frames,
            &[pair],
            &reconstruction,
            &[0; 2],
            &[0; 2],
            &HashSet::new(),
            &config,
            &camera_priors(&reconstruction),
            &[true],
            &registration_stats(&reconstruction),
        )
        .expect("COLMAP accepts a refined absolute pose independently of pair rotation metadata");

        assert_eq!(choice.source, "pnp");
        assert!(choice.inlier_ratio > 0.9);
        assert!(choice.mean_error_px < 1.0);
        assert!(choice.pair_rot_error > absolute_pose_pair_rotation_limit_deg());
    }

    #[cfg(feature = "poselib")]
    #[test]
    fn generalized_frame_registration_uses_gp3p_and_registers_full_rig_frame() {
        let camera = CameraModel::new_pinhole(200, 160, 80.0, 80.0, 100.0, 80.0);
        let provider_pose = SE3::identity();
        let ref_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 11,
        };
        let aux_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 12,
        };
        let ref_from_rig = SE3::identity();
        let aux_from_rig =
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(0.35, 0.0, 0.0));
        let rig_from_world = SE3::from_quat_translation(
            glam::Quat::from_rotation_y(0.04) * glam::Quat::from_rotation_x(-0.02),
            glam::Vec3::new(-0.18, 0.02, 0.06),
        );
        let points = (0..36)
            .map(|idx| {
                let col = (idx % 6) as f32;
                let row = (idx / 6) as f32;
                [
                    -0.45 + col * 0.18,
                    -0.35 + row * 0.16,
                    3.2 + idx as f32 * 0.025,
                ]
            })
            .collect::<Vec<_>>();
        let mut frames = vec![
            minimal_frame(0, "provider.jpg"),
            minimal_frame(1, "rig_ref.jpg"),
            minimal_frame(2, "rig_aux.jpg"),
        ];
        for frame in &mut frames {
            frame.width = camera.width;
            frame.height = camera.height;
            frame.keypoints.clear();
            frame.colors.clear();
        }
        for (idx, &point) in points.iter().enumerate() {
            frames[0]
                .keypoints
                .push(project_test_point(camera, provider_pose, point));
            let sensor_pose = if idx % 2 == 0 {
                ref_from_rig
            } else {
                aux_from_rig
            };
            let target_image = if idx % 2 == 0 { 1 } else { 2 };
            let cam_from_world = sensor_pose.compose(&rig_from_world);
            frames[target_image]
                .keypoints
                .push(project_test_point(camera, cam_from_world, point));
        }
        for frame in &mut frames {
            frame.colors = vec![[0, 0, 0]; frame.keypoints.len()];
        }

        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.cameras = vec![camera, camera];
        reconstruction.camera = camera;
        reconstruction.camera_ids = vec![11, 12];
        reconstruction.image_camera_indices = vec![0, 0, 1];
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
                    sensor_from_rig: Some(Rigid3::from_se3(aux_from_rig)),
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
                    data_id: reconstruction.image_id(1) as u64,
                },
                DataId {
                    sensor_id: aux_sensor,
                    data_id: reconstruction.image_id(2) as u64,
                },
            ],
        }];
        reconstruction.image_frame_indices = vec![None, Some(0), Some(0)];
        reconstruction.poses[0] = Some(provider_pose);
        for (idx, &point) in points.iter().enumerate() {
            reconstruction.observations[0][idx] = Some(idx);
            reconstruction.points.push(Point3D {
                xyz: point,
                color: [0, 0, 0],
                error: 0.0,
                track: vec![TrackObservation {
                    image: 0,
                    feature: idx,
                }],
            });
            reconstruction.point_ids.push(idx as u64 + 1);
        }
        let ref_matches = (0..points.len())
            .filter(|idx| idx % 2 == 0)
            .enumerate()
            .map(|(feature, provider_feature)| (provider_feature as u32, feature as u32))
            .collect::<Vec<_>>();
        let aux_matches = (0..points.len())
            .filter(|idx| idx % 2 == 1)
            .enumerate()
            .map(|(feature, provider_feature)| (provider_feature as u32, feature as u32))
            .collect::<Vec<_>>();
        let pairs = vec![
            pair_with_inliers(0, 1, &ref_matches),
            pair_with_inliers(0, 2, &aux_matches),
        ];
        let config = MapperConfig {
            abs_pose_min_num_inliers: 24,
            pnp_iterations: 128,
            random_seed: 23,
            ..MapperConfig::default()
        };

        let choice = choose_next_registration(
            &frames,
            &pairs,
            &reconstruction,
            &[0; 3],
            &[0; 3],
            &HashSet::new(),
            &config,
            &camera_priors(&reconstruction),
            &camera_prior_focal_flags(&reconstruction, true),
            &registration_stats(&reconstruction),
        )
        .expect("generalized frame registration");

        assert_eq!(choice.source, "generalized_frame");
        assert_eq!(choice.frame_image_poses.len(), 2);
        assert_eq!(choice.generalized_inliers.len(), points.len());
        assert!(choice
            .frame_image_poses
            .iter()
            .any(|&(image, _)| image == 1));
        assert!(choice
            .frame_image_poses
            .iter()
            .any(|&(image, _)| image == 2));
        let ref_pose = choice
            .frame_image_poses
            .iter()
            .find_map(|&(image, pose)| (image == 1).then_some(pose))
            .unwrap();
        let aux_pose = choice
            .frame_image_poses
            .iter()
            .find_map(|&(image, pose)| (image == 2).then_some(pose))
            .unwrap();
        assert!(crate::geometry::relative_rotation_deg(ref_pose, rig_from_world) < 3.0);
        assert!(
            crate::geometry::relative_rotation_deg(aux_pose, aux_from_rig.compose(&rig_from_world))
                < 3.0
        );
    }

    #[cfg(feature = "poselib")]
    #[test]
    fn generalized_frame_refinement_improves_rig_pose_reprojection_error() {
        let camera = CameraModel::new_pinhole(220, 180, 90.0, 90.0, 110.0, 90.0);
        let ref_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 21,
        };
        let aux_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 22,
        };
        let aux_from_rig =
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(0.42, 0.02, 0.0));
        let true_rig_from_world = SE3::from_quat_translation(
            glam::Quat::from_rotation_y(0.03) * glam::Quat::from_rotation_x(-0.015),
            glam::Vec3::new(-0.12, 0.04, 0.05),
        );
        let initial_rig_from_world = SE3::from_quat_translation(
            glam::Quat::from_rotation_y(0.065) * glam::Quat::from_rotation_x(-0.035),
            glam::Vec3::new(-0.22, 0.0, 0.09),
        );
        let points = (0..30)
            .map(|idx| {
                let col = (idx % 6) as f32;
                let row = (idx / 6) as f32;
                [
                    -0.5 + col * 0.18,
                    -0.32 + row * 0.15,
                    3.0 + idx as f32 * 0.02,
                ]
            })
            .collect::<Vec<_>>();
        let mut frames = vec![
            minimal_frame(0, "rig_ref.jpg"),
            minimal_frame(1, "rig_aux.jpg"),
        ];
        for frame in &mut frames {
            frame.width = camera.width;
            frame.height = camera.height;
            frame.keypoints.clear();
            frame.colors.clear();
        }
        for (idx, &point) in points.iter().enumerate() {
            let (target_image, sensor_pose) = if idx % 2 == 0 {
                (0, SE3::identity())
            } else {
                (1, aux_from_rig)
            };
            let cam_from_world = sensor_pose.compose(&true_rig_from_world);
            frames[target_image]
                .keypoints
                .push(project_test_point(camera, cam_from_world, point));
        }
        for frame in &mut frames {
            frame.colors = vec![[0, 0, 0]; frame.keypoints.len()];
        }

        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.camera = camera;
        reconstruction.cameras = vec![camera, camera];
        reconstruction.camera_ids = vec![21, 22];
        reconstruction.image_camera_indices = vec![0, 1];
        reconstruction.rigs = vec![Rig {
            rig_id: 4,
            ref_sensor_id: Some(ref_sensor.clone()),
            sensors: vec![
                RigSensor {
                    sensor_id: ref_sensor.clone(),
                    sensor_from_rig: None,
                },
                RigSensor {
                    sensor_id: aux_sensor.clone(),
                    sensor_from_rig: Some(Rigid3::from_se3(aux_from_rig)),
                },
            ],
        }];
        reconstruction.frames = vec![Frame {
            frame_id: 12,
            rig_id: 4,
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
        for (idx, &point) in points.iter().enumerate() {
            reconstruction.points.push(Point3D {
                xyz: point,
                color: [0, 0, 0],
                error: 0.0,
                track: Vec::new(),
            });
            reconstruction.point_ids.push(idx as u64 + 1);
        }

        let mut problem = GeneralizedFrameAbsolutePoseProblem {
            points2d: Vec::new(),
            points3d: Vec::new(),
            camera_idxs: Vec::new(),
            cams_from_rig: vec![SE3::identity(), aux_from_rig],
            cameras: vec![camera, camera],
            correspondences: Vec::new(),
        };
        let mut per_image_feature = [0usize; 2];
        for (idx, &point) in points.iter().enumerate() {
            let image = if idx % 2 == 0 { 0 } else { 1 };
            let feature = per_image_feature[image];
            per_image_feature[image] += 1;
            let kp = &frames[image].keypoints[feature];
            problem.points2d.push([kp.x() as f64, kp.y() as f64]);
            problem
                .points3d
                .push([point[0] as f64, point[1] as f64, point[2] as f64]);
            problem.camera_idxs.push(image);
            problem.correspondences.push(GeneralizedFrameInlier {
                image,
                feature,
                point_id: idx,
            });
        }
        let inlier_mask = vec![true; problem.correspondences.len()];
        let config = MapperConfig {
            abs_pose_min_num_inliers: 20,
            pnp_threshold_px: 6.0,
            ..MapperConfig::default()
        };
        let initial_inliers = generalized_frame_inliers_for_pose(
            initial_rig_from_world,
            &reconstruction,
            &problem,
            100.0,
        );
        let initial_mean = generalized_frame_mean_error_px(
            initial_rig_from_world,
            &reconstruction,
            &problem,
            &initial_inliers,
        );

        let refined = refine_generalized_frame_absolute_pose(
            0,
            &frames,
            &reconstruction,
            &config,
            &problem,
            initial_rig_from_world,
            &inlier_mask,
        )
        .expect("generalized frame refinement");
        let refined_ref_pose = refined
            .frame_image_poses
            .iter()
            .find_map(|&(image, pose)| (image == 0).then_some(pose))
            .unwrap();

        assert!(refined.mean_error_px < initial_mean);
        assert!(refined.inliers.len() >= config.abs_pose_min_num_inliers);
        assert!(
            crate::geometry::relative_rotation_deg(refined_ref_pose, true_rig_from_world)
                < crate::geometry::relative_rotation_deg(
                    initial_rig_from_world,
                    true_rig_from_world,
                )
        );
    }

    #[test]
    fn absolute_pose_refinement_uses_ransac_inliers_only() {
        let pose = SE3::identity();
        let camera = CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0);
        let good = AbsolutePoseObservation {
            feature: 0,
            point_id: 0,
            xy: [50.0, 50.0],
            xyz: [0.0, 0.0, 3.0],
        };
        let outlier = AbsolutePoseObservation {
            feature: 1,
            point_id: 1,
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

        let mut obs_manager = ObservationManager::new(&frames, &[], &reconstruction);
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
            &mut obs_manager,
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

        let mut obs_manager = ObservationManager::new(&frames, &[], &reconstruction);
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
            &mut obs_manager,
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
    fn default_structureless_path_uses_pair_pose_fallback_without_poselib() {
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
            &[0; 4],
            &HashSet::new(),
            &config,
            &camera_priors(&reconstruction),
            &camera_prior_focal_flags(&reconstruction, true),
            &registration_stats(&reconstruction),
        );

        if cfg!(feature = "poselib") {
            let choice = choice
                .expect("COLMAP structureless should register via correspondence graph + PoseLib");
            assert_eq!(choice.image, 1);
            assert_eq!(choice.source, "structureless");
        } else {
            let choice =
                choice.expect("default experimental structureless fallback should register");
            assert_eq!(choice.image, 1);
            assert_eq!(choice.source, "structureless");
            assert!(config.experimental_structureless_pair_pose_fallback);
        }
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
            &[0; 4],
            &HashSet::new(),
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
            &[0; 3],
            &HashSet::new(),
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
            &[0; 3],
            &HashSet::new(),
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
            &[0; 3],
            &HashSet::new(),
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
            &[0; 4],
            &HashSet::new(),
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

        mark_unregistered_images_with_no_absolute_pose_for_test(
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
    fn registration_trial_increment_applies_to_full_frame() {
        let frames = vec![
            minimal_frame(0, "provider.jpg"),
            minimal_frame(1, "rig_ref.jpg"),
            minimal_frame(2, "rig_aux.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        attach_two_image_rig_frame(&mut reconstruction, 1, 2);
        let mut reg_trials = vec![0; 3];

        increment_registration_unit_trials(&reconstruction, 1, &mut reg_trials);

        assert_eq!(reg_trials, vec![0, 1, 1]);
    }

    #[test]
    fn registration_unit_trial_count_uses_full_frame_max() {
        let frames = vec![
            minimal_frame(0, "provider.jpg"),
            minimal_frame(1, "rig_ref.jpg"),
            minimal_frame(2, "rig_aux.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        attach_two_image_rig_frame(&mut reconstruction, 1, 2);
        let reg_trials = vec![0, 0, 3];

        assert_eq!(
            registration_unit_num_trials(&reconstruction, 1, &reg_trials),
            3
        );
        assert_eq!(
            registration_unit_num_trials(&reconstruction, 2, &reg_trials),
            3
        );
    }

    #[test]
    fn structureless_registration_trials_are_tracked_separately() {
        let frames = vec![
            minimal_frame(0, "provider.jpg"),
            minimal_frame(1, "candidate.jpg"),
        ];
        let reconstruction = test_reconstruction(&frames);
        let mut reg_trials = vec![0; 2];
        let mut structureless_reg_trials = vec![0; 2];

        increment_registration_unit_trials_for_mode(
            &reconstruction,
            1,
            NextImageRegistrationMode::StructureBased,
            &mut reg_trials,
            &mut structureless_reg_trials,
        );
        assert_eq!(reg_trials, vec![0, 1]);
        assert_eq!(structureless_reg_trials, vec![0, 0]);

        increment_registration_unit_trials_for_mode(
            &reconstruction,
            1,
            NextImageRegistrationMode::StructureLess,
            &mut reg_trials,
            &mut structureless_reg_trials,
        );
        assert_eq!(reg_trials, vec![0, 1]);
        assert_eq!(structureless_reg_trials, vec![0, 1]);
        assert_eq!(
            registration_unit_num_trials_for_mode(
                &reconstruction,
                1,
                &reg_trials,
                &structureless_reg_trials,
                NextImageRegistrationMode::StructureBased,
            ),
            1
        );
        assert_eq!(
            registration_unit_num_trials_for_mode(
                &reconstruction,
                1,
                &reg_trials,
                &structureless_reg_trials,
                NextImageRegistrationMode::StructureLess,
            ),
            1
        );
    }

    #[test]
    fn mark_unregistered_no_pose_increments_frame_once() {
        let frames = vec![
            minimal_frame(0, "provider.jpg"),
            minimal_frame(1, "rig_ref.jpg"),
            minimal_frame(2, "rig_aux.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        attach_two_image_rig_frame(&mut reconstruction, 1, 2);
        reconstruction.poses[0] = Some(SE3::identity());
        let mut reg_trials = vec![0; 3];
        let config = MapperConfig {
            abs_pose_min_num_inliers: 20,
            ..MapperConfig::default()
        };

        mark_unregistered_images_with_no_absolute_pose_for_test(
            &frames,
            &[],
            &reconstruction,
            &mut reg_trials,
            &config,
            &camera_priors(&reconstruction),
            &camera_prior_focal_flags(&reconstruction, true),
            &registration_stats(&reconstruction),
        );

        assert_eq!(reg_trials, vec![0, 1, 1]);
    }

    #[test]
    fn mark_unregistered_skips_frame_when_any_sibling_hits_max_trials() {
        let frames = vec![
            minimal_frame(0, "provider.jpg"),
            minimal_frame(1, "rig_ref.jpg"),
            minimal_frame(2, "rig_aux.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        attach_two_image_rig_frame(&mut reconstruction, 1, 2);
        reconstruction.poses[0] = Some(SE3::identity());
        let mut reg_trials = vec![0, 0, 3];
        let config = MapperConfig {
            max_reg_trials: 3,
            abs_pose_min_num_inliers: 20,
            ..MapperConfig::default()
        };

        mark_unregistered_images_with_no_absolute_pose_for_test(
            &frames,
            &[],
            &reconstruction,
            &mut reg_trials,
            &config,
            &camera_priors(&reconstruction),
            &camera_prior_focal_flags(&reconstruction, true),
            &registration_stats(&reconstruction),
        );

        assert_eq!(reg_trials, vec![0, 0, 3]);
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
                    point_id: feature,
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
    fn global_ba_schedule_absolute_frequency_triggers_on_image_or_point_growth() {
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
            global_ba_images_freq: 1,
            global_ba_points_freq: 1,
            global_ba_images_ratio: 10.0,
            global_ba_points_ratio: 10.0,
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
    fn global_ba_schedule_ratio_triggers_on_first_frame_past_ratio() {
        // COLMAP's CheckRunGlobalRefinement applies the frame ratio threshold
        // with no minimum-growth gate: from a 2-frame baseline with ratio 1.1,
        // the third registered frame (3 >= 2.2) must trigger immediately.
        let frames = (0..3)
            .map(|image| minimal_frame(image, &format!("image_{image}.jpg")))
            .collect::<Vec<_>>();
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::identity());
        let point = Point3D {
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
        };
        reconstruction.points.push(point.clone());
        let schedule = GlobalBaSchedule::new(&reconstruction);
        let config = MapperConfig {
            global_ba_images_freq: 0,
            global_ba_points_freq: 0,
            global_ba_images_ratio: 1.1,
            global_ba_points_ratio: 10.0,
            ..MapperConfig::default()
        };

        reconstruction.points.push(point);
        assert!(!should_run_global_ba(&schedule, &reconstruction, &config));

        reconstruction.poses[2] = Some(SE3::identity());
        assert!(should_run_global_ba(&schedule, &reconstruction, &config));
    }

    #[test]
    fn pose_prior_alignment_keeps_pose_and_point_transforms_consistent() {
        let alignment = PosePriorAlignment {
            scale: 0.4,
            rotation: glam::Mat3::from_quat(glam::Quat::from_rotation_y(0.7)),
            translation: glam::Vec3::new(3.0, -1.0, 2.0),
        };
        let pose = SE3::from_quat_translation(
            glam::Quat::from_rotation_x(0.3),
            glam::Vec3::new(1.0, 2.0, 3.0),
        );
        let point = glam::Vec3::new(0.5, -0.25, 4.0);
        let camera_point = glam::Vec3::from_array(pose.transform_point(&point.to_array()));
        let transformed_camera_point = glam::Vec3::from_array(
            alignment
                .transform_pose(pose)
                .transform_point(&alignment.transform_point(point).to_array()),
        );
        let expected = alignment.scale * camera_point;
        assert!(
            (transformed_camera_point - expected).length() < 1.0e-4,
            "pose and point transforms must stay projection-consistent"
        );
    }

    #[test]
    fn degenerate_prior_alignment_rotation_removes_line_component() {
        // Nearly collinear prior positions: the rotation about the position
        // line is unobservable and must be removed from the alignment.
        let src = [[0.0, 0.0, 0.0], [1.0, 0.001, 0.002], [2.0, 0.003, 0.001]];
        let mut rotation = glam::Mat3::from_quat(glam::Quat::from_rotation_x(0.2));
        remove_degenerate_alignment_rotation(&mut rotation, &src);
        let (_, angle) = glam::Quat::from_mat3(&rotation).to_axis_angle();
        assert!(
            angle.abs() < 1.0e-2,
            "line rotation component must be removed"
        );
    }

    #[test]
    fn well_conditioned_prior_alignment_rotation_is_kept() {
        let src = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let mut rotation = glam::Mat3::from_quat(glam::Quat::from_rotation_z(0.3));
        remove_degenerate_alignment_rotation(&mut rotation, &src);
        let (_, angle) = glam::Quat::from_mat3(&rotation).to_axis_angle();
        assert!(
            (angle - 0.3).abs() < 1.0e-4,
            "well-conditioned rotation must stay"
        );
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
    fn incremental_global_ba_normalization_default_matches_colmap() {
        assert!(incremental_global_ba_normalizes_reconstruction(
            &MapperConfig::default()
        ));
    }

    #[test]
    fn scheduled_global_ba_caps_refinements_while_initial_and_final_keep_config() {
        let mut config = MapperConfig::default();
        config.global_ba_max_refinements = 5;
        assert_eq!(
            global_ba_max_refinements_for_reason(&config, "scheduled"),
            2
        );
        assert_eq!(global_ba_max_refinements_for_reason(&config, "initial"), 5);
        assert_eq!(global_ba_max_refinements_for_reason(&config, "final"), 5);

        config.global_ba_max_refinements = 1;
        assert_eq!(
            global_ba_max_refinements_for_reason(&config, "scheduled"),
            1
        );
        assert_eq!(global_ba_max_refinements_for_reason(&config, "final"), 1);

        let reconstruction = test_reconstruction(&structureless_frames(3));
        config.global_ba_iterations = 50;
        assert_eq!(
            global_ba_iterations_for_reason(&config, &reconstruction, "scheduled"),
            20
        );
        assert_eq!(
            global_ba_iterations_for_reason(&config, &reconstruction, "final"),
            50
        );
        config.global_ba_iterations = 10;
        assert_eq!(
            global_ba_iterations_for_reason(&config, &reconstruction, "scheduled"),
            10
        );
    }

    #[test]
    fn final_global_ba_normalization_disabled_like_colmap_final_all() {
        assert!(!final_global_ba_normalizes_reconstruction(
            &MapperConfig::default()
        ));
    }

    #[test]
    fn global_ba_prior_position_disables_incremental_normalization_like_colmap() {
        let frames = vec![
            minimal_frame(0, "a.jpg"),
            minimal_frame(1, "b.jpg"),
            minimal_frame(2, "c.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.camera_ids = vec![11];
        reconstruction.image_ids = vec![101, 102, 103];
        for image in 0..3 {
            reconstruction.poses[image] = Some(SE3::identity());
        }
        let config = MapperConfig {
            pose_priors: vec![
                test_pose_prior(1, 11, 101, [0.0, 0.0, 0.0], [0.0; 9]),
                test_pose_prior(2, 11, 102, [1.0, 0.0, 0.0], [0.0; 9]),
                test_pose_prior(3, 11, 103, [2.0, 0.0, 0.0], [0.0; 9]),
            ],
            ..MapperConfig::default()
        };
        let options =
            mapper_global_ba_options(&config, &reconstruction, 5, None, vec![], None, None);

        assert!(incremental_global_ba_normalizes_reconstruction(&config));
        assert!(global_ba_uses_prior_position(&options, &reconstruction));
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
        )
        .unwrap();

        assert_eq!(setup.camera_ids, vec![1, 2]);
        assert_eq!(setup.image_ids, vec![1, 2]);
        assert_eq!(setup.image_camera_indices, vec![0, 1]);
        assert_eq!(setup.cameras[0].width, 120);
        assert_eq!(setup.cameras[1].height, 150);
        assert_eq!(setup.cameras[0].fx, 90.0);
        assert_eq!(setup.cameras[1].cx, 100.0);
    }

    #[test]
    fn local_image_fallback_can_share_one_camera_across_same_sized_images() {
        let mut frames = vec![minimal_frame(0, "left.jpg"), minimal_frame(1, "right.jpg")];
        for frame in &mut frames {
            frame.width = 120;
            frame.height = 80;
        }

        let setup = local_image_camera_setup(
            &frames,
            &MapperConfig {
                single_camera: true,
                fx: Some(90.0),
                cy: Some(35.0),
                ..MapperConfig::default()
            },
        )
        .unwrap();

        assert_eq!(setup.camera_ids, vec![1]);
        assert_eq!(setup.image_ids, vec![1, 2]);
        assert_eq!(setup.image_camera_indices, vec![0, 0]);
        assert_eq!(setup.cameras.len(), 1);
        assert_eq!(setup.cameras[0].width, 120);
        assert_eq!(setup.cameras[0].height, 80);
        assert_eq!(setup.cameras[0].fx, 90.0);
        assert_eq!(setup.cameras[0].fy, 144.0);
        assert_eq!(setup.cameras[0].cx, 60.0);
        assert_eq!(setup.cameras[0].cy, 35.0);
    }

    #[test]
    fn local_image_fallback_rejects_mixed_dimensions_for_a_shared_camera() {
        let mut frames = vec![minimal_frame(0, "left.jpg"), minimal_frame(1, "right.jpg")];
        frames[0].width = 120;
        frames[0].height = 80;
        frames[1].width = 200;
        frames[1].height = 150;

        let err = local_image_camera_setup(
            &frames,
            &MapperConfig {
                single_camera: true,
                ..MapperConfig::default()
            },
        )
        .unwrap_err();

        assert!(err.to_string().contains("single camera"));
        assert!(err.to_string().contains("120x80"));
        assert!(err.to_string().contains("200x150"));
    }

    #[test]
    fn populate_local_matching_database_writes_features_matches_and_geometries() -> Result<()> {
        let dir = tempdir()?;
        let db_path = dir.path().join("database.db");
        let mut frames = vec![minimal_frame(0, "left.jpg"), minimal_frame(1, "right.jpg")];
        frames[0].sift.keypoints = frames[0].keypoints.clone();
        frames[1].sift.keypoints = frames[1].keypoints.clone();
        frames[0].sift.descriptors = vec![
            lowe_sift::Descriptor::new([1.0; lowe_sift::DESCRIPTOR_LEN]),
            lowe_sift::Descriptor::new([0.5; lowe_sift::DESCRIPTOR_LEN]),
        ];
        frames[1].sift.descriptors = vec![
            lowe_sift::Descriptor::new([0.9; lowe_sift::DESCRIPTOR_LEN]),
            lowe_sift::Descriptor::new([0.4; lowe_sift::DESCRIPTOR_LEN]),
        ];
        let setup = local_image_camera_setup(&frames, &MapperConfig::default()).unwrap();
        let mut pair = pair_with_inliers(0, 1, &[(0, 1)]);
        pair.matches = pair.inlier_matches.clone();
        pair.two_view_config = crate::database::COLMAP_TWO_VIEW_CALIBRATED;
        pair.qvec = Some([1.0, 0.0, 0.0, 0.0]);
        pair.tvec = Some([0.0, 0.0, 1.0]);

        let written = populate_local_matching_database(
            &db_path,
            &frames,
            &setup,
            std::slice::from_ref(&pair),
            FeatureType::Sift,
        )?;
        assert!(written >= 7);

        let db = ColmapDatabase::open(&db_path)?;
        assert_eq!(db.read_all_cameras()?.len(), 2);
        assert_eq!(db.read_all_images()?.len(), 2);
        assert_eq!(db.read_keypoints(1)?.len(), 2);
        assert_eq!(db.read_descriptors(1)?.rows, 2);
        assert_eq!(db.read_matches(1, 2)?.len(), 1);
        let geometry = db.read_two_view_geometry(1, 2)?;
        assert_eq!(geometry.config, crate::database::COLMAP_TWO_VIEW_CALIBRATED);
        assert_eq!(geometry.inlier_matches.len(), 1);
        Ok(())
    }

    #[test]
    fn resolve_mapper_database_path_allows_missing_output_for_local_write() -> Result<()> {
        let dir = tempdir()?;
        let missing = dir.path().join("new.db");
        let config = MapperConfig {
            input: dir.path().to_path_buf(),
            database: Some(missing.clone()),
            local_matching: true,
            write_database: true,
            ..MapperConfig::default()
        };
        assert_eq!(resolve_mapper_database_path(&config)?, Some(missing));
        Ok(())
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
        assert_eq!(pair.qvec, Some([1.0, 0.0, 0.0, 0.0]));
        assert_eq!(pair.tvec, Some([-1.0, 0.0, 0.0]));
        let stored = pair_geometry_to_colmap_two_view_geometry(&pair);
        assert_eq!(stored.config, crate::database::COLMAP_TWO_VIEW_CALIBRATED);
        assert_eq!(stored.qvec, pair.qvec);
        assert_eq!(stored.tvec, pair.tvec);
        assert_eq!(stored.inlier_matches.len(), pair.inlier_matches.len());
        Ok(())
    }

    #[test]
    fn estimate_database_pair_geometries_rejects_under_supported_stored_wide_baseline() -> Result<()>
    {
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
                config: crate::database::COLMAP_TWO_VIEW_CALIBRATED,
                inlier_matches: (0..4).map(|idx| FeatureMatch::new(idx, idx)).collect(),
                qvec: Some([1.0, 0.0, 0.0, 0.0]),
                tvec: Some([-1.0, 0.0, 0.0]),
                ..ColmapTwoViewGeometry::default()
            },
        )?;
        let mut frames = vec![
            minimal_frame(0, "left.jpg"),
            minimal_frame(1, "middle_1.jpg"),
            minimal_frame(2, "middle_2.jpg"),
            minimal_frame(3, "middle_3.jpg"),
            minimal_frame(4, "right.jpg"),
        ];
        let database = load_mapper_database(Some(&db_path), &frames, 0)?.expect("database input");
        apply_database_keypoints(&mut frames, &database.keypoints_by_name);
        let camera = CameraModel::from_colmap(
            crate::types::COLMAP_PINHOLE,
            100,
            100,
            &[50.0, 50.0, 50.0, 50.0],
        )
        .unwrap();

        let pairs = estimate_database_pair_geometries(
            &frames,
            &database.cache,
            &database.two_view_geometries,
            camera,
            None,
            &MapperConfig {
                min_inliers: 4,
                min_triangulated: 4,
                max_reprojection_error_px: 1.0,
                ..MapperConfig::default()
            },
        )?;

        assert!(pairs.is_empty());
        Ok(())
    }

    #[test]
    fn stored_database_pair_rejects_high_reprojection_error() {
        let mut pair = test_pair(0, 20, 60, 50, 12.0, [1.0, 0.0, 0.0]);
        pair.mean_reprojection_error_px = 2.0;
        let config = MapperConfig {
            max_reprojection_error_px: 1.5,
            ..MapperConfig::default()
        };

        assert!(!keep_pair_for_mapping(&pair, &config));
    }

    #[test]
    fn stored_database_pair_rejects_weak_triangulated_support() {
        let pair = test_pair(0, 20, 60, 3, 12.0, [1.0, 0.0, 0.0]);

        assert!(!keep_pair_for_mapping(&pair, &MapperConfig::default()));
    }

    #[test]
    fn stored_database_pair_accepts_supported_wide_baseline_rotation() {
        let mut pair = test_pair(0, 20, 60, 50, 12.0, [1.0, 0.0, 0.0]);
        pair.rotation_deg = 135.0;

        assert!(keep_pair_for_mapping(&pair, &MapperConfig::default()));
    }

    #[test]
    fn estimate_database_pair_geometries_keeps_stored_verified_edge_after_estimating_pose(
    ) -> Result<()> {
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
        let camera = CameraModel::from_colmap(
            crate::types::COLMAP_PINHOLE,
            100,
            100,
            &[50.0, 50.0, 50.0, 50.0],
        )
        .unwrap();
        let left_pose = SE3::identity();
        let right_pose =
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(-1.0, 0.0, 0.0));
        let points = [
            [-0.2, -0.1, 3.0],
            [0.2, -0.1, 3.2],
            [-0.25, 0.2, 3.4],
            [0.25, 0.2, 3.6],
            [0.0, 0.0, 4.0],
            [0.1, -0.25, 4.2],
            [-0.1, 0.25, 4.4],
            [0.3, 0.05, 4.6],
        ];
        let left_keypoints = points
            .iter()
            .map(|&point| {
                let kp = project_test_point(camera, left_pose, point);
                ColmapKeypoint::new(kp.x(), kp.y())
            })
            .collect::<Vec<_>>();
        let right_keypoints = points
            .iter()
            .map(|&point| {
                let kp = project_test_point(camera, right_pose, point);
                ColmapKeypoint::new(kp.x(), kp.y())
            })
            .collect::<Vec<_>>();
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
                config: crate::database::COLMAP_TWO_VIEW_UNCALIBRATED,
                inlier_matches: (0..points.len() as u32)
                    .map(|idx| FeatureMatch::new(idx, idx))
                    .collect(),
                qvec: None,
                tvec: None,
                ..ColmapTwoViewGeometry::default()
            },
        )?;
        let mut frames = vec![minimal_frame(0, "left.jpg"), minimal_frame(1, "right.jpg")];
        let database = load_mapper_database(Some(&db_path), &frames, 0)?.expect("database input");
        apply_database_keypoints(&mut frames, &database.keypoints_by_name);

        let pairs = estimate_database_pair_geometries(
            &frames,
            &database.cache,
            &database.two_view_geometries,
            camera,
            None,
            &MapperConfig {
                min_inliers: points.len(),
                min_triangulated: points.len(),
                random_seed: 0,
                ..MapperConfig::default()
            },
        )?;

        assert_eq!(pairs.len(), 1);
        assert_eq!((pairs[0].left, pairs[0].right), (0, 1));
        assert_eq!(
            pairs[0].two_view_config,
            crate::database::COLMAP_TWO_VIEW_UNCALIBRATED
        );
        assert_eq!(pairs[0].inliers, points.len());
        assert!(pairs[0].qvec.is_some());
        assert!(pairs[0].tvec.is_some());
        assert!(keep_pair_for_mapping(
            &pairs[0],
            &MapperConfig {
                min_inliers: points.len(),
                min_triangulated: points.len(),
                ..MapperConfig::default()
            }
        ));
        Ok(())
    }

    #[test]
    fn database_pair_geometry_orients_stored_pose_when_image_ids_descend() -> Result<()> {
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
        let first_keypoints = vec![
            ColmapKeypoint::new(45.0, 45.0),
            ColmapKeypoint::new(55.0, 45.0),
            ColmapKeypoint::new(45.0, 55.0),
            ColmapKeypoint::new(55.0, 55.0),
        ];
        let second_keypoints = vec![
            ColmapKeypoint::new(40.0, 45.0),
            ColmapKeypoint::new(50.0, 45.0),
            ColmapKeypoint::new(40.0, 55.0),
            ColmapKeypoint::new(50.0, 55.0),
        ];
        for (image_id, name, keypoints) in [
            (5, "first.jpg", first_keypoints),
            (2, "second.jpg", second_keypoints),
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
            5,
            2,
            &ColmapTwoViewGeometry {
                config: crate::database::COLMAP_TWO_VIEW_CALIBRATED,
                inlier_matches: (0..4).map(|idx| FeatureMatch::new(idx, idx)).collect(),
                qvec: Some([1.0, 0.0, 0.0, 0.0]),
                tvec: Some([-1.0, 0.0, 0.0]),
                ..ColmapTwoViewGeometry::default()
            },
        )?;
        let mut frames = vec![
            minimal_frame(0, "first.jpg"),
            minimal_frame(1, "second.jpg"),
        ];
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
        .expect("stored pose pair with descending image ids");

        assert_eq!(pair.inliers, 4);
        assert_eq!(pair.relative_pose.translation(), [-1.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn database_pair_geometry_preserves_calibrated_rig_config_for_initial_pair_gate() -> Result<()>
    {
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
                config: crate::database::COLMAP_TWO_VIEW_CALIBRATED_RIG,
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

        assert_eq!(
            pair.two_view_config,
            crate::database::COLMAP_TWO_VIEW_CALIBRATED_RIG
        );

        let mut reconstruction = test_reconstruction(&frames);
        let ref_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 1,
        };
        let aux_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 2,
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
                        tvec: [0.2, 0.0, 0.0],
                    }),
                },
            ],
        }];
        reconstruction.frames = vec![
            Frame {
                frame_id: 11,
                rig_id: 3,
                rig_from_world: Rigid3::identity(),
                data_ids: vec![DataId {
                    sensor_id: ref_sensor,
                    data_id: 1,
                }],
            },
            Frame {
                frame_id: 12,
                rig_id: 3,
                rig_from_world: Rigid3::identity(),
                data_ids: vec![DataId {
                    sensor_id: aux_sensor,
                    data_id: 2,
                }],
            },
        ];
        reconstruction.image_frame_indices = vec![Some(0), Some(1)];

        assert!(generalized_initial_pair_gate(&pair, &reconstruction));
        Ok(())
    }

    #[test]
    fn writes_pair_geometries_to_database_with_colmap_direction() -> Result<()> {
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
        for (image_id, name) in [(2, "left.jpg"), (1, "right.jpg")] {
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
        db.write_two_view_geometry(
            2,
            1,
            &ColmapTwoViewGeometry {
                config: crate::database::COLMAP_TWO_VIEW_CALIBRATED,
                inlier_matches: vec![FeatureMatch::new(9, 9)],
                ..ColmapTwoViewGeometry::default()
            },
        )?;

        let frames = vec![minimal_frame(0, "left.jpg"), minimal_frame(1, "right.jpg")];
        let mut pair = pair_with_inliers(0, 1, &[(0, 1), (2, 3)]);
        pair.two_view_config = crate::database::COLMAP_TWO_VIEW_PLANAR;
        pair.f_matrix = Some([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        pair.h_matrix = Some([1.0, 0.0, 2.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0]);
        pair.qvec = Some([1.0, 0.0, 0.0, 0.0]);
        pair.tvec = Some([0.0, 0.0, 1.0]);

        let written = write_pair_geometries_to_database(&db_path, &frames, &[pair.clone()])?;

        assert_eq!(written, 1);
        let read_logical = db.read_two_view_geometry(2, 1)?;
        assert_eq!(
            read_logical,
            pair_geometry_to_colmap_two_view_geometry(&pair)
        );
        let read_sorted = db.read_two_view_geometry(1, 2)?;
        assert_eq!(
            read_sorted.inlier_matches,
            vec![FeatureMatch::new(1, 0), FeatureMatch::new(3, 2)]
        );
        assert_eq!(read_sorted.qvec, Some([1.0, -0.0, -0.0, -0.0]));
        assert_eq!(read_sorted.tvec, Some([-0.0, -0.0, -1.0]));
        Ok(())
    }

    #[test]
    fn run_incremental_pipeline_reports_success_status() {
        let camera = CameraModel::new_pinhole(200, 160, 80.0, 80.0, 100.0, 80.0);
        let poses = [
            SE3::identity(),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(0.02),
                glam::Vec3::new(-0.35, 0.0, 0.0),
            ),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(-0.03),
                glam::Vec3::new(0.45, 0.0, 0.0),
            ),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(0.04),
                glam::Vec3::new(0.9, 0.0, 0.0),
            ),
        ];
        let points = (0..12)
            .map(|idx| {
                let col = (idx % 4) as f32;
                let row = (idx / 4) as f32;
                [-0.3 + col * 0.2, -0.2 + row * 0.18, 3.0 + idx as f32 * 0.03]
            })
            .collect::<Vec<_>>();
        let mut frames = (0..4)
            .map(|idx| minimal_frame(idx, &format!("image_{idx}.jpg")))
            .collect::<Vec<_>>();
        for (image, pose) in poses.iter().copied().enumerate() {
            frames[image].width = camera.width;
            frames[image].height = camera.height;
            frames[image].keypoints = points
                .iter()
                .map(|&point| project_test_point(camera, pose, point))
                .collect();
            frames[image].colors = vec![[image as u8, 0, 0]; points.len()];
        }
        let pairs = vec![initial_pair_from_projected_points(
            2,
            3,
            poses[2],
            poses[3],
            points.len(),
        )];
        let config = MapperConfig {
            multiple_models: false,
            local_ba: false,
            global_ba: false,
            init_num_trials: 1,
            init_min_num_inliers: 4,
            init_min_tri_angle_deg: 0.5,
            min_triangulated: 0,
            abs_pose_min_num_inliers: 4,
            ignore_two_view_tracks: false,
            ..MapperConfig::default()
        };

        let result = run_incremental_pipeline(&frames, camera, None, &pairs, &config, None);

        assert_eq!(result.status, IncrementalPipelineStatus::Success);
        assert_eq!(result.reconstructions.len(), 1);
        assert!(registered_image_count(&result.reconstructions[0]) >= 2);
    }

    #[test]
    fn rig_seed_continuation_registers_new_images_in_single_attempt() -> Result<()> {
        let dir = tempdir()?;
        let sparse = dir.path().join("sparse/0");
        let (camera, poses, points, frames, mut sparse_model) = colmap_rig_frame_seed_fixture();
        densify_colmap_rig_sparse_seed(&mut sparse_model, &frames, &points);
        write_colmap_sparse_text(&sparse, &sparse_model)?;
        let setup = reference_camera_setup(
            dir.path(),
            &frames.iter().map(|f| f.path.clone()).collect::<Vec<_>>(),
        )?;
        let pairs = vec![
            initial_pair_from_projected_points(0, 2, poses[0], poses[2], points.len()),
            initial_pair_from_projected_points(1, 2, poses[1], poses[2], points.len()),
            initial_pair_from_projected_points(2, 3, poses[2], poses[3], points.len()),
        ];
        let config = MapperConfig {
            multiple_models: false,
            local_ba: false,
            global_ba: false,
            init_num_trials: 1,
            init_min_num_inliers: 4,
            init_min_tri_angle_deg: 0.5,
            min_triangulated: 0,
            abs_pose_min_num_inliers: 1,
            max_reg_trials: 8,
            ..MapperConfig::default()
        };
        assert!(
            setup
                .seed_reconstruction
                .as_ref()
                .is_some_and(|seed| seed.poses.iter().any(Option::is_some)),
            "COLMAP sparse reference should seed existing reconstruction"
        );
        let mut session = IncrementalMapperSession::default();
        let (reconstruction, log) = incremental_map_single_attempt(
            &frames,
            camera,
            Some(&setup),
            &pairs,
            &config,
            &mut session,
            0,
            &mut MapperEventBridge::Silent,
        )?;

        assert!(log
            .iter()
            .any(|line| line.starts_with("continue_reconstruction registered_images=2 points=12")));
        assert!(!log.iter().any(|line| line.starts_with("initial_pair ")));
        assert!(reconstruction.poses[0].is_some());
        assert!(reconstruction.poses[1].is_some());
        assert!(
            reconstruction.poses[2].is_some(),
            "continuation should register image_2 from COLMAP rig seed: {:?}",
            log
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires the external test_data/flowers2_colmap fixture"]
    fn real_colmap_sparse_seed_registers_neighbor_with_mapper_pnp() -> Result<()> {
        let (camera, frames, setup, pairs, reference_pose) =
            real_colmap_sparse_seed_mapper_fixture()?;
        let config = MapperConfig {
            multiple_models: false,
            local_ba: false,
            global_ba: false,
            extract_colors: false,
            init_num_trials: 1,
            abs_pose_min_num_inliers: 120,
            pnp_iterations: 10_000,
            random_seed: 0,
            ..MapperConfig::default()
        };
        let mut session = IncrementalMapperSession::default();
        let (reconstruction, log) = incremental_map_single_attempt(
            &frames,
            camera,
            Some(&setup),
            &pairs,
            &config,
            &mut session,
            0,
            &mut MapperEventBridge::Silent,
        )?;

        assert!(log
            .iter()
            .any(|line| line.starts_with("continue_reconstruction registered_images=1 ")));
        assert!(!log.iter().any(|line| line.starts_with("initial_pair ")));
        let registration = log
            .iter()
            .find(|line| line.starts_with("register frame_0003.jpg source=pnp"))
            .unwrap_or_else(|| panic!("missing mapper PnP registration log: {log:?}"));
        assert!(
            registration.contains("pnp_inliers=256") || registration.contains("pnp_inliers=255"),
            "unexpected registration log: {registration}"
        );

        let estimated = reconstruction.poses[1].expect("candidate registered");
        let rotation_error = relative_rotation_deg(estimated, reference_pose);
        let translation_error = pose_translation_error(estimated, reference_pose);
        assert!(
            rotation_error < 0.2,
            "rotation_error={rotation_error}deg estimated={:?} reference={:?}",
            estimated.quaternion(),
            reference_pose.quaternion()
        );
        assert!(
            translation_error < 0.05,
            "translation_error={translation_error} estimated={:?} reference={:?}",
            estimated.translation(),
            reference_pose.translation()
        );
        Ok(())
    }

    #[test]
    fn single_target_registration_returns_candidate_without_bundle_adjustment() -> Result<()> {
        let camera = CameraModel::new_pinhole(320, 240, 220.0, 220.0, 160.0, 120.0);
        let seed_pose = SE3::identity();
        let target_pose = SE3::from_quat_translation(
            glam::Quat::from_rotation_y(0.035),
            glam::Vec3::new(-0.3, 0.02, 0.01),
        );
        let points = (0..64)
            .map(|index| {
                let column = (index % 8) as f32;
                let row = (index / 8) as f32;
                [
                    -0.7 + column * 0.2,
                    -0.55 + row * 0.16,
                    3.0 + (index % 5) as f32 * 0.12,
                ]
            })
            .collect::<Vec<_>>();
        let mut frames = vec![minimal_frame(0, "seed.jpg"), minimal_frame(1, "target.jpg")];
        for (frame, pose) in frames.iter_mut().zip([seed_pose, target_pose]) {
            frame.width = camera.width;
            frame.height = camera.height;
            frame.keypoints = points
                .iter()
                .map(|&point| project_test_point(camera, pose, point))
                .collect();
        }
        let seed_points = points
            .iter()
            .enumerate()
            .map(|(feature, &xyz)| Point3D {
                xyz,
                color: [feature as u8, 2, 3],
                error: 0.25,
                track: vec![TrackObservation { image: 0, feature }],
            })
            .collect::<Vec<_>>();
        let mut seed_observations = vec![vec![None; points.len()], vec![None; points.len()]];
        for feature in 0..points.len() {
            seed_observations[0][feature] = Some(feature);
        }
        let seed_point_ids = (0..points.len())
            .map(|index| index as u64 + 500)
            .collect::<Vec<_>>();
        let setup = ReferenceCameraSetup {
            cameras: vec![camera],
            camera_ids: vec![1],
            camera_has_prior_focal_length: vec![true],
            rigs: Vec::new(),
            frames: Vec::new(),
            image_ids: vec![1, 2],
            image_camera_indices: vec![0, 0],
            image_frame_indices: vec![None, None],
            seed_reconstruction: Some(ReconstructionSeed {
                poses: vec![Some(seed_pose), None],
                observations: seed_observations,
                point_ids: seed_point_ids.clone(),
                points: seed_points.clone(),
            }),
        };
        let pairs = vec![initial_pair_from_projected_points(
            0,
            1,
            seed_pose,
            target_pose,
            points.len(),
        )];
        let config = MapperConfig {
            reference: Some(PathBuf::from("required-by-contract")),
            fix_existing_frames: true,
            local_ba: false,
            global_ba: false,
            extract_colors: false,
            abs_pose_min_num_inliers: 16,
            pnp_iterations: 10_000,
            random_seed: 0,
            ..MapperConfig::default()
        };

        let attempt = register_single_target_from_seed(&frames, &pairs, &setup, 1, &config, None)?;
        let candidate = attempt
            .candidate
            .expect("target should have a PnP candidate");
        assert_eq!(attempt.debug_log.len(), 1);

        assert!(candidate.inlier_count >= 63);
        assert!(candidate.inlier_ratio.is_finite());
        assert!(candidate.mean_reprojection_error.is_finite());
        let preserved_seed = candidate.reconstruction.poses[0].expect("seed pose preserved");
        assert!(relative_rotation_deg(preserved_seed, seed_pose) < 1.0e-6);
        assert!(pose_translation_error(preserved_seed, seed_pose) < 1.0e-6);
        let estimated =
            candidate.reconstruction.poses[1].expect("target pose committed to candidate");
        assert!(relative_rotation_deg(estimated, target_pose) < 0.2);
        assert!(pose_translation_error(estimated, target_pose) < 0.05);
        assert_eq!(candidate.reconstruction.point_ids, seed_point_ids);
        assert_eq!(candidate.reconstruction.points.len(), seed_points.len());
        let committed_target_observations = candidate.reconstruction.observations[1]
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(committed_target_observations.len(), candidate.inlier_count);
        for (feature, point_id) in candidate.reconstruction.observations[1]
            .iter()
            .enumerate()
            .filter_map(|(feature, point_id)| point_id.map(|point_id| (feature, point_id)))
        {
            assert_eq!(point_id, feature);
            assert!(candidate.reconstruction.points[point_id]
                .track
                .contains(&TrackObservation { image: 1, feature }));
        }
        for (point_id, (actual, expected)) in candidate
            .reconstruction
            .points
            .iter()
            .zip(&seed_points)
            .enumerate()
        {
            assert_eq!(actual.xyz, expected.xyz);
            assert_eq!(actual.color, expected.color);
            assert_eq!(actual.error, expected.error);
            assert!(actual.track.starts_with(&expected.track));
            if !committed_target_observations.contains(&point_id) {
                assert_eq!(actual.track, expected.track);
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires the external test_data/flowers2_colmap fixture"]
    fn real_colmap_sparse_seed_mapper_pnp_survives_local_ba() -> Result<()> {
        let (camera, frames, setup, pairs, reference_pose) =
            real_colmap_sparse_seed_local_ba_fixture()?;
        let config = MapperConfig {
            multiple_models: false,
            local_ba: true,
            local_ba_iterations: 10,
            local_ba_max_refinements: 1,
            global_ba: false,
            extract_colors: false,
            init_num_trials: 1,
            abs_pose_min_num_inliers: 120,
            pnp_iterations: 10_000,
            random_seed: 0,
            ..MapperConfig::default()
        };
        let mut session = IncrementalMapperSession::default();
        let (reconstruction, log) = incremental_map_single_attempt(
            &frames,
            camera,
            Some(&setup),
            &pairs,
            &config,
            &mut session,
            0,
            &mut MapperEventBridge::Silent,
        )?;

        assert!(log
            .iter()
            .any(|line| line.starts_with("continue_reconstruction registered_images=2 ")));
        assert!(!log.iter().any(|line| line.starts_with("initial_pair ")));
        let registration = log
            .iter()
            .find(|line| line.starts_with("register frame_0003.jpg source=pnp"))
            .unwrap_or_else(|| panic!("missing mapper PnP registration log: {log:?}"));
        assert!(
            registration.contains("pnp_inliers=256") || registration.contains("pnp_inliers=255"),
            "unexpected registration log: {registration}"
        );
        let local_ba = log
            .iter()
            .find(|line| line.starts_with("local_ba image=frame_0003.jpg"))
            .unwrap_or_else(|| panic!("missing local BA log after mapper PnP: {log:?}"));
        assert!(
            local_ba.contains("local_images=1")
                && local_ba.contains("variable_images=2")
                && local_ba.contains("points=256"),
            "unexpected local BA log: {local_ba}"
        );

        let estimated = reconstruction.poses[2].expect("candidate registered after local BA");
        let rotation_error = relative_rotation_deg(estimated, reference_pose);
        let translation_error = pose_translation_error(estimated, reference_pose);
        assert!(
            rotation_error < 0.05,
            "rotation_error={rotation_error}deg estimated={:?} reference={:?}",
            estimated.quaternion(),
            reference_pose.quaternion()
        );
        assert!(
            translation_error < 0.05,
            "translation_error={translation_error} estimated={:?} reference={:?}",
            estimated.translation(),
            reference_pose.translation()
        );
        let candidate_observations = reconstruction.observations[2]
            .iter()
            .filter(|point| point.is_some())
            .count();
        assert!(
            candidate_observations >= 255,
            "candidate observations should survive local BA: {candidate_observations}"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires the external test_data/flowers2_colmap fixture"]
    fn real_colmap_sparse_seed_mapper_pnp_survives_scheduled_global_ba() -> Result<()> {
        let (camera, frames, setup, pairs, reference_pose) =
            real_colmap_sparse_seed_local_ba_fixture()?;
        let reference_relative = pairs
            .iter()
            .find(|pair| pair.left == 1 && pair.right == 2)
            .expect("seed-candidate reference pair")
            .relative_pose;
        let config = MapperConfig {
            multiple_models: false,
            local_ba: false,
            global_ba: true,
            global_ba_iterations: 10,
            global_ba_max_refinements: 1,
            global_ba_images_freq: 999,
            global_ba_points_freq: 999_999,
            global_ba_images_ratio: 1.1,
            global_ba_points_ratio: 10.0,
            extract_colors: false,
            init_num_trials: 1,
            abs_pose_min_num_inliers: 120,
            pnp_iterations: 10_000,
            random_seed: 0,
            ..MapperConfig::default()
        };
        let mut session = IncrementalMapperSession::default();
        let (reconstruction, log) = incremental_map_single_attempt(
            &frames,
            camera,
            Some(&setup),
            &pairs,
            &config,
            &mut session,
            0,
            &mut MapperEventBridge::Silent,
        )?;

        assert!(log
            .iter()
            .any(|line| line.starts_with("continue_reconstruction registered_images=2 ")));
        assert!(!log.iter().any(|line| line.starts_with("initial_pair ")));
        assert!(
            log.iter()
                .any(|line| line.starts_with("register frame_0003.jpg source=pnp")),
            "missing mapper PnP registration log: {log:?}"
        );
        let global_ba = log
            .iter()
            .find(|line| line.starts_with("global_ba reason=scheduled round=1"))
            .unwrap_or_else(|| panic!("missing scheduled global BA log: {log:?}"));
        assert!(
            global_ba.contains("size=small")
                && global_ba.contains("observations=")
                && global_ba.contains("residuals="),
            "unexpected scheduled global BA log: {global_ba}"
        );
        assert!(
            log.iter()
                .any(|line| line.starts_with("global_ba_normalize reason=scheduled round=1")),
            "scheduled global BA should normalize incremental reconstructions: {log:?}"
        );
        assert!(
            !log.iter()
                .any(|line| line.starts_with("registration_rollback ")),
            "global BA fixture should not roll back registration: {log:?}"
        );

        let estimated = reconstruction.poses[2].expect("candidate registered after global BA");
        let rotation_error = relative_rotation_deg(estimated, reference_pose);
        assert!(
            rotation_error < 0.1,
            "rotation_error={rotation_error}deg estimated={:?} reference={:?}",
            estimated.quaternion(),
            reference_pose.quaternion()
        );
        let estimated_relative = estimated.compose(
            &reconstruction.poses[1]
                .expect("seed remains registered")
                .inverse(),
        );
        let relative_rotation_error = relative_rotation_deg(estimated_relative, reference_relative);
        let relative_translation_error =
            pose_translation_direction_error_deg(estimated_relative, reference_relative);
        assert!(
            relative_rotation_error < 0.05,
            "relative_rotation_error={relative_rotation_error}deg"
        );
        assert!(
            relative_translation_error < 0.5,
            "relative_translation_direction_error={relative_translation_error}deg"
        );
        let candidate_observations = reconstruction.observations[2]
            .iter()
            .filter(|point| point.is_some())
            .count();
        assert!(
            candidate_observations >= 255,
            "candidate observations should survive scheduled global BA: {candidate_observations}"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires the external test_data/flowers2_colmap fixture"]
    fn real_colmap_sparse_pipeline_seeded_global_ba_prepare_completes_tracks() -> Result<()> {
        let (camera, frames, mut setup, pairs, reference_pose) =
            real_colmap_sparse_seed_local_ba_fixture()?;
        let seed = setup
            .seed_reconstruction
            .as_mut()
            .expect("fixture seed reconstruction");
        seed.poses[2] = Some(reference_pose);
        assert_eq!(
            seed.observations[2]
                .iter()
                .filter(|point| point.is_some())
                .count(),
            0
        );
        let config = MapperConfig {
            multiple_models: false,
            local_ba: false,
            global_ba: true,
            global_ba_iterations: 3,
            global_ba_max_refinements: 1,
            global_ba_images_freq: 999,
            global_ba_points_freq: 999_999,
            global_ba_images_ratio: 10.0,
            global_ba_points_ratio: 10.0,
            extract_colors: false,
            init_num_trials: 1,
            random_seed: 0,
            ..MapperConfig::default()
        };

        let result = run_incremental_pipeline(&frames, camera, Some(&setup), &pairs, &config, None);

        assert_eq!(result.status, IncrementalPipelineStatus::Success);
        assert_eq!(result.reconstructions.len(), 1);
        assert!(result
            .debug_log
            .iter()
            .any(|line| line.starts_with("continue_reconstruction registered_images=3 ")));
        let prepare = result
            .debug_log
            .iter()
            .find(|line| line.starts_with("global_ba_prepare reason=initial"))
            .unwrap_or_else(|| {
                panic!(
                    "missing pipeline global BA prepare log: {:?}",
                    result.debug_log
                )
            });
        let completed = prepare
            .split_whitespace()
            .find_map(|field| field.strip_prefix("completed="))
            .and_then(|value| value.parse::<usize>().ok())
            .expect("prepare completed count");
        assert!(
            completed > 0,
            "pipeline global BA prepare should complete true COLMAP tracks: {prepare}"
        );
        let reconstruction = &result.reconstructions[0];
        assert_eq!(
            reconstruction.observations[2]
                .iter()
                .filter(|point| point.is_some())
                .count(),
            256
        );
        assert!(
            !result
                .debug_log
                .iter()
                .any(|line| line.starts_with("registration_rollback ")),
            "seeded pipeline prepare fixture should not roll back: {:?}",
            result.debug_log
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires the external test_data/flowers2_colmap fixture"]
    fn real_colmap_sparse_seed_mapper_pnp_prior_global_ba_skips_normalization() -> Result<()> {
        let (camera, frames, setup, pairs, reference_pose) =
            real_colmap_sparse_seed_local_ba_fixture()?;
        let seed = setup
            .seed_reconstruction
            .as_ref()
            .expect("fixture seed reconstruction");
        let pose_priors = [
            seed.poses[0].expect("gauge pose"),
            seed.poses[1].expect("seed pose"),
            reference_pose,
        ]
        .into_iter()
        .enumerate()
        .map(|(image, pose)| {
            let center = camera_center(pose);
            test_pose_prior(
                image as u32 + 1,
                setup.camera_ids[setup.image_camera_indices[image]],
                setup.image_ids[image] as u64,
                [center.x as f64, center.y as f64, center.z as f64],
                position_prior_covariance(0.01),
            )
        })
        .collect::<Vec<_>>();
        let config = MapperConfig {
            multiple_models: false,
            local_ba: false,
            global_ba: true,
            global_ba_iterations: 10,
            global_ba_max_refinements: 1,
            global_ba_images_freq: 999,
            global_ba_points_freq: 999_999,
            global_ba_images_ratio: 1.1,
            global_ba_points_ratio: 10.0,
            pose_priors,
            extract_colors: false,
            init_num_trials: 1,
            abs_pose_min_num_inliers: 120,
            pnp_iterations: 10_000,
            random_seed: 0,
            ..MapperConfig::default()
        };
        let mut session = IncrementalMapperSession::default();
        let (reconstruction, log) = incremental_map_single_attempt(
            &frames,
            camera,
            Some(&setup),
            &pairs,
            &config,
            &mut session,
            0,
            &mut MapperEventBridge::Silent,
        )?;

        assert!(
            log.iter()
                .any(|line| line.starts_with("global_ba reason=scheduled round=1")),
            "missing scheduled global BA log: {log:?}"
        );
        assert!(
            !log.iter()
                .any(|line| line.starts_with("global_ba_normalize reason=scheduled")),
            "COLMAP prior-position global BA should skip normalization: {log:?}"
        );

        let estimated =
            reconstruction.poses[2].expect("candidate registered after prior-position global BA");
        let rotation_error = relative_rotation_deg(estimated, reference_pose);
        let estimated_center = camera_center(estimated);
        let reference_center = camera_center(reference_pose);
        assert!(
            rotation_error < 3.0,
            "rotation_error={rotation_error}deg estimated={:?} reference={:?}",
            estimated.quaternion(),
            reference_pose.quaternion()
        );
        assert!(
            estimated_center.distance(reference_center) < 0.25,
            "center_error={} estimated={:?} reference={:?}",
            estimated_center.distance(reference_center),
            estimated_center,
            reference_center
        );
        Ok(())
    }

    #[test]
    fn initial_pair_prefers_strong_non_adjacent_colmap_style_candidate() {
        let weak_adjacent = test_pair(0, 1, 120, 60, 20.0, [1.0, 0.0, 0.0]);
        let strong_non_adjacent = test_pair(0, 3, 220, 140, 25.0, [1.0, 0.1, 0.0]);
        let pairs = vec![weak_adjacent, strong_non_adjacent];
        let frames = structureless_frames(4);
        let reconstruction = test_reconstruction(&frames);

        let chosen = choose_initial_pair_for_test(
            &pairs,
            &reconstruction,
            &MapperConfig::default(),
            &camera_prior_focal_flags(&reconstruction, true),
        )
        .unwrap();

        assert_eq!((chosen.left, chosen.right), (0, 3));
    }

    #[test]
    fn parallel_initial_pair_selection_matches_sequential_colmap_order() {
        let weak_adjacent = test_pair(0, 1, 120, 60, 20.0, [1.0, 0.0, 0.0]);
        let strong_non_adjacent = test_pair(0, 3, 220, 140, 25.0, [1.0, 0.1, 0.0]);
        let pairs = vec![weak_adjacent, strong_non_adjacent];
        let frames = structureless_frames(4);
        let reconstruction = test_reconstruction(&frames);
        let camera_flags = camera_prior_focal_flags(&reconstruction, true);

        let sequential = {
            let mut selection_state =
                InitialPairSelectionState::from_reconstruction(&reconstruction);
            choose_initial_pair(
                &pairs,
                &reconstruction,
                &MapperConfig::default(),
                &camera_flags,
                &mut selection_state,
            )
            .unwrap()
        };
        let parallel = {
            let mut selection_state =
                InitialPairSelectionState::from_reconstruction(&reconstruction);
            choose_initial_pair(
                &pairs,
                &reconstruction,
                &MapperConfig {
                    threads: Some(4),
                    ..MapperConfig::default()
                },
                &camera_flags,
                &mut selection_state,
            )
            .unwrap()
        };

        assert_eq!(
            (sequential.left, sequential.right),
            (parallel.left, parallel.right)
        );
        assert_eq!((sequential.left, sequential.right), (0, 3));
    }

    #[test]
    fn reconstruction_with_reset_frame_cameras_restores_bogus_sibling_prior() {
        let frames = vec![
            minimal_frame(0, "rig_ref.jpg"),
            minimal_frame(1, "rig_aux.jpg"),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        let good = CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0);
        let bogus = CameraModel::new_pinhole(100, 100, 1.0e6, 1.0e6, 50.0, 50.0);
        let ref_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 11,
        };
        let aux_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 12,
        };
        reconstruction.cameras = vec![good, bogus];
        reconstruction.camera_ids = vec![11, 12];
        reconstruction.image_camera_indices = vec![0, 1];
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
                    sensor_from_rig: Some(Rigid3::identity()),
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
        let priors = vec![good, good];
        let config = MapperConfig::default();

        let snapshot =
            reconstruction_with_reset_frame_cameras(&reconstruction, 0, &config, &priors);

        assert!(!camera_has_bogus_params(
            snapshot.camera_for_image(1),
            &config
        ));
        assert!(camera_has_bogus_params(
            reconstruction.camera_for_image(1),
            &config
        ));
    }

    #[test]
    fn initial_pair_selection_follows_colmap_prior_focal_first_ordering() {
        let strong_unknown_focal = test_pair(0, 1, 600, 500, 30.0, [1.0, 0.0, 0.0]);
        let weaker_prior_focal = test_pair(2, 3, 180, 140, 18.0, [1.0, 0.0, 0.0]);
        let pairs = vec![strong_unknown_focal, weaker_prior_focal];
        let frames = structureless_frames(4);
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.cameras = vec![
            CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0),
            CameraModel::new_pinhole(100, 100, 55.0, 55.0, 50.0, 50.0),
        ];
        reconstruction.camera_ids = vec![11, 12];
        reconstruction.image_camera_indices = vec![0, 0, 1, 1];

        let chosen = choose_initial_pair_for_test(
            &pairs,
            &reconstruction,
            &MapperConfig::default(),
            &[false, true],
        )
        .unwrap();

        assert_eq!((chosen.left, chosen.right), (2, 3));
    }

    #[test]
    fn initial_pair_selection_orients_pair_to_colmap_first_image_order() {
        let stored_as_reverse = pair_with_inliers(1, 0, &[(7, 3), (8, 4)]);
        let pairs = vec![stored_as_reverse];
        let frames = structureless_frames(2);
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.cameras = vec![
            CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0),
            CameraModel::new_pinhole(100, 100, 55.0, 55.0, 50.0, 50.0),
        ];
        reconstruction.camera_ids = vec![11, 12];
        reconstruction.image_camera_indices = vec![0, 1];

        let chosen = choose_initial_pair_for_test(
            &pairs,
            &reconstruction,
            &MapperConfig {
                init_min_num_inliers: 2,
                init_min_tri_angle_deg: 0.5,
                min_triangulated: 0,
                ..MapperConfig::default()
            },
            &[true, false],
        )
        .unwrap();

        assert_eq!((chosen.left, chosen.right), (0, 1));
        assert_eq!(chosen.inlier_matches[0].query_idx, 3);
        assert_eq!(chosen.inlier_matches[0].train_idx, 7);
        assert_eq!(chosen.relative_pose.translation(), [-1.0, -0.0, -0.0]);
    }

    #[test]
    fn initial_pair_rejects_forward_motion_and_low_triangulation_in_strict_pass() {
        let forward_motion = test_pair(0, 1, 300, 200, 30.0, [0.0, 0.0, 1.0]);
        let low_angle = test_pair(0, 2, 260, 180, 2.0, [1.0, 0.0, 0.0]);
        let stable = test_pair(1, 3, 140, 90, 18.0, [1.0, 0.0, 0.1]);
        let pairs = vec![forward_motion, low_angle, stable];
        let frames = structureless_frames(4);
        let reconstruction = test_reconstruction(&frames);

        let chosen = choose_initial_pair_for_test(
            &pairs,
            &reconstruction,
            &MapperConfig::default(),
            &camera_prior_focal_flags(&reconstruction, true),
        )
        .unwrap();

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

        assert!(choose_initial_pair_for_test(
            &pairs,
            &reconstruction,
            &MapperConfig::default(),
            &camera_prior_focal_flags(&reconstruction, true),
        )
        .is_none());
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

        let chosen = choose_initial_pair_for_test(
            &pairs,
            &reconstruction,
            &MapperConfig::default(),
            &camera_prior_focal_flags(&reconstruction, true),
        )
        .unwrap();

        assert_eq!((chosen.left, chosen.right), (1, 2));
    }

    #[test]
    fn initial_pair_requires_calibrated_rig_config_for_non_trivial_rigs() {
        let mut regular = test_pair(0, 1, 500, 450, 25.0, [1.0, 0.0, 0.0]);
        regular.two_view_config = crate::database::COLMAP_TWO_VIEW_CALIBRATED;
        let mut rig_pair = test_pair(0, 2, 300, 240, 20.0, [1.0, 0.0, 0.0]);
        rig_pair.two_view_config = crate::database::COLMAP_TWO_VIEW_CALIBRATED_RIG;
        let pairs = vec![regular, rig_pair];
        let frames = structureless_frames(3);
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
                data_ids: vec![DataId {
                    sensor_id: ref_sensor,
                    data_id: 1,
                }],
            },
            Frame {
                frame_id: 10,
                rig_id: 3,
                rig_from_world: Rigid3::identity(),
                data_ids: vec![DataId {
                    sensor_id: aux_sensor,
                    data_id: 2,
                }],
            },
            Frame {
                frame_id: 11,
                rig_id: 3,
                rig_from_world: Rigid3::identity(),
                data_ids: Vec::new(),
            },
        ];
        reconstruction.image_frame_indices = vec![Some(0), Some(1), Some(2)];

        let chosen = choose_initial_pair_for_test(
            &pairs,
            &reconstruction,
            &MapperConfig::default(),
            &camera_prior_focal_flags(&reconstruction, true),
        )
        .unwrap();

        assert_eq!((chosen.left, chosen.right), (0, 2));
        assert_eq!(
            chosen.two_view_config,
            crate::database::COLMAP_TWO_VIEW_CALIBRATED_RIG
        );
    }

    #[test]
    fn initial_pair_default_init_trial_limit_matches_colmap() {
        assert_eq!(MapperConfig::default().init_max_reg_trials, 2);
    }

    #[test]
    fn initialization_defaults_match_colmap_pipeline_trials() {
        assert_eq!(MapperConfig::default().init_num_trials, 200);
    }

    #[test]
    fn initialization_relaxation_schedule_matches_colmap_order() {
        let config = MapperConfig {
            init_num_trials: 3,
            init_min_num_inliers: 100,
            init_min_tri_angle_deg: 16.0,
            ..MapperConfig::default()
        };
        let stages = initialization_stage_configs(&config);

        assert_eq!(
            stages.iter().map(|stage| stage.stage).collect::<Vec<_>>(),
            vec![
                InitializationRelaxationStage::Strict,
                InitializationRelaxationStage::MinInliersRelaxed(1),
                InitializationRelaxationStage::MinTriAngleRelaxed(1),
                InitializationRelaxationStage::MinInliersRelaxed(2),
                InitializationRelaxationStage::MinTriAngleRelaxed(2),
            ]
        );
        assert_eq!(
            stages
                .iter()
                .map(|stage| stage.config.init_min_num_inliers)
                .collect::<Vec<_>>(),
            vec![100, 50, 50, 25, 25]
        );
        assert_eq!(
            stages
                .iter()
                .map(|stage| stage.config.init_min_tri_angle_deg)
                .collect::<Vec<_>>(),
            vec![16.0, 16.0, 8.0, 8.0, 4.0]
        );
        assert!(stages.iter().all(|stage| stage.config.init_num_trials == 3));
    }

    #[test]
    fn relaxed_initialization_can_select_pair_rejected_by_strict_inlier_gate() {
        let low_inlier_pair = test_pair(0, 1, 60, 50, 20.0, [1.0, 0.0, 0.0]);
        let pairs = vec![low_inlier_pair];
        let frames = structureless_frames(2);
        let reconstruction = test_reconstruction(&frames);
        let strict_config = MapperConfig {
            init_min_num_inliers: 100,
            ..MapperConfig::default()
        };
        let relaxed_config = MapperConfig {
            init_min_num_inliers: strict_config.init_min_num_inliers / 2,
            ..strict_config.clone()
        };
        let focal_flags = camera_prior_focal_flags(&reconstruction, true);
        let mut session = IncrementalMapperSession::default();
        let mut strict_state = session.initial_pair_selection_state(&reconstruction);

        assert!(choose_initial_pair(
            &pairs,
            &reconstruction,
            &strict_config,
            &focal_flags,
            &mut strict_state,
        )
        .is_none());

        session.commit_initial_pair_selection_state(&reconstruction, &strict_state);
        session.reset_initialization_stats();
        let mut relaxed_state = session.initial_pair_selection_state(&reconstruction);
        let relaxed = choose_initial_pair(
            &pairs,
            &reconstruction,
            &relaxed_config,
            &focal_flags,
            &mut relaxed_state,
        )
        .unwrap();

        assert_eq!((relaxed.left, relaxed.right), (0, 1));
    }

    #[test]
    fn bad_initial_pair_retries_next_pair_in_same_stage_like_colmap() {
        let camera = CameraModel::new_pinhole(200, 160, 80.0, 80.0, 100.0, 80.0);
        let poses = [
            SE3::identity(),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(0.02),
                glam::Vec3::new(-0.35, 0.0, 0.0),
            ),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(-0.03),
                glam::Vec3::new(0.45, 0.0, 0.0),
            ),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(0.04),
                glam::Vec3::new(0.9, 0.0, 0.0),
            ),
        ];
        let points = (0..12)
            .map(|idx| {
                let col = (idx % 4) as f32;
                let row = (idx / 4) as f32;
                [-0.3 + col * 0.2, -0.2 + row * 0.18, 3.0 + idx as f32 * 0.03]
            })
            .collect::<Vec<_>>();
        let mut frames = (0..4)
            .map(|idx| minimal_frame(idx, &format!("image_{idx}.jpg")))
            .collect::<Vec<_>>();
        for (image, pose) in poses.iter().copied().enumerate() {
            frames[image].width = camera.width;
            frames[image].height = camera.height;
            frames[image].keypoints = points
                .iter()
                .map(|&point| project_test_point(camera, pose, point))
                .collect();
            frames[image].colors = vec![[image as u8, 0, 0]; points.len()];
        }
        let bad_pair = test_pair(0, 1, 400, 300, 20.0, [1.0, 0.0, 0.0]);
        let good_pair = initial_pair_from_projected_points(2, 3, poses[2], poses[3], points.len());
        let config = MapperConfig {
            init_num_trials: 2,
            init_min_num_inliers: 4,
            init_min_tri_angle_deg: 0.5,
            min_triangulated: 0,
            abs_pose_min_num_inliers: 4,
            local_ba: false,
            global_ba: false,
            ignore_two_view_tracks: false,
            threads: Some(1),
            ..MapperConfig::default()
        };
        let mut session = IncrementalMapperSession::default();

        let (reconstruction, log) = incremental_map_with_session(
            &frames,
            camera,
            None,
            &[bad_pair, good_pair],
            &config,
            &mut session,
        )
        .expect("second initial pair should recover");

        assert!(log.iter().any(|line| line
            == "initialization_attempt_failed stage=strict trial=0 error=bad initial pair"));
        assert!(log
            .iter()
            .any(|line| line.starts_with("initialization_attempt stage=strict trial=1 ")));
        assert!(log.iter().any(
            |line| line == "initial_pair image_2.jpg -> image_3.jpg inliers=12 triangulated=12"
        ));
        assert!(reconstruction.poses[2].is_some());
        assert!(reconstruction.poses[3].is_some());
        assert!(reconstruction.points.len() >= config.abs_pose_min_num_inliers);
    }

    #[test]
    fn pipeline_keeps_first_small_model_and_discards_later_small_models_like_colmap() {
        let camera = CameraModel::new_pinhole(200, 160, 80.0, 80.0, 100.0, 80.0);
        let poses = [
            SE3::identity(),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(0.03),
                glam::Vec3::new(-0.35, 0.0, 0.0),
            ),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(-0.02),
                glam::Vec3::new(0.45, 0.0, 0.0),
            ),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(0.04),
                glam::Vec3::new(0.9, 0.0, 0.0),
            ),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(-0.01),
                glam::Vec3::new(1.3, 0.0, 0.0),
            ),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(0.02),
                glam::Vec3::new(1.7, 0.0, 0.0),
            ),
        ];
        let points = (0..12)
            .map(|idx| {
                let col = (idx % 4) as f32;
                let row = (idx / 4) as f32;
                [-0.3 + col * 0.2, -0.2 + row * 0.18, 3.0 + idx as f32 * 0.03]
            })
            .collect::<Vec<_>>();
        let mut frames = (0..6)
            .map(|idx| minimal_frame(idx, &format!("image_{idx}.jpg")))
            .collect::<Vec<_>>();
        for (image, pose) in poses.iter().copied().enumerate() {
            frames[image].width = camera.width;
            frames[image].height = camera.height;
            frames[image].keypoints = points
                .iter()
                .map(|&point| project_test_point(camera, pose, point))
                .collect();
            frames[image].colors = vec![[image as u8, 0, 0]; points.len()];
        }
        let pairs = vec![
            initial_pair_from_projected_points(0, 1, poses[0], poses[1], points.len()),
            initial_pair_from_projected_points(2, 3, poses[2], poses[3], points.len()),
        ];
        let config = MapperConfig {
            multiple_models: true,
            max_num_models: 2,
            min_model_size: 3,
            init_num_trials: 1,
            init_min_num_inliers: 4,
            init_min_tri_angle_deg: 0.5,
            min_triangulated: 0,
            abs_pose_min_num_inliers: 4,
            local_ba: false,
            global_ba: false,
            ignore_two_view_tracks: false,
            ..MapperConfig::default()
        };

        let result = incremental_pipeline_map(&frames, camera, None, &pairs, &config)
            .expect("first small model should be kept");

        assert_eq!(result.reconstructions.len(), 1);
        assert!(result.reconstructions[0].poses[0].is_some());
        assert!(result.reconstructions[0].poses[1].is_some());
        assert!(result
            .debug_log
            .iter()
            .any(|line| line.starts_with("pipeline_submodel index=0 status=kept")));
        assert!(result
            .debug_log
            .iter()
            .any(|line| line
                .starts_with("pipeline_submodel index=1 status=discarded_insufficient_size")));
        assert!(result.debug_log.iter().any(|line| line
            .starts_with("pipeline_submodel index=1 status=discarded_insufficient_size")
            && line.contains("total_registered_images=2")));
    }

    #[test]
    fn pipeline_writes_sparse_snapshots_after_registered_frame_frequency_like_colmap() {
        let camera = CameraModel::new_pinhole(200, 160, 80.0, 80.0, 100.0, 80.0);
        let poses = [
            SE3::identity(),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(0.03),
                glam::Vec3::new(-0.35, 0.0, 0.0),
            ),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(-0.02),
                glam::Vec3::new(0.45, 0.0, 0.0),
            ),
        ];
        let points = (0..12)
            .map(|idx| {
                let col = (idx % 4) as f32;
                let row = (idx / 4) as f32;
                [-0.3 + col * 0.2, -0.2 + row * 0.18, 3.0 + idx as f32 * 0.03]
            })
            .collect::<Vec<_>>();
        let mut frames = (0..3)
            .map(|idx| minimal_frame(idx, &format!("image_{idx}.jpg")))
            .collect::<Vec<_>>();
        for (image, pose) in poses.iter().copied().enumerate() {
            frames[image].width = camera.width;
            frames[image].height = camera.height;
            frames[image].keypoints = points
                .iter()
                .map(|&point| project_test_point(camera, pose, point))
                .collect();
            frames[image].colors = vec![[image as u8, 0, 0]; points.len()];
        }
        let pairs = vec![
            initial_pair_from_projected_points(0, 1, poses[0], poses[1], points.len()),
            initial_pair_from_projected_points(0, 2, poses[0], poses[2], points.len()),
            initial_pair_from_projected_points(1, 2, poses[1], poses[2], points.len()),
        ];
        let dir = tempdir().unwrap();
        let config = MapperConfig {
            multiple_models: false,
            snapshot_path: Some(dir.path().join("snapshots")),
            snapshot_frames_freq: 1,
            init_num_trials: 1,
            init_min_num_inliers: 4,
            init_min_tri_angle_deg: 0.5,
            min_triangulated: 0,
            abs_pose_min_num_inliers: 4,
            local_ba: false,
            global_ba: false,
            ..MapperConfig::default()
        };

        let result = incremental_pipeline_map(&frames, camera, None, &pairs, &config)
            .expect("snapshot-enabled pipeline");
        let snapshot_dir = dir.path().join("snapshots").join("0000000001");

        assert_eq!(result.reconstructions.len(), 1);
        assert!(result.reconstructions[0].poses[2].is_some());
        assert!(snapshot_dir.join("cameras.txt").exists());
        assert!(snapshot_dir.join("images.txt").exists());
        assert!(snapshot_dir.join("points3D.txt").exists());
        assert!(result
            .debug_log
            .iter()
            .any(|line| line.starts_with("pipeline_snapshot path=")
                && line.contains("registered_frames=3")));
    }

    #[test]
    fn pipeline_final_global_ba_skips_normalization_like_colmap_final_all() {
        let camera = CameraModel::new_pinhole(200, 160, 80.0, 80.0, 100.0, 80.0);
        let poses = [
            SE3::identity(),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(0.03),
                glam::Vec3::new(-0.35, 0.0, 0.0),
            ),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(-0.02),
                glam::Vec3::new(0.45, 0.0, 0.0),
            ),
        ];
        let points = (0..12)
            .map(|idx| {
                let col = (idx % 4) as f32;
                let row = (idx / 4) as f32;
                [-0.3 + col * 0.2, -0.2 + row * 0.18, 3.0 + idx as f32 * 0.03]
            })
            .collect::<Vec<_>>();
        let mut frames = (0..3)
            .map(|idx| minimal_frame(idx, &format!("image_{idx}.jpg")))
            .collect::<Vec<_>>();
        for (image, pose) in poses.iter().copied().enumerate() {
            frames[image].width = camera.width;
            frames[image].height = camera.height;
            frames[image].keypoints = points
                .iter()
                .map(|&point| project_test_point(camera, pose, point))
                .collect();
            frames[image].colors = vec![[image as u8, 0, 0]; points.len()];
        }
        let pairs = vec![
            initial_pair_from_projected_points(0, 1, poses[0], poses[1], points.len()),
            initial_pair_from_projected_points(0, 2, poses[0], poses[2], points.len()),
            initial_pair_from_projected_points(1, 2, poses[1], poses[2], points.len()),
        ];
        let config = MapperConfig {
            multiple_models: false,
            init_num_trials: 1,
            init_min_num_inliers: 4,
            init_min_tri_angle_deg: 0.5,
            min_triangulated: 0,
            abs_pose_min_num_inliers: 4,
            local_ba: false,
            global_ba: true,
            global_ba_iterations: 3,
            global_ba_max_refinements: 1,
            global_ba_images_freq: 999,
            global_ba_points_freq: 999_999,
            global_ba_images_ratio: 10.0,
            global_ba_points_ratio: 10.0,
            extract_colors: false,
            ..MapperConfig::default()
        };

        let result = incremental_pipeline_map(&frames, camera, None, &pairs, &config)
            .expect("final global BA pipeline");

        assert_eq!(result.reconstructions.len(), 1);
        assert!(result.reconstructions[0].poses.iter().all(Option::is_some));
        assert!(
            result
                .debug_log
                .iter()
                .any(|line| line.starts_with("global_ba reason=initial round=1")),
            "initial incremental global BA should still run: {:?}",
            result.debug_log
        );
        let initial_global_ba = result
            .debug_log
            .iter()
            .find(|line| line.starts_with("global_ba reason=initial round=1"))
            .expect("initial global BA log");
        for field in [
            "solver=",
            "preconditioner=",
            "sparse_backend=",
            "setup_ms=",
            "solve_ms=",
            "postprocess_ms=",
            "ba_elapsed_ms=",
        ] {
            assert!(
                initial_global_ba.contains(field),
                "missing {field} in {initial_global_ba}"
            );
        }
        assert!(
            result
                .debug_log
                .iter()
                .any(|line| line.starts_with("global_ba_normalize reason=initial round=1")),
            "initial incremental global BA should still normalize: {:?}",
            result.debug_log
        );
        assert!(
            result
                .debug_log
                .iter()
                .any(|line| line.starts_with("global_ba reason=final round=1")),
            "final global BA should run after the reconstruction changed: {:?}",
            result.debug_log
        );
        assert!(
            !result
                .debug_log
                .iter()
                .any(|line| line.starts_with("global_ba_normalize reason=final")),
            "COLMAP final-all global BA should not normalize: {:?}",
            result.debug_log
        );
    }

    #[test]
    fn pipeline_callback_events_follow_colmap_controller_order() {
        let camera = CameraModel::new_pinhole(200, 160, 80.0, 80.0, 100.0, 80.0);
        let poses = [
            SE3::identity(),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(0.03),
                glam::Vec3::new(-0.35, 0.0, 0.0),
            ),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(-0.02),
                glam::Vec3::new(0.45, 0.0, 0.0),
            ),
        ];
        let points = (0..12)
            .map(|idx| {
                let col = (idx % 4) as f32;
                let row = (idx / 4) as f32;
                [-0.3 + col * 0.2, -0.2 + row * 0.18, 3.0 + idx as f32 * 0.03]
            })
            .collect::<Vec<_>>();
        let mut frames = (0..3)
            .map(|idx| minimal_frame(idx, &format!("image_{idx}.jpg")))
            .collect::<Vec<_>>();
        for (image, pose) in poses.iter().copied().enumerate() {
            frames[image].width = camera.width;
            frames[image].height = camera.height;
            frames[image].keypoints = points
                .iter()
                .map(|&point| project_test_point(camera, pose, point))
                .collect();
            frames[image].colors = vec![[(image as u8) + 1, 20, 30]; points.len()];
        }
        let pairs = vec![
            initial_pair_from_projected_points(0, 1, poses[0], poses[1], points.len()),
            initial_pair_from_projected_points(0, 2, poses[0], poses[2], points.len()),
            initial_pair_from_projected_points(1, 2, poses[1], poses[2], points.len()),
        ];
        let dir = tempdir().unwrap();
        let config = MapperConfig {
            multiple_models: false,
            snapshot_path: Some(dir.path().join("snapshots")),
            snapshot_frames_freq: 1,
            init_num_trials: 1,
            init_min_num_inliers: 4,
            init_min_tri_angle_deg: 0.5,
            min_triangulated: 0,
            abs_pose_min_num_inliers: 4,
            local_ba: false,
            global_ba: false,
            ..MapperConfig::default()
        };

        let result = incremental_pipeline_map(&frames, camera, None, &pairs, &config)
            .expect("callback event pipeline");
        let initial_color = log_index(
            &result.debug_log,
            "extract_colors image=image_1.jpg images=1 updated_points=0",
        );
        let initial_callback = log_index(&result.debug_log, "callback initial_image_pair_reg");
        let next_color = log_index(
            &result.debug_log,
            "extract_colors image=image_2.jpg images=1 updated_points=0",
        );
        let snapshot = result
            .debug_log
            .iter()
            .position(|line| line.starts_with("pipeline_snapshot path="))
            .expect("snapshot event");
        let next_callback = log_index(&result.debug_log, "callback next_image_reg");
        let submodel = result
            .debug_log
            .iter()
            .position(|line| line.starts_with("pipeline_submodel index=0 status=kept"))
            .expect("submodel event");
        let last_callback = log_index(&result.debug_log, "callback last_image_reg");

        assert!(initial_color < initial_callback);
        assert!(initial_callback < next_color);
        assert!(next_color < snapshot);
        assert!(snapshot < next_callback);
        assert!(submodel < last_callback);
    }

    #[derive(Default)]
    struct CallbackCollector {
        events: Vec<PipelineCallbackEvent>,
    }

    impl PipelineCallbackSink for CallbackCollector {
        fn on_pipeline_callback(&mut self, event: &PipelineCallbackEvent) {
            self.events.push(event.clone());
        }
    }

    #[test]
    fn task_callback_adapter_maps_registration_events_with_monotonic_metadata() {
        use crate::task::{
            SfmTaskContext, SfmTaskControl, SfmTaskEvent, SfmTaskEventKind, SfmTaskOperation,
            SfmTaskStage,
        };

        let control = SfmTaskControl::new();
        let mut task_events = Vec::<SfmTaskEvent>::new();
        {
            let mut sink = |event| task_events.push(event);
            let mut task = SfmTaskContext::new(&control, &mut sink);
            let mut bridge = MapperEventBridge::Task(&mut task);

            for (callback, registered_images, points) in [
                (IncrementalPipelineCallback::InitialImagePairReg, 2, 11),
                (IncrementalPipelineCallback::NextImageReg, 3, 17),
                (IncrementalPipelineCallback::LastImageReg, 3, 17),
            ] {
                bridge.callback(PipelineCallbackEvent {
                    callback,
                    model_index: 0,
                    registered_images,
                    registered_frames: registered_images,
                    points,
                });
            }
        }

        assert_eq!(
            task_events
                .iter()
                .map(|event| event.operation)
                .collect::<Vec<_>>(),
            vec![
                SfmTaskOperation::RegisterInitialPair,
                SfmTaskOperation::RegisterImage,
                SfmTaskOperation::RegisterImage,
            ]
        );
        assert!(task_events
            .windows(2)
            .all(|events| events[0].sequence < events[1].sequence));
        assert!(task_events.iter().all(|event| {
            event.stage == SfmTaskStage::IncrementalMapping
                && event.kind == SfmTaskEventKind::Progress
        }));
        assert_eq!(task_events[0].registered_images, Some(2));
        assert_eq!(task_events[0].sparse_points, Some(11));
        assert_eq!(task_events.last().unwrap().registered_images, Some(3));
        assert_eq!(task_events.last().unwrap().sparse_points, Some(17));
    }

    fn controlled_mapper_fixture() -> (CameraModel, Vec<ImageFrame>, Vec<PairGeometry>) {
        let camera = CameraModel::new_pinhole(200, 160, 80.0, 80.0, 100.0, 80.0);
        let poses = [
            SE3::identity(),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(0.03),
                glam::Vec3::new(-0.35, 0.0, 0.0),
            ),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(-0.02),
                glam::Vec3::new(0.45, 0.0, 0.0),
            ),
        ];
        let points = (0..12)
            .map(|idx| {
                let col = (idx % 4) as f32;
                let row = (idx / 4) as f32;
                [-0.3 + col * 0.2, -0.2 + row * 0.18, 3.0 + idx as f32 * 0.03]
            })
            .collect::<Vec<_>>();
        let mut frames = (0..3)
            .map(|idx| minimal_frame(idx, &format!("image_{idx}.jpg")))
            .collect::<Vec<_>>();
        for (image, pose) in poses.iter().copied().enumerate() {
            frames[image].width = camera.width;
            frames[image].height = camera.height;
            frames[image].keypoints = points
                .iter()
                .map(|&point| project_test_point(camera, pose, point))
                .collect();
            frames[image].colors = vec![[(image as u8) + 1, 20, 30]; points.len()];
        }
        let pairs = vec![
            initial_pair_from_projected_points(0, 1, poses[0], poses[1], points.len()),
            initial_pair_from_projected_points(0, 2, poses[0], poses[2], points.len()),
            initial_pair_from_projected_points(1, 2, poses[1], poses[2], points.len()),
        ];
        (camera, frames, pairs)
    }

    #[test]
    fn controlled_mapper_pauses_after_committed_registration_and_keeps_snapshot_exportable() {
        use crate::task::{
            SfmTaskContext, SfmTaskControl, SfmTaskEvent, SfmTaskEventKind, SfmTaskOperation,
            SfmTaskStage, SfmTaskStop,
        };

        let (camera, frames, pairs) = controlled_mapper_fixture();
        let dir = tempdir().unwrap();
        let snapshot_root = dir.path().join("snapshots");
        let config = MapperConfig {
            multiple_models: false,
            snapshot_path: Some(snapshot_root.clone()),
            snapshot_frames_freq: 1,
            init_num_trials: 1,
            init_min_num_inliers: 4,
            init_min_tri_angle_deg: 0.5,
            min_triangulated: 0,
            abs_pose_min_num_inliers: 4,
            local_ba: false,
            global_ba: false,
            ..MapperConfig::default()
        };
        let control = SfmTaskControl::new();
        let pause = control.clone();
        let mut task_events = Vec::<SfmTaskEvent>::new();
        let error = {
            let mut sink = |event: SfmTaskEvent| {
                if event.operation == SfmTaskOperation::RegisterImage {
                    pause.request_pause();
                }
                task_events.push(event);
            };
            let mut task = SfmTaskContext::new(&control, &mut sink);
            let mut bridge = MapperEventBridge::Task(&mut task);
            incremental_pipeline_map_with_pnp_scorer_and_events(
                &frames,
                camera,
                None,
                &pairs,
                &config,
                &mut bridge,
                None,
            )
            .expect_err("the task should pause after the committed registration")
        };

        assert_eq!(
            error.downcast_ref::<SfmTaskStop>(),
            Some(&SfmTaskStop::Paused)
        );
        let register_event = task_events
            .iter()
            .find(|event| event.operation == SfmTaskOperation::RegisterImage)
            .expect("registered image event");
        assert_eq!(register_event.registered_images, Some(3));
        assert_eq!(register_event.sparse_points, Some(12));
        assert_eq!(
            task_events
                .iter()
                .filter(|event| {
                    event.stage == SfmTaskStage::Export
                        && event.operation == SfmTaskOperation::WriteArtifacts
                })
                .map(|event| event.kind)
                .collect::<Vec<_>>(),
            vec![SfmTaskEventKind::Started, SfmTaskEventKind::Completed]
        );

        let snapshot = snapshot_root.join("0000000001");
        let exported = read_colmap_sparse_model(&snapshot).expect("committed snapshot is readable");
        assert_eq!(
            exported
                .reconstruction
                .poses
                .iter()
                .filter(|pose| pose.is_some())
                .count(),
            3
        );
        assert_eq!(exported.reconstruction.points.len(), 12);
    }

    #[test]
    fn controlled_mapper_reports_local_and_global_ba_operation_boundaries() {
        use crate::task::{
            SfmTaskContext, SfmTaskControl, SfmTaskEvent, SfmTaskEventKind, SfmTaskOperation,
        };

        let (camera, frames, pairs) = controlled_mapper_fixture();
        let config = MapperConfig {
            multiple_models: false,
            init_num_trials: 1,
            init_min_num_inliers: 4,
            init_min_tri_angle_deg: 0.5,
            min_triangulated: 0,
            abs_pose_min_num_inliers: 4,
            local_ba: true,
            local_ba_num_images: 2,
            local_ba_min_shared_points: 4,
            local_ba_iterations: 1,
            local_ba_max_refinements: 1,
            global_ba: true,
            global_ba_iterations: 1,
            global_ba_max_refinements: 1,
            global_ba_images_freq: 1,
            global_ba_points_freq: 1,
            extract_colors: false,
            ..MapperConfig::default()
        };
        let control = SfmTaskControl::new();
        let mut task_events = Vec::<SfmTaskEvent>::new();
        {
            let mut sink = |event| task_events.push(event);
            let mut task = SfmTaskContext::new(&control, &mut sink);
            let mut bridge = MapperEventBridge::Task(&mut task);
            incremental_pipeline_map_with_pnp_scorer_and_events(
                &frames,
                camera,
                None,
                &pairs,
                &config,
                &mut bridge,
                None,
            )
            .expect("controlled BA pipeline");
        }

        for operation in [
            SfmTaskOperation::LocalBundleAdjustment,
            SfmTaskOperation::GlobalBundleAdjustment,
        ] {
            let boundary_kinds = task_events
                .iter()
                .filter(|event| event.operation == operation)
                .map(|event| event.kind)
                .collect::<Vec<_>>();
            assert!(!boundary_kinds.is_empty(), "missing {operation:?} events");
            assert_eq!(boundary_kinds.len() % 2, 0, "unpaired {operation:?} events");
            assert!(boundary_kinds
                .chunks_exact(2)
                .all(|pair| { pair == [SfmTaskEventKind::Started, SfmTaskEventKind::Completed] }));
        }
    }

    #[test]
    fn pipeline_callback_sink_receives_colmap_controller_events_with_payloads() {
        let camera = CameraModel::new_pinhole(200, 160, 80.0, 80.0, 100.0, 80.0);
        let poses = [
            SE3::identity(),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(0.03),
                glam::Vec3::new(-0.35, 0.0, 0.0),
            ),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(-0.02),
                glam::Vec3::new(0.45, 0.0, 0.0),
            ),
        ];
        let points = (0..12)
            .map(|idx| {
                let col = (idx % 4) as f32;
                let row = (idx / 4) as f32;
                [-0.3 + col * 0.2, -0.2 + row * 0.18, 3.0 + idx as f32 * 0.03]
            })
            .collect::<Vec<_>>();
        let mut frames = (0..3)
            .map(|idx| minimal_frame(idx, &format!("image_{idx}.jpg")))
            .collect::<Vec<_>>();
        for (image, pose) in poses.iter().copied().enumerate() {
            frames[image].width = camera.width;
            frames[image].height = camera.height;
            frames[image].keypoints = points
                .iter()
                .map(|&point| project_test_point(camera, pose, point))
                .collect();
            frames[image].colors = vec![[(image as u8) + 1, 20, 30]; points.len()];
        }
        let pairs = vec![
            initial_pair_from_projected_points(0, 1, poses[0], poses[1], points.len()),
            initial_pair_from_projected_points(0, 2, poses[0], poses[2], points.len()),
            initial_pair_from_projected_points(1, 2, poses[1], poses[2], points.len()),
        ];
        let config = MapperConfig {
            multiple_models: false,
            init_num_trials: 1,
            init_min_num_inliers: 4,
            init_min_tri_angle_deg: 0.5,
            min_triangulated: 0,
            abs_pose_min_num_inliers: 4,
            local_ba: false,
            global_ba: false,
            ..MapperConfig::default()
        };
        let mut collector = CallbackCollector::default();

        let mut events = MapperEventBridge::Legacy(&mut collector);
        let result = incremental_pipeline_map_with_pnp_scorer_and_events(
            &frames,
            camera,
            None,
            &pairs,
            &config,
            &mut events,
            None,
        )
        .expect("callback sink pipeline");

        assert_eq!(
            collector
                .events
                .iter()
                .map(|event| event.callback)
                .collect::<Vec<_>>(),
            vec![
                IncrementalPipelineCallback::InitialImagePairReg,
                IncrementalPipelineCallback::NextImageReg,
                IncrementalPipelineCallback::LastImageReg,
            ]
        );
        assert_eq!(collector.events[0].model_index, 0);
        assert_eq!(collector.events[0].registered_images, 2);
        assert_eq!(collector.events[0].registered_frames, 2);
        assert_eq!(collector.events[0].points, 12);
        assert_eq!(collector.events[1].registered_images, 3);
        assert_eq!(collector.events[1].registered_frames, 3);
        assert_eq!(collector.events[1].points, 12);
        assert_eq!(collector.events[2].registered_images, 3);
        assert_eq!(collector.events[2].registered_frames, 3);
        assert_eq!(collector.events[2].points, 12);
        assert_eq!(result.reconstructions.len(), 1);
        assert!(result.reconstructions[0].poses.iter().all(Option::is_some));
        assert!(result
            .debug_log
            .iter()
            .any(|line| line == "callback initial_image_pair_reg"));
        assert!(result
            .debug_log
            .iter()
            .any(|line| line == "callback next_image_reg"));
        assert!(result
            .debug_log
            .iter()
            .any(|line| line == "callback last_image_reg"));
    }

    #[test]
    fn pipeline_extracts_colors_after_each_successful_registration_like_colmap() {
        let camera = CameraModel::new_pinhole(200, 160, 80.0, 80.0, 100.0, 80.0);
        let poses = [
            SE3::identity(),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(0.03),
                glam::Vec3::new(-0.35, 0.0, 0.0),
            ),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(-0.02),
                glam::Vec3::new(0.45, 0.0, 0.0),
            ),
        ];
        let points = (0..12)
            .map(|idx| {
                let col = (idx % 4) as f32;
                let row = (idx / 4) as f32;
                [-0.3 + col * 0.2, -0.2 + row * 0.18, 3.0 + idx as f32 * 0.03]
            })
            .collect::<Vec<_>>();
        let mut frames = (0..3)
            .map(|idx| minimal_frame(idx, &format!("image_{idx}.jpg")))
            .collect::<Vec<_>>();
        for (image, pose) in poses.iter().copied().enumerate() {
            frames[image].width = camera.width;
            frames[image].height = camera.height;
            frames[image].keypoints = points
                .iter()
                .map(|&point| project_test_point(camera, pose, point))
                .collect();
            frames[image].colors = vec![[(image as u8) + 1, 20, 30]; points.len()];
        }
        let pairs = vec![
            initial_pair_from_projected_points(0, 1, poses[0], poses[1], points.len()),
            initial_pair_from_projected_points(0, 2, poses[0], poses[2], points.len()),
            initial_pair_from_projected_points(1, 2, poses[1], poses[2], points.len()),
        ];
        let config = MapperConfig {
            init_num_trials: 1,
            init_min_num_inliers: 4,
            init_min_tri_angle_deg: 0.5,
            min_triangulated: 0,
            abs_pose_min_num_inliers: 4,
            local_ba: false,
            global_ba: false,
            ..MapperConfig::default()
        };

        let result = incremental_pipeline_map(&frames, camera, None, &pairs, &config)
            .expect("color extraction pipeline");

        assert_eq!(result.reconstructions.len(), 1);
        assert!(result.reconstructions[0].poses.iter().all(Option::is_some));
        assert!(result
            .debug_log
            .iter()
            .any(|line| line == "extract_colors image=image_0.jpg images=1 updated_points=12"));
        assert!(result
            .debug_log
            .iter()
            .any(|line| { line == "extract_colors image=image_1.jpg images=1 updated_points=0" }));
        assert!(result
            .debug_log
            .iter()
            .any(|line| { line == "extract_colors image=image_2.jpg images=1 updated_points=0" }));
        assert!(result
            .debug_log
            .iter()
            .any(|line| line == "extract_colors_all images=3 updated_points=12"));
        assert!(result.reconstructions[0]
            .points
            .iter()
            .all(|point| point.color == [2, 20, 30]));
    }

    #[test]
    fn seeded_reconstruction_continues_without_initial_pair_like_colmap() {
        let camera = CameraModel::new_pinhole(200, 160, 80.0, 80.0, 100.0, 80.0);
        let poses = [
            SE3::identity(),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(0.03),
                glam::Vec3::new(-0.35, 0.0, 0.0),
            ),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(-0.02),
                glam::Vec3::new(0.45, 0.0, 0.0),
            ),
        ];
        let points = (0..12)
            .map(|idx| {
                let col = (idx % 4) as f32;
                let row = (idx / 4) as f32;
                [-0.3 + col * 0.2, -0.2 + row * 0.18, 3.0 + idx as f32 * 0.03]
            })
            .collect::<Vec<_>>();
        let mut frames = (0..3)
            .map(|idx| minimal_frame(idx, &format!("image_{idx}.jpg")))
            .collect::<Vec<_>>();
        for (image, pose) in poses.iter().copied().enumerate() {
            frames[image].width = camera.width;
            frames[image].height = camera.height;
            frames[image].keypoints = points
                .iter()
                .map(|&point| project_test_point(camera, pose, point))
                .collect();
            frames[image].colors = vec![[image as u8, 0, 0]; points.len()];
        }

        let seed_points = points
            .iter()
            .enumerate()
            .map(|(idx, &xyz)| Point3D {
                xyz,
                color: [idx as u8, 1, 2],
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
            })
            .collect::<Vec<_>>();
        let mut seed_observations =
            vec![vec![None; points.len()], vec![None; points.len()], vec![]];
        for idx in 0..points.len() {
            seed_observations[0][idx] = Some(idx);
            seed_observations[1][idx] = Some(idx);
        }
        let setup = ReferenceCameraSetup {
            cameras: vec![camera],
            camera_ids: vec![1],
            camera_has_prior_focal_length: vec![true],
            rigs: Vec::new(),
            frames: Vec::new(),
            image_ids: vec![1, 2, 3],
            image_camera_indices: vec![0, 0, 0],
            image_frame_indices: vec![None, None, None],
            seed_reconstruction: Some(ReconstructionSeed {
                poses: vec![Some(poses[0]), Some(poses[1]), None],
                observations: seed_observations,
                point_ids: (0..points.len()).map(|idx| idx as u64 + 100).collect(),
                points: seed_points,
            }),
        };
        let pairs = vec![
            initial_pair_from_projected_points(0, 1, poses[0], poses[1], points.len()),
            initial_pair_from_projected_points(0, 2, poses[0], poses[2], points.len()),
            initial_pair_from_projected_points(1, 2, poses[1], poses[2], points.len()),
        ];
        let config = MapperConfig {
            init_num_trials: 1,
            init_min_num_inliers: 4,
            init_min_tri_angle_deg: 0.5,
            min_triangulated: 0,
            abs_pose_min_num_inliers: 4,
            local_ba: false,
            global_ba: false,
            ..MapperConfig::default()
        };
        let mut session = IncrementalMapperSession::default();
        let mut events = MapperEventBridge::Silent;

        let (reconstruction, log) = incremental_map_single_attempt(
            &frames,
            camera,
            Some(&setup),
            &pairs,
            &config,
            &mut session,
            0,
            &mut events,
        )
        .expect("seeded reconstruction should continue");

        assert!(log
            .iter()
            .any(|line| line == "continue_reconstruction registered_images=2 points=12"));
        assert!(!log.iter().any(|line| line.starts_with("initial_pair ")));
        assert!(log
            .iter()
            .any(|line| line.starts_with("register image_2.jpg source=pnp")));
        let maintenance = log
            .iter()
            .find(|line| line.starts_with("sparse_maintenance "))
            .expect("sparse maintenance telemetry");
        assert!(
            maintenance.contains("full_filter_calls=0")
                && maintenance.contains("subset_filter_calls=1"),
            "{maintenance}"
        );
        assert!(reconstruction.poses.iter().all(Option::is_some));
        assert_eq!(reconstruction.point_ids[0], 100);
        assert!(reconstruction
            .points
            .iter()
            .any(|point| point.track.iter().any(|obs| obs.image == 2)));
    }

    #[test]
    fn continuation_seed_is_only_reused_for_first_trial_like_colmap_manager_index_zero() {
        let camera = CameraModel::new_pinhole(200, 160, 80.0, 80.0, 100.0, 80.0);
        let poses = [
            SE3::identity(),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(0.03),
                glam::Vec3::new(-0.35, 0.0, 0.0),
            ),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(-0.02),
                glam::Vec3::new(0.45, 0.0, 0.0),
            ),
        ];
        let points = (0..12)
            .map(|idx| {
                let col = (idx % 4) as f32;
                let row = (idx / 4) as f32;
                [-0.3 + col * 0.2, -0.2 + row * 0.18, 3.0 + idx as f32 * 0.03]
            })
            .collect::<Vec<_>>();
        let mut frames = (0..3)
            .map(|idx| minimal_frame(idx, &format!("image_{idx}.jpg")))
            .collect::<Vec<_>>();
        for (image, pose) in poses.iter().copied().enumerate() {
            frames[image].width = camera.width;
            frames[image].height = camera.height;
            frames[image].keypoints = points
                .iter()
                .map(|&point| project_test_point(camera, pose, point))
                .collect();
            frames[image].colors = vec![[image as u8, 0, 0]; points.len()];
        }
        let setup = ReferenceCameraSetup {
            cameras: vec![camera],
            camera_ids: vec![1],
            camera_has_prior_focal_length: vec![true],
            rigs: Vec::new(),
            frames: Vec::new(),
            image_ids: vec![1, 2, 3],
            image_camera_indices: vec![0, 0, 0],
            image_frame_indices: vec![None, None, None],
            seed_reconstruction: Some(ReconstructionSeed {
                poses: vec![Some(poses[0]), None, None],
                observations: vec![vec![Some(0)], vec![], vec![]],
                point_ids: vec![500],
                points: vec![Point3D {
                    xyz: points[0],
                    color: [9, 9, 9],
                    error: 0.0,
                    track: vec![TrackObservation {
                        image: 0,
                        feature: 0,
                    }],
                }],
            }),
        };
        let pairs = vec![initial_pair_from_projected_points(
            1,
            2,
            poses[1],
            poses[2],
            points.len(),
        )];
        let config = MapperConfig {
            multiple_models: true,
            max_num_models: 2,
            min_model_size: 2,
            init_num_trials: 2,
            init_min_num_inliers: 4,
            init_min_tri_angle_deg: 0.5,
            min_triangulated: 0,
            abs_pose_min_num_inliers: 1,
            local_ba: false,
            global_ba: false,
            ignore_two_view_tracks: false,
            ..MapperConfig::default()
        };

        let result = incremental_pipeline_map(&frames, camera, Some(&setup), &pairs, &config)
            .expect("seeded continuation followed by new submodel");

        assert_eq!(result.reconstructions.len(), 2);
        assert_eq!(
            result
                .debug_log
                .iter()
                .filter(|line| line.starts_with("continue_reconstruction "))
                .count(),
            1
        );
        assert!(result.debug_log.iter().any(
            |line| line == "initial_pair image_1.jpg -> image_2.jpg inliers=12 triangulated=12"
        ));
        assert!(result.reconstructions[0].poses[0].is_some());
        assert!(result.reconstructions[0].poses[1].is_none());
        assert!(result.reconstructions[1].poses[0].is_none());
        assert!(result.reconstructions[1].poses[1].is_some());
        assert!(result.reconstructions[1].poses[2].is_some());
    }

    #[test]
    fn rig_frame_continuation_seed_is_only_reused_for_first_trial_like_colmap() {
        let camera = CameraModel::new_pinhole(200, 160, 80.0, 80.0, 100.0, 80.0);
        let poses = [
            SE3::identity(),
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(-0.35, 0.0, 0.0)),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(0.03),
                glam::Vec3::new(0.45, 0.0, 0.0),
            ),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(-0.02),
                glam::Vec3::new(0.9, 0.0, 0.0),
            ),
        ];
        let points = (0..12)
            .map(|idx| {
                let col = (idx % 4) as f32;
                let row = (idx / 4) as f32;
                [-0.3 + col * 0.2, -0.2 + row * 0.18, 3.0 + idx as f32 * 0.03]
            })
            .collect::<Vec<_>>();
        let mut frames = (0..4)
            .map(|idx| minimal_frame(idx, &format!("image_{idx}.jpg")))
            .collect::<Vec<_>>();
        for (image, pose) in poses.iter().copied().enumerate() {
            frames[image].width = camera.width;
            frames[image].height = camera.height;
            frames[image].keypoints = points
                .iter()
                .map(|&point| project_test_point(camera, pose, point))
                .collect();
            frames[image].colors = vec![[image as u8, 0, 0]; points.len()];
        }
        let ref_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 1,
        };
        let aux_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: 2,
        };
        let setup = ReferenceCameraSetup {
            cameras: vec![camera],
            camera_ids: vec![1],
            camera_has_prior_focal_length: vec![true],
            rigs: vec![Rig {
                rig_id: 10,
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
                            tvec: [0.35, 0.0, 0.0],
                        }),
                    },
                ],
            }],
            frames: vec![Frame {
                frame_id: 20,
                rig_id: 10,
                rig_from_world: Rigid3::identity(),
                data_ids: vec![
                    DataId {
                        sensor_id: ref_sensor,
                        data_id: 1,
                    },
                    DataId {
                        sensor_id: aux_sensor,
                        data_id: 2,
                    },
                ],
            }],
            image_ids: vec![1, 2, 3, 4],
            image_camera_indices: vec![0, 0, 0, 0],
            image_frame_indices: vec![Some(0), Some(0), None, None],
            seed_reconstruction: Some(ReconstructionSeed {
                poses: vec![Some(poses[0]), Some(poses[1]), None, None],
                observations: vec![vec![Some(0)], vec![Some(0)], vec![], vec![]],
                point_ids: vec![700],
                points: vec![Point3D {
                    xyz: points[0],
                    color: [7, 7, 7],
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
                }],
            }),
        };
        let pairs = vec![initial_pair_from_projected_points(
            2,
            3,
            poses[2],
            poses[3],
            points.len(),
        )];
        let config = MapperConfig {
            multiple_models: true,
            max_num_models: 2,
            min_model_size: 3,
            init_num_trials: 2,
            init_min_num_inliers: 4,
            init_min_tri_angle_deg: 0.5,
            min_triangulated: 0,
            abs_pose_min_num_inliers: 1,
            local_ba: false,
            global_ba: false,
            ignore_two_view_tracks: false,
            ..MapperConfig::default()
        };

        let result = incremental_pipeline_map(&frames, camera, Some(&setup), &pairs, &config)
            .expect("seeded rig-frame continuation followed by new submodel");

        assert_eq!(result.reconstructions.len(), 2);
        assert_eq!(
            result
                .debug_log
                .iter()
                .filter(|line| line.starts_with("continue_reconstruction "))
                .count(),
            1
        );
        assert!(result.debug_log.iter().any(
            |line| line == "initial_pair image_2.jpg -> image_3.jpg inliers=12 triangulated=12"
        ));
        assert_eq!(registered_frame_count(&result.reconstructions[0]), 1);
        assert!(result.reconstructions[0].poses[0].is_some());
        assert!(result.reconstructions[0].poses[1].is_some());
        assert!(result.reconstructions[1].poses[0].is_none());
        assert!(result.reconstructions[1].poses[1].is_none());
        assert!(result.reconstructions[1].poses[2].is_some());
        assert!(result.reconstructions[1].poses[3].is_some());
    }

    #[test]
    fn real_colmap_rig_frame_seed_continuation_uses_sparse_fixture_like_colmap() -> Result<()> {
        let dir = tempdir()?;
        let sparse = dir.path().join("sparse/0");
        let (camera, poses, points, frames, sparse_model) = colmap_rig_frame_seed_fixture();
        write_colmap_sparse_text(&sparse, &sparse_model)?;

        let setup = reference_camera_setup(
            dir.path(),
            &frames.iter().map(|f| f.path.clone()).collect::<Vec<_>>(),
        )?;
        let seed = setup
            .seed_reconstruction
            .as_ref()
            .expect("COLMAP sparse reference should seed existing reconstruction");
        assert_eq!(setup.camera_ids, vec![11]);
        assert_eq!(setup.image_ids, vec![101, 205, 3, 4]);
        assert_eq!(
            setup.image_frame_indices,
            vec![Some(0), Some(0), None, None]
        );
        assert_eq!(seed.point_ids, vec![700]);
        assert_eq!(seed.points[0].track.len(), 2);

        let pairs = vec![initial_pair_from_projected_points(
            2,
            3,
            poses[2],
            poses[3],
            points.len(),
        )];
        let config = MapperConfig {
            multiple_models: true,
            max_num_models: 2,
            min_model_size: 3,
            init_num_trials: 2,
            init_min_num_inliers: 4,
            init_min_tri_angle_deg: 0.5,
            min_triangulated: 0,
            abs_pose_min_num_inliers: 1,
            local_ba: false,
            global_ba: false,
            ignore_two_view_tracks: false,
            ..MapperConfig::default()
        };

        let result = incremental_pipeline_map(&frames, camera, Some(&setup), &pairs, &config)?;

        assert_eq!(result.reconstructions.len(), 2);
        assert_eq!(
            result
                .debug_log
                .iter()
                .filter(|line| line.starts_with("continue_reconstruction "))
                .count(),
            1
        );
        assert_eq!(registered_frame_count(&result.reconstructions[0]), 1);
        assert_eq!(result.reconstructions[0].image_ids[0], 101);
        assert_eq!(result.reconstructions[0].image_ids[1], 205);
        assert!(result.reconstructions[0].poses[0].is_some());
        assert!(result.reconstructions[0].poses[1].is_some());
        assert_eq!(result.reconstructions[0].point_ids, vec![700]);
        assert!(result.reconstructions[1].poses[0].is_none());
        assert!(result.reconstructions[1].poses[1].is_none());
        assert!(result.reconstructions[1].poses[2].is_some());
        assert!(result.reconstructions[1].poses[3].is_some());
        Ok(())
    }

    #[test]
    fn real_colmap_rig_frame_fix_existing_frames_protects_sparse_seed() -> Result<()> {
        let dir = tempdir()?;
        let sparse = dir.path().join("sparse/0");
        let (camera, _, _, seed_frames, sparse_model) = colmap_rig_frame_seed_fixture();
        write_colmap_sparse_text(&sparse, &sparse_model)?;
        let setup = reference_camera_setup(
            dir.path(),
            &seed_frames
                .iter()
                .map(|f| f.path.clone())
                .collect::<Vec<_>>(),
        )?;
        let seed = setup
            .seed_reconstruction
            .clone()
            .expect("COLMAP sparse reference should seed existing reconstruction");
        let mut frames = seed_frames;
        frames.extend((4..21).map(|idx| minimal_frame(idx, &format!("extra_{idx}.jpg"))));
        let mut expanded_setup = setup.clone();
        expanded_setup.image_ids.extend(5..22);
        expanded_setup.image_camera_indices.extend(vec![0; 17]);
        expanded_setup.image_frame_indices.extend(vec![None; 17]);
        expanded_setup.seed_reconstruction = Some(seed);
        let config = MapperConfig {
            fix_existing_frames: true,
            ..MapperConfig::default()
        };
        let mut reconstruction =
            reconstruction_from_reference_setup_for_test(&frames, camera, &expanded_setup);
        for image in 2..reconstruction.poses.len() {
            reconstruction.poses[image] = Some(SE3::identity());
        }
        let mut stats = registration_stats(&reconstruction);
        let mut begin_reconstruction = reconstruction.clone();
        for image in 2..begin_reconstruction.poses.len() {
            begin_reconstruction.poses[image] = None;
        }
        stats.set_existing_registration_units_from_reconstruction(&begin_reconstruction);
        let options = mapper_local_ba_options(
            &config,
            &reconstruction,
            &stats,
            5,
            (0..reconstruction.poses.len()).collect(),
            Vec::new(),
            None,
            None,
        );
        let mut filtered_units = HashSet::new();
        let mut tri_state = IncrementalTriangulatorState::new(&frames, &[], &reconstruction);

        let filtered = filter_registered_frames(
            &frames,
            &[],
            &mut reconstruction,
            &config,
            &mut stats,
            Some(&mut filtered_units),
            &mut tri_state,
        );

        assert_eq!(options.constant_images, vec![0, 1]);
        assert_eq!(filtered, 19);
        assert!(reconstruction.poses[0].is_some());
        assert!(reconstruction.poses[1].is_some());
        assert!(reconstruction.poses[2..].iter().all(Option::is_none));
        assert!(!filtered_units.contains(&RegistrationUnitKey::Frame(0)));
        assert_eq!(registered_frame_count(&reconstruction), 1);
        Ok(())
    }

    #[test]
    fn initial_pair_skips_first_images_that_reached_init_trial_limit() {
        let strong_exhausted = test_pair(0, 1, 600, 500, 30.0, [1.0, 0.0, 0.0]);
        let weaker_available = test_pair(2, 3, 180, 140, 18.0, [1.0, 0.0, 0.0]);
        let pairs = vec![strong_exhausted, weaker_available];
        let frames = structureless_frames(4);
        let reconstruction = test_reconstruction(&frames);
        let mut selection_state = InitialPairSelectionState::from_reconstruction(&reconstruction);
        selection_state.init_num_reg_trials[0] = MapperConfig::default().init_max_reg_trials;
        selection_state.init_num_reg_trials[1] = MapperConfig::default().init_max_reg_trials;
        let config = MapperConfig {
            threads: Some(1),
            ..MapperConfig::default()
        };

        let chosen = choose_initial_pair(
            &pairs,
            &reconstruction,
            &config,
            &camera_prior_focal_flags(&reconstruction, true),
            &mut selection_state,
        )
        .unwrap();

        assert_eq!((chosen.left, chosen.right), (2, 3));
    }

    #[test]
    fn initial_pair_skips_images_registered_in_other_reconstruction() {
        let registered_strong = test_pair(0, 1, 600, 500, 30.0, [1.0, 0.0, 0.0]);
        let available = test_pair(2, 3, 180, 140, 18.0, [1.0, 0.0, 0.0]);
        let pairs = vec![registered_strong, available];
        let frames = structureless_frames(4);
        let reconstruction = test_reconstruction(&frames);
        let mut selection_state = InitialPairSelectionState::from_reconstruction(&reconstruction);
        selection_state.num_registrations[0] = 1;

        let chosen = choose_initial_pair(
            &pairs,
            &reconstruction,
            &MapperConfig::default(),
            &camera_prior_focal_flags(&reconstruction, true),
            &mut selection_state,
        )
        .unwrap();

        assert_eq!((chosen.left, chosen.right), (2, 3));
    }

    #[test]
    fn initial_pair_suppresses_already_tried_pairs() {
        let already_tried = test_pair(0, 1, 600, 500, 30.0, [1.0, 0.0, 0.0]);
        let available = test_pair(2, 3, 180, 140, 18.0, [1.0, 0.0, 0.0]);
        let pairs = vec![already_tried, available];
        let frames = structureless_frames(4);
        let reconstruction = test_reconstruction(&frames);
        let mut selection_state = InitialPairSelectionState::from_reconstruction(&reconstruction);
        selection_state.mark_initial_pair_tried(&reconstruction, 0, 1);

        let chosen = choose_initial_pair(
            &pairs,
            &reconstruction,
            &MapperConfig::default(),
            &camera_prior_focal_flags(&reconstruction, true),
            &mut selection_state,
        )
        .unwrap();

        assert_eq!((chosen.left, chosen.right), (2, 3));
    }

    #[test]
    fn initial_pair_session_keeps_tried_pairs_across_reconstruction_attempts() {
        let first_attempt_best = test_pair(0, 1, 600, 500, 30.0, [1.0, 0.0, 0.0]);
        let second_attempt_pair = test_pair(2, 3, 180, 140, 18.0, [1.0, 0.0, 0.0]);
        let pairs = vec![first_attempt_best, second_attempt_pair];
        let frames = structureless_frames(4);
        let reconstruction = test_reconstruction(&frames);
        let config = MapperConfig {
            threads: Some(1),
            ..MapperConfig::default()
        };
        let focal_flags = camera_prior_focal_flags(&reconstruction, true);
        let mut session = IncrementalMapperSession::default();

        let mut first_state = session.initial_pair_selection_state(&reconstruction);
        let first = choose_initial_pair(
            &pairs,
            &reconstruction,
            &config,
            &focal_flags,
            &mut first_state,
        )
        .unwrap();
        first_state.register_initial_pair(&reconstruction, first.left, first.right);
        session.commit_initial_pair_selection_state(&reconstruction, &first_state);

        let mut second_state = session.initial_pair_selection_state(&reconstruction);
        let second = choose_initial_pair(
            &pairs,
            &reconstruction,
            &config,
            &focal_flags,
            &mut second_state,
        )
        .unwrap();

        assert_eq!((first.left, first.right), (0, 1));
        assert_eq!((second.left, second.right), (2, 3));
    }

    #[test]
    fn initial_pair_session_reset_matches_colmap_relaxation_reset() {
        let first_attempt_best = test_pair(0, 1, 600, 500, 30.0, [1.0, 0.0, 0.0]);
        let second_attempt_pair = test_pair(2, 3, 180, 140, 18.0, [1.0, 0.0, 0.0]);
        let pairs = vec![first_attempt_best, second_attempt_pair];
        let frames = structureless_frames(4);
        let reconstruction = test_reconstruction(&frames);
        let config = MapperConfig {
            threads: Some(1),
            ..MapperConfig::default()
        };
        let focal_flags = camera_prior_focal_flags(&reconstruction, true);
        let mut session = IncrementalMapperSession::default();

        let mut first_state = session.initial_pair_selection_state(&reconstruction);
        let first = choose_initial_pair(
            &pairs,
            &reconstruction,
            &config,
            &focal_flags,
            &mut first_state,
        )
        .unwrap();
        first_state.register_initial_pair(&reconstruction, first.left, first.right);
        session.commit_initial_pair_selection_state(&reconstruction, &first_state);

        session.reset_initialization_stats();
        let mut reset_state = session.initial_pair_selection_state(&reconstruction);
        let reset_choice = choose_initial_pair(
            &pairs,
            &reconstruction,
            &config,
            &focal_flags,
            &mut reset_state,
        )
        .unwrap();

        assert_eq!((reset_choice.left, reset_choice.right), (0, 1));
    }

    #[test]
    fn initial_pair_session_counts_kept_and_discarded_registrations_like_colmap() {
        let frames = structureless_frames(2);
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::identity());
        let mut session = IncrementalMapperSession::default();

        session.end_reconstruction(&reconstruction, false);
        let kept_state = session.initial_pair_selection_state(&reconstruction);
        assert_eq!(kept_state.num_registrations, vec![1, 1]);

        session.end_reconstruction(&reconstruction, true);
        let discarded_state = session.initial_pair_selection_state(&reconstruction);
        assert_eq!(discarded_state.num_registrations, vec![0, 0]);
    }

    #[test]
    fn initial_pair_session_shared_images_are_current_submodel_only_like_colmap() {
        let frames = structureless_frames(5);
        let mut first = test_reconstruction(&frames);
        first.poses[0] = Some(SE3::identity());
        first.poses[1] = Some(SE3::identity());
        let mut session = IncrementalMapperSession::default();
        session.begin_reconstruction(&first);
        assert_eq!(session.num_shared_registered_image_events(), 0);
        session.end_reconstruction(&first, false);
        assert_eq!(session.num_total_registered_images(), 2);
        assert_eq!(session.num_shared_registered_image_events(), 0);

        let mut second = test_reconstruction(&frames);
        second.poses[0] = Some(SE3::identity());
        second.poses[2] = Some(SE3::identity());
        session.begin_reconstruction(&second);
        assert_eq!(session.num_shared_registered_image_events(), 1);
        session.end_reconstruction(&second, false);
        assert_eq!(session.num_total_registered_images(), 3);
        assert_eq!(session.num_shared_registered_image_events(), 1);

        let mut third = test_reconstruction(&frames);
        third.poses[3] = Some(SE3::identity());
        third.poses[4] = Some(SE3::identity());
        session.begin_reconstruction(&third);
        assert_eq!(session.num_shared_registered_image_events(), 0);
        session.end_reconstruction(&third, false);
        assert_eq!(session.num_total_registered_images(), 5);
        assert_eq!(session.num_shared_registered_image_events(), 0);

        let mut fourth = test_reconstruction(&frames);
        fourth.poses[0] = Some(SE3::identity());
        fourth.poses[3] = Some(SE3::identity());
        session.begin_reconstruction(&fourth);
        assert_eq!(session.num_shared_registered_image_events(), 2);
        session.end_reconstruction(&fourth, true);
        assert_eq!(session.num_total_registered_images(), 5);
        assert_eq!(session.num_shared_registered_image_events(), 0);
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
            frame_image_poses: Vec::new(),
            generalized_inliers: Vec::new(),
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
            frame_image_poses: Vec::new(),
            generalized_inliers: Vec::new(),
        };

        assert!(registration_score(&strong, &manager) > registration_score(&weak, &manager));
    }

    #[test]
    fn registration_rank_matches_colmap_image_selection_methods() {
        let mut frames = vec![
            minimal_frame(0, "seed.jpg"),
            minimal_frame(1, "many_visible.jpg"),
            minimal_frame(2, "high_ratio.jpg"),
        ];
        frames[0].keypoints = (0..8)
            .map(|idx| rustslam::KeyPoint::new(idx as f32, 0.0))
            .collect();
        frames[1].keypoints = (0..8)
            .map(|idx| rustslam::KeyPoint::new(idx as f32, 1.0))
            .collect();
        frames[2].keypoints = (0..4)
            .map(|idx| rustslam::KeyPoint::new(idx as f32, 2.0))
            .collect();
        let pairs = vec![
            pair_with_inliers(
                0,
                1,
                &[
                    (0, 0),
                    (1, 1),
                    (2, 2),
                    (3, 3),
                    (4, 4),
                    (5, 5),
                    (6, 6),
                    (7, 7),
                ],
            ),
            pair_with_inliers(0, 2, &[(0, 0), (1, 1)]),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        for idx in 0..4 {
            reconstruction.observations[0][idx] = Some(idx);
            reconstruction.point_ids.push(idx as u64 + 1);
            reconstruction.points.push(Point3D {
                xyz: [idx as f32 * 0.1, 0.0, 2.0],
                color: [0, 0, 0],
                error: 0.0,
                track: vec![TrackObservation {
                    image: 0,
                    feature: idx,
                }],
            });
        }
        let manager = ObservationManager::new(&frames, &pairs, &reconstruction);
        let many_visible = RegistrationChoice {
            image: 1,
            pose: SE3::identity(),
            camera: reconstruction.camera,
            source: "pnp",
            pnp_inliers: 99,
            inlier_ratio: 1.0,
            visible_points: 4,
            visible_points_ratio: 0.5,
            mean_error_px: 0.0,
            pair_rot_error: 0.0,
            structureless_inliers: Vec::new(),
            frame_image_poses: Vec::new(),
            generalized_inliers: Vec::new(),
        };
        let high_ratio = RegistrationChoice {
            image: 2,
            pose: SE3::identity(),
            camera: reconstruction.camera,
            source: "pnp",
            pnp_inliers: 1,
            inlier_ratio: 1.0,
            visible_points: 2,
            visible_points_ratio: 0.5,
            mean_error_px: 0.0,
            pair_rot_error: 0.0,
            structureless_inliers: Vec::new(),
            frame_image_poses: Vec::new(),
            generalized_inliers: Vec::new(),
        };

        assert!(
            registration_rank(
                &many_visible,
                &reconstruction,
                &manager,
                &MapperConfig {
                    image_selection_method: ImageSelectionMethod::MaxVisiblePointsNum,
                    ..MapperConfig::default()
                }
            ) > registration_rank(
                &high_ratio,
                &reconstruction,
                &manager,
                &MapperConfig {
                    image_selection_method: ImageSelectionMethod::MaxVisiblePointsNum,
                    ..MapperConfig::default()
                }
            )
        );
        assert!(
            registration_rank(
                &high_ratio,
                &reconstruction,
                &manager,
                &MapperConfig {
                    image_selection_method: ImageSelectionMethod::MaxVisiblePointsRatio,
                    ..MapperConfig::default()
                }
            ) > registration_rank(
                &many_visible,
                &reconstruction,
                &manager,
                &MapperConfig {
                    image_selection_method: ImageSelectionMethod::MaxVisiblePointsRatio,
                    ..MapperConfig::default()
                }
            )
        );
        assert_eq!(
            registration_rank(
                &many_visible,
                &reconstruction,
                &manager,
                &MapperConfig {
                    image_selection_method: ImageSelectionMethod::MinUncertainty,
                    ..MapperConfig::default()
                }
            ),
            manager.point3d_visibility_score(1) as f32
        );
        assert_eq!(
            registration_rank(
                &many_visible,
                &reconstruction,
                &manager,
                &MapperConfig::default()
            ),
            manager.point3d_visibility_score(1) as f32
        );
    }

    #[test]
    fn find_next_registration_images_matches_colmap_bucketed_queue() {
        let mut frames = vec![
            minimal_frame(0, "seed.jpg"),
            minimal_frame(1, "clean_many_visible.jpg"),
            minimal_frame(2, "filtered_high_score.jpg"),
            minimal_frame(3, "tried_medium_score.jpg"),
            minimal_frame(4, "too_few_visible.jpg"),
        ];
        frames[0].keypoints = (0..8)
            .map(|idx| rustslam::KeyPoint::new(idx as f32, 0.0))
            .collect();
        frames[1].keypoints = (0..6)
            .map(|idx| rustslam::KeyPoint::new(idx as f32, 1.0))
            .collect();
        frames[2].keypoints = (0..8)
            .map(|idx| rustslam::KeyPoint::new(idx as f32, 2.0))
            .collect();
        frames[3].keypoints = (0..5)
            .map(|idx| rustslam::KeyPoint::new(idx as f32, 3.0))
            .collect();
        frames[4].keypoints = vec![rustslam::KeyPoint::new(0.0, 4.0)];
        let pairs = vec![
            pair_with_inliers(0, 1, &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]),
            pair_with_inliers(
                0,
                2,
                &[
                    (0, 0),
                    (1, 1),
                    (2, 2),
                    (3, 3),
                    (4, 4),
                    (5, 5),
                    (6, 6),
                    (7, 7),
                ],
            ),
            pair_with_inliers(0, 3, &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]),
            pair_with_inliers(0, 4, &[(0, 0)]),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        for idx in 0..8 {
            reconstruction.observations[0][idx] = Some(idx);
            reconstruction.point_ids.push(idx as u64 + 1);
            reconstruction.points.push(Point3D {
                xyz: [idx as f32 * 0.1, 0.0, 2.0],
                color: [0, 0, 0],
                error: 0.0,
                track: vec![TrackObservation {
                    image: 0,
                    feature: idx,
                }],
            });
        }
        let obs_manager = ObservationManager::new(&frames, &pairs, &reconstruction);
        let mut filtered_units = HashSet::new();
        filtered_units.insert(RegistrationUnitKey::Image(2));
        let reg_trials = vec![0, 0, 0, 1, 3];
        let config = MapperConfig {
            abs_pose_min_num_inliers: 2,
            max_reg_trials: 3,
            image_selection_method: ImageSelectionMethod::MaxVisiblePointsNum,
            ..MapperConfig::default()
        };

        let ranked = find_next_registration_images(
            &reconstruction,
            &reg_trials,
            &reg_trials,
            &filtered_units,
            &config,
            &obs_manager,
            NextImageRegistrationMode::StructureBased,
        );

        assert_eq!(ranked, vec![1, 2, 3]);
    }

    #[test]
    fn find_next_registration_images_fallback_retries_exhausted_candidate() {
        let mut frames = vec![
            minimal_frame(0, "seed.jpg"),
            minimal_frame(1, "candidate.jpg"),
        ];
        for frame in &mut frames {
            frame.keypoints = vec![
                rustslam::KeyPoint::new(10.0, 10.0),
                rustslam::KeyPoint::new(20.0, 20.0),
            ];
        }
        let pairs = vec![pair_with_inliers(0, 1, &[(0, 0), (1, 1)])];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        for idx in 0..2 {
            reconstruction.observations[0][idx] = Some(idx);
            reconstruction.point_ids.push(idx as u64 + 1);
            reconstruction.points.push(Point3D {
                xyz: [idx as f32 * 0.1, 0.0, 2.0],
                color: [0, 0, 0],
                error: 0.0,
                track: vec![TrackObservation {
                    image: 0,
                    feature: idx,
                }],
            });
        }
        let obs_manager = ObservationManager::new(&frames, &pairs, &reconstruction);
        let retry_state = RegistrationRetryState::from_trial_vectors(&[0, 3], &[0, 0]);
        let config = MapperConfig {
            abs_pose_min_num_inliers: 2,
            max_reg_trials: 3,
            image_selection_method: ImageSelectionMethod::MaxVisiblePointsNum,
            ..MapperConfig::default()
        };
        let mut normal_telemetry = IncrementalRegistrationTelemetry::default();
        let mut fallback_telemetry = IncrementalRegistrationTelemetry::default();

        let normal = find_next_registration_images_with_retry_state(
            &reconstruction,
            &retry_state,
            &HashSet::new(),
            &config,
            &obs_manager,
            NextImageRegistrationMode::StructureBased,
            RegistrationPass::Normal,
            &mut normal_telemetry,
        );
        let fallback = find_next_registration_images_with_retry_state(
            &reconstruction,
            &retry_state,
            &HashSet::new(),
            &config,
            &obs_manager,
            NextImageRegistrationMode::StructureBased,
            RegistrationPass::ExhaustiveFallback,
            &mut fallback_telemetry,
        );

        assert!(normal.is_empty());
        assert_eq!(normal_telemetry.skipped_unchanged, 1);
        assert_eq!(fallback, vec![1]);
        assert_eq!(fallback_telemetry.skipped_unchanged, 0);
    }

    #[test]
    fn find_next_registration_images_deduplicates_rig_frame_siblings() {
        let camera = CameraModel::new_pinhole(160, 120, 70.0, 70.0, 80.0, 60.0);
        let mut frames = vec![
            minimal_frame(0, "seed.jpg"),
            minimal_frame(1, "rig_ref.jpg"),
            minimal_frame(2, "rig_aux.jpg"),
        ];
        for frame in &mut frames {
            frame.width = camera.width;
            frame.height = camera.height;
            frame.keypoints = (0..12)
                .map(|idx| rustslam::KeyPoint::new(10.0 + idx as f32 * 4.0, 12.0))
                .collect();
            frame.colors = vec![[0, 0, 0]; frame.keypoints.len()];
        }
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.camera = camera;
        reconstruction.cameras = vec![camera, camera];
        reconstruction.poses[0] = Some(SE3::identity());
        attach_two_image_rig_frame(&mut reconstruction, 1, 2);
        for idx in 0..12 {
            reconstruction.observations[0][idx] = Some(idx);
            reconstruction.point_ids.push(idx as u64 + 1);
            reconstruction.points.push(Point3D {
                xyz: [idx as f32 * 0.05, 0.0, 2.0],
                color: [0, 0, 0],
                error: 0.0,
                track: vec![TrackObservation {
                    image: 0,
                    feature: idx,
                }],
            });
        }
        let matches = (0..12).map(|idx| (idx, idx)).collect::<Vec<_>>();
        let pairs = vec![
            pair_with_inliers(0, 1, &matches),
            pair_with_inliers(0, 2, &matches),
        ];
        let obs_manager = ObservationManager::new(&frames, &pairs, &reconstruction);
        let config = MapperConfig {
            abs_pose_min_num_inliers: 20,
            image_selection_method: ImageSelectionMethod::MaxVisiblePointsNum,
            ..MapperConfig::default()
        };

        let ranked = find_next_registration_images(
            &reconstruction,
            &[0; 3],
            &[0; 3],
            &HashSet::new(),
            &config,
            &obs_manager,
            NextImageRegistrationMode::StructureBased,
        );

        assert_eq!(ranked, vec![1]);
    }

    #[test]
    fn choose_next_registration_deprioritizes_filtered_or_failed_units() {
        let camera = CameraModel::new_pinhole(160, 120, 70.0, 70.0, 80.0, 60.0);
        let provider_pose = SE3::identity();
        let candidate_pose = SE3::from_quat_translation(
            glam::Quat::from_rotation_y(0.03),
            glam::Vec3::new(-0.2, 0.01, 0.05),
        );
        let points = (0..40)
            .map(|idx| {
                let col = (idx % 8) as f32;
                let row = (idx / 8) as f32;
                [
                    -0.55 + col * 0.16,
                    -0.3 + row * 0.16,
                    3.0 + idx as f32 * 0.015,
                ]
            })
            .collect::<Vec<_>>();
        let mut frames = vec![
            minimal_frame(0, "provider.jpg"),
            minimal_frame(1, "filtered_high_score.jpg"),
            minimal_frame(2, "clean_lower_score.jpg"),
        ];
        for frame in &mut frames {
            frame.width = camera.width;
            frame.height = camera.height;
            frame.keypoints.clear();
            frame.colors.clear();
        }
        for &point in &points {
            frames[0]
                .keypoints
                .push(project_test_point(camera, provider_pose, point));
            frames[1]
                .keypoints
                .push(project_test_point(camera, candidate_pose, point));
            frames[2]
                .keypoints
                .push(project_test_point(camera, candidate_pose, point));
        }
        for frame in &mut frames {
            frame.colors = vec![[0, 0, 0]; frame.keypoints.len()];
        }
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.camera = camera;
        reconstruction.cameras = vec![camera];
        reconstruction.poses[0] = Some(provider_pose);
        for (idx, &point) in points.iter().enumerate() {
            reconstruction.observations[0][idx] = Some(idx);
            reconstruction.point_ids.push(idx as u64 + 1);
            reconstruction.points.push(Point3D {
                xyz: point,
                color: [0, 0, 0],
                error: 0.0,
                track: vec![TrackObservation {
                    image: 0,
                    feature: idx,
                }],
            });
        }
        let high_score_pair =
            pair_with_inliers(0, 1, &(0..36).map(|idx| (idx, idx)).collect::<Vec<_>>());
        let lower_score_pair =
            pair_with_inliers(0, 2, &(0..28).map(|idx| (idx, idx)).collect::<Vec<_>>());
        let config = MapperConfig {
            abs_pose_min_num_inliers: 20,
            pnp_iterations: 512,
            random_seed: 13,
            ..MapperConfig::default()
        };

        let clean_choice = choose_next_registration(
            &frames,
            &[high_score_pair.clone(), lower_score_pair.clone()],
            &reconstruction,
            &[0; 3],
            &[0; 3],
            &HashSet::new(),
            &config,
            &camera_priors(&reconstruction),
            &camera_prior_focal_flags(&reconstruction, true),
            &registration_stats(&reconstruction),
        )
        .expect("clean high-score candidate");
        assert_eq!(clean_choice.image, 1);

        let mut filtered_units = HashSet::new();
        filtered_units.insert(RegistrationUnitKey::Image(1));
        let filtered_choice = choose_next_registration(
            &frames,
            &[high_score_pair, lower_score_pair],
            &reconstruction,
            &[0; 3],
            &[0; 3],
            &filtered_units,
            &config,
            &camera_priors(&reconstruction),
            &camera_prior_focal_flags(&reconstruction, true),
            &registration_stats(&reconstruction),
        )
        .expect("clean lower-score candidate");
        assert_eq!(filtered_choice.image, 2);
    }

    #[test]
    fn choose_next_registration_records_failed_queue_candidates_and_continues_like_colmap() {
        let camera = CameraModel::new_pinhole(160, 120, 70.0, 70.0, 80.0, 60.0);
        let provider_pose = SE3::identity();
        let good_pose = SE3::from_quat_translation(
            glam::Quat::from_rotation_y(0.03),
            glam::Vec3::new(-0.2, 0.01, 0.05),
        );
        let points = (0..48)
            .map(|idx| {
                let col = (idx % 8) as f32;
                let row = (idx / 8) as f32;
                [
                    -0.55 + col * 0.16,
                    -0.36 + row * 0.14,
                    3.0 + idx as f32 * 0.012,
                ]
            })
            .collect::<Vec<_>>();
        let mut frames = vec![
            minimal_frame(0, "provider.jpg"),
            minimal_frame(1, "bad_first.jpg"),
            minimal_frame(2, "good_second.jpg"),
        ];
        for frame in &mut frames {
            frame.width = camera.width;
            frame.height = camera.height;
            frame.keypoints.clear();
            frame.colors.clear();
        }
        for &point in &points {
            frames[0]
                .keypoints
                .push(project_test_point(camera, provider_pose, point));
            frames[1]
                .keypoints
                .push(rustslam::KeyPoint::new(camera.cx, camera.cy));
            frames[2]
                .keypoints
                .push(project_test_point(camera, good_pose, point));
        }
        for frame in &mut frames {
            frame.colors = vec![[0, 0, 0]; frame.keypoints.len()];
        }
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.camera = camera;
        reconstruction.cameras = vec![camera];
        reconstruction.poses[0] = Some(provider_pose);
        for (idx, &point) in points.iter().enumerate() {
            reconstruction.observations[0][idx] = Some(idx);
            reconstruction.point_ids.push(idx as u64 + 1);
            reconstruction.points.push(Point3D {
                xyz: point,
                color: [0, 0, 0],
                error: 0.0,
                track: vec![TrackObservation {
                    image: 0,
                    feature: idx,
                }],
            });
        }
        let bad_pair = pair_with_inliers(0, 1, &(0..48).map(|idx| (idx, idx)).collect::<Vec<_>>());
        let good_pair = pair_with_inliers(0, 2, &(0..36).map(|idx| (idx, idx)).collect::<Vec<_>>());
        let config = MapperConfig {
            abs_pose_min_num_inliers: 20,
            pnp_iterations: 256,
            random_seed: 19,
            image_selection_method: ImageSelectionMethod::MaxVisiblePointsNum,
            ..MapperConfig::default()
        };

        let obs_manager = ObservationManager::new(
            &frames,
            &[bad_pair.clone(), good_pair.clone()],
            &reconstruction,
        );
        let selection = choose_next_registration_with_failures(
            &frames,
            &[bad_pair, good_pair],
            &reconstruction,
            &[0; 3],
            &[0; 3],
            &HashSet::new(),
            &config,
            &camera_priors(&reconstruction),
            &camera_prior_focal_flags(&reconstruction, true),
            &registration_stats(&reconstruction),
            &obs_manager,
        );

        assert_eq!(
            selection.failed_attempts,
            vec![(1, NextImageRegistrationMode::StructureBased)]
        );
        let choice = selection
            .choice
            .expect("second queue candidate should register");
        assert_eq!(choice.image, 2);
        assert_eq!(choice.source, "pnp");
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
    fn track_filter_updates_error_without_retriangulating_xyz() {
        let camera = CameraModel::new_pinhole(100, 100, 50.0, 50.0, 50.0, 50.0);
        let pose0 = SE3::identity();
        let pose1 =
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(1.0, 0.0, 0.0));
        let pose2 =
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(0.0, 1.0, 0.0));
        let original_xyz = [0.0, 0.0, 4.0];
        let measured_xyz = [0.04, -0.03, 4.08];
        let mut frames = vec![
            minimal_frame(0, "a.jpg"),
            minimal_frame(1, "b.jpg"),
            minimal_frame(2, "c.jpg"),
        ];
        frames[0].keypoints[0] = project_test_point(camera, pose0, measured_xyz);
        frames[1].keypoints[0] = project_test_point(camera, pose1, measured_xyz);
        frames[2].keypoints[0] = project_test_point(camera, pose2, measured_xyz);
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.camera = camera;
        reconstruction.cameras = vec![camera];
        reconstruction.poses[0] = Some(pose0);
        reconstruction.poses[1] = Some(pose1);
        reconstruction.poses[2] = Some(pose2);
        reconstruction.observations[0][0] = Some(0);
        reconstruction.observations[1][0] = Some(0);
        reconstruction.observations[2][0] = Some(0);
        reconstruction.point_ids.push(17);
        reconstruction.points.push(Point3D {
            xyz: original_xyz,
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

        let removed = filter_reprojection_tracks_with_policy(
            &frames,
            &[],
            &mut reconstruction,
            &MapperConfig {
                max_reprojection_error_px: 5.0,
                ..MapperConfig::default()
            },
            None,
            5.0,
            0.1,
            2,
        );

        assert_eq!(removed, 0);
        assert_eq!(reconstruction.points.len(), 1);
        assert_eq!(reconstruction.points[0].xyz, original_xyz);
        assert!(reconstruction.points[0].error > 0.0);
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
            None,
            1.0,
            0.1,
            3,
        );

        assert_eq!(removed, 2);
        assert!(reconstruction.points.is_empty());
    }

    #[test]
    fn subset_track_filter_leaves_unselected_invalid_points_for_full_boundary() {
        let mut frames = (0..6)
            .map(|id| minimal_frame(id, &format!("{id}.jpg")))
            .collect::<Vec<_>>();
        for frame in &mut frames {
            frame.keypoints[0] = rustslam::KeyPoint::new(50.0, 50.0);
        }
        frames[5].keypoints[0] = rustslam::KeyPoint::new(75.0, 50.0);
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses.fill(Some(SE3::identity()));
        reconstruction.poses[5] = Some(SE3::from_quat_translation(
            glam::Quat::IDENTITY,
            glam::Vec3::new(1.0, 0.0, 0.0),
        ));
        let tracks = [
            vec![
                TrackObservation {
                    image: 0,
                    feature: 0,
                },
                TrackObservation {
                    image: 1,
                    feature: 0,
                },
            ],
            vec![
                TrackObservation {
                    image: 2,
                    feature: 0,
                },
                TrackObservation {
                    image: 3,
                    feature: 0,
                },
            ],
            vec![
                TrackObservation {
                    image: 4,
                    feature: 0,
                },
                TrackObservation {
                    image: 5,
                    feature: 0,
                },
            ],
        ];
        add_test_point3d(&mut reconstruction, 11, tracks[0].clone());
        add_test_point3d(&mut reconstruction, 22, tracks[1].clone());
        add_test_point3d(&mut reconstruction, 33, tracks[2].clone());
        reconstruction.points[0].xyz = [0.0, 0.0, -1.0];
        reconstruction.points[1].xyz = [0.0, 0.0, -1.0];
        reconstruction.points[2].xyz = [0.0, 0.0, 2.0];
        let mut state = IncrementalTriangulatorState::new(&frames, &[], &reconstruction);

        let removed = filter_reprojection_tracks_subset_with_state(
            &frames,
            &[],
            &mut reconstruction,
            &MapperConfig::default(),
            &mut state,
            &HashSet::from([0]),
        );

        assert_eq!(removed, 2);
        assert_eq!(reconstruction.point_ids, vec![33, 22]);
        assert!(reconstruction.points[1].xyz[2] < 0.0);
        assert_eq!(
            filter_reprojection_tracks_with_state(
                &frames,
                &[],
                &mut reconstruction,
                &MapperConfig::default(),
                &mut state,
            ),
            2
        );
        assert_eq!(reconstruction.point_ids, vec![33]);
        let log = state.observation_manager().sparse_maintenance_log();
        for expected in [
            "full_filter_calls=1",
            "subset_filter_calls=1",
            "filter_points=3",
            "filter_observations=6",
            "filtered_observations=4",
        ] {
            assert!(log.contains(expected), "missing {expected}: {log}");
        }
    }

    #[test]
    fn modified_track_filter_consumes_frontier_without_full_scan() {
        let mut frames = (0..4)
            .map(|id| minimal_frame(id, &format!("{id}.jpg")))
            .collect::<Vec<_>>();
        for frame in &mut frames {
            frame.keypoints[0] = rustslam::KeyPoint::new(50.0, 50.0);
        }
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses.fill(Some(SE3::identity()));
        add_test_point3d(
            &mut reconstruction,
            11,
            vec![
                TrackObservation {
                    image: 0,
                    feature: 0,
                },
                TrackObservation {
                    image: 1,
                    feature: 0,
                },
            ],
        );
        add_test_point3d(
            &mut reconstruction,
            22,
            vec![
                TrackObservation {
                    image: 2,
                    feature: 0,
                },
                TrackObservation {
                    image: 3,
                    feature: 0,
                },
            ],
        );
        reconstruction.points[0].xyz = [0.0, 0.0, -1.0];
        reconstruction.points[1].xyz = [0.0, 0.0, -1.0];
        let mut state = IncrementalTriangulatorState::new(&frames, &[], &reconstruction);
        state.observation_manager_mut().mark_point3d_modified(0);

        assert_eq!(
            filter_modified_reprojection_tracks_with_state(
                &frames,
                &[],
                &mut reconstruction,
                &MapperConfig::default(),
                &mut state,
            ),
            2
        );

        assert_eq!(reconstruction.point_ids, vec![22]);
        assert!(state
            .observation_manager()
            .modified_point3d_ids()
            .is_empty());
        let log = state.observation_manager().sparse_maintenance_log();
        assert!(log.contains("full_filter_calls=0"), "{log}");
        assert!(log.contains("subset_filter_calls=1"), "{log}");
        assert!(log.contains("frontier_cycles=1"), "{log}");
    }

    #[test]
    fn synthetic_local_ba_filters_only_its_modified_frontier() {
        let camera = CameraModel::new_pinhole(200, 160, 80.0, 80.0, 100.0, 80.0);
        let poses = [
            SE3::identity(),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(0.03),
                glam::Vec3::new(-0.35, 0.0, 0.0),
            ),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(-0.02),
                glam::Vec3::new(0.45, 0.0, 0.0),
            ),
        ];
        let points = (0..12)
            .map(|idx| {
                let col = (idx % 4) as f32;
                let row = (idx / 4) as f32;
                [-0.3 + col * 0.2, -0.2 + row * 0.18, 3.0 + idx as f32 * 0.03]
            })
            .collect::<Vec<_>>();
        let mut frames = (0..3)
            .map(|idx| minimal_frame(idx, &format!("image_{idx}.jpg")))
            .collect::<Vec<_>>();
        for (image, pose) in poses.iter().copied().enumerate() {
            frames[image].width = camera.width;
            frames[image].height = camera.height;
            frames[image].keypoints = points
                .iter()
                .map(|&point| project_test_point(camera, pose, point))
                .collect();
            frames[image].colors = vec![[image as u8, 0, 0]; points.len()];
        }
        let pairs = vec![
            initial_pair_from_projected_points(0, 1, poses[0], poses[1], points.len()),
            initial_pair_from_projected_points(0, 2, poses[0], poses[2], points.len()),
            initial_pair_from_projected_points(1, 2, poses[1], poses[2], points.len()),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.camera = camera;
        reconstruction.cameras = vec![camera];
        reconstruction.poses = poses.into_iter().map(Some).collect();
        for (point_id, xyz) in points.into_iter().enumerate() {
            let track = (0..3)
                .map(|image| TrackObservation {
                    image,
                    feature: point_id,
                })
                .collect();
            add_test_point3d(&mut reconstruction, point_id as u64 + 1, track);
            reconstruction.points[point_id].xyz = xyz;
        }
        let mut state = IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        let stats = RegistrationStats::from_reconstruction(&reconstruction);
        let config = MapperConfig {
            local_ba: true,
            local_ba_num_images: 2,
            local_ba_min_shared_points: 4,
            local_ba_iterations: 1,
            local_ba_max_refinements: 1,
            global_ba: false,
            extract_colors: false,
            ..MapperConfig::default()
        };

        refine_local_bundle_after_registration(
            &frames,
            &pairs,
            &mut reconstruction,
            2,
            0,
            &mapper_triangulator_options(&config),
            &config,
            &stats,
            &mut state,
        )
        .expect("synthetic local BA");

        assert!(state
            .observation_manager()
            .modified_point3d_ids()
            .is_empty());
        let log = state.observation_manager().sparse_maintenance_log();
        assert!(log.contains("full_filter_calls=0"), "{log}");
        assert!(log.contains("subset_filter_calls=1"), "{log}");
    }

    #[test]
    fn stateful_track_filter_updates_candidate_visibility_stats() {
        let mut frames = vec![
            minimal_frame(0, "registered_a.jpg"),
            minimal_frame(1, "registered_b.jpg"),
            minimal_frame(2, "candidate.jpg"),
        ];
        frames[0].keypoints[0] = rustslam::KeyPoint::new(50.0, 50.0);
        frames[1].keypoints[0] = rustslam::KeyPoint::new(90.0, 50.0);
        frames[2].keypoints[0] = rustslam::KeyPoint::new(55.0, 50.0);
        let pairs = vec![
            pair_with_inliers(0, 2, &[(0, 0)]),
            pair_with_inliers(1, 2, &[(0, 0)]),
        ];
        let mut reconstruction = test_reconstruction(&frames);
        reconstruction.poses[0] = Some(SE3::identity());
        reconstruction.poses[1] = Some(SE3::from_quat_translation(
            glam::Quat::IDENTITY,
            glam::Vec3::new(1.0, 0.0, 0.0),
        ));
        reconstruction.observations[0][0] = Some(0);
        reconstruction.observations[1][0] = Some(0);
        reconstruction.point_ids.push(21);
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
        let mut triangulation_state =
            IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);

        assert_eq!(
            triangulation_state
                .observation_manager()
                .num_visible_correspondences(2),
            2
        );
        assert_eq!(
            triangulation_state
                .observation_manager()
                .num_visible_points3d(2),
            1
        );
        assert!(
            triangulation_state
                .observation_manager()
                .point3d_visibility_score(2)
                > 0
        );
        assert_eq!(
            triangulation_state
                .observation_manager()
                .num_correspondences_have_point3d(2, 0),
            2
        );

        let removed = filter_reprojection_tracks_with_state(
            &frames,
            &pairs,
            &mut reconstruction,
            &MapperConfig {
                max_reprojection_error_px: 1.0,
                ..MapperConfig::default()
            },
            &mut triangulation_state,
        );

        assert_eq!(removed, 2);
        assert!(reconstruction.points.is_empty());
        assert_observation_manager_matches_fresh(
            &frames,
            &pairs,
            &reconstruction,
            &triangulation_state,
        );
        let state_manager = triangulation_state.observation_manager();
        assert_eq!(state_manager.num_visible_correspondences(2), 2);
        assert_eq!(state_manager.num_visible_points3d(2), 0);
        assert_eq!(state_manager.point3d_visibility_score(2), 0);
        assert_eq!(state_manager.num_correspondences_have_point3d(2, 0), 0);
    }

    #[test]
    #[ignore = "requires the external test_data/flowers2_colmap fixture"]
    fn real_colmap_sparse_track_filter_keeps_observation_manager_in_sync() -> Result<()> {
        let (camera, frames, setup, pairs, _) = real_colmap_sparse_seed_local_ba_fixture()?;
        let mut reconstruction =
            reconstruction_from_reference_setup_for_test(&frames, camera, &setup);
        assert_eq!(reconstruction.points.len(), 256);
        let removed_track = reconstruction.points[0].track.clone();
        let mut triangulation_state =
            IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);

        reconstruction.points[0].xyz = [0.0, 0.0, -10.0];
        let removed = filter_reprojection_tracks_with_policy(
            &frames,
            &pairs,
            &mut reconstruction,
            &MapperConfig::default(),
            Some(triangulation_state.observation_manager_mut()),
            4.0,
            0.0,
            2,
        );

        assert_eq!(removed, removed_track.len());
        assert_eq!(reconstruction.points.len(), 255);
        for obs in removed_track {
            assert_eq!(reconstruction.observations[obs.image][obs.feature], None);
        }
        assert_observation_manager_matches_fresh(
            &frames,
            &pairs,
            &reconstruction,
            &triangulation_state,
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires the external test_data/flowers2_colmap fixture"]
    fn real_colmap_sparse_initial_pair_keeps_observation_manager_in_sync() -> Result<()> {
        let (camera, frames, setup, pairs, _) = real_colmap_sparse_seed_mapper_fixture()?;
        let setup = ReferenceCameraSetup {
            seed_reconstruction: None,
            ..setup
        };
        let mut reconstruction =
            reconstruction_from_reference_setup_for_test(&frames, camera, &setup);
        let initial = pairs[0].clone();
        let mut triangulation_state =
            IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        let tri_options = IncrementalTriangulatorOptions {
            ignore_two_view_tracks: false,
            ..IncrementalTriangulatorOptions::from_mapper_threshold(4.0)
        };

        register_and_triangulate_initial_image_pair(
            &frames,
            &pairs,
            &mut reconstruction,
            &mut triangulation_state,
            &tri_options,
            initial.median_triangulation_angle_deg.max(0.1) * 0.5,
            &initial,
        );
        assert!(
            reconstruction.points.len() > 0,
            "initial pair should triangulate real COLMAP correspondences"
        );
        filter_reprojection_tracks_with_state(
            &frames,
            &pairs,
            &mut reconstruction,
            &MapperConfig::default(),
            &mut triangulation_state,
        );
        assert_observation_manager_matches_fresh(
            &frames,
            &pairs,
            &reconstruction,
            &triangulation_state,
        );
        Ok(())
    }

    #[test]
    fn structureless_tracks_keep_observation_manager_in_sync_with_fresh_rebuild() {
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
        reconstruction.observations[0][0] = Some(0);
        reconstruction.points.push(Point3D {
            xyz: point,
            color: [0, 0, 0],
            error: 0.0,
            track: vec![TrackObservation {
                image: 0,
                feature: 0,
            }],
        });
        reconstruction.point_ids.push(1);
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
        let mut triangulation_state =
            IncrementalTriangulatorState::new(&frames, &[], &reconstruction);
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
            triangulation_state.observation_manager_mut(),
        );
        assert_eq!(report.continued_observations, 1);
        assert_observation_manager_matches_fresh(
            &frames,
            &[],
            &reconstruction,
            &triangulation_state,
        );
    }

    #[test]
    #[ignore = "requires the external test_data/flowers2_colmap fixture"]
    fn real_colmap_sparse_local_ba_post_filter_keeps_observation_manager_in_sync() -> Result<()> {
        let (camera, frames, setup, mut pairs, reference_pose) =
            real_colmap_sparse_seed_local_ba_fixture()?;
        let mut reconstruction =
            reconstruction_from_reference_setup_for_test(&frames, camera, &setup);
        reconstruction.poses[2] = Some(reference_pose);
        assert_eq!(reconstruction.points.len(), 256);

        let gauge_candidate_pair = pairs
            .iter()
            .find(|pair| pair.left == 0 && pair.right == 2)
            .expect("gauge-candidate pair");
        for m in &gauge_candidate_pair.inlier_matches {
            let gauge_feature = m.query_idx as usize;
            let candidate_feature = m.train_idx as usize;
            let Some(point_id) = reconstruction.observations[0][gauge_feature] else {
                continue;
            };
            if reconstruction.observations[2][candidate_feature].is_none() {
                reconstruction.observations[2][candidate_feature] = Some(point_id);
                reconstruction.points[point_id]
                    .track
                    .push(TrackObservation {
                        image: 2,
                        feature: candidate_feature,
                    });
            }
        }
        assert_eq!(reconstruction.points[0].track.len(), 3);

        let removed_track = reconstruction.points[0].track.clone();
        remove_track_features_from_pairs(&mut pairs, &removed_track);
        let candidate_obs = removed_track
            .iter()
            .copied()
            .find(|obs| obs.image == 2)
            .expect("candidate observation");
        for obs in &removed_track {
            reconstruction.observations[obs.image][obs.feature] = None;
        }
        reconstruction.observations[candidate_obs.image][candidate_obs.feature] = Some(0);
        reconstruction.points[0].track = vec![candidate_obs];

        let mut triangulation_state =
            IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        let stats = RegistrationStats::from_reconstruction(&reconstruction);
        let config = MapperConfig {
            local_ba: true,
            local_ba_iterations: 3,
            local_ba_max_refinements: 1,
            global_ba: false,
            extract_colors: false,
            ..MapperConfig::default()
        };

        let report = refine_local_bundle_after_registration(
            &frames,
            &pairs,
            &mut reconstruction,
            2,
            0,
            &mapper_triangulator_options(&config),
            &config,
            &stats,
            &mut triangulation_state,
        )
        .expect("local BA should run on real COLMAP sparse fixture");

        assert!(
            report.filtered_observations > 0,
            "local BA post-filter should delete observations from the real fixture: {report:?}"
        );
        assert!(reconstruction.points.len() < 256);
        for obs in removed_track {
            assert_eq!(reconstruction.observations[obs.image][obs.feature], None);
        }
        assert_observation_manager_matches_fresh(
            &frames,
            &pairs,
            &reconstruction,
            &triangulation_state,
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires the external test_data/flowers2_colmap fixture"]
    fn real_colmap_sparse_local_ba_merges_tracks_keeps_observation_manager_in_sync() -> Result<()> {
        let (camera, frames, setup, pairs, reference_pose) =
            real_colmap_sparse_seed_local_ba_fixture()?;
        let mut reconstruction =
            reconstruction_from_reference_setup_for_test(&frames, camera, &setup);
        reconstruction.poses[2] = Some(reference_pose);
        assert_eq!(reconstruction.points.len(), 256);

        let gauge_candidate_pair = pairs
            .iter()
            .find(|pair| pair.left == 0 && pair.right == 2)
            .expect("gauge-candidate pair");
        let mut split_candidate_feature = None;
        for (idx, m) in gauge_candidate_pair.inlier_matches.iter().enumerate() {
            let gauge_feature = m.query_idx as usize;
            let candidate_feature = m.train_idx as usize;
            let Some(point_id) = reconstruction.observations[0][gauge_feature] else {
                continue;
            };
            if idx == 0 {
                split_candidate_feature = Some((point_id, candidate_feature));
                continue;
            }
            if reconstruction.observations[2][candidate_feature].is_none() {
                reconstruction.observations[2][candidate_feature] = Some(point_id);
                reconstruction.points[point_id]
                    .track
                    .push(TrackObservation {
                        image: 2,
                        feature: candidate_feature,
                    });
            }
        }
        let (target_point, candidate_feature) =
            split_candidate_feature.expect("real COLMAP split candidate feature");
        assert_eq!(reconstruction.observations[2][candidate_feature], None);

        let split_point_id = reconstruction.points.len();
        let split_point = Point3D {
            xyz: reconstruction.points[target_point].xyz,
            color: reconstruction.points[target_point].color,
            error: reconstruction.points[target_point].error,
            track: vec![TrackObservation {
                image: 2,
                feature: candidate_feature,
            }],
        };
        reconstruction.observations[2][candidate_feature] = Some(split_point_id);
        reconstruction
            .point_ids
            .push(reconstruction.point3d_id(target_point) + 2_000_000);
        reconstruction.points.push(split_point);
        assert_eq!(reconstruction.points.len(), 257);

        let mut triangulation_state =
            IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        assert_observation_manager_matches_fresh(
            &frames,
            &pairs,
            &reconstruction,
            &triangulation_state,
        );
        let stats = RegistrationStats::from_reconstruction(&reconstruction);
        let config = MapperConfig {
            local_ba: true,
            local_ba_iterations: 3,
            local_ba_max_refinements: 1,
            global_ba: false,
            extract_colors: false,
            ..MapperConfig::default()
        };

        let report = refine_local_bundle_after_registration(
            &frames,
            &pairs,
            &mut reconstruction,
            2,
            0,
            &mapper_triangulator_options(&config),
            &config,
            &stats,
            &mut triangulation_state,
        )
        .expect("local BA should run on real COLMAP sparse fixture");

        assert!(
            report.merged_observations > 0,
            "local BA post-merge should merge split real COLMAP tracks: {report:?}"
        );
        assert_eq!(reconstruction.points.len(), 256);
        assert_eq!(
            reconstruction.observations[2][candidate_feature],
            Some(target_point)
        );
        assert!(reconstruction.points[target_point]
            .track
            .iter()
            .any(|obs| obs.image == 2 && obs.feature == candidate_feature));
        assert_observation_manager_matches_fresh(
            &frames,
            &pairs,
            &reconstruction,
            &triangulation_state,
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires the external test_data/flowers2_colmap fixture"]
    fn real_colmap_sparse_global_ba_prepare_completes_tracks_keeps_observation_manager_in_sync(
    ) -> Result<()> {
        let (camera, frames, setup, pairs, reference_pose) =
            real_colmap_sparse_seed_local_ba_fixture()?;
        let mut reconstruction =
            reconstruction_from_reference_setup_for_test(&frames, camera, &setup);
        reconstruction.poses[2] = Some(reference_pose);
        assert_eq!(reconstruction.points.len(), 256);
        assert_eq!(
            reconstruction.observations[2]
                .iter()
                .filter(|point| point.is_some())
                .count(),
            0
        );

        let mut triangulation_state =
            IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        assert_observation_manager_matches_fresh(
            &frames,
            &pairs,
            &reconstruction,
            &triangulation_state,
        );
        let mut stats = RegistrationStats::from_reconstruction(&reconstruction);
        let mut filtered_units = HashSet::new();
        let mut debug_log = Vec::new();
        let config = MapperConfig {
            global_ba: true,
            global_ba_iterations: 3,
            global_ba_max_refinements: 1,
            global_ba_images_freq: 999,
            global_ba_points_freq: 999_999,
            global_ba_images_ratio: 10.0,
            global_ba_points_ratio: 10.0,
            extract_colors: false,
            ..MapperConfig::default()
        };

        assert!(refine_global_bundle_with_postprocessing(
            &frames,
            &pairs,
            &mut reconstruction,
            &mapper_triangulator_options(&config),
            &config,
            "prepare_fixture",
            false,
            &mut debug_log,
            Some(&mut stats),
            Some(&mut filtered_units),
            &mut triangulation_state,
        ));

        let prepare = debug_log
            .iter()
            .find(|line| line.starts_with("global_ba_prepare reason=prepare_fixture"))
            .unwrap_or_else(|| panic!("missing global BA prepare log: {debug_log:?}"));
        let completed = prepare
            .split_whitespace()
            .find_map(|field| field.strip_prefix("completed="))
            .and_then(|value| value.parse::<usize>().ok())
            .expect("prepare completed count");
        assert!(
            completed > 0,
            "global BA prepare should complete real COLMAP tracks: {prepare}"
        );
        assert_eq!(
            reconstruction.observations[2]
                .iter()
                .filter(|point| point.is_some())
                .count(),
            256
        );
        assert_observation_manager_matches_fresh(
            &frames,
            &pairs,
            &reconstruction,
            &triangulation_state,
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires the external test_data/flowers2_colmap fixture"]
    fn real_colmap_sparse_final_ba_prepare_keeps_observation_manager_in_sync() -> Result<()> {
        let (camera, frames, setup, pairs, reference_pose) =
            real_colmap_sparse_seed_local_ba_fixture()?;
        let mut reconstruction =
            reconstruction_from_reference_setup_for_test(&frames, camera, &setup);
        reconstruction.poses[2] = Some(reference_pose);
        assert_eq!(reconstruction.points.len(), 256);
        assert_eq!(
            reconstruction.observations[2]
                .iter()
                .filter(|point| point.is_some())
                .count(),
            0
        );

        let mut triangulation_state =
            IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        assert_observation_manager_matches_fresh(
            &frames,
            &pairs,
            &reconstruction,
            &triangulation_state,
        );
        let mut stats = RegistrationStats::from_reconstruction(&reconstruction);
        let mut filtered_units = HashSet::new();
        let mut debug_log = Vec::new();
        let config = MapperConfig {
            global_ba: true,
            global_ba_iterations: 3,
            global_ba_max_refinements: 1,
            global_ba_images_freq: 999,
            global_ba_points_freq: 999_999,
            global_ba_images_ratio: 10.0,
            global_ba_points_ratio: 10.0,
            extract_colors: false,
            ..MapperConfig::default()
        };

        assert!(refine_global_bundle_with_postprocessing(
            &frames,
            &pairs,
            &mut reconstruction,
            &mapper_triangulator_options(&config),
            &config,
            "final",
            final_global_ba_normalizes_reconstruction(&config),
            &mut debug_log,
            Some(&mut stats),
            Some(&mut filtered_units),
            &mut triangulation_state,
        ));

        let prepare = debug_log
            .iter()
            .find(|line| line.starts_with("global_ba_prepare reason=final"))
            .unwrap_or_else(|| panic!("missing final global BA prepare log: {debug_log:?}"));
        let completed = prepare
            .split_whitespace()
            .find_map(|field| field.strip_prefix("completed="))
            .and_then(|value| value.parse::<usize>().ok())
            .expect("final prepare completed count");
        assert!(
            completed > 0,
            "final global BA prepare should complete true COLMAP tracks: {prepare}"
        );
        assert!(
            debug_log
                .iter()
                .any(|line| line.starts_with("global_ba reason=final round=1")),
            "missing final global BA report: {debug_log:?}"
        );
        assert!(
            !debug_log
                .iter()
                .any(|line| line.starts_with("global_ba_normalize reason=final")),
            "final global BA should not normalize: {debug_log:?}"
        );
        assert_eq!(
            reconstruction.observations[2]
                .iter()
                .filter(|point| point.is_some())
                .count(),
            256
        );
        assert_observation_manager_matches_fresh(
            &frames,
            &pairs,
            &reconstruction,
            &triangulation_state,
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires the external test_data/flowers2_colmap fixture"]
    fn real_colmap_sparse_global_ba_prepare_merges_tracks_keeps_observation_manager_in_sync(
    ) -> Result<()> {
        let (camera, frames, setup, pairs, reference_pose) =
            real_colmap_sparse_seed_local_ba_fixture()?;
        let mut reconstruction =
            reconstruction_from_reference_setup_for_test(&frames, camera, &setup);
        reconstruction.poses[2] = Some(reference_pose);
        assert_eq!(reconstruction.points.len(), 256);

        let target_point = 0usize;
        let gauge_feature = reconstruction.points[target_point]
            .track
            .iter()
            .find(|obs| obs.image == 0)
            .map(|obs| obs.feature)
            .expect("target point has gauge observation");
        let candidate_feature = pairs
            .iter()
            .find(|pair| pair.left == 0 && pair.right == 2)
            .and_then(|pair| {
                pair.inlier_matches
                    .iter()
                    .find(|m| m.query_idx as usize == gauge_feature)
            })
            .map(|m| m.train_idx as usize)
            .expect("target point has real COLMAP gauge-candidate match");
        assert_eq!(reconstruction.observations[2][candidate_feature], None);

        let split_point_id = reconstruction.points.len();
        let split_point = Point3D {
            xyz: reconstruction.points[target_point].xyz,
            color: reconstruction.points[target_point].color,
            error: reconstruction.points[target_point].error,
            track: vec![TrackObservation {
                image: 2,
                feature: candidate_feature,
            }],
        };
        reconstruction.observations[2][candidate_feature] = Some(split_point_id);
        reconstruction
            .point_ids
            .push(reconstruction.point3d_id(target_point) + 1_000_000);
        reconstruction.points.push(split_point);
        assert_eq!(reconstruction.points.len(), 257);

        let mut triangulation_state =
            IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        assert_observation_manager_matches_fresh(
            &frames,
            &pairs,
            &reconstruction,
            &triangulation_state,
        );
        let mut stats = RegistrationStats::from_reconstruction(&reconstruction);
        let mut filtered_units = HashSet::new();
        let mut debug_log = Vec::new();
        let config = MapperConfig {
            global_ba: true,
            global_ba_iterations: 3,
            global_ba_max_refinements: 1,
            global_ba_images_freq: 999,
            global_ba_points_freq: 999_999,
            global_ba_images_ratio: 10.0,
            global_ba_points_ratio: 10.0,
            extract_colors: false,
            ..MapperConfig::default()
        };

        assert!(refine_global_bundle_with_postprocessing(
            &frames,
            &pairs,
            &mut reconstruction,
            &mapper_triangulator_options(&config),
            &config,
            "prepare_merge_fixture",
            false,
            &mut debug_log,
            Some(&mut stats),
            Some(&mut filtered_units),
            &mut triangulation_state,
        ));

        let prepare = debug_log
            .iter()
            .find(|line| line.starts_with("global_ba_prepare reason=prepare_merge_fixture"))
            .unwrap_or_else(|| panic!("missing global BA prepare merge log: {debug_log:?}"));
        let merged = prepare
            .split_whitespace()
            .find_map(|field| field.strip_prefix("merged="))
            .and_then(|value| value.parse::<usize>().ok())
            .expect("prepare merged count");
        assert!(
            merged > 0,
            "global BA prepare should merge split real COLMAP tracks: {prepare}"
        );
        assert_eq!(reconstruction.points.len(), 256);
        assert_eq!(
            reconstruction.observations[2][candidate_feature],
            Some(target_point)
        );
        assert!(reconstruction.points[target_point]
            .track
            .iter()
            .any(|obs| obs.image == 2 && obs.feature == candidate_feature));
        assert_observation_manager_matches_fresh(
            &frames,
            &pairs,
            &reconstruction,
            &triangulation_state,
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires the external test_data/flowers2_colmap fixture"]
    fn real_colmap_sparse_global_ba_prepare_retriangulates_tracks_keeps_observation_manager_in_sync(
    ) -> Result<()> {
        let (camera, frames, setup, pairs, reference_pose) =
            real_colmap_sparse_seed_local_ba_fixture()?;
        let mut reconstruction =
            reconstruction_from_reference_setup_for_test(&frames, camera, &setup);
        reconstruction.poses[2] = Some(reference_pose);
        assert_eq!(reconstruction.points.len(), 256);

        let kept_points = 8usize;
        for image_observations in &mut reconstruction.observations {
            for observation in image_observations {
                if observation.is_some_and(|point_id| point_id >= kept_points) {
                    *observation = None;
                }
            }
        }
        reconstruction.points.truncate(kept_points);
        reconstruction.point_ids.truncate(kept_points);
        assert_eq!(reconstruction.points.len(), kept_points);
        assert_eq!(
            reconstruction.observations[2]
                .iter()
                .filter(|point| point.is_some())
                .count(),
            0
        );

        let mut triangulation_state =
            IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        assert_observation_manager_matches_fresh(
            &frames,
            &pairs,
            &reconstruction,
            &triangulation_state,
        );
        let mut stats = RegistrationStats::from_reconstruction(&reconstruction);
        let mut filtered_units = HashSet::new();
        let mut debug_log = Vec::new();
        let config = MapperConfig {
            global_ba: true,
            global_ba_iterations: 3,
            global_ba_max_refinements: 1,
            global_ba_images_freq: 999,
            global_ba_points_freq: 999_999,
            global_ba_images_ratio: 10.0,
            global_ba_points_ratio: 10.0,
            extract_colors: false,
            ..MapperConfig::default()
        };

        assert!(refine_global_bundle_with_postprocessing(
            &frames,
            &pairs,
            &mut reconstruction,
            &mapper_triangulator_options(&config),
            &config,
            "prepare_retriangulate_fixture",
            false,
            &mut debug_log,
            Some(&mut stats),
            Some(&mut filtered_units),
            &mut triangulation_state,
        ));

        let prepare = debug_log
            .iter()
            .find(|line| line.starts_with("global_ba_prepare reason=prepare_retriangulate_fixture"))
            .unwrap_or_else(|| {
                panic!("missing global BA prepare retriangulate log: {debug_log:?}")
            });
        let retriangulated = prepare
            .split_whitespace()
            .find_map(|field| field.strip_prefix("retriangulated="))
            .and_then(|value| value.parse::<usize>().ok())
            .expect("prepare retriangulated count");
        assert!(
            retriangulated > 0,
            "global BA prepare should retriangulate under-reconstructed real COLMAP pairs: {prepare}"
        );
        assert!(
            reconstruction.points.len() > kept_points,
            "retriangulation should restore real COLMAP sparse points"
        );
        assert!(
            reconstruction.observations[2]
                .iter()
                .filter(|point| point.is_some())
                .count()
                > 0,
            "candidate image should gain retriangulated observations"
        );
        assert_eq!(
            triangulation_state
                .retriangulation_trials()
                .get(&(0, 2))
                .copied(),
            Some(1)
        );
        assert_observation_manager_matches_fresh(
            &frames,
            &pairs,
            &reconstruction,
            &triangulation_state,
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires the external test_data/flowers2_colmap fixture"]
    fn real_colmap_sparse_global_ba_post_filter_keeps_observation_manager_in_sync() -> Result<()> {
        let (camera, frames, setup, mut pairs, _) = real_colmap_sparse_seed_local_ba_fixture()?;
        let mut reconstruction =
            reconstruction_from_reference_setup_for_test(&frames, camera, &setup);
        assert_eq!(reconstruction.points.len(), 256);

        let removed_track = reconstruction.points[0].track.clone();
        assert_eq!(removed_track.len(), 2);
        remove_track_features_from_pairs(&mut pairs, &removed_track);
        let orphaned_obs = removed_track[1];
        reconstruction.observations[orphaned_obs.image][orphaned_obs.feature] = None;
        reconstruction.points[0].track.truncate(1);

        let mut triangulation_state =
            IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        let mut stats = RegistrationStats::from_reconstruction(&reconstruction);
        let mut filtered_units = HashSet::new();
        let mut debug_log = Vec::new();
        let config = MapperConfig {
            global_ba: true,
            global_ba_iterations: 3,
            global_ba_max_refinements: 1,
            global_ba_images_freq: 999,
            global_ba_points_freq: 999_999,
            global_ba_images_ratio: 10.0,
            global_ba_points_ratio: 10.0,
            extract_colors: false,
            ..MapperConfig::default()
        };

        assert!(refine_global_bundle_with_postprocessing(
            &frames,
            &pairs,
            &mut reconstruction,
            &mapper_triangulator_options(&config),
            &config,
            "post_filter_fixture",
            false,
            &mut debug_log,
            Some(&mut stats),
            Some(&mut filtered_units),
            &mut triangulation_state,
        ));

        let global_ba = debug_log
            .iter()
            .find(|line| line.starts_with("global_ba reason=post_filter_fixture round=1"))
            .unwrap_or_else(|| panic!("missing global BA log: {debug_log:?}"));
        let filtered = global_ba
            .split_whitespace()
            .find_map(|field| field.strip_prefix("filtered="))
            .and_then(|value| value.parse::<usize>().ok())
            .expect("global BA filtered count");
        assert!(
            filtered > 0,
            "global BA post-filter should delete observations from the real fixture: {global_ba}"
        );
        assert!(reconstruction.points.len() < 256);
        for obs in removed_track {
            assert_eq!(reconstruction.observations[obs.image][obs.feature], None);
        }
        assert_observation_manager_matches_fresh(
            &frames,
            &pairs,
            &reconstruction,
            &triangulation_state,
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires the external test_data/flowers2_colmap fixture"]
    fn real_colmap_sparse_registered_frame_filter_keeps_observation_manager_in_sync() -> Result<()>
    {
        let (frames, pairs, mut reconstruction) = real_colmap_sparse_full_registration_fixture()?;
        assert_eq!(registered_frame_count(&reconstruction), 24);
        let target_image = reconstruction
            .image_names
            .iter()
            .position(|name| name == "frame_0009.jpg")
            .expect("registered real COLMAP target image");
        let target_observations = reconstruction.observations[target_image]
            .iter()
            .filter(|point| point.is_some())
            .count();
        assert!(
            target_observations > 0,
            "target image should start with true COLMAP sparse observations"
        );

        let mut triangulation_state =
            IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        assert_observation_manager_matches_fresh(
            &frames,
            &pairs,
            &reconstruction,
            &triangulation_state,
        );
        {
            let mut triangulator = IncrementalTriangulator::new(
                &frames,
                &pairs,
                &mut reconstruction,
                &mut triangulation_state,
            );
            let _ = triangulator.retriangulate(&IncrementalTriangulatorOptions {
                re_min_ratio: 0.5,
                re_max_trials: 1,
                min_angle_deg: 0.1,
                merge_max_reproj_error_px: 10.0,
                ignore_two_view_tracks: false,
                random_seed: 0,
                ..IncrementalTriangulatorOptions::default()
            });
        }
        let trial_snapshot = triangulation_state.retriangulation_trials().clone();

        let bogus_camera_id = 999_001;
        reconstruction.cameras.push(CameraModel::new_pinhole(
            1000, 1000, 1.0e9, 1.0e9, 500.0, 500.0,
        ));
        reconstruction.camera_ids.push(bogus_camera_id);
        reconstruction.image_camera_indices[target_image] = reconstruction.cameras.len() - 1;
        assert!(camera_has_bogus_params(
            reconstruction.camera_for_image(target_image),
            &MapperConfig::default()
        ));

        let mut stats = RegistrationStats::from_reconstruction(&reconstruction);
        let mut filtered_units = HashSet::new();
        let filtered = filter_registered_frames(
            &frames,
            &pairs,
            &mut reconstruction,
            &MapperConfig::default(),
            &mut stats,
            Some(&mut filtered_units),
            &mut triangulation_state,
        );

        assert_eq!(filtered, 1);
        assert!(filtered_units.contains(&RegistrationUnitKey::Frame(0)));
        assert_eq!(registered_frame_count(&reconstruction), 23);
        assert!(reconstruction.poses[target_image].is_none());
        assert!(reconstruction.observations[target_image]
            .iter()
            .all(|point| point.is_none()));
        assert_eq!(stats.registered_images_with_camera_id(bogus_camera_id), 0);
        assert_eq!(stats.num_total_reg_images, 23);
        assert_eq!(
            triangulation_state.retriangulation_trials(),
            &trial_snapshot
        );
        assert_observation_manager_matches_fresh(
            &frames,
            &pairs,
            &reconstruction,
            &triangulation_state,
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires the external test_data/flowers2_colmap fixture"]
    fn real_colmap_sparse_post_registration_filter_keeps_observation_manager_in_sync() -> Result<()>
    {
        let (camera, frames, setup, pairs, reference_pose) =
            real_colmap_sparse_seed_local_ba_fixture()?;
        let mut reconstruction =
            reconstruction_from_reference_setup_for_test(&frames, camera, &setup);
        assert_eq!(reconstruction.points.len(), 256);
        let mut triangulation_state =
            IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        assert!(triangulation_state
            .observation_manager_mut()
            .register_image(&frames, &pairs, &mut reconstruction, 2, reference_pose));
        let tri_options = mapper_triangulator_options(&MapperConfig {
            random_seed: 0,
            ..MapperConfig::default()
        });
        {
            let mut triangulator = IncrementalTriangulator::new(
                &frames,
                &pairs,
                &mut reconstruction,
                &mut triangulation_state,
            );
            let completed = triangulator.retriangulate(&tri_options);
            assert!(
                completed > 0,
                "real COLMAP sparse post-registration fixture should add candidate observations"
            );
        }
        assert_observation_manager_matches_fresh(
            &frames,
            &pairs,
            &reconstruction,
            &triangulation_state,
        );

        let target_point = reconstruction
            .points
            .iter()
            .position(|point| {
                point.track.len() >= 3 && point.track.iter().any(|obs| obs.image == 2)
            })
            .expect("continued real COLMAP candidate track");
        let removed_track = reconstruction.points[target_point].track.clone();
        reconstruction.points[target_point].xyz = [0.0, 0.0, -10.0];

        let removed = filter_reprojection_tracks_with_state(
            &frames,
            &pairs,
            &mut reconstruction,
            &MapperConfig::default(),
            &mut triangulation_state,
        );

        assert_eq!(removed, removed_track.len());
        assert_eq!(
            triangulation_state
                .retriangulation_trials()
                .get(&(0, 2))
                .copied(),
            Some(1)
        );
        for obs in removed_track {
            assert_eq!(reconstruction.observations[obs.image][obs.feature], None);
        }
        assert_observation_manager_matches_fresh(
            &frames,
            &pairs,
            &reconstruction,
            &triangulation_state,
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires the external test_data/flowers2_colmap fixture"]
    fn real_colmap_sparse_registration_rollback_keeps_observation_manager_in_sync() -> Result<()> {
        let (camera, frames, setup, pairs, reference_pose) =
            real_colmap_sparse_seed_local_ba_fixture()?;
        let mut reconstruction =
            reconstruction_from_reference_setup_for_test(&frames, camera, &setup);
        let registration_snapshot = reconstruction.clone();
        assert_eq!(reconstruction.points.len(), 256);
        assert!(reconstruction.poses[2].is_none());
        assert_eq!(
            reconstruction.observations[2]
                .iter()
                .filter(|point| point.is_some())
                .count(),
            0
        );

        let mut triangulation_state =
            IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        assert!(triangulation_state
            .observation_manager_mut()
            .register_image(&frames, &pairs, &mut reconstruction, 2, reference_pose));
        assert_observation_manager_matches_fresh(
            &frames,
            &pairs,
            &reconstruction,
            &triangulation_state,
        );
        let tri_options = mapper_triangulator_options(&MapperConfig {
            random_seed: 0,
            ..MapperConfig::default()
        });
        {
            let mut triangulator = IncrementalTriangulator::new(
                &frames,
                &pairs,
                &mut reconstruction,
                &mut triangulation_state,
            );
            let completed = triangulator.retriangulate(&tri_options);
            assert!(
                completed > 0,
                "real COLMAP sparse rollback fixture should add candidate observations"
            );
        }
        let candidate_observations = reconstruction.observations[2]
            .iter()
            .filter(|point| point.is_some())
            .count();
        assert!(
            candidate_observations > 0,
            "candidate observations should be present before rollback"
        );
        assert_eq!(
            triangulation_state
                .retriangulation_trials()
                .get(&(0, 2))
                .copied(),
            Some(1)
        );
        assert_observation_manager_matches_fresh(
            &frames,
            &pairs,
            &reconstruction,
            &triangulation_state,
        );

        assert_eq!(
            registration_rollback_reason(&reconstruction, 2, true, false, &MapperConfig::default()),
            Some("local_ba_failed")
        );
        reconstruction = registration_snapshot;
        triangulation_state.sync_after_reconstruction_rollback(&frames, &pairs, &reconstruction);

        assert!(reconstruction.poses[2].is_none());
        assert_eq!(
            reconstruction.observations[2]
                .iter()
                .filter(|point| point.is_some())
                .count(),
            0
        );
        assert_eq!(
            triangulation_state
                .retriangulation_trials()
                .get(&(0, 2))
                .copied(),
            Some(1)
        );
        assert_observation_manager_matches_fresh(
            &frames,
            &pairs,
            &reconstruction,
            &triangulation_state,
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires the external test_data/flowers2_colmap fixture"]
    fn real_colmap_sparse_bogus_camera_rollback_keeps_observation_manager_in_sync() -> Result<()> {
        let (camera, frames, setup, pairs, reference_pose) =
            real_colmap_sparse_seed_local_ba_fixture()?;
        let mut reconstruction =
            reconstruction_from_reference_setup_for_test(&frames, camera, &setup);
        let registration_snapshot = reconstruction.clone();

        let mut triangulation_state =
            IncrementalTriangulatorState::new(&frames, &pairs, &reconstruction);
        assert!(triangulation_state
            .observation_manager_mut()
            .register_image(&frames, &pairs, &mut reconstruction, 2, reference_pose));
        let tri_options = mapper_triangulator_options(&MapperConfig {
            random_seed: 0,
            ..MapperConfig::default()
        });
        {
            let mut triangulator = IncrementalTriangulator::new(
                &frames,
                &pairs,
                &mut reconstruction,
                &mut triangulation_state,
            );
            let completed = triangulator.retriangulate(&tri_options);
            assert!(
                completed > 0,
                "real COLMAP sparse rollback fixture should add candidate observations"
            );
        }
        let trial_snapshot = triangulation_state.retriangulation_trials().clone();
        assert_eq!(trial_snapshot.get(&(0, 2)).copied(), Some(1));
        assert_observation_manager_matches_fresh(
            &frames,
            &pairs,
            &reconstruction,
            &triangulation_state,
        );

        let bogus_camera_id = 999_002;
        reconstruction.cameras.push(CameraModel::new_pinhole(
            1000, 1000, 1.0e9, 1.0e9, 500.0, 500.0,
        ));
        reconstruction.camera_ids.push(bogus_camera_id);
        reconstruction.image_camera_indices[2] = reconstruction.cameras.len() - 1;
        assert!(registration_state_has_bogus_camera(
            &reconstruction,
            2,
            &MapperConfig::default()
        ));
        assert_eq!(
            registration_rollback_reason(
                &reconstruction,
                2,
                false,
                false,
                &MapperConfig::default()
            ),
            Some("bogus_camera")
        );

        reconstruction = registration_snapshot;
        triangulation_state.sync_after_reconstruction_rollback(&frames, &pairs, &reconstruction);

        assert!(reconstruction.poses[2].is_none());
        assert_eq!(
            reconstruction.observations[2]
                .iter()
                .filter(|point| point.is_some())
                .count(),
            0
        );
        assert_eq!(
            triangulation_state.retriangulation_trials(),
            &trial_snapshot
        );
        assert_observation_manager_matches_fresh(
            &frames,
            &pairs,
            &reconstruction,
            &triangulation_state,
        );
        Ok(())
    }

    fn remove_track_features_from_pairs(pairs: &mut [PairGeometry], track: &[TrackObservation]) {
        for pair in pairs {
            pair.matches.retain(|m| {
                !track.iter().any(|obs| {
                    (pair.left == obs.image && m.query_idx as usize == obs.feature)
                        || (pair.right == obs.image && m.train_idx as usize == obs.feature)
                })
            });
            pair.inlier_matches.retain(|m| {
                !track.iter().any(|obs| {
                    (pair.left == obs.image && m.query_idx as usize == obs.feature)
                        || (pair.right == obs.image && m.train_idx as usize == obs.feature)
                })
            });
            pair.inliers = pair.inlier_matches.len();
            pair.triangulated = pair.triangulated.min(pair.inliers);
        }
    }

    fn real_colmap_sparse_seed_mapper_fixture() -> Result<(
        CameraModel,
        Vec<ImageFrame>,
        ReferenceCameraSetup,
        Vec<PairGeometry>,
        SE3,
    )> {
        let sparse =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../test_data/flowers2_colmap/sparse/text");
        let reference = read_colmap_sparse_model(&sparse)?.reconstruction;
        let seed_ref = reference
            .image_names
            .iter()
            .position(|name| name == "frame_0002.jpg")
            .expect("flowers2 seed image");
        let candidate_ref = reference
            .image_names
            .iter()
            .position(|name| name == "frame_0003.jpg")
            .expect("flowers2 candidate image");
        let seed_pose = reference.poses[seed_ref].expect("registered seed pose");
        let candidate_pose = reference.poses[candidate_ref].expect("registered candidate pose");
        let frames = vec![
            image_frame_from_reference(&reference, seed_ref, 0),
            image_frame_from_reference(&reference, candidate_ref, 1),
        ];

        let mut candidate_feature_by_point = HashMap::<usize, usize>::new();
        for (feature, point_idx) in reference.observations[candidate_ref].iter().enumerate() {
            if let Some(point_idx) = point_idx {
                candidate_feature_by_point
                    .entry(*point_idx)
                    .or_insert(feature);
            }
        }

        let mut observations = vec![
            vec![None; frames[0].keypoints.len()],
            vec![None; frames[1].keypoints.len()],
        ];
        let mut point_ids = Vec::new();
        let mut points = Vec::new();
        let mut matches = Vec::new();
        for (seed_feature, point_idx) in reference.observations[seed_ref].iter().enumerate() {
            let Some(reference_point_idx) = *point_idx else {
                continue;
            };
            let Some(&candidate_feature) = candidate_feature_by_point.get(&reference_point_idx)
            else {
                continue;
            };
            let seed_point_idx = points.len();
            observations[0][seed_feature] = Some(seed_point_idx);
            let point = &reference.points[reference_point_idx];
            point_ids.push(reference.point3d_id(reference_point_idx));
            points.push(Point3D {
                xyz: point.xyz,
                color: point.color,
                error: point.error,
                track: vec![TrackObservation {
                    image: 0,
                    feature: seed_feature,
                }],
            });
            matches.push(rustslam::Match {
                query_idx: seed_feature as u32,
                train_idx: candidate_feature as u32,
                distance: 0.0,
            });
            if matches.len() == 256 {
                break;
            }
        }
        assert_eq!(matches.len(), 256);

        let relative_pose = candidate_pose.compose(&seed_pose.inverse());
        let pair = PairGeometry {
            left: 0,
            right: 1,
            two_view_config: crate::database::COLMAP_TWO_VIEW_CALIBRATED,
            f_matrix: None,
            e_matrix: None,
            h_matrix: None,
            qvec: None,
            tvec: None,
            matches: matches.clone(),
            inlier_matches: matches,
            relative_pose,
            inliers: 256,
            triangulated: 256,
            mean_reprojection_error_px: 0.0,
            rotation_deg: relative_rotation_deg(relative_pose, SE3::identity()),
            median_triangulation_angle_deg: 6.0,
            pose_graph_only: false,
        };
        let setup = ReferenceCameraSetup {
            cameras: reference.cameras.clone(),
            camera_ids: reference.camera_ids.clone(),
            camera_has_prior_focal_length: vec![true; reference.cameras.len()],
            rigs: Vec::new(),
            frames: Vec::new(),
            image_ids: vec![
                reference.image_id(seed_ref),
                reference.image_id(candidate_ref),
            ],
            image_camera_indices: vec![
                reference.image_camera_indices[seed_ref],
                reference.image_camera_indices[candidate_ref],
            ],
            image_frame_indices: vec![None, None],
            seed_reconstruction: Some(ReconstructionSeed {
                poses: vec![Some(seed_pose), None],
                observations,
                point_ids,
                points,
            }),
        };

        Ok((
            reference.camera_for_image(seed_ref),
            frames,
            setup,
            vec![pair],
            candidate_pose,
        ))
    }

    fn real_colmap_sparse_seed_local_ba_fixture() -> Result<(
        CameraModel,
        Vec<ImageFrame>,
        ReferenceCameraSetup,
        Vec<PairGeometry>,
        SE3,
    )> {
        let sparse =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../test_data/flowers2_colmap/sparse/text");
        let reference = read_colmap_sparse_model(&sparse)?.reconstruction;
        let gauge_ref = reference
            .image_names
            .iter()
            .position(|name| name == "frame_0001.jpg")
            .expect("flowers2 gauge image");
        let seed_ref = reference
            .image_names
            .iter()
            .position(|name| name == "frame_0002.jpg")
            .expect("flowers2 seed image");
        let candidate_ref = reference
            .image_names
            .iter()
            .position(|name| name == "frame_0003.jpg")
            .expect("flowers2 candidate image");
        let gauge_pose = reference.poses[gauge_ref].expect("registered gauge pose");
        let seed_pose = reference.poses[seed_ref].expect("registered seed pose");
        let candidate_pose = reference.poses[candidate_ref].expect("registered candidate pose");
        let frames = vec![
            image_frame_from_reference(&reference, gauge_ref, 0),
            image_frame_from_reference(&reference, seed_ref, 1),
            image_frame_from_reference(&reference, candidate_ref, 2),
        ];

        let mut seed_feature_by_point = HashMap::<usize, usize>::new();
        for (feature, point_idx) in reference.observations[seed_ref].iter().enumerate() {
            if let Some(point_idx) = point_idx {
                seed_feature_by_point.entry(*point_idx).or_insert(feature);
            }
        }
        let mut candidate_feature_by_point = HashMap::<usize, usize>::new();
        for (feature, point_idx) in reference.observations[candidate_ref].iter().enumerate() {
            if let Some(point_idx) = point_idx {
                candidate_feature_by_point
                    .entry(*point_idx)
                    .or_insert(feature);
            }
        }

        let mut observations = vec![
            vec![None; frames[0].keypoints.len()],
            vec![None; frames[1].keypoints.len()],
            vec![None; frames[2].keypoints.len()],
        ];
        let mut point_ids = Vec::new();
        let mut points = Vec::new();
        let mut gauge_candidate_matches = Vec::new();
        let mut seed_candidate_matches = Vec::new();
        for (gauge_feature, point_idx) in reference.observations[gauge_ref].iter().enumerate() {
            let Some(reference_point_idx) = *point_idx else {
                continue;
            };
            let Some(&seed_feature) = seed_feature_by_point.get(&reference_point_idx) else {
                continue;
            };
            let Some(&candidate_feature) = candidate_feature_by_point.get(&reference_point_idx)
            else {
                continue;
            };
            let seed_point_idx = points.len();
            observations[0][gauge_feature] = Some(seed_point_idx);
            observations[1][seed_feature] = Some(seed_point_idx);
            let point = &reference.points[reference_point_idx];
            point_ids.push(reference.point3d_id(reference_point_idx));
            points.push(Point3D {
                xyz: point.xyz,
                color: point.color,
                error: point.error,
                track: vec![
                    TrackObservation {
                        image: 0,
                        feature: gauge_feature,
                    },
                    TrackObservation {
                        image: 1,
                        feature: seed_feature,
                    },
                ],
            });
            gauge_candidate_matches.push(rustslam::Match {
                query_idx: gauge_feature as u32,
                train_idx: candidate_feature as u32,
                distance: 0.0,
            });
            seed_candidate_matches.push(rustslam::Match {
                query_idx: seed_feature as u32,
                train_idx: candidate_feature as u32,
                distance: 0.0,
            });
            if points.len() == 256 {
                break;
            }
        }
        assert_eq!(points.len(), 256);

        let gauge_candidate_pair =
            real_colmap_sparse_seed_pair(0, 2, gauge_pose, candidate_pose, gauge_candidate_matches);
        let seed_candidate_pair =
            real_colmap_sparse_seed_pair(1, 2, seed_pose, candidate_pose, seed_candidate_matches);
        let setup = ReferenceCameraSetup {
            cameras: reference.cameras.clone(),
            camera_ids: reference.camera_ids.clone(),
            camera_has_prior_focal_length: vec![true; reference.cameras.len()],
            rigs: Vec::new(),
            frames: Vec::new(),
            image_ids: vec![
                reference.image_id(gauge_ref),
                reference.image_id(seed_ref),
                reference.image_id(candidate_ref),
            ],
            image_camera_indices: vec![
                reference.image_camera_indices[gauge_ref],
                reference.image_camera_indices[seed_ref],
                reference.image_camera_indices[candidate_ref],
            ],
            image_frame_indices: vec![None, None, None],
            seed_reconstruction: Some(ReconstructionSeed {
                poses: vec![Some(gauge_pose), Some(seed_pose), None],
                observations,
                point_ids,
                points,
            }),
        };

        Ok((
            reference.camera_for_image(candidate_ref),
            frames,
            setup,
            vec![gauge_candidate_pair, seed_candidate_pair],
            candidate_pose,
        ))
    }

    fn real_colmap_sparse_full_registration_fixture(
    ) -> Result<(Vec<ImageFrame>, Vec<PairGeometry>, Reconstruction)> {
        let sparse =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../test_data/flowers2_colmap/sparse/text");
        let reference = read_colmap_sparse_model(&sparse)?.reconstruction;
        assert_eq!(
            reference.poses.iter().filter(|pose| pose.is_some()).count(),
            24
        );
        let frames = (0..reference.image_names.len())
            .map(|image| image_frame_from_reference(&reference, image, image))
            .collect::<Vec<_>>();
        let pairs = pair_geometries_from_reference_tracks(&reference);
        assert!(
            pairs.iter().any(|pair| pair.left == 0 || pair.right == 0),
            "real COLMAP fixture should include correspondences for the target image"
        );
        Ok((frames, pairs, reference))
    }

    fn pair_geometries_from_reference_tracks(reference: &Reconstruction) -> Vec<PairGeometry> {
        let mut matches_by_pair = BTreeMap::<(usize, usize), Vec<rustslam::Match>>::new();
        for point in &reference.points {
            for i in 0..point.track.len() {
                for j in (i + 1)..point.track.len() {
                    let obs1 = point.track[i];
                    let obs2 = point.track[j];
                    if obs1.image == obs2.image {
                        continue;
                    }
                    let (left, left_feature, right, right_feature) = if obs1.image < obs2.image {
                        (obs1.image, obs1.feature, obs2.image, obs2.feature)
                    } else {
                        (obs2.image, obs2.feature, obs1.image, obs1.feature)
                    };
                    matches_by_pair
                        .entry((left, right))
                        .or_default()
                        .push(rustslam::Match {
                            query_idx: left_feature as u32,
                            train_idx: right_feature as u32,
                            distance: 0.0,
                        });
                }
            }
        }

        matches_by_pair
            .into_iter()
            .filter_map(|((left, right), mut matches)| {
                matches.sort_by_key(|m| (m.query_idx, m.train_idx));
                matches.dedup_by_key(|m| (m.query_idx, m.train_idx));
                if matches.is_empty() {
                    return None;
                }
                let left_pose = reference.poses[left]?;
                let right_pose = reference.poses[right]?;
                Some(real_colmap_sparse_seed_pair(
                    left, right, left_pose, right_pose, matches,
                ))
            })
            .collect()
    }

    fn real_colmap_sparse_seed_pair(
        left: usize,
        right: usize,
        left_pose: SE3,
        right_pose: SE3,
        matches: Vec<rustslam::Match>,
    ) -> PairGeometry {
        let inliers = matches.len();
        let relative_pose = right_pose.compose(&left_pose.inverse());
        PairGeometry {
            left,
            right,
            two_view_config: crate::database::COLMAP_TWO_VIEW_CALIBRATED,
            f_matrix: None,
            e_matrix: None,
            h_matrix: None,
            qvec: None,
            tvec: None,
            matches: matches.clone(),
            inlier_matches: matches,
            relative_pose,
            inliers,
            triangulated: inliers,
            mean_reprojection_error_px: 0.0,
            rotation_deg: relative_rotation_deg(relative_pose, SE3::identity()),
            median_triangulation_angle_deg: 6.0,
            pose_graph_only: false,
        }
    }

    fn image_frame_from_reference(
        reference: &Reconstruction,
        reference_image: usize,
        current_id: usize,
    ) -> ImageFrame {
        let camera = reference.camera_for_image(reference_image);
        let keypoints = reference.keypoints[reference_image].clone();
        ImageFrame {
            id: current_id,
            name: reference.image_names[reference_image].clone(),
            path: PathBuf::from(&reference.image_names[reference_image]),
            width: camera.width,
            height: camera.height,
            keypoints,
            descriptors: rustslam::Descriptors::new(),
            sift: crate::sift::SiftFeatures::default(),
            wide_descriptors: crate::wide::WideDescriptors {
                data: Vec::new(),
                dim: 0,
                count: 0,
            },
            strong_feature_indices: Vec::new(),
            colors: vec![[0, 0, 0]; reference.keypoints[reference_image].len()],
        }
    }

    fn pose_translation_error(a: SE3, b: SE3) -> f32 {
        let ta = a.translation();
        let tb = b.translation();
        ((ta[0] - tb[0]).powi(2) + (ta[1] - tb[1]).powi(2) + (ta[2] - tb[2]).powi(2)).sqrt()
    }

    fn pose_translation_direction_error_deg(a: SE3, b: SE3) -> f32 {
        let ta = glam::Vec3::from_array(a.translation());
        let tb = glam::Vec3::from_array(b.translation());
        let Some(ta) = ta.try_normalize() else {
            return f32::INFINITY;
        };
        let Some(tb) = tb.try_normalize() else {
            return f32::INFINITY;
        };
        ta.dot(tb).clamp(-1.0, 1.0).acos().to_degrees()
    }

    fn position_prior_covariance(stddev: f64) -> [f64; 9] {
        let variance = stddev * stddev;
        [variance, 0.0, 0.0, 0.0, variance, 0.0, 0.0, 0.0, variance]
    }

    fn colmap_rig_frame_seed_fixture() -> (
        CameraModel,
        [SE3; 4],
        Vec<[f32; 3]>,
        Vec<ImageFrame>,
        ColmapSparseFiles,
    ) {
        let camera = CameraModel::new_pinhole(200, 160, 80.0, 80.0, 100.0, 80.0);
        let poses = [
            SE3::identity(),
            SE3::from_quat_translation(glam::Quat::IDENTITY, glam::Vec3::new(-0.35, 0.0, 0.0)),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(0.03),
                glam::Vec3::new(0.45, 0.0, 0.0),
            ),
            SE3::from_quat_translation(
                glam::Quat::from_rotation_y(-0.02),
                glam::Vec3::new(0.9, 0.0, 0.0),
            ),
        ];
        let points = (0..12)
            .map(|idx| {
                let col = (idx % 4) as f32;
                let row = (idx / 4) as f32;
                [-0.3 + col * 0.2, -0.2 + row * 0.18, 3.0 + idx as f32 * 0.03]
            })
            .collect::<Vec<_>>();
        let mut frames = (0..4)
            .map(|idx| minimal_frame(idx, &format!("image_{idx}.jpg")))
            .collect::<Vec<_>>();
        for (image, pose) in poses.iter().copied().enumerate() {
            frames[image].width = camera.width;
            frames[image].height = camera.height;
            frames[image].keypoints = points
                .iter()
                .map(|&point| project_test_point(camera, pose, point))
                .collect();
            frames[image].colors = vec![[image as u8, 0, 0]; points.len()];
        }

        let ref_sensor = ColmapSensorId {
            sensor_type: ColmapSensorType::Camera,
            sensor_id: 11,
        };
        let aux_sensor = ColmapSensorId {
            sensor_type: ColmapSensorType::Camera,
            sensor_id: 12,
        };
        let sparse_model = ColmapSparseFiles {
            cameras: vec![ColmapCamera {
                camera_id: 11,
                model_id: camera.model_id,
                width: camera.width,
                height: camera.height,
                params: camera.params[..camera.num_params].to_vec(),
            }],
            rigs: vec![ColmapRig {
                rig_id: 77,
                ref_sensor_id: Some(ref_sensor.clone()),
                sensors: vec![
                    ColmapRigSensor {
                        sensor_id: ref_sensor.clone(),
                        sensor_from_rig: None,
                    },
                    ColmapRigSensor {
                        sensor_id: aux_sensor.clone(),
                        sensor_from_rig: Some(ColmapRigid3 {
                            qvec: [1.0, 0.0, 0.0, 0.0],
                            tvec: [0.35, 0.0, 0.0],
                        }),
                    },
                ],
            }],
            frames: vec![ColmapFrame {
                frame_id: 99,
                rig_id: 77,
                rig_from_world: ColmapRigid3 {
                    qvec: [1.0, 0.0, 0.0, 0.0],
                    tvec: [0.0, 0.0, 0.0],
                },
                data_ids: vec![
                    ColmapDataId {
                        sensor_id: ref_sensor,
                        data_id: 101,
                    },
                    ColmapDataId {
                        sensor_id: aux_sensor,
                        data_id: 205,
                    },
                ],
            }],
            images: vec![
                ColmapImage {
                    image_id: 101,
                    camera_id: 11,
                    name: "image_0.jpg".to_string(),
                    qvec: [1.0, 0.0, 0.0, 0.0],
                    tvec: [0.0, 0.0, 0.0],
                    points2d: vec![ColmapPoint2D {
                        xy: [
                            frames[0].keypoints[0].x() as f64,
                            frames[0].keypoints[0].y() as f64,
                        ],
                        point3d_id: Some(700),
                    }],
                },
                ColmapImage {
                    image_id: 205,
                    camera_id: 11,
                    name: "image_1.jpg".to_string(),
                    qvec: [1.0, 0.0, 0.0, 0.0],
                    tvec: [-0.35, 0.0, 0.0],
                    points2d: vec![ColmapPoint2D {
                        xy: [
                            frames[1].keypoints[0].x() as f64,
                            frames[1].keypoints[0].y() as f64,
                        ],
                        point3d_id: Some(700),
                    }],
                },
            ],
            points3d: vec![ColmapPoint3D {
                point3d_id: 700,
                xyz: [
                    points[0][0] as f64,
                    points[0][1] as f64,
                    points[0][2] as f64,
                ],
                color: [7, 7, 7],
                error: 0.0,
                track: vec![
                    ColmapTrackElement {
                        image_id: 101,
                        point2d_idx: 0,
                    },
                    ColmapTrackElement {
                        image_id: 205,
                        point2d_idx: 0,
                    },
                ],
            }],
        };
        (camera, poses, points, frames, sparse_model)
    }

    fn densify_colmap_rig_sparse_seed(
        sparse_model: &mut ColmapSparseFiles,
        frames: &[ImageFrame],
        points: &[[f32; 3]],
    ) {
        let seeded_images = [(101_u32, 0_usize), (205, 1)];
        for (image_id, frame_idx) in seeded_images {
            let Some(image) = sparse_model
                .images
                .iter_mut()
                .find(|image| image.image_id == image_id)
            else {
                continue;
            };
            image.points2d = (0..points.len())
                .map(|feature| ColmapPoint2D {
                    xy: [
                        frames[frame_idx].keypoints[feature].x() as f64,
                        frames[frame_idx].keypoints[feature].y() as f64,
                    ],
                    point3d_id: Some(700 + feature as u64),
                })
                .collect();
        }
        sparse_model.points3d = (0..points.len())
            .map(|feature| ColmapPoint3D {
                point3d_id: 700 + feature as u64,
                xyz: [
                    points[feature][0] as f64,
                    points[feature][1] as f64,
                    points[feature][2] as f64,
                ],
                color: [7, 7, 7],
                error: 0.0,
                track: vec![
                    ColmapTrackElement {
                        image_id: 101,
                        point2d_idx: feature as u64,
                    },
                    ColmapTrackElement {
                        image_id: 205,
                        point2d_idx: feature as u64,
                    },
                ],
            })
            .collect();
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
            f_matrix: None,
            e_matrix: None,
            h_matrix: None,
            qvec: None,
            tvec: None,
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
            f_matrix: None,
            e_matrix: None,
            h_matrix: None,
            qvec: None,
            tvec: None,
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

    fn initial_pair_from_projected_points(
        left: usize,
        right: usize,
        left_pose: SE3,
        right_pose: SE3,
        inliers: usize,
    ) -> PairGeometry {
        let mut pair = structureless_pair_from_poses(left, right, left_pose, right_pose, inliers);
        pair.triangulated = inliers;
        pair.median_triangulation_angle_deg = 6.0;
        pair
    }

    fn camera_priors(reconstruction: &Reconstruction) -> Vec<CameraModel> {
        reconstruction.cameras.clone()
    }

    fn camera_prior_focal_flags(reconstruction: &Reconstruction, value: bool) -> Vec<bool> {
        vec![value; reconstruction.cameras.len()]
    }

    fn log_index(log: &[String], expected: &str) -> usize {
        log.iter()
            .position(|line| line == expected)
            .unwrap_or_else(|| panic!("missing log line: {expected}"))
    }

    fn choose_initial_pair_for_test(
        pairs: &[PairGeometry],
        reconstruction: &Reconstruction,
        config: &MapperConfig,
        camera_has_prior_focal_length: &[bool],
    ) -> Option<PairGeometry> {
        let mut selection_state = InitialPairSelectionState::from_reconstruction(reconstruction);
        let mut config = config.clone();
        config.threads.get_or_insert(1);
        choose_initial_pair(
            pairs,
            reconstruction,
            &config,
            camera_has_prior_focal_length,
            &mut selection_state,
        )
    }

    fn registration_stats(reconstruction: &Reconstruction) -> RegistrationStats {
        RegistrationStats::from_reconstruction(reconstruction)
    }

    fn assert_observation_manager_matches_fresh(
        frames: &[ImageFrame],
        pairs: &[PairGeometry],
        reconstruction: &Reconstruction,
        state: &IncrementalTriangulatorState,
    ) {
        let fresh_manager = ObservationManager::new(frames, pairs, reconstruction);
        let state_manager = state.observation_manager();
        assert_eq!(state_manager.image_pairs(), fresh_manager.image_pairs());
        for image in 0..frames.len() {
            assert_eq!(
                state_manager.num_observations(image),
                fresh_manager.num_observations(image),
                "observations image {image}"
            );
            assert_eq!(
                state_manager.num_correspondences(image),
                fresh_manager.num_correspondences(image),
                "correspondences image {image}"
            );
            assert_eq!(
                state_manager.num_visible_correspondences(image),
                fresh_manager.num_visible_correspondences(image),
                "visible correspondences image {image}"
            );
            assert_eq!(
                state_manager.num_visible_points3d(image),
                fresh_manager.num_visible_points3d(image),
                "visible points image {image}"
            );
            assert_eq!(
                state_manager.point3d_visibility_score(image),
                fresh_manager.point3d_visibility_score(image),
                "visibility score image {image}"
            );
            for feature in 0..frames[image].keypoints.len() {
                assert_eq!(
                    state_manager.num_correspondences_have_point3d(image, feature),
                    fresh_manager.num_correspondences_have_point3d(image, feature),
                    "correspondence point3D count image {image} feature {feature}"
                );
            }
        }
    }

    fn test_pose_prior(
        pose_prior_id: u32,
        camera_id: u32,
        image_id: u64,
        position: [f64; 3],
        position_covariance: [f64; 9],
    ) -> ColmapPosePrior {
        ColmapPosePrior {
            pose_prior_id,
            corr_data_id: ColmapDataId {
                sensor_id: ColmapSensorId {
                    sensor_type: ColmapSensorType::Camera,
                    sensor_id: camera_id,
                },
                data_id: image_id,
            },
            position,
            position_covariance,
            coordinate_system: ColmapPosePriorCoordinateSystem::Cartesian,
            gravity: [f64::NAN; 3],
        }
    }

    fn reconstruction_from_reference_setup_for_test(
        frames: &[ImageFrame],
        camera: CameraModel,
        setup: &ReferenceCameraSetup,
    ) -> Reconstruction {
        let mut reconstruction = Reconstruction {
            camera,
            cameras: setup.cameras.clone(),
            camera_ids: setup.camera_ids.clone(),
            rigs: setup.rigs.clone(),
            frames: setup.frames.clone(),
            image_names: frames.iter().map(|f| f.name.clone()).collect(),
            image_paths: frames.iter().map(|f| f.path.clone()).collect(),
            image_ids: setup.image_ids.clone(),
            image_camera_indices: setup.image_camera_indices.clone(),
            image_frame_indices: setup.image_frame_indices.clone(),
            poses: vec![None; frames.len()],
            observations: frames
                .iter()
                .map(|f| vec![None; f.keypoints.len()])
                .collect(),
            keypoints: frames.iter().map(|f| f.keypoints.clone()).collect(),
            point_ids: Vec::new(),
            points: Vec::new(),
        };
        if let Some(seed) = setup.seed_reconstruction.clone() {
            apply_reconstruction_seed(&mut reconstruction, seed, frames);
        }
        reconstruction
    }

    fn attach_two_image_rig_frame(
        reconstruction: &mut Reconstruction,
        ref_image: usize,
        aux_image: usize,
    ) {
        let ref_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: reconstruction.image_id(ref_image),
        };
        let aux_sensor = SensorId {
            sensor_type: SensorType::Camera,
            sensor_id: reconstruction.image_id(aux_image),
        };
        reconstruction.rigs = vec![Rig {
            rig_id: 99,
            ref_sensor_id: Some(ref_sensor.clone()),
            sensors: vec![
                RigSensor {
                    sensor_id: ref_sensor.clone(),
                    sensor_from_rig: None,
                },
                RigSensor {
                    sensor_id: aux_sensor.clone(),
                    sensor_from_rig: Some(Rigid3::identity()),
                },
            ],
        }];
        reconstruction.frames = vec![Frame {
            frame_id: 199,
            rig_id: 99,
            rig_from_world: Rigid3::identity(),
            data_ids: vec![
                DataId {
                    sensor_id: ref_sensor,
                    data_id: reconstruction.image_id(ref_image) as u64,
                },
                DataId {
                    sensor_id: aux_sensor,
                    data_id: reconstruction.image_id(aux_image) as u64,
                },
            ],
        }];
        reconstruction.image_frame_indices = vec![None; reconstruction.poses.len()];
        reconstruction.image_frame_indices[ref_image] = Some(0);
        reconstruction.image_frame_indices[aux_image] = Some(0);
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

    fn add_test_point3d(
        reconstruction: &mut Reconstruction,
        stable_point_id: u64,
        track: Vec<TrackObservation>,
    ) -> usize {
        let point_id = reconstruction.points.len();
        for obs in &track {
            if let Some(image_observations) = reconstruction.observations.get_mut(obs.image) {
                if obs.feature >= image_observations.len() {
                    image_observations.resize(obs.feature + 1, None);
                }
                image_observations[obs.feature] = Some(point_id);
            }
        }
        reconstruction.point_ids.push(stable_point_id);
        reconstruction.points.push(Point3D {
            xyz: [point_id as f32, 0.0, 1.0],
            color: [0, 0, 0],
            error: 0.0,
            track,
        });
        point_id
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
