use super::*;
use std::fmt;

pub(super) fn global_reconstruction_options_from_config(
    config: &MapperConfig,
) -> GlobalReconstructionOptions {
    GlobalReconstructionOptions {
        view_graph_calibration: ViewGraphCalibrationOptions {
            enabled: true,
            max_epipolar_error_px: config.essential_threshold_px,
            min_inliers_per_pair: config.min_inliers,
            essential_threshold_px: config.essential_threshold_px,
            essential_iterations: config.essential_iterations.min(512),
            min_triangulated_per_pair: config.min_triangulated,
            refine_relative_poses: false,
            ..ViewGraphCalibrationOptions::default()
        },
        mapper: GlobalMapperOptions {
            rotation_averaging: RotationAveragingOptions::default(),
            ..GlobalMapperOptions::default()
        },
        tracks: TrackEstablishmentOptions {
            min_track_length: 2,
            max_track_length: 0,
        },
        triangulation: TrackTriangulationOptions {
            // `init_min_tri_angle_deg` is for incremental init-pair selection (default 16°),
            // not global point establishment (COLMAP/GLOMAP use ~1.5°).
            min_triangulation_angle_deg: TrackTriangulationOptions::default()
                .min_triangulation_angle_deg,
            max_reprojection_error_px: std::env::var("RUSTSFM_GLOBAL_TRIANGULATION_MAX_REPROJ_PX")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .filter(|value| value.is_finite() && *value > 0.0)
                .unwrap_or(
                    config
                        .max_reprojection_error_px
                        .min(TrackTriangulationOptions::default().max_reprojection_error_px),
                ),
            require_cheirality: true,
        },
        use_joint_positioning: true,
        joint_positioning: JointGlobalPositioningOptions::default(),
        refinement: GlobalRefinementOptions {
            max_refinements: config.global_ba_max_refinements,
            max_refinement_change: std::env::var("RUSTSFM_GLOBAL_BA_MAX_REFINEMENT_CHANGE")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .filter(|value| value.is_finite() && *value >= 0.0)
                .unwrap_or(config.global_ba_max_refinement_change),
            filter_max_reprojection_error_px: std::env::var("RUSTSFM_GLOBAL_FILTER_MAX_REPROJ_PX")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .filter(|value| value.is_finite() && *value > 0.0)
                .unwrap_or(
                    config
                        .max_reprojection_error_px
                        .min(GlobalRefinementOptions::default().filter_max_reprojection_error_px),
                ),
            filter_min_track_length: GlobalRefinementOptions::default().filter_min_track_length,
            filter_min_triangulation_angle_deg: TrackTriangulationOptions::default()
                .min_triangulation_angle_deg,
            complete_max_reprojection_error_px: std::env::var(
                "RUSTSFM_GLOBAL_COMPLETE_MAX_REPROJ_PX",
            )
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(
                config
                    .max_reprojection_error_px
                    .min(GlobalRefinementOptions::default().complete_max_reprojection_error_px),
            ),
        },
        incremental_triangulation: {
            let max_reproj = std::env::var("RUSTSFM_GLOBAL_TRIANGULATION_MAX_REPROJ_PX")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .filter(|value| value.is_finite() && *value > 0.0)
                .unwrap_or(
                    config
                        .max_reprojection_error_px
                        .min(TrackTriangulationOptions::default().max_reprojection_error_px),
                );
            let mut tri = IncrementalTriangulatorOptions::from_mapper_config_values(
                max_reproj,
                config.min_focal_length_ratio,
                config.max_focal_length_ratio,
                config.max_extra_param,
                config.random_seed,
                std::env::var("RUSTSFM_GLOBAL_IGNORE_TWO_VIEW_TRACKS")
                    .ok()
                    .and_then(|value| value.parse::<bool>().ok())
                    .unwrap_or(true),
                1,
            );
            tri.re_max_trials = std::env::var("RUSTSFM_GLOBAL_RE_MAX_TRIALS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(tri.re_max_trials);
            tri
        },
        run_global_ba: config.global_ba,
        global_ba_iterations: global_ba_iterations(config),
        component_splitting: ViewGraphComponentSplittingOptions {
            enabled: config.multiple_models,
            min_component_size: config.min_model_size.max(2),
            max_components: if config.multiple_models {
                config.max_num_models.max(1)
            } else {
                1
            },
        },
    }
}

pub(super) fn global_ba_iterations(config: &MapperConfig) -> usize {
    std::env::var("RUSTSFM_BA_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(config.global_ba_iterations)
}

pub(super) fn global_ba_iterations_for_reconstruction(
    config: &MapperConfig,
    reconstruction: &Reconstruction,
) -> usize {
    let base = global_ba_iterations(config);
    let observations = reconstruction_num_observations(reconstruction);
    if observations >= 20_000 {
        base.saturating_mul(3)
    } else if observations >= 10_000 {
        base.saturating_mul(2)
    } else {
        base
    }
}

pub(super) fn global_ba_huber_delta_px() -> f64 {
    std::env::var("RUSTSFM_BA_HUBER_PX")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(4.0)
}

/// COLMAP's incremental pipeline uses a trivial loss for mapper BA by default.
/// The loss type and scale can be overridden via
/// `RUSTSFM_BA_LOSS` (`trivial|huber|softl1|cauchy`) and
/// `RUSTSFM_BA_LOSS_SCALE`; for backward compatibility a `huber` selection still
/// honors `RUSTSFM_BA_HUBER_PX` when no explicit scale is provided.
pub(super) fn mapper_ba_loss_function(
    colmap_default: crate::ba::BundleAdjustmentLoss,
) -> crate::ba::BundleAdjustmentLoss {
    use crate::ba::BundleAdjustmentLoss;
    let kind = std::env::var("RUSTSFM_BA_LOSS")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase());
    let scale_override = std::env::var("RUSTSFM_BA_LOSS_SCALE")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0);
    match kind.as_deref() {
        Some("trivial") => BundleAdjustmentLoss::Trivial,
        Some("huber") => BundleAdjustmentLoss::Huber {
            scale: scale_override.unwrap_or_else(global_ba_huber_delta_px),
        },
        Some("softl1") | Some("soft_l1") => BundleAdjustmentLoss::SoftL1 {
            scale: scale_override.unwrap_or(1.0),
        },
        Some("cauchy") => BundleAdjustmentLoss::Cauchy {
            scale: scale_override.unwrap_or(1.0),
        },
        None | Some(_) => mapper_ba_loss_with_scale_override(colmap_default, scale_override),
    }
}

pub(super) fn mapper_ba_loss_with_scale_override(
    loss: crate::ba::BundleAdjustmentLoss,
    scale_override: Option<f64>,
) -> crate::ba::BundleAdjustmentLoss {
    match loss {
        crate::ba::BundleAdjustmentLoss::Trivial => crate::ba::BundleAdjustmentLoss::Trivial,
        crate::ba::BundleAdjustmentLoss::Huber { scale } => {
            crate::ba::BundleAdjustmentLoss::Huber {
                scale: scale_override.unwrap_or(scale),
            }
        }
        crate::ba::BundleAdjustmentLoss::SoftL1 { scale } => {
            crate::ba::BundleAdjustmentLoss::SoftL1 {
                scale: scale_override.unwrap_or(scale),
            }
        }
        crate::ba::BundleAdjustmentLoss::Cauchy { scale } => {
            crate::ba::BundleAdjustmentLoss::Cauchy {
                scale: scale_override.unwrap_or(scale),
            }
        }
    }
}

pub(super) fn mapper_global_ba_loss_function() -> crate::ba::BundleAdjustmentLoss {
    mapper_ba_loss_function(crate::ba::BundleAdjustmentLoss::Trivial)
}

pub(super) fn mapper_local_ba_loss_function() -> crate::ba::BundleAdjustmentLoss {
    mapper_ba_loss_function(crate::ba::BundleAdjustmentLoss::Trivial)
}

pub(super) fn mapper_local_ba_refinement_loss_function(
    round: usize,
) -> crate::ba::BundleAdjustmentLoss {
    if round == 0 {
        mapper_local_ba_loss_function()
    } else {
        crate::ba::BundleAdjustmentLoss::Trivial
    }
}

pub(super) fn global_ba_max_observation_error_px(config: &MapperConfig) -> f64 {
    std::env::var("RUSTSFM_BA_MAX_OBS_ERROR_PX")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(config.max_reprojection_error_px as f64 * 2.0)
}

pub(super) fn mapper_triangulator_options(config: &MapperConfig) -> IncrementalTriangulatorOptions {
    IncrementalTriangulatorOptions::from_mapper_config_values(
        config.max_reprojection_error_px,
        config.min_focal_length_ratio,
        config.max_focal_length_ratio,
        config.max_extra_param,
        config.random_seed,
        config.ignore_two_view_tracks,
        // COLMAP's `EstimateTriangulation` uses `CombinationSampler`, and
        // LORANSAC parallel execution is only supported for `RandomSampler`.
        1,
    )
}

pub(super) fn mapper_ba_options(
    config: &MapperConfig,
    reconstruction: &Reconstruction,
    iterations: usize,
    variable_images: Option<Vec<usize>>,
    constant_images: Vec<usize>,
    point_ids: Option<Vec<usize>>,
    constant_point_ids: Option<Vec<usize>>,
) -> crate::ba::BundleAdjustmentOptions {
    let constant_images = expanded_constant_images(reconstruction, constant_images);
    let pose_priors = mapper_pose_priors(
        config,
        reconstruction,
        variable_images.as_deref(),
        &constant_images,
    );
    let mut options = crate::ba::BundleAdjustmentOptions {
        iterations,
        loss_function: mapper_global_ba_loss_function(),
        max_observation_error_px: global_ba_max_observation_error_px(config),
        variable_images,
        constant_images,
        pose_priors,
        variable_cameras: None,
        constant_cameras: ba_constant_camera_indices(config, reconstruction),
        constant_rigs: ba_constant_rig_ids(config, reconstruction),
        constant_sensor_from_rig: ba_constant_sensor_from_rig_ids(config, reconstruction),
        refine_focal_length: config.ba_refine_focal_length,
        refine_principal_point: config.ba_refine_principal_point,
        refine_extra_params: config.ba_refine_extra_params,
        num_threads: config.threads.map(|threads| threads as isize).unwrap_or(-1),
        point_ids,
        constant_point_ids,
        ..crate::ba::BundleAdjustmentOptions::default()
    };
    apply_colmap_global_ba_solver_options(&mut options);
    options
}

pub(super) fn mapper_pose_priors(
    config: &MapperConfig,
    reconstruction: &Reconstruction,
    variable_images: Option<&[usize]>,
    constant_images: &[usize],
) -> Vec<crate::ba::BundleAdjustmentPosePrior> {
    if config.pose_priors.is_empty() {
        return Vec::new();
    }

    let variable_set = variable_images.map(|images| images.iter().copied().collect::<HashSet<_>>());
    let constant_set = constant_images.iter().copied().collect::<HashSet<_>>();
    let mut priors = config
        .pose_priors
        .iter()
        .filter_map(|prior| {
            let image = pose_prior_image_index(reconstruction, &prior.corr_data_id)?;
            if reconstruction
                .poses
                .get(image)
                .is_none_or(|pose| pose.is_none())
            {
                return None;
            }
            if constant_set.contains(&image) {
                return None;
            }
            if variable_set
                .as_ref()
                .is_some_and(|images| !images.contains(&image))
            {
                return None;
            }
            Some(
                crate::ba::BundleAdjustmentPosePrior::new(image, prior.position)
                    .with_covariance(prior.position_covariance),
            )
        })
        .collect::<Vec<_>>();
    priors.sort_by_key(|prior| prior.image);
    priors.dedup_by_key(|prior| prior.image);
    priors
}

pub(super) fn pose_prior_image_index(
    reconstruction: &Reconstruction,
    data_id: &ColmapDataId,
) -> Option<usize> {
    if data_id.sensor_id.sensor_type != ColmapSensorType::Camera {
        return None;
    }
    for image in 0..reconstruction.poses.len() {
        if reconstruction.image_id(image) as u64 != data_id.data_id {
            continue;
        }
        if let Some(frame_idx) = reconstruction.frame_index_for_image(image) {
            let frame = reconstruction.frames.get(frame_idx)?;
            if frame.data_ids.iter().any(|frame_data_id| {
                frame_data_id.data_id == data_id.data_id
                    && frame_data_id.sensor_id == sensor_id_from_colmap(&data_id.sensor_id)
            }) {
                return Some(image);
            }
            continue;
        }
        if reconstruction.camera_id_for_image(image) == data_id.sensor_id.sensor_id {
            return Some(image);
        }
    }
    None
}

pub(super) fn expanded_constant_images(
    reconstruction: &Reconstruction,
    constant_images: Vec<usize>,
) -> Vec<usize> {
    expand_images_to_registration_frames(reconstruction, &constant_images)
}

pub(super) fn apply_colmap_global_ba_solver_options(
    options: &mut crate::ba::BundleAdjustmentOptions,
) {
    options.gradient_tolerance = 1.0;
    options.parameter_tolerance = 0.0;
    options.max_linear_solver_iterations = 100;
}

pub(super) fn mapper_global_ba_options(
    config: &MapperConfig,
    reconstruction: &Reconstruction,
    iterations: usize,
    variable_images: Option<Vec<usize>>,
    constant_images: Vec<usize>,
    point_ids: Option<Vec<usize>>,
    constant_point_ids: Option<Vec<usize>>,
) -> crate::ba::BundleAdjustmentOptions {
    let mut options = mapper_ba_options(
        config,
        reconstruction,
        iterations,
        variable_images,
        constant_images,
        point_ids,
        constant_point_ids,
    );
    apply_colmap_small_reconstruction_global_ba_solver_options(reconstruction, &mut options);
    options
}

pub(super) fn apply_colmap_small_reconstruction_global_ba_solver_options(
    reconstruction: &Reconstruction,
    options: &mut crate::ba::BundleAdjustmentOptions,
) {
    const MIN_NUM_REG_FRAMES_FOR_FAST_BA: usize = 10;
    if registered_frame_count(reconstruction) >= MIN_NUM_REG_FRAMES_FOR_FAST_BA {
        return;
    }
    options.function_tolerance /= 10.0;
    options.gradient_tolerance /= 10.0;
    options.parameter_tolerance /= 10.0;
    options.iterations = options.iterations.saturating_mul(2);
    options.max_linear_solver_iterations = 200;
}

pub(super) fn apply_colmap_local_ba_solver_options(
    options: &mut crate::ba::BundleAdjustmentOptions,
) {
    options.gradient_tolerance = 10.0;
    options.parameter_tolerance = 0.0;
    options.max_linear_solver_iterations = 100;
}

pub(super) fn mapper_local_ba_options(
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
    let mut constant_images =
        expand_images_to_registration_frames(reconstruction, &constant_images);
    if config.fix_existing_frames {
        constant_images.extend(registration_stats.existing_registered_images(reconstruction));
    }
    constant_images = expand_images_to_registration_frames(reconstruction, &constant_images);
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
    options.loss_function = mapper_local_ba_loss_function();
    apply_colmap_local_ba_solver_options(&mut options);
    options
}

#[derive(Debug, Clone)]
pub(super) enum BundleAdjustmentSkipReason {
    PreBogusCameras(Vec<usize>),
    SolverReturnedNone,
    UnusableSolution(crate::ba::BundleAdjustmentReport),
    PostBogusCameras(Vec<usize>),
}

impl fmt::Display for BundleAdjustmentSkipReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PreBogusCameras(indices) => {
                write!(f, "pre_bogus_cameras={indices:?}")
            }
            Self::SolverReturnedNone => f.write_str("solver_returned_none"),
            Self::UnusableSolution(report) => {
                write!(f, "unusable_solution {}", report.brief_report())
            }
            Self::PostBogusCameras(indices) => {
                write!(f, "post_bogus_cameras={indices:?}")
            }
        }
    }
}

pub(super) fn refine_bundle_adjustment_checked(
    frames: &[ImageFrame],
    reconstruction: &mut Reconstruction,
    config: &MapperConfig,
    options: crate::ba::BundleAdjustmentOptions,
) -> std::result::Result<crate::ba::BundleAdjustmentReport, BundleAdjustmentSkipReason> {
    let pre_bogus = bogus_registered_camera_indices(reconstruction, config);
    if !pre_bogus.is_empty() {
        return Err(BundleAdjustmentSkipReason::PreBogusCameras(pre_bogus));
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
    let Some(report) = report else {
        restore_ba_state(
            reconstruction,
            base_camera,
            &base_cameras,
            &base_poses,
            &base_points,
        );
        return Err(BundleAdjustmentSkipReason::SolverReturnedNone);
    };
    if !report.is_solution_usable() {
        restore_ba_state(
            reconstruction,
            base_camera,
            &base_cameras,
            &base_poses,
            &base_points,
        );
        return Err(BundleAdjustmentSkipReason::UnusableSolution(report));
    }
    let post_bogus = bogus_registered_camera_indices(reconstruction, config);
    if !post_bogus.is_empty() {
        restore_ba_state(
            reconstruction,
            base_camera,
            &base_cameras,
            &base_poses,
            &base_points,
        );
        return Err(BundleAdjustmentSkipReason::PostBogusCameras(post_bogus));
    }
    sync_registered_frame_poses_from_images(reconstruction);
    Ok(report)
}

pub(super) fn sync_registered_frame_poses_from_images(reconstruction: &mut Reconstruction) {
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

pub(super) fn frame_consistent_poses_from_registered_images(
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

pub(super) fn restore_ba_state(
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

pub(super) fn ba_constant_camera_indices(
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

pub(super) fn ba_constant_rig_ids(
    config: &MapperConfig,
    reconstruction: &Reconstruction,
) -> Vec<u32> {
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

pub(super) fn ba_constant_sensor_from_rig_ids(
    config: &MapperConfig,
    reconstruction: &Reconstruction,
) -> Vec<SensorId> {
    constant_sensor_from_rig_ids_for_rigs(
        reconstruction,
        ba_constant_rig_ids(config, reconstruction).into_iter(),
    )
}

pub(super) fn local_ba_constant_sensor_from_rig_ids(
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

pub(super) fn local_ba_partial_rig_ids(
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

pub(super) fn constant_sensor_from_rig_ids_for_rigs(
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

pub(super) fn local_ba_constant_camera_indices(
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

pub(super) fn expand_images_to_registration_frames(
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
