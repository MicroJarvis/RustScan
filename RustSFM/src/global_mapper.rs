//! GLOMAP-style global mapper orchestration.
//!
//! This module provides the global SfM pipeline stages:
//!
//! 1. [`crate::view_graph_calibration`] — filter/refine the two-view graph
//!    before rotation averaging (GLOMAP §3.1).
//! 2. [`crate::view_graph_splitting`] — split the covisibility graph into
//!    connected components and reconstruct each independently.
//! 3. [`crate::rotation_averaging`] — global rotation averaging from the
//!    relative rotations of the view graph.
//! 2. [`crate::global_positioning`] — global positioning (translation
//!    averaging) from the relative translation directions.
//! 3. [`crate::track_establishment`] — fuse pairwise inlier matches into
//!    multi-view feature tracks.
//! 4. [`crate::track_triangulation`] — triangulate tracks into 3D points.
//! 5. Optional global bundle adjustment via [`crate::ba::refine_bundle_adjustment`].
//!
//! [`run_global_mapper`] recovers per-view world→camera poses only.
//! [`run_global_reconstruction`] runs the full pipeline and returns a populated
//! [`crate::types::Reconstruction`].

use crate::ba::{refine_bundle_adjustment, BundleAdjustmentOptions};
use crate::global_positioning::{
    estimate_global_positions, relative_translations_from_pairs, GlobalPositioningOptions,
};
use crate::incremental_triangulator::{
    IncrementalTriangulator, IncrementalTriangulatorOptions, IncrementalTriangulatorState,
};
use crate::joint_global_positioning::{
    estimate_joint_global_positions, JointGlobalPositioningOptions,
};
use crate::rotation_averaging::{
    estimate_global_rotations, relative_rotations_from_pairs, RotationAveragingOptions,
};
use crate::track_establishment::{
    establish_tracks, Track, TrackEstablishmentOptions, TrackEstablishmentStats,
};
use crate::track_triangulation::{
    triangulate_tracks, TrackTriangulationOptions, TrackTriangulationStats,
};
use crate::types::{CameraModel, ImageFrame, PairGeometry, Point3D, Reconstruction, TrackObservation};
use crate::view_graph_calibration::{
    calibrate_view_graph, ViewGraphCalibrationOptions, ViewGraphCalibrationStats,
};
use crate::view_graph_splitting::{
    components_for_reconstruction, remap_pairs_for_component, subset_frames_for_component,
    ViewGraphComponent, ViewGraphComponentSplittingOptions, ViewGraphComponentSplittingStats,
};
use glam::Vec3;
use rustslam::SE3;

/// Options for [`run_global_mapper`].
#[derive(Debug, Clone, Copy, Default)]
pub struct GlobalMapperOptions {
    /// Rotation-averaging stage options.
    pub rotation_averaging: RotationAveragingOptions,
    /// Global-positioning stage options.
    pub global_positioning: GlobalPositioningOptions,
}

/// Options for multi-pass global bundle adjustment after triangulation.
#[derive(Debug, Clone, Copy)]
pub struct GlobalRefinementOptions {
    /// Maximum global-BA / filter rounds (COLMAP `ba_global_max_refinements`).
    pub max_refinements: usize,
    /// Stop when the relative observation change drops below this threshold.
    pub max_refinement_change: f32,
    /// Remove observations whose reprojection error exceeds this threshold.
    pub filter_max_reprojection_error_px: f32,
    /// Drop tracks shorter than this after filtering (COLMAP default: 2).
    pub filter_min_track_length: usize,
    /// Drop tracks whose best triangulation angle is below this (degrees).
    pub filter_min_triangulation_angle_deg: f32,
    /// Reprojection gate for `complete` / `merge` between BA rounds (can be looser
    /// than [`Self::filter_max_reprojection_error_px`]).
    pub complete_max_reprojection_error_px: f32,
}

impl Default for GlobalRefinementOptions {
    fn default() -> Self {
        Self {
            max_refinements: 5,
            max_refinement_change: 0.0005,
            filter_max_reprojection_error_px: 4.0,
            filter_min_track_length: 2,
            filter_min_triangulation_angle_deg: TrackTriangulationOptions::default()
                .min_triangulation_angle_deg,
            complete_max_reprojection_error_px: 4.0,
        }
    }
}

/// Options for the full global reconstruction pipeline.
#[derive(Debug, Clone)]
pub struct GlobalReconstructionOptions {
    /// View-graph calibration before rotation averaging and track establishment.
    pub view_graph_calibration: ViewGraphCalibrationOptions,
    /// Pose-estimation options (rotation averaging + global positioning).
    pub mapper: GlobalMapperOptions,
    /// Track-establishment options.
    pub tracks: TrackEstablishmentOptions,
    /// Track-triangulation options (legacy path and joint-point filtering).
    pub triangulation: TrackTriangulationOptions,
    /// Use GLOMAP-style joint camera+point positioning instead of translation
    /// averaging followed by independent DLT triangulation.
    pub use_joint_positioning: bool,
    /// Joint-positioning options when [`Self::use_joint_positioning`] is true.
    pub joint_positioning: JointGlobalPositioningOptions,
    /// Multi-pass global BA / outlier filtering after structure initialization.
    pub refinement: GlobalRefinementOptions,
    /// Incremental triangulator options for complete/merge/retriangulate passes
    /// between global-BA rounds.
    pub incremental_triangulation: IncrementalTriangulatorOptions,
    /// Run global bundle adjustment after triangulation.
    pub run_global_ba: bool,
    /// Global BA iteration count per refinement round.
    pub global_ba_iterations: usize,
    /// Split the view graph into connected components and reconstruct each
    /// qualifying component as an independent model.
    pub component_splitting: ViewGraphComponentSplittingOptions,
}

impl Default for GlobalReconstructionOptions {
    fn default() -> Self {
        Self {
            view_graph_calibration: ViewGraphCalibrationOptions::default(),
            mapper: GlobalMapperOptions::default(),
            tracks: TrackEstablishmentOptions::default(),
            triangulation: TrackTriangulationOptions::default(),
            use_joint_positioning: true,
            joint_positioning: JointGlobalPositioningOptions::default(),
            refinement: GlobalRefinementOptions::default(),
            incremental_triangulation: IncrementalTriangulatorOptions {
                ignore_two_view_tracks: true,
                ..IncrementalTriangulatorOptions::default()
            },
            run_global_ba: true,
            global_ba_iterations: 50,
            component_splitting: ViewGraphComponentSplittingOptions::default(),
        }
    }
}

/// Result of the global mapper pose stage.
#[derive(Debug, Clone)]
pub struct GlobalMapperResult {
    /// Per-view world→camera poses. `None` for views that are not connected to
    /// view `0` through the view graph (and therefore unconstrained).
    pub poses: Vec<Option<SE3>>,
    /// Number of views that received a pose.
    pub num_registered: usize,
    /// Rotation-averaging iterations performed.
    pub rotation_iterations: usize,
    /// Global-positioning iterations performed.
    pub position_iterations: usize,
    /// Mean per-edge rotation residual (degrees).
    pub mean_rotation_residual_deg: f64,
    /// Mean per-edge position residual (normalized units).
    pub mean_position_residual: f64,
}

/// Run the global mapper over a view graph of two-view geometries.
///
/// `num_views` is the number of cameras (view indices in `pairs` must be in
/// `0..num_views`). Returns `None` if the view graph is too small or yields no
/// usable rotation/position constraints.
pub fn run_global_mapper(
    num_views: usize,
    pairs: &[PairGeometry],
    options: &GlobalMapperOptions,
) -> Option<GlobalMapperResult> {
    if num_views < 2 {
        return None;
    }

    let rotation_edges = relative_rotations_from_pairs(pairs);
    let rotation = estimate_global_rotations(num_views, &rotation_edges, &options.rotation_averaging)?;

    let translation_edges = relative_translations_from_pairs(pairs);
    let position = estimate_global_positions(
        &rotation.global_rotations,
        &translation_edges,
        &options.global_positioning,
    )?;

    let mut poses = vec![None; num_views];
    let mut num_registered = 0usize;
    for view in 0..num_views {
        let connected = *rotation.connected.get(view).unwrap_or(&false)
            && *position.connected.get(view).unwrap_or(&false);
        if !connected {
            continue;
        }
        let rotation_i = rotation.global_rotations[view];
        let center_i = position.centers[view];
        // world→cam translation: t = -R c.
        let translation = -(rotation_i * center_i);
        poses[view] = Some(SE3::from_quat_translation(rotation_i, translation));
        num_registered += 1;
    }

    Some(GlobalMapperResult {
        poses,
        num_registered,
        rotation_iterations: rotation.num_iterations,
        position_iterations: position.num_iterations,
        mean_rotation_residual_deg: rotation.mean_residual_deg,
        mean_position_residual: position.mean_residual,
    })
}

/// Summary of incremental complete/merge/retriangulate passes during global
/// structure refinement.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GlobalStructureRefinementStats {
    /// New 3D points created by per-image `triangulate_image` / `complete_image`.
    pub created_points: usize,
    /// Observations added by per-image triangulation passes.
    pub image_completed_observations: usize,
    /// Observations added by `complete_all_tracks`.
    pub completed_observations: usize,
    /// Tracks merged by `merge_all_tracks`.
    pub merged_tracks: usize,
    /// New/changed points from `retriangulate`.
    pub retriangulated_points: usize,
    /// Observations removed by reprojection filtering.
    pub filtered_observations: usize,
}

/// Result of the full global reconstruction pipeline.
#[derive(Debug, Clone)]
pub struct GlobalReconstructionResult {
    /// Sparse reconstruction with registered poses and triangulated points.
    pub reconstruction: Reconstruction,
    /// Pose-estimation stage summary.
    pub mapper: GlobalMapperResult,
    /// Track-establishment statistics.
    pub track_stats: TrackEstablishmentStats,
    /// Track-triangulation statistics.
    pub triangulation_stats: TrackTriangulationStats,
    /// Whether global bundle adjustment ran successfully on the last round.
    pub global_ba_success: bool,
    /// Number of global-BA refinement rounds executed.
    pub refinement_rounds: usize,
    /// Incremental triangulation + filtering statistics across all BA rounds.
    pub structure_refinement: GlobalStructureRefinementStats,
    /// Whether joint camera+point positioning was used.
    pub used_joint_positioning: bool,
    /// View-graph calibration statistics.
    pub view_graph_calibration: ViewGraphCalibrationStats,
    /// Original view indices represented by this reconstruction model.
    pub component_views: Vec<usize>,
    /// Index of this model among selected connected components.
    pub component_index: usize,
}

/// Result of reconstructing one or more view-graph connected components.
#[derive(Debug, Clone)]
pub struct GlobalReconstructionsResult {
    /// One entry per successfully reconstructed component (largest first).
    pub reconstructions: Vec<GlobalReconstructionResult>,
    /// Connected-component splitting statistics.
    pub component_splitting: ViewGraphComponentSplittingStats,
}

/// Build pairwise inlier matches from verified two-view geometries.
pub fn pairwise_matches_from_pairs(pairs: &[PairGeometry]) -> Vec<crate::track_establishment::PairwiseMatches> {
    pairs
        .iter()
        .filter(|pair| !pair.pose_graph_only && !pair.inlier_matches.is_empty())
        .map(|pair| crate::track_establishment::PairwiseMatches {
            image_i: pair.left,
            image_j: pair.right,
            matches: pair
                .inlier_matches
                .iter()
                .map(|m| (m.query_idx as usize, m.train_idx as usize))
                .collect(),
        })
        .collect()
}

/// Run the full global reconstruction pipeline on every selected connected
/// component of the view graph.
pub fn run_global_reconstructions(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    camera: CameraModel,
    options: &GlobalReconstructionOptions,
) -> Option<GlobalReconstructionsResult> {
    if frames.len() < 2 {
        return None;
    }

    let (pairs, camera, view_graph_calibration) = {
        let (calibrated_pairs, calibrated_camera, stats) =
            calibrate_view_graph(frames, pairs, camera, &options.view_graph_calibration);
        if calibrated_pairs.is_empty() {
            return None;
        }
        (calibrated_pairs, calibrated_camera, stats)
    };

    let (components, mut component_splitting) =
        components_for_reconstruction(frames.len(), &pairs, &options.component_splitting);
    if components.is_empty() {
        return None;
    }

    let mut reconstructions = Vec::new();
    for (component_index, component) in components.iter().enumerate() {
        if let Some(result) = run_global_reconstruction_component(
            frames,
            &pairs,
            camera,
            options,
            component,
            component_index,
            view_graph_calibration,
        ) {
            reconstructions.push(result);
        }
    }

    if reconstructions.is_empty() {
        return None;
    }

    component_splitting.num_reconstructed = reconstructions.len();
    Some(GlobalReconstructionsResult {
        reconstructions,
        component_splitting,
    })
}

/// Run the full global reconstruction pipeline: pose estimation, track
/// establishment, structure initialization (joint positioning or legacy DLT
/// triangulation), and optional multi-pass global bundle adjustment.
///
/// When component splitting is enabled and multiple components qualify, this
/// returns the largest successfully reconstructed component.
pub fn run_global_reconstruction(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    camera: CameraModel,
    options: &GlobalReconstructionOptions,
) -> Option<GlobalReconstructionResult> {
    run_global_reconstructions(frames, pairs, camera, options)
        .and_then(|result| result.reconstructions.into_iter().next())
}

fn run_global_reconstruction_component(
    all_frames: &[ImageFrame],
    all_pairs: &[PairGeometry],
    camera: CameraModel,
    options: &GlobalReconstructionOptions,
    component: &ViewGraphComponent,
    component_index: usize,
    view_graph_calibration: ViewGraphCalibrationStats,
) -> Option<GlobalReconstructionResult> {
    let frames = subset_frames_for_component(all_frames, component);
    let pairs = remap_pairs_for_component(all_pairs, component);
    if frames.len() < 2 || pairs.is_empty() {
        return None;
    }

    let rotation_edges = relative_rotations_from_pairs(&pairs);
    let rotation = estimate_global_rotations(
        frames.len(),
        &rotation_edges,
        &options.mapper.rotation_averaging,
    )?;

    let match_pairs = pairwise_matches_from_pairs(&pairs);
    let (tracks, track_stats) = establish_tracks(&match_pairs, &options.tracks);
    if tracks.is_empty() {
        return None;
    }

    let used_joint_positioning = options.use_joint_positioning;
    let (mapper, mut triangulation_stats) = if used_joint_positioning {
        let joint = estimate_joint_global_positions(
            &rotation.global_rotations,
            &tracks,
            &frames,
            camera,
            &pairs,
            &options.joint_positioning,
        )?;
        let mapper = mapper_result_from_joint(&rotation, &joint);
        if mapper.num_registered < 2 {
            return None;
        }
        let mut reconstruction =
            build_reconstruction_scaffold(&frames, camera, &mapper.poses);
        // Seed long tracks with DLT; len-2 points are created by per-image triangulation.
        let triangulation_stats = triangulate_bulk_tracks(
            &tracks,
            &frames,
            &mut reconstruction,
            &options.triangulation,
        );
        let (refinement_rounds, structure_refinement, global_ba_success) =
            run_iterative_global_refinement(&frames, &pairs, &mut reconstruction, options);
        return Some(GlobalReconstructionResult {
            reconstruction,
            mapper,
            track_stats,
            triangulation_stats,
            global_ba_success,
            refinement_rounds,
            structure_refinement,
            used_joint_positioning: true,
            view_graph_calibration,
            component_views: component.views.clone(),
            component_index,
        });
    } else {
        let mapper = run_global_mapper(frames.len(), &pairs, &options.mapper)?;
        if mapper.num_registered < 2 {
            return None;
        }
        (mapper, TrackTriangulationStats::default())
    };

    let mut reconstruction = build_reconstruction_scaffold(&frames, camera, &mapper.poses);
    triangulation_stats =
        triangulate_bulk_tracks(&tracks, &frames, &mut reconstruction, &options.triangulation);

    let (refinement_rounds, structure_refinement, global_ba_success) =
        run_iterative_global_refinement(&frames, &pairs, &mut reconstruction, options);

    Some(GlobalReconstructionResult {
        reconstruction,
        mapper,
        track_stats,
        triangulation_stats,
        global_ba_success,
        refinement_rounds,
        structure_refinement,
        used_joint_positioning,
        view_graph_calibration,
        component_views: component.views.clone(),
        component_index,
    })
}

/// Minimum track length for bulk DLT triangulation before per-image refinement.
/// COLMAP-style incremental triangulation creates len-2 points only via
/// `complete_image`; bulk triangulation is restricted to longer tracks.
fn global_mapper_bulk_triangulation_min_track_length() -> usize {
    std::env::var("RUSTSFM_GLOBAL_BULK_TRIANGULATION_MIN_TRACK_LENGTH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 2)
        .unwrap_or(3)
}

fn triangulate_bulk_tracks(
    tracks: &[Track],
    frames: &[ImageFrame],
    reconstruction: &mut Reconstruction,
    options: &TrackTriangulationOptions,
) -> TrackTriangulationStats {
    let min_track_length = global_mapper_bulk_triangulation_min_track_length();
    let multi_view_tracks = tracks
        .iter()
        .filter(|track| track.len() >= min_track_length)
        .cloned()
        .collect::<Vec<_>>();
    triangulate_tracks(&multi_view_tracks, frames, reconstruction, options)
}

fn mapper_result_from_joint(
    rotation: &crate::rotation_averaging::RotationAveragingResult,
    joint: &crate::joint_global_positioning::JointGlobalPositioningResult,
) -> GlobalMapperResult {
    let num_views = rotation.global_rotations.len();
    let mut poses = vec![None; num_views];
    let mut num_registered = 0usize;
    for view in 0..num_views {
        let connected = rotation.connected.get(view).copied().unwrap_or(false)
            && joint.connected.get(view).copied().unwrap_or(false);
        if !connected {
            continue;
        }
        let rotation_i = rotation.global_rotations[view];
        let center_i = joint.centers[view];
        let translation = -(rotation_i * center_i);
        poses[view] = Some(SE3::from_quat_translation(rotation_i, translation));
        num_registered += 1;
    }
    GlobalMapperResult {
        poses,
        num_registered,
        rotation_iterations: rotation.num_iterations,
        position_iterations: joint.num_iterations,
        mean_rotation_residual_deg: rotation.mean_residual_deg,
        mean_position_residual: joint.mean_residual,
    }
}

fn incremental_triangulation_options(
    options: &GlobalReconstructionOptions,
) -> IncrementalTriangulatorOptions {
    IncrementalTriangulatorOptions {
        min_angle_deg: options.triangulation.min_triangulation_angle_deg,
        merge_max_reproj_error_px: options.refinement.complete_max_reprojection_error_px,
        complete_max_reproj_error_px: options.refinement.complete_max_reprojection_error_px,
        ..options.incremental_triangulation
    }
}

fn run_per_image_triangulation_pass(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    tri_options: &IncrementalTriangulatorOptions,
    tri_state: &mut IncrementalTriangulatorState,
    registered: &[usize],
) -> (usize, usize) {
    let mut triangulator =
        IncrementalTriangulator::new(frames, pairs, reconstruction, tri_state);
    let mut created_points = 0usize;
    let mut completed_observations = 0usize;
    for &image in registered {
        let mut image_report = triangulator.triangulate_image(tri_options, image);
        let complete_report = triangulator.complete_image(tri_options, image);
        image_report.completed_observations += complete_report.completed_observations;
        image_report.created_points += complete_report.created_points;
        created_points += image_report.created_points;
        completed_observations += image_report.total_observations();
    }
    (created_points, completed_observations)
}

fn run_incremental_triangulation_pass(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    tri_options: &IncrementalTriangulatorOptions,
    tri_state: &mut IncrementalTriangulatorState,
    registered: &[usize],
    retriangulate: bool,
) -> GlobalStructureRefinementStats {
    let (created_points, image_completed_observations) = run_per_image_triangulation_pass(
        frames,
        pairs,
        reconstruction,
        tri_options,
        tri_state,
        registered,
    );
    let mut triangulator =
        IncrementalTriangulator::new(frames, pairs, reconstruction, tri_state);
    let completed_observations = triangulator.complete_all_tracks(tri_options);
    let merged_tracks = triangulator.merge_all_tracks(tri_options);
    let retriangulated_points = if retriangulate {
        triangulator.retriangulate(tri_options)
    } else {
        0
    };
    GlobalStructureRefinementStats {
        created_points,
        image_completed_observations,
        completed_observations,
        merged_tracks,
        retriangulated_points,
        filtered_observations: 0,
    }
}

fn accumulate_structure_refinement(
    total: &mut GlobalStructureRefinementStats,
    pass: GlobalStructureRefinementStats,
) {
    total.created_points += pass.created_points;
    total.image_completed_observations += pass.image_completed_observations;
    total.completed_observations += pass.completed_observations;
    total.merged_tracks += pass.merged_tracks;
    total.retriangulated_points += pass.retriangulated_points;
    total.filtered_observations += pass.filtered_observations;
}

fn run_iterative_global_refinement(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    options: &GlobalReconstructionOptions,
) -> (usize, GlobalStructureRefinementStats, bool) {
    let mut stats = GlobalStructureRefinementStats::default();

    let registered: Vec<usize> = reconstruction
        .poses
        .iter()
        .enumerate()
        .filter_map(|(idx, pose)| pose.is_some().then_some(idx))
        .collect();
    if registered.len() < 2 {
        return (0, stats, false);
    }

    let tri_options = incremental_triangulation_options(options);
    let mut tri_state = IncrementalTriangulatorState::new(frames, pairs, reconstruction);

    accumulate_structure_refinement(
        &mut stats,
        run_incremental_triangulation_pass(
            frames,
            pairs,
            reconstruction,
            &tri_options,
            &mut tri_state,
            &registered,
            true,
        ),
    );

    if !options.run_global_ba || options.refinement.max_refinements == 0 {
        return (0, stats, false);
    }

    if reconstruction.points.is_empty() {
        return (0, stats, false);
    }

    let mut global_ba_success = false;
    let mut rounds = 0usize;
    for round in 0..options.refinement.max_refinements {
        let observations_before = reconstruction_num_observations(reconstruction);
        if observations_before == 0 {
            break;
        }
        rounds = round + 1;
        let ba_options = BundleAdjustmentOptions {
            iterations: options.global_ba_iterations,
            constant_images: vec![registered[0]],
            variable_images: Some(registered.clone()),
            allow_single_observation_points: false,
            ..BundleAdjustmentOptions::default()
        };
        global_ba_success =
            refine_bundle_adjustment(frames, reconstruction, ba_options).is_some();

        let pass_stats = run_incremental_triangulation_pass(
            frames,
            pairs,
            reconstruction,
            &tri_options,
            &mut tri_state,
            &registered,
            true,
        );
        accumulate_structure_refinement(&mut stats, pass_stats);

        let filtered = filter_reconstruction_tracks_with_state(
            frames,
            pairs,
            reconstruction,
            &mut tri_state,
            &options.refinement,
        );
        stats.filtered_observations += filtered;
        tri_state.sync_after_reconstruction_rollback(frames, pairs, reconstruction);

        let observations_after = reconstruction_num_observations(reconstruction);
        if observations_after == 0 {
            break;
        }
        let changed = (observations_before.saturating_sub(observations_after)
            + pass_stats.created_points
            + pass_stats.image_completed_observations
            + pass_stats.completed_observations
            + pass_stats.merged_tracks
            + pass_stats.retriangulated_points
            + filtered) as f32
            / observations_before.max(1) as f32;
        if changed <= options.refinement.max_refinement_change {
            break;
        }
    }

    if !reconstruction.points.is_empty() {
        stats.filtered_observations += filter_reconstruction_tracks_with_state(
            frames,
            pairs,
            reconstruction,
            &mut tri_state,
            &options.refinement,
        );
        tri_state.sync_after_reconstruction_rollback(frames, pairs, reconstruction);
    }

    (rounds, stats, global_ba_success)
}

fn reconstruction_num_observations(reconstruction: &Reconstruction) -> usize {
    reconstruction
        .observations
        .iter()
        .flat_map(|image| image.iter())
        .filter(|obs| obs.is_some())
        .count()
}

fn filter_reconstruction_tracks_with_state(
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
    reconstruction: &mut Reconstruction,
    tri_state: &mut IncrementalTriangulatorState,
    options: &GlobalRefinementOptions,
) -> usize {
    let max_error_px = options.filter_max_reprojection_error_px;
    let min_track_length = options.filter_min_track_length;
    let min_tri_angle_deg = options.filter_min_triangulation_angle_deg;
    let image_cameras = (0..reconstruction.poses.len())
        .map(|image| reconstruction.camera_for_image(image))
        .collect::<Vec<_>>();
    let observation_manager = tri_state.observation_manager_mut();
    let mut removed = 0usize;
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
                let err = crate::geometry::reprojection_error_px(
                    point_xyz,
                    pose,
                    [kp.x(), kp.y()],
                    camera,
                );
                !err.is_finite() || err > max_error_px
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
        if !track_has_min_triangulation_angle_filter(
            reconstruction.points[point_id].xyz,
            &track,
            reconstruction,
            min_tri_angle_deg,
        ) {
            removed += track.len();
            observation_manager.delete_point3d(frames, pairs, reconstruction, point_id);
            continue;
        }
        point_id += 1;
    }
    removed
}

fn point_has_positive_depth(point: [f32; 3], pose: SE3) -> bool {
    pose.transform_point(&point)[2] > 1.0e-6
}

fn track_has_min_triangulation_angle_filter(
    point: [f32; 3],
    track: &[TrackObservation],
    reconstruction: &Reconstruction,
    min_angle_deg: f32,
) -> bool {
    if min_angle_deg <= 0.0 {
        return true;
    }
    let point3 = nalgebra::Vector3::new(point[0] as f64, point[1] as f64, point[2] as f64);
    let mut best = 0.0f64;
    for i in 0..track.len() {
        for j in i + 1..track.len() {
            let Some(pose_i) = reconstruction.poses[track[i].image] else {
                continue;
            };
            let Some(pose_j) = reconstruction.poses[track[j].image] else {
                continue;
            };
            let c1 = camera_center_vec3(pose_i);
            let c2 = camera_center_vec3(pose_j);
            let angle_rad =
                crate::triangulation::calculate_triangulation_angle(&c1, &c2, &point3);
            best = best.max(angle_rad);
        }
    }
    best.to_degrees() >= min_angle_deg as f64
}

fn camera_center_vec3(pose: SE3) -> nalgebra::Vector3<f64> {
    let c = crate::geometry::camera_center(pose);
    nalgebra::Vector3::new(c.x as f64, c.y as f64, c.z as f64)
}

fn build_reconstruction_scaffold(
    frames: &[ImageFrame],
    camera: CameraModel,
    poses: &[Option<SE3>],
) -> Reconstruction {
    Reconstruction {
        camera,
        cameras: vec![camera],
        camera_ids: vec![1],
        rigs: Vec::new(),
        frames: Vec::new(),
        image_names: frames.iter().map(|frame| frame.name.clone()).collect(),
        image_paths: frames.iter().map(|frame| frame.path.clone()).collect(),
        image_ids: (0..frames.len()).map(|idx| idx as u32 + 1).collect(),
        image_camera_indices: vec![0; frames.len()],
        image_frame_indices: vec![None; frames.len()],
        poses: poses.to_vec(),
        observations: frames
            .iter()
            .map(|frame| vec![None; frame.keypoints.len()])
            .collect(),
        keypoints: frames.iter().map(|frame| frame.keypoints.clone()).collect(),
        point_ids: Vec::new(),
        points: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{camera_center, pose_rotation};
    use glam::Quat;
    use rustslam::ColmapMt19937;
    use rustslam::Match;

    fn unit(rng: &mut ColmapMt19937) -> f32 {
        rng.next_u32() as f32 / u32::MAX as f32
    }

    fn random_quat(rng: &mut ColmapMt19937) -> Quat {
        let axis = Vec3::new(unit(rng) - 0.5, unit(rng) - 0.5, unit(rng) - 0.5)
            .normalize_or_zero();
        let axis = if axis.length_squared() < 1.0e-6 {
            Vec3::X
        } else {
            axis
        };
        Quat::from_axis_angle(axis, unit(rng) * std::f32::consts::PI)
    }

    fn random_center(rng: &mut ColmapMt19937) -> Vec3 {
        Vec3::new(
            unit(rng) * 2.0 - 1.0,
            unit(rng) * 2.0 - 1.0,
            unit(rng) * 2.0 - 1.0,
        )
    }

    /// Build a `PairGeometry` from ground-truth global rotations/centers.
    /// Relative pose maps cam_i -> cam_j: R_ij = R_j R_i^T, t_ij = -R_j(c_j-c_i).
    fn synth_pair(
        i: usize,
        j: usize,
        rotations: &[Quat],
        centers: &[Vec3],
        inliers: usize,
    ) -> PairGeometry {
        let r_ij = (rotations[j] * rotations[i].inverse()).normalize();
        let t_ij = -(rotations[j] * (centers[j] - centers[i]));
        PairGeometry {
            left: i,
            right: j,
            two_view_config: 2,
            f_matrix: None,
            e_matrix: None,
            h_matrix: None,
            qvec: None,
            tvec: None,
            matches: Vec::new(),
            inlier_matches: Vec::new(),
            relative_pose: SE3::from_quat_translation(r_ij, t_ij),
            inliers,
            triangulated: inliers,
            mean_reprojection_error_px: 0.5,
            rotation_deg: 0.0,
            median_triangulation_angle_deg: 5.0,
            pose_graph_only: false,
        }
    }

    fn angle_between_deg(a: Quat, b: Quat) -> f32 {
        let r = (a * b.inverse()).normalize();
        (2.0 * r.w.abs().clamp(-1.0, 1.0).acos()).to_degrees()
    }

    #[test]
    fn recovers_global_poses_from_synthetic_scene() {
        let mut rng = ColmapMt19937::new(2024);
        let n = 9;
        // Gauge: view 0 at identity rotation and origin to match the solver gauge.
        let mut rotations = vec![Quat::IDENTITY];
        let mut centers = vec![Vec3::ZERO];
        for _ in 1..n {
            rotations.push(random_quat(&mut rng));
            centers.push(random_center(&mut rng));
        }
        let mut pairs = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if j - i > 4 {
                    continue;
                }
                pairs.push(synth_pair(i, j, &rotations, &centers, 100));
            }
        }

        let result = run_global_mapper(n, &pairs, &GlobalMapperOptions::default()).unwrap();
        assert_eq!(result.num_registered, n);

        // Recover the global scale by comparing recovered centers to gt.
        let est_centers: Vec<Vec3> = result
            .poses
            .iter()
            .map(|p| camera_center(p.unwrap()))
            .collect();
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for (e, g) in est_centers.iter().zip(centers.iter()) {
            num += e.dot(*g);
            den += e.dot(*e);
        }
        let scale = if den < 1.0e-12 { 1.0 } else { num / den };

        for view in 0..n {
            let pose = result.poses[view].unwrap();
            let rot_err = angle_between_deg(pose_rotation(pose), rotations[view]);
            assert!(rot_err < 0.1, "view {view} rotation error {rot_err} deg");
            let center_err = (est_centers[view] * scale - centers[view]).length();
            assert!(center_err < 1.0e-2, "view {view} center error {center_err}");
        }
        assert!(result.mean_rotation_residual_deg < 0.1);
        assert!(result.mean_position_residual < 1.0e-2);
    }

    #[test]
    fn leaves_disconnected_views_unregistered() {
        let rotations = vec![Quat::IDENTITY; 4];
        let centers = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
        // Only views 0,1 are connected.
        let pairs = vec![synth_pair(0, 1, &rotations, &centers, 50)];
        let result = run_global_mapper(4, &pairs, &GlobalMapperOptions::default()).unwrap();
        assert!(result.poses[0].is_some());
        assert!(result.poses[1].is_some());
        assert!(result.poses[2].is_none());
        assert!(result.poses[3].is_none());
        assert_eq!(result.num_registered, 2);
    }

    #[test]
    fn rejects_trivial_view_graph() {
        assert!(run_global_mapper(1, &[], &GlobalMapperOptions::default()).is_none());
        assert!(run_global_mapper(3, &[], &GlobalMapperOptions::default()).is_none());
    }

    fn test_camera() -> crate::types::CameraModel {
        crate::types::CameraModel::new_pinhole(640, 480, 500.0, 500.0, 320.0, 240.0)
    }

    fn project_keypoint(
        camera: crate::types::CameraModel,
        pose: SE3,
        point: [f32; 3],
    ) -> rustslam::KeyPoint {
        let p = pose.transform_point(&point);
        let xy = camera
            .img_from_cam(p[0] as f64, p[1] as f64, p[2] as f64)
            .unwrap();
        rustslam::KeyPoint::new(xy[0] as f32, xy[1] as f32)
    }

    fn synth_frame(id: usize, keypoints: Vec<rustslam::KeyPoint>) -> ImageFrame {
        let colors = vec![[128, 128, 128]; keypoints.len()];
        ImageFrame {
            id,
            name: format!("img_{id:03}.jpg"),
            path: std::path::PathBuf::from(format!("img_{id:03}.jpg")),
            width: 640,
            height: 480,
            keypoints,
            descriptors: rustslam::Descriptors::new(),
            sift: crate::sift::SiftFeatures::default(),
            wide_descriptors: crate::wide::WideDescriptors {
                data: Vec::new(),
                dim: 0,
                count: 0,
            },
            strong_feature_indices: Vec::new(),
            colors,
        }
    }

    fn synth_pair_with_matches(
        i: usize,
        j: usize,
        rotations: &[Quat],
        centers: &[Vec3],
        feature_idx: usize,
        inliers: usize,
    ) -> PairGeometry {
        let mut pair = synth_pair(i, j, rotations, centers, inliers);
        pair.inlier_matches = vec![Match {
            query_idx: feature_idx as u32,
            train_idx: feature_idx as u32,
            distance: 0.0,
        }];
        pair
    }

    #[test]
    fn run_global_reconstruction_triangulates_shared_feature() {
        let camera = test_camera();
        let point = [0.0, 0.0, 5.0];
        let n = 6;
        let mut rotations = vec![Quat::IDENTITY];
        let mut centers = vec![Vec3::ZERO];
        for view in 1..n {
            rotations.push(Quat::IDENTITY);
            centers.push(Vec3::new(view as f32 * 0.25, 0.0, 0.0));
        }

        let mut pairs = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if j - i > 3 {
                    continue;
                }
                pairs.push(synth_pair_with_matches(
                    i, j, &rotations, &centers, 0, 100,
                ));
            }
        }

        let mapper = run_global_mapper(n, &pairs, &GlobalMapperOptions::default()).unwrap();
        assert_eq!(mapper.num_registered, n);

        // Project the shared 3D point with the *recovered* poses so observations
        // are consistent with the global mapper geometry.
        let frames: Vec<ImageFrame> = (0..n)
            .map(|view| {
                let pose = mapper.poses[view].unwrap();
                synth_frame(view, vec![project_keypoint(camera, pose, point)])
            })
            .collect();

        let mut options = GlobalReconstructionOptions::default();
        options.triangulation.min_triangulation_angle_deg = 0.5;
        options.triangulation.max_reprojection_error_px = 4.0;
        options.run_global_ba = false;

        let match_pairs = pairwise_matches_from_pairs(&pairs);
        let (tracks, track_stats) = establish_tracks(&match_pairs, &options.tracks);
        let mut reconstruction =
            build_reconstruction_scaffold(&frames, camera, &mapper.poses);
        let triangulation_stats =
            triangulate_tracks(&tracks, &frames, &mut reconstruction, &options.triangulation);

        assert!(track_stats.num_tracks >= 1);
        assert!(triangulation_stats.num_triangulated >= 1);
        assert!(!reconstruction.points.is_empty());
    }

    #[test]
    fn global_refinement_retriangulates_under_reconstructed_pairs() {
        let camera = test_camera();
        let mut frames = (0..2)
            .map(|view| synth_frame(view, vec![rustslam::KeyPoint::new(320.0, 240.0)]))
            .collect::<Vec<_>>();
        frames[1].keypoints = vec![
            rustslam::KeyPoint::new(330.0, 240.0),
            rustslam::KeyPoint::new(340.0, 240.0),
        ];
        frames[1].colors = vec![[128, 128, 128]; frames[1].keypoints.len()];

        let pairs = vec![PairGeometry {
            left: 0,
            right: 1,
            two_view_config: 2,
            f_matrix: None,
            e_matrix: None,
            h_matrix: None,
            qvec: None,
            tvec: None,
            matches: Vec::new(),
            inlier_matches: vec![
                rustslam::Match {
                    query_idx: 0,
                    train_idx: 0,
                    distance: 0.0,
                },
                rustslam::Match {
                    query_idx: 0,
                    train_idx: 1,
                    distance: 0.0,
                },
            ],
            relative_pose: SE3::from_quat_translation(
                Quat::IDENTITY,
                Vec3::new(1.0, 0.0, 0.0),
            ),
            inliers: 2,
            triangulated: 2,
            mean_reprojection_error_px: 0.5,
            rotation_deg: 0.0,
            median_triangulation_angle_deg: 5.0,
            pose_graph_only: false,
        }];

        let mut reconstruction = build_reconstruction_scaffold(
            &frames,
            camera,
            &[
                Some(SE3::identity()),
                Some(SE3::from_quat_translation(
                    Quat::IDENTITY,
                    Vec3::new(1.0, 0.0, 0.0),
                )),
            ],
        );

        let mut options = GlobalReconstructionOptions::default();
        options.triangulation.min_triangulation_angle_deg = 0.1;
        options.triangulation.max_reprojection_error_px = 10.0;
        options.refinement.max_refinements = 1;
        options.global_ba_iterations = 5;
        options.incremental_triangulation = IncrementalTriangulatorOptions {
            re_min_ratio: 0.5,
            min_angle_deg: 0.1,
            merge_max_reproj_error_px: 10.0,
            complete_max_reproj_error_px: 10.0,
            ignore_two_view_tracks: false,
            ..IncrementalTriangulatorOptions::default()
        };
        options.refinement.filter_max_reprojection_error_px = 10.0;
        options.refinement.filter_min_triangulation_angle_deg = 0.1;

        let (_rounds, stats, _ba_ok) = super::run_iterative_global_refinement(
            &frames,
            &pairs,
            &mut reconstruction,
            &options,
        );

        assert!(
            stats.created_points >= 1 || stats.retriangulated_points >= 1,
            "created_points={} retriangulated_points={}",
            stats.created_points,
            stats.retriangulated_points
        );
        assert!(!reconstruction.points.is_empty());
    }

    #[test]
    fn per_image_triangulation_creates_orphan_points_without_global_ba() {
        let camera = test_camera();
        let point = [0.0, 0.0, 5.0];
        let mut frames = (0..3)
            .map(|view| {
                let pose = SE3::from_quat_translation(
                    Quat::IDENTITY,
                    Vec3::new(view as f32, 0.0, 0.0),
                );
                synth_frame(view, vec![project_keypoint(camera, pose, point)])
            })
            .collect::<Vec<_>>();
        frames[1].keypoints[0] = rustslam::KeyPoint::new(330.0, 240.0);
        frames[2].keypoints[0] = rustslam::KeyPoint::new(340.0, 240.0);

        let pairs = vec![
            synth_pair_with_matches(0, 1, &[Quat::IDENTITY; 3], &[Vec3::ZERO, Vec3::X, Vec3::splat(2.0)], 0, 2),
            synth_pair_with_matches(1, 2, &[Quat::IDENTITY; 3], &[Vec3::ZERO, Vec3::X, Vec3::splat(2.0)], 0, 2),
        ];

        let mut reconstruction = build_reconstruction_scaffold(
            &frames,
            camera,
            &[
                Some(SE3::identity()),
                Some(SE3::from_quat_translation(Quat::IDENTITY, Vec3::X)),
                Some(SE3::from_quat_translation(Quat::IDENTITY, Vec3::splat(2.0))),
            ],
        );
        assert!(reconstruction.points.is_empty());

        let mut options = GlobalReconstructionOptions::default();
        options.triangulation.min_triangulation_angle_deg = 0.1;
        options.triangulation.max_reprojection_error_px = 10.0;
        options.run_global_ba = false;
        options.incremental_triangulation = IncrementalTriangulatorOptions {
            complete_max_reproj_error_px: 10.0,
            min_angle_deg: 0.1,
            ignore_two_view_tracks: false,
            ..IncrementalTriangulatorOptions::default()
        };

        let (_rounds, stats, ba_ok) = super::run_iterative_global_refinement(
            &frames,
            &pairs,
            &mut reconstruction,
            &options,
        );

        assert!(!ba_ok);
        assert!(stats.created_points >= 1);
        assert!(!reconstruction.points.is_empty());
    }

    #[test]
    fn run_global_reconstructions_splits_disconnected_components() {
        let camera = test_camera();
        let point = [0.0, 0.0, 5.0];
        let n = 6;
        let mut rotations = vec![Quat::IDENTITY; n];
        let mut centers = vec![Vec3::ZERO; n];
        for view in 0..n {
            centers[view] = Vec3::new((view % 3) as f32 * 0.25, (view / 3) as f32, 0.0);
        }

        let mut pairs = Vec::new();
        for i in 0..3 {
            for j in (i + 1)..3 {
                pairs.push(synth_pair_with_matches(
                    i, j, &rotations, &centers, 0, 100,
                ));
            }
        }
        for i in 3..6 {
            for j in (i + 1)..6 {
                pairs.push(synth_pair_with_matches(
                    i, j, &rotations, &centers, 0, 100,
                ));
            }
        }

        let mapper = run_global_mapper(n, &pairs, &GlobalMapperOptions::default()).unwrap();
        assert_eq!(mapper.num_registered, 3);
        let frames: Vec<ImageFrame> = (0..n)
            .map(|view| {
                let rotation_i = rotations[view];
                let center_i = centers[view];
                let translation = -(rotation_i * center_i);
                let pose = SE3::from_quat_translation(rotation_i, translation);
                synth_frame(view, vec![project_keypoint(camera, pose, point)])
            })
            .collect();

        let mut options = GlobalReconstructionOptions::default();
        options.view_graph_calibration.enabled = false;
        options.component_splitting.enabled = true;
        options.component_splitting.min_component_size = 3;
        options.component_splitting.max_components = 0;
        options.triangulation.min_triangulation_angle_deg = 0.5;
        options.triangulation.max_reprojection_error_px = 4.0;
        options.run_global_ba = false;

        let result = run_global_reconstructions(&frames, &pairs, camera, &options).unwrap();
        assert_eq!(result.component_splitting.num_components, 2);
        assert_eq!(result.component_splitting.num_reconstructed, 2);
        assert_eq!(result.reconstructions.len(), 2);
        assert_eq!(result.reconstructions[0].component_views, vec![0, 1, 2]);
        assert_eq!(result.reconstructions[1].component_views, vec![3, 4, 5]);
        assert_eq!(result.reconstructions[0].mapper.num_registered, 3);
        assert_eq!(result.reconstructions[1].mapper.num_registered, 3);
        assert_eq!(result.reconstructions[0].reconstruction.poses.len(), 3);
        assert_eq!(result.reconstructions[1].reconstruction.poses.len(), 3);
    }
}
