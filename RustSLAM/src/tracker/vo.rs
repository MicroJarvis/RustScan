//! Visual Odometry

use crate::config::{FeatureType, TrackerParams};
use crate::core::{Camera, Frame, FrameFeatures, KeyFrame, Map, MapPoint, SE3};
use crate::features::{
    Descriptors, FastExtractor, FastParams, FeatureExtractor, FeatureMatcher, HammingMatcher,
    HarrisExtractor, HarrisParams, KeyPoint, OrbExtractor,
};
use crate::loop_closing::Relocalizer;
use crate::tracker::solver::{EssentialSolver, PnPSolver, Triangulator};
use glam::Vec3;
use serde::Serialize;
use std::collections::VecDeque;

const MAX_RELOCALIZATION_KEYFRAMES: usize = 24;
const RELOCALIZATION_ANCHOR_INTERVAL_FRAMES: u64 = 15;

fn relocalization_min_inliers_from_tracking(min_inliers: usize) -> usize {
    min_inliers.saturating_sub(2).max(6)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameEstimateKind {
    InitializedMonocular,
    TrackedPnP,
    RelocalizedPnP,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct RelocalizationStats {
    pub lost_events: u64,
    pub direct_retrack_attempts: u64,
    pub direct_retrack_successes: u64,
    pub anchor_store_successes: u64,
    pub anchor_relocalization_calls: u64,
    pub anchor_candidates_tested: u64,
    pub anchor_relocalization_successes: u64,
    pub monocular_reinit_attempts: u64,
    pub monocular_reinit_successes: u64,
    pub cached_anchor_keyframes: usize,
}

/// Visual Odometry state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VOState {
    /// Not initialized
    NotInitialized,
    /// Initializing (need more frames)
    Initializing,
    /// Tracking OK
    TrackingOk,
    /// Lost tracking
    TrackingLost,
}

/// Visual Odometry result
#[derive(Debug, Clone)]
pub struct VOResult {
    /// Estimated pose
    pub pose: SE3,
    /// Number of matches
    pub num_matches: usize,
    /// Number of inliers
    pub num_inliers: usize,
    /// Whether tracking is successful
    pub success: bool,
}

impl VOResult {
    fn failure() -> Self {
        Self {
            pose: SE3::identity(),
            num_matches: 0,
            num_inliers: 0,
            success: false,
        }
    }
}

/// Visual Odometry
pub struct VisualOdometry {
    /// Feature extractor
    extractor: Box<dyn FeatureExtractor>,
    /// Feature matcher
    matcher: Box<dyn FeatureMatcher>,
    /// Camera model
    camera: Camera,
    /// Current state
    state: VOState,
    /// PnP solver for 3D-2D
    pnp_solver: PnPSolver,
    /// Essential matrix solver for 2D-2D
    essential_solver: EssentialSolver,
    /// Triangulator
    triangulator: Triangulator,
    /// Relocalizer for absolute pose recovery against cached keyframes
    relocalizer: Relocalizer,
    /// Lightweight keyframe/map cache for PnP relocalization
    relocalization_map: Map,
    /// Cached keyframe IDs in insertion order
    relocalization_keyframes: VecDeque<u64>,
    /// Last frame id stored as a relocalization anchor
    last_relocalization_anchor_frame: Option<u64>,
    /// Runtime statistics for relocalization behavior on real sequences
    relocalization_stats: RelocalizationStats,
    /// Previous frame ID
    prev_frame_id: Option<u64>,
    /// Previous frame pose
    prev_frame_pose: Option<SE3>,
    /// Previous keypoints
    prev_keypoints: Vec<KeyPoint>,
    /// Previous descriptors
    prev_descriptors: Descriptors,
    /// Previous 3D points (triangulated)
    prev_3d_points: Vec<Option<[f32; 3]>>,
    /// Frame count
    frame_count: u64,
    /// Min matches to proceed
    min_matches: usize,
    /// Min inliers to maintain tracking
    min_inliers: usize,
    /// Median scene depth (camera-space Z) from the last successful PnP frame.
    /// Used to restore metric scale when a fresh monocular re-initialisation is
    /// forced, which would otherwise produce a unit-baseline reconstruction at
    /// an arbitrary scale relative to the global map.
    last_good_median_depth: f32,
}

impl VisualOdometry {
    /// Create a new VO with default feature extraction
    pub fn new(camera: Camera) -> Self {
        Self::with_params(camera, TrackerParams::default())
    }

    /// Create a new VO with tracker parameters
    pub fn with_params(camera: Camera, params: TrackerParams) -> Self {
        let extractor: Box<dyn FeatureExtractor> = match params.feature_type {
            FeatureType::Orb => Box::new(OrbExtractor::new(params.max_features)),
            FeatureType::Harris => Box::new(HarrisExtractor::new(
                params.max_features,
                HarrisParams::default(),
            )),
            FeatureType::Fast => Box::new(FastExtractor::new(
                params.max_features,
                FastParams::default(),
            )),
        };
        // All feature types now produce binary BRIEF descriptors → use Hamming distance
        let matcher: Box<dyn FeatureMatcher> =
            Box::new(HammingMatcher::new(2).with_ratio_threshold(params.match_ratio as f64));

        // Get intrinsics from camera
        let fx = camera.focal.x;
        let fy = camera.focal.y;
        let cx = camera.principal.x;
        let cy = camera.principal.y;

        let pnp_solver = PnPSolver::new(fx, fy, cx, cy);
        let mut relocalizer = Relocalizer::new(fx, fy, cx, cy);
        relocalizer.set_min_inliers(relocalization_min_inliers_from_tracking(params.min_inliers));

        Self {
            extractor,
            matcher,
            camera,
            state: VOState::NotInitialized,
            pnp_solver,
            essential_solver: EssentialSolver::new(),
            triangulator: Triangulator::new(),
            relocalizer,
            relocalization_map: Map::new(),
            relocalization_keyframes: VecDeque::new(),
            last_relocalization_anchor_frame: None,
            relocalization_stats: RelocalizationStats::default(),
            prev_frame_id: None,
            prev_frame_pose: None,
            prev_keypoints: Vec::new(),
            prev_descriptors: Descriptors::new(),
            prev_3d_points: Vec::new(),
            frame_count: 0,
            min_matches: params.min_matches,
            min_inliers: params.min_inliers,
            last_good_median_depth: 0.0,
        }
    }

    /// Create a new VO with custom extractor and matcher
    pub fn with_extractor(
        extractor: Box<dyn FeatureExtractor>,
        matcher: Box<dyn FeatureMatcher>,
        camera: Camera,
    ) -> Self {
        // Get intrinsics from camera
        let fx = camera.focal.x;
        let fy = camera.focal.y;
        let cx = camera.principal.x;
        let cy = camera.principal.y;

        let pnp_solver = PnPSolver::new(fx, fy, cx, cy);
        let mut relocalizer = Relocalizer::new(fx, fy, cx, cy);
        relocalizer.set_min_inliers(relocalization_min_inliers_from_tracking(10));

        Self {
            extractor,
            matcher,
            camera,
            state: VOState::NotInitialized,
            pnp_solver,
            essential_solver: EssentialSolver::new(),
            triangulator: Triangulator::new(),
            relocalizer,
            relocalization_map: Map::new(),
            relocalization_keyframes: VecDeque::new(),
            last_relocalization_anchor_frame: None,
            relocalization_stats: RelocalizationStats::default(),
            prev_frame_id: None,
            prev_frame_pose: None,
            prev_keypoints: Vec::new(),
            prev_descriptors: Descriptors::new(),
            prev_3d_points: Vec::new(),
            frame_count: 0,
            min_matches: 50,
            min_inliers: 10,
            last_good_median_depth: 0.0,
        }
    }

    /// Process a new frame and estimate pose
    pub fn process_frame(&mut self, image: &[u8], width: u32, height: u32) -> VOResult {
        self.frame_count += 1;

        // Extract features
        let (keypoints, descriptors) = match self.extractor.detect_and_compute(image, width, height)
        {
            Ok(result) => result,
            Err(_) => return VOResult::failure(),
        };

        if keypoints.is_empty() {
            self.state = VOState::TrackingLost;
            return VOResult::failure();
        }

        let (result, estimate_kind) = match self.state {
            VOState::NotInitialized | VOState::Initializing => (
                self.initialize(keypoints.clone(), descriptors.clone()),
                Some(FrameEstimateKind::InitializedMonocular),
            ),
            VOState::TrackingOk => (
                self.track(keypoints.clone(), descriptors.clone()),
                Some(FrameEstimateKind::TrackedPnP),
            ),
            VOState::TrackingLost => self.relocalize(keypoints.clone(), descriptors.clone()),
        };

        // Store for next iteration
        if result.success {
            self.prev_frame_id = Some(self.frame_count);
            self.prev_keypoints = keypoints;
            self.prev_descriptors = descriptors;
            self.prev_frame_pose = Some(result.pose);
            if let Some(kind) = estimate_kind {
                self.maybe_store_relocalization_keyframe(kind, result.pose);
            }
        }

        result
    }

    /// Initialize with first few frames (2D-2D motion)
    fn initialize(&mut self, keypoints: Vec<KeyPoint>, descriptors: Descriptors) -> VOResult {
        if self.prev_keypoints.is_empty() {
            // First frame - just store
            self.state = VOState::Initializing;
            self.prev_keypoints = keypoints;
            self.prev_descriptors = descriptors;
            return VOResult {
                pose: SE3::identity(),
                num_matches: 0,
                num_inliers: 0,
                success: true,
            };
        }

        // Second frame - estimate motion using essential matrix
        let matches = match self
            .matcher
            .match_descriptors(&descriptors, &self.prev_descriptors)
        {
            Ok(m) => m,
            Err(_) => {
                self.state = VOState::TrackingLost;
                return VOResult::failure();
            }
        };

        if matches.len() < self.min_matches {
            self.state = VOState::TrackingLost;
            return VOResult::failure();
        }

        // Extract matched points
        let mut pts1 = Vec::with_capacity(matches.len());
        let mut pts2 = Vec::with_capacity(matches.len());
        let mut query_indices = Vec::with_capacity(matches.len());
        for m in &matches {
            let train_idx = m.train_idx as usize;
            let query_idx = m.query_idx as usize;
            if train_idx < self.prev_keypoints.len() && query_idx < keypoints.len() {
                pts1.push(self.normalize_keypoint(&self.prev_keypoints[train_idx]));
                pts2.push(self.normalize_keypoint(&keypoints[query_idx]));
                query_indices.push(query_idx);
            }
        }

        // Compute essential matrix
        if let Some((e, inliers)) = self.essential_solver.compute(&matches, &pts1, &pts2) {
            let inlier_count = inliers.iter().filter(|&&x| x).count();

            if inlier_count >= self.min_inliers && matches.len() >= self.min_matches {
                let anchor_pose = self.prev_frame_pose.unwrap_or(SE3::identity());
                let anchor_to_world = anchor_pose.inverse();
                let mut inlier_pts1 = Vec::with_capacity(inlier_count);
                let mut inlier_pts2 = Vec::with_capacity(inlier_count);
                let mut inlier_query_indices = Vec::with_capacity(inlier_count);
                for (idx, &is_inlier) in inliers.iter().enumerate() {
                    if !is_inlier {
                        continue;
                    }
                    inlier_pts1.push(pts1[idx]);
                    inlier_pts2.push(pts2[idx]);
                    inlier_query_indices.push(query_indices[idx]);
                }

                // Recover pose from essential matrix
                let poses = self.essential_solver.recover_pose(e);
                let prev_pose = SE3::identity();
                let mut best_pose = poses[0];
                let mut best_points = Vec::new();
                let mut best_count = 0usize;

                for pose in poses {
                    let triangulated = self.triangulator.triangulate(
                        &prev_pose,
                        &pose,
                        &inlier_pts1,
                        &inlier_pts2,
                    );
                    let count = triangulated.iter().filter(|pt| pt.is_some()).count();
                    if count > best_count {
                        best_count = count;
                        best_pose = pose;
                        best_points = triangulated;
                    }
                }

                // Require minimum triangulated points to accept monocular init.
                // Too few points indicates degenerate geometry (e.g., pure rotation
                // or insufficient parallax) that will cause unreliable tracking.
                const MIN_TRIANGULATED_POINTS: usize = 8;
                if best_count < MIN_TRIANGULATED_POINTS {
                    self.state = VOState::TrackingLost;
                    return VOResult::failure();
                }

                // Map triangulated points to current keypoint indices
                // (process_frame stores current keypoints as prev_keypoints)
                let mut kp_3d_points = vec![None; keypoints.len()];
                for (i, &query_idx) in inlier_query_indices.iter().enumerate() {
                    if query_idx < keypoints.len() && i < best_points.len() {
                        kp_3d_points[query_idx] = best_points[i].map(|point_local_prev| {
                            anchor_to_world.transform_point(&point_local_prev)
                        });
                    }
                }
                self.prev_3d_points = kp_3d_points;

                self.state = VOState::TrackingOk;
                let global_pose = best_pose.compose(&anchor_pose);

                return VOResult {
                    pose: global_pose,
                    num_matches: matches.len(),
                    num_inliers: inlier_count,
                    success: true,
                };
            }
        }

        self.state = VOState::TrackingLost;
        VOResult::failure()
    }

    /// Track using 3D-2D PnP
    fn track(&mut self, keypoints: Vec<KeyPoint>, descriptors: Descriptors) -> VOResult {
        // Match with previous frame
        let matches = match self
            .matcher
            .match_descriptors(&descriptors, &self.prev_descriptors)
        {
            Ok(m) => m,
            Err(_) => return VOResult::failure(),
        };

        if matches.len() < self.min_matches {
            self.state = VOState::TrackingLost;
            return VOResult::failure();
        }

        // Build PnP problem from 3D-2D correspondences
        let mut pnp_problem = crate::tracker::solver::PnPProblem::new();

        for m in &matches {
            let query_idx = m.query_idx as usize;
            let train_idx = m.train_idx as usize;

            if query_idx < keypoints.len() && train_idx < self.prev_3d_points.len() {
                if let Some(pt3d) = self.prev_3d_points[train_idx] {
                    let pt2d = [keypoints[query_idx].x(), keypoints[query_idx].y()];
                    pnp_problem.add_correspondence(pt2d, pt3d);
                }
            }
        }

        // Solve PnP with RANSAC
        if let Some((pose, inliers)) = self.pnp_solver.solve(&pnp_problem) {
            let inlier_count = inliers.iter().filter(|&&x| x).count();

            if inlier_count >= self.min_inliers {
                // Sanity check: reject poses implying unrealistically large
                // inter-frame camera jumps.  Degenerate 3D point configurations
                // can cause DLT/P3P to produce wildly wrong poses that still
                // accumulate enough "inliers" via reprojection coincidence.
                //
                // Empirical observation from sofa.MOV: normal inter-frame motion
                // is ~0.07 world units (baseline from E-matrix).  A threshold of
                // 1.0 units ≈ 14× normal motion, allowing reasonable acceleration
                // while rejecting clearly degenerate poses.
                if let Some(prev) = self.prev_frame_pose {
                    let prev_center = prev.inverse().translation();
                    let curr_center = pose.inverse().translation();
                    let jump_sq: f32 = prev_center
                        .iter()
                        .zip(curr_center.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    if jump_sq > 1.0 {
                        // Jump > 1.0 world units: this PnP result is suspect
                        self.state = VOState::TrackingLost;
                        return VOResult::failure();
                    }
                }

                // Update prev_3d_points: propagate existing + triangulate new.
                // All matched 3D points are propagated regardless of PnP inlier
                // status; restricting to inliers caused the pool to starve over
                // time, leaving too few correspondences for subsequent frames.
                let prev_pose = self.prev_frame_pose.unwrap_or(SE3::identity());
                let mut new_3d_points = vec![None; keypoints.len()];

                // Collect matches needing triangulation
                let mut tri_prev_pts = Vec::new();
                let mut tri_curr_pts = Vec::new();
                let mut tri_query_indices = Vec::new();

                for m in &matches {
                    let query_idx = m.query_idx as usize;
                    let train_idx = m.train_idx as usize;
                    if query_idx >= keypoints.len() {
                        continue;
                    }

                    // Propagate all matched 3D points into the next frame.
                    if train_idx < self.prev_3d_points.len() {
                        if let Some(pt3d) = self.prev_3d_points[train_idx] {
                            new_3d_points[query_idx] = Some(pt3d);
                            continue;
                        }
                    }
                    // No existing 3D point — queue for triangulation
                    if train_idx < self.prev_keypoints.len() {
                        tri_prev_pts.push(self.normalize_keypoint(&self.prev_keypoints[train_idx]));
                        tri_curr_pts.push(self.normalize_keypoint(&keypoints[query_idx]));
                        tri_query_indices.push(query_idx);
                    }
                }

                // Triangulate new points from two-view geometry
                if !tri_prev_pts.is_empty() {
                    let triangulated = self.triangulator.triangulate(
                        &prev_pose,
                        &pose,
                        &tri_prev_pts,
                        &tri_curr_pts,
                    );
                    for (i, pt) in triangulated.into_iter().enumerate() {
                        if let Some(p) = pt {
                            new_3d_points[tri_query_indices[i]] = Some(p);
                        }
                    }
                }

                self.prev_3d_points = new_3d_points;

                // Record the median camera-space depth so that a subsequent
                // monocular re-init can restore the correct metric scale.
                let mut depths: Vec<f32> = self
                    .prev_3d_points
                    .iter()
                    .filter_map(|p| *p)
                    .filter_map(|pt| {
                        let z = pose.transform_point(&pt)[2];
                        (z > 0.01 && z < 100.0 && z.is_finite()).then_some(z)
                    })
                    .collect();
                if depths.len() >= 4 {
                    depths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    self.last_good_median_depth = depths[depths.len() / 2];
                }

                return VOResult {
                    pose,
                    num_matches: matches.len(),
                    num_inliers: inlier_count,
                    success: true,
                };
            }
        }

        // Fallback to 2D-2D if PnP fails
        self.state = VOState::TrackingLost;
        VOResult::failure()
    }

    /// Try to relocalize after tracking lost
    fn relocalize(
        &mut self,
        keypoints: Vec<KeyPoint>,
        descriptors: Descriptors,
    ) -> (VOResult, Option<FrameEstimateKind>) {
        self.relocalization_stats.lost_events += 1;

        // First try to recover directly against the last retained 3D landmarks.
        // This preserves the existing global frame and avoids re-injecting
        // unit-scale monocular translations after short tracking dropouts.
        if self.prev_3d_points.iter().any(|point| point.is_some()) {
            self.relocalization_stats.direct_retrack_attempts += 1;
            let attempt = self.track(keypoints.clone(), descriptors.clone());
            if attempt.success {
                self.relocalization_stats.direct_retrack_successes += 1;
                self.state = VOState::TrackingOk;
                return (attempt, Some(FrameEstimateKind::RelocalizedPnP));
            }
        }

        // If the most recent frame-specific 3D points are no longer enough,
        // try to relocalize against older cached keyframes that still carry
        // valid 3D landmarks in the current global frame.
        self.relocalization_stats.anchor_relocalization_calls += 1;
        if let Some(attempt) = self.try_anchor_relocalization(&keypoints, &descriptors) {
            self.state = VOState::TrackingOk;
            return (attempt, Some(FrameEstimateKind::RelocalizedPnP));
        }

        // Fall back to a fresh monocular initialization only when absolute
        // relocalization is not available. This keeps the pipeline alive, but
        // it may reintroduce scale ambiguity for the new segment.
        let attempt = if !self.prev_keypoints.is_empty() && !self.prev_descriptors.is_empty() {
            self.relocalization_stats.monocular_reinit_attempts += 1;
            self.initialize(keypoints.clone(), descriptors.clone())
        } else {
            VOResult::failure()
        };
        if attempt.success {
            self.relocalization_stats.monocular_reinit_successes += 1;
            return (attempt, Some(FrameEstimateKind::InitializedMonocular));
        }

        // If re-initialization failed, stage the current frame as the new seed.
        // Keep prev_frame_pose so the next successful initialize() anchors the
        // new segment at the last known camera position instead of world origin.
        self.prev_keypoints = keypoints;
        self.prev_descriptors = descriptors;
        self.prev_3d_points.clear();
        // NOTE: intentionally do NOT clear prev_frame_pose here.
        self.state = VOState::Initializing;

        (VOResult::failure(), None)
    }

    fn maybe_store_relocalization_keyframe(
        &mut self,
        _estimate_kind: FrameEstimateKind,
        pose: SE3,
    ) {
        let valid_points = self
            .prev_3d_points
            .iter()
            .filter(|point| point.is_some())
            .count();
        let min_anchor_points = relocalization_min_inliers_from_tracking(self.min_inliers);
        if valid_points < min_anchor_points {
            return;
        }

        let frame_id = self.prev_frame_id.unwrap_or(self.frame_count);
        if let Some(last_anchor_frame) = self.last_relocalization_anchor_frame {
            if frame_id.saturating_sub(last_anchor_frame) < RELOCALIZATION_ANCHOR_INTERVAL_FRAMES {
                return;
            }
        }

        let mut features = FrameFeatures {
            keypoints: self
                .prev_keypoints
                .iter()
                .map(|keypoint| [keypoint.x(), keypoint.y()])
                .collect(),
            descriptors: self.prev_descriptors.data.clone(),
            map_points: vec![None; self.prev_keypoints.len()],
        };
        for (idx, point) in self.prev_3d_points.iter().enumerate() {
            let Some(point_world) = point else {
                continue;
            };
            if !point_world.iter().all(|value| value.is_finite()) {
                continue;
            }
            let point_id = self.relocalization_map.add_point(MapPoint::new(
                0,
                Vec3::from(*point_world),
                frame_id,
            ));
            if idx < features.map_points.len() {
                features.map_points[idx] = Some(point_id);
            }
        }

        let mut frame = Frame::new(
            frame_id,
            frame_id as f64,
            self.camera.width,
            self.camera.height,
        );
        frame.set_pose(pose);
        frame.mark_as_keyframe();
        let keyframe_id = self
            .relocalization_map
            .add_keyframe(KeyFrame::new(frame, features));

        self.relocalization_keyframes.push_back(keyframe_id);
        self.last_relocalization_anchor_frame = Some(frame_id);
        self.relocalization_stats.anchor_store_successes += 1;
        while self.relocalization_keyframes.len() > MAX_RELOCALIZATION_KEYFRAMES {
            self.relocalization_keyframes.pop_front();
        }
    }

    fn try_anchor_relocalization(
        &mut self,
        keypoints: &[KeyPoint],
        descriptors: &Descriptors,
    ) -> Option<VOResult> {
        if self.relocalization_keyframes.is_empty() || descriptors.is_empty() {
            return None;
        }

        let current_keypoints: Vec<[f32; 2]> = keypoints
            .iter()
            .map(|keypoint| [keypoint.x(), keypoint.y()])
            .collect();
        for &keyframe_id in self.relocalization_keyframes.iter().rev() {
            self.relocalization_stats.anchor_candidates_tested += 1;
            let result = self.relocalizer.relocalize_against_keyframe(
                &self.relocalization_map,
                keyframe_id,
                &descriptors.data,
                &current_keypoints,
            );
            if !result.success {
                continue;
            }
            let seeded_points = self.seed_points_from_anchor(keyframe_id, keypoints, descriptors);
            let seeded_count = seeded_points.iter().filter(|point| point.is_some()).count();
            if seeded_count < 4 {
                continue;
            }

            self.prev_3d_points = seeded_points;
            self.relocalization_stats.anchor_relocalization_successes += 1;
            return Some(VOResult {
                pose: result.pose.unwrap_or(SE3::identity()),
                num_matches: seeded_count,
                num_inliers: result.num_inliers,
                success: true,
            });
        }

        None
    }

    fn seed_points_from_anchor(
        &self,
        keyframe_id: u64,
        keypoints: &[KeyPoint],
        descriptors: &Descriptors,
    ) -> Vec<Option<[f32; 3]>> {
        let mut seeded_points = vec![None; keypoints.len()];
        let Some(anchor_keyframe) = self.relocalization_map.get_keyframe(keyframe_id) else {
            return seeded_points;
        };
        let Some(anchor_descriptors) = descriptors_from_frame_features(&anchor_keyframe.features)
        else {
            return seeded_points;
        };
        let Ok(matches) = self
            .matcher
            .match_descriptors(descriptors, &anchor_descriptors)
        else {
            return seeded_points;
        };

        for matched in matches {
            let query_idx = matched.query_idx as usize;
            let train_idx = matched.train_idx as usize;
            if query_idx >= seeded_points.len()
                || train_idx >= anchor_keyframe.features.map_points.len()
            {
                continue;
            }
            let Some(point_id) = anchor_keyframe.features.map_points[train_idx] else {
                continue;
            };
            let Some(map_point) = self.relocalization_map.get_point(point_id) else {
                continue;
            };
            if map_point.is_outlier || !map_point.position.is_finite() {
                continue;
            }
            seeded_points[query_idx] = Some([
                map_point.position.x,
                map_point.position.y,
                map_point.position.z,
            ]);
        }

        seeded_points
    }

    /// Get current state
    pub fn state(&self) -> VOState {
        self.state
    }

    pub fn relocalization_stats(&self) -> RelocalizationStats {
        let mut stats = self.relocalization_stats;
        stats.cached_anchor_keyframes = self.relocalization_keyframes.len();
        stats
    }

    /// Reset the VO
    pub fn reset(&mut self) {
        self.state = VOState::NotInitialized;
        self.prev_frame_id = None;
        self.prev_frame_pose = None;
        self.prev_keypoints.clear();
        self.prev_descriptors = Descriptors::new();
        self.prev_3d_points.clear();
        self.frame_count = 0;
        self.relocalization_map = Map::new();
        self.relocalization_keyframes.clear();
        self.last_relocalization_anchor_frame = None;
        self.relocalization_stats = RelocalizationStats::default();
        self.last_good_median_depth = 0.0;
    }

    /// Set minimum matches threshold
    pub fn set_min_matches(&mut self, min: usize) {
        self.min_matches = min;
    }

    /// Set minimum inliers threshold
    pub fn set_min_inliers(&mut self, min: usize) {
        self.min_inliers = min;
        self.relocalizer
            .set_min_inliers(relocalization_min_inliers_from_tracking(min));
    }

    /// Get features from the last successfully processed frame.
    pub fn last_features(&self) -> Option<FrameFeatures> {
        if self.prev_keypoints.is_empty() || self.prev_descriptors.is_empty() {
            return None;
        }

        Some(FrameFeatures {
            keypoints: self
                .prev_keypoints
                .iter()
                .map(|kp| [kp.x(), kp.y()])
                .collect(),
            descriptors: self.prev_descriptors.data.clone(),
            map_points: vec![None; self.prev_keypoints.len()],
        })
    }

    /// Get the 3D points tracked for the last successfully processed frame.
    pub fn last_sparse_points(&self) -> Vec<[f32; 3]> {
        self.prev_3d_points
            .iter()
            .filter_map(|point| *point)
            .filter(|point| point.iter().all(|value| value.is_finite()))
            .collect()
    }

    fn normalize_keypoint(&self, keypoint: &KeyPoint) -> [f32; 2] {
        [
            (keypoint.x() - self.camera.principal.x) / self.camera.focal.x.max(1.0),
            (keypoint.y() - self.camera.principal.y) / self.camera.focal.y.max(1.0),
        ]
    }
}

fn descriptors_from_frame_features(features: &FrameFeatures) -> Option<Descriptors> {
    if features.keypoints.is_empty() || features.descriptors.is_empty() {
        return None;
    }
    if features.descriptors.len() % features.keypoints.len() != 0 {
        return None;
    }

    let size = features.descriptors.len() / features.keypoints.len();
    if size == 0 {
        return None;
    }

    Some(Descriptors {
        data: features.descriptors.clone(),
        size,
        count: features.keypoints.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::base::{FeatureError, Match};

    struct TestExtractor {
        keypoints: Vec<KeyPoint>,
        descriptors: Descriptors,
    }

    impl FeatureExtractor for TestExtractor {
        fn detect_and_compute(
            &mut self,
            _image: &[u8],
            _width: u32,
            _height: u32,
        ) -> Result<(Vec<KeyPoint>, Descriptors), FeatureError> {
            Ok((self.keypoints.clone(), self.descriptors.clone()))
        }

        fn detect(
            &mut self,
            _image: &[u8],
            _width: u32,
            _height: u32,
        ) -> Result<Vec<KeyPoint>, FeatureError> {
            Ok(self.keypoints.clone())
        }

        fn num_features(&self) -> usize {
            self.keypoints.len()
        }

        fn set_num_features(&mut self, _num: usize) {}
    }

    struct TestMatcher;

    impl FeatureMatcher for TestMatcher {
        fn match_descriptors(
            &self,
            _query: &Descriptors,
            _train: &Descriptors,
        ) -> Result<Vec<Match>, FeatureError> {
            Ok(Vec::new())
        }
    }

    struct FixedMatcher(Vec<Match>);

    impl FeatureMatcher for FixedMatcher {
        fn match_descriptors(
            &self,
            _query: &Descriptors,
            _train: &Descriptors,
        ) -> Result<Vec<Match>, FeatureError> {
            Ok(self.0.clone())
        }
    }

    fn make_descriptors(count: usize) -> Descriptors {
        let mut descriptors = Descriptors::with_capacity(count, 32);
        for idx in 0..count {
            for byte_idx in 0..32 {
                descriptors.data[idx * 32 + byte_idx] =
                    ((idx * 37 + byte_idx * 13 + 17) % 251) as u8;
            }
        }
        descriptors
    }

    #[test]
    fn test_vo_state() {
        assert_eq!(VOState::NotInitialized, VOState::NotInitialized);
        assert_ne!(VOState::TrackingOk, VOState::TrackingLost);
    }

    #[test]
    fn test_vo_result() {
        let result = VOResult {
            pose: SE3::identity(),
            num_matches: 100,
            num_inliers: 50,
            success: true,
        };

        assert!(result.success);
        assert_eq!(result.num_inliers, 50);
    }

    #[test]
    fn test_relocalize_failure_preserves_anchor_pose_for_reinit() {
        let camera = Camera::new(525.0, 525.0, 319.5, 239.5, 640, 480);
        let current_kp = KeyPoint::new(10.0, 10.0);
        let mut current_desc = Descriptors::with_capacity(1, 32);
        current_desc.data[0] = 1;

        let extractor = Box::new(TestExtractor {
            keypoints: vec![current_kp.clone()],
            descriptors: current_desc.clone(),
        });
        let matcher = Box::new(TestMatcher);
        let mut vo = VisualOdometry::with_extractor(extractor, matcher, camera);

        vo.state = VOState::TrackingLost;
        vo.prev_keypoints = vec![KeyPoint::new(5.0, 5.0)];
        vo.prev_descriptors = Descriptors::with_capacity(1, 32);
        vo.prev_descriptors.data[0] = 2;
        vo.prev_frame_pose = Some(SE3::identity());
        vo.prev_3d_points = vec![Some([0.0, 0.0, 1.0])];

        let result = vo.process_frame(&vec![0; 640 * 480], 640, 480);

        assert!(!result.success);
        assert_eq!(vo.state(), VOState::Initializing);
        assert_eq!(vo.prev_keypoints.len(), 1);
        assert_eq!(vo.prev_keypoints[0].x(), current_kp.x());
        assert!(vo.prev_3d_points.is_empty());
        assert_eq!(
            vo.prev_frame_pose
                .expect("anchor pose should be preserved")
                .translation(),
            SE3::identity().translation()
        );
    }

    #[test]
    fn test_initialize_keeps_relocalized_pose_in_global_frame() {
        let camera = Camera::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let relative_pose = SE3::from_axis_angle(&[0.01, -0.02, 0.005], &[0.2, 0.0, 0.98]);
        let anchor_pose = SE3::from_axis_angle(&[0.0, 0.0, 0.0], &[0.6, -0.1, 0.25]);

        let world_points_prev = [
            [-0.8, -0.4, 4.0],
            [-0.2, -0.3, 4.4],
            [0.3, -0.5, 4.8],
            [0.7, -0.2, 5.1],
            [-0.6, 0.1, 4.2],
            [-0.1, 0.3, 4.6],
            [0.4, 0.2, 4.9],
            [0.8, 0.4, 5.3],
            [-0.5, 0.6, 4.5],
            [0.0, 0.7, 5.0],
            [0.55, 0.65, 4.7],
            [-0.75, -0.65, 4.8],
            [0.15, -0.7, 5.2],
            [0.65, 0.05, 5.5],
            [-0.35, 0.85, 5.4],
            [0.9, -0.15, 5.8],
        ];

        let prev_keypoints: Vec<KeyPoint> = world_points_prev
            .iter()
            .map(|point_world| {
                let pixel = camera
                    .project(&glam::Vec3::from(*point_world))
                    .expect("point visible in previous view");
                KeyPoint::new(pixel.x, pixel.y)
            })
            .collect();
        let curr_keypoints: Vec<KeyPoint> = world_points_prev
            .iter()
            .map(|point_world| {
                let point_camera = relative_pose.transform_point(point_world);
                let pixel = camera
                    .project(&glam::Vec3::from(point_camera))
                    .expect("point visible in current view");
                KeyPoint::new(pixel.x, pixel.y)
            })
            .collect();

        let matches: Vec<Match> = (0..world_points_prev.len())
            .map(|idx| Match {
                query_idx: idx as u32,
                train_idx: idx as u32,
                distance: 0.0,
            })
            .collect();
        let expected_matches = matches.clone();
        let expected_prev_keypoints = prev_keypoints.clone();
        let expected_curr_keypoints = curr_keypoints.clone();
        let descriptors = make_descriptors(world_points_prev.len());

        let extractor = Box::new(TestExtractor {
            keypoints: curr_keypoints.clone(),
            descriptors: descriptors.clone(),
        });
        let matcher = Box::new(FixedMatcher(matches));
        let mut vo = VisualOdometry::with_extractor(extractor, matcher, camera);
        vo.prev_keypoints = prev_keypoints;
        vo.prev_descriptors = descriptors.clone();
        vo.prev_frame_pose = Some(anchor_pose);
        vo.state = VOState::TrackingLost;
        vo.set_min_matches(8);
        vo.set_min_inliers(6);

        let result = vo.initialize(curr_keypoints, descriptors);

        assert!(result.success);
        let t = result.pose.translation();
        let rel_t = relative_pose.translation();
        assert!(
            (t[0] - rel_t[0]).abs() > 0.25 || (t[2] - rel_t[2]).abs() > 0.25,
            "relocalized pose should remain in the global frame: got {:?}, relative {:?}",
            t,
            rel_t
        );

        let pts1: Vec<[f32; 2]> = expected_prev_keypoints
            .iter()
            .map(|kp| vo.normalize_keypoint(kp))
            .collect();
        let pts2: Vec<[f32; 2]> = expected_curr_keypoints
            .iter()
            .map(|kp| vo.normalize_keypoint(kp))
            .collect();
        let (essential, inliers) = vo
            .essential_solver
            .compute(&expected_matches, &pts1, &pts2)
            .expect("essential matrix");
        let mut inlier_pts1 = Vec::new();
        let mut inlier_pts2 = Vec::new();
        for (idx, &is_inlier) in inliers.iter().enumerate() {
            if is_inlier {
                inlier_pts1.push(pts1[idx]);
                inlier_pts2.push(pts2[idx]);
            }
        }
        let poses = vo.essential_solver.recover_pose(essential);
        let mut expected_local_points = Vec::new();
        let mut best_count = 0usize;
        for pose_candidate in poses {
            let triangulated = vo.triangulator.triangulate(
                &SE3::identity(),
                &pose_candidate,
                &inlier_pts1,
                &inlier_pts2,
            );
            let count = triangulated.iter().filter(|pt| pt.is_some()).count();
            if count > best_count {
                best_count = count;
                expected_local_points = triangulated;
            }
        }

        let anchor_to_world = anchor_pose.inverse();
        let expected_local = expected_local_points
            .into_iter()
            .flatten()
            .next()
            .expect("expected local triangulated point");
        let expected_first_world = anchor_to_world.transform_point(&expected_local);
        let recovered_first_world = vo
            .prev_3d_points
            .iter()
            .flatten()
            .copied()
            .next()
            .expect("triangulated point");
        let dx = recovered_first_world[0] - expected_first_world[0];
        let dy = recovered_first_world[1] - expected_first_world[1];
        let dz = recovered_first_world[2] - expected_first_world[2];
        let err = (dx * dx + dy * dy + dz * dz).sqrt();
        assert!(
            err < 0.5,
            "triangulated point should be converted into global frame, err={err}"
        );
    }

    #[test]
    fn test_relocalize_from_cached_keyframe_uses_absolute_pnp() {
        let camera = Camera::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let current_pose = SE3::from_axis_angle(&[0.01, -0.015, 0.005], &[0.35, -0.1, 0.55]);
        let world_points: Vec<[f32; 3]> = (0..24)
            .map(|idx| {
                let x = ((idx % 6) as f32 - 2.5) * 0.35;
                let y = ((idx / 6) as f32 - 1.5) * 0.3;
                let z = 4.5 + (idx % 4) as f32 * 0.35;
                [x, y, z]
            })
            .collect();

        let anchor_keypoints: Vec<KeyPoint> = world_points
            .iter()
            .map(|point_world| {
                let pixel = camera
                    .project(&glam::Vec3::from(*point_world))
                    .expect("anchor point visible");
                KeyPoint::new(pixel.x, pixel.y)
            })
            .collect();
        let current_keypoints: Vec<KeyPoint> = world_points
            .iter()
            .map(|point_world| {
                let point_camera = current_pose.transform_point(point_world);
                let pixel = camera
                    .project(&glam::Vec3::from(point_camera))
                    .expect("current point visible");
                KeyPoint::new(pixel.x, pixel.y)
            })
            .collect();
        let descriptors = make_descriptors(world_points.len());
        let matches: Vec<Match> = (0..world_points.len())
            .map(|idx| Match {
                query_idx: idx as u32,
                train_idx: idx as u32,
                distance: 0.0,
            })
            .collect();

        let extractor = Box::new(TestExtractor {
            keypoints: current_keypoints.clone(),
            descriptors: descriptors.clone(),
        });
        let matcher = Box::new(FixedMatcher(matches));
        let mut vo = VisualOdometry::with_extractor(extractor, matcher, camera);
        vo.set_min_matches(12);
        vo.set_min_inliers(8);
        vo.prev_frame_id = Some(1);
        vo.frame_count = 1;
        vo.prev_frame_pose = Some(SE3::identity());
        vo.prev_keypoints = anchor_keypoints;
        vo.prev_descriptors = descriptors.clone();
        vo.prev_3d_points = world_points.iter().copied().map(Some).collect();
        vo.maybe_store_relocalization_keyframe(FrameEstimateKind::TrackedPnP, SE3::identity());
        let direct_reloc = vo.relocalizer.relocalize_against_keyframe(
            &vo.relocalization_map,
            *vo.relocalization_keyframes.back().expect("cached keyframe"),
            &descriptors.data,
            &current_keypoints
                .iter()
                .map(|keypoint| [keypoint.x(), keypoint.y()])
                .collect::<Vec<_>>(),
        );
        assert!(
            direct_reloc.success,
            "direct relocalizer result should succeed: inliers={}",
            direct_reloc.num_inliers
        );
        vo.prev_3d_points = vec![None; world_points.len()];
        vo.state = VOState::TrackingLost;

        let result = vo.process_frame(&vec![0; 640 * 480], 640, 480);

        assert!(result.success);
        let expected_t = current_pose.translation();
        let estimated_t = result.pose.translation();
        let err_t = ((estimated_t[0] - expected_t[0]).powi(2)
            + (estimated_t[1] - expected_t[1]).powi(2)
            + (estimated_t[2] - expected_t[2]).powi(2))
        .sqrt();
        assert!(
            err_t < 0.25,
            "expected translation {:?}, got {:?}, err={err_t}",
            expected_t,
            estimated_t
        );
        assert!(
            vo.prev_3d_points
                .iter()
                .filter(|point| point.is_some())
                .count()
                >= vo.min_inliers
        );
    }

    #[test]
    fn test_monocular_segments_can_continue_populating_relocalization_anchors() {
        let camera = Camera::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let extractor = Box::new(TestExtractor {
            keypoints: Vec::new(),
            descriptors: Descriptors::new(),
        });
        let matcher = Box::new(TestMatcher);
        let mut vo = VisualOdometry::with_extractor(extractor, matcher, camera);

        let anchor_points = relocalization_min_inliers_from_tracking(vo.min_inliers);
        vo.prev_3d_points = (0..anchor_points)
            .map(|idx| Some([idx as f32 * 0.1, 0.0, 4.0]))
            .collect();
        vo.prev_keypoints = (0..anchor_points)
            .map(|idx| KeyPoint::new(100.0 + idx as f32, 120.0))
            .collect();
        vo.prev_descriptors = make_descriptors(anchor_points);
        vo.prev_frame_id = Some(16);
        vo.frame_count = 16;
        vo.maybe_store_relocalization_keyframe(FrameEstimateKind::TrackedPnP, SE3::identity());
        assert_eq!(vo.relocalization_keyframes.len(), 1);

        vo.prev_frame_id = Some(32);
        vo.frame_count = 32;
        vo.maybe_store_relocalization_keyframe(
            FrameEstimateKind::InitializedMonocular,
            SE3::identity(),
        );

        assert_eq!(
            vo.relocalization_keyframes.len(),
            2,
            "successful monocular segments should still refresh the anchor pool"
        );
    }
}
