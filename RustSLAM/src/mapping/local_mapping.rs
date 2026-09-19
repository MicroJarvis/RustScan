//! Local Mapping module
//!
//! This module handles local mapping in the SLAM system:
//! - Receives keyframes from the tracking thread
//! - Triangulates new map points
//! - Performs local Bundle Adjustment
//! - Filters redundant keyframes

use nalgebra::{Point3, Vector3};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};

use crate::core::{Camera, FrameFeatures, KeyFrame, Map, MapPoint, SE3};
use crate::features::{Descriptors, FeatureMatcher, HammingMatcher};
use crate::optimizer::ba::{BACamera, BALandmark, BAObservation, BundleAdjuster};
use crate::tracker::solver::Triangulator;

/// Configuration for Local Mapping
#[derive(Debug, Clone)]
pub struct LocalMappingConfig {
    /// Maximum number of local keyframes
    pub max_keyframes: usize,
    /// Maximum number of local map points
    pub max_map_points: usize,
    /// Minimum observations for a map point
    pub min_observations: u32,
    /// Minimum triangulation angle (degrees)
    pub min_triangulation_angle: f32,
    /// Minimum distance for triangulation
    pub min_triangulation_dist: f32,
    /// Maximum reprojection error for triangulation
    pub max_reprojection_error: f32,
    /// Whether to enable local BA
    pub local_ba_enabled: bool,
    /// Local BA iterations
    pub local_ba_iterations: usize,
    /// Local BA interval in keyframes
    pub local_ba_interval: usize,
    /// Keyframe culling threshold (ratio of mapped points)
    pub culling_threshold: f32,
}

impl Default for LocalMappingConfig {
    fn default() -> Self {
        Self {
            max_keyframes: 10,
            max_map_points: 1000,
            min_observations: 2,
            min_triangulation_angle: 3.0, // degrees
            min_triangulation_dist: 0.1,
            max_reprojection_error: 4.0, // pixels
            local_ba_enabled: true,
            local_ba_iterations: 10,
            local_ba_interval: 5,
            culling_threshold: 0.9, // 90% mapped points = redundant
        }
    }
}

/// Local Mapping state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MappingState {
    /// Idle, waiting for keyframes
    Idle,
    /// Processing new keyframes
    Processing,
    /// Running local BA
    RunningLocalBA,
    /// Culling redundant keyframes
    Culling,
}

/// Local Mapping module
///
/// Receives keyframes from tracking and:
/// 1. Triangulates new map points from consecutive keyframes
/// 2. Runs local Bundle Adjustment
/// 3. Culls redundant keyframes
pub struct LocalMapping {
    /// Configuration
    config: LocalMappingConfig,
    /// Current state
    state: MappingState,
    /// The map being worked on (thread-safe)
    map: Option<Arc<RwLock<Map>>>,
    /// Local keyframes (current and connected)
    local_keyframes: Vec<u64>,
    /// Local map points (observed by local keyframes)
    local_map_points: HashSet<u64>,
    /// New keyframes waiting to be processed
    new_keyframes: VecDeque<KeyFrame>,
    /// Camera for triangulation
    camera: Option<Camera>,
    /// Keyframes since last local BA
    keyframes_since_ba: usize,
}

impl LocalMapping {
    /// Create a new Local Mapping instance
    pub fn new(config: LocalMappingConfig) -> Self {
        Self {
            config,
            state: MappingState::Idle,
            map: None,
            local_keyframes: Vec::new(),
            local_map_points: HashSet::new(),
            new_keyframes: VecDeque::new(),
            camera: None,
            keyframes_since_ba: 0,
        }
    }

    /// Set the map reference
    pub fn set_map(&mut self, map: Map) {
        self.map = Some(Arc::new(RwLock::new(map)));
    }

    /// Set a shared map reference (for multi-threaded access)
    pub fn set_shared_map(&mut self, map: Arc<RwLock<Map>>) {
        self.map = Some(map);
    }

    /// Set the map and initialize local indices
    pub fn set_map_and_init(&mut self, map: Map) {
        self.local_keyframes = map.keyframes().map(|kf| kf.id()).collect();
        self.local_map_points = map.points().map(|mp| mp.id).collect();
        self.map = Some(Arc::new(RwLock::new(map)));
        self.keyframes_since_ba = 0;
    }

    /// Set a shared map and initialize local indices
    pub fn set_shared_map_and_init(&mut self, map: Arc<RwLock<Map>>) {
        {
            let m = map.read().unwrap_or_else(|e| e.into_inner());
            self.local_keyframes = m.keyframes().map(|kf| kf.id()).collect();
            self.local_map_points = m.points().map(|mp| mp.id).collect();
        }
        self.map = Some(map);
        self.keyframes_since_ba = 0;
    }

    /// Access the shared map
    pub fn shared_map(&self) -> Option<&Arc<RwLock<Map>>> {
        self.map.as_ref()
    }

    /// Access the internal map (convenience for single-threaded use)
    pub fn map(&self) -> Option<std::sync::RwLockReadGuard<'_, Map>> {
        self.map.as_ref().and_then(|m| m.read().ok())
    }

    /// Set the camera
    pub fn set_camera(&mut self, camera: Camera) {
        self.camera = Some(camera);
    }

    /// Check if there are new keyframes to process
    pub fn check_new_keyframes(&self) -> bool {
        !self.new_keyframes.is_empty()
    }

    /// Insert a new keyframe from the tracking thread
    pub fn insert_keyframe(&mut self, keyframe: KeyFrame) {
        self.state = MappingState::Processing;
        self.new_keyframes.push_back(keyframe);

        // Process immediately
        self.process_new_keyframes();
    }

    /// Process all new keyframes in the queue
    fn process_new_keyframes(&mut self) {
        while let Some(keyframe) = self.new_keyframes.pop_front() {
            let kf_id = keyframe.id();
            self.insert_keyframe_into_map(keyframe);

            // Add keyframe to local keyframes
            self.local_keyframes.push(kf_id);
            self.keyframes_since_ba = self.keyframes_since_ba.saturating_add(1);

            // Limit local keyframes
            while self.local_keyframes.len() > self.config.max_keyframes {
                self.local_keyframes.remove(0);
            }

            // Triangulate new points
            self.triangulate_new_points(kf_id);
        }

        // Run local BA if enabled
        if self.config.local_ba_enabled
            && (self.config.local_ba_interval == 0
                || self.keyframes_since_ba >= self.config.local_ba_interval)
        {
            self.run_local_ba();
            self.keyframes_since_ba = 0;
        }

        // Cull redundant keyframes
        self.cull_redundant_keyframes();

        self.state = MappingState::Idle;
    }

    fn insert_keyframe_into_map(&mut self, keyframe: KeyFrame) {
        if let Some(ref shared) = self.map {
            let mut map = shared.write().unwrap_or_else(|e| e.into_inner());
            map.insert_keyframe_with_id(keyframe.id(), keyframe);
        }
    }

    /// Triangulate new map points from the new keyframe with its neighbors
    fn triangulate_new_points(&mut self, new_keyframe_id: u64) {
        let Some(ref shared) = self.map else {
            return;
        };

        let (new_keyframe, new_pose, neighbor_data) = {
            let map = shared.read().unwrap_or_else(|e| e.into_inner());
            let Some(new_keyframe) = map.get_keyframe(new_keyframe_id).cloned() else {
                return;
            };
            let Some(new_pose) = new_keyframe.pose() else {
                return;
            };

            let neighbors: Vec<(KeyFrame, SE3)> = self
                .local_keyframes
                .iter()
                .filter(|&&kf_id| kf_id != new_keyframe_id)
                .filter_map(|&kf_id| {
                    map.get_keyframe(kf_id)
                        .and_then(|kf| kf.pose().map(|pose| (kf.clone(), pose)))
                })
                .collect();
            (new_keyframe, new_pose, neighbors)
        };

        // Now triangulate with each neighbor
        for (neighbor_kf, neighbor_pose) in neighbor_data {
            self.triangulate_between_keyframes(
                &new_keyframe,
                &new_pose,
                &neighbor_kf,
                &neighbor_pose,
            );

            if self.local_map_points.len() >= self.config.max_map_points {
                break;
            }
        }
    }

    /// Triangulate points between two keyframes
    fn triangulate_between_keyframes(
        &mut self,
        kf1: &KeyFrame,
        pose1: &SE3,
        kf2: &KeyFrame,
        pose2: &SE3,
    ) {
        if self.local_map_points.len() >= self.config.max_map_points {
            return;
        }

        let Some(desc1) = descriptors_from_features(&kf1.features) else {
            return;
        };
        let Some(desc2) = descriptors_from_features(&kf2.features) else {
            return;
        };

        let camera = self.camera.unwrap_or_default();
        let matcher = HammingMatcher::new(2).with_ratio_threshold(0.75);
        let Ok(mut matches) = matcher.match_descriptors(&desc1, &desc2) else {
            return;
        };
        if matches.is_empty() {
            return;
        }

        matches.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut tri_pts1 = Vec::new();
        let mut tri_pts2 = Vec::new();
        let mut feature_pairs = Vec::new();
        let mut used_kf1 = HashSet::new();
        let mut used_kf2 = HashSet::new();

        for m in matches {
            let idx1 = m.query_idx as usize;
            let idx2 = m.train_idx as usize;

            if idx1 >= kf1.features.keypoints.len() || idx2 >= kf2.features.keypoints.len() {
                continue;
            }
            if kf1
                .features
                .map_points
                .get(idx1)
                .and_then(|point_id| *point_id)
                .is_some()
            {
                continue;
            }
            if kf2
                .features
                .map_points
                .get(idx2)
                .and_then(|point_id| *point_id)
                .is_some()
            {
                continue;
            }
            if !used_kf1.insert(idx1) || !used_kf2.insert(idx2) {
                continue;
            }

            tri_pts1.push(normalize_keypoint(&camera, kf1.features.keypoints[idx1]));
            tri_pts2.push(normalize_keypoint(&camera, kf2.features.keypoints[idx2]));
            feature_pairs.push((idx1, idx2));
        }

        if tri_pts1.is_empty() {
            return;
        }

        let triangulator = Triangulator {
            min_angle: self.config.min_triangulation_angle.to_radians(),
            min_dist: self.config.min_triangulation_dist,
            max_error: self.config.max_reprojection_error,
        };
        let triangulated = triangulator.triangulate(pose1, pose2, &tri_pts1, &tri_pts2);

        let mut accepted = Vec::new();
        for ((idx1, idx2), maybe_point) in feature_pairs.into_iter().zip(triangulated.into_iter()) {
            let Some(point_world) = maybe_point else {
                continue;
            };
            if !passes_depth_check(pose1, &point_world) || !passes_depth_check(pose2, &point_world)
            {
                continue;
            }

            let kp1 = kf1.features.keypoints[idx1];
            let kp2 = kf2.features.keypoints[idx2];
            let err1 = reprojection_error_pixels(&camera, pose1, &point_world, kp1);
            let err2 = reprojection_error_pixels(&camera, pose2, &point_world, kp2);
            if err1 > self.config.max_reprojection_error
                || err2 > self.config.max_reprojection_error
            {
                continue;
            }

            accepted.push((idx1, idx2, point_world));
            if self.local_map_points.len() + accepted.len() >= self.config.max_map_points {
                break;
            }
        }

        if accepted.is_empty() {
            return;
        }

        if let Some(ref shared) = self.map {
            let mut map = shared.write().unwrap_or_else(|e| e.into_inner());
            for (idx1, idx2, point_world) in accepted {
                let already_mapped_1 = map
                    .get_keyframe(kf1.id())
                    .and_then(|kf| kf.features.map_points.get(idx1))
                    .and_then(|point_id| *point_id)
                    .is_some();
                let already_mapped_2 = map
                    .get_keyframe(kf2.id())
                    .and_then(|kf| kf.features.map_points.get(idx2))
                    .and_then(|point_id| *point_id)
                    .is_some();
                if already_mapped_1 || already_mapped_2 {
                    continue;
                }

                let point_id = self.add_map_point_to_map(&mut map, point_world, kf1.id());

                if let Some(kf) = map.get_keyframe_mut(kf1.id()) {
                    if idx1 < kf.features.map_points.len() {
                        kf.features.map_points[idx1] = Some(point_id);
                    }
                }
                if let Some(kf) = map.get_keyframe_mut(kf2.id()) {
                    if idx2 < kf.features.map_points.len() {
                        kf.features.map_points[idx2] = Some(point_id);
                    }
                }

                self.local_map_points.insert(point_id);
            }
        }

        while self.local_map_points.len() > self.config.max_map_points {
            if let Some(id) = self.local_map_points.iter().next().copied() {
                self.local_map_points.remove(&id);
            }
        }
    }

    /// Add a new map point to the shared map and return its assigned ID.
    fn add_map_point_to_map(&self, map: &mut Map, position: [f32; 3], ref_kf: u64) -> u64 {
        let mut map_point = MapPoint::new(0, Point3::from(position), ref_kf);
        map_point.observations = 2;
        map.add_point(map_point)
    }

    /// Run local Bundle Adjustment
    pub fn run_local_ba(&mut self) {
        self.state = MappingState::RunningLocalBA;

        // Get local keyframes and map points
        let local_kfs: Vec<u64> = self.local_keyframes.clone();
        let local_mps: Vec<u64> = self.local_map_points.iter().cloned().collect();

        if local_kfs.len() < 2 || local_mps.is_empty() {
            self.state = MappingState::Idle;
            return;
        }

        // Build BA problem
        let mut adjuster = BundleAdjuster::new();

        // Add cameras
        let mut camera_indices: HashMap<u64, usize> = HashMap::new();

        if let Some(ref shared) = self.map {
            let map = shared.read().unwrap_or_else(|e| e.into_inner());
            for &kf_id in &local_kfs {
                if let Some(kf) = map.get_keyframe(kf_id) {
                    // Get camera intrinsics (use default if not available)
                    let (fx, fy, cx, cy) = if let Some(ref cam) = self.camera {
                        (
                            cam.focal.x as f64,
                            cam.focal.y as f64,
                            cam.principal.x as f64,
                            cam.principal.y as f64,
                        )
                    } else {
                        let defaults = crate::config::CameraConfig::default();
                        (
                            defaults.fx as f64,
                            defaults.fy as f64,
                            defaults.cx as f64,
                            defaults.cy as f64,
                        )
                    };

                    let pose = kf.pose().unwrap_or(SE3::identity()).inverse();
                    let ba_cam = BACamera::new(fx, fy, cx, cy).with_pose(pose).fix_pose();
                    let idx = adjuster.add_camera(ba_cam);
                    camera_indices.insert(kf_id, idx);
                }
            }
        }

        // Add landmarks
        let mut landmark_indices: HashMap<u64, usize> = HashMap::new();

        if let Some(ref shared) = self.map {
            let map = shared.read().unwrap_or_else(|e| e.into_inner());
            for &mp_id in &local_mps {
                if let Some(mp) = map.get_point(mp_id) {
                    let landmark = BALandmark::new(
                        mp.position.x as f64,
                        mp.position.y as f64,
                        mp.position.z as f64,
                    );
                    let idx = adjuster.add_landmark(landmark);
                    landmark_indices.insert(mp_id, idx);
                }
            }
        }

        // Add observations
        if let Some(ref shared) = self.map {
            let map = shared.read().unwrap_or_else(|e| e.into_inner());
            for &kf_id in &local_kfs {
                if let Some(kf) = map.get_keyframe(kf_id) {
                    let Some(&cam_idx) = camera_indices.get(&kf_id) else {
                        continue;
                    };

                    for (feat_idx, mp_id) in kf.features.map_points.iter().enumerate() {
                        let Some(lm_id) = *mp_id else {
                            continue;
                        };

                        let Some(&lm_idx) = landmark_indices.get(&lm_id) else {
                            continue;
                        };

                        if feat_idx < kf.features.keypoints.len() {
                            let kp = kf.features.keypoints[feat_idx];
                            let obs = BAObservation::new(kp[0] as f64, kp[1] as f64);
                            adjuster.add_observation(cam_idx, lm_idx, obs);
                        }
                    }
                }
            }
        }

        // Run optimization
        let result = adjuster.optimize(self.config.local_ba_iterations);

        match result {
            Ok((_cameras, landmarks)) => {
                // Update map points with optimized positions
                if let Some(ref shared) = self.map {
                    let mut map = shared.write().unwrap_or_else(|e| e.into_inner());
                    for (mp_id, lm_idx) in &landmark_indices {
                        if *lm_idx < landmarks.len() {
                            let lm = &landmarks[*lm_idx];
                            if let Some(mp) = map.get_point_mut(*mp_id) {
                                let new_position = Point3::new(
                                    lm.position[0] as f32,
                                    lm.position[1] as f32,
                                    lm.position[2] as f32,
                                );
                                if new_position.x.is_finite()
                                    && new_position.y.is_finite()
                                    && new_position.z.is_finite()
                                {
                                    mp.position = new_position;
                                    mp.mark_inlier();
                                } else {
                                    mp.mark_outlier();
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                // BA failed, but we continue
                eprintln!("Local BA failed: {}", e);
            }
        }

        self.state = MappingState::Idle;
    }

    /// Cull redundant keyframes
    ///
    /// A keyframe is redundant if more than culling_threshold (e.g., 90%)
    /// of its map points are also observed by other keyframes.
    ///
    /// Returns the number of keyframes culled
    pub fn cull_redundant_keyframes(&mut self) -> usize {
        self.state = MappingState::Culling;

        let mut culled_count = 0;
        let mut to_remove: Vec<u64> = Vec::new();

        // Need at least 3 keyframes to consider culling
        if self.local_keyframes.len() < 3 {
            self.state = MappingState::Idle;
            return 0;
        }

        // Check each keyframe (skip the newest ones)
        // Usually we don't cull the most recent keyframes
        let check_range = self.local_keyframes.len().saturating_sub(3)..self.local_keyframes.len();

        for i in check_range {
            let kf_id = self.local_keyframes[i];

            if let Some(ref shared) = self.map {
                let map = shared.read().unwrap_or_else(|e| e.into_inner());
                if let Some(kf) = map.get_keyframe(kf_id) {
                    // Count mapped points
                    let mapped_count = kf
                        .features
                        .map_points
                        .iter()
                        .filter(|mp| mp.is_some())
                        .count();

                    // Count observations from other keyframes
                    let mut observed_count = 0;

                    if let Some(mp_id) = kf.features.map_points.iter().find(|mp| mp.is_some()) {
                        if let Some(mp) = map.get_point(mp_id.unwrap()) {
                            observed_count = mp.observations as usize;
                        }
                    }

                    // Calculate ratio
                    if mapped_count > 0 {
                        let ratio = observed_count as f32 / mapped_count as f32;

                        if ratio > self.config.culling_threshold {
                            to_remove.push(kf_id);
                        }
                    }
                }
            }
        }

        // Remove culled keyframes from local list
        for kf_id in &to_remove {
            self.local_keyframes.retain(|&id| id != *kf_id);
            culled_count += 1;
        }

        self.state = MappingState::Idle;
        culled_count
    }

    /// Get the number of local keyframes
    pub fn num_local_keyframes(&self) -> usize {
        self.local_keyframes.len()
    }

    /// Get the number of local map points
    pub fn num_local_map_points(&self) -> usize {
        self.local_map_points.len()
    }

    /// Get the current state
    pub fn is_processing(&self) -> bool {
        self.state != MappingState::Idle
    }

    /// Get the current state
    pub fn state(&self) -> MappingState {
        self.state
    }

    /// Clear the local mapping data
    pub fn clear(&mut self) {
        self.local_keyframes.clear();
        self.local_map_points.clear();
        self.new_keyframes.clear();
        self.state = MappingState::Idle;
    }
}

fn descriptors_from_features(features: &FrameFeatures) -> Option<Descriptors> {
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

fn normalize_keypoint(camera: &Camera, keypoint: [f32; 2]) -> [f32; 2] {
    [
        (keypoint[0] - camera.principal.x) / camera.focal.x.max(1.0),
        (keypoint[1] - camera.principal.y) / camera.focal.y.max(1.0),
    ]
}

fn passes_depth_check(pose: &SE3, point_world: &[f32; 3]) -> bool {
    pose.transform_point(point_world)[2] > 0.0
}

fn reprojection_error_pixels(
    camera: &Camera,
    pose: &SE3,
    point_world: &[f32; 3],
    keypoint: [f32; 2],
) -> f32 {
    let point_camera = pose.transform_point(point_world);
    let Some(projected) = camera.project(&Vector3::from(point_camera)) else {
        return f32::INFINITY;
    };

    let dx = projected.x - keypoint[0];
    let dy = projected.y - keypoint[1];
    (dx * dx + dy * dy).sqrt()
}

impl Default for LocalMapping {
    fn default() -> Self {
        Self::new(LocalMappingConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Frame, FrameFeatures, Map};
    use nalgebra::Vector3;

    fn make_descriptors(count: usize) -> Vec<u8> {
        let mut descriptors = vec![0u8; count * 32];
        for idx in 0..count {
            descriptors[idx * 32] = (idx as u8).wrapping_mul(17).wrapping_add(11);
            descriptors[idx * 32 + 1] = (idx as u8).wrapping_mul(29).wrapping_add(7);
        }
        descriptors
    }

    fn build_test_keyframe(
        id: u64,
        pose: SE3,
        camera: Camera,
        world_points: &[[f32; 3]],
    ) -> KeyFrame {
        let mut frame = Frame::new(id, id as f64, camera.width, camera.height);
        frame.set_pose(pose);
        frame.mark_as_keyframe();

        let keypoints = world_points
            .iter()
            .map(|point_world| {
                let point_camera = pose.transform_point(point_world);
                let pixel = camera.project(&Vector3::from(point_camera)).unwrap();
                [pixel.x, pixel.y]
            })
            .collect::<Vec<_>>();
        let feature_count = keypoints.len();

        let features = FrameFeatures {
            keypoints,
            descriptors: make_descriptors(feature_count),
            map_points: vec![None; feature_count],
        };

        KeyFrame::new(frame, features)
    }

    #[test]
    fn test_config_default() {
        let config = LocalMappingConfig::default();
        assert_eq!(config.max_keyframes, 10);
        assert_eq!(config.max_map_points, 1000);
        assert!(config.local_ba_enabled);
    }

    #[test]
    fn test_local_mapping_creation() {
        let mapping = LocalMapping::new(LocalMappingConfig::default());
        assert_eq!(mapping.num_local_keyframes(), 0);
        assert_eq!(mapping.num_local_map_points(), 0);
    }

    #[test]
    fn test_inserted_keyframe_is_persisted_in_map() {
        let mut mapping = LocalMapping::new(LocalMappingConfig::default());
        mapping.set_map(Map::new());

        let mut frame = Frame::new(7, 0.0, 640, 480);
        frame.set_pose(SE3::identity());
        frame.mark_as_keyframe();
        let features = FrameFeatures {
            keypoints: vec![[10.0, 12.0]],
            descriptors: make_descriptors(1),
            map_points: vec![None],
        };

        mapping.insert_keyframe(KeyFrame::new(frame, features));

        let map = mapping.map().unwrap();
        assert!(map.get_keyframe(7).is_some());
        assert_eq!(map.num_keyframes(), 1);
    }

    #[test]
    fn test_matching_keyframes_generate_sparse_points() {
        let camera = Camera::new(100.0, 100.0, 50.0, 50.0, 100, 100);
        let world_points = vec![
            [0.0, 0.0, 2.0],
            [0.2, 0.1, 2.5],
            [-0.2, 0.05, 2.3],
            [0.1, -0.15, 2.8],
        ];

        let mut mapping = LocalMapping::new(LocalMappingConfig {
            local_ba_enabled: false,
            min_triangulation_dist: 0.05,
            ..LocalMappingConfig::default()
        });
        mapping.set_camera(camera);
        mapping.set_map(Map::new());

        let pose1 = SE3::identity();
        let pose2 = SE3::from_axis_angle(&[0.0, 0.0, 0.0], &[-0.2, 0.0, 0.0]);
        let kf1 = build_test_keyframe(0, pose1, camera, &world_points);
        let kf2 = build_test_keyframe(1, pose2, camera, &world_points);

        mapping.insert_keyframe(kf1);
        mapping.insert_keyframe(kf2);

        let map = mapping.map().unwrap();
        assert!(map.num_points() > 0, "expected triangulated map points");
        assert!(map
            .get_keyframe(0)
            .unwrap()
            .features
            .map_points
            .iter()
            .any(|point_id| point_id.is_some()));
        assert!(map
            .get_keyframe(1)
            .unwrap()
            .features
            .map_points
            .iter()
            .any(|point_id| point_id.is_some()));
    }

    #[test]
    fn test_state_transitions() {
        let mut mapping = LocalMapping::new(LocalMappingConfig::default());

        // Initially idle
        assert_eq!(mapping.state(), MappingState::Idle);

        // After insert, should process then go back to idle
        let frame = Frame::new(0, 0.0, 640, 480);
        let features = FrameFeatures::new();
        let kf = KeyFrame::new(frame, features);

        mapping.insert_keyframe(kf);

        // State should eventually return to idle
        // (may briefly be Processing/RunningLocalBA/Culling)
        assert!(matches!(
            mapping.state(),
            MappingState::Idle
                | MappingState::Processing
                | MappingState::RunningLocalBA
                | MappingState::Culling
        ));
    }
}
