//! Relocalization module for recovering tracking
//!
//! Provides functionality to relocalize the camera when tracking is lost
//! by querying the BoW database for similar keyframes and attempting
//! PnP pose estimation against top-N candidates.

use crate::core::{KeyFrame, Map, SE3};
use crate::features::base::{Descriptors, FeatureMatcher, ORB_DESCRIPTOR_SIZE};
use crate::features::HammingMatcher;
use crate::loop_closing::{KeyFrameDatabase, Vocabulary};
use crate::tracker::solver::{PnPProblem, PnPSolver};

/// Relocalization result
#[derive(Debug, Clone)]
pub struct RelocalizationResult {
    /// Whether relocalization was successful
    pub success: bool,
    /// Recovered pose
    pub pose: Option<SE3>,
    /// Number of inliers
    pub num_inliers: usize,
    /// Matched keyframe ID
    pub matched_keyframe_id: Option<u64>,
}

impl RelocalizationResult {
    /// Create a failure result
    pub fn failed() -> Self {
        Self {
            success: false,
            pose: None,
            num_inliers: 0,
            matched_keyframe_id: None,
        }
    }
}

/// Relocalizer for recovering from tracking loss
pub struct Relocalizer {
    /// PnP solver for pose estimation
    pnp_solver: PnPSolver,
    /// Minimum inliers for successful relocalization
    min_inliers: usize,
    /// Minimum score for candidate keyframes
    min_score: f32,
    /// Maximum candidates to try
    max_candidates: usize,
}

impl Relocalizer {
    /// Create a new relocalizer
    pub fn new(fx: f32, fy: f32, cx: f32, cy: f32) -> Self {
        Self {
            pnp_solver: PnPSolver::new(fx, fy, cx, cy),
            min_inliers: 15,
            min_score: 0.5,
            max_candidates: 20,
        }
    }

    /// Set minimum inliers
    pub fn set_min_inliers(&mut self, min: usize) {
        self.min_inliers = min;
    }

    /// Set minimum score
    pub fn set_min_score(&mut self, score: f32) {
        self.min_score = score;
    }

    /// Attempt to relocalize using keyframe database
    ///
    /// Queries the BoW database for similar keyframes, then attempts PnP
    /// pose estimation against the top candidates using feature matching.
    pub fn relocalize(
        &self,
        map: &Map,
        database: &KeyFrameDatabase,
        vocabulary: &Vocabulary,
        current_descriptors: &[u8],
        current_keypoints: &[[f32; 2]],
    ) -> RelocalizationResult {
        // Transform current descriptors to BoW
        let bow_vector = vocabulary.transform_bow(&Descriptors::from_slice(current_descriptors));

        if bow_vector.is_empty() {
            return RelocalizationResult::failed();
        }

        // Get candidates from database
        let word_ids: Vec<u32> = bow_vector.keys().copied().collect();
        let candidates = database.get_candidates(&word_ids, 10);

        if candidates.is_empty() {
            return RelocalizationResult::failed();
        }

        // Try each candidate until we find a good match
        for (kf_id, _) in candidates.iter().take(self.max_candidates) {
            if let Some(keyframe) = map.get_keyframe(*kf_id) {
                if let Some(result) =
                    self.try_pnp(keyframe, current_descriptors, current_keypoints, map)
                {
                    if result.success {
                        return result;
                    }
                }
            }
        }

        RelocalizationResult::failed()
    }

    /// Try PnP pose estimation against a keyframe.
    ///
    /// 1. Match current descriptors against keyframe descriptors (Hamming)
    /// 2. Collect 2D-3D correspondences from matched map points
    /// 3. Solve PnP with RANSAC
    fn try_pnp(
        &self,
        keyframe: &KeyFrame,
        current_descriptors: &[u8],
        current_keypoints: &[[f32; 2]],
        map: &Map,
    ) -> Option<RelocalizationResult> {
        let current_desc = Descriptors::from_slice(current_descriptors);
        let kf_desc = Descriptors::from_slice(&keyframe.features.descriptors);

        if current_desc.count == 0 || kf_desc.count == 0 {
            return None;
        }

        // Match features using Hamming distance with ratio test
        let matcher = HammingMatcher::new(2).with_ratio_threshold(0.75);
        let matches = matcher.match_descriptors(&current_desc, &kf_desc).ok()?;

        if matches.len() < 4 {
            return None;
        }

        // Build PnP problem: for each match, look up the keyframe's 3D map point
        let mut problem = PnPProblem::new();

        for m in &matches {
            let kf_idx = m.train_idx as usize;
            let q_idx = m.query_idx as usize;

            if q_idx >= current_keypoints.len() {
                continue;
            }
            if kf_idx >= keyframe.features.map_points.len() {
                continue;
            }

            if let Some(point_id) = keyframe.features.map_points[kf_idx] {
                if let Some(mp) = map.get_point(point_id) {
                    if !mp.is_outlier {
                        problem.add_correspondence(
                            current_keypoints[q_idx],
                            [mp.position.x, mp.position.y, mp.position.z],
                        );
                    }
                }
            }
        }

        if !problem.is_solvable() {
            return None;
        }

        let (pose, inlier_mask) = self.pnp_solver.solve(&problem)?;
        let num_inliers = inlier_mask.iter().filter(|&&b| b).count();

        Some(RelocalizationResult {
            success: num_inliers >= self.min_inliers,
            pose: if num_inliers >= self.min_inliers {
                Some(pose)
            } else {
                None
            },
            num_inliers,
            matched_keyframe_id: Some(keyframe.id()),
        })
    }

    /// Relocalize using a specific reference keyframe (no BoW query).
    ///
    /// Matches current features against the reference keyframe and attempts
    /// PnP if enough 3D correspondences are found.
    pub fn relocalize_against_keyframe(
        &self,
        map: &Map,
        reference_keyframe_id: u64,
        current_descriptors: &[u8],
        current_keypoints: &[[f32; 2]],
    ) -> RelocalizationResult {
        let keyframe = match map.get_keyframe(reference_keyframe_id) {
            Some(kf) => kf,
            None => return RelocalizationResult::failed(),
        };

        match self.try_pnp(keyframe, current_descriptors, current_keypoints, map) {
            Some(result) => result,
            None => RelocalizationResult::failed(),
        }
    }
}

impl Default for Relocalizer {
    fn default() -> Self {
        let cam = crate::config::CameraConfig::default();
        Self::new(cam.fx, cam.fy, cam.cx, cam.cy)
    }
}

/// Extended Descriptors to support slice conversion
pub trait DescriptorsExt {
    fn from_slice(data: &[u8]) -> Descriptors;
}

impl DescriptorsExt for Descriptors {
    fn from_slice(data: &[u8]) -> Descriptors {
        Descriptors {
            data: data.to_vec(),
            size: ORB_DESCRIPTOR_SIZE,
            count: if !data.is_empty() {
                data.len() / ORB_DESCRIPTOR_SIZE
            } else {
                0
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Frame, FrameFeatures, MapPoint};
    use nalgebra::Point3;

    #[test]
    fn test_relocalizer_creation() {
        let reloc = Relocalizer::new(500.0, 500.0, 320.0, 240.0);
        assert_eq!(reloc.min_inliers, 15);
    }

    #[test]
    fn test_relocalization_failed() {
        let result = RelocalizationResult::failed();
        assert!(!result.success);
        assert!(result.pose.is_none());
    }

    #[test]
    fn test_set_parameters() {
        let mut reloc = Relocalizer::default();
        reloc.set_min_inliers(20);
        reloc.set_min_score(0.7);

        assert_eq!(reloc.min_inliers, 20);
    }

    #[test]
    fn test_try_pnp_no_map_points_returns_none() {
        let reloc = Relocalizer::new(500.0, 500.0, 320.0, 240.0);
        let map = Map::new();

        // Keyframe with descriptors but no map points
        let mut features = FrameFeatures::new();
        features.keypoints = vec![[100.0, 100.0]; 10];
        features.descriptors = vec![0xAA; 10 * ORB_DESCRIPTOR_SIZE];
        features.map_points = vec![None; 10];

        let frame = Frame::new(1, 0.0, 640, 480);
        let kf = KeyFrame::new(frame, features);

        let current_desc = vec![0xAA; 5 * ORB_DESCRIPTOR_SIZE];
        let current_kps: Vec<[f32; 2]> = vec![[200.0, 200.0]; 5];

        // No map points → PnP can't be solved
        let result = reloc.try_pnp(&kf, &current_desc, &current_kps, &map);
        assert!(result.is_none() || !result.unwrap().success);
    }

    #[test]
    fn test_try_pnp_with_correspondences() {
        let reloc = Relocalizer::new(500.0, 500.0, 320.0, 240.0);
        let mut map = Map::new();

        // 8 non-coplanar 3D points with correct projections for identity pose
        // u = fx * X/Z + cx, v = fy * Y/Z + cy  (fx=fy=500, cx=320, cy=240)
        let positions = [
            Point3::new(0.0, 0.0, 5.0),
            Point3::new(0.5, 0.0, 5.0),
            Point3::new(0.0, 0.5, 5.0),
            Point3::new(0.5, 0.5, 4.0),
            Point3::new(-0.3, -0.3, 6.0),
            Point3::new(0.3, -0.2, 3.0),
            Point3::new(-0.2, 0.4, 4.0),
            Point3::new(0.4, 0.3, 5.0),
        ];
        for (i, pos) in positions.iter().enumerate() {
            let mp = MapPoint::new(i as u64, *pos, 1);
            map.add_point(mp);
        }

        // Keyframe with matching descriptors and map points
        let mut features = FrameFeatures::new();
        for i in 0..8 {
            features
                .keypoints
                .push([250.0 + i as f32 * 20.0, 240.0 + i as f32 * 10.0]);
            let mut desc = vec![0u8; ORB_DESCRIPTOR_SIZE];
            desc[0] = i as u8;
            desc[1] = (i * 7) as u8;
            features.descriptors.extend_from_slice(&desc);
            features.map_points.push(Some(i as u64));
        }

        let mut frame = Frame::new(1, 0.0, 640, 480);
        frame.set_pose(SE3::identity());
        let kf = KeyFrame::new(frame, features.clone());

        // Current frame: correct projections of the 3D points (identity pose)
        let current_kps: Vec<[f32; 2]> = positions
            .iter()
            .map(|p| [500.0 * p.x / p.z + 320.0, 500.0 * p.y / p.z + 240.0])
            .collect();

        let result = reloc.try_pnp(&kf, &features.descriptors, &current_kps, &map);
        // Should find correspondences and solve PnP successfully
        assert!(result.is_some());
    }

    #[test]
    fn test_relocalize_against_missing_keyframe() {
        let reloc = Relocalizer::default();
        let map = Map::new();

        let result = reloc.relocalize_against_keyframe(&map, 999, &[], &[]);
        assert!(!result.success);
    }
}
