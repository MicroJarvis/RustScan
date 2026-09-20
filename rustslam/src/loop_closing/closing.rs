//! Loop closing coordinator that runs detection and triggers global BA.

use crate::core::Map;
use crate::features::base::ORB_DESCRIPTOR_SIZE;
use crate::features::{Descriptors, FeatureMatcher, HammingMatcher};
use crate::loop_closing::detector::{
    compute_sim3_from_matches, LoopCandidate, LoopDetectionResult, LoopDetector,
};
use crate::optimizer::ba::{BACamera, BALandmark, BAObservation, BundleAdjuster};

/// Loop closing coordinator.
pub struct LoopClosing {
    detector: LoopDetector,
    last_loop_frame_id: Option<u64>,
    global_ba_iterations: usize,
    ba_intrinsics: [f64; 4],
}

impl LoopClosing {
    pub fn new() -> Self {
        let default_cam = crate::core::Camera::default();
        Self {
            detector: LoopDetector::new(),
            last_loop_frame_id: None,
            global_ba_iterations: 50,
            ba_intrinsics: [
                default_cam.focal.x as f64,
                default_cam.focal.y as f64,
                default_cam.principal.x as f64,
                default_cam.principal.y as f64,
            ],
        }
    }

    /// Configure BA intrinsics used during loop-closure global optimization.
    pub fn set_ba_intrinsics(&mut self, fx: f32, fy: f32, cx: f32, cy: f32) {
        self.ba_intrinsics = [fx as f64, fy as f64, cx as f64, cy as f64];
    }

    pub fn last_loop_frame_id(&self) -> Option<u64> {
        self.last_loop_frame_id
    }

    pub fn process(
        &mut self,
        map: &mut Map,
        current_frame_id: u64,
        descriptors: &Descriptors,
    ) -> LoopDetectionResult {
        if map.num_keyframes() < 2 {
            return LoopDetectionResult::no_loop();
        }

        let candidates =
            self.detector
                .compute_loop_candidates(map, current_frame_id, &descriptors.data);
        let consistent = self
            .detector
            .compute_loop_consistency(&candidates, current_frame_id);
        let Some(best) = consistent.first().cloned() else {
            return LoopDetectionResult::no_loop();
        };

        let Some(candidate_kf) = map.get_keyframe(best.keyframe_id) else {
            return LoopDetectionResult::no_loop();
        };
        let Some(current_kf) = map.get_keyframe(current_frame_id) else {
            return LoopDetectionResult::no_loop();
        };

        let current_desc = descriptors;
        let candidate_desc = Descriptors {
            data: candidate_kf.features.descriptors.clone(),
            size: ORB_DESCRIPTOR_SIZE,
            count: candidate_kf
                .features
                .descriptors
                .len()
                .checked_div(ORB_DESCRIPTOR_SIZE)
                .unwrap_or(0),
        };

        let matcher = HammingMatcher::new(2).with_ratio_threshold(0.75);
        let matches = matcher
            .match_descriptors(current_desc, &candidate_desc)
            .unwrap_or_default();

        if matches.len() < self.detector.min_matches() {
            return LoopDetectionResult::no_loop();
        }

        let mut points1 = Vec::new();
        let mut points2 = Vec::new();
        for m in &matches {
            let q_idx = m.query_idx as usize;
            let t_idx = m.train_idx as usize;

            let Some(q_mp_id) = current_kf.features.map_points.get(q_idx).and_then(|id| *id) else {
                continue;
            };
            let Some(t_mp_id) = candidate_kf
                .features
                .map_points
                .get(t_idx)
                .and_then(|id| *id)
            else {
                continue;
            };

            let Some(mp1) = map.get_point(q_mp_id) else {
                continue;
            };
            let Some(mp2) = map.get_point(t_mp_id) else {
                continue;
            };

            points1.push([mp1.position.x, mp1.position.y, mp1.position.z]);
            points2.push([mp2.position.x, mp2.position.y, mp2.position.z]);
        }

        let sim3 = compute_sim3_from_matches(&points1, &points2);
        if sim3.is_none() {
            return LoopDetectionResult::no_loop();
        }

        self.detector.add_confirmed_loop(best.keyframe_id);
        self.last_loop_frame_id = Some(best.keyframe_id);

        let _ = run_global_ba(map, self.global_ba_iterations, self.ba_intrinsics);

        LoopDetectionResult {
            loop_detected: true,
            candidate: Some(LoopCandidate {
                keyframe_id: best.keyframe_id,
                score: best.score,
                timestamp: best.timestamp,
            }),
            sim3,
            num_matches: matches.len(),
            num_inliers: points1.len(),
        }
    }
}

fn run_global_ba(map: &mut Map, iterations: usize, intrinsics: [f64; 4]) -> Result<(), String> {
    let mut adjuster = BundleAdjuster::new();

    let mut camera_indices = std::collections::HashMap::new();
    for kf in map.keyframes() {
        let mut cam = BACamera::new(intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]);
        if let Some(pose) = kf.pose() {
            cam = cam.with_pose(pose);
        }
        let idx = adjuster.add_camera(cam);
        camera_indices.insert(kf.id(), idx);
    }

    let mut landmark_indices = std::collections::HashMap::new();
    for mp in map.points() {
        let idx = adjuster.add_landmark(BALandmark::new(
            mp.position.x as f64,
            mp.position.y as f64,
            mp.position.z as f64,
        ));
        landmark_indices.insert(mp.id, idx);
    }

    for kf in map.keyframes() {
        let Some(&cam_idx) = camera_indices.get(&kf.id()) else {
            continue;
        };
        for (feat_idx, mp_id) in kf.features.map_points.iter().enumerate() {
            let Some(mp_id) = *mp_id else {
                continue;
            };
            let Some(&lm_idx) = landmark_indices.get(&mp_id) else {
                continue;
            };
            if feat_idx < kf.features.keypoints.len() {
                let kp = kf.features.keypoints[feat_idx];
                adjuster.add_observation(
                    cam_idx,
                    lm_idx,
                    BAObservation::new(kp[0] as f64, kp[1] as f64),
                );
            }
        }
    }

    let (cameras, landmarks) = adjuster.optimize(iterations)?;

    for (mp_id, lm_idx) in &landmark_indices {
        if *lm_idx < landmarks.len() {
            let lm = &landmarks[*lm_idx];
            if let Some(mp) = map.get_point_mut(*mp_id) {
                mp.position.x = lm.position[0] as f32;
                mp.position.y = lm.position[1] as f32;
                mp.position.z = lm.position[2] as f32;
            }
        }
    }

    for (kf_id, cam_idx) in &camera_indices {
        if *cam_idx < cameras.len() {
            if let Some(kf) = map.get_keyframe_mut(*kf_id) {
                kf.frame.set_pose(cameras[*cam_idx].pose);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Frame, FrameFeatures, KeyFrame, SE3};

    #[test]
    fn test_loop_closing_empty_map() {
        let mut loop_closing = LoopClosing::new();
        let mut map = Map::new();
        let result = loop_closing.process(&mut map, 0, &Descriptors::new());
        assert!(!result.loop_detected);
    }

    #[test]
    fn test_loop_closing_with_minimal_keyframes() {
        let mut loop_closing = LoopClosing::new();
        let mut map = Map::new();

        let mut frame = Frame::new(0, 0.0, 10, 10);
        frame.set_pose(SE3::identity());
        let features = FrameFeatures::new();
        let kf = KeyFrame::new(frame, features);
        map.add_keyframe(kf);

        let result = loop_closing.process(&mut map, 1, &Descriptors::new());
        assert!(!result.loop_detected);
    }
}
