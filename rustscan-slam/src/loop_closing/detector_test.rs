//! Tests for loop closing module

use crate::core::{Frame, FrameFeatures, KeyFrame, Map, SE3};
use crate::features::{Descriptors, KeyPoint};
use crate::features::base::ORB_DESCRIPTOR_SIZE;

mod detector_tests {
    use super::*;

    /// Create a test keyframe with given ID
    fn create_test_keyframe(id: u64, timestamp: f64) -> KeyFrame {
        let frame = Frame::new(id, timestamp, 640, 480);
        let features = FrameFeatures::new();
        KeyFrame::new(frame, features)
    }

    /// Create test descriptors with controlled values
    fn create_test_descriptors(count: usize, seed: u8) -> Descriptors {
        let size = ORB_DESCRIPTOR_SIZE;
        let mut data = vec![0u8; count * size];
        for i in 0..data.len() {
            data[i] = (seed as u8).wrapping_add(i as u8);
        }
        Descriptors {
            data,
            size,
            count,
        }
    }

    #[test]
    fn test_loop_detector_creation() {
        let detector = crate::loop_closing::LoopDetector::new();
        assert!(detector.min_loop_score() > 0.0);
    }

    #[test]
    fn test_loop_candidates_empty_map() {
        let detector = crate::loop_closing::LoopDetector::new();
        let map = Map::new();
        
        let result = detector.compute_loop_candidates(&map, 0, &Descriptors::new());
        assert!(result.is_empty());
    }

    #[test]
    fn test_loop_consistency_no_candidates() {
        let detector = crate::loop_closing::LoopDetector::new();
        let candidates = vec![];
        
        let result = detector.compute_loop_consistency(&candidates, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_loop_consistency_with_candidates() {
        let detector = crate::loop_closing::LoopDetector::new();
        // Current frame is 100, candidates are at 10, 20, 30
        // Time difference: 100 - 10 = 90, 100 - 20 = 80, 100 - 30 = 70
        let candidates = vec![
            crate::loop_closing::LoopCandidate {
                keyframe_id: 1,
                score: 0.8,
                timestamp: 10.0,
            },
            crate::loop_closing::LoopCandidate {
                keyframe_id: 2,
                score: 0.7,
                timestamp: 20.0,
            },
            crate::loop_closing::LoopCandidate {
                keyframe_id: 3,
                score: 0.6,
                timestamp: 30.0,
            },
        ];
        
        let result = detector.compute_loop_consistency(&candidates, 100);
        // All candidates should pass (no minimum distance requirement)
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_similarity_transform_identity() {
        let result = crate::loop_closing::compute_similarity_transform(
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], // R
            &[0.0, 0.0, 0.0], // t
            1.0, // scale
        );
        
        assert!(result.is_some());
        let sim3 = result.unwrap();
        assert!((sim3.scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_loop_closing_initialization() {
        let loop_closing = crate::loop_closing::LoopClosing::new();
        assert_eq!(loop_closing.last_loop_frame_id(), None);
    }

    #[test]
    fn test_loop_closing_process_no_keyframes() {
        let mut loop_closing = crate::loop_closing::LoopClosing::new();
        let map = Map::new();
        
        let result = loop_closing.process(&map, 0, &Descriptors::new());
        assert!(!result.loop_detected);
    }

    #[test]
    fn test_minimum_distance_filter() {
        let detector = crate::loop_closing::LoopDetector::new();
        detector.set_min_distance_between_frames(50);
        
        let candidates = vec![
            crate::loop_closing::LoopCandidate {
                keyframe_id: 1,
                score: 0.8,
                timestamp: 10.0,
            },
            crate::loop_closing::LoopCandidate {
                keyframe_id: 2,
                score: 0.7,
                timestamp: 90.0, // Only 10 frames away from current
            },
        ];
        
        let result = detector.compute_loop_consistency(&candidates, 100);
        // First candidate passes (100 - 10 = 90 >= 50)
        // Second candidate fails (100 - 90 = 10 < 50)
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].keyframe_id, 1);
    }
}
