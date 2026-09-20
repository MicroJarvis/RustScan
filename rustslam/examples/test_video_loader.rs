//! Integration tests for VideoLoader
//!
//! Tests video frame handling and mock data generation.

use rustslam::test_utils::*;

fn test_mock_frame_creation() {
    // Test creating mock frames with different patterns
    let patterns = vec![
        FramePattern::Solid,
        FramePattern::Checkerboard,
        FramePattern::Gradient,
    ];

    for pattern in patterns {
        let frame = create_mock_frame(640, 480, pattern);

        // Verify frame properties
        assert_eq!(frame.width, 640, "Frame width should be 640");
        assert_eq!(frame.height, 480, "Frame height should be 480");
        assert_eq!(
            frame.rgb.len(),
            640 * 480 * 3,
            "RGB data should have correct size"
        );
        assert_eq!(frame.timestamp, 0.0, "Initial timestamp should be 0");
    }
}

fn test_mock_frame_patterns() {
    // Test solid pattern
    let solid = create_mock_frame(100, 100, FramePattern::Solid);
    // All pixels should be gray (128, 128, 128)
    for chunk in solid.rgb.chunks(3) {
        assert_eq!(chunk[0], 128, "Solid pattern R should be 128");
        assert_eq!(chunk[1], 128, "Solid pattern G should be 128");
        assert_eq!(chunk[2], 128, "Solid pattern B should be 128");
    }

    // Test checkerboard pattern
    let checkerboard = create_mock_frame(16, 16, FramePattern::Checkerboard);
    // First pixel should be white (block 0,0)
    assert_eq!(checkerboard.rgb[0], 255, "First pixel should be white");
    assert_eq!(checkerboard.rgb[1], 255);
    assert_eq!(checkerboard.rgb[2], 255);

    // Test gradient pattern
    let gradient = create_mock_frame(256, 100, FramePattern::Gradient);
    // First pixel should be black
    assert_eq!(gradient.rgb[0], 0, "First pixel should be black");
    // Last pixel in first row should be white
    let last_idx = (255 * 3) as usize;
    assert!(
        gradient.rgb[last_idx] > 250,
        "Last pixel should be nearly white"
    );
}

fn test_frame_dimensions() {
    // Test various resolutions
    let resolutions = vec![(640, 480), (1280, 720), (1920, 1080)];

    for (width, height) in resolutions {
        let frame = create_mock_frame(width, height, FramePattern::Solid);

        assert_eq!(frame.width, width);
        assert_eq!(frame.height, height);
        assert_eq!(frame.rgb.len(), (width * height * 3) as usize);
    }
}

fn main() {
    println!("Running VideoLoader integration tests...");

    test_mock_frame_creation();
    println!("✓ Mock frame creation test passed");

    test_mock_frame_patterns();
    println!("✓ Mock frame patterns test passed");

    test_frame_dimensions();
    println!("✓ Frame dimensions test passed");

    println!("\nAll VideoLoader integration tests passed!");
}
