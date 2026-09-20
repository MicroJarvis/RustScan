#![cfg(feature = "opencv")]

use rustslam::io::VideoLoader;
use rustslam::pipeline::realtime::RealtimePipeline;

fn sample_video_path() -> std::path::PathBuf {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("..");
    path.push("test_data");
    path.push("video");
    path.push("sofa_sample_01.MOV");
    path
}

#[test]
#[cfg(feature = "opencv")]
fn run_realtime_pipeline_on_sample_video() {
    let path = sample_video_path();
    assert!(path.exists(), "sample video should exist at {:?}", path);

    // Open loader directly to inspect metadata
    let loader = VideoLoader::open(&path).expect("failed to open video");
    assert!(loader.total_frames() > 0);

    // Create realtime pipeline from video path (uses VideoLoader internally)
    let mut pipeline = RealtimePipeline::from_video_path(&path)
        .expect("failed to create realtime pipeline from video");

    // Process a few frames
    let max_frames = usize::min(5, loader.total_frames());
    for _ in 0..max_frames {
        if !pipeline.step().expect("pipeline step failed") {
            break;
        }
    }

    // Basic sanity: pipeline should have processed at least one frame
    assert!(pipeline.state().processed_frames > 0);
}
