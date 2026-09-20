#![cfg(feature = "opencv")]

use rustslam::io::VideoLoader;

fn sample_video_path() -> std::path::PathBuf {
    // Use small sample video under repo test_data
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("..");
    path.push("test_data");
    path.push("video");
    path.push("sofa_sample_01.MOV");
    path
}

#[test]
#[cfg(feature = "opencv")]
fn open_video_and_read_metadata() {
    let path = sample_video_path();
    assert!(path.exists(), "sample video should exist at {:?}", path);

    let loader = VideoLoader::open(&path).expect("failed to open sample video");

    assert!(loader.fps() > 0.0, "fps should be positive");
    assert!(
        loader.total_frames() > 0,
        "video should have at least one frame"
    );
}

#[test]
#[cfg(feature = "opencv")]
fn decode_first_and_last_frames() {
    let path = sample_video_path();
    let loader = VideoLoader::open(&path).expect("failed to open sample video");

    let total = loader.total_frames();
    assert!(total > 0, "video must have frames");

    // First frame
    let first = loader.get_frame(0).expect("failed to get first frame");
    assert_eq!(first.index(), 0);
    assert_eq!(
        first.color().len(),
        (first.camera().width() * first.camera().height() * 3) as usize
    );

    // Last frame (best-effort)
    let last = loader
        .get_frame(total - 1)
        .expect("failed to get last frame");
    assert_eq!(last.index(), total - 1);

    // Timestamp should be monotonic and consistent with fps
    assert!(last.timestamp() >= first.timestamp());
}

#[test]
#[cfg(feature = "opencv")]
fn out_of_bounds_frame_index_fails() {
    let path = sample_video_path();
    let loader = VideoLoader::open(&path).expect("failed to open sample video");

    let total = loader.total_frames();
    let err = loader
        .get_frame(total)
        .err()
        .expect("expected error for out-of-bounds frame index");
    let msg = format!("{}", err);
    assert!(msg.contains("frame index"), "unexpected error: {}", msg);
}
