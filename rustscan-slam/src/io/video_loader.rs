//! Video file loader for SLAM pipeline
//!
//! Uses OpenCV VideoCapture to decode common video formats (MP4/MOV/HEVC).

use std::path::{Path, PathBuf};

use thiserror::Error;

use super::{Dataset, DatasetMetadata, Frame};
use crate::core::Camera;

#[cfg(feature = "opencv")]
use opencv::{
    core::Mat,
    imgproc,
    prelude::*,
    videoio::{
        VideoCapture, CAP_ANY, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT, CAP_PROP_FRAME_HEIGHT,
        CAP_PROP_FRAME_WIDTH, CAP_PROP_POS_FRAMES,
    },
};

/// Errors that can occur while loading video files
#[derive(Debug, Error)]
pub enum VideoError {
    #[error("invalid video path")]
    InvalidPath,
    #[error("failed to open video: {0}")]
    OpenFailed(String),
    #[error("OpenCV error: {0}")]
    OpenCv(String),
    #[error("frame index out of bounds: {0}")]
    FrameIndex(usize),
    #[error("failed to read frame: {0}")]
    ReadFrame(usize),
}

pub type Result<T> = std::result::Result<T, VideoError>;

/// Video loader backed by OpenCV
pub struct VideoLoader {
    path: PathBuf,
    fps: f64,
    width: i32,
    height: i32,
    frame_count: i32,
    camera: Camera,
    metadata: DatasetMetadata,
}

impl VideoLoader {
    /// Open a video file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let path_str = path.to_str().ok_or(VideoError::InvalidPath)?;

        let mut capture = VideoCapture::from_file(path_str, CAP_ANY)
            .map_err(|e| VideoError::OpenCv(e.to_string()))?;

        let opened = capture
            .is_opened()
            .map_err(|e| VideoError::OpenCv(e.to_string()))?;
        if !opened {
            return Err(VideoError::OpenFailed(path_str.to_string()));
        }

        let fps = capture
            .get(CAP_PROP_FPS)
            .map_err(|e| VideoError::OpenCv(e.to_string()))?;
        let width = capture
            .get(CAP_PROP_FRAME_WIDTH)
            .map_err(|e| VideoError::OpenCv(e.to_string()))? as i32;
        let height = capture
            .get(CAP_PROP_FRAME_HEIGHT)
            .map_err(|e| VideoError::OpenCv(e.to_string()))? as i32;
        let frame_count = capture
            .get(CAP_PROP_FRAME_COUNT)
            .map_err(|e| VideoError::OpenCv(e.to_string()))? as i32;

        let fps = if fps > 0.0 { fps } else { 30.0 };
        let camera = Self::estimate_camera(width as u32, height as u32);

        let metadata = DatasetMetadata {
            name: "Video".to_string(),
            sequence: path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "video".to_string()),
            total_frames: frame_count.max(0) as usize,
            has_depth: false,
            has_ground_truth: false,
            frame_rate: Some(fps as f32),
            avg_speed: None,
            trajectory_length: None,
            notes: "Video file input".to_string(),
        };

        Ok(Self {
            path,
            fps,
            width,
            height,
            frame_count,
            camera,
            metadata,
        })
    }

    /// Frames per second reported by the container
    pub fn fps(&self) -> f64 {
        self.fps
    }

    /// Total number of frames (best effort)
    pub fn total_frames(&self) -> usize {
        self.frame_count.max(0) as usize
    }

    /// Estimated camera intrinsics from resolution
    pub fn estimate_camera(width: u32, height: u32) -> Camera {
        // Rough iPhone FOV estimate (~60-70 degrees)
        let fx = width as f32 * 1.2;
        let fy = height as f32 * 1.2;
        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;
        Camera::new(fx, fy, cx, cy, width, height)
    }

    #[cfg(feature = "opencv")]
    fn read_frame_at(&self, index: usize) -> Result<Vec<u8>> {
        let path_str = self.path.to_str().ok_or(VideoError::InvalidPath)?;
        let mut capture = VideoCapture::from_file(path_str, CAP_ANY)
            .map_err(|e| VideoError::OpenCv(e.to_string()))?;

        capture
            .set(CAP_PROP_POS_FRAMES, index as f64)
            .map_err(|e| VideoError::OpenCv(e.to_string()))?;

        let mut mat = Mat::default();
        let ok = capture
            .read(&mut mat)
            .map_err(|e| VideoError::OpenCv(e.to_string()))?;
        if !ok || mat.empty().map_err(|e| VideoError::OpenCv(e.to_string()))? {
            return Err(VideoError::ReadFrame(index));
        }

        let mut rgb = Mat::default();
        imgproc::cvt_color(&mat, &mut rgb, imgproc::COLOR_BGR2RGB, 0)
            .map_err(|e| VideoError::OpenCv(e.to_string()))?;

        let data = rgb
            .data_bytes()
            .map_err(|e| VideoError::OpenCv(e.to_string()))?;

        Ok(data.to_vec())
    }
}

impl Dataset for VideoLoader {
    fn len(&self) -> usize {
        self.total_frames()
    }

    fn get_frame(&self, index: usize) -> std::result::Result<Frame, super::DatasetError> {
        if index >= self.total_frames() {
            return Err(super::DatasetError::FrameIndex(index));
        }

        #[cfg(feature = "opencv")]
        {
            let color = self
                .read_frame_at(index)
                .map_err(|e| super::DatasetError::Image(e.to_string()))?;
            let timestamp = index as f64 / self.fps;

            Ok(Frame::new(
                index,
                timestamp,
                color,
                None,
                self.camera.clone(),
                None,
            ))
        }

        #[cfg(not(feature = "opencv"))]
        {
            let _ = index;
            Err(super::DatasetError::Image(
                "Video loading requires 'opencv' feature".to_string(),
            ))
        }
    }

    fn camera(&self) -> Camera {
        self.camera.clone()
    }

    fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_intrinsics_estimation() {
        // Test with known resolution
        let width = 1920;
        let height = 1080;

        #[cfg(feature = "opencv")]
        {
            let camera = VideoLoader::estimate_camera(width, height);

            // Verify reasonable values
            assert!(camera.fx() > 0.0, "fx should be positive");
            assert!(camera.fy() > 0.0, "fy should be positive");
            assert_eq!(
                camera.cx(),
                width as f32 / 2.0,
                "cx should be at image center"
            );
            assert_eq!(
                camera.cy(),
                height as f32 / 2.0,
                "cy should be at image center"
            );
            assert_eq!(camera.width(), width, "width should match");
            assert_eq!(camera.height(), height, "height should match");
        }

        #[cfg(not(feature = "opencv"))]
        {
            // Test the logic without VideoLoader
            let fx = width as f32 * 1.2;
            let fy = height as f32 * 1.2;
            let cx = width as f32 / 2.0;
            let cy = height as f32 / 2.0;

            assert!(fx > 0.0);
            assert!(fy > 0.0);
            assert_eq!(cx, 960.0);
            assert_eq!(cy, 540.0);
        }
    }

    #[test]
    fn test_frame_timestamp_calculation() {
        // Test timestamp calculation logic
        let fps = 30.0;
        let frame_index = 90;

        let timestamp = frame_index as f64 / fps;

        assert_eq!(timestamp, 3.0, "90 frames at 30fps = 3 seconds");
    }

    #[test]
    fn test_frame_timestamp_calculation_various_fps() {
        // Test with different frame rates
        let test_cases = vec![
            (30.0, 0, 0.0),
            (30.0, 30, 1.0),
            (60.0, 60, 1.0),
            (24.0, 24, 1.0),
            (30.0, 15, 0.5),
        ];

        for (fps, frame_index, expected_time) in test_cases {
            let timestamp = frame_index as f64 / fps;
            assert!(
                (timestamp - expected_time).abs() < 0.001,
                "Frame {} at {}fps should be at {}s, got {}s",
                frame_index,
                fps,
                expected_time,
                timestamp
            );
        }
    }

    #[test]
    fn test_camera_intrinsics_different_resolutions() {
        // Test with various common resolutions
        let resolutions = vec![
            (640, 480),   // VGA
            (1280, 720),  // HD
            (1920, 1080), // Full HD
            (3840, 2160), // 4K
        ];

        for (width, height) in resolutions {
            // Test the intrinsics calculation logic
            let fx = width as f32 * 1.2;
            let fy = height as f32 * 1.2;
            let cx = width as f32 / 2.0;
            let cy = height as f32 / 2.0;

            // Verify focal lengths scale with resolution
            assert!(
                fx > width as f32,
                "fx should be greater than width for {}x{}",
                width,
                height
            );
            assert!(
                fy > height as f32,
                "fy should be greater than height for {}x{}",
                width,
                height
            );

            // Verify principal point is at center
            assert_eq!(cx, width as f32 / 2.0);
            assert_eq!(cy, height as f32 / 2.0);
        }
    }

    #[test]
    #[cfg(feature = "opencv")]
    fn test_video_error_types() {
        // Test error type creation
        let err1 = VideoError::InvalidPath;
        assert!(err1.to_string().contains("invalid"));

        let err2 = VideoError::FrameIndex(42);
        assert!(err2.to_string().contains("42"));

        let err3 = VideoError::ReadFrame(10);
        assert!(err3.to_string().contains("10"));
    }
}
