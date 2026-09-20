//! Frame representation

use crate::core::SE3;

/// A frame in the SLAM system
#[derive(Debug, Clone)]
pub struct Frame {
    /// Unique frame ID
    pub id: u64,
    /// Timestamp in seconds
    pub timestamp: f64,
    /// Camera pose in world frame
    pub pose: Option<SE3>,
    /// Whether this is a keyframe
    pub is_keyframe: bool,
    /// Image width
    pub width: u32,
    /// Image height
    pub height: u32,
}

impl Frame {
    /// Create a new frame
    pub fn new(id: u64, timestamp: f64, width: u32, height: u32) -> Self {
        Self {
            id,
            timestamp,
            pose: None,
            is_keyframe: false,
            width,
            height,
        }
    }

    /// Set the camera pose
    pub fn set_pose(&mut self, pose: SE3) {
        self.pose = Some(pose);
    }

    /// Mark this frame as a keyframe
    pub fn mark_as_keyframe(&mut self) {
        self.is_keyframe = true;
    }

    /// Check if frame has a valid pose
    pub fn has_pose(&self) -> bool {
        self.pose.is_some()
    }

    /// Get the camera center in world coordinates
    ///
    /// For a camera pose T_wc (world from camera), the camera center
    /// in world coordinates is: C_w = -R_wc^T * t_wc
    pub fn camera_center(&self) -> Option<[f32; 3]> {
        self.pose.map(|p| {
            let t = p.translation();
            let r = p.rotation_matrix();
            // C = -R^T * t
            let mut center = [0.0f32; 3];
            center[0] = -(r[0][0] * t[0] + r[1][0] * t[1] + r[2][0] * t[2]);
            center[1] = -(r[0][1] * t[0] + r[1][1] * t[1] + r[2][1] * t[2]);
            center[2] = -(r[0][2] * t[0] + r[1][2] * t[1] + r[2][2] * t[2]);
            center
        })
    }
}

/// Feature data associated with a frame
#[derive(Debug, Clone)]
pub struct FrameFeatures {
    /// Keypoints in pixel coordinates [x, y]
    pub keypoints: Vec<[f32; 2]>,
    /// Descriptors (ORB, etc.) - raw bytes
    pub descriptors: Vec<u8>,
    /// 3D map points (if triangulated) - MapPoint IDs
    pub map_points: Vec<Option<u64>>,
}

impl FrameFeatures {
    /// Create empty features
    pub fn new() -> Self {
        Self {
            keypoints: Vec::new(),
            descriptors: Vec::new(),
            map_points: Vec::new(),
        }
    }

    /// Number of features
    pub fn len(&self) -> usize {
        self.keypoints.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.keypoints.is_empty()
    }
}

impl Default for FrameFeatures {
    fn default() -> Self {
        Self::new()
    }
}
