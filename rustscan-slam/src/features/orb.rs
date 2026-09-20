//! ORB feature extractor
//!
//! This module provides the ORB feature extractor interface.
//! The actual OpenCV implementation requires opencv-rust with system OpenCV.
//! For now, this provides a placeholder that can be replaced with actual implementation.

use crate::features::base::{Descriptors, FeatureError, FeatureExtractor, KeyPoint};
use crate::features::pure_rust::{FastDetector, FastParams, HarrisDetector, HarrisParams};
use crate::features::utils::{
    build_brief_descriptors, filter_by_response, select_keypoints_grid, to_grayscale,
};

/// ORB (Oriented FAST and Rotated BRIEF) feature extractor
///
/// This is the interface for ORB feature extraction.
///
/// To enable OpenCV support, ensure OpenCV 4.x is installed and
/// compile with the `opencv` feature:
///
/// ```toml
/// rustrslam = { version = "0.1", features = ["opencv"] }
/// ```
pub struct OrbExtractor {
    /// Number of features to extract
    num_features: usize,
    /// Scale factor between pyramid levels
    scale_factor: f64,
    /// Number of pyramid levels
    num_levels: i32,
    /// Edge threshold
    edge_threshold: i32,
    /// First level
    first_level: i32,
    /// WTA_K (points for BRIEF)
    wta_k: i32,
    /// Patch size
    patch_size: i32,
}

impl OrbExtractor {
    /// Create a new ORB extractor with default parameters
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            scale_factor: 1.2,
            num_levels: 8,
            edge_threshold: 31,
            first_level: 0,
            wta_k: 2, // 2 points for each BRIEF comparison
            patch_size: 31,
        }
    }

    /// Create with custom parameters
    pub fn with_params(num_features: usize, scale_factor: f64, num_levels: i32) -> Self {
        Self {
            num_features,
            scale_factor,
            num_levels,
            edge_threshold: 31,
            first_level: 0,
            wta_k: 2,
            patch_size: 31,
        }
    }

    /// Full configuration
    pub fn configure(
        num_features: usize,
        scale_factor: f64,
        num_levels: i32,
        edge_threshold: i32,
        first_level: i32,
        wta_k: i32,
        patch_size: i32,
    ) -> Self {
        Self {
            num_features,
            scale_factor,
            num_levels,
            edge_threshold,
            first_level,
            wta_k,
            patch_size,
        }
    }

    /// Get parameters as tuple for reference
    pub fn params(
        &self,
    ) -> (
        usize, // num_features
        f64,   // scale_factor
        i32,   // num_levels
        i32,   // edge_threshold
        i32,   // first_level
        i32,   // wta_k
        i32,   // patch_size
    ) {
        (
            self.num_features,
            self.scale_factor,
            self.num_levels,
            self.edge_threshold,
            self.first_level,
            self.wta_k,
            self.patch_size,
        )
    }
}

impl Default for OrbExtractor {
    fn default() -> Self {
        Self::new(2000)
    }
}

impl FeatureExtractor for OrbExtractor {
    fn detect_and_compute(
        &mut self,
        image: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(Vec<KeyPoint>, Descriptors), FeatureError> {
        #[cfg(feature = "opencv")]
        {
            return self.detect_and_compute_opencv(image, width as i32, height as i32);
        }

        let keypoints = self.detect(image, width, height)?;
        let gray = to_grayscale(image, width, height);
        let descriptors = build_brief_descriptors(&gray, width, height, &keypoints);
        Ok((keypoints, descriptors))
    }

    fn detect(
        &mut self,
        image: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<KeyPoint>, FeatureError> {
        #[cfg(feature = "opencv")]
        {
            let (kps, _desc) =
                self.detect_and_compute_opencv(image, width as i32, height as i32)?;
            return Ok(kps);
        }

        let gray = to_grayscale(image, width, height);
        if gray.is_empty() {
            return Ok(Vec::new());
        }

        let detector = FastDetector::new(FastParams::default());
        let mut keypoints: Vec<KeyPoint> = detector
            .detect(&gray, width, height)
            .into_iter()
            .map(|kp| {
                let mut base = KeyPoint::new(kp.x, kp.y);
                base.response = kp.response.abs();
                base
            })
            .collect();

        if keypoints.is_empty() {
            let harris = HarrisDetector::new(HarrisParams::default());
            keypoints = harris
                .detect(&gray, width, height)
                .into_iter()
                .map(|kp| {
                    let mut base = KeyPoint::new(kp.x, kp.y);
                    base.response = kp.response.abs();
                    base
                })
                .collect();
        }

        if keypoints.is_empty() {
            return Ok(keypoints);
        }

        let max_response = keypoints
            .iter()
            .map(|kp| kp.response)
            .fold(0.0f32, f32::max);
        let threshold = max_response * 0.01;
        let filtered = filter_by_response(keypoints, threshold);
        let mut selected = select_keypoints_grid(filtered, width, height, self.num_features, 4, 4);
        let half_patch = (self.patch_size.max(3) / 2) as i32;
        for kp in &mut selected {
            kp.angle =
                compute_intensity_centroid_angle(&gray, width, height, kp.x(), kp.y(), half_patch);
            kp.size = self.patch_size as f32;
        }

        Ok(selected)
    }

    fn num_features(&self) -> usize {
        self.num_features
    }

    fn set_num_features(&mut self, num: usize) {
        self.num_features = num;
    }
}

fn compute_intensity_centroid_angle(
    gray: &[u8],
    width: u32,
    height: u32,
    x: f32,
    y: f32,
    half_patch: i32,
) -> f32 {
    let cx = x.round() as i32;
    let cy = y.round() as i32;
    let w = width as i32;
    let h = height as i32;

    let mut m10 = 0.0f32;
    let mut m01 = 0.0f32;

    for dy in -half_patch..=half_patch {
        for dx in -half_patch..=half_patch {
            if dx * dx + dy * dy > half_patch * half_patch {
                continue;
            }
            let px = cx + dx;
            let py = cy + dy;
            if px < 0 || py < 0 || px >= w || py >= h {
                continue;
            }
            let idx = (py * w + px) as usize;
            let intensity = gray.get(idx).copied().unwrap_or(0) as f32;
            m10 += dx as f32 * intensity;
            m01 += dy as f32 * intensity;
        }
    }

    if m10.abs() < 1e-6 && m01.abs() < 1e-6 {
        0.0
    } else {
        m01.atan2(m10)
    }
}

// OpenCV implementation - enabled when opencv feature is available
#[cfg(feature = "opencv")]
mod opencv_impl {
    use super::*;
    use opencv::core::{no_array, Mat, Vector};
    use opencv::features::ORB;

    impl OrbExtractor {
        /// Detect and compute with OpenCV
        pub fn detect_and_compute_opencv(
            &mut self,
            image: &[u8],
            width: i32,
            height: i32,
        ) -> Result<(Vec<KeyPoint>, Descriptors), FeatureError> {
            // Create Mat from image buffer (grayscale)
            let mat = unsafe {
                Mat::new_rows_cols_with_data(
                    height,
                    width,
                    opencv::core::CV_8UC1,
                    image.as_ptr() as *mut std::ffi::c_void,
                    opencv::core::Mat_AUTO_STEP,
                )
            }
            .map_err(|e| FeatureError::OpenCV(e.to_string()))?;

            // Create ORB detector
            let mut orb = ORB::create(
                self.num_features as i32,
                self.scale_factor,
                self.num_levels,
                self.edge_threshold,
                self.first_level,
                self.patch_size,
                self.wta_k,
                0, // score_type (0 = HARRIS, 1 = FAST)
            )
            .map_err(|e| FeatureError::OpenCV(e.to_string()))?;

            // Detect keypoints and compute descriptors
            let mut keypoints = Vector::<opencv::core::KeyPoint>::new();
            let mut descriptors = Mat::new().map_err(|e| FeatureError::OpenCV(e.to_string()))?;

            orb.detect_and_compute(&mat, &no_array(), &mut keypoints, &mut descriptors)
                .map_err(|e| FeatureError::OpenCV(e.to_string()))?;

            // Convert OpenCV keypoints to our format
            let mut kps = Vec::with_capacity(keypoints.len());
            for kp in keypoints.iter() {
                let pt = kp.pt();
                kps.push(KeyPoint {
                    pt: (pt.x, pt.y),
                    size: kp.size() as f32,
                    angle: kp.angle() as f32,
                    response: kp.response() as f32,
                    octave: kp.octave(),
                });
            }

            // Convert descriptors
            let mut des_data = Vec::new();
            let descriptor_size = if !descriptors.empty() {
                descriptors.elem_size() as usize
            } else {
                crate::features::base::ORB_DESCRIPTOR_SIZE
            };

            if !descriptors.data().is_empty() {
                let rows = descriptors.rows();
                if rows > 0 {
                    for i in 0..rows {
                        let row = descriptors
                            .row(i)
                            .map_err(|e| FeatureError::OpenCV(e.to_string()))?;
                        let slice = row
                            .data()
                            .map_err(|e| FeatureError::OpenCV(e.to_string()))?;
                        let len = row.elem_size() as usize * row.cols();
                        des_data
                            .extend_from_slice(unsafe { std::slice::from_raw_parts(slice, len) });
                    }
                }
            }

            let des = Descriptors {
                data: des_data,
                size: descriptor_size,
                count: kps.len(),
            };

            Ok((kps, des))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::base::ORB_DESCRIPTOR_SIZE;

    fn checkerboard(width: u32, height: u32, block: u32) -> Vec<u8> {
        let mut img = vec![0u8; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let bx = (x / block) % 2;
                let by = (y / block) % 2;
                let v = if (bx ^ by) == 0 { 30 } else { 220 };
                img[(y * width + x) as usize] = v;
            }
        }
        img
    }

    #[test]
    fn test_orb_creation() {
        let orb = OrbExtractor::new(1000);
        assert_eq!(orb.num_features(), 1000);
    }

    #[test]
    fn test_orb_params() {
        let orb = OrbExtractor::with_params(500, 1.5, 4);
        let (num, scale, levels, _, _, _, _) = orb.params();
        assert_eq!(num, 500);
        assert!((scale - 1.5).abs() < 1e-6);
        assert_eq!(levels, 4);
    }

    #[test]
    fn test_orb_default() {
        let orb = OrbExtractor::default();
        assert_eq!(orb.num_features(), 2000);
    }

    #[test]
    fn test_orb_configure() {
        let orb = OrbExtractor::configure(
            3000, // num_features
            1.5,  // scale_factor
            6,    // num_levels
            20,   // edge_threshold
            1,    // first_level
            3,    // wta_k
            25,   // patch_size
        );
        let (num, scale, levels, edge, first, wta, patch) = orb.params();
        assert_eq!(num, 3000);
        assert!((scale - 1.5).abs() < 1e-6);
        assert_eq!(levels, 6);
        assert_eq!(edge, 20);
        assert_eq!(first, 1);
        assert_eq!(wta, 3);
        assert_eq!(patch, 25);
    }

    #[test]
    #[cfg(not(feature = "opencv"))]
    fn test_orb_fallback_extracts_and_distributes_keypoints() {
        let img = checkerboard(128, 128, 8);
        let mut extractor = OrbExtractor::new(2000);
        let (kps, desc) = extractor.detect_and_compute(&img, 128, 128).unwrap();
        assert!(!kps.is_empty());
        assert_eq!(desc.count, kps.len());
        assert_eq!(desc.size, ORB_DESCRIPTOR_SIZE);
        assert!(kps.len() <= 2000);

        let mut cells = std::collections::HashSet::new();
        let cell_w = 128.0 / 4.0;
        let cell_h = 128.0 / 4.0;
        for kp in &kps {
            let col = (kp.x() / cell_w).floor() as i32;
            let row = (kp.y() / cell_h).floor() as i32;
            cells.insert((row, col));
        }
        assert!(
            cells.len() >= 4,
            "expected keypoints in multiple grid cells"
        );
    }

    #[test]
    fn test_intensity_centroid_angle_points_to_bright_side() {
        let width = 31u32;
        let height = 31u32;
        let mut img = vec![20u8; (width * height) as usize];

        // Right half brighter than left half -> dominant moment on +x.
        for y in 0..height as usize {
            for x in (width as usize / 2)..width as usize {
                img[y * width as usize + x] = 220;
            }
        }

        let angle = compute_intensity_centroid_angle(&img, width, height, 15.0, 15.0, 15);
        assert!(angle.is_finite());
        assert!(
            angle.abs() < 0.6,
            "angle should roughly point to +x, got {angle}"
        );
    }
}
