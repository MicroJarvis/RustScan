//! Pure Rust feature detection algorithms
//!
//! Implements Harris corner detection and FAST detector in pure Rust.

use crate::features::base::{
    Descriptors, FeatureError, FeatureExtractor, KeyPoint as BaseKeyPoint,
};
use crate::features::utils::{
    build_brief_descriptors, filter_by_response, select_keypoints_grid, to_grayscale,
};

/// A detected keypoint
#[derive(Debug, Clone, Copy)]
pub struct KeyPoint {
    /// x coordinate (pixel)
    pub x: f32,
    /// y coordinate (pixel)
    pub y: f32,
    /// Response (corner strength for Harris)
    pub response: f32,
    /// Scale (for multi-scale detection)
    pub scale: f32,
    /// Orientation in radians
    pub angle: f32,
}

impl KeyPoint {
    /// Create a new keypoint
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            response: 0.0,
            scale: 1.0,
            angle: 0.0,
        }
    }

    /// Create with response
    pub fn with_response(x: f32, y: f32, response: f32) -> Self {
        Self {
            x,
            y,
            response,
            scale: 1.0,
            angle: 0.0,
        }
    }
}

/// Harris corner detector parameters
#[derive(Debug, Clone, Copy)]
pub struct HarrisParams {
    /// Sobel aperture size (must be 3, 5, or 7)
    pub aperture_size: i32,
    /// Free parameter of Harris detector
    pub k: f32,
    /// Threshold for corner detection
    pub threshold: f32,
    /// Non-maximum suppression radius
    pub nms_radius: i32,
}

impl Default for HarrisParams {
    fn default() -> Self {
        Self {
            aperture_size: 3,
            k: 0.04,
            threshold: 0.01,
            nms_radius: 3,
        }
    }
}

/// Harris corner detector
pub struct HarrisDetector {
    params: HarrisParams,
}

impl HarrisDetector {
    /// Create a new Harris detector
    pub fn new(params: HarrisParams) -> Self {
        Self { params }
    }

    /// Create with default parameters
    pub fn default() -> Self {
        Self::new(HarrisParams::default())
    }

    /// Detect corners in a grayscale image
    ///
    /// Input image should be in row-major format (y fastest)
    pub fn detect(&self, image: &[u8], width: u32, height: u32) -> Vec<KeyPoint> {
        let w = width as usize;
        let h = height as usize;

        if image.len() != w * h || w < 3 || h < 3 {
            return Vec::new();
        }

        // Compute gradients
        let mut ix = vec![0i32; w * h];
        let mut iy = vec![0i32; w * h];

        self.compute_gradients(image, width, height, &mut ix, &mut iy);

        // Compute Ixx, Iyy, Ixy
        let mut ixx = vec![0i64; w * h];
        let mut iyy = vec![0i64; w * h];
        let mut ixy = vec![0i64; w * h];

        for i in 1..(h - 1) {
            for j in 1..(w - 1) {
                let idx = i * w + j;
                let gx = ix[idx] as i64;
                let gy = iy[idx] as i64;
                ixx[idx] = gx * gx;
                iyy[idx] = gy * gy;
                ixy[idx] = gx * gy;
            }
        }

        // Compute Harris response using Gaussian window
        let mut response = vec![0.0f32; w * h];
        let kernel_size = 5isize;
        let half_k = kernel_size / 2;
        let gaussian = Kernels::gaussian(kernel_size as usize, 1.2);

        for i in half_k..(h as isize - half_k) {
            for j in half_k..(w as isize - half_k) {
                let mut sum_ixx = 0.0f32;
                let mut sum_iyy = 0.0f32;
                let mut sum_ixy = 0.0f32;

                // Gaussian-weighted second moment matrix improves Harris stability.
                for di in -half_k..=half_k {
                    for dj in -half_k..=half_k {
                        let idx = ((i + di) * w as isize + (j + dj)) as usize;
                        let k_idx = ((di + half_k) * kernel_size + (dj + half_k)) as usize;
                        let weight = gaussian[k_idx];
                        sum_ixx += ixx[idx] as f32 * weight;
                        sum_iyy += iyy[idx] as f32 * weight;
                        sum_ixy += ixy[idx] as f32 * weight;
                    }
                }

                // Compute Harris response: det(M) - k * trace(M)^2
                let det = sum_ixx * sum_iyy - sum_ixy * sum_ixy;
                let trace = sum_ixx + sum_iyy;
                response[(i * w as isize + j) as usize] = det - self.params.k * trace * trace;
            }
        }

        // Non-maximum suppression and threshold
        self.nms_and_threshold(&response, width, height)
    }

    /// Compute image gradients using Sobel operator
    fn compute_gradients(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        ix: &mut [i32],
        iy: &mut [i32],
    ) {
        let w = width as usize;
        let h = height as usize;

        for i in 1..(h - 1) {
            for j in 1..(w - 1) {
                // Sobel X
                let gx = -1 * image[(i - 1) * w + (j - 1)] as i32
                    + -2 * image[i * w + (j - 1)] as i32
                    + -1 * image[(i + 1) * w + (j - 1)] as i32
                    + 1 * image[(i - 1) * w + (j + 1)] as i32
                    + 2 * image[i * w + (j + 1)] as i32
                    + 1 * image[(i + 1) * w + (j + 1)] as i32;

                // Sobel Y
                let gy = -1 * image[(i - 1) * w + (j - 1)] as i32
                    + -2 * image[(i - 1) * w + j] as i32
                    + -1 * image[(i - 1) * w + (j + 1)] as i32
                    + 1 * image[(i + 1) * w + (j - 1)] as i32
                    + 2 * image[(i + 1) * w + j] as i32
                    + 1 * image[(i + 1) * w + (j + 1)] as i32;

                ix[i * w + j] = gx;
                iy[i * w + j] = gy;
            }
        }
    }

    /// Non-maximum suppression and thresholding
    fn nms_and_threshold(&self, response: &[f32], width: u32, height: u32) -> Vec<KeyPoint> {
        let w = width as usize;
        let h = height as usize;
        let r = self.params.nms_radius as isize;
        let thresh = self.params.threshold;

        let mut corners = Vec::new();

        for i in (r as usize + 1)..(h - r as usize - 1) {
            for j in (r as usize + 1)..(w - r as usize - 1) {
                let val = response[i * w + j];

                if val > thresh {
                    // Check if it's a local maximum
                    let mut is_max = true;
                    for di in -r..=r {
                        for dj in -r..=r {
                            if di == 0 && dj == 0 {
                                continue;
                            }
                            let ni = (i as isize + di) as usize;
                            let nj = (j as isize + dj) as usize;
                            if response[ni * w + nj] > val {
                                is_max = false;
                                break;
                            }
                        }
                        if !is_max {
                            break;
                        }
                    }

                    if is_max {
                        corners.push(KeyPoint::with_response(j as f32, i as f32, val));
                    }
                }
            }
        }

        // Sort by response (strongest first)
        corners.sort_by(|a, b| {
            b.response
                .partial_cmp(&a.response)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        corners
    }
}

/// FAST corner detector parameters
#[derive(Debug, Clone, Copy)]
pub struct FastParams {
    /// Threshold for pixel intensity difference
    pub threshold: u8,
    /// Minimum consecutive pixels for corner
    pub min_consecutive: usize,
    /// Non-maximum suppression
    pub nms: bool,
}

impl Default for FastParams {
    fn default() -> Self {
        Self {
            threshold: 10,
            min_consecutive: 9,
            nms: true,
        }
    }
}

/// FAST (Features from Accelerated Segment Test) detector
pub struct FastDetector {
    params: FastParams,
}

impl FastDetector {
    /// Create a new FAST detector
    pub fn new(params: FastParams) -> Self {
        Self { params }
    }

    /// Create with default parameters
    pub fn default() -> Self {
        Self::new(FastParams::default())
    }

    /// Detect corners in a grayscale image
    pub fn detect(&self, image: &[u8], width: u32, height: u32) -> Vec<KeyPoint> {
        let w = width as usize;
        let h = height as usize;

        if image.len() != w * h || w < 7 || h < 7 {
            return Vec::new();
        }

        let mut corners = Vec::new();
        let t = self.params.threshold as i32;

        // FAST corner detection (simplified)
        // Check 16 pixels in a circle around the center pixel
        let circle_16 = [
            (0isize, -3isize),
            (1isize, -3isize),
            (2isize, -2isize),
            (3isize, -1isize),
            (3isize, 0isize),
            (3isize, 1isize),
            (2isize, 2isize),
            (1isize, 3isize),
            (0isize, 3isize),
            (-1isize, 3isize),
            (-2isize, 2isize),
            (-3isize, 1isize),
            (-3isize, 0isize),
            (-3isize, -1isize),
            (-2isize, -2isize),
            (-1isize, -3isize),
        ];

        for i in 3isize..(h as isize - 3) {
            for j in 3isize..(w as isize - 3) {
                let center = image[(i * w as isize + j) as usize] as i32;
                let mut ring = [0i32; 16];
                for (idx, (dx, dy)) in circle_16.iter().enumerate() {
                    ring[idx] = image[((i + dy) * w as isize + (j + dx)) as usize] as i32;
                }

                if let Some(score) = self.fast_score(&ring, center, t) {
                    corners.push(KeyPoint::with_response(j as f32, i as f32, score));
                }
            }
        }

        if !self.params.nms {
            return corners;
        }

        self.nonmax_suppression(corners, w, h)
    }

    fn fast_score(&self, ring: &[i32; 16], center: i32, threshold: i32) -> Option<f32> {
        let min_run = self.params.min_consecutive.max(1).min(16);
        let mut bright = [false; 32];
        let mut dark = [false; 32];
        let mut diff = [0i32; 32];

        for i in 0..32 {
            let val = ring[i % 16];
            bright[i] = val >= center + threshold;
            dark[i] = val <= center - threshold;
            diff[i] = (val - center).abs();
        }

        let has_run = |mask: &[bool; 32]| -> bool {
            let mut run = 0usize;
            for &is_on in mask {
                if is_on {
                    run += 1;
                    if run >= min_run {
                        return true;
                    }
                } else {
                    run = 0;
                }
            }
            false
        };

        if !has_run(&bright) && !has_run(&dark) {
            return None;
        }

        let mut max_diff = 0i32;
        for d in diff.iter().take(16) {
            max_diff = max_diff.max(*d);
        }
        Some(max_diff as f32)
    }

    fn nonmax_suppression(
        &self,
        corners: Vec<KeyPoint>,
        width: usize,
        height: usize,
    ) -> Vec<KeyPoint> {
        if corners.is_empty() {
            return corners;
        }

        let mut response = vec![0.0f32; width * height];
        for kp in &corners {
            let x = kp.x as usize;
            let y = kp.y as usize;
            if x < width && y < height {
                let idx = y * width + x;
                response[idx] = response[idx].max(kp.response);
            }
        }

        let mut filtered = Vec::new();
        for kp in corners {
            let x = kp.x as isize;
            let y = kp.y as isize;
            let val = kp.response;
            let mut is_max = true;

            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                        continue;
                    }
                    let nidx = ny as usize * width + nx as usize;
                    if response[nidx] > val {
                        is_max = false;
                        break;
                    }
                }
                if !is_max {
                    break;
                }
            }

            if is_max {
                filtered.push(kp);
            }
        }

        filtered
    }
}

/// Harris feature extractor wrapper implementing FeatureExtractor.
pub struct HarrisExtractor {
    detector: HarrisDetector,
    max_features: usize,
}

impl HarrisExtractor {
    pub fn new(max_features: usize, params: HarrisParams) -> Self {
        Self {
            detector: HarrisDetector::new(params),
            max_features,
        }
    }
}

impl FeatureExtractor for HarrisExtractor {
    fn detect_and_compute(
        &mut self,
        image: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(Vec<BaseKeyPoint>, Descriptors), FeatureError> {
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
    ) -> Result<Vec<BaseKeyPoint>, FeatureError> {
        let gray = to_grayscale(image, width, height);
        if gray.is_empty() {
            return Ok(Vec::new());
        }

        let keypoints: Vec<BaseKeyPoint> = self
            .detector
            .detect(&gray, width, height)
            .into_iter()
            .map(|kp| {
                let mut base = BaseKeyPoint::new(kp.x, kp.y);
                base.response = kp.response.abs();
                base
            })
            .collect();

        if keypoints.is_empty() {
            return Ok(keypoints);
        }

        let max_response = keypoints
            .iter()
            .map(|kp| kp.response)
            .fold(0.0f32, f32::max);
        let threshold = max_response * 0.01;
        let filtered = filter_by_response(keypoints, threshold);
        let selected = select_keypoints_grid(filtered, width, height, self.max_features, 4, 4);
        Ok(selected)
    }

    fn num_features(&self) -> usize {
        self.max_features
    }

    fn set_num_features(&mut self, num: usize) {
        self.max_features = num;
    }
}

/// FAST feature extractor wrapper implementing FeatureExtractor.
pub struct FastExtractor {
    detector: FastDetector,
    max_features: usize,
}

impl FastExtractor {
    pub fn new(max_features: usize, params: FastParams) -> Self {
        Self {
            detector: FastDetector::new(params),
            max_features,
        }
    }
}

impl FeatureExtractor for FastExtractor {
    fn detect_and_compute(
        &mut self,
        image: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(Vec<BaseKeyPoint>, Descriptors), FeatureError> {
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
    ) -> Result<Vec<BaseKeyPoint>, FeatureError> {
        let gray = to_grayscale(image, width, height);
        if gray.is_empty() {
            return Ok(Vec::new());
        }

        let keypoints: Vec<BaseKeyPoint> = self
            .detector
            .detect(&gray, width, height)
            .into_iter()
            .map(|kp| {
                let mut base = BaseKeyPoint::new(kp.x, kp.y);
                base.response = kp.response.abs();
                base
            })
            .collect();

        if keypoints.is_empty() {
            return Ok(keypoints);
        }

        let max_response = keypoints
            .iter()
            .map(|kp| kp.response)
            .fold(0.0f32, f32::max);
        let threshold = max_response * 0.01;
        let filtered = filter_by_response(keypoints, threshold);
        let selected = select_keypoints_grid(filtered, width, height, self.max_features, 4, 4);
        Ok(selected)
    }

    fn num_features(&self) -> usize {
        self.max_features
    }

    fn set_num_features(&mut self, num: usize) {
        self.max_features = num;
    }
}

/// Simple image convolution kernels
pub struct Kernels;

impl Kernels {
    /// Gaussian kernel (simplified)
    pub fn gaussian(size: usize, sigma: f32) -> Vec<f32> {
        let mut kernel = Vec::with_capacity(size * size);
        let half = size / 2;

        for dy in 0..size {
            for dx in 0..size {
                let x = dx as f32 - half as f32;
                let y = dy as f32 - half as f32;
                let val = (-(x * x + y * y) / (2.0 * sigma * sigma)).exp();
                kernel.push(val);
            }
        }

        // Normalize
        let sum: f32 = kernel.iter().sum();
        for v in &mut kernel {
            *v /= sum;
        }

        kernel
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
                let v = if (bx ^ by) == 0 { 40 } else { 210 };
                img[(y * width + x) as usize] = v;
            }
        }
        img
    }

    #[test]
    fn test_harris_params() {
        let params = HarrisParams::default();
        assert_eq!(params.aperture_size, 3);
        assert_eq!(params.nms_radius, 3);
    }

    #[test]
    fn test_keypoint_creation() {
        let kp = KeyPoint::new(100.0, 200.0);
        assert_eq!(kp.x, 100.0);
        assert_eq!(kp.y, 200.0);
    }

    #[test]
    fn test_gaussian_kernel() {
        let kernel = Kernels::gaussian(5, 1.0);
        assert_eq!(kernel.len(), 25);

        // Check sum is approximately 1
        let sum: f32 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_harris_detector() {
        let detector = HarrisDetector::default();

        // Create a simple test image with corners
        let mut image = vec![0u8; 100 * 100];
        // Add a corner-like pattern
        for i in 40..60 {
            for j in 40..60 {
                image[i * 100 + j] = 255;
            }
        }

        let corners = detector.detect(&image, 100, 100);
        // Should detect at least some corners
        assert!(corners.len() > 0);
    }

    #[test]
    fn test_fast_detector() {
        let detector = FastDetector::default();

        // Create a simple test image
        let image = vec![128u8; 100 * 100];

        let corners = detector.detect(&image, 100, 100);
        // On uniform image, should find no corners
        assert_eq!(corners.len(), 0);
    }

    #[test]
    fn test_fast_detector_wraparound_consecutive_arc() {
        let width = 15u32;
        let height = 15u32;
        let mut image = vec![100u8; (width * height) as usize];
        let cx = 7isize;
        let cy = 7isize;

        let circle_16 = [
            (0isize, -3isize),
            (1isize, -3isize),
            (2isize, -2isize),
            (3isize, -1isize),
            (3isize, 0isize),
            (3isize, 1isize),
            (2isize, 2isize),
            (1isize, 3isize),
            (0isize, 3isize),
            (-1isize, 3isize),
            (-2isize, 2isize),
            (-3isize, 1isize),
            (-3isize, 0isize),
            (-3isize, -1isize),
            (-2isize, -2isize),
            (-1isize, -3isize),
        ];

        // Construct a bright run that wraps around ring end -> start.
        let wrap_run = [14usize, 15, 0, 1, 2, 3, 4, 5, 6];
        for &idx in &wrap_run {
            let (dx, dy) = circle_16[idx];
            let x = (cx + dx) as usize;
            let y = (cy + dy) as usize;
            image[y * width as usize + x] = 150;
        }

        let detector = FastDetector::new(FastParams {
            threshold: 20,
            min_consecutive: 9,
            nms: false,
        });
        let corners = detector.detect(&image, width, height);

        assert!(corners
            .iter()
            .any(|kp| kp.x as i32 == cx as i32 && kp.y as i32 == cy as i32));
    }

    #[test]
    fn test_harris_extractor_descriptors() {
        let img = checkerboard(128, 128, 8);
        let mut extractor = HarrisExtractor::new(500, HarrisParams::default());
        let (kps, desc) = extractor.detect_and_compute(&img, 128, 128).unwrap();
        assert_eq!(desc.size, ORB_DESCRIPTOR_SIZE);
        assert_eq!(desc.count, kps.len());
        assert!(kps.len() <= 500);
    }

    #[test]
    fn test_fast_extractor_descriptors() {
        let img = checkerboard(128, 128, 8);
        let mut extractor = FastExtractor::new(500, FastParams::default());
        let (kps, desc) = extractor.detect_and_compute(&img, 128, 128).unwrap();
        assert_eq!(desc.size, ORB_DESCRIPTOR_SIZE);
        assert_eq!(desc.count, kps.len());
        assert!(kps.len() <= 500);
    }
}
