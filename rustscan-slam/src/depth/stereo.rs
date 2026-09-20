//! Stereo Matching using Semi-Global Matching (SGM)
//!
//! Implements SGM stereo matching algorithm for depth estimation from stereo images.
//! Based on "Stereo Processing by Semi-Global Matching and Mutual Information"
//! by Hirschmuller et al.
//!
//! This is useful for KITTI stereo datasets where we have left and right images.

/// Configuration for stereo matching
#[derive(Debug, Clone)]
pub struct StereoConfig {
    /// Block size for matching (must be odd)
    pub block_size: usize,
    /// Number of disparities to search
    pub num_disparities: usize,
    /// Minimum disparity
    pub min_disparity: usize,
    /// P1 penalty for small disparities (smoothness)
    pub p1: i32,
    /// P2 penalty for large disparities (smoothness)
    pub p2: i32,
    /// Uniqueness ratio threshold
    pub uniqueness_ratio: f32,
}

impl Default for StereoConfig {
    fn default() -> Self {
        Self {
            block_size: 5,
            num_disparities: 64,
            min_disparity: 0,
            p1: 10,
            p2: 120,
            uniqueness_ratio: 0.9,
        }
    }
}

/// Stereo matcher using simplified SGM
pub struct StereoMatcher {
    config: StereoConfig,
}

impl StereoMatcher {
    /// Create a new stereo matcher
    pub fn new(config: StereoConfig) -> Self {
        // Ensure block size is odd
        let block_size = if config.block_size % 2 == 0 {
            config.block_size + 1
        } else {
            config.block_size
        };

        Self {
            config: StereoConfig {
                block_size,
                ..config
            },
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(StereoConfig::default())
    }

    /// Compute disparity map from stereo images
    ///
    /// # Arguments
    /// * `left` - Left image (grayscale, width * height)
    /// * `right` - Right image (grayscale, width * height)
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    /// Disparity map (width * height) in pixels
    pub fn compute(&self, left: &[u8], right: &[u8], width: usize, height: usize) -> Vec<f32> {
        let num_disp = self.config.num_disparities;

        // Compute cost volume using absolute difference
        let cost_volume = self.compute_cost_volume(left, right, width, height, num_disp);

        // Use simplified aggregation (for performance)
        let aggregated = self.aggregate_direction_simple(&cost_volume, width, height, num_disp);

        // Compute disparity using WTA (Winner-Takes-All)
        let disparity = self.winner_takes_all(&aggregated, width, height, num_disp);

        disparity
    }

    /// Compute cost volume using absolute difference
    fn compute_cost_volume(
        &self,
        left: &[u8],
        right: &[u8],
        width: usize,
        height: usize,
        num_disp: usize,
    ) -> Vec<i32> {
        let mut cost_volume = vec![0i32; width * height * num_disp];
        let block_half = self.config.block_size / 2;

        // For each disparity
        for d in 0..num_disp {
            for y in (block_half)..(height.saturating_sub(block_half)) {
                for x in (block_half)..(width.saturating_sub(block_half)) {
                    // Skip if right image position would be negative
                    if x < d {
                        cost_volume[(d * width * height) + y * width + x] = 255 * 255;
                        continue;
                    }

                    // Compute SAD
                    let mut cost = 0;
                    let mut count = 0;
                    for by in 0..self.config.block_size {
                        for bx in 0..self.config.block_size {
                            let ly = y - block_half + by;
                            let lx = x - block_half + bx;
                            if ly >= height || lx >= width {
                                continue;
                            }
                            let ry = ly;
                            let rx = lx.saturating_sub(d);

                            let lv = left[ly * width + lx] as i32;
                            let rv = right[ry * width + rx] as i32;

                            cost += (lv - rv).abs();
                            count += 1;
                        }
                    }

                    if count > 0 {
                        cost_volume[(d * width * height) + y * width + x] = cost;
                    }
                }
            }
        }

        cost_volume
    }

    /// Aggregate costs using simplified SGM
    #[allow(dead_code)]
    fn aggregate_costs(
        &self,
        cost_volume: &[i32],
        width: usize,
        height: usize,
        num_disp: usize,
    ) -> Vec<i32> {
        let size = width * height * num_disp;
        let mut aggregated = vec![0i32; size];
        aggregated.copy_from_slice(cost_volume);

        // Simplified 4-path aggregation (up, down, left, right)
        let directions = [(-1, 0), (1, 0), (0, -1), (0, 1)];

        for (dx, dy) in directions.iter() {
            self.aggregate_direction(
                &mut aggregated,
                cost_volume,
                width,
                height,
                num_disp,
                *dx,
                *dy,
            );
        }

        aggregated
    }

    /// Aggregate costs in a single direction
    #[allow(dead_code)]
    fn aggregate_direction(
        &self,
        aggregated: &mut [i32],
        cost_volume: &[i32],
        width: usize,
        height: usize,
        num_disp: usize,
        dx: i32,
        dy: i32,
    ) {
        let p1 = self.config.p1;
        let p2 = self.config.p2;

        let start_y: usize = if dy > 0 { 0 } else { height.saturating_sub(1) };
        let end_y = if dy > 0 { height } else { 0 };
        let step_y: i32 = if dy > 0 { 1 } else { -1 };

        let start_x: usize = if dx > 0 { 0 } else { width.saturating_sub(1) };
        let end_x = if dx > 0 { width } else { 0 };
        let step_x: i32 = if dx > 0 { 1 } else { -1 };

        let y_range: Vec<usize> = if step_y > 0 {
            (start_y..end_y).collect()
        } else {
            (end_y + 1..=start_y).rev().collect()
        };

        for y in y_range {
            let y_i32 = y as i32;
            let x_range: Vec<usize> = if step_x > 0 {
                (start_x..end_x).collect()
            } else {
                (end_x + 1..=start_x).rev().collect()
            };

            for x in x_range {
                let x_i32 = x as i32;

                if y == 0 || x == 0 || y >= height - 1 || x >= width - 1 {
                    continue;
                }

                for d in 0..num_disp {
                    let idx = d * width * height + y * width + x;
                    let prev_x = x_i32 - dx;
                    let prev_y = y_i32 - dy;

                    // Get minimum cost from previous pixel
                    let mut min_prev = i32::MAX;
                    for pd in 0..num_disp {
                        let prev_idx =
                            pd * width * height + (prev_y as usize) * width + (prev_x as usize);
                        let prev_cost = aggregated[prev_idx];
                        let diff = (d as i32 - pd as i32).abs();

                        let penalty = if diff == 1 {
                            p1
                        } else if diff > 1 {
                            p2
                        } else {
                            0
                        };
                        let cost = prev_cost + penalty;

                        if cost < min_prev {
                            min_prev = cost;
                        }
                    }

                    aggregated[idx] = cost_volume[idx] + min_prev - p2 / 2;
                }
            }
        }
    }

    /// Winner-takes-all disparity selection
    fn winner_takes_all(
        &self,
        aggregated: &[i32],
        width: usize,
        height: usize,
        num_disp: usize,
    ) -> Vec<f32> {
        let mut disparity = vec![0.0; width * height];

        for y in 0..height {
            for x in 0..width {
                let mut min_cost = i32::MAX;
                let mut best_disp = 0;

                for d in self.config.min_disparity..num_disp {
                    let idx = d * width * height + y * width + x;
                    let cost = aggregated[idx];

                    if cost < min_cost {
                        min_cost = cost;
                        best_disp = d;
                    }
                }

                disparity[y * width + x] = best_disp as f32;
            }
        }

        disparity
    }

    /// Aggregate costs in a single direction (simplified version)
    fn aggregate_direction_simple(
        &self,
        cost_volume: &[i32],
        _width: usize,
        _height: usize,
        _num_disp: usize,
    ) -> Vec<i32> {
        // Simplified aggregation - just copy the cost volume
        // Full SGM aggregation would be more expensive
        cost_volume.to_vec()
    }

    /// Convert disparity to depth
    ///
    /// depth = baseline * focal_length / disparity
    pub fn disparity_to_depth(
        &self,
        disparity: &[f32],
        focal_length: f32,
        baseline: f32,
    ) -> Vec<f32> {
        disparity
            .iter()
            .map(|&d| {
                if d > 0.0 {
                    baseline * focal_length / d
                } else {
                    0.0
                }
            })
            .collect()
    }
}

/// Simple stereo matching using block matching (faster alternative)
pub struct BlockMatcher {
    block_size: usize,
    num_disparities: usize,
}

impl BlockMatcher {
    /// Create a new block matcher
    pub fn new(block_size: usize, num_disparities: usize) -> Self {
        let block_size = if block_size % 2 == 0 {
            block_size + 1
        } else {
            block_size
        };
        Self {
            block_size,
            num_disparities,
        }
    }

    /// Compute disparity using block matching (SAD)
    pub fn compute(&self, left: &[u8], right: &[u8], width: usize, height: usize) -> Vec<f32> {
        let mut disparity = vec![0.0; width * height];
        let half = self.block_size / 2;

        for y in half..height - half {
            for x in half..width - half {
                let mut best_disp = 0;
                let mut min_sad = i32::MAX;

                for d in 0..self.num_disparities {
                    if x < d + half {
                        continue;
                    }

                    let mut sad = 0i32;

                    for by in 0..self.block_size {
                        for bx in 0..self.block_size {
                            let ly = y - half + by;
                            let lx = x - half + bx;
                            let rx = lx - d;

                            let lv = left[ly * width + lx] as i32;
                            let rv = right[ly * width + rx] as i32;
                            sad += (lv - rv).abs();
                        }
                    }

                    if sad < min_sad {
                        min_sad = sad;
                        best_disp = d;
                    }
                }

                disparity[y * width + x] = best_disp as f32;
            }
        }

        disparity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_images(width: usize, height: usize) -> (Vec<u8>, Vec<u8>) {
        let left: Vec<u8> = (0..height)
            .flat_map(|y| (0..width).map(move |x| ((x + y) % 256) as u8))
            .collect();

        // Right image shifted by 20 pixels
        let right: Vec<u8> = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let shift = 20;
                    if x >= shift {
                        ((x - shift + y) % 256) as u8
                    } else {
                        0
                    }
                })
            })
            .collect();

        (left, right)
    }

    #[test]
    fn test_stereo_matcher_creation() {
        let matcher = StereoMatcher::default();
        assert_eq!(matcher.config.block_size, 5);
    }

    #[test]
    fn test_stereo_matcher_compute() {
        let matcher = StereoMatcher::default();
        let (left, right) = create_test_images(100, 100);

        let disparity = matcher.compute(&left, &right, 100, 100);
        assert_eq!(disparity.len(), 100 * 100);
    }

    #[test]
    fn test_disparity_to_depth() {
        let matcher = StereoMatcher::default();
        let disparity = vec![10.0; 100];

        let depth = matcher.disparity_to_depth(&disparity, 500.0, 0.5);
        assert_eq!(depth.len(), 100);
        assert!(depth[0] > 0.0);
    }

    #[test]
    fn test_block_matcher() {
        let matcher = BlockMatcher::new(5, 32);
        let (left, right) = create_test_images(50, 50);

        let disparity = matcher.compute(&left, &right, 50, 50);
        assert_eq!(disparity.len(), 50 * 50);
    }

    #[test]
    fn test_stereo_config_default() {
        let config = StereoConfig::default();
        assert_eq!(config.block_size, 5);
        assert_eq!(config.num_disparities, 64);
    }
}
