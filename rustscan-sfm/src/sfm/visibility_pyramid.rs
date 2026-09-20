//! COLMAP `VisibilityPyramid` for spatially distributed 3D point visibility scoring.

pub const NUM_POINT3D_VISIBILITY_PYRAMID_LEVELS: usize = 6;

#[derive(Debug, Clone)]
pub struct VisibilityPyramid {
    num_levels: usize,
    width: usize,
    height: usize,
    score: usize,
    max_score: usize,
    pyramid: Vec<u32>,
}

impl VisibilityPyramid {
    pub fn new(width: usize, height: usize) -> Self {
        Self::with_num_levels(NUM_POINT3D_VISIBILITY_PYRAMID_LEVELS, width, height)
    }

    pub fn with_num_levels(num_levels: usize, width: usize, height: usize) -> Self {
        let mut level_size = 0usize;
        let mut max_score = 0usize;
        for level in 0..num_levels {
            let dim = 1usize << (level + 1);
            level_size += dim * dim;
            max_score += dim * dim * dim * dim;
        }
        Self {
            num_levels,
            width,
            height,
            score: 0,
            max_score,
            pyramid: vec![0; level_size],
        }
    }

    pub fn num_levels(&self) -> usize {
        self.num_levels
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn score(&self) -> usize {
        self.score
    }

    pub fn max_score(&self) -> usize {
        self.max_score
    }

    pub fn set_point(&mut self, x: f32, y: f32) {
        if self.num_levels == 0 || self.width == 0 || self.height == 0 {
            return;
        }
        let (mut col, mut row) = self.cell_for_point(x, y);
        let mut level_idx = self.pyramid.len();
        for level in (0..self.num_levels).rev() {
            let dim = 1usize << (level + 1);
            level_idx -= dim * dim;
            let idx = level_idx + row * dim + col;
            if self.pyramid[idx] == 0 {
                self.score += dim * dim;
            }
            self.pyramid[idx] += 1;
            col >>= 1;
            row >>= 1;
        }
        debug_assert!(self.score <= self.max_score);
    }

    pub fn reset_point(&mut self, x: f32, y: f32) {
        if self.num_levels == 0 || self.width == 0 || self.height == 0 {
            return;
        }
        let (mut col, mut row) = self.cell_for_point(x, y);
        let mut level_idx = self.pyramid.len();
        for level in (0..self.num_levels).rev() {
            let dim = 1usize << (level + 1);
            level_idx -= dim * dim;
            let idx = level_idx + row * dim + col;
            debug_assert!(self.pyramid[idx] > 0, "visibility pyramid underflow");
            if self.pyramid[idx] == 1 {
                self.score = self.score.saturating_sub(dim * dim);
            }
            self.pyramid[idx] -= 1;
            col >>= 1;
            row >>= 1;
        }
        debug_assert!(self.score <= self.max_score);
    }

    fn cell_for_point(&self, x: f32, y: f32) -> (usize, usize) {
        let max_dim = 1usize << self.num_levels;
        (
            clamp_col_row(x, self.width, max_dim),
            clamp_col_row(y, self.height, max_dim),
        )
    }
}

impl Default for VisibilityPyramid {
    fn default() -> Self {
        Self::with_num_levels(0, 0, 0)
    }
}

impl PartialEq for VisibilityPyramid {
    fn eq(&self, other: &Self) -> bool {
        self.num_levels == other.num_levels
            && self.width == other.width
            && self.height == other.height
            && self.score == other.score
            && self.max_score == other.max_score
            && self.pyramid == other.pyramid
    }
}

fn clamp_col_row(coord: f32, extent: usize, dim: usize) -> usize {
    if extent == 0 || dim == 0 {
        return 0;
    }
    let scaled = (coord.max(0.0) / extent as f32) * dim as f32;
    (scaled.floor() as usize).min(dim - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_and_reset_point_updates_score() {
        let mut pyramid = VisibilityPyramid::new(100, 100);
        assert_eq!(pyramid.score(), 0);
        pyramid.set_point(10.0, 20.0);
        assert_eq!(pyramid.score(), 5460);
        let score_after_set = pyramid.score();
        pyramid.set_point(10.0, 20.0);
        assert_eq!(pyramid.score(), score_after_set);
        pyramid.reset_point(10.0, 20.0);
        assert_eq!(pyramid.score(), score_after_set);
        pyramid.reset_point(10.0, 20.0);
        assert_eq!(pyramid.score(), 0);
    }

    #[test]
    fn exposes_colmap_constructor_metadata_and_max_score() {
        let pyramid = VisibilityPyramid::with_num_levels(3, 640, 480);
        assert_eq!(pyramid.num_levels(), 3);
        assert_eq!(pyramid.width(), 640);
        assert_eq!(pyramid.height(), 480);
        assert_eq!(pyramid.max_score(), 16 + 256 + 4096);
        assert_eq!(pyramid.score(), 0);
    }

    #[test]
    fn batch_score_matches_incremental_updates() {
        let width = 100usize;
        let height = 100usize;
        let positions = [(10.0, 20.0), (80.0, 70.0), (80.0, 71.0)];
        let mut pyramid = VisibilityPyramid::new(width, height);
        for &(x, y) in &positions {
            pyramid.set_point(x, y);
        }
        let incremental = pyramid.score();
        let batch = batch_visibility_score(width, height, &positions);
        assert_eq!(incremental, batch);
    }

    fn batch_visibility_score(width: usize, height: usize, positions: &[(f32, f32)]) -> usize {
        let mut score = 0usize;
        for level in 0..NUM_POINT3D_VISIBILITY_PYRAMID_LEVELS {
            let dim = 1usize << (level + 1);
            let mut occupied = std::collections::HashSet::new();
            for &(x, y) in positions {
                let max_dim = 1usize << NUM_POINT3D_VISIBILITY_PYRAMID_LEVELS;
                let max_col = clamp_col_row(x, width, max_dim);
                let max_row = clamp_col_row(y, height, max_dim);
                let shift = NUM_POINT3D_VISIBILITY_PYRAMID_LEVELS - level - 1;
                let col = max_col >> shift;
                let row = max_row >> shift;
                occupied.insert((col, row));
            }
            score += occupied.len() * dim * dim;
        }
        score
    }
}
