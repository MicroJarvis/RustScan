//! COLMAP `VisibilityPyramid` for spatially distributed 3D point visibility scoring.

pub const NUM_POINT3D_VISIBILITY_PYRAMID_LEVELS: usize = 6;

#[derive(Debug, Clone)]
pub struct VisibilityPyramid {
    width: usize,
    height: usize,
    score: usize,
    pyramid: Vec<u32>,
}

impl VisibilityPyramid {
    pub fn new(width: usize, height: usize) -> Self {
        let mut level_size = 0usize;
        for level in 0..NUM_POINT3D_VISIBILITY_PYRAMID_LEVELS {
            let dim = 1usize << level;
            level_size += dim * dim;
        }
        Self {
            width,
            height,
            score: 0,
            pyramid: vec![0; level_size],
        }
    }

    pub fn score(&self) -> usize {
        self.score
    }

    pub fn set_point(&mut self, x: f32, y: f32) {
        if self.width == 0 || self.height == 0 {
            return;
        }
        let mut level_idx = 0usize;
        for level in 0..NUM_POINT3D_VISIBILITY_PYRAMID_LEVELS {
            let dim = 1usize << level;
            let col = clamp_col_row(x, self.width, dim);
            let row = clamp_col_row(y, self.height, dim);
            let idx = level_idx + row * dim + col;
            if self.pyramid[idx] == 0 {
                self.score += 1;
            }
            self.pyramid[idx] += 1;
            level_idx += dim * dim;
        }
    }

    pub fn reset_point(&mut self, x: f32, y: f32) {
        if self.width == 0 || self.height == 0 {
            return;
        }
        let mut level_idx = 0usize;
        for level in 0..NUM_POINT3D_VISIBILITY_PYRAMID_LEVELS {
            let dim = 1usize << level;
            let col = clamp_col_row(x, self.width, dim);
            let row = clamp_col_row(y, self.height, dim);
            let idx = level_idx + row * dim + col;
            debug_assert!(self.pyramid[idx] > 0, "visibility pyramid underflow");
            if self.pyramid[idx] == 1 {
                self.score = self.score.saturating_sub(1);
            }
            self.pyramid[idx] -= 1;
            level_idx += dim * dim;
        }
    }
}

impl Default for VisibilityPyramid {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl PartialEq for VisibilityPyramid {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width
            && self.height == other.height
            && self.score == other.score
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
        assert!(pyramid.score() > 0);
        let score_after_set = pyramid.score();
        pyramid.set_point(10.0, 20.0);
        assert_eq!(pyramid.score(), score_after_set);
        pyramid.reset_point(10.0, 20.0);
        assert_eq!(pyramid.score(), score_after_set);
        pyramid.reset_point(10.0, 20.0);
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
            let dim = 1usize << level;
            let mut occupied = std::collections::HashSet::new();
            for &(x, y) in positions {
                let col = clamp_col_row(x, width, dim);
                let row = clamp_col_row(y, height, dim);
                occupied.insert((col, row));
            }
            score += occupied.len();
        }
        score
    }
}
