//! Utility helpers for feature extraction and descriptors.

use crate::features::base::{Descriptors, KeyPoint, ORB_DESCRIPTOR_SIZE};

/// Number of binary comparison tests (256 bits = 32 bytes).
const NUM_BRIEF_TESTS: usize = 256;

/// 256 pre-computed BRIEF point pairs `(x1, y1, x2, y2)` within a 31×31 patch (±15 pixels).
/// Generated deterministically via LCG so no `rand` crate is needed.
const BRIEF_PAIRS: [(i8, i8, i8, i8); NUM_BRIEF_TESTS] = generate_brief_pairs();

const fn generate_brief_pairs() -> [(i8, i8, i8, i8); NUM_BRIEF_TESTS] {
    let mut pairs = [(0i8, 0i8, 0i8, 0i8); NUM_BRIEF_TESTS];
    let mut state: u32 = 0x12345678; // fixed seed
    let mut i = 0;
    while i < NUM_BRIEF_TESTS {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let x1 = ((state >> 16) % 31) as i8 - 15;
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let y1 = ((state >> 16) % 31) as i8 - 15;
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let x2 = ((state >> 16) % 31) as i8 - 15;
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let y2 = ((state >> 16) % 31) as i8 - 15;
        pairs[i] = (x1, y1, x2, y2);
        i += 1;
    }
    pairs
}

/// Build true binary BRIEF descriptors (rBRIEF — rotation-aware).
///
/// For each keypoint, 256 point-pair intensity comparisons are performed and
/// packed into 32 bytes (256 bits). When the keypoint has a valid orientation
/// (`angle >= 0`), the pairs are rotated by that angle before sampling.
pub fn build_brief_descriptors(
    gray: &[u8],
    width: u32,
    height: u32,
    keypoints: &[KeyPoint],
) -> Descriptors {
    let mut descriptors = Descriptors::with_capacity(keypoints.len(), ORB_DESCRIPTOR_SIZE);
    if keypoints.is_empty() || gray.is_empty() {
        return descriptors;
    }

    let w = width as i32;
    let h = height as i32;

    for (kp_idx, kp) in keypoints.iter().enumerate() {
        let cx = kp.x().round() as i32;
        let cy = kp.y().round() as i32;
        let use_rotation = kp.angle.is_finite() && kp.angle >= 0.0;
        let (sin_a, cos_a) = if use_rotation {
            kp.angle.sin_cos()
        } else {
            (0.0f32, 1.0f32)
        };

        let base = kp_idx * ORB_DESCRIPTOR_SIZE;

        for (i, &(x1, y1, x2, y2)) in BRIEF_PAIRS.iter().enumerate() {
            let (rx1, ry1) = rotate_offset(x1 as f32, y1 as f32, sin_a, cos_a, use_rotation);
            let (rx2, ry2) = rotate_offset(x2 as f32, y2 as f32, sin_a, cos_a, use_rotation);

            let i1 = safe_pixel(gray, w, h, cx + rx1, cy + ry1);
            let i2 = safe_pixel(gray, w, h, cx + rx2, cy + ry2);

            if i1 > i2 {
                descriptors.data[base + i / 8] |= 1 << (i % 8);
            }
        }
    }

    descriptors
}

#[inline]
fn rotate_offset(x: f32, y: f32, sin_a: f32, cos_a: f32, rotate: bool) -> (i32, i32) {
    if rotate {
        (
            (x * cos_a - y * sin_a).round() as i32,
            (x * sin_a + y * cos_a).round() as i32,
        )
    } else {
        (x as i32, y as i32)
    }
}

#[inline]
fn safe_pixel(gray: &[u8], w: i32, h: i32, x: i32, y: i32) -> u8 {
    if x >= 0 && x < w && y >= 0 && y < h {
        gray[(y * w + x) as usize]
    } else {
        0
    }
}

pub fn filter_by_response(mut keypoints: Vec<KeyPoint>, min_response: f32) -> Vec<KeyPoint> {
    if min_response <= 0.0 {
        return keypoints;
    }

    keypoints.retain(|kp| kp.response >= min_response);
    keypoints
}

pub fn select_keypoints_grid(
    keypoints: Vec<KeyPoint>,
    width: u32,
    height: u32,
    max_features: usize,
    grid_rows: usize,
    grid_cols: usize,
) -> Vec<KeyPoint> {
    if keypoints.len() <= max_features || max_features == 0 {
        return keypoints;
    }

    let rows = grid_rows.max(1);
    let cols = grid_cols.max(1);
    let cell_w = (width.max(1) as f32) / cols as f32;
    let cell_h = (height.max(1) as f32) / rows as f32;
    let per_cell = (max_features + rows * cols - 1) / (rows * cols);

    let mut cells: Vec<Vec<KeyPoint>> = vec![Vec::new(); rows * cols];

    for kp in keypoints {
        let mut col = (kp.x() / cell_w).floor() as isize;
        let mut row = (kp.y() / cell_h).floor() as isize;
        if col < 0 {
            col = 0;
        }
        if row < 0 {
            row = 0;
        }
        if col as usize >= cols {
            col = (cols - 1) as isize;
        }
        if row as usize >= rows {
            row = (rows - 1) as isize;
        }
        cells[row as usize * cols + col as usize].push(kp);
    }

    let mut selected = Vec::with_capacity(max_features);
    for cell in &mut cells {
        cell.sort_by(|a, b| {
            b.response
                .partial_cmp(&a.response)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        selected.extend(cell.iter().take(per_cell).cloned());
    }

    if selected.len() > max_features {
        selected.sort_by(|a, b| {
            b.response
                .partial_cmp(&a.response)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        selected.truncate(max_features);
    }

    selected
}

pub fn to_grayscale(image: &[u8], width: u32, height: u32) -> Vec<u8> {
    let expected_gray = (width as usize) * (height as usize);
    let expected_rgb = expected_gray * 3;

    if image.len() == expected_gray {
        return image.to_vec();
    }

    if image.len() < expected_rgb {
        return Vec::new();
    }

    let mut gray = Vec::with_capacity(expected_gray);
    for chunk in image.chunks_exact(3) {
        let r = chunk[0] as f32;
        let g = chunk[1] as f32;
        let b = chunk[2] as f32;
        let luma = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
        gray.push(luma);
    }

    gray
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::base::ORB_DESCRIPTOR_SIZE;

    #[test]
    fn test_brief_descriptor_size_is_32_bytes() {
        let width = 64u32;
        let height = 64u32;
        let gray = vec![128u8; (width * height) as usize];
        let mut kp = KeyPoint::new(32.0, 32.0);
        kp.angle = 0.0;

        let desc = build_brief_descriptors(&gray, width, height, &[kp]);
        assert_eq!(desc.size, ORB_DESCRIPTOR_SIZE);
        assert_eq!(desc.count, 1);
        assert_eq!(desc.data.len(), ORB_DESCRIPTOR_SIZE);
    }

    #[test]
    fn test_brief_descriptors_are_binary() {
        // On a gradient image, descriptors should have a mix of 0 and 1 bits.
        let width = 64u32;
        let height = 64u32;
        let mut gray = vec![0u8; (width * height) as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                gray[y * width as usize + x] = (x * 4).min(255) as u8;
            }
        }

        let mut kp = KeyPoint::new(32.0, 32.0);
        kp.angle = 0.0;
        let desc = build_brief_descriptors(&gray, width, height, &[kp]);

        let ones: u32 = desc.data.iter().map(|b| b.count_ones()).sum();
        // On a gradient image, expect a healthy mix of bits (not all-zero or all-one).
        assert!(ones > 30, "expected >30 set bits, got {ones}");
        assert!(ones < 226, "expected <226 set bits, got {ones}");
    }

    #[test]
    fn test_brief_rotation_changes_descriptor() {
        let width = 64u32;
        let height = 64u32;
        let mut gray = vec![0u8; (width * height) as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                gray[y * width as usize + x] = (x * 4).min(255) as u8;
            }
        }

        let mut kp0 = KeyPoint::new(32.0, 32.0);
        kp0.angle = 0.0;
        let mut kp90 = KeyPoint::new(32.0, 32.0);
        kp90.angle = std::f32::consts::FRAC_PI_2;

        let d0 = build_brief_descriptors(&gray, width, height, &[kp0]);
        let d90 = build_brief_descriptors(&gray, width, height, &[kp90]);

        assert_eq!(d0.size, ORB_DESCRIPTOR_SIZE);
        assert_eq!(d90.size, ORB_DESCRIPTOR_SIZE);
        assert_ne!(d0.data, d90.data, "rotated descriptor should differ");
    }

    #[test]
    fn test_brief_uniform_image_all_zero() {
        // On a uniform image all comparisons yield I(p1) == I(p2), so no bit is set.
        let width = 64u32;
        let height = 64u32;
        let gray = vec![100u8; (width * height) as usize];
        let mut kp = KeyPoint::new(32.0, 32.0);
        kp.angle = 0.0;

        let desc = build_brief_descriptors(&gray, width, height, &[kp]);
        assert!(
            desc.data.iter().all(|&b| b == 0),
            "uniform image should give all-zero descriptor"
        );
    }

    #[test]
    fn test_brief_pairs_are_deterministic() {
        // Calling generate_brief_pairs twice must yield identical results.
        let a = generate_brief_pairs();
        let b = generate_brief_pairs();
        assert_eq!(a, b);
    }
}
