use rustslam::{KeyPoint, Match};

#[derive(Debug, Clone)]
pub struct WideDescriptors {
    pub data: Vec<f32>,
    pub dim: usize,
    pub count: usize,
}

impl WideDescriptors {
    pub fn descriptor(&self, idx: usize) -> Option<&[f32]> {
        if idx >= self.count {
            return None;
        }
        let start = idx * self.dim;
        Some(&self.data[start..start + self.dim])
    }
}

pub fn rgb_to_gray(rgb: &[u8], width: u32, height: u32) -> Vec<f32> {
    let pixels = (width as usize).saturating_mul(height as usize);
    let mut gray = Vec::with_capacity(pixels);
    for px in rgb.chunks_exact(3).take(pixels) {
        gray.push(0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32);
    }
    gray
}

pub fn build_wide_descriptors(
    gray: &[f32],
    width: u32,
    height: u32,
    keypoints: &[KeyPoint],
) -> WideDescriptors {
    const CELLS: usize = 4;
    const BINS: usize = 8;
    const DIM: usize = CELLS * CELLS * BINS;
    let mut data = vec![0.0f32; keypoints.len() * DIM];
    let w = width as i32;
    let h = height as i32;

    for (kp_idx, kp) in keypoints.iter().enumerate() {
        let angle = if kp.angle.is_finite() { kp.angle } else { 0.0 };
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let radius = 8i32;
        let base = kp_idx * DIM;

        for dy in -radius..radius {
            for dx in -radius..radius {
                let rx = cos_a * dx as f32 - sin_a * dy as f32;
                let ry = sin_a * dx as f32 + cos_a * dy as f32;
                let x = kp.x() + rx;
                let y = kp.y() + ry;
                let ix = x.round() as i32;
                let iy = y.round() as i32;
                if ix <= 0 || iy <= 0 || ix >= w - 1 || iy >= h - 1 {
                    continue;
                }
                let gx = gray[(iy * w + ix + 1) as usize] - gray[(iy * w + ix - 1) as usize];
                let gy = gray[((iy + 1) * w + ix) as usize] - gray[((iy - 1) * w + ix) as usize];
                let mag = (gx * gx + gy * gy).sqrt();
                if mag <= 1.0e-6 {
                    continue;
                }
                let mut theta = gy.atan2(gx) - angle;
                while theta < 0.0 {
                    theta += std::f32::consts::TAU;
                }
                while theta >= std::f32::consts::TAU {
                    theta -= std::f32::consts::TAU;
                }
                let cx = ((dx + radius) as usize * CELLS / (2 * radius) as usize).min(CELLS - 1);
                let cy = ((dy + radius) as usize * CELLS / (2 * radius) as usize).min(CELLS - 1);
                let bin = ((theta / std::f32::consts::TAU) * BINS as f32).floor() as usize % BINS;
                let spatial = (-(dx * dx + dy * dy) as f32 / (2.0 * 6.0 * 6.0)).exp();
                data[base + (cy * CELLS + cx) * BINS + bin] += mag * spatial;
            }
        }

        let desc = &mut data[base..base + DIM];
        let norm = desc.iter().map(|v| v * v).sum::<f32>().sqrt().max(1.0e-12);
        for v in desc.iter_mut() {
            *v = (*v / norm).min(0.2);
        }
        let norm = desc.iter().map(|v| v * v).sum::<f32>().sqrt().max(1.0e-12);
        for v in desc.iter_mut() {
            *v = (*v / norm).sqrt();
        }
    }

    WideDescriptors {
        data,
        dim: DIM,
        count: keypoints.len(),
    }
}

pub fn match_wide_mutual(
    left: &WideDescriptors,
    right: &WideDescriptors,
    ratio: f32,
    max_distance: f32,
) -> Vec<Match> {
    let left_indices = (0..left.count).collect::<Vec<_>>();
    let right_indices = (0..right.count).collect::<Vec<_>>();
    match_wide_mutual_indices(
        left,
        right,
        &left_indices,
        &right_indices,
        ratio,
        max_distance,
    )
}

pub fn match_wide_mutual_indices(
    left: &WideDescriptors,
    right: &WideDescriptors,
    left_indices: &[usize],
    right_indices: &[usize],
    ratio: f32,
    max_distance: f32,
) -> Vec<Match> {
    let forward = match_wide_one_way_indices(
        left,
        right,
        left_indices,
        right_indices,
        ratio,
        max_distance,
    );
    let reverse = match_wide_one_way_indices(
        right,
        left,
        right_indices,
        left_indices,
        ratio,
        max_distance,
    );
    let reverse_pairs = reverse
        .into_iter()
        .map(|m| ((m.query_idx, m.train_idx), m.distance))
        .collect::<std::collections::HashMap<_, _>>();
    forward
        .into_iter()
        .filter(|m| reverse_pairs.contains_key(&(m.train_idx, m.query_idx)))
        .collect()
}

fn match_wide_one_way_indices(
    query: &WideDescriptors,
    train: &WideDescriptors,
    query_indices: &[usize],
    train_indices: &[usize],
    ratio: f32,
    max_distance: f32,
) -> Vec<Match> {
    if query.count == 0 || train.count == 0 || query.dim != train.dim {
        return Vec::new();
    }
    let mut matches = Vec::new();
    for &q_idx in query_indices {
        if q_idx >= query.count {
            continue;
        }
        let Some(q) = query.descriptor(q_idx) else {
            continue;
        };
        let mut best = (usize::MAX, f32::INFINITY);
        let mut second = f32::INFINITY;
        for &t_idx in train_indices {
            if t_idx >= train.count {
                continue;
            }
            let Some(t) = train.descriptor(t_idx) else {
                continue;
            };
            let dist = l2(q, t);
            if dist < best.1 {
                second = best.1;
                best = (t_idx, dist);
            } else if dist < second {
                second = dist;
            }
        }
        if best.0 != usize::MAX && best.1 <= max_distance && best.1 < ratio * second {
            matches.push(Match {
                query_idx: q_idx as u32,
                train_idx: best.0 as u32,
                distance: best.1,
            });
        }
    }
    matches
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}
