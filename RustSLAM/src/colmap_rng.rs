//! COLMAP-compatible MT19937 and random sampling.
//!
//! Matches COLMAP `RandomSampler` / libc++ `std::uniform_int_distribution` behavior
//! used by two-view geometry and absolute-pose RANSAC.

/// COLMAP-compatible MT19937-32 generator.
#[derive(Debug, Clone)]
pub struct ColmapMt19937 {
    state: [u32; 624],
    index: usize,
}

impl ColmapMt19937 {
    pub fn new(seed: u64) -> Self {
        let mut rng = Self {
            state: [0; 624],
            index: 624,
        };
        rng.state[0] = seed as u32;
        for i in 1..624 {
            rng.state[i] = 1_812_433_253u32
                .wrapping_mul(rng.state[i - 1] ^ (rng.state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        rng
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    fn twist(&mut self) {
        const UPPER_MASK: u32 = 0x8000_0000;
        const LOWER_MASK: u32 = 0x7fff_ffff;
        const MATRIX_A: u32 = 0x9908_b0df;

        for i in 0..624 {
            let x = (self.state[i] & UPPER_MASK) | (self.state[(i + 1) % 624] & LOWER_MASK);
            let mut xa = x >> 1;
            if x & 1 != 0 {
                xa ^= MATRIX_A;
            }
            self.state[i] = self.state[(i + 397) % 624] ^ xa;
        }
        self.index = 0;
    }

    pub fn uniform_u32(&mut self, min: u32, max: u32) -> u32 {
        let range = max.wrapping_sub(min).wrapping_add(1);
        if range == 1 {
            return min;
        }
        let width = if range == 0 {
            u32::BITS
        } else {
            let floor_log2 = u32::BITS - range.leading_zeros() - 1;
            let is_power_of_two = range & ((u32::MAX) >> (u32::BITS - floor_log2)) == 0;
            floor_log2 + u32::from(!is_power_of_two)
        };
        loop {
            let sample = self.independent_bits(width);
            if range == 0 || sample < range {
                return sample.wrapping_add(min);
            }
        }
    }

    pub fn uniform_usize(&mut self, min: usize, max: usize) -> usize {
        if min >= max {
            return min;
        }
        self.uniform_u32(min as u32, max as u32) as usize
    }

    fn independent_bits(&mut self, width: u32) -> u32 {
        if width == 0 {
            return 0;
        }
        let mask = if width < u32::BITS {
            u32::MAX >> (u32::BITS - width)
        } else {
            u32::MAX
        };
        self.next_u32() & mask
    }
}

/// COLMAP `RandomSampler` with a persistent index pool and partial Fisher-Yates draws.
#[derive(Debug, Clone)]
pub struct ColmapRandomSampler {
    rng: ColmapMt19937,
    sample_indices: Vec<usize>,
}

impl ColmapRandomSampler {
    pub fn new(seed: u64, indices: &[usize]) -> Self {
        Self {
            rng: ColmapMt19937::new(seed),
            sample_indices: indices.to_vec(),
        }
    }

    pub fn sample(&mut self, k: usize) -> Vec<usize> {
        if k > self.sample_indices.len() {
            return Vec::new();
        }
        let last = self.sample_indices.len() - 1;
        for i in 0..k {
            let j = self.rng.uniform_u32(i as u32, last as u32) as usize;
            self.sample_indices.swap(i, j);
        }
        self.sample_indices[..k].to_vec()
    }
}

/// Draw `k` unique indices from `[0, n)` using COLMAP's per-iteration sampling shape.
pub fn sample_unique_indices(rng: &mut ColmapMt19937, n: usize, k: usize) -> Vec<usize> {
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let target = k.min(n);
    let mut indices: Vec<usize> = (0..n).collect();
    let last = n - 1;
    for i in 0..target {
        let j = rng.uniform_u32(i as u32, last as u32) as usize;
        indices.swap(i, j);
    }
    indices.truncate(target);
    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colmap_random_sampler_uses_stateful_partial_shuffle() {
        let mut sampler = ColmapRandomSampler::new(42, &[0, 1, 2, 3, 4, 5]);

        assert_eq!(sampler.sample(3), vec![3, 5, 4]);
        assert_eq!(sampler.sample(3), vec![4, 1, 3]);

        let mut sampler = ColmapRandomSampler::new(1, &[0, 1, 2, 3, 4, 5]);
        assert_eq!(sampler.sample(3), vec![5, 4, 2]);
        assert_eq!(sampler.sample(3), vec![5, 2, 0]);
    }

    #[test]
    fn colmap_mt19937_matches_reference_outputs() {
        let mut rng = ColmapMt19937::new(42);

        let outputs = [
            rng.next_u32(),
            rng.next_u32(),
            rng.next_u32(),
            rng.next_u32(),
            rng.next_u32(),
            rng.next_u32(),
        ];

        assert_eq!(
            outputs,
            [
                1_608_637_542,
                3_421_126_067,
                4_083_286_876,
                787_846_414,
                3_143_890_026,
                3_348_747_335,
            ]
        );
    }

    #[test]
    fn sample_unique_indices_uses_fresh_pool_per_call() {
        let mut rng = ColmapMt19937::new(42);
        assert_eq!(sample_unique_indices(&mut rng, 6, 3), vec![3, 5, 4]);
        assert_eq!(sample_unique_indices(&mut rng, 6, 3), vec![2, 5, 0]);

        let mut sampler = ColmapRandomSampler::new(42, &[0, 1, 2, 3, 4, 5]);
        assert_eq!(sampler.sample(3), vec![3, 5, 4]);
        assert_eq!(sampler.sample(3), vec![4, 1, 3]);
    }
}
