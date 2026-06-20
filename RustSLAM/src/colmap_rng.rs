//! COLMAP-compatible MT19937 and random sampling.
//!
//! Matches COLMAP `RandomSampler` / `CombinationSampler` /
//! `ProgressiveSampler` behavior used by RANSAC-based estimators.

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

/// COLMAP `math::NChooseK`.
///
/// Mirrors COLMAP's edge cases: `n_choose_k(0, 0) == 0`, and `n < k` returns
/// zero. Arithmetic intentionally uses `u64` like COLMAP's implementation.
pub fn n_choose_k(mut n: u64, k: u64) -> u64 {
    if n == 0 || n < k {
        return 0;
    }
    let mut r = 1u64;
    for d in 1..=k {
        r *= n;
        r /= d;
        n -= 1;
    }
    r
}

/// COLMAP `CombinationSampler`, which enumerates unique sorted combinations.
#[derive(Debug, Clone)]
pub struct ColmapCombinationSampler {
    num_samples: usize,
    total_sample_indices: Vec<usize>,
    initialized: bool,
}

impl ColmapCombinationSampler {
    pub fn new(num_samples: usize) -> Self {
        Self {
            num_samples,
            total_sample_indices: Vec::new(),
            initialized: false,
        }
    }

    pub fn initialize(&mut self, total_num_samples: usize) -> bool {
        if self.num_samples > total_num_samples {
            self.total_sample_indices.clear();
            self.initialized = false;
            return false;
        }
        self.total_sample_indices = (0..total_num_samples).collect();
        self.initialized = true;
        true
    }

    pub fn max_num_samples(&self) -> u64 {
        n_choose_k(
            self.total_sample_indices.len() as u64,
            self.num_samples as u64,
        )
    }

    pub fn sample(&mut self) -> Vec<usize> {
        if !self.initialized || self.num_samples > self.total_sample_indices.len() {
            return Vec::new();
        }
        let sample = self.total_sample_indices[..self.num_samples].to_vec();
        if !next_combination(&mut self.total_sample_indices, self.num_samples) {
            self.total_sample_indices
                .iter_mut()
                .enumerate()
                .for_each(|(idx, value)| *value = idx);
        }
        sample
    }
}

fn next_combination(values: &mut [usize], middle: usize) -> bool {
    if middle == 0 || middle >= values.len() {
        return false;
    }
    let n = values.len();
    let mut i = middle - 1;
    while values[i] == i + n - middle {
        if i == 0 {
            return false;
        }
        i -= 1;
    }
    values[i] += 1;
    for j in (i + 1)..middle {
        values[j] = values[j - 1] + 1;
    }
    true
}

/// COLMAP `ProgressiveSampler` (PROSAC).
///
/// The returned values are raw sample indices, just like COLMAP's
/// `Sampler::Sample`. Callers are expected to apply these indices to
/// quality-sorted data.
#[derive(Debug, Clone)]
pub struct ColmapProgressiveSampler {
    rng: ColmapMt19937,
    num_samples: usize,
    total_num_samples: usize,
    initialized: bool,
    t: usize,
    n: usize,
    t_n: f64,
    t_n_p: f64,
}

impl ColmapProgressiveSampler {
    pub fn new(seed: u64, num_samples: usize) -> Self {
        Self {
            rng: ColmapMt19937::new(seed),
            num_samples,
            total_num_samples: 0,
            initialized: false,
            t: 0,
            n: 0,
            t_n: 0.0,
            t_n_p: 0.0,
        }
    }

    pub fn initialize(&mut self, total_num_samples: usize) -> bool {
        if self.num_samples > total_num_samples {
            self.total_num_samples = 0;
            self.initialized = false;
            return false;
        }

        self.total_num_samples = total_num_samples;
        self.initialized = true;
        self.t = 0;
        self.n = self.num_samples;

        // COLMAP uses the PROSAC paper's recommended progressive iteration
        // count before the sampler behaves like ordinary RANSAC.
        const NUM_PROGRESSIVE_ITERATIONS: f64 = 200_000.0;
        self.t_n = NUM_PROGRESSIVE_ITERATIONS;
        self.t_n_p = 1.0;
        for i in 0..self.num_samples {
            self.t_n *=
                (self.num_samples - i) as f64 / (self.total_num_samples - i) as f64;
        }
        true
    }

    pub fn max_num_samples(&self) -> usize {
        usize::MAX
    }

    pub fn sample(&mut self) -> Vec<usize> {
        if !self.initialized || self.num_samples > self.total_num_samples {
            return Vec::new();
        }

        self.t += 1;

        if self.t as f64 == self.t_n_p && self.n < self.total_num_samples {
            let t_n_plus_1 = self.t_n * (self.n as f64 + 1.0)
                / (self.n as f64 + 1.0 - self.num_samples as f64);
            self.t_n_p += (t_n_plus_1 - self.t_n).ceil();
            self.t_n = t_n_plus_1;
            self.n += 1;
        }

        let mut num_random_samples = self.num_samples;
        let mut max_random_sample_idx = self.n - 1;
        if self.t_n_p >= self.t as f64 {
            num_random_samples -= 1;
            max_random_sample_idx = max_random_sample_idx.wrapping_sub(1);
        }

        let mut sampled_idxs = Vec::with_capacity(self.num_samples);
        for _ in 0..num_random_samples {
            loop {
                let random_idx = self.rng.uniform_u32(0, max_random_sample_idx as u32) as usize;
                if !sampled_idxs.contains(&random_idx) {
                    sampled_idxs.push(random_idx);
                    break;
                }
            }
        }

        if self.t_n_p >= self.t as f64 {
            sampled_idxs.push(self.n);
        }

        sampled_idxs
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

    #[test]
    fn n_choose_k_matches_colmap_math_helper() {
        assert_eq!(n_choose_k(0, 0), 0);
        assert_eq!(n_choose_k(1, 0), 1);
        assert_eq!(n_choose_k(2, 0), 1);
        assert_eq!(n_choose_k(3, 0), 1);
        assert_eq!(n_choose_k(1, 1), 1);
        assert_eq!(n_choose_k(2, 1), 2);
        assert_eq!(n_choose_k(3, 1), 3);
        assert_eq!(n_choose_k(2, 2), 1);
        assert_eq!(n_choose_k(2, 3), 0);
        assert_eq!(n_choose_k(3, 2), 3);
        assert_eq!(n_choose_k(4, 2), 6);
        assert_eq!(n_choose_k(5, 2), 10);
        assert_eq!(n_choose_k(500, 3), 20_708_500);
        assert_eq!(n_choose_k(500, 7), 1_486_071_034_734_000);
        assert_eq!(n_choose_k(10_000, 5), 832_500_291_625_002_000);
    }

    #[test]
    fn colmap_combination_sampler_enumerates_and_wraps() {
        let mut sampler = ColmapCombinationSampler::new(2);
        assert!(sampler.initialize(5));
        assert_eq!(sampler.max_num_samples(), 10);

        let expected = [
            vec![0, 1],
            vec![0, 2],
            vec![0, 3],
            vec![0, 4],
            vec![1, 2],
            vec![1, 3],
            vec![1, 4],
            vec![2, 3],
            vec![2, 4],
            vec![3, 4],
        ];
        for sample in expected {
            assert_eq!(sampler.sample(), sample);
        }
        assert_eq!(sampler.sample(), vec![0, 1]);
    }

    #[test]
    fn colmap_combination_sampler_handles_equal_samples() {
        let mut sampler = ColmapCombinationSampler::new(5);
        assert!(sampler.initialize(5));
        assert_eq!(sampler.max_num_samples(), 1);
        for _ in 0..10 {
            assert_eq!(sampler.sample(), vec![0, 1, 2, 3, 4]);
        }
    }

    #[test]
    fn colmap_progressive_sampler_matches_reference_seeded_sequence() {
        let mut sampler = ColmapProgressiveSampler::new(42, 5);
        assert!(sampler.initialize(50));
        assert_eq!(sampler.max_num_samples(), usize::MAX);

        let expected = [
            vec![3, 4, 2, 1, 6],
            vec![2, 4, 3, 5, 7],
            vec![4, 1, 3, 5, 7],
            vec![5, 1, 3, 4, 8],
            vec![0, 3, 1, 5, 8],
            vec![4, 3, 0, 2, 8],
            vec![2, 6, 1, 3, 8],
            vec![3, 7, 6, 5, 9],
        ];
        for sample in expected {
            assert_eq!(sampler.sample(), sample);
        }
    }

    #[test]
    fn colmap_progressive_sampler_preserves_prosac_invariants() {
        let mut sampler = ColmapProgressiveSampler::new(1, 5);
        assert!(sampler.initialize(50));
        let mut prev_last_sample = 5;
        for _ in 0..100 {
            let sample = sampler.sample();
            assert_eq!(sample.len(), 5);
            let mut sorted = sample.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), 5);

            let last = *sample.last().unwrap();
            assert!(last >= prev_last_sample);
            for &idx in &sample[..sample.len() - 1] {
                assert!(idx < last);
            }
            prev_last_sample = last;
        }
    }

    #[test]
    fn colmap_progressive_sampler_handles_equal_samples_like_colmap() {
        let mut sampler = ColmapProgressiveSampler::new(42, 5);
        assert!(sampler.initialize(5));
        for _ in 0..100 {
            let sample = sampler.sample();
            assert_eq!(sample.len(), 5);
            let mut sorted = sample.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), 5);
        }
    }
}
