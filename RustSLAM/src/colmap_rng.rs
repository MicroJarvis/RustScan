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
    configured_num_samples: Option<usize>,
}

impl ColmapRandomSampler {
    pub fn new(seed: u64, indices: &[usize]) -> Self {
        Self {
            rng: ColmapMt19937::new(seed),
            sample_indices: indices.to_vec(),
            configured_num_samples: None,
        }
    }

    pub fn with_num_samples(seed: u64, num_samples: usize) -> Self {
        Self {
            rng: ColmapMt19937::new(seed),
            sample_indices: Vec::new(),
            configured_num_samples: Some(num_samples),
        }
    }

    pub fn initialize(&mut self, total_num_samples: usize) -> bool {
        let Some(num_samples) = self.configured_num_samples else {
            return false;
        };
        if num_samples > total_num_samples {
            self.sample_indices.clear();
            return false;
        }
        self.sample_indices = (0..total_num_samples).collect();
        true
    }

    pub fn max_num_samples(&self) -> usize {
        usize::MAX
    }

    pub fn sample_configured(&mut self) -> Vec<usize> {
        match self.configured_num_samples {
            Some(num_samples) => self.sample(num_samples),
            None => Vec::new(),
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

/// COLMAP `RANSAC::ComputeNumTrials` with an explicit minimal sample size.
pub fn colmap_ransac_num_trials(
    num_inliers: usize,
    num_samples: usize,
    min_num_samples: usize,
    confidence: f64,
    num_trials_multiplier: f64,
) -> usize {
    let prob_failure = 1.0 - confidence;
    if prob_failure <= 0.0 {
        return usize::MAX;
    }

    if num_samples < min_num_samples || num_inliers < min_num_samples {
        return usize::MAX;
    }

    let mut prob_inlier = 1.0;
    for i in 0..min_num_samples {
        prob_inlier *= (num_inliers - i) as f64 / (num_samples - i) as f64;
    }

    let prob_outlier = 1.0 - prob_inlier;
    if prob_outlier <= 0.0 {
        return 1;
    }
    if prob_outlier == 1.0 {
        return usize::MAX;
    }

    let trials = (prob_failure.ln() / prob_outlier.ln() * num_trials_multiplier).ceil();
    if trials.is_finite() && trials > 0.0 {
        trials as usize
    } else {
        usize::MAX
    }
}

/// COLMAP stores `RANSACOptions::max_num_trials` as a signed `int`.
pub const COLMAP_RANSAC_DEFAULT_MAX_NUM_TRIALS: usize = i32::MAX as usize;

/// COLMAP `RANSACOptions` default surface and validation rules.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColmapRansacOptions {
    pub max_error: f64,
    pub min_inlier_ratio: f64,
    pub confidence: f64,
    pub dyn_num_trials_multiplier: f64,
    pub min_num_trials: usize,
    pub max_num_trials: usize,
    pub random_seed: i32,
    pub num_threads: isize,
}

impl Default for ColmapRansacOptions {
    fn default() -> Self {
        Self {
            max_error: 0.0,
            min_inlier_ratio: 0.1,
            confidence: 0.99,
            dyn_num_trials_multiplier: 3.0,
            min_num_trials: 0,
            max_num_trials: COLMAP_RANSAC_DEFAULT_MAX_NUM_TRIALS,
            random_seed: -1,
            num_threads: 1,
        }
    }
}

impl ColmapRansacOptions {
    pub fn check(&self) -> Result<(), &'static str> {
        if self.max_error <= 0.0 {
            return Err("max_error must be positive");
        }
        if !(0.0..=1.0).contains(&self.min_inlier_ratio) {
            return Err("min_inlier_ratio must be in [0, 1]");
        }
        if !(0.0..=1.0).contains(&self.confidence) {
            return Err("confidence must be in [0, 1]");
        }
        if self.min_num_trials > self.max_num_trials {
            return Err("min_num_trials must not exceed max_num_trials");
        }
        if self.random_seed < -1 {
            return Err("random_seed must be >= -1");
        }
        if self.num_threads == 0 || self.num_threads < -1 {
            return Err("num_threads must be -1 or positive");
        }
        Ok(())
    }

    pub fn initial_max_num_trials(&self, min_num_samples: usize) -> usize {
        let assumed_samples = 100_000usize;
        let assumed_inliers =
            (self.min_inlier_ratio.clamp(0.0, 1.0) * assumed_samples as f64) as usize;
        self.max_num_trials.min(colmap_ransac_num_trials(
            assumed_inliers,
            assumed_samples,
            min_num_samples,
            self.confidence,
            self.dyn_num_trials_multiplier,
        ))
    }

    pub fn with_initial_max_num_trials(
        mut self,
        min_num_samples: usize,
    ) -> Result<Self, &'static str> {
        self.check()?;
        self.max_num_trials = self.initial_max_num_trials(min_num_samples);
        Ok(self)
    }
}

/// COLMAP `RANSAC::Report` / `LORANSAC::Report` shape.
#[derive(Debug, Clone, PartialEq)]
pub struct ColmapRansacReport<Support, Model> {
    pub success: bool,
    pub num_trials: usize,
    pub support: Support,
    pub inlier_mask: Vec<bool>,
    pub model: Option<Model>,
}

impl<Support: Default, Model> Default for ColmapRansacReport<Support, Model> {
    fn default() -> Self {
        Self {
            success: false,
            num_trials: 0,
            support: Support::default(),
            inlier_mask: Vec::new(),
            model: None,
        }
    }
}

impl<Support, Model> ColmapRansacReport<Support, Model> {
    pub fn from_success(
        num_trials: usize,
        support: Support,
        inlier_mask: Vec<bool>,
        model: Model,
    ) -> Self {
        Self {
            success: true,
            num_trials,
            support,
            inlier_mask,
            model: Some(model),
        }
    }

    pub fn inlier_mask_from_residuals(residuals: &[f64], max_error: f64) -> Vec<bool> {
        let max_residual = max_error * max_error;
        residuals
            .iter()
            .map(|&residual| residual <= max_residual)
            .collect()
    }
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
            self.t_n *= (self.num_samples - i) as f64 / (self.total_num_samples - i) as f64;
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
            let t_n_plus_1 =
                self.t_n * (self.n as f64 + 1.0) / (self.n as f64 + 1.0 - self.num_samples as f64);
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
    fn colmap_random_sampler_official_less_samples_shape() {
        let mut sampler = ColmapRandomSampler::with_num_samples(42, 2);
        assert!(sampler.initialize(5));
        assert_eq!(sampler.max_num_samples(), usize::MAX);

        for _ in 0..100 {
            let samples = sampler.sample_configured();
            assert_eq!(samples.len(), 2);
            let mut unique = samples.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), 2);
            assert!(samples.iter().all(|&idx| idx < 5));
        }
    }

    #[test]
    fn colmap_random_sampler_official_equal_samples_shape() {
        let mut sampler = ColmapRandomSampler::with_num_samples(42, 5);
        assert!(sampler.initialize(5));
        assert_eq!(sampler.max_num_samples(), usize::MAX);

        for _ in 0..100 {
            let samples = sampler.sample_configured();
            assert_eq!(samples.len(), 5);
            let mut unique = samples.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), 5);
            assert_eq!(unique, vec![0, 1, 2, 3, 4]);
        }
    }

    #[test]
    fn colmap_random_sampler_initialize_rejects_oversized_requests() {
        let mut sampler = ColmapRandomSampler::with_num_samples(1, 3);
        assert!(!sampler.initialize(2));
        assert!(sampler.sample_configured().is_empty());
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
    fn colmap_ransac_num_trials_matches_official_examples() {
        assert_eq!(colmap_ransac_num_trials(1, 100, 3, 0.99, 1.0), usize::MAX);
        assert_eq!(colmap_ransac_num_trials(10, 100, 3, 0.99, 1.0), 6204);
        assert_eq!(colmap_ransac_num_trials(10, 100, 3, 0.999, 1.0), 9305);
        assert_eq!(colmap_ransac_num_trials(10, 100, 3, 0.999, 2.0), 18610);
        assert_eq!(colmap_ransac_num_trials(50, 100, 3, 0.99, 1.0), 36);
        assert_eq!(colmap_ransac_num_trials(50, 100, 3, 0.999, 1.0), 54);
        assert_eq!(colmap_ransac_num_trials(100, 100, 3, 0.99, 1.0), 1);
        assert_eq!(colmap_ransac_num_trials(100, 100, 3, 0.999, 1.0), 1);
        assert_eq!(colmap_ransac_num_trials(100, 100, 3, 0.0, 1.0), 1);
    }

    #[test]
    fn colmap_ransac_options_match_official_defaults() {
        let options = ColmapRansacOptions::default();
        assert_eq!(options.max_error, 0.0);
        assert_eq!(options.min_inlier_ratio, 0.1);
        assert_eq!(options.confidence, 0.99);
        assert_eq!(options.dyn_num_trials_multiplier, 3.0);
        assert_eq!(options.min_num_trials, 0);
        assert_eq!(options.max_num_trials, COLMAP_RANSAC_DEFAULT_MAX_NUM_TRIALS);
        assert_eq!(options.random_seed, -1);
        assert_eq!(options.num_threads, 1);
    }

    #[test]
    fn colmap_ransac_options_check_matches_official_bounds() {
        let mut options = ColmapRansacOptions {
            max_error: 1.0,
            ..ColmapRansacOptions::default()
        };
        assert!(options.check().is_ok());

        options.max_error = 0.0;
        assert!(options.check().is_err());
        options.max_error = 1.0;
        options.min_inlier_ratio = -1.0e-6;
        assert!(options.check().is_err());
        options.min_inlier_ratio = 0.1;
        options.confidence = 1.0 + 1.0e-6;
        assert!(options.check().is_err());
        options.confidence = 0.99;
        options.min_num_trials = 2;
        options.max_num_trials = 1;
        assert!(options.check().is_err());
        options.min_num_trials = 0;
        options.max_num_trials = COLMAP_RANSAC_DEFAULT_MAX_NUM_TRIALS;
        options.random_seed = -2;
        assert!(options.check().is_err());
        options.random_seed = -1;
        options.num_threads = 0;
        assert!(options.check().is_err());
        options.num_threads = -2;
        assert!(options.check().is_err());
    }

    #[test]
    fn colmap_ransac_options_constructor_clamps_initial_trials() {
        let options = ColmapRansacOptions {
            max_error: 1.0,
            min_inlier_ratio: 0.5,
            confidence: 0.999,
            dyn_num_trials_multiplier: 1.0,
            max_num_trials: 10_000,
            ..ColmapRansacOptions::default()
        };
        let checked = options.with_initial_max_num_trials(3).unwrap();
        assert_eq!(checked.max_num_trials, 52);

        let options = ColmapRansacOptions {
            max_error: 1.0,
            min_inlier_ratio: 0.1,
            confidence: 0.99,
            dyn_num_trials_multiplier: 3.0,
            max_num_trials: 25,
            ..ColmapRansacOptions::default()
        };
        let checked = options.with_initial_max_num_trials(3).unwrap();
        assert_eq!(checked.max_num_trials, 25);

        let invalid = ColmapRansacOptions::default();
        assert!(invalid.with_initial_max_num_trials(3).is_err());
    }

    #[test]
    fn colmap_ransac_report_matches_official_default_shape() {
        let report = ColmapRansacReport::<usize, [f64; 3]>::default();
        assert!(!report.success);
        assert_eq!(report.num_trials, 0);
        assert_eq!(report.support, 0);
        assert!(report.inlier_mask.is_empty());
        assert!(report.model.is_none());
    }

    #[test]
    fn colmap_ransac_report_builds_success_and_mask_from_residuals() {
        let mask = ColmapRansacReport::<usize, [f64; 3]>::inlier_mask_from_residuals(
            &[0.0, 1.0, 4.0, 4.01],
            2.0,
        );
        assert_eq!(mask, vec![true, true, true, false]);

        let report = ColmapRansacReport::from_success(7, 3usize, mask, [1.0, 2.0, 3.0]);
        assert!(report.success);
        assert_eq!(report.num_trials, 7);
        assert_eq!(report.support, 3);
        assert_eq!(report.inlier_mask, vec![true, true, true, false]);
        assert_eq!(report.model, Some([1.0, 2.0, 3.0]));
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
