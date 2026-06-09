//! LiteGS-compatible Density Controller implementation.
//!
//! This module replicates the behavior of LiteGS's DensityControllerOfficial
//! and DensityControllerTamingGS from `litegs/training/densify.py`.

/// Statistics helper for tracking per-Gaussian metrics during training.
///
/// Mirrors LiteGS's `StatisticsHelper` for accumulating mean2d_grad,
/// fragment_weight, fragment_err, and visibility counts.
#[derive(Debug, Clone)]
pub struct StatisticsHelper {
    /// Per-Gaussian visible count (how many times each Gaussian was rendered).
    pub visible_count: Vec<u32>,
    /// Accumulated mean2d_grad: (sum, count) for each Gaussian.
    pub mean2d_grad: Vec<(f32, f32)>,
    /// Accumulated fragment_weight: (sum, count) for each Gaussian.
    pub fragment_weight: Vec<(f32, f32)>,
    /// Accumulated fragment_err: (sum, square_sum, count) for each Gaussian.
    /// Used for variance computation: var = square_sum/count - (sum/count)^2
    pub fragment_err: Vec<(f32, f32, f32)>,
    /// Current opacity values (sigmoid space).
    pub opacity: Vec<f32>,
    /// Current max scale for each Gaussian.
    pub max_scale: Vec<f32>,
}

impl StatisticsHelper {
    pub fn new(gaussian_count: usize) -> Self {
        Self {
            visible_count: vec![0; gaussian_count],
            mean2d_grad: vec![(0.0, 0.0); gaussian_count],
            fragment_weight: vec![(0.0, 0.0); gaussian_count],
            fragment_err: vec![(0.0, 0.0, 0.0); gaussian_count],
            opacity: vec![0.0; gaussian_count],
            max_scale: vec![0.0; gaussian_count],
        }
    }

    pub fn len(&self) -> usize {
        self.visible_count.len()
    }

    /// Resize to match new Gaussian count after topology changes.
    pub fn resize(&mut self, new_len: usize) {
        let old_len = self.len();
        if new_len > old_len {
            self.visible_count.extend(vec![0; new_len - old_len]);
            self.mean2d_grad.extend(vec![(0.0, 0.0); new_len - old_len]);
            self.fragment_weight
                .extend(vec![(0.0, 0.0); new_len - old_len]);
            self.fragment_err
                .extend(vec![(0.0, 0.0, 0.0); new_len - old_len]);
            self.opacity.extend(vec![0.0; new_len - old_len]);
            self.max_scale.extend(vec![0.0; new_len - old_len]);
        } else if new_len < old_len {
            self.visible_count.truncate(new_len);
            self.mean2d_grad.truncate(new_len);
            self.fragment_weight.truncate(new_len);
            self.fragment_err.truncate(new_len);
            self.opacity.truncate(new_len);
            self.max_scale.truncate(new_len);
        }
    }

    /// Get mean of mean2d_grad for a Gaussian.
    pub fn get_mean2d_grad(&self, idx: usize) -> f32 {
        self.mean2d_grad
            .get(idx)
            .map(|(sum, count)| if *count > 0.0 { sum / count } else { 0.0 })
            .unwrap_or(0.0)
    }

    /// Get mean of fragment_weight for a Gaussian.
    pub fn get_fragment_weight_mean(&self, idx: usize) -> f32 {
        self.fragment_weight
            .get(idx)
            .map(|(sum, count)| if *count > 0.0 { sum / count } else { 0.0 })
            .unwrap_or(0.0)
    }

    /// Get weight_sum = fragment_weight_mean * visible_count (LiteGS semantics).
    pub fn get_weight_sum(&self, idx: usize) -> f32 {
        let weight_mean = self.get_fragment_weight_mean(idx);
        let count = self.visible_count.get(idx).copied().unwrap_or(0) as f32;
        weight_mean * count
    }
}

impl Default for StatisticsHelper {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Configuration for the density controller.
#[derive(Debug, Clone)]
pub struct DensityControllerConfig {
    /// Gradient threshold for densification.
    pub densify_grad_threshold: f32,
    /// Opacity threshold for pruning.
    pub opacity_threshold: f32,
    /// Percentage of screen space that defines "small" vs "large" Gaussians.
    pub percent_dense: f32,
    /// Screen extent for scale threshold computation.
    pub screen_extent: f32,
    /// Initial point count for budget calculations.
    pub init_points_num: usize,
    /// Target number of primitives (TamingGS mode).
    pub target_primitives: usize,
    /// Epoch to start densification.
    pub densify_from: usize,
    /// Epoch to stop densification.
    pub densify_until: usize,
    /// Interval between densification steps.
    pub densification_interval: usize,
    /// Prune mode: "weight" or "threshold".
    pub prune_mode: PruneMode,
}

impl Default for DensityControllerConfig {
    fn default() -> Self {
        Self {
            densify_grad_threshold: 0.0002,
            opacity_threshold: 0.005,
            percent_dense: 0.01,
            screen_extent: 1.0,
            init_points_num: 0,
            target_primitives: 1_000_000,
            densify_from: 500,
            densify_until: 15000,
            densification_interval: 100,
            prune_mode: PruneMode::Weight,
        }
    }
}

/// Prune mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruneMode {
    /// Prune based on fragment weight (TamingGS).
    Weight,
    /// Prune based on opacity threshold (Official).
    Threshold,
}

/// Density controller implementing LiteGS parity.
///
/// This struct mirrors the behavior of `DensityControllerOfficial` and
/// `DensityControllerTamingGS` from LiteGS.
#[derive(Debug, Clone)]
pub struct DensityController {
    pub config: DensityControllerConfig,
    pub stats: StatisticsHelper,
    /// Whether to use TamingGS-style budget-aware densification.
    pub taming_gs_mode: bool,
}

impl DensityController {
    pub fn new(config: DensityControllerConfig, taming_gs_mode: bool) -> Self {
        let stats = StatisticsHelper::new(config.init_points_num.max(1));
        Self {
            config,
            stats,
            taming_gs_mode,
        }
    }

    /// Check if densification is active at the given epoch.
    pub fn is_densify_active(&self, epoch: usize) -> bool {
        epoch >= self.config.densify_from
            && epoch < self.config.densify_until
            && epoch.is_multiple_of(self.config.densification_interval)
    }

    /// Get prune mask based on opacity and visibility.
    ///
    /// LiteGS Official:
    /// - transparent = opacity < threshold
    /// - invisible = global_culling (never visible)
    /// - prune_mask = transparent | invisible
    ///
    /// LiteGS TamingGS (weight mode):
    /// - weight_sum = fragment_weight * fragment_count
    /// - prune when weight_sum == 0
    pub fn get_prune_mask(&self) -> Vec<bool> {
        let n = self.stats.len();
        let mut prune_mask = vec![false; n];

        for (i, prune) in prune_mask.iter_mut().enumerate().take(n) {
            let opacity = self.stats.opacity.get(i).copied().unwrap_or(0.0);
            let visible = self.stats.visible_count.get(i).copied().unwrap_or(0) > 0;

            match self.config.prune_mode {
                PruneMode::Threshold => {
                    // Official: prune if opacity < threshold or never visible
                    let transparent = opacity < self.config.opacity_threshold;
                    let invisible = !visible;
                    *prune = transparent || invisible;
                }
                PruneMode::Weight => {
                    // TamingGS: prune when weight_sum == 0
                    let weight_sum = self.stats.get_weight_sum(i);
                    *prune = weight_sum == 0.0;
                }
            }
        }

        prune_mask
    }

    /// Get clone mask based on gradient and scale.
    ///
    /// LiteGS: clone when grad >= threshold AND scale is small.
    pub fn get_clone_mask(&self) -> Vec<bool> {
        let n = self.stats.len();
        let mut clone_mask = vec![false; n];
        let scale_threshold = self.config.percent_dense * self.config.screen_extent;

        for (i, clone) in clone_mask.iter_mut().enumerate().take(n) {
            let grad = self.stats.get_mean2d_grad(i);
            let max_scale = self.stats.max_scale.get(i).copied().unwrap_or(0.0);
            let opacity = self.stats.opacity.get(i).copied().unwrap_or(0.0);

            // LiteGS: abnormal (high grad) AND tiny (small scale)
            let abnormal = grad >= self.config.densify_grad_threshold;
            let tiny = max_scale <= scale_threshold;
            let has_opacity = opacity > self.config.opacity_threshold;

            *clone = abnormal && tiny && has_opacity;
        }

        clone_mask
    }

    /// Get split mask based on gradient and scale.
    ///
    /// LiteGS: split when grad >= threshold AND scale is large.
    pub fn get_split_mask(&self) -> Vec<bool> {
        let n = self.stats.len();
        let mut split_mask = vec![false; n];
        let scale_threshold = self.config.percent_dense * self.config.screen_extent;

        for (i, split) in split_mask.iter_mut().enumerate().take(n) {
            let grad = self.stats.get_mean2d_grad(i);
            let max_scale = self.stats.max_scale.get(i).copied().unwrap_or(0.0);
            let opacity = self.stats.opacity.get(i).copied().unwrap_or(0.0);

            // LiteGS: abnormal (high grad) AND large (big scale)
            let abnormal = grad >= self.config.densify_grad_threshold;
            let large = max_scale > scale_threshold;
            let has_opacity = opacity > self.config.opacity_threshold;

            *split = abnormal && large && has_opacity;
        }

        split_mask
    }

    /// Compute densify budget for TamingGS mode.
    ///
    /// LiteGS TamingGS:
    /// cur_target = (target - init) / (densify_until - densify_from) * (epoch - densify_from) + init
    /// budget = min(max(cur_target - current_count, 1) + prune_count, current_count)
    pub fn compute_densify_budget(
        &self,
        current_count: usize,
        prune_count: usize,
        epoch: usize,
    ) -> usize {
        if !self.taming_gs_mode {
            // Official mode: no budget limit
            return current_count;
        }

        let init = self.config.init_points_num.max(1);
        let densify_from = self.config.densify_from;
        let densify_until = self.config.densify_until;
        let target = self.config.target_primitives;

        let span = densify_until.saturating_sub(densify_from).max(1);
        let progressed = epoch.saturating_sub(densify_from);

        let cur_target =
            init as f32 + ((target.saturating_sub(init)) as f32 / span as f32) * progressed as f32;
        let cur_target = cur_target.round() as usize;

        let deficit = cur_target.saturating_sub(current_count).max(1);
        deficit.saturating_add(prune_count).min(current_count)
    }
}
