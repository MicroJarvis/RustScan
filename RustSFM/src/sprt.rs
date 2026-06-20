//! Rust port of COLMAP's `optim/sprt` module.
//!
//! Implements the Sequential Probability Ratio Test used by randomized RANSAC.

/// COLMAP `SPRT::Options`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SprtOptions {
    /// Probability of rejecting a good model.
    pub delta: f64,
    /// A priori assumed minimum inlier ratio.
    pub epsilon: f64,
    /// Ratio of model-estimation time over single-sample evaluation time.
    pub eval_time_ratio: f64,
    /// Number of candidate models generated per random sample.
    pub num_models_per_sample: i32,
}

impl Default for SprtOptions {
    fn default() -> Self {
        Self {
            delta: 0.01,
            epsilon: 0.1,
            eval_time_ratio: 200.0,
            num_models_per_sample: 1,
        }
    }
}

/// COLMAP `SPRT`.
#[derive(Debug, Clone)]
pub struct Sprt {
    options: SprtOptions,
    delta_epsilon: f64,
    delta_1_epsilon_1: f64,
    decision_threshold: f64,
}

impl Sprt {
    pub fn new(options: SprtOptions) -> Self {
        let mut sprt = Self {
            options,
            delta_epsilon: 0.0,
            delta_1_epsilon_1: 0.0,
            decision_threshold: 0.0,
        };
        sprt.update(options);
        sprt
    }

    pub fn update(&mut self, options: SprtOptions) {
        self.options = options;
        self.delta_epsilon = options.delta / options.epsilon;
        self.delta_1_epsilon_1 = (1.0 - options.delta) / (1.0 - options.epsilon);
        self.update_decision_threshold();
    }

    pub fn evaluate(&self, residuals: &[f64], max_residual: f64) -> SprtEvaluation {
        let mut num_inliers = 0usize;
        let mut likelihood_ratio = 1.0;

        for (idx, &residual) in residuals.iter().enumerate() {
            if residual.abs() <= max_residual {
                num_inliers += 1;
                likelihood_ratio *= self.delta_epsilon;
            } else {
                likelihood_ratio *= self.delta_1_epsilon_1;
            }

            if likelihood_ratio > self.decision_threshold {
                return SprtEvaluation {
                    accepted: false,
                    num_inliers,
                    num_eval_samples: idx + 1,
                };
            }
        }

        SprtEvaluation {
            accepted: true,
            num_inliers,
            num_eval_samples: residuals.len(),
        }
    }

    pub fn decision_threshold(&self) -> f64 {
        self.decision_threshold
    }

    fn update_decision_threshold(&mut self) {
        let c = (1.0 - self.options.delta)
            * ((1.0 - self.options.delta) / (1.0 - self.options.epsilon)).ln()
            + self.options.delta * (self.options.delta / self.options.epsilon).ln();

        let a0 =
            self.options.eval_time_ratio * c / self.options.num_models_per_sample as f64 + 1.0;
        let mut a = a0;
        const EPS: f64 = 1.5e-8;

        for _ in 0..100 {
            let a1 = a0 + a.ln();
            if (a1 - a).abs() < EPS {
                break;
            }
            a = a1;
        }

        self.decision_threshold = a;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SprtEvaluation {
    pub accepted: bool,
    pub num_inliers: usize,
    pub num_eval_samples: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sprt_evaluate_all_inliers() {
        let options = SprtOptions {
            delta: 0.05,
            epsilon: 0.5,
            ..SprtOptions::default()
        };
        let sprt = Sprt::new(options);
        let residuals = vec![0.1; 100];

        let report = sprt.evaluate(&residuals, 1.0);
        assert!(report.accepted);
        assert_eq!(report.num_inliers, 100);
        assert_eq!(report.num_eval_samples, 100);
    }

    #[test]
    fn sprt_evaluate_all_outliers() {
        let options = SprtOptions {
            delta: 0.05,
            epsilon: 0.5,
            ..SprtOptions::default()
        };
        let sprt = Sprt::new(options);
        let residuals = vec![10.0; 100];

        let report = sprt.evaluate(&residuals, 1.0);
        assert!(!report.accepted);
        assert_eq!(report.num_inliers, 0);
        assert!(report.num_eval_samples < 100);
    }

    #[test]
    fn sprt_evaluate_mixed_early_reject() {
        let options = SprtOptions {
            delta: 0.05,
            epsilon: 0.9,
            ..SprtOptions::default()
        };
        let sprt = Sprt::new(options);
        let mut residuals = vec![10.0; 1000];
        residuals[0] = 0.1;
        residuals[10] = 0.1;

        let report = sprt.evaluate(&residuals, 1.0);
        assert!(!report.accepted);
        assert_eq!(report.num_inliers, 1);
        assert_eq!(report.num_eval_samples, 5);
    }

    #[test]
    fn sprt_evaluate_empty() {
        let sprt = Sprt::new(SprtOptions::default());
        let report = sprt.evaluate(&[], 1.0);
        assert!(report.accepted);
        assert_eq!(report.num_inliers, 0);
        assert_eq!(report.num_eval_samples, 0);
    }
}
