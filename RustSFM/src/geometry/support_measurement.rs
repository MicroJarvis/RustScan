//! Rust port of COLMAP's `optim/support_measurement` module.
//!
//! These support measurers are shared by RANSAC/LORANSAC-style estimators.

use std::collections::HashSet;

/// COLMAP `InlierSupportMeasurer::Support`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InlierSupport {
    pub num_inliers: usize,
    pub residual_sum: f64,
}

impl Default for InlierSupport {
    fn default() -> Self {
        Self {
            num_inliers: 0,
            residual_sum: f64::MAX,
        }
    }
}

/// COLMAP `InlierSupportMeasurer`.
#[derive(Debug, Clone, Copy, Default)]
pub struct InlierSupportMeasurer;

impl InlierSupportMeasurer {
    pub fn evaluate(&self, residuals: &[f64], max_residual: f64) -> InlierSupport {
        let mut support = InlierSupport {
            num_inliers: 0,
            residual_sum: 0.0,
        };
        for &residual in residuals {
            if residual <= max_residual {
                support.num_inliers += 1;
                support.residual_sum += residual;
            }
        }
        support
    }

    pub fn is_left_better(&self, left: &InlierSupport, right: &InlierSupport) -> bool {
        left.num_inliers > right.num_inliers
            || (left.num_inliers == right.num_inliers && left.residual_sum < right.residual_sum)
    }
}

/// COLMAP `UniqueInlierSupportMeasurer::Support`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UniqueInlierSupport {
    pub num_unique_inliers: usize,
    pub num_inliers: usize,
    pub residual_sum: f64,
}

impl Default for UniqueInlierSupport {
    fn default() -> Self {
        Self {
            num_unique_inliers: 0,
            num_inliers: 0,
            residual_sum: f64::MAX,
        }
    }
}

/// COLMAP `UniqueInlierSupportMeasurer`.
#[derive(Debug, Clone)]
pub struct UniqueInlierSupportMeasurer {
    unique_sample_ids: Vec<usize>,
}

impl UniqueInlierSupportMeasurer {
    pub fn new(unique_sample_ids: Vec<usize>) -> Self {
        Self { unique_sample_ids }
    }

    pub fn evaluate(&self, residuals: &[f64], max_residual: f64) -> UniqueInlierSupport {
        assert_eq!(residuals.len(), self.unique_sample_ids.len());

        let mut support = UniqueInlierSupport {
            num_unique_inliers: 0,
            num_inliers: 0,
            residual_sum: 0.0,
        };
        let mut inlier_point_ids = HashSet::new();
        for (idx, &residual) in residuals.iter().enumerate() {
            if residual <= max_residual {
                support.num_inliers += 1;
                inlier_point_ids.insert(self.unique_sample_ids[idx]);
                support.residual_sum += residual;
            }
        }
        support.num_unique_inliers = inlier_point_ids.len();
        support
    }

    pub fn is_left_better(&self, left: &UniqueInlierSupport, right: &UniqueInlierSupport) -> bool {
        if left.num_unique_inliers > right.num_unique_inliers {
            true
        } else if left.num_unique_inliers == right.num_unique_inliers {
            left.num_inliers > right.num_inliers
                || (left.num_inliers == right.num_inliers && left.residual_sum < right.residual_sum)
        } else {
            false
        }
    }
}

/// COLMAP `MEstimatorSupportMeasurer::Support`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MEstimatorSupport {
    pub num_inliers: usize,
    pub score: f64,
}

impl Default for MEstimatorSupport {
    fn default() -> Self {
        Self {
            num_inliers: 0,
            score: f64::MAX,
        }
    }
}

/// COLMAP `MEstimatorSupportMeasurer`.
#[derive(Debug, Clone, Copy, Default)]
pub struct MEstimatorSupportMeasurer;

impl MEstimatorSupportMeasurer {
    pub fn evaluate(&self, residuals: &[f64], max_residual: f64) -> MEstimatorSupport {
        let mut support = MEstimatorSupport {
            num_inliers: 0,
            score: 0.0,
        };
        for &residual in residuals {
            if residual <= max_residual {
                support.num_inliers += 1;
                support.score += residual;
            } else {
                support.score += max_residual;
            }
        }
        support
    }

    pub fn is_left_better(&self, left: &MEstimatorSupport, right: &MEstimatorSupport) -> bool {
        left.score < right.score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inlier_support_measurer_matches_colmap_nominal() {
        let support1 = InlierSupport::default();
        assert_eq!(support1.num_inliers, 0);
        assert_eq!(support1.residual_sum, f64::MAX);

        let measurer = InlierSupportMeasurer;
        let residuals = [-1.0, 0.0, 1.0, 2.0];
        let support1 = measurer.evaluate(&residuals, 1.0);
        assert_eq!(support1.num_inliers, 3);
        assert_eq!(support1.residual_sum, 0.0);

        let mut support2 = InlierSupport::default();
        support2.num_inliers = 2;
        assert!(measurer.is_left_better(&support1, &support2));
        assert!(!measurer.is_left_better(&support2, &support1));

        support2.residual_sum = support1.residual_sum;
        assert!(measurer.is_left_better(&support1, &support2));
        assert!(!measurer.is_left_better(&support2, &support1));

        support2.num_inliers = support1.num_inliers;
        support2.residual_sum += 0.01;
        assert!(measurer.is_left_better(&support1, &support2));
        assert!(!measurer.is_left_better(&support2, &support1));

        support2.residual_sum -= 0.01;
        assert!(!measurer.is_left_better(&support1, &support2));
        assert!(!measurer.is_left_better(&support2, &support1));

        support2.residual_sum -= 0.01;
        assert!(!measurer.is_left_better(&support1, &support2));
        assert!(measurer.is_left_better(&support2, &support1));
    }

    #[test]
    fn unique_inlier_support_measurer_matches_colmap_nominal() {
        let support1 = UniqueInlierSupport::default();
        assert_eq!(support1.num_inliers, 0);
        assert_eq!(support1.num_unique_inliers, 0);
        assert_eq!(support1.residual_sum, f64::MAX);

        let measurer = UniqueInlierSupportMeasurer::new(vec![1, 2, 2, 3]);
        let residuals = [-1.0, 0.0, 1.0, 2.0];
        let support1 = measurer.evaluate(&residuals, 1.0);
        assert_eq!(support1.num_inliers, 3);
        assert_eq!(support1.num_unique_inliers, 2);
        assert_eq!(support1.residual_sum, 0.0);

        let mut support2 = UniqueInlierSupport::default();
        support2.num_unique_inliers = support1.num_unique_inliers - 1;
        assert!(measurer.is_left_better(&support1, &support2));
        assert!(!measurer.is_left_better(&support2, &support1));

        support2.num_inliers = support1.num_inliers + 1;
        assert!(measurer.is_left_better(&support1, &support2));
        assert!(!measurer.is_left_better(&support2, &support1));

        support2.num_inliers = support1.num_inliers;
        assert!(measurer.is_left_better(&support1, &support2));
        assert!(!measurer.is_left_better(&support2, &support1));

        support2.residual_sum = support1.residual_sum - 0.01;
        assert!(measurer.is_left_better(&support1, &support2));
        assert!(!measurer.is_left_better(&support2, &support1));

        support2.residual_sum = support1.residual_sum;
        assert!(measurer.is_left_better(&support1, &support2));
        assert!(!measurer.is_left_better(&support2, &support1));

        support2.num_unique_inliers = support1.num_unique_inliers;
        assert!(!measurer.is_left_better(&support1, &support2));
        assert!(!measurer.is_left_better(&support2, &support1));

        support2.residual_sum = support1.residual_sum - 0.01;
        assert!(!measurer.is_left_better(&support1, &support2));
        assert!(measurer.is_left_better(&support2, &support1));

        support2.num_inliers = support1.num_inliers + 1;
        support2.residual_sum = support1.residual_sum + 0.01;
        assert!(!measurer.is_left_better(&support1, &support2));
        assert!(measurer.is_left_better(&support2, &support1));

        support2.num_unique_inliers = support1.num_unique_inliers + 1;
        support2.num_inliers = support1.num_inliers - 1;
        support2.residual_sum = support1.residual_sum + 0.01;
        assert!(!measurer.is_left_better(&support1, &support2));
        assert!(measurer.is_left_better(&support2, &support1));
    }

    #[test]
    fn m_estimator_support_measurer_matches_colmap_nominal() {
        let support1 = MEstimatorSupport::default();
        assert_eq!(support1.num_inliers, 0);
        assert_eq!(support1.score, f64::MAX);

        let measurer = MEstimatorSupportMeasurer;
        let residuals = [-1.0, 0.0, 1.0, 2.0];
        let support1 = measurer.evaluate(&residuals, 1.0);
        assert_eq!(support1.num_inliers, 3);
        assert_eq!(support1.score, 1.0);

        let mut support2 = support1;
        assert!(!measurer.is_left_better(&support1, &support2));
        assert!(!measurer.is_left_better(&support2, &support1));

        support2.num_inliers -= 1;
        support2.score += 0.01;
        assert!(measurer.is_left_better(&support1, &support2));
        assert!(!measurer.is_left_better(&support2, &support1));

        support2.score -= 0.01;
        assert!(!measurer.is_left_better(&support1, &support2));
        assert!(!measurer.is_left_better(&support2, &support1));

        support2.score -= 0.01;
        assert!(!measurer.is_left_better(&support1, &support2));
        assert!(measurer.is_left_better(&support2, &support1));
    }
}
