use nalgebra::{DMatrix, SMatrix};
use std::collections::HashMap;

pub type Mat6 = SMatrix<f64, 6, 6>;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BundleAdjustmentCovariance {
    pub pose_blocks: HashMap<usize, Mat6>,
}

#[derive(Debug, Clone, Copy)]
pub struct CovariancePoseBlock {
    pub image: usize,
    pub offset: usize,
    pub dim: usize,
}

const MAX_FULL_COVARIANCE_DIM: usize = 192;

pub fn compute_pose_covariances(
    hessian: &DMatrix<f64>,
    blocks: &[CovariancePoseBlock],
) -> Option<BundleAdjustmentCovariance> {
    if blocks.is_empty() || hessian.nrows() != hessian.ncols() {
        return None;
    }
    let n = hessian.nrows();
    let mut pose_blocks = HashMap::new();
    if n <= MAX_FULL_COVARIANCE_DIM {
        let inverse = hessian.clone().try_inverse()?;
        for block in blocks {
            if block.dim != 6 {
                continue;
            }
            let mut cov = Mat6::zeros();
            for row in 0..6 {
                for col in 0..6 {
                    cov[(row, col)] = inverse[(block.offset + row, block.offset + col)];
                }
            }
            pose_blocks.insert(block.image, cov);
        }
    } else {
        for block in blocks {
            if block.dim != 6 {
                continue;
            }
            let mut sub = DMatrix::<f64>::zeros(block.dim, block.dim);
            for row in 0..block.dim {
                for col in 0..block.dim {
                    sub[(row, col)] = hessian[(block.offset + row, block.offset + col)];
                }
            }
            let inverse = sub.try_inverse()?;
            let mut cov = Mat6::zeros();
            for row in 0..block.dim {
                for col in 0..block.dim {
                    cov[(row, col)] = inverse[(row, col)];
                }
            }
            pose_blocks.insert(block.image, cov);
        }
    }
    Some(BundleAdjustmentCovariance { pose_blocks })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_covariance_recovers_pose_block_from_inverse_hessian() {
        let hessian = DMatrix::<f64>::from_diagonal(&nalgebra::DVector::from_element(6, 2.0));
        let blocks = vec![CovariancePoseBlock {
            image: 0,
            offset: 0,
            dim: 6,
        }];
        let covariance = compute_pose_covariances(&hessian, &blocks).expect("covariance");
        let block = covariance.pose_blocks.get(&0).expect("pose block");
        assert!((block[(0, 0)] - 0.5).abs() < 1.0e-12);
    }
}
