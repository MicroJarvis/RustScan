use nalgebra::{SMatrix, SVector};
use rustslam::SE3;

pub type Mat3x6 = SMatrix<f64, 3, 6>;
pub type Mat3 = SMatrix<f64, 3, 3>;
pub type Vec3d = SVector<f64, 3>;
pub(crate) const POSE_PRIOR_JACOBIAN_EPS: f64 = 1.0e-5;

#[derive(Debug, Clone, PartialEq)]
pub struct BundleAdjustmentPosePrior {
    pub image: usize,
    pub position: [f64; 3],
    pub position_covariance: [f64; 9],
}

impl BundleAdjustmentPosePrior {
    pub fn new(image: usize, position: [f64; 3]) -> Self {
        Self {
            image,
            position,
            position_covariance: [0.0; 9],
        }
    }

    pub fn with_covariance(mut self, position_covariance: [f64; 9]) -> Self {
        self.position_covariance = position_covariance;
        self
    }
}

pub fn camera_center_world(pose: SE3) -> Vec3d {
    let center = pose.inverse().translation();
    Vec3d::new(center[0] as f64, center[1] as f64, center[2] as f64)
}

pub fn position_prior_information_matrix(
    position_covariance: &[f64; 9],
    fallback_stddev: f64,
) -> Mat3 {
    if is_valid_covariance(position_covariance) {
        let cov = Mat3::from_row_slice(position_covariance);
        if let Some(inv) = cov.try_inverse() {
            return inv;
        }
    }
    let stddev = fallback_stddev.max(1.0e-12);
    let weight = 1.0 / (stddev * stddev);
    Mat3::from_diagonal(&Vec3d::new(weight, weight, weight))
}

pub fn camera_center_pose_jacobian(pose: SE3) -> Mat3x6 {
    let mut jacobian = Mat3x6::zeros();
    for axis in 0..6 {
        let mut plus = [0.0; 6];
        let mut minus = [0.0; 6];
        plus[axis] = POSE_PRIOR_JACOBIAN_EPS;
        minus[axis] = -POSE_PRIOR_JACOBIAN_EPS;
        let plus_center = camera_center_world(apply_pose_delta_f64(pose, plus));
        let minus_center = camera_center_world(apply_pose_delta_f64(pose, minus));
        jacobian.set_column(
            axis,
            &((plus_center - minus_center) / (2.0 * POSE_PRIOR_JACOBIAN_EPS)),
        );
    }
    jacobian
}

fn is_valid_covariance(position_covariance: &[f64; 9]) -> bool {
    position_covariance.iter().all(|value| value.is_finite())
        && position_covariance[0] > 0.0
        && position_covariance[4] > 0.0
        && position_covariance[8] > 0.0
}

fn apply_pose_delta_f64(pose: SE3, delta: [f64; 6]) -> SE3 {
    let omega = glam::Vec3::new(delta[0] as f32, delta[1] as f32, delta[2] as f32);
    let angle = omega.length();
    let delta_rotation = if angle <= 1.0e-12 {
        glam::Quat::IDENTITY
    } else {
        glam::Quat::from_axis_angle(omega / angle, angle)
    };
    let delta_translation = glam::Vec3::new(delta[3] as f32, delta[4] as f32, delta[5] as f32);
    SE3::from_quat_translation(delta_rotation, delta_translation).compose(&pose)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    #[test]
    fn position_prior_fallback_uses_isotropic_information() {
        let info = position_prior_information_matrix(&[0.0; 9], 2.0);
        assert!((info[(0, 0)] - 0.25).abs() < 1.0e-12);
        assert!((info[(1, 1)] - 0.25).abs() < 1.0e-12);
    }

    #[test]
    fn camera_center_jacobian_is_nonzero_for_translated_pose() {
        let pose = SE3::from_quat_translation(Quat::IDENTITY, Vec3::new(1.0, 2.0, 3.0));
        let jacobian = camera_center_pose_jacobian(pose);
        assert!(jacobian.norm() > 1.0e-6);
    }
}
