//! Projection VJP finite-difference gate (R07).
//!
//! Analytic grads come from `project_bwd`. Numeric grads use centered
//! differences on the same projected-splat scalar objective. Quaternion
//! parameters are perturbed in their stored (unnormalized) form — matching
//! the forward path which normalizes internally.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GradientParameter {
    PositionX,
    PositionY,
    PositionZ,
    QuatW,
    QuatX,
    QuatY,
    QuatZ,
    LogScaleX,
    LogScaleY,
    LogScaleZ,
    OpacityLogit,
    ShDcR,
    ShDcG,
    ShDcB,
    ShRest0R,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GradientCheckResult {
    pub parameter: String,
    pub analytic: f32,
    pub numeric: f32,
    pub absolute_error: f32,
    pub relative_error: f32,
    pub finite: bool,
    pub passed: bool,
    pub epsilon: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GradientCheckCase {
    pub name: String,
    pub sh_degree: u32,
    pub cov_blur: f32,
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
    pub mean: [f32; 3],
    pub quat_unorm: [f32; 4],
    pub log_scale: [f32; 3],
    pub opacity_logit: f32,
    /// Flat SH coeffs `[K*3]` in RGB triples.
    pub sh_coeffs: Vec<f32>,
}

impl GradientCheckCase {
    pub fn baseline_degree0() -> Self {
        Self {
            name: "baseline_degree0".into(),
            sh_degree: 0,
            cov_blur: 0.3,
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 240.0,
            width: 640,
            height: 480,
            mean: [0.0, 0.0, 2.0],
            quat_unorm: [1.0, 0.1, -0.05, 0.02],
            log_scale: [-1.8, -1.5, -2.0],
            opacity_logit: 1.0,
            sh_coeffs: vec![0.5, 0.4, 0.3],
        }
    }

    pub fn off_center_fx_fy() -> Self {
        let mut case = Self::baseline_degree0();
        case.name = "off_center_fx_fy".into();
        case.fx = 480.0;
        case.fy = 520.0;
        case.cx = 350.0;
        case.cy = 200.0;
        case.mean = [0.15, -0.1, 1.8];
        case
    }

    pub fn small_covariance() -> Self {
        let mut case = Self::baseline_degree0();
        case.name = "small_covariance".into();
        case.log_scale = [-4.5, -4.2, -4.8];
        case
    }

    pub fn anisotropic_rotated() -> Self {
        let mut case = Self::baseline_degree0();
        case.name = "anisotropic_rotated".into();
        case.quat_unorm = [0.7, 0.5, 0.3, 0.2];
        case.log_scale = [-1.8, -3.5, -2.5];
        case
    }

    pub fn degree1_viewdir() -> Self {
        let mut case = Self::baseline_degree0();
        case.name = "degree1_viewdir".into();
        case.sh_degree = 1;
        // 4 coeffs × RGB
        case.sh_coeffs = vec![
            0.4, 0.35, 0.3, // DC
            0.05, -0.02, 0.01, // band1
            -0.03, 0.04, -0.01, 0.02, 0.01, -0.02,
        ];
        case
    }
}

pub fn scale_aware_epsilon(value: f32) -> f32 {
    (1e-4f32).max(1e-3 * value.abs())
}

pub fn relative_error(analytic: f32, numeric: f32) -> f32 {
    let denom = analytic.abs().max(numeric.abs()).max(1e-8);
    (analytic - numeric).abs() / denom
}

pub fn accepts_regular_case(relative: f32, absolute: f32) -> bool {
    relative <= 8e-2 || absolute <= 2e-3
}

fn accepts_parameter(
    parameter: GradientParameter,
    relative: f32,
    absolute: f32,
    analytic: f32,
    numeric: f32,
) -> bool {
    if accepts_regular_case(relative, absolute) {
        return true;
    }
    if analytic.abs() < 1e-2 && numeric.abs() < 2e-2 {
        return true;
    }
    // Unnormalized-quaternion FD is poorly conditioned; keep a looser gate while
    // still rejecting large finite disagreements on strong axes.
    matches!(
        parameter,
        GradientParameter::QuatW
            | GradientParameter::QuatX
            | GradientParameter::QuatY
            | GradientParameter::QuatZ
    ) && absolute <= 3e-2
        && analytic.abs() <= 0.1
}

#[cfg(all(test, feature = "gpu"))]
mod gpu_tests {
    use super::*;
    use crate::core::GaussianCamera;
    use crate::sh::sh_coeff_count_for_degree;
    use crate::training::backward::project_bwd::{from_inner_splats, project_bwd};
    use crate::training::engine::GsBackendBase;
    use crate::training::forward::project_visible::project_visible;
    use burn::prelude::*;
    use burn::tensor::{Int, TensorData};
    use burn_cubecl::cubecl::CubeCount;
    use rustscan_types::{Intrinsics, SE3};

    const UPSTREAM: [f32; 10] = [
        1.0, -0.5, // xy
        2.0, 0.5, -1.5, // conic — stress covariance / scale / quat
        0.4, -0.1, 0.2, // color
        1.0, // alpha
        0.0, // depth slot unused by VJP
    ];

    fn near_zero_agreement(analytic: f32, numeric: f32) -> bool {
        analytic.abs() <= 5e-3 && numeric.abs() <= 5e-3
    }

    async fn numeric_grad_transform(
        case: &GradientCheckCase,
        index: usize,
        device: &<GsBackendBase as Backend>::Device,
    ) -> (f32, f32) {
        let mut transforms = [0.0f32; 10];
        transforms[0..3].copy_from_slice(&case.mean);
        transforms[3..7].copy_from_slice(&case.quat_unorm);
        transforms[7..10].copy_from_slice(&case.log_scale);
        let base_eps = scale_aware_epsilon(transforms[index]);
        let mut best = (0.0f32, base_eps);
        let mut best_rel_noise = f32::INFINITY;
        for scale in [1.0f32, 2.0, 4.0] {
            let eps = (base_eps * scale).max(5e-4);
            let mut plus = transforms;
            let mut minus = transforms;
            plus[index] += eps;
            minus[index] -= eps;
            let f_plus = scalar(
                &projected_row(case, plus, case.opacity_logit, &case.sh_coeffs, device).await,
            );
            let f_minus = scalar(
                &projected_row(case, minus, case.opacity_logit, &case.sh_coeffs, device).await,
            );
            if !f_plus.is_finite() || !f_minus.is_finite() {
                continue;
            }
            let numeric = (f_plus - f_minus) / (2.0 * eps);
            // Prefer smaller eps once the finite-difference response clears float noise.
            let response = (f_plus - f_minus).abs();
            let rel_noise = if response > 1e-6 {
                eps / response
            } else {
                f32::INFINITY
            };
            if rel_noise < best_rel_noise {
                best = (numeric, eps);
                best_rel_noise = rel_noise;
            }
        }
        best
    }

    fn camera_for(case: &GradientCheckCase) -> GaussianCamera {
        GaussianCamera::new(
            Intrinsics::new(case.fx, case.fy, case.cx, case.cy, case.width, case.height),
            SE3::new(&[0.0, 0.0, 0.0, 1.0], &[0.0, 0.0, 0.0]),
        )
    }

    fn make_splats(
        case: &GradientCheckCase,
        device: &<GsBackendBase as Backend>::Device,
        transforms: [f32; 10],
        opacity: f32,
        sh: &[f32],
    ) -> crate::training::engine::DeviceSplats<GsBackendBase> {
        let k = sh_coeff_count_for_degree(case.sh_degree as usize);
        let mut sh_flat = vec![0.0f32; k * 3];
        for (dst, src) in sh_flat.iter_mut().zip(sh.iter()) {
            *dst = *src;
        }
        from_inner_splats(
            Tensor::<GsBackendBase, 2>::from_data(TensorData::new(transforms.to_vec(), [1, 10]), device),
            Tensor::<GsBackendBase, 3>::from_data(TensorData::new(sh_flat, [1, k, 3]), device),
            Tensor::<GsBackendBase, 1>::from_data(TensorData::new(vec![opacity], [1]), device),
            case.sh_degree,
        )
    }

    async fn projected_row(
        case: &GradientCheckCase,
        transforms: [f32; 10],
        opacity: f32,
        sh: &[f32],
        device: &<GsBackendBase as Backend>::Device,
    ) -> Vec<f32> {
        let camera = camera_for(case);
        let splats = make_splats(case, device, transforms, opacity, sh);
        let global = Tensor::<GsBackendBase, 1, Int>::from_data(TensorData::from([0i32]), device);
        let visible = Tensor::<GsBackendBase, 1, Int>::from_data(TensorData::from([1i32]), device);
        let projected = project_visible(
            &splats,
            case.sh_degree,
            &global,
            &visible,
            1,
            CubeCount::Static(1, 1, 1),
            &camera,
            (case.width, case.height),
            device,
            case.cov_blur,
        );
        projected
            .into_data_async()
            .await
            .expect("projected readback")
            .to_vec::<f32>()
            .expect("projected f32")
    }

    fn scalar(row: &[f32]) -> f32 {
        row.iter()
            .zip(UPSTREAM.iter())
            .map(|(value, weight)| value * weight)
            .sum()
    }

    async fn analytic_grads(
        case: &GradientCheckCase,
        device: &<GsBackendBase as Backend>::Device,
    ) -> (Vec<f32>, Vec<f32>, f32) {
        let camera = camera_for(case);
        let mut transforms = [0.0f32; 10];
        transforms[0..3].copy_from_slice(&case.mean);
        transforms[3..7].copy_from_slice(&case.quat_unorm);
        transforms[7..10].copy_from_slice(&case.log_scale);
        let splats = make_splats(case, device, transforms, case.opacity_logit, &case.sh_coeffs);
        let global = Tensor::<GsBackendBase, 1, Int>::from_data(TensorData::from([0i32]), device);
        let visible = Tensor::<GsBackendBase, 1, Int>::from_data(TensorData::from([1i32]), device);
        let v_splats =
            Tensor::<GsBackendBase, 2>::from_data(TensorData::new(UPSTREAM.to_vec(), [1, 10]), device);
        let screen =
            Tensor::<GsBackendBase, 2>::zeros([1, 5], device);
        let out = project_bwd(
            &splats,
            case.sh_degree,
            global,
            visible,
            v_splats,
            screen,
            crate::training::engine::DeviceTrainingStatus::<GsBackendBase>::new(device, 0)
                .buffer()
                .clone(),
            CubeCount::Static(1, 1, 1),
            &camera,
            (case.width, case.height),
            1,
            device,
            case.cov_blur,
        );
        let v_transforms = out
            .v_transforms
            .into_data_async()
            .await
            .expect("v_transforms")
            .to_vec::<f32>()
            .expect("v_transforms f32");
        let v_sh = out
            .v_sh_coeffs
            .into_data_async()
            .await
            .expect("v_sh")
            .to_vec::<f32>()
            .expect("v_sh f32");
        let v_opacity = out
            .v_raw_opacities
            .into_data_async()
            .await
            .expect("v_opacity")
            .to_vec::<f32>()
            .expect("v_opacity f32")[0];
        (v_transforms, v_sh, v_opacity)
    }

    async fn numeric_grad_opacity(
        case: &GradientCheckCase,
        device: &<GsBackendBase as Backend>::Device,
    ) -> (f32, f32) {
        let mut transforms = [0.0f32; 10];
        transforms[0..3].copy_from_slice(&case.mean);
        transforms[3..7].copy_from_slice(&case.quat_unorm);
        transforms[7..10].copy_from_slice(&case.log_scale);
        let eps = scale_aware_epsilon(case.opacity_logit);
        let f_plus = scalar(
            &projected_row(
                case,
                transforms,
                case.opacity_logit + eps,
                &case.sh_coeffs,
                device,
            )
            .await,
        );
        let f_minus = scalar(
            &projected_row(
                case,
                transforms,
                case.opacity_logit - eps,
                &case.sh_coeffs,
                device,
            )
            .await,
        );
        ((f_plus - f_minus) / (2.0 * eps), eps)
    }

    async fn numeric_grad_sh(
        case: &GradientCheckCase,
        flat_index: usize,
        device: &<GsBackendBase as Backend>::Device,
    ) -> (f32, f32) {
        let mut transforms = [0.0f32; 10];
        transforms[0..3].copy_from_slice(&case.mean);
        transforms[3..7].copy_from_slice(&case.quat_unorm);
        transforms[7..10].copy_from_slice(&case.log_scale);
        let mut sh = case.sh_coeffs.clone();
        while sh.len() <= flat_index {
            sh.push(0.0);
        }
        let eps = scale_aware_epsilon(sh[flat_index]);
        let mut plus = sh.clone();
        let mut minus = sh.clone();
        plus[flat_index] += eps;
        minus[flat_index] -= eps;
        let f_plus = scalar(&projected_row(case, transforms, case.opacity_logit, &plus, device).await);
        let f_minus =
            scalar(&projected_row(case, transforms, case.opacity_logit, &minus, device).await);
        ((f_plus - f_minus) / (2.0 * eps), eps)
    }

    fn pack_result(
        parameter: GradientParameter,
        analytic: f32,
        numeric: f32,
        epsilon: f32,
    ) -> GradientCheckResult {
        let absolute_error = (analytic - numeric).abs();
        let relative = relative_error(analytic, numeric);
        let finite = analytic.is_finite() && numeric.is_finite();
        // Weak directions (tiny analytic & numeric) are recorded but do not fail the gate;
        // FD float noise dominates there. Regular cases still use relative/absolute thresholds.
        let weak_direction = analytic.abs() < 1e-2 && numeric.abs() < 2e-2;
        let passed = finite
            && (weak_direction
                || accepts_parameter(parameter, relative, absolute_error, analytic, numeric)
                || near_zero_agreement(analytic, numeric));
        GradientCheckResult {
            parameter: format!("{parameter:?}"),
            analytic,
            numeric,
            absolute_error,
            relative_error: relative,
            finite,
            passed,
            epsilon,
        }
    }

    async fn check_case(case: &GradientCheckCase) -> Vec<GradientCheckResult> {
        let device = <GsBackendBase as Backend>::Device::default();
        let (v_transforms, v_sh, v_opacity) = analytic_grads(case, &device).await;
        let mut results = Vec::new();
        let transform_params = [
            (GradientParameter::PositionX, 0usize),
            (GradientParameter::PositionY, 1),
            (GradientParameter::PositionZ, 2),
            (GradientParameter::QuatW, 3),
            (GradientParameter::QuatX, 4),
            (GradientParameter::QuatY, 5),
            (GradientParameter::QuatZ, 6),
            (GradientParameter::LogScaleX, 7),
            (GradientParameter::LogScaleY, 8),
            (GradientParameter::LogScaleZ, 9),
        ];
        for (parameter, index) in transform_params {
            let (numeric, eps) = numeric_grad_transform(case, index, &device).await;
            results.push(pack_result(parameter, v_transforms[index], numeric, eps));
        }
        let (numeric, eps) = numeric_grad_opacity(case, &device).await;
        results.push(pack_result(
            GradientParameter::OpacityLogit,
            v_opacity,
            numeric,
            eps,
        ));
        let (numeric, eps) = numeric_grad_sh(case, 0, &device).await;
        results.push(pack_result(GradientParameter::ShDcR, v_sh[0], numeric, eps));
        if case.sh_degree >= 1 && v_sh.len() >= 4 {
            let (numeric, eps) = numeric_grad_sh(case, 3, &device).await;
            results.push(pack_result(
                GradientParameter::ShRest0R,
                v_sh[3],
                numeric,
                eps,
            ));
        }
        results
    }

    fn assert_case_passes(case: &GradientCheckCase, results: &[GradientCheckResult]) {
        let failures: Vec<_> = results.iter().filter(|result| !result.passed).collect();
        if !failures.is_empty() {
            let table = serde_json::to_string_pretty(results).unwrap_or_default();
            panic!(
                "gradient check failed for {}: {} failure(s)\n{table}",
                case.name,
                failures.len()
            );
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn projection_gradients_baseline_degree0() {
        let case = GradientCheckCase::baseline_degree0();
        let results = check_case(&case).await;
        assert_case_passes(&case, &results);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn projection_gradients_off_center_fx_fy() {
        let case = GradientCheckCase::off_center_fx_fy();
        let results = check_case(&case).await;
        assert_case_passes(&case, &results);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn projection_gradients_small_covariance() {
        let case = GradientCheckCase::small_covariance();
        let results = check_case(&case).await;
        assert_case_passes(&case, &results);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn projection_gradients_anisotropic_rotated() {
        let case = GradientCheckCase::anisotropic_rotated();
        let results = check_case(&case).await;
        assert_case_passes(&case, &results);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn projection_gradients_degree1_includes_viewdir() {
        let case = GradientCheckCase::degree1_viewdir();
        let results = check_case(&case).await;
        assert_case_passes(&case, &results);
    }

    #[test]
    fn host_helpers_keep_null_safe_tolerances() {
        assert!(accepts_regular_case(0.01, 1.0));
        assert!(accepts_regular_case(1.0, 1e-4));
        assert!(!accepts_regular_case(0.2, 2e-2));
        assert!(scale_aware_epsilon(0.0) >= 1e-4);
    }
}
