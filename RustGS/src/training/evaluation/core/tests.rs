use super::{
    compute_gradient_sharpness_f32, compute_laplacian_sharpness_f32, select_evaluation_frames,
    summarize_psnr_samples, summarize_training_metrics, worst_frame_metrics, EvaluationDevice,
    EvaluationFrameMetric, FinalTrainingMetrics,
};
use crate::{Intrinsics, ScenePose, TrainingDataset, SE3};
use std::path::PathBuf;

#[test]
fn summarize_training_metrics_tracks_last_epoch_mean_and_last_step() {
    let history = [0.9f32, 0.8, 0.7, 0.6, 0.5];
    assert_eq!(
        summarize_training_metrics(&history, 2),
        FinalTrainingMetrics {
            final_loss: 0.55,
            final_step_loss: 0.5,
        }
    );
}

#[test]
fn summarize_psnr_samples_tracks_distribution() {
    let summary = summarize_psnr_samples(&[1.0, 2.0, 3.0, 4.0]);
    assert!((summary.mean_db - 2.5).abs() < 1e-6);
    assert!((summary.median_db - 2.5).abs() < 1e-6);
    assert_eq!(summary.min_db, 1.0);
    assert_eq!(summary.max_db, 4.0);
    assert!((summary.stddev_db - 1.118_034).abs() < 1e-5);
}

#[test]
fn evaluation_device_parses_gpu_aliases() {
    assert_eq!(
        "cpu".parse::<EvaluationDevice>().unwrap(),
        EvaluationDevice::Cpu
    );
    assert_eq!(
        "gpu".parse::<EvaluationDevice>().unwrap(),
        EvaluationDevice::Gpu
    );
    assert_eq!(
        "wgpu".parse::<EvaluationDevice>().unwrap(),
        EvaluationDevice::Gpu
    );
    assert_eq!(
        "metal".parse::<EvaluationDevice>().unwrap(),
        EvaluationDevice::Gpu
    );
}

#[test]
fn sharpness_metrics_detect_edges() {
    let flat = vec![0.5f32; 4 * 4 * 3];
    let mut edge = vec![0.0f32; 4 * 4 * 3];
    for y in 0..4 {
        for x in 2..4 {
            let base = (y * 4 + x) * 3;
            edge[base] = 1.0;
            edge[base + 1] = 1.0;
            edge[base + 2] = 1.0;
        }
    }

    assert_eq!(compute_gradient_sharpness_f32(&flat, 4, 4), 0.0);
    assert!(compute_gradient_sharpness_f32(&edge, 4, 4) > 0.0);
    assert_eq!(compute_laplacian_sharpness_f32(&flat, 4, 4), 0.0);
    assert!(compute_laplacian_sharpness_f32(&edge, 4, 4) > 0.0);
}

#[test]
fn worst_frame_metrics_returns_low_psnr_prefix() {
    let metrics = vec![
        EvaluationFrameMetric {
            dataset_index: 0,
            frame_id: 0,
            psnr_db: 9.0,
            sharpness_grad_ratio: 0.9,
            sharpness_lap_ratio: 0.7,
            image_path: PathBuf::from("a.png"),
        },
        EvaluationFrameMetric {
            dataset_index: 1,
            frame_id: 1,
            psnr_db: 3.0,
            sharpness_grad_ratio: 0.9,
            sharpness_lap_ratio: 0.7,
            image_path: PathBuf::from("b.png"),
        },
        EvaluationFrameMetric {
            dataset_index: 2,
            frame_id: 2,
            psnr_db: 6.0,
            sharpness_grad_ratio: 0.9,
            sharpness_lap_ratio: 0.7,
            image_path: PathBuf::from("c.png"),
        },
    ];
    let worst = worst_frame_metrics(&metrics, 2);
    assert_eq!(worst.len(), 2);
    assert_eq!(worst[0].frame_id, 1);
    assert_eq!(worst[1].frame_id, 2);
}

#[test]
fn select_evaluation_frames_copies_initial_points_and_stride() {
    let mut dataset = TrainingDataset::new(Intrinsics::from_focal(500.0, 32, 32));
    dataset.add_point([0.0, 0.0, 0.0], Some([1.0, 0.0, 0.0]));
    for idx in 0..6 {
        dataset.add_pose(ScenePose::new(
            idx as u64,
            PathBuf::from(format!("frame-{idx}.png")),
            SE3::identity(),
            idx as f64,
        ));
    }

    let selected = select_evaluation_frames(&dataset, 5, 2);
    assert_eq!(selected.initial_points.len(), 1);
    assert_eq!(selected.poses.len(), 3);
    assert_eq!(selected.poses[0].frame_id, 0);
    assert_eq!(selected.poses[1].frame_id, 2);
    assert_eq!(selected.poses[2].frame_id, 4);
}

#[cfg(feature = "gpu")]
#[test]
fn splat_evaluation_renderer_gpu_renders_rgb_frame() {
    let splats = test_gpu_splats();
    let camera = crate::GaussianCamera::new(Intrinsics::from_focal(500.0, 32, 32), SE3::identity());
    let mut renderer = super::SplatEvaluationRenderer::new(
        32,
        32,
        EvaluationDevice::Gpu,
        crate::DEFAULT_RASTER_COV_BLUR,
    )
    .expect("gpu renderer");

    let rgb = renderer.render(&splats, &camera).expect("render frame");

    assert_eq!(rgb.len(), 32 * 32 * 3);
    assert!(rgb.iter().any(|value| *value > 0.0));
}

#[cfg(feature = "gpu")]
#[test]
fn splat_evaluation_renderer_gpu_renders_rgba_frame() {
    let splats = test_gpu_splats();
    let camera = crate::GaussianCamera::new(Intrinsics::from_focal(500.0, 32, 32), SE3::identity());
    let mut renderer = super::SplatEvaluationRenderer::new(
        32,
        32,
        EvaluationDevice::Gpu,
        crate::DEFAULT_RASTER_COV_BLUR,
    )
    .expect("gpu renderer");

    let rgba = renderer
        .render_rgba_f32(&splats, &camera)
        .expect("render frame");

    assert_eq!(rgba.len(), 32 * 32 * 4);
    assert!(rgba.iter().any(|value| *value > 0.0));
}

#[cfg(feature = "gpu")]
fn test_gpu_splats() -> crate::HostSplats {
    crate::HostSplats::from_components(
        vec![0.0, 0.0, 2.0],
        vec![0.2f32.ln(), 0.2f32.ln(), 0.2f32.ln()],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0],
        [1.0, 0.5, 0.25].map(crate::sh::rgb_to_sh0_value).into(),
        0,
    )
    .expect("valid test splats")
}
