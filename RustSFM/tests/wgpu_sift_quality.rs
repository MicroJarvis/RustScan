#![cfg(feature = "gpu-wgpu")]

use anyhow::Result;
use rustsfm::gpu::{WgpuContext, WgpuSiftExtractor};
use rustsfm::sift::{extract_sift_from_grayscale_u8, SiftExtractionOptions, SiftFeatures};

#[test]
fn gpu_sift_preserves_textured_geometric_content() -> Result<()> {
    let Some(context) = WgpuContext::try_new_optional()? else {
        eprintln!("skipping GPU quality test: no compatible adapter");
        return Ok(());
    };
    let gray = textured_fixture(512, 384);
    let cpu_options = SiftExtractionOptions {
        max_num_features: 1024,
        ..Default::default()
    };
    let gpu_options = SiftExtractionOptions {
        use_gpu: true,
        ..cpu_options.clone()
    };
    let cpu = extract_sift_from_grayscale_u8(&gray, 512, 384, &cpu_options)?;
    let gpu = WgpuSiftExtractor::from_context(context)?.extract_grayscale(
        &gray,
        512,
        384,
        &gpu_options,
    )?;
    assert!(
        gpu.keypoints.len() >= cpu.keypoints.len() / 3,
        "GPU keypoints={} CPU keypoints={}",
        gpu.keypoints.len(),
        cpu.keypoints.len()
    );
    assert!(
        gpu.keypoints.len() <= cpu.keypoints.len() * 3 + 1,
        "GPU keypoints={} CPU keypoints={}",
        gpu.keypoints.len(),
        cpu.keypoints.len()
    );
    assert!(
        nearest_keypoint_repeatability(&cpu, &gpu, 2.0) >= 0.55,
        "GPU/CPU repeatability was below threshold"
    );
    Ok(())
}

fn textured_fixture(width: u32, height: u32) -> Vec<u8> {
    (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| {
                let checker = if ((x / 16) + (y / 16)) % 2 == 0 {
                    32.0
                } else {
                    -32.0
                };
                let wave = ((x as f32 * 0.071).sin() + (y as f32 * 0.053).cos()) * 18.0;
                (128.0 + checker + wave).round().clamp(0.0, 255.0) as u8
            })
        })
        .collect()
}

fn nearest_keypoint_repeatability(
    left: &SiftFeatures,
    right: &SiftFeatures,
    radius_px: f32,
) -> f32 {
    if left.keypoints.is_empty() || right.keypoints.is_empty() {
        return 0.0;
    }
    let radius2 = radius_px * radius_px;
    let matched = left
        .keypoints
        .iter()
        .filter(|left_point| {
            right.keypoints.iter().any(|right_point| {
                let dx = left_point.x() - right_point.x();
                let dy = left_point.y() - right_point.y();
                dx * dx + dy * dy <= radius2
            })
        })
        .count();
    matched as f32 / left.keypoints.len() as f32
}
