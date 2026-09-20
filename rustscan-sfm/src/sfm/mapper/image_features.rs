use super::FeatureType;
use crate::colmap_image::load_colmap_grayscale_u8;
use crate::feature_extraction::FeatureMemoryPlan;
use crate::feature_matching_db::load_rgb_image_for_frame;
use crate::sift::{
    extract_sift_from_grayscale_u8, extract_sift_from_grayscale_u8_with_timing,
    SiftExtractionOptions,
};
use crate::task::SfmTaskControl;
use crate::types::ImageFrame;
use crate::wide::build_wide_descriptors;
use anyhow::{Context, Result};
use rayon::prelude::*;
use rustscan_slam::{FeatureExtractor, OrbExtractor};
use serde::Serialize;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Clone, Serialize)]
pub(super) struct FeatureImageTiming {
    pub image: String,
    pub grayscale_load_ms: f64,
    pub sift_ms: f64,
    pub sift_preparation_ms: f64,
    pub sift_input_conversion_ms: f64,
    pub sift_kernel_ms: f64,
    pub sift_backend_input_conversion_ms: f64,
    pub sift_backend_scale_space_ms: f64,
    pub sift_backend_detection_ms: f64,
    pub sift_backend_orientation_ms: f64,
    pub sift_backend_descriptor_ms: f64,
    pub sift_backend_output_assembly_ms: f64,
    pub sift_result_conversion_ms: f64,
    pub rgb_load_ms: f64,
    pub color_sampling_ms: f64,
    pub wide_descriptor_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct FeatureProfile {
    pub image_count: usize,
    pub wall_clock_span_ms: f64,
    pub p50_total_ms: f64,
    pub p95_total_ms: f64,
    pub image_timings: Vec<FeatureImageTiming>,
    pub slowest_images: Vec<FeatureImageTiming>,
}

pub(super) fn extract_frames(
    paths: &[PathBuf],
    max_features: usize,
    feature_type: FeatureType,
    sift_options: &SiftExtractionOptions,
    plan: Option<&FeatureMemoryPlan>,
    control: &SfmTaskControl,
) -> Result<(Vec<ImageFrame>, Option<FeatureProfile>)> {
    let profiling = std::env::var("RUSTSFM_PROFILE_FEATURES").is_ok_and(|value| value == "1");
    extract_frames_profiled(
        paths,
        max_features,
        feature_type,
        sift_options,
        plan,
        control,
        profiling,
    )
}

fn extract_frames_profiled(
    paths: &[PathBuf],
    max_features: usize,
    feature_type: FeatureType,
    sift_options: &SiftExtractionOptions,
    plan: Option<&FeatureMemoryPlan>,
    control: &SfmTaskControl,
    profiling: bool,
) -> Result<(Vec<ImageFrame>, Option<FeatureProfile>)> {
    let paths = plan.map_or(paths, FeatureMemoryPlan::paths);
    let sift_options = plan.map_or(sift_options, FeatureMemoryPlan::options);
    let batch_size = plan.map_or(paths.len().max(1), FeatureMemoryPlan::batch_size);
    control.checkpoint()?;
    let profile_start = profiling.then(Instant::now);
    let mut results = Vec::with_capacity(paths.len());
    for (batch_id, batch) in paths.chunks(batch_size).enumerate() {
        control.checkpoint()?;
        let batch_results = crate::execution::parallel(|| {
            batch
                .par_iter()
                .enumerate()
                .map(
                    |(offset, path)| -> Result<(ImageFrame, Option<FeatureImageTiming>)> {
                        control.checkpoint()?;
                        let id = batch_id * batch_size + offset;
                        let total_start = profiling.then(Instant::now);
                        let mut timing = profiling.then(|| FeatureImageTiming {
                            image: path.file_name().unwrap().to_string_lossy().to_string(),
                            grayscale_load_ms: 0.0,
                            sift_ms: 0.0,
                            sift_preparation_ms: 0.0,
                            sift_input_conversion_ms: 0.0,
                            sift_kernel_ms: 0.0,
                            sift_backend_input_conversion_ms: 0.0,
                            sift_backend_scale_space_ms: 0.0,
                            sift_backend_detection_ms: 0.0,
                            sift_backend_orientation_ms: 0.0,
                            sift_backend_descriptor_ms: 0.0,
                            sift_backend_output_assembly_ms: 0.0,
                            sift_result_conversion_ms: 0.0,
                            rgb_load_ms: 0.0,
                            color_sampling_ms: 0.0,
                            wide_descriptor_ms: 0.0,
                            total_ms: 0.0,
                        });
                        let (keypoints, descriptors, sift, width, height, colors) =
                            match feature_type {
                                FeatureType::Orb => {
                                    let rgb_start = profiling.then(Instant::now);
                                    let (rgb, width, height) = load_rgb_image_for_frame(path)?;
                                    if let (Some(timing), Some(rgb_start)) =
                                        (&mut timing, rgb_start)
                                    {
                                        timing.rgb_load_ms +=
                                            rgb_start.elapsed().as_secs_f64() * 1000.0;
                                    }
                                    let mut extractor = OrbExtractor::new(max_features);
                                    let (keypoints, descriptors) =
                                        extractor.detect_and_compute(&rgb, width, height).map_err(
                                            |e| anyhow::anyhow!("feature extraction failed: {e}"),
                                        )?;
                                    let color_start = profiling.then(Instant::now);
                                    let colors =
                                        sample_colors_from_rgb(&rgb, width, height, &keypoints);
                                    if let (Some(timing), Some(color_start)) =
                                        (&mut timing, color_start)
                                    {
                                        timing.color_sampling_ms +=
                                            color_start.elapsed().as_secs_f64() * 1000.0;
                                    }
                                    (
                                        keypoints,
                                        descriptors,
                                        Default::default(),
                                        width,
                                        height,
                                        colors,
                                    )
                                }
                                FeatureType::Sift => {
                                    let gray_start = profiling.then(Instant::now);
                                    let gray =
                                        load_colmap_grayscale_u8(path).with_context(|| {
                                            format!("failed to load {}", path.display())
                                        })?;
                                    if let (Some(timing), Some(gray_start)) =
                                        (&mut timing, gray_start)
                                    {
                                        timing.grayscale_load_ms +=
                                            gray_start.elapsed().as_secs_f64() * 1000.0;
                                    }
                                    let sift_start = profiling.then(Instant::now);
                                    let sift = if profiling {
                                        let (sift, sift_timing) =
                                            extract_sift_from_grayscale_u8_with_timing(
                                                &gray.data,
                                                gray.width,
                                                gray.height,
                                                sift_options,
                                            )?;
                                        if let Some(timing) = timing.as_mut() {
                                            timing.sift_preparation_ms +=
                                                sift_timing.preparation_ms;
                                            timing.sift_input_conversion_ms +=
                                                sift_timing.input_conversion_ms;
                                            timing.sift_kernel_ms += sift_timing.backend_kernel_ms;
                                            timing.sift_backend_input_conversion_ms +=
                                                sift_timing.backend_input_conversion_ms;
                                            timing.sift_backend_scale_space_ms +=
                                                sift_timing.backend_scale_space_ms;
                                            timing.sift_backend_detection_ms +=
                                                sift_timing.backend_detection_ms;
                                            timing.sift_backend_orientation_ms +=
                                                sift_timing.backend_orientation_ms;
                                            timing.sift_backend_descriptor_ms +=
                                                sift_timing.backend_descriptor_ms;
                                            timing.sift_backend_output_assembly_ms +=
                                                sift_timing.backend_output_assembly_ms;
                                            timing.sift_result_conversion_ms +=
                                                sift_timing.result_conversion_ms;
                                        }
                                        sift
                                    } else {
                                        extract_sift_from_grayscale_u8(
                                            &gray.data,
                                            gray.width,
                                            gray.height,
                                            sift_options,
                                        )?
                                    };
                                    if let (Some(timing), Some(sift_start)) =
                                        (&mut timing, sift_start)
                                    {
                                        timing.sift_ms +=
                                            sift_start.elapsed().as_secs_f64() * 1000.0;
                                    }
                                    let rgb_start = profiling.then(Instant::now);
                                    let (rgb, width, height) = load_rgb_image_for_frame(path)?;
                                    if let (Some(timing), Some(rgb_start)) =
                                        (&mut timing, rgb_start)
                                    {
                                        timing.rgb_load_ms +=
                                            rgb_start.elapsed().as_secs_f64() * 1000.0;
                                    }
                                    let color_start = profiling.then(Instant::now);
                                    let colors = sample_colors_from_rgb(
                                        &rgb,
                                        width,
                                        height,
                                        &sift.keypoints,
                                    );
                                    if let (Some(timing), Some(color_start)) =
                                        (&mut timing, color_start)
                                    {
                                        timing.color_sampling_ms +=
                                            color_start.elapsed().as_secs_f64() * 1000.0;
                                    }
                                    (
                                        sift.keypoints.clone(),
                                        rustscan_slam::Descriptors::new(),
                                        sift,
                                        width,
                                        height,
                                        colors,
                                    )
                                }
                            };
                        let gray_start = profiling.then(Instant::now);
                        let gray = load_colmap_grayscale_u8(path)
                            .with_context(|| format!("failed to load {}", path.display()))?;
                        if let (Some(timing), Some(gray_start)) = (&mut timing, gray_start) {
                            timing.grayscale_load_ms += gray_start.elapsed().as_secs_f64() * 1000.0;
                        }
                        let wide_start = profiling.then(Instant::now);
                        let gray_f32 = gray
                            .data
                            .iter()
                            .map(|value| *value as f32 / 255.0)
                            .collect::<Vec<_>>();
                        let wide_descriptors =
                            build_wide_descriptors(&gray_f32, gray.width, gray.height, &keypoints);
                        if let (Some(timing), Some(wide_start)) = (&mut timing, wide_start) {
                            timing.wide_descriptor_ms +=
                                wide_start.elapsed().as_secs_f64() * 1000.0;
                        }
                        let strong_feature_indices = strong_feature_indices(&keypoints, 1024);
                        if let (Some(timing), Some(total_start)) = (&mut timing, total_start) {
                            timing.total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
                        }
                        control.checkpoint()?;
                        Ok((
                            ImageFrame {
                                id,
                                name: path.file_name().unwrap().to_string_lossy().to_string(),
                                path: path.clone(),
                                width,
                                height,
                                keypoints,
                                descriptors,
                                sift,
                                wide_descriptors,
                                strong_feature_indices,
                                colors,
                            },
                            timing,
                        ))
                    },
                )
                .collect::<Result<Vec<_>>>()
        })?;
        results.extend(batch_results);
    }
    control.checkpoint()?;
    let (frames, timings): (Vec<_>, Vec<_>) = results.into_iter().unzip();
    let profile = profile_start.map(|start| {
        let mut timings = timings.into_iter().flatten().collect::<Vec<_>>();
        let mut totals = timings
            .iter()
            .map(|timing| timing.total_ms)
            .collect::<Vec<_>>();
        totals.sort_by(f64::total_cmp);
        let p50_total_ms = percentile(&totals, 0.50);
        let p95_total_ms = percentile(&totals, 0.95);
        let image_timings = timings.clone();
        timings.sort_by(|left, right| right.total_ms.total_cmp(&left.total_ms));
        timings.truncate(10);
        FeatureProfile {
            image_count: frames.len(),
            wall_clock_span_ms: start.elapsed().as_secs_f64() * 1000.0,
            p50_total_ms,
            p95_total_ms,
            image_timings,
            slowest_images: timings,
        }
    });
    Ok((frames, profile))
}

#[cfg(test)]
#[path = "image_features_tests.rs"]
mod tests;

fn percentile(sorted_values: &[f64], percentile: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let index = ((sorted_values.len() - 1) as f64 * percentile).round() as usize;
    sorted_values[index.min(sorted_values.len() - 1)]
}

fn sample_colors_from_rgb(
    rgb: &[u8],
    width: u32,
    height: u32,
    keypoints: &[rustscan_slam::KeyPoint],
) -> Vec<[u8; 3]> {
    keypoints
        .iter()
        .map(|kp| {
            let x = kp.x().round().clamp(0.0, (width.saturating_sub(1)) as f32) as u32;
            let y = kp.y().round().clamp(0.0, (height.saturating_sub(1)) as f32) as u32;
            let idx = ((y * width + x) * 3) as usize;
            if idx + 2 < rgb.len() {
                [rgb[idx], rgb[idx + 1], rgb[idx + 2]]
            } else {
                [0, 0, 0]
            }
        })
        .collect()
}

fn strong_feature_indices(keypoints: &[rustscan_slam::KeyPoint], limit: usize) -> Vec<usize> {
    let mut indices = (0..keypoints.len()).collect::<Vec<_>>();
    indices.sort_by(|&a, &b| {
        keypoints[b]
            .response
            .partial_cmp(&keypoints[a].response)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices.truncate(limit.min(indices.len()));
    indices
}
