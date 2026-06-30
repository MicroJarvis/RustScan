use super::FeatureType;
use crate::colmap_image::load_colmap_grayscale_u8;
use crate::feature_matching_db::load_rgb_image_for_frame;
use crate::sift::{extract_sift_from_grayscale_u8, SiftExtractionOptions};
use crate::types::ImageFrame;
use crate::wide::build_wide_descriptors;
use anyhow::{Context, Result};
use rayon::prelude::*;
use rustslam::{FeatureExtractor, OrbExtractor};
use std::path::PathBuf;

pub(super) fn extract_frames(
    paths: &[PathBuf],
    max_features: usize,
    feature_type: FeatureType,
    sift_options: &SiftExtractionOptions,
) -> Result<Vec<ImageFrame>> {
    paths
        .par_iter()
        .enumerate()
        .map(|(id, path)| -> Result<ImageFrame> {
            let (keypoints, descriptors, sift, width, height, colors) = match feature_type {
                FeatureType::Orb => {
                    let (rgb, width, height) = load_rgb_image_for_frame(path)?;
                    let mut extractor = OrbExtractor::new(max_features);
                    let (keypoints, descriptors) = extractor
                        .detect_and_compute(&rgb, width, height)
                        .map_err(|e| anyhow::anyhow!("feature extraction failed: {e}"))?;
                    let colors = sample_colors_from_rgb(&rgb, width, height, &keypoints);
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
                    let gray = load_colmap_grayscale_u8(path)
                        .with_context(|| format!("failed to load {}", path.display()))?;
                    let sift = extract_sift_from_grayscale_u8(
                        &gray.data,
                        gray.width,
                        gray.height,
                        sift_options,
                    )?;
                    let (rgb, width, height) = load_rgb_image_for_frame(path)?;
                    let colors = sample_colors_from_rgb(&rgb, width, height, &sift.keypoints);
                    (
                        sift.keypoints.clone(),
                        rustslam::Descriptors::new(),
                        sift,
                        width,
                        height,
                        colors,
                    )
                }
            };
            let gray = load_colmap_grayscale_u8(path)
                .with_context(|| format!("failed to load {}", path.display()))?;
            let gray_f32 = gray
                .data
                .iter()
                .map(|value| *value as f32 / 255.0)
                .collect::<Vec<_>>();
            let wide_descriptors =
                build_wide_descriptors(&gray_f32, gray.width, gray.height, &keypoints);
            let strong_feature_indices = strong_feature_indices(&keypoints, 1024);
            Ok(ImageFrame {
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
            })
        })
        .collect()
}

fn sample_colors_from_rgb(
    rgb: &[u8],
    width: u32,
    height: u32,
    keypoints: &[rustslam::KeyPoint],
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

fn strong_feature_indices(keypoints: &[rustslam::KeyPoint], limit: usize) -> Vec<usize> {
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
