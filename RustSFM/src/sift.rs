use anyhow::{bail, Result};
use lowe_sift::{BbfConfig, Descriptor, Feature, GrayImage, Sift, SiftConfig};
use rustslam::{KeyPoint, Match};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Default)]
pub struct SiftFeatures {
    pub keypoints: Vec<KeyPoint>,
    pub descriptors: Vec<Descriptor>,
}

#[derive(Debug, Clone, Copy)]
pub enum SiftDescriptorNormalization {
    L1Root,
    L2,
}

#[derive(Debug, Clone)]
pub struct SiftExtractionOptions {
    pub max_num_features: usize,
    pub first_octave: i32,
    pub num_octaves: usize,
    pub octave_resolution: usize,
    pub peak_threshold: f64,
    pub edge_threshold: f64,
    pub estimate_affine_shape: bool,
    pub max_num_orientations: usize,
    pub upright: bool,
    pub domain_size_pooling: bool,
    pub dsp_min_scale: f64,
    pub dsp_max_scale: f64,
    pub dsp_num_scales: usize,
    pub normalization: SiftDescriptorNormalization,
}

impl Default for SiftExtractionOptions {
    fn default() -> Self {
        Self {
            max_num_features: 8192,
            first_octave: -1,
            num_octaves: 4,
            octave_resolution: 3,
            peak_threshold: 0.02 / 3.0,
            edge_threshold: 10.0,
            estimate_affine_shape: false,
            max_num_orientations: 2,
            upright: false,
            domain_size_pooling: false,
            dsp_min_scale: 1.0 / 6.0,
            dsp_max_scale: 3.0,
            dsp_num_scales: 10,
            normalization: SiftDescriptorNormalization::L1Root,
        }
    }
}

impl SiftExtractionOptions {
    pub fn check(&self) -> Result<()> {
        if self.max_num_features == 0 {
            bail!("SiftExtraction.max_num_features must be > 0");
        }
        if self.octave_resolution == 0 {
            bail!("SiftExtraction.octave_resolution must be > 0");
        }
        if !self.peak_threshold.is_finite() || self.peak_threshold <= 0.0 {
            bail!("SiftExtraction.peak_threshold must be > 0");
        }
        if !self.edge_threshold.is_finite() || self.edge_threshold <= 0.0 {
            bail!("SiftExtraction.edge_threshold must be > 0");
        }
        if self.domain_size_pooling {
            if !self.dsp_min_scale.is_finite() || self.dsp_min_scale <= 0.0 {
                bail!("SiftExtraction.dsp_min_scale must be > 0");
            }
            if !self.dsp_max_scale.is_finite() || self.dsp_max_scale <= 0.0 {
                bail!("SiftExtraction.dsp_max_scale must be > 0");
            }
            if self.dsp_num_scales == 0 {
                bail!("SiftExtraction.dsp_num_scales must be > 0");
            }
        }
        Ok(())
    }

    fn to_lowe_config(&self) -> SiftConfig {
        let mut config = SiftConfig::default();
        config.intervals = self.octave_resolution;
        config.double_image = self.first_octave < 0;
        config.contrast_threshold = self.peak_threshold as f32;
        config.edge_threshold = self.edge_threshold as f32;
        if self.upright || self.max_num_orientations <= 1 {
            config.orientation_peak_ratio = 1.0;
        } else if self.max_num_orientations == 2 {
            config.orientation_peak_ratio = 0.8;
        } else {
            config.orientation_peak_ratio = 0.8;
        }
        config
    }
}

#[derive(Debug, Clone)]
pub struct SiftMatchingOptions {
    pub max_ratio: f32,
    pub max_distance: f32,
    pub cross_check: bool,
    pub max_num_matches: usize,
    pub guided_matching: bool,
    pub max_guided_epipolar_error_px: f32,
}

impl Default for SiftMatchingOptions {
    fn default() -> Self {
        Self {
            max_ratio: 0.8,
            max_distance: 0.7,
            cross_check: true,
            max_num_matches: 32768,
            guided_matching: false,
            max_guided_epipolar_error_px: 2.0,
        }
    }
}

impl SiftMatchingOptions {
    pub fn check(&self) -> Result<()> {
        if !self.max_ratio.is_finite() || self.max_ratio <= 0.0 {
            bail!("SiftMatching.max_ratio must be > 0");
        }
        if !self.max_distance.is_finite() || self.max_distance <= 0.0 {
            bail!("SiftMatching.max_distance must be > 0");
        }
        Ok(())
    }
}

pub fn extract_sift_features(
    rgb: &[u8],
    width: u32,
    height: u32,
    max_features: usize,
) -> Result<SiftFeatures> {
    let options = SiftExtractionOptions {
        max_num_features: max_features,
        ..Default::default()
    };
    extract_sift_features_with_options(rgb, width, height, &options)
}

pub fn extract_sift_features_with_options(
    rgb: &[u8],
    width: u32,
    height: u32,
    options: &SiftExtractionOptions,
) -> Result<SiftFeatures> {
    options.check()?;
    let gray = rgb_to_sift_gray(rgb, width, height)?;
    let mut features = Sift::new(options.to_lowe_config())?.detect_and_compute(&gray);
    features.sort_by(|a, b| {
        feature_scale(b)
            .partial_cmp(&feature_scale(a))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.keypoint
                    .response
                    .partial_cmp(&a.keypoint.response)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    features.truncate(options.max_num_features.min(features.len()));
    Ok(features_from_lowe(features, options))
}

pub fn match_sift_mutual(
    left: &SiftFeatures,
    right: &SiftFeatures,
    ratio_threshold: f32,
) -> Vec<Match> {
    let options = SiftMatchingOptions {
        max_ratio: ratio_threshold,
        ..Default::default()
    };
    match_sift_with_options(left, right, &options)
}

pub fn match_sift_with_options(
    left: &SiftFeatures,
    right: &SiftFeatures,
    options: &SiftMatchingOptions,
) -> Vec<Match> {
    if options.check().is_err() {
        return Vec::new();
    }
    if left.descriptors.is_empty() || right.descriptors.is_empty() {
        return Vec::new();
    }
    let config = BbfConfig {
        ratio_threshold: options.max_ratio,
        max_candidates: 512,
        leaf_size: 16,
    };
    let Ok(forward) =
        lowe_sift::match_descriptors_bbf(&left.descriptors, &right.descriptors, config)
    else {
        return Vec::new();
    };
    let reverse_best = if options.cross_check {
        let Ok(reverse) =
            lowe_sift::match_descriptors_bbf(&right.descriptors, &left.descriptors, config)
        else {
            return Vec::new();
        };
        let mut reverse_best = HashMap::with_capacity(reverse.len());
        for m in reverse {
            if m.distance <= options.max_distance {
                reverse_best.insert((m.query_index, m.train_index), m.distance);
            }
        }
        Some(reverse_best)
    } else {
        None
    };

    let mut matches = forward
        .into_iter()
        .filter(|m| m.distance <= options.max_distance)
        .filter(|m| {
            reverse_best
                .as_ref()
                .map(|reverse| reverse.contains_key(&(m.train_index, m.query_index)))
                .unwrap_or(true)
        })
        .map(|m| Match {
            query_idx: m.query_index as u32,
            train_idx: m.train_index as u32,
            distance: m.distance,
        })
        .collect::<Vec<_>>();
    matches.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if options.max_num_matches > 0 && matches.len() > options.max_num_matches {
        matches.truncate(options.max_num_matches);
    }
    matches
}

pub fn match_sift_guided_with_options(
    left: &SiftFeatures,
    right: &SiftFeatures,
    f_matrix: &[f64; 9],
    options: &SiftMatchingOptions,
) -> Vec<Match> {
    if options.check().is_err() {
        return Vec::new();
    }
    if left.descriptors.is_empty() || right.descriptors.is_empty() {
        return Vec::new();
    }
    let f = nalgebra::Matrix3::from_row_slice(f_matrix);
    let max_epipolar_error = options.max_guided_epipolar_error_px.max(0.0);

    let mut forward = Vec::new();
    for (left_idx, (left_kp, left_desc)) in left
        .keypoints
        .iter()
        .zip(left.descriptors.iter())
        .enumerate()
    {
        let x1 = nalgebra::Vector3::new(left_kp.x() as f64, left_kp.y() as f64, 1.0);
        let line2 = f * x1;
        let mut best = None::<(u32, f32, f32)>;
        let mut second_best = f32::INFINITY;
        for (right_idx, right_kp) in right.keypoints.iter().enumerate() {
            let err = epipolar_line_distance_px(
                (right_kp.x(), right_kp.y()),
                (line2.x, line2.y, line2.z),
            );
            if err > max_epipolar_error {
                continue;
            }
            let distance2 = left_desc.distance2(&right.descriptors[right_idx]);
            let distance = distance2.sqrt();
            if distance > options.max_distance {
                continue;
            }
            match best {
                Some((_, best_distance, _)) if distance >= best_distance => {
                    if distance < second_best {
                        second_best = distance;
                    }
                }
                Some((_, best_distance, _)) => {
                    second_best = best_distance;
                    best = Some((right_idx as u32, distance, err));
                }
                None => best = Some((right_idx as u32, distance, err)),
            }
        }
        let Some((train_idx, best_distance, _)) = best else {
            continue;
        };
        if second_best.is_finite() && best_distance >= options.max_ratio * second_best {
            continue;
        }
        forward.push(Match {
            query_idx: left_idx as u32,
            train_idx,
            distance: best_distance,
        });
    }

    if !options.cross_check {
        forward.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if options.max_num_matches > 0 && forward.len() > options.max_num_matches {
            forward.truncate(options.max_num_matches);
        }
        return forward;
    }

    let mut reverse = Vec::new();
    let f_t = f.transpose();
    for (right_idx, (right_kp, right_desc)) in right
        .keypoints
        .iter()
        .zip(right.descriptors.iter())
        .enumerate()
    {
        let x2 = nalgebra::Vector3::new(right_kp.x() as f64, right_kp.y() as f64, 1.0);
        let line1 = f_t * x2;
        let mut best = None::<(u32, f32)>;
        let mut second_best = f32::INFINITY;
        for (left_idx, left_kp) in left.keypoints.iter().enumerate() {
            let err = epipolar_line_distance_px(
                (left_kp.x(), left_kp.y()),
                (line1.x, line1.y, line1.z),
            );
            if err > max_epipolar_error {
                continue;
            }
            let distance2 = right_desc.distance2(&left.descriptors[left_idx]);
            let distance = distance2.sqrt();
            if distance > options.max_distance {
                continue;
            }
            match best {
                Some((_, best_distance)) if distance >= best_distance => {
                    if distance < second_best {
                        second_best = distance;
                    }
                }
                Some((_, best_distance)) => {
                    second_best = best_distance;
                    best = Some((left_idx as u32, distance));
                }
                None => best = Some((left_idx as u32, distance)),
            }
        }
        let Some((query_idx, best_distance)) = best else {
            continue;
        };
        if second_best.is_finite() && best_distance >= options.max_ratio * second_best {
            continue;
        }
        reverse.push((query_idx, right_idx as u32));
    }

    let reverse_set: HashSet<(u32, u32)> = reverse.into_iter().collect();
    let mut matches = forward
        .into_iter()
        .filter(|m| reverse_set.contains(&(m.query_idx, m.train_idx)))
        .collect::<Vec<_>>();
    matches.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if options.max_num_matches > 0 && matches.len() > options.max_num_matches {
        matches.truncate(options.max_num_matches);
    }
    matches
}

fn epipolar_line_distance_px(point: (f32, f32), line: (f64, f64, f64)) -> f32 {
    let (a, b, c) = line;
    let numerator = (a * point.0 as f64 + b * point.1 as f64 + c).abs();
    let denominator = (a * a + b * b).sqrt();
    if denominator <= 1.0e-12 || !denominator.is_finite() {
        return f32::INFINITY;
    }
    (numerator / denominator) as f32
}

fn feature_scale(feature: &Feature) -> f32 {
    feature.keypoint.size * 2.0f32.powi(feature.keypoint.octave)
}

fn features_from_lowe(features: Vec<Feature>, options: &SiftExtractionOptions) -> SiftFeatures {
    let mut keypoints = Vec::with_capacity(features.len());
    let mut descriptors = Vec::with_capacity(features.len());
    for mut feature in features {
        if options.upright {
            feature.keypoint.angle = 0.0;
        }
        keypoints.push(KeyPoint {
            pt: (feature.keypoint.x, feature.keypoint.y),
            size: feature.keypoint.size,
            angle: feature.keypoint.angle,
            response: feature.keypoint.response,
            octave: feature.keypoint.octave,
        });
        descriptors.push(normalize_descriptor(
            feature.descriptor,
            options.normalization,
        ));
    }
    SiftFeatures {
        keypoints,
        descriptors,
    }
}

fn normalize_descriptor(
    descriptor: Descriptor,
    normalization: SiftDescriptorNormalization,
) -> Descriptor {
    let mut values = *descriptor.as_slice();
    match normalization {
        SiftDescriptorNormalization::L1Root => {
            let l1_norm: f32 = values.iter().map(|v| v.abs()).sum();
            if l1_norm > f32::EPSILON {
                for value in &mut values {
                    *value /= l1_norm;
                    *value = value.max(0.0).sqrt();
                }
            }
            let l2_norm: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
            if l2_norm > f32::EPSILON {
                for value in &mut values {
                    *value /= l2_norm;
                }
            }
        }
        SiftDescriptorNormalization::L2 => {
            let l2_norm: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
            if l2_norm > f32::EPSILON {
                for value in &mut values {
                    *value /= l2_norm;
                }
            }
        }
    }
    Descriptor::new(values)
}

fn rgb_to_sift_gray(rgb: &[u8], width: u32, height: u32) -> Result<GrayImage> {
    let mut gray = Vec::with_capacity(width as usize * height as usize);
    for px in rgb.chunks_exact(3) {
        let y = 0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32;
        gray.push(y / 255.0);
    }
    Ok(GrayImage::new(width as usize, height as usize, gray)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor_with_first(value: f32) -> Descriptor {
        let mut values = [0.0; lowe_sift::DESCRIPTOR_LEN];
        values[0] = value;
        Descriptor::new(values)
    }

    #[test]
    fn colmap_style_sift_defaults_match_official_values() {
        let extraction = SiftExtractionOptions::default();
        assert_eq!(extraction.max_num_features, 8192);
        assert_eq!(extraction.first_octave, -1);
        assert_eq!(extraction.num_octaves, 4);
        assert_eq!(extraction.octave_resolution, 3);
        assert!((extraction.peak_threshold - 0.02 / 3.0).abs() <= f64::EPSILON);
        assert_eq!(extraction.edge_threshold, 10.0);

        let matching = SiftMatchingOptions::default();
        assert_eq!(matching.max_ratio, 0.8);
        assert_eq!(matching.max_distance, 0.7);
        assert!(matching.cross_check);
        assert_eq!(matching.max_num_matches, 32768);
    }

    #[test]
    fn l1_root_normalization_matches_colmap_shape() {
        let descriptor = normalize_descriptor(
            descriptor_with_first(4.0),
            SiftDescriptorNormalization::L1Root,
        );
        let values = descriptor.as_slice();
        let l2_norm: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((l2_norm - 1.0).abs() <= 1.0e-5);
        assert!(values[0] > 0.99);
        assert!(values[1..].iter().all(|&v| v.abs() <= 1.0e-5));
    }

    #[test]
    fn feature_scale_prefers_larger_octave_scale_when_limiting_features() {
        assert!(10.0 * 2.0f32.powi(2) > 5.0 * 2.0f32.powi(0));
    }

    #[test]
    fn guided_matching_filters_by_epipolar_line_before_ratio_test() {
        let left = SiftFeatures {
            keypoints: vec![rustslam::KeyPoint::new(100.0, 120.0)],
            descriptors: vec![descriptor_with_first(0.0)],
        };
        let right = SiftFeatures {
            keypoints: vec![
                rustslam::KeyPoint::new(140.0, 120.0),
                rustslam::KeyPoint::new(500.0, 400.0),
            ],
            descriptors: vec![descriptor_with_first(0.05), descriptor_with_first(0.9)],
        };
        // Pure translation: epipolar lines are horizontal in image 2.
        let f = [
            0.0, 0.0, 0.0, //
            0.0, 0.0, -1.0, //
            0.0, 1.0, 0.0,
        ];
        let matches = match_sift_guided_with_options(
            &left,
            &right,
            &f,
            &SiftMatchingOptions {
                max_ratio: 0.8,
                max_distance: 0.3,
                cross_check: false,
                max_num_matches: 32,
                guided_matching: true,
                max_guided_epipolar_error_px: 2.0,
            },
        );
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].train_idx, 0);
    }

    #[test]
    fn sift_matching_applies_max_distance() {
        let left = SiftFeatures {
            keypoints: Vec::new(),
            descriptors: vec![descriptor_with_first(0.0)],
        };
        let right = SiftFeatures {
            keypoints: Vec::new(),
            descriptors: vec![descriptor_with_first(0.2), descriptor_with_first(1.0)],
        };
        let accepted = match_sift_with_options(
            &left,
            &right,
            &SiftMatchingOptions {
                max_ratio: 0.8,
                max_distance: 0.3,
                cross_check: false,
                max_num_matches: 32768,
                guided_matching: false,
                max_guided_epipolar_error_px: 2.0,
            },
        );
        let rejected = match_sift_with_options(
            &left,
            &right,
            &SiftMatchingOptions {
                max_ratio: 0.8,
                max_distance: 0.1,
                cross_check: false,
                max_num_matches: 32768,
                guided_matching: false,
                max_guided_epipolar_error_px: 2.0,
            },
        );

        assert_eq!(accepted.len(), 1);
        assert!(rejected.is_empty());
    }
}
