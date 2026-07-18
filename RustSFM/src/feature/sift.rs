use anyhow::{bail, Context, Result};
use lowe_sift::Descriptor;
use rustslam::{KeyPoint, Match};
#[cfg(any(not(feature = "vlfeat-sift"), feature = "lowe-sift-backend"))]
use std::collections::HashMap;
use std::collections::HashSet;

use crate::database::ColmapKeypoint;
use crate::sift_index::SiftDescriptorIndex;
#[cfg(any(not(feature = "vlfeat-sift"), feature = "lowe-sift-backend"))]
use lowe_sift::{BbfConfig, Feature, GrayImage, Sift, SiftConfig};
use rayon::prelude::*;

#[derive(Debug, Clone, Default)]
pub struct SiftFeatures {
    pub keypoints: Vec<KeyPoint>,
    pub descriptors: Vec<Descriptor>,
    pub colmap_keypoints: Vec<ColmapKeypoint>,
    pub descriptors_u8: Vec<[u8; lowe_sift::DESCRIPTOR_LEN]>,
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
    pub force_covariant_extractor: bool,
    pub use_gpu: bool,
    /// COLMAP `SiftExtraction.max_image_size` / `FeatureExtraction.max_image_size`.
    /// Downscale when max(width, height) exceeds this value. `0` disables rescaling.
    pub max_image_size: usize,
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
            force_covariant_extractor: false,
            use_gpu: false,
            max_image_size: 3200,
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

    pub fn uses_covariant_extractor(&self) -> bool {
        self.force_covariant_extractor || self.estimate_affine_shape || self.domain_size_pooling
    }

    #[cfg(any(not(feature = "vlfeat-sift"), feature = "lowe-sift-backend"))]
    fn to_lowe_config(&self) -> SiftConfig {
        let mut config = SiftConfig::default();
        config.intervals = self.octave_resolution;
        config.double_image = self.first_octave < 0;
        config.contrast_threshold = self.peak_threshold as f32;
        config.edge_threshold = self.edge_threshold as f32;
        // COLMAP/VLFeat stop building octaves once the shorter side is below ~32 px.
        config.min_octave_size = 32;
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
    pub cpu_brute_force_matcher: bool,
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
            cpu_brute_force_matcher: false,
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
    let (gray, width, height) =
        prepare_colmap_grayscale(rgb, width, height, options.max_image_size)?;
    extract_sift_from_grayscale_u8(&gray, width, height, options)
}

pub fn extract_sift_from_grayscale_u8(
    gray: &[u8],
    width: u32,
    height: u32,
    options: &SiftExtractionOptions,
) -> Result<SiftFeatures> {
    options.check()?;
    let expected = width as usize * height as usize;
    if gray.len() != expected {
        bail!(
            "grayscale buffer length {} does not match {}x{}",
            gray.len(),
            width,
            height
        );
    }
    let (gray, width, height) =
        prepare_grayscale_for_extraction(gray, width, height, options.max_image_size)?;
    #[cfg(all(feature = "vlfeat-sift", not(feature = "lowe-sift-backend")))]
    {
        return extract_sift_features_vlfeat(&gray, width, height, options);
    }
    #[cfg(any(not(feature = "vlfeat-sift"), feature = "lowe-sift-backend"))]
    {
        extract_sift_features_lowe_from_gray(&gray, width, height, options)
    }
}

#[cfg(any(not(feature = "vlfeat-sift"), feature = "lowe-sift-backend"))]
fn extract_sift_features_lowe_from_gray(
    gray: &[u8],
    width: u32,
    height: u32,
    options: &SiftExtractionOptions,
) -> Result<SiftFeatures> {
    let gray_f32 = gray.iter().map(|&pixel| f32::from(pixel) / 255.0).collect();
    let gray_image = GrayImage::new(width as usize, height as usize, gray_f32)?;
    let mut features = Sift::new(options.to_lowe_config())?.detect_and_compute(&gray_image);
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

#[cfg(all(feature = "vlfeat-sift", not(feature = "lowe-sift-backend")))]
fn extract_sift_features_vlfeat(
    gray_u8: &[u8],
    width: u32,
    height: u32,
    options: &SiftExtractionOptions,
) -> Result<SiftFeatures> {
    use std::ffi::CStr;

    let c_options = RustSfmVlfeatSiftOptions {
        max_num_features: options.max_num_features as i32,
        first_octave: options.first_octave,
        num_octaves: options.num_octaves as i32,
        octave_resolution: options.octave_resolution as i32,
        peak_threshold: options.peak_threshold as f32,
        edge_threshold: options.edge_threshold as f32,
        max_num_orientations: options.max_num_orientations as i32,
        upright: i32::from(options.upright),
        normalization_l1_root: i32::from(matches!(
            options.normalization,
            SiftDescriptorNormalization::L1Root
        )),
        estimate_affine_shape: i32::from(options.estimate_affine_shape),
        domain_size_pooling: i32::from(options.domain_size_pooling),
        dsp_min_scale: options.dsp_min_scale as f32,
        dsp_max_scale: options.dsp_max_scale as f32,
        dsp_num_scales: options.dsp_num_scales as i32,
        force_covariant_extractor: i32::from(options.force_covariant_extractor),
    };

    let mut out = RustSfmVlfeatSiftFeatures {
        keypoints: std::ptr::null_mut(),
        descriptors: std::ptr::null_mut(),
        count: 0,
        error_message: std::ptr::null_mut(),
    };

    let ok = unsafe {
        rustsfm_vlfeat_extract_sift(
            gray_u8.as_ptr(),
            width as i32,
            height as i32,
            &c_options,
            &mut out,
        )
    };
    if ok == 0 {
        let message = unsafe {
            if out.error_message.is_null() {
                "VLFeat SIFT extraction failed".to_string()
            } else {
                CStr::from_ptr(out.error_message)
                    .to_string_lossy()
                    .into_owned()
            }
        };
        unsafe {
            rustsfm_vlfeat_free_features(&mut out);
        }
        bail!(message);
    }

    let features = features_from_vlfeat(&out, options);
    unsafe {
        rustsfm_vlfeat_free_features(&mut out);
    }
    Ok(features)
}

#[cfg(all(feature = "vlfeat-sift", not(feature = "lowe-sift-backend")))]
fn features_from_vlfeat(
    out: &RustSfmVlfeatSiftFeatures,
    options: &SiftExtractionOptions,
) -> SiftFeatures {
    let mut keypoints = Vec::with_capacity(out.count);
    let mut descriptors = Vec::with_capacity(out.count);
    let mut colmap_keypoints = Vec::with_capacity(out.count);
    let mut descriptors_u8 = Vec::with_capacity(out.count);
    if out.count == 0 {
        return SiftFeatures {
            keypoints,
            descriptors,
            colmap_keypoints,
            descriptors_u8: Vec::new(),
        };
    }

    unsafe {
        for i in 0..out.count {
            let kp = &*out.keypoints.add(i);
            let mut angle = kp.angle;
            if options.upright {
                angle = 0.0;
            }
            keypoints.push(KeyPoint {
                pt: (kp.x, kp.y),
                size: kp.size,
                angle,
                response: kp.response,
                octave: kp.octave,
            });
            colmap_keypoints.push(ColmapKeypoint {
                x: kp.x,
                y: kp.y,
                a11: kp.a11,
                a12: kp.a12,
                a21: kp.a21,
                a22: kp.a22,
            });

            let mut values = [0.0f32; lowe_sift::DESCRIPTOR_LEN];
            let src = out.descriptors.add(i * lowe_sift::DESCRIPTOR_LEN);
            values.copy_from_slice(std::slice::from_raw_parts(src, lowe_sift::DESCRIPTOR_LEN));
            descriptors_u8.push(descriptor_to_uint8_from_float(&values));
            descriptors.push(Descriptor::new(values));
        }
    }

    SiftFeatures {
        keypoints,
        descriptors,
        colmap_keypoints,
        descriptors_u8,
    }
}

#[cfg(all(feature = "vlfeat-sift", not(feature = "lowe-sift-backend")))]
mod vlfeat_ffi {
    use std::os::raw::{c_char, c_float, c_int};

    #[repr(C)]
    pub struct RustSfmVlfeatSiftOptions {
        pub max_num_features: c_int,
        pub first_octave: c_int,
        pub num_octaves: c_int,
        pub octave_resolution: c_int,
        pub peak_threshold: c_float,
        pub edge_threshold: c_float,
        pub max_num_orientations: c_int,
        pub upright: c_int,
        pub normalization_l1_root: c_int,
        pub estimate_affine_shape: c_int,
        pub domain_size_pooling: c_int,
        pub dsp_min_scale: c_float,
        pub dsp_max_scale: c_float,
        pub dsp_num_scales: c_int,
        pub force_covariant_extractor: c_int,
    }

    #[repr(C)]
    pub struct RustSfmVlfeatSiftKeypoint {
        pub x: c_float,
        pub y: c_float,
        pub size: c_float,
        pub angle: c_float,
        pub response: c_float,
        pub octave: c_int,
        pub a11: c_float,
        pub a12: c_float,
        pub a21: c_float,
        pub a22: c_float,
    }

    #[repr(C)]
    pub struct RustSfmVlfeatSiftFeatures {
        pub keypoints: *mut RustSfmVlfeatSiftKeypoint,
        pub descriptors: *mut c_float,
        pub count: usize,
        pub error_message: *mut c_char,
    }

    extern "C" {
        pub fn rustsfm_vlfeat_extract_sift(
            gray_u8: *const u8,
            width: c_int,
            height: c_int,
            options: *const RustSfmVlfeatSiftOptions,
            out: *mut RustSfmVlfeatSiftFeatures,
        ) -> c_int;

        pub fn rustsfm_vlfeat_free_features(out: *mut RustSfmVlfeatSiftFeatures);

        pub fn rustsfm_vlfeat_test_paired_allocation_failure(
            growth_path: c_int,
            fail_allocation: c_int,
        ) -> c_int;
    }
}

#[cfg(all(feature = "vlfeat-sift", not(feature = "lowe-sift-backend")))]
use vlfeat_ffi::{
    rustsfm_vlfeat_extract_sift, rustsfm_vlfeat_free_features, RustSfmVlfeatSiftFeatures,
    RustSfmVlfeatSiftOptions,
};

const COLMAP_SIFT_DESCRIPTOR_NORM: f32 = 512.0 * 512.0;

fn descriptor_to_uint8(descriptor: &Descriptor) -> [u8; lowe_sift::DESCRIPTOR_LEN] {
    descriptor_to_uint8_from_float(descriptor.as_slice())
}

fn descriptor_to_uint8_from_float(
    values: &[f32; lowe_sift::DESCRIPTOR_LEN],
) -> [u8; lowe_sift::DESCRIPTOR_LEN] {
    let mut out = [0u8; lowe_sift::DESCRIPTOR_LEN];
    for (value, slot) in values.iter().zip(out.iter_mut()) {
        *slot = (value.clamp(0.0, 1.0) * 512.0).round() as u8;
    }
    out
}

fn sift_features_u8(features: &SiftFeatures) -> Vec<[u8; lowe_sift::DESCRIPTOR_LEN]> {
    if !features.descriptors_u8.is_empty() {
        return features.descriptors_u8.clone();
    }
    features
        .descriptors
        .iter()
        .map(descriptor_to_uint8)
        .collect()
}

fn colmap_uint8_l2_distance2(
    left: &[u8; lowe_sift::DESCRIPTOR_LEN],
    right: &[u8; lowe_sift::DESCRIPTOR_LEN],
) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| {
            let delta = i32::from(*a) - i32::from(*b);
            (delta * delta) as f32
        })
        .sum()
}

fn colmap_normalized_distance(l2_dist: f32) -> f32 {
    (l2_dist / COLMAP_SIFT_DESCRIPTOR_NORM).sqrt()
}

#[cfg(all(feature = "vlfeat-sift", not(feature = "lowe-sift-backend")))]
fn sift_pair_l2_distance2(left: &Descriptor, right: &Descriptor) -> f32 {
    colmap_uint8_l2_distance2(&descriptor_to_uint8(left), &descriptor_to_uint8(right))
}

#[cfg(any(not(feature = "vlfeat-sift"), feature = "lowe-sift-backend"))]
fn sift_pair_l2_distance2(left: &Descriptor, right: &Descriptor) -> f32 {
    left.distance2(right)
}

fn sift_pair_distance(left: &Descriptor, right: &Descriptor) -> f32 {
    let l2_dist = sift_pair_l2_distance2(left, right);
    #[cfg(all(feature = "vlfeat-sift", not(feature = "lowe-sift-backend")))]
    {
        colmap_normalized_distance(l2_dist)
    }
    #[cfg(any(not(feature = "vlfeat-sift"), feature = "lowe-sift-backend"))]
    {
        l2_dist.sqrt()
    }
}

#[cfg(all(feature = "vlfeat-sift", not(feature = "lowe-sift-backend")))]
fn match_sift_colmap_uint8(
    left: &SiftFeatures,
    right: &SiftFeatures,
    options: &SiftMatchingOptions,
) -> Vec<Match> {
    let left_u8 = sift_features_u8(left);
    let right_u8 = sift_features_u8(right);
    let forward = if options.cpu_brute_force_matcher {
        match_sift_colmap_one_way_brute(&left_u8, &right_u8, options)
    } else {
        let index = SiftDescriptorIndex::build(&right_u8);
        match_sift_colmap_one_way_indexed(&left_u8, &index, options)
    };
    if options.cross_check {
        let reverse_pairs: HashSet<(u32, u32)> = if options.cpu_brute_force_matcher {
            match_sift_colmap_one_way_brute(&right_u8, &left_u8, options)
        } else {
            let index = SiftDescriptorIndex::build(&left_u8);
            match_sift_colmap_one_way_indexed(&right_u8, &index, options)
        }
        .into_iter()
        .collect();
        let mut matches = Vec::new();
        for (query_idx, train_idx) in forward {
            if reverse_pairs.contains(&(train_idx, query_idx)) {
                let distance = colmap_normalized_distance(colmap_uint8_l2_distance2(
                    &left_u8[query_idx as usize],
                    &right_u8[train_idx as usize],
                ));
                matches.push(Match {
                    query_idx,
                    train_idx,
                    distance,
                });
            }
        }
        finalize_matches(matches, options)
    } else {
        let matches = forward
            .into_iter()
            .map(|(query_idx, train_idx)| {
                let distance = colmap_normalized_distance(colmap_uint8_l2_distance2(
                    &left_u8[query_idx as usize],
                    &right_u8[train_idx as usize],
                ));
                Match {
                    query_idx,
                    train_idx,
                    distance,
                }
            })
            .collect();
        finalize_matches(matches, options)
    }
}

fn match_sift_colmap_one_way_brute(
    left: &[[u8; lowe_sift::DESCRIPTOR_LEN]],
    right: &[[u8; lowe_sift::DESCRIPTOR_LEN]],
    options: &SiftMatchingOptions,
) -> Vec<(u32, u32)> {
    let max_l2_dist = COLMAP_SIFT_DESCRIPTOR_NORM * options.max_distance * options.max_distance;
    left.par_iter()
        .enumerate()
        .filter_map(|(query_idx, left_desc)| {
            let mut best_train = None::<usize>;
            let mut best_l2 = f32::INFINITY;
            let mut second_best_l2 = f32::INFINITY;
            for (train_idx, right_desc) in right.iter().enumerate() {
                let l2_dist = colmap_uint8_l2_distance2(left_desc, right_desc);
                if l2_dist < best_l2 {
                    second_best_l2 = best_l2;
                    best_l2 = l2_dist;
                    best_train = Some(train_idx);
                } else if l2_dist < second_best_l2 {
                    second_best_l2 = l2_dist;
                }
            }
            let train_idx = best_train?;
            if best_l2 > max_l2_dist {
                return None;
            }
            if best_l2.sqrt() >= options.max_ratio * second_best_l2.sqrt() {
                return None;
            }
            Some((query_idx as u32, train_idx as u32))
        })
        .collect()
}

fn match_sift_colmap_one_way_indexed(
    left: &[[u8; lowe_sift::DESCRIPTOR_LEN]],
    index: &SiftDescriptorIndex,
    options: &SiftMatchingOptions,
) -> Vec<(u32, u32)> {
    let max_l2_dist = COLMAP_SIFT_DESCRIPTOR_NORM * options.max_distance * options.max_distance;
    left.par_iter()
        .enumerate()
        .filter_map(|(query_idx, left_desc)| {
            let neighbors = index.search_two_nearest(left_desc)?;
            let (best_l2, best_index, second_best_l2) =
                if neighbors.best_l2 <= neighbors.second_best_l2 {
                    (
                        neighbors.best_l2,
                        neighbors.best_index,
                        neighbors.second_best_l2,
                    )
                } else {
                    (
                        neighbors.second_best_l2,
                        neighbors.second_best_index,
                        neighbors.best_l2,
                    )
                };
            if best_l2 > max_l2_dist {
                return None;
            }
            if best_l2.sqrt() >= options.max_ratio * second_best_l2.sqrt() {
                return None;
            }
            Some((query_idx as u32, best_index))
        })
        .collect()
}

#[cfg(any(not(feature = "vlfeat-sift"), feature = "lowe-sift-backend"))]
fn match_sift_lowe_bbf(
    left: &SiftFeatures,
    right: &SiftFeatures,
    options: &SiftMatchingOptions,
) -> Vec<Match> {
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

    let matches = forward
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
        .collect();
    finalize_matches(matches, options)
}

fn finalize_matches(mut matches: Vec<Match>, options: &SiftMatchingOptions) -> Vec<Match> {
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
    #[cfg(all(feature = "vlfeat-sift", not(feature = "lowe-sift-backend")))]
    {
        return match_sift_colmap_uint8(left, right, options);
    }
    #[cfg(any(not(feature = "vlfeat-sift"), feature = "lowe-sift-backend"))]
    {
        match_sift_lowe_bbf(left, right, options)
    }
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
            let distance = sift_pair_distance(left_desc, &right.descriptors[right_idx]);
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
            let err =
                epipolar_line_distance_px((left_kp.x(), left_kp.y()), (line1.x, line1.y, line1.z));
            if err > max_epipolar_error {
                continue;
            }
            let distance = sift_pair_distance(right_desc, &left.descriptors[left_idx]);
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

#[cfg(any(not(feature = "vlfeat-sift"), feature = "lowe-sift-backend"))]
fn feature_scale(feature: &Feature) -> f32 {
    feature.keypoint.size * 2.0f32.powi(feature.keypoint.octave)
}

#[cfg(any(not(feature = "vlfeat-sift"), feature = "lowe-sift-backend"))]
fn features_from_lowe(features: Vec<Feature>, options: &SiftExtractionOptions) -> SiftFeatures {
    let mut keypoints = Vec::with_capacity(features.len());
    let mut descriptors = Vec::with_capacity(features.len());
    let mut colmap_keypoints = Vec::with_capacity(features.len());
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
        colmap_keypoints.push(ColmapKeypoint::from_scale_orientation(
            feature.keypoint.x,
            feature.keypoint.y,
            feature.keypoint.size,
            feature.keypoint.angle,
        ));
        descriptors.push(normalize_descriptor(
            feature.descriptor,
            options.normalization,
        ));
    }
    let descriptors_u8 = descriptors
        .iter()
        .map(|descriptor| descriptor_to_uint8(descriptor))
        .collect();
    SiftFeatures {
        keypoints,
        descriptors,
        colmap_keypoints,
        descriptors_u8,
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

pub(crate) fn rgb_to_colmap_gray_u8(rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let expected = width as usize * height as usize * 3;
    if rgb.len() != expected {
        bail!(
            "rgb buffer length {} does not match {}x{}x3",
            rgb.len(),
            width,
            height
        );
    }
    let mut gray = Vec::with_capacity(width as usize * height as usize);
    for px in rgb.chunks_exact(3) {
        // COLMAP Bitmap::CloneAsGrey (BT.709 luminance, rounded to u8).
        let y = 0.2126 * px[0] as f64 + 0.7152 * px[1] as f64 + 0.0722 * px[2] as f64;
        gray.push(y.round().clamp(0.0, 255.0) as u8);
    }
    Ok(gray)
}

fn prepare_colmap_grayscale(
    rgb: &[u8],
    width: u32,
    height: u32,
    max_image_size: usize,
) -> Result<(Vec<u8>, u32, u32)> {
    let gray = rgb_to_colmap_gray_u8(rgb, width, height)?;
    prepare_grayscale_for_extraction(&gray, width, height, max_image_size)
}

pub(crate) fn prepare_grayscale_for_extraction(
    gray: &[u8],
    width: u32,
    height: u32,
    max_image_size: usize,
) -> Result<(Vec<u8>, u32, u32)> {
    let mut gray = gray.to_vec();
    let mut w = width;
    let mut h = height;
    if max_image_size > 0 && (w.max(h) as usize) > max_image_size {
        let scale = max_image_size as f64 / f64::from(w.max(h));
        let new_w = ((f64::from(w) * scale).round() as u32).max(1);
        let new_h = ((f64::from(h) * scale).round() as u32).max(1);
        let image = image::GrayImage::from_raw(w, h, gray)
            .context("failed to build grayscale image for rescaling")?;
        let resized =
            image::imageops::resize(&image, new_w, new_h, image::imageops::FilterType::Triangle);
        w = resized.width();
        h = resized.height();
        gray = resized.into_raw();
    }
    Ok((gray, w, h))
}

#[cfg(any(not(feature = "vlfeat-sift"), feature = "lowe-sift-backend"))]
fn rgb_to_sift_gray(rgb: &[u8], width: u32, height: u32) -> Result<GrayImage> {
    let gray = rgb_to_colmap_gray_u8(rgb, width, height)?;
    let mut values = Vec::with_capacity(gray.len());
    for value in gray {
        values.push(value as f32 / 255.0);
    }
    Ok(GrayImage::new(width as usize, height as usize, values)?)
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SiftImageBenchmark {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub num_features: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SiftBenchmarkReport {
    pub backend: &'static str,
    pub image_count: usize,
    pub total_features: usize,
    pub mean_features: f64,
    pub extraction_seconds: f64,
    pub uses_covariant_extractor: bool,
    pub images: Vec<SiftImageBenchmark>,
}

pub fn benchmark_sift_extraction(
    input: &std::path::Path,
    options: &SiftExtractionOptions,
) -> Result<SiftBenchmarkReport> {
    use crate::colmap_image::load_colmap_grayscale_u8;
    use std::time::Instant;

    options.check()?;
    #[cfg(feature = "gpu-wgpu")]
    let gpu_extractor = if options.use_gpu {
        Some(crate::gpu::WgpuSiftExtractor::try_new()?)
    } else {
        None
    };
    #[cfg(not(feature = "gpu-wgpu"))]
    if options.use_gpu {
        bail!("RustSFM was built without gpu-wgpu support");
    }
    let mut paths = std::fs::read_dir(input)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| {
                    matches!(
                        ext.to_ascii_lowercase().as_str(),
                        "jpg" | "jpeg" | "png" | "bmp" | "tif" | "tiff" | "webp"
                    )
                })
        })
        .collect::<Vec<_>>();
    paths.sort();

    let started = Instant::now();
    let mut images = Vec::with_capacity(paths.len());
    let mut total_features = 0usize;
    for path in paths {
        let decoded = load_colmap_grayscale_u8(&path)
            .with_context(|| format!("failed to load {}", path.display()))?;
        #[cfg(feature = "gpu-wgpu")]
        let features = if let Some(extractor) = gpu_extractor.as_ref() {
            extractor.extract_grayscale(&decoded.data, decoded.width, decoded.height, options)?
        } else {
            extract_sift_from_grayscale_u8(&decoded.data, decoded.width, decoded.height, options)?
        };
        #[cfg(not(feature = "gpu-wgpu"))]
        let features =
            extract_sift_from_grayscale_u8(&decoded.data, decoded.width, decoded.height, options)?;
        total_features += features.keypoints.len();
        images.push(SiftImageBenchmark {
            name: path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string()),
            width: decoded.width,
            height: decoded.height,
            num_features: features.keypoints.len(),
        });
    }
    let extraction_seconds = started.elapsed().as_secs_f64();
    let image_count = images.len();
    let mean_features = if image_count == 0 {
        0.0
    } else {
        total_features as f64 / image_count as f64
    };

    Ok(SiftBenchmarkReport {
        backend: if options.use_gpu {
            "wgpu"
        } else if cfg!(all(
            feature = "vlfeat-sift",
            not(feature = "lowe-sift-backend")
        )) {
            "vlfeat"
        } else {
            "lowe-sift"
        },
        image_count,
        total_features,
        mean_features,
        extraction_seconds,
        uses_covariant_extractor: options.uses_covariant_extractor(),
        images,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vlfeat_paired_buffer_growth_does_not_use_realloc() {
        let source = include_str!("../native/vlfeat_sift.c");
        assert!(
            !source.contains("realloc("),
            "paired native buffers must use allocate-copy-commit ownership"
        );
    }

    #[cfg(all(feature = "vlfeat-sift", not(feature = "lowe-sift-backend")))]
    #[test]
    fn vlfeat_paired_buffer_growth_is_failure_atomic() {
        for growth_path in 0..=1 {
            for fail_allocation in 0..=1 {
                let preserved = unsafe {
                    vlfeat_ffi::rustsfm_vlfeat_test_paired_allocation_failure(
                        growth_path,
                        fail_allocation,
                    )
                };
                assert_eq!(
                    preserved, 1,
                    "growth_path={growth_path} fail_allocation={fail_allocation}"
                );
            }
        }
    }

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
        assert_eq!(extraction.max_image_size, 3200);

        let matching = SiftMatchingOptions::default();
        assert_eq!(matching.max_ratio, 0.8);
        assert_eq!(matching.max_distance, 0.7);
        assert!(matching.cross_check);
        assert_eq!(matching.max_num_matches, 32768);
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn generic_sift_options_allow_explicit_gpu_selection() {
        let options = SiftExtractionOptions {
            use_gpu: true,
            ..SiftExtractionOptions::default()
        };
        assert!(options.check().is_ok());
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn benchmark_reports_wgpu_for_explicit_gpu_options() -> Result<()> {
        let input = tempfile::tempdir()?;
        let report = benchmark_sift_extraction(
            input.path(),
            &SiftExtractionOptions {
                use_gpu: true,
                ..Default::default()
            },
        )?;
        assert_eq!(report.backend, "wgpu");
        assert_eq!(report.image_count, 0);
        Ok(())
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
            colmap_keypoints: vec![],
            descriptors_u8: vec![],
        };
        let right = SiftFeatures {
            keypoints: vec![
                rustslam::KeyPoint::new(140.0, 120.0),
                rustslam::KeyPoint::new(500.0, 400.0),
            ],
            descriptors: vec![descriptor_with_first(0.05), descriptor_with_first(0.9)],
            colmap_keypoints: vec![],
            descriptors_u8: vec![],
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
                cpu_brute_force_matcher: true,
            },
        );
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].train_idx, 0);
    }

    #[test]
    fn covdet_backend_extracts_features_with_domain_size_pooling() {
        let width = 256u32;
        let height = 256u32;
        let mut rgb = vec![0u8; (width * height * 3) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                let value = if ((x / 16) + (y / 16)) % 2 == 0 {
                    240
                } else {
                    20
                };
                rgb[idx..idx + 3].fill(value);
            }
        }
        let options = SiftExtractionOptions {
            domain_size_pooling: true,
            max_num_features: 1024,
            ..Default::default()
        };
        let features =
            extract_sift_features_with_options(&rgb, width, height, &options).expect("extract");
        assert!(!features.keypoints.is_empty());
        assert_eq!(features.keypoints.len(), features.descriptors.len());
        assert_eq!(features.colmap_keypoints.len(), features.keypoints.len());
        assert!(features.colmap_keypoints.iter().any(|kp| kp.a11 != 0.0));
    }

    #[test]
    fn vlfeat_backend_extracts_features_from_checkerboard() {
        let width = 256u32;
        let height = 256u32;
        let mut rgb = vec![0u8; (width * height * 3) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                let value = if ((x / 16) + (y / 16)) % 2 == 0 {
                    240
                } else {
                    20
                };
                rgb[idx..idx + 3].fill(value);
            }
        }
        let features = extract_sift_features(&rgb, width, height, 8192).expect("extract");
        assert!(!features.keypoints.is_empty());
        assert_eq!(features.keypoints.len(), features.descriptors.len());
    }

    #[test]
    fn colmap_uint8_matching_applies_distance_and_ratio_thresholds() {
        let mut left = [0u8; lowe_sift::DESCRIPTOR_LEN];
        let mut near = [0u8; lowe_sift::DESCRIPTOR_LEN];
        let mut far = [0u8; lowe_sift::DESCRIPTOR_LEN];
        left.fill(255);
        near.fill(255);
        near[0] = 253;
        far.fill(0);

        let near_l2 = colmap_uint8_l2_distance2(&left, &near);
        let far_l2 = colmap_uint8_l2_distance2(&left, &far);
        assert!(near_l2 < far_l2);
        assert!(colmap_normalized_distance(near_l2) < 0.7);
        assert!(colmap_normalized_distance(far_l2) > 0.7);

        let left_features = SiftFeatures {
            keypoints: vec![KeyPoint::new(10.0, 10.0)],
            descriptors: vec![Descriptor::new(left.map(|v| v as f32 / 512.0))],
            colmap_keypoints: vec![],
            descriptors_u8: vec![left],
        };
        let right_features = SiftFeatures {
            keypoints: vec![KeyPoint::new(20.0, 20.0), KeyPoint::new(30.0, 30.0)],
            descriptors: vec![
                Descriptor::new(near.map(|v| v as f32 / 512.0)),
                Descriptor::new(far.map(|v| v as f32 / 512.0)),
            ],
            colmap_keypoints: vec![],
            descriptors_u8: vec![near, far],
        };

        let accepted = match_sift_with_options(
            &left_features,
            &right_features,
            &SiftMatchingOptions {
                max_ratio: 0.8,
                max_distance: 0.7,
                cross_check: false,
                max_num_matches: 32768,
                guided_matching: false,
                max_guided_epipolar_error_px: 2.0,
                cpu_brute_force_matcher: true,
            },
        );
        assert_eq!(accepted.len(), 1);
        assert_eq!(accepted[0].train_idx, 0);
    }

    #[test]
    fn indexed_matching_agrees_with_brute_force_on_toy_descriptors() {
        let mut left = [0u8; lowe_sift::DESCRIPTOR_LEN];
        let mut near = [0u8; lowe_sift::DESCRIPTOR_LEN];
        let mut far = [0u8; lowe_sift::DESCRIPTOR_LEN];
        left.fill(255);
        near.fill(255);
        near[0] = 253;
        far.fill(0);

        let left_set = vec![left];
        let right_set = vec![near, far];
        let options = SiftMatchingOptions {
            max_ratio: 0.8,
            max_distance: 0.7,
            cross_check: false,
            max_num_matches: 32768,
            guided_matching: false,
            max_guided_epipolar_error_px: 2.0,
            cpu_brute_force_matcher: false,
        };
        let brute = match_sift_colmap_one_way_brute(&left_set, &right_set, &options);
        let indexed = match_sift_colmap_one_way_indexed(
            &left_set,
            &SiftDescriptorIndex::build(&right_set),
            &options,
        );
        assert_eq!(brute, indexed);
    }

    #[test]
    fn sift_matching_applies_max_distance() {
        let left = SiftFeatures {
            keypoints: Vec::new(),
            descriptors: vec![descriptor_with_first(0.0)],
            colmap_keypoints: vec![],
            descriptors_u8: vec![],
        };
        let right = SiftFeatures {
            keypoints: Vec::new(),
            descriptors: vec![descriptor_with_first(0.2), descriptor_with_first(1.0)],
            colmap_keypoints: vec![],
            descriptors_u8: vec![],
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
                cpu_brute_force_matcher: true,
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
                cpu_brute_force_matcher: true,
            },
        );

        assert_eq!(accepted.len(), 1);
        assert!(rejected.is_empty());
    }
}
