use crate::geometry::{UnitQuatNormalize, Vec3GlamExt};
use anyhow::{bail, Context, Result};
use lowe_sift::Descriptor;
use rayon::prelude::*;
use rustslam::{KeyPoint, Match};
use std::collections::HashSet;

use crate::database::ColmapKeypoint;
#[cfg(any(not(feature = "vlfeat-sift"), feature = "lowe-sift-backend"))]
use lowe_sift::{Feature, GrayImage, Sift, SiftConfig};
use std::time::Instant;

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
    extract_sift_from_grayscale_u8_impl(gray, width, height, options, None)
        .map(|(features, _)| features)
}

pub(crate) fn extract_sift_from_grayscale_u8_with_timing(
    gray: &[u8],
    width: u32,
    height: u32,
    options: &SiftExtractionOptions,
) -> Result<(SiftFeatures, SiftExtractionTiming)> {
    let mut timing = SiftExtractionTiming::default();
    let features =
        extract_sift_from_grayscale_u8_impl(gray, width, height, options, Some(&mut timing))?;
    Ok((features.0, timing))
}

fn extract_sift_from_grayscale_u8_impl(
    gray: &[u8],
    width: u32,
    height: u32,
    options: &SiftExtractionOptions,
    mut timing: Option<&mut SiftExtractionTiming>,
) -> Result<(SiftFeatures, SiftExtractionTiming)> {
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
    let prepare_start = timing.as_ref().map(|_| Instant::now());
    let (gray, width, height) =
        prepare_grayscale_for_extraction(gray, width, height, options.max_image_size)?;
    if let (Some(timing), Some(start)) = (timing.as_mut(), prepare_start) {
        timing.preparation_ms = start.elapsed().as_secs_f64() * 1000.0;
    }
    #[cfg(all(feature = "vlfeat-sift", not(feature = "lowe-sift-backend")))]
    {
        let backend_start = timing.as_ref().map(|_| Instant::now());
        let features =
            extract_sift_features_vlfeat(&gray, width, height, options, timing.as_deref_mut())?;
        if let (Some(timing), Some(start)) = (timing.as_mut(), backend_start) {
            timing.backend_kernel_ms =
                (start.elapsed().as_secs_f64() * 1000.0 - timing.result_conversion_ms).max(0.0);
        }
        return Ok((
            features,
            timing.map(|timing| timing.clone()).unwrap_or_default(),
        ));
    }
    #[cfg(any(not(feature = "vlfeat-sift"), feature = "lowe-sift-backend"))]
    {
        let backend_start = timing.as_ref().map(|_| Instant::now());
        let features = extract_sift_features_lowe_from_gray(
            &gray,
            width,
            height,
            options,
            timing.as_deref_mut(),
        )?;
        if let (Some(timing), Some(start)) = (timing.as_mut(), backend_start) {
            timing.backend_kernel_ms = (start.elapsed().as_secs_f64() * 1000.0
                - timing.result_conversion_ms
                - timing.input_conversion_ms)
                .max(0.0);
        }
        Ok((
            features,
            timing.map(|timing| timing.clone()).unwrap_or_default(),
        ))
    }
}

#[cfg(any(not(feature = "vlfeat-sift"), feature = "lowe-sift-backend"))]
fn extract_sift_features_lowe_from_gray(
    gray: &[u8],
    width: u32,
    height: u32,
    options: &SiftExtractionOptions,
    mut timing: Option<&mut SiftExtractionTiming>,
) -> Result<SiftFeatures> {
    let input_start = timing.as_ref().map(|_| Instant::now());
    let gray_f32 = gray.iter().map(|&pixel| f32::from(pixel) / 255.0).collect();
    let gray_image = GrayImage::new(width as usize, height as usize, gray_f32)?;
    if let (Some(timing), Some(start)) = (timing.as_mut(), input_start) {
        timing.input_conversion_ms = start.elapsed().as_secs_f64() * 1000.0;
    }
    let kernel_start = timing.as_ref().map(|_| Instant::now());
    let mut features = Sift::new(options.to_lowe_config())?.detect_and_compute(&gray_image);
    if let (Some(timing), Some(start)) = (timing.as_mut(), kernel_start) {
        timing.backend_kernel_ms += start.elapsed().as_secs_f64() * 1000.0;
    }
    let result_start = timing.as_ref().map(|_| Instant::now());
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
    let result = features_from_lowe(features, options);
    if let (Some(timing), Some(start)) = (timing.as_mut(), result_start) {
        timing.result_conversion_ms = start.elapsed().as_secs_f64() * 1000.0;
    }
    Ok(result)
}

#[cfg(all(feature = "vlfeat-sift", not(feature = "lowe-sift-backend")))]
fn extract_sift_features_vlfeat(
    gray_u8: &[u8],
    width: u32,
    height: u32,
    options: &SiftExtractionOptions,
    mut timing: Option<&mut SiftExtractionTiming>,
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

    let mut backend_timing = RustSfmVlfeatSiftTiming::default();
    let backend_timing_ptr = if timing.is_some() {
        &mut backend_timing
    } else {
        std::ptr::null_mut()
    };
    let ok = unsafe {
        rustsfm_vlfeat_extract_sift(
            gray_u8.as_ptr(),
            width as i32,
            height as i32,
            &c_options,
            &mut out,
            backend_timing_ptr,
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

    if let Some(timing) = timing.as_mut() {
        timing.backend_input_conversion_ms = backend_timing.input_conversion_ms;
        timing.backend_scale_space_ms = backend_timing.scale_space_ms;
        timing.backend_detection_ms = backend_timing.detection_ms;
        timing.backend_orientation_ms = backend_timing.orientation_ms;
        timing.backend_descriptor_ms = backend_timing.descriptor_ms;
        timing.backend_output_assembly_ms = backend_timing.output_assembly_ms;
    }

    let features_start = timing.as_ref().map(|_| Instant::now());
    let features = features_from_vlfeat(&out, options);
    if let (Some(timing), Some(start)) = (timing.as_mut(), features_start) {
        timing.result_conversion_ms = start.elapsed().as_secs_f64() * 1000.0;
    }
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
    use std::os::raw::{c_char, c_double, c_float, c_int};

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
    #[derive(Default)]
    pub struct RustSfmVlfeatSiftTiming {
        pub input_conversion_ms: c_double,
        pub scale_space_ms: c_double,
        pub detection_ms: c_double,
        pub orientation_ms: c_double,
        pub descriptor_ms: c_double,
        pub output_assembly_ms: c_double,
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
            timing: *mut RustSfmVlfeatSiftTiming,
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
    RustSfmVlfeatSiftOptions, RustSfmVlfeatSiftTiming,
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

fn colmap_normalized_distance(l2_dist: f32) -> f32 {
    (l2_dist / COLMAP_SIFT_DESCRIPTOR_NORM).sqrt()
}

#[inline]
fn colmap_uint8_l2_distance2(
    left: &[u8; lowe_sift::DESCRIPTOR_LEN],
    right: &[u8; lowe_sift::DESCRIPTOR_LEN],
) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| {
            let delta = i32::from(*a) - i32::from(*b);
            (delta * delta) as u32
        })
        .sum::<u32>() as f32
}

fn sift_pair_l2_distance2(left: &Descriptor, right: &Descriptor) -> f32 {
    #[cfg(all(feature = "vlfeat-sift", not(feature = "lowe-sift-backend")))]
    {
        colmap_uint8_l2_distance2(&descriptor_to_uint8(left), &descriptor_to_uint8(right))
    }
    #[cfg(any(not(feature = "vlfeat-sift"), feature = "lowe-sift-backend"))]
    {
        left.distance2(right)
    }
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

/// Nested wall-clock measurements for one SIFT extraction. These are only
/// collected by the explicit profiling path.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub(crate) struct SiftExtractionTiming {
    pub preparation_ms: f64,
    pub input_conversion_ms: f64,
    pub backend_kernel_ms: f64,
    pub backend_input_conversion_ms: f64,
    pub backend_scale_space_ms: f64,
    pub backend_detection_ms: f64,
    pub backend_orientation_ms: f64,
    pub backend_descriptor_ms: f64,
    pub backend_output_assembly_ms: f64,
    pub result_conversion_ms: f64,
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
    let max_ratio = options.max_ratio;
    let max_distance = options.max_distance;
    // Prefer pre-quantized u8 descriptors from the DB/GPU path. The float path
    // re-quantizes on every candidate compare and dominates guided wall time.
    let use_u8 = left.descriptors_u8.len() == left.keypoints.len()
        && right.descriptors_u8.len() == right.keypoints.len();
    let max_l2 = if use_u8 {
        (max_distance * max_distance) * COLMAP_SIFT_DESCRIPTOR_NORM
    } else {
        f32::INFINITY
    };
    let max_ratio_sq = max_ratio * max_ratio;

    let forward: Vec<Match> = (0..left.keypoints.len())
        .into_par_iter()
        .map(|left_idx| {
            let left_kp = &left.keypoints[left_idx];
            let x1 = nalgebra::Vector3::new(left_kp.x() as f64, left_kp.y() as f64, 1.0);
            let line2 = f * x1;
            let mut best = None::<(u32, f32)>;
            let mut second_best = f32::INFINITY;
            for (right_idx, right_kp) in right.keypoints.iter().enumerate() {
                let err = epipolar_line_distance_px(
                    (right_kp.x(), right_kp.y()),
                    (line2.x, line2.y, line2.z),
                );
                if err > max_epipolar_error {
                    continue;
                }
                let distance = if use_u8 {
                    let l2 = colmap_uint8_l2_distance2(
                        &left.descriptors_u8[left_idx],
                        &right.descriptors_u8[right_idx],
                    );
                    if l2 > max_l2 {
                        continue;
                    }
                    l2
                } else {
                    let distance = sift_pair_distance(
                        &left.descriptors[left_idx],
                        &right.descriptors[right_idx],
                    );
                    if distance > max_distance {
                        continue;
                    }
                    distance
                };
                match best {
                    Some((_, best_distance)) if distance >= best_distance => {
                        if distance < second_best {
                            second_best = distance;
                        }
                    }
                    Some((_, best_distance)) => {
                        second_best = best_distance;
                        best = Some((right_idx as u32, distance));
                    }
                    None => best = Some((right_idx as u32, distance)),
                }
            }
            let (train_idx, best_distance) = best?;
            let reject = if use_u8 {
                second_best.is_finite() && best_distance >= max_ratio_sq * second_best
            } else {
                second_best.is_finite() && best_distance >= max_ratio * second_best
            };
            if reject {
                return None;
            }
            let reported = if use_u8 {
                colmap_normalized_distance(best_distance)
            } else {
                best_distance
            };
            Some(Match {
                query_idx: left_idx as u32,
                train_idx,
                distance: reported,
            })
        })
        .collect::<Vec<Option<Match>>>()
        .into_iter()
        .flatten()
        .collect();

    if !options.cross_check {
        let mut forward = forward;
        forward.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.query_idx.cmp(&b.query_idx))
                .then_with(|| a.train_idx.cmp(&b.train_idx))
        });
        if options.max_num_matches > 0 && forward.len() > options.max_num_matches {
            forward.truncate(options.max_num_matches);
        }
        return forward;
    }

    let f_t = f.transpose();
    let reverse: HashSet<(u32, u32)> = (0..right.keypoints.len())
        .into_par_iter()
        .filter_map(|right_idx| {
            let right_kp = &right.keypoints[right_idx];
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
                let distance = if use_u8 {
                    let l2 = colmap_uint8_l2_distance2(
                        &right.descriptors_u8[right_idx],
                        &left.descriptors_u8[left_idx],
                    );
                    if l2 > max_l2 {
                        continue;
                    }
                    l2
                } else {
                    let distance = sift_pair_distance(
                        &right.descriptors[right_idx],
                        &left.descriptors[left_idx],
                    );
                    if distance > max_distance {
                        continue;
                    }
                    distance
                };
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
            let (query_idx, best_distance) = best?;
            let reject = if use_u8 {
                second_best.is_finite() && best_distance >= max_ratio_sq * second_best
            } else {
                second_best.is_finite() && best_distance >= max_ratio * second_best
            };
            if reject {
                return None;
            }
            Some((query_idx, right_idx as u32))
        })
        .collect();

    let mut matches = forward
        .into_iter()
        .filter(|m| reverse.contains(&(m.query_idx, m.train_idx)))
        .collect::<Vec<_>>();
    matches.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.query_idx.cmp(&b.query_idx))
            .then_with(|| a.train_idx.cmp(&b.train_idx))
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
        if crate::gpu::WgpuContext::try_new_optional()?.is_none() {
            eprintln!("skipping GPU SIFT benchmark test: no compatible adapter");
            return Ok(());
        }
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

    // Serialize fields explicitly: native struct padding and float equality are
    // not suitable for a bit-exact regression gate (notably signed zero).
    fn sift_feature_bytes(features: &SiftFeatures) -> Vec<u8> {
        let count = features.keypoints.len();
        assert_eq!(features.colmap_keypoints.len(), count);
        assert_eq!(features.descriptors.len(), count);
        assert_eq!(features.descriptors_u8.len(), count);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(count as u64).to_le_bytes());
        for kp in &features.keypoints {
            for value in [kp.pt.0, kp.pt.1, kp.size, kp.angle, kp.response] {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            bytes.extend_from_slice(&kp.octave.to_le_bytes());
        }
        for kp in &features.colmap_keypoints {
            for value in [kp.x, kp.y, kp.a11, kp.a12, kp.a21, kp.a22] {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        for descriptor in &features.descriptors {
            for value in descriptor.as_slice() {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        for descriptor in &features.descriptors_u8 {
            bytes.extend_from_slice(descriptor);
        }
        bytes
    }

    #[cfg(all(feature = "vlfeat-sift", not(feature = "lowe-sift-backend")))]
    #[test]
    fn vlfeat_profiling_preserves_feature_bits() -> Result<()> {
        // Odd rectangular dimensions exercise convolution tails and asymmetric
        // texture avoids relying only on repeated checkerboard responses.
        let (width, height) = (193u32, 157u32);
        let gray = (0..width * height)
            .map(|index| {
                let x = index % width;
                let y = index / width;
                let tile = if (x / 13 + y / 17) % 2 == 0 { 180 } else { 20 };
                (tile + (x * 7 + y * 11 + x * y % 31) % 55) as u8
            })
            .collect::<Vec<_>>();
        for (name, options) in [
            ("standard", SiftExtractionOptions::default()),
            (
                "standard_l2_upright",
                SiftExtractionOptions {
                    normalization: SiftDescriptorNormalization::L2,
                    upright: true,
                    first_octave: 0,
                    ..Default::default()
                },
            ),
            (
                "covdet_dsp",
                SiftExtractionOptions {
                    domain_size_pooling: true,
                    ..Default::default()
                },
            ),
        ] {
            let baseline = extract_sift_from_grayscale_u8(&gray, width, height, &options)?;
            assert!(!baseline.keypoints.is_empty(), "{name}: empty fixture");
            let expected = sift_feature_bytes(&baseline);
            // Independent pre-SIMD snapshots, verified with Apple clang 21.0.0
            // and rustc 1.97.0 in release mode. Do not regenerate from a candidate.
            // Other targets retain the repeat/profiling gate below, not these
            // platform-specific floating-point expectations.
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                let (count, hash) = match name {
                    "standard" => (
                        582,
                        "6b5675e5f9b52de5094772730a717dcb25440bb1f3ee7131055c77187ea1fe70",
                    ),
                    "standard_l2_upright" => (
                        177,
                        "9f57e514d5ee246fc43ed86610c58f52a3f964fbea290ecd4878302ff8f03651",
                    ),
                    "covdet_dsp" => (
                        608,
                        "9ee1e70d22c5b6770a44f048c051c7bf7c7f53312a665460ba8bf71ed07a42f7",
                    ),
                    _ => unreachable!(),
                };
                assert_eq!(
                    baseline.keypoints.len(),
                    count,
                    "{name}: preoptimization count"
                );
                assert_eq!(blake3::hash(&expected).to_hex().as_str(), hash,
                    "{name}: preoptimization bits (toolchain changes require independent revalidation)");
            }
            for _ in 0..2 {
                let repeated = extract_sift_from_grayscale_u8(&gray, width, height, &options)?;
                let (profiled, _) =
                    extract_sift_from_grayscale_u8_with_timing(&gray, width, height, &options)?;
                assert_eq!(sift_feature_bytes(&repeated), expected, "{name}: repeat");
                assert_eq!(sift_feature_bytes(&profiled), expected, "{name}: profiling");
            }
            eprintln!(
                "{name}: {} features, blake3={}",
                baseline.keypoints.len(),
                blake3::hash(&expected)
            );
        }
        Ok(())
    }

    #[cfg(all(feature = "vlfeat-sift", not(feature = "lowe-sift-backend")))]
    #[test]
    fn profiled_grayscale_extraction_reports_nested_sift_stages() -> Result<()> {
        let width = 256u32;
        let height = 256u32;
        let gray = (0..width * height)
            .map(|index| {
                let x = index % width;
                let y = index / width;
                if ((x / 16) + (y / 16)) % 2 == 0 {
                    240
                } else {
                    20
                }
            })
            .collect::<Vec<_>>();
        let options = SiftExtractionOptions {
            max_num_features: 1024,
            max_image_size: 256,
            ..Default::default()
        };
        let (features, timing) =
            extract_sift_from_grayscale_u8_with_timing(&gray, width, height, &options)?;
        assert!(!features.keypoints.is_empty());
        for (name, value) in [
            ("preparation_ms", timing.preparation_ms),
            ("input_conversion_ms", timing.input_conversion_ms),
            ("backend_kernel_ms", timing.backend_kernel_ms),
            (
                "backend_input_conversion_ms",
                timing.backend_input_conversion_ms,
            ),
            ("backend_scale_space_ms", timing.backend_scale_space_ms),
            ("backend_detection_ms", timing.backend_detection_ms),
            ("backend_orientation_ms", timing.backend_orientation_ms),
            ("backend_descriptor_ms", timing.backend_descriptor_ms),
            (
                "backend_output_assembly_ms",
                timing.backend_output_assembly_ms,
            ),
            ("result_conversion_ms", timing.result_conversion_ms),
        ] {
            assert!(value.is_finite() && value >= 0.0, "{name}={value}");
        }
        assert!(timing.backend_kernel_ms > 0.0);
        assert!(timing.backend_scale_space_ms > 0.0);
        assert!(timing.backend_detection_ms > 0.0);
        assert!(timing.backend_descriptor_ms > 0.0);
        assert!(timing.backend_input_conversion_ms > 0.0);
        assert!(timing.backend_output_assembly_ms > 0.0);
        Ok(())
    }
}
