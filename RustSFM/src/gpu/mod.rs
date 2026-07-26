//! GPU backends for COLMAP-parity feature extraction and matching.
//!
//! Platform strategy: **wgpu** only (Vulkan / Metal / DX12). CUDA and SiftGPU are
//! intentionally out of scope.

use crate::sift::{SiftExtractionOptions, SiftFeatures};
#[cfg(feature = "gpu-wgpu")]
use anyhow::Context;
use anyhow::{bail, Result};

#[cfg(feature = "gpu-wgpu")]
mod context;
#[cfg(feature = "gpu-wgpu")]
mod matcher;
#[cfg(feature = "gpu-wgpu")]
mod pnp_scorer;
#[cfg(feature = "gpu-wgpu")]
mod scorer;
#[cfg(feature = "gpu-wgpu")]
mod sift;

#[cfg(feature = "gpu-wgpu")]
pub use context::WgpuContext;
#[cfg(feature = "gpu-wgpu")]
pub use matcher::WgpuSiftMatcher;
#[cfg(feature = "gpu-wgpu")]
pub use pnp_scorer::WgpuPnpModelScorer;
#[cfg(all(feature = "gpu-wgpu", test))]
pub(crate) use pnp_scorer::{GpuPnpImagePoint, GpuPnpModel, GpuPnpObjectPoint};
#[cfg(feature = "gpu-wgpu")]
pub(crate) use scorer::WgpuModelScoringSession;
#[cfg(feature = "gpu-wgpu")]
pub use scorer::{GpuModelSupport, TwoViewModelKind, WgpuModelScorer};

#[cfg(feature = "gpu-wgpu")]
use self::sift::{SiftDescriptorComputer, SiftDetector, SiftOrientationAssigner, SiftPyramid};
#[cfg(feature = "gpu-wgpu")]
use crate::database::ColmapKeypoint;
#[cfg(feature = "gpu-wgpu")]
use crate::sift::SiftDescriptorNormalization;
#[cfg(feature = "gpu-wgpu")]
use lowe_sift::Descriptor;
#[cfg(feature = "gpu-wgpu")]
use rustslam::KeyPoint;
#[cfg(feature = "gpu-wgpu")]
use std::cmp::Ordering;
#[cfg(feature = "gpu-wgpu")]
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackendKind {
    Wgpu,
    Vulkan,
}

#[derive(Debug, Clone)]
pub struct GpuSiftCapabilities {
    pub backend: GpuBackendKind,
    pub device_name: String,
}

pub trait GpuSiftExtractor {
    fn capabilities(&self) -> &GpuSiftCapabilities;

    fn extract_sift(
        &self,
        rgb: &[u8],
        width: u32,
        height: u32,
        options: &SiftExtractionOptions,
    ) -> Result<SiftFeatures>;
}

pub fn validate_gpu_sift_options(options: &SiftExtractionOptions) -> Result<()> {
    if options.estimate_affine_shape {
        bail!("wgpu SIFT does not support affine shape estimation");
    }
    if options.domain_size_pooling {
        bail!("wgpu SIFT does not support domain-size pooling");
    }
    if options.force_covariant_extractor {
        bail!("wgpu SIFT does not support the covariant extractor");
    }
    if options.first_octave < -1 {
        bail!("wgpu SIFT first_octave must be >= -1");
    }
    Ok(())
}

#[cfg(feature = "gpu-wgpu")]
struct GpuFeatureRecord {
    keypoint: sift::GpuKeypoint,
    descriptor: [f32; lowe_sift::DESCRIPTOR_LEN],
}

#[cfg(feature = "gpu-wgpu")]
pub struct WgpuSiftExtractor {
    pyramid: SiftPyramid,
    detector: SiftDetector,
    orientation: SiftOrientationAssigner,
    descriptor: SiftDescriptorComputer,
    capabilities: GpuSiftCapabilities,
}

#[cfg(not(feature = "gpu-wgpu"))]
pub struct WgpuSiftExtractor;

#[cfg(feature = "gpu-wgpu")]
impl WgpuSiftExtractor {
    pub fn try_new() -> Result<Self> {
        Self::from_context(WgpuContext::try_new()?)
    }

    pub fn from_context(context: Arc<WgpuContext>) -> Result<Self> {
        Ok(Self {
            capabilities: context.capabilities().clone(),
            pyramid: SiftPyramid::new(context.clone())?,
            detector: SiftDetector::new(context.clone())?,
            orientation: SiftOrientationAssigner::new(context.clone())?,
            descriptor: SiftDescriptorComputer::new(context.clone())?,
        })
    }

    pub fn extract_grayscale(
        &self,
        gray: &[u8],
        width: u32,
        height: u32,
        options: &SiftExtractionOptions,
    ) -> Result<SiftFeatures> {
        validate_gpu_sift_options(options)?;
        validate_gray_buffer(gray, width, height)?;
        let (gray, width, height) = crate::sift::prepare_grayscale_for_extraction(
            gray,
            width,
            height,
            options.max_image_size,
        )?;
        let plan = sift::SiftPlan::new(width, height, options)?;
        let Some(first_octave) = plan.octaves.first() else {
            return Ok(SiftFeatures::default());
        };
        let base = gray
            .iter()
            .map(|&value| f32::from(value) / 255.0)
            .collect::<Vec<_>>();
        let mut octave_base = resize_f32(
            &base,
            width,
            height,
            first_octave.width,
            first_octave.height,
        )?;
        let mut records = Vec::new();
        let intervals = options.octave_resolution;
        let sigma0 = 1.6f32;
        let root_sift = matches!(options.normalization, SiftDescriptorNormalization::L1Root);

        for (ordinal, octave) in plan.octaves.iter().enumerate() {
            let (octave_width, octave_height) = octave.dimensions();
            if octave_base.len() != octave.pixel_count()? {
                bail!(
                    "GPU SIFT octave base has {} pixels, expected {}",
                    octave_base.len(),
                    octave.pixel_count()?
                );
            }

            let mut gaussians = Vec::with_capacity(octave.gaussian_levels);
            let first = self
                .pyramid
                .gaussian(&octave_base, octave_width, octave_height, sigma0)?;
            gaussians.push(first);
            for level in 1..octave.gaussian_levels {
                let previous_sigma = sigma0 * 2.0f32.powf((level - 1) as f32 / intervals as f32);
                let sigma = sigma0 * 2.0f32.powf(level as f32 / intervals as f32);
                let incremental_sigma = (sigma * sigma - previous_sigma * previous_sigma)
                    .max(1.0e-6)
                    .sqrt();
                let next = self.pyramid.gaussian(
                    gaussians
                        .last()
                        .context("GPU SIFT Gaussian level missing")?,
                    octave_width,
                    octave_height,
                    incremental_sigma,
                )?;
                gaussians.push(next);
            }

            let mut dogs = Vec::with_capacity(octave.dog_levels);
            for level in 0..octave.dog_levels {
                dogs.push(self.pyramid.dog(
                    &gaussians[level],
                    &gaussians[level + 1],
                    octave_width,
                    octave_height,
                )?);
            }
            let pixels = octave.pixel_count()?;
            let mut dog_volume = Vec::with_capacity(pixels * dogs.len());
            for dog in &dogs {
                dog_volume.extend_from_slice(dog);
            }
            let max_capacity = plan
                .candidate_capacity
                .saturating_mul(8)
                .max(plan.candidate_capacity);
            let candidates = self.detector.detect_volume_with_retry(
                &dog_volume,
                sift::DetectorParams {
                    width: octave_width,
                    height: octave_height,
                    levels: octave.dog_levels as u32,
                    capacity: plan.candidate_capacity,
                    peak_threshold: options.peak_threshold as f32,
                    edge_threshold: options.edge_threshold as f32,
                    sigma0,
                    octave_scale: 1.0,
                    octave: octave.octave,
                    octave_resolution: intervals as u32,
                    pad0: 0,
                    pad1: 0,
                },
                max_capacity,
            )?;

            for level in 1..octave.dog_levels.saturating_sub(1) {
                let level_points = candidates
                    .iter()
                    .filter(|point| point.level == level as i32)
                    .copied()
                    .collect::<Vec<_>>();
                if level_points.is_empty() {
                    continue;
                }
                let level_image = &gaussians[level];
                let oriented = self.orientation.assign(
                    level_image,
                    octave_width,
                    octave_height,
                    &level_points,
                    options.max_num_orientations as u32,
                    options.upright,
                )?;
                let descriptors = self.descriptor.compute(
                    level_image,
                    octave_width,
                    octave_height,
                    &oriented,
                    root_sift,
                )?;
                records.extend(oriented.into_iter().zip(descriptors).map(
                    |(keypoint, descriptor)| GpuFeatureRecord {
                        keypoint,
                        descriptor,
                    },
                ));
            }

            if ordinal + 1 < plan.octaves.len() {
                octave_base =
                    self.pyramid
                        .downsample(&gaussians[intervals], octave_width, octave_height)?;
            }
        }

        records.sort_by(gpu_feature_order);
        records.truncate(options.max_num_features);
        Ok(records_to_sift_features(records))
    }
}

#[cfg(not(feature = "gpu-wgpu"))]
impl WgpuSiftExtractor {
    pub fn try_new() -> Result<Self> {
        bail!("RustSFM was built without gpu-wgpu support")
    }
}

#[cfg(feature = "gpu-wgpu")]
impl GpuSiftExtractor for WgpuSiftExtractor {
    fn capabilities(&self) -> &GpuSiftCapabilities {
        &self.capabilities
    }

    fn extract_sift(
        &self,
        rgb: &[u8],
        width: u32,
        height: u32,
        options: &SiftExtractionOptions,
    ) -> Result<SiftFeatures> {
        let gray = crate::sift::rgb_to_colmap_gray_u8(rgb, width, height)?;
        self.extract_grayscale(&gray, width, height, options)
    }
}

#[cfg(not(feature = "gpu-wgpu"))]
impl GpuSiftExtractor for WgpuSiftExtractor {
    fn capabilities(&self) -> &GpuSiftCapabilities {
        static CAPS: std::sync::OnceLock<GpuSiftCapabilities> = std::sync::OnceLock::new();
        CAPS.get_or_init(|| GpuSiftCapabilities {
            backend: GpuBackendKind::Wgpu,
            device_name: "unavailable".to_string(),
        })
    }

    fn extract_sift(
        &self,
        _rgb: &[u8],
        _width: u32,
        _height: u32,
        _options: &SiftExtractionOptions,
    ) -> Result<SiftFeatures> {
        bail!("RustSFM was built without gpu-wgpu support")
    }
}

#[cfg(feature = "gpu-wgpu")]
fn validate_gray_buffer(gray: &[u8], width: u32, height: u32) -> Result<()> {
    let expected = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|count| usize::try_from(count).ok())
        .context("GPU SIFT grayscale image size overflow")?;
    if gray.len() != expected {
        bail!(
            "GPU SIFT grayscale buffer length {} does not match {}x{}",
            gray.len(),
            width,
            height
        );
    }
    Ok(())
}

#[cfg(feature = "gpu-wgpu")]
fn resize_f32(
    input: &[f32],
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
) -> Result<Vec<f32>> {
    if input_width == output_width && input_height == output_height {
        return Ok(input.to_vec());
    }
    if input_width == 0 || input_height == 0 || output_width == 0 || output_height == 0 {
        bail!("GPU SIFT resize dimensions must be non-zero");
    }
    let output_count = u64::from(output_width)
        .checked_mul(u64::from(output_height))
        .and_then(|count| usize::try_from(count).ok())
        .context("GPU SIFT resized image size overflow")?;
    let mut output = vec![0.0; output_count];
    let x_scale = input_width as f32 / output_width as f32;
    let y_scale = input_height as f32 / output_height as f32;
    for y in 0..output_height {
        let source_y = ((y as f32 + 0.5) * y_scale - 0.5).clamp(0.0, (input_height - 1) as f32);
        let y0 = source_y.floor() as u32;
        let y1 = (y0 + 1).min(input_height - 1);
        let fy = source_y - y0 as f32;
        for x in 0..output_width {
            let source_x = ((x as f32 + 0.5) * x_scale - 0.5).clamp(0.0, (input_width - 1) as f32);
            let x0 = source_x.floor() as u32;
            let x1 = (x0 + 1).min(input_width - 1);
            let fx = source_x - x0 as f32;
            let top = input[(y0 * input_width + x0) as usize] * (1.0 - fx)
                + input[(y0 * input_width + x1) as usize] * fx;
            let bottom = input[(y1 * input_width + x0) as usize] * (1.0 - fx)
                + input[(y1 * input_width + x1) as usize] * fx;
            output[(y * output_width + x) as usize] = top * (1.0 - fy) + bottom * fy;
        }
    }
    Ok(output)
}

#[cfg(feature = "gpu-wgpu")]
fn gpu_feature_order(left: &GpuFeatureRecord, right: &GpuFeatureRecord) -> Ordering {
    let left_scale = left.keypoint.sigma * 2.0f32.powi(left.keypoint.octave);
    let right_scale = right.keypoint.sigma * 2.0f32.powi(right.keypoint.octave);
    right_scale
        .total_cmp(&left_scale)
        .then_with(|| right.keypoint.response.total_cmp(&left.keypoint.response))
        .then_with(|| left.keypoint.octave.cmp(&right.keypoint.octave))
        .then_with(|| left.keypoint.level.cmp(&right.keypoint.level))
        .then_with(|| left.keypoint.y.total_cmp(&right.keypoint.y))
        .then_with(|| left.keypoint.x.total_cmp(&right.keypoint.x))
        .then_with(|| left.keypoint.angle.total_cmp(&right.keypoint.angle))
}

#[cfg(feature = "gpu-wgpu")]
fn records_to_sift_features(records: Vec<GpuFeatureRecord>) -> SiftFeatures {
    let mut output = SiftFeatures {
        keypoints: Vec::with_capacity(records.len()),
        descriptors: Vec::with_capacity(records.len()),
        colmap_keypoints: Vec::with_capacity(records.len()),
        descriptors_u8: Vec::with_capacity(records.len()),
    };
    for record in records {
        let factor = 2.0f32.powi(record.keypoint.octave);
        let x = record.keypoint.x * factor;
        let y = record.keypoint.y * factor;
        let scale = record.keypoint.sigma * factor;
        let size = 2.0 * scale;
        let angle = record.keypoint.angle;
        output.keypoints.push(KeyPoint {
            pt: (x, y),
            size,
            angle,
            response: record.keypoint.response,
            octave: record.keypoint.octave,
        });
        output
            .colmap_keypoints
            .push(ColmapKeypoint::from_scale_orientation(x, y, size, angle));
        output.descriptors.push(Descriptor::new(record.descriptor));
        output
            .descriptors_u8
            .push(sift::quantize_gpu_descriptor(&record.descriptor));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "gpu-wgpu")]
    use wgpu::util::DeviceExt;

    #[cfg(all(feature = "gpu-wgpu", not(feature = "gpu-vulkan")))]
    #[test]
    fn wgpu_context_reports_a_real_adapter_when_available() -> Result<()> {
        let Some(context) = WgpuContext::try_new_optional()? else {
            eprintln!("skipping GPU smoke test: no compatible adapter");
            return Ok(());
        };
        assert!(!context.capabilities().device_name.trim().is_empty());
        assert_eq!(context.capabilities().backend, GpuBackendKind::Wgpu);
        Ok(())
    }

    #[cfg(feature = "gpu-vulkan")]
    #[test]
    fn wgpu_context_requires_vulkan_adapter() -> Result<()> {
        let context = WgpuContext::try_new()?;

        assert_eq!(context.backend(), wgpu::Backend::Vulkan);
        assert_eq!(context.capabilities().backend, GpuBackendKind::Vulkan);
        Ok(())
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn wgpu_context_reads_back_a_storage_buffer() -> Result<()> {
        let Some(context) = WgpuContext::try_new_optional()? else {
            eprintln!("skipping GPU readback test: no compatible adapter");
            return Ok(());
        };
        let expected = [3u32, 5, 8, 13];
        let buffer = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rustsfm readback test input"),
                contents: bytemuck::cast_slice(&expected),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let actual = context.read_buffer::<u32>(&buffer, expected.len())?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn gpu_sift_rejects_covariant_modes_before_device_creation() {
        let options = SiftExtractionOptions {
            use_gpu: true,
            estimate_affine_shape: true,
            ..SiftExtractionOptions::default()
        };
        let error = validate_gpu_sift_options(&options).unwrap_err();
        assert!(error.to_string().contains("affine shape"));
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn wgpu_sift_checkerboard_produces_aligned_colmap_outputs() -> Result<()> {
        let Some(context) = WgpuContext::try_new_optional()? else {
            eprintln!("skipping GPU extractor test: no compatible adapter");
            return Ok(());
        };
        let extractor = WgpuSiftExtractor::from_context(context)?;
        let gray = checkerboard_u8(256, 256, 16);
        let options = SiftExtractionOptions {
            use_gpu: true,
            max_num_features: 512,
            ..SiftExtractionOptions::default()
        };
        let features = extractor.extract_grayscale(&gray, 256, 256, &options)?;
        assert!(!features.keypoints.is_empty());
        assert!(features.keypoints.len() <= 512);
        assert_eq!(features.keypoints.len(), features.descriptors.len());
        assert_eq!(features.keypoints.len(), features.colmap_keypoints.len());
        assert_eq!(features.keypoints.len(), features.descriptors_u8.len());
        assert!(features
            .keypoints
            .iter()
            .all(|point| { point.x().is_finite() && point.y().is_finite() && point.size > 0.0 }));
        Ok(())
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn wgpu_sift_constant_image_returns_no_features() -> Result<()> {
        let Some(context) = WgpuContext::try_new_optional()? else {
            eprintln!("skipping GPU extractor test: no compatible adapter");
            return Ok(());
        };
        let extractor = WgpuSiftExtractor::from_context(context)?;
        let features = extractor.extract_grayscale(
            &vec![127; 128 * 96],
            128,
            96,
            &SiftExtractionOptions {
                use_gpu: true,
                ..SiftExtractionOptions::default()
            },
        )?;
        assert!(features.keypoints.is_empty());
        assert!(features.descriptors.is_empty());
        assert!(features.colmap_keypoints.is_empty());
        assert!(features.descriptors_u8.is_empty());
        Ok(())
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn wgpu_sift_matcher_applies_ratio_distance_and_cross_check() -> Result<()> {
        let Some(context) = WgpuContext::try_new_optional()? else {
            eprintln!("skipping GPU matcher test: no compatible adapter");
            return Ok(());
        };
        let mut zero = [0u8; 128];
        let mut full = [255u8; 128];
        let mut middle = [128u8; 128];
        zero[0] = 1;
        full[0] = 254;
        middle[0] = 127;
        let left = [zero, full];
        let right = [zero, full, middle];
        let options = crate::sift::SiftMatchingOptions {
            use_gpu: true,
            max_ratio: 0.8,
            max_distance: 0.7,
            cross_check: true,
            max_num_matches: 16,
            ..Default::default()
        };
        let matches =
            WgpuSiftMatcher::from_context(context)?.match_descriptors(&left, &right, &options)?;
        assert_eq!(
            matches
                .iter()
                .map(|value| (value.query_idx, value.train_idx))
                .collect::<Vec<_>>(),
            vec![(0, 0), (1, 1)]
        );
        Ok(())
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn wgpu_model_scorer_scores_homographies_and_reads_mask() -> Result<()> {
        let Some(context) = WgpuContext::try_new_optional()? else {
            eprintln!("skipping GPU model scorer test: no compatible adapter");
            return Ok(());
        };
        let scorer = WgpuModelScorer::from_context(context)?;
        let points1 = [[0.0, 0.0], [1.0, 2.0], [-3.0, 4.0]];
        let points2 = points1;
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let translated = [1.0, 0.0, 10.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let summaries = scorer.score_two_view_models(
            &[identity, translated],
            &points1,
            &points2,
            0.1,
            TwoViewModelKind::HomographyForward,
        )?;
        assert_eq!(summaries[0].inliers, 3);
        assert!(summaries[0].residual_sum.abs() < 1.0e-6);
        assert_eq!(summaries[1].inliers, 0);
        assert_eq!(
            scorer.inlier_mask(
                &identity,
                &points1,
                &points2,
                0.1,
                TwoViewModelKind::HomographyForward,
            )?,
            vec![true, true, true]
        );
        Ok(())
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn wgpu_model_scorer_matches_sampson_support() -> Result<()> {
        let Some(context) = WgpuContext::try_new_optional()? else {
            eprintln!("skipping GPU model scorer test: no compatible adapter");
            return Ok(());
        };
        let scorer = WgpuModelScorer::from_context(context)?;
        let model = [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0];
        let points1 = [[0.0, 0.0], [1.0, 2.0], [-3.0, 4.0]];
        let points2 = [[5.0, 0.0], [2.0, 2.0], [1.0, 5.0]];
        let summaries = scorer.score_two_view_models(
            &[model],
            &points1,
            &points2,
            0.1,
            TwoViewModelKind::Sampson,
        )?;
        assert_eq!(summaries[0].inliers, 2);
        assert!(summaries[0].residual_sum.abs() < 1.0e-6);
        assert_eq!(
            scorer.inlier_mask(&model, &points1, &points2, 0.1, TwoViewModelKind::Sampson,)?,
            vec![true, true, false]
        );
        let boundary = scorer.score_two_view_models(
            &[model],
            &points1,
            &points2,
            1.0,
            TwoViewModelKind::Sampson,
        )?;
        assert_eq!(boundary[0].inliers, 3);
        assert!((boundary[0].residual_sum - 0.5).abs() < 1.0e-6);
        Ok(())
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn wgpu_model_scorer_preserves_homogeneous_sampson_scaling() -> Result<()> {
        let Some(context) = WgpuContext::try_new_optional()? else {
            eprintln!("skipping GPU model scorer test: no compatible adapter");
            return Ok(());
        };
        let scorer = WgpuModelScorer::from_context(context)?;
        let model = [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0];
        let points1 = [[1.0, 4.0, 2.0]];
        let points2 = [[3.0, 6.0, 2.0]];
        let rejected = scorer.score_homogeneous_two_view_models(
            &[model],
            &points1,
            &points2,
            1.0,
            TwoViewModelKind::Sampson,
        )?;
        assert_eq!(rejected[0].inliers, 0);
        let accepted = scorer.score_homogeneous_two_view_models(
            &[model],
            &points1,
            &points2,
            2.0,
            TwoViewModelKind::Sampson,
        )?;
        assert_eq!(accepted[0].inliers, 1);
        assert!((accepted[0].residual_sum - 2.0).abs() < 1.0e-6);
        Ok(())
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn wgpu_model_scorer_keeps_degenerate_models_outliers() -> Result<()> {
        let Some(context) = WgpuContext::try_new_optional()? else {
            eprintln!("skipping GPU model scorer test: no compatible adapter");
            return Ok(());
        };
        let scorer = WgpuModelScorer::from_context(context)?;
        let support = scorer.score_two_view_models(
            &[[0.0; 9]],
            &[[1.0, 2.0]],
            &[[1.0, 2.0]],
            f32::MAX,
            TwoViewModelKind::HomographyForward,
        )?;
        assert_eq!(support[0].inliers, 0);
        Ok(())
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn wgpu_model_scorer_validates_inputs_and_handles_empty_observations() -> Result<()> {
        let Some(context) = WgpuContext::try_new_optional()? else {
            eprintln!("skipping GPU model scorer test: no compatible adapter");
            return Ok(());
        };
        let scorer = WgpuModelScorer::from_context(context)?;
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(
            scorer.score_two_view_models(
                &[identity],
                &[],
                &[],
                1.0,
                TwoViewModelKind::HomographyForward,
            )?,
            vec![GpuModelSupport::default()]
        );
        assert!(scorer
            .inlier_mask(
                &identity,
                &[],
                &[],
                1.0,
                TwoViewModelKind::HomographyForward,
            )?
            .is_empty());
        assert!(scorer
            .score_two_view_models(
                &[identity],
                &[[0.0, 0.0]],
                &[],
                1.0,
                TwoViewModelKind::HomographyForward,
            )
            .unwrap_err()
            .to_string()
            .contains("point count mismatch"));
        assert!(scorer
            .score_two_view_models(
                &[identity],
                &[],
                &[],
                -1.0,
                TwoViewModelKind::HomographyForward,
            )
            .unwrap_err()
            .to_string()
            .contains("finite and non-negative"));
        Ok(())
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn wgpu_pnp_abi_records_are_wgsl_aligned() {
        assert_eq!(std::mem::size_of::<GpuPnpImagePoint>(), 16);
        assert_eq!(std::mem::size_of::<GpuPnpObjectPoint>(), 16);
        assert_eq!(std::mem::size_of::<GpuPnpModel>(), 48);
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn wgpu_pnp_scorer_matches_cpu_projection_and_mask() -> Result<()> {
        let Some(context) = WgpuContext::try_new_optional()? else {
            eprintln!("skipping GPU PnP scorer test: no compatible adapter");
            return Ok(());
        };
        let mut scorer = WgpuPnpModelScorer::from_context(context)?;
        let image = [[0.0, 0.0], [0.1, 0.0], [-0.2, 0.2], [0.0, 0.0]];
        let world = [
            [0.0, 0.0, 2.0],
            [0.2, 0.0, 2.0],
            [-0.4, 0.4, 2.0],
            [0.0, 0.0, -1.0],
        ];
        scorer.prepare(&image, &world, 0.01)?;
        let supports = scorer.score_models(&[rustslam::SE3::identity()])?;
        let mask = scorer.inlier_mask(&rustslam::SE3::identity())?;
        assert_eq!(supports[0].inliers, 4);
        assert_eq!(mask, vec![true, true, true, true]);
        Ok(())
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn wgpu_pnp_mask_requires_model_from_latest_scoring_batch() -> Result<()> {
        let Some(context) = WgpuContext::try_new_optional()? else {
            eprintln!("skipping GPU PnP scorer test: no compatible adapter");
            return Ok(());
        };
        let mut scorer = WgpuPnpModelScorer::from_context(context)?;
        let image = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [-0.1, 0.0]];
        let world = [
            [0.0, 0.0, 2.0],
            [0.2, 0.0, 2.0],
            [0.0, 0.2, 2.0],
            [-0.2, 0.0, 2.0],
        ];
        let model = rustslam::SE3::identity();
        scorer.prepare(&image, &world, 0.01)?;

        let error = scorer
            .inlier_mask(&model)
            .expect_err("mask lookup must not rescore an unscored model");
        assert!(error.to_string().contains("latest scoring batch"));

        scorer.score_models(&[model])?;
        assert_eq!(scorer.inlier_mask(&model)?, vec![true; 4]);
        scorer.prepare(&image, &world, 0.01)?;
        let error = scorer
            .inlier_mask(&model)
            .expect_err("prepare must invalidate the previous scoring batch");
        assert!(error.to_string().contains("latest scoring batch"));
        Ok(())
    }

    #[cfg(feature = "gpu-wgpu")]
    fn checkerboard_u8(width: u32, height: u32, tile: u32) -> Vec<u8> {
        (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    if ((x / tile) + (y / tile)) % 2 == 0 {
                        240
                    } else {
                        20
                    }
                })
            })
            .collect()
    }
}
