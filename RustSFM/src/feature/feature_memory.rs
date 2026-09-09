//! Allocation planning, NOT an RSS/allocator upper bound. Codec scratch, allocator
//! fragmentation and data-dependent detector candidates can exceed this estimate.
use crate::sift::SiftExtractionOptions;
use crate::task::SfmTaskControl;
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

pub(super) fn standard(options: &SiftExtractionOptions) -> bool {
    cfg!(all(
        feature = "vlfeat-sift",
        not(feature = "lowe-sift-backend")
    )) && !options.use_gpu
        && !options.uses_covariant_extractor()
}

fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).context("SIFT memory estimate overflow")
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).context("SIFT memory estimate overflow")
}

fn dimensions(w: u32, h: u32, limit: usize) -> Result<(u32, u32)> {
    if w == 0 || h == 0 {
        bail!("cannot estimate an empty image");
    }
    if limit > 0 && w.max(h) as usize > limit {
        let scale = limit as f64 / f64::from(w.max(h));
        Ok((
            ((f64::from(w) * scale).round() as u32).max(1),
            ((f64::from(h) * scale).round() as u32).max(1),
        ))
    } else {
        Ok((w, h))
    }
}

fn core(w: u32, h: u32, options: &SiftExtractionOptions) -> Result<u64> {
    let shift = options.first_octave.unsigned_abs();
    if shift >= 31
        || options.octave_resolution > (i32::MAX - 3) as usize
        || options.num_octaves > i32::MAX as usize
        || options.max_num_features > i32::MAX as usize
        || options.max_num_orientations > i32::MAX as usize
        || w > i32::MAX as u32
        || h > i32::MAX as u32
    {
        bail!("SIFT dimensions/options exceed native integer range");
    }
    let scale = |v: u32| -> Result<u64> {
        if options.first_octave < 0 {
            mul(u64::from(v), 1u64 << shift)
        } else {
            Ok(u64::from(v) >> shift)
        }
    };
    let n = mul(scale(w)?, scale(h)?)?;
    // vl_sift_new computes w*h in signed int before allocating its four arrays.
    if n == 0 || n > i32::MAX as u64 {
        bail!("SIFT octave pixels exceed native integer range");
    }
    mul(
        mul(4, n)?,
        add(mul(4, options.octave_resolution as u64)?, 10)?,
    )
}

pub(super) fn estimate(
    w: u32,
    h: u32,
    encoded: u64,
    options: &SiftExtractionOptions,
) -> Result<u64> {
    options.check()?;
    let original = mul(u64::from(w), u64::from(h))?;
    let (w, h) = dimensions(w, h, options.max_image_size)?;
    let pixels = mul(u64::from(w), u64::from(h))?;
    let scratch = core(w, h, options)?;
    // Decode planning: up to RGBA32F (16 B/pixel), RGB8 conversion (3),
    // grayscale/native-to-Rust copy and preparation clone (3), plus encoded file.
    // Resize: gray output + float intermediate (image::resize), native float input.
    let input = add(add(mul(original, 22)?, mul(pixels, 9)?)?, encoded)?;
    // Native levels retain candidates BEFORE max_num_features filtering and may
    // keep a whole extra scale. Reserve additional scratch-sized headroom / 4;
    // this is a planning allowance, not a proven bound on candidate cardinality.
    // Planned output rows include orientations. 2048 B/row covers native paired
    // keypoints/u8, float descriptors, Rust copies and DB serialization together.
    // max_num_features sizes ONLY planned output, never total detector allocation.
    let rows = mul(
        options.max_num_features as u64,
        options.max_num_orientations.max(1) as u64,
    )?;
    add(add(add(scratch, scratch / 4)?, input)?, mul(rows, 2048)?)
}

pub(super) fn image_estimate(
    path: &Path,
    options: &SiftExtractionOptions,
    floor: u64,
    control: &SfmTaskControl,
) -> Result<u64> {
    control.checkpoint()?;
    if !standard(options) {
        // Unsupported backends retain the caller/default stage estimate contract;
        // do not misrepresent it as a dimension-derived conservative formula.
        return Ok(floor);
    }
    let (w, h, encoded) = image_header(path, control)?;
    Ok(estimate(w, h, encoded, options)?.max(floor))
}

fn image_header(path: &Path, control: &SfmTaskControl) -> Result<(u32, u32, u64)> {
    control.checkpoint()?;
    let metadata = match std::fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            bail!(
                "missing image file before memory estimation: {}",
                path.display()
            );
        }
        Err(error) => {
            return Err(error).with_context(|| format!("cannot stat image: {}", path.display()));
        }
    };
    let (w, h) = image::image_dimensions(path).with_context(|| {
        format!(
            "cannot read image dimensions before decode: {}",
            path.display()
        )
    })?;
    let encoded = metadata.len();
    control.checkpoint()?;
    Ok((w, h, encoded))
}

/// Fixed-size chunks match the retained Vec of outputs in the parallel DB path.
/// Return the true single-image request even if it exceeds budget: admission must
/// reject it (and report it), rather than silently clamping the estimate.
pub(super) fn batch_plan(
    estimates: &[u64],
    threads: usize,
    budget: u64,
    floor: u64,
) -> Result<(usize, u64)> {
    let largest = estimates.iter().copied().max().unwrap_or(0).max(floor);
    let count = if largest == 0 {
        1
    } else {
        (budget / largest).max(1)
    };
    let size = threads
        .max(1)
        .min(usize::try_from(count).unwrap_or(usize::MAX));
    let mut request = floor;
    for batch in estimates.chunks(size) {
        let sum = batch.iter().try_fold(0, |sum, &bytes| add(sum, bytes))?;
        request = request.max(sum);
    }
    Ok((size, request))
}

/// Immutable association between ordered inputs, effective SIFT options and
/// fixed chunk boundaries. Execute these inputs/options unchanged; fewer granted
/// CPUs must not regroup chunks. Estimates are not allocator/RSS bounds.
#[derive(Debug, Clone)]
pub(crate) struct FeatureMemoryPlan {
    paths: Vec<PathBuf>,
    options: SiftExtractionOptions,
    batch_size: usize,
    request_bytes: u64,
    retained_bytes: u64,
    headers: Vec<(u32, u32, u64)>,
    retains_frames: bool,
}

impl FeatureMemoryPlan {
    pub(crate) fn paths(&self) -> &[PathBuf] {
        &self.paths
    }
    pub(crate) fn options(&self) -> &SiftExtractionOptions {
        &self.options
    }
    pub(crate) fn batch_size(&self) -> usize {
        self.batch_size
    }
    pub(crate) fn request_bytes(&self) -> u64 {
        self.request_bytes
    }
    pub(crate) fn retained_bytes(&self) -> u64 {
        self.retained_bytes
    }

    /// Recheck saved dimensions and encoded lengths before decode, including
    /// retained/image-only inputs. Does not replan, rebind or mutate this plan.
    /// Header equality is not content authentication or a filesystem snapshot.
    pub(crate) fn validate_inputs(&self, control: &SfmTaskControl) -> Result<()> {
        self.verify_headers(&self.paths, control)
    }

    fn verify_headers(&self, paths: &[PathBuf], control: &SfmTaskControl) -> Result<()> {
        control.checkpoint()?;
        if paths.len() != self.headers.len() {
            bail!("feature memory plan path/header count mismatch");
        }
        for (path, expected) in paths.iter().zip(&self.headers) {
            if image_header(path, control)? != *expected {
                bail!(
                    "planned extraction header differs from the memory plan: {}",
                    path.display()
                );
            }
        }
        control.checkpoint()?;
        Ok(())
    }

    /// Positional source -> staged mapping; only paths may change. Validate all
    /// headers before extraction, including encoded length (copies/links only,
    /// not re-encoding). This is a memory-input check, not content authentication.
    pub(super) fn rebind_db_paths(
        &self,
        paths: &[PathBuf],
        options: &SiftExtractionOptions,
        control: &SfmTaskControl,
    ) -> Result<Self> {
        control.checkpoint()?;
        if self.retains_frames || !standard(options) {
            bail!("planned selected extraction requires a standard CPU SIFT DB plan");
        }
        options.check()?;
        if !same_options(&self.options, options) {
            bail!("planned selected extraction options differ from the memory plan");
        }
        if paths.len() != self.paths.len() {
            bail!("planned selected extraction path count differs from the memory plan");
        }
        self.verify_headers(paths, control)?;
        let mut rebound = self.clone();
        rebound.paths = paths.to_vec();
        Ok(rebound)
    }
}

fn same_options(left: &SiftExtractionOptions, right: &SiftExtractionOptions) -> bool {
    // Exhaustive destructuring makes new options a compile-time review point.
    let SiftExtractionOptions {
        max_num_features,
        first_octave,
        num_octaves,
        octave_resolution,
        peak_threshold,
        edge_threshold,
        estimate_affine_shape,
        max_num_orientations,
        upright,
        domain_size_pooling,
        dsp_min_scale,
        dsp_max_scale,
        dsp_num_scales,
        normalization,
        force_covariant_extractor,
        use_gpu,
        max_image_size,
    } = right;
    left.max_num_features == *max_num_features
        && left.first_octave == *first_octave
        && left.num_octaves == *num_octaves
        && left.octave_resolution == *octave_resolution
        && left.peak_threshold.to_bits() == peak_threshold.to_bits()
        && left.edge_threshold.to_bits() == edge_threshold.to_bits()
        && left.estimate_affine_shape == *estimate_affine_shape
        && left.max_num_orientations == *max_num_orientations
        && left.upright == *upright
        && left.domain_size_pooling == *domain_size_pooling
        && left.dsp_min_scale.to_bits() == dsp_min_scale.to_bits()
        && left.dsp_max_scale.to_bits() == dsp_max_scale.to_bits()
        && left.dsp_num_scales == *dsp_num_scales
        && std::mem::discriminant(&left.normalization) == std::mem::discriminant(normalization)
        && left.force_covariant_extractor == *force_covariant_extractor
        && left.use_gpu == *use_gpu
        && left.max_image_size == *max_image_size
}

/// Plan only the missing-feature paths selected by the caller's DB/cache logic.
/// No database is opened or mutated here. `None` preserves the legacy contract
/// for unbound contexts or unsupported backends (including GPU/covariant SIFT).
/// `other_work_bytes` is ADDITIVE live caller work, not a floor or remaining grant.
/// `budget` is used only to choose chunks; admission decides whether to wait/fail.
pub(crate) fn db_memory_plan(
    paths: &[PathBuf],
    options: &SiftExtractionOptions,
    explicit: bool,
    threads: usize,
    budget: u64,
    other_work_bytes: u64,
    control: &SfmTaskControl,
) -> Result<Option<FeatureMemoryPlan>> {
    composition_plan(
        paths,
        options,
        explicit,
        threads,
        budget,
        other_work_bytes,
        control,
        false,
    )
}

/// SIFT image-only plan retaining ALL frames, including duplicate keypoints,
/// float SIFT/wide descriptors, colors and indices. Call only for SIFT, not ORB.
/// Caller must supply effective options (notably the max_features override).
/// Conservatively counts output again inside the existing VLFeat estimate;
/// original-size grayscale/f32/wide preparation gets extra transient headroom.
pub(crate) fn retained_memory_plan(
    paths: &[PathBuf],
    options: &SiftExtractionOptions,
    explicit: bool,
    threads: usize,
    budget: u64,
    other_work_bytes: u64,
    control: &SfmTaskControl,
) -> Result<Option<FeatureMemoryPlan>> {
    composition_plan(
        paths,
        options,
        explicit,
        threads,
        budget,
        other_work_bytes,
        control,
        true,
    )
}

fn composition_batch_plan(
    transient: &[u64],
    retained: u64,
    other: u64,
    threads: usize,
    budget: u64,
) -> Result<(usize, u64)> {
    let resident = add(other, retained)?;
    // Saturation only selects concurrency; the actual request is never clamped.
    let (size, peak) = batch_plan(transient, threads, budget.saturating_sub(resident), 0)?;
    Ok((size, add(resident, peak)?))
}

#[allow(clippy::too_many_arguments)]
fn composition_plan(
    paths: &[PathBuf],
    options: &SiftExtractionOptions,
    explicit: bool,
    threads: usize,
    budget: u64,
    other: u64,
    control: &SfmTaskControl,
    retain: bool,
) -> Result<Option<FeatureMemoryPlan>> {
    control.checkpoint()?;
    if !explicit || !standard(options) {
        return Ok(None);
    }
    options.check()?;
    let rows = mul(
        options.max_num_features as u64,
        options.max_num_orientations.max(1) as u64,
    )?;
    let mut retained = 0;
    let mut transient = Vec::with_capacity(paths.len());
    let mut headers = Vec::with_capacity(paths.len());
    for path in paths {
        let (w, h, encoded) = image_header(path, control)?;
        headers.push((w, h, encoded));
        let mut bytes = estimate(w, h, encoded, options)?;
        if retain {
            // 4096 B/planned row includes both descriptor families and keypoint
            // copies with allocation headroom; cardinality remains an estimate.
            let metadata = add(
                std::mem::size_of::<crate::types::ImageFrame>() as u64,
                mul(path.as_os_str().len() as u64, 2)?,
            )?;
            retained = add(retained, add(mul(rows, 4096)?, metadata)?)?;
            bytes = add(bytes, mul(mul(u64::from(w), u64::from(h))?, 8)?)?;
        }
        transient.push(bytes);
    }
    control.checkpoint()?;
    let (batch_size, request_bytes) =
        composition_batch_plan(&transient, retained, other, threads, budget)?;
    Ok(Some(FeatureMemoryPlan {
        paths: paths.to_vec(),
        options: options.clone(),
        batch_size,
        request_bytes,
        retained_bytes: retained,
        headers,
        retains_frames: retain,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn memory_review_validate_retained_inputs_without_replanning() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("input.bmp");
        image::GrayImage::new(8, 9).save(&path)?;
        let original = std::fs::read(&path)?;
        let control = SfmTaskControl::new();
        // Header validation is backend-independent; no native extractor or
        // standard-backend gate is needed for this immutable-plan fixture.
        let plan = FeatureMemoryPlan {
            paths: vec![path.clone()],
            options: SiftExtractionOptions::default(),
            batch_size: 2,
            request_bytes: 123,
            retained_bytes: 45,
            headers: vec![image_header(&path, &control)?],
            retains_frames: true,
        };
        let before = plan.clone();
        plan.validate_inputs(&control)?;
        let mut longer = original.clone();
        longer.push(0);
        std::fs::write(&path, longer)?;
        assert_eq!(image::image_dimensions(&path)?, (8, 9));
        assert!(plan.validate_inputs(&control).is_err());
        // BMP width/height swap preserves the encoded length: dimensions are
        // validated separately from the file-size check, without pixel decode.
        let mut resized_header = original.clone();
        resized_header[18..22].copy_from_slice(&9u32.to_le_bytes());
        resized_header[22..26].copy_from_slice(&8u32.to_le_bytes());
        std::fs::write(&path, resized_header)?;
        assert_eq!(std::fs::metadata(&path)?.len(), plan.headers[0].2);
        assert_eq!(image::image_dimensions(&path)?, (9, 8));
        assert!(plan.validate_inputs(&control).is_err());
        std::fs::write(&path, original)?;
        plan.validate_inputs(&control)?;
        std::fs::remove_file(&path)?;
        assert!(plan.validate_inputs(&control).is_err());
        for expected in [
            crate::task::SfmTaskStop::Paused,
            crate::task::SfmTaskStop::Cancelled,
        ] {
            let stopped = SfmTaskControl::new();
            match expected {
                crate::task::SfmTaskStop::Paused => stopped.request_pause(),
                crate::task::SfmTaskStop::Cancelled => stopped.request_cancel(),
            }
            // Typed stop precedes missing-file IO.
            assert_eq!(
                plan.validate_inputs(&stopped)
                    .unwrap_err()
                    .downcast_ref::<crate::task::SfmTaskStop>(),
                Some(&expected)
            );
        }
        assert_eq!(plan.paths, before.paths);
        assert!(same_options(&plan.options, &before.options));
        assert_eq!(plan.headers, before.headers);
        assert_eq!(plan.batch_size, before.batch_size);
        assert_eq!(plan.request_bytes, before.request_bytes);
        assert_eq!(plan.retained_bytes, before.retained_bytes);
        assert_eq!(plan.retains_frames, before.retains_frames);
        Ok(())
    }

    #[test]
    fn planned_selected_options_are_exact() {
        let options = SiftExtractionOptions::default();
        assert!(same_options(&options, &options.clone()));
        macro_rules! changed {
            ($field:ident, $value:expr) => {{
                let mut other = options.clone();
                other.$field = $value;
                assert!(!same_options(&options, &other), stringify!($field));
            }};
        }
        changed!(max_num_features, 1);
        changed!(first_octave, 0);
        changed!(num_octaves, 1);
        changed!(octave_resolution, 1);
        changed!(peak_threshold, 0.1);
        changed!(edge_threshold, 1.0);
        changed!(estimate_affine_shape, true);
        changed!(max_num_orientations, 1);
        changed!(upright, true);
        changed!(domain_size_pooling, true);
        changed!(dsp_min_scale, 0.1);
        changed!(dsp_max_scale, 1.0);
        changed!(dsp_num_scales, 1);
        changed!(normalization, crate::sift::SiftDescriptorNormalization::L2);
        changed!(force_covariant_extractor, true);
        changed!(use_gpu, true);
        changed!(max_image_size, 1);
    }

    #[test]
    fn planned_selected_rebind_preserves_estimates_and_checks_headers() -> Result<()> {
        let options = SiftExtractionOptions {
            max_num_features: 8,
            ..Default::default()
        };
        if !standard(&options) {
            return Ok(());
        }
        let dir = tempfile::tempdir()?;
        let original = dir.path().join("original.png");
        let staged = dir.path().join("staged.png");
        image::GrayImage::new(8, 9).save(&original)?;
        let control = SfmTaskControl::new();
        let plan =
            db_memory_plan(&[original.clone()], &options, true, 4, 0, 77, &control)?.unwrap();
        assert!(!staged.exists());
        std::fs::copy(&original, &staged)?;
        let rebound = plan.rebind_db_paths(&[staged.clone()], &options, &control)?;
        assert_eq!(plan.paths(), &[original.clone()]);
        assert_eq!(rebound.paths(), &[staged.clone()]);
        assert_eq!(rebound.headers, plan.headers);
        assert_eq!(rebound.request_bytes(), plan.request_bytes());
        assert_eq!(rebound.batch_size(), plan.batch_size());
        assert_eq!(rebound.retained_bytes(), plan.retained_bytes());
        assert!(plan.rebind_db_paths(&[], &options, &control).is_err());
        let changed = SiftExtractionOptions {
            upright: true,
            ..options.clone()
        };
        assert!(plan
            .rebind_db_paths(&[staged.clone()], &changed, &control)
            .is_err());
        let retained =
            retained_memory_plan(&[original], &options, true, 4, 0, 77, &control)?.unwrap();
        assert!(retained
            .rebind_db_paths(&[staged.clone()], &options, &control)
            .is_err());
        // Same dimensions but a changed encoded length must also fail.
        use std::io::Write;
        std::fs::OpenOptions::new()
            .append(true)
            .open(&staged)?
            .write_all(&[0])?;
        assert!(plan
            .rebind_db_paths(&[staged.clone()], &options, &control)
            .is_err());
        image::GrayImage::new(9, 8).save(&staged)?;
        assert!(plan
            .rebind_db_paths(&[staged.clone()], &options, &control)
            .is_err());
        control.request_cancel();
        assert!(plan
            .rebind_db_paths(&[staged], &options, &control)
            .unwrap_err()
            .downcast_ref::<crate::task::SfmTaskStop>()
            .is_some());
        Ok(())
    }

    #[test]
    fn composition_memory_additive_retained_chunks_and_overflow() -> Result<()> {
        assert_eq!(
            composition_batch_plan(&[10, 20, 30], 40, 10, 2, 110)?,
            (2, 80)
        );
        assert_eq!(
            composition_batch_plan(&[10, 20, 30], 40, 10, 2, 0)?,
            (1, 80)
        );
        assert_eq!(composition_batch_plan(&[], 0, 50, 4, 0)?.1, 50);
        assert!(composition_batch_plan(&[1], u64::MAX, 0, 1, 0).is_err());
        assert!(composition_batch_plan(&[], u64::MAX, 1, 1, 0).is_err());
        Ok(())
    }

    #[test]
    fn composition_memory_fallback_cache_subset_and_options() -> Result<()> {
        let control = SfmTaskControl::new();
        let missing = PathBuf::from("absent-cached-image.png");
        let options = SiftExtractionOptions::default();
        assert!(db_memory_plan(&[missing.clone()], &options, false, 4, 0, 77, &control)?.is_none());
        let unsupported = SiftExtractionOptions {
            use_gpu: true,
            ..options.clone()
        };
        assert!(
            retained_memory_plan(&[missing.clone()], &unsupported, true, 4, 0, 77, &control)?
                .is_none()
        );
        if standard(&options) {
            let dir = tempfile::tempdir()?;
            let first = dir.path().join("b.png");
            let second = dir.path().join("a.png");
            image::GrayImage::new(8, 9).save(&first)?;
            image::GrayImage::new(9, 8).save(&second)?;
            let paths = vec![first, second];
            let effective = SiftExtractionOptions {
                max_num_features: 7,
                ..options.clone()
            };
            let db = db_memory_plan(&paths, &effective, true, 4, 0, 77, &control)?.unwrap();
            let retained =
                retained_memory_plan(&paths, &effective, true, 4, 0, 77, &control)?.unwrap();
            assert_eq!(db.paths(), paths);
            assert_eq!(db.options().max_num_features, 7);
            assert_eq!(db.batch_size(), 1);
            assert_eq!(db.retained_bytes(), 0);
            assert!(retained.retained_bytes() > 0);
            assert!(retained.request_bytes() > db.request_bytes() + retained.retained_bytes());
            // Cache selection supplies no missing paths: no read of missing headers.
            assert_eq!(
                db_memory_plan(&[], &effective, true, 4, 0, 77, &control)?
                    .unwrap()
                    .request_bytes(),
                77
            );
            assert!(
                db_memory_plan(&[missing.clone()], &effective, true, 4, 0, 77, &control).is_err()
            );
        }
        control.request_cancel();
        assert!(
            db_memory_plan(&[missing], &options, false, 4, 0, 77, &control)
                .unwrap_err()
                .downcast_ref::<crate::task::SfmTaskStop>()
                .is_some()
        );
        Ok(())
    }

    #[test]
    fn standard_core_is_1056_mib() {
        assert_eq!(
            core(1536, 2048, &SiftExtractionOptions::default()).unwrap(),
            1056 * 1024 * 1024
        );
    }
    #[test]
    fn resize_rounding_and_overflow() {
        assert_eq!(dimensions(7, 13, 6).unwrap(), (3, 6));
        assert_eq!(dimensions(1, 10000, 1).unwrap(), (1, 1));
        assert!(dimensions(0, 1, 0).is_err());
        let mut options = SiftExtractionOptions::default();
        options.first_octave = i32::MIN;
        assert!(estimate(1, 1, 0, &options).is_err());
        options.first_octave = -1;
        assert!(estimate(32, 32, u64::MAX, &options).is_err());
    }
    #[test]
    fn budget_limits_aggregate_without_clamping() {
        let options = SiftExtractionOptions::default();
        let one = estimate(1536, 2048, 0, &options).unwrap();
        assert!(one > 1056 * 1024 * 1024);
        assert!(one < 2 * 1024 * 1024 * 1024);
        assert_eq!(
            batch_plan(&[one, one], 4, 2 * 1024 * 1024 * 1024, 512 * 1024 * 1024).unwrap(),
            (1, one)
        );
        assert_eq!(batch_plan(&[10, 20, 30], 2, 60, 1).unwrap(), (2, 30));
        assert_eq!(batch_plan(&[70], 4, 60, 1).unwrap(), (1, 70));
        assert_eq!(batch_plan(&[10], 4, 60, 80).unwrap(), (1, 80));
    }
    #[test]
    fn memory_header_checkpoint_precedes_io_and_estimate_is_a_floor() -> Result<()> {
        let control = SfmTaskControl::new();
        control.request_cancel();
        assert!(image_estimate(
            Path::new("does-not-exist.png"),
            &SiftExtractionOptions::default(),
            1,
            &control
        )
        .unwrap_err()
        .downcast_ref::<crate::task::SfmTaskStop>()
        .is_some());
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("tiny.png");
        image::GrayImage::new(3, 7).save(&path)?;
        let options = SiftExtractionOptions::default();
        if standard(&options) {
            let expected = estimate(3, 7, std::fs::metadata(&path)?.len(), &options)?;
            assert_eq!(
                image_estimate(&path, &options, 1, &SfmTaskControl::new())?,
                expected
            );
            assert_eq!(
                image_estimate(&path, &options, expected + 1, &SfmTaskControl::new())?,
                expected + 1
            );
        }
        let unsupported = SiftExtractionOptions {
            force_covariant_extractor: true,
            ..options
        };
        assert!(!standard(&unsupported));
        assert_eq!(
            image_estimate(
                Path::new("does-not-exist.png"),
                &unsupported,
                123,
                &SfmTaskControl::new()
            )?,
            123
        );
        Ok(())
    }

    #[test]
    fn max_features_is_not_a_scratch_cap() {
        let mut options = SiftExtractionOptions::default();
        options.max_num_features = 1;
        assert!(estimate(1536, 2048, 0, &options).unwrap() > 1056 * 1024 * 1024);
    }
}
