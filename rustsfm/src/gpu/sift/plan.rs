use crate::sift::SiftExtractionOptions;
use anyhow::{bail, Context, Result};

const MIN_OCTAVE_SIZE: u32 = 32;
const MAX_LEVEL_BYTES: u64 = 512 * 1024 * 1024;
const MIN_CANDIDATE_CAPACITY: usize = 1024;
const CANDIDATES_PER_FEATURE: usize = 8;

#[derive(Debug, Clone, Copy)]
pub(crate) struct OctavePlan {
    pub(crate) octave: i32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) gaussian_levels: usize,
    pub(crate) dog_levels: usize,
}

impl OctavePlan {
    pub(crate) fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub(crate) fn pixel_count(&self) -> Result<usize> {
        let pixels = u64::from(self.width)
            .checked_mul(u64::from(self.height))
            .context("GPU SIFT octave pixel count overflow")?;
        usize::try_from(pixels).context("GPU SIFT octave pixel count does not fit usize")
    }
}

#[derive(Debug)]
pub(crate) struct SiftPlan {
    pub(crate) first_octave: i32,
    pub(crate) sigma_step: f32,
    pub(crate) octaves: Vec<OctavePlan>,
    pub(crate) candidate_capacity: u32,
}

impl SiftPlan {
    pub(crate) fn new(width: u32, height: u32, options: &SiftExtractionOptions) -> Result<Self> {
        options.check()?;
        if width == 0 || height == 0 {
            bail!("GPU SIFT image dimensions must be non-zero");
        }
        if options.first_octave < -1 {
            bail!("wgpu SIFT first_octave must be >= -1");
        }

        let (mut octave_width, mut octave_height) =
            scaled_first_octave_dimensions(width, height, options.first_octave)?;
        let gaussian_levels = options
            .octave_resolution
            .checked_add(3)
            .context("GPU SIFT Gaussian level count overflow")?;
        let dog_levels = options
            .octave_resolution
            .checked_add(2)
            .context("GPU SIFT DoG level count overflow")?;
        let max_octaves = if options.num_octaves == 0 {
            usize::BITS as usize
        } else {
            options.num_octaves
        };

        let mut octaves = Vec::with_capacity(max_octaves.min(16));
        for ordinal in 0..max_octaves {
            if octave_width.min(octave_height) < MIN_OCTAVE_SIZE {
                break;
            }
            validate_level_size(octave_width, octave_height)?;
            let octave = options
                .first_octave
                .checked_add(i32::try_from(ordinal).context("GPU SIFT octave index overflow")?)
                .context("GPU SIFT octave index overflow")?;
            octaves.push(OctavePlan {
                octave,
                width: octave_width,
                height: octave_height,
                gaussian_levels,
                dog_levels,
            });
            octave_width /= 2;
            octave_height /= 2;
        }

        let candidate_capacity = options
            .max_num_features
            .checked_mul(CANDIDATES_PER_FEATURE)
            .unwrap_or(usize::MAX)
            .max(MIN_CANDIDATE_CAPACITY);
        let candidate_capacity =
            u32::try_from(candidate_capacity).context("GPU SIFT candidate capacity exceeds u32")?;

        Ok(Self {
            first_octave: options.first_octave,
            sigma_step: 2.0f32.powf(1.0 / options.octave_resolution as f32),
            octaves,
            candidate_capacity,
        })
    }
}

fn scaled_first_octave_dimensions(
    width: u32,
    height: u32,
    first_octave: i32,
) -> Result<(u32, u32)> {
    if first_octave == -1 {
        return Ok((
            width
                .checked_mul(2)
                .context("GPU SIFT doubled image width overflow")?,
            height
                .checked_mul(2)
                .context("GPU SIFT doubled image height overflow")?,
        ));
    }
    let shift = u32::try_from(first_octave).context("GPU SIFT octave shift is negative")?;
    Ok(((width >> shift).max(1), (height >> shift).max(1)))
}

fn validate_level_size(width: u32, height: u32) -> Result<()> {
    let bytes = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(std::mem::size_of::<f32>() as u64))
        .context("GPU SIFT level byte count overflow")?;
    if bytes > MAX_LEVEL_BYTES {
        bail!(
            "GPU SIFT level {}x{} requires {} bytes, exceeding the {} byte limit",
            width,
            height,
            bytes,
            MAX_LEVEL_BYTES
        );
    }
    Ok(())
}
