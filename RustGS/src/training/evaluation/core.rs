use crate::core::GaussianCamera;
use crate::{SplatMetadata, TrainingDataset};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

#[cfg(feature = "gpu")]
use crate::core::HostSplats;
#[cfg(feature = "gpu")]
use crate::training::engine::{host_splats_to_device, DeviceSplats, GsBackendBase};
#[cfg(feature = "gpu")]
use crate::training::forward;
#[cfg(feature = "gpu")]
use burn::prelude::Backend;

pub const MIN_RENDER_SCALE: f32 = 0.0625;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FinalTrainingMetrics {
    pub final_loss: f32,
    pub final_step_loss: f32,
}

pub fn summarize_training_metrics(
    loss_history: &[f32],
    frame_count: usize,
) -> FinalTrainingMetrics {
    FinalTrainingMetrics {
        final_loss: summarized_final_loss(loss_history, frame_count),
        final_step_loss: loss_history.last().copied().unwrap_or(0.0),
    }
}

fn summarized_final_loss(loss_history: &[f32], frame_count: usize) -> f32 {
    if loss_history.is_empty() {
        return 0.0;
    }

    let window = frame_count.max(1).min(loss_history.len());
    let start = loss_history.len() - window;
    loss_history[start..].iter().sum::<f32>() / window as f32
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationDevice {
    Gpu,
}

impl std::fmt::Display for EvaluationDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gpu => write!(f, "gpu"),
        }
    }
}

impl FromStr for EvaluationDevice {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "gpu" | "wgpu" | "metal" => Ok(Self::Gpu),
            other => Err(format!(
                "unsupported evaluation device '{other}'. Expected gpu"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SplatEvaluationConfig {
    pub render_scale: f32,
    pub raster_cov_blur: f32,
    pub frame_stride: usize,
    pub max_frames: usize,
    pub worst_frame_count: usize,
}

impl Default for SplatEvaluationConfig {
    fn default() -> Self {
        Self {
            render_scale: 0.5,
            raster_cov_blur: crate::training::DEFAULT_RASTER_COV_BLUR,
            frame_stride: 1,
            max_frames: 0,
            worst_frame_count: 5,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationFrameMetric {
    pub dataset_index: usize,
    pub frame_id: u64,
    pub psnr_db: f32,
    #[serde(default)]
    pub sharpness_grad_ratio: f32,
    #[serde(default)]
    pub sharpness_lap_ratio: f32,
    pub image_path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PsnrSummary {
    pub mean_db: f32,
    pub median_db: f32,
    pub min_db: f32,
    pub max_db: f32,
    pub stddev_db: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplatEvaluationSummary {
    pub device: EvaluationDevice,
    pub render_scale: f32,
    #[serde(default = "default_raster_cov_blur")]
    pub raster_cov_blur: f32,
    pub render_width: usize,
    pub render_height: usize,
    pub frame_stride: usize,
    pub max_frames: usize,
    pub frame_count: usize,
    #[serde(alias = "scene_iterations")]
    pub splat_iterations: usize,
    #[serde(alias = "scene_gaussian_count")]
    pub splat_count: usize,
    pub final_loss: f32,
    pub final_step_loss: Option<f32>,
    pub elapsed_seconds: f32,
    pub psnr_mean_db: f32,
    pub psnr_median_db: f32,
    pub psnr_min_db: f32,
    pub psnr_max_db: f32,
    pub psnr_std_db: f32,
    #[serde(default)]
    pub sharpness_grad_ratio_mean: f32,
    #[serde(default)]
    pub sharpness_lap_ratio_mean: f32,
    pub worst_frames: Vec<EvaluationFrameMetric>,
    #[serde(default)]
    pub crop_outputs: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SplatEvaluationResult {
    pub summary: SplatEvaluationSummary,
    pub frame_metrics: Vec<EvaluationFrameMetric>,
}

#[derive(Debug, thiserror::Error)]
pub enum SplatEvaluationError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("image error: {0}")]
    Image(#[from] image::ImageError),

    #[error("invalid evaluation input: {0}")]
    InvalidInput(String),
}

pub fn select_evaluation_frames(
    dataset: &TrainingDataset,
    max_frames: usize,
    frame_stride: usize,
) -> TrainingDataset {
    let mut selected =
        TrainingDataset::new(dataset.intrinsics).with_depth_scale(dataset.depth_scale);
    selected.initial_points = dataset.initial_points.clone();
    for pose in dataset
        .poses
        .iter()
        .take(if max_frames == 0 {
            dataset.poses.len()
        } else {
            max_frames
        })
        .step_by(frame_stride.max(1))
    {
        selected.add_pose(pose.clone());
    }
    selected
}

pub fn summarize_psnr_samples(values: &[f32]) -> PsnrSummary {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mean = values.iter().copied().sum::<f32>() / values.len().max(1) as f32;
    let median = if sorted.is_empty() {
        0.0
    } else if sorted.len().is_multiple_of(2) {
        let hi = sorted.len() / 2;
        (sorted[hi - 1] + sorted[hi]) * 0.5
    } else {
        sorted[sorted.len() / 2]
    };
    let min = sorted.first().copied().unwrap_or(0.0);
    let max = sorted.last().copied().unwrap_or(0.0);
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f32>()
        / values.len().max(1) as f32;

    PsnrSummary {
        mean_db: mean,
        median_db: median,
        min_db: min,
        max_db: max,
        stddev_db: variance.sqrt(),
    }
}

fn mean_f32(values: &[f32]) -> f32 {
    values.iter().copied().sum::<f32>() / values.len().max(1) as f32
}

fn default_raster_cov_blur() -> f32 {
    crate::training::DEFAULT_RASTER_COV_BLUR
}

pub fn worst_frame_metrics(
    frame_metrics: &[EvaluationFrameMetric],
    count: usize,
) -> Vec<EvaluationFrameMetric> {
    let mut sorted = frame_metrics.to_vec();
    sorted.sort_by(|a, b| {
        a.psnr_db
            .partial_cmp(&b.psnr_db)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sorted
        .into_iter()
        .take(count.min(frame_metrics.len()))
        .collect()
}

pub fn compute_psnr_f32(rendered: &[f32], target: &[f32]) -> f32 {
    if rendered.len() != target.len() || rendered.is_empty() {
        return 0.0;
    }

    let mse = rendered
        .iter()
        .zip(target.iter())
        .map(|(r, t)| {
            let diff = r - t;
            diff * diff
        })
        .sum::<f32>()
        / rendered.len() as f32;

    if mse <= 1e-12 {
        100.0
    } else {
        10.0 * (1.0 / mse).log10()
    }
}

pub fn compute_gradient_sharpness_f32(rgb: &[f32], width: usize, height: usize) -> f32 {
    if rgb.len() != width.saturating_mul(height).saturating_mul(3) || width < 2 || height < 2 {
        return 0.0;
    }

    let mut sum = 0.0f32;
    let mut count = 0usize;
    for y in 0..height {
        for x in 0..width {
            let luminance = luminance_at(rgb, width, x, y);
            if x + 1 < width {
                sum += (luminance_at(rgb, width, x + 1, y) - luminance).abs();
                count += 1;
            }
            if y + 1 < height {
                sum += (luminance_at(rgb, width, x, y + 1) - luminance).abs();
                count += 1;
            }
        }
    }
    sum / count.max(1) as f32
}

pub fn compute_laplacian_sharpness_f32(rgb: &[f32], width: usize, height: usize) -> f32 {
    if rgb.len() != width.saturating_mul(height).saturating_mul(3) || width < 3 || height < 3 {
        return 0.0;
    }

    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    let mut count = 0usize;
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let center = luminance_at(rgb, width, x, y);
            let lap = -4.0 * center
                + luminance_at(rgb, width, x - 1, y)
                + luminance_at(rgb, width, x + 1, y)
                + luminance_at(rgb, width, x, y - 1)
                + luminance_at(rgb, width, x, y + 1);
            sum += lap;
            sum_sq += lap * lap;
            count += 1;
        }
    }
    let mean = sum / count.max(1) as f32;
    (sum_sq / count.max(1) as f32 - mean * mean).max(0.0)
}

fn luminance_at(rgb: &[f32], width: usize, x: usize, y: usize) -> f32 {
    let base = (y * width + x) * 3;
    0.2126 * rgb[base] + 0.7152 * rgb[base + 1] + 0.0722 * rgb[base + 2]
}

fn sharpness_ratio(rendered: f32, target: f32) -> f32 {
    if target <= 1e-12 {
        0.0
    } else {
        rendered / target
    }
}

pub fn scaled_dimensions(width: usize, height: usize, render_scale: f32) -> (usize, usize) {
    let scale = render_scale.clamp(MIN_RENDER_SCALE, 1.0);
    let scaled_width = ((width as f32) * scale).round().max(1.0) as usize;
    let scaled_height = ((height as f32) * scale).round().max(1.0) as usize;
    (scaled_width, scaled_height)
}

#[cfg(feature = "gpu")]
pub fn evaluation_device(
    device: EvaluationDevice,
) -> Result<EvaluationDevice, SplatEvaluationError> {
    Ok(device)
}

#[cfg(feature = "gpu")]
pub fn runtime_from_splats(
    splats: &HostSplats,
    _device: &EvaluationDevice,
) -> Result<HostSplats, SplatEvaluationError> {
    Ok(splats.clone())
}

#[cfg(feature = "gpu")]
pub struct SplatEvaluationRenderer {
    backend: SplatEvaluationRendererBackend,
}

#[cfg(feature = "gpu")]
enum SplatEvaluationRendererBackend {
    Gpu(GpuSplatEvaluationRenderer),
}

#[cfg(feature = "gpu")]
#[derive(Clone, Debug)]
pub struct SharedWgpuContext {
    device: <GsBackendBase as Backend>::Device,
}

#[cfg(feature = "gpu")]
impl SharedWgpuContext {
    pub fn from_wgpu_setup(setup: burn_wgpu::WgpuSetup) -> Self {
        let device = burn_wgpu::init_device(setup, shared_wgpu_runtime_options());
        Self { device }
    }

    pub fn from_wgpu_parts(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        backend: wgpu::Backend,
    ) -> Self {
        Self::from_wgpu_setup(burn_wgpu::WgpuSetup {
            instance,
            adapter,
            device,
            queue,
            backend,
        })
    }

    fn device(&self) -> <GsBackendBase as Backend>::Device {
        self.device.clone()
    }
}

#[cfg(feature = "gpu")]
fn shared_wgpu_runtime_options() -> burn_wgpu::RuntimeOptions {
    burn_wgpu::RuntimeOptions {
        memory_config: burn_wgpu::MemoryConfiguration::ExclusivePages,
        ..burn_wgpu::RuntimeOptions::default()
    }
}

#[cfg(feature = "gpu")]
struct GpuSplatEvaluationRenderer {
    render_width: usize,
    render_height: usize,
    raster_cov_blur: f32,
    device: <GsBackendBase as Backend>::Device,
    runtime: tokio::runtime::Runtime,
    cached_splats: Option<CachedGpuSplats>,
}

#[cfg(feature = "gpu")]
struct CachedGpuSplats {
    key: SplatCacheKey,
    splats: DeviceSplats<GsBackendBase>,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SplatCacheKey {
    len: usize,
    sh_degree: usize,
    positions: usize,
    log_scales: usize,
    rotations: usize,
    opacity_logits: usize,
    sh_coeffs: usize,
}

#[cfg(feature = "gpu")]
impl SplatEvaluationRenderer {
    pub fn new(
        render_width: usize,
        render_height: usize,
        device: EvaluationDevice,
        raster_cov_blur: f32,
    ) -> Result<Self, SplatEvaluationError> {
        let backend = match device {
            EvaluationDevice::Gpu => SplatEvaluationRendererBackend::Gpu(
                GpuSplatEvaluationRenderer::new(render_width, render_height, raster_cov_blur)?,
            ),
        };
        Ok(Self { backend })
    }

    pub fn new_with_wgpu_context(
        render_width: usize,
        render_height: usize,
        context: SharedWgpuContext,
        raster_cov_blur: f32,
    ) -> Result<Self, SplatEvaluationError> {
        Ok(Self {
            backend: SplatEvaluationRendererBackend::Gpu(
                GpuSplatEvaluationRenderer::new_with_device(
                    render_width,
                    render_height,
                    raster_cov_blur,
                    context.device(),
                )?,
            ),
        })
    }

    pub fn render(
        &mut self,
        splats: &HostSplats,
        camera: &GaussianCamera,
    ) -> Result<Vec<f32>, SplatEvaluationError> {
        match &mut self.backend {
            SplatEvaluationRendererBackend::Gpu(renderer) => renderer.render(splats, camera),
        }
    }
}

#[cfg(feature = "gpu")]
impl GpuSplatEvaluationRenderer {
    fn new(
        render_width: usize,
        render_height: usize,
        raster_cov_blur: f32,
    ) -> Result<Self, SplatEvaluationError> {
        Self::new_with_device(
            render_width,
            render_height,
            raster_cov_blur,
            <GsBackendBase as Backend>::Device::default(),
        )
    }

    fn new_with_device(
        render_width: usize,
        render_height: usize,
        raster_cov_blur: f32,
        device: <GsBackendBase as Backend>::Device,
    ) -> Result<Self, SplatEvaluationError> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|err| {
                SplatEvaluationError::InvalidInput(format!(
                    "failed to initialize gpu renderer runtime: {err}"
                ))
            })?;

        Ok(Self {
            render_width,
            render_height,
            raster_cov_blur,
            device,
            runtime,
            cached_splats: None,
        })
    }

    fn render(
        &mut self,
        splats: &HostSplats,
        camera: &GaussianCamera,
    ) -> Result<Vec<f32>, SplatEvaluationError> {
        self.ensure_device_splats(splats);
        let cached = self.cached_splats.as_ref().ok_or_else(|| {
            SplatEvaluationError::InvalidInput("gpu splat cache was not initialized".to_string())
        })?;
        let img_size = (
            u32::try_from(self.render_width).map_err(|_| {
                SplatEvaluationError::InvalidInput(format!(
                    "render width {} exceeds u32",
                    self.render_width
                ))
            })?,
            u32::try_from(self.render_height).map_err(|_| {
                SplatEvaluationError::InvalidInput(format!(
                    "render height {} exceeds u32",
                    self.render_height
                ))
            })?,
        );

        self.runtime.block_on(render_gpu_rgb_f32(
            &cached.splats,
            camera,
            img_size,
            &self.device,
            self.raster_cov_blur,
        ))
    }

    fn ensure_device_splats(&mut self, splats: &HostSplats) {
        let key = SplatCacheKey::from_splats(splats);
        if self
            .cached_splats
            .as_ref()
            .map(|cached| cached.key == key)
            .unwrap_or(false)
        {
            return;
        }

        self.cached_splats = Some(CachedGpuSplats {
            key,
            splats: host_splats_to_device::<GsBackendBase>(splats, &self.device),
        });
    }
}

#[cfg(feature = "gpu")]
impl SplatCacheKey {
    fn from_splats(splats: &HostSplats) -> Self {
        let view = splats.as_view();
        Self {
            len: splats.len(),
            sh_degree: splats.sh_degree(),
            positions: view.positions.as_ptr() as usize,
            log_scales: view.log_scales.as_ptr() as usize,
            rotations: view.rotations.as_ptr() as usize,
            opacity_logits: view.opacity_logits.as_ptr() as usize,
            sh_coeffs: view.sh_coeffs.as_ptr() as usize,
        }
    }
}

#[cfg(feature = "gpu")]
async fn render_gpu_rgb_f32(
    splats: &DeviceSplats<GsBackendBase>,
    camera: &GaussianCamera,
    img_size: (u32, u32),
    device: &<GsBackendBase as Backend>::Device,
    raster_cov_blur: f32,
) -> Result<Vec<f32>, SplatEvaluationError> {
    let rendered = forward::render_forward::<GsBackendBase>(
        splats,
        camera,
        img_size,
        [0.0, 0.0, 0.0],
        device,
        raster_cov_blur,
    )
    .await;
    let rgba = rendered
        .out_img
        .into_data_async()
        .await
        .map_err(|err| {
            SplatEvaluationError::InvalidInput(format!("gpu renderer readback failed: {err}"))
        })?
        .into_vec::<f32>()
        .map_err(|err| {
            SplatEvaluationError::InvalidInput(format!("gpu renderer output was not f32: {err}"))
        })?;

    let expected = img_size.0 as usize * img_size.1 as usize * 4;
    if rgba.len() != expected {
        return Err(SplatEvaluationError::InvalidInput(format!(
            "gpu renderer output length {} does not match expected {}",
            rgba.len(),
            expected
        )));
    }

    let mut rgb = Vec::with_capacity(img_size.0 as usize * img_size.1 as usize * 3);
    for px in rgba.chunks_exact(4) {
        rgb.extend_from_slice(&px[..3]);
    }

    Ok(rgb)
}

#[cfg(feature = "gpu")]
pub fn render_evaluation_frame(
    dataset: &TrainingDataset,
    pose: &rustscan_types::ScenePose,
    render_width: usize,
    render_height: usize,
    _device: &EvaluationDevice,
    trainable: &HostSplats,
    renderer: &mut SplatEvaluationRenderer,
) -> Result<(Vec<f32>, Vec<f32>), SplatEvaluationError> {
    let target = load_resized_target(
        &pose.image_path,
        dataset.intrinsics.width as usize,
        dataset.intrinsics.height as usize,
        render_width,
        render_height,
    )?;
    let camera = scaled_camera_for_pose(pose.pose, dataset.intrinsics, render_width, render_height);
    let rendered = renderer.render(trainable, &camera)?;
    Ok((target, rendered))
}

#[cfg(feature = "gpu")]
pub fn evaluate_splats(
    dataset: &TrainingDataset,
    splats: &HostSplats,
    metadata: &SplatMetadata,
    config: &SplatEvaluationConfig,
    device: &EvaluationDevice,
    training_metrics: Option<FinalTrainingMetrics>,
) -> Result<SplatEvaluationResult, SplatEvaluationError> {
    let dataset = select_evaluation_frames(dataset, config.max_frames, config.frame_stride);
    if dataset.poses.is_empty() {
        return Err(SplatEvaluationError::InvalidInput(
            "evaluation dataset resolved to zero frames".to_string(),
        ));
    }

    let (render_width, render_height) = scaled_dimensions(
        dataset.intrinsics.width as usize,
        dataset.intrinsics.height as usize,
        config.render_scale,
    );
    let runtime_splats = runtime_from_splats(splats, device)?;
    let mut renderer =
        SplatEvaluationRenderer::new(render_width, render_height, *device, config.raster_cov_blur)?;

    let start = Instant::now();
    let mut psnrs = Vec::with_capacity(dataset.poses.len());
    let mut sharpness_grad_ratios = Vec::with_capacity(dataset.poses.len());
    let mut sharpness_lap_ratios = Vec::with_capacity(dataset.poses.len());
    let mut frame_metrics = Vec::with_capacity(dataset.poses.len());

    for (idx, pose) in dataset.poses.iter().enumerate() {
        let (target, rendered) = render_evaluation_frame(
            &dataset,
            pose,
            render_width,
            render_height,
            device,
            &runtime_splats,
            &mut renderer,
        )?;
        let psnr_db = compute_psnr_f32(&rendered, &target);
        let render_grad = compute_gradient_sharpness_f32(&rendered, render_width, render_height);
        let target_grad = compute_gradient_sharpness_f32(&target, render_width, render_height);
        let render_lap = compute_laplacian_sharpness_f32(&rendered, render_width, render_height);
        let target_lap = compute_laplacian_sharpness_f32(&target, render_width, render_height);
        let sharpness_grad_ratio = sharpness_ratio(render_grad, target_grad);
        let sharpness_lap_ratio = sharpness_ratio(render_lap, target_lap);
        psnrs.push(psnr_db);
        sharpness_grad_ratios.push(sharpness_grad_ratio);
        sharpness_lap_ratios.push(sharpness_lap_ratio);
        frame_metrics.push(EvaluationFrameMetric {
            dataset_index: idx,
            frame_id: pose.frame_id,
            psnr_db,
            sharpness_grad_ratio,
            sharpness_lap_ratio,
            image_path: pose.image_path.clone(),
        });
    }

    let psnr = summarize_psnr_samples(&psnrs);
    let final_loss = training_metrics
        .map(|metrics| metrics.final_loss)
        .unwrap_or(metadata.final_loss);
    let final_step_loss = training_metrics.map(|metrics| metrics.final_step_loss);
    let summary = SplatEvaluationSummary {
        device: *device,
        render_scale: config.render_scale,
        raster_cov_blur: config.raster_cov_blur,
        render_width,
        render_height,
        frame_stride: config.frame_stride,
        max_frames: config.max_frames,
        frame_count: dataset.poses.len(),
        splat_iterations: metadata.iterations,
        splat_count: metadata.gaussian_count,
        final_loss,
        final_step_loss,
        elapsed_seconds: start.elapsed().as_secs_f32(),
        psnr_mean_db: psnr.mean_db,
        psnr_median_db: psnr.median_db,
        psnr_min_db: psnr.min_db,
        psnr_max_db: psnr.max_db,
        psnr_std_db: psnr.stddev_db,
        sharpness_grad_ratio_mean: mean_f32(&sharpness_grad_ratios),
        sharpness_lap_ratio_mean: mean_f32(&sharpness_lap_ratios),
        worst_frames: worst_frame_metrics(&frame_metrics, config.worst_frame_count),
        crop_outputs: Vec::new(),
    };

    Ok(SplatEvaluationResult {
        summary,
        frame_metrics,
    })
}

#[cfg(feature = "gpu")]
fn scaled_camera_for_pose(
    pose_c2w: rustscan_types::SE3,
    intrinsics: rustscan_types::Intrinsics,
    dst_width: usize,
    dst_height: usize,
) -> GaussianCamera {
    let view_pose = pose_c2w.inverse();
    let sx = dst_width as f32 / intrinsics.width as f32;
    let sy = dst_height as f32 / intrinsics.height as f32;
    GaussianCamera::new(
        rustscan_types::Intrinsics::new(
            intrinsics.fx * sx,
            intrinsics.fy * sy,
            intrinsics.cx * sx,
            intrinsics.cy * sy,
            dst_width as u32,
            dst_height as u32,
        ),
        view_pose,
    )
}

fn load_resized_target(
    path: &Path,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Result<Vec<f32>, SplatEvaluationError> {
    let image = image::ImageReader::open(path)?
        .with_guessed_format()?
        .decode()?;
    let rgb = image.to_rgb8();
    let (actual_width, actual_height) = rgb.dimensions();
    if actual_width as usize != src_width || actual_height as usize != src_height {
        return Err(SplatEvaluationError::InvalidInput(format!(
            "image {} has size {}x{}, expected {}x{}",
            path.display(),
            actual_width,
            actual_height,
            src_width,
            src_height
        )));
    }
    let src: Vec<f32> = rgb
        .into_raw()
        .into_iter()
        .map(|value| value as f32 / 255.0)
        .collect();
    Ok(resize_rgb_box(
        &src, src_width, src_height, dst_width, dst_height,
    ))
}

fn resize_rgb_box(
    src: &[f32],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Vec<f32> {
    let mut dst = vec![0.0f32; dst_width * dst_height * 3];
    for dy in 0..dst_height {
        let sy0 = dy * src_height / dst_height;
        let sy1 = ((dy + 1) * src_height / dst_height)
            .max(sy0 + 1)
            .min(src_height);
        for dx in 0..dst_width {
            let sx0 = dx * src_width / dst_width;
            let sx1 = ((dx + 1) * src_width / dst_width)
                .max(sx0 + 1)
                .min(src_width);
            let mut acc = [0.0f32; 3];
            let mut count = 0usize;
            for sy in sy0..sy1 {
                for sx in sx0..sx1 {
                    let src_idx = (sy * src_width + sx) * 3;
                    acc[0] += src[src_idx];
                    acc[1] += src[src_idx + 1];
                    acc[2] += src[src_idx + 2];
                    count += 1;
                }
            }
            let dst_idx = (dy * dst_width + dx) * 3;
            let inv = 1.0 / count.max(1) as f32;
            dst[dst_idx] = acc[0] * inv;
            dst[dst_idx + 1] = acc[1] * inv;
            dst[dst_idx + 2] = acc[2] * inv;
        }
    }
    dst
}

#[cfg(test)]
mod tests;
