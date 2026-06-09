#![allow(clippy::too_many_arguments)]

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use image::{DynamicImage, GenericImageView, ImageReader};

use super::frame_targets::resize_rgb_u8_to_f32;
use crate::TrainingConfig;
use crate::{TrainingDataset, TrainingError};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DecodedFrame {
    pub(crate) color_u8: Vec<u8>,
    pub(crate) target_rgb: Option<Arc<Vec<f32>>>,
    pub(crate) depth: Vec<f32>,
    pub(crate) used_real_depth: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FrameLoaderOptions {
    pub(crate) cache_capacity: usize,
    pub(crate) prefetch_ahead: usize,
    pub(crate) rgb_target_size: Option<(usize, usize)>,
}

impl Default for FrameLoaderOptions {
    fn default() -> Self {
        Self {
            cache_capacity: 8,
            prefetch_ahead: 4,
            rgb_target_size: None,
        }
    }
}

#[derive(Debug, Clone)]
struct FrameLoadSpec {
    image_path: PathBuf,
    depth_path: Option<PathBuf>,
}

pub(crate) struct PrefetchFrameLoader {
    specs: Arc<Vec<FrameLoadSpec>>,
    options: FrameLoaderOptions,
    cache: HashMap<usize, Arc<DecodedFrame>>,
    lru: VecDeque<usize>,
    pending: HashSet<usize>,
    request_tx: Option<Sender<usize>>,
    result_rx: Receiver<(usize, Result<DecodedFrame, TrainingError>)>,
    workers: Vec<JoinHandle<()>>,
}

#[derive(Debug, Clone)]
pub(super) struct DeterministicFrameBatchIter {
    order: Vec<usize>,
    batch_size: usize,
    cursor: usize,
}

impl DeterministicFrameBatchIter {
    pub(super) fn new(frame_count: usize, batch_size: usize, seed: u64) -> Self {
        let mut order = (0..frame_count).collect::<Vec<_>>();
        if seed != 0 {
            deterministic_shuffle(&mut order, seed);
        }
        Self {
            order,
            batch_size: batch_size.max(1),
            cursor: 0,
        }
    }
}

pub(crate) fn ordered_frame_indices(
    frame_count: usize,
    batch_size: usize,
    seed: u64,
) -> Vec<usize> {
    DeterministicFrameBatchIter::new(frame_count, batch_size, seed)
        .flat_map(|batch| batch.into_iter())
        .collect()
}

impl Iterator for DeterministicFrameBatchIter {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.order.len() {
            return None;
        }

        let end = (self.cursor + self.batch_size).min(self.order.len());
        let batch = self.order[self.cursor..end].to_vec();
        self.cursor = end;
        Some(batch)
    }
}

impl PrefetchFrameLoader {
    pub(crate) fn new(
        dataset: &TrainingDataset,
        config: &TrainingConfig,
        options: FrameLoaderOptions,
    ) -> Result<Self, TrainingError> {
        if dataset.poses.is_empty() {
            return Err(TrainingError::InvalidInput(
                "training dataset does not contain any poses".to_string(),
            ));
        }

        let specs = Arc::new(
            dataset
                .poses
                .iter()
                .map(|pose| FrameLoadSpec {
                    image_path: pose.image_path.clone(),
                    depth_path: pose.depth_path.clone(),
                })
                .collect::<Vec<_>>(),
        );
        let width = dataset.intrinsics.width as usize;
        let height = dataset.intrinsics.height as usize;
        let depth_scale = dataset.depth_scale;
        let use_synthetic_depth = config.initialization.use_synthetic_depth;
        let min_depth = config.initialization.min_depth;
        let max_depth = config.initialization.max_depth;
        let rgb_target_size = options.rgb_target_size;
        let (request_tx, request_rx) = mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        let request_rx = Arc::new(Mutex::new(request_rx));
        let worker_count = frame_loader_worker_count(specs.len(), options);
        let mut workers = Vec::with_capacity(worker_count);

        for worker_idx in 0..worker_count {
            let worker_specs = Arc::clone(&specs);
            let worker_request_rx = Arc::clone(&request_rx);
            let worker_result_tx = result_tx.clone();
            let worker = thread::Builder::new()
                .name(format!("rustgs-frame-prefetch-{worker_idx}"))
                .spawn(move || loop {
                    let frame_idx = {
                        let request_rx = worker_request_rx
                            .lock()
                            .expect("frame prefetch receiver lock poisoned");
                        match request_rx.recv() {
                            Ok(frame_idx) => frame_idx,
                            Err(_) => break,
                        }
                    };

                    let result = if let Some(spec) = worker_specs.get(frame_idx) {
                        let spec: &FrameLoadSpec = spec;
                        decode_frame(
                            &spec.image_path,
                            spec.depth_path.as_deref(),
                            width,
                            height,
                            rgb_target_size,
                            depth_scale,
                            use_synthetic_depth,
                            min_depth,
                            max_depth,
                        )
                    } else {
                        Err(TrainingError::InvalidInput(format!(
                            "frame index {frame_idx} is outside the dataset bounds"
                        )))
                    };
                    if worker_result_tx.send((frame_idx, result)).is_err() {
                        break;
                    }
                })
                .map_err(|err| {
                    TrainingError::TrainingFailed(format!(
                        "failed to start frame prefetch worker {worker_idx}: {err}"
                    ))
                })?;
            workers.push(worker);
        }

        Ok(Self {
            specs,
            options,
            cache: HashMap::new(),
            lru: VecDeque::new(),
            pending: HashSet::new(),
            request_tx: Some(request_tx),
            result_rx,
            workers,
        })
    }

    pub(crate) fn prefetch_order_window(
        &mut self,
        order: &[usize],
        cursor: usize,
    ) -> Result<(), TrainingError> {
        let end = (cursor + self.options.prefetch_ahead).min(order.len());
        for &frame_idx in &order[cursor..end] {
            self.queue_frame(frame_idx)?;
        }
        self.drain_ready()?;
        Ok(())
    }

    pub(crate) fn get(&mut self, frame_idx: usize) -> Result<Arc<DecodedFrame>, TrainingError> {
        self.drain_ready()?;
        if let Some(frame) = self.cache.get(&frame_idx).cloned() {
            self.touch(frame_idx);
            return Ok(frame);
        }

        self.queue_frame(frame_idx)?;
        loop {
            let (ready_idx, result) = self.result_rx.recv().map_err(|err| {
                TrainingError::TrainingFailed(format!(
                    "frame prefetch worker stopped before delivering frame {frame_idx}: {err}"
                ))
            })?;
            self.pending.remove(&ready_idx);
            match result {
                Ok(decoded) => {
                    let cached = Arc::new(decoded);
                    let requested = ready_idx == frame_idx;
                    self.insert(ready_idx, Arc::clone(&cached));
                    if requested {
                        return Ok(cached);
                    }
                }
                Err(err) if ready_idx == frame_idx => return Err(err),
                Err(err) => {
                    log::warn!("Dropping failed prefetched frame {}: {}", ready_idx, err);
                }
            }
        }
    }

    fn queue_frame(&mut self, frame_idx: usize) -> Result<(), TrainingError> {
        if frame_idx >= self.specs.len() {
            return Err(TrainingError::InvalidInput(format!(
                "frame index {frame_idx} is outside the dataset bounds"
            )));
        }
        if self.cache.contains_key(&frame_idx) || self.pending.contains(&frame_idx) {
            return Ok(());
        }
        self.request_tx
            .as_ref()
            .ok_or_else(|| {
                TrainingError::TrainingFailed(
                    "frame prefetch worker is no longer available".to_string(),
                )
            })?
            .send(frame_idx)
            .map_err(|err| {
                TrainingError::TrainingFailed(format!(
                    "failed to queue frame {frame_idx} for prefetch: {err}"
                ))
            })?;
        self.pending.insert(frame_idx);
        Ok(())
    }

    fn drain_ready(&mut self) -> Result<(), TrainingError> {
        loop {
            match self.result_rx.try_recv() {
                Ok((ready_idx, result)) => {
                    self.pending.remove(&ready_idx);
                    match result {
                        Ok(decoded) => {
                            self.insert(ready_idx, Arc::new(decoded));
                        }
                        Err(err) => {
                            return Err(TrainingError::TrainingFailed(format!(
                                "prefetch failed for frame {}: {}",
                                ready_idx, err
                            )));
                        }
                    }
                }
                Err(TryRecvError::Empty) => return Ok(()),
                Err(TryRecvError::Disconnected) => {
                    return Err(TrainingError::TrainingFailed(
                        "frame prefetch worker disconnected unexpectedly".to_string(),
                    ));
                }
            }
        }
    }

    fn insert(&mut self, frame_idx: usize, frame: Arc<DecodedFrame>) {
        self.cache.insert(frame_idx, frame);
        self.touch(frame_idx);
        let capacity = self.options.cache_capacity.max(1);
        while self.cache.len() > capacity {
            if let Some(evicted) = self.lru.pop_front() {
                self.cache.remove(&evicted);
            }
        }
    }

    fn touch(&mut self, frame_idx: usize) {
        if let Some(position) = self.lru.iter().position(|cached| *cached == frame_idx) {
            self.lru.remove(position);
        }
        self.lru.push_back(frame_idx);
    }
}

impl Drop for PrefetchFrameLoader {
    fn drop(&mut self) {
        self.request_tx.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn frame_loader_worker_count(frame_count: usize, options: FrameLoaderOptions) -> usize {
    let parallelism = thread::available_parallelism().map_or(4, |count| count.get());
    frame_count
        .max(1)
        .min(options.prefetch_ahead.max(1))
        .min(parallelism.max(1))
}

pub(super) fn decode_frame(
    image_path: &Path,
    depth_path: Option<&Path>,
    width: usize,
    height: usize,
    rgb_target_size: Option<(usize, usize)>,
    depth_scale: f32,
    use_synthetic_depth: bool,
    min_depth: f32,
    max_depth: f32,
) -> Result<DecodedFrame, TrainingError> {
    let expected_color = width.saturating_mul(height).saturating_mul(3);
    let expected_depth = width.saturating_mul(height);
    let color_u8 = load_color_image(image_path, width, height)?;
    if color_u8.len() != expected_color {
        return Err(TrainingError::InvalidInput(format!(
            "image {} produced {} bytes, expected {}",
            image_path.display(),
            color_u8.len(),
            expected_color,
        )));
    }
    let target_rgb = rgb_target_size.map(|(target_width, target_height)| {
        Arc::new(resize_rgb_u8_to_f32(
            &color_u8,
            width,
            height,
            target_width,
            target_height,
        ))
    });

    let (depth, used_real_depth) = match depth_path {
        Some(path) => (load_depth_image(path, width, height, depth_scale)?, true),
        None if use_synthetic_depth => (
            synthetic_depth(&color_u8, width, height, min_depth, max_depth),
            false,
        ),
        None => (vec![0.0; expected_depth], false),
    };
    if depth.len() != expected_depth {
        return Err(TrainingError::InvalidInput(format!(
            "decoded depth for {} produced {} values, expected {}",
            image_path.display(),
            depth.len(),
            expected_depth,
        )));
    }

    Ok(DecodedFrame {
        color_u8,
        target_rgb,
        depth,
        used_real_depth,
    })
}

pub(super) fn load_color_image(
    path: &Path,
    width: usize,
    height: usize,
) -> Result<Vec<u8>, TrainingError> {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("rgb") => {
            let bytes = fs::read(path)?;
            let expected = width.saturating_mul(height).saturating_mul(3);
            if bytes.len() != expected {
                return Err(TrainingError::InvalidInput(format!(
                    "raw RGB frame {} has {} bytes, expected {}",
                    path.display(),
                    bytes.len(),
                    expected,
                )));
            }
            Ok(bytes)
        }
        _ => {
            let image = ImageReader::open(path)
                .map_err(TrainingError::Io)?
                .with_guessed_format()
                .map_err(|err| TrainingError::InvalidInput(err.to_string()))?
                .decode()
                .map_err(|err| TrainingError::InvalidInput(err.to_string()))?;

            let (actual_width, actual_height) = image.dimensions();
            if actual_width as usize != width || actual_height as usize != height {
                return Err(TrainingError::InvalidInput(format!(
                    "image {} has size {}x{}, expected {}x{}",
                    path.display(),
                    actual_width,
                    actual_height,
                    width,
                    height,
                )));
            }

            Ok(image.to_rgb8().into_raw())
        }
    }
}

pub(super) fn load_depth_image(
    path: &Path,
    width: usize,
    height: usize,
    depth_scale: f32,
) -> Result<Vec<f32>, TrainingError> {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("depth") => load_raw_depth(path, width, height),
        _ => load_raster_depth(path, width, height, depth_scale),
    }
}

fn load_raw_depth(path: &Path, width: usize, height: usize) -> Result<Vec<f32>, TrainingError> {
    let bytes = fs::read(path)?;
    let expected = width
        .saturating_mul(height)
        .saturating_mul(std::mem::size_of::<f32>());
    if bytes.len() != expected {
        return Err(TrainingError::InvalidInput(format!(
            "raw depth frame {} has {} bytes, expected {}",
            path.display(),
            bytes.len(),
            expected,
        )));
    }

    Ok(bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn load_raster_depth(
    path: &Path,
    width: usize,
    height: usize,
    depth_scale: f32,
) -> Result<Vec<f32>, TrainingError> {
    if depth_scale <= 0.0 {
        return Err(TrainingError::InvalidInput(format!(
            "depth scale must be > 0, got {depth_scale}",
        )));
    }

    let image = ImageReader::open(path)
        .map_err(TrainingError::Io)?
        .with_guessed_format()
        .map_err(|err| TrainingError::InvalidInput(err.to_string()))?
        .decode()
        .map_err(|err| TrainingError::InvalidInput(err.to_string()))?;

    let (actual_width, actual_height) = image.dimensions();
    if actual_width as usize != width || actual_height as usize != height {
        return Err(TrainingError::InvalidInput(format!(
            "depth image {} has size {}x{}, expected {}x{}",
            path.display(),
            actual_width,
            actual_height,
            width,
            height,
        )));
    }

    match image {
        DynamicImage::ImageLuma16(luma) => Ok(luma
            .pixels()
            .map(|pixel| pixel.0[0] as f32 / depth_scale)
            .collect()),
        _ => Err(TrainingError::InvalidInput(format!(
            "unsupported depth image format for {}",
            path.display(),
        ))),
    }
}

pub(super) fn synthetic_depth(
    color: &[u8],
    width: usize,
    height: usize,
    min_depth: f32,
    max_depth: f32,
) -> Vec<f32> {
    let expected = width.saturating_mul(height).saturating_mul(3);
    let mut depth = Vec::with_capacity(width.saturating_mul(height));
    if color.len() < expected || min_depth >= max_depth {
        depth.resize(width.saturating_mul(height), min_depth.max(0.01));
        return depth;
    }

    let range = max_depth - min_depth;
    for chunk in color.chunks_exact(3) {
        let r = chunk[0] as f32 / 255.0;
        let g = chunk[1] as f32 / 255.0;
        let b = chunk[2] as f32 / 255.0;
        let luma = 0.299 * r + 0.587 * g + 0.114 * b;
        let value = max_depth - luma * range;
        depth.push(value.clamp(min_depth, max_depth));
    }

    depth
}

#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9e37_79b9_7f4a_7c15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut z = self.state;
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn next_index(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            0
        } else {
            (self.next_u64() % upper as u64) as usize
        }
    }
}

fn deterministic_shuffle(indices: &mut [usize], seed: u64) {
    let mut rng = SplitMix64::new(seed ^ ((indices.len() as u64) << 32));
    for idx in (1..indices.len()).rev() {
        let swap_idx = rng.next_index(idx + 1);
        indices.swap(idx, swap_idx);
    }
}
