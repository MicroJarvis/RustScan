use super::{GpuBackendKind, GpuSiftCapabilities};
use anyhow::{Context, Result};
use bytemuck::Pod;
use std::sync::{mpsc, Arc};
#[cfg(test)]
use std::sync::{Mutex, MutexGuard, OnceLock, TryLockError};
use std::time::Instant;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct WgpuReadbackTiming {
    pub(crate) total_seconds: f64,
    pub(crate) copy_submit_seconds: f64,
    pub(crate) wait_seconds: f64,
    pub(crate) map_decode_seconds: f64,
    pub(crate) calls: usize,
    pub(crate) bytes: u64,
}

#[derive(Debug)]
pub struct WgpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    backend: wgpu::Backend,
    capabilities: GpuSiftCapabilities,
    #[cfg(test)]
    // Keep Metal shader compilation from overlapping across unit-test threads.
    _test_gpu_context_lease: MutexGuard<'static, ()>,
}

impl WgpuContext {
    pub fn try_new() -> Result<Arc<Self>> {
        Self::try_new_optional()?.context(no_compatible_adapter_message())
    }

    pub fn try_new_optional() -> Result<Option<Arc<Self>>> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Option<Arc<Self>>> {
        #[cfg(test)]
        let test_gpu_context_lease = test_gpu_context_lease();
        #[cfg(feature = "gpu-vulkan")]
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::from_build_config().with_env(),
            backend_options: wgpu::BackendOptions::from_env_or_default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });
        #[cfg(not(feature = "gpu-vulkan"))]
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            Ok(adapter) => adapter,
            Err(_) => return Ok(None),
        };
        let info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("rustsfm-wgpu-sift"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await
            .context("failed to request the wgpu SIFT device")?;

        Ok(Some(Arc::new(Self {
            device,
            queue,
            backend: info.backend,
            capabilities: GpuSiftCapabilities {
                backend: gpu_backend_kind(info.backend),
                device_name: info.name,
            },
            #[cfg(test)]
            _test_gpu_context_lease: test_gpu_context_lease,
        })))
    }

    pub fn capabilities(&self) -> &GpuSiftCapabilities {
        &self.capabilities
    }

    pub fn backend(&self) -> wgpu::Backend {
        self.backend
    }

    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub(crate) fn queue(&self) -> &wgpu::Queue {
        crate::execution::register_gpu_device(self as *const Self as usize, &self.device);
        &self.queue
    }

    pub(crate) fn wait_for(&self, submission: wgpu::SubmissionIndex) -> Result<()> {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .context("wgpu device wait failed")?;
        Ok(())
    }

    pub(crate) fn read_buffer<T: Pod>(
        &self,
        source: &wgpu::Buffer,
        element_count: usize,
    ) -> Result<Vec<T>> {
        self.read_buffer_profiled(source, element_count)
            .map(|(values, _)| values)
    }

    pub(crate) fn read_buffer_profiled<T: Pod>(
        &self,
        source: &wgpu::Buffer,
        element_count: usize,
    ) -> Result<(Vec<T>, WgpuReadbackTiming)> {
        let (values, timing, ()) =
            self.read_buffer_profiled_overlapping(source, element_count, || ())?;
        Ok((values, timing))
    }

    /// Reads one buffer, running `overlap` after the copy is submitted and
    /// mapped so CPU work can hide the device wait. `overlap` must not submit
    /// GPU work against this device or reuse matcher/scorer scratch.
    pub(crate) fn read_buffer_profiled_overlapping<T, F, R>(
        &self,
        source: &wgpu::Buffer,
        element_count: usize,
        overlap: F,
    ) -> Result<(Vec<T>, WgpuReadbackTiming, R)>
    where
        T: Pod,
        F: FnOnce() -> R,
    {
        if element_count == 0 {
            let extra = overlap();
            return Ok((Vec::new(), WgpuReadbackTiming::default(), extra));
        }
        let byte_len = element_count
            .checked_mul(std::mem::size_of::<T>())
            .context("wgpu readback byte count overflow")?;
        let pending = self.start_read_regions(&[(source, byte_len)])?;
        let extra = overlap();
        let (mut regions, timing) = self.finish_read_regions(pending)?;
        let values = decode_pod_region::<T>(&regions.remove(0));
        Ok((values, timing, extra))
    }

    /// Reads the leading `element_count` elements of two buffers with one
    /// staging buffer, one copy submission and one device wait.
    pub(crate) fn read_two_buffers_profiled<A: Pod, B: Pod>(
        &self,
        first: &wgpu::Buffer,
        first_count: usize,
        second: &wgpu::Buffer,
        second_count: usize,
    ) -> Result<(Vec<A>, Vec<B>, WgpuReadbackTiming)> {
        let (first, second, timing, ()) = self.read_two_buffers_profiled_overlapping(
            first,
            first_count,
            second,
            second_count,
            || (),
        )?;
        Ok((first, second, timing))
    }

    /// Reads two buffers, running `overlap` after the copy is submitted and
    /// mapped so CPU work can hide the device wait. `overlap` must not submit
    /// GPU work against this device or reuse scorer scratch.
    pub(crate) fn read_two_buffers_profiled_overlapping<A, B, F, T>(
        &self,
        first: &wgpu::Buffer,
        first_count: usize,
        second: &wgpu::Buffer,
        second_count: usize,
        overlap: F,
    ) -> Result<(Vec<A>, Vec<B>, WgpuReadbackTiming, T)>
    where
        A: Pod,
        B: Pod,
        F: FnOnce() -> T,
    {
        let first_bytes = first_count
            .checked_mul(std::mem::size_of::<A>())
            .context("wgpu readback byte count overflow")?;
        let second_bytes = second_count
            .checked_mul(std::mem::size_of::<B>())
            .context("wgpu readback byte count overflow")?;
        if first_bytes == 0 && second_bytes == 0 {
            let extra = overlap();
            return Ok((Vec::new(), Vec::new(), WgpuReadbackTiming::default(), extra));
        }
        let pending = self.start_read_regions(&[(first, first_bytes), (second, second_bytes)])?;
        let extra = overlap();
        let (regions, timing) = self.finish_read_regions(pending)?;
        Ok((
            decode_pod_region::<A>(&regions[0]),
            decode_pod_region::<B>(&regions[1]),
            timing,
            extra,
        ))
    }

    /// Copies the leading byte range of each source into one staging buffer
    /// (each region aligned to `COPY_BUFFER_ALIGNMENT`), waits once, and returns
    /// the raw bytes of each region in order.
    fn read_regions_profiled(
        &self,
        sources: &[(&wgpu::Buffer, usize)],
    ) -> Result<(Vec<Vec<u8>>, WgpuReadbackTiming)> {
        let pending = self.start_read_regions(sources)?;
        self.finish_read_regions(pending)
    }

    fn start_read_regions(&self, sources: &[(&wgpu::Buffer, usize)]) -> Result<PendingReadback> {
        let total_started = Instant::now();
        let alignment = wgpu::COPY_BUFFER_ALIGNMENT;
        let mut offsets = Vec::with_capacity(sources.len());
        let mut total_len = 0u64;
        for &(_, bytes) in sources {
            let bytes = u64::try_from(bytes).context("wgpu readback does not fit u64")?;
            offsets.push((total_len, bytes));
            let padded = bytes.div_ceil(alignment) * alignment;
            total_len = total_len
                .checked_add(padded)
                .context("wgpu readback staging size overflow")?;
        }
        let copy_submit_started = Instant::now();
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm wgpu readback staging"),
            size: total_len.max(alignment),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rustsfm wgpu readback encoder"),
            });
        for (&(source, _), &(offset, bytes)) in sources.iter().zip(&offsets) {
            if bytes > 0 {
                encoder.copy_buffer_to_buffer(source, 0, &staging, offset, bytes);
            }
        }
        let submission = self.queue().submit(Some(encoder.finish()));
        let copy_submit_seconds = copy_submit_started.elapsed().as_secs_f64();

        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::sync_channel(1);
        let map_decode_started = Instant::now();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        Ok(PendingReadback {
            staging,
            offsets,
            receiver,
            submission,
            wait_started: Instant::now(),
            copy_submit_seconds,
            total_started,
            map_decode_started,
        })
    }

    fn finish_read_regions(
        &self,
        pending: PendingReadback,
    ) -> Result<(Vec<Vec<u8>>, WgpuReadbackTiming)> {
        self.wait_for(pending.submission)?;
        let wait_seconds = pending.wait_started.elapsed().as_secs_f64();
        pending
            .receiver
            .recv()
            .context("wgpu readback callback was dropped")?
            .context("wgpu readback mapping failed")?;

        let slice = pending.staging.slice(..);
        let mapped = slice.get_mapped_range();
        let regions = pending
            .offsets
            .iter()
            .map(|&(offset, bytes)| {
                let start = usize::try_from(offset).unwrap_or(usize::MAX);
                let end = usize::try_from(offset + bytes).unwrap_or(usize::MAX);
                mapped[start..end].to_vec()
            })
            .collect();
        drop(mapped);
        pending.staging.unmap();
        let map_decode_seconds = pending.map_decode_started.elapsed().as_secs_f64();
        Ok((
            regions,
            WgpuReadbackTiming {
                total_seconds: pending.total_started.elapsed().as_secs_f64(),
                copy_submit_seconds: pending.copy_submit_seconds,
                wait_seconds,
                map_decode_seconds,
                calls: 1,
                bytes: pending.offsets.iter().map(|&(_, bytes)| bytes).sum(),
            },
        ))
    }
}

struct PendingReadback {
    staging: wgpu::Buffer,
    offsets: Vec<(u64, u64)>,
    receiver: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    submission: wgpu::SubmissionIndex,
    wait_started: Instant,
    copy_submit_seconds: f64,
    total_started: Instant,
    map_decode_started: Instant,
}

fn decode_pod_region<T: Pod>(bytes: &[u8]) -> Vec<T> {
    bytes
        .chunks_exact(std::mem::size_of::<T>())
        .map(bytemuck::pod_read_unaligned)
        .collect()
}

fn no_compatible_adapter_message() -> &'static str {
    #[cfg(feature = "gpu-vulkan")]
    {
        "no compatible Vulkan adapter is available"
    }
    #[cfg(not(feature = "gpu-vulkan"))]
    {
        "no compatible wgpu adapter is available"
    }
}

fn gpu_backend_kind(backend: wgpu::Backend) -> GpuBackendKind {
    if backend == wgpu::Backend::Vulkan {
        GpuBackendKind::Vulkan
    } else {
        GpuBackendKind::Wgpu
    }
}

#[cfg(test)]
fn test_gpu_context_lease_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[cfg(test)]
fn test_gpu_context_lease() -> MutexGuard<'static, ()> {
    test_gpu_context_lease_lock()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

#[cfg(test)]
fn try_test_gpu_context_lease() -> Option<MutexGuard<'static, ()>> {
    match test_gpu_context_lease_lock().try_lock() {
        Ok(lease) => Some(lease),
        Err(TryLockError::WouldBlock) => None,
        Err(TryLockError::Poisoned(error)) => Some(error.into_inner()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_test_context_lease_excludes_parallel_initialization() {
        let first = test_gpu_context_lease();
        assert!(try_test_gpu_context_lease().is_none());
        drop(first);
        assert!(try_test_gpu_context_lease().is_some());
    }
}
