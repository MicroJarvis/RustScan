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
        if element_count == 0 {
            return Ok((Vec::new(), WgpuReadbackTiming::default()));
        }
        let byte_len = element_count
            .checked_mul(std::mem::size_of::<T>())
            .context("wgpu readback byte count overflow")?;
        let (mut regions, timing) = self.read_regions_profiled(&[(source, byte_len)])?;
        let values = decode_pod_region::<T>(&regions.remove(0));
        Ok((values, timing))
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
        let first_bytes = first_count
            .checked_mul(std::mem::size_of::<A>())
            .context("wgpu readback byte count overflow")?;
        let second_bytes = second_count
            .checked_mul(std::mem::size_of::<B>())
            .context("wgpu readback byte count overflow")?;
        if first_bytes == 0 && second_bytes == 0 {
            return Ok((Vec::new(), Vec::new(), WgpuReadbackTiming::default()));
        }
        let (regions, timing) =
            self.read_regions_profiled(&[(first, first_bytes), (second, second_bytes)])?;
        Ok((
            decode_pod_region::<A>(&regions[0]),
            decode_pod_region::<B>(&regions[1]),
            timing,
        ))
    }

    /// Copies the leading byte range of each source into one staging buffer
    /// (each region aligned to `COPY_BUFFER_ALIGNMENT`), waits once, and returns
    /// the raw bytes of each region in order.
    fn read_regions_profiled(
        &self,
        sources: &[(&wgpu::Buffer, usize)],
    ) -> Result<(Vec<Vec<u8>>, WgpuReadbackTiming)> {
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
        let wait_started = Instant::now();
        self.wait_for(submission)?;
        let wait_seconds = wait_started.elapsed().as_secs_f64();
        receiver
            .recv()
            .context("wgpu readback callback was dropped")?
            .context("wgpu readback mapping failed")?;

        let mapped = slice.get_mapped_range();
        let regions = offsets
            .iter()
            .map(|&(offset, bytes)| {
                let start = usize::try_from(offset).unwrap_or(usize::MAX);
                let end = usize::try_from(offset + bytes).unwrap_or(usize::MAX);
                mapped[start..end].to_vec()
            })
            .collect();
        drop(mapped);
        staging.unmap();
        let map_decode_seconds = map_decode_started.elapsed().as_secs_f64();
        Ok((
            regions,
            WgpuReadbackTiming {
                total_seconds: total_started.elapsed().as_secs_f64(),
                copy_submit_seconds,
                wait_seconds,
                map_decode_seconds,
                calls: 1,
                bytes: offsets.iter().map(|&(_, bytes)| bytes).sum(),
            },
        ))
    }
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
