use super::{GpuBackendKind, GpuSiftCapabilities};
use anyhow::{Context, Result};
use bytemuck::Pod;
use std::sync::{mpsc, Arc};

#[derive(Debug)]
pub struct WgpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    capabilities: GpuSiftCapabilities,
}

impl WgpuContext {
    pub fn try_new() -> Result<Arc<Self>> {
        Self::try_new_optional()?.context("no compatible wgpu adapter is available")
    }

    pub fn try_new_optional() -> Result<Option<Arc<Self>>> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Option<Arc<Self>>> {
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
            capabilities: GpuSiftCapabilities {
                backend: GpuBackendKind::Wgpu,
                device_name: info.name,
            },
        })))
    }

    pub fn capabilities(&self) -> &GpuSiftCapabilities {
        &self.capabilities
    }

    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub(crate) fn queue(&self) -> &wgpu::Queue {
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
        if element_count == 0 {
            return Ok(Vec::new());
        }
        let byte_len = element_count
            .checked_mul(std::mem::size_of::<T>())
            .context("wgpu readback byte count overflow")?;
        let byte_len = u64::try_from(byte_len).context("wgpu readback does not fit u64")?;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustsfm wgpu readback staging"),
            size: byte_len,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rustsfm wgpu readback encoder"),
            });
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, byte_len);
        let submission = self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.wait_for(submission)?;
        receiver
            .recv()
            .context("wgpu readback callback was dropped")?
            .context("wgpu readback mapping failed")?;

        let mapped = slice.get_mapped_range();
        let element_size = std::mem::size_of::<T>();
        let values = mapped
            .chunks_exact(element_size)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        drop(mapped);
        staging.unmap();
        Ok(values)
    }
}
