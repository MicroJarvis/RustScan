use std::sync::Arc;

use crate::{Completion, DeviceId, TaskContext, TaskError};

/// An adapter over an EXISTING device/queue. The owner (e.g. eframe) must keep
/// polling the device to deliver completion callbacks. It must also handle wgpu
/// validation/device-lost errors; queue completion alone is not error validation.
/// No second device, polling thread, or independent scheduler is created here.
#[derive(Clone)]
pub struct WgpuBackend {
    id: DeviceId,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}
impl WgpuBackend {
    /// `id` must identify this device in RuntimeConfig. Pair the queue with the
    /// same device and reserve enough working/output bytes before allocating.
    pub fn new(id: DeviceId, device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self { id, device, queue }
    }
    pub fn id(&self) -> DeviceId {
        self.id
    }
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Submit valid command buffers and publish `output` after queue completion.
    /// The callback covers all work previously submitted to this shared queue,
    /// so concurrent rendering may conservatively delay completion. Readback
    /// mapping, validation errors and device loss can instead be handled by a
    /// custom TaskVariant::asynchronous adapter using its Completion directly.
    pub fn submit<T: Send + Sync + 'static>(
        &self,
        context: &TaskContext,
        commands: impl IntoIterator<Item = wgpu::CommandBuffer>,
        output: T,
        completion: Completion<T>,
    ) {
        if let Err(error) = context.check_cancelled() {
            completion.complete(Err(error));
            return;
        }
        if context.grant().gpu.as_ref().map(|g| g.device) != Some(self.id) {
            completion.complete(Err(TaskError::Failed(
                "wgpu device does not match the execution grant".into(),
            )));
            return;
        }
        self.queue.submit(commands);
        self.queue
            .on_submitted_work_done(move || completion.complete(Ok(output)));
    }
}
