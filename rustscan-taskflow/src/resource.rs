use std::collections::BTreeMap;

use crate::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeviceId(pub u32);

/// CPU allowances are cooperative concurrency limits, not OS core reservations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuRequest {
    pub min: usize,
    pub preferred: usize,
    pub max: usize,
}

impl CpuRequest {
    pub const fn fixed(threads: usize) -> Self {
        Self {
            min: threads,
            preferred: threads,
            max: threads,
        }
    }

    pub const fn scalable(min: usize, preferred: usize, max: usize) -> Self {
        Self {
            min,
            preferred,
            max,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuRequest {
    /// None lets the scheduler choose any configured device.
    pub device: Option<DeviceId>,
    pub working_memory_bytes: u64,
    pub output_memory_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceRequest {
    pub cpu: CpuRequest,
    pub working_memory_bytes: u64,
    /// Reserved until every handle to the published artifact has been dropped.
    pub output_memory_bytes: u64,
    pub gpu: Option<GpuRequest>,
    pub io_slots: usize,
    /// Exclusive application resources, e.g. "project-42/reconstruction".
    pub exclusive: Vec<String>,
}

impl ResourceRequest {
    pub fn cpu(cpu: CpuRequest) -> Self {
        Self {
            cpu,
            working_memory_bytes: 0,
            output_memory_bytes: 0,
            gpu: None,
            io_slots: 0,
            exclusive: Vec::new(),
        }
    }

    pub(crate) fn validate(&self) -> Result<(), Error> {
        if self.cpu.min == 0
            || self.cpu.min > self.cpu.preferred
            || self.cpu.preferred > self.cpu.max
        {
            return Err(Error::Invalid(
                "CPU request requires 1 <= min <= preferred <= max".into(),
            ));
        }
        self.working_memory_bytes
            .checked_add(self.output_memory_bytes)
            .ok_or_else(|| Error::Invalid("memory request overflow".into()))?;
        if let Some(gpu) = &self.gpu {
            gpu.working_memory_bytes
                .checked_add(gpu.output_memory_bytes)
                .ok_or_else(|| Error::Invalid("GPU memory request overflow".into()))?;
        }
        let mut keys = self.exclusive.clone();
        keys.sort();
        keys.dedup();
        if keys.len() != self.exclusive.len() || keys.iter().any(String::is_empty) {
            return Err(Error::Invalid(
                "exclusive resource keys must be nonempty and unique".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuCapacity {
    pub id: DeviceId,
    pub memory_bytes: u64,
    pub max_in_flight: usize,
    /// Charge GPU memory against the host budget too (Apple unified memory).
    pub shared_host_memory: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    pub cpu_threads: usize,
    pub memory_bytes: u64,
    pub io_slots: usize,
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub budget: Budget,
    pub gpus: Vec<GpuCapacity>,
    /// Maximum number of unfinished tasks admitted across workflows.
    pub max_pending_tasks: usize,
    /// Per-run event queue. Overflow drops telemetry, never completion messages.
    pub event_capacity: usize,
    /// After this many bypasses, reserve CPU admission for a waiting task
    /// whose other resources already fit. Never block memory-releasing consumers.
    pub max_bypasses: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        let cpu = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        Self {
            budget: Budget {
                cpu_threads: cpu.saturating_sub(1).max(1),
                memory_bytes: 1024 * 1024 * 1024,
                io_slots: 2,
            },
            gpus: Vec::new(),
            max_pending_tasks: 100_000,
            event_capacity: 1024,
            max_bypasses: 8,
        }
    }
}

impl RuntimeConfig {
    pub(crate) fn validate(&self) -> Result<(), Error> {
        if self.budget.cpu_threads == 0 || self.max_pending_tasks == 0 || self.max_bypasses == 0 {
            return Err(Error::Invalid(
                "CPU capacity, pending capacity and max_bypasses must be positive".into(),
            ));
        }
        let mut devices = std::collections::HashSet::new();
        for gpu in &self.gpus {
            if gpu.max_in_flight == 0 || !devices.insert(gpu.id) {
                return Err(Error::Invalid(
                    "GPU IDs must be unique and concurrency positive".into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuGrant {
    pub device: DeviceId,
    pub working_memory_bytes: u64,
    pub output_memory_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionGrant {
    pub variant: String,
    pub cpu_threads: usize,
    /// Includes GPU memory on unified-memory devices.
    pub working_memory_bytes: u64,
    pub output_memory_bytes: u64,
    pub gpu: Option<GpuGrant>,
    pub io_slots: usize,
    pub exclusive: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GpuUsage {
    pub memory_bytes: u64,
    pub in_flight: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceSnapshot {
    pub budget: Budget,
    pub cpu_threads: usize,
    pub memory_bytes: u64,
    pub io_slots: usize,
    pub gpus: BTreeMap<DeviceId, GpuUsage>,
    pub exclusive: Vec<String>,
    pub pending_tasks: usize,
}

pub(crate) struct Ledger {
    pub config: RuntimeConfig,
    pub usage: ResourceSnapshot,
}

impl Ledger {
    pub fn new(config: RuntimeConfig) -> Self {
        Self {
            usage: ResourceSnapshot {
                budget: config.budget,
                cpu_threads: 0,
                memory_bytes: 0,
                io_slots: 0,
                gpus: config
                    .gpus
                    .iter()
                    .map(|g| (g.id, GpuUsage::default()))
                    .collect(),
                exclusive: Vec::new(),
                pending_tasks: 0,
            },
            config,
        }
    }

    pub fn fits(&self, name: &str, req: &ResourceRequest) -> Option<ExecutionGrant> {
        self.fits_with_cpu_usage(name, req, self.usage.cpu_threads)
    }

    pub fn can_reserve_cpu(&self, req: &ResourceRequest) -> bool {
        self.fits_with_cpu_usage("", req, 0).is_some()
    }

    fn fits_with_cpu_usage(
        &self,
        name: &str,
        req: &ResourceRequest,
        used_cpu: usize,
    ) -> Option<ExecutionGrant> {
        if self.usage.budget.cpu_threads.saturating_sub(used_cpu) < req.cpu.min
            || self
                .usage
                .budget
                .io_slots
                .saturating_sub(self.usage.io_slots)
                < req.io_slots
            || req
                .exclusive
                .iter()
                .any(|key| self.usage.exclusive.contains(key))
        {
            return None;
        }
        let candidates: Vec<Option<&GpuCapacity>> = match &req.gpu {
            None => vec![None],
            Some(gpu) => self
                .config
                .gpus
                .iter()
                .filter(|d| gpu.device.is_none_or(|id| id == d.id))
                .map(Some)
                .collect(),
        };
        for device in candidates {
            let mut working = req.working_memory_bytes;
            let mut output = req.output_memory_bytes;
            let gpu = if let (Some(device), Some(request)) = (device, &req.gpu) {
                let used = &self.usage.gpus[&device.id];
                let memory = request
                    .working_memory_bytes
                    .checked_add(request.output_memory_bytes)?;
                if used.in_flight >= device.max_in_flight
                    || memory > device.memory_bytes.saturating_sub(used.memory_bytes)
                {
                    continue;
                }
                if device.shared_host_memory {
                    let (Some(w), Some(o)) = (
                        working.checked_add(request.working_memory_bytes),
                        output.checked_add(request.output_memory_bytes),
                    ) else {
                        continue;
                    };
                    working = w;
                    output = o;
                }
                Some(GpuGrant {
                    device: device.id,
                    working_memory_bytes: request.working_memory_bytes,
                    output_memory_bytes: request.output_memory_bytes,
                })
            } else {
                None
            };
            let Some(total) = working.checked_add(output) else {
                continue;
            };
            if total
                > self
                    .usage
                    .budget
                    .memory_bytes
                    .saturating_sub(self.usage.memory_bytes)
            {
                continue;
            }
            return Some(ExecutionGrant {
                variant: name.into(),
                cpu_threads: req
                    .cpu
                    .preferred
                    .min(req.cpu.max)
                    .min(self.usage.budget.cpu_threads.saturating_sub(used_cpu)),
                working_memory_bytes: working,
                output_memory_bytes: output,
                gpu,
                io_slots: req.io_slots,
                exclusive: req.exclusive.clone(),
            });
        }
        None
    }

    pub fn acquire(&mut self, grant: &ExecutionGrant) {
        self.usage.cpu_threads += grant.cpu_threads;
        self.usage.memory_bytes += grant.working_memory_bytes + grant.output_memory_bytes;
        self.usage.io_slots += grant.io_slots;
        self.usage.exclusive.extend(grant.exclusive.iter().cloned());
        if let Some(gpu) = &grant.gpu {
            let usage = self.usage.gpus.get_mut(&gpu.device).unwrap();
            usage.in_flight += 1;
            usage.memory_bytes += gpu.working_memory_bytes + gpu.output_memory_bytes;
        }
    }

    pub fn release_cpu(&mut self, grant: &ExecutionGrant) {
        self.usage.cpu_threads -= grant.cpu_threads;
    }

    pub fn finish(&mut self, grant: &ExecutionGrant, retain_output: bool) {
        self.usage.memory_bytes -= grant.working_memory_bytes;
        self.usage.io_slots -= grant.io_slots;
        self.usage
            .exclusive
            .retain(|key| !grant.exclusive.contains(key));
        if let Some(gpu) = &grant.gpu {
            let usage = self.usage.gpus.get_mut(&gpu.device).unwrap();
            usage.in_flight -= 1;
            usage.memory_bytes -= gpu.working_memory_bytes;
        }
        if !retain_output {
            self.release_output(
                grant.output_memory_bytes,
                grant
                    .gpu
                    .as_ref()
                    .map(|g| (g.device, g.output_memory_bytes)),
            );
        }
    }

    pub fn release_output(&mut self, memory: u64, gpu: Option<(DeviceId, u64)>) {
        self.usage.memory_bytes -= memory;
        if let Some((id, bytes)) = gpu {
            self.usage.gpus.get_mut(&id).unwrap().memory_bytes -= bytes;
        }
    }
}
