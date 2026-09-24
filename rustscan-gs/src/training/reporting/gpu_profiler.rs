//! GPU completion profiler and unified pipeline timing.
//!
//! CPU spans use host `Instant` and are reported under explicit timing_kind
//! labels. GPU percentiles cover **forward render only** (`gpu_timing_scope =
//! "forward"`): loss, backward, optimizer, and topology are outside the device
//! timestamp bracket. System/host waits (including `into_scalar_async`) must
//! never be labeled as GPU kernel time.
//!
//! Warmup drops the first [`PIPELINE_TIMING_WARMUP_SAMPLES`] samples per series
//! before aggregation. Aborted runs report only samples collected so far.
//! Resume constructs a new collector (samples do not include prior-run steps).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

pub use super::optimization_report::PipelineSpanStats;
use super::optimization_report::{duration_millis, percentile_f64};

/// Drop the first N samples per span before percentile aggregation.
pub const PIPELINE_TIMING_WARMUP_SAMPLES: usize = 1;

/// Stable reason when the adapter lacks `TIMESTAMP_QUERY`.
pub const UNSUPPORTED_TIMESTAMP_QUERY_UNAVAILABLE: &str = "timestamp_query_unavailable";

/// Stable reason when cubecl reports system timing instead of device timestamps.
pub const UNSUPPORTED_TIMING_METHOD_SYSTEM: &str = "timing_method_system";

/// Stable reason when adapter metadata cannot be resolved from the training device.
pub const UNSUPPORTED_ADAPTER_PROBE_FAILED: &str = "adapter_probe_failed";

/// Stable reason when cubecl memory usage is unavailable.
pub const UNSUPPORTED_RUNTIME_PEAK_UNAVAILABLE: &str = "runtime_peak_device_bytes_unavailable";

/// Observed high-water of allocator `bytes_in_use` samples — not a true allocator peak.
pub const SAMPLED_BYTES_IN_USE_HIGH_WATER: &str = "sampled_bytes_in_use_high_water";

/// Existing/shared device without stored adapter metadata.
pub const EXISTING_DEVICE_ADAPTER_METADATA_UNAVAILABLE: &str =
    "existing_device_adapter_metadata_unavailable";

/// DefaultDevice / BestAvailable cannot re-pick an adapter for name/driver.
///
/// The live training client may have been selected via
/// `CUBECL_WGPU_DEFAULT_DEVICE` or other runtime policy; guessing via
/// `request_adapter(HighPerformance)` can attribute the wrong GPU.
pub const DEFAULT_DEVICE_ADAPTER_METADATA_UNAVAILABLE: &str =
    "default_device_adapter_metadata_unavailable";

/// GPU samples cover render forward only.
pub const GPU_TIMING_SCOPE_FORWARD: &str = "forward";

/// Prefix-sum workspace ownership scope for workspace_* / fresh_step_allocations.
pub const WORKSPACE_SCOPE_PREFIX_SUM: &str = "prefix_sum";

/// Timing-kind labels for pipeline spans (honest classification).
pub mod timing_kind {
    /// Host thread blocked waiting for prefetch / frame availability.
    pub const HOST_WAIT: &str = "host_wait";
    /// Prefetch-worker wall (decode/resize); not on the train-step thread.
    pub const WORKER_WALL: &str = "worker_wall";
    /// Host Instant around CPU-side submit / orchestration.
    pub const CPU_SUBMIT: &str = "cpu_submit";
    /// Host Instant that may include GPU timestamp resolve / sync.
    pub const SYNCHRONIZED_BOUNDARY: &str = "synchronized_boundary";
    /// Full outer-loop iteration wall (frame wait through step end).
    pub const HOST_WALL: &str = "host_wall";
    /// Train-step wall after frame wait / upload prep.
    pub const STEP_WALL: &str = "step_wall";
}

/// Unified pipeline span names (CPU submit / wall unless documented otherwise).
pub mod span {
    pub const FRAME_WAIT: &str = "frame_wait";
    pub const DECODE: &str = "decode";
    pub const RESIZE: &str = "resize";
    pub const UPLOAD: &str = "upload";
    pub const FORWARD: &str = "forward";
    pub const BACKWARD: &str = "backward";
    pub const OPTIMIZER: &str = "optimizer";
    pub const TOPOLOGY_SNAPSHOT: &str = "topology_snapshot";
    pub const TOPOLOGY_PLAN: &str = "topology_plan";
    pub const TOPOLOGY_APPLY: &str = "topology_apply";
    pub const TOPOLOGY_UPLOAD: &str = "topology_upload";
    /// Visibility-state remap after a topology mutation.
    pub const TOPOLOGY_REMAP_VISIBILITY: &str = "topology_remap_visibility";
    /// Adam optimizer origin remap after a topology mutation.
    pub const TOPOLOGY_REMAP_OPTIMIZER: &str = "topology_remap_optimizer";
    /// Outer-loop iteration wall: starts before frame_wait, ends with step_cpu.
    pub const ITERATION_WALL: &str = "iteration_wall";
    /// Train-step wall after frame wait / target upload prep (not full iteration).
    pub const STEP_CPU: &str = "step_cpu";

    pub const ALL: &[&str] = &[
        ITERATION_WALL,
        FRAME_WAIT,
        DECODE,
        RESIZE,
        UPLOAD,
        FORWARD,
        BACKWARD,
        OPTIMIZER,
        TOPOLOGY_SNAPSHOT,
        TOPOLOGY_PLAN,
        TOPOLOGY_APPLY,
        TOPOLOGY_UPLOAD,
        TOPOLOGY_REMAP_VISIBILITY,
        TOPOLOGY_REMAP_OPTIMIZER,
        STEP_CPU,
    ];
}

/// Stable timing_kind for a named pipeline span.
pub fn span_timing_kind(name: &str) -> &'static str {
    match name {
        span::FRAME_WAIT => timing_kind::HOST_WAIT,
        span::DECODE | span::RESIZE => timing_kind::WORKER_WALL,
        span::FORWARD => timing_kind::SYNCHRONIZED_BOUNDARY,
        span::ITERATION_WALL => timing_kind::HOST_WALL,
        span::STEP_CPU => timing_kind::STEP_WALL,
        _ => timing_kind::CPU_SUBMIT,
    }
}

/// Serializable GPU profiler report distinguishing CPU submit from GPU completion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct GpuProfilerReport {
    /// True when the training device can produce device-timestamp GPU samples.
    pub supported: bool,
    pub unsupported_reason: Option<String>,
    pub backend: String,
    pub adapter: Option<String>,
    pub driver: Option<String>,
    pub adapter_unavailable_reason: Option<String>,
    pub driver_unavailable_reason: Option<String>,
    /// Hardware/capability: training adapter exposes timestamp queries (Device timing).
    pub timestamp_query_available: bool,
    /// CPU span collection enabled (host Instant).
    pub profiler_enabled: bool,
    /// Device timestamp profiling requested by config.
    pub gpu_timing_enabled: bool,
    /// True when at least one accepted GPU forward sample survived warmup.
    pub measurement_success: bool,
    /// Sample every N iterations when GPU timing is enabled.
    pub gpu_sample_every: usize,
    /// Scope of GPU timestamp samples (`"forward"` — not full train step).
    pub gpu_timing_scope: Option<String>,
    /// Sum of accepted (post-warmup) GPU forward samples in milliseconds.
    pub gpu_forward_sum_ms: Option<f64>,
    /// Forward-only GPU completion p50; null when unsupported or no samples.
    pub gpu_step_p50_ms: Option<f64>,
    /// Forward-only GPU completion p95; null when unsupported or no samples.
    pub gpu_step_p95_ms: Option<f64>,
    /// Accepted GPU forward sample count after warmup.
    pub sample_count: u64,
    /// Illegal (non-finite / negative) timing samples dropped at record sites.
    pub rejected_timing_samples: u64,
    /// Profile start failed before work ran (work then executed unprofiled).
    pub profile_start_failures: u64,
    /// Profile end failed after work already ran (timing dropped).
    pub profile_end_failures: u64,
    /// Profile Ok but device-ms resolve yielded no usable sample.
    pub profile_resolve_failures: u64,
    /// GPU timing samples dropped due to start/end/resolve failure (not rejects).
    pub dropped_profile_samples: u64,
    /// Scope of workspace_* / fresh_step_allocations (`"prefix_sum"`).
    pub workspace_scope: String,
    pub workspace_current_bytes: u64,
    pub workspace_peak_bytes: u64,
    pub workspace_growth_count: u64,
    /// Fresh prefix-sum workspace allocations only — not whole-runtime.
    pub fresh_step_allocations: u64,
    /// Max observed allocator `bytes_in_use` across sample points.
    pub runtime_peak_device_bytes: Option<u64>,
    pub runtime_peak_device_bytes_reason: Option<String>,
    /// Explicit CPU submit-side loop percentiles (`Instant`); never GPU.
    pub cpu_step_p50_ms: Option<f64>,
    pub cpu_step_p95_ms: Option<f64>,
    pub cpu_timing_kind: Option<String>,
    pub pipeline_spans: BTreeMap<String, PipelineSpanStats>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct GpuEnvironmentProbe {
    pub backend: String,
    pub adapter: Option<String>,
    pub driver: Option<String>,
    pub adapter_unavailable_reason: Option<String>,
    pub driver_unavailable_reason: Option<String>,
    pub timestamp_query_available: bool,
    pub timing_method_device: bool,
    pub unsupported_reason: Option<String>,
}

impl GpuEnvironmentProbe {
    pub fn gpu_timing_supported(&self) -> bool {
        self.timestamp_query_available && self.timing_method_device
    }
}

/// Optional adapter metadata captured from a shared wgpu setup (training device).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TrainingAdapterMetadata {
    pub backend: String,
    pub adapter: Option<String>,
    pub driver: Option<String>,
    pub timestamp_query_available: bool,
}

impl TrainingAdapterMetadata {
    pub fn from_wgpu_adapter(adapter: &wgpu::Adapter, backend: wgpu::Backend) -> Self {
        let info = adapter.get_info();
        let (adapter_name, driver) = adapter_name_and_driver(&info);
        Self {
            backend: format!("{backend:?}"),
            adapter: adapter_name,
            driver,
            timestamp_query_available: adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY),
        }
    }
}

fn adapter_name_and_driver(info: &wgpu::AdapterInfo) -> (Option<String>, Option<String>) {
    let adapter_name = if info.name.is_empty() {
        None
    } else {
        Some(info.name.clone())
    };
    let driver = {
        let mut parts = Vec::new();
        if !info.driver.is_empty() {
            parts.push(info.driver.clone());
        }
        if !info.driver_info.is_empty() {
            parts.push(info.driver_info.clone());
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(" / "))
        }
    };
    (adapter_name, driver)
}

/// Probe a fresh HighPerformance adapter — diagnostic only, not for training reports.
///
/// Training reports must use [`probe_training_device`] (or SharedWgpuContext metadata).
pub fn probe_wgpu_environment() -> GpuEnvironmentProbe {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = match burn_cubecl::cubecl::future::block_on(instance.request_adapter(
        &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        },
    )) {
        Ok(adapter) => adapter,
        Err(_) => {
            return GpuEnvironmentProbe {
                backend: "wgpu".into(),
                adapter: None,
                driver: None,
                adapter_unavailable_reason: Some(UNSUPPORTED_ADAPTER_PROBE_FAILED.into()),
                driver_unavailable_reason: Some(UNSUPPORTED_ADAPTER_PROBE_FAILED.into()),
                timestamp_query_available: false,
                timing_method_device: false,
                unsupported_reason: Some(UNSUPPORTED_ADAPTER_PROBE_FAILED.into()),
            };
        }
    };

    let info = adapter.get_info();
    let backend = format!("{:?}", info.backend);
    let (adapter_name, driver) = adapter_name_and_driver(&info);
    let timestamp_query_available = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);

    let mut unsupported_reason = None;
    if !timestamp_query_available {
        unsupported_reason = Some(UNSUPPORTED_TIMESTAMP_QUERY_UNAVAILABLE.into());
    }

    GpuEnvironmentProbe {
        backend,
        adapter: adapter_name.clone(),
        driver: driver.clone(),
        adapter_unavailable_reason: adapter_name.is_none().then(|| "adapter_name_empty".into()),
        driver_unavailable_reason: driver.is_none().then(|| "driver_info_empty".into()),
        timestamp_query_available,
        timing_method_device: timestamp_query_available,
        unsupported_reason,
    }
}

/// Build a probe from SharedWgpuContext / setup metadata, refined with client timing.
pub fn probe_from_training_adapter_metadata(
    meta: &TrainingAdapterMetadata,
    timing_method_device: bool,
) -> GpuEnvironmentProbe {
    let mut probe = GpuEnvironmentProbe {
        backend: meta.backend.clone(),
        adapter: meta.adapter.clone(),
        driver: meta.driver.clone(),
        adapter_unavailable_reason: meta.adapter.is_none().then(|| "adapter_name_empty".into()),
        driver_unavailable_reason: meta.driver.is_none().then(|| "driver_info_empty".into()),
        timestamp_query_available: meta.timestamp_query_available,
        timing_method_device,
        unsupported_reason: None,
    };
    refine_probe_unsupported_reason(&mut probe);
    probe
}

fn refine_probe_unsupported_reason(probe: &mut GpuEnvironmentProbe) {
    if probe.timestamp_query_available && !probe.timing_method_device {
        probe.unsupported_reason = Some(UNSUPPORTED_TIMING_METHOD_SYSTEM.into());
    } else if !probe.timestamp_query_available {
        probe.unsupported_reason = Some(UNSUPPORTED_TIMESTAMP_QUERY_UNAVAILABLE.into());
    } else {
        probe.unsupported_reason = None;
    }
}

/// Refine probe with live cubecl client timing method.
pub fn refine_probe_with_client_timing(
    mut probe: GpuEnvironmentProbe,
    timing_method_device: bool,
) -> GpuEnvironmentProbe {
    probe.timing_method_device = timing_method_device;
    // Capability follows the training client's timing method, not a separate adapter.
    if timing_method_device {
        probe.timestamp_query_available = true;
    }
    refine_probe_unsupported_reason(&mut probe);
    probe
}

fn is_valid_timing_ms(ms: f64) -> bool {
    ms.is_finite() && ms >= 0.0
}

#[derive(Debug, Default)]
pub struct PipelineTimingCollector {
    spans: BTreeMap<String, Vec<f64>>,
    gpu_step_ms: Vec<f64>,
    cpu_step_ms: Vec<f64>,
    workspace_current_bytes: u64,
    workspace_peak_bytes: u64,
    workspace_growth_count: u64,
    /// Cumulative fresh prefix-sum allocations across observed steps (monotonic).
    fresh_step_allocations: u64,
    runtime_peak_device_bytes: Option<u64>,
    runtime_peak_device_bytes_reason: Option<String>,
    probe: GpuEnvironmentProbe,
    profiler_enabled: bool,
    gpu_timing_enabled: bool,
    gpu_sample_every: usize,
    rejected_timing_samples: u64,
    profile_start_failures: u64,
    profile_end_failures: u64,
    profile_resolve_failures: u64,
    dropped_profile_samples: u64,
}

impl PipelineTimingCollector {
    pub fn new(probe: GpuEnvironmentProbe) -> Self {
        Self {
            probe,
            runtime_peak_device_bytes_reason: Some(UNSUPPORTED_RUNTIME_PEAK_UNAVAILABLE.into()),
            profiler_enabled: true,
            gpu_timing_enabled: false,
            gpu_sample_every: 20,
            ..Self::default()
        }
    }

    pub fn probe(&self) -> &GpuEnvironmentProbe {
        &self.probe
    }

    pub fn set_probe(&mut self, probe: GpuEnvironmentProbe) {
        self.probe = probe;
    }

    /// Record configured profiling mode for the report.
    pub fn set_profiler_mode(
        &mut self,
        enabled: bool,
        gpu_timing_enabled: bool,
        gpu_sample_every: usize,
    ) {
        self.profiler_enabled = enabled;
        self.gpu_timing_enabled = gpu_timing_enabled;
        self.gpu_sample_every = gpu_sample_every.max(1);
    }

    pub fn rejected_timing_samples(&self) -> u64 {
        self.rejected_timing_samples
    }

    /// Record structured GPU profile attempt failure / drop counters.
    pub fn record_profile_attempt_stats(&mut self, stats: &ProfileAttemptStats) {
        self.profile_start_failures = self
            .profile_start_failures
            .saturating_add(stats.start_failures);
        self.profile_end_failures = self.profile_end_failures.saturating_add(stats.end_failures);
        self.profile_resolve_failures = self
            .profile_resolve_failures
            .saturating_add(stats.resolve_failures);
        self.dropped_profile_samples = self
            .dropped_profile_samples
            .saturating_add(stats.dropped_samples);
    }

    pub fn record_span_ms(&mut self, name: &str, ms: f64) {
        if !self.profiler_enabled {
            return;
        }
        if !is_valid_timing_ms(ms) {
            self.rejected_timing_samples = self.rejected_timing_samples.saturating_add(1);
            return;
        }
        self.spans.entry(name.to_string()).or_default().push(ms);
    }

    pub fn record_span(&mut self, name: &str, duration: Duration) {
        self.record_span_ms(name, duration_millis(duration));
    }

    pub fn record_cpu_step(&mut self, duration: Duration) {
        if !self.profiler_enabled {
            return;
        }
        let ms = duration_millis(duration);
        if !is_valid_timing_ms(ms) {
            self.rejected_timing_samples = self.rejected_timing_samples.saturating_add(1);
            return;
        }
        self.cpu_step_ms.push(ms);
        self.record_span_ms(span::STEP_CPU, ms);
    }

    /// Record a GPU forward completion sample only when device timestamps are supported.
    pub fn record_gpu_step_ms(&mut self, ms: f64) {
        if !is_valid_timing_ms(ms) {
            self.rejected_timing_samples = self.rejected_timing_samples.saturating_add(1);
            return;
        }
        if self.probe.gpu_timing_supported() {
            self.gpu_step_ms.push(ms);
        }
    }

    pub fn observe_workspace(
        &mut self,
        current_bytes: u64,
        growth_count: u64,
        step_fresh_allocations: u64,
    ) {
        self.workspace_current_bytes = current_bytes;
        self.workspace_peak_bytes = self.workspace_peak_bytes.max(current_bytes);
        // Growth count is cumulative from the workspace; take max to stay monotonic.
        self.workspace_growth_count = self.workspace_growth_count.max(growth_count);
        self.fresh_step_allocations = self
            .fresh_step_allocations
            .saturating_add(step_fresh_allocations);
    }

    pub fn observe_runtime_device_bytes(&mut self, bytes: Option<u64>) {
        match bytes {
            Some(bytes) => {
                self.runtime_peak_device_bytes =
                    Some(self.runtime_peak_device_bytes.unwrap_or(0).max(bytes));
                // Honesty: this is sampled bytes_in_use HWM, not allocator true peak.
                self.runtime_peak_device_bytes_reason =
                    Some(SAMPLED_BYTES_IN_USE_HIGH_WATER.into());
            }
            None => {
                if self.runtime_peak_device_bytes.is_none() {
                    self.runtime_peak_device_bytes_reason =
                        Some(UNSUPPORTED_RUNTIME_PEAK_UNAVAILABLE.into());
                }
            }
        }
    }

    pub fn build_report(&self) -> GpuProfilerReport {
        let supported = self.probe.gpu_timing_supported();
        let gpu_samples = apply_warmup(&self.gpu_step_ms);
        let cpu_samples = apply_warmup(&self.cpu_step_ms);

        let (gpu_step_p50_ms, gpu_step_p95_ms, sample_count, gpu_forward_sum_ms) = if supported {
            let mut sorted = gpu_samples;
            let sum = if sorted.is_empty() {
                None
            } else {
                Some(sorted.iter().sum::<f64>())
            };
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let count = sorted.len() as u64;
            if count == 0 {
                (None, None, 0, None)
            } else {
                (
                    percentile_f64(&sorted, 50.0),
                    percentile_f64(&sorted, 95.0),
                    count,
                    sum,
                )
            }
        } else {
            (None, None, 0, None)
        };

        let mut cpu_sorted = cpu_samples;
        cpu_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let (cpu_step_p50_ms, cpu_step_p95_ms) = if cpu_sorted.is_empty() {
            (None, None)
        } else {
            (
                percentile_f64(&cpu_sorted, 50.0),
                percentile_f64(&cpu_sorted, 95.0),
            )
        };

        let mut pipeline_spans = BTreeMap::new();
        for name in span::ALL {
            let raw = self.spans.get(*name).cloned().unwrap_or_default();
            let warmed = apply_warmup(&raw);
            let sample_count = warmed.len() as u64;
            let mut sorted = warmed;
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let (p50_ms, p95_ms) = if sample_count == 0 {
                (None, None)
            } else {
                (percentile_f64(&sorted, 50.0), percentile_f64(&sorted, 95.0))
            };
            let timing_kind = span_timing_kind(name);
            pipeline_spans.insert(
                (*name).to_string(),
                PipelineSpanStats {
                    sample_count,
                    p50_ms,
                    p95_ms,
                    timing_kind: timing_kind.into(),
                },
            );
        }

        GpuProfilerReport {
            supported,
            unsupported_reason: if supported {
                None
            } else {
                self.probe
                    .unsupported_reason
                    .clone()
                    .or_else(|| Some(UNSUPPORTED_TIMESTAMP_QUERY_UNAVAILABLE.into()))
            },
            backend: self.probe.backend.clone(),
            adapter: self.probe.adapter.clone(),
            driver: self.probe.driver.clone(),
            adapter_unavailable_reason: self.probe.adapter_unavailable_reason.clone(),
            driver_unavailable_reason: self.probe.driver_unavailable_reason.clone(),
            timestamp_query_available: self.probe.timestamp_query_available,
            profiler_enabled: self.profiler_enabled,
            gpu_timing_enabled: self.gpu_timing_enabled,
            measurement_success: sample_count > 0,
            gpu_sample_every: self.gpu_sample_every,
            gpu_timing_scope: Some(GPU_TIMING_SCOPE_FORWARD.into()),
            gpu_forward_sum_ms,
            gpu_step_p50_ms,
            gpu_step_p95_ms,
            sample_count,
            rejected_timing_samples: self.rejected_timing_samples,
            profile_start_failures: self.profile_start_failures,
            profile_end_failures: self.profile_end_failures,
            profile_resolve_failures: self.profile_resolve_failures,
            dropped_profile_samples: self.dropped_profile_samples,
            workspace_scope: WORKSPACE_SCOPE_PREFIX_SUM.into(),
            workspace_current_bytes: self.workspace_current_bytes,
            workspace_peak_bytes: self.workspace_peak_bytes,
            workspace_growth_count: self.workspace_growth_count,
            fresh_step_allocations: self.fresh_step_allocations,
            runtime_peak_device_bytes: self.runtime_peak_device_bytes,
            runtime_peak_device_bytes_reason: self.runtime_peak_device_bytes_reason.clone(),
            cpu_step_p50_ms,
            cpu_step_p95_ms,
            cpu_timing_kind: Some(timing_kind::STEP_WALL.into()),
            pipeline_spans,
        }
    }
}

fn apply_warmup(samples: &[f64]) -> Vec<f64> {
    if samples.len() <= PIPELINE_TIMING_WARMUP_SAMPLES {
        return Vec::new();
    }
    samples[PIPELINE_TIMING_WARMUP_SAMPLES..].to_vec()
}

/// RAII CPU span timer for host Instant spans.
pub struct CpuSpanTimer {
    name: &'static str,
    started: Instant,
}

impl CpuSpanTimer {
    pub fn start(name: &'static str) -> Self {
        Self {
            name,
            started: Instant::now(),
        }
    }

    pub fn finish(self, collector: &mut PipelineTimingCollector) {
        collector.record_span(self.name, self.started.elapsed());
    }
}

/// Build environment probe from the actual training `GsDevice` / cubecl client.
///
/// Adapter name/driver are resolved against the training client's backend and
/// device selection when possible. `WgpuDevice::Existing` without registered
/// SharedWgpuContext metadata yields null + reason (callers should
/// [`PipelineTimingCollector::set_probe`] from SharedWgpuContext metadata).
pub fn probe_training_device(device: &crate::training::engine::GsDevice) -> GpuEnvironmentProbe {
    use burn_cubecl::cubecl::profile::TimingMethod;
    use burn_cubecl::cubecl::Runtime;
    use burn_wgpu::WgpuRuntime;

    let client = WgpuRuntime::client(device);
    let backend = *client.info();
    let timing_method_device = client.properties().timing_method == TimingMethod::Device;
    // CubeCL sets Device timing only when the training adapter has TIMESTAMP_QUERY.
    let timestamp_query_available = timing_method_device;

    let (adapter, driver, adapter_reason, driver_reason) =
        resolve_training_adapter_metadata(device, backend);

    let mut probe = GpuEnvironmentProbe {
        backend: format!("{backend:?}"),
        adapter,
        driver,
        adapter_unavailable_reason: adapter_reason,
        driver_unavailable_reason: driver_reason,
        timestamp_query_available,
        timing_method_device,
        unsupported_reason: None,
    };
    refine_probe_unsupported_reason(&mut probe);
    probe
}

fn resolve_training_adapter_metadata(
    device: &crate::training::engine::GsDevice,
    backend: wgpu::Backend,
) -> (
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
) {
    use burn_wgpu::WgpuDevice;

    match device {
        WgpuDevice::Existing(_) => (
            None,
            None,
            Some(EXISTING_DEVICE_ADAPTER_METADATA_UNAVAILABLE.into()),
            Some(EXISTING_DEVICE_ADAPTER_METADATA_UNAVAILABLE.into()),
        ),
        // Do not re-request HighPerformance — may diverge from the live client
        // (e.g. CUBECL_WGPU_DEFAULT_DEVICE override). TimingMethod still comes
        // from the training client in probe_training_device.
        WgpuDevice::DefaultDevice => (
            None,
            None,
            Some(DEFAULT_DEVICE_ADAPTER_METADATA_UNAVAILABLE.into()),
            Some(DEFAULT_DEVICE_ADAPTER_METADATA_UNAVAILABLE.into()),
        ),
        #[allow(deprecated)]
        WgpuDevice::BestAvailable => (
            None,
            None,
            Some(DEFAULT_DEVICE_ADAPTER_METADATA_UNAVAILABLE.into()),
            Some(DEFAULT_DEVICE_ADAPTER_METADATA_UNAVAILABLE.into()),
        ),
        other => match select_adapter_matching_device(other, backend) {
            Some(adapter) => {
                let info = adapter.get_info();
                let (name, driver) = adapter_name_and_driver(&info);
                (
                    name.clone(),
                    driver.clone(),
                    name.is_none().then(|| "adapter_name_empty".into()),
                    driver.is_none().then(|| "driver_info_empty".into()),
                )
            }
            None => (
                None,
                None,
                Some(UNSUPPORTED_ADAPTER_PROBE_FAILED.into()),
                Some(UNSUPPORTED_ADAPTER_PROBE_FAILED.into()),
            ),
        },
    }
}

fn select_adapter_matching_device(
    device: &burn_wgpu::WgpuDevice,
    backend: wgpu::Backend,
) -> Option<wgpu::Adapter> {
    use burn_wgpu::WgpuDevice;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: backend.into(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapters =
        burn_cubecl::cubecl::future::block_on(instance.enumerate_adapters(backend.into()));

    let filtered: Vec<_> = adapters
        .into_iter()
        .filter(|adapter| adapter.get_info().backend == backend)
        .collect();

    match device {
        WgpuDevice::DiscreteGpu(index) => filtered
            .into_iter()
            .filter(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
            .nth(*index),
        WgpuDevice::IntegratedGpu(index) => filtered
            .into_iter()
            .filter(|a| a.get_info().device_type == wgpu::DeviceType::IntegratedGpu)
            .nth(*index),
        WgpuDevice::VirtualGpu(index) => filtered
            .into_iter()
            .filter(|a| a.get_info().device_type == wgpu::DeviceType::VirtualGpu)
            .nth(*index),
        WgpuDevice::Cpu => filtered
            .into_iter()
            .find(|a| a.get_info().device_type == wgpu::DeviceType::Cpu),
        // Handled in resolve_training_adapter_metadata — never re-pick.
        WgpuDevice::DefaultDevice | WgpuDevice::Existing(_) => None,
        #[allow(deprecated)]
        WgpuDevice::BestAvailable => None,
    }
}

/// Read cubecl allocator bytes-in-use for runtime peak tracking.
pub fn runtime_device_bytes_in_use(device: &crate::training::engine::GsDevice) -> Option<u64> {
    use burn_cubecl::cubecl::Runtime;
    use burn_wgpu::WgpuRuntime;

    let client = WgpuRuntime::client(device);
    client.memory_usage().ok().map(|usage| usage.bytes_in_use)
}

/// Resolve a cubecl profile duration into GPU milliseconds only for Device timing.
///
/// # Panic / failure honesty
///
/// CubeCL's wgpu timestamp resolve path may `expect`/panic on map-buffer failure
/// (see cubecl-wgpu timings). This wrapper cannot convert that into a silent
/// `None` success — a panic aborts the process. When timing_method is not
/// Device, the future is drained and `None` is returned (not counted as GPU time).
pub async fn resolve_device_gpu_ms(
    profile: burn_cubecl::cubecl::profile::ProfileDuration,
) -> Option<f64> {
    use burn_cubecl::cubecl::profile::TimingMethod;

    if profile.timing_method() != TimingMethod::Device {
        // Drain the future so the map buffer is released; discard as GPU time.
        let _ = profile.resolve().await;
        return None;
    }
    let ticks = profile.resolve().await;
    Some(duration_millis(ticks.duration()))
}

/// Whether this iteration should capture a device-timestamp GPU sample.
///
/// Cadence: iteration 1, then every `sample_every` iterations. When a total
/// iteration count is known, the final iteration is also sampled.
pub fn should_profile_gpu(
    iteration: usize,
    sample_every: usize,
    total_iterations: Option<usize>,
) -> bool {
    let every = sample_every.max(1);
    if iteration == 0 {
        return false;
    }
    if iteration == 1 || iteration.is_multiple_of(every) {
        return true;
    }
    matches!(total_iterations, Some(total) if total > 0 && iteration == total)
}

/// Counters for a single GPU profile attempt (start / end / resolve).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProfileAttemptStats {
    pub start_failures: u64,
    pub end_failures: u64,
    pub resolve_failures: u64,
    pub dropped_samples: u64,
}

/// Classification of profile Err recovery (shared by production and tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileRecoveryKind {
    /// Work already ran and output is stored — do not re-run; drop timing.
    EndFailed,
    /// Profile never started — run work once unprofiled.
    StartFailed,
    /// Both slots empty (protocol violation).
    Impossible,
}

/// Classify profile Err slots without consuming them.
pub fn classify_profile_recovery<T, F>(
    work_slot: &Option<F>,
    output_slot: &Option<T>,
) -> ProfileRecoveryKind {
    if output_slot.is_some() {
        ProfileRecoveryKind::EndFailed
    } else if work_slot.is_some() {
        ProfileRecoveryKind::StartFailed
    } else {
        ProfileRecoveryKind::Impossible
    }
}

/// Recover after a profile attempt that returned `Err`.
///
/// Work must run at most once: prefer an already-stored output (end-profile
/// failure after the closure ran), else run remaining work once (start failure).
/// Production [`profile_device_gpu_step`] uses the same classification via
/// [`classify_profile_recovery`].
#[cfg(test)]
pub fn recover_profile_result<T, F>(
    work_slot: &mut Option<F>,
    output_slot: &mut Option<T>,
) -> Result<(T, Option<f64>), String>
where
    F: FnOnce() -> T,
{
    match classify_profile_recovery(work_slot, output_slot) {
        ProfileRecoveryKind::EndFailed => {
            let output = output_slot
                .take()
                .expect("EndFailed requires stored output");
            Ok((output, None))
        }
        ProfileRecoveryKind::StartFailed => {
            let work = work_slot.take().expect("StartFailed requires work");
            Ok((work(), None))
        }
        ProfileRecoveryKind::Impossible => Err(
            "gpu profile recovery: work already consumed and no output stored (impossible)".into(),
        ),
    }
}

/// Run `work` under cubecl device timestamp profiling when supported.
///
/// The future is driven with `cubecl::future::block_on` inside `client.profile`
/// so timestamp start/end bracket the submitted GPU work. Callers must not
/// include loss `into_scalar_async` readback inside `work` — that host sync is
/// CPU time, not GPU completion.
///
/// When profiling cannot start or end fails after work ran, `work` still runs
/// at most once (unprofiled on start failure; timing dropped on end failure).
/// Failure counters are returned in [`ProfileAttemptStats`].
///
/// Resolve path honesty: cubecl may panic inside `ProfileDuration::resolve` on
/// map failure; that is not mapped to a silent successful unsampled step.
pub async fn profile_device_gpu_step<F, Fut, T>(
    device: &crate::training::engine::GsDevice,
    work: F,
) -> (T, Option<f64>, ProfileAttemptStats)
where
    F: FnOnce() -> Fut + Send,
    Fut: std::future::Future<Output = T>,
    T: Send + 'static,
{
    use burn_cubecl::cubecl::profile::TimingMethod;
    use burn_cubecl::cubecl::Runtime;
    use burn_wgpu::WgpuRuntime;
    use std::sync::{Arc, Mutex};

    let mut stats = ProfileAttemptStats::default();
    let client = WgpuRuntime::client(device);
    if client.properties().timing_method != TimingMethod::Device {
        return (work().await, None, stats);
    }

    let work_slot = Arc::new(Mutex::new(Some(work)));
    let output_slot: Arc<Mutex<Option<T>>> = Arc::new(Mutex::new(None));
    let work_slot_for_profile = Arc::clone(&work_slot);
    let output_slot_for_profile = Arc::clone(&output_slot);
    match client.profile(
        move || {
            let work = work_slot_for_profile
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .take();
            let Some(work) = work else {
                // Closure must not re-enter after work was taken.
                return;
            };
            let output = burn_cubecl::cubecl::future::block_on(work());
            *output_slot_for_profile
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(output);
        },
        "gpu_forward",
    ) {
        Ok(((), profile)) => {
            let output = {
                output_slot
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .take()
            };
            match output {
                Some(output) => {
                    let ms = resolve_device_gpu_ms(profile).await;
                    if ms.is_none() {
                        stats.resolve_failures = 1;
                        stats.dropped_samples = 1;
                    }
                    (output, ms, stats)
                }
                None => {
                    // Profile Ok without stored output cannot happen with the
                    // slot protocol above; treat as unrecoverable.
                    panic!("gpu profile Ok without output: work closure did not store a result");
                }
            }
        }
        Err(_) => {
            let kind = {
                let work_guard = work_slot
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let output_guard = output_slot
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                classify_profile_recovery(&*work_guard, &*output_guard)
            };
            match kind {
                ProfileRecoveryKind::EndFailed => {
                    stats.end_failures = 1;
                    stats.dropped_samples = 1;
                    let output = output_slot
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .take()
                        .expect("EndFailed requires stored output");
                    (output, None, stats)
                }
                ProfileRecoveryKind::StartFailed => {
                    stats.start_failures = 1;
                    stats.dropped_samples = 1;
                    let work = work_slot
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .take()
                        .expect("StartFailed requires work");
                    (work().await, None, stats)
                }
                ProfileRecoveryKind::Impossible => panic!(
                    "gpu profile recovery: work already consumed and no output stored (impossible)"
                ),
            }
        }
    }
}

/// Validate percentile pair against sample_count.
///
/// - `sample_count == 0` ⇒ both percentiles must be `None`
/// - `sample_count > 0` ⇒ both `Some`, finite, non-negative, and `p95 >= p50`
fn percentile_pair_valid(
    sample_count: u64,
    p50: Option<f64>,
    p95: Option<f64>,
) -> Result<(), String> {
    if sample_count == 0 {
        return match (p50, p95) {
            (None, None) => Ok(()),
            _ => Err(format!(
                "sample_count==0 requires null percentiles, got p50={p50:?} p95={p95:?}"
            )),
        };
    }
    match (p50, p95) {
        (Some(p50), Some(p95)) => {
            if !p50.is_finite() || !p95.is_finite() || p50 < 0.0 || p95 < 0.0 {
                return Err(format!(
                    "percentiles must be finite non-negative: p50={p50} p95={p95}"
                ));
            }
            if p95 < p50 {
                return Err(format!("p95 ({p95}) < p50 ({p50})"));
            }
            Ok(())
        }
        _ => Err(format!(
            "sample_count={sample_count}>0 requires both percentiles Some, got p50={p50:?} p95={p95:?}"
        )),
    }
}

/// Validate profiler report self-consistency (finite non-negative, null semantics).
pub fn assert_report_self_consistent(report: &GpuProfilerReport) -> Result<(), String> {
    if report.supported {
        if report.unsupported_reason.is_some() {
            return Err("supported report must not set unsupported_reason".into());
        }
        if report.sample_count > 0 {
            percentile_pair_valid(
                report.sample_count,
                report.gpu_step_p50_ms,
                report.gpu_step_p95_ms,
            )?;
            match report.gpu_forward_sum_ms {
                Some(sum) if sum.is_finite() && sum >= 0.0 => {}
                Some(sum) => {
                    return Err(format!(
                        "gpu_forward_sum_ms must be finite non-negative: {sum}"
                    ));
                }
                None => {
                    return Err("supported report with samples must have gpu_forward_sum_ms".into());
                }
            }
            if !report.measurement_success {
                return Err("measurement_success must be true when sample_count > 0".into());
            }
        } else {
            percentile_pair_valid(0, report.gpu_step_p50_ms, report.gpu_step_p95_ms)?;
            if report.gpu_forward_sum_ms.is_some() {
                return Err("zero GPU samples ⇒ null gpu_forward_sum_ms".into());
            }
            if report.measurement_success {
                return Err("measurement_success must be false when sample_count is 0".into());
            }
        }
    } else {
        if report
            .unsupported_reason
            .as_deref()
            .unwrap_or("")
            .is_empty()
        {
            return Err("unsupported report requires unsupported_reason".into());
        }
        if report.gpu_step_p50_ms.is_some() || report.gpu_step_p95_ms.is_some() {
            return Err("unsupported report must null gpu_step percentiles".into());
        }
        if report.gpu_forward_sum_ms.is_some() {
            return Err("unsupported report must null gpu_forward_sum_ms".into());
        }
        if report.sample_count != 0 {
            return Err("unsupported report sample_count must be 0".into());
        }
    }

    if report.gpu_timing_scope.as_deref() != Some(GPU_TIMING_SCOPE_FORWARD) {
        return Err("gpu_timing_scope must be \"forward\"".into());
    }
    if report.workspace_scope != WORKSPACE_SCOPE_PREFIX_SUM {
        return Err("workspace_scope must be \"prefix_sum\"".into());
    }

    let cpu_span_count = report
        .pipeline_spans
        .get(span::STEP_CPU)
        .map(|s| s.sample_count)
        .unwrap_or(0);
    // cpu_step_* mirrors step_cpu span after warmup; require matching nullness.
    percentile_pair_valid(
        cpu_span_count,
        report.cpu_step_p50_ms,
        report.cpu_step_p95_ms,
    )?;
    if let Some(kind) = report.cpu_timing_kind.as_deref() {
        if kind != timing_kind::STEP_WALL {
            return Err(format!(
                "cpu_timing_kind must be \"{}\", got {kind}",
                timing_kind::STEP_WALL
            ));
        }
    }

    for (name, stats) in &report.pipeline_spans {
        let expected_kind = span_timing_kind(name);
        if stats.timing_kind != expected_kind {
            return Err(format!(
                "span {name} must use {expected_kind}, got {}",
                stats.timing_kind
            ));
        }
        percentile_pair_valid(stats.sample_count, stats.p50_ms, stats.p95_ms)?;
    }

    if report.runtime_peak_device_bytes.is_none()
        && report
            .runtime_peak_device_bytes_reason
            .as_deref()
            .unwrap_or("")
            .is_empty()
    {
        return Err("missing runtime_peak_device_bytes requires a reason".into());
    }
    if report.runtime_peak_device_bytes.is_some()
        && report.runtime_peak_device_bytes_reason.as_deref()
            != Some(SAMPLED_BYTES_IN_USE_HIGH_WATER)
    {
        return Err(
            "sampled runtime_peak_device_bytes must reason sampled_bytes_in_use_high_water".into(),
        );
    }

    Ok(())
}

/// Map profiler fields into optimization-report GPU/train/memory slices.
///
/// `gpu_completion_seconds` is always null: we never invent totals via p50×N.
/// Use `gpu_forward_sum_seconds` for the real accepted-sample sum (forward scope
/// only; excludes warmup drops, unsampled steps, and resume-prior work).
pub fn optimization_gpu_fields_from_profiler(report: &GpuProfilerReport) -> OptimizationGpuFields {
    OptimizationGpuFields {
        timestamp_query_available: Some(report.timestamp_query_available),
        adapter_name: report.adapter.clone(),
        backend: Some(report.backend.clone()),
        driver: report.driver.clone(),
        adapter_unavailable_reason: report.adapter_unavailable_reason.clone(),
        driver_unavailable_reason: report.driver_unavailable_reason.clone(),
        gpu_timing_scope: report.gpu_timing_scope.clone(),
        gpu_completion_seconds: None,
        gpu_forward_sum_seconds: report.gpu_forward_sum_ms.map(|ms| ms / 1000.0),
        gpu_step_p50_ms: report.gpu_step_p50_ms,
        gpu_step_p95_ms: report.gpu_step_p95_ms,
        gpu_step_sample_count: Some(report.sample_count),
        gpu_profiler_unsupported_reason: report.unsupported_reason.clone(),
        profiler_enabled: Some(report.profiler_enabled),
        gpu_timing_enabled: Some(report.gpu_timing_enabled),
        gpu_sample_every: Some(report.gpu_sample_every),
        measurement_success: Some(report.measurement_success),
        rejected_timing_samples: Some(report.rejected_timing_samples),
        profile_start_failures: Some(report.profile_start_failures),
        profile_end_failures: Some(report.profile_end_failures),
        profile_resolve_failures: Some(report.profile_resolve_failures),
        dropped_profile_samples: Some(report.dropped_profile_samples),
        pipeline_spans: report.pipeline_spans.clone(),
        peak_device_bytes: report.runtime_peak_device_bytes,
        peak_device_bytes_reason: report.runtime_peak_device_bytes_reason.clone(),
        workspace_scope: Some(report.workspace_scope.clone()),
        workspace_current_bytes: Some(report.workspace_current_bytes),
        workspace_peak_bytes: Some(report.workspace_peak_bytes),
        workspace_growth_count: Some(report.workspace_growth_count),
        fresh_step_allocations: Some(report.fresh_step_allocations),
    }
}

/// GPU-related slices for optimization JSON mapping.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OptimizationGpuFields {
    pub timestamp_query_available: Option<bool>,
    pub adapter_name: Option<String>,
    pub backend: Option<String>,
    pub driver: Option<String>,
    pub adapter_unavailable_reason: Option<String>,
    pub driver_unavailable_reason: Option<String>,
    pub gpu_timing_scope: Option<String>,
    pub gpu_completion_seconds: Option<f64>,
    pub gpu_forward_sum_seconds: Option<f64>,
    pub gpu_step_p50_ms: Option<f64>,
    pub gpu_step_p95_ms: Option<f64>,
    pub gpu_step_sample_count: Option<u64>,
    pub gpu_profiler_unsupported_reason: Option<String>,
    pub profiler_enabled: Option<bool>,
    pub gpu_timing_enabled: Option<bool>,
    pub gpu_sample_every: Option<usize>,
    pub measurement_success: Option<bool>,
    pub rejected_timing_samples: Option<u64>,
    pub profile_start_failures: Option<u64>,
    pub profile_end_failures: Option<u64>,
    pub profile_resolve_failures: Option<u64>,
    pub dropped_profile_samples: Option<u64>,
    pub pipeline_spans: BTreeMap<String, PipelineSpanStats>,
    pub peak_device_bytes: Option<u64>,
    pub peak_device_bytes_reason: Option<String>,
    pub workspace_scope: Option<String>,
    pub workspace_current_bytes: Option<u64>,
    pub workspace_peak_bytes: Option<u64>,
    pub workspace_growth_count: Option<u64>,
    pub fresh_step_allocations: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn supported_probe() -> GpuEnvironmentProbe {
        GpuEnvironmentProbe {
            backend: "Metal".into(),
            adapter: Some("Test GPU".into()),
            driver: Some("TestDriver".into()),
            timestamp_query_available: true,
            timing_method_device: true,
            ..Default::default()
        }
    }

    #[test]
    fn percentile_p95_ge_p50_and_sample_count_matches() {
        let mut collector = PipelineTimingCollector::new(supported_probe());
        // warmup + 4 samples
        for ms in [1.0, 10.0, 20.0, 30.0, 40.0] {
            collector.record_gpu_step_ms(ms);
            collector.record_cpu_step(Duration::from_millis(ms as u64));
        }
        collector.observe_workspace(100, 2, 1);
        collector.observe_workspace(120, 3, 0);
        collector.observe_runtime_device_bytes(Some(4096));

        let report = collector.build_report();
        assert!(report.supported);
        assert_eq!(report.sample_count, 4);
        assert_eq!(
            report.gpu_timing_scope.as_deref(),
            Some(GPU_TIMING_SCOPE_FORWARD)
        );
        assert_eq!(report.workspace_scope, WORKSPACE_SCOPE_PREFIX_SUM);
        // Warmup drops 1.0; remaining [10,20,30,40]. Rank for p50 is round(0.5*3)=2 → 30.
        assert_eq!(report.gpu_step_p50_ms, Some(30.0));
        assert_eq!(report.gpu_step_p95_ms, Some(40.0));
        assert_eq!(report.gpu_forward_sum_ms, Some(100.0));
        assert!(report.measurement_success);
        assert_eq!(
            report.runtime_peak_device_bytes_reason.as_deref(),
            Some(SAMPLED_BYTES_IN_USE_HIGH_WATER)
        );
        assert_report_self_consistent(&report).expect("consistent");
    }

    #[test]
    fn forward_sum_differs_from_p50_times_sample_count_for_uneven_samples() {
        let mut collector = PipelineTimingCollector::new(supported_probe());
        // warmup + uneven: 1, then 10, 100, 1000
        for ms in [1.0, 10.0, 100.0, 1000.0] {
            collector.record_gpu_step_ms(ms);
        }
        let report = collector.build_report();
        assert_eq!(report.sample_count, 3);
        let sum = report.gpu_forward_sum_ms.expect("sum");
        let p50 = report.gpu_step_p50_ms.expect("p50");
        let invented = p50 * report.sample_count as f64;
        assert!(
            (sum - invented).abs() > 1.0,
            "sum={sum} must not equal p50*N={invented}"
        );
        let fields = optimization_gpu_fields_from_profiler(&report);
        assert!(fields.gpu_completion_seconds.is_none());
        assert_eq!(fields.gpu_forward_sum_seconds, Some(sum / 1000.0));
        assert_eq!(fields.gpu_timing_scope.as_deref(), Some("forward"));
    }

    #[test]
    fn rejects_nan_inf_and_negative_timings() {
        let mut collector = PipelineTimingCollector::new(supported_probe());
        collector.record_gpu_step_ms(f64::NAN);
        collector.record_gpu_step_ms(f64::INFINITY);
        collector.record_gpu_step_ms(-1.0);
        collector.record_span_ms(span::FORWARD, f64::NAN);
        collector.record_span_ms(span::DECODE, -0.5);
        collector.record_cpu_step(Duration::from_millis(5));
        // One valid GPU sample that will be dropped by warmup alone.
        collector.record_gpu_step_ms(12.0);
        assert_eq!(collector.rejected_timing_samples(), 5);
        let report = collector.build_report();
        assert_eq!(report.sample_count, 0);
        assert!(report.gpu_step_p50_ms.is_none());
        assert!(report.gpu_forward_sum_ms.is_none());
        assert_eq!(report.rejected_timing_samples, 5);
        assert_report_self_consistent(&report).expect("empty after rejects/warmup");
    }

    #[test]
    fn empty_after_warmup_only_yields_null_percentiles() {
        let mut collector = PipelineTimingCollector::new(supported_probe());
        collector.record_gpu_step_ms(5.0); // warmup only
        collector.record_cpu_step(Duration::from_millis(3));
        let report = collector.build_report();
        assert_eq!(report.sample_count, 0);
        assert!(report.gpu_step_p50_ms.is_none());
        assert!(report.gpu_step_p95_ms.is_none());
        assert!(report.gpu_forward_sum_ms.is_none());
        assert!(!report.measurement_success);
        assert!(report.cpu_step_p50_ms.is_none());
        assert_report_self_consistent(&report).expect("warmup-empty consistent");
    }

    #[test]
    fn unsupported_nulls_gpu_percentiles_and_sets_stable_reason() {
        let mut collector = PipelineTimingCollector::new(GpuEnvironmentProbe {
            backend: "Metal".into(),
            adapter: Some("Apple GPU".into()),
            driver: Some("Metal".into()),
            timestamp_query_available: false,
            timing_method_device: false,
            unsupported_reason: Some(UNSUPPORTED_TIMESTAMP_QUERY_UNAVAILABLE.into()),
            ..Default::default()
        });
        // Attempted GPU samples must be ignored when unsupported.
        collector.record_gpu_step_ms(12.0);
        collector.record_cpu_step(Duration::from_millis(18));
        collector.record_cpu_step(Duration::from_millis(22));
        collector.record_span(span::FORWARD, Duration::from_millis(5));
        collector.record_span(span::FORWARD, Duration::from_millis(7));

        let report = collector.build_report();
        assert!(!report.supported);
        assert!(!report.timestamp_query_available);
        assert_eq!(
            report.unsupported_reason.as_deref(),
            Some(UNSUPPORTED_TIMESTAMP_QUERY_UNAVAILABLE)
        );
        assert!(report.gpu_step_p50_ms.is_none());
        assert!(report.gpu_step_p95_ms.is_none());
        assert_eq!(report.sample_count, 0);
        assert_eq!(
            report.cpu_timing_kind.as_deref(),
            Some(timing_kind::STEP_WALL)
        );
        assert_eq!(
            report
                .pipeline_spans
                .get(span::FORWARD)
                .map(|s| s.timing_kind.as_str()),
            Some(timing_kind::SYNCHRONIZED_BOUNDARY)
        );
        assert_eq!(
            report
                .pipeline_spans
                .get(span::DECODE)
                .map(|s| s.timing_kind.as_str()),
            Some(timing_kind::WORKER_WALL)
        );
        assert_eq!(
            report
                .pipeline_spans
                .get(span::FRAME_WAIT)
                .map(|s| s.timing_kind.as_str()),
            Some(timing_kind::HOST_WAIT)
        );
        assert!(report.cpu_step_p50_ms.is_some());
        assert_report_self_consistent(&report).expect("consistent");
    }

    #[test]
    fn gpu_profiler_report_json_round_trip_fixture() {
        let report = GpuProfilerReport {
            supported: false,
            unsupported_reason: Some(UNSUPPORTED_TIMESTAMP_QUERY_UNAVAILABLE.into()),
            backend: "Metal".into(),
            adapter: Some("Apple M-series".into()),
            driver: Some("Metal".into()),
            adapter_unavailable_reason: None,
            driver_unavailable_reason: None,
            timestamp_query_available: false,
            profiler_enabled: true,
            gpu_timing_enabled: false,
            measurement_success: false,
            gpu_sample_every: 20,
            gpu_timing_scope: Some(GPU_TIMING_SCOPE_FORWARD.into()),
            gpu_forward_sum_ms: None,
            gpu_step_p50_ms: None,
            gpu_step_p95_ms: None,
            sample_count: 0,
            rejected_timing_samples: 0,
            profile_start_failures: 0,
            profile_end_failures: 0,
            profile_resolve_failures: 0,
            dropped_profile_samples: 0,
            workspace_scope: WORKSPACE_SCOPE_PREFIX_SUM.into(),
            workspace_current_bytes: 2048,
            workspace_peak_bytes: 4096,
            workspace_growth_count: 2,
            fresh_step_allocations: 3,
            runtime_peak_device_bytes: None,
            runtime_peak_device_bytes_reason: Some(UNSUPPORTED_RUNTIME_PEAK_UNAVAILABLE.into()),
            cpu_step_p50_ms: Some(18.0),
            cpu_step_p95_ms: Some(22.0),
            cpu_timing_kind: Some(timing_kind::STEP_WALL.into()),
            pipeline_spans: BTreeMap::from([
                (
                    span::FORWARD.to_string(),
                    PipelineSpanStats {
                        sample_count: 2,
                        p50_ms: Some(5.0),
                        p95_ms: Some(7.0),
                        timing_kind: timing_kind::SYNCHRONIZED_BOUNDARY.into(),
                    },
                ),
                (
                    span::STEP_CPU.to_string(),
                    PipelineSpanStats {
                        sample_count: 2,
                        p50_ms: Some(18.0),
                        p95_ms: Some(22.0),
                        timing_kind: timing_kind::STEP_WALL.into(),
                    },
                ),
            ]),
        };
        let json = serde_json::to_string_pretty(&report).expect("serialize");
        assert!(json.contains("\"gpu_step_p50_ms\": null"));
        assert!(json.contains("\"gpu_forward_sum_ms\": null"));
        assert!(json.contains("\"gpu_timing_scope\": \"forward\""));
        assert!(json.contains("timestamp_query_unavailable"));
        assert!(json.contains(timing_kind::SYNCHRONIZED_BOUNDARY));
        assert!(json.contains("prefix_sum"));
        let decoded: GpuProfilerReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded, report);
        assert_report_self_consistent(&decoded).expect("fixture consistent");
    }

    #[test]
    fn workspace_growth_and_fresh_allocations_do_not_regress() {
        let mut collector = PipelineTimingCollector::new(GpuEnvironmentProbe::default());
        collector.observe_workspace(100, 1, 2);
        collector.observe_workspace(100, 1, 0);
        collector.observe_workspace(80, 1, 0);
        assert_eq!(collector.workspace_peak_bytes, 100);
        assert_eq!(collector.workspace_growth_count, 1);
        assert_eq!(collector.fresh_step_allocations, 2);
        // Later smaller current must not shrink peak.
        assert_eq!(collector.workspace_current_bytes, 80);
    }

    #[test]
    fn should_profile_gpu_cadence_first_and_multiples() {
        assert!(!should_profile_gpu(0, 20, None));
        assert!(should_profile_gpu(1, 20, None));
        assert!(!should_profile_gpu(2, 20, None));
        assert!(should_profile_gpu(20, 20, None));
        assert!(should_profile_gpu(40, 20, None));
        assert!(!should_profile_gpu(41, 20, None));
        // Final iteration when total is known.
        assert!(should_profile_gpu(17, 20, Some(17)));
        assert!(!should_profile_gpu(16, 20, Some(17)));
        // sample_every < 1 treated as 1.
        assert!(should_profile_gpu(3, 0, None));
    }

    #[test]
    fn recover_profile_result_runs_work_at_most_once_on_start_fail() {
        use std::cell::Cell;
        let runs = Cell::new(0);
        let mut work_slot = Some(|| {
            runs.set(runs.get() + 1);
            42
        });
        let mut output_slot = None;
        // Start-fail: work still in slot, no output yet.
        let (out, ms) = recover_profile_result(&mut work_slot, &mut output_slot).expect("recover");
        assert_eq!(out, 42);
        assert!(ms.is_none());
        assert_eq!(runs.get(), 1);
        assert!(work_slot.is_none());
        assert!(output_slot.is_none());
        // Second recovery must fail rather than re-run.
        let err = recover_profile_result(&mut work_slot, &mut output_slot).unwrap_err();
        assert!(err.contains("impossible"));
        assert_eq!(runs.get(), 1);
    }

    #[test]
    fn recover_profile_result_prefers_stored_output_on_end_fail() {
        use std::cell::Cell;
        let runs = Cell::new(0);
        let mut work_slot = Some(|| {
            runs.set(runs.get() + 1);
            7
        });
        let mut output_slot = None;
        // Simulate profile closure: take work once, store output.
        if let Some(work) = work_slot.take() {
            output_slot = Some(work());
        }
        assert_eq!(runs.get(), 1);
        let (out, ms) = recover_profile_result(&mut work_slot, &mut output_slot).expect("recover");
        assert_eq!(out, 7);
        assert!(ms.is_none());
        assert_eq!(runs.get(), 1, "end-fail must not re-run work");
        assert!(work_slot.is_none());
        assert!(output_slot.is_none());
    }

    #[test]
    fn build_report_records_profiler_mode() {
        let mut collector = PipelineTimingCollector::new(GpuEnvironmentProbe::default());
        collector.set_profiler_mode(true, true, 20);
        let report = collector.build_report();
        assert!(report.profiler_enabled);
        assert!(report.gpu_timing_enabled);
        assert_eq!(report.gpu_sample_every, 20);
        assert!(!report.measurement_success);
    }

    #[test]
    fn profiler_disabled_records_no_cpu_span_samples() {
        let mut collector = PipelineTimingCollector::new(supported_probe());
        collector.set_profiler_mode(false, false, 20);
        collector.record_span(span::FORWARD, Duration::from_millis(5));
        collector.record_span(span::DECODE, Duration::from_millis(3));
        collector.record_cpu_step(Duration::from_millis(12));
        collector.record_span_ms(span::UPLOAD, 1.5);
        // Invalid samples must not bump rejects when disabled.
        collector.record_span_ms(span::FORWARD, f64::NAN);
        assert_eq!(collector.rejected_timing_samples(), 0);
        let report = collector.build_report();
        assert!(!report.profiler_enabled);
        assert!(report.cpu_step_p50_ms.is_none());
        assert!(report.cpu_step_p95_ms.is_none());
        for name in span::ALL {
            let stats = report.pipeline_spans.get(*name).expect("span");
            assert_eq!(
                stats.sample_count, 0,
                "{name} must stay empty when profiler disabled"
            );
            assert!(stats.p50_ms.is_none());
            assert!(stats.p95_ms.is_none());
        }
        assert_report_self_consistent(&report).expect("disabled consistent");
    }

    #[test]
    fn hand_built_nonzero_samples_without_percentiles_fail_validator() {
        let mut report = GpuProfilerReport {
            supported: true,
            unsupported_reason: None,
            backend: "Metal".into(),
            adapter: Some("Test".into()),
            driver: Some("Test".into()),
            timestamp_query_available: true,
            profiler_enabled: true,
            gpu_timing_enabled: true,
            measurement_success: true,
            gpu_sample_every: 1,
            gpu_timing_scope: Some(GPU_TIMING_SCOPE_FORWARD.into()),
            gpu_forward_sum_ms: Some(10.0),
            gpu_step_p50_ms: None,
            gpu_step_p95_ms: None,
            sample_count: 1,
            workspace_scope: WORKSPACE_SCOPE_PREFIX_SUM.into(),
            runtime_peak_device_bytes_reason: Some(UNSUPPORTED_RUNTIME_PEAK_UNAVAILABLE.into()),
            cpu_timing_kind: Some(timing_kind::STEP_WALL.into()),
            pipeline_spans: BTreeMap::from([(
                span::UPLOAD.to_string(),
                PipelineSpanStats {
                    sample_count: 1,
                    p50_ms: None,
                    p95_ms: None,
                    timing_kind: timing_kind::CPU_SUBMIT.into(),
                },
            )]),
            ..Default::default()
        };
        let err = assert_report_self_consistent(&report).expect_err("must reject");
        assert!(
            err.contains("sample_count") || err.contains("percentiles"),
            "unexpected err: {err}"
        );

        // JSON round-trip of the bad report must still fail validation.
        let json = serde_json::to_string(&report).expect("serialize");
        let decoded: GpuProfilerReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded.sample_count, 1);
        assert!(decoded.gpu_step_p50_ms.is_none());
        assert!(assert_report_self_consistent(&decoded).is_err());

        // Repair GPU percentiles but leave upload span broken.
        report.gpu_step_p50_ms = Some(10.0);
        report.gpu_step_p95_ms = Some(10.0);
        let err = assert_report_self_consistent(&report).expect_err("upload still bad");
        assert!(err.contains("upload") || err.contains("sample_count"));
    }

    #[test]
    fn classify_profile_recovery_matches_recover_core() {
        let mut work_slot = Some(|| 1);
        let mut output_slot = None::<i32>;
        assert_eq!(
            classify_profile_recovery(&work_slot, &output_slot),
            ProfileRecoveryKind::StartFailed
        );
        let (out, _) = recover_profile_result(&mut work_slot, &mut output_slot).expect("start");
        assert_eq!(out, 1);
        assert_eq!(
            classify_profile_recovery(&work_slot, &output_slot),
            ProfileRecoveryKind::Impossible
        );

        let mut work_slot = Some(|| 2);
        let mut output_slot = Some(9);
        assert_eq!(
            classify_profile_recovery(&work_slot, &output_slot),
            ProfileRecoveryKind::EndFailed
        );
        let (out, _) = recover_profile_result(&mut work_slot, &mut output_slot).expect("end");
        assert_eq!(out, 9);
        assert!(work_slot.is_some(), "end-fail must not consume work");
    }

    #[test]
    fn timestamp_capability_independent_of_enablement() {
        let mut probe = supported_probe();
        probe.timestamp_query_available = true;
        probe.timing_method_device = true;
        let mut collector = PipelineTimingCollector::new(probe);
        collector.set_profiler_mode(true, false, 20);
        let report = collector.build_report();
        assert!(report.timestamp_query_available);
        assert!(report.supported);
        assert!(!report.gpu_timing_enabled);
        assert!(!report.measurement_success);
        let fields = optimization_gpu_fields_from_profiler(&report);
        assert_eq!(fields.timestamp_query_available, Some(true));
        assert_eq!(fields.profiler_enabled, Some(true));
        assert_eq!(fields.gpu_timing_enabled, Some(false));
        assert!(!fields.pipeline_spans.is_empty());
    }

    #[test]
    fn default_device_probe_does_not_fabricate_adapter_metadata() {
        use crate::training::engine::GsDevice;
        let probe = probe_training_device(&GsDevice::default());
        assert!(probe.adapter.is_none());
        assert!(probe.driver.is_none());
        assert_eq!(
            probe.adapter_unavailable_reason.as_deref(),
            Some(DEFAULT_DEVICE_ADAPTER_METADATA_UNAVAILABLE)
        );
        assert_eq!(
            probe.driver_unavailable_reason.as_deref(),
            Some(DEFAULT_DEVICE_ADAPTER_METADATA_UNAVAILABLE)
        );
        // TimingMethod still comes from the live client.
        assert!(!probe.backend.is_empty());
    }

    #[test]
    fn cpu_span_timer_records_span() {
        let mut collector = PipelineTimingCollector::new(supported_probe());
        let timer = CpuSpanTimer::start(span::UPLOAD);
        std::thread::sleep(Duration::from_millis(1));
        timer.finish(&mut collector);
        let report = collector.build_report();
        // Warmup drops the single sample.
        assert_eq!(
            report
                .pipeline_spans
                .get(span::UPLOAD)
                .map(|s| s.sample_count),
            Some(0)
        );
        collector.record_span(span::UPLOAD, Duration::from_millis(2));
        let report = collector.build_report();
        assert_eq!(
            report
                .pipeline_spans
                .get(span::UPLOAD)
                .map(|s| s.sample_count),
            Some(1)
        );
    }
}
