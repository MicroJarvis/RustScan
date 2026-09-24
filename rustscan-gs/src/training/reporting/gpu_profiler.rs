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
    /// Host Instant around CPU-side submit / orchestration (no GPU resolve wait).
    pub const CPU_SUBMIT: &str = "cpu_submit";
    /// Host Instant that brackets GPU timestamp queries **and** resolve wait.
    ///
    /// Only for iterations that actually sample device timestamps. Do not mix
    /// with unsampled forward submit walls in the same percentile series.
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
    /// Forward host wall on GPU-sampled iterations (includes timestamp resolve wait).
    pub const FORWARD_GPU_SAMPLED: &str = "forward_gpu_sampled";
    /// Forward host wall on unsampled iterations (CPU submit / orchestration only).
    pub const FORWARD_CPU_SUBMIT: &str = "forward_cpu_submit";
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
        FORWARD_GPU_SAMPLED,
        FORWARD_CPU_SUBMIT,
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
        span::FORWARD_GPU_SAMPLED => timing_kind::SYNCHRONIZED_BOUNDARY,
        span::FORWARD_CPU_SUBMIT => timing_kind::CPU_SUBMIT,
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

    /// Record a GPU forward completion sample only when profiling + GPU timing
    /// are enabled and device timestamps are supported.
    pub fn record_gpu_step_ms(&mut self, ms: f64) {
        if !self.profiler_enabled || !self.gpu_timing_enabled {
            return;
        }
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
        note_profiling_instant_created();
        Self {
            name,
            started: Instant::now(),
        }
    }

    /// Start only when profiling is enabled — skips Instant creation when off.
    pub fn start_enabled(enabled: bool, name: &'static str) -> Option<Self> {
        enabled.then(|| Self::start(name))
    }

    pub fn finish(self, collector: &mut PipelineTimingCollector) {
        collector.record_span(self.name, self.started.elapsed());
    }
}

/// Optional Instant for profiler-only host walls. When `enabled` is false, no
/// Instant is created (verified by [`take_profiling_instant_creations`] in tests).
#[inline]
pub fn profiling_instant(enabled: bool) -> Option<Instant> {
    if !enabled {
        return None;
    }
    note_profiling_instant_created();
    Some(Instant::now())
}

#[cfg(test)]
thread_local! {
    static PROFILING_INSTANT_CREATIONS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[inline]
fn note_profiling_instant_created() {
    #[cfg(test)]
    PROFILING_INSTANT_CREATIONS.with(|c| c.set(c.get().saturating_add(1)));
}

/// Test helper: return and clear profiling Instant creation count.
#[cfg(test)]
pub fn take_profiling_instant_creations() -> u64 {
    PROFILING_INSTANT_CREATIONS.with(|c| c.replace(0))
}

#[cfg(test)]
thread_local! {
    static DEBUG_PROFILE_READBACKS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Count a debug-only `into_scalar_async` profiling readback (test observability).
#[inline]
pub fn note_debug_profile_readback() {
    #[cfg(test)]
    DEBUG_PROFILE_READBACKS.with(|c| c.set(c.get().saturating_add(1)));
}

/// Test helper: return and clear debug profile readback count.
#[cfg(test)]
pub fn take_debug_profile_readbacks() -> u64 {
    DEBUG_PROFILE_READBACKS.with(|c| c.replace(0))
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

/// Measurement-only failure while resolving device timestamps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileMeasureError {
    /// CubeCL timestamp map/resolve panicked (or equivalent hard failure).
    ResolvePanic { message: String },
}

/// Injected faults for production-shared profile completion paths (tests + drills).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProfileFaultStage {
    #[default]
    None,
    /// Pretend profile never started; run work once unprofiled.
    Start,
    /// Pretend profile end failed after work; keep output, drop timing.
    End,
    /// Pretend resolve panicked while the device remains usable.
    Resolve,
}

impl ProfileFaultStage {
    /// Injectable stages (excludes [`Self::None`]). Referenced so the API surface
    /// stays live for production-shared fault drills without silencing dead_code.
    pub const INJECTABLE: &[Self] = &[Self::Start, Self::End, Self::Resolve];
}

fn panic_payload_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(msg) = payload.downcast_ref::<&'static str>() {
        (*msg).to_string()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "profile resolve panicked".to_string()
    }
}

/// Whether the training device can still execute GPU work after a measure fault.
///
/// Cached `properties()` / host allocator stats are **not** sufficient. This probe
/// runs a tiny tensor write/readback round-trip on the training device; panics or
/// I/O failures mean the device is not usable for continued training. When health
/// cannot be confirmed, returns `false` so callers surface [`crate::TrainingError::Gpu`].
pub fn device_measurement_path_healthy(device: &crate::training::engine::GsDevice) -> bool {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    catch_unwind(AssertUnwindSafe(|| probe_device_executes_gpu_work(device))).unwrap_or(false)
}

fn probe_device_executes_gpu_work(device: &crate::training::engine::GsDevice) -> bool {
    use burn::prelude::*;
    use burn::tensor::TensorData;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use crate::training::engine::GsBackendBase;

    // Force a device round-trip: host→GPU write then GPU→host readback.
    let round_trip = catch_unwind(AssertUnwindSafe(|| {
        let tensor = Tensor::<GsBackendBase, 1>::from_data(
            TensorData::new(vec![0.125_f32, 0.25, 0.5], [3]),
            device,
        );
        let data = burn_cubecl::cubecl::future::block_on(tensor.into_data_async())
            .map_err(|error| format!("readback failed: {error}"))?;
        let values = data
            .into_vec::<f32>()
            .map_err(|error| format!("typed readback failed: {error:?}"))?;
        if values.len() != 3 {
            return Err(format!("unexpected readback length {}", values.len()));
        }
        Ok::<_, String>(values)
    }));
    match round_trip {
        Ok(Ok(values)) => {
            (values[0] - 0.125).abs() < 1e-5
                && (values[1] - 0.25).abs() < 1e-5
                && (values[2] - 0.5).abs() < 1e-5
        }
        Ok(Err(_)) | Err(_) => false,
    }
}

/// Resolve a cubecl profile duration into GPU milliseconds only for Device timing.
///
/// CubeCL's wgpu path may `expect`/panic inside the resolve future on map-buffer
/// failure. This wrapper catches that panic and returns
/// [`ProfileMeasureError::ResolvePanic`] instead of aborting the process.
pub async fn resolve_device_gpu_ms(
    profile: burn_cubecl::cubecl::profile::ProfileDuration,
) -> Result<Option<f64>, ProfileMeasureError> {
    use burn_cubecl::cubecl::profile::TimingMethod;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    let timing_method = profile.timing_method();
    // Drive resolve via block_on so catch_unwind can wrap the panic-prone path.
    let resolved = catch_unwind(AssertUnwindSafe(|| {
        burn_cubecl::cubecl::future::block_on(profile.resolve())
    }));
    match resolved {
        Ok(ticks) => {
            if timing_method != TimingMethod::Device {
                // Drained non-device timing; not a GPU sample.
                Ok(None)
            } else {
                Ok(Some(duration_millis(ticks.duration())))
            }
        }
        Err(payload) => Err(ProfileMeasureError::ResolvePanic {
            message: panic_payload_message(payload),
        }),
    }
}

/// Finish a profiled step after work produced `output` and resolve completed
/// (or failed). Shared by the live GPU path and fault-injection tests.
///
/// On resolve panic: if [`device_measurement_path_healthy`] confirms the device,
/// drop the sample and continue; otherwise return [`crate::TrainingError::Gpu`].
pub fn finish_profiled_output<T>(
    device: &crate::training::engine::GsDevice,
    output: T,
    resolve: Result<Option<f64>, ProfileMeasureError>,
) -> Result<(T, Option<f64>, ProfileAttemptStats), crate::TrainingError> {
    let mut stats = ProfileAttemptStats::default();
    match resolve {
        Ok(Some(ms)) => Ok((output, Some(ms), stats)),
        Ok(None) => {
            stats.resolve_failures = 1;
            stats.dropped_samples = 1;
            Ok((output, None, stats))
        }
        Err(ProfileMeasureError::ResolvePanic { message }) => {
            if !device_measurement_path_healthy(device) {
                return Err(crate::TrainingError::Gpu(format!(
                    "GPU unavailable after profile resolve failure: {message}"
                )));
            }
            stats.resolve_failures = 1;
            stats.dropped_samples = 1;
            Ok((output, None, stats))
        }
    }
}

/// Complete a start/end profile recovery without re-running already-executed work.
pub async fn complete_profile_recovery<T, F, Fut>(
    kind: ProfileRecoveryKind,
    work_slot: &mut Option<F>,
    output_slot: &mut Option<T>,
) -> Result<(T, Option<f64>, ProfileAttemptStats), crate::TrainingError>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = T>,
{
    let mut stats = ProfileAttemptStats::default();
    match kind {
        ProfileRecoveryKind::EndFailed => {
            stats.end_failures = 1;
            stats.dropped_samples = 1;
            let output = output_slot.take().ok_or_else(|| {
                crate::TrainingError::TrainingFailed(
                    "gpu profile end-fail recovery missing stored output".into(),
                )
            })?;
            Ok((output, None, stats))
        }
        ProfileRecoveryKind::StartFailed => {
            stats.start_failures = 1;
            stats.dropped_samples = 1;
            let work = work_slot.take().ok_or_else(|| {
                crate::TrainingError::TrainingFailed(
                    "gpu profile start-fail recovery missing work".into(),
                )
            })?;
            Ok((work().await, None, stats))
        }
        ProfileRecoveryKind::Impossible => Err(crate::TrainingError::TrainingFailed(
            "gpu profile recovery: work already consumed and no output stored (impossible)".into(),
        )),
    }
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
/// When profiling cannot start or end fails after work ran, `work` still runs
/// at most once. Resolve panics are caught: if the device remains healthy the
/// sample is dropped and training continues; if the device is unusable a
/// [`crate::TrainingError::Gpu`] is returned. Training errors from `work` are
/// not swallowed (work returns `T` directly; callers wrap fallible work).
pub async fn profile_device_gpu_step<F, Fut, T>(
    device: &crate::training::engine::GsDevice,
    work: F,
) -> Result<(T, Option<f64>, ProfileAttemptStats), crate::TrainingError>
where
    F: FnOnce() -> Fut + Send,
    Fut: std::future::Future<Output = T>,
    T: Send + 'static,
{
    profile_device_gpu_step_with_fault(device, work, ProfileFaultStage::None).await
}

/// Production profile path with optional fault injection at start/end/resolve.
pub async fn profile_device_gpu_step_with_fault<F, Fut, T>(
    device: &crate::training::engine::GsDevice,
    work: F,
    fault: ProfileFaultStage,
) -> Result<(T, Option<f64>, ProfileAttemptStats), crate::TrainingError>
where
    F: FnOnce() -> Fut + Send,
    Fut: std::future::Future<Output = T>,
    T: Send + 'static,
{
    use burn_cubecl::cubecl::profile::TimingMethod;
    use burn_cubecl::cubecl::Runtime;
    use burn_wgpu::WgpuRuntime;
    use std::sync::{Arc, Mutex};

    match fault {
        ProfileFaultStage::Start => {
            let mut work_slot = Some(work);
            let mut output_slot = None;
            return complete_profile_recovery(
                ProfileRecoveryKind::StartFailed,
                &mut work_slot,
                &mut output_slot,
            )
            .await;
        }
        ProfileFaultStage::End => {
            let output = work().await;
            let mut work_slot = None::<F>;
            let mut output_slot = Some(output);
            return complete_profile_recovery(
                ProfileRecoveryKind::EndFailed,
                &mut work_slot,
                &mut output_slot,
            )
            .await;
        }
        ProfileFaultStage::Resolve => {
            let output = work().await;
            return finish_profiled_output(
                device,
                output,
                Err(ProfileMeasureError::ResolvePanic {
                    message: "injected resolve failure".into(),
                }),
            );
        }
        ProfileFaultStage::None => {
            // Keep INJECTABLE referenced in production code so fault stages stay live.
            debug_assert_eq!(ProfileFaultStage::INJECTABLE.len(), 3);
        }
    }

    let client = WgpuRuntime::client(device);
    if client.properties().timing_method != TimingMethod::Device {
        return Ok((work().await, None, ProfileAttemptStats::default()));
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
                    let resolve = resolve_device_gpu_ms(profile).await;
                    finish_profiled_output(device, output, resolve)
                }
                None => Err(crate::TrainingError::TrainingFailed(
                    "gpu profile Ok without output: work closure did not store a result".into(),
                )),
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
            // Take slots before await so MutexGuards are not held across .await.
            let (mut work_opt, mut output_opt) = {
                let mut work_guard = work_slot
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let mut output_guard = output_slot
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                (work_guard.take(), output_guard.take())
            };
            complete_profile_recovery(kind, &mut work_opt, &mut output_opt).await
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
        if report.measurement_success {
            return Err("unsupported report must set measurement_success=false".into());
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
        collector.set_profiler_mode(true, true, 1);
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
        collector.set_profiler_mode(true, true, 1);
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
        collector.set_profiler_mode(true, true, 1);
        collector.record_gpu_step_ms(f64::NAN);
        collector.record_gpu_step_ms(f64::INFINITY);
        collector.record_gpu_step_ms(-1.0);
        collector.record_span_ms(span::FORWARD_CPU_SUBMIT, f64::NAN);
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
        collector.set_profiler_mode(true, true, 1);
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
        collector.record_span(span::FORWARD_CPU_SUBMIT, Duration::from_millis(5));
        collector.record_span(span::FORWARD_CPU_SUBMIT, Duration::from_millis(7));

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
                .get(span::FORWARD_CPU_SUBMIT)
                .map(|s| s.timing_kind.as_str()),
            Some(timing_kind::CPU_SUBMIT)
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
                    span::FORWARD_CPU_SUBMIT.to_string(),
                    PipelineSpanStats {
                        sample_count: 2,
                        p50_ms: Some(5.0),
                        p95_ms: Some(7.0),
                        timing_kind: timing_kind::CPU_SUBMIT.into(),
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
        assert!(json.contains(timing_kind::CPU_SUBMIT));
        assert!(json.contains(span::FORWARD_CPU_SUBMIT));
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
        collector.record_span(span::FORWARD_CPU_SUBMIT, Duration::from_millis(5));
        collector.record_span(span::DECODE, Duration::from_millis(3));
        collector.record_cpu_step(Duration::from_millis(12));
        collector.record_span_ms(span::UPLOAD, 1.5);
        // Invalid samples must not bump rejects when disabled.
        collector.record_span_ms(span::FORWARD_CPU_SUBMIT, f64::NAN);
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

    #[tokio::test(flavor = "current_thread")]
    async fn production_fault_path_start_end_resolve_work_at_most_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let device = crate::training::engine::GsDevice::default();
        let runs = AtomicUsize::new(0);

        let (out, ms, stats) = profile_device_gpu_step_with_fault(
            &device,
            || async {
                runs.fetch_add(1, Ordering::SeqCst);
                11_i32
            },
            ProfileFaultStage::Start,
        )
        .await
        .expect("start fault");
        assert_eq!(out, 11);
        assert!(ms.is_none());
        assert_eq!(stats.start_failures, 1);
        assert_eq!(stats.dropped_samples, 1);
        assert_eq!(runs.load(Ordering::SeqCst), 1);

        let (out, ms, stats) = profile_device_gpu_step_with_fault(
            &device,
            || async {
                runs.fetch_add(1, Ordering::SeqCst);
                22_i32
            },
            ProfileFaultStage::End,
        )
        .await
        .expect("end fault");
        assert_eq!(out, 22);
        assert!(ms.is_none());
        assert_eq!(stats.end_failures, 1);
        assert_eq!(runs.load(Ordering::SeqCst), 2);

        let (out, ms, stats) = profile_device_gpu_step_with_fault(
            &device,
            || async {
                runs.fetch_add(1, Ordering::SeqCst);
                33_i32
            },
            ProfileFaultStage::Resolve,
        )
        .await
        .expect("resolve fault keeps training");
        assert_eq!(out, 33);
        assert!(ms.is_none());
        assert_eq!(stats.resolve_failures, 1);
        assert_eq!(stats.dropped_samples, 1);
        assert_eq!(runs.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn finish_profiled_output_resolve_panic_keeps_prior_success_flag_semantics() {
        let device = crate::training::engine::GsDevice::default();
        let mut collector = PipelineTimingCollector::new(supported_probe());
        collector.set_profiler_mode(true, true, 1);
        collector.record_gpu_step_ms(10.0); // warmup
        collector.record_gpu_step_ms(12.0); // accepted
        let (out, ms, stats) = finish_profiled_output(
            &device,
            7_u32,
            Err(ProfileMeasureError::ResolvePanic {
                message: "map failed".into(),
            }),
        )
        .expect("healthy device drops sample");
        assert_eq!(out, 7);
        assert!(ms.is_none());
        assert_eq!(stats.resolve_failures, 1);
        collector.record_profile_attempt_stats(&stats);
        let report = collector.build_report();
        assert!(report.measurement_success, "prior accepted samples remain");
        assert_eq!(report.sample_count, 1);
        assert_eq!(report.profile_resolve_failures, 1);
        assert_eq!(report.dropped_profile_samples, 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn destroyed_isolated_device_fails_health_without_force_flag() {
        // Run the destructive probe in an isolated child process so cubecl/wgpu
        // teardown after `Device::destroy` cannot abort the parent test binary.
        if std::env::var_os("RUSTGS_DESTROY_DEVICE_WORKER").is_none() {
            let exe = std::env::current_exe().expect("current test exe");
            let output = std::process::Command::new(exe)
                .env("RUSTGS_DESTROY_DEVICE_WORKER", "1")
                .env("RUST_BACKTRACE", "0")
                .args([
                    "--exact",
                    "training::reporting::gpu_profiler::tests::destroyed_isolated_device_fails_health_without_force_flag",
                    "--nocapture",
                ])
                .output()
                .expect("spawn destroyed-device worker");
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                output.status.success(),
                "destroyed-device worker failed status={:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
                output.status
            );
            assert!(
                stdout.contains("DESTROYED_DEVICE_WORKER_OK"),
                "worker missing success marker\nstdout:\n{stdout}\nstderr:\n{stderr}"
            );
            return;
        }

        use burn_wgpu::{init_device, RuntimeOptions, WgpuSetup};
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Build an isolated wgpu device owned by this worker (not the shared default).
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("adapter for destroyed-device probe");
        let (raw_device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("wgpu device");
        let backend = adapter.get_info().backend;
        let setup = WgpuSetup {
            instance,
            adapter,
            device: raw_device.clone(),
            queue,
            backend,
        };
        let gs_device = init_device(
            setup,
            RuntimeOptions {
                memory_config: burn_wgpu::MemoryConfiguration::ExclusivePages,
                ..RuntimeOptions::default()
            },
        );

        assert!(
            device_measurement_path_healthy(&gs_device),
            "fresh isolated device must pass the live GPU probe"
        );

        let runs = AtomicUsize::new(0);
        let (out, ms, stats) = finish_profiled_output(
            &gs_device,
            {
                runs.fetch_add(1, Ordering::SeqCst);
                42_i32
            },
            Err(ProfileMeasureError::ResolvePanic {
                message: "injected resolve on healthy device".into(),
            }),
        )
        .expect("healthy device: measurement-only failure continues");
        assert_eq!(out, 42);
        assert!(ms.is_none());
        assert_eq!(stats.resolve_failures, 1);
        assert_eq!(stats.dropped_samples, 1);
        assert_eq!(runs.load(Ordering::SeqCst), 1, "work must not re-run");

        raw_device.destroy();
        let _ = raw_device.poll(wgpu::PollType::Poll);

        assert!(
            !device_measurement_path_healthy(&gs_device),
            "destroyed device must fail the live GPU probe"
        );
        let err = finish_profiled_output(
            &gs_device,
            99_i32,
            Err(ProfileMeasureError::ResolvePanic {
                message: "injected resolve after destroy".into(),
            }),
        )
        .expect_err("destroyed device must return TrainingError::Gpu");
        match &err {
            crate::TrainingError::Gpu(message) => {
                assert!(
                    message.contains("unavailable") || message.contains("resolve"),
                    "unexpected gpu error text: {message}"
                );
            }
            other => panic!("expected TrainingError::Gpu, got {other:?}"),
        }
        println!("DESTROYED_DEVICE_WORKER_OK");
        // Skip destructors that would touch the destroyed device / TLS.
        std::process::exit(0);
    }

    #[test]
    fn split_forward_series_kinds_and_sparse_sample_counts() {
        let mut collector = PipelineTimingCollector::new(supported_probe());
        collector.set_profiler_mode(true, true, 5);
        // iterations 1..12 with sample every 5 (+ final): sampled 1,5,10,12
        for iter in 1..=12 {
            if should_profile_gpu(iter, 5, Some(12)) {
                collector.record_span_ms(span::FORWARD_GPU_SAMPLED, 3.0 + iter as f64);
                collector.record_gpu_step_ms(2.0 + iter as f64);
            } else {
                collector.record_span_ms(span::FORWARD_CPU_SUBMIT, 1.0 + iter as f64 * 0.1);
            }
        }
        let report = collector.build_report();
        let sampled = report
            .pipeline_spans
            .get(span::FORWARD_GPU_SAMPLED)
            .expect("sampled");
        let submit = report
            .pipeline_spans
            .get(span::FORWARD_CPU_SUBMIT)
            .expect("submit");
        assert_eq!(sampled.timing_kind, timing_kind::SYNCHRONIZED_BOUNDARY);
        assert_eq!(submit.timing_kind, timing_kind::CPU_SUBMIT);
        // 4 sampled raw → 3 after warmup; 8 submit raw → 7 after warmup
        assert_eq!(sampled.sample_count, 3);
        assert_eq!(submit.sample_count, 7);
        assert_eq!(report.sample_count, 3);
        assert_report_self_consistent(&report).expect("split series consistent");
    }

    #[test]
    fn report_consistency_json_round_trip_rejects_illegal_combinations() {
        let mut good = PipelineTimingCollector::new(supported_probe());
        good.set_profiler_mode(true, true, 20);
        // warmup-only: one GPU sample → success false, still consistent
        good.record_gpu_step_ms(5.0);
        let warmup_only = good.build_report();
        assert!(!warmup_only.measurement_success);
        let json = serde_json::to_string(&warmup_only).expect("ser");
        let decoded: GpuProfilerReport = serde_json::from_str(&json).expect("de");
        assert_report_self_consistent(&decoded).expect("warmup-only ok");

        let mut empty = PipelineTimingCollector::new(supported_probe());
        empty.set_profiler_mode(true, false, 20);
        let no_samples = empty.build_report();
        let json = serde_json::to_string(&no_samples).expect("ser");
        let decoded: GpuProfilerReport = serde_json::from_str(&json).expect("de");
        assert_report_self_consistent(&decoded).expect("no-sample ok");

        let mut bad = warmup_only.clone();
        bad.supported = false;
        bad.unsupported_reason = Some(UNSUPPORTED_TIMESTAMP_QUERY_UNAVAILABLE.into());
        bad.measurement_success = true; // illegal with unsupported
        bad.sample_count = 0;
        bad.gpu_step_p50_ms = None;
        bad.gpu_step_p95_ms = None;
        bad.gpu_forward_sum_ms = None;
        let json = serde_json::to_string(&bad).expect("ser");
        let decoded: GpuProfilerReport = serde_json::from_str(&json).expect("de");
        let err = assert_report_self_consistent(&decoded).expect_err("unsupported+success");
        assert!(
            err.contains("measurement_success") || err.contains("unsupported"),
            "{err}"
        );

        let mut bad2 = warmup_only.clone();
        bad2.measurement_success = true; // sample_count 0
        let json = serde_json::to_string(&bad2).expect("ser");
        let decoded: GpuProfilerReport = serde_json::from_str(&json).expect("de");
        assert!(assert_report_self_consistent(&decoded).is_err());
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
