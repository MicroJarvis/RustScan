//! GPU completion profiler and unified pipeline timing.
//!
//! CPU spans use host `Instant` and are reported under explicit `cpu_*` field
//! names. GPU step percentiles are filled only after wgpu timestamp-query
//! resolve/readback completes with `TimingMethod::Device`. System/host waits
//! (including `into_scalar_async`) must never be labeled as GPU kernel time.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use super::optimization_report::{duration_millis, percentile_f64};

/// Drop the first N samples per span before percentile aggregation.
pub const PIPELINE_TIMING_WARMUP_SAMPLES: usize = 1;

/// Stable reason when the adapter lacks `TIMESTAMP_QUERY`.
pub const UNSUPPORTED_TIMESTAMP_QUERY_UNAVAILABLE: &str = "timestamp_query_unavailable";

/// Stable reason when cubecl reports system timing instead of device timestamps.
pub const UNSUPPORTED_TIMING_METHOD_SYSTEM: &str = "timing_method_system";

/// Stable reason when adapter metadata cannot be probed.
pub const UNSUPPORTED_ADAPTER_PROBE_FAILED: &str = "adapter_probe_failed";

/// Stable reason when cubecl memory usage is unavailable.
pub const UNSUPPORTED_RUNTIME_PEAK_UNAVAILABLE: &str = "runtime_peak_device_bytes_unavailable";

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
    pub const TOPOLOGY_REMAP: &str = "topology_remap";
    /// Parent CPU wall covering one outer-loop iteration (submit-side Instant).
    pub const STEP_CPU: &str = "step_cpu";

    pub const ALL: &[&str] = &[
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
        TOPOLOGY_REMAP,
        STEP_CPU,
    ];
}

/// Serializable GPU profiler report distinguishing CPU submit from GPU completion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct GpuProfilerReport {
    pub supported: bool,
    pub unsupported_reason: Option<String>,
    pub backend: String,
    pub adapter: Option<String>,
    pub driver: Option<String>,
    pub adapter_unavailable_reason: Option<String>,
    pub driver_unavailable_reason: Option<String>,
    /// GPU completion p50 from timestamp-query resolve; null when unsupported.
    pub gpu_step_p50_ms: Option<f64>,
    /// GPU completion p95 from timestamp-query resolve; null when unsupported.
    pub gpu_step_p95_ms: Option<f64>,
    pub sample_count: u64,
    pub workspace_current_bytes: u64,
    pub workspace_peak_bytes: u64,
    pub workspace_growth_count: u64,
    pub fresh_step_allocations: u64,
    pub runtime_peak_device_bytes: Option<u64>,
    pub runtime_peak_device_bytes_reason: Option<String>,
    /// Explicit CPU submit-side loop percentiles (`Instant`); never GPU.
    pub cpu_step_p50_ms: Option<f64>,
    pub cpu_step_p95_ms: Option<f64>,
    pub cpu_timing_kind: Option<String>,
    pub pipeline_spans: BTreeMap<String, PipelineSpanStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct PipelineSpanStats {
    pub sample_count: u64,
    pub p50_ms: Option<f64>,
    pub p95_ms: Option<f64>,
    /// Always `cpu_submit_instant` for these spans; parent/child are not summed.
    pub timing_kind: String,
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

/// Probe adapter/backend/driver and timestamp-query capability via wgpu.
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

/// Refine probe with live cubecl client timing method and memory.
pub fn refine_probe_with_client_timing(
    mut probe: GpuEnvironmentProbe,
    timing_method_device: bool,
) -> GpuEnvironmentProbe {
    probe.timing_method_device = timing_method_device;
    if probe.timestamp_query_available && !timing_method_device {
        probe.unsupported_reason = Some(UNSUPPORTED_TIMING_METHOD_SYSTEM.into());
    } else if !probe.timestamp_query_available {
        probe.unsupported_reason = Some(UNSUPPORTED_TIMESTAMP_QUERY_UNAVAILABLE.into());
    } else {
        probe.unsupported_reason = None;
    }
    probe
}

#[derive(Debug, Default)]
pub struct PipelineTimingCollector {
    spans: BTreeMap<String, Vec<f64>>,
    gpu_step_ms: Vec<f64>,
    cpu_step_ms: Vec<f64>,
    workspace_current_bytes: u64,
    workspace_peak_bytes: u64,
    workspace_growth_count: u64,
    /// Cumulative fresh allocations across observed steps (monotonic).
    fresh_step_allocations: u64,
    runtime_peak_device_bytes: Option<u64>,
    runtime_peak_device_bytes_reason: Option<String>,
    probe: GpuEnvironmentProbe,
}

impl PipelineTimingCollector {
    pub fn new(probe: GpuEnvironmentProbe) -> Self {
        Self {
            probe,
            runtime_peak_device_bytes_reason: Some(UNSUPPORTED_RUNTIME_PEAK_UNAVAILABLE.into()),
            ..Self::default()
        }
    }

    pub fn probe(&self) -> &GpuEnvironmentProbe {
        &self.probe
    }

    pub fn set_probe(&mut self, probe: GpuEnvironmentProbe) {
        self.probe = probe;
    }

    pub fn record_span_ms(&mut self, name: &str, ms: f64) {
        self.spans.entry(name.to_string()).or_default().push(ms);
    }

    pub fn record_span(&mut self, name: &str, duration: Duration) {
        self.record_span_ms(name, duration_millis(duration));
    }

    pub fn record_cpu_step(&mut self, duration: Duration) {
        let ms = duration_millis(duration);
        self.cpu_step_ms.push(ms);
        self.record_span_ms(span::STEP_CPU, ms);
    }

    /// Record a GPU completion sample only when device timestamps are supported.
    pub fn record_gpu_step_ms(&mut self, ms: f64) {
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
                self.runtime_peak_device_bytes_reason = None;
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

        let (gpu_step_p50_ms, gpu_step_p95_ms, sample_count) = if supported {
            let mut sorted = gpu_samples;
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            (
                percentile_f64(&sorted, 50.0),
                percentile_f64(&sorted, 95.0),
                sorted.len() as u64,
            )
        } else {
            (None, None, 0)
        };

        let mut cpu_sorted = cpu_samples;
        cpu_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut pipeline_spans = BTreeMap::new();
        for name in span::ALL {
            let raw = self.spans.get(*name).cloned().unwrap_or_default();
            let warmed = apply_warmup(&raw);
            let sample_count = warmed.len() as u64;
            let mut sorted = warmed;
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            pipeline_spans.insert(
                (*name).to_string(),
                PipelineSpanStats {
                    sample_count,
                    p50_ms: percentile_f64(&sorted, 50.0),
                    p95_ms: percentile_f64(&sorted, 95.0),
                    timing_kind: "cpu_submit_instant".into(),
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
            gpu_step_p50_ms,
            gpu_step_p95_ms,
            sample_count,
            workspace_current_bytes: self.workspace_current_bytes,
            workspace_peak_bytes: self.workspace_peak_bytes,
            workspace_growth_count: self.workspace_growth_count,
            fresh_step_allocations: self.fresh_step_allocations,
            runtime_peak_device_bytes: self.runtime_peak_device_bytes,
            runtime_peak_device_bytes_reason: self.runtime_peak_device_bytes_reason.clone(),
            cpu_step_p50_ms: percentile_f64(&cpu_sorted, 50.0),
            cpu_step_p95_ms: percentile_f64(&cpu_sorted, 95.0),
            cpu_timing_kind: Some("cpu_submit_instant".into()),
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

/// RAII CPU span timer.
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

/// Build environment probe refined with the live Burn/cubecl client's timing method.
pub fn probe_training_device(device: &crate::training::engine::GsDevice) -> GpuEnvironmentProbe {
    use burn_cubecl::cubecl::profile::TimingMethod;
    use burn_cubecl::cubecl::Runtime;
    use burn_wgpu::WgpuRuntime;

    let probe = probe_wgpu_environment();
    let client = WgpuRuntime::client(device);
    let timing_method_device = client.properties().timing_method == TimingMethod::Device;
    refine_probe_with_client_timing(probe, timing_method_device)
}

/// Read cubecl allocator bytes-in-use for runtime peak tracking.
pub fn runtime_device_bytes_in_use(device: &crate::training::engine::GsDevice) -> Option<u64> {
    use burn_cubecl::cubecl::Runtime;
    use burn_wgpu::WgpuRuntime;

    let client = WgpuRuntime::client(device);
    client.memory_usage().ok().map(|usage| usage.bytes_in_use)
}

/// Resolve a cubecl profile duration into GPU milliseconds only for Device timing.
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

/// Send-capable raw pointer for exclusive same-thread GPU profile closures.
pub(crate) struct SendMutPtr<T>(pub(crate) *mut T);

// SAFETY: The pointer is only dereferenced inside `profile_device_gpu_step`, which
// completes the future on the calling thread before returning to any other access.
unsafe impl<T> Send for SendMutPtr<T> {}

impl<T> SendMutPtr<T> {
    /// # Safety
    /// Caller must ensure exclusive access for the duration of the returned borrow.
    pub(crate) unsafe fn as_mut<'a>(self) -> &'a mut T {
        unsafe { &mut *self.0 }
    }
}

pub(crate) struct SendConstPtr<T>(pub(crate) *const T);

// SAFETY: Same exclusive completion guarantee as `SendMutPtr`.
unsafe impl<T> Send for SendConstPtr<T> {}

impl<T> SendConstPtr<T> {
    /// # Safety
    /// Caller must ensure the pointees remain valid for the returned borrow.
    pub(crate) unsafe fn as_ref<'a>(self) -> &'a T {
        unsafe { &*self.0 }
    }
}

/// Run `work` under cubecl device timestamp profiling when supported.
///
/// The future is driven with `cubecl::future::block_on` inside `client.profile`
/// so timestamp start/end bracket the submitted GPU work. Callers must not
/// include loss `into_scalar_async` readback inside `work` — that host sync is
/// CPU time, not GPU completion.
///
/// When profiling cannot start, `work` still runs once (unprofiled).
pub async fn profile_device_gpu_step<F, Fut, T>(
    device: &crate::training::engine::GsDevice,
    work: F,
) -> (T, Option<f64>)
where
    F: FnOnce() -> Fut + Send,
    Fut: std::future::Future<Output = T>,
    T: Send + 'static,
{
    use burn_cubecl::cubecl::profile::TimingMethod;
    use burn_cubecl::cubecl::Runtime;
    use burn_wgpu::WgpuRuntime;
    use std::sync::{Arc, Mutex};

    let client = WgpuRuntime::client(device);
    if client.properties().timing_method != TimingMethod::Device {
        return (work().await, None);
    }

    let work_slot = Arc::new(Mutex::new(Some(work)));
    let work_slot_for_profile = Arc::clone(&work_slot);
    match client.profile(
        move || {
            let work = work_slot_for_profile
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .take()
                .expect("gpu profile work slot");
            burn_cubecl::cubecl::future::block_on(work())
        },
        "gpu_step",
    ) {
        Ok((output, profile)) => {
            let ms = resolve_device_gpu_ms(profile).await;
            (output, ms)
        }
        Err(_) => {
            let work = work_slot
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .take()
                .expect("gpu profile work slot after start failure");
            (work().await, None)
        }
    }
}

/// Validate profiler report self-consistency (p95>=p50, unsupported nulls).
pub fn assert_report_self_consistent(report: &GpuProfilerReport) -> Result<(), String> {
    if report.supported {
        if report.unsupported_reason.is_some() {
            return Err("supported report must not set unsupported_reason".into());
        }
        if report.sample_count > 0 {
            match (report.gpu_step_p50_ms, report.gpu_step_p95_ms) {
                (Some(p50), Some(p95)) => {
                    if p95 < p50 {
                        return Err(format!("gpu p95 ({p95}) < p50 ({p50})"));
                    }
                }
                _ => {
                    return Err("supported report with samples must have gpu p50/p95".into());
                }
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
        if report.sample_count != 0 {
            return Err("unsupported report sample_count must be 0".into());
        }
    }

    if let (Some(p50), Some(p95)) = (report.cpu_step_p50_ms, report.cpu_step_p95_ms) {
        if p95 < p50 {
            return Err(format!("cpu p95 ({p95}) < p50 ({p50})"));
        }
    }

    for (name, stats) in &report.pipeline_spans {
        if stats.timing_kind != "cpu_submit_instant" {
            return Err(format!("span {name} must use cpu_submit_instant"));
        }
        if let (Some(p50), Some(p95)) = (stats.p50_ms, stats.p95_ms) {
            if p95 < p50 {
                return Err(format!("span {name} p95 ({p95}) < p50 ({p50})"));
            }
        }
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

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_p95_ge_p50_and_sample_count_matches() {
        let mut collector = PipelineTimingCollector::new(GpuEnvironmentProbe {
            backend: "Metal".into(),
            adapter: Some("Test GPU".into()),
            driver: Some("TestDriver".into()),
            timestamp_query_available: true,
            timing_method_device: true,
            ..Default::default()
        });
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
        // Warmup drops 1.0; remaining [10,20,30,40]. Rank for p50 is round(0.5*3)=2 → 30.
        assert_eq!(report.gpu_step_p50_ms, Some(30.0));
        assert_eq!(report.gpu_step_p95_ms, Some(40.0));
        assert!(report.gpu_step_p95_ms.unwrap() >= report.gpu_step_p50_ms.unwrap());
        assert_eq!(report.workspace_current_bytes, 120);
        assert_eq!(report.workspace_peak_bytes, 120);
        assert_eq!(report.workspace_growth_count, 3);
        assert_eq!(report.fresh_step_allocations, 1);
        assert_eq!(report.runtime_peak_device_bytes, Some(4096));
        assert_report_self_consistent(&report).expect("consistent");
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
        assert_eq!(
            report.unsupported_reason.as_deref(),
            Some(UNSUPPORTED_TIMESTAMP_QUERY_UNAVAILABLE)
        );
        assert!(report.gpu_step_p50_ms.is_none());
        assert!(report.gpu_step_p95_ms.is_none());
        assert_eq!(report.sample_count, 0);
        assert_eq!(
            report.cpu_timing_kind.as_deref(),
            Some("cpu_submit_instant")
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
            gpu_step_p50_ms: None,
            gpu_step_p95_ms: None,
            sample_count: 0,
            workspace_current_bytes: 2048,
            workspace_peak_bytes: 4096,
            workspace_growth_count: 2,
            fresh_step_allocations: 3,
            runtime_peak_device_bytes: None,
            runtime_peak_device_bytes_reason: Some(UNSUPPORTED_RUNTIME_PEAK_UNAVAILABLE.into()),
            cpu_step_p50_ms: Some(18.0),
            cpu_step_p95_ms: Some(22.0),
            cpu_timing_kind: Some("cpu_submit_instant".into()),
            pipeline_spans: BTreeMap::from([(
                span::FORWARD.to_string(),
                PipelineSpanStats {
                    sample_count: 2,
                    p50_ms: Some(5.0),
                    p95_ms: Some(7.0),
                    timing_kind: "cpu_submit_instant".into(),
                },
            )]),
        };
        let json = serde_json::to_string_pretty(&report).expect("serialize");
        assert!(json.contains("\"gpu_step_p50_ms\": null"));
        assert!(json.contains("\"gpu_step_p95_ms\": null"));
        assert!(json.contains("timestamp_query_unavailable"));
        assert!(json.contains("cpu_submit_instant"));
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
}
