//! Process-wide admission for synchronous SfM stages. Borrowed callbacks stay
//! on their caller; only Send numerical work enters a bounded Rayon pool.
use crate::task::SfmTaskControl;
use anyhow::{bail, Result};
use rustscan_taskflow::{
    CpuRequest, ExecutionGrant, ResourceRequest, RunHandle, Runtime, RuntimeConfig, TaskError,
    TaskGraph, TaskResult, TaskVariant,
};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
#[cfg(feature = "gpu-wgpu")]
use std::collections::HashMap;
use std::sync::{mpsc, Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

const SCRATCH: u64 = 512 * 1024 * 1024;
const GPU_KEY: &str = "rustsfm/default-gpu";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SfmStageReport {
    pub stage_name: String,
    pub requested_threads: usize,
    pub granted_threads: usize,
    pub requested_memory: u64,
    pub granted_memory: u64,
    pub queue_ms: f64,
    pub service_ms: f64,
    pub total_ms: f64,
    pub cancelled_or_failed: bool,
}

pub(crate) type StageReportSink = Arc<Mutex<Vec<SfmStageReport>>>;

pub(crate) fn new_stage_report_sink() -> StageReportSink {
    Arc::new(Mutex::new(Vec::new()))
}

pub(crate) fn stage_reports(sink: &StageReportSink) -> Vec<SfmStageReport> {
    sink.lock()
        .unwrap_or_else(|error| error.into_inner())
        .clone()
}

fn record_stage_report(sink: &StageReportSink, report: SfmStageReport) {
    sink.lock()
        .unwrap_or_else(|error| error.into_inner())
        .push(report);
}

/// Share one instance across independently submitted workflows. The default
/// instance is process-wide. Memory requests are estimates, not allocator caps.
#[derive(Clone)]
pub struct SfmTaskflow {
    runtime: Arc<Runtime>,
    scratch_bytes: u64,
}

struct Scope {
    runtime: Arc<Runtime>,
    grant: ExecutionGrant,
    control: SfmTaskControl,
    pool: OnceLock<rayon::ThreadPool>,
    #[cfg(feature = "gpu-wgpu")]
    devices: Mutex<HashMap<usize, wgpu::Device>>,
}
// Only the borrowed-call owner retains the context across the stage boundary.
// Worker TLS and pool handlers must not keep their own pool (or Runtime) alive.
thread_local! { static ACTIVE: RefCell<Option<Weak<Scope>>> = const { RefCell::new(None) }; }

fn active_scope() -> Option<Arc<Scope>> {
    ACTIVE.with(|active| {
        active.borrow().as_ref().map(|scope| {
            scope.upgrade().expect(
                "expired SfM admission context: detached work must not escape an admitted stage",
            )
        })
    })
}

struct Restore {
    previous: Option<Weak<Scope>>,
    owner: Option<Arc<Scope>>,
    armed: bool,
}
impl Restore {
    fn drain(&mut self) -> Result<()> {
        if !std::mem::replace(&mut self.armed, false) {
            return Ok(());
        }
        ACTIVE.with(|active| active.replace(self.previous.take()));
        let finished = self.owner.take();
        #[cfg(feature = "gpu-wgpu")]
        if let Some(scope) = &finished {
            // Covers early errors/unwind, including caller-owned GPU contexts.
            // The native device must finish before the outer admission is dropped.
            let devices =
                std::mem::take(&mut *scope.devices.lock().unwrap_or_else(|e| e.into_inner()));
            let mut cleanup = Ok(());
            for device in devices.into_values() {
                let status = device
                    .poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: None,
                    })
                    .map(|_| ())
                    .map_err(|error| anyhow::anyhow!("GPU stage cleanup failed: {error}"));
                // Preserve the first error, but still drain every other device.
                cleanup = cleanup.and(status);
            }
            cleanup?;
        }
        drop(finished);
        Ok(())
    }
}
impl Drop for Restore {
    fn drop(&mut self) {
        if let Err(error) = self.drain() {
            log::warn!("{error:#}");
        }
    }
}

impl SfmTaskflow {
    pub fn new(runtime: Arc<Runtime>, scratch_bytes: u64) -> Result<Self> {
        if scratch_bytes == 0 {
            bail!("SfM stage scratch estimate must be positive");
        }
        Ok(Self {
            runtime,
            scratch_bytes,
        })
    }

    pub fn shared() -> Result<Self> {
        static SHARED: OnceLock<Result<SfmTaskflow, String>> = OnceLock::new();
        SHARED
            .get_or_init(|| {
                let mut config = RuntimeConfig::default();
                config.budget.memory_bytes = 4 * SCRATCH;
                Runtime::new(config)
                    .map(|runtime| Self {
                        runtime: Arc::new(runtime),
                        scratch_bytes: SCRATCH,
                    })
                    .map_err(|error| error.to_string())
            })
            .clone()
            .map_err(anyhow::Error::msg)
    }

    pub(crate) fn merge(&self, other: &Self) -> Result<Self> {
        if !Arc::ptr_eq(&self.runtime, &other.runtime) {
            bail!("all stages in an SfM context must share one Taskflow Runtime");
        }
        Self::new(
            self.runtime.clone(),
            self.scratch_bytes.max(other.scratch_bytes),
        )
    }

    pub fn runtime(&self) -> &Arc<Runtime> {
        &self.runtime
    }

    /// Configured working-memory estimate, not an active grant or an RSS cap.
    pub(crate) fn stage_memory_bytes(&self) -> u64 {
        self.scratch_bytes
    }

    /// Raise the estimate for subsequent stage requests without changing Runtime.
    /// Zero or a smaller floor is a no-op. This does not enlarge an active grant:
    /// nested execution still rejects a request exceeding its parent's allowance.
    pub(crate) fn with_stage_memory_floor(self, bytes: u64) -> Result<Self> {
        Self::new(self.runtime, self.scratch_bytes.max(bytes))
    }

    pub(crate) fn run<R>(
        &self,
        name: &str,
        gpu: bool,
        threads: usize,
        control: &SfmTaskControl,
        work: impl FnOnce() -> Result<R>,
    ) -> Result<R> {
        self.run_internal(name, gpu, threads, control, None, work)
    }

    pub(crate) fn run_with_reports<R>(
        &self,
        name: &str,
        gpu: bool,
        threads: usize,
        control: &SfmTaskControl,
        reports: &StageReportSink,
        work: impl FnOnce() -> Result<R>,
    ) -> Result<R> {
        self.run_internal(name, gpu, threads, control, Some(reports), work)
    }

    fn run_internal<R>(
        &self,
        name: &str,
        gpu: bool,
        threads: usize,
        control: &SfmTaskControl,
        reports: Option<&StageReportSink>,
        work: impl FnOnce() -> Result<R>,
    ) -> Result<R> {
        let requested_threads = threads.max(1);
        let requested_memory = self.scratch_bytes;
        let admission_start = Instant::now();
        if let Err(error) = control.checkpoint() {
            if let Some(reports) = reports {
                record_stage_report(
                    reports,
                    SfmStageReport {
                        stage_name: name.to_owned(),
                        requested_threads,
                        granted_threads: 0,
                        requested_memory,
                        granted_memory: 0,
                        queue_ms: admission_start.elapsed().as_secs_f64() * 1000.0,
                        service_ms: 0.0,
                        total_ms: admission_start.elapsed().as_secs_f64() * 1000.0,
                        cancelled_or_failed: true,
                    },
                );
            }
            return Err(error.into());
        }
        let active = active_scope();
        if let Some(active) = active {
            active.control.checkpoint()?;
            if !Arc::ptr_eq(&active.runtime, &self.runtime) {
                bail!("cannot switch Taskflow runtime inside an admitted stage");
            }
            // Pipeline composition is synchronous: children borrow the parent's
            // allowance. Never hold an allowance while queueing on that same runtime.
            if gpu && !active.grant.exclusive.iter().any(|key| key == GPU_KEY) {
                bail!("GPU stage {name} nested inside a CPU-only stage");
            }
            if requested_memory > active.grant.working_memory_bytes {
                bail!("nested stage {name} exceeds its stage memory allowance: requested {requested_memory}, granted {}", active.grant.working_memory_bytes);
            }
            let service_start = Instant::now();
            let outcome = match reports {
                Some(_) => std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)),
                None => Ok(work()),
            };
            match outcome {
                Ok(result) => {
                    if let Some(reports) = reports {
                        let service_ms = service_start.elapsed().as_secs_f64() * 1000.0;
                        record_stage_report(
                            reports,
                            SfmStageReport {
                                stage_name: name.to_owned(),
                                requested_threads,
                                granted_threads: active.grant.cpu_threads,
                                requested_memory,
                                granted_memory: active.grant.working_memory_bytes,
                                queue_ms: 0.0,
                                service_ms,
                                total_ms: service_ms,
                                cancelled_or_failed: result.is_err(),
                            },
                        );
                    }
                    result
                }
                Err(panic) => {
                    if let Some(reports) = reports {
                        let service_ms = service_start.elapsed().as_secs_f64() * 1000.0;
                        record_stage_report(
                            reports,
                            SfmStageReport {
                                stage_name: name.to_owned(),
                                requested_threads,
                                granted_threads: active.grant.cpu_threads,
                                requested_memory,
                                granted_memory: active.grant.working_memory_bytes,
                                queue_ms: 0.0,
                                service_ms,
                                total_ms: service_ms,
                                cancelled_or_failed: true,
                            },
                        );
                    }
                    std::panic::resume_unwind(panic)
                }
            }
        } else {
            let mut request = ResourceRequest::cpu(CpuRequest::scalable(
                1,
                requested_threads,
                requested_threads,
            ));
            request.working_memory_bytes = requested_memory;
            if gpu {
                request.exclusive.push(GPU_KEY.into());
            }
            let reported = reports.is_some();
            let report_recorded = std::cell::Cell::new(false);
            let run = || {
                admit_outer(&self.runtime, name, request, control, |grant, queue_ms| {
                    let service_start = Instant::now();
                    let result = Self::scoped(&self.runtime, grant, control, work);
                    if let Some(reports) = reports {
                        let service_ms = service_start.elapsed().as_secs_f64() * 1000.0;
                        record_stage_report(
                            reports,
                            SfmStageReport {
                                stage_name: name.to_owned(),
                                requested_threads,
                                granted_threads: grant.cpu_threads,
                                requested_memory,
                                granted_memory: grant.working_memory_bytes,
                                queue_ms,
                                service_ms,
                                total_ms: queue_ms + service_ms,
                                cancelled_or_failed: result.is_err(),
                            },
                        );
                        report_recorded.set(true);
                    }
                    result
                })
            };
            let result = if reported {
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
                    Ok(result) => result,
                    Err(panic) => {
                        if !report_recorded.get() {
                            record_stage_report(
                                reports.unwrap(),
                                SfmStageReport {
                                    stage_name: name.to_owned(),
                                    requested_threads,
                                    granted_threads: 0,
                                    requested_memory,
                                    granted_memory: 0,
                                    queue_ms: admission_start.elapsed().as_secs_f64() * 1000.0,
                                    service_ms: 0.0,
                                    total_ms: admission_start.elapsed().as_secs_f64() * 1000.0,
                                    cancelled_or_failed: true,
                                },
                            );
                        }
                        std::panic::resume_unwind(panic)
                    }
                }
            } else {
                run()
            };
            if reported && !report_recorded.get() && result.is_err() {
                record_stage_report(
                    reports.unwrap(),
                    SfmStageReport {
                        stage_name: name.to_owned(),
                        requested_threads,
                        granted_threads: 0,
                        requested_memory,
                        granted_memory: 0,
                        queue_ms: admission_start.elapsed().as_secs_f64() * 1000.0,
                        service_ms: 0.0,
                        total_ms: admission_start.elapsed().as_secs_f64() * 1000.0,
                        cancelled_or_failed: true,
                    },
                );
            }
            result
        }
    }

    fn scoped<R>(
        runtime: &Arc<Runtime>,
        grant: &ExecutionGrant,
        control: &SfmTaskControl,
        work: impl FnOnce() -> Result<R>,
    ) -> Result<R> {
        let owner = Arc::new(Scope {
            runtime: runtime.clone(),
            grant: grant.clone(),
            control: control.clone(),
            pool: OnceLock::new(),
            #[cfg(feature = "gpu-wgpu")]
            devices: Default::default(),
        });
        let previous = ACTIVE.with(|active| active.replace(Some(Arc::downgrade(&owner))));
        let mut restore = Restore {
            previous,
            owner: Some(owner),
            armed: true,
        };
        let result = work();
        let cleanup = restore.drain();
        result.and_then(|value| cleanup.map(|()| value))
    }

    pub(crate) fn sequence(
        &self,
        stages: &[(&str, bool, usize)],
        control: &SfmTaskControl,
        work: impl FnMut(usize) -> Result<()>,
    ) -> Result<()> {
        self.sequence_internal(stages, control, None, work)
    }

    pub(crate) fn sequence_with_reports(
        &self,
        stages: &[(&str, bool, usize)],
        control: &SfmTaskControl,
        reports: &StageReportSink,
        work: impl FnMut(usize) -> Result<()>,
    ) -> Result<()> {
        self.sequence_internal(stages, control, Some(reports), work)
    }

    fn sequence_internal(
        &self,
        stages: &[(&str, bool, usize)],
        control: &SfmTaskControl,
        reports: Option<&StageReportSink>,
        mut work: impl FnMut(usize) -> Result<()>,
    ) -> Result<()> {
        if active_threads().is_some() {
            for (index, &(name, gpu, threads)) in stages.iter().enumerate() {
                match reports {
                    Some(reports) => {
                        self.run_with_reports(name, gpu, threads, control, reports, || work(index))
                    }
                    None => self.run(name, gpu, threads, control, || work(index)),
                }?;
            }
            return Ok(());
        }
        let requests = stages
            .iter()
            .map(|&(name, gpu, threads)| {
                let mut request =
                    ResourceRequest::cpu(CpuRequest::scalable(1, threads.max(1), threads.max(1)));
                request.working_memory_bytes = self.scratch_bytes;
                if gpu {
                    request.exclusive.push(GPU_KEY.into());
                }
                (name, request)
            })
            .collect::<Vec<_>>();
        admit_chain(
            &self.runtime,
            &requests,
            control,
            |index, grant, queue_ms| {
                let service_start = Instant::now();
                let result = Self::scoped(&self.runtime, grant, control, || work(index));
                if let Some(reports) = reports {
                    let service_ms = service_start.elapsed().as_secs_f64() * 1000.0;
                    record_stage_report(
                        reports,
                        SfmStageReport {
                            stage_name: stages[index].0.to_owned(),
                            requested_threads: stages[index].2.max(1),
                            granted_threads: grant.cpu_threads,
                            requested_memory: self.scratch_bytes,
                            granted_memory: grant.working_memory_bytes,
                            queue_ms,
                            service_ms,
                            total_ms: queue_ms + service_ms,
                            cancelled_or_failed: result.is_err(),
                        },
                    );
                }
                result
            },
        )
    }
}

pub(crate) fn active_threads() -> Option<usize> {
    active_scope().map(|scope| scope.grant.cpu_threads)
}

pub(crate) fn active_control() -> Option<SfmTaskControl> {
    active_scope().map(|scope| scope.control.clone())
}

/// Actual parent working-memory allowance on the caller or an inherited worker.
/// This is the total grant, not unconsumed memory; nested work shares it and must
/// account for simultaneous siblings. No active admission returns None.
pub(crate) fn active_memory_bytes() -> Option<u64> {
    active_scope().map(|scope| scope.grant.working_memory_bytes)
}

#[cfg(feature = "gpu-wgpu")]
pub(crate) fn register_gpu_device(key: usize, device: &wgpu::Device) {
    if let Some(scope) = active_scope() {
        assert!(
            scope.grant.exclusive.iter().any(|key| key == GPU_KEY),
            "GPU submission without a GPU stage allowance"
        );
        scope
            .devices
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(key)
            .or_insert_with(|| device.clone());
    }
}

/// Each admission lazily owns exactly its granted number of compute workers.
/// All workers inherit its context, including work stolen from a par_iter.
/// Only structured Rayon work is supported: detached spawn/global-pool work
/// must not outlive this call or carry the admission onto another thread.
pub(crate) fn parallel<R: Send>(work: impl FnOnce() -> R + Send) -> R {
    let Some(scope) = active_scope() else {
        return work();
    };
    let pool = scope.pool.get_or_init(|| {
        let context = Arc::downgrade(&scope);
        rayon::ThreadPoolBuilder::new()
            .num_threads(scope.grant.cpu_threads)
            .start_handler(move |_| {
                ACTIVE.with(|active| {
                    active.replace(Some(context.clone()));
                });
            })
            .exit_handler(|_| {
                // Worker exit never drains the owner's GPU registry.
                ACTIVE.with(|active| {
                    active.take();
                });
            })
            .build()
            .expect("could not create bounded SfM Rayon pool")
    });
    pool.install(work)
}

/// One borrowed native-call bridge, shared by full stages and the BA adapter.
/// Completion/error/unwind drains the dispatch node before releasing resources.
pub(crate) fn admit<R>(
    runtime: &Arc<Runtime>,
    name: &str,
    request: ResourceRequest,
    control: &SfmTaskControl,
    work: impl FnOnce(&ExecutionGrant, f64) -> Result<R>,
) -> Result<R> {
    control.checkpoint()?;
    let inherited = active_scope();
    if let Some(scope) = inherited {
        scope.control.checkpoint()?;
        if !Arc::ptr_eq(&scope.runtime, runtime) {
            bail!("nested task {name} uses a different Runtime; bind all adapters to the stage Runtime");
        }
        if request.cpu.min == 0
            || request.cpu.min > request.cpu.preferred
            || request.cpu.preferred > request.cpu.max
            || request.cpu.min > scope.grant.cpu_threads
            || request.working_memory_bytes > scope.grant.working_memory_bytes
            || request.output_memory_bytes != 0
            || request.gpu.is_some()
            || request.io_slots > scope.grant.io_slots
            || request
                .exclusive
                .iter()
                .any(|key| !scope.grant.exclusive.contains(key))
        {
            bail!("nested task {name} exceeds its stage resource allowance");
        }
        // Each compute worker already occupies one lane of this admission.
        // Giving every worker the full native grant would multiply concurrency.
        // Query this pool, not Rayon's global index: an unrelated Rayon worker
        // may be the borrowed-call coordinator and still use the full allowance.
        let on_compute_worker = scope
            .pool
            .get()
            .is_some_and(|pool| pool.current_thread_index().is_some());
        if on_compute_worker && request.cpu.min > 1 {
            bail!(
                "nested native task {name} on an SfM compute worker requires at least {} CPU threads, but its worker allowance is 1",
                request.cpu.min
            );
        }
        let mut grant = scope.grant.clone();
        grant.cpu_threads = if on_compute_worker {
            1
        } else {
            grant.cpu_threads.min(request.cpu.max)
        };
        return work(&grant, 0.0);
    }
    admit_outer(runtime, name, request, control, |grant, queue_ms| {
        SfmTaskflow::scoped(runtime, grant, control, || work(grant, queue_ms))
    })
}

// The caller must establish the owner scope inside work and drain it before
// returning. Keeping this bridge separate lets stage reports include cleanup.
fn admit_outer<R>(
    runtime: &Arc<Runtime>,
    name: &str,
    request: ResourceRequest,
    control: &SfmTaskControl,
    work: impl FnOnce(&ExecutionGrant, f64) -> Result<R>,
) -> Result<R> {
    let mut work = Some(work);
    let mut result = None;
    admit_chain(
        runtime,
        &[(name, request)],
        control,
        |_, grant, queue_ms| {
            result = Some(work.take().unwrap()(grant, queue_ms)?);
            Ok(())
        },
    )?;
    Ok(result.expect("completed single-stage graph must have a result"))
}

fn admit_chain(
    runtime: &Arc<Runtime>,
    stages: &[(&str, ResourceRequest)],
    control: &SfmTaskControl,
    mut work: impl FnMut(usize, &ExecutionGrant, f64) -> Result<()>,
) -> Result<()> {
    control.checkpoint()?;
    if stages.is_empty() {
        return Ok(());
    }
    let (grants, receive) = mpsc::sync_channel(1);
    let (finish, finished) = mpsc::sync_channel::<TaskResult<()>>(1);
    // DAG dependencies guarantee that only one node consumes an acknowledgement.
    let finished = Arc::new(Mutex::new(finished));
    let mut graph = TaskGraph::new();
    let mut previous = None;
    for (index, (name, request)) in stages.iter().enumerate() {
        let grants = grants.clone();
        let finished = finished.clone();
        let node = graph.task(
            *name,
            vec![TaskVariant::cpu(
                "borrowed-stage",
                request.clone(),
                move |ctx| {
                    ctx.check_cancelled()?;
                    grants
                        .send((index, ctx.grant().clone()))
                        .map_err(|_| TaskError::Cancelled)?;
                    finished
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .recv()
                        .map_err(|_| TaskError::Cancelled)?
                },
            )],
        )?;
        if let Some(parent) = previous {
            graph.depends_on(node.id(), parent)?;
        }
        previous = Some(node.id());
    }
    drop(grants);
    let active = ActiveCall {
        run: runtime.submit(graph)?,
        finish: Some(finish),
    };
    let mut stage_queue_start = Instant::now();
    for expected in 0..stages.len() {
        let (index, grant) = loop {
            control.checkpoint()?;
            match receive.recv_timeout(Duration::from_millis(5)) {
                Ok(grant) => break grant,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => bail!(
                    "stage admission ended before grant: {:?}",
                    active.run.wait().tasks
                ),
            }
        };
        anyhow::ensure!(index == expected, "stage DAG executed out of order");
        control.checkpoint()?;
        let queue_ms = stage_queue_start.elapsed().as_secs_f64() * 1000.0;
        let result = work(index, &grant, queue_ms);
        let status = result
            .as_ref()
            .map(|_| ())
            .map_err(|error| TaskError::Failed(format!("{error:#}")));
        let _ = active.finish.as_ref().unwrap().send(status);
        result?;
        stage_queue_start = Instant::now();
    }
    let report = active.run.wait();
    anyhow::ensure!(report.succeeded(), "stage graph failed: {:?}", report.tasks);
    Ok(())
}

#[cfg(test)]
#[path = "execution_tests.rs"]
mod tests;

struct ActiveCall {
    run: RunHandle,
    finish: Option<mpsc::SyncSender<TaskResult<()>>>,
}
impl Drop for ActiveCall {
    fn drop(&mut self) {
        self.finish.take();
        self.run.cancel();
        self.run.wait();
    }
}
