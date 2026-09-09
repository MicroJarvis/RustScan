use serde::{Deserialize, Serialize};
use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc,
};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SfmTaskStage {
    FeatureExtraction,
    FeatureMatching,
    KeyframeSelection,
    IncrementalMapping,
    BundleAdjustment,
    FullFrameRegistration,
    Export,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SfmTaskOperation {
    Begin,
    ExtractImage,
    MatchPairBatch,
    EvaluateKeyframePair,
    SelectKeyframe,
    RegisterInitialPair,
    RegisterImage,
    LocalBundleAdjustment,
    GlobalBundleAdjustment,
    RegisterFrameAttempt,
    ValidateArtifacts,
    WriteArtifacts,
    Complete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SfmTaskEventKind {
    Started,
    Progress,
    Warning,
    Error,
    Completed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SfmTaskIssue {
    pub code: String,
    pub summary: String,
    pub detail: String,
    pub retryable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SfmTaskEvent {
    pub sequence: u64,
    pub elapsed_ms: u64,
    pub stage: SfmTaskStage,
    pub operation: SfmTaskOperation,
    pub kind: SfmTaskEventKind,
    pub completed: Option<usize>,
    pub total: Option<usize>,
    pub registered_images: Option<usize>,
    pub sparse_points: Option<usize>,
    pub image_id: Option<u32>,
    pub pair: Option<(u32, u32)>,
    pub message: Option<String>,
    pub issue: Option<SfmTaskIssue>,
}

pub trait SfmTaskEventSink {
    fn on_sfm_event(&mut self, event: SfmTaskEvent);
}

impl<F> SfmTaskEventSink for F
where
    F: FnMut(SfmTaskEvent),
{
    fn on_sfm_event(&mut self, event: SfmTaskEvent) {
        self(event);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SfmControlState {
    Running = 0,
    PauseRequested = 1,
    CancelRequested = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SfmTaskStop {
    #[error("SFM task paused at a safe boundary")]
    Paused,
    #[error("SFM task cancelled at a safe boundary")]
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct SfmTaskControl {
    state: Arc<AtomicU8>,
}

impl Default for SfmTaskControl {
    fn default() -> Self {
        Self {
            state: Arc::new(AtomicU8::new(SfmControlState::Running as u8)),
        }
    }
}

impl SfmTaskControl {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn request_pause(&self) {
        let _ = self.state.compare_exchange(
            SfmControlState::Running as u8,
            SfmControlState::PauseRequested as u8,
            Ordering::SeqCst,
            Ordering::SeqCst,
        );
    }

    pub fn request_cancel(&self) {
        self.state
            .store(SfmControlState::CancelRequested as u8, Ordering::SeqCst);
    }

    pub fn state(&self) -> SfmControlState {
        match self.state.load(Ordering::SeqCst) {
            0 => SfmControlState::Running,
            1 => SfmControlState::PauseRequested,
            2 => SfmControlState::CancelRequested,
            state => unreachable!("invalid SFM control state: {state}"),
        }
    }

    pub fn checkpoint(&self) -> Result<(), SfmTaskStop> {
        match self.state() {
            SfmControlState::Running => Ok(()),
            SfmControlState::PauseRequested => Err(SfmTaskStop::Paused),
            SfmControlState::CancelRequested => Err(SfmTaskStop::Cancelled),
        }
    }
}

pub struct SfmTaskContext<'a> {
    control: &'a SfmTaskControl,
    sink: &'a mut dyn SfmTaskEventSink,
    started_at: Instant,
    next_sequence: u64,
    cpu_feature_taskflow: Option<&'a crate::feature_extraction::CpuFeatureTaskflow>,
    ceres_ba_taskflow: Option<&'a crate::ba::CeresBaTaskflow>,
    taskflow: Option<crate::execution::SfmTaskflow>,
    stage_reports: crate::execution::StageReportSink,
}

impl<'a> SfmTaskContext<'a> {
    pub fn new(control: &'a SfmTaskControl, sink: &'a mut dyn SfmTaskEventSink) -> Self {
        Self {
            control,
            sink,
            started_at: Instant::now(),
            next_sequence: 0,
            cpu_feature_taskflow: None,
            ceres_ba_taskflow: None,
            taskflow: None,
            stage_reports: crate::execution::new_stage_report_sink(),
        }
    }

    /// Override the process-wide stage scheduler for this workflow.
    /// For built-in standard CPU SIFT database extraction this also opts into
    /// automatic allocation planning, with this executor's estimate as a floor.
    /// Unbound contexts retain the legacy estimate without that guarantee.
    pub fn with_taskflow(mut self, executor: crate::execution::SfmTaskflow) -> Self {
        self.taskflow = Some(executor);
        self
    }

    pub(crate) fn execute<R>(
        &mut self,
        name: &str,
        gpu: bool,
        threads: usize,
        work: impl FnOnce(&mut Self) -> anyhow::Result<R>,
    ) -> anyhow::Result<R> {
        let executor = self.executor()?;
        let control = self.control.clone();
        let reports = self.stage_reports.clone();
        executor.run_with_reports(name, gpu, threads, &control, &reports, || work(self))
    }

    pub(crate) fn has_feature_memory_estimate(&self) -> bool {
        self.taskflow.is_some() || self.cpu_feature_taskflow.is_some()
    }

    pub(crate) fn feature_memory_limits(&self) -> anyhow::Result<(u64, u64)> {
        let executor = self.executor()?;
        let budget = match crate::execution::active_memory_bytes() {
            Some(grant) => grant,
            None => executor.runtime().snapshot()?.budget.memory_bytes,
        };
        Ok((executor.stage_memory_bytes(), budget))
    }

    pub(crate) fn execute_with_memory<R>(
        &mut self,
        name: &str,
        bytes: u64,
        threads: usize,
        work: impl FnOnce(&mut Self) -> anyhow::Result<R>,
    ) -> anyhow::Result<R> {
        self.execute_with_memory_and_gpu(name, false, bytes, threads, work)
    }

    /// Apply a request floor for this call only, preserving GPU exclusivity.
    pub(crate) fn execute_with_memory_and_gpu<R>(
        &mut self,
        name: &str,
        gpu: bool,
        bytes: u64,
        threads: usize,
        work: impl FnOnce(&mut Self) -> anyhow::Result<R>,
    ) -> anyhow::Result<R> {
        let executor = self.executor()?.with_stage_memory_floor(bytes)?;
        let requested = executor.stage_memory_bytes();
        // A parent's actual grant is a hard boundary, not remaining capacity.
        // Transient Runtime budgets are NOT ceilings: submit validates against
        // immutable config and queues until budget recovery or cancellation.
        if let Some(budget) =
            crate::execution::active_memory_bytes().filter(|&grant| requested > grant)
        {
            self.stage_reports
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(crate::SfmStageReport {
                    stage_name: name.to_owned(),
                    requested_threads: threads.max(1),
                    granted_threads: 0,
                    requested_memory: requested,
                    granted_memory: 0,
                    queue_ms: 0.0,
                    service_ms: 0.0,
                    total_ms: 0.0,
                    cancelled_or_failed: true,
                });
            // Match execution's checkpoint precedence even on this early
            // rejection path. A fresh child control must not hide a stopped
            // inherited scope; retain the failed report for either outcome.
            self.control.checkpoint()?;
            if let Some(control) = crate::execution::active_control() {
                control.checkpoint()?;
            }
            anyhow::bail!("feature allocation estimate {requested} bytes exceeds parent grant {budget} bytes before decode (estimate, not RSS cap)");
        }
        let control = self.control.clone();
        let reports = self.stage_reports.clone();
        let started = std::cell::Cell::new(false);
        executor.run_with_reports(name, gpu, threads, &control, &reports, || {
            started.set(true);
            work(self)
        }).map_err(|error| {
            if !started.get() && matches!(error.downcast_ref::<rustscan_taskflow::Error>(),
                Some(rustscan_taskflow::Error::Unschedulable(_))) {
                error.context(format!("feature allocation estimate {requested} bytes cannot fit immutable runtime resources before decode (estimate, not RSS cap)"))
            } else {
                error
            }
        })
    }

    pub(crate) fn sequence(
        &mut self,
        stages: &[(&str, bool, usize)],
        mut work: impl FnMut(usize, &mut Self) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        let executor = self.executor()?;
        let control = self.control.clone();
        let reports = self.stage_reports.clone();
        executor.sequence_with_reports(stages, &control, &reports, |index| work(index, self))
    }

    /// Common request floor for the existing chain, not an enclosing admission.
    /// Each node retains its own grant, GPU flag and report. The configured floor
    /// is unchanged; a nested chain still borrows its parent's total allowance.
    pub(crate) fn sequence_with_memory(
        &mut self,
        stages: &[(&str, bool, usize)],
        bytes: u64,
        mut work: impl FnMut(usize, &mut Self) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        let executor = self.executor()?.with_stage_memory_floor(bytes)?;
        let control = self.control.clone();
        let reports = self.stage_reports.clone();
        executor.sequence_with_reports(stages, &control, &reports, |index| work(index, self))
    }

    fn executor(&self) -> anyhow::Result<crate::execution::SfmTaskflow> {
        let mut executor = self.taskflow.clone();
        for component in [
            self.cpu_feature_taskflow
                .map(|adapter| adapter.stage_taskflow())
                .transpose()?,
            self.ceres_ba_taskflow
                .map(|adapter| adapter.stage_taskflow())
                .transpose()?,
        ]
        .into_iter()
        .flatten()
        {
            executor = Some(match executor {
                Some(existing) => existing.merge(&component)?,
                None => component,
            });
        }
        executor
            .map(Ok)
            .unwrap_or_else(crate::execution::SfmTaskflow::shared)
    }

    /// Opt this context's CPU database feature extraction into a shared runtime.
    /// Composed workflows inherit this Runtime; pair it with BA on the same Runtime.
    pub fn with_cpu_feature_taskflow(
        mut self,
        executor: &'a crate::feature_extraction::CpuFeatureTaskflow,
    ) -> Self {
        self.cpu_feature_taskflow = Some(executor);
        self
    }

    /// Bind Ceres BA to the same shared runtime as other scheduled stages.
    pub fn with_ceres_ba_taskflow(mut self, executor: &'a crate::ba::CeresBaTaskflow) -> Self {
        self.ceres_ba_taskflow = Some(executor);
        self
    }

    pub(crate) fn inherit_ba_taskflow(
        &mut self,
        config: &crate::MapperConfig,
    ) -> anyhow::Result<()> {
        if let Some(adapter) = &config.ba_taskflow {
            let component = adapter.stage_taskflow()?;
            let executor = match &self.taskflow {
                Some(existing) => existing.merge(&component)?,
                None => component,
            };
            self.taskflow = Some(executor);
        }
        Ok(())
    }

    pub(crate) fn bind_ba_taskflow(&self, config: &mut crate::MapperConfig) {
        if let Some(executor) = self.ceres_ba_taskflow.or(config.ba_taskflow.as_ref()) {
            config.ba_taskflow = Some(executor.clone().with_control(self.control.clone()));
        }
    }

    pub fn stage_reports(&self) -> Vec<crate::SfmStageReport> {
        crate::execution::stage_reports(&self.stage_reports)
    }

    pub(crate) fn stage_report_sink(&self) -> crate::execution::StageReportSink {
        self.stage_reports.clone()
    }

    pub(crate) fn control(&self) -> SfmTaskControl {
        self.control.clone()
    }

    pub(crate) fn feature_batch_threads(&self) -> usize {
        if crate::execution::active_threads().is_some() && self.cpu_feature_taskflow.is_some() {
            // A legacy adapter's estimate covers one image, not a parallel batch.
            1
        } else {
            crate::execution::active_threads()
                .unwrap_or_else(rayon::current_num_threads)
                .max(1)
        }
    }

    pub(crate) fn cpu_feature_taskflow(
        &self,
    ) -> anyhow::Result<Option<&'a crate::feature_extraction::CpuFeatureTaskflow>> {
        // A whole-stage allowance already covers nested extraction. Submitting
        // child graphs while retaining that allowance can deadlock a small budget.
        if crate::execution::active_threads().is_some() {
            return Ok(None);
        }
        if self.cpu_feature_taskflow.is_some() {
            // Standalone image DAGs bypass stage admission, not binding validation.
            self.executor()?;
        }
        Ok(self.cpu_feature_taskflow)
    }

    pub fn emit(&mut self, mut event: SfmTaskEvent) {
        event.sequence = self.next_sequence;
        event.elapsed_ms = self.started_at.elapsed().as_millis().min(u64::MAX as u128) as u64;
        self.next_sequence = self.next_sequence.saturating_add(1);
        self.sink.on_sfm_event(event);
    }

    pub fn checkpoint(&self) -> Result<(), SfmTaskStop> {
        self.control.checkpoint()
    }
}

#[cfg(test)]
mod composition_memory_tests {
    use super::*;

    #[test]
    fn memory_review_typed_stop_precedes_parent_limit_with_fresh_child_control(
    ) -> anyhow::Result<()> {
        for expected in [SfmTaskStop::Paused, SfmTaskStop::Cancelled] {
            for stop_parent in [false, true] {
                let runtime = Arc::new(rustscan_taskflow::Runtime::new(
                    rustscan_taskflow::RuntimeConfig {
                        budget: rustscan_taskflow::Budget {
                            cpu_threads: 1,
                            memory_bytes: 256,
                            io_slots: 1,
                        },
                        ..Default::default()
                    },
                )?);
                let executor = crate::SfmTaskflow::new(runtime.clone(), 128)?;
                let parent_control = SfmTaskControl::new();
                let child_control = SfmTaskControl::new();
                executor.run("parent", false, 1, &parent_control, || {
                    let stopped = if stop_parent {
                        &parent_control
                    } else {
                        &child_control
                    };
                    match expected {
                        SfmTaskStop::Paused => stopped.request_pause(),
                        SfmTaskStop::Cancelled => stopped.request_cancel(),
                    }
                    if stop_parent {
                        assert_eq!(child_control.state(), SfmControlState::Running);
                    }
                    let mut sink = |_| {};
                    let mut child = SfmTaskContext::new(&child_control, &mut sink)
                        .with_taskflow(executor.clone());
                    let mut entered = false;
                    let error = child
                        .execute_with_memory_and_gpu(
                            "stopped oversized child",
                            false,
                            129,
                            1,
                            |_| {
                                entered = true;
                                Ok(())
                            },
                        )
                        .unwrap_err();
                    assert!(!entered);
                    assert_eq!(error.downcast_ref::<SfmTaskStop>(), Some(&expected));
                    let reports = child.stage_reports();
                    assert_eq!(reports.len(), 1);
                    let report = &reports[0];
                    assert_eq!(report.stage_name, "stopped oversized child");
                    assert_eq!(report.requested_memory, 129);
                    assert_eq!(
                        (
                            report.requested_threads,
                            report.granted_threads,
                            report.granted_memory
                        ),
                        (1, 0, 0)
                    );
                    assert_eq!(
                        (report.queue_ms, report.service_ms, report.total_ms),
                        (0.0, 0.0, 0.0)
                    );
                    assert!(report.cancelled_or_failed);
                    Ok(())
                })?;
                let snapshot = runtime.snapshot()?;
                assert_eq!(
                    (
                        snapshot.cpu_threads,
                        snapshot.memory_bytes,
                        snapshot.pending_tasks
                    ),
                    (0, 0, 0)
                );
                assert!(crate::execution::active_control().is_none());
            }
        }
        Ok(())
    }

    #[test]
    fn composition_memory_helpers_keep_gpu_chain_and_configured_floor() -> anyhow::Result<()> {
        let runtime = Arc::new(rustscan_taskflow::Runtime::new(
            rustscan_taskflow::RuntimeConfig {
                budget: rustscan_taskflow::Budget {
                    cpu_threads: 1,
                    memory_bytes: 1024,
                    io_slots: 1,
                },
                ..Default::default()
            },
        )?);
        let executor = crate::SfmTaskflow::new(runtime.clone(), 128)?;
        let control = SfmTaskControl::new();
        let mut sink = |_| {};
        let mut task = SfmTaskContext::new(&control, &mut sink).with_taskflow(executor);
        task.execute_with_memory_and_gpu("gpu parent", true, 256, 1, |task| {
            assert_eq!(crate::execution::active_memory_bytes(), Some(256));
            task.execute("gpu child", true, 1, |_| Ok(()))
        })?;
        assert_eq!(task.feature_memory_limits()?.0, 128);
        let report_start = task.stage_reports().len();
        let mut visited = Vec::new();
        task.sequence_with_memory(
            &[("one", false, 1), ("two", true, 1)],
            384,
            |index, task| {
                assert_eq!(crate::execution::active_memory_bytes(), Some(384));
                assert_eq!(runtime.snapshot()?.memory_bytes, 384);
                if index == 1 {
                    task.execute("nested gpu", true, 1, |_| Ok(()))?;
                }
                visited.push(index);
                Ok(())
            },
        )?;
        assert_eq!(visited, [0, 1]);
        let reports = task.stage_reports();
        let chain = reports[report_start..]
            .iter()
            .filter(|r| r.stage_name == "one" || r.stage_name == "two")
            .collect::<Vec<_>>();
        assert_eq!(chain.len(), 2);
        assert!(chain
            .iter()
            .all(|r| r.requested_memory == 384 && r.granted_memory == 384));
        assert_eq!(task.feature_memory_limits()?.0, 128);
        task.execute("small parent", false, 1, |task| {
            let mut entered = false;
            assert!(task
                .execute_with_memory_and_gpu("too large", false, 129, 1, |_| {
                    entered = true;
                    Ok(())
                })
                .is_err());
            assert!(task
                .sequence_with_memory(&[("too large chain", false, 1)], 129, |_, _| {
                    entered = true;
                    Ok(())
                })
                .is_err());
            assert!(task
                .execute_with_memory_and_gpu("gpu under cpu", true, 128, 1, |_| {
                    entered = true;
                    Ok(())
                })
                .is_err());
            assert!(!entered);
            Ok(())
        })?;
        let snapshot = runtime.snapshot()?;
        assert_eq!(
            (
                snapshot.cpu_threads,
                snapshot.memory_bytes,
                snapshot.pending_tasks
            ),
            (0, 0, 0)
        );
        assert_eq!(snapshot.budget.memory_bytes, 1024);
        assert_eq!(task.feature_memory_limits()?.0, 128);
        Ok(())
    }
}
