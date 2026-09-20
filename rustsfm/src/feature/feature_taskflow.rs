use super::{
    extract_cpu_image, ColmapDatabaseImage, CpuSiftExtractor, ExtractedCpuImage,
    SiftExtractionOptions,
};
use crate::task::SfmTaskControl;
use anyhow::{bail, Context, Result};
use rustscan_taskflow::{
    Artifact, CpuRequest, ResourceRequest, RunHandle, Runtime, TaskError, TaskGraph, TaskVariant,
};
use std::path::Path;
use std::sync::{mpsc, Arc};
use std::time::Duration;

/// Opt-in CPU SIFT adapter. Share the same Runtime across workflows.
///
/// One single-threaded decode/SIFT node feeds each ordered database-write node.
/// The caller supplies a per-image allocation estimate (decoded image,
/// SIFT scratch, conversions and output), raised to the known standard VLFeat
/// working-set estimate. Other backends rely on the caller's conservative
/// estimate; no automatic bound is claimed. That whole reservation stays alive
/// until the image is committed or discarded. This is not an allocator limit.
/// Every image in a graph window reserves the same maximum image estimate, so
/// an ordered writer cannot retain a lighter image that overtook a blocked heavy
/// predecessor. Equal resource requests preserve extraction admission order in
/// the runtime ready queue while still allowing parallel extraction when it fits.
/// Transient budget reductions queue work until recovery/cancellation; only the
/// immutable runtime ceiling rejects it. The window also bounds graph metadata.
#[derive(Clone)]
pub struct CpuFeatureTaskflow {
    runtime: Arc<Runtime>,
    memory_per_image_bytes: u64,
    max_in_flight_images: usize,
}

impl CpuFeatureTaskflow {
    pub fn new(
        runtime: Arc<Runtime>,
        memory_per_image_bytes: u64,
        max_in_flight_images: usize,
    ) -> Result<Self> {
        if memory_per_image_bytes == 0 || max_in_flight_images == 0 {
            bail!("Taskflow per-image memory and image window must be positive");
        }
        Ok(Self {
            runtime,
            memory_per_image_bytes,
            max_in_flight_images,
        })
    }

    pub(crate) fn stage_taskflow(&self) -> Result<crate::SfmTaskflow> {
        crate::SfmTaskflow::new(self.runtime.clone(), self.memory_per_image_bytes)
    }

    pub fn max_in_flight_images(&self) -> usize {
        self.max_in_flight_images
    }

    pub(super) fn extract_batch(
        &self,
        batch: ExtractionBatch<'_>,
        control: &SfmTaskControl,
        write: &mut impl FnMut(&ExtractedCpuImage) -> Result<()>,
    ) -> Result<()> {
        let ExtractionBatch {
            images,
            database_path,
            images_dir,
            options,
        } = batch;
        control.checkpoint()?;
        // Headers only: every image in the graph is estimated before any decode.
        // Output reservations remain charged through ordered commit, so the
        // runtime admits the aggregate rather than just individual scratch.
        let estimates = images
            .iter()
            .map(|image| {
                super::memory::image_estimate(
                    &images_dir.join(&image.name),
                    options,
                    self.memory_per_image_bytes,
                    control,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let images_dir = images_dir.to_path_buf();
        let options = options.clone();
        self.extract_estimated_batch(
            images,
            database_path,
            &estimates,
            control,
            move |image| extract_cpu_image(image, &images_dir, &options, &CpuSiftExtractor),
            write,
        )
    }

    // Separate graph admission from decoding so small synthetic tests can prove
    // liveness under external occupancy without allocating SIFT-sized buffers.
    #[allow(clippy::too_many_arguments)]
    fn extract_estimated_batch(
        &self,
        images: &[ColmapDatabaseImage],
        database_path: &Path,
        estimates: &[u64],
        control: &SfmTaskControl,
        extract: impl Fn(&ColmapDatabaseImage) -> Result<ExtractedCpuImage> + Send + Sync + 'static,
        write: &mut impl FnMut(&ExtractedCpuImage) -> Result<()>,
    ) -> Result<()> {
        control.checkpoint()?;
        anyhow::ensure!(
            images.len() == estimates.len(),
            "one memory estimate is required per image"
        );
        // Without a common floor, 80 then 30 under budget 100 with external 40
        // lets 30 overtake and retain its output forever waiting for 80 to fit.
        let window_memory = estimates
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .max(self.memory_per_image_bytes);
        let extract = Arc::new(extract);
        let database_key = format!("sfm/features/{}", database_path.canonicalize()?.display());
        let (sender, receiver) = mpsc::sync_channel(1);
        let mut graph = TaskGraph::new();
        let mut previous_write = None;
        for image in images {
            let mut request = ResourceRequest::cpu(CpuRequest::fixed(1));
            // Conservatively retain scratch headroom with the output. This lets
            // a writer always run without acquiring more memory and avoids
            // decode-ahead consuming the memory needed by its own consumers.
            request.output_memory_bytes = window_memory;
            request.io_slots = 1;
            let image = image.clone();
            let name = image.name.clone();
            let extract = extract.clone();
            let extraction_control = control.clone();
            let extracted = graph.task(
                format!("extract {name}"),
                vec![TaskVariant::cpu("cpu-sift", request, move |ctx| {
                    ctx.check_cancelled()?;
                    extraction_control.checkpoint().map_err(task_error)?;
                    // VLFeat is built with VL_DISABLE_OPENMP. Parallelism is across
                    // images, not independent per-image/global Rayon pools.
                    let result = extract(&image).map_err(task_error)?;
                    ctx.check_cancelled()?;
                    Ok(result)
                })],
            )?;
            let extracted_id = extracted.id();
            let mut request = ResourceRequest::cpu(CpuRequest::fixed(1));
            request.io_slots = 1;
            request.exclusive.push(database_key.clone());
            let sender = sender.clone();
            let writer = graph.task(
                format!("commit {name}"),
                vec![TaskVariant::cpu("sqlite", request, move |ctx| {
                    ctx.check_cancelled()?;
                    let image = ctx.input(&extracted)?;
                    let (reply, receive) = mpsc::sync_channel(1);
                    sender
                        .send(WriteRequest { image, reply })
                        .map_err(|_| TaskError::Cancelled)?;
                    // The dispatch worker parks while the caller does the actual
                    // SQLite work. Its CPU/I/O/exclusive grant remains held. Keeping
                    // callbacks here preserves borrowed/non-Send event sink APIs.
                    receive.recv().map_err(|_| TaskError::Cancelled)?;
                    Ok(())
                })],
            )?;
            graph.depends_on(writer.id(), extracted_id)?;
            if let Some(previous) = previous_write {
                graph.depends_on(writer.id(), previous)?;
            }
            previous_write = Some(writer.id());
        }
        drop(sender);
        let run = self
            .runtime
            .submit(graph)
            .with_context(|| format!("CPU feature window allocation estimate {window_memory} bytes could not be admitted before decode"))?;
        let active = ActiveBatch {
            receiver: Some(receiver),
            run,
        };
        loop {
            control.checkpoint()?;
            match active
                .receiver
                .as_ref()
                .unwrap()
                .recv_timeout(Duration::from_millis(5))
            {
                Ok(request) => {
                    write(&request.image)?;
                    // The write callback checks pause/cancel after emitting its
                    // progress event, before acknowledging the next write edge.
                    let _ = request.reply.send(());
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // A failed extraction can drop every writer sender before
                    // its CPU closure has returned. Wait briefly, not a hot loop.
                    if let Some(report) = active.run.wait_timeout(Duration::from_millis(5)) {
                        control.checkpoint()?;
                        if let Some(failure) = report.tasks.iter().find(|task| task.error.is_some())
                        {
                            bail!("{}: {}", failure.name, failure.error.as_ref().unwrap());
                        }
                        if !report.succeeded() {
                            bail!("CPU feature graph did not complete successfully");
                        }
                        return Ok(());
                    }
                }
            }
        }
    }
}

pub(super) struct ExtractionBatch<'a> {
    pub images: &'a [ColmapDatabaseImage],
    pub database_path: &'a Path,
    pub images_dir: &'a Path,
    pub options: &'a SiftExtractionOptions,
}

fn task_error(error: impl std::fmt::Display) -> TaskError {
    TaskError::Failed(format!("{error:#}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Instant;

    // Resource numbers are bytes, not real allocations. Every wait is bounded;
    // cleanup cancels the window and releases both fake extraction and occupancy.
    fn exercise_window(
        ceiling: u64,
        initial: u64,
        external: u64,
        parallel: bool,
        cancel: bool,
    ) -> Result<()> {
        let runtime = Arc::new(Runtime::new(rustscan_taskflow::RuntimeConfig {
            budget: rustscan_taskflow::Budget {
                cpu_threads: 3,
                memory_bytes: ceiling,
                io_slots: 3,
            },
            ..Default::default()
        })?);
        let mut budget = runtime.snapshot()?.budget;
        budget.memory_bytes = initial;
        runtime.set_budget(budget)?;
        let (release_external, external_release) = mpsc::sync_channel(1);
        let (external_started, external_start) = mpsc::sync_channel(1);
        let external_run = if external != 0 {
            let mut graph = TaskGraph::new();
            let mut request = ResourceRequest::cpu(CpuRequest::fixed(1));
            request.working_memory_bytes = external;
            graph.task(
                "external occupancy",
                vec![TaskVariant::cpu("hold", request, move |_| {
                    external_started.send(()).map_err(task_error)?;
                    external_release
                        .recv_timeout(Duration::from_secs(5))
                        .map_err(task_error)?;
                    Ok(())
                })],
            )?;
            let run = runtime.submit(graph)?;
            external_start.recv_timeout(Duration::from_secs(5))?;
            Some(run)
        } else {
            None
        };
        let file = tempfile::NamedTempFile::new()?;
        let images = [1, 2].map(|image_id| ColmapDatabaseImage {
            image_id,
            name: format!("{image_id}"),
            camera_id: 1,
            frame_id: None,
        });
        let adapter = CpuFeatureTaskflow::new(runtime.clone(), 1, 2)?;
        let control = SfmTaskControl::new();
        let finish = Arc::new(AtomicBool::new(!parallel));
        let (started, start) = mpsc::channel();
        let (done, result) = mpsc::sync_channel(1);
        let extraction_finish = finish.clone();
        let extraction_control = control.clone();
        let extract = move |image: &ColmapDatabaseImage| -> Result<ExtractedCpuImage> {
            started.send(image.image_id)?;
            let deadline = Instant::now() + Duration::from_secs(5);
            while !extraction_finish.load(Ordering::SeqCst) {
                extraction_control.checkpoint()?;
                anyhow::ensure!(Instant::now() < deadline, "fake extraction timed out");
                std::thread::sleep(Duration::from_millis(1));
            }
            Ok(ExtractedCpuImage {
                image_id: image.image_id,
                image_name: image.name.clone(),
                keypoints: Vec::new(),
                descriptors: crate::database::ColmapDescriptors::new(
                    crate::database::COLMAP_FEATURE_SIFT,
                    0,
                    128,
                    Vec::new(),
                )?,
                extract_ms: 0.0,
            })
        };
        let outcome = std::thread::scope(|scope| -> Result<()> {
            let worker = scope.spawn(|| {
                let mut committed = Vec::new();
                let status = adapter.extract_estimated_batch(
                    &images,
                    file.path(),
                    &[80, 30],
                    &control,
                    extract,
                    &mut |image| {
                        committed.push(image.image_id);
                        Ok(())
                    },
                );
                let _ = done.send((status, committed));
            });
            let outcome = (|| -> Result<()> {
                if parallel {
                    let mut ids = vec![
                        start.recv_timeout(Duration::from_secs(5))?,
                        start.recv_timeout(Duration::from_secs(5))?,
                    ];
                    ids.sort();
                    anyhow::ensure!(ids == vec![1, 2], "both extractions must run concurrently");
                    anyhow::ensure!(
                        runtime.snapshot()?.memory_bytes == external + 160,
                        "both images must retain the common 80-byte floor"
                    );
                } else {
                    let deadline = Instant::now() + Duration::from_secs(5);
                    while runtime.snapshot()?.pending_tasks < 4 {
                        anyhow::ensure!(Instant::now() < deadline, "window did not queue");
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    anyhow::ensure!(
                        matches!(
                            start.recv_timeout(Duration::from_millis(50)),
                            Err(mpsc::RecvTimeoutError::Timeout)
                        ),
                        "a lighter image overtook the heavy predecessor"
                    );
                    anyhow::ensure!(
                        runtime.snapshot()?.memory_bytes == external,
                        "queued images must not acquire memory"
                    );
                }
                if cancel {
                    control.request_cancel();
                } else {
                    finish.store(true, Ordering::SeqCst);
                    budget.memory_bytes = ceiling;
                    runtime.set_budget(budget)?;
                    let _ = release_external.try_send(());
                }
                let (status, committed) = result.recv_timeout(Duration::from_secs(5))?;
                if cancel {
                    anyhow::ensure!(
                        status.unwrap_err().downcast_ref::<crate::SfmTaskStop>()
                            == Some(&crate::SfmTaskStop::Cancelled),
                        "cancellation must remain typed"
                    );
                    anyhow::ensure!(committed.is_empty(), "cancelled queue must not commit");
                } else {
                    status?;
                    anyhow::ensure!(committed == vec![1, 2], "commit order changed");
                }
                Ok(())
            })();
            finish.store(true, Ordering::SeqCst);
            control.request_cancel();
            let _ = release_external.try_send(());
            worker.join().expect("window worker panicked");
            outcome
        });
        if let Some(run) = external_run {
            run.wait_timeout(Duration::from_secs(5))
                .context("external hold did not drain")?;
        }
        outcome?;
        let usage = runtime.snapshot()?;
        assert_eq!(
            (
                usage.cpu_threads,
                usage.memory_bytes,
                usage.io_slots,
                usage.pending_tasks
            ),
            (0, 0, 0, 0)
        );
        Ok(())
    }

    #[test]
    fn memory_heterogeneous_window_with_external_occupancy_does_not_deadlock() -> Result<()> {
        exercise_window(100, 100, 40, false, false)
    }
    #[test]
    fn memory_uniform_window_floor_preserves_parallel_extraction() -> Result<()> {
        exercise_window(200, 200, 40, true, false)
    }
    #[test]
    fn memory_nonzero_transient_budget_waits_for_recovery() -> Result<()> {
        exercise_window(100, 50, 0, false, false)
    }
    #[test]
    fn memory_nonzero_transient_budget_cancels_without_decode() -> Result<()> {
        exercise_window(100, 50, 0, false, true)
    }
}

struct WriteRequest {
    image: Artifact<ExtractedCpuImage>,
    reply: mpsc::SyncSender<()>,
}

struct ActiveBatch {
    receiver: Option<mpsc::Receiver<WriteRequest>>,
    run: RunHandle,
}

impl Drop for ActiveBatch {
    fn drop(&mut self) {
        // Also runs on callback panic. Disconnect first so a writer waiting for
        // acknowledgement can exit, then cancel queued work and drain native
        // SIFT calls before the caller may remove its database/input files.
        self.receiver.take();
        self.run.cancel();
        self.run.wait();
    }
}
