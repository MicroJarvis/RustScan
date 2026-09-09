use std::collections::VecDeque;
use std::sync::{
    mpsc::{self, Receiver, Sender, SyncSender},
    Arc, Condvar, Mutex,
};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::resource::Ledger;
use crate::task::{panic_error, ErasedCompletion, MemoryLease, StoredArtifact, Value};
use crate::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    Pending,
    Ready,
    Running,
    Succeeded,
    Failed,
    Cancelled,
    DependencyFailed,
}
impl TaskStatus {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Succeeded | Self::Failed | Self::Cancelled | Self::DependencyFailed
        )
    }
}

#[derive(Debug, Clone)]
pub struct TaskReport {
    pub id: TaskId,
    pub name: String,
    pub status: TaskStatus,
    pub error: Option<TaskError>,
    pub grant: Option<ExecutionGrant>,
    pub queue_time: Option<Duration>,
    pub execution_time: Option<Duration>,
}

#[derive(Debug, Clone)]
pub struct RunReport {
    pub graph_id: u64,
    pub tasks: Vec<TaskReport>,
    pub elapsed: Duration,
    pub dropped_events: usize,
}
impl RunReport {
    pub fn succeeded(&self) -> bool {
        self.tasks.iter().all(|t| t.status == TaskStatus::Succeeded)
    }
}

#[derive(Debug, Clone)]
pub enum Event {
    Submitted {
        graph_id: u64,
        tasks: usize,
    },
    Started {
        task: TaskId,
        grant: ExecutionGrant,
        queue_time: Duration,
    },
    /// CPU submission work and its scoped pool have finished; a GPU/backend
    /// may still be running. Other resource reservations remain held.
    SubmissionFinished {
        task: TaskId,
    },
    Finished {
        task: TaskId,
        status: TaskStatus,
        error: Option<TaskError>,
        execution_time: Option<Duration>,
    },
    WorkflowFinished {
        graph_id: u64,
    },
}

pub(crate) struct SharedRun {
    report: Mutex<Option<RunReport>>,
    ready: Condvar,
    cancel: CancellationToken,
}

pub struct RunHandle {
    id: u64,
    sender: Sender<Command>,
    shared: Arc<SharedRun>,
    events: Mutex<Receiver<Event>>,
}
impl RunHandle {
    pub fn graph_id(&self) -> u64 {
        self.id
    }
    pub fn cancel(&self) {
        self.shared.cancel.cancel();
        let _ = self.sender.send(Command::Cancel(self.id));
    }
    pub fn wait(&self) -> RunReport {
        let report = self.shared.report.lock().unwrap();
        self.shared
            .ready
            .wait_while(report, |r| r.is_none())
            .unwrap()
            .as_ref()
            .unwrap()
            .clone()
    }
    pub fn wait_timeout(&self, timeout: Duration) -> Option<RunReport> {
        let report = self.shared.report.lock().unwrap();
        self.shared
            .ready
            .wait_timeout_while(report, timeout, |r| r.is_none())
            .unwrap()
            .0
            .clone()
    }
    pub fn try_event(&self) -> Option<Event> {
        self.events.lock().unwrap().try_recv().ok()
    }
    /// Only published outputs are visible. Cancellation/failure never publishes
    /// partial results. A handle from another workflow is rejected.
    pub fn output<T: Send + Sync + 'static>(
        &self,
        task: &TaskHandle<T>,
    ) -> TaskResult<Artifact<T>> {
        if task.id.graph != self.id {
            return Err(TaskError::InvalidArtifact);
        }
        task.read()
    }
}

/// One scheduler shared by all workflows. Dropping it cancels outstanding work
/// and waits for cooperative backend completion. Never drop it on a task worker
/// or GUI thread while a non-cooperative task is running.
pub struct Runtime {
    pub(crate) sender: Sender<Command>,
    thread: Option<JoinHandle<()>>,
}
impl Runtime {
    pub fn new(config: RuntimeConfig) -> Result<Self, Error> {
        config.validate()?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.budget.cpu_threads)
            .thread_name(|i| format!("taskflow-dispatch-{i}"))
            .build()
            .map_err(|e| Error::Executor(e.to_string()))?;
        let (sender, receiver) = mpsc::channel();
        let coordinator_sender = sender.clone();
        let thread = std::thread::Builder::new()
            .name("taskflow-scheduler".into())
            .spawn(move || Scheduler::new(config, pool, coordinator_sender).run(receiver))
            .map_err(|e| Error::Executor(e.to_string()))?;
        Ok(Self {
            sender,
            thread: Some(thread),
        })
    }

    pub fn submit(&self, graph: TaskGraph) -> Result<RunHandle, Error> {
        graph.validate()?;
        let id = graph.id;
        let shared = Arc::new(SharedRun {
            report: Mutex::new(None),
            ready: Condvar::new(),
            cancel: CancellationToken::default(),
        });
        let (reply, receive) = mpsc::sync_channel(1);
        let (event_tx, event_rx) = mpsc::channel();
        // The coordinator supplies the bounded event receiver after admission.
        self.sender
            .send(Command::Submit(graph, shared.clone(), reply, event_tx))
            .map_err(|_| Error::Closed)?;
        receive.recv().map_err(|_| Error::Closed)??;
        let events = event_rx.recv().map_err(|_| Error::Closed)?;
        Ok(RunHandle {
            id,
            sender: self.sender.clone(),
            shared,
            events: Mutex::new(events),
        })
    }

    pub fn snapshot(&self) -> Result<ResourceSnapshot, Error> {
        let (tx, rx) = mpsc::sync_channel(1);
        self.sender
            .send(Command::Snapshot(tx))
            .map_err(|_| Error::Closed)?;
        rx.recv().map_err(|_| Error::Closed)
    }

    /// Changes admission limits, not running tasks. Limits cannot exceed the
    /// original physical/application ceiling. Zero temporarily pauses a lane.
    pub fn set_budget(&self, budget: Budget) -> Result<(), Error> {
        set_budget(&self.sender, budget)
    }

    /// Cooperatively cancel and join all workflows. This may wait indefinitely
    /// for a backend that violates its completion contract.
    pub fn shutdown(mut self) {
        self.stop();
    }
    fn stop(&mut self) {
        if let Some(thread) = self.thread.take() {
            let _ = self.sender.send(Command::Shutdown);
            let _ = thread.join();
        }
    }
}
impl Drop for Runtime {
    fn drop(&mut self) {
        self.stop();
    }
}

pub(crate) fn set_budget(sender: &Sender<Command>, budget: Budget) -> Result<(), Error> {
    let (tx, rx) = mpsc::sync_channel(1);
    sender
        .send(Command::Budget(budget, tx))
        .map_err(|_| Error::Closed)?;
    rx.recv().map_err(|_| Error::Closed)?
}

pub(crate) enum Command {
    Submit(
        TaskGraph,
        Arc<SharedRun>,
        SyncSender<Result<(), Error>>,
        Sender<Receiver<Event>>,
    ),
    Cancel(u64),
    Completed(TaskId, TaskResult<Value>),
    Returned(TaskId, Option<TaskError>),
    ReleaseOutput(u64, Option<(DeviceId, u64)>),
    Snapshot(SyncSender<ResourceSnapshot>),
    Budget(Budget, SyncSender<Result<(), Error>>),
    Shutdown,
}

struct Running {
    grant: ExecutionGrant,
    started: Instant,
    returned: bool,
    submission_error: Option<TaskError>,
    result: Option<TaskResult<Value>>,
}

struct Job {
    graph: TaskGraph,
    shared: Arc<SharedRun>,
    events: SyncSender<Event>,
    dropped_events: usize,
    reports: Vec<TaskReport>,
    consumers: Vec<Vec<usize>>,
    remaining_deps: Vec<usize>,
    ready: VecDeque<usize>,
    bypasses: Vec<usize>,
    running: Vec<Option<Running>>,
    remaining: usize,
    created: Instant,
}
impl Job {
    fn new(graph: TaskGraph, shared: Arc<SharedRun>, events: SyncSender<Event>) -> Self {
        let len = graph.len();
        let mut consumers = vec![Vec::new(); len];
        let mut ready = VecDeque::new();
        let reports = graph
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| {
                for &dep in &n.deps {
                    consumers[dep].push(i);
                }
                let status = if n.deps.is_empty() {
                    ready.push_back(i);
                    TaskStatus::Ready
                } else {
                    TaskStatus::Pending
                };
                TaskReport {
                    id: TaskId {
                        graph: graph.id,
                        index: i,
                    },
                    name: n.name.clone(),
                    status,
                    error: None,
                    grant: None,
                    queue_time: None,
                    execution_time: None,
                }
            })
            .collect();
        Self {
            remaining_deps: graph.nodes.iter().map(|n| n.deps.len()).collect(),
            graph,
            shared,
            events,
            dropped_events: 0,
            reports,
            consumers,
            ready,
            bypasses: vec![0; len],
            running: (0..len).map(|_| None).collect(),
            remaining: len,
            created: Instant::now(),
        }
    }
    fn emit(&mut self, event: Event) {
        if self.events.try_send(event).is_err() {
            self.dropped_events += 1;
        }
    }
    fn mark(
        &mut self,
        i: usize,
        status: TaskStatus,
        error: Option<TaskError>,
        elapsed: Option<Duration>,
    ) {
        let report = &mut self.reports[i];
        report.status = status;
        report.error = error.clone();
        report.execution_time = elapsed;
        self.remaining -= 1;
        // Release captures of skipped implementations/dependencies promptly.
        self.graph.nodes[i].variants.clear();
        self.emit(Event::Finished {
            task: TaskId {
                graph: self.graph.id,
                index: i,
            },
            status,
            error,
            execution_time: elapsed,
        });
    }
    fn skip_descendants(&mut self, root: usize) -> usize {
        let mut queue: VecDeque<_> = self.consumers[root].iter().copied().collect();
        let mut skipped = 0;
        while let Some(i) = queue.pop_front() {
            if self.reports[i].status.is_terminal() {
                continue;
            }
            self.mark(i, TaskStatus::DependencyFailed, None, None);
            skipped += 1;
            queue.extend(self.consumers[i].iter().copied());
        }
        skipped
    }
    fn cancel_pending(&mut self) -> usize {
        self.shared.cancel.cancel();
        let mut count = 0;
        for i in 0..self.reports.len() {
            if matches!(
                self.reports[i].status,
                TaskStatus::Pending | TaskStatus::Ready
            ) {
                self.mark(i, TaskStatus::Cancelled, Some(TaskError::Cancelled), None);
                count += 1;
            }
        }
        self.ready.clear();
        count
    }
}

struct Scheduler {
    ledger: Ledger,
    pool: rayon::ThreadPool,
    sender: Sender<Command>,
    jobs: VecDeque<Job>,
    stopping: bool,
}
impl Scheduler {
    fn new(config: RuntimeConfig, pool: rayon::ThreadPool, sender: Sender<Command>) -> Self {
        Self {
            ledger: Ledger::new(config),
            pool,
            sender,
            jobs: VecDeque::new(),
            stopping: false,
        }
    }
    fn run(mut self, receiver: Receiver<Command>) {
        loop {
            self.reap();
            if self.stopping && self.jobs.is_empty() {
                break;
            }
            // Bounded draining keeps submission floods from starving scheduling.
            for _ in 0..256 {
                match receiver.try_recv() {
                    Ok(command) => self.command(command),
                    Err(_) => break,
                }
            }
            self.reap();
            if self.stopping && self.jobs.is_empty() {
                break;
            }
            if self.dispatch_one() {
                continue;
            }
            match receiver.recv() {
                Ok(command) => self.command(command),
                Err(_) => break,
            }
        }
    }
    fn command(&mut self, command: Command) {
        match command {
            Command::Submit(graph, shared, reply, event_reply) => {
                let result = if self.stopping {
                    Err(Error::Closed)
                } else if graph.len()
                    > self
                        .ledger
                        .config
                        .max_pending_tasks
                        .saturating_sub(self.ledger.usage.pending_tasks)
                {
                    Err(Error::QueueFull)
                } else {
                    graph.validate_resources(&self.ledger.config)
                };
                if result.is_ok() {
                    let (tx, rx) = mpsc::sync_channel(self.ledger.config.event_capacity);
                    self.ledger.usage.pending_tasks += graph.len();
                    let mut job = Job::new(graph, shared, tx);
                    job.emit(Event::Submitted {
                        graph_id: job.graph.id,
                        tasks: job.graph.len(),
                    });
                    self.jobs.push_back(job);
                    let _ = event_reply.send(rx);
                }
                let _ = reply.send(result);
            }
            Command::Cancel(id) => {
                if let Some(job) = self.jobs.iter_mut().find(|j| j.graph.id == id) {
                    self.ledger.usage.pending_tasks -= job.cancel_pending();
                }
            }
            Command::Completed(id, result) => {
                if let Some(job) = self.jobs.iter_mut().find(|j| j.graph.id == id.graph) {
                    if let Some(running) = job.running[id.index].as_mut() {
                        running.result = Some(result);
                    }
                }
                self.finish(id);
            }
            Command::Returned(id, error) => {
                if let Some(job) = self.jobs.iter_mut().find(|j| j.graph.id == id.graph) {
                    if let Some(running) = job.running[id.index].as_mut() {
                        if !running.returned {
                            running.returned = true;
                            running.submission_error = error;
                            self.ledger.release_cpu(&running.grant);
                            job.emit(Event::SubmissionFinished { task: id });
                        }
                    }
                }
                self.finish(id);
            }
            Command::ReleaseOutput(memory, gpu) => self.ledger.release_output(memory, gpu),
            Command::Snapshot(reply) => {
                let _ = reply.send(self.ledger.usage.clone());
            }
            Command::Budget(budget, reply) => {
                let ceiling = self.ledger.config.budget;
                let result = if budget.cpu_threads > ceiling.cpu_threads
                    || budget.memory_bytes > ceiling.memory_bytes
                    || budget.io_slots > ceiling.io_slots
                {
                    Err(Error::Invalid("budget exceeds configured ceiling".into()))
                } else {
                    self.ledger.usage.budget = budget;
                    Ok(())
                };
                let _ = reply.send(result);
            }
            Command::Shutdown => {
                self.stopping = true;
                for job in &mut self.jobs {
                    self.ledger.usage.pending_tasks -= job.cancel_pending();
                }
            }
        }
    }

    fn dispatch_one(&mut self) -> bool {
        if self.stopping {
            return false;
        }
        // Cancel requests become visible even before their message is consumed.
        for job in &mut self.jobs {
            if job.shared.cancel.is_cancelled() {
                self.ledger.usage.pending_tasks -= job.cancel_pending();
            }
        }
        let protected = self.jobs.iter().find_map(|job| {
            job.ready
                .iter()
                .find(|&&i| {
                    // Only wait for CPU already held by running submissions.
                    // Memory may need a ready consumer to run before it is freed.
                    job.bypasses[i] >= self.ledger.config.max_bypasses
                        && job.graph.nodes[i]
                            .variants
                            .iter()
                            .any(|variant| self.ledger.can_reserve_cpu(&variant.resources))
                })
                .map(|&i| (job.graph.id, i))
        });
        let mut selection = None;
        'jobs: for (j, job) in self.jobs.iter().enumerate() {
            for (position, &i) in job.ready.iter().enumerate() {
                if protected.is_some_and(|key| key != (job.graph.id, i)) {
                    continue;
                }
                for (v, variant) in job.graph.nodes[i].variants.iter().enumerate() {
                    if let Some(grant) = self.ledger.fits(&variant.name, &variant.resources) {
                        selection = Some((j, position, i, v, grant));
                        break 'jobs;
                    }
                }
            }
        }
        let Some((j, position, i, v, grant)) = selection else {
            return false;
        };
        // Count actual overtakes rather than polling iterations.
        for (job_index, job) in self.jobs.iter_mut().enumerate() {
            for &node in &job.ready {
                if job_index == j && node == i {
                    break;
                }
                job.bypasses[node] = job.bypasses[node].saturating_add(1);
            }
            if job_index == j {
                break;
            }
        }
        let mut job = self.jobs.remove(j).unwrap();
        job.ready.remove(position);
        let node = &mut job.graph.nodes[i];
        let work = node.variants[v].work.take().unwrap();
        node.variants.clear();
        let deps = node
            .deps
            .iter()
            .map(|&index| TaskId {
                graph: job.graph.id,
                index,
            })
            .collect();
        let id = TaskId {
            graph: job.graph.id,
            index: i,
        };
        let context = TaskContext::new(grant.clone(), deps, job.shared.cancel.clone());
        self.ledger.acquire(&grant);
        let queue_time = job.created.elapsed();
        job.reports[i].status = TaskStatus::Running;
        job.reports[i].grant = Some(grant.clone());
        job.reports[i].queue_time = Some(queue_time);
        job.running[i] = Some(Running {
            grant: grant.clone(),
            started: Instant::now(),
            returned: false,
            submission_error: None,
            result: None,
        });
        job.emit(Event::Started {
            task: id,
            grant,
            queue_time,
        });
        self.jobs.push_back(job);
        let sender = self.sender.clone();
        self.pool.spawn(move || {
            let done = ErasedCompletion::new(sender.clone(), id);
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if context.cancel.is_cancelled() {
                    done.complete(Err(TaskError::Cancelled));
                } else {
                    work(&context, done);
                }
            }));
            // The compute pool and its workers must be gone before Returned.
            drop(context);
            let _ = sender.send(Command::Returned(id, outcome.err().map(panic_error)));
        });
        true
    }

    fn finish(&mut self, id: TaskId) {
        let Some(job) = self.jobs.iter_mut().find(|j| j.graph.id == id.graph) else {
            return;
        };
        let Some(running) = job.running[id.index].as_ref() else {
            return;
        };
        if !running.returned || running.result.is_none() {
            return;
        }
        let running = job.running[id.index].take().unwrap();
        let elapsed = running.started.elapsed();
        let result = if let Some(error) = running.submission_error {
            Err(error)
        } else if job.shared.cancel.is_cancelled() {
            Err(TaskError::Cancelled)
        } else {
            running.result.unwrap()
        };
        self.ledger.finish(&running.grant, result.is_ok());
        let (status, error) = match result {
            Ok(value) => {
                let lease = Arc::new(MemoryLease {
                    sender: self.sender.clone(),
                    memory: running.grant.output_memory_bytes,
                    gpu: running
                        .grant
                        .gpu
                        .as_ref()
                        .map(|g| (g.device, g.output_memory_bytes)),
                });
                // Runtime relinquishes its copy immediately; consumers and user
                // handles, not completed job records, determine output lifetime.
                let slot = std::mem::replace(
                    &mut job.graph.nodes[id.index].slot,
                    Arc::new(std::sync::OnceLock::new()),
                );
                let _ = slot.set(StoredArtifact { value, lease });
                (TaskStatus::Succeeded, None)
            }
            Err(TaskError::Cancelled) => (TaskStatus::Cancelled, Some(TaskError::Cancelled)),
            Err(error) => (TaskStatus::Failed, Some(error)),
        };
        job.mark(id.index, status, error, Some(elapsed));
        self.ledger.usage.pending_tasks -= 1;
        if status == TaskStatus::Succeeded {
            for &consumer in &job.consumers[id.index] {
                job.remaining_deps[consumer] -= 1;
                if job.remaining_deps[consumer] == 0
                    && job.reports[consumer].status == TaskStatus::Pending
                {
                    job.reports[consumer].status = TaskStatus::Ready;
                    job.ready.push_back(consumer);
                }
            }
        } else {
            self.ledger.usage.pending_tasks -= job.skip_descendants(id.index);
        }
    }

    fn reap(&mut self) {
        let mut i = 0;
        while i < self.jobs.len() {
            if self.jobs[i].remaining != 0 {
                i += 1;
                continue;
            }
            let mut job = self.jobs.remove(i).unwrap();
            job.emit(Event::WorkflowFinished {
                graph_id: job.graph.id,
            });
            let report = RunReport {
                graph_id: job.graph.id,
                tasks: job.reports,
                elapsed: job.created.elapsed(),
                dropped_events: job.dropped_events,
            };
            *job.shared.report.lock().unwrap() = Some(report);
            job.shared.ready.notify_all();
        }
    }
}
