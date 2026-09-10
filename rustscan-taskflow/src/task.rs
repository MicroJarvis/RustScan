use std::any::Any;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::Sender,
    Arc, OnceLock,
};
use std::time::Duration;

use crate::runtime::Command;
use crate::{DeviceId, ExecutionGrant, ResourceRequest};

pub type TaskResult<T> = Result<T, TaskError>;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TaskError {
    #[error("{0}")]
    Failed(String),
    #[error("task panicked: {0}")]
    Panicked(String),
    #[error("task cancelled")]
    Cancelled,
    #[error("asynchronous completion was dropped without a result")]
    CompletionDropped,
    #[error("artifact is unavailable or is not a declared dependency")]
    InvalidArtifact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId {
    pub(crate) graph: u64,
    pub(crate) index: usize,
}

impl TaskId {
    pub fn index(self) -> usize {
        self.index
    }
    pub fn graph_id(self) -> u64 {
        self.graph
    }
}

#[derive(Clone, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
    pub fn check(&self) -> TaskResult<()> {
        if self.is_cancelled() {
            Err(TaskError::Cancelled)
        } else {
            Ok(())
        }
    }
    pub(crate) fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
}

pub(crate) type Value = Arc<dyn Any + Send + Sync>;
pub(crate) type Slot = Arc<OnceLock<StoredArtifact>>;

pub(crate) struct MemoryLease {
    pub sender: Sender<Command>,
    pub memory: u64,
    pub gpu: Option<(DeviceId, u64)>,
}

impl Drop for MemoryLease {
    fn drop(&mut self) {
        let _ = self
            .sender
            .send(Command::ReleaseOutput(self.memory, self.gpu));
    }
}

pub(crate) struct StoredArtifact {
    pub value: Value,
    pub lease: Arc<MemoryLease>,
}

/// A shared immutable output. Cloning it keeps its reservation alive; extracting
/// a bare Arc is deliberately not supported because that would lose accounting.
pub struct Artifact<T> {
    value: Arc<T>,
    _lease: Arc<MemoryLease>,
}

impl<T> Clone for Artifact<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            _lease: self._lease.clone(),
        }
    }
}

impl<T> Deref for Artifact<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.value
    }
}

pub struct TaskHandle<T> {
    pub(crate) id: TaskId,
    pub(crate) slot: Slot,
    pub(crate) marker: PhantomData<fn() -> T>,
}

impl<T> Clone for TaskHandle<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            slot: self.slot.clone(),
            marker: PhantomData,
        }
    }
}

impl<T> TaskHandle<T> {
    pub fn id(&self) -> TaskId {
        self.id
    }
}

impl<T: Send + Sync + 'static> TaskHandle<T> {
    pub(crate) fn read(&self) -> TaskResult<Artifact<T>> {
        let artifact = self.slot.get().ok_or(TaskError::InvalidArtifact)?;
        Ok(Artifact {
            value: artifact
                .value
                .clone()
                .downcast::<T>()
                .map_err(|_| TaskError::InvalidArtifact)?,
            _lease: artifact.lease.clone(),
        })
    }
}

/// Valid only during the invocation of a task's CPU/submission closure. It is
/// intentionally borrowed, so an async backend cannot retain CPU privileges
/// after returning from submission. Clone the cancellation token if needed.
pub struct TaskContext {
    pub(crate) grant: ExecutionGrant,
    pub(crate) dependencies: Vec<TaskId>,
    pub(crate) cancel: CancellationToken,
    dependency_wait_time: Duration,
    resource_wait_time: Duration,
    queue_time: Duration,
    pool: OnceLock<Result<rayon::ThreadPool, String>>,
}

impl TaskContext {
    pub(crate) fn new(
        grant: ExecutionGrant,
        dependencies: Vec<TaskId>,
        cancel: CancellationToken,
        dependency_wait_time: Duration,
        resource_wait_time: Duration,
        queue_time: Duration,
    ) -> Self {
        Self {
            grant,
            dependencies,
            cancel,
            dependency_wait_time,
            resource_wait_time,
            queue_time,
            pool: OnceLock::new(),
        }
    }
    pub fn grant(&self) -> &ExecutionGrant {
        &self.grant
    }
    pub fn queue_time(&self) -> Duration {
        self.queue_time
    }

    /// Time spent waiting for prerequisite tasks after workflow submission.
    pub fn dependency_wait_time(&self) -> Duration {
        self.dependency_wait_time
    }

    /// Time spent ready but waiting for resource admission.
    pub fn resource_wait_time(&self) -> Duration {
        self.resource_wait_time
    }

    pub fn cancellation(&self) -> CancellationToken {
        self.cancel.clone()
    }
    pub fn check_cancelled(&self) -> TaskResult<()> {
        self.cancel.check()
    }
    pub fn input<T: Send + Sync + 'static>(
        &self,
        handle: &TaskHandle<T>,
    ) -> TaskResult<Artifact<T>> {
        if !self.dependencies.contains(&handle.id) {
            return Err(TaskError::InvalidArtifact);
        }
        handle.read()
    }

    /// Runs Rayon work in an isolated, lazily created pool of exactly the
    /// granted size. The dispatch thread waits; it is not an extra compute
    /// worker. The pool is destroyed before the CPU allowance is released.
    /// Do not let detached Rayon work escape this scope or use the global pool.
    pub fn parallel<R: Send>(&self, work: impl FnOnce() -> R + Send) -> TaskResult<R> {
        self.check_cancelled()?;
        let pool = self.pool.get_or_init(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(self.grant.cpu_threads)
                .thread_name(|i| format!("taskflow-compute-{i}"))
                .build()
                .map_err(|e| e.to_string())
        });
        match pool {
            Ok(pool) => Ok(pool.install(work)),
            Err(error) => Err(TaskError::Failed(error.clone())),
        }
    }
}

/// One-shot external completion. Keep this token alive until the device/backend
/// has really stopped accessing all task resources, even after cancellation.
/// Dropping it reports failure; it does NOT cancel hardware commands.
pub struct Completion<T> {
    pub(crate) inner: Option<ErasedCompletion>,
    pub(crate) marker: PhantomData<fn(T)>,
}

impl<T: Send + Sync + 'static> Completion<T> {
    pub fn complete(mut self, result: TaskResult<T>) {
        if let Some(inner) = self.inner.take() {
            inner.complete(result.map(|v| Arc::new(v) as Value));
        }
    }
}

pub(crate) struct ErasedCompletion {
    sender: Sender<Command>,
    id: TaskId,
    sent: bool,
}

impl ErasedCompletion {
    pub fn new(sender: Sender<Command>, id: TaskId) -> Self {
        Self {
            sender,
            id,
            sent: false,
        }
    }
    pub fn complete(mut self, result: TaskResult<Value>) {
        self.sent = true;
        let _ = self.sender.send(Command::Completed(self.id, result));
    }
}

impl Drop for ErasedCompletion {
    fn drop(&mut self) {
        if !self.sent {
            let _ = self.sender.send(Command::Completed(
                self.id,
                Err(TaskError::CompletionDropped),
            ));
        }
    }
}

pub(crate) type Work = Box<dyn FnOnce(&TaskContext, ErasedCompletion) + Send>;

pub struct TaskVariant<T> {
    pub(crate) name: String,
    pub(crate) resources: ResourceRequest,
    pub(crate) work: Work,
    marker: PhantomData<fn() -> T>,
}

impl<T: Send + Sync + 'static> TaskVariant<T> {
    pub fn cpu(
        name: impl Into<String>,
        resources: ResourceRequest,
        work: impl FnOnce(&TaskContext) -> TaskResult<T> + Send + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            resources,
            work: Box::new(move |context, done| {
                let result =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(context)))
                        .unwrap_or_else(|panic| Err(panic_error(panic)));
                done.complete(result.map(|v| Arc::new(v) as Value));
            }),
            marker: PhantomData,
        }
    }

    /// The closure submits work and returns promptly. `done` may be moved into
    /// an existing wgpu completion callback or another external backend; this
    /// crate neither creates GPU devices nor assumes submission means completion.
    pub fn asynchronous(
        name: impl Into<String>,
        resources: ResourceRequest,
        submit: impl FnOnce(&TaskContext, Completion<T>) + Send + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            resources,
            work: Box::new(move |context, done| {
                submit(
                    context,
                    Completion {
                        inner: Some(done),
                        marker: PhantomData,
                    },
                )
            }),
            marker: PhantomData,
        }
    }
}

pub(crate) fn panic_error(payload: Box<dyn Any + Send>) -> TaskError {
    let message = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).into()))
        .unwrap_or_else(|| "non-string panic payload".into());
    TaskError::Panicked(message)
}
