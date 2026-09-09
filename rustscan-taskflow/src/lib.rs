#![doc = include_str!("../README.md")]

mod graph;
mod pressure;
mod resource;
mod runtime;
mod task;
#[cfg(feature = "wgpu-backend")]
mod wgpu_backend;
#[cfg(feature = "wgpu-backend")]
pub use wgpu_backend::WgpuBackend;

pub use graph::TaskGraph;
pub use pressure::*;
pub use resource::{
    Budget, CpuRequest, DeviceId, ExecutionGrant, GpuCapacity, GpuGrant, GpuRequest, GpuUsage,
    ResourceRequest, ResourceSnapshot, RuntimeConfig,
};
pub use runtime::{Event, RunHandle, RunReport, Runtime, TaskReport, TaskStatus};
pub use task::{
    Artifact, CancellationToken, Completion, TaskContext, TaskError, TaskHandle, TaskId,
    TaskResult, TaskVariant,
};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("invalid configuration: {0}")]
    Invalid(String),
    #[error("task handle belongs to another graph")]
    ForeignTask,
    #[error("graph contains a dependency cycle")]
    Cycle,
    #[error("no implementation fits configured resources for task {0}")]
    Unschedulable(String),
    #[error("task queue capacity exceeded")]
    QueueFull,
    #[error("runtime is closed")]
    Closed,
    #[error("executor could not be created: {0}")]
    Executor(String),
}
