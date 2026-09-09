use super::{BundleAdjustmentOptions, BundleAdjustmentReport};
use crate::task::SfmTaskControl;
use crate::types::{ImageFrame, Reconstruction};
use anyhow::{bail, Result};
use rustscan_taskflow::Runtime;
use std::sync::Arc;

#[cfg(feature = "ceres-ba")]
use rustscan_taskflow::{CpuRequest, ExecutionGrant, ResourceRequest};
#[cfg(all(test, feature = "ceres-ba"))]
use rustscan_taskflow::{TaskGraph, TaskVariant};
#[cfg(all(test, feature = "ceres-ba"))]
use std::{
    sync::mpsc,
    time::{Duration, Instant},
};

#[derive(Debug, Clone)]
pub struct BaSchedulingReport {
    pub granted_cpu_threads: usize,
    pub queue_ms: f64,
}

/// Shared, opt-in admission for the entire Ceres BA call (setup through cleanup).
/// No reconstruction/image clones or new per-BA compute pool are required.
/// Memory is a caller-supplied scratch estimate; existing model storage is not
/// counted. Native BLAS/OpenMP pools must be limited separately by the host.
#[derive(Clone)]
pub struct CeresBaTaskflow {
    runtime: Arc<Runtime>,
    max_cpu_threads: usize,
    working_memory_bytes: u64,
    control: SfmTaskControl,
}

impl std::fmt::Debug for CeresBaTaskflow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CeresBaTaskflow")
            .field("max_cpu_threads", &self.max_cpu_threads)
            .field("working_memory_bytes", &self.working_memory_bytes)
            .field("control", &self.control)
            .finish_non_exhaustive()
    }
}

impl CeresBaTaskflow {
    pub fn new(
        runtime: Arc<Runtime>,
        max_cpu_threads: usize,
        working_memory_bytes: u64,
    ) -> Result<Self> {
        if max_cpu_threads == 0 || max_cpu_threads > i32::MAX as usize || working_memory_bytes == 0
        {
            bail!("Ceres Taskflow requires 1..=i32::MAX CPU threads and positive scratch memory");
        }
        Ok(Self {
            runtime,
            max_cpu_threads,
            working_memory_bytes,
            control: SfmTaskControl::new(),
        })
    }

    pub(crate) fn stage_taskflow(&self) -> Result<crate::SfmTaskflow> {
        crate::SfmTaskflow::new(self.runtime.clone(), self.working_memory_bytes)
    }

    /// Bind a workflow's control. Clones still share the original Runtime.
    pub fn with_control(mut self, control: SfmTaskControl) -> Self {
        self.control = control;
        self
    }

    pub fn refine(
        &self,
        frames: &[ImageFrame],
        reconstruction: &mut Reconstruction,
        options: BundleAdjustmentOptions,
    ) -> Result<Option<BundleAdjustmentReport>> {
        self.control.checkpoint()?;
        #[cfg(not(feature = "ceres-ba"))]
        {
            let _ = (frames, reconstruction, options, &self.runtime);
            bail!("Ceres Taskflow requires the ceres-ba feature");
        }
        #[cfg(feature = "ceres-ba")]
        {
            if !options.loss_function.has_colmap_valid_scale() {
                return Ok(None);
            }
            let threads = self.preferred_threads(reconstruction, &options);
            self.run_admitted(threads, |grant, queue_ms| {
                let mut options = options;
                options.taskflow = None;
                options.num_threads = grant.cpu_threads as isize;
                let mut result = super::ceres_problem::solve_bundle_adjustment_ceres(
                    frames,
                    reconstruction,
                    options,
                    Some(&self.control),
                );
                if let Some(report) = &mut result {
                    report.scheduling = Some(BaSchedulingReport {
                        granted_cpu_threads: grant.cpu_threads,
                        queue_ms,
                    });
                } else {
                    // A stop during native Solve is detected before writeback.
                    self.control.checkpoint()?;
                }
                Ok(result)
            })
        }
    }

    #[cfg(feature = "ceres-ba")]
    fn preferred_threads(
        &self,
        reconstruction: &Reconstruction,
        options: &BundleAdjustmentOptions,
    ) -> usize {
        // Conservative upper bound, not a second observation collection pass.
        // Filtering can make Ceres use fewer threads than this reservation.
        let tracks = if options.point_ids.is_some() || options.constant_point_ids.is_some() {
            options
                .point_ids
                .iter()
                .flatten()
                .chain(options.constant_point_ids.iter().flatten())
                .filter_map(|&id| reconstruction.points.get(id))
                .fold(0usize, |sum, point| sum.saturating_add(point.track.len()))
        } else {
            reconstruction
                .points
                .iter()
                .fold(0usize, |sum, point| sum.saturating_add(point.track.len()))
        };
        let residual_bound = tracks
            .saturating_mul(2)
            .saturating_add(options.pose_priors.len().saturating_mul(3));
        if residual_bound < options.min_num_residuals_for_multi_threading {
            return 1;
        }
        if options.num_threads > 0 {
            self.max_cpu_threads.min(options.num_threads as usize)
        } else {
            self.max_cpu_threads
        }
    }

    #[cfg(feature = "ceres-ba")]
    fn run_admitted<R>(
        &self,
        threads: usize,
        work: impl FnOnce(&ExecutionGrant, f64) -> Result<R>,
    ) -> Result<R> {
        let mut request = ResourceRequest::cpu(CpuRequest::scalable(1, threads, threads));
        request.working_memory_bytes = self.working_memory_bytes;
        crate::execution::admit(&self.runtime, "Ceres BA", request, &self.control, work)
    }
}

#[cfg(all(test, feature = "ceres-ba"))]
#[path = "taskflow_tests.rs"]
mod tests;
