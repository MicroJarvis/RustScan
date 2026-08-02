use super::{WgpuContext, WgpuPnpModelScorer};
use anyhow::{bail, Result};
use bytemuck::{Pod, Zeroable};
use rustslam::tracker::PnPModelScorer;
use rustslam::SE3;
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuPnpFocalModel {
    pub(crate) row0: [f32; 4],
    pub(crate) row1: [f32; 4],
    pub(crate) row2: [f32; 4],
    pub(crate) log_focal_and_padding: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuPnpFocalResult {
    pub(crate) selected_model: u32,
    pub(crate) inliers: u32,
    pub(crate) valid: u32,
    pub(crate) pad: u32,
    pub(crate) residual_sum: f32,
    pub(crate) focal: f32,
    pub(crate) pad1: f32,
    pub(crate) pad2: f32,
}

pub(crate) const PNP_FOCAL_SHADER: &str = include_str!("shaders/pnp_focal.wgsl");

#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuPnpFocalCandidate {
    pub(crate) pose: SE3,
    pub(crate) focal: f32,
}

pub(crate) struct WgpuPnPFocalScorer {
    scorer: WgpuPnpModelScorer,
    centered_points: Vec<[f32; 2]>,
    object_points: Vec<[f32; 3]>,
    threshold_px: f32,
}

impl WgpuPnPFocalScorer {
    pub(crate) fn from_context(context: Arc<WgpuContext>) -> Result<Self> {
        Ok(Self { scorer: WgpuPnpModelScorer::from_context(context)?, centered_points: Vec::new(), object_points: Vec::new(), threshold_px: 0.0 })
    }

    pub(crate) fn prepare(&mut self, centered_points: &[[f32; 2]], object_points: &[[f32; 3]], threshold_px: f32) -> Result<()> {
        if centered_points.len() != object_points.len() || centered_points.len() < 4 || !threshold_px.is_finite() || threshold_px <= 0.0 { bail!("invalid gpu focal pnp observations"); }
        self.centered_points = centered_points.to_vec(); self.object_points = object_points.to_vec(); self.threshold_px = threshold_px; Ok(())
    }

    pub(crate) fn score(&mut self, candidate: GpuPnpFocalCandidate) -> Result<rustslam::tracker::PnPModelSupport> {
        if !candidate.focal.is_finite() || candidate.focal <= 0.0 { bail!("gpu focal pnp candidate has invalid focal length"); }
        let normalized = self.centered_points.iter().map(|p| [p[0] / candidate.focal, p[1] / candidate.focal]).collect::<Vec<_>>();
        self.scorer.prepare(&normalized, &self.object_points, self.threshold_px / candidate.focal)?;
        self.scorer.score_models(&[candidate.pose]).map(|mut values| values.remove(0))
    }
}
