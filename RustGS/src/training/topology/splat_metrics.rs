use crate::core::splats::sigmoid_scalar;
use crate::core::HostSplats;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TopologySplatMetrics {
    positions: Vec<f32>,
    log_scales: Vec<f32>,
    rotations: Vec<f32>,
    opacity_logits: Vec<f32>,
    retainable: Vec<bool>,
}

impl TopologySplatMetrics {
    pub(super) fn from_snapshot(splats: &HostSplats) -> Self {
        Self {
            positions: splats.positions.clone(),
            log_scales: splats.log_scales.clone(),
            rotations: splats.rotations.clone(),
            opacity_logits: splats.opacity_logits.clone(),
            retainable: (0..splats.len())
                .map(|idx| {
                    splats.position(idx).iter().all(|value| value.is_finite())
                        && splats.rotation(idx).iter().all(|value| value.is_finite())
                        && splats
                            .sh_coeffs_row(idx)
                            .iter()
                            .all(|value| value.is_finite())
                })
                .collect(),
        }
    }

    pub(super) fn len(&self) -> usize {
        self.opacity_logits.len()
    }

    pub(super) fn position(&self, idx: usize) -> [f32; 3] {
        let base = idx * 3;
        [
            self.positions.get(base).copied().unwrap_or_default(),
            self.positions.get(base + 1).copied().unwrap_or_default(),
            self.positions.get(base + 2).copied().unwrap_or_default(),
        ]
    }

    pub(super) fn log_scale(&self, idx: usize) -> [f32; 3] {
        let base = idx * 3;
        [
            self.log_scales.get(base).copied().unwrap_or_default(),
            self.log_scales.get(base + 1).copied().unwrap_or_default(),
            self.log_scales.get(base + 2).copied().unwrap_or_default(),
        ]
    }

    pub(super) fn scale(&self, idx: usize) -> [f32; 3] {
        let log = self.log_scale(idx);
        [log[0].exp(), log[1].exp(), log[2].exp()]
    }

    pub(super) fn max_scale(&self, idx: usize) -> f32 {
        let scale = self.scale(idx);
        scale[0].max(scale[1]).max(scale[2])
    }

    pub(super) fn opacity(&self, idx: usize) -> f32 {
        self.opacity_logits
            .get(idx)
            .copied()
            .map(sigmoid_scalar)
            .unwrap_or_default()
    }

    pub(super) fn retainable(&self, idx: usize) -> bool {
        self.retainable.get(idx).copied().unwrap_or(false)
    }

    pub(super) fn brush_bounds_center_extent(&self) -> ([f32; 3], f32) {
        if self.len() == 0 {
            return ([0.0, 0.0, 0.0], 1.0);
        }

        let mut center = [0.0f32; 3];
        for idx in 0..self.len() {
            let position = self.position(idx);
            center[0] += position[0];
            center[1] += position[1];
            center[2] += position[2];
        }
        let inv = 1.0 / self.len() as f32;
        center[0] *= inv;
        center[1] *= inv;
        center[2] *= inv;

        let mut extent = 0.0f32;
        for idx in 0..self.len() {
            let position = self.position(idx);
            extent = extent.max((position[0] - center[0]).abs());
            extent = extent.max((position[1] - center[1]).abs());
            extent = extent.max((position[2] - center[2]).abs());
        }
        (center, extent.max(1e-6))
    }
}
