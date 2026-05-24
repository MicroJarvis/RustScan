use serde::{Deserialize, Serialize};

use crate::sh::{sh0_to_rgb_value, sh_coeff_count_for_degree};
use crate::TrainingError;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HostSplats {
    pub(crate) positions: Vec<f32>,
    pub(crate) log_scales: Vec<f32>,
    pub(crate) rotations: Vec<f32>,
    pub(crate) opacity_logits: Vec<f32>,
    pub(crate) sh_coeffs: Vec<f32>,
    pub(crate) sh_degree: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct SplatView<'a> {
    pub positions: &'a [f32],
    pub log_scales: &'a [f32],
    pub rotations: &'a [f32],
    pub opacity_logits: &'a [f32],
    pub sh_coeffs: &'a [f32],
    pub sh_degree: usize,
}

impl HostSplats {
    /// Build a host-side splat set from its packed component arrays.
    pub fn from_components(
        positions: Vec<f32>,
        log_scales: Vec<f32>,
        rotations: Vec<f32>,
        opacity_logits: Vec<f32>,
        sh_coeffs: Vec<f32>,
        sh_degree: usize,
    ) -> Result<Self, TrainingError> {
        let splats = Self {
            positions,
            log_scales,
            rotations,
            opacity_logits,
            sh_coeffs,
            sh_degree,
        };
        splats.validate()?;
        Ok(splats)
    }

    pub fn as_view(&self) -> SplatView<'_> {
        SplatView {
            positions: &self.positions,
            log_scales: &self.log_scales,
            rotations: &self.rotations,
            opacity_logits: &self.opacity_logits,
            sh_coeffs: &self.sh_coeffs,
            sh_degree: self.sh_degree,
        }
    }

    pub fn validate(&self) -> Result<(), TrainingError> {
        let row_count = self.opacity_logits.len();
        validate_component_len("positions", self.positions.len(), row_count, 3)?;
        validate_component_len("log_scales", self.log_scales.len(), row_count, 3)?;
        validate_component_len("rotations", self.rotations.len(), row_count, 4)?;
        validate_component_len(
            "sh_coeffs",
            self.sh_coeffs.len(),
            row_count,
            self.sh_coeffs_row_width(),
        )?;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.opacity_logits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn position(&self, idx: usize) -> [f32; 3] {
        let base = idx * 3;
        [
            self.positions[base],
            self.positions[base + 1],
            self.positions[base + 2],
        ]
    }

    pub fn log_scale(&self, idx: usize) -> [f32; 3] {
        let base = idx * 3;
        [
            self.log_scales[base],
            self.log_scales[base + 1],
            self.log_scales[base + 2],
        ]
    }

    pub fn rotation(&self, idx: usize) -> [f32; 4] {
        let base = idx * 4;
        [
            self.rotations[base],
            self.rotations[base + 1],
            self.rotations[base + 2],
            self.rotations[base + 3],
        ]
    }

    pub fn sh_0(&self, idx: usize) -> [f32; 3] {
        let base = idx * self.sh_coeffs_row_width();
        [
            self.sh_coeffs[base],
            self.sh_coeffs[base + 1],
            self.sh_coeffs[base + 2],
        ]
    }

    pub fn rgb_color(&self, idx: usize) -> [f32; 3] {
        self.sh_0(idx).map(sh0_to_rgb_value)
    }

    pub fn sh_degree(&self) -> usize {
        self.sh_degree
    }

    pub(crate) fn sh_coeffs_row_width(&self) -> usize {
        sh_coeff_count_for_degree(self.sh_degree) * 3
    }

    pub fn sh_coeffs_row(&self, idx: usize) -> &[f32] {
        row_slice(&self.sh_coeffs, self.sh_coeffs_row_width(), idx)
    }

    pub fn sh_rest(&self, idx: usize) -> &[f32] {
        self.sh_coeffs_row(idx).get(3..).unwrap_or(&[])
    }

    pub fn scale(&self, idx: usize) -> [f32; 3] {
        let log = self.log_scale(idx);
        [log[0].exp(), log[1].exp(), log[2].exp()]
    }

    pub fn opacity_logit(&self, idx: usize) -> f32 {
        self.opacity_logits.get(idx).copied().unwrap_or_default()
    }

    pub fn opacity(&self, idx: usize) -> f32 {
        sigmoid_scalar(self.opacity_logit(idx)).clamp(0.0, 1.0)
    }

    #[allow(dead_code)]
    pub(crate) fn push(
        &mut self,
        position: [f32; 3],
        log_scale: [f32; 3],
        rotation: [f32; 4],
        opacity_logit: f32,
        sh_coeffs: &[f32],
    ) {
        self.positions.extend_from_slice(&position);
        self.log_scales.extend_from_slice(&log_scale);
        self.rotations.extend_from_slice(&rotation);
        self.opacity_logits.push(opacity_logit);
        self.push_sh_coeffs_row(sh_coeffs);
    }

    pub(crate) fn truncate_rows(&mut self, row_count: usize) {
        if row_count >= self.len() {
            return;
        }

        self.positions.truncate(row_count * 3);
        self.log_scales.truncate(row_count * 3);
        self.rotations.truncate(row_count * 4);
        self.opacity_logits.truncate(row_count);
        self.sh_coeffs
            .truncate(row_count * self.sh_coeffs_row_width());
    }

    #[allow(dead_code)]
    pub(crate) fn retained_view(&self, row_count: usize) -> Self {
        Self {
            positions: Vec::with_capacity(row_count * 3),
            log_scales: Vec::with_capacity(row_count * 3),
            rotations: Vec::with_capacity(row_count * 4),
            opacity_logits: Vec::with_capacity(row_count),
            sh_coeffs: Vec::with_capacity(row_count * self.sh_coeffs_row_width()),
            sh_degree: self.sh_degree,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn downsample_evenly(&mut self, target_count: usize) {
        if target_count == 0 || self.len() <= target_count {
            return;
        }

        let len = self.len();
        let mut sampled = self.retained_view(target_count);
        for out_idx in 0..target_count {
            let src_idx = out_idx.saturating_mul(len) / target_count;
            let src_idx = src_idx.min(len.saturating_sub(1));
            sampled.push(
                self.position(src_idx),
                self.log_scale(src_idx),
                self.rotation(src_idx),
                self.opacity_logits[src_idx],
                self.sh_coeffs_row(src_idx),
            );
        }
        *self = sampled;
    }

    pub fn positions_vec3(&self) -> Vec<[f32; 3]> {
        (0..self.len()).map(|idx| self.position(idx)).collect()
    }

    pub fn to_splat_metadata(&self, iterations: usize, final_loss: f32) -> crate::SplatMetadata {
        crate::SplatMetadata {
            iterations,
            final_loss,
            gaussian_count: self.len(),
            sh_degree: self.sh_degree(),
        }
    }

    pub(crate) fn scene_extent(&self) -> f32 {
        if self.is_empty() {
            return 1.0;
        }

        let mut center = [0.0f32; 3];
        for idx in 0..self.len() {
            let position = self.position(idx);
            center[0] += position[0];
            center[1] += position[1];
            center[2] += position[2];
        }
        let inv = 1.0 / self.len().max(1) as f32;
        center[0] *= inv;
        center[1] *= inv;
        center[2] *= inv;

        let mut max_dist = 0.0f32;
        for idx in 0..self.len() {
            let position = self.position(idx);
            let dx = position[0] - center[0];
            let dy = position[1] - center[1];
            let dz = position[2] - center[2];
            max_dist = max_dist.max((dx * dx + dy * dy + dz * dz).sqrt());
        }
        max_dist.max(1e-3)
    }

    #[allow(dead_code)]
    fn push_sh_coeffs_row(&mut self, sh_coeffs: &[f32]) {
        let row_width = self.sh_coeffs_row_width();
        if row_width == 0 {
            return;
        }
        let copied = sh_coeffs.len().min(row_width);
        self.sh_coeffs.extend_from_slice(&sh_coeffs[..copied]);
        if copied < row_width {
            self.sh_coeffs
                .resize(self.sh_coeffs.len() + (row_width - copied), 0.0);
        }
    }
}

pub(crate) fn row_slice(values: &[f32], width: usize, idx: usize) -> &[f32] {
    let start = idx.saturating_mul(width);
    let end = start.saturating_add(width);
    values.get(start..end).unwrap_or(&[])
}

fn validate_component_len(
    name: &str,
    actual: usize,
    row_count: usize,
    row_width: usize,
) -> Result<(), TrainingError> {
    let expected = row_count.saturating_mul(row_width);
    if actual != expected {
        return Err(TrainingError::TrainingFailed(format!(
            "splats invariant violated: {name} expected {expected} values for {row_count} gaussians, got {actual}"
        )));
    }
    Ok(())
}

pub(crate) fn sigmoid_scalar(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
#[path = "splats_tests.rs"]
mod tests;
