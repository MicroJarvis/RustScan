use serde::{Deserialize, Serialize};

use crate::sh::sh0_to_rgb_value;
use crate::TrainingError;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
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

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HostSplatsCacheKey {
    len: usize,
    sh_degree: usize,
    positions: usize,
    log_scales: usize,
    rotations: usize,
    opacity_logits: usize,
    sh_coeffs: usize,
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

    #[cfg(feature = "gpu")]
    pub(crate) fn cache_key(&self) -> HostSplatsCacheKey {
        let view = self.as_view();
        HostSplatsCacheKey {
            len: self.len(),
            sh_degree: self.sh_degree(),
            positions: view.positions.as_ptr() as usize,
            log_scales: view.log_scales.as_ptr() as usize,
            rotations: view.rotations.as_ptr() as usize,
            opacity_logits: view.opacity_logits.as_ptr() as usize,
            sh_coeffs: view.sh_coeffs.as_ptr() as usize,
        }
    }

    pub fn validate(&self) -> Result<(), TrainingError> {
        let row_count = self.opacity_logits.len();
        validate_component_len("positions", self.positions.len(), row_count, 3)?;
        validate_component_len("log_scales", self.log_scales.len(), row_count, 3)?;
        validate_component_len("rotations", self.rotations.len(), row_count, 4)?;
        let sh_coeffs_row_width = self.checked_sh_coeffs_row_width()?;
        validate_component_len(
            "sh_coeffs",
            self.sh_coeffs.len(),
            row_count,
            sh_coeffs_row_width,
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
        component_array(&self.positions, idx)
    }

    pub fn log_scale(&self, idx: usize) -> [f32; 3] {
        component_array(&self.log_scales, idx)
    }

    pub fn rotation(&self, idx: usize) -> [f32; 4] {
        component_array(&self.rotations, idx)
    }

    pub fn sh_0(&self, idx: usize) -> [f32; 3] {
        component_array(self.sh_coeffs_row(idx), 0)
    }

    pub fn rgb_color(&self, idx: usize) -> [f32; 3] {
        self.sh_0(idx).map(sh0_to_rgb_value)
    }

    pub fn sh_degree(&self) -> usize {
        self.sh_degree
    }

    pub(crate) fn sh_coeffs_row_width(&self) -> usize {
        self.checked_sh_coeffs_row_width().unwrap_or(usize::MAX)
    }

    pub fn sh_coeffs_row(&self, idx: usize) -> &[f32] {
        row_slice(&self.sh_coeffs, self.sh_coeffs_row_width(), idx)
    }

    pub fn sh_rest(&self, idx: usize) -> &[f32] {
        self.sh_coeffs_row(idx).get(3..).unwrap_or(&[])
    }

    fn checked_sh_coeffs_row_width(&self) -> Result<usize, TrainingError> {
        let order = self
            .sh_degree
            .checked_add(1)
            .ok_or_else(|| sh_width_overflow(self.sh_degree))?;
        order
            .checked_mul(order)
            .and_then(|coefficient_count| coefficient_count.checked_mul(3))
            .ok_or_else(|| sh_width_overflow(self.sh_degree))
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

    #[cfg(feature = "gpu")]
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
}

fn component_array<const N: usize>(values: &[f32], idx: usize) -> [f32; N] {
    let start = idx.saturating_mul(N);
    std::array::from_fn(|offset| {
        values
            .get(start.saturating_add(offset))
            .copied()
            .unwrap_or_default()
    })
}

fn sh_width_overflow(sh_degree: usize) -> TrainingError {
    TrainingError::TrainingFailed(format!(
        "splats invariant violated: SH width overflow for degree {sh_degree}"
    ))
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
mod tests {
    use std::panic::catch_unwind;

    use super::HostSplats;

    #[test]
    fn corrupt_splat_component_reads_do_not_panic() {
        let splats = HostSplats {
            positions: vec![],
            log_scales: vec![],
            rotations: vec![],
            opacity_logits: vec![0.0],
            sh_coeffs: vec![],
            sh_degree: usize::MAX,
        };

        let reads = catch_unwind(|| {
            assert_eq!(splats.position(0), [0.0; 3]);
            assert_eq!(splats.log_scale(0), [0.0; 3]);
            assert_eq!(splats.rotation(0), [0.0; 4]);
            assert_eq!(splats.sh_0(0), [0.0; 3]);
            assert!(splats.sh_coeffs_row(0).is_empty());
        });

        assert!(reads.is_ok(), "corrupt splat reads must not panic");
    }
}
