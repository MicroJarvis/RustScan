use burn::module::Param;
use burn::prelude::*;
use glam::{Quat, Vec3};

use crate::core::HostSplats;
use crate::training::engine::{host_splats_to_device, DeviceSplats};

use super::{TopologyMutationPlan, TopologyPlanRow, TopologyRefineDecay};

pub(crate) fn apply_mutations<B: Backend>(
    splats: &mut DeviceSplats<B>,
    snapshot: &HostSplats,
    plan: &TopologyMutationPlan,
    device: &B::Device,
) {
    if plan.rows.is_empty() {
        return;
    }

    let rebuilt = rebuild_host_snapshot(snapshot, plan);
    let updated = host_splats_to_device(&rebuilt, device);

    splats.transforms = Param::initialized(splats.transforms.id, updated.transforms.val());
    splats.sh_coeffs = Param::initialized(splats.sh_coeffs.id, updated.sh_coeffs.val());
    splats.raw_opacities = Param::initialized(splats.raw_opacities.id, updated.raw_opacities.val());
    splats.sh_degree = updated.sh_degree;
}

fn rebuild_host_snapshot(snapshot: &HostSplats, plan: &TopologyMutationPlan) -> HostSplats {
    let sh_row_width = if snapshot.is_empty() {
        3
    } else {
        snapshot.sh_coeffs_row(0).len()
    };
    let mut positions = Vec::with_capacity(plan.rows.len() * 3);
    let mut log_scales = Vec::with_capacity(plan.rows.len() * 3);
    let mut rotations = Vec::with_capacity(plan.rows.len() * 4);
    let mut opacity_logits = Vec::with_capacity(plan.rows.len());
    let mut sh_coeffs = Vec::with_capacity(plan.rows.len() * sh_row_width);

    for row in &plan.rows {
        let source_idx = row.source_idx();
        let mut position = snapshot.position(source_idx);
        let mut log_scale = snapshot.log_scale(source_idx);
        let rotation = snapshot.rotation(source_idx);
        let mut opacity_logit = snapshot.opacity_logit(source_idx);

        match *row {
            TopologyPlanRow::Existing { .. } => {}
            TopologyPlanRow::BrushRefineExisting { sample_scalar, .. } => {
                let offset =
                    brush_refine_offset(rotation, snapshot.scale(source_idx), sample_scalar);
                let refined_scale = brush_refine_scale(snapshot.scale(source_idx));
                position[0] -= offset[0];
                position[1] -= offset[1];
                position[2] -= offset[2];
                log_scale = refined_scale.map(|value| value.max(1e-6).ln());
                opacity_logit = brush_refine_opacity_logit(opacity_logit);
            }
            TopologyPlanRow::BrushRefineNew { sample_scalar, .. } => {
                let offset =
                    brush_refine_offset(rotation, snapshot.scale(source_idx), sample_scalar);
                let refined_scale = brush_refine_scale(snapshot.scale(source_idx));
                position[0] += offset[0];
                position[1] += offset[1];
                position[2] += offset[2];
                log_scale = refined_scale.map(|value| value.max(1e-6).ln());
                opacity_logit = brush_refine_opacity_logit(opacity_logit);
            }
        }

        if let Some(decay) = plan.refine_decay {
            apply_refine_decay(&mut log_scale, &mut opacity_logit, decay);
        }
        if plan.aftermath.apply_opacity_reset {
            opacity_logit = opacity_reset_logit(opacity_logit);
        }

        positions.extend_from_slice(&position);
        log_scales.extend_from_slice(&log_scale);
        rotations.extend_from_slice(&rotation);
        opacity_logits.push(opacity_logit);
        sh_coeffs.extend_from_slice(snapshot.sh_coeffs_row(source_idx));
    }

    HostSplats::from_components(
        positions,
        log_scales,
        rotations,
        opacity_logits,
        sh_coeffs,
        snapshot.sh_degree(),
    )
    .expect("valid topology mutation rebuild")
}

fn brush_refine_offset(rotation: [f32; 4], scale: [f32; 3], sample_scalar: f32) -> [f32; 3] {
    let quat = Quat::from_xyzw(rotation[1], rotation[2], rotation[3], rotation[0]);
    let quat = if quat.length_squared() > 0.0 {
        quat.normalize()
    } else {
        Quat::IDENTITY
    };
    let rotated = quat * (Vec3::from_array(scale) * sample_scalar);
    rotated.to_array()
}

fn brush_refine_scale(scale: [f32; 3]) -> [f32; 3] {
    let mut refined = scale;
    let mut max_axis = 0usize;
    for axis in 1..3 {
        if refined[axis] > refined[max_axis] {
            max_axis = axis;
        }
    }
    refined[max_axis] *= 0.5;
    refined
}

fn brush_refine_opacity_logit(opacity_logit: f32) -> f32 {
    let opacity = sigmoid_scalar(opacity_logit).clamp(1.0 / 255.0, 1.0 - 1.0 / 255.0);
    let refined = (1.0 - (1.0 - opacity).sqrt()).clamp(1.0 / 255.0, 1.0 - 1.0 / 255.0);
    (refined / (1.0 - refined)).ln()
}

fn opacity_reset_logit(opacity_logit: f32) -> f32 {
    let opacity = sigmoid_scalar(opacity_logit).clamp(1e-12, 1.0 - 1e-12);
    let reset = opacity.min(super::LITEGS_OPACITY_RESET_CAP);
    (reset / (1.0 - reset)).ln()
}

fn apply_refine_decay(
    log_scale: &mut [f32; 3],
    opacity_logit: &mut f32,
    decay: TopologyRefineDecay,
) {
    let shrink_strength = 1.0 - decay.train_t.clamp(0.0, 1.0);
    if decay.scale_decay > 0.0 {
        let scale_scaling = (1.0 - decay.scale_decay * shrink_strength).max(1e-6);
        for value in log_scale {
            *value = (value.exp() * scale_scaling).max(1e-12).ln();
        }
    }

    if decay.opacity_decay > 0.0 {
        let opacity = sigmoid_scalar(*opacity_logit);
        let decayed = (opacity - decay.opacity_decay * shrink_strength).clamp(1e-12, 1.0 - 1e-12);
        *opacity_logit = (decayed / (1.0 - decayed)).ln();
    }
}

fn sigmoid_scalar(value: f32) -> f32 {
    if value >= 0.0 {
        let exp = (-value).exp();
        1.0 / (1.0 + exp)
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brush_refine_offset_normalizes_non_unit_rotation() {
        let unit = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let rotation = [unit.w * 2.0, unit.x * 2.0, unit.y * 2.0, unit.z * 2.0];

        let offset = brush_refine_offset(rotation, [2.0, 0.0, 0.0], 1.0);

        assert!(offset[0].abs() < 1e-5, "x offset was {}", offset[0]);
        assert!(
            (offset[1] - 2.0).abs() < 1e-5,
            "y offset was {}",
            offset[1]
        );
        assert!(offset[2].abs() < 1e-5, "z offset was {}", offset[2]);
    }

    #[test]
    fn brush_refine_offset_uses_identity_for_zero_rotation() {
        let offset = brush_refine_offset([0.0; 4], [1.0, 2.0, 3.0], 0.5);

        assert_eq!(offset, [0.5, 1.0, 1.5]);
    }
}
