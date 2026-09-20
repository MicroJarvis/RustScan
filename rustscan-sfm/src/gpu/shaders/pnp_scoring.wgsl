struct PnpScoringParams {
    model_count: u32,
    observation_count: u32,
    selected_model: u32,
    pad0: u32,
    max_residual: f32,
    pad1: f32,
    pad2: f32,
    pad3: f32,
}

struct PnpImagePoint {
    x: f32,
    y: f32,
    pad0: f32,
    pad1: f32,
}

struct PnpObjectPoint {
    x: f32,
    y: f32,
    z: f32,
    pad: f32,
}

struct PnpModel {
    row0: vec4<f32>,
    row1: vec4<f32>,
    row2: vec4<f32>,
}

struct SupportSummary {
    inliers: u32,
    residual_sum: f32,
}

@group(0) @binding(0) var<storage, read> models: array<PnpModel>;
@group(0) @binding(1) var<storage, read> image_points: array<PnpImagePoint>;
@group(0) @binding(2) var<storage, read> object_points: array<PnpObjectPoint>;
@group(0) @binding(3) var<storage, read_write> summaries: array<SupportSummary>;
@group(0) @binding(4) var<storage, read_write> mask: array<u32>;
@group(0) @binding(5) var<uniform> params: PnpScoringParams;

var<workgroup> local_inliers: array<u32, 64>;
var<workgroup> local_residual_sums: array<f32, 64>;

const MAX_FINITE_F32: f32 = 3.402823466e+38;

fn is_finite_f32(value: f32) -> bool {
    return value == value && abs(value) <= MAX_FINITE_F32;
}

fn invalid_residual() -> f32 {
    return bitcast<f32>(0x7f800000u);
}

fn camera_point(model: PnpModel, point: PnpObjectPoint) -> vec3<f32> {
    let object = vec3<f32>(point.x, point.y, point.z);
    return vec3<f32>(
        dot(model.row0.xyz, object) + model.row0.w,
        dot(model.row1.xyz, object) + model.row1.w,
        dot(model.row2.xyz, object) + model.row2.w,
    );
}

fn reprojection_residual(model_index: u32, observation_index: u32) -> f32 {
    let camera = camera_point(models[model_index], object_points[observation_index]);
    let observed = image_points[observation_index];
    var projected = vec2<f32>(0.0, 0.0);
    if (camera.z > 0.0) {
        projected = camera.xy / camera.z;
    }
    let delta = vec2<f32>(observed.x, observed.y) - projected;
    let residual = dot(delta, delta);
    if (!is_finite_f32(residual)) {
        return invalid_residual();
    }
    return residual;
}

fn is_inlier(residual: f32) -> bool {
    return is_finite_f32(residual) && residual <= params.max_residual;
}

@compute @workgroup_size(64)
fn score_models(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let model_index = workgroup_id.x;
    let lane = local_id.x;
    if (model_index >= params.model_count) {
        return;
    }

    var inliers = 0u;
    var residual_sum = 0.0;
    var observation_index = lane;
    loop {
        if (observation_index >= params.observation_count) {
            break;
        }
        let residual = reprojection_residual(model_index, observation_index);
        if (is_inlier(residual)) {
            inliers = inliers + 1u;
            residual_sum = residual_sum + residual;
        }
        observation_index = observation_index + 64u;
    }
    local_inliers[lane] = inliers;
    local_residual_sums[lane] = residual_sum;
    workgroupBarrier();

    var stride = 32u;
    loop {
        if (lane < stride) {
            local_inliers[lane] = local_inliers[lane] + local_inliers[lane + stride];
            local_residual_sums[lane] = local_residual_sums[lane]
                + local_residual_sums[lane + stride];
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    if (lane == 0u) {
        summaries[model_index].inliers = local_inliers[0];
        summaries[model_index].residual_sum = local_residual_sums[0];
    }
}

@compute @workgroup_size(64)
fn write_mask(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let observation_index = global_id.x;
    if (observation_index >= params.observation_count) {
        return;
    }
    let residual = reprojection_residual(params.selected_model, observation_index);
    mask[observation_index] = select(0u, 1u, is_inlier(residual));
}
