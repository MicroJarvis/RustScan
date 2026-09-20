struct ScoringParams {
    model_count: u32,
    observation_count: u32,
    model_kind: u32,
    selected_model: u32,
    max_residual: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct SupportSummary {
    inliers: u32,
    residual_sum: f32,
}

struct HomogeneousPoint {
    x: f32,
    y: f32,
    z: f32,
    pad: f32,
}

@group(0) @binding(0) var<storage, read> models: array<f32>;
@group(0) @binding(1) var<storage, read> points1: array<HomogeneousPoint>;
@group(0) @binding(2) var<storage, read> points2: array<HomogeneousPoint>;
@group(0) @binding(3) var<storage, read_write> summaries: array<SupportSummary>;
@group(0) @binding(4) var<storage, read_write> mask: array<u32>;
@group(0) @binding(5) var<uniform> params: ScoringParams;

var<workgroup> local_inliers: array<u32, 64>;
var<workgroup> local_residual_sums: array<f32, 64>;

const MODEL_HOMOGRAPHY_FORWARD: u32 = 0u;
const MODEL_SAMPSON: u32 = 1u;
const MAX_FINITE_F32: f32 = 3.402823466e+38;

fn is_finite_f32(value: f32) -> bool {
    return value == value && abs(value) <= MAX_FINITE_F32;
}

fn invalid_residual() -> f32 {
    return bitcast<f32>(0x7f800000u);
}

fn model_value(model_index: u32, row: u32, column: u32) -> f32 {
    return models[model_index * 9u + row * 3u + column];
}

fn point_vector(point: HomogeneousPoint) -> vec3<f32> {
    return vec3<f32>(point.x, point.y, point.z);
}

fn homography_forward_residual(
    model_index: u32,
    point1: vec3<f32>,
    point2: vec3<f32>,
) -> f32 {
    let predicted_z = model_value(model_index, 2u, 0u) * point1.x
        + model_value(model_index, 2u, 1u) * point1.y
        + model_value(model_index, 2u, 2u) * point1.z;
    if (!is_finite_f32(predicted_z) || abs(predicted_z) <= 1.0e-12
        || !is_finite_f32(point2.z) || abs(point2.z) <= 1.0e-12) {
        return invalid_residual();
    }
    let predicted = vec2<f32>(
        (model_value(model_index, 0u, 0u) * point1.x
            + model_value(model_index, 0u, 1u) * point1.y
            + model_value(model_index, 0u, 2u) * point1.z) / predicted_z,
        (model_value(model_index, 1u, 0u) * point1.x
            + model_value(model_index, 1u, 1u) * point1.y
            + model_value(model_index, 1u, 2u) * point1.z) / predicted_z,
    );
    let observed = point2.xy / point2.z;
    let delta = predicted - observed;
    let residual = dot(delta, delta);
    if (!is_finite_f32(residual)) {
        return invalid_residual();
    }
    return residual;
}

fn sampson_residual(
    model_index: u32,
    point1: vec3<f32>,
    point2: vec3<f32>,
) -> f32 {
    let fx1 = vec3<f32>(
        model_value(model_index, 0u, 0u) * point1.x
            + model_value(model_index, 0u, 1u) * point1.y
            + model_value(model_index, 0u, 2u) * point1.z,
        model_value(model_index, 1u, 0u) * point1.x
            + model_value(model_index, 1u, 1u) * point1.y
            + model_value(model_index, 1u, 2u) * point1.z,
        model_value(model_index, 2u, 0u) * point1.x
            + model_value(model_index, 2u, 1u) * point1.y
            + model_value(model_index, 2u, 2u) * point1.z,
    );
    let ftx2 = vec3<f32>(
        model_value(model_index, 0u, 0u) * point2.x
            + model_value(model_index, 1u, 0u) * point2.y
            + model_value(model_index, 2u, 0u) * point2.z,
        model_value(model_index, 0u, 1u) * point2.x
            + model_value(model_index, 1u, 1u) * point2.y
            + model_value(model_index, 2u, 1u) * point2.z,
        model_value(model_index, 0u, 2u) * point2.x
            + model_value(model_index, 1u, 2u) * point2.y
            + model_value(model_index, 2u, 2u) * point2.z,
    );
    let numerator = point2.x * fx1.x + point2.y * fx1.y + point2.z * fx1.z;
    let denominator = fx1.x * fx1.x + fx1.y * fx1.y
        + ftx2.x * ftx2.x + ftx2.y * ftx2.y;
    if (!is_finite_f32(denominator) || denominator <= 1.0e-24) {
        return invalid_residual();
    }
    let residual = numerator * numerator / denominator;
    if (!is_finite_f32(residual)) {
        return invalid_residual();
    }
    return residual;
}

fn model_residual(model_index: u32, observation_index: u32) -> f32 {
    if (params.model_kind == MODEL_HOMOGRAPHY_FORWARD) {
        return homography_forward_residual(
            model_index,
            point_vector(points1[observation_index]),
            point_vector(points2[observation_index]),
        );
    }
    if (params.model_kind == MODEL_SAMPSON) {
        return sampson_residual(
            model_index,
            point_vector(points1[observation_index]),
            point_vector(points2[observation_index]),
        );
    }
    return invalid_residual();
}

fn is_inlier(residual: f32) -> bool {
    return is_finite_f32(residual) && residual <= params.max_residual;
}

@compute @workgroup_size(64)
fn score_models(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let model_index = group_id.x;
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
        let residual = model_residual(model_index, observation_index);
        if (is_inlier(residual)) {
            inliers += 1u;
            residual_sum += residual;
        }
        observation_index += 64u;
    }
    local_inliers[lane] = inliers;
    local_residual_sums[lane] = residual_sum;
    workgroupBarrier();

    var stride = 32u;
    loop {
        if (lane < stride) {
            local_inliers[lane] += local_inliers[lane + stride];
            local_residual_sums[lane] += local_residual_sums[lane + stride];
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride /= 2u;
    }

    if (lane == 0u) {
        summaries[model_index].inliers = local_inliers[0];
        summaries[model_index].residual_sum = local_residual_sums[0];
    }
}

var<workgroup> local_bits: array<u32, 64>;

// Scores every model exactly like `score_models` (same per-lane observation
// order, same reduction) and additionally writes a packed inlier bitmask per
// model: word `w` of model `m` lives at `m * words_per_model + w` and bit `b`
// of that word is observation `32 * w + b`. Bits at or beyond
// `observation_count` are zero. Every word below `ceil(observation_count / 32)`
// is written, so the mask buffer needs no clearing between dispatches.
@compute @workgroup_size(64)
fn score_models_with_masks(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let model_index = group_id.x;
    let lane = local_id.x;
    if (model_index >= params.model_count) {
        return;
    }
    let words_per_model = (params.observation_count + 31u) / 32u;
    let mask_base = model_index * words_per_model;

    var inliers = 0u;
    var residual_sum = 0.0;
    var base = 0u;
    loop {
        if (base >= params.observation_count) {
            break;
        }
        let observation_index = base + lane;
        var bit = 0u;
        if (observation_index < params.observation_count) {
            let residual = model_residual(model_index, observation_index);
            if (is_inlier(residual)) {
                inliers += 1u;
                residual_sum += residual;
                bit = 1u;
            }
        }
        local_bits[lane] = bit;
        workgroupBarrier();
        if ((lane == 0u || lane == 32u) && observation_index < params.observation_count) {
            var word = 0u;
            for (var i = 0u; i < 32u; i++) {
                word |= local_bits[lane + i] << i;
            }
            mask[mask_base + observation_index / 32u] = word;
        }
        workgroupBarrier();
        base += 64u;
    }
    local_inliers[lane] = inliers;
    local_residual_sums[lane] = residual_sum;
    workgroupBarrier();

    var stride = 32u;
    loop {
        if (lane < stride) {
            local_inliers[lane] += local_inliers[lane + stride];
            local_residual_sums[lane] += local_residual_sums[lane + stride];
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride /= 2u;
    }

    if (lane == 0u) {
        summaries[model_index].inliers = local_inliers[0];
        summaries[model_index].residual_sum = local_residual_sums[0];
    }
}

@compute @workgroup_size(64)
fn write_mask(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let observation_index = global_id.x;
    let model_index = global_id.y;
    if (observation_index >= params.observation_count || model_index >= params.model_count) {
        return;
    }
    let residual = model_residual(model_index, observation_index);
    let output_offset = model_index * params.observation_count + observation_index;
    mask[output_offset] = select(0u, 1u, is_inlier(residual));
}
