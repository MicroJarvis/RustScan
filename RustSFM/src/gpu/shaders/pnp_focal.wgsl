struct PnpFocalModel {
    row0: vec4<f32>, row1: vec4<f32>, row2: vec4<f32>,
    log_focal: f32, pad0: f32, pad1: f32, pad2: f32,
}

struct PnpFocalResult {
    selected_model: u32, inliers: u32, valid: u32, pad0: u32,
    residual_sum: f32, focal: f32, pad1: f32, pad2: f32,
}

struct PnpFocalSample {
    indices: vec4<u32>,
}

struct PnpFocalSamplingParams {
    seed: u32,
    trial_count: u32,
    observation_count: u32,
    pad: u32,
}

@group(0) @binding(0)
var<storage, read_write> samples: array<PnpFocalSample>;

@group(0) @binding(1)
var<uniform> params: PnpFocalSamplingParams;

fn sample_hash(seed: u32, trial: u32, lane: u32, attempt: u32) -> u32 {
    var value = seed ^ (trial * 0x9e3779b9u) ^ (lane * 0x85ebca6bu) ^ (attempt * 0xc2b2ae35u);
    value = value ^ (value >> 16u);
    value = value * 0x7feb352du;
    value = value ^ (value >> 15u);
    value = value * 0x846ca68bu;
    return value ^ (value >> 16u);
}

@compute @workgroup_size(1)
fn sample_four_points(@builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let trial = workgroup_id.x;
    if (trial >= params.trial_count) {
        return;
    }

    var selected: array<u32, 4>;
    var lane = 0u;
    loop {
        if (lane == 4u) {
            break;
        }
        var attempt = 0u;
        loop {
            let candidate = sample_hash(params.seed, trial, lane, attempt) % params.observation_count;
            var duplicate = false;
            var previous = 0u;
            loop {
                if (previous == lane) {
                    break;
                }
                if (selected[previous] == candidate) {
                    duplicate = true;
                    break;
                }
                previous = previous + 1u;
            }
            if (!duplicate) {
                selected[lane] = candidate;
                break;
            }
            attempt = attempt + 1u;
        }
        lane = lane + 1u;
    }
    samples[trial].indices = vec4<u32>(selected[0], selected[1], selected[2], selected[3]);
}

struct PnpFocalImagePoint {
    x: f32,
    y: f32,
    pad0: f32,
    pad1: f32,
}

struct PnpFocalObjectPoint {
    x: f32,
    y: f32,
    z: f32,
    pad: f32,
}

struct PnpFocalCandidateParams {
    sample: vec4<u32>,
    focal: f32,
    triple: u32,
    observation_count: u32,
    model_offset: u32,
}

@group(1) @binding(0)
var<storage, read> candidate_image_points: array<PnpFocalImagePoint>;
@group(1) @binding(1)
var<storage, read> candidate_object_points: array<PnpFocalObjectPoint>;
@group(1) @binding(2)
var<storage, read_write> single_candidate_models: array<PnpFocalModel>;
@group(1) @binding(3)
var<uniform> single_candidate_params: PnpFocalCandidateParams;

@group(3) @binding(0)
var<storage, read> batch_candidate_image_points: array<PnpFocalImagePoint>;
@group(3) @binding(1)
var<storage, read> batch_candidate_object_points: array<PnpFocalObjectPoint>;
@group(3) @binding(2)
var<storage, read> batch_candidate_params: array<PnpFocalCandidateParams>;
@group(3) @binding(3)
var<storage, read_write> batch_candidate_models: array<PnpFocalModel>;

var<private> candidate_params: PnpFocalCandidateParams;
var<private> use_batch_candidates: bool;

const PNP_EPS: f32 = 1.0e-6;

fn finite(value: f32) -> bool {
    return value == value && abs(value) < 3.402823466e+38;
}

fn invalid_model() -> PnpFocalModel {
    var model: PnpFocalModel;
    model.row0 = vec4<f32>(0.0);
    model.row1 = vec4<f32>(0.0);
    model.row2 = vec4<f32>(0.0);
    model.log_focal = bitcast<f32>(0x7f800000u);
    model.pad0 = 0.0;
    model.pad1 = 0.0;
    model.pad2 = 0.0;
    return model;
}

fn inverse3(matrix: mat3x3<f32>) -> mat3x3<f32> {
    let c0 = cross(matrix[1], matrix[2]);
    let c1 = cross(matrix[2], matrix[0]);
    let c2 = cross(matrix[0], matrix[1]);
    let determinant = dot(matrix[0], c0);
    if (!finite(determinant) || abs(determinant) <= PNP_EPS) {
        return mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    }
    let inverse_determinant = 1.0 / determinant;
    let adjugate = transpose(mat3x3<f32>(c0, c1, c2));
    return mat3x3<f32>(
        adjugate[0] * inverse_determinant,
        adjugate[1] * inverse_determinant,
        adjugate[2] * inverse_determinant,
    );
}

fn cubic_single_real(c2: f32, c1: f32, c0: f32) -> vec2<f32> {
    let a = c1 - c2 * c2 / 3.0;
    let b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
    let discriminant = b * b / 4.0 + a * a * a / 27.0;
    if (discriminant > 0.0) {
        let root_disc = sqrt(discriminant);
        let root = sign(-0.5 * b + root_disc) * pow(abs(-0.5 * b + root_disc), 1.0 / 3.0)
            + sign(-0.5 * b - root_disc) * pow(abs(-0.5 * b - root_disc), 1.0 / 3.0)
            - c2 / 3.0;
        return vec2<f32>(1.0, root);
    }
    if (a >= -PNP_EPS) {
        return vec2<f32>(0.0, -c2 / 3.0);
    }
    let angle = acos(clamp(3.0 * b / (2.0 * a) * sqrt(-3.0 / a), -1.0, 1.0));
    return vec2<f32>(0.0, 2.0 * sqrt(-a / 3.0) * cos(angle / 3.0) - c2 / 3.0);
}

fn quadratic_roots(b: f32, c: f32) -> vec2<f32> {
    let disc = b * b - 4.0 * c;
    if (disc < -PNP_EPS) { return vec2<f32>(bitcast<f32>(0x7f800000u)); }
    let root_disc = sqrt(max(disc, 0.0));
    if (b < 0.0) { return vec2<f32>(0.5 * (-b + root_disc), 0.5 * (-b - root_disc)); }
    let denom0 = -b + root_disc;
    let denom1 = -b - root_disc;
    return vec2<f32>(select(-0.5 * b, 2.0 * c / denom0, abs(denom0) > PNP_EPS), select(-0.5 * b, 2.0 * c / denom1, abs(denom1) > PNP_EPS));
}

fn refine_lambda(lambda: vec3<f32>, a01: f32, a02: f32, a12: f32, m01: f32, m02: f32, m12: f32) -> vec3<f32> {
    var value = lambda;
    for (var iter = 0u; iter < 5u; iter = iter + 1u) {
        let residual = vec3<f32>(
            value.x * value.x - 2.0 * value.x * value.y * m01 + value.y * value.y - a01,
            value.x * value.x - 2.0 * value.x * value.z * m02 + value.z * value.z - a02,
            value.y * value.y - 2.0 * value.y * value.z * m12 + value.z * value.z - a12);
        let x11 = value.x - value.y * m01; let x12 = value.y - value.x * m01;
        let x21 = value.x - value.z * m02; let x23 = value.z - value.x * m02;
        let x32 = value.y - value.z * m12; let x33 = value.z - value.y * m12;
        let denom = x11 * x23 * x32 + x12 * x21 * x33;
        if (abs(denom) <= PNP_EPS) { break; }
        let scale = 0.5 / denom;
        value.x += (-x23*x32*residual.x - x12*x33*residual.y + x12*x23*residual.z) * scale;
        value.y += (-x21*x33*residual.x + x11*x33*residual.y - x11*x23*residual.z) * scale;
        value.z += (x21*x32*residual.x - x11*x32*residual.y - x12*x21*residual.z) * scale;
    }
    return value;
}

struct PqPair { first: vec3<f32>, second: vec3<f32> }

fn compute_pq(c: mat3x3<f32>) -> PqPair {
    let adj0 = vec3<f32>(
        c[1].z * c[2].y - c[1].y * c[2].z,
        c[0].y * c[2].z - c[0].z * c[2].y,
        c[0].y * c[1].z - c[0].z * c[1].y,
    );
    let adj1 = vec3<f32>(adj0.y, c[0].z * c[2].x - c[0].x * c[2].z, c[0].x * c[1].z - c[0].z * c[1].x);
    let adj2 = vec3<f32>(adj0.z, adj1.z, c[0].y * c[1].x - c[0].x * c[1].y);
    let diagonal = vec3<f32>(adj0.x, adj1.y, adj2.z);
    var index = 0u;
    if (diagonal.y > diagonal.x && diagonal.y >= diagonal.z) { index = 1u; }
    if (diagonal.z > diagonal.x && diagonal.z > diagonal.y) { index = 2u; }
    let column = select(select(adj0, adj1, index == 1u), adj2, index == 2u);
    let normalizer = sqrt(max(abs(diagonal[index]), PNP_EPS));
    let vector = column / normalizer;
    var shifted = c;
    shifted[1].x -= vector.z;
    shifted[2].x += vector.y;
    shifted[0].y += vector.z;
    shifted[2].y -= vector.x;
    shifted[0].z -= vector.y;
    shifted[1].z += vector.x;
    return PqPair(shifted[0], vec3<f32>(shifted[0].x, shifted[1].x, shifted[2].x));
}

fn object_at(index: u32) -> vec3<f32> {
    var point: PnpFocalObjectPoint;
    if (use_batch_candidates) {
        point = batch_candidate_object_points[index];
    } else {
        point = candidate_object_points[index];
    }
    return vec3<f32>(point.x, point.y, point.z);
}

fn ray_at(index: u32) -> vec3<f32> {
    var point: PnpFocalImagePoint;
    if (use_batch_candidates) {
        point = batch_candidate_image_points[index];
    } else {
        point = candidate_image_points[index];
    }
    return normalize(vec3<f32>(point.x / candidate_params.focal, point.y / candidate_params.focal, 1.0));
}

fn triple_indices(slot: u32) -> vec3<u32> {
    if (slot == 0u) { return vec3<u32>(0u, 1u, 2u); }
    if (slot == 1u) { return vec3<u32>(0u, 1u, 3u); }
    if (slot == 2u) { return vec3<u32>(0u, 2u, 3u); }
    return vec3<u32>(1u, 2u, 3u);
}

fn sample_index(slot: u32) -> u32 {
    if (slot == 0u) { return candidate_params.sample.x; }
    if (slot == 1u) { return candidate_params.sample.y; }
    if (slot == 2u) { return candidate_params.sample.z; }
    return candidate_params.sample.w;
}

fn p3p_setup(triple: u32) -> mat3x3<f32> {
    let indices = triple_indices(triple);
    let original0 = sample_index(indices.x);
    let original1 = sample_index(indices.y);
    let original2 = sample_index(indices.z);
    let world0 = object_at(original0);
    let world1 = object_at(original1);
    let world2 = object_at(original2);
    let a01 = dot(world0 - world1, world0 - world1);
    let a02 = dot(world0 - world2, world0 - world2);
    let a12 = dot(world1 - world2, world1 - world2);

    // The P3P elimination divides by a12, so retain the longest baseline as
    // the second/third pair while moving the matching ray with each point.
    var first = original0;
    var second = original1;
    var third = original2;
    if (a02 >= a01 && a02 >= a12) {
        first = original1;
        second = original0;
        third = original2;
    } else if (a01 >= a02 && a01 >= a12) {
        first = original2;
        second = original0;
        third = original1;
    }
    let ordered0 = object_at(first);
    let ordered1 = object_at(second);
    let ordered2 = object_at(third);
    let ray0 = ray_at(first);
    let ray1 = ray_at(second);
    let ray2 = ray_at(third);
    let ordered_a01 = dot(ordered0 - ordered1, ordered0 - ordered1);
    let ordered_a02 = dot(ordered0 - ordered2, ordered0 - ordered2);
    let ordered_a12 = dot(ordered1 - ordered2, ordered1 - ordered2);
    return mat3x3<f32>(
        vec3<f32>(ordered_a01, ordered_a02, ordered_a12),
        vec3<f32>(dot(ray0, ray1), dot(ray0, ray2), dot(ray1, ray2)),
        vec3<f32>(f32(first), f32(second), f32(third)),
    );
}

fn p3p_cubic_coefficients(geometry: mat3x3<f32>) -> vec3<f32> {
    let a01 = geometry[0].x; let a02 = geometry[0].y; let a12 = geometry[0].z;
    let m01 = geometry[1].x; let m02 = geometry[1].y; let m12 = geometry[1].z;
    if (min(a01, min(a02, a12)) <= PNP_EPS) { return vec3<f32>(bitcast<f32>(0x7f800000u)); }
    let a = a01 / a12; let b = a02 / a12;
    let m12sq = 1.0 - m12 * m12;
    let m02sq = m02 * m02 - 1.0;
    let m01sq = m01 * m01 - 1.0;
    let ab = a * b;
    let denominator = b * b * m12sq + b * m02sq;
    if (abs(denominator) <= PNP_EPS) { return vec3<f32>(bitcast<f32>(0x7f800000u)); }
    let inverse = 1.0 / denominator;
    let mixed = -2.0 + 2.0 * m01 * m02 * m12;
    let k2 = inverse * ((a - 1.0) * m02sq + 2.0 * ab * m12sq + b*b*m12sq + b*mixed);
    let k1 = inverse * (a*a*m12sq + 2.0*ab*m12sq + a*mixed + (b - 1.0)*m01sq);
    let k0 = inverse * (a*a*m12sq + a*m01sq);
    return vec3<f32>(k2, k1, k0);
}

fn p3p_constraint_matrix(geometry: mat3x3<f32>, s: f32) -> mat3x3<f32> {
    let a01 = geometry[0].x / geometry[0].z;
    let a02 = geometry[0].y / geometry[0].z;
    let m01 = geometry[1].x; let m02 = geometry[1].y; let m12 = geometry[1].z;
    return mat3x3<f32>(
        vec3<f32>(-a01 + s * (1.0 - a02), -m02 * s, a01 * m12 + a02 * m12 * s),
        vec3<f32>(-m02 * s, s + 1.0, -m01),
        vec3<f32>(a01 * m12 + a02 * m12 * s, -m01, -a01 - a02 * s + 1.0),
    );
}

fn model_from_depths(geometry: mat3x3<f32>, depths: vec3<f32>) -> PnpFocalModel {
    if (any(depths <= vec3<f32>(0.0)) || !finite(depths.x) || !finite(depths.y) || !finite(depths.z)) { return invalid_model(); }
    let first = u32(geometry[2].x); let second = u32(geometry[2].y); let third = u32(geometry[2].z);
    let world0 = object_at(first); let world1 = object_at(second); let world2 = object_at(third);
    let ray0 = ray_at(first); let ray1 = ray_at(second); let ray2 = ray_at(third);
    let x01 = world0 - world1; let x02 = world0 - world2;
    let y01 = depths.x * ray0 - depths.y * ray1;
    let y02 = depths.x * ray0 - depths.z * ray2;
    let basis_world = mat3x3<f32>(x01, x02, cross(x01, x02));
    let basis_camera = mat3x3<f32>(y01, y02, cross(y01, y02));
    let rotation = basis_camera * inverse3(basis_world);
    let translation = depths.x * ray0 - rotation * world0;
    let rotation_determinant = dot(rotation[0], cross(rotation[1], rotation[2]));
    if (abs(rotation_determinant) <= PNP_EPS || !finite(rotation[0].x) || !finite(rotation[1].y) || !finite(rotation[2].z) || !finite(translation.x) || !finite(translation.y) || !finite(translation.z)) { return invalid_model(); }
    var model: PnpFocalModel;
    model.row0 = vec4<f32>(rotation[0].x, rotation[1].x, rotation[2].x, translation.x);
    model.row1 = vec4<f32>(rotation[0].y, rotation[1].y, rotation[2].y, translation.y);
    model.row2 = vec4<f32>(rotation[0].z, rotation[1].z, rotation[2].z, translation.z);
    model.log_focal = log(candidate_params.focal);
    model.pad0 = 0.0; model.pad1 = 0.0; model.pad2 = 0.0;
    return model;
}

fn depths_from_pq(geometry: mat3x3<f32>, pq: vec3<f32>, use_first_elimination: bool, root_index: u32) -> vec3<f32> {
    let a01 = geometry[0].x; let a02 = geometry[0].y; let a12 = geometry[0].z;
    let m01 = geometry[1].x; let m02 = geometry[1].y; let m12 = geometry[1].z;
    if (use_first_elimination) {
        if (abs(pq.y) <= PNP_EPS) { return vec3<f32>(0.0); }
        let w0 = -pq.x / pq.y; let w1 = -pq.z / pq.y;
        let denominator = w1*w1 - a02/a12;
        if (abs(denominator) <= PNP_EPS) { return vec3<f32>(0.0); }
        let cb = 2.0 * ((a02/a12)*m12 - m02*w1 + w0*w1) / denominator;
        let cc = (w0*w0 - 2.0*m02*w0 - a02/a12 + 1.0) / denominator;
        let tau = quadratic_roots(cb, cc)[root_index];
        let depth2 = sqrt(max(0.0, a12 / (tau*(tau - 2.0*m12) + 1.0)));
        return refine_lambda(vec3<f32>(w0*depth2 + w1*tau*depth2, tau*depth2, depth2), a01, a02, a12, m01, m02, m12);
    }
    if (abs(pq.x) <= PNP_EPS) { return vec3<f32>(0.0); }
    let w0 = -pq.y / pq.x; let w1 = -pq.z / pq.x;
    let a = a01 / a12;
    let denominator = -a*w1*w1 + 2.0*a*m12*w1 - a + 1.0;
    if (abs(denominator) <= PNP_EPS) { return vec3<f32>(0.0); }
    let cb = 2.0 * (a*m12*w0 - m01 - a*w0*w1) / denominator;
    let cc = (1.0 - a*w0*w0) / denominator;
    let tau = quadratic_roots(cb, cc)[root_index];
    let depth0 = sqrt(max(0.0, a01 / (tau*(tau - 2.0*m01) + 1.0)));
    return refine_lambda(vec3<f32>(depth0, tau*depth0, w0*depth0 + w1*tau*depth0), a01, a02, a12, m01, m02, m12);
}

@compute @workgroup_size(1)
fn generate_p3p_candidates() {
    candidate_params = single_candidate_params;
    use_batch_candidates = false;
    let geometry = p3p_setup(candidate_params.triple);
    let coefficients = p3p_cubic_coefficients(geometry);
    if (!finite(coefficients.x) || !finite(coefficients.y) || !finite(coefficients.z)) {
        single_candidate_models[0] = invalid_model(); single_candidate_models[1] = invalid_model();
        single_candidate_models[2] = invalid_model(); single_candidate_models[3] = invalid_model();
        return;
    }
    let cubic = cubic_single_real(coefficients.x, coefficients.y, coefficients.z);
    let pair = compute_pq(p3p_constraint_matrix(geometry, cubic.y));
    let first_elimination = abs(pair.first.x) <= abs(pair.first.y);
    single_candidate_models[0] = model_from_depths(geometry, depths_from_pq(geometry, pair.first, first_elimination, 0u));
    single_candidate_models[1] = model_from_depths(geometry, depths_from_pq(geometry, pair.first, first_elimination, 1u));
    let second_elimination = abs(pair.second.x) <= abs(pair.second.y);
    single_candidate_models[2] = model_from_depths(geometry, depths_from_pq(geometry, pair.second, second_elimination, 0u));
    single_candidate_models[3] = model_from_depths(geometry, depths_from_pq(geometry, pair.second, second_elimination, 1u));
}

@compute @workgroup_size(1)
fn generate_p3p_candidate_batch(@builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let parameter_index = workgroup_id.x;
    candidate_params = batch_candidate_params[parameter_index];
    use_batch_candidates = true;
    let output_index = candidate_params.model_offset;
    let geometry = p3p_setup(candidate_params.triple);
    let coefficients = p3p_cubic_coefficients(geometry);
    if (!finite(coefficients.x) || !finite(coefficients.y) || !finite(coefficients.z)) {
        batch_candidate_models[output_index] = invalid_model();
        batch_candidate_models[output_index + 1u] = invalid_model();
        batch_candidate_models[output_index + 2u] = invalid_model();
        batch_candidate_models[output_index + 3u] = invalid_model();
        return;
    }
    let cubic = cubic_single_real(coefficients.x, coefficients.y, coefficients.z);
    let pair = compute_pq(p3p_constraint_matrix(geometry, cubic.y));
    let first_elimination = abs(pair.first.x) <= abs(pair.first.y);
    batch_candidate_models[output_index] = model_from_depths(geometry, depths_from_pq(geometry, pair.first, first_elimination, 0u));
    batch_candidate_models[output_index + 1u] = model_from_depths(geometry, depths_from_pq(geometry, pair.first, first_elimination, 1u));
    let second_elimination = abs(pair.second.x) <= abs(pair.second.y);
    batch_candidate_models[output_index + 2u] = model_from_depths(geometry, depths_from_pq(geometry, pair.second, second_elimination, 0u));
    batch_candidate_models[output_index + 3u] = model_from_depths(geometry, depths_from_pq(geometry, pair.second, second_elimination, 1u));
}

struct PnpFocalScoringParams {
    model_count: u32,
    observation_count: u32,
    selected_model: u32,
    pad1: u32,
    threshold_squared: f32,
    pad2: f32,
    pad3: f32,
    pad4: f32,
}

struct PnpFocalSupport {
    inliers: u32,
    pad0: u32,
    residual_sum: f32,
    pad1: f32,
}

@group(2) @binding(0)
var<storage, read> scoring_models: array<PnpFocalModel>;
@group(2) @binding(1)
var<storage, read> scoring_image_points: array<PnpFocalImagePoint>;
@group(2) @binding(2)
var<storage, read> scoring_object_points: array<PnpFocalObjectPoint>;
@group(2) @binding(3)
var<storage, read_write> scoring_supports: array<PnpFocalSupport>;
@group(2) @binding(4)
var<storage, read_write> scoring_mask: array<u32>;
@group(2) @binding(5)
var<uniform> scoring_params: PnpFocalScoringParams;

var<workgroup> scoring_inliers: array<u32, 64>;
var<workgroup> scoring_residuals: array<f32, 64>;

fn rigid_focal_rotation(model: PnpFocalModel) -> bool {
    let row0 = model.row0.xyz;
    let row1 = model.row1.xyz;
    let row2 = model.row2.xyz;
    let determinant = dot(row0, cross(row1, row2));
    return abs(dot(row0, row0) - 1.0) <= 5.0e-2
        && abs(dot(row1, row1) - 1.0) <= 5.0e-2
        && abs(dot(row2, row2) - 1.0) <= 5.0e-2
        && abs(dot(row0, row1)) <= 5.0e-2
        && abs(dot(row0, row2)) <= 5.0e-2
        && abs(dot(row1, row2)) <= 5.0e-2
        && abs(determinant - 1.0) <= 5.0e-2;
}

fn focal_residual(model: PnpFocalModel, index: u32) -> f32 {
    let focal = exp(model.log_focal);
    let object = scoring_object_points[index];
    let camera = vec3<f32>(
        dot(model.row0.xyz, vec3<f32>(object.x, object.y, object.z)) + model.row0.w,
        dot(model.row1.xyz, vec3<f32>(object.x, object.y, object.z)) + model.row1.w,
        dot(model.row2.xyz, vec3<f32>(object.x, object.y, object.z)) + model.row2.w,
    );
    if (!rigid_focal_rotation(model) || !finite(focal) || focal <= 0.0 || !finite(camera.z) || camera.z <= PNP_EPS) {
        return bitcast<f32>(0x7f800000u);
    }
    let observed = scoring_image_points[index];
    let delta = vec2<f32>(observed.x, observed.y) - focal * camera.xy / camera.z;
    let residual = dot(delta, delta);
    return select(bitcast<f32>(0x7f800000u), residual, finite(residual));
}

@compute @workgroup_size(64)
fn score_focal_models(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let model_index = workgroup_id.x;
    let lane = local_id.x;
    if (model_index >= scoring_params.model_count) {
        return;
    }

    var inliers = 0u;
    var residual_sum = 0.0;
    var observation = lane;
    loop {
        if (observation >= scoring_params.observation_count) {
            break;
        }
        let residual = focal_residual(scoring_models[model_index], observation);
        if (finite(residual) && residual <= scoring_params.threshold_squared) {
            inliers = inliers + 1u;
            residual_sum = residual_sum + residual;
        }
        observation = observation + 64u;
    }
    scoring_inliers[lane] = inliers;
    scoring_residuals[lane] = residual_sum;
    workgroupBarrier();

    var stride = 32u;
    loop {
        if (lane < stride) {
            scoring_inliers[lane] = scoring_inliers[lane] + scoring_inliers[lane + stride];
            scoring_residuals[lane] = scoring_residuals[lane] + scoring_residuals[lane + stride];
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    if (lane == 0u) {
        scoring_supports[model_index].inliers = scoring_inliers[0];
        scoring_supports[model_index].pad0 = 0u;
        scoring_supports[model_index].residual_sum = scoring_residuals[0];
        scoring_supports[model_index].pad1 = 0.0;
    }
}

@compute @workgroup_size(64)
fn write_focal_mask(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let observation = global_id.x;
    if (observation >= scoring_params.observation_count) {
        return;
    }
    let residual = focal_residual(scoring_models[scoring_params.selected_model], observation);
    scoring_mask[observation] = select(0u, 1u, finite(residual) && residual <= scoring_params.threshold_squared);
}

struct PnpFocalSelectionParams {
    model_count: u32,
    observation_count: u32,
    pad0: u32,
    pad1: u32,
    min_focal: f32,
    max_focal: f32,
    pad2: f32,
    pad3: f32,
}

@group(5) @binding(0)
var<storage, read> selection_models: array<PnpFocalModel>;
@group(5) @binding(1)
var<storage, read> selection_supports: array<PnpFocalSupport>;
@group(5) @binding(2)
var<storage, read_write> selection_result: array<PnpFocalResult>;
@group(5) @binding(3)
var<storage, read_write> selected_focal_model: array<PnpFocalModel>;
@group(5) @binding(4)
var<uniform> selection_params: PnpFocalSelectionParams;
@group(5) @binding(5)
var<storage, read_write> selected_focal_support: array<PnpFocalSupport>;

var<workgroup> selection_indices: array<u32, 64>;
var<workgroup> selection_inliers: array<u32, 64>;
var<workgroup> selection_residuals: array<f32, 64>;
var<workgroup> selection_valid: array<u32, 64>;

fn valid_focal_model(model: PnpFocalModel) -> bool {
    let focal = exp(model.log_focal);
    return finite(focal) && focal >= selection_params.min_focal && focal <= selection_params.max_focal
        && finite(model.row0.x) && finite(model.row0.y) && finite(model.row0.z) && finite(model.row0.w)
        && finite(model.row1.x) && finite(model.row1.y) && finite(model.row1.z) && finite(model.row1.w)
        && finite(model.row2.x) && finite(model.row2.y) && finite(model.row2.z) && finite(model.row2.w)
        && rigid_focal_rotation(model);
}

fn selection_is_better(
    candidate_valid: u32,
    candidate_inliers: u32,
    candidate_residual: f32,
    candidate_index: u32,
    current_valid: u32,
    current_inliers: u32,
    current_residual: f32,
    current_index: u32,
) -> bool {
    if (candidate_valid == 0u) { return false; }
    if (current_valid == 0u) { return true; }
    return candidate_inliers > current_inliers
        || (candidate_inliers == current_inliers
            && (candidate_residual < current_residual
                || (candidate_residual == current_residual && candidate_index < current_index)));
}

@compute @workgroup_size(64)
fn select_focal_model(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let lane = local_id.x;
    var best_index = 0u;
    var best_inliers = 0u;
    var best_residual = bitcast<f32>(0x7f800000u);
    var best_valid = 0u;
    var model_index = lane;
    loop {
        if (model_index >= selection_params.model_count) { break; }
        let model = selection_models[model_index];
        let support = selection_supports[model_index];
        let valid = select(0u, 1u,
            valid_focal_model(model) && support.inliers <= selection_params.observation_count
                && finite(support.residual_sum) && support.residual_sum >= 0.0);
        if (selection_is_better(
            valid, support.inliers, support.residual_sum, model_index,
            best_valid, best_inliers, best_residual, best_index,
        )) {
            best_index = model_index;
            best_inliers = support.inliers;
            best_residual = support.residual_sum;
            best_valid = valid;
        }
        model_index = model_index + 64u;
    }
    selection_indices[lane] = best_index;
    selection_inliers[lane] = best_inliers;
    selection_residuals[lane] = best_residual;
    selection_valid[lane] = best_valid;
    workgroupBarrier();

    var stride = 32u;
    loop {
        if (lane < stride && selection_is_better(
            selection_valid[lane + stride], selection_inliers[lane + stride],
            selection_residuals[lane + stride], selection_indices[lane + stride],
            selection_valid[lane], selection_inliers[lane],
            selection_residuals[lane], selection_indices[lane],
        )) {
            selection_indices[lane] = selection_indices[lane + stride];
            selection_inliers[lane] = selection_inliers[lane + stride];
            selection_residuals[lane] = selection_residuals[lane + stride];
            selection_valid[lane] = selection_valid[lane + stride];
        }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride = stride / 2u;
    }
    if (lane == 0u) {
        let selected = selection_indices[0];
        selection_result[0].selected_model = selected;
        selection_result[0].inliers = selection_inliers[0];
        selection_result[0].valid = selection_valid[0];
        selection_result[0].pad0 = 0u;
        selection_result[0].residual_sum = selection_residuals[0];
        selection_result[0].focal = select(0.0, exp(selection_models[selected].log_focal), selection_valid[0] != 0u);
        selection_result[0].pad1 = 0.0;
        selection_result[0].pad2 = 0.0;
        if (selection_valid[0] != 0u) {
            selected_focal_model[0] = selection_models[selected];
            selected_focal_support[0].inliers = selection_inliers[0];
            selected_focal_support[0].pad0 = selection_inliers[0];
            selected_focal_support[0].residual_sum = selection_residuals[0];
            selected_focal_support[0].pad1 = 0.0;
        } else {
            selected_focal_model[0] = invalid_model();
            selected_focal_support[0].inliers = 0u;
            selected_focal_support[0].pad0 = 0u;
            selected_focal_support[0].residual_sum = 0.0;
            selected_focal_support[0].pad1 = 0.0;
        }
    }
}

struct PnpFocalAcceptanceParams {
    observation_count: u32,
    pad0: u32,
    min_focal: f32,
    max_focal: f32,
}

@group(6) @binding(0)
var<storage, read_write> current_focal_model: array<PnpFocalModel>;
@group(6) @binding(1)
var<storage, read_write> current_focal_support: array<PnpFocalSupport>;
@group(6) @binding(2)
var<storage, read_write> current_focal_mask: array<u32>;
@group(6) @binding(3)
var<storage, read> candidate_focal_model: array<PnpFocalModel>;
@group(6) @binding(4)
var<storage, read> candidate_focal_support: array<PnpFocalSupport>;
@group(6) @binding(5)
var<storage, read> candidate_focal_mask: array<u32>;
@group(6) @binding(6)
var<storage, read> candidate_refine_status: array<u32>;
@group(6) @binding(7)
var<uniform> acceptance_params: PnpFocalAcceptanceParams;

var<workgroup> accept_refined_focal_model: u32;

@compute @workgroup_size(64)
fn accept_refined_focal_candidate(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let lane = local_id.x;
    if (lane == 0u) {
        let current = current_focal_support[0];
        let candidate = candidate_focal_support[0];
        let candidate_valid = candidate_refine_status[0] != 0u
            && candidate.inliers <= acceptance_params.observation_count
            && finite(candidate.residual_sum) && candidate.residual_sum >= 0.0
            && finite(exp(candidate_focal_model[0].log_focal))
            && exp(candidate_focal_model[0].log_focal) >= acceptance_params.min_focal
            && exp(candidate_focal_model[0].log_focal) <= acceptance_params.max_focal
            && rigid_focal_rotation(candidate_focal_model[0]);
        accept_refined_focal_model = select(0u, 1u,
            candidate_valid && (candidate.inliers > current.inliers
                || (candidate.inliers == current.inliers
                    && candidate.residual_sum < current.residual_sum)));
        if (accept_refined_focal_model != 0u) {
            current_focal_model[0] = candidate_focal_model[0];
            current_focal_support[0] = candidate;
            current_focal_support[0].pad0 = current.pad0;
        }
    }
    workgroupBarrier();
    if (accept_refined_focal_model != 0u) {
        var observation = lane;
        loop {
            if (observation >= acceptance_params.observation_count) { break; }
            current_focal_mask[observation] = candidate_focal_mask[observation];
            observation = observation + 64u;
        }
    }
}

struct PnpFocalRefineParams {
    observation_count: u32,
    pad0: u32,
    damping: f32,
    pad1: f32,
}

@group(4) @binding(0)
var<storage, read> refine_input: array<PnpFocalModel>;
@group(4) @binding(1)
var<storage, read> refine_image_points: array<PnpFocalImagePoint>;
@group(4) @binding(2)
var<storage, read> refine_object_points: array<PnpFocalObjectPoint>;
@group(4) @binding(3)
var<storage, read> refine_mask: array<u32>;
@group(4) @binding(4)
var<storage, read_write> refine_output: array<PnpFocalModel>;
@group(4) @binding(5)
var<storage, read_write> refine_status: array<u32>;
@group(4) @binding(6)
var<uniform> refine_params: PnpFocalRefineParams;

var<workgroup> refine_terms: array<array<f32, 56>, 64>;
var<workgroup> refine_invalid_observations: array<u32, 64>;

fn refine_camera(model: PnpFocalModel, object: PnpFocalObjectPoint) -> vec3<f32> {
    let point = vec3<f32>(object.x, object.y, object.z);
    return vec3<f32>(
        dot(model.row0.xyz, point) + model.row0.w,
        dot(model.row1.xyz, point) + model.row1.w,
        dot(model.row2.xyz, point) + model.row2.w,
    );
}

@compute @workgroup_size(64)
fn refine_focal_model(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let lane = local_id.x;
    var term = 0u;
    loop {
        if (term == 56u) { break; }
        refine_terms[lane][term] = 0.0;
        term = term + 1u;
    }
    refine_invalid_observations[lane] = 0u;

    let model = refine_input[0];
    let focal = exp(model.log_focal);
    var observation = lane;
    loop {
        if (observation >= refine_params.observation_count) { break; }
        if (refine_mask[observation] != 0u) {
            let camera = refine_camera(model, refine_object_points[observation]);
            if (finite(focal) && focal > 0.0 && finite(camera.x) && finite(camera.y) && finite(camera.z) && camera.z > PNP_EPS) {
                let inverse_z = 1.0 / camera.z;
                let projected = focal * camera.xy * inverse_z;
                let observed = refine_image_points[observation];
                let residual = vec2<f32>(observed.x, observed.y) - projected;
                let axes = array<vec3<f32>, 3>(
                    vec3<f32>(0.0, -camera.z, camera.y),
                    vec3<f32>(camera.z, 0.0, -camera.x),
                    vec3<f32>(-camera.y, camera.x, 0.0),
                );
                var ju: array<f32, 7>;
                var jv: array<f32, 7>;
                var parameter = 0u;
                loop {
                    if (parameter == 7u) { break; }
                    var differential = vec3<f32>(0.0);
                    if (parameter < 3u) {
                        differential = axes[parameter];
                    } else if (parameter < 6u) {
                        differential[parameter - 3u] = 1.0;
                    }
                    ju[parameter] = focal * (differential.x * camera.z - camera.x * differential.z) * inverse_z * inverse_z;
                    jv[parameter] = focal * (differential.y * camera.z - camera.y * differential.z) * inverse_z * inverse_z;
                    if (parameter == 6u) {
                        ju[parameter] = projected.x;
                        jv[parameter] = projected.y;
                    }
                    parameter = parameter + 1u;
                }
                var row = 0u;
                loop {
                    if (row == 7u) { break; }
                    refine_terms[lane][49u + row] = refine_terms[lane][49u + row] + ju[row] * residual.x + jv[row] * residual.y;
                    var column = 0u;
                    loop {
                        if (column == 7u) { break; }
                        let index = row * 7u + column;
                        refine_terms[lane][index] = refine_terms[lane][index] + ju[row] * ju[column] + jv[row] * jv[column];
                        column = column + 1u;
                    }
                    row = row + 1u;
                }
            } else {
                refine_invalid_observations[lane] = 1u;
            }
        }
        observation = observation + 64u;
    }
    workgroupBarrier();

    var stride = 32u;
    loop {
        if (lane < stride) {
            refine_invalid_observations[lane] = max(
                refine_invalid_observations[lane],
                refine_invalid_observations[lane + stride],
            );
            var index = 0u;
            loop {
                if (index == 56u) { break; }
                refine_terms[lane][index] = refine_terms[lane][index] + refine_terms[lane + stride][index];
                index = index + 1u;
            }
        }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride = stride / 2u;
    }

    if (lane != 0u) { return; }
    if (refine_invalid_observations[0] != 0u) {
        refine_status[0] = 0u;
        return;
    }
    var system: array<f32, 49>;
    var rank_system: array<f32, 49>;
    var rhs: array<f32, 7>;
    var index = 0u;
    var matrix_scale = 0.0;
    loop {
        if (index == 49u) { break; }
        system[index] = refine_terms[0][index];
        rank_system[index] = system[index];
        matrix_scale = max(matrix_scale, abs(system[index]));
        index = index + 1u;
    }

    // Damping must not turn a rank-deficient raw system into a usable pose update.
    let rank_tolerance = max(PNP_EPS, matrix_scale * 1.0e-5);
    var rank_pivot = 0u;
    loop {
        if (rank_pivot == 7u) { break; }
        var pivot_row = rank_pivot;
        var pivot_magnitude = abs(rank_system[rank_pivot * 7u + rank_pivot]);
        var candidate_row = rank_pivot + 1u;
        loop {
            if (candidate_row == 7u) { break; }
            let candidate_magnitude = abs(rank_system[candidate_row * 7u + rank_pivot]);
            if (candidate_magnitude > pivot_magnitude) {
                pivot_row = candidate_row;
                pivot_magnitude = candidate_magnitude;
            }
            candidate_row = candidate_row + 1u;
        }
        if (!finite(pivot_magnitude) || pivot_magnitude <= rank_tolerance) {
            refine_status[0] = 0u;
            return;
        }
        if (pivot_row != rank_pivot) {
            var swap_column = 0u;
            loop {
                if (swap_column == 7u) { break; }
                let left_index = rank_pivot * 7u + swap_column;
                let right_index = pivot_row * 7u + swap_column;
                let value = rank_system[left_index];
                rank_system[left_index] = rank_system[right_index];
                rank_system[right_index] = value;
                swap_column = swap_column + 1u;
            }
        }
        let pivot_value = rank_system[rank_pivot * 7u + rank_pivot];
        var lower_row = rank_pivot + 1u;
        loop {
            if (lower_row == 7u) { break; }
            let factor = rank_system[lower_row * 7u + rank_pivot] / pivot_value;
            var column = rank_pivot;
            loop {
                if (column == 7u) { break; }
                let destination = lower_row * 7u + column;
                rank_system[destination] = rank_system[destination]
                    - factor * rank_system[rank_pivot * 7u + column];
                column = column + 1u;
            }
            lower_row = lower_row + 1u;
        }
        rank_pivot = rank_pivot + 1u;
    }

    var row = 0u;
    loop {
        if (row == 7u) { break; }
        let diagonal = row * 7u + row;
        system[diagonal] = system[diagonal] + refine_params.damping * max(1.0, abs(system[diagonal]));
        rhs[row] = refine_terms[0][49u + row];
        row = row + 1u;
    }

    var pivot = 0u;
    loop {
        if (pivot == 7u) { break; }
        let pivot_value = system[pivot * 7u + pivot];
        if (!finite(pivot_value) || abs(pivot_value) <= PNP_EPS || !finite(rhs[pivot])) {
            refine_status[0] = 0u;
            return;
        }
        var lower = pivot + 1u;
        loop {
            if (lower == 7u) { break; }
            let factor = system[lower * 7u + pivot] / pivot_value;
            var column = pivot;
            loop {
                if (column == 7u) { break; }
                system[lower * 7u + column] = system[lower * 7u + column] - factor * system[pivot * 7u + column];
                column = column + 1u;
            }
            rhs[lower] = rhs[lower] - factor * rhs[pivot];
            lower = lower + 1u;
        }
        pivot = pivot + 1u;
    }
    var delta: array<f32, 7>;
    var reverse = 7u;
    loop {
        if (reverse == 0u) { break; }
        reverse = reverse - 1u;
        var value = rhs[reverse];
        var column = reverse + 1u;
        loop {
            if (column == 7u) { break; }
            value = value - system[reverse * 7u + column] * delta[column];
            column = column + 1u;
        }
        let diagonal = system[reverse * 7u + reverse];
        if (!finite(value) || !finite(diagonal) || abs(diagonal) <= PNP_EPS) {
            refine_status[0] = 0u;
            return;
        }
        delta[reverse] = clamp(value / diagonal, -0.25, 0.25);
    }

    let rotation = vec3<f32>(delta[0], delta[1], delta[2]);
    let translation = vec3<f32>(model.row0.w, model.row1.w, model.row2.w);
    let rotated_translation = translation + cross(rotation, translation) + vec3<f32>(delta[3], delta[4], delta[5]);
    var refined: PnpFocalModel;
    refined.row0 = vec4<f32>(model.row0.xyz - rotation.z * model.row1.xyz + rotation.y * model.row2.xyz, rotated_translation.x);
    refined.row1 = vec4<f32>(rotation.z * model.row0.xyz + model.row1.xyz - rotation.x * model.row2.xyz, rotated_translation.y);
    refined.row2 = vec4<f32>(-rotation.y * model.row0.xyz + rotation.x * model.row1.xyz + model.row2.xyz, rotated_translation.z);
    refined.log_focal = model.log_focal + delta[6];
    refined.pad0 = 0.0;
    refined.pad1 = 0.0;
    refined.pad2 = 0.0;
    let refined_focal = exp(refined.log_focal);
    if (!finite(refined_focal) || refined_focal <= 0.0
        || !finite(refined.row0.x) || !finite(refined.row0.y) || !finite(refined.row0.z) || !finite(refined.row0.w)
        || !finite(refined.row1.x) || !finite(refined.row1.y) || !finite(refined.row1.z) || !finite(refined.row1.w)
        || !finite(refined.row2.x) || !finite(refined.row2.y) || !finite(refined.row2.z) || !finite(refined.row2.w)) {
        refine_status[0] = 0u;
        return;
    }
    refine_output[0] = refined;
    refine_status[0] = 1u;
}
