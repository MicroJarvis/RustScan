struct GpuKeypoint {
    x: f32,
    y: f32,
    sigma: f32,
    response: f32,
    angle: f32,
    octave: i32,
    level: i32,
    valid: u32,
}

struct DetectorCounters {
    count: atomic<u32>,
    overflow: atomic<u32>,
    pad0: u32,
    pad1: u32,
}

struct DetectorParams {
    width: u32,
    height: u32,
    levels: u32,
    capacity: u32,
    peak_threshold: f32,
    edge_threshold: f32,
    sigma0: f32,
    octave_scale: f32,
    octave: i32,
    octave_resolution: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> dogs: array<f32>;
@group(0) @binding(1) var<storage, read_write> candidates: array<GpuKeypoint>;
@group(0) @binding(2) var<storage, read_write> counters: DetectorCounters;
@group(0) @binding(3) var<uniform> params: DetectorParams;

fn dog_value(x: i32, y: i32, level: i32) -> f32 {
    return dogs[(u32(level) * params.height + u32(y)) * params.width + u32(x)];
}

fn hessian_offset(x: i32, y: i32, level: i32) -> vec4<f32> {
    let center = dog_value(x, y, level);
    let gx = 0.5 * (dog_value(x + 1, y, level) - dog_value(x - 1, y, level));
    let gy = 0.5 * (dog_value(x, y + 1, level) - dog_value(x, y - 1, level));
    let gs = 0.5 * (dog_value(x, y, level + 1) - dog_value(x, y, level - 1));

    let a = dog_value(x + 1, y, level) + dog_value(x - 1, y, level) - 2.0 * center;
    let d = dog_value(x, y + 1, level) + dog_value(x, y - 1, level) - 2.0 * center;
    let f = dog_value(x, y, level + 1) + dog_value(x, y, level - 1) - 2.0 * center;
    let b = 0.25 * (dog_value(x + 1, y + 1, level) - dog_value(x + 1, y - 1, level)
        - dog_value(x - 1, y + 1, level) + dog_value(x - 1, y - 1, level));
    let c = 0.25 * (dog_value(x + 1, y, level + 1) - dog_value(x + 1, y, level - 1)
        - dog_value(x - 1, y, level + 1) + dog_value(x - 1, y, level - 1));
    let e = 0.25 * (dog_value(x, y + 1, level + 1) - dog_value(x, y + 1, level - 1)
        - dog_value(x, y - 1, level + 1) + dog_value(x, y - 1, level - 1));

    let det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
    if (abs(det) < 1.0e-10) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let inv00 = d * f - e * e;
    let inv01 = c * e - b * f;
    let inv02 = b * e - c * d;
    let inv11 = a * f - c * c;
    let inv12 = b * c - a * e;
    let inv22 = a * d - b * b;
    let ox = -(inv00 * gx + inv01 * gy + inv02 * gs) / det;
    let oy = -(inv01 * gx + inv11 * gy + inv12 * gs) / det;
    let os = -(inv02 * gx + inv12 * gy + inv22 * gs) / det;
    return vec4<f32>(ox, oy, os, det);
}

fn is_strict_extremum(x: i32, y: i32, level: i32, center: f32) -> bool {
    var is_maximum = true;
    var is_minimum = true;
    for (var ds = -1; ds <= 1; ds = ds + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                if (dx == 0 && dy == 0 && ds == 0) {
                    continue;
                }
                let neighbor = dog_value(x + dx, y + dy, level + ds);
                if (center <= neighbor) {
                    is_maximum = false;
                }
                if (center >= neighbor) {
                    is_minimum = false;
                }
            }
        }
    }
    return is_maximum || is_minimum;
}

@compute @workgroup_size(8, 8, 1)
fn detect_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (params.levels < 3u || id.z >= params.levels - 2u) {
        return;
    }
    if (id.x == 0u || id.y == 0u || id.x + 1u >= params.width || id.y + 1u >= params.height) {
        return;
    }

    var x = i32(id.x);
    var y = i32(id.y);
    var level = i32(id.z + 1u);
    let initial = dog_value(x, y, level);
    if (abs(initial) < params.peak_threshold || !is_strict_extremum(x, y, level, initial)) {
        return;
    }

    var offset = vec4<f32>(0.0);
    var converged = false;
    for (var iteration = 0; iteration < 5; iteration = iteration + 1) {
        if (x <= 0 || y <= 0 || x + 1 >= i32(params.width) || y + 1 >= i32(params.height)
            || level <= 0 || level + 1 >= i32(params.levels)) {
            return;
        }
        offset = hessian_offset(x, y, level);
        if (offset.w == 0.0 || any(abs(offset.xyz) > vec3<f32>(1.5))) {
            return;
        }
        if (all(abs(offset.xyz) <= vec3<f32>(0.5))) {
            converged = true;
            break;
        }
        if (offset.x > 0.5) { x = x + 1; }
        if (offset.x < -0.5) { x = x - 1; }
        if (offset.y > 0.5) { y = y + 1; }
        if (offset.y < -0.5) { y = y - 1; }
        if (offset.z > 0.5) { level = level + 1; }
        if (offset.z < -0.5) { level = level - 1; }
    }
    if (!converged) {
        return;
    }

    let center = dog_value(x, y, level);
    let gradient = vec3<f32>(
        0.5 * (dog_value(x + 1, y, level) - dog_value(x - 1, y, level)),
        0.5 * (dog_value(x, y + 1, level) - dog_value(x, y - 1, level)),
        0.5 * (dog_value(x, y, level + 1) - dog_value(x, y, level - 1)),
    );
    let contrast = center + 0.5 * dot(gradient, offset.xyz);
    if (abs(contrast) < params.peak_threshold) {
        return;
    }

    let dxx = dog_value(x + 1, y, level) + dog_value(x - 1, y, level) - 2.0 * center;
    let dyy = dog_value(x, y + 1, level) + dog_value(x, y - 1, level) - 2.0 * center;
    let dxy = 0.25 * (dog_value(x + 1, y + 1, level) - dog_value(x + 1, y - 1, level)
        - dog_value(x - 1, y + 1, level) + dog_value(x - 1, y - 1, level));
    let spatial_det = dxx * dyy - dxy * dxy;
    let trace = dxx + dyy;
    let edge = params.edge_threshold;
    if (spatial_det <= 0.0 || trace * trace * edge >= (edge + 1.0) * (edge + 1.0) * spatial_det) {
        return;
    }

    let slot = atomicAdd(&counters.count, 1u);
    if (slot >= params.capacity) {
        atomicStore(&counters.overflow, 1u);
        return;
    }
    let scale_exponent = (f32(level) + offset.z) / f32(params.octave_resolution);
    candidates[slot] = GpuKeypoint(
        f32(x) + offset.x,
        f32(y) + offset.y,
        params.sigma0 * pow(2.0, scale_exponent) * params.octave_scale,
        abs(contrast),
        0.0,
        params.octave,
        level,
        1u,
    );
}
