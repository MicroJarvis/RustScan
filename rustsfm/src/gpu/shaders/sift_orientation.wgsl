const PI: f32 = 3.141592653589793;
const TAU: f32 = 6.283185307179586;
const ORIENTATION_BINS: u32 = 36u;

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

struct OrientationCounters {
    count: atomic<u32>,
    overflow: atomic<u32>,
    pad0: u32,
    pad1: u32,
}

struct OrientationParams {
    width: u32,
    height: u32,
    keypoint_count: u32,
    capacity: u32,
    max_orientations: u32,
    upright: u32,
    peak_ratio: f32,
    pad0: u32,
}

@group(0) @binding(0) var<storage, read> image: array<f32>;
@group(0) @binding(1) var<storage, read> keypoints: array<GpuKeypoint>;
@group(0) @binding(2) var<storage, read_write> oriented: array<GpuKeypoint>;
@group(0) @binding(3) var<storage, read_write> counters: OrientationCounters;
@group(0) @binding(4) var<uniform> params: OrientationParams;

fn image_value(x: i32, y: i32) -> f32 {
    return image[u32(y) * params.width + u32(x)];
}

fn append_orientation(keypoint: GpuKeypoint, angle: f32) {
    let slot = atomicAdd(&counters.count, 1u);
    if (slot >= params.capacity) {
        atomicStore(&counters.overflow, 1u);
        return;
    }
    oriented[slot] = GpuKeypoint(
        keypoint.x,
        keypoint.y,
        keypoint.sigma,
        keypoint.response,
        angle,
        keypoint.octave,
        keypoint.level,
        keypoint.valid,
    );
}

@compute @workgroup_size(64)
fn orientation_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.keypoint_count) {
        return;
    }
    let keypoint = keypoints[id.x];
    if (keypoint.valid == 0u) {
        return;
    }
    if (params.upright != 0u) {
        append_orientation(keypoint, 0.0);
        return;
    }

    var histogram: array<f32, 36>;
    var scratch: array<f32, 36>;
    var selected: array<u32, 36>;
    for (var bin = 0u; bin < ORIENTATION_BINS; bin = bin + 1u) {
        histogram[bin] = 0.0;
        scratch[bin] = 0.0;
        selected[bin] = 0u;
    }

    let window_sigma = max(1.5 * keypoint.sigma, 0.5);
    let radius = i32(ceil(3.0 * window_sigma));
    let center_x = i32(round(keypoint.x));
    let center_y = i32(round(keypoint.y));
    let gaussian_denom = 2.0 * window_sigma * window_sigma;
    for (var dy = -radius; dy <= radius; dy = dy + 1) {
        let y = center_y + dy;
        if (y <= 0 || y + 1 >= i32(params.height)) {
            continue;
        }
        for (var dx = -radius; dx <= radius; dx = dx + 1) {
            let x = center_x + dx;
            if (x <= 0 || x + 1 >= i32(params.width)) {
                continue;
            }
            let gradient_x = image_value(x + 1, y) - image_value(x - 1, y);
            let gradient_y = image_value(x, y + 1) - image_value(x, y - 1);
            let magnitude = sqrt(gradient_x * gradient_x + gradient_y * gradient_y);
            if (magnitude <= 0.0) {
                continue;
            }
            var angle = atan2(gradient_y, gradient_x);
            if (angle < 0.0) {
                angle = angle + TAU;
            }
            let bin = u32(floor(angle * f32(ORIENTATION_BINS) / TAU)) % ORIENTATION_BINS;
            let distance2 = f32(dx * dx + dy * dy);
            let weight = exp(-distance2 / gaussian_denom);
            histogram[bin] = histogram[bin] + weight * magnitude;
        }
    }

    for (var iteration = 0u; iteration < 6u; iteration = iteration + 1u) {
        for (var bin = 0u; bin < ORIENTATION_BINS; bin = bin + 1u) {
            let left = (bin + ORIENTATION_BINS - 1u) % ORIENTATION_BINS;
            let right = (bin + 1u) % ORIENTATION_BINS;
            scratch[bin] = (histogram[left] + histogram[bin] + histogram[right]) / 3.0;
        }
        for (var bin = 0u; bin < ORIENTATION_BINS; bin = bin + 1u) {
            histogram[bin] = scratch[bin];
        }
    }

    var maximum = 0.0;
    for (var bin = 0u; bin < ORIENTATION_BINS; bin = bin + 1u) {
        maximum = max(maximum, histogram[bin]);
    }
    if (maximum <= 0.0) {
        return;
    }

    var emitted = 0u;
    loop {
        if (emitted >= params.max_orientations) {
            break;
        }
        var best_bin = -1;
        var best_value = -1.0;
        for (var bin = 0u; bin < ORIENTATION_BINS; bin = bin + 1u) {
            if (selected[bin] != 0u) {
                continue;
            }
            let left = (bin + ORIENTATION_BINS - 1u) % ORIENTATION_BINS;
            let right = (bin + 1u) % ORIENTATION_BINS;
            let value = histogram[bin];
            if (value >= params.peak_ratio * maximum && value > histogram[left]
                && value > histogram[right] && value > best_value) {
                best_bin = i32(bin);
                best_value = value;
            }
        }
        if (best_bin < 0) {
            break;
        }
        let bin = u32(best_bin);
        selected[bin] = 1u;
        let left = histogram[(bin + ORIENTATION_BINS - 1u) % ORIENTATION_BINS];
        let center = histogram[bin];
        let right = histogram[(bin + 1u) % ORIENTATION_BINS];
        let denominator = left - 2.0 * center + right;
        var offset = 0.0;
        if (abs(denominator) > 1.0e-12) {
            offset = clamp(0.5 * (left - right) / denominator, -0.5, 0.5);
        }
        var angle = TAU * (f32(bin) + offset) / f32(ORIENTATION_BINS);
        if (angle < 0.0) { angle = angle + TAU; }
        if (angle >= TAU) { angle = angle - TAU; }
        append_orientation(keypoint, angle);
        emitted = emitted + 1u;
    }
}
