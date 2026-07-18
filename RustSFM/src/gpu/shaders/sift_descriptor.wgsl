const TAU: f32 = 6.283185307179586;
const DESCRIPTOR_BINS: u32 = 8u;
const DESCRIPTOR_CELLS: u32 = 4u;
const DESCRIPTOR_SIZE: u32 = 128u;

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

struct DescriptorParams {
    width: u32,
    height: u32,
    keypoint_count: u32,
    root_sift: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

@group(0) @binding(0) var<storage, read> image: array<f32>;
@group(0) @binding(1) var<storage, read> keypoints: array<GpuKeypoint>;
@group(0) @binding(2) var<storage, read_write> descriptors: array<f32>;
@group(0) @binding(3) var<uniform> params: DescriptorParams;

fn image_value(x: i32, y: i32) -> f32 {
    return image[u32(y) * params.width + u32(x)];
}

fn descriptor_index(cell_x: u32, cell_y: u32, bin: u32) -> u32 {
    return (cell_y * DESCRIPTOR_CELLS + cell_x) * DESCRIPTOR_BINS + bin;
}

@compute @workgroup_size(64)
fn descriptor_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.keypoint_count) {
        return;
    }
    let keypoint = keypoints[id.x];
    if (keypoint.valid == 0u) {
        return;
    }

    var histogram: array<f32, 128>;
    for (var index = 0u; index < DESCRIPTOR_SIZE; index = index + 1u) {
        histogram[index] = 0.0;
    }

    let sigma = max(keypoint.sigma, 0.5);
    let sample_scale = 4.0 * sigma;
    let radius = i32(ceil(8.0 * sigma));
    let cosine = cos(keypoint.angle);
    let sine = sin(keypoint.angle);
    let window_denom = 2.0 * (0.5 * f32(radius)) * (0.5 * f32(radius));
    let center_x = i32(round(keypoint.x));
    let center_y = i32(round(keypoint.y));

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
            let local_x = (cosine * f32(dx) + sine * f32(dy)) / sample_scale + 2.0;
            let local_y = (-sine * f32(dx) + cosine * f32(dy)) / sample_scale + 2.0;
            if (local_x < -1.0 || local_x >= 4.0 || local_y < -1.0 || local_y >= 4.0) {
                continue;
            }
            let cell_x = i32(floor(local_x));
            let cell_y = i32(floor(local_y));
            let fraction_x = local_x - f32(cell_x);
            let fraction_y = local_y - f32(cell_y);
            var relative_angle = atan2(gradient_y, gradient_x) - keypoint.angle;
            while (relative_angle < 0.0) { relative_angle = relative_angle + TAU; }
            while (relative_angle >= TAU) { relative_angle = relative_angle - TAU; }
            let orientation = relative_angle * f32(DESCRIPTOR_BINS) / TAU;
            let orientation_bin = u32(floor(orientation)) % DESCRIPTOR_BINS;
            let orientation_fraction = orientation - floor(orientation);
            let spatial_weight = exp(-(f32(dx * dx + dy * dy)) / window_denom);

            for (var sy = 0; sy < 2; sy = sy + 1) {
                let target_y = cell_y + sy;
                if (target_y < 0 || target_y >= 4) { continue; }
                let weight_y = select(1.0 - fraction_y, fraction_y, sy == 1);
                for (var sx = 0; sx < 2; sx = sx + 1) {
                    let target_x = cell_x + sx;
                    if (target_x < 0 || target_x >= 4) { continue; }
                    let weight_x = select(1.0 - fraction_x, fraction_x, sx == 1);
                    let contribution = magnitude * spatial_weight * weight_x * weight_y;
                    let first = descriptor_index(u32(target_x), u32(target_y), orientation_bin);
                    let second = descriptor_index(
                        u32(target_x),
                        u32(target_y),
                        (orientation_bin + 1u) % DESCRIPTOR_BINS,
                    );
                    histogram[first] = histogram[first] + contribution * (1.0 - orientation_fraction);
                    histogram[second] = histogram[second] + contribution * orientation_fraction;
                }
            }
        }
    }

    var l2 = 0.0;
    var l1 = 0.0;
    for (var index = 0u; index < DESCRIPTOR_SIZE; index = index + 1u) {
        l1 = l1 + max(histogram[index], 0.0);
        l2 = l2 + histogram[index] * histogram[index];
    }
    if (params.root_sift != 0u) {
        for (var index = 0u; index < DESCRIPTOR_SIZE; index = index + 1u) {
            histogram[index] = sqrt(max(histogram[index], 0.0) / max(l1, 1.0e-12));
        }
        l2 = 0.0;
        for (var index = 0u; index < DESCRIPTOR_SIZE; index = index + 1u) {
            l2 = l2 + histogram[index] * histogram[index];
        }
    } else {
        let norm = max(sqrt(l2), 1.0e-12);
        for (var index = 0u; index < DESCRIPTOR_SIZE; index = index + 1u) {
            histogram[index] = histogram[index] / norm;
        }
    }
    if (params.root_sift != 0u) {
        let norm = max(sqrt(l2), 1.0e-12);
        for (var index = 0u; index < DESCRIPTOR_SIZE; index = index + 1u) {
            histogram[index] = histogram[index] / norm;
        }
    }
    var clipped_l2 = 0.0;
    for (var index = 0u; index < DESCRIPTOR_SIZE; index = index + 1u) {
        histogram[index] = min(histogram[index], 0.2);
        clipped_l2 = clipped_l2 + histogram[index] * histogram[index];
    }
    let output_norm = max(sqrt(clipped_l2), 1.0e-12);
    let base = id.x * DESCRIPTOR_SIZE;
    for (var index = 0u; index < DESCRIPTOR_SIZE; index = index + 1u) {
        descriptors[base + index] = histogram[index] / output_norm;
    }
}
