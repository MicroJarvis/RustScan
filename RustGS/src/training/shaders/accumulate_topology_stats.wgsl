struct Params {
    num_splats: u32,
    sh_coeffs: u32,
    use_actual_visibility: u32,
    collect_actual_visibility: u32,
}

@group(0) @binding(0) var<storage, read> transforms_grad: array<f32>;
@group(0) @binding(1) var<storage, read> screen_grad_stats: array<f32>;
@group(0) @binding(2) var<storage, read> sh_grad: array<f32>;
@group(0) @binding(3) var<storage, read> visible: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_2d_accum: array<f32>;
@group(0) @binding(5) var<storage, read_write> screen_grad_2d_accum: array<f32>;
@group(0) @binding(6) var<storage, read_write> abs_grad_2d_accum: array<f32>;
@group(0) @binding(7) var<storage, read_write> abs_pixel_grad_2d_accum: array<f32>;
@group(0) @binding(8) var<storage, read_write> pixel_coverage_accum: array<f32>;
@group(0) @binding(9) var<storage, read_write> camera_depth_accum: array<f32>;
@group(0) @binding(10) var<storage, read_write> grad_color_accum: array<f32>;
@group(0) @binding(11) var<storage, read_write> num_observations: array<f32>;
@group(0) @binding(12) var<storage, read_write> visible_observations: array<f32>;
@group(0) @binding(13) var<storage, read_write> actual_visible_observations: array<f32>;
@group(0) @binding(14) var<storage, read> params: Params;

fn abs_finite(value: f32) -> f32 {
    let abs_value = abs(value);
    if (abs_value == abs_value) {
        return abs_value;
    }
    return 0.0;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.num_splats) {
        return;
    }

    let transform_base = idx * 10u;
    let grad_2d = (
        abs_finite(transforms_grad[transform_base]) +
        abs_finite(transforms_grad[transform_base + 1u]) +
        abs_finite(transforms_grad[transform_base + 2u])
    ) / 3.0;

    let screen_base = idx * 7u;
    let screen_x = screen_grad_stats[screen_base];
    let screen_y = screen_grad_stats[screen_base + 1u];
    let abs_x = screen_grad_stats[screen_base + 2u];
    let abs_y = screen_grad_stats[screen_base + 3u];
    let abs_pixel = screen_grad_stats[screen_base + 4u];
    let coverage = screen_grad_stats[screen_base + 5u];
    let camera_depth = screen_grad_stats[screen_base + 6u];

    let screen_grad_2d = sqrt(max(screen_x * screen_x + screen_y * screen_y, 0.0));
    let abs_grad_2d = sqrt(max(abs_x * abs_x + abs_y * abs_y, 0.0));

    var grad_color_sum = 0.0;
    let sh_base = idx * params.sh_coeffs * 3u;
    for (var coeff = 0u; coeff < params.sh_coeffs; coeff++) {
        let coeff_base = sh_base + coeff * 3u;
        grad_color_sum += (
            abs_finite(sh_grad[coeff_base]) +
            abs_finite(sh_grad[coeff_base + 1u]) +
            abs_finite(sh_grad[coeff_base + 2u])
        ) / 3.0;
    }
    let grad_color = grad_color_sum / max(f32(params.sh_coeffs), 1.0);

    grad_2d_accum[idx] += grad_2d;
    screen_grad_2d_accum[idx] += screen_grad_2d;
    abs_grad_2d_accum[idx] += abs_grad_2d;
    abs_pixel_grad_2d_accum[idx] += abs_pixel;
    pixel_coverage_accum[idx] += coverage;
    camera_depth_accum[idx] += camera_depth;
    grad_color_accum[idx] += grad_color;
    num_observations[idx] += 1.0;

    let visible_value = visible[idx];
    if (params.collect_actual_visibility != 0u) {
        actual_visible_observations[idx] += visible_value;
    }
    if (params.use_actual_visibility != 0u) {
        visible_observations[idx] += visible_value;
    } else {
        visible_observations[idx] += 1.0;
    }
}
