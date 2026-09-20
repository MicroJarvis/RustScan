struct PyramidParams {
    width: u32,
    height: u32,
    radius: u32,
    direction: u32,
}

@group(0) @binding(0) var<storage, read> source_a: array<f32>;
@group(0) @binding(1) var<storage, read> source_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> destination: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<uniform> params: PyramidParams;

fn source_index(x: i32, y: i32) -> u32 {
    let clamped_x = clamp(x, 0, i32(params.width) - 1);
    let clamped_y = clamp(y, 0, i32(params.height) - 1);
    return u32(clamped_y) * params.width + u32(clamped_x);
}

@compute @workgroup_size(16, 16)
fn gaussian_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }

    var sum = 0.0;
    let radius = i32(params.radius);
    for (var offset = -radius; offset <= radius; offset = offset + 1) {
        var sample_x = i32(id.x);
        var sample_y = i32(id.y);
        if (params.direction == 0u) {
            sample_x = sample_x + offset;
        } else {
            sample_y = sample_y + offset;
        }
        let weight_index = u32(offset + radius);
        sum = sum + source_a[source_index(sample_x, sample_y)] * weights[weight_index];
    }
    destination[id.y * params.width + id.x] = sum;
}

@compute @workgroup_size(16, 16)
fn dog_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }
    let pixel = id.y * params.width + id.x;
    destination[pixel] = source_b[pixel] - source_a[pixel];
}

@compute @workgroup_size(16, 16)
fn downsample_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let output_width = params.width / 2u;
    let output_height = params.height / 2u;
    if (id.x >= output_width || id.y >= output_height) {
        return;
    }
    let source_x = id.x * 2u;
    let source_y = id.y * 2u;
    destination[id.y * output_width + id.x] = source_a[source_y * params.width + source_x];
}
