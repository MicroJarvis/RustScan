struct Params {
    len: u32,
    value: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> output_values: array<u32>;
@group(0) @binding(1) var<storage, read> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.len) {
        output_values[idx] = params.value;
    }
}
