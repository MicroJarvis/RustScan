struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> output_values: array<u32>;
@group(0) @binding(1) var<storage, read> scanned_block_sums: array<u32>;
@group(0) @binding(2) var<storage, read> block_sums: array<u32>;
@group(0) @binding(3) var<storage, read> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let block = idx / 256u;
    let prefix = scanned_block_sums[block] - block_sums[block];
    output_values[idx] = output_values[idx] + prefix;
}
