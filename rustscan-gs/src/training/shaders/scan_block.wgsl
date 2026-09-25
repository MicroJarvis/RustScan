struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input_values: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_values: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(3) var<storage, read> params: Params;

var<workgroup> values: array<u32, 256>;

fn scan_step(local_id: u32, offset: u32) {
    var next = values[local_id];
    if (local_id >= offset) {
        next = next + values[local_id - offset];
    }
    workgroupBarrier();
    values[local_id] = next;
    workgroupBarrier();
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) local_id: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let idx = gid.x;
    // `select` would still evaluate `input_values[idx]` on tail lanes past `len`.
    if (idx < params.len) {
        values[local_id] = input_values[idx];
    } else {
        values[local_id] = 0u;
    }
    workgroupBarrier();

    // 256 lanes, inclusive Hillis-Steele. Offset is uniform so barriers stay uniform.
    scan_step(local_id, 1u);
    scan_step(local_id, 2u);
    scan_step(local_id, 4u);
    scan_step(local_id, 8u);
    scan_step(local_id, 16u);
    scan_step(local_id, 32u);
    scan_step(local_id, 64u);
    scan_step(local_id, 128u);

    if (idx < params.len) {
        output_values[idx] = values[local_id];
    }

    let block_start = group_id.x * 256u;
    if (block_start < params.len) {
        let last_local = min(256u, params.len - block_start) - 1u;
        if (local_id == last_local) {
            block_sums[group_id.x] = values[local_id];
        }
    }
}
