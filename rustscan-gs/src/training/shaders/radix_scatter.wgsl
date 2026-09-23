struct Params {
    shift: u32,
    num_blocks: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<u32>;
@group(0) @binding(2) var<storage, read> count: array<u32>;
@group(0) @binding(3) var<storage, read> hist: array<u32>;
@group(0) @binding(4) var<storage, read> scanned: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_keys: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_values: array<u32>;
@group(0) @binding(7) var<storage, read> params: Params;

var<workgroup> shared_keys: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) local_id: u32,
) {
    let n = count[0];
    let idx = gid.x;
    let block = idx / 256u;
    let block_start = block * 256u;
    let in_range = idx < n;
    // `select` would still evaluate `keys[idx]` on tail lanes past `n`.
    if (in_range) {
        shared_keys[local_id] = keys[idx];
    } else {
        shared_keys[local_id] = 0u;
    }
    workgroupBarrier();

    if !in_range {
        return;
    }

    let digit = (shared_keys[local_id] >> params.shift) & 15u;
    let active_in_block = min(256u, n - block_start);
    var rank = 0u;
    for (var lane = 0u; lane < local_id; lane++) {
        if lane < active_in_block && ((shared_keys[lane] >> params.shift) & 15u) == digit {
            rank += 1u;
        }
    }

    let hist_index = digit * params.num_blocks + block;
    // Inclusive scan minus this block's count is the exclusive global base.
    let dest = (scanned[hist_index] - hist[hist_index]) + rank;
    out_keys[dest] = shared_keys[local_id];
    out_values[dest] = values[idx];
}
