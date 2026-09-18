struct Params {
    shift: u32,
    num_blocks: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> count: array<u32>;
@group(0) @binding(2) var<storage, read_write> hist: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= count[0] {
        return;
    }

    let digit = (keys[idx] >> params.shift) & 15u;
    let block = idx / 256u;
    atomicAdd(&hist[digit * params.num_blocks + block], 1u);
}
