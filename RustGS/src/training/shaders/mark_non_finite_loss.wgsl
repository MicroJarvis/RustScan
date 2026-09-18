struct Params {
    iteration: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const STATUS_NON_FINITE_LOSS: u32 = 2u;
const STATUS_FIRST_ITERATION_UNSET: u32 = 0xffffffffu;

@group(0) @binding(0) var<storage, read> loss: array<f32>;
@group(0) @binding(1) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> params: Params;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }
    let value = loss[0];
    // IEEE754: all-ones exponent marks NaN / Inf.
    let bits = bitcast<u32>(value);
    let exponent = (bits >> 23u) & 0xffu;
    let non_finite = exponent == 0xffu;
    if (non_finite) {
        atomicOr(&status[0], STATUS_NON_FINITE_LOSS);
        atomicMin(&status[1], max(params.iteration, 1u));
    }
}
