struct Params {
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

const STATUS_FORWARD_OVERFLOW: u32 = 1u;
const STATUS_NON_FINITE_LOSS: u32 = 2u;

@group(0) @binding(0) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> params: Params;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }
    _ = params._pad0;
    let flags = atomicLoad(&status[0]);
    let healthy = (flags & (STATUS_FORWARD_OVERFLOW | STATUS_NON_FINITE_LOSS)) == 0u;
    if (healthy) {
        atomicStore(&status[5], 1u);
        atomicAdd(&status[4], 1u);
    } else {
        atomicStore(&status[5], 0u);
        // Count device-side optimizer gate skips for safety-point telemetry.
        atomicAdd(&status[6], 1u);
    }
}
