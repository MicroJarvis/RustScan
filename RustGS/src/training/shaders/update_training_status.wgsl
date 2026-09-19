// Shared device training status buffer (exactly 8 contiguous u32 words).
// Layout must match `device_status.rs` / WGSL atomic producers.
//
// word 0: sticky flags (atomicOr)
// word 1: first_invalid_iteration (atomicMin; init 0xffffffff)
// word 2: first overflow requested_intersections
// word 3: first overflow intersection_capacity
// word 4: committed_optimizer_steps
// word 5: mutation_gate for the current step (0 = blocked, 1 = allowed)
// word 6: gpu_gate_optimizer_skips (atomicAdd when prepare blocks)
// word 7: reserved

const STATUS_FORWARD_OVERFLOW: u32 = 1u;
const STATUS_NON_FINITE_LOSS: u32 = 2u;
const STATUS_FIRST_ITERATION_UNSET: u32 = 0xffffffffu;

struct StatusParams {
    iteration: u32,
    requested_intersections: u32,
    intersection_capacity: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<storage, read_write> status: array<u32>;
@group(0) @binding(1) var<storage, read> params: StatusParams;

// Placeholder entry point for Task 1.1; overflow / non-finite kernels land in 1.2–1.3.
@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }
    let _ = params.iteration;
    let _ = status[0];
}
