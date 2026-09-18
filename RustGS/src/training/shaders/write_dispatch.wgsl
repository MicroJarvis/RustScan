struct Params {
    intersection_capacity: u32,
    workgroup_size: u32,
    iteration: u32,
    _pad0: u32,
}

const STATUS_FORWARD_OVERFLOW: u32 = 1u;
const STATUS_FIRST_ITERATION_UNSET: u32 = 0xffffffffu;

@group(0) @binding(0) var<storage, read_write> num_visible: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> num_intersections: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> logical_visible: array<u32>;
@group(0) @binding(3) var<storage, read_write> logical_intersections: array<u32>;
@group(0) @binding(4) var<storage, read_write> visible_dispatch: array<u32>;
@group(0) @binding(5) var<storage, read_write> intersection_dispatch: array<u32>;
@group(0) @binding(6) var<storage, read_write> overflow: array<u32>;
@group(0) @binding(7) var<storage, read_write> requested_intersections: array<u32>;
@group(0) @binding(8) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read> params: Params;

@compute @workgroup_size(1, 1, 1)
fn main() {
    let visible = atomicLoad(&num_visible[0]);
    let intersections = atomicLoad(&num_intersections[0]);
    let clamped_intersections = min(intersections, params.intersection_capacity);
    let workgroup = max(params.workgroup_size, 1u);
    let overflowed = intersections > params.intersection_capacity;

    logical_visible[0] = visible;
    // Kernels must see the workspace-safe count; telemetry keeps the true request.
    logical_intersections[0] = clamped_intersections;
    requested_intersections[0] = intersections;
    visible_dispatch[0] = (visible + workgroup - 1u) / workgroup;
    visible_dispatch[1] = 1u;
    visible_dispatch[2] = 1u;
    intersection_dispatch[0] = (clamped_intersections + workgroup - 1u) / workgroup;
    intersection_dispatch[1] = 1u;
    intersection_dispatch[2] = 1u;
    overflow[0] = select(0u, 1u, overflowed);

    if (overflowed) {
        atomicOr(&status[0], STATUS_FORWARD_OVERFLOW);
        let previous = atomicMin(&status[1], max(params.iteration, 1u));
        if (previous == STATUS_FIRST_ITERATION_UNSET) {
            // First sticky anomaly owns requested/capacity telemetry.
            atomicStore(&status[2], intersections);
            atomicStore(&status[3], params.intersection_capacity);
        }
    }
}
