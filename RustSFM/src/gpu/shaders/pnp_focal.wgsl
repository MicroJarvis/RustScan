struct PnpFocalModel {
    row0: vec4<f32>, row1: vec4<f32>, row2: vec4<f32>,
    log_focal: f32, pad0: f32, pad1: f32, pad2: f32,
}

struct PnpFocalResult {
    selected_model: u32, inliers: u32, valid: u32, pad0: u32,
    residual_sum: f32, focal: f32, pad1: f32, pad2: f32,
}

struct PnpFocalSample {
    indices: vec4<u32>,
}

struct PnpFocalSamplingParams {
    seed: u32,
    trial_count: u32,
    observation_count: u32,
    pad: u32,
}

@group(0) @binding(0)
var<storage, read_write> samples: array<PnpFocalSample>;

@group(0) @binding(1)
var<uniform> params: PnpFocalSamplingParams;

fn sample_hash(seed: u32, trial: u32, lane: u32, attempt: u32) -> u32 {
    var value = seed ^ (trial * 0x9e3779b9u) ^ (lane * 0x85ebca6bu) ^ (attempt * 0xc2b2ae35u);
    value = value ^ (value >> 16u);
    value = value * 0x7feb352du;
    value = value ^ (value >> 15u);
    value = value * 0x846ca68bu;
    return value ^ (value >> 16u);
}

@compute @workgroup_size(1)
fn sample_four_points(@builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let trial = workgroup_id.x;
    if (trial >= params.trial_count) {
        return;
    }

    var selected: array<u32, 4>;
    var lane = 0u;
    loop {
        if (lane == 4u) {
            break;
        }
        var attempt = 0u;
        loop {
            let candidate = sample_hash(params.seed, trial, lane, attempt) % params.observation_count;
            var duplicate = false;
            var previous = 0u;
            loop {
                if (previous == lane) {
                    break;
                }
                if (selected[previous] == candidate) {
                    duplicate = true;
                    break;
                }
                previous = previous + 1u;
            }
            if (!duplicate) {
                selected[lane] = candidate;
                break;
            }
            attempt = attempt + 1u;
        }
        lane = lane + 1u;
    }
    samples[trial].indices = vec4<u32>(selected[0], selected[1], selected[2], selected[3]);
}
