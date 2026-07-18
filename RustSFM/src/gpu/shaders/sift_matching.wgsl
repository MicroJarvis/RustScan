struct MatchParams {
    query_count: u32,
    target_count: u32,
    max_l2_distance: f32,
    ratio_squared: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

struct MatchCandidate {
    best_index: u32,
    second_index: u32,
    best_distance: f32,
    second_distance: f32,
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<storage, read> queries: array<u32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read_write> candidates: array<MatchCandidate>;
@group(0) @binding(3) var<uniform> params: MatchParams;

const DESCRIPTOR_WORDS: u32 = 32u;

@compute @workgroup_size(64)
fn match_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.query_count) {
        return;
    }

    var best_index = 0xffffffffu;
    var second_index = 0xffffffffu;
    var best_distance = 3.402823466e+38;
    var second_distance = 3.402823466e+38;
    let query_base = id.x * DESCRIPTOR_WORDS;
    for (var target_index = 0u; target_index < params.target_count; target_index = target_index + 1u) {
        let target_base = target_index * DESCRIPTOR_WORDS;
        var distance = 0.0;
        for (var word_index = 0u; word_index < DESCRIPTOR_WORDS; word_index = word_index + 1u) {
            let query_word = queries[query_base + word_index];
            let target_word = targets[target_base + word_index];
            for (var byte_index = 0u; byte_index < 4u; byte_index = byte_index + 1u) {
                let shift = byte_index * 8u;
                let query_value = (query_word >> shift) & 0xffu;
                let target_value = (target_word >> shift) & 0xffu;
                let delta = f32(query_value) - f32(target_value);
                distance = distance + delta * delta;
            }
        }
        if (distance < best_distance || (distance == best_distance && target_index < best_index)) {
            second_distance = best_distance;
            second_index = best_index;
            best_distance = distance;
            best_index = target_index;
        } else if (distance < second_distance
            || (distance == second_distance && target_index < second_index)) {
            second_distance = distance;
            second_index = target_index;
        }
    }

    let ratio_ok = best_distance < params.ratio_squared * second_distance;
    let valid = select(0u, 1u,
        best_index != 0xffffffffu
            && best_distance <= params.max_l2_distance
            && ratio_ok);
    candidates[id.x] = MatchCandidate(
        best_index,
        second_index,
        best_distance,
        second_distance,
        valid,
        0u,
        0u,
        0u,
    );
}
