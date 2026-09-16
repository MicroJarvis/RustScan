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

const DESCRIPTOR_PACKED_WORDS: u32 = 32u;
const DESCRIPTOR_STRIDE: u32 = 33u;
const WORKGROUP_SIZE: u32 = 64u;

var<workgroup> target_tile: array<u32, 2112>;

@compute @workgroup_size(64)
fn match_main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let query_index = id.x;
    let query_valid = query_index < params.query_count;

    var query_desc: array<u32, 33>;
    if (query_valid) {
        let query_base = query_index * DESCRIPTOR_STRIDE;
        for (var word_index = 0u; word_index < DESCRIPTOR_STRIDE; word_index = word_index + 1u) {
            query_desc[word_index] = queries[query_base + word_index];
        }
    }

    var best_index = 0xffffffffu;
    var second_index = 0xffffffffu;
    var best_distance = 3.402823466e+38;
    var second_distance = 3.402823466e+38;

    // Tile targets into workgroup memory. Target index order stays 0..N-1, so
    // equal-distance tie-breaks match the previous per-thread global scan.
    var tile_start = 0u;
    loop {
        if (tile_start >= params.target_count) {
            break;
        }
        let tile_count = min(WORKGROUP_SIZE, params.target_count - tile_start);
        let tile_word_count = tile_count * DESCRIPTOR_STRIDE;
        let src_base = tile_start * DESCRIPTOR_STRIDE;
        for (var word = lid; word < tile_word_count; word = word + WORKGROUP_SIZE) {
            target_tile[word] = targets[src_base + word];
        }
        workgroupBarrier();

        if (query_valid) {
            for (var t = 0u; t < tile_count; t = t + 1u) {
                let target_index = tile_start + t;
                let tile_base = t * DESCRIPTOR_STRIDE;
                var cross_term = 0u;
                for (var word_index = 0u; word_index < DESCRIPTOR_PACKED_WORDS; word_index = word_index + 4u) {
                    cross_term = cross_term + dot4U8Packed(
                        query_desc[word_index],
                        target_tile[tile_base + word_index],
                    );
                    cross_term = cross_term + dot4U8Packed(
                        query_desc[word_index + 1u],
                        target_tile[tile_base + word_index + 1u],
                    );
                    cross_term = cross_term + dot4U8Packed(
                        query_desc[word_index + 2u],
                        target_tile[tile_base + word_index + 2u],
                    );
                    cross_term = cross_term + dot4U8Packed(
                        query_desc[word_index + 3u],
                        target_tile[tile_base + word_index + 3u],
                    );
                }
                let distance_u32 = query_desc[DESCRIPTOR_PACKED_WORDS]
                    + target_tile[tile_base + DESCRIPTOR_PACKED_WORDS]
                    - 2u * cross_term;
                // 128 * 255^2 = 8,323,200 < 2^24, so conversion to f32 is exact.
                let distance = f32(distance_u32);
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
        }
        workgroupBarrier();
        tile_start = tile_start + WORKGROUP_SIZE;
    }

    if (!query_valid) {
        return;
    }
    let ratio_ok = best_distance < params.ratio_squared * second_distance;
    let valid = select(0u, 1u,
        best_index != 0xffffffffu
            && best_distance <= params.max_l2_distance
            && ratio_ok);
    candidates[query_index] = MatchCandidate(
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
