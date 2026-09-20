#import helpers;

struct MGIUniforms {
    tile_bounds: vec2<u32>,
    num_visible: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> projected: array<helpers::ProjectedSplat>;
@group(0) @binding(1) var<storage, read> cum_tiles_hit: array<u32>;
@group(0) @binding(2) var<storage, read_write> tile_id_from_isect: array<u32>;
@group(0) @binding(3) var<storage, read_write> compact_gid_from_isect: array<u32>;
@group(0) @binding(4) var<storage, read> uniforms: MGIUniforms;
@group(0) @binding(5) var<storage, read> logical_visible: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let compact_gid = gid.x;
    if compact_gid >= logical_visible[0] {
        return;
    }

    let splat = projected[compact_gid];
    let mean2d = vec2<f32>(splat.xy_x, splat.xy_y);
    let conic = vec3<f32>(splat.conic_x, splat.conic_y, splat.conic_z);
    let opacity = splat.color_a;
    let power_threshold = log(opacity * 255.0);
    let cov2d = helpers::inverse2x2(mat2x2<f32>(conic.x, conic.y, conic.y, conic.z));
    let extent = helpers::compute_bbox_extent(cov2d, power_threshold);
    let tile_bbox = helpers::get_tile_bbox(mean2d, extent, uniforms.tile_bounds);
    let bbox_min = tile_bbox.xy;
    let bbox_max = tile_bbox.zw;
    let bbox_width = bbox_max.x - bbox_min.x;
    let bbox_tiles = (bbox_max.y - bbox_min.y) * bbox_width;
    let base_isect_id = select(cum_tiles_hit[compact_gid - 1u], 0u, compact_gid == 0u);

    var offset = 0u;
    for (var tile_idx = 0u; tile_idx < bbox_tiles; tile_idx++) {
        let tx = bbox_min.x + (tile_idx % bbox_width);
        let ty = bbox_min.y + (tile_idx / bbox_width);
        let rect = vec4<u32>(tx, ty, tx + 1u, ty + 1u);
        if helpers::will_primitive_contribute(rect, mean2d, conic, power_threshold) {
            let isect_id = base_isect_id + offset;
            let tile_id = tx + ty * uniforms.tile_bounds.x;
            tile_id_from_isect[isect_id] = tile_id;
            compact_gid_from_isect[isect_id] = compact_gid;
            offset += 1u;
        }
    }
}
