#import helpers;

struct RasterizeUniforms {
    tile_bounds: vec2<u32>,
    img_size: vec2<u32>,
    background: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> compact_gid_from_isect: array<u32>;
@group(0) @binding(1) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> projected: array<helpers::ProjectedSplat>;
@group(0) @binding(3) var<storage, read_write> out_img: array<f32>;
@group(0) @binding(4) var<storage, read> global_from_compact_gid: array<u32>;
@group(0) @binding(5) var<storage, read_write> visible: array<f32>;
@group(0) @binding(6) var<storage, read_write> depth: array<f32>;
@group(0) @binding(7) var<storage, read> uniforms: RasterizeUniforms;

var<workgroup> range_uniform: vec2<u32>;
var<workgroup> local_batch: array<helpers::ProjectedSplat, helpers::TILE_SIZE>;
var<workgroup> load_gid: array<u32, helpers::TILE_SIZE>;

@compute @workgroup_size(helpers::TILE_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    let num_tiles = uniforms.tile_bounds.x * uniforms.tile_bounds.y;
    let tile_id = wg_id.x;
    if tile_id >= num_tiles {
        return;
    }

    let tile_x = tile_id % uniforms.tile_bounds.x;
    let tile_y = tile_id / uniforms.tile_bounds.x;
    let pixel = vec2<u32>(
        tile_x * helpers::TILE_WIDTH + (local_idx % helpers::TILE_WIDTH),
        tile_y * helpers::TILE_WIDTH + (local_idx / helpers::TILE_WIDTH),
    );
    let inside = pixel.x < uniforms.img_size.x && pixel.y < uniforms.img_size.y;
    let pixel_coord = vec2<f32>(pixel) + vec2<f32>(0.5);

    if local_idx == 0u {
        range_uniform = vec2<u32>(
            tile_offsets[tile_id * 2u],
            tile_offsets[tile_id * 2u + 1u],
        );
    }
    workgroupBarrier();

    let range = range_uniform;
    var T = 1.0;
    var pix_out = vec3<f32>(0.0);
    var depth_sum = 0.0;
    var done = !inside;

    for (var batch_start = range.x; batch_start < range.y; batch_start += helpers::TILE_SIZE) {
        let remaining = min(helpers::TILE_SIZE, range.y - batch_start);

        if local_idx < remaining {
            let isect_id = batch_start + local_idx;
            let compact_gid = compact_gid_from_isect[isect_id];
            local_batch[local_idx] = projected[compact_gid];
            load_gid[local_idx] = global_from_compact_gid[compact_gid];
        }
        workgroupBarrier();

        for (var t = 0u; !done && t < remaining; t++) {
            let splat = local_batch[t];
            let delta = vec2<f32>(splat.xy_x, splat.xy_y) - pixel_coord;
            let sigma = 0.5 * (
                splat.conic_x * delta.x * delta.x +
                2.0 * splat.conic_y * delta.x * delta.y +
                splat.conic_z * delta.y * delta.y
            );
            let alpha = min(0.999, splat.color_a * exp(-sigma));

            if sigma >= 0.0 && alpha >= (1.0 / 255.0) {
                let vis = alpha * T;
                pix_out += max(vec3<f32>(splat.color_r, splat.color_g, splat.color_b), vec3<f32>(0.0)) * vis;
                depth_sum += splat.depth * vis;
                T *= (1.0 - alpha);
                visible[load_gid[t]] = 1.0;

                if T < 1e-4 {
                    done = true;
                    break;
                }
            }
        }
        workgroupBarrier();
    }

    if inside {
        let pixel_id = pixel.x + pixel.y * uniforms.img_size.x;
        let base = pixel_id * 4u;
        let final_rgb = pix_out + T * uniforms.background.rgb;
        out_img[base] = final_rgb.r;
        out_img[base + 1u] = final_rgb.g;
        out_img[base + 2u] = final_rgb.b;
        out_img[base + 3u] = 1.0 - T;
        let alpha = 1.0 - T;
        depth[pixel_id] = select(0.0, depth_sum / max(alpha, 1e-6), alpha > (1.0 / 255.0));
    }
}
