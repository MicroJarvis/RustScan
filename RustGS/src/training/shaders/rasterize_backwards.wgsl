#import helpers;

@group(0) @binding(0) var<storage, read> compact_gid_from_isect: array<u32>;
@group(0) @binding(1) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> projected: array<helpers::ProjectedSplat>;
@group(0) @binding(3) var<storage, read> output: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> v_output: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> v_splats: array<atomic<f32>>;
@group(0) @binding(6) var<storage, read_write> screen_grad_splats: array<atomic<f32>>;
@group(0) @binding(7) var<storage, read> uniforms: helpers::RasterizeUniforms;

var<workgroup> range_uniform: vec2<u32>;
var<workgroup> local_batch: array<helpers::ProjectedSplat, helpers::TILE_SIZE>;
var<workgroup> local_gid: array<u32, helpers::TILE_SIZE>;

fn write_grads_atomic(id: u32, grads: f32) {
    atomicAdd(&v_splats[id], grads);
}

fn write_screen_grad_atomic(id: u32, grads: f32) {
    atomicAdd(&screen_grad_splats[id], grads);
}

@compute
@workgroup_size(helpers::TILE_SIZE, 1, 1)
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
    let pixel_id = pixel.x + pixel.y * uniforms.img_size.x;

    var v_out = vec4<f32>(0.0);
    var rgb_pixel_final = vec4<f32>(0.0);
    if inside {
        let final_color = output[pixel_id];
        let T_final = 1.0f - final_color.a;
        rgb_pixel_final = vec4<f32>(
            final_color.rgb - T_final * uniforms.background.rgb,
            final_color.a,
        );
        let grad_out = v_output[pixel_id];
        v_out = vec4<f32>(
            grad_out.rgb,
            (grad_out.a - dot(uniforms.background.rgb, grad_out.rgb)) * T_final,
        );
    }

    var pix_out = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    var done = !inside;

    if local_idx == 0u {
        range_uniform = vec2<u32>(
            tile_offsets[tile_id * 2u],
            tile_offsets[tile_id * 2u + 1u],
        );
    }
    workgroupBarrier();

    let range = range_uniform;

    for (var batch_start = range.x; batch_start < range.y; batch_start += helpers::TILE_SIZE) {
        let remaining = min(helpers::TILE_SIZE, range.y - batch_start);

        if local_idx < remaining {
            let load_isect_id = batch_start + local_idx;
            let compact_gid = compact_gid_from_isect[load_isect_id];
            local_batch[local_idx] = projected[compact_gid];
            local_gid[local_idx] = compact_gid;
        }
        workgroupBarrier();

        for (var t = 0u; t < remaining; t++) {
            if done {
                continue;
            }

            let proj = local_batch[t];
            let xy = vec2<f32>(proj.xy_x, proj.xy_y);
            let conic = vec3<f32>(proj.conic_x, proj.conic_y, proj.conic_z);
            let color = vec4<f32>(proj.color_r, proj.color_g, proj.color_b, proj.color_a);
            let clamped_rgb = max(color.rgb, vec3<f32>(0.0));
            let delta = xy - pixel_coord;
            let sigma = 0.5f * (
                conic.x * delta.x * delta.x +
                conic.z * delta.y * delta.y
            ) + conic.y * delta.x * delta.y;
            let gaussian = exp(-sigma);
            let alpha = min(0.999f, color.a * gaussian);

            if sigma < 0.0f || alpha < 1.0f / 255.0f {
                continue;
            }

            let next_T = pix_out.a * (1.0f - alpha);
            let stop_after_current = next_T < 1e-4f;

            let vis = alpha * pix_out.a;
            let v_rgb = select(vec3<f32>(0.0), vis * v_out.rgb, color.rgb >= vec3<f32>(0.0));
            pix_out = vec4<f32>(pix_out.rgb + vis * clamped_rgb, pix_out.a);

            let ra = 1.0f / (1.0f - alpha);
            let v_alpha = dot(
                pix_out.a * clamped_rgb + (pix_out.rgb - rgb_pixel_final.rgb) * ra,
                v_out.rgb,
            ) + v_out.a * ra;
            let v_sigma = -alpha * v_alpha;
            let v_xy_local = v_sigma * vec2<f32>(
                conic.x * delta.x + conic.y * delta.y,
                conic.y * delta.x + conic.z * delta.y,
            );

            var v_conic = vec3<f32>(0.0);
            var v_xy = vec2<f32>(0.0);
            var v_alpha_term = 0.0f;
            var v_refine = 0.0f;
            if color.a * gaussian <= 0.999f {
                v_conic = vec3<f32>(
                    0.5f * v_sigma * delta.x * delta.x,
                    v_sigma * delta.x * delta.y,
                    0.5f * v_sigma * delta.y * delta.y,
                );
                v_xy = v_xy_local;
                v_alpha_term = v_alpha * gaussian;
                let final_a = max(rgb_pixel_final.a, 1e-5f);
                v_refine = length(v_xy_local * vec2<f32>(uniforms.img_size.xy)) / final_a;
            }

            let base = local_gid[t] * 10u;
            write_grads_atomic(base + 0u, v_xy.x);
            write_grads_atomic(base + 1u, v_xy.y);
            write_grads_atomic(base + 2u, v_conic.x);
            write_grads_atomic(base + 3u, v_conic.y);
            write_grads_atomic(base + 4u, v_conic.z);
            write_grads_atomic(base + 5u, v_rgb.x);
            write_grads_atomic(base + 6u, v_rgb.y);
            write_grads_atomic(base + 7u, v_rgb.z);
            write_grads_atomic(base + 8u, v_alpha_term);
            write_grads_atomic(base + 9u, v_refine);

            let screen_base = local_gid[t] * 5u;
            write_screen_grad_atomic(screen_base + 0u, v_xy.x);
            write_screen_grad_atomic(screen_base + 1u, v_xy.y);
            write_screen_grad_atomic(screen_base + 2u, abs(v_xy.x));
            write_screen_grad_atomic(screen_base + 3u, abs(v_xy.y));
            write_screen_grad_atomic(screen_base + 4u, 1.0);

            pix_out.a = next_T;
            if stop_after_current {
                done = true;
            }
        }

        workgroupBarrier();
    }
}
