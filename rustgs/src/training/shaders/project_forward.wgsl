#import helpers;

@group(0) @binding(0) var<storage, read> transforms: array<f32>;
@group(0) @binding(1) var<storage, read> raw_opacities: array<f32>;
@group(0) @binding(2) var<storage, read_write> global_from_presort_gid: array<u32>;
@group(0) @binding(3) var<storage, read_write> depths: array<f32>;
@group(0) @binding(4) var<storage, read_write> num_visible: atomic<u32>;
@group(0) @binding(5) var<storage, read_write> intersect_counts: array<u32>;
@group(0) @binding(6) var<storage, read_write> num_intersections: atomic<u32>;
@group(0) @binding(7) var<storage, read> uniforms: helpers::ProjectUniforms;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let global_gid = gid.x;
    if global_gid >= uniforms.total_splats {
        return;
    }

    let base = global_gid * 10u;
    let mean = vec3<f32>(transforms[base], transforms[base + 1u], transforms[base + 2u]);
    var quat = vec4<f32>(
        transforms[base + 3u],
        transforms[base + 4u],
        transforms[base + 5u],
        transforms[base + 6u],
    );
    let scale = exp(vec3<f32>(
        transforms[base + 7u],
        transforms[base + 8u],
        transforms[base + 9u],
    ));

    let mean_c = (uniforms.viewmat * vec4<f32>(mean, 1.0)).xyz;
    if mean_c.z < 0.01 || mean_c.z > 1e10 {
        return;
    }

    let quat_norm_sqr = dot(quat, quat);
    if quat_norm_sqr < 1e-6 {
        return;
    }
    quat *= inverseSqrt(quat_norm_sqr);

    var opacity = helpers::sigmoid(raw_opacities[global_gid]);
    let cov3d = helpers::calc_cov3d(scale, quat);
    var cov2d = helpers::calc_cov2d(
        cov3d,
        mean_c,
        uniforms.focal,
        uniforms.img_size,
        uniforms.pixel_center,
        uniforms.viewmat,
    );
    opacity *= helpers::compensate_cov2d(&cov2d, uniforms.cov_blur);
    if opacity < (1.0 / 255.0) {
        return;
    }

    let mean2d = uniforms.focal * mean_c.xy * (1.0 / mean_c.z) + uniforms.pixel_center;
    let conic = helpers::inverse2x2(cov2d);
    let power_threshold = log(opacity * 255.0);
    let extent = helpers::compute_bbox_extent(cov2d, power_threshold);
    if extent.x < 0.0 || extent.y < 0.0 {
        return;
    }

    if mean2d.x + extent.x <= 0.0 ||
        mean2d.x - extent.x >= f32(uniforms.img_size.x) ||
        mean2d.y + extent.y <= 0.0 ||
        mean2d.y - extent.y >= f32(uniforms.img_size.y) {
        return;
    }

    let conic_packed = vec3<f32>(conic[0][0], conic[0][1], conic[1][1]);
    let tile_bbox = helpers::get_tile_bbox(mean2d, extent, uniforms.tile_bounds);
    let bbox_min = tile_bbox.xy;
    let bbox_max = tile_bbox.zw;
    let bbox_width = bbox_max.x - bbox_min.x;
    let num_tiles_bbox = (bbox_max.y - bbox_min.y) * bbox_width;

    var num_tiles_hit = 0u;
    for (var tile_idx = 0u; tile_idx < num_tiles_bbox; tile_idx++) {
        let tx = bbox_min.x + (tile_idx % bbox_width);
        let ty = bbox_min.y + (tile_idx / bbox_width);
        let rect = vec4<u32>(tx, ty, tx + 1u, ty + 1u);
        if helpers::will_primitive_contribute(rect, mean2d, conic_packed, power_threshold) {
            num_tiles_hit += 1u;
        }
    }

    intersect_counts[global_gid] = num_tiles_hit;
    atomicAdd(&num_intersections, num_tiles_hit);

    let write_id = atomicAdd(&num_visible, 1u);
    global_from_presort_gid[write_id] = global_gid;
    depths[write_id] = mean_c.z;
}
