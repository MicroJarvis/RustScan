#import helpers;

@group(0) @binding(0) var<storage, read> params: array<f32>;
@group(0) @binding(1) var<storage, read> global_from_compact_gid: array<u32>;
@group(0) @binding(2) var<storage, read> v_splats: array<f32>;
@group(0) @binding(3) var<storage, read_write> v_params: array<f32>;
@group(0) @binding(4) var<storage, read_write> v_sh_coeffs: array<f32>;
@group(0) @binding(5) var<storage, read> uniforms: helpers::ProjectUniforms;
@group(0) @binding(6) var<storage, read> screen_grad_splats: array<f32>;
@group(0) @binding(7) var<storage, read_write> screen_grad_stats: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let compact_gid = gid.x;
    if compact_gid >= uniforms.num_visible {
        return;
    }

    let global_gid = global_from_compact_gid[compact_gid];
    let rg_base = compact_gid * 10u;
    let screen_compact_base = compact_gid * 5u;
    let screen_global_base = global_gid * 7u;
    let signed_screen_grad = vec2<f32>(
        screen_grad_splats[screen_compact_base + 0u],
        screen_grad_splats[screen_compact_base + 1u],
    );
    let abs_screen_grad = vec2<f32>(
        screen_grad_splats[screen_compact_base + 2u],
        screen_grad_splats[screen_compact_base + 3u],
    );
    let pixel_coverage = screen_grad_splats[screen_compact_base + 4u];

    screen_grad_stats[screen_global_base + 0u] = signed_screen_grad.x;
    screen_grad_stats[screen_global_base + 1u] = signed_screen_grad.y;
    screen_grad_stats[screen_global_base + 2u] = abs_screen_grad.x;
    screen_grad_stats[screen_global_base + 3u] = abs_screen_grad.y;

    let v_xy = vec2<f32>(v_splats[rg_base], v_splats[rg_base + 1u]);
    let v_conic = vec3<f32>(
        v_splats[rg_base + 2u],
        v_splats[rg_base + 3u],
        v_splats[rg_base + 4u],
    );
    let v_color = vec3<f32>(
        v_splats[rg_base + 5u],
        v_splats[rg_base + 6u],
        v_splats[rg_base + 7u],
    );
    let v_color_a = v_splats[rg_base + 8u];

    let t_base = global_gid * 11u;
    let p_base = global_gid * 11u;
    let mean = vec3<f32>(params[t_base], params[t_base + 1u], params[t_base + 2u]);
    let quat_unorm = vec4<f32>(
        params[t_base + 3u],
        params[t_base + 4u],
        params[t_base + 5u],
        params[t_base + 6u],
    );
    let quat = normalize(quat_unorm);
    let log_scale = vec3<f32>(
        params[t_base + 7u],
        params[t_base + 8u],
        params[t_base + 9u],
    );
    let scale = exp(log_scale);

    let viewdir_delta = mean - uniforms.camera_position.xyz;
    let viewdir = viewdir_delta * inverseSqrt(max(dot(viewdir_delta, viewdir_delta), 1e-12));
    let sh_grads = helpers::sh_to_color_vjp(uniforms.sh_degree, viewdir, v_color);
    let num_active_coeffs = helpers::num_sh_coeffs(uniforms.sh_degree);
    let num_storage_coeffs = helpers::num_sh_coeffs(uniforms.storage_sh_degree);
    let sh_base = global_gid * num_storage_coeffs * 3u;
    for (var coeff_idx = 0u; coeff_idx < num_active_coeffs; coeff_idx++) {
        let grad = sh_grads[coeff_idx];
        let dst = sh_base + coeff_idx * 3u;
        v_sh_coeffs[dst] = grad.x;
        v_sh_coeffs[dst + 1u] = grad.y;
        v_sh_coeffs[dst + 2u] = grad.z;
    }

    let rotation = mat3x3<f32>(
        uniforms.viewmat[0].xyz,
        uniforms.viewmat[1].xyz,
        uniforms.viewmat[2].xyz,
    );
    let mean_c = rotation * mean + uniforms.viewmat[3].xyz;
    if mean_c.z < 0.01 {
        return;
    }
    let rotmat = helpers::quat_to_mat3(quat);
    let scale_mat = helpers::scale_to_mat3(scale);
    let m = rotmat * scale_mat;
    let cov3d = m * transpose(m);
    let cov_cam = rotation * cov3d * transpose(rotation);
    let j = helpers::calc_cam_j(
        mean_c,
        uniforms.focal,
        uniforms.img_size,
        uniforms.pixel_center,
    );
    var cov2d = j * cov_cam * transpose(j);
    let filter_comp = helpers::compensate_cov2d(&cov2d, uniforms.cov_blur);

    let opac = helpers::sigmoid(params[t_base + 10u]);
    let abs_score = length(abs_screen_grad);
    screen_grad_stats[screen_global_base + 4u] = pixel_coverage * abs_score;
    screen_grad_stats[screen_global_base + 5u] = pixel_coverage;
    screen_grad_stats[screen_global_base + 6u] = pixel_coverage * mean_c.z;

    v_params[p_base + 10u] = filter_comp * v_color_a * opac * (1.0 - opac);

    let cov2d_inv = helpers::inverse2x2(cov2d);
    let v_cov2d_inv = mat2x2<f32>(
        vec2<f32>(v_conic.x, 0.5 * v_conic.y),
        vec2<f32>(0.5 * v_conic.y, v_conic.z),
    );
    let v_cov2d = helpers::inverse2x2_vjp(cov2d_inv, v_cov2d_inv);
    let v_mean_c = helpers::persp_proj_vjp(
        j,
        mean_c,
        cov_cam,
        uniforms.focal,
        uniforms.pixel_center,
        uniforms.img_size,
        v_cov2d,
        v_xy,
    );
    let v_cov_cam = transpose(j) * v_cov2d * j;
    let v_cov3d = transpose(rotation) * v_cov_cam * rotation;
    let v_m = (v_cov3d + transpose(v_cov3d)) * m;
    let v_scale = vec3<f32>(
        dot(rotmat[0], v_m[0]),
        dot(rotmat[1], v_m[1]),
        dot(rotmat[2], v_m[2]),
    );
    let v_log_scale = v_scale * scale;
    let v_quat = helpers::normalize_vjp(quat_unorm) *
        helpers::quat_to_mat3_vjp(quat, v_m * scale_mat);
    let v_mean = transpose(rotation) * v_mean_c;

    v_params[p_base] = v_mean.x;
    v_params[p_base + 1u] = v_mean.y;
    v_params[p_base + 2u] = v_mean.z;
    v_params[p_base + 3u] = v_quat.x;
    v_params[p_base + 4u] = v_quat.y;
    v_params[p_base + 5u] = v_quat.z;
    v_params[p_base + 6u] = v_quat.w;
    v_params[p_base + 7u] = v_log_scale.x;
    v_params[p_base + 8u] = v_log_scale.y;
    v_params[p_base + 9u] = v_log_scale.z;
}
