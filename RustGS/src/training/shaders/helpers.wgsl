#define_import_path helpers

const TILE_WIDTH: u32 = 16u;
const TILE_SIZE: u32 = 256u;
const SH_C0: f32 = 0.28209479177387814;

struct ProjectUniforms {
    viewmat: mat4x4<f32>,
    focal: vec2<f32>,
    img_size: vec2<u32>,
    tile_bounds: vec2<u32>,
    pixel_center: vec2<f32>,
    camera_position: vec4<f32>,
    sh_degree: u32,
    total_splats: u32,
    num_visible: u32,
    cov_blur: f32,
}

struct RasterizeUniforms {
    tile_bounds: vec2<u32>,
    img_size: vec2<u32>,
    background: vec4<f32>,
}

struct ProjectedSplat {
    xy_x: f32,
    xy_y: f32,
    conic_x: f32,
    conic_y: f32,
    conic_z: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    color_a: f32,
    depth: f32,
}

fn create_projected_splat(xy: vec2<f32>, conic: vec3<f32>, color: vec4<f32>, depth: f32) -> ProjectedSplat {
    return ProjectedSplat(
        xy.x,
        xy.y,
        conic.x,
        conic.y,
        conic.z,
        color.r,
        color.g,
        color.b,
        color.a,
        depth,
    );
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let w = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;

    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;

    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y2 + z2), 2.0 * (xy + wz), 2.0 * (xz - wy)),
        vec3<f32>(2.0 * (xy - wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz + wx)),
        vec3<f32>(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (x2 + y2)),
    );
}

fn quat_to_mat3_vjp(quat: vec4<f32>, v_r: mat3x3<f32>) -> vec4<f32> {
    let w = quat.x;
    let x = quat.y;
    let y = quat.z;
    let z = quat.w;

    return vec4<f32>(
        x * (v_r[1][2] - v_r[2][1]) +
            y * (v_r[2][0] - v_r[0][2]) +
            z * (v_r[0][1] - v_r[1][0]),
        -2.0 * x * (v_r[1][1] + v_r[2][2]) +
            y * (v_r[0][1] + v_r[1][0]) +
            z * (v_r[0][2] + v_r[2][0]) +
            w * (v_r[1][2] - v_r[2][1]),
        x * (v_r[0][1] + v_r[1][0]) -
            2.0 * y * (v_r[0][0] + v_r[2][2]) +
            z * (v_r[1][2] + v_r[2][1]) +
            w * (v_r[2][0] - v_r[0][2]),
        x * (v_r[0][2] + v_r[2][0]) +
            y * (v_r[1][2] + v_r[2][1]) -
            2.0 * z * (v_r[0][0] + v_r[1][1]) +
            w * (v_r[0][1] - v_r[1][0]),
    ) * 2.0;
}

fn normalize_vjp(v: vec4<f32>) -> mat4x4<f32> {
    let v_sqr = v * v;
    let v_len_sqr = dot(v, v);
    let v_len = sqrt(max(v_len_sqr, 1e-20));

    let cross_complex = -v.xyz * v.yzx;
    let cross_scalar = -v.xyz * v.w;

    return mat4x4<f32>(
        vec4<f32>(v_len_sqr - v_sqr.x, cross_complex.x, cross_complex.z, cross_scalar.x),
        vec4<f32>(cross_complex.x, v_len_sqr - v_sqr.y, cross_complex.y, cross_scalar.y),
        vec4<f32>(cross_complex.z, cross_complex.y, v_len_sqr - v_sqr.z, cross_scalar.z),
        vec4<f32>(cross_scalar.x, cross_scalar.y, cross_scalar.z, v_len_sqr - v_sqr.w),
    ) * (1.0 / (v_len * v_len_sqr));
}

fn scale_to_mat3(s: vec3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(
        vec3<f32>(s.x, 0.0, 0.0),
        vec3<f32>(0.0, s.y, 0.0),
        vec3<f32>(0.0, 0.0, s.z),
    );
}

fn calc_cov3d(scale: vec3<f32>, quat: vec4<f32>) -> mat3x3<f32> {
    let s = mat3x3<f32>(
        vec3<f32>(scale.x, 0.0, 0.0),
        vec3<f32>(0.0, scale.y, 0.0),
        vec3<f32>(0.0, 0.0, scale.z),
    );
    let m = quat_to_mat3(quat) * s;
    return m * transpose(m);
}

fn calc_cam_j(
    mean_c: vec3<f32>,
    focal: vec2<f32>,
    img_size: vec2<u32>,
    pixel_center: vec2<f32>,
) -> mat3x2<f32> {
    let lims_pos = (1.15 * vec2<f32>(img_size) - pixel_center) / focal;
    let lims_neg = (-0.15 * vec2<f32>(img_size) - pixel_center) / focal;
    let rz = 1.0 / max(mean_c.z, 0.01);
    let uv_clipped = clamp(mean_c.xy * rz, lims_neg, lims_pos);
    let duv_dxy = focal * rz;
    return mat3x2<f32>(
        vec2<f32>(duv_dxy.x, 0.0),
        vec2<f32>(0.0, duv_dxy.y),
        -duv_dxy * uv_clipped,
    );
}

fn calc_cov2d(
    cov3d: mat3x3<f32>,
    mean_c: vec3<f32>,
    focal: vec2<f32>,
    img_size: vec2<u32>,
    pixel_center: vec2<f32>,
    viewmat: mat4x4<f32>,
) -> mat2x2<f32> {
    let rotation = mat3x3<f32>(viewmat[0].xyz, viewmat[1].xyz, viewmat[2].xyz);
    let covar_cam = rotation * cov3d * transpose(rotation);
    let j = calc_cam_j(mean_c, focal, img_size, pixel_center);
    return j * covar_cam * transpose(j);
}

fn compensate_cov2d(cov2d: ptr<function, mat2x2<f32>>, cov_blur: f32) -> f32 {
    let cov_start = *cov2d;
    var cov_end = *cov2d;
    cov_end[0][0] += cov_blur;
    cov_end[1][1] += cov_blur;
    *cov2d = cov_end;

    let det_raw = max(determinant(cov_start), 0.0);
    let det_end = max(determinant(cov_end), 1e-12);
    if det_raw <= 0.0 {
        return 0.0;
    }
    return sqrt(det_raw / det_end);
}

fn inverse2x2(m: mat2x2<f32>) -> mat2x2<f32> {
    var m_reg = m;
    m_reg[0][0] += 1e-6;
    m_reg[1][1] += 1e-6;
    let det = determinant(m_reg);
    if det <= 1e-10 {
        return mat2x2<f32>(vec2<f32>(0.0), vec2<f32>(0.0));
    }
    let inv_det = 1.0 / det;
    return mat2x2<f32>(
        vec2<f32>(m_reg[1][1] * inv_det, -m_reg[0][1] * inv_det),
        vec2<f32>(-m_reg[0][1] * inv_det, m_reg[0][0] * inv_det),
    );
}

fn inverse2x2_vjp(m_inv: mat2x2<f32>, v_m_inv: mat2x2<f32>) -> mat2x2<f32> {
    return mat2x2<f32>(-m_inv[0], -m_inv[1]) * v_m_inv * m_inv;
}

fn get_bbox(center: vec2<f32>, extent: vec2<f32>, bounds: vec2<u32>) -> vec4<u32> {
    let min_value = vec2<u32>(clamp(center - extent, vec2<f32>(0.0), vec2<f32>(bounds)));
    let max_value = vec2<u32>(
        clamp(center + extent + vec2<f32>(1.0), vec2<f32>(0.0), vec2<f32>(bounds)),
    );
    return vec4<u32>(min_value, max_value);
}

fn get_tile_bbox(center: vec2<f32>, extent: vec2<f32>, tile_bounds: vec2<u32>) -> vec4<u32> {
    let tile_center = center / f32(TILE_WIDTH);
    let tile_extent = extent / f32(TILE_WIDTH);
    return get_bbox(tile_center, tile_extent, tile_bounds);
}

fn compute_bbox_extent(cov2d: mat2x2<f32>, power_threshold: f32) -> vec2<f32> {
    let safe_power = max(power_threshold, 0.0);
    return vec2<f32>(
        sqrt(max(0.0, 2.0 * safe_power * cov2d[0][0])),
        sqrt(max(0.0, 2.0 * safe_power * cov2d[1][1])),
    );
}

fn tile_rect(tile: vec2<u32>) -> vec4<f32> {
    let rect_min = vec2<f32>(tile * TILE_WIDTH);
    let rect_max = rect_min + f32(TILE_WIDTH);
    return vec4<f32>(rect_min.x, rect_min.y, rect_max.x, rect_max.y);
}

fn calc_sigma(pixel_coord: vec2<f32>, conic: vec3<f32>, mean: vec2<f32>) -> f32 {
    let delta = pixel_coord - mean;
    return 0.5 * (conic.x * delta.x * delta.x + 2.0 * conic.y * delta.x * delta.y + conic.z * delta.y * delta.y);
}

fn will_primitive_contribute(
    rect: vec4<u32>,
    mean2d: vec2<f32>,
    conic: vec3<f32>,
    threshold: f32,
) -> bool {
    let rectf = tile_rect(rect.xy);
    let x_left = mean2d.x < rectf.x;
    let x_right = mean2d.x > rectf.z;
    let in_x_range = !(x_left || x_right);

    let y_above = mean2d.y < rectf.y;
    let y_below = mean2d.y > rectf.w;
    let in_y_range = !(y_above || y_below);

    if in_x_range && in_y_range {
        return true;
    }

    let closest_corner = vec2<f32>(
        select(rectf.z, rectf.x, x_left),
        select(rectf.w, rectf.y, y_above),
    );
    let width = rectf.z - rectf.x;
    let height = rectf.w - rectf.y;
    let d = vec2<f32>(
        select(-width, width, x_left),
        select(-height, height, y_above),
    );

    let diff = mean2d - closest_corner;
    let t_max = vec2<f32>(
        select(
            clamp((d.x * conic.x * diff.x + d.x * conic.y * diff.y) / max(d.x * conic.x * d.x, 1e-12), 0.0, 1.0),
            0.0,
            in_y_range,
        ),
        select(
            clamp((d.y * conic.y * diff.x + d.y * conic.z * diff.y) / max(d.y * conic.z * d.y, 1e-12), 0.0, 1.0),
            0.0,
            in_x_range,
        ),
    );
    let max_point = closest_corner + t_max * d;
    let max_power = calc_sigma(max_point, conic, mean2d);
    return max_power <= threshold;
}

fn num_sh_coeffs(degree: u32) -> u32 {
    return (degree + 1u) * (degree + 1u);
}

fn sh_to_color_vjp(
    degree: u32,
    viewdir: vec3<f32>,
    v_color: vec3<f32>,
) -> array<vec3<f32>, 25> {
    var grads = array<vec3<f32>, 25>();

    grads[0] = SH_C0 * v_color;
    if degree == 0u {
        return grads;
    }

    let x = viewdir.x;
    let y = viewdir.y;
    let z = viewdir.z;

    let fTmp0A = 0.48860251190292;
    grads[1] = -fTmp0A * y * v_color;
    grads[2] = fTmp0A * z * v_color;
    grads[3] = -fTmp0A * x * v_color;
    if degree == 1u {
        return grads;
    }

    let z2 = z * z;
    let fTmp0B = -1.092548430592079 * z;
    let fTmp1A = 0.5462742152960395;
    let fC1 = x * x - y * y;
    let fS1 = 2.0 * x * y;
    let pSH6 = 0.9461746957575601 * z2 - 0.3153915652525201;
    let pSH7 = fTmp0B * x;
    let pSH5 = fTmp0B * y;
    let pSH8 = fTmp1A * fC1;
    let pSH4 = fTmp1A * fS1;
    grads[4] = pSH4 * v_color;
    grads[5] = pSH5 * v_color;
    grads[6] = pSH6 * v_color;
    grads[7] = pSH7 * v_color;
    grads[8] = pSH8 * v_color;
    if degree == 2u {
        return grads;
    }

    let fTmp0C = -2.285228997322329 * z2 + 0.4570457994644658;
    let fTmp1B = 1.445305721320277 * z;
    let fTmp2A = -0.5900435899266435;
    let fC2 = x * fC1 - y * fS1;
    let fS2 = x * fS1 + y * fC1;
    let pSH12 = z * (1.865881662950577 * z2 - 1.119528997770346);
    let pSH13 = fTmp0C * x;
    let pSH11 = fTmp0C * y;
    let pSH14 = fTmp1B * fC1;
    let pSH10 = fTmp1B * fS1;
    let pSH15 = fTmp2A * fC2;
    let pSH9 = fTmp2A * fS2;
    grads[9] = pSH9 * v_color;
    grads[10] = pSH10 * v_color;
    grads[11] = pSH11 * v_color;
    grads[12] = pSH12 * v_color;
    grads[13] = pSH13 * v_color;
    grads[14] = pSH14 * v_color;
    grads[15] = pSH15 * v_color;
    if degree == 3u {
        return grads;
    }

    let fTmp0D = z * (-4.683325804901025 * z2 + 2.007139630671868);
    let fTmp1C = 3.31161143515146 * z2 - 0.47308734787878;
    let fTmp2B = -1.770130769779931 * z;
    let fTmp3A = 0.6258357354491763;
    let fC3 = x * fC2 - y * fS2;
    let fS3 = x * fS2 + y * fC2;
    let pSH20 = 1.984313483298443 * z * pSH12 - 1.006230589874905 * pSH6;
    let pSH21 = fTmp0D * x;
    let pSH19 = fTmp0D * y;
    let pSH22 = fTmp1C * fC1;
    let pSH18 = fTmp1C * fS1;
    let pSH23 = fTmp2B * fC2;
    let pSH17 = fTmp2B * fS2;
    let pSH24 = fTmp3A * fC3;
    let pSH16 = fTmp3A * fS3;
    grads[16] = pSH16 * v_color;
    grads[17] = pSH17 * v_color;
    grads[18] = pSH18 * v_color;
    grads[19] = pSH19 * v_color;
    grads[20] = pSH20 * v_color;
    grads[21] = pSH21 * v_color;
    grads[22] = pSH22 * v_color;
    grads[23] = pSH23 * v_color;
    grads[24] = pSH24 * v_color;
    return grads;
}

fn persp_proj_vjp(
    j: mat3x2<f32>,
    mean3d: vec3<f32>,
    cov3d: mat3x3<f32>,
    focal: vec2<f32>,
    pixel_center: vec2<f32>,
    img_size: vec2<u32>,
    v_cov2d: mat2x2<f32>,
    v_mean2d: vec2<f32>,
) -> vec3<f32> {
    let x = mean3d.x;
    let y = mean3d.y;
    let z = mean3d.z;

    let rz = 1.0 / max(abs(z), 0.01);
    let rz2 = rz * rz;

    var v_mean3d = vec3<f32>(
        focal.x * rz * v_mean2d.x,
        focal.y * rz * v_mean2d.y,
        -(focal.x * x * v_mean2d.x + focal.y * y * v_mean2d.y) * rz2,
    );

    let v_j = v_cov2d * j * transpose(cov3d) + transpose(v_cov2d) * j * cov3d;
    let lims_pos = (1.15 * vec2<f32>(img_size) - pixel_center) / focal;
    let lims_neg = (-0.15 * vec2<f32>(img_size) - pixel_center) / focal;

    if x * rz <= lims_pos.x && x * rz >= lims_neg.x {
        v_mean3d.x += -focal.x * rz2 * v_j[2].x;
        v_mean3d.z += 2.0 * focal.x * x * rz2 * rz * v_j[2].x;
    } else {
        let clipped_x = clamp(x * rz, lims_neg.x, lims_pos.x);
        v_mean3d.z += focal.x * clipped_x * rz2 * v_j[2].x;
    }

    if y * rz <= lims_pos.y && y * rz >= lims_neg.y {
        v_mean3d.y += -focal.y * rz2 * v_j[2].y;
        v_mean3d.z += 2.0 * focal.y * y * rz2 * rz * v_j[2].y;
    } else {
        let clipped_y = clamp(y * rz, lims_neg.y, lims_pos.y);
        v_mean3d.z += focal.y * clipped_y * rz2 * v_j[2].y;
    }

    v_mean3d.z += -focal.x * rz2 * v_j[0].x - focal.y * rz2 * v_j[1].y;
    return v_mean3d;
}
