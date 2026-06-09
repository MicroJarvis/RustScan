#import helpers;

@group(0) @binding(0) var<storage, read> transforms: array<f32>;
@group(0) @binding(1) var<storage, read> sh_coeffs: array<f32>;
@group(0) @binding(2) var<storage, read> raw_opacities: array<f32>;
@group(0) @binding(3) var<storage, read> global_from_compact_gid: array<u32>;
@group(0) @binding(4) var<storage, read_write> projected: array<helpers::ProjectedSplat>;
@group(0) @binding(5) var<storage, read> uniforms: helpers::ProjectUniforms;

struct ShCoeffs {
    b0_c0: vec3<f32>,
    b1_c0: vec3<f32>,
    b1_c1: vec3<f32>,
    b1_c2: vec3<f32>,
    b2_c0: vec3<f32>,
    b2_c1: vec3<f32>,
    b2_c2: vec3<f32>,
    b2_c3: vec3<f32>,
    b2_c4: vec3<f32>,
    b3_c0: vec3<f32>,
    b3_c1: vec3<f32>,
    b3_c2: vec3<f32>,
    b3_c3: vec3<f32>,
    b3_c4: vec3<f32>,
    b3_c5: vec3<f32>,
    b3_c6: vec3<f32>,
    b4_c0: vec3<f32>,
    b4_c1: vec3<f32>,
    b4_c2: vec3<f32>,
    b4_c3: vec3<f32>,
    b4_c4: vec3<f32>,
    b4_c5: vec3<f32>,
    b4_c6: vec3<f32>,
    b4_c7: vec3<f32>,
    b4_c8: vec3<f32>,
}

fn read_coeff(base_id: ptr<function, u32>) -> vec3<f32> {
    let idx = *base_id;
    *base_id += 3u;
    return vec3<f32>(sh_coeffs[idx], sh_coeffs[idx + 1u], sh_coeffs[idx + 2u]);
}

fn sh_coeffs_to_color(degree: u32, viewdir: vec3<f32>, sh: ShCoeffs) -> vec3<f32> {
    var color = helpers::SH_C0 * sh.b0_c0;
    if degree == 0u {
        return color;
    }

    let x = viewdir.x;
    let y = viewdir.y;
    let z = viewdir.z;

    let fTmp0A = 0.48860251190292;
    color += fTmp0A * (-y * sh.b1_c0 + z * sh.b1_c1 - x * sh.b1_c2);
    if degree == 1u {
        return color;
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
    color += pSH4 * sh.b2_c0 +
        pSH5 * sh.b2_c1 +
        pSH6 * sh.b2_c2 +
        pSH7 * sh.b2_c3 +
        pSH8 * sh.b2_c4;
    if degree == 2u {
        return color;
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
    color += pSH9 * sh.b3_c0 +
        pSH10 * sh.b3_c1 +
        pSH11 * sh.b3_c2 +
        pSH12 * sh.b3_c3 +
        pSH13 * sh.b3_c4 +
        pSH14 * sh.b3_c5 +
        pSH15 * sh.b3_c6;
    if degree == 3u {
        return color;
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
    color += pSH16 * sh.b4_c0 +
        pSH17 * sh.b4_c1 +
        pSH18 * sh.b4_c2 +
        pSH19 * sh.b4_c3 +
        pSH20 * sh.b4_c4 +
        pSH21 * sh.b4_c5 +
        pSH22 * sh.b4_c6 +
        pSH23 * sh.b4_c7 +
        pSH24 * sh.b4_c8;
    return color;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let compact_gid = gid.x;
    if compact_gid >= uniforms.num_visible {
        return;
    }

    let global_gid = global_from_compact_gid[compact_gid];
    let base = global_gid * 10u;
    let mean = vec3<f32>(transforms[base], transforms[base + 1u], transforms[base + 2u]);
    let scale = exp(vec3<f32>(
        transforms[base + 7u],
        transforms[base + 8u],
        transforms[base + 9u],
    ));

    let quat = normalize(vec4<f32>(
        transforms[base + 3u],
        transforms[base + 4u],
        transforms[base + 5u],
        transforms[base + 6u],
    ));
    let mean_c = (uniforms.viewmat * vec4<f32>(mean, 1.0)).xyz;
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

    let conic = helpers::inverse2x2(cov2d);
    let mean2d = uniforms.focal * mean_c.xy * (1.0 / mean_c.z) + uniforms.pixel_center;

    let dir = mean - uniforms.camera_position.xyz;
    let viewdir = dir * inverseSqrt(max(dot(dir, dir), 1e-12));
    let num_coeffs = helpers::num_sh_coeffs(uniforms.sh_degree);
    var coeff_base = global_gid * num_coeffs * 3u;

    var sh = ShCoeffs();
    sh.b0_c0 = read_coeff(&coeff_base);
    if uniforms.sh_degree >= 1u {
        sh.b1_c0 = read_coeff(&coeff_base);
        sh.b1_c1 = read_coeff(&coeff_base);
        sh.b1_c2 = read_coeff(&coeff_base);
    }
    if uniforms.sh_degree >= 2u {
        sh.b2_c0 = read_coeff(&coeff_base);
        sh.b2_c1 = read_coeff(&coeff_base);
        sh.b2_c2 = read_coeff(&coeff_base);
        sh.b2_c3 = read_coeff(&coeff_base);
        sh.b2_c4 = read_coeff(&coeff_base);
    }
    if uniforms.sh_degree >= 3u {
        sh.b3_c0 = read_coeff(&coeff_base);
        sh.b3_c1 = read_coeff(&coeff_base);
        sh.b3_c2 = read_coeff(&coeff_base);
        sh.b3_c3 = read_coeff(&coeff_base);
        sh.b3_c4 = read_coeff(&coeff_base);
        sh.b3_c5 = read_coeff(&coeff_base);
        sh.b3_c6 = read_coeff(&coeff_base);
    }
    if uniforms.sh_degree >= 4u {
        sh.b4_c0 = read_coeff(&coeff_base);
        sh.b4_c1 = read_coeff(&coeff_base);
        sh.b4_c2 = read_coeff(&coeff_base);
        sh.b4_c3 = read_coeff(&coeff_base);
        sh.b4_c4 = read_coeff(&coeff_base);
        sh.b4_c5 = read_coeff(&coeff_base);
        sh.b4_c6 = read_coeff(&coeff_base);
        sh.b4_c7 = read_coeff(&coeff_base);
        sh.b4_c8 = read_coeff(&coeff_base);
    }

    let color = sh_coeffs_to_color(uniforms.sh_degree, viewdir, sh) + vec3<f32>(0.5);
    projected[compact_gid] = helpers::create_projected_splat(
        mean2d,
        vec3<f32>(conic[0][0], conic[0][1], conic[1][1]),
        vec4<f32>(color, opacity),
        mean_c.z,
    );
}
