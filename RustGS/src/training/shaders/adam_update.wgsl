struct Params {
    len: u32,
    scale_len: u32,
    scale_inner_repeat: u32,
    step: u32,
    beta1: f32,
    beta2: f32,
    lr: f32,
    eps: f32,
    weight_decay: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> param: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> moment1: array<f32>;
@group(0) @binding(3) var<storage, read_write> moment2: array<f32>;
@group(0) @binding(4) var<storage, read> scale: array<f32>;
@group(0) @binding(5) var<storage, read> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }

    let param_value = param[idx];
    let grad_value = grad[idx] + param_value * params.weight_decay;
    let m1 = moment1[idx] * params.beta1 + grad_value * (1.0 - params.beta1);
    let m2 = moment2[idx] * params.beta2 + grad_value * grad_value * (1.0 - params.beta2);

    moment1[idx] = m1;
    moment2[idx] = m2;

    let bias_correction1 = 1.0 - pow(params.beta1, f32(params.step));
    let bias_correction2 = 1.0 - pow(params.beta2, f32(params.step));
    let update = (m1 / bias_correction1) / (sqrt(m2 / bias_correction2) + params.eps);

    var scale_idx = 0u;
    if (params.scale_len > 1u) {
        scale_idx = (idx / max(params.scale_inner_repeat, 1u)) % params.scale_len;
    }
    param[idx] = param_value - update * scale[scale_idx] * params.lr;
}
