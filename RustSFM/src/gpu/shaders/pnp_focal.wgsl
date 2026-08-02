struct PnpFocalModel {
    row0: vec4<f32>, row1: vec4<f32>, row2: vec4<f32>,
    log_focal: f32, pad0: f32, pad1: f32, pad2: f32,
}

struct PnpFocalResult {
    selected_model: u32, inliers: u32, valid: u32, pad0: u32,
    residual_sum: f32, focal: f32, pad1: f32, pad2: f32,
}
