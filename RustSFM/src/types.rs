use rustslam::{Descriptors, KeyPoint, Match, SE3};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::sift::SiftFeatures;
use crate::wide::WideDescriptors;

pub const COLMAP_SIMPLE_PINHOLE: i32 = 0;
pub const COLMAP_PINHOLE: i32 = 1;
pub const COLMAP_SIMPLE_RADIAL: i32 = 2;
pub const COLMAP_RADIAL: i32 = 3;
pub const COLMAP_OPENCV: i32 = 4;
pub const COLMAP_OPENCV_FISHEYE: i32 = 5;
pub const COLMAP_FULL_OPENCV: i32 = 6;
pub const COLMAP_FOV: i32 = 7;
pub const COLMAP_SIMPLE_RADIAL_FISHEYE: i32 = 8;
pub const COLMAP_RADIAL_FISHEYE: i32 = 9;
pub const COLMAP_THIN_PRISM_FISHEYE: i32 = 10;
pub const COLMAP_RAD_TAN_THIN_PRISM_FISHEYE: i32 = 11;
pub const COLMAP_SIMPLE_DIVISION: i32 = 12;
pub const COLMAP_DIVISION: i32 = 13;
pub const COLMAP_SIMPLE_FISHEYE: i32 = 14;
pub const COLMAP_FISHEYE: i32 = 15;
pub const COLMAP_EUCM: i32 = 16;
pub const COLMAP_MAX_CAMERA_PARAMS: usize = 16;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CameraModel {
    pub model_id: i32,
    pub num_params: usize,
    pub params: [f64; COLMAP_MAX_CAMERA_PARAMS],
    pub width: u32,
    pub height: u32,
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

impl CameraModel {
    pub fn new_pinhole(width: u32, height: u32, fx: f32, fy: f32, cx: f32, cy: f32) -> Self {
        let mut params = [0.0; COLMAP_MAX_CAMERA_PARAMS];
        params[0] = fx as f64;
        params[1] = fy as f64;
        params[2] = cx as f64;
        params[3] = cy as f64;
        Self {
            model_id: COLMAP_PINHOLE,
            num_params: 4,
            params,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
        }
    }

    pub fn from_colmap(
        model_id: i32,
        width: u32,
        height: u32,
        input_params: &[f64],
    ) -> Option<Self> {
        let expected = colmap_camera_model_num_params(model_id)?;
        if input_params.len() != expected || expected > COLMAP_MAX_CAMERA_PARAMS {
            return None;
        }

        let mut params = [0.0; COLMAP_MAX_CAMERA_PARAMS];
        params[..expected].copy_from_slice(input_params);
        let (fx, fy) = focal_lengths_from_params(model_id, &params)?;
        let (cx, cy) = principal_point_from_params(model_id, &params)?;
        Some(Self {
            model_id,
            num_params: expected,
            params,
            width,
            height,
            fx: fx as f32,
            fy: fy as f32,
            cx: cx as f32,
            cy: cy as f32,
        })
    }

    pub fn cam_from_img_f32(&self, x: f32, y: f32) -> Option<[f32; 2]> {
        let uv = self.cam_from_img(x as f64, y as f64)?;
        Some([uv[0] as f32, uv[1] as f32])
    }

    pub fn img_from_cam_f32(&self, u: f32, v: f32, w: f32) -> Option<[f32; 2]> {
        let xy = self.img_from_cam(u as f64, v as f64, w as f64)?;
        Some([xy[0] as f32, xy[1] as f32])
    }

    pub fn cam_from_img(&self, x: f64, y: f64) -> Option<[f64; 2]> {
        let p = &self.params;
        let uv = match self.model_id {
            COLMAP_SIMPLE_PINHOLE => [(x - p[1]) / p[0], (y - p[2]) / p[0]],
            COLMAP_PINHOLE => [(x - p[2]) / p[0], (y - p[3]) / p[1]],
            COLMAP_SIMPLE_RADIAL => {
                let mut u = (x - p[1]) / p[0];
                let mut v = (y - p[2]) / p[0];
                iterative_undistortion(self.model_id, &p[3..4], &mut u, &mut v)?;
                [u, v]
            }
            COLMAP_RADIAL => {
                let mut u = (x - p[1]) / p[0];
                let mut v = (y - p[2]) / p[0];
                iterative_undistortion(self.model_id, &p[3..5], &mut u, &mut v)?;
                [u, v]
            }
            COLMAP_OPENCV => {
                let mut u = (x - p[2]) / p[0];
                let mut v = (y - p[3]) / p[1];
                iterative_undistortion(self.model_id, &p[4..8], &mut u, &mut v)?;
                [u, v]
            }
            COLMAP_OPENCV_FISHEYE => {
                let mut uu = (x - p[2]) / p[0];
                let mut vv = (y - p[3]) / p[1];
                iterative_undistortion(self.model_id, &p[4..8], &mut uu, &mut vv)?;
                normal_from_fisheye(uu, vv)
            }
            COLMAP_FULL_OPENCV => {
                let mut u = (x - p[2]) / p[0];
                let mut v = (y - p[3]) / p[1];
                iterative_undistortion(self.model_id, &p[4..12], &mut u, &mut v)?;
                [u, v]
            }
            COLMAP_FOV => {
                let uu = (x - p[2]) / p[0];
                let vv = (y - p[3]) / p[1];
                fov_undistortion(p[4], uu, vv)
            }
            COLMAP_SIMPLE_RADIAL_FISHEYE => {
                let mut uu = (x - p[1]) / p[0];
                let mut vv = (y - p[2]) / p[0];
                iterative_undistortion(self.model_id, &p[3..4], &mut uu, &mut vv)?;
                normal_from_fisheye(uu, vv)
            }
            COLMAP_RADIAL_FISHEYE => {
                let mut uu = (x - p[1]) / p[0];
                let mut vv = (y - p[2]) / p[0];
                iterative_undistortion(self.model_id, &p[3..5], &mut uu, &mut vv)?;
                normal_from_fisheye(uu, vv)
            }
            COLMAP_THIN_PRISM_FISHEYE => {
                let mut uu = (x - p[2]) / p[0];
                let mut vv = (y - p[3]) / p[1];
                iterative_undistortion(self.model_id, &p[4..12], &mut uu, &mut vv)?;
                normal_from_fisheye(uu, vv)
            }
            COLMAP_RAD_TAN_THIN_PRISM_FISHEYE => {
                let mut uu = (x - p[2]) / p[0];
                let mut vv = (y - p[3]) / p[1];
                iterative_undistortion(self.model_id, &p[4..16], &mut uu, &mut vv)?;
                normal_from_fisheye(uu, vv)
            }
            COLMAP_SIMPLE_DIVISION => {
                let x0 = (x - p[1]) / p[0];
                let y0 = (y - p[2]) / p[0];
                let r2 = x0 * x0 + y0 * y0;
                let denom = 1.0 + p[3] * r2;
                [x0 / denom, y0 / denom]
            }
            COLMAP_DIVISION => {
                let x0 = (x - p[2]) / p[0];
                let y0 = (y - p[3]) / p[1];
                let r2 = x0 * x0 + y0 * y0;
                let denom = 1.0 + p[4] * r2;
                [x0 / denom, y0 / denom]
            }
            COLMAP_SIMPLE_FISHEYE => {
                let uu = (x - p[1]) / p[0];
                let vv = (y - p[2]) / p[0];
                normal_from_fisheye(uu, vv)
            }
            COLMAP_FISHEYE => {
                let uu = (x - p[2]) / p[0];
                let vv = (y - p[3]) / p[1];
                normal_from_fisheye(uu, vv)
            }
            COLMAP_EUCM => {
                let mut u = (x - p[2]) / p[0];
                let mut v = (y - p[3]) / p[1];
                let alpha = p[4];
                let beta = p[5];
                let r2 = u * u + v * v;
                let gamma = 1.0 - alpha;
                let radicand = 1.0 - (alpha - gamma) * beta * r2;
                if radicand < 0.0 {
                    return None;
                }
                let helper_den = alpha * radicand.sqrt() + gamma;
                if helper_den < f64::EPSILON {
                    return None;
                }
                let helper = (1.0 - alpha * alpha * beta * r2) / helper_den;
                if helper < f64::EPSILON {
                    return None;
                }
                u /= helper;
                v /= helper;
                [u, v]
            }
            _ => return None,
        };
        finite2(uv).then_some(uv)
    }

    pub fn cam_ray_from_img(&self, x: f64, y: f64) -> Option<[f64; 3]> {
        let [u, v] = self.cam_from_img(x, y)?;
        let norm = (u * u + v * v + 1.0).sqrt();
        (norm > 0.0 && norm.is_finite()).then_some([u / norm, v / norm, 1.0 / norm])
    }

    pub fn img_from_cam(&self, u: f64, v: f64, w: f64) -> Option<[f64; 2]> {
        let p = &self.params;
        let xy = match self.model_id {
            COLMAP_SIMPLE_PINHOLE => {
                if w < f64::EPSILON {
                    return None;
                }
                [p[0] * u / w + p[1], p[0] * v / w + p[2]]
            }
            COLMAP_PINHOLE => {
                if w < f64::EPSILON {
                    return None;
                }
                [p[0] * u / w + p[2], p[1] * v / w + p[3]]
            }
            COLMAP_SIMPLE_RADIAL => {
                if w < f64::EPSILON {
                    return None;
                }
                let uu = u / w;
                let vv = v / w;
                let [du, dv] = distortion(self.model_id, &p[3..4], uu, vv)?;
                [p[0] * (uu + du) + p[1], p[0] * (vv + dv) + p[2]]
            }
            COLMAP_RADIAL => {
                if w < f64::EPSILON {
                    return None;
                }
                let uu = u / w;
                let vv = v / w;
                let [du, dv] = distortion(self.model_id, &p[3..5], uu, vv)?;
                [p[0] * (uu + du) + p[1], p[0] * (vv + dv) + p[2]]
            }
            COLMAP_OPENCV => {
                if w < f64::EPSILON {
                    return None;
                }
                let uu = u / w;
                let vv = v / w;
                let [du, dv] = distortion(self.model_id, &p[4..8], uu, vv)?;
                [p[0] * (uu + du) + p[2], p[1] * (vv + dv) + p[3]]
            }
            COLMAP_OPENCV_FISHEYE => {
                if w < f64::EPSILON {
                    return None;
                }
                let [uu, vv] = fisheye_from_normal(u / w, v / w);
                let [duu, dvv] = distortion(self.model_id, &p[4..8], uu, vv)?;
                [p[0] * (uu + duu) + p[2], p[1] * (vv + dvv) + p[3]]
            }
            COLMAP_FULL_OPENCV => {
                if w < f64::EPSILON {
                    return None;
                }
                let uu = u / w;
                let vv = v / w;
                let [du, dv] = distortion(self.model_id, &p[4..12], uu, vv)?;
                [p[0] * (uu + du) + p[2], p[1] * (vv + dv) + p[3]]
            }
            COLMAP_FOV => {
                if w < f64::EPSILON {
                    return None;
                }
                let [xd, yd] = fov_distortion(p[4], u / w, v / w);
                [p[0] * xd + p[2], p[1] * yd + p[3]]
            }
            COLMAP_SIMPLE_RADIAL_FISHEYE => {
                if w < f64::EPSILON {
                    return None;
                }
                let [uu, vv] = fisheye_from_normal(u / w, v / w);
                let [duu, dvv] = distortion(self.model_id, &p[3..4], uu, vv)?;
                [p[0] * (uu + duu) + p[1], p[0] * (vv + dvv) + p[2]]
            }
            COLMAP_RADIAL_FISHEYE => {
                if w < f64::EPSILON {
                    return None;
                }
                let [uu, vv] = fisheye_from_normal(u / w, v / w);
                let [duu, dvv] = distortion(self.model_id, &p[3..5], uu, vv)?;
                [p[0] * (uu + duu) + p[1], p[0] * (vv + dvv) + p[2]]
            }
            COLMAP_THIN_PRISM_FISHEYE => {
                if w < f64::EPSILON {
                    return None;
                }
                let [uu, vv] = fisheye_from_normal(u / w, v / w);
                let [duu, dvv] = distortion(self.model_id, &p[4..12], uu, vv)?;
                [p[0] * (uu + duu) + p[2], p[1] * (vv + dvv) + p[3]]
            }
            COLMAP_RAD_TAN_THIN_PRISM_FISHEYE => {
                if w < f64::EPSILON {
                    return None;
                }
                let [uu, vv] = fisheye_from_normal(u / w, v / w);
                let [duu, dvv] = distortion(self.model_id, &p[4..16], uu, vv)?;
                [p[0] * (uu + duu) + p[2], p[1] * (vv + dvv) + p[3]]
            }
            COLMAP_SIMPLE_DIVISION => {
                let rho = (u * u + v * v).sqrt();
                let disc_sq = w * w - 4.0 * rho * rho * p[3];
                if disc_sq < 0.0 {
                    return None;
                }
                let r = 2.0 / (w + disc_sq.sqrt());
                [p[0] * r * u + p[1], p[0] * r * v + p[2]]
            }
            COLMAP_DIVISION => {
                let rho = (u * u + v * v).sqrt();
                let disc_sq = w * w - 4.0 * rho * rho * p[4];
                if disc_sq < 0.0 {
                    return None;
                }
                let r = 2.0 / (w + disc_sq.sqrt());
                [p[0] * r * u + p[2], p[1] * r * v + p[3]]
            }
            COLMAP_SIMPLE_FISHEYE => {
                if w < f64::EPSILON {
                    return None;
                }
                let [uu, vv] = fisheye_from_normal(u / w, v / w);
                [p[0] * uu + p[1], p[0] * vv + p[2]]
            }
            COLMAP_FISHEYE => {
                if w < f64::EPSILON {
                    return None;
                }
                let [uu, vv] = fisheye_from_normal(u / w, v / w);
                [p[0] * uu + p[2], p[1] * vv + p[3]]
            }
            COLMAP_EUCM => {
                if w < f64::EPSILON {
                    return None;
                }
                let alpha = p[4];
                let beta = p[5];
                let rho2 = beta * (u * u + v * v) + w * w;
                if rho2 < 0.0 {
                    return None;
                }
                let rho = rho2.sqrt();
                let den = alpha * rho + (1.0 - alpha) * w;
                if den < f64::EPSILON {
                    return None;
                }
                [p[0] * (u / den) + p[2], p[1] * (v / den) + p[3]]
            }
            _ => return None,
        };
        finite2(xy).then_some(xy)
    }

    pub fn cam_from_img_threshold(&self, threshold_px: f64) -> f64 {
        let Some(focal_idxs) = colmap_camera_model_focal_idxs(self.model_id) else {
            return threshold_px / self.fx.max(self.fy).max(1.0) as f64;
        };
        let mean_focal_length =
            focal_idxs.iter().map(|&idx| self.params[idx]).sum::<f64>() / focal_idxs.len() as f64;
        threshold_px / mean_focal_length
    }

    pub fn model_name(&self) -> &'static str {
        colmap_camera_model_name(self.model_id).unwrap_or("INVALID")
    }

    pub fn params_slice(&self) -> &[f64] {
        &self.params[..self.num_params]
    }

    pub fn set_fx(&mut self, fx: f32) {
        self.fx = fx;
        self.sync_params_from_intrinsics();
    }

    pub fn set_fy(&mut self, fy: f32) {
        self.fy = fy;
        self.sync_params_from_intrinsics();
    }

    pub fn set_cx(&mut self, cx: f32) {
        self.cx = cx;
        self.sync_params_from_intrinsics();
    }

    pub fn set_cy(&mut self, cy: f32) {
        self.cy = cy;
        self.sync_params_from_intrinsics();
    }

    pub fn sync_params_from_intrinsics(&mut self) {
        if let Some(focal_idxs) = colmap_camera_model_focal_idxs(self.model_id) {
            match focal_idxs {
                [idx] => {
                    self.params[*idx] = ((self.fx as f64) + (self.fy as f64)) * 0.5;
                }
                [idx_x, idx_y] => {
                    self.params[*idx_x] = self.fx as f64;
                    self.params[*idx_y] = self.fy as f64;
                }
                _ => {}
            }
        }
        if let Some([idx_x, idx_y]) = colmap_camera_model_principal_point_idxs(self.model_id) {
            self.params[idx_x] = self.cx as f64;
            self.params[idx_y] = self.cy as f64;
        }
    }

    pub fn sync_intrinsics_from_params(&mut self) {
        if let Some((fx, fy)) = focal_lengths_from_params(self.model_id, &self.params) {
            self.fx = fx as f32;
            self.fy = fy as f32;
        }
        if let Some((cx, cy)) = principal_point_from_params(self.model_id, &self.params) {
            self.cx = cx as f32;
            self.cy = cy as f32;
        }
    }

    pub fn has_bogus_params(
        &self,
        min_focal_length_ratio: f64,
        max_focal_length_ratio: f64,
        max_extra_param: f64,
    ) -> bool {
        self.has_bogus_focal_length(min_focal_length_ratio, max_focal_length_ratio)
            || self.has_bogus_principal_point()
            || self.has_bogus_extra_params(max_extra_param)
    }

    pub fn has_bogus_focal_length(
        &self,
        min_focal_length_ratio: f64,
        max_focal_length_ratio: f64,
    ) -> bool {
        let max_dim = self.width.max(self.height).max(1) as f64;
        let Some(focal_idxs) = colmap_camera_model_focal_idxs(self.model_id) else {
            return true;
        };
        focal_idxs.iter().any(|&idx| {
            idx >= self.num_params
                || !self.params[idx].is_finite()
                || self.params[idx] / max_dim < min_focal_length_ratio
                || self.params[idx] / max_dim > max_focal_length_ratio
        })
    }

    pub fn has_bogus_principal_point(&self) -> bool {
        let Some([idx_x, idx_y]) = colmap_camera_model_principal_point_idxs(self.model_id) else {
            return true;
        };
        if idx_x >= self.num_params || idx_y >= self.num_params {
            return true;
        }
        let cx = self.params[idx_x];
        let cy = self.params[idx_y];
        !cx.is_finite()
            || !cy.is_finite()
            || cx < 0.0
            || cx > self.width as f64
            || cy < 0.0
            || cy > self.height as f64
    }

    pub fn has_bogus_extra_params(&self, max_extra_param: f64) -> bool {
        let max_extra_param = max_extra_param.abs();
        colmap_camera_model_extra_idxs(self.model_id)
            .map(|idxs| {
                idxs.iter().any(|&idx| {
                    idx >= self.num_params
                        || !self.params[idx].is_finite()
                        || self.params[idx].abs() > max_extra_param
                })
            })
            .unwrap_or(true)
    }
}

pub fn colmap_camera_model_id(model_name: &str) -> Option<i32> {
    match model_name {
        "SIMPLE_PINHOLE" => Some(COLMAP_SIMPLE_PINHOLE),
        "PINHOLE" => Some(COLMAP_PINHOLE),
        "SIMPLE_RADIAL" => Some(COLMAP_SIMPLE_RADIAL),
        "RADIAL" => Some(COLMAP_RADIAL),
        "OPENCV" => Some(COLMAP_OPENCV),
        "OPENCV_FISHEYE" => Some(COLMAP_OPENCV_FISHEYE),
        "FULL_OPENCV" => Some(COLMAP_FULL_OPENCV),
        "FOV" => Some(COLMAP_FOV),
        "SIMPLE_RADIAL_FISHEYE" => Some(COLMAP_SIMPLE_RADIAL_FISHEYE),
        "RADIAL_FISHEYE" => Some(COLMAP_RADIAL_FISHEYE),
        "THIN_PRISM_FISHEYE" => Some(COLMAP_THIN_PRISM_FISHEYE),
        "RAD_TAN_THIN_PRISM_FISHEYE" => Some(COLMAP_RAD_TAN_THIN_PRISM_FISHEYE),
        "SIMPLE_DIVISION" => Some(COLMAP_SIMPLE_DIVISION),
        "DIVISION" => Some(COLMAP_DIVISION),
        "SIMPLE_FISHEYE" => Some(COLMAP_SIMPLE_FISHEYE),
        "FISHEYE" => Some(COLMAP_FISHEYE),
        "EUCM" => Some(COLMAP_EUCM),
        _ => None,
    }
}

pub fn colmap_camera_model_name(model_id: i32) -> Option<&'static str> {
    match model_id {
        COLMAP_SIMPLE_PINHOLE => Some("SIMPLE_PINHOLE"),
        COLMAP_PINHOLE => Some("PINHOLE"),
        COLMAP_SIMPLE_RADIAL => Some("SIMPLE_RADIAL"),
        COLMAP_RADIAL => Some("RADIAL"),
        COLMAP_OPENCV => Some("OPENCV"),
        COLMAP_OPENCV_FISHEYE => Some("OPENCV_FISHEYE"),
        COLMAP_FULL_OPENCV => Some("FULL_OPENCV"),
        COLMAP_FOV => Some("FOV"),
        COLMAP_SIMPLE_RADIAL_FISHEYE => Some("SIMPLE_RADIAL_FISHEYE"),
        COLMAP_RADIAL_FISHEYE => Some("RADIAL_FISHEYE"),
        COLMAP_THIN_PRISM_FISHEYE => Some("THIN_PRISM_FISHEYE"),
        COLMAP_RAD_TAN_THIN_PRISM_FISHEYE => Some("RAD_TAN_THIN_PRISM_FISHEYE"),
        COLMAP_SIMPLE_DIVISION => Some("SIMPLE_DIVISION"),
        COLMAP_DIVISION => Some("DIVISION"),
        COLMAP_SIMPLE_FISHEYE => Some("SIMPLE_FISHEYE"),
        COLMAP_FISHEYE => Some("FISHEYE"),
        COLMAP_EUCM => Some("EUCM"),
        _ => None,
    }
}

pub fn colmap_camera_model_num_params(model_id: i32) -> Option<usize> {
    match model_id {
        COLMAP_SIMPLE_PINHOLE => Some(3),
        COLMAP_PINHOLE => Some(4),
        COLMAP_SIMPLE_RADIAL => Some(4),
        COLMAP_RADIAL => Some(5),
        COLMAP_OPENCV => Some(8),
        COLMAP_OPENCV_FISHEYE => Some(8),
        COLMAP_FULL_OPENCV => Some(12),
        COLMAP_FOV => Some(5),
        COLMAP_SIMPLE_RADIAL_FISHEYE => Some(4),
        COLMAP_RADIAL_FISHEYE => Some(5),
        COLMAP_THIN_PRISM_FISHEYE => Some(12),
        COLMAP_RAD_TAN_THIN_PRISM_FISHEYE => Some(16),
        COLMAP_SIMPLE_DIVISION => Some(4),
        COLMAP_DIVISION => Some(5),
        COLMAP_SIMPLE_FISHEYE => Some(3),
        COLMAP_FISHEYE => Some(4),
        COLMAP_EUCM => Some(6),
        _ => None,
    }
}

pub fn colmap_camera_model_focal_idxs(model_id: i32) -> Option<&'static [usize]> {
    match model_id {
        COLMAP_SIMPLE_PINHOLE
        | COLMAP_SIMPLE_RADIAL
        | COLMAP_RADIAL
        | COLMAP_SIMPLE_RADIAL_FISHEYE
        | COLMAP_RADIAL_FISHEYE
        | COLMAP_SIMPLE_DIVISION
        | COLMAP_SIMPLE_FISHEYE => Some(&[0]),
        COLMAP_PINHOLE
        | COLMAP_OPENCV
        | COLMAP_OPENCV_FISHEYE
        | COLMAP_FULL_OPENCV
        | COLMAP_FOV
        | COLMAP_THIN_PRISM_FISHEYE
        | COLMAP_RAD_TAN_THIN_PRISM_FISHEYE
        | COLMAP_DIVISION
        | COLMAP_FISHEYE
        | COLMAP_EUCM => Some(&[0, 1]),
        _ => None,
    }
}

pub fn colmap_camera_model_principal_point_idxs(model_id: i32) -> Option<[usize; 2]> {
    match model_id {
        COLMAP_SIMPLE_PINHOLE
        | COLMAP_SIMPLE_RADIAL
        | COLMAP_RADIAL
        | COLMAP_SIMPLE_RADIAL_FISHEYE
        | COLMAP_RADIAL_FISHEYE
        | COLMAP_SIMPLE_DIVISION
        | COLMAP_SIMPLE_FISHEYE => Some([1, 2]),
        COLMAP_PINHOLE
        | COLMAP_OPENCV
        | COLMAP_OPENCV_FISHEYE
        | COLMAP_FULL_OPENCV
        | COLMAP_FOV
        | COLMAP_THIN_PRISM_FISHEYE
        | COLMAP_RAD_TAN_THIN_PRISM_FISHEYE
        | COLMAP_DIVISION
        | COLMAP_FISHEYE
        | COLMAP_EUCM => Some([2, 3]),
        _ => None,
    }
}

pub fn colmap_camera_model_extra_idxs(model_id: i32) -> Option<&'static [usize]> {
    match model_id {
        COLMAP_SIMPLE_PINHOLE | COLMAP_PINHOLE | COLMAP_SIMPLE_FISHEYE | COLMAP_FISHEYE => {
            Some(&[])
        }
        COLMAP_SIMPLE_RADIAL | COLMAP_SIMPLE_RADIAL_FISHEYE | COLMAP_SIMPLE_DIVISION => Some(&[3]),
        COLMAP_RADIAL | COLMAP_RADIAL_FISHEYE | COLMAP_DIVISION | COLMAP_FOV => Some(&[3, 4]),
        COLMAP_OPENCV | COLMAP_OPENCV_FISHEYE => Some(&[4, 5, 6, 7]),
        COLMAP_FULL_OPENCV | COLMAP_THIN_PRISM_FISHEYE => Some(&[4, 5, 6, 7, 8, 9, 10, 11]),
        COLMAP_RAD_TAN_THIN_PRISM_FISHEYE => Some(&[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        COLMAP_EUCM => Some(&[4, 5]),
        _ => None,
    }
}

fn focal_lengths_from_params(
    model_id: i32,
    params: &[f64; COLMAP_MAX_CAMERA_PARAMS],
) -> Option<(f64, f64)> {
    let idxs = colmap_camera_model_focal_idxs(model_id)?;
    match idxs {
        [idx] => Some((params[*idx], params[*idx])),
        [idx_x, idx_y] => Some((params[*idx_x], params[*idx_y])),
        _ => None,
    }
}

fn principal_point_from_params(
    model_id: i32,
    params: &[f64; COLMAP_MAX_CAMERA_PARAMS],
) -> Option<(f64, f64)> {
    let [idx_x, idx_y] = colmap_camera_model_principal_point_idxs(model_id)?;
    Some((params[idx_x], params[idx_y]))
}

fn finite2(values: [f64; 2]) -> bool {
    values[0].is_finite() && values[1].is_finite()
}

fn fisheye_from_normal(u: f64, v: f64) -> [f64; 2] {
    let mut uu = u;
    let mut vv = v;
    let r = (u * u + v * v).sqrt();
    if r > f64::EPSILON {
        let theta = r.atan();
        uu *= theta / r;
        vv *= theta / r;
    }
    [uu, vv]
}

fn normal_from_fisheye(uu: f64, vv: f64) -> [f64; 2] {
    let mut u = uu;
    let mut v = vv;
    let theta = (uu * uu + vv * vv).sqrt();
    let theta_cos_theta = theta * theta.cos();
    if theta_cos_theta > f64::EPSILON {
        let scale = theta.sin() / theta_cos_theta;
        u *= scale;
        v *= scale;
    }
    [u, v]
}

fn fov_distortion(omega: f64, u: f64, v: f64) -> [f64; 2] {
    const EPSILON: f64 = 1.0e-4;
    let radius2 = u * u + v * v;
    let omega2 = omega * omega;
    let factor = if omega2 < EPSILON {
        omega2 * radius2 / 3.0 - omega2 / 12.0 + 1.0
    } else if radius2 < EPSILON {
        let tan_half_omega = (omega / 2.0).tan();
        -2.0 * tan_half_omega * (4.0 * radius2 * tan_half_omega * tan_half_omega - 3.0)
            / (3.0 * omega)
    } else {
        let radius = radius2.sqrt();
        (radius * 2.0 * (omega / 2.0).tan()).atan() / (radius * omega)
    };
    [u * factor, v * factor]
}

fn fov_undistortion(omega: f64, u: f64, v: f64) -> [f64; 2] {
    const EPSILON: f64 = 1.0e-4;
    let radius2 = u * u + v * v;
    let omega2 = omega * omega;
    let factor = if omega2 < EPSILON {
        omega2 * radius2 / 3.0 - omega2 / 12.0 + 1.0
    } else if radius2 < EPSILON {
        omega * (omega * omega * radius2 + 3.0) / (6.0 * (omega / 2.0).tan())
    } else {
        let radius = radius2.sqrt();
        (radius * omega).tan() / (radius * 2.0 * (omega / 2.0).tan())
    };
    [u * factor, v * factor]
}

fn iterative_undistortion(model_id: i32, params: &[f64], u: &mut f64, v: &mut f64) -> Option<()> {
    const NUM_ITERATIONS: usize = 100;
    const MIN_STEP_SQUARED_NORM: f64 = 1.0e-10;
    const REL_STEP_RADIUS: f64 = 0.1;
    const STEP_RADIUS: f64 = 0.1;

    let x0 = [*u, *v];
    let mut x = x0;
    for _ in 0..NUM_ITERATIONS {
        let [dx0, dx1] = distortion(model_id, params, x[0], x[1])?;
        let j = distortion_jacobian(model_id, params, x[0], x[1])?;
        let a00 = 1.0 + j[0][0];
        let a01 = j[0][1];
        let a10 = j[1][0];
        let a11 = 1.0 + j[1][1];
        let b0 = x[0] + dx0 - x0[0];
        let b1 = x[1] + dx1 - x0[1];
        let det = a00 * a11 - a01 * a10;
        if det.abs() <= 1.0e-24 || !det.is_finite() {
            return None;
        }
        let mut step = [(a11 * b0 - a01 * b1) / det, (-a10 * b0 + a00 * b1) / det];
        let radius_sqr = (x[0] * x[0] + x[1] * x[1])
            .mul_add(REL_STEP_RADIUS * REL_STEP_RADIUS, 0.0)
            .max(STEP_RADIUS * STEP_RADIUS);
        let step_norm_sqr = step[0] * step[0] + step[1] * step[1];
        if step_norm_sqr > radius_sqr {
            let scale = (radius_sqr / step_norm_sqr).sqrt();
            step[0] *= scale;
            step[1] *= scale;
        }
        x[0] -= step[0];
        x[1] -= step[1];
        let clamped_step_norm_sqr = step[0] * step[0] + step[1] * step[1];
        if clamped_step_norm_sqr < MIN_STEP_SQUARED_NORM {
            *u = x[0];
            *v = x[1];
            return finite2(x).then_some(());
        }
    }

    *u = x[0];
    *v = x[1];
    None
}

fn distortion_jacobian(model_id: i32, params: &[f64], u: f64, v: f64) -> Option<[[f64; 2]; 2]> {
    let r2 = u * u + v * v;
    let j = match model_id {
        COLMAP_SIMPLE_RADIAL | COLMAP_SIMPLE_RADIAL_FISHEYE => {
            let radial = params[0] * r2;
            let radial_derivative = params[0];
            radial_offset_jacobian(u, v, radial, radial_derivative)
        }
        COLMAP_RADIAL | COLMAP_RADIAL_FISHEYE | COLMAP_OPENCV_FISHEYE => {
            let k1 = params[0];
            let k2 = params[1];
            let k3 = params.get(2).copied().unwrap_or(0.0);
            let k4 = params.get(3).copied().unwrap_or(0.0);
            let r4 = r2 * r2;
            let r6 = r4 * r2;
            let r8 = r4 * r4;
            let radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
            let radial_derivative = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4 + 4.0 * k4 * r6;
            radial_offset_jacobian(u, v, radial, radial_derivative)
        }
        COLMAP_OPENCV => {
            let k1 = params[0];
            let k2 = params[1];
            let p1 = params[2];
            let p2 = params[3];
            let radial = k1 * r2 + k2 * r2 * r2;
            let radial_derivative = k1 + 2.0 * k2 * r2;
            radial_tangential_jacobian(u, v, radial, radial_derivative, p1, p2)
        }
        COLMAP_FULL_OPENCV => {
            let k1 = params[0];
            let k2 = params[1];
            let p1 = params[2];
            let p2 = params[3];
            let k3 = params[4];
            let k4 = params[5];
            let k5 = params[6];
            let k6 = params[7];
            let r4 = r2 * r2;
            let r6 = r4 * r2;
            let num = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
            let den = 1.0 + k4 * r2 + k5 * r4 + k6 * r6;
            if den.abs() <= f64::EPSILON {
                return None;
            }
            let dnum_dr2 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;
            let dden_dr2 = k4 + 2.0 * k5 * r2 + 3.0 * k6 * r4;
            let radial = num / den;
            let radial_derivative = (dnum_dr2 * den - num * dden_dr2) / (den * den);
            radial_tangential_jacobian(u, v, radial - 1.0, radial_derivative, p1, p2)
        }
        COLMAP_THIN_PRISM_FISHEYE => {
            let k1 = params[0];
            let k2 = params[1];
            let p1 = params[2];
            let p2 = params[3];
            let k3 = params[4];
            let k4 = params[5];
            let sx1 = params[6];
            let sy1 = params[7];
            let r4 = r2 * r2;
            let r6 = r4 * r2;
            let r8 = r4 * r4;
            let radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
            let radial_derivative = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4 + 4.0 * k4 * r6;
            let mut j = radial_tangential_jacobian(u, v, radial, radial_derivative, p1, p2);
            j[0][0] += 2.0 * sx1 * u;
            j[0][1] += 2.0 * sx1 * v;
            j[1][0] += 2.0 * sy1 * u;
            j[1][1] += 2.0 * sy1 * v;
            j
        }
        COLMAP_RAD_TAN_THIN_PRISM_FISHEYE => {
            let p0 = params[6];
            let p1 = params[7];
            let s0 = params[8];
            let s1 = params[9];
            let s2 = params[10];
            let s3 = params[11];
            let theta2 = r2;
            let mut th_radial = 1.0;
            let mut th_radial_derivative = 0.0;
            let mut theta_power = 1.0;
            for (idx, coeff) in params[..6].iter().enumerate() {
                th_radial_derivative += (idx as f64 + 1.0) * coeff * theta_power;
                theta_power *= theta2;
                th_radial += coeff * theta_power;
            }

            let x = th_radial * u;
            let y = th_radial * v;
            let dx_du = th_radial + 2.0 * u * u * th_radial_derivative;
            let dx_dv = 2.0 * u * v * th_radial_derivative;
            let dy_du = dx_dv;
            let dy_dv = th_radial + 2.0 * v * v * th_radial_derivative;

            let x2 = x * x;
            let y2 = y * y;
            let r2_distorted = x2 + y2;
            let dtx_dx = 2.0 * p1 * y + 6.0 * p0 * x + 2.0 * s0 * x + 4.0 * s1 * r2_distorted * x;
            let dtx_dy = 2.0 * p1 * x + 2.0 * p0 * y + 2.0 * s0 * y + 4.0 * s1 * r2_distorted * y;
            let dty_dx = 2.0 * p0 * y + 2.0 * p1 * x + 2.0 * s2 * x + 4.0 * s3 * r2_distorted * x;
            let dty_dy = 2.0 * p0 * x + 6.0 * p1 * y + 2.0 * s2 * y + 4.0 * s3 * r2_distorted * y;

            [
                [
                    (1.0 + dtx_dx) * dx_du + dtx_dy * dy_du - 1.0,
                    (1.0 + dtx_dx) * dx_dv + dtx_dy * dy_dv,
                ],
                [
                    dty_dx * dx_du + (1.0 + dty_dy) * dy_du,
                    dty_dx * dx_dv + (1.0 + dty_dy) * dy_dv - 1.0,
                ],
            ]
        }
        COLMAP_SIMPLE_DIVISION | COLMAP_DIVISION => {
            let k = params[0];
            let den = 1.0 + k * r2;
            if den.abs() <= f64::EPSILON {
                return None;
            }
            let factor = k * r2 / den;
            let factor_derivative = k / (den * den);
            [
                [
                    -(factor + 2.0 * u * u * factor_derivative),
                    -2.0 * u * v * factor_derivative,
                ],
                [
                    -2.0 * u * v * factor_derivative,
                    -(factor + 2.0 * v * v * factor_derivative),
                ],
            ]
        }
        _ => return None,
    };
    finite_jacobian(j).then_some(j)
}

fn radial_offset_jacobian(u: f64, v: f64, radial: f64, radial_derivative: f64) -> [[f64; 2]; 2] {
    let uv_radial_derivative = 2.0 * u * v * radial_derivative;
    [
        [
            radial + 2.0 * u * u * radial_derivative,
            uv_radial_derivative,
        ],
        [
            uv_radial_derivative,
            radial + 2.0 * v * v * radial_derivative,
        ],
    ]
}

fn radial_tangential_jacobian(
    u: f64,
    v: f64,
    radial: f64,
    radial_derivative: f64,
    p1: f64,
    p2: f64,
) -> [[f64; 2]; 2] {
    [
        [
            radial + 2.0 * u * u * radial_derivative + 2.0 * p1 * v + 6.0 * p2 * u,
            2.0 * u * v * radial_derivative + 2.0 * p1 * u + 2.0 * p2 * v,
        ],
        [
            2.0 * u * v * radial_derivative + 2.0 * p2 * v + 2.0 * p1 * u,
            radial + 2.0 * v * v * radial_derivative + 6.0 * p1 * v + 2.0 * p2 * u,
        ],
    ]
}

fn finite_jacobian(j: [[f64; 2]; 2]) -> bool {
    j.iter().flatten().all(|value| value.is_finite())
}

fn distortion(model_id: i32, params: &[f64], u: f64, v: f64) -> Option<[f64; 2]> {
    let out = match model_id {
        COLMAP_SIMPLE_RADIAL => {
            let k = params[0];
            let r2 = u * u + v * v;
            let radial = k * r2;
            [u * radial, v * radial]
        }
        COLMAP_RADIAL => {
            let k1 = params[0];
            let k2 = params[1];
            let r2 = u * u + v * v;
            let radial = k1 * r2 + k2 * r2 * r2;
            [u * radial, v * radial]
        }
        COLMAP_OPENCV => {
            let k1 = params[0];
            let k2 = params[1];
            let p1 = params[2];
            let p2 = params[3];
            let u2 = u * u;
            let uv = u * v;
            let v2 = v * v;
            let r2 = u2 + v2;
            let radial = k1 * r2 + k2 * r2 * r2;
            [
                u * radial + 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2),
                v * radial + 2.0 * p2 * uv + p1 * (r2 + 2.0 * v2),
            ]
        }
        COLMAP_OPENCV_FISHEYE => {
            let k1 = params[0];
            let k2 = params[1];
            let k3 = params[2];
            let k4 = params[3];
            let theta2 = u * u + v * v;
            let theta4 = theta2 * theta2;
            let theta6 = theta4 * theta2;
            let theta8 = theta4 * theta4;
            let radial = k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8;
            [u * radial, v * radial]
        }
        COLMAP_FULL_OPENCV => {
            let k1 = params[0];
            let k2 = params[1];
            let p1 = params[2];
            let p2 = params[3];
            let k3 = params[4];
            let k4 = params[5];
            let k5 = params[6];
            let k6 = params[7];
            let u2 = u * u;
            let uv = u * v;
            let v2 = v * v;
            let r2 = u2 + v2;
            let r4 = r2 * r2;
            let r6 = r4 * r2;
            let radial = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6);
            [
                u * radial + 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2) - u,
                v * radial + 2.0 * p2 * uv + p1 * (r2 + 2.0 * v2) - v,
            ]
        }
        COLMAP_SIMPLE_RADIAL_FISHEYE => {
            let k = params[0];
            let theta2 = u * u + v * v;
            let radial = k * theta2;
            [u * radial, v * radial]
        }
        COLMAP_RADIAL_FISHEYE => {
            let k1 = params[0];
            let k2 = params[1];
            let theta2 = u * u + v * v;
            let theta4 = theta2 * theta2;
            let radial = k1 * theta2 + k2 * theta4;
            [u * radial, v * radial]
        }
        COLMAP_THIN_PRISM_FISHEYE => {
            let k1 = params[0];
            let k2 = params[1];
            let p1 = params[2];
            let p2 = params[3];
            let k3 = params[4];
            let k4 = params[5];
            let sx1 = params[6];
            let sy1 = params[7];
            let u2 = u * u;
            let uv = u * v;
            let v2 = v * v;
            let r2 = u2 + v2;
            let r4 = r2 * r2;
            let r6 = r4 * r2;
            let r8 = r6 * r2;
            let radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
            [
                u * radial + 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2) + sx1 * r2,
                v * radial + 2.0 * p2 * uv + p1 * (r2 + 2.0 * v2) + sy1 * r2,
            ]
        }
        COLMAP_RAD_TAN_THIN_PRISM_FISHEYE => {
            let p0 = params[6];
            let p1 = params[7];
            let s0 = params[8];
            let s1 = params[9];
            let s2 = params[10];
            let s3 = params[11];
            let theta2 = u * u + v * v;
            let mut th_radial = 1.0;
            let mut theta_power = 1.0;
            for coeff in &params[..6] {
                theta_power *= theta2;
                th_radial += coeff * theta_power;
            }
            let x = th_radial * u;
            let y = th_radial * v;
            let x2 = x * x;
            let y2 = y * y;
            let xy = x * y;
            let r2 = x2 + y2;
            let r4 = r2 * r2;
            let dx_tang = 2.0 * p1 * xy + p0 * (r2 + 2.0 * x2);
            let dy_tang = 2.0 * p0 * xy + p1 * (r2 + 2.0 * y2);
            let dx_tp = s0 * r2 + s1 * r4;
            let dy_tp = s2 * r2 + s3 * r4;
            [x + dx_tang + dx_tp - u, y + dy_tang + dy_tp - v]
        }
        COLMAP_SIMPLE_DIVISION | COLMAP_DIVISION => {
            let k = params[0];
            let r2 = u * u + v * v;
            let factor = k * r2 / (1.0 + k * r2);
            [-u * factor, -v * factor]
        }
        _ => return None,
    };
    finite2(out).then_some(out)
}

#[derive(Debug, Clone)]
pub struct ImageFrame {
    pub id: usize,
    pub name: String,
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub keypoints: Vec<KeyPoint>,
    pub descriptors: Descriptors,
    pub sift: SiftFeatures,
    pub wide_descriptors: WideDescriptors,
    pub strong_feature_indices: Vec<usize>,
    pub colors: Vec<[u8; 3]>,
}

#[derive(Debug, Clone)]
pub struct PairGeometry {
    pub left: usize,
    pub right: usize,
    pub two_view_config: i32,
    pub matches: Vec<Match>,
    pub inlier_matches: Vec<Match>,
    pub relative_pose: SE3,
    pub inliers: usize,
    pub triangulated: usize,
    pub mean_reprojection_error_px: f32,
    pub rotation_deg: f32,
    pub median_triangulation_angle_deg: f32,
    pub pose_graph_only: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrackObservation {
    pub image: usize,
    pub feature: usize,
}

#[derive(Debug, Clone)]
pub struct Point3D {
    pub xyz: [f32; 3],
    pub color: [u8; 3],
    pub error: f32,
    pub track: Vec<TrackObservation>,
}

#[derive(Debug, Clone)]
pub struct Reconstruction {
    pub camera: CameraModel,
    pub cameras: Vec<CameraModel>,
    pub camera_ids: Vec<u32>,
    pub image_names: Vec<String>,
    pub image_paths: Vec<PathBuf>,
    pub image_ids: Vec<u32>,
    pub image_camera_indices: Vec<usize>,
    pub poses: Vec<Option<SE3>>,
    pub observations: Vec<Vec<Option<usize>>>,
    pub keypoints: Vec<Vec<KeyPoint>>,
    pub point_ids: Vec<u64>,
    pub points: Vec<Point3D>,
}

impl Reconstruction {
    pub fn camera_for_image(&self, image: usize) -> CameraModel {
        self.image_camera_indices
            .get(image)
            .and_then(|&camera_idx| self.cameras.get(camera_idx))
            .copied()
            .unwrap_or(self.camera)
    }

    pub fn camera_id_for_image(&self, image: usize) -> u32 {
        self.image_camera_indices
            .get(image)
            .and_then(|&camera_idx| self.camera_ids.get(camera_idx))
            .copied()
            .unwrap_or(1)
    }

    pub fn image_id(&self, image: usize) -> u32 {
        self.image_ids
            .get(image)
            .copied()
            .unwrap_or_else(|| image as u32 + 1)
    }

    pub fn point3d_id(&self, point: usize) -> u64 {
        self.point_ids
            .get(point)
            .copied()
            .unwrap_or_else(|| point as u64 + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close2(actual: [f64; 2], expected: [f64; 2], eps: f64) {
        assert!(
            (actual[0] - expected[0]).abs() <= eps && (actual[1] - expected[1]).abs() <= eps,
            "actual={actual:?} expected={expected:?} eps={eps}"
        );
    }

    fn numerical_distortion_jacobian(
        model_id: i32,
        params: &[f64],
        u: f64,
        v: f64,
    ) -> [[f64; 2]; 2] {
        let eps = 1.0e-6_f64.max((u.abs() + v.abs()) * 1.0e-8);
        let plus_u = distortion(model_id, params, u + eps, v).unwrap();
        let minus_u = distortion(model_id, params, u - eps, v).unwrap();
        let plus_v = distortion(model_id, params, u, v + eps).unwrap();
        let minus_v = distortion(model_id, params, u, v - eps).unwrap();
        [
            [
                (plus_u[0] - minus_u[0]) / (2.0 * eps),
                (plus_v[0] - minus_v[0]) / (2.0 * eps),
            ],
            [
                (plus_u[1] - minus_u[1]) / (2.0 * eps),
                (plus_v[1] - minus_v[1]) / (2.0 * eps),
            ],
        ]
    }

    fn assert_roundtrip(camera: CameraModel, uv: [f64; 2]) {
        let xy = camera.img_from_cam(uv[0], uv[1], 1.0).unwrap();
        let lifted = camera.cam_from_img(xy[0], xy[1]).unwrap();
        assert_close2(lifted, uv, 1.0e-7);
        let ray = camera.cam_ray_from_img(xy[0], xy[1]).unwrap();
        let norm = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();
        assert!((norm - 1.0).abs() <= 1.0e-12, "ray={ray:?}");
    }

    #[test]
    fn pinhole_projection_matches_colmap_formula() {
        let camera =
            CameraModel::from_colmap(COLMAP_PINHOLE, 800, 600, &[500.0, 510.0, 400.0, 300.0])
                .unwrap();

        assert_close2(
            camera.img_from_cam(0.2, -0.1, 2.0).unwrap(),
            [450.0, 274.5],
            1.0e-12,
        );
        assert_close2(
            camera.cam_from_img(450.0, 274.5).unwrap(),
            [0.1, -0.05],
            1.0e-12,
        );
        assert!((camera.cam_from_img_threshold(2.0) - 2.0 / 505.0).abs() <= 1.0e-12);
    }

    #[test]
    fn cam_from_img_reports_undistortion_failure_without_pinhole_fallback() {
        let camera =
            CameraModel::from_colmap(COLMAP_RADIAL, 800, 600, &[700.0, 400.0, 300.0, -10.0, 0.0])
                .unwrap();

        assert!(camera.cam_from_img(1400.0, 300.0).is_none());
        assert!(camera.cam_from_img_f32(1400.0, 300.0).is_none());
    }

    #[test]
    fn distortion_jacobians_match_finite_differences() {
        let cases = [
            (COLMAP_SIMPLE_RADIAL, vec![0.02]),
            (COLMAP_RADIAL, vec![0.02, -0.001]),
            (COLMAP_OPENCV, vec![0.02, -0.001, 0.0005, -0.0003]),
            (COLMAP_OPENCV_FISHEYE, vec![0.01, -0.001, 0.0001, -0.00001]),
            (
                COLMAP_FULL_OPENCV,
                vec![
                    0.02, -0.001, 0.0005, -0.0003, 0.00001, 0.00002, -0.00001, 0.000005,
                ],
            ),
            (COLMAP_SIMPLE_RADIAL_FISHEYE, vec![0.01]),
            (COLMAP_RADIAL_FISHEYE, vec![0.01, -0.0005]),
            (
                COLMAP_THIN_PRISM_FISHEYE,
                vec![
                    0.01, -0.0005, 0.0002, -0.0001, 0.00001, -0.000005, 0.0003, -0.0002,
                ],
            ),
            (
                COLMAP_RAD_TAN_THIN_PRISM_FISHEYE,
                vec![
                    0.01,
                    -0.0005,
                    0.00004,
                    -0.000003,
                    0.0000002,
                    -0.00000001,
                    0.0002,
                    -0.0001,
                    0.0003,
                    -0.0002,
                    0.00015,
                    -0.00012,
                ],
            ),
            (COLMAP_SIMPLE_DIVISION, vec![0.02]),
            (COLMAP_DIVISION, vec![0.02]),
        ];

        for (model_id, params) in cases {
            let analytic = distortion_jacobian(model_id, &params, 0.08, -0.05).unwrap();
            let numerical = numerical_distortion_jacobian(model_id, &params, 0.08, -0.05);
            for row in 0..2 {
                for col in 0..2 {
                    assert!(
                        (analytic[row][col] - numerical[row][col]).abs() < 1.0e-8,
                        "model={model_id} row={row} col={col} analytic={} numerical={}",
                        analytic[row][col],
                        numerical[row][col]
                    );
                }
            }
        }
    }

    #[test]
    fn camera_bogus_params_match_colmap_threshold_categories() {
        let mut camera = CameraModel::from_colmap(
            COLMAP_OPENCV,
            1000,
            800,
            &[800.0, 810.0, 500.0, 400.0, 0.1, -0.05, 0.01, 0.0],
        )
        .unwrap();
        assert!(!camera.has_bogus_params(0.1, 10.0, 1.0));

        camera.params[0] = 50.0;
        assert!(camera.has_bogus_focal_length(0.1, 10.0));
        camera.params[0] = 800.0;

        camera.params[2] = 1001.0;
        assert!(camera.has_bogus_principal_point());
        camera.params[2] = 500.0;

        camera.params[4] = 1.25;
        assert!(camera.has_bogus_extra_params(1.0));
    }

    #[test]
    fn common_distorted_models_roundtrip_projection() {
        let cases = [
            (COLMAP_SIMPLE_RADIAL, vec![700.0, 400.0, 300.0, 0.02]),
            (COLMAP_RADIAL, vec![700.0, 400.0, 300.0, 0.02, -0.001]),
            (
                COLMAP_OPENCV,
                vec![700.0, 710.0, 400.0, 300.0, 0.02, -0.001, 0.0005, -0.0003],
            ),
            (
                COLMAP_FULL_OPENCV,
                vec![
                    700.0, 710.0, 400.0, 300.0, 0.02, -0.001, 0.0005, -0.0003, 0.00001, 0.00002,
                    -0.00001, 0.000005,
                ],
            ),
            (COLMAP_FOV, vec![700.0, 710.0, 400.0, 300.0, 0.25]),
            (COLMAP_SIMPLE_DIVISION, vec![700.0, 400.0, 300.0, 0.02]),
            (COLMAP_DIVISION, vec![700.0, 710.0, 400.0, 300.0, 0.02]),
            (COLMAP_EUCM, vec![700.0, 710.0, 400.0, 300.0, 0.4, 1.2]),
        ];

        for (model_id, params) in cases {
            let camera = CameraModel::from_colmap(model_id, 800, 600, &params).unwrap();
            assert_roundtrip(camera, [0.08, -0.05]);
        }
    }

    #[test]
    fn fisheye_models_roundtrip_projection() {
        let cases = [
            (
                COLMAP_OPENCV_FISHEYE,
                vec![700.0, 710.0, 400.0, 300.0, 0.01, -0.001, 0.0001, -0.00001],
            ),
            (
                COLMAP_SIMPLE_RADIAL_FISHEYE,
                vec![700.0, 400.0, 300.0, 0.01],
            ),
            (
                COLMAP_RADIAL_FISHEYE,
                vec![700.0, 400.0, 300.0, 0.01, -0.0005],
            ),
            (
                COLMAP_THIN_PRISM_FISHEYE,
                vec![
                    700.0, 710.0, 400.0, 300.0, 0.01, -0.0005, 0.0001, -0.0001, 0.00001, -0.00001,
                    0.00002, -0.00002,
                ],
            ),
            (
                COLMAP_RAD_TAN_THIN_PRISM_FISHEYE,
                vec![
                    700.0, 710.0, 400.0, 300.0, 0.01, -0.0005, 0.00001, -0.000001, 0.0, 0.0,
                    0.0001, -0.0001, 0.00002, -0.00002, 0.00001, -0.00001,
                ],
            ),
            (COLMAP_SIMPLE_FISHEYE, vec![700.0, 400.0, 300.0]),
            (COLMAP_FISHEYE, vec![700.0, 710.0, 400.0, 300.0]),
        ];

        for (model_id, params) in cases {
            let camera = CameraModel::from_colmap(model_id, 800, 600, &params).unwrap();
            assert_roundtrip(camera, [0.08, -0.05]);
        }
    }
}
