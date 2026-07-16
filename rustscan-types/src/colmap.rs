//! Authoritative COLMAP camera-model metadata shared across RustScan crates.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColmapCameraModelSpec {
    pub id: i32,
    pub name: &'static str,
    pub num_params: usize,
    pub focal_idxs: &'static [usize],
    pub principal_point_idxs: [usize; 2],
    pub extra_idxs: &'static [usize],
}

impl ColmapCameraModelSpec {
    pub const fn has_distortion(self) -> bool {
        !self.extra_idxs.is_empty()
    }
}

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

pub const COLMAP_CAMERA_MODELS: [ColmapCameraModelSpec; 17] = [
    ColmapCameraModelSpec {
        id: 0,
        name: "SIMPLE_PINHOLE",
        num_params: 3,
        focal_idxs: &[0],
        principal_point_idxs: [1, 2],
        extra_idxs: &[],
    },
    ColmapCameraModelSpec {
        id: 1,
        name: "PINHOLE",
        num_params: 4,
        focal_idxs: &[0, 1],
        principal_point_idxs: [2, 3],
        extra_idxs: &[],
    },
    ColmapCameraModelSpec {
        id: 2,
        name: "SIMPLE_RADIAL",
        num_params: 4,
        focal_idxs: &[0],
        principal_point_idxs: [1, 2],
        extra_idxs: &[3],
    },
    ColmapCameraModelSpec {
        id: 3,
        name: "RADIAL",
        num_params: 5,
        focal_idxs: &[0],
        principal_point_idxs: [1, 2],
        extra_idxs: &[3, 4],
    },
    ColmapCameraModelSpec {
        id: 4,
        name: "OPENCV",
        num_params: 8,
        focal_idxs: &[0, 1],
        principal_point_idxs: [2, 3],
        extra_idxs: &[4, 5, 6, 7],
    },
    ColmapCameraModelSpec {
        id: 5,
        name: "OPENCV_FISHEYE",
        num_params: 8,
        focal_idxs: &[0, 1],
        principal_point_idxs: [2, 3],
        extra_idxs: &[4, 5, 6, 7],
    },
    ColmapCameraModelSpec {
        id: 6,
        name: "FULL_OPENCV",
        num_params: 12,
        focal_idxs: &[0, 1],
        principal_point_idxs: [2, 3],
        extra_idxs: &[4, 5, 6, 7, 8, 9, 10, 11],
    },
    ColmapCameraModelSpec {
        id: 7,
        name: "FOV",
        num_params: 5,
        focal_idxs: &[0, 1],
        principal_point_idxs: [2, 3],
        extra_idxs: &[4],
    },
    ColmapCameraModelSpec {
        id: 8,
        name: "SIMPLE_RADIAL_FISHEYE",
        num_params: 4,
        focal_idxs: &[0],
        principal_point_idxs: [1, 2],
        extra_idxs: &[3],
    },
    ColmapCameraModelSpec {
        id: 9,
        name: "RADIAL_FISHEYE",
        num_params: 5,
        focal_idxs: &[0],
        principal_point_idxs: [1, 2],
        extra_idxs: &[3, 4],
    },
    ColmapCameraModelSpec {
        id: 10,
        name: "THIN_PRISM_FISHEYE",
        num_params: 12,
        focal_idxs: &[0, 1],
        principal_point_idxs: [2, 3],
        extra_idxs: &[4, 5, 6, 7, 8, 9, 10, 11],
    },
    ColmapCameraModelSpec {
        id: 11,
        name: "RAD_TAN_THIN_PRISM_FISHEYE",
        num_params: 16,
        focal_idxs: &[0, 1],
        principal_point_idxs: [2, 3],
        extra_idxs: &[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    },
    ColmapCameraModelSpec {
        id: 12,
        name: "SIMPLE_DIVISION",
        num_params: 4,
        focal_idxs: &[0],
        principal_point_idxs: [1, 2],
        extra_idxs: &[3],
    },
    ColmapCameraModelSpec {
        id: 13,
        name: "DIVISION",
        num_params: 5,
        focal_idxs: &[0, 1],
        principal_point_idxs: [2, 3],
        extra_idxs: &[4],
    },
    ColmapCameraModelSpec {
        id: 14,
        name: "SIMPLE_FISHEYE",
        num_params: 3,
        focal_idxs: &[0],
        principal_point_idxs: [1, 2],
        extra_idxs: &[],
    },
    ColmapCameraModelSpec {
        id: 15,
        name: "FISHEYE",
        num_params: 4,
        focal_idxs: &[0, 1],
        principal_point_idxs: [2, 3],
        extra_idxs: &[],
    },
    ColmapCameraModelSpec {
        id: 16,
        name: "EUCM",
        num_params: 6,
        focal_idxs: &[0, 1],
        principal_point_idxs: [2, 3],
        extra_idxs: &[4, 5],
    },
];

pub fn colmap_camera_model_by_id(id: i32) -> Option<&'static ColmapCameraModelSpec> {
    let index = usize::try_from(id).ok()?;
    COLMAP_CAMERA_MODELS
        .get(index)
        .filter(|model| model.id == id)
}

pub fn colmap_camera_model_by_name(name: &str) -> Option<&'static ColmapCameraModelSpec> {
    COLMAP_CAMERA_MODELS.iter().find(|model| model.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_model_table_is_contiguous_and_roundtrips() {
        assert_eq!(COLMAP_CAMERA_MODELS.len(), 17);
        for (id, model) in COLMAP_CAMERA_MODELS.iter().enumerate() {
            assert_eq!(model.id, id as i32);
            assert_eq!(colmap_camera_model_by_id(model.id), Some(model));
            assert_eq!(colmap_camera_model_by_name(model.name), Some(model));
            assert!(model.num_params <= COLMAP_MAX_CAMERA_PARAMS);
        }
    }

    #[test]
    fn division_distortion_index_follows_fx_fy_cx_cy_k_layout() {
        assert_eq!(
            colmap_camera_model_by_id(COLMAP_DIVISION)
                .unwrap()
                .extra_idxs,
            &[4]
        );
    }
}
