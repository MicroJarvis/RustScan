use super::shared::{project_point, BaObservation};
use super::{
    camera_center_pose_jacobian, camera_center_world, compute_pose_covariances,
    position_prior_information_matrix, BundleAdjustmentCovariance, BundleAdjustmentLoss,
    BundleAdjustmentOptions, CovariancePoseBlock, POSE_PRIOR_JACOBIAN_EPS,
};
use crate::sparse_cholesky::{SymmetricSparseMatrix, DENSE_SCHUR_MAX_POSE_ENTITIES};
use crate::types::{
    colmap_camera_model_focal_idxs, colmap_camera_model_principal_point_idxs, CameraModel,
    Reconstruction, Rigid3, SensorId, COLMAP_DIVISION, COLMAP_EUCM, COLMAP_FISHEYE, COLMAP_FOV,
    COLMAP_FULL_OPENCV, COLMAP_OPENCV, COLMAP_OPENCV_FISHEYE, COLMAP_PINHOLE, COLMAP_RADIAL,
    COLMAP_RADIAL_FISHEYE, COLMAP_RAD_TAN_THIN_PRISM_FISHEYE, COLMAP_SIMPLE_DIVISION,
    COLMAP_SIMPLE_FISHEYE, COLMAP_SIMPLE_PINHOLE, COLMAP_SIMPLE_RADIAL,
    COLMAP_SIMPLE_RADIAL_FISHEYE, COLMAP_THIN_PRISM_FISHEYE,
};
type Quat = nalgebra::UnitQuaternion<f32>;
type Vec3 = nalgebra::Vector3<f32>;
use crate::geometry::{UnitQuatNormalize, Vec3GlamExt};
use nalgebra::{DMatrix, DVector, SMatrix, SVector};
use rustscan_slam::SE3;
use std::collections::{BTreeMap, HashSet};

type Mat2x3 = SMatrix<f64, 2, 3>;
type Mat2x6 = SMatrix<f64, 2, 6>;
type Mat2 = SMatrix<f64, 2, 2>;
type Mat3 = SMatrix<f64, 3, 3>;
type Vec2 = SVector<f64, 2>;
type Vec3d = SVector<f64, 3>;
type Vec6 = SVector<f64, 6>;

pub(crate) fn count_variable_residuals(
    reconstruction: &Reconstruction,
    observations: &[BaObservation],
    pose_blocks: &PoseBlockSet,
    sensor_pose_specs: &[SensorPoseSpec],
    camera_param_specs: &[CameraParamSpec],
    constant_point_filter: &HashSet<usize>,
) -> usize {
    let variable_sensors = sensor_pose_specs
        .iter()
        .map(|spec| spec.key.clone())
        .collect::<HashSet<_>>();
    let variable_cameras = camera_param_specs
        .iter()
        .map(|spec| spec.camera)
        .collect::<HashSet<_>>();
    let variable_observations = observations
        .iter()
        .filter(|obs| {
            !constant_point_filter.contains(&obs.point)
                || pose_blocks
                    .image_to_block
                    .get(obs.image)
                    .copied()
                    .flatten()
                    .and_then(|block| pose_blocks.blocks.get(block))
                    .is_some_and(|block| pose_block_dim(block) > 0)
                || frame_sensor_key_for_image(reconstruction, obs.image)
                    .is_some_and(|key| variable_sensors.contains(&key))
                || camera_index_for_image(reconstruction, obs.image)
                    .is_some_and(|camera| variable_cameras.contains(&camera))
        })
        .count();
    variable_observations * 2
}

#[derive(Clone)]
struct SchurSystem {
    h: SchurHessian,
}

#[derive(Debug, Clone)]
enum SchurHessian {
    Dense(DMatrix<f64>),
    Sparse(SymmetricSparseMatrix),
}

impl SchurHessian {
    fn new_dense(n: usize) -> Self {
        Self::Dense(DMatrix::zeros(n, n))
    }

    fn new_for_pose_entities(n: usize, pose_entities: usize) -> Self {
        if pose_entities > DENSE_SCHUR_MAX_POSE_ENTITIES {
            Self::Sparse(SymmetricSparseMatrix::new(n))
        } else {
            Self::new_dense(n)
        }
    }

    fn add(&mut self, row: usize, col: usize, value: f64) {
        match self {
            Self::Dense(matrix) => matrix[(row, col)] += value,
            Self::Sparse(matrix) => matrix.add(row, col, value),
        }
    }

    fn add_lm_damping(&mut self, radius: f64) {
        match self {
            Self::Dense(matrix) => add_lm_damping_to_matrix_diagonal(matrix, radius),
            Self::Sparse(matrix) => {
                matrix.add_lm_damping_to_diagonal(radius, clamp_lm_diagonal);
            }
        }
    }

    fn to_dense(&self) -> DMatrix<f64> {
        match self {
            Self::Dense(matrix) => matrix.clone(),
            Self::Sparse(matrix) => matrix.to_dense(),
        }
    }
}

fn schur_covariance_pose_blocks(pose_blocks: &PoseBlockSet) -> Vec<CovariancePoseBlock> {
    pose_blocks
        .blocks
        .iter()
        .filter_map(|block| {
            let PoseBlockKind::Image(image) = block.kind else {
                return None;
            };
            (pose_block_dim(block) == 6).then_some(CovariancePoseBlock {
                image,
                offset: block.offset,
                dim: 6,
            })
        })
        .collect()
}

pub(crate) fn compute_bundle_adjustment_covariance(
    reconstruction: &Reconstruction,
    observations: &[BaObservation],
    pose_blocks: &PoseBlockSet,
    sensor_pose_specs: &[SensorPoseSpec],
    camera_param_specs: &[CameraParamSpec],
    constant_point_filter: &HashSet<usize>,
    loss_function: BundleAdjustmentLoss,
    pose_priors: &[super::BundleAdjustmentPosePrior],
    prior_position_fallback_stddev: f64,
) -> Option<BundleAdjustmentCovariance> {
    const COVARIANCE_TRUST_REGION_RADIUS: f64 = 1.0e32;
    let system = build_schur_system(
        reconstruction,
        observations,
        pose_blocks,
        sensor_pose_specs,
        camera_param_specs,
        constant_point_filter,
        loss_function,
        COVARIANCE_TRUST_REGION_RADIUS,
        pose_priors,
        prior_position_fallback_stddev,
    )?;
    compute_pose_covariances(
        &system.h.to_dense(),
        &schur_covariance_pose_blocks(pose_blocks),
    )
}

fn pose_block_jacobian_3d(
    jacobian: crate::ba::pose_prior::Mat3x6,
    block: &PoseBlock,
) -> Option<DMatrix<f64>> {
    let dim = pose_block_dim(block);
    if dim == 0 {
        return None;
    }
    let mut out = DMatrix::<f64>::zeros(3, dim);
    let mut local_col = 0usize;
    for axis in 0..6 {
        if block.free_axes[axis] {
            out.set_column(local_col, &jacobian.column(axis));
            local_col += 1;
        }
    }
    Some(out)
}

fn mat3x6_to_dmatrix(matrix: crate::ba::pose_prior::Mat3x6) -> DMatrix<f64> {
    DMatrix::from_fn(3, 6, |row, col| matrix[(row, col)])
}

fn accumulate_pose_priors(
    reconstruction: &Reconstruction,
    pose_blocks: &PoseBlockSet,
    sensor_pose_specs: &[SensorPoseSpec],
    pose_priors: &[super::BundleAdjustmentPosePrior],
    fallback_stddev: f64,
    h_cc: &mut SchurHessian,
    g_c: &mut DVector<f64>,
) {
    let sensor_pose_lookup = sensor_pose_specs
        .iter()
        .map(|spec| (spec.key.clone(), spec.offset))
        .collect::<BTreeMap<_, _>>();
    for prior in pose_priors {
        let Some(pose) = reconstruction.poses.get(prior.image).copied().flatten() else {
            continue;
        };
        let mut nonpoint_jacobians = Vec::new();
        if let Some(block_idx) = pose_blocks
            .image_to_block
            .get(prior.image)
            .copied()
            .flatten()
        {
            let block = &pose_blocks.blocks[block_idx];
            let jacobian = match block.kind {
                PoseBlockKind::Image(_) => camera_center_pose_jacobian(pose),
                PoseBlockKind::Frame(frame_idx) => {
                    let Some(frame_jacobian) =
                        frame_camera_center_jacobian(reconstruction, frame_idx, prior.image)
                    else {
                        continue;
                    };
                    frame_jacobian
                }
            };
            if let Some(jacobian) = pose_block_jacobian_3d(jacobian, block) {
                nonpoint_jacobians.push((block.offset, jacobian));
            }
        }
        if let Some(key) = frame_sensor_key_for_image(reconstruction, prior.image) {
            if let Some(&offset) = sensor_pose_lookup.get(&key) {
                let Some(jacobian) = sensor_camera_center_jacobian(reconstruction, prior.image)
                else {
                    continue;
                };
                nonpoint_jacobians.push((offset, mat3x6_to_dmatrix(jacobian)));
            }
        }
        if nonpoint_jacobians.is_empty() {
            continue;
        }
        let center = camera_center_world(pose);
        let residual = center - DVector::from_column_slice(&prior.position);
        let information =
            position_prior_information_matrix(&prior.position_covariance, fallback_stddev);
        let info_residual = information * residual;
        for (offset, jacobian) in &nonpoint_jacobians {
            for col in 0..jacobian.ncols() {
                g_c[*offset + col] += jacobian.column(col).dot(&info_residual);
            }
        }
        for (offset_i, jacobian_i) in &nonpoint_jacobians {
            for (offset_j, jacobian_j) in &nonpoint_jacobians {
                for row in 0..jacobian_i.ncols() {
                    for col in 0..jacobian_j.ncols() {
                        h_cc.add(
                            *offset_i + row,
                            *offset_j + col,
                            jacobian_i
                                .column(row)
                                .dot(&(information * jacobian_j.column(col))),
                        );
                    }
                }
            }
        }
    }
}

const LM_MIN_DIAGONAL: f64 = 1.0e-6;
const LM_MAX_DIAGONAL: f64 = 1.0e32;
const MIN_TRUST_REGION_RADIUS: f64 = 1.0e-32;

fn clamp_lm_diagonal(value: f64) -> f64 {
    if !value.is_finite() {
        return LM_MIN_DIAGONAL;
    }
    value.clamp(LM_MIN_DIAGONAL, LM_MAX_DIAGONAL)
}

fn add_lm_damping_to_matrix_diagonal(matrix: &mut DMatrix<f64>, radius: f64) {
    let radius = radius.max(MIN_TRUST_REGION_RADIUS);
    for idx in 0..matrix.nrows() {
        let diag = clamp_lm_diagonal(matrix[(idx, idx)]);
        matrix[(idx, idx)] += diag / radius;
    }
}

fn add_lm_damping_to_mat3_diagonal(matrix: &mut Mat3, radius: f64) {
    let radius = radius.max(MIN_TRUST_REGION_RADIUS);
    for d in 0..3 {
        let diag = clamp_lm_diagonal(matrix[(d, d)]);
        matrix[(d, d)] += diag / radius;
    }
}

#[derive(Debug, Clone)]
struct PointBlock {
    h_inv: Mat3,
    g: Vec3d,
    nonpoint_blocks: Vec<NonPointBlock>,
}

#[derive(Debug, Clone)]
struct NonPointBlock {
    offset: usize,
    jacobian: DMatrix<f64>,
}

#[derive(Debug, Clone)]
pub(crate) struct PoseBlockSet {
    pub(crate) blocks: Vec<PoseBlock>,
    pub(crate) image_to_block: Vec<Option<usize>>,
    pub(crate) dim: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct PoseBlock {
    pub(crate) kind: PoseBlockKind,
    pub(crate) images: Vec<usize>,
    pub(crate) offset: usize,
    pub(crate) free_axes: [bool; 6],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PoseBlockKind {
    Image(usize),
    Frame(usize),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct SensorPoseKey {
    pub(crate) rig_id: u32,
    pub(crate) sensor_id: SensorId,
}

#[derive(Debug, Clone)]
pub(crate) struct SensorPoseSpec {
    pub(crate) key: SensorPoseKey,
    pub(crate) offset: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CameraParamSpec {
    pub(crate) camera: usize,
    pub(crate) param: usize,
    pub(crate) offset: usize,
}

pub(crate) fn variable_pose_blocks(
    reconstruction: &Reconstruction,
    variable_images: Option<&[usize]>,
    constant_images: &[usize],
    constant_rigs: &[u32],
    apply_default_gauge: bool,
) -> PoseBlockSet {
    let mut constant_images = constant_images.iter().copied().collect::<HashSet<_>>();
    let constant_rigs = constant_rigs.iter().copied().collect::<HashSet<_>>();
    let mut constant_frames = constant_images
        .iter()
        .filter_map(|&image| reconstruction.frame_index_for_image(image))
        .collect::<HashSet<_>>();
    for (frame_idx, frame) in reconstruction.frames.iter().enumerate() {
        if constant_rigs.contains(&frame.rig_id) {
            constant_frames.insert(frame_idx);
        }
    }
    if apply_default_gauge
        && variable_images.is_none()
        && constant_images.is_empty()
        && constant_frames.is_empty()
        && !reconstruction.poses.is_empty()
    {
        constant_images.insert(0);
        if let Some(frame_idx) = reconstruction.frame_index_for_image(0) {
            constant_frames.insert(frame_idx);
        }
    }
    let candidate_images = if let Some(images) = variable_images {
        images.to_vec()
    } else {
        reconstruction
            .poses
            .iter()
            .enumerate()
            .filter_map(|(idx, pose)| pose.is_some().then_some(idx))
            .collect()
    };

    let mut frame_candidates = BTreeMap::<usize, ()>::new();
    let mut image_candidates = Vec::new();
    for image in candidate_images {
        if image >= reconstruction.poses.len() || reconstruction.poses[image].is_none() {
            continue;
        }
        if let Some(frame_idx) = reconstruction.frame_index_for_image(image) {
            if constant_frames.contains(&frame_idx)
                || frame_registered_images_with_sensors(reconstruction, frame_idx).is_none()
            {
                continue;
            }
            frame_candidates.insert(frame_idx, ());
        } else if !constant_images.contains(&image) {
            image_candidates.push(image);
        }
    }

    image_candidates.sort_unstable();
    image_candidates.dedup();
    let mut blocks = Vec::new();
    let mut image_to_block = vec![None; reconstruction.poses.len()];
    for frame_idx in frame_candidates.keys().copied() {
        let Some(images) = frame_registered_images_with_sensors(reconstruction, frame_idx) else {
            continue;
        };
        let block_idx = blocks.len();
        for &image in &images {
            image_to_block[image] = Some(block_idx);
        }
        blocks.push(PoseBlock {
            kind: PoseBlockKind::Frame(frame_idx),
            images,
            offset: block_idx * 6,
            free_axes: [true; 6],
        });
    }
    for image in image_candidates {
        let block_idx = blocks.len();
        image_to_block[image] = Some(block_idx);
        blocks.push(PoseBlock {
            kind: PoseBlockKind::Image(image),
            images: vec![image],
            offset: block_idx * 6,
            free_axes: [true; 6],
        });
    }

    let mut pose_blocks = PoseBlockSet {
        blocks,
        image_to_block,
        dim: 0,
    };
    reindex_pose_blocks(&mut pose_blocks);
    pose_blocks
}

fn reindex_pose_blocks(pose_blocks: &mut PoseBlockSet) {
    let mut offset = 0usize;
    for block in &mut pose_blocks.blocks {
        block.offset = offset;
        offset += pose_block_dim(block);
    }
    pose_blocks.dim = offset;
}

pub(crate) fn pose_block_dim(block: &PoseBlock) -> usize {
    block.free_axes.iter().filter(|&&free| free).count()
}

fn pose_block_active_axes(block: &PoseBlock) -> Vec<usize> {
    block
        .free_axes
        .iter()
        .enumerate()
        .filter_map(|(axis, &free)| free.then_some(axis))
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum GaugeUnitKey {
    Image(usize),
    Frame(usize),
}

#[derive(Debug, Clone, Copy)]
struct GaugePoseUnit {
    key: GaugeUnitKey,
    block: Option<usize>,
    pose: SE3,
}

pub(crate) fn apply_two_cams_from_world_gauge(
    pose_blocks: &mut PoseBlockSet,
    reconstruction: &Reconstruction,
    options: &BundleAdjustmentOptions,
    observations: &[BaObservation],
) -> bool {
    if pose_blocks.dim == 0 {
        return true;
    }

    let units = gauge_pose_units(reconstruction, pose_blocks, options, observations);
    let mut fixed_unit: Option<GaugePoseUnit> = None;
    for unit in &units {
        if !gauge_unit_pose_is_constant(*unit, pose_blocks) {
            continue;
        }
        if let Some(first) = fixed_unit {
            if first.key != unit.key {
                return true;
            }
        } else {
            fixed_unit = Some(*unit);
        }
    }

    let mut first_unit = fixed_unit;
    let mut second_unit = None;
    let mut fixed_dim = 0usize;
    for unit in units {
        if first_unit.is_none() {
            first_unit = Some(unit);
            continue;
        }
        let first = first_unit.unwrap();
        if first.key == unit.key || !gauge_unit_pose_is_variable(unit, pose_blocks) {
            continue;
        }
        if let Some(dim) = two_cam_gauge_fixed_translation_dim(first.pose, unit.pose) {
            second_unit = Some(unit);
            fixed_dim = dim;
            break;
        }
    }

    let (Some(first), Some(second)) = (first_unit, second_unit) else {
        return false;
    };
    if let Some(block_idx) = first.block {
        pose_blocks.blocks[block_idx].free_axes = [false; 6];
    }
    if let Some(block_idx) = second.block {
        pose_blocks.blocks[block_idx].free_axes[3 + fixed_dim] = false;
    }
    true
}

fn gauge_pose_units(
    reconstruction: &Reconstruction,
    pose_blocks: &PoseBlockSet,
    options: &BundleAdjustmentOptions,
    observations: &[BaObservation],
) -> Vec<GaugePoseUnit> {
    let mut units = Vec::new();
    let mut seen = HashSet::new();
    let mut images = observations.iter().map(|obs| obs.image).collect::<Vec<_>>();
    images.sort_unstable();
    images.dedup();
    for image in images {
        if reconstruction.poses[image].is_none()
            || !gauge_image_sensor_is_constant(reconstruction, image, options)
        {
            continue;
        }
        let key = reconstruction
            .frame_index_for_image(image)
            .map(GaugeUnitKey::Frame)
            .unwrap_or(GaugeUnitKey::Image(image));
        if !seen.insert(key) {
            continue;
        }
        let pose = match key {
            GaugeUnitKey::Image(_) => reconstruction.poses[image],
            GaugeUnitKey::Frame(frame_idx) => reconstruction
                .frames
                .get(frame_idx)
                .map(|frame| frame.rig_from_world.to_se3()),
        };
        let Some(pose) = pose else {
            continue;
        };
        let block = pose_blocks.image_to_block.get(image).copied().flatten();
        units.push(GaugePoseUnit { key, block, pose });
    }
    units
}

fn gauge_image_sensor_is_constant(
    reconstruction: &Reconstruction,
    image: usize,
    options: &BundleAdjustmentOptions,
) -> bool {
    let Some(key) = frame_sensor_key_for_image(reconstruction, image) else {
        return true;
    };
    ref_sensor_key(reconstruction, &key)
        || options
            .constant_sensor_from_rig
            .iter()
            .any(|sensor_id| sensor_id == &key.sensor_id)
}

fn gauge_unit_pose_is_constant(unit: GaugePoseUnit, pose_blocks: &PoseBlockSet) -> bool {
    unit.block
        .and_then(|block| pose_blocks.blocks.get(block))
        .is_none_or(|block| pose_block_dim(block) == 0)
}

fn gauge_unit_pose_is_variable(unit: GaugePoseUnit, pose_blocks: &PoseBlockSet) -> bool {
    unit.block
        .and_then(|block| pose_blocks.blocks.get(block))
        .is_some_and(|block| pose_block_dim(block) > 0)
}

fn two_cam_gauge_fixed_translation_dim(first: SE3, second: SE3) -> Option<usize> {
    let baseline = first.compose(&second.inverse()).translation();
    let mut fixed_dim = 0usize;
    let mut max_abs = baseline[0].abs();
    for dim in 1..3 {
        let value = baseline[dim].abs();
        if value > max_abs {
            max_abs = value;
            fixed_dim = dim;
        }
    }
    (max_abs > 1.0e-9).then_some(fixed_dim)
}

fn frame_registered_images_with_sensors(
    reconstruction: &Reconstruction,
    frame_idx: usize,
) -> Option<Vec<usize>> {
    let images = reconstruction
        .image_indices_for_frame_index(frame_idx)
        .into_iter()
        .filter(|&image| reconstruction.poses.get(image).copied().flatten().is_some())
        .collect::<Vec<_>>();
    if images.is_empty()
        || images
            .iter()
            .any(|&image| frame_sensor_from_rig(reconstruction, frame_idx, image).is_none())
    {
        None
    } else {
        Some(images)
    }
}

pub(crate) fn camera_param_specs(
    reconstruction: &Reconstruction,
    observations: &[BaObservation],
    options: &BundleAdjustmentOptions,
    first_offset: usize,
) -> Vec<CameraParamSpec> {
    if !(options.refine_focal_length
        || options.refine_principal_point
        || options.refine_extra_params)
    {
        return Vec::new();
    }

    let constant_cameras = options
        .constant_cameras
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let mut camera_indices = if let Some(cameras) = options.variable_cameras.as_deref() {
        cameras.to_vec()
    } else {
        observations
            .iter()
            .filter_map(|obs| camera_index_for_image(reconstruction, obs.image))
            .collect::<Vec<_>>()
    };
    camera_indices.sort_unstable();
    camera_indices.dedup();

    let mut specs = Vec::new();
    for camera_idx in camera_indices {
        if constant_cameras.contains(&camera_idx) {
            continue;
        }
        let Some(camera) = camera_by_index(reconstruction, camera_idx) else {
            continue;
        };
        for param in selected_camera_params(
            camera,
            options.refine_focal_length,
            options.refine_principal_point,
            options.refine_extra_params,
        ) {
            specs.push(CameraParamSpec {
                camera: camera_idx,
                param,
                offset: first_offset + specs.len(),
            });
        }
    }
    specs
}

pub(crate) fn sensor_pose_specs(
    reconstruction: &Reconstruction,
    pose_blocks: &PoseBlockSet,
    options: &BundleAdjustmentOptions,
) -> Vec<SensorPoseSpec> {
    let constant_sensors = options
        .constant_sensor_from_rig
        .iter()
        .cloned()
        .collect::<HashSet<_>>();
    let mut keys = BTreeMap::<SensorPoseKey, ()>::new();
    for block in &pose_blocks.blocks {
        let PoseBlockKind::Frame(_) = block.kind else {
            continue;
        };
        for &image in &block.images {
            let Some(key) = frame_sensor_key_for_image(reconstruction, image) else {
                continue;
            };
            if constant_sensors.contains(&key.sensor_id) || ref_sensor_key(reconstruction, &key) {
                continue;
            }
            keys.insert(key, ());
        }
    }
    keys.keys()
        .cloned()
        .enumerate()
        .map(|(idx, key)| SensorPoseSpec {
            key,
            offset: pose_blocks.dim + idx * 6,
        })
        .collect()
}

fn ref_sensor_key(reconstruction: &Reconstruction, key: &SensorPoseKey) -> bool {
    reconstruction
        .rigs
        .iter()
        .find(|rig| rig.rig_id == key.rig_id)
        .and_then(|rig| rig.ref_sensor_id.as_ref())
        .is_some_and(|ref_sensor_id| ref_sensor_id == &key.sensor_id)
}

fn selected_camera_params(
    camera: CameraModel,
    refine_focal_length: bool,
    refine_principal_point: bool,
    refine_extra_params: bool,
) -> Vec<usize> {
    let focal = colmap_camera_model_focal_idxs(camera.model_id).unwrap_or(&[]);
    let principal = colmap_camera_model_principal_point_idxs(camera.model_id)
        .map(|idxs| idxs.to_vec())
        .unwrap_or_default();
    let mut selected = Vec::new();
    if refine_focal_length {
        selected.extend(focal.iter().copied());
    }
    if refine_principal_point {
        selected.extend(principal.iter().copied());
    }
    if refine_extra_params {
        selected.extend(
            (0..camera.num_params).filter(|idx| !focal.contains(idx) && !principal.contains(idx)),
        );
    }
    selected.retain(|&idx| idx < camera.num_params);
    selected.sort_unstable();
    selected.dedup();
    selected
}

pub(crate) fn camera_index_for_image(
    reconstruction: &Reconstruction,
    image: usize,
) -> Option<usize> {
    match reconstruction.image_camera_indices.get(image).copied() {
        Some(camera_idx) if camera_idx < reconstruction.cameras.len() => Some(camera_idx),
        Some(0) if reconstruction.cameras.is_empty() => Some(0),
        Some(_) => None,
        None => {
            if reconstruction.cameras.is_empty() || !reconstruction.poses.is_empty() {
                Some(0)
            } else {
                None
            }
        }
    }
}

pub(crate) fn camera_by_index(
    reconstruction: &Reconstruction,
    camera_idx: usize,
) -> Option<CameraModel> {
    reconstruction
        .cameras
        .get(camera_idx)
        .copied()
        .or_else(|| (camera_idx == 0).then_some(reconstruction.camera))
}

fn build_schur_system(
    reconstruction: &Reconstruction,
    observations: &[BaObservation],
    pose_blocks: &PoseBlockSet,
    sensor_pose_specs: &[SensorPoseSpec],
    camera_param_specs: &[CameraParamSpec],
    constant_point_filter: &HashSet<usize>,
    loss_function: BundleAdjustmentLoss,
    radius: f64,
    pose_priors: &[super::BundleAdjustmentPosePrior],
    prior_position_fallback_stddev: f64,
) -> Option<SchurSystem> {
    let mut camera_param_lookup = vec![
        Vec::new();
        camera_param_specs
            .iter()
            .map(|s| s.camera)
            .max()
            .unwrap_or(0)
            + 1
    ];
    for (idx, spec) in camera_param_specs.iter().enumerate() {
        camera_param_lookup[spec.camera].push(idx);
    }
    let sensor_pose_lookup = sensor_pose_specs
        .iter()
        .map(|spec| (spec.key.clone(), spec.offset))
        .collect::<BTreeMap<_, _>>();

    let nonpoint_dim = pose_blocks.dim + sensor_pose_specs.len() * 6 + camera_param_specs.len();
    let pose_entities = pose_blocks.blocks.len() + sensor_pose_specs.len();
    let mut h_cc = SchurHessian::new_for_pose_entities(nonpoint_dim, pose_entities);
    let mut g_c = DVector::<f64>::zeros(nonpoint_dim);
    let mut point_blocks = (0..reconstruction.points.len())
        .map(|_| PointBlock {
            h_inv: Mat3::zeros(),
            g: Vec3d::zeros(),
            nonpoint_blocks: Vec::new(),
        })
        .collect::<Vec<_>>();

    for obs in observations {
        let pose = reconstruction.poses[obs.image]?;
        let point = reconstruction.points.get(obs.point)?.xyz;
        let (residual, j_pose, j_point) = residual_and_jacobians(
            reconstruction.camera_for_image(obs.image),
            pose,
            point,
            obs.xy,
        )?;
        let err = residual.norm();
        let weight = loss_function.weight(err * err);
        let sqrt_w = weight.sqrt();
        let residual = residual * sqrt_w;
        let j_pose = j_pose * sqrt_w;
        let j_point = j_point * sqrt_w;

        let mut nonpoint_jacobians = Vec::new();
        if let Some(block_idx) = pose_blocks.image_to_block.get(obs.image).copied().flatten() {
            let block = &pose_blocks.blocks[block_idx];
            let j_pose = match block.kind {
                PoseBlockKind::Image(_) => j_pose,
                PoseBlockKind::Frame(frame_idx) => {
                    frame_pose_jacobian(
                        reconstruction,
                        frame_idx,
                        obs.image,
                        reconstruction.camera_for_image(obs.image),
                        point,
                    )? * sqrt_w
                }
            };
            if let Some(j_pose) = pose_block_jacobian(j_pose, block) {
                nonpoint_jacobians.push((block.offset, j_pose));
            }
        }
        if let Some(key) = frame_sensor_key_for_image(reconstruction, obs.image) {
            if let Some(&offset) = sensor_pose_lookup.get(&key) {
                let j_sensor = sensor_pose_jacobian(
                    reconstruction,
                    obs.image,
                    reconstruction.camera_for_image(obs.image),
                    point,
                )? * sqrt_w;
                nonpoint_jacobians.push((offset, mat2x6_to_dmatrix(j_sensor)));
            }
        }
        if let Some(camera_idx) = camera_index_for_image(reconstruction, obs.image) {
            if camera_idx < camera_param_lookup.len() {
                let camera = camera_by_index(reconstruction, camera_idx)?;
                for &spec_idx in &camera_param_lookup[camera_idx] {
                    let spec = camera_param_specs[spec_idx];
                    if let Some(j_param) = camera_param_jacobian(camera, spec.param, pose, point) {
                        nonpoint_jacobians.push((spec.offset, vec2_to_dmatrix(j_param * sqrt_w)));
                    }
                }
            }
        }

        let residual_d = DVector::from_column_slice(&[residual[0], residual[1]]);
        let point_is_constant = constant_point_filter.contains(&obs.point);
        if !point_is_constant {
            let point_block = &mut point_blocks[obs.point];
            point_block.h_inv += j_point.transpose() * j_point;
            point_block.g += j_point.transpose() * residual;
            for (offset, jacobian) in &nonpoint_jacobians {
                point_block.nonpoint_blocks.push(NonPointBlock {
                    offset: *offset,
                    jacobian: point_nonpoint_cross(j_point, jacobian),
                });
            }
        }
        for (offset, jacobian) in &nonpoint_jacobians {
            let g = jacobian.transpose() * &residual_d;
            for r in 0..jacobian.ncols() {
                g_c[*offset + r] += g[r];
            }
        }
        for (offset_i, jacobian_i) in &nonpoint_jacobians {
            for (offset_j, jacobian_j) in &nonpoint_jacobians {
                let h = jacobian_i.transpose() * jacobian_j;
                for r in 0..jacobian_i.ncols() {
                    for c in 0..jacobian_j.ncols() {
                        h_cc.add(*offset_i + r, *offset_j + c, h[(r, c)]);
                    }
                }
            }
        }
    }

    h_cc.add_lm_damping(radius);
    accumulate_pose_priors(
        reconstruction,
        pose_blocks,
        sensor_pose_specs,
        pose_priors,
        prior_position_fallback_stddev,
        &mut h_cc,
        &mut g_c,
    );

    for block in &mut point_blocks {
        if block.nonpoint_blocks.is_empty() {
            continue;
        }
        add_lm_damping_to_mat3_diagonal(&mut block.h_inv, radius);
        block.h_inv = block.h_inv.try_inverse()?;
        for e_i in &block.nonpoint_blocks {
            let schur_g = e_i.jacobian.transpose() * block.h_inv * block.g;
            for r in 0..e_i.jacobian.ncols() {
                g_c[e_i.offset + r] -= schur_g[r];
            }
            for e_j in &block.nonpoint_blocks {
                let schur_h = e_i.jacobian.transpose() * block.h_inv * &e_j.jacobian;
                for r in 0..e_i.jacobian.ncols() {
                    for c in 0..e_j.jacobian.ncols() {
                        h_cc.add(e_i.offset + r, e_j.offset + c, -schur_h[(r, c)]);
                    }
                }
            }
        }
    }

    Some(SchurSystem { h: h_cc })
}

pub(crate) fn set_frame_pose_block(
    reconstruction: &mut Reconstruction,
    frame_idx: usize,
    images: &[usize],
    rig_from_world: SE3,
) {
    let image_poses = images
        .iter()
        .filter_map(|&image| {
            let sensor_from_rig = frame_sensor_from_rig(reconstruction, frame_idx, image)?;
            Some((image, sensor_from_rig.compose(&rig_from_world)))
        })
        .collect::<Vec<_>>();
    if let Some(frame) = reconstruction.frames.get_mut(frame_idx) {
        frame.rig_from_world = Rigid3::from_se3(rig_from_world);
    }
    for (image, pose) in image_poses {
        if let Some(slot) = reconstruction.poses.get_mut(image) {
            *slot = Some(pose);
        }
    }
}

pub(crate) fn frame_sensor_from_rig(
    reconstruction: &Reconstruction,
    frame_idx: usize,
    image: usize,
) -> Option<SE3> {
    let frame = reconstruction.frames.get(frame_idx)?;
    let sensor_id = reconstruction.frame_sensor_id_for_image(frame_idx, image)?;
    reconstruction.sensor_from_rig(frame.rig_id, sensor_id)
}

pub(crate) fn frame_sensor_key_for_image(
    reconstruction: &Reconstruction,
    image: usize,
) -> Option<SensorPoseKey> {
    let frame_idx = reconstruction.frame_index_for_image(image)?;
    let frame = reconstruction.frames.get(frame_idx)?;
    let sensor_id = reconstruction
        .frame_sensor_id_for_image(frame_idx, image)?
        .clone();
    Some(SensorPoseKey {
        rig_id: frame.rig_id,
        sensor_id,
    })
}

fn frame_pose_jacobian(
    reconstruction: &Reconstruction,
    frame_idx: usize,
    image: usize,
    camera: CameraModel,
    point: [f32; 3],
) -> Option<Mat2x6> {
    let frame = reconstruction.frames.get(frame_idx)?;
    let rig_from_world = frame.rig_from_world.to_se3();
    let sensor_from_rig = frame_sensor_from_rig(reconstruction, frame_idx, image)?;
    analytic_frame_pose_jacobian(camera, sensor_from_rig, rig_from_world, point)
        .or_else(|| numerical_frame_pose_jacobian(camera, sensor_from_rig, rig_from_world, point))
}

fn sensor_pose_jacobian(
    reconstruction: &Reconstruction,
    image: usize,
    camera: CameraModel,
    point: [f32; 3],
) -> Option<Mat2x6> {
    let frame_idx = reconstruction.frame_index_for_image(image)?;
    let frame = reconstruction.frames.get(frame_idx)?;
    let rig_from_world = frame.rig_from_world.to_se3();
    let sensor_from_rig = frame_sensor_from_rig(reconstruction, frame_idx, image)?;
    analytic_sensor_pose_jacobian(camera, sensor_from_rig, rig_from_world, point)
        .or_else(|| numerical_sensor_pose_jacobian(camera, sensor_from_rig, rig_from_world, point))
}

fn frame_camera_center_jacobian(
    reconstruction: &Reconstruction,
    frame_idx: usize,
    image: usize,
) -> Option<crate::ba::pose_prior::Mat3x6> {
    let frame = reconstruction.frames.get(frame_idx)?;
    let rig_from_world = frame.rig_from_world.to_se3();
    let sensor_from_rig = frame_sensor_from_rig(reconstruction, frame_idx, image)?;
    let mut jacobian = crate::ba::pose_prior::Mat3x6::zeros();
    for axis in 0..6 {
        let mut plus = Vec6::zeros();
        plus[axis] = POSE_PRIOR_JACOBIAN_EPS;
        let mut minus = Vec6::zeros();
        minus[axis] = -POSE_PRIOR_JACOBIAN_EPS;
        let plus_center = camera_center_world(
            sensor_from_rig.compose(&apply_pose_delta_f64(rig_from_world, plus)),
        );
        let minus_center = camera_center_world(
            sensor_from_rig.compose(&apply_pose_delta_f64(rig_from_world, minus)),
        );
        jacobian.set_column(
            axis,
            &((plus_center - minus_center) / (2.0 * POSE_PRIOR_JACOBIAN_EPS)),
        );
    }
    Some(jacobian)
}

fn sensor_camera_center_jacobian(
    reconstruction: &Reconstruction,
    image: usize,
) -> Option<crate::ba::pose_prior::Mat3x6> {
    let frame_idx = reconstruction.frame_index_for_image(image)?;
    let frame = reconstruction.frames.get(frame_idx)?;
    let rig_from_world = frame.rig_from_world.to_se3();
    let sensor_from_rig = frame_sensor_from_rig(reconstruction, frame_idx, image)?;
    let mut jacobian = crate::ba::pose_prior::Mat3x6::zeros();
    for axis in 0..6 {
        let mut plus = Vec6::zeros();
        plus[axis] = POSE_PRIOR_JACOBIAN_EPS;
        let mut minus = Vec6::zeros();
        minus[axis] = -POSE_PRIOR_JACOBIAN_EPS;
        let plus_center = camera_center_world(
            apply_pose_delta_f64(sensor_from_rig, plus).compose(&rig_from_world),
        );
        let minus_center = camera_center_world(
            apply_pose_delta_f64(sensor_from_rig, minus).compose(&rig_from_world),
        );
        jacobian.set_column(
            axis,
            &((plus_center - minus_center) / (2.0 * POSE_PRIOR_JACOBIAN_EPS)),
        );
    }
    Some(jacobian)
}

pub(crate) fn analytic_frame_pose_jacobian(
    camera: CameraModel,
    sensor_from_rig: SE3,
    rig_from_world: SE3,
    point: [f32; 3],
) -> Option<Mat2x6> {
    let image_pose = sensor_from_rig.compose(&rig_from_world);
    let (j_image_pose, _) = analytic_projection_jacobians(camera, image_pose, point)?;
    let r_sensor = mat3_from_pose_rotation(sensor_from_rig);
    let mut jacobian = Mat2x6::zeros();
    for row in 0..2 {
        for col in 0..3 {
            jacobian[(row, col)] = j_image_pose[(row, 0)] * r_sensor[(0, col)]
                + j_image_pose[(row, 1)] * r_sensor[(1, col)]
                + j_image_pose[(row, 2)] * r_sensor[(2, col)];
            jacobian[(row, col + 3)] = j_image_pose[(row, 3)] * r_sensor[(0, col)]
                + j_image_pose[(row, 4)] * r_sensor[(1, col)]
                + j_image_pose[(row, 5)] * r_sensor[(2, col)];
        }
    }
    Some(jacobian)
}

fn numerical_frame_pose_jacobian(
    camera: CameraModel,
    sensor_from_rig: SE3,
    rig_from_world: SE3,
    point: [f32; 3],
) -> Option<Mat2x6> {
    let mut jacobian = Mat2x6::zeros();
    let eps = [1.0e-4; 6];
    for axis in 0..6 {
        let mut plus = Vec6::zeros();
        plus[axis] = eps[axis];
        let mut minus = Vec6::zeros();
        minus[axis] = -eps[axis];
        let p_plus = project_point(
            camera,
            sensor_from_rig.compose(&apply_pose_delta_f64(rig_from_world, plus)),
            point,
        )?;
        let p_minus = project_point(
            camera,
            sensor_from_rig.compose(&apply_pose_delta_f64(rig_from_world, minus)),
            point,
        )?;
        jacobian[(0, axis)] = (p_plus[0] - p_minus[0]) / (2.0 * eps[axis]);
        jacobian[(1, axis)] = (p_plus[1] - p_minus[1]) / (2.0 * eps[axis]);
    }
    Some(jacobian)
}

pub(crate) fn analytic_sensor_pose_jacobian(
    camera: CameraModel,
    sensor_from_rig: SE3,
    rig_from_world: SE3,
    point: [f32; 3],
) -> Option<Mat2x6> {
    let image_pose = sensor_from_rig.compose(&rig_from_world);
    let (j_image_pose, _) = analytic_projection_jacobians(camera, image_pose, point)?;
    let r_sensor = mat3_from_pose_rotation(sensor_from_rig);
    let t_rig = vec3_from_pose_translation(rig_from_world);
    let dt_domega = -cross_matrix(&(r_sensor * t_rig));
    let mut jacobian = Mat2x6::zeros();
    for row in 0..2 {
        for col in 0..3 {
            jacobian[(row, col)] = j_image_pose[(row, col)]
                + j_image_pose[(row, 3)] * dt_domega[(0, col)]
                + j_image_pose[(row, 4)] * dt_domega[(1, col)]
                + j_image_pose[(row, 5)] * dt_domega[(2, col)];
            jacobian[(row, col + 3)] = j_image_pose[(row, col + 3)];
        }
    }
    Some(jacobian)
}

fn numerical_sensor_pose_jacobian(
    camera: CameraModel,
    sensor_from_rig: SE3,
    rig_from_world: SE3,
    point: [f32; 3],
) -> Option<Mat2x6> {
    let mut jacobian = Mat2x6::zeros();
    let eps = [1.0e-4; 6];
    for axis in 0..6 {
        let mut plus = Vec6::zeros();
        plus[axis] = eps[axis];
        let mut minus = Vec6::zeros();
        minus[axis] = -eps[axis];
        let p_plus = project_point(
            camera,
            apply_pose_delta_f64(sensor_from_rig, plus).compose(&rig_from_world),
            point,
        )?;
        let p_minus = project_point(
            camera,
            apply_pose_delta_f64(sensor_from_rig, minus).compose(&rig_from_world),
            point,
        )?;
        jacobian[(0, axis)] = (p_plus[0] - p_minus[0]) / (2.0 * eps[axis]);
        jacobian[(1, axis)] = (p_plus[1] - p_minus[1]) / (2.0 * eps[axis]);
    }
    Some(jacobian)
}

fn mat3_from_pose_rotation(pose: SE3) -> Mat3 {
    let rotation = pose.rotation_matrix();
    Mat3::from_row_slice(&[
        rotation[0][0] as f64,
        rotation[0][1] as f64,
        rotation[0][2] as f64,
        rotation[1][0] as f64,
        rotation[1][1] as f64,
        rotation[1][2] as f64,
        rotation[2][0] as f64,
        rotation[2][1] as f64,
        rotation[2][2] as f64,
    ])
}

fn vec3_from_pose_translation(pose: SE3) -> Vec3d {
    let translation = pose.translation();
    Vec3d::new(
        translation[0] as f64,
        translation[1] as f64,
        translation[2] as f64,
    )
}

fn cross_matrix(vector: &Vec3d) -> Mat3 {
    Mat3::new(
        0.0, -vector[2], vector[1], vector[2], 0.0, -vector[0], -vector[1], vector[0], 0.0,
    )
}

pub(crate) fn sync_pose_blocks_for_sensor_changes(
    reconstruction: &mut Reconstruction,
    pose_blocks: &PoseBlockSet,
    changed_sensors: &[SensorPoseKey],
) {
    let changed_sensors = changed_sensors.iter().collect::<HashSet<_>>();
    for block in &pose_blocks.blocks {
        let PoseBlockKind::Frame(frame_idx) = block.kind else {
            continue;
        };
        let frame_uses_changed_sensor = block
            .images
            .iter()
            .filter_map(|&image| frame_sensor_key_for_image(reconstruction, image))
            .any(|key| changed_sensors.contains(&key));
        if frame_uses_changed_sensor {
            let Some(frame) = reconstruction.frames.get(frame_idx) else {
                continue;
            };
            set_frame_pose_block(
                reconstruction,
                frame_idx,
                &block.images,
                frame.rig_from_world.to_se3(),
            );
        }
    }
}

pub(crate) fn projection_jacobians(
    camera: CameraModel,
    pose: SE3,
    point: [f32; 3],
) -> Option<(Mat2x6, Mat2x3)> {
    analytic_projection_jacobians(camera, pose, point).or_else(|| {
        Some((
            numerical_pose_jacobian(camera, pose, point)?,
            numerical_point_jacobian(camera, pose, point)?,
        ))
    })
}

fn residual_and_jacobians(
    camera: CameraModel,
    pose: SE3,
    point: [f32; 3],
    xy: [f64; 2],
) -> Option<(Vec2, Mat2x6, Mat2x3)> {
    let predicted = project_point(camera, pose, point)?;
    let residual = Vec2::new(predicted[0] - xy[0], predicted[1] - xy[1]);
    if !residual.iter().all(|v| v.is_finite()) {
        return None;
    }
    let (j_pose, j_point) = analytic_projection_jacobians(camera, pose, point).unwrap_or((
        numerical_pose_jacobian(camera, pose, point)?,
        numerical_point_jacobian(camera, pose, point)?,
    ));
    Some((residual, j_pose, j_point))
}

fn analytic_projection_jacobians(
    camera: CameraModel,
    pose: SE3,
    point: [f32; 3],
) -> Option<(Mat2x6, Mat2x3)> {
    let cam_point = pose.transform_point(&point);
    let x = cam_point[0] as f64;
    let y = cam_point[1] as f64;
    let z = cam_point[2] as f64;
    let j_cam = analytic_img_from_cam_jacobian(camera, x, y, z)?;
    if z <= f64::EPSILON || ![x, y, z].iter().all(|v| v.is_finite()) {
        return None;
    }

    let translation = pose.translation();
    let rx = x - translation[0] as f64;
    let ry = y - translation[1] as f64;
    let rz = z - translation[2] as f64;

    // COLMAP/Ceres stores pose as separate rotation and translation parameter
    // blocks: R <- delta_R * R, t <- t + delta_t.
    let dcam_dpose = [
        [0.0, rz, -ry, 1.0, 0.0, 0.0],
        [-rz, 0.0, rx, 0.0, 1.0, 0.0],
        [ry, -rx, 0.0, 0.0, 0.0, 1.0],
    ];
    let mut j_pose = Mat2x6::zeros();
    for col in 0..6 {
        j_pose[(0, col)] = j_cam[(0, 0)] * dcam_dpose[0][col]
            + j_cam[(0, 1)] * dcam_dpose[1][col]
            + j_cam[(0, 2)] * dcam_dpose[2][col];
        j_pose[(1, col)] = j_cam[(1, 0)] * dcam_dpose[0][col]
            + j_cam[(1, 1)] * dcam_dpose[1][col]
            + j_cam[(1, 2)] * dcam_dpose[2][col];
    }

    let rotation = pose.rotation_matrix();
    let mut j_point = Mat2x3::zeros();
    for col in 0..3 {
        j_point[(0, col)] = j_cam[(0, 0)] * rotation[0][col] as f64
            + j_cam[(0, 1)] * rotation[1][col] as f64
            + j_cam[(0, 2)] * rotation[2][col] as f64;
        j_point[(1, col)] = j_cam[(1, 0)] * rotation[0][col] as f64
            + j_cam[(1, 1)] * rotation[1][col] as f64
            + j_cam[(1, 2)] * rotation[2][col] as f64;
    }

    Some((j_pose, j_point))
}

pub(crate) fn analytic_img_from_cam_jacobian(
    camera: CameraModel,
    x: f64,
    y: f64,
    z: f64,
) -> Option<Mat2x3> {
    if z <= f64::EPSILON || ![x, y, z].iter().all(|v| v.is_finite()) {
        return None;
    }
    let inv_z = 1.0 / z;
    let inv_z2 = inv_z * inv_z;
    let u = x * inv_z;
    let v = y * inv_z;
    let du_dcam = [inv_z, 0.0, -x * inv_z2];
    let dv_dcam = [0.0, inv_z, -y * inv_z2];

    let mut j_norm = SMatrix::<f64, 2, 2>::zeros();
    match camera.model_id {
        COLMAP_SIMPLE_PINHOLE => {
            let f = camera.params_slice()[0];
            j_norm[(0, 0)] = f;
            j_norm[(1, 1)] = f;
        }
        COLMAP_PINHOLE => {
            j_norm[(0, 0)] = camera.params_slice()[0];
            j_norm[(1, 1)] = camera.params_slice()[1];
        }
        COLMAP_SIMPLE_RADIAL | COLMAP_RADIAL => {
            let f = camera.params_slice()[0];
            let k1 = camera.params_slice()[3];
            let k2 = if camera.model_id == COLMAP_RADIAL {
                camera.params_slice()[4]
            } else {
                0.0
            };
            let r2 = u * u + v * v;
            let radial = 1.0 + k1 * r2 + k2 * r2 * r2;
            let radial_derivative = k1 + 2.0 * k2 * r2;
            j_norm[(0, 0)] = f * (radial + 2.0 * u * u * radial_derivative);
            j_norm[(0, 1)] = f * (2.0 * u * v * radial_derivative);
            j_norm[(1, 0)] = f * (2.0 * u * v * radial_derivative);
            j_norm[(1, 1)] = f * (radial + 2.0 * v * v * radial_derivative);
        }
        COLMAP_OPENCV => {
            let fx = camera.params_slice()[0];
            let fy = camera.params_slice()[1];
            let k1 = camera.params_slice()[4];
            let k2 = camera.params_slice()[5];
            let p1 = camera.params_slice()[6];
            let p2 = camera.params_slice()[7];
            let u2 = u * u;
            let v2 = v * v;
            let r2 = u2 + v2;
            let radial = 1.0 + k1 * r2 + k2 * r2 * r2;
            let radial_derivative = k1 + 2.0 * k2 * r2;
            let dx_du = radial + 2.0 * u2 * radial_derivative + 2.0 * p1 * v + 6.0 * p2 * u;
            let dx_dv = 2.0 * u * v * radial_derivative + 2.0 * p1 * u + 2.0 * p2 * v;
            let dy_du = 2.0 * u * v * radial_derivative + 2.0 * p2 * v + 2.0 * p1 * u;
            let dy_dv = radial + 2.0 * v2 * radial_derivative + 6.0 * p1 * v + 2.0 * p2 * u;
            j_norm[(0, 0)] = fx * dx_du;
            j_norm[(0, 1)] = fx * dx_dv;
            j_norm[(1, 0)] = fy * dy_du;
            j_norm[(1, 1)] = fy * dy_dv;
        }
        COLMAP_FULL_OPENCV => {
            let fx = camera.params_slice()[0];
            let fy = camera.params_slice()[1];
            let k1 = camera.params_slice()[4];
            let k2 = camera.params_slice()[5];
            let p1 = camera.params_slice()[6];
            let p2 = camera.params_slice()[7];
            let k3 = camera.params_slice()[8];
            let k4 = camera.params_slice()[9];
            let k5 = camera.params_slice()[10];
            let k6 = camera.params_slice()[11];
            let u2 = u * u;
            let v2 = v * v;
            let terms = full_opencv_radial_terms(u, v, k1, k2, k3, k4, k5, k6)?;
            let dx_du =
                terms.radial + 2.0 * u2 * terms.radial_derivative + 2.0 * p1 * v + 6.0 * p2 * u;
            let dx_dv = 2.0 * u * v * terms.radial_derivative + 2.0 * p1 * u + 2.0 * p2 * v;
            let dy_du = 2.0 * u * v * terms.radial_derivative + 2.0 * p2 * v + 2.0 * p1 * u;
            let dy_dv =
                terms.radial + 2.0 * v2 * terms.radial_derivative + 6.0 * p1 * v + 2.0 * p2 * u;
            j_norm[(0, 0)] = fx * dx_du;
            j_norm[(0, 1)] = fx * dx_dv;
            j_norm[(1, 0)] = fy * dy_du;
            j_norm[(1, 1)] = fy * dy_dv;
        }
        COLMAP_FOV => {
            let fx = camera.params_slice()[0];
            let fy = camera.params_slice()[1];
            let terms = fov_distortion_terms(camera.params_slice()[4], u, v)?;
            j_norm[(0, 0)] = fx * (terms.factor + 2.0 * u * u * terms.factor_derivative_r2);
            j_norm[(0, 1)] = fx * (2.0 * u * v * terms.factor_derivative_r2);
            j_norm[(1, 0)] = fy * (2.0 * u * v * terms.factor_derivative_r2);
            j_norm[(1, 1)] = fy * (terms.factor + 2.0 * v * v * terms.factor_derivative_r2);
        }
        COLMAP_SIMPLE_FISHEYE | COLMAP_FISHEYE => {
            let fx = camera.params_slice()[0];
            let fy = if camera.model_id == COLMAP_SIMPLE_FISHEYE {
                camera.params_slice()[0]
            } else {
                camera.params_slice()[1]
            };
            let fisheye = fisheye_normal_terms(u, v)?;
            j_norm[(0, 0)] = fx * fisheye.jacobian[(0, 0)];
            j_norm[(0, 1)] = fx * fisheye.jacobian[(0, 1)];
            j_norm[(1, 0)] = fy * fisheye.jacobian[(1, 0)];
            j_norm[(1, 1)] = fy * fisheye.jacobian[(1, 1)];
        }
        COLMAP_SIMPLE_RADIAL_FISHEYE | COLMAP_RADIAL_FISHEYE | COLMAP_OPENCV_FISHEYE => {
            let fx = camera.params_slice()[0];
            let fy = match camera.model_id {
                COLMAP_OPENCV_FISHEYE => camera.params_slice()[1],
                _ => camera.params_slice()[0],
            };
            let fisheye = fisheye_normal_terms(u, v)?;
            let terms = fisheye_radial_terms(camera, fisheye.u, fisheye.v)?;
            let dx_duu = terms.radial + 2.0 * fisheye.u * fisheye.u * terms.radial_derivative;
            let dx_dvv = 2.0 * fisheye.u * fisheye.v * terms.radial_derivative;
            let dy_duu = dx_dvv;
            let dy_dvv = terms.radial + 2.0 * fisheye.v * fisheye.v * terms.radial_derivative;
            j_norm[(0, 0)] =
                fx * (dx_duu * fisheye.jacobian[(0, 0)] + dx_dvv * fisheye.jacobian[(1, 0)]);
            j_norm[(0, 1)] =
                fx * (dx_duu * fisheye.jacobian[(0, 1)] + dx_dvv * fisheye.jacobian[(1, 1)]);
            j_norm[(1, 0)] =
                fy * (dy_duu * fisheye.jacobian[(0, 0)] + dy_dvv * fisheye.jacobian[(1, 0)]);
            j_norm[(1, 1)] =
                fy * (dy_duu * fisheye.jacobian[(0, 1)] + dy_dvv * fisheye.jacobian[(1, 1)]);
        }
        COLMAP_THIN_PRISM_FISHEYE | COLMAP_RAD_TAN_THIN_PRISM_FISHEYE => {
            let fx = camera.params_slice()[0];
            let fy = camera.params_slice()[1];
            let fisheye = fisheye_normal_terms(u, v)?;
            let distortion = fisheye_distortion_terms(camera, fisheye.u, fisheye.v)?;
            let j_total = Mat2::identity() + distortion.jacobian;
            j_norm[(0, 0)] = fx
                * (j_total[(0, 0)] * fisheye.jacobian[(0, 0)]
                    + j_total[(0, 1)] * fisheye.jacobian[(1, 0)]);
            j_norm[(0, 1)] = fx
                * (j_total[(0, 0)] * fisheye.jacobian[(0, 1)]
                    + j_total[(0, 1)] * fisheye.jacobian[(1, 1)]);
            j_norm[(1, 0)] = fy
                * (j_total[(1, 0)] * fisheye.jacobian[(0, 0)]
                    + j_total[(1, 1)] * fisheye.jacobian[(1, 0)]);
            j_norm[(1, 1)] = fy
                * (j_total[(1, 0)] * fisheye.jacobian[(0, 1)]
                    + j_total[(1, 1)] * fisheye.jacobian[(1, 1)]);
        }
        COLMAP_SIMPLE_DIVISION | COLMAP_DIVISION => {
            let fx = camera.params_slice()[0];
            let fy = if camera.model_id == COLMAP_SIMPLE_DIVISION {
                camera.params_slice()[0]
            } else {
                camera.params_slice()[1]
            };
            let k = if camera.model_id == COLMAP_SIMPLE_DIVISION {
                camera.params_slice()[3]
            } else {
                camera.params_slice()[4]
            };
            let terms = division_projection_terms(x, y, z, k)?;
            return Some(Mat2x3::from_row_slice(&[
                fx * terms.j_cam[(0, 0)],
                fx * terms.j_cam[(0, 1)],
                fx * terms.j_cam[(0, 2)],
                fy * terms.j_cam[(1, 0)],
                fy * terms.j_cam[(1, 1)],
                fy * terms.j_cam[(1, 2)],
            ]));
        }
        COLMAP_EUCM => {
            let fx = camera.params_slice()[0];
            let fy = camera.params_slice()[1];
            let terms =
                eucm_projection_terms(x, y, z, camera.params_slice()[4], camera.params_slice()[5])?;
            return Some(Mat2x3::from_row_slice(&[
                fx * terms.j_cam[(0, 0)],
                fx * terms.j_cam[(0, 1)],
                fx * terms.j_cam[(0, 2)],
                fy * terms.j_cam[(1, 0)],
                fy * terms.j_cam[(1, 1)],
                fy * terms.j_cam[(1, 2)],
            ]));
        }
        _ => return None,
    }
    if !j_norm.iter().all(|value| value.is_finite()) {
        return None;
    }

    let mut j_cam = Mat2x3::zeros();
    for col in 0..3 {
        j_cam[(0, col)] = j_norm[(0, 0)] * du_dcam[col] + j_norm[(0, 1)] * dv_dcam[col];
        j_cam[(1, col)] = j_norm[(1, 0)] * du_dcam[col] + j_norm[(1, 1)] * dv_dcam[col];
    }
    Some(j_cam)
}

fn numerical_pose_jacobian(camera: CameraModel, pose: SE3, point: [f32; 3]) -> Option<Mat2x6> {
    let mut jacobian = Mat2x6::zeros();
    let eps = [1.0e-4; 6];
    for axis in 0..6 {
        let mut plus = Vec6::zeros();
        plus[axis] = eps[axis];
        let mut minus = Vec6::zeros();
        minus[axis] = -eps[axis];
        let p_plus = project_point(camera, apply_pose_delta_f64(pose, plus), point)?;
        let p_minus = project_point(camera, apply_pose_delta_f64(pose, minus), point)?;
        jacobian[(0, axis)] = (p_plus[0] - p_minus[0]) / (2.0 * eps[axis]);
        jacobian[(1, axis)] = (p_plus[1] - p_minus[1]) / (2.0 * eps[axis]);
    }
    Some(jacobian)
}

fn numerical_point_jacobian(camera: CameraModel, pose: SE3, point: [f32; 3]) -> Option<Mat2x3> {
    let mut jacobian = Mat2x3::zeros();
    let eps = 1.0e-4;
    for axis in 0..3 {
        let mut plus = point;
        let mut minus = point;
        plus[axis] += eps as f32;
        minus[axis] -= eps as f32;
        let p_plus = project_point(camera, pose, plus)?;
        let p_minus = project_point(camera, pose, minus)?;
        jacobian[(0, axis)] = (p_plus[0] - p_minus[0]) / (2.0 * eps);
        jacobian[(1, axis)] = (p_plus[1] - p_minus[1]) / (2.0 * eps);
    }
    Some(jacobian)
}

pub(crate) fn camera_param_jacobian(
    camera: CameraModel,
    param: usize,
    pose: SE3,
    point: [f32; 3],
) -> Option<Vec2> {
    analytic_camera_param_jacobian(camera, param, pose, point)
        .or_else(|| finite_difference_camera_param_jacobian(camera, param, pose, point))
}

fn analytic_camera_param_jacobian(
    camera: CameraModel,
    param: usize,
    pose: SE3,
    point: [f32; 3],
) -> Option<Vec2> {
    if param >= camera.num_params {
        return None;
    }
    let cam_point = pose.transform_point(&point);
    let x = cam_point[0] as f64;
    let y = cam_point[1] as f64;
    let z = cam_point[2] as f64;
    if z <= f64::EPSILON || ![x, y, z].iter().all(|v| v.is_finite()) {
        return None;
    }
    let nx = x / z;
    let ny = y / z;
    let r2 = nx * nx + ny * ny;
    match (camera.model_id, param) {
        (COLMAP_SIMPLE_PINHOLE, 0) => Some(Vec2::new(nx, ny)),
        (COLMAP_SIMPLE_PINHOLE, 1) => Some(Vec2::new(1.0, 0.0)),
        (COLMAP_SIMPLE_PINHOLE, 2) => Some(Vec2::new(0.0, 1.0)),
        (COLMAP_PINHOLE, 0) => Some(Vec2::new(nx, 0.0)),
        (COLMAP_PINHOLE, 1) => Some(Vec2::new(0.0, ny)),
        (COLMAP_PINHOLE, 2) => Some(Vec2::new(1.0, 0.0)),
        (COLMAP_PINHOLE, 3) => Some(Vec2::new(0.0, 1.0)),
        (COLMAP_SIMPLE_RADIAL, 0) => {
            let radial = 1.0 + camera.params_slice()[3] * r2;
            Some(Vec2::new(nx * radial, ny * radial))
        }
        (COLMAP_SIMPLE_RADIAL, 1) => Some(Vec2::new(1.0, 0.0)),
        (COLMAP_SIMPLE_RADIAL, 2) => Some(Vec2::new(0.0, 1.0)),
        (COLMAP_SIMPLE_RADIAL, 3) => {
            let f = camera.params_slice()[0];
            Some(Vec2::new(f * nx * r2, f * ny * r2))
        }
        (COLMAP_RADIAL, 0) => {
            let radial = 1.0 + camera.params_slice()[3] * r2 + camera.params_slice()[4] * r2 * r2;
            Some(Vec2::new(nx * radial, ny * radial))
        }
        (COLMAP_RADIAL, 1) => Some(Vec2::new(1.0, 0.0)),
        (COLMAP_RADIAL, 2) => Some(Vec2::new(0.0, 1.0)),
        (COLMAP_RADIAL, 3) => {
            let f = camera.params_slice()[0];
            Some(Vec2::new(f * nx * r2, f * ny * r2))
        }
        (COLMAP_RADIAL, 4) => {
            let f = camera.params_slice()[0];
            Some(Vec2::new(f * nx * r2 * r2, f * ny * r2 * r2))
        }
        (COLMAP_OPENCV, 0) => {
            let k1 = camera.params_slice()[4];
            let k2 = camera.params_slice()[5];
            let p1 = camera.params_slice()[6];
            let p2 = camera.params_slice()[7];
            let distorted = opencv_distorted_normal(nx, ny, k1, k2, p1, p2);
            Some(Vec2::new(distorted[0], 0.0))
        }
        (COLMAP_OPENCV, 1) => {
            let k1 = camera.params_slice()[4];
            let k2 = camera.params_slice()[5];
            let p1 = camera.params_slice()[6];
            let p2 = camera.params_slice()[7];
            let distorted = opencv_distorted_normal(nx, ny, k1, k2, p1, p2);
            Some(Vec2::new(0.0, distorted[1]))
        }
        (COLMAP_OPENCV, 2) => Some(Vec2::new(1.0, 0.0)),
        (COLMAP_OPENCV, 3) => Some(Vec2::new(0.0, 1.0)),
        (COLMAP_OPENCV, 4) => Some(Vec2::new(
            camera.params_slice()[0] * nx * r2,
            camera.params_slice()[1] * ny * r2,
        )),
        (COLMAP_OPENCV, 5) => Some(Vec2::new(
            camera.params_slice()[0] * nx * r2 * r2,
            camera.params_slice()[1] * ny * r2 * r2,
        )),
        (COLMAP_OPENCV, 6) => Some(Vec2::new(
            camera.params_slice()[0] * 2.0 * nx * ny,
            camera.params_slice()[1] * (r2 + 2.0 * ny * ny),
        )),
        (COLMAP_OPENCV, 7) => Some(Vec2::new(
            camera.params_slice()[0] * (r2 + 2.0 * nx * nx),
            camera.params_slice()[1] * 2.0 * nx * ny,
        )),
        (COLMAP_FOV, 0..=4) => {
            let terms = fov_distortion_terms(camera.params_slice()[4], nx, ny)?;
            match param {
                0 => Some(Vec2::new(terms.x, 0.0)),
                1 => Some(Vec2::new(0.0, terms.y)),
                2 => Some(Vec2::new(1.0, 0.0)),
                3 => Some(Vec2::new(0.0, 1.0)),
                4 => Some(Vec2::new(
                    camera.params_slice()[0] * nx * terms.factor_derivative_omega,
                    camera.params_slice()[1] * ny * terms.factor_derivative_omega,
                )),
                _ => None,
            }
        }
        (COLMAP_SIMPLE_FISHEYE, 0..=2) => {
            let fisheye = fisheye_normal_terms(nx, ny)?;
            match param {
                0 => Some(Vec2::new(fisheye.u, fisheye.v)),
                1 => Some(Vec2::new(1.0, 0.0)),
                2 => Some(Vec2::new(0.0, 1.0)),
                _ => None,
            }
        }
        (COLMAP_FISHEYE, 0..=3) => {
            let fisheye = fisheye_normal_terms(nx, ny)?;
            match param {
                0 => Some(Vec2::new(fisheye.u, 0.0)),
                1 => Some(Vec2::new(0.0, fisheye.v)),
                2 => Some(Vec2::new(1.0, 0.0)),
                3 => Some(Vec2::new(0.0, 1.0)),
                _ => None,
            }
        }
        (COLMAP_SIMPLE_RADIAL_FISHEYE, 0..=3) => {
            let fisheye = fisheye_normal_terms(nx, ny)?;
            let terms = fisheye_radial_terms(camera, fisheye.u, fisheye.v)?;
            let distorted_u = fisheye.u * terms.radial;
            let distorted_v = fisheye.v * terms.radial;
            match param {
                0 => Some(Vec2::new(distorted_u, distorted_v)),
                1 => Some(Vec2::new(1.0, 0.0)),
                2 => Some(Vec2::new(0.0, 1.0)),
                3 => Some(Vec2::new(
                    camera.params_slice()[0] * fisheye.u * terms.r2,
                    camera.params_slice()[0] * fisheye.v * terms.r2,
                )),
                _ => None,
            }
        }
        (COLMAP_RADIAL_FISHEYE, 0..=4) => {
            let fisheye = fisheye_normal_terms(nx, ny)?;
            let terms = fisheye_radial_terms(camera, fisheye.u, fisheye.v)?;
            let distorted_u = fisheye.u * terms.radial;
            let distorted_v = fisheye.v * terms.radial;
            match param {
                0 => Some(Vec2::new(distorted_u, distorted_v)),
                1 => Some(Vec2::new(1.0, 0.0)),
                2 => Some(Vec2::new(0.0, 1.0)),
                3 => Some(Vec2::new(
                    camera.params_slice()[0] * fisheye.u * terms.r2,
                    camera.params_slice()[0] * fisheye.v * terms.r2,
                )),
                4 => Some(Vec2::new(
                    camera.params_slice()[0] * fisheye.u * terms.r4,
                    camera.params_slice()[0] * fisheye.v * terms.r4,
                )),
                _ => None,
            }
        }
        (COLMAP_OPENCV_FISHEYE, 0..=7) => {
            let fisheye = fisheye_normal_terms(nx, ny)?;
            let terms = fisheye_radial_terms(camera, fisheye.u, fisheye.v)?;
            let distorted_u = fisheye.u * terms.radial;
            let distorted_v = fisheye.v * terms.radial;
            match param {
                0 => Some(Vec2::new(distorted_u, 0.0)),
                1 => Some(Vec2::new(0.0, distorted_v)),
                2 => Some(Vec2::new(1.0, 0.0)),
                3 => Some(Vec2::new(0.0, 1.0)),
                4 => Some(Vec2::new(
                    camera.params_slice()[0] * fisheye.u * terms.r2,
                    camera.params_slice()[1] * fisheye.v * terms.r2,
                )),
                5 => Some(Vec2::new(
                    camera.params_slice()[0] * fisheye.u * terms.r4,
                    camera.params_slice()[1] * fisheye.v * terms.r4,
                )),
                6 => Some(Vec2::new(
                    camera.params_slice()[0] * fisheye.u * terms.r6,
                    camera.params_slice()[1] * fisheye.v * terms.r6,
                )),
                7 => Some(Vec2::new(
                    camera.params_slice()[0] * fisheye.u * terms.r8,
                    camera.params_slice()[1] * fisheye.v * terms.r8,
                )),
                _ => None,
            }
        }
        (COLMAP_THIN_PRISM_FISHEYE, 0..=11) => {
            let fx = camera.params_slice()[0];
            let fy = camera.params_slice()[1];
            let fisheye = fisheye_normal_terms(nx, ny)?;
            let terms = fisheye_distortion_terms(camera, fisheye.u, fisheye.v)?;
            let r2 = fisheye.u * fisheye.u + fisheye.v * fisheye.v;
            let r4 = r2 * r2;
            let r6 = r4 * r2;
            let r8 = r4 * r4;
            match param {
                0 => Some(Vec2::new(terms.x, 0.0)),
                1 => Some(Vec2::new(0.0, terms.y)),
                2 => Some(Vec2::new(1.0, 0.0)),
                3 => Some(Vec2::new(0.0, 1.0)),
                4 => Some(Vec2::new(fx * fisheye.u * r2, fy * fisheye.v * r2)),
                5 => Some(Vec2::new(fx * fisheye.u * r4, fy * fisheye.v * r4)),
                6 => Some(Vec2::new(
                    fx * 2.0 * fisheye.u * fisheye.v,
                    fy * (r2 + 2.0 * fisheye.v * fisheye.v),
                )),
                7 => Some(Vec2::new(
                    fx * (r2 + 2.0 * fisheye.u * fisheye.u),
                    fy * 2.0 * fisheye.u * fisheye.v,
                )),
                8 => Some(Vec2::new(fx * fisheye.u * r6, fy * fisheye.v * r6)),
                9 => Some(Vec2::new(fx * fisheye.u * r8, fy * fisheye.v * r8)),
                10 => Some(Vec2::new(fx * r2, 0.0)),
                11 => Some(Vec2::new(0.0, fy * r2)),
                _ => None,
            }
        }
        (COLMAP_RAD_TAN_THIN_PRISM_FISHEYE, 0..=15) => {
            let fx = camera.params_slice()[0];
            let fy = camera.params_slice()[1];
            let fisheye = fisheye_normal_terms(nx, ny)?;
            let terms = fisheye_distortion_terms(camera, fisheye.u, fisheye.v)?;
            let theta2 = fisheye.u * fisheye.u + fisheye.v * fisheye.v;
            let mut th_radial = 1.0;
            let mut theta_power = 1.0;
            for coeff in &camera.params_slice()[4..10] {
                theta_power *= theta2;
                th_radial += coeff * theta_power;
            }
            let x_dist = th_radial * fisheye.u;
            let y_dist = th_radial * fisheye.v;
            let x2 = x_dist * x_dist;
            let y2 = y_dist * y_dist;
            let xy = x_dist * y_dist;
            let r2_dist = x2 + y2;
            let r4_dist = r2_dist * r2_dist;
            match param {
                0 => Some(Vec2::new(terms.x, 0.0)),
                1 => Some(Vec2::new(0.0, terms.y)),
                2 => Some(Vec2::new(1.0, 0.0)),
                3 => Some(Vec2::new(0.0, 1.0)),
                4..=9 => {
                    let intermediate =
                        rad_tan_thin_prism_intermediate_terms(camera, fisheye.u, fisheye.v)?;
                    let power = theta2.powi((param - 3) as i32);
                    let radial_du = intermediate.j_xy[(0, 0)] * fisheye.u * power
                        + intermediate.j_xy[(0, 1)] * fisheye.v * power;
                    let radial_dv = intermediate.j_xy[(1, 0)] * fisheye.u * power
                        + intermediate.j_xy[(1, 1)] * fisheye.v * power;
                    Some(Vec2::new(fx * radial_du, fy * radial_dv))
                }
                10 => Some(Vec2::new(fx * (r2_dist + 2.0 * x2), fy * 2.0 * xy)),
                11 => Some(Vec2::new(fx * 2.0 * xy, fy * (r2_dist + 2.0 * y2))),
                12 => Some(Vec2::new(fx * r2_dist, 0.0)),
                13 => Some(Vec2::new(fx * r4_dist, 0.0)),
                14 => Some(Vec2::new(0.0, fy * r2_dist)),
                15 => Some(Vec2::new(0.0, fy * r4_dist)),
                _ => None,
            }
        }
        (COLMAP_SIMPLE_DIVISION, 0..=3) => {
            let terms = division_projection_terms(x, y, z, camera.params_slice()[3])?;
            match param {
                0 => Some(Vec2::new(terms.x, terms.y)),
                1 => Some(Vec2::new(1.0, 0.0)),
                2 => Some(Vec2::new(0.0, 1.0)),
                3 => {
                    let q = x * x + y * y;
                    let dscale_dk = 4.0 * q / (terms.disc_sqrt * (z + terms.disc_sqrt).powi(2));
                    Some(Vec2::new(
                        camera.params_slice()[0] * x * dscale_dk,
                        camera.params_slice()[0] * y * dscale_dk,
                    ))
                }
                _ => None,
            }
        }
        (COLMAP_DIVISION, 0..=4) => {
            let terms = division_projection_terms(x, y, z, camera.params_slice()[4])?;
            match param {
                0 => Some(Vec2::new(terms.x, 0.0)),
                1 => Some(Vec2::new(0.0, terms.y)),
                2 => Some(Vec2::new(1.0, 0.0)),
                3 => Some(Vec2::new(0.0, 1.0)),
                4 => {
                    let q = x * x + y * y;
                    let dscale_dk = 4.0 * q / (terms.disc_sqrt * (z + terms.disc_sqrt).powi(2));
                    Some(Vec2::new(
                        camera.params_slice()[0] * x * dscale_dk,
                        camera.params_slice()[1] * y * dscale_dk,
                    ))
                }
                _ => None,
            }
        }
        (COLMAP_EUCM, 0..=5) => {
            let alpha = camera.params_slice()[4];
            let beta = camera.params_slice()[5];
            let terms = eucm_projection_terms(x, y, z, alpha, beta)?;
            match param {
                0 => Some(Vec2::new(terms.x, 0.0)),
                1 => Some(Vec2::new(0.0, terms.y)),
                2 => Some(Vec2::new(1.0, 0.0)),
                3 => Some(Vec2::new(0.0, 1.0)),
                4 => {
                    let dden_dalpha = terms.rho - z;
                    Some(Vec2::new(
                        -camera.params_slice()[0] * x * dden_dalpha / (terms.den * terms.den),
                        -camera.params_slice()[1] * y * dden_dalpha / (terms.den * terms.den),
                    ))
                }
                5 => {
                    let q = x * x + y * y;
                    let dden_dbeta = alpha * q / (2.0 * terms.rho);
                    Some(Vec2::new(
                        -camera.params_slice()[0] * x * dden_dbeta / (terms.den * terms.den),
                        -camera.params_slice()[1] * y * dden_dbeta / (terms.den * terms.den),
                    ))
                }
                _ => None,
            }
        }
        (COLMAP_FULL_OPENCV, 0..=11) => {
            let fx = camera.params_slice()[0];
            let fy = camera.params_slice()[1];
            let k1 = camera.params_slice()[4];
            let k2 = camera.params_slice()[5];
            let p1 = camera.params_slice()[6];
            let p2 = camera.params_slice()[7];
            let k3 = camera.params_slice()[8];
            let k4 = camera.params_slice()[9];
            let k5 = camera.params_slice()[10];
            let k6 = camera.params_slice()[11];
            let terms = full_opencv_radial_terms(nx, ny, k1, k2, k3, k4, k5, k6)?;
            let distorted = opencv_distorted_normal_from_radial(nx, ny, terms.radial, p1, p2);
            match param {
                0 => Some(Vec2::new(distorted[0], 0.0)),
                1 => Some(Vec2::new(0.0, distorted[1])),
                2 => Some(Vec2::new(1.0, 0.0)),
                3 => Some(Vec2::new(0.0, 1.0)),
                4 => Some(Vec2::new(
                    fx * nx * terms.r2 / terms.den,
                    fy * ny * terms.r2 / terms.den,
                )),
                5 => Some(Vec2::new(
                    fx * nx * terms.r4 / terms.den,
                    fy * ny * terms.r4 / terms.den,
                )),
                6 => Some(Vec2::new(fx * 2.0 * nx * ny, fy * (r2 + 2.0 * ny * ny))),
                7 => Some(Vec2::new(fx * (r2 + 2.0 * nx * nx), fy * 2.0 * nx * ny)),
                8 => Some(Vec2::new(
                    fx * nx * terms.r6 / terms.den,
                    fy * ny * terms.r6 / terms.den,
                )),
                9 => {
                    let scale = -terms.num * terms.r2 / (terms.den * terms.den);
                    Some(Vec2::new(fx * nx * scale, fy * ny * scale))
                }
                10 => {
                    let scale = -terms.num * terms.r4 / (terms.den * terms.den);
                    Some(Vec2::new(fx * nx * scale, fy * ny * scale))
                }
                11 => {
                    let scale = -terms.num * terms.r6 / (terms.den * terms.den);
                    Some(Vec2::new(fx * nx * scale, fy * ny * scale))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn opencv_distorted_normal(u: f64, v: f64, k1: f64, k2: f64, p1: f64, p2: f64) -> [f64; 2] {
    let r2 = u * u + v * v;
    let radial = 1.0 + k1 * r2 + k2 * r2 * r2;
    opencv_distorted_normal_from_radial(u, v, radial, p1, p2)
}

fn opencv_distorted_normal_from_radial(u: f64, v: f64, radial: f64, p1: f64, p2: f64) -> [f64; 2] {
    let u2 = u * u;
    let uv = u * v;
    let v2 = v * v;
    let r2 = u2 + v2;
    [
        u * radial + 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2),
        v * radial + 2.0 * p2 * uv + p1 * (r2 + 2.0 * v2),
    ]
}

struct FullOpenCvRadialTerms {
    r2: f64,
    r4: f64,
    r6: f64,
    num: f64,
    den: f64,
    radial: f64,
    radial_derivative: f64,
}

fn full_opencv_radial_terms(
    u: f64,
    v: f64,
    k1: f64,
    k2: f64,
    k3: f64,
    k4: f64,
    k5: f64,
    k6: f64,
) -> Option<FullOpenCvRadialTerms> {
    let r2 = u * u + v * v;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let num = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
    let den = 1.0 + k4 * r2 + k5 * r4 + k6 * r6;
    if den.abs() <= f64::EPSILON {
        return None;
    }
    let radial = num / den;
    let dnum_dr2 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;
    let dden_dr2 = k4 + 2.0 * k5 * r2 + 3.0 * k6 * r4;
    let radial_derivative = (dnum_dr2 * den - num * dden_dr2) / (den * den);
    let terms = FullOpenCvRadialTerms {
        r2,
        r4,
        r6,
        num,
        den,
        radial,
        radial_derivative,
    };
    if [
        terms.r2,
        terms.r4,
        terms.r6,
        terms.num,
        terms.den,
        terms.radial,
        terms.radial_derivative,
    ]
    .iter()
    .all(|value| value.is_finite())
    {
        Some(terms)
    } else {
        None
    }
}

struct FisheyeNormalTerms {
    u: f64,
    v: f64,
    jacobian: Mat2,
}

fn fisheye_normal_terms(u: f64, v: f64) -> Option<FisheyeNormalTerms> {
    let r2 = u * u + v * v;
    let r = r2.sqrt();
    let mut uu = u;
    let mut vv = v;
    let mut jacobian = Mat2::identity();
    if r > f64::EPSILON {
        let theta = r.atan();
        let scale = theta / r;
        uu *= scale;
        vv *= scale;
        let dscale_dr = (r / (1.0 + r2) - theta) / r2;
        let dscale_du = dscale_dr * u / r;
        let dscale_dv = dscale_dr * v / r;
        jacobian[(0, 0)] = scale + u * dscale_du;
        jacobian[(0, 1)] = u * dscale_dv;
        jacobian[(1, 0)] = v * dscale_du;
        jacobian[(1, 1)] = scale + v * dscale_dv;
    }
    if [uu, vv].iter().all(|value| value.is_finite())
        && jacobian.iter().all(|value| value.is_finite())
    {
        Some(FisheyeNormalTerms {
            u: uu,
            v: vv,
            jacobian,
        })
    } else {
        None
    }
}

struct FisheyeRadialTerms {
    r2: f64,
    r4: f64,
    r6: f64,
    r8: f64,
    radial: f64,
    radial_derivative: f64,
}

fn fisheye_radial_terms(camera: CameraModel, u: f64, v: f64) -> Option<FisheyeRadialTerms> {
    let r2 = u * u + v * v;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let r8 = r4 * r4;
    let (k1, k2, k3, k4) = match camera.model_id {
        COLMAP_SIMPLE_RADIAL_FISHEYE => (camera.params_slice()[3], 0.0, 0.0, 0.0),
        COLMAP_RADIAL_FISHEYE => (camera.params_slice()[3], camera.params_slice()[4], 0.0, 0.0),
        COLMAP_OPENCV_FISHEYE => (
            camera.params_slice()[4],
            camera.params_slice()[5],
            camera.params_slice()[6],
            camera.params_slice()[7],
        ),
        _ => return None,
    };
    let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
    let radial_derivative = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4 + 4.0 * k4 * r6;
    let terms = FisheyeRadialTerms {
        r2,
        r4,
        r6,
        r8,
        radial,
        radial_derivative,
    };
    if [
        terms.r2,
        terms.r4,
        terms.r6,
        terms.r8,
        terms.radial,
        terms.radial_derivative,
    ]
    .iter()
    .all(|value| value.is_finite())
    {
        Some(terms)
    } else {
        None
    }
}

struct FovDistortionTerms {
    x: f64,
    y: f64,
    factor: f64,
    factor_derivative_r2: f64,
    factor_derivative_omega: f64,
}

fn fov_distortion_terms(omega: f64, u: f64, v: f64) -> Option<FovDistortionTerms> {
    const EPSILON: f64 = 1.0e-4;
    let r2 = u * u + v * v;
    let omega2 = omega * omega;
    let (factor, factor_derivative_r2, factor_derivative_omega) = if omega2 < EPSILON {
        (
            omega2 * r2 / 3.0 - omega2 / 12.0 + 1.0,
            omega2 / 3.0,
            2.0 * omega * (r2 / 3.0 - 1.0 / 12.0),
        )
    } else if r2 < EPSILON {
        let t = (omega / 2.0).tan();
        let dt_domega = 0.5 * (1.0 + t * t);
        let factor = -2.0 * t * (4.0 * r2 * t * t - 3.0) / (3.0 * omega);
        let factor_derivative_r2 = -8.0 * t * t * t / (3.0 * omega);
        let numerator = -8.0 * r2 * t * t * t + 6.0 * t;
        let numerator_derivative = dt_domega * (6.0 - 24.0 * r2 * t * t);
        let factor_derivative_omega =
            (numerator_derivative * omega - numerator) / (3.0 * omega * omega);
        (factor, factor_derivative_r2, factor_derivative_omega)
    } else {
        let r = r2.sqrt();
        let t = (omega / 2.0).tan();
        let a = 2.0 * r * t;
        let numerator = a.atan();
        let den = r * omega;
        let factor = numerator / den;
        let dnum_dr = 2.0 * t / (1.0 + a * a);
        let dfactor_dr = (dnum_dr * den - numerator * omega) / (den * den);
        let factor_derivative_r2 = dfactor_dr / (2.0 * r);
        let da_domega = r * (1.0 + t * t);
        let dnum_domega = da_domega / (1.0 + a * a);
        let factor_derivative_omega = (dnum_domega * den - numerator * r) / (den * den);
        (factor, factor_derivative_r2, factor_derivative_omega)
    };
    let terms = FovDistortionTerms {
        x: u * factor,
        y: v * factor,
        factor,
        factor_derivative_r2,
        factor_derivative_omega,
    };
    if [
        terms.x,
        terms.y,
        terms.factor,
        terms.factor_derivative_r2,
        terms.factor_derivative_omega,
    ]
    .iter()
    .all(|value| value.is_finite())
    {
        Some(terms)
    } else {
        None
    }
}

struct FisheyeDistortionTerms {
    x: f64,
    y: f64,
    dx: f64,
    dy: f64,
    jacobian: Mat2,
}

fn fisheye_distortion_terms(camera: CameraModel, u: f64, v: f64) -> Option<FisheyeDistortionTerms> {
    match camera.model_id {
        COLMAP_THIN_PRISM_FISHEYE => thin_prism_fisheye_distortion_terms(camera, u, v),
        COLMAP_RAD_TAN_THIN_PRISM_FISHEYE => {
            rad_tan_thin_prism_fisheye_distortion_terms(camera, u, v)
        }
        _ => None,
    }
}

fn thin_prism_fisheye_distortion_terms(
    camera: CameraModel,
    u: f64,
    v: f64,
) -> Option<FisheyeDistortionTerms> {
    let k1 = camera.params_slice()[4];
    let k2 = camera.params_slice()[5];
    let p1 = camera.params_slice()[6];
    let p2 = camera.params_slice()[7];
    let k3 = camera.params_slice()[8];
    let k4 = camera.params_slice()[9];
    let sx1 = camera.params_slice()[10];
    let sy1 = camera.params_slice()[11];
    let r2 = u * u + v * v;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let r8 = r4 * r4;
    let radial_offset = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
    let radial_derivative = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4 + 4.0 * k4 * r6;
    let dx = u * radial_offset + 2.0 * p1 * u * v + p2 * (r2 + 2.0 * u * u) + sx1 * r2;
    let dy = v * radial_offset + 2.0 * p2 * u * v + p1 * (r2 + 2.0 * v * v) + sy1 * r2;
    let mut jacobian =
        radial_tangential_offset_jacobian(u, v, radial_offset, radial_derivative, p1, p2);
    jacobian[(0, 0)] += 2.0 * sx1 * u;
    jacobian[(0, 1)] += 2.0 * sx1 * v;
    jacobian[(1, 0)] += 2.0 * sy1 * u;
    jacobian[(1, 1)] += 2.0 * sy1 * v;
    finite_fisheye_distortion_terms(u, v, dx, dy, jacobian)
}

fn rad_tan_thin_prism_fisheye_distortion_terms(
    camera: CameraModel,
    u: f64,
    v: f64,
) -> Option<FisheyeDistortionTerms> {
    let p0 = camera.params_slice()[10];
    let p1 = camera.params_slice()[11];
    let s0 = camera.params_slice()[12];
    let s1 = camera.params_slice()[13];
    let s2 = camera.params_slice()[14];
    let s3 = camera.params_slice()[15];
    let theta2 = u * u + v * v;
    let mut th_radial = 1.0;
    let mut th_radial_derivative = 0.0;
    let mut theta_power = 1.0;
    for (idx, coeff) in camera.params_slice()[4..10].iter().enumerate() {
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
    let xy = x * y;
    let r2 = x2 + y2;
    let r4 = r2 * r2;
    let dx_tang = 2.0 * p1 * xy + p0 * (r2 + 2.0 * x2);
    let dy_tang = 2.0 * p0 * xy + p1 * (r2 + 2.0 * y2);
    let dx_tp = s0 * r2 + s1 * r4;
    let dy_tp = s2 * r2 + s3 * r4;
    let dx = x + dx_tang + dx_tp - u;
    let dy = y + dy_tang + dy_tp - v;

    let dtx_dx = 2.0 * p1 * y + 6.0 * p0 * x + 2.0 * s0 * x + 4.0 * s1 * r2 * x;
    let dtx_dy = 2.0 * p1 * x + 2.0 * p0 * y + 2.0 * s0 * y + 4.0 * s1 * r2 * y;
    let dty_dx = 2.0 * p0 * y + 2.0 * p1 * x + 2.0 * s2 * x + 4.0 * s3 * r2 * x;
    let dty_dy = 2.0 * p0 * x + 6.0 * p1 * y + 2.0 * s2 * y + 4.0 * s3 * r2 * y;
    let jacobian = Mat2::from_row_slice(&[
        (1.0 + dtx_dx) * dx_du + dtx_dy * dy_du - 1.0,
        (1.0 + dtx_dx) * dx_dv + dtx_dy * dy_dv,
        dty_dx * dx_du + (1.0 + dty_dy) * dy_du,
        dty_dx * dx_dv + (1.0 + dty_dy) * dy_dv - 1.0,
    ]);
    finite_fisheye_distortion_terms(u, v, dx, dy, jacobian)
}

struct RadTanThinPrismIntermediateTerms {
    x: f64,
    y: f64,
    r2: f64,
    r4: f64,
    j_xy: Mat2,
}

fn rad_tan_thin_prism_intermediate_terms(
    camera: CameraModel,
    u: f64,
    v: f64,
) -> Option<RadTanThinPrismIntermediateTerms> {
    let p0 = camera.params_slice()[10];
    let p1 = camera.params_slice()[11];
    let s0 = camera.params_slice()[12];
    let s1 = camera.params_slice()[13];
    let s2 = camera.params_slice()[14];
    let s3 = camera.params_slice()[15];
    let theta2 = u * u + v * v;
    let mut th_radial = 1.0;
    let mut theta_power = 1.0;
    for coeff in &camera.params_slice()[4..10] {
        theta_power *= theta2;
        th_radial += coeff * theta_power;
    }
    let x = th_radial * u;
    let y = th_radial * v;
    let r2 = x * x + y * y;
    let r4 = r2 * r2;
    let dtx_dx = 2.0 * p1 * y + 6.0 * p0 * x + 2.0 * s0 * x + 4.0 * s1 * r2 * x;
    let dtx_dy = 2.0 * p1 * x + 2.0 * p0 * y + 2.0 * s0 * y + 4.0 * s1 * r2 * y;
    let dty_dx = 2.0 * p0 * y + 2.0 * p1 * x + 2.0 * s2 * x + 4.0 * s3 * r2 * x;
    let dty_dy = 2.0 * p0 * x + 6.0 * p1 * y + 2.0 * s2 * y + 4.0 * s3 * r2 * y;
    let terms = RadTanThinPrismIntermediateTerms {
        x,
        y,
        r2,
        r4,
        j_xy: Mat2::from_row_slice(&[1.0 + dtx_dx, dtx_dy, dty_dx, 1.0 + dty_dy]),
    };
    if [terms.x, terms.y, terms.r2, terms.r4]
        .iter()
        .all(|value| value.is_finite())
        && terms.j_xy.iter().all(|value| value.is_finite())
    {
        Some(terms)
    } else {
        None
    }
}

fn finite_fisheye_distortion_terms(
    u: f64,
    v: f64,
    dx: f64,
    dy: f64,
    jacobian: Mat2,
) -> Option<FisheyeDistortionTerms> {
    let terms = FisheyeDistortionTerms {
        x: u + dx,
        y: v + dy,
        dx,
        dy,
        jacobian,
    };
    if [terms.x, terms.y, terms.dx, terms.dy]
        .iter()
        .all(|value| value.is_finite())
        && terms.jacobian.iter().all(|value| value.is_finite())
    {
        Some(terms)
    } else {
        None
    }
}

fn radial_tangential_offset_jacobian(
    u: f64,
    v: f64,
    radial: f64,
    radial_derivative: f64,
    p1: f64,
    p2: f64,
) -> Mat2 {
    Mat2::from_row_slice(&[
        radial + 2.0 * u * u * radial_derivative + 2.0 * p1 * v + 6.0 * p2 * u,
        2.0 * u * v * radial_derivative + 2.0 * p1 * u + 2.0 * p2 * v,
        2.0 * u * v * radial_derivative + 2.0 * p2 * v + 2.0 * p1 * u,
        radial + 2.0 * v * v * radial_derivative + 6.0 * p1 * v + 2.0 * p2 * u,
    ])
}

struct DivisionProjectionTerms {
    x: f64,
    y: f64,
    scale: f64,
    disc_sqrt: f64,
    j_cam: Mat2x3,
}

fn division_projection_terms(x: f64, y: f64, z: f64, k: f64) -> Option<DivisionProjectionTerms> {
    let q = x * x + y * y;
    let disc_sq = z * z - 4.0 * k * q;
    if disc_sq < 0.0 {
        return None;
    }
    let disc_sqrt = disc_sq.sqrt();
    let den = z + disc_sqrt;
    if den.abs() <= f64::EPSILON {
        return None;
    }
    let scale = 2.0 / den;
    let den_derivative_x = -4.0 * k * x / disc_sqrt;
    let den_derivative_y = -4.0 * k * y / disc_sqrt;
    let den_derivative_z = 1.0 + z / disc_sqrt;
    let scale_derivative_x = -2.0 * den_derivative_x / (den * den);
    let scale_derivative_y = -2.0 * den_derivative_y / (den * den);
    let scale_derivative_z = -2.0 * den_derivative_z / (den * den);
    let j_cam = Mat2x3::from_row_slice(&[
        scale + x * scale_derivative_x,
        x * scale_derivative_y,
        x * scale_derivative_z,
        y * scale_derivative_x,
        scale + y * scale_derivative_y,
        y * scale_derivative_z,
    ]);
    let terms = DivisionProjectionTerms {
        x: scale * x,
        y: scale * y,
        scale,
        disc_sqrt,
        j_cam,
    };
    if [terms.x, terms.y, terms.scale, terms.disc_sqrt]
        .iter()
        .all(|value| value.is_finite())
        && terms.j_cam.iter().all(|value| value.is_finite())
    {
        Some(terms)
    } else {
        None
    }
}

struct EucmProjectionTerms {
    x: f64,
    y: f64,
    rho: f64,
    den: f64,
    j_cam: Mat2x3,
}

fn eucm_projection_terms(
    x: f64,
    y: f64,
    z: f64,
    alpha: f64,
    beta: f64,
) -> Option<EucmProjectionTerms> {
    let q = x * x + y * y;
    let rho2 = beta * q + z * z;
    if rho2 < 0.0 {
        return None;
    }
    let rho = rho2.sqrt();
    let den = alpha * rho + (1.0 - alpha) * z;
    if den < f64::EPSILON {
        return None;
    }
    let inv_den = 1.0 / den;
    let dden_dx = alpha * beta * x / rho;
    let dden_dy = alpha * beta * y / rho;
    let dden_dz = alpha * z / rho + (1.0 - alpha);
    let inv_den2 = inv_den * inv_den;
    let j_cam = Mat2x3::from_row_slice(&[
        inv_den - x * dden_dx * inv_den2,
        -x * dden_dy * inv_den2,
        -x * dden_dz * inv_den2,
        -y * dden_dx * inv_den2,
        inv_den - y * dden_dy * inv_den2,
        -y * dden_dz * inv_den2,
    ]);
    let terms = EucmProjectionTerms {
        x: x * inv_den,
        y: y * inv_den,
        rho,
        den,
        j_cam,
    };
    if [terms.x, terms.y, terms.rho, terms.den]
        .iter()
        .all(|value| value.is_finite())
        && terms.j_cam.iter().all(|value| value.is_finite())
    {
        Some(terms)
    } else {
        None
    }
}

fn finite_difference_camera_param_jacobian(
    camera: CameraModel,
    param: usize,
    pose: SE3,
    point: [f32; 3],
) -> Option<Vec2> {
    if param >= camera.num_params {
        return None;
    }
    let current = camera.param(param)?;
    let eps = current.abs().max(1.0) * 1.0e-6;
    let mut plus = camera;
    let mut minus = camera;
    plus.set_param(param, current + eps).ok()?;
    minus.set_param(param, current - eps).ok()?;
    let p_plus = project_point(plus, pose, point)?;
    let p_minus = project_point(minus, pose, point)?;
    Some(Vec2::new(
        (p_plus[0] - p_minus[0]) / (2.0 * eps),
        (p_plus[1] - p_minus[1]) / (2.0 * eps),
    ))
}

fn mat2x6_to_dmatrix(matrix: Mat2x6) -> DMatrix<f64> {
    DMatrix::from_fn(2, 6, |row, col| matrix[(row, col)])
}

fn pose_block_jacobian(matrix: Mat2x6, block: &PoseBlock) -> Option<DMatrix<f64>> {
    let axes = pose_block_active_axes(block);
    if axes.is_empty() {
        return None;
    }
    Some(DMatrix::from_fn(2, axes.len(), |row, col| {
        matrix[(row, axes[col])]
    }))
}

fn vec2_to_dmatrix(vector: Vec2) -> DMatrix<f64> {
    DMatrix::from_column_slice(2, 1, &[vector[0], vector[1]])
}

fn point_nonpoint_cross(j_point: Mat2x3, j_nonpoint: &DMatrix<f64>) -> DMatrix<f64> {
    DMatrix::from_fn(3, j_nonpoint.ncols(), |row, col| {
        j_point[(0, row)] * j_nonpoint[(0, col)] + j_point[(1, row)] * j_nonpoint[(1, col)]
    })
}

pub(crate) fn sync_camera_intrinsics_from_params(camera: &mut CameraModel) {
    // Intrinsics are derived from params; keep the helper as a no-op call site
    // seam for BA code that previously mirrored fields.
    camera.sync_intrinsics_from_params();
}

pub(crate) fn apply_pose_delta_f64(pose: SE3, delta: Vec6) -> SE3 {
    let q = pose.quaternion();
    let base_rotation = crate::geometry::quat_from_xyzw(q[0], q[1], q[2], q[3]).normalize();
    let omega = Vec3::new(delta[0] as f32, delta[1] as f32, delta[2] as f32);
    let angle = omega.length();
    let delta_rotation = if angle > 1.0e-12 {
        crate::geometry::quat_from_axis_angle(omega / angle, angle)
    } else {
        Quat::identity()
    };
    let t = pose.translation();
    let translation = Vec3::new(
        t[0] + delta[3] as f32,
        t[1] + delta[4] as f32,
        t[2] + delta[5] as f32,
    );
    SE3::from_quat_translation((delta_rotation * base_rotation).normalize(), translation)
}
