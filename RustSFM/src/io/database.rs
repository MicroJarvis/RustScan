use crate::colmap::{
    ColmapCamera, ColmapDataId, ColmapRig, ColmapRigSensor, ColmapRigid3, ColmapSensorId,
    ColmapSensorType,
};
use crate::correspondence_graph::{
    image_pair_to_pair_id, pair_id_to_image_pair, should_swap_image_pair, CorrespondenceGraph,
    FeatureMatch, ImageId, ImagePairId, TwoViewGeometryRecord,
};
use crate::types::colmap_camera_model_num_params;
use anyhow::{bail, Context, Result};
use nalgebra::Matrix3;
use rand::seq::SliceRandom;
use rusqlite::{params, types::Type, Connection, OpenFlags, OptionalExtension, Row, Rows};
use rustslam::{Descriptors, KeyPoint};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

pub const COLMAP_TWO_VIEW_UNDEFINED: i32 = 0;
pub const COLMAP_TWO_VIEW_DEGENERATE: i32 = 1;
pub const COLMAP_TWO_VIEW_CALIBRATED: i32 = 2;
pub const COLMAP_TWO_VIEW_UNCALIBRATED: i32 = 3;
pub const COLMAP_TWO_VIEW_PLANAR: i32 = 4;
pub const COLMAP_TWO_VIEW_PANORAMIC: i32 = 5;
pub const COLMAP_TWO_VIEW_PLANAR_OR_PANORAMIC: i32 = 6;
pub const COLMAP_TWO_VIEW_WATERMARK: i32 = 7;
pub const COLMAP_TWO_VIEW_MULTIPLE: i32 = 8;
pub const COLMAP_TWO_VIEW_CALIBRATED_RIG: i32 = 9;

pub const COLMAP_FEATURE_UNDEFINED: i32 = -1;
pub const COLMAP_FEATURE_SIFT: i32 = 0;
pub const COLMAP_FEATURE_ALIKED_N16ROT: i32 = 1;
pub const COLMAP_FEATURE_ALIKED_N32: i32 = 2;

pub const COLMAP_DATABASE_VERSION_3_13_0_0: i32 = make_colmap_database_version_number(3, 13, 0, 0);
pub const COLMAP_DATABASE_VERSION_3_14_0_0: i32 = make_colmap_database_version_number(3, 14, 0, 0);
pub const COLMAP_DATABASE_VERSION_3_14_0_1: i32 = make_colmap_database_version_number(3, 14, 0, 1);
pub const COLMAP_CURRENT_DATABASE_VERSION: i32 = make_colmap_database_version_number(4, 1, 0, 0);

pub const fn make_colmap_database_version_number(
    major: i32,
    minor: i32,
    patch: i32,
    revision: i32,
) -> i32 {
    major * 1_000_000 + minor * 10_000 + patch * 100 + revision
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColmapKeypoint {
    pub x: f32,
    pub y: f32,
    pub a11: f32,
    pub a12: f32,
    pub a21: f32,
    pub a22: f32,
}

impl ColmapKeypoint {
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            a11: 1.0,
            a12: 0.0,
            a21: 0.0,
            a22: 1.0,
        }
    }

    pub fn from_scale_orientation(x: f32, y: f32, scale: f32, orientation: f32) -> Self {
        Self {
            x,
            y,
            a11: scale,
            a12: orientation,
            a21: 0.0,
            a22: 0.0,
        }
    }

    pub fn to_keypoint(self) -> KeyPoint {
        KeyPoint::new(self.x, self.y)
    }
}

impl From<&KeyPoint> for ColmapKeypoint {
    fn from(value: &KeyPoint) -> Self {
        Self::from_scale_orientation(value.x(), value.y(), value.size, value.angle)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColmapKeypointsBlob {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<u8>,
}

impl ColmapKeypointsBlob {
    pub fn new(rows: usize, cols: usize, data: Vec<u8>) -> Result<Self> {
        validate_dynamic_blob("keypoints", rows, cols, std::mem::size_of::<f32>(), &data)?;
        Ok(Self { rows, cols, data })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColmapDescriptors {
    pub feature_type: i32,
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<u8>,
}

impl ColmapDescriptors {
    pub fn new(feature_type: i32, rows: usize, cols: usize, data: Vec<u8>) -> Result<Self> {
        if data.len() != rows.saturating_mul(cols) {
            bail!(
                "descriptor blob has {} bytes, expected rows*cols={}*{}",
                data.len(),
                rows,
                cols
            );
        }
        Ok(Self {
            feature_type,
            rows,
            cols,
            data,
        })
    }

    pub fn from_rustslam(feature_type: i32, descriptors: &Descriptors) -> Self {
        Self {
            feature_type,
            rows: descriptors.count,
            cols: descriptors.size,
            data: descriptors.data.clone(),
        }
    }

    pub fn to_rustslam(&self) -> Descriptors {
        Descriptors {
            data: self.data.clone(),
            size: self.cols,
            count: self.rows,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColmapDescriptorsFloat {
    pub feature_type: i32,
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl ColmapDescriptorsFloat {
    pub fn empty() -> Self {
        Self {
            feature_type: COLMAP_FEATURE_UNDEFINED,
            rows: 0,
            cols: 0,
            data: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColmapMatchesBlob {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<u8>,
}

impl ColmapMatchesBlob {
    pub fn new(rows: usize, cols: usize, data: Vec<u8>) -> Result<Self> {
        if cols != 2 {
            bail!("COLMAP match blob has unsupported column count {cols}");
        }
        validate_dynamic_blob("matches", rows, cols, std::mem::size_of::<u32>(), &data)?;
        Ok(Self { rows, cols, data })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColmapTwoViewGeometry {
    pub config: i32,
    pub inlier_matches: Vec<FeatureMatch>,
    pub f_matrix: Option<[f64; 9]>,
    pub e_matrix: Option<[f64; 9]>,
    pub h_matrix: Option<[f64; 9]>,
    pub qvec: Option<[f64; 4]>,
    pub tvec: Option<[f64; 3]>,
}

impl Default for ColmapTwoViewGeometry {
    fn default() -> Self {
        Self {
            config: COLMAP_TWO_VIEW_UNDEFINED,
            inlier_matches: Vec::new(),
            f_matrix: None,
            e_matrix: None,
            h_matrix: None,
            qvec: None,
            tvec: None,
        }
    }
}

impl ColmapTwoViewGeometry {
    pub fn invert(&mut self) {
        self.f_matrix = self.f_matrix.map(transpose3);
        self.e_matrix = self.e_matrix.map(transpose3);
        self.h_matrix = self.h_matrix.and_then(invert_matrix3);
        if let (Some(qvec), Some(tvec)) = (self.qvec, self.tvec) {
            let rotation = glam::DQuat::from_xyzw(qvec[1], qvec[2], qvec[3], qvec[0]).normalize();
            let translation = glam::DVec3::from_array(tvec);
            let inverse_rotation = rotation.inverse();
            let inverse_translation = -(inverse_rotation * translation);
            self.qvec = Some([
                inverse_rotation.w,
                inverse_rotation.x,
                inverse_rotation.y,
                inverse_rotation.z,
            ]);
            self.tvec = Some(inverse_translation.to_array());
        }
        for match_ in &mut self.inlier_matches {
            std::mem::swap(&mut match_.point2d_idx1, &mut match_.point2d_idx2);
        }
    }
}

pub struct ColmapDatabase {
    conn: Connection,
    database_entry_deleted: Cell<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColmapDatabaseCamera {
    pub camera: ColmapCamera,
    pub has_prior_focal_length: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColmapDatabaseImage {
    pub image_id: ImageId,
    pub name: String,
    pub camera_id: u32,
    pub frame_id: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColmapDatabaseFrame {
    pub frame_id: u32,
    pub rig_id: u32,
    pub data_ids: Vec<ColmapDataId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColmapPosePriorCoordinateSystem {
    Undefined,
    Wgs84,
    Cartesian,
    Other(i64),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColmapPosePrior {
    pub pose_prior_id: u32,
    pub corr_data_id: ColmapDataId,
    pub position: [f64; 3],
    pub position_covariance: [f64; 9],
    pub coordinate_system: ColmapPosePriorCoordinateSystem,
    pub gravity: [f64; 3],
}

#[derive(Debug, Clone)]
pub struct DatabaseCacheOptions {
    pub min_num_matches: usize,
    pub ignore_watermarks: bool,
    pub image_names: BTreeSet<String>,
    pub load_all_images: bool,
    pub convert_pose_priors_to_enu: bool,
}

impl Default for DatabaseCacheOptions {
    fn default() -> Self {
        Self {
            min_num_matches: 0,
            ignore_watermarks: false,
            image_names: BTreeSet::new(),
            load_all_images: false,
            convert_pose_priors_to_enu: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DatabaseCache {
    pub rigs: BTreeMap<u32, ColmapRig>,
    pub cameras: BTreeMap<u32, ColmapDatabaseCamera>,
    pub frames: BTreeMap<u32, ColmapDatabaseFrame>,
    pub images: BTreeMap<ImageId, ColmapDatabaseImage>,
    pub pose_priors: Vec<ColmapPosePrior>,
    pub correspondence_graph: CorrespondenceGraph,
}

impl DatabaseCache {
    pub fn new() -> Self {
        Self {
            rigs: BTreeMap::new(),
            cameras: BTreeMap::new(),
            frames: BTreeMap::new(),
            images: BTreeMap::new(),
            pose_priors: Vec::new(),
            correspondence_graph: CorrespondenceGraph::new(),
        }
    }

    pub fn create_from_cache(
        database_cache: &DatabaseCache,
        options: &DatabaseCacheOptions,
    ) -> Result<Self> {
        let candidate_image_ids = database_cache
            .images
            .iter()
            .filter(|(_, image)| {
                options.image_names.is_empty() || options.image_names.contains(&image.name)
            })
            .map(|(&image_id, _)| image_id)
            .collect::<BTreeSet<_>>();

        let mut connected_image_ids = BTreeSet::new();
        if !options.load_all_images {
            for (pair_id, num_matches) in database_cache
                .correspondence_graph
                .num_matches_between_all_images()
            {
                let (image_id1, image_id2) =
                    pair_id_to_image_pair(pair_id).map_err(|err| anyhow::anyhow!("{err:?}"))?;
                if !candidate_image_ids.contains(&image_id1)
                    || !candidate_image_ids.contains(&image_id2)
                {
                    continue;
                }
                let two_view_geometry = database_cache
                    .correspondence_graph
                    .extract_two_view_geometry(image_id1, image_id2, false)
                    .map_err(|err| anyhow::anyhow!("{err:?}"))?;
                if !use_inlier_matches(options, two_view_geometry.config, num_matches as usize) {
                    continue;
                }
                connected_image_ids.insert(image_id1);
                connected_image_ids.insert(image_id2);
            }
        }
        let load_image_ids = if options.load_all_images {
            candidate_image_ids
        } else {
            connected_image_ids
        };

        let filtered_frame_ids = load_image_ids
            .iter()
            .filter_map(|image_id| database_cache.images.get(image_id))
            .filter_map(|image| image.frame_id)
            .collect::<BTreeSet<_>>();

        let images = database_cache
            .images
            .iter()
            .filter(|(_, image)| {
                image
                    .frame_id
                    .is_some_and(|id| filtered_frame_ids.contains(&id))
            })
            .map(|(&image_id, image)| (image_id, image.clone()))
            .collect::<BTreeMap<_, _>>();

        let filtered_camera_ids = images
            .values()
            .map(|image| image.camera_id)
            .collect::<BTreeSet<_>>();

        let frames = database_cache
            .frames
            .iter()
            .filter(|(&frame_id, _)| filtered_frame_ids.contains(&frame_id))
            .map(|(&frame_id, frame)| (frame_id, frame.clone()))
            .collect::<BTreeMap<_, _>>();

        let filtered_rig_ids = frames
            .values()
            .map(|frame| frame.rig_id)
            .collect::<BTreeSet<_>>();

        let cameras = database_cache
            .cameras
            .iter()
            .filter(|(&camera_id, _)| filtered_camera_ids.contains(&camera_id))
            .map(|(&camera_id, camera)| (camera_id, camera.clone()))
            .collect::<BTreeMap<_, _>>();

        let rigs = database_cache
            .rigs
            .iter()
            .filter(|(&rig_id, _)| filtered_rig_ids.contains(&rig_id))
            .map(|(&rig_id, rig)| (rig_id, rig.clone()))
            .collect::<BTreeMap<_, _>>();

        let mut pose_priors = database_cache.pose_priors.clone();
        if options.convert_pose_priors_to_enu {
            convert_pose_priors_to_enu(&mut pose_priors)?;
        }

        let mut graph = CorrespondenceGraph::new();
        for &image_id in images.keys() {
            let num_points2d = database_cache
                .correspondence_graph
                .num_points2d_for_image(image_id)
                .map_err(|err| anyhow::anyhow!("{err:?}"))?;
            graph
                .add_image(image_id, num_points2d)
                .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        }
        for pair_id in database_cache.correspondence_graph.image_pairs() {
            let (image_id1, image_id2) =
                pair_id_to_image_pair(pair_id).map_err(|err| anyhow::anyhow!("{err:?}"))?;
            if images.contains_key(&image_id1) && images.contains_key(&image_id2) {
                let geometry = database_cache
                    .correspondence_graph
                    .extract_two_view_geometry(image_id1, image_id2, true)
                    .map_err(|err| anyhow::anyhow!("{err:?}"))?;
                graph
                    .add_two_view_geometry(image_id1, image_id2, geometry)
                    .map_err(|err| anyhow::anyhow!("{err:?}"))?;
            }
        }
        graph.finalize().map_err(|err| anyhow::anyhow!("{err:?}"))?;

        Ok(Self {
            rigs,
            cameras,
            frames,
            images,
            pose_priors,
            correspondence_graph: graph,
        })
    }

    pub fn num_rigs(&self) -> usize {
        self.rigs.len()
    }

    pub fn num_cameras(&self) -> usize {
        self.cameras.len()
    }

    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    pub fn num_images(&self) -> usize {
        self.images.len()
    }

    pub fn num_pose_priors(&self) -> usize {
        self.pose_priors.len()
    }

    pub fn add_rig(&mut self, rig: ColmapRig) -> Result<()> {
        if self.exists_rig(rig.rig_id) {
            bail!("rig_id={} already exists in database cache", rig.rig_id);
        }
        self.rigs.insert(rig.rig_id, rig);
        Ok(())
    }

    pub fn add_camera(&mut self, camera: ColmapDatabaseCamera) -> Result<()> {
        let camera_id = camera.camera.camera_id;
        if self.exists_camera(camera_id) {
            bail!("camera_id={camera_id} already exists in database cache");
        }
        self.cameras.insert(camera_id, camera);
        Ok(())
    }

    pub fn add_frame(&mut self, frame: ColmapDatabaseFrame) -> Result<()> {
        if self.exists_frame(frame.frame_id) {
            bail!(
                "frame_id={} already exists in database cache",
                frame.frame_id
            );
        }
        self.frames.insert(frame.frame_id, frame);
        Ok(())
    }

    pub fn add_image(&mut self, image: ColmapDatabaseImage, num_points2d: usize) -> Result<()> {
        if self.exists_image(image.image_id) {
            bail!(
                "image_id={} already exists in database cache",
                image.image_id
            );
        }
        self.correspondence_graph
            .add_image(image.image_id, num_points2d)
            .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        self.images.insert(image.image_id, image);
        Ok(())
    }

    pub fn add_pose_prior(&mut self, pose_prior: ColmapPosePrior) {
        self.pose_priors.push(pose_prior);
    }

    pub fn rig(&self, rig_id: u32) -> Result<&ColmapRig> {
        self.rigs
            .get(&rig_id)
            .with_context(|| format!("rig_id={rig_id} does not exist in database cache"))
    }

    pub fn rig_mut(&mut self, rig_id: u32) -> Result<&mut ColmapRig> {
        self.rigs
            .get_mut(&rig_id)
            .with_context(|| format!("rig_id={rig_id} does not exist in database cache"))
    }

    pub fn camera(&self, camera_id: u32) -> Result<&ColmapDatabaseCamera> {
        self.cameras
            .get(&camera_id)
            .with_context(|| format!("camera_id={camera_id} does not exist in database cache"))
    }

    pub fn camera_mut(&mut self, camera_id: u32) -> Result<&mut ColmapDatabaseCamera> {
        self.cameras
            .get_mut(&camera_id)
            .with_context(|| format!("camera_id={camera_id} does not exist in database cache"))
    }

    pub fn frame(&self, frame_id: u32) -> Result<&ColmapDatabaseFrame> {
        self.frames
            .get(&frame_id)
            .with_context(|| format!("frame_id={frame_id} does not exist in database cache"))
    }

    pub fn frame_mut(&mut self, frame_id: u32) -> Result<&mut ColmapDatabaseFrame> {
        self.frames
            .get_mut(&frame_id)
            .with_context(|| format!("frame_id={frame_id} does not exist in database cache"))
    }

    pub fn image(&self, image_id: ImageId) -> Result<&ColmapDatabaseImage> {
        self.images
            .get(&image_id)
            .with_context(|| format!("image_id={image_id} does not exist in database cache"))
    }

    pub fn image_mut(&mut self, image_id: ImageId) -> Result<&mut ColmapDatabaseImage> {
        self.images
            .get_mut(&image_id)
            .with_context(|| format!("image_id={image_id} does not exist in database cache"))
    }

    pub fn exists_rig(&self, rig_id: u32) -> bool {
        self.rigs.contains_key(&rig_id)
    }

    pub fn exists_camera(&self, camera_id: u32) -> bool {
        self.cameras.contains_key(&camera_id)
    }

    pub fn exists_frame(&self, frame_id: u32) -> bool {
        self.frames.contains_key(&frame_id)
    }

    pub fn exists_image(&self, image_id: ImageId) -> bool {
        self.images.contains_key(&image_id)
    }

    pub fn find_image_with_name(&self, name: &str) -> Option<&ColmapDatabaseImage> {
        self.images.values().find(|image| image.name == name)
    }

    pub fn correspondence_graph(&self) -> &CorrespondenceGraph {
        &self.correspondence_graph
    }

    pub fn correspondence_graph_mut(&mut self) -> &mut CorrespondenceGraph {
        &mut self.correspondence_graph
    }
}

impl Default for DatabaseCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ColmapDatabase {
    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
            .context("open COLMAP database read-only")?;
        conn.execute_batch(
            "PRAGMA foreign_keys = ON;
             PRAGMA query_only = ON;",
        )?;
        Ok(Self {
            conn,
            database_entry_deleted: Cell::new(false),
        })
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path).context("open COLMAP database")?;
        let db = Self {
            conn,
            database_entry_deleted: Cell::new(false),
        };
        db.conn.execute_batch(
            "PRAGMA foreign_keys = ON;
             PRAGMA auto_vacuum = 1;",
        )?;
        db.pre_migrate_tables()?;
        db.create_core_tables()?;
        db.post_migrate_tables()?;
        Ok(db)
    }

    pub fn close(self) -> Result<()> {
        if self.database_entry_deleted.get() {
            self.conn.execute_batch("VACUUM;")?;
            self.database_entry_deleted.set(false);
        }
        Ok(())
    }

    pub fn merge(
        database1: &ColmapDatabase,
        database2: &ColmapDatabase,
        merged_database: &ColmapDatabase,
    ) -> Result<()> {
        let (camera_ids1, rig_ids1, image_ids1) = merge_database_side(database1, merged_database)?;
        let (camera_ids2, rig_ids2, image_ids2) = merge_database_side(database2, merged_database)?;

        merge_database_frames(
            database1,
            merged_database,
            &camera_ids1,
            &rig_ids1,
            &image_ids1,
        )?;
        merge_database_frames(
            database2,
            merged_database,
            &camera_ids2,
            &rig_ids2,
            &image_ids2,
        )?;
        merge_database_pose_priors(database1, merged_database, &camera_ids1, &image_ids1)?;
        merge_database_pose_priors(database2, merged_database, &camera_ids2, &image_ids2)?;
        merge_database_matches(database1, merged_database, &image_ids1)?;
        merge_database_matches(database2, merged_database, &image_ids2)?;
        merge_database_two_view_geometries(database1, merged_database, &image_ids1)?;
        merge_database_two_view_geometries(database2, merged_database, &image_ids2)?;
        Ok(())
    }

    pub fn create_core_tables(&self) -> Result<()> {
        self.conn.execute_batch(
            "PRAGMA foreign_keys = ON;
             CREATE TABLE IF NOT EXISTS cameras
                (camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                 model INTEGER NOT NULL,
                 width INTEGER NOT NULL,
                 height INTEGER NOT NULL,
                 params BLOB,
                 prior_focal_length INTEGER NOT NULL);
             CREATE TABLE IF NOT EXISTS images
                (image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                 name TEXT NOT NULL UNIQUE,
                 camera_id INTEGER NOT NULL,
                 CHECK(image_id >= 0 and image_id < 2147483647),
                 FOREIGN KEY(camera_id) REFERENCES cameras(camera_id));
             CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name);
             CREATE TABLE IF NOT EXISTS rigs
                (rig_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                 ref_sensor_id INTEGER NOT NULL,
                 ref_sensor_type INTEGER NOT NULL);
             CREATE UNIQUE INDEX IF NOT EXISTS rig_ref_sensor_assignment
                ON rigs(ref_sensor_id, ref_sensor_type);
             CREATE TABLE IF NOT EXISTS rig_sensors
                (rig_id INTEGER NOT NULL,
                 sensor_id INTEGER NOT NULL,
                 sensor_type INTEGER NOT NULL,
                 sensor_from_rig BLOB,
                 FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE);
             CREATE UNIQUE INDEX IF NOT EXISTS rig_sensor_assignment
                ON rig_sensors(sensor_id, sensor_type);
             CREATE TABLE IF NOT EXISTS frames
                (frame_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                 rig_id INTEGER NOT NULL,
                 FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE);
             CREATE TABLE IF NOT EXISTS frame_data
                (frame_id INTEGER NOT NULL,
                 data_id INTEGER NOT NULL,
                 sensor_id INTEGER NOT NULL,
                 sensor_type INTEGER NOT NULL,
                 FOREIGN KEY(frame_id) REFERENCES frames(frame_id) ON DELETE CASCADE);
             CREATE UNIQUE INDEX IF NOT EXISTS frame_sensor_assignment
                ON frame_data(data_id, sensor_type);
             CREATE TABLE IF NOT EXISTS pose_priors
                (pose_prior_id INTEGER PRIMARY KEY NOT NULL,
                 image_id INTEGER,
                 corr_data_id INTEGER NOT NULL,
                 corr_sensor_id INTEGER NOT NULL,
                 corr_sensor_type INTEGER NOT NULL,
                 position BLOB,
                 position_covariance BLOB,
                 gravity BLOB,
                 coordinate_system INTEGER NOT NULL);
             CREATE UNIQUE INDEX IF NOT EXISTS pose_prior_data_assignment
                ON pose_priors(corr_data_id, corr_sensor_id, corr_sensor_type);",
        )?;
        self.create_feature_tables()?;
        Ok(())
    }

    pub fn create_feature_tables(&self) -> Result<()> {
        if self.exists_table("inlier_matches")? && !self.exists_table("two_view_geometries")? {
            self.conn
                .execute_batch("ALTER TABLE inlier_matches RENAME TO two_view_geometries;")?;
        }
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS keypoints
                (image_id INTEGER PRIMARY KEY NOT NULL,
                 rows INTEGER NOT NULL,
                 cols INTEGER NOT NULL,
                 data BLOB,
                 FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
             CREATE TABLE IF NOT EXISTS descriptors
                (image_id INTEGER PRIMARY KEY NOT NULL,
                 type INTEGER NOT NULL,
                 rows INTEGER NOT NULL,
                 cols INTEGER NOT NULL,
                 data BLOB,
                 FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
             CREATE TABLE IF NOT EXISTS matches
                (pair_id INTEGER PRIMARY KEY NOT NULL,
                 rows INTEGER NOT NULL,
                 cols INTEGER NOT NULL,
                 data BLOB);
             CREATE TABLE IF NOT EXISTS two_view_geometries
                (pair_id INTEGER PRIMARY KEY NOT NULL,
                 rows INTEGER NOT NULL,
                 cols INTEGER NOT NULL,
                 data BLOB,
                 config INTEGER NOT NULL,
                 F BLOB,
                 E BLOB,
                 H BLOB,
                 qvec BLOB,
                 tvec BLOB);",
        )?;
        Ok(())
    }

    fn pre_migrate_tables(&self) -> Result<()> {
        if self.exists_legacy_image_pose_prior_table()? && !self.exists_table("pose_priors_old")? {
            self.conn
                .execute_batch("ALTER TABLE pose_priors RENAME TO pose_priors_old;")?;
        }
        Ok(())
    }

    fn post_migrate_tables(&self) -> Result<()> {
        for column_name in ["F", "E", "H", "qvec", "tvec"] {
            self.add_blob_column_if_missing("two_view_geometries", column_name)?;
        }

        let user_version = self.user_version()?;
        if user_version <= COLMAP_DATABASE_VERSION_3_13_0_0 {
            self.migrate_legacy_two_view_pose_sentinels()?;
        }
        if user_version <= COLMAP_DATABASE_VERSION_3_14_0_0 {
            self.migrate_legacy_zero_two_view_matrices()?;
        }
        if user_version <= COLMAP_DATABASE_VERSION_3_14_0_1
            && !self.exists_column("descriptors", "type")?
        {
            self.conn.execute_batch(
                "ALTER TABLE descriptors ADD COLUMN type INTEGER NOT NULL DEFAULT 0;",
            )?;
        }
        if self.exists_table("pose_priors_old")? {
            self.migrate_legacy_pose_priors()?;
        }
        self.add_colmap_legacy_pose_prior_image_id_column()?;
        self.set_user_version(COLMAP_CURRENT_DATABASE_VERSION)?;
        Ok(())
    }

    fn exists_legacy_image_pose_prior_table(&self) -> Result<bool> {
        Ok(self.exists_column("pose_priors", "image_id")?
            && !self.exists_column("pose_priors", "corr_data_id")?)
    }

    fn add_colmap_legacy_pose_prior_image_id_column(&self) -> Result<()> {
        if self.exists_table("pose_priors")?
            && self.exists_column("pose_priors", "corr_data_id")?
            && !self.exists_column("pose_priors", "image_id")?
        {
            self.conn
                .execute_batch("ALTER TABLE pose_priors ADD COLUMN image_id INTEGER;")?;
        }
        self.sync_colmap_legacy_pose_prior_image_ids()
    }

    fn sync_colmap_legacy_pose_prior_image_ids(&self) -> Result<()> {
        if self.exists_table("pose_priors")?
            && self.exists_column("pose_priors", "image_id")?
            && self.exists_column("pose_priors", "corr_data_id")?
        {
            self.conn.execute_batch(
                "UPDATE pose_priors
                 SET image_id = corr_data_id
                 WHERE corr_sensor_type = 0;",
            )?;
        }
        Ok(())
    }

    fn add_blob_column_if_missing(&self, table_name: &str, column_name: &str) -> Result<()> {
        if !self.exists_column(table_name, column_name)? {
            self.conn.execute_batch(&format!(
                "ALTER TABLE {table_name} ADD COLUMN {column_name} BLOB;"
            ))?;
        }
        Ok(())
    }

    fn migrate_legacy_two_view_pose_sentinels(&self) -> Result<()> {
        let rows = {
            let mut stmt = self.conn.prepare(
                "SELECT pair_id, qvec, tvec FROM two_view_geometries
                 WHERE qvec IS NOT NULL AND tvec IS NOT NULL;",
            )?;
            let mut rows = stmt.query([])?;
            let mut out = Vec::new();
            while let Some(row) = rows.next()? {
                out.push((
                    row.get::<_, i64>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                ));
            }
            out
        };

        for (pair_id, q_blob, t_blob) in rows {
            let qvec = decode_static_f64_blob_or_zero::<4>(&q_blob)?;
            let tvec = decode_static_f64_blob_or_zero::<3>(&t_blob)?;
            let is_identity = qvec == [1.0, 0.0, 0.0, 0.0] && tvec == [0.0, 0.0, 0.0];
            let is_zero = qvec == [0.0, 0.0, 0.0, 0.0];
            if is_identity || is_zero {
                self.conn.execute(
                    "UPDATE two_view_geometries SET qvec = NULL, tvec = NULL WHERE pair_id = ?1;",
                    params![pair_id],
                )?;
            }
        }
        Ok(())
    }

    fn migrate_legacy_zero_two_view_matrices(&self) -> Result<()> {
        let rows = {
            let mut stmt = self.conn.prepare(
                "SELECT pair_id, F, E, H FROM two_view_geometries
                 WHERE F IS NOT NULL OR E IS NOT NULL OR H IS NOT NULL;",
            )?;
            let mut rows = stmt.query([])?;
            let mut out = Vec::new();
            while let Some(row) = rows.next()? {
                out.push((
                    row.get::<_, i64>(0)?,
                    row.get::<_, Option<Vec<u8>>>(1)?,
                    row.get::<_, Option<Vec<u8>>>(2)?,
                    row.get::<_, Option<Vec<u8>>>(3)?,
                ));
            }
            out
        };

        for (pair_id, f_blob, e_blob, h_blob) in rows {
            if f_blob
                .as_deref()
                .map(is_zero_static_f64_blob::<9>)
                .transpose()?
                .unwrap_or(false)
            {
                self.conn.execute(
                    "UPDATE two_view_geometries SET F = NULL WHERE pair_id = ?1;",
                    params![pair_id],
                )?;
            }
            if e_blob
                .as_deref()
                .map(is_zero_static_f64_blob::<9>)
                .transpose()?
                .unwrap_or(false)
            {
                self.conn.execute(
                    "UPDATE two_view_geometries SET E = NULL WHERE pair_id = ?1;",
                    params![pair_id],
                )?;
            }
            if h_blob
                .as_deref()
                .map(is_zero_static_f64_blob::<9>)
                .transpose()?
                .unwrap_or(false)
            {
                self.conn.execute(
                    "UPDATE two_view_geometries SET H = NULL WHERE pair_id = ?1;",
                    params![pair_id],
                )?;
            }
        }
        Ok(())
    }

    fn migrate_legacy_pose_priors(&self) -> Result<()> {
        let gravity = encode_f64_blob(&[f64::NAN; 3]);
        self.conn.execute(
            "INSERT INTO pose_priors(
                pose_prior_id, image_id, corr_data_id, corr_sensor_id, corr_sensor_type,
                position, position_covariance, coordinate_system, gravity)
             SELECT pose_priors_old.image_id, pose_priors_old.image_id,
                    pose_priors_old.image_id, images.camera_id, ?1, pose_priors_old.position,
                    pose_priors_old.position_covariance,
                    pose_priors_old.coordinate_system, ?2
             FROM pose_priors_old
             JOIN images ON pose_priors_old.image_id = images.image_id;",
            params![sensor_type_to_i64(&ColmapSensorType::Camera)?, gravity],
        )?;
        self.conn.execute_batch("DROP TABLE pose_priors_old;")?;
        Ok(())
    }

    fn user_version(&self) -> Result<i32> {
        self.conn
            .query_row("PRAGMA user_version;", [], |row| row.get(0))
            .map_err(Into::into)
    }

    fn set_user_version(&self, version: i32) -> Result<()> {
        self.conn
            .execute_batch(&format!("PRAGMA user_version = {version};"))?;
        Ok(())
    }

    fn exists_table(&self, table_name: &str) -> Result<bool> {
        let exists = self
            .conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1;",
                params![table_name],
                |_| Ok(true),
            )
            .optional()?
            .unwrap_or(false);
        Ok(exists)
    }

    fn exists_column(&self, table_name: &str, column_name: &str) -> Result<bool> {
        let mut stmt = self
            .conn
            .prepare(&format!("PRAGMA table_info({table_name});"))?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            if row.get::<_, String>(1)? == column_name {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn begin_transaction(&self) -> Result<()> {
        self.conn.execute_batch("BEGIN TRANSACTION;")?;
        Ok(())
    }

    pub fn end_transaction(&self) -> Result<()> {
        self.conn.execute_batch("END TRANSACTION;")?;
        Ok(())
    }

    pub fn with_transaction<T>(&self, operation: impl FnOnce() -> Result<T>) -> Result<T> {
        let deleted_before = self.database_entry_deleted.get();
        self.conn.execute_batch("BEGIN IMMEDIATE TRANSACTION;")?;
        match operation() {
            Ok(value) => {
                if let Err(commit_error) = self.conn.execute_batch("COMMIT;") {
                    let _ = self.conn.execute_batch("ROLLBACK;");
                    self.database_entry_deleted.set(deleted_before);
                    return Err(commit_error.into());
                }
                Ok(value)
            }
            Err(error) => {
                let rollback = self.conn.execute_batch("ROLLBACK;");
                self.database_entry_deleted.set(deleted_before);
                rollback.context("roll back COLMAP database transaction")?;
                Err(error)
            }
        }
    }

    pub fn exists_rig(&self, rig_id: u32) -> Result<bool> {
        self.exists_row_id("rigs", "rig_id", rig_id as i64)
    }

    pub fn exists_camera(&self, camera_id: u32) -> Result<bool> {
        self.exists_row_id("cameras", "camera_id", camera_id as i64)
    }

    pub fn exists_frame(&self, frame_id: u32) -> Result<bool> {
        self.exists_row_id("frames", "frame_id", frame_id as i64)
    }

    pub fn exists_image(&self, image_id: ImageId) -> Result<bool> {
        self.exists_row_id("images", "image_id", image_id as i64)
    }

    pub fn exists_image_with_name(&self, name: &str) -> Result<bool> {
        self.conn
            .query_row(
                "SELECT 1 FROM images WHERE name = ?1;",
                params![name],
                |_| Ok(true),
            )
            .optional()
            .map(|value| value.unwrap_or(false))
            .map_err(Into::into)
    }

    pub fn exists_pose_prior(&self, pose_prior_id: u32) -> Result<bool> {
        self.exists_row_id("pose_priors", "pose_prior_id", pose_prior_id as i64)
    }

    pub fn exists_keypoints(&self, image_id: ImageId) -> Result<bool> {
        self.exists_row_id("keypoints", "image_id", image_id as i64)
    }

    pub fn exists_descriptors(&self, image_id: ImageId) -> Result<bool> {
        self.exists_row_id("descriptors", "image_id", image_id as i64)
    }

    pub fn exists_matches(&self, image_id1: ImageId, image_id2: ImageId) -> Result<bool> {
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)
            .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        self.exists_row_id("matches", "pair_id", pair_id as i64)
    }

    pub fn exists_two_view_geometry(&self, image_id1: ImageId, image_id2: ImageId) -> Result<bool> {
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)
            .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        self.exists_row_id("two_view_geometries", "pair_id", pair_id as i64)
    }

    fn exists_row_id(&self, table_name: &str, column_name: &str, value: i64) -> Result<bool> {
        self.conn
            .query_row(
                &format!("SELECT 1 FROM {table_name} WHERE {column_name} = ?1;"),
                params![value],
                |_| Ok(true),
            )
            .optional()
            .map(|value| value.unwrap_or(false))
            .map_err(Into::into)
    }

    pub fn num_rigs(&self) -> Result<usize> {
        self.count_rows("rigs")
    }

    pub fn num_cameras(&self) -> Result<usize> {
        self.count_rows("cameras")
    }

    pub fn num_frames(&self) -> Result<usize> {
        self.count_rows("frames")
    }

    pub fn num_images(&self) -> Result<usize> {
        self.count_rows("images")
    }

    pub fn num_pose_priors(&self) -> Result<usize> {
        self.count_rows("pose_priors")
    }

    pub fn num_keypoints(&self) -> Result<usize> {
        self.sum_rows("keypoints")
    }

    pub fn max_num_keypoints(&self) -> Result<usize> {
        self.max_rows("keypoints")
    }

    pub fn num_keypoints_for_image(&self, image_id: ImageId) -> Result<usize> {
        self.rows_for_entry("keypoints", "image_id", image_id as i64)
    }

    pub fn num_descriptors(&self) -> Result<usize> {
        self.sum_rows("descriptors")
    }

    pub fn max_num_descriptors(&self) -> Result<usize> {
        self.max_rows("descriptors")
    }

    pub fn num_descriptors_for_image(&self, image_id: ImageId) -> Result<usize> {
        self.rows_for_entry("descriptors", "image_id", image_id as i64)
    }

    pub fn num_matches(&self) -> Result<usize> {
        self.sum_rows("matches")
    }

    pub fn num_inlier_matches(&self) -> Result<usize> {
        self.sum_rows("two_view_geometries")
    }

    pub fn num_matched_image_pairs(&self) -> Result<usize> {
        self.count_rows("matches")
    }

    pub fn num_verified_image_pairs(&self) -> Result<usize> {
        self.count_rows("two_view_geometries")
    }

    fn count_rows(&self, table_name: &str) -> Result<usize> {
        self.conn
            .query_row(&format!("SELECT COUNT(*) FROM {table_name};"), [], |row| {
                checked_row_integer(row, 0)
            })
            .map_err(Into::into)
    }

    fn sum_rows(&self, table_name: &str) -> Result<usize> {
        self.conn
            .query_row(&format!("SELECT SUM(rows) FROM {table_name};"), [], |row| {
                checked_optional_row_integer(row, 0)
            })
            .map(|value: Option<usize>| value.unwrap_or(0))
            .map_err(Into::into)
    }

    fn max_rows(&self, table_name: &str) -> Result<usize> {
        self.conn
            .query_row(&format!("SELECT MAX(rows) FROM {table_name};"), [], |row| {
                checked_optional_row_integer(row, 0)
            })
            .map(|value: Option<usize>| value.unwrap_or(0))
            .map_err(Into::into)
    }

    fn rows_for_entry(&self, table_name: &str, column_name: &str, value: i64) -> Result<usize> {
        self.conn
            .query_row(
                &format!("SELECT rows FROM {table_name} WHERE {column_name} = ?1;"),
                params![value],
                |row| checked_row_integer(row, 0),
            )
            .optional()
            .map(|value: Option<usize>| value.unwrap_or(0))
            .map_err(Into::into)
    }

    pub fn write_camera(&self, camera: &ColmapDatabaseCamera, use_camera_id: bool) -> Result<u32> {
        validate_camera_params(&camera.camera)?;
        let params_blob = encode_f64_blob(&camera.camera.params);
        if use_camera_id {
            self.conn.execute(
                "INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length)
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6);",
                params![
                    camera.camera.camera_id,
                    camera.camera.model_id,
                    camera.camera.width,
                    camera.camera.height,
                    params_blob,
                    camera.has_prior_focal_length as i64
                ],
            )?;
            Ok(camera.camera.camera_id)
        } else {
            self.conn.execute(
                "INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length)
                 VALUES(NULL, ?1, ?2, ?3, ?4, ?5);",
                params![
                    camera.camera.model_id,
                    camera.camera.width,
                    camera.camera.height,
                    params_blob,
                    camera.has_prior_focal_length as i64
                ],
            )?;
            Ok(self.conn.last_insert_rowid() as u32)
        }
    }

    pub fn read_camera(&self, camera_id: u32) -> Result<Option<ColmapDatabaseCamera>> {
        self.conn
            .query_row(
                "SELECT camera_id, model, width, height, params, prior_focal_length
                 FROM cameras WHERE camera_id = ?1;",
                params![camera_id],
                read_camera_row,
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn read_all_cameras(&self) -> Result<Vec<ColmapDatabaseCamera>> {
        let mut stmt = self.conn.prepare(
            "SELECT camera_id, model, width, height, params, prior_focal_length
             FROM cameras;",
        )?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(read_camera_row(row)?);
        }
        Ok(out)
    }

    pub fn write_image(&self, image: &ColmapDatabaseImage, use_image_id: bool) -> Result<ImageId> {
        if use_image_id {
            self.conn.execute(
                "INSERT INTO images(image_id, name, camera_id) VALUES(?1, ?2, ?3);",
                params![image.image_id, &image.name, image.camera_id],
            )?;
            Ok(image.image_id)
        } else {
            self.conn.execute(
                "INSERT INTO images(image_id, name, camera_id) VALUES(NULL, ?1, ?2);",
                params![&image.name, image.camera_id],
            )?;
            Ok(self.conn.last_insert_rowid() as ImageId)
        }
    }

    pub fn read_image(&self, image_id: ImageId) -> Result<Option<ColmapDatabaseImage>> {
        self.conn
            .query_row(
                "SELECT images.image_id, images.name, images.camera_id, frame_data.frame_id
                 FROM images
                 LEFT JOIN frame_data
                   ON images.image_id = frame_data.data_id AND frame_data.sensor_type = 0
                 WHERE images.image_id = ?1;",
                params![image_id],
                read_image_row,
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn read_image_with_name(&self, name: &str) -> Result<Option<ColmapDatabaseImage>> {
        self.conn
            .query_row(
                "SELECT images.image_id, images.name, images.camera_id, frame_data.frame_id
                 FROM images
                 LEFT JOIN frame_data
                   ON images.image_id = frame_data.data_id AND frame_data.sensor_type = 0
                 WHERE images.name = ?1;",
                params![name],
                read_image_row,
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn read_all_images(&self) -> Result<Vec<ColmapDatabaseImage>> {
        let mut stmt = self.conn.prepare(
            "SELECT images.image_id, images.name, images.camera_id, frame_data.frame_id
             FROM images
             LEFT JOIN frame_data
               ON images.image_id = frame_data.data_id AND frame_data.sensor_type = 0;",
        )?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(read_image_row(row)?);
        }
        Ok(out)
    }

    pub fn write_rig(&self, rig: &ColmapRig, use_rig_id: bool) -> Result<u32> {
        let ref_sensor_id = rig
            .ref_sensor_id
            .as_ref()
            .context("COLMAP database rig requires a reference sensor")?;
        if rig.sensors.is_empty() {
            bail!("COLMAP database rig requires at least one sensor");
        }
        if use_rig_id {
            self.conn.execute(
                "INSERT INTO rigs(rig_id, ref_sensor_id, ref_sensor_type) VALUES(?1, ?2, ?3);",
                params![
                    rig.rig_id,
                    ref_sensor_id.sensor_id,
                    sensor_type_to_i64(&ref_sensor_id.sensor_type)?
                ],
            )?;
            write_rig_sensors(&self.conn, rig.rig_id, rig)?;
            Ok(rig.rig_id)
        } else {
            self.conn.execute(
                "INSERT INTO rigs(rig_id, ref_sensor_id, ref_sensor_type) VALUES(NULL, ?1, ?2);",
                params![
                    ref_sensor_id.sensor_id,
                    sensor_type_to_i64(&ref_sensor_id.sensor_type)?
                ],
            )?;
            let rig_id = self.conn.last_insert_rowid() as u32;
            write_rig_sensors(&self.conn, rig_id, rig)?;
            Ok(rig_id)
        }
    }

    pub fn read_rig(&self, rig_id: u32) -> Result<Option<ColmapRig>> {
        let mut stmt = self.conn.prepare(
            "SELECT rigs.rig_id, rigs.ref_sensor_id, rigs.ref_sensor_type,
                    rig_sensors.sensor_id, rig_sensors.sensor_type, rig_sensors.sensor_from_rig
             FROM rigs
             LEFT OUTER JOIN rig_sensors ON rigs.rig_id = rig_sensors.rig_id
             WHERE rigs.rig_id = ?1
             ORDER BY rigs.rig_id;",
        )?;
        let mut rows = stmt.query(params![rig_id])?;
        Ok(collect_rig_rows(&mut rows)?.into_iter().next())
    }

    pub fn read_rig_with_sensor(&self, sensor_id: &ColmapSensorId) -> Result<Option<ColmapRig>> {
        let sensor_type = sensor_type_to_i64(&sensor_id.sensor_type)?;
        let rig_id = if let Some(rig_id) = self
            .conn
            .query_row(
                "SELECT rig_id FROM rig_sensors WHERE sensor_id = ?1 AND sensor_type = ?2;",
                params![sensor_id.sensor_id, sensor_type],
                |row| row.get::<_, i64>(0),
            )
            .optional()?
        {
            Some(rig_id as u32)
        } else {
            self.conn
                .query_row(
                    "SELECT rig_id FROM rigs WHERE ref_sensor_id = ?1 AND ref_sensor_type = ?2;",
                    params![sensor_id.sensor_id, sensor_type],
                    |row| row.get::<_, i64>(0),
                )
                .optional()?
                .map(|rig_id| rig_id as u32)
        };

        match rig_id {
            Some(rig_id) => self.read_rig(rig_id),
            None => Ok(None),
        }
    }

    pub fn read_all_rigs(&self) -> Result<Vec<ColmapRig>> {
        let mut stmt = self.conn.prepare(
            "SELECT rigs.rig_id, rigs.ref_sensor_id, rigs.ref_sensor_type,
                    rig_sensors.sensor_id, rig_sensors.sensor_type, rig_sensors.sensor_from_rig
             FROM rigs
             LEFT OUTER JOIN rig_sensors ON rigs.rig_id = rig_sensors.rig_id
             ORDER BY rigs.rig_id;",
        )?;
        let mut rows = stmt.query([])?;
        collect_rig_rows(&mut rows)
    }

    pub fn write_frame(&self, frame: &ColmapDatabaseFrame, use_frame_id: bool) -> Result<u32> {
        if use_frame_id {
            self.conn.execute(
                "INSERT INTO frames(frame_id, rig_id) VALUES(?1, ?2);",
                params![frame.frame_id, frame.rig_id],
            )?;
            write_frame_data(&self.conn, frame.frame_id, frame)?;
            Ok(frame.frame_id)
        } else {
            self.conn.execute(
                "INSERT INTO frames(frame_id, rig_id) VALUES(NULL, ?1);",
                params![frame.rig_id],
            )?;
            let frame_id = self.conn.last_insert_rowid() as u32;
            write_frame_data(&self.conn, frame_id, frame)?;
            Ok(frame_id)
        }
    }

    pub fn read_frame(&self, frame_id: u32) -> Result<Option<ColmapDatabaseFrame>> {
        let mut stmt = self.conn.prepare(
            "SELECT frames.frame_id, frames.rig_id, frame_data.data_id,
                    frame_data.sensor_id, frame_data.sensor_type
             FROM frames
             LEFT OUTER JOIN frame_data ON frames.frame_id = frame_data.frame_id
             WHERE frames.frame_id = ?1
             ORDER BY frames.frame_id;",
        )?;
        let mut rows = stmt.query(params![frame_id])?;
        Ok(collect_frame_rows(&mut rows)?.into_iter().next())
    }

    pub fn read_all_frames(&self) -> Result<Vec<ColmapDatabaseFrame>> {
        let mut stmt = self.conn.prepare(
            "SELECT frames.frame_id, frames.rig_id, frame_data.data_id,
                    frame_data.sensor_id, frame_data.sensor_type
             FROM frames
             LEFT OUTER JOIN frame_data ON frames.frame_id = frame_data.frame_id
             ORDER BY frames.frame_id;",
        )?;
        let mut rows = stmt.query([])?;
        collect_frame_rows(&mut rows)
    }

    pub fn write_pose_prior(
        &self,
        pose_prior: &ColmapPosePrior,
        use_pose_prior_id: bool,
    ) -> Result<u32> {
        let position = encode_f64_blob(&pose_prior.position);
        let position_covariance = encode_f64_blob(&pose_prior.position_covariance);
        let gravity = encode_f64_blob(&pose_prior.gravity);
        if use_pose_prior_id {
            self.conn.execute(
                "INSERT INTO pose_priors(
                    pose_prior_id, image_id, corr_data_id, corr_sensor_id, corr_sensor_type,
                    position, position_covariance, coordinate_system, gravity)
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9);",
                params![
                    pose_prior.pose_prior_id,
                    pose_prior_legacy_image_id(pose_prior)?,
                    pose_prior.corr_data_id.data_id as i64,
                    pose_prior.corr_data_id.sensor_id.sensor_id,
                    sensor_type_to_i64(&pose_prior.corr_data_id.sensor_id.sensor_type)?,
                    position,
                    position_covariance,
                    coordinate_system_to_i64(&pose_prior.coordinate_system)?,
                    gravity
                ],
            )?;
            Ok(pose_prior.pose_prior_id)
        } else {
            self.conn.execute(
                "INSERT INTO pose_priors(
                    pose_prior_id, image_id, corr_data_id, corr_sensor_id, corr_sensor_type,
                    position, position_covariance, coordinate_system, gravity)
                 VALUES(NULL, ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8);",
                params![
                    pose_prior_legacy_image_id(pose_prior)?,
                    pose_prior.corr_data_id.data_id as i64,
                    pose_prior.corr_data_id.sensor_id.sensor_id,
                    sensor_type_to_i64(&pose_prior.corr_data_id.sensor_id.sensor_type)?,
                    position,
                    position_covariance,
                    coordinate_system_to_i64(&pose_prior.coordinate_system)?,
                    gravity
                ],
            )?;
            Ok(self.conn.last_insert_rowid() as u32)
        }
    }

    pub fn read_pose_prior(&self, pose_prior_id: u32) -> Result<Option<ColmapPosePrior>> {
        self.conn
            .query_row(
                "SELECT pose_prior_id, corr_data_id, corr_sensor_id, corr_sensor_type,
                        position, position_covariance, coordinate_system, gravity
                 FROM pose_priors WHERE pose_prior_id = ?1;",
                params![pose_prior_id],
                read_pose_prior_row,
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn read_all_pose_priors(&self) -> Result<Vec<ColmapPosePrior>> {
        let mut stmt = self.conn.prepare(
            "SELECT pose_prior_id, corr_data_id, corr_sensor_id, corr_sensor_type,
                    position, position_covariance, coordinate_system, gravity
             FROM pose_priors;",
        )?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(read_pose_prior_row(row)?);
        }
        Ok(out)
    }

    pub fn write_keypoints(&self, image_id: ImageId, keypoints: &[ColmapKeypoint]) -> Result<()> {
        let (rows, cols, data) = encode_keypoints_blob(keypoints);
        self.write_keypoints_blob(image_id, &ColmapKeypointsBlob { rows, cols, data })
    }

    pub fn write_keypoints_blob(
        &self,
        image_id: ImageId,
        blob: &ColmapKeypointsBlob,
    ) -> Result<()> {
        validate_keypoints_blob(blob)?;
        self.conn.execute(
            "INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?1, ?2, ?3, ?4);",
            params![image_id, blob.rows as i64, blob.cols as i64, &blob.data],
        )?;
        Ok(())
    }

    pub fn read_keypoints(&self, image_id: ImageId) -> Result<Vec<ColmapKeypoint>> {
        let blob = self.read_keypoints_blob(image_id)?;
        decode_keypoints_blob(blob.rows, blob.cols, &blob.data)
    }

    pub fn read_keypoints_blob(&self, image_id: ImageId) -> Result<ColmapKeypointsBlob> {
        self.read_dynamic_blob(
            "SELECT rows, cols, data FROM keypoints WHERE image_id = ?1;",
            image_id as i64,
            "keypoints",
        )
    }

    fn read_dynamic_blob(&self, sql: &str, id: i64, label: &str) -> Result<ColmapKeypointsBlob> {
        let row = self
            .conn
            .query_row(sql, params![id], |row| {
                Ok((
                    checked_row_integer(row, 0)?,
                    checked_row_integer(row, 1)?,
                    row.get::<_, Vec<u8>>(2)?,
                ))
            })
            .optional()?;
        let Some((rows, cols, data)) = row else {
            return Ok(ColmapKeypointsBlob {
                rows: 0,
                cols: 0,
                data: Vec::new(),
            });
        };
        validate_dynamic_blob(label, rows, cols, std::mem::size_of::<f32>(), &data)?;
        Ok(ColmapKeypointsBlob { rows, cols, data })
    }

    pub fn write_descriptors(
        &self,
        image_id: ImageId,
        descriptors: &ColmapDescriptors,
    ) -> Result<()> {
        if descriptors.data.len() != descriptors.rows.saturating_mul(descriptors.cols) {
            bail!("descriptor data length does not match rows*cols");
        }
        self.conn.execute(
            "INSERT INTO descriptors(image_id, rows, cols, data, type) VALUES(?1, ?2, ?3, ?4, ?5);",
            params![
                image_id,
                descriptors.rows as i64,
                descriptors.cols as i64,
                descriptors.data,
                descriptors.feature_type
            ],
        )?;
        Ok(())
    }

    pub fn read_descriptors(&self, image_id: ImageId) -> Result<ColmapDescriptors> {
        let row = self
            .conn
            .query_row(
                "SELECT rows, cols, data, type FROM descriptors WHERE image_id = ?1;",
                params![image_id],
                |row| {
                    Ok((
                        checked_row_integer(row, 0)?,
                        checked_row_integer(row, 1)?,
                        row.get::<_, Vec<u8>>(2)?,
                        row.get::<_, i32>(3)?,
                    ))
                },
            )
            .optional()?;
        let Some((rows, cols, data, feature_type)) = row else {
            return ColmapDescriptors::new(-1, 0, 0, Vec::new());
        };
        ColmapDescriptors::new(feature_type, rows, cols, data)
    }

    pub fn read_keypoint_counts(&self) -> Result<Vec<(ImageId, usize)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT image_id, rows FROM keypoints ORDER BY image_id;")?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push((checked_row_integer(row, 0)?, checked_row_integer(row, 1)?));
        }
        Ok(out)
    }

    pub fn write_matches(
        &self,
        image_id1: ImageId,
        image_id2: ImageId,
        matches: &[FeatureMatch],
    ) -> Result<()> {
        let data = encode_matches_blob(matches);
        self.write_matches_blob(
            image_id1,
            image_id2,
            &ColmapMatchesBlob {
                rows: matches.len(),
                cols: 2,
                data,
            },
        )
    }

    pub fn write_matches_blob(
        &self,
        image_id1: ImageId,
        image_id2: ImageId,
        blob: &ColmapMatchesBlob,
    ) -> Result<()> {
        validate_matches_blob(blob)?;
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)
            .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        let stored_blob = maybe_swapped_matches_blob(image_id1, image_id2, blob)?;
        self.conn.execute(
            "INSERT INTO matches(pair_id, rows, cols, data) VALUES(?1, ?2, ?3, ?4);",
            params![
                pair_id as i64,
                stored_blob.rows as i64,
                stored_blob.cols as i64,
                stored_blob.data
            ],
        )?;
        Ok(())
    }

    pub fn read_matches(
        &self,
        image_id1: ImageId,
        image_id2: ImageId,
    ) -> Result<Vec<FeatureMatch>> {
        let blob = self.read_matches_blob(image_id1, image_id2)?;
        decode_matches_blob(blob.rows, blob.cols, &blob.data)
    }

    pub fn read_matches_blob(
        &self,
        image_id1: ImageId,
        image_id2: ImageId,
    ) -> Result<ColmapMatchesBlob> {
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)
            .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        let blob = self.read_matches_blob_by_pair_id(pair_id)?;
        maybe_swapped_matches_blob(image_id1, image_id2, &blob)
    }

    fn read_matches_blob_by_pair_id(&self, pair_id: ImagePairId) -> Result<ColmapMatchesBlob> {
        let row = self
            .conn
            .query_row(
                "SELECT rows, cols, data FROM matches WHERE pair_id = ?1;",
                params![pair_id as i64],
                |row| {
                    Ok((
                        checked_row_integer(row, 0)?,
                        checked_row_integer(row, 1)?,
                        row.get::<_, Vec<u8>>(2)?,
                    ))
                },
            )
            .optional()?;
        let Some((rows, cols, data)) = row else {
            return Ok(ColmapMatchesBlob {
                rows: 0,
                cols: 2,
                data: Vec::new(),
            });
        };
        ColmapMatchesBlob::new(rows, cols, data)
    }

    pub fn read_all_matches_blob(&self) -> Result<Vec<(ImagePairId, ColmapMatchesBlob)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT pair_id, rows, cols, data FROM matches WHERE rows > 0;")?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push((
                checked_row_integer(row, 0)?,
                ColmapMatchesBlob::new(
                    checked_row_integer(row, 1)?,
                    checked_row_integer(row, 2)?,
                    row.get::<_, Vec<u8>>(3)?,
                )?,
            ));
        }
        Ok(out)
    }

    pub fn read_all_matches(&self) -> Result<Vec<(ImagePairId, Vec<FeatureMatch>)>> {
        self.read_all_matches_blob()?
            .into_iter()
            .map(|(pair_id, blob)| {
                Ok((
                    pair_id,
                    decode_matches_blob(blob.rows, blob.cols, &blob.data)?,
                ))
            })
            .collect()
    }

    pub fn read_num_matches(&self) -> Result<Vec<(ImagePairId, i32)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT pair_id, rows FROM matches WHERE rows > 0;")?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push((checked_row_integer(row, 0)?, checked_row_integer(row, 1)?));
        }
        Ok(out)
    }

    pub fn write_two_view_geometry(
        &self,
        image_id1: ImageId,
        image_id2: ImageId,
        geometry: &ColmapTwoViewGeometry,
    ) -> Result<()> {
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)
            .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        let mut stored = geometry.clone();
        if should_swap_image_pair(image_id1, image_id2) {
            stored.invert();
        }
        let data = encode_matches_blob(&stored.inlier_matches);
        let f_blob = stored
            .f_matrix
            .as_ref()
            .map(|m| encode_matrix3_colmap_blob(*m));
        let e_blob = stored
            .e_matrix
            .as_ref()
            .map(|m| encode_matrix3_colmap_blob(*m));
        let h_blob = stored
            .h_matrix
            .as_ref()
            .map(|m| encode_matrix3_colmap_blob(*m));
        let q_blob = stored.qvec.as_ref().map(|v| encode_f64_blob(v));
        let t_blob = stored.tvec.as_ref().map(|v| encode_f64_blob(v));
        self.conn.execute(
            "INSERT INTO two_view_geometries(pair_id, rows, cols, data, config, F, E, H, qvec, tvec)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10);",
            params![
                pair_id as i64,
                stored.inlier_matches.len() as i64,
                2i64,
                data,
                stored.config,
                f_blob,
                e_blob,
                h_blob,
                q_blob,
                t_blob
            ],
        )?;
        Ok(())
    }

    pub fn update_rig(&self, rig: &ColmapRig) -> Result<()> {
        let ref_sensor_id = rig
            .ref_sensor_id
            .as_ref()
            .context("COLMAP database rig requires a reference sensor")?;
        self.conn.execute(
            "UPDATE rigs SET ref_sensor_id = ?1, ref_sensor_type = ?2 WHERE rig_id = ?3;",
            params![
                ref_sensor_id.sensor_id,
                sensor_type_to_i64(&ref_sensor_id.sensor_type)?,
                rig.rig_id
            ],
        )?;
        self.conn.execute(
            "DELETE FROM rig_sensors WHERE rig_id = ?1;",
            params![rig.rig_id],
        )?;
        write_rig_sensors(&self.conn, rig.rig_id, rig)?;
        Ok(())
    }

    pub fn update_camera(&self, camera: &ColmapDatabaseCamera) -> Result<()> {
        validate_camera_params(&camera.camera)?;
        self.conn.execute(
            "UPDATE cameras
             SET model = ?1, width = ?2, height = ?3, params = ?4, prior_focal_length = ?5
             WHERE camera_id = ?6;",
            params![
                camera.camera.model_id,
                camera.camera.width,
                camera.camera.height,
                encode_f64_blob(&camera.camera.params),
                camera.has_prior_focal_length as i64,
                camera.camera.camera_id
            ],
        )?;
        Ok(())
    }

    pub fn update_frame(&self, frame: &ColmapDatabaseFrame) -> Result<()> {
        self.conn.execute(
            "UPDATE frames SET rig_id = ?1 WHERE frame_id = ?2;",
            params![frame.rig_id, frame.frame_id],
        )?;
        self.conn.execute(
            "DELETE FROM frame_data WHERE frame_id = ?1;",
            params![frame.frame_id],
        )?;
        write_frame_data(&self.conn, frame.frame_id, frame)?;
        Ok(())
    }

    pub fn update_image(&self, image: &ColmapDatabaseImage) -> Result<()> {
        self.conn.execute(
            "UPDATE images SET name = ?1, camera_id = ?2 WHERE image_id = ?3;",
            params![&image.name, image.camera_id, image.image_id],
        )?;
        Ok(())
    }

    pub fn update_pose_prior(&self, pose_prior: &ColmapPosePrior) -> Result<()> {
        self.conn.execute(
            "UPDATE pose_priors
             SET image_id = ?1, corr_data_id = ?2, corr_sensor_id = ?3, corr_sensor_type = ?4,
                 position = ?5, position_covariance = ?6, coordinate_system = ?7, gravity = ?8
             WHERE pose_prior_id = ?9;",
            params![
                pose_prior_legacy_image_id(pose_prior)?,
                pose_prior.corr_data_id.data_id as i64,
                pose_prior.corr_data_id.sensor_id.sensor_id,
                sensor_type_to_i64(&pose_prior.corr_data_id.sensor_id.sensor_type)?,
                encode_f64_blob(&pose_prior.position),
                encode_f64_blob(&pose_prior.position_covariance),
                coordinate_system_to_i64(&pose_prior.coordinate_system)?,
                encode_f64_blob(&pose_prior.gravity),
                pose_prior.pose_prior_id
            ],
        )?;
        Ok(())
    }

    pub fn upsert_keypoints(&self, image_id: ImageId, keypoints: &[ColmapKeypoint]) -> Result<()> {
        if self.exists_keypoints(image_id)? {
            self.update_keypoints(image_id, keypoints)
        } else {
            self.write_keypoints(image_id, keypoints)
        }
    }

    pub fn upsert_descriptors(
        &self,
        image_id: ImageId,
        descriptors: &ColmapDescriptors,
    ) -> Result<()> {
        if self.exists_descriptors(image_id)? {
            self.conn.execute(
                "UPDATE descriptors SET rows = ?1, cols = ?2, data = ?3, type = ?4 WHERE image_id = ?5;",
                params![
                    descriptors.rows as i64,
                    descriptors.cols as i64,
                    descriptors.data,
                    descriptors.feature_type,
                    image_id
                ],
            )?;
            Ok(())
        } else {
            self.write_descriptors(image_id, descriptors)
        }
    }

    pub fn update_keypoints(&self, image_id: ImageId, keypoints: &[ColmapKeypoint]) -> Result<()> {
        let (rows, cols, data) = encode_keypoints_blob(keypoints);
        self.update_keypoints_blob(image_id, &ColmapKeypointsBlob { rows, cols, data })
    }

    pub fn update_keypoints_blob(
        &self,
        image_id: ImageId,
        blob: &ColmapKeypointsBlob,
    ) -> Result<()> {
        validate_keypoints_blob(blob)?;
        self.conn.execute(
            "UPDATE keypoints SET rows = ?1, cols = ?2, data = ?3 WHERE image_id = ?4;",
            params![blob.rows as i64, blob.cols as i64, &blob.data, image_id],
        )?;
        Ok(())
    }

    pub fn update_two_view_geometry(
        &self,
        image_id1: ImageId,
        image_id2: ImageId,
        geometry: &ColmapTwoViewGeometry,
    ) -> Result<()> {
        if self.exists_two_view_geometry(image_id1, image_id2)? {
            self.delete_two_view_geometry(image_id1, image_id2)?;
            self.write_two_view_geometry(image_id1, image_id2, geometry)?;
        }
        Ok(())
    }

    pub fn delete_matches(&self, image_id1: ImageId, image_id2: ImageId) -> Result<()> {
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)
            .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        self.conn.execute(
            "DELETE FROM matches WHERE pair_id = ?1;",
            params![pair_id as i64],
        )?;
        self.mark_database_entry_deleted();
        Ok(())
    }

    pub fn delete_two_view_geometry(&self, image_id1: ImageId, image_id2: ImageId) -> Result<()> {
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)
            .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        self.conn.execute(
            "DELETE FROM two_view_geometries WHERE pair_id = ?1;",
            params![pair_id as i64],
        )?;
        self.mark_database_entry_deleted();
        Ok(())
    }

    pub fn delete_inlier_matches(&self, image_id1: ImageId, image_id2: ImageId) -> Result<()> {
        if !self.exists_two_view_geometry(image_id1, image_id2)? {
            return Ok(());
        }
        let mut geometry = self.read_two_view_geometry(image_id1, image_id2)?;
        geometry.inlier_matches.clear();
        self.update_two_view_geometry(image_id1, image_id2, &geometry)
    }

    pub fn clear_all_tables(&self) -> Result<()> {
        self.clear_matches()?;
        self.clear_two_view_geometries()?;
        self.clear_descriptors()?;
        self.clear_keypoints()?;
        self.clear_pose_priors()?;
        self.clear_frames()?;
        self.clear_images()?;
        self.clear_rigs()?;
        self.clear_cameras()?;
        Ok(())
    }

    pub fn clear_rigs(&self) -> Result<()> {
        self.conn
            .execute_batch("DELETE FROM rigs; DELETE FROM rig_sensors;")?;
        self.mark_database_entry_deleted();
        Ok(())
    }

    pub fn clear_cameras(&self) -> Result<()> {
        self.conn.execute_batch("DELETE FROM cameras;")?;
        self.mark_database_entry_deleted();
        Ok(())
    }

    pub fn clear_frames(&self) -> Result<()> {
        self.conn
            .execute_batch("DELETE FROM frames; DELETE FROM frame_data;")?;
        self.mark_database_entry_deleted();
        Ok(())
    }

    pub fn clear_images(&self) -> Result<()> {
        self.conn.execute_batch("DELETE FROM images;")?;
        self.mark_database_entry_deleted();
        Ok(())
    }

    pub fn clear_pose_priors(&self) -> Result<()> {
        self.conn.execute_batch("DELETE FROM pose_priors;")?;
        self.mark_database_entry_deleted();
        Ok(())
    }

    pub fn clear_keypoints(&self) -> Result<()> {
        self.conn.execute_batch("DELETE FROM keypoints;")?;
        self.mark_database_entry_deleted();
        Ok(())
    }

    pub fn clear_descriptors(&self) -> Result<()> {
        self.conn.execute_batch("DELETE FROM descriptors;")?;
        self.mark_database_entry_deleted();
        Ok(())
    }

    pub fn clear_matches(&self) -> Result<()> {
        self.conn.execute_batch("DELETE FROM matches;")?;
        self.mark_database_entry_deleted();
        Ok(())
    }

    pub fn clear_two_view_geometries(&self) -> Result<()> {
        self.conn
            .execute_batch("DELETE FROM two_view_geometries;")?;
        self.mark_database_entry_deleted();
        Ok(())
    }

    fn mark_database_entry_deleted(&self) {
        self.database_entry_deleted.set(true);
    }

    pub fn read_two_view_geometry(
        &self,
        image_id1: ImageId,
        image_id2: ImageId,
    ) -> Result<ColmapTwoViewGeometry> {
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)
            .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        let row = self
            .conn
            .query_row(
                "SELECT rows, cols, data, config, F, E, H, qvec, tvec
                 FROM two_view_geometries WHERE pair_id = ?1;",
                params![pair_id as i64],
                |row| {
                    Ok((
                        checked_row_integer(row, 0)?,
                        checked_row_integer(row, 1)?,
                        row.get::<_, Vec<u8>>(2)?,
                        row.get::<_, i32>(3)?,
                        row.get::<_, Option<Vec<u8>>>(4)?,
                        row.get::<_, Option<Vec<u8>>>(5)?,
                        row.get::<_, Option<Vec<u8>>>(6)?,
                        row.get::<_, Option<Vec<u8>>>(7)?,
                        row.get::<_, Option<Vec<u8>>>(8)?,
                    ))
                },
            )
            .optional()?;
        let Some((rows, cols, data, config, f_blob, e_blob, h_blob, q_blob, t_blob)) = row else {
            return Ok(ColmapTwoViewGeometry::default());
        };
        let mut geometry = ColmapTwoViewGeometry {
            config,
            inlier_matches: decode_matches_blob(rows, cols, &data)?,
            f_matrix: f_blob
                .as_deref()
                .map(decode_matrix3_colmap_blob)
                .transpose()?,
            e_matrix: e_blob
                .as_deref()
                .map(decode_matrix3_colmap_blob)
                .transpose()?,
            h_matrix: h_blob
                .as_deref()
                .map(decode_matrix3_colmap_blob)
                .transpose()?,
            qvec: q_blob.as_deref().map(decode_vec4_blob).transpose()?,
            tvec: t_blob.as_deref().map(decode_vec3_blob).transpose()?,
        };
        if should_swap_image_pair(image_id1, image_id2) {
            geometry.invert();
        }
        Ok(geometry)
    }

    pub fn read_two_view_geometry_num_inliers(&self) -> Result<Vec<(ImagePairId, i32)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0;")?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push((checked_row_integer(row, 0)?, checked_row_integer(row, 1)?));
        }
        Ok(out)
    }

    pub fn read_two_view_geometries(&self) -> Result<Vec<(ImagePairId, ColmapTwoViewGeometry)>> {
        let mut stmt = self.conn.prepare(
            "SELECT pair_id, rows, cols, data, config, F, E, H, qvec, tvec
             FROM two_view_geometries
             WHERE rows > 0 OR F IS NOT NULL OR E IS NOT NULL OR H IS NOT NULL
                OR qvec IS NOT NULL OR tvec IS NOT NULL;",
        )?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            let pair_id = checked_row_integer(row, 0)?;
            let match_rows = checked_row_integer(row, 1)?;
            let cols = checked_row_integer(row, 2)?;
            let data = row.get::<_, Vec<u8>>(3)?;
            out.push((
                pair_id,
                ColmapTwoViewGeometry {
                    config: row.get::<_, i32>(4)?,
                    inlier_matches: decode_matches_blob(match_rows, cols, &data)?,
                    f_matrix: row
                        .get::<_, Option<Vec<u8>>>(5)?
                        .as_deref()
                        .map(decode_matrix3_colmap_blob)
                        .transpose()?,
                    e_matrix: row
                        .get::<_, Option<Vec<u8>>>(6)?
                        .as_deref()
                        .map(decode_matrix3_colmap_blob)
                        .transpose()?,
                    h_matrix: row
                        .get::<_, Option<Vec<u8>>>(7)?
                        .as_deref()
                        .map(decode_matrix3_colmap_blob)
                        .transpose()?,
                    qvec: row
                        .get::<_, Option<Vec<u8>>>(8)?
                        .as_deref()
                        .map(decode_vec4_blob)
                        .transpose()?,
                    tvec: row
                        .get::<_, Option<Vec<u8>>>(9)?
                        .as_deref()
                        .map(decode_vec3_blob)
                        .transpose()?,
                },
            ));
        }
        Ok(out)
    }

    pub fn build_correspondence_graph(&self) -> Result<CorrespondenceGraph> {
        let mut graph = CorrespondenceGraph::new();
        for (image_id, num_points2d) in self.read_keypoint_counts()? {
            graph
                .add_image(image_id, num_points2d)
                .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        }
        for (pair_id, geometry) in self.read_two_view_geometries()? {
            let (image_id1, image_id2) =
                pair_id_to_image_pair(pair_id).map_err(|err| anyhow::anyhow!("{err:?}"))?;
            if !graph.exists_image(image_id1) || !graph.exists_image(image_id2) {
                continue;
            }
            graph
                .add_two_view_geometry(
                    image_id1,
                    image_id2,
                    colmap_two_view_geometry_to_record(geometry),
                )
                .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        }
        graph.finalize().map_err(|err| anyhow::anyhow!("{err:?}"))?;
        Ok(graph)
    }

    pub fn load_cache(&self, options: &DatabaseCacheOptions) -> Result<DatabaseCache> {
        let mut rigs = self
            .read_all_rigs()?
            .into_iter()
            .map(|rig| (rig.rig_id, rig))
            .collect::<BTreeMap<_, _>>();
        let mut cameras = self
            .read_all_cameras()?
            .into_iter()
            .map(|camera| (camera.camera.camera_id, camera))
            .collect::<BTreeMap<_, _>>();
        let mut frames = self
            .read_all_frames()?
            .into_iter()
            .map(|frame| (frame.frame_id, frame))
            .collect::<BTreeMap<_, _>>();
        let mut images = self
            .read_all_images()?
            .into_iter()
            .map(|image| (image.image_id, image))
            .collect::<BTreeMap<_, _>>();
        if rigs.is_empty() {
            for camera in cameras.values() {
                let sensor_id = ColmapSensorId {
                    sensor_type: ColmapSensorType::Camera,
                    sensor_id: camera.camera.camera_id,
                };
                rigs.insert(
                    camera.camera.camera_id,
                    ColmapRig {
                        rig_id: camera.camera.camera_id,
                        ref_sensor_id: Some(sensor_id.clone()),
                        sensors: vec![ColmapRigSensor {
                            sensor_id,
                            sensor_from_rig: None,
                        }],
                    },
                );
            }
        }
        if frames.is_empty() {
            for image in images.values_mut() {
                let frame_id = image.image_id;
                image.frame_id = Some(frame_id);
                frames.insert(
                    frame_id,
                    ColmapDatabaseFrame {
                        frame_id,
                        rig_id: image.camera_id,
                        data_ids: vec![ColmapDataId {
                            sensor_id: ColmapSensorId {
                                sensor_type: ColmapSensorType::Camera,
                                sensor_id: image.camera_id,
                            },
                            data_id: image.image_id as u64,
                        }],
                    },
                );
            }
        }
        let keypoint_counts = self.read_keypoint_counts()?;
        let two_view_geometries = self.read_two_view_geometries()?;
        let image_to_frame = images
            .iter()
            .map(|(&image_id, image)| (image_id, image.frame_id.unwrap_or(image_id)))
            .collect::<BTreeMap<_, _>>();
        let candidate_frames = candidate_frame_ids(options, &images);

        if !options.load_all_images {
            let connected_frames = connected_frames_from_geometries(
                options,
                &two_view_geometries,
                &image_to_frame,
                &candidate_frames,
            )?;
            images.retain(|_, image| {
                let frame_id = image.frame_id.unwrap_or(image.image_id);
                candidate_frames.contains(&frame_id) && connected_frames.contains(&frame_id)
            });
        } else {
            images.retain(|_, image| {
                let frame_id = image.frame_id.unwrap_or(image.image_id);
                candidate_frames.contains(&frame_id)
            });
        }
        let loaded_frame_ids = images
            .values()
            .map(|image| image.frame_id.unwrap_or(image.image_id))
            .collect::<BTreeSet<_>>();
        frames.retain(|frame_id, _| loaded_frame_ids.contains(frame_id));
        let loaded_camera_ids = images
            .values()
            .map(|image| image.camera_id)
            .collect::<BTreeSet<_>>();
        cameras.retain(|camera_id, _| loaded_camera_ids.contains(camera_id));
        let loaded_rig_ids = frames
            .values()
            .map(|frame| frame.rig_id)
            .collect::<BTreeSet<_>>();
        rigs.retain(|rig_id, _| loaded_rig_ids.contains(rig_id));

        let mut graph = CorrespondenceGraph::new();
        for (image_id, num_points2d) in keypoint_counts {
            if images.contains_key(&image_id) {
                graph
                    .add_image(image_id, num_points2d)
                    .map_err(|err| anyhow::anyhow!("{err:?}"))?;
            }
        }
        for (pair_id, geometry) in two_view_geometries {
            if !use_inlier_matches(options, geometry.config, geometry.inlier_matches.len()) {
                continue;
            }
            let (image_id1, image_id2) =
                pair_id_to_image_pair(pair_id).map_err(|err| anyhow::anyhow!("{err:?}"))?;
            if !graph.exists_image(image_id1) || !graph.exists_image(image_id2) {
                continue;
            }
            graph
                .add_two_view_geometry(
                    image_id1,
                    image_id2,
                    colmap_two_view_geometry_to_record(geometry),
                )
                .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        }
        graph.finalize().map_err(|err| anyhow::anyhow!("{err:?}"))?;

        let mut pose_priors = self.read_all_pose_priors()?;
        if options.convert_pose_priors_to_enu {
            convert_pose_priors_to_enu(&mut pose_priors)?;
        }

        Ok(DatabaseCache {
            rigs,
            cameras,
            frames,
            images,
            pose_priors,
            correspondence_graph: graph,
        })
    }
}

impl Drop for ColmapDatabase {
    fn drop(&mut self) {
        if self.database_entry_deleted.get() {
            let _ = self.conn.execute_batch("VACUUM;");
            self.database_entry_deleted.set(false);
        }
    }
}

pub fn load_random_database_descriptors(
    database: &ColmapDatabase,
    max_num_descriptors: i32,
) -> Result<ColmapDescriptorsFloat> {
    let images = database.read_all_images()?;
    let total_num_descriptors = database.num_descriptors()?;
    if total_num_descriptors == 0 {
        return Ok(ColmapDescriptorsFloat::empty());
    }

    let mut descriptor_idxs =
        if max_num_descriptors < 0 || max_num_descriptors as usize >= total_num_descriptors {
            (0..total_num_descriptors).collect::<Vec<_>>()
        } else {
            let all_idxs = (0..total_num_descriptors).collect::<Vec<_>>();
            let mut rng = rand::thread_rng();
            all_idxs
                .choose_multiple(&mut rng, max_num_descriptors as usize)
                .copied()
                .collect::<Vec<_>>()
        };
    descriptor_idxs.sort_unstable();

    let mut result = ColmapDescriptorsFloat::empty();
    let mut image_idx = 0usize;
    let mut image_descriptors = ColmapDescriptors::new(COLMAP_FEATURE_UNDEFINED, 0, 0, Vec::new())?;
    let mut image_descriptor_start = 0usize;
    let mut image_descriptor_end = 0usize;

    for descriptor_idx in descriptor_idxs {
        while descriptor_idx >= image_descriptor_end {
            let image = images
                .get(image_idx)
                .with_context(|| "descriptor rows exceed images in database")?;
            image_descriptors = database.read_descriptors(image.image_id)?;
            image_descriptor_start = image_descriptor_end;
            image_descriptor_end = image_descriptor_start + image_descriptors.rows;
            image_idx += 1;
        }

        if result.feature_type == COLMAP_FEATURE_UNDEFINED {
            if image_descriptors.feature_type == COLMAP_FEATURE_UNDEFINED {
                bail!("database descriptors have undefined feature type");
            }
            result.feature_type = image_descriptors.feature_type;
            result.cols = image_descriptors.cols;
            result.data.reserve(result.cols * (result.rows + 1));
        } else {
            if result.feature_type != image_descriptors.feature_type {
                bail!("all images must have the same feature type");
            }
            if result.cols != image_descriptors.cols {
                bail!("all images must have the same descriptor dimensionality");
            }
        }

        let local_row = descriptor_idx - image_descriptor_start;
        let begin = local_row * image_descriptors.cols;
        let end = begin + image_descriptors.cols;
        result.data.extend(
            image_descriptors.data[begin..end]
                .iter()
                .map(|&value| value as f32),
        );
        result.rows += 1;
    }

    Ok(result)
}

fn colmap_two_view_geometry_to_record(geometry: ColmapTwoViewGeometry) -> TwoViewGeometryRecord {
    TwoViewGeometryRecord {
        config: geometry.config,
        inlier_matches: geometry.inlier_matches,
        f_matrix: geometry.f_matrix,
        e_matrix: geometry.e_matrix,
        h_matrix: geometry.h_matrix,
        qvec: geometry.qvec,
        tvec: geometry.tvec,
    }
}

fn merge_database_side(
    source: &ColmapDatabase,
    target: &ColmapDatabase,
) -> Result<(
    HashMap<u32, u32>,
    HashMap<u32, u32>,
    HashMap<ImageId, ImageId>,
)> {
    let mut camera_ids = HashMap::new();
    for mut camera in source.read_all_cameras()? {
        let old_camera_id = camera.camera.camera_id;
        camera.camera.camera_id = 0;
        let new_camera_id = target.write_camera(&camera, false)?;
        camera_ids.insert(old_camera_id, new_camera_id);
    }

    let mut rig_ids = HashMap::new();
    for rig in source.read_all_rigs()? {
        let old_rig_id = rig.rig_id;
        let updated_rig = remap_rig_for_merge(&rig, &camera_ids)?;
        let new_rig_id = target.write_rig(&updated_rig, false)?;
        rig_ids.insert(old_rig_id, new_rig_id);
    }

    let mut image_ids = HashMap::new();
    for image in source.read_all_images()? {
        if target.exists_image_with_name(&image.name)? {
            bail!(
                "the two databases must not contain images with the same name: {}",
                image.name
            );
        }
        let new_camera_id = remap_id(&camera_ids, image.camera_id, "camera")?;
        let new_image_id = target.write_image(
            &ColmapDatabaseImage {
                image_id: 0,
                name: image.name.clone(),
                camera_id: new_camera_id,
                frame_id: None,
            },
            false,
        )?;
        image_ids.insert(image.image_id, new_image_id);
        target.write_keypoints(new_image_id, &source.read_keypoints(image.image_id)?)?;
        target.write_descriptors(new_image_id, &source.read_descriptors(image.image_id)?)?;
    }

    Ok((camera_ids, rig_ids, image_ids))
}

fn merge_database_frames(
    source: &ColmapDatabase,
    target: &ColmapDatabase,
    camera_ids: &HashMap<u32, u32>,
    rig_ids: &HashMap<u32, u32>,
    image_ids: &HashMap<ImageId, ImageId>,
) -> Result<()> {
    for frame in source.read_all_frames()? {
        target.write_frame(
            &ColmapDatabaseFrame {
                frame_id: 0,
                rig_id: remap_id(rig_ids, frame.rig_id, "rig")?,
                data_ids: frame
                    .data_ids
                    .iter()
                    .map(|data_id| remap_data_id_for_merge(data_id, camera_ids, image_ids))
                    .collect::<Result<Vec<_>>>()?,
            },
            false,
        )?;
    }
    Ok(())
}

fn merge_database_pose_priors(
    source: &ColmapDatabase,
    target: &ColmapDatabase,
    camera_ids: &HashMap<u32, u32>,
    image_ids: &HashMap<ImageId, ImageId>,
) -> Result<()> {
    for pose_prior in source.read_all_pose_priors()? {
        let mut updated = pose_prior;
        updated.pose_prior_id = 0;
        updated.corr_data_id =
            remap_data_id_for_merge(&updated.corr_data_id, camera_ids, image_ids)?;
        target.write_pose_prior(&updated, false)?;
    }
    Ok(())
}

fn merge_database_matches(
    source: &ColmapDatabase,
    target: &ColmapDatabase,
    image_ids: &HashMap<ImageId, ImageId>,
) -> Result<()> {
    for (pair_id, matches) in source.read_all_matches()? {
        let (image_id1, image_id2) =
            pair_id_to_image_pair(pair_id).map_err(|err| anyhow::anyhow!("{err:?}"))?;
        target.write_matches(
            remap_id(image_ids, image_id1, "image")?,
            remap_id(image_ids, image_id2, "image")?,
            &matches,
        )?;
    }
    Ok(())
}

fn merge_database_two_view_geometries(
    source: &ColmapDatabase,
    target: &ColmapDatabase,
    image_ids: &HashMap<ImageId, ImageId>,
) -> Result<()> {
    for (pair_id, geometry) in source.read_two_view_geometries()? {
        let (image_id1, image_id2) =
            pair_id_to_image_pair(pair_id).map_err(|err| anyhow::anyhow!("{err:?}"))?;
        target.write_two_view_geometry(
            remap_id(image_ids, image_id1, "image")?,
            remap_id(image_ids, image_id2, "image")?,
            &geometry,
        )?;
    }
    Ok(())
}

fn remap_rig_for_merge(rig: &ColmapRig, camera_ids: &HashMap<u32, u32>) -> Result<ColmapRig> {
    let ref_sensor_id = rig
        .ref_sensor_id
        .as_ref()
        .context("COLMAP database rig requires a reference sensor")?;
    let updated_ref_sensor_id = remap_sensor_id_for_merge(ref_sensor_id, camera_ids)?;
    let mut sensors = vec![ColmapRigSensor {
        sensor_id: updated_ref_sensor_id.clone(),
        sensor_from_rig: None,
    }];
    for sensor in &rig.sensors {
        if &sensor.sensor_id == ref_sensor_id {
            continue;
        }
        sensors.push(ColmapRigSensor {
            sensor_id: remap_sensor_id_for_merge(&sensor.sensor_id, camera_ids)?,
            sensor_from_rig: sensor.sensor_from_rig.clone(),
        });
    }
    Ok(ColmapRig {
        rig_id: 0,
        ref_sensor_id: Some(updated_ref_sensor_id),
        sensors,
    })
}

fn remap_data_id_for_merge(
    data_id: &ColmapDataId,
    camera_ids: &HashMap<u32, u32>,
    image_ids: &HashMap<ImageId, ImageId>,
) -> Result<ColmapDataId> {
    match data_id.sensor_id.sensor_type {
        ColmapSensorType::Camera => Ok(ColmapDataId {
            sensor_id: ColmapSensorId {
                sensor_type: ColmapSensorType::Camera,
                sensor_id: remap_id(camera_ids, data_id.sensor_id.sensor_id, "camera")?,
            },
            data_id: remap_id(image_ids, data_id.data_id as ImageId, "image")? as u64,
        }),
        _ => bail!(
            "data type not supported for COLMAP database merge: {:?}",
            data_id.sensor_id.sensor_type
        ),
    }
}

fn remap_sensor_id_for_merge(
    sensor_id: &ColmapSensorId,
    camera_ids: &HashMap<u32, u32>,
) -> Result<ColmapSensorId> {
    match sensor_id.sensor_type {
        ColmapSensorType::Camera => Ok(ColmapSensorId {
            sensor_type: ColmapSensorType::Camera,
            sensor_id: remap_id(camera_ids, sensor_id.sensor_id, "camera")?,
        }),
        _ => bail!(
            "sensor type not supported for COLMAP database merge: {:?}",
            sensor_id.sensor_type
        ),
    }
}

fn remap_id(ids: &HashMap<u32, u32>, old_id: u32, kind: &str) -> Result<u32> {
    ids.get(&old_id)
        .copied()
        .with_context(|| format!("missing remapped {kind} id for {old_id}"))
}

fn read_camera_row(row: &Row<'_>) -> rusqlite::Result<ColmapDatabaseCamera> {
    let camera_id = checked_row_integer(row, 0)?;
    let model_id = checked_row_integer(row, 1)?;
    let width = checked_row_integer(row, 2)?;
    let height = checked_row_integer(row, 3)?;
    let params_blob = row.get::<_, Vec<u8>>(4)?;
    let params = decode_f64_values_vec(&params_blob).map_err(to_sql_error)?;
    let camera = ColmapCamera {
        camera_id,
        model_id,
        width,
        height,
        params,
    };
    validate_camera_params(&camera).map_err(to_sql_error)?;
    Ok(ColmapDatabaseCamera {
        camera,
        has_prior_focal_length: row.get::<_, i64>(5)? != 0,
    })
}

fn use_inlier_matches(
    options: &DatabaseCacheOptions,
    two_view_geometry_config: i32,
    num_matches: usize,
) -> bool {
    num_matches >= options.min_num_matches
        && (!options.ignore_watermarks || two_view_geometry_config != COLMAP_TWO_VIEW_WATERMARK)
}

pub fn is_colmap_two_view_geometry_with_inliers(config: i32) -> bool {
    matches!(
        config,
        COLMAP_TWO_VIEW_CALIBRATED
            | COLMAP_TWO_VIEW_UNCALIBRATED
            | COLMAP_TWO_VIEW_PLANAR
            | COLMAP_TWO_VIEW_PANORAMIC
            | COLMAP_TWO_VIEW_PLANAR_OR_PANORAMIC
            | COLMAP_TWO_VIEW_WATERMARK
            | COLMAP_TWO_VIEW_MULTIPLE
            | COLMAP_TWO_VIEW_CALIBRATED_RIG
    )
}

fn convert_pose_priors_to_enu(pose_priors: &mut [ColmapPosePrior]) -> Result<()> {
    let mut coordinate_system = None;
    let mut gps_positions = Vec::new();
    for pose_prior in pose_priors.iter() {
        if let Some(existing) = coordinate_system.as_ref() {
            if existing != &pose_prior.coordinate_system {
                bail!("inconsistent coordinate systems defined in pose priors");
            }
        } else {
            coordinate_system = Some(pose_prior.coordinate_system.clone());
        }
        if pose_prior.coordinate_system == ColmapPosePriorCoordinateSystem::Wgs84 {
            gps_positions.push(pose_prior.position);
        }
    }

    if coordinate_system == Some(ColmapPosePriorCoordinateSystem::Wgs84)
        && !gps_positions.is_empty()
    {
        let ref_lat_lon_alt = gps_positions[0];
        let enu_positions = wgs84_ellipsoid_to_enu(&gps_positions, ref_lat_lon_alt)?;
        for (pose_prior, enu_position) in pose_priors.iter_mut().zip(enu_positions) {
            pose_prior.position = enu_position;
            pose_prior.coordinate_system = ColmapPosePriorCoordinateSystem::Cartesian;
        }
    }
    Ok(())
}

fn wgs84_ellipsoid_to_enu(
    lat_lon_alt: &[[f64; 3]],
    ref_lat_lon_alt: [f64; 3],
) -> Result<Vec<[f64; 3]>> {
    let xyz_in_ecef = lat_lon_alt
        .iter()
        .map(|position| wgs84_ellipsoid_to_ecef(*position))
        .collect::<Result<Vec<_>>>()?;
    let ref_ecef = wgs84_ellipsoid_to_ecef(ref_lat_lon_alt)?;
    let ref_ellipsoid = wgs84_ecef_to_ellipsoid(ref_ecef)?;
    Ok(ecef_to_enu(
        &xyz_in_ecef,
        ref_ecef,
        ref_ellipsoid[0],
        ref_ellipsoid[1],
    ))
}

fn wgs84_ellipsoid_to_ecef(lat_lon_alt: [f64; 3]) -> Result<[f64; 3]> {
    if !lat_lon_alt.iter().all(|value| value.is_finite()) {
        bail!("WGS84 pose prior position must be finite for ENU conversion");
    }
    const WGS84_A: f64 = 6_378_137.0;
    const WGS84_F: f64 = 1.0 / 298.257_223_563;
    let e2 = WGS84_F * (2.0 - WGS84_F);
    let lat = lat_lon_alt[0].to_radians();
    let lon = lat_lon_alt[1].to_radians();
    let alt = lat_lon_alt[2];
    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let sin_lon = lon.sin();
    let cos_lon = lon.cos();
    let n = WGS84_A / (1.0 - e2 * sin_lat * sin_lat).sqrt();
    Ok([
        (n + alt) * cos_lat * cos_lon,
        (n + alt) * cos_lat * sin_lon,
        (n * (1.0 - e2) + alt) * sin_lat,
    ])
}

fn wgs84_ecef_to_ellipsoid(xyz_in_ecef: [f64; 3]) -> Result<[f64; 3]> {
    if !xyz_in_ecef.iter().all(|value| value.is_finite()) {
        bail!("ECEF position must be finite for WGS84 conversion");
    }
    const WGS84_A: f64 = 6_378_137.0;
    const WGS84_F: f64 = 1.0 / 298.257_223_563;
    let e2 = WGS84_F * (2.0 - WGS84_F);
    let x = xyz_in_ecef[0];
    let y = xyz_in_ecef[1];
    let z = xyz_in_ecef[2];
    let radius_xy = (x * x + y * y).sqrt();
    if radius_xy == 0.0 {
        bail!("cannot convert polar ECEF reference to WGS84 ellipsoid");
    }
    let mut lat = z.atan2(radius_xy);
    let mut alt = 0.0;
    for _ in 0..100 {
        let sin_lat = lat.sin();
        let n = WGS84_A / (1.0 - e2 * sin_lat * sin_lat).sqrt();
        let prev_alt = alt;
        alt = radius_xy / lat.cos() - n;
        let prev_lat = lat;
        lat = ((z / radius_xy) / (1.0 - e2 * n / (n + alt))).atan();
        if (prev_lat - lat).abs() < 1.0e-12 && (prev_alt - alt).abs() < 1.0e-12 {
            break;
        }
    }
    Ok([lat.to_degrees(), y.atan2(x).to_degrees(), alt])
}

fn ecef_to_enu(
    xyz_in_ecef: &[[f64; 3]],
    ref_ecef: [f64; 3],
    ref_lat: f64,
    ref_lon: f64,
) -> Vec<[f64; 3]> {
    let cos_lat = ref_lat.to_radians().cos();
    let sin_lat = ref_lat.to_radians().sin();
    let cos_lon = ref_lon.to_radians().cos();
    let sin_lon = ref_lon.to_radians().sin();

    xyz_in_ecef
        .iter()
        .map(|xyz| {
            let dx = xyz[0] - ref_ecef[0];
            let dy = xyz[1] - ref_ecef[1];
            let dz = xyz[2] - ref_ecef[2];
            [
                -sin_lon * dx + cos_lon * dy,
                -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz,
                cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz,
            ]
        })
        .collect()
}

fn candidate_frame_ids(
    options: &DatabaseCacheOptions,
    images: &BTreeMap<ImageId, ColmapDatabaseImage>,
) -> BTreeSet<u32> {
    images
        .values()
        .filter(|image| options.image_names.is_empty() || options.image_names.contains(&image.name))
        .map(|image| image.frame_id.unwrap_or(image.image_id))
        .collect()
}

fn connected_frames_from_geometries(
    options: &DatabaseCacheOptions,
    geometries: &[(ImagePairId, ColmapTwoViewGeometry)],
    image_to_frame: &BTreeMap<ImageId, u32>,
    candidate_frames: &BTreeSet<u32>,
) -> Result<BTreeSet<u32>> {
    let mut connected = BTreeSet::new();
    for (pair_id, geometry) in geometries {
        if !use_inlier_matches(options, geometry.config, geometry.inlier_matches.len()) {
            continue;
        }
        let (image_id1, image_id2) =
            pair_id_to_image_pair(*pair_id).map_err(|err| anyhow::anyhow!("{err:?}"))?;
        let Some(&frame_id1) = image_to_frame.get(&image_id1) else {
            continue;
        };
        let Some(&frame_id2) = image_to_frame.get(&image_id2) else {
            continue;
        };
        if candidate_frames.contains(&frame_id1) && candidate_frames.contains(&frame_id2) {
            connected.insert(frame_id1);
            connected.insert(frame_id2);
        }
    }
    Ok(connected)
}

fn read_image_row(row: &Row<'_>) -> rusqlite::Result<ColmapDatabaseImage> {
    Ok(ColmapDatabaseImage {
        image_id: checked_row_integer(row, 0)?,
        name: row.get(1)?,
        camera_id: checked_row_integer(row, 2)?,
        frame_id: checked_optional_row_integer(row, 3)?,
    })
}

fn read_pose_prior_row(row: &Row<'_>) -> rusqlite::Result<ColmapPosePrior> {
    let position_blob = row.get::<_, Vec<u8>>(4)?;
    let position_covariance_blob = row.get::<_, Vec<u8>>(5)?;
    let gravity_blob = row.get::<_, Vec<u8>>(7)?;
    Ok(ColmapPosePrior {
        pose_prior_id: checked_row_integer(row, 0)?,
        corr_data_id: ColmapDataId {
            data_id: checked_row_integer(row, 1)?,
            sensor_id: ColmapSensorId {
                sensor_id: checked_row_integer(row, 2)?,
                sensor_type: sensor_type_from_i64(row.get::<_, i64>(3)?),
            },
        },
        position: decode_f64_values::<3>(&position_blob).map_err(to_sql_error)?,
        position_covariance: decode_f64_values::<9>(&position_covariance_blob)
            .map_err(to_sql_error)?,
        coordinate_system: coordinate_system_from_i64(row.get::<_, i64>(6)?),
        gravity: decode_f64_values::<3>(&gravity_blob).map_err(to_sql_error)?,
    })
}

fn collect_rig_rows(rows: &mut Rows<'_>) -> Result<Vec<ColmapRig>> {
    let mut rigs = BTreeMap::<u32, ColmapRig>::new();
    while let Some(row) = rows.next()? {
        let rig_id = checked_row_integer(row, 0)?;
        if let std::collections::btree_map::Entry::Vacant(entry) = rigs.entry(rig_id) {
            entry.insert(ColmapRig {
                rig_id,
                ref_sensor_id: Some(ColmapSensorId {
                    sensor_id: checked_row_integer(row, 1)?,
                    sensor_type: sensor_type_from_i64(row.get::<_, i64>(2)?),
                }),
                sensors: Vec::new(),
            });
        }
        let rig = rigs.get_mut(&rig_id).expect("rig inserted");
        push_rig_sensor_from_row(rig, row)?;
    }
    Ok(rigs.into_values().collect())
}

fn push_rig_sensor_from_row(rig: &mut ColmapRig, row: &Row<'_>) -> Result<()> {
    let sensor_id = row.get::<_, Option<i64>>(3)?;
    let Some(sensor_id) = sensor_id else {
        return Ok(());
    };
    let sensor_from_rig = row
        .get::<_, Option<Vec<u8>>>(5)?
        .as_deref()
        .map(decode_rigid3_blob)
        .transpose()?;
    rig.sensors.push(ColmapRigSensor {
        sensor_id: ColmapSensorId {
            sensor_id: checked_integer(sensor_id, 3)?,
            sensor_type: sensor_type_from_i64(row.get::<_, i64>(4)?),
        },
        sensor_from_rig,
    });
    Ok(())
}

fn collect_frame_rows(rows: &mut Rows<'_>) -> Result<Vec<ColmapDatabaseFrame>> {
    let mut frames = BTreeMap::<u32, ColmapDatabaseFrame>::new();
    while let Some(row) = rows.next()? {
        let frame_id = checked_row_integer(row, 0)?;
        if let std::collections::btree_map::Entry::Vacant(entry) = frames.entry(frame_id) {
            entry.insert(ColmapDatabaseFrame {
                frame_id,
                rig_id: checked_row_integer(row, 1)?,
                data_ids: Vec::new(),
            });
        }
        let frame = frames.get_mut(&frame_id).expect("frame inserted");
        push_frame_data_from_row(frame, row)?;
    }
    Ok(frames.into_values().collect())
}

fn push_frame_data_from_row(frame: &mut ColmapDatabaseFrame, row: &Row<'_>) -> Result<()> {
    let data_id = row.get::<_, Option<i64>>(2)?;
    let Some(data_id) = data_id else {
        return Ok(());
    };
    frame.data_ids.push(ColmapDataId {
        data_id: checked_integer(data_id, 2)?,
        sensor_id: ColmapSensorId {
            sensor_id: checked_row_integer(row, 3)?,
            sensor_type: sensor_type_from_i64(row.get::<_, i64>(4)?),
        },
    });
    Ok(())
}

fn write_rig_sensors(conn: &Connection, rig_id: u32, rig: &ColmapRig) -> Result<()> {
    for sensor in &rig.sensors {
        if rig.ref_sensor_id.as_ref() == Some(&sensor.sensor_id) {
            continue;
        }
        let pose_blob = sensor.sensor_from_rig.as_ref().map(encode_rigid3_blob);
        conn.execute(
            "INSERT INTO rig_sensors(rig_id, sensor_id, sensor_type, sensor_from_rig)
             VALUES(?1, ?2, ?3, ?4);",
            params![
                rig_id,
                sensor.sensor_id.sensor_id,
                sensor_type_to_i64(&sensor.sensor_id.sensor_type)?,
                pose_blob
            ],
        )?;
    }
    Ok(())
}

fn write_frame_data(conn: &Connection, frame_id: u32, frame: &ColmapDatabaseFrame) -> Result<()> {
    for data_id in &frame.data_ids {
        conn.execute(
            "INSERT INTO frame_data(frame_id, data_id, sensor_id, sensor_type)
             VALUES(?1, ?2, ?3, ?4);",
            params![
                frame_id,
                data_id.data_id as i64,
                data_id.sensor_id.sensor_id,
                sensor_type_to_i64(&data_id.sensor_id.sensor_type)?
            ],
        )?;
    }
    Ok(())
}

fn pose_prior_legacy_image_id(pose_prior: &ColmapPosePrior) -> Result<Option<i64>> {
    if pose_prior.corr_data_id.sensor_id.sensor_type == ColmapSensorType::Camera {
        Ok(Some(i64::try_from(pose_prior.corr_data_id.data_id)?))
    } else {
        Ok(None)
    }
}

fn sensor_type_to_i64(sensor_type: &ColmapSensorType) -> Result<i64> {
    match sensor_type {
        ColmapSensorType::Invalid => Ok(-1),
        ColmapSensorType::Camera => Ok(0),
        ColmapSensorType::Imu => Ok(1),
        ColmapSensorType::Other(value) => {
            bail!("cannot write non-COLMAP sensor type {value:?} to SQLite database")
        }
    }
}

fn sensor_type_from_i64(value: i64) -> ColmapSensorType {
    match value {
        -1 => ColmapSensorType::Invalid,
        0 => ColmapSensorType::Camera,
        1 => ColmapSensorType::Imu,
        other => ColmapSensorType::Other(other.to_string()),
    }
}

fn coordinate_system_to_i64(coordinate_system: &ColmapPosePriorCoordinateSystem) -> Result<i64> {
    match coordinate_system {
        ColmapPosePriorCoordinateSystem::Undefined => Ok(-1),
        ColmapPosePriorCoordinateSystem::Wgs84 => Ok(0),
        ColmapPosePriorCoordinateSystem::Cartesian => Ok(1),
        ColmapPosePriorCoordinateSystem::Other(value) => {
            bail!("cannot write non-COLMAP pose prior coordinate system {value}")
        }
    }
}

fn coordinate_system_from_i64(value: i64) -> ColmapPosePriorCoordinateSystem {
    match value {
        -1 => ColmapPosePriorCoordinateSystem::Undefined,
        0 => ColmapPosePriorCoordinateSystem::Wgs84,
        1 => ColmapPosePriorCoordinateSystem::Cartesian,
        other => ColmapPosePriorCoordinateSystem::Other(other),
    }
}

fn validate_camera_params(camera: &ColmapCamera) -> Result<()> {
    let expected = colmap_camera_model_num_params(camera.model_id)
        .with_context(|| format!("unsupported COLMAP camera model id {}", camera.model_id))?;
    if camera.params.len() != expected {
        bail!(
            "camera_id={} model_id={} has {} params, expected {}",
            camera.camera_id,
            camera.model_id,
            camera.params.len(),
            expected
        );
    }
    Ok(())
}

fn checked_integer<T>(value: i64, column: usize) -> rusqlite::Result<T>
where
    T: TryFrom<i64>,
    T::Error: std::error::Error + Send + Sync + 'static,
{
    T::try_from(value).map_err(|err| {
        rusqlite::Error::FromSqlConversionFailure(column, Type::Integer, Box::new(err))
    })
}

fn checked_row_integer<T>(row: &Row<'_>, column: usize) -> rusqlite::Result<T>
where
    T: TryFrom<i64>,
    T::Error: std::error::Error + Send + Sync + 'static,
{
    checked_integer(row.get::<_, i64>(column)?, column)
}

fn checked_optional_row_integer<T>(row: &Row<'_>, column: usize) -> rusqlite::Result<Option<T>>
where
    T: TryFrom<i64>,
    T::Error: std::error::Error + Send + Sync + 'static,
{
    row.get::<_, Option<i64>>(column)?
        .map(|value| checked_integer(value, column))
        .transpose()
}

fn to_sql_error(err: anyhow::Error) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(
        0,
        rusqlite::types::Type::Blob,
        Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, err)),
    )
}

fn validate_dynamic_blob(
    label: &str,
    rows: usize,
    cols: usize,
    scalar_size: usize,
    data: &[u8],
) -> Result<()> {
    let expected = rows.saturating_mul(cols).saturating_mul(scalar_size);
    if data.len() != expected {
        bail!(
            "COLMAP {label} blob has {} bytes, expected rows*cols*scalar_size={}*{}*{}={}",
            data.len(),
            rows,
            cols,
            scalar_size,
            expected
        );
    }
    Ok(())
}

fn validate_keypoints_blob(blob: &ColmapKeypointsBlob) -> Result<()> {
    validate_dynamic_blob(
        "keypoints",
        blob.rows,
        blob.cols,
        std::mem::size_of::<f32>(),
        &blob.data,
    )
}

fn validate_matches_blob(blob: &ColmapMatchesBlob) -> Result<()> {
    if blob.cols != 2 {
        bail!(
            "COLMAP match blob has unsupported column count {}",
            blob.cols
        );
    }
    validate_dynamic_blob(
        "matches",
        blob.rows,
        blob.cols,
        std::mem::size_of::<u32>(),
        &blob.data,
    )
}

fn encode_keypoints_blob(keypoints: &[ColmapKeypoint]) -> (usize, usize, Vec<u8>) {
    let mut values = Vec::with_capacity(keypoints.len() * 6);
    for kp in keypoints {
        values.extend_from_slice(&[kp.x, kp.y, kp.a11, kp.a12, kp.a21, kp.a22]);
    }
    (keypoints.len(), 6, encode_f32_blob(&values))
}

fn decode_keypoints_blob(rows: usize, cols: usize, data: &[u8]) -> Result<Vec<ColmapKeypoint>> {
    if rows == 0 && cols == 0 && data.is_empty() {
        return Ok(Vec::new());
    }
    if !matches!(cols, 2 | 4 | 6) {
        bail!("COLMAP keypoint blob has unsupported column count {cols}");
    }
    let values = decode_f32_values(data)?;
    if values.len() != rows.saturating_mul(cols) {
        bail!(
            "COLMAP keypoint blob has {} floats, expected rows*cols={}*{}",
            values.len(),
            rows,
            cols
        );
    }
    let mut keypoints = Vec::with_capacity(rows);
    for row in values.chunks_exact(cols) {
        let keypoint = match cols {
            2 => ColmapKeypoint::new(row[0], row[1]),
            4 => ColmapKeypoint::from_scale_orientation(row[0], row[1], row[2], row[3]),
            6 => ColmapKeypoint {
                x: row[0],
                y: row[1],
                a11: row[2],
                a12: row[3],
                a21: row[4],
                a22: row[5],
            },
            _ => unreachable!(),
        };
        keypoints.push(keypoint);
    }
    Ok(keypoints)
}

fn maybe_swapped_matches_blob(
    image_id1: ImageId,
    image_id2: ImageId,
    blob: &ColmapMatchesBlob,
) -> Result<ColmapMatchesBlob> {
    validate_matches_blob(blob)?;
    if !should_swap_image_pair(image_id1, image_id2) {
        return Ok(blob.clone());
    }
    let mut matches = decode_matches_blob(blob.rows, blob.cols, &blob.data)?;
    swap_matches(&mut matches);
    Ok(ColmapMatchesBlob {
        rows: blob.rows,
        cols: blob.cols,
        data: encode_matches_blob(&matches),
    })
}

fn swap_matches(matches: &mut [FeatureMatch]) {
    for match_ in matches {
        std::mem::swap(&mut match_.point2d_idx1, &mut match_.point2d_idx2);
    }
}

fn encode_matches_blob(matches: &[FeatureMatch]) -> Vec<u8> {
    let mut out = Vec::with_capacity(matches.len() * 2 * std::mem::size_of::<u32>());
    for match_ in matches {
        out.extend_from_slice(&match_.point2d_idx1.to_ne_bytes());
        out.extend_from_slice(&match_.point2d_idx2.to_ne_bytes());
    }
    out
}

fn decode_matches_blob(rows: usize, cols: usize, data: &[u8]) -> Result<Vec<FeatureMatch>> {
    if cols != 2 {
        bail!("COLMAP match blob has unsupported column count {cols}");
    }
    let expected = rows
        .saturating_mul(cols)
        .saturating_mul(std::mem::size_of::<u32>());
    if data.len() != expected {
        bail!(
            "COLMAP match blob has {} bytes, expected {}",
            data.len(),
            expected
        );
    }
    let mut matches = Vec::with_capacity(rows);
    for row in data.chunks_exact(8) {
        matches.push(FeatureMatch {
            point2d_idx1: u32::from_ne_bytes(row[0..4].try_into().expect("chunk length")),
            point2d_idx2: u32::from_ne_bytes(row[4..8].try_into().expect("chunk length")),
        });
    }
    Ok(matches)
}

fn encode_f32_blob(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        out.extend_from_slice(&value.to_ne_bytes());
    }
    out
}

fn decode_f32_values(data: &[u8]) -> Result<Vec<f32>> {
    if data.len() % std::mem::size_of::<f32>() != 0 {
        bail!("float32 blob byte length is not divisible by 4");
    }
    Ok(data
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("chunk length")))
        .collect())
}

fn encode_f64_blob(values: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        out.extend_from_slice(&value.to_ne_bytes());
    }
    out
}

fn decode_f64_values<const N: usize>(data: &[u8]) -> Result<[f64; N]> {
    if data.len() != N * std::mem::size_of::<f64>() {
        bail!(
            "float64 blob byte length {} does not match {}",
            data.len(),
            N * 8
        );
    }
    let mut out = [0.0; N];
    for (idx, chunk) in data.chunks_exact(8).enumerate() {
        out[idx] = f64::from_ne_bytes(chunk.try_into().expect("chunk length"));
    }
    Ok(out)
}

fn decode_static_f64_blob_or_zero<const N: usize>(data: &[u8]) -> Result<[f64; N]> {
    if data.is_empty() {
        return Ok([0.0; N]);
    }
    decode_f64_values(data)
}

fn is_zero_static_f64_blob<const N: usize>(data: &[u8]) -> Result<bool> {
    Ok(decode_static_f64_blob_or_zero::<N>(data)?
        .iter()
        .all(|value| *value == 0.0))
}

fn decode_f64_values_vec(data: &[u8]) -> Result<Vec<f64>> {
    if data.len() % std::mem::size_of::<f64>() != 0 {
        bail!("float64 blob byte length is not divisible by 8");
    }
    Ok(data
        .chunks_exact(8)
        .map(|chunk| f64::from_ne_bytes(chunk.try_into().expect("chunk length")))
        .collect())
}

fn encode_rigid3_blob(rigid: &ColmapRigid3) -> Vec<u8> {
    let mut values = Vec::with_capacity(7);
    values.extend_from_slice(&rigid.qvec);
    values.extend_from_slice(&rigid.tvec);
    encode_f64_blob(&values)
}

fn decode_rigid3_blob(data: &[u8]) -> Result<ColmapRigid3> {
    let values = decode_f64_values::<7>(data)?;
    Ok(ColmapRigid3 {
        qvec: [values[0], values[1], values[2], values[3]],
        tvec: [values[4], values[5], values[6]],
    })
}

fn encode_matrix3_colmap_blob(matrix: [f64; 9]) -> Vec<u8> {
    encode_f64_blob(&transpose3(matrix))
}

fn decode_matrix3_colmap_blob(data: &[u8]) -> Result<[f64; 9]> {
    Ok(transpose3(decode_f64_values::<9>(data)?))
}

fn decode_vec4_blob(data: &[u8]) -> Result<[f64; 4]> {
    decode_f64_values(data)
}

fn decode_vec3_blob(data: &[u8]) -> Result<[f64; 3]> {
    decode_f64_values(data)
}

fn transpose3(matrix: [f64; 9]) -> [f64; 9] {
    [
        matrix[0], matrix[3], matrix[6], matrix[1], matrix[4], matrix[7], matrix[2], matrix[5],
        matrix[8],
    ]
}

fn invert_matrix3(matrix: [f64; 9]) -> Option<[f64; 9]> {
    let inverse = Matrix3::from_row_slice(&matrix).try_inverse()?;
    Some([
        inverse[(0, 0)],
        inverse[(0, 1)],
        inverse[(0, 2)],
        inverse[(1, 0)],
        inverse[(1, 1)],
        inverse[(1, 2)],
        inverse[(2, 0)],
        inverse[(2, 1)],
        inverse[(2, 2)],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn m(point2d_idx1: u32, point2d_idx2: u32) -> FeatureMatch {
        FeatureMatch::new(point2d_idx1, point2d_idx2)
    }

    fn write_test_camera(db: &ColmapDatabase, camera_id: u32) {
        db.write_camera(
            &ColmapDatabaseCamera {
                camera: ColmapCamera {
                    camera_id,
                    model_id: crate::types::COLMAP_PINHOLE,
                    width: 100,
                    height: 100,
                    params: vec![50.0, 50.0, 50.0, 50.0],
                },
                has_prior_focal_length: true,
            },
            true,
        )
        .unwrap();
    }

    fn write_test_images(db: &ColmapDatabase, camera_id: u32, image_ids: &[ImageId]) {
        for &image_id in image_ids {
            db.write_image(
                &ColmapDatabaseImage {
                    image_id,
                    name: format!("{image_id}.jpg"),
                    camera_id,
                    frame_id: None,
                },
                true,
            )
            .unwrap();
        }
    }

    #[test]
    fn cameras_and_images_roundtrip_through_colmap_tables() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();
        let camera = ColmapDatabaseCamera {
            camera: ColmapCamera {
                camera_id: 7,
                model_id: crate::types::COLMAP_PINHOLE,
                width: 1920,
                height: 1080,
                params: vec![1000.0, 1001.0, 960.0, 540.0],
            },
            has_prior_focal_length: true,
        };

        assert_eq!(db.write_camera(&camera, true).unwrap(), 7);
        assert_eq!(db.read_camera(7).unwrap(), Some(camera.clone()));
        assert_eq!(db.read_all_cameras().unwrap(), vec![camera.clone()]);

        let image = ColmapDatabaseImage {
            image_id: 11,
            name: "images/0001.jpg".to_string(),
            camera_id: 7,
            frame_id: None,
        };
        assert_eq!(db.write_image(&image, true).unwrap(), 11);
        assert_eq!(db.read_image(11).unwrap(), Some(image.clone()));
        assert_eq!(
            db.read_image_with_name("images/0001.jpg").unwrap(),
            Some(image.clone())
        );
        assert_eq!(db.read_all_images().unwrap(), vec![image]);
    }

    #[test]
    fn database_read_only_open_preserves_file_bytes() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("database.db");
        {
            let db = ColmapDatabase::open(&path).unwrap();
            write_test_camera(&db, 1);
            write_test_images(&db, 1, &[1]);
        }
        let before = std::fs::read(&path).unwrap();

        {
            let db = ColmapDatabase::open_read_only(&path).unwrap();
            assert_eq!(db.read_all_cameras().unwrap().len(), 1);
            assert_eq!(db.read_all_images().unwrap().len(), 1);
        }

        assert_eq!(std::fs::read(path).unwrap(), before);
    }

    #[test]
    fn negative_keypoint_rows_are_rejected_before_usize_conversion() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();
        write_test_camera(&db, 1);
        write_test_images(&db, 1, &[1]);
        db.conn
            .execute(
                "INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?1, -1, 2, ?2);",
                params![1i64, Vec::<u8>::new()],
            )
            .unwrap();

        assert!(db.read_keypoint_counts().is_err());
        assert!(db.read_keypoints(1).is_err());
    }

    #[test]
    fn transaction_error_rolls_back_rows_and_deletion_state() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();
        write_test_camera(&db, 1);
        write_test_images(&db, 1, &[1, 2]);
        let original = vec![m(0, 1)];
        db.write_matches(1, 2, &original).unwrap();

        let result: Result<()> = db.with_transaction(|| {
            db.clear_matches()?;
            db.write_matches(1, 2, &[m(1, 0)])?;
            bail!("injected transaction failure")
        });

        assert!(result.is_err());
        assert_eq!(db.read_matches(1, 2).unwrap(), original);
        assert!(!db.database_entry_deleted.get());
    }

    #[test]
    fn camera_and_image_autoincrement_ids_match_sqlite_rowid() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();
        let camera_id = db
            .write_camera(
                &ColmapDatabaseCamera {
                    camera: ColmapCamera {
                        camera_id: 0,
                        model_id: crate::types::COLMAP_SIMPLE_PINHOLE,
                        width: 640,
                        height: 480,
                        params: vec![500.0, 320.0, 240.0],
                    },
                    has_prior_focal_length: false,
                },
                false,
            )
            .unwrap();
        assert_eq!(camera_id, 1);

        let image_id = db
            .write_image(
                &ColmapDatabaseImage {
                    image_id: 0,
                    name: "auto.jpg".to_string(),
                    camera_id,
                    frame_id: None,
                },
                false,
            )
            .unwrap();
        assert_eq!(image_id, 1);
        let image = db.read_image(image_id).unwrap().unwrap();
        assert_eq!(image.name, "auto.jpg");
        assert_eq!(image.camera_id, camera_id);
    }

    #[test]
    fn rigs_and_frames_roundtrip_through_colmap_tables() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();
        let ref_sensor = ColmapSensorId {
            sensor_type: ColmapSensorType::Camera,
            sensor_id: 10,
        };
        let other_sensor = ColmapSensorId {
            sensor_type: ColmapSensorType::Camera,
            sensor_id: 11,
        };
        let rig = ColmapRig {
            rig_id: 5,
            ref_sensor_id: Some(ref_sensor.clone()),
            sensors: vec![
                ColmapRigSensor {
                    sensor_id: ref_sensor.clone(),
                    sensor_from_rig: None,
                },
                ColmapRigSensor {
                    sensor_id: other_sensor.clone(),
                    sensor_from_rig: Some(ColmapRigid3 {
                        qvec: [1.0, 0.0, 0.0, 0.0],
                        tvec: [0.1, 0.2, 0.3],
                    }),
                },
            ],
        };

        assert_eq!(db.write_rig(&rig, true).unwrap(), 5);
        let mut expected_rig = rig.clone();
        expected_rig.sensors = vec![rig.sensors[1].clone()];
        assert_eq!(db.read_rig(5).unwrap(), Some(expected_rig.clone()));
        assert_eq!(
            db.read_rig_with_sensor(&ref_sensor).unwrap(),
            Some(expected_rig.clone())
        );
        assert_eq!(
            db.read_rig_with_sensor(&other_sensor).unwrap(),
            Some(expected_rig.clone())
        );
        assert_eq!(db.read_all_rigs().unwrap(), vec![expected_rig]);

        let frame = ColmapDatabaseFrame {
            frame_id: 7,
            rig_id: 5,
            data_ids: vec![
                ColmapDataId {
                    sensor_id: ref_sensor,
                    data_id: 101,
                },
                ColmapDataId {
                    sensor_id: other_sensor,
                    data_id: 102,
                },
            ],
        };
        assert_eq!(db.write_frame(&frame, true).unwrap(), 7);
        assert_eq!(db.read_frame(7).unwrap(), Some(frame.clone()));
        assert_eq!(db.read_all_frames().unwrap(), vec![frame]);
    }

    #[test]
    fn rig_and_frame_autoincrement_ids_match_sqlite_rowid() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();
        let rig_id = db
            .write_rig(
                &ColmapRig {
                    rig_id: 0,
                    ref_sensor_id: Some(ColmapSensorId {
                        sensor_type: ColmapSensorType::Camera,
                        sensor_id: 1,
                    }),
                    sensors: vec![ColmapRigSensor {
                        sensor_id: ColmapSensorId {
                            sensor_type: ColmapSensorType::Camera,
                            sensor_id: 1,
                        },
                        sensor_from_rig: None,
                    }],
                },
                false,
            )
            .unwrap();
        assert_eq!(rig_id, 1);
        let frame_id = db
            .write_frame(
                &ColmapDatabaseFrame {
                    frame_id: 0,
                    rig_id,
                    data_ids: vec![ColmapDataId {
                        sensor_id: ColmapSensorId {
                            sensor_type: ColmapSensorType::Camera,
                            sensor_id: 1,
                        },
                        data_id: 44,
                    }],
                },
                false,
            )
            .unwrap();
        assert_eq!(frame_id, 1);
    }

    #[test]
    fn pose_priors_roundtrip_through_colmap_table() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();
        let pose_prior = ColmapPosePrior {
            pose_prior_id: 12,
            corr_data_id: ColmapDataId {
                data_id: 42,
                sensor_id: ColmapSensorId {
                    sensor_type: ColmapSensorType::Camera,
                    sensor_id: 3,
                },
            },
            position: [1.0, 2.0, 3.0],
            position_covariance: [1.0, 0.1, 0.2, 0.1, 2.0, 0.3, 0.2, 0.3, 3.0],
            coordinate_system: ColmapPosePriorCoordinateSystem::Cartesian,
            gravity: [0.0, 1.0, 0.0],
        };

        assert_eq!(db.write_pose_prior(&pose_prior, true).unwrap(), 12);
        assert_eq!(db.read_pose_prior(12).unwrap(), Some(pose_prior.clone()));
        assert_eq!(db.read_all_pose_priors().unwrap(), vec![pose_prior]);
    }

    #[test]
    fn pose_prior_autoincrement_id_matches_sqlite_rowid() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();
        let pose_prior_id = db
            .write_pose_prior(
                &ColmapPosePrior {
                    pose_prior_id: 0,
                    corr_data_id: ColmapDataId {
                        data_id: 5,
                        sensor_id: ColmapSensorId {
                            sensor_type: ColmapSensorType::Camera,
                            sensor_id: 1,
                        },
                    },
                    position: [f64::NAN; 3],
                    position_covariance: [f64::NAN; 9],
                    coordinate_system: ColmapPosePriorCoordinateSystem::Wgs84,
                    gravity: [0.0, 0.0, 1.0],
                },
                false,
            )
            .unwrap();
        assert_eq!(pose_prior_id, 1);
        let read = db.read_pose_prior(pose_prior_id).unwrap().unwrap();
        assert_eq!(read.pose_prior_id, 1);
        assert_eq!(
            read.coordinate_system,
            ColmapPosePriorCoordinateSystem::Wgs84
        );
        assert!(read.position.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn pose_prior_table_keeps_legacy_image_id_compatibility_column() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("database.db");
        {
            let db = ColmapDatabase::open(&path).unwrap();
            assert!(db.exists_column("pose_priors", "image_id").unwrap());
            assert!(db.exists_column("pose_priors", "corr_data_id").unwrap());
            let mut pose_prior = ColmapPosePrior {
                pose_prior_id: 9,
                corr_data_id: ColmapDataId {
                    data_id: 42,
                    sensor_id: ColmapSensorId {
                        sensor_type: ColmapSensorType::Camera,
                        sensor_id: 3,
                    },
                },
                position: [1.0, 2.0, 3.0],
                position_covariance: [f64::NAN; 9],
                coordinate_system: ColmapPosePriorCoordinateSystem::Cartesian,
                gravity: [f64::NAN; 3],
            };
            db.write_pose_prior(&pose_prior, true).unwrap();

            let image_id = db
                .conn
                .query_row(
                    "SELECT image_id FROM pose_priors WHERE pose_prior_id = ?1;",
                    params![pose_prior.pose_prior_id],
                    |row| row.get::<_, Option<i64>>(0),
                )
                .unwrap();
            assert_eq!(image_id, Some(42));

            pose_prior.corr_data_id.data_id = 43;
            db.update_pose_prior(&pose_prior).unwrap();
            let image_id = db
                .conn
                .query_row(
                    "SELECT image_id FROM pose_priors WHERE pose_prior_id = ?1;",
                    params![pose_prior.pose_prior_id],
                    |row| row.get::<_, Option<i64>>(0),
                )
                .unwrap();
            assert_eq!(image_id, Some(43));
        }

        let db = ColmapDatabase::open(&path).unwrap();
        assert!(!db.exists_table("pose_priors_old").unwrap());
        let pose_prior = db.read_pose_prior(9).unwrap().unwrap();
        assert_eq!(pose_prior.corr_data_id.data_id, 43);
    }

    #[test]
    fn database_cache_filters_pairs_and_unconnected_images() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();
        let camera = ColmapDatabaseCamera {
            camera: ColmapCamera {
                camera_id: 1,
                model_id: crate::types::COLMAP_PINHOLE,
                width: 100,
                height: 100,
                params: vec![50.0, 50.0, 50.0, 50.0],
            },
            has_prior_focal_length: true,
        };
        db.write_camera(&camera, true).unwrap();
        for image_id in 1..=3 {
            db.write_image(
                &ColmapDatabaseImage {
                    image_id,
                    name: format!("{image_id}.jpg"),
                    camera_id: 1,
                    frame_id: None,
                },
                true,
            )
            .unwrap();
            db.write_keypoints(
                image_id,
                &[
                    ColmapKeypoint::new(0.0, 0.0),
                    ColmapKeypoint::new(1.0, 1.0),
                    ColmapKeypoint::new(2.0, 2.0),
                ],
            )
            .unwrap();
        }
        db.write_two_view_geometry(
            1,
            2,
            &ColmapTwoViewGeometry {
                config: 2,
                inlier_matches: vec![m(0, 0), m(1, 1)],
                ..ColmapTwoViewGeometry::default()
            },
        )
        .unwrap();
        db.write_two_view_geometry(
            2,
            3,
            &ColmapTwoViewGeometry {
                config: COLMAP_TWO_VIEW_WATERMARK,
                inlier_matches: vec![m(0, 0), m(1, 1), m(2, 2)],
                ..ColmapTwoViewGeometry::default()
            },
        )
        .unwrap();

        let cache = db
            .load_cache(&DatabaseCacheOptions {
                min_num_matches: 2,
                ignore_watermarks: true,
                load_all_images: false,
                ..DatabaseCacheOptions::default()
            })
            .unwrap();
        assert_eq!(cache.images.keys().copied().collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(cache.cameras.len(), 1);
        assert_eq!(cache.correspondence_graph.num_image_pairs(), 1);
        assert_eq!(
            cache
                .correspondence_graph
                .num_matches_between_images(1, 2)
                .unwrap(),
            2
        );
        assert_eq!(
            cache
                .correspondence_graph
                .num_matches_between_images(2, 3)
                .unwrap(),
            0
        );

        let all_cache = db
            .load_cache(&DatabaseCacheOptions {
                load_all_images: true,
                ..DatabaseCacheOptions::default()
            })
            .unwrap();
        assert_eq!(
            all_cache.images.keys().copied().collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn database_cache_uses_colmap_inlier_match_filter_without_rejecting_configs() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();
        let camera = ColmapDatabaseCamera {
            camera: ColmapCamera {
                camera_id: 1,
                model_id: crate::types::COLMAP_PINHOLE,
                width: 100,
                height: 100,
                params: vec![50.0, 50.0, 50.0, 50.0],
            },
            has_prior_focal_length: true,
        };
        db.write_camera(&camera, true).unwrap();
        for image_id in 1..=3 {
            db.write_image(
                &ColmapDatabaseImage {
                    image_id,
                    name: format!("{image_id}.jpg"),
                    camera_id: 1,
                    frame_id: None,
                },
                true,
            )
            .unwrap();
            db.write_keypoints(
                image_id,
                &[ColmapKeypoint::new(0.0, 0.0), ColmapKeypoint::new(1.0, 1.0)],
            )
            .unwrap();
        }
        db.write_two_view_geometry(
            1,
            2,
            &ColmapTwoViewGeometry {
                config: COLMAP_TWO_VIEW_DEGENERATE,
                inlier_matches: vec![m(0, 0), m(1, 1)],
                ..ColmapTwoViewGeometry::default()
            },
        )
        .unwrap();
        db.write_two_view_geometry(
            2,
            3,
            &ColmapTwoViewGeometry {
                config: COLMAP_TWO_VIEW_UNDEFINED,
                inlier_matches: vec![m(0, 0), m(1, 1)],
                ..ColmapTwoViewGeometry::default()
            },
        )
        .unwrap();

        let cache = db.load_cache(&DatabaseCacheOptions::default()).unwrap();

        assert_eq!(
            cache.images.keys().copied().collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(cache.correspondence_graph.num_image_pairs(), 2);
        assert_eq!(
            cache
                .correspondence_graph
                .num_matches_between_images(1, 2)
                .unwrap(),
            2
        );
        assert_eq!(
            cache
                .correspondence_graph
                .extract_two_view_geometry(1, 2, false)
                .unwrap()
                .config,
            COLMAP_TWO_VIEW_DEGENERATE
        );
        assert_eq!(
            cache
                .correspondence_graph
                .extract_two_view_geometry(2, 3, false)
                .unwrap()
                .config,
            COLMAP_TWO_VIEW_UNDEFINED
        );
        assert!(!is_colmap_two_view_geometry_with_inliers(
            COLMAP_TWO_VIEW_DEGENERATE
        ));
        assert!(!is_colmap_two_view_geometry_with_inliers(
            COLMAP_TWO_VIEW_UNDEFINED
        ));
        assert!(is_colmap_two_view_geometry_with_inliers(
            COLMAP_TWO_VIEW_CALIBRATED
        ));
    }

    #[test]
    fn database_cache_correspondence_graph_preserves_two_view_metadata() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();
        write_test_camera(&db, 1);
        write_test_images(&db, 1, &[1, 2]);
        db.write_keypoints(
            1,
            &[ColmapKeypoint::new(0.0, 0.0), ColmapKeypoint::new(1.0, 1.0)],
        )
        .unwrap();
        db.write_keypoints(
            2,
            &[ColmapKeypoint::new(0.0, 0.0), ColmapKeypoint::new(1.0, 1.0)],
        )
        .unwrap();
        let geometry = ColmapTwoViewGeometry {
            config: COLMAP_TWO_VIEW_PLANAR_OR_PANORAMIC,
            inlier_matches: vec![m(0, 0), m(1, 1)],
            f_matrix: Some([1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]),
            e_matrix: Some([4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0]),
            h_matrix: Some([1.0, 0.0, 2.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0]),
            qvec: Some([1.0, 0.0, 0.0, 0.0]),
            tvec: Some([0.1, 0.2, 0.3]),
        };
        db.write_two_view_geometry(1, 2, &geometry).unwrap();

        let cache = db
            .load_cache(&DatabaseCacheOptions {
                load_all_images: true,
                ..DatabaseCacheOptions::default()
            })
            .unwrap();
        let cached = cache
            .correspondence_graph
            .extract_two_view_geometry(1, 2, true)
            .unwrap();

        assert_eq!(cached.config, geometry.config);
        assert_eq!(cached.inlier_matches, geometry.inlier_matches);
        assert_eq!(cached.f_matrix, geometry.f_matrix);
        assert_eq!(cached.e_matrix, geometry.e_matrix);
        assert_eq!(cached.h_matrix, geometry.h_matrix);
        assert_eq!(cached.qvec, geometry.qvec);
        assert_eq!(cached.tvec, geometry.tvec);
    }

    #[test]
    fn database_cache_image_name_filter_loads_entire_frames() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();
        db.write_camera(
            &ColmapDatabaseCamera {
                camera: ColmapCamera {
                    camera_id: 1,
                    model_id: crate::types::COLMAP_PINHOLE,
                    width: 100,
                    height: 100,
                    params: vec![50.0, 50.0, 50.0, 50.0],
                },
                has_prior_focal_length: true,
            },
            true,
        )
        .unwrap();
        db.write_rig(
            &ColmapRig {
                rig_id: 1,
                ref_sensor_id: Some(ColmapSensorId {
                    sensor_type: ColmapSensorType::Camera,
                    sensor_id: 1,
                }),
                sensors: vec![ColmapRigSensor {
                    sensor_id: ColmapSensorId {
                        sensor_type: ColmapSensorType::Camera,
                        sensor_id: 1,
                    },
                    sensor_from_rig: None,
                }],
            },
            true,
        )
        .unwrap();
        for (image_id, name) in [(1, "left.jpg"), (2, "right.jpg"), (3, "other.jpg")] {
            db.write_image(
                &ColmapDatabaseImage {
                    image_id,
                    name: name.to_string(),
                    camera_id: 1,
                    frame_id: None,
                },
                true,
            )
            .unwrap();
            db.write_keypoints(
                image_id,
                &[ColmapKeypoint::new(0.0, 0.0), ColmapKeypoint::new(1.0, 1.0)],
            )
            .unwrap();
        }
        db.write_frame(
            &ColmapDatabaseFrame {
                frame_id: 10,
                rig_id: 1,
                data_ids: vec![
                    ColmapDataId {
                        sensor_id: ColmapSensorId {
                            sensor_type: ColmapSensorType::Camera,
                            sensor_id: 1,
                        },
                        data_id: 1,
                    },
                    ColmapDataId {
                        sensor_id: ColmapSensorId {
                            sensor_type: ColmapSensorType::Camera,
                            sensor_id: 1,
                        },
                        data_id: 2,
                    },
                ],
            },
            true,
        )
        .unwrap();
        db.write_frame(
            &ColmapDatabaseFrame {
                frame_id: 11,
                rig_id: 1,
                data_ids: vec![ColmapDataId {
                    sensor_id: ColmapSensorId {
                        sensor_type: ColmapSensorType::Camera,
                        sensor_id: 1,
                    },
                    data_id: 3,
                }],
            },
            true,
        )
        .unwrap();
        db.write_two_view_geometry(
            1,
            2,
            &ColmapTwoViewGeometry {
                config: 2,
                inlier_matches: vec![m(0, 0), m(1, 1)],
                ..ColmapTwoViewGeometry::default()
            },
        )
        .unwrap();
        db.write_two_view_geometry(
            2,
            3,
            &ColmapTwoViewGeometry {
                config: 2,
                inlier_matches: vec![m(0, 0), m(1, 1)],
                ..ColmapTwoViewGeometry::default()
            },
        )
        .unwrap();

        let mut image_names = BTreeSet::new();
        image_names.insert("left.jpg".to_string());
        let cache = db
            .load_cache(&DatabaseCacheOptions {
                image_names: image_names.clone(),
                load_all_images: true,
                ..DatabaseCacheOptions::default()
            })
            .unwrap();
        assert_eq!(cache.images.keys().copied().collect::<Vec<_>>(), vec![1, 2]);

        let connected_cache = db
            .load_cache(&DatabaseCacheOptions {
                image_names,
                min_num_matches: 2,
                load_all_images: false,
                ..DatabaseCacheOptions::default()
            })
            .unwrap();
        assert_eq!(
            connected_cache.images.keys().copied().collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(connected_cache.correspondence_graph.num_image_pairs(), 1);
    }

    #[test]
    fn database_cache_create_from_cache_matches_colmap_frame_filtering_and_metadata_copy() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();
        for camera_id in [1, 2] {
            db.write_camera(
                &ColmapDatabaseCamera {
                    camera: ColmapCamera {
                        camera_id,
                        model_id: crate::types::COLMAP_PINHOLE,
                        width: 100,
                        height: 100,
                        params: vec![50.0, 50.0, 50.0, 50.0],
                    },
                    has_prior_focal_length: true,
                },
                true,
            )
            .unwrap();
            db.write_rig(
                &ColmapRig {
                    rig_id: camera_id,
                    ref_sensor_id: Some(ColmapSensorId {
                        sensor_type: ColmapSensorType::Camera,
                        sensor_id: camera_id,
                    }),
                    sensors: vec![ColmapRigSensor {
                        sensor_id: ColmapSensorId {
                            sensor_type: ColmapSensorType::Camera,
                            sensor_id: camera_id,
                        },
                        sensor_from_rig: None,
                    }],
                },
                true,
            )
            .unwrap();
        }
        for (image_id, camera_id, name) in [
            (1, 1, "a_left.jpg"),
            (2, 1, "a_right.jpg"),
            (3, 2, "b_left.jpg"),
            (4, 2, "b_right.jpg"),
        ] {
            db.write_image(
                &ColmapDatabaseImage {
                    image_id,
                    name: name.to_string(),
                    camera_id,
                    frame_id: None,
                },
                true,
            )
            .unwrap();
            db.write_keypoints(
                image_id,
                &[ColmapKeypoint::new(0.0, 0.0), ColmapKeypoint::new(1.0, 1.0)],
            )
            .unwrap();
        }
        db.write_frame(
            &ColmapDatabaseFrame {
                frame_id: 10,
                rig_id: 1,
                data_ids: vec![
                    ColmapDataId {
                        sensor_id: ColmapSensorId {
                            sensor_type: ColmapSensorType::Camera,
                            sensor_id: 1,
                        },
                        data_id: 1,
                    },
                    ColmapDataId {
                        sensor_id: ColmapSensorId {
                            sensor_type: ColmapSensorType::Camera,
                            sensor_id: 1,
                        },
                        data_id: 2,
                    },
                ],
            },
            true,
        )
        .unwrap();
        db.write_frame(
            &ColmapDatabaseFrame {
                frame_id: 20,
                rig_id: 2,
                data_ids: vec![
                    ColmapDataId {
                        sensor_id: ColmapSensorId {
                            sensor_type: ColmapSensorType::Camera,
                            sensor_id: 2,
                        },
                        data_id: 3,
                    },
                    ColmapDataId {
                        sensor_id: ColmapSensorId {
                            sensor_type: ColmapSensorType::Camera,
                            sensor_id: 2,
                        },
                        data_id: 4,
                    },
                ],
            },
            true,
        )
        .unwrap();
        let geometry = ColmapTwoViewGeometry {
            config: COLMAP_TWO_VIEW_PLANAR,
            inlier_matches: vec![m(0, 0), m(1, 1)],
            f_matrix: Some([1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]),
            h_matrix: Some([1.0, 0.0, 2.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0]),
            qvec: Some([1.0, 0.0, 0.0, 0.0]),
            tvec: Some([0.1, 0.2, 0.3]),
            ..ColmapTwoViewGeometry::default()
        };
        db.write_two_view_geometry(1, 2, &geometry).unwrap();
        db.write_two_view_geometry(
            3,
            4,
            &ColmapTwoViewGeometry {
                config: COLMAP_TWO_VIEW_CALIBRATED,
                inlier_matches: vec![m(0, 0), m(1, 1)],
                ..ColmapTwoViewGeometry::default()
            },
        )
        .unwrap();

        let base_cache = db
            .load_cache(&DatabaseCacheOptions {
                load_all_images: true,
                ..DatabaseCacheOptions::default()
            })
            .unwrap();
        let mut image_names = BTreeSet::new();
        image_names.insert("a_left.jpg".to_string());
        let filtered = DatabaseCache::create_from_cache(
            &base_cache,
            &DatabaseCacheOptions {
                image_names,
                load_all_images: true,
                ..DatabaseCacheOptions::default()
            },
        )
        .unwrap();

        assert_eq!(
            filtered.images.keys().copied().collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(
            filtered.frames.keys().copied().collect::<Vec<_>>(),
            vec![10]
        );
        assert_eq!(
            filtered.cameras.keys().copied().collect::<Vec<_>>(),
            vec![1]
        );
        assert_eq!(filtered.rigs.keys().copied().collect::<Vec<_>>(), vec![1]);
        assert_eq!(filtered.correspondence_graph.num_image_pairs(), 1);
        let copied = filtered
            .correspondence_graph
            .extract_two_view_geometry(1, 2, true)
            .unwrap();
        assert_eq!(copied.config, COLMAP_TWO_VIEW_PLANAR);
        assert_eq!(copied.inlier_matches, vec![m(0, 0), m(1, 1)]);
        assert_eq!(copied.f_matrix, geometry.f_matrix);
        assert_eq!(copied.h_matrix, geometry.h_matrix);
        assert_eq!(copied.qvec, geometry.qvec);
        assert_eq!(copied.tvec, geometry.tvec);
    }

    #[test]
    fn database_cache_creates_trivial_frames_for_legacy_databases() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();
        db.write_camera(
            &ColmapDatabaseCamera {
                camera: ColmapCamera {
                    camera_id: 9,
                    model_id: crate::types::COLMAP_PINHOLE,
                    width: 100,
                    height: 100,
                    params: vec![50.0, 50.0, 50.0, 50.0],
                },
                has_prior_focal_length: true,
            },
            true,
        )
        .unwrap();
        for image_id in 1..=2 {
            db.write_image(
                &ColmapDatabaseImage {
                    image_id,
                    name: format!("legacy_{image_id}.jpg"),
                    camera_id: 9,
                    frame_id: None,
                },
                true,
            )
            .unwrap();
            db.write_keypoints(
                image_id,
                &[ColmapKeypoint::new(0.0, 0.0), ColmapKeypoint::new(1.0, 1.0)],
            )
            .unwrap();
        }
        db.write_two_view_geometry(
            1,
            2,
            &ColmapTwoViewGeometry {
                config: 2,
                inlier_matches: vec![m(0, 0), m(1, 1)],
                ..ColmapTwoViewGeometry::default()
            },
        )
        .unwrap();

        let cache = db.load_cache(&DatabaseCacheOptions::default()).unwrap();

        assert_eq!(cache.rigs.keys().copied().collect::<Vec<_>>(), vec![9]);
        assert_eq!(
            cache.rigs.get(&9).unwrap(),
            &ColmapRig {
                rig_id: 9,
                ref_sensor_id: Some(ColmapSensorId {
                    sensor_type: ColmapSensorType::Camera,
                    sensor_id: 9,
                }),
                sensors: vec![ColmapRigSensor {
                    sensor_id: ColmapSensorId {
                        sensor_type: ColmapSensorType::Camera,
                        sensor_id: 9,
                    },
                    sensor_from_rig: None,
                }],
            }
        );
        assert_eq!(cache.frames.keys().copied().collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(cache.images.get(&1).unwrap().frame_id, Some(1));
        assert_eq!(cache.images.get(&2).unwrap().frame_id, Some(2));
        assert_eq!(cache.frames.get(&1).unwrap().rig_id, 9);
        assert_eq!(
            cache.frames.get(&1).unwrap().data_ids,
            vec![ColmapDataId {
                sensor_id: ColmapSensorId {
                    sensor_type: ColmapSensorType::Camera,
                    sensor_id: 9,
                },
                data_id: 1,
            }]
        );
    }

    #[test]
    fn database_cache_helpers_match_colmap_api_shape() {
        let mut cache = DatabaseCache::new();
        let sensor_id = ColmapSensorId {
            sensor_type: ColmapSensorType::Camera,
            sensor_id: 7,
        };
        cache
            .add_rig(ColmapRig {
                rig_id: 7,
                ref_sensor_id: Some(sensor_id.clone()),
                sensors: vec![ColmapRigSensor {
                    sensor_id: sensor_id.clone(),
                    sensor_from_rig: None,
                }],
            })
            .unwrap();
        cache
            .add_camera(ColmapDatabaseCamera {
                camera: ColmapCamera {
                    camera_id: 7,
                    model_id: crate::types::COLMAP_PINHOLE,
                    width: 100,
                    height: 100,
                    params: vec![50.0, 50.0, 50.0, 50.0],
                },
                has_prior_focal_length: true,
            })
            .unwrap();
        cache
            .add_frame(ColmapDatabaseFrame {
                frame_id: 11,
                rig_id: 7,
                data_ids: vec![ColmapDataId {
                    sensor_id,
                    data_id: 3,
                }],
            })
            .unwrap();
        cache
            .add_image(
                ColmapDatabaseImage {
                    image_id: 3,
                    name: "cache.jpg".to_string(),
                    camera_id: 7,
                    frame_id: Some(11),
                },
                4,
            )
            .unwrap();
        cache.add_pose_prior(ColmapPosePrior {
            pose_prior_id: 5,
            corr_data_id: ColmapDataId {
                sensor_id: ColmapSensorId {
                    sensor_type: ColmapSensorType::Camera,
                    sensor_id: 7,
                },
                data_id: 3,
            },
            position: [1.0, 2.0, 3.0],
            position_covariance: [0.0; 9],
            coordinate_system: ColmapPosePriorCoordinateSystem::Cartesian,
            gravity: [0.0, 0.0, 1.0],
        });

        assert_eq!(cache.num_rigs(), 1);
        assert_eq!(cache.num_cameras(), 1);
        assert_eq!(cache.num_frames(), 1);
        assert_eq!(cache.num_images(), 1);
        assert_eq!(cache.num_pose_priors(), 1);
        assert!(cache.exists_rig(7));
        assert!(cache.exists_camera(7));
        assert!(cache.exists_frame(11));
        assert!(cache.exists_image(3));
        assert_eq!(cache.rig(7).unwrap().rig_id, 7);
        assert_eq!(cache.camera(7).unwrap().camera.camera_id, 7);
        assert_eq!(cache.frame(11).unwrap().frame_id, 11);
        assert_eq!(cache.image(3).unwrap().name, "cache.jpg");
        assert_eq!(cache.find_image_with_name("cache.jpg").unwrap().image_id, 3);
        assert_eq!(cache.correspondence_graph().num_images(), 1);
        assert!(cache.add_rig(cache.rig(7).unwrap().clone()).is_err());
        assert!(cache.find_image_with_name("missing.jpg").is_none());
    }

    #[test]
    fn database_cache_converts_wgs84_pose_priors_to_enu_when_requested() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();
        write_test_camera(&db, 1);
        write_test_images(&db, 1, &[1, 2]);
        for pose_prior in [
            ColmapPosePrior {
                pose_prior_id: 1,
                corr_data_id: ColmapDataId {
                    data_id: 1,
                    sensor_id: ColmapSensorId {
                        sensor_type: ColmapSensorType::Camera,
                        sensor_id: 1,
                    },
                },
                position: [47.0, 8.0, 500.0],
                position_covariance: [f64::NAN; 9],
                coordinate_system: ColmapPosePriorCoordinateSystem::Wgs84,
                gravity: [f64::NAN; 3],
            },
            ColmapPosePrior {
                pose_prior_id: 2,
                corr_data_id: ColmapDataId {
                    data_id: 2,
                    sensor_id: ColmapSensorId {
                        sensor_type: ColmapSensorType::Camera,
                        sensor_id: 1,
                    },
                },
                position: [47.0001, 8.0002, 520.0],
                position_covariance: [f64::NAN; 9],
                coordinate_system: ColmapPosePriorCoordinateSystem::Wgs84,
                gravity: [f64::NAN; 3],
            },
        ] {
            db.write_pose_prior(&pose_prior, true).unwrap();
        }

        let raw_cache = db
            .load_cache(&DatabaseCacheOptions {
                load_all_images: true,
                ..DatabaseCacheOptions::default()
            })
            .unwrap();
        assert_eq!(
            raw_cache.pose_priors[0].coordinate_system,
            ColmapPosePriorCoordinateSystem::Wgs84
        );
        assert_eq!(raw_cache.pose_priors[0].position, [47.0, 8.0, 500.0]);

        let enu_cache = db
            .load_cache(&DatabaseCacheOptions {
                load_all_images: true,
                convert_pose_priors_to_enu: true,
                ..DatabaseCacheOptions::default()
            })
            .unwrap();

        assert_eq!(
            enu_cache.pose_priors[0].coordinate_system,
            ColmapPosePriorCoordinateSystem::Cartesian
        );
        assert_eq!(
            enu_cache.pose_priors[1].coordinate_system,
            ColmapPosePriorCoordinateSystem::Cartesian
        );
        for value in enu_cache.pose_priors[0].position {
            assert!(value.abs() < 1.0e-8);
        }
        let expected = wgs84_ellipsoid_to_enu(
            &[[47.0, 8.0, 500.0], [47.0001, 8.0002, 520.0]],
            [47.0, 8.0, 500.0],
        )
        .unwrap();
        for (actual, expected) in enu_cache.pose_priors[1]
            .position
            .iter()
            .zip(expected[1].iter())
        {
            assert!((actual - expected).abs() < 1.0e-8);
        }
        assert!(enu_cache.pose_priors[1].position[0] > 0.0);
        assert!(enu_cache.pose_priors[1].position[1] > 0.0);
    }

    #[test]
    fn database_cache_rejects_mixed_pose_prior_coordinate_systems_for_enu_conversion() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();
        for pose_prior in [
            ColmapPosePrior {
                pose_prior_id: 1,
                corr_data_id: ColmapDataId {
                    data_id: 1,
                    sensor_id: ColmapSensorId {
                        sensor_type: ColmapSensorType::Camera,
                        sensor_id: 1,
                    },
                },
                position: [47.0, 8.0, 500.0],
                position_covariance: [f64::NAN; 9],
                coordinate_system: ColmapPosePriorCoordinateSystem::Wgs84,
                gravity: [f64::NAN; 3],
            },
            ColmapPosePrior {
                pose_prior_id: 2,
                corr_data_id: ColmapDataId {
                    data_id: 2,
                    sensor_id: ColmapSensorId {
                        sensor_type: ColmapSensorType::Camera,
                        sensor_id: 1,
                    },
                },
                position: [1.0, 2.0, 3.0],
                position_covariance: [f64::NAN; 9],
                coordinate_system: ColmapPosePriorCoordinateSystem::Cartesian,
                gravity: [f64::NAN; 3],
            },
        ] {
            db.write_pose_prior(&pose_prior, true).unwrap();
        }

        let err = db
            .load_cache(&DatabaseCacheOptions {
                convert_pose_priors_to_enu: true,
                ..DatabaseCacheOptions::default()
            })
            .unwrap_err();

        assert!(err
            .to_string()
            .contains("inconsistent coordinate systems defined in pose priors"));
    }

    #[test]
    fn keypoints_descriptors_roundtrip_through_colmap_tables() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("database.db");
        let db = ColmapDatabase::open(&path).unwrap();
        write_test_camera(&db, 1);
        write_test_images(&db, 1, &[7]);
        let keypoints = vec![
            ColmapKeypoint::new(10.5, 20.25),
            ColmapKeypoint {
                x: 1.0,
                y: 2.0,
                a11: 3.0,
                a12: 4.0,
                a21: 5.0,
                a22: 6.0,
            },
        ];
        let descriptors = ColmapDescriptors::new(0, 2, 4, vec![1, 2, 3, 4, 5, 6, 7, 8]).unwrap();

        db.write_keypoints(7, &keypoints).unwrap();
        db.write_descriptors(7, &descriptors).unwrap();
        drop(db);

        let db = ColmapDatabase::open(&path).unwrap();
        assert_eq!(db.read_keypoints(7).unwrap(), keypoints);
        assert_eq!(db.read_descriptors(7).unwrap(), descriptors);
    }

    #[test]
    fn load_random_database_descriptors_matches_colmap_helper_shape() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();
        write_test_camera(&db, 1);
        for image_id in 1..=3 {
            db.write_image(
                &ColmapDatabaseImage {
                    image_id,
                    name: format!("desc_{image_id}.jpg"),
                    camera_id: 1,
                    frame_id: None,
                },
                true,
            )
            .unwrap();
        }
        db.write_descriptors(
            1,
            &ColmapDescriptors::new(COLMAP_FEATURE_SIFT, 2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap(),
        )
        .unwrap();
        db.write_descriptors(
            2,
            &ColmapDescriptors::new(COLMAP_FEATURE_SIFT, 0, 3, Vec::new()).unwrap(),
        )
        .unwrap();
        db.write_descriptors(
            3,
            &ColmapDescriptors::new(COLMAP_FEATURE_SIFT, 2, 3, vec![7, 8, 9, 10, 11, 12]).unwrap(),
        )
        .unwrap();

        let empty = load_random_database_descriptors(&db, 0).unwrap();
        assert_eq!(empty.rows, 0);
        assert_eq!(empty.cols, 0);
        assert_eq!(empty.feature_type, COLMAP_FEATURE_UNDEFINED);

        let all = load_random_database_descriptors(&db, -1).unwrap();
        assert_eq!(all.feature_type, COLMAP_FEATURE_SIFT);
        assert_eq!(all.rows, 4);
        assert_eq!(all.cols, 3);
        assert_eq!(
            all.data,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        );

        let subset = load_random_database_descriptors(&db, 2).unwrap();
        assert_eq!(subset.feature_type, COLMAP_FEATURE_SIFT);
        assert_eq!(subset.rows, 2);
        assert_eq!(subset.cols, 3);
        assert_eq!(subset.data.len(), 6);
    }

    #[test]
    fn matches_roundtrip_with_colmap_pair_ordering() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();
        let matches = vec![m(3, 4), m(5, 6)];

        db.write_matches(9, 2, &matches).unwrap();

        assert_eq!(db.read_matches(9, 2).unwrap(), matches);
        assert_eq!(db.read_matches(2, 9).unwrap(), vec![m(4, 3), m(6, 5)]);
        let all = db.read_all_matches().unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].1, vec![m(4, 3), m(6, 5)]);
        assert_eq!(
            db.read_num_matches().unwrap(),
            vec![(image_pair_to_pair_id(9, 2).unwrap(), 2)]
        );
    }

    #[test]
    fn keypoint_blob_overloads_preserve_raw_colmap_matrix_shape() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();
        write_test_camera(&db, 1);
        write_test_images(&db, 1, &[7]);
        let blob4 = ColmapKeypointsBlob::new(
            2,
            4,
            encode_f32_blob(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        )
        .unwrap();

        db.write_keypoints_blob(7, &blob4).unwrap();

        assert_eq!(db.read_keypoints_blob(7).unwrap(), blob4);
        assert_eq!(
            db.read_keypoints(7).unwrap(),
            vec![
                ColmapKeypoint::from_scale_orientation(1.0, 2.0, 3.0, 4.0),
                ColmapKeypoint::from_scale_orientation(5.0, 6.0, 7.0, 8.0),
            ]
        );

        let blob2 = ColmapKeypointsBlob::new(1, 2, encode_f32_blob(&[9.0, 10.0])).unwrap();
        db.update_keypoints_blob(7, &blob2).unwrap();

        assert_eq!(db.read_keypoints_blob(7).unwrap(), blob2);
        assert_eq!(
            db.read_keypoints(7).unwrap(),
            vec![ColmapKeypoint::new(9.0, 10.0)]
        );
    }

    #[test]
    fn match_blob_overloads_follow_colmap_pair_direction() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();
        let query_direction =
            ColmapMatchesBlob::new(2, 2, encode_matches_blob(&[m(3, 4), m(5, 6)])).unwrap();
        let stored_direction =
            ColmapMatchesBlob::new(2, 2, encode_matches_blob(&[m(4, 3), m(6, 5)])).unwrap();

        db.write_matches_blob(9, 2, &query_direction).unwrap();

        assert_eq!(db.read_matches_blob(9, 2).unwrap(), query_direction);
        assert_eq!(db.read_matches_blob(2, 9).unwrap(), stored_direction);
        assert_eq!(db.read_matches(9, 2).unwrap(), vec![m(3, 4), m(5, 6)]);
        assert_eq!(db.read_matches(2, 9).unwrap(), vec![m(4, 3), m(6, 5)]);
        assert_eq!(
            db.read_all_matches_blob().unwrap(),
            vec![(image_pair_to_pair_id(9, 2).unwrap(), stored_direction)]
        );
    }

    #[test]
    fn two_view_geometry_roundtrip_preserves_optional_blobs() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();
        let geometry = ColmapTwoViewGeometry {
            config: 3,
            inlier_matches: vec![m(1, 2), m(3, 4)],
            f_matrix: Some([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
            e_matrix: None,
            h_matrix: Some([1.0, 0.0, 2.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0]),
            qvec: Some([1.0, 0.0, 0.0, 0.0]),
            tvec: Some([0.3, 0.4, 0.5]),
        };

        db.write_two_view_geometry(8, 4, &geometry).unwrap();

        assert_eq!(db.read_two_view_geometry(8, 4).unwrap(), geometry);
        let reversed = db.read_two_view_geometry(4, 8).unwrap();
        assert_eq!(reversed.config, geometry.config);
        assert_eq!(reversed.inlier_matches, vec![m(2, 1), m(4, 3)]);
        assert_eq!(reversed.f_matrix, geometry.f_matrix.map(transpose3));
        let expected_h = geometry.h_matrix.and_then(invert_matrix3).unwrap();
        for (actual, expected) in reversed.h_matrix.unwrap().iter().zip(expected_h.iter()) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
        let expected_q = [1.0, -0.0, -0.0, -0.0];
        for (actual, expected) in reversed.qvec.unwrap().iter().zip(expected_q.iter()) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
        let expected_t = [-0.3, -0.4, -0.5];
        for (actual, expected) in reversed.tvec.unwrap().iter().zip(expected_t.iter()) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
        assert_eq!(
            db.read_two_view_geometry_num_inliers().unwrap(),
            vec![(image_pair_to_pair_id(8, 4).unwrap(), 2)]
        );
    }

    #[test]
    fn two_view_geometry_write_rejects_duplicate_and_update_replaces_existing_row() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();

        db.write_two_view_geometry(
            1,
            2,
            &ColmapTwoViewGeometry {
                config: COLMAP_TWO_VIEW_CALIBRATED,
                inlier_matches: vec![m(0, 0)],
                ..ColmapTwoViewGeometry::default()
            },
        )
        .unwrap();
        assert!(db
            .write_two_view_geometry(
                1,
                2,
                &ColmapTwoViewGeometry {
                    config: COLMAP_TWO_VIEW_CALIBRATED,
                    inlier_matches: vec![m(1, 1)],
                    ..ColmapTwoViewGeometry::default()
                },
            )
            .is_err());
        let replacement = ColmapTwoViewGeometry {
            config: COLMAP_TWO_VIEW_PLANAR,
            inlier_matches: vec![m(1, 2), m(3, 4)],
            h_matrix: Some([1.0, 0.0, 2.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0]),
            qvec: Some([1.0, 0.0, 0.0, 0.0]),
            tvec: Some([0.0, 0.0, 1.0]),
            ..ColmapTwoViewGeometry::default()
        };

        db.update_two_view_geometry(1, 2, &replacement).unwrap();

        assert_eq!(db.read_two_view_geometry(1, 2).unwrap(), replacement);
        assert_eq!(
            db.read_two_view_geometry_num_inliers().unwrap(),
            vec![(image_pair_to_pair_id(1, 2).unwrap(), 2)]
        );
    }

    #[test]
    fn database_exists_counts_delete_and_clear_match_colmap_database_api_shape() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();
        write_test_camera(&db, 1);
        write_test_images(&db, 1, &[1, 2]);
        db.write_keypoints(
            1,
            &[ColmapKeypoint::new(0.0, 0.0), ColmapKeypoint::new(1.0, 1.0)],
        )
        .unwrap();
        db.write_keypoints(2, &[ColmapKeypoint::new(2.0, 2.0)])
            .unwrap();
        db.write_descriptors(
            1,
            &ColmapDescriptors::new(COLMAP_FEATURE_SIFT, 2, 2, vec![1, 2, 3, 4]).unwrap(),
        )
        .unwrap();
        db.write_matches(1, 2, &[m(0, 0), m(1, 0)]).unwrap();
        db.write_two_view_geometry(
            1,
            2,
            &ColmapTwoViewGeometry {
                config: COLMAP_TWO_VIEW_CALIBRATED,
                inlier_matches: vec![m(0, 0)],
                f_matrix: Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
                ..ColmapTwoViewGeometry::default()
            },
        )
        .unwrap();

        assert!(db.exists_camera(1).unwrap());
        assert!(db.exists_image(1).unwrap());
        assert!(db.exists_image_with_name("1.jpg").unwrap());
        assert!(db.exists_keypoints(1).unwrap());
        assert!(db.exists_descriptors(1).unwrap());
        assert!(db.exists_matches(2, 1).unwrap());
        assert!(db.exists_two_view_geometry(2, 1).unwrap());
        assert_eq!(db.num_cameras().unwrap(), 1);
        assert_eq!(db.num_images().unwrap(), 2);
        assert_eq!(db.num_keypoints().unwrap(), 3);
        assert_eq!(db.max_num_keypoints().unwrap(), 2);
        assert_eq!(db.num_keypoints_for_image(2).unwrap(), 1);
        assert_eq!(db.num_descriptors().unwrap(), 2);
        assert_eq!(db.max_num_descriptors().unwrap(), 2);
        assert_eq!(db.num_descriptors_for_image(1).unwrap(), 2);
        assert_eq!(db.num_matches().unwrap(), 2);
        assert_eq!(db.num_inlier_matches().unwrap(), 1);
        assert_eq!(db.num_matched_image_pairs().unwrap(), 1);
        assert_eq!(db.num_verified_image_pairs().unwrap(), 1);

        db.delete_matches(2, 1).unwrap();
        assert!(!db.exists_matches(1, 2).unwrap());
        db.delete_inlier_matches(1, 2).unwrap();
        assert_eq!(
            db.read_two_view_geometry(1, 2).unwrap().inlier_matches,
            vec![]
        );
        assert_eq!(db.num_inlier_matches().unwrap(), 0);
        assert_eq!(db.num_verified_image_pairs().unwrap(), 1);
        db.delete_two_view_geometry(1, 2).unwrap();
        assert!(!db.exists_two_view_geometry(1, 2).unwrap());

        db.clear_keypoints().unwrap();
        assert_eq!(db.num_keypoints().unwrap(), 0);
        db.clear_images().unwrap();
        assert_eq!(db.num_images().unwrap(), 0);
        assert!(!db.exists_keypoints(1).unwrap());
        assert_eq!(db.num_descriptors().unwrap(), 0);
    }

    #[test]
    fn database_close_vacuums_after_delete_or_clear_like_colmap() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("database.db");
        {
            let db = ColmapDatabase::open(&path).unwrap();
            write_test_camera(&db, 1);
            write_test_images(&db, 1, &[1, 2]);
            let keypoints = (0..400)
                .map(|idx| ColmapKeypoint::new(idx as f32, idx as f32 + 1.0))
                .collect::<Vec<_>>();
            db.write_keypoints(1, &keypoints).unwrap();
            db.write_keypoints(2, &keypoints).unwrap();
            db.write_matches(1, 2, &(0..400).map(|idx| m(idx, idx)).collect::<Vec<_>>())
                .unwrap();
            db.clear_keypoints().unwrap();
            db.close().unwrap();
        }

        let conn = Connection::open(&path).unwrap();
        let freelist_count = conn
            .query_row("PRAGMA freelist_count;", [], |row| row.get::<_, i64>(0))
            .unwrap();
        assert_eq!(freelist_count, 0);
    }

    #[test]
    fn database_update_methods_replace_existing_table_payloads() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();
        write_test_camera(&db, 1);
        db.update_camera(&ColmapDatabaseCamera {
            camera: ColmapCamera {
                camera_id: 1,
                model_id: crate::types::COLMAP_PINHOLE,
                width: 200,
                height: 150,
                params: vec![70.0, 71.0, 100.0, 75.0],
            },
            has_prior_focal_length: false,
        })
        .unwrap();
        assert_eq!(db.read_camera(1).unwrap().unwrap().camera.width, 200);
        assert!(!db.read_camera(1).unwrap().unwrap().has_prior_focal_length);

        db.write_image(
            &ColmapDatabaseImage {
                image_id: 10,
                name: "old.jpg".to_string(),
                camera_id: 1,
                frame_id: None,
            },
            true,
        )
        .unwrap();
        db.update_image(&ColmapDatabaseImage {
            image_id: 10,
            name: "new.jpg".to_string(),
            camera_id: 1,
            frame_id: None,
        })
        .unwrap();
        assert!(db.exists_image_with_name("new.jpg").unwrap());
        assert!(!db.exists_image_with_name("old.jpg").unwrap());

        db.write_keypoints(10, &[ColmapKeypoint::new(0.0, 0.0)])
            .unwrap();
        db.update_keypoints(
            10,
            &[ColmapKeypoint::new(1.0, 1.0), ColmapKeypoint::new(2.0, 2.0)],
        )
        .unwrap();
        assert_eq!(db.num_keypoints_for_image(10).unwrap(), 2);

        let pose_prior = ColmapPosePrior {
            pose_prior_id: 3,
            corr_data_id: ColmapDataId {
                data_id: 10,
                sensor_id: ColmapSensorId {
                    sensor_type: ColmapSensorType::Camera,
                    sensor_id: 1,
                },
            },
            position: [0.0, 0.0, 0.0],
            position_covariance: [f64::NAN; 9],
            coordinate_system: ColmapPosePriorCoordinateSystem::Undefined,
            gravity: [f64::NAN; 3],
        };
        db.write_pose_prior(&pose_prior, true).unwrap();
        let mut updated_pose_prior = pose_prior;
        updated_pose_prior.position = [3.0, 2.0, 1.0];
        updated_pose_prior.coordinate_system = ColmapPosePriorCoordinateSystem::Cartesian;
        db.update_pose_prior(&updated_pose_prior).unwrap();
        assert_eq!(
            db.read_pose_prior(3).unwrap().unwrap().position,
            [3.0, 2.0, 1.0]
        );

        let ref_sensor = ColmapSensorId {
            sensor_type: ColmapSensorType::Camera,
            sensor_id: 1,
        };
        db.write_rig(
            &ColmapRig {
                rig_id: 5,
                ref_sensor_id: Some(ref_sensor.clone()),
                sensors: vec![ColmapRigSensor {
                    sensor_id: ref_sensor.clone(),
                    sensor_from_rig: None,
                }],
            },
            true,
        )
        .unwrap();
        let imu_sensor = ColmapSensorId {
            sensor_type: ColmapSensorType::Imu,
            sensor_id: 8,
        };
        db.update_rig(&ColmapRig {
            rig_id: 5,
            ref_sensor_id: Some(ref_sensor),
            sensors: vec![
                ColmapRigSensor {
                    sensor_id: ColmapSensorId {
                        sensor_type: ColmapSensorType::Camera,
                        sensor_id: 1,
                    },
                    sensor_from_rig: None,
                },
                ColmapRigSensor {
                    sensor_id: imu_sensor.clone(),
                    sensor_from_rig: Some(ColmapRigid3 {
                        qvec: [1.0, 0.0, 0.0, 0.0],
                        tvec: [1.0, 2.0, 3.0],
                    }),
                },
            ],
        })
        .unwrap();
        assert_eq!(
            db.read_rig(5).unwrap().unwrap().sensors[0].sensor_id,
            imu_sensor
        );

        db.write_frame(
            &ColmapDatabaseFrame {
                frame_id: 6,
                rig_id: 5,
                data_ids: vec![ColmapDataId {
                    sensor_id: ColmapSensorId {
                        sensor_type: ColmapSensorType::Camera,
                        sensor_id: 1,
                    },
                    data_id: 10,
                }],
            },
            true,
        )
        .unwrap();
        db.update_frame(&ColmapDatabaseFrame {
            frame_id: 6,
            rig_id: 5,
            data_ids: vec![ColmapDataId {
                sensor_id: ColmapSensorId {
                    sensor_type: ColmapSensorType::Camera,
                    sensor_id: 1,
                },
                data_id: 11,
            }],
        })
        .unwrap();
        assert_eq!(db.read_frame(6).unwrap().unwrap().data_ids[0].data_id, 11);
    }

    #[test]
    fn database_merge_remaps_ids_and_preserves_feature_pair_payloads() {
        let dir = tempdir().unwrap();
        let db1 = ColmapDatabase::open(dir.path().join("database1.db")).unwrap();
        let db2 = ColmapDatabase::open(dir.path().join("database2.db")).unwrap();
        let merged = ColmapDatabase::open(dir.path().join("merged.db")).unwrap();

        let source_camera1 = ColmapDatabaseCamera {
            camera: ColmapCamera {
                camera_id: 11,
                model_id: crate::types::COLMAP_PINHOLE,
                width: 640,
                height: 480,
                params: vec![500.0, 501.0, 320.0, 240.0],
            },
            has_prior_focal_length: true,
        };
        let source_camera2 = ColmapDatabaseCamera {
            camera: ColmapCamera {
                camera_id: 22,
                model_id: crate::types::COLMAP_SIMPLE_PINHOLE,
                width: 320,
                height: 240,
                params: vec![300.0, 160.0, 120.0],
            },
            has_prior_focal_length: false,
        };
        db1.write_camera(&source_camera1, true).unwrap();
        db2.write_camera(&source_camera2, true).unwrap();

        db1.write_image(
            &ColmapDatabaseImage {
                image_id: 101,
                name: "a/left.jpg".to_string(),
                camera_id: 11,
                frame_id: None,
            },
            true,
        )
        .unwrap();
        db1.write_image(
            &ColmapDatabaseImage {
                image_id: 103,
                name: "a/right.jpg".to_string(),
                camera_id: 11,
                frame_id: None,
            },
            true,
        )
        .unwrap();
        db2.write_image(
            &ColmapDatabaseImage {
                image_id: 201,
                name: "b/left.jpg".to_string(),
                camera_id: 22,
                frame_id: None,
            },
            true,
        )
        .unwrap();
        db2.write_image(
            &ColmapDatabaseImage {
                image_id: 205,
                name: "b/right.jpg".to_string(),
                camera_id: 22,
                frame_id: None,
            },
            true,
        )
        .unwrap();

        db1.write_keypoints(
            101,
            &[
                ColmapKeypoint::new(1.0, 2.0),
                ColmapKeypoint::from_scale_orientation(3.0, 4.0, 5.0, 6.0),
            ],
        )
        .unwrap();
        db1.write_keypoints(103, &[ColmapKeypoint::new(7.0, 8.0)])
            .unwrap();
        db2.write_keypoints(201, &[ColmapKeypoint::new(9.0, 10.0)])
            .unwrap();
        db2.write_keypoints(
            205,
            &[
                ColmapKeypoint::new(11.0, 12.0),
                ColmapKeypoint::new(13.0, 14.0),
            ],
        )
        .unwrap();

        db1.write_descriptors(
            101,
            &ColmapDescriptors::new(COLMAP_FEATURE_SIFT, 2, 2, vec![1, 2, 3, 4]).unwrap(),
        )
        .unwrap();
        db1.write_descriptors(
            103,
            &ColmapDescriptors::new(COLMAP_FEATURE_ALIKED_N16ROT, 1, 2, vec![5, 6]).unwrap(),
        )
        .unwrap();
        db2.write_descriptors(
            201,
            &ColmapDescriptors::new(COLMAP_FEATURE_ALIKED_N32, 1, 2, vec![7, 8]).unwrap(),
        )
        .unwrap();
        db2.write_descriptors(
            205,
            &ColmapDescriptors::new(COLMAP_FEATURE_SIFT, 2, 2, vec![9, 10, 11, 12]).unwrap(),
        )
        .unwrap();

        db1.write_rig(
            &ColmapRig {
                rig_id: 31,
                ref_sensor_id: Some(ColmapSensorId {
                    sensor_type: ColmapSensorType::Camera,
                    sensor_id: 11,
                }),
                sensors: vec![ColmapRigSensor {
                    sensor_id: ColmapSensorId {
                        sensor_type: ColmapSensorType::Camera,
                        sensor_id: 11,
                    },
                    sensor_from_rig: None,
                }],
            },
            true,
        )
        .unwrap();
        db2.write_rig(
            &ColmapRig {
                rig_id: 41,
                ref_sensor_id: Some(ColmapSensorId {
                    sensor_type: ColmapSensorType::Camera,
                    sensor_id: 22,
                }),
                sensors: vec![ColmapRigSensor {
                    sensor_id: ColmapSensorId {
                        sensor_type: ColmapSensorType::Camera,
                        sensor_id: 22,
                    },
                    sensor_from_rig: None,
                }],
            },
            true,
        )
        .unwrap();
        db1.write_frame(
            &ColmapDatabaseFrame {
                frame_id: 51,
                rig_id: 31,
                data_ids: vec![ColmapDataId {
                    sensor_id: ColmapSensorId {
                        sensor_type: ColmapSensorType::Camera,
                        sensor_id: 11,
                    },
                    data_id: 101,
                }],
            },
            true,
        )
        .unwrap();
        db2.write_frame(
            &ColmapDatabaseFrame {
                frame_id: 61,
                rig_id: 41,
                data_ids: vec![ColmapDataId {
                    sensor_id: ColmapSensorId {
                        sensor_type: ColmapSensorType::Camera,
                        sensor_id: 22,
                    },
                    data_id: 205,
                }],
            },
            true,
        )
        .unwrap();
        db1.write_pose_prior(
            &ColmapPosePrior {
                pose_prior_id: 71,
                corr_data_id: ColmapDataId {
                    sensor_id: ColmapSensorId {
                        sensor_type: ColmapSensorType::Camera,
                        sensor_id: 11,
                    },
                    data_id: 101,
                },
                position: [1.0, 2.0, 3.0],
                position_covariance: [f64::NAN; 9],
                coordinate_system: ColmapPosePriorCoordinateSystem::Cartesian,
                gravity: [0.0, 0.0, 1.0],
            },
            true,
        )
        .unwrap();

        db1.write_matches(101, 103, &[m(0, 0), m(1, 0)]).unwrap();
        db2.write_matches(205, 201, &[]).unwrap();
        let two_view = ColmapTwoViewGeometry {
            config: COLMAP_TWO_VIEW_CALIBRATED,
            inlier_matches: vec![m(0, 0)],
            f_matrix: Some([1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]),
            qvec: Some([1.0, 0.0, 0.0, 0.0]),
            tvec: Some([0.1, 0.2, 0.3]),
            ..ColmapTwoViewGeometry::default()
        };
        db1.write_two_view_geometry(101, 103, &two_view).unwrap();

        ColmapDatabase::merge(&db1, &db2, &merged).unwrap();

        assert_eq!(merged.num_cameras().unwrap(), 2);
        assert_eq!(merged.num_images().unwrap(), 4);
        assert_eq!(merged.num_frames().unwrap(), 2);
        assert_eq!(merged.num_pose_priors().unwrap(), 1);
        assert_eq!(merged.num_matched_image_pairs().unwrap(), 1);
        assert_eq!(merged.num_verified_image_pairs().unwrap(), 1);

        let a_left = merged
            .read_image_with_name("a/left.jpg")
            .unwrap()
            .expect("merged image");
        let a_right = merged
            .read_image_with_name("a/right.jpg")
            .unwrap()
            .expect("merged image");
        let b_left = merged
            .read_image_with_name("b/left.jpg")
            .unwrap()
            .expect("merged image");
        let b_right = merged
            .read_image_with_name("b/right.jpg")
            .unwrap()
            .expect("merged image");
        assert_eq!(
            [
                a_left.image_id,
                a_right.image_id,
                b_left.image_id,
                b_right.image_id
            ],
            [1, 2, 3, 4]
        );
        assert_eq!(
            [
                a_left.camera_id,
                a_right.camera_id,
                b_left.camera_id,
                b_right.camera_id
            ],
            [1, 1, 2, 2]
        );
        assert_eq!(a_left.frame_id, Some(1));
        assert_eq!(b_right.frame_id, Some(2));

        assert_eq!(merged.read_keypoints(a_left.image_id).unwrap()[0].x, 1.0);
        assert_eq!(merged.read_keypoints(b_right.image_id).unwrap().len(), 2);
        assert_eq!(
            merged
                .read_descriptors(a_right.image_id)
                .unwrap()
                .feature_type,
            COLMAP_FEATURE_ALIKED_N16ROT
        );
        assert_eq!(
            merged.read_descriptors(b_right.image_id).unwrap().data,
            vec![9, 10, 11, 12]
        );
        assert_eq!(
            merged
                .read_matches(a_left.image_id, a_right.image_id)
                .unwrap(),
            vec![m(0, 0), m(1, 0)]
        );
        assert!(!merged
            .exists_matches(b_left.image_id, b_right.image_id)
            .unwrap());
        assert_eq!(
            merged
                .read_two_view_geometry(a_left.image_id, a_right.image_id)
                .unwrap(),
            two_view
        );

        let frames = merged.read_all_frames().unwrap();
        assert_eq!(frames[0].rig_id, 1);
        assert_eq!(frames[0].data_ids[0].sensor_id.sensor_id, 1);
        assert_eq!(frames[0].data_ids[0].data_id, a_left.image_id as u64);
        assert_eq!(frames[1].rig_id, 2);
        assert_eq!(frames[1].data_ids[0].sensor_id.sensor_id, 2);
        assert_eq!(frames[1].data_ids[0].data_id, b_right.image_id as u64);

        let pose_prior = merged.read_all_pose_priors().unwrap().pop().unwrap();
        assert_eq!(pose_prior.pose_prior_id, 1);
        assert_eq!(pose_prior.corr_data_id.sensor_id.sensor_id, 1);
        assert_eq!(pose_prior.corr_data_id.data_id, a_left.image_id as u64);
        assert_eq!(pose_prior.position, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn database_merge_rejects_duplicate_image_names() {
        let dir = tempdir().unwrap();
        let db1 = ColmapDatabase::open(dir.path().join("database1.db")).unwrap();
        let db2 = ColmapDatabase::open(dir.path().join("database2.db")).unwrap();
        let merged = ColmapDatabase::open(dir.path().join("merged.db")).unwrap();
        write_test_camera(&db1, 1);
        write_test_camera(&db2, 1);
        db1.write_image(
            &ColmapDatabaseImage {
                image_id: 1,
                name: "shared.jpg".to_string(),
                camera_id: 1,
                frame_id: None,
            },
            true,
        )
        .unwrap();
        db2.write_image(
            &ColmapDatabaseImage {
                image_id: 2,
                name: "shared.jpg".to_string(),
                camera_id: 1,
                frame_id: None,
            },
            true,
        )
        .unwrap();

        let err = ColmapDatabase::merge(&db1, &db2, &merged).unwrap_err();

        assert!(err
            .to_string()
            .contains("must not contain images with the same name"));
    }

    #[test]
    fn builds_correspondence_graph_from_database_two_view_geometries() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();
        write_test_camera(&db, 1);
        write_test_images(&db, 1, &[1, 2, 3]);
        let keypoints = vec![
            ColmapKeypoint::new(0.0, 0.0),
            ColmapKeypoint::new(1.0, 1.0),
            ColmapKeypoint::new(2.0, 2.0),
        ];
        db.write_keypoints(1, &keypoints).unwrap();
        db.write_keypoints(2, &keypoints).unwrap();
        db.write_keypoints(3, &keypoints).unwrap();
        db.write_two_view_geometry(
            1,
            2,
            &ColmapTwoViewGeometry {
                config: 2,
                inlier_matches: vec![m(0, 0), m(1, 1), m(3, 2)],
                ..ColmapTwoViewGeometry::default()
            },
        )
        .unwrap();
        db.write_two_view_geometry(
            2,
            3,
            &ColmapTwoViewGeometry {
                config: 2,
                inlier_matches: vec![m(1, 2)],
                ..ColmapTwoViewGeometry::default()
            },
        )
        .unwrap();

        let graph = db.build_correspondence_graph().unwrap();

        assert_eq!(graph.num_images(), 3);
        assert_eq!(graph.num_image_pairs(), 2);
        assert_eq!(graph.num_observations_for_image(1).unwrap(), 2);
        assert_eq!(graph.num_matches_between_images(1, 2).unwrap(), 2);
        assert_eq!(
            graph.extract_matches_between_images(1, 2).unwrap(),
            vec![m(0, 0), m(1, 1)]
        );
        let mut transitive = graph.extract_transitive_correspondences(1, 1, 2).unwrap();
        transitive.sort();
        assert_eq!(
            transitive,
            vec![
                crate::correspondence_graph::Correspondence::new(2, 1),
                crate::correspondence_graph::Correspondence::new(3, 2),
            ]
        );
    }

    #[test]
    fn reads_legacy_two_and_four_column_keypoint_blobs() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();
        write_test_camera(&db, 1);
        write_test_images(&db, 1, &[1, 2]);
        let data_2 = encode_f32_blob(&[1.0, 2.0, 3.0, 4.0]);
        db.conn
            .execute(
                "INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?1, ?2, ?3, ?4);",
                params![1u32, 2i64, 2i64, data_2],
            )
            .unwrap();
        let data_4 = encode_f32_blob(&[5.0, 6.0, 7.0, 8.0]);
        db.conn
            .execute(
                "INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?1, ?2, ?3, ?4);",
                params![2u32, 1i64, 4i64, data_4],
            )
            .unwrap();

        assert_eq!(
            db.read_keypoints(1).unwrap(),
            vec![ColmapKeypoint::new(1.0, 2.0), ColmapKeypoint::new(3.0, 4.0)]
        );
        assert_eq!(
            db.read_keypoints(2).unwrap(),
            vec![ColmapKeypoint::from_scale_orientation(5.0, 6.0, 7.0, 8.0)]
        );
    }

    #[test]
    fn open_migrates_legacy_inlier_matches_and_two_view_blob_columns() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("database.db");
        let pair_id = image_pair_to_pair_id(1, 2).unwrap() as i64;
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE inlier_matches
                    (pair_id INTEGER PRIMARY KEY NOT NULL,
                     rows INTEGER NOT NULL,
                     cols INTEGER NOT NULL,
                     data BLOB,
                     config INTEGER NOT NULL);
                 PRAGMA user_version = 3130000;",
            )
            .unwrap();
            conn.execute(
                "INSERT INTO inlier_matches(pair_id, rows, cols, data, config)
                 VALUES(?1, ?2, ?3, ?4, ?5);",
                params![pair_id, 1i64, 2i64, encode_matches_blob(&[m(3, 4)]), 2i32],
            )
            .unwrap();
        }

        let db = ColmapDatabase::open(&path).unwrap();

        assert!(db.exists_table("two_view_geometries").unwrap());
        assert!(!db.exists_table("inlier_matches").unwrap());
        for column_name in ["F", "E", "H", "qvec", "tvec"] {
            assert!(db
                .exists_column("two_view_geometries", column_name)
                .unwrap());
        }
        let geometry = db.read_two_view_geometry(1, 2).unwrap();
        assert_eq!(geometry.config, COLMAP_TWO_VIEW_CALIBRATED);
        assert_eq!(geometry.inlier_matches, vec![m(3, 4)]);
        assert_eq!(db.user_version().unwrap(), COLMAP_CURRENT_DATABASE_VERSION);
    }

    #[test]
    fn open_migrates_legacy_two_view_pose_and_matrix_sentinels() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("database.db");
        let pair_id = image_pair_to_pair_id(1, 2).unwrap() as i64;
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE two_view_geometries
                    (pair_id INTEGER PRIMARY KEY NOT NULL,
                     rows INTEGER NOT NULL,
                     cols INTEGER NOT NULL,
                     data BLOB,
                     config INTEGER NOT NULL,
                     F BLOB,
                     E BLOB,
                     H BLOB,
                     qvec BLOB,
                     tvec BLOB);
                 PRAGMA user_version = 3130000;",
            )
            .unwrap();
            conn.execute(
                "INSERT INTO two_view_geometries(
                    pair_id, rows, cols, data, config, F, E, H, qvec, tvec)
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10);",
                params![
                    pair_id,
                    0i64,
                    2i64,
                    Vec::<u8>::new(),
                    COLMAP_TWO_VIEW_CALIBRATED,
                    encode_matrix3_colmap_blob([0.0; 9]),
                    encode_matrix3_colmap_blob([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
                    encode_matrix3_colmap_blob([0.0; 9]),
                    encode_f64_blob(&[1.0, 0.0, 0.0, 0.0]),
                    encode_f64_blob(&[0.0, 0.0, 0.0])
                ],
            )
            .unwrap();
        }

        let db = ColmapDatabase::open(&path).unwrap();
        let geometry = db.read_two_view_geometry(1, 2).unwrap();

        assert_eq!(geometry.f_matrix, None);
        assert_eq!(geometry.h_matrix, None);
        assert_eq!(
            geometry.e_matrix,
            Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        );
        assert_eq!(geometry.qvec, None);
        assert_eq!(geometry.tvec, None);
    }

    #[test]
    fn open_migrates_legacy_descriptor_type_column_to_sift() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("database.db");
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE descriptors
                    (image_id INTEGER PRIMARY KEY NOT NULL,
                     rows INTEGER NOT NULL,
                     cols INTEGER NOT NULL,
                     data BLOB);
                 PRAGMA user_version = 3140001;",
            )
            .unwrap();
            conn.execute(
                "INSERT INTO descriptors(image_id, rows, cols, data)
                 VALUES(?1, ?2, ?3, ?4);",
                params![7u32, 1i64, 4i64, vec![1u8, 2, 3, 4]],
            )
            .unwrap();
        }

        let db = ColmapDatabase::open(&path).unwrap();

        assert!(db.exists_column("descriptors", "type").unwrap());
        assert_eq!(
            db.read_descriptors(7).unwrap(),
            ColmapDescriptors::new(COLMAP_FEATURE_SIFT, 1, 4, vec![1, 2, 3, 4]).unwrap()
        );
    }

    #[test]
    fn open_migrates_legacy_image_pose_priors_to_sensor_pose_priors() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("database.db");
        let position = encode_f64_blob(&[1.0, 2.0, 3.0]);
        let covariance = encode_f64_blob(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE cameras
                    (camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                     model INTEGER NOT NULL,
                     width INTEGER NOT NULL,
                     height INTEGER NOT NULL,
                     params BLOB,
                     prior_focal_length INTEGER NOT NULL);
                 CREATE TABLE images
                    (image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                     name TEXT NOT NULL UNIQUE,
                     camera_id INTEGER NOT NULL);
                 CREATE TABLE pose_priors
                    (image_id INTEGER PRIMARY KEY NOT NULL,
                     position BLOB,
                     position_covariance BLOB,
                     coordinate_system INTEGER NOT NULL);
                 PRAGMA user_version = 3130000;",
            )
            .unwrap();
            conn.execute(
                "INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length)
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6);",
                params![
                    3u32,
                    crate::types::COLMAP_PINHOLE,
                    100u32,
                    100u32,
                    encode_f64_blob(&[50.0, 50.0, 50.0, 50.0]),
                    1i32
                ],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO images(image_id, name, camera_id) VALUES(?1, ?2, ?3);",
                params![42u32, "prior.jpg", 3u32],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO pose_priors(
                    image_id, position, position_covariance, coordinate_system)
                 VALUES(?1, ?2, ?3, ?4);",
                params![
                    42u32,
                    position,
                    covariance,
                    coordinate_system_to_i64(&ColmapPosePriorCoordinateSystem::Cartesian).unwrap()
                ],
            )
            .unwrap();
        }

        let db = ColmapDatabase::open(&path).unwrap();
        let pose_priors = db.read_all_pose_priors().unwrap();

        assert_eq!(pose_priors.len(), 1);
        assert!(!db.exists_table("pose_priors_old").unwrap());
        assert_eq!(pose_priors[0].pose_prior_id, 42);
        assert_eq!(
            pose_priors[0].corr_data_id,
            ColmapDataId {
                data_id: 42,
                sensor_id: ColmapSensorId {
                    sensor_type: ColmapSensorType::Camera,
                    sensor_id: 3,
                },
            }
        );
        assert_eq!(pose_priors[0].position, [1.0, 2.0, 3.0]);
        assert_eq!(
            pose_priors[0].coordinate_system,
            ColmapPosePriorCoordinateSystem::Cartesian
        );
        assert!(pose_priors[0].gravity.iter().all(|value| value.is_nan()));
    }
}
