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
use rusqlite::{params, Connection, OptionalExtension, Row, Rows};
use rustslam::{Descriptors, KeyPoint};
use std::collections::BTreeMap;
use std::path::Path;

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
            config: 0,
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
        for match_ in &mut self.inlier_matches {
            std::mem::swap(&mut match_.point2d_idx1, &mut match_.point2d_idx2);
        }
    }
}

pub struct ColmapDatabase {
    conn: Connection,
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

impl ColmapDatabase {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path).context("open COLMAP database")?;
        let db = Self { conn };
        db.create_core_tables()?;
        Ok(db)
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
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS keypoints
                (image_id INTEGER PRIMARY KEY NOT NULL,
                 rows INTEGER NOT NULL,
                 cols INTEGER NOT NULL,
                 data BLOB);
             CREATE TABLE IF NOT EXISTS descriptors
                (image_id INTEGER PRIMARY KEY NOT NULL,
                 type INTEGER NOT NULL,
                 rows INTEGER NOT NULL,
                 cols INTEGER NOT NULL,
                 data BLOB);
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
                "SELECT image_id, name, camera_id, NULL as frame_id
                 FROM images WHERE image_id = ?1;",
                params![image_id],
                read_image_row,
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn read_image_with_name(&self, name: &str) -> Result<Option<ColmapDatabaseImage>> {
        self.conn
            .query_row(
                "SELECT image_id, name, camera_id, NULL as frame_id
                 FROM images WHERE name = ?1;",
                params![name],
                read_image_row,
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn read_all_images(&self) -> Result<Vec<ColmapDatabaseImage>> {
        let mut stmt = self
            .conn
            .prepare("SELECT image_id, name, camera_id, NULL as frame_id FROM images;")?;
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
                    pose_prior_id, corr_data_id, corr_sensor_id, corr_sensor_type,
                    position, position_covariance, coordinate_system, gravity)
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8);",
                params![
                    pose_prior.pose_prior_id,
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
                    pose_prior_id, corr_data_id, corr_sensor_id, corr_sensor_type,
                    position, position_covariance, coordinate_system, gravity)
                 VALUES(NULL, ?1, ?2, ?3, ?4, ?5, ?6, ?7);",
                params![
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
        self.conn.execute(
            "INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?1, ?2, ?3, ?4);",
            params![image_id, rows as i64, cols as i64, data],
        )?;
        Ok(())
    }

    pub fn read_keypoints(&self, image_id: ImageId) -> Result<Vec<ColmapKeypoint>> {
        let row = self
            .conn
            .query_row(
                "SELECT rows, cols, data FROM keypoints WHERE image_id = ?1;",
                params![image_id],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)? as usize,
                        row.get::<_, i64>(1)? as usize,
                        row.get::<_, Vec<u8>>(2)?,
                    ))
                },
            )
            .optional()?;
        let Some((rows, cols, data)) = row else {
            return Ok(Vec::new());
        };
        decode_keypoints_blob(rows, cols, &data)
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
                        row.get::<_, i64>(0)? as usize,
                        row.get::<_, i64>(1)? as usize,
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
            out.push((
                row.get::<_, i64>(0)? as ImageId,
                row.get::<_, i64>(1)? as usize,
            ));
        }
        Ok(out)
    }

    pub fn write_matches(
        &self,
        image_id1: ImageId,
        image_id2: ImageId,
        matches: &[FeatureMatch],
    ) -> Result<()> {
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)
            .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        let stored_matches = maybe_swapped_matches(image_id1, image_id2, matches);
        let data = encode_matches_blob(&stored_matches);
        self.conn.execute(
            "INSERT INTO matches(pair_id, rows, cols, data) VALUES(?1, ?2, ?3, ?4);",
            params![pair_id as i64, stored_matches.len() as i64, 2i64, data],
        )?;
        Ok(())
    }

    pub fn read_matches(
        &self,
        image_id1: ImageId,
        image_id2: ImageId,
    ) -> Result<Vec<FeatureMatch>> {
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)
            .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        let row = self
            .conn
            .query_row(
                "SELECT rows, cols, data FROM matches WHERE pair_id = ?1;",
                params![pair_id as i64],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)? as usize,
                        row.get::<_, i64>(1)? as usize,
                        row.get::<_, Vec<u8>>(2)?,
                    ))
                },
            )
            .optional()?;
        let Some((rows, cols, data)) = row else {
            return Ok(Vec::new());
        };
        let mut matches = decode_matches_blob(rows, cols, &data)?;
        if should_swap_image_pair(image_id1, image_id2) {
            swap_matches(&mut matches);
        }
        Ok(matches)
    }

    pub fn read_all_matches(&self) -> Result<Vec<(ImagePairId, Vec<FeatureMatch>)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT pair_id, rows, cols, data FROM matches WHERE rows > 0;")?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            let pair_id = row.get::<_, i64>(0)? as ImagePairId;
            let rows = row.get::<_, i64>(1)? as usize;
            let cols = row.get::<_, i64>(2)? as usize;
            let data = row.get::<_, Vec<u8>>(3)?;
            out.push((pair_id, decode_matches_blob(rows, cols, &data)?));
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
                        row.get::<_, i64>(0)? as usize,
                        row.get::<_, i64>(1)? as usize,
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
            out.push((
                row.get::<_, i64>(0)? as ImagePairId,
                row.get::<_, i64>(1)? as i32,
            ));
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
            let pair_id = row.get::<_, i64>(0)? as ImagePairId;
            let match_rows = row.get::<_, i64>(1)? as usize;
            let cols = row.get::<_, i64>(2)? as usize;
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
                    TwoViewGeometryRecord {
                        config: geometry.config,
                        inlier_matches: geometry.inlier_matches,
                    },
                )
                .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        }
        graph.finalize().map_err(|err| anyhow::anyhow!("{err:?}"))?;
        Ok(graph)
    }
}

fn read_camera_row(row: &Row<'_>) -> rusqlite::Result<ColmapDatabaseCamera> {
    let camera_id = row.get::<_, i64>(0)? as u32;
    let model_id = row.get::<_, i64>(1)? as i32;
    let width = row.get::<_, i64>(2)? as u32;
    let height = row.get::<_, i64>(3)? as u32;
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

fn read_image_row(row: &Row<'_>) -> rusqlite::Result<ColmapDatabaseImage> {
    Ok(ColmapDatabaseImage {
        image_id: row.get::<_, i64>(0)? as ImageId,
        name: row.get(1)?,
        camera_id: row.get::<_, i64>(2)? as u32,
        frame_id: row.get::<_, Option<i64>>(3)?.map(|id| id as u32),
    })
}

fn read_pose_prior_row(row: &Row<'_>) -> rusqlite::Result<ColmapPosePrior> {
    let position_blob = row.get::<_, Vec<u8>>(4)?;
    let position_covariance_blob = row.get::<_, Vec<u8>>(5)?;
    let gravity_blob = row.get::<_, Vec<u8>>(7)?;
    Ok(ColmapPosePrior {
        pose_prior_id: row.get::<_, i64>(0)? as u32,
        corr_data_id: ColmapDataId {
            data_id: row.get::<_, i64>(1)? as u64,
            sensor_id: ColmapSensorId {
                sensor_id: row.get::<_, i64>(2)? as u32,
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
        let rig_id = row.get::<_, i64>(0)? as u32;
        if let std::collections::btree_map::Entry::Vacant(entry) = rigs.entry(rig_id) {
            entry.insert(ColmapRig {
                rig_id,
                ref_sensor_id: Some(ColmapSensorId {
                    sensor_id: row.get::<_, i64>(1)? as u32,
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
            sensor_id: sensor_id as u32,
            sensor_type: sensor_type_from_i64(row.get::<_, i64>(4)?),
        },
        sensor_from_rig,
    });
    Ok(())
}

fn collect_frame_rows(rows: &mut Rows<'_>) -> Result<Vec<ColmapDatabaseFrame>> {
    let mut frames = BTreeMap::<u32, ColmapDatabaseFrame>::new();
    while let Some(row) = rows.next()? {
        let frame_id = row.get::<_, i64>(0)? as u32;
        if let std::collections::btree_map::Entry::Vacant(entry) = frames.entry(frame_id) {
            entry.insert(ColmapDatabaseFrame {
                frame_id,
                rig_id: row.get::<_, i64>(1)? as u32,
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
        data_id: data_id as u64,
        sensor_id: ColmapSensorId {
            sensor_id: row.get::<_, i64>(3)? as u32,
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

fn to_sql_error(err: anyhow::Error) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(
        0,
        rusqlite::types::Type::Blob,
        Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, err)),
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

fn maybe_swapped_matches(
    image_id1: ImageId,
    image_id2: ImageId,
    matches: &[FeatureMatch],
) -> Vec<FeatureMatch> {
    let mut out = matches.to_vec();
    if should_swap_image_pair(image_id1, image_id2) {
        swap_matches(&mut out);
    }
    out
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn m(point2d_idx1: u32, point2d_idx2: u32) -> FeatureMatch {
        FeatureMatch::new(point2d_idx1, point2d_idx2)
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
    fn keypoints_descriptors_roundtrip_through_colmap_tables() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("database.db");
        let db = ColmapDatabase::open(&path).unwrap();
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
            h_matrix: Some([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
            qvec: Some([1.0, 0.0, 0.1, 0.2]),
            tvec: Some([0.3, 0.4, 0.5]),
        };

        db.write_two_view_geometry(8, 4, &geometry).unwrap();

        assert_eq!(db.read_two_view_geometry(8, 4).unwrap(), geometry);
        let reversed = db.read_two_view_geometry(4, 8).unwrap();
        assert_eq!(reversed.config, geometry.config);
        assert_eq!(reversed.inlier_matches, vec![m(2, 1), m(4, 3)]);
        assert_eq!(reversed.f_matrix, geometry.f_matrix);
        assert_eq!(reversed.h_matrix, geometry.h_matrix);
        assert_eq!(reversed.qvec, geometry.qvec);
        assert_eq!(reversed.tvec, geometry.tvec);
        assert_eq!(
            db.read_two_view_geometry_num_inliers().unwrap(),
            vec![(image_pair_to_pair_id(8, 4).unwrap(), 2)]
        );
    }

    #[test]
    fn builds_correspondence_graph_from_database_two_view_geometries() {
        let dir = tempdir().unwrap();
        let db = ColmapDatabase::open(dir.path().join("database.db")).unwrap();
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
}
