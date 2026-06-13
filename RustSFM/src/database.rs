use crate::correspondence_graph::{
    image_pair_to_pair_id, should_swap_image_pair, FeatureMatch, ImageId, ImagePairId,
};
use anyhow::{bail, Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use rustslam::{Descriptors, KeyPoint};
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

impl ColmapDatabase {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path).context("open COLMAP database")?;
        let db = Self { conn };
        db.create_feature_tables()?;
        Ok(db)
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
