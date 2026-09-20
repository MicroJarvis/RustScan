//! SLAM checkpointing utilities for saving and resuming pipeline state.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::core::{Frame, FrameFeatures, KeyFrame, Map, MapPoint, SE3};
use nalgebra::Point3;

#[derive(Debug, Error)]
pub enum CheckpointError {
    #[error("failed to create checkpoint directory {path}: {source}")]
    CreateDir {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to write checkpoint {path}: {source}")]
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to read checkpoint {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to serialize checkpoint {path}: {source}")]
    Serialize {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("failed to deserialize checkpoint {path}: {source}")]
    Deserialize {
        path: PathBuf,
        source: serde_json::Error,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlamCheckpoint {
    pub version: u32,
    pub frame_index: usize,
    pub keyframes: Vec<CheckpointKeyFrame>,
    pub map_points: Vec<CheckpointMapPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointKeyFrame {
    pub id: u64,
    pub timestamp: f64,
    pub pose: Option<CheckpointPose>,
    pub width: u32,
    pub height: u32,
    pub is_keyframe: bool,
    pub features: CheckpointFeatures,
    pub connected_keyframes: Vec<(u64, u32)>,
    pub bow_vector: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointPose {
    pub translation: [f32; 3],
    pub quaternion: [f32; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointFeatures {
    pub keypoints: Vec<[f32; 2]>,
    pub descriptors: Vec<u8>,
    pub map_points: Vec<Option<u64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMapPoint {
    pub id: u64,
    pub position: [f32; 3],
    #[serde(default)]
    pub color: Option<[f32; 3]>,
    pub reference_kf: u64,
    pub observations: u32,
    pub is_outlier: bool,
}

#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    pub dir: PathBuf,
    pub interval: usize,
}

pub struct CheckpointManager {
    config: CheckpointConfig,
    last_saved_frame: usize,
}

impl CheckpointManager {
    pub fn new(config: CheckpointConfig) -> Result<Self, CheckpointError> {
        if !config.dir.exists() {
            fs::create_dir_all(&config.dir).map_err(|source| CheckpointError::CreateDir {
                path: config.dir.clone(),
                source,
            })?;
        }

        Ok(Self {
            config,
            last_saved_frame: 0,
        })
    }

    pub fn maybe_save(
        &mut self,
        frame_index: usize,
        map: &Map,
    ) -> Result<PathBuf, CheckpointError> {
        if self.config.interval == 0 {
            return Ok(PathBuf::new());
        }

        if frame_index < self.last_saved_frame + self.config.interval {
            return Ok(PathBuf::new());
        }

        let checkpoint = SlamCheckpoint::from_map(frame_index, map);
        let path = checkpoint_path(&self.config.dir, frame_index);
        save_checkpoint(&checkpoint, &path)?;
        self.last_saved_frame = frame_index;
        Ok(path)
    }
}

impl SlamCheckpoint {
    pub fn from_map(frame_index: usize, map: &Map) -> Self {
        let mut keyframes = Vec::new();
        for kf in map.keyframes() {
            keyframes.push(CheckpointKeyFrame::from_keyframe(kf));
        }

        let mut map_points = Vec::new();
        for mp in map.points() {
            map_points.push(CheckpointMapPoint::from_map_point(mp));
        }

        Self {
            version: 1,
            frame_index,
            keyframes,
            map_points,
        }
    }

    pub fn to_map(&self) -> Map {
        let mut map = Map::new();
        let mut max_kf = 0u64;
        let mut max_mp = 0u64;

        for mp in &self.map_points {
            let point = mp.to_map_point();
            map.insert_point_with_id(mp.id, point);
            max_mp = max_mp.max(mp.id);
        }

        for kf in &self.keyframes {
            let keyframe = kf.to_keyframe();
            map.insert_keyframe_with_id(kf.id, keyframe);
            max_kf = max_kf.max(kf.id);
        }

        map.set_next_ids(max_mp + 1, max_kf + 1);
        map
    }
}

impl CheckpointKeyFrame {
    fn from_keyframe(kf: &KeyFrame) -> Self {
        let pose = kf.pose().map(|p| CheckpointPose {
            translation: p.translation(),
            quaternion: p.quaternion(),
        });

        Self {
            id: kf.id(),
            timestamp: kf.frame.timestamp,
            pose,
            width: kf.frame.width,
            height: kf.frame.height,
            is_keyframe: kf.frame.is_keyframe,
            features: CheckpointFeatures {
                keypoints: kf.features.keypoints.clone(),
                descriptors: kf.features.descriptors.clone(),
                map_points: kf.features.map_points.clone(),
            },
            connected_keyframes: kf.connected_keyframes.clone(),
            bow_vector: kf.bow_vector.clone(),
        }
    }

    fn to_keyframe(&self) -> KeyFrame {
        let mut frame = Frame::new(self.id, self.timestamp, self.width, self.height);
        if let Some(pose) = &self.pose {
            let se3 = SE3::new(&pose.quaternion, &pose.translation);
            frame.set_pose(se3);
        }
        if self.is_keyframe {
            frame.mark_as_keyframe();
        }

        let features = FrameFeatures {
            keypoints: self.features.keypoints.clone(),
            descriptors: self.features.descriptors.clone(),
            map_points: self.features.map_points.clone(),
        };

        let mut kf = KeyFrame::new(frame, features);
        kf.connected_keyframes = self.connected_keyframes.clone();
        kf.bow_vector = self.bow_vector.clone();
        kf
    }
}

impl CheckpointMapPoint {
    fn from_map_point(mp: &MapPoint) -> Self {
        Self {
            id: mp.id,
            position: [mp.position.x, mp.position.y, mp.position.z],
            color: mp.color,
            reference_kf: mp.reference_kf,
            observations: mp.observations,
            is_outlier: mp.is_outlier,
        }
    }

    fn to_map_point(&self) -> MapPoint {
        MapPoint {
            id: self.id,
            position: Point3::new(self.position[0], self.position[1], self.position[2]),
            normal: None,
            color: self.color,
            reference_kf: self.reference_kf,
            observations: self.observations,
            is_outlier: self.is_outlier,
        }
    }
}

pub fn save_checkpoint(checkpoint: &SlamCheckpoint, path: &Path) -> Result<(), CheckpointError> {
    let file = File::create(path).map_err(|source| CheckpointError::Write {
        path: path.to_path_buf(),
        source,
    })?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, checkpoint).map_err(|source| {
        CheckpointError::Serialize {
            path: path.to_path_buf(),
            source,
        }
    })?;
    Ok(())
}

pub fn load_checkpoint(path: &Path) -> Result<SlamCheckpoint, CheckpointError> {
    let file = File::open(path).map_err(|source| CheckpointError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let reader = BufReader::new(file);
    serde_json::from_reader(reader).map_err(|source| CheckpointError::Deserialize {
        path: path.to_path_buf(),
        source,
    })
}

pub fn load_latest_checkpoint(dir: &Path) -> Result<Option<SlamCheckpoint>, CheckpointError> {
    if !dir.exists() {
        return Ok(None);
    }

    let mut best: Option<(usize, PathBuf)> = None;
    for entry in fs::read_dir(dir).map_err(|source| CheckpointError::Read {
        path: dir.to_path_buf(),
        source,
    })? {
        let entry = entry.map_err(|source| CheckpointError::Read {
            path: dir.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        if let Some(frame_index) = parse_checkpoint_name(&path) {
            if best
                .as_ref()
                .map(|(idx, _)| frame_index > *idx)
                .unwrap_or(true)
            {
                best = Some((frame_index, path));
            }
        }
    }

    let Some((_idx, path)) = best else {
        return Ok(None);
    };
    load_checkpoint(&path).map(Some)
}

pub fn checkpoint_path(dir: &Path, frame_index: usize) -> PathBuf {
    dir.join(format!("slam_{frame_index}.ckpt"))
}

fn parse_checkpoint_name(path: &Path) -> Option<usize> {
    let file_name = path.file_name()?.to_string_lossy();
    if !file_name.starts_with("slam_") || !file_name.ends_with(".ckpt") {
        return None;
    }
    let number = &file_name[5..file_name.len() - 5];
    number.parse::<usize>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Frame, FrameFeatures, KeyFrame, Map, MapPoint, SE3};
    use nalgebra::Point3;

    fn build_map() -> Map {
        let mut map = Map::new();
        let mut frame = Frame::new(0, 0.0, 640, 480);
        frame.set_pose(SE3::identity());
        frame.mark_as_keyframe();

        let features = FrameFeatures {
            keypoints: vec![[10.0, 12.0], [20.0, 24.0]],
            descriptors: vec![1, 2, 3, 4, 5, 6, 7, 8],
            map_points: vec![None, None],
        };

        let keyframe = KeyFrame::new(frame, features);
        let kf_id = map.add_keyframe(keyframe);

        let mut map_point = MapPoint::new(0, Point3::new(1.0, 2.0, 3.0), kf_id);
        map_point.set_color([0.1, 0.2, 0.3]);
        map.add_point(map_point);
        map
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let map = build_map();
        let checkpoint = SlamCheckpoint::from_map(12, &map);

        let dir = tempfile::tempdir().unwrap();
        let path = checkpoint_path(dir.path(), 12);
        save_checkpoint(&checkpoint, &path).unwrap();

        let loaded = load_checkpoint(&path).unwrap();
        assert_eq!(loaded.frame_index, 12);

        let mut loaded_map = loaded.to_map();
        assert_eq!(loaded_map.num_keyframes(), 1);
        assert_eq!(loaded_map.num_points(), 1);

        let keyframe = loaded_map.get_keyframe(0).unwrap();
        assert_eq!(keyframe.features.keypoints.len(), 2);
        assert_eq!(keyframe.features.descriptors.len(), 8);

        let map_point = loaded_map.get_point(0).unwrap();
        assert_eq!(map_point.color, Some([0.1, 0.2, 0.3]));

        let new_id = loaded_map.add_point(MapPoint::new(1, Point3::origin(), 0));
        assert_eq!(new_id, 1);
    }

    #[test]
    fn test_load_latest_checkpoint() {
        let map = build_map();

        let dir = tempfile::tempdir().unwrap();
        let ckpt1 = SlamCheckpoint::from_map(3, &map);
        let ckpt2 = SlamCheckpoint::from_map(8, &map);

        save_checkpoint(&ckpt1, &checkpoint_path(dir.path(), 3)).unwrap();
        save_checkpoint(&ckpt2, &checkpoint_path(dir.path(), 8)).unwrap();

        let latest = load_latest_checkpoint(dir.path()).unwrap().unwrap();
        assert_eq!(latest.frame_index, 8);
    }
}
