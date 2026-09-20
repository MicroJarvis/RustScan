//! Loader for SLAM checkpoint JSON files.
//!
//! Parses the `pipeline.json` format produced by the RustSLAM pipeline.

use serde::Deserialize;
use std::path::Path;
use thiserror::Error;

use crate::renderer::scene::Scene;

#[derive(Debug, Error)]
pub enum LoadError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("OBJ parse error: {0}")]
    ObjParse(String),
    #[error("PLY parse error: {0}")]
    PlyParse(String),
    #[error("Unsupported file format: {0}")]
    UnsupportedFormat(String),
    #[error("COLMAP sparse structure missing: {0}")]
    ColmapSparseStructureMissing(String),
    #[error("COLMAP images missing: {0}")]
    ColmapImagesMissing(String),
    #[error("COLMAP load failed: {0}")]
    ColmapLoadFailed(String),
}

// ── Mirror structs for pipeline.json ────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct PipelineCheckpoint {
    slam: Option<SlamSection>,
}

#[derive(Debug, Deserialize)]
struct SlamSection {
    keyframes: Vec<KeyframeEntry>,
    #[serde(default)]
    map_points: Vec<MapPointEntry>,
}

#[derive(Debug, Deserialize)]
struct KeyframeEntry {
    pose: PoseEntry,
}

#[derive(Debug, Deserialize)]
struct PoseEntry {
    translation: [f32; 3],
}

#[derive(Debug, Deserialize)]
struct MapPointEntry {
    position: [f32; 3],
    #[serde(default)]
    color: Option<[f32; 3]>,
}

// ────────────────────────────────────────────────────────────────────────────

/// Load a pipeline checkpoint JSON and populate trajectory + map points in the scene.
///
/// Accepts the `pipeline.json` format written by the RustSLAM pipeline
/// (`checkpoints/pipeline.json`).
///
/// # Example
/// ```no_run
/// use std::path::Path;
/// use rust_viewer::loader::checkpoint::load_checkpoint;
/// use rust_viewer::renderer::scene::Scene;
///
/// let mut scene = Scene::default();
/// load_checkpoint(Path::new("pipeline.json"), &mut scene).unwrap();
/// println!("Loaded {} keyframes", scene.trajectory.len());
/// ```
pub fn load_checkpoint(path: &Path, scene: &mut Scene) -> Result<(), LoadError> {
    let data = std::fs::read(path)?;
    let checkpoint: PipelineCheckpoint = serde_json::from_slice(&data)?;

    let Some(slam) = checkpoint.slam else {
        return Ok(());
    };

    // Extract camera trajectory from keyframe poses.
    for kf in &slam.keyframes {
        let t = kf.pose.translation;
        scene.trajectory.push(t);
        scene.bounds.extend(t);
    }

    // Extract map points if present.
    if !slam.map_points.is_empty() {
        let ys: Vec<f32> = slam.map_points.iter().map(|mp| mp.position[1]).collect();
        let y_min = ys.iter().cloned().fold(f32::INFINITY, f32::min);
        let y_max = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let y_range = (y_max - y_min).max(1e-6);

        for mp in &slam.map_points {
            let pos = mp.position;
            scene.map_points.push(pos);
            scene.bounds.extend(pos);

            // Use stored color when available; otherwise depth-shade by Y.
            let color = mp.color.unwrap_or_else(|| {
                let t = ((pos[1] - y_min) / y_range).clamp(0.0, 1.0);
                [t, 1.0 - t, 0.2]
            });
            scene.map_point_colors.push(color);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_load_checkpoint_no_slam() {
        // PipelineCheckpoint with no slam section → empty scene, no error.
        let json = r#"{"version":1,"video_completed":true,"slam_completed":false,"gaussian_completed":false,"mesh_completed":false}"#;
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        tmpfile.write_all(json.as_bytes()).unwrap();

        let mut scene = Scene::default();
        let result = load_checkpoint(tmpfile.path(), &mut scene);
        assert!(
            result.is_ok(),
            "missing slam section should succeed: {:?}",
            result
        );
        assert!(scene.trajectory.is_empty());
        assert!(scene.map_points.is_empty());
        assert!(!scene.bounds.is_valid());
    }

    #[test]
    fn test_load_checkpoint_with_keyframes() {
        // PipelineCheckpoint with slam.keyframes → trajectory populated.
        let json = r#"{
            "version": 1,
            "slam": {
                "camera": {"fx":525,"fy":525,"cx":320,"cy":240,"width":640,"height":480},
                "frame_count": 2,
                "keyframes": [
                    {"index":0,"timestamp":0.0,"width":640,"height":480,
                     "pose":{"rotation":[[1,0,0],[0,1,0],[0,0,1]],"translation":[0.0,0.0,0.0]},
                     "color_path":""},
                    {"index":5,"timestamp":0.16,"width":640,"height":480,
                     "pose":{"rotation":[[1,0,0],[0,1,0],[0,0,1]],"translation":[1.0,2.0,3.0]},
                     "color_path":""}
                ]
            }
        }"#;
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        tmpfile.write_all(json.as_bytes()).unwrap();

        let mut scene = Scene::default();
        let result = load_checkpoint(tmpfile.path(), &mut scene);
        assert!(
            result.is_ok(),
            "checkpoint with keyframes should succeed: {:?}",
            result
        );
        assert_eq!(scene.trajectory.len(), 2);
        assert_eq!(scene.trajectory[0], [0.0, 0.0, 0.0]);
        assert_eq!(scene.trajectory[1], [1.0, 2.0, 3.0]);
        assert!(scene.bounds.is_valid());
    }

    #[test]
    fn test_load_checkpoint_with_map_points() {
        // PipelineCheckpoint with slam.map_points → map points populated with color.
        let json = r#"{
            "version": 1,
            "slam": {
                "camera": {"fx":525,"fy":525,"cx":320,"cy":240,"width":640,"height":480},
                "frame_count": 1,
                "keyframes": [
                    {"index":0,"timestamp":0.0,"width":640,"height":480,
                     "pose":{"rotation":[[1,0,0],[0,1,0],[0,0,1]],"translation":[0.0,0.0,0.0]},
                     "color_path":""}
                ],
                "map_points": [
                    {"position":[1.0,2.0,3.0],"color":[0.1,0.2,0.3]},
                    {"position":[4.0,5.0,6.0]}
                ]
            }
        }"#;
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        tmpfile.write_all(json.as_bytes()).unwrap();

        let mut scene = Scene::default();
        let result = load_checkpoint(tmpfile.path(), &mut scene);
        assert!(
            result.is_ok(),
            "checkpoint with map_points should succeed: {:?}",
            result
        );
        assert_eq!(scene.map_points.len(), 2);
        assert_eq!(scene.map_point_colors.len(), 2);
        // First point has explicit color
        assert_eq!(scene.map_point_colors[0], [0.1, 0.2, 0.3]);
    }
}
