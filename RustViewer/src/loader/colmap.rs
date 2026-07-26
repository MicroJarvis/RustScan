//! Loader and scene mapping utilities for COLMAP datasets.

use std::path::{Path, PathBuf};

use rustgs::{load_colmap_dataset as rustgs_load_colmap_dataset, ColmapConfig, TrainingDataset};

use crate::loader::checkpoint::LoadError;
use crate::renderer::scene::Scene;

#[derive(Debug, Clone)]
pub struct ColmapDatasetSummary {
    pub input_dir: PathBuf,
    pub sparse_dir: PathBuf,
    pub image_dir: PathBuf,
    pub frame_count: usize,
    pub sparse_point_count: usize,
    pub intrinsics: rustgs::Intrinsics,
}

#[derive(Debug, Clone)]
pub struct LoadedColmapDataset {
    pub dataset: TrainingDataset,
    pub summary: ColmapDatasetSummary,
}

/// Load a COLMAP directory into a RustGS training dataset with summary metadata.
pub fn load_colmap_training_dataset(
    input: &Path,
    config: &ColmapConfig,
) -> Result<LoadedColmapDataset, LoadError> {
    let sparse_dir = resolve_sparse_dir(input)?;
    let image_dir = resolve_image_dir(&sparse_dir)?;

    let dataset = rustgs_load_colmap_dataset(input, config)
        .map_err(|err| classify_colmap_error(input, err))?;

    Ok(LoadedColmapDataset {
        summary: ColmapDatasetSummary {
            input_dir: input.to_path_buf(),
            sparse_dir,
            image_dir,
            frame_count: dataset.poses.len(),
            sparse_point_count: dataset.initial_points.len(),
            intrinsics: dataset.intrinsics,
        },
        dataset,
    })
}

/// Map dataset trajectory and sparse points into the viewer scene.
pub fn map_training_dataset_to_scene(dataset: &TrainingDataset, scene: &mut Scene) {
    scene.trajectory.clear();
    scene.map_points.clear();
    scene.map_point_colors.clear();

    for pose in &dataset.poses {
        scene.trajectory.push(pose.pose.translation());
    }

    let y_min = dataset
        .initial_points
        .iter()
        .map(|(position, _)| position[1])
        .fold(f32::INFINITY, f32::min);
    let y_max = dataset
        .initial_points
        .iter()
        .map(|(position, _)| position[1])
        .fold(f32::NEG_INFINITY, f32::max);
    let y_range = (y_max - y_min).max(1e-6);

    for (position, color) in &dataset.initial_points {
        scene.map_points.push(*position);
        scene.map_point_colors.push(color.unwrap_or_else(|| {
            let t = ((position[1] - y_min) / y_range).clamp(0.0, 1.0);
            [t, 1.0 - t, 0.2]
        }));
    }

    scene.recompute_bounds();
}

/// Load COLMAP data and immediately map it into the viewer scene.
pub fn load_colmap_into_scene(
    input: &Path,
    config: &ColmapConfig,
    scene: &mut Scene,
) -> Result<LoadedColmapDataset, LoadError> {
    let loaded = load_colmap_training_dataset(input, config)?;
    map_training_dataset_to_scene(&loaded.dataset, scene);
    Ok(loaded)
}

fn resolve_sparse_dir(input: &Path) -> Result<PathBuf, LoadError> {
    let candidates = [
        input.to_path_buf(),
        input.join("sparse"),
        input.join("sparse").join("0"),
    ];

    for candidate in &candidates {
        if is_sparse_dir(candidate) {
            if has_sparse_points(candidate) {
                return Ok(candidate.clone());
            }

            return Err(LoadError::ColmapSparseStructureMissing(format!(
                "missing points3D.bin/points3D.txt in {}",
                candidate.display(),
            )));
        }
    }

    Err(LoadError::ColmapSparseStructureMissing(format!(
        "could not find COLMAP sparse reconstruction in {}",
        input.display(),
    )))
}

fn resolve_image_dir(sparse_dir: &Path) -> Result<PathBuf, LoadError> {
    let mut candidates = vec![sparse_dir.join("images")];
    if let Some(parent) = sparse_dir.parent() {
        candidates.push(parent.join("images"));
        if let Some(grand_parent) = parent.parent() {
            candidates.push(grand_parent.join("images"));
        }
    }

    for candidate in candidates {
        if candidate.is_dir() {
            return Ok(candidate);
        }
    }

    Err(LoadError::ColmapImagesMissing(format!(
        "could not find images directory for sparse path {}",
        sparse_dir.display(),
    )))
}

fn classify_colmap_error(input: &Path, err: rustgs::TrainingError) -> LoadError {
    match err {
        rustgs::TrainingError::InvalidInput(message) => {
            if is_sparse_error_message(&message) {
                return LoadError::ColmapSparseStructureMissing(message);
            }
            if is_images_error_message(&message) {
                return LoadError::ColmapImagesMissing(message);
            }
            LoadError::ColmapLoadFailed(message)
        }
        rustgs::TrainingError::Io(source) => LoadError::ColmapLoadFailed(format!(
            "failed to read COLMAP inputs under {}: {}",
            input.display(),
            source
        )),
        rustgs::TrainingError::Gpu(message) | rustgs::TrainingError::TrainingFailed(message) => {
            LoadError::ColmapLoadFailed(message)
        }
    }
}

fn is_sparse_dir(path: &Path) -> bool {
    path.is_dir() && has_cameras(path) && has_images(path)
}

fn has_cameras(path: &Path) -> bool {
    path.join("cameras.bin").is_file() || path.join("cameras.txt").is_file()
}

fn has_images(path: &Path) -> bool {
    path.join("images.bin").is_file() || path.join("images.txt").is_file()
}

fn has_sparse_points(path: &Path) -> bool {
    path.join("points3D.bin").is_file() || path.join("points3D.txt").is_file()
}

fn is_sparse_error_message(message: &str) -> bool {
    message.contains("could not find COLMAP sparse reconstruction")
        || message.contains("missing COLMAP sparse points")
        || message.contains("no sparse points found")
        || message.contains("no cameras file found")
        || message.contains("no images file found")
        || message.contains("no cameras found in COLMAP dataset")
        || message.contains("no images found in COLMAP dataset")
}

fn is_images_error_message(message: &str) -> bool {
    message.contains("could not find images directory")
        || message.contains("no valid frames found")
        || message.contains("image")
            && message.contains("not found")
            && message.contains("after image path validation")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_colmap_text_dataset(root: &Path) {
        let sparse = root.join("sparse").join("0");
        std::fs::create_dir_all(&sparse).unwrap();
        let images = root.join("images");
        std::fs::create_dir_all(&images).unwrap();

        std::fs::write(
            sparse.join("cameras.txt"),
            "# Camera list with one line of data per camera:\n1 PINHOLE 1920 1080 1500 1500 960 540\n",
        )
        .unwrap();

        std::fs::write(
            sparse.join("images.txt"),
            "# Image list with two lines of data per image:\n1 1.0 0.0 0.0 0.0 0.0 0.0 1.0 1 frame_0001.jpg\n\n2 1.0 0.0 0.0 0.0 1.0 0.0 2.0 1 frame_0002.jpg\n\n",
        )
        .unwrap();

        std::fs::write(
            sparse.join("points3D.txt"),
            "# 3D point list with one line of data per point:\n1 0.0 0.0 1.0 128 128 128 0.1\n2 1.0 2.0 1.0 64 64 64 0.1\n",
        )
        .unwrap();

        let placeholder: Vec<u8> = vec![0u8; 1920 * 1080 * 3];
        std::fs::write(images.join("frame_0001.jpg"), &placeholder).unwrap();
        std::fs::write(images.join("frame_0002.jpg"), &placeholder).unwrap();
    }

    #[test]
    fn loads_colmap_and_maps_scene() {
        let temp = tempfile::tempdir().unwrap();
        write_colmap_text_dataset(temp.path());

        let mut scene = Scene::default();
        let loaded = load_colmap_into_scene(temp.path(), &ColmapConfig::default(), &mut scene)
            .expect("colmap dataset should load");

        assert_eq!(loaded.summary.frame_count, 2);
        assert_eq!(loaded.summary.sparse_point_count, 2);
        assert_eq!(scene.trajectory.len(), 2);
        assert_eq!(scene.trajectory[0], [0.0, 0.0, -1.0]);
        assert_eq!(scene.trajectory[1], [-1.0, 0.0, -2.0]);
        assert_eq!(scene.map_points.len(), 2);
        assert_eq!(scene.map_point_colors.len(), 2);
        assert!(scene.bounds.is_valid());
    }

    #[test]
    fn returns_sparse_structure_error_when_sparse_files_missing() {
        let temp = tempfile::tempdir().unwrap();
        let err = load_colmap_training_dataset(temp.path(), &ColmapConfig::default()).unwrap_err();
        assert!(matches!(err, LoadError::ColmapSparseStructureMissing(_)));
    }

    #[test]
    fn returns_images_missing_error_when_images_directory_missing() {
        let temp = tempfile::tempdir().unwrap();
        let sparse = temp.path().join("sparse").join("0");
        std::fs::create_dir_all(&sparse).unwrap();
        std::fs::write(
            sparse.join("cameras.txt"),
            "1 PINHOLE 640 480 500 500 320 240\n",
        )
        .unwrap();
        std::fs::write(
            sparse.join("images.txt"),
            "1 1.0 0.0 0.0 0.0 0.0 0.0 1.0 1 frame_0001.jpg\n",
        )
        .unwrap();
        std::fs::write(
            sparse.join("points3D.txt"),
            "1 0.0 0.0 1.0 255 255 255 0.1\n",
        )
        .unwrap();

        let err = load_colmap_training_dataset(temp.path(), &ColmapConfig::default()).unwrap_err();
        assert!(matches!(err, LoadError::ColmapImagesMissing(_)));
    }
}
