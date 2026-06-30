use crate::colmap_image::load_colmap_grayscale_u8;
use crate::compare::{compare_feature_counts, FeaturesCompareReport};
use crate::database::{ColmapDatabase, ColmapDescriptors, ColmapKeypoint, COLMAP_FEATURE_SIFT};
use crate::sift::{extract_sift_from_grayscale_u8, SiftExtractionOptions, SiftFeatures};
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractFeaturesImageReport {
    pub image_name: String,
    pub num_keypoints: usize,
    pub elapsed_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractFeaturesReport {
    pub database: PathBuf,
    pub images_dir: PathBuf,
    pub backend: &'static str,
    pub image_count: usize,
    pub total_keypoints: usize,
    pub mean_keypoints: f64,
    pub extraction_seconds: f64,
    pub images: Vec<ExtractFeaturesImageReport>,
}

pub fn sift_features_to_colmap_keypoints(features: &SiftFeatures) -> Vec<ColmapKeypoint> {
    if !features.colmap_keypoints.is_empty() {
        return features.colmap_keypoints.clone();
    }
    features
        .keypoints
        .iter()
        .map(ColmapKeypoint::from)
        .collect()
}

pub fn sift_features_to_colmap_descriptors(features: &SiftFeatures) -> Result<ColmapDescriptors> {
    const DESCRIPTOR_LEN: usize = 128;
    let rows = features.descriptors_u8.len();
    let data = features
        .descriptors_u8
        .iter()
        .flat_map(|descriptor| descriptor.iter().copied())
        .collect::<Vec<_>>();
    ColmapDescriptors::new(COLMAP_FEATURE_SIFT, rows, DESCRIPTOR_LEN, data)
}

pub fn extract_features_to_database(
    database_path: &Path,
    images_dir: &Path,
    options: &SiftExtractionOptions,
) -> Result<ExtractFeaturesReport> {
    options.check()?;
    let db = ColmapDatabase::open(database_path)?;
    let images = db.read_all_images()?;
    if images.is_empty() {
        bail!("database has no images; import images before feature extraction");
    }

    let started = Instant::now();
    let mut reports = Vec::with_capacity(images.len());
    for image in images {
        let image_path = images_dir.join(&image.name);
        if !image_path.exists() {
            bail!(
                "missing image file for database image {}: {}",
                image.name,
                image_path.display()
            );
        }
        let extract_started = Instant::now();
        let decoded = load_colmap_grayscale_u8(&image_path)
            .with_context(|| format!("failed to load {}", image_path.display()))?;
        let features =
            extract_sift_from_grayscale_u8(&decoded.data, decoded.width, decoded.height, options)?;
        let keypoints = sift_features_to_colmap_keypoints(&features);
        let descriptors = sift_features_to_colmap_descriptors(&features)?;
        db.upsert_keypoints(image.image_id, &keypoints)?;
        db.upsert_descriptors(image.image_id, &descriptors)?;
        reports.push(ExtractFeaturesImageReport {
            image_name: image.name.clone(),
            num_keypoints: keypoints.len(),
            elapsed_ms: extract_started.elapsed().as_secs_f64() * 1000.0,
        });
    }
    reports.sort_unstable_by(|left, right| left.image_name.cmp(&right.image_name));

    let total_keypoints = reports
        .iter()
        .map(|image| image.num_keypoints)
        .sum::<usize>();
    let image_count = reports.len();
    let mean_keypoints = if image_count == 0 {
        0.0
    } else {
        total_keypoints as f64 / image_count as f64
    };

    Ok(ExtractFeaturesReport {
        database: database_path.to_path_buf(),
        images_dir: images_dir.to_path_buf(),
        backend: sift_backend_name(),
        image_count,
        total_keypoints,
        mean_keypoints,
        extraction_seconds: started.elapsed().as_secs_f64(),
        images: reports,
    })
}

pub fn compare_extracted_sift_features(
    reference_database: &Path,
    images_dir: &Path,
    options: &SiftExtractionOptions,
) -> Result<FeaturesCompareReport> {
    options.check()?;
    let reference = reference_keypoint_counts(reference_database)?;
    let candidate = extracted_keypoint_counts(images_dir, options)?;
    compare_feature_counts(&reference, &candidate)
}

fn reference_keypoint_counts(database_path: &Path) -> Result<HashMap<String, usize>> {
    let db = ColmapDatabase::open(database_path)?;
    db.read_keypoint_counts()?
        .into_iter()
        .map(|(image_id, count)| {
            let image = db
                .read_image(image_id)?
                .with_context(|| format!("missing image_id={image_id}"))?;
            Ok((image.name, count))
        })
        .collect()
}

fn extracted_keypoint_counts(
    images_dir: &Path,
    options: &SiftExtractionOptions,
) -> Result<HashMap<String, usize>> {
    let mut paths = std::fs::read_dir(images_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| {
                    matches!(
                        ext.to_ascii_lowercase().as_str(),
                        "jpg" | "jpeg" | "png" | "bmp" | "tif" | "tiff" | "webp"
                    )
                })
        })
        .collect::<Vec<_>>();
    paths.sort();

    paths
        .par_iter()
        .map(|path| -> Result<(String, usize)> {
            let name = path
                .file_name()
                .map(|value| value.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string());
            let decoded = load_colmap_grayscale_u8(path)
                .with_context(|| format!("failed to load {}", path.display()))?;
            let features = extract_sift_from_grayscale_u8(
                &decoded.data,
                decoded.width,
                decoded.height,
                options,
            )?;
            Ok((name, features.keypoints.len()))
        })
        .collect()
}

fn sift_backend_name() -> &'static str {
    if cfg!(all(
        feature = "vlfeat-sift",
        not(feature = "lowe-sift-backend")
    )) {
        "vlfeat"
    } else {
        "lowe-sift"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colmap::ColmapCamera;
    use crate::database::{ColmapDatabase, ColmapDatabaseCamera, ColmapDatabaseImage};
    use crate::types::COLMAP_PINHOLE;
    use tempfile::tempdir;

    #[test]
    fn extract_features_to_database_updates_existing_rows() -> Result<()> {
        let dir = tempdir()?;
        let images_dir = dir.path().join("images");
        std::fs::create_dir_all(&images_dir)?;
        let image_path = images_dir.join("left.jpg");
        write_checkerboard_image(&image_path, 256, 256)?;

        let db_path = dir.path().join("database.db");
        let db = ColmapDatabase::open(&db_path)?;
        db.write_camera(
            &ColmapDatabaseCamera {
                camera: ColmapCamera {
                    camera_id: 1,
                    model_id: COLMAP_PINHOLE,
                    width: 256,
                    height: 256,
                    params: vec![200.0, 200.0, 128.0, 128.0],
                },
                has_prior_focal_length: true,
            },
            true,
        )?;
        db.write_image(
            &ColmapDatabaseImage {
                image_id: 1,
                name: "left.jpg".to_string(),
                camera_id: 1,
                frame_id: None,
            },
            true,
        )?;
        db.write_keypoints(1, &[ColmapKeypoint::new(1.0, 1.0)])?;
        db.write_descriptors(
            1,
            &ColmapDescriptors::new(COLMAP_FEATURE_SIFT, 1, 128, vec![0u8; 128])?,
        )?;

        let report = extract_features_to_database(
            &db_path,
            &images_dir,
            &SiftExtractionOptions {
                max_num_features: 256,
                ..SiftExtractionOptions::default()
            },
        )?;
        assert_eq!(report.image_count, 1);
        assert!(report.total_keypoints > 1);

        let updated = db.read_keypoints(1)?;
        assert_eq!(updated.len(), report.total_keypoints);
        let descriptors = db.read_descriptors(1)?;
        assert_eq!(descriptors.rows, report.total_keypoints);
        Ok(())
    }

    fn write_checkerboard_image(path: &Path, width: u32, height: u32) -> Result<()> {
        let mut image = image::RgbImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let value = if ((x / 16) + (y / 16)) % 2 == 0 {
                    240u8
                } else {
                    20u8
                };
                image.put_pixel(x, y, image::Rgb([value, value, value]));
            }
        }
        image.save(path)?;
        Ok(())
    }
}
