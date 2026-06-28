use crate::colmap_image::load_colmap_grayscale_u8;
use crate::correspondence_graph::FeatureMatch;
use crate::database::{ColmapDatabase, ColmapDatabaseImage, ColmapDescriptors, ColmapKeypoint};
use crate::feature_matching::{generate_matching_pairs, MatchingPairStrategy};
use crate::geometry::{estimate_pair_geometry_with_options_and_cameras, PairEstimationOptions};
use crate::mapper::pair_geometry_to_colmap_two_view_geometry;
use crate::sift::{match_sift_with_options, SiftFeatures, SiftMatchingOptions};
use crate::types::{CameraModel, ImageFrame};
use anyhow::{bail, Context, Result};
use lowe_sift::Descriptor;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchFeaturesPairReport {
    pub left_image: String,
    pub right_image: String,
    pub num_matches: usize,
    pub num_inliers: usize,
    pub triangulated: usize,
    pub two_view_config: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchFeaturesReport {
    pub database: PathBuf,
    pub pair_count: usize,
    pub matched_pairs: usize,
    pub verified_pairs: usize,
    pub total_matches: usize,
    pub matching_seconds: f64,
    pub pairs: Vec<MatchFeaturesPairReport>,
}

#[derive(Debug, Clone)]
pub struct MatchFeaturesOptions {
    pub pair_strategy: MatchingPairStrategy,
    pub sift_matching: SiftMatchingOptions,
    pub essential_threshold_px: f32,
    pub essential_iterations: u32,
    pub min_inliers: usize,
    pub min_triangulated: usize,
    pub min_num_matches: usize,
    pub random_seed: i32,
    pub clear_existing: bool,
}

impl Default for MatchFeaturesOptions {
    fn default() -> Self {
        Self {
            pair_strategy: MatchingPairStrategy::default(),
            sift_matching: SiftMatchingOptions::default(),
            essential_threshold_px: 2.0,
            essential_iterations: 10_000,
            min_inliers: 15,
            min_triangulated: 4,
            min_num_matches: 15,
            random_seed: -1,
            clear_existing: true,
        }
    }
}

pub fn match_features_to_database(
    database_path: &Path,
    options: &MatchFeaturesOptions,
) -> Result<MatchFeaturesReport> {
    options.sift_matching.check()?;
    let started = Instant::now();
    let db = ColmapDatabase::open(database_path)?;
    if options.clear_existing {
        db.clear_matches()?;
        db.clear_two_view_geometries()?;
    }

    let mut images = db.read_all_images()?;
    if images.len() < 2 {
        bail!("database needs at least two images for matching");
    }
    images.sort_by(|left, right| left.name.cmp(&right.name));

    let mut frames = Vec::with_capacity(images.len());
    let mut cameras = Vec::with_capacity(images.len());
    for (idx, image) in images.iter().enumerate() {
        let db_camera = db
            .read_camera(image.camera_id)?
            .with_context(|| format!("missing camera_id={}", image.camera_id))?;
        let camera = CameraModel::from_colmap(
            db_camera.camera.model_id,
            db_camera.camera.width,
            db_camera.camera.height,
            &db_camera.camera.params,
        )
        .with_context(|| format!("unsupported camera_id={}", image.camera_id))?;
        let keypoints = db.read_keypoints(image.image_id)?;
        let descriptors = db.read_descriptors(image.image_id)?;
        let sift = sift_features_from_database(&keypoints, &descriptors)?;
        let rust_keypoints = keypoints.iter().map(|kp| kp.to_keypoint()).collect::<Vec<_>>();
        frames.push(ImageFrame {
            id: idx,
            name: image.name.clone(),
            path: PathBuf::from(&image.name),
            width: db_camera.camera.width,
            height: db_camera.camera.height,
            keypoints: rust_keypoints,
            descriptors: rustslam::Descriptors::new(),
            sift,
            wide_descriptors: crate::wide::WideDescriptors {
                data: Vec::new(),
                dim: 0,
                count: 0,
            },
            strong_feature_indices: Vec::new(),
            colors: Vec::new(),
        });
        cameras.push(camera);
    }

    let image_id_by_index = images
        .iter()
        .enumerate()
        .map(|(idx, image)| (idx, image.image_id))
        .collect::<Vec<_>>();

    let pairs = match options.pair_strategy {
        MatchingPairStrategy::VocabTree { num_images } => {
            vocab_tree_pairs_from_frames(&frames, num_images, options.random_seed)
        }
        strategy => generate_matching_pairs(frames.len(), strategy),
    };
    let pair_reports = pairs
        .par_iter()
        .filter_map(|&(left, right)| {
            let matches = match_sift_with_options(
                &frames[left].sift,
                &frames[right].sift,
                &options.sift_matching,
            );
            if matches.len() < options.min_num_matches {
                return None;
            }
            let geometry = estimate_pair_geometry_with_options_and_cameras(
                left,
                right,
                &frames[left],
                &frames[right],
                &matches,
                cameras[left],
                cameras[right],
                options.essential_threshold_px,
                options.essential_iterations,
                options.min_inliers,
                options.min_triangulated,
                PairEstimationOptions {
                    max_pose_matches: 0,
                    ransac_random_seed: options.random_seed,
                    ..PairEstimationOptions::default()
                },
            );
            Some((
                left,
                right,
                matches,
                geometry,
            ))
        })
        .collect::<Vec<_>>();

    let mut reports = Vec::with_capacity(pair_reports.len());
    let mut total_matches = 0usize;
    for (left, right, matches, geometry) in pair_reports {
        let left_image_id = image_id_by_index[left].1;
        let right_image_id = image_id_by_index[right].1;
        let feature_matches = matches
            .iter()
            .map(|match_| FeatureMatch {
                point2d_idx1: match_.query_idx,
                point2d_idx2: match_.train_idx,
            })
            .collect::<Vec<_>>();
        if db.exists_matches(left_image_id, right_image_id)? {
            db.delete_matches(left_image_id, right_image_id)?;
        }
        db.write_matches(left_image_id, right_image_id, &feature_matches)?;
        total_matches += matches.len();

        let (inliers, triangulated, config) = if let Some(geometry) = geometry.as_ref() {
            let colmap_geometry = pair_geometry_to_colmap_two_view_geometry(geometry);
            if db.exists_two_view_geometry(left_image_id, right_image_id)? {
                db.update_two_view_geometry(left_image_id, right_image_id, &colmap_geometry)?;
            } else {
                db.write_two_view_geometry(left_image_id, right_image_id, &colmap_geometry)?;
            }
            (
                geometry.inliers,
                geometry.triangulated,
                geometry.two_view_config,
            )
        } else {
            if db.exists_two_view_geometry(left_image_id, right_image_id)? {
                db.delete_two_view_geometry(left_image_id, right_image_id)?;
            }
            (0, 0, -1)
        };

        reports.push(MatchFeaturesPairReport {
            left_image: frames[left].name.clone(),
            right_image: frames[right].name.clone(),
            num_matches: matches.len(),
            num_inliers: inliers,
            triangulated,
            two_view_config: config,
        });
    }
    reports.sort_by(|left, right| {
        left.left_image
            .cmp(&right.left_image)
            .then_with(|| left.right_image.cmp(&right.right_image))
    });

    Ok(MatchFeaturesReport {
        database: database_path.to_path_buf(),
        pair_count: pairs.len(),
        matched_pairs: reports.len(),
        verified_pairs: reports
            .iter()
            .filter(|pair| pair.num_inliers >= options.min_inliers)
            .count(),
        total_matches,
        matching_seconds: started.elapsed().as_secs_f64(),
        pairs: reports,
    })
}

/// Build vocabulary-tree candidate pairs from the in-memory frame descriptors.
///
/// Frames are keyed by their slice index, so the returned `(i32, i32)` image-id
/// pairs map directly back to `(left, right)` frame indices. The vocabulary
/// tree is trained on the union of all frame SIFT descriptors.
pub fn vocab_tree_pairs_from_frames(
    frames: &[ImageFrame],
    num_images: usize,
    random_seed: i32,
) -> Vec<(usize, usize)> {
    let dim = lowe_sift::DESCRIPTOR_LEN;
    let images: Vec<(i32, Vec<f32>)> = frames
        .iter()
        .enumerate()
        .map(|(idx, frame)| {
            let mut descriptors = Vec::with_capacity(frame.sift.descriptors_u8.len() * dim);
            for desc in &frame.sift.descriptors_u8 {
                descriptors.extend(desc.iter().map(|&b| b as f32));
            }
            (idx as i32, descriptors)
        })
        .collect();

    let build_options = crate::retrieval::VocabTreeBuildOptions {
        random_seed: if random_seed < 0 { 0 } else { random_seed as i64 },
        ..crate::retrieval::VocabTreeBuildOptions::default()
    };
    let pair_options = crate::retrieval::VocabTreePairOptions {
        num_images: num_images.max(1),
    };
    crate::retrieval::build_vocab_tree_pairs(&images, dim, &build_options, &pair_options)
        .into_iter()
        .filter_map(|(a, b)| {
            let (a, b) = (a as usize, b as usize);
            (a < frames.len() && b < frames.len()).then_some((a.min(b), a.max(b)))
        })
        .collect()
}

fn sift_features_from_database(
    keypoints: &[ColmapKeypoint],
    descriptors: &ColmapDescriptors,
) -> Result<SiftFeatures> {
    const DESCRIPTOR_LEN: usize = 128;
    if descriptors.cols != DESCRIPTOR_LEN {
        bail!(
            "expected SIFT descriptors with {} columns, got {}",
            DESCRIPTOR_LEN,
            descriptors.cols
        );
    }
    if keypoints.len() != descriptors.rows {
        bail!(
            "keypoint/descriptor row mismatch: {} vs {}",
            keypoints.len(),
            descriptors.rows
        );
    }

    let mut descriptors_u8 = Vec::with_capacity(descriptors.rows);
    let mut float_descriptors = Vec::with_capacity(descriptors.rows);
    for row in 0..descriptors.rows {
        let start = row * DESCRIPTOR_LEN;
        let end = start + DESCRIPTOR_LEN;
        let mut values = [0u8; DESCRIPTOR_LEN];
        values.copy_from_slice(&descriptors.data[start..end]);
        let mut floats = [0f32; DESCRIPTOR_LEN];
        for (slot, value) in floats.iter_mut().zip(values.iter()) {
            *slot = *value as f32 / 512.0;
        }
        descriptors_u8.push(values);
        float_descriptors.push(Descriptor::new(floats));
    }

    Ok(SiftFeatures {
        keypoints: keypoints.iter().map(|kp| kp.to_keypoint()).collect(),
        descriptors: float_descriptors,
        colmap_keypoints: keypoints.to_vec(),
        descriptors_u8,
    })
}

pub fn load_rgb_image_for_frame(path: &Path) -> Result<(Vec<u8>, u32, u32)> {
    use image::ImageReader;

    let decoded = ImageReader::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?
        .decode()
        .with_context(|| format!("failed to decode {}", path.display()))?
        .to_rgb8();
    let (width, height) = decoded.dimensions();
    Ok((decoded.into_raw(), width, height))
}

pub fn load_grayscale_image_for_frame(path: &Path) -> Result<(Vec<u8>, u32, u32)> {
    if cfg!(colmap_freeimage) {
        let gray = load_colmap_grayscale_u8(path)?;
        return Ok((gray.data, gray.width, gray.height));
    }
    let rgb = load_rgb_image_for_frame(path)?;
    crate::sift::prepare_grayscale_for_extraction(
        &crate::sift::rgb_to_colmap_gray_u8(&rgb.0, rgb.1, rgb.2)?,
        rgb.1,
        rgb.2,
        crate::sift::SiftExtractionOptions::default().max_image_size,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sift_features_from_database_roundtrips_rows() -> Result<()> {
        let keypoints = vec![ColmapKeypoint::new(1.0, 2.0), ColmapKeypoint::new(3.0, 4.0)];
        let descriptors = ColmapDescriptors::new(
            crate::database::COLMAP_FEATURE_SIFT,
            keypoints.len(),
            128,
            (0..keypoints.len() * 128).map(|v| v as u8).collect(),
        )?;
        let features = sift_features_from_database(&keypoints, &descriptors)?;
        assert_eq!(features.keypoints.len(), 2);
        assert_eq!(features.descriptors_u8.len(), 2);
        Ok(())
    }
}
