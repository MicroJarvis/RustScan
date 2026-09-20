use super::{pair_geometry_to_colmap_two_view_geometry, FeatureType, ReferenceCameraSetup};
use crate::colmap::ColmapCamera;
use crate::correspondence_graph::FeatureMatch;
use crate::database::{
    ColmapDatabase, ColmapDatabaseCamera, ColmapDatabaseImage, ColmapDescriptors, ColmapKeypoint,
    COLMAP_FEATURE_SIFT,
};
use crate::types::{ImageFrame, PairGeometry};
use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

pub(super) fn write_pair_geometries_to_database(
    database_path: &Path,
    frames: &[ImageFrame],
    pairs: &[PairGeometry],
) -> Result<usize> {
    let db = ColmapDatabase::open(database_path)?;
    let image_by_name = db
        .read_all_images()?
        .into_iter()
        .map(|image| (image.name, image.image_id))
        .collect::<HashMap<_, _>>();
    let mut written = 0usize;
    for pair in pairs {
        let Some(left_frame) = frames.get(pair.left) else {
            continue;
        };
        let Some(right_frame) = frames.get(pair.right) else {
            continue;
        };
        let Some(&left_image_id) = image_by_name.get(&left_frame.name) else {
            continue;
        };
        let Some(&right_image_id) = image_by_name.get(&right_frame.name) else {
            continue;
        };
        let geometry = pair_geometry_to_colmap_two_view_geometry(pair);
        if db.exists_two_view_geometry(left_image_id, right_image_id)? {
            db.update_two_view_geometry(left_image_id, right_image_id, &geometry)?;
        } else {
            db.write_two_view_geometry(left_image_id, right_image_id, &geometry)?;
        }
        written += 1;
    }
    Ok(written)
}

pub(super) fn populate_local_matching_database(
    database_path: &Path,
    frames: &[ImageFrame],
    setup: &ReferenceCameraSetup,
    pairs: &[PairGeometry],
    feature_type: FeatureType,
) -> Result<usize> {
    if let Some(parent) = database_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let db = ColmapDatabase::open(database_path)?;
    let mut written = 0usize;
    for (camera_idx, camera) in setup.cameras.iter().enumerate() {
        let camera_id = setup.camera_ids[camera_idx];
        db.write_camera(
            &ColmapDatabaseCamera {
                camera: ColmapCamera {
                    camera_id,
                    model_id: camera.model_id,
                    width: camera.width,
                    height: camera.height,
                    params: camera.params[..camera.num_params].to_vec(),
                },
                has_prior_focal_length: setup
                    .camera_has_prior_focal_length
                    .get(camera_idx)
                    .copied()
                    .unwrap_or(true),
            },
            true,
        )?;
        written += 1;
    }
    for (frame_idx, frame) in frames.iter().enumerate() {
        let image_id = setup.image_ids[frame_idx];
        let camera_id = setup.camera_ids[setup.image_camera_indices[frame_idx]];
        db.write_image(
            &ColmapDatabaseImage {
                image_id,
                name: frame.name.clone(),
                camera_id,
                frame_id: None,
            },
            true,
        )?;
        written += 1;
        let keypoints = frame_keypoints_for_database(frame, feature_type);
        db.write_keypoints(image_id, &keypoints)?;
        written += 1;
        let descriptors = frame_descriptors_for_database(frame, feature_type)?;
        db.write_descriptors(image_id, &descriptors)?;
        written += 1;
    }
    for pair in pairs {
        let left_image_id = setup.image_ids[pair.left];
        let right_image_id = setup.image_ids[pair.right];
        let matches = pair
            .matches
            .iter()
            .map(|match_| FeatureMatch {
                point2d_idx1: match_.query_idx,
                point2d_idx2: match_.train_idx,
            })
            .collect::<Vec<_>>();
        if db.exists_matches(left_image_id, right_image_id)? {
            db.delete_matches(left_image_id, right_image_id)?;
        }
        if !matches.is_empty() {
            db.write_matches(left_image_id, right_image_id, &matches)?;
            written += 1;
        }
        let geometry = pair_geometry_to_colmap_two_view_geometry(pair);
        if db.exists_two_view_geometry(left_image_id, right_image_id)? {
            db.update_two_view_geometry(left_image_id, right_image_id, &geometry)?;
        } else {
            db.write_two_view_geometry(left_image_id, right_image_id, &geometry)?;
        }
        written += 1;
    }
    Ok(written)
}

fn frame_keypoints_for_database(
    frame: &ImageFrame,
    feature_type: FeatureType,
) -> Vec<ColmapKeypoint> {
    match feature_type {
        FeatureType::Sift => {
            if !frame.sift.colmap_keypoints.is_empty() {
                return frame.sift.colmap_keypoints.clone();
            }
            frame
                .sift
                .keypoints
                .iter()
                .map(ColmapKeypoint::from)
                .collect()
        }
        FeatureType::Orb => frame.keypoints.iter().map(ColmapKeypoint::from).collect(),
    }
}

fn frame_descriptors_for_database(
    frame: &ImageFrame,
    feature_type: FeatureType,
) -> Result<ColmapDescriptors> {
    match feature_type {
        FeatureType::Sift => {
            let rows = frame.sift.descriptors.len();
            const DESCRIPTOR_LEN: usize = 128;
            let cols = DESCRIPTOR_LEN;
            let mut data = Vec::with_capacity(rows.saturating_mul(cols));
            for descriptor in &frame.sift.descriptors {
                for value in descriptor.as_slice() {
                    data.push((value.clamp(0.0, 1.0) * 512.0).round() as u8);
                }
            }
            ColmapDescriptors::new(COLMAP_FEATURE_SIFT, rows, cols, data)
        }
        FeatureType::Orb => Ok(ColmapDescriptors::from_rustslam(
            COLMAP_FEATURE_SIFT,
            &frame.descriptors,
        )),
    }
}
