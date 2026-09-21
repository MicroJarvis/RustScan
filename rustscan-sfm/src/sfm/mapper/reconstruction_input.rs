use super::*;
use crate::colmap::{read_colmap_poses, resolve_sparse_dir};

pub(super) fn collect_images(input: &Path, max_images: Option<usize>) -> Result<Vec<PathBuf>> {
    let mut paths = fs::read_dir(input)
        .with_context(|| format!("failed to read {}", input.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.extension()
                .and_then(|e| e.to_str())
                .map(|e| matches!(e.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    paths.sort();
    if let Some(max) = max_images {
        paths.truncate(max);
    }
    Ok(paths)
}

pub(super) fn resolve_mapper_database_path(config: &MapperConfig) -> Result<Option<PathBuf>> {
    if let Some(database) = &config.database {
        if !database.exists() {
            if config.write_database && config.local_matching {
                return Ok(Some(database.clone()));
            }
            bail!("database path does not exist: {}", database.display());
        }
        return Ok(Some(database.clone()));
    }

    if config.discover_database {
        for candidate in default_database_candidates(&config.input) {
            if candidate.exists() {
                return Ok(Some(candidate));
            }
        }
    }

    if config.write_database && config.local_matching {
        return Ok(Some(config.input.join("database.db")));
    }

    Ok(None)
}

pub(super) fn default_database_candidates(input: &Path) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    push_unique_path(&mut candidates, input.join("database.db"));
    if let Some(parent) = input.parent() {
        push_unique_path(&mut candidates, parent.join("database.db"));
    }
    candidates
}

pub(super) fn push_unique_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.iter().any(|existing| existing == &path) {
        paths.push(path);
    }
}

#[derive(Debug, Clone)]
pub struct ReferenceCameraSetup {
    pub cameras: Vec<CameraModel>,
    pub camera_ids: Vec<u32>,
    pub camera_has_prior_focal_length: Vec<bool>,
    pub rigs: Vec<Rig>,
    pub frames: Vec<Frame>,
    pub image_ids: Vec<u32>,
    pub image_camera_indices: Vec<usize>,
    pub image_frame_indices: Vec<Option<usize>>,
    pub seed_reconstruction: Option<ReconstructionSeed>,
}

#[derive(Debug, Clone)]
pub struct ReconstructionSeed {
    pub poses: Vec<Option<SE3>>,
    pub observations: Vec<Vec<Option<usize>>>,
    pub point_ids: Vec<u64>,
    pub points: Vec<Point3D>,
}

/// One database-backed image that survived filtering, in final mapper order.
///
/// `source_index` is the position in the unfiltered input path list. `frame.id`
/// is the index of this record in the retained collection. Later camera, seed,
/// target, and pair lookups must use that retained index or the stable name /
/// database image ID, never the unfiltered path position.
#[derive(Debug, Clone)]
pub(super) struct RetainedMapperImage {
    pub source_index: usize,
    pub name: String,
    pub path: PathBuf,
    pub frame: ImageFrame,
    pub database_image_id: u32,
    pub database_camera_id: u32,
    /// Index into the aligned `ReferenceCameraSetup::cameras` list.
    pub camera_index: usize,
    pub rig_frame_index: Option<usize>,
    /// Index into the reference reconstruction image list, when this image is present there.
    pub reference_image_index: Option<usize>,
}

#[derive(Debug, Clone)]
pub(super) struct MapperDatabaseInput {
    pub(super) cache: DatabaseCache,
    pub(super) keypoints_by_name: HashMap<String, Vec<rustscan_slam::KeyPoint>>,
    pub(super) two_view_geometries: HashMap<ImagePairId, ColmapTwoViewGeometry>,
}

pub(super) fn setup_for_reconstruction_attempt(
    setup: Option<&ReferenceCameraSetup>,
    use_seed: bool,
) -> Option<ReferenceCameraSetup> {
    let mut setup = setup.cloned()?;
    if !use_seed {
        setup.seed_reconstruction = None;
    }
    Some(setup)
}

pub fn reference_camera_setup(
    reference: &Path,
    image_paths: &[PathBuf],
) -> Result<ReferenceCameraSetup> {
    let mut retained = retained_images_for_paths(image_paths)?;
    reference_camera_setup_for_retained(reference, &mut retained, None)
}

pub(super) fn reference_camera_setup_for_retained(
    reference: &Path,
    retained: &mut [RetainedMapperImage],
    database: Option<&DatabaseCache>,
) -> Result<ReferenceCameraSetup> {
    let cameras_with_ids = read_colmap_cameras(reference)?;
    if cameras_with_ids.is_empty() {
        bail!("reference model has no cameras");
    }
    let mut camera_ids = cameras_with_ids
        .iter()
        .map(|(camera_id, _)| *camera_id)
        .collect::<Vec<_>>();
    let mut cameras = cameras_with_ids
        .iter()
        .map(|(_, camera)| *camera)
        .collect::<Vec<_>>();
    let mut camera_has_prior_focal_length = vec![true; cameras.len()];
    let camera_index_by_id = camera_ids
        .iter()
        .enumerate()
        .map(|(idx, &camera_id)| (camera_id, idx))
        .collect::<HashMap<_, _>>();
    let poses = read_colmap_poses(reference)?;
    let pose_by_name = poses
        .iter()
        .map(|pose| (pose.name.as_str(), pose))
        .collect::<HashMap<_, _>>();
    let sparse_dir = resolve_sparse_dir(reference)?;
    let has_points =
        sparse_dir.join("points3D.bin").exists() || sparse_dir.join("points3D.txt").exists();
    let sparse_model = if has_points {
        Some(read_colmap_sparse_model(reference).with_context(|| {
            format!(
                "failed to load reference model from {}",
                reference.display()
            )
        })?)
    } else {
        None
    };
    let mut rigs = sparse_model
        .as_ref()
        .map(|model| model.reconstruction.rigs.clone())
        .unwrap_or_default();
    let mut frames = sparse_model
        .as_ref()
        .map(|model| model.reconstruction.frames.clone())
        .unwrap_or_default();
    let image_frame_by_id = sparse_model
        .as_ref()
        .map(|model| {
            model
                .reconstruction
                .image_ids
                .iter()
                .copied()
                .zip(model.reconstruction.image_frame_indices.iter().copied())
                .collect::<HashMap<_, _>>()
        })
        .unwrap_or_default();

    let reference_index_by_name = sparse_model
        .as_ref()
        .map(|model| {
            model
                .reconstruction
                .image_names
                .iter()
                .enumerate()
                .map(|(index, name)| (name.as_str(), index))
                .collect::<HashMap<_, _>>()
        })
        .unwrap_or_default();
    let retained_paths = retained
        .iter()
        .map(|image| image.path.clone())
        .collect::<Vec<_>>();

    let mut image_ids = Vec::with_capacity(retained.len());
    let mut image_camera_indices = Vec::with_capacity(retained.len());
    let mut image_frame_indices = Vec::with_capacity(retained.len());
    let mut present_in_reference = Vec::with_capacity(retained.len());
    for (idx, image) in retained.iter().enumerate() {
        if let Some(pose) = pose_by_name.get(image.name.as_str()) {
            present_in_reference.push(true);
            if let Some(cache) = database {
                let reference_frame = image_frame_by_id
                    .get(&pose.image_id)
                    .copied()
                    .flatten()
                    .and_then(|frame_index| frames.get(frame_index));
                validate_overlapping_reference_database_identity(
                    &image.name,
                    pose.image_id,
                    pose.camera_id,
                    reference_frame.map(|frame| frame.frame_id),
                    reference_frame.map(|frame| frame.rig_id),
                    cache,
                )?;
            }
            image_ids.push(pose.image_id);
            image_camera_indices.push(*camera_index_by_id.get(&pose.camera_id).with_context(
                || {
                    format!(
                        "reference image '{}' uses missing camera id {}",
                        pose.name, pose.camera_id
                    )
                },
            )?);
            image_frame_indices.push(*image_frame_by_id.get(&pose.image_id).unwrap_or(&None));
        } else if let Some(cache) = database {
            // A database row is the identity. Do not invent idx+1 or camera 0.
            present_in_reference.push(false);
            let assigned = assign_database_identity_for_unreferenced_image(
                &image.name,
                cache,
                &image_ids,
                &mut camera_ids,
                &mut cameras,
                &mut camera_has_prior_focal_length,
                &mut rigs,
                &mut frames,
            )?;
            image_ids.push(assigned.image_id);
            image_camera_indices.push(assigned.camera_index);
            image_frame_indices.push(assigned.frame_index);
        } else {
            // Reference-only inputs have no database metadata to preserve.
            present_in_reference.push(false);
            image_ids.push(idx as u32 + 1);
            image_camera_indices.push(0);
            image_frame_indices.push(None);
        }
    }
    if database.is_some() {
        let mut seen_ids = HashSet::new();
        for (index, &image_id) in image_ids.iter().enumerate() {
            if !seen_ids.insert(image_id) {
                bail!(
                    "retained image '{}' database id {image_id} collides with another image",
                    retained[index].name
                );
            }
        }
    }

    let seed_reconstruction = sparse_model.as_ref().and_then(|model| {
        seed_reconstruction_from_reference(&model.reconstruction, &retained_paths)
    });

    let setup = ReferenceCameraSetup {
        cameras,
        camera_ids,
        camera_has_prior_focal_length,
        rigs,
        frames,
        image_ids,
        image_camera_indices,
        image_frame_indices,
        seed_reconstruction,
    };
    anyhow::ensure!(
        setup.image_camera_indices.len() == retained.len()
            && setup.image_frame_indices.len() == retained.len()
            && setup.camera_ids.len() == setup.cameras.len()
            && setup.camera_has_prior_focal_length.len() == setup.cameras.len(),
        "reference camera setup did not produce one entry per retained image"
    );
    for (index, image) in retained.iter_mut().enumerate() {
        image.reference_image_index = reference_index_by_name.get(image.name.as_str()).copied();
        let camera_index = setup.image_camera_indices[index];
        anyhow::ensure!(
            camera_index < setup.camera_ids.len(),
            "retained image '{}' camera index {camera_index} is out of range",
            image.name
        );
        image.camera_index = camera_index;
        image.rig_frame_index = setup.image_frame_indices[index];
        if database.is_some() && !present_in_reference[index] {
            image.database_image_id = setup.image_ids[index];
            image.database_camera_id = setup.camera_ids[camera_index];
        }
    }
    validate_retained_image_setup(&setup, retained)?;
    Ok(setup)
}

struct AssignedDatabaseIdentity {
    image_id: u32,
    camera_index: usize,
    frame_index: Option<usize>,
}

/// Reject a same-named reference/database pair whose stable identity disagrees.
/// Images present on only one side keep that side's identity.
fn validate_overlapping_reference_database_identity(
    image_name: &str,
    reference_image_id: u32,
    reference_camera_id: u32,
    reference_frame_id: Option<u32>,
    reference_rig_id: Option<u32>,
    cache: &DatabaseCache,
) -> Result<()> {
    let mut matched = None;
    for image in cache.images.values() {
        if image.name != image_name {
            continue;
        }
        if matched.is_some() {
            bail!("retained image '{image_name}' database id is ambiguous");
        }
        matched = Some(image);
    }
    let Some(db_image) = matched else {
        return Ok(());
    };
    if db_image.image_id != reference_image_id {
        bail!(
            "retained image '{image_name}' image_id conflict: reference={reference_image_id} \
             database={}",
            db_image.image_id
        );
    }
    if db_image.camera_id != reference_camera_id {
        bail!(
            "retained image '{image_name}' camera_id conflict: reference={reference_camera_id} \
             database={}",
            db_image.camera_id
        );
    }
    if let (Some(reference_frame_id), Some(database_frame_id)) =
        (reference_frame_id, db_image.frame_id)
    {
        if reference_frame_id != database_frame_id {
            bail!(
                "retained image '{image_name}' frame_id conflict: reference={reference_frame_id} \
                 database={database_frame_id}"
            );
        }
        if let (Some(reference_rig_id), Some(db_frame)) =
            (reference_rig_id, cache.frames.get(&database_frame_id))
        {
            if reference_rig_id != db_frame.rig_id {
                bail!(
                    "retained image '{image_name}' rig_id conflict: reference={reference_rig_id} \
                     database={}",
                    db_frame.rig_id
                );
            }
        }
    }
    Ok(())
}

/// Keep the database image, camera, and frame for an image that is not in the
/// reference model. Cameras and frames are merged by stable ID. A database
/// frame index is never reused as an index into the reference frame list.
fn assign_database_identity_for_unreferenced_image(
    image_name: &str,
    cache: &DatabaseCache,
    assigned_image_ids: &[u32],
    camera_ids: &mut Vec<u32>,
    cameras: &mut Vec<CameraModel>,
    camera_has_prior_focal_length: &mut Vec<bool>,
    rigs: &mut Vec<Rig>,
    frames: &mut Vec<Frame>,
) -> Result<AssignedDatabaseIdentity> {
    let mut matched = None;
    for image in cache.images.values() {
        if image.name != image_name {
            continue;
        }
        if matched.is_some() {
            bail!("retained image '{image_name}' database id is ambiguous");
        }
        matched = Some((image.image_id, image.camera_id, image.frame_id));
    }
    let Some((image_id, camera_id, frame_id)) = matched else {
        bail!("retained image '{image_name}' is missing from the database");
    };
    if assigned_image_ids.contains(&image_id) {
        bail!("retained image '{image_name}' database id {image_id} collides with another image");
    }
    let camera_index = if let Some(index) = camera_ids.iter().position(|&id| id == camera_id) {
        index
    } else {
        let db_camera = cache.cameras.get(&camera_id).with_context(|| {
            format!("retained image '{image_name}' is missing camera_id={camera_id}")
        })?;
        let camera = CameraModel::from_colmap(
            db_camera.camera.model_id,
            db_camera.camera.width,
            db_camera.camera.height,
            &db_camera.camera.params,
        )
        .with_context(|| {
            format!("retained image '{image_name}' has unsupported camera_id={camera_id}")
        })?;
        camera_ids.push(camera_id);
        cameras.push(camera);
        camera_has_prior_focal_length.push(db_camera.has_prior_focal_length);
        cameras.len() - 1
    };
    let frame_index = match frame_id {
        Some(frame_id) => Some(merge_database_frame_by_id(
            frame_id, image_name, cache, rigs, frames,
        )?),
        None => None,
    };
    Ok(AssignedDatabaseIdentity {
        image_id,
        camera_index,
        frame_index,
    })
}

fn merge_database_frame_by_id(
    frame_id: u32,
    image_name: &str,
    cache: &DatabaseCache,
    rigs: &mut Vec<Rig>,
    frames: &mut Vec<Frame>,
) -> Result<usize> {
    let db_frame = cache.frames.get(&frame_id).with_context(|| {
        format!("retained image '{image_name}' frame id {frame_id} cannot be mapped")
    })?;
    merge_database_rig_by_id(db_frame.rig_id, image_name, cache, rigs)?;
    let incoming = database_frame_to_frame(db_frame);
    let mut matched = None;
    for (index, existing) in frames.iter().enumerate() {
        if existing.frame_id != frame_id {
            continue;
        }
        if matched.is_some() || !frame_identity_matches(existing, &incoming) {
            bail!("retained image '{image_name}' frame id {frame_id} cannot be mapped");
        }
        matched = Some(index);
    }
    if let Some(index) = matched {
        return Ok(index);
    }
    frames.push(incoming);
    Ok(frames.len() - 1)
}

fn frame_identity_matches(existing: &Frame, incoming: &Frame) -> bool {
    existing.frame_id == incoming.frame_id
        && existing.rig_id == incoming.rig_id
        && existing.data_ids == incoming.data_ids
}

fn merge_database_rig_by_id(
    rig_id: u32,
    image_name: &str,
    cache: &DatabaseCache,
    rigs: &mut Vec<Rig>,
) -> Result<()> {
    let db_rig = cache.rigs.get(&rig_id).with_context(|| {
        format!("retained image '{image_name}' frame rig id {rig_id} cannot be mapped")
    })?;
    let incoming = rig_from_colmap(db_rig);
    let mut matched = false;
    for existing in rigs.iter() {
        if existing.rig_id != rig_id {
            continue;
        }
        if matched || existing != &incoming {
            bail!("retained image '{image_name}' frame rig id {rig_id} cannot be mapped");
        }
        matched = true;
    }
    if !matched {
        rigs.push(incoming);
    }
    Ok(())
}

pub(super) fn seed_reconstruction_from_reference(
    reference: &Reconstruction,
    image_paths: &[PathBuf],
) -> Option<ReconstructionSeed> {
    let reference_image_by_name = reference
        .image_names
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.as_str(), idx))
        .collect::<HashMap<_, _>>();
    let mut reference_to_current = HashMap::<usize, usize>::new();
    let mut current_to_reference = vec![None; image_paths.len()];
    for (current_idx, path) in image_paths.iter().enumerate() {
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(&reference_idx) = reference_image_by_name.get(name) else {
            continue;
        };
        reference_to_current.insert(reference_idx, current_idx);
        current_to_reference[current_idx] = Some(reference_idx);
    }
    if reference_to_current.is_empty() {
        return None;
    }

    let poses = current_to_reference
        .iter()
        .map(|reference_idx| {
            reference_idx.and_then(|idx| reference.poses.get(idx).copied().flatten())
        })
        .collect::<Vec<_>>();
    let mut observations = image_paths
        .iter()
        .map(|_| Vec::<Option<usize>>::new())
        .collect::<Vec<_>>();
    for (current_idx, reference_idx) in current_to_reference.iter().enumerate() {
        let Some(reference_idx) = reference_idx else {
            continue;
        };
        if let Some(reference_observations) = reference.observations.get(*reference_idx) {
            observations[current_idx] = vec![None; reference_observations.len()];
        }
    }

    let mut point_ids = Vec::new();
    let mut points = Vec::new();
    let mut reference_point_to_seed = HashMap::<usize, usize>::new();
    for (reference_point_idx, reference_point) in reference.points.iter().enumerate() {
        let track = reference_point
            .track
            .iter()
            .filter_map(|obs| {
                let &image = reference_to_current.get(&obs.image)?;
                let reference_keypoints = reference.keypoints.get(obs.image)?;
                (obs.feature < reference_keypoints.len()).then_some(TrackObservation {
                    image,
                    feature: obs.feature,
                })
            })
            .collect::<Vec<_>>();
        if track.is_empty() {
            continue;
        }
        let seed_point_idx = points.len();
        reference_point_to_seed.insert(reference_point_idx, seed_point_idx);
        point_ids.push(reference.point3d_id(reference_point_idx));
        points.push(Point3D {
            xyz: reference_point.xyz,
            color: reference_point.color,
            error: reference_point.error,
            track,
        });
    }

    for (current_idx, reference_idx) in current_to_reference.iter().enumerate() {
        let Some(reference_idx) = reference_idx else {
            continue;
        };
        let Some(reference_observations) = reference.observations.get(*reference_idx) else {
            continue;
        };
        if observations[current_idx].len() < reference_observations.len() {
            observations[current_idx].resize(reference_observations.len(), None);
        }
        for (feature, reference_point_idx) in reference_observations.iter().enumerate() {
            let Some(reference_point_idx) = reference_point_idx else {
                continue;
            };
            let Some(&seed_point_idx) = reference_point_to_seed.get(reference_point_idx) else {
                continue;
            };
            observations[current_idx][feature] = Some(seed_point_idx);
        }
    }

    for (point_idx, point) in points.iter().enumerate() {
        for obs in &point.track {
            if observations[obs.image].len() <= obs.feature {
                observations[obs.image].resize(obs.feature + 1, None);
            }
            if observations[obs.image][obs.feature].is_none() {
                observations[obs.image][obs.feature] = Some(point_idx);
            }
        }
    }

    let has_registered_pose = poses.iter().any(Option::is_some);
    (has_registered_pose || !points.is_empty()).then_some(ReconstructionSeed {
        poses,
        observations,
        point_ids,
        points,
    })
}

#[cfg(test)]
pub(super) fn load_mapper_database(
    database: Option<&Path>,
    frames: &[ImageFrame],
    min_num_matches: usize,
) -> Result<Option<MapperDatabaseInput>> {
    let image_names = frames
        .iter()
        .map(|frame| frame.name.clone())
        .collect::<BTreeSet<_>>();
    load_mapper_database_for_names(database, image_names, min_num_matches)
}

pub(super) fn load_mapper_database_for_paths(
    database: Option<&Path>,
    paths: &[PathBuf],
    min_num_matches: usize,
) -> Result<Option<MapperDatabaseInput>> {
    let image_names = paths
        .iter()
        .filter_map(|path| path.file_name())
        .map(|name| name.to_string_lossy().into_owned())
        .collect::<BTreeSet<_>>();
    load_mapper_database_for_names(database, image_names, min_num_matches)
}

fn load_mapper_database_for_names(
    database: Option<&Path>,
    image_names: BTreeSet<String>,
    min_num_matches: usize,
) -> Result<Option<MapperDatabaseInput>> {
    let Some(database) = database else {
        return Ok(None);
    };
    let db = ColmapDatabase::open_read_only(database)?;
    let cache = db.load_cache(&DatabaseCacheOptions {
        min_num_matches,
        ignore_watermarks: false,
        image_names,
        load_all_images: false,
        ..DatabaseCacheOptions::default()
    })?;
    let mut keypoints_by_name = HashMap::new();
    for image in cache.images.values() {
        let keypoints = db
            .read_keypoints(image.image_id)?
            .into_iter()
            .map(|kp| kp.to_keypoint())
            .collect::<Vec<_>>();
        keypoints_by_name.insert(image.name.clone(), keypoints);
    }
    let two_view_geometries = db
        .read_two_view_geometries()?
        .into_iter()
        .collect::<HashMap<_, _>>();
    Ok(Some(MapperDatabaseInput {
        cache,
        keypoints_by_name,
        two_view_geometries,
    }))
}

#[cfg(test)]
pub(super) fn database_frames(
    paths: &[PathBuf],
    database: &MapperDatabaseInput,
) -> Result<Vec<ImageFrame>> {
    Ok(retained_database_images(paths, database)?
        .into_iter()
        .map(|image| image.frame)
        .collect())
}

pub(super) fn retained_database_images(
    paths: &[PathBuf],
    database: &MapperDatabaseInput,
) -> Result<Vec<RetainedMapperImage>> {
    let images_by_name = database
        .cache
        .images
        .values()
        .map(|image| (image.name.as_str(), image))
        .collect::<HashMap<_, _>>();
    let mut camera_ids = Vec::new();
    for &camera_id in database.cache.cameras.keys() {
        if !camera_ids.contains(&camera_id) {
            camera_ids.push(camera_id);
        }
    }
    camera_ids.sort_unstable();
    let camera_index_by_id = camera_ids
        .iter()
        .enumerate()
        .map(|(index, &camera_id)| (camera_id, index))
        .collect::<HashMap<_, _>>();
    let mut rig_frame_ids = database
        .cache
        .frames
        .values()
        .map(|frame| frame.frame_id)
        .collect::<Vec<_>>();
    rig_frame_ids.sort_unstable();
    rig_frame_ids.dedup();
    let rig_frame_index_by_id = rig_frame_ids
        .iter()
        .enumerate()
        .map(|(index, &frame_id)| (frame_id, index))
        .collect::<HashMap<_, _>>();

    // COLMAP's DatabaseCache only loads match-connected images. Skip files under
    // image_path that are absent from the cache. The retained index is the mapper
    // index; source_index remembers the unfiltered path position.
    let mut retained = Vec::new();
    for (source_index, path) in paths.iter().enumerate() {
        let name = path
            .file_name()
            .with_context(|| format!("image path has no file name: {}", path.display()))?
            .to_string_lossy()
            .into_owned();
        let Some(image) = images_by_name.get(name.as_str()) else {
            continue;
        };
        let (width, height) = database
            .cache
            .cameras
            .get(&image.camera_id)
            .map(|camera| (camera.camera.width, camera.camera.height))
            .with_context(|| {
                format!(
                    "database image '{}' is missing camera_id={}",
                    name, image.camera_id
                )
            })?;
        let camera_index = *camera_index_by_id.get(&image.camera_id).with_context(|| {
            format!(
                "database image '{}' is missing camera_id={}",
                name, image.camera_id
            )
        })?;
        let keypoints = database
            .keypoints_by_name
            .get(name.as_str())
            .cloned()
            .unwrap_or_default();
        let id = retained.len();
        retained.push(RetainedMapperImage {
            source_index,
            name: name.clone(),
            path: path.clone(),
            frame: ImageFrame {
                id,
                name,
                path: path.clone(),
                width,
                height,
                keypoints,
                descriptors: rustscan_slam::Descriptors::new(),
                sift: crate::sift::SiftFeatures::default(),
                wide_descriptors: crate::wide::WideDescriptors {
                    data: Vec::new(),
                    dim: 0,
                    count: 0,
                },
                strong_feature_indices: Vec::new(),
                colors: Vec::new(),
            },
            database_image_id: image.image_id,
            database_camera_id: image.camera_id,
            camera_index,
            rig_frame_index: image
                .frame_id
                .and_then(|frame_id| rig_frame_index_by_id.get(&frame_id).copied()),
            reference_image_index: None,
        });
    }
    anyhow::ensure!(
        !retained.is_empty(),
        "database contains no images that match files under the image path"
    );
    Ok(retained)
}

pub(super) fn merge_reference_registered_images(
    reference: &Reconstruction,
    ordered_names: &[String],
    mut retained: Vec<RetainedMapperImage>,
) -> Result<Vec<RetainedMapperImage>> {
    let mut by_name = retained
        .drain(..)
        .map(|image| (image.name.clone(), image))
        .collect::<HashMap<_, _>>();
    let mut merged = Vec::with_capacity(ordered_names.len());
    for (source_index, name) in ordered_names.iter().enumerate() {
        let mut image = if let Some(image) = by_name.remove(name) {
            image
        } else {
            retained_image_from_reference(reference, name, source_index).with_context(|| {
                format!("registered image '{name}' is missing from database frames")
            })?
        };
        let retained_index = merged.len();
        image.frame.id = retained_index;
        image.source_index = source_index;
        merged.push(image);
    }
    Ok(merged)
}

fn retained_image_from_reference(
    reference: &Reconstruction,
    name: &str,
    source_index: usize,
) -> Result<RetainedMapperImage> {
    let reference_index = reference
        .image_names
        .iter()
        .position(|image_name| image_name == name)
        .with_context(|| format!("reference model is missing registered image '{name}'"))?;
    let pose_registered = reference
        .poses
        .get(reference_index)
        .and_then(|pose| *pose)
        .is_some();
    anyhow::ensure!(
        pose_registered,
        "reference image '{name}' has no registered pose"
    );
    let camera_index = *reference
        .image_camera_indices
        .get(reference_index)
        .with_context(|| format!("reference image '{name}' has no camera index"))?;
    let camera = reference
        .cameras
        .get(camera_index)
        .copied()
        .with_context(|| format!("reference image '{name}' camera index is out of range"))?;
    let camera_id = *reference
        .camera_ids
        .get(camera_index)
        .with_context(|| format!("reference image '{name}' camera id is out of range"))?;
    let image_id = *reference
        .image_ids
        .get(reference_index)
        .with_context(|| format!("reference image '{name}' has no database id"))?;
    let path = reference
        .image_paths
        .get(reference_index)
        .cloned()
        .unwrap_or_else(|| PathBuf::from(name));
    let keypoints = reference
        .keypoints
        .get(reference_index)
        .cloned()
        .unwrap_or_default();
    Ok(RetainedMapperImage {
        source_index,
        name: name.to_owned(),
        path: path.clone(),
        frame: ImageFrame {
            id: reference_index,
            name: name.to_owned(),
            path,
            width: camera.width,
            height: camera.height,
            keypoints,
            descriptors: rustscan_slam::Descriptors::new(),
            sift: crate::sift::SiftFeatures::default(),
            wide_descriptors: crate::wide::WideDescriptors {
                data: Vec::new(),
                dim: 0,
                count: 0,
            },
            strong_feature_indices: Vec::new(),
            colors: Vec::new(),
        },
        database_image_id: image_id,
        database_camera_id: camera_id,
        camera_index,
        rig_frame_index: reference
            .image_frame_indices
            .get(reference_index)
            .copied()
            .flatten(),
        reference_image_index: Some(reference_index),
    })
}

pub(super) fn retained_image_index_by_name(
    retained: &[RetainedMapperImage],
    name: &str,
) -> Option<usize> {
    retained.iter().position(|image| image.name == name)
}

pub(super) fn retained_image_index_by_database_id(
    retained: &[RetainedMapperImage],
    database_image_id: u32,
) -> Option<usize> {
    retained
        .iter()
        .position(|image| image.database_image_id == database_image_id)
}

pub(super) fn retained_images_for_paths(paths: &[PathBuf]) -> Result<Vec<RetainedMapperImage>> {
    paths
        .iter()
        .enumerate()
        .map(|(source_index, path)| {
            let name = path
                .file_name()
                .with_context(|| format!("image path has no file name: {}", path.display()))?
                .to_string_lossy()
                .into_owned();
            Ok(RetainedMapperImage {
                source_index,
                name: name.clone(),
                path: path.clone(),
                frame: empty_retained_frame(source_index, &name, path),
                database_image_id: 0,
                database_camera_id: 0,
                camera_index: 0,
                rig_frame_index: None,
                reference_image_index: None,
            })
        })
        .collect()
}

fn empty_retained_frame(id: usize, name: &str, path: &Path) -> ImageFrame {
    ImageFrame {
        id,
        name: name.to_string(),
        path: path.to_path_buf(),
        width: 0,
        height: 0,
        keypoints: Vec::new(),
        descriptors: rustscan_slam::Descriptors::new(),
        sift: crate::sift::SiftFeatures::default(),
        wide_descriptors: crate::wide::WideDescriptors {
            data: Vec::new(),
            dim: 0,
            count: 0,
        },
        strong_feature_indices: Vec::new(),
        colors: Vec::new(),
    }
}

pub(super) fn validate_retained_image_setup(
    setup: &ReferenceCameraSetup,
    retained: &[RetainedMapperImage],
) -> Result<()> {
    let names = retained
        .iter()
        .map(|image| image.name.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    anyhow::ensure!(
        setup.image_ids.len() == retained.len()
            && setup.image_camera_indices.len() == retained.len()
            && setup.image_frame_indices.len() == retained.len(),
        "per-image camera setup does not match retained images [{names}]"
    );
    if let Some(seed) = &setup.seed_reconstruction {
        anyhow::ensure!(
            seed.poses.len() == retained.len() && seed.observations.len() == retained.len(),
            "reference seed does not match retained images [{names}]"
        );
    }
    for (index, image) in retained.iter().enumerate() {
        anyhow::ensure!(
            image.frame.id == index && image.frame.name == image.name,
            "retained image '{}' is not at mapper index {index}",
            image.name
        );
        let camera_index = setup.image_camera_indices[index];
        anyhow::ensure!(
            camera_index < setup.cameras.len() && camera_index < setup.camera_ids.len(),
            "retained image '{}' camera index {camera_index} is out of range",
            image.name
        );
        anyhow::ensure!(
            image.camera_index == camera_index
                && image.rig_frame_index == setup.image_frame_indices[index],
            "retained image '{}' identity does not match the camera setup",
            image.name
        );
    }
    Ok(())
}

pub(super) fn validate_setup_matches_frames(
    setup: &ReferenceCameraSetup,
    frames: &[ImageFrame],
) -> Result<()> {
    let names = frames
        .iter()
        .map(|frame| frame.name.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    anyhow::ensure!(
        setup.image_ids.len() == frames.len()
            && setup.image_camera_indices.len() == frames.len()
            && setup.image_frame_indices.len() == frames.len(),
        "per-image camera setup does not match retained frames [{names}]"
    );
    if let Some(seed) = &setup.seed_reconstruction {
        anyhow::ensure!(
            seed.poses.len() == frames.len() && seed.observations.len() == frames.len(),
            "reference seed does not match retained frames [{names}]"
        );
    }
    for (index, frame) in frames.iter().enumerate() {
        let camera_index = setup.image_camera_indices[index];
        anyhow::ensure!(
            camera_index < setup.cameras.len(),
            "retained frame '{}' camera index {camera_index} is out of range",
            frame.name
        );
    }
    Ok(())
}

pub(super) fn database_camera_setup(
    cache: &DatabaseCache,
    image_paths: &[PathBuf],
) -> Result<ReferenceCameraSetup> {
    if cache.cameras.is_empty() {
        bail!("database cache has no cameras");
    }
    let mut camera_ids = Vec::with_capacity(cache.cameras.len());
    let mut cameras = Vec::with_capacity(cache.cameras.len());
    let mut camera_has_prior_focal_length = Vec::with_capacity(cache.cameras.len());
    for (&camera_id, db_camera) in &cache.cameras {
        let camera = CameraModel::from_colmap(
            db_camera.camera.model_id,
            db_camera.camera.width,
            db_camera.camera.height,
            &db_camera.camera.params,
        )
        .with_context(|| format!("unsupported database camera_id={camera_id}"))?;
        camera_ids.push(camera_id);
        cameras.push(camera);
        camera_has_prior_focal_length.push(db_camera.has_prior_focal_length);
    }
    let camera_index_by_id = camera_ids
        .iter()
        .enumerate()
        .map(|(idx, &camera_id)| (camera_id, idx))
        .collect::<HashMap<_, _>>();
    let image_by_name = cache
        .images
        .values()
        .map(|image| (image.name.as_str(), image))
        .collect::<HashMap<_, _>>();
    let rigs = cache.rigs.values().map(rig_from_colmap).collect::<Vec<_>>();
    let frames = cache
        .frames
        .values()
        .map(database_frame_to_frame)
        .collect::<Vec<_>>();
    let frame_index_by_id = frames
        .iter()
        .enumerate()
        .map(|(idx, frame)| (frame.frame_id, idx))
        .collect::<HashMap<_, _>>();

    let mut image_ids = Vec::with_capacity(image_paths.len());
    let mut image_camera_indices = Vec::with_capacity(image_paths.len());
    let mut image_frame_indices = Vec::with_capacity(image_paths.len());
    for (idx, path) in image_paths.iter().enumerate() {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if let Some(image) = image_by_name.get(name) {
            image_ids.push(image.image_id);
            image_camera_indices.push(*camera_index_by_id.get(&image.camera_id).unwrap_or(&0));
            image_frame_indices.push(
                image
                    .frame_id
                    .and_then(|frame_id| frame_index_by_id.get(&frame_id).copied()),
            );
        } else {
            image_ids.push(idx as u32 + 1);
            image_camera_indices.push(0);
            image_frame_indices.push(None);
        }
    }

    Ok(ReferenceCameraSetup {
        cameras,
        camera_ids,
        camera_has_prior_focal_length,
        rigs,
        frames,
        image_ids,
        image_camera_indices,
        image_frame_indices,
        seed_reconstruction: None,
    })
}

pub(super) fn database_camera_setup_for_retained(
    cache: &DatabaseCache,
    retained: &mut [RetainedMapperImage],
) -> Result<ReferenceCameraSetup> {
    let paths = retained
        .iter()
        .map(|image| image.path.clone())
        .collect::<Vec<_>>();
    let setup = database_camera_setup(cache, &paths)?;
    anyhow::ensure!(
        setup.image_ids.len() == retained.len()
            && setup.image_camera_indices.len() == retained.len()
            && setup.image_frame_indices.len() == retained.len(),
        "database camera setup does not match retained images"
    );
    for (index, image) in retained.iter_mut().enumerate() {
        anyhow::ensure!(
            setup.image_ids[index] == image.database_image_id,
            "database image '{}' id {} does not match retained id {}",
            image.name,
            setup.image_ids[index],
            image.database_image_id
        );
        let camera_index = setup.image_camera_indices[index];
        anyhow::ensure!(
            camera_index < setup.camera_ids.len(),
            "retained image '{}' camera index {camera_index} is out of range",
            image.name
        );
        image.camera_index = camera_index;
        image.database_camera_id = setup.camera_ids[camera_index];
        image.rig_frame_index = setup.image_frame_indices[index];
    }
    validate_retained_image_setup(&setup, retained)?;
    Ok(setup)
}

pub(super) fn local_image_camera_setup(
    frames: &[ImageFrame],
    config: &MapperConfig,
) -> Result<ReferenceCameraSetup> {
    if config.single_camera {
        if let Some(first) = frames.first() {
            if let Some(mismatched) = frames
                .iter()
                .skip(1)
                .find(|frame| frame.width != first.width || frame.height != first.height)
            {
                bail!(
                    "single camera local reconstruction requires matching image dimensions; '{}' is {}x{} but '{}' is {}x{}",
                    first.name,
                    first.width,
                    first.height,
                    mismatched.name,
                    mismatched.width,
                    mismatched.height,
                );
            }

            return Ok(ReferenceCameraSetup {
                cameras: vec![local_image_camera(first, config)],
                camera_ids: vec![1],
                camera_has_prior_focal_length: vec![true],
                rigs: Vec::new(),
                frames: Vec::new(),
                image_ids: (1..=frames.len() as u32).collect(),
                image_camera_indices: vec![0; frames.len()],
                image_frame_indices: vec![None; frames.len()],
                seed_reconstruction: None,
            });
        }
    }

    let mut cameras = Vec::with_capacity(frames.len());
    let mut camera_ids = Vec::with_capacity(frames.len());
    let mut image_ids = Vec::with_capacity(frames.len());
    let mut image_camera_indices = Vec::with_capacity(frames.len());
    for (idx, frame) in frames.iter().enumerate() {
        cameras.push(local_image_camera(frame, config));
        camera_ids.push(idx as u32 + 1);
        image_ids.push(idx as u32 + 1);
        image_camera_indices.push(idx);
    }
    Ok(ReferenceCameraSetup {
        cameras,
        camera_ids,
        camera_has_prior_focal_length: vec![true; frames.len()],
        rigs: Vec::new(),
        frames: Vec::new(),
        image_ids,
        image_camera_indices,
        image_frame_indices: vec![None; frames.len()],
        seed_reconstruction: None,
    })
}

fn local_image_camera(frame: &ImageFrame, config: &MapperConfig) -> CameraModel {
    let focal = frame.width.max(frame.height) as f32 * 1.2;
    let mut camera = CameraModel::new_pinhole(
        frame.width,
        frame.height,
        focal,
        focal,
        frame.width as f32 * 0.5,
        frame.height as f32 * 0.5,
    );
    if let Some(fx) = config.fx {
        camera.set_fx(fx);
    }
    if let Some(fy) = config.fy {
        camera.set_fy(fy);
    }
    if let Some(cx) = config.cx {
        camera.set_cx(cx);
    }
    if let Some(cy) = config.cy {
        camera.set_cy(cy);
    }
    camera
}

pub(super) fn rig_from_colmap(rig: &ColmapRig) -> Rig {
    Rig {
        rig_id: rig.rig_id,
        ref_sensor_id: rig.ref_sensor_id.as_ref().map(sensor_id_from_colmap),
        sensors: rig.sensors.iter().map(rig_sensor_from_colmap).collect(),
    }
}

pub(super) fn rig_sensor_from_colmap(sensor: &ColmapRigSensor) -> RigSensor {
    RigSensor {
        sensor_id: sensor_id_from_colmap(&sensor.sensor_id),
        sensor_from_rig: sensor.sensor_from_rig.as_ref().map(rigid3_from_colmap),
    }
}

pub(super) fn database_frame_to_frame(frame: &crate::database::ColmapDatabaseFrame) -> Frame {
    Frame {
        frame_id: frame.frame_id,
        rig_id: frame.rig_id,
        rig_from_world: Rigid3 {
            qvec: [1.0, 0.0, 0.0, 0.0],
            tvec: [0.0, 0.0, 0.0],
        },
        data_ids: frame.data_ids.iter().map(data_id_from_colmap).collect(),
    }
}

pub(super) fn sensor_id_from_colmap(sensor_id: &ColmapSensorId) -> SensorId {
    SensorId {
        sensor_type: sensor_type_from_colmap(&sensor_id.sensor_type),
        sensor_id: sensor_id.sensor_id,
    }
}

pub(super) fn sensor_type_from_colmap(sensor_type: &ColmapSensorType) -> SensorType {
    match sensor_type {
        ColmapSensorType::Invalid => SensorType::Invalid,
        ColmapSensorType::Camera => SensorType::Camera,
        ColmapSensorType::Imu => SensorType::Imu,
        ColmapSensorType::Other(value) => SensorType::Other(value.clone()),
    }
}

pub(super) fn rigid3_from_colmap(rigid: &ColmapRigid3) -> Rigid3 {
    Rigid3 {
        qvec: rigid.qvec,
        tvec: rigid.tvec,
    }
}

pub(super) fn data_id_from_colmap(data_id: &ColmapDataId) -> DataId {
    DataId {
        sensor_id: sensor_id_from_colmap(&data_id.sensor_id),
        data_id: data_id.data_id,
    }
}

#[cfg(test)]
pub(super) fn apply_database_keypoints(
    frames: &mut [ImageFrame],
    keypoints_by_name: &HashMap<String, Vec<rustscan_slam::KeyPoint>>,
) {
    for frame in frames {
        let Some(keypoints) = keypoints_by_name.get(frame.name.as_str()) else {
            continue;
        };
        if !keypoints.is_empty() {
            frame.keypoints = keypoints.clone();
            frame.descriptors = rustscan_slam::Descriptors::new();
            frame.sift = crate::sift::SiftFeatures::default();
            frame.wide_descriptors = crate::wide::WideDescriptors {
                data: Vec::new(),
                dim: 0,
                count: 0,
            };
            frame.strong_feature_indices = Vec::new();
            frame.colors = sample_keypoint_colors(frame);
        }
    }
}

pub(super) fn sample_keypoint_colors(frame: &ImageFrame) -> Vec<[u8; 3]> {
    let Ok(reader) = ImageReader::open(&frame.path) else {
        return vec![[0, 0, 0]; frame.keypoints.len()];
    };
    let Ok(image) = reader.decode() else {
        return vec![[0, 0, 0]; frame.keypoints.len()];
    };
    let image = image.to_rgb8();
    let width = image.width().max(1);
    let height = image.height().max(1);
    frame
        .keypoints
        .iter()
        .map(|kp| {
            let x = kp.x().round().clamp(0.0, (width - 1) as f32) as u32;
            let y = kp.y().round().clamp(0.0, (height - 1) as f32) as u32;
            image.get_pixel(x, y).0
        })
        .collect()
}

pub(super) fn apply_color_extraction_policy(frames: &mut [ImageFrame], extract_colors: bool) {
    if !extract_colors {
        for frame in frames {
            frame.colors = vec![[0, 0, 0]; frame.keypoints.len()];
        }
    }
}

pub(super) fn fallback_camera(first_image: &Path) -> CameraModel {
    let image = ImageReader::open(first_image)
        .ok()
        .and_then(|r| r.decode().ok());
    let (width, height) = image
        .as_ref()
        .map(|i| (i.width(), i.height()))
        .unwrap_or((1536, 2048));
    let focal = width.max(height) as f32 * 1.2;
    CameraModel::new_pinhole(
        width,
        height,
        focal,
        focal,
        width as f32 * 0.5,
        height as f32 * 0.5,
    )
}
