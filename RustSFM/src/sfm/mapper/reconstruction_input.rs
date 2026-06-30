use super::*;

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

    for candidate in default_database_candidates(&config.input) {
        if candidate.exists() {
            return Ok(Some(candidate));
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

#[derive(Debug, Clone)]
pub(super) struct MapperDatabaseInput {
    pub(super) cache: DatabaseCache,
    pub(super) keypoints_by_name: HashMap<String, Vec<rustslam::KeyPoint>>,
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
    let cameras_with_ids = read_colmap_cameras(reference)?;
    if cameras_with_ids.is_empty() {
        bail!("reference model has no cameras");
    }
    let camera_ids = cameras_with_ids
        .iter()
        .map(|(camera_id, _)| *camera_id)
        .collect::<Vec<_>>();
    let cameras = cameras_with_ids
        .iter()
        .map(|(_, camera)| *camera)
        .collect::<Vec<_>>();
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
    let sparse_model = read_colmap_sparse_model(reference).ok();
    let rigs = sparse_model
        .as_ref()
        .map(|model| model.reconstruction.rigs.clone())
        .unwrap_or_default();
    let frames = sparse_model
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

    let mut image_ids = Vec::with_capacity(image_paths.len());
    let mut image_camera_indices = Vec::with_capacity(image_paths.len());
    let mut image_frame_indices = Vec::with_capacity(image_paths.len());
    for (idx, path) in image_paths.iter().enumerate() {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if let Some(pose) = pose_by_name.get(name) {
            image_ids.push(pose.image_id);
            image_camera_indices.push(*camera_index_by_id.get(&pose.camera_id).unwrap_or(&0));
            image_frame_indices.push(*image_frame_by_id.get(&pose.image_id).unwrap_or(&None));
        } else {
            image_ids.push(idx as u32 + 1);
            image_camera_indices.push(0);
            image_frame_indices.push(None);
        }
    }

    let seed_reconstruction = sparse_model
        .as_ref()
        .and_then(|model| seed_reconstruction_from_reference(&model.reconstruction, image_paths));

    Ok(ReferenceCameraSetup {
        cameras,
        camera_ids,
        camera_has_prior_focal_length: vec![true; cameras_with_ids.len()],
        rigs,
        frames,
        image_ids,
        image_camera_indices,
        image_frame_indices,
        seed_reconstruction,
    })
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

pub(super) fn load_mapper_database(
    database: Option<&Path>,
    frames: &[ImageFrame],
    min_num_matches: usize,
) -> Result<Option<MapperDatabaseInput>> {
    let Some(database) = database else {
        return Ok(None);
    };
    let image_names = frames
        .iter()
        .map(|frame| frame.name.clone())
        .collect::<BTreeSet<_>>();
    let db = ColmapDatabase::open(database)?;
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

pub(super) fn local_image_camera_setup(
    frames: &[ImageFrame],
    config: &MapperConfig,
) -> ReferenceCameraSetup {
    let mut cameras = Vec::with_capacity(frames.len());
    let mut camera_ids = Vec::with_capacity(frames.len());
    let mut image_ids = Vec::with_capacity(frames.len());
    let mut image_camera_indices = Vec::with_capacity(frames.len());
    for (idx, frame) in frames.iter().enumerate() {
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
        cameras.push(camera);
        camera_ids.push(idx as u32 + 1);
        image_ids.push(idx as u32 + 1);
        image_camera_indices.push(idx);
    }
    ReferenceCameraSetup {
        cameras,
        camera_ids,
        camera_has_prior_focal_length: vec![true; frames.len()],
        rigs: Vec::new(),
        frames: Vec::new(),
        image_ids,
        image_camera_indices,
        image_frame_indices: vec![None; frames.len()],
        seed_reconstruction: None,
    }
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

pub(super) fn apply_database_keypoints(
    frames: &mut [ImageFrame],
    keypoints_by_name: &HashMap<String, Vec<rustslam::KeyPoint>>,
) {
    for frame in frames {
        let Some(keypoints) = keypoints_by_name.get(frame.name.as_str()) else {
            continue;
        };
        if !keypoints.is_empty() {
            frame.keypoints = keypoints.clone();
            frame.descriptors = rustslam::Descriptors::new();
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
    if extract_colors {
        for frame in frames {
            if frame.colors.len() != frame.keypoints.len() {
                frame.colors = sample_keypoint_colors(frame);
            }
        }
    } else {
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
