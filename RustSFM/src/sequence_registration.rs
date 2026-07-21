use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BinaryHeap;
use std::collections::{BTreeSet, HashSet};
use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};

use crate::colmap::{
    export_colmap, export_colmap_sparse_snapshot, read_colmap_sparse_files_with_format,
    read_colmap_sparse_model, write_colmap_sparse_binary, ColmapCamera, ColmapSparseFormat,
};
use crate::database::{ColmapDatabase, ColmapDatabaseCamera, ColmapDatabaseImage};
use crate::feature_extraction::extract_selected_features_to_database_with_task;
use crate::feature_matching::{generate_matching_pairs, MatchingPairStrategy};
use crate::feature_matching_db::{
    match_explicit_image_pairs_to_database_with_task, MatchFeaturesOptions,
};
use crate::mapper::{
    register_single_target_from_database, run_reconstruction_with_task, FeatureType, MapperConfig,
};
use crate::task::{SfmTaskContext, SfmTaskEvent, SfmTaskEventKind, SfmTaskOperation, SfmTaskStage};
use crate::types::{Reconstruction, COLMAP_PINHOLE};

pub const MAX_SEQUENCE_PLAN_FRAMES: usize = 1_000_000;
pub const MAX_SEQUENCE_NEIGHBORS: usize = 1_024;
pub const MAX_TOTAL_SUPPORT_ENTRIES: usize = 32_000_000;
pub const MAX_TIMESTAMP_PLATEAU: usize = MAX_SEQUENCE_NEIGHBORS;
pub const MAX_DYNAMIC_SUPPORT_CANDIDATES: usize = 65_536;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SequenceFrame {
    pub id: u32,
    pub image_path: PathBuf,
    pub timestamp_us: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceRegistrationConfig {
    pub narrow_neighbors_each_side: usize,
    pub wide_neighbors_each_side: usize,
    pub min_inliers: usize,
    pub min_inlier_ratio: f64,
    pub max_reprojection_error: f64,
    pub use_gpu_pnp: bool,
}

impl Default for SequenceRegistrationConfig {
    fn default() -> Self {
        Self {
            narrow_neighbors_each_side: 2,
            wide_neighbors_each_side: 4,
            min_inliers: 24,
            min_inlier_ratio: 0.20,
            max_reprojection_error: 4.0,
            use_gpu_pnp: true,
        }
    }
}

impl SequenceRegistrationConfig {
    pub fn validate(&self) -> Result<(), SequenceRegistrationError> {
        if !self.min_inlier_ratio.is_finite() || !(0.0..=1.0).contains(&self.min_inlier_ratio) {
            return Err(SequenceRegistrationError::InvalidConfigMetric {
                field: "min_inlier_ratio",
            });
        }
        if !self.max_reprojection_error.is_finite() || self.max_reprojection_error < 0.0 {
            return Err(SequenceRegistrationError::InvalidConfigMetric {
                field: "max_reprojection_error",
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegistrationRound {
    Narrow,
    Wide,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameRegistrationStatus {
    Keyframe,
    Registered,
    Unresolved,
    Excluded,
}

impl FrameRegistrationStatus {
    pub fn is_registered(self) -> bool {
        matches!(self, Self::Keyframe | Self::Registered)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameRegistrationDiagnostic {
    pub frame_id: u32,
    pub status: FrameRegistrationStatus,
    pub attempts: usize,
    pub support_frame_ids: Vec<u32>,
    pub inlier_count: usize,
    pub inlier_ratio: f64,
    pub mean_reprojection_error: Option<f64>,
    pub message: Option<String>,
}

impl FrameRegistrationDiagnostic {
    pub fn new(frame_id: u32, status: FrameRegistrationStatus) -> Self {
        Self {
            frame_id,
            status,
            attempts: 0,
            support_frame_ids: Vec::new(),
            inlier_count: 0,
            inlier_ratio: 0.0,
            mean_reprojection_error: None,
            message: None,
        }
    }

    pub fn record_attempt(
        &mut self,
        status: FrameRegistrationStatus,
        support_frame_ids: Vec<u32>,
        inlier_count: usize,
        inlier_ratio: f64,
        mean_reprojection_error: Option<f64>,
        message: Option<String>,
    ) {
        self.status = status;
        self.attempts = self.attempts.saturating_add(1);
        self.support_frame_ids = support_frame_ids;
        self.inlier_count = inlier_count;
        self.inlier_ratio = inlier_ratio;
        self.mean_reprojection_error = mean_reprojection_error;
        self.message = message;
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceRegistrationResult {
    pub imported_frames: usize,
    pub registered_frames: usize,
    pub frame_ids: Vec<u32>,
    pub diagnostics: Vec<FrameRegistrationDiagnostic>,
    pub sparse_model: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyframeReconstructionResult {
    pub imported_frames: usize,
    pub keyframe_ids: Vec<u32>,
    pub registered_keyframes: usize,
    pub database: PathBuf,
    pub sparse_model: PathBuf,
}

pub fn run_keyframe_reconstruction(
    frames: &[SequenceFrame],
    keyframe_ids: &[u32],
    mapper_config: &MapperConfig,
    output: &Path,
    task: &mut SfmTaskContext<'_>,
) -> anyhow::Result<KeyframeReconstructionResult> {
    let keyframe_indices = validate_runner_inputs(frames, keyframe_ids)?;
    if mapper_config.feature_type != FeatureType::Sift {
        anyhow::bail!("sequence registration requires SIFT features");
    }
    if matches!(
        mapper_config.matching_pair_strategy,
        MatchingPairStrategy::VocabTree { .. }
    ) {
        anyhow::bail!("keyframe-only vocabulary-tree matching is unsupported");
    }
    task.checkpoint().map_err(anyhow::Error::new)?;

    let cache = output.join("Cache");
    let keyframe_input = cache.join("keyframes");
    let database = cache.join("database.db");
    std::fs::create_dir_all(&keyframe_input)?;
    for &index in &keyframe_indices {
        link_or_copy_stable_image(&frames[index].image_path, &keyframe_input)?;
    }
    import_database_images(frames, &keyframe_indices, mapper_config, &database)?;
    let keyframe_database_ids =
        database_image_ids_for_indices(frames, &keyframe_indices, &database)?;

    let mut sift_extraction = mapper_config.sift_extraction.clone();
    sift_extraction.max_num_features = mapper_config.max_features;
    let mut missing_feature_image_ids = Vec::new();
    for &image_id in &keyframe_database_ids {
        if !database_features_exist(&database, image_id)? {
            missing_feature_image_ids.push(image_id);
        }
    }
    if !missing_feature_image_ids.is_empty() {
        extract_selected_features_to_database_with_task(
            &database,
            &keyframe_input,
            &sift_extraction,
            &missing_feature_image_ids,
            task,
        )?;
    }

    let keyframe_pairs = generate_matching_pairs(
        keyframe_database_ids.len(),
        mapper_config.matching_pair_strategy,
    )
    .into_iter()
    .map(|(left, right)| (keyframe_database_ids[left], keyframe_database_ids[right]))
    .collect::<Vec<_>>();
    let mut match_options = sequence_match_options(mapper_config);
    match_options.task_pair_batch_size = 1;
    match_explicit_image_pairs_to_database_with_task(
        &database,
        &keyframe_pairs,
        &match_options,
        task,
    )?;

    let mut reconstruction_config = mapper_config.clone();
    reconstruction_config.input = keyframe_input;
    reconstruction_config.output = output.to_path_buf();
    reconstruction_config.reference = None;
    reconstruction_config.database = Some(database.clone());
    reconstruction_config.local_matching = false;
    reconstruction_config.write_database = false;
    let summary = run_reconstruction_with_task(&reconstruction_config, task)?;
    let sparse_model = output.join("sparse").join("0");
    let sparse_files =
        read_colmap_sparse_files_with_format(&sparse_model, ColmapSparseFormat::Text)?;
    write_colmap_sparse_binary(&sparse_model, &sparse_files)?;
    let model = read_colmap_sparse_model(&sparse_model)?;
    validate_sparse_reconstruction(&model.reconstruction)?;
    let registered_keyframes = model.reconstruction.poses.iter().flatten().count();
    if summary.registered_images != registered_keyframes {
        anyhow::bail!(
            "keyframe reconstruction summary reports {} registered images but sparse model contains {registered_keyframes}",
            summary.registered_images
        );
    }

    Ok(KeyframeReconstructionResult {
        imported_frames: frames.len(),
        keyframe_ids: keyframe_indices
            .iter()
            .map(|&index| frames[index].id)
            .collect(),
        registered_keyframes,
        database,
        sparse_model,
    })
}

fn link_or_copy_stable_image(source: &Path, destination_dir: &Path) -> anyhow::Result<PathBuf> {
    let name = source
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("input image {} has no file name", source.display()))?;
    let destination = destination_dir.join(name);
    if destination.exists() {
        if files_have_same_contents(source, &destination)? {
            return Ok(destination);
        }
        anyhow::bail!(
            "existing stable image {} does not match source {}",
            destination.display(),
            source.display()
        );
    }
    if std::fs::hard_link(source, &destination).is_err() {
        std::fs::copy(source, &destination)?;
    }
    Ok(destination)
}

fn files_have_same_contents(left: &Path, right: &Path) -> anyhow::Result<bool> {
    use std::io::{BufReader, Read};

    if std::fs::metadata(left)?.len() != std::fs::metadata(right)?.len() {
        return Ok(false);
    }
    let mut left = BufReader::new(std::fs::File::open(left)?);
    let mut right = BufReader::new(std::fs::File::open(right)?);
    let mut left_buffer = [0u8; 64 * 1024];
    let mut right_buffer = [0u8; 64 * 1024];
    loop {
        let left_count = left.read(&mut left_buffer)?;
        let right_count = right.read(&mut right_buffer)?;
        if left_count != right_count || left_buffer[..left_count] != right_buffer[..right_count] {
            return Ok(false);
        }
        if left_count == 0 {
            return Ok(true);
        }
    }
}

fn import_database_images(
    frames: &[SequenceFrame],
    indices: &[usize],
    mapper_config: &MapperConfig,
    database: &Path,
) -> anyhow::Result<()> {
    let db = ColmapDatabase::open(database)?;
    for &index in indices {
        let frame = &frames[index];
        let image_id = u32::try_from(index + 1)?;
        let expected_name = frame
            .image_path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("stable file name was validated");
        let expected_camera = expected_database_camera(frame, mapper_config, image_id)?;
        if let Some(existing_image) = db.read_image(image_id)? {
            if existing_image.name != expected_name || existing_image.frame_id.is_some() {
                anyhow::bail!("database image_id={image_id} metadata does not match frame");
            }
            let existing_camera = db.read_camera(existing_image.camera_id)?.ok_or_else(|| {
                anyhow::anyhow!(
                    "database image_id={image_id} references missing camera_id={}",
                    existing_image.camera_id
                )
            })?;
            if !database_camera_metadata_matches(&existing_camera, &expected_camera) {
                anyhow::bail!("database image_id={image_id} camera metadata does not match frame");
            }
            continue;
        }

        if let Some(existing_camera) = db.read_camera(image_id)? {
            if !database_camera_metadata_matches(&existing_camera, &expected_camera) {
                anyhow::bail!("database camera_id={image_id} metadata does not match frame");
            }
        } else {
            db.write_camera(&expected_camera, true)?;
        }
        db.write_image(
            &ColmapDatabaseImage {
                image_id,
                name: expected_name.to_owned(),
                camera_id: image_id,
                frame_id: None,
            },
            true,
        )?;
    }
    Ok(())
}

fn expected_database_camera(
    frame: &SequenceFrame,
    mapper_config: &MapperConfig,
    camera_id: u32,
) -> anyhow::Result<ColmapDatabaseCamera> {
    let (width, height) = image::image_dimensions(&frame.image_path)?;
    let focal = width.max(height) as f64 * 1.2;
    let fx = mapper_config.fx.map(f64::from).unwrap_or(focal);
    let fy = mapper_config.fy.map(f64::from).unwrap_or(focal);
    let cx = mapper_config
        .cx
        .map(f64::from)
        .unwrap_or(width as f64 * 0.5);
    let cy = mapper_config
        .cy
        .map(f64::from)
        .unwrap_or(height as f64 * 0.5);
    Ok(ColmapDatabaseCamera {
        camera: ColmapCamera {
            camera_id,
            model_id: COLMAP_PINHOLE,
            width,
            height,
            params: vec![fx, fy, cx, cy],
        },
        has_prior_focal_length: mapper_config.fx.is_some() && mapper_config.fy.is_some(),
    })
}

fn database_camera_metadata_matches(
    actual: &ColmapDatabaseCamera,
    expected: &ColmapDatabaseCamera,
) -> bool {
    actual.camera.model_id == expected.camera.model_id
        && actual.camera.width == expected.camera.width
        && actual.camera.height == expected.camera.height
        && actual.camera.params == expected.camera.params
        && actual.has_prior_focal_length == expected.has_prior_focal_length
}

fn database_image_ids_for_indices(
    frames: &[SequenceFrame],
    indices: &[usize],
    database: &Path,
) -> anyhow::Result<Vec<u32>> {
    let database = ColmapDatabase::open_read_only(database)?;
    indices
        .iter()
        .map(|&index| {
            let name = stable_image_name(&frames[index])?;
            database
                .read_image_with_name(name)?
                .map(|image| image.image_id)
                .ok_or_else(|| anyhow::anyhow!("database is missing image '{name}'"))
        })
        .collect()
}

fn validate_keyframe_artifacts(
    frames: &[SequenceFrame],
    keyframe_indices: &[usize],
    keyframe_result: &KeyframeReconstructionResult,
    mapper_config: &MapperConfig,
    output: &Path,
) -> anyhow::Result<Reconstruction> {
    let expected_sparse_model = output.join("sparse").join("0");
    if keyframe_result.sparse_model != expected_sparse_model {
        anyhow::bail!(
            "keyframe sparse model must remain at {}",
            expected_sparse_model.display()
        );
    }
    if !keyframe_result.sparse_model.is_dir() {
        anyhow::bail!(
            "missing keyframe sparse model {}",
            keyframe_result.sparse_model.display()
        );
    }

    let database = ColmapDatabase::open_read_only(&keyframe_result.database)?;
    let mut expected_sparse_images = BTreeSet::<(String, u32)>::new();
    for &index in keyframe_indices {
        let frame = &frames[index];
        let name = stable_image_name(frame)?;
        let image = database
            .read_image_with_name(name)?
            .ok_or_else(|| anyhow::anyhow!("keyframe database is missing image '{name}'"))?;
        if image.frame_id.is_some() {
            anyhow::bail!("keyframe database image '{name}' has unexpected rig frame metadata");
        }
        let camera = database.read_camera(image.camera_id)?.ok_or_else(|| {
            anyhow::anyhow!(
                "keyframe database image '{name}' references missing camera_id={}",
                image.camera_id
            )
        })?;
        let expected_camera = expected_database_camera(frame, mapper_config, image.camera_id)?;
        if !database_camera_metadata_matches(&camera, &expected_camera) {
            anyhow::bail!(
                "keyframe database camera metadata does not match frame {}",
                frame.id
            );
        }
        expected_sparse_images.insert((name.to_owned(), image.image_id));
    }
    drop(database);

    let model = read_colmap_sparse_model(&keyframe_result.sparse_model)?;
    validate_sparse_reconstruction(&model.reconstruction)?;
    if model.reconstruction.image_names.len() != model.reconstruction.image_ids.len() {
        anyhow::bail!("keyframe sparse image name/ID metadata length mismatch");
    }
    let actual_sparse_images = model
        .reconstruction
        .image_names
        .iter()
        .cloned()
        .zip(model.reconstruction.image_ids.iter().copied())
        .collect::<BTreeSet<_>>();
    if actual_sparse_images.len() != model.reconstruction.image_names.len()
        || !actual_sparse_images.is_subset(&expected_sparse_images)
    {
        anyhow::bail!("keyframe sparse image names/IDs do not match database");
    }
    let registered_keyframes = model.reconstruction.poses.iter().flatten().count();
    if keyframe_result.registered_keyframes != registered_keyframes {
        anyhow::bail!(
            "keyframe artifacts do not match registered keyframe count: result={} sparse={registered_keyframes}",
            keyframe_result.registered_keyframes
        );
    }
    Ok(model.reconstruction)
}

fn validate_sparse_reconstruction(reconstruction: &Reconstruction) -> anyhow::Result<()> {
    if reconstruction.points.is_empty() {
        anyhow::bail!("keyframe reconstruction contains no sparse points");
    }
    if reconstruction.poses.iter().flatten().count() < 2 {
        anyhow::bail!("keyframe reconstruction contains fewer than two registered images");
    }
    if reconstruction
        .points
        .iter()
        .any(|point| point.xyz.iter().any(|value| !value.is_finite()))
    {
        anyhow::bail!("keyframe reconstruction contains non-finite points");
    }
    if reconstruction.poses.iter().flatten().any(|pose| {
        pose.translation().iter().any(|value| !value.is_finite())
            || pose.quaternion().iter().any(|value| !value.is_finite())
    }) {
        anyhow::bail!("keyframe reconstruction contains non-finite poses");
    }
    Ok(())
}

fn validate_runner_inputs(
    frames: &[SequenceFrame],
    keyframe_ids: &[u32],
) -> anyhow::Result<Vec<usize>> {
    if frames.is_empty() {
        anyhow::bail!("sequence must contain at least one frame");
    }
    if keyframe_ids.is_empty() {
        anyhow::bail!("sequence registration requires at least one keyframe");
    }

    let mut frame_index_by_id = std::collections::BTreeMap::new();
    let mut image_names = BTreeSet::new();
    for (index, frame) in frames.iter().enumerate() {
        if frame_index_by_id.insert(frame.id, index).is_some() {
            anyhow::bail!("duplicate frame id {}", frame.id);
        }
        let image_name = frame
            .image_path
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "frame {} image path has no stable UTF-8 file name",
                    frame.id
                )
            })?;
        if !image_names.insert(image_name.to_owned()) {
            anyhow::bail!("duplicate stable image name {image_name}");
        }
    }

    let mut seen_keyframes = BTreeSet::new();
    let mut keyframe_indices = Vec::with_capacity(keyframe_ids.len());
    for &keyframe_id in keyframe_ids {
        if !seen_keyframes.insert(keyframe_id) {
            anyhow::bail!("duplicate keyframe id {keyframe_id}");
        }
        let index = frame_index_by_id
            .get(&keyframe_id)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("unknown keyframe id {keyframe_id}"))?;
        keyframe_indices.push(index);
    }
    keyframe_indices.sort_unstable();
    Ok(keyframe_indices)
}

pub fn register_remaining_sequence_frames(
    frames: &[SequenceFrame],
    keyframe_ids: &[u32],
    keyframe_result: &KeyframeReconstructionResult,
    mapper_config: &MapperConfig,
    config: &SequenceRegistrationConfig,
    output: &Path,
    task: &mut SfmTaskContext<'_>,
) -> anyhow::Result<SequenceRegistrationResult> {
    let keyframe_indices = validate_runner_inputs(frames, keyframe_ids)?;
    config.validate().map_err(anyhow::Error::new)?;
    let normalized_keyframe_ids = keyframe_indices
        .iter()
        .map(|&index| frames[index].id)
        .collect::<Vec<_>>();
    if keyframe_result.imported_frames != frames.len()
        || keyframe_result.keyframe_ids != normalized_keyframe_ids
    {
        anyhow::bail!("keyframe artifacts do not match the requested sequence and keyframe IDs");
    }
    let expected_database = output.join("Cache").join("database.db");
    if keyframe_result.database != expected_database {
        anyhow::bail!(
            "keyframe database must remain at {}",
            expected_database.display()
        );
    }
    if !keyframe_result.database.is_file() {
        anyhow::bail!(
            "missing keyframe database {}",
            keyframe_result.database.display()
        );
    }
    let initial_reconstruction = validate_keyframe_artifacts(
        frames,
        &keyframe_indices,
        keyframe_result,
        mapper_config,
        output,
    )?;
    let plan = SequenceRegistrationPlan::build_from_frames(
        frames,
        &keyframe_indices,
        config.narrow_neighbors_each_side,
        config.wide_neighbors_each_side,
    )?;
    task.checkpoint().map_err(anyhow::Error::new)?;

    let sequence_input = output.join("Cache").join("sequence");
    std::fs::create_dir_all(&sequence_input)?;
    for frame in frames {
        link_or_copy_stable_image(&frame.image_path, &sequence_input)?;
    }
    let mut diagnostics = frames
        .iter()
        .map(|frame| {
            FrameRegistrationDiagnostic::new(frame.id, FrameRegistrationStatus::Unresolved)
        })
        .collect::<Vec<_>>();
    let initial_registered_names = registered_image_names(&initial_reconstruction);
    for &keyframe in &keyframe_indices {
        let name = stable_image_name(&frames[keyframe])?;
        if initial_registered_names.contains(name) {
            diagnostics[keyframe].status = FrameRegistrationStatus::Keyframe;
        } else {
            diagnostics[keyframe].message = Some("keyframe was not registered".to_owned());
        }
    }
    task.emit(SfmTaskEvent {
        sequence: 0,
        elapsed_ms: 0,
        stage: SfmTaskStage::FullFrameRegistration,
        operation: SfmTaskOperation::Begin,
        kind: SfmTaskEventKind::Started,
        completed: Some(0),
        total: Some(plan.pending_frames().len()),
        registered_images: Some(initial_registered_names.len()),
        sparse_points: Some(initial_reconstruction.points.len()),
        image_id: None,
        pair: None,
        message: None,
        issue: None,
    });

    let mut current_reconstruction = initial_reconstruction;
    let mut current_reference = keyframe_result.sparse_model.clone();
    let mut available_from_prior_rounds = Vec::<usize>::new();
    let mut extracted_targets = HashSet::<usize>::new();
    let match_options = sequence_match_options(mapper_config);
    let mut target_mapper_config = mapper_config.clone();
    target_mapper_config.abs_pose_min_num_inliers = config.min_inliers.max(4);
    target_mapper_config.abs_pose_min_inlier_ratio = config.min_inlier_ratio as f32;
    target_mapper_config.pnp_threshold_px = config.max_reprojection_error as f32;
    target_mapper_config.use_gpu_pnp = config.use_gpu_pnp;
    target_mapper_config.local_ba = false;
    target_mapper_config.global_ba = false;
    target_mapper_config.fix_existing_frames = true;

    for round in [RegistrationRound::Narrow, RegistrationRound::Wide] {
        let mut accepted_this_round = Vec::<usize>::new();
        for &target in plan.pending_frames() {
            if diagnostics[target].status == FrameRegistrationStatus::Registered {
                continue;
            }
            if extracted_targets.insert(target) {
                import_database_images(
                    frames,
                    &[target],
                    mapper_config,
                    &keyframe_result.database,
                )?;
                let target_database_id = u32::try_from(target + 1)?;
                if !database_features_exist(&keyframe_result.database, target_database_id)? {
                    let mut sift_extraction = mapper_config.sift_extraction.clone();
                    sift_extraction.max_num_features = mapper_config.max_features;
                    extract_selected_features_to_database_with_task(
                        &keyframe_result.database,
                        &sequence_input,
                        &sift_extraction,
                        &[target_database_id],
                        task,
                    )?;
                }
            }
            let mut support =
                plan.attempts_for_with_sorted_support(target, round, &available_from_prior_rounds)?;
            let registered_names = registered_image_names(&current_reconstruction);
            support.retain(|&index| {
                stable_image_name(&frames[index])
                    .map(|name| registered_names.contains(name))
                    .unwrap_or(false)
            });
            let support_frame_ids = support
                .iter()
                .map(|&index| frames[index].id)
                .collect::<Vec<_>>();
            let attempt_seed =
                sequence_attempt_random_seed(mapper_config.random_seed, frames[target].id, round);
            task.checkpoint().map_err(anyhow::Error::new)?;
            task.emit(SfmTaskEvent {
                sequence: 0,
                elapsed_ms: 0,
                stage: SfmTaskStage::FullFrameRegistration,
                operation: SfmTaskOperation::RegisterFrameAttempt,
                kind: SfmTaskEventKind::Progress,
                completed: Some(diagnostics.iter().map(|item| item.attempts).sum::<usize>() + 1),
                total: None,
                registered_images: Some(registered_names.len()),
                sparse_points: Some(current_reconstruction.points.len()),
                image_id: Some(frames[target].id),
                pair: None,
                message: Some(format!(
                    "round={round:?} seed={attempt_seed} support={support_frame_ids:?}"
                )),
                issue: None,
            });
            if support.is_empty() {
                diagnostics[target].record_attempt(
                    FrameRegistrationStatus::Unresolved,
                    support_frame_ids,
                    0,
                    0.0,
                    None,
                    Some("no registered temporal support".to_owned()),
                );
                continue;
            }

            let target_database_id = u32::try_from(target + 1)?;
            let pairs = support
                .iter()
                .map(|&index| Ok((target_database_id, u32::try_from(index + 1)?)))
                .collect::<anyhow::Result<Vec<_>>>()?;
            let mut attempt_match_options = match_options.clone();
            attempt_match_options.random_seed = attempt_seed;
            match_explicit_image_pairs_to_database_with_task(
                &keyframe_result.database,
                &pairs,
                &attempt_match_options,
                task,
            )?;
            let target_name = stable_image_name(&frames[target])?;
            let support_names = support
                .iter()
                .map(|&index| stable_image_name(&frames[index]).map(str::to_owned))
                .collect::<anyhow::Result<Vec<_>>>()?;
            let mut attempt_mapper_config = target_mapper_config.clone();
            attempt_mapper_config.random_seed = attempt_seed;
            let candidate = register_single_target_from_database(
                &sequence_input,
                &keyframe_result.database,
                &current_reference,
                target_name,
                &support_names,
                &attempt_mapper_config,
            )?;
            let (inlier_count, inlier_ratio, mean_error) = candidate
                .as_ref()
                .map(|candidate| {
                    (
                        candidate.inlier_count,
                        candidate.inlier_ratio,
                        Some(candidate.mean_reprojection_error),
                    )
                })
                .unwrap_or((0, 0.0, None));
            diagnostics[target].record_attempt(
                FrameRegistrationStatus::Unresolved,
                support_frame_ids,
                inlier_count,
                inlier_ratio,
                mean_error,
                candidate
                    .is_none()
                    .then(|| "PnP did not produce a finite pose".to_owned()),
            );
            if !accepts_registration(&diagnostics[target], config) {
                continue;
            }
            let candidate = candidate.expect("accepted diagnostic requires a candidate model");
            validate_sparse_reconstruction(&candidate.reconstruction)
                .map_err(|error| error.context("invalid accepted registration model"))?;
            let accepted_root = output
                .join("Cache")
                .join("accepted")
                .join(match round {
                    RegistrationRound::Narrow => "narrow",
                    RegistrationRound::Wide => "wide",
                })
                .join(frames[target].id.to_string());
            if accepted_root.exists() {
                std::fs::remove_dir_all(&accepted_root)?;
            }
            export_colmap(&accepted_root, &candidate.reconstruction, false)?;
            let accepted_sparse = accepted_root.join("sparse").join("0");
            let accepted_model = read_colmap_sparse_model(&accepted_sparse)?;
            validate_sparse_reconstruction(&accepted_model.reconstruction)
                .map_err(|error| error.context("invalid exported registration model"))?;
            let accepted_files =
                read_colmap_sparse_files_with_format(&accepted_sparse, ColmapSparseFormat::Text)?;
            write_colmap_sparse_binary(&accepted_sparse, &accepted_files)?;
            task.checkpoint().map_err(anyhow::Error::new)?;

            current_reconstruction = candidate.reconstruction;
            current_reference = accepted_sparse;
            diagnostics[target].status = FrameRegistrationStatus::Registered;
            diagnostics[target].message = Some(format!("registered in {round:?} round"));
            match accepted_this_round.binary_search(&target) {
                Ok(_) => {}
                Err(position) => accepted_this_round.insert(position, target),
            }
        }
        available_from_prior_rounds = merge_sorted_support(
            &available_from_prior_rounds,
            &accepted_this_round,
            usize::MAX,
        );
    }

    validate_sparse_reconstruction(&current_reconstruction)
        .map_err(|error| error.context("invalid merged sequence model"))?;
    let sparse_model = publish_sparse_model_atomic(output, &current_reconstruction, task)?;
    let merged_model = read_colmap_sparse_model(&sparse_model)?;
    validate_sparse_reconstruction(&merged_model.reconstruction)
        .map_err(|error| error.context("invalid exported merged sequence model"))?;
    let registered_frames = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.status.is_registered())
        .count();
    let result = SequenceRegistrationResult {
        imported_frames: frames.len(),
        registered_frames,
        frame_ids: frames.iter().map(|frame| frame.id).collect(),
        diagnostics,
        sparse_model,
    };
    write_registration_result_atomic(output, &result)?;
    task.emit(SfmTaskEvent {
        sequence: 0,
        elapsed_ms: 0,
        stage: SfmTaskStage::FullFrameRegistration,
        operation: SfmTaskOperation::Complete,
        kind: SfmTaskEventKind::Completed,
        completed: Some(registered_frames),
        total: Some(frames.len()),
        registered_images: Some(registered_frames),
        sparse_points: Some(merged_model.reconstruction.points.len()),
        image_id: None,
        pair: None,
        message: None,
        issue: None,
    });
    Ok(result)
}

fn publish_sparse_model_atomic(
    output: &Path,
    reconstruction: &Reconstruction,
    task: &mut SfmTaskContext<'_>,
) -> anyhow::Result<PathBuf> {
    let sparse_root = output.join("sparse");
    let destination = sparse_root.join("0");
    let temporary = sparse_root.join("0.tmp");
    let backup = sparse_root.join("0.backup");
    std::fs::create_dir_all(&sparse_root)?;
    recover_sparse_publish_paths(&destination, &temporary, &backup)?;

    let stage_result = (|| -> anyhow::Result<()> {
        export_colmap_sparse_snapshot(&temporary, reconstruction)?;
        let text_files =
            read_colmap_sparse_files_with_format(&temporary, ColmapSparseFormat::Text)?;
        write_colmap_sparse_binary(&temporary, &text_files)?;
        sync_directory_files(&temporary)?;
        let staged = read_colmap_sparse_model(&temporary)?;
        validate_sparse_reconstruction(&staged.reconstruction)
            .map_err(|error| error.context("invalid staged merged sequence model"))?;
        Ok(())
    })();
    if let Err(error) = stage_result {
        let _ = remove_file_or_directory(&temporary);
        return Err(error);
    }

    task.emit(SfmTaskEvent {
        sequence: 0,
        elapsed_ms: 0,
        stage: SfmTaskStage::Export,
        operation: SfmTaskOperation::ValidateArtifacts,
        kind: SfmTaskEventKind::Progress,
        completed: None,
        total: None,
        registered_images: Some(reconstruction.poses.iter().flatten().count()),
        sparse_points: Some(reconstruction.points.len()),
        image_id: None,
        pair: None,
        message: Some("validated staged sparse model".to_owned()),
        issue: None,
    });
    if let Err(stop) = task.checkpoint() {
        let _ = remove_file_or_directory(&temporary);
        return Err(anyhow::Error::new(stop));
    }

    replace_sparse_directory(&destination, &temporary, &backup)?;
    task.emit(SfmTaskEvent {
        sequence: 0,
        elapsed_ms: 0,
        stage: SfmTaskStage::Export,
        operation: SfmTaskOperation::WriteArtifacts,
        kind: SfmTaskEventKind::Progress,
        completed: None,
        total: None,
        registered_images: Some(reconstruction.poses.iter().flatten().count()),
        sparse_points: Some(reconstruction.points.len()),
        image_id: None,
        pair: None,
        message: Some("published sparse model".to_owned()),
        issue: None,
    });
    task.checkpoint().map_err(anyhow::Error::new)?;
    Ok(destination)
}

fn recover_sparse_publish_paths(
    destination: &Path,
    temporary: &Path,
    backup: &Path,
) -> anyhow::Result<()> {
    if backup.exists() {
        if destination.exists() {
            remove_file_or_directory(backup)?;
        } else {
            std::fs::rename(backup, destination)?;
        }
    }
    if temporary.exists() {
        remove_file_or_directory(temporary)?;
    }
    Ok(())
}

fn replace_sparse_directory(
    destination: &Path,
    temporary: &Path,
    backup: &Path,
) -> anyhow::Result<()> {
    let had_destination = destination.exists();
    if had_destination {
        std::fs::rename(destination, backup)?;
    }
    if let Err(publish_error) = std::fs::rename(temporary, destination) {
        let restore_result = if had_destination {
            std::fs::rename(backup, destination)
        } else {
            Ok(())
        };
        let _ = remove_file_or_directory(temporary);
        return match restore_result {
            Ok(()) => Err(publish_error.into()),
            Err(restore_error) => anyhow::bail!(
                "failed to publish sparse model ({publish_error}) and restore backup ({restore_error})"
            ),
        };
    }
    if had_destination {
        if let Err(cleanup_error) = remove_file_or_directory(backup) {
            let moved_new_model = std::fs::rename(destination, temporary);
            let restored_old_model = std::fs::rename(backup, destination);
            if moved_new_model.is_ok() && restored_old_model.is_ok() {
                let _ = remove_file_or_directory(temporary);
                return Err(cleanup_error.into());
            }
            anyhow::bail!(
                "published sparse model but failed to remove backup ({cleanup_error}); rollback new={moved_new_model:?} old={restored_old_model:?}"
            );
        }
    }
    std::fs::File::open(
        destination
            .parent()
            .expect("sparse destination has a parent"),
    )?
    .sync_all()?;
    Ok(())
}

fn sync_directory_files(root: &Path) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_file() {
            std::fs::File::open(path)?.sync_all()?;
        }
    }
    std::fs::File::open(root)?.sync_all()?;
    Ok(())
}

fn remove_file_or_directory(path: &Path) -> std::io::Result<()> {
    if path.is_dir() {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    }
}

pub fn run_sequence_registration(
    frames: &[SequenceFrame],
    keyframe_ids: &[u32],
    mapper_config: &MapperConfig,
    config: &SequenceRegistrationConfig,
    output: &Path,
    task: &mut SfmTaskContext<'_>,
) -> anyhow::Result<SequenceRegistrationResult> {
    let keyframes = run_keyframe_reconstruction(frames, keyframe_ids, mapper_config, output, task)?;
    register_remaining_sequence_frames(
        frames,
        keyframe_ids,
        &keyframes,
        mapper_config,
        config,
        output,
        task,
    )
}

fn stable_image_name(frame: &SequenceFrame) -> anyhow::Result<&str> {
    frame
        .image_path
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| anyhow::anyhow!("frame {} has no stable UTF-8 image name", frame.id))
}

fn registered_image_names(reconstruction: &Reconstruction) -> HashSet<&str> {
    reconstruction
        .image_names
        .iter()
        .zip(&reconstruction.poses)
        .filter_map(|(name, pose)| pose.is_some().then_some(name.as_str()))
        .collect()
}

fn sequence_match_options(mapper_config: &MapperConfig) -> MatchFeaturesOptions {
    let mut sift_matching = mapper_config.sift_matching.clone();
    sift_matching.max_ratio = mapper_config.match_ratio as f32;
    MatchFeaturesOptions {
        pair_strategy: mapper_config.matching_pair_strategy,
        sift_matching,
        essential_threshold_px: mapper_config.essential_threshold_px,
        essential_iterations: mapper_config.essential_iterations,
        min_inliers: mapper_config.min_inliers,
        min_triangulated: mapper_config.min_triangulated,
        min_num_matches: mapper_config.min_matches,
        random_seed: mapper_config.random_seed,
        clear_existing: false,
        use_existing_matches: false,
        task_pair_batch_size: 1,
        ..MatchFeaturesOptions::default()
    }
}

fn sequence_attempt_random_seed(
    configured_seed: i32,
    frame_id: u32,
    round: RegistrationRound,
) -> i32 {
    if configured_seed >= 0 {
        return configured_seed;
    }
    let round_tag = match round {
        RegistrationRound::Narrow => 0x9e37_79b9_7f4a_7c15,
        RegistrationRound::Wide => 0xbf58_476d_1ce4_e5b9,
    };
    let mut value = u64::from(frame_id) ^ round_tag;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    (value & i32::MAX as u64) as i32
}

fn database_features_exist(database: &Path, image_id: u32) -> anyhow::Result<bool> {
    let database = ColmapDatabase::open_read_only(database)?;
    Ok(database.exists_keypoints(image_id)? && database.exists_descriptors(image_id)?)
}

pub fn require_complete_pose_coverage(result: &SequenceRegistrationResult) -> anyhow::Result<()> {
    if result.has_complete_coverage() {
        Ok(())
    } else {
        let failed = result
            .imported_frames
            .saturating_sub(result.registered_frames);
        anyhow::bail!("{failed} frames could not be registered")
    }
}

fn accepts_registration(
    diagnostic: &FrameRegistrationDiagnostic,
    config: &SequenceRegistrationConfig,
) -> bool {
    diagnostic.inlier_count >= config.min_inliers
        && diagnostic.inlier_ratio >= config.min_inlier_ratio
        && diagnostic
            .mean_reprojection_error
            .is_some_and(|error| error.is_finite() && error <= config.max_reprojection_error)
}

fn write_registration_result_atomic(
    output: &Path,
    result: &SequenceRegistrationResult,
) -> anyhow::Result<()> {
    use std::io::Write;

    std::fs::create_dir_all(output)?;
    let destination = output.join("registration.json");
    let temporary = output.join("registration.json.tmp");
    let write_result = (|| -> anyhow::Result<()> {
        let mut file = std::fs::File::create(&temporary)?;
        serde_json::to_writer_pretty(&mut file, result)?;
        file.write_all(b"\n")?;
        file.flush()?;
        file.sync_all()?;
        std::fs::rename(&temporary, &destination)?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    write_result
}

impl SequenceRegistrationResult {
    pub fn has_complete_coverage(&self) -> bool {
        self.validate_complete_coverage().is_ok()
    }

    pub fn validate_complete_coverage(&self) -> Result<(), SequenceRegistrationError> {
        if self.imported_frames == 0 {
            return Err(SequenceRegistrationError::EmptySequence);
        }
        if self.diagnostics.len() != self.imported_frames {
            return Err(SequenceRegistrationError::DiagnosticCountMismatch {
                imported_frames: self.imported_frames,
                diagnostic_count: self.diagnostics.len(),
            });
        }
        if self.imported_frames as u128 > u32::MAX as u128 + 1 {
            return Err(SequenceRegistrationError::FrameCountExceedsFrameIdRange {
                imported_frames: self.imported_frames,
            });
        }
        if self.frame_ids.len() != self.imported_frames {
            return Err(SequenceRegistrationError::InvalidFrameIds {
                imported_frames: self.imported_frames,
                frame_id_count: self.frame_ids.len(),
                duplicate_frame_ids: Vec::new(),
            });
        }

        let mut expected_frame_ids = BTreeSet::new();
        let mut duplicate_expected_frame_ids = BTreeSet::new();
        for frame_id in self.frame_ids.iter().copied() {
            if !expected_frame_ids.insert(frame_id) {
                duplicate_expected_frame_ids.insert(frame_id);
            }
        }
        if !duplicate_expected_frame_ids.is_empty() {
            return Err(SequenceRegistrationError::InvalidFrameIds {
                imported_frames: self.imported_frames,
                frame_id_count: self.frame_ids.len(),
                duplicate_frame_ids: duplicate_expected_frame_ids.into_iter().collect(),
            });
        }

        let mut observed_frame_ids = BTreeSet::new();
        let mut duplicate_frame_ids = BTreeSet::new();
        for diagnostic in &self.diagnostics {
            if !observed_frame_ids.insert(diagnostic.frame_id) {
                duplicate_frame_ids.insert(diagnostic.frame_id);
            }
        }
        let missing_frame_ids: Vec<_> = expected_frame_ids
            .difference(&observed_frame_ids)
            .copied()
            .collect();
        let duplicate_frame_ids: Vec<_> = duplicate_frame_ids.into_iter().collect();
        let unexpected_frame_ids: Vec<_> = observed_frame_ids
            .difference(&expected_frame_ids)
            .copied()
            .collect();
        if !missing_frame_ids.is_empty()
            || !duplicate_frame_ids.is_empty()
            || !unexpected_frame_ids.is_empty()
        {
            return Err(SequenceRegistrationError::InvalidDiagnostics {
                imported_frames: self.imported_frames,
                diagnostic_count: self.diagnostics.len(),
                missing_frame_ids,
                duplicate_frame_ids,
                unexpected_frame_ids,
            });
        }

        for diagnostic in &self.diagnostics {
            if !diagnostic.inlier_ratio.is_finite()
                || !(0.0..=1.0).contains(&diagnostic.inlier_ratio)
            {
                return Err(SequenceRegistrationError::InvalidDiagnosticMetric {
                    frame_id: diagnostic.frame_id,
                    field: "inlier_ratio",
                });
            }
            if diagnostic
                .mean_reprojection_error
                .is_some_and(|error| !error.is_finite() || error < 0.0)
            {
                return Err(SequenceRegistrationError::InvalidDiagnosticMetric {
                    frame_id: diagnostic.frame_id,
                    field: "mean_reprojection_error",
                });
            }
        }

        let unresolved_frame_ids = self
            .diagnostics
            .iter()
            .filter(|diagnostic| !diagnostic.status.is_registered())
            .map(|diagnostic| diagnostic.frame_id)
            .collect::<Vec<_>>();
        let diagnostic_registered_frames = self.diagnostics.len() - unresolved_frame_ids.len();
        if self.registered_frames != diagnostic_registered_frames {
            return Err(SequenceRegistrationError::RegistrationStatusCountMismatch {
                registered_frames: self.registered_frames,
                diagnostic_registered_frames,
                unresolved_frame_ids,
            });
        }
        if self.imported_frames != self.registered_frames || !unresolved_frame_ids.is_empty() {
            return Err(SequenceRegistrationError::IncompleteCoverage {
                imported_frames: self.imported_frames,
                registered_frames: self.registered_frames,
                unresolved_frame_ids,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SequenceRegistrationError {
    EmptySequence,
    EmptyKeyframes,
    TooManyKeyframes {
        frame_count: usize,
        keyframe_count: usize,
    },
    DuplicateKeyframe {
        frame: usize,
    },
    KeyframeOutOfRange {
        frame: usize,
        frame_count: usize,
    },
    UnsortedKeyframes {
        previous: usize,
        current: usize,
    },
    FrameCountExceedsFrameIdRange {
        imported_frames: usize,
    },
    SequencePlanTooLarge {
        frame_count: usize,
        max_frame_count: usize,
    },
    SequenceNeighborLimitExceeded {
        round: RegistrationRound,
        requested: usize,
        max_neighbors: usize,
    },
    SequenceSupportBudgetExceeded {
        frame_count: usize,
        estimated_support_entries: u128,
        max_support_entries: usize,
    },
    TimestampPlateauTooLarge {
        timestamp_us: i64,
        plateau_size: usize,
        max_plateau_size: usize,
    },
    DynamicSupportLimitExceeded {
        candidate_count: usize,
        max_candidates: usize,
    },
    DynamicSupportNotSortedUnique,
    DynamicSupportFrameOutOfRange {
        frame: usize,
        frame_count: usize,
    },
    InvalidFrameIds {
        imported_frames: usize,
        frame_id_count: usize,
        duplicate_frame_ids: Vec<u32>,
    },
    TimestampCountMismatch {
        frame_count: usize,
        timestamp_count: usize,
    },
    UnsortedTimestamps {
        previous_frame: usize,
        current_frame: usize,
    },
    DiagnosticCountMismatch {
        imported_frames: usize,
        diagnostic_count: usize,
    },
    InvalidConfigMetric {
        field: &'static str,
    },
    InvalidDiagnosticMetric {
        frame_id: u32,
        field: &'static str,
    },
    InvalidDiagnostics {
        imported_frames: usize,
        diagnostic_count: usize,
        missing_frame_ids: Vec<u32>,
        duplicate_frame_ids: Vec<u32>,
        unexpected_frame_ids: Vec<u32>,
    },
    RegistrationStatusCountMismatch {
        registered_frames: usize,
        diagnostic_registered_frames: usize,
        unresolved_frame_ids: Vec<u32>,
    },
    IncompleteCoverage {
        imported_frames: usize,
        registered_frames: usize,
        unresolved_frame_ids: Vec<u32>,
    },
}

impl fmt::Display for SequenceRegistrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySequence => formatter.write_str("sequence must contain at least one frame"),
            Self::EmptyKeyframes => {
                formatter.write_str("sequence registration requires at least one keyframe")
            }
            Self::TooManyKeyframes {
                frame_count,
                keyframe_count,
            } => write!(
                formatter,
                "{keyframe_count} keyframes exceed sequence length {frame_count}"
            ),
            Self::DuplicateKeyframe { frame } => {
                write!(formatter, "duplicate keyframe index {frame}")
            }
            Self::KeyframeOutOfRange { frame, frame_count } => write!(
                formatter,
                "keyframe index {frame} is out of range for {frame_count} frames"
            ),
            Self::UnsortedKeyframes { previous, current } => write!(
                formatter,
                "keyframe indices must be sorted: {current} follows {previous}"
            ),
            Self::FrameCountExceedsFrameIdRange { imported_frames } => write!(
                formatter,
                "{imported_frames} imported frames cannot be represented by u32 frame IDs"
            ),
            Self::SequencePlanTooLarge {
                frame_count,
                max_frame_count,
            } => write!(
                formatter,
                "sequence plan frame count {frame_count} exceeds supported maximum {max_frame_count}"
            ),
            Self::SequenceNeighborLimitExceeded {
                round,
                requested,
                max_neighbors,
            } => write!(
                formatter,
                "{round:?} registration neighbor count {requested} exceeds supported maximum {max_neighbors}"
            ),
            Self::SequenceSupportBudgetExceeded {
                frame_count,
                estimated_support_entries,
                max_support_entries,
            } => write!(
                formatter,
                "sequence plan for {frame_count} frames may cache {estimated_support_entries} support entries, exceeding maximum {max_support_entries}"
            ),
            Self::TimestampPlateauTooLarge {
                timestamp_us,
                plateau_size,
                max_plateau_size,
            } => write!(
                formatter,
                "timestamp {timestamp_us} plateau contains {plateau_size} frames, exceeding maximum {max_plateau_size}"
            ),
            Self::DynamicSupportLimitExceeded {
                candidate_count,
                max_candidates,
            } => write!(
                formatter,
                "dynamic support contains {candidate_count} candidates, exceeding maximum {max_candidates}"
            ),
            Self::DynamicSupportNotSortedUnique => {
                formatter.write_str("dynamic support must be sorted and unique")
            }
            Self::DynamicSupportFrameOutOfRange { frame, frame_count } => write!(
                formatter,
                "dynamic support frame {frame} is out of range for {frame_count} frames"
            ),
            Self::InvalidFrameIds {
                imported_frames,
                frame_id_count,
                duplicate_frame_ids,
            } => write!(
                formatter,
                "invalid expected frame IDs: expected {imported_frames}, found {frame_id_count}; duplicate frame IDs {duplicate_frame_ids:?}"
            ),
            Self::TimestampCountMismatch {
                frame_count,
                timestamp_count,
            } => write!(
                formatter,
                "timestamp count {timestamp_count} does not match frame count {frame_count}"
            ),
            Self::UnsortedTimestamps {
                previous_frame,
                current_frame,
            } => write!(
                formatter,
                "frame {current_frame} timestamp precedes frame {previous_frame}"
            ),
            Self::DiagnosticCountMismatch {
                imported_frames,
                diagnostic_count,
            } => write!(
                formatter,
                "diagnostic count {diagnostic_count} does not match imported frame count {imported_frames}"
            ),
            Self::InvalidConfigMetric { field } => {
                write!(formatter, "sequence registration config metric {field} is invalid")
            }
            Self::InvalidDiagnosticMetric { frame_id, field } => write!(
                formatter,
                "frame {frame_id} registration diagnostic metric {field} is invalid"
            ),
            Self::InvalidDiagnostics {
                imported_frames,
                diagnostic_count,
                missing_frame_ids,
                duplicate_frame_ids,
                unexpected_frame_ids,
            } => write!(
                formatter,
                "invalid sequence diagnostics: expected {imported_frames} records, found {diagnostic_count}; missing frame IDs {missing_frame_ids:?}; duplicate frame IDs {duplicate_frame_ids:?}; unexpected frame IDs {unexpected_frame_ids:?}"
            ),
            Self::RegistrationStatusCountMismatch {
                registered_frames,
                diagnostic_registered_frames,
                unresolved_frame_ids,
            } => {
                write!(
                    formatter,
                    "registered frame count {registered_frames} disagrees with {diagnostic_registered_frames} registered diagnostics"
                )?;
                if !unresolved_frame_ids.is_empty() {
                    formatter.write_str("; unresolved frame")?;
                    if unresolved_frame_ids.len() != 1 {
                        formatter.write_str("s")?;
                    }
                    for (index, frame_id) in unresolved_frame_ids.iter().enumerate() {
                        if index == 0 {
                            write!(formatter, " {frame_id}")?;
                        } else {
                            write!(formatter, ", {frame_id}")?;
                        }
                    }
                }
                Ok(())
            }
            Self::IncompleteCoverage {
                imported_frames,
                registered_frames,
                unresolved_frame_ids,
            } => {
                write!(
                    formatter,
                    "incomplete sequence pose coverage: registered {registered_frames} of {imported_frames} imported frames"
                )?;
                if !unresolved_frame_ids.is_empty() {
                    formatter.write_str("; unresolved frame")?;
                    if unresolved_frame_ids.len() != 1 {
                        formatter.write_str("s")?;
                    }
                    for (index, frame_id) in unresolved_frame_ids.iter().enumerate() {
                        if index == 0 {
                            write!(formatter, " {frame_id}")?;
                        } else {
                            write!(formatter, ", {frame_id}")?;
                        }
                    }
                }
                Ok(())
            }
        }
    }
}

impl Error for SequenceRegistrationError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceRegistrationPlan {
    frame_count: usize,
    keyframes: Vec<usize>,
    narrow_neighbors_each_side: usize,
    wide_neighbors_each_side: usize,
    frame_ids: Vec<u32>,
    timestamps_us: Option<Vec<i64>>,
    pending: Vec<usize>,
    narrow_support: Vec<Vec<usize>>,
    wide_support: Vec<Vec<usize>>,
}

impl SequenceRegistrationPlan {
    pub fn build(
        frame_count: usize,
        keyframes: &[usize],
        narrow_neighbors_each_side: usize,
        wide_neighbors_each_side: usize,
    ) -> Result<Self, SequenceRegistrationError> {
        validate_plan_limits(
            frame_count,
            narrow_neighbors_each_side,
            wide_neighbors_each_side,
        )?;
        validate_keyframes(frame_count, keyframes)?;
        let frame_ids = (0..frame_count).map(|frame| frame as u32).collect();
        Self::build_validated(
            frame_count,
            keyframes,
            narrow_neighbors_each_side,
            wide_neighbors_each_side,
            frame_ids,
            None,
        )
    }

    pub fn build_from_frames(
        frames: &[SequenceFrame],
        keyframes: &[usize],
        narrow_neighbors_each_side: usize,
        wide_neighbors_each_side: usize,
    ) -> Result<Self, SequenceRegistrationError> {
        validate_plan_limits(
            frames.len(),
            narrow_neighbors_each_side,
            wide_neighbors_each_side,
        )?;
        validate_keyframes(frames.len(), keyframes)?;
        validate_timestamp_inputs(frames.len(), frames.iter().map(|frame| frame.timestamp_us))?;
        let frame_ids = frames.iter().map(|frame| frame.id).collect();
        let timestamps_us = frames.iter().map(|frame| frame.timestamp_us).collect();
        Self::build_validated(
            frames.len(),
            keyframes,
            narrow_neighbors_each_side,
            wide_neighbors_each_side,
            frame_ids,
            timestamps_us,
        )
    }

    fn build_validated(
        frame_count: usize,
        keyframes: &[usize],
        narrow_neighbors_each_side: usize,
        wide_neighbors_each_side: usize,
        frame_ids: Vec<u32>,
        timestamps_us: Option<Vec<i64>>,
    ) -> Result<Self, SequenceRegistrationError> {
        validate_plan_limits(
            frame_count,
            narrow_neighbors_each_side,
            wide_neighbors_each_side,
        )?;
        validate_plan_ordering(frame_count, &frame_ids, timestamps_us.as_deref())?;

        let pending = (0..frame_count)
            .filter(|frame| keyframes.binary_search(frame).is_err())
            .collect();
        let narrow_support = build_support_lists(
            frame_count,
            keyframes,
            narrow_neighbors_each_side,
            &frame_ids,
            timestamps_us.as_deref(),
        );
        let wide_support = build_support_lists(
            frame_count,
            keyframes,
            wide_neighbors_each_side,
            &frame_ids,
            timestamps_us.as_deref(),
        );

        Ok(Self {
            frame_count,
            keyframes: keyframes.to_vec(),
            narrow_neighbors_each_side,
            wide_neighbors_each_side,
            frame_ids,
            timestamps_us,
            pending,
            narrow_support,
            wide_support,
        })
    }

    pub fn pending_frames(&self) -> &[usize] {
        &self.pending
    }

    pub fn attempts_for(&self, frame: usize, round: RegistrationRound) -> &[usize] {
        let support = match round {
            RegistrationRound::Narrow => &self.narrow_support,
            RegistrationRound::Wide => &self.wide_support,
        };
        support.get(frame).map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn attempts_for_with_support(
        &self,
        frame: usize,
        round: RegistrationRound,
        registered_support: &[usize],
    ) -> Vec<usize> {
        if frame >= self.frame_count || self.keyframes.binary_search(&frame).is_ok() {
            return Vec::new();
        }
        if registered_support.len() > MAX_DYNAMIC_SUPPORT_CANDIDATES {
            return self.attempts_for(frame, round).to_vec();
        }

        let mut registered_support: Vec<_> = registered_support
            .iter()
            .copied()
            .filter(|support| *support < self.frame_count && *support != frame)
            .collect();
        registered_support.sort_unstable();
        registered_support.dedup();
        self.attempts_for_with_sorted_support(frame, round, &registered_support)
            .unwrap_or_else(|_| self.attempts_for(frame, round).to_vec())
    }

    pub fn attempts_for_with_sorted_support(
        &self,
        frame: usize,
        round: RegistrationRound,
        registered_support: &[usize],
    ) -> Result<Vec<usize>, SequenceRegistrationError> {
        if registered_support.len() > MAX_DYNAMIC_SUPPORT_CANDIDATES {
            return Err(SequenceRegistrationError::DynamicSupportLimitExceeded {
                candidate_count: registered_support.len(),
                max_candidates: MAX_DYNAMIC_SUPPORT_CANDIDATES,
            });
        }
        if registered_support.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(SequenceRegistrationError::DynamicSupportNotSortedUnique);
        }
        if let Some(out_of_range) = registered_support
            .iter()
            .copied()
            .find(|support| *support >= self.frame_count)
        {
            return Err(SequenceRegistrationError::DynamicSupportFrameOutOfRange {
                frame: out_of_range,
                frame_count: self.frame_count,
            });
        }
        if frame >= self.frame_count || self.keyframes.binary_search(&frame).is_ok() {
            return Ok(Vec::new());
        }

        let mut keyframe_support = self.attempts_for(frame, round).to_vec();
        keyframe_support.sort_unstable();
        let candidates = merge_sorted_support(&keyframe_support, registered_support, frame);
        let neighbors_each_side = match round {
            RegistrationRound::Narrow => self.narrow_neighbors_each_side,
            RegistrationRound::Wide => self.wide_neighbors_each_side,
        };
        Ok(support_for(
            frame,
            &candidates,
            neighbors_each_side,
            &self.frame_ids,
            self.timestamps_us.as_deref(),
        ))
    }
}

#[derive(Serialize)]
struct SequenceRegistrationPlanRef<'a> {
    frame_count: usize,
    keyframes: &'a [usize],
    narrow_neighbors_each_side: usize,
    wide_neighbors_each_side: usize,
    frame_ids: &'a [u32],
    timestamps_us: Option<&'a [i64]>,
}

#[derive(Deserialize)]
struct SequenceRegistrationPlanWire {
    frame_count: usize,
    keyframes: Vec<usize>,
    narrow_neighbors_each_side: usize,
    wide_neighbors_each_side: usize,
    frame_ids: Vec<u32>,
    timestamps_us: Option<Vec<i64>>,
}

impl Serialize for SequenceRegistrationPlan {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SequenceRegistrationPlanRef {
            frame_count: self.frame_count,
            keyframes: &self.keyframes,
            narrow_neighbors_each_side: self.narrow_neighbors_each_side,
            wide_neighbors_each_side: self.wide_neighbors_each_side,
            frame_ids: &self.frame_ids,
            timestamps_us: self.timestamps_us.as_deref(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SequenceRegistrationPlan {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SequenceRegistrationPlanWire::deserialize(deserializer)?;
        validate_plan_limits(
            wire.frame_count,
            wire.narrow_neighbors_each_side,
            wire.wide_neighbors_each_side,
        )
        .map_err(de::Error::custom)?;
        validate_keyframes(wire.frame_count, &wire.keyframes).map_err(de::Error::custom)?;
        Self::build_validated(
            wire.frame_count,
            &wire.keyframes,
            wire.narrow_neighbors_each_side,
            wire.wide_neighbors_each_side,
            wire.frame_ids,
            wire.timestamps_us,
        )
        .map_err(de::Error::custom)
    }
}

fn validate_plan_limits(
    frame_count: usize,
    narrow_neighbors_each_side: usize,
    wide_neighbors_each_side: usize,
) -> Result<(), SequenceRegistrationError> {
    if frame_count > MAX_SEQUENCE_PLAN_FRAMES {
        return Err(SequenceRegistrationError::SequencePlanTooLarge {
            frame_count,
            max_frame_count: MAX_SEQUENCE_PLAN_FRAMES,
        });
    }
    for (round, requested) in [
        (RegistrationRound::Narrow, narrow_neighbors_each_side),
        (RegistrationRound::Wide, wide_neighbors_each_side),
    ] {
        if requested > MAX_SEQUENCE_NEIGHBORS {
            return Err(SequenceRegistrationError::SequenceNeighborLimitExceeded {
                round,
                requested,
                max_neighbors: MAX_SEQUENCE_NEIGHBORS,
            });
        }
    }

    let estimated_support_entries = frame_count as u128
        * 2
        * (narrow_neighbors_each_side as u128 + wide_neighbors_each_side as u128);
    if estimated_support_entries > MAX_TOTAL_SUPPORT_ENTRIES as u128 {
        return Err(SequenceRegistrationError::SequenceSupportBudgetExceeded {
            frame_count,
            estimated_support_entries,
            max_support_entries: MAX_TOTAL_SUPPORT_ENTRIES,
        });
    }
    Ok(())
}

fn merge_sorted_support(
    keyframe_support: &[usize],
    registered_support: &[usize],
    target_frame: usize,
) -> Vec<usize> {
    let mut merged = Vec::with_capacity(
        keyframe_support
            .len()
            .saturating_add(registered_support.len()),
    );
    let mut keyframe_index = 0;
    let mut registered_index = 0;
    while keyframe_index < keyframe_support.len() || registered_index < registered_support.len() {
        let next = match (
            keyframe_support.get(keyframe_index),
            registered_support.get(registered_index),
        ) {
            (Some(keyframe), Some(registered)) if keyframe < registered => {
                keyframe_index += 1;
                *keyframe
            }
            (Some(keyframe), Some(registered)) if registered < keyframe => {
                registered_index += 1;
                *registered
            }
            (Some(keyframe), Some(_)) => {
                keyframe_index += 1;
                registered_index += 1;
                *keyframe
            }
            (Some(keyframe), None) => {
                keyframe_index += 1;
                *keyframe
            }
            (None, Some(registered)) => {
                registered_index += 1;
                *registered
            }
            (None, None) => break,
        };
        if next != target_frame {
            merged.push(next);
        }
    }
    merged
}

fn validate_keyframes(
    frame_count: usize,
    keyframes: &[usize],
) -> Result<(), SequenceRegistrationError> {
    if frame_count > MAX_SEQUENCE_PLAN_FRAMES {
        return Err(SequenceRegistrationError::SequencePlanTooLarge {
            frame_count,
            max_frame_count: MAX_SEQUENCE_PLAN_FRAMES,
        });
    }
    if frame_count as u128 > u32::MAX as u128 + 1 {
        return Err(SequenceRegistrationError::FrameCountExceedsFrameIdRange {
            imported_frames: frame_count,
        });
    }
    if frame_count == 0 {
        return Err(SequenceRegistrationError::EmptySequence);
    }
    if keyframes.len() > frame_count {
        return Err(SequenceRegistrationError::TooManyKeyframes {
            frame_count,
            keyframe_count: keyframes.len(),
        });
    }
    if keyframes.is_empty() {
        return Err(SequenceRegistrationError::EmptyKeyframes);
    }

    let first = keyframes[0];
    if first >= frame_count {
        return Err(SequenceRegistrationError::KeyframeOutOfRange {
            frame: first,
            frame_count,
        });
    }
    for pair in keyframes.windows(2) {
        let current = pair[1];
        if current >= frame_count {
            return Err(SequenceRegistrationError::KeyframeOutOfRange {
                frame: current,
                frame_count,
            });
        }
        if pair[0] == current {
            return Err(SequenceRegistrationError::DuplicateKeyframe { frame: current });
        }
        if pair[0] > pair[1] {
            return Err(SequenceRegistrationError::UnsortedKeyframes {
                previous: pair[0],
                current: pair[1],
            });
        }
    }
    Ok(())
}

fn validate_plan_ordering(
    frame_count: usize,
    frame_ids: &[u32],
    timestamps_us: Option<&[i64]>,
) -> Result<(), SequenceRegistrationError> {
    if frame_ids.len() != frame_count {
        return Err(SequenceRegistrationError::InvalidFrameIds {
            imported_frames: frame_count,
            frame_id_count: frame_ids.len(),
            duplicate_frame_ids: Vec::new(),
        });
    }
    if let Some(timestamps_us) = timestamps_us {
        validate_timestamp_inputs(frame_count, timestamps_us.iter().copied().map(Some))?;
    }
    let mut observed = HashSet::with_capacity(frame_ids.len());
    let mut duplicates = BTreeSet::new();
    for frame_id in frame_ids.iter().copied() {
        if !observed.insert(frame_id) {
            duplicates.insert(frame_id);
        }
    }
    if !duplicates.is_empty() {
        return Err(SequenceRegistrationError::InvalidFrameIds {
            imported_frames: frame_count,
            frame_id_count: frame_ids.len(),
            duplicate_frame_ids: duplicates.into_iter().collect(),
        });
    }
    Ok(())
}

fn validate_timestamp_inputs<I>(
    frame_count: usize,
    timestamps: I,
) -> Result<(), SequenceRegistrationError>
where
    I: IntoIterator<Item = Option<i64>>,
{
    let mut timestamp_count = 0;
    let mut all_present = true;
    let mut previous_timestamp = None;
    let mut plateau_start = 0;
    let mut first_error = None;

    for (current_frame, timestamp) in timestamps.into_iter().enumerate() {
        timestamp_count += 1;
        let Some(timestamp) = timestamp else {
            all_present = false;
            continue;
        };
        if !all_present {
            continue;
        }

        if let Some(previous_timestamp_value) = previous_timestamp {
            if previous_timestamp_value > timestamp {
                if first_error.is_none() {
                    first_error = Some(SequenceRegistrationError::UnsortedTimestamps {
                        previous_frame: current_frame - 1,
                        current_frame,
                    });
                }
            } else if previous_timestamp_value != timestamp {
                if first_error.is_none() {
                    first_error = timestamp_plateau_error(
                        previous_timestamp_value,
                        current_frame.saturating_sub(plateau_start),
                    )
                    .err();
                }
                plateau_start = current_frame;
            }
        } else {
            plateau_start = current_frame;
        }
        previous_timestamp = Some(timestamp);
    }

    if timestamp_count != frame_count {
        return Err(SequenceRegistrationError::TimestampCountMismatch {
            frame_count,
            timestamp_count,
        });
    }
    if !all_present {
        return Ok(());
    }
    if let Some(error) = first_error {
        return Err(error);
    }
    if let Some(timestamp) = previous_timestamp {
        return timestamp_plateau_error(timestamp, timestamp_count.saturating_sub(plateau_start));
    }
    Ok(())
}

#[cfg(test)]
mod task6_tests {
    use super::*;

    fn single_frame(path: PathBuf) -> SequenceFrame {
        SequenceFrame {
            id: 42,
            image_path: path,
            timestamp_us: Some(0),
        }
    }

    fn diagnostic(inliers: usize, ratio: f64, error: Option<f64>) -> FrameRegistrationDiagnostic {
        FrameRegistrationDiagnostic {
            frame_id: 42,
            status: FrameRegistrationStatus::Unresolved,
            attempts: 1,
            support_frame_ids: vec![7],
            inlier_count: inliers,
            inlier_ratio: ratio,
            mean_reprojection_error: error,
            message: None,
        }
    }

    #[test]
    fn registration_acceptance_requires_all_finite_thresholds() {
        let config = SequenceRegistrationConfig {
            min_inliers: 24,
            min_inlier_ratio: 0.2,
            max_reprojection_error: 4.0,
            use_gpu_pnp: false,
            ..Default::default()
        };

        assert!(accepts_registration(
            &diagnostic(24, 0.2, Some(4.0)),
            &config
        ));
        assert!(!accepts_registration(
            &diagnostic(23, 0.2, Some(4.0)),
            &config
        ));
        assert!(!accepts_registration(
            &diagnostic(24, 0.19, Some(4.0)),
            &config
        ));
        assert!(!accepts_registration(
            &diagnostic(24, 0.2, Some(f64::NAN)),
            &config
        ));
        assert!(!accepts_registration(&diagnostic(24, 0.2, None), &config));
    }

    #[test]
    fn registration_json_is_atomic_and_cleans_temporary_file() -> anyhow::Result<()> {
        let output = tempfile::tempdir()?;
        let result = SequenceRegistrationResult {
            imported_frames: 1,
            registered_frames: 1,
            frame_ids: vec![42],
            diagnostics: vec![FrameRegistrationDiagnostic::new(
                42,
                FrameRegistrationStatus::Keyframe,
            )],
            sparse_model: output.path().join("sparse/0"),
        };

        write_registration_result_atomic(output.path(), &result)?;

        assert!(!output.path().join("registration.json.tmp").exists());
        let restored: SequenceRegistrationResult = serde_json::from_reader(std::fs::File::open(
            output.path().join("registration.json"),
        )?)?;
        assert_eq!(restored, result);
        Ok(())
    }

    #[test]
    fn registration_json_failure_preserves_existing_destination() -> anyhow::Result<()> {
        let output = tempfile::tempdir()?;
        let destination = output.path().join("registration.json");
        let temporary = output.path().join("registration.json.tmp");
        std::fs::write(&destination, b"previous-valid-json\n")?;
        std::fs::create_dir(&temporary)?;
        let result = SequenceRegistrationResult {
            imported_frames: 1,
            registered_frames: 1,
            frame_ids: vec![42],
            diagnostics: vec![FrameRegistrationDiagnostic::new(
                42,
                FrameRegistrationStatus::Keyframe,
            )],
            sparse_model: output.path().join("sparse/0"),
        };

        let error = write_registration_result_atomic(output.path(), &result).unwrap_err();

        assert!(error.to_string().contains("Is a directory"));
        assert_eq!(std::fs::read(&destination)?, b"previous-valid-json\n");
        assert!(temporary.is_dir());
        Ok(())
    }

    #[test]
    fn stable_image_reuse_rejects_different_existing_contents() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let source_dir = temp.path().join("source");
        let destination_dir = temp.path().join("destination");
        std::fs::create_dir_all(&source_dir)?;
        std::fs::create_dir_all(&destination_dir)?;
        let source = source_dir.join("frame.png");
        let destination = destination_dir.join("frame.png");
        std::fs::write(&source, b"current-frame")?;
        std::fs::write(&destination, b"stale-frame")?;

        let error = link_or_copy_stable_image(&source, &destination_dir).unwrap_err();

        assert!(error.to_string().contains("does not match source"));
        assert_eq!(std::fs::read(destination)?, b"stale-frame");
        Ok(())
    }

    #[test]
    fn database_import_rejects_stale_existing_image_metadata() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let image_path = temp.path().join("frame.png");
        image::GrayImage::new(32, 24).save(&image_path)?;
        let database_path = temp.path().join("database.db");
        let database = ColmapDatabase::open(&database_path)?;
        database.write_camera(
            &ColmapDatabaseCamera {
                camera: ColmapCamera {
                    camera_id: 1,
                    model_id: COLMAP_PINHOLE,
                    width: 32,
                    height: 24,
                    params: vec![38.4, 38.4, 16.0, 12.0],
                },
                has_prior_focal_length: false,
            },
            true,
        )?;
        database.write_image(
            &ColmapDatabaseImage {
                image_id: 1,
                name: "wrong.png".to_owned(),
                camera_id: 1,
                frame_id: None,
            },
            true,
        )?;
        drop(database);

        let error = import_database_images(
            &[single_frame(image_path)],
            &[0],
            &MapperConfig::default(),
            &database_path,
        )
        .unwrap_err();

        assert!(error
            .to_string()
            .contains("image_id=1 metadata does not match"));
        Ok(())
    }

    #[test]
    fn sparse_directory_replace_restores_destination_when_staged_rename_fails() -> anyhow::Result<()>
    {
        let temp = tempfile::tempdir()?;
        let destination = temp.path().join("0");
        let missing_staged = temp.path().join("0.tmp");
        let backup = temp.path().join("0.backup");
        std::fs::create_dir(&destination)?;
        std::fs::write(destination.join("marker"), b"original-model")?;

        let error = replace_sparse_directory(&destination, &missing_staged, &backup).unwrap_err();

        assert!(error.to_string().contains("No such file"));
        assert_eq!(
            std::fs::read(destination.join("marker"))?,
            b"original-model"
        );
        assert!(!backup.exists());
        assert!(!missing_staged.exists());
        Ok(())
    }
}

fn timestamp_plateau_error(
    timestamp_us: i64,
    plateau_size: usize,
) -> Result<(), SequenceRegistrationError> {
    if plateau_size > MAX_TIMESTAMP_PLATEAU {
        return Err(SequenceRegistrationError::TimestampPlateauTooLarge {
            timestamp_us,
            plateau_size,
            max_plateau_size: MAX_TIMESTAMP_PLATEAU,
        });
    }
    Ok(())
}

fn build_support_lists(
    frame_count: usize,
    keyframes: &[usize],
    neighbors_each_side: usize,
    frame_ids: &[u32],
    timestamps_us: Option<&[i64]>,
) -> Vec<Vec<usize>> {
    (0..frame_count)
        .map(|frame| {
            if keyframes.binary_search(&frame).is_ok() {
                Vec::new()
            } else {
                support_for(
                    frame,
                    keyframes,
                    neighbors_each_side,
                    frame_ids,
                    timestamps_us,
                )
            }
        })
        .collect()
}

fn support_for(
    frame: usize,
    candidates: &[usize],
    neighbors_each_side: usize,
    frame_ids: &[u32],
    timestamps_us: Option<&[i64]>,
) -> Vec<usize> {
    let left_end = candidates.partition_point(|candidate| *candidate < frame);
    let right_start = candidates.partition_point(|candidate| *candidate <= frame);
    let mut support = if let Some(timestamps_us) = timestamps_us {
        let left_candidates = bounded_left_timestamp_candidates(
            candidates,
            left_end,
            neighbors_each_side,
            timestamps_us,
        );
        let right_candidates = bounded_right_timestamp_candidates(
            candidates,
            right_start,
            neighbors_each_side,
            timestamps_us,
        );
        let mut support = select_top_support(
            frame,
            left_candidates,
            neighbors_each_side,
            frame_ids,
            Some(timestamps_us),
        );
        support.extend(select_top_support(
            frame,
            right_candidates,
            neighbors_each_side,
            frame_ids,
            Some(timestamps_us),
        ));
        support
    } else {
        let left_start = left_end.saturating_sub(neighbors_each_side);
        let right_end = right_start
            .saturating_add(neighbors_each_side)
            .min(candidates.len());
        let mut support = Vec::with_capacity(
            left_end.saturating_sub(left_start) + right_end.saturating_sub(right_start),
        );
        support.extend_from_slice(&candidates[left_start..left_end]);
        support.extend_from_slice(&candidates[right_start..right_end]);
        support
    };
    support.sort_by_key(|candidate| support_key(frame, *candidate, frame_ids, timestamps_us));
    support
}

fn bounded_left_timestamp_candidates<'a>(
    candidates: &'a [usize],
    left_end: usize,
    limit: usize,
    timestamps_us: &[i64],
) -> &'a [usize] {
    if limit == 0 || left_end == 0 {
        return &candidates[left_end..left_end];
    }

    let initial_start = left_end.saturating_sub(limit);
    let cutoff_timestamp = timestamps_us[candidates[initial_start]];
    let plateau_start = candidates[..initial_start]
        .partition_point(|candidate| timestamps_us[*candidate] < cutoff_timestamp);
    &candidates[plateau_start..left_end]
}

fn bounded_right_timestamp_candidates<'a>(
    candidates: &'a [usize],
    right_start: usize,
    limit: usize,
    timestamps_us: &[i64],
) -> &'a [usize] {
    let right_candidates = &candidates[right_start..];
    if limit == 0 || right_candidates.is_empty() {
        return &right_candidates[..0];
    }

    let initial_len = limit.min(right_candidates.len());
    let cutoff_timestamp = timestamps_us[right_candidates[initial_len - 1]];
    let plateau_end =
        right_candidates.partition_point(|candidate| timestamps_us[*candidate] <= cutoff_timestamp);
    &right_candidates[..plateau_end]
}

fn select_top_support(
    frame: usize,
    candidates: &[usize],
    limit: usize,
    frame_ids: &[u32],
    timestamps_us: Option<&[i64]>,
) -> Vec<usize> {
    if limit == 0 || candidates.is_empty() {
        return Vec::new();
    }

    let mut selected = BinaryHeap::with_capacity(limit.min(candidates.len()));
    for candidate in candidates.iter().copied() {
        let entry = (
            support_key(frame, candidate, frame_ids, timestamps_us),
            candidate,
        );
        if selected.len() < limit {
            selected.push(entry);
        } else if selected.peek().is_some_and(|worst| entry < *worst) {
            selected.pop();
            selected.push(entry);
        }
    }
    selected
        .into_iter()
        .map(|(_, candidate)| candidate)
        .collect()
}

fn support_key(
    frame: usize,
    candidate: usize,
    frame_ids: &[u32],
    timestamps_us: Option<&[i64]>,
) -> (u128, u32) {
    let distance = if let Some(timestamps_us) = timestamps_us {
        timestamps_us[candidate].abs_diff(timestamps_us[frame]) as u128
    } else {
        candidate.abs_diff(frame) as u128
    };
    (distance, frame_ids[candidate])
}
