use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path, PathBuf};

use thiserror::Error;

use crate::media::ImportedFrame;
use crate::pipeline::{
    ArtifactValidation, PendingArtifact, PipelineProgressDetail, PnpWorker, SfmWorker,
    StageRequest, WorkerControl, WorkerEventSink, WorkerOutcome,
};
use crate::project::{
    KeyframeSelectionMode, ProjectErrorRecord, ProjectStage, SourceKind, StageState,
    SuggestedAction,
};

#[derive(Debug, Default, Clone, Copy)]
pub struct RustSfmWorker;

struct ImportedSequence {
    frames: Vec<rustsfm::SequenceFrame>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct KeyframeStageResult {
    mode: KeyframeSelectionMode,
    imported_frames: usize,
    selected_keyframe_count: usize,
    selected_keyframe_ids: Vec<u32>,
    registered_keyframes: usize,
    selection_config: Option<rustsfm::AdaptiveKeyframeSelectionConfig>,
    evaluated_pairs: usize,
    diagnostics: Vec<rustsfm::AdaptiveKeyframePairDiagnostic>,
}

#[derive(Debug, Clone, PartialEq)]
struct ResolvedKeyframeSelection {
    selected_frame_ids: Vec<u32>,
    selection_config: Option<rustsfm::AdaptiveKeyframeSelectionConfig>,
    evaluated_pairs: usize,
    diagnostics: Vec<rustsfm::AdaptiveKeyframePairDiagnostic>,
}

#[derive(Debug, Error)]
pub(crate) enum RustSfmWorkerError {
    #[error("RustSFM is only available for image-sequence projects")]
    UnsupportedSource,
    #[error("the imported project is missing frame metadata")]
    MissingFrameMetadata,
    #[error("project manifest is invalid for RustSFM: {0}")]
    Manifest(String),
    #[error("imported frame metadata is invalid: {0}")]
    FrameMetadata(#[from] serde_json::Error),
    #[error("imported frame path is unsafe: {0}")]
    UnsafeFramePath(String),
    #[error("imported frame is missing or not a regular file: {0}")]
    MissingFrame(PathBuf),
    #[error("RustSFM failed: {0}")]
    Sfm(String),
    #[error(
        "RustSFM registered {registered_frames} of {imported_frames} frames; RustGS remains blocked"
    )]
    IncompletePoseCoverage {
        imported_frames: usize,
        registered_frames: usize,
    },
    #[error("filesystem error: {0}")]
    Io(#[from] std::io::Error),
}

impl SfmWorker for RustSfmWorker {
    fn run(
        &self,
        request: StageRequest,
        control: WorkerControl,
        events: WorkerEventSink,
    ) -> WorkerOutcome {
        let sequence = match load_imported_sequence(&request) {
            Ok(sequence) => sequence,
            Err(error) => return worker_failure(ProjectStage::KeyframeSfm, error),
        };
        let output = match worker_output_directory(&request) {
            Ok(output) => output,
            Err(error) => return worker_failure(ProjectStage::KeyframeSfm, error),
        };
        let mapper_config = mapper_config_for(&request);

        let sfm_control = rustsfm::SfmTaskControl::new();
        let mut task_sink = progress_sink(&control, &events, &sfm_control);
        let mut task = rustsfm::SfmTaskContext::new(&sfm_control, &mut task_sink);
        let selection = resolve_keyframe_selection_with(
            request.manifest.sfm_config.keyframe_selection,
            &request.manifest.sfm_config.adaptive_keyframes,
            &sequence.frames,
            &mapper_config,
            &output,
            &mut task,
            rustsfm::run_adaptive_keyframe_selection,
        );
        let selection = match selection {
            Ok(selection) => selection,
            Err(error) => {
                drop(task);
                drop(task_sink);
                let _ = fs::remove_dir_all(&output);
                return worker_failure(ProjectStage::KeyframeSfm, error);
            }
        };
        if let Some(outcome) = requested_stop(&control) {
            drop(task);
            drop(task_sink);
            let _ = fs::remove_dir_all(&output);
            return outcome;
        }
        let result = rustsfm::run_keyframe_reconstruction(
            &sequence.frames,
            &selection.selected_frame_ids,
            &mapper_config,
            &output,
            &mut task,
        );
        drop(task);
        drop(task_sink);

        if let Some(outcome) = requested_stop(&control) {
            let _ = fs::remove_dir_all(&output);
            return outcome;
        }
        let result = match result {
            Ok(result) => result,
            Err(error) => {
                let _ = fs::remove_dir_all(&output);
                return worker_failure(
                    ProjectStage::KeyframeSfm,
                    RustSfmWorkerError::Sfm(error.to_string()),
                );
            }
        };
        if result.keyframe_ids != selection.selected_frame_ids {
            let _ = fs::remove_dir_all(&output);
            return worker_failure(
                ProjectStage::KeyframeSfm,
                RustSfmWorkerError::Sfm(
                    "RustSFM reconstruction returned different keyframe IDs than selection"
                        .to_owned(),
                ),
            );
        }
        let artifacts = match keyframe_stage_artifacts(
            request.manifest.sfm_config.keyframe_selection,
            &selection,
            &result,
        ) {
            Ok(artifacts) => artifacts,
            Err(error) => {
                let _ = fs::remove_dir_all(&output);
                return worker_failure(ProjectStage::KeyframeSfm, error);
            }
        };
        if let Err(error) = fs::remove_dir_all(&output) {
            return worker_failure(ProjectStage::KeyframeSfm, RustSfmWorkerError::Io(error));
        }
        WorkerOutcome::Succeeded(artifacts)
    }
}

impl PnpWorker for RustSfmWorker {
    fn run(
        &self,
        request: StageRequest,
        control: WorkerControl,
        events: WorkerEventSink,
    ) -> WorkerOutcome {
        let sequence = match load_imported_sequence(&request) {
            Ok(sequence) => sequence,
            Err(error) => return worker_failure(ProjectStage::FullFramePnp, error),
        };
        let output = match worker_output_directory(&request) {
            Ok(output) => output,
            Err(error) => return worker_failure(ProjectStage::FullFramePnp, error),
        };
        let mapper_config = mapper_config_for(&request);
        let registration_config = registration_config_for(&request);
        let keyframes = match hydrate_keyframe_result(&request, &sequence.frames, &output) {
            Ok(keyframes) => keyframes,
            Err(error) => {
                let _ = fs::remove_dir_all(&output);
                return worker_failure(ProjectStage::FullFramePnp, error);
            }
        };

        let sfm_control = rustsfm::SfmTaskControl::new();
        let mut task_sink = progress_sink(&control, &events, &sfm_control);
        let mut task = rustsfm::SfmTaskContext::new(&sfm_control, &mut task_sink);
        let result = run_remaining_registration_with(
            &sequence.frames,
            &keyframes,
            &mapper_config,
            &registration_config,
            &output,
            &mut task,
            rustsfm::register_remaining_sequence_frames,
        );
        drop(task);
        drop(task_sink);

        if let Some(outcome) = requested_stop(&control) {
            let _ = fs::remove_dir_all(&output);
            return outcome;
        }
        let result = match result {
            Ok(result) => result,
            Err(error) => {
                let _ = fs::remove_dir_all(&output);
                return worker_failure(
                    ProjectStage::FullFramePnp,
                    RustSfmWorkerError::Sfm(error.to_string()),
                );
            }
        };
        if !result.has_complete_coverage() {
            let _ = fs::remove_dir_all(&output);
            return worker_failure(
                ProjectStage::FullFramePnp,
                RustSfmWorkerError::IncompletePoseCoverage {
                    imported_frames: result.imported_frames,
                    registered_frames: result.registered_frames,
                },
            );
        }
        let artifacts = match final_colmap_artifacts(&request, &output, &sequence.frames, &result) {
            Ok(artifacts) => artifacts,
            Err(error) => {
                let _ = fs::remove_dir_all(&output);
                return worker_failure(ProjectStage::FullFramePnp, error);
            }
        };
        if let Err(error) = fs::remove_dir_all(&output) {
            return worker_failure(ProjectStage::FullFramePnp, RustSfmWorkerError::Io(error));
        }
        registration_outcome_with_backend_progress(
            &events,
            &result,
            registration_config.use_gpu_pnp,
            artifacts,
        )
    }
}

fn mapper_config_for(request: &StageRequest) -> rustsfm::MapperConfig {
    let mut mapper_config = rustsfm::MapperConfig::default();
    mapper_config.local_matching = true;
    mapper_config.single_camera = true;
    mapper_config.discover_database = false;
    mapper_config.copy_images = true;
    mapper_config.max_features = 4096;
    mapper_config.matching_pair_strategy = rustsfm::MatchingPairStrategy::LocalWindow { window: 5 };
    mapper_config.sift_extraction.use_gpu = request.manifest.sfm_config.use_gpu_sift;
    mapper_config.sift_matching.use_gpu = request.manifest.sfm_config.use_gpu_matching;
    mapper_config.use_gpu_pnp = request.manifest.pnp_config.use_gpu_pnp;
    mapper_config
}

fn registration_config_for(request: &StageRequest) -> rustsfm::SequenceRegistrationConfig {
    let pnp_config = &request.manifest.pnp_config;
    rustsfm::SequenceRegistrationConfig {
        narrow_neighbors_each_side: pnp_config.narrow_neighbors_each_side,
        wide_neighbors_each_side: pnp_config.wide_neighbors_each_side,
        min_inliers: pnp_config.min_inliers,
        min_inlier_ratio: pnp_config.min_inlier_ratio,
        max_reprojection_error: pnp_config.max_reprojection_error,
        use_gpu_pnp: pnp_config.use_gpu_pnp,
    }
}

fn resolve_keyframe_selection_with<F>(
    mode: KeyframeSelectionMode,
    config: &rustsfm::AdaptiveKeyframeSelectionConfig,
    frames: &[rustsfm::SequenceFrame],
    mapper_config: &rustsfm::MapperConfig,
    output: &Path,
    task: &mut rustsfm::SfmTaskContext<'_>,
    run_adaptive: F,
) -> Result<ResolvedKeyframeSelection, RustSfmWorkerError>
where
    F: FnOnce(
        &[rustsfm::SequenceFrame],
        &rustsfm::AdaptiveKeyframeSelectionConfig,
        &rustsfm::MapperConfig,
        &Path,
        &mut rustsfm::SfmTaskContext<'_>,
    ) -> anyhow::Result<rustsfm::AdaptiveKeyframeSelectionResult>,
{
    if frames.len() < 2 {
        return Err(RustSfmWorkerError::Sfm(
            "at least two imported frames are required for reconstruction".to_owned(),
        ));
    }
    let resolved = match mode {
        KeyframeSelectionMode::Adaptive => {
            let result = run_adaptive(frames, config, mapper_config, output, task)
                .map_err(|error| RustSfmWorkerError::Sfm(error.to_string()))?;
            ResolvedKeyframeSelection {
                selected_frame_ids: result.selected_frame_ids,
                selection_config: Some(result.config),
                evaluated_pairs: result.evaluated_pairs,
                diagnostics: result.diagnostics,
            }
        }
        KeyframeSelectionMode::AllImages => ResolvedKeyframeSelection {
            selected_frame_ids: frames.iter().map(|frame| frame.id).collect(),
            selection_config: None,
            evaluated_pairs: 0,
            diagnostics: Vec::new(),
        },
    };
    validate_selected_frame_ids(frames, &resolved.selected_frame_ids)?;
    Ok(resolved)
}

fn validate_selected_frame_ids(
    frames: &[rustsfm::SequenceFrame],
    selected_frame_ids: &[u32],
) -> Result<(), RustSfmWorkerError> {
    if selected_frame_ids.len() < 2 {
        return Err(RustSfmWorkerError::Sfm(
            "RustSFM keyframe selection returned fewer than two frames".to_owned(),
        ));
    }
    let mut next_selected = 0;
    for frame in frames {
        if selected_frame_ids.get(next_selected) == Some(&frame.id) {
            next_selected += 1;
        }
    }
    if next_selected != selected_frame_ids.len() {
        return Err(RustSfmWorkerError::Sfm(
            "RustSFM keyframe selection must contain known IDs in stable sequence order".to_owned(),
        ));
    }
    Ok(())
}

fn keyframe_stage_artifacts(
    mode: KeyframeSelectionMode,
    selection: &ResolvedKeyframeSelection,
    reconstruction: &rustsfm::KeyframeReconstructionResult,
) -> Result<Vec<PendingArtifact>, RustSfmWorkerError> {
    ensure_regular_project_file(&reconstruction.database)?;
    let stage_result = KeyframeStageResult {
        mode,
        imported_frames: reconstruction.imported_frames,
        selected_keyframe_count: selection.selected_frame_ids.len(),
        selected_keyframe_ids: selection.selected_frame_ids.clone(),
        registered_keyframes: reconstruction.registered_keyframes,
        selection_config: selection.selection_config.clone(),
        evaluated_pairs: selection.evaluated_pairs,
        diagnostics: selection.diagnostics.clone(),
    };
    let mut artifacts = vec![PendingArtifact::new(
        "keyframe-result.json",
        serde_json::to_vec_pretty(&stage_result)
            .expect("serializable RustSFM keyframe stage result"),
        ArtifactValidation::Json,
    )];
    artifacts.push(PendingArtifact::new(
        "rustsfm/database.db",
        fs::read(&reconstruction.database)?,
        ArtifactValidation::ReadableFile,
    ));
    for name in [
        "cameras.txt",
        "images.txt",
        "points3D.txt",
        "cameras.bin",
        "images.bin",
        "points3D.bin",
    ] {
        let path = reconstruction.sparse_model.join(name);
        ensure_regular_project_file(&path)?;
        artifacts.push(PendingArtifact::new(
            format!("rustsfm/keyframe-sparse/0/{name}"),
            fs::read(path)?,
            ArtifactValidation::ReadableFile,
        ));
    }
    Ok(artifacts)
}

fn hydrate_keyframe_result(
    request: &StageRequest,
    frames: &[rustsfm::SequenceFrame],
    output: &Path,
) -> Result<rustsfm::KeyframeReconstructionResult, RustSfmWorkerError> {
    let stage = request
        .manifest
        .try_stage(ProjectStage::KeyframeSfm)
        .map_err(|error| RustSfmWorkerError::Manifest(error.to_string()))?;
    if stage.state() != StageState::Succeeded {
        return Err(RustSfmWorkerError::Sfm(
            "the keyframe stage has not committed successful artifacts".to_owned(),
        ));
    }

    let result_path = committed_keyframe_artifact_path(request, "keyframe-result.json")?;
    ensure_regular_project_file(&result_path)?;
    let stage_result = serde_json::from_slice::<KeyframeStageResult>(&fs::read(result_path)?)
        .map_err(|error| {
            RustSfmWorkerError::Sfm(format!("invalid committed keyframe result: {error}"))
        })?;
    if stage_result.imported_frames != frames.len() {
        return Err(RustSfmWorkerError::Sfm(format!(
            "committed keyframe result describes {} imported frames, current sequence has {}",
            stage_result.imported_frames,
            frames.len()
        )));
    }
    if stage_result.selected_keyframe_count != stage_result.selected_keyframe_ids.len() {
        return Err(RustSfmWorkerError::Sfm(
            "committed keyframe count does not match selected IDs".to_owned(),
        ));
    }
    validate_selected_frame_ids(frames, &stage_result.selected_keyframe_ids)?;
    if stage_result.registered_keyframes > stage_result.selected_keyframe_count {
        return Err(RustSfmWorkerError::Sfm(
            "committed registered keyframe count exceeds selected count".to_owned(),
        ));
    }

    let database = output.join("Cache/database.db");
    copy_committed_keyframe_artifact(request, "rustsfm/database.db", &database)?;
    let sparse_model = output.join("Cache/keyframe-sparse/0");
    for name in [
        "cameras.txt",
        "images.txt",
        "points3D.txt",
        "cameras.bin",
        "images.bin",
        "points3D.bin",
    ] {
        copy_committed_keyframe_artifact(
            request,
            &format!("rustsfm/keyframe-sparse/0/{name}"),
            &sparse_model.join(name),
        )?;
    }

    Ok(rustsfm::KeyframeReconstructionResult {
        imported_frames: stage_result.imported_frames,
        keyframe_ids: stage_result.selected_keyframe_ids,
        registered_keyframes: stage_result.registered_keyframes,
        database,
        sparse_model,
    })
}

fn committed_keyframe_artifact_path(
    request: &StageRequest,
    suffix: &str,
) -> Result<PathBuf, RustSfmWorkerError> {
    let stage = request
        .manifest
        .try_stage(ProjectStage::KeyframeSfm)
        .map_err(|error| RustSfmWorkerError::Manifest(error.to_string()))?;
    let suffix = Path::new(suffix);
    let mut matches = Vec::new();
    for artifact in stage.artifacts() {
        let relative = safe_relative_path(&artifact.relative_path)?;
        if relative.ends_with(suffix) {
            matches.push(request.project_root.join(relative));
        }
    }
    match matches.as_slice() {
        [path] => Ok(path.clone()),
        [] => Err(RustSfmWorkerError::Sfm(format!(
            "missing committed keyframe artifact suffix {}",
            suffix.display()
        ))),
        _ => Err(RustSfmWorkerError::Sfm(format!(
            "duplicate committed keyframe artifact suffix {}",
            suffix.display()
        ))),
    }
}

fn copy_committed_keyframe_artifact(
    request: &StageRequest,
    suffix: &str,
    destination: &Path,
) -> Result<(), RustSfmWorkerError> {
    let source = committed_keyframe_artifact_path(request, suffix)?;
    ensure_regular_project_file(&source)?;
    let parent = destination
        .parent()
        .ok_or_else(|| RustSfmWorkerError::UnsafeFramePath(destination.display().to_string()))?;
    fs::create_dir_all(parent)?;
    fs::copy(source, destination)?;
    ensure_regular_project_file(destination)
}

fn run_remaining_registration_with<F>(
    frames: &[rustsfm::SequenceFrame],
    keyframes: &rustsfm::KeyframeReconstructionResult,
    mapper_config: &rustsfm::MapperConfig,
    registration_config: &rustsfm::SequenceRegistrationConfig,
    output: &Path,
    task: &mut rustsfm::SfmTaskContext<'_>,
    run_remaining: F,
) -> anyhow::Result<rustsfm::SequenceRegistrationResult>
where
    F: FnOnce(
        &[rustsfm::SequenceFrame],
        &[u32],
        &rustsfm::KeyframeReconstructionResult,
        &rustsfm::MapperConfig,
        &rustsfm::SequenceRegistrationConfig,
        &Path,
        &mut rustsfm::SfmTaskContext<'_>,
    ) -> anyhow::Result<rustsfm::SequenceRegistrationResult>,
{
    run_remaining(
        frames,
        &keyframes.keyframe_ids,
        keyframes,
        mapper_config,
        registration_config,
        output,
        task,
    )
}

fn load_imported_sequence(request: &StageRequest) -> Result<ImportedSequence, RustSfmWorkerError> {
    if request.manifest.source.kind != SourceKind::ImageSequence {
        return Err(RustSfmWorkerError::UnsupportedSource);
    }
    let import = request
        .manifest
        .try_stage(ProjectStage::Import)
        .map_err(|error| RustSfmWorkerError::Manifest(error.to_string()))?;
    let metadata = import
        .artifacts()
        .iter()
        .find(|artifact| artifact.relative_path.ends_with("/Cache/frames.json"))
        .ok_or(RustSfmWorkerError::MissingFrameMetadata)?;
    let metadata_path = request
        .project_root
        .join(safe_relative_path(&metadata.relative_path)?);
    ensure_regular_project_file(&metadata_path)?;
    let imported = serde_json::from_slice::<Vec<ImportedFrame>>(&fs::read(metadata_path)?)?;
    if imported.is_empty() {
        return Err(RustSfmWorkerError::MissingFrameMetadata);
    }

    let mut frame_ids = BTreeSet::new();
    let mut frames = Vec::with_capacity(imported.len());
    for frame in imported {
        if !frame_ids.insert(frame.id) {
            return Err(RustSfmWorkerError::Sfm(format!(
                "imported frame metadata contains duplicate id {}",
                frame.id
            )));
        }
        let image_path = request
            .project_root
            .join(safe_relative_path(&frame.normalized_image)?);
        if !image_path.starts_with(&request.project_root) {
            return Err(RustSfmWorkerError::UnsafeFramePath(frame.normalized_image));
        }
        ensure_regular_project_file(&image_path)?;
        frames.push(rustsfm::SequenceFrame {
            id: frame.id,
            image_path,
            timestamp_us: frame.presentation_time_us,
        });
    }
    Ok(ImportedSequence { frames })
}

fn worker_output_directory(request: &StageRequest) -> Result<PathBuf, RustSfmWorkerError> {
    if !request.workspace_path.starts_with(&request.project_root) {
        return Err(RustSfmWorkerError::UnsafeFramePath(
            request.workspace_path.display().to_string(),
        ));
    }
    let output = request.workspace_path.join("rustsfm");
    fs::create_dir_all(&output)?;
    Ok(output)
}

fn progress_sink<'a>(
    control: &'a WorkerControl,
    events: &'a WorkerEventSink,
    sfm_control: &'a rustsfm::SfmTaskControl,
) -> impl FnMut(rustsfm::SfmTaskEvent) + 'a {
    move |event| {
        if control.cancel_requested() {
            sfm_control.request_cancel();
        } else if control.pause_requested() {
            sfm_control.request_pause();
        }
        events.progress(
            event.completed.map(|value| value as u64),
            event.total.map(|value| value as u64),
            PipelineProgressDetail::Sfm {
                operation: progress_operation_label(event.operation),
                image_id: event.image_id,
                pair: event.pair,
                registered_images: event.registered_images,
                sparse_points: event.sparse_points,
            },
        );
    }
}

fn progress_operation_label(operation: rustsfm::SfmTaskOperation) -> String {
    match operation {
        rustsfm::SfmTaskOperation::MatchPairBatch => "Matching image pairs".to_owned(),
        operation => format!("{operation:?}"),
    }
}

fn requested_stop(control: &WorkerControl) -> Option<WorkerOutcome> {
    if control.cancel_requested() {
        Some(WorkerOutcome::Cancelled(Vec::new()))
    } else if control.pause_requested() {
        Some(WorkerOutcome::Paused(Vec::new()))
    } else {
        None
    }
}

fn final_colmap_artifacts(
    request: &StageRequest,
    output: &Path,
    frames: &[rustsfm::SequenceFrame],
    registration: &rustsfm::SequenceRegistrationResult,
) -> Result<Vec<PendingArtifact>, RustSfmWorkerError> {
    let sparse = output.join("sparse/0");
    let mut artifacts = Vec::new();
    for name in ["cameras.txt", "images.txt", "points3D.txt"] {
        let path = sparse.join(name);
        ensure_regular_project_file(&path)?;
        artifacts.push(PendingArtifact::new(
            format!("colmap/sparse/0/{name}"),
            fs::read(path)?,
            ArtifactValidation::ReadableFile,
        ));
    }

    let mut image_names = BTreeSet::new();
    for frame in frames {
        let name = frame
            .image_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                RustSfmWorkerError::UnsafeFramePath(frame.image_path.display().to_string())
            })?;
        if !image_names.insert(name.to_owned()) {
            return Err(RustSfmWorkerError::Sfm(format!(
                "normalized image name is not unique: {name}"
            )));
        }
        ensure_regular_project_file(&frame.image_path)?;
        artifacts.push(PendingArtifact::new(
            format!("colmap/images/{name}"),
            fs::read(&frame.image_path)?,
            ArtifactValidation::ReadableFile,
        ));
    }
    let payload = serde_json::to_vec_pretty(&serde_json::json!({
        "imported_frames": registration.imported_frames,
        "registered_frames": registration.registered_frames,
        "complete": registration.has_complete_coverage(),
        "source_project": request.project_root,
    }))
    .expect("JSON serialization of scalar RustSFM result");
    artifacts.push(PendingArtifact::new(
        "pnp-result.json",
        payload,
        ArtifactValidation::PnpCoverage {
            imported_frames: registration.imported_frames,
            registered_frames: registration.registered_frames,
        },
    ));
    Ok(artifacts)
}

fn outcome_for_registration(
    imported_frames: usize,
    registered_frames: usize,
    artifacts: Vec<PendingArtifact>,
) -> WorkerOutcome {
    if imported_frames != registered_frames {
        return worker_failure(
            ProjectStage::FullFramePnp,
            RustSfmWorkerError::IncompletePoseCoverage {
                imported_frames,
                registered_frames,
            },
        );
    }
    WorkerOutcome::Succeeded(artifacts)
}

fn registration_outcome_with_backend_progress(
    events: &WorkerEventSink,
    registration: &rustsfm::SequenceRegistrationResult,
    gpu_pnp_enabled: bool,
    artifacts: Vec<PendingArtifact>,
) -> WorkerOutcome {
    let outcome = outcome_for_registration(
        registration.imported_frames,
        registration.registered_frames,
        artifacts,
    );
    if matches!(outcome, WorkerOutcome::Succeeded(_)) {
        events.progress(
            Some(registration.registered_frames as u64),
            Some(registration.imported_frames as u64),
            PipelineProgressDetail::Sfm {
                operation: gpu_pnp_focal_backend_progress(registration, gpu_pnp_enabled),
                image_id: None,
                pair: None,
                registered_images: Some(registration.registered_frames),
                sparse_points: None,
            },
        );
    }
    outcome
}

fn gpu_pnp_focal_backend_progress(
    registration: &rustsfm::SequenceRegistrationResult,
    gpu_pnp_enabled: bool,
) -> String {
    if !gpu_pnp_enabled {
        return "CPU PnP configured".to_owned();
    }
    if let Some(reason) = registration.diagnostics.iter().find_map(|diagnostic| {
        diagnostic
            .message
            .as_deref()
            .and_then(|message| message.split_once("gpu_pnp_focal_fallback="))
            .map(|(_, reason)| reason.trim())
            .filter(|reason| !reason.is_empty())
    }) {
        return format!("GPU PnP-focal CPU fallback: {reason}");
    }
    "GPU PnP configured; no focal-solver fallback reported".to_owned()
}

fn worker_failure(stage: ProjectStage, error: RustSfmWorkerError) -> WorkerOutcome {
    WorkerOutcome::Failed(ProjectErrorRecord {
        code: "rustsfm_failed".to_owned(),
        stage,
        summary: "RustSFM pose solve failed".to_owned(),
        detail: error.to_string(),
        frame_id: None,
        pair: None,
        retryable: true,
        suggested_actions: vec![SuggestedAction::Retry, SuggestedAction::OpenLog],
    })
}

fn safe_relative_path(path: &str) -> Result<&Path, RustSfmWorkerError> {
    let path = Path::new(path);
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(RustSfmWorkerError::UnsafeFramePath(
            path.display().to_string(),
        ));
    }
    Ok(path)
}

fn ensure_regular_project_file(path: &Path) -> Result<(), RustSfmWorkerError> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|_| RustSfmWorkerError::MissingFrame(path.to_path_buf()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(RustSfmWorkerError::MissingFrame(path.to_path_buf()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::fs;
    use std::path::{Path, PathBuf};

    use crate::media::{
        import_image_sequence, ImageSequenceImportRequest, MediaEventSink, MediaImportEvent,
    };
    use crate::pipeline::{
        ArtifactValidation, PendingArtifact, PipelineProgressDetail, StageRequest, WorkerOutcome,
    };
    use crate::project::{
        ArtifactRef, KeyframeSelectionMode, ProjectCreateRequest, ProjectStage, ProjectStore,
        SourceSpec, StageState,
    };

    #[derive(Default)]
    struct DiscardEvents;

    impl MediaEventSink for DiscardEvents {
        fn on_media_event(&mut self, _event: MediaImportEvent) {}
    }

    #[test]
    fn imported_frames_resolve_only_inside_the_committed_project_package() {
        let (_temp, request) = fixture_request();

        let frames = super::load_imported_sequence(&request).unwrap().frames;

        assert_eq!(frames.len(), 2);
        assert!(frames.iter().all(|frame| frame.image_path.is_file()));
        assert!(frames
            .iter()
            .all(|frame| frame.image_path.starts_with(&request.project_root)));
    }

    #[test]
    fn project_gpu_configuration_enables_all_rustsfm_gpu_paths() {
        let (_temp, request) = fixture_request();

        let mapper_config = super::mapper_config_for(&request);
        let registration_config = super::registration_config_for(&request);

        assert!(mapper_config.sift_extraction.use_gpu);
        assert!(mapper_config.sift_matching.use_gpu);
        assert!(mapper_config.use_gpu_pnp);
        assert!(mapper_config.local_matching);
        assert!(mapper_config.single_camera);
        assert!(!mapper_config.discover_database);
        assert!(mapper_config.copy_images);
        assert_eq!(mapper_config.max_features, 4096);
        assert_eq!(
            mapper_config.matching_pair_strategy,
            rustsfm::MatchingPairStrategy::LocalWindow { window: 5 }
        );
        assert!(registration_config.use_gpu_pnp);
    }

    #[test]
    fn project_gpu_configuration_can_disable_each_rustsfm_gpu_path() {
        let (_temp, mut request) = fixture_request();
        request.manifest.sfm_config.use_gpu_sift = false;
        request.manifest.sfm_config.use_gpu_matching = false;
        request.manifest.pnp_config.use_gpu_pnp = false;

        let mapper_config = super::mapper_config_for(&request);
        let registration_config = super::registration_config_for(&request);

        assert!(!mapper_config.sift_extraction.use_gpu);
        assert!(!mapper_config.sift_matching.use_gpu);
        assert!(!mapper_config.use_gpu_pnp);
        assert!(!registration_config.use_gpu_pnp);
    }

    #[test]
    fn adaptive_selection_uses_only_rustsfm_selected_ids() {
        let (_temp, request) = fixture_request();
        let sequence = super::load_imported_sequence(&request).unwrap();
        let mapper_config = super::mapper_config_for(&request);
        let output = request.workspace_path.join("selection");
        let control = rustsfm::SfmTaskControl::new();
        let mut sink = |_| {};
        let mut task = rustsfm::SfmTaskContext::new(&control, &mut sink);
        let calls = Cell::new(0);

        let resolved = super::resolve_keyframe_selection_with(
            KeyframeSelectionMode::Adaptive,
            &request.manifest.sfm_config.adaptive_keyframes,
            &sequence.frames,
            &mapper_config,
            &output,
            &mut task,
            |frames, config, mapper, received_output, _task| {
                calls.set(calls.get() + 1);
                assert_eq!(frames, sequence.frames);
                assert_eq!(config, &request.manifest.sfm_config.adaptive_keyframes);
                assert_eq!(mapper.max_features, 4096);
                assert_eq!(received_output, output);
                Ok(adaptive_result(&[frames[0].id, frames[1].id]))
            },
        )
        .unwrap();

        assert_eq!(calls.get(), 1);
        assert_eq!(
            resolved.selected_frame_ids,
            sequence
                .frames
                .iter()
                .map(|frame| frame.id)
                .collect::<Vec<_>>()
        );
        assert!(request.manifest.sfm_config.use_all_images);
    }

    #[test]
    fn all_images_selection_never_invokes_adaptive_selector() {
        let (_temp, request) = fixture_request();
        let sequence = super::load_imported_sequence(&request).unwrap();
        let mapper_config = super::mapper_config_for(&request);
        let control = rustsfm::SfmTaskControl::new();
        let mut sink = |_| {};
        let mut task = rustsfm::SfmTaskContext::new(&control, &mut sink);

        let resolved = super::resolve_keyframe_selection_with(
            KeyframeSelectionMode::AllImages,
            &request.manifest.sfm_config.adaptive_keyframes,
            &sequence.frames,
            &mapper_config,
            &request.workspace_path,
            &mut task,
            |_, _, _, _, _| panic!("all-images mode must not run adaptive selection"),
        )
        .unwrap();

        assert_eq!(
            resolved.selected_frame_ids,
            sequence
                .frames
                .iter()
                .map(|frame| frame.id)
                .collect::<Vec<_>>()
        );
        assert_eq!(resolved.evaluated_pairs, 0);
        assert!(resolved.diagnostics.is_empty());
    }

    #[test]
    fn invalid_adaptive_selection_results_are_rejected() {
        let (_temp, request) = fixture_request();
        let sequence = super::load_imported_sequence(&request).unwrap();
        let mapper_config = super::mapper_config_for(&request);

        for selected in [vec![1], vec![1, 1], vec![2, 1], vec![1, 99]] {
            let control = rustsfm::SfmTaskControl::new();
            let mut sink = |_| {};
            let mut task = rustsfm::SfmTaskContext::new(&control, &mut sink);
            let result = super::resolve_keyframe_selection_with(
                KeyframeSelectionMode::Adaptive,
                &request.manifest.sfm_config.adaptive_keyframes,
                &sequence.frames,
                &mapper_config,
                &request.workspace_path,
                &mut task,
                |_, _, _, _, _| Ok(adaptive_result(&selected)),
            );

            assert!(result.is_err(), "selection {selected:?} must be rejected");
        }
    }

    #[test]
    fn keyframe_stage_artifacts_persist_selection_and_reconstruction() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("rustsfm");
        let database = output.join("Cache/database.db");
        let sparse = output.join("Cache/keyframe-sparse/0");
        fs::create_dir_all(&sparse).unwrap();
        fs::write(&database, b"database").unwrap();
        for name in [
            "cameras.txt",
            "images.txt",
            "points3D.txt",
            "cameras.bin",
            "images.bin",
            "points3D.bin",
        ] {
            fs::write(sparse.join(name), name.as_bytes()).unwrap();
        }
        let selection = super::ResolvedKeyframeSelection {
            selected_frame_ids: vec![1, 3],
            selection_config: Some(rustsfm::AdaptiveKeyframeSelectionConfig::default()),
            evaluated_pairs: 2,
            diagnostics: adaptive_result(&[1, 3]).diagnostics,
        };
        let reconstruction = rustsfm::KeyframeReconstructionResult {
            imported_frames: 3,
            keyframe_ids: vec![1, 3],
            registered_keyframes: 2,
            database,
            sparse_model: sparse,
        };

        let artifacts = super::keyframe_stage_artifacts(
            KeyframeSelectionMode::Adaptive,
            &selection,
            &reconstruction,
        )
        .unwrap();
        let names = artifacts
            .iter()
            .map(|artifact| artifact.relative_path.as_str())
            .collect::<Vec<_>>();

        assert_eq!(
            names,
            vec![
                "keyframe-result.json",
                "rustsfm/database.db",
                "rustsfm/keyframe-sparse/0/cameras.txt",
                "rustsfm/keyframe-sparse/0/images.txt",
                "rustsfm/keyframe-sparse/0/points3D.txt",
                "rustsfm/keyframe-sparse/0/cameras.bin",
                "rustsfm/keyframe-sparse/0/images.bin",
                "rustsfm/keyframe-sparse/0/points3D.bin",
            ]
        );
        let payload: serde_json::Value = serde_json::from_slice(&artifacts[0].payload).unwrap();
        assert_eq!(payload["imported_frames"], 3);
        assert_eq!(payload["selected_keyframe_count"], 2);
        assert_eq!(payload["selected_keyframe_ids"], serde_json::json!([1, 3]));
        assert!(payload["selection_config"].is_object());
        assert_eq!(payload["evaluated_pairs"], 2);
        assert!(payload["diagnostics"].is_array());
    }

    #[test]
    fn hydrate_keyframe_result_copies_committed_database_and_sparse_model() {
        let (_temp, mut request) = fixture_request();
        let stage_result = commit_keyframe_stage_fixture(&mut request);
        let sequence = super::load_imported_sequence(&request).unwrap();
        let output = super::worker_output_directory(&request).unwrap();

        let hydrated = super::hydrate_keyframe_result(&request, &sequence.frames, &output).unwrap();

        assert_eq!(hydrated.keyframe_ids, stage_result.selected_keyframe_ids);
        assert_eq!(hydrated.database, output.join("Cache/database.db"));
        assert_eq!(
            hydrated.sparse_model,
            output.join("Cache/keyframe-sparse/0")
        );
        assert_eq!(fs::read(&hydrated.database).unwrap(), b"committed database");
    }

    #[test]
    fn hydrate_keyframe_result_rejects_missing_and_duplicate_artifact_suffixes() {
        let (_temp, mut missing_request) = fixture_request();
        commit_keyframe_stage_fixture(&mut missing_request);
        missing_request
            .manifest
            .stage_mut(ProjectStage::KeyframeSfm)
            .artifacts
            .retain(|artifact| !artifact.relative_path.ends_with("/cameras.bin"));
        let frames = super::load_imported_sequence(&missing_request)
            .unwrap()
            .frames;
        let output = super::worker_output_directory(&missing_request).unwrap();
        assert!(super::hydrate_keyframe_result(&missing_request, &frames, &output).is_err());

        let (_temp, mut duplicate_request) = fixture_request();
        commit_keyframe_stage_fixture(&mut duplicate_request);
        let duplicate_relative =
            "Artifacts/keyframe_sfm/attempt-00000001/duplicate/rustsfm/database.db".to_owned();
        let duplicate_path = duplicate_request.project_root.join(&duplicate_relative);
        fs::create_dir_all(duplicate_path.parent().unwrap()).unwrap();
        fs::write(&duplicate_path, b"duplicate").unwrap();
        duplicate_request
            .manifest
            .stage_mut(ProjectStage::KeyframeSfm)
            .artifacts
            .push(artifact_ref(duplicate_relative, 9));
        let frames = super::load_imported_sequence(&duplicate_request)
            .unwrap()
            .frames;
        let output = super::worker_output_directory(&duplicate_request).unwrap();
        assert!(super::hydrate_keyframe_result(&duplicate_request, &frames, &output).is_err());
    }

    #[test]
    fn hydrate_keyframe_result_rejects_unsafe_paths_and_sequence_mismatches() {
        let (_temp, mut unsafe_request) = fixture_request();
        commit_keyframe_stage_fixture(&mut unsafe_request);
        unsafe_request
            .manifest
            .stage_mut(ProjectStage::KeyframeSfm)
            .artifacts[0]
            .relative_path = "../keyframe-result.json".to_owned();
        let frames = super::load_imported_sequence(&unsafe_request)
            .unwrap()
            .frames;
        let output = super::worker_output_directory(&unsafe_request).unwrap();
        assert!(super::hydrate_keyframe_result(&unsafe_request, &frames, &output).is_err());

        for (imported_frames, selected_ids) in [(3, vec![0, 1]), (2, vec![0, 99])] {
            let (_temp, mut request) = fixture_request();
            let mut result = commit_keyframe_stage_fixture(&mut request);
            result.imported_frames = imported_frames;
            result.selected_keyframe_ids = selected_ids;
            result.selected_keyframe_count = result.selected_keyframe_ids.len();
            overwrite_keyframe_stage_result(&request, &result);
            let frames = super::load_imported_sequence(&request).unwrap().frames;
            let output = super::worker_output_directory(&request).unwrap();
            assert!(super::hydrate_keyframe_result(&request, &frames, &output).is_err());
        }
    }

    #[test]
    fn remaining_registration_boundary_receives_hydrated_keyframes_and_all_frames() {
        let (_temp, request) = fixture_request();
        let sequence = super::load_imported_sequence(&request).unwrap();
        let mapper = super::mapper_config_for(&request);
        let registration = super::registration_config_for(&request);
        let output = request.workspace_path.join("remaining");
        let selected = sequence
            .frames
            .iter()
            .map(|frame| frame.id)
            .collect::<Vec<_>>();
        let keyframes = rustsfm::KeyframeReconstructionResult {
            imported_frames: sequence.frames.len(),
            keyframe_ids: selected.clone(),
            registered_keyframes: selected.len(),
            database: output.join("Cache/database.db"),
            sparse_model: output.join("Cache/keyframe-sparse/0"),
        };
        let control = rustsfm::SfmTaskControl::new();
        let mut sink = |_| {};
        let mut task = rustsfm::SfmTaskContext::new(&control, &mut sink);
        let called = Cell::new(false);

        let result = super::run_remaining_registration_with(
            &sequence.frames,
            &keyframes,
            &mapper,
            &registration,
            &output,
            &mut task,
            |frames, ids, received_keyframes, _, _, received_output, _| {
                called.set(true);
                assert_eq!(frames, sequence.frames);
                assert_eq!(ids, selected);
                assert_eq!(received_keyframes, &keyframes);
                assert_eq!(received_output, output);
                Ok(rustsfm::SequenceRegistrationResult {
                    imported_frames: frames.len(),
                    registered_frames: frames.len(),
                    frame_ids: ids.to_vec(),
                    diagnostics: Vec::new(),
                    sparse_model: output.join("sparse/0"),
                })
            },
        )
        .unwrap();

        assert!(called.get());
        assert_eq!(result.registered_frames, sequence.frames.len());
    }

    #[test]
    fn imported_keyframe_flags_do_not_filter_the_stable_sequence() {
        let (_temp, mut request) = fixture_request();
        request.manifest.sfm_config.use_all_images = false;
        let metadata = request
            .project_root
            .join("Artifacts/import/attempt-00000001/Cache/frames.json");
        let mut frames = serde_json::from_slice::<Vec<crate::media::ImportedFrame>>(
            &fs::read(&metadata).unwrap(),
        )
        .unwrap();
        for frame in &mut frames {
            frame.is_keyframe = false;
        }
        fs::write(&metadata, serde_json::to_vec(&frames).unwrap()).unwrap();

        let sequence = super::load_imported_sequence(&request).unwrap();

        assert_eq!(
            sequence
                .frames
                .iter()
                .map(|frame| frame.id)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn duplicate_imported_frame_ids_are_rejected() {
        let (_temp, request) = fixture_request();
        let metadata = request
            .project_root
            .join("Artifacts/import/attempt-00000001/Cache/frames.json");
        let mut frames = serde_json::from_slice::<Vec<crate::media::ImportedFrame>>(
            &fs::read(&metadata).unwrap(),
        )
        .unwrap();
        frames[1].id = frames[0].id;
        fs::write(&metadata, serde_json::to_vec(&frames).unwrap()).unwrap();

        let error = match super::load_imported_sequence(&request) {
            Ok(_) => panic!("duplicate frame ids must be rejected"),
            Err(error) => error,
        };

        assert!(
            matches!(error, super::RustSfmWorkerError::Sfm(detail) if detail.contains("duplicate id"))
        );
    }

    #[test]
    fn parent_components_in_imported_frame_paths_are_rejected() {
        let (_temp, request) = fixture_request();
        let metadata = request
            .project_root
            .join("Artifacts/import/attempt-00000001/Cache/frames.json");
        let mut frames = serde_json::from_slice::<Vec<crate::media::ImportedFrame>>(
            &fs::read(&metadata).unwrap(),
        )
        .unwrap();
        frames[0].normalized_image = "../outside.png".to_owned();
        fs::write(&metadata, serde_json::to_vec(&frames).unwrap()).unwrap();

        let error = match super::load_imported_sequence(&request) {
            Ok(_) => panic!("parent components must be rejected"),
            Err(error) => error,
        };

        assert!(matches!(
            error,
            super::RustSfmWorkerError::UnsafeFramePath(_)
        ));
    }

    #[test]
    fn incomplete_full_frame_registration_never_returns_success() {
        let outcome = super::outcome_for_registration(2, 1, Vec::new());

        assert!(matches!(outcome, WorkerOutcome::Failed(_)));
    }

    #[test]
    fn gpu_pnp_focal_backend_progress_distinguishes_cpu_fallback_reason() {
        let mut diagnostic = rustsfm::FrameRegistrationDiagnostic::new(
            7,
            rustsfm::FrameRegistrationStatus::Registered,
        );
        diagnostic.message = Some(
            "registered in Narrow round; gpu_pnp_focal_fallback=gpu dispatch failed".to_owned(),
        );
        let mut registration = rustsfm::SequenceRegistrationResult {
            imported_frames: 2,
            registered_frames: 2,
            frame_ids: vec![0, 7],
            diagnostics: vec![
                rustsfm::FrameRegistrationDiagnostic::new(
                    0,
                    rustsfm::FrameRegistrationStatus::Keyframe,
                ),
                diagnostic,
            ],
            sparse_model: std::path::PathBuf::from("fixture/sparse/0"),
        };

        assert_eq!(
            super::gpu_pnp_focal_backend_progress(&registration, true),
            "GPU PnP-focal CPU fallback: gpu dispatch failed"
        );
        assert_eq!(
            super::gpu_pnp_focal_backend_progress(&registration, false),
            "CPU PnP configured"
        );
        registration.diagnostics[1].message = None;
        assert_eq!(
            super::gpu_pnp_focal_backend_progress(&registration, true),
            "GPU PnP configured; no focal-solver fallback reported"
        );
    }

    #[test]
    fn match_pair_batch_progress_uses_user_facing_label() {
        let control = crate::pipeline::WorkerControl::new();
        let sfm_control = rustsfm::SfmTaskControl::new();
        let (sender, receiver) = crossbeam_channel::bounded(1);
        let events = crate::pipeline::WorkerEventSink::new(ProjectStage::KeyframeSfm, 3, sender);
        let mut sink = super::progress_sink(&control, &events, &sfm_control);

        sink(rustsfm::SfmTaskEvent {
            sequence: 11,
            elapsed_ms: 240,
            stage: rustsfm::SfmTaskStage::FeatureMatching,
            operation: rustsfm::SfmTaskOperation::MatchPairBatch,
            kind: rustsfm::SfmTaskEventKind::Progress,
            completed: Some(32),
            total: Some(100),
            registered_images: None,
            sparse_points: None,
            image_id: None,
            pair: Some((7, 8)),
            message: Some("ignored event message".to_owned()),
            issue: None,
        });

        assert!(matches!(
            receiver.recv().unwrap(),
            crate::pipeline::PipelineEvent::StageProgress {
                stage: ProjectStage::KeyframeSfm,
                attempt: 3,
                completed: Some(32),
                total: Some(100),
                detail: PipelineProgressDetail::Sfm { operation, pair, .. },
            } if operation == "Matching image pairs" && pair == Some((7, 8))
        ));
    }

    #[test]
    fn backend_completion_progress_is_emitted_only_for_successful_registration() {
        let registration = rustsfm::SequenceRegistrationResult {
            imported_frames: 2,
            registered_frames: 2,
            frame_ids: vec![0, 1],
            diagnostics: Vec::new(),
            sparse_model: std::path::PathBuf::from("fixture/sparse/0"),
        };
        let (sender, receiver) = crossbeam_channel::bounded(1);
        let events = crate::pipeline::WorkerEventSink::new(ProjectStage::FullFramePnp, 1, sender);

        let outcome = super::registration_outcome_with_backend_progress(
            &events,
            &registration,
            true,
            vec![PendingArtifact::new(
                "result.json",
                br#"{}"#.to_vec(),
                ArtifactValidation::Json,
            )],
        );

        assert!(matches!(outcome, WorkerOutcome::Succeeded(_)));
        assert!(matches!(
            receiver.recv().unwrap(),
            crate::pipeline::PipelineEvent::StageProgress {
                stage: ProjectStage::FullFramePnp,
                attempt: 1,
                completed: Some(2),
                total: Some(2),
                detail: PipelineProgressDetail::Sfm { operation, .. },
            } if operation == "GPU PnP configured; no focal-solver fallback reported"
        ));

        let incomplete = rustsfm::SequenceRegistrationResult {
            registered_frames: 1,
            ..registration
        };
        let outcome = super::registration_outcome_with_backend_progress(
            &events,
            &incomplete,
            true,
            Vec::new(),
        );

        assert!(matches!(outcome, WorkerOutcome::Failed(_)));
        assert!(receiver.try_recv().is_err());
    }

    fn adaptive_result(selected_frame_ids: &[u32]) -> rustsfm::AdaptiveKeyframeSelectionResult {
        rustsfm::AdaptiveKeyframeSelectionResult {
            imported_frames: 2,
            usable_frames: 2,
            selected_frame_ids: selected_frame_ids.to_vec(),
            config: rustsfm::AdaptiveKeyframeSelectionConfig::default(),
            evaluated_pairs: 1,
            diagnostics: vec![rustsfm::AdaptiveKeyframePairDiagnostic {
                metrics: rustsfm::AdaptiveKeyframePairMetrics {
                    anchor_frame_id: 1,
                    candidate_frame_id: 2,
                    descriptor_matches: 40,
                    inliers: 30,
                    triangulated: 20,
                    inlier_ratio: 0.75,
                    feature_coverage: 0.5,
                },
                decision: rustsfm::AdaptiveKeyframeSelectionDecision::Boundary,
            }],
        }
    }

    fn commit_keyframe_stage_fixture(request: &mut StageRequest) -> super::KeyframeStageResult {
        let selected_keyframe_ids = vec![0, 1];
        let result = super::KeyframeStageResult {
            mode: KeyframeSelectionMode::Adaptive,
            imported_frames: 2,
            selected_keyframe_count: selected_keyframe_ids.len(),
            selected_keyframe_ids,
            registered_keyframes: 2,
            selection_config: Some(rustsfm::AdaptiveKeyframeSelectionConfig::default()),
            evaluated_pairs: 1,
            diagnostics: adaptive_result(&[0, 1]).diagnostics,
        };
        let root = "Artifacts/keyframe_sfm/attempt-00000001";
        let files = [
            ("keyframe-result.json", serde_json::to_vec(&result).unwrap()),
            ("rustsfm/database.db", b"committed database".to_vec()),
            (
                "rustsfm/keyframe-sparse/0/cameras.txt",
                b"cameras text".to_vec(),
            ),
            (
                "rustsfm/keyframe-sparse/0/images.txt",
                b"images text".to_vec(),
            ),
            (
                "rustsfm/keyframe-sparse/0/points3D.txt",
                b"points text".to_vec(),
            ),
            (
                "rustsfm/keyframe-sparse/0/cameras.bin",
                b"cameras binary".to_vec(),
            ),
            (
                "rustsfm/keyframe-sparse/0/images.bin",
                b"images binary".to_vec(),
            ),
            (
                "rustsfm/keyframe-sparse/0/points3D.bin",
                b"points binary".to_vec(),
            ),
        ];
        let mut artifacts = Vec::new();
        for (suffix, payload) in files {
            let relative = format!("{root}/{suffix}");
            let path = request.project_root.join(&relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, &payload).unwrap();
            artifacts.push(artifact_ref(relative, payload.len()));
        }
        let stage = request.manifest.stage_mut(ProjectStage::KeyframeSfm);
        stage.state = StageState::Succeeded;
        stage.artifacts = artifacts;
        result
    }

    fn overwrite_keyframe_stage_result(
        request: &StageRequest,
        result: &super::KeyframeStageResult,
    ) {
        let artifact = request
            .manifest
            .try_stage(ProjectStage::KeyframeSfm)
            .unwrap()
            .artifacts()
            .iter()
            .find(|artifact| artifact.relative_path.ends_with("/keyframe-result.json"))
            .unwrap();
        fs::write(
            request.project_root.join(&artifact.relative_path),
            serde_json::to_vec(result).unwrap(),
        )
        .unwrap();
    }

    fn artifact_ref(relative_path: String, byte_len: usize) -> ArtifactRef {
        ArtifactRef {
            relative_path,
            content_hash: "0".repeat(64),
            byte_len: byte_len as u64,
        }
    }

    fn fixture_request() -> (tempfile::TempDir, StageRequest) {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        fs::create_dir(&source).unwrap();
        let first = write_fixture_image(&source, "frame-001.png");
        let second = write_fixture_image(&source, "frame-002.png");
        let project_path = temp.path().join("Sequence.rustscanproject");
        let mut store = ProjectStore::create(
            &project_path,
            ProjectCreateRequest::new("Sequence", SourceSpec::managed_images("pending-import")),
        )
        .unwrap();
        import_image_sequence(
            &ImageSequenceImportRequest::managed(vec![first, second]),
            &mut store,
            &mut DiscardEvents,
        )
        .unwrap();

        let request = StageRequest {
            stage: ProjectStage::KeyframeSfm,
            attempt: 1,
            project_root: store.root().to_path_buf(),
            workspace_path: store.root().join("Cache/.staging/keyframe_sfm-1"),
            manifest: store.manifest().clone(),
        };

        (temp, request)
    }

    fn write_fixture_image(directory: &Path, name: &str) -> PathBuf {
        let path = directory.join(name);
        image::RgbaImage::from_pixel(2, 2, image::Rgba([12, 34, 56, 255]))
            .save(&path)
            .unwrap();
        path
    }
}
