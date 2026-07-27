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
    KeyframeSelectionMode, ProjectErrorRecord, ProjectStage, SourceKind, SuggestedAction,
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
        if sequence.frames.len() < 2 {
            return worker_failure(
                ProjectStage::FullFramePnp,
                RustSfmWorkerError::Sfm(
                    "at least two keyframes are required for reconstruction".to_owned(),
                ),
            );
        }
        let output = match worker_output_directory(&request) {
            Ok(output) => output,
            Err(error) => return worker_failure(ProjectStage::FullFramePnp, error),
        };
        let mapper_config = mapper_config_for(&request);
        let registration_config = registration_config_for(&request);

        let sfm_control = rustsfm::SfmTaskControl::new();
        let mut task_sink = progress_sink(&control, &events, &sfm_control);
        let mut task = rustsfm::SfmTaskContext::new(&sfm_control, &mut task_sink);
        let keyframe_ids = sequence
            .frames
            .iter()
            .map(|frame| frame.id)
            .collect::<Vec<_>>();
        let result = rustsfm::run_sequence_registration(
            &sequence.frames,
            &keyframe_ids,
            &mapper_config,
            &registration_config,
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
        outcome_for_registration(result.imported_frames, result.registered_frames, artifacts)
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
                operation: format!("{:?}", event.operation),
                image_id: event.image_id,
                pair: event.pair,
                registered_images: event.registered_images,
                sparse_points: event.sparse_points,
            },
        );
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
    use crate::pipeline::{StageRequest, WorkerOutcome};
    use crate::project::{
        KeyframeSelectionMode, ProjectCreateRequest, ProjectStage, ProjectStore, SourceSpec,
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
