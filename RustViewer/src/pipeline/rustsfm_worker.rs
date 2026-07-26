use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path, PathBuf};

use thiserror::Error;

use crate::media::ImportedFrame;
use crate::pipeline::{
    ArtifactValidation, PendingArtifact, PipelineProgressDetail, PnpWorker, SfmWorker,
    StageRequest, WorkerControl, WorkerEventSink, WorkerOutcome,
};
use crate::project::{ProjectErrorRecord, ProjectStage, SourceKind, SuggestedAction};

#[derive(Debug, Default, Clone, Copy)]
pub struct RustSfmWorker;

struct ImportedSequence {
    frames: Vec<rustsfm::SequenceFrame>,
    keyframe_ids: Vec<u32>,
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
        if sequence.keyframe_ids.len() < 2 {
            return worker_failure(
                ProjectStage::KeyframeSfm,
                RustSfmWorkerError::Sfm(
                    "at least two keyframes are required for reconstruction".to_owned(),
                ),
            );
        }
        let output = match worker_output_directory(&request) {
            Ok(output) => output,
            Err(error) => return worker_failure(ProjectStage::KeyframeSfm, error),
        };
        let mapper_config = mapper_config_for(&request);

        let sfm_control = rustsfm::SfmTaskControl::new();
        let mut task_sink = progress_sink(&control, &events, &sfm_control);
        let mut task = rustsfm::SfmTaskContext::new(&sfm_control, &mut task_sink);
        let result = rustsfm::run_keyframe_reconstruction(
            &sequence.frames,
            &sequence.keyframe_ids,
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
        let payload = serde_json::to_vec_pretty(&serde_json::json!({
            "imported_frames": result.imported_frames,
            "registered_keyframes": result.registered_keyframes,
            "keyframe_ids": result.keyframe_ids,
        }))
        .expect("JSON serialization of scalar RustSFM result");
        if let Err(error) = fs::remove_dir_all(&output) {
            return worker_failure(ProjectStage::KeyframeSfm, RustSfmWorkerError::Io(error));
        }
        WorkerOutcome::Succeeded(vec![PendingArtifact::new(
            "keyframe-result.json",
            payload,
            ArtifactValidation::Json,
        )])
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
        if sequence.keyframe_ids.len() < 2 {
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
        let result = rustsfm::run_sequence_registration(
            &sequence.frames,
            &sequence.keyframe_ids,
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
    let mut keyframe_ids = Vec::new();
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
        if request.manifest.sfm_config.use_all_images || frame.is_keyframe {
            keyframe_ids.push(frame.id);
        }
        frames.push(rustsfm::SequenceFrame {
            id: frame.id,
            image_path,
            timestamp_us: frame.presentation_time_us,
        });
    }
    if keyframe_ids.is_empty() {
        return Err(RustSfmWorkerError::Sfm(
            "the imported project has no selected keyframes".to_owned(),
        ));
    }
    Ok(ImportedSequence {
        frames,
        keyframe_ids,
    })
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
    use std::fs;
    use std::path::{Path, PathBuf};

    use crate::media::{
        import_image_sequence, ImageSequenceImportRequest, MediaEventSink, MediaImportEvent,
    };
    use crate::pipeline::{StageRequest, WorkerOutcome};
    use crate::project::{ProjectCreateRequest, ProjectStage, ProjectStore, SourceSpec};

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
    fn empty_keyframe_selection_is_rejected_when_all_images_are_disabled() {
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

        let error = match super::load_imported_sequence(&request) {
            Ok(_) => panic!("an empty keyframe selection must be rejected"),
            Err(error) => error,
        };

        assert!(
            matches!(error, super::RustSfmWorkerError::Sfm(detail) if detail.contains("keyframes"))
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
