use std::cmp::Ordering;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, ImageDecoder, ImageFormat, ImageReader};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::project::artifacts::{ArtifactValidationKind, StagedArtifact};
use crate::project::source::{
    image_sequence_identity, ImportedSourceFrame, ImportedSourceRecord, SourceBookmark,
};
use crate::project::{
    ProjectErrorRecord, ProjectStage, ProjectStore, ProjectStoreError, SourceKind, SourceOwnership,
    SourceSpec, SuggestedAction,
};

const SOURCE_METADATA_PAYLOAD: &str = "Sources/source.json";
const FRAMES_METADATA_PAYLOAD: &str = "Cache/frames.json";
const KEYFRAMES_METADATA_PAYLOAD: &str = "Cache/keyframes.json";
const MANAGED_SOURCE_DIRECTORY: &str = "Sources/managed";
const FRAME_DIRECTORY: &str = "Cache/frames";
const THUMBNAIL_DIRECTORY: &str = "Cache/thumbnails";
const THUMBNAIL_LONG_EDGE: u32 = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageSequenceImportRequest {
    pub paths: Vec<PathBuf>,
    pub ownership: SourceOwnership,
}

impl ImageSequenceImportRequest {
    pub fn managed(paths: Vec<PathBuf>) -> Self {
        Self {
            paths,
            ownership: SourceOwnership::ManagedCopy,
        }
    }

    pub fn referenced(paths: Vec<PathBuf>) -> Self {
        Self {
            paths,
            ownership: SourceOwnership::Referenced,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportedFrame {
    pub id: u32,
    pub source_name: String,
    pub presentation_time_us: Option<i64>,
    pub normalized_image: String,
    pub thumbnail: String,
    pub width: u32,
    pub height: u32,
    pub sharpness: f64,
    pub perceptual_hash: u64,
    pub is_keyframe: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportResult {
    pub source_identity: String,
    pub frames: Vec<ImportedFrame>,
    pub duration_us: Option<i64>,
    pub source_metadata: String,
    pub frames_metadata: String,
    pub keyframes_metadata: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaImportEvent {
    Started {
        total: Option<usize>,
    },
    FrameCommitted {
        frame_id: u32,
        completed: usize,
        total: Option<usize>,
    },
    Completed {
        frame_count: usize,
        keyframe_count: usize,
    },
}

pub trait MediaEventSink {
    fn on_media_event(&mut self, event: MediaImportEvent);
}

#[derive(Debug, Error)]
pub enum MediaImportError {
    #[error("invalid image-sequence source: {0}")]
    InvalidSource(String),
    #[error("image decode failed: {0}")]
    Decode(String),
    #[error("image sequence changed during import: expected {expected}, found {found}")]
    SourceChangedDuringImport { expected: String, found: String },
    #[error("media import I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("image conversion failed: {0}")]
    Image(#[from] image::ImageError),
    #[error("project storage failed: {0}")]
    Project(#[from] ProjectStoreError),
    #[error("image import failed ({import}) and its failed stage could not be persisted: {state}")]
    FailedStagePersistence { import: String, state: String },
    #[error("image import commit failed ({commit}) and recovery could not complete: {recovery}")]
    CommitRecoveryRequired { commit: String, recovery: String },
}

pub fn import_image_sequence(
    request: &ImageSequenceImportRequest,
    store: &mut ProjectStore,
    sink: &mut impl MediaEventSink,
) -> Result<ImportResult, MediaImportError> {
    let sources = match collect_image_sources(request) {
        Ok(sources) => sources,
        Err(error) => return Err(persist_preflight_import_failure(store, error)),
    };
    let identity = match identity_for_sources(&sources) {
        Ok(identity) => identity,
        Err(error) => return Err(persist_preflight_import_failure(store, error)),
    };
    let display_paths = sources
        .iter()
        .map(|source| source.canonical_path.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let bookmark = if request.ownership == SourceOwnership::Referenced {
        Some(
            SourceBookmark::from_canonical_paths(display_paths.clone())
                .map_err(|error| MediaImportError::InvalidSource(error.to_string()))?
                .encode()
                .map_err(|error| {
                    MediaImportError::InvalidSource(format!(
                        "failed to encode source bookmark: {error}"
                    ))
                })?,
        )
    } else {
        None
    };
    store.update_source(SourceSpec {
        kind: SourceKind::ImageSequence,
        ownership: request.ownership,
        identity: identity.clone(),
        display_paths,
        bookmark,
    })?;

    sink.on_media_event(MediaImportEvent::Started {
        total: Some(sources.len()),
    });
    let workspace = store.begin_stage(ProjectStage::Import)?;
    let import = import_into_workspace(&sources, request.ownership, &identity, &workspace, sink);
    match import {
        Ok((result, declarations, payloads)) => {
            if let Err(error) = store
                .validate_stage_payloads(&workspace, &payloads)
                .map_err(MediaImportError::Project)
            {
                return Err(persist_import_failure(store, error, |store, record| {
                    store.mark_stage_failed(ProjectStage::Import, record)
                }));
            }
            commit_import_or_recover(store, |store| {
                store.commit_stage_success(&workspace, &declarations, false)
            })?;
            sink.on_media_event(MediaImportEvent::Completed {
                frame_count: result.frames.len(),
                keyframe_count: result.frames.len(),
            });
            Ok(result)
        }
        Err(error) => Err(persist_import_failure(store, error, |store, record| {
            store.mark_stage_failed(ProjectStage::Import, record)
        })),
    }
}

fn commit_import_or_recover(
    store: &mut ProjectStore,
    commit: impl FnOnce(&mut ProjectStore) -> Result<(), ProjectStoreError>,
) -> Result<(), MediaImportError> {
    match commit(store) {
        Ok(()) => Ok(()),
        Err(commit) => match store.recover_interrupted_stage() {
            Ok(()) => Err(MediaImportError::Project(commit)),
            Err(recovery) => Err(MediaImportError::CommitRecoveryRequired {
                commit: commit.to_string(),
                recovery: recovery.to_string(),
            }),
        },
    }
}

struct ImageSource {
    input_path: PathBuf,
    canonical_path: PathBuf,
    source_name: String,
}

fn collect_image_sources(
    request: &ImageSequenceImportRequest,
) -> Result<Vec<ImageSource>, MediaImportError> {
    if request.paths.len() < 2 {
        return Err(MediaImportError::InvalidSource(
            "at least two readable image files are required".to_owned(),
        ));
    }
    let mut sources = request
        .paths
        .iter()
        .map(|input_path| {
            let extension = input_path
                .extension()
                .and_then(|extension| extension.to_str())
                .map(str::to_ascii_lowercase)
                .ok_or_else(|| {
                    MediaImportError::InvalidSource(format!(
                        "unsupported image file: {}",
                        input_path.display()
                    ))
                })?;
            if !matches!(
                extension.as_str(),
                "jpg" | "jpeg" | "png" | "bmp" | "tif" | "tiff" | "webp"
            ) {
                return Err(MediaImportError::InvalidSource(format!(
                    "unsupported image file: {}",
                    input_path.display()
                )));
            }
            let canonical_path = fs::canonicalize(input_path).map_err(|error| {
                MediaImportError::InvalidSource(format!("{}: {error}", input_path.display()))
            })?;
            if !canonical_path.is_file() {
                return Err(MediaImportError::InvalidSource(format!(
                    "not a regular image file: {}",
                    input_path.display()
                )));
            }
            let source_name = canonical_path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| {
                    MediaImportError::InvalidSource(format!(
                        "image name is not valid UTF-8: {}",
                        input_path.display()
                    ))
                })?
                .to_owned();
            Ok(ImageSource {
                input_path: input_path.clone(),
                canonical_path,
                source_name,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    sources.sort_by(|left, right| {
        natural_compare(&left.source_name, &right.source_name)
            .then_with(|| left.canonical_path.cmp(&right.canonical_path))
    });
    Ok(sources)
}

fn import_into_workspace(
    sources: &[ImageSource],
    ownership: SourceOwnership,
    identity: &str,
    workspace: &crate::project::artifacts::StageWorkspace,
    sink: &mut impl MediaEventSink,
) -> Result<(ImportResult, Vec<StagedArtifact>, Vec<StagedArtifact>), MediaImportError> {
    let final_prefix = format!("Artifacts/import/attempt-{:08}", workspace.attempt());
    let mut frames = Vec::with_capacity(sources.len());
    let mut source_frames = Vec::with_capacity(sources.len());
    let mut declarations = Vec::new();
    let mut payloads = Vec::new();

    for (index, source) in sources.iter().enumerate() {
        let id = u32::try_from(index).map_err(|_| {
            MediaImportError::InvalidSource("image sequence contains too many frames".to_owned())
        })?;
        let mut decoder = ImageReader::open(&source.canonical_path)
            .map_err(|error| {
                MediaImportError::Decode(format!("{}: {error}", source.input_path.display()))
            })?
            .into_decoder()
            .map_err(|error| {
                MediaImportError::Decode(format!("{}: {error}", source.input_path.display()))
            })?;
        let orientation = decoder.orientation().map_err(|error| {
            MediaImportError::Decode(format!("{}: {error}", source.input_path.display()))
        })?;
        let mut decoded = DynamicImage::from_decoder(decoder).map_err(|error| {
            MediaImportError::Decode(format!("{}: {error}", source.input_path.display()))
        })?;
        decoded.apply_orientation(orientation);
        let (width, height) = decoded.dimensions();
        let frame_payload = format!("{FRAME_DIRECTORY}/{id:08}.png");
        let thumbnail_payload = format!("{THUMBNAIL_DIRECTORY}/{id:08}.jpg");
        write_png(&workspace.path().join(&frame_payload), &decoded)?;
        write_thumbnail(&workspace.path().join(&thumbnail_payload), &decoded)?;
        payloads.push(readable_payload(&frame_payload)?);
        payloads.push(readable_payload(&thumbnail_payload)?);
        let managed_copy = if ownership == SourceOwnership::ManagedCopy {
            let extension = source
                .canonical_path
                .extension()
                .and_then(|extension| extension.to_str())
                .expect("validated source extension");
            let managed_payload = format!("{MANAGED_SOURCE_DIRECTORY}/{id:08}.{extension}");
            copy_regular_file(
                &source.canonical_path,
                &workspace.path().join(&managed_payload),
            )?;
            payloads.push(readable_payload(&managed_payload)?);
            Some(format!("{final_prefix}/{managed_payload}"))
        } else {
            None
        };
        frames.push(ImportedFrame {
            id,
            source_name: source.source_name.clone(),
            presentation_time_us: None,
            normalized_image: format!("{final_prefix}/{frame_payload}"),
            thumbnail: format!("{final_prefix}/{thumbnail_payload}"),
            width,
            height,
            sharpness: sharpness(&decoded),
            perceptual_hash: perceptual_hash(&decoded),
            is_keyframe: true,
        });
        source_frames.push(ImportedSourceFrame {
            id,
            source_name: source.source_name.clone(),
            original_path: source.canonical_path.to_string_lossy().into_owned(),
            managed_copy,
        });
        sink.on_media_event(MediaImportEvent::FrameCommitted {
            frame_id: id,
            completed: index + 1,
            total: Some(sources.len()),
        });
    }

    let final_identity = identity_for_sources(sources)?;
    if final_identity != identity {
        return Err(MediaImportError::SourceChangedDuringImport {
            expected: identity.to_owned(),
            found: final_identity,
        });
    }

    let source_metadata = format!("{final_prefix}/{SOURCE_METADATA_PAYLOAD}");
    let frames_metadata = format!("{final_prefix}/{FRAMES_METADATA_PAYLOAD}");
    let keyframes_metadata = format!("{final_prefix}/{KEYFRAMES_METADATA_PAYLOAD}");
    let source_record = ImportedSourceRecord {
        schema_version: 1,
        ownership,
        identity: identity.to_owned(),
        frames: source_frames,
    };
    write_json(
        &workspace.path().join(SOURCE_METADATA_PAYLOAD),
        &source_record,
    )?;
    write_json(&workspace.path().join(FRAMES_METADATA_PAYLOAD), &frames)?;
    write_json(
        &workspace.path().join(KEYFRAMES_METADATA_PAYLOAD),
        &frames.iter().map(|frame| frame.id).collect::<Vec<_>>(),
    )?;
    for payload in [
        SOURCE_METADATA_PAYLOAD,
        FRAMES_METADATA_PAYLOAD,
        KEYFRAMES_METADATA_PAYLOAD,
    ] {
        let declaration = StagedArtifact::new(payload, ArtifactValidationKind::Json)
            .map_err(|error| MediaImportError::InvalidSource(error.to_string()))?;
        payloads.push(declaration.clone());
        declarations.push(declaration);
    }
    Ok((
        ImportResult {
            source_identity: identity.to_owned(),
            frames,
            duration_us: None,
            source_metadata,
            frames_metadata,
            keyframes_metadata,
        },
        declarations,
        payloads,
    ))
}

fn persist_preflight_import_failure(
    store: &mut ProjectStore,
    error: MediaImportError,
) -> MediaImportError {
    persist_import_failure(store, error, |store, record| {
        store.mark_stage_preflight_failed(ProjectStage::Import, record)
    })
}

fn persist_import_failure(
    store: &mut ProjectStore,
    error: MediaImportError,
    mark_failed: impl FnOnce(&mut ProjectStore, ProjectErrorRecord) -> Result<(), ProjectStoreError>,
) -> MediaImportError {
    let detail = error.to_string();
    let record = ProjectErrorRecord {
        code: "media_import_failed".to_owned(),
        stage: ProjectStage::Import,
        summary: "Image sequence import failed".to_owned(),
        detail: detail.clone(),
        frame_id: None,
        pair: None,
        retryable: true,
        suggested_actions: vec![SuggestedAction::Retry, SuggestedAction::RevealSource],
    };
    match mark_failed(store, record) {
        Ok(()) => error,
        Err(state) => MediaImportError::FailedStagePersistence {
            import: detail,
            state: state.to_string(),
        },
    }
}

fn identity_for_sources(sources: &[ImageSource]) -> Result<String, MediaImportError> {
    image_sequence_identity(
        &sources
            .iter()
            .map(|source| (source.source_name.clone(), source.canonical_path.clone()))
            .collect::<Vec<_>>(),
    )
    .map_err(MediaImportError::Io)
}

fn readable_payload(payload: &str) -> Result<StagedArtifact, MediaImportError> {
    StagedArtifact::new(payload, ArtifactValidationKind::ReadableFile)
        .map_err(|error| MediaImportError::InvalidSource(error.to_string()))
}

fn write_png(path: &Path, image: &DynamicImage) -> Result<(), MediaImportError> {
    create_parent(path)?;
    let mut output = File::create(path)?;
    image.write_to(&mut output, ImageFormat::Png)?;
    Ok(())
}

fn write_thumbnail(path: &Path, image: &DynamicImage) -> Result<(), MediaImportError> {
    create_parent(path)?;
    let (width, height) = image.dimensions();
    let scale = THUMBNAIL_LONG_EDGE as f64 / width.max(height) as f64;
    let thumbnail = if scale < 1.0 {
        image.resize(
            (width as f64 * scale).round().max(1.0) as u32,
            (height as f64 * scale).round().max(1.0) as u32,
            FilterType::Lanczos3,
        )
    } else {
        image.clone()
    };
    let mut output = File::create(path)?;
    JpegEncoder::new_with_quality(&mut output, 88).encode_image(&thumbnail)?;
    Ok(())
}

fn copy_regular_file(source: &Path, destination: &Path) -> Result<(), MediaImportError> {
    create_parent(destination)?;
    fs::copy(source, destination)?;
    Ok(())
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), MediaImportError> {
    create_parent(path)?;
    fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|error| {
            MediaImportError::InvalidSource(format!("failed to serialize import metadata: {error}"))
        })?,
    )?;
    Ok(())
}

fn create_parent(path: &Path) -> Result<(), MediaImportError> {
    let parent = path.parent().ok_or_else(|| {
        MediaImportError::InvalidSource(format!("payload has no parent: {}", path.display()))
    })?;
    fs::create_dir_all(parent)?;
    Ok(())
}

fn natural_compare(left: &str, right: &str) -> Ordering {
    let left = left.as_bytes();
    let right = right.as_bytes();
    let mut left_index = 0;
    let mut right_index = 0;
    while left_index < left.len() && right_index < right.len() {
        if left[left_index].is_ascii_digit() && right[right_index].is_ascii_digit() {
            let left_start = left_index;
            let right_start = right_index;
            while left_index < left.len() && left[left_index].is_ascii_digit() {
                left_index += 1;
            }
            while right_index < right.len() && right[right_index].is_ascii_digit() {
                right_index += 1;
            }
            let left_digits = trim_leading_zeros(&left[left_start..left_index]);
            let right_digits = trim_leading_zeros(&right[right_start..right_index]);
            let order = left_digits
                .len()
                .cmp(&right_digits.len())
                .then_with(|| left_digits.cmp(right_digits));
            if order != Ordering::Equal {
                return order;
            }
            continue;
        }
        let order = left[left_index]
            .to_ascii_lowercase()
            .cmp(&right[right_index].to_ascii_lowercase());
        if order != Ordering::Equal {
            return order;
        }
        left_index += 1;
        right_index += 1;
    }
    left.len().cmp(&right.len()).then_with(|| left.cmp(right))
}

fn trim_leading_zeros(digits: &[u8]) -> &[u8] {
    let first_nonzero = digits.iter().position(|digit| *digit != b'0');
    first_nonzero.map_or(&digits[digits.len() - 1..], |index| &digits[index..])
}

fn sharpness(image: &DynamicImage) -> f64 {
    let preview = image
        .grayscale()
        .resize(320, 320, FilterType::Triangle)
        .to_luma8();
    let (width, height) = preview.dimensions();
    if width < 3 || height < 3 {
        return 0.0;
    }
    let mut values = Vec::with_capacity(((width - 2) * (height - 2)) as usize);
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let center = preview.get_pixel(x, y)[0] as f64;
            let laplacian = 4.0 * center
                - preview.get_pixel(x - 1, y)[0] as f64
                - preview.get_pixel(x + 1, y)[0] as f64
                - preview.get_pixel(x, y - 1)[0] as f64
                - preview.get_pixel(x, y + 1)[0] as f64;
            values.push(laplacian);
        }
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64
}

fn perceptual_hash(image: &DynamicImage) -> u64 {
    let pixels = image
        .grayscale()
        .resize_exact(32, 32, FilterType::Triangle)
        .to_luma8();
    let mut coefficients = [[0.0_f64; 8]; 8];
    for (v, row) in coefficients.iter_mut().enumerate() {
        for (u, coefficient) in row.iter_mut().enumerate() {
            let mut sum = 0.0;
            for y in 0..32 {
                for x in 0..32 {
                    let sample = pixels.get_pixel(x, y)[0] as f64;
                    sum += sample
                        * ((std::f64::consts::PI * (2 * x + 1) as f64 * u as f64) / 64.0).cos()
                        * ((std::f64::consts::PI * (2 * y + 1) as f64 * v as f64) / 64.0).cos();
                }
            }
            *coefficient = sum;
        }
    }
    hash_dct_coefficients(coefficients)
}

fn hash_dct_coefficients(coefficients: [[f64; 8]; 8]) -> u64 {
    let mut values = coefficients.iter().flatten().copied().collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    let median = values[values.len() / 2];
    coefficients
        .iter()
        .flatten()
        .copied()
        .enumerate()
        .fold(0_u64, |hash, (index, value)| {
            if value > median {
                hash | (1_u64 << index)
            } else {
                hash
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::{ProjectCreateRequest, SourceSpec};

    #[test]
    fn failed_stage_persistence_is_reported_to_the_import_caller() {
        let temp = tempfile::tempdir().unwrap();
        let mut store = ProjectStore::create(
            temp.path().join("Import.rustscanproject"),
            ProjectCreateRequest::new("Import", SourceSpec::managed_images("pending")),
        )
        .unwrap();

        let error = persist_import_failure(
            &mut store,
            MediaImportError::Decode("fixture decode failure".to_owned()),
            |_, _| {
                Err(ProjectStoreError::Io(io::Error::other(
                    "injected manifest failure",
                )))
            },
        );

        assert!(matches!(
            error,
            MediaImportError::FailedStagePersistence { .. }
        ));
    }

    #[test]
    fn failed_commit_recovers_the_active_import_attempt_before_returning() {
        let temp = tempfile::tempdir().unwrap();
        let mut store = ProjectStore::create(
            temp.path().join("Commit.rustscanproject"),
            ProjectCreateRequest::new("Commit", SourceSpec::managed_images("pending")),
        )
        .unwrap();
        store.begin_stage(ProjectStage::Import).unwrap();

        let error = commit_import_or_recover(&mut store, |_| {
            Err(ProjectStoreError::Io(io::Error::other(
                "injected commit failure",
            )))
        })
        .unwrap_err();

        assert!(matches!(error, MediaImportError::Project(_)));
        assert!(store.manifest().lease().is_none());
        assert_eq!(
            store
                .manifest()
                .try_stage(ProjectStage::Import)
                .unwrap()
                .state(),
            crate::project::StageState::Failed
        );
    }

    #[test]
    fn dct_hash_maps_the_64th_coefficient_to_the_high_bit() {
        let mut coefficients = [[0.0_f64; 8]; 8];
        coefficients[7][7] = 100.0;

        assert_ne!(hash_dct_coefficients(coefficients) & (1_u64 << 63), 0);
    }
}
