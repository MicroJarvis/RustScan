use std::collections::HashSet;
use std::fs::{self, File};
use std::io;
use std::path::Path;

use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb};
use serde::Serialize;

use crate::project::artifacts::{ArtifactValidationKind, StagedArtifact};
use crate::project::source::{ImportedSourceFrame, ImportedSourceRecord};
use crate::project::{
    ProjectErrorRecord, ProjectStage, ProjectStore, ProjectStoreError, SourceKind, SourceSpec,
    SuggestedAction,
};

use super::{
    select_keyframes, ImportResult, ImportedFrame, KeyframeSelectionConfig, MediaEventSink,
    MediaImportError, MediaImportEvent,
};

const SOURCE_METADATA_PAYLOAD: &str = "Sources/source.json";
const FRAMES_METADATA_PAYLOAD: &str = "Cache/frames.json";
const KEYFRAMES_METADATA_PAYLOAD: &str = "Cache/keyframes.json";
const FRAME_DIRECTORY: &str = "Cache/frames";
const THUMBNAIL_DIRECTORY: &str = "Cache/thumbnails";
const THUMBNAIL_LONG_EDGE: u32 = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedVideoFrame {
    pub presentation_time_us: i64,
    pub width: u32,
    pub height: u32,
    pub bgra: Vec<u8>,
    pub bytes_per_row: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoMetadata {
    pub duration_us: i64,
    pub width: u32,
    pub height: u32,
    pub nominal_fps: f64,
}

pub trait VideoDecoder: Send {
    fn metadata(&self) -> Result<VideoMetadata, MediaImportError>;
    fn next_frame(&mut self) -> Result<Option<DecodedVideoFrame>, MediaImportError>;
}

pub fn import_video(
    decoder: &mut impl VideoDecoder,
    source: SourceSpec,
    store: &mut ProjectStore,
    sink: &mut impl MediaEventSink,
) -> Result<ImportResult, MediaImportError> {
    let metadata = match validate_video_source(decoder, &source) {
        Ok(metadata) => metadata,
        Err(error) => return Err(persist_video_preflight_failure(store, error)),
    };
    store.update_source(source.clone())?;

    sink.on_media_event(MediaImportEvent::Started { total: None });
    let workspace = store.begin_stage(ProjectStage::Import)?;
    let keyframe_config = KeyframeSelectionConfig {
        target_per_second: store.manifest().import_config.video_keyframes_per_second,
        max_gap_us: store.manifest().import_config.maximum_keyframe_gap_us,
        duplicate_hamming_threshold: KeyframeSelectionConfig::default().duplicate_hamming_threshold,
    };
    let import = import_into_workspace(
        decoder,
        &source,
        metadata,
        keyframe_config,
        &workspace,
        sink,
    );
    match import {
        Ok((result, declarations, payloads)) => {
            if let Err(error) = store
                .validate_stage_payloads(&workspace, &payloads)
                .map_err(MediaImportError::Project)
            {
                return Err(persist_video_failure(store, error, |store, record| {
                    store.mark_stage_failed(ProjectStage::Import, record)
                }));
            }
            match store.commit_stage_success(&workspace, &declarations, false) {
                Ok(()) => {
                    sink.on_media_event(MediaImportEvent::Completed {
                        frame_count: result.frames.len(),
                        keyframe_count: result
                            .frames
                            .iter()
                            .filter(|frame| frame.is_keyframe)
                            .count(),
                    });
                    Ok(result)
                }
                Err(commit) => match store.recover_interrupted_stage() {
                    Ok(()) => Err(MediaImportError::Project(commit)),
                    Err(recovery) => Err(MediaImportError::CommitRecoveryRequired {
                        commit: commit.to_string(),
                        recovery: recovery.to_string(),
                    }),
                },
            }
        }
        Err(error) => Err(persist_video_failure(store, error, |store, record| {
            store.mark_stage_failed(ProjectStage::Import, record)
        })),
    }
}

fn validate_video_source(
    decoder: &impl VideoDecoder,
    source: &SourceSpec,
) -> Result<VideoMetadata, MediaImportError> {
    if source.kind != SourceKind::Video {
        return Err(MediaImportError::InvalidSource(
            "video import requires a video source specification".to_owned(),
        ));
    }
    let metadata = decoder.metadata()?;
    if metadata.duration_us < 0
        || metadata.width == 0
        || metadata.height == 0
        || !metadata.nominal_fps.is_finite()
        || metadata.nominal_fps < 0.0
    {
        return Err(MediaImportError::InvalidSource(
            "video decoder returned invalid metadata".to_owned(),
        ));
    }
    Ok(metadata)
}

fn import_into_workspace(
    decoder: &mut impl VideoDecoder,
    source: &SourceSpec,
    metadata: VideoMetadata,
    keyframe_config: KeyframeSelectionConfig,
    workspace: &crate::project::artifacts::StageWorkspace,
    sink: &mut impl MediaEventSink,
) -> Result<(ImportResult, Vec<StagedArtifact>, Vec<StagedArtifact>), MediaImportError> {
    let final_prefix = format!("Artifacts/import/attempt-{:08}", workspace.attempt());
    let mut frames = Vec::new();
    let mut source_frames = Vec::new();
    let mut payloads = Vec::new();
    let mut previous_time_us = None;

    while let Some(frame) = decoder.next_frame()? {
        let id = u32::try_from(frames.len()).map_err(|_| {
            MediaImportError::InvalidSource("video contains too many frames".to_owned())
        })?;
        validate_frame(&frame, metadata, previous_time_us)?;
        previous_time_us = Some(frame.presentation_time_us);
        let decoded = bgra_to_rgb(&frame)?;
        let frame_payload = format!("{FRAME_DIRECTORY}/{id:08}.png");
        let thumbnail_payload = format!("{THUMBNAIL_DIRECTORY}/{id:08}.jpg");
        write_png(&workspace.path().join(&frame_payload), &decoded)?;
        write_thumbnail(&workspace.path().join(&thumbnail_payload), &decoded)?;
        payloads.push(readable_payload(&frame_payload)?);
        payloads.push(readable_payload(&thumbnail_payload)?);
        frames.push(ImportedFrame {
            id,
            source_name: format!("frame{id:08}.png"),
            presentation_time_us: Some(frame.presentation_time_us),
            normalized_image: format!("{final_prefix}/{frame_payload}"),
            thumbnail: format!("{final_prefix}/{thumbnail_payload}"),
            width: frame.width,
            height: frame.height,
            sharpness: sharpness(&decoded),
            perceptual_hash: perceptual_hash(&decoded),
            is_keyframe: false,
        });
        source_frames.push(ImportedSourceFrame {
            id,
            source_name: format!("frame{id:08}.png"),
            original_path: source.display_paths.first().cloned().unwrap_or_default(),
            managed_copy: None,
        });
        sink.on_media_event(MediaImportEvent::FrameCommitted {
            frame_id: id,
            completed: frames.len(),
            total: None,
        });
    }

    if frames.is_empty() {
        return Err(MediaImportError::InvalidSource(
            "video decoder produced no frames".to_owned(),
        ));
    }
    let selected = select_keyframes(&frames, keyframe_config)
        .map_err(|error| MediaImportError::InvalidSource(error.to_string()))?
        .into_iter()
        .collect::<HashSet<_>>();
    for frame in &mut frames {
        frame.is_keyframe = selected.contains(&frame.id);
    }

    let source_metadata = format!("{final_prefix}/{SOURCE_METADATA_PAYLOAD}");
    let frames_metadata = format!("{final_prefix}/{FRAMES_METADATA_PAYLOAD}");
    let keyframes_metadata = format!("{final_prefix}/{KEYFRAMES_METADATA_PAYLOAD}");
    let source_record = ImportedSourceRecord {
        schema_version: 1,
        ownership: source.ownership,
        identity: source.identity.clone(),
        frames: source_frames,
    };
    write_json(
        &workspace.path().join(SOURCE_METADATA_PAYLOAD),
        &source_record,
    )?;
    write_json(&workspace.path().join(FRAMES_METADATA_PAYLOAD), &frames)?;
    write_json(
        &workspace.path().join(KEYFRAMES_METADATA_PAYLOAD),
        &frames
            .iter()
            .filter(|frame| frame.is_keyframe)
            .map(|frame| frame.id)
            .collect::<Vec<_>>(),
    )?;
    let declarations = [
        SOURCE_METADATA_PAYLOAD,
        FRAMES_METADATA_PAYLOAD,
        KEYFRAMES_METADATA_PAYLOAD,
    ]
    .into_iter()
    .map(|payload| {
        StagedArtifact::new(payload, ArtifactValidationKind::Json)
            .map_err(|error| MediaImportError::InvalidSource(error.to_string()))
    })
    .collect::<Result<Vec<_>, _>>()?;
    payloads.extend(declarations.iter().cloned());
    Ok((
        ImportResult {
            source_identity: source.identity.clone(),
            frames,
            duration_us: Some(metadata.duration_us),
            source_metadata,
            frames_metadata,
            keyframes_metadata,
        },
        declarations,
        payloads,
    ))
}

fn validate_frame(
    frame: &DecodedVideoFrame,
    metadata: VideoMetadata,
    previous_time_us: Option<i64>,
) -> Result<(), MediaImportError> {
    if frame.presentation_time_us < 0 {
        return Err(MediaImportError::InvalidSource(
            "video frame has a negative presentation timestamp".to_owned(),
        ));
    }
    if let Some(previous_time_us) = previous_time_us {
        if frame.presentation_time_us <= previous_time_us {
            return Err(MediaImportError::InvalidSource(
                "video frame presentation timestamps must be strictly increasing".to_owned(),
            ));
        }
    }
    if frame.width == 0
        || frame.height == 0
        || frame.width != metadata.width
        || frame.height != metadata.height
    {
        return Err(MediaImportError::InvalidSource(
            "video frame dimensions do not match decoder metadata".to_owned(),
        ));
    }
    let minimum_row_bytes = usize::try_from(frame.width)
        .ok()
        .and_then(|width| width.checked_mul(4))
        .ok_or_else(|| {
            MediaImportError::InvalidSource("video frame width is too large".to_owned())
        })?;
    let required_bytes = frame
        .bytes_per_row
        .checked_mul(usize::try_from(frame.height).map_err(|_| {
            MediaImportError::InvalidSource("video frame height is too large".to_owned())
        })?)
        .ok_or_else(|| {
            MediaImportError::InvalidSource("video frame buffer is too large".to_owned())
        })?;
    if frame.bytes_per_row < minimum_row_bytes || frame.bgra.len() < required_bytes {
        return Err(MediaImportError::InvalidSource(
            "video frame BGRA buffer does not match its row stride".to_owned(),
        ));
    }
    Ok(())
}

fn bgra_to_rgb(frame: &DecodedVideoFrame) -> Result<DynamicImage, MediaImportError> {
    let pixel_count = usize::try_from(frame.width)
        .ok()
        .and_then(|width| {
            usize::try_from(frame.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or_else(|| {
            MediaImportError::InvalidSource("video frame dimensions are too large".to_owned())
        })?;
    let mut rgb = Vec::with_capacity(pixel_count * 3);
    for row in 0..usize::try_from(frame.height).unwrap_or_default() {
        let start = row * frame.bytes_per_row;
        for pixel in frame.bgra[start..start + usize::try_from(frame.width).unwrap_or_default() * 4]
            .chunks_exact(4)
        {
            rgb.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
        }
    }
    let image =
        ImageBuffer::<Rgb<u8>, _>::from_raw(frame.width, frame.height, rgb).ok_or_else(|| {
            MediaImportError::InvalidSource(
                "video RGB conversion produced an invalid image".to_owned(),
            )
        })?;
    Ok(DynamicImage::ImageRgb8(image))
}

fn persist_video_preflight_failure(
    store: &mut ProjectStore,
    error: MediaImportError,
) -> MediaImportError {
    persist_video_failure(store, error, |store, record| {
        store.mark_stage_preflight_failed(ProjectStage::Import, record)
    })
}

fn persist_video_failure(
    store: &mut ProjectStore,
    error: MediaImportError,
    mark_failed: impl FnOnce(&mut ProjectStore, ProjectErrorRecord) -> Result<(), ProjectStoreError>,
) -> MediaImportError {
    let detail = error.to_string();
    let record = ProjectErrorRecord {
        code: "video_import_failed".to_owned(),
        stage: ProjectStage::Import,
        summary: "Video import failed".to_owned(),
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
    let (width, height) = (image.width(), image.height());
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

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), MediaImportError> {
    create_parent(path)?;
    fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|error| {
            MediaImportError::InvalidSource(format!("failed to serialize video metadata: {error}"))
        })?,
    )?;
    Ok(())
}

fn create_parent(path: &Path) -> Result<(), io::Error> {
    let parent = path.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("payload has no parent: {}", path.display()),
        )
    })?;
    fs::create_dir_all(parent)
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
            values.push(
                4.0 * center
                    - preview.get_pixel(x - 1, y)[0] as f64
                    - preview.get_pixel(x + 1, y)[0] as f64
                    - preview.get_pixel(x, y - 1)[0] as f64
                    - preview.get_pixel(x, y + 1)[0] as f64,
            );
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
                    sum += pixels.get_pixel(x, y)[0] as f64
                        * ((std::f64::consts::PI * (2 * x + 1) as f64 * u as f64) / 64.0).cos()
                        * ((std::f64::consts::PI * (2 * y + 1) as f64 * v as f64) / 64.0).cos();
                }
            }
            *coefficient = sum;
        }
    }
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
