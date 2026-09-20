use std::fs;
use std::path::{Path, PathBuf};
use std::slice;

use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2::AnyThread;
use objc2_av_foundation::{
    AVAssetReader, AVAssetReaderStatus, AVAssetReaderTrackOutput, AVMediaTypeVideo, AVURLAsset,
};
use objc2_core_media::CMSampleBuffer;
use objc2_core_video::{
    kCVPixelFormatType_32BGRA, kCVReturnSuccess, CVPixelBufferGetBaseAddress,
    CVPixelBufferGetBytesPerRow, CVPixelBufferGetHeight, CVPixelBufferGetWidth,
    CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress,
};
use objc2_foundation::{NSDictionary, NSNumber, NSString, NSURL};

use crate::project::source::{image_sequence_identity, ScopedSourcePath, SourceBookmark};
use crate::project::{SourceKind, SourceOwnership, SourceSpec};

use super::{DecodedVideoFrame, MediaImportError, VideoDecoder, VideoMetadata};

pub struct AvFoundationVideoDecoder {
    reader: Retained<AVAssetReader>,
    output: Retained<AVAssetReaderTrackOutput>,
    metadata: VideoMetadata,
    // Kept last so the reader/output fields are released before its security scope ends.
    #[allow(dead_code)]
    source_access: Option<ScopedSourcePath>,
}

// AVAssetReader allows a single reader/output owner to be moved to a worker; all reads require
// `&mut self`, and AVFoundation documents that concurrent `copyNextSampleBuffer` calls are invalid.
unsafe impl Send for AvFoundationVideoDecoder {}

impl AvFoundationVideoDecoder {
    pub fn open_referenced(path: impl AsRef<Path>) -> Result<(Self, SourceSpec), MediaImportError> {
        let canonical_path = canonical_video_path(path.as_ref())?;
        let source = video_source_spec(&canonical_path, SourceOwnership::Referenced, true)?;
        let decoder = Self::open_saved_referenced(&source)?;
        Ok((decoder, source))
    }

    /// Reopens a persisted referenced source through its bookmark, never through display paths.
    pub fn open_saved_referenced(source: &SourceSpec) -> Result<Self, MediaImportError> {
        if source.kind != SourceKind::Video || source.ownership != SourceOwnership::Referenced {
            return Err(MediaImportError::InvalidSource(
                "saved source is not a referenced video".to_owned(),
            ));
        }
        let bookmark = SourceBookmark::decode(source.bookmark.as_deref().ok_or_else(|| {
            MediaImportError::InvalidSource(
                "referenced video source is missing its security-scoped bookmark".to_owned(),
            )
        })?)
        .map_err(|error| {
            MediaImportError::InvalidSource(format!("invalid referenced video bookmark: {error}"))
        })?;
        let access = bookmark.resolve_single_path_with_scope().map_err(|error| {
            MediaImportError::InvalidSource(format!("referenced video bookmark: {error}"))
        })?;
        let scoped_path = access.path().to_path_buf();
        Self::open_path(&scoped_path, Some(access))
    }

    pub fn open_managed(path: impl AsRef<Path>) -> Result<(Self, SourceSpec), MediaImportError> {
        let canonical_path = canonical_video_path(path.as_ref())?;
        let source = video_source_spec(&canonical_path, SourceOwnership::ManagedCopy, false)?;
        let decoder = Self::open_path(&canonical_path, None)?;
        Ok((decoder, source))
    }

    fn open_path(
        path: &Path,
        source_access: Option<ScopedSourcePath>,
    ) -> Result<Self, MediaImportError> {
        let path = path.to_str().ok_or_else(|| {
            MediaImportError::InvalidSource(format!(
                "video path is not valid UTF-8: {}",
                path.display()
            ))
        })?;
        let ns_path = NSString::from_str(path);
        let url = NSURL::fileURLWithPath(&ns_path);
        let asset = unsafe { AVURLAsset::URLAssetWithURL_options(&url, None) };
        let video_type = unsafe { AVMediaTypeVideo }.ok_or_else(|| {
            MediaImportError::VideoDecoderFailed("AVMediaTypeVideo is unavailable".to_owned())
        })?;
        let tracks = unsafe { asset.tracks() };
        let track = tracks
            .iter()
            .find(|track| unsafe { &*track.mediaType() == video_type })
            .ok_or_else(|| {
                MediaImportError::InvalidSource("video asset contains no video track".to_owned())
            })?
            .to_owned();
        let size = unsafe { track.naturalSize() };
        let width = f64_to_dimension(size.width)?;
        let height = f64_to_dimension(size.height)?;
        let duration_us = seconds_to_microseconds(unsafe { asset.duration().seconds() })?;
        let nominal_fps = unsafe { track.nominalFrameRate() } as f64;
        if !nominal_fps.is_finite() || nominal_fps < 0.0 {
            return Err(MediaImportError::VideoDecoderFailed(
                "video track has an invalid nominal frame rate".to_owned(),
            ));
        }

        let key = NSString::from_str("PixelFormatType");
        let value = NSNumber::new_u32(kCVPixelFormatType_32BGRA);
        let settings = NSDictionary::<NSString, AnyObject>::from_slices(&[&*key], &[&*value]);
        let output = unsafe {
            AVAssetReaderTrackOutput::assetReaderTrackOutputWithTrack_outputSettings(
                &track,
                Some(&settings),
            )
        };
        let reader = unsafe { AVAssetReader::initWithAsset_error(AVAssetReader::alloc(), &asset) }
            .map_err(|error| MediaImportError::VideoDecoderFailed(error.to_string()))?;
        if !unsafe { reader.canAddOutput(&output) } {
            return Err(MediaImportError::VideoDecoderFailed(
                "AVAssetReader rejected the video track output".to_owned(),
            ));
        }
        unsafe { reader.addOutput(&output) };
        if !unsafe { reader.startReading() } {
            return Err(reader_failure(&reader));
        }

        Ok(Self {
            reader,
            output,
            metadata: VideoMetadata {
                duration_us,
                width,
                height,
                nominal_fps,
            },
            source_access,
        })
    }
}

impl Drop for AvFoundationVideoDecoder {
    fn drop(&mut self) {
        if unsafe { self.reader.status() } == AVAssetReaderStatus::Reading {
            unsafe { self.reader.cancelReading() };
        }
    }
}

impl VideoDecoder for AvFoundationVideoDecoder {
    fn metadata(&self) -> Result<VideoMetadata, MediaImportError> {
        Ok(self.metadata)
    }

    fn next_frame(&mut self) -> Result<Option<DecodedVideoFrame>, MediaImportError> {
        let Some(sample) = (unsafe { self.output.copyNextSampleBuffer() }) else {
            return match unsafe { self.reader.status() } {
                AVAssetReaderStatus::Completed => Ok(None),
                AVAssetReaderStatus::Cancelled => Err(MediaImportError::VideoDecoderCancelled),
                AVAssetReaderStatus::Failed => Err(reader_failure(&self.reader)),
                status => Err(MediaImportError::VideoDecoderFailed(format!(
                    "AVAssetReader ended with unexpected status {}",
                    status.0
                ))),
            };
        };
        decode_sample(&sample)
    }
}

fn decode_sample(sample: &CMSampleBuffer) -> Result<Option<DecodedVideoFrame>, MediaImportError> {
    let pixel_buffer = unsafe { sample.image_buffer() }.ok_or_else(|| {
        MediaImportError::VideoDecoderFailed(
            "video sample did not contain a pixel buffer".to_owned(),
        )
    })?;
    let timestamp = seconds_to_microseconds(unsafe { sample.presentation_time_stamp().seconds() })?;
    let lock_flags = CVPixelBufferLockFlags::ReadOnly;
    if unsafe { CVPixelBufferLockBaseAddress(&pixel_buffer, lock_flags) } != kCVReturnSuccess {
        return Err(MediaImportError::VideoDecoderFailed(
            "unable to lock CVPixelBuffer for read access".to_owned(),
        ));
    }
    let copied = (|| {
        let width = u32::try_from(CVPixelBufferGetWidth(&pixel_buffer)).map_err(|_| {
            MediaImportError::VideoDecoderFailed("CVPixelBuffer width exceeds u32".to_owned())
        })?;
        let height = u32::try_from(CVPixelBufferGetHeight(&pixel_buffer)).map_err(|_| {
            MediaImportError::VideoDecoderFailed("CVPixelBuffer height exceeds u32".to_owned())
        })?;
        let bytes_per_row = CVPixelBufferGetBytesPerRow(&pixel_buffer);
        let byte_len = bytes_per_row.checked_mul(height as usize).ok_or_else(|| {
            MediaImportError::VideoDecoderFailed("CVPixelBuffer is too large".to_owned())
        })?;
        let base = CVPixelBufferGetBaseAddress(&pixel_buffer);
        if base.is_null() {
            return Err(MediaImportError::VideoDecoderFailed(
                "locked CVPixelBuffer has no base address".to_owned(),
            ));
        }
        let bgra = unsafe { slice::from_raw_parts(base.cast::<u8>(), byte_len) }.to_vec();
        Ok(DecodedVideoFrame {
            presentation_time_us: timestamp,
            width,
            height,
            bgra,
            bytes_per_row,
        })
    })();
    let unlock = unsafe { CVPixelBufferUnlockBaseAddress(&pixel_buffer, lock_flags) };
    if unlock != kCVReturnSuccess {
        return Err(MediaImportError::VideoDecoderFailed(
            "unable to unlock CVPixelBuffer".to_owned(),
        ));
    }
    copied.map(Some)
}

fn reader_failure(reader: &AVAssetReader) -> MediaImportError {
    let detail = unsafe { reader.error() }
        .map(|error| error.to_string())
        .unwrap_or_else(|| "AVAssetReader could not start or complete reading".to_owned());
    MediaImportError::VideoDecoderFailed(detail)
}

fn seconds_to_microseconds(seconds: f64) -> Result<i64, MediaImportError> {
    let microseconds = seconds * 1_000_000.0;
    if !microseconds.is_finite() || microseconds < 0.0 || microseconds > i64::MAX as f64 {
        return Err(MediaImportError::VideoDecoderFailed(
            "video timestamp is invalid".to_owned(),
        ));
    }
    Ok(microseconds.round() as i64)
}

fn f64_to_dimension(value: f64) -> Result<u32, MediaImportError> {
    if !value.is_finite() || value <= 0.0 || value > u32::MAX as f64 {
        return Err(MediaImportError::VideoDecoderFailed(
            "video track has invalid dimensions".to_owned(),
        ));
    }
    Ok(value.round() as u32)
}

fn canonical_video_path(path: &Path) -> Result<PathBuf, MediaImportError> {
    let path = fs::canonicalize(path)
        .map_err(|error| MediaImportError::InvalidSource(format!("{}: {error}", path.display())))?;
    if !path.is_file() {
        return Err(MediaImportError::InvalidSource(format!(
            "not a regular video file: {}",
            path.display()
        )));
    }
    Ok(path)
}

fn video_source_spec(
    canonical_path: &Path,
    ownership: SourceOwnership,
    create_bookmark: bool,
) -> Result<SourceSpec, MediaImportError> {
    let display_path = canonical_path.to_string_lossy().into_owned();
    let name = canonical_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            MediaImportError::InvalidSource(format!(
                "video name is not valid UTF-8: {}",
                canonical_path.display()
            ))
        })?
        .to_owned();
    let identity = image_sequence_identity(&[(name, canonical_path.to_path_buf())])
        .map_err(MediaImportError::Io)?;
    let bookmark = if create_bookmark {
        Some(
            SourceBookmark::from_canonical_paths(vec![display_path.clone()])
                .map_err(|error| MediaImportError::InvalidSource(error.to_string()))?
                .encode()
                .map_err(|error| MediaImportError::InvalidSource(error.to_string()))?,
        )
    } else {
        None
    };
    Ok(SourceSpec {
        kind: SourceKind::Video,
        ownership,
        identity,
        display_paths: vec![display_path],
        bookmark,
    })
}
