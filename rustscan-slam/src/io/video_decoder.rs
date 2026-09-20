//! Video input and decoding helpers for RustScan.

use std::collections::VecDeque;
use std::fs::File;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use lru::LruCache;
use thiserror::Error;

use ffmpeg::codec;
use ffmpeg::format;
use ffmpeg::media;
use ffmpeg::software::scaling::{context::Context as ScalingContext, flag::Flags as ScalingFlags};
use ffmpeg::util::format::pixel::Pixel;
use ffmpeg::util::frame;
use ffmpeg_next as ffmpeg;

#[derive(Debug, Error)]
pub enum VideoError {
    #[error("invalid video path: {0}")]
    InvalidPath(PathBuf),
    #[error("video file not found: {0}")]
    MissingPath(PathBuf),
    #[error("video path is not a file: {0}")]
    NotAFile(PathBuf),
    #[error("video file is empty: {0}")]
    EmptyFile(PathBuf),
    #[error("video file is not readable: {0}")]
    Unreadable(PathBuf),
    #[error("failed to initialize ffmpeg: {0}")]
    InitFailed(String),
    #[error("failed to open input: {0}")]
    OpenFailed(String),
    #[error("missing video stream")]
    StreamMissing,
    #[error("unsupported container format: {format} (extension: {extension})")]
    UnsupportedContainer { format: String, extension: String },
    #[error("unsupported video codec: {0}")]
    UnsupportedCodec(String),
    #[error("failed to create decoder: {0}")]
    Decoder(String),
    #[error("failed to configure scaler: {0}")]
    Scaler(String),
    #[error("frame index out of range: {0}")]
    FrameIndex(usize),
    #[error("decode error: {0}")]
    Decode(String),
}

pub type Result<T> = std::result::Result<T, VideoError>;

#[derive(Debug, Clone)]
pub struct VideoInfo {
    pub container: String,
    pub codec: String,
    pub width: u32,
    pub height: u32,
    pub frame_rate: f64,
    pub frame_count: Option<usize>,
    pub decoder: String,
    pub hardware_accel: bool,
}

#[derive(Debug, Clone)]
pub struct VideoFrame {
    pub index: usize,
    pub timestamp: f64,
    pub width: u32,
    pub height: u32,
    pub stride: usize,
    pub data: Arc<Vec<u8>>,
}

#[derive(Debug, Clone)]
pub struct VideoDecoderConfig {
    pub cache_capacity: usize,
    pub prefer_hardware: bool,
}

impl Default for VideoDecoderConfig {
    fn default() -> Self {
        Self {
            cache_capacity: 100,
            prefer_hardware: true,
        }
    }
}

pub struct VideoDecoder {
    path: PathBuf,
    info: VideoInfo,
    input: format::context::Input,
    stream_index: usize,
    decoder: ffmpeg::decoder::Video,
    scaler: ScalingContext,
    cache: LruCache<usize, Arc<VideoFrame>>,
    pending: VecDeque<Arc<VideoFrame>>,
    next_index: usize,
    prefer_hardware: bool,
}

impl VideoDecoder {
    pub fn open<P: AsRef<Path>>(path: P, config: VideoDecoderConfig) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let path_str = path
            .to_str()
            .ok_or_else(|| VideoError::InvalidPath(path.clone()))?;

        validate_file(&path)?;

        ffmpeg::init().map_err(|err| VideoError::InitFailed(err.to_string()))?;

        let input =
            format::input(&path_str).map_err(|err| VideoError::OpenFailed(err.to_string()))?;

        let format_name = input.format().name().to_string();
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("unknown")
            .to_ascii_lowercase();

        if !is_supported_container(&format_name, &extension) {
            return Err(VideoError::UnsupportedContainer {
                format: format_name,
                extension,
            });
        }

        let stream = input
            .streams()
            .best(media::Type::Video)
            .ok_or(VideoError::StreamMissing)?;
        let stream_index = stream.index();

        let codec_context = codec::context::Context::from_parameters(stream.parameters())
            .map_err(|err| VideoError::Decoder(err.to_string()))?;
        let codec_id = codec_context.id();

        if !is_supported_codec(codec_id) {
            return Err(VideoError::UnsupportedCodec(format!("{codec_id:?}")));
        }

        let (decoder, decoder_name, hardware_accel) =
            open_decoder(&stream, codec_id, config.prefer_hardware)?;

        let scaler = ScalingContext::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            Pixel::RGB24,
            decoder.width(),
            decoder.height(),
            ScalingFlags::BILINEAR,
        )
        .map_err(|err| VideoError::Scaler(err.to_string()))?;

        let frame_rate = frame_rate_from_stream(&stream);
        let frame_count = match stream.frames() {
            count if count > 0 => Some(count as usize),
            _ => None,
        };

        let info = VideoInfo {
            container: format_name,
            codec: codec_label(codec_id).to_string(),
            width: decoder.width(),
            height: decoder.height(),
            frame_rate,
            frame_count,
            decoder: decoder_name,
            hardware_accel,
        };

        let capacity = NonZeroUsize::new(config.cache_capacity.max(1))
            .unwrap_or_else(|| NonZeroUsize::new(1).expect("non-zero"));
        let cache = LruCache::new(capacity);

        Ok(Self {
            path,
            info,
            input,
            stream_index,
            decoder,
            scaler,
            cache,
            pending: VecDeque::new(),
            next_index: 0,
            prefer_hardware: config.prefer_hardware,
        })
    }

    pub fn info(&self) -> &VideoInfo {
        &self.info
    }

    pub fn frame(&mut self, index: usize) -> Result<Arc<VideoFrame>> {
        if let Some(frame) = self.cache.get(&index) {
            return Ok(frame.clone());
        }

        if index < self.next_index {
            self.reset_decoder()?;
        }

        while self.next_index <= index {
            let next = self
                .decode_next_frame()?
                .ok_or(VideoError::FrameIndex(index))?;
            self.cache.put(next.index, next.clone());
        }

        self.cache
            .get(&index)
            .cloned()
            .ok_or(VideoError::FrameIndex(index))
    }

    fn reset_decoder(&mut self) -> Result<()> {
        let path_str = self
            .path
            .to_str()
            .ok_or_else(|| VideoError::InvalidPath(self.path.clone()))?;
        self.input =
            format::input(path_str).map_err(|err| VideoError::OpenFailed(err.to_string()))?;

        let stream = self
            .input
            .streams()
            .best(media::Type::Video)
            .ok_or(VideoError::StreamMissing)?;
        self.stream_index = stream.index();

        let codec_context = codec::context::Context::from_parameters(stream.parameters())
            .map_err(|err| VideoError::Decoder(err.to_string()))?;
        let codec_id = codec_context.id();

        let (decoder, decoder_name, hardware_accel) =
            open_decoder(&stream, codec_id, self.prefer_hardware)?;
        self.decoder = decoder;
        self.info.decoder = decoder_name;
        self.info.hardware_accel = hardware_accel;

        self.scaler = ScalingContext::get(
            self.decoder.format(),
            self.decoder.width(),
            self.decoder.height(),
            Pixel::RGB24,
            self.decoder.width(),
            self.decoder.height(),
            ScalingFlags::BILINEAR,
        )
        .map_err(|err| VideoError::Scaler(err.to_string()))?;

        self.pending.clear();
        self.next_index = 0;
        Ok(())
    }

    fn decode_next_frame(&mut self) -> Result<Option<Arc<VideoFrame>>> {
        if let Some(frame) = self.pending.pop_front() {
            return Ok(Some(frame));
        }

        let stream_index = self.stream_index;
        let mut decoded = frame::Video::empty();
        let VideoDecoder {
            input,
            decoder,
            scaler,
            pending,
            next_index,
            info,
            ..
        } = self;

        for (stream, packet) in input.packets() {
            if stream.index() != stream_index {
                continue;
            }

            decoder
                .send_packet(&packet)
                .map_err(|err| VideoError::Decode(err.to_string()))?;

            while decoder.receive_frame(&mut decoded).is_ok() {
                let frame = convert_frame_with(scaler, info, next_index, &decoded)?;
                pending.push_back(frame);
            }

            if let Some(frame) = pending.pop_front() {
                return Ok(Some(frame));
            }
        }

        decoder
            .send_eof()
            .map_err(|err| VideoError::Decode(err.to_string()))?;
        while decoder.receive_frame(&mut decoded).is_ok() {
            let frame = convert_frame_with(scaler, info, next_index, &decoded)?;
            pending.push_back(frame);
        }

        Ok(pending.pop_front())
    }

    #[allow(dead_code)]
    fn convert_frame(&mut self, decoded: &frame::Video) -> Result<Arc<VideoFrame>> {
        convert_frame_with(&mut self.scaler, &self.info, &mut self.next_index, decoded)
    }
}

fn convert_frame_with(
    scaler: &mut ScalingContext,
    info: &VideoInfo,
    next_index: &mut usize,
    decoded: &frame::Video,
) -> Result<Arc<VideoFrame>> {
    let mut rgb = frame::Video::empty();
    scaler
        .run(decoded, &mut rgb)
        .map_err(|err| VideoError::Decode(err.to_string()))?;

    let index = *next_index;
    *next_index += 1;

    let stride = rgb.stride(0);
    let height = rgb.height();
    let data = rgb.data(0).to_vec();

    let frame = VideoFrame {
        index,
        timestamp: index as f64 / info.frame_rate,
        width: rgb.width(),
        height,
        stride,
        data: Arc::new(data),
    };

    Ok(Arc::new(frame))
}
fn is_supported_container(format_name: &str, extension: &str) -> bool {
    let format_name = format_name.to_ascii_lowercase();
    let extension = extension.to_ascii_lowercase();

    if matches!(extension.as_str(), "mp4" | "mov" | "hevc" | "m4v") {
        return true;
    }

    format_name.contains("mov") || format_name.contains("mp4")
}

fn is_supported_codec(codec_id: codec::Id) -> bool {
    matches!(codec_id, codec::Id::H264 | codec::Id::HEVC)
}

fn codec_label(codec_id: codec::Id) -> &'static str {
    match codec_id {
        codec::Id::H264 => "H.264",
        codec::Id::HEVC => "H.265/HEVC",
        _ => "Unknown",
    }
}

fn validate_file(path: &Path) -> Result<()> {
    let metadata = match std::fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(err) => {
            return Err(match err.kind() {
                std::io::ErrorKind::NotFound => VideoError::MissingPath(path.to_path_buf()),
                _ => VideoError::Unreadable(path.to_path_buf()),
            })
        }
    };
    if !metadata.is_file() {
        return Err(VideoError::NotAFile(path.to_path_buf()));
    }
    if metadata.len() == 0 {
        return Err(VideoError::EmptyFile(path.to_path_buf()));
    }
    File::open(path).map_err(|_| VideoError::Unreadable(path.to_path_buf()))?;
    Ok(())
}

fn open_decoder(
    stream: &format::stream::Stream,
    codec_id: codec::Id,
    prefer_hardware: bool,
) -> Result<(ffmpeg::decoder::Video, String, bool)> {
    if prefer_hardware {
        if let Some(name) = hardware_decoder_name(codec_id) {
            if let Some(codec) = ffmpeg::decoder::find_by_name(name) {
                let context = codec::context::Context::from_parameters(stream.parameters())
                    .map_err(|err| VideoError::Decoder(err.to_string()))?;
                if let Ok(opened) = context.decoder().open_as(codec) {
                    if let Ok(video) = opened.video() {
                        return Ok((video, name.to_string(), true));
                    }
                }
            }
        }
    }

    let context = codec::context::Context::from_parameters(stream.parameters())
        .map_err(|err| VideoError::Decoder(err.to_string()))?;
    let video = context
        .decoder()
        .video()
        .map_err(|err| VideoError::Decoder(err.to_string()))?;

    Ok((video, format!("{codec_id:?}"), false))
}

#[cfg(target_os = "macos")]
fn hardware_decoder_name(codec_id: codec::Id) -> Option<&'static str> {
    match codec_id {
        codec::Id::H264 => Some("h264_videotoolbox"),
        codec::Id::HEVC => Some("hevc_videotoolbox"),
        _ => None,
    }
}

#[cfg(not(target_os = "macos"))]
fn hardware_decoder_name(_codec_id: codec::Id) -> Option<&'static str> {
    None
}

fn frame_rate_from_stream(stream: &format::stream::Stream) -> f64 {
    let fps = f64::from(stream.rate());
    if fps > 1.0 {
        fps
    } else {
        30.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(label: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        path.push(format!("rustscan_video_test_{label}_{nanos}"));
        path
    }

    #[test]
    fn test_supported_containers() {
        assert!(is_supported_container("mov,mp4,m4a", "mov"));
        assert!(is_supported_container("mp4", "mp4"));
        assert!(is_supported_container("mov,mp4", "m4v"));
        assert!(!is_supported_container("matroska", "mkv"));
    }

    #[test]
    fn test_supported_codecs() {
        assert!(is_supported_codec(codec::Id::H264));
        assert!(is_supported_codec(codec::Id::HEVC));
        assert!(!is_supported_codec(codec::Id::VP9));
    }

    #[test]
    fn test_default_decoder_config_prefers_hardware() {
        let config = VideoDecoderConfig::default();
        assert!(config.prefer_hardware);
    }

    #[test]
    fn test_default_decoder_config_cache_capacity() {
        let config = VideoDecoderConfig::default();
        assert_eq!(config.cache_capacity, 100);
    }

    #[test]
    fn test_codec_label() {
        assert_eq!(codec_label(codec::Id::H264), "H.264");
        assert_eq!(codec_label(codec::Id::HEVC), "H.265/HEVC");
    }

    #[test]
    fn test_validate_file_missing() {
        let path = temp_path("missing");
        let err = validate_file(&path).unwrap_err();
        assert!(matches!(err, VideoError::MissingPath(_)));
    }

    #[test]
    fn test_validate_file_not_file() {
        let path = temp_path("dir");
        std::fs::create_dir_all(&path).unwrap();
        let err = validate_file(&path).unwrap_err();
        std::fs::remove_dir_all(&path).ok();
        assert!(matches!(err, VideoError::NotAFile(_)));
    }

    #[test]
    fn test_validate_file_empty() {
        let path = temp_path("empty");
        File::create(&path).unwrap();
        let err = validate_file(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(matches!(err, VideoError::EmptyFile(_)));
    }

    #[test]
    fn test_validate_file_readable() {
        let path = temp_path("readable");
        std::fs::write(&path, b"ok").unwrap();
        let result = validate_file(&path);
        std::fs::remove_file(&path).ok();
        assert!(result.is_ok());
    }
}
