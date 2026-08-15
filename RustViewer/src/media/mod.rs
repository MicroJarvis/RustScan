#[cfg(target_os = "macos")]
mod avfoundation;
mod images;
mod keyframes;
mod video;

#[cfg(target_os = "macos")]
pub use avfoundation::AvFoundationVideoDecoder;
pub use images::{
    import_image_sequence, ImageSequenceImportRequest, ImportResult, ImportedFrame, MediaEventSink,
    MediaImportError, MediaImportEvent,
};
pub use keyframes::{select_keyframes, KeyframeSelectionConfig, KeyframeSelectionError};
pub use video::{import_video, DecodedVideoFrame, VideoDecoder, VideoMetadata};
