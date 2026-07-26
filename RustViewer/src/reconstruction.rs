use anyhow::{bail, ensure, Context};
use std::{
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageSource {
    root: PathBuf,
    images: Vec<PathBuf>,
}

impl ImageSource {
    pub fn open(root: impl AsRef<Path>) -> anyhow::Result<Self> {
        let root = root.as_ref().to_path_buf();
        let mut images = Vec::new();

        for entry in fs::read_dir(&root)
            .with_context(|| format!("failed to read image directory {}", root.display()))?
        {
            let entry = entry?;
            if !entry.file_type()?.is_file() || !is_supported_image(&entry.path()) {
                continue;
            }
            images.push(entry.path());
        }

        images.sort();
        if images.len() < 2 {
            bail!(
                "{} must contain at least two JPEG or PNG images",
                root.display()
            );
        }

        Ok(Self { root, images })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn image_count(&self) -> usize {
        self.images.len()
    }

    pub fn images(&self) -> &[PathBuf] {
        &self.images
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconstructionProgress {
    pub registered_images: usize,
    pub registered_frames: usize,
    pub points: usize,
    pub stage: &'static str,
}

#[derive(Debug, Clone)]
pub struct ReconstructionOutput {
    pub output_dir: PathBuf,
    pub summary: rustsfm::ReconstructionSummary,
}

pub trait ReconstructionRunner: Send + 'static {
    fn run(
        &self,
        source: &ImageSource,
        output_dir: PathBuf,
        emit: &mut dyn FnMut(ReconstructionProgress),
    ) -> anyhow::Result<ReconstructionOutput>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RustSfmRunner;

impl ReconstructionRunner for RustSfmRunner {
    fn run(
        &self,
        source: &ImageSource,
        output_dir: PathBuf,
        emit: &mut dyn FnMut(ReconstructionProgress),
    ) -> anyhow::Result<ReconstructionOutput> {
        let config = mapper_config_for(source.root(), &output_dir);
        let mut sink = ProgressSink { emit };
        let summary = rustsfm::run_reconstruction_with_callbacks(&config, Some(&mut sink))?;
        validate_completed_summary(&summary)?;

        Ok(ReconstructionOutput {
            output_dir,
            summary,
        })
    }
}

pub fn create_run_directory(input: &Path) -> anyhow::Result<PathBuf> {
    let root = input.join(".rustviewer/rustsfm");
    fs::create_dir_all(&root)
        .with_context(|| format!("failed to create RustSFM output root {}", root.display()))?;

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_nanos();
    let mut counter = 0_u64;

    loop {
        let output_dir = root.join(format!("run-{nanos}-{counter}"));
        match fs::create_dir(&output_dir) {
            Ok(()) => return Ok(output_dir),
            Err(error) if error.kind() == ErrorKind::AlreadyExists => {
                counter = counter
                    .checked_add(1)
                    .context("exhausted RustSFM run directory counter")?;
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "failed to create RustSFM output directory {}",
                        output_dir.display()
                    )
                })
            }
        }
    }
}

pub fn mapper_config_for(input: &Path, output: &Path) -> rustsfm::MapperConfig {
    rustsfm::MapperConfig {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
        local_matching: true,
        single_camera: true,
        copy_images: true,
        ..rustsfm::MapperConfig::default()
    }
}

pub fn validate_completed_summary(summary: &rustsfm::ReconstructionSummary) -> anyhow::Result<()> {
    ensure!(
        summary.registered_images > 0,
        "RustSFM completed without registered images"
    );
    ensure!(
        summary.points > 0,
        "RustSFM completed without sparse points"
    );
    Ok(())
}

struct ProgressSink<'a> {
    emit: &'a mut dyn FnMut(ReconstructionProgress),
}

impl rustsfm::PipelineCallbackSink for ProgressSink<'_> {
    fn on_pipeline_callback(&mut self, event: &rustsfm::PipelineCallbackEvent) {
        (self.emit)(ReconstructionProgress {
            registered_images: event.registered_images,
            registered_frames: event.registered_frames,
            points: event.points,
            stage: event.callback.as_str(),
        });
    }
}

fn is_supported_image(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("jpg" | "jpeg" | "png")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn image_source_requires_two_supported_images() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("only.jpg"), b"image").unwrap();
        let error = ImageSource::open(temp.path()).unwrap_err();
        assert!(error
            .to_string()
            .contains("at least two JPEG or PNG images"));
    }

    #[test]
    fn run_directory_is_unique_and_stays_below_the_source() {
        let temp = tempfile::tempdir().unwrap();
        let first = create_run_directory(temp.path()).unwrap();
        let second = create_run_directory(temp.path()).unwrap();
        assert_ne!(first, second);
        assert!(first.starts_with(temp.path().join(".rustviewer/rustsfm")));
    }

    #[test]
    fn mapper_config_enables_shared_camera_local_matching_and_image_copying() {
        let input = PathBuf::from("/captures/set-a");
        let output = PathBuf::from("/captures/set-a/.rustviewer/rustsfm/run-1");
        let config = mapper_config_for(&input, &output);
        assert_eq!(config.input, input);
        assert_eq!(config.output, output);
        assert!(config.local_matching && config.single_camera && config.copy_images);
        assert!(config.fx.is_none() && config.fy.is_none());
    }

    #[test]
    fn completed_summary_requires_registered_images_and_sparse_points() {
        let empty = rustsfm::ReconstructionSummary {
            images: 12,
            registered_images: 0,
            points: 0,
            pairs: 0,
            models: 0,
            elapsed_ms: 1.0,
            debug_log: Vec::new(),
        };
        assert!(validate_completed_summary(&empty).is_err());
    }

    #[test]
    fn completed_summary_accepts_registered_sparse_model() {
        let complete = rustsfm::ReconstructionSummary {
            images: 12,
            registered_images: 8,
            points: 120,
            pairs: 24,
            models: 1,
            elapsed_ms: 1.0,
            debug_log: Vec::new(),
        };
        assert!(validate_completed_summary(&complete).is_ok());
    }
}
