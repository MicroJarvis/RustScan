use crate::colmap_image::load_colmap_grayscale_u8;
use crate::compare::{compare_feature_counts, FeaturesCompareReport};
use crate::database::{ColmapDatabase, ColmapDescriptors, ColmapKeypoint, COLMAP_FEATURE_SIFT};
#[cfg(feature = "gpu-wgpu")]
use crate::gpu::WgpuSiftExtractor;
use crate::sift::{extract_sift_from_grayscale_u8, SiftExtractionOptions, SiftFeatures};
use crate::task::{
    SfmTaskContext, SfmTaskControl, SfmTaskEvent, SfmTaskEventKind, SfmTaskOperation, SfmTaskStage,
};
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractFeaturesImageReport {
    pub image_name: String,
    pub num_keypoints: usize,
    pub elapsed_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractFeaturesReport {
    pub database: PathBuf,
    pub images_dir: PathBuf,
    pub backend: &'static str,
    pub image_count: usize,
    pub total_keypoints: usize,
    pub mean_keypoints: f64,
    pub extraction_seconds: f64,
    pub images: Vec<ExtractFeaturesImageReport>,
}

pub trait SiftFeatureExtractor {
    fn backend_name(&self) -> &'static str;

    fn extract_grayscale(
        &self,
        gray: &[u8],
        width: u32,
        height: u32,
        options: &SiftExtractionOptions,
    ) -> Result<SiftFeatures>;
}

struct CpuSiftExtractor;

impl SiftFeatureExtractor for CpuSiftExtractor {
    fn backend_name(&self) -> &'static str {
        sift_backend_name(&SiftExtractionOptions::default())
    }

    fn extract_grayscale(
        &self,
        gray: &[u8],
        width: u32,
        height: u32,
        options: &SiftExtractionOptions,
    ) -> Result<SiftFeatures> {
        extract_sift_from_grayscale_u8(gray, width, height, options)
    }
}

#[cfg(feature = "gpu-wgpu")]
impl SiftFeatureExtractor for WgpuSiftExtractor {
    fn backend_name(&self) -> &'static str {
        "wgpu"
    }

    fn extract_grayscale(
        &self,
        gray: &[u8],
        width: u32,
        height: u32,
        options: &SiftExtractionOptions,
    ) -> Result<SiftFeatures> {
        self.extract_grayscale(gray, width, height, options)
    }
}

enum SiftExtractionBackend {
    Cpu(CpuSiftExtractor),
    #[cfg(feature = "gpu-wgpu")]
    Wgpu(WgpuSiftExtractor),
}

impl SiftExtractionBackend {
    fn from_options(options: &SiftExtractionOptions) -> Result<Self> {
        if options.use_gpu {
            #[cfg(feature = "gpu-wgpu")]
            {
                return Ok(Self::Wgpu(WgpuSiftExtractor::try_new()?));
            }
            #[cfg(not(feature = "gpu-wgpu"))]
            {
                bail!("RustSFM was built without gpu-wgpu support");
            }
        }
        Ok(Self::Cpu(CpuSiftExtractor))
    }
}

impl SiftFeatureExtractor for SiftExtractionBackend {
    fn backend_name(&self) -> &'static str {
        match self {
            Self::Cpu(extractor) => extractor.backend_name(),
            #[cfg(feature = "gpu-wgpu")]
            Self::Wgpu(extractor) => extractor.backend_name(),
        }
    }

    fn extract_grayscale(
        &self,
        gray: &[u8],
        width: u32,
        height: u32,
        options: &SiftExtractionOptions,
    ) -> Result<SiftFeatures> {
        match self {
            Self::Cpu(extractor) => extractor.extract_grayscale(gray, width, height, options),
            #[cfg(feature = "gpu-wgpu")]
            Self::Wgpu(extractor) => extractor.extract_grayscale(gray, width, height, options),
        }
    }
}

pub fn sift_features_to_colmap_keypoints(features: &SiftFeatures) -> Vec<ColmapKeypoint> {
    if !features.colmap_keypoints.is_empty() {
        return features.colmap_keypoints.clone();
    }
    features
        .keypoints
        .iter()
        .map(ColmapKeypoint::from)
        .collect()
}

pub fn sift_features_to_colmap_descriptors(features: &SiftFeatures) -> Result<ColmapDescriptors> {
    const DESCRIPTOR_LEN: usize = 128;
    let rows = features.descriptors_u8.len();
    let data = features
        .descriptors_u8
        .iter()
        .flat_map(|descriptor| descriptor.iter().copied())
        .collect::<Vec<_>>();
    ColmapDescriptors::new(COLMAP_FEATURE_SIFT, rows, DESCRIPTOR_LEN, data)
}

pub fn extract_features_to_database(
    database_path: &Path,
    images_dir: &Path,
    options: &SiftExtractionOptions,
) -> Result<ExtractFeaturesReport> {
    let backend = SiftExtractionBackend::from_options(options)?;
    extract_features_to_database_with_extractor(database_path, images_dir, options, &backend)
}

pub fn extract_features_to_database_with_extractor<E: SiftFeatureExtractor>(
    database_path: &Path,
    images_dir: &Path,
    options: &SiftExtractionOptions,
    extractor: &E,
) -> Result<ExtractFeaturesReport> {
    let control = SfmTaskControl::new();
    let mut sink = |_event: SfmTaskEvent| {};
    let mut task = SfmTaskContext::new(&control, &mut sink);
    extract_features_to_database_with_extractor_and_task(
        database_path,
        images_dir,
        options,
        extractor,
        &mut task,
    )
}

pub fn extract_features_to_database_with_extractor_and_task<E: SiftFeatureExtractor>(
    database_path: &Path,
    images_dir: &Path,
    options: &SiftExtractionOptions,
    extractor: &E,
    task: &mut SfmTaskContext<'_>,
) -> Result<ExtractFeaturesReport> {
    options.check()?;
    let db = ColmapDatabase::open(database_path)?;
    let mut images = db.read_all_images()?;
    if images.is_empty() {
        bail!("database has no images; import images before feature extraction");
    }
    images.sort_unstable_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| left.image_id.cmp(&right.image_id))
    });

    let started = Instant::now();
    let image_count = images.len();
    let mut reports = Vec::with_capacity(images.len());
    for image in images {
        task.checkpoint()?;
        let image_path = images_dir.join(&image.name);
        if !image_path.exists() {
            bail!(
                "missing image file for database image {}: {}",
                image.name,
                image_path.display()
            );
        }
        let extract_started = Instant::now();
        let decoded = load_colmap_grayscale_u8(&image_path)
            .with_context(|| format!("failed to load {}", image_path.display()))?;
        let features =
            extractor.extract_grayscale(&decoded.data, decoded.width, decoded.height, options)?;
        let keypoints = sift_features_to_colmap_keypoints(&features);
        let descriptors = sift_features_to_colmap_descriptors(&features)?;
        db.with_transaction(|| {
            db.upsert_keypoints(image.image_id, &keypoints)?;
            db.upsert_descriptors(image.image_id, &descriptors)?;
            Ok(())
        })?;
        reports.push(ExtractFeaturesImageReport {
            image_name: image.name.clone(),
            num_keypoints: keypoints.len(),
            elapsed_ms: extract_started.elapsed().as_secs_f64() * 1000.0,
        });
        task.emit(SfmTaskEvent {
            sequence: 0,
            elapsed_ms: 0,
            stage: SfmTaskStage::FeatureExtraction,
            operation: SfmTaskOperation::ExtractImage,
            kind: SfmTaskEventKind::Progress,
            completed: Some(reports.len()),
            total: Some(image_count),
            registered_images: None,
            sparse_points: None,
            image_id: Some(image.image_id),
            pair: None,
            message: Some(image.name.clone()),
            issue: None,
        });
        task.checkpoint()?;
    }
    reports.sort_unstable_by(|left, right| left.image_name.cmp(&right.image_name));

    let total_keypoints = reports
        .iter()
        .map(|image| image.num_keypoints)
        .sum::<usize>();
    let image_count = reports.len();
    let mean_keypoints = if image_count == 0 {
        0.0
    } else {
        total_keypoints as f64 / image_count as f64
    };

    Ok(ExtractFeaturesReport {
        database: database_path.to_path_buf(),
        images_dir: images_dir.to_path_buf(),
        backend: extractor.backend_name(),
        image_count,
        total_keypoints,
        mean_keypoints,
        extraction_seconds: started.elapsed().as_secs_f64(),
        images: reports,
    })
}

pub fn compare_extracted_sift_features(
    reference_database: &Path,
    images_dir: &Path,
    options: &SiftExtractionOptions,
) -> Result<FeaturesCompareReport> {
    options.check()?;
    let reference = reference_keypoint_counts(reference_database)?;
    let candidate = extracted_keypoint_counts(images_dir, options)?;
    compare_feature_counts(&reference, &candidate)
}

fn reference_keypoint_counts(database_path: &Path) -> Result<HashMap<String, usize>> {
    let db = ColmapDatabase::open_read_only(database_path)?;
    db.read_keypoint_counts()?
        .into_iter()
        .map(|(image_id, count)| {
            let image = db
                .read_image(image_id)?
                .with_context(|| format!("missing image_id={image_id}"))?;
            Ok((image.name, count))
        })
        .collect()
}

fn extracted_keypoint_counts(
    images_dir: &Path,
    options: &SiftExtractionOptions,
) -> Result<HashMap<String, usize>> {
    let mut paths = std::fs::read_dir(images_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| {
                    matches!(
                        ext.to_ascii_lowercase().as_str(),
                        "jpg" | "jpeg" | "png" | "bmp" | "tif" | "tiff" | "webp"
                    )
                })
        })
        .collect::<Vec<_>>();
    paths.sort();

    paths
        .par_iter()
        .map(|path| -> Result<(String, usize)> {
            let name = path
                .file_name()
                .map(|value| value.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string());
            let decoded = load_colmap_grayscale_u8(path)
                .with_context(|| format!("failed to load {}", path.display()))?;
            let features = extract_sift_from_grayscale_u8(
                &decoded.data,
                decoded.width,
                decoded.height,
                options,
            )?;
            Ok((name, features.keypoints.len()))
        })
        .collect()
}

fn sift_backend_name(options: &SiftExtractionOptions) -> &'static str {
    if options.use_gpu {
        return "wgpu";
    }
    if cfg!(all(
        feature = "vlfeat-sift",
        not(feature = "lowe-sift-backend")
    )) {
        "vlfeat"
    } else {
        "lowe-sift"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colmap::ColmapCamera;
    use crate::database::{ColmapDatabase, ColmapDatabaseCamera, ColmapDatabaseImage};
    #[cfg(feature = "gpu-wgpu")]
    use crate::gpu::{WgpuContext, WgpuSiftExtractor};
    use crate::types::COLMAP_PINHOLE;
    use tempfile::tempdir;

    #[test]
    fn extraction_backend_name_reports_wgpu_for_gpu_options() {
        let options = SiftExtractionOptions {
            use_gpu: true,
            ..Default::default()
        };
        assert_eq!(sift_backend_name(&options), "wgpu");
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn gpu_database_extraction_reuses_one_backend() -> Result<()> {
        let Some(context) = WgpuContext::try_new_optional()? else {
            eprintln!("skipping GPU database test: no compatible adapter");
            return Ok(());
        };
        let dir = tempdir()?;
        let images_dir = dir.path().join("images");
        std::fs::create_dir_all(&images_dir)?;
        write_checkerboard_image(&images_dir.join("left.jpg"), 256, 256)?;
        write_checkerboard_image(&images_dir.join("right.jpg"), 256, 256)?;
        let db_path = dir.path().join("database.db");
        let db = ColmapDatabase::open(&db_path)?;
        db.write_camera(
            &ColmapDatabaseCamera {
                camera: ColmapCamera {
                    camera_id: 1,
                    model_id: COLMAP_PINHOLE,
                    width: 256,
                    height: 256,
                    params: vec![200.0, 200.0, 128.0, 128.0],
                },
                has_prior_focal_length: true,
            },
            true,
        )?;
        for (image_id, name) in [(1, "left.jpg"), (2, "right.jpg")] {
            db.write_image(
                &ColmapDatabaseImage {
                    image_id,
                    name: name.to_string(),
                    camera_id: 1,
                    frame_id: None,
                },
                true,
            )?;
        }
        let extractor = WgpuSiftExtractor::from_context(context)?;
        let report = extract_features_to_database_with_extractor(
            &db_path,
            &images_dir,
            &SiftExtractionOptions {
                use_gpu: true,
                max_num_features: 256,
                ..Default::default()
            },
            &extractor,
        )?;
        assert_eq!(report.backend, "wgpu");
        assert_eq!(report.image_count, 2);
        assert!(report.total_keypoints > 0);
        Ok(())
    }

    #[test]
    fn extract_features_to_database_updates_existing_rows() -> Result<()> {
        let dir = tempdir()?;
        let images_dir = dir.path().join("images");
        std::fs::create_dir_all(&images_dir)?;
        let image_path = images_dir.join("left.jpg");
        write_checkerboard_image(&image_path, 256, 256)?;

        let db_path = dir.path().join("database.db");
        let db = ColmapDatabase::open(&db_path)?;
        db.write_camera(
            &ColmapDatabaseCamera {
                camera: ColmapCamera {
                    camera_id: 1,
                    model_id: COLMAP_PINHOLE,
                    width: 256,
                    height: 256,
                    params: vec![200.0, 200.0, 128.0, 128.0],
                },
                has_prior_focal_length: true,
            },
            true,
        )?;
        db.write_image(
            &ColmapDatabaseImage {
                image_id: 1,
                name: "left.jpg".to_string(),
                camera_id: 1,
                frame_id: None,
            },
            true,
        )?;
        db.write_keypoints(1, &[ColmapKeypoint::new(1.0, 1.0)])?;
        db.write_descriptors(
            1,
            &ColmapDescriptors::new(COLMAP_FEATURE_SIFT, 1, 128, vec![0u8; 128])?,
        )?;

        let report = extract_features_to_database(
            &db_path,
            &images_dir,
            &SiftExtractionOptions {
                max_num_features: 256,
                ..SiftExtractionOptions::default()
            },
        )?;
        assert_eq!(report.image_count, 1);
        assert!(report.total_keypoints > 1);

        let updated = db.read_keypoints(1)?;
        assert_eq!(updated.len(), report.total_keypoints);
        let descriptors = db.read_descriptors(1)?;
        assert_eq!(descriptors.rows, report.total_keypoints);
        Ok(())
    }

    #[test]
    fn controlled_extraction_pauses_after_committing_one_image() -> Result<()> {
        use crate::task::{
            SfmTaskControl, SfmTaskEvent, SfmTaskEventKind, SfmTaskOperation, SfmTaskStage,
            SfmTaskStop,
        };

        let (_dir, db_path, images_dir) = two_image_fixture()?;
        let extractor = DeterministicExtractor;
        let control = SfmTaskControl::new();
        let sink_control = control.clone();
        let mut events = Vec::new();
        let mut sink = |event: SfmTaskEvent| {
            if event.operation == SfmTaskOperation::ExtractImage && event.completed == Some(1) {
                sink_control.request_pause();
            }
            events.push(event);
        };
        let mut task = crate::task::SfmTaskContext::new(&control, &mut sink);
        let error = extract_features_to_database_with_extractor_and_task(
            &db_path,
            &images_dir,
            &SiftExtractionOptions::default(),
            &extractor,
            &mut task,
        )
        .expect_err("pause requested from the first progress event");
        assert_eq!(
            error.downcast_ref::<SfmTaskStop>(),
            Some(&SfmTaskStop::Paused)
        );

        let db = ColmapDatabase::open(&db_path)?;
        assert!(db.exists_keypoints(2)?);
        assert!(db.exists_descriptors(2)?);
        assert!(!db.exists_keypoints(1)?);
        assert!(!db.exists_descriptors(1)?);
        assert_eq!(db.read_keypoints(2)?.len(), 1);
        assert_eq!(db.read_descriptors(2)?.rows, 1);
        let event = events.last().expect("first image progress event");
        assert_eq!(event.stage, SfmTaskStage::FeatureExtraction);
        assert_eq!(event.operation, SfmTaskOperation::ExtractImage);
        assert_eq!(event.kind, SfmTaskEventKind::Progress);
        assert_eq!(event.completed, Some(1));
        assert_eq!(event.total, Some(2));
        assert_eq!(event.image_id, Some(2));
        assert_eq!(event.message.as_deref(), Some("left.jpg"));
        Ok(())
    }

    #[test]
    fn controlled_extraction_honors_pre_requested_cancellation() -> Result<()> {
        use crate::task::{SfmTaskControl, SfmTaskEvent, SfmTaskStop};

        let (_dir, db_path, images_dir) = two_image_fixture()?;
        let extractor = DeterministicExtractor;
        let control = SfmTaskControl::new();
        control.request_cancel();
        let mut events = Vec::new();
        let mut sink = |event: SfmTaskEvent| events.push(event);
        let mut task = crate::task::SfmTaskContext::new(&control, &mut sink);
        let error = extract_features_to_database_with_extractor_and_task(
            &db_path,
            &images_dir,
            &SiftExtractionOptions::default(),
            &extractor,
            &mut task,
        )
        .expect_err("pre-requested cancellation");
        assert_eq!(
            error.downcast_ref::<SfmTaskStop>(),
            Some(&SfmTaskStop::Cancelled)
        );

        let db = ColmapDatabase::open(&db_path)?;
        for image_id in [1, 2] {
            assert!(!db.exists_keypoints(image_id)?);
            assert!(!db.exists_descriptors(image_id)?);
        }
        assert!(events.is_empty());
        Ok(())
    }

    #[test]
    fn controlled_extraction_checks_cancellation_before_missing_file_validation() -> Result<()> {
        use crate::task::{SfmTaskControl, SfmTaskEvent, SfmTaskStop};

        let (_dir, db_path, images_dir) = two_image_fixture()?;
        std::fs::remove_file(images_dir.join("right.jpg"))?;
        let extractor = DeterministicExtractor;
        let control = SfmTaskControl::new();
        control.request_cancel();
        let mut events = Vec::new();
        let mut sink = |event: SfmTaskEvent| events.push(event);
        let mut task = crate::task::SfmTaskContext::new(&control, &mut sink);
        let error = extract_features_to_database_with_extractor_and_task(
            &db_path,
            &images_dir,
            &SiftExtractionOptions::default(),
            &extractor,
            &mut task,
        )
        .expect_err("pre-requested cancellation");
        assert_eq!(
            error.downcast_ref::<SfmTaskStop>(),
            Some(&SfmTaskStop::Cancelled)
        );
        assert!(events.is_empty());
        Ok(())
    }

    #[test]
    fn controlled_extraction_rolls_back_keypoints_when_descriptor_upsert_fails() -> Result<()> {
        use crate::task::SfmTaskEvent;
        use rusqlite::Connection;

        let (_dir, db_path, images_dir) = two_image_fixture()?;
        let trigger_connection = Connection::open(&db_path)?;
        trigger_connection.execute_batch(
            "CREATE TRIGGER fail_descriptor_insert
             BEFORE INSERT ON descriptors
             WHEN NEW.image_id = 2
             BEGIN
                 SELECT RAISE(ABORT, 'descriptor insert failed');
             END;",
        )?;
        let extractor = DeterministicExtractor;
        let control = SfmTaskControl::new();
        let mut sink = |_event: SfmTaskEvent| {};
        let mut task = crate::task::SfmTaskContext::new(&control, &mut sink);
        let error = extract_features_to_database_with_extractor_and_task(
            &db_path,
            &images_dir,
            &SiftExtractionOptions::default(),
            &extractor,
            &mut task,
        )
        .expect_err("descriptor trigger failure");
        assert!(error.to_string().contains("descriptor insert failed"));

        let db = ColmapDatabase::open(&db_path)?;
        assert!(!db.exists_keypoints(2)?);
        assert!(!db.exists_descriptors(2)?);
        Ok(())
    }

    struct DeterministicExtractor;

    impl SiftFeatureExtractor for DeterministicExtractor {
        fn backend_name(&self) -> &'static str {
            "deterministic-test"
        }

        fn extract_grayscale(
            &self,
            _gray: &[u8],
            _width: u32,
            _height: u32,
            _options: &SiftExtractionOptions,
        ) -> Result<SiftFeatures> {
            Ok(SiftFeatures {
                colmap_keypoints: vec![ColmapKeypoint::new(1.0, 1.0)],
                descriptors_u8: vec![[7u8; 128]],
                ..SiftFeatures::default()
            })
        }
    }

    fn two_image_fixture() -> Result<(tempfile::TempDir, PathBuf, PathBuf)> {
        let dir = tempdir()?;
        let images_dir = dir.path().join("images");
        std::fs::create_dir_all(&images_dir)?;
        write_checkerboard_image(&images_dir.join("left.jpg"), 32, 32)?;
        write_checkerboard_image(&images_dir.join("right.jpg"), 32, 32)?;
        let db_path = dir.path().join("database.db");
        let db = ColmapDatabase::open(&db_path)?;
        db.write_camera(
            &ColmapDatabaseCamera {
                camera: ColmapCamera {
                    camera_id: 1,
                    model_id: COLMAP_PINHOLE,
                    width: 32,
                    height: 32,
                    params: vec![20.0, 20.0, 16.0, 16.0],
                },
                has_prior_focal_length: true,
            },
            true,
        )?;
        for (image_id, name) in [(1, "right.jpg"), (2, "left.jpg")] {
            db.write_image(
                &ColmapDatabaseImage {
                    image_id,
                    name: name.to_string(),
                    camera_id: 1,
                    frame_id: None,
                },
                true,
            )?;
        }
        Ok((dir, db_path, images_dir))
    }

    fn write_checkerboard_image(path: &Path, width: u32, height: u32) -> Result<()> {
        let mut image = image::RgbImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let value = if ((x / 16) + (y / 16)) % 2 == 0 {
                    240u8
                } else {
                    20u8
                };
                image.put_pixel(x, y, image::Rgb([value, value, value]));
            }
        }
        image.save(path)?;
        Ok(())
    }
}
