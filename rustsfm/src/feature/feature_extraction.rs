use crate::colmap_image::load_colmap_grayscale_u8;
use crate::compare::{compare_feature_counts, FeaturesCompareReport};
use crate::database::{
    ColmapDatabase, ColmapDatabaseImage, ColmapDescriptors, ColmapKeypoint, COLMAP_FEATURE_SIFT,
};
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

#[path = "feature_memory.rs"]
mod memory;
// Composition-only facade: callers select missing DB inputs before planning.
pub(crate) use memory::{db_memory_plan, retained_memory_plan, FeatureMemoryPlan};
#[path = "feature_taskflow.rs"]
mod taskflow;
pub use taskflow::CpuFeatureTaskflow;

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
    let control = SfmTaskControl::new();
    let mut sink = |_| {};
    let mut task = SfmTaskContext::new(&control, &mut sink);
    extract_features_to_database_with_task(database_path, images_dir, options, &mut task)
}

/// Binding SfmTaskflow or CpuFeatureTaskflow explicitly opts standard VLFeat
/// into automatic image/batch allocation planning. Without a binding, including
/// the no-task API, the legacy default estimate and behavior are retained: there
/// is NO automatic image working-set guarantee. Other backends and custom
/// extractors always rely on the caller/default estimate contract, not a proven
/// conservative formula. No algorithm parameters or public defaults are changed.
pub fn extract_features_to_database_with_task(
    database_path: &Path,
    images_dir: &Path,
    options: &SiftExtractionOptions,
    task: &mut SfmTaskContext<'_>,
) -> Result<ExtractFeaturesReport> {
    task.checkpoint()?;
    if !memory::standard(options) && !task.has_feature_memory_estimate() {
        log::warn!("non-standard SIFT backend uses the existing default stage allocation estimate; no automatic conservative memory formula is available. Configure SfmTaskflow/CpuFeatureTaskflow to cover this workload's decode, backend scratch and output");
    }
    if crate::execution::active_threads().is_none()
        && (options.use_gpu || !task.has_feature_memory_estimate())
    {
        return task.execute("feature extraction", options.use_gpu, 4, |task| {
            extract_features_to_database_with_task(database_path, images_dir, options, task)
        });
    }
    let backend = SiftExtractionBackend::from_options(options)?;
    match &backend {
        SiftExtractionBackend::Cpu(extractor) => {
            extract_features_to_database_cpu_parallel_with_task(
                database_path,
                images_dir,
                options,
                extractor,
                task,
            )
        }
        #[cfg(feature = "gpu-wgpu")]
        SiftExtractionBackend::Wgpu(extractor) => {
            extract_features_to_database_with_extractor_and_task(
                database_path,
                images_dir,
                options,
                extractor,
                task,
            )
        }
    }
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

/// Custom extractors retain the caller-estimate contract: configure SfmTaskflow
/// with an estimate covering decode, backend scratch and retained output. No
/// standard-backend formula is inferred from an extractor's name or public trait.
pub fn extract_features_to_database_with_extractor_and_task<E: SiftFeatureExtractor>(
    database_path: &Path,
    images_dir: &Path,
    options: &SiftExtractionOptions,
    extractor: &E,
    task: &mut SfmTaskContext<'_>,
) -> Result<ExtractFeaturesReport> {
    extract_selected_features_to_database_with_extractor_and_task(
        database_path,
        images_dir,
        options,
        &[],
        extractor,
        task,
    )
}

fn extract_features_to_database_cpu_parallel_with_task(
    database_path: &Path,
    images_dir: &Path,
    options: &SiftExtractionOptions,
    extractor: &CpuSiftExtractor,
    task: &mut SfmTaskContext<'_>,
) -> Result<ExtractFeaturesReport> {
    extract_selected_features_to_database_cpu_parallel_with_task(
        database_path,
        images_dir,
        options,
        &[],
        extractor,
        task,
    )
}

pub(crate) fn extract_selected_features_to_database_with_task(
    database_path: &Path,
    images_dir: &Path,
    options: &SiftExtractionOptions,
    image_ids: &[u32],
    task: &mut SfmTaskContext<'_>,
) -> Result<ExtractFeaturesReport> {
    task.checkpoint()?;
    if !memory::standard(options) && !task.has_feature_memory_estimate() {
        log::warn!("non-standard SIFT backend uses the existing default stage allocation estimate; no automatic conservative memory formula is available. Configure SfmTaskflow/CpuFeatureTaskflow to cover this workload's decode, backend scratch and output");
    }
    if crate::execution::active_threads().is_none()
        && (options.use_gpu || !task.has_feature_memory_estimate())
    {
        return task.execute("selected feature extraction", options.use_gpu, 4, |task| {
            extract_selected_features_to_database_with_task(
                database_path,
                images_dir,
                options,
                image_ids,
                task,
            )
        });
    }
    let backend = SiftExtractionBackend::from_options(options)?;
    match &backend {
        SiftExtractionBackend::Cpu(extractor) => {
            extract_selected_features_to_database_cpu_parallel_with_task(
                database_path,
                images_dir,
                options,
                image_ids,
                extractor,
                task,
            )
        }
        #[cfg(feature = "gpu-wgpu")]
        SiftExtractionBackend::Wgpu(extractor) => {
            extract_selected_features_to_database_with_extractor_and_task(
                database_path,
                images_dir,
                options,
                image_ids,
                extractor,
                task,
            )
        }
    }
}

/// Execute a previously admitted/planned selection without recomputing chunks.
/// `planned_images` is exactly one (DB id, original plan path) per plan row,
/// sorted by DB (name, image_id), NOT caller ID order. Staged files may be created
/// after planning: their paths are images_dir.join(DB name), validated against
/// the original header dimensions/encoded length before any feature writes.
/// Empty selection is rejected (unlike the legacy empty-ID = all-images API).
/// Plans must cover this exact extraction call; no subset/regrouping is allowed.
pub(crate) fn extract_selected_features_to_database_with_plan_and_task(
    database_path: &Path,
    images_dir: &Path,
    options: &SiftExtractionOptions,
    planned_images: &[(u32, PathBuf)],
    plan: &FeatureMemoryPlan,
    task: &mut SfmTaskContext<'_>,
) -> Result<ExtractFeaturesReport> {
    task.checkpoint()?;
    // Validate merged runtime bindings even when a CPU image adapter is bound.
    let _ = task.feature_memory_limits()?;
    let db = ColmapDatabase::open_read_only(database_path)?;
    let images = validate_planned_selection(&db, planned_images, plan)?;
    let staged_paths = images
        .iter()
        .map(|image| images_dir.join(&image.name))
        .collect::<Vec<_>>();
    let rebound = plan.rebind_db_paths(&staged_paths, options, &task.control())?;
    drop(db);
    let threads = crate::execution::active_threads().unwrap_or(4);
    task.execute_with_memory(
        "selected feature extraction",
        plan.request_bytes(),
        threads,
        |task| {
            // A real outer admission (or inherited parent) suppresses image-adapter
            // DAG submission. Always use the immutable plan's chunks in this scope.
            anyhow::ensure!(
                crate::execution::active_threads().is_some(),
                "planned extraction requires an active stage"
            );
            let db = ColmapDatabase::open(database_path)?;
            let current = validate_planned_selection(&db, planned_images, plan)?;
            let current_paths = current
                .iter()
                .map(|image| images_dir.join(&image.name))
                .collect::<Vec<_>>();
            anyhow::ensure!(
                current_paths == rebound.paths(),
                "planned selected DB paths changed while awaiting admission"
            );
            // Recheck headers after queueing too; files may have changed meanwhile.
            plan.rebind_db_paths(&current_paths, options, &task.control())?;
            extract_cpu_images(
                &db,
                &current,
                database_path,
                images_dir,
                rebound.options(),
                &CpuSiftExtractor,
                rebound.batch_size(),
                task,
            )
        },
    )
}

fn validate_planned_selection(
    db: &ColmapDatabase,
    planned_images: &[(u32, PathBuf)],
    plan: &FeatureMemoryPlan,
) -> Result<Vec<ColmapDatabaseImage>> {
    anyhow::ensure!(
        !planned_images.is_empty(),
        "planned selected extraction requires a nonempty explicit selection"
    );
    anyhow::ensure!(
        planned_images.len() == plan.paths().len(),
        "planned selected extraction mapping length differs from the memory plan"
    );
    let ids = planned_images.iter().map(|(id, _)| *id).collect::<Vec<_>>();
    let images = read_selected_feature_images(db, &ids)?;
    for ((image, (id, source)), planned_path) in images.iter().zip(planned_images).zip(plan.paths())
    {
        anyhow::ensure!(image.image_id == *id && source == planned_path,
            "planned selected extraction mapping must match DB name/id order and original plan path order");
        // Do not allow DB names to redirect staged IO outside the input directory.
        anyhow::ensure!(
            Path::new(&image.name)
                .components()
                .all(|part| matches!(part, std::path::Component::Normal(_)))
                && !image.name.is_empty(),
            "planned selected extraction requires a relative DB image name"
        );
    }
    Ok(images)
}

fn read_selected_feature_images(
    db: &ColmapDatabase,
    image_ids: &[u32],
) -> Result<Vec<ColmapDatabaseImage>> {
    let mut images = db.read_all_images()?;
    if images.is_empty() {
        bail!("database has no images; import images before feature extraction");
    }
    images.sort_unstable_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| left.image_id.cmp(&right.image_id))
    });
    if image_ids.is_empty() {
        return Ok(images);
    }
    let selected = image_ids
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    if selected.len() != image_ids.len() {
        bail!("selected feature extraction image IDs must be unique");
    }
    let available = images
        .iter()
        .map(|image| image.image_id)
        .collect::<std::collections::BTreeSet<_>>();
    if let Some(missing) = selected.difference(&available).next() {
        bail!("selected feature extraction references missing image_id={missing}");
    }
    images.retain(|image| selected.contains(&image.image_id));
    Ok(images)
}

fn extract_selected_features_to_database_cpu_parallel_with_task(
    database_path: &Path,
    images_dir: &Path,
    options: &SiftExtractionOptions,
    image_ids: &[u32],
    extractor: &CpuSiftExtractor,
    task: &mut SfmTaskContext<'_>,
) -> Result<ExtractFeaturesReport> {
    task.checkpoint()?;
    options.check()?;
    // Preserve binding validation before any database creation or mutation.
    let image_taskflow = task.cpu_feature_taskflow()?;
    let stage_limits = if image_taskflow.is_none() && task.has_feature_memory_estimate() {
        Some(task.feature_memory_limits()?)
    } else {
        None
    };
    let db = ColmapDatabase::open(database_path)?;
    let images = read_selected_feature_images(&db, image_ids)?;
    if let Some((floor, budget)) = stage_limits {
        let control = task.control();
        let estimates = images
            .iter()
            .map(|image| {
                // A whole-stage caller estimate is a floor on the aggregate, not
                // a per-image multiplier. Unsupported backends run one image at a time.
                memory::image_estimate(
                    &images_dir.join(&image.name),
                    options,
                    if memory::standard(options) { 0 } else { floor },
                    &control,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let threads = if memory::standard(options) {
            crate::execution::active_threads().unwrap_or(4)
        } else {
            1
        };
        let (batch_size, request) = memory::batch_plan(&estimates, threads, budget, floor)?;
        let stage_name = if image_ids.is_empty() {
            "feature extraction"
        } else {
            "selected feature extraction"
        };
        return task.execute_with_memory(stage_name, request, threads, |task| {
            // Keep planned chunk boundaries even if admission grants fewer CPUs:
            // regrouping heterogeneous images could increase the aggregate.
            extract_cpu_images(
                &db,
                &images,
                database_path,
                images_dir,
                options,
                extractor,
                batch_size,
                task,
            )
        });
    }
    let batch_size = image_taskflow.map_or_else(
        || task.feature_batch_threads(),
        CpuFeatureTaskflow::max_in_flight_images,
    );
    extract_cpu_images(
        &db,
        &images,
        database_path,
        images_dir,
        options,
        extractor,
        batch_size,
        task,
    )
}

#[allow(clippy::too_many_arguments)]
fn extract_cpu_images(
    db: &ColmapDatabase,
    images: &[ColmapDatabaseImage],
    database_path: &Path,
    images_dir: &Path,
    options: &SiftExtractionOptions,
    extractor: &CpuSiftExtractor,
    batch_size: usize,
    task: &mut SfmTaskContext<'_>,
) -> Result<ExtractFeaturesReport> {
    let started = Instant::now();
    let image_count = images.len();
    let mut reports = Vec::with_capacity(image_count);
    // Bound graph metadata and legacy batch buffering. Both paths keep SQLite
    // commits and progress callbacks ordered on this calling thread.

    for batch in images.chunks(batch_size) {
        task.checkpoint()?;
        if let Some(executor) = task.cpu_feature_taskflow()? {
            let control = task.control();
            executor.extract_batch(
                taskflow::ExtractionBatch {
                    images: batch,
                    database_path,
                    images_dir,
                    options,
                },
                &control,
                &mut |extracted| record_cpu_image(db, extracted, &mut reports, image_count, task),
            )?;
        } else {
            let extracted = crate::execution::parallel(|| {
                batch
                    .par_iter()
                    .map(|image| extract_cpu_image(image, images_dir, options, extractor))
                    .collect::<Result<Vec<_>>>()
            })?;
            for image in extracted {
                record_cpu_image(db, &image, &mut reports, image_count, task)?;
            }
        }
    }
    reports.sort_unstable_by(|left, right| left.image_name.cmp(&right.image_name));
    let total_keypoints = reports
        .iter()
        .map(|image| image.num_keypoints)
        .sum::<usize>();
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

struct ExtractedCpuImage {
    image_id: u32,
    image_name: String,
    keypoints: Vec<ColmapKeypoint>,
    descriptors: ColmapDescriptors,
    extract_ms: f64,
}

fn extract_cpu_image(
    image: &ColmapDatabaseImage,
    images_dir: &Path,
    options: &SiftExtractionOptions,
    extractor: &CpuSiftExtractor,
) -> Result<ExtractedCpuImage> {
    let image_path = images_dir.join(&image.name);
    if !image_path.exists() {
        bail!(
            "missing image file for database image {}: {}",
            image.name,
            image_path.display()
        );
    }
    let started = Instant::now();
    let decoded = load_colmap_grayscale_u8(&image_path)
        .with_context(|| format!("failed to load {}", image_path.display()))?;
    let features =
        extractor.extract_grayscale(&decoded.data, decoded.width, decoded.height, options)?;
    Ok(ExtractedCpuImage {
        image_id: image.image_id,
        image_name: image.name.clone(),
        keypoints: sift_features_to_colmap_keypoints(&features),
        descriptors: sift_features_to_colmap_descriptors(&features)?,
        extract_ms: started.elapsed().as_secs_f64() * 1000.0,
    })
}

fn record_cpu_image(
    db: &ColmapDatabase,
    image: &ExtractedCpuImage,
    reports: &mut Vec<ExtractFeaturesImageReport>,
    image_count: usize,
    task: &mut SfmTaskContext<'_>,
) -> Result<()> {
    task.checkpoint()?;
    let started = Instant::now();
    db.with_transaction(|| {
        db.upsert_keypoints(image.image_id, &image.keypoints)?;
        db.upsert_descriptors(image.image_id, &image.descriptors)?;
        Ok(())
    })?;
    reports.push(ExtractFeaturesImageReport {
        image_name: image.image_name.clone(),
        num_keypoints: image.keypoints.len(),
        elapsed_ms: image.extract_ms + started.elapsed().as_secs_f64() * 1000.0,
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
        message: Some(image.image_name.clone()),
        issue: None,
    });
    task.checkpoint()?;
    Ok(())
}

fn extract_selected_features_to_database_with_extractor_and_task<E: SiftFeatureExtractor>(
    database_path: &Path,
    images_dir: &Path,
    options: &SiftExtractionOptions,
    image_ids: &[u32],
    extractor: &E,
    task: &mut SfmTaskContext<'_>,
) -> Result<ExtractFeaturesReport> {
    if crate::execution::active_threads().is_none() {
        return task.execute(
            "feature extraction and commit",
            options.use_gpu,
            1,
            |task| {
                extract_selected_features_to_database_with_extractor_and_task(
                    database_path,
                    images_dir,
                    options,
                    image_ids,
                    extractor,
                    task,
                )
            },
        );
    }
    options.check()?;
    let db = ColmapDatabase::open(database_path)?;
    let images = read_selected_feature_images(&db, image_ids)?;

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
    fn controlled_selected_extraction_does_not_touch_existing_keyframes() -> Result<()> {
        use crate::task::{SfmTaskControl, SfmTaskEvent};

        let (_dir, db_path, images_dir) = two_image_fixture()?;
        let db = ColmapDatabase::open(&db_path)?;
        db.write_keypoints(1, &[ColmapKeypoint::new(9.0, 9.0)])?;
        db.write_descriptors(
            1,
            &ColmapDescriptors::new(COLMAP_FEATURE_SIFT, 1, 128, vec![3u8; 128])?,
        )?;
        let control = SfmTaskControl::new();
        let mut events = Vec::<SfmTaskEvent>::new();
        let mut sink = |event: SfmTaskEvent| events.push(event);
        let mut task = crate::task::SfmTaskContext::new(&control, &mut sink);

        let report = extract_selected_features_to_database_with_extractor_and_task(
            &db_path,
            &images_dir,
            &SiftExtractionOptions::default(),
            &[2],
            &DeterministicExtractor,
            &mut task,
        )?;

        assert_eq!(report.image_count, 1);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].image_id, Some(2));
        assert_eq!(db.read_keypoints(1)?, vec![ColmapKeypoint::new(9.0, 9.0)]);
        assert_eq!(db.read_descriptors(1)?.data, vec![3u8; 128]);
        assert_eq!(db.read_keypoints(2)?, vec![ColmapKeypoint::new(1.0, 1.0)]);
        Ok(())
    }

    #[test]
    fn controlled_extraction_checks_cancellation_before_missing_file_validation() -> Result<()> {
        use crate::task::{SfmTaskControl, SfmTaskEvent, SfmTaskStop};

        let (_dir, db_path, images_dir) = two_image_fixture()?;
        std::fs::remove_file(images_dir.join("left.jpg"))?;
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

    fn feature_runtime(
        cpu: usize,
        images_in_memory: usize,
    ) -> std::sync::Arc<rustscan_taskflow::Runtime> {
        std::sync::Arc::new(
            rustscan_taskflow::Runtime::new(rustscan_taskflow::RuntimeConfig {
                budget: rustscan_taskflow::Budget {
                    cpu_threads: cpu,
                    memory_bytes: images_in_memory as u64 * 64 * 1024 * 1024,
                    io_slots: cpu,
                },
                ..Default::default()
            })
            .unwrap(),
        )
    }

    fn assert_feature_resources_released(runtime: &rustscan_taskflow::Runtime) {
        let usage = runtime.snapshot().unwrap();
        assert_eq!(usage.cpu_threads, 0);
        assert_eq!(usage.memory_bytes, 0);
        assert_eq!(usage.io_slots, 0);
        assert_eq!(usage.pending_tasks, 0);
        assert!(usage.exclusive.is_empty());
    }

    #[test]
    fn taskflow_single_image_budget_matches_legacy_database_exactly() -> Result<()> {
        let (_dir, db_path, images_dir) = two_image_fixture()?;
        for name in ["left.jpg", "right.jpg"] {
            write_checkerboard_image(&images_dir.join(name), 256, 256)?;
        }
        let options = SiftExtractionOptions {
            max_num_features: 256,
            ..Default::default()
        };
        let legacy = extract_features_to_database(&db_path, &images_dir, &options)?;
        assert!(legacy.total_keypoints > 0);
        let db = ColmapDatabase::open(&db_path)?;
        let expected = [1, 2].map(|id| {
            (
                db.read_keypoints(id).unwrap(),
                db.read_descriptors(id).unwrap().data,
            )
        });
        let runtime = feature_runtime(1, 1);
        let executor = CpuFeatureTaskflow::new(runtime.clone(), 64 * 1024 * 1024, 8)?;
        let control = SfmTaskControl::new();
        let thread = std::thread::current().id();
        let mut events = Vec::new();
        let mut sink = |event| {
            assert_eq!(std::thread::current().id(), thread);
            events.push(event);
        };
        let mut task =
            SfmTaskContext::new(&control, &mut sink).with_cpu_feature_taskflow(&executor);
        let report =
            extract_features_to_database_with_task(&db_path, &images_dir, &options, &mut task)?;
        assert_eq!(legacy.total_keypoints, report.total_keypoints);
        assert_eq!(legacy.backend, report.backend);
        for (id, (keypoints, descriptors)) in [1, 2].into_iter().zip(expected) {
            assert_eq!(db.read_keypoints(id)?, keypoints);
            assert_eq!(db.read_descriptors(id)?.data, descriptors);
        }
        assert_eq!(
            events
                .iter()
                .map(|event| event.image_id.unwrap())
                .collect::<Vec<_>>(),
            vec![2, 1]
        );
        assert_feature_resources_released(&runtime);
        Ok(())
    }

    #[test]
    fn taskflow_pause_after_first_commit_preserves_safe_boundary() -> Result<()> {
        let (_dir, db_path, images_dir) = two_image_fixture()?;
        let runtime = feature_runtime(2, 2);
        let executor = CpuFeatureTaskflow::new(runtime.clone(), 64 * 1024 * 1024, 8)?;
        let control = SfmTaskControl::new();
        let mut sink = |event: SfmTaskEvent| {
            if event.completed == Some(1) {
                control.request_pause();
            }
        };
        let mut task =
            SfmTaskContext::new(&control, &mut sink).with_cpu_feature_taskflow(&executor);
        let error = extract_features_to_database_with_task(
            &db_path,
            &images_dir,
            &Default::default(),
            &mut task,
        )
        .unwrap_err();
        assert_eq!(
            error.downcast_ref::<crate::SfmTaskStop>(),
            Some(&crate::SfmTaskStop::Paused)
        );
        let db = ColmapDatabase::open(&db_path)?;
        assert!(db.exists_keypoints(2)?);
        assert!(!db.exists_keypoints(1)?);
        assert_feature_resources_released(&runtime);
        Ok(())
    }

    #[test]
    fn taskflow_cancel_while_budget_paused_drains_queued_graph() -> Result<()> {
        let (_dir, db_path, images_dir) = two_image_fixture()?;
        let runtime = feature_runtime(1, 1);
        let executor = CpuFeatureTaskflow::new(runtime.clone(), 64 * 1024 * 1024, 8)?;
        let mut budget = runtime.snapshot()?.budget;
        budget.cpu_threads = 0;
        runtime.set_budget(budget)?;
        let control = SfmTaskControl::new();
        std::thread::scope(|scope| -> Result<()> {
            let worker = scope.spawn(|| {
                let mut sink = |_| {};
                let mut task =
                    SfmTaskContext::new(&control, &mut sink).with_cpu_feature_taskflow(&executor);
                extract_features_to_database_with_task(
                    &db_path,
                    &images_dir,
                    &Default::default(),
                    &mut task,
                )
            });
            let deadline = Instant::now() + std::time::Duration::from_secs(5);
            while runtime.snapshot()?.pending_tasks == 0 && Instant::now() < deadline {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            let queued = runtime.snapshot()?.pending_tasks;
            control.request_cancel();
            let error = worker.join().unwrap().unwrap_err();
            assert_eq!(queued, 4);
            assert_eq!(
                error.downcast_ref::<crate::SfmTaskStop>(),
                Some(&crate::SfmTaskStop::Cancelled)
            );
            Ok(())
        })?;
        let db = ColmapDatabase::open(&db_path)?;
        assert!(!db.exists_keypoints(1)? && !db.exists_keypoints(2)?);
        assert_feature_resources_released(&runtime);
        Ok(())
    }

    #[test]
    fn taskflow_write_failure_rolls_back_and_releases_waiting_writers() -> Result<()> {
        let (_dir, db_path, images_dir) = two_image_fixture()?;
        rusqlite::Connection::open(&db_path)?.execute_batch(
            "CREATE TRIGGER fail_taskflow_descriptor BEFORE INSERT ON descriptors
             BEGIN SELECT RAISE(ABORT, 'taskflow descriptor failure'); END;",
        )?;
        let runtime = feature_runtime(2, 2);
        let executor = CpuFeatureTaskflow::new(runtime.clone(), 64 * 1024 * 1024, 8)?;
        let control = SfmTaskControl::new();
        let mut sink = |_| {};
        let mut task =
            SfmTaskContext::new(&control, &mut sink).with_cpu_feature_taskflow(&executor);
        let error = extract_features_to_database_with_task(
            &db_path,
            &images_dir,
            &Default::default(),
            &mut task,
        )
        .unwrap_err();
        assert!(error.to_string().contains("taskflow descriptor failure"));
        let db = ColmapDatabase::open(&db_path)?;
        assert!(!db.exists_keypoints(1)? && !db.exists_keypoints(2)?);
        assert_feature_resources_released(&runtime);
        Ok(())
    }

    #[test]
    fn taskflow_missing_input_fails_without_stalling_downstream() -> Result<()> {
        let (_dir, db_path, images_dir) = two_image_fixture()?;
        std::fs::remove_file(images_dir.join("left.jpg"))?;
        let runtime = feature_runtime(1, 1);
        let executor = CpuFeatureTaskflow::new(runtime.clone(), 64 * 1024 * 1024, 8)?;
        let control = SfmTaskControl::new();
        let mut sink = |_| {};
        let mut task =
            SfmTaskContext::new(&control, &mut sink).with_cpu_feature_taskflow(&executor);
        let error = extract_features_to_database_with_task(
            &db_path,
            &images_dir,
            &Default::default(),
            &mut task,
        )
        .unwrap_err();
        assert!(error.to_string().contains("missing image file"));
        assert_feature_resources_released(&runtime);
        Ok(())
    }

    #[test]
    fn taskflow_callback_panic_drains_native_work_and_releases_grants() -> Result<()> {
        let (_dir, db_path, images_dir) = two_image_fixture()?;
        let runtime = feature_runtime(2, 2);
        let executor = CpuFeatureTaskflow::new(runtime.clone(), 64 * 1024 * 1024, 8)?;
        let control = SfmTaskControl::new();
        let mut sink = |_| panic!("test callback panic");
        let mut task =
            SfmTaskContext::new(&control, &mut sink).with_cpu_feature_taskflow(&executor);
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            extract_features_to_database_with_task(
                &db_path,
                &images_dir,
                &Default::default(),
                &mut task,
            )
        }));
        assert!(panic.is_err());
        assert_feature_resources_released(&runtime);
        Ok(())
    }

    #[test]
    fn taskflow_two_workflows_share_budget_and_selected_image_semantics() -> Result<()> {
        let fixtures = [two_image_fixture()?, two_image_fixture()?];
        let runtime = feature_runtime(2, 2);
        let executor = CpuFeatureTaskflow::new(runtime.clone(), 64 * 1024 * 1024, 8)?;
        let start = std::sync::Barrier::new(3);
        let finished = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|scope| -> Result<()> {
            let mut workers = Vec::new();
            for (_, db_path, images_dir) in &fixtures {
                let executor = &executor;
                let start = &start;
                let finished = &finished;
                workers.push(scope.spawn(move || -> Result<()> {
                    let control = SfmTaskControl::new();
                    let mut sink = |_| {};
                    let mut task = SfmTaskContext::new(&control, &mut sink)
                        .with_cpu_feature_taskflow(executor);
                    start.wait();
                    let result = extract_selected_features_to_database_with_task(
                        db_path,
                        images_dir,
                        &Default::default(),
                        &[2],
                        &mut task,
                    );
                    finished.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    assert_eq!(result?.image_count, 1);
                    let db = ColmapDatabase::open(db_path)?;
                    assert!(db.exists_keypoints(2)? && !db.exists_keypoints(1)?);
                    Ok(())
                }));
            }
            start.wait();
            while finished.load(std::sync::atomic::Ordering::SeqCst) != 2 {
                let usage = runtime.snapshot()?;
                assert!(usage.cpu_threads <= 2);
                assert!(usage.io_slots <= 2);
                assert!(usage.memory_bytes <= 128 * 1024 * 1024);
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            for worker in workers {
                worker.join().unwrap()?;
            }
            Ok(())
        })?;
        assert_feature_resources_released(&runtime);
        Ok(())
    }

    #[cfg(feature = "ceres-ba")]
    #[test]
    fn taskflow_ceres_and_sift_share_one_resource_budget() -> Result<()> {
        let (_dir, db_path, images_dir) = two_image_fixture()?;
        for name in ["left.jpg", "right.jpg"] {
            write_checkerboard_image(&images_dir.join(name), 256, 256)?;
        }
        let runtime = feature_runtime(2, 2);
        let features = CpuFeatureTaskflow::new(runtime.clone(), 64 * 1024 * 1024, 8)?;
        let ba = crate::CeresBaTaskflow::new(runtime.clone(), 4, 64 * 1024 * 1024)?;
        let (frames, base, ..) = crate::ba::rig_sensor_ba_fixture();
        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|scope| -> Result<()> {
            let feature_job = scope.spawn(|| {
                let control = SfmTaskControl::new();
                let mut sink = |_| {};
                let mut task =
                    SfmTaskContext::new(&control, &mut sink).with_cpu_feature_taskflow(&features);
                barrier.wait();
                extract_features_to_database_with_task(
                    &db_path,
                    &images_dir,
                    &Default::default(),
                    &mut task,
                )
            });
            barrier.wait();
            for _ in 0..10 {
                let mut reconstruction = base.clone();
                let report = crate::ba::try_refine_bundle_adjustment(
                    &frames,
                    &mut reconstruction,
                    crate::ba::BundleAdjustmentOptions {
                        iterations: 30,
                        max_observation_error_px: 200.0,
                        variable_images: Some(vec![0, 1]),
                        constant_images: vec![2, 3],
                        point_ids: Some((0..8).collect()),
                        min_num_residuals_for_multi_threading: 0,
                        taskflow: Some(ba.clone()),
                        ..Default::default()
                    },
                )?
                .unwrap();
                assert!(report.is_solution_usable());
                assert!(report.final_cost <= report.initial_cost);
                assert!(report.solver_num_threads <= 2);
                assert_eq!(
                    report.solver_num_threads,
                    report.scheduling.unwrap().granted_cpu_threads
                );
                let usage = runtime.snapshot()?;
                assert!(usage.cpu_threads <= 2 && usage.memory_bytes <= 128 * 1024 * 1024);
            }
            assert!(feature_job.join().unwrap()?.total_keypoints > 0);
            Ok(())
        })?;
        assert_feature_resources_released(&runtime);
        Ok(())
    }

    #[cfg(all(feature = "vlfeat-sift", not(feature = "lowe-sift-backend")))]
    #[test]
    fn memory_db_over_budget_rejects_before_decode_and_reports_request() -> Result<()> {
        let (_dir, db_path, images_dir) = two_image_fixture()?;
        // Valid BMP dimensions, deliberately absent pixel payload. A decoder
        // would fail; budget rejection must win and report zero service/grant.
        let mut header = vec![0u8; 54];
        header[..2].copy_from_slice(b"BM");
        header[2..6].copy_from_slice(&54u32.to_le_bytes());
        header[10..14].copy_from_slice(&54u32.to_le_bytes());
        header[14..18].copy_from_slice(&40u32.to_le_bytes());
        header[18..22].copy_from_slice(&32u32.to_le_bytes());
        header[22..26].copy_from_slice(&32u32.to_le_bytes());
        header[26..28].copy_from_slice(&1u16.to_le_bytes());
        header[28..30].copy_from_slice(&24u16.to_le_bytes());
        // Use a BMP extension too: the fallback decoder guesses from extension.
        let db = ColmapDatabase::open(&db_path)?;
        for id in [1, 2] {
            let mut image = db.read_image(id)?.unwrap();
            image.name = format!("{id}.bmp");
            db.update_image(&image)?;
            let path = images_dir.join(&image.name);
            std::fs::write(&path, &header)?;
            assert_eq!(image::image_dimensions(&path)?, (32, 32));
            assert!(image::open(&path).is_err());
        }
        let runtime = std::sync::Arc::new(rustscan_taskflow::Runtime::new(
            rustscan_taskflow::RuntimeConfig {
                budget: rustscan_taskflow::Budget {
                    cpu_threads: 2,
                    memory_bytes: 1024,
                    io_slots: 2,
                },
                ..Default::default()
            },
        )?);
        let options = SiftExtractionOptions::default();
        let expected = memory::estimate(32, 32, 54, &options)?;
        for selected in [false, true] {
            let control = SfmTaskControl::new();
            let mut sink = |_| {};
            let mut task = SfmTaskContext::new(&control, &mut sink)
                .with_taskflow(crate::SfmTaskflow::new(runtime.clone(), 1)?);
            let result = if selected {
                extract_selected_features_to_database_with_task(
                    &db_path,
                    &images_dir,
                    &options,
                    &[1],
                    &mut task,
                )
            } else {
                extract_features_to_database_with_task(&db_path, &images_dir, &options, &mut task)
            };
            assert!(result.unwrap_err().to_string().contains("before decode"));
            let reports = task.stage_reports();
            assert_eq!(reports.len(), 1);
            assert_eq!(reports[0].requested_memory, expected);
            assert_eq!(reports[0].granted_memory, 0);
            assert_eq!(reports[0].service_ms, 0.0);
            assert!(reports[0].cancelled_or_failed);
        }
        let adapter = CpuFeatureTaskflow::new(runtime.clone(), 1, 8)?;
        let control = SfmTaskControl::new();
        let mut sink = |_| {};
        let mut task = SfmTaskContext::new(&control, &mut sink).with_cpu_feature_taskflow(&adapter);
        let error =
            extract_features_to_database_with_task(&db_path, &images_dir, &options, &mut task)
                .unwrap_err();
        assert!(error
            .to_string()
            .contains(&format!("estimate {expected} bytes")));
        assert!(error.to_string().contains("before decode"));
        assert!(db.read_keypoint_counts()?.is_empty());
        assert_feature_resources_released(&runtime);
        Ok(())
    }

    #[cfg(all(feature = "vlfeat-sift", not(feature = "lowe-sift-backend")))]
    #[test]
    fn memory_small_db_stage_request_matches_batch_plan() -> Result<()> {
        let (_dir, db_path, images_dir) = two_image_fixture()?;
        let options = SiftExtractionOptions {
            max_num_features: 16,
            ..Default::default()
        };
        let estimates = ["left.jpg", "right.jpg"]
            .iter()
            .map(|name| {
                memory::image_estimate(&images_dir.join(name), &options, 0, &SfmTaskControl::new())
            })
            .collect::<Result<Vec<_>>>()?;
        let budget = *estimates.iter().max().unwrap();
        let runtime = std::sync::Arc::new(rustscan_taskflow::Runtime::new(
            rustscan_taskflow::RuntimeConfig {
                budget: rustscan_taskflow::Budget {
                    cpu_threads: 4,
                    memory_bytes: budget,
                    io_slots: 4,
                },
                ..Default::default()
            },
        )?);
        let control = SfmTaskControl::new();
        let mut sink = |_| {};
        let mut task = SfmTaskContext::new(&control, &mut sink)
            .with_taskflow(crate::SfmTaskflow::new(runtime.clone(), 1)?);
        let result =
            extract_features_to_database_with_task(&db_path, &images_dir, &options, &mut task)?;
        assert_eq!(result.image_count, 2);
        let reports = task.stage_reports();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].requested_memory, budget);
        assert_eq!(reports[0].granted_memory, budget);
        assert!(!reports[0].cancelled_or_failed);
        assert_feature_resources_released(&runtime);
        Ok(())
    }

    #[test]
    fn memory_nonstandard_preserves_default_and_caller_estimate_contract() -> Result<()> {
        let (_dir, db_path, images_dir) = two_image_fixture()?;
        let control = SfmTaskControl::new();
        let mut sink = |_| {};
        let mut task = SfmTaskContext::new(&control, &mut sink);
        let options = SiftExtractionOptions {
            force_covariant_extractor: true,
            ..Default::default()
        };
        assert!(!memory::standard(&options));
        let (default_estimate, _) = task.feature_memory_limits()?;
        let result =
            extract_features_to_database_with_task(&db_path, &images_dir, &options, &mut task)?;
        assert_eq!(result.image_count, 2);
        let reports = task.stage_reports();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].requested_memory, default_estimate);
        assert_eq!(reports[0].requested_threads, 4);
        assert!(!reports[0].cancelled_or_failed);

        let runtime = feature_runtime(1, 1);
        let caller_estimate = runtime.snapshot()?.budget.memory_bytes;
        let mut caller_sink = |_| {};
        let mut caller_task = SfmTaskContext::new(&control, &mut caller_sink)
            .with_taskflow(crate::SfmTaskflow::new(runtime.clone(), caller_estimate)?);
        let selected = extract_selected_features_to_database_with_task(
            &db_path,
            &images_dir,
            &options,
            &[1],
            &mut caller_task,
        )?;
        assert_eq!(selected.image_count, 1);
        let reports = caller_task.stage_reports();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].requested_memory, caller_estimate);
        assert_eq!(reports[0].granted_memory, caller_estimate);
        assert!(!reports[0].cancelled_or_failed);
        assert_feature_resources_released(&runtime);

        control.request_cancel();
        let error = extract_features_to_database_with_task(
            Path::new("missing.db"),
            Path::new("missing"),
            &options,
            &mut task,
        )
        .unwrap_err();
        assert!(error.downcast_ref::<crate::task::SfmTaskStop>().is_some());
        Ok(())
    }

    #[test]
    fn memory_automatic_planning_requires_explicit_binding() -> Result<()> {
        // Dimensions only: never allocate/decode this large image. Automatic
        // planning would exceed 2 GiB after the default resize, so it must not
        // silently become mandatory for the legacy no-task/default API.
        assert!(
            memory::estimate(4000, 3000, 0, &SiftExtractionOptions::default())?
                > 2 * 1024 * 1024 * 1024
        );
        let (_dir, db_path, images_dir) = two_image_fixture()?;
        let control = SfmTaskControl::new();
        let mut sink = |_| {};
        let mut task = SfmTaskContext::new(&control, &mut sink);
        assert!(!task.has_feature_memory_estimate());
        let (legacy_floor, _) = task.feature_memory_limits()?;
        assert_eq!(
            extract_features_to_database_with_task(
                &db_path,
                &images_dir,
                &Default::default(),
                &mut task
            )?
            .image_count,
            2
        );
        assert_eq!(task.stage_reports()[0].requested_memory, legacy_floor);
        assert_eq!(
            extract_features_to_database(&db_path, &images_dir, &Default::default())?.image_count,
            2
        );
        assert_eq!(
            extract_features_to_database_with_extractor(
                &db_path,
                &images_dir,
                &Default::default(),
                &DeterministicExtractor
            )?
            .image_count,
            2
        );
        let runtime = feature_runtime(1, 1);
        let mut explicit_sink = |_| {};
        let explicit = SfmTaskContext::new(&control, &mut explicit_sink)
            .with_taskflow(crate::SfmTaskflow::new(runtime, 1)?);
        assert!(explicit.has_feature_memory_estimate());
        Ok(())
    }

    #[test]
    fn memory_whole_stage_transient_budget_recovers_or_cancels() -> Result<()> {
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            mpsc, Arc,
        };
        use std::time::Duration;
        for cancel in [false, true] {
            let runtime = Arc::new(rustscan_taskflow::Runtime::new(
                rustscan_taskflow::RuntimeConfig {
                    budget: rustscan_taskflow::Budget {
                        cpu_threads: 1,
                        memory_bytes: 100,
                        io_slots: 1,
                    },
                    ..Default::default()
                },
            )?);
            let ceiling = runtime.snapshot()?.budget;
            let mut low = ceiling;
            low.memory_bytes = 50;
            runtime.set_budget(low)?;
            let executor = crate::SfmTaskflow::new(runtime.clone(), 1)?;
            let control = SfmTaskControl::new();
            let entered = AtomicBool::new(false);
            let (done, result) = mpsc::sync_channel(1);
            std::thread::scope(|scope| -> Result<()> {
                let worker = scope.spawn(|| {
                    let mut sink = |_| {};
                    let mut task = SfmTaskContext::new(&control, &mut sink).with_taskflow(executor);
                    let status = task.execute_with_memory("memory transient", 80, 1, |_| {
                        entered.store(true, Ordering::SeqCst);
                        Ok(())
                    });
                    let _ = done.send((status, task.stage_reports()));
                });
                let outcome = (|| -> Result<()> {
                    let deadline = Instant::now() + Duration::from_secs(5);
                    while runtime.snapshot()?.pending_tasks == 0 {
                        anyhow::ensure!(Instant::now() < deadline, "stage did not queue");
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    anyhow::ensure!(
                        !entered.load(Ordering::SeqCst),
                        "work started below its required memory budget"
                    );
                    anyhow::ensure!(
                        runtime.snapshot()?.memory_bytes == 0,
                        "queued stage acquired memory"
                    );
                    if cancel {
                        control.request_cancel();
                    } else {
                        runtime.set_budget(ceiling)?;
                    }
                    let (status, reports) = result.recv_timeout(Duration::from_secs(5))?;
                    assert_eq!(reports.len(), 1);
                    assert_eq!(reports[0].requested_memory, 80);
                    if cancel {
                        assert_eq!(
                            status.unwrap_err().downcast_ref::<crate::SfmTaskStop>(),
                            Some(&crate::SfmTaskStop::Cancelled)
                        );
                        assert!(!entered.load(Ordering::SeqCst));
                        assert_eq!(reports[0].granted_memory, 0);
                        assert!(reports[0].cancelled_or_failed);
                    } else {
                        status?;
                        assert!(entered.load(Ordering::SeqCst));
                        assert_eq!(reports[0].granted_memory, 80);
                        assert!(!reports[0].cancelled_or_failed);
                    }
                    Ok(())
                })();
                control.request_cancel();
                worker.join().expect("stage worker panicked");
                outcome
            })?;
            assert_feature_resources_released(&runtime);
        }
        Ok(())
    }

    #[test]
    fn memory_parent_actual_grant_remains_a_hard_boundary() -> Result<()> {
        let runtime = feature_runtime(1, 1);
        let control = SfmTaskControl::new();
        let mut sink = |_| {};
        let mut task = SfmTaskContext::new(&control, &mut sink)
            .with_taskflow(crate::SfmTaskflow::new(runtime.clone(), 40)?);
        let mut entered = false;
        let error = task
            .execute("parent", false, 1, |task| {
                task.execute_with_memory("memory child", 80, 1, |_| {
                    entered = true;
                    Ok(())
                })
            })
            .unwrap_err();
        assert!(error.to_string().contains("parent grant 40"));
        assert!(!entered);
        let reports = task.stage_reports();
        assert_eq!(reports.len(), 2);
        assert_eq!(reports[0].requested_memory, 80);
        assert_eq!(reports[0].granted_memory, 0);
        assert_eq!(reports[0].service_ms, 0.0);
        assert!(reports[0].cancelled_or_failed);
        assert_feature_resources_released(&runtime);
        Ok(())
    }

    #[test]
    fn planned_selected_mapping_and_fixed_chunks_override_parent_and_adapter() -> Result<()> {
        let options = SiftExtractionOptions {
            max_num_features: 8,
            ..Default::default()
        };
        if !memory::standard(&options) {
            return Ok(());
        }
        let (dir, db_path, _) = two_image_fixture()?;
        let original_a = dir.path().join("original-a.bmp");
        let original_b = dir.path().join("original-b.bmp");
        image::GrayImage::new(32, 32).save(&original_a)?;
        // Header-only second image: regrouping into a two-image batch would
        // decode this and fail BEFORE the first image's commit/pause callback.
        let mut header = vec![0u8; 54];
        header[..2].copy_from_slice(b"BM");
        header[2..6].copy_from_slice(&54u32.to_le_bytes());
        header[10..14].copy_from_slice(&54u32.to_le_bytes());
        header[14..18].copy_from_slice(&40u32.to_le_bytes());
        header[18..22].copy_from_slice(&32u32.to_le_bytes());
        header[22..26].copy_from_slice(&32u32.to_le_bytes());
        header[26..28].copy_from_slice(&1u16.to_le_bytes());
        header[28..30].copy_from_slice(&24u16.to_le_bytes());
        std::fs::write(&original_b, &header)?;
        let db = ColmapDatabase::open(&db_path)?;
        for (id, name) in [(2, "a.bmp"), (1, "b.bmp")] {
            let mut image = db.read_image(id)?.unwrap();
            image.name = name.to_owned();
            db.update_image(&image)?;
        }
        let control = SfmTaskControl::new();
        let plan = db_memory_plan(
            &[original_a.clone(), original_b.clone()],
            &options,
            true,
            4,
            0,
            77,
            &control,
        )?
        .unwrap();
        assert_eq!(plan.batch_size(), 1);
        let mapping = vec![(2, original_a.clone()), (1, original_b.clone())];
        assert!(validate_planned_selection(&db, &[], &plan).is_err());
        assert!(validate_planned_selection(
            &db,
            &[(1, original_a.clone()), (2, original_b.clone())],
            &plan
        )
        .is_err());
        assert!(validate_planned_selection(
            &db,
            &[(2, original_b.clone()), (1, original_a.clone())],
            &plan
        )
        .is_err());
        assert!(validate_planned_selection(
            &db,
            &[(2, original_a.clone()), (2, original_b.clone())],
            &plan
        )
        .is_err());
        assert!(validate_planned_selection(&db, &[(2, original_a.clone())], &plan).is_err());
        assert_eq!(validate_planned_selection(&db, &mapping, &plan)?.len(), 2);
        drop(db);
        let staged = dir.path().join("staged");
        std::fs::create_dir(&staged)?;
        std::fs::copy(original_a, staged.join("a.bmp"))?;
        std::fs::copy(original_b, staged.join("b.bmp"))?;
        let runtime = std::sync::Arc::new(rustscan_taskflow::Runtime::new(
            rustscan_taskflow::RuntimeConfig {
                budget: rustscan_taskflow::Budget {
                    cpu_threads: 2,
                    memory_bytes: plan.request_bytes() * 4,
                    io_slots: 2,
                },
                ..Default::default()
            },
        )?);
        let adapter = CpuFeatureTaskflow::new(runtime.clone(), 1, 8)?;
        let mut sink = |event: SfmTaskEvent| {
            if event.operation == SfmTaskOperation::ExtractImage && event.completed == Some(1) {
                control.request_pause();
            }
        };
        let mut task = SfmTaskContext::new(&control, &mut sink)
            .with_taskflow(crate::SfmTaskflow::new(runtime.clone(), 1)?)
            .with_cpu_feature_taskflow(&adapter);
        let changed_options = SiftExtractionOptions {
            upright: true,
            ..options.clone()
        };
        assert!(extract_selected_features_to_database_with_plan_and_task(
            &db_path,
            &staged,
            &changed_options,
            &mapping,
            &plan,
            &mut task,
        )
        .is_err());
        task.execute_with_memory("small parent", plan.request_bytes() - 1, 1, |task| {
            let error = extract_selected_features_to_database_with_plan_and_task(
                &db_path, &staged, &options, &mapping, &plan, task,
            )
            .unwrap_err();
            assert!(error.to_string().contains("exceeds parent grant"));
            Ok(())
        })?;
        let result =
            task.execute_with_memory("large parent", plan.request_bytes() * 3, 2, |task| {
                extract_selected_features_to_database_with_plan_and_task(
                    &db_path, &staged, &options, &mapping, &plan, task,
                )
            });
        assert_eq!(
            result
                .unwrap_err()
                .downcast_ref::<crate::task::SfmTaskStop>(),
            Some(&crate::task::SfmTaskStop::Paused)
        );
        let db = ColmapDatabase::open_read_only(&db_path)?;
        assert!(db.exists_descriptors(2)?);
        assert!(!db.exists_descriptors(1)?);
        let reports = task.stage_reports();
        let selected = reports
            .iter()
            .find(|r| r.stage_name == "selected feature extraction" && r.granted_memory > 0)
            .unwrap();
        assert_eq!(selected.requested_memory, plan.request_bytes());
        assert_eq!(selected.granted_memory, plan.request_bytes() * 3);
        assert_feature_resources_released(&runtime);
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
