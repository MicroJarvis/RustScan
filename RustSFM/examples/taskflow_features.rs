//! CPU SIFT stage admission vs per-image DAG; temporary synthetic inputs only.
//! Both modes use the same Runtime with four CPU slots across two workflows.
use anyhow::Result;
use rustscan_taskflow::{Budget, Runtime, RuntimeConfig};
use rustsfm::colmap::ColmapCamera;
use rustsfm::database::{
    ColmapDatabase, ColmapDatabaseCamera, ColmapDatabaseImage, ColmapKeypoint,
};
use rustsfm::{
    CpuFeatureTaskflow, SfmTaskContext, SfmTaskControl, SfmTaskflow, SiftExtractionOptions,
};
use std::path::PathBuf;
use std::sync::{Arc, Barrier};
use std::time::Instant;

const IMAGES: u32 = 8;
struct Fixture {
    _directory: tempfile::TempDir,
    database: PathBuf,
    images: PathBuf,
}

fn fixture() -> Result<Fixture> {
    let directory = tempfile::tempdir()?;
    let images = directory.path().join("images");
    std::fs::create_dir(&images)?;
    let database = directory.path().join("database.db");
    let db = ColmapDatabase::open(&database)?;
    db.write_camera(
        &ColmapDatabaseCamera {
            camera: ColmapCamera {
                camera_id: 1,
                model_id: rustsfm::types::COLMAP_PINHOLE,
                width: 512,
                height: 512,
                params: vec![400.0, 400.0, 256.0, 256.0],
            },
            has_prior_focal_length: true,
        },
        true,
    )?;
    for id in 1..=IMAGES {
        let name = format!("{id:03}.png");
        let image = image::GrayImage::from_fn(512, 512, |x, y| {
            let texture = ((x * 13 + id * 7) ^ (y * 17) ^ ((x / 23 + y / 19) * 43)) % 256;
            image::Luma([texture as u8])
        });
        image.save(images.join(&name))?;
        db.write_image(
            &ColmapDatabaseImage {
                image_id: id,
                name,
                camera_id: 1,
                frame_id: None,
            },
            true,
        )?;
    }
    Ok(Fixture {
        _directory: directory,
        database,
        images,
    })
}

type Rows = Vec<(Vec<ColmapKeypoint>, Vec<u8>)>;
fn rows(fixture: &Fixture) -> Result<Rows> {
    let db = ColmapDatabase::open(&fixture.database)?;
    (1..=IMAGES)
        .map(|id| Ok((db.read_keypoints(id)?, db.read_descriptors(id)?.data)))
        .collect()
}

fn measure(
    label: &str,
    fixtures: &[Fixture],
    stage: &SfmTaskflow,
    executor: Option<&CpuFeatureTaskflow>,
) -> Result<Vec<Rows>> {
    let start = Instant::now();
    let barrier = Barrier::new(fixtures.len());
    let reports = std::thread::scope(|scope| -> Result<Vec<_>> {
        let mut workers = Vec::new();
        for fixture in fixtures {
            let barrier = &barrier;
            workers.push(scope.spawn(move || {
                let options = SiftExtractionOptions {
                    max_num_features: 512,
                    ..Default::default()
                };
                barrier.wait();
                let control = SfmTaskControl::new();
                let mut sink = |_| {};
                let mut task =
                    SfmTaskContext::new(&control, &mut sink).with_taskflow(stage.clone());
                if let Some(executor) = executor {
                    task = task.with_cpu_feature_taskflow(executor);
                }
                rustsfm::extract_features_to_database_with_task(
                    &fixture.database,
                    &fixture.images,
                    &options,
                    &mut task,
                )
            }));
        }
        workers
            .into_iter()
            .map(|worker| worker.join().expect("benchmark worker panic"))
            .collect()
    })?;
    let elapsed = start.elapsed();
    assert!(reports
        .iter()
        .all(|report| report.image_count == IMAGES as usize && report.total_keypoints > 0));
    println!(
        "{label}: wall={:.3}s, workflow_seconds={:?}, backend={}, keypoints={}",
        elapsed.as_secs_f64(),
        reports
            .iter()
            .map(|report| report.extraction_seconds)
            .collect::<Vec<_>>(),
        reports[0].backend,
        reports
            .iter()
            .map(|report| report.total_keypoints)
            .sum::<usize>()
    );
    // Database parity checks are outside the timed region.
    fixtures.iter().map(rows).collect()
}

fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        anyhow::bail!("run with --release");
    }
    let fixtures = [fixture()?, fixture()?];

    let runtime = Arc::new(Runtime::new(RuntimeConfig {
        budget: Budget {
            cpu_threads: 4,
            memory_bytes: 1024 * 1024 * 1024,
            io_slots: 4,
        },
        ..Default::default()
    })?);
    let stage = SfmTaskflow::new(runtime.clone(), 512 * 1024 * 1024)?;
    let executor = CpuFeatureTaskflow::new(runtime.clone(), 256 * 1024 * 1024, 32)?;
    println!("2 concurrent workflows x {IMAGES} synthetic 512x512 images; shared CPU budget=4, memory=1024MiB; stage estimate=512MiB, image estimate=256MiB; warm-up then alternating order");
    let expected = measure("warmup stage", &fixtures, &stage, None)?;
    assert_eq!(
        measure("warmup image-dag", &fixtures, &stage, Some(&executor))?,
        expected
    );
    for round in 0..3 {
        for scheduled in if round % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let label = format!(
                "round {} {}",
                round + 1,
                if scheduled { "image-dag" } else { "stage" }
            );
            assert_eq!(
                measure(&label, &fixtures, &stage, scheduled.then_some(&executor))?,
                expected
            );
        }
    }
    let usage = runtime.snapshot()?;
    assert_eq!(usage.pending_tasks, 0);
    assert_eq!(usage.cpu_threads, 0);
    assert_eq!(usage.memory_bytes, 0);
    assert_eq!(usage.io_slots, 0);
    assert!(usage.exclusive.is_empty());
    println!("Exact keypoint/descriptor parity passed; Taskflow reservations returned to zero.");
    Ok(())
}
