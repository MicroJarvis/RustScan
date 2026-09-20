//! C5: real CPU feature-only whole-stage admission. No image DAG or reconstruction.
use anyhow::{ensure, Context, Result};
use clap::Parser;
use rustscan_sfm::colmap::ColmapCamera;
use rustscan_sfm::database::{
    ColmapDatabase, ColmapDatabaseCamera, ColmapDatabaseImage, ColmapDescriptors, ColmapKeypoint,
};
use rustscan_sfm::{SfmTaskContext, SfmTaskControl, SfmTaskflow, SiftExtractionOptions};
use rustscan_taskflow::{Budget, ResourceSnapshot, Runtime, RuntimeConfig};
use serde_json::{json, Value};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Barrier};
use std::time::{Duration, Instant};

const MIB: u64 = 1024 * 1024;
const SCRATCH: u64 = 512 * MIB;
const IMAGE_COUNT: usize = 8;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_parser = clap::value_parser!(u8).range(1..=2))]
    jobs: u8,
    #[arg(long)]
    memory_mib: u64,
}

fn save(path: &Path, value: &Value) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    writeln!(file)?;
    Ok(())
}

fn select_images(input: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(input)? {
        let path = entry?.path();
        if path.is_file()
            && path
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|s| matches!(s.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"))
        {
            paths.push(path);
        }
    }
    paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    ensure!(paths.len() >= IMAGE_COUNT, "need eight real images");
    paths.truncate(IMAGE_COUNT);
    Ok(paths)
}

fn prepare(path: &Path, images: &[PathBuf]) -> Result<Value> {
    // Reserve the filename atomically: never open an existing DB for extraction.
    File::create_new(path)?;
    let db = ColmapDatabase::open(path)?;
    let mut metadata = Vec::new();
    for (index, image) in images.iter().enumerate() {
        let id = index as u32 + 1;
        let (width, height) = image::image_dimensions(image)?;
        let name = image
            .file_name()
            .context("image filename")?
            .to_str()
            .context("UTF-8 name")?;
        let focal = width.max(height) as f64;
        let camera = ColmapDatabaseCamera {
            camera: ColmapCamera {
                camera_id: id,
                model_id: rustscan_sfm::types::COLMAP_PINHOLE,
                width: width.into(),
                height: height.into(),
                params: vec![focal, focal, width as f64 / 2.0, height as f64 / 2.0],
            },
            has_prior_focal_length: false,
        };
        db.write_camera(&camera, true)?;
        db.write_image(
            &ColmapDatabaseImage {
                image_id: id,
                name: name.into(),
                camera_id: id,
                frame_id: None,
            },
            true,
        )?;
        metadata.push(
            json!({"image_id":id,"name":name,"width":width,"height":height,
            "camera_id":id,"model":"PINHOLE","params":camera.camera.params}),
        );
    }
    ensure!(
        db.read_keypoint_counts()?.is_empty(),
        "new DB contains features"
    );
    Ok(json!(metadata))
}

fn hashes(keypoints: &[ColmapKeypoint], descriptors: &ColmapDescriptors) -> Result<Value> {
    ensure!(!keypoints.is_empty(), "empty keypoints");
    ensure!(
        descriptors.rows == keypoints.len() && descriptors.cols == 128,
        "descriptor shape"
    );
    ensure!(
        descriptors.data.len() == descriptors.rows * descriptors.cols,
        "descriptor bytes"
    );
    let mut key_hash = blake3::Hasher::new();
    for kp in keypoints {
        for value in [kp.x, kp.y, kp.a11, kp.a12, kp.a21, kp.a22] {
            ensure!(value.is_finite(), "nonfinite keypoint");
            key_hash.update(&value.to_bits().to_le_bytes());
        }
    }
    let mut complete = blake3::Hasher::new();
    complete.update(b"c5-features-v1\0");
    complete.update(&(keypoints.len() as u64).to_le_bytes());
    complete.update(key_hash.finalize().as_bytes());
    complete.update(&descriptors.feature_type.to_le_bytes());
    complete.update(&(descriptors.rows as u64).to_le_bytes());
    complete.update(&(descriptors.cols as u64).to_le_bytes());
    complete.update(&descriptors.data);
    Ok(json!({"keypoints":keypoints.len(),"keypoint_fields":6,
        "keypoint_bits_blake3":key_hash.finalize().to_hex().to_string(),
        "descriptor_rows":descriptors.rows,"descriptor_cols":descriptors.cols,
        "descriptor_type":descriptors.feature_type,"descriptor_bytes":descriptors.data.len(),
        "descriptor_blake3":blake3::hash(&descriptors.data).to_hex().to_string(),
        "complete_blake3":complete.finalize().to_hex().to_string()}))
}

fn quality(path: &Path, images: &[PathBuf]) -> Result<Value> {
    let db = ColmapDatabase::open(path)?;
    let mut result = Vec::new();
    for (index, image) in images.iter().enumerate() {
        let id = index as u32 + 1;
        let mut row = hashes(&db.read_keypoints(id)?, &db.read_descriptors(id)?)?;
        row["image_id"] = json!(id);
        row["name"] = json!(image.file_name().unwrap().to_str().unwrap());
        result.push(row);
    }
    Ok(json!(result))
}

fn drained(s: &ResourceSnapshot) -> bool {
    s.cpu_threads == 0
        && s.memory_bytes == 0
        && s.io_slots == 0
        && s.pending_tasks == 0
        && s.exclusive.is_empty()
        && s.gpus
            .values()
            .all(|g| g.memory_bytes == 0 && g.in_flight == 0)
}

fn snapshot(runtime: &Runtime, epoch: Instant, phase: &str) -> Result<Value> {
    let s = runtime.snapshot()?;
    let gpus: Vec<_> = s
        .gpus
        .iter()
        .map(|(id, g)| json!({"id":id.0,"memory_bytes":g.memory_bytes,"in_flight":g.in_flight}))
        .collect();
    Ok(
        json!({"ms":epoch.elapsed().as_secs_f64()*1000.0,"phase":phase,
        "budget":{"cpu_threads":s.budget.cpu_threads,"memory_bytes":s.budget.memory_bytes,"io_slots":s.budget.io_slots},
        "cpu_threads":s.cpu_threads,"memory_bytes":s.memory_bytes,"io_slots":s.io_slots,
        "pending_tasks":s.pending_tasks,"exclusive":s.exclusive,"gpus":gpus,"drained":drained(&s)}),
    )
}

fn sample_line(file: &mut File, runtime: &Runtime, epoch: Instant, phase: &str) -> Result<Value> {
    let value = snapshot(runtime, epoch, phase)?;
    serde_json::to_writer(&mut *file, &value)?;
    writeln!(file)?;
    file.flush()?;
    Ok(value)
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!cfg!(debug_assertions), "run --release");
    ensure!(
        cfg!(feature = "vlfeat-sift"),
        "C5 measurements require default VLFeat backend"
    );
    ensure!(
        matches!(args.memory_mib, 512 | 1024),
        "C5 memory budget must be 512 or 1024MiB"
    );
    ensure!(
        args.output.is_dir(),
        "runner must create a fresh output directory"
    );
    let claim = args.output.join("child.claim");
    File::create_new(&claim)?;
    let epoch = Instant::now();
    let images = select_images(&args.input)?;
    let options = SiftExtractionOptions::default();
    let config = json!({"jobs":args.jobs,"images_per_job":IMAGE_COUNT,"cpu_budget":8,
        "io_budget":2,"memory_budget_bytes":args.memory_mib*MIB,"stage_estimate_bytes":SCRATCH,
        "requested_threads":4,"sift_options_debug":format!("{options:#?}"),
        "cpu_image_adapter":false,"snapshot_interval_ms":10});
    save(&args.output.join("config.json"), &config)?;
    let databases: Vec<_> = (0..args.jobs)
        .map(|id| args.output.join(format!("job-{id}.db")))
        .collect();
    for (id, path) in databases.iter().enumerate() {
        save(
            &args.output.join(format!("metadata-{id}.json")),
            &prepare(path, &images)?,
        )?;
    }
    let prepared_ms = epoch.elapsed().as_secs_f64() * 1000.0;
    let runtime = Arc::new(Runtime::new(RuntimeConfig {
        budget: Budget {
            cpu_threads: 8,
            memory_bytes: args.memory_mib * MIB,
            io_slots: 2,
        },
        ..Default::default()
    })?);
    let executor = SfmTaskflow::new(runtime.clone(), SCRATCH)?;
    let mut samples = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(args.output.join("snapshots.jsonl"))?;
    let initial = sample_line(&mut samples, &runtime, epoch, "initial")?;
    ensure!(initial["drained"] == true, "initial resource leak");
    let (stop_tx, stop_rx) = mpsc::channel();
    let sample_runtime = runtime.clone();
    let sampler = std::thread::spawn(move || -> Result<()> {
        loop {
            sample_line(&mut samples, &sample_runtime, epoch, "running")?;
            match stop_rx.recv_timeout(Duration::from_millis(10)) {
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                _ => break,
            }
        }
        sample_line(&mut samples, &sample_runtime, epoch, "joined")?;
        Ok(())
    });
    let barrier = Barrier::new(args.jobs as usize + 1);
    let mut submitted_ms = 0.0;
    let results = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (id, path) in databases.iter().enumerate() {
            let barrier = &barrier;
            let executor = executor.clone();
            let options = &options;
            let input = &args.input;
            handles.push(scope.spawn(move || -> Result<Value> {
                let control = SfmTaskControl::new();
                let mut events = Vec::new();
                let mut sink = |event| events.push(event);
                let mut task = SfmTaskContext::new(&control, &mut sink).with_taskflow(executor);
                barrier.wait();
                let start_ms = epoch.elapsed().as_secs_f64()*1000.0;
                let result = rustscan_sfm::extract_features_to_database_with_task(path, input, options, &mut task);
                let completed_ms = epoch.elapsed().as_secs_f64()*1000.0;
                let stages = task.stage_reports();
                drop(task);
                Ok(match result {
                    Ok(report) => json!({"id":id,"start_ms":start_ms,"completed_ms":completed_ms,
                        "latency_ms":completed_ms-start_ms,"stages":stages,"extraction":report,"events":events}),
                    Err(error) => json!({"id":id,"start_ms":start_ms,"completed_ms":completed_ms,
                        "stages":stages,"error":format!("{error:#}"),"events":events}),
                })
            }));
        }
        submitted_ms = epoch.elapsed().as_secs_f64() * 1000.0;
        barrier.wait();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .map_err(|_| anyhow::anyhow!("workflow panic"))
                    .and_then(|r| r)
            })
            .collect::<Vec<_>>()
    });
    let _ = stop_tx.send(());
    let sampling = sampler
        .join()
        .map_err(|_| anyhow::anyhow!("sampler panic"))
        .and_then(|r| r);
    let final_snapshot = snapshot(&runtime, epoch, "final")?;
    let mut jobs = Vec::new();
    let mut failures = Vec::new();
    for result in results {
        match result {
            Ok(job) => jobs.push(job),
            Err(error) => failures.push(format!("{error:#}")),
        }
    }
    if let Err(error) = sampling {
        failures.push(format!("sampler: {error:#}"));
    }
    // All extraction callers have returned before any quality reread. Process RSS
    // still includes this sequential, one-image-at-a-time validation and output.
    let validation_started_ms = epoch.elapsed().as_secs_f64() * 1000.0;
    for job in &mut jobs {
        let id = job["id"].as_u64().unwrap() as usize;
        if job.get("error").is_none() {
            match quality(&databases[id], &images) {
                Ok(value) => job["quality"] = value,
                Err(error) => failures.push(format!("quality job {id}: {error:#}")),
            }
        }
    }
    let clean = final_snapshot["drained"] == true;
    let makespan_ms = jobs
        .iter()
        .filter_map(|j| j["completed_ms"].as_f64())
        .fold(submitted_ms, f64::max)
        - submitted_ms;
    let report = json!({"config":config,"prepared_ms":prepared_ms,"submitted_ms":submitted_ms,
        "makespan_ms":makespan_ms,"validation_started_ms":validation_started_ms,
        "validation_finished_ms":epoch.elapsed().as_secs_f64()*1000.0,
        "jobs":jobs,"initial_snapshot":initial,"final_snapshot":final_snapshot,"clean":clean,"failures":failures});
    save(&args.output.join("report.json"), &report)?;
    ensure!(
        clean && failures.is_empty(),
        "C5 cleanup/sampling/quality failure"
    );
    ensure!(
        jobs.len() == args.jobs as usize && jobs.iter().all(|j| j.get("error").is_none()),
        "C5 extraction failed"
    );
    println!(
        "C5 complete: jobs={} makespan_ms={makespan_ms:.3} clean={clean}",
        args.jobs
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_covers_all_bits_bytes_and_shape() -> Result<()> {
        let kp = ColmapKeypoint::new(0.0, 1.0);
        let d = ColmapDescriptors::new(0, 1, 128, vec![7; 128])?;
        let base = hashes(&[kp], &d)?;
        for index in 0..6 {
            let mut fields = [kp.x, kp.y, kp.a11, kp.a12, kp.a21, kp.a22];
            fields[index] = f32::from_bits(fields[index].to_bits() ^ 1);
            let changed = ColmapKeypoint {
                x: fields[0],
                y: fields[1],
                a11: fields[2],
                a12: fields[3],
                a21: fields[4],
                a22: fields[5],
            };
            assert_ne!(base, hashes(&[changed], &d)?);
        }
        let mut negative_zero = kp;
        negative_zero.x = -0.0;
        assert_ne!(base, hashes(&[negative_zero], &d)?);
        for index in 0..128 {
            let mut changed = d.clone();
            changed.data[index] ^= 1;
            assert_ne!(base, hashes(&[kp], &changed)?);
        }
        let mut invalid = d.clone();
        invalid.cols = 64;
        assert!(hashes(&[kp], &invalid).is_err());
        let mut invalid = kp;
        invalid.a22 = f32::NAN;
        assert!(hashes(&[invalid], &d).is_err());
        Ok(())
    }

    #[test]
    fn output_refuses_overwrite_and_selection_is_sorted() -> Result<()> {
        let dir = tempfile::tempdir()?;
        for id in (0..10).rev() {
            File::create_new(dir.path().join(format!("{id:02}.jpg")))?;
        }
        File::create_new(dir.path().join("ignore.txt"))?;
        let selected = select_images(dir.path())?;
        assert_eq!(selected.len(), 8);
        assert_eq!(selected[0].file_name().unwrap(), "00.jpg");
        assert_eq!(selected[7].file_name().unwrap(), "07.jpg");
        let path = dir.path().join("report.json");
        save(&path, &json!({"original":true}))?;
        assert!(save(&path, &json!({"original":false})).is_err());
        assert_eq!(
            serde_json::from_slice::<Value>(&std::fs::read(path)?)?,
            json!({"original":true})
        );
        Ok(())
    }
}
