use super::super::{
    mapper_uses_gpu, prepare_mapper_features, run_reconstruction_with_task, MapperConfig,
};
use super::*;
use crate::feature_extraction::retained_memory_plan;
use crate::task::SfmTaskContext;
use std::sync::Arc;

fn executor(memory: u64, floor: u64, threads: usize) -> Result<crate::SfmTaskflow> {
    crate::SfmTaskflow::new(
        Arc::new(rustscan_taskflow::Runtime::new(
            rustscan_taskflow::RuntimeConfig {
                budget: rustscan_taskflow::Budget {
                    cpu_threads: threads,
                    memory_bytes: memory,
                    io_slots: 1,
                },
                ..Default::default()
            },
        )?),
        floor,
    )
}

fn config(input: &std::path::Path) -> MapperConfig {
    MapperConfig {
        input: input.to_owned(),
        discover_database: false,
        feature_type: FeatureType::Sift,
        max_features: 32,
        threads: Some(2),
        local_matching: true,
        ..Default::default()
    }
}

fn small_images(dir: &std::path::Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for id in 0..3 {
        let path = dir.join(format!("{id}.png"));
        image::RgbImage::from_fn(128, 128, |x, y| {
            let checker = if (x / 12 + y / 12) % 2 == 0 { 40 } else { 180 };
            let value = (checker + (x * 17 + y * 31 + id * 7) % 53) as u8;
            image::Rgb([value, value.saturating_add(13), value.saturating_sub(11)])
        })
        .save(&path)?;
        paths.push(path);
    }
    Ok(paths)
}

fn float_bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn keypoint_bits(points: &[rustslam::KeyPoint]) -> Vec<([u32; 5], i32)> {
    points
        .iter()
        .map(|p| {
            (
                [
                    p.x().to_bits(),
                    p.y().to_bits(),
                    p.size.to_bits(),
                    p.angle.to_bits(),
                    p.response.to_bits(),
                ],
                p.octave,
            )
        })
        .collect()
}

fn assert_frame_bits(a: &ImageFrame, b: &ImageFrame) {
    assert_eq!(
        (a.id, &a.name, &a.path, a.width, a.height),
        (b.id, &b.name, &b.path, b.width, b.height)
    );
    assert_eq!(keypoint_bits(&a.keypoints), keypoint_bits(&b.keypoints));
    assert_eq!(
        (&a.descriptors.data, a.descriptors.size, a.descriptors.count),
        (&b.descriptors.data, b.descriptors.size, b.descriptors.count)
    );
    assert_eq!(
        keypoint_bits(&a.sift.keypoints),
        keypoint_bits(&b.sift.keypoints)
    );
    assert_eq!(a.sift.descriptors.len(), b.sift.descriptors.len());
    for (a, b) in a.sift.descriptors.iter().zip(&b.sift.descriptors) {
        assert_eq!(float_bits(a.as_slice()), float_bits(b.as_slice()));
    }
    let colmap = |frame: &ImageFrame| {
        frame
            .sift
            .colmap_keypoints
            .iter()
            .map(|p| [p.x, p.y, p.a11, p.a12, p.a21, p.a22].map(f32::to_bits))
            .collect::<Vec<_>>()
    };
    assert_eq!(colmap(a), colmap(b));
    assert_eq!(a.sift.descriptors_u8, b.sift.descriptors_u8);
    assert_eq!(
        (a.wide_descriptors.dim, a.wide_descriptors.count),
        (b.wide_descriptors.dim, b.wide_descriptors.count)
    );
    assert_eq!(
        float_bits(&a.wide_descriptors.data),
        float_bits(&b.wide_descriptors.data)
    );
    assert_eq!(a.strong_feature_indices, b.strong_feature_indices);
    assert_eq!(a.colors, b.colors);
}

#[test]
#[cfg(all(feature = "vlfeat-sift", not(feature = "lowe-sift-backend")))]
fn composition_memory_mapper_planned_frames_match_legacy_bits_and_profile() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let paths = small_images(dir.path())?;
    let control = SfmTaskControl::new();
    let options = SiftExtractionOptions {
        max_num_features: 32,
        ..Default::default()
    };
    let mut sink = |_| {};
    let mut task =
        SfmTaskContext::new(&control, &mut sink).with_taskflow(executor(64 << 20, 128, 2)?);
    let (legacy, _) = task.execute("legacy", false, 2, |_| {
        extract_frames_profiled(
            &paths,
            32,
            FeatureType::Sift,
            &options,
            None,
            &control,
            false,
        )
    })?;
    assert!(legacy
        .iter()
        .all(|frame| !frame.sift.descriptors.is_empty()));
    for budget in [0, 64 << 20] {
        let plan = retained_memory_plan(&paths, &options, true, 2, budget, 128, &control)?.unwrap();
        assert_eq!(plan.batch_size(), if budget == 0 { 1 } else { 2 });
        let original_batch = plan.batch_size();
        // The second plan has two-image chunks but only one granted CPU.
        let (planned, profile) =
            task.execute_with_memory_and_gpu("planned", false, plan.request_bytes(), 1, |_| {
                assert_eq!(crate::execution::active_threads(), Some(1));
                extract_frames_profiled(
                    &[],
                    32,
                    FeatureType::Sift,
                    &options,
                    Some(&plan),
                    &control,
                    true,
                )
            })?;
        assert_eq!(plan.batch_size(), original_batch);
        assert_eq!(planned.len(), legacy.len());
        for (id, (a, b)) in legacy.iter().zip(&planned).enumerate() {
            assert_eq!(b.id, id);
            assert_frame_bits(a, b);
        }
        let profile = profile.unwrap();
        assert_eq!(profile.image_count, paths.len());
        assert_eq!(
            profile
                .image_timings
                .iter()
                .map(|t| t.image.as_str())
                .collect::<Vec<_>>(),
            vec!["0.png", "1.png", "2.png"]
        );
        assert!(profile.wall_clock_span_ms >= 0.0 && profile.p95_total_ms >= profile.p50_total_ms);
        assert_eq!(task.feature_memory_limits()?.0, 128);
    }
    Ok(())
}

#[test]
#[cfg(all(feature = "vlfeat-sift", not(feature = "lowe-sift-backend")))]
fn composition_memory_mapper_planned_chunk_preserves_load_errors() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let paths = small_images(dir.path())?;
    let control = SfmTaskControl::new();
    let options = SiftExtractionOptions {
        max_num_features: 32,
        ..Default::default()
    };
    let plan = retained_memory_plan(&paths, &options, true, 2, 0, 128, &control)?.unwrap();
    assert_eq!(plan.batch_size(), 1);
    std::fs::remove_file(&paths[1])?;
    let mut sink = |_| {};
    let mut task =
        SfmTaskContext::new(&control, &mut sink).with_taskflow(executor(64 << 20, 128, 1)?);
    let error = task
        .execute_with_memory_and_gpu("planned error", false, plan.request_bytes(), 1, |_| {
            extract_frames(
                &paths,
                32,
                FeatureType::Sift,
                &options,
                Some(&plan),
                &control,
            )
        })
        .unwrap_err();
    assert!(format!("{error:#}").contains(&format!("failed to load {}", paths[1].display())));
    assert!(task
        .stage_reports()
        .iter()
        .any(|r| r.stage_name == "planned error" && r.cancelled_or_failed));
    Ok(())
}

#[test]
#[cfg(all(feature = "vlfeat-sift", not(feature = "lowe-sift-backend")))]
fn composition_memory_mapper_waiting_admission_rejects_replaced_header() -> Result<()> {
    use std::sync::mpsc;
    use std::time::Duration;

    let dir = tempfile::tempdir()?;
    let paths = [dir.path().join("0.png"), dir.path().join("1.png")];
    for path in &paths {
        image::RgbImage::from_pixel(16, 16, image::Rgb([40, 80, 120])).save(path)?;
    }
    let original_image = std::fs::read(&paths[0])?;
    let mut cfg = config(dir.path());
    cfg.max_features = 8;
    let executor = executor(8 << 20, 128, 1)?;
    let runtime = executor.runtime().clone();
    let ceiling = runtime.snapshot()?.budget;
    let mut low = ceiling;
    low.memory_bytes = 128;
    runtime.set_budget(low)?;
    let control = SfmTaskControl::new();
    let mut sink = |_| {};
    let task = SfmTaskContext::new(&control, &mut sink).with_taskflow(executor.clone());
    let prepared = prepare_mapper_features(&cfg, &task)?.unwrap();
    let original_plan = prepared.plan.as_ref().unwrap().clone();
    let original_signature = format!("{original_plan:?}");
    let request = original_plan.request_bytes();
    assert!(request > low.memory_bytes && request <= ceiling.memory_bytes);
    assert_eq!(original_plan.batch_size(), 1);
    original_plan.validate_inputs(&control)?;

    let (done, result) = mpsc::sync_channel(1);
    std::thread::scope(|scope| -> Result<()> {
        let worker = scope.spawn(|| {
            let mut sink = |_| {};
            let mut task = SfmTaskContext::new(&control, &mut sink).with_taskflow(executor);
            let reports = task.stage_report_sink();
            let status = task.execute_with_memory_and_gpu(
                "reconstruction",
                mapper_uses_gpu(&cfg),
                request,
                cfg.threads.unwrap_or(4),
                |task| {
                    let mut events = super::super::MapperEventBridge::Task(task);
                    super::super::run_reconstruction_prepared(
                        &cfg,
                        &mut events,
                        reports,
                        Some(prepared),
                    )
                },
            );
            let _ = done.send((status, task.stage_reports(), task.feature_memory_limits()));
        });
        let outcome = (|| -> Result<()> {
            let deadline = Instant::now() + Duration::from_secs(5);
            while runtime.snapshot()?.pending_tasks == 0 {
                anyhow::ensure!(Instant::now() < deadline, "mapper admission did not queue");
                std::thread::sleep(Duration::from_millis(1));
            }
            let queued = runtime.snapshot()?;
            anyhow::ensure!(
                queued.memory_bytes == 0 && queued.cpu_threads == 0,
                "mapper acquired resources before budget recovery"
            );
            anyhow::ensure!(
                matches!(result.try_recv(), Err(mpsc::TryRecvError::Empty)),
                "mapper completed before budget recovery"
            );

            // Replace at the same path only after its original request is queued.
            // All images remain bounded and valid; the larger one is just 64x64.
            image::RgbImage::from_pixel(64, 64, image::Rgb([10, 20, 30])).save(&paths[0])?;
            anyhow::ensure!(image::image_dimensions(&paths[0])? == (64, 64));
            runtime.set_budget(ceiling)?;
            let (status, reports, limits) = result.recv_timeout(Duration::from_secs(5))?;
            let error = status.unwrap_err();
            let message = format!("{error:#}");
            anyhow::ensure!(
                message.contains("mapper planned inputs changed before decode"),
                "{message}"
            );
            anyhow::ensure!(
                message.contains("planned extraction header differs from the memory plan"),
                "{message}"
            );
            anyhow::ensure!(
                message.contains(&paths[0].display().to_string()),
                "{message}"
            );
            anyhow::ensure!(
                reports.len() == 1,
                "unexpected readmission or downstream stage: {reports:?}"
            );
            let report = &reports[0];
            anyhow::ensure!(report.stage_name == "reconstruction" && report.cancelled_or_failed);
            anyhow::ensure!(
                report.requested_memory == request && report.granted_memory == request,
                "request expanded after header replacement: {report:?}"
            );
            anyhow::ensure!(
                report.granted_threads > 0,
                "validation must run after admission"
            );
            anyhow::ensure!(limits?.0 == 128, "configured floor was changed");
            Ok(())
        })();
        // Always unblock the worker on polling, replacement or assertion errors.
        control.request_cancel();
        worker.join().expect("mapper admission worker panicked");
        outcome
    })?;

    let usage = runtime.snapshot()?;
    assert_eq!(
        (
            usage.cpu_threads,
            usage.memory_bytes,
            usage.io_slots,
            usage.pending_tasks
        ),
        (0, 0, 0, 0)
    );
    assert!(usage.exclusive.is_empty());
    assert_eq!(usage.budget.memory_bytes, ceiling.memory_bytes);
    assert_eq!(format!("{original_plan:?}"), original_signature);
    let fresh_control = SfmTaskControl::new();
    assert!(original_plan.validate_inputs(&fresh_control).is_err());
    std::fs::write(&paths[0], original_image)?;
    original_plan.validate_inputs(&fresh_control)?;
    assert_eq!(format!("{original_plan:?}"), original_signature);
    Ok(())
}

#[test]
fn composition_memory_mapper_fallback_and_cancellation_do_not_read_headers() -> Result<()> {
    let control = SfmTaskControl::new();
    let mut sink = |_| {};
    let cfg = config(std::path::Path::new("absent-mapper-input"));
    let unbound = SfmTaskContext::new(&control, &mut sink);
    assert!(prepare_mapper_features(&cfg, &unbound)?.is_none());
    let mut sink = |_| {};
    let task = SfmTaskContext::new(&control, &mut sink).with_taskflow(executor(1024, 128, 1)?);
    let mut orb = cfg.clone();
    orb.feature_type = FeatureType::Orb;
    assert!(prepare_mapper_features(&orb, &task)?.is_none());
    let mut unsupported = cfg.clone();
    unsupported.sift_extraction.force_covariant_extractor = true;
    assert!(prepare_mapper_features(&unsupported, &task)?.is_none());
    if !cfg!(all(
        feature = "vlfeat-sift",
        not(feature = "lowe-sift-backend")
    )) {
        assert!(prepare_mapper_features(&cfg, &task)?.is_none());
    }
    control.request_cancel();
    assert!(prepare_mapper_features(&cfg, &task)
        .err()
        .unwrap()
        .downcast_ref::<crate::task::SfmTaskStop>()
        .is_some());
    assert!(extract_frames(
        &[PathBuf::from("absent.png")],
        32,
        FeatureType::Sift,
        &cfg.sift_extraction,
        None,
        &control
    )
    .unwrap_err()
    .downcast_ref::<crate::task::SfmTaskStop>()
    .is_some());
    Ok(())
}

#[test]
#[cfg(all(feature = "vlfeat-sift", not(feature = "lowe-sift-backend")))]
fn composition_memory_mapper_selection_and_missing_output_db_use_retained_plan() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let paths = small_images(dir.path())?;
    // Excluded inputs must not enter the header plan.
    std::fs::write(&paths[2], b"not an image")?;
    let control = SfmTaskControl::new();
    let mut sink = |_| {};
    let task = SfmTaskContext::new(&control, &mut sink).with_taskflow(executor(64 << 20, 128, 2)?);
    let mut cfg = config(dir.path());
    cfg.max_images = Some(2);
    cfg.database = Some(dir.path().join("new.db"));
    cfg.write_database = true;
    cfg.sift_extraction.max_num_features = 999;
    let prepared = prepare_mapper_features(&cfg, &task)?.unwrap();
    assert!(prepared.database.is_none());
    assert!(!cfg.database.as_ref().unwrap().exists());
    let plan = prepared.plan.unwrap();
    assert_eq!(plan.paths(), &paths[..2]);
    assert_eq!(plan.options().max_num_features, cfg.max_features);
    assert!(plan.retained_bytes() > 0);
    Ok(())
}

#[test]
#[cfg(all(feature = "vlfeat-sift", not(feature = "lowe-sift-backend")))]
fn composition_memory_mapper_cached_db_skips_invalid_image_headers() -> Result<()> {
    use crate::database::{
        ColmapDatabase, ColmapDatabaseCamera, ColmapDatabaseImage, ColmapKeypoint,
        ColmapTwoViewGeometry,
    };
    let dir = tempfile::tempdir()?;
    let db_path = dir.path().join("database.db");
    let db = ColmapDatabase::open(&db_path)?;
    db.write_camera(
        &ColmapDatabaseCamera {
            camera: crate::colmap::ColmapCamera {
                camera_id: 1,
                model_id: crate::types::COLMAP_PINHOLE,
                width: 16,
                height: 12,
                params: vec![10.0, 10.0, 8.0, 6.0],
            },
            has_prior_focal_length: true,
        },
        true,
    )?;
    for id in 1..=2 {
        let name = format!("{id}.png");
        std::fs::write(dir.path().join(&name), b"cached; no valid header")?;
        db.write_image(
            &ColmapDatabaseImage {
                image_id: id,
                name,
                camera_id: 1,
                frame_id: None,
            },
            true,
        )?;
        db.write_keypoints(id, &[ColmapKeypoint::new(3.0, 4.0)])?;
    }
    db.write_two_view_geometry(
        1,
        2,
        &ColmapTwoViewGeometry {
            config: 2,
            inlier_matches: vec![crate::correspondence_graph::FeatureMatch::new(0, 0)],
            ..Default::default()
        },
    )?;
    drop(db);
    let control = SfmTaskControl::new();
    let mut sink = |_| {};
    let task = SfmTaskContext::new(&control, &mut sink).with_taskflow(executor(1024, 128, 1)?);
    let mut cfg = config(dir.path());
    cfg.discover_database = true;
    cfg.min_matches = 0;
    let prepared = prepare_mapper_features(&cfg, &task)?.unwrap();
    let plan = prepared.plan.as_ref().unwrap();
    assert!(plan.paths().is_empty());
    assert_eq!(plan.request_bytes(), 128);
    assert_eq!(plan.retained_bytes(), 0);
    let frames =
        super::super::database_frames(&prepared.paths, prepared.database.as_ref().unwrap())?;
    assert_eq!(frames.len(), 2);
    assert!(frames
        .iter()
        .all(|f| f.width == 16 && f.keypoints.len() == 1 && f.sift.descriptors.is_empty()));
    Ok(())
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb88320 & (0u32.wrapping_sub(crc & 1)));
        }
    }
    !crc
}

#[test]
#[cfg(all(feature = "vlfeat-sift", not(feature = "lowe-sift-backend")))]
fn composition_memory_mapper_oversize_header_rejects_before_decode_and_reports() -> Result<()> {
    let dir = tempfile::tempdir()?;
    for id in 0..2 {
        let path = dir.path().join(format!("{id}.png"));
        image::GrayImage::new(1, 1).save(&path)?;
        let mut png = std::fs::read(&path)?;
        png[16..20].copy_from_slice(&16384u32.to_be_bytes());
        png[20..24].copy_from_slice(&16384u32.to_be_bytes());
        let crc = crc32(&png[12..29]);
        png[29..33].copy_from_slice(&crc.to_be_bytes());
        std::fs::write(&path, png)?;
        assert_eq!(image::image_dimensions(&path)?, (16384, 16384));
        // Never decode: IDAT contains only the original one-pixel payload.
    }
    let cfg = config(dir.path());
    let control = SfmTaskControl::new();
    let mut sink = |_| {};
    let mut task =
        SfmTaskContext::new(&control, &mut sink).with_taskflow(executor(1 << 20, 128, 2)?);
    let error = run_reconstruction_with_task(&cfg, &mut task).unwrap_err();
    assert!(format!("{error:#}").contains("before decode"));
    let reports = task.stage_reports();
    let report = reports
        .iter()
        .find(|r| r.stage_name == "reconstruction")
        .unwrap();
    assert!(report.requested_memory > 1 << 20);
    assert_eq!(report.granted_memory, 0);
    assert_eq!(report.service_ms, 0.0);
    assert!(report.cancelled_or_failed);
    assert_eq!(task.feature_memory_limits()?.0, 128);
    Ok(())
}

#[test]
#[cfg(all(feature = "vlfeat-sift", not(feature = "lowe-sift-backend")))]
fn composition_memory_mapper_parent_grant_rejects_without_readmission() -> Result<()> {
    let dir = tempfile::tempdir()?;
    small_images(dir.path())?;
    let cfg = config(dir.path());
    let control = SfmTaskControl::new();
    let mut sink = |_| {};
    let mut task =
        SfmTaskContext::new(&control, &mut sink).with_taskflow(executor(64 << 20, 128, 2)?);
    task.execute("parent", false, 1, |task| {
        let error = run_reconstruction_with_task(&cfg, task).unwrap_err();
        assert!(format!("{error:#}").contains("exceeds parent grant 128 bytes before decode"));
        assert_eq!(crate::execution::active_memory_bytes(), Some(128));
        let reports = task.stage_reports();
        let report = reports
            .iter()
            .find(|r| r.stage_name == "reconstruction")
            .unwrap();
        assert_eq!(report.granted_threads, 0);
        assert_eq!(report.granted_memory, 0);
        assert!(report.cancelled_or_failed);
        Ok(())
    })?;
    Ok(())
}

#[test]
fn composition_memory_mapper_gpu_union_and_floor_survive_planned_admission() -> Result<()> {
    let control = SfmTaskControl::new();
    let mut sink = |_| {};
    let mut task = SfmTaskContext::new(&control, &mut sink).with_taskflow(executor(1024, 128, 1)?);
    for route in 0..3 {
        let mut cfg = config(std::path::Path::new("unused"));
        cfg.sift_extraction.use_gpu = route == 0;
        cfg.sift_matching.use_gpu = route == 1;
        cfg.use_gpu_pnp = route == 2;
        assert!(mapper_uses_gpu(&cfg));
        task.execute_with_memory_and_gpu(
            "reconstruction",
            mapper_uses_gpu(&cfg),
            256,
            1,
            |task| task.execute("synthetic GPU child", true, 1, |_| Ok(())),
        )?;
        assert_eq!(task.feature_memory_limits()?.0, 128);
    }
    assert!(!mapper_uses_gpu(&config(std::path::Path::new("unused"))));
    Ok(())
}
