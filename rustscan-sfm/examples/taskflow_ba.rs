//! Two real Ceres solves plus CPU SIFT sharing one runtime. Synthetic in-memory
//! inputs; demonstrates admission and numerical checks, not an SfM speedup claim.
use anyhow::Result;
type Quat = nalgebra::UnitQuaternion<f32>;
type Vec3 = nalgebra::Vector3<f32>;
use rustscan_sfm::ba::{try_refine_bundle_adjustment, BundleAdjustmentOptions};
use rustscan_sfm::geometry::{UnitQuatNormalize, Vec3GlamExt};
use rustscan_sfm::sift::{SiftExtractionOptions, SiftFeatures};
use rustscan_sfm::types::{CameraModel, ImageFrame, Point3D, Reconstruction, TrackObservation};
use rustscan_sfm::wide::WideDescriptors;
use rustscan_sfm::CeresBaTaskflow;
use rustscan_slam::{Descriptors, KeyPoint, SE3};
use rustscan_taskflow::{
    Budget, CpuRequest, ResourceRequest, Runtime, RuntimeConfig, TaskGraph, TaskVariant,
};
use std::sync::{mpsc, Arc, Barrier};
use std::time::Duration;

fn fixture() -> (Vec<ImageFrame>, Reconstruction) {
    fixture_with_points(4000)
}

pub(crate) fn fixture_with_points(point_count: usize) -> (Vec<ImageFrame>, Reconstruction) {
    let camera = CameraModel::new_pinhole(512, 512, 350.0, 350.0, 256.0, 256.0);
    let mut frames: Vec<_> = (0..5)
        .map(|id| ImageFrame {
            id,
            name: format!("{id}.png"),
            path: format!("{id}.png").into(),
            width: 512,
            height: 512,
            keypoints: Vec::new(),
            descriptors: Descriptors::new(),
            sift: SiftFeatures::default(),
            wide_descriptors: WideDescriptors {
                data: Vec::new(),
                dim: 0,
                count: 0,
            },
            strong_feature_indices: Vec::new(),
            colors: Vec::new(),
        })
        .collect();
    let mut points = Vec::new();
    for id in 0..point_count {
        let [x, y, z] = [
            (id % 80) as f32 / 40.0 - 1.0,
            (id / 80) as f32 / 50.0 - 0.5,
            3.0 + (id % 11) as f32 * 0.15,
        ];
        for (image, frame) in frames.iter_mut().enumerate() {
            frame.keypoints.push(KeyPoint::new(
                256.0 + 350.0 * (x + image as f32 * 0.15) / z,
                256.0 + 350.0 * y / z,
            ));
        }
        points.push(Point3D {
            xyz: [x + 0.02, y - 0.01, z + 0.015],
            color: [0; 3],
            error: 0.0,
            track: (0..frames.len())
                .map(|image| TrackObservation { image, feature: id })
                .collect(),
        });
    }
    let reconstruction = Reconstruction {
        camera,
        cameras: vec![camera],
        camera_ids: vec![1],
        rigs: Vec::new(),
        frames: Vec::new(),
        image_names: frames.iter().map(|frame| frame.name.clone()).collect(),
        image_paths: frames.iter().map(|frame| frame.path.clone()).collect(),
        image_ids: (1..=5).collect(),
        image_camera_indices: vec![0; 5],
        image_frame_indices: vec![None; 5],
        poses: (0..5)
            .map(|image| {
                Some(SE3::from_quat_translation(
                    Quat::identity(),
                    Vec3::new(image as f32 * 0.15, 0.0, 0.0),
                ))
            })
            .collect(),
        observations: (0..5)
            .map(|_| (0..points.len()).map(Some).collect())
            .collect(),
        keypoints: frames.iter().map(|frame| frame.keypoints.clone()).collect(),
        point_ids: (1..=points.len() as u64).collect(),
        points,
    };
    (frames, reconstruction)
}

fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        anyhow::bail!("run with --release");
    }
    if !cfg!(feature = "ceres-ba") {
        anyhow::bail!("enable ceres-ba");
    }
    let runtime = Arc::new(Runtime::new(RuntimeConfig {
        budget: Budget {
            cpu_threads: 4,
            memory_bytes: 1024 * 1024 * 1024,
            io_slots: 2,
        },
        ..Default::default()
    })?);
    // A per-BA cap leaves room for another workflow; grants do not resize a
    // solve already in progress. SIFT consumes one of the four global slots.
    let executor = CeresBaTaskflow::new(runtime.clone(), 2, 256 * 1024 * 1024)?;
    let (frames, base) = fixture();
    let models = [base.clone(), base];
    let (entered, started) = mpsc::sync_channel(1);
    let mut graph = TaskGraph::new();
    let mut request = ResourceRequest::cpu(CpuRequest::fixed(1));
    request.working_memory_bytes = 256 * 1024 * 1024;
    let sift = graph.task(
        "CPU SIFT",
        vec![TaskVariant::cpu("sift", request, move |_| {
            entered.send(()).unwrap();
            let gray: Vec<u8> = (0..512 * 512)
                .map(|index| {
                    let x = index % 512;
                    let y = index / 512;
                    (((x * 13) ^ (y * 17) ^ ((x / 23 + y / 19) * 43)) % 256) as u8
                })
                .collect();
            rustscan_sfm::sift::extract_sift_from_grayscale_u8(
                &gray,
                512,
                512,
                &SiftExtractionOptions::default(),
            )
            .map(|features| features.keypoints.len())
            .map_err(|error| rustscan_taskflow::TaskError::Failed(format!("{error:#}")))
        })],
    )?;
    let sift_run = runtime.submit(graph)?;
    started.recv_timeout(Duration::from_secs(5))?;
    let barrier = Barrier::new(2);
    let reports = std::thread::scope(|scope| -> Result<Vec<_>> {
        let mut workers = Vec::new();
        for mut reconstruction in models {
            let executor = executor.clone();
            let frames = &frames;
            let barrier = &barrier;
            workers.push(scope.spawn(move || {
                barrier.wait();
                try_refine_bundle_adjustment(
                    frames,
                    &mut reconstruction,
                    BundleAdjustmentOptions {
                        iterations: 15,
                        constant_images: vec![0, 1],
                        // Deliberately exercise scalable grants on this small fixture;
                        // production retains its default 50,000-residual gate.
                        min_num_residuals_for_multi_threading: 0,
                        taskflow: Some(executor),
                        ..Default::default()
                    },
                )?
                .ok_or_else(|| anyhow::anyhow!("Ceres returned no report"))
            }));
        }
        workers
            .into_iter()
            .map(|worker| worker.join().expect("BA thread panic"))
            .collect()
    })?;
    for (index, report) in reports.iter().enumerate() {
        assert!(report.is_solution_usable());
        assert!(report.final_cost <= report.initial_cost);
        let scheduling = report.scheduling.as_ref().unwrap();
        assert_eq!(report.solver_num_threads, scheduling.granted_cpu_threads);
        assert!(report.solver_num_threads <= 2);
        println!("BA {index}: {}", report.brief_report());
    }
    assert!(sift_run
        .wait_timeout(Duration::from_secs(10))
        .unwrap()
        .succeeded());
    assert!(*sift_run.output(&sift)? > 0);
    let used = runtime.snapshot()?;
    assert_eq!(used.cpu_threads, 0);
    assert_eq!(used.memory_bytes, 0);
    assert_eq!(used.pending_tasks, 0);
    println!(
        "SIFT keypoints={}; remaining reservations={used:?}",
        *sift_run.output(&sift)?
    );
    Ok(())
}
