#![cfg(feature = "gpu")]

use std::cell::RefCell;
use std::rc::Rc;

use rustgs::{
    train_splats, Intrinsics, LiteGsOpacityResetMode, ScenePose, TrainingConfig, TrainingDataset,
    TrainingEvent, TrainingOptions, SE3,
};

#[test]
#[ignore = "requires a working wgpu adapter"]
fn tiny_wgpu_training_compiles_shaders_and_survives_optimizer_rebuild() {
    let temp = tempfile::tempdir().unwrap();
    let image_path = temp.path().join("frame.rgb");
    let mut pixels = Vec::with_capacity(16 * 16 * 3);
    for y in 0..16_u8 {
        for x in 0..16_u8 {
            pixels.extend_from_slice(&[
                x.saturating_mul(12),
                y.saturating_mul(12),
                x.saturating_add(y).saturating_mul(6),
            ]);
        }
    }
    std::fs::write(&image_path, pixels).unwrap();

    let mut dataset = TrainingDataset::new(Intrinsics::new(12.0, 12.0, 8.0, 8.0, 16, 16));
    dataset.add_pose(ScenePose::new(0, image_path, SE3::identity(), 0.0));
    dataset.add_point([0.0, 0.0, 2.0], Some([0.25, 0.5, 0.75]));
    dataset.add_point([0.25, -0.2, 2.5], Some([0.75, 0.25, 0.5]));

    let config = TrainingConfig {
        iterations: 3,
        raster: rustgs::TrainingRasterConfig {
            render_scale: 1.0,
            ..Default::default()
        },
        initialization: rustgs::TrainingInitializationConfig {
            max_initial_gaussians: 2,
            ..Default::default()
        },
        data: rustgs::TrainingDataConfig {
            frame_cache_capacity: 1,
            frame_prefetch_ahead: 1,
            ..Default::default()
        },
        litegs: rustgs::LiteGsConfig {
            topology: rustgs::LiteGsTopologyConfig {
                refine_every: 1,
                opacity_reset_interval: 1,
                opacity_reset_mode: LiteGsOpacityResetMode::Reset,
                topology_freeze_after_epoch: Some(2),
                target_primitives: 4,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let events = Rc::new(RefCell::new(Vec::new()));
    let captured_events = Rc::clone(&events);
    let run = train_splats(
        &dataset,
        &config,
        TrainingOptions::new().with_event_sink(move |event| {
            captured_events.borrow_mut().push(event);
        }),
    )
    .unwrap();

    assert_eq!(run.report.completed_iterations, 3);
    assert_eq!(run.report.final_loss, run.report.final_step_loss);
    assert!(run.report.final_loss.unwrap().is_finite());
    assert!(run.splats.len() <= config.litegs.topology.target_primitives);
    assert_eq!(
        run.report
            .telemetry
            .as_ref()
            .unwrap()
            .topology
            .opacity_reset_events,
        1
    );
    assert!(matches!(
        events.borrow().last(),
        Some(TrainingEvent::RunCompleted(_))
    ));
}
