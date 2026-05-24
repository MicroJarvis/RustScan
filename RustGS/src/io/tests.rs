use super::TrainingCheckpoint;
use crate::sh::rgb_to_sh0_value;
use tempfile::tempdir;

#[test]
fn checkpoint_round_trips_host_splats() {
    let tempdir = tempdir().unwrap();
    let path = tempdir.path().join("checkpoint.json");
    let splats = crate::core::HostSplats::from_components(
        vec![0.0, 0.0, 1.0],
        vec![0.1f32.ln(), 0.1f32.ln(), 0.1f32.ln()],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0],
        vec![
            rgb_to_sh0_value(0.2),
            rgb_to_sh0_value(0.3),
            rgb_to_sh0_value(0.4),
        ],
        0,
    )
    .unwrap();
    let checkpoint = TrainingCheckpoint {
        iteration: 12,
        loss: 0.25,
        splats,
    };

    checkpoint.save(&path).unwrap();
    let loaded = TrainingCheckpoint::load(&path).unwrap();

    assert_eq!(loaded.iteration, 12);
    assert!((loaded.loss - 0.25).abs() < 1e-6);
    assert_eq!(loaded.splats.len(), 1);
}
