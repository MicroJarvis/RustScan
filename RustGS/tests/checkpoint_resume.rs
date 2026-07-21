use std::fs;

use rustgs::{
    load_training_checkpoint, save_training_checkpoint, AdamCheckpoint, AdamParameterCheckpoint,
    HostSplats, Intrinsics, TensorCheckpoint, TopologyCheckpoint, TrainingCheckpoint,
    TrainingConfig, TrainingDataset, TrainingError, TrainingIdentity, TRAINING_CHECKPOINT_VERSION,
};

fn tensor(values: &[f32]) -> TensorCheckpoint {
    TensorCheckpoint {
        shape: vec![values.len()],
        values: values.to_vec(),
    }
}

fn adam_parameter(step: usize) -> AdamParameterCheckpoint {
    AdamParameterCheckpoint {
        moment1: Some(tensor(&[0.1])),
        moment2: Some(tensor(&[0.2])),
        scaling: Some(tensor(&[0.3])),
        step,
    }
}

fn checkpoint_fixture(completed_iterations: usize) -> TrainingCheckpoint {
    let topology_tensor = tensor(&[0.25]);
    TrainingCheckpoint {
        version: TRAINING_CHECKPOINT_VERSION,
        identity: TrainingIdentity {
            dataset: "dataset-a".to_string(),
            reconstruction: "reconstruction-a".to_string(),
            config: "config-a".to_string(),
        },
        completed_iterations,
        latest_loss: Some(0.125),
        splats: HostSplats::from_components(
            vec![1.0, 2.0, 3.0],
            vec![-1.0, -1.0, -1.0],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0],
            vec![0.25, 0.5, 0.75],
            0,
        )
        .unwrap(),
        optimizer: AdamCheckpoint {
            transforms: adam_parameter(completed_iterations),
            sh_coeffs: adam_parameter(completed_iterations),
            raw_opacities: adam_parameter(completed_iterations),
        },
        topology: TopologyCheckpoint {
            grad_2d: topology_tensor.clone(),
            screen_grad_2d: topology_tensor.clone(),
            abs_grad_2d: topology_tensor.clone(),
            abs_pixel_grad_2d: topology_tensor.clone(),
            pixel_coverage: topology_tensor.clone(),
            camera_depth: topology_tensor.clone(),
            grad_color: topology_tensor.clone(),
            num_observations: topology_tensor.clone(),
            visible_observations: topology_tensor.clone(),
            actual_visible_observations: topology_tensor,
            splat_birth_iterations: vec![0],
            splat_invisible_windows: vec![1],
        },
        frame_shuffle_seed: 7,
        active_sh_degree: 0,
    }
}

fn write_unchecked(path: &std::path::Path, checkpoint: &TrainingCheckpoint) {
    fs::write(path, bincode::serialize(checkpoint).unwrap()).unwrap();
}

fn assert_invalid_input_contains(error: TrainingError, expected: &str) {
    match error {
        TrainingError::InvalidInput(message) => assert!(
            message.contains(expected),
            "expected invalid-input message to contain {expected:?}, got {message:?}"
        ),
        other => panic!("expected TrainingError::InvalidInput, got {other:?}"),
    }
}

#[test]
fn checkpoint_store_round_trips_and_atomically_replaces() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("iteration-000010.rgscp");
    let first = checkpoint_fixture(10);
    save_training_checkpoint(&path, &first).unwrap();
    assert_eq!(load_training_checkpoint(&path).unwrap(), first);

    let second = checkpoint_fixture(20);
    save_training_checkpoint(&path, &second).unwrap();
    assert_eq!(load_training_checkpoint(&path).unwrap(), second);
    assert!(!path.with_extension("rgscp.tmp").exists());
}

#[test]
fn checkpoint_store_cleans_up_temporary_file_after_io_error() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("blocked.rgscp");
    fs::create_dir(&path).unwrap();
    fs::write(path.join("keep"), b"not empty").unwrap();

    let error = save_training_checkpoint(&path, &checkpoint_fixture(10)).unwrap_err();

    assert!(matches!(error, TrainingError::Io(_)));
    assert!(!path.with_extension("rgscp.tmp").exists());
}

#[test]
fn checkpoint_load_rejects_truncated_data() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("truncated.rgscp");
    fs::write(&path, [1, 2, 3]).unwrap();

    assert_invalid_input_contains(
        load_training_checkpoint(&path).unwrap_err(),
        "decode checkpoint",
    );
}

#[test]
fn checkpoint_load_rejects_wrong_version() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("wrong-version.rgscp");
    let mut checkpoint = checkpoint_fixture(10);
    checkpoint.version = TRAINING_CHECKPOINT_VERSION + 1;
    write_unchecked(&path, &checkpoint);

    assert_invalid_input_contains(
        load_training_checkpoint(&path).unwrap_err(),
        "checkpoint version",
    );
}

#[test]
fn checkpoint_load_rejects_non_finite_values() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("non-finite.rgscp");
    let mut checkpoint = checkpoint_fixture(10);
    checkpoint.latest_loss = Some(f32::NAN);
    write_unchecked(&path, &checkpoint);

    assert_invalid_input_contains(
        load_training_checkpoint(&path).unwrap_err(),
        "latest loss must be finite",
    );

    checkpoint.latest_loss = Some(0.125);
    checkpoint
        .optimizer
        .transforms
        .moment1
        .as_mut()
        .unwrap()
        .values[0] = f32::INFINITY;
    write_unchecked(&path, &checkpoint);
    assert_invalid_input_contains(
        load_training_checkpoint(&path).unwrap_err(),
        "tensor values must be finite",
    );
}

#[test]
fn checkpoint_validation_rejects_shape_overflow_and_length_mismatch() {
    let mut checkpoint = checkpoint_fixture(10);
    let moment = checkpoint.optimizer.transforms.moment1.as_mut().unwrap();
    moment.shape = vec![usize::MAX, 2];
    moment.values.clear();
    assert_invalid_input_contains(checkpoint.validate().unwrap_err(), "tensor shape overflows");

    let mut checkpoint = checkpoint_fixture(10);
    checkpoint.topology.splat_birth_iterations.push(3);
    assert_invalid_input_contains(checkpoint.validate().unwrap_err(), "splat_birth_iterations");
}

#[test]
fn checkpoint_validation_rejects_empty_identity_fields() {
    let mut checkpoint = checkpoint_fixture(10);
    checkpoint.identity.dataset.clear();

    assert_invalid_input_contains(checkpoint.validate().unwrap_err(), "identity dataset");
}

#[test]
fn checkpoint_identity_hashes_inputs_but_ignores_iteration_target() {
    let mut dataset = TrainingDataset::new(Intrinsics::new(12.0, 12.0, 8.0, 8.0, 16, 16));
    dataset.add_point([0.0, 0.0, 2.0], Some([0.25, 0.5, 0.75]));
    let config = TrainingConfig {
        iterations: 10,
        ..Default::default()
    };

    let first = TrainingIdentity::from_inputs(&dataset, "reconstruction-a", &config).unwrap();
    let second = TrainingIdentity::from_inputs(
        &dataset,
        "reconstruction-a",
        &TrainingConfig {
            iterations: 20,
            ..config.clone()
        },
    )
    .unwrap();
    assert_eq!(first, second);
    assert_eq!(first.dataset.len(), 64);
    assert_eq!(first.config.len(), 64);
    assert_eq!(first.reconstruction, "reconstruction-a");

    let mut changed_dataset = dataset.clone();
    changed_dataset.add_point([1.0, 0.0, 2.0], None);
    let changed_dataset_identity =
        TrainingIdentity::from_inputs(&changed_dataset, "reconstruction-a", &config).unwrap();
    assert_ne!(changed_dataset_identity.dataset, first.dataset);

    let mut changed_config = config.clone();
    changed_config.data.frame_shuffle_seed = 8;
    let changed_config_identity =
        TrainingIdentity::from_inputs(&dataset, "reconstruction-a", &changed_config).unwrap();
    assert_ne!(changed_config_identity.config, first.config);
}
