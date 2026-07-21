use std::fs;
use std::io::Write;
use std::panic::catch_unwind;
use std::sync::{Arc, Barrier};

use bincode::Options;
use rustgs::{
    load_training_checkpoint, save_training_checkpoint, AdamCheckpoint, AdamParameterCheckpoint,
    HostSplats, Intrinsics, ScenePose, TensorCheckpoint, TopologyCheckpoint, TrainingCheckpoint,
    TrainingConfig, TrainingDataset, TrainingError, TrainingIdentity,
    MAX_TRAINING_CHECKPOINT_BYTES, MAX_TRAINING_CHECKPOINT_SPLATS,
    MAX_TRAINING_CHECKPOINT_TENSOR_ELEMENTS, MAX_TRAINING_CHECKPOINT_TENSOR_RANK,
    MAX_TRAINING_IDENTITY_BYTES, SE3, TRAINING_CHECKPOINT_FORMAT_VERSION,
    TRAINING_CHECKPOINT_MAGIC, TRAINING_CHECKPOINT_VERSION,
};
use serde::Serialize;

#[derive(Serialize)]
struct SerializedHostSplats {
    positions: Vec<f32>,
    log_scales: Vec<f32>,
    rotations: Vec<f32>,
    opacity_logits: Vec<f32>,
    sh_coeffs: Vec<f32>,
    sh_degree: usize,
}

#[derive(Serialize)]
struct SerializedTrainingCheckpoint {
    version: u32,
    identity: TrainingIdentity,
    completed_iterations: usize,
    latest_loss: Option<f32>,
    splats: SerializedHostSplats,
    optimizer: AdamCheckpoint,
    topology: TopologyCheckpoint,
    frame_shuffle_seed: u64,
    active_sh_degree: usize,
}

type SplatMutation = fn(&mut SerializedHostSplats);

fn tensor(values: &[f32]) -> TensorCheckpoint {
    TensorCheckpoint {
        shape: vec![values.len()],
        values: values.to_vec(),
    }
}

fn filled_tensor(shape: &[usize], value: f32) -> TensorCheckpoint {
    TensorCheckpoint {
        shape: shape.to_vec(),
        values: vec![value; shape.iter().product()],
    }
}

fn adam_parameter(
    step: usize,
    parameter_shape: &[usize],
    scaling_shape: &[usize],
) -> AdamParameterCheckpoint {
    AdamParameterCheckpoint {
        moment1: Some(filled_tensor(parameter_shape, 0.1)),
        moment2: Some(filled_tensor(parameter_shape, 0.2)),
        scaling: Some(filled_tensor(scaling_shape, 0.3)),
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
            transforms: adam_parameter(completed_iterations, &[1, 10], &[1, 10]),
            sh_coeffs: adam_parameter(completed_iterations, &[1, 1, 3], &[1, 1, 1]),
            raw_opacities: adam_parameter(completed_iterations, &[1], &[1]),
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
    fs::write(path, encode_unchecked(checkpoint)).unwrap();
}

fn encode_unchecked(value: &impl Serialize) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&TRAINING_CHECKPOINT_MAGIC);
    bytes.extend_from_slice(&TRAINING_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(MAX_TRAINING_CHECKPOINT_BYTES)
        .reject_trailing_bytes()
        .serialize_into(&mut bytes, value)
        .unwrap();
    bytes
}

fn serialize_with_splat_mutation(
    checkpoint: TrainingCheckpoint,
    mutate: impl FnOnce(&mut SerializedHostSplats),
) -> Vec<u8> {
    let view = checkpoint.splats.as_view();
    let mut splats = SerializedHostSplats {
        positions: view.positions.to_vec(),
        log_scales: view.log_scales.to_vec(),
        rotations: view.rotations.to_vec(),
        opacity_logits: view.opacity_logits.to_vec(),
        sh_coeffs: view.sh_coeffs.to_vec(),
        sh_degree: view.sh_degree,
    };
    mutate(&mut splats);
    encode_unchecked(&SerializedTrainingCheckpoint {
        version: checkpoint.version,
        identity: checkpoint.identity,
        completed_iterations: checkpoint.completed_iterations,
        latest_loss: checkpoint.latest_loss,
        splats,
        optimizer: checkpoint.optimizer,
        topology: checkpoint.topology,
        frame_shuffle_seed: checkpoint.frame_shuffle_seed,
        active_sh_degree: checkpoint.active_sh_degree,
    })
}

fn serialize_with_sh_degree(checkpoint: TrainingCheckpoint, sh_degree: usize) -> Vec<u8> {
    serialize_with_splat_mutation(checkpoint, |splats| splats.sh_degree = sh_degree)
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
fn checkpoint_store_writes_a_versioned_format_envelope() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("envelope.rgscp");
    save_training_checkpoint(&path, &checkpoint_fixture(10)).unwrap();

    let bytes = fs::read(path).unwrap();
    assert_eq!(
        &bytes[..TRAINING_CHECKPOINT_MAGIC.len()],
        &TRAINING_CHECKPOINT_MAGIC
    );
    assert_eq!(
        &bytes[TRAINING_CHECKPOINT_MAGIC.len()..TRAINING_CHECKPOINT_MAGIC.len() + 4],
        &TRAINING_CHECKPOINT_FORMAT_VERSION.to_le_bytes()
    );
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
fn checkpoint_store_supports_concurrent_writers_to_one_target() {
    const WRITERS: usize = 12;

    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("concurrent.rgscp");
    let barrier = Arc::new(Barrier::new(WRITERS));
    let handles: Vec<_> = (0..WRITERS)
        .map(|writer| {
            let path = path.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let checkpoint = checkpoint_fixture(writer + 1);
                barrier.wait();
                save_training_checkpoint(&path, &checkpoint)
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap().unwrap();
    }
    let loaded = load_training_checkpoint(&path).unwrap();
    assert!((1..=WRITERS).contains(&loaded.completed_iterations));
    assert_eq!(
        fs::read_dir(temp.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"))
            .count(),
        0
    );
}

#[test]
fn checkpoint_store_does_not_clobber_stale_or_same_stem_temporary_files() {
    let temp = tempfile::tempdir().unwrap();
    let checkpoint_path = temp.path().join("model.rgscp");
    let same_stem_path = temp.path().join("model.backup");
    let stale_temp = checkpoint_path.with_extension("rgscp.tmp");
    fs::write(&stale_temp, b"leave this file alone").unwrap();

    save_training_checkpoint(&checkpoint_path, &checkpoint_fixture(10)).unwrap();
    save_training_checkpoint(&same_stem_path, &checkpoint_fixture(20)).unwrap();

    assert_eq!(fs::read(&stale_temp).unwrap(), b"leave this file alone");
    assert_eq!(
        load_training_checkpoint(&checkpoint_path)
            .unwrap()
            .completed_iterations,
        10
    );
    assert_eq!(
        load_training_checkpoint(&same_stem_path)
            .unwrap()
            .completed_iterations,
        20
    );
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
fn checkpoint_load_rejects_wrong_magic_and_format_version() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("wrong-envelope.rgscp");
    let mut bytes = encode_unchecked(&checkpoint_fixture(10));
    bytes[0] ^= 0xff;
    fs::write(&path, &bytes).unwrap();
    assert_invalid_input_contains(load_training_checkpoint(&path).unwrap_err(), "magic");

    let mut bytes = encode_unchecked(&checkpoint_fixture(10));
    let offset = TRAINING_CHECKPOINT_MAGIC.len();
    bytes[offset..offset + 4]
        .copy_from_slice(&(TRAINING_CHECKPOINT_FORMAT_VERSION + 1).to_le_bytes());
    fs::write(&path, &bytes).unwrap();
    assert_invalid_input_contains(
        load_training_checkpoint(&path).unwrap_err(),
        "format version",
    );
}

#[test]
fn checkpoint_load_rejects_trailing_bytes() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("trailing.rgscp");
    save_training_checkpoint(&path, &checkpoint_fixture(10)).unwrap();
    let mut file = fs::OpenOptions::new().append(true).open(&path).unwrap();
    file.write_all(b"trailing").unwrap();

    assert_invalid_input_contains(load_training_checkpoint(&path).unwrap_err(), "trailing");
}

#[test]
fn checkpoint_load_rejects_oversized_file_before_decode() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("oversized.rgscp");
    let file = fs::File::create(&path).unwrap();
    file.set_len(MAX_TRAINING_CHECKPOINT_BYTES + 1).unwrap();

    assert_invalid_input_contains(
        load_training_checkpoint(&path).unwrap_err(),
        "exceeds maximum size",
    );
}

#[test]
fn checkpoint_load_rejects_declared_oversized_vector_without_panicking() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("declared-oversized-vector.rgscp");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&TRAINING_CHECKPOINT_MAGIC);
    bytes.extend_from_slice(&TRAINING_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&TRAINING_CHECKPOINT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&u64::MAX.to_le_bytes());
    fs::write(&path, bytes).unwrap();

    let loaded = catch_unwind(|| load_training_checkpoint(&path))
        .expect("oversized declared vectors must not panic or abort");
    assert_invalid_input_contains(loaded.unwrap_err(), "decode checkpoint");
}

#[test]
fn checkpoint_load_rejects_declared_oversized_float_vec_without_panicking() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("declared-oversized-float-vec.rgscp");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&TRAINING_CHECKPOINT_MAGIC);
    bytes.extend_from_slice(&TRAINING_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&TRAINING_CHECKPOINT_VERSION.to_le_bytes());
    for identity in [b"dataset".as_slice(), b"reconstruction", b"config"] {
        bytes.extend_from_slice(&(identity.len() as u64).to_le_bytes());
        bytes.extend_from_slice(identity);
    }
    bytes.extend_from_slice(&10u64.to_le_bytes());
    bytes.push(1);
    bytes.extend_from_slice(&0.125f32.to_le_bytes());
    bytes.extend_from_slice(&u64::MAX.to_le_bytes());
    fs::write(&path, bytes).unwrap();

    let loaded = catch_unwind(|| load_training_checkpoint(&path))
        .expect("oversized declared float vectors must not panic or abort");
    assert_invalid_input_contains(loaded.unwrap_err(), "decode checkpoint");
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
fn checkpoint_load_rejects_non_finite_splat_components() {
    let temp = tempfile::tempdir().unwrap();
    let cases: [(&str, SplatMutation); 5] = [
        ("positions", |splats| splats.positions[0] = f32::NAN),
        ("log_scales", |splats| splats.log_scales[0] = f32::INFINITY),
        ("rotations", |splats| {
            splats.rotations[0] = f32::NEG_INFINITY
        }),
        ("opacity_logits", |splats| {
            splats.opacity_logits[0] = f32::NAN
        }),
        ("sh_coeffs", |splats| splats.sh_coeffs[0] = f32::INFINITY),
    ];

    for (field, mutate) in cases {
        let path = temp.path().join(format!("non-finite-{field}.rgscp"));
        fs::write(
            &path,
            serialize_with_splat_mutation(checkpoint_fixture(10), mutate),
        )
        .unwrap();
        assert_invalid_input_contains(
            load_training_checkpoint(&path).unwrap_err(),
            &format!("{field} values must be finite"),
        );
    }
}

#[test]
fn checkpoint_load_rejects_sh_width_overflow_without_panicking() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("sh-width-overflow.rgscp");
    fs::write(
        &path,
        serialize_with_sh_degree(checkpoint_fixture(10), usize::MAX),
    )
    .unwrap();

    let loaded = catch_unwind(|| load_training_checkpoint(&path))
        .expect("loading a corrupt checkpoint must not panic");
    assert_invalid_input_contains(loaded.unwrap_err(), "SH width overflow");
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
fn checkpoint_validation_rejects_unpaired_adam_moments() {
    let mut checkpoint = checkpoint_fixture(10);
    checkpoint.optimizer.transforms.moment2 = None;

    assert_invalid_input_contains(
        checkpoint.validate().unwrap_err(),
        "moment1 and moment2 must both be present or both be absent",
    );
}

#[test]
fn checkpoint_validation_rejects_adam_step_mismatch() {
    let mut checkpoint = checkpoint_fixture(10);
    checkpoint.optimizer.sh_coeffs.step = 9;

    assert_invalid_input_contains(
        checkpoint.validate().unwrap_err(),
        "optimizer.sh_coeffs.step must equal completed iterations",
    );
}

#[test]
fn checkpoint_validation_rejects_inexact_adam_shapes() {
    let mut checkpoint = checkpoint_fixture(10);
    checkpoint
        .optimizer
        .transforms
        .moment1
        .as_mut()
        .unwrap()
        .shape = vec![10];
    assert_invalid_input_contains(checkpoint.validate().unwrap_err(), "expected shape [1, 10]");

    let mut checkpoint = checkpoint_fixture(10);
    checkpoint
        .optimizer
        .sh_coeffs
        .scaling
        .as_mut()
        .unwrap()
        .shape = vec![1];
    assert_invalid_input_contains(
        checkpoint.validate().unwrap_err(),
        "expected shape [1, 1, 1]",
    );

    let mut checkpoint = checkpoint_fixture(10);
    checkpoint
        .optimizer
        .raw_opacities
        .moment1
        .as_mut()
        .unwrap()
        .shape = vec![1, 1];
    assert_invalid_input_contains(checkpoint.validate().unwrap_err(), "expected shape [1]");
}

#[test]
fn checkpoint_validation_enforces_identity_rank_and_tensor_element_limits() {
    let mut checkpoint = checkpoint_fixture(10);
    checkpoint.identity.dataset = "d".repeat(MAX_TRAINING_IDENTITY_BYTES + 1);
    assert_invalid_input_contains(
        checkpoint.validate().unwrap_err(),
        "identity dataset exceeds",
    );

    let mut checkpoint = checkpoint_fixture(10);
    let moment = checkpoint.optimizer.transforms.moment1.as_mut().unwrap();
    moment.shape = vec![1; MAX_TRAINING_CHECKPOINT_TENSOR_RANK + 1];
    moment.values = vec![0.1];
    assert_invalid_input_contains(checkpoint.validate().unwrap_err(), "tensor rank");

    let mut checkpoint = checkpoint_fixture(10);
    let moment = checkpoint.optimizer.transforms.moment1.as_mut().unwrap();
    moment.shape = vec![MAX_TRAINING_CHECKPOINT_TENSOR_ELEMENTS + 1];
    moment.values.clear();
    assert_invalid_input_contains(checkpoint.validate().unwrap_err(), "tensor element count");
}

#[test]
fn checkpoint_load_enforces_splat_count_limit() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("too-many-splats.rgscp");
    let bytes = serialize_with_splat_mutation(checkpoint_fixture(10), |splats| {
        splats.opacity_logits = vec![0.0; MAX_TRAINING_CHECKPOINT_SPLATS + 1];
    });
    fs::write(&path, bytes).unwrap();

    assert_invalid_input_contains(
        load_training_checkpoint(&path).unwrap_err(),
        "splat count exceeds",
    );
}

#[test]
fn checkpoint_validation_rejects_invalid_active_sh_degree() {
    let mut checkpoint = checkpoint_fixture(10);
    checkpoint.active_sh_degree = checkpoint.splats.sh_degree() + 1;

    assert_invalid_input_contains(checkpoint.validate().unwrap_err(), "active SH degree");
}

#[test]
fn checkpoint_validation_rejects_future_topology_iterations() {
    let mut checkpoint = checkpoint_fixture(10);
    checkpoint.topology.splat_birth_iterations[0] = 11;
    assert_invalid_input_contains(
        checkpoint.validate().unwrap_err(),
        "splat_birth_iterations contains future iteration",
    );

    let mut checkpoint = checkpoint_fixture(10);
    checkpoint.topology.splat_invisible_windows[0] = 11;
    assert_invalid_input_contains(
        checkpoint.validate().unwrap_err(),
        "splat_invisible_windows exceeds completed iterations",
    );
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

    let first =
        TrainingIdentity::from_canonical_content(&dataset, b"reconstruction-a", &config).unwrap();
    let second = TrainingIdentity::from_canonical_content(
        &dataset,
        b"reconstruction-a",
        &TrainingConfig {
            iterations: 20,
            ..config.clone()
        },
    )
    .unwrap();
    assert_eq!(first, second);
    assert_eq!(first.dataset.len(), 64);
    assert_eq!(first.config.len(), 64);
    assert_eq!(first.reconstruction.len(), 64);

    let mut changed_dataset = dataset.clone();
    changed_dataset.add_point([1.0, 0.0, 2.0], None);
    let changed_dataset_identity =
        TrainingIdentity::from_canonical_content(&changed_dataset, b"reconstruction-a", &config)
            .unwrap();
    assert_ne!(changed_dataset_identity.dataset, first.dataset);

    let mut changed_config = config.clone();
    changed_config.data.frame_shuffle_seed = 8;
    let changed_config_identity =
        TrainingIdentity::from_canonical_content(&dataset, b"reconstruction-a", &changed_config)
            .unwrap();
    assert_ne!(changed_config_identity.config, first.config);
}

#[test]
fn checkpoint_identity_hashes_reconstruction_file_content_not_path() {
    let temp = tempfile::tempdir().unwrap();
    let first_path = temp.path().join("first-reconstruction.bin");
    let second_path = temp.path().join("second-reconstruction.bin");
    fs::write(&first_path, b"same reconstruction").unwrap();
    fs::write(&second_path, b"same reconstruction").unwrap();
    let dataset = TrainingDataset::new(Intrinsics::new(12.0, 12.0, 8.0, 8.0, 16, 16));
    let config = TrainingConfig::default();

    let first = TrainingIdentity::from_inputs(&dataset, &first_path, &config).unwrap();
    let relocated = TrainingIdentity::from_inputs(&dataset, &second_path, &config).unwrap();
    assert_eq!(first.reconstruction, relocated.reconstruction);

    fs::write(&first_path, b"changed reconstruction").unwrap();
    let changed = TrainingIdentity::from_inputs(&dataset, &first_path, &config).unwrap();
    assert_ne!(first.reconstruction, changed.reconstruction);
}

#[test]
fn checkpoint_identity_hashes_reconstruction_directory_canonically() {
    let temp = tempfile::tempdir().unwrap();
    let first = temp.path().join("first");
    let relocated = temp.path().join("relocated");
    fs::create_dir_all(first.join("nested")).unwrap();
    fs::create_dir_all(relocated.join("nested")).unwrap();
    fs::write(first.join("cameras.bin"), b"cameras").unwrap();
    fs::write(first.join("nested/images.bin"), b"images").unwrap();
    fs::write(relocated.join("nested/images.bin"), b"images").unwrap();
    fs::write(relocated.join("cameras.bin"), b"cameras").unwrap();
    let dataset = TrainingDataset::new(Intrinsics::new(12.0, 12.0, 8.0, 8.0, 16, 16));
    let config = TrainingConfig::default();

    let original = TrainingIdentity::from_inputs(&dataset, &first, &config).unwrap();
    let copied = TrainingIdentity::from_inputs(&dataset, &relocated, &config).unwrap();
    assert_eq!(original.reconstruction, copied.reconstruction);

    fs::write(first.join("nested/images.bin"), b"changed images").unwrap();
    let changed = TrainingIdentity::from_inputs(&dataset, &first, &config).unwrap();
    assert_ne!(original.reconstruction, changed.reconstruction);
}

#[test]
fn checkpoint_identity_hashes_dataset_file_content_not_path() {
    let temp = tempfile::tempdir().unwrap();
    let reconstruction = temp.path().join("reconstruction.bin");
    let first_image = temp.path().join("first.rgb");
    let second_image = temp.path().join("second.rgb");
    fs::write(&reconstruction, b"reconstruction").unwrap();
    fs::write(&first_image, b"same image").unwrap();
    fs::write(&second_image, b"same image").unwrap();
    let config = TrainingConfig::default();

    let mut first_dataset = TrainingDataset::new(Intrinsics::new(12.0, 12.0, 8.0, 8.0, 16, 16));
    first_dataset.add_pose(ScenePose::new(0, first_image, SE3::identity(), 0.0));
    let mut relocated_dataset = TrainingDataset::new(Intrinsics::new(12.0, 12.0, 8.0, 8.0, 16, 16));
    relocated_dataset.add_pose(ScenePose::new(
        0,
        second_image.clone(),
        SE3::identity(),
        0.0,
    ));

    let first = TrainingIdentity::from_inputs(&first_dataset, &reconstruction, &config).unwrap();
    let relocated =
        TrainingIdentity::from_inputs(&relocated_dataset, &reconstruction, &config).unwrap();
    assert_eq!(first.dataset, relocated.dataset);

    fs::write(second_image, b"changed image").unwrap();
    let changed =
        TrainingIdentity::from_inputs(&relocated_dataset, &reconstruction, &config).unwrap();
    assert_ne!(first.dataset, changed.dataset);
}

#[test]
#[allow(deprecated)]
fn legacy_json_checkpoint_has_an_explicit_migration_api() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("legacy-checkpoint.json");
    let splats = checkpoint_fixture(10).splats;
    let legacy = rustgs::LegacyTrainingCheckpoint {
        iteration: 10,
        loss: 0.125,
        splats: splats.clone(),
    };
    legacy.save(&path).unwrap();

    let loaded = rustgs::load_legacy_training_checkpoint(&path).unwrap();
    assert_eq!(loaded.iteration, 10);
    assert_eq!(loaded.loss, 0.125);
    assert_eq!(loaded.splats, splats);
    let io_alias: rustgs::io::TrainingCheckpoint = loaded.clone();
    let module_alias: rustgs::legacy::TrainingCheckpoint = io_alias;
    assert_eq!(module_alias.into_splats(), splats);

    assert_invalid_input_contains(load_training_checkpoint(&path).unwrap_err(), "magic");
}
