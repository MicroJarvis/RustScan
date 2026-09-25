use std::cell::RefCell;
use std::fs;
use std::io::Write;
use std::panic::catch_unwind;
use std::rc::Rc;
use std::sync::{Arc, Barrier};

use bincode::Options;
use rustscan_gs::{
    fingerprint_frame_selection, load_training_checkpoint, load_training_checkpoint_with_migration,
    resolve_checkpoint_selection, save_training_checkpoint, train_splats, AdamCheckpoint,
    AdamParameterCheckpoint, CheckpointFrameSelectionMeta, CheckpointMigration,
    FrameSelectionRequest, HostSplats, Intrinsics, ScenePose, SelectionMetaProvenance,
    TensorCheckpoint, TopologyCheckpoint, TrainingCheckpoint, TrainingCheckpointPolicy,
    TrainingCheckpointReason, TrainingConfig, TrainingControl, TrainingDataset, TrainingError,
    TrainingEvent, TrainingEventCadence, TrainingIdentity, TrainingOptions, TrainingRunDisposition,
    MAX_TRAINING_CHECKPOINT_BYTES, MAX_TRAINING_CHECKPOINT_SPLATS,
    MAX_TRAINING_CHECKPOINT_TENSOR_ELEMENTS, MAX_TRAINING_CHECKPOINT_TENSOR_RANK,
    MAX_TRAINING_IDENTITY_BYTES, MAX_TRAINING_ITERATIONS, SE3, TRAINING_CHECKPOINT_FORMAT_VERSION,
    TRAINING_CHECKPOINT_MAGIC, TRAINING_CHECKPOINT_VERSION, TRAINING_CHECKPOINT_VERSION_V1,
    TRAINING_CHECKPOINT_VERSION_V2, TRAINING_CHECKPOINT_VERSION_V3,
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
    selection: Option<rustscan_gs::CheckpointFrameSelectionMeta>,
}

/// Pre-v3 on-disk layout used to prove migration leaves selection absent.
#[derive(Serialize)]
struct SerializedTrainingCheckpointV2 {
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

/// Pre-v4 on-disk layout: selection without request fields.
#[derive(Serialize)]
struct SerializedTrainingCheckpointV3 {
    version: u32,
    identity: TrainingIdentity,
    completed_iterations: usize,
    latest_loss: Option<f32>,
    splats: SerializedHostSplats,
    optimizer: AdamCheckpoint,
    topology: TopologyCheckpoint,
    frame_shuffle_seed: u64,
    active_sh_degree: usize,
    selection: Option<SerializedSelectionMetaV3>,
}

#[derive(Serialize)]
struct SerializedSelectionMetaV3 {
    eval_split_kind: Option<String>,
    train_stable_ids: Vec<u64>,
    eval_stable_ids: Vec<u64>,
    train_loader_frame_ids: Vec<u64>,
    manifest_fingerprint: Option<String>,
    selection_fingerprint: Option<String>,
    eval_selection_fingerprint: Option<String>,
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
            visibility_window_baseline: vec![0.0],
            actual_visibility_window_baseline: vec![0.0],
        },
        frame_shuffle_seed: 7,
        active_sh_degree: 0,
        selection: None,
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
        selection: checkpoint.selection,
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

fn invalid_input_message(error: TrainingError) -> String {
    match error {
        TrainingError::InvalidInput(message) => message,
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
fn checkpoint_identity_wire_round_trips_utf8() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("identity-utf8.rgscp");
    let mut checkpoint = checkpoint_fixture(10);
    checkpoint.identity = TrainingIdentity {
        dataset: "dataset-with-spaces".to_string(),
        reconstruction: "reconstruction/path".to_string(),
        config: "config-\u{2603}".to_string(),
    };

    save_training_checkpoint(&path, &checkpoint).unwrap();

    assert_eq!(load_training_checkpoint(&path).unwrap(), checkpoint);
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
fn checkpoint_load_rejects_declared_oversized_identity_without_allocating_payload() {
    const DECLARED_IDENTITY_BYTES: u64 = 1024 * 1024 * 1024;

    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("declared-oversized-identity.rgscp");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&TRAINING_CHECKPOINT_MAGIC);
    bytes.extend_from_slice(&TRAINING_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&TRAINING_CHECKPOINT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&DECLARED_IDENTITY_BYTES.to_le_bytes());
    fs::write(&path, bytes).unwrap();

    let loaded = catch_unwind(|| load_training_checkpoint(&path))
        .expect("oversized declared identity must not panic or abort");
    assert_invalid_input_contains(loaded.unwrap_err(), "identity field exceeds maximum length");
}

#[test]
fn checkpoint_load_rejects_non_utf8_identity_bytes() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("non-utf8-identity.rgscp");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&TRAINING_CHECKPOINT_MAGIC);
    bytes.extend_from_slice(&TRAINING_CHECKPOINT_FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&TRAINING_CHECKPOINT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.push(0xff);
    fs::write(&path, bytes).unwrap();

    assert_invalid_input_contains(
        load_training_checkpoint(&path).unwrap_err(),
        "identity field must be valid UTF-8",
    );
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
fn checkpoint_v2_roundtrip_preserves_visibility_baselines() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("v2-baselines.rgscp");
    let mut checkpoint = checkpoint_fixture(10);
    checkpoint.topology.visibility_window_baseline = vec![4.0];
    checkpoint.topology.actual_visibility_window_baseline = vec![5.0];
    save_training_checkpoint(&path, &checkpoint).unwrap();
    let loaded = load_training_checkpoint(&path).unwrap();
    assert_eq!(loaded.version, TRAINING_CHECKPOINT_VERSION);
    assert_eq!(loaded.topology.visibility_window_baseline, vec![4.0]);
    assert_eq!(loaded.topology.actual_visibility_window_baseline, vec![5.0]);
}

#[test]
fn checkpoint_v2_rejects_baseline_length_and_nan() {
    let mut checkpoint = checkpoint_fixture(10);
    checkpoint.topology.visibility_window_baseline = vec![1.0, 2.0];
    assert_invalid_input_contains(
        checkpoint.validate().unwrap_err(),
        "visibility_window_baseline length",
    );

    let mut checkpoint = checkpoint_fixture(10);
    checkpoint.topology.actual_visibility_window_baseline = vec![f32::NAN];
    assert_invalid_input_contains(
        checkpoint.validate().unwrap_err(),
        "actual_visibility_window_baseline[0] must be finite",
    );

    let mut checkpoint = checkpoint_fixture(10);
    checkpoint.topology.visibility_window_baseline = vec![-1.0];
    assert_invalid_input_contains(
        checkpoint.validate().unwrap_err(),
        "visibility_window_baseline[0] must be non-negative",
    );
}

#[derive(Serialize)]
struct TopologyCheckpointV1Wire {
    grad_2d: TensorCheckpoint,
    screen_grad_2d: TensorCheckpoint,
    abs_grad_2d: TensorCheckpoint,
    abs_pixel_grad_2d: TensorCheckpoint,
    pixel_coverage: TensorCheckpoint,
    camera_depth: TensorCheckpoint,
    grad_color: TensorCheckpoint,
    num_observations: TensorCheckpoint,
    visible_observations: TensorCheckpoint,
    actual_visible_observations: TensorCheckpoint,
    splat_birth_iterations: Vec<usize>,
    splat_invisible_windows: Vec<usize>,
}

#[derive(Serialize)]
struct TrainingCheckpointV1Wire {
    version: u32,
    identity: TrainingIdentity,
    completed_iterations: usize,
    latest_loss: Option<f32>,
    splats: HostSplats,
    optimizer: AdamCheckpoint,
    topology: TopologyCheckpointV1Wire,
    frame_shuffle_seed: u64,
    active_sh_degree: usize,
}

#[test]
fn checkpoint_v1_fixture_migrates_baselines_from_cumulative() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("v1-migrate.rgscp");
    let v2 = checkpoint_fixture(10);
    let v1 = TrainingCheckpointV1Wire {
        version: TRAINING_CHECKPOINT_VERSION_V1,
        identity: v2.identity.clone(),
        completed_iterations: v2.completed_iterations,
        latest_loss: v2.latest_loss,
        splats: v2.splats.clone(),
        optimizer: v2.optimizer.clone(),
        topology: TopologyCheckpointV1Wire {
            grad_2d: v2.topology.grad_2d.clone(),
            screen_grad_2d: v2.topology.screen_grad_2d.clone(),
            abs_grad_2d: v2.topology.abs_grad_2d.clone(),
            abs_pixel_grad_2d: v2.topology.abs_pixel_grad_2d.clone(),
            pixel_coverage: v2.topology.pixel_coverage.clone(),
            camera_depth: v2.topology.camera_depth.clone(),
            grad_color: v2.topology.grad_color.clone(),
            num_observations: v2.topology.num_observations.clone(),
            visible_observations: TensorCheckpoint {
                shape: vec![1],
                values: vec![7.0],
            },
            actual_visible_observations: TensorCheckpoint {
                shape: vec![1],
                values: vec![9.0],
            },
            splat_birth_iterations: v2.topology.splat_birth_iterations.clone(),
            splat_invisible_windows: v2.topology.splat_invisible_windows.clone(),
        },
        frame_shuffle_seed: v2.frame_shuffle_seed,
        active_sh_degree: v2.active_sh_degree,
    };
    fs::write(&path, encode_unchecked(&v1)).unwrap();

    let (loaded, migration) = load_training_checkpoint_with_migration(&path).unwrap();
    assert_eq!(migration, CheckpointMigration::V1BaselineReset);
    assert_eq!(loaded.version, TRAINING_CHECKPOINT_VERSION);
    assert_eq!(loaded.topology.visibility_window_baseline, vec![7.0]);
    assert_eq!(loaded.topology.actual_visibility_window_baseline, vec![9.0]);
    assert_eq!(loaded.topology.visible_observations.values, vec![7.0]);
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
fn checkpoint_load_rejects_unsupported_sh_degree_before_width_calculation() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("sh-width-overflow.rgscp");
    fs::write(
        &path,
        serialize_with_sh_degree(checkpoint_fixture(10), usize::MAX),
    )
    .unwrap();

    let loaded = catch_unwind(|| load_training_checkpoint(&path))
        .expect("loading a corrupt checkpoint must not panic");
    assert_invalid_input_contains(
        loaded.unwrap_err(),
        "stored SH degree exceeds supported maximum 3",
    );
}

#[test]
fn checkpoint_load_rejects_unsupported_stored_sh_degree_for_empty_splats() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("unsupported-empty-sh-degree.rgscp");
    let bytes = serialize_with_splat_mutation(checkpoint_fixture(10), |splats| {
        splats.positions.clear();
        splats.log_scales.clear();
        splats.rotations.clear();
        splats.opacity_logits.clear();
        splats.sh_coeffs.clear();
        splats.sh_degree = usize::MAX;
    });
    fs::write(&path, bytes).unwrap();

    let loaded = catch_unwind(|| load_training_checkpoint(&path))
        .expect("unsupported stored SH degree must not panic or abort");
    assert_invalid_input_contains(
        loaded.unwrap_err(),
        "stored SH degree exceeds supported maximum 3",
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
fn checkpoint_validation_rejects_unpaired_adam_moments() {
    let mut checkpoint = checkpoint_fixture(10);
    checkpoint.optimizer.transforms.moment2 = None;

    assert_invalid_input_contains(
        checkpoint.validate().unwrap_err(),
        "moment1 and moment2 must both be present or both be absent",
    );
}

#[test]
fn checkpoint_validation_accepts_reset_adam_state_with_scaling() {
    let mut checkpoint = checkpoint_fixture(10);
    for parameter in [
        &mut checkpoint.optimizer.transforms,
        &mut checkpoint.optimizer.sh_coeffs,
        &mut checkpoint.optimizer.raw_opacities,
    ] {
        parameter.step = 0;
        parameter.moment1 = None;
        parameter.moment2 = None;
        assert!(parameter.scaling.is_some());
    }

    checkpoint.validate().unwrap();
}

#[test]
fn checkpoint_validation_rejects_adam_step_after_completed_iterations() {
    let mut checkpoint = checkpoint_fixture(10);
    checkpoint.optimizer.sh_coeffs.step = 11;

    assert_invalid_input_contains(
        checkpoint.validate().unwrap_err(),
        "optimizer.sh_coeffs.step must not exceed completed iterations",
    );
}

#[test]
fn checkpoint_validation_rejects_step_beyond_i32_boundary() {
    let checkpoint = checkpoint_fixture(MAX_TRAINING_ITERATIONS + 1);

    assert_invalid_input_contains(checkpoint.validate().unwrap_err(), "maximum safe step");
}

#[test]
fn checkpoint_validation_rejects_adam_moments_at_step_zero() {
    let mut checkpoint = checkpoint_fixture(10);
    checkpoint.optimizer.transforms.step = 0;

    assert_invalid_input_contains(
        checkpoint.validate().unwrap_err(),
        "optimizer.transforms moments must be absent when step is zero",
    );
}

#[test]
fn checkpoint_validation_rejects_missing_adam_moments_after_step_zero() {
    let mut checkpoint = checkpoint_fixture(10);
    checkpoint.optimizer.raw_opacities.moment1 = None;
    checkpoint.optimizer.raw_opacities.moment2 = None;

    assert_invalid_input_contains(
        checkpoint.validate().unwrap_err(),
        "optimizer.raw_opacities moments must be present when step is non-zero",
    );
}

#[test]
fn checkpoint_validation_rejects_divergent_adam_steps() {
    let mut checkpoint = checkpoint_fixture(10);
    checkpoint.optimizer.transforms.step = 0;
    checkpoint.optimizer.transforms.moment1 = None;
    checkpoint.optimizer.transforms.moment2 = None;
    checkpoint.optimizer.sh_coeffs.step = 7;
    checkpoint.optimizer.raw_opacities.step = 3;

    assert_invalid_input_contains(
        checkpoint.validate().unwrap_err(),
        "optimizer parameter steps must be equal",
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
    let legacy = rustscan_gs::LegacyTrainingCheckpoint {
        iteration: 10,
        loss: 0.125,
        splats: splats.clone(),
    };
    legacy.save(&path).unwrap();

    let loaded = rustscan_gs::load_legacy_training_checkpoint(&path).unwrap();
    assert_eq!(loaded.iteration, 10);
    assert_eq!(loaded.loss, 0.125);
    assert_eq!(loaded.splats, splats);
    let io_alias: rustscan_gs::io::TrainingCheckpoint = loaded.clone();
    let module_alias: rustscan_gs::legacy::TrainingCheckpoint = io_alias;
    assert_eq!(module_alias.into_splats(), splats);

    assert_invalid_input_contains(load_training_checkpoint(&path).unwrap_err(), "magic");
}

fn tiny_training_dataset(
    temp: &tempfile::TempDir,
    stem: &str,
    frame_count: usize,
) -> TrainingDataset {
    let image_path = temp.path().join(format!("{stem}.rgb"));
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
    fs::write(&image_path, pixels).unwrap();

    let mut dataset = TrainingDataset::new(Intrinsics::new(12.0, 12.0, 8.0, 8.0, 16, 16));
    for frame_idx in 0..frame_count {
        dataset.add_pose(ScenePose::new(
            frame_idx as u64,
            image_path.clone(),
            SE3::identity(),
            frame_idx as f64,
        ));
    }
    dataset.add_point([0.0, 0.0, 2.0], Some([0.25, 0.5, 0.75]));
    dataset.add_point([0.25, -0.2, 2.5], Some([0.75, 0.25, 0.5]));
    dataset
}

fn verified_selection_meta(
    train_ids: &[u64],
    eval_ids: &[u64],
    loader_ids: &[u64],
    eval_split_kind: Option<&str>,
) -> CheckpointFrameSelectionMeta {
    let train_request = FrameSelectionRequest {
        max_frames: 0,
        frame_stride: 1,
        allowed_ids: Some(train_ids.to_vec()),
        ..Default::default()
    };
    let eval_request = FrameSelectionRequest {
        max_frames: 0,
        frame_stride: 1,
        allowed_ids: if eval_ids.is_empty() {
            None
        } else {
            Some(eval_ids.to_vec())
        },
        ..Default::default()
    };
    CheckpointFrameSelectionMeta {
        eval_split_kind: eval_split_kind.map(str::to_string),
        train_stable_ids: train_ids.to_vec(),
        eval_stable_ids: eval_ids.to_vec(),
        train_loader_frame_ids: loader_ids.to_vec(),
        manifest_fingerprint: None,
        selection_fingerprint: Some(fingerprint_frame_selection(train_ids, &train_request)),
        eval_selection_fingerprint: if eval_ids.is_empty() {
            None
        } else {
            Some(fingerprint_frame_selection(eval_ids, &eval_request))
        },
        max_frames: 0,
        frame_stride: 1,
        include_ranges: Vec::new(),
        exclude_ranges: Vec::new(),
        allowed_ids: Some(train_ids.to_vec()),
        eval_max_frames: 0,
        eval_frame_stride: 1,
        eval_include_ranges: Vec::new(),
        eval_exclude_ranges: Vec::new(),
        eval_allowed_ids: if eval_ids.is_empty() {
            None
        } else {
            Some(eval_ids.to_vec())
        },
        provenance: rustscan_gs::SelectionMetaProvenance::Verified,
    }
}

fn tiny_training_config(iterations: usize) -> TrainingConfig {
    TrainingConfig {
        iterations,
        raster: rustscan_gs::TrainingRasterConfig {
            render_scale: 1.0,
            ..Default::default()
        },
        initialization: rustscan_gs::TrainingInitializationConfig {
            max_initial_gaussians: 2,
            ..Default::default()
        },
        data: rustscan_gs::TrainingDataConfig {
            frame_cache_capacity: 2,
            frame_prefetch_ahead: 1,
            frame_shuffle_seed: 0,
        },
        ..Default::default()
    }
}

fn continuity_training_dataset(temp: &tempfile::TempDir) -> TrainingDataset {
    let mut dataset = TrainingDataset::new(Intrinsics::new(12.0, 12.0, 8.0, 8.0, 16, 16));
    for frame_idx in 0..3_u8 {
        let image_path = temp
            .path()
            .join(format!("continuity-frame-{frame_idx}.rgb"));
        let mut pixels = Vec::with_capacity(16 * 16 * 3);
        for y in 0..16_u8 {
            for x in 0..16_u8 {
                pixels.extend_from_slice(&[
                    x.saturating_mul(10).saturating_add(frame_idx * 19),
                    y.saturating_mul(10).saturating_add(frame_idx * 23),
                    x.saturating_add(y)
                        .saturating_mul(5)
                        .saturating_add(frame_idx * 29),
                ]);
            }
        }
        fs::write(&image_path, pixels).unwrap();
        dataset.add_pose(ScenePose::new(
            frame_idx as u64,
            image_path,
            SE3::new(&[0.0, 0.0, 0.0, 1.0], &[frame_idx as f32 * 0.05, 0.0, 0.0]),
            frame_idx as f64,
        ));
    }
    dataset.add_point([0.0, 0.0, 2.0], Some([0.25, 0.5, 0.75]));
    dataset.add_point([0.25, -0.2, 2.5], Some([0.75, 0.25, 0.5]));
    dataset
}

#[test]
fn pause_checkpoint_sink_failure_fails_run_without_paused_or_completed_events() {
    let temp = tempfile::tempdir().unwrap();
    let dataset = tiny_training_dataset(&temp, "sink-failure-frame", 1);
    let config = tiny_training_config(3);
    let identity =
        TrainingIdentity::from_canonical_content(&dataset, b"sink-failure", &config).unwrap();
    let control = TrainingControl::new(TrainingEventCadence::default());
    control.request_pause();
    let events = Rc::new(RefCell::new(Vec::new()));
    let captured_events = Rc::clone(&events);

    let error = train_splats(
        &dataset,
        &config,
        TrainingOptions::new()
            .with_control(control)
            .with_identity(identity)
            .with_checkpoint_sink(|_ready| {
                Err(TrainingError::TrainingFailed(
                    "checkpoint commit failed".to_string(),
                ))
            })
            .with_event_sink(move |event| captured_events.borrow_mut().push(event)),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        TrainingError::TrainingFailed(ref message) if message == "checkpoint commit failed"
    ));
    assert!(matches!(
        events.borrow().last(),
        Some(TrainingEvent::RunFailed(_))
    ));
    assert!(!events.borrow().iter().any(|event| matches!(
        event,
        TrainingEvent::CheckpointReady(_)
            | TrainingEvent::RunPaused(_)
            | TrainingEvent::RunCompleted(_)
    )));
}

#[test]
fn cancel_requested_by_checkpoint_sink_commits_then_finishes_cancelled() {
    let temp = tempfile::tempdir().unwrap();
    let dataset = tiny_training_dataset(&temp, "sink-cancel-frame", 1);
    let config = tiny_training_config(3);
    let identity =
        TrainingIdentity::from_canonical_content(&dataset, b"sink-cancel", &config).unwrap();
    let control = TrainingControl::new(TrainingEventCadence::default());
    control.request_pause();
    let sink_control = control.clone();
    let sequence = Rc::new(RefCell::new(Vec::new()));
    let sink_sequence = Rc::clone(&sequence);
    let event_sequence = Rc::clone(&sequence);

    let run = train_splats(
        &dataset,
        &config,
        TrainingOptions::new()
            .with_control(control)
            .with_identity(identity)
            .with_checkpoint_sink(move |ready| {
                sink_sequence
                    .borrow_mut()
                    .push(format!("sink:{}", ready.iteration));
                sink_control.request_cancel();
                Ok(())
            })
            .with_event_sink(move |event| match event {
                TrainingEvent::CheckpointReady(ready) => event_sequence
                    .borrow_mut()
                    .push(format!("checkpoint:{}", ready.iteration)),
                TrainingEvent::RunPaused(_) => {
                    event_sequence.borrow_mut().push("paused".to_string())
                }
                TrainingEvent::RunCancelled(cancelled) => event_sequence
                    .borrow_mut()
                    .push(format!("cancelled:{}", cancelled.completed_iterations)),
                TrainingEvent::RunCompleted(completed) => event_sequence
                    .borrow_mut()
                    .push(format!("completed:{:?}", completed.report.disposition)),
                _ => {}
            }),
    )
    .unwrap();

    assert_eq!(run.report.completed_iterations, 1);
    assert!(run.report.cancelled);
    assert_eq!(run.report.disposition, TrainingRunDisposition::Cancelled);
    assert_eq!(
        sequence.borrow().as_slice(),
        [
            "sink:1",
            "checkpoint:1",
            "cancelled:1",
            "completed:Cancelled"
        ]
    );
}

#[test]
fn cancel_requested_by_checkpoint_event_finishes_cancelled_after_commit() {
    let temp = tempfile::tempdir().unwrap();
    let dataset = tiny_training_dataset(&temp, "event-cancel-frame", 1);
    let config = tiny_training_config(3);
    let identity =
        TrainingIdentity::from_canonical_content(&dataset, b"event-cancel", &config).unwrap();
    let control = TrainingControl::new(TrainingEventCadence::default());
    control.request_pause();
    let event_control = control.clone();
    let sequence = Rc::new(RefCell::new(Vec::new()));
    let sink_sequence = Rc::clone(&sequence);
    let event_sequence = Rc::clone(&sequence);

    let run = train_splats(
        &dataset,
        &config,
        TrainingOptions::new()
            .with_control(control)
            .with_identity(identity)
            .with_checkpoint_sink(move |ready| {
                sink_sequence
                    .borrow_mut()
                    .push(format!("sink:{}", ready.iteration));
                Ok(())
            })
            .with_event_sink(move |event| match event {
                TrainingEvent::CheckpointReady(ready) => {
                    event_sequence
                        .borrow_mut()
                        .push(format!("checkpoint:{}", ready.iteration));
                    event_control.request_cancel();
                }
                TrainingEvent::RunPaused(_) => {
                    event_sequence.borrow_mut().push("paused".to_string())
                }
                TrainingEvent::RunCancelled(cancelled) => event_sequence
                    .borrow_mut()
                    .push(format!("cancelled:{}", cancelled.completed_iterations)),
                TrainingEvent::RunCompleted(completed) => event_sequence
                    .borrow_mut()
                    .push(format!("completed:{:?}", completed.report.disposition)),
                _ => {}
            }),
    )
    .unwrap();

    assert_eq!(run.report.completed_iterations, 1);
    assert!(run.report.cancelled);
    assert_eq!(run.report.disposition, TrainingRunDisposition::Cancelled);
    assert_eq!(
        sequence.borrow().as_slice(),
        [
            "sink:1",
            "checkpoint:1",
            "cancelled:1",
            "completed:Cancelled"
        ]
    );
}

#[test]
fn resume_past_loss_cadence_pause_runs_one_more_iteration() {
    let temp = tempfile::tempdir().unwrap();
    let dataset = tiny_training_dataset(&temp, "resume-frame", 3);
    // Pause can only take effect at a declared safety point. With loss-cadence
    // confirmation (interval 20), progress for mid-run iterations arrives in the
    // catch-up batch at iteration 20, so the pause checkpoint commits 20.
    let pause_config = tiny_training_config(20);
    let identity =
        TrainingIdentity::from_canonical_content(&dataset, b"resume-reconstruction", &pause_config)
            .unwrap();
    let control = TrainingControl::new(TrainingEventCadence {
        progress_every: 1,
        snapshot_every: None,
    });
    let captured_checkpoint = Rc::new(RefCell::new(None));
    let sink_checkpoint = Rc::clone(&captured_checkpoint);
    let event_control = control.clone();

    let paused = train_splats(
        &dataset,
        &pause_config,
        TrainingOptions::new()
            .with_control(control)
            .with_identity(identity.clone())
            .with_checkpoint_sink(move |ready| {
                *sink_checkpoint.borrow_mut() = Some(ready.checkpoint.clone());
                Ok(())
            })
            .with_event_sink(move |event| {
                if matches!(
                    event,
                    TrainingEvent::IterationProgress(progress) if progress.iteration == 20
                ) {
                    event_control.request_pause();
                }
            }),
    )
    .unwrap();
    assert_eq!(paused.report.disposition, TrainingRunDisposition::Paused);
    assert_eq!(paused.report.completed_iterations, 20);
    let checkpoint = captured_checkpoint
        .borrow()
        .clone()
        .expect("pause checkpoint");
    assert_eq!(checkpoint.completed_iterations, 20);

    let resume_config = tiny_training_config(21);
    let resume_identity = TrainingIdentity::from_canonical_content(
        &dataset,
        b"resume-reconstruction",
        &resume_config,
    )
    .unwrap();
    let resumed_iterations = Rc::new(RefCell::new(Vec::new()));
    let captured_iterations = Rc::clone(&resumed_iterations);
    let resumed = train_splats(
        &dataset,
        &resume_config,
        TrainingOptions::new()
            .with_identity(resume_identity)
            .with_resume_checkpoint(checkpoint)
            .with_event_sink(move |event| {
                if let TrainingEvent::IterationProgress(progress) = event {
                    captured_iterations.borrow_mut().push(progress.iteration);
                }
            }),
    )
    .unwrap();

    assert_eq!(resumed_iterations.borrow().as_slice(), [21]);
    assert_eq!(resumed.report.completed_iterations, 21);
    assert!(!resumed.report.cancelled);
    assert_eq!(
        resumed.report.disposition,
        TrainingRunDisposition::Completed
    );
    let last_sample = resumed
        .report
        .telemetry
        .as_ref()
        .unwrap()
        .loss_curve_samples
        .last()
        .expect("target iteration is retained in telemetry");
    assert_eq!(last_sample.iteration, 21);
    // zero-based sample index 20 with 3 frames → frame 2
    assert_eq!(last_sample.frame_idx, 2);
}

#[test]
fn train_splats_rejects_resume_identity_errors_before_gpu_training_events() {
    let temp = tempfile::tempdir().unwrap();
    let dataset = tiny_training_dataset(&temp, "identity-validation-frame", 1);
    let config = tiny_training_config(8);
    let current =
        TrainingIdentity::from_canonical_content(&dataset, b"identity-validation", &config)
            .unwrap();
    let cases = [
        (
            Some(current.clone()),
            TrainingIdentity {
                dataset: "other".to_string(),
                ..current.clone()
            },
            "checkpoint dataset does not match the current training dataset",
        ),
        (
            Some(current.clone()),
            TrainingIdentity {
                reconstruction: "other".to_string(),
                ..current.clone()
            },
            "checkpoint reconstruction does not match the current sparse reconstruction",
        ),
        (
            Some(current.clone()),
            TrainingIdentity {
                config: "other".to_string(),
                ..current.clone()
            },
            "checkpoint configuration does not match the current training configuration",
        ),
        (
            None,
            current.clone(),
            "resuming training requires the current training identity",
        ),
    ];

    for (current_identity, checkpoint_identity, expected) in cases {
        let mut checkpoint = checkpoint_fixture(7);
        checkpoint.identity = checkpoint_identity;
        checkpoint.frame_shuffle_seed = config.data.frame_shuffle_seed;
        let events = Rc::new(RefCell::new(Vec::new()));
        let captured_events = Rc::clone(&events);
        let mut options = TrainingOptions::new()
            .with_resume_checkpoint(checkpoint)
            .with_event_sink(move |event| captured_events.borrow_mut().push(event));
        if let Some(current_identity) = current_identity {
            options = options.with_identity(current_identity);
        }

        let error = train_splats(&dataset, &config, options).unwrap_err();
        assert_eq!(invalid_input_message(error), expected);
        assert!(matches!(
            events.borrow().last(),
            Some(TrainingEvent::RunFailed(_))
        ));
        assert!(!events.borrow().iter().any(|event| matches!(
            event,
            TrainingEvent::IterationProgress(_)
                | TrainingEvent::SnapshotReady(_)
                | TrainingEvent::CheckpointReady(_)
                | TrainingEvent::RunPaused(_)
                | TrainingEvent::RunCancelled(_)
                | TrainingEvent::RunCompleted(_)
        )));
    }
}

#[test]
fn periodic_checkpoint_policy_commits_once_at_cadence_before_completed_event() {
    let temp = tempfile::tempdir().unwrap();
    let dataset = tiny_training_dataset(&temp, "periodic-frame", 1);
    let config = tiny_training_config(3);
    let identity =
        TrainingIdentity::from_canonical_content(&dataset, b"periodic-reconstruction", &config)
            .unwrap();
    let sequence = Rc::new(RefCell::new(Vec::new()));
    let sink_sequence = Rc::clone(&sequence);
    let event_sequence = Rc::clone(&sequence);

    let run = train_splats(
        &dataset,
        &config,
        TrainingOptions::new()
            .with_identity(identity)
            .with_checkpoint_policy(TrainingCheckpointPolicy { every: Some(2) })
            .with_checkpoint_sink(move |ready| {
                sink_sequence
                    .borrow_mut()
                    .push(format!("sink:{}:{:?}", ready.iteration, ready.reason));
                Ok(())
            })
            .with_event_sink(move |event| match event {
                TrainingEvent::CheckpointReady(ready) => event_sequence
                    .borrow_mut()
                    .push(format!("event:{}:{:?}", ready.iteration, ready.reason)),
                TrainingEvent::RunCompleted(completed) => event_sequence
                    .borrow_mut()
                    .push(format!("completed:{:?}", completed.report.disposition)),
                _ => {}
            }),
    )
    .unwrap();

    assert_eq!(run.report.disposition, TrainingRunDisposition::Completed);
    assert_eq!(run.report.completed_iterations, 3);
    assert_eq!(
        sequence.borrow().as_slice(),
        ["sink:2:Periodic", "event:2:Periodic", "completed:Completed"]
    );
}

#[test]
fn periodic_checkpoint_sink_pause_reuses_commit_and_finishes_current_iteration() {
    let temp = tempfile::tempdir().unwrap();
    let dataset = tiny_training_dataset(&temp, "periodic-sink-pause-frame", 1);
    let config = tiny_training_config(3);
    let identity =
        TrainingIdentity::from_canonical_content(&dataset, b"periodic-sink-pause", &config)
            .unwrap();
    let control = TrainingControl::default();
    let sink_control = control.clone();
    let sequence = Rc::new(RefCell::new(Vec::new()));
    let sink_sequence = Rc::clone(&sequence);
    let event_sequence = Rc::clone(&sequence);

    let run = train_splats(
        &dataset,
        &config,
        TrainingOptions::new()
            .with_control(control)
            .with_identity(identity)
            .with_checkpoint_policy(TrainingCheckpointPolicy { every: Some(1) })
            .with_checkpoint_sink(move |ready| {
                sink_sequence
                    .borrow_mut()
                    .push(format!("sink:{}:{:?}", ready.iteration, ready.reason));
                sink_control.request_pause();
                Ok(())
            })
            .with_event_sink(move |event| match event {
                TrainingEvent::CheckpointReady(ready) => event_sequence
                    .borrow_mut()
                    .push(format!("checkpoint:{}:{:?}", ready.iteration, ready.reason)),
                TrainingEvent::RunPaused(paused) => event_sequence
                    .borrow_mut()
                    .push(format!("paused:{}", paused.completed_iterations)),
                TrainingEvent::RunCompleted(completed) => event_sequence
                    .borrow_mut()
                    .push(format!("completed:{:?}", completed.report.disposition)),
                _ => {}
            }),
    )
    .unwrap();

    assert_eq!(run.report.completed_iterations, 1);
    assert!(!run.report.cancelled);
    assert_eq!(run.report.disposition, TrainingRunDisposition::Paused);
    assert_eq!(
        sequence.borrow().as_slice(),
        [
            "sink:1:Periodic",
            "checkpoint:1:Periodic",
            "paused:1",
            "completed:Paused"
        ]
    );
}

#[test]
fn periodic_checkpoint_event_pause_reuses_commit_and_finishes_current_iteration() {
    let temp = tempfile::tempdir().unwrap();
    let dataset = tiny_training_dataset(&temp, "periodic-event-pause-frame", 1);
    let config = tiny_training_config(3);
    let identity =
        TrainingIdentity::from_canonical_content(&dataset, b"periodic-event-pause", &config)
            .unwrap();
    let control = TrainingControl::default();
    let event_control = control.clone();
    let sequence = Rc::new(RefCell::new(Vec::new()));
    let sink_sequence = Rc::clone(&sequence);
    let event_sequence = Rc::clone(&sequence);

    let run = train_splats(
        &dataset,
        &config,
        TrainingOptions::new()
            .with_control(control)
            .with_identity(identity)
            .with_checkpoint_policy(TrainingCheckpointPolicy { every: Some(1) })
            .with_checkpoint_sink(move |ready| {
                sink_sequence
                    .borrow_mut()
                    .push(format!("sink:{}:{:?}", ready.iteration, ready.reason));
                Ok(())
            })
            .with_event_sink(move |event| match event {
                TrainingEvent::CheckpointReady(ready) => {
                    event_sequence
                        .borrow_mut()
                        .push(format!("checkpoint:{}:{:?}", ready.iteration, ready.reason));
                    event_control.request_pause();
                }
                TrainingEvent::RunPaused(paused) => event_sequence
                    .borrow_mut()
                    .push(format!("paused:{}", paused.completed_iterations)),
                TrainingEvent::RunCompleted(completed) => event_sequence
                    .borrow_mut()
                    .push(format!("completed:{:?}", completed.report.disposition)),
                _ => {}
            }),
    )
    .unwrap();

    assert_eq!(run.report.completed_iterations, 1);
    assert!(!run.report.cancelled);
    assert_eq!(run.report.disposition, TrainingRunDisposition::Paused);
    assert_eq!(
        sequence.borrow().as_slice(),
        [
            "sink:1:Periodic",
            "checkpoint:1:Periodic",
            "paused:1",
            "completed:Paused"
        ]
    );
}

#[test]
fn cancel_requested_after_pause_wins_at_complete_iteration_boundary() {
    let temp = tempfile::tempdir().unwrap();
    let dataset = tiny_training_dataset(&temp, "cancel-frame", 1);
    let config = tiny_training_config(3);
    let identity =
        TrainingIdentity::from_canonical_content(&dataset, b"cancel-reconstruction", &config)
            .unwrap();
    let control = TrainingControl::new(TrainingEventCadence {
        progress_every: 1,
        snapshot_every: None,
    });
    let event_control = control.clone();
    let terminal = Rc::new(RefCell::new(Vec::new()));
    let captured_terminal = Rc::clone(&terminal);

    let run = train_splats(
        &dataset,
        &config,
        TrainingOptions::new()
            .with_control(control)
            .with_identity(identity)
            .with_event_sink(move |event| match event {
                TrainingEvent::IterationProgress(progress) if progress.iteration == 1 => {
                    event_control.request_pause();
                    event_control.request_cancel();
                }
                TrainingEvent::CheckpointReady(_) => captured_terminal
                    .borrow_mut()
                    .push("checkpoint".to_string()),
                TrainingEvent::RunPaused(_) => {
                    captured_terminal.borrow_mut().push("paused".to_string())
                }
                TrainingEvent::RunCancelled(cancelled) => captured_terminal
                    .borrow_mut()
                    .push(format!("cancelled:{}", cancelled.completed_iterations)),
                TrainingEvent::RunCompleted(completed) => captured_terminal
                    .borrow_mut()
                    .push(format!("completed:{:?}", completed.report.disposition)),
                _ => {}
            }),
    )
    .unwrap();

    assert_eq!(run.report.completed_iterations, 1);
    assert!(run.report.cancelled);
    assert_eq!(run.report.disposition, TrainingRunDisposition::Cancelled);
    assert_eq!(
        terminal.borrow().as_slice(),
        ["cancelled:1", "completed:Cancelled"]
    );
}

#[test]
fn resume_at_completed_target_runs_zero_iterations_and_reports_completed() {
    let temp = tempfile::tempdir().unwrap();
    let dataset = tiny_training_dataset(&temp, "zero-resume-frame", 1);
    let config = tiny_training_config(7);
    let identity =
        TrainingIdentity::from_canonical_content(&dataset, b"zero-resume", &config).unwrap();
    let mut checkpoint = checkpoint_fixture(7);
    checkpoint.identity = identity.clone();
    checkpoint.frame_shuffle_seed = config.data.frame_shuffle_seed;
    let events = Rc::new(RefCell::new(Vec::new()));
    let captured_events = Rc::clone(&events);

    let run = train_splats(
        &dataset,
        &config,
        TrainingOptions::new()
            .with_identity(identity)
            .with_resume_checkpoint(checkpoint)
            .with_event_sink(move |event| match event {
                TrainingEvent::IterationProgress(progress) => captured_events
                    .borrow_mut()
                    .push(format!("progress:{}", progress.iteration)),
                TrainingEvent::RunCompleted(completed) => captured_events
                    .borrow_mut()
                    .push(format!("completed:{:?}", completed.report.disposition)),
                TrainingEvent::RunPaused(_) => {
                    captured_events.borrow_mut().push("paused".to_string())
                }
                TrainingEvent::RunCancelled(_) => {
                    captured_events.borrow_mut().push("cancelled".to_string())
                }
                _ => {}
            }),
    )
    .unwrap();

    assert_eq!(run.report.completed_iterations, 7);
    assert_eq!(run.report.final_loss, Some(0.125));
    let telemetry = run.report.telemetry.as_ref().unwrap();
    assert_eq!(telemetry.final_loss, Some(0.125));
    assert_eq!(telemetry.final_step_loss, Some(0.125));
    assert_eq!(run.report.gaussian_count, 1);
    assert_eq!(run.splats.len(), 1);
    assert!(!run.report.cancelled);
    assert_eq!(run.report.disposition, TrainingRunDisposition::Completed);
    assert_eq!(events.borrow().as_slice(), ["completed:Completed"]);
}

#[test]
fn pause_after_iteration_seven_commits_checkpoint_before_terminal_events() {
    let temp = tempfile::tempdir().unwrap();
    let image_path = temp.path().join("pause-frame.rgb");
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
    fs::write(&image_path, pixels).unwrap();

    let mut dataset = TrainingDataset::new(Intrinsics::new(12.0, 12.0, 8.0, 8.0, 16, 16));
    dataset.add_pose(ScenePose::new(0, image_path, SE3::identity(), 0.0));
    dataset.add_point([0.0, 0.0, 2.0], Some([0.25, 0.5, 0.75]));
    dataset.add_point([0.25, -0.2, 2.5], Some([0.75, 0.25, 0.5]));
    let config = TrainingConfig {
        iterations: 20,
        raster: rustscan_gs::TrainingRasterConfig {
            render_scale: 1.0,
            ..Default::default()
        },
        initialization: rustscan_gs::TrainingInitializationConfig {
            max_initial_gaussians: 2,
            ..Default::default()
        },
        data: rustscan_gs::TrainingDataConfig {
            frame_cache_capacity: 1,
            frame_prefetch_ahead: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let identity =
        TrainingIdentity::from_canonical_content(&dataset, b"pause-reconstruction", &config)
            .unwrap();
    let control = TrainingControl::new(TrainingEventCadence {
        progress_every: 1,
        snapshot_every: None,
    });
    let sequence = Rc::new(RefCell::new(Vec::new()));
    let captured_checkpoint = Rc::new(RefCell::new(None));

    let event_control = control.clone();
    let event_sequence = Rc::clone(&sequence);
    let checkpoint_sequence = Rc::clone(&sequence);
    let sink_checkpoint = Rc::clone(&captured_checkpoint);
    let run = train_splats(
        &dataset,
        &config,
        TrainingOptions::new()
            .with_control(control)
            .with_identity(identity)
            .with_checkpoint_policy(TrainingCheckpointPolicy { every: Some(7) })
            .with_checkpoint_sink(move |ready| {
                checkpoint_sequence
                    .borrow_mut()
                    .push(format!("sink:{}", ready.iteration));
                *sink_checkpoint.borrow_mut() = Some(ready.checkpoint.clone());
                Ok(())
            })
            .with_event_sink(move |event| match event {
                TrainingEvent::IterationProgress(progress) => {
                    event_sequence
                        .borrow_mut()
                        .push(format!("progress:{}", progress.iteration));
                    if progress.iteration == 7 {
                        event_control.request_pause();
                    }
                }
                TrainingEvent::CheckpointReady(ready) => event_sequence
                    .borrow_mut()
                    .push(format!("checkpoint:{}:{:?}", ready.iteration, ready.reason)),
                TrainingEvent::RunPaused(paused) => event_sequence
                    .borrow_mut()
                    .push(format!("paused:{}", paused.completed_iterations)),
                TrainingEvent::RunCompleted(completed) => event_sequence
                    .borrow_mut()
                    .push(format!("completed:{:?}", completed.report.disposition)),
                _ => {}
            }),
    )
    .unwrap();

    let checkpoint = captured_checkpoint.borrow();
    let checkpoint = checkpoint.as_ref().expect("pause checkpoint committed");
    assert_eq!(run.report.completed_iterations, 7);
    assert!(!run.report.cancelled);
    assert_eq!(run.report.disposition, TrainingRunDisposition::Paused);
    assert_eq!(checkpoint.completed_iterations, 7);
    assert_eq!(
        sequence.borrow().as_slice(),
        [
            "progress:1",
            "progress:2",
            "progress:3",
            "progress:4",
            "progress:5",
            "progress:6",
            "progress:7",
            "sink:7",
            "checkpoint:7:Pause",
            "paused:7",
            "completed:Paused",
        ]
    );
}

#[test]
fn resumed_training_matches_uninterrupted_training_state() {
    const FRAME_ORDER: [usize; 3] = [1, 0, 2];

    let temp = tempfile::tempdir().unwrap();
    let dataset = continuity_training_dataset(&temp);
    for left in 0..dataset.poses.len() {
        for right in left + 1..dataset.poses.len() {
            assert_ne!(
                dataset.poses[left].image_path,
                dataset.poses[right].image_path
            );
            assert_ne!(dataset.poses[left].pose, dataset.poses[right].pose);
        }
    }
    let mut config = tiny_training_config(12);
    config.data.frame_prefetch_ahead = 2;
    config.data.frame_shuffle_seed = 0x5eed_cafe;
    let identity =
        TrainingIdentity::from_canonical_content(&dataset, b"resumed-continuity", &config).unwrap();

    let captured_checkpoint = Rc::new(RefCell::new(None));
    let sink_checkpoint = Rc::clone(&captured_checkpoint);
    let uninterrupted = train_splats(
        &dataset,
        &config,
        TrainingOptions::new()
            .with_identity(identity.clone())
            .with_checkpoint_policy(TrainingCheckpointPolicy { every: Some(7) })
            .with_checkpoint_sink(move |ready| {
                assert_eq!(ready.iteration, 7);
                assert_eq!(ready.reason, TrainingCheckpointReason::Periodic);
                *sink_checkpoint.borrow_mut() = Some(ready.checkpoint.clone());
                Ok(())
            }),
    )
    .unwrap();
    assert_eq!(uninterrupted.report.completed_iterations, 12);
    assert_eq!(
        uninterrupted.report.disposition,
        TrainingRunDisposition::Completed
    );
    assert!(!uninterrupted.report.cancelled);
    let uninterrupted_samples = &uninterrupted
        .report
        .telemetry
        .as_ref()
        .unwrap()
        .loss_curve_samples;
    assert_eq!(
        uninterrupted_samples
            .iter()
            .map(|sample| sample.iteration)
            .collect::<Vec<_>>(),
        [12]
    );
    assert_eq!(
        uninterrupted_samples[0].frame_idx,
        FRAME_ORDER[(12 - 1) % dataset.poses.len()]
    );
    let checkpoint = captured_checkpoint
        .borrow()
        .clone()
        .expect("periodic checkpoint");
    assert_eq!(checkpoint.completed_iterations, 7);
    assert_eq!(
        checkpoint.frame_shuffle_seed,
        config.data.frame_shuffle_seed
    );

    let resumed = train_splats(
        &dataset,
        &config,
        TrainingOptions::new()
            .with_identity(identity)
            .with_resume_checkpoint(checkpoint),
    )
    .unwrap();
    assert_eq!(resumed.report.completed_iterations, 12);
    assert_eq!(
        resumed.report.disposition,
        TrainingRunDisposition::Completed
    );
    assert!(!resumed.report.cancelled);
    assert_eq!(
        resumed.report.gaussian_count,
        uninterrupted.report.gaussian_count
    );
    assert_eq!(resumed.report.sh_degree, uninterrupted.report.sh_degree);
    assert_eq!(resumed.splats.len(), uninterrupted.splats.len());

    let expected = uninterrupted.splats.as_view();
    let actual = resumed.splats.as_view();
    assert_eq!(actual.sh_degree, expected.sh_degree);
    for (name, expected, actual) in [
        ("positions", expected.positions, actual.positions),
        ("log_scales", expected.log_scales, actual.log_scales),
        ("rotations", expected.rotations, actual.rotations),
        (
            "opacity_logits",
            expected.opacity_logits,
            actual.opacity_logits,
        ),
        ("sh_coeffs", expected.sh_coeffs, actual.sh_coeffs),
    ] {
        assert_eq!(actual.len(), expected.len(), "{name} length");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                actual.is_finite(),
                "{name}[{index}] is not finite: {actual}"
            );
            assert!(
                expected.is_finite(),
                "{name}[{index}] is not finite: {expected}"
            );
            assert!(
                (actual - expected).abs() <= 1e-5,
                "{name}[{index}] differs: resumed={actual}, uninterrupted={expected}"
            );
        }
    }

    let telemetry = resumed.report.telemetry.as_ref().unwrap();
    let first_sample = telemetry
        .loss_curve_samples
        .first()
        .expect("resume telemetry sample");
    assert_eq!(first_sample.iteration, 8);
    assert_eq!(
        first_sample.frame_idx,
        FRAME_ORDER[(8 - 1) % dataset.poses.len()]
    );
    assert!(first_sample.total.is_some_and(f32::is_finite));
    assert!(resumed.report.final_loss.is_some_and(f32::is_finite));
    assert!(resumed.report.final_step_loss.is_some_and(f32::is_finite));
    assert!(
        (resumed.report.final_loss.unwrap() - uninterrupted.report.final_loss.unwrap()).abs()
            <= 1e-5
    );
    assert!(
        (resumed.report.final_step_loss.unwrap() - uninterrupted.report.final_step_loss.unwrap())
            .abs()
            <= 1e-5
    );
}

fn assert_tensor_checkpoint_close(label: &str, left: &TensorCheckpoint, right: &TensorCheckpoint) {
    assert_eq!(left.shape, right.shape, "{label} shape");
    assert_eq!(left.values.len(), right.values.len(), "{label} length");
    for (index, (&a, &b)) in left.values.iter().zip(&right.values).enumerate() {
        assert!(
            (a - b).abs() <= 1e-5,
            "{label}[{index}] differs: {a} vs {b}"
        );
    }
}

fn assert_topology_checkpoints_close(left: &TopologyCheckpoint, right: &TopologyCheckpoint) {
    assert_tensor_checkpoint_close("grad_2d", &left.grad_2d, &right.grad_2d);
    assert_tensor_checkpoint_close(
        "screen_grad_2d",
        &left.screen_grad_2d,
        &right.screen_grad_2d,
    );
    assert_tensor_checkpoint_close("abs_grad_2d", &left.abs_grad_2d, &right.abs_grad_2d);
    assert_tensor_checkpoint_close(
        "abs_pixel_grad_2d",
        &left.abs_pixel_grad_2d,
        &right.abs_pixel_grad_2d,
    );
    assert_tensor_checkpoint_close(
        "pixel_coverage",
        &left.pixel_coverage,
        &right.pixel_coverage,
    );
    assert_tensor_checkpoint_close("camera_depth", &left.camera_depth, &right.camera_depth);
    assert_tensor_checkpoint_close("grad_color", &left.grad_color, &right.grad_color);
    assert_tensor_checkpoint_close(
        "num_observations",
        &left.num_observations,
        &right.num_observations,
    );
    assert_tensor_checkpoint_close(
        "visible_observations",
        &left.visible_observations,
        &right.visible_observations,
    );
    assert_tensor_checkpoint_close(
        "actual_visible_observations",
        &left.actual_visible_observations,
        &right.actual_visible_observations,
    );
    assert_eq!(left.splat_birth_iterations, right.splat_birth_iterations);
    assert_eq!(left.splat_invisible_windows, right.splat_invisible_windows);
    assert_eq!(
        left.visibility_window_baseline,
        right.visibility_window_baseline
    );
    assert_eq!(
        left.actual_visibility_window_baseline,
        right.actual_visibility_window_baseline
    );
}

fn assert_adam_checkpoints_close(left: &AdamCheckpoint, right: &AdamCheckpoint) {
    for (label, a, b) in [
        ("transforms", &left.transforms, &right.transforms),
        ("sh_coeffs", &left.sh_coeffs, &right.sh_coeffs),
        ("raw_opacities", &left.raw_opacities, &right.raw_opacities),
    ] {
        assert_eq!(a.step, b.step, "{label} step");
        match (&a.moment1, &b.moment1) {
            (Some(am), Some(bm)) => {
                assert_tensor_checkpoint_close(&format!("{label}.moment1"), am, bm)
            }
            (None, None) => {}
            _ => panic!("{label}.moment1 presence mismatch"),
        }
        match (&a.moment2, &b.moment2) {
            (Some(am), Some(bm)) => {
                assert_tensor_checkpoint_close(&format!("{label}.moment2"), am, bm)
            }
            (None, None) => {}
            _ => panic!("{label}.moment2 presence mismatch"),
        }
    }
}

#[test]
fn mid_window_topology_resume_matches_uninterrupted_at_iteration_eight() {
    let temp = tempfile::tempdir().unwrap();
    let dataset = continuity_training_dataset(&temp);
    let mut config = tiny_training_config(8);
    config.data.frame_shuffle_seed = 0x71c0_1067;
    config.litegs.topology.refine_every = 4;
    config.litegs.topology.opacity_reset_interval = 10_000;
    config.litegs.topology.target_primitives = 64;
    let identity =
        TrainingIdentity::from_canonical_content(&dataset, b"mid-window-topology", &config)
            .unwrap();

    let uninterrupted_final = Rc::new(RefCell::new(None));
    let sink_uninterrupted = Rc::clone(&uninterrupted_final);
    let uninterrupted = train_splats(
        &dataset,
        &config,
        TrainingOptions::new()
            .with_identity(identity.clone())
            .with_checkpoint_policy(TrainingCheckpointPolicy { every: Some(8) })
            .with_checkpoint_sink(move |ready| {
                assert_eq!(ready.iteration, 8);
                *sink_uninterrupted.borrow_mut() = Some(ready.checkpoint.clone());
                Ok(())
            }),
    )
    .unwrap();
    assert_eq!(uninterrupted.report.completed_iterations, 8);
    let uninterrupted_ckpt = uninterrupted_final
        .borrow()
        .clone()
        .expect("uninterrupted final checkpoint");

    let mid_checkpoint = Rc::new(RefCell::new(None));
    let sink_mid = Rc::clone(&mid_checkpoint);
    let control = TrainingControl::default();
    let pause_control = control.clone();
    let paused = train_splats(
        &dataset,
        &config,
        TrainingOptions::new()
            .with_control(control)
            .with_identity(identity.clone())
            .with_checkpoint_policy(TrainingCheckpointPolicy { every: Some(2) })
            .with_checkpoint_sink(move |ready| {
                if ready.iteration == 2 {
                    assert_eq!(ready.reason, TrainingCheckpointReason::Periodic);
                    // Mid densify window (refine_every=4) and outside loss cadence (20).
                    *sink_mid.borrow_mut() = Some(ready.checkpoint.clone());
                    pause_control.request_pause();
                }
                Ok(())
            }),
    )
    .unwrap();
    assert_eq!(paused.report.disposition, TrainingRunDisposition::Paused);
    assert_eq!(paused.report.completed_iterations, 2);
    let resume_from = mid_checkpoint
        .borrow()
        .clone()
        .expect("mid-window checkpoint at iteration 2");
    assert_eq!(resume_from.completed_iterations, 2);
    assert_eq!(
        resume_from.topology.visibility_window_baseline.len(),
        resume_from.splats.len()
    );

    let resumed_final = Rc::new(RefCell::new(None));
    let sink_resumed = Rc::clone(&resumed_final);
    let resumed = train_splats(
        &dataset,
        &config,
        TrainingOptions::new()
            .with_identity(identity)
            .with_resume_checkpoint(resume_from)
            .with_checkpoint_policy(TrainingCheckpointPolicy { every: Some(8) })
            .with_checkpoint_sink(move |ready| {
                assert_eq!(ready.iteration, 8);
                *sink_resumed.borrow_mut() = Some(ready.checkpoint.clone());
                Ok(())
            }),
    )
    .unwrap();
    assert_eq!(resumed.report.completed_iterations, 8);
    let resumed_ckpt = resumed_final
        .borrow()
        .clone()
        .expect("resumed final checkpoint");

    assert_eq!(
        resumed_ckpt.completed_iterations,
        uninterrupted_ckpt.completed_iterations
    );
    assert_eq!(
        resumed_ckpt.optimizer.transforms.step,
        uninterrupted_ckpt.optimizer.transforms.step
    );
    assert_topology_checkpoints_close(&resumed_ckpt.topology, &uninterrupted_ckpt.topology);
    assert_adam_checkpoints_close(&resumed_ckpt.optimizer, &uninterrupted_ckpt.optimizer);

    let expected = uninterrupted_ckpt.splats.as_view();
    let actual = resumed_ckpt.splats.as_view();
    for (name, expected, actual) in [
        ("positions", expected.positions, actual.positions),
        ("log_scales", expected.log_scales, actual.log_scales),
        ("rotations", expected.rotations, actual.rotations),
        (
            "opacity_logits",
            expected.opacity_logits,
            actual.opacity_logits,
        ),
        ("sh_coeffs", expected.sh_coeffs, actual.sh_coeffs),
    ] {
        assert_eq!(actual.len(), expected.len(), "{name} length");
        for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5,
                "{name}[{index}] differs: resumed={a}, uninterrupted={b}"
            );
        }
    }

    let left_samples = &uninterrupted
        .report
        .telemetry
        .as_ref()
        .unwrap()
        .topology
        .topology_step_samples;
    let right_samples = &resumed
        .report
        .telemetry
        .as_ref()
        .unwrap()
        .topology
        .topology_step_samples;
    assert_eq!(left_samples.len(), right_samples.len());
    for (left, right) in left_samples.iter().zip(right_samples) {
        assert_eq!(left.iteration, right.iteration);
        assert_eq!(left.gaussian_count, right.gaussian_count);
        assert_eq!(left.clone_candidates, right.clone_candidates);
        assert_eq!(left.split_candidates, right.split_candidates);
        assert_eq!(left.prune_candidates, right.prune_candidates);
        assert_eq!(left.growth_candidates, right.growth_candidates);
    }
    assert_eq!(
        uninterrupted
            .report
            .telemetry
            .as_ref()
            .unwrap()
            .topology
            .densify_added,
        resumed
            .report
            .telemetry
            .as_ref()
            .unwrap()
            .topology
            .densify_added
    );
    assert_eq!(
        uninterrupted
            .report
            .telemetry
            .as_ref()
            .unwrap()
            .topology
            .prune_removed,
        resumed
            .report
            .telemetry
            .as_ref()
            .unwrap()
            .topology
            .prune_removed
    );
}

#[test]
fn selection_metadata_roundtrips_and_v2_loads_as_absent() {
    let mut checkpoint = checkpoint_fixture(3);
    checkpoint.selection = Some(CheckpointFrameSelectionMeta {
        eval_split_kind: Some("in-view".into()),
        train_stable_ids: vec![10, 20, 30],
        eval_stable_ids: vec![40],
        train_loader_frame_ids: vec![20, 10, 30],
        manifest_fingerprint: Some("manifest-fp".into()),
        selection_fingerprint: Some("sel-fp".into()),
        eval_selection_fingerprint: Some("eval-sel-fp".into()),
        ..Default::default()
    });

    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("selection-v3.rgscp");
    save_training_checkpoint(&path, &checkpoint).unwrap();
    let loaded = load_training_checkpoint(&path).unwrap();
    assert_eq!(loaded.version, TRAINING_CHECKPOINT_VERSION);
    assert_eq!(loaded.selection, checkpoint.selection);

    // Write a raw v2 payload (no selection field) and migrate.
    let view = checkpoint.splats.as_view();
    let v2_bytes = encode_unchecked(&SerializedTrainingCheckpointV2 {
        version: TRAINING_CHECKPOINT_VERSION_V2,
        identity: checkpoint.identity.clone(),
        completed_iterations: checkpoint.completed_iterations,
        latest_loss: checkpoint.latest_loss,
        splats: SerializedHostSplats {
            positions: view.positions.to_vec(),
            log_scales: view.log_scales.to_vec(),
            rotations: view.rotations.to_vec(),
            opacity_logits: view.opacity_logits.to_vec(),
            sh_coeffs: view.sh_coeffs.to_vec(),
            sh_degree: view.sh_degree,
        },
        optimizer: checkpoint.optimizer.clone(),
        topology: checkpoint.topology.clone(),
        frame_shuffle_seed: checkpoint.frame_shuffle_seed,
        active_sh_degree: checkpoint.active_sh_degree,
    });
    let v2_path = temp.path().join("selection-v2.rgscp");
    fs::write(&v2_path, v2_bytes).unwrap();
    let (migrated, migration) = load_training_checkpoint_with_migration(&v2_path).unwrap();
    assert_eq!(migration, CheckpointMigration::V2SelectionMetaAbsent);
    assert_eq!(migrated.version, TRAINING_CHECKPOINT_VERSION);
    assert!(
        migrated.selection.is_none(),
        "v2 migration must mark selection absent, not invent IDs"
    );
}

#[test]
fn resume_rejects_mismatched_selection_metadata() {
    let temp = tempfile::tempdir().unwrap();
    let dataset = tiny_training_dataset(&temp, "sel-meta", 2);
    let config = tiny_training_config(2);
    let identity =
        TrainingIdentity::from_canonical_content(&dataset, b"sel-recon", &config).unwrap();
    let mut checkpoint = checkpoint_fixture(1);
    checkpoint.identity = identity.clone();
    checkpoint.selection = Some(CheckpointFrameSelectionMeta {
        selection_fingerprint: Some("old".into()),
        train_stable_ids: vec![0, 1],
        ..Default::default()
    });
    let expected = CheckpointFrameSelectionMeta {
        selection_fingerprint: Some("new".into()),
        train_stable_ids: vec![0, 1],
        ..Default::default()
    };
    let error = train_splats(
        &dataset,
        &config,
        TrainingOptions::new()
            .with_identity(identity)
            .with_selection(expected)
            .with_resume_checkpoint(checkpoint),
    )
    .unwrap_err();
    assert_invalid_input_contains(error, "frame-selection metadata");
}

#[test]
fn resume_without_with_selection_inherits_v3_selection_through_save_reload() {
    let temp = tempfile::tempdir().unwrap();
    let dataset = tiny_training_dataset(&temp, "sel-inherit", 2);
    let mut config = tiny_training_config(1);
    let identity =
        TrainingIdentity::from_canonical_content(&dataset, b"sel-inherit-recon", &config).unwrap();
    let selection = {
        let mut meta = verified_selection_meta(&[0, 1], &[1], &[0, 1], Some("in-view"));
        meta.manifest_fingerprint = Some("manifest-fp".into());
        meta
    };

    let captured = Rc::new(RefCell::new(None));
    let sink = Rc::clone(&captured);
    train_splats(
        &dataset,
        &config,
        TrainingOptions::new()
            .with_identity(identity.clone())
            .with_selection(selection.clone())
            .with_checkpoint_policy(TrainingCheckpointPolicy { every: Some(1) })
            .with_checkpoint_sink(move |ready| {
                *sink.borrow_mut() = Some(ready.checkpoint.clone());
                Ok(())
            }),
    )
    .unwrap();
    let first = captured.borrow().clone().expect("first checkpoint");
    assert_eq!(first.selection.as_ref(), Some(&selection));
    let first_path = temp.path().join("first.rgscp");
    save_training_checkpoint(&first_path, &first).unwrap();
    let loaded = load_training_checkpoint(&first_path).unwrap();
    assert_eq!(loaded.selection.as_ref(), Some(&selection));

    config.iterations = 2;
    let identity2 =
        TrainingIdentity::from_canonical_content(&dataset, b"sel-inherit-recon", &config).unwrap();
    let captured2 = Rc::new(RefCell::new(None));
    let sink2 = Rc::clone(&captured2);
    train_splats(
        &dataset,
        &config,
        TrainingOptions::new()
            .with_identity(identity2)
            .with_resume_checkpoint(loaded)
            // Intentionally omit with_selection — must inherit from checkpoint.
            .with_checkpoint_policy(TrainingCheckpointPolicy { every: Some(2) })
            .with_checkpoint_sink(move |ready| {
                *sink2.borrow_mut() = Some(ready.checkpoint.clone());
                Ok(())
            }),
    )
    .unwrap();
    let second = captured2.borrow().clone().expect("second checkpoint");
    assert_eq!(
        second.selection.as_ref(),
        Some(&selection),
        "resume without with_selection must not drop v3 selection metadata"
    );
    let second_path = temp.path().join("second.rgscp");
    save_training_checkpoint(&second_path, &second).unwrap();
    let reloaded = load_training_checkpoint(&second_path).unwrap();
    assert_eq!(reloaded.selection.as_ref(), Some(&selection));
}

#[test]
fn selection_meta_rejects_duplicate_train_stable_ids() {
    let temp = tempfile::tempdir().unwrap();
    let dataset = tiny_training_dataset(&temp, "sel-dup-train", 3);
    let mut meta = verified_selection_meta(&[0, 1], &[], &[0, 1], None);
    meta.train_stable_ids = vec![0, 0, 1];
    // Fingerprint no longer matches after mutating IDs; either duplicate or fingerprint
    // rejection is acceptable — prefer the duplicate check by clearing fingerprint.
    meta.selection_fingerprint = None;
    let err = resolve_checkpoint_selection(None, Some(&meta), &dataset).unwrap_err();
    assert_invalid_input_contains(err, "duplicate stable id");
}

#[test]
fn selection_meta_rejects_holdout_train_eval_overlap() {
    let temp = tempfile::tempdir().unwrap();
    let dataset = tiny_training_dataset(&temp, "sel-holdout-overlap", 3);
    let meta = verified_selection_meta(&[0, 1], &[1, 2], &[0, 1], Some("holdout"));
    let err = resolve_checkpoint_selection(None, Some(&meta), &dataset).unwrap_err();
    assert_invalid_input_contains(err, "train/eval");
}

#[test]
fn selection_meta_rejects_loader_id_outside_train_selection() {
    let temp = tempfile::tempdir().unwrap();
    let dataset = tiny_training_dataset(&temp, "sel-loader-oob", 3);
    let meta = {
        let mut meta = verified_selection_meta(&[0, 1], &[], &[0, 1], None);
        meta.train_loader_frame_ids = vec![0, 1, 2];
        meta
    };
    let err = resolve_checkpoint_selection(None, Some(&meta), &dataset).unwrap_err();
    assert_invalid_input_contains(err, "outside the train selection");
}

#[test]
fn selection_meta_allows_loader_oversample_duplicates_within_train() {
    let temp = tempfile::tempdir().unwrap();
    let dataset = tiny_training_dataset(&temp, "sel-loader-dup-ok", 3);
    let meta = verified_selection_meta(&[0, 1], &[], &[0, 1, 0, 1, 0], None);
    let resolved = resolve_checkpoint_selection(None, Some(&meta), &dataset).unwrap();
    assert_eq!(resolved.as_ref(), Some(&meta));
}

#[test]
fn selection_meta_rejects_fingerprint_inconsistent_with_selection_fields() {
    let temp = tempfile::tempdir().unwrap();
    let dataset = tiny_training_dataset(&temp, "sel-fp-mismatch", 3);
    let mut meta = verified_selection_meta(&[0, 1], &[], &[0, 1], None);
    meta.selection_fingerprint = Some("not-the-recomputed-fingerprint".into());
    let err = resolve_checkpoint_selection(None, Some(&meta), &dataset).unwrap_err();
    assert_invalid_input_contains(err, "selection_fingerprint does not match");
}

#[test]
fn selection_meta_rejects_request_stable_id_mismatch_on_reselect() {
    let temp = tempfile::tempdir().unwrap();
    let dataset = tiny_training_dataset(&temp, "sel-reselect-mismatch", 4);
    // Request includes only 0..=1, but stored train IDs claim 0..=2.
    let request = FrameSelectionRequest {
        include_ranges: vec![rustscan_gs::FrameIdRange { start: 0, end: 1 }],
        max_frames: 0,
        frame_stride: 1,
        ..Default::default()
    };
    let mut meta = verified_selection_meta(&[0, 1], &[], &[0, 1], None);
    meta.include_ranges = request.include_ranges.clone();
    meta.allowed_ids = None;
    meta.train_stable_ids = vec![0, 1, 2];
    meta.train_loader_frame_ids = vec![0, 1, 2];
    meta.selection_fingerprint = Some(fingerprint_frame_selection(&[0, 1, 2], &request));
    let err = resolve_checkpoint_selection(None, Some(&meta), &dataset).unwrap_err();
    assert_invalid_input_contains(err, "does not reproduce train_stable_ids");
}

#[test]
fn legacy_v3_checkpoint_loads_and_resumes_as_unverified_selection() {
    let temp = tempfile::tempdir().unwrap();
    let dataset = tiny_training_dataset(&temp, "sel-v3-legacy", 2);
    let mut config = tiny_training_config(1);
    let identity =
        TrainingIdentity::from_canonical_content(&dataset, b"sel-v3-legacy-recon", &config)
            .unwrap();

    // Build a current checkpoint, then rewrite bytes as the historical v3 layout
    // (selection without request fields).
    let mut seed = checkpoint_fixture(1);
    seed.identity = identity.clone();
    let view = seed.splats.as_view();
    let v3_selection = SerializedSelectionMetaV3 {
        eval_split_kind: Some("in-view".into()),
        train_stable_ids: vec![0, 1],
        eval_stable_ids: vec![],
        train_loader_frame_ids: vec![0, 1],
        manifest_fingerprint: None,
        // Historical fingerprint string — must not be re-verified against empty request.
        selection_fingerprint: Some("legacy-v3-sel-fp".into()),
        eval_selection_fingerprint: None,
    };
    let v3_bytes = encode_unchecked(&SerializedTrainingCheckpointV3 {
        version: TRAINING_CHECKPOINT_VERSION_V3,
        identity: seed.identity.clone(),
        completed_iterations: seed.completed_iterations,
        latest_loss: seed.latest_loss,
        splats: SerializedHostSplats {
            positions: view.positions.to_vec(),
            log_scales: view.log_scales.to_vec(),
            rotations: view.rotations.to_vec(),
            opacity_logits: view.opacity_logits.to_vec(),
            sh_coeffs: view.sh_coeffs.to_vec(),
            sh_degree: view.sh_degree,
        },
        optimizer: seed.optimizer.clone(),
        topology: seed.topology.clone(),
        frame_shuffle_seed: seed.frame_shuffle_seed,
        active_sh_degree: seed.active_sh_degree,
        selection: Some(v3_selection),
    });
    let v3_path = temp.path().join("legacy-v3.rgscp");
    fs::write(&v3_path, v3_bytes).unwrap();

    let (loaded, migration) = load_training_checkpoint_with_migration(&v3_path).unwrap();
    assert_eq!(
        migration,
        CheckpointMigration::V3SelectionMetaLegacyUnverified
    );
    assert_eq!(loaded.version, TRAINING_CHECKPOINT_VERSION);
    let selection = loaded.selection.as_ref().expect("selection preserved");
    assert_eq!(
        selection.provenance,
        SelectionMetaProvenance::LegacyUnverified
    );
    assert!(!selection.is_verified());
    assert!(selection.include_ranges.is_empty());
    assert!(selection.allowed_ids.is_none());
    assert_eq!(selection.frame_stride, 0);
    assert_eq!(
        selection.selection_fingerprint.as_deref(),
        Some("legacy-v3-sel-fp")
    );
    assert_eq!(selection.train_stable_ids, vec![0, 1]);

    // Resume without with_selection inherits legacy/unverified meta safely.
    config.iterations = 2;
    let identity2 =
        TrainingIdentity::from_canonical_content(&dataset, b"sel-v3-legacy-recon", &config)
            .unwrap();
    let captured = Rc::new(RefCell::new(None));
    let sink = Rc::clone(&captured);
    train_splats(
        &dataset,
        &config,
        TrainingOptions::new()
            .with_identity(identity2)
            .with_resume_checkpoint(loaded)
            .with_checkpoint_policy(TrainingCheckpointPolicy { every: Some(2) })
            .with_checkpoint_sink(move |ready| {
                *sink.borrow_mut() = Some(ready.checkpoint.clone());
                Ok(())
            }),
    )
    .unwrap();
    let resumed = captured.borrow().clone().expect("resumed checkpoint");
    let resumed_sel = resumed.selection.as_ref().expect("selection inherited");
    assert_eq!(
        resumed_sel.provenance,
        SelectionMetaProvenance::LegacyUnverified
    );
    assert_eq!(
        resumed_sel.selection_fingerprint.as_deref(),
        Some("legacy-v3-sel-fp")
    );
    assert!(resumed_sel.allowed_ids.is_none());
}

#[test]
fn resume_v2_without_selection_keeps_selection_absent() {
    let temp = tempfile::tempdir().unwrap();
    let dataset = tiny_training_dataset(&temp, "sel-v2-absent", 2);
    let mut config = tiny_training_config(1);
    let identity =
        TrainingIdentity::from_canonical_content(&dataset, b"sel-v2-recon", &config).unwrap();

    let captured = Rc::new(RefCell::new(None));
    let sink = Rc::clone(&captured);
    train_splats(
        &dataset,
        &config,
        TrainingOptions::new()
            .with_identity(identity.clone())
            .with_checkpoint_policy(TrainingCheckpointPolicy { every: Some(1) })
            .with_checkpoint_sink(move |ready| {
                *sink.borrow_mut() = Some(ready.checkpoint.clone());
                Ok(())
            }),
    )
    .unwrap();
    let first = captured.borrow().clone().expect("checkpoint");
    // Downgrade on-disk bytes to v2 layout (no selection field).
    let view = first.splats.as_view();
    let v2_path = temp.path().join("v2.rgscp");
    fs::write(
        &v2_path,
        encode_unchecked(&SerializedTrainingCheckpointV2 {
            version: TRAINING_CHECKPOINT_VERSION_V2,
            identity: first.identity.clone(),
            completed_iterations: first.completed_iterations,
            latest_loss: first.latest_loss,
            splats: SerializedHostSplats {
                positions: view.positions.to_vec(),
                log_scales: view.log_scales.to_vec(),
                rotations: view.rotations.to_vec(),
                opacity_logits: view.opacity_logits.to_vec(),
                sh_coeffs: view.sh_coeffs.to_vec(),
                sh_degree: view.sh_degree,
            },
            optimizer: first.optimizer.clone(),
            topology: first.topology.clone(),
            frame_shuffle_seed: first.frame_shuffle_seed,
            active_sh_degree: first.active_sh_degree,
        }),
    )
    .unwrap();
    let (loaded, migration) = load_training_checkpoint_with_migration(&v2_path).unwrap();
    assert_eq!(migration, CheckpointMigration::V2SelectionMetaAbsent);
    assert!(loaded.selection.is_none());

    config.iterations = 2;
    let identity2 =
        TrainingIdentity::from_canonical_content(&dataset, b"sel-v2-recon", &config).unwrap();
    let captured2 = Rc::new(RefCell::new(None));
    let sink2 = Rc::clone(&captured2);
    train_splats(
        &dataset,
        &config,
        TrainingOptions::new()
            .with_identity(identity2)
            .with_resume_checkpoint(loaded)
            .with_checkpoint_policy(TrainingCheckpointPolicy { every: Some(2) })
            .with_checkpoint_sink(move |ready| {
                *sink2.borrow_mut() = Some(ready.checkpoint.clone());
                Ok(())
            }),
    )
    .unwrap();
    let second = captured2.borrow().clone().expect("resumed checkpoint");
    assert!(
        second.selection.is_none(),
        "v2 resume must keep selection absent rather than inventing metadata"
    );
}

#[test]
fn public_train_splats_accepts_pre_c5_gap_free_identity_then_resaves_current() {
    let temp = tempfile::tempdir().unwrap();
    // Stable IDs are non-enumerated so current != pre-C5 gap-free digests.
    let image_path = temp.path().join("legacy-frame.rgb");
    let mut pixels = Vec::with_capacity(16 * 16 * 3);
    for y in 0..16_u8 {
        for x in 0..16_u8 {
            pixels.extend_from_slice(&[x, y, x.saturating_add(y)]);
        }
    }
    fs::write(&image_path, pixels).unwrap();
    let mut dataset = TrainingDataset::new(Intrinsics::new(12.0, 12.0, 8.0, 8.0, 16, 16));
    for &stable_id in &[11u64, 22, 33] {
        dataset.add_pose(ScenePose::new(
            stable_id,
            image_path.clone(),
            SE3::identity(),
            0.0,
        ));
    }
    dataset.add_point([0.0, 0.0, 2.0], Some([0.25, 0.5, 0.75]));

    let first_config = tiny_training_config(4);
    let current_hash = rustscan_gs::TrainingIdentity::from_canonical_content(
        &dataset,
        b"legacy-recon",
        &first_config,
    )
    .unwrap()
    .dataset;
    let legacy_hash =
        rustscan_gs::hash_training_dataset_with_pre_c5_frame_ids(&dataset, &[0, 1, 2]).unwrap();
    assert_ne!(current_hash, legacy_hash);

    let legacy_identity = TrainingIdentity {
        dataset: legacy_hash,
        reconstruction: blake3::hash(b"legacy-recon").to_hex().to_string(),
        config: TrainingIdentity::from_canonical_content(&dataset, b"legacy-recon", &first_config)
            .unwrap()
            .config,
    };

    let captured = Rc::new(RefCell::new(None));
    let sink = Rc::clone(&captured);
    let first = train_splats(
        &dataset,
        &first_config,
        TrainingOptions::new()
            .with_identity(legacy_identity)
            .with_checkpoint_policy(TrainingCheckpointPolicy { every: Some(4) })
            .with_checkpoint_sink(move |ready| {
                *sink.borrow_mut() = Some(ready.checkpoint.clone());
                Ok(())
            }),
    )
    .expect("train with pre-C5 identity must succeed");
    assert_eq!(first.report.completed_iterations, 4);
    let legacy_checkpoint = captured
        .borrow()
        .clone()
        .expect("periodic checkpoint from first run");
    assert_eq!(legacy_checkpoint.completed_iterations, 4);

    let resume_config = tiny_training_config(6);
    let current_identity =
        TrainingIdentity::from_canonical_content(&dataset, b"legacy-recon", &resume_config)
            .unwrap();
    let captured2 = Rc::new(RefCell::new(None));
    let sink2 = Rc::clone(&captured2);
    let second = train_splats(
        &dataset,
        &resume_config,
        TrainingOptions::new()
            .with_identity(current_identity.clone())
            .with_resume_checkpoint(legacy_checkpoint)
            .with_checkpoint_policy(TrainingCheckpointPolicy { every: Some(6) })
            .with_checkpoint_sink(move |ready| {
                *sink2.borrow_mut() = Some(ready.checkpoint.clone());
                Ok(())
            }),
    )
    .expect("resume of pre-C5 gap-free checkpoint must succeed via public train_splats");
    assert_eq!(second.report.completed_iterations, 6);
    let upgraded = captured2
        .borrow()
        .clone()
        .expect("checkpoint after re-save");
    assert_eq!(upgraded.identity.dataset, current_identity.dataset);
    assert_ne!(
        upgraded.identity.dataset,
        rustscan_gs::hash_training_dataset_with_pre_c5_frame_ids(&dataset, &[0, 1, 2]).unwrap()
    );

    let third_config = tiny_training_config(7);
    let third_identity =
        TrainingIdentity::from_canonical_content(&dataset, b"legacy-recon", &third_config).unwrap();
    let third = train_splats(
        &dataset,
        &third_config,
        TrainingOptions::new()
            .with_identity(third_identity)
            .with_resume_checkpoint(upgraded),
    )
    .expect("second resume with new identity must succeed");
    assert_eq!(third.report.completed_iterations, 7);
}
