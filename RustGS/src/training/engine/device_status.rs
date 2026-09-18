//! Shared device-resident training status (sticky overflow / non-finite).
//!
//! Buffer layout is exactly eight contiguous `u32` words so Rust and WGSL agree
//! without struct-padding surprises. See `update_training_status.wgsl`.

use burn::prelude::*;
use burn::tensor::Int;

use crate::TrainingError;

#[allow(dead_code)]
const UPDATE_TRAINING_STATUS_SHADER: &str = include_str!("../shaders/update_training_status.wgsl");

pub(crate) const STATUS_WORD_COUNT: usize = 8;
pub(crate) const STATUS_FORWARD_OVERFLOW: u32 = 1 << 0;
pub(crate) const STATUS_NON_FINITE_LOSS: u32 = 1 << 1;
pub(crate) const STATUS_FIRST_ITERATION_UNSET: u32 = u32::MAX;

const WORD_FLAGS: usize = 0;
const WORD_FIRST_ITERATION: usize = 1;
const WORD_REQUESTED: usize = 2;
const WORD_CAPACITY: usize = 3;
const WORD_COMMITTED_STEPS: usize = 4;
const WORD_MUTATION_GATE: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TrainingStatusSnapshot {
    pub flags: u32,
    pub first_invalid_iteration: u32,
    pub requested_intersections: u32,
    pub intersection_capacity: u32,
    pub committed_optimizer_steps: u32,
    pub mutation_gate: u32,
}

impl Default for TrainingStatusSnapshot {
    fn default() -> Self {
        Self {
            flags: 0,
            first_invalid_iteration: STATUS_FIRST_ITERATION_UNSET,
            requested_intersections: 0,
            intersection_capacity: 0,
            committed_optimizer_steps: 0,
            mutation_gate: 0,
        }
    }
}

impl TrainingStatusSnapshot {
    pub(crate) fn is_healthy(&self) -> bool {
        self.flags == 0
    }

    pub(crate) fn has_forward_overflow(&self) -> bool {
        self.flags & STATUS_FORWARD_OVERFLOW != 0
    }

    pub(crate) fn has_non_finite_loss(&self) -> bool {
        self.flags & STATUS_NON_FINITE_LOSS != 0
    }

    /// Prefer capacity errors when both sticky flags are set so callers can
    /// distinguish the two failure modes with dedicated variants.
    pub(crate) fn to_error(self) -> Option<TrainingError> {
        if self.has_forward_overflow() {
            return Some(TrainingError::ForwardCapacityExceeded {
                logical_intersections: self.requested_intersections,
                capacity: self.intersection_capacity,
                first_iteration: self.first_invalid_iteration,
            });
        }
        if self.has_non_finite_loss() {
            return Some(TrainingError::NonFiniteLoss {
                first_iteration: self.first_invalid_iteration,
            });
        }
        None
    }
}

pub(crate) fn encode_training_status(snapshot: TrainingStatusSnapshot) -> [u32; STATUS_WORD_COUNT] {
    let mut words = [0u32; STATUS_WORD_COUNT];
    words[WORD_FLAGS] = snapshot.flags;
    words[WORD_FIRST_ITERATION] = snapshot.first_invalid_iteration;
    words[WORD_REQUESTED] = snapshot.requested_intersections;
    words[WORD_CAPACITY] = snapshot.intersection_capacity;
    words[WORD_COMMITTED_STEPS] = snapshot.committed_optimizer_steps;
    words[WORD_MUTATION_GATE] = snapshot.mutation_gate;
    words
}

pub(crate) fn decode_training_status(words: &[u32; STATUS_WORD_COUNT]) -> TrainingStatusSnapshot {
    TrainingStatusSnapshot {
        flags: words[WORD_FLAGS],
        first_invalid_iteration: words[WORD_FIRST_ITERATION],
        requested_intersections: words[WORD_REQUESTED],
        intersection_capacity: words[WORD_CAPACITY],
        committed_optimizer_steps: words[WORD_COMMITTED_STEPS],
        mutation_gate: words[WORD_MUTATION_GATE],
    }
}

/// Merge a step overflow into sticky status. Exact-full capacity is not overflow.
pub(crate) fn note_forward_overflow(
    mut snapshot: TrainingStatusSnapshot,
    step_overflowed: bool,
    iteration: u32,
    requested_intersections: u32,
    intersection_capacity: u32,
) -> TrainingStatusSnapshot {
    if !step_overflowed {
        return snapshot;
    }
    if snapshot.has_forward_overflow() {
        return snapshot;
    }
    snapshot.flags |= STATUS_FORWARD_OVERFLOW;
    if snapshot.first_invalid_iteration == STATUS_FIRST_ITERATION_UNSET {
        snapshot.first_invalid_iteration = iteration.max(1);
        snapshot.requested_intersections = requested_intersections;
        snapshot.intersection_capacity = intersection_capacity;
    }
    snapshot
}

pub(crate) fn note_non_finite_loss(
    mut snapshot: TrainingStatusSnapshot,
    iteration: u32,
) -> TrainingStatusSnapshot {
    if snapshot.has_non_finite_loss() {
        return snapshot;
    }
    snapshot.flags |= STATUS_NON_FINITE_LOSS;
    if snapshot.first_invalid_iteration == STATUS_FIRST_ITERATION_UNSET {
        snapshot.first_invalid_iteration = iteration.max(1);
    }
    snapshot
}

pub(crate) struct DeviceTrainingStatus<B: Backend> {
    buffer: Tensor<B, 1, Int>,
    host: TrainingStatusSnapshot,
    /// Device-side sticky until Task 1.3 marks `STATUS_NON_FINITE_LOSS` from a GPU kernel.
    non_finite_steps: Option<Tensor<B, 1, Int>>,
}

impl<B: Backend> DeviceTrainingStatus<B> {
    pub(crate) fn new(device: &B::Device, restored_step: u32) -> Self {
        let mut host = TrainingStatusSnapshot::default();
        host.committed_optimizer_steps = restored_step;
        let words = encode_training_status(host);
        let ints: [i32; STATUS_WORD_COUNT] = words.map(|word| word as i32);
        Self {
            buffer: Tensor::<B, 1, Int>::from_ints(ints.as_slice(), device),
            host,
            non_finite_steps: None,
        }
    }

    pub(crate) fn host_snapshot(&self) -> TrainingStatusSnapshot {
        self.host
    }

    pub(crate) fn buffer(&self) -> &Tensor<B, 1, Int> {
        &self.buffer
    }

    pub(crate) fn set_host_snapshot(&mut self, snapshot: TrainingStatusSnapshot) {
        self.host = snapshot;
        self.sync_host_to_device();
    }

    pub(crate) fn adopt_device_snapshot(&mut self, snapshot: TrainingStatusSnapshot) {
        self.host = snapshot;
    }

    #[allow(dead_code)]
    pub(crate) fn note_forward_overflow_host(
        &mut self,
        step_overflowed: bool,
        iteration: u32,
        requested_intersections: u32,
        intersection_capacity: u32,
    ) {
        self.host = note_forward_overflow(
            self.host,
            step_overflowed,
            iteration,
            requested_intersections,
            intersection_capacity,
        );
        self.sync_host_to_device();
    }

    pub(crate) fn note_non_finite_loss_device(&mut self, bad: Tensor<B, 1, Int>) {
        self.non_finite_steps = Some(match self.non_finite_steps.clone() {
            Some(accumulated) => accumulated + bad,
            None => bad,
        });
    }

    pub(crate) async fn non_finite_loss_seen(&self) -> Result<bool, TrainingError> {
        let Some(flag) = self.non_finite_steps.clone() else {
            return Ok(self.host.has_non_finite_loss());
        };
        let value = flag.into_scalar_async().await.map_err(|err| {
            TrainingError::TrainingFailed(format!("failed to read loss finite flag: {err}"))
        })?;
        let value_i32 = burn::tensor::cast::ToElement::to_i32(&value);
        Ok(value_i32 != 0 || self.host.has_non_finite_loss())
    }

    fn sync_host_to_device(&mut self) {
        let device = self.buffer.device();
        let words = encode_training_status(self.host);
        let ints: [i32; STATUS_WORD_COUNT] = words.map(|word| word as i32);
        self.buffer = Tensor::<B, 1, Int>::from_ints(ints.as_slice(), &device);
    }

    pub(crate) async fn read(&self) -> Result<TrainingStatusSnapshot, TrainingError> {
        let data = self.buffer.clone().into_data_async().await.map_err(|err| {
            TrainingError::TrainingFailed(format!("failed to read training status: {err}"))
        })?;
        let values = data.to_vec::<i32>().map_err(|err| {
            TrainingError::TrainingFailed(format!("failed to decode training status: {err}"))
        })?;
        if values.len() != STATUS_WORD_COUNT {
            return Err(TrainingError::TrainingFailed(format!(
                "training status buffer length {}, expected {STATUS_WORD_COUNT}",
                values.len()
            )));
        }
        let mut words = [0u32; STATUS_WORD_COUNT];
        for (idx, value) in values.into_iter().enumerate() {
            words[idx] = value as u32;
        }
        let mut snapshot = decode_training_status(&words);
        if self.non_finite_loss_seen().await? {
            snapshot = note_non_finite_loss(snapshot, snapshot.first_invalid_iteration.max(1));
        }
        Ok(snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn healthy_status_decodes_without_error() {
        let snapshot = decode_training_status(&encode_training_status(TrainingStatusSnapshot {
            committed_optimizer_steps: 7,
            ..TrainingStatusSnapshot::default()
        }));
        assert!(snapshot.is_healthy());
        assert_eq!(snapshot.committed_optimizer_steps, 7);
        assert_eq!(
            snapshot.first_invalid_iteration,
            STATUS_FIRST_ITERATION_UNSET
        );
        assert!(snapshot.to_error().is_none());
    }

    #[test]
    fn overflow_status_maps_to_capacity_error() {
        let snapshot =
            note_forward_overflow(TrainingStatusSnapshot::default(), true, 2, 9_000, 8_000);
        assert!(snapshot.has_forward_overflow());
        assert!(!snapshot.has_non_finite_loss());
        assert_eq!(snapshot.first_invalid_iteration, 2);
        assert_eq!(snapshot.requested_intersections, 9_000);
        assert_eq!(snapshot.intersection_capacity, 8_000);
        assert!(matches!(
            snapshot.to_error(),
            Some(TrainingError::ForwardCapacityExceeded {
                first_iteration: 2,
                logical_intersections: 9_000,
                capacity: 8_000,
            })
        ));
    }

    #[test]
    fn non_finite_status_maps_to_non_finite_error() {
        let snapshot = note_non_finite_loss(TrainingStatusSnapshot::default(), 5);
        assert!(snapshot.has_non_finite_loss());
        assert!(!snapshot.has_forward_overflow());
        assert_eq!(snapshot.first_invalid_iteration, 5);
        assert!(matches!(
            snapshot.to_error(),
            Some(TrainingError::NonFiniteLoss { first_iteration: 5 })
        ));
    }

    #[test]
    fn dual_flags_keep_both_bits_and_prefer_capacity_error() {
        let mut snapshot =
            note_forward_overflow(TrainingStatusSnapshot::default(), true, 2, 9_000, 8_000);
        snapshot = note_non_finite_loss(snapshot, 9);
        assert!(snapshot.has_forward_overflow());
        assert!(snapshot.has_non_finite_loss());
        assert_eq!(snapshot.first_invalid_iteration, 2);
        assert!(matches!(
            snapshot.to_error(),
            Some(TrainingError::ForwardCapacityExceeded { .. })
        ));
    }

    #[test]
    fn first_invalid_iteration_is_sticky_across_later_anomalies() {
        let mut snapshot =
            note_forward_overflow(TrainingStatusSnapshot::default(), true, 3, 5_000, 4_000);
        snapshot = note_forward_overflow(snapshot, true, 4, 6_000, 4_000);
        snapshot = note_non_finite_loss(snapshot, 8);
        assert_eq!(snapshot.first_invalid_iteration, 3);
        assert_eq!(snapshot.requested_intersections, 5_000);
        assert_eq!(snapshot.intersection_capacity, 4_000);
        let words = encode_training_status(snapshot);
        let decoded = decode_training_status(&words);
        assert_eq!(decoded, snapshot);
    }
}
