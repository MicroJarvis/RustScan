use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use super::ImportedFrame;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyframeSelectionConfig {
    pub target_per_second: f64,
    pub max_gap_us: i64,
    pub duplicate_hamming_threshold: u32,
}

impl Default for KeyframeSelectionConfig {
    fn default() -> Self {
        Self {
            target_per_second: 3.0,
            max_gap_us: 1_000_000,
            duplicate_hamming_threshold: 6,
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum KeyframeSelectionError {
    #[error("keyframe selection requires at least one frame")]
    EmptyFrames,
    #[error("keyframe selection target_per_second must be finite and greater than zero")]
    InvalidTargetPerSecond,
    #[error("keyframe selection max_gap_us must be greater than zero")]
    InvalidMaximumGap,
    #[error("keyframe selection duplicate_hamming_threshold must not exceed 64")]
    InvalidDuplicateHammingThreshold,
    #[error("frame {frame_id} has no presentation timestamp")]
    MissingPresentationTimestamp { frame_id: u32 },
    #[error("frame {frame_id} has a negative presentation timestamp ({presentation_time_us})")]
    NegativePresentationTimestamp {
        frame_id: u32,
        presentation_time_us: i64,
    },
    #[error("frame {frame_id} has a non-finite sharpness value")]
    NonFiniteSharpness { frame_id: u32 },
    #[error("frame ID {frame_id} is duplicated")]
    DuplicateFrameId { frame_id: u32 },
    #[error(
        "frame {frame_id} presentation timestamp ({presentation_time_us}) is not strictly later than the previous frame ({previous_presentation_time_us})"
    )]
    NonMonotonicPresentationTime {
        frame_id: u32,
        presentation_time_us: i64,
        previous_presentation_time_us: i64,
    },
    #[error("keyframe selection target rate cannot form a timeline window")]
    WindowIndexOverflow,
    #[error(
        "cannot meet the maximum keyframe gap of {max_gap_us}us between frames {from_frame_id} and {to_frame_id}"
    )]
    UnsatisfiableGap {
        from_frame_id: u32,
        to_frame_id: u32,
        max_gap_us: i64,
    },
}

struct TimelineFrame<'a> {
    frame: &'a ImportedFrame,
    presentation_time_us: i64,
}

/// Selects deterministic video keyframes from chronologically ordered frame metadata.
///
/// The first and last input frames are always retained. Interior frames compete within
/// fixed target-rate windows. Frames whose perceptual-hash Hamming distance is at or
/// below the configured threshold from an already selected frame are suppressed there;
/// the later gap-fill pass can still use them when they are required for coverage.
pub fn select_keyframes(
    frames: &[ImportedFrame],
    config: KeyframeSelectionConfig,
) -> Result<Vec<u32>, KeyframeSelectionError> {
    validate_config(config)?;
    let timeline = validate_frames(frames)?;

    let mut selected = BTreeSet::new();
    selected.insert(0);
    selected.insert(timeline.len() - 1);

    let windows = target_rate_windows(&timeline, config.target_per_second)?;
    for window in windows.values() {
        let candidate = best_candidate(
            window
                .iter()
                .copied()
                .filter(|index| !is_near_duplicate(*index, &selected, &timeline, config)),
            &selected,
            &timeline,
        );
        if let Some(candidate) = candidate {
            selected.insert(candidate);
        }
    }

    fill_overlong_gaps(&mut selected, &timeline, config)?;

    let mut ids = selected
        .into_iter()
        .map(|index| timeline[index].frame.id)
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    Ok(ids)
}

fn validate_config(config: KeyframeSelectionConfig) -> Result<(), KeyframeSelectionError> {
    if !config.target_per_second.is_finite() || config.target_per_second <= 0.0 {
        return Err(KeyframeSelectionError::InvalidTargetPerSecond);
    }
    if config.max_gap_us <= 0 {
        return Err(KeyframeSelectionError::InvalidMaximumGap);
    }
    if config.duplicate_hamming_threshold > 64 {
        return Err(KeyframeSelectionError::InvalidDuplicateHammingThreshold);
    }
    Ok(())
}

fn validate_frames(
    frames: &[ImportedFrame],
) -> Result<Vec<TimelineFrame<'_>>, KeyframeSelectionError> {
    if frames.is_empty() {
        return Err(KeyframeSelectionError::EmptyFrames);
    }

    let mut ids = BTreeSet::new();
    let mut previous_presentation_time_us = None;
    frames
        .iter()
        .map(|frame| {
            if !ids.insert(frame.id) {
                return Err(KeyframeSelectionError::DuplicateFrameId { frame_id: frame.id });
            }
            if !frame.sharpness.is_finite() {
                return Err(KeyframeSelectionError::NonFiniteSharpness { frame_id: frame.id });
            }
            let presentation_time_us = frame.presentation_time_us.ok_or(
                KeyframeSelectionError::MissingPresentationTimestamp { frame_id: frame.id },
            )?;
            if presentation_time_us < 0 {
                return Err(KeyframeSelectionError::NegativePresentationTimestamp {
                    frame_id: frame.id,
                    presentation_time_us,
                });
            }
            if let Some(previous_presentation_time_us) = previous_presentation_time_us {
                if presentation_time_us <= previous_presentation_time_us {
                    return Err(KeyframeSelectionError::NonMonotonicPresentationTime {
                        frame_id: frame.id,
                        presentation_time_us,
                        previous_presentation_time_us,
                    });
                }
            }
            previous_presentation_time_us = Some(presentation_time_us);
            Ok(TimelineFrame {
                frame,
                presentation_time_us,
            })
        })
        .collect()
}

fn target_rate_windows(
    timeline: &[TimelineFrame<'_>],
    target_per_second: f64,
) -> Result<BTreeMap<u64, Vec<usize>>, KeyframeSelectionError> {
    let first_presentation_time_us = timeline[0].presentation_time_us;
    let mut windows = BTreeMap::<u64, Vec<usize>>::new();
    for (index, frame) in timeline.iter().enumerate() {
        let elapsed_us = frame.presentation_time_us - first_presentation_time_us;
        let window = (elapsed_us as f64 * target_per_second / 1_000_000.0).floor();
        if !window.is_finite() || window < 0.0 || window > u64::MAX as f64 {
            return Err(KeyframeSelectionError::WindowIndexOverflow);
        }
        windows.entry(window as u64).or_default().push(index);
    }
    Ok(windows)
}

fn fill_overlong_gaps(
    selected: &mut BTreeSet<usize>,
    timeline: &[TimelineFrame<'_>],
    config: KeyframeSelectionConfig,
) -> Result<(), KeyframeSelectionError> {
    loop {
        let selected_indices = selected.iter().copied().collect::<Vec<_>>();
        let mut inserted = false;
        for pair in selected_indices.windows(2) {
            let previous = pair[0];
            let next = pair[1];
            let gap_us =
                timeline[next].presentation_time_us - timeline[previous].presentation_time_us;
            if gap_us <= config.max_gap_us {
                continue;
            }
            let deadline_us = timeline[previous]
                .presentation_time_us
                .checked_add(config.max_gap_us)
                .unwrap_or(i64::MAX);
            let candidate = best_candidate(
                (previous + 1..next)
                    .filter(|index| timeline[*index].presentation_time_us <= deadline_us),
                selected,
                timeline,
            )
            .ok_or(KeyframeSelectionError::UnsatisfiableGap {
                from_frame_id: timeline[previous].frame.id,
                to_frame_id: timeline[next].frame.id,
                max_gap_us: config.max_gap_us,
            })?;
            selected.insert(candidate);
            inserted = true;
            break;
        }
        if !inserted {
            return Ok(());
        }
    }
}

fn is_near_duplicate(
    candidate: usize,
    selected: &BTreeSet<usize>,
    timeline: &[TimelineFrame<'_>],
    config: KeyframeSelectionConfig,
) -> bool {
    selected.iter().any(|selected_index| {
        hamming_distance(
            timeline[candidate].frame.perceptual_hash,
            timeline[*selected_index].frame.perceptual_hash,
        ) <= config.duplicate_hamming_threshold
    })
}

fn best_candidate(
    candidates: impl Iterator<Item = usize>,
    selected: &BTreeSet<usize>,
    timeline: &[TimelineFrame<'_>],
) -> Option<usize> {
    candidates.max_by(|left, right| compare_candidates(*left, *right, selected, timeline))
}

fn compare_candidates(
    left: usize,
    right: usize,
    selected: &BTreeSet<usize>,
    timeline: &[TimelineFrame<'_>],
) -> Ordering {
    let left_frame = timeline[left].frame;
    let right_frame = timeline[right].frame;
    left_frame
        .sharpness
        .total_cmp(&right_frame.sharpness)
        .then_with(|| {
            // The selection contract uses reverse duplicate distance: closer ties win.
            Reverse(minimum_duplicate_distance(left, selected, timeline)).cmp(&Reverse(
                minimum_duplicate_distance(right, selected, timeline),
            ))
        })
        .then_with(|| Reverse(left_frame.id).cmp(&Reverse(right_frame.id)))
}

fn minimum_duplicate_distance(
    candidate: usize,
    selected: &BTreeSet<usize>,
    timeline: &[TimelineFrame<'_>],
) -> u32 {
    selected
        .iter()
        .map(|selected_index| {
            hamming_distance(
                timeline[candidate].frame.perceptual_hash,
                timeline[*selected_index].frame.perceptual_hash,
            )
        })
        .min()
        .unwrap_or(u32::MAX)
}

fn hamming_distance(left: u64, right: u64) -> u32 {
    (left ^ right).count_ones()
}
