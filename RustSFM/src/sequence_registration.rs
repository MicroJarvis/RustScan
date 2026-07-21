use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BinaryHeap;
use std::collections::{BTreeSet, HashSet};
use std::error::Error;
use std::fmt;
use std::path::PathBuf;

pub const MAX_SEQUENCE_PLAN_FRAMES: usize = 1_000_000;
pub const MAX_SEQUENCE_NEIGHBORS: usize = 1_024;
pub const MAX_TOTAL_SUPPORT_ENTRIES: usize = 32_000_000;
pub const MAX_TIMESTAMP_PLATEAU: usize = MAX_SEQUENCE_NEIGHBORS;
pub const MAX_DYNAMIC_SUPPORT_CANDIDATES: usize = 65_536;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SequenceFrame {
    pub id: u32,
    pub image_path: PathBuf,
    pub timestamp_us: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceRegistrationConfig {
    pub narrow_neighbors_each_side: usize,
    pub wide_neighbors_each_side: usize,
    pub min_inliers: usize,
    pub min_inlier_ratio: f64,
    pub max_reprojection_error: f64,
    pub use_gpu_pnp: bool,
}

impl Default for SequenceRegistrationConfig {
    fn default() -> Self {
        Self {
            narrow_neighbors_each_side: 2,
            wide_neighbors_each_side: 4,
            min_inliers: 24,
            min_inlier_ratio: 0.20,
            max_reprojection_error: 4.0,
            use_gpu_pnp: true,
        }
    }
}

impl SequenceRegistrationConfig {
    pub fn validate(&self) -> Result<(), SequenceRegistrationError> {
        if !self.min_inlier_ratio.is_finite() || !(0.0..=1.0).contains(&self.min_inlier_ratio) {
            return Err(SequenceRegistrationError::InvalidConfigMetric {
                field: "min_inlier_ratio",
            });
        }
        if !self.max_reprojection_error.is_finite() || self.max_reprojection_error < 0.0 {
            return Err(SequenceRegistrationError::InvalidConfigMetric {
                field: "max_reprojection_error",
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegistrationRound {
    Narrow,
    Wide,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameRegistrationStatus {
    Keyframe,
    Registered,
    Unresolved,
    Excluded,
}

impl FrameRegistrationStatus {
    pub fn is_registered(self) -> bool {
        matches!(self, Self::Keyframe | Self::Registered)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameRegistrationDiagnostic {
    pub frame_id: u32,
    pub status: FrameRegistrationStatus,
    pub attempts: usize,
    pub support_frame_ids: Vec<u32>,
    pub inlier_count: usize,
    pub inlier_ratio: f64,
    pub mean_reprojection_error: Option<f64>,
    pub message: Option<String>,
}

impl FrameRegistrationDiagnostic {
    pub fn new(frame_id: u32, status: FrameRegistrationStatus) -> Self {
        Self {
            frame_id,
            status,
            attempts: 0,
            support_frame_ids: Vec::new(),
            inlier_count: 0,
            inlier_ratio: 0.0,
            mean_reprojection_error: None,
            message: None,
        }
    }

    pub fn record_attempt(
        &mut self,
        status: FrameRegistrationStatus,
        support_frame_ids: Vec<u32>,
        inlier_count: usize,
        inlier_ratio: f64,
        mean_reprojection_error: Option<f64>,
        message: Option<String>,
    ) {
        self.status = status;
        self.attempts = self.attempts.saturating_add(1);
        self.support_frame_ids = support_frame_ids;
        self.inlier_count = inlier_count;
        self.inlier_ratio = inlier_ratio;
        self.mean_reprojection_error = mean_reprojection_error;
        self.message = message;
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceRegistrationResult {
    pub imported_frames: usize,
    pub registered_frames: usize,
    pub frame_ids: Vec<u32>,
    pub diagnostics: Vec<FrameRegistrationDiagnostic>,
    pub sparse_model: PathBuf,
}

impl SequenceRegistrationResult {
    pub fn has_complete_coverage(&self) -> bool {
        self.validate_complete_coverage().is_ok()
    }

    pub fn validate_complete_coverage(&self) -> Result<(), SequenceRegistrationError> {
        if self.imported_frames == 0 {
            return Err(SequenceRegistrationError::EmptySequence);
        }
        if self.diagnostics.len() != self.imported_frames {
            return Err(SequenceRegistrationError::DiagnosticCountMismatch {
                imported_frames: self.imported_frames,
                diagnostic_count: self.diagnostics.len(),
            });
        }
        if self.imported_frames as u128 > u32::MAX as u128 + 1 {
            return Err(SequenceRegistrationError::FrameCountExceedsFrameIdRange {
                imported_frames: self.imported_frames,
            });
        }
        if self.frame_ids.len() != self.imported_frames {
            return Err(SequenceRegistrationError::InvalidFrameIds {
                imported_frames: self.imported_frames,
                frame_id_count: self.frame_ids.len(),
                duplicate_frame_ids: Vec::new(),
            });
        }

        let mut expected_frame_ids = BTreeSet::new();
        let mut duplicate_expected_frame_ids = BTreeSet::new();
        for frame_id in self.frame_ids.iter().copied() {
            if !expected_frame_ids.insert(frame_id) {
                duplicate_expected_frame_ids.insert(frame_id);
            }
        }
        if !duplicate_expected_frame_ids.is_empty() {
            return Err(SequenceRegistrationError::InvalidFrameIds {
                imported_frames: self.imported_frames,
                frame_id_count: self.frame_ids.len(),
                duplicate_frame_ids: duplicate_expected_frame_ids.into_iter().collect(),
            });
        }

        let mut observed_frame_ids = BTreeSet::new();
        let mut duplicate_frame_ids = BTreeSet::new();
        for diagnostic in &self.diagnostics {
            if !observed_frame_ids.insert(diagnostic.frame_id) {
                duplicate_frame_ids.insert(diagnostic.frame_id);
            }
        }
        let missing_frame_ids: Vec<_> = expected_frame_ids
            .difference(&observed_frame_ids)
            .copied()
            .collect();
        let duplicate_frame_ids: Vec<_> = duplicate_frame_ids.into_iter().collect();
        let unexpected_frame_ids: Vec<_> = observed_frame_ids
            .difference(&expected_frame_ids)
            .copied()
            .collect();
        if !missing_frame_ids.is_empty()
            || !duplicate_frame_ids.is_empty()
            || !unexpected_frame_ids.is_empty()
        {
            return Err(SequenceRegistrationError::InvalidDiagnostics {
                imported_frames: self.imported_frames,
                diagnostic_count: self.diagnostics.len(),
                missing_frame_ids,
                duplicate_frame_ids,
                unexpected_frame_ids,
            });
        }

        for diagnostic in &self.diagnostics {
            if !diagnostic.inlier_ratio.is_finite()
                || !(0.0..=1.0).contains(&diagnostic.inlier_ratio)
            {
                return Err(SequenceRegistrationError::InvalidDiagnosticMetric {
                    frame_id: diagnostic.frame_id,
                    field: "inlier_ratio",
                });
            }
            if diagnostic
                .mean_reprojection_error
                .is_some_and(|error| !error.is_finite() || error < 0.0)
            {
                return Err(SequenceRegistrationError::InvalidDiagnosticMetric {
                    frame_id: diagnostic.frame_id,
                    field: "mean_reprojection_error",
                });
            }
        }

        let unresolved_frame_ids = self
            .diagnostics
            .iter()
            .filter(|diagnostic| !diagnostic.status.is_registered())
            .map(|diagnostic| diagnostic.frame_id)
            .collect::<Vec<_>>();
        let diagnostic_registered_frames = self.diagnostics.len() - unresolved_frame_ids.len();
        if self.registered_frames != diagnostic_registered_frames {
            return Err(SequenceRegistrationError::RegistrationStatusCountMismatch {
                registered_frames: self.registered_frames,
                diagnostic_registered_frames,
                unresolved_frame_ids,
            });
        }
        if self.imported_frames != self.registered_frames || !unresolved_frame_ids.is_empty() {
            return Err(SequenceRegistrationError::IncompleteCoverage {
                imported_frames: self.imported_frames,
                registered_frames: self.registered_frames,
                unresolved_frame_ids,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SequenceRegistrationError {
    EmptySequence,
    EmptyKeyframes,
    TooManyKeyframes {
        frame_count: usize,
        keyframe_count: usize,
    },
    DuplicateKeyframe {
        frame: usize,
    },
    KeyframeOutOfRange {
        frame: usize,
        frame_count: usize,
    },
    UnsortedKeyframes {
        previous: usize,
        current: usize,
    },
    FrameCountExceedsFrameIdRange {
        imported_frames: usize,
    },
    SequencePlanTooLarge {
        frame_count: usize,
        max_frame_count: usize,
    },
    SequenceNeighborLimitExceeded {
        round: RegistrationRound,
        requested: usize,
        max_neighbors: usize,
    },
    SequenceSupportBudgetExceeded {
        frame_count: usize,
        estimated_support_entries: u128,
        max_support_entries: usize,
    },
    TimestampPlateauTooLarge {
        timestamp_us: i64,
        plateau_size: usize,
        max_plateau_size: usize,
    },
    DynamicSupportLimitExceeded {
        candidate_count: usize,
        max_candidates: usize,
    },
    DynamicSupportNotSortedUnique,
    DynamicSupportFrameOutOfRange {
        frame: usize,
        frame_count: usize,
    },
    InvalidFrameIds {
        imported_frames: usize,
        frame_id_count: usize,
        duplicate_frame_ids: Vec<u32>,
    },
    TimestampCountMismatch {
        frame_count: usize,
        timestamp_count: usize,
    },
    UnsortedTimestamps {
        previous_frame: usize,
        current_frame: usize,
    },
    DiagnosticCountMismatch {
        imported_frames: usize,
        diagnostic_count: usize,
    },
    InvalidConfigMetric {
        field: &'static str,
    },
    InvalidDiagnosticMetric {
        frame_id: u32,
        field: &'static str,
    },
    InvalidDiagnostics {
        imported_frames: usize,
        diagnostic_count: usize,
        missing_frame_ids: Vec<u32>,
        duplicate_frame_ids: Vec<u32>,
        unexpected_frame_ids: Vec<u32>,
    },
    RegistrationStatusCountMismatch {
        registered_frames: usize,
        diagnostic_registered_frames: usize,
        unresolved_frame_ids: Vec<u32>,
    },
    IncompleteCoverage {
        imported_frames: usize,
        registered_frames: usize,
        unresolved_frame_ids: Vec<u32>,
    },
}

impl fmt::Display for SequenceRegistrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySequence => formatter.write_str("sequence must contain at least one frame"),
            Self::EmptyKeyframes => {
                formatter.write_str("sequence registration requires at least one keyframe")
            }
            Self::TooManyKeyframes {
                frame_count,
                keyframe_count,
            } => write!(
                formatter,
                "{keyframe_count} keyframes exceed sequence length {frame_count}"
            ),
            Self::DuplicateKeyframe { frame } => {
                write!(formatter, "duplicate keyframe index {frame}")
            }
            Self::KeyframeOutOfRange { frame, frame_count } => write!(
                formatter,
                "keyframe index {frame} is out of range for {frame_count} frames"
            ),
            Self::UnsortedKeyframes { previous, current } => write!(
                formatter,
                "keyframe indices must be sorted: {current} follows {previous}"
            ),
            Self::FrameCountExceedsFrameIdRange { imported_frames } => write!(
                formatter,
                "{imported_frames} imported frames cannot be represented by u32 frame IDs"
            ),
            Self::SequencePlanTooLarge {
                frame_count,
                max_frame_count,
            } => write!(
                formatter,
                "sequence plan frame count {frame_count} exceeds supported maximum {max_frame_count}"
            ),
            Self::SequenceNeighborLimitExceeded {
                round,
                requested,
                max_neighbors,
            } => write!(
                formatter,
                "{round:?} registration neighbor count {requested} exceeds supported maximum {max_neighbors}"
            ),
            Self::SequenceSupportBudgetExceeded {
                frame_count,
                estimated_support_entries,
                max_support_entries,
            } => write!(
                formatter,
                "sequence plan for {frame_count} frames may cache {estimated_support_entries} support entries, exceeding maximum {max_support_entries}"
            ),
            Self::TimestampPlateauTooLarge {
                timestamp_us,
                plateau_size,
                max_plateau_size,
            } => write!(
                formatter,
                "timestamp {timestamp_us} plateau contains {plateau_size} frames, exceeding maximum {max_plateau_size}"
            ),
            Self::DynamicSupportLimitExceeded {
                candidate_count,
                max_candidates,
            } => write!(
                formatter,
                "dynamic support contains {candidate_count} candidates, exceeding maximum {max_candidates}"
            ),
            Self::DynamicSupportNotSortedUnique => {
                formatter.write_str("dynamic support must be sorted and unique")
            }
            Self::DynamicSupportFrameOutOfRange { frame, frame_count } => write!(
                formatter,
                "dynamic support frame {frame} is out of range for {frame_count} frames"
            ),
            Self::InvalidFrameIds {
                imported_frames,
                frame_id_count,
                duplicate_frame_ids,
            } => write!(
                formatter,
                "invalid expected frame IDs: expected {imported_frames}, found {frame_id_count}; duplicate frame IDs {duplicate_frame_ids:?}"
            ),
            Self::TimestampCountMismatch {
                frame_count,
                timestamp_count,
            } => write!(
                formatter,
                "timestamp count {timestamp_count} does not match frame count {frame_count}"
            ),
            Self::UnsortedTimestamps {
                previous_frame,
                current_frame,
            } => write!(
                formatter,
                "frame {current_frame} timestamp precedes frame {previous_frame}"
            ),
            Self::DiagnosticCountMismatch {
                imported_frames,
                diagnostic_count,
            } => write!(
                formatter,
                "diagnostic count {diagnostic_count} does not match imported frame count {imported_frames}"
            ),
            Self::InvalidConfigMetric { field } => {
                write!(formatter, "sequence registration config metric {field} is invalid")
            }
            Self::InvalidDiagnosticMetric { frame_id, field } => write!(
                formatter,
                "frame {frame_id} registration diagnostic metric {field} is invalid"
            ),
            Self::InvalidDiagnostics {
                imported_frames,
                diagnostic_count,
                missing_frame_ids,
                duplicate_frame_ids,
                unexpected_frame_ids,
            } => write!(
                formatter,
                "invalid sequence diagnostics: expected {imported_frames} records, found {diagnostic_count}; missing frame IDs {missing_frame_ids:?}; duplicate frame IDs {duplicate_frame_ids:?}; unexpected frame IDs {unexpected_frame_ids:?}"
            ),
            Self::RegistrationStatusCountMismatch {
                registered_frames,
                diagnostic_registered_frames,
                unresolved_frame_ids,
            } => {
                write!(
                    formatter,
                    "registered frame count {registered_frames} disagrees with {diagnostic_registered_frames} registered diagnostics"
                )?;
                if !unresolved_frame_ids.is_empty() {
                    formatter.write_str("; unresolved frame")?;
                    if unresolved_frame_ids.len() != 1 {
                        formatter.write_str("s")?;
                    }
                    for (index, frame_id) in unresolved_frame_ids.iter().enumerate() {
                        if index == 0 {
                            write!(formatter, " {frame_id}")?;
                        } else {
                            write!(formatter, ", {frame_id}")?;
                        }
                    }
                }
                Ok(())
            }
            Self::IncompleteCoverage {
                imported_frames,
                registered_frames,
                unresolved_frame_ids,
            } => {
                write!(
                    formatter,
                    "incomplete sequence pose coverage: registered {registered_frames} of {imported_frames} imported frames"
                )?;
                if !unresolved_frame_ids.is_empty() {
                    formatter.write_str("; unresolved frame")?;
                    if unresolved_frame_ids.len() != 1 {
                        formatter.write_str("s")?;
                    }
                    for (index, frame_id) in unresolved_frame_ids.iter().enumerate() {
                        if index == 0 {
                            write!(formatter, " {frame_id}")?;
                        } else {
                            write!(formatter, ", {frame_id}")?;
                        }
                    }
                }
                Ok(())
            }
        }
    }
}

impl Error for SequenceRegistrationError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceRegistrationPlan {
    frame_count: usize,
    keyframes: Vec<usize>,
    narrow_neighbors_each_side: usize,
    wide_neighbors_each_side: usize,
    frame_ids: Vec<u32>,
    timestamps_us: Option<Vec<i64>>,
    pending: Vec<usize>,
    narrow_support: Vec<Vec<usize>>,
    wide_support: Vec<Vec<usize>>,
}

impl SequenceRegistrationPlan {
    pub fn build(
        frame_count: usize,
        keyframes: &[usize],
        narrow_neighbors_each_side: usize,
        wide_neighbors_each_side: usize,
    ) -> Result<Self, SequenceRegistrationError> {
        validate_plan_limits(
            frame_count,
            narrow_neighbors_each_side,
            wide_neighbors_each_side,
        )?;
        validate_keyframes(frame_count, keyframes)?;
        let frame_ids = (0..frame_count).map(|frame| frame as u32).collect();
        Self::build_validated(
            frame_count,
            keyframes,
            narrow_neighbors_each_side,
            wide_neighbors_each_side,
            frame_ids,
            None,
        )
    }

    pub fn build_from_frames(
        frames: &[SequenceFrame],
        keyframes: &[usize],
        narrow_neighbors_each_side: usize,
        wide_neighbors_each_side: usize,
    ) -> Result<Self, SequenceRegistrationError> {
        validate_plan_limits(
            frames.len(),
            narrow_neighbors_each_side,
            wide_neighbors_each_side,
        )?;
        validate_keyframes(frames.len(), keyframes)?;
        validate_timestamp_inputs(frames.len(), frames.iter().map(|frame| frame.timestamp_us))?;
        let frame_ids = frames.iter().map(|frame| frame.id).collect();
        let timestamps_us = frames.iter().map(|frame| frame.timestamp_us).collect();
        Self::build_validated(
            frames.len(),
            keyframes,
            narrow_neighbors_each_side,
            wide_neighbors_each_side,
            frame_ids,
            timestamps_us,
        )
    }

    fn build_validated(
        frame_count: usize,
        keyframes: &[usize],
        narrow_neighbors_each_side: usize,
        wide_neighbors_each_side: usize,
        frame_ids: Vec<u32>,
        timestamps_us: Option<Vec<i64>>,
    ) -> Result<Self, SequenceRegistrationError> {
        validate_plan_limits(
            frame_count,
            narrow_neighbors_each_side,
            wide_neighbors_each_side,
        )?;
        validate_plan_ordering(frame_count, &frame_ids, timestamps_us.as_deref())?;

        let pending = (0..frame_count)
            .filter(|frame| keyframes.binary_search(frame).is_err())
            .collect();
        let narrow_support = build_support_lists(
            frame_count,
            keyframes,
            narrow_neighbors_each_side,
            &frame_ids,
            timestamps_us.as_deref(),
        );
        let wide_support = build_support_lists(
            frame_count,
            keyframes,
            wide_neighbors_each_side,
            &frame_ids,
            timestamps_us.as_deref(),
        );

        Ok(Self {
            frame_count,
            keyframes: keyframes.to_vec(),
            narrow_neighbors_each_side,
            wide_neighbors_each_side,
            frame_ids,
            timestamps_us,
            pending,
            narrow_support,
            wide_support,
        })
    }

    pub fn pending_frames(&self) -> &[usize] {
        &self.pending
    }

    pub fn attempts_for(&self, frame: usize, round: RegistrationRound) -> &[usize] {
        let support = match round {
            RegistrationRound::Narrow => &self.narrow_support,
            RegistrationRound::Wide => &self.wide_support,
        };
        support.get(frame).map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn attempts_for_with_support(
        &self,
        frame: usize,
        round: RegistrationRound,
        registered_support: &[usize],
    ) -> Vec<usize> {
        if frame >= self.frame_count || self.keyframes.binary_search(&frame).is_ok() {
            return Vec::new();
        }
        if registered_support.len() > MAX_DYNAMIC_SUPPORT_CANDIDATES {
            return self.attempts_for(frame, round).to_vec();
        }

        let mut registered_support: Vec<_> = registered_support
            .iter()
            .copied()
            .filter(|support| *support < self.frame_count && *support != frame)
            .collect();
        registered_support.sort_unstable();
        registered_support.dedup();
        self.attempts_for_with_sorted_support(frame, round, &registered_support)
            .unwrap_or_else(|_| self.attempts_for(frame, round).to_vec())
    }

    pub fn attempts_for_with_sorted_support(
        &self,
        frame: usize,
        round: RegistrationRound,
        registered_support: &[usize],
    ) -> Result<Vec<usize>, SequenceRegistrationError> {
        if registered_support.len() > MAX_DYNAMIC_SUPPORT_CANDIDATES {
            return Err(SequenceRegistrationError::DynamicSupportLimitExceeded {
                candidate_count: registered_support.len(),
                max_candidates: MAX_DYNAMIC_SUPPORT_CANDIDATES,
            });
        }
        if registered_support.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(SequenceRegistrationError::DynamicSupportNotSortedUnique);
        }
        if let Some(out_of_range) = registered_support
            .iter()
            .copied()
            .find(|support| *support >= self.frame_count)
        {
            return Err(SequenceRegistrationError::DynamicSupportFrameOutOfRange {
                frame: out_of_range,
                frame_count: self.frame_count,
            });
        }
        if frame >= self.frame_count || self.keyframes.binary_search(&frame).is_ok() {
            return Ok(Vec::new());
        }

        let mut keyframe_support = self.attempts_for(frame, round).to_vec();
        keyframe_support.sort_unstable();
        let candidates = merge_sorted_support(&keyframe_support, registered_support, frame);
        let neighbors_each_side = match round {
            RegistrationRound::Narrow => self.narrow_neighbors_each_side,
            RegistrationRound::Wide => self.wide_neighbors_each_side,
        };
        Ok(support_for(
            frame,
            &candidates,
            neighbors_each_side,
            &self.frame_ids,
            self.timestamps_us.as_deref(),
        ))
    }
}

#[derive(Serialize)]
struct SequenceRegistrationPlanRef<'a> {
    frame_count: usize,
    keyframes: &'a [usize],
    narrow_neighbors_each_side: usize,
    wide_neighbors_each_side: usize,
    frame_ids: &'a [u32],
    timestamps_us: Option<&'a [i64]>,
}

#[derive(Deserialize)]
struct SequenceRegistrationPlanWire {
    frame_count: usize,
    keyframes: Vec<usize>,
    narrow_neighbors_each_side: usize,
    wide_neighbors_each_side: usize,
    frame_ids: Vec<u32>,
    timestamps_us: Option<Vec<i64>>,
}

impl Serialize for SequenceRegistrationPlan {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SequenceRegistrationPlanRef {
            frame_count: self.frame_count,
            keyframes: &self.keyframes,
            narrow_neighbors_each_side: self.narrow_neighbors_each_side,
            wide_neighbors_each_side: self.wide_neighbors_each_side,
            frame_ids: &self.frame_ids,
            timestamps_us: self.timestamps_us.as_deref(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SequenceRegistrationPlan {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SequenceRegistrationPlanWire::deserialize(deserializer)?;
        validate_plan_limits(
            wire.frame_count,
            wire.narrow_neighbors_each_side,
            wire.wide_neighbors_each_side,
        )
        .map_err(de::Error::custom)?;
        validate_keyframes(wire.frame_count, &wire.keyframes).map_err(de::Error::custom)?;
        Self::build_validated(
            wire.frame_count,
            &wire.keyframes,
            wire.narrow_neighbors_each_side,
            wire.wide_neighbors_each_side,
            wire.frame_ids,
            wire.timestamps_us,
        )
        .map_err(de::Error::custom)
    }
}

fn validate_plan_limits(
    frame_count: usize,
    narrow_neighbors_each_side: usize,
    wide_neighbors_each_side: usize,
) -> Result<(), SequenceRegistrationError> {
    if frame_count > MAX_SEQUENCE_PLAN_FRAMES {
        return Err(SequenceRegistrationError::SequencePlanTooLarge {
            frame_count,
            max_frame_count: MAX_SEQUENCE_PLAN_FRAMES,
        });
    }
    for (round, requested) in [
        (RegistrationRound::Narrow, narrow_neighbors_each_side),
        (RegistrationRound::Wide, wide_neighbors_each_side),
    ] {
        if requested > MAX_SEQUENCE_NEIGHBORS {
            return Err(SequenceRegistrationError::SequenceNeighborLimitExceeded {
                round,
                requested,
                max_neighbors: MAX_SEQUENCE_NEIGHBORS,
            });
        }
    }

    let estimated_support_entries = frame_count as u128
        * 2
        * (narrow_neighbors_each_side as u128 + wide_neighbors_each_side as u128);
    if estimated_support_entries > MAX_TOTAL_SUPPORT_ENTRIES as u128 {
        return Err(SequenceRegistrationError::SequenceSupportBudgetExceeded {
            frame_count,
            estimated_support_entries,
            max_support_entries: MAX_TOTAL_SUPPORT_ENTRIES,
        });
    }
    Ok(())
}

fn merge_sorted_support(
    keyframe_support: &[usize],
    registered_support: &[usize],
    target_frame: usize,
) -> Vec<usize> {
    let mut merged = Vec::with_capacity(
        keyframe_support
            .len()
            .saturating_add(registered_support.len()),
    );
    let mut keyframe_index = 0;
    let mut registered_index = 0;
    while keyframe_index < keyframe_support.len() || registered_index < registered_support.len() {
        let next = match (
            keyframe_support.get(keyframe_index),
            registered_support.get(registered_index),
        ) {
            (Some(keyframe), Some(registered)) if keyframe < registered => {
                keyframe_index += 1;
                *keyframe
            }
            (Some(keyframe), Some(registered)) if registered < keyframe => {
                registered_index += 1;
                *registered
            }
            (Some(keyframe), Some(_)) => {
                keyframe_index += 1;
                registered_index += 1;
                *keyframe
            }
            (Some(keyframe), None) => {
                keyframe_index += 1;
                *keyframe
            }
            (None, Some(registered)) => {
                registered_index += 1;
                *registered
            }
            (None, None) => break,
        };
        if next != target_frame {
            merged.push(next);
        }
    }
    merged
}

fn validate_keyframes(
    frame_count: usize,
    keyframes: &[usize],
) -> Result<(), SequenceRegistrationError> {
    if frame_count > MAX_SEQUENCE_PLAN_FRAMES {
        return Err(SequenceRegistrationError::SequencePlanTooLarge {
            frame_count,
            max_frame_count: MAX_SEQUENCE_PLAN_FRAMES,
        });
    }
    if frame_count as u128 > u32::MAX as u128 + 1 {
        return Err(SequenceRegistrationError::FrameCountExceedsFrameIdRange {
            imported_frames: frame_count,
        });
    }
    if frame_count == 0 {
        return Err(SequenceRegistrationError::EmptySequence);
    }
    if keyframes.len() > frame_count {
        return Err(SequenceRegistrationError::TooManyKeyframes {
            frame_count,
            keyframe_count: keyframes.len(),
        });
    }
    if keyframes.is_empty() {
        return Err(SequenceRegistrationError::EmptyKeyframes);
    }

    let first = keyframes[0];
    if first >= frame_count {
        return Err(SequenceRegistrationError::KeyframeOutOfRange {
            frame: first,
            frame_count,
        });
    }
    for pair in keyframes.windows(2) {
        let current = pair[1];
        if current >= frame_count {
            return Err(SequenceRegistrationError::KeyframeOutOfRange {
                frame: current,
                frame_count,
            });
        }
        if pair[0] == current {
            return Err(SequenceRegistrationError::DuplicateKeyframe { frame: current });
        }
        if pair[0] > pair[1] {
            return Err(SequenceRegistrationError::UnsortedKeyframes {
                previous: pair[0],
                current: pair[1],
            });
        }
    }
    Ok(())
}

fn validate_plan_ordering(
    frame_count: usize,
    frame_ids: &[u32],
    timestamps_us: Option<&[i64]>,
) -> Result<(), SequenceRegistrationError> {
    if frame_ids.len() != frame_count {
        return Err(SequenceRegistrationError::InvalidFrameIds {
            imported_frames: frame_count,
            frame_id_count: frame_ids.len(),
            duplicate_frame_ids: Vec::new(),
        });
    }
    if let Some(timestamps_us) = timestamps_us {
        validate_timestamp_inputs(frame_count, timestamps_us.iter().copied().map(Some))?;
    }
    let mut observed = HashSet::with_capacity(frame_ids.len());
    let mut duplicates = BTreeSet::new();
    for frame_id in frame_ids.iter().copied() {
        if !observed.insert(frame_id) {
            duplicates.insert(frame_id);
        }
    }
    if !duplicates.is_empty() {
        return Err(SequenceRegistrationError::InvalidFrameIds {
            imported_frames: frame_count,
            frame_id_count: frame_ids.len(),
            duplicate_frame_ids: duplicates.into_iter().collect(),
        });
    }
    Ok(())
}

fn validate_timestamp_inputs<I>(
    frame_count: usize,
    timestamps: I,
) -> Result<(), SequenceRegistrationError>
where
    I: IntoIterator<Item = Option<i64>>,
{
    let mut timestamp_count = 0;
    let mut all_present = true;
    let mut previous_timestamp = None;
    let mut plateau_start = 0;
    let mut first_error = None;

    for (current_frame, timestamp) in timestamps.into_iter().enumerate() {
        timestamp_count += 1;
        let Some(timestamp) = timestamp else {
            all_present = false;
            continue;
        };
        if !all_present {
            continue;
        }

        if let Some(previous_timestamp_value) = previous_timestamp {
            if previous_timestamp_value > timestamp {
                if first_error.is_none() {
                    first_error = Some(SequenceRegistrationError::UnsortedTimestamps {
                        previous_frame: current_frame - 1,
                        current_frame,
                    });
                }
            } else if previous_timestamp_value != timestamp {
                if first_error.is_none() {
                    first_error = timestamp_plateau_error(
                        previous_timestamp_value,
                        current_frame.saturating_sub(plateau_start),
                    )
                    .err();
                }
                plateau_start = current_frame;
            }
        } else {
            plateau_start = current_frame;
        }
        previous_timestamp = Some(timestamp);
    }

    if timestamp_count != frame_count {
        return Err(SequenceRegistrationError::TimestampCountMismatch {
            frame_count,
            timestamp_count,
        });
    }
    if !all_present {
        return Ok(());
    }
    if let Some(error) = first_error {
        return Err(error);
    }
    if let Some(timestamp) = previous_timestamp {
        return timestamp_plateau_error(timestamp, timestamp_count.saturating_sub(plateau_start));
    }
    Ok(())
}

fn timestamp_plateau_error(
    timestamp_us: i64,
    plateau_size: usize,
) -> Result<(), SequenceRegistrationError> {
    if plateau_size > MAX_TIMESTAMP_PLATEAU {
        return Err(SequenceRegistrationError::TimestampPlateauTooLarge {
            timestamp_us,
            plateau_size,
            max_plateau_size: MAX_TIMESTAMP_PLATEAU,
        });
    }
    Ok(())
}

fn build_support_lists(
    frame_count: usize,
    keyframes: &[usize],
    neighbors_each_side: usize,
    frame_ids: &[u32],
    timestamps_us: Option<&[i64]>,
) -> Vec<Vec<usize>> {
    (0..frame_count)
        .map(|frame| {
            if keyframes.binary_search(&frame).is_ok() {
                Vec::new()
            } else {
                support_for(
                    frame,
                    keyframes,
                    neighbors_each_side,
                    frame_ids,
                    timestamps_us,
                )
            }
        })
        .collect()
}

fn support_for(
    frame: usize,
    candidates: &[usize],
    neighbors_each_side: usize,
    frame_ids: &[u32],
    timestamps_us: Option<&[i64]>,
) -> Vec<usize> {
    let left_end = candidates.partition_point(|candidate| *candidate < frame);
    let right_start = candidates.partition_point(|candidate| *candidate <= frame);
    let mut support = if let Some(timestamps_us) = timestamps_us {
        let left_candidates = bounded_left_timestamp_candidates(
            candidates,
            left_end,
            neighbors_each_side,
            timestamps_us,
        );
        let right_candidates = bounded_right_timestamp_candidates(
            candidates,
            right_start,
            neighbors_each_side,
            timestamps_us,
        );
        let mut support = select_top_support(
            frame,
            left_candidates,
            neighbors_each_side,
            frame_ids,
            Some(timestamps_us),
        );
        support.extend(select_top_support(
            frame,
            right_candidates,
            neighbors_each_side,
            frame_ids,
            Some(timestamps_us),
        ));
        support
    } else {
        let left_start = left_end.saturating_sub(neighbors_each_side);
        let right_end = right_start
            .saturating_add(neighbors_each_side)
            .min(candidates.len());
        let mut support = Vec::with_capacity(
            left_end.saturating_sub(left_start) + right_end.saturating_sub(right_start),
        );
        support.extend_from_slice(&candidates[left_start..left_end]);
        support.extend_from_slice(&candidates[right_start..right_end]);
        support
    };
    support.sort_by_key(|candidate| support_key(frame, *candidate, frame_ids, timestamps_us));
    support
}

fn bounded_left_timestamp_candidates<'a>(
    candidates: &'a [usize],
    left_end: usize,
    limit: usize,
    timestamps_us: &[i64],
) -> &'a [usize] {
    if limit == 0 || left_end == 0 {
        return &candidates[left_end..left_end];
    }

    let initial_start = left_end.saturating_sub(limit);
    let cutoff_timestamp = timestamps_us[candidates[initial_start]];
    let plateau_start = candidates[..initial_start]
        .partition_point(|candidate| timestamps_us[*candidate] < cutoff_timestamp);
    &candidates[plateau_start..left_end]
}

fn bounded_right_timestamp_candidates<'a>(
    candidates: &'a [usize],
    right_start: usize,
    limit: usize,
    timestamps_us: &[i64],
) -> &'a [usize] {
    let right_candidates = &candidates[right_start..];
    if limit == 0 || right_candidates.is_empty() {
        return &right_candidates[..0];
    }

    let initial_len = limit.min(right_candidates.len());
    let cutoff_timestamp = timestamps_us[right_candidates[initial_len - 1]];
    let plateau_end =
        right_candidates.partition_point(|candidate| timestamps_us[*candidate] <= cutoff_timestamp);
    &right_candidates[..plateau_end]
}

fn select_top_support(
    frame: usize,
    candidates: &[usize],
    limit: usize,
    frame_ids: &[u32],
    timestamps_us: Option<&[i64]>,
) -> Vec<usize> {
    if limit == 0 || candidates.is_empty() {
        return Vec::new();
    }

    let mut selected = BinaryHeap::with_capacity(limit.min(candidates.len()));
    for candidate in candidates.iter().copied() {
        let entry = (
            support_key(frame, candidate, frame_ids, timestamps_us),
            candidate,
        );
        if selected.len() < limit {
            selected.push(entry);
        } else if selected.peek().is_some_and(|worst| entry < *worst) {
            selected.pop();
            selected.push(entry);
        }
    }
    selected
        .into_iter()
        .map(|(_, candidate)| candidate)
        .collect()
}

fn support_key(
    frame: usize,
    candidate: usize,
    frame_ids: &[u32],
    timestamps_us: Option<&[i64]>,
) -> (u128, u32) {
    let distance = if let Some(timestamps_us) = timestamps_us {
        timestamps_us[candidate].abs_diff(timestamps_us[frame]) as u128
    } else {
        candidate.abs_diff(frame) as u128
    };
    (distance, frame_ids[candidate])
}
