use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::path::PathBuf;

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
    pub diagnostics: Vec<FrameRegistrationDiagnostic>,
    pub sparse_model: PathBuf,
}

impl SequenceRegistrationResult {
    pub fn has_complete_coverage(&self) -> bool {
        self.validate_complete_coverage().is_ok()
    }

    pub fn validate_complete_coverage(&self) -> Result<(), SequenceRegistrationError> {
        if self.imported_frames as u128 > u32::MAX as u128 + 1 {
            return Err(SequenceRegistrationError::FrameCountExceedsFrameIdRange {
                imported_frames: self.imported_frames,
            });
        }

        let mut observed_frame_ids = BTreeSet::new();
        let mut duplicate_frame_ids = BTreeSet::new();
        let mut unexpected_frame_ids = BTreeSet::new();
        for diagnostic in &self.diagnostics {
            if diagnostic.frame_id as u128 >= self.imported_frames as u128 {
                unexpected_frame_ids.insert(diagnostic.frame_id);
            }
            if !observed_frame_ids.insert(diagnostic.frame_id) {
                duplicate_frame_ids.insert(diagnostic.frame_id);
            }
        }
        let missing_frame_ids: Vec<_> = (0..self.imported_frames)
            .map(|frame| frame as u32)
            .filter(|frame| !observed_frame_ids.contains(frame))
            .collect();
        let duplicate_frame_ids: Vec<_> = duplicate_frame_ids.into_iter().collect();
        let unexpected_frame_ids: Vec<_> = unexpected_frame_ids.into_iter().collect();
        if self.diagnostics.len() != self.imported_frames
            || !missing_frame_ids.is_empty()
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
        validate_keyframes(frame_count, keyframes)?;

        let pending = (0..frame_count)
            .filter(|frame| keyframes.binary_search(frame).is_err())
            .collect();
        let narrow_support =
            build_support_lists(frame_count, keyframes, narrow_neighbors_each_side);
        let wide_support = build_support_lists(frame_count, keyframes, wide_neighbors_each_side);

        Ok(Self {
            frame_count,
            keyframes: keyframes.to_vec(),
            narrow_neighbors_each_side,
            wide_neighbors_each_side,
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
}

#[derive(Serialize)]
struct SequenceRegistrationPlanRef<'a> {
    frame_count: usize,
    keyframes: &'a [usize],
    narrow_neighbors_each_side: usize,
    wide_neighbors_each_side: usize,
}

#[derive(Deserialize)]
struct SequenceRegistrationPlanWire {
    frame_count: usize,
    keyframes: Vec<usize>,
    narrow_neighbors_each_side: usize,
    wide_neighbors_each_side: usize,
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
        Self::build(
            wire.frame_count,
            &wire.keyframes,
            wire.narrow_neighbors_each_side,
            wire.wide_neighbors_each_side,
        )
        .map_err(de::Error::custom)
    }
}

fn validate_keyframes(
    frame_count: usize,
    keyframes: &[usize],
) -> Result<(), SequenceRegistrationError> {
    if frame_count == 0 {
        return Err(SequenceRegistrationError::EmptySequence);
    }
    if keyframes.is_empty() {
        return Err(SequenceRegistrationError::EmptyKeyframes);
    }

    for (index, frame) in keyframes.iter().copied().enumerate() {
        if frame >= frame_count {
            return Err(SequenceRegistrationError::KeyframeOutOfRange { frame, frame_count });
        }
        if keyframes[..index].contains(&frame) {
            return Err(SequenceRegistrationError::DuplicateKeyframe { frame });
        }
    }

    for pair in keyframes.windows(2) {
        if pair[0] > pair[1] {
            return Err(SequenceRegistrationError::UnsortedKeyframes {
                previous: pair[0],
                current: pair[1],
            });
        }
    }
    Ok(())
}

fn build_support_lists(
    frame_count: usize,
    keyframes: &[usize],
    neighbors_each_side: usize,
) -> Vec<Vec<usize>> {
    (0..frame_count)
        .map(|frame| {
            if keyframes.binary_search(&frame).is_ok() {
                Vec::new()
            } else {
                support_for(frame, keyframes, neighbors_each_side)
            }
        })
        .collect()
}

fn support_for(frame: usize, keyframes: &[usize], neighbors_each_side: usize) -> Vec<usize> {
    let mut support: Vec<_> = keyframes
        .iter()
        .rev()
        .copied()
        .filter(|keyframe| *keyframe < frame)
        .take(neighbors_each_side)
        .chain(
            keyframes
                .iter()
                .copied()
                .filter(|keyframe| *keyframe > frame)
                .take(neighbors_each_side),
        )
        .collect();
    support.sort_by_key(|keyframe| (keyframe.abs_diff(frame), *keyframe));
    support
}
