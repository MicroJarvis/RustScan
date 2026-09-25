//! Canonical frame selection and split manifests for RustGS.
//!
//! Selection order is fixed:
//! 1. resolve manifest / include / exclude by stable COLMAP `image_id`
//! 2. apply `max_frames`
//! 3. apply `frame_stride`
//! 4. fingerprint the resulting ordered stable IDs
//!
//! Train, eval, crop, and report must reuse one [`FrameSelection`] result rather
//! than re-implementing filters.

use crate::TrainingDataset;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashSet};
use std::fs;
use std::io;
use std::path::Path;
use std::str::FromStr;

/// How post-training evaluation frames were chosen relative to the train split.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationSplitKind {
    /// Frames drawn from the training / in-view set (not a held-out split).
    InView,
    /// Frames drawn from an explicit holdout set in a split manifest.
    Holdout,
}

impl std::fmt::Display for EvaluationSplitKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InView => write!(f, "in-view"),
            Self::Holdout => write!(f, "holdout"),
        }
    }
}

impl FromStr for EvaluationSplitKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "in-view" | "in_view" | "inview" => Ok(Self::InView),
            "holdout" | "held-out" | "held_out" => Ok(Self::Holdout),
            other => Err(format!(
                "unsupported eval split '{other}'. Expected in-view or holdout"
            )),
        }
    }
}

/// Inclusive stable-ID range used by include/exclude filters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameIdRange {
    pub start: u64,
    pub end: u64,
}

impl FrameIdRange {
    #[must_use]
    pub fn contains(self, frame_id: u64) -> bool {
        self.start <= frame_id && frame_id <= self.end
    }
}

/// Parse comma-separated `<id>` or `<start>-<end>` / `<start>..<end>` tokens.
pub fn parse_frame_id_ranges(value: Option<&str>) -> Result<Vec<FrameIdRange>, String> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let mut ranges = Vec::new();
    for raw_token in value.split(',') {
        let token = raw_token.trim();
        if token.is_empty() {
            continue;
        }
        let (start, end) = if let Some((start, end)) = token.split_once("..") {
            (start.trim(), end.trim())
        } else if let Some((start, end)) = token.split_once('-') {
            (start.trim(), end.trim())
        } else {
            (token, token)
        };
        if start.is_empty() || end.is_empty() {
            return Err(format!(
                "frame range '{token}' must be <frame_id> or <start>-<end>"
            ));
        }
        let start = start
            .parse::<u64>()
            .map_err(|_| format!("invalid frame range start in '{token}'"))?;
        let end = end
            .parse::<u64>()
            .map_err(|_| format!("invalid frame range end in '{token}'"))?;
        if start > end {
            return Err(format!("frame range '{token}' has start greater than end"));
        }
        ranges.push(FrameIdRange { start, end });
    }
    Ok(ranges)
}

/// Explicit train / in-view / holdout partition keyed by stable COLMAP image IDs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameSplitManifest {
    pub dataset_fingerprint: String,
    pub train_ids: Vec<u64>,
    pub in_view_ids: Vec<u64>,
    pub holdout_ids: Vec<u64>,
}

impl FrameSplitManifest {
    pub fn load_path(path: &Path) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|err| {
            format!(
                "failed to read frame-split manifest {}: {err}",
                path.display()
            )
        })?;
        serde_json::from_slice(&bytes).map_err(|err| {
            format!(
                "failed to parse frame-split manifest {}: {err}",
                path.display()
            )
        })
    }

    pub fn write_path(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let mut json = serde_json::to_vec_pretty(self)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        json.push(b'\n');
        let temp = path.with_extension("json.tmp");
        fs::write(&temp, &json)?;
        fs::rename(&temp, path)?;
        Ok(())
    }

    /// Validate against a loaded dataset and expected sparse-model fingerprint.
    pub fn validate(
        &self,
        dataset: &TrainingDataset,
        expected_dataset_fingerprint: &str,
    ) -> Result<(), String> {
        if self.dataset_fingerprint != expected_dataset_fingerprint {
            return Err(format!(
                "frame-split manifest dataset fingerprint mismatch: manifest={} dataset={}",
                self.dataset_fingerprint, expected_dataset_fingerprint
            ));
        }

        let available: HashSet<u64> = dataset.poses.iter().map(|pose| pose.frame_id).collect();
        self.validate_id_list(&self.train_ids, "train", &available)?;
        self.validate_id_list(&self.in_view_ids, "in_view", &available)?;
        self.validate_id_list(&self.holdout_ids, "holdout", &available)?;

        let train: HashSet<u64> = self.train_ids.iter().copied().collect();
        let holdout: HashSet<u64> = self.holdout_ids.iter().copied().collect();
        let overlap: BTreeSet<u64> = train.intersection(&holdout).copied().collect();
        if !overlap.is_empty() {
            return Err(format!(
                "frame-split manifest train/holdout overlap on stable IDs: {:?}",
                overlap.into_iter().collect::<Vec<_>>()
            ));
        }
        Ok(())
    }

    fn validate_id_list(
        &self,
        ids: &[u64],
        label: &str,
        available: &HashSet<u64>,
    ) -> Result<(), String> {
        let mut seen = HashSet::new();
        for &id in ids {
            if !seen.insert(id) {
                return Err(format!(
                    "frame-split manifest {label} IDs contain duplicate stable id {id}"
                ));
            }
            if !available.contains(&id) {
                return Err(format!(
                    "frame-split manifest {label} IDs contain unknown stable id {id}"
                ));
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn fingerprint(&self) -> String {
        let bytes = serde_json::to_vec(self).expect("FrameSplitManifest serializes");
        blake3::hash(&bytes).to_hex().to_string()
    }
}

/// Reconstruct the historical `static_162` evaluation set as stable IDs.
///
/// Pre-C5 semantics: take the first 180 poses in load order (old `max_frames=180`),
/// then drop enumerated indices `76..=93` (old `--exclude-frame-ranges 76-93` matched
/// post-filter frame_idx, not COLMAP `image_id`). Pinning via `allowed_ids` keeps
/// FrameSelection's include/exclude→max→stride order without pulling post-prefix frames.
pub fn static_162_allowed_stable_ids(source: &TrainingDataset) -> Result<Vec<u64>, String> {
    if source.poses.len() < 180 {
        return Err(format!(
            "static_162 requires at least 180 source frames, got {}",
            source.poses.len()
        ));
    }
    Ok(source
        .poses
        .iter()
        .take(180)
        .enumerate()
        .filter(|(idx, _)| !(76..=93).contains(idx))
        .map(|(_, pose)| pose.frame_id)
        .collect())
}

/// Parameters for one canonical selection pass.
#[derive(Debug, Clone, Default)]
pub struct FrameSelectionRequest {
    pub include_ranges: Vec<FrameIdRange>,
    pub exclude_ranges: Vec<FrameIdRange>,
    /// Restrict candidates to these stable IDs before include/exclude (manifest sets).
    pub allowed_ids: Option<Vec<u64>>,
    /// 0 keeps all remaining frames after include/exclude.
    pub max_frames: usize,
    pub frame_stride: usize,
}

/// Result of one canonical selection: ordered stable IDs + filtered dataset.
#[derive(Debug, Clone)]
pub struct FrameSelection {
    pub stable_ids: Vec<u64>,
    pub dataset: TrainingDataset,
    pub selection_fingerprint: String,
    pub max_frames: usize,
    pub frame_stride: usize,
}

impl FrameSelection {
    /// Apply the fixed selection order to `source`.
    ///
    /// Unknown IDs in `allowed_ids` are rejected. Empty results after filtering
    /// are allowed only when the source itself was empty (caller decides).
    pub fn select(
        source: &TrainingDataset,
        request: &FrameSelectionRequest,
    ) -> Result<Self, String> {
        let stride = request.frame_stride.max(1);
        let available: HashSet<u64> = source.poses.iter().map(|pose| pose.frame_id).collect();

        if let Some(allowed) = &request.allowed_ids {
            let mut seen = HashSet::new();
            for &id in allowed {
                if !seen.insert(id) {
                    return Err(format!(
                        "frame selection allowed_ids contain duplicate stable id {id}"
                    ));
                }
                if !available.contains(&id) {
                    return Err(format!(
                        "frame selection allowed_ids contain unknown stable id {id}"
                    ));
                }
            }
        }

        let mut candidates: Vec<_> = source.poses.iter().collect();
        if let Some(allowed) = &request.allowed_ids {
            let allowed_set: HashSet<u64> = allowed.iter().copied().collect();
            candidates.retain(|pose| allowed_set.contains(&pose.frame_id));
        }

        if !request.include_ranges.is_empty() {
            candidates.retain(|pose| {
                request
                    .include_ranges
                    .iter()
                    .any(|range| range.contains(pose.frame_id))
            });
        }

        if !request.exclude_ranges.is_empty() {
            candidates.retain(|pose| {
                !request
                    .exclude_ranges
                    .iter()
                    .any(|range| range.contains(pose.frame_id))
            });
        }

        if request.max_frames > 0 && candidates.len() > request.max_frames {
            candidates.truncate(request.max_frames);
        }

        let selected_poses: Vec<_> = candidates.into_iter().step_by(stride).cloned().collect();
        let stable_ids: Vec<u64> = selected_poses.iter().map(|pose| pose.frame_id).collect();
        let selection_fingerprint =
            fingerprint_selection(&stable_ids, request.max_frames, stride, request);

        let mut dataset =
            TrainingDataset::new(source.intrinsics).with_depth_scale(source.depth_scale);
        dataset.initial_points = source.initial_points.clone();
        for pose in selected_poses {
            dataset.add_pose(pose);
        }

        Ok(Self {
            stable_ids,
            dataset,
            selection_fingerprint,
            max_frames: request.max_frames,
            frame_stride: stride,
        })
    }
}

fn fingerprint_selection(
    stable_ids: &[u64],
    max_frames: usize,
    frame_stride: usize,
    request: &FrameSelectionRequest,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"rustgs-frame-selection-v1\0");
    hasher.update(&max_frames.to_le_bytes());
    hasher.update(&frame_stride.to_le_bytes());
    update_ranges(&mut hasher, b"include", &request.include_ranges);
    update_ranges(&mut hasher, b"exclude", &request.exclude_ranges);
    if let Some(allowed) = &request.allowed_ids {
        hasher.update(b"allowed\0");
        hasher.update(&(allowed.len() as u64).to_le_bytes());
        for id in allowed {
            hasher.update(&id.to_le_bytes());
        }
    } else {
        hasher.update(b"allowed-none\0");
    }
    hasher.update(b"ids\0");
    hasher.update(&(stable_ids.len() as u64).to_le_bytes());
    for id in stable_ids {
        hasher.update(&id.to_le_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

fn update_ranges(hasher: &mut blake3::Hasher, label: &[u8], ranges: &[FrameIdRange]) {
    hasher.update(label);
    hasher.update(b"\0");
    hasher.update(&(ranges.len() as u64).to_le_bytes());
    for range in ranges {
        hasher.update(&range.start.to_le_bytes());
        hasher.update(&range.end.to_le_bytes());
    }
}

/// Reject frame IDs that cannot be represented in a legacy `u32` boundary.
pub fn require_frame_id_u32(frame_id: u64, context: &str) -> Result<u32, String> {
    u32::try_from(frame_id).map_err(|_| {
        format!("{context}: frame_id {frame_id} exceeds u32::MAX and cannot be truncated")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustscan_types::{Intrinsics, ScenePose, SE3};
    use std::path::PathBuf;

    fn pose(frame_id: u64) -> ScenePose {
        ScenePose::new(
            frame_id,
            PathBuf::from(format!("frame_{frame_id}.png")),
            SE3::identity(),
            0.0,
        )
    }

    fn dataset_with_ids(ids: &[u64]) -> TrainingDataset {
        let mut dataset = TrainingDataset::new(Intrinsics::new(1.0, 1.0, 0.0, 0.0, 8, 8));
        for id in ids {
            dataset.add_pose(pose(*id));
        }
        dataset
    }

    #[test]
    fn empty_split_yields_empty_selection() {
        let source = dataset_with_ids(&[]);
        let selection = FrameSelection::select(&source, &FrameSelectionRequest::default()).unwrap();
        assert!(selection.stable_ids.is_empty());
        assert!(selection.dataset.poses.is_empty());
    }

    #[test]
    fn single_frame_split_is_stable() {
        let source = dataset_with_ids(&[42]);
        let selection = FrameSelection::select(&source, &FrameSelectionRequest::default()).unwrap();
        assert_eq!(selection.stable_ids, vec![42]);
        assert_eq!(selection.dataset.poses[0].frame_id, 42);
    }

    #[test]
    fn duplicate_allowed_ids_are_rejected() {
        let source = dataset_with_ids(&[1, 2, 3]);
        let err = FrameSelection::select(
            &source,
            &FrameSelectionRequest {
                allowed_ids: Some(vec![1, 1]),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(err.contains("duplicate"));
    }

    #[test]
    fn unknown_allowed_ids_are_rejected() {
        let source = dataset_with_ids(&[1, 2, 3]);
        let err = FrameSelection::select(
            &source,
            &FrameSelectionRequest {
                allowed_ids: Some(vec![99]),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(err.contains("unknown"));
    }

    #[test]
    fn selection_fingerprint_is_stable_across_runs() {
        let source = dataset_with_ids(&[10, 20, 30, 40]);
        let request = FrameSelectionRequest {
            include_ranges: vec![FrameIdRange { start: 20, end: 40 }],
            max_frames: 2,
            frame_stride: 1,
            ..Default::default()
        };
        let a = FrameSelection::select(&source, &request).unwrap();
        let b = FrameSelection::select(&source, &request).unwrap();
        assert_eq!(a.stable_ids, b.stable_ids);
        assert_eq!(a.selection_fingerprint, b.selection_fingerprint);
        assert_eq!(a.stable_ids, vec![20, 30]);
    }

    #[test]
    fn include_late_sorted_id_before_max_frames() {
        // COLMAP order is 1,2,3,...,10. Including only id 10 must survive even
        // when max_frames would have dropped a prefix-only selection.
        let source = dataset_with_ids(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let selection = FrameSelection::select(
            &source,
            &FrameSelectionRequest {
                include_ranges: vec![FrameIdRange { start: 10, end: 10 }],
                max_frames: 3,
                frame_stride: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(selection.stable_ids, vec![10]);
    }

    #[test]
    fn include_before_max_keeps_late_id_that_prefix_max_would_drop() {
        // Regression: applying max_frames before include would truncate to [1,2,3]
        // and drop stable id 9. Canonical order must include first.
        let source = dataset_with_ids(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let wrong_order_prefix: Vec<u64> = source
            .poses
            .iter()
            .take(3)
            .map(|pose| pose.frame_id)
            .collect();
        assert_eq!(wrong_order_prefix, vec![1, 2, 3]);
        assert!(!wrong_order_prefix.contains(&9));

        let selection = FrameSelection::select(
            &source,
            &FrameSelectionRequest {
                include_ranges: vec![FrameIdRange { start: 9, end: 9 }],
                max_frames: 3,
                frame_stride: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            selection.stable_ids,
            vec![9],
            "include of late stable id must run before max_frames"
        );
    }

    #[test]
    fn max_then_stride_apply_after_include() {
        let source = dataset_with_ids(&[1, 2, 3, 4, 5, 6]);
        let selection = FrameSelection::select(
            &source,
            &FrameSelectionRequest {
                include_ranges: vec![FrameIdRange { start: 2, end: 6 }],
                max_frames: 4,
                frame_stride: 2,
                ..Default::default()
            },
        )
        .unwrap();
        // include → [2,3,4,5,6], max 4 → [2,3,4,5], stride 2 → [2,4]
        assert_eq!(selection.stable_ids, vec![2, 4]);
    }

    #[test]
    fn static_162_allowed_ids_pin_prefix_without_backfill() {
        // Fixture: COLMAP image_id == 1..=300 in load order (matches common Home layouts).
        // Old mapping: enumerated idx i → image_id i+1; exclude idx 76..=93 → drop IDs 77..=94.
        let source = dataset_with_ids(&(1..=300).collect::<Vec<_>>());
        let allowed = static_162_allowed_stable_ids(&source).unwrap();
        let expected: Vec<u64> = (1..=76).chain(95..=180).collect();
        assert_eq!(allowed.len(), 162);
        assert_eq!(allowed, expected);
        assert_eq!(*allowed.last().unwrap(), 180);

        let selection = FrameSelection::select(
            &source,
            &FrameSelectionRequest {
                allowed_ids: Some(allowed.clone()),
                max_frames: 180,
                frame_stride: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(selection.stable_ids.len(), 162);
        assert_eq!(selection.stable_ids, expected);
        assert!(
            selection.stable_ids.iter().all(|&id| id <= 180),
            "must not backfill frames beyond the original 180-prefix"
        );
        assert!(
            !selection
                .stable_ids
                .iter()
                .any(|&id| (77..=94).contains(&id)),
            "must not include the historically excluded band"
        );

        // Drift regression: exclude-then-max without allowed_ids pulls id 198.
        let drifted = FrameSelection::select(
            &source,
            &FrameSelectionRequest {
                exclude_ranges: vec![FrameIdRange { start: 76, end: 93 }],
                max_frames: 180,
                frame_stride: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(drifted.stable_ids.len(), 180);
        assert_eq!(*drifted.stable_ids.last().unwrap(), 198);
        assert_ne!(drifted.stable_ids, selection.stable_ids);
    }

    #[test]
    fn manifest_rejects_fingerprint_mismatch() {
        let source = dataset_with_ids(&[1, 2]);
        let manifest = FrameSplitManifest {
            dataset_fingerprint: "aaa".into(),
            train_ids: vec![1],
            in_view_ids: vec![1],
            holdout_ids: vec![2],
        };
        let err = manifest.validate(&source, "bbb").unwrap_err();
        assert!(err.contains("fingerprint mismatch"));
    }

    #[test]
    fn manifest_rejects_train_holdout_overlap() {
        let source = dataset_with_ids(&[1, 2, 3]);
        let manifest = FrameSplitManifest {
            dataset_fingerprint: "fp".into(),
            train_ids: vec![1, 2],
            in_view_ids: vec![1],
            holdout_ids: vec![2, 3],
        };
        let err = manifest.validate(&source, "fp").unwrap_err();
        assert!(err.contains("overlap"));
    }

    #[test]
    fn manifest_rejects_unknown_and_duplicate_ids() {
        let source = dataset_with_ids(&[1, 2]);
        let unknown = FrameSplitManifest {
            dataset_fingerprint: "fp".into(),
            train_ids: vec![1, 9],
            in_view_ids: vec![],
            holdout_ids: vec![],
        };
        assert!(unknown
            .validate(&source, "fp")
            .unwrap_err()
            .contains("unknown"));

        let dup = FrameSplitManifest {
            dataset_fingerprint: "fp".into(),
            train_ids: vec![1, 1],
            in_view_ids: vec![],
            holdout_ids: vec![],
        };
        assert!(dup
            .validate(&source, "fp")
            .unwrap_err()
            .contains("duplicate"));
    }

    #[test]
    fn require_frame_id_u32_rejects_overflow() {
        let err = require_frame_id_u32(u64::from(u32::MAX) + 1, "report").unwrap_err();
        assert!(err.contains("exceeds u32::MAX"));
        assert_eq!(require_frame_id_u32(42, "report").unwrap(), 42);
    }

    #[test]
    fn parse_eval_split_kind() {
        assert_eq!(
            "in-view".parse::<EvaluationSplitKind>().unwrap(),
            EvaluationSplitKind::InView
        );
        assert_eq!(
            "holdout".parse::<EvaluationSplitKind>().unwrap(),
            EvaluationSplitKind::Holdout
        );
        assert!("bogus".parse::<EvaluationSplitKind>().is_err());
    }
}
