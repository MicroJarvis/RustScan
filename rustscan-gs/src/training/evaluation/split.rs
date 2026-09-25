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

pub use crate::io::colmap_dataset::ColmapFrameCandidate;

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

/// Historical `static_162` exclude band on pre-C5 enumerated indices (not COLMAP IDs).
const STATIC_162_EXCLUDE_ENUMERATED: std::ops::RangeInclusive<usize> = 76..=93;
const STATIC_162_PREFIX_LEN: usize = 180;

/// Reconstruct the historical `static_162` allowed set from the original COLMAP
/// candidate list (model order, including missing files).
///
/// Pre-C5 semantics: `take(180)` on the COLMAP list, enumerate indices `0..179`,
/// keep entries that exist and whose enumerated index is outside `76..=93`.
/// A filtered [`TrainingDataset`] alone cannot supply those indices when images
/// were skipped — callers must pass candidates or an explicit stable-ID manifest.
pub fn static_162_allowed_stable_ids_from_candidates(
    candidates: &[ColmapFrameCandidate],
) -> Result<Vec<u64>, String> {
    if candidates.len() < STATIC_162_PREFIX_LEN {
        return Err(format!(
            "static_162 requires at least {STATIC_162_PREFIX_LEN} COLMAP candidates, got {}; \
             old quality gate is inapplicable without the original candidate list",
            candidates.len()
        ));
    }
    Ok(candidates
        .iter()
        .take(STATIC_162_PREFIX_LEN)
        .enumerate()
        .filter(|(idx, candidate)| {
            candidate.file_exists && !STATIC_162_EXCLUDE_ENUMERATED.contains(idx)
        })
        .map(|(_, candidate)| candidate.image_id)
        .collect())
}

/// Reject reconstructing `static_162` from a filtered dataset alone.
///
/// Missing images change enumerated indices; use
/// [`static_162_allowed_stable_ids_from_candidates`] with the original COLMAP
/// list, or pin an explicit stable-ID manifest.
pub fn static_162_allowed_stable_ids(_source: &TrainingDataset) -> Result<Vec<u64>, String> {
    Err(
        "static_162 cannot be reconstructed from a filtered TrainingDataset alone \
         (missing images invalidate re-enumeration of poses); pass the original \
         COLMAP candidate list to static_162_allowed_stable_ids_from_candidates \
         or an explicit stable-ID manifest — old quality gate must not run on a \
         guessed frame set"
            .into(),
    )
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
    pub include_ranges: Vec<FrameIdRange>,
    pub exclude_ranges: Vec<FrameIdRange>,
    pub allowed_ids: Option<Vec<u64>>,
}

/// Serializable selection provenance for eval / suite reports.
///
/// Captures the *actual* request parameters and resulting stable IDs so reports
/// remain reproducible even when `SplatEvaluationSummary` records post-preselect
/// `max_frames=0` / `frame_stride=1`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameSelectionReport {
    pub stable_frame_ids: Vec<u64>,
    pub selection_fingerprint: String,
    pub max_frames: usize,
    pub frame_stride: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_frame_ranges: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude_frame_ranges: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_split_kind: Option<String>,
}

impl FrameSelectionReport {
    #[must_use]
    pub fn from_selection(
        selection: &FrameSelection,
        manifest_fingerprint: Option<String>,
        eval_split_kind: Option<String>,
    ) -> Self {
        Self {
            stable_frame_ids: selection.stable_ids.clone(),
            selection_fingerprint: selection.selection_fingerprint.clone(),
            max_frames: selection.max_frames,
            frame_stride: selection.frame_stride,
            include_frame_ranges: format_frame_id_ranges(&selection.include_ranges),
            exclude_frame_ranges: format_frame_id_ranges(&selection.exclude_ranges),
            manifest_fingerprint,
            eval_split_kind,
        }
    }

    /// Build a report for in-view evaluation paths (evaluate_psnr / eval suite).
    #[must_use]
    pub fn from_in_view_selection(
        selection: &FrameSelection,
        manifest_fingerprint: Option<String>,
    ) -> Self {
        Self::from_selection(selection, manifest_fingerprint, Some("in-view".to_string()))
    }
}

/// Aggregate quality-gate check statuses: Failed > Inapplicable > Passed.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationGateStatus {
    Passed,
    Failed,
    /// Threshold must not be applied (e.g. historical static_162 unrestorable).
    Inapplicable,
}

/// Collapse per-check statuses with Failed beating Inapplicable beating Passed.
#[must_use]
pub fn aggregate_evaluation_gate_status(
    statuses: impl IntoIterator<Item = EvaluationGateStatus>,
) -> EvaluationGateStatus {
    let mut saw_inapplicable = false;
    for status in statuses {
        match status {
            EvaluationGateStatus::Failed => return EvaluationGateStatus::Failed,
            EvaluationGateStatus::Inapplicable => saw_inapplicable = true,
            EvaluationGateStatus::Passed => {}
        }
    }
    if saw_inapplicable {
        EvaluationGateStatus::Inapplicable
    } else {
        EvaluationGateStatus::Passed
    }
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
        let resolved_request = FrameSelectionRequest {
            include_ranges: request.include_ranges.clone(),
            exclude_ranges: request.exclude_ranges.clone(),
            allowed_ids: request.allowed_ids.clone(),
            max_frames: request.max_frames,
            frame_stride: stride,
        };
        let selection_fingerprint = fingerprint_frame_selection(&stable_ids, &resolved_request);

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
            max_frames: resolved_request.max_frames,
            frame_stride: stride,
            include_ranges: resolved_request.include_ranges,
            exclude_ranges: resolved_request.exclude_ranges,
            allowed_ids: resolved_request.allowed_ids,
        })
    }

    /// Rebuild the request that produced this selection (for fingerprint checks).
    #[must_use]
    pub fn request(&self) -> FrameSelectionRequest {
        FrameSelectionRequest {
            include_ranges: self.include_ranges.clone(),
            exclude_ranges: self.exclude_ranges.clone(),
            allowed_ids: self.allowed_ids.clone(),
            max_frames: self.max_frames,
            frame_stride: self.frame_stride,
        }
    }
}

/// Canonical blake3 fingerprint of a selection (request params + ordered stable IDs).
///
/// Independently recomputable from stored checkpoint / report metadata.
#[must_use]
pub fn fingerprint_frame_selection(stable_ids: &[u64], request: &FrameSelectionRequest) -> String {
    let frame_stride = request.frame_stride.max(1);
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"rustgs-frame-selection-v1\0");
    hasher.update(&request.max_frames.to_le_bytes());
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

/// Format inclusive ranges as comma-separated `<id>` / `<start>-<end>` tokens.
#[must_use]
pub fn format_frame_id_ranges(ranges: &[FrameIdRange]) -> Option<String> {
    if ranges.is_empty() {
        return None;
    }
    Some(
        ranges
            .iter()
            .map(|range| {
                if range.start == range.end {
                    range.start.to_string()
                } else {
                    format!("{}-{}", range.start, range.end)
                }
            })
            .collect::<Vec<_>>()
            .join(","),
    )
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
    fn static_162_from_candidates_covers_gap_free_and_missing_cases() {
        // Frozen old-loader expectation helper (independent of the migration API):
        // take(180) COLMAP slots → keep exist && enumerated idx ∉ 76..=93.
        fn expected_from_old_semantics(candidates: &[ColmapFrameCandidate]) -> Vec<u64> {
            candidates
                .iter()
                .take(180)
                .enumerate()
                .filter(|(idx, c)| c.file_exists && !(76..=93).contains(idx))
                .map(|(_, c)| c.image_id)
                .collect()
        }

        // 1) Complete 300-frame consecutive IDs, all present.
        let complete: Vec<ColmapFrameCandidate> = (1..=300)
            .map(|image_id| ColmapFrameCandidate {
                image_id,
                file_exists: true,
            })
            .collect();
        let expected_complete: Vec<u64> = (1..=76).chain(95..=180).collect();
        assert_eq!(expected_from_old_semantics(&complete), expected_complete);
        let allowed = static_162_allowed_stable_ids_from_candidates(&complete).unwrap();
        assert_eq!(allowed, expected_complete);
        assert_eq!(allowed.len(), 162);
        assert_eq!(*allowed.last().unwrap(), 180);

        let source = dataset_with_ids(&(1..=300).collect::<Vec<_>>());
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
        assert_eq!(selection.stable_ids, expected_complete);
        assert!(selection.stable_ids.iter().all(|&id| id <= 180));

        // Filtered-dataset helper must refuse (cannot see COLMAP gaps).
        let err = static_162_allowed_stable_ids(&source).unwrap_err();
        assert!(
            err.contains("cannot be reconstructed") || err.contains("candidate"),
            "{err}"
        );

        // 2) Missing image inside prefix, outside exclude band (id 50 at slot 49).
        let mut missing_outside = complete.clone();
        missing_outside[49].file_exists = false; // image_id 50
        let expected_missing_outside = expected_from_old_semantics(&missing_outside);
        assert_eq!(expected_missing_outside.len(), 161);
        assert_eq!(*expected_missing_outside.last().unwrap(), 180);
        assert!(!expected_missing_outside.contains(&50));
        assert!(!expected_missing_outside.contains(&77));
        assert!(!expected_missing_outside.contains(&181));
        assert!(expected_missing_outside.contains(&95));
        assert_eq!(
            static_162_allowed_stable_ids_from_candidates(&missing_outside).unwrap(),
            expected_missing_outside
        );

        // 3) Missing image inside exclude band (id 80 at slot 79) — kept set unchanged.
        let mut missing_inside = complete.clone();
        missing_inside[79].file_exists = false; // image_id 80, enumerated idx 79 ∈ 76..=93
        let expected_missing_inside = expected_from_old_semantics(&missing_inside);
        assert_eq!(expected_missing_inside, expected_complete);
        assert_eq!(
            static_162_allowed_stable_ids_from_candidates(&missing_inside).unwrap(),
            expected_complete
        );

        // 4) Non-contiguous COLMAP image_ids (still 300 ordered candidates).
        let sparse_ids: Vec<u64> = (0..300).map(|i| 1_000 + i * 7).collect();
        let sparse: Vec<ColmapFrameCandidate> = sparse_ids
            .iter()
            .map(|&image_id| ColmapFrameCandidate {
                image_id,
                file_exists: true,
            })
            .collect();
        let expected_sparse = expected_from_old_semantics(&sparse);
        assert_eq!(expected_sparse.len(), 162);
        assert_eq!(
            expected_sparse,
            sparse_ids
                .iter()
                .take(180)
                .enumerate()
                .filter(|(idx, _)| !(76..=93).contains(idx))
                .map(|(_, id)| *id)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            static_162_allowed_stable_ids_from_candidates(&sparse).unwrap(),
            expected_sparse
        );
        assert!(
            !expected_sparse
                .iter()
                .any(|&id| id == sparse_ids[180] || id == sparse_ids[198]),
            "must not backfill past the COLMAP take(180) prefix"
        );
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

    #[test]
    fn gate_status_aggregates_failed_over_inapplicable_over_passed() {
        assert_eq!(
            aggregate_evaluation_gate_status([
                EvaluationGateStatus::Passed,
                EvaluationGateStatus::Passed,
            ]),
            EvaluationGateStatus::Passed
        );
        assert_eq!(
            aggregate_evaluation_gate_status([
                EvaluationGateStatus::Passed,
                EvaluationGateStatus::Inapplicable,
            ]),
            EvaluationGateStatus::Inapplicable
        );
        assert_eq!(
            aggregate_evaluation_gate_status([
                EvaluationGateStatus::Inapplicable,
                EvaluationGateStatus::Failed,
            ]),
            EvaluationGateStatus::Failed
        );
    }

    #[test]
    fn unrestorable_static_162_gate_is_not_passed() {
        // Mirrors eval-suite behavior: static_162 check marked Inapplicable while
        // other checks may still Pass — overall must not report Passed.
        let status = aggregate_evaluation_gate_status([
            EvaluationGateStatus::Passed,       // full_180
            EvaluationGateStatus::Inapplicable, // static_162 unrestorable
        ]);
        assert_ne!(status, EvaluationGateStatus::Passed);
        assert_eq!(status, EvaluationGateStatus::Inapplicable);
    }

    #[test]
    fn selection_report_serializes_full_stable_ids_and_fingerprint() {
        let source = dataset_with_ids(&[10, 20, 30, 40, 50]);
        let selection = FrameSelection::select(
            &source,
            &FrameSelectionRequest {
                include_ranges: vec![FrameIdRange { start: 20, end: 40 }],
                max_frames: 2,
                frame_stride: 1,
                ..Default::default()
            },
        )
        .unwrap();
        let report = FrameSelectionReport::from_selection(
            &selection,
            Some("manifest-fp".into()),
            Some("holdout".into()),
        );
        assert_eq!(report.stable_frame_ids, vec![20, 30]);
        assert_eq!(
            report.selection_fingerprint,
            selection.selection_fingerprint
        );
        assert_eq!(report.max_frames, 2);
        assert_eq!(report.frame_stride, 1);
        assert_eq!(report.include_frame_ranges.as_deref(), Some("20-40"));
        assert_eq!(report.manifest_fingerprint.as_deref(), Some("manifest-fp"));
        assert_eq!(report.eval_split_kind.as_deref(), Some("holdout"));

        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(
            json["stable_frame_ids"],
            serde_json::json!([20, 30]),
            "report must retain full stable IDs"
        );
        assert_eq!(
            json["selection_fingerprint"],
            serde_json::json!(selection.selection_fingerprint)
        );
        assert_eq!(json["max_frames"], serde_json::json!(2));
        assert_eq!(json["frame_stride"], serde_json::json!(1));
    }

    #[test]
    fn evaluate_psnr_and_suite_selection_reports_record_in_view_split() {
        // evaluate_psnr / rustgs_eval_suite both use from_in_view_selection.
        let source = dataset_with_ids(&[1, 2, 3]);
        let selection = FrameSelection::select(
            &source,
            &FrameSelectionRequest {
                max_frames: 2,
                frame_stride: 1,
                ..Default::default()
            },
        )
        .unwrap();
        let report = FrameSelectionReport::from_in_view_selection(&selection, None);
        assert_eq!(report.eval_split_kind.as_deref(), Some("in-view"));
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["eval_split_kind"], serde_json::json!("in-view"));
        let markdown_line = format!(
            "eval_split_kind: {}",
            report.eval_split_kind.as_deref().unwrap_or("-")
        );
        assert!(
            markdown_line.contains("in-view"),
            "suite markdown must surface in-view split: {markdown_line}"
        );
    }

    #[test]
    fn fingerprint_is_independently_recomputable_from_request_fields() {
        let source = dataset_with_ids(&[1, 2, 3, 4]);
        let request = FrameSelectionRequest {
            exclude_ranges: vec![FrameIdRange { start: 2, end: 2 }],
            max_frames: 3,
            frame_stride: 1,
            allowed_ids: Some(vec![1, 2, 3, 4]),
            ..Default::default()
        };
        let selection = FrameSelection::select(&source, &request).unwrap();
        let recomputed = fingerprint_frame_selection(&selection.stable_ids, &selection.request());
        assert_eq!(recomputed, selection.selection_fingerprint);
    }
}
