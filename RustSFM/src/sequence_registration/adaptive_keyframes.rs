use std::collections::HashSet;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveKeyframeSelectionConfig {
    pub retention_feature_coverage: f64,
    pub min_inliers: usize,
    pub min_inlier_ratio: f64,
    pub min_triangulated: usize,
}

impl Default for AdaptiveKeyframeSelectionConfig {
    fn default() -> Self {
        Self {
            retention_feature_coverage: 0.35,
            min_inliers: 15,
            min_inlier_ratio: 0.20,
            min_triangulated: 4,
        }
    }
}

impl AdaptiveKeyframeSelectionConfig {
    pub fn validate(&self) -> Result<(), AdaptiveKeyframeSelectionError> {
        validate_unit_interval(
            self.retention_feature_coverage,
            "retention_feature_coverage",
        )?;
        if self.min_inliers == 0 {
            return Err(AdaptiveKeyframeSelectionError::InvalidConfigMetric {
                field: "min_inliers",
            });
        }
        validate_unit_interval(self.min_inlier_ratio, "min_inlier_ratio")?;
        if self.min_triangulated == 0 {
            return Err(AdaptiveKeyframeSelectionError::InvalidConfigMetric {
                field: "min_triangulated",
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveKeyframePairMetrics {
    pub anchor_frame_id: u32,
    pub candidate_frame_id: u32,
    pub descriptor_matches: usize,
    pub inliers: usize,
    pub triangulated: usize,
    pub inlier_ratio: f64,
    pub feature_coverage: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdaptiveKeyframeSelectionDecision {
    Redundant,
    ConnectedTransition,
    ConnectivityBridge,
    ForcedProgress,
    Boundary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveKeyframePairDiagnostic {
    pub metrics: AdaptiveKeyframePairMetrics,
    pub decision: AdaptiveKeyframeSelectionDecision,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveKeyframeSelectionResult {
    pub imported_frames: usize,
    pub usable_frames: usize,
    pub selected_frame_ids: Vec<u32>,
    pub config: AdaptiveKeyframeSelectionConfig,
    pub evaluated_pairs: usize,
    pub diagnostics: Vec<AdaptiveKeyframePairDiagnostic>,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum AdaptiveKeyframeSelectionError {
    #[error("invalid adaptive keyframe configuration field {field}")]
    InvalidConfigMetric { field: &'static str },
    #[error(
        "adaptive keyframe selection requires at least two usable frames, found {usable_frames}"
    )]
    InsufficientFrames { usable_frames: usize },
    #[error("adaptive keyframe selection contains duplicate frame id {frame_id}")]
    DuplicateFrameId { frame_id: u32 },
    #[error(
        "missing pair evidence for anchor {anchor_frame_id} and candidate {candidate_frame_id}"
    )]
    MissingPairEvidence {
        anchor_frame_id: u32,
        candidate_frame_id: u32,
    },
    #[error(
        "pair evidence for {anchor_frame_id}-{candidate_frame_id} contains non-finite {field}"
    )]
    NonFinitePairMetric {
        anchor_frame_id: u32,
        candidate_frame_id: u32,
        field: &'static str,
    },
}

pub fn select_adaptive_keyframes_from_metrics(
    frame_ids: &[u32],
    evidence: &[AdaptiveKeyframePairMetrics],
    config: &AdaptiveKeyframeSelectionConfig,
) -> Result<AdaptiveKeyframeSelectionResult, AdaptiveKeyframeSelectionError> {
    select_adaptive_keyframes_with(frame_ids, config, |anchor_frame_id, candidate_frame_id| {
        evidence
            .iter()
            .find(|metrics| {
                metrics.anchor_frame_id == anchor_frame_id
                    && metrics.candidate_frame_id == candidate_frame_id
            })
            .cloned()
            .ok_or(AdaptiveKeyframeSelectionError::MissingPairEvidence {
                anchor_frame_id,
                candidate_frame_id,
            })
    })
}

fn select_adaptive_keyframes_with<F>(
    frame_ids: &[u32],
    config: &AdaptiveKeyframeSelectionConfig,
    mut lookup: F,
) -> Result<AdaptiveKeyframeSelectionResult, AdaptiveKeyframeSelectionError>
where
    F: FnMut(u32, u32) -> Result<AdaptiveKeyframePairMetrics, AdaptiveKeyframeSelectionError>,
{
    config.validate()?;
    validate_frame_ids(frame_ids)?;

    let mut selected_frame_ids = vec![frame_ids[0]];
    let mut diagnostics = Vec::new();
    let mut evaluated_pairs = 0;
    let mut anchor_index = 0;
    let mut candidate_index = 1;
    let mut bridge_index = None;

    while candidate_index < frame_ids.len() {
        let anchor_frame_id = frame_ids[anchor_index];
        let candidate_frame_id = frame_ids[candidate_index];
        let metrics = lookup(anchor_frame_id, candidate_frame_id)?;
        evaluated_pairs += 1;
        validate_pair_metrics(&metrics, anchor_frame_id, candidate_frame_id)?;

        let connected = is_geometrically_connected(&metrics, config);
        let decision = if metrics.feature_coverage >= config.retention_feature_coverage {
            if connected {
                bridge_index = Some(candidate_index);
            }
            candidate_index += 1;
            AdaptiveKeyframeSelectionDecision::Redundant
        } else if connected {
            selected_frame_ids.push(candidate_frame_id);
            anchor_index = candidate_index;
            candidate_index += 1;
            bridge_index = None;
            AdaptiveKeyframeSelectionDecision::ConnectedTransition
        } else if let Some(bridge_frame_index) = bridge_index {
            selected_frame_ids.push(frame_ids[bridge_frame_index]);
            anchor_index = bridge_frame_index;
            bridge_index = None;
            AdaptiveKeyframeSelectionDecision::ConnectivityBridge
        } else {
            selected_frame_ids.push(candidate_frame_id);
            anchor_index = candidate_index;
            candidate_index += 1;
            bridge_index = None;
            AdaptiveKeyframeSelectionDecision::ForcedProgress
        };

        diagnostics.push(AdaptiveKeyframePairDiagnostic { metrics, decision });
    }

    let boundary_frame_id = *frame_ids.last().expect("validated frame IDs are non-empty");
    if selected_frame_ids.last() != Some(&boundary_frame_id) {
        selected_frame_ids.push(boundary_frame_id);
    }

    Ok(AdaptiveKeyframeSelectionResult {
        imported_frames: frame_ids.len(),
        usable_frames: frame_ids.len(),
        selected_frame_ids,
        config: config.clone(),
        evaluated_pairs,
        diagnostics,
    })
}

fn validate_unit_interval(
    value: f64,
    field: &'static str,
) -> Result<(), AdaptiveKeyframeSelectionError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) || value == 0.0 {
        return Err(AdaptiveKeyframeSelectionError::InvalidConfigMetric { field });
    }
    Ok(())
}

fn validate_frame_ids(frame_ids: &[u32]) -> Result<(), AdaptiveKeyframeSelectionError> {
    if frame_ids.len() < 2 {
        return Err(AdaptiveKeyframeSelectionError::InsufficientFrames {
            usable_frames: frame_ids.len(),
        });
    }

    let mut unique_ids = HashSet::with_capacity(frame_ids.len());
    for &frame_id in frame_ids {
        if !unique_ids.insert(frame_id) {
            return Err(AdaptiveKeyframeSelectionError::DuplicateFrameId { frame_id });
        }
    }
    Ok(())
}

fn validate_pair_metrics(
    metrics: &AdaptiveKeyframePairMetrics,
    anchor_frame_id: u32,
    candidate_frame_id: u32,
) -> Result<(), AdaptiveKeyframeSelectionError> {
    if metrics.anchor_frame_id != anchor_frame_id
        || metrics.candidate_frame_id != candidate_frame_id
    {
        return Err(AdaptiveKeyframeSelectionError::MissingPairEvidence {
            anchor_frame_id,
            candidate_frame_id,
        });
    }
    if !metrics.inlier_ratio.is_finite() {
        return Err(AdaptiveKeyframeSelectionError::NonFinitePairMetric {
            anchor_frame_id,
            candidate_frame_id,
            field: "inlier_ratio",
        });
    }
    if !metrics.feature_coverage.is_finite() {
        return Err(AdaptiveKeyframeSelectionError::NonFinitePairMetric {
            anchor_frame_id,
            candidate_frame_id,
            field: "feature_coverage",
        });
    }
    Ok(())
}

fn is_geometrically_connected(
    metrics: &AdaptiveKeyframePairMetrics,
    config: &AdaptiveKeyframeSelectionConfig,
) -> bool {
    metrics.inliers >= config.min_inliers
        && metrics.inlier_ratio >= config.min_inlier_ratio
        && metrics.triangulated >= config.min_triangulated
}
