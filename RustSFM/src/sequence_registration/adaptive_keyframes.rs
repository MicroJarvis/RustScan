use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::path::Path;

use anyhow::Context;
use serde::{Deserialize, Serialize};

use super::{
    database_features_exist, database_image_ids_for_indices, import_database_images,
    link_or_copy_stable_image, sequence_match_options, validate_runner_inputs, SequenceFrame,
};
use crate::database::ColmapDatabase;
use crate::feature_extraction::extract_selected_features_to_database_with_task;
use crate::feature_matching_db::{
    match_explicit_image_pairs_to_database_with_session, ExplicitPairMatchingSession,
};
use crate::mapper::{FeatureType, MapperConfig};
use crate::task::{SfmTaskContext, SfmTaskEvent, SfmTaskEventKind, SfmTaskOperation, SfmTaskStage};

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
    #[error("frame {frame_id} has no usable SIFT feature evidence")]
    MissingFeatureEvidence { frame_id: u32 },
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
    select_adaptive_keyframes_with(
        frame_ids,
        config,
        |anchor_frame_id, candidate_frame_id| {
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
        },
        |_, _, _| {},
    )
}

fn select_adaptive_keyframes_with<E, F, S>(
    frame_ids: &[u32],
    config: &AdaptiveKeyframeSelectionConfig,
    mut lookup: F,
    mut on_selection: S,
) -> Result<AdaptiveKeyframeSelectionResult, E>
where
    E: From<AdaptiveKeyframeSelectionError>,
    F: FnMut(u32, u32) -> Result<AdaptiveKeyframePairMetrics, E>,
    S: FnMut(AdaptiveKeyframeSelectionDecision, u32, usize),
{
    config.validate().map_err(E::from)?;
    validate_frame_ids(frame_ids).map_err(E::from)?;

    let mut selected_frame_ids = vec![frame_ids[0]];
    on_selection(
        AdaptiveKeyframeSelectionDecision::Boundary,
        frame_ids[0],
        selected_frame_ids.len(),
    );
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
        validate_pair_metrics(&metrics, anchor_frame_id, candidate_frame_id).map_err(E::from)?;

        let connected = is_geometrically_connected(&metrics, config);
        let decision = if metrics.feature_coverage >= config.retention_feature_coverage {
            if connected {
                bridge_index = Some(candidate_index);
            }
            candidate_index += 1;
            AdaptiveKeyframeSelectionDecision::Redundant
        } else if connected {
            selected_frame_ids.push(candidate_frame_id);
            on_selection(
                AdaptiveKeyframeSelectionDecision::ConnectedTransition,
                candidate_frame_id,
                selected_frame_ids.len(),
            );
            anchor_index = candidate_index;
            candidate_index += 1;
            bridge_index = None;
            AdaptiveKeyframeSelectionDecision::ConnectedTransition
        } else if let Some(bridge_frame_index) = bridge_index {
            selected_frame_ids.push(frame_ids[bridge_frame_index]);
            on_selection(
                AdaptiveKeyframeSelectionDecision::ConnectivityBridge,
                frame_ids[bridge_frame_index],
                selected_frame_ids.len(),
            );
            anchor_index = bridge_frame_index;
            bridge_index = None;
            AdaptiveKeyframeSelectionDecision::ConnectivityBridge
        } else {
            selected_frame_ids.push(candidate_frame_id);
            on_selection(
                AdaptiveKeyframeSelectionDecision::ForcedProgress,
                candidate_frame_id,
                selected_frame_ids.len(),
            );
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
        on_selection(
            AdaptiveKeyframeSelectionDecision::Boundary,
            boundary_frame_id,
            selected_frame_ids.len(),
        );
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

pub fn run_adaptive_keyframe_selection(
    frames: &[SequenceFrame],
    config: &AdaptiveKeyframeSelectionConfig,
    mapper_config: &MapperConfig,
    output: &Path,
    task: &mut SfmTaskContext<'_>,
) -> anyhow::Result<AdaptiveKeyframeSelectionResult> {
    config.validate().map_err(anyhow::Error::new)?;
    if frames.len() < 2 {
        return Err(anyhow::Error::new(
            AdaptiveKeyframeSelectionError::InsufficientFrames {
                usable_frames: frames.len(),
            },
        ));
    }
    if mapper_config.feature_type != FeatureType::Sift {
        anyhow::bail!("adaptive keyframe selection requires SIFT features");
    }
    let frame_ids = frames.iter().map(|frame| frame.id).collect::<Vec<_>>();
    let frame_indices = validate_runner_inputs(frames, &frame_ids)?;
    task.checkpoint().map_err(anyhow::Error::new)?;
    task.emit(selection_event(
        SfmTaskOperation::Begin,
        SfmTaskEventKind::Started,
        Some(0),
        Some(frames.len()),
        None,
        None,
    ));

    let cache = output.join("Cache");
    let sequence_input = cache.join("sequence");
    let database = cache.join("database.db");
    std::fs::create_dir_all(&sequence_input)?;
    for frame in frames {
        link_or_copy_stable_image(&frame.image_path, &sequence_input)?;
    }
    import_database_images(frames, &frame_indices, mapper_config, &database)?;
    let database_ids = database_image_ids_for_indices(frames, &frame_indices, &database)?;

    let mut sift_extraction = mapper_config.sift_extraction.clone();
    sift_extraction.max_num_features = mapper_config.max_features;
    let missing_feature_ids = database_ids
        .iter()
        .copied()
        .map(|image_id| Ok((image_id, database_features_exist(&database, image_id)?)))
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .filter_map(|(image_id, exists)| (!exists).then_some(image_id))
        .collect::<Vec<_>>();
    if !missing_feature_ids.is_empty() {
        task.checkpoint().map_err(anyhow::Error::new)?;
        extract_selected_features_to_database_with_task(
            &database,
            &sequence_input,
            &sift_extraction,
            &missing_feature_ids,
            task,
        )?;
    }

    let database_reader = ColmapDatabase::open_read_only(&database)?;
    let feature_counts = database_ids
        .iter()
        .map(|&image_id| database_reader.num_keypoints_for_image(image_id))
        .collect::<anyhow::Result<Vec<_>>>()?;
    drop(database_reader);
    let usable = frames
        .iter()
        .zip(database_ids.iter().copied())
        .zip(feature_counts.iter().copied())
        .filter_map(|((frame, database_id), feature_count)| {
            (feature_count > 0).then_some((frame.id, database_id, feature_count))
        })
        .collect::<Vec<_>>();
    if usable.len() < 2 {
        return Err(anyhow::Error::new(
            AdaptiveKeyframeSelectionError::InsufficientFrames {
                usable_frames: usable.len(),
            },
        ));
    }
    for boundary in [
        frames.first().expect("non-empty"),
        frames.last().expect("non-empty"),
    ] {
        if !usable
            .iter()
            .any(|(frame_id, _, _)| *frame_id == boundary.id)
        {
            return Err(anyhow::Error::new(
                AdaptiveKeyframeSelectionError::MissingFeatureEvidence {
                    frame_id: boundary.id,
                },
            ));
        }
    }

    let usable_frame_ids = usable
        .iter()
        .map(|(frame_id, _, _)| *frame_id)
        .collect::<Vec<_>>();
    let mut match_options = sequence_match_options(mapper_config);
    match_options.task_pair_batch_size = 1;
    // Selection needs a zero-valued metric report when descriptor overlap is
    // already lost; geometry acceptance still uses the configured thresholds.
    match_options.min_num_matches = 0;
    let session = ExplicitPairMatchingSession::new(&match_options)?;
    let task_cell = RefCell::new(task);
    let selected_count = Cell::new(0usize);
    let mut result = select_adaptive_keyframes_with::<anyhow::Error, _, _>(
        &usable_frame_ids,
        config,
        |anchor_frame_id, candidate_frame_id| {
            let (_, anchor_database_id, anchor_features) = usable
                .iter()
                .find(|(frame_id, _, _)| *frame_id == anchor_frame_id)
                .copied()
                .context("adaptive selection anchor is not usable")?;
            let (_, candidate_database_id, candidate_features) = usable
                .iter()
                .find(|(frame_id, _, _)| *frame_id == candidate_frame_id)
                .copied()
                .context("adaptive selection candidate is not usable")?;
            let mut task = task_cell.borrow_mut();
            task.checkpoint().map_err(anyhow::Error::new)?;
            let report = match_explicit_image_pairs_to_database_with_session(
                &database,
                &[(anchor_database_id, candidate_database_id)],
                &match_options,
                &session,
                &mut task,
            )?;
            let pair = report
                .pairs
                .into_iter()
                .next()
                .context("explicit adaptive pair produced no report")?;
            let inlier_ratio = if pair.num_matches == 0 {
                0.0
            } else {
                pair.num_inliers as f64 / pair.num_matches as f64
            };
            let feature_coverage =
                pair.num_inliers as f64 / anchor_features.min(candidate_features) as f64;
            task.emit(selection_event(
                SfmTaskOperation::EvaluateKeyframePair,
                SfmTaskEventKind::Progress,
                Some(selected_count.get()),
                Some(frames.len()),
                Some((anchor_frame_id, candidate_frame_id)),
                None,
            ));
            Ok(AdaptiveKeyframePairMetrics {
                anchor_frame_id,
                candidate_frame_id,
                descriptor_matches: pair.num_matches,
                inliers: pair.num_inliers,
                triangulated: pair.triangulated,
                inlier_ratio,
                feature_coverage,
            })
        },
        |decision, frame_id, current_selected_count| {
            selected_count.set(current_selected_count);
            task_cell.borrow_mut().emit(selection_event(
                SfmTaskOperation::SelectKeyframe,
                SfmTaskEventKind::Progress,
                Some(current_selected_count),
                Some(frames.len()),
                None,
                Some(format!("{decision:?}: selected frame {frame_id}")),
            ));
        },
    )?;
    result.imported_frames = frames.len();
    result.usable_frames = usable.len();
    task_cell.borrow_mut().emit(selection_event(
        SfmTaskOperation::Complete,
        SfmTaskEventKind::Completed,
        Some(result.selected_frame_ids.len()),
        Some(frames.len()),
        None,
        None,
    ));
    Ok(result)
}

fn selection_event(
    operation: SfmTaskOperation,
    kind: SfmTaskEventKind,
    completed: Option<usize>,
    total: Option<usize>,
    pair: Option<(u32, u32)>,
    message: Option<String>,
) -> SfmTaskEvent {
    SfmTaskEvent {
        sequence: 0,
        elapsed_ms: 0,
        stage: SfmTaskStage::KeyframeSelection,
        operation,
        kind,
        completed,
        total,
        registered_images: None,
        sparse_points: None,
        image_id: None,
        pair,
        message,
        issue: None,
    }
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
