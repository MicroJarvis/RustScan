use crate::task::{SfmTaskContext, SfmTaskEvent, SfmTaskEventKind, SfmTaskOperation, SfmTaskStage};
use crate::types::Reconstruction;
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IncrementalPipelineStatus {
    Success,
    NoInitialPair,
    BadInitialPair,
    NoModelsKept,
}

#[derive(Debug, Clone)]
pub struct IncrementalPipelineResult {
    pub status: IncrementalPipelineStatus,
    pub reconstructions: Vec<Reconstruction>,
    pub debug_log: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct IncrementalPipelineMapResult {
    pub reconstructions: Vec<Reconstruction>,
    pub debug_log: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IncrementalPipelineCallback {
    InitialImagePairReg,
    NextImageReg,
    LastImageReg,
}

impl IncrementalPipelineCallback {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InitialImagePairReg => "initial_image_pair_reg",
            Self::NextImageReg => "next_image_reg",
            Self::LastImageReg => "last_image_reg",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineCallbackEvent {
    pub callback: IncrementalPipelineCallback,
    pub model_index: usize,
    pub registered_images: usize,
    pub registered_frames: usize,
    pub points: usize,
}

pub trait PipelineCallbackSink {
    fn on_pipeline_callback(&mut self, event: &PipelineCallbackEvent);
}

pub(super) enum MapperEventBridge<'bridge, 'task> {
    Silent,
    Legacy(&'bridge mut dyn PipelineCallbackSink),
    Task(&'bridge mut SfmTaskContext<'task>),
}

impl MapperEventBridge<'_, '_> {
    pub(super) fn callback(&mut self, event: PipelineCallbackEvent) {
        match self {
            Self::Silent => {}
            Self::Legacy(sink) => sink.on_pipeline_callback(&event),
            Self::Task(task) => {
                let operation = match event.callback {
                    IncrementalPipelineCallback::InitialImagePairReg => {
                        SfmTaskOperation::RegisterInitialPair
                    }
                    IncrementalPipelineCallback::NextImageReg
                    | IncrementalPipelineCallback::LastImageReg => SfmTaskOperation::RegisterImage,
                };
                task.emit(SfmTaskEvent {
                    sequence: 0,
                    elapsed_ms: 0,
                    stage: SfmTaskStage::IncrementalMapping,
                    operation,
                    kind: SfmTaskEventKind::Progress,
                    completed: Some(event.registered_images),
                    total: None,
                    registered_images: Some(event.registered_images),
                    sparse_points: Some(event.points),
                    image_id: None,
                    pair: None,
                    message: Some(format!(
                        "model={} registered_frames={}",
                        event.model_index, event.registered_frames
                    )),
                    issue: None,
                });
            }
        }
    }

    pub(super) fn emit_operation(
        &mut self,
        stage: SfmTaskStage,
        operation: SfmTaskOperation,
        kind: SfmTaskEventKind,
    ) {
        if let Self::Task(task) = self {
            task.emit(SfmTaskEvent {
                sequence: 0,
                elapsed_ms: 0,
                stage,
                operation,
                kind,
                completed: None,
                total: None,
                registered_images: None,
                sparse_points: None,
                image_id: None,
                pair: None,
                message: None,
                issue: None,
            });
        }
    }

    pub(super) fn checkpoint(&self) -> Result<()> {
        match self {
            Self::Task(task) => task.checkpoint().map_err(anyhow::Error::new),
            Self::Silent | Self::Legacy(_) => Ok(()),
        }
    }

    pub(super) fn is_task(&self) -> bool {
        matches!(self, Self::Task(_))
    }
}
