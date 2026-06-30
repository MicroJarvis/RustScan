use crate::types::Reconstruction;
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
