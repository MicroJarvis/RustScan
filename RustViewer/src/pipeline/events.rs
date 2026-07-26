use std::sync::Arc;

use crate::project::{ProjectErrorRecord, ProjectManifest, ProjectStage};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineCommand {
    StartAutomatic,
    StartThrough {
        stage: ProjectStage,
    },
    Pause,
    Cancel,
    Retry {
        stage: ProjectStage,
    },
    RestartFrom {
        stage: ProjectStage,
        confirmed: bool,
    },
    Shutdown {
        disposition: ShutdownDisposition,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownDisposition {
    PauseAndQuit,
    CancelAndQuit,
    KeepRunning,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PipelineProgressDetail {
    Media {
        frame_id: Option<u32>,
    },
    Sfm {
        operation: String,
        image_id: Option<u32>,
        pair: Option<(u32, u32)>,
        registered_images: Option<usize>,
        sparse_points: Option<usize>,
    },
    Training {
        iteration: usize,
        loss: f32,
        gaussian_count: usize,
        elapsed_ms: u64,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum PipelineEvent {
    ManifestChanged(ProjectManifest),
    StageProgress {
        stage: ProjectStage,
        completed: Option<u64>,
        total: Option<u64>,
        detail: PipelineProgressDetail,
    },
    SceneSnapshot(Arc<rustgs::HostSplats>),
    NeedsAttention(ProjectErrorRecord),
    Idle,
}
