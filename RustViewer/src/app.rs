//! Main application struct implementing eframe::App.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui::{self, Color32, Rect, Vec2};
use eframe::egui_wgpu;
use eframe::wgpu;
use glam::{Vec3, Vec4};
use rustgs::{ColmapConfig, HostSplats, SharedWgpuContext, TrainingConfig};

use crate::loader::checkpoint::LoadError;
use crate::loader::{
    load_colmap_training_dataset, map_training_dataset_to_scene, LoadedColmapDataset,
};
use crate::media::{
    import_image_sequence, ImageSequenceImportRequest, MediaEventSink, MediaImportEvent,
};
use crate::pipeline::{
    ImportWorker, PipelineCommand, PipelineCoordinator, PipelineEvent, PipelineProgressDetail,
    PipelineWorkers, RustSfmWorker, StageRequest, TrainingWorker, WorkerControl, WorkerEventSink,
    WorkerOutcome,
};
use crate::project::{
    ProjectCreateRequest, ProjectErrorRecord, ProjectSessionSummary, ProjectStage, ProjectStore,
    SourceSpec, SuggestedAction,
};
use crate::renderer::camera::ArcballCamera;
use crate::renderer::scene::{GaussianSplat, Scene};
use crate::renderer::ViewerCallback;
use crate::robot::{GroundPlane, NavigationMode, RobotController, RobotInput};
use crate::training::gpu_viewport::{viewport_render_scale, GpuViewportBridge};
use crate::training::preview::PreviewResolution;
use crate::training::{TrainingControlOptions, TrainingManager, TrainingSessionEvent};
use crate::ui::panel::{DatasetUiSummary, PanelAction, UiState};
use crate::ui::theme::{PANEL_BG, TEXT_PRIMARY, TEXT_SECONDARY, VIEWPORT_BG, WINDOW_BG};
use crate::ui::workbench::{
    self, PipelineStageState, WorkbenchActivity, WorkbenchLayout, WorkbenchSnapshot,
};

const VIEWPORT_INTERACTIVE_IDLE_DELAY: Duration = Duration::from_millis(180);
const GROUND_PICK_POINT_COUNT: usize = 4;
const MAX_WORKBENCH_ACTIVITY_ENTRIES: usize = 4;
const IMAGE_FILE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CachedTexture {
    id: egui::TextureId,
    resolution: PreviewResolution,
}

#[derive(Debug, Clone, Copy)]
enum AssetLoadKind {
    Checkpoint,
    Gaussian,
    Mesh,
}

#[derive(Debug, Clone, Copy)]
enum ImageImportSource {
    Files,
    Folder,
}

enum AppCommand {
    LoadAsset { kind: AssetLoadKind, path: PathBuf },
    ColmapLoaded(Result<LoadedColmapDataset, String>),
    ImageSequenceImported(Result<ProjectImportSummary, String>),
    ImageSequenceImportCancelled,
}

#[derive(Debug, Clone)]
struct ProjectImportSummary {
    path: PathBuf,
}

#[derive(Debug, Clone)]
enum ImageImportInput {
    Files(Vec<PathBuf>),
    Folder(PathBuf),
}

#[derive(Debug, Clone)]
struct ImageImportRequest {
    input: ImageImportInput,
    destination: PathBuf,
}

fn image_import_request(
    input: Option<ImageImportInput>,
    destination: Option<PathBuf>,
) -> Option<ImageImportRequest> {
    Some(ImageImportRequest {
        input: input?,
        destination: destination?,
    })
}

fn send_image_import_completion(
    tx: &mpsc::Sender<AppCommand>,
    ctx: &egui::Context,
    result: Result<ProjectImportSummary, String>,
) {
    if tx.send(AppCommand::ImageSequenceImported(result)).is_ok() {
        ctx.request_repaint();
    }
}

fn send_image_import_cancellation(tx: &mpsc::Sender<AppCommand>, ctx: &egui::Context) {
    if tx.send(AppCommand::ImageSequenceImportCancelled).is_ok() {
        ctx.request_repaint();
    }
}

struct DiscardMediaEvents;

impl MediaEventSink for DiscardMediaEvents {
    fn on_media_event(&mut self, _event: MediaImportEvent) {}
}

#[derive(Debug, Default, Clone, Copy)]
struct DisabledProjectStageWorker;

impl DisabledProjectStageWorker {
    fn outcome(stage: ProjectStage) -> WorkerOutcome {
        WorkerOutcome::Failed(ProjectErrorRecord {
            code: "unsupported_project_stage".to_owned(),
            stage,
            summary: "Project pipeline stage is not available".to_owned(),
            detail: "ViewerApp only runs RustSFM reconstruction stages through FullFramePnp."
                .to_owned(),
            frame_id: None,
            pair: None,
            retryable: false,
            suggested_actions: vec![SuggestedAction::OpenLog],
        })
    }
}

impl ImportWorker for DisabledProjectStageWorker {
    fn run(
        &self,
        request: StageRequest,
        _control: WorkerControl,
        _events: WorkerEventSink,
    ) -> WorkerOutcome {
        Self::outcome(request.stage)
    }
}

impl TrainingWorker for DisabledProjectStageWorker {
    fn run(
        &self,
        request: StageRequest,
        _control: WorkerControl,
        _events: WorkerEventSink,
    ) -> WorkerOutcome {
        Self::outcome(request.stage)
    }
}

#[derive(Debug, Default)]
struct GroundPickState {
    points: Vec<Vec3>,
}

pub struct ViewerApp {
    scene: Arc<Mutex<Scene>>,
    camera: ArcballCamera,
    robot: RobotController,
    ui_state: UiState,
    loaded_colmap: Option<LoadedColmapDataset>,
    project_summary: Option<ProjectSessionSummary>,
    project_path: Option<PathBuf>,
    project_pipeline: Option<PipelineCoordinator>,
    activity: VecDeque<WorkbenchActivity>,
    command_rx: mpsc::Receiver<AppCommand>,
    command_tx: mpsc::Sender<AppCommand>,
    training_manager: TrainingManager,
    loaded_splats: Option<Arc<HostSplats>>,
    viewport_bridge: Option<GpuViewportBridge>,
    viewport_dirty: bool,
    viewport_texture: Option<CachedTexture>,
    viewport_last_motion: Option<Instant>,
    ground_pick: Option<GroundPickState>,
    viewport_render_error: Option<String>,
    shared_wgpu_context: Option<SharedWgpuContext>,
    wgpu_render_state: Option<egui_wgpu::RenderState>,
    /// Actual wgpu surface format read from eframe at startup.
    surface_format: wgpu::TextureFormat,
}

impl ViewerApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        Self::new_with_startup_asset(cc, None)
    }

    pub fn new_with_startup_asset(
        cc: &eframe::CreationContext<'_>,
        startup_asset: Option<PathBuf>,
    ) -> Self {
        let (command_tx, command_rx) = mpsc::channel();
        let surface_format = cc
            .wgpu_render_state
            .as_ref()
            .map(|rs| rs.target_format)
            .unwrap_or(wgpu::TextureFormat::Bgra8Unorm);
        let shared_wgpu_context = cc
            .wgpu_render_state
            .as_ref()
            .map(shared_wgpu_context_from_render_state);
        let viewport_bridge =
            new_gpu_viewport_bridge(&shared_wgpu_context, cc.wgpu_render_state.as_ref());
        let app = Self {
            scene: Arc::new(Mutex::new(Scene::default())),
            camera: ArcballCamera::default(),
            robot: RobotController::default(),
            ui_state: UiState::default(),
            loaded_colmap: None,
            project_summary: None,
            project_path: None,
            project_pipeline: None,
            activity: VecDeque::new(),
            command_rx,
            command_tx,
            training_manager: TrainingManager::new(),
            loaded_splats: None,
            viewport_bridge,
            viewport_dirty: true,
            viewport_texture: None,
            viewport_last_motion: None,
            ground_pick: None,
            viewport_render_error: None,
            shared_wgpu_context,
            wgpu_render_state: cc.wgpu_render_state.clone(),
            surface_format,
        };

        if let Some(path) = startup_asset {
            let _ = app.command_tx.send(AppCommand::LoadAsset {
                kind: startup_asset_kind(&path),
                path,
            });
        }

        app
    }

    fn poll_commands(&mut self) {
        while let Ok(command) = self.command_rx.try_recv() {
            match command {
                AppCommand::LoadAsset { kind, path } => self.handle_asset_load(kind, path),
                AppCommand::ColmapLoaded(result) => self.handle_colmap_loaded(result),
                AppCommand::ImageSequenceImported(result) => {
                    self.handle_image_sequence_import(result)
                }
                AppCommand::ImageSequenceImportCancelled => {
                    self.ui_state.is_loading = false;
                    self.ui_state.loading_message = None;
                    self.record_activity(
                        "Import",
                        "Image sequence import cancelled",
                        PipelineStageState::Waiting,
                    );
                }
            }
        }
    }

    fn handle_asset_load(&mut self, kind: AssetLoadKind, path: PathBuf) {
        let mut next_loaded_splats = None;
        let mut load_succeeded = false;

        if let Ok(mut scene) = self.scene.lock() {
            clear_scene_preserving_layers(&mut scene);

            let result: Result<(), LoadError> = match kind {
                AssetLoadKind::Checkpoint => {
                    crate::loader::checkpoint::load_checkpoint(&path, &mut scene)
                }
                AssetLoadKind::Gaussian => {
                    match crate::loader::gaussian::load_gaussians_with_splats(&path, &mut scene) {
                        Ok(splats) => {
                            next_loaded_splats = splats.map(Arc::new);
                            Ok(())
                        }
                        Err(err) => Err(err),
                    }
                }
                AssetLoadKind::Mesh => crate::loader::mesh::load_mesh(&path, &mut scene),
            };

            match result {
                Ok(()) => {
                    load_succeeded = true;
                    self.ui_state.load_error = None;
                    if scene.has_data() {
                        self.camera.fit_scene(&scene.bounds);
                        self.robot.reset_to_scene(&scene.bounds);
                    } else {
                        scene.recompute_bounds();
                    }
                }
                Err(err) => {
                    clear_scene_preserving_layers(&mut scene);
                    self.ui_state.load_error = Some(err.to_string());
                }
            }
        }

        self.loaded_splats = load_succeeded.then_some(next_loaded_splats).flatten();
        if load_succeeded {
            self.project_summary = None;
            self.project_path = None;
            self.project_pipeline = None;
        }
        self.viewport_bridge = self.new_gpu_viewport_bridge();
        self.viewport_texture = None;
        self.viewport_dirty = true;
        self.viewport_last_motion = None;
        self.viewport_render_error = None;
    }

    fn handle_colmap_loaded(&mut self, result: Result<LoadedColmapDataset, String>) {
        self.ui_state.is_loading = false;
        self.ui_state.loading_message = None;

        match result {
            Ok(loaded) => {
                let frame_count = loaded.summary.frame_count;
                if let Ok(mut scene) = self.scene.lock() {
                    clear_scene_preserving_layers(&mut scene);
                    map_training_dataset_to_scene(&loaded.dataset, &mut scene);
                    if scene.has_data() {
                        self.camera.fit_scene(&scene.bounds);
                        self.robot.reset_to_scene(&scene.bounds);
                    }
                }

                self.ui_state.load_error = None;
                self.ui_state.dataset_summary = Some(DatasetUiSummary {
                    root_path: loaded.summary.input_dir.display().to_string(),
                    frame_count: loaded.summary.frame_count,
                    sparse_point_count: loaded.summary.sparse_point_count,
                    width: loaded.summary.intrinsics.width,
                    height: loaded.summary.intrinsics.height,
                });
                self.ui_state.training_error = None;
                self.ui_state.preview_error = None;
                self.loaded_colmap = Some(loaded);
                self.project_summary = None;
                self.project_path = None;
                self.project_pipeline = None;
                self.loaded_splats = None;
                self.viewport_bridge = self.new_gpu_viewport_bridge();
                self.viewport_texture = None;
                self.viewport_dirty = true;
                self.viewport_last_motion = None;
                self.viewport_render_error = None;
                self.record_activity(
                    "Dataset",
                    format!("{frame_count} COLMAP frames loaded"),
                    PipelineStageState::Completed,
                );
            }
            Err(err) => {
                self.record_activity("Dataset", err.clone(), PipelineStageState::Failed);
                self.ui_state.load_error = Some(err);
            }
        }
    }

    fn handle_image_sequence_import(&mut self, result: Result<ProjectImportSummary, String>) {
        self.ui_state.is_loading = false;
        self.ui_state.loading_message = None;

        match result {
            Ok(summary) => {
                let pipeline = match new_project_pipeline(&summary.path) {
                    Ok(pipeline) => pipeline,
                    Err(error) => {
                        self.ui_state.load_error = Some(error);
                        return;
                    }
                };
                if let Ok(mut scene) = self.scene.lock() {
                    clear_scene_preserving_layers(&mut scene);
                }
                self.loaded_colmap = None;
                self.loaded_splats = None;
                let project_summary =
                    ProjectSessionSummary::from_manifest(pipeline.store().manifest());
                self.record_project_activity(&project_summary);
                self.project_summary = Some(project_summary);
                self.project_path = Some(pipeline.store().root().to_path_buf());
                self.project_pipeline = Some(pipeline);
                self.ui_state.dataset_summary = None;
                self.ui_state.load_error = None;
                self.ui_state.training_error = None;
                self.ui_state.training_state = crate::training::TrainingSessionState::Idle;
                self.ui_state.training_progress = Default::default();
                self.viewport_bridge = self.new_gpu_viewport_bridge();
                self.viewport_texture = None;
                self.viewport_dirty = true;
                self.viewport_last_motion = None;
                self.viewport_render_error = None;
            }
            Err(error) => {
                self.record_activity("Import", error.clone(), PipelineStageState::Failed);
                self.ui_state.load_error = Some(error);
            }
        }
    }

    fn drive_project_pipeline(&mut self) {
        let should_load_dataset = self.loaded_colmap.is_none();
        let mut colmap_root = None;
        let mut pipeline_error = None;
        let mut project_root = None;
        let mut events = Vec::new();

        if let Some(pipeline) = self.project_pipeline.as_mut() {
            if let Err(error) = pipeline.drive_once() {
                pipeline_error = Some(error.to_string());
            }
            project_root = Some(pipeline.store().root().to_path_buf());
            while let Some(event) = pipeline.try_next_event() {
                events.push(event);
            }
        }

        for event in events {
            match event {
                PipelineEvent::ManifestChanged(manifest) => {
                    let project_summary = ProjectSessionSummary::from_manifest(&manifest);
                    self.record_project_activity(&project_summary);
                    self.project_summary = Some(project_summary);
                    if let Some(error) = reconstruction_failure_message(&manifest) {
                        self.ui_state.is_loading = false;
                        self.ui_state.loading_message = None;
                        self.ui_state.load_error = Some(error);
                    } else if should_load_dataset {
                        colmap_root = project_root
                            .as_deref()
                            .and_then(|root| committed_project_colmap_root(root, &manifest));
                    }
                }
                PipelineEvent::StageProgress { stage, detail, .. } => {
                    if should_ignore_pipeline_progress(self.project_summary.as_ref(), stage) {
                        continue;
                    }
                    self.ui_state.is_loading = true;
                    let message = pipeline_progress_message(&detail);
                    self.record_activity(
                        pipeline_activity_title(&detail),
                        message.clone(),
                        PipelineStageState::Running,
                    );
                    self.ui_state.loading_message = Some(message);
                }
                PipelineEvent::NeedsAttention(error) => {
                    self.ui_state.is_loading = false;
                    self.record_activity(
                        project_stage_title(error.stage),
                        error.detail.clone(),
                        PipelineStageState::Failed,
                    );
                    self.ui_state.load_error = Some(error.detail);
                }
                PipelineEvent::Idle => {
                    self.ui_state.is_loading = false;
                    self.ui_state.loading_message = None;
                }
                PipelineEvent::SceneSnapshot(_) => {}
            }
        }

        if let Some(error) = pipeline_error {
            self.ui_state.is_loading = false;
            self.ui_state.load_error = Some(error);
        }
        if let Some(root) = colmap_root {
            self.load_committed_project_colmap(root);
        }
    }

    fn load_committed_project_colmap(&mut self, root: PathBuf) {
        let result = load_colmap_training_dataset(&root, &ColmapConfig::default())
            .map_err(|error| error.to_string());
        let should_start_training = should_start_training_after_project_colmap_load(result.is_ok());
        self.ui_state.is_loading = false;
        self.ui_state.loading_message = None;

        match result {
            Ok(loaded) => {
                if let Ok(mut scene) = self.scene.lock() {
                    clear_scene_preserving_layers(&mut scene);
                    map_training_dataset_to_scene(&loaded.dataset, &mut scene);
                    if scene.has_data() {
                        self.camera.fit_scene(&scene.bounds);
                        self.robot.reset_to_scene(&scene.bounds);
                    }
                }
                self.ui_state.load_error = None;
                self.ui_state.dataset_summary = Some(DatasetUiSummary {
                    root_path: loaded.summary.input_dir.display().to_string(),
                    frame_count: loaded.summary.frame_count,
                    sparse_point_count: loaded.summary.sparse_point_count,
                    width: loaded.summary.intrinsics.width,
                    height: loaded.summary.intrinsics.height,
                });
                self.ui_state.training_error = None;
                self.ui_state.preview_error = None;
                self.loaded_colmap = Some(loaded);
                self.loaded_splats = None;
                self.viewport_bridge = self.new_gpu_viewport_bridge();
                self.viewport_texture = None;
                self.viewport_dirty = true;
                self.viewport_last_motion = None;
                self.viewport_render_error = None;
                self.record_activity(
                    "Dataset",
                    "Verified COLMAP dataset loaded",
                    PipelineStageState::Completed,
                );
            }
            Err(error) => {
                self.record_activity("Dataset", error.clone(), PipelineStageState::Failed);
                self.ui_state.load_error = Some(error);
            }
        }

        if should_start_training {
            self.start_training();
        }
    }

    fn new_gpu_viewport_bridge(&self) -> Option<GpuViewportBridge> {
        self.wgpu_render_state
            .as_ref()
            .zip(self.shared_wgpu_context.clone())
            .and_then(|(render_state, context)| {
                GpuViewportBridge::new(
                    context,
                    render_state.device.clone(),
                    render_state.queue.clone(),
                    render_state,
                )
                .ok()
            })
    }

    fn process_panel_actions(&mut self, ctx: &egui::Context, actions: Vec<PanelAction>) {
        for action in actions {
            match action {
                PanelAction::OpenImages => {
                    self.spawn_image_sequence_import(ctx, ImageImportSource::Folder)
                }
                PanelAction::RunReconstruction => self.spawn_image_sequence_sfm(),
                PanelAction::ImportImageFiles => {
                    self.spawn_image_sequence_import(ctx, ImageImportSource::Files)
                }
                PanelAction::ImportImageFolder => {
                    self.spawn_image_sequence_import(ctx, ImageImportSource::Folder)
                }
                PanelAction::SolvePoses => self.spawn_image_sequence_sfm(),
                PanelAction::OpenCheckpoint => self.spawn_file_dialog(AssetLoadKind::Checkpoint),
                PanelAction::OpenGaussian => self.spawn_file_dialog(AssetLoadKind::Gaussian),
                PanelAction::OpenMesh => self.spawn_file_dialog(AssetLoadKind::Mesh),
                PanelAction::OpenColmap => self.spawn_colmap_load(),
                PanelAction::StartTraining => self.start_training(),
                PanelAction::StopTraining => self.stop_training(),
                PanelAction::AutoFitScene => {
                    self.viewport_dirty = true;
                    self.viewport_last_motion = None;
                }
                PanelAction::ResetRobot => {
                    if let Ok(scene) = self.scene.lock() {
                        self.robot.reset_to_scene(&scene.bounds);
                    }
                    self.ui_state.robot_move_speed = self.robot.move_speed;
                    self.viewport_dirty = true;
                    self.viewport_last_motion = None;
                }
                PanelAction::SnapRobotToGround => {
                    if let Ok(scene) = self.scene.lock() {
                        self.robot.snap_to_scene_ground(&scene);
                    }
                    self.viewport_dirty = true;
                    self.viewport_last_motion = None;
                }
                PanelAction::PickRobotGround => {
                    self.ground_pick = Some(GroundPickState::default());
                    self.ui_state.navigation_mode = NavigationMode::Orbit;
                    self.viewport_dirty = true;
                    self.viewport_last_motion = None;
                }
                PanelAction::FlipRobotGround => {
                    self.robot.flip_ground_plane();
                    self.viewport_dirty = true;
                    self.viewport_last_motion = None;
                }
                PanelAction::PlaceRobotInView => {
                    let camera = self.camera.clone();
                    if let Ok(scene) = self.scene.lock() {
                        self.robot.place_in_camera_view(&camera, &scene);
                    }
                    self.ui_state.robot_visible = true;
                    self.robot.visible = true;
                    self.viewport_dirty = true;
                    self.viewport_last_motion = None;
                }
            }
        }
    }

    fn spawn_file_dialog(&self, kind: AssetLoadKind) {
        let tx = self.command_tx.clone();
        std::thread::spawn(move || {
            let dialog = match kind {
                AssetLoadKind::Checkpoint => rfd::FileDialog::new().add_filter("JSON", &["json"]),
                AssetLoadKind::Gaussian => {
                    rfd::FileDialog::new().add_filter("Splats", &["splat", "ply"])
                }
                AssetLoadKind::Mesh => rfd::FileDialog::new().add_filter("Mesh", &["obj", "ply"]),
            };

            if let Some(path) = dialog.pick_file() {
                let _ = tx.send(AppCommand::LoadAsset { kind, path });
            }
        });
    }

    fn spawn_colmap_load(&mut self) {
        self.ui_state.is_loading = true;
        self.ui_state.loading_message = Some("Loading COLMAP dataset…".to_string());
        self.ui_state.load_error = None;
        self.record_activity(
            "Dataset",
            "Opening COLMAP workspace",
            PipelineStageState::Running,
        );

        let tx = self.command_tx.clone();
        std::thread::spawn(move || {
            let Some(path) = rfd::FileDialog::new().pick_folder() else {
                return;
            };

            let result = load_colmap_training_dataset(&path, &ColmapConfig::default())
                .map_err(|err| err.to_string());
            let _ = tx.send(AppCommand::ColmapLoaded(result));
        });
    }

    fn spawn_image_sequence_import(&mut self, ctx: &egui::Context, source: ImageImportSource) {
        if matches!(
            self.ui_state.training_state,
            crate::training::TrainingSessionState::Loading
                | crate::training::TrainingSessionState::Starting
                | crate::training::TrainingSessionState::Training
                | crate::training::TrainingSessionState::Stopping
        ) {
            self.ui_state.load_error =
                Some("Stop training before importing a new sequence.".to_owned());
            return;
        }

        self.ui_state.is_loading = true;
        self.ui_state.loading_message = Some(match source {
            ImageImportSource::Files => "Importing selected images...".to_owned(),
            ImageImportSource::Folder => "Importing images from folder...".to_owned(),
        });
        self.ui_state.load_error = None;
        self.record_activity(
            "Import",
            match source {
                ImageImportSource::Files => "Choosing image files",
                ImageImportSource::Folder => "Choosing image folder",
            },
            PipelineStageState::Running,
        );

        let tx = self.command_tx.clone();
        let input = match source {
            ImageImportSource::Files => rfd::FileDialog::new()
                .add_filter("Images", IMAGE_FILE_EXTENSIONS)
                .pick_files()
                .map(ImageImportInput::Files),
            ImageImportSource::Folder => rfd::FileDialog::new()
                .pick_folder()
                .map(ImageImportInput::Folder),
        };
        let destination = if input.is_some() {
            rfd::FileDialog::new()
                .add_filter("RustScan Project", &["rustscanproject"])
                .save_file()
        } else {
            None
        };
        let Some(request) = image_import_request(input, destination) else {
            send_image_import_cancellation(&tx, ctx);
            return;
        };

        let completion_ctx = ctx.clone();
        std::thread::spawn(move || {
            let ImageImportRequest { input, destination } = request;
            let result = match input {
                ImageImportInput::Files(paths) => create_image_sequence_project(paths, destination),
                ImageImportInput::Folder(folder) => image_files_in_folder(&folder)
                    .and_then(|paths| create_image_sequence_project(paths, destination)),
            };
            send_image_import_completion(&tx, &completion_ctx, result);
        });
    }

    fn spawn_image_sequence_sfm(&mut self) {
        if self.ui_state.is_loading {
            return;
        }
        let result = match self.project_pipeline.as_ref() {
            Some(pipeline) => image_project_reconstruction_commands(pipeline.store().manifest())
                .into_iter()
                .try_for_each(|command| pipeline.send(command)),
            None => {
                self.ui_state.load_error =
                    Some("Import an image sequence before solving poses.".to_owned());
                return;
            }
        };

        self.ui_state.is_loading = true;
        self.ui_state.loading_message = Some("RustSFM is solving camera poses...".to_owned());
        self.ui_state.load_error = None;
        self.record_activity(
            "Pose solve",
            "Starting RustSFM",
            PipelineStageState::Running,
        );
        if let Err(error) = result {
            self.ui_state.is_loading = false;
            self.ui_state.loading_message = None;
            self.ui_state.load_error = Some(error.to_string());
        }
    }

    fn start_training(&mut self) {
        let Some(loaded) = self.loaded_colmap.as_ref() else {
            self.ui_state.training_error =
                Some("Load a COLMAP dataset before training.".to_string());
            return;
        };

        let mut config = TrainingConfig::default();
        config.iterations = self.ui_state.training_controls.iterations;
        config.raster.render_scale = self.ui_state.training_controls.render_scale;

        let options = TrainingControlOptions {
            progress_every: self.ui_state.training_controls.progress_every,
            snapshot_every: Some(self.ui_state.training_controls.snapshot_every),
            retain_snapshot_on_cancel: true,
        };

        match self
            .training_manager
            .start(loaded.dataset.clone(), config, options)
        {
            Ok(()) => {
                self.ui_state.training_error = None;
                self.ui_state.preview_error = None;
                self.viewport_dirty = true;
                self.record_activity(
                    "Gaussian train",
                    "Preparing RustGS",
                    PipelineStageState::Running,
                );
            }
            Err(err) => {
                let error = err.to_string();
                self.record_activity("Gaussian train", error.clone(), PipelineStageState::Failed);
                self.ui_state.training_error = Some(error);
            }
        }
    }

    fn stop_training(&mut self) {
        if let Err(err) = self.training_manager.stop() {
            let error = err.to_string();
            self.record_activity("Gaussian train", error.clone(), PipelineStageState::Failed);
            self.ui_state.training_error = Some(error);
        } else {
            self.record_activity(
                "Gaussian train",
                "Stopping RustGS",
                PipelineStageState::Running,
            );
        }
    }

    fn poll_training_events(&mut self, ctx: &egui::Context) {
        for event in self.training_manager.poll_events() {
            match event {
                TrainingSessionEvent::StateChanged { to, .. } => {
                    self.ui_state.training_state = to;
                }
                TrainingSessionEvent::ProgressUpdated(progress) => {
                    self.ui_state.training_progress = progress;
                }
                TrainingSessionEvent::SnapshotUpdated { .. } => {
                    self.sync_latest_snapshot_to_scene();
                    self.viewport_dirty = true;
                }
                TrainingSessionEvent::Completed(report) => {
                    self.ui_state.training_progress.latest_loss = report.final_loss;
                    self.ui_state.training_progress.gaussian_count = Some(report.gaussian_count);
                    self.sync_latest_snapshot_to_scene();
                    self.viewport_dirty = true;
                }
                TrainingSessionEvent::Failed(error) => {
                    self.ui_state.training_error = Some(error);
                }
                TrainingSessionEvent::Cancelled => {
                    self.viewport_dirty = true;
                }
                TrainingSessionEvent::BackendEvent(_) => {}
            }
        }

        self.ui_state.training_state = self.training_manager.state();
        self.ui_state.training_progress = self.training_manager.progress();
        if let Some(error) = self.training_manager.latest_error() {
            self.ui_state.training_error = Some(error);
        }
        if self.loaded_colmap.is_some() {
            self.record_training_activity();
        }

        if matches!(
            self.ui_state.training_state,
            crate::training::TrainingSessionState::Starting
                | crate::training::TrainingSessionState::Training
                | crate::training::TrainingSessionState::Stopping
        ) {
            ctx.request_repaint();
        }
    }

    fn sync_latest_snapshot_to_scene(&mut self) {
        let Some(snapshot) = self.training_manager.latest_snapshot() else {
            return;
        };

        if let Ok(mut scene) = self.scene.lock() {
            scene.gaussians = host_splats_to_scene_gaussians(&snapshot);
            scene.recompute_bounds();
            self.robot.sync_ground_from_scene(&scene.bounds);
        }
        self.loaded_splats = Some(snapshot);
        self.viewport_dirty = true;
        self.viewport_render_error = None;
    }

    fn record_project_activity(&mut self, summary: &ProjectSessionSummary) {
        let import_detail = summary
            .imported_frame_count
            .map(|count| format!("{count} frames imported"))
            .unwrap_or_else(|| stage_presentation_detail(summary.import_state, "Import"));
        self.record_activity(
            "Import",
            import_detail,
            presentation_pipeline_state(summary.import_state),
        );
        self.record_activity(
            "Pose solve",
            summary.pose_detail.clone(),
            presentation_pipeline_state(summary.pose_state),
        );
    }

    fn record_training_activity(&mut self) {
        let (detail, state) = match self.ui_state.training_state {
            crate::training::TrainingSessionState::Idle => {
                ("Ready to train".to_owned(), PipelineStageState::Ready)
            }
            crate::training::TrainingSessionState::Loading
            | crate::training::TrainingSessionState::Starting => {
                ("Preparing RustGS".to_owned(), PipelineStageState::Running)
            }
            crate::training::TrainingSessionState::Training => (
                self.ui_state
                    .training_progress
                    .latest_iteration
                    .map(|iteration| format!("Training iteration {iteration}"))
                    .unwrap_or_else(|| "RustGS training".to_owned()),
                PipelineStageState::Running,
            ),
            crate::training::TrainingSessionState::Stopping => {
                ("Stopping RustGS".to_owned(), PipelineStageState::Running)
            }
            crate::training::TrainingSessionState::Completed => {
                ("RustGS completed".to_owned(), PipelineStageState::Completed)
            }
            crate::training::TrainingSessionState::Failed => (
                self.ui_state
                    .training_error
                    .clone()
                    .unwrap_or_else(|| "RustGS training failed".to_owned()),
                PipelineStageState::Failed,
            ),
            crate::training::TrainingSessionState::Cancelled => (
                "RustGS training cancelled".to_owned(),
                PipelineStageState::Waiting,
            ),
        };
        self.record_activity("Gaussian train", detail, state);
    }

    fn record_activity(
        &mut self,
        title: impl Into<String>,
        detail: impl Into<String>,
        state: PipelineStageState,
    ) {
        record_workbench_activity(&mut self.activity, title, detail, state);
    }

    fn display_camera(&self) -> ArcballCamera {
        if self.ui_state.navigation_mode == NavigationMode::Robot {
            self.robot.camera()
        } else {
            self.camera.clone()
        }
    }

    fn pick_viewport_world_point(
        &self,
        viewport_rect: Rect,
        pointer_pos: egui::Pos2,
    ) -> Option<Vec3> {
        let scene = self.scene.lock().ok()?;
        let use_splat_depth =
            self.loaded_splats.is_some() && scene.layers.gaussians && !self.viewport_dirty;
        let splat_depth = use_splat_depth
            .then(|| {
                self.viewport_bridge
                    .as_ref()
                    .and_then(|bridge| bridge.depth_at_viewport_pos(viewport_rect, pointer_pos))
            })
            .flatten();
        pick_viewport_point(
            &scene,
            &self.camera,
            viewport_rect,
            pointer_pos,
            splat_depth,
        )
    }

    fn record_ground_pick_point(&mut self, point: Vec3, camera_eye: Vec3) {
        let Some(ground_pick) = self.ground_pick.as_mut() else {
            return;
        };
        ground_pick.points.push(point);
        if ground_pick.points.len() < GROUND_PICK_POINT_COUNT {
            return;
        }

        let points = ground_pick.points.clone();
        self.ground_pick = None;
        if let Some(ground_plane) = GroundPlane::from_points(&points, camera_eye) {
            self.robot.set_ground_plane(ground_plane);
            self.robot.snap_to_ground();
            self.viewport_dirty = true;
            self.viewport_last_motion = None;
        }
    }

    fn refresh_viewport_texture_id(
        &mut self,
        ctx: &egui::Context,
        size: Vec2,
    ) -> Option<egui::TextureId> {
        let Some(splats) = self.loaded_splats.clone() else {
            return None;
        };
        let camera = self.display_camera();
        let Some(viewport_bridge) = self.viewport_bridge.as_mut() else {
            self.viewport_render_error = Some("wgpu render state is unavailable".to_string());
            return None;
        };

        let scale =
            viewport_render_scale(self.viewport_last_motion, VIEWPORT_INTERACTIVE_IDLE_DELAY);
        if scale < 1.0 {
            ctx.request_repaint_after(VIEWPORT_INTERACTIVE_IDLE_DELAY);
        }
        let Some(resolution) = PreviewResolution::from_panel_size_scaled(size, scale) else {
            return self.viewport_texture.map(|texture| texture.id);
        };
        if !needs_texture_render(self.viewport_dirty, self.viewport_texture, resolution) {
            return self.viewport_texture.map(|texture| texture.id);
        }
        match viewport_bridge.render_texture_id(
            self.wgpu_render_state.as_ref(),
            splats,
            &camera,
            size,
            scale,
        ) {
            Ok(texture_id) => {
                self.viewport_render_error = None;
                self.viewport_dirty = false;
                self.viewport_texture = texture_id.map(|id| CachedTexture { id, resolution });
                texture_id
            }
            Err(err) => {
                self.viewport_render_error = Some(err.to_string());
                self.viewport_dirty = false;
                None
            }
        }
    }
}

fn startup_asset_kind(path: &std::path::Path) -> AssetLoadKind {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .as_deref()
    {
        Some("json") => AssetLoadKind::Checkpoint,
        Some("obj") => AssetLoadKind::Mesh,
        Some("ply") | Some("splat") => AssetLoadKind::Gaussian,
        _ => AssetLoadKind::Gaussian,
    }
}

fn create_image_sequence_project(
    paths: Vec<PathBuf>,
    destination: PathBuf,
) -> Result<ProjectImportSummary, String> {
    let destination = project_package_path(destination);
    let display_name = destination
        .file_stem()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("Untitled reconstruction")
        .to_owned();
    let mut store = ProjectStore::create(
        destination,
        ProjectCreateRequest::new(display_name, SourceSpec::managed_images("pending-import")),
    )
    .map_err(|error| error.to_string())?;
    let mut events = DiscardMediaEvents;
    import_image_sequence(
        &ImageSequenceImportRequest::managed(paths),
        &mut store,
        &mut events,
    )
    .map_err(|error| error.to_string())?;

    Ok(ProjectImportSummary {
        path: store.root().to_path_buf(),
    })
}

fn image_files_in_folder(folder: &std::path::Path) -> Result<Vec<PathBuf>, String> {
    let mut paths = std::fs::read_dir(folder)
        .map_err(|error| format!("failed to read image folder {}: {error}", folder.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file() && is_supported_image_file(path))
        .collect::<Vec<_>>();
    paths.sort();

    if paths.len() < 2 {
        return Err(format!(
            "image folder {} must contain at least two supported image files",
            folder.display()
        ));
    }

    Ok(paths)
}

fn is_supported_image_file(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            IMAGE_FILE_EXTENSIONS
                .iter()
                .any(|supported| extension.eq_ignore_ascii_case(supported))
        })
}

fn new_project_pipeline(project_path: &std::path::Path) -> Result<PipelineCoordinator, String> {
    let store = ProjectStore::open(project_path).map_err(|error| error.to_string())?;
    let workers = PipelineWorkers::new(
        DisabledProjectStageWorker,
        RustSfmWorker,
        RustSfmWorker,
        DisabledProjectStageWorker,
    );
    PipelineCoordinator::new(store, workers).map_err(|error| error.to_string())
}

fn image_project_reconstruction_command() -> PipelineCommand {
    PipelineCommand::StartThrough {
        stage: ProjectStage::FullFramePnp,
    }
}

fn image_project_reconstruction_commands(
    manifest: &crate::project::ProjectManifest,
) -> Vec<PipelineCommand> {
    let mut commands = Vec::new();
    if let Some(stage) = [ProjectStage::KeyframeSfm, ProjectStage::FullFramePnp]
        .into_iter()
        .find(|stage| {
            manifest
                .try_stage(*stage)
                .is_ok_and(|record| record.state() == crate::project::StageState::Failed)
        })
    {
        commands.push(PipelineCommand::RestartFrom {
            stage,
            confirmed: true,
        });
    }
    commands.push(image_project_reconstruction_command());
    commands
}

fn committed_project_colmap_root(
    project_root: &std::path::Path,
    manifest: &crate::project::ProjectManifest,
) -> Option<PathBuf> {
    let pnp = manifest.try_stage(ProjectStage::FullFramePnp).ok()?;
    if pnp.state() != crate::project::StageState::Succeeded {
        return None;
    }
    let artifact = pnp.artifacts().iter().find(|artifact| {
        artifact
            .relative_path
            .ends_with("/colmap/sparse/0/cameras.txt")
    })?;
    let sparse = project_root
        .join(&artifact.relative_path)
        .parent()?
        .to_path_buf();
    sparse.parent()?.parent().map(std::path::Path::to_path_buf)
}

fn should_start_training_after_project_colmap_load(load_succeeded: bool) -> bool {
    load_succeeded
}

fn reconstruction_failure_message(manifest: &crate::project::ProjectManifest) -> Option<String> {
    [ProjectStage::KeyframeSfm, ProjectStage::FullFramePnp]
        .into_iter()
        .find_map(|stage| {
            let record = manifest.try_stage(stage).ok()?;
            (record.state() == crate::project::StageState::Failed).then(|| {
                record
                    .error()
                    .map(|error| error.detail.clone())
                    .unwrap_or_else(|| "RustSFM pose solve failed.".to_owned())
            })
        })
}

fn pipeline_progress_message(detail: &PipelineProgressDetail) -> String {
    match detail {
        PipelineProgressDetail::Media { frame_id } => frame_id
            .map(|id| format!("Importing frame {id}"))
            .unwrap_or_else(|| "Importing media".to_owned()),
        PipelineProgressDetail::Sfm {
            operation,
            registered_images,
            ..
        } => registered_images
            .map(|count| format!("{operation}: {count} poses"))
            .unwrap_or_else(|| operation.clone()),
        PipelineProgressDetail::Training { iteration, .. } => {
            format!("Training iteration {iteration}")
        }
    }
}

fn pipeline_activity_title(detail: &PipelineProgressDetail) -> &'static str {
    match detail {
        PipelineProgressDetail::Media { .. } => "Import",
        PipelineProgressDetail::Sfm { .. } => "Pose solve",
        PipelineProgressDetail::Training { .. } => "Gaussian train",
    }
}

fn should_ignore_pipeline_progress(
    project: Option<&ProjectSessionSummary>,
    stage: ProjectStage,
) -> bool {
    matches!(
        stage,
        ProjectStage::KeyframeSfm | ProjectStage::FullFramePnp
    ) && project.is_some_and(|summary| {
        summary.pose_state == crate::project::ProjectStagePresentation::Failed
    })
}

fn project_stage_title(stage: ProjectStage) -> &'static str {
    match stage {
        ProjectStage::Import => "Import",
        ProjectStage::KeyframeSfm | ProjectStage::FullFramePnp => "Pose solve",
        ProjectStage::Training => "Gaussian train",
        ProjectStage::Complete => "Pipeline",
    }
}

fn presentation_pipeline_state(
    state: crate::project::ProjectStagePresentation,
) -> PipelineStageState {
    match state {
        crate::project::ProjectStagePresentation::Ready => PipelineStageState::Ready,
        crate::project::ProjectStagePresentation::Waiting => PipelineStageState::Waiting,
        crate::project::ProjectStagePresentation::Running => PipelineStageState::Running,
        crate::project::ProjectStagePresentation::Completed => PipelineStageState::Completed,
        crate::project::ProjectStagePresentation::Failed => PipelineStageState::Failed,
    }
}

fn stage_presentation_detail(
    state: crate::project::ProjectStagePresentation,
    name: &'static str,
) -> String {
    match state {
        crate::project::ProjectStagePresentation::Ready => format!("{name} ready"),
        crate::project::ProjectStagePresentation::Waiting => format!("Waiting for {name}"),
        crate::project::ProjectStagePresentation::Running => format!("{name} running"),
        crate::project::ProjectStagePresentation::Completed => format!("{name} completed"),
        crate::project::ProjectStagePresentation::Failed => format!("{name} failed"),
    }
}

fn record_workbench_activity(
    activity: &mut VecDeque<WorkbenchActivity>,
    title: impl Into<String>,
    detail: impl Into<String>,
    state: PipelineStageState,
) {
    let entry = WorkbenchActivity::new(title, detail, state);
    if let Some(index) = activity
        .iter()
        .position(|current| current.title == entry.title)
    {
        if activity[index] == entry {
            return;
        }
        activity.remove(index);
    }
    if activity.len() == MAX_WORKBENCH_ACTIVITY_ENTRIES {
        activity.pop_front();
    }
    activity.push_back(entry);
}

fn project_pipeline_repaint_delay(
    has_project_pipeline: bool,
    is_loading: bool,
) -> Option<Duration> {
    (has_project_pipeline && is_loading).then_some(Duration::from_millis(16))
}

#[cfg(test)]
fn snapshot_after_project_colmap_result(
    root: &std::path::Path,
) -> Result<WorkbenchSnapshot, String> {
    let loaded = load_colmap_training_dataset(root, &ColmapConfig::default())
        .map_err(|error| error.to_string())?;
    Ok(WorkbenchSnapshot {
        has_dataset: true,
        frame_count: loaded.summary.frame_count,
        sparse_point_count: loaded.summary.sparse_point_count,
        ..Default::default()
    })
}

fn project_package_path(path: PathBuf) -> PathBuf {
    if path.extension().and_then(|extension| extension.to_str()) == Some("rustscanproject") {
        path
    } else {
        path.with_extension("rustscanproject")
    }
}

fn clear_scene_preserving_layers(scene: &mut Scene) {
    let layers = scene.layers.clone();
    *scene = Scene::default();
    scene.layers = layers;
}

fn needs_texture_render(
    dirty: bool,
    cached: Option<CachedTexture>,
    resolution: PreviewResolution,
) -> bool {
    dirty
        || cached
            .map(|texture| texture.resolution != resolution)
            .unwrap_or(true)
}

fn new_gpu_viewport_bridge(
    context: &Option<SharedWgpuContext>,
    render_state: Option<&egui_wgpu::RenderState>,
) -> Option<GpuViewportBridge> {
    render_state
        .zip(context.clone())
        .and_then(|(render_state, context)| {
            GpuViewportBridge::new(
                context,
                render_state.device.clone(),
                render_state.queue.clone(),
                render_state,
            )
            .ok()
        })
}

fn shared_wgpu_context_from_render_state(
    render_state: &egui_wgpu::RenderState,
) -> SharedWgpuContext {
    let backend = render_state.adapter.get_info().backend;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: backend.into(),
        flags: wgpu::InstanceFlags::from_build_config().with_env(),
        backend_options: wgpu::BackendOptions::from_env_or_default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    });
    SharedWgpuContext::from_wgpu_parts(
        instance,
        render_state.adapter.clone(),
        render_state.device.clone(),
        render_state.queue.clone(),
        backend,
    )
}

impl eframe::App for ViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        crate::ui::theme::configure_theme(&ctx);

        self.poll_commands();
        self.drive_project_pipeline();
        self.poll_training_events(&ctx);
        if let Some(delay) = project_pipeline_repaint_delay(
            self.project_pipeline.is_some(),
            self.ui_state.is_loading,
        ) {
            ctx.request_repaint_after(delay);
        }
        self.robot.visible = self.ui_state.robot_visible;
        self.robot.camera_mode = self.ui_state.robot_camera_mode;
        self.robot.move_speed = self.ui_state.robot_move_speed;

        let (has_data, scene_bounds) = self
            .scene
            .lock()
            .map(|scene| (scene.has_data(), scene.bounds.clone()))
            .unwrap_or_default();
        let gaussian_count = self
            .scene
            .lock()
            .map(|scene| scene.gaussian_count())
            .ok()
            .filter(|count| *count > 0);
        let snapshot = WorkbenchSnapshot {
            has_dataset: self.ui_state.dataset_summary.is_some(),
            project: self.project_summary.clone(),
            frame_count: self
                .ui_state
                .dataset_summary
                .as_ref()
                .map(|summary| summary.frame_count)
                .unwrap_or_default(),
            sparse_point_count: self
                .ui_state
                .dataset_summary
                .as_ref()
                .map(|summary| summary.sparse_point_count)
                .unwrap_or_default(),
            training_state: self.ui_state.training_state,
            gaussian_count: self
                .ui_state
                .training_progress
                .gaussian_count
                .or(gaussian_count),
            has_rendered_gaussians: self.loaded_splats.is_some(),
            error: self
                .ui_state
                .training_error
                .clone()
                .or_else(|| self.ui_state.load_error.clone()),
            activity: self.activity.iter().cloned().collect(),
        };
        let layout = WorkbenchLayout::for_window(ui.available_size());

        egui::Panel::top("workbench_command_bar")
            .exact_size(layout.top_bar_height)
            .frame(egui::Frame::new().fill(WINDOW_BG))
            .show_inside(ui, |ui| {
                workbench::draw_command_bar(ui, &snapshot);
            });

        let mut panel_actions = Vec::new();
        egui::Panel::left("workbench_stage_rail")
            .exact_size(layout.stage_rail_width)
            .frame(
                egui::Frame::new()
                    .fill(PANEL_BG)
                    .inner_margin(egui::Margin::same(14)),
            )
            .show_inside(ui, |ui| {
                panel_actions.extend(workbench::draw_stage_rail(ui, &snapshot));
            });
        egui::Panel::right("workbench_inspector")
            .exact_size(layout.inspector_width)
            .frame(
                egui::Frame::new()
                    .fill(PANEL_BG)
                    .inner_margin(egui::Margin::same(14)),
            )
            .show_inside(ui, |ui| {
                if let Ok(mut scene) = self.scene.lock() {
                    panel_actions.extend(workbench::draw_inspector(
                        ui,
                        &mut self.ui_state,
                        &mut scene,
                        &snapshot,
                    ));
                }
            });
        egui::Panel::bottom("workbench_activity")
            .exact_size(layout.activity_height)
            .frame(
                egui::Frame::new()
                    .fill(PANEL_BG)
                    .inner_margin(egui::Margin::same(14)),
            )
            .show_inside(ui, |ui| {
                workbench::draw_activity_strip(ui, &snapshot);
            });
        egui::Panel::top("workbench_viewport_toolbar")
            .exact_size(layout.viewport_toolbar_height)
            .frame(
                egui::Frame::new()
                    .fill(PANEL_BG)
                    .inner_margin(egui::Margin::symmetric(0, 6)),
            )
            .show_inside(ui, |ui| {
                panel_actions.extend(workbench::draw_viewport_toolbar(ui, &snapshot));
            });
        self.process_panel_actions(&ctx, panel_actions);
        self.robot.visible = self.ui_state.robot_visible;
        self.robot.camera_mode = self.ui_state.robot_camera_mode;
        self.robot.move_speed = self.ui_state.robot_move_speed;

        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(VIEWPORT_BG))
            .show_inside(ui, |ui| {
                if !has_data {
                    workbench::draw_viewport_empty_state(ui, &snapshot);
                    return;
                }

                let viewport_rect = ui.max_rect();
                let viewport_size = [viewport_rect.width(), viewport_rect.height()];

                let response = ui.allocate_rect(viewport_rect, egui::Sense::click_and_drag());
                if response.clicked() {
                    response.request_focus();
                }

                let ground_pick_was_active = self.ground_pick.is_some();
                if ground_pick_was_active && response.clicked_by(egui::PointerButton::Primary) {
                    if let Some(pointer_pos) = response.interact_pointer_pos() {
                        if let Some(point) =
                            self.pick_viewport_world_point(viewport_rect, pointer_pos)
                        {
                            self.record_ground_pick_point(point, self.camera.eye());
                            ctx.request_repaint();
                        }
                    }
                }

                let mut camera_moved = false;
                if self.ui_state.navigation_mode == NavigationMode::Robot {
                    let accepts_keyboard = response.hovered()
                        || response.has_focus()
                        || response.dragged()
                        || !ctx.egui_wants_keyboard_input();
                    if accepts_keyboard {
                        let (robot_input, dt) = ui.input(|input| {
                            let mut robot_input = RobotInput::default();
                            if input.key_down(egui::Key::W) || input.key_down(egui::Key::ArrowUp) {
                                robot_input.forward += 1.0;
                            }
                            if input.key_down(egui::Key::S) || input.key_down(egui::Key::ArrowDown)
                            {
                                robot_input.forward -= 1.0;
                            }
                            if input.key_down(egui::Key::A) {
                                robot_input.turn += 1.0;
                            }
                            if input.key_down(egui::Key::D) {
                                robot_input.turn -= 1.0;
                            }
                            if input.key_down(egui::Key::Q) || input.key_down(egui::Key::ArrowLeft)
                            {
                                robot_input.strafe -= 1.0;
                            }
                            if input.key_down(egui::Key::E) || input.key_down(egui::Key::ArrowRight)
                            {
                                robot_input.strafe += 1.0;
                            }
                            (robot_input, input.stable_dt.min(0.05))
                        });
                        if self.robot.apply_input(robot_input, dt) {
                            camera_moved = true;
                        }
                    }
                    if response.dragged_by(egui::PointerButton::Primary)
                        || response.dragged_by(egui::PointerButton::Secondary)
                    {
                        let delta = response.drag_motion();
                        self.robot.look(delta.x, delta.y);
                        camera_moved = true;
                    }
                    if response.hovered() {
                        let scroll = ui.input(|input| input.smooth_scroll_delta.y);
                        if scroll != 0.0 {
                            self.robot.adjust_speed(scroll);
                            self.ui_state.robot_move_speed = self.robot.move_speed;
                        }
                    }
                } else {
                    if !ground_pick_was_active && response.double_clicked() {
                        if let Some(pointer_pos) = response.interact_pointer_pos() {
                            if let Ok(scene) = self.scene.lock() {
                                let use_splat_depth = self.loaded_splats.is_some()
                                    && scene.layers.gaussians
                                    && !self.viewport_dirty;
                                let splat_depth = use_splat_depth
                                    .then(|| {
                                        self.viewport_bridge.as_ref().and_then(|bridge| {
                                            bridge.depth_at_viewport_pos(viewport_rect, pointer_pos)
                                        })
                                    })
                                    .flatten();
                                if let Some(target) = pick_viewport_focus(
                                    &scene,
                                    &self.camera,
                                    viewport_rect,
                                    pointer_pos,
                                    splat_depth,
                                ) {
                                    self.camera.focus_on(target);
                                    camera_moved = true;
                                }
                            }
                        }
                    }
                    if response.dragged_by(egui::PointerButton::Primary) {
                        let delta = response.drag_motion();
                        if ui.input(|input| input.modifiers.shift) {
                            self.camera.roll(delta.x);
                        } else {
                            self.camera.orbit(delta.x, delta.y);
                        }
                        camera_moved = true;
                    }
                    if response.dragged_by(egui::PointerButton::Middle) {
                        let delta = response.drag_motion();
                        self.camera.roll(delta.x);
                        camera_moved = true;
                    }
                    if response.dragged_by(egui::PointerButton::Secondary) {
                        let delta = response.drag_motion();
                        self.camera.pan(delta.x, delta.y);
                        camera_moved = true;
                    }
                    if response.hovered() {
                        let scroll = ui.input(|input| input.smooth_scroll_delta.y);
                        if scroll != 0.0 {
                            self.camera.zoom(scroll);
                            camera_moved = true;
                        }
                    }
                }
                if camera_moved {
                    self.viewport_dirty = true;
                    self.viewport_last_motion = Some(Instant::now());
                    ctx.request_repaint();
                }

                let true_splat_view = self.loaded_splats.is_some()
                    && self
                        .scene
                        .lock()
                        .map(|scene| scene.layers.gaussians)
                        .unwrap_or(false);
                if true_splat_view {
                    let texture_id = self.refresh_viewport_texture_id(&ctx, viewport_rect.size());
                    ui.painter().rect_filled(viewport_rect, 0.0, VIEWPORT_BG);
                    if let Some(texture_id) = texture_id {
                        ui.painter().image(
                            texture_id,
                            viewport_rect,
                            Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                            Color32::WHITE,
                        );
                    } else {
                        draw_viewport_placeholder(
                            ui,
                            viewport_rect,
                            self.viewport_render_error
                                .as_deref()
                                .unwrap_or("Rendering 3DGS view…"),
                        );
                    }
                }

                let callback = egui_wgpu::Callback::new_paint_callback(
                    viewport_rect,
                    ViewerCallback {
                        scene: Arc::clone(&self.scene),
                        camera: self.display_camera(),
                        viewport_size,
                        robot_mesh: self.robot.render_mesh(),
                        surface_format: self.surface_format,
                    },
                );
                ui.painter().add(callback);

                if let Some(ground_pick) = &self.ground_pick {
                    draw_ground_pick_overlay(ui, viewport_rect, &self.camera, ground_pick);
                }

                workbench::draw_viewport_grid(ui, viewport_rect);

                let _ = scene_bounds;
            });
    }
}

fn host_splats_to_scene_gaussians(splats: &HostSplats) -> Vec<GaussianSplat> {
    let mut gaussians = Vec::with_capacity(splats.len());
    for idx in 0..splats.len() {
        gaussians.push(GaussianSplat {
            position: splats.position(idx),
            scale: splats.scale(idx),
            rotation: splats.rotation(idx),
            opacity: splats.opacity(idx),
            color: splats.rgb_color(idx),
        });
    }
    gaussians
}

fn draw_viewport_placeholder(ui: &egui::Ui, rect: Rect, message: &str) {
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        message,
        egui::FontId::proportional(13.0),
        TEXT_SECONDARY,
    );
}

fn draw_ground_pick_overlay(
    ui: &egui::Ui,
    viewport_rect: Rect,
    camera: &ArcballCamera,
    ground_pick: &GroundPickState,
) {
    let painter = ui.painter();
    let label_pos = viewport_rect.left_top() + Vec2::new(16.0, 16.0);
    painter.text(
        label_pos,
        egui::Align2::LEFT_TOP,
        format!(
            "Ground points {}/{}",
            ground_pick.points.len(),
            GROUND_PICK_POINT_COUNT
        ),
        egui::FontId::proportional(13.0),
        TEXT_PRIMARY,
    );

    for point in &ground_pick.points {
        if let Some(screen) = world_to_screen(camera, viewport_rect, *point) {
            painter.circle_filled(screen, 5.0, Color32::from_rgb(48, 209, 88));
            painter.circle_stroke(screen, 7.0, egui::Stroke::new(1.5, Color32::BLACK));
        }
    }
}

fn world_to_screen(camera: &ArcballCamera, viewport_rect: Rect, point: Vec3) -> Option<egui::Pos2> {
    let size = viewport_rect.size();
    if size.x <= 1.0 || size.y <= 1.0 {
        return None;
    }
    let clip = camera.view_proj(size.x / size.y.max(1.0)) * point.extend(1.0);
    if clip.w <= 1e-8 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    if ndc.z < -1.0 || ndc.z > 1.0 {
        return None;
    }
    Some(egui::pos2(
        viewport_rect.left() + (ndc.x + 1.0) * 0.5 * size.x,
        viewport_rect.top() + (1.0 - ndc.y) * 0.5 * size.y,
    ))
}

fn pick_viewport_focus(
    scene: &Scene,
    camera: &ArcballCamera,
    viewport_rect: Rect,
    pointer_pos: egui::Pos2,
    splat_depth: Option<f32>,
) -> Option<Vec3> {
    pick_viewport_point(scene, camera, viewport_rect, pointer_pos, splat_depth)
}

fn pick_viewport_point(
    scene: &Scene,
    camera: &ArcballCamera,
    viewport_rect: Rect,
    pointer_pos: egui::Pos2,
    splat_depth: Option<f32>,
) -> Option<Vec3> {
    let ray = viewport_ray(camera, viewport_rect, pointer_pos)?;
    if let Some(depth) = splat_depth {
        return focus_from_camera_depth(camera, ray.1, depth);
    }
    pick_mesh_focus(scene, ray)
}

fn focus_from_camera_depth(camera: &ArcballCamera, ray_dir: Vec3, depth: f32) -> Option<Vec3> {
    if !depth.is_finite() || depth <= 0.0 {
        return None;
    }
    let forward = -camera.backward();
    let denom = ray_dir.dot(forward);
    if !denom.is_finite() || denom <= 1e-6 {
        return None;
    }
    let distance = depth / denom;
    (distance.is_finite() && distance > 0.0).then_some(camera.eye() + ray_dir * distance)
}

fn viewport_ray(
    camera: &ArcballCamera,
    viewport_rect: Rect,
    pointer_pos: egui::Pos2,
) -> Option<(Vec3, Vec3)> {
    let size = viewport_rect.size();
    if size.x <= 1.0 || size.y <= 1.0 {
        return None;
    }

    let x = ((pointer_pos.x - viewport_rect.left()) / size.x) * 2.0 - 1.0;
    let y = 1.0 - ((pointer_pos.y - viewport_rect.top()) / size.y) * 2.0;
    let aspect = size.x / size.y.max(1.0);
    let inv_view_proj = camera.view_proj(aspect).inverse();
    let near = inv_view_proj * Vec4::new(x, y, -1.0, 1.0);
    let far = inv_view_proj * Vec4::new(x, y, 1.0, 1.0);
    if near.w.abs() <= 1e-8 || far.w.abs() <= 1e-8 {
        return None;
    }

    let near = near.truncate() / near.w;
    let far = far.truncate() / far.w;
    let dir = (far - near).normalize_or_zero();
    (dir.length_squared() > 0.0).then_some((near, dir))
}

fn pick_mesh_focus(scene: &Scene, ray: (Vec3, Vec3)) -> Option<Vec3> {
    if scene.mesh_indices.len() < 3 || scene.mesh_vertices.is_empty() {
        return None;
    }

    let (origin, dir) = ray;
    let mut best_t = f32::INFINITY;
    let mut best_hit = None;
    for tri in scene.mesh_indices.chunks_exact(3) {
        let Some(a) = scene.mesh_vertices.get(tri[0] as usize) else {
            continue;
        };
        let Some(b) = scene.mesh_vertices.get(tri[1] as usize) else {
            continue;
        };
        let Some(c) = scene.mesh_vertices.get(tri[2] as usize) else {
            continue;
        };
        if let Some(t) = ray_triangle_t(
            origin,
            dir,
            Vec3::from_array(a.position),
            Vec3::from_array(b.position),
            Vec3::from_array(c.position),
        ) {
            if t < best_t {
                best_t = t;
                best_hit = Some(origin + dir * t);
            }
        }
    }

    best_hit
}

fn ray_triangle_t(origin: Vec3, dir: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    let edge1 = b - a;
    let edge2 = c - a;
    let h = dir.cross(edge2);
    let det = edge1.dot(h);
    if det.abs() < 1e-8 {
        return None;
    }

    let inv_det = 1.0 / det;
    let s = origin - a;
    let u = inv_det * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let q = s.cross(edge1);
    let v = inv_det * dir.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = inv_det * edge2.dot(q);
    (t > 1e-5).then_some(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::scene::MeshGpuVertex;
    use std::fs;

    fn viewer_app_for_test() -> ViewerApp {
        let (command_tx, command_rx) = mpsc::channel();
        ViewerApp {
            scene: Arc::new(Mutex::new(Scene::default())),
            camera: ArcballCamera::default(),
            robot: RobotController::default(),
            ui_state: UiState::default(),
            loaded_colmap: None,
            project_summary: None,
            project_path: None,
            project_pipeline: None,
            activity: VecDeque::new(),
            command_rx,
            command_tx,
            training_manager: TrainingManager::new(),
            loaded_splats: None,
            viewport_bridge: None,
            viewport_dirty: true,
            viewport_texture: None,
            viewport_last_motion: None,
            ground_pick: None,
            viewport_render_error: None,
            shared_wgpu_context: None,
            wgpu_render_state: None,
            surface_format: wgpu::TextureFormat::Bgra8Unorm,
        }
    }

    #[test]
    fn image_sequence_action_routes_to_project_import() {
        assert_eq!(
            crate::ui::panel::panel_action_for_primary_import(
                crate::ui::panel::SourceSelection::Images,
            ),
            PanelAction::ImportImageFiles,
        );
    }

    #[test]
    fn image_import_request_requires_selected_images_and_destination() {
        let paths = vec![PathBuf::from("frame-01.png"), PathBuf::from("frame-02.png")];
        let input = ImageImportInput::Files(paths);
        let destination = PathBuf::from("Capture.rustscanproject");
        assert!(image_import_request(Some(input.clone()), Some(destination.clone())).is_some());
        assert!(image_import_request(None, Some(destination.clone())).is_none());
        assert!(image_import_request(Some(input), None).is_none());
    }

    #[test]
    fn image_import_request_keeps_folder_selection_for_worker() {
        let folder = PathBuf::from("Capture");
        let destination = PathBuf::from("Capture.rustscanproject");

        let request = image_import_request(
            Some(ImageImportInput::Folder(folder.clone())),
            Some(destination),
        )
        .unwrap();

        assert!(matches!(
            request.input,
            ImageImportInput::Folder(selected_folder) if selected_folder == folder
        ));
    }

    #[test]
    fn image_import_completion_enqueues_result_before_requesting_repaint() {
        let (tx, rx) = mpsc::channel();
        let rx = Arc::new(Mutex::new(rx));
        let received_command = Arc::new(Mutex::new(None));
        let callback_rx = Arc::clone(&rx);
        let callback_command = Arc::clone(&received_command);
        let ctx = egui::Context::default();
        ctx.set_request_repaint_callback(move |_| {
            *callback_command.lock().unwrap() = Some(
                callback_rx
                    .lock()
                    .unwrap()
                    .try_recv()
                    .expect("image import result should be queued before repaint"),
            );
        });

        send_image_import_completion(&tx, &ctx, Err("import failed".to_owned()));

        assert!(matches!(
            received_command.lock().unwrap().take(),
            Some(AppCommand::ImageSequenceImported(Err(error))) if error == "import failed"
        ));
    }

    #[test]
    fn image_import_cancellation_enqueues_command_before_requesting_repaint() {
        let (tx, rx) = mpsc::channel();
        let rx = Arc::new(Mutex::new(rx));
        let received_command = Arc::new(Mutex::new(None));
        let callback_rx = Arc::clone(&rx);
        let callback_command = Arc::clone(&received_command);
        let ctx = egui::Context::default();
        ctx.set_request_repaint_callback(move |_| {
            *callback_command.lock().unwrap() = Some(
                callback_rx
                    .lock()
                    .unwrap()
                    .try_recv()
                    .expect("image import cancellation should be queued before repaint"),
            );
        });

        send_image_import_cancellation(&tx, &ctx);

        assert!(matches!(
            received_command.lock().unwrap().take(),
            Some(AppCommand::ImageSequenceImportCancelled)
        ));
    }

    #[test]
    fn image_folder_selection_collects_only_supported_image_files() {
        let directory = tempfile::tempdir().unwrap();
        let first = directory.path().join("frame-01.JPG");
        let second = directory.path().join("frame-02.png");
        fs::write(&first, []).unwrap();
        fs::write(&second, []).unwrap();
        fs::write(directory.path().join("notes.txt"), []).unwrap();
        fs::create_dir(directory.path().join("nested")).unwrap();

        let paths = image_files_in_folder(directory.path()).unwrap();

        assert_eq!(paths, vec![first, second]);
    }

    #[test]
    fn imported_image_project_waits_for_explicit_reconstruction_start() {
        let directory = tempfile::tempdir().unwrap();
        let first = directory.path().join("frame-01.png");
        let second = directory.path().join("frame-02.png");
        image::RgbaImage::from_pixel(2, 2, image::Rgba([12, 34, 56, 255]))
            .save(&first)
            .unwrap();
        image::RgbaImage::from_pixel(2, 2, image::Rgba([56, 34, 12, 255]))
            .save(&second)
            .unwrap();
        let imported = create_image_sequence_project(
            vec![first, second],
            directory.path().join("Sample.rustscanproject"),
        )
        .unwrap();
        let mut app = viewer_app_for_test();

        app.handle_image_sequence_import(Ok(imported));

        assert!(!app.ui_state.is_loading);
        let snapshot = WorkbenchSnapshot {
            project: app.project_summary.clone(),
            ..Default::default()
        };
        assert_eq!(snapshot.primary_command().label, "运行重建");
        assert!(snapshot.primary_command().enabled);
    }

    #[test]
    fn committed_full_frame_result_is_loaded_before_training_is_enabled() {
        let root = fixture_colmap_root();

        let snapshot = snapshot_after_project_colmap_result(&root).unwrap();

        assert!(snapshot.has_dataset);
        assert!(snapshot.primary_command().enabled);
    }

    #[test]
    fn project_colmap_load_starts_training_only_after_success() {
        assert!(should_start_training_after_project_colmap_load(true));
        assert!(!should_start_training_after_project_colmap_load(false));
    }

    #[test]
    fn imported_image_project_starts_through_full_frame_pose_coverage() {
        assert_eq!(
            image_project_reconstruction_command(),
            PipelineCommand::StartThrough {
                stage: ProjectStage::FullFramePnp,
            }
        );
    }

    #[test]
    fn failed_keyframe_pose_solve_restarts_through_full_frame_coverage() {
        let mut manifest = crate::project::ProjectManifest::new(
            "Retry fixture",
            SourceSpec::managed_images("fixture"),
        );
        manifest.stage_mut(ProjectStage::KeyframeSfm).state = crate::project::StageState::Failed;

        assert_eq!(
            image_project_reconstruction_commands(&manifest),
            [
                PipelineCommand::RestartFrom {
                    stage: ProjectStage::KeyframeSfm,
                    confirmed: true,
                },
                PipelineCommand::StartThrough {
                    stage: ProjectStage::FullFramePnp,
                },
            ]
        );
    }

    #[test]
    fn active_project_pipeline_requests_a_follow_up_frame() {
        assert_eq!(
            project_pipeline_repaint_delay(true, true),
            Some(Duration::from_millis(16))
        );
        assert_eq!(project_pipeline_repaint_delay(true, false), None);
        assert_eq!(project_pipeline_repaint_delay(false, true), None);
    }

    #[test]
    fn failed_pose_stage_reports_the_persisted_project_error() {
        let mut manifest = crate::project::ProjectManifest::new(
            "Failure fixture",
            SourceSpec::managed_images("fixture"),
        );
        let stage = manifest.stage_mut(ProjectStage::KeyframeSfm);
        stage.state = crate::project::StageState::Failed;
        stage.error = Some(ProjectErrorRecord {
            code: "rustsfm_failed".to_owned(),
            stage: ProjectStage::KeyframeSfm,
            summary: "RustSFM pose solve failed".to_owned(),
            detail: "Feature matching exhausted the available pairs.".to_owned(),
            frame_id: None,
            pair: None,
            retryable: true,
            suggested_actions: vec![SuggestedAction::Retry],
        });

        assert_eq!(
            reconstruction_failure_message(&manifest).as_deref(),
            Some("Feature matching exhausted the available pairs.")
        );
    }

    #[test]
    fn persisted_pose_failure_ignores_late_pnp_progress() {
        let mut manifest = crate::project::ProjectManifest::new(
            "Failure fixture",
            SourceSpec::managed_images("fixture"),
        );
        manifest.stage_mut(ProjectStage::Import).state = crate::project::StageState::Succeeded;
        manifest.stage_mut(ProjectStage::KeyframeSfm).state = crate::project::StageState::Succeeded;
        let stage = manifest.stage_mut(ProjectStage::FullFramePnp);
        stage.state = crate::project::StageState::Failed;
        stage.error = Some(ProjectErrorRecord {
            code: "rustsfm_failed".to_owned(),
            stage: ProjectStage::FullFramePnp,
            summary: "RustSFM pose solve failed".to_owned(),
            detail: "GPU PnP-focal fallback could not solve".to_owned(),
            frame_id: None,
            pair: None,
            retryable: true,
            suggested_actions: vec![SuggestedAction::Retry],
        });
        let summary = ProjectSessionSummary::from_manifest(&manifest);

        assert!(should_ignore_pipeline_progress(
            Some(&summary),
            ProjectStage::FullFramePnp
        ));
        assert!(!should_ignore_pipeline_progress(
            Some(&summary),
            ProjectStage::Import
        ));
    }

    fn fixture_colmap_root() -> PathBuf {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.keep();
        let sparse = root.join("sparse/0");
        let images = root.join("images");
        fs::create_dir_all(&sparse).unwrap();
        fs::create_dir_all(&images).unwrap();
        image::RgbaImage::from_pixel(2, 2, image::Rgba([12, 34, 56, 255]))
            .save(images.join("00000000.png"))
            .unwrap();
        fs::write(
            sparse.join("cameras.txt"),
            "1 PINHOLE 2 2 1.0 1.0 1.0 1.0\n",
        )
        .unwrap();
        fs::write(
            sparse.join("images.txt"),
            "1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 00000000.png\n\n",
        )
        .unwrap();
        fs::write(
            sparse.join("points3D.txt"),
            "1 0.0 0.0 1.0 128 128 128 0.1 1 0\n",
        )
        .unwrap();
        root
    }

    #[test]
    fn activity_updates_replace_the_latest_event_for_the_same_stage() {
        let mut activity = std::collections::VecDeque::new();
        record_workbench_activity(
            &mut activity,
            "Pose solve",
            "Feature matching: 4 poses",
            workbench::PipelineStageState::Running,
        );
        record_workbench_activity(
            &mut activity,
            "Pose solve",
            "Feature matching: 12 poses",
            workbench::PipelineStageState::Running,
        );

        assert_eq!(activity.len(), 1);
        assert_eq!(activity[0].detail, "Feature matching: 12 poses");
    }

    #[test]
    fn clear_scene_preserving_layers_removes_data_without_resetting_visibility() {
        let mut scene = Scene::default();
        scene.layers.trajectory = false;
        scene.layers.map_points = false;
        scene.layers.gaussians = true;
        scene.layers.mesh_wireframe = true;
        scene.layers.mesh_solid = false;
        scene.trajectory.push([1.0, 2.0, 3.0]);
        scene.map_points.push([4.0, 5.0, 6.0]);
        scene.map_point_colors.push([0.1, 0.2, 0.3]);
        scene.gaussians.push(GaussianSplat {
            position: [7.0, 8.0, 9.0],
            scale: [1.0, 1.0, 1.0],
            rotation: [1.0, 0.0, 0.0, 0.0],
            opacity: 0.5,
            color: [0.3, 0.4, 0.5],
        });
        scene.mesh_vertices.push(MeshGpuVertex {
            position: [0.0, 1.0, 2.0],
            normal: [0.0, 0.0, 1.0],
            color: [1.0, 1.0, 1.0],
        });
        scene.mesh_indices.push(0);
        scene.mesh_edge_indices.push(0);
        scene.bounds.extend([7.0, 8.0, 9.0]);

        clear_scene_preserving_layers(&mut scene);

        assert!(!scene.has_data());
        assert!(scene.map_point_colors.is_empty());
        assert!(scene.mesh_indices.is_empty());
        assert!(scene.mesh_edge_indices.is_empty());
        assert!(!scene.bounds.is_valid());
        assert!(!scene.layers.trajectory);
        assert!(!scene.layers.map_points);
        assert!(scene.layers.gaussians);
        assert!(scene.layers.mesh_wireframe);
        assert!(!scene.layers.mesh_solid);
    }

    #[test]
    fn texture_render_only_needed_when_dirty_missing_or_resized() {
        let resolution = PreviewResolution::new(640, 480).unwrap();
        let cached = Some(CachedTexture {
            id: egui::TextureId::User(7),
            resolution,
        });

        assert!(needs_texture_render(true, cached, resolution));
        assert!(needs_texture_render(false, None, resolution));
        assert!(needs_texture_render(
            false,
            cached,
            PreviewResolution::new(320, 240).unwrap()
        ));
        assert!(!needs_texture_render(false, cached, resolution));
    }

    #[test]
    fn viewport_center_ray_points_at_camera_target() {
        let camera = ArcballCamera::default();
        let rect = Rect::from_min_size(egui::pos2(0.0, 0.0), Vec2::new(800.0, 600.0));
        let (origin, dir) = viewport_ray(&camera, rect, rect.center()).unwrap();
        let to_target = (camera.target - origin).normalize();

        assert!(dir.dot(to_target) > 0.999);
    }

    #[test]
    fn focus_from_camera_depth_unprojects_center_ray() {
        let camera = ArcballCamera::default();
        let rect = Rect::from_min_size(egui::pos2(0.0, 0.0), Vec2::new(800.0, 600.0));
        let (_, dir) = viewport_ray(&camera, rect, rect.center()).unwrap();

        let picked = focus_from_camera_depth(&camera, dir, camera.distance).unwrap();

        assert!((picked - Vec3::ZERO).length() < 1e-4);
    }

    #[test]
    fn pick_mesh_focus_hits_triangle() {
        let mut scene = Scene::default();
        scene.mesh_vertices = vec![
            MeshGpuVertex {
                position: [-1.0, -1.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                color: [1.0, 1.0, 1.0],
            },
            MeshGpuVertex {
                position: [1.0, -1.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                color: [1.0, 1.0, 1.0],
            },
            MeshGpuVertex {
                position: [0.0, 1.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                color: [1.0, 1.0, 1.0],
            },
        ];
        scene.mesh_indices = vec![0, 1, 2];

        let hit = pick_mesh_focus(&scene, (Vec3::new(0.0, 0.0, 5.0), Vec3::NEG_Z)).unwrap();

        assert!((hit - Vec3::ZERO).length() < 1e-4);
    }
}
