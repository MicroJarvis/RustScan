//! Main application struct implementing eframe::App.

use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui::{self, Color32, Rect, Sense, Vec2};
use eframe::egui_wgpu;
use eframe::wgpu;
use rustgs::{ColmapConfig, HostSplats, SharedWgpuContext, TrainingConfig};

use crate::loader::checkpoint::LoadError;
use crate::loader::{
    load_colmap_training_dataset, map_training_dataset_to_scene, LoadedColmapDataset,
};
use crate::renderer::camera::ArcballCamera;
use crate::renderer::scene::{GaussianSplat, Scene};
use crate::renderer::ViewerCallback;
use crate::training::gpu_viewport::{viewport_render_scale, GpuViewportBridge};
use crate::training::preview::PreviewResolution;
use crate::training::{TrainingControlOptions, TrainingManager, TrainingSessionEvent};
use crate::ui::panel::{draw_side_panel, DatasetUiSummary, PanelAction, UiState};
use crate::ui::theme::{
    overlay_bg, PANEL_BG, TEXT_PRIMARY, TEXT_SECONDARY, VIEWPORT_BG, WINDOW_BG,
};
use crate::ui::viewport::{draw_empty_state, draw_viewport_overlay};

const VIEWPORT_INTERACTIVE_IDLE_DELAY: Duration = Duration::from_millis(180);

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

enum AppCommand {
    LoadAsset { kind: AssetLoadKind, path: PathBuf },
    ColmapLoaded(Result<LoadedColmapDataset, String>),
}

pub struct ViewerApp {
    scene: Arc<Mutex<Scene>>,
    camera: ArcballCamera,
    preview_camera: ArcballCamera,
    ui_state: UiState,
    loaded_colmap: Option<LoadedColmapDataset>,
    command_rx: mpsc::Receiver<AppCommand>,
    command_tx: mpsc::Sender<AppCommand>,
    training_manager: TrainingManager,
    loaded_splats: Option<Arc<HostSplats>>,
    preview_bridge: Option<GpuViewportBridge>,
    preview_dirty: bool,
    preview_texture: Option<CachedTexture>,
    viewport_bridge: Option<GpuViewportBridge>,
    viewport_dirty: bool,
    viewport_texture: Option<CachedTexture>,
    viewport_last_motion: Option<Instant>,
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
        let preview_bridge =
            new_gpu_viewport_bridge(&shared_wgpu_context, cc.wgpu_render_state.as_ref());
        let viewport_bridge =
            new_gpu_viewport_bridge(&shared_wgpu_context, cc.wgpu_render_state.as_ref());
        let app = Self {
            scene: Arc::new(Mutex::new(Scene::default())),
            camera: ArcballCamera::default(),
            preview_camera: ArcballCamera::default(),
            ui_state: UiState::default(),
            loaded_colmap: None,
            command_rx,
            command_tx,
            training_manager: TrainingManager::new(),
            loaded_splats: None,
            preview_bridge,
            preview_dirty: true,
            preview_texture: None,
            viewport_bridge,
            viewport_dirty: true,
            viewport_texture: None,
            viewport_last_motion: None,
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
        self.preview_bridge = self.new_gpu_viewport_bridge();
        self.viewport_bridge = self.new_gpu_viewport_bridge();
        self.preview_texture = None;
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
                if let Ok(mut scene) = self.scene.lock() {
                    clear_scene_preserving_layers(&mut scene);
                    map_training_dataset_to_scene(&loaded.dataset, &mut scene);
                    if scene.has_data() {
                        self.camera.fit_scene(&scene.bounds);
                        self.preview_camera = self.camera.clone();
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
                self.preview_bridge = self.new_gpu_viewport_bridge();
                self.viewport_bridge = self.new_gpu_viewport_bridge();
                self.preview_texture = None;
                self.viewport_texture = None;
                self.viewport_dirty = true;
                self.viewport_last_motion = None;
                self.viewport_render_error = None;
                self.preview_dirty = true;
            }
            Err(err) => {
                self.ui_state.load_error = Some(err);
            }
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

    fn process_panel_actions(&mut self, actions: Vec<PanelAction>) {
        for action in actions {
            match action {
                PanelAction::OpenCheckpoint => self.spawn_file_dialog(AssetLoadKind::Checkpoint),
                PanelAction::OpenGaussian => self.spawn_file_dialog(AssetLoadKind::Gaussian),
                PanelAction::OpenMesh => self.spawn_file_dialog(AssetLoadKind::Mesh),
                PanelAction::OpenColmap => self.spawn_colmap_load(),
                PanelAction::StartTraining => self.start_training(),
                PanelAction::StopTraining => self.stop_training(),
                PanelAction::AutoFitScene => {
                    self.preview_dirty = true;
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
                self.preview_dirty = true;
            }
            Err(err) => {
                self.ui_state.training_error = Some(err.to_string());
            }
        }
    }

    fn stop_training(&mut self) {
        if let Err(err) = self.training_manager.stop() {
            self.ui_state.training_error = Some(err.to_string());
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
                    self.preview_dirty = true;
                }
                TrainingSessionEvent::Completed(report) => {
                    self.ui_state.training_progress.latest_loss = report.final_loss;
                    self.ui_state.training_progress.gaussian_count = Some(report.gaussian_count);
                    self.sync_latest_snapshot_to_scene();
                    self.preview_dirty = true;
                }
                TrainingSessionEvent::Failed(error) => {
                    self.ui_state.training_error = Some(error);
                }
                TrainingSessionEvent::Cancelled => {
                    self.preview_dirty = true;
                }
                TrainingSessionEvent::BackendEvent(_) => {}
            }
        }

        self.ui_state.training_state = self.training_manager.state();
        self.ui_state.training_progress = self.training_manager.progress();
        if let Some(error) = self.training_manager.latest_error() {
            self.ui_state.training_error = Some(error);
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
        }
        self.loaded_splats = Some(snapshot);
        self.viewport_dirty = true;
        self.viewport_render_error = None;
    }

    fn refresh_preview_texture_id(&mut self, size: Vec2) -> Option<egui::TextureId> {
        self.loaded_colmap.as_ref()?;
        let Some(snapshot) = self.training_manager.latest_snapshot() else {
            self.ui_state.preview_error = None;
            self.preview_dirty = false;
            self.preview_texture = None;
            return None;
        };
        if snapshot.is_empty() {
            self.ui_state.preview_error = None;
            self.preview_dirty = false;
            self.preview_texture = None;
            return None;
        }
        let Some(resolution) = PreviewResolution::from_panel_size_scaled(size, 1.0) else {
            return self.preview_texture.map(|texture| texture.id);
        };
        if !needs_texture_render(self.preview_dirty, self.preview_texture, resolution) {
            return self.preview_texture.map(|texture| texture.id);
        }
        let Some(preview_bridge) = self.preview_bridge.as_mut() else {
            self.ui_state.preview_error = Some("wgpu render state is unavailable".to_string());
            self.preview_dirty = false;
            return None;
        };

        match preview_bridge.render_texture_id(
            self.wgpu_render_state.as_ref(),
            snapshot,
            &self.preview_camera,
            size,
            1.0,
        ) {
            Ok(texture_id) => {
                self.ui_state.preview_error = None;
                self.preview_dirty = false;
                self.preview_texture = texture_id.map(|id| CachedTexture { id, resolution });
                texture_id
            }
            Err(err) => {
                self.ui_state.preview_error = Some(err.to_string());
                self.preview_dirty = false;
                None
            }
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
            &self.camera,
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

    fn draw_preview_panel(&mut self, parent_ui: &mut egui::Ui) {
        if self.loaded_colmap.is_none() {
            return;
        }

        let ctx = parent_ui.ctx().clone();
        egui::Panel::right("training_preview")
            .min_size(360.0)
            .default_size(420.0)
            .frame(
                egui::Frame::new()
                    .fill(PANEL_BG)
                    .inner_margin(egui::Margin::same(20)),
            )
            .show_inside(parent_ui, |ui| {
                ui.vertical(|ui| {
                    ui.label(
                        egui::RichText::new("Live Preview")
                            .size(13.0)
                            .strong()
                            .color(TEXT_PRIMARY),
                    );
                    ui.label(
                        egui::RichText::new(
                            "Drag left to orbit, drag right to pan, scroll to zoom.",
                        )
                        .size(11.0)
                        .color(TEXT_SECONDARY),
                    );
                    ui.add_space(12.0);

                    let desired_size =
                        Vec2::new(ui.available_width(), ui.available_height().max(220.0));
                    let (rect, response) = ui.allocate_exact_size(desired_size, Sense::drag());

                    let mut preview_moved = false;
                    if response.dragged_by(egui::PointerButton::Primary) {
                        let delta = response.drag_motion();
                        if ui.input(|input| input.modifiers.shift) {
                            self.preview_camera.roll(delta.x);
                        } else {
                            self.preview_camera.orbit(delta.x, delta.y);
                        }
                        preview_moved = true;
                    }
                    if response.dragged_by(egui::PointerButton::Middle) {
                        let delta = response.drag_motion();
                        self.preview_camera.roll(delta.x);
                        preview_moved = true;
                    }
                    if response.dragged_by(egui::PointerButton::Secondary) {
                        let delta = response.drag_motion();
                        self.preview_camera.pan(delta.x, delta.y);
                        preview_moved = true;
                    }
                    if response.hovered() {
                        let scroll = ui.input(|input| input.smooth_scroll_delta.y);
                        if scroll != 0.0 {
                            self.preview_camera.zoom(scroll);
                            preview_moved = true;
                        }
                    }
                    if preview_moved {
                        self.preview_dirty = true;
                        ctx.request_repaint();
                    }

                    let texture_id = self.refresh_preview_texture_id(rect.size());
                    ui.painter().rect_filled(rect, 10.0, overlay_bg());

                    if let Some(texture_id) = texture_id {
                        ui.painter().image(
                            texture_id,
                            rect,
                            Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                            Color32::WHITE,
                        );
                    } else {
                        draw_preview_placeholder(
                            ui,
                            rect,
                            match self.ui_state.training_state {
                                crate::training::TrainingSessionState::Training
                                | crate::training::TrainingSessionState::Starting
                                | crate::training::TrainingSessionState::Stopping => {
                                    "Waiting for training snapshot…"
                                }
                                _ => "Start training to see live preview.",
                            },
                        );
                    }
                });
            });
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
        self.poll_training_events(&ctx);

        let (has_data, scene_bounds) = self
            .scene
            .lock()
            .map(|scene| (scene.has_data(), scene.bounds.clone()))
            .unwrap_or_default();

        egui::Panel::top("title_bar")
            .exact_size(52.0)
            .frame(egui::Frame::new().fill(WINDOW_BG))
            .show_inside(ui, |ui| {
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    ui.add_space(24.0);
                    ui.label(
                        egui::RichText::new("RustViewer")
                            .size(13.0)
                            .strong()
                            .color(TEXT_PRIMARY),
                    );
                });
            });

        let mut panel_actions = Vec::new();
        egui::Panel::left("control_panel")
            .exact_size(300.0)
            .frame(
                egui::Frame::new()
                    .fill(PANEL_BG)
                    .inner_margin(egui::Margin::same(24)),
            )
            .show_inside(ui, |ui| {
                if let Ok(mut scene) = self.scene.lock() {
                    panel_actions =
                        draw_side_panel(ui, &mut self.ui_state, &mut scene, &mut self.camera);
                }
            });
        self.process_panel_actions(panel_actions);

        self.draw_preview_panel(ui);

        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(VIEWPORT_BG))
            .show_inside(ui, |ui| {
                if !has_data {
                    draw_empty_state(ui);
                    draw_viewport_overlay(ui, &self.camera, has_data);
                    return;
                }

                let viewport_rect = ui.max_rect();
                let viewport_size = [viewport_rect.width(), viewport_rect.height()];

                let response = ui.allocate_rect(viewport_rect, egui::Sense::drag());

                let mut camera_moved = false;
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
                        draw_preview_placeholder(
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
                        camera: self.camera.clone(),
                        viewport_size,
                        surface_format: self.surface_format,
                    },
                );
                ui.painter().add(callback);

                draw_viewport_overlay(ui, &self.camera, has_data);

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

fn draw_preview_placeholder(ui: &egui::Ui, rect: Rect, message: &str) {
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        message,
        egui::FontId::proportional(13.0),
        TEXT_SECONDARY,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::scene::MeshGpuVertex;

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
}
