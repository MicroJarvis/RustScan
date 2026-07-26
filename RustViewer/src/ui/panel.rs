//! Left-side control panel UI.

use egui::{Color32, Vec2};

use crate::renderer::camera::ArcballCamera;
use crate::renderer::scene::Scene;
use crate::robot::{NavigationMode, RobotCameraMode};
use crate::training::{TrainingProgress, TrainingSessionState};
use crate::ui::theme::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelAction {
    OpenCheckpoint,
    OpenGaussian,
    OpenMesh,
    OpenColmap,
    OpenImages,
    RunReconstruction,
    StartTraining,
    StopTraining,
    AutoFitScene,
    ResetRobot,
    SnapRobotToGround,
    PlaceRobotInView,
    PickRobotGround,
    FlipRobotGround,
}

#[derive(Debug, Clone)]
pub struct DatasetUiSummary {
    pub root_path: String,
    pub frame_count: usize,
    pub sparse_point_count: usize,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageSourceSummary {
    pub root_path: String,
    pub image_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconstructionUiState {
    Idle,
    Ready,
    Running,
    Completed,
    Failed,
}

impl Default for ReconstructionUiState {
    fn default() -> Self {
        Self::Idle
    }
}

#[derive(Debug, Clone)]
pub struct TrainingControls {
    pub iterations: usize,
    pub render_scale: f32,
    pub litegs_mode: bool,
    pub progress_every: usize,
    pub snapshot_every: usize,
}

impl Default for TrainingControls {
    fn default() -> Self {
        Self {
            iterations: 1000,
            render_scale: 0.5,
            litegs_mode: false,
            progress_every: 5,
            snapshot_every: 25,
        }
    }
}

/// State shared between the UI and app logic.
#[derive(Debug, Clone)]
pub struct UiState {
    pub load_error: Option<String>,
    pub is_loading: bool,
    pub loading_message: Option<String>,
    pub dataset_summary: Option<DatasetUiSummary>,
    pub image_source: Option<ImageSourceSummary>,
    pub reconstruction_state: ReconstructionUiState,
    pub reconstruction_registered_images: usize,
    pub reconstruction_points: usize,
    pub reconstruction_error: Option<String>,
    pub training_controls: TrainingControls,
    pub training_state: TrainingSessionState,
    pub training_progress: TrainingProgress,
    pub training_error: Option<String>,
    pub preview_error: Option<String>,
    pub navigation_mode: NavigationMode,
    pub robot_camera_mode: RobotCameraMode,
    pub robot_visible: bool,
    pub robot_move_speed: f32,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            load_error: None,
            is_loading: false,
            loading_message: None,
            dataset_summary: None,
            image_source: None,
            reconstruction_state: ReconstructionUiState::Idle,
            reconstruction_registered_images: 0,
            reconstruction_points: 0,
            reconstruction_error: None,
            training_controls: TrainingControls::default(),
            training_state: TrainingSessionState::Idle,
            training_progress: TrainingProgress::default(),
            training_error: None,
            preview_error: None,
            navigation_mode: NavigationMode::Orbit,
            robot_camera_mode: RobotCameraMode::Follow,
            robot_visible: true,
            robot_move_speed: 1.0,
        }
    }
}

impl UiState {
    pub fn can_load_colmap(&self) -> bool {
        !self.is_loading
            && !reconstruction_is_running(self.reconstruction_state)
            && !reconstruction_is_blocked_by_training(self.training_state)
    }

    pub fn can_run_reconstruction(&self) -> bool {
        self.image_source.is_some()
            && !self.is_loading
            && !reconstruction_is_running(self.reconstruction_state)
            && !reconstruction_is_blocked_by_training(self.training_state)
    }

    pub fn can_start_training(&self) -> bool {
        self.dataset_summary.is_some()
            && !self.is_loading
            && !reconstruction_is_running(self.reconstruction_state)
    }
}

pub(crate) fn reconstruction_is_running(state: ReconstructionUiState) -> bool {
    matches!(state, ReconstructionUiState::Running)
}

pub(crate) fn reconstruction_is_blocked_by_training(state: TrainingSessionState) -> bool {
    matches!(
        state,
        TrainingSessionState::Loading
            | TrainingSessionState::Starting
            | TrainingSessionState::Training
            | TrainingSessionState::Stopping
    )
}

/// Draw the left-side control panel.
pub fn draw_side_panel(
    ui: &mut egui::Ui,
    state: &mut UiState,
    scene: &mut Scene,
    camera: &mut ArcballCamera,
) -> Vec<PanelAction> {
    let mut actions = Vec::new();
    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing = Vec2::new(0.0, 12.0);

        // Loading indicator
        if state.is_loading {
            draw_loading_indicator(ui, state);
        }

        // Error alert
        if state.load_error.is_some() {
            let error = state.load_error.clone().unwrap();
            draw_error_alert(ui, &error, state);
        }

        // FILE OPERATIONS section
        draw_section_header(ui, "FILE OPERATIONS");

        if draw_blue_button(ui, "📂", "Load Checkpoint") {
            actions.push(PanelAction::OpenCheckpoint);
        }

        if draw_blue_button(ui, "✨", "Load Gaussians") {
            actions.push(PanelAction::OpenGaussian);
        }

        if draw_blue_button(ui, "🔷", "Load Mesh") {
            actions.push(PanelAction::OpenMesh);
        }

        let load_colmap_clicked = ui
            .add_enabled_ui(state.can_load_colmap(), |ui| {
                draw_blue_button(ui, "🗂", "Load COLMAP")
            })
            .inner;
        if load_colmap_clicked {
            actions.push(PanelAction::OpenColmap);
        }

        let importing_images_enabled =
            !matches!(state.reconstruction_state, ReconstructionUiState::Running);
        let import_images_clicked = ui
            .add_enabled_ui(importing_images_enabled, |ui| {
                draw_blue_button(ui, "🖼", "Import Images")
            })
            .inner;
        if import_images_clicked {
            actions.push(PanelAction::OpenImages);
        }

        // Divider
        draw_divider(ui);

        draw_section_header(ui, "DATASET");
        draw_dataset_summary(ui, state);

        draw_divider(ui);

        draw_section_header(ui, "TRAINING");
        draw_training_controls(ui, state, &mut actions);

        draw_divider(ui);

        draw_section_header(ui, "ROBOT NAVIGATION");
        draw_robot_controls(ui, state, &mut actions);

        draw_divider(ui);

        // SCENE LAYERS section
        draw_section_header(ui, "SCENE LAYERS");

        draw_layer_toggle(
            ui,
            &mut scene.layers.trajectory,
            "Camera Trajectory",
            SYSTEM_BLUE,
        );
        draw_layer_toggle(ui, &mut scene.layers.map_points, "Map Points", SYSTEM_GREEN);
        draw_layer_toggle(ui, &mut scene.layers.gaussians, "Gaussians", SYSTEM_ORANGE);
        draw_layer_toggle(
            ui,
            &mut scene.layers.mesh_wireframe,
            "Mesh Wireframe",
            SYSTEM_GRAY,
        );
        draw_layer_toggle(ui, &mut scene.layers.mesh_solid, "Mesh Solid", SYSTEM_GRAY);

        // Divider
        draw_divider(ui);

        // SCENE STATISTICS section
        draw_section_header(ui, "SCENE STATISTICS");

        // Divider
        draw_divider(ui);

        // Auto Fit button
        if draw_auto_fit_button(ui) && scene.has_data() {
            camera.fit_scene(&scene.bounds);
            actions.push(PanelAction::AutoFitScene);
        }

        // Statistics cards - always visible
        draw_stats_cards(ui, scene);
    });

    actions
}

fn draw_loading_indicator(ui: &mut egui::Ui, state: &UiState) {
    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing = Vec2::new(0.0, 6.0);

        // Progress bar
        let (rect, _) =
            ui.allocate_exact_size(Vec2::new(ui.available_width(), 3.0), egui::Sense::hover());
        ui.painter().rect_filled(rect, 2.0, SYSTEM_BLUE);

        // Loading text
        let msg = state.loading_message.as_deref().unwrap_or("Loading...");
        ui.label(egui::RichText::new(msg).size(12.0).color(TEXT_PRIMARY));
    });
}

fn draw_error_alert(ui: &mut egui::Ui, error: &str, state: &mut UiState) {
    let frame = egui::Frame::new()
        .fill(Color32::from_rgba_unmultiplied(255, 59, 48, 36))
        .corner_radius(6.0)
        .inner_margin(12.0);

    frame.show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("⚠️").size(16.0));
            ui.label(egui::RichText::new(error).size(12.0).color(SYSTEM_RED));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .button(egui::RichText::new("×").size(16.0).color(SYSTEM_GRAY))
                    .clicked()
                {
                    state.load_error = None;
                }
            });
        });
    });
}

fn draw_section_header(ui: &mut egui::Ui, text: &str) {
    ui.label(
        egui::RichText::new(text)
            .size(11.0)
            .strong()
            .color(SYSTEM_GRAY),
    );
}

fn draw_dataset_summary(ui: &mut egui::Ui, state: &UiState) {
    let frame = egui::Frame::new()
        .fill(CARD_BG)
        .corner_radius(8.0)
        .inner_margin(12.0);

    frame.show(ui, |ui| {
        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing = Vec2::new(0.0, 8.0);

            match &state.dataset_summary {
                Some(summary) => {
                    ui.label(
                        egui::RichText::new(summary.root_path.as_str())
                            .size(11.0)
                            .color(TEXT_SECONDARY),
                    );
                    draw_metric_row(ui, "Frames", summary.frame_count.to_string());
                    draw_metric_row(ui, "Sparse Points", summary.sparse_point_count.to_string());
                    draw_metric_row(
                        ui,
                        "Resolution",
                        format!("{}x{}", summary.width, summary.height),
                    );
                }
                None => {
                    ui.label(
                        egui::RichText::new("No COLMAP dataset loaded")
                            .size(12.0)
                            .color(TEXT_SECONDARY),
                    );
                }
            }
        });
    });
}

fn draw_training_controls(ui: &mut egui::Ui, state: &mut UiState, actions: &mut Vec<PanelAction>) {
    let frame = egui::Frame::new()
        .fill(CARD_BG)
        .corner_radius(8.0)
        .inner_margin(12.0);

    frame.show(ui, |ui| {
        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing = Vec2::new(0.0, 10.0);

            draw_reconstruction_controls(ui, state, actions);
            draw_divider(ui);

            draw_metric_row(ui, "State", training_state_label(state.training_state));

            ui.add(
                egui::Slider::new(&mut state.training_controls.iterations, 10..=30_000)
                    .text("Iterations")
                    .logarithmic(true),
            );
            ui.add(
                egui::Slider::new(&mut state.training_controls.render_scale, 0.125..=1.0)
                    .text("Render Scale"),
            );
            ui.add(
                egui::Slider::new(&mut state.training_controls.progress_every, 1..=500)
                    .text("Progress Every"),
            );
            ui.add(
                egui::Slider::new(&mut state.training_controls.snapshot_every, 1..=1000)
                    .text("Snapshot Every"),
            );
            ui.checkbox(&mut state.training_controls.litegs_mode, "LiteGS Mode");

            if let Some(error) = &state.training_error {
                ui.label(egui::RichText::new(error).size(11.0).color(SYSTEM_RED));
            }
            if let Some(error) = &state.preview_error {
                ui.label(
                    egui::RichText::new(format!("Preview: {error}"))
                        .size(11.0)
                        .color(SYSTEM_ORANGE),
                );
            }

            draw_metric_row(
                ui,
                "Iteration",
                state
                    .training_progress
                    .latest_iteration
                    .map(|value| {
                        if let Some(total) = state.training_progress.total_iterations {
                            format!("{value}/{total}")
                        } else {
                            value.to_string()
                        }
                    })
                    .unwrap_or_else(|| "—".to_string()),
            );
            draw_metric_row(
                ui,
                "Loss",
                state
                    .training_progress
                    .latest_loss
                    .map(|value| format!("{value:.5}"))
                    .unwrap_or_else(|| "—".to_string()),
            );
            draw_metric_row(
                ui,
                "Gaussians",
                state
                    .training_progress
                    .gaussian_count
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "—".to_string()),
            );
            draw_metric_row(
                ui,
                "Elapsed",
                format_duration(state.training_progress.elapsed),
            );

            let training_active = reconstruction_is_blocked_by_training(state.training_state);

            let button_label = if training_active {
                "Stop Training"
            } else {
                "Start Training"
            };
            let button_icon = if training_active { "⏹" } else { "▶" };
            let clicked = if training_active {
                draw_secondary_button(ui, button_icon, button_label)
            } else {
                ui.add_enabled_ui(state.can_start_training(), |ui| {
                    draw_blue_button(ui, button_icon, button_label)
                })
                .inner
            };

            if clicked {
                actions.push(if training_active {
                    PanelAction::StopTraining
                } else {
                    PanelAction::StartTraining
                });
            }
        });
    });
}

fn draw_reconstruction_controls(
    ui: &mut egui::Ui,
    state: &UiState,
    actions: &mut Vec<PanelAction>,
) {
    draw_metric_row(
        ui,
        "Reconstruction",
        reconstruction_state_label(state.reconstruction_state),
    );

    if let Some(source) = &state.image_source {
        ui.label(
            egui::RichText::new(source.root_path.as_str())
                .size(11.0)
                .color(TEXT_SECONDARY),
        );
        draw_metric_row(ui, "Images", source.image_count.to_string());
    }

    if matches!(state.reconstruction_state, ReconstructionUiState::Running) {
        draw_metric_row(
            ui,
            "Registered Images",
            state.reconstruction_registered_images.to_string(),
        );
        draw_metric_row(ui, "Sparse Points", state.reconstruction_points.to_string());
    }

    if matches!(state.reconstruction_state, ReconstructionUiState::Failed) {
        if let Some(error) = &state.reconstruction_error {
            ui.label(egui::RichText::new(error).size(11.0).color(SYSTEM_RED));
        }
    }

    let run_reconstruction_clicked = ui
        .add_enabled_ui(state.can_run_reconstruction(), |ui| {
            draw_blue_button(ui, "▶", "Run Reconstruction")
        })
        .inner;
    if run_reconstruction_clicked {
        actions.push(PanelAction::RunReconstruction);
    }
}

fn draw_robot_controls(ui: &mut egui::Ui, state: &mut UiState, actions: &mut Vec<PanelAction>) {
    ui.horizontal(|ui| {
        let orbit = state.navigation_mode == NavigationMode::Orbit;
        if ui.selectable_label(orbit, "Orbit").clicked() {
            state.navigation_mode = NavigationMode::Orbit;
        }
        let robot = state.navigation_mode == NavigationMode::Robot;
        if ui.selectable_label(robot, "Robot").clicked() {
            state.navigation_mode = NavigationMode::Robot;
        }
    });

    ui.horizontal(|ui| {
        let follow = state.robot_camera_mode == RobotCameraMode::Follow;
        if ui.selectable_label(follow, "Follow").clicked() {
            state.robot_camera_mode = RobotCameraMode::Follow;
        }
        let first_person = state.robot_camera_mode == RobotCameraMode::FirstPerson;
        if ui.selectable_label(first_person, "First Person").clicked() {
            state.robot_camera_mode = RobotCameraMode::FirstPerson;
        }
    });

    ui.add(
        egui::Slider::new(&mut state.robot_move_speed, 0.05..=10.0)
            .text("Speed")
            .clamping(egui::SliderClamping::Always),
    );
    ui.checkbox(&mut state.robot_visible, "Show Unitree G1");

    ui.horizontal(|ui| {
        if draw_secondary_button(ui, "⊙", "Place In View") {
            actions.push(PanelAction::PlaceRobotInView);
        }
        if draw_secondary_button(ui, "⟲", "Reset Robot") {
            actions.push(PanelAction::ResetRobot);
        }
    });

    ui.horizontal(|ui| {
        if draw_secondary_button(ui, "⌄", "Snap Ground") {
            actions.push(PanelAction::SnapRobotToGround);
        }
        if draw_secondary_button(ui, "◎", "Set Ground") {
            actions.push(PanelAction::PickRobotGround);
        }
    });

    ui.horizontal(|ui| {
        if draw_secondary_button(ui, "⇅", "Flip Ground") {
            actions.push(PanelAction::FlipRobotGround);
        }
    });
}

fn draw_metric_row(ui: &mut egui::Ui, label: &str, value: impl Into<String>) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).size(11.0).color(TEXT_SECONDARY));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                egui::RichText::new(value.into())
                    .size(12.0)
                    .strong()
                    .color(TEXT_PRIMARY),
            );
        });
    });
}

fn training_state_label(state: TrainingSessionState) -> String {
    match state {
        TrainingSessionState::Idle => "idle",
        TrainingSessionState::Loading => "loading",
        TrainingSessionState::Starting => "starting",
        TrainingSessionState::Training => "training",
        TrainingSessionState::Stopping => "stopping",
        TrainingSessionState::Completed => "completed",
        TrainingSessionState::Failed => "failed",
        TrainingSessionState::Cancelled => "cancelled",
    }
    .to_string()
}

pub fn reconstruction_state_label(state: ReconstructionUiState) -> &'static str {
    match state {
        ReconstructionUiState::Idle => "idle",
        ReconstructionUiState::Ready => "ready",
        ReconstructionUiState::Running => "solving poses",
        ReconstructionUiState::Completed => "completed",
        ReconstructionUiState::Failed => "failed",
    }
}

fn format_duration(duration: std::time::Duration) -> String {
    let secs = duration.as_secs();
    format!("{:02}:{:02}", secs / 60, secs % 60)
}

fn draw_blue_button(ui: &mut egui::Ui, icon: &str, label: &str) -> bool {
    let button_height = 32.0;
    let (rect, response) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), button_height),
        egui::Sense::click(),
    );

    let bg_color = if response.clicked() {
        Color32::from_rgb(0, 85, 200)
    } else if response.hovered() {
        Color32::from_rgb(0, 110, 230)
    } else {
        SYSTEM_BLUE
    };

    ui.painter().rect_filled(rect, 6.0, bg_color);

    // Calculate proper centering with 8px gap between icon and text
    let font_id = egui::FontId::proportional(13.0);
    let icon_width = ui.fonts_mut(|f| f.glyph_width(&font_id, icon.chars().next().unwrap()));
    let text_width = ui.fonts_mut(|f| {
        f.layout_no_wrap(label.to_string(), font_id.clone(), Color32::WHITE)
            .size()
            .x
    });
    let gap = 8.0;
    let total_width = icon_width + gap + text_width;

    let start_x = rect.center().x - total_width / 2.0;
    let center_y = rect.center().y;

    ui.painter().text(
        egui::pos2(start_x + icon_width / 2.0, center_y),
        egui::Align2::CENTER_CENTER,
        icon,
        font_id.clone(),
        Color32::WHITE,
    );
    ui.painter().text(
        egui::pos2(start_x + icon_width + gap + text_width / 2.0, center_y),
        egui::Align2::CENTER_CENTER,
        label,
        font_id,
        Color32::WHITE,
    );

    response.clicked()
}

fn draw_divider(ui: &mut egui::Ui) {
    let (rect, _) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), 1.0), egui::Sense::hover());
    ui.painter().rect_filled(rect, 0.0, SEPARATOR);
}

fn draw_secondary_button(ui: &mut egui::Ui, icon: &str, label: &str) -> bool {
    let button_height = 32.0;
    let (rect, response) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), button_height),
        egui::Sense::click(),
    );

    let bg_color = if response.hovered() {
        hover_bg()
    } else {
        CARD_BG
    };

    ui.painter().rect_filled(rect, 6.0, bg_color);
    ui.painter().rect_stroke(
        rect,
        6.0,
        egui::Stroke::new(1.0, SEPARATOR),
        egui::StrokeKind::Outside,
    );

    let font_id = egui::FontId::proportional(13.0);
    let icon_width = ui.fonts_mut(|f| f.glyph_width(&font_id, icon.chars().next().unwrap()));
    let text_width = ui.fonts_mut(|f| {
        f.layout_no_wrap(label.to_string(), font_id.clone(), TEXT_PRIMARY)
            .size()
            .x
    });
    let gap = 8.0;
    let total_width = icon_width + gap + text_width;
    let start_x = rect.center().x - total_width / 2.0;
    let center_y = rect.center().y;

    ui.painter().text(
        egui::pos2(start_x + icon_width / 2.0, center_y),
        egui::Align2::CENTER_CENTER,
        icon,
        font_id.clone(),
        TEXT_PRIMARY,
    );
    ui.painter().text(
        egui::pos2(start_x + icon_width + gap + text_width / 2.0, center_y),
        egui::Align2::CENTER_CENTER,
        label,
        font_id,
        TEXT_PRIMARY,
    );

    response.clicked()
}

fn draw_layer_toggle(ui: &mut egui::Ui, checked: &mut bool, label: &str, color: Color32) {
    let toggle_height = 32.0;
    let (rect, response) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), toggle_height),
        egui::Sense::click(),
    );

    if response.clicked() {
        *checked = !*checked;
    }

    // Draw checkbox
    let checkbox_size = 16.0;
    let checkbox_pos = rect.left_center() + Vec2::new(12.0, -checkbox_size / 2.0);
    let checkbox_rect =
        egui::Rect::from_min_size(checkbox_pos.into(), Vec2::new(checkbox_size, checkbox_size));

    if *checked {
        ui.painter().rect_filled(checkbox_rect, 4.0, SYSTEM_BLUE);
        // Draw checkmark
        ui.painter().text(
            checkbox_rect.center(),
            egui::Align2::CENTER_CENTER,
            "✓",
            egui::FontId::proportional(10.0),
            Color32::WHITE,
        );
    } else {
        // Draw empty checkbox with rounded border
        ui.painter().rect_stroke(
            checkbox_rect,
            4.0,
            egui::Stroke::new(1.0, SYSTEM_GRAY),
            egui::StrokeKind::Outside,
        );
    }

    // Draw color badge
    let badge_size = 12.0;
    let badge_pos =
        checkbox_pos + Vec2::new(checkbox_size + 16.0, (checkbox_size - badge_size) / 2.0);
    let badge_rect = egui::Rect::from_min_size(badge_pos.into(), Vec2::new(badge_size, badge_size));
    ui.painter()
        .circle_filled(badge_rect.center(), badge_size / 2.0, color);

    // Draw label
    let label_pos = badge_pos + Vec2::new(badge_size + 8.0, badge_size / 2.0);
    ui.painter().text(
        label_pos.into(),
        egui::Align2::LEFT_CENTER,
        label,
        egui::FontId::proportional(13.0),
        TEXT_PRIMARY,
    );
}

fn draw_auto_fit_button(ui: &mut egui::Ui) -> bool {
    let button_height = 32.0;
    let (rect, response) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), button_height),
        egui::Sense::click(),
    );

    let bg_color = if response.hovered() {
        hover_bg()
    } else {
        Color32::TRANSPARENT
    };

    ui.painter().rect_filled(rect, 6.0, bg_color);

    // Calculate proper centering with 8px gap between icon and text
    let icon = "🎯";
    let label = "Auto Fit Scene";
    let font_id = egui::FontId::proportional(13.0);
    let icon_width = ui.fonts_mut(|f| f.glyph_width(&font_id, icon.chars().next().unwrap()));
    let text_width = ui.fonts_mut(|f| {
        f.layout_no_wrap(label.to_string(), font_id.clone(), TEXT_PRIMARY)
            .size()
            .x
    });
    let gap = 8.0;
    let total_width = icon_width + gap + text_width;

    let start_x = rect.center().x - total_width / 2.0;
    let center_y = rect.center().y;

    ui.painter().text(
        egui::pos2(start_x + icon_width / 2.0, center_y),
        egui::Align2::CENTER_CENTER,
        icon,
        font_id.clone(),
        TEXT_PRIMARY,
    );
    ui.painter().text(
        egui::pos2(start_x + icon_width + gap + text_width / 2.0, center_y),
        egui::Align2::CENTER_CENTER,
        label,
        font_id,
        TEXT_PRIMARY,
    );

    response.clicked()
}

fn draw_stats_cards(ui: &mut egui::Ui, scene: &Scene) {
    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing = Vec2::new(0.0, 8.0);

        // Get values - show "—" when no data loaded
        let mesh_vertices = if scene.has_data() {
            scene.mesh_vertex_count()
        } else {
            0
        };

        let keyframes = if scene.has_data() {
            scene.keyframe_count()
        } else {
            0
        };

        let map_points = if scene.has_data() {
            scene.map_point_count()
        } else {
            0
        };

        let gaussians = if scene.has_data() {
            scene.gaussian_count()
        } else {
            0
        };

        // Mesh vertices card
        let frame = egui::Frame::new()
            .fill(CARD_BG)
            .corner_radius(8.0)
            .inner_margin(12.0);

        frame.show(ui, |ui| {
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing = Vec2::new(0.0, 8.0);
                ui.label(
                    egui::RichText::new("Mesh Vertices")
                        .size(11.0)
                        .color(SYSTEM_GRAY),
                );
                ui.label(
                    egui::RichText::new(format!("{}", mesh_vertices))
                        .size(15.0)
                        .strong()
                        .color(TEXT_PRIMARY),
                );
            });
        });

        // Stats card with multiple rows
        let frame = egui::Frame::new()
            .fill(CARD_BG)
            .corner_radius(8.0)
            .inner_margin(16.0);

        frame.show(ui, |ui| {
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing = Vec2::new(0.0, 12.0);

                // Keyframes row
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Keyframes")
                            .size(11.0)
                            .color(TEXT_SECONDARY),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(format!("{}", keyframes))
                                .size(15.0)
                                .strong()
                                .color(TEXT_PRIMARY),
                        );
                    });
                });

                // Map Points row
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Map Points")
                            .size(11.0)
                            .color(TEXT_SECONDARY),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(format!("{}", map_points))
                                .size(15.0)
                                .strong()
                                .color(TEXT_PRIMARY),
                        );
                    });
                });

                // Gaussians row
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Gaussians")
                            .size(11.0)
                            .color(TEXT_SECONDARY),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(format!("{}", gaussians))
                                .size(15.0)
                                .strong()
                                .color(TEXT_PRIMARY),
                        );
                    });
                });
            });
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstruction_can_run_only_with_source_and_idle_state() {
        let mut state = UiState::default();
        assert!(!state.can_run_reconstruction());
        state.image_source = Some(ImageSourceSummary {
            root_path: "/captures/chair".to_owned(),
            image_count: 24,
        });
        assert!(state.can_run_reconstruction());
        state.reconstruction_state = ReconstructionUiState::Running;
        assert!(!state.can_run_reconstruction());
        state.reconstruction_state = ReconstructionUiState::Ready;
        state.training_state = TrainingSessionState::Training;
        assert!(!state.can_run_reconstruction());
    }

    #[test]
    fn reconstruction_and_colmap_operations_exclude_conflicts() {
        let mut state = UiState::default();
        state.dataset_summary = Some(DatasetUiSummary {
            root_path: "/captures/chair/sparse".to_owned(),
            frame_count: 24,
            sparse_point_count: 1_024,
            width: 1_920,
            height: 1_080,
        });
        state.image_source = Some(ImageSourceSummary {
            root_path: "/captures/chair".to_owned(),
            image_count: 24,
        });

        assert!(state.can_load_colmap());
        assert!(state.can_run_reconstruction());

        state.is_loading = true;
        assert!(!state.can_load_colmap());
        assert!(!state.can_run_reconstruction());

        state.is_loading = false;
        state.reconstruction_state = ReconstructionUiState::Running;
        assert!(!state.can_load_colmap());

        state.reconstruction_state = ReconstructionUiState::Ready;
        state.training_state = TrainingSessionState::Training;
        assert!(!state.can_load_colmap());
    }

    #[test]
    fn reconstruction_running_blocks_training_start() {
        let mut state = UiState::default();
        state.dataset_summary = Some(DatasetUiSummary {
            root_path: "/captures/chair".to_owned(),
            frame_count: 24,
            sparse_point_count: 1_024,
            width: 1_920,
            height: 1_080,
        });
        assert!(state.can_start_training());

        state.is_loading = true;
        assert!(!state.can_start_training());

        state.is_loading = false;
        state.reconstruction_state = ReconstructionUiState::Running;
        assert!(!state.can_start_training());
    }

    #[test]
    fn reconstruction_labels_are_operator_facing() {
        assert_eq!(
            reconstruction_state_label(ReconstructionUiState::Ready),
            "ready"
        );
        assert_eq!(
            reconstruction_state_label(ReconstructionUiState::Running),
            "solving poses"
        );
        assert_eq!(
            reconstruction_state_label(ReconstructionUiState::Failed),
            "failed"
        );
    }
}
