use egui::{Color32, Pos2, Rect, RichText, Stroke, Ui, Vec2};

use crate::project::{ProjectSessionSummary, ProjectStagePresentation};
use crate::renderer::scene::Scene;
use crate::training::TrainingSessionState;
use crate::ui::panel::{panel_action_for_primary_import, PanelAction, SourceSelection, UiState};
use crate::ui::theme::{
    CARD_BG, PANEL_BG, SEPARATOR, SYSTEM_BLUE, SYSTEM_GRAY, SYSTEM_GREEN, SYSTEM_ORANGE,
    SYSTEM_RED, TEXT_DISABLED, TEXT_PRIMARY, TEXT_SECONDARY,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStageState {
    Ready,
    Waiting,
    Running,
    Completed,
    Failed,
}

/// The target workbench keeps the reconstruction viewport as the dominant region.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorkbenchLayout {
    pub top_bar_height: f32,
    pub viewport_toolbar_height: f32,
    pub stage_rail_width: f32,
    pub inspector_width: f32,
    pub activity_height: f32,
    window_size: Vec2,
}

impl WorkbenchLayout {
    pub fn for_window(window_size: Vec2) -> Self {
        Self {
            top_bar_height: 52.0,
            viewport_toolbar_height: 48.0,
            stage_rail_width: 184.0,
            inspector_width: 288.0,
            activity_height: 128.0,
            window_size,
        }
    }

    pub fn viewport_width(self) -> f32 {
        self.window_size.x - self.stage_rail_width - self.inspector_width
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineStage {
    pub name: &'static str,
    pub detail: String,
    pub state: PipelineStageState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkbenchActivity {
    pub title: String,
    pub detail: String,
    pub state: PipelineStageState,
}

impl WorkbenchActivity {
    pub fn new(
        title: impl Into<String>,
        detail: impl Into<String>,
        state: PipelineStageState,
    ) -> Self {
        Self {
            title: title.into(),
            detail: detail.into(),
            state,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimaryAction {
    ImportImages,
    OpenColmap,
    SolvePoses,
    StartTraining,
    CancelTraining,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrimaryCommand {
    pub label: &'static str,
    pub action: PrimaryAction,
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub struct WorkbenchSnapshot {
    pub has_dataset: bool,
    pub project: Option<ProjectSessionSummary>,
    pub frame_count: usize,
    pub sparse_point_count: usize,
    pub training_state: TrainingSessionState,
    pub gaussian_count: Option<usize>,
    pub has_rendered_gaussians: bool,
    pub error: Option<String>,
    pub activity: Vec<WorkbenchActivity>,
}

impl Default for WorkbenchSnapshot {
    fn default() -> Self {
        Self {
            has_dataset: false,
            project: None,
            frame_count: 0,
            sparse_point_count: 0,
            training_state: TrainingSessionState::Idle,
            gaussian_count: None,
            has_rendered_gaussians: false,
            error: None,
            activity: Vec::new(),
        }
    }
}

impl WorkbenchSnapshot {
    pub fn primary_command(&self) -> PrimaryCommand {
        if matches!(
            self.training_state,
            TrainingSessionState::Loading
                | TrainingSessionState::Starting
                | TrainingSessionState::Training
                | TrainingSessionState::Stopping
        ) {
            return PrimaryCommand {
                label: "Cancel training",
                action: PrimaryAction::CancelTraining,
                enabled: true,
            };
        }

        if let Some(project) = &self.project {
            if project.source_kind == crate::project::SourceKind::Video {
                return PrimaryCommand {
                    label: "Video pose solve unavailable",
                    action: PrimaryAction::SolvePoses,
                    enabled: false,
                };
            }
            match project.pose_state {
                ProjectStagePresentation::Ready => {
                    return PrimaryCommand {
                        label: "运行重建",
                        action: PrimaryAction::SolvePoses,
                        enabled: true,
                    };
                }
                ProjectStagePresentation::Running => {
                    return PrimaryCommand {
                        label: "Solving poses",
                        action: PrimaryAction::SolvePoses,
                        enabled: false,
                    };
                }
                ProjectStagePresentation::Failed => {
                    return PrimaryCommand {
                        label: "Retry pose solve",
                        action: PrimaryAction::SolvePoses,
                        enabled: true,
                    };
                }
                ProjectStagePresentation::Waiting if !self.has_dataset => {
                    return PrimaryCommand {
                        label: "Waiting for poses",
                        action: PrimaryAction::SolvePoses,
                        enabled: false,
                    };
                }
                ProjectStagePresentation::Waiting | ProjectStagePresentation::Completed => {}
            }
        }

        if !self.has_dataset {
            return PrimaryCommand {
                label: "导入图像",
                action: PrimaryAction::ImportImages,
                enabled: true,
            };
        }

        PrimaryCommand {
            label: "Start training",
            action: PrimaryAction::StartTraining,
            enabled: self.project.as_ref().map_or(true, |project| {
                project.training_state == ProjectStagePresentation::Ready
            }),
        }
    }

    pub fn activity_feed(&self) -> &[WorkbenchActivity] {
        &self.activity
    }

    pub fn visible_frame_count(&self) -> usize {
        self.project
            .as_ref()
            .and_then(|project| project.imported_frame_count)
            .map(|count| count as usize)
            .unwrap_or(self.frame_count)
    }

    pub fn stages(&self) -> [PipelineStage; 4] {
        if let Some(project) = &self.project {
            return self.project_stages(project);
        }

        let imported = self.has_dataset;
        let import = PipelineStage {
            name: "导入采集",
            detail: if imported {
                format!("{} frames ready", self.frame_count)
            } else {
                "选择图像序列".to_string()
            },
            state: if imported {
                PipelineStageState::Completed
            } else {
                PipelineStageState::Ready
            },
        };
        let pose = PipelineStage {
            name: "位姿解算",
            detail: if imported {
                format!("{} sparse points", self.sparse_point_count)
            } else {
                "等待开始".to_string()
            },
            state: if imported {
                PipelineStageState::Completed
            } else {
                PipelineStageState::Waiting
            },
        };
        let train = self.training_stage(imported);
        let render = PipelineStage {
            name: "渲染检查",
            detail: if self.has_rendered_gaussians {
                self.gaussian_count
                    .map(|count| format!("{} Gaussians", count))
                    .unwrap_or_else(|| "Gaussian preview ready".to_string())
            } else {
                "等待训练结果".to_string()
            },
            state: if self.has_rendered_gaussians {
                PipelineStageState::Completed
            } else {
                PipelineStageState::Waiting
            },
        };

        [import, pose, train, render]
    }

    fn project_stages(&self, project: &ProjectSessionSummary) -> [PipelineStage; 4] {
        let import = PipelineStage {
            name: "导入采集",
            detail: project
                .imported_frame_count
                .map(|count| format!("{count} 张已验证"))
                .unwrap_or_else(|| match project.source_kind {
                    crate::project::SourceKind::ImageSequence => "等待导入图像".to_owned(),
                    crate::project::SourceKind::Video => "等待导入视频".to_owned(),
                }),
            state: stage_presentation_state(project.import_state),
        };
        let pose = PipelineStage {
            name: "位姿解算",
            detail: project.pose_detail.clone(),
            state: stage_presentation_state(project.pose_state),
        };
        let train = if matches!(
            self.training_state,
            TrainingSessionState::Loading
                | TrainingSessionState::Starting
                | TrainingSessionState::Training
                | TrainingSessionState::Stopping
                | TrainingSessionState::Completed
                | TrainingSessionState::Failed
                | TrainingSessionState::Cancelled
        ) {
            self.training_stage(
                self.has_dataset && project.pose_state == ProjectStagePresentation::Completed,
            )
        } else {
            PipelineStage {
                name: "高斯训练",
                detail: project.training_detail.clone(),
                state: stage_presentation_state(project.training_state),
            }
        };
        let render = PipelineStage {
            name: "渲染检查",
            detail: if self.has_rendered_gaussians {
                self.gaussian_count
                    .map(|count| format!("{count} Gaussians"))
                    .unwrap_or_else(|| "高斯预览就绪".to_owned())
            } else {
                "等待训练结果".to_owned()
            },
            state: if self.has_rendered_gaussians {
                PipelineStageState::Completed
            } else {
                PipelineStageState::Waiting
            },
        };

        [import, pose, train, render]
    }

    fn training_stage(&self, imported: bool) -> PipelineStage {
        let (detail, state) = match self.training_state {
            TrainingSessionState::Starting | TrainingSessionState::Loading => {
                ("Preparing RustGS".to_string(), PipelineStageState::Running)
            }
            TrainingSessionState::Training => (
                self.gaussian_count
                    .map(|count| format!("{} Gaussians", count))
                    .unwrap_or_else(|| "RustGS training".to_string()),
                PipelineStageState::Running,
            ),
            TrainingSessionState::Completed => (
                self.gaussian_count
                    .map(|count| format!("{} Gaussians generated", count))
                    .unwrap_or_else(|| "RustGS completed".to_string()),
                PipelineStageState::Completed,
            ),
            TrainingSessionState::Failed => (
                self.error
                    .clone()
                    .unwrap_or_else(|| "RustGS training failed".to_string()),
                PipelineStageState::Failed,
            ),
            TrainingSessionState::Cancelled => {
                ("Training cancelled".to_string(), PipelineStageState::Failed)
            }
            TrainingSessionState::Idle if imported => {
                ("Ready to train".to_string(), PipelineStageState::Ready)
            }
            TrainingSessionState::Idle => {
                ("Waiting for poses".to_string(), PipelineStageState::Waiting)
            }
            TrainingSessionState::Stopping => {
                ("Stopping RustGS".to_string(), PipelineStageState::Running)
            }
        };

        PipelineStage {
            name: "高斯训练",
            detail,
            state,
        }
    }
}

pub fn draw_command_bar(ui: &mut Ui, snapshot: &WorkbenchSnapshot) {
    ui.horizontal(|ui| {
        ui.add_space(14.0);
        ui.label(
            RichText::new("RustViewer")
                .strong()
                .size(14.0)
                .color(TEXT_PRIMARY),
        );
        ui.label(RichText::new("⌄").size(13.0).color(TEXT_SECONDARY));
        ui.separator();
        let project_name = if snapshot.project.is_some() {
            "当前项目"
        } else {
            "未命名项目"
        };
        ui.label(
            RichText::new(project_name)
                .strong()
                .size(13.0)
                .color(TEXT_PRIMARY),
        );
        ui.label(RichText::new("/").size(12.0).color(TEXT_DISABLED));
        let source = if snapshot.project.is_some() {
            "图像序列"
        } else if snapshot.has_dataset {
            "COLMAP 数据集"
        } else {
            "未载入素材"
        };
        ui.label(RichText::new(source).size(12.0).color(TEXT_SECONDARY));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_space(10.0);
            ui.label(RichText::new("?").size(13.0).color(TEXT_SECONDARY));
            ui.add_space(18.0);
            ui.label(RichText::new("GPU: Metal").size(11.0).color(TEXT_SECONDARY));
            ui.colored_label(SYSTEM_GREEN, "●");
        });
    });
}

pub fn draw_stage_rail(ui: &mut Ui, snapshot: &WorkbenchSnapshot) -> Vec<PanelAction> {
    let mut actions = Vec::new();
    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing = Vec2::new(0.0, 9.0);
        ui.label(
            RichText::new("CAPTURE SOURCE")
                .size(10.0)
                .strong()
                .color(SYSTEM_GRAY),
        );
        ui.horizontal(|ui| {
            draw_image_import_menu(ui, RichText::new("图像").size(12.0), &mut actions);
            ui.add_enabled(
                false,
                egui::Button::new(RichText::new("视频").size(12.0)).min_size(Vec2::new(76.0, 30.0)),
            );
        });
        ui.add_space(10.0);
        ui.label(
            RichText::new("PIPELINE")
                .size(10.0)
                .strong()
                .color(SYSTEM_GRAY),
        );
        for (index, stage) in snapshot.stages().iter().enumerate() {
            draw_stage(ui, stage, index == 0);
        }
        ui.add_space(14.0);
        let command = snapshot.primary_command();
        if command.action == PrimaryAction::ImportImages {
            ui.add_enabled_ui(command.enabled, |ui| {
                draw_image_import_menu(ui, RichText::new(command.label).strong(), &mut actions);
            });
        } else if ui
            .add_enabled(
                command.enabled,
                egui::Button::new(RichText::new(command.label).strong())
                    .min_size(Vec2::new(ui.available_width(), 36.0)),
            )
            .clicked()
        {
            actions.push(match command.action {
                PrimaryAction::ImportImages => unreachable!("import actions use the import menu"),
                PrimaryAction::OpenColmap => PanelAction::OpenColmap,
                PrimaryAction::SolvePoses => PanelAction::SolvePoses,
                PrimaryAction::StartTraining => PanelAction::StartTraining,
                PrimaryAction::CancelTraining => PanelAction::StopTraining,
            });
        }
        ui.add_enabled(
            false,
            egui::Button::new(RichText::new("暂停").strong())
                .min_size(Vec2::new(ui.available_width(), 34.0)),
        );
    });
    actions
}

fn draw_image_import_menu(ui: &mut Ui, label: RichText, actions: &mut Vec<PanelAction>) {
    ui.menu_button(label, |ui| {
        ui.set_min_width(176.0);
        if ui.button("批量选择图片...").clicked() {
            actions.push(panel_action_for_primary_import(SourceSelection::Images));
            ui.close();
        }
        if ui.button("选择图片文件夹...").clicked() {
            actions.push(PanelAction::ImportImageFolder);
            ui.close();
        }
        ui.separator();
        if ui.button("打开 RustScan 项目...").clicked() {
            actions.push(PanelAction::OpenProject);
            ui.close();
        }
        if ui.button("打开 COLMAP 工作区...").clicked() {
            actions.push(PanelAction::OpenColmap);
            ui.close();
        }
    });
}

pub fn draw_inspector(
    ui: &mut Ui,
    state: &mut UiState,
    scene: &mut Scene,
    snapshot: &WorkbenchSnapshot,
) -> Vec<PanelAction> {
    let mut actions = Vec::new();
    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing = Vec2::new(0.0, 10.0);
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("Scene Inspector")
                    .size(14.0)
                    .strong()
                    .color(TEXT_PRIMARY),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new("↻").size(15.0).color(TEXT_SECONDARY));
            });
        });
        draw_divider(ui);
        draw_section_label(ui, "CAPTURE");
        ui.horizontal(|ui| {
            ui.colored_label(SYSTEM_BLUE, "▣");
            ui.label(
                RichText::new(format!("{} 张已选图像", snapshot.visible_frame_count()))
                    .size(13.0)
                    .strong(),
            );
        });
        metric_row(ui, "格式", "JPEG sequence".to_owned());
        if let Some(summary) = &state.dataset_summary {
            metric_row(
                ui,
                "分辨率",
                format!("{} × {}", summary.width, summary.height),
            );
        } else {
            metric_row(ui, "分辨率", "—".to_string());
        }
        metric_row(ui, "相机模型", "PINHOLE".to_owned());
        draw_divider(ui);
        draw_section_label(ui, "RECONSTRUCTION");
        draw_reconstruction_metrics(ui, snapshot);
        draw_divider(ui);
        draw_section_label(ui, "TRAINING CONTROLS");
        numeric_setting(
            ui,
            "训练迭代",
            &mut state.training_controls.iterations,
            10..=30_000,
        );
        scalar_setting(
            ui,
            "渲染缩放",
            &mut state.training_controls.render_scale,
            0.125..=1.0,
        );
        ui.checkbox(&mut scene.layers.gaussians, "实时快照");
        ui.checkbox(&mut scene.layers.trajectory, "相机轨迹");
        draw_divider(ui);
        draw_section_label(ui, "OUTPUT");
        metric_row(ui, "目标", "scene.splat".to_owned());
        metric_row(ui, "状态", training_label(state.training_state).to_owned());
        if ui.button("适配场景").clicked() && scene.has_data() {
            actions.push(PanelAction::AutoFitScene);
        }
        if let Some(error) = snapshot.error.as_deref() {
            draw_divider(ui);
            ui.label(RichText::new(error).size(11.0).color(SYSTEM_RED));
        }
    });
    actions
}

pub fn draw_activity_strip(ui: &mut Ui, snapshot: &WorkbenchSnapshot) {
    let activity_width = ui.available_width() * 0.76;
    ui.horizontal_top(|ui| {
        ui.allocate_ui_with_layout(
            Vec2::new(activity_width, ui.available_height()),
            egui::Layout::top_down(egui::Align::LEFT),
            |ui| {
                ui.label(
                    RichText::new("Activity")
                        .size(12.0)
                        .strong()
                        .color(TEXT_PRIMARY),
                );
                if snapshot.activity_feed().is_empty() {
                    ui.label(
                        RichText::new("等待导入图像序列")
                            .size(11.0)
                            .color(TEXT_DISABLED),
                    );
                } else {
                    for entry in snapshot.activity_feed().iter().rev().take(2) {
                        ui.horizontal(|ui| {
                            ui.colored_label(stage_color(entry.state), "●");
                            ui.label(
                                RichText::new(format!("{}: {}", entry.title, entry.detail))
                                    .size(11.0)
                                    .color(TEXT_SECONDARY),
                            );
                        });
                    }
                }
            },
        );
        ui.separator();
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Frame sequence").size(12.0).strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        RichText::new(format!("0 / {}", snapshot.visible_frame_count()))
                            .size(11.0)
                            .color(SYSTEM_GREEN),
                    );
                });
            });
            draw_frame_sequence(ui, snapshot.visible_frame_count());
            ui.label(
                RichText::new("■ 已处理   ◆ 关键帧")
                    .size(10.0)
                    .color(TEXT_SECONDARY),
            );
        });
    });
}

pub fn draw_viewport_toolbar(ui: &mut Ui, snapshot: &WorkbenchSnapshot) -> Vec<PanelAction> {
    let mut actions = Vec::new();
    ui.horizontal(|ui| {
        ui.add_space(12.0);
        ui.label(RichText::new("Scene viewport").size(12.0).strong());
        ui.separator();
        ui.add_enabled(false, egui::Button::new(RichText::new("Orbit").size(11.0)));
        if ui.button(RichText::new("适配").size(11.0)).clicked() {
            actions.push(PanelAction::AutoFitScene);
        }
        ui.add_enabled(false, egui::Button::new(RichText::new("测量").size(11.0)));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let resolution = if snapshot.has_dataset {
                "已加载"
            } else {
                "等待重建"
            };
            ui.label(RichText::new(resolution).size(11.0).color(TEXT_SECONDARY));
        });
    });
    actions
}

pub fn draw_viewport_empty_state(ui: &mut Ui, snapshot: &WorkbenchSnapshot) {
    let rect = ui.max_rect();
    draw_viewport_grid(ui, rect);
    let heading = if snapshot.project.is_some() {
        "等待重建"
    } else {
        "等待导入图像"
    };
    let detail = if snapshot.project.is_some() {
        format!(
            "{} 张图像已验证，可运行重建",
            snapshot.visible_frame_count()
        )
    } else {
        "从左侧选择图像序列开始".to_owned()
    };
    let card = Rect::from_min_size(
        Pos2::new(rect.right() - 230.0, rect.top() + 16.0),
        Vec2::new(214.0, 102.0),
    );
    ui.painter().rect_filled(card, 6.0, PANEL_BG);
    ui.painter().rect_stroke(
        card,
        6.0,
        Stroke::new(1.0_f32, SEPARATOR),
        egui::StrokeKind::Inside,
    );
    ui.scope_builder(egui::UiBuilder::new().max_rect(card.shrink(12.0)), |ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new(heading).size(13.0).strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new("导入").size(11.0).color(TEXT_SECONDARY));
            });
        });
        ui.separator();
        ui.label(RichText::new(detail).size(11.0).color(TEXT_SECONDARY));
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("Pose").size(10.0).color(TEXT_DISABLED));
            ui.add_space(62.0);
            ui.label(RichText::new("PSNR").size(10.0).color(TEXT_DISABLED));
        });
    });
    let chip = Rect::from_min_size(
        Pos2::new(rect.left() + 16.0, rect.bottom() - 36.0),
        Vec2::new(222.0, 28.0),
    );
    ui.painter().rect_filled(chip, 5.0, PANEL_BG);
    ui.painter().rect_stroke(
        chip,
        5.0,
        Stroke::new(1.0_f32, SEPARATOR),
        egui::StrokeKind::Inside,
    );
    ui.painter().text(
        chip.center(),
        egui::Align2::CENTER_CENTER,
        "Active view   Orbit   ·   无选中对象",
        egui::FontId::proportional(10.0),
        TEXT_SECONDARY,
    );
}

pub fn draw_viewport_grid(ui: &Ui, rect: Rect) {
    let spacing = 44.0;
    let stroke = Stroke::new(1.0_f32, Color32::from_rgb(29, 39, 41));
    let mut x = rect.left();
    while x <= rect.right() {
        ui.painter().line_segment(
            [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
            stroke,
        );
        x += spacing;
    }
    let mut y = rect.top();
    while y <= rect.bottom() {
        ui.painter().line_segment(
            [Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)],
            stroke,
        );
        y += spacing;
    }
}

fn draw_stage(ui: &mut Ui, stage: &PipelineStage, selected: bool) {
    let color = stage_color(stage.state);
    let frame = egui::Frame::new()
        .inner_margin(egui::Margin::symmetric(8, 7))
        .corner_radius(6.0);
    let frame = if selected {
        frame.fill(CARD_BG).stroke(Stroke::new(1.0_f32, SEPARATOR))
    } else {
        frame
    };
    frame.show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.colored_label(color, "●");
            ui.vertical(|ui| {
                ui.label(
                    RichText::new(stage.name)
                        .size(12.0)
                        .strong()
                        .color(TEXT_PRIMARY),
                );
                ui.label(
                    RichText::new(&stage.detail)
                        .size(10.0)
                        .color(TEXT_SECONDARY),
                );
            });
        });
    });
}

fn draw_reconstruction_metrics(ui: &mut Ui, snapshot: &WorkbenchSnapshot) {
    egui::Grid::new("reconstruction_metrics")
        .num_columns(2)
        .spacing(Vec2::new(14.0, 8.0))
        .show(ui, |ui| {
            metric_cell(
                ui,
                "已注册相机",
                if snapshot.has_dataset {
                    snapshot.visible_frame_count().to_string()
                } else {
                    "—".to_owned()
                },
            );
            metric_cell(
                ui,
                "稀疏点",
                if snapshot.sparse_point_count > 0 {
                    snapshot.sparse_point_count.to_string()
                } else {
                    "—".to_owned()
                },
            );
            ui.end_row();
            metric_cell(
                ui,
                "高斯基元",
                snapshot
                    .gaussian_count
                    .map(|count| count.to_string())
                    .unwrap_or_else(|| "—".to_owned()),
            );
            metric_cell(ui, "渲染 PSNR", "—".to_owned());
        });
}

fn metric_cell(ui: &mut Ui, label: &str, value: String) {
    ui.vertical(|ui| {
        ui.label(RichText::new(label).size(10.0).color(TEXT_DISABLED));
        ui.label(RichText::new(value).size(13.0).strong().color(TEXT_PRIMARY));
    });
}

fn numeric_setting(
    ui: &mut Ui,
    label: &str,
    value: &mut usize,
    range: std::ops::RangeInclusive<usize>,
) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).size(11.0).color(TEXT_SECONDARY));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add(
                egui::DragValue::new(value)
                    .range(range)
                    .speed(100.0)
                    .min_decimals(0),
            );
        });
    });
}

fn scalar_setting(ui: &mut Ui, label: &str, value: &mut f32, range: std::ops::RangeInclusive<f32>) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).size(11.0).color(TEXT_SECONDARY));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add(
                egui::DragValue::new(value)
                    .range(range)
                    .speed(0.05)
                    .max_decimals(2),
            );
        });
    });
}

fn draw_frame_sequence(ui: &mut Ui, count: usize) {
    let visible = count.min(12).max(1);
    let available = ui.available_width();
    let cell_width =
        ((available - (visible.saturating_sub(1) as f32 * 4.0)) / visible as f32).clamp(11.0, 18.0);
    let (rect, _) = ui.allocate_exact_size(Vec2::new(available, 34.0), egui::Sense::hover());
    let painter = ui.painter();
    for index in 0..visible {
        let left = rect.left() + index as f32 * (cell_width + 4.0);
        let cell = Rect::from_min_size(Pos2::new(left, rect.top()), Vec2::new(cell_width, 30.0));
        painter.rect_stroke(
            cell,
            2.0,
            Stroke::new(1.0_f32, SEPARATOR),
            egui::StrokeKind::Inside,
        );
    }
}

fn draw_section_label(ui: &mut Ui, label: &str) {
    ui.label(RichText::new(label).size(10.0).strong().color(SYSTEM_GRAY));
}

fn draw_divider(ui: &mut Ui) {
    let width = ui.available_width();
    ui.add_space(2.0);
    ui.painter().hline(
        ui.min_rect().x_range(),
        ui.cursor().top(),
        egui::Stroke::new(1.0_f32, SEPARATOR),
    );
    ui.allocate_space(Vec2::new(width, 4.0));
}

fn metric_row(ui: &mut Ui, label: &str, value: String) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).size(11.0).color(TEXT_SECONDARY));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(RichText::new(value).size(11.0).strong().color(TEXT_PRIMARY));
        });
    });
}

fn stage_color(state: PipelineStageState) -> Color32 {
    match state {
        PipelineStageState::Ready => SYSTEM_BLUE,
        PipelineStageState::Waiting => TEXT_DISABLED,
        PipelineStageState::Running => SYSTEM_ORANGE,
        PipelineStageState::Completed => SYSTEM_GREEN,
        PipelineStageState::Failed => SYSTEM_RED,
    }
}

fn stage_presentation_state(state: ProjectStagePresentation) -> PipelineStageState {
    match state {
        ProjectStagePresentation::Ready => PipelineStageState::Ready,
        ProjectStagePresentation::Waiting => PipelineStageState::Waiting,
        ProjectStagePresentation::Running => PipelineStageState::Running,
        ProjectStagePresentation::Completed => PipelineStageState::Completed,
        ProjectStagePresentation::Failed => PipelineStageState::Failed,
    }
}

fn training_label(state: TrainingSessionState) -> &'static str {
    match state {
        TrainingSessionState::Idle => "Idle",
        TrainingSessionState::Loading => "Loading",
        TrainingSessionState::Starting => "Starting",
        TrainingSessionState::Training => "Training",
        TrainingSessionState::Stopping => "Stopping",
        TrainingSessionState::Completed => "Completed",
        TrainingSessionState::Failed => "Failed",
        TrainingSessionState::Cancelled => "Cancelled",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_workbench_only_enables_import() {
        let snapshot = WorkbenchSnapshot::default();
        let stages = snapshot.stages();

        assert_eq!(stages[0].state, PipelineStageState::Ready);
        assert_eq!(stages[1].state, PipelineStageState::Waiting);
        assert_eq!(stages[2].state, PipelineStageState::Waiting);
        assert_eq!(stages[3].state, PipelineStageState::Waiting);
    }

    #[test]
    fn primary_command_follows_real_dataset_and_training_state() {
        let empty = WorkbenchSnapshot::default();
        assert_eq!(empty.primary_command().label, "导入图像");
        assert_eq!(empty.primary_command().action, PrimaryAction::ImportImages);

        let loaded = WorkbenchSnapshot {
            has_dataset: true,
            ..Default::default()
        };
        assert_eq!(loaded.primary_command().label, "Start training");
        assert_eq!(
            loaded.primary_command().action,
            PrimaryAction::StartTraining
        );

        let training = WorkbenchSnapshot {
            has_dataset: true,
            training_state: TrainingSessionState::Training,
            ..Default::default()
        };
        assert_eq!(
            training.primary_command().action,
            PrimaryAction::CancelTraining
        );
    }

    #[test]
    fn imported_image_project_exposes_pose_solve_before_colmap_is_loaded() {
        let snapshot = WorkbenchSnapshot {
            project: Some(crate::project::ProjectSessionSummary::from_states(
                crate::project::SourceKind::ImageSequence,
                crate::project::StageState::Succeeded,
                crate::project::StageState::Ready,
                crate::project::StageState::NotStarted,
                crate::project::StageState::NotStarted,
            )),
            ..Default::default()
        };

        assert_eq!(snapshot.primary_command().label, "运行重建");
        assert_eq!(snapshot.primary_command().action, PrimaryAction::SolvePoses);
        assert!(snapshot.primary_command().enabled);
    }

    #[test]
    fn imported_image_project_labels_the_primary_action_as_run_reconstruction() {
        let snapshot = WorkbenchSnapshot {
            project: Some(crate::project::ProjectSessionSummary::from_states(
                crate::project::SourceKind::ImageSequence,
                crate::project::StageState::Succeeded,
                crate::project::StageState::Ready,
                crate::project::StageState::NotStarted,
                crate::project::StageState::NotStarted,
            )),
            ..Default::default()
        };

        assert_eq!(snapshot.primary_command().label, "运行重建");
    }

    #[test]
    fn workbench_layout_keeps_the_viewport_dominant_at_minimum_window_size() {
        let layout = WorkbenchLayout::for_window(egui::vec2(1280.0, 800.0));

        assert_eq!(layout.top_bar_height, 52.0);
        assert_eq!(layout.viewport_toolbar_height, 48.0);
        assert_eq!(layout.stage_rail_width, 184.0);
        assert_eq!(layout.inspector_width, 288.0);
        assert_eq!(layout.activity_height, 128.0);
        assert!(layout.viewport_width() >= 800.0);
    }

    #[test]
    fn imported_project_surfaces_its_committed_frame_count_before_colmap_loads() {
        let snapshot = WorkbenchSnapshot {
            project: Some(crate::project::ProjectSessionSummary {
                source_kind: crate::project::SourceKind::ImageSequence,
                imported_frame_count: Some(87),
                import_state: crate::project::ProjectStagePresentation::Completed,
                pose_state: crate::project::ProjectStagePresentation::Ready,
                pose_detail: "Ready to start".to_owned(),
                training_state: crate::project::ProjectStagePresentation::Waiting,
                training_detail: "Waiting for pose coverage".to_owned(),
            }),
            ..Default::default()
        };

        assert_eq!(snapshot.visible_frame_count(), 87);
    }

    #[test]
    fn project_stage_summary_overrides_legacy_colmap_training_readiness() {
        let snapshot = WorkbenchSnapshot {
            has_dataset: true,
            project: Some(crate::project::ProjectSessionSummary::from_states(
                crate::project::SourceKind::Video,
                crate::project::StageState::Succeeded,
                crate::project::StageState::Succeeded,
                crate::project::StageState::Ready,
                crate::project::StageState::NotStarted,
            )),
            ..Default::default()
        };

        let stages = snapshot.stages();
        assert_eq!(stages[2].state, PipelineStageState::Waiting);
        assert_eq!(stages[2].detail, "Waiting for full-frame poses");
        assert!(!snapshot.primary_command().enabled);
    }

    #[test]
    fn project_stage_summary_shows_live_rustgs_training_state() {
        let snapshot = WorkbenchSnapshot {
            has_dataset: true,
            project: Some(crate::project::ProjectSessionSummary::from_states(
                crate::project::SourceKind::ImageSequence,
                crate::project::StageState::Succeeded,
                crate::project::StageState::Succeeded,
                crate::project::StageState::Succeeded,
                crate::project::StageState::Ready,
            )),
            training_state: TrainingSessionState::Training,
            gaussian_count: Some(64),
            ..Default::default()
        };

        let stages = snapshot.stages();

        assert_eq!(stages[2].state, PipelineStageState::Running);
        assert_eq!(stages[2].detail, "64 Gaussians");
    }

    #[test]
    fn activity_feed_uses_recorded_pipeline_events() {
        let snapshot = WorkbenchSnapshot {
            activity: vec![WorkbenchActivity::new(
                "Pose solve",
                "Feature matching: 12 poses",
                PipelineStageState::Running,
            )],
            ..Default::default()
        };

        assert_eq!(
            snapshot.activity_feed(),
            [WorkbenchActivity::new(
                "Pose solve",
                "Feature matching: 12 poses",
                PipelineStageState::Running,
            )]
        );
    }
}
