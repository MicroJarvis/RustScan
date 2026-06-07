//! Dark theme configuration for RustViewer.

use egui::{Color32, FontId, Stroke, Vec2};

// ── Accent Colors ────────────────────────────────────────────────────────────

pub const SYSTEM_BLUE: Color32 = Color32::from_rgb(0, 122, 255);
pub const SYSTEM_GREEN: Color32 = Color32::from_rgb(52, 199, 89);
pub const SYSTEM_ORANGE: Color32 = Color32::from_rgb(255, 149, 0);
pub const SYSTEM_RED: Color32 = Color32::from_rgb(255, 59, 48);
pub const SYSTEM_GRAY: Color32 = Color32::from_rgb(142, 142, 147);

// ── Background Colors ────────────────────────────────────────────────────────

pub const WINDOW_BG: Color32 = Color32::from_rgb(10, 12, 16);
pub const PANEL_BG: Color32 = Color32::from_rgb(15, 18, 24);
pub const CARD_BG: Color32 = Color32::from_rgb(22, 26, 34);
pub const VIEWPORT_BG: Color32 = Color32::from_rgb(0, 0, 0);
pub const SEPARATOR: Color32 = Color32::from_rgb(42, 48, 60);

// ── Text Colors ──────────────────────────────────────────────────────────────

pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(242, 245, 250);
pub const TEXT_SECONDARY: Color32 = Color32::from_rgb(160, 168, 184);
pub const TEXT_DISABLED: Color32 = Color32::from_rgb(108, 116, 132);

// ── 3D Scene Colors ─────────────────────────────────────────────────────────

pub const COLOR_TRAJECTORY: Color32 = Color32::from_rgb(0, 122, 255);
pub const COLOR_MAP_POINTS: Color32 = Color32::from_rgb(52, 199, 89);
pub const COLOR_GAUSSIANS: Color32 = Color32::from_rgb(255, 149, 0);
pub const COLOR_MESH: Color32 = Color32::from_rgb(142, 142, 147);
pub const COLOR_MESH_SOLID: Color32 = Color32::from_rgb(140, 140, 200);

// ── Spacing (8pt grid) ──────────────────────────────────────────────────────

pub const SP_XS: f32 = 4.0;
pub const SP_SM: f32 = 8.0;
pub const SP_MD: f32 = 12.0;
pub const SP_LG: f32 = 16.0;
pub const SP_XL: f32 = 24.0;
pub const SP_XXL: f32 = 32.0;

// ── Typography ──────────────────────────────────────────────────────────────

pub fn font_heading() -> FontId {
    FontId::proportional(20.0)
}

pub fn font_title() -> FontId {
    FontId::proportional(17.0)
}

pub fn font_body() -> FontId {
    FontId::proportional(13.0)
}

pub fn font_small() -> FontId {
    FontId::proportional(11.0)
}

pub fn font_caption() -> FontId {
    FontId::proportional(10.0)
}

pub fn font_mono() -> FontId {
    FontId::monospace(13.0)
}

pub fn font_mono_small() -> FontId {
    FontId::monospace(11.0)
}

pub fn overlay_bg() -> Color32 {
    Color32::from_rgba_unmultiplied(18, 22, 30, 230)
}

pub fn hover_bg() -> Color32 {
    Color32::from_rgba_unmultiplied(255, 255, 255, 18)
}

// ── Corner Radius ───────────────────────────────────────────────────────────

pub const RADIUS_CARD: u8 = 8;
pub const RADIUS_BUTTON: u8 = 6;
pub const RADIUS_BADGE: u8 = 4;

// ── Apply Theme ──────────────────────────────────────────────────────────────

/// Configure the egui context to use a dark high-contrast styling.
/// Call this once at the start of each frame in `App::update()`.
pub fn configure_theme(ctx: &egui::Context) {
    let mut style = (*ctx.global_style()).clone();

    // Font sizes
    style.text_styles = [
        (egui::TextStyle::Heading, FontId::proportional(20.0)),
        (egui::TextStyle::Body, FontId::proportional(13.0)),
        (egui::TextStyle::Button, FontId::proportional(13.0)),
        (egui::TextStyle::Small, FontId::proportional(11.0)),
        (egui::TextStyle::Monospace, FontId::monospace(13.0)),
    ]
    .into();

    // Spacing (8pt grid)
    style.spacing.item_spacing = Vec2::new(8.0, 8.0);
    style.spacing.button_padding = Vec2::new(12.0, 6.0);
    style.spacing.indent = 16.0;

    // Visuals
    let mut visuals = egui::Visuals::dark();

    // Window background
    visuals.window_fill = WINDOW_BG;
    visuals.panel_fill = PANEL_BG;
    visuals.extreme_bg_color = WINDOW_BG;
    visuals.faint_bg_color = CARD_BG;
    visuals.override_text_color = Some(TEXT_PRIMARY);

    // Widget colors — inactive (normal)
    visuals.widgets.inactive.weak_bg_fill = SYSTEM_BLUE;
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    visuals.widgets.inactive.bg_fill = CARD_BG;
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, SEPARATOR);

    // Widget colors — hovered
    visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(0, 110, 230);
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    visuals.widgets.hovered.bg_fill = hover_bg();
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, Color32::from_rgb(76, 86, 102));

    // Widget colors — active (pressed)
    visuals.widgets.active.weak_bg_fill = Color32::from_rgb(0, 85, 200);
    visuals.widgets.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    visuals.widgets.active.bg_fill = hover_bg();
    visuals.widgets.active.bg_stroke = Stroke::new(1.0, SYSTEM_BLUE);

    // Separator
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(0.5, SEPARATOR);
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT_SECONDARY);
    visuals.selection.bg_fill = Color32::from_rgba_unmultiplied(0, 122, 255, 80);
    visuals.selection.stroke = Stroke::new(1.0, SYSTEM_BLUE);

    style.visuals = visuals;
    ctx.set_global_style(style);
}
