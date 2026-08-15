//! Viewport overlay and empty-state UI.

use crate::renderer::camera::ArcballCamera;
use crate::ui::theme::*;
use egui::Vec2;

struct EmptyStateCopy {
    heading: &'static str,
    detail: &'static str,
}

fn empty_state_copy() -> EmptyStateCopy {
    EmptyStateCopy {
        heading: "No reconstruction loaded",
        detail: "Open a project or load a COLMAP workspace to begin.",
    }
}

/// Draw an empty-state overlay when no scene data is loaded.
pub fn draw_empty_state(ui: &mut egui::Ui) {
    let copy = empty_state_copy();
    ui.centered_and_justified(|ui| {
        ui.vertical_centered(|ui| {
            ui.label(
                egui::RichText::new(copy.heading)
                    .size(18.0)
                    .strong()
                    .color(TEXT_PRIMARY),
            );

            ui.add_space(10.0);

            ui.label(
                egui::RichText::new(copy.detail)
                    .size(12.0)
                    .color(TEXT_SECONDARY),
            );
        });
    });
}

/// Draw axis indicator and camera info overlay at the bottom-right of the viewport.
pub fn draw_viewport_overlay(ui: &mut egui::Ui, camera: &ArcballCamera, has_data: bool) {
    let rect = ui.max_rect();
    let indicator_size = 80.0;
    let margin = 20.0;
    let pos = egui::Pos2::new(
        rect.right() - indicator_size - margin,
        rect.bottom() - indicator_size - margin,
    );

    let indicator_rect = egui::Rect::from_min_size(pos, Vec2::new(indicator_size, indicator_size));

    // Background
    ui.painter().rect_filled(indicator_rect, 8.0, overlay_bg());

    // Draw axis labels
    let center = indicator_rect.center();

    // X axis (red)
    ui.painter().text(
        center + Vec2::new(-30.0, 5.0),
        egui::Align2::CENTER_CENTER,
        "X",
        egui::FontId::proportional(12.0),
        SYSTEM_RED,
    );

    // Y axis (green)
    ui.painter().text(
        center + Vec2::new(5.0, -25.0),
        egui::Align2::CENTER_CENTER,
        "Y",
        egui::FontId::proportional(12.0),
        SYSTEM_GREEN,
    );

    // Z axis (blue)
    ui.painter().text(
        center + Vec2::new(5.0, 30.0),
        egui::Align2::CENTER_CENTER,
        "Z",
        egui::FontId::proportional(12.0),
        SYSTEM_BLUE,
    );

    // Camera info (only when data is loaded)
    if has_data {
        let info_pos = indicator_rect.min + Vec2::new(6.0, 50.0);
        let (yaw, pitch, roll) = camera.display_angles();
        ui.painter().text(
            info_pos,
            egui::Align2::LEFT_TOP,
            format!(
                "yaw: {:.1}°\npitch: {:.1}°\nroll: {:.1}°\ndist: {:.2}",
                yaw.to_degrees(),
                pitch.to_degrees(),
                roll.to_degrees(),
                camera.distance
            ),
            egui::FontId::proportional(10.0),
            TEXT_PRIMARY,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_state_copy_guides_users_to_a_project_or_colmap_workspace() {
        let copy = empty_state_copy();

        assert_eq!(copy.heading, "No reconstruction loaded");
        assert_eq!(
            copy.detail,
            "Open a project or load a COLMAP workspace to begin."
        );
    }
}
