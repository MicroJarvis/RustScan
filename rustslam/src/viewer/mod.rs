//! MapViewer - 3D Map Visualization Module
//!
//! This module provides visualization capabilities for SLAM maps,
//! including drawing camera trajectories, map points, keyframes, and current frame.
//!
//! # Features
//! - 3D to 2D orthographic projection for map visualization
//! - Draw camera trajectories with view cones
//! - Draw 3D map points with depth-based coloring
//! - Draw keyframes with camera representations
//! - Draw current frame pose
//! - Save visualization as PNG images
//!
//! # Usage
//!
//! ```ignore
//! use rustslam::{Map, MapViewer};
//!
//! let mut viewer = MapViewer::new(800, 600);
//! viewer.draw_map_points(map.points()).unwrap();
//! viewer.draw_keyframes(map.keyframes()).unwrap();
//! viewer.draw_cameras(map.keyframes()).unwrap();
//! viewer.save_image("map.png").unwrap();
//! ```

use crate::core::{Frame, KeyFrame, MapPoint};

/// RGBA color type
pub type Color = [u8; 4];

/// 3D Point for visualization
#[derive(Debug, Clone, Copy)]
pub struct Point3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Point3D {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn from_array(arr: &[f32; 3]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
            z: arr[2],
        }
    }
}

/// MapViewer provides visualization for SLAM maps
///
/// Uses orthographic projection to render 3D map data as 2D images.
/// The view is from above (top-down view) by default.
#[derive(Debug)]
pub struct MapViewer {
    /// Image width
    width: u32,
    /// Image height
    height: u32,
    /// Image buffer (RGBA)
    buffer: Vec<u8>,
    /// Background color
    background_color: Color,
    /// Point size for drawing
    point_size: u32,
    /// Camera scale factor
    camera_scale: f32,
    /// View center X
    center_x: f32,
    /// View center Y
    center_y: f32,
    /// View scale (zoom)
    scale: f32,
}

impl MapViewer {
    /// Create a new MapViewer with default settings
    ///
    /// # Arguments
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            buffer: vec![0u8; (width * height * 4) as usize],
            background_color: [20, 20, 20, 255], // Dark gray background
            point_size: 3,
            camera_scale: 0.5,
            center_x: 0.0,
            center_y: 0.0,
            scale: 100.0, // 100 pixels per unit
        }
    }

    /// Create a MapViewer with custom settings
    ///
    /// # Arguments
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `point_size` - Size of points to draw
    pub fn with_settings(width: u32, height: u32, point_size: u32) -> Self {
        let mut viewer = Self::new(width, height);
        viewer.point_size = point_size;
        viewer
    }

    /// Create a MapViewer with full custom settings
    ///
    /// # Arguments
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `point_size` - Size of points
    /// * `camera_scale` - Scale factor for camera visualization
    pub fn with_settings_full(width: u32, height: u32, point_size: u32, camera_scale: f32) -> Self {
        let mut viewer = Self::new(width, height);
        viewer.point_size = point_size;
        viewer.camera_scale = camera_scale;
        viewer
    }

    /// Get the image width
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the image height
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get the background color
    pub fn background_color(&self) -> Color {
        self.background_color
    }

    /// Set the background color
    pub fn set_background_color(&mut self, color: Color) {
        self.background_color = color;
    }

    /// Get the point size
    pub fn point_size(&self) -> u32 {
        self.point_size
    }

    /// Set the point size
    pub fn set_point_size(&mut self, size: u32) {
        self.point_size = size;
    }

    /// Get the camera scale
    pub fn camera_scale(&self) -> f32 {
        self.camera_scale
    }

    /// Set the camera scale
    pub fn set_camera_scale(&mut self, scale: f32) {
        self.camera_scale = scale;
    }

    /// Get the image buffer (RGBA)
    pub fn get_image_buffer(&self) -> &[u8] {
        &self.buffer
    }

    /// Get mutable image buffer
    pub fn get_image_buffer_mut(&mut self) -> &mut [u8] {
        &mut self.buffer
    }

    /// Clear the image buffer with background color
    pub fn clear(&mut self) {
        let bg = self.background_color;
        for i in (0..self.buffer.len()).step_by(4) {
            self.buffer[i] = bg[0]; // R
            self.buffer[i + 1] = bg[1]; // G
            self.buffer[i + 2] = bg[2]; // B
            self.buffer[i + 3] = bg[3]; // A
        }
    }

    /// Project a 3D point to 2D viewport coordinates
    ///
    /// Uses orthographic projection (top-down view)
    /// X axis -> viewport X, Z axis -> viewport Y (inverted)
    pub fn project_to_viewport(&self, point: &[f32; 3]) -> [f32; 2] {
        let x = (point[0] - self.center_x) * self.scale + self.width as f32 / 2.0;
        // Invert Y so positive Z goes up in the image
        let y = -(point[2] - self.center_y) * self.scale + self.height as f32 / 2.0;
        [x, y]
    }

    /// Get depth-based color for map points
    ///
    /// Near points are green, far points are red
    fn get_depth_color(&self, z: f32, min_z: f32, max_z: f32) -> Color {
        let range = max_z - min_z;
        if range < 1e-6 {
            return [0, 255, 0, 255]; // Green for unknown depth
        }

        let t = ((z - min_z) / range).clamp(0.0, 1.0);

        // Green to Red gradient
        let r = (t * 255.0) as u8;
        let g = ((1.0 - t) * 255.0) as u8;
        let b = 50u8;

        [r, g, b, 255]
    }

    /// Draw a filled circle at (x, y) with given radius and color
    fn draw_circle(&mut self, cx: i32, cy: i32, radius: i32, color: Color) {
        let r2 = radius * radius;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= r2 {
                    let px = cx + dx;
                    let py = cy + dy;
                    if px >= 0 && px < self.width as i32 && py >= 0 && py < self.height as i32 {
                        let idx = ((py as u32) * self.width + (px as u32)) as usize * 4;
                        self.buffer[idx] = color[0];
                        self.buffer[idx + 1] = color[1];
                        self.buffer[idx + 2] = color[2];
                        self.buffer[idx + 3] = color[3];
                    }
                }
            }
        }
    }

    /// Draw a line between two points
    fn draw_line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: Color) {
        // Bresenham's line algorithm
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;

        let mut x = x0;
        let mut y = y0;

        loop {
            if x >= 0 && x < self.width as i32 && y >= 0 && y < self.height as i32 {
                let idx = ((y as usize) * (self.width as usize) + (x as usize)) * 4;
                self.buffer[idx] = color[0];
                self.buffer[idx + 1] = color[1];
                self.buffer[idx + 2] = color[2];
                self.buffer[idx + 3] = color[3];
            }

            if x == x1 && y == y1 {
                break;
            }

            let e2 = 2 * err;
            if e2 >= dy {
                if x == x1 {
                    break;
                }
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                if y == y1 {
                    break;
                }
                err += dx;
                y += sy;
            }
        }
    }

    /// Draw map points from an iterator
    ///
    /// # Arguments
    /// * `points` - Iterator of MapPoint references
    ///
    /// # Returns
    /// Ok(()) on success
    pub fn draw_map_points<'a, I>(&mut self, points: I) -> Result<(), String>
    where
        I: Iterator<Item = &'a MapPoint>,
    {
        // Collect points and compute depth range
        let points_data: Vec<_> = points
            .filter(|p| !p.is_outlier)
            .map(|p| (p.position.x, p.position.y, p.position.z))
            .collect();

        if points_data.is_empty() {
            return Ok(());
        }

        let min_z = points_data
            .iter()
            .map(|(_, _, z)| *z)
            .fold(f32::INFINITY, f32::min);
        let max_z = points_data
            .iter()
            .map(|(_, _, z)| *z)
            .fold(f32::NEG_INFINITY, f32::max);

        // Draw each point
        for (x, y, z) in points_data {
            let projected = self.project_to_viewport(&[x, y, z]);
            let px = projected[0] as i32;
            let py = projected[1] as i32;

            let color = self.get_depth_color(z, min_z, max_z);
            self.draw_circle(px, py, self.point_size as i32, color);
        }

        Ok(())
    }

    /// Draw keyframes as camera representations
    ///
    /// Each keyframe is drawn as a camera icon with its pose
    ///
    /// # Arguments
    /// * `keyframes` - Iterator of KeyFrame references
    ///
    /// # Returns
    /// Ok(()) on success
    pub fn draw_keyframes<'a, I>(&mut self, keyframes: I) -> Result<(), String>
    where
        I: Iterator<Item = &'a KeyFrame>,
    {
        let camera_size = (30.0 * self.camera_scale) as i32;

        for kf in keyframes {
            if let Some(pose) = kf.pose() {
                let t = pose.translation();
                let projected = self.project_to_viewport(&[t[0], t[1], t[2]]);
                let cx = projected[0] as i32;
                let cy = projected[1] as i32;

                // Get rotation to determine view direction
                let rot = pose.rotation_matrix();
                let fx = rot[0][2]; // Forward direction (Z axis of camera)
                let fz = rot[2][2];

                // Project view cone
                let view_length: f32 = camera_size as f32;
                let cone_angle: f32 = 0.5_f32; // Half angle in radians

                // Calculate view frustum points
                let (sin_angle, cos_angle) = (cone_angle.sin(), cone_angle.cos());

                // Left and right view rays
                let left_x = cx + ((fx * cos_angle - fz * sin_angle) * view_length) as i32;
                let left_y = cy + ((fz * cos_angle + fx * sin_angle) * view_length) as i32;
                let right_x = cx + ((fx * cos_angle + fz * sin_angle) * view_length) as i32;
                let right_y = cy + ((fz * cos_angle - fx * sin_angle) * view_length) as i32;

                // Draw camera triangle
                let cam_color: Color = [255, 200, 0, 255]; // Gold color
                self.draw_line(cx, cy, left_x, left_y, cam_color);
                self.draw_line(cx, cy, right_x, right_y, cam_color);
                self.draw_line(left_x, left_y, right_x, right_y, cam_color);
            }
        }

        Ok(())
    }

    /// Draw camera trajectory
    ///
    /// Draws lines connecting sequential camera poses
    ///
    /// # Arguments
    /// * `keyframes` - Iterator of KeyFrame references (ordered by ID)
    ///
    /// # Returns
    /// Ok(()) on success
    pub fn draw_cameras<'a, I>(&mut self, keyframes: I) -> Result<(), String>
    where
        I: Iterator<Item = &'a KeyFrame>,
    {
        // Collect keyframes sorted by ID
        let mut kf_list: Vec<_> = keyframes
            .filter(|kf| kf.pose().is_some())
            .map(|kf| (kf.id(), kf.pose().unwrap()))
            .collect();

        kf_list.sort_by_key(|(id, _)| *id);

        if kf_list.len() < 2 {
            return Ok(());
        }

        // Draw trajectory lines
        let traj_color: Color = [0, 150, 255, 255]; // Light blue

        for i in 0..kf_list.len() - 1 {
            let (_, pose1) = kf_list[i];
            let (_, pose2) = kf_list[i + 1];

            let t1 = pose1.translation();
            let t2 = pose2.translation();

            let p1 = self.project_to_viewport(&[t1[0], t1[1], t1[2]]);
            let p2 = self.project_to_viewport(&[t2[0], t2[1], t2[2]]);

            self.draw_line(
                p1[0] as i32,
                p1[1] as i32,
                p2[0] as i32,
                p2[1] as i32,
                traj_color,
            );
        }

        // Draw current camera (last one) with different color
        if let Some((_, pose)) = kf_list.last() {
            let t = pose.translation();
            let projected = self.project_to_viewport(&[t[0], t[1], t[2]]);
            let cx = projected[0] as i32;
            let cy = projected[1] as i32;

            // Draw current camera as larger circle
            let curr_color: Color = [255, 50, 50, 255]; // Red
            self.draw_circle(cx, cy, (self.point_size + 2) as i32, curr_color);
        }

        Ok(())
    }

    /// Draw the current frame pose
    ///
    /// Similar to draw_keyframes but for a single frame
    ///
    /// # Arguments
    /// * `frame` - Reference to the current Frame
    ///
    /// # Returns
    /// Ok(()) on success
    pub fn draw_current_frame(&mut self, frame: &Frame) -> Result<(), String> {
        if let Some(pose) = frame.pose {
            let t = pose.translation();
            let projected = self.project_to_viewport(&[t[0], t[1], t[2]]);
            let cx = projected[0] as i32;
            let cy = projected[1] as i32;

            // Draw current frame as a distinct marker
            let color: Color = [255, 0, 255, 255]; // Magenta

            // Draw larger marker for current frame
            self.draw_circle(cx, cy, (self.point_size * 2) as i32, color);

            // Draw direction indicator
            let rot = pose.rotation_matrix();
            let fx = rot[0][2];
            let fz = rot[2][2];

            let dir_len = 40.0 * self.camera_scale;
            let dir_x = cx + (fx * dir_len) as i32;
            let dir_y = cy + (fz * dir_len) as i32;

            self.draw_line(cx, cy, dir_x, dir_y, color);
        }

        Ok(())
    }

    /// Save the current image to a file
    ///
    /// Saves as PNG format
    ///
    /// # Arguments
    /// * `path` - Output file path
    ///
    /// # Returns
    /// Ok(()) on success, Err(message) on failure
    pub fn save_image(&self, path: &str) -> Result<(), String> {
        // Try to use image crate if available, otherwise return error
        #[cfg(feature = "image")]
        {
            use image::{ImageBuffer, Rgba};

            let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
                ImageBuffer::from_raw(self.width, self.height, self.buffer.clone())
                    .ok_or("Failed to create image buffer")?;

            img.save(path)
                .map_err(|e| format!("Failed to save image: {}", e))?;
            Ok(())
        }

        #[cfg(not(feature = "image"))]
        {
            // Fallback: try to encode as raw RGBA and save
            // This is a minimal fallback - in practice you'd want the image crate
            use std::fs::File;
            use std::io::Write;

            let mut file = File::create(path).map_err(|e| e.to_string())?;

            // Write simple PPM format (easier to debug without dependencies)
            // PPM format: P6 width height maxval\n followed by RGB data
            let header = format!("P6\n{} {}\n255\n", self.width, self.height);
            file.write_all(header.as_bytes())
                .map_err(|e| e.to_string())?;

            // Convert RGBA to RGB for PPM
            for chunk in self.buffer.chunks(4) {
                file.write_all(&[chunk[0], chunk[1], chunk[2]])
                    .map_err(|e| e.to_string())?;
            }

            // Also save RGBA as separate file with .rgba extension for debugging
            let rgba_path = format!("{}.rgba", path);
            let mut rgba_file = File::create(&rgba_path).map_err(|e| e.to_string())?;
            rgba_file
                .write_all(&self.buffer)
                .map_err(|e| e.to_string())?;

            Ok(())
        }
    }

    /// Set the view center
    pub fn set_center(&mut self, x: f32, y: f32) {
        self.center_x = x;
        self.center_y = y;
    }

    /// Set the view scale (zoom)
    pub fn set_scale(&mut self, scale: f32) {
        self.scale = scale;
    }

    /// Auto-center the view based on map data
    pub fn auto_center(&mut self, keyframes: impl Iterator<Item = impl AsRef<KeyFrame>>) {
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_z = f32::INFINITY;
        let mut max_z = f32::NEG_INFINITY;

        for kf in keyframes {
            if let Some(pose) = kf.as_ref().pose() {
                let t = pose.translation();
                min_x = min_x.min(t[0]);
                max_x = max_x.max(t[0]);
                min_z = min_z.min(t[2]);
                max_z = max_z.max(t[2]);
            }
        }

        if min_x.is_finite() && max_x.is_finite() {
            self.center_x = (min_x + max_x) / 2.0;
        }
        if min_z.is_finite() && max_z.is_finite() {
            self.center_y = (min_z + max_z) / 2.0;
        }
    }

    /// Auto-zoom to fit all keyframes
    pub fn auto_zoom(&mut self, keyframes: impl Iterator<Item = impl AsRef<KeyFrame>>) {
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_z = f32::INFINITY;
        let mut max_z = f32::NEG_INFINITY;

        for kf in keyframes {
            if let Some(pose) = kf.as_ref().pose() {
                let t = pose.translation();
                min_x = min_x.min(t[0]);
                max_x = max_x.max(t[0]);
                min_z = min_z.min(t[2]);
                max_z = max_z.max(t[2]);
            }
        }

        if min_x.is_finite() && max_x.is_finite() && min_z.is_finite() && max_z.is_finite() {
            let range_x = max_x - min_x;
            let range_z = max_z - min_z;
            let range = range_x.max(range_z);

            if range > 0.0 {
                // Add some padding
                let padding = 1.2;
                let scale_x = self.width as f32 / (range_x * padding);
                let scale_z = self.height as f32 / (range_z * padding);
                self.scale = scale_x.min(scale_z).min(500.0).max(10.0);
            }
        }
    }
}

impl Default for MapViewer {
    fn default() -> Self {
        Self::new(800, 600)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viewer_creation() {
        let viewer = MapViewer::new(800, 600);
        assert_eq!(viewer.width(), 800);
        assert_eq!(viewer.height(), 600);
    }

    #[test]
    fn test_projection() {
        let viewer = MapViewer::new(800, 600);
        let projected = viewer.project_to_viewport(&[0.0, 0.0, 0.0]);

        // Center should be at image center
        assert!((projected[0] - 400.0).abs() < 1.0);
        assert!((projected[1] - 300.0).abs() < 1.0);
    }

    #[test]
    fn test_depth_color() {
        let viewer = MapViewer::new(800, 600);

        // Near point should be green
        let color_near = viewer.get_depth_color(0.0, 0.0, 10.0);
        assert!(color_near[1] > color_near[0]); // More green than red

        // Far point should be red
        let color_far = viewer.get_depth_color(10.0, 0.0, 10.0);
        assert!(color_far[0] > color_far[1]); // More red than green
    }

    #[test]
    fn test_clear() {
        let mut viewer = MapViewer::new(100, 100);

        // Draw something
        viewer.draw_circle(50, 50, 10, [255, 0, 0, 255]);

        // Clear
        viewer.clear();

        // All pixels should be background color
        let bg = viewer.background_color();
        for i in (0..viewer.buffer.len()).step_by(4) {
            assert_eq!(viewer.buffer[i], bg[0]);
        }
    }
}
