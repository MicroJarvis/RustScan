//! Arcball camera controller for 3D navigation.

use crate::renderer::scene::SceneBounds;
use glam::{Mat3, Mat4, Quat, Vec3};

/// Arcball camera: orbits around a target point.
#[derive(Debug, Clone)]
pub struct ArcballCamera {
    /// World-space look-at target.
    pub target: Vec3,
    /// Distance from target to eye.
    pub distance: f32,
    /// Camera orientation. Local +Z points from target to eye.
    orientation: Quat,
    /// Vertical field-of-view in radians.
    pub fov_y: f32,
}

impl Default for ArcballCamera {
    fn default() -> Self {
        Self::from_angles(Vec3::ZERO, 5.0, 0.0, 0.3, 0.0, std::f32::consts::FRAC_PI_4)
    }
}

impl ArcballCamera {
    const MIN_DISTANCE: f32 = 0.1;
    const ORBIT_SENSITIVITY: f32 = 0.005;
    const ZOOM_FACTOR: f32 = 1.1;
    const PAN_SCALE: f32 = 0.001;
    const FIT_MARGIN: f32 = 1.5;

    pub fn from_angles(
        target: Vec3,
        distance: f32,
        yaw: f32,
        pitch: f32,
        roll: f32,
        fov_y: f32,
    ) -> Self {
        let yaw = Quat::from_rotation_y(yaw);
        let pitch = Quat::from_rotation_x(-pitch);
        let roll = Quat::from_rotation_z(roll);
        Self {
            target,
            distance,
            orientation: (yaw * pitch * roll).normalize(),
            fov_y,
        }
    }

    pub fn from_raw(target: Vec3, distance: f32, orientation: Quat, fov_y: f32) -> Self {
        Self {
            target,
            distance: distance.max(Self::MIN_DISTANCE),
            orientation: orientation.normalize(),
            fov_y,
        }
    }

    pub fn from_eye_target(eye: Vec3, target: Vec3, up_hint: Vec3, fov_y: f32) -> Self {
        let distance = eye.distance(target).max(Self::MIN_DISTANCE);
        let target = if eye.distance_squared(target) <= 1e-8 {
            eye + Vec3::NEG_Z
        } else {
            target
        };
        let backward = (eye - target).normalize_or_zero();
        let right = up_hint.cross(backward).normalize_or_zero();
        let up = backward.cross(right).normalize_or_zero();
        let orientation = if backward.length_squared() > 0.0
            && right.length_squared() > 0.0
            && up.length_squared() > 0.0
        {
            Quat::from_mat3(&Mat3::from_cols(right, up, backward)).normalize()
        } else {
            Quat::IDENTITY
        };
        Self::from_raw(target, distance, orientation, fov_y)
    }

    /// Eye position in world space.
    pub fn eye(&self) -> Vec3 {
        self.target + self.backward() * self.distance
    }

    /// Camera right direction in world space.
    pub fn right(&self) -> Vec3 {
        (self.orientation * Vec3::X).normalize()
    }

    /// Camera up direction in world space.
    pub fn up(&self) -> Vec3 {
        (self.orientation * Vec3::Y).normalize()
    }

    /// Direction from target to eye in world space.
    pub fn backward(&self) -> Vec3 {
        (self.orientation * Vec3::Z).normalize()
    }

    /// View matrix (right-handed, looking from eye toward target).
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, self.up())
    }

    /// Perspective projection matrix.
    pub fn proj_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, aspect, 0.01, 1000.0)
    }

    /// Combined view-projection matrix.
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.proj_matrix(aspect) * self.view_matrix()
    }

    /// Orbit around target (left-button drag).
    pub fn orbit(&mut self, delta_x: f32, delta_y: f32) {
        // Match common DCC/model-viewer interaction:
        // dragging the pointer should make the scene appear to move
        // in the same direction as the drag.
        let yaw = Quat::from_axis_angle(self.up(), -delta_x * Self::ORBIT_SENSITIVITY);
        let pitch = Quat::from_axis_angle(self.right(), -delta_y * Self::ORBIT_SENSITIVITY);
        self.orientation = (yaw * pitch * self.orientation).normalize();
    }

    /// Roll the camera around the current viewing direction.
    pub fn roll(&mut self, delta_x: f32) {
        let roll = Quat::from_axis_angle(self.backward(), -delta_x * Self::ORBIT_SENSITIVITY);
        self.orientation = (roll * self.orientation).normalize();
    }

    /// Pan the target point (right-button drag).
    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let scale = self.distance * Self::PAN_SCALE;
        self.target -= self.right() * delta_x * scale;
        self.target += self.up() * delta_y * scale;
    }

    /// Zoom by scaling distance (scroll wheel).
    pub fn zoom(&mut self, delta: f32) {
        if delta > 0.0 {
            self.distance /= Self::ZOOM_FACTOR;
        } else if delta < 0.0 {
            self.distance *= Self::ZOOM_FACTOR;
        }
        self.distance = self.distance.max(Self::MIN_DISTANCE);
    }

    /// Re-center orbit controls on a world-space point while keeping the eye fixed.
    pub fn focus_on(&mut self, target: Vec3) {
        let eye = self.eye();
        *self = Self::from_eye_target(eye, target, self.up(), self.fov_y);
    }

    /// Fit the camera to display the full scene bounds.
    pub fn fit_scene(&mut self, bounds: &SceneBounds) {
        if !bounds.is_valid() {
            return;
        }
        let center = bounds.center();
        self.target = Vec3::new(center[0], center[1], center[2]);
        let diag = bounds.diagonal();
        self.distance = (diag * Self::FIT_MARGIN).max(Self::MIN_DISTANCE);
    }

    pub fn display_angles(&self) -> (f32, f32, f32) {
        let backward = self.backward();
        let yaw = backward.x.atan2(backward.z);
        let pitch = backward.y.clamp(-1.0, 1.0).asin();
        let reference_up = if backward.dot(Vec3::Y).abs() > 0.999 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        let reference_up = (reference_up - backward * reference_up.dot(backward)).normalize();
        let up = self.up();
        let roll = reference_up
            .cross(up)
            .dot(backward)
            .atan2(reference_up.dot(up));
        (yaw, pitch, roll)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eye_position() {
        let cam =
            ArcballCamera::from_angles(Vec3::ZERO, 5.0, 0.0, 0.0, 0.0, std::f32::consts::FRAC_PI_4);
        let eye = cam.eye();
        assert!(
            (eye.z - 5.0).abs() < 1e-4,
            "eye.z should be ~5.0, got {}",
            eye.z
        );
        assert!(eye.x.abs() < 1e-4);
        assert!(eye.y.abs() < 1e-4);
    }

    #[test]
    fn test_orbit_can_cross_poles_without_losing_up_vector() {
        let mut cam = ArcballCamera::default();
        cam.orbit(0.0, -400.0);
        assert!(cam.up().is_finite());
        assert!(cam.backward().is_finite());
        assert!(cam.up().dot(cam.backward()).abs() < 1e-4);
    }

    #[test]
    fn test_orbit_drag_direction_follows_pointer() {
        let mut cam = ArcballCamera::default();
        let (initial_yaw, initial_pitch, _) = cam.display_angles();

        // Dragging right should rotate the view so the scene appears to move right.
        cam.orbit(100.0, 0.0);
        assert!(cam.display_angles().0 < initial_yaw);

        // Dragging up should rotate the view so the scene appears to move up.
        cam.orbit(0.0, -100.0);
        assert!(cam.display_angles().1 < initial_pitch);
    }

    #[test]
    fn test_roll_keeps_eye_position_and_changes_up_vector() {
        let mut cam = ArcballCamera::default();
        let eye = cam.eye();
        let up = cam.up();

        cam.roll(100.0);

        assert!((cam.eye() - eye).length() < 1e-4);
        assert!(cam.up().dot(up) < 0.99);
    }

    #[test]
    fn test_zoom_clamp() {
        let mut cam = ArcballCamera::default();
        // Zoom in extremely
        for _ in 0..1000 {
            cam.zoom(1.0);
        }
        assert!(cam.distance >= ArcballCamera::MIN_DISTANCE);
    }

    #[test]
    fn focus_on_keeps_eye_and_updates_target() {
        let mut cam = ArcballCamera::default();
        let eye = cam.eye();
        let target = Vec3::new(1.0, 2.0, -3.0);

        cam.focus_on(target);

        assert!((cam.eye() - eye).length() < 1e-4);
        assert!((cam.target - target).length() < 1e-4);
    }
}
