//! Arcball camera controller for 3D navigation.

use crate::renderer::scene::SceneBounds;
use nalgebra::{Matrix3, Matrix4, Point3, Unit, UnitQuaternion, Vector3, Vector4};
use rustscan_types::matrix4_to_column_major_array;

pub type Vec3 = Vector3<f32>;
pub type Quat = UnitQuaternion<f32>;

pub(crate) fn normalize_or_zero(v: Vec3) -> Vec3 {
    v.try_normalize(f32::EPSILON).unwrap_or_else(Vec3::zeros)
}

pub(crate) fn vec3_is_finite(v: Vec3) -> bool {
    v.iter().copied().all(f32::is_finite)
}

pub(crate) fn vec4_xyz(v: Vector4<f32>) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

/// wgpu/Vulkan clip-space perspective: NDC z in `[0, 1]`.
///
/// nalgebra's `new_perspective` uses OpenGL `[-1, 1]` depth, so this keeps the
/// previous glam `perspective_rh` convention used by the wgpu mesh uniforms.
pub(crate) fn perspective_rh_wgpu(
    fov_y: f32,
    aspect: f32,
    z_near: f32,
    z_far: f32,
) -> Matrix4<f32> {
    let (sin_fov, cos_fov) = (0.5 * fov_y).sin_cos();
    let h = cos_fov / sin_fov;
    let w = h / aspect;
    let r = z_far / (z_near - z_far);
    Matrix4::new(
        w,
        0.0,
        0.0,
        0.0,
        0.0,
        h,
        0.0,
        0.0,
        0.0,
        0.0,
        r,
        r * z_near,
        0.0,
        0.0,
        -1.0,
        0.0,
    )
}

pub(crate) fn look_at_rh(eye: Vec3, target: Vec3, up: Vec3) -> Matrix4<f32> {
    Matrix4::look_at_rh(&Point3::from(eye), &Point3::from(target), &up)
}

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
        Self::from_angles(
            Vec3::zeros(),
            5.0,
            0.0,
            0.3,
            0.0,
            std::f32::consts::FRAC_PI_4,
        )
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
        let yaw = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), yaw);
        let pitch = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), -pitch);
        let roll = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), roll);
        Self {
            target,
            distance,
            orientation: yaw * pitch * roll,
            fov_y,
        }
    }

    pub fn from_raw(target: Vec3, distance: f32, orientation: Quat, fov_y: f32) -> Self {
        Self {
            target,
            distance: distance.max(Self::MIN_DISTANCE),
            orientation,
            fov_y,
        }
    }

    pub fn from_eye_target(eye: Vec3, target: Vec3, up_hint: Vec3, fov_y: f32) -> Self {
        let distance = (eye - target).norm().max(Self::MIN_DISTANCE);
        let target = if (eye - target).norm_squared() <= 1e-8 {
            eye - Vector3::z()
        } else {
            target
        };
        let backward = normalize_or_zero(eye - target);
        let right = normalize_or_zero(up_hint.cross(&backward));
        let up = normalize_or_zero(backward.cross(&right));
        let orientation = if backward.norm_squared() > 0.0
            && right.norm_squared() > 0.0
            && up.norm_squared() > 0.0
        {
            UnitQuaternion::from_matrix(&Matrix3::from_columns(&[right, up, backward]))
        } else {
            UnitQuaternion::identity()
        };
        Self::from_raw(target, distance, orientation, fov_y)
    }

    /// Eye position in world space.
    pub fn eye(&self) -> Vec3 {
        self.target + self.backward() * self.distance
    }

    /// Camera right direction in world space.
    pub fn right(&self) -> Vec3 {
        (self.orientation * Vector3::x()).normalize()
    }

    /// Camera up direction in world space.
    pub fn up(&self) -> Vec3 {
        (self.orientation * Vector3::y()).normalize()
    }

    /// Direction from target to eye in world space.
    pub fn backward(&self) -> Vec3 {
        (self.orientation * Vector3::z()).normalize()
    }

    /// View matrix (right-handed, looking from eye toward target).
    pub fn view_matrix(&self) -> Matrix4<f32> {
        look_at_rh(self.eye(), self.target, self.up())
    }

    /// Perspective projection matrix in wgpu clip space.
    pub fn proj_matrix(&self, aspect: f32) -> Matrix4<f32> {
        perspective_rh_wgpu(self.fov_y, aspect, 0.01, 1000.0)
    }

    /// Combined view-projection matrix.
    pub fn view_proj(&self, aspect: f32) -> Matrix4<f32> {
        self.proj_matrix(aspect) * self.view_matrix()
    }

    /// Column-major `[[column]; 4]` packing for wgpu uniforms: `array[col][row]`.
    pub fn view_proj_column_major_array(&self, aspect: f32) -> [[f32; 4]; 4] {
        matrix4_to_column_major_array(self.view_proj(aspect))
    }

    /// Orbit around target (left-button drag).
    pub fn orbit(&mut self, delta_x: f32, delta_y: f32) {
        // Match common DCC/model-viewer interaction:
        // dragging the pointer should make the scene appear to move
        // in the same direction as the drag.
        let yaw = UnitQuaternion::from_axis_angle(
            &Unit::new_normalize(self.up()),
            -delta_x * Self::ORBIT_SENSITIVITY,
        );
        let pitch = UnitQuaternion::from_axis_angle(
            &Unit::new_normalize(self.right()),
            -delta_y * Self::ORBIT_SENSITIVITY,
        );
        self.orientation = yaw * pitch * self.orientation;
    }

    /// Roll the camera around the current viewing direction.
    pub fn roll(&mut self, delta_x: f32) {
        let roll = UnitQuaternion::from_axis_angle(
            &Unit::new_normalize(self.backward()),
            -delta_x * Self::ORBIT_SENSITIVITY,
        );
        self.orientation = roll * self.orientation;
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
        let reference_up = if backward.dot(&Vector3::y()).abs() > 0.999 {
            Vector3::z()
        } else {
            Vector3::y()
        };
        let reference_up = (reference_up - backward * reference_up.dot(&backward)).normalize();
        let up = self.up();
        let roll = reference_up
            .cross(&up)
            .dot(&backward)
            .atan2(reference_up.dot(&up));
        (yaw, pitch, roll)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustscan_types::matrix4_from_column_major_array;

    #[test]
    fn test_eye_position() {
        let cam = ArcballCamera::from_angles(
            Vec3::zeros(),
            5.0,
            0.0,
            0.0,
            0.0,
            std::f32::consts::FRAC_PI_4,
        );
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
        assert!(vec3_is_finite(cam.up()));
        assert!(vec3_is_finite(cam.backward()));
        assert!(cam.up().dot(&cam.backward()).abs() < 1e-4);
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

        assert!((cam.eye() - eye).norm() < 1e-4);
        assert!(cam.up().dot(&up) < 0.99);
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

        assert!((cam.eye() - eye).norm() < 1e-4);
        assert!((cam.target - target).norm() < 1e-4);
    }

    #[test]
    fn view_proj_column_major_array_round_trips_non_symmetric_pose() {
        let cam = ArcballCamera::from_eye_target(
            Vec3::new(1.25, -2.5, 3.75),
            Vec3::new(0.4, -1.1, 2.2),
            Vec3::new(0.2, 0.9, 0.1),
            std::f32::consts::FRAC_PI_4,
        );
        let matrix = cam.view_proj(16.0 / 9.0);
        let columns = cam.view_proj_column_major_array(16.0 / 9.0);
        let restored = matrix4_from_column_major_array(&columns);
        let point = Vector4::new(0.7, -1.2, 2.4, 1.0);
        assert!((restored * point - matrix * point).norm() < 1e-5);
        assert!((columns[3][0] - matrix[(0, 3)]).abs() < 1e-6);
    }
}
