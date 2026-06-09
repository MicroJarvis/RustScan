//! Robot navigation state and proxy rendering helpers.

use std::sync::OnceLock;

use glam::{Mat3, Quat, Vec3};

use crate::renderer::camera::ArcballCamera;
use crate::renderer::scene::{MeshGpuVertex, Scene, SceneBounds};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavigationMode {
    Orbit,
    Robot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobotCameraMode {
    Follow,
    FirstPerson,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobotController {
    pub visible: bool,
    pub camera_mode: RobotCameraMode,
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub ground_plane: Option<GroundPlane>,
    pub ground_height: f32,
    pub walk_bounds: Option<WalkBounds>,
    pub model_height: f32,
    pub move_speed: f32,
    pub turn_speed: f32,
    pub fov_y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WalkBounds {
    pub min_x: f32,
    pub max_x: f32,
    pub min_z: f32,
    pub max_z: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundPlane {
    pub origin: Vec3,
    pub up: Vec3,
}

impl GroundPlane {
    pub fn new(origin: Vec3, up: Vec3) -> Option<Self> {
        let up = up.normalize_or_zero();
        (origin.is_finite() && up.length_squared() > 0.0).then_some(Self { origin, up })
    }

    pub fn from_points(points: &[Vec3], camera_eye: Vec3) -> Option<Self> {
        if points.len() < 3 {
            return None;
        }
        let origin = points.iter().copied().sum::<Vec3>() / points.len() as f32;
        let mut normal = Vec3::ZERO;
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                for k in (j + 1)..points.len() {
                    let tri_normal = (points[j] - points[i]).cross(points[k] - points[i]);
                    if tri_normal.length_squared() <= 1e-10 {
                        continue;
                    }
                    normal += if normal.dot(tri_normal) < 0.0 {
                        -tri_normal
                    } else {
                        tri_normal
                    };
                }
            }
        }
        if normal.length_squared() <= 1e-10 {
            return None;
        }
        let mut up = normal.normalize();
        if camera_eye.is_finite() && up.dot(camera_eye - origin) < 0.0 {
            up = -up;
        }
        Self::new(origin, up)
    }

    pub fn project_point(&self, point: Vec3) -> Vec3 {
        point - self.up * (point - self.origin).dot(self.up)
    }

    pub fn project_direction(&self, direction: Vec3) -> Vec3 {
        direction - self.up * direction.dot(self.up)
    }
}

#[derive(Debug, Clone)]
pub struct RobotRenderMesh {
    pub vertices: Vec<MeshGpuVertex>,
    pub indices: Vec<u32>,
    pub edge_indices: Vec<u32>,
}

struct BakedRobotMesh {
    vertices: Vec<MeshGpuVertex>,
    indices: Vec<u32>,
    edge_indices: Vec<u32>,
    height: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RobotInput {
    pub forward: f32,
    pub strafe: f32,
    pub turn: f32,
}

impl Default for RobotController {
    fn default() -> Self {
        Self {
            visible: true,
            camera_mode: RobotCameraMode::Follow,
            position: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            ground_plane: None,
            ground_height: 0.0,
            walk_bounds: Some(WalkBounds {
                min_x: -10.0,
                max_x: 10.0,
                min_z: -10.0,
                max_z: 10.0,
            }),
            model_height: 1.32,
            move_speed: 1.0,
            turn_speed: 1.8,
            fov_y: std::f32::consts::FRAC_PI_4,
        }
    }
}

impl RobotController {
    const MOUSE_LOOK_SENSITIVITY: f32 = 0.006;
    const BAKED_G1_HEIGHT: f32 = 1.322_845_5;
    const MIN_MODEL_HEIGHT: f32 = 0.25;
    const MAX_MODEL_HEIGHT: f32 = 2.0;
    const MIN_PITCH: f32 = -1.2;
    const MAX_PITCH: f32 = 1.2;

    pub fn reset_to_scene(&mut self, bounds: &SceneBounds) {
        if !bounds.is_valid() {
            self.position = Vec3::ZERO;
            self.ground_plane = None;
            self.ground_height = 0.0;
            self.walk_bounds = Some(WalkBounds {
                min_x: -10.0,
                max_x: 10.0,
                min_z: -10.0,
                max_z: 10.0,
            });
            return;
        }

        let center = bounds.center();
        let diagonal = bounds.diagonal().max(1.0);
        self.ground_plane = None;
        self.ground_height = bounds.min[1];
        self.walk_bounds = Some(WalkBounds {
            min_x: bounds.min[0],
            max_x: bounds.max[0],
            min_z: bounds.min[2],
            max_z: bounds.max[2],
        });
        self.model_height = (diagonal * 0.18).clamp(Self::MIN_MODEL_HEIGHT, Self::MAX_MODEL_HEIGHT);
        self.move_speed = (diagonal * 0.18).clamp(0.15, 3.0);
        self.position = Vec3::new(center[0], self.ground_height, center[2]);
        self.yaw = 0.0;
        self.pitch = 0.0;
        self.snap_to_ground();
    }

    pub fn sync_ground_from_scene(&mut self, bounds: &SceneBounds) {
        if !bounds.is_valid() {
            return;
        }
        if self.ground_plane.is_some() {
            self.position = self.constrain_to_ground(self.position);
            return;
        }
        self.set_ground_from_bounds(bounds);
        self.position = self.constrain_to_ground(self.position);
    }

    pub fn snap_to_scene_ground(&mut self, scene: &Scene) {
        if self.ground_plane.is_some() {
            self.position = self.constrain_to_ground(self.position);
            return;
        }
        if let Some(ground_height) = estimate_ground_height(scene, self.position) {
            self.ground_height = ground_height;
        } else if scene.bounds.is_valid() {
            self.set_ground_from_bounds(&scene.bounds);
        }
        self.position = self.constrain_to_ground(self.position);
    }

    fn set_ground_from_bounds(&mut self, bounds: &SceneBounds) {
        self.ground_height = bounds.min[1];
        self.walk_bounds = Some(WalkBounds {
            min_x: bounds.min[0],
            max_x: bounds.max[0],
            min_z: bounds.min[2],
            max_z: bounds.max[2],
        });
    }

    pub fn set_ground_plane(&mut self, ground_plane: GroundPlane) {
        self.ground_plane = Some(ground_plane);
        self.position = self.constrain_to_ground(self.position);
        self.pitch = 0.0;
    }

    pub fn flip_ground_plane(&mut self) {
        if let Some(ground_plane) = self.ground_plane.as_mut() {
            ground_plane.up = -ground_plane.up;
            self.position = self.constrain_to_ground(self.position);
            self.pitch = 0.0;
        }
    }

    pub fn place_in_camera_view(&mut self, camera: &ArcballCamera, scene: &Scene) {
        self.sync_ground_from_scene(&scene.bounds);

        let eye = camera.eye();
        let view_forward = (camera.target - eye).normalize_or_zero();
        let forward = self.project_direction_to_ground(view_forward);
        let forward = if forward.length_squared() > 1e-8 {
            forward.normalize()
        } else {
            self.project_direction_to_ground(-camera.backward())
                .normalize_or_zero()
        };
        let mut position = camera.target;
        if !position.is_finite() {
            position = eye + forward * (self.model_height * 3.0).max(0.75);
        }
        if let Some(ground_plane) = self.ground_plane {
            position = ground_plane.project_point(position);
        } else {
            position.y = self.ground_height;
            if let Some(ground_height) = estimate_ground_height(scene, position) {
                self.ground_height = ground_height;
                position.y = ground_height;
            }
        }
        self.position = self.constrain_to_ground(position);

        if forward.length_squared() > 1e-8 {
            self.yaw = (-forward.x).atan2(-forward.z);
        }
        self.pitch = 0.0;
    }

    pub fn apply_input(&mut self, input: RobotInput, dt: f32) -> bool {
        if dt <= 0.0 {
            return false;
        }

        let mut changed = false;
        if input.turn != 0.0 {
            self.yaw += input.turn * self.turn_speed * dt;
            changed = true;
        }

        let forward = self.forward_flat();
        let right = self.right_flat();
        let movement = forward * input.forward + right * input.strafe;
        if movement.length_squared() > 1e-8 {
            let direction = movement.normalize();
            self.position += direction * self.move_speed * dt;
            self.position = self.constrain_to_ground(self.position);
            changed = true;
        }

        changed
    }

    pub fn look(&mut self, delta_x: f32, delta_y: f32) {
        self.yaw -= delta_x * Self::MOUSE_LOOK_SENSITIVITY;
        self.pitch = (self.pitch - delta_y * Self::MOUSE_LOOK_SENSITIVITY)
            .clamp(Self::MIN_PITCH, Self::MAX_PITCH);
    }

    pub fn adjust_speed(&mut self, scroll_delta: f32) {
        if scroll_delta == 0.0 {
            return;
        }
        let factor = if scroll_delta > 0.0 { 1.12 } else { 1.0 / 1.12 };
        self.move_speed = (self.move_speed * factor).clamp(0.05, 10.0);
    }

    pub fn snap_to_ground(&mut self) {
        self.position = self.constrain_to_ground(self.position);
    }

    pub fn camera(&self) -> ArcballCamera {
        match self.camera_mode {
            RobotCameraMode::Follow => self.follow_camera(),
            RobotCameraMode::FirstPerson => self.first_person_camera(),
        }
    }

    pub fn render_mesh(&self) -> Option<RobotRenderMesh> {
        if !self.visible {
            return None;
        }

        Some(self.g1_render_mesh())
    }

    pub fn forward_flat(&self) -> Vec3 {
        let up = self.ground_up();
        (Quat::from_axis_angle(up, self.yaw) * self.ground_reference_forward()).normalize()
    }

    fn right_flat(&self) -> Vec3 {
        self.forward_flat().cross(self.ground_up()).normalize()
    }

    fn constrain_to_ground(&self, mut position: Vec3) -> Vec3 {
        if let Some(ground_plane) = self.ground_plane {
            position = ground_plane.project_point(position);
            if let Some(bounds) = self.walk_bounds {
                position.x = position.x.clamp(bounds.min_x, bounds.max_x);
                position.z = position.z.clamp(bounds.min_z, bounds.max_z);
                position = ground_plane.project_point(position);
            }
            return position;
        }
        position.y = self.ground_height;
        if let Some(bounds) = self.walk_bounds {
            position.x = position.x.clamp(bounds.min_x, bounds.max_x);
            position.z = position.z.clamp(bounds.min_z, bounds.max_z);
        }
        position
    }

    fn first_person_camera(&self) -> ArcballCamera {
        let up = self.ground_up();
        let eye = self.position + up * self.model_height * 0.86;
        let forward = self.forward_with_pitch();
        ArcballCamera::from_eye_target(eye, eye + forward, up, self.fov_y)
    }

    fn follow_camera(&self) -> ArcballCamera {
        let up = self.ground_up();
        let forward = self.forward_flat();
        let target = self.position + up * self.model_height * 0.55;
        let eye = target - forward * self.model_height * 2.6 + up * self.model_height * 0.95;
        ArcballCamera::from_eye_target(eye, target, up, self.fov_y)
    }

    fn forward_with_pitch(&self) -> Vec3 {
        let right = self.right_flat();
        let forward = self.forward_flat();
        (Quat::from_axis_angle(right, self.pitch) * forward).normalize()
    }

    fn g1_render_mesh(&self) -> RobotRenderMesh {
        let baked = baked_g1_mesh();
        let scale = self.model_height.max(Self::MIN_MODEL_HEIGHT) / baked.height.max(1e-6);
        let rotation = self.body_rotation();
        let mut vertices = Vec::with_capacity(baked.vertices.len());

        for vertex in &baked.vertices {
            let local = Vec3::from_array(vertex.position) * scale;
            let world = self.position + rotation * local;
            let normal = rotation * Vec3::from_array(vertex.normal);
            vertices.push(MeshGpuVertex {
                position: world.to_array(),
                normal: normal.normalize_or_zero().to_array(),
                color: vertex.color,
            });
        }

        RobotRenderMesh {
            vertices,
            indices: baked.indices.clone(),
            edge_indices: baked.edge_indices.clone(),
        }
    }

    fn ground_up(&self) -> Vec3 {
        self.ground_plane
            .map(|ground_plane| ground_plane.up)
            .unwrap_or(Vec3::Y)
    }

    fn ground_reference_forward(&self) -> Vec3 {
        for candidate in [Vec3::NEG_Z, Vec3::X, Vec3::Z] {
            let forward = self.project_direction_to_ground(candidate);
            if forward.length_squared() > 1e-8 {
                return forward.normalize();
            }
        }
        Vec3::NEG_Z
    }

    fn project_direction_to_ground(&self, direction: Vec3) -> Vec3 {
        if let Some(ground_plane) = self.ground_plane {
            ground_plane.project_direction(direction)
        } else {
            Vec3::new(direction.x, 0.0, direction.z)
        }
    }

    fn body_rotation(&self) -> Quat {
        let up = self.ground_up();
        let forward = self.forward_flat();
        let right = forward.cross(up).normalize();
        Quat::from_mat3(&Mat3::from_cols(right, up, -forward)).normalize()
    }
}

fn baked_g1_mesh() -> &'static BakedRobotMesh {
    static MESH: OnceLock<BakedRobotMesh> = OnceLock::new();
    MESH.get_or_init(|| {
        parse_baked_mesh(include_bytes!("../assets/robots/g1/g1_29dof_rev_1_0.bmesh"))
            .unwrap_or_else(|| fallback_proxy_mesh(RobotController::BAKED_G1_HEIGHT))
    })
}

fn parse_baked_mesh(bytes: &[u8]) -> Option<BakedRobotMesh> {
    const HEADER: &[u8; 8] = b"G1BMESH1";
    if bytes.len() < 12 || &bytes[..8] != HEADER {
        return None;
    }

    let count = u32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize;
    let expected = 12 + count * 9 * 4;
    if bytes.len() != expected || count % 3 != 0 {
        return None;
    }

    let mut vertices = Vec::with_capacity(count);
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut offset = 12;
    for _ in 0..count {
        let mut values = [0.0f32; 9];
        for value in &mut values {
            *value = f32::from_le_bytes(bytes[offset..offset + 4].try_into().ok()?);
            offset += 4;
        }
        min_y = min_y.min(values[1]);
        max_y = max_y.max(values[1]);
        vertices.push(MeshGpuVertex {
            position: [values[0], values[1], values[2]],
            normal: [values[3], values[4], values[5]],
            color: [values[6], values[7], values[8]],
        });
    }

    let indices = (0..count as u32).collect();
    let edge_indices = triangle_edges(count);
    Some(BakedRobotMesh {
        vertices,
        indices,
        edge_indices,
        height: (max_y - min_y).max(RobotController::BAKED_G1_HEIGHT),
    })
}

fn triangle_edges(vertex_count: usize) -> Vec<u32> {
    let mut edges = Vec::with_capacity(vertex_count * 2);
    for tri in (0..vertex_count as u32).step_by(3) {
        edges.extend([tri, tri + 1, tri + 1, tri + 2, tri + 2, tri]);
    }
    edges
}

fn fallback_proxy_mesh(height: f32) -> BakedRobotMesh {
    let h = height;
    let mut vertices = Vec::new();
    let mut add_box = |center: Vec3, half_extents: Vec3, color: [f32; 3]| {
        let corners = [
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
        ];
        let faces = [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 6, 2],
            [3, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [0, 4, 7],
            [0, 7, 3],
        ];
        for face in faces {
            let p0 = center + corners[face[0]] * half_extents;
            let p1 = center + corners[face[1]] * half_extents;
            let p2 = center + corners[face[2]] * half_extents;
            let normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();
            for point in [p0, p1, p2] {
                vertices.push(MeshGpuVertex {
                    position: point.to_array(),
                    normal: normal.to_array(),
                    color,
                });
            }
        }
    };
    add_box(
        Vec3::new(0.0, h * 0.5, 0.0),
        Vec3::new(h * 0.16, h * 0.28, h * 0.11),
        [0.75, 0.78, 0.80],
    );
    add_box(
        Vec3::new(0.0, h * 0.86, 0.0),
        Vec3::new(h * 0.13, h * 0.11, h * 0.12),
        [0.12, 0.14, 0.16],
    );

    let indices = (0..vertices.len() as u32).collect();
    let edge_indices = triangle_edges(vertices.len());
    BakedRobotMesh {
        vertices,
        indices,
        edge_indices,
        height,
    }
}

fn estimate_ground_height(scene: &Scene, position: Vec3) -> Option<f32> {
    raycast_mesh_ground(scene, position).or_else(|| estimate_point_cloud_ground(scene, position))
}

fn raycast_mesh_ground(scene: &Scene, position: Vec3) -> Option<f32> {
    if scene.mesh_indices.len() < 3 || scene.mesh_vertices.is_empty() {
        return None;
    }

    let mut best_y = f32::NEG_INFINITY;
    for tri in scene.mesh_indices.chunks_exact(3) {
        let a = scene.mesh_vertices.get(tri[0] as usize)?.position;
        let b = scene.mesh_vertices.get(tri[1] as usize)?.position;
        let c = scene.mesh_vertices.get(tri[2] as usize)?.position;
        if let Some(y) = triangle_height_at_xz(position.x, position.z, a, b, c) {
            if y <= position.y + scene.bounds.diagonal().max(1.0) && y > best_y {
                best_y = y;
            }
        }
    }

    best_y.is_finite().then_some(best_y)
}

fn triangle_height_at_xz(x: f32, z: f32, a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> Option<f32> {
    let ax = a[0];
    let az = a[2];
    let bx = b[0];
    let bz = b[2];
    let cx = c[0];
    let cz = c[2];
    let denom = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz);
    if denom.abs() < 1e-8 {
        return None;
    }

    let w0 = ((bz - cz) * (x - cx) + (cx - bx) * (z - cz)) / denom;
    let w1 = ((cz - az) * (x - cx) + (ax - cx) * (z - cz)) / denom;
    let w2 = 1.0 - w0 - w1;
    let epsilon = -1e-4;
    (w0 >= epsilon && w1 >= epsilon && w2 >= epsilon).then_some(w0 * a[1] + w1 * b[1] + w2 * c[1])
}

fn estimate_point_cloud_ground(scene: &Scene, position: Vec3) -> Option<f32> {
    if !scene.bounds.is_valid() {
        return None;
    }

    let diagonal = scene.bounds.diagonal().max(1.0);
    let mut radius = (diagonal * 0.04).max(0.15);
    let max_radius = diagonal * 0.35;
    while radius <= max_radius {
        let radius2 = radius * radius;
        let mut heights = Vec::new();
        collect_nearby_heights(&scene.map_points, position, radius2, &mut heights);
        for gaussian in &scene.gaussians {
            let p = gaussian.position;
            let dx = p[0] - position.x;
            let dz = p[2] - position.z;
            if dx * dx + dz * dz <= radius2 {
                heights.push(p[1]);
            }
        }

        if heights.len() >= 8 {
            heights.sort_by(|a, b| a.total_cmp(b));
            let idx = (heights.len() / 10).min(heights.len() - 1);
            return Some(heights[idx]);
        }

        radius *= 2.0;
    }

    Some(scene.bounds.min[1])
}

fn collect_nearby_heights(
    points: &[[f32; 3]],
    position: Vec3,
    radius2: f32,
    heights: &mut Vec<f32>,
) {
    for p in points {
        let dx = p[0] - position.x;
        let dz = p[2] - position.z;
        if dx * dx + dz * dz <= radius2 {
            heights.push(p[1]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn robot_stays_on_ground_and_inside_bounds() {
        let mut robot = RobotController::default();
        robot.ground_height = 2.0;
        robot.walk_bounds = Some(WalkBounds {
            min_x: -1.0,
            max_x: 1.0,
            min_z: -1.0,
            max_z: 1.0,
        });
        robot.position = Vec3::new(0.0, 20.0, 0.0);
        robot.move_speed = 100.0;
        robot.apply_input(
            RobotInput {
                forward: 1.0,
                strafe: 1.0,
                turn: 0.0,
            },
            1.0,
        );
        assert_eq!(robot.position.y, 2.0);
        assert!(robot.position.x <= 1.0);
        assert!(robot.position.z >= -1.0);
    }

    #[test]
    fn robot_camera_is_finite() {
        let robot = RobotController::default();
        let camera = robot.camera();
        assert!(camera.eye().is_finite());
        assert!(camera.target.is_finite());
    }

    #[test]
    fn ground_plane_from_points_chooses_up_toward_camera() {
        let points = [
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(-1.0, 0.0, 1.0),
        ];

        let plane = GroundPlane::from_points(&points, Vec3::new(0.0, 5.0, 0.0)).unwrap();
        assert!(plane.up.dot(Vec3::Y) > 0.999);

        let flipped = GroundPlane::from_points(&points, Vec3::new(0.0, -5.0, 0.0)).unwrap();
        assert!(flipped.up.dot(Vec3::NEG_Y) > 0.999);
    }

    #[test]
    fn robot_ground_plane_projects_position_and_mesh_above_ground() {
        let mut robot = RobotController::default();
        let up = Vec3::new(0.0, 1.0, 1.0).normalize();
        let plane = GroundPlane::new(Vec3::ZERO, up).unwrap();
        robot.set_ground_plane(plane);
        robot.position = Vec3::new(0.0, 5.0, 0.0);
        robot.snap_to_ground();

        assert!((robot.position - plane.project_point(robot.position)).length() < 1e-5);

        let mesh = robot.render_mesh().expect("robot mesh");
        let min_ground_distance = mesh
            .vertices
            .iter()
            .map(|vertex| (Vec3::from_array(vertex.position) - plane.origin).dot(plane.up))
            .fold(f32::INFINITY, f32::min);
        assert!(
            min_ground_distance >= -1e-4,
            "mesh should stay above fitted ground plane, min distance {min_ground_distance}"
        );
    }

    #[test]
    fn robot_can_be_placed_in_camera_view() {
        let mut scene = Scene::default();
        scene.bounds = SceneBounds {
            min: [-10.0, -0.2, -10.0],
            max: [10.0, 0.8, 10.0],
        };
        let camera = ArcballCamera::from_eye_target(
            Vec3::new(0.0, 1.0, 8.0),
            Vec3::new(3.0, 0.4, -4.0),
            Vec3::Y,
            std::f32::consts::FRAC_PI_4,
        );
        let mut robot = RobotController::default();

        robot.place_in_camera_view(&camera, &scene);

        assert!((robot.position.x - camera.target.x).abs() < 1e-4);
        assert_eq!(robot.position.y, scene.bounds.min[1]);
        assert!((robot.position.z - camera.target.z).abs() < 1e-4);
    }

    #[test]
    fn baked_g1_mesh_loads() {
        let mesh = baked_g1_mesh();
        assert!(mesh.vertices.len() > 10_000);
        assert_eq!(mesh.indices.len(), mesh.vertices.len());
        assert!(mesh.height > 1.0);
        assert!(mesh.height < 1.5);
    }

    #[test]
    fn snap_ground_uses_mesh_under_robot() {
        let mut scene = Scene::default();
        scene.mesh_vertices = vec![
            MeshGpuVertex {
                position: [-1.0, 0.25, -1.0],
                normal: [0.0, 1.0, 0.0],
                color: [1.0, 1.0, 1.0],
            },
            MeshGpuVertex {
                position: [1.0, 0.25, -1.0],
                normal: [0.0, 1.0, 0.0],
                color: [1.0, 1.0, 1.0],
            },
            MeshGpuVertex {
                position: [1.0, 0.25, 1.0],
                normal: [0.0, 1.0, 0.0],
                color: [1.0, 1.0, 1.0],
            },
            MeshGpuVertex {
                position: [-1.0, 0.25, 1.0],
                normal: [0.0, 1.0, 0.0],
                color: [1.0, 1.0, 1.0],
            },
        ];
        scene.mesh_indices = vec![0, 1, 2, 0, 2, 3];
        scene.recompute_bounds();
        let mut robot = RobotController {
            position: Vec3::new(0.0, 5.0, 0.0),
            ..Default::default()
        };

        robot.snap_to_scene_ground(&scene);

        assert!((robot.position.y - 0.25).abs() < 1e-6);
    }

    #[test]
    fn snap_ground_uses_nearby_low_points() {
        let mut scene = Scene::default();
        for x in -2..=2 {
            for z in -2..=2 {
                scene.map_points.push([x as f32 * 0.1, 0.4, z as f32 * 0.1]);
            }
        }
        scene.map_points.push([0.0, 2.0, 0.0]);
        scene.recompute_bounds();
        let mut robot = RobotController {
            position: Vec3::new(0.0, 5.0, 0.0),
            ..Default::default()
        };

        robot.snap_to_scene_ground(&scene);

        assert!((robot.position.y - 0.4).abs() < 1e-6);
    }
}
