//! Scene data model — pure CPU data, no GPU resources.

use bytemuck::{Pod, Zeroable};

/// Scene visibility flags per layer.
#[derive(Debug, Clone)]
pub struct LayerVisibility {
    pub trajectory: bool,
    pub map_points: bool,
    pub gaussians: bool,
    pub mesh_wireframe: bool,
    pub mesh_solid: bool,
}

impl Default for LayerVisibility {
    fn default() -> Self {
        Self {
            trajectory: true,
            map_points: true,
            gaussians: true,
            mesh_wireframe: false,
            mesh_solid: true,
        }
    }
}

/// Axis-aligned bounding box of the loaded scene.
#[derive(Debug, Clone)]
pub struct SceneBounds {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Default for SceneBounds {
    fn default() -> Self {
        Self {
            min: [f32::MAX; 3],
            max: [f32::MIN; 3],
        }
    }
}

impl SceneBounds {
    pub fn is_valid(&self) -> bool {
        self.min[0] < self.max[0]
    }

    pub fn extend(&mut self, p: [f32; 3]) {
        for i in 0..3 {
            if p[i] < self.min[i] {
                self.min[i] = p[i];
            }
            if p[i] > self.max[i] {
                self.max[i] = p[i];
            }
        }
    }

    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    pub fn diagonal(&self) -> f32 {
        let dx = self.max[0] - self.min[0];
        let dy = self.max[1] - self.min[1];
        let dz = self.max[2] - self.min[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// A single 3D Gaussian primitive for splat rendering.
#[derive(Debug, Clone, Copy)]
pub struct GaussianSplat {
    pub position: [f32; 3],
    pub scale: [f32; 3],
    pub rotation: [f32; 4],
    pub opacity: f32,
    pub color: [f32; 3],
}

/// GPU-ready mesh vertex.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MeshGpuVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

/// All scene data loaded from files.
#[derive(Debug, Default)]
pub struct Scene {
    /// Camera position sequence (world space).
    pub trajectory: Vec<[f32; 3]>,
    /// Map point positions (world space).
    pub map_points: Vec<[f32; 3]>,
    /// Map point display colors (depth-shaded).
    pub map_point_colors: Vec<[f32; 3]>,
    /// Gaussian splats.
    pub gaussians: Vec<GaussianSplat>,
    /// Mesh vertices (GPU ready).
    pub mesh_vertices: Vec<MeshGpuVertex>,
    /// Mesh triangle indices.
    pub mesh_indices: Vec<u32>,
    /// Mesh edge indices for wireframe rendering.
    pub mesh_edge_indices: Vec<u32>,
    /// Layer visibility flags.
    pub layers: LayerVisibility,
    /// Axis-aligned bounding box of the full scene.
    pub bounds: SceneBounds,
}

impl Scene {
    pub fn has_data(&self) -> bool {
        !self.trajectory.is_empty()
            || !self.map_points.is_empty()
            || !self.gaussians.is_empty()
            || !self.mesh_vertices.is_empty()
    }

    pub fn keyframe_count(&self) -> usize {
        self.trajectory.len()
    }

    pub fn map_point_count(&self) -> usize {
        self.map_points.len()
    }

    pub fn gaussian_count(&self) -> usize {
        self.gaussians.len()
    }

    pub fn mesh_vertex_count(&self) -> usize {
        self.mesh_vertices.len()
    }

    pub fn recompute_bounds(&mut self) {
        let mut bounds = SceneBounds::default();

        for &position in &self.trajectory {
            bounds.extend(position);
        }
        for &position in &self.map_points {
            bounds.extend(position);
        }
        for gaussian in &self.gaussians {
            bounds.extend(gaussian.position);
        }
        for vertex in &self.mesh_vertices {
            bounds.extend(vertex.position);
        }

        self.bounds = bounds;
    }
}
