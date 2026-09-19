//! Mesh Extractor for 3DGS -> Mesh Pipeline
//!
//! High-level API for extracting mesh from Gaussian Splatting representation.
//! Combines TSDF fusion and Marching Cubes with post-processing.
//!
//! Based on PGSR strategy:
//! 1. Render depth from Gaussians
//! 2. TSDF Fusion
//! 3. Marching Cubes extraction
//! 4. Post-processing (clustering, floaters removal)

use nalgebra::{Matrix4, Point3, Vector3};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub use crate::fusion::marching_cubes::{MarchingCubes, Mesh, MeshTriangle, MeshVertex};
use crate::fusion::mesh_io::{export_mesh, MeshIoError};
use crate::fusion::mesh_metadata::{
    export_mesh_metadata, BoundingBox, MeshMetadata, MeshMetadataError, MeshTimings, TsdfMetadata,
};
pub use crate::fusion::tsdf_volume::{TsdfConfig, TsdfVolume};

/// Mesh extraction configuration
#[derive(Debug, Clone)]
pub struct MeshExtractionConfig {
    /// TSDF volume configuration
    pub tsdf_config: TsdfConfig,
    /// Minimum number of triangles in a cluster to keep
    pub min_cluster_size: usize,
    /// Number of largest clusters to keep
    pub num_largest_clusters: usize,
    /// Whether to compute smooth normals
    pub smooth_normals: bool,
    /// Normal smoothing iterations
    pub normal_smoothing_iterations: usize,
}

impl Default for MeshExtractionConfig {
    fn default() -> Self {
        Self {
            tsdf_config: TsdfConfig {
                voxel_size: 0.01, // 1cm
                sdf_trunc: 0.03,  // 3cm truncation
                min_bound: Point3::new(-1.0, -1.0, -1.0),
                max_bound: Point3::new(1.0, 1.0, 1.0),
                max_weight: 100.0,
                integration_weight: 1.0,
            },
            min_cluster_size: 100,
            num_largest_clusters: 1,
            smooth_normals: true,
            normal_smoothing_iterations: 3,
        }
    }
}

/// Mesh extractor combining TSDF and Marching Cubes
pub struct MeshExtractor {
    config: MeshExtractionConfig,
    volume: TsdfVolume,
    timings_ms: MeshTimings,
}

struct ClusterStats {
    removed_clusters: usize,
    removed_triangles: usize,
    kept_clusters: usize,
    kept_triangles: usize,
}

struct TopologyStats {
    boundary_edges: usize,
    non_manifold_edges: usize,
}

/// Results from post-processed mesh extraction.
#[derive(Debug)]
pub struct MeshExtractionReport {
    pub mesh: Mesh,
    pub isolated_triangle_percentage: f32,
}

impl MeshExtractor {
    pub fn new(config: MeshExtractionConfig) -> Self {
        Self {
            config: config.clone(),
            volume: TsdfVolume::new(config.tsdf_config),
            timings_ms: MeshTimings::default(),
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(MeshExtractionConfig::default())
    }

    /// Create volume centered at a point with given size
    pub fn centered(center: Point3<f32>, size_meters: f32, voxel_size: f32) -> Self {
        let half = size_meters / 2.0;
        let config = TsdfConfig {
            voxel_size,
            sdf_trunc: voxel_size * 3.0,
            min_bound: center - Vector3::repeat(half),
            max_bound: center + Vector3::repeat(half),
            max_weight: 100.0,
            integration_weight: 1.0,
        };
        let config_clone = config.clone();

        Self {
            config: MeshExtractionConfig {
                tsdf_config: config,
                ..Default::default()
            },
            volume: TsdfVolume::new(config_clone),
            timings_ms: MeshTimings::default(),
        }
    }

    /// Integrate a depth frame into the TSDF volume
    pub fn integrate_frame(
        &mut self,
        depth: &[f32],
        color: Option<&[[u8; 3]]>,
        width: usize,
        height: usize,
        intrinsics: [f32; 4],
        extrinsics: &Matrix4<f32>,
    ) {
        let start = Instant::now();
        self.volume
            .integrate(depth, color, width, height, intrinsics, extrinsics);
        self.timings_ms.tsdf_fusion_ms = self
            .timings_ms
            .tsdf_fusion_ms
            .saturating_add(start.elapsed().as_millis() as u64);
    }

    /// Integrate depth rendered from Gaussians
    pub fn integrate_from_gaussians<F>(
        &mut self,
        render_depth_fn: F,
        color: Option<&[[u8; 3]]>,
        width: usize,
        height: usize,
        intrinsics: [f32; 4],
        extrinsics: &Matrix4<f32>,
    ) where
        F: Fn(usize) -> f32,
    {
        let start = Instant::now();
        let color_fn: Option<Box<dyn Fn(usize) -> Option<[u8; 3]>>> = color.map(|c| {
            let c = c.to_vec();
            Box::new(move |i: usize| c.get(i).copied()) as Box<dyn Fn(usize) -> Option<[u8; 3]>>
        });
        self.volume.integrate_from_gaussians(
            render_depth_fn,
            color_fn
                .as_ref()
                .map(|f| f.as_ref() as &dyn Fn(usize) -> Option<[u8; 3]>),
            width,
            height,
            intrinsics,
            extrinsics,
        );
        self.timings_ms.tsdf_fusion_ms = self
            .timings_ms
            .tsdf_fusion_ms
            .saturating_add(start.elapsed().as_millis() as u64);
    }

    /// Extract mesh from the TSDF volume
    pub fn extract(&self) -> Mesh {
        extract_mesh_from_tsdf(&self.volume)
    }

    /// Extract and post-process mesh
    pub fn extract_with_postprocessing(&mut self) -> Mesh {
        self.extract_with_postprocessing_report().mesh
    }

    /// Extract and post-process mesh with statistics for metadata export.
    pub fn extract_with_postprocessing_report(&mut self) -> MeshExtractionReport {
        let march_start = Instant::now();
        let mut mesh = self.extract();
        self.timings_ms.marching_cubes_ms = self
            .timings_ms
            .marching_cubes_ms
            .saturating_add(march_start.elapsed().as_millis() as u64);

        let post_start = Instant::now();
        let pre_triangles = mesh.triangles.len();
        let pre_vertices = mesh.vertices.len();
        let mut cluster_stats = ClusterStats {
            removed_clusters: 0,
            removed_triangles: 0,
            kept_clusters: 0,
            kept_triangles: pre_triangles,
        };

        // Post-processing
        if self.config.min_cluster_size > 0 || self.config.num_largest_clusters > 0 {
            let (filtered, stats) = self.filter_clusters(mesh);
            mesh = filtered;
            cluster_stats = stats;
        }

        if self.config.smooth_normals && self.config.normal_smoothing_iterations > 0 {
            mesh = self.smooth_normals(mesh);
        }

        let topo_stats = self.validate_topology(&mesh);
        let isolated_pct = if pre_triangles > 0 {
            (cluster_stats.removed_triangles as f32 / pre_triangles as f32) * 100.0
        } else {
            0.0
        };

        log::info!(
            "Mesh stats: vertices {} -> {}, triangles {} -> {}, kept_clusters {}, kept_triangles {}, removed_clusters {}, removed_triangles {}, isolated_pct {:.2}%",
            pre_vertices,
            mesh.vertices.len(),
            pre_triangles,
            mesh.triangles.len(),
            cluster_stats.kept_clusters,
            cluster_stats.kept_triangles,
            cluster_stats.removed_clusters,
            cluster_stats.removed_triangles,
            isolated_pct,
        );
        log::info!(
            "Mesh topology: boundary_edges={}, non_manifold_edges={}, watertight={}",
            topo_stats.boundary_edges,
            topo_stats.non_manifold_edges,
            topo_stats.boundary_edges == 0 && topo_stats.non_manifold_edges == 0
        );
        if isolated_pct > 1.0 {
            log::warn!(
                "Isolated triangle percentage {:.2}% exceeds 1% threshold before filtering",
                isolated_pct
            );
        }

        self.timings_ms.post_process_ms = self
            .timings_ms
            .post_process_ms
            .saturating_add(post_start.elapsed().as_millis() as u64);

        MeshExtractionReport {
            mesh,
            isolated_triangle_percentage: isolated_pct,
        }
    }

    /// Export mesh to OBJ and PLY under the output directory.
    pub fn export_mesh_files(
        &self,
        mesh: &Mesh,
        output_dir: &Path,
    ) -> Result<(PathBuf, PathBuf), MeshIoError> {
        export_mesh(output_dir, mesh)
    }

    /// Export mesh metadata JSON under the output directory.
    pub fn export_mesh_metadata_files(
        &self,
        mesh: &Mesh,
        isolated_triangle_percentage: f32,
        output_dir: &Path,
    ) -> Result<PathBuf, MeshMetadataError> {
        let metadata = self.build_mesh_metadata(mesh, isolated_triangle_percentage);
        export_mesh_metadata(output_dir, &metadata)
    }

    /// Build mesh metadata for export.
    pub fn build_mesh_metadata(
        &self,
        mesh: &Mesh,
        isolated_triangle_percentage: f32,
    ) -> MeshMetadata {
        let (min, max) = mesh_bounding_box(mesh);
        let tsdf = self.volume.config();
        MeshMetadata {
            vertex_count: mesh.vertices.len(),
            triangle_count: mesh.triangles.len(),
            bounding_box: BoundingBox { min, max },
            isolated_triangle_percentage,
            tsdf: TsdfMetadata {
                voxel_size: tsdf.voxel_size,
                truncation_distance: tsdf.sdf_trunc,
            },
            viewpoint_count: self.frame_count(),
            timings_ms: self.timings_ms.clone(),
        }
    }

    /// Filter out small clusters (floaters)
    fn filter_clusters(&self, mesh: Mesh) -> (Mesh, ClusterStats) {
        if mesh.triangles.is_empty() {
            return (
                mesh,
                ClusterStats {
                    removed_clusters: 0,
                    removed_triangles: 0,
                    kept_clusters: 0,
                    kept_triangles: 0,
                },
            );
        }

        let tri_count = mesh.triangles.len();
        let mut vert_tris: Vec<Vec<usize>> = vec![Vec::new(); mesh.vertices.len()];
        for (tri_idx, tri) in mesh.triangles.iter().enumerate() {
            for &v in &tri.indices {
                vert_tris[v].push(tri_idx);
            }
        }

        let mut tri_adj: Vec<Vec<usize>> = vec![Vec::new(); tri_count];
        for (tri_idx, tri) in mesh.triangles.iter().enumerate() {
            let mut neighbors: HashSet<usize> = HashSet::new();
            for &v in &tri.indices {
                for &other in &vert_tris[v] {
                    if other != tri_idx {
                        neighbors.insert(other);
                    }
                }
            }
            tri_adj[tri_idx] = neighbors.into_iter().collect();
        }

        // Connected components on triangles
        let mut visited = vec![false; tri_count];
        let mut clusters: Vec<Vec<usize>> = Vec::new();
        for start in 0..tri_count {
            if visited[start] {
                continue;
            }

            let mut cluster = Vec::new();
            let mut stack = vec![start];
            visited[start] = true;
            while let Some(t) = stack.pop() {
                cluster.push(t);
                for &n in &tri_adj[t] {
                    if !visited[n] {
                        visited[n] = true;
                        stack.push(n);
                    }
                }
            }

            clusters.push(cluster);
        }

        let mut cluster_indices: Vec<usize> = (0..clusters.len()).collect();
        cluster_indices.sort_by(|a, b| clusters[*b].len().cmp(&clusters[*a].len()));

        let mut keep_cluster = vec![false; clusters.len()];
        for (rank, &cluster_idx) in cluster_indices.iter().enumerate() {
            let cluster_size = clusters[cluster_idx].len();
            if self.config.min_cluster_size > 0 && cluster_size < self.config.min_cluster_size {
                continue;
            }
            if self.config.num_largest_clusters > 0 && rank >= self.config.num_largest_clusters {
                continue;
            }
            keep_cluster[cluster_idx] = true;
        }

        let mut keep_tri = vec![false; tri_count];
        let mut removed_triangles = 0usize;
        let mut kept_triangles = 0usize;
        let mut removed_clusters = 0usize;
        let mut kept_clusters = 0usize;

        for (idx, cluster) in clusters.iter().enumerate() {
            if keep_cluster[idx] {
                kept_clusters += 1;
                for &tri_idx in cluster {
                    keep_tri[tri_idx] = true;
                }
                kept_triangles += cluster.len();
            } else {
                removed_clusters += 1;
                removed_triangles += cluster.len();
            }
        }

        // Rebuild mesh
        let mut index_map = vec![usize::MAX; mesh.vertices.len()];
        let mut new_vertices = Vec::new();
        let mut new_triangles = Vec::new();

        for (tri_idx, tri) in mesh.triangles.iter().enumerate() {
            if !keep_tri[tri_idx] {
                continue;
            }

            let mut new_indices = [0usize; 3];
            for (i, &v) in tri.indices.iter().enumerate() {
                let mapped = if index_map[v] == usize::MAX {
                    let new_idx = new_vertices.len();
                    index_map[v] = new_idx;
                    new_vertices.push(mesh.vertices[v].clone());
                    new_idx
                } else {
                    index_map[v]
                };
                new_indices[i] = mapped;
            }

            new_triangles.push(MeshTriangle {
                indices: new_indices,
            });
        }

        (
            Mesh {
                vertices: new_vertices,
                triangles: new_triangles,
            },
            ClusterStats {
                removed_clusters,
                removed_triangles,
                kept_clusters,
                kept_triangles,
            },
        )
    }

    /// Smooth normals using neighbor averaging
    fn smooth_normals(&self, mut mesh: Mesh) -> Mesh {
        if mesh.triangles.is_empty() {
            return mesh;
        }

        // Build vertex to triangles mapping
        let mut vert_tris: HashMap<usize, Vec<usize>> = HashMap::new();
        for (tri_idx, tri) in mesh.triangles.iter().enumerate() {
            for &v in &tri.indices {
                vert_tris.entry(v).or_insert_with(Vec::new).push(tri_idx);
            }
        }

        for _ in 0..self.config.normal_smoothing_iterations {
            let mut new_normals = Vec::new();

            for (i, vert) in mesh.vertices.iter().enumerate() {
                if let Some(tris) = vert_tris.get(&i) {
                    let mut avg_normal = Vector3::zeros();

                    for &tri_idx in tris {
                        let tri = &mesh.triangles[tri_idx];
                        let v0 = mesh.vertices[tri.indices[0]].position;
                        let v1 = mesh.vertices[tri.indices[1]].position;
                        let v2 = mesh.vertices[tri.indices[2]].position;

                        let e1 = v1 - v0;
                        let e2 = v2 - v0;
                        avg_normal += e1.cross(&e2);
                    }

                    avg_normal = avg_normal.normalize();

                    // Blend with original
                    let smoothed = (vert.normal + avg_normal).normalize();
                    new_normals.push(smoothed);
                } else {
                    new_normals.push(vert.normal);
                }
            }

            // Apply smoothed normals
            for (i, normal) in new_normals.into_iter().enumerate() {
                mesh.vertices[i].normal = normal;
            }
        }

        mesh
    }

    fn validate_topology(&self, mesh: &Mesh) -> TopologyStats {
        let mut edges: HashMap<(usize, usize), usize> = HashMap::new();
        for tri in &mesh.triangles {
            let [a, b, c] = tri.indices;
            for (u, v) in [(a, b), (b, c), (c, a)] {
                let key = if u < v { (u, v) } else { (v, u) };
                *edges.entry(key).or_insert(0) += 1;
            }
        }

        let mut boundary_edges = 0usize;
        let mut non_manifold_edges = 0usize;
        for count in edges.values() {
            if *count == 1 {
                boundary_edges += 1;
            } else if *count > 2 {
                non_manifold_edges += 1;
            }
        }

        TopologyStats {
            boundary_edges,
            non_manifold_edges,
        }
    }

    /// Get current volume
    pub fn volume(&self) -> &TsdfVolume {
        &self.volume
    }

    /// Get mutable volume
    pub fn volume_mut(&mut self) -> &mut TsdfVolume {
        &mut self.volume
    }

    /// Clear the volume
    pub fn clear(&mut self) {
        self.volume.clear();
        self.timings_ms = MeshTimings::default();
    }

    /// Get number of integrated frames
    pub fn frame_count(&self) -> usize {
        self.volume.frame_count()
    }

    /// Get current timing metrics.
    pub fn timings(&self) -> &MeshTimings {
        &self.timings_ms
    }

    /// Reset timing metrics.
    pub fn reset_timings(&mut self) {
        self.timings_ms = MeshTimings::default();
    }
}

// Re-export for convenience
pub use crate::fusion::marching_cubes::extract_mesh_from_tsdf;

fn mesh_bounding_box(mesh: &Mesh) -> ([f32; 3], [f32; 3]) {
    if mesh.vertices.is_empty() {
        return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    }

    let mut min = mesh.vertices[0].position;
    let mut max = mesh.vertices[0].position;
    for v in &mesh.vertices[1..] {
        min = Point3::from(min.coords.inf(&v.position.coords));
        max = Point3::from(max.coords.sup(&v.position.coords));
    }

    ([min.x, min.y, min.z], [max.x, max.y, max.z])
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_filter_clusters_removes_small_components() {
        let mut mesh = Mesh::new();
        let verts = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(11.0, 0.0, 0.0),
            Point3::new(10.0, 1.0, 0.0),
        ];
        for v in verts {
            mesh.vertices.push(MeshVertex {
                position: v,
                normal: Vector3::z(),
                color: [1.0, 1.0, 1.0],
            });
        }
        mesh.triangles.push(MeshTriangle { indices: [0, 1, 2] });
        mesh.triangles.push(MeshTriangle { indices: [1, 3, 2] });
        mesh.triangles.push(MeshTriangle { indices: [6, 7, 8] });

        let extractor = MeshExtractor::new(MeshExtractionConfig {
            min_cluster_size: 2,
            num_largest_clusters: 0,
            ..Default::default()
        });

        let (filtered, stats) = extractor.filter_clusters(mesh);
        assert_eq!(filtered.triangles.len(), 2);
        assert_eq!(stats.removed_triangles, 1);
        assert_eq!(stats.removed_clusters, 1);
        assert_eq!(stats.kept_clusters, 1);
    }

    #[test]
    fn test_export_mesh_metadata_files() {
        let extractor = MeshExtractor::default();
        let mut mesh = Mesh::new();
        let verts = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 2.0, 3.0),
        ];
        for v in verts {
            mesh.vertices.push(MeshVertex {
                position: v,
                normal: Vector3::z(),
                color: [1.0, 1.0, 1.0],
            });
        }
        mesh.triangles.push(MeshTriangle { indices: [0, 1, 2] });

        let dir = tempdir().unwrap();
        let path = extractor
            .export_mesh_metadata_files(&mesh, 12.5, dir.path())
            .unwrap();
        let payload = std::fs::read_to_string(path).unwrap();
        assert!(payload.contains("\"vertex_count\": 3"));
        assert!(payload.contains("\"triangle_count\": 1"));
        assert!(payload.contains("\"isolated_triangle_percentage\": 12.5"));
        assert!(payload.contains("\"voxel_size\": 0.01"));
    }
}
