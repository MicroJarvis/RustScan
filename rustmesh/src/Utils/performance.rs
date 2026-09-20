//! # High-Performance Mesh Iteration
//!
//! This module provides zero-overhead iteration primitives for mesh data.
//! Designed to match or exceed C++ performance while maintaining safety.

#[cfg(target_arch = "wasm32")]
use std::arch::wasm32;

/// Vertex with tightly packed layout (no padding)
#[repr(C, align(16))]
pub struct PackedVertex {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Packed face (3 vertex indices)
#[repr(C, align(4))]
pub struct PackedFace {
    pub v0: u32,
    pub v1: u32,
    pub v2: u32,
}

/// High-performance vertex iterator (returns raw indices)
#[derive(Debug)]
pub struct FastVertexIter {
    current: usize,
    end: usize,
}

impl FastVertexIter {
    #[inline]
    pub fn new(n_vertices: usize) -> Self {
        Self {
            current: 0,
            end: n_vertices,
        }
    }
}

impl Iterator for FastVertexIter {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.end {
            let idx = self.current;
            self.current += 1;
            Some(idx)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.current;
        (remaining, Some(remaining))
    }
}

/// High-performance face iterator (returns raw indices)
#[derive(Debug)]
pub struct FastFaceIter {
    current: usize,
    end: usize,
}

impl FastFaceIter {
    #[inline]
    pub fn new(n_faces: usize) -> Self {
        Self {
            current: 0,
            end: n_faces,
        }
    }
}

impl Iterator for FastFaceIter {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.end {
            let idx = self.current;
            self.current += 1;
            Some(idx)
        } else {
            None
        }
    }
}

/// Chunk-based iterator for SIMD processing
pub struct ChunkedVertexIter<'a> {
    vertices: &'a [PackedVertex],
    chunk_size: usize,
    current: usize,
}

impl<'a> ChunkedVertexIter<'a> {
    #[inline]
    pub fn new(vertices: &'a [PackedVertex], chunk_size: usize) -> Self {
        Self {
            vertices,
            chunk_size,
            current: 0,
        }
    }
}

impl<'a> Iterator for ChunkedVertexIter<'a> {
    type Item = &'a [PackedVertex];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.vertices.len() {
            let end = (self.current + self.chunk_size).min(self.vertices.len());
            let chunk = &self.vertices[self.current..end];
            self.current = end;
            Some(chunk)
        } else {
            None
        }
    }
}

/// Compute centroid using SIMD (ARM NEON / x86 SSE)
#[inline]
pub fn compute_centroid_simd(vertices: &[PackedVertex]) -> (f32, f32, f32) {
    if vertices.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let len = vertices.len();

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;

        let mut acc_x = vdupq_n_f32(0.0);
        let mut acc_y = vdupq_n_f32(0.0);
        let mut acc_z = vdupq_n_f32(0.0);

        // Process 4 vertices at a time
        let chunks = len / 4;
        for i in 0..chunks {
            let offset = i * 4;

            // Load 4 x values
            let vx = vld1q_f32(
                [
                    vertices[offset].x,
                    vertices[offset + 1].x,
                    vertices[offset + 2].x,
                    vertices[offset + 3].x,
                ]
                .as_ptr(),
            );

            // Load 4 y values
            let vy = vld1q_f32(
                [
                    vertices[offset].y,
                    vertices[offset + 1].y,
                    vertices[offset + 2].y,
                    vertices[offset + 3].y,
                ]
                .as_ptr(),
            );

            // Load 4 z values
            let vz = vld1q_f32(
                [
                    vertices[offset].z,
                    vertices[offset + 1].z,
                    vertices[offset + 2].z,
                    vertices[offset + 3].z,
                ]
                .as_ptr(),
            );

            acc_x = vaddq_f32(acc_x, vx);
            acc_y = vaddq_f32(acc_y, vy);
            acc_z = vaddq_f32(acc_z, vz);
        }

        // Horizontal sum
        let sum_x = vaddvq_f32(acc_x);
        let sum_y = vaddvq_f32(acc_y);
        let sum_z = vaddvq_f32(acc_z);

        // Handle remainder
        let mut remainder_x = 0.0f32;
        let mut remainder_y = 0.0f32;
        let mut remainder_z = 0.0f32;
        for i in (chunks * 4)..len {
            remainder_x += vertices[i].x;
            remainder_y += vertices[i].y;
            remainder_z += vertices[i].z;
        }

        let count = len as f32;
        (
            (sum_x + remainder_x) / count,
            (sum_y + remainder_y) / count,
            (sum_z + remainder_z) / count,
        )
    }

    #[cfg(all(not(target_arch = "aarch64"), target_feature = "sse"))]
    unsafe {
        use std::arch::x86_64::*;

        let mut acc_x = _mm_setzero_ps();
        let mut acc_y = _mm_setzero_ps();
        let mut acc_z = _mm_setzero_ps();

        let chunks = len / 4;
        for i in 0..chunks {
            let offset = i * 4;

            let vx = _mm_set_ps(
                vertices[offset + 3].x,
                vertices[offset + 2].x,
                vertices[offset + 1].x,
                vertices[offset].x,
            );
            let vy = _mm_set_ps(
                vertices[offset + 3].y,
                vertices[offset + 2].y,
                vertices[offset + 1].y,
                vertices[offset].y,
            );
            let vz = _mm_set_ps(
                vertices[offset + 3].z,
                vertices[offset + 2].z,
                vertices[offset + 1].z,
                vertices[offset].z,
            );

            acc_x = _mm_add_ps(acc_x, vx);
            acc_y = _mm_add_ps(acc_y, vy);
            acc_z = _mm_add_ps(acc_z, vz);
        }

        // Horizontal sum using SSE3 hadd
        acc_x = _mm_hadd_ps(acc_x, acc_x);
        acc_x = _mm_hadd_ps(acc_x, acc_x);
        let sum_x = _mm_cvtss_f32(acc_x);

        acc_y = _mm_hadd_ps(acc_y, acc_y);
        acc_y = _mm_hadd_ps(acc_y, acc_y);
        let sum_y = _mm_cvtss_f32(acc_y);

        acc_z = _mm_hadd_ps(acc_z, acc_z);
        acc_z = _mm_hadd_ps(acc_z, acc_z);
        let sum_z = _mm_cvtss_f32(acc_z);

        // Handle remainder
        let mut remainder_x = 0.0f32;
        let mut remainder_y = 0.0f32;
        let mut remainder_z = 0.0f32;
        for i in (chunks * 4)..len {
            remainder_x += vertices[i].x;
            remainder_y += vertices[i].y;
            remainder_z += vertices[i].z;
        }

        let count = len as f32;
        (
            (sum_x + remainder_x) / count,
            (sum_y + remainder_y) / count,
            (sum_z + remainder_z) / count,
        )
    }

    #[cfg(all(not(target_arch = "aarch64"), not(target_feature = "sse")))]
    {
        // Scalar fallback
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_z = 0.0f32;

        for v in vertices {
            sum_x += v.x;
            sum_y += v.y;
            sum_z += v.z;
        }

        let count = len as f32;
        (sum_x / count, sum_y / count, sum_z / count)
    }
}

/// Compute bounding box using SIMD
#[inline]
pub fn compute_bbox_simd(vertices: &[PackedVertex]) -> (f32, f32, f32, f32, f32, f32) {
    if vertices.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let len = vertices.len();

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;

        // Initialize with first vertex
        let mut min_x = vdupq_n_f32(vertices[0].x);
        let mut min_y = vdupq_n_f32(vertices[0].y);
        let mut min_z = vdupq_n_f32(vertices[0].z);
        let mut max_x = min_x;
        let mut max_y = min_y;
        let mut max_z = min_z;

        let chunks = len / 4;
        for i in 0..chunks {
            let offset = i * 4;

            let vx = vld1q_f32(
                [
                    vertices[offset].x,
                    vertices[offset + 1].x,
                    vertices[offset + 2].x,
                    vertices[offset + 3].x,
                ]
                .as_ptr(),
            );
            let vy = vld1q_f32(
                [
                    vertices[offset].y,
                    vertices[offset + 1].y,
                    vertices[offset + 2].y,
                    vertices[offset + 3].y,
                ]
                .as_ptr(),
            );
            let vz = vld1q_f32(
                [
                    vertices[offset].z,
                    vertices[offset + 1].z,
                    vertices[offset + 2].z,
                    vertices[offset + 3].z,
                ]
                .as_ptr(),
            );

            min_x = vminq_f32(min_x, vx);
            min_y = vminq_f32(min_y, vy);
            min_z = vminq_f32(min_z, vz);
            max_x = vmaxq_f32(max_x, vx);
            max_y = vmaxq_f32(max_y, vy);
            max_z = vmaxq_f32(max_z, vz);
        }

        // Horizontal min/max
        let final_min_x = vminvq_f32(min_x);
        let final_min_y = vminvq_f32(min_y);
        let final_min_z = vminvq_f32(min_z);
        let final_max_x = vmaxvq_f32(max_x);
        let final_max_y = vmaxvq_f32(max_y);
        let final_max_z = vmaxvq_f32(max_z);

        // Handle remainder
        let mut rmin_x = final_min_x;
        let mut rmin_y = final_min_y;
        let mut rmin_z = final_min_z;
        let mut rmax_x = final_max_x;
        let mut rmax_y = final_max_y;
        let mut rmax_z = final_max_z;

        for i in (chunks * 4)..len {
            rmin_x = rmin_x.min(vertices[i].x);
            rmin_y = rmin_y.min(vertices[i].y);
            rmin_z = rmin_z.min(vertices[i].z);
            rmax_x = rmax_x.max(vertices[i].x);
            rmax_y = rmax_y.max(vertices[i].y);
            rmax_z = rmax_z.max(vertices[i].z);
        }

        (rmin_x, rmin_y, rmin_z, rmax_x, rmax_y, rmax_z)
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        compute_bbox(vertices)
    }
}

/// Compute bounding box in single pass
#[inline]
pub fn compute_bbox(vertices: &[PackedVertex]) -> (f32, f32, f32, f32, f32, f32) {
    if vertices.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let mut min_x = vertices[0].x;
    let mut min_y = vertices[0].y;
    let mut min_z = vertices[0].z;
    let mut max_x = vertices[0].x;
    let mut max_y = vertices[0].y;
    let mut max_z = vertices[0].z;

    for v in vertices {
        if v.x < min_x {
            min_x = v.x;
        }
        if v.x > max_x {
            max_x = v.x;
        }
        if v.y < min_y {
            min_y = v.y;
        }
        if v.y > max_y {
            max_y = v.y;
        }
        if v.z < min_z {
            min_z = v.z;
        }
        if v.z > max_z {
            max_z = v.z;
        }
    }

    (min_x, min_y, min_z, max_x, max_y, max_z)
}

/// Parallel iterator helper (rayon integration ready)
#[cfg(feature = "parallel")]
pub use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[cfg(feature = "parallel")]
pub fn parallel_vertex_sum(vertices: &[PackedVertex]) -> f32 {
    use rayon::prelude::*;
    vertices.par_iter().map(|v| v.x + v.y + v.z).sum()
}

/// Parallel vertex processing with chunking for better cache locality
#[cfg(feature = "parallel")]
pub fn parallel_vertex_process<F>(vertices: &[PackedVertex], f: F) -> Vec<f32>
where
    F: Fn(&PackedVertex) -> f32 + Sync + Send,
{
    use rayon::prelude::*;
    vertices.par_iter().map(f).collect()
}

/// Parallel face processing
#[cfg(feature = "parallel")]
pub fn parallel_face_process<F>(faces: &[PackedFace], f: F) -> Vec<f32>
where
    F: Fn(&PackedFace) -> f32 + Sync + Send,
{
    use rayon::prelude::*;
    faces.par_iter().map(f).collect()
}

// ============================================================================
// Batch Operations
// ============================================================================

/// Batch vertex position updates
///
/// More efficient than individual updates due to reduced bounds checking
#[inline]
pub fn batch_update_positions(positions: &mut [f32], indices: &[usize], new_values: &[[f32; 3]]) {
    debug_assert_eq!(indices.len(), new_values.len());

    for (&idx, &val) in indices.iter().zip(new_values.iter()) {
        let base = idx * 3;
        if base + 2 < positions.len() {
            positions[base] = val[0];
            positions[base + 1] = val[1];
            positions[base + 2] = val[2];
        }
    }
}

/// Batch gather vertex positions
#[inline]
pub fn batch_gather_positions(positions: &[f32], indices: &[usize]) -> Vec<[f32; 3]> {
    indices
        .iter()
        .map(|&idx| {
            let base = idx * 3;
            if base + 2 < positions.len() {
                [positions[base], positions[base + 1], positions[base + 2]]
            } else {
                [0.0, 0.0, 0.0]
            }
        })
        .collect()
}

/// Prefetch hint for cache optimization
#[inline(always)]
pub fn prefetch_read(data: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(data.as_ptr() as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!(
            "prfm pldl1keep, [{0}]",
            in(reg) data.as_ptr(),
        );
    }
}

// ============================================================================
// Circulator Optimizations
// ============================================================================

/// Cached vertex neighborhood for faster circulator iterations
///
/// Pre-computes the one-ring neighborhood to avoid repeated halfedge traversals
#[derive(Debug, Clone)]
pub struct CachedVertexNeighborhood {
    pub center: usize,
    pub neighbors: Vec<usize>,
    pub halfedges: Vec<usize>,
}

impl CachedVertexNeighborhood {
    pub fn new(center: usize, capacity: usize) -> Self {
        Self {
            center,
            neighbors: Vec::with_capacity(capacity),
            halfedges: Vec::with_capacity(capacity),
        }
    }

    pub fn add_neighbor(&mut self, neighbor: usize, halfedge: usize) {
        self.neighbors.push(neighbor);
        self.halfedges.push(halfedge);
    }

    pub fn len(&self) -> usize {
        self.neighbors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.neighbors.is_empty()
    }
}

/// Build cached neighborhoods for all vertices
///
/// This is useful when you need to iterate over neighborhoods multiple times
#[cfg(feature = "parallel")]
pub fn build_cached_neighborhoods(mesh: &crate::RustMesh) -> Vec<CachedVertexNeighborhood> {
    use rayon::prelude::*;

    (0..mesh.n_vertices())
        .into_par_iter()
        .map(|i| {
            let vh = crate::VertexHandle::new(i as u32);
            let mut cache = CachedVertexNeighborhood::new(i, 8);

            if let Some(vv) = mesh.vertex_vertices(vh) {
                for neighbor in vv {
                    cache.add_neighbor(neighbor.idx_usize(), 0);
                }
            }

            cache
        })
        .collect()
}

#[cfg(not(feature = "parallel"))]
pub fn build_cached_neighborhoods(mesh: &crate::RustMesh) -> Vec<CachedVertexNeighborhood> {
    (0..mesh.n_vertices())
        .map(|i| {
            let vh = crate::VertexHandle::new(i as u32);
            let mut cache = CachedVertexNeighborhood::new(i, 8);

            if let Some(vv) = mesh.vertex_vertices(vh) {
                for neighbor in vv {
                    cache.add_neighbor(neighbor.idx_usize(), 0);
                }
            }

            cache
        })
        .collect()
}

/// Benchmark results storage
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub iterations: usize,
    pub total_time_ns: f64,
    pub time_per_iter_ns: f64,
    pub throughput_mps: f64, // millions per second
}

impl BenchmarkResult {
    #[inline]
    pub fn new(name: String, iterations: usize, total_time_ns: f64) -> Self {
        let time_per_iter_ns = total_time_ns / iterations as f64;
        let throughput_mps = iterations as f64 / (total_time_ns / 1_000_000.0);
        Self {
            name,
            iterations,
            total_time_ns,
            time_per_iter_ns,
            throughput_mps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_centroid_simd() {
        let vertices: Vec<PackedVertex> = (0..100)
            .map(|i| PackedVertex {
                x: i as f32,
                y: (i * 2) as f32,
                z: (i * 3) as f32,
            })
            .collect();

        let (cx, cy, cz) = compute_centroid_simd(&vertices);

        println!("Centroid: ({}, {}, {})", cx, cy, cz);

        // Expected: average of 0..99 = 49.5, average of 0..198 = 99, average of 0..297 = 148.5
        let expected_x = (0..100).sum::<i32>() as f32 / 100.0;
        let expected_y = (0..100).map(|i| i * 2).sum::<i32>() as f32 / 100.0;
        let expected_z = (0..100).map(|i| i * 3).sum::<i32>() as f32 / 100.0;

        assert!((cx - expected_x).abs() < 0.1);
        assert!((cy - expected_y).abs() < 0.1);
        assert!((cz - expected_z).abs() < 0.1);
    }

    #[test]
    fn test_compute_bbox_simd() {
        let vertices = vec![
            PackedVertex {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            PackedVertex {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            PackedVertex {
                x: -1.0,
                y: -2.0,
                z: -3.0,
            },
            PackedVertex {
                x: 0.5,
                y: 1.0,
                z: 1.5,
            },
        ];

        let (min_x, min_y, min_z, max_x, max_y, max_z) = compute_bbox_simd(&vertices);

        println!(
            "BBox: min=({}, {}, {}), max=({}, {}, {})",
            min_x, min_y, min_z, max_x, max_y, max_z
        );

        assert!((min_x - (-1.0)).abs() < 0.01);
        assert!((min_y - (-2.0)).abs() < 0.01);
        assert!((min_z - (-3.0)).abs() < 0.01);
        assert!((max_x - 1.0).abs() < 0.01);
        assert!((max_y - 2.0).abs() < 0.01);
        assert!((max_z - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_batch_gather_positions() {
        let positions = vec![
            0.0, 0.0, 0.0, // vertex 0
            1.0, 0.0, 0.0, // vertex 1
            0.0, 1.0, 0.0, // vertex 2
        ];

        let indices = vec![0, 2];
        let gathered = batch_gather_positions(&positions, &indices);

        assert_eq!(gathered.len(), 2);
        assert_eq!(gathered[0], [0.0, 0.0, 0.0]);
        assert_eq!(gathered[1], [0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_fast_vertex_iter() {
        let iter = FastVertexIter::new(10);
        let collected: Vec<_> = iter.collect();

        assert_eq!(collected, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_cached_vertex_neighborhood() {
        let mut cache = CachedVertexNeighborhood::new(0, 4);
        cache.add_neighbor(1, 10);
        cache.add_neighbor(2, 20);

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.neighbors, vec![1, 2]);
    }
}
