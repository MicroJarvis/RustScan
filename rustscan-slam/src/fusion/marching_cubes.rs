//! Marching Cubes Algorithm for Mesh Extraction
//!
//! Implements the Marching Cubes algorithm to extract isosurface mesh
//! from a TSDF volume.
//!
//! Based on: "Marching Cubes: A High Resolution Surface Reconstruction Algorithm"

use super::tsdf_volume::TsdfVolume;
use nalgebra::{Point3, Vector3};
use std::collections::{HashMap, HashSet};

/// A vertex in the extracted mesh
#[derive(Debug, Clone)]
pub struct MeshVertex {
    pub position: Point3<f32>,
    pub normal: Vector3<f32>,
    pub color: [f32; 3],
}

/// A triangle in the mesh
#[derive(Debug, Clone)]
pub struct MeshTriangle {
    pub indices: [usize; 3],
}

/// Extracted mesh from TSDF volume
#[derive(Debug, Default)]
pub struct Mesh {
    pub vertices: Vec<MeshVertex>,
    pub triangles: Vec<MeshTriangle>,
}

impl Mesh {
    /// Create empty mesh
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            triangles: Vec::new(),
        }
    }

    /// Get number of vertices
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Get number of triangles
    pub fn num_triangles(&self) -> usize {
        self.triangles.len()
    }

    /// Clear mesh
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.triangles.clear();
    }
}

/// Edge connection table (which edges cross the isosurface)
const EDGE_TABLE: [u16; 256] = [
    0x0, 0x109, 0x203, 0x30a, 0x80c, 0x905, 0xa0f, 0xb06, 0x406, 0x50f, 0x605, 0x70c, 0xc0a, 0xd03,
    0xe09, 0xf00, 0x190, 0x99, 0x393, 0x29a, 0x99c, 0x895, 0xb9f, 0xa96, 0x596, 0x49f, 0x795,
    0x69c, 0xd9a, 0xc93, 0xf99, 0xe90, 0x230, 0x339, 0x33, 0x13a, 0xa3c, 0xb35, 0x83f, 0x936,
    0x636, 0x73f, 0x435, 0x53c, 0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9, 0x1a3, 0xaa, 0xbac,
    0xaa5, 0x9af, 0x8a6, 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xfaa, 0xea3, 0xda9, 0xca0, 0x8c0, 0x9c9,
    0xac3, 0xbca, 0xcc, 0x1c5, 0x2cf, 0x3c6, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0x4ca, 0x5c3, 0x6c9,
    0x7c0, 0x950, 0x859, 0xb53, 0xa5a, 0x15c, 0x55, 0x35f, 0x256, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x55a, 0x453, 0x759, 0x650, 0xaf0, 0xbf9, 0x8f3, 0x9fa, 0x2fc, 0x3f5, 0xff, 0x1f6, 0xef6,
    0xfff, 0xcf5, 0xdfc, 0x6fa, 0x7f3, 0x4f9, 0x5f0, 0xb60, 0xa69, 0x963, 0x86a, 0x36c, 0x265,
    0x16f, 0x66, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x76a, 0x663, 0x569, 0x460, 0x460, 0x569, 0x663,
    0x76a, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x66, 0x16f, 0x265, 0x36c, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x1f6, 0xff, 0x3f5, 0x2fc, 0x9fa,
    0x8f3, 0xbf9, 0xaf0, 0x650, 0x759, 0x453, 0x55a, 0xe5c, 0xf55, 0xc5f, 0xd56, 0x256, 0x35f,
    0x55, 0x15c, 0xa5a, 0xb53, 0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0xfcc, 0xec5, 0xdcf,
    0xcc6, 0x3c6, 0x2cf, 0x1c5, 0xcc, 0xbca, 0xac3, 0x9c9, 0x8c0, 0xca0, 0xda9, 0xea3, 0xfaa,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0x8a6, 0x9af, 0xaa5, 0xbac, 0xaa, 0x1a3, 0x2a9, 0x3a0, 0xd30,
    0xc39, 0xf33, 0xe3a, 0x53c, 0x435, 0x73f, 0x636, 0x936, 0x83f, 0xb35, 0xa3c, 0x13a, 0x33,
    0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0x69c, 0x795, 0x49f, 0x596, 0xa96, 0xb9f, 0x895,
    0x99c, 0x29a, 0x393, 0x99, 0x190, 0xf00, 0xe09, 0xd03, 0xc0a, 0x70c, 0x605, 0x50f, 0x406,
    0xb06, 0xa0f, 0x905, 0x80c, 0x30a, 0x203, 0x109, 0x0,
];

/// Triangle connection table for all 256 cases
/// Each entry contains edge indices for triangles (0-11), -1 marks end
/// Vertex ordering uses bit layout: x=(i&1), y=(i&2)>>1, z=(i&4)>>2
const TRI_TABLE: [[i8; 16]; 256] = [
    [
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    ],
    [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 8, 1, 1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 0, 11, 11, 0, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 2, 11, 1, 0, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 1, 2, 11, 9, 1, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
    [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 3, 8, 2, 1, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 2, 9, 9, 2, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 2, 3, 8, 10, 2, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
    [11, 3, 10, 10, 3, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 0, 1, 10, 8, 0, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
    [9, 3, 0, 9, 11, 3, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    [8, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 4, 3, 3, 4, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 8, 7, 0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 4, 9, 1, 7, 4, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
    [8, 7, 4, 11, 3, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 11, 7, 4, 2, 11, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
    [0, 9, 1, 8, 7, 4, 11, 3, 2, -1, -1, -1, -1, -1, -1, -1],
    [7, 4, 11, 11, 4, 2, 2, 4, 9, 2, 9, 1, -1, -1, -1, -1],
    [4, 8, 7, 2, 1, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 4, 3, 3, 4, 0, 10, 2, 1, -1, -1, -1, -1, -1, -1, -1],
    [10, 2, 9, 9, 2, 0, 7, 4, 8, -1, -1, -1, -1, -1, -1, -1],
    [10, 2, 3, 10, 3, 4, 3, 7, 4, 9, 10, 4, -1, -1, -1, -1],
    [1, 10, 3, 3, 10, 11, 4, 8, 7, -1, -1, -1, -1, -1, -1, -1],
    [10, 11, 1, 11, 7, 4, 1, 11, 4, 1, 4, 0, -1, -1, -1, -1],
    [7, 4, 8, 9, 3, 0, 9, 11, 3, 9, 10, 11, -1, -1, -1, -1],
    [7, 4, 11, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 4, 5, 8, 0, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 5, 0, 0, 5, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [5, 8, 4, 5, 3, 8, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
    [9, 4, 5, 11, 3, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 11, 0, 0, 11, 8, 5, 9, 4, -1, -1, -1, -1, -1, -1, -1],
    [4, 5, 0, 0, 5, 1, 11, 3, 2, -1, -1, -1, -1, -1, -1, -1],
    [5, 1, 4, 1, 2, 11, 4, 1, 11, 4, 11, 8, -1, -1, -1, -1],
    [1, 10, 2, 5, 9, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 4, 5, 0, 3, 8, 2, 1, 10, -1, -1, -1, -1, -1, -1, -1],
    [2, 5, 10, 2, 4, 5, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
    [10, 2, 5, 5, 2, 4, 4, 2, 3, 4, 3, 8, -1, -1, -1, -1],
    [11, 3, 10, 10, 3, 1, 4, 5, 9, -1, -1, -1, -1, -1, -1, -1],
    [4, 5, 9, 10, 0, 1, 10, 8, 0, 10, 11, 8, -1, -1, -1, -1],
    [11, 3, 0, 11, 0, 5, 0, 4, 5, 10, 11, 5, -1, -1, -1, -1],
    [4, 5, 8, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
    [8, 7, 9, 9, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 9, 0, 3, 5, 9, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
    [7, 0, 8, 7, 1, 0, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
    [7, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [5, 9, 7, 7, 9, 8, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1],
    [2, 11, 7, 2, 7, 9, 7, 5, 9, 0, 2, 9, -1, -1, -1, -1],
    [2, 11, 3, 7, 0, 8, 7, 1, 0, 7, 5, 1, -1, -1, -1, -1],
    [2, 11, 1, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
    [8, 7, 9, 9, 7, 5, 2, 1, 10, -1, -1, -1, -1, -1, -1, -1],
    [10, 2, 1, 3, 9, 0, 3, 5, 9, 3, 7, 5, -1, -1, -1, -1],
    [7, 5, 8, 5, 10, 2, 8, 5, 2, 8, 2, 0, -1, -1, -1, -1],
    [10, 2, 5, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
    [8, 7, 5, 8, 5, 9, 11, 3, 10, 3, 1, 10, -1, -1, -1, -1],
    [5, 11, 7, 10, 11, 5, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
    [11, 5, 10, 7, 5, 11, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
    [5, 11, 7, 10, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [6, 7, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 11, 6, 3, 8, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [6, 7, 11, 0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 1, 8, 8, 1, 3, 6, 7, 11, -1, -1, -1, -1, -1, -1, -1],
    [3, 2, 7, 7, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 7, 8, 0, 6, 7, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
    [6, 7, 2, 2, 7, 3, 9, 1, 0, -1, -1, -1, -1, -1, -1, -1],
    [6, 7, 8, 6, 8, 1, 8, 9, 1, 2, 6, 1, -1, -1, -1, -1],
    [11, 6, 7, 10, 2, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 8, 0, 11, 6, 7, 10, 2, 1, -1, -1, -1, -1, -1, -1, -1],
    [0, 9, 2, 2, 9, 10, 7, 11, 6, -1, -1, -1, -1, -1, -1, -1],
    [6, 7, 11, 8, 2, 3, 8, 10, 2, 8, 9, 10, -1, -1, -1, -1],
    [7, 10, 6, 7, 1, 10, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
    [8, 0, 7, 7, 0, 6, 6, 0, 1, 6, 1, 10, -1, -1, -1, -1],
    [7, 3, 6, 3, 0, 9, 6, 3, 9, 6, 9, 10, -1, -1, -1, -1],
    [6, 7, 10, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
    [11, 6, 8, 8, 6, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [6, 3, 11, 6, 0, 3, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
    [11, 6, 8, 8, 6, 4, 1, 0, 9, -1, -1, -1, -1, -1, -1, -1],
    [1, 3, 9, 3, 11, 6, 9, 3, 6, 9, 6, 4, -1, -1, -1, -1],
    [2, 8, 3, 2, 4, 8, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
    [4, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 1, 0, 2, 8, 3, 2, 4, 8, 2, 6, 4, -1, -1, -1, -1],
    [9, 1, 4, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
    [4, 8, 6, 6, 8, 11, 1, 10, 2, -1, -1, -1, -1, -1, -1, -1],
    [1, 10, 2, 6, 3, 11, 6, 0, 3, 6, 4, 0, -1, -1, -1, -1],
    [11, 6, 4, 11, 4, 8, 10, 2, 9, 2, 0, 9, -1, -1, -1, -1],
    [10, 4, 9, 6, 4, 10, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
    [4, 8, 3, 4, 3, 10, 3, 1, 10, 6, 4, 10, -1, -1, -1, -1],
    [1, 10, 0, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
    [4, 10, 6, 9, 10, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
    [4, 10, 6, 9, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [6, 7, 11, 4, 5, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 5, 9, 7, 11, 6, 3, 8, 0, -1, -1, -1, -1, -1, -1, -1],
    [1, 0, 5, 5, 0, 4, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1],
    [11, 6, 7, 5, 8, 4, 5, 3, 8, 5, 1, 3, -1, -1, -1, -1],
    [3, 2, 7, 7, 2, 6, 9, 4, 5, -1, -1, -1, -1, -1, -1, -1],
    [5, 9, 4, 0, 7, 8, 0, 6, 7, 0, 2, 6, -1, -1, -1, -1],
    [3, 2, 6, 3, 6, 7, 1, 0, 5, 0, 4, 5, -1, -1, -1, -1],
    [6, 1, 2, 5, 1, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
    [10, 2, 1, 6, 7, 11, 4, 5, 9, -1, -1, -1, -1, -1, -1, -1],
    [0, 3, 8, 4, 5, 9, 11, 6, 7, 10, 2, 1, -1, -1, -1, -1],
    [7, 11, 6, 2, 5, 10, 2, 4, 5, 2, 0, 4, -1, -1, -1, -1],
    [8, 4, 7, 5, 10, 6, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
    [9, 4, 5, 7, 10, 6, 7, 1, 10, 7, 3, 1, -1, -1, -1, -1],
    [10, 6, 5, 7, 8, 4, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
    [4, 3, 0, 7, 3, 4, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
    [10, 6, 5, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 6, 5, 9, 11, 6, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
    [11, 6, 3, 3, 6, 0, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
    [11, 6, 5, 11, 5, 0, 5, 1, 0, 8, 11, 0, -1, -1, -1, -1],
    [11, 6, 3, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
    [9, 8, 5, 8, 3, 2, 5, 8, 2, 5, 2, 6, -1, -1, -1, -1],
    [5, 9, 6, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 5, 2, 6, 1, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 1, 10, 9, 6, 5, 9, 11, 6, 9, 8, 11, -1, -1, -1, -1],
    [9, 0, 1, 3, 11, 2, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
    [11, 0, 8, 2, 0, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
    [3, 11, 2, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 8, 3, 9, 8, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
    [6, 5, 10, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 0, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [6, 5, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 3, 8, 6, 10, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 5, 6, 9, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 8, 1, 1, 8, 9, 6, 10, 5, -1, -1, -1, -1, -1, -1, -1],
    [2, 11, 3, 6, 10, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 0, 11, 11, 0, 2, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1],
    [1, 0, 9, 2, 11, 3, 6, 10, 5, -1, -1, -1, -1, -1, -1, -1],
    [5, 6, 10, 11, 1, 2, 11, 9, 1, 11, 8, 9, -1, -1, -1, -1],
    [5, 6, 1, 1, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [5, 6, 1, 1, 6, 2, 8, 0, 3, -1, -1, -1, -1, -1, -1, -1],
    [6, 9, 5, 6, 0, 9, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
    [6, 2, 5, 2, 3, 8, 5, 2, 8, 5, 8, 9, -1, -1, -1, -1],
    [3, 6, 11, 3, 5, 6, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
    [8, 0, 1, 8, 1, 6, 1, 5, 6, 11, 8, 6, -1, -1, -1, -1],
    [11, 3, 6, 6, 3, 5, 5, 3, 0, 5, 0, 9, -1, -1, -1, -1],
    [5, 6, 9, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
    [5, 6, 10, 7, 4, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 3, 4, 4, 3, 7, 10, 5, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 6, 10, 4, 8, 7, 0, 9, 1, -1, -1, -1, -1, -1, -1, -1],
    [6, 10, 5, 1, 4, 9, 1, 7, 4, 1, 3, 7, -1, -1, -1, -1],
    [7, 4, 8, 6, 10, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1],
    [10, 5, 6, 4, 11, 7, 4, 2, 11, 4, 0, 2, -1, -1, -1, -1],
    [4, 8, 7, 6, 10, 5, 3, 2, 11, 1, 0, 9, -1, -1, -1, -1],
    [1, 2, 10, 11, 7, 6, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
    [2, 1, 6, 6, 1, 5, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1],
    [0, 3, 7, 0, 7, 4, 2, 1, 6, 1, 5, 6, -1, -1, -1, -1],
    [8, 7, 4, 6, 9, 5, 6, 0, 9, 6, 2, 0, -1, -1, -1, -1],
    [7, 2, 3, 6, 2, 7, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
    [4, 8, 7, 3, 6, 11, 3, 5, 6, 3, 1, 5, -1, -1, -1, -1],
    [5, 0, 1, 4, 0, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, 6, 11, 7, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
    [11, 7, 6, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [6, 10, 4, 4, 10, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [6, 10, 4, 4, 10, 9, 3, 8, 0, -1, -1, -1, -1, -1, -1, -1],
    [0, 10, 1, 0, 6, 10, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    [6, 10, 1, 6, 1, 8, 1, 3, 8, 4, 6, 8, -1, -1, -1, -1],
    [9, 4, 10, 10, 4, 6, 3, 2, 11, -1, -1, -1, -1, -1, -1, -1],
    [2, 11, 8, 2, 8, 0, 6, 10, 4, 10, 9, 4, -1, -1, -1, -1],
    [11, 3, 2, 0, 10, 1, 0, 6, 10, 0, 4, 6, -1, -1, -1, -1],
    [6, 8, 4, 11, 8, 6, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
    [4, 1, 9, 4, 2, 1, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
    [3, 8, 0, 4, 1, 9, 4, 2, 1, 4, 6, 2, -1, -1, -1, -1],
    [6, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 8, 2, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
    [4, 6, 9, 6, 11, 3, 9, 6, 3, 9, 3, 1, -1, -1, -1, -1],
    [8, 6, 11, 4, 6, 8, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
    [11, 3, 6, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    [8, 6, 11, 4, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 7, 6, 10, 8, 7, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
    [3, 7, 0, 7, 6, 10, 0, 7, 10, 0, 10, 9, -1, -1, -1, -1],
    [6, 10, 7, 7, 10, 8, 8, 10, 1, 8, 1, 0, -1, -1, -1, -1],
    [6, 10, 7, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
    [3, 2, 11, 10, 7, 6, 10, 8, 7, 10, 9, 8, -1, -1, -1, -1],
    [2, 9, 0, 10, 9, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 7, 6, 11, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
    [7, 6, 11, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 1, 9, 2, 9, 7, 9, 8, 7, 6, 2, 7, -1, -1, -1, -1],
    [2, 7, 6, 3, 7, 2, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
    [8, 7, 0, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
    [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 1, 9, 3, 1, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
    [11, 7, 6, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [6, 11, 7, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 11, 5, 5, 11, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 5, 11, 11, 5, 7, 0, 3, 8, -1, -1, -1, -1, -1, -1, -1],
    [7, 11, 5, 5, 11, 10, 0, 9, 1, -1, -1, -1, -1, -1, -1, -1],
    [7, 11, 10, 7, 10, 5, 3, 8, 1, 8, 9, 1, -1, -1, -1, -1],
    [5, 2, 10, 5, 3, 2, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
    [5, 7, 10, 7, 8, 0, 10, 7, 0, 10, 0, 2, -1, -1, -1, -1],
    [0, 9, 1, 5, 2, 10, 5, 3, 2, 5, 7, 3, -1, -1, -1, -1],
    [9, 7, 8, 5, 7, 9, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
    [1, 11, 2, 1, 7, 11, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
    [8, 0, 3, 1, 11, 2, 1, 7, 11, 1, 5, 7, -1, -1, -1, -1],
    [7, 11, 2, 7, 2, 9, 2, 0, 9, 5, 7, 9, -1, -1, -1, -1],
    [7, 9, 5, 8, 9, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
    [3, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 0, 7, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
    [0, 9, 3, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
    [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 5, 4, 8, 10, 5, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [0, 3, 11, 0, 11, 5, 11, 10, 5, 4, 0, 5, -1, -1, -1, -1],
    [1, 0, 9, 8, 5, 4, 8, 10, 5, 8, 11, 10, -1, -1, -1, -1],
    [10, 3, 11, 1, 3, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
    [3, 2, 8, 8, 2, 4, 4, 2, 10, 4, 10, 5, -1, -1, -1, -1],
    [10, 5, 2, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
    [5, 4, 9, 8, 3, 0, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
    [2, 10, 1, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 11, 4, 11, 2, 1, 4, 11, 1, 4, 1, 5, -1, -1, -1, -1],
    [0, 5, 4, 1, 5, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
    [0, 11, 2, 8, 11, 0, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
    [5, 4, 9, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 8, 5, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
    [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [5, 4, 9, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [5, 4, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 4, 7, 11, 9, 4, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
    [0, 3, 8, 11, 4, 7, 11, 9, 4, 11, 10, 9, -1, -1, -1, -1],
    [11, 10, 7, 10, 1, 0, 7, 10, 0, 7, 0, 4, -1, -1, -1, -1],
    [3, 10, 1, 11, 10, 3, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
    [3, 2, 10, 3, 10, 4, 10, 9, 4, 7, 3, 4, -1, -1, -1, -1],
    [9, 2, 10, 0, 2, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
    [3, 4, 7, 0, 4, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
    [7, 8, 4, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 11, 4, 4, 11, 9, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
    [1, 9, 0, 4, 7, 8, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
    [7, 11, 4, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
    [4, 7, 8, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 4, 1, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
    [7, 8, 4, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 4, 7, 0, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 8, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 3, 9, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
    [1, 0, 10, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [10, 3, 11, 1, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 2, 8, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
    [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 0, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 10, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 1, 11, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
    [11, 2, 3, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 0, 8, 2, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    ],
];

/// Cube vertex positions (in unit cube coordinates)
const CUBE_VERTICES: [[f32; 3]; 8] = [
    [0.0, 0.0, 0.0], // 0: (0,0,0)
    [1.0, 0.0, 0.0], // 1: (1,0,0)
    [0.0, 1.0, 0.0], // 2: (0,1,0)
    [1.0, 1.0, 0.0], // 3: (1,1,0)
    [0.0, 0.0, 1.0], // 4: (0,0,1)
    [1.0, 0.0, 1.0], // 5: (1,0,1)
    [0.0, 1.0, 1.0], // 6: (0,1,1)
    [1.0, 1.0, 1.0], // 7: (1,1,1)
];

/// Edge vertex indices
const EDGE_VERTS: [[usize; 2]; 12] = [
    [0, 1],
    [1, 3],
    [3, 2],
    [2, 0],
    [4, 5],
    [5, 7],
    [7, 6],
    [6, 4],
    [0, 4],
    [1, 5],
    [3, 7],
    [2, 6],
];

/// Marching Cubes for mesh extraction
pub struct MarchingCubes {
    voxel_size: f32,
    min_bound: Point3<f32>,
}

impl MarchingCubes {
    pub fn new(voxel_size: f32, min_bound: Point3<f32>) -> Self {
        Self {
            voxel_size,
            min_bound,
        }
    }

    /// Interpolate vertex position along an edge
    fn interpolate_edge(
        &self,
        v1: Point3<f32>,
        v2: Point3<f32>,
        tsdf1: f32,
        tsdf2: f32,
    ) -> Point3<f32> {
        if tsdf1.abs() < 0.00001 {
            return v1;
        }
        if tsdf2.abs() < 0.00001 {
            return v2;
        }
        if (tsdf1 - tsdf2).abs() < 0.00001 {
            return v1;
        }

        let t = tsdf1 / (tsdf1 - tsdf2);
        v1 + (v2 - v1) * t
    }

    /// Extract mesh from TSDF volume
    pub fn extract_mesh<V: Fn(i32, i32, i32) -> Option<(f32, [f32; 3])>>(
        &self,
        dims: (i32, i32, i32),
        get_voxel: V,
    ) -> Mesh {
        let mut mesh = Mesh::new();
        let (dim_x, dim_y, dim_z) = dims;

        // Iterate through all voxels
        for z in 0..dim_z - 1 {
            for y in 0..dim_y - 1 {
                for x in 0..dim_x - 1 {
                    // Get values at 8 corners of cube
                    let mut cube_values = [0.0f32; 8];
                    let mut cube_colors = [[0.0f32; 3]; 8];
                    let mut has_data = true;

                    for i in 0..8 {
                        let cx = x + (i & 1) as i32;
                        let cy = y + ((i & 2) >> 1) as i32;
                        let cz = z + ((i & 4) >> 2) as i32;

                        if let Some((tsdf, color)) = get_voxel(cx, cy, cz) {
                            cube_values[i] = tsdf;
                            cube_colors[i] = color;
                        } else {
                            has_data = false;
                            break;
                        }
                    }

                    if !has_data {
                        continue;
                    }

                    // Calculate cube index
                    let mut cube_index = 0u8;
                    for i in 0..8 {
                        if cube_values[i] < 0.0 {
                            cube_index |= 1 << i;
                        }
                    }

                    // Skip empty or full cubes
                    let edge_bits = EDGE_TABLE[cube_index as usize];
                    if edge_bits == 0 {
                        continue;
                    }

                    // Get triangle table entry
                    let tri_edges = TRI_TABLE[cube_index as usize];

                    // Interpolate vertices on edges
                    let mut edge_vertices = [Point3::origin(); 12];
                    let mut edge_colors = [[0.0f32; 3]; 12];

                    for e in 0..12 {
                        if edge_bits & (1 << e) != 0 {
                            let [a, b] = EDGE_VERTS[e];

                            // Convert to world coordinates
                            let pos_a = self.voxel_to_world(x, y, z, a);
                            let pos_b = self.voxel_to_world(x, y, z, b);

                            edge_vertices[e] =
                                self.interpolate_edge(pos_a, pos_b, cube_values[a], cube_values[b]);

                            // Interpolate color
                            let t = cube_values[a] / (cube_values[a] - cube_values[b]);
                            edge_colors[e] = [
                                cube_colors[a][0] * (1.0 - t) + cube_colors[b][0] * t,
                                cube_colors[a][1] * (1.0 - t) + cube_colors[b][1] * t,
                                cube_colors[a][2] * (1.0 - t) + cube_colors[b][2] * t,
                            ];
                        }
                    }

                    // Create triangles
                    let mut i = 0;
                    while i + 2 < 16 && tri_edges[i] >= 0 {
                        let e0 = tri_edges[i] as usize;
                        let e1 = tri_edges[i + 1] as usize;
                        let e2 = tri_edges[i + 2] as usize;

                        let v0 = edge_vertices[e0];
                        let v1 = edge_vertices[e1];
                        let v2 = edge_vertices[e2];

                        // Calculate normal
                        let edge1 = v1 - v0;
                        let edge2 = v2 - v0;
                        let normal = if edge1.norm_squared() > 0.0 && edge2.norm_squared() > 0.0 {
                            edge1.cross(&edge2).normalize()
                        } else {
                            Vector3::y()
                        };

                        // Add vertices
                        let i0 = mesh.vertices.len();
                        mesh.vertices.push(MeshVertex {
                            position: v0,
                            normal,
                            color: edge_colors[e0],
                        });
                        let i1 = mesh.vertices.len();
                        mesh.vertices.push(MeshVertex {
                            position: v1,
                            normal,
                            color: edge_colors[e1],
                        });
                        let i2 = mesh.vertices.len();
                        mesh.vertices.push(MeshVertex {
                            position: v2,
                            normal,
                            color: edge_colors[e2],
                        });

                        // Swap i1 and i2 to reverse winding order for consistent normals
                        mesh.triangles.push(MeshTriangle {
                            indices: [i0, i2, i1],
                        });

                        i += 3;
                    }
                }
            }
        }

        mesh
    }

    fn voxel_to_world(&self, x: i32, y: i32, z: i32, corner: usize) -> Point3<f32> {
        let [cx, cy, cz] = CUBE_VERTICES[corner];
        self.min_bound
            + Vector3::new(
                (x as f32 + cx) * self.voxel_size,
                (y as f32 + cy) * self.voxel_size,
                (z as f32 + cz) * self.voxel_size,
            )
    }
}

/// Simple mesh extraction from TSDF volume (simplified version)
pub fn extract_mesh_from_tsdf(volume: &TsdfVolume) -> Mesh {
    let config = volume.config();
    let dims = volume.dimensions();

    let mc = MarchingCubes::new(config.voxel_size, config.min_bound);

    // For sparse volumes, only iterate over cubes that have active voxels
    // A cube at (x, y, z) uses voxels at (x, y, z), (x+1, y, z), ..., (x+1, y+1, z+1)
    // So any active voxel contributes to up to 8 cubes
    let mut cubes_to_process: HashSet<(i32, i32, i32)> = HashSet::new();

    for (voxel_pos, voxel) in volume.voxels_iter() {
        // Only consider voxels near the surface
        if voxel.weight > 0.0 {
            let (vx, vy, vz) = voxel_pos;
            // This voxel could be any of the 8 corners of a cube
            // Add all 8 possible cube origins
            for dx in 0i32..=1 {
                for dy in 0i32..=1 {
                    for dz in 0i32..=1 {
                        let cx = vx - dx;
                        let cy = vy - dy;
                        let cz = vz - dz;
                        if cx >= 0
                            && cy >= 0
                            && cz >= 0
                            && cx < dims.0 - 1
                            && cy < dims.1 - 1
                            && cz < dims.2 - 1
                        {
                            cubes_to_process.insert((cx, cy, cz));
                        }
                    }
                }
            }
        }
    }

    let mut mesh = Mesh::new();

    // Edge vertex cache: key = (cube_x, cube_y, cube_z, edge_index), value = vertex index in mesh
    // This ensures shared vertices along cube edges
    let mut edge_vertex_cache: HashMap<(i32, i32, i32, usize), usize> = HashMap::new();

    // Process only the cubes that might contain the surface
    for (x, y, z) in cubes_to_process {
        // Get values at 8 corners of cube
        let mut cube_values = [0.0f32; 8];
        let mut cube_colors = [[0.0f32; 3]; 8];

        for i in 0..8 {
            let cx = x + (i & 1) as i32;
            let cy = y + ((i & 2) >> 1) as i32;
            let cz = z + ((i & 4) >> 2) as i32;

            if let Some(voxel) = volume.get_voxel(cx, cy, cz) {
                if voxel.weight > 0.0 {
                    cube_values[i] = voxel.tsdf;
                    cube_colors[i] = voxel.color;
                } else {
                    // Voxel exists but has no data - treat as outside
                    cube_values[i] = 1.0;
                }
            } else {
                // Voxel doesn't exist in sparse map - treat as outside (tsdf = 1.0)
                cube_values[i] = 1.0;
            }
        }

        // Calculate cube index
        let mut cube_index = 0u8;
        for i in 0..8 {
            if cube_values[i] < 0.0 {
                cube_index |= 1 << i;
            }
        }

        // Skip empty or full cubes
        let edge_bits = EDGE_TABLE[cube_index as usize];
        if edge_bits == 0 {
            continue;
        }

        // Get triangle table entry
        let tri_edges = TRI_TABLE[cube_index as usize];

        // Canonical edge keys: map each of the 12 edges to (canonical_cube_x, canonical_cube_y, canonical_cube_z, axis)
        // where axis: 0=X, 1=Y, 2=Z. This ensures adjacent cubes share the same cache key
        // for their shared edges.
        //
        // Cube vertex layout: v_i where bit0=dx, bit1=dy, bit2=dz
        //   0=(0,0,0) 1=(1,0,0) 2=(0,1,0) 3=(1,1,0)
        //   4=(0,0,1) 5=(1,0,1) 6=(0,1,1) 7=(1,1,1)
        //
        // For EDGE_VERTS = [[0,1],[1,3],[3,2],[2,0],[4,5],[5,7],[7,6],[6,4],[0,4],[1,5],[3,7],[2,6]]:
        // Each edge is an axis-aligned segment; we key it by the lower-indexed vertex's voxel coords + axis.
        const EDGE_CANONICAL: [(i32, i32, i32, usize); 12] = [
            (0, 0, 0, 0), // edge 0: v0->v1, +X from (x,y,z)
            (1, 0, 0, 1), // edge 1: v1->v3, +Y from (x+1,y,z)
            (0, 1, 0, 0), // edge 2: v3->v2, +X from (x,y+1,z)  [note: EDGE_VERTS has [3,2] but direction is -X; we flip to canonical +X at (x,y+1,z)]
            (0, 0, 0, 1), // edge 3: v2->v0, +Y from (x,y,z)    [flipped to canonical]
            (0, 0, 1, 0), // edge 4: v4->v5, +X from (x,y,z+1)
            (1, 0, 1, 1), // edge 5: v5->v7, +Y from (x+1,y,z+1)
            (0, 1, 1, 0), // edge 6: v7->v6, +X from (x,y+1,z+1)
            (0, 0, 1, 1), // edge 7: v6->v4, +Y from (x,y,z+1)  [flipped]
            (0, 0, 0, 2), // edge 8: v0->v4, +Z from (x,y,z)
            (1, 0, 0, 2), // edge 9: v1->v5, +Z from (x+1,y,z)
            (1, 1, 0, 2), // edge 10: v3->v7, +Z from (x+1,y+1,z)
            (0, 1, 0, 2), // edge 11: v2->v6, +Z from (x,y+1,z)
        ];

        // Compute or retrieve cached vertices for each edge
        let mut edge_vertex_indices: [usize; 12] = [0; 12];

        for e in 0..12 {
            if edge_bits & (1 << e) != 0 {
                // Build canonical cache key shared across adjacent cubes
                let (dx, dy, dz, axis) = EDGE_CANONICAL[e];
                let cache_key = (x + dx, y + dy, z + dz, axis);
                if let Some(&idx) = edge_vertex_cache.get(&cache_key) {
                    edge_vertex_indices[e] = idx;
                } else {
                    // Compute new vertex
                    let [a, b] = EDGE_VERTS[e];

                    // Convert to world coordinates
                    let pos_a = mc.voxel_to_world(x, y, z, a);
                    let pos_b = mc.voxel_to_world(x, y, z, b);

                    let position =
                        mc.interpolate_edge(pos_a, pos_b, cube_values[a], cube_values[b]);

                    // Interpolate color
                    let denom = cube_values[a] - cube_values[b];
                    let t = if denom.abs() > 1e-8 {
                        cube_values[a] / denom
                    } else {
                        0.5
                    };
                    let color = [
                        cube_colors[a][0] * (1.0 - t) + cube_colors[b][0] * t,
                        cube_colors[a][1] * (1.0 - t) + cube_colors[b][1] * t,
                        cube_colors[a][2] * (1.0 - t) + cube_colors[b][2] * t,
                    ];

                    // Normal will be computed later from triangle edges
                    let normal = Vector3::zeros();

                    let idx = mesh.vertices.len();
                    mesh.vertices.push(MeshVertex {
                        position,
                        normal,
                        color,
                    });

                    // Cache this vertex for this edge
                    edge_vertex_cache.insert(cache_key, idx);
                    edge_vertex_indices[e] = idx;
                }
            }
        }

        // Create triangles using cached vertex indices
        let mut i = 0;
        while i + 2 < 16 && tri_edges[i] >= 0 {
            let e0 = tri_edges[i] as usize;
            let e1 = tri_edges[i + 1] as usize;
            let e2 = tri_edges[i + 2] as usize;

            let i0 = edge_vertex_indices[e0];
            let i1 = edge_vertex_indices[e1];
            let i2 = edge_vertex_indices[e2];

            // Swap i1 and i2 to reverse winding order for consistent normals
            mesh.triangles.push(MeshTriangle {
                indices: [i0, i2, i1],
            });

            i += 3;
        }
    }

    // Compute normals from triangle faces
    let mut vertex_normals: Vec<Vector3<f32>> = vec![Vector3::zeros(); mesh.vertices.len()];
    let mut vertex_counts: Vec<u32> = vec![0; mesh.vertices.len()];

    for tri in &mesh.triangles {
        let v0 = mesh.vertices[tri.indices[0]].position;
        let v1 = mesh.vertices[tri.indices[1]].position;
        let v2 = mesh.vertices[tri.indices[2]].position;

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let normal = edge1.cross(&edge2);

        for &idx in &tri.indices {
            vertex_normals[idx] += normal;
            vertex_counts[idx] += 1;
        }
    }

    // Normalize and apply
    for (i, vertex) in mesh.vertices.iter_mut().enumerate() {
        if vertex_counts[i] > 0 {
            vertex.normal = vertex_normals[i].normalize();
        } else {
            vertex.normal = Vector3::y();
        }
    }

    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::tsdf_volume::{TsdfConfig, TsdfVolume};
    use crate::test_utils::*;

    #[test]
    fn test_marching_cubes_creation() {
        let mc = MarchingCubes::new(0.01, Point3::new(-1.0, -1.0, -1.0));
        assert_eq!(mc.voxel_size, 0.01);
    }

    #[test]
    fn test_tri_table_completeness() {
        // Verify all 256 entries exist
        assert_eq!(TRI_TABLE.len(), 256);

        // Verify each entry is properly terminated with -1
        for (i, entry) in TRI_TABLE.iter().enumerate() {
            let mut found_terminator = false;
            for &val in entry.iter() {
                if val == -1 {
                    found_terminator = true;
                    break;
                }
                // Verify vertex indices are in valid range [0, 11]
                assert!(
                    val >= 0 && val <= 11,
                    "Case {}: Invalid edge index {} (must be 0-11)",
                    i,
                    val
                );
            }
            assert!(found_terminator, "Case {}: Missing -1 terminator", i);
        }
    }

    #[test]
    fn test_edge_table_completeness() {
        // Verify all 256 entries exist
        assert_eq!(EDGE_TABLE.len(), 256);

        // Each entry should be a valid 12-bit mask (edges 0-11)
        for (i, &entry) in EDGE_TABLE.iter().enumerate() {
            assert!(
                entry <= 0xFFF,
                "Case {}: Edge mask {} exceeds 12 bits",
                i,
                entry
            );
        }
    }

    #[test]
    fn test_cube_vertices() {
        // Verify CUBE_VERTICES has 8 corners
        assert_eq!(CUBE_VERTICES.len(), 8);

        // Verify each corner is either 0.0 or 1.0
        for (i, vertex) in CUBE_VERTICES.iter().enumerate() {
            for &coord in vertex.iter() {
                assert!(
                    coord == 0.0 || coord == 1.0,
                    "Corner {}: Invalid coordinate {}",
                    i,
                    coord
                );
            }
        }

        // Verify specific corners
        assert_eq!(CUBE_VERTICES[0], [0.0, 0.0, 0.0]);
        assert_eq!(CUBE_VERTICES[7], [1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_simple_case_empty() {
        // Case 0: All vertices outside (no triangles)
        let entry = TRI_TABLE[0];
        assert_eq!(entry[0], -1, "Case 0 should have no triangles");
    }

    #[test]
    fn test_simple_case_one_corner() {
        // Case 1: One corner inside (should generate 1 triangle)
        let entry = TRI_TABLE[1];
        assert_eq!(entry[0], 0);
        assert_eq!(entry[1], 3);
        assert_eq!(entry[2], 8);
        assert_eq!(entry[3], -1);
    }

    #[test]
    fn test_sphere_extraction() {
        // Create synthetic sphere TSDF
        let tsdf = create_sphere_tsdf(Point3::origin(), 1.0, 0.1);

        // Extract mesh
        let mesh = extract_mesh_from_tsdf(&tsdf);

        // Verify mesh properties
        assert!(mesh.vertices.len() > 0, "Sphere should have vertices");
        assert!(mesh.triangles.len() > 0, "Sphere should have triangles");

        // Verify vertices are roughly on sphere surface
        let tolerance = 0.2; // Allow some tolerance due to voxelization
        for vertex in &mesh.vertices {
            let dist_from_center = vertex.position.coords.norm();
            assert!(
                (dist_from_center - 1.0).abs() < tolerance,
                "Vertex at {:?} is {} from center (expected ~1.0)",
                vertex.position,
                dist_from_center
            );
        }
    }

    #[test]
    fn test_edge_cases_empty_volume() {
        // Create empty TSDF (all values > 0, outside surface)
        let config = TsdfConfig {
            voxel_size: 0.1,
            sdf_trunc: 0.4,
            min_bound: Point3::new(-1.0, -1.0, -1.0),
            max_bound: Point3::new(1.0, 1.0, 1.0),
            max_weight: 100.0,
            integration_weight: 1.0,
        };

        let volume = TsdfVolume::new(config);
        let mesh = extract_mesh_from_tsdf(&volume);

        // Empty volume should produce no mesh
        assert_eq!(
            mesh.vertices.len(),
            0,
            "Empty volume should have no vertices"
        );
        assert_eq!(
            mesh.triangles.len(),
            0,
            "Empty volume should have no triangles"
        );
    }

    #[test]
    fn test_sparse_voxels_still_extract() {
        let config = TsdfConfig {
            voxel_size: 1.0,
            sdf_trunc: 3.0,
            min_bound: Point3::origin(),
            max_bound: Point3::new(2.0, 2.0, 2.0),
            max_weight: 1.0,
            integration_weight: 1.0,
        };
        let mut volume = TsdfVolume::new(config);
        let voxel = volume.get_voxel_mut(0, 0, 0);
        voxel.tsdf = -0.5;
        voxel.weight = 1.0;

        let mesh = extract_mesh_from_tsdf(&volume);
        assert!(
            !mesh.triangles.is_empty(),
            "Sparse voxel grid should still extract triangles"
        );
    }

    #[test]
    fn test_sparse_volume_with_few_surface_voxels() {
        // Create a large volume but only populate a small surface patch
        // This tests that sparse iteration works correctly
        let config = TsdfConfig {
            voxel_size: 0.01, // 1cm voxels
            sdf_trunc: 0.03,
            min_bound: Point3::new(-2.0, -2.0, -2.0), // 4m volume (400x400x400 = 64 million voxels if dense)
            max_bound: Point3::new(2.0, 2.0, 2.0),
            max_weight: 100.0,
            integration_weight: 1.0,
        };
        let mut volume = TsdfVolume::new(config);

        // Create a small surface patch - only ~300 voxels out of 64 million possible
        let center_voxel = 200i32; // Near world origin (0, 0, 0)
        for dx in 0..10 {
            for dy in 0..10 {
                let x = center_voxel + dx;
                let y = center_voxel + dy;

                // Create a plane surface
                let voxel = volume.get_voxel_mut(x, y, center_voxel + 5);
                voxel.tsdf = 0.0; // On surface
                voxel.weight = 1.0;

                let voxel = volume.get_voxel_mut(x, y, center_voxel + 4);
                voxel.tsdf = -0.5; // Inside
                voxel.weight = 1.0;

                let voxel = volume.get_voxel_mut(x, y, center_voxel + 6);
                voxel.tsdf = 0.5; // Outside
                voxel.weight = 1.0;
            }
        }

        // Verify we have only ~300 voxels (sparse!)
        let num_voxels = volume.num_voxels();
        assert!(
            num_voxels < 500,
            "Volume should be sparse, got {}",
            num_voxels
        );
        assert!(num_voxels > 200, "Volume should have surface voxels");

        // Extract mesh - should work with sparse iteration
        let mesh = extract_mesh_from_tsdf(&volume);

        // Should generate triangles from the surface
        assert!(
            mesh.triangles.len() > 0,
            "Sparse volume should extract triangles, got {} vertices, {} triangles",
            mesh.vertices.len(),
            mesh.triangles.len()
        );

        // All vertices should be within the volume bounds
        for vertex in &mesh.vertices {
            assert!(
                vertex.position.x >= -2.0 && vertex.position.x <= 2.0,
                "Vertex x {} should be in volume bounds",
                vertex.position.x
            );
            assert!(
                vertex.position.y >= -2.0 && vertex.position.y <= 2.0,
                "Vertex y {} should be in volume bounds",
                vertex.position.y
            );
            assert!(
                vertex.position.z >= -2.0 && vertex.position.z <= 2.0,
                "Vertex z {} should be in volume bounds",
                vertex.position.z
            );
        }
    }

    #[test]
    fn test_large_sparse_volume_performance() {
        // Test that sparse iteration handles large volumes efficiently
        // This simulates the real-world case: 200x200x200 volume with few voxels
        use std::time::Instant;

        let config = TsdfConfig {
            voxel_size: 0.01,
            sdf_trunc: 0.03,
            min_bound: Point3::new(-1.0, -1.0, -1.0),
            max_bound: Point3::new(1.0, 1.0, 1.0),
            max_weight: 100.0,
            integration_weight: 1.0,
        };
        let mut volume = TsdfVolume::new(config);

        // Add a small sphere of surface voxels (not the full 8 million!)
        let center = Point3::new(0.0, 0.0, 0.0);
        let radius = 0.3; // 30cm radius sphere
        let r_voxels = (radius / 0.01) as i32;
        let center_voxel = (1.0 / 0.01) as i32 / 2; // center at (100, 100, 100)

        for dx in -r_voxels..=r_voxels {
            for dy in -r_voxels..=r_voxels {
                for dz in -r_voxels..=r_voxels {
                    let dist = ((dx * dx + dy * dy + dz * dz) as f32).sqrt() * 0.01;
                    if (dist - radius).abs() < 0.02 {
                        // Near surface
                        let voxel = volume.get_voxel_mut(
                            center_voxel + dx,
                            center_voxel + dy,
                            center_voxel + dz,
                        );
                        voxel.tsdf = (dist - radius) / 0.03; // Normalized TSDF
                        voxel.weight = 1.0;
                    }
                }
            }
        }

        let num_voxels = volume.num_voxels();
        println!(
            "Sparse volume has {} voxels (vs 8 million full)",
            num_voxels
        );
        assert!(num_voxels < 100000, "Should have sparse voxels");

        let start = Instant::now();
        let mesh = extract_mesh_from_tsdf(&volume);
        let duration = start.elapsed();

        println!(
            "Sparse extraction: {} voxels -> {} vertices, {} triangles in {:?}",
            num_voxels,
            mesh.vertices.len(),
            mesh.triangles.len(),
            duration
        );

        assert!(
            mesh.triangles.len() > 0,
            "Should extract triangles from sparse sphere"
        );
        // Should complete in reasonable time (< 1 second for sparse)
        assert!(
            duration.as_millis() < 1000,
            "Sparse extraction should be fast"
        );
    }

    fn run_single_cube(case_index: u8) -> Mesh {
        let voxel_size = 1.0;
        let min_bound = Point3::origin();
        let mc = MarchingCubes::new(voxel_size, min_bound);

        let dims = (2, 2, 2);
        let mut cube_values = [1.0f32; 8];
        for i in 0..8 {
            if (case_index & (1 << i)) != 0 {
                cube_values[i] = -1.0;
            }
        }

        mc.extract_mesh(dims, move |x, y, z| {
            if x == 0 && y == 0 && z == 0 {
                Some((cube_values[0], [1.0, 0.0, 0.0]))
            } else if x == 1 && y == 0 && z == 0 {
                Some((cube_values[1], [0.0, 1.0, 0.0]))
            } else if x == 0 && y == 1 && z == 0 {
                Some((cube_values[2], [0.0, 0.0, 1.0]))
            } else if x == 1 && y == 1 && z == 0 {
                Some((cube_values[3], [1.0, 1.0, 0.0]))
            } else if x == 0 && y == 0 && z == 1 {
                Some((cube_values[4], [1.0, 0.0, 1.0]))
            } else if x == 1 && y == 0 && z == 1 {
                Some((cube_values[5], [0.0, 1.0, 1.0]))
            } else if x == 0 && y == 1 && z == 1 {
                Some((cube_values[6], [0.5, 0.5, 0.5]))
            } else if x == 1 && y == 1 && z == 1 {
                Some((cube_values[7], [0.25, 0.75, 0.25]))
            } else {
                None
            }
        })
    }

    #[test]
    fn test_all_cases_vertices_and_triangles_are_consistent() {
        for case_index in 0u8..=255u8 {
            let mesh = run_single_cube(case_index);
            for tri in &mesh.triangles {
                assert!(tri.indices.len() == 3);
                for &idx in &tri.indices {
                    assert!(
                        idx < mesh.vertices.len(),
                        "Triangle index out of bounds for case {}",
                        case_index
                    );
                }
            }
        }
    }

    #[test]
    fn test_case_single_triangle_exact_edges() {
        let case_index = 1u8;
        let mesh = run_single_cube(case_index);

        assert_eq!(
            mesh.triangles.len(),
            1,
            "Case 1 should generate exactly one triangle"
        );
        assert_eq!(
            mesh.vertices.len(),
            3,
            "Case 1 should have exactly three vertices"
        );

        let tri = &mesh.triangles[0];
        let v0 = mesh.vertices[tri.indices[0]].position;
        let v1 = mesh.vertices[tri.indices[1]].position;
        let v2 = mesh.vertices[tri.indices[2]].position;

        for v in [v0, v1, v2] {
            assert!(v.x >= 0.0 && v.x <= 1.0);
            assert!(v.y >= 0.0 && v.y <= 1.0);
            assert!(v.z >= 0.0 && v.z <= 1.0);
        }
    }

    #[test]
    fn test_all_vertices_within_unit_cube() {
        for case_index in 0u8..=255u8 {
            let mesh = run_single_cube(case_index);
            for vertex in &mesh.vertices {
                let p = vertex.position;
                assert!(
                    p.x >= 0.0 && p.x <= 1.0,
                    "x out of bounds for case {}: {}",
                    case_index,
                    p.x
                );
                assert!(
                    p.y >= 0.0 && p.y <= 1.0,
                    "y out of bounds for case {}: {}",
                    case_index,
                    p.y
                );
                assert!(
                    p.z >= 0.0 && p.z <= 1.0,
                    "z out of bounds for case {}: {}",
                    case_index,
                    p.z
                );
            }
        }
    }
}
