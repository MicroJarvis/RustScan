//! # SoA Kernel
//!
//! Structure of Arrays (SoA) layout for SIMD-friendly mesh storage.
//! This provides better cache locality and enables efficient SIMD operations.
//!
//! Memory Layout:
//! - x: Vec<f32> - all x coordinates (contiguous)
//! - y: Vec<f32> - all y coordinates (contiguous)
//! - z: Vec<f32> - all z coordinates (contiguous)
//! - halfedge_handles: Vec<Option<HalfedgeHandle>>
//!

use crate::attrib_soa_kernel::{
    EPropHandle, FPropHandle, HPropHandle, PropValue, PropertyStore, VPropHandle,
};
use crate::handles::{EdgeHandle, FaceHandle, HalfedgeHandle, VertexHandle};
use crate::items::{Edge, Face, Halfedge};
use crate::{Point3, Vec2, Vec3, Vec4};

#[derive(Debug, Clone, Copy)]
struct EdgeLookupEntry {
    other_vertex: u32,
    halfedge: HalfedgeHandle,
}

/// SoA Kernel - SIMD-friendly mesh storage
#[derive(Debug, Clone)]
pub struct SoAKernel {
    // Vertex position data (SoA layout for SIMD)
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,

    // Vertex auxiliary data
    halfedge_handles: Vec<Option<HalfedgeHandle>>,

    // Connectivity data
    halfedges: Vec<Halfedge>,
    edges: Vec<Edge>,
    faces: Vec<Face>,

    // Edge lookup bucketed by the lower vertex index, then sorted by the higher
    // vertex index. This keeps lookups cache-friendly on triangle insertion.
    edge_lookup: Vec<Vec<EdgeLookupEntry>>,

    // Track which halfedges have had next set
    next_set: Vec<bool>,

    // ========================
    // Attribute arrays (SoA layout)
    // ========================

    // Vertex attributes
    vertex_normals: Option<Vec<Vec3>>,
    vertex_colors: Option<Vec<Vec4>>,
    vertex_texcoords: Option<Vec<Vec2>>,

    // Halfedge attributes
    halfedge_normals: Option<Vec<Vec3>>,
    halfedge_colors: Option<Vec<Vec4>>,
    halfedge_texcoords: Option<Vec<Vec2>>,

    // Edge attributes
    edge_colors: Option<Vec<Vec4>>,

    // Face attributes
    face_normals: Option<Vec<Vec3>>,
    face_colors: Option<Vec<Vec4>>,

    // Dynamic properties
    vertex_props: PropertyStore,
    halfedge_props: PropertyStore,
    edge_props: PropertyStore,
    face_props: PropertyStore,

    // Deleted-state flags used by local topology edits before garbage collection.
    vertex_deleted: Vec<bool>,
    halfedge_deleted: Vec<bool>,
    edge_deleted: Vec<bool>,
    face_deleted: Vec<bool>,
}

impl SoAKernel {
    fn resize_halfedge_attrs(&mut self) {
        let size = self.halfedges.len();
        if let Some(ref mut normals) = self.halfedge_normals {
            normals.resize(size, Vec3::zeros());
        }
        if let Some(ref mut colors) = self.halfedge_colors {
            colors.resize(size, Vec4::new(0.5, 0.5, 0.5, 1.0));
        }
        if let Some(ref mut texcoords) = self.halfedge_texcoords {
            texcoords.resize(size, Vec2::zeros());
        }
    }

    fn resize_edge_attrs(&mut self) {
        let size = self.edges.len();
        if let Some(ref mut colors) = self.edge_colors {
            colors.resize(size, Vec4::new(0.5, 0.5, 0.5, 1.0));
        }
    }

    fn resize_face_attrs(&mut self) {
        let size = self.faces.len();
        if let Some(ref mut normals) = self.face_normals {
            normals.resize(size, Vec3::zeros());
        }
        if let Some(ref mut colors) = self.face_colors {
            colors.resize(size, Vec4::new(0.8, 0.8, 0.8, 1.0));
        }
    }

    /// Create a new empty SoA kernel
    #[inline]
    pub fn new() -> Self {
        Self {
            x: Vec::new(),
            y: Vec::new(),
            z: Vec::new(),
            halfedge_handles: Vec::new(),
            halfedges: Vec::new(),
            edges: Vec::new(),
            faces: Vec::new(),
            edge_lookup: Vec::new(),
            next_set: Vec::new(),
            // Attributes
            vertex_normals: None,
            vertex_colors: None,
            vertex_texcoords: None,
            halfedge_normals: None,
            halfedge_colors: None,
            halfedge_texcoords: None,
            edge_colors: None,
            face_normals: None,
            face_colors: None,
            vertex_props: PropertyStore::default(),
            halfedge_props: PropertyStore::default(),
            edge_props: PropertyStore::default(),
            face_props: PropertyStore::default(),
            vertex_deleted: Vec::new(),
            halfedge_deleted: Vec::new(),
            edge_deleted: Vec::new(),
            face_deleted: Vec::new(),
        }
    }

    /// Clear all data
    #[inline]
    pub fn clear(&mut self) {
        self.x.clear();
        self.y.clear();
        self.z.clear();
        self.halfedge_handles.clear();
        self.halfedges.clear();
        self.edges.clear();
        self.faces.clear();
        self.edge_lookup.clear();
        self.next_set.clear();
        // Clear attributes
        self.vertex_normals = None;
        self.vertex_colors = None;
        self.vertex_texcoords = None;
        self.halfedge_normals = None;
        self.halfedge_colors = None;
        self.halfedge_texcoords = None;
        self.edge_colors = None;
        self.face_normals = None;
        self.face_colors = None;
        self.vertex_props.clear();
        self.halfedge_props.clear();
        self.edge_props.clear();
        self.face_props.clear();
        self.vertex_deleted.clear();
        self.halfedge_deleted.clear();
        self.edge_deleted.clear();
        self.face_deleted.clear();
    }

    // --- Vertex operations ---

    /// Add a new vertex and return its handle
    #[inline]
    pub fn add_vertex(&mut self, point: Point3) -> VertexHandle {
        let idx = self.x.len() as u32;
        self.x.push(point.x);
        self.y.push(point.y);
        self.z.push(point.z);
        self.halfedge_handles.push(None);
        self.edge_lookup.push(Vec::new());
        self.vertex_deleted.push(false);

        // Resize attribute arrays if they exist
        if let Some(ref mut normals) = self.vertex_normals {
            normals.push(Vec3::zeros());
        }
        if let Some(ref mut colors) = self.vertex_colors {
            colors.push(Vec4::new(1.0, 1.0, 1.0, 1.0));
        }
        if let Some(ref mut texcoords) = self.vertex_texcoords {
            texcoords.push(Vec2::zeros());
        }
        self.vertex_props.resize_all(self.x.len());

        VertexHandle::new(idx)
    }

    /// Get vertex count
    #[inline]
    pub fn n_vertices(&self) -> usize {
        self.x.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    // --- Position access (SIMD friendly) ---

    /// Get x coordinates as slice
    #[inline]
    pub fn x_slice(&self) -> &[f32] {
        &self.x
    }

    /// Get y coordinates as slice
    #[inline]
    pub fn y_slice(&self) -> &[f32] {
        &self.y
    }

    /// Get z coordinates as slice
    #[inline]
    pub fn z_slice(&self) -> &[f32] {
        &self.z
    }

    /// Get x pointer for SIMD
    #[inline]
    pub fn x_ptr(&self) -> *const f32 {
        self.x.as_ptr()
    }

    /// Get y pointer for SIMD
    #[inline]
    pub fn y_ptr(&self) -> *const f32 {
        self.y.as_ptr()
    }

    /// Get z pointer for SIMD
    #[inline]
    pub fn z_ptr(&self) -> *const f32 {
        self.z.as_ptr()
    }

    /// Get vertex position by index
    #[inline]
    pub fn point(&self, idx: usize) -> Option<Point3> {
        if idx < self.x.len() {
            Some(Point3::new(self.x[idx], self.y[idx], self.z[idx]))
        } else {
            None
        }
    }

    /// Get vertex position by index (unsafe, unchecked)
    #[inline]
    pub unsafe fn point_unchecked(&self, idx: usize) -> Point3 {
        Point3::new(self.x[idx], self.y[idx], self.z[idx])
    }

    /// Get x by index
    #[inline]
    pub fn x(&self, idx: usize) -> Option<f32> {
        self.x.get(idx).copied()
    }

    /// Get y by index
    #[inline]
    pub fn y(&self, idx: usize) -> Option<f32> {
        self.y.get(idx).copied()
    }

    /// Get z by index
    #[inline]
    pub fn z(&self, idx: usize) -> Option<f32> {
        self.z.get(idx).copied()
    }

    /// Get x by index (unchecked)
    #[inline]
    pub unsafe fn x_unchecked(&self, idx: usize) -> f32 {
        *self.x_ptr().add(idx)
    }

    /// Get y by index (unchecked)
    #[inline]
    pub unsafe fn y_unchecked(&self, idx: usize) -> f32 {
        *self.y_ptr().add(idx)
    }

    /// Get z by index (unchecked)
    #[inline]
    pub unsafe fn z_unchecked(&self, idx: usize) -> f32 {
        *self.z_ptr().add(idx)
    }

    /// Set vertex position
    #[inline]
    pub fn set_point(&mut self, idx: usize, point: Point3) {
        if idx < self.x.len() {
            self.x[idx] = point.x;
            self.y[idx] = point.y;
            self.z[idx] = point.z;
        }
    }

    // --- Halfedge operations ---

    /// Add a new edge and return the handle to the first halfedge
    #[inline]
    pub fn add_edge(&mut self, start_vh: VertexHandle, end_vh: VertexHandle) -> HalfedgeHandle {
        let (bucket_idx, other_vertex) = Self::edge_lookup_key(start_vh.idx(), end_vh.idx());

        // Check if edge already exists
        if let Some(existing_heh) = self.edge_lookup_halfedge(bucket_idx, other_vertex) {
            // Return the halfedge pointing to end_vh
            let to_v = self
                .halfedges
                .get(existing_heh.idx_usize())
                .map(|he| he.vertex_handle.idx());
            if to_v == Some(end_vh.idx()) {
                return existing_heh;
            } else {
                // Return opposite halfedge (safe because every halfedge has an opposite)
                return self
                    .opposite_halfedge_handle(existing_heh)
                    .unwrap_or(existing_heh);
            }
        }

        // Create new edge
        let edge_idx = self.edges.len() as u32;
        let he0_idx = self.halfedges.len() as u32;
        let he1_idx = he0_idx + 1;

        // Create halfedges
        let he0 = Halfedge {
            vertex_handle: end_vh,
            face_handle: None,
            next_halfedge_handle: None,
            prev_halfedge_handle: None,
            opposite_halfedge_handle: Some(HalfedgeHandle::new(he1_idx)),
            edge_idx,
        };

        let he1 = Halfedge {
            vertex_handle: start_vh,
            face_handle: None,
            next_halfedge_handle: None,
            prev_halfedge_handle: None,
            opposite_halfedge_handle: Some(HalfedgeHandle::new(he0_idx)),
            edge_idx,
        };

        let he0_handle = HalfedgeHandle::new(he0_idx);
        let he1_handle = HalfedgeHandle::new(he1_idx);
        self.edges.push(Edge::new(he0_handle, he1_handle));
        self.edge_deleted.push(false);

        self.halfedges.push(he0);
        self.halfedges.push(he1);
        self.next_set.push(false);
        self.next_set.push(false);
        self.halfedge_deleted.push(false);
        self.halfedge_deleted.push(false);
        self.resize_edge_attrs();
        self.resize_halfedge_attrs();
        self.halfedge_props.resize_all(self.halfedges.len());
        self.edge_props.resize_all(self.edges.len());

        self.insert_edge_lookup(bucket_idx, other_vertex, he0_handle);

        he0_handle
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn find_halfedge(
        &self,
        start_vh: VertexHandle,
        end_vh: VertexHandle,
    ) -> Option<HalfedgeHandle> {
        let (bucket_idx, other_vertex) = Self::edge_lookup_key(start_vh.idx(), end_vh.idx());
        let existing_heh = self.edge_lookup_halfedge(bucket_idx, other_vertex)?;
        if self.to_vertex_handle(existing_heh) == end_vh {
            Some(existing_heh)
        } else {
            self.opposite_halfedge_handle(existing_heh)
        }
    }

    /// Check if edge already exists
    #[inline]
    pub fn edge_exists(&self, v0: u32, v1: u32) -> bool {
        let (bucket_idx, other_vertex) = Self::edge_lookup_key(v0, v1);
        self.edge_lookup_halfedge(bucket_idx, other_vertex)
            .is_some()
    }

    /// Get edge count
    #[inline]
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get halfedge count
    #[inline]
    pub fn n_halfedges(&self) -> usize {
        self.halfedges.len()
    }

    /// Get halfedge by handle
    #[inline]
    pub fn halfedge(&self, heh: HalfedgeHandle) -> Option<&Halfedge> {
        self.halfedges.get(heh.idx_usize())
    }

    /// Get halfedge by handle (mutable)
    #[inline]
    pub fn halfedge_mut(&mut self, heh: HalfedgeHandle) -> Option<&mut Halfedge> {
        self.halfedges.get_mut(heh.idx_usize())
    }

    /// Get edge by handle
    #[inline]
    pub fn edge(&self, eh: EdgeHandle) -> Option<&Edge> {
        self.edges.get(eh.idx_usize())
    }

    // --- Face operations ---

    /// Add a new face
    #[inline]
    pub fn add_face(&mut self, halfedge_handle: Option<HalfedgeHandle>) -> FaceHandle {
        let idx = self.faces.len() as u32;
        self.faces.push(Face::new(halfedge_handle));
        self.face_deleted.push(false);
        self.resize_face_attrs();
        self.face_props.resize_all(self.faces.len());
        FaceHandle::new(idx)
    }

    pub fn add_vertex_property<T: PropValue>(&mut self, name: &str) -> VPropHandle<T> {
        self.vertex_props.add(name, self.n_vertices())
    }

    pub fn add_halfedge_property<T: PropValue>(&mut self, name: &str) -> HPropHandle<T> {
        self.halfedge_props.add(name, self.n_halfedges())
    }

    pub fn add_edge_property<T: PropValue>(&mut self, name: &str) -> EPropHandle<T> {
        self.edge_props.add(name, self.n_edges())
    }

    pub fn add_face_property<T: PropValue>(&mut self, name: &str) -> FPropHandle<T> {
        self.face_props.add(name, self.n_faces())
    }

    pub fn vertex_property<T: PropValue>(
        &self,
        handle: VPropHandle<T>,
        vh: VertexHandle,
    ) -> Option<T> {
        self.vertex_props.get(handle, vh.idx_usize())
    }

    pub fn halfedge_property<T: PropValue>(
        &self,
        handle: HPropHandle<T>,
        heh: HalfedgeHandle,
    ) -> Option<T> {
        self.halfedge_props.get(handle, heh.idx_usize())
    }

    pub fn edge_property<T: PropValue>(&self, handle: EPropHandle<T>, eh: EdgeHandle) -> Option<T> {
        self.edge_props.get(handle, eh.idx_usize())
    }

    pub fn face_property<T: PropValue>(&self, handle: FPropHandle<T>, fh: FaceHandle) -> Option<T> {
        self.face_props.get(handle, fh.idx_usize())
    }

    pub fn set_vertex_property<T: PropValue>(
        &mut self,
        handle: VPropHandle<T>,
        vh: VertexHandle,
        value: T,
    ) -> bool {
        self.vertex_props.set(handle, vh.idx_usize(), value)
    }

    pub fn set_halfedge_property<T: PropValue>(
        &mut self,
        handle: HPropHandle<T>,
        heh: HalfedgeHandle,
        value: T,
    ) -> bool {
        self.halfedge_props.set(handle, heh.idx_usize(), value)
    }

    pub fn set_edge_property<T: PropValue>(
        &mut self,
        handle: EPropHandle<T>,
        eh: EdgeHandle,
        value: T,
    ) -> bool {
        self.edge_props.set(handle, eh.idx_usize(), value)
    }

    pub fn set_face_property<T: PropValue>(
        &mut self,
        handle: FPropHandle<T>,
        fh: FaceHandle,
        value: T,
    ) -> bool {
        self.face_props.set(handle, fh.idx_usize(), value)
    }

    pub fn has_vertex_property<T>(&self, handle: VPropHandle<T>) -> bool {
        self.vertex_props.contains(handle)
    }

    pub fn has_halfedge_property<T>(&self, handle: HPropHandle<T>) -> bool {
        self.halfedge_props.contains(handle)
    }

    pub fn has_edge_property<T>(&self, handle: EPropHandle<T>) -> bool {
        self.edge_props.contains(handle)
    }

    pub fn has_face_property<T>(&self, handle: FPropHandle<T>) -> bool {
        self.face_props.contains(handle)
    }

    pub fn vertex_property_name<T>(&self, handle: VPropHandle<T>) -> Option<&str> {
        self.vertex_props.name(handle)
    }

    pub fn halfedge_property_name<T>(&self, handle: HPropHandle<T>) -> Option<&str> {
        self.halfedge_props.name(handle)
    }

    pub fn edge_property_name<T>(&self, handle: EPropHandle<T>) -> Option<&str> {
        self.edge_props.name(handle)
    }

    pub fn face_property_name<T>(&self, handle: FPropHandle<T>) -> Option<&str> {
        self.face_props.name(handle)
    }

    #[allow(dead_code)]
    pub(crate) fn interpolate_vertex_props_pair(
        &mut self,
        a: VertexHandle,
        b: VertexHandle,
        dst: VertexHandle,
    ) {
        self.vertex_props
            .blend2_index(a.idx_usize(), b.idx_usize(), dst.idx_usize());
    }

    #[allow(dead_code)]
    pub(crate) fn interpolate_vertex_props_triangle(
        &mut self,
        a: VertexHandle,
        b: VertexHandle,
        c: VertexHandle,
        dst: VertexHandle,
    ) {
        self.vertex_props.blend3_index(
            a.idx_usize(),
            b.idx_usize(),
            c.idx_usize(),
            dst.idx_usize(),
        );
    }

    #[allow(dead_code)]
    pub(crate) fn merge_vertex_props(&mut self, removed: VertexHandle, kept: VertexHandle) {
        self.vertex_props
            .blend2_index(removed.idx_usize(), kept.idx_usize(), kept.idx_usize());
    }

    #[allow(dead_code)]
    pub(crate) fn copy_edge_props(&mut self, from: EdgeHandle, to: EdgeHandle) {
        self.edge_props.copy_index(from.idx_usize(), to.idx_usize());
    }

    #[allow(dead_code)]
    pub(crate) fn copy_face_props(&mut self, from: FaceHandle, to: FaceHandle) {
        self.face_props.copy_index(from.idx_usize(), to.idx_usize());
    }

    /// Get face count (includes deleted faces)
    #[inline]
    pub fn n_faces(&self) -> usize {
        self.faces.len()
    }

    /// Get active (non-deleted) face count
    #[inline]
    pub fn n_active_faces(&self) -> usize {
        self.faces
            .iter()
            .zip(self.face_deleted.iter())
            .filter(|(f, deleted)| f.halfedge_handle.is_some() && !**deleted)
            .count()
    }

    /// Get face by handle
    #[inline]
    pub fn face(&self, fh: FaceHandle) -> Option<&Face> {
        self.faces.get(fh.idx_usize())
    }

    /// Get face by handle (mutable)
    #[inline]
    pub fn face_mut(&mut self, fh: FaceHandle) -> Option<&mut Face> {
        self.faces.get_mut(fh.idx_usize())
    }

    /// Get the halfedge handle associated with a face
    #[inline]
    pub fn face_halfedge_handle(&self, fh: FaceHandle) -> Option<HalfedgeHandle> {
        if self.is_face_deleted(fh) {
            return None;
        }
        self.face(fh).map(|f| f.halfedge_handle).flatten()
    }

    #[inline]
    pub fn set_face_halfedge_handle(&mut self, fh: FaceHandle, heh: Option<HalfedgeHandle>) {
        if let Some(face) = self.face_mut(fh) {
            face.halfedge_handle = heh;
        }
    }

    // --- Connectivity queries ---

    /// Get the to-vertex of a halfedge
    #[inline]
    pub fn to_vertex_handle(&self, heh: HalfedgeHandle) -> VertexHandle {
        if self.is_halfedge_deleted(heh) {
            return VertexHandle::invalid();
        }
        self.halfedge(heh)
            .map(|he| he.vertex_handle)
            .unwrap_or(VertexHandle::invalid())
    }

    /// Get the from-vertex of a halfedge (via opposite halfedge)
    #[inline]
    pub fn from_vertex_handle(&self, heh: HalfedgeHandle) -> VertexHandle {
        self.opposite_halfedge_handle(heh)
            .and_then(|opp| self.halfedge(opp))
            .map(|he| he.vertex_handle)
            .unwrap_or(VertexHandle::invalid())
    }

    /// Get the opposite halfedge
    #[inline]
    pub fn opposite_halfedge_handle(&self, heh: HalfedgeHandle) -> Option<HalfedgeHandle> {
        self.halfedge(heh)
            .and_then(|he| he.opposite_halfedge_handle)
    }

    /// Get the edge handle from a halfedge
    #[inline]
    pub fn edge_handle(&self, heh: HalfedgeHandle) -> EdgeHandle {
        EdgeHandle::new(heh.idx() >> 1)
    }

    /// Get the halfedge handle from an edge (0 or 1)
    #[inline]
    pub fn edge_halfedge_handle(&self, eh: EdgeHandle, idx: usize) -> HalfedgeHandle {
        HalfedgeHandle::new((eh.idx() << 1) + idx as u32)
    }

    /// Get the face handle from a halfedge
    #[inline]
    pub fn face_handle(&self, heh: HalfedgeHandle) -> Option<FaceHandle> {
        if self.is_halfedge_deleted(heh) {
            return None;
        }
        self.halfedge(heh).and_then(|he| he.face_handle)
    }

    #[inline]
    pub fn triangle_face_vertex_indices_and_area_normal(
        &self,
        fh: FaceHandle,
    ) -> Option<([usize; 3], Vec3)> {
        if self.is_face_deleted(fh) {
            return None;
        }

        let he0 = self.face_halfedge_handle(fh)?;
        let he1 = self.next_halfedge_handle(he0)?;
        if he1 == he0 {
            return None;
        }

        let he2 = self.next_halfedge_handle(he1)?;
        if he2 == he1 || he2 == he0 {
            return None;
        }

        if self.next_halfedge_handle(he2)? != he0 {
            return None;
        }

        let v0 = self.to_vertex_handle(he0);
        let v1 = self.to_vertex_handle(he1);
        let v2 = self.to_vertex_handle(he2);
        if !v0.is_valid()
            || !v1.is_valid()
            || !v2.is_valid()
            || self.is_vertex_deleted(v0)
            || self.is_vertex_deleted(v1)
            || self.is_vertex_deleted(v2)
        {
            return None;
        }

        let i0 = v0.idx_usize();
        let i1 = v1.idx_usize();
        let i2 = v2.idx_usize();
        if i0 >= self.x.len() || i1 >= self.x.len() || i2 >= self.x.len() {
            return None;
        }

        let ux = self.x[i1] - self.x[i0];
        let uy = self.y[i1] - self.y[i0];
        let uz = self.z[i1] - self.z[i0];
        let vx = self.x[i2] - self.x[i0];
        let vy = self.y[i2] - self.y[i0];
        let vz = self.z[i2] - self.z[i0];

        Some((
            [i0, i1, i2],
            Vec3::new(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx),
        ))
    }

    /// Set the face handle for a halfedge
    #[inline]
    pub fn set_face_handle(&mut self, heh: HalfedgeHandle, fh: FaceHandle) {
        if let Some(he) = self.halfedge_mut(heh) {
            he.face_handle = Some(fh);
        }
    }

    #[inline]
    pub fn clear_face_handle(&mut self, heh: HalfedgeHandle) {
        if let Some(he) = self.halfedge_mut(heh) {
            he.face_handle = None;
        }
    }

    /// Check if a halfedge is a boundary
    #[inline]
    pub fn is_boundary(&self, heh: HalfedgeHandle) -> bool {
        self.face_handle(heh).is_none()
    }

    /// Get the next halfedge in the cycle
    #[inline]
    pub fn next_halfedge_handle(&self, heh: HalfedgeHandle) -> Option<HalfedgeHandle> {
        self.halfedge(heh).and_then(|he| he.next_halfedge_handle)
    }

    /// Set the next halfedge in the cycle
    #[inline]
    pub fn set_next_halfedge_handle(&mut self, heh: HalfedgeHandle, next_heh: HalfedgeHandle) {
        let idx = heh.idx_usize();
        if idx >= self.next_set.len() {
            self.next_set.resize(idx + 1, false);
        }

        if let Some(he) = self.halfedge_mut(heh) {
            he.next_halfedge_handle = Some(next_heh);
        }
        if let Some(he) = self.halfedge_mut(next_heh) {
            he.prev_halfedge_handle = Some(heh);
        }
        self.next_set[idx] = true;
    }

    /// Get the previous halfedge in the cycle
    #[inline]
    pub fn prev_halfedge_handle(&self, heh: HalfedgeHandle) -> Option<HalfedgeHandle> {
        self.halfedge(heh).and_then(|he| he.prev_halfedge_handle)
    }

    /// Set the previous halfedge handle
    #[inline]
    pub fn set_prev_halfedge_handle(&mut self, heh: HalfedgeHandle, prev_heh: HalfedgeHandle) {
        if let Some(he) = self.halfedge_mut(heh) {
            he.prev_halfedge_handle = Some(prev_heh);
        }
    }

    /// Get vertex halfedge handle
    #[inline]
    pub fn halfedge_handle(&self, vh: VertexHandle) -> Option<HalfedgeHandle> {
        if self.is_vertex_deleted(vh) {
            return None;
        }
        self.halfedge_handles.get(vh.idx_usize()).copied().flatten()
    }

    /// Set vertex halfedge handle
    #[inline]
    pub fn set_halfedge_handle(&mut self, vh: VertexHandle, heh: HalfedgeHandle) {
        let idx = vh.idx_usize();
        if idx >= self.halfedge_handles.len() {
            self.halfedge_handles.resize(idx + 1, None);
        }
        self.halfedge_handles[idx] = Some(heh);
    }

    #[inline]
    pub fn clear_halfedge_handle(&mut self, vh: VertexHandle) {
        let idx = vh.idx_usize();
        if idx < self.halfedge_handles.len() {
            self.halfedge_handles[idx] = None;
        }
    }

    /// Set halfedge's to_vertex (target vertex)
    #[inline]
    pub fn set_halfedge_to_vertex(&mut self, heh: HalfedgeHandle, vh: VertexHandle) {
        let idx = heh.idx_usize();
        if idx >= self.halfedges.len() {
            return;
        }
        self.halfedges[idx].vertex_handle = vh;
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn remap_edge_lookup(
        &mut self,
        old_v0: VertexHandle,
        old_v1: VertexHandle,
        new_v0: VertexHandle,
        new_v1: VertexHandle,
        lookup_halfedge: HalfedgeHandle,
    ) {
        let (old_bucket, old_other) = Self::edge_lookup_key(old_v0.idx(), old_v1.idx());
        self.remove_edge_lookup(old_bucket, old_other);

        let (new_bucket, new_other) = Self::edge_lookup_key(new_v0.idx(), new_v1.idx());
        self.insert_edge_lookup(new_bucket, new_other, lookup_halfedge);
    }

    #[inline]
    pub fn is_vertex_deleted(&self, vh: VertexHandle) -> bool {
        self.vertex_deleted
            .get(vh.idx_usize())
            .copied()
            .unwrap_or(true)
    }

    #[inline]
    pub fn is_halfedge_deleted(&self, heh: HalfedgeHandle) -> bool {
        self.halfedge_deleted
            .get(heh.idx_usize())
            .copied()
            .unwrap_or(true)
    }

    #[inline]
    pub fn is_edge_deleted(&self, eh: EdgeHandle) -> bool {
        self.edge_deleted
            .get(eh.idx_usize())
            .copied()
            .unwrap_or(true)
    }

    #[inline]
    pub fn is_face_deleted(&self, fh: FaceHandle) -> bool {
        self.face_deleted
            .get(fh.idx_usize())
            .copied()
            .unwrap_or(true)
    }

    #[inline]
    pub fn mark_vertex_deleted(&mut self, vh: VertexHandle, deleted: bool) {
        let idx = vh.idx_usize();
        if idx < self.vertex_deleted.len() {
            self.vertex_deleted[idx] = deleted;
        }
        if deleted {
            self.clear_halfedge_handle(vh);
        }
    }

    #[inline]
    pub fn mark_halfedge_deleted(&mut self, heh: HalfedgeHandle, deleted: bool) {
        let idx = heh.idx_usize();
        if idx < self.halfedge_deleted.len() {
            self.halfedge_deleted[idx] = deleted;
        }
        if deleted {
            self.clear_face_handle(heh);
        }
    }

    #[inline]
    pub fn mark_edge_deleted(&mut self, eh: EdgeHandle, deleted: bool) {
        let idx = eh.idx_usize();
        if idx < self.edge_deleted.len() {
            self.edge_deleted[idx] = deleted;
        }
    }

    #[inline]
    pub fn mark_face_deleted(&mut self, fh: FaceHandle, deleted: bool) {
        let idx = fh.idx_usize();
        if idx < self.face_deleted.len() {
            self.face_deleted[idx] = deleted;
        }
        if deleted {
            self.set_face_halfedge_handle(fh, None);
        }
    }

    /// Mark a vertex as deleted
    #[inline]
    pub fn delete_vertex(&mut self, vh: VertexHandle) {
        self.mark_vertex_deleted(vh, true);
    }

    /// Mark a face as deleted
    #[inline]
    pub fn delete_face(&mut self, fh: FaceHandle) {
        let idx = fh.idx_usize();
        if idx < self.faces.len() {
            if let Some(start_heh) = self.faces[idx].halfedge_handle {
                let n_halfedges = self.halfedges.len();
                let mut visited = vec![false; n_halfedges];
                let mut current = start_heh;
                loop {
                    let curr_idx = current.idx_usize();
                    if curr_idx >= n_halfedges || visited[curr_idx] {
                        break;
                    }
                    visited[curr_idx] = true;

                    // IMPORTANT: Save next handle BEFORE clearing it
                    let next_handle = self.next_halfedge_handle(current);

                    if let Some(he) = self.halfedge_mut(current) {
                        he.face_handle = None;
                    }

                    // Move to next before clearing next pointer
                    match next_handle {
                        Some(next) if next.is_valid() => current = next,
                        _ => break,
                    }
                }
            }
            self.faces[idx].halfedge_handle = None;
            self.face_deleted[idx] = true;
        }
    }

    /// Mark an edge as deleted
    #[inline]
    pub fn delete_edge(&mut self, eh: EdgeHandle) {
        let idx = eh.idx_usize();
        if idx < self.edges.len() {
            // Get the two halfedges before marking as invalid
            let he0 = self.edges[idx].halfedges[0];
            let he1 = self.edges[idx].halfedges[1];
            let to0 = self
                .halfedges
                .get(he0.idx_usize())
                .map(|he| he.vertex_handle.idx());
            let to1 = self
                .halfedges
                .get(he1.idx_usize())
                .map(|he| he.vertex_handle.idx());

            if let (Some(v0), Some(v1)) = (to0, to1) {
                let (bucket_idx, other_vertex) = Self::edge_lookup_key(v0, v1);
                self.remove_edge_lookup(bucket_idx, other_vertex);
            }
            self.edge_deleted[idx] = true;
            self.mark_halfedge_deleted(he0, true);
            self.mark_halfedge_deleted(he1, true);

            // Mark halfedges as invalid
            self.edges[idx].halfedges =
                [HalfedgeHandle::new(u32::MAX), HalfedgeHandle::new(u32::MAX)];
        }
    }
}

impl SoAKernel {
    #[inline]
    fn edge_lookup_key(v0: u32, v1: u32) -> (usize, u32) {
        let min_v = v0.min(v1);
        let max_v = v0.max(v1);
        (min_v as usize, max_v)
    }

    #[inline]
    fn edge_lookup_halfedge(&self, bucket_idx: usize, other_vertex: u32) -> Option<HalfedgeHandle> {
        let bucket = self.edge_lookup.get(bucket_idx)?;
        let pos = bucket
            .binary_search_by_key(&other_vertex, |entry| entry.other_vertex)
            .ok()?;
        Some(bucket[pos].halfedge)
    }

    #[inline]
    fn insert_edge_lookup(
        &mut self,
        bucket_idx: usize,
        other_vertex: u32,
        halfedge: HalfedgeHandle,
    ) {
        let bucket = &mut self.edge_lookup[bucket_idx];
        let insert_at = match bucket.binary_search_by_key(&other_vertex, |entry| entry.other_vertex)
        {
            Ok(pos) | Err(pos) => pos,
        };
        bucket.insert(
            insert_at,
            EdgeLookupEntry {
                other_vertex,
                halfedge,
            },
        );
    }

    #[inline]
    fn remove_edge_lookup(&mut self, bucket_idx: usize, other_vertex: u32) {
        let Some(bucket) = self.edge_lookup.get_mut(bucket_idx) else {
            return;
        };
        if let Ok(pos) = bucket.binary_search_by_key(&other_vertex, |entry| entry.other_vertex) {
            bucket.remove(pos);
        }
    }
}

// ============================================================================
// SIMD helper functions
// ============================================================================

impl SoAKernel {
    /// Compute bounding box using SIMD-friendly access
    pub fn bounding_box(&self) -> (f32, f32, f32, f32, f32, f32) {
        let n = self.x.len();
        if n == 0 {
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        let mut min_x = self.x[0];
        let mut max_x = self.x[0];
        let mut min_y = self.y[0];
        let mut max_y = self.y[0];
        let mut min_z = self.z[0];
        let mut max_z = self.z[0];

        for i in 0..n {
            let vx = self.x[i];
            let vy = self.y[i];
            let vz = self.z[i];

            if vx < min_x {
                min_x = vx;
            }
            if vx > max_x {
                max_x = vx;
            }
            if vy < min_y {
                min_y = vy;
            }
            if vy > max_y {
                max_y = vy;
            }
            if vz < min_z {
                min_z = vz;
            }
            if vz > max_z {
                max_z = vz;
            }
        }

        (min_x, max_x, min_y, max_y, min_z, max_z)
    }

    /// Compute centroid
    pub fn centroid(&self) -> (f32, f32, f32) {
        let n = self.n_vertices();
        if n == 0 {
            return (0.0, 0.0, 0.0);
        }

        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_z = 0.0f32;

        for i in 0..n {
            sum_x += self.x[i];
            sum_y += self.y[i];
            sum_z += self.z[i];
        }

        (sum_x / n as f32, sum_y / n as f32, sum_z / n as f32)
    }

    /// Compute surface area of triangular faces
    pub fn surface_area(&self, faces: &[u32]) -> f32 {
        let n_faces = faces.len() / 3;
        let mut area = 0.0f32;

        for i in 0..n_faces {
            let i0 = faces[i * 3] as usize;
            let i1 = faces[i * 3 + 1] as usize;
            let i2 = faces[i * 3 + 2] as usize;

            let ax = self.x[i0];
            let ay = self.y[i0];
            let az = self.z[i0];
            let bx = self.x[i1];
            let by = self.y[i1];
            let bz = self.z[i1];
            let cx = self.x[i2];
            let cy = self.y[i2];
            let cz = self.z[i2];

            let bax = bx - ax;
            let bay = by - ay;
            let baz = bz - az;
            let cax = cx - ax;
            let cay = cy - ay;
            let caz = cz - az;

            let cx1 = bay * caz - baz * cay;
            let cy1 = baz * cax - bax * caz;
            let cz1 = bax * cay - bay * cax;

            area += 0.5 * (cx1 * cx1 + cy1 * cy1 + cz1 * cz1).sqrt();
        }

        area
    }

    // =========================================================================
    // SIMD-optimized operations
    // =========================================================================

    /// Compute bounding box using NEON SIMD
    #[inline]
    pub unsafe fn bounding_box_simd(&self) -> (f32, f32, f32, f32, f32, f32) {
        let n = self.x.len();
        if n == 0 {
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        let mut i = 0;
        let n_simd = (n / 4) * 4;

        let ptr_x = self.x_ptr();
        let ptr_y = self.y_ptr();
        let ptr_z = self.z_ptr();

        // Initialize with first element
        let mut min_x = self.x[0];
        let mut max_x = self.x[0];
        let mut min_y = self.y[0];
        let mut max_y = self.y[0];
        let mut min_z = self.z[0];
        let mut max_z = self.z[0];

        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;

            let mut acc_min_x = vdupq_n_f32(min_x);
            let mut acc_max_x = vdupq_n_f32(max_x);
            let mut acc_min_y = vdupq_n_f32(min_y);
            let mut acc_max_y = vdupq_n_f32(max_y);
            let mut acc_min_z = vdupq_n_f32(min_z);
            let mut acc_max_z = vdupq_n_f32(max_z);

            while i < n_simd {
                let vx = vld1q_f32(ptr_x.add(i));
                let vy = vld1q_f32(ptr_y.add(i));
                let vz = vld1q_f32(ptr_z.add(i));

                acc_min_x = vminq_f32(acc_min_x, vx);
                acc_max_x = vmaxq_f32(acc_max_x, vx);
                acc_min_y = vminq_f32(acc_min_y, vy);
                acc_max_y = vmaxq_f32(acc_max_y, vy);
                acc_min_z = vminq_f32(acc_min_z, vz);
                acc_max_z = vmaxq_f32(acc_max_z, vz);

                i += 4;
            }

            min_x = vminvq_f32(acc_min_x);
            max_x = vmaxvq_f32(acc_max_x);
            min_y = vminvq_f32(acc_min_y);
            max_y = vmaxvq_f32(acc_max_y);
            min_z = vminvq_f32(acc_min_z);
            max_z = vmaxvq_f32(acc_max_z);
        }

        // Handle remaining elements
        while i < n {
            let vx = *ptr_x.add(i);
            let vy = *ptr_y.add(i);
            let vz = *ptr_z.add(i);

            if vx < min_x {
                min_x = vx;
            }
            if vx > max_x {
                max_x = vx;
            }
            if vy < min_y {
                min_y = vy;
            }
            if vy > max_y {
                max_y = vy;
            }
            if vz < min_z {
                min_z = vz;
            }
            if vz > max_z {
                max_z = vz;
            }

            i += 1;
        }

        (min_x, max_x, min_y, max_y, min_z, max_z)
    }

    /// Compute centroid using NEON SIMD
    #[inline]
    pub unsafe fn centroid_simd(&self) -> (f32, f32, f32) {
        let n = self.n_vertices();
        if n == 0 {
            return (0.0, 0.0, 0.0);
        }

        let ptr_x = self.x_ptr();
        let ptr_y = self.y_ptr();
        let ptr_z = self.z_ptr();

        let mut i = 0;
        let n_simd = (n / 4) * 4;

        #[cfg(target_arch = "aarch64")]
        let (mut sum_x, mut sum_y, mut sum_z) = unsafe {
            use std::arch::aarch64::*;

            let mut acc_x = vdupq_n_f32(0.0);
            let mut acc_y = vdupq_n_f32(0.0);
            let mut acc_z = vdupq_n_f32(0.0);

            while i < n_simd {
                let vx = vld1q_f32(ptr_x.add(i));
                let vy = vld1q_f32(ptr_y.add(i));
                let vz = vld1q_f32(ptr_z.add(i));
                acc_x = vaddq_f32(acc_x, vx);
                acc_y = vaddq_f32(acc_y, vy);
                acc_z = vaddq_f32(acc_z, vz);
                i += 4;
            }

            (vaddvq_f32(acc_x), vaddvq_f32(acc_y), vaddvq_f32(acc_z))
        };

        #[cfg(not(target_arch = "aarch64"))]
        let (mut sum_x, mut sum_y, mut sum_z) = (0.0f32, 0.0f32, 0.0f32);

        // Handle remaining elements
        while i < n {
            sum_x += *ptr_x.add(i);
            sum_y += *ptr_y.add(i);
            sum_z += *ptr_z.add(i);
            i += 1;
        }

        (sum_x / n as f32, sum_y / n as f32, sum_z / n as f32)
    }

    /// Compute vertex sum using NEON SIMD
    #[inline]
    pub unsafe fn vertex_sum_simd(&self) -> (f32, f32, f32) {
        self.centroid_simd()
    }

    // =========================================================================
    // Attribute management
    // =========================================================================

    // --- Vertex attributes ---

    /// Request vertex normals
    pub fn request_vertex_normals(&mut self) {
        if self.vertex_normals.is_none() {
            let size = self.x.len();
            self.vertex_normals = Some(vec![Vec3::zeros(); size]);
        }
    }

    /// Check if vertex normals are available
    pub fn has_vertex_normals(&self) -> bool {
        self.vertex_normals.is_some()
    }

    /// Get vertex normal
    pub fn vertex_normal(&self, vh: VertexHandle) -> Option<Vec3> {
        self.vertex_normals
            .as_ref()
            .and_then(|n| n.get(vh.idx_usize()).copied())
    }

    /// Set vertex normal
    pub fn set_vertex_normal(&mut self, vh: VertexHandle, n: Vec3) {
        if let Some(ref mut normals) = self.vertex_normals {
            if let Some(normal) = normals.get_mut(vh.idx_usize()) {
                *normal = n;
            }
        }
    }

    #[inline]
    pub fn vertex_normals_mut_ptr(&mut self) -> Option<*mut Vec3> {
        self.vertex_normals.as_mut().map(Vec::as_mut_ptr)
    }

    /// Request vertex colors
    pub fn request_vertex_colors(&mut self) {
        if self.vertex_colors.is_none() {
            let size = self.x.len();
            self.vertex_colors = Some(vec![Vec4::new(1.0, 1.0, 1.0, 1.0); size]);
        }
    }

    /// Check if vertex colors are available
    pub fn has_vertex_colors(&self) -> bool {
        self.vertex_colors.is_some()
    }

    /// Get vertex color
    pub fn vertex_color(&self, vh: VertexHandle) -> Option<Vec4> {
        self.vertex_colors
            .as_ref()
            .and_then(|c| c.get(vh.idx_usize()).copied())
    }

    /// Set vertex color
    pub fn set_vertex_color(&mut self, vh: VertexHandle, c: Vec4) {
        if let Some(ref mut colors) = self.vertex_colors {
            if let Some(color) = colors.get_mut(vh.idx_usize()) {
                *color = c;
            }
        }
    }

    /// Request vertex texture coordinates
    pub fn request_vertex_texcoords(&mut self) {
        if self.vertex_texcoords.is_none() {
            let size = self.x.len();
            self.vertex_texcoords = Some(vec![Vec2::zeros(); size]);
        }
    }

    /// Check if vertex texcoords are available
    pub fn has_vertex_texcoords(&self) -> bool {
        self.vertex_texcoords.is_some()
    }

    /// Get vertex texcoord
    pub fn vertex_texcoord(&self, vh: VertexHandle) -> Option<Vec2> {
        self.vertex_texcoords
            .as_ref()
            .and_then(|t| t.get(vh.idx_usize()).copied())
    }

    /// Set vertex texcoord
    pub fn set_vertex_texcoord(&mut self, vh: VertexHandle, t: Vec2) {
        if let Some(ref mut texcoords) = self.vertex_texcoords {
            if let Some(texcoord) = texcoords.get_mut(vh.idx_usize()) {
                *texcoord = t;
            }
        }
    }

    // --- Halfedge attributes ---

    /// Request halfedge normals
    pub fn request_halfedge_normals(&mut self) {
        if self.halfedge_normals.is_none() {
            let size = self.halfedges.len();
            self.halfedge_normals = Some(vec![Vec3::zeros(); size]);
        }
    }

    /// Check if halfedge normals are available
    pub fn has_halfedge_normals(&self) -> bool {
        self.halfedge_normals.is_some()
    }

    /// Get halfedge normal
    pub fn halfedge_normal(&self, heh: HalfedgeHandle) -> Option<Vec3> {
        self.halfedge_normals
            .as_ref()
            .and_then(|n| n.get(heh.idx_usize()).copied())
    }

    /// Set halfedge normal
    pub fn set_halfedge_normal(&mut self, heh: HalfedgeHandle, n: Vec3) {
        if let Some(ref mut normals) = self.halfedge_normals {
            if let Some(normal) = normals.get_mut(heh.idx_usize()) {
                *normal = n;
            }
        }
    }

    /// Request halfedge colors
    pub fn request_halfedge_colors(&mut self) {
        if self.halfedge_colors.is_none() {
            let size = self.halfedges.len();
            self.halfedge_colors = Some(vec![Vec4::new(0.5, 0.5, 0.5, 1.0); size]);
        }
    }

    /// Check if halfedge colors are available
    pub fn has_halfedge_colors(&self) -> bool {
        self.halfedge_colors.is_some()
    }

    /// Get halfedge color
    pub fn halfedge_color(&self, heh: HalfedgeHandle) -> Option<Vec4> {
        self.halfedge_colors
            .as_ref()
            .and_then(|c| c.get(heh.idx_usize()).copied())
    }

    /// Set halfedge color
    pub fn set_halfedge_color(&mut self, heh: HalfedgeHandle, c: Vec4) {
        if let Some(ref mut colors) = self.halfedge_colors {
            if let Some(color) = colors.get_mut(heh.idx_usize()) {
                *color = c;
            }
        }
    }

    /// Request halfedge texcoords
    pub fn request_halfedge_texcoords(&mut self) {
        if self.halfedge_texcoords.is_none() {
            let size = self.halfedges.len();
            self.halfedge_texcoords = Some(vec![Vec2::zeros(); size]);
        }
    }

    /// Check if halfedge texcoords are available
    pub fn has_halfedge_texcoords(&self) -> bool {
        self.halfedge_texcoords.is_some()
    }

    /// Get halfedge texcoord
    pub fn halfedge_texcoord(&self, heh: HalfedgeHandle) -> Option<Vec2> {
        self.halfedge_texcoords
            .as_ref()
            .and_then(|t| t.get(heh.idx_usize()).copied())
    }

    /// Set halfedge texcoord
    pub fn set_halfedge_texcoord(&mut self, heh: HalfedgeHandle, t: Vec2) {
        if let Some(ref mut texcoords) = self.halfedge_texcoords {
            if let Some(texcoord) = texcoords.get_mut(heh.idx_usize()) {
                *texcoord = t;
            }
        }
    }

    // --- Edge attributes ---

    /// Request edge colors
    pub fn request_edge_colors(&mut self) {
        if self.edge_colors.is_none() {
            let size = self.edges.len();
            self.edge_colors = Some(vec![Vec4::new(0.5, 0.5, 0.5, 1.0); size]);
        }
    }

    /// Check if edge colors are available
    pub fn has_edge_colors(&self) -> bool {
        self.edge_colors.is_some()
    }

    /// Get edge color
    pub fn edge_color(&self, eh: EdgeHandle) -> Option<Vec4> {
        self.edge_colors
            .as_ref()
            .and_then(|c| c.get(eh.idx_usize()).copied())
    }

    /// Set edge color
    pub fn set_edge_color(&mut self, eh: EdgeHandle, c: Vec4) {
        if let Some(ref mut colors) = self.edge_colors {
            if let Some(color) = colors.get_mut(eh.idx_usize()) {
                *color = c;
            }
        }
    }

    // --- Face attributes ---

    /// Request face normals
    pub fn request_face_normals(&mut self) {
        if self.face_normals.is_none() {
            let size = self.faces.len();
            self.face_normals = Some(vec![Vec3::zeros(); size]);
        }
    }

    /// Check if face normals are available
    pub fn has_face_normals(&self) -> bool {
        self.face_normals.is_some()
    }

    /// Get face normal
    pub fn face_normal(&self, fh: FaceHandle) -> Option<Vec3> {
        self.face_normals
            .as_ref()
            .and_then(|n| n.get(fh.idx_usize()).copied())
    }

    /// Set face normal
    pub fn set_face_normal(&mut self, fh: FaceHandle, n: Vec3) {
        if let Some(ref mut normals) = self.face_normals {
            if let Some(normal) = normals.get_mut(fh.idx_usize()) {
                *normal = n;
            }
        }
    }

    #[inline]
    pub fn face_normals_mut_ptr(&mut self) -> Option<*mut Vec3> {
        self.face_normals.as_mut().map(Vec::as_mut_ptr)
    }

    /// Request face colors
    pub fn request_face_colors(&mut self) {
        if self.face_colors.is_none() {
            let size = self.faces.len();
            self.face_colors = Some(vec![Vec4::new(0.8, 0.8, 0.8, 1.0); size]);
        }
    }

    /// Check if face colors are available
    pub fn has_face_colors(&self) -> bool {
        self.face_colors.is_some()
    }

    /// Get face color
    pub fn face_color(&self, fh: FaceHandle) -> Option<Vec4> {
        self.face_colors
            .as_ref()
            .and_then(|c| c.get(fh.idx_usize()).copied())
    }

    /// Set face color
    pub fn set_face_color(&mut self, fh: FaceHandle, c: Vec4) {
        if let Some(ref mut colors) = self.face_colors {
            if let Some(color) = colors.get_mut(fh.idx_usize()) {
                *color = c;
            }
        }
    }
}
