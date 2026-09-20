//! # AttribSoAKernel
//!
//! Unified kernel combining SoA layout with dynamic attribute system.
//! Provides SIMD-friendly storage with flexible property support.

use crate::handles::{EdgeHandle, FaceHandle, HalfedgeHandle, VertexHandle};
use crate::items::{Edge, Face, Halfedge};
use crate::{Point3, Vec2, Vec3, Vec4};

use std::collections::HashMap;
use std::marker::PhantomData;

/// Attribute type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttributeType {
    VertexNormal,
    VertexColor,
    VertexTexCoord,
    HalfedgeNormal,
    HalfedgeColor,
    HalfedgeTexCoord,
    EdgeColor,
    FaceNormal,
    FaceColor,
    Custom(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PropertyHandle<T, Domain> {
    id: u32,
    _marker: PhantomData<fn() -> (T, Domain)>,
}

impl<T, Domain> PropertyHandle<T, Domain> {
    fn new(id: u32) -> Self {
        Self {
            id,
            _marker: PhantomData,
        }
    }

    fn id(self) -> u32 {
        self.id
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexPropertyTag {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgePropertyTag {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FacePropertyTag {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HalfedgePropertyTag {}

pub type VPropHandle<T> = PropertyHandle<T, VertexPropertyTag>;
pub type EPropHandle<T> = PropertyHandle<T, EdgePropertyTag>;
pub type FPropHandle<T> = PropertyHandle<T, FacePropertyTag>;
pub type HPropHandle<T> = PropertyHandle<T, HalfedgePropertyTag>;
pub type PropHandle<T> = VPropHandle<T>;

/// Dynamic property storage
#[derive(Debug, Clone)]
pub enum DynamicProperty {
    Float(Vec<f32>),
    Vec2(Vec<Vec2>),
    Vec3(Vec<Vec3>),
    Vec4(Vec<Vec4>),
    Int(Vec<i32>),
}

#[allow(dead_code)]
impl DynamicProperty {
    fn resize(&mut self, size: usize) {
        match self {
            DynamicProperty::Float(v) => v.resize(size, 0.0),
            DynamicProperty::Vec2(v) => v.resize(size, Vec2::zeros()),
            DynamicProperty::Vec3(v) => v.resize(size, Vec3::zeros()),
            DynamicProperty::Vec4(v) => v.resize(size, Vec4::zeros()),
            DynamicProperty::Int(v) => v.resize(size, 0),
        }
    }

    fn copy_index(&mut self, from: usize, to: usize) {
        match self {
            DynamicProperty::Float(values) => {
                if let (Some(src), Some(dst)) = (values.get(from).copied(), values.get_mut(to)) {
                    *dst = src;
                }
            }
            DynamicProperty::Vec2(values) => {
                if let (Some(src), Some(dst)) = (values.get(from).copied(), values.get_mut(to)) {
                    *dst = src;
                }
            }
            DynamicProperty::Vec3(values) => {
                if let (Some(src), Some(dst)) = (values.get(from).copied(), values.get_mut(to)) {
                    *dst = src;
                }
            }
            DynamicProperty::Vec4(values) => {
                if let (Some(src), Some(dst)) = (values.get(from).copied(), values.get_mut(to)) {
                    *dst = src;
                }
            }
            DynamicProperty::Int(values) => {
                if let (Some(src), Some(dst)) = (values.get(from).copied(), values.get_mut(to)) {
                    *dst = src;
                }
            }
        }
    }

    fn blend2_index(&mut self, a: usize, b: usize, to: usize) {
        match self {
            DynamicProperty::Float(values) => {
                if let (Some(va), Some(vb), Some(dst)) = (
                    values.get(a).copied(),
                    values.get(b).copied(),
                    values.get_mut(to),
                ) {
                    *dst = (va + vb) * 0.5;
                }
            }
            DynamicProperty::Vec2(values) => {
                if let (Some(va), Some(vb), Some(dst)) = (
                    values.get(a).copied(),
                    values.get(b).copied(),
                    values.get_mut(to),
                ) {
                    *dst = (va + vb) * 0.5;
                }
            }
            DynamicProperty::Vec3(values) => {
                if let (Some(va), Some(vb), Some(dst)) = (
                    values.get(a).copied(),
                    values.get(b).copied(),
                    values.get_mut(to),
                ) {
                    *dst = (va + vb) * 0.5;
                }
            }
            DynamicProperty::Vec4(values) => {
                if let (Some(va), Some(vb), Some(dst)) = (
                    values.get(a).copied(),
                    values.get(b).copied(),
                    values.get_mut(to),
                ) {
                    *dst = (va + vb) * 0.5;
                }
            }
            DynamicProperty::Int(values) => {
                if let (Some(va), Some(vb), Some(dst)) = (
                    values.get(a).copied(),
                    values.get(b).copied(),
                    values.get_mut(to),
                ) {
                    *dst = ((va as i64 + vb as i64) / 2) as i32;
                }
            }
        }
    }

    fn blend3_index(&mut self, a: usize, b: usize, c: usize, to: usize) {
        match self {
            DynamicProperty::Float(values) => {
                if let (Some(va), Some(vb), Some(vc), Some(dst)) = (
                    values.get(a).copied(),
                    values.get(b).copied(),
                    values.get(c).copied(),
                    values.get_mut(to),
                ) {
                    *dst = (va + vb + vc) / 3.0;
                }
            }
            DynamicProperty::Vec2(values) => {
                if let (Some(va), Some(vb), Some(vc), Some(dst)) = (
                    values.get(a).copied(),
                    values.get(b).copied(),
                    values.get(c).copied(),
                    values.get_mut(to),
                ) {
                    *dst = (va + vb + vc) / 3.0;
                }
            }
            DynamicProperty::Vec3(values) => {
                if let (Some(va), Some(vb), Some(vc), Some(dst)) = (
                    values.get(a).copied(),
                    values.get(b).copied(),
                    values.get(c).copied(),
                    values.get_mut(to),
                ) {
                    *dst = (va + vb + vc) / 3.0;
                }
            }
            DynamicProperty::Vec4(values) => {
                if let (Some(va), Some(vb), Some(vc), Some(dst)) = (
                    values.get(a).copied(),
                    values.get(b).copied(),
                    values.get(c).copied(),
                    values.get_mut(to),
                ) {
                    *dst = (va + vb + vc) / 3.0;
                }
            }
            DynamicProperty::Int(values) => {
                if let (Some(va), Some(vb), Some(vc), Some(dst)) = (
                    values.get(a).copied(),
                    values.get(b).copied(),
                    values.get(c).copied(),
                    values.get_mut(to),
                ) {
                    *dst = ((va as i64 + vb as i64 + vc as i64) / 3) as i32;
                }
            }
        }
    }
}

/// Trait for property value types
pub trait PropValue: 'static + Copy + Clone + Default {
    fn create_dynamic() -> DynamicProperty;
    fn get_dynamic(prop: &DynamicProperty, idx: usize) -> Option<Self>;
    fn set_dynamic(prop: &mut DynamicProperty, idx: usize, value: Self) -> bool;
}

impl PropValue for f32 {
    fn create_dynamic() -> DynamicProperty {
        DynamicProperty::Float(Vec::new())
    }

    fn get_dynamic(prop: &DynamicProperty, idx: usize) -> Option<Self> {
        match prop {
            DynamicProperty::Float(v) => v.get(idx).copied(),
            _ => None,
        }
    }

    fn set_dynamic(prop: &mut DynamicProperty, idx: usize, value: Self) -> bool {
        match prop {
            DynamicProperty::Float(v) => match v.get_mut(idx) {
                Some(slot) => {
                    *slot = value;
                    true
                }
                None => false,
            },
            _ => false,
        }
    }
}

impl PropValue for Vec2 {
    fn create_dynamic() -> DynamicProperty {
        DynamicProperty::Vec2(Vec::new())
    }

    fn get_dynamic(prop: &DynamicProperty, idx: usize) -> Option<Self> {
        match prop {
            DynamicProperty::Vec2(v) => v.get(idx).copied(),
            _ => None,
        }
    }

    fn set_dynamic(prop: &mut DynamicProperty, idx: usize, value: Self) -> bool {
        match prop {
            DynamicProperty::Vec2(v) => match v.get_mut(idx) {
                Some(slot) => {
                    *slot = value;
                    true
                }
                None => false,
            },
            _ => false,
        }
    }
}

impl PropValue for Vec3 {
    fn create_dynamic() -> DynamicProperty {
        DynamicProperty::Vec3(Vec::new())
    }

    fn get_dynamic(prop: &DynamicProperty, idx: usize) -> Option<Self> {
        match prop {
            DynamicProperty::Vec3(v) => v.get(idx).copied(),
            _ => None,
        }
    }

    fn set_dynamic(prop: &mut DynamicProperty, idx: usize, value: Self) -> bool {
        match prop {
            DynamicProperty::Vec3(v) => match v.get_mut(idx) {
                Some(slot) => {
                    *slot = value;
                    true
                }
                None => false,
            },
            _ => false,
        }
    }
}

impl PropValue for Vec4 {
    fn create_dynamic() -> DynamicProperty {
        DynamicProperty::Vec4(Vec::new())
    }

    fn get_dynamic(prop: &DynamicProperty, idx: usize) -> Option<Self> {
        match prop {
            DynamicProperty::Vec4(v) => v.get(idx).copied(),
            _ => None,
        }
    }

    fn set_dynamic(prop: &mut DynamicProperty, idx: usize, value: Self) -> bool {
        match prop {
            DynamicProperty::Vec4(v) => match v.get_mut(idx) {
                Some(slot) => {
                    *slot = value;
                    true
                }
                None => false,
            },
            _ => false,
        }
    }
}

impl PropValue for i32 {
    fn create_dynamic() -> DynamicProperty {
        DynamicProperty::Int(Vec::new())
    }

    fn get_dynamic(prop: &DynamicProperty, idx: usize) -> Option<Self> {
        match prop {
            DynamicProperty::Int(v) => v.get(idx).copied(),
            _ => None,
        }
    }

    fn set_dynamic(prop: &mut DynamicProperty, idx: usize, value: Self) -> bool {
        match prop {
            DynamicProperty::Int(v) => match v.get_mut(idx) {
                Some(slot) => {
                    *slot = value;
                    true
                }
                None => false,
            },
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
struct NamedProperty {
    name: String,
    values: DynamicProperty,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) enum VertexPropertyRef<'a> {
    Float { name: &'a str, values: &'a [f32] },
    Vec2 { name: &'a str, values: &'a [Vec2] },
    Vec3 { name: &'a str, values: &'a [Vec3] },
    Vec4 { name: &'a str, values: &'a [Vec4] },
    Int { name: &'a str, values: &'a [i32] },
}

impl DynamicProperty {
    fn as_vertex_property_ref<'a>(&'a self, name: &'a str) -> VertexPropertyRef<'a> {
        match self {
            DynamicProperty::Float(values) => VertexPropertyRef::Float { name, values },
            DynamicProperty::Vec2(values) => VertexPropertyRef::Vec2 { name, values },
            DynamicProperty::Vec3(values) => VertexPropertyRef::Vec3 { name, values },
            DynamicProperty::Vec4(values) => VertexPropertyRef::Vec4 { name, values },
            DynamicProperty::Int(values) => VertexPropertyRef::Int { name, values },
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct PropertyStore {
    props: HashMap<u32, NamedProperty>,
    next_prop_id: u32,
}

#[allow(dead_code)]
impl PropertyStore {
    pub(crate) fn clear(&mut self) {
        self.props.clear();
        self.next_prop_id = 0;
    }

    pub(crate) fn add<T: PropValue, Domain>(
        &mut self,
        name: &str,
        size: usize,
    ) -> PropertyHandle<T, Domain> {
        let id = self.next_prop_id;
        self.next_prop_id += 1;

        let mut values = T::create_dynamic();
        values.resize(size);
        self.props.insert(
            id,
            NamedProperty {
                name: name.to_string(),
                values,
            },
        );

        PropertyHandle::new(id)
    }

    pub(crate) fn get<T: PropValue, Domain>(
        &self,
        handle: PropertyHandle<T, Domain>,
        idx: usize,
    ) -> Option<T> {
        self.props
            .get(&handle.id())
            .and_then(|prop| T::get_dynamic(&prop.values, idx))
    }

    pub(crate) fn set<T: PropValue, Domain>(
        &mut self,
        handle: PropertyHandle<T, Domain>,
        idx: usize,
        value: T,
    ) -> bool {
        match self.props.get_mut(&handle.id()) {
            Some(prop) => T::set_dynamic(&mut prop.values, idx, value),
            None => false,
        }
    }

    pub(crate) fn contains<T, Domain>(&self, handle: PropertyHandle<T, Domain>) -> bool {
        self.props.contains_key(&handle.id())
    }

    pub(crate) fn name<T, Domain>(&self, handle: PropertyHandle<T, Domain>) -> Option<&str> {
        self.props.get(&handle.id()).map(|prop| prop.name.as_str())
    }

    pub(crate) fn resize_all(&mut self, size: usize) {
        for prop in self.props.values_mut() {
            prop.values.resize(size);
        }
    }

    pub(crate) fn copy_index(&mut self, from: usize, to: usize) {
        for prop in self.props.values_mut() {
            prop.values.copy_index(from, to);
        }
    }

    pub(crate) fn blend2_index(&mut self, a: usize, b: usize, to: usize) {
        for prop in self.props.values_mut() {
            prop.values.blend2_index(a, b, to);
        }
    }

    pub(crate) fn blend3_index(&mut self, a: usize, b: usize, c: usize, to: usize) {
        for prop in self.props.values_mut() {
            prop.values.blend3_index(a, b, c, to);
        }
    }

    fn sorted_refs(&self) -> Vec<&NamedProperty> {
        let mut props: Vec<_> = self.props.iter().collect();
        props.sort_by_key(|(id, _)| *id);
        props.into_iter().map(|(_, prop)| prop).collect()
    }
}

/// AttribSoAKernel - Unified Kernel with SoA layout and dynamic attributes
#[derive(Debug, Clone)]
pub struct AttribSoAKernel {
    // === SoA Position Data ===
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,

    // Vertex auxiliary data
    halfedge_handles: Vec<Option<HalfedgeHandle>>,

    // === Connectivity Data ===
    halfedges: Vec<Halfedge>,
    edges: Vec<Edge>,
    faces: Vec<Face>,

    // Edge lookup
    edge_map: HashMap<(u32, u32), HalfedgeHandle>,

    // Track which halfedges have had next set
    next_set: Vec<bool>,

    // === Preset Attributes (SoA layout) ===
    vertex_normals: Option<Vec<Vec3>>,
    vertex_colors: Option<Vec<Vec4>>,
    vertex_texcoords: Option<Vec<Vec2>>,

    halfedge_normals: Option<Vec<Vec3>>,
    halfedge_colors: Option<Vec<Vec4>>,
    halfedge_texcoords: Option<Vec<Vec2>>,

    edge_colors: Option<Vec<Vec4>>,

    face_normals: Option<Vec<Vec3>>,
    face_colors: Option<Vec<Vec4>>,

    // === Dynamic Properties ===
    vertex_props: PropertyStore,
    halfedge_props: PropertyStore,
    edge_props: PropertyStore,
    face_props: PropertyStore,
}

impl AttribSoAKernel {
    /// Create a new empty AttribSoAKernel
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
            edge_map: HashMap::new(),
            next_set: Vec::new(),
            // Preset attributes
            vertex_normals: None,
            vertex_colors: None,
            vertex_texcoords: None,
            halfedge_normals: None,
            halfedge_colors: None,
            halfedge_texcoords: None,
            edge_colors: None,
            face_normals: None,
            face_colors: None,
            // Dynamic properties
            vertex_props: PropertyStore::default(),
            halfedge_props: PropertyStore::default(),
            edge_props: PropertyStore::default(),
            face_props: PropertyStore::default(),
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
        self.edge_map.clear();
        self.next_set.clear();
        // Clear preset attributes
        self.vertex_normals = None;
        self.vertex_colors = None;
        self.vertex_texcoords = None;
        self.halfedge_normals = None;
        self.halfedge_colors = None;
        self.halfedge_texcoords = None;
        self.edge_colors = None;
        self.face_normals = None;
        self.face_colors = None;
        // Clear dynamic properties
        self.vertex_props.clear();
        self.halfedge_props.clear();
        self.edge_props.clear();
        self.face_props.clear();
    }

    // ========================
    // Vertex Operations
    // ========================

    /// Add a new vertex and return its handle
    #[inline]
    pub fn add_vertex(&mut self, point: Point3) -> VertexHandle {
        let idx = self.x.len() as u32;
        self.x.push(point.x);
        self.y.push(point.y);
        self.z.push(point.z);
        self.halfedge_handles.push(None);

        // Resize preset attribute arrays if they exist
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

    // ========================
    // Position Access (SIMD friendly)
    // ========================

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

    /// Get vertex position (unchecked)
    #[inline]
    pub unsafe fn point_unchecked(&self, idx: usize) -> Point3 {
        Point3::new(self.x[idx], self.y[idx], self.z[idx])
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

    // ========================
    // Edge Operations
    // ========================

    /// Add an edge between two vertices
    #[inline]
    pub fn add_edge(&mut self, start_vh: VertexHandle, end_vh: VertexHandle) -> HalfedgeHandle {
        let start_idx = start_vh.idx();
        let end_idx = end_vh.idx();

        // Check if edge already exists
        let min_idx = start_idx.min(end_idx);
        let max_idx = start_idx.max(end_idx);

        if let Some(&existing_he) = self.edge_map.get(&(min_idx, max_idx)) {
            return existing_he;
        }

        // Create two halfedges
        let heh1 = HalfedgeHandle::new(self.halfedges.len() as u32);
        let heh2 = HalfedgeHandle::new(heh1.idx() + 1);

        // Add halfedges
        self.halfedges.push(Halfedge::new(heh1, end_vh));
        self.halfedges.push(Halfedge::new(heh2, start_vh));

        // Initialize next_set
        self.next_set.push(false);
        self.next_set.push(false);

        // Create edge
        let _eh = EdgeHandle::new(self.edges.len() as u32);
        self.edges.push(Edge::new(heh1, heh2));

        // Set halfedge handles in vertices
        self.halfedge_handles[start_idx as usize] = Some(heh1);
        self.halfedge_handles[end_idx as usize] = Some(heh2);

        // Set opposite halfedge handles
        if let Some(he1) = self.halfedge_mut(heh1) {
            he1.set_opposite_halfedge(heh2);
        }
        if let Some(he2) = self.halfedge_mut(heh2) {
            he2.set_opposite_halfedge(heh1);
        }

        // Store in edge map
        self.edge_map.insert((min_idx, max_idx), heh1);

        // Resize attribute arrays
        self.resize_halfedge_attrs();
        self.resize_edge_attrs();
        self.halfedge_props.resize_all(self.halfedges.len());
        self.edge_props.resize_all(self.edges.len());

        heh1
    }

    /// Check if edge exists
    #[inline]
    pub fn edge_exists(&self, v0: u32, v1: u32) -> bool {
        let min_idx = v0.min(v1);
        let max_idx = v0.max(v1);
        self.edge_map.contains_key(&(min_idx, max_idx))
    }

    /// Get number of edges
    #[inline]
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get number of halfedges
    #[inline]
    pub fn n_halfedges(&self) -> usize {
        self.halfedges.len()
    }

    // ========================
    // Halfedge Access
    // ========================

    /// Get halfedge
    #[inline]
    pub fn halfedge(&self, heh: HalfedgeHandle) -> Option<&Halfedge> {
        self.halfedges.get(heh.idx() as usize)
    }

    /// Get mutable halfedge
    #[inline]
    pub fn halfedge_mut(&mut self, heh: HalfedgeHandle) -> Option<&mut Halfedge> {
        self.halfedges.get_mut(heh.idx() as usize)
    }

    /// Get edge
    #[inline]
    pub fn edge(&self, eh: EdgeHandle) -> Option<&Edge> {
        self.edges.get(eh.idx() as usize)
    }

    // ========================
    // Face Operations
    // ========================

    /// Add a face
    #[inline]
    pub fn add_face(&mut self, halfedge_handle: Option<HalfedgeHandle>) -> FaceHandle {
        let fh = FaceHandle::new(self.faces.len() as u32);
        self.faces.push(Face::new(halfedge_handle));
        self.resize_face_attrs();
        self.face_props.resize_all(self.faces.len());
        fh
    }

    /// Get number of faces
    #[inline]
    pub fn n_faces(&self) -> usize {
        self.faces.len()
    }

    /// Get face
    #[inline]
    pub fn face(&self, fh: FaceHandle) -> Option<&Face> {
        self.faces.get(fh.idx() as usize)
    }

    /// Get mutable face
    #[inline]
    pub fn face_mut(&mut self, fh: FaceHandle) -> Option<&mut Face> {
        self.faces.get_mut(fh.idx() as usize)
    }

    /// Get face's halfedge handle
    #[inline]
    pub fn face_halfedge_handle(&self, fh: FaceHandle) -> Option<HalfedgeHandle> {
        self.face(fh).and_then(|f| f.halfedge_handle())
    }

    // ========================
    // Connectivity Helpers
    // ========================

    /// Get vertex handle from halfedge
    #[inline]
    pub fn to_vertex_handle(&self, heh: HalfedgeHandle) -> VertexHandle {
        self.halfedge(heh)
            .map(|h| h.to_vertex_handle())
            .unwrap_or(VertexHandle::new(0))
    }

    /// Get from vertex handle from halfedge
    #[inline]
    pub fn from_vertex_handle(&self, heh: HalfedgeHandle) -> VertexHandle {
        // Get the opposite halfedge and return its to_vertex
        // This is the correct way to get from_vertex without mesh context
        self.opposite_halfedge_handle(heh)
            .and_then(|opp_heh| self.halfedge(opp_heh))
            .map(|h| h.to_vertex_handle())
            .unwrap_or(VertexHandle::invalid())
    }

    /// Get opposite halfedge handle
    #[inline]
    pub fn opposite_halfedge_handle(&self, heh: HalfedgeHandle) -> Option<HalfedgeHandle> {
        self.halfedge(heh)
            .and_then(|h| h.opposite_halfedge_handle())
    }

    /// Get edge handle from halfedge
    #[inline]
    pub fn edge_handle(&self, heh: HalfedgeHandle) -> EdgeHandle {
        self.halfedge(heh)
            .and_then(|h| h.edge_handle())
            .unwrap_or(EdgeHandle::new(0))
    }

    /// Get halfedge handle from edge
    #[inline]
    pub fn edge_halfedge_handle(&self, eh: EdgeHandle, idx: usize) -> HalfedgeHandle {
        self.edge(eh)
            .map(|e| e.halfedge_handle(idx))
            .unwrap_or(HalfedgeHandle::new(0))
    }

    /// Get face handle from halfedge
    #[inline]
    pub fn face_handle(&self, heh: HalfedgeHandle) -> Option<FaceHandle> {
        self.halfedge(heh).and_then(|h| h.face_handle())
    }

    /// Set face handle for halfedge
    #[inline]
    pub fn set_face_handle(&mut self, heh: HalfedgeHandle, fh: FaceHandle) {
        if let Some(he) = self.halfedge_mut(heh) {
            he.set_face_handle(fh);
        }
    }

    /// Check if halfedge is boundary
    #[inline]
    pub fn is_boundary(&self, heh: HalfedgeHandle) -> bool {
        self.face_handle(heh).is_none()
    }

    /// Get next halfedge handle
    #[inline]
    pub fn next_halfedge_handle(&self, heh: HalfedgeHandle) -> Option<HalfedgeHandle> {
        self.halfedge(heh).and_then(|h| h.next_halfedge_handle())
    }

    /// Set next halfedge handle
    #[inline]
    pub fn set_next_halfedge_handle(&mut self, heh: HalfedgeHandle, next_heh: HalfedgeHandle) {
        if let Some(he) = self.halfedge_mut(heh) {
            he.set_next_halfedge_handle(next_heh);
            if (next_heh.idx() as usize) < self.next_set.len() {
                self.next_set[next_heh.idx() as usize] = true;
            }
        }
    }

    /// Get previous halfedge handle
    #[inline]
    pub fn prev_halfedge_handle(&self, heh: HalfedgeHandle) -> Option<HalfedgeHandle> {
        self.halfedge(heh).and_then(|h| h.prev_halfedge_handle())
    }

    /// Get vertex's halfedge handle
    #[inline]
    pub fn halfedge_handle(&self, vh: VertexHandle) -> Option<HalfedgeHandle> {
        self.halfedge_handles
            .get(vh.idx() as usize)
            .and_then(|h| *h)
    }

    /// Set vertex's halfedge handle
    #[inline]
    pub fn set_halfedge_handle(&mut self, vh: VertexHandle, heh: HalfedgeHandle) {
        if (vh.idx() as usize) < self.halfedge_handles.len() {
            self.halfedge_handles[vh.idx() as usize] = Some(heh);
        }
    }

    /// Set halfedge's to vertex
    #[inline]
    pub fn set_halfedge_to_vertex(&mut self, heh: HalfedgeHandle, vh: VertexHandle) {
        if let Some(he) = self.halfedge_mut(heh) {
            he.set_to_vertex_handle(vh);
        }
    }

    fn resize_halfedge_attrs(&mut self) {
        let size = self.halfedges.len();
        if let Some(ref mut normals) = self.halfedge_normals {
            normals.resize(size, Vec3::zeros());
        }
        if let Some(ref mut colors) = self.halfedge_colors {
            colors.resize(size, Vec4::zeros());
        }
        if let Some(ref mut texcoords) = self.halfedge_texcoords {
            texcoords.resize(size, Vec2::zeros());
        }
    }

    fn resize_edge_attrs(&mut self) {
        let size = self.edges.len();
        if let Some(ref mut colors) = self.edge_colors {
            colors.resize(size, Vec4::zeros());
        }
    }

    fn resize_face_attrs(&mut self) {
        let size = self.faces.len();
        if let Some(ref mut normals) = self.face_normals {
            normals.resize(size, Vec3::zeros());
        }
        if let Some(ref mut colors) = self.face_colors {
            colors.resize(size, Vec4::zeros());
        }
    }

    // ========================
    // Preset Attributes - Vertex
    // ========================

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
            .and_then(|n| n.get(vh.idx() as usize).copied())
    }

    /// Set vertex normal
    pub fn set_vertex_normal(&mut self, vh: VertexHandle, n: Vec3) {
        if let Some(ref mut normals) = self.vertex_normals {
            if let Some(normal) = normals.get_mut(vh.idx() as usize) {
                *normal = n;
            }
        }
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
            .and_then(|c| c.get(vh.idx() as usize).copied())
    }

    /// Set vertex color
    pub fn set_vertex_color(&mut self, vh: VertexHandle, c: Vec4) {
        if let Some(ref mut colors) = self.vertex_colors {
            if let Some(color) = colors.get_mut(vh.idx() as usize) {
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
            .and_then(|t| t.get(vh.idx() as usize).copied())
    }

    /// Set vertex texcoord
    pub fn set_vertex_texcoord(&mut self, vh: VertexHandle, t: Vec2) {
        if let Some(ref mut texcoords) = self.vertex_texcoords {
            if let Some(texcoord) = texcoords.get_mut(vh.idx() as usize) {
                *texcoord = t;
            }
        }
    }

    // ========================
    // Preset Attributes - Halfedge
    // ========================

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
            .and_then(|n| n.get(heh.idx() as usize).copied())
    }

    /// Set halfedge normal
    pub fn set_halfedge_normal(&mut self, heh: HalfedgeHandle, n: Vec3) {
        if let Some(ref mut normals) = self.halfedge_normals {
            if let Some(normal) = normals.get_mut(heh.idx() as usize) {
                *normal = n;
            }
        }
    }

    /// Request halfedge colors
    pub fn request_halfedge_colors(&mut self) {
        if self.halfedge_colors.is_none() {
            let size = self.halfedges.len();
            self.halfedge_colors = Some(vec![Vec4::zeros(); size]);
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
            .and_then(|c| c.get(heh.idx() as usize).copied())
    }

    /// Set halfedge color
    pub fn set_halfedge_color(&mut self, heh: HalfedgeHandle, c: Vec4) {
        if let Some(ref mut colors) = self.halfedge_colors {
            if let Some(color) = colors.get_mut(heh.idx() as usize) {
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
            .and_then(|t| t.get(heh.idx() as usize).copied())
    }

    /// Set halfedge texcoord
    pub fn set_halfedge_texcoord(&mut self, heh: HalfedgeHandle, t: Vec2) {
        if let Some(ref mut texcoords) = self.halfedge_texcoords {
            if let Some(texcoord) = texcoords.get_mut(heh.idx() as usize) {
                *texcoord = t;
            }
        }
    }

    // ========================
    // Preset Attributes - Edge
    // ========================

    /// Request edge colors
    pub fn request_edge_colors(&mut self) {
        if self.edge_colors.is_none() {
            let size = self.edges.len();
            self.edge_colors = Some(vec![Vec4::zeros(); size]);
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
            .and_then(|c| c.get(eh.idx() as usize).copied())
    }

    /// Set edge color
    pub fn set_edge_color(&mut self, eh: EdgeHandle, c: Vec4) {
        if let Some(ref mut colors) = self.edge_colors {
            if let Some(color) = colors.get_mut(eh.idx() as usize) {
                *color = c;
            }
        }
    }

    // ========================
    // Preset Attributes - Face
    // ========================

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
            .and_then(|n| n.get(fh.idx() as usize).copied())
    }

    /// Set face normal
    pub fn set_face_normal(&mut self, fh: FaceHandle, n: Vec3) {
        if let Some(ref mut normals) = self.face_normals {
            if let Some(normal) = normals.get_mut(fh.idx() as usize) {
                *normal = n;
            }
        }
    }

    /// Request face colors
    pub fn request_face_colors(&mut self) {
        if self.face_colors.is_none() {
            let size = self.faces.len();
            self.face_colors = Some(vec![Vec4::zeros(); size]);
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
            .and_then(|c| c.get(fh.idx() as usize).copied())
    }

    /// Set face color
    pub fn set_face_color(&mut self, fh: FaceHandle, c: Vec4) {
        if let Some(ref mut colors) = self.face_colors {
            if let Some(color) = colors.get_mut(fh.idx() as usize) {
                *color = c;
            }
        }
    }

    // ========================
    // Dynamic Properties
    // ========================

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
        self.vertex_props.get(handle, vh.idx() as usize)
    }

    pub fn halfedge_property<T: PropValue>(
        &self,
        handle: HPropHandle<T>,
        heh: HalfedgeHandle,
    ) -> Option<T> {
        self.halfedge_props.get(handle, heh.idx() as usize)
    }

    pub fn edge_property<T: PropValue>(&self, handle: EPropHandle<T>, eh: EdgeHandle) -> Option<T> {
        self.edge_props.get(handle, eh.idx() as usize)
    }

    pub fn face_property<T: PropValue>(&self, handle: FPropHandle<T>, fh: FaceHandle) -> Option<T> {
        self.face_props.get(handle, fh.idx() as usize)
    }

    pub fn set_vertex_property<T: PropValue>(
        &mut self,
        handle: VPropHandle<T>,
        vh: VertexHandle,
        value: T,
    ) -> bool {
        self.vertex_props.set(handle, vh.idx() as usize, value)
    }

    pub fn set_halfedge_property<T: PropValue>(
        &mut self,
        handle: HPropHandle<T>,
        heh: HalfedgeHandle,
        value: T,
    ) -> bool {
        self.halfedge_props.set(handle, heh.idx() as usize, value)
    }

    pub fn set_edge_property<T: PropValue>(
        &mut self,
        handle: EPropHandle<T>,
        eh: EdgeHandle,
        value: T,
    ) -> bool {
        self.edge_props.set(handle, eh.idx() as usize, value)
    }

    pub fn set_face_property<T: PropValue>(
        &mut self,
        handle: FPropHandle<T>,
        fh: FaceHandle,
        value: T,
    ) -> bool {
        self.face_props.set(handle, fh.idx() as usize, value)
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

    // Compatibility wrappers for the legacy vertex-only dynamic property helpers.
    pub fn add_property<T: PropValue>(&mut self, name: &str) -> PropHandle<T> {
        self.add_vertex_property(name)
    }

    pub fn get_property<T: PropValue>(&self, handle: PropHandle<T>, idx: usize) -> Option<T> {
        self.vertex_props.get(handle, idx)
    }

    pub fn set_property<T: PropValue>(
        &mut self,
        handle: PropHandle<T>,
        idx: usize,
        value: T,
    ) -> bool {
        self.vertex_props.set(handle, idx, value)
    }

    pub fn set_property_f32(&mut self, handle: PropHandle<f32>, idx: usize, value: f32) -> bool {
        self.set_property(handle, idx, value)
    }

    pub fn set_property_vec2(&mut self, handle: PropHandle<Vec2>, idx: usize, value: Vec2) -> bool {
        self.set_property(handle, idx, value)
    }

    pub fn set_property_vec3(&mut self, handle: PropHandle<Vec3>, idx: usize, value: Vec3) -> bool {
        self.set_property(handle, idx, value)
    }

    pub fn set_property_vec4(&mut self, handle: PropHandle<Vec4>, idx: usize, value: Vec4) -> bool {
        self.set_property(handle, idx, value)
    }

    pub fn set_property_i32(&mut self, handle: PropHandle<i32>, idx: usize, value: i32) -> bool {
        self.set_property(handle, idx, value)
    }

    pub fn has_property<T>(&self, handle: PropHandle<T>) -> bool {
        self.has_vertex_property(handle)
    }

    pub(crate) fn vertex_property_refs(&self) -> Vec<VertexPropertyRef<'_>> {
        self.vertex_props
            .sorted_refs()
            .into_iter()
            .map(|prop| prop.values.as_vertex_property_ref(prop.name.as_str()))
            .collect()
    }
}

impl Default for AttribSoAKernel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_kernel() {
        let kernel = AttribSoAKernel::new();
        assert_eq!(kernel.n_vertices(), 0);
        assert!(kernel.is_empty());
    }

    #[test]
    fn test_add_vertex() {
        let mut kernel = AttribSoAKernel::new();
        let vh = kernel.add_vertex(Point3::new(1.0, 2.0, 3.0));

        assert_eq!(kernel.n_vertices(), 1);
        assert_eq!(
            kernel.point(vh.idx() as usize),
            Some(Point3::new(1.0, 2.0, 3.0))
        );
    }

    #[test]
    fn test_vertex_attributes() {
        let mut kernel = AttribSoAKernel::new();
        let vh = kernel.add_vertex(Point3::new(1.0, 2.0, 3.0));

        // Request and set normals
        kernel.request_vertex_normals();
        assert!(kernel.has_vertex_normals());

        kernel.set_vertex_normal(vh, Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(kernel.vertex_normal(vh), Some(Vec3::new(0.0, 1.0, 0.0)));

        // Request and set colors
        kernel.request_vertex_colors();
        assert!(kernel.has_vertex_colors());

        kernel.set_vertex_color(vh, Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(kernel.vertex_color(vh), Some(Vec4::new(1.0, 0.0, 0.0, 1.0)));
    }

    #[test]
    fn test_dynamic_property() {
        let mut kernel = AttribSoAKernel::new();
        let vh = kernel.add_vertex(Point3::new(1.0, 2.0, 3.0));
        let prop = kernel.add_vertex_property::<f32>("custom_float");

        assert!(kernel.has_vertex_property(prop));
        assert_eq!(kernel.vertex_property_name(prop), Some("custom_float"));
        assert_eq!(kernel.vertex_property(prop, vh), Some(0.0));
        assert!(kernel.set_vertex_property(prop, vh, 2.5));
        assert_eq!(kernel.vertex_property(prop, vh), Some(2.5));
    }

    #[test]
    fn test_vertex_property_auto_resizes() {
        let mut kernel = AttribSoAKernel::new();
        let first = kernel.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let quality = kernel.add_vertex_property::<f32>("quality");
        assert!(kernel.set_vertex_property(quality, first, 1.0));

        let second = kernel.add_vertex(Point3::new(1.0, 0.0, 0.0));
        assert_eq!(kernel.vertex_property(quality, first), Some(1.0));
        assert_eq!(kernel.vertex_property(quality, second), Some(0.0));
    }

    #[test]
    fn test_edge_and_halfedge_properties_auto_resize() {
        let mut kernel = AttribSoAKernel::new();
        let v0 = kernel.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = kernel.add_vertex(Point3::new(1.0, 0.0, 0.0));

        let edge_quality = kernel.add_edge_property::<i32>("edge_quality");
        let halfedge_flow = kernel.add_halfedge_property::<Vec2>("halfedge_flow");

        let heh = kernel.add_edge(v0, v1);
        let opp = kernel.opposite_halfedge_handle(heh).unwrap();
        let eh = kernel.edge_handle(heh);

        assert_eq!(
            kernel.edge_property_name(edge_quality),
            Some("edge_quality")
        );
        assert_eq!(
            kernel.halfedge_property_name(halfedge_flow),
            Some("halfedge_flow")
        );
        assert_eq!(kernel.edge_property(edge_quality, eh), Some(0));
        assert_eq!(
            kernel.halfedge_property(halfedge_flow, heh),
            Some(Vec2::zeros())
        );
        assert_eq!(
            kernel.halfedge_property(halfedge_flow, opp),
            Some(Vec2::zeros())
        );

        assert!(kernel.set_edge_property(edge_quality, eh, 7));
        assert!(kernel.set_halfedge_property(halfedge_flow, heh, Vec2::new(1.0, 2.0)));

        assert_eq!(kernel.edge_property(edge_quality, eh), Some(7));
        assert_eq!(
            kernel.halfedge_property(halfedge_flow, heh),
            Some(Vec2::new(1.0, 2.0))
        );
        assert_eq!(
            kernel.halfedge_property(halfedge_flow, opp),
            Some(Vec2::zeros())
        );
    }

    #[test]
    fn test_face_property_auto_resizes() {
        let mut kernel = AttribSoAKernel::new();
        let priority = kernel.add_face_property::<Vec3>("priority");
        let fh = kernel.add_face(None);

        assert!(kernel.has_face_property(priority));
        assert_eq!(kernel.face_property_name(priority), Some("priority"));
        assert_eq!(kernel.face_property(priority, fh), Some(Vec3::zeros()));
        assert!(kernel.set_face_property(priority, fh, Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!(
            kernel.face_property(priority, fh),
            Some(Vec3::new(1.0, 2.0, 3.0))
        );
    }
}
